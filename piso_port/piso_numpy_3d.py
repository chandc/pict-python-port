"""
Phase 5: PISO orchestration on a 3D curvilinear collocated grid.

Layout follows the PICT C++ implementation: pressure AND velocity both live at cell centres
(Block::CreatePressure -> CreateDataTensor(1), Block::CreateVelocity -> CreateDataTensor(dims),
both on the same grid). Face fluxes are a DERIVED quantity, interpolated from the cell-centred
contravariant components. PICT uses no Rhie-Chow interpolation; consistency comes instead from
expressing the divergence and the pressure operator on the same faces (see phase5_fluxes.py).

Per time step, mirroring PISOtorch_simulation.py (corrector_steps default 2):

    predictor   [J/dt + J*adv - nu*lap_diag] u* = J/dt u^n - J grad p + nu J cross(u*)
    correctors  repeat corrector_steps times:
                    F   <- face fluxes of u*
                    solve  M p' = J*(D(cross(p')) - div F)      # M = div( (J/Adiag) grad )
                    F   <- F - Phi(p')                          # fluxes now divergence-free
                    u*  <- u* - (J/Adiag) grad p'               # cell-centred correction
                    p   <- p + p'

Backward Euler matches SetupAdvectionMatrixEulerImplicit (:4483) and the plan's section 2.
The velocity correction u = hbyA - (1/Adiag) grad p matches PISO_update_velocity (:5948),
PICT's default velocity_corrector="FD".

A note on what is and is not divergence-free: the FLUXES are, to machine precision. The
cell-centred velocity is not exactly so, and cannot be on a collocated arrangement -- PICT has
the same property. Continuity is carried by the fluxes.
"""
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

from phase1_grid_metrics import (analytical_wavy_grid_mms, compute_numerical_metrics,
                                 make_grid, as_periodic)
from phase2_operators import compute_gradient
from phase3_momentum import (build_momentum_matrix_7point, build_conservative_diffusion_matrix,
                             compute_cross_diffusion, boundary_masks, deferred_correction)
from phase5_fluxes import (compute_face_fluxes, divergence_from_fluxes,
                           pressure_face_fluxes, correct_fluxes)

class PISOSolver:
    def __init__(self, n, warp=0.05, nu=0.01, dt=0.01, corrector_steps=2,
                 momentum_dc_iters=2, pressure_dc_iters=500, pressure_tol=1e-12,
                 boundary_flux_mode='from_velocity', periodic=None,
                 scheme='chorin', time_scheme='be', convection='sou',
                 pressure_coef='auto', picard_iters=1, implicit_cross=False):
        """
        scheme:
          'chorin'      -- non-incremental projection: the predictor carries NO pressure
                           gradient and p is recomputed each step. PICT's default
                           (apply_pressure_gradient = False). 1st order; the velocity
                           Dirichlet BCs impose a spurious dp/dn = 0 at walls.
          'incremental' -- predictor carries -grad p^n, solve for the increment phi,
                           p <- p + phi. Better pressure, but STILL imposes dp/dn = 0,
                           so the wall boundary layer survives.
          'rotational'  -- incremental plus the rotational correction
                               p <- p + phi - nu * div(u*)
                           which cancels that spurious wall condition at leading order.
                           Uses the identity  nu lap(u) = nu grad(div u) - nu curl(curl u):
                           div(u*) is NOT zero on the predictor field, and it is exactly the
                           piece the standard update throws away.

        time_scheme:
          'be'   -- Backward Euler, 1st order.
          'bdf2' -- 2nd-order backward difference. Required to SEE the rotational term's
                    benefit: with Backward Euler the O(dt) time truncation error dominates
                    and the rotational correction buys nothing measurable.
        """
        self.n, self.nu, self.dt = n, nu, dt
        self.corrector_steps = corrector_steps
        self.momentum_dc_iters = momentum_dc_iters
        self.pressure_dc_iters = pressure_dc_iters
        self.pressure_tol = pressure_tol
        self.boundary_flux_mode = boundary_flux_mode
        self.per = as_periodic(periodic)
        self.scheme = scheme
        self.time_scheme = time_scheme
        self.convection = convection
        # How Gamma (the pressure-correction coefficient) approximates the response of A^{-1}
        # to a pressure gradient:
        #   'diag'   -- Gamma = J/A_diag. The textbook PISO/SIMPLE choice, and what PICT uses
        #               (raP = 1/Adiag).
        #   'rowsum' -- Gamma = J/rowsum(A), the SIMPLEC choice. For a SMOOTH gradient this is
        #               the correct measure: the diffusion operator has ZERO row sum, so its
        #               diagonal is entirely cancelled by its neighbours and 'diag' understates
        #               the response by a factor 1 + 2 nu dt sum(1/h_i^2). That factor reaches
        #               3.4 at nu*dt/h^2 ~ 1.2 and 10 at ~4.8 -- and it is exactly where the
        #               incremental/rotational feedback loop goes unstable.
        #   'auto'   -- 'diag' for chorin (matching PICT exactly, and harmless there since
        #               chorin never feeds p back), 'rowsum' for the accumulating schemes.
        if pressure_coef == 'auto':
            pressure_coef = 'diag' if scheme == 'chorin' else 'rowsum'
        self.pressure_coef = pressure_coef
        # Outer Picard iterations per step. A is assembled from the OLD velocity, which is an
        # O(dt) lag; repeating the step with A rebuilt from the latest u* removes it. Costs
        # nothing when advection vanishes (a Cartesian channel), matters when it does not.
        self.picard_iters = picard_iters
        # Treat the non-orthogonal cross terms IMPLICITLY instead of by deferred correction.
        # Applied matrix-free: the operator is exactly M_orth - J*cross(.), i.e. precisely the
        # fixed point the deferred correction converges to, so both paths must agree.
        #
        # Solver choice is forced by symmetry, which is NOT the same in both cases:
        #   fully periodic -> symmetric to 1e-15, CG is valid
        #   any wall axis  -> asymmetric at ~1e-2, because the one-sided edge differences in
        #                     np.gradient are not self-adjoint. CG is then INVALID and
        #                     BiCGStab is used instead.
        # (The 7-point deferred-correction path keeps CG valid in both cases, since its
        # implicit operator is the symmetric orthogonal part alone.)
        self.implicit_cross = implicit_cross
        self.u_prev = None          # for BDF2
        if all(self.per):
            # Only a FULLY periodic domain has no prescribed boundary faces at all. With mixed
            # periodicity (e.g. walls in y, periodic in x and z) the periodic axes are handled
            # per-axis inside compute_face_fluxes, and the wall axes still need a real boundary
            # mode -- overriding it here would silently discard the caller's choice.
            self.boundary_flux_mode = 'periodic'

        ns = (n, n, n) if isinstance(n, (int, np.integer)) else tuple(n)
        self.shape = ns
        self.nx, self.ny, self.nz = ns
        if any(self.per) or len(set(ns)) > 1:
            x, y, z, dxi, deta, dzeta = make_grid(ns, warp=warp, periodic=self.per)
        else:
            x, y, z, dxi, deta, dzeta, _, _ = analytical_wavy_grid_mms(n, n, n,
                                                                      Ax=warp, Ay=warp, Az=warp)
        self.x, self.y, self.z = x, y, z
        self.h = (dxi, deta, dzeta)
        self.J, self.metrics = compute_numerical_metrics(x, y, z, dxi, deta, dzeta,
                                                         periodic=self.per)

        self.ib, self.bb, self.bmask = boundary_masks(*ns, periodic=self.per)

        # fields
        self.u = np.zeros(ns)
        self.v = np.zeros(ns)
        self.w = np.zeros(ns)
        self.p = np.zeros(ns)

        # Dirichlet velocity boundary values (default: all walls at rest)
        # Optional body force (3 arrays), added to the momentum RHS. This is PICT's
        # `velocitySource` hook -- forcing_version=4 in its plane-Poiseuille validation.
        self.velocity_source = None

        self.u_bc = np.zeros(ns)
        self.v_bc = np.zeros(ns)
        self.w_bc = np.zeros(ns)

    # ------------------------------------------------------------------ setup
    def set_lid_driven_cavity(self, lid_velocity=1.0):
        """No-slip on all walls; the k = n-1 face slides in +x."""
        self.u_bc[:] = 0.0; self.v_bc[:] = 0.0; self.w_bc[:] = 0.0
        self.u_bc[:, :, -1] = lid_velocity
        # A cavity is closed: force the wall fluxes to zero rather than deriving them from the
        # lid velocity. On a warped grid a lid velocity fixed in physical x is not exactly
        # tangent to the (curved) top face, so 'from_velocity' would leak a little mass through
        # it and make the singular Neumann pressure system inconsistent.
        self.boundary_flux_mode = 'impermeable'
        self._apply_velocity_bc()

    def _apply_velocity_bc(self):
        m = self.bmask
        self.u[m] = self.u_bc[m]
        self.v[m] = self.v_bc[m]
        self.w[m] = self.w_bc[m]

    # -------------------------------------------------------------- momentum
    def _momentum_matrix(self, convect=None):
        J = self.J
        A = build_momentum_matrix_7point(*self.shape, J, self.metrics, *self.h,
                                         *(convect if convect is not None else (self.u, self.v, self.w)),
                                         self.nu, periodic=self.per,
                                         convection=self.convection)
        # Backward Euler transient term: J/dt on the diagonal (volume-integrated, so that
        # every term carries the cell volume consistently). This also strengthens diagonal
        # dominance, which is why the momentum solve needs far fewer deferred-correction
        # sweeps here than the steady Phase 3 test did.
        # BDF2 puts 3/(2 dt) on the diagonal instead of 1/dt
        c0 = 1.5 / self.dt if (self.time_scheme == 'bdf2' and self.u_prev is not None) \
             else 1.0 / self.dt
        self._c0 = c0
        return (A + sparse.diags(J.ravel() * c0)).tocsr()

    def _solve_momentum(self, A):
        """
        Backward-Euler predictor for all three components, returning PICT's `hbyA`.

        The pressure gradient is deliberately NOT applied here: PISOtorch_simulation.py:1173
        sets `apply_pressure_gradient = False`, so the predictor produces H/A and the FULL
        pressure is applied once at correction time (PISO_update_velocity: velocityResult =
        pressureRHS - rDiag*grad p). Applying grad p in the predictor as well and then
        accumulating the correction into p makes the pressure drift without bound, because on
        a collocated grid the cell-centred gradient used by the predictor and the face operator
        used by the projection are not the same operator.
        """
        J, ib, bb = self.J, self.ib, self.bb
        A_ii = A[ib][:, ib].tocsr()
        A_ib = A[ib][:, bb].tocsr()

        # 'incremental'/'rotational' carry the old pressure gradient into the
        # predictor (PICT's applyPressureGradient = True); 'chorin' does not.
        if self.scheme in ('incremental', 'rotational'):
            gp = compute_gradient(self.p, self.metrics, *self.h, periodic=self.per)
        else:
            gp = (0.0, 0.0, 0.0)

        bdf2 = self.time_scheme == 'bdf2' and self.u_prev is not None
        prevs = self.u_prev if bdf2 else (None, None, None)

        out = []
        for comp, (phi_n, bc) in enumerate(((self.u, self.u_bc),
                                            (self.v, self.v_bc),
                                            (self.w, self.w_bc))):
            phi = phi_n.copy()
            phi[self.bmask] = bc[self.bmask]
            phi_b = phi.flat[bb]
            # transient source: BE uses u^n/dt; BDF2 uses (4u^n - u^{n-1})/(2 dt)
            trans = ((2.0 * phi_n - 0.5 * prevs[comp]) / self.dt) if bdf2 \
                    else (phi_n / self.dt)
            for _ in range(self.momentum_dc_iters):
                cross = compute_cross_diffusion(phi, J, self.metrics, *self.h,
                                            periodic=self.per)
                src = 0.0 if self.velocity_source is None else self.velocity_source[comp]
                rhs = (J * (trans - gp[comp] + self.nu * cross + src)).flat[ib] \
                      - A_ib @ phi_b
                sol, info = splinalg.bicgstab(A_ii, rhs, x0=phi.flat[ib], rtol=1e-10, maxiter=5000)
                if info != 0:
                    print(f"  warning: momentum BiCGStab info={info}")
                phi.flat[ib] = sol
            out.append(phi)
        return out

    # -------------------------------------------------------------- pressure
    def _solve_pressure_implicit(self, M, F, coef, free, div_F):
        """
        One solve with the FULL operator (orthogonal + cross), applied matrix-free.

        Removes the deferred-correction loop entirely: the operator here IS the fixed point
        that loop converges to, so the two paths must agree to solver tolerance (verified to
        1e-14..1e-12 in test_implicit_cross.py).

        What this does NOT do is buy extra warp range. Removing the Picard iteration removes
        its contraction limit, but that limit sits at warp ~0.18 and so does the point where
        this grid family tangles (min(J) < 0), so there is no valid mesh on which the extra
        range could be used. The demonstrated gain is speed at warps that actually mesh:
        2.1x-17.8x, growing with warp because that is where the lag costs most.
        """
        J = self.J
        N = J.size

        def apply(xf):
            # The operator must be EXACTLY the fixed point of the deferred correction:
            #     M p - J * D(Phi_cross(p))  =  -J * div F
            # so the cross part has to come from pressure_face_fluxes with the SAME `coef`,
            # not from compute_cross_diffusion -- the latter is unscaled by coef and uses a
            # different discretisation, which would silently solve a different system.
            x = np.zeros(N)
            x[free] = xf
            xg = x.reshape(self.shape)
            Phi_c = pressure_face_fluxes(xg, J, self.metrics, self.h, coef=coef,
                                         include_orth=False, include_cross=True,
                                         periodic=self.per)
            full = (M @ x) - (J * divergence_from_fluxes(Phi_c, J, self.h)).ravel()
            return full[free]

        nf = len(free)
        op = splinalg.LinearOperator((nf, nf), matvec=apply, dtype=float)
        b = J * (-div_F)
        b = b - b.mean()                       # compatibility: M is singular
        rhs = b.flat[free]

        # Precondition with the ORTHOGONAL part alone -- i.e. exactly the operator the deferred
        # correction inverts on every sweep. That choice is not arbitrary:
        #
        #     M^-1 A  =  I - M^-1 J D(Phi_cross(.))
        #
        # and the spectral radius of that second term IS the deferred-correction contraction
        # ratio (0.31 / 0.59 / 0.92 at warp 0.05 / 0.10 / 0.15). So the preconditioned spectrum
        # clusters at 1 with spread rho, which Krylov resolves in a handful of iterations
        # instead of the ~rho^k of a fixed-point sweep -- and, unlike that sweep, it keeps
        # converging when rho > 1. Without this the matrix-free operator is a net loss: one
        # application costs ~32x a sparse matvec (191.5 us vs 5.9 us at 1024 cells), so the
        # iteration count has to come down hard for the implicit path to pay for itself.
        # Measured with the preconditioner: 4.8x / 8.4x / 17.8x on the cavity at warp
        # 0.05 / 0.10 / 0.15, replacing 22 / 48 / 121 sweeps with 8 / 12 / 17 Krylov steps.
        lu = splinalg.splu(M[free][:, free].tocsc())
        prec = splinalg.LinearOperator((nf, nf), matvec=lu.solve, dtype=float)

        self._implicit_its = 0
        def _count(_x):
            self._implicit_its += 1

        # CG needs a symmetric operator. With a wall axis the one-sided edge differences make
        # the full operator non-self-adjoint, so BiCGStab is required there.
        if all(self.per):
            sol, info = splinalg.cg(op, rhs, M=prec, rtol=1e-13, maxiter=20000,
                                    callback=_count)
        else:
            sol, info = splinalg.bicgstab(op, rhs, M=prec, rtol=1e-13, maxiter=20000,
                                          callback=_count)
            if info != 0:
                sol, info = splinalg.lgmres(op, rhs, M=prec, rtol=1e-13, maxiter=5000,
                                            callback=_count)
        if info != 0:
            print(f"  warning: implicit-cross pressure solve info={info}")

        p = np.zeros_like(J)
        p.flat[free] = sol
        Phi = pressure_face_fluxes(p, J, self.metrics, self.h, coef=coef,
                                   include_cross=True, periodic=self.per)
        return p, correct_fluxes(F, Phi)

    def _solve_pressure(self, F, coef):
        """
        Solve  M p' = J*(D(cross(p')) - div F)  by deferred correction, then return p' and
        the corrected (divergence-free) fluxes.

        M is the zero-flux/Neumann operator -- it builds no face at the domain boundary, so its
        row sums vanish and it is singular with a constant null space. The RHS is compatible by
        construction (the flux divergence telescopes to the prescribed boundary fluxes, which
        are zero for a closed domain); we project it anyway for safety and pin one cell to
        remove the constant.
        """
        J = self.J
        free = np.arange(J.size)[1:]                    # pin cell 0 to fix the null space
        div_F = divergence_from_fluxes(F, J, self.h)

        # Non-orthogonal terms are carried explicitly (PICT's nonOrthoFlags). This is a Picard
        # iteration whose contraction ratio grows with grid skewness. Measured at n=16:
        #
        #     warp 0.05 -> 0.31 (21 sweeps)     warp 0.15 -> 0.92 (321 sweeps)
        #     warp 0.10 -> 0.59 (49 sweeps)     warp 0.20 -> 1.27  DIVERGES
        #
        # CAVEAT on that last row, measured later: this grid family TANGLES at warp 0.18 --
        # min(J) goes negative -- so the ratio > 1 at warp 0.20 was measured on a mesh with
        # negative cell volumes. The deferred-correction limit and the grid-validity limit
        # coincide here, and it is wrong to read the 1.27 as evidence of a solver limit alone.
        # Whatever the cause, under-relaxation does not lift it. Nor does boosting the implicit
        # coefficient while compensating the explicit term by the same amount -- that leaves
        # the fixed point identical and merely drives the iteration toward the identity map,
        # converging more slowly, which I measured before discarding the idea.
        #
        # implicit_cross=True removes the loop entirely (see _solve_pressure_implicit) and is
        # 2.1x-17.8x faster at warps that actually mesh. It does NOT buy extra warp range, because
        # there is no valid grid past 0.18 on which to use it. Past the limit this warns loudly
        # rather than returning a quietly wrong field.
        M = build_conservative_diffusion_matrix(*self.shape, *self.h, J, self.metrics,
                                                coef=coef, periodic=self.per)

        if self.implicit_cross:
            return self._solve_pressure_implicit(M, F, coef, free, div_F)
        M_ff = M[free][:, free].tocsr()

        def rhs_fn(p):
            Phi_cross = pressure_face_fluxes(p, J, self.metrics, self.h, coef=coef,
                                             include_orth=False, include_cross=True,
                                             periodic=self.per)
            b = J * (divergence_from_fluxes(Phi_cross, J, self.h) - div_F)
            return (b - b.mean()).flat[free]        # enforce compatibility (M is singular)

        def solve_fn(rhs, x0):
            sol, info = splinalg.cg(M_ff, rhs, x0=x0, rtol=1e-13, maxiter=20000)
            if info != 0:
                print(f"  warning: pressure CG info={info}")
            return sol

        p, dc = deferred_correction(np.zeros_like(J), free, rhs_fn, solve_fn,
                                    max_iters=self.pressure_dc_iters, tol=self.pressure_tol)
        self._dc_sweeps = dc['iters']   # counterpart of _implicit_its, for like-for-like cost
        if not dc['converged']:
            print(f"  warning: pressure deferred correction did not converge "
                  f"({dc['iters']} sweeps at omega={dc['omega']}) -- grid warp likely past ~0.18")

        Phi = pressure_face_fluxes(p, J, self.metrics, self.h, coef=coef,
                                   include_cross=True, periodic=self.per)
        return p, correct_fluxes(F, Phi)

    # ------------------------------------------------------------------ step
    def step(self):
        """
        Advance one step, optionally iterating the Picard linearisation.

        A is assembled from the convecting velocity, which by default is u^n -- an O(dt) lag.
        Repeating the step with A rebuilt from the latest u*, restoring the starting state each
        time so time advances only once, removes that lag.
        """
        if self.picard_iters <= 1:
            return self._step_once()
        u0, v0, w0, p0 = self.u.copy(), self.v.copy(), self.w.copy(), self.p.copy()
        prev0 = self.u_prev
        convect, out = None, None
        for _k in range(self.picard_iters):
            if _k > 0:
                self.u, self.v, self.w = u0.copy(), v0.copy(), w0.copy()
                self.p, self.u_prev = p0.copy(), prev0
            out = self._step_once(convect)
            convect = (self.u.copy(), self.v.copy(), self.w.copy())
        return out

    def _step_once(self, convect=None):
        J = self.J

        A = self._momentum_matrix(convect)
        us, vs, ws = self._solve_momentum(A)           # hbyA

        rowsum = np.asarray(A.sum(axis=1)).ravel().reshape(J.shape)
        diag = A.diagonal().reshape(J.shape)
        if self.pressure_coef == 'rowsum':
            denom = rowsum
        else:
            denom = diag
            # Guard the combination that is genuinely unsound: an accumulating scheme with the
            # diagonal coefficient. 'diag' understates the response of A^{-1} to a smooth
            # pressure gradient by rowsum/diag = 1 + 2 nu dt sum(1/h_i^2), because the diffusion
            # operator has ZERO row sum. Chorin is unaffected -- it replaces p rather than
            # accumulating it -- but the incremental schemes feed that deficit back every step
            # and diverge once the ratio passes ~3. Fail loudly, not 100 steps into a run.
            if self.scheme != 'chorin':
                # NOTE the direction: the deficit is diag/rowsum, i.e. Gamma_rowsum/Gamma_diag.
                # rowsum/diag is always < 1 and would never trip this.
                ratio = float(np.mean(diag / rowsum))
                if ratio > 3.0:
                    raise RuntimeError(
                        f"pressure_coef='diag' with scheme='{self.scheme}' is unstable here: "
                        f"diag/rowsum = {ratio:.2f} (> 3). The correction under-corrects by that "
                        f"factor and the incremental loop amplifies it. Use "
                        f"pressure_coef='rowsum', or reduce nu*dt/h^2.")
        coef = J / denom

        # Pressure is solved afresh every step (never carried over), matching PICT, where
        # CopyPressureResultToBlocks replaces the pressure field each corrector.
        phi_total = np.zeros_like(J)
        div_star = None
        for c in range(self.corrector_steps):
            F = compute_face_fluxes(us, vs, ws, J, self.metrics,
                                    boundary=self.boundary_flux_mode, periodic=self.per)
            if c == 0:
                # divergence of the PREDICTOR field -- exactly what the rotational term needs
                div_star = divergence_from_fluxes(F, J, self.h)
            pp, F = self._solve_pressure(F, coef)

            dgx, dgy, dgz = compute_gradient(pp, self.metrics, *self.h,
                                             periodic=self.per)
            us = us - coef * dgx
            vs = vs - coef * dgy
            ws = ws - coef * dgz
            for f, bc in ((us, self.u_bc), (vs, self.v_bc), (ws, self.w_bc)):
                f[self.bmask] = bc[self.bmask]
            phi_total += pp

        if self.scheme == "chorin":
            self.p = phi_total                       # recomputed, never accumulated
        elif self.scheme == "incremental":
            self.p = self.p + phi_total
        elif self.scheme == "rotational":
            # p <- p + phi - nu * div(u*): cancels the spurious dp/dn = 0 that the
            # projection otherwise imposes at walls, which is what caps the standard scheme.
            self.p = self.p + phi_total - self.nu * div_star
        else:
            raise ValueError(f"unknown scheme {self.scheme!r}")

        self.u_prev = (self.u.copy(), self.v.copy(), self.w.copy())
        self.u, self.v, self.w = us, vs, ws
        self.last_flux_divergence = divergence_from_fluxes(F, J, self.h)
        return np.abs(self.last_flux_divergence).max()
