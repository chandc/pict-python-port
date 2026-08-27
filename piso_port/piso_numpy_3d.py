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
                 scheme='chorin', time_scheme='be', convection='sou'):
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
        self.u_prev = None          # for BDF2
        if any(self.per):
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
    def _momentum_matrix(self):
        J = self.J
        A = build_momentum_matrix_7point(*self.shape, J, self.metrics, *self.h,
                                         self.u, self.v, self.w, self.nu, periodic=self.per,
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
                rhs = (J * (trans - gp[comp] + self.nu * cross)).flat[ib] \
                      - A_ib @ phi_b
                sol, info = splinalg.bicgstab(A_ii, rhs, x0=phi.flat[ib], rtol=1e-10, maxiter=5000)
                if info != 0:
                    print(f"  warning: momentum BiCGStab info={info}")
                phi.flat[ib] = sol
            out.append(phi)
        return out

    # -------------------------------------------------------------- pressure
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
        # So there is a hard usable limit near warp ~0.18, and it is a property of deferred
        # correction itself, not of the solver around it. Under-relaxation does not lift it
        # (the ratio is real and > 1). Nor does boosting the implicit coefficient while
        # compensating the explicit term by the same amount -- that leaves the fixed point
        # identical and merely drives the iteration toward the identity map, converging more
        # slowly, which I measured before discarding the idea. Lifting the limit properly
        # means making the cross terms implicit (a 19- or 27-point matrix), which is a
        # deliberate future change, not a tuning knob. Past the limit this warns loudly rather
        # than returning a quietly wrong field.
        M = build_conservative_diffusion_matrix(*self.shape, *self.h, J, self.metrics,
                                                coef=coef, periodic=self.per)
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
        if not dc['converged']:
            print(f"  warning: pressure deferred correction did not converge "
                  f"({dc['iters']} sweeps at omega={dc['omega']}) -- grid warp likely past ~0.18")

        Phi = pressure_face_fluxes(p, J, self.metrics, self.h, coef=coef,
                                   include_cross=True, periodic=self.per)
        return p, correct_fluxes(F, Phi)

    # ------------------------------------------------------------------ step
    def step(self):
        J = self.J

        A = self._momentum_matrix()
        us, vs, ws = self._solve_momentum(A)           # hbyA

        coef = J / A.diagonal().reshape(J.shape)       # PICT's 1/Adiag, volume-weighted

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
