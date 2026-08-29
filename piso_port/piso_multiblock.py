"""
A PISO step across multiple blocks.

Every operator this needs was built and verified against the single-block solver first
(multiblock.py): seam metrics, field padding, face fluxes, divergence, the pressure matrix and
the momentum matrix, each exact to machine precision. What remains here is ORCHESTRATION, and
the three things that go wrong in it are all global-vs-local mistakes:

  * THE PRESSURE PIN IS GLOBAL. One cell for the whole domain, not one per block. Pinning per
    block lets each block's pressure level float independently -- and it still converges, so
    nothing complains.
  * GAMMA COMES FROM THE GLOBAL MATRIX. coef = J / rowsum(A) must use the assembled global A,
    or seam rows get a coefficient computed as if their neighbours did not exist.
  * THE GRADIENT AND FLUXES MUST CROSS SEAMS. Using the block-local operators here would treat
    every connection as a wall.

SCOPE: Cartesian, central convection; periodic OR no-slip walls (the face-type
registry is consumed via Domain.wall_mask); BDF2 in time (BE on the
first step, which has no previous level). That is deliberate -- it is the
configuration where the multi-block result can be compared against the single-block solver
EXACTLY, with no cross terms (Cartesian) and no boundary conditions (periodic) to confound it.
Warped multi-block additionally needs the implicit cross operator across seams; walls need the
face-type registry. Both are separate increments.
"""
import numpy as np
import scipy.sparse.linalg as spla


class MultiBlockPISO:
    def __init__(self, domain, nu, dt, corrector_steps=2, tol=1e-13, time_scheme='bdf2',
                 scheme='rotational', picard_iters=2, implicit_cross=False):
        self.d = domain
        self.nu, self.dt = nu, dt
        # BDF2 by default, matching the single-block solver. The single-block Stokes study
        # measured order 2.00 with no-slip walls once the Richardson triple was taken inside
        # the asymptotic range, so second-order time is a real property worth carrying across
        # blocks rather than quietly dropping to Backward Euler.
        self.time_scheme = time_scheme
        self.u_prev = None
        # 'rotational' and picard_iters=2 by default, because BOTH are needed for second order
        # once convection is active. chorin's splitting error is O(dt) regardless of the
        # predictor, and the momentum matrix is assembled from the lagged u^n, which is a
        # second O(dt) error. Measured: 0.93 / 1.20 / 2.16 for chorin+bdf2, rotational with
        # picard_iters=1, and rotational with picard_iters=2.
        self.scheme = scheme
        self.picard_iters = picard_iters
        # Non-orthogonal cross terms treated IMPLICITLY across blocks -- the design chosen for
        # multi-block because the deferred-correction alternative would need cross-term fluxes
        # exchanged across every seam on every Picard sweep. Here the cross operator is applied
        # per block on PADDED fields using the same seam-aware pressure_face_fluxes the
        # orthogonal part uses, and the exact global 7-point matrix serves as preconditioner.
        # Required for WARPED multi-block; on a Cartesian grid the cross terms vanish and this
        # is pure overhead.
        self.implicit_cross = implicit_cross
        self.momentum_dc_iters = 2
        self.dong_delta = 0.01
        self._prec = None
        self.corrector_steps = corrector_steps
        self.tol = tol
        self.Js, self.ms = [], []
        for b in range(len(domain.blocks)):
            J, m = domain.block_metrics(b)
            self.Js.append(J); self.ms.append(m)
        self.u = {b: np.zeros(bl.shape) for b, bl in enumerate(domain.blocks)}
        self.v = {b: np.zeros(bl.shape) for b, bl in enumerate(domain.blocks)}
        self.w = {b: np.zeros(bl.shape) for b, bl in enumerate(domain.blocks)}
        self.p = {b: np.zeros(bl.shape) for b, bl in enumerate(domain.blocks)}
        # Dirichlet velocity on wall faces. Default zero = no-slip; set per block to drive a lid
        # or an inlet. Walls are found from the face-type registry, so a block whose '+x' is a
        # wall and whose neighbour's '+x' is a connection is handled correctly.
        self.u_bc = {b: np.zeros(bl.shape) for b, bl in enumerate(domain.blocks)}
        self.v_bc = {b: np.zeros(bl.shape) for b, bl in enumerate(domain.blocks)}
        self.w_bc = {b: np.zeros(bl.shape) for b, bl in enumerate(domain.blocks)}
        # Optional body force, PICT's `velocitySource`: a list of three per-block dicts (or
        # scalars). Without it a periodic channel has nothing to drive it and the velocity stays
        # identically zero -- which looks like a solver failure and is not one.
        self.velocity_source = None
        # Outflow faces as (block, face_id, U_c). They stay Dirichlet VELOCITY boundaries --
        # PICT's design -- advected out each step and then rescaled so the DOMAIN-WIDE flux
        # balances. Balancing per block would wrongly force each block to be individually
        # conservative when mass legitimately crosses a seam.
        self.outflow = []
        self.wall = domain.wall_mask()
        self.interior = np.where(~self.wall)[0]
        self.bnd = np.where(self.wall)[0]

    # ---- helpers ---------------------------------------------------------------
    def _flat(self, fields):
        return np.concatenate([fields[b].ravel() for b in range(len(self.d.blocks))])

    def _unflat(self, vec):
        out, o = {}, 0
        for b, bl in enumerate(self.d.blocks):
            n = int(np.prod(bl.shape))
            out[b] = vec[o:o + n].reshape(bl.shape); o += n
        return out

    def step(self):
        """Advance one step, repeating the Picard linearisation if asked."""
        if self.picard_iters <= 1:
            return self._step_once()
        u0 = (dict(self.u), dict(self.v), dict(self.w))
        p0 = dict(self.p)
        prev0 = self.u_prev
        convect, out = None, None
        for _k in range(self.picard_iters):
            if _k > 0:
                # restore the starting state so time advances ONCE, and rebuild A from the
                # latest u* -- that is what removes the O(dt) lag in the convecting velocity
                self.u, self.v, self.w = dict(u0[0]), dict(u0[1]), dict(u0[2])
                self.p, self.u_prev = dict(p0), prev0
            out = self._step_once(convect)
            convect = (dict(self.u), dict(self.v), dict(self.w))
        return out

    def _update_outflow(self):
        """Advective outflow update, then GLOBAL flux balancing."""
        from multiblock import face_axis_side, face_slice
        d = self.d
        for spec in self.outflow:
            b, fid, U_c = spec[0], spec[1], spec[2]
            kind = spec[3] if len(spec) > 3 else "convective"
            axis, side = face_axis_side(fid)
            blk = d.blocks[b]
            bs = face_slice(fid)
            isl = [slice(None)] * 3
            isl[axis] = 1 if side == 0 else -2
            isl = tuple(isl)
            # distance from the boundary node to the first interior node
            dn = np.sqrt((blk.x[bs] - blk.x[isl]) ** 2 + (blk.y[bs] - blk.y[isl]) ** 2
                         + (blk.z[bs] - blk.z[isl]) ** 2)
            # alpha = dt*U_c/h_n, WITHOUT PICT's factor 2: they are cell-centred, our nodes sit
            # ON the boundary, so the distance is h_n not h_n/2.
            # Dong's tangential condition is zero normal-gradient; the traction is carried by
            # the DIRICHLET PRESSURE instead, so the velocity is simply copied inward.
            t = 1.0 if kind == "dong" else 1.0 - 1.0 / (1.0 + self.dt * abs(U_c) / dn)
            for arr, bc in ((self.u, self.u_bc), (self.v, self.v_bc), (self.w, self.w_bc)):
                bc[b][bs] = bc[b][bs] - t * (bc[b][bs] - arr[b][isl])
                arr[b][bs] = bc[b][bs]
        # Flux balancing is needed ONLY for the singular all-Neumann system. A Dong outlet
        # carries a Dirichlet pressure, which makes the system non-singular, so its flux must
        # NOT be rescaled -- mass leaves as the solution dictates.
        conv = [sp for sp in self.outflow if (sp[3] if len(sp) > 3 else "convective") != "dong"]
        if not conv:
            return
        faces = [(sp[0], sp[1]) for sp in conv]
        fixed, free = d.boundary_flux_totals(self.u, self.v, self.w, faces)
        if abs(fixed + free) < 1e-14 or abs(free) < 1e-30:
            return
        scale = -fixed / free
        for sp in conv:
            b, fid = sp[0], sp[1]
            bs = face_slice(fid)
            for arr, bc in ((self.u, self.u_bc), (self.v, self.v_bc), (self.w, self.w_bc)):
                bc[b][bs] = bc[b][bs] * scale
                arr[b][bs] = bc[b][bs]

    def _step_once(self, convect=None):
        d, nb = self.d, len(self.d.blocks)
        if self.outflow:
            self._update_outflow()
        if convect is None:
            convect = (self.u, self.v, self.w)
        # BDF2 needs two levels, so the FIRST step must fall back to Backward Euler -- there
        # is no u^{n-1} yet. Using a zero previous level instead would silently corrupt the
        # first step and, with it, every convergence study built on short runs.
        bdf2 = (self.time_scheme == 'bdf2' and self.u_prev is not None)
        A = d.build_momentum_matrix(self.Js, self.ms, convect[0], convect[1], convect[2],
                                    self.nu, self.dt, bdf2=bdf2)
        Jg = self._flat({b: self.Js[b] for b in range(nb)})

        if self.scheme in ('incremental', 'rotational'):
            g = {b: d.gradient(b, self.p) for b in range(nb)}
            gp = [self._flat({b: g[b][c] for b in range(nb)}) for c in range(3)]
        else:
            gp = None

        # --- momentum predictor, all three components on the GLOBAL matrix
        has_wall = self.bnd.size > 0
        if has_wall:
            A_ii = A[self.interior][:, self.interior].tocsr()
            A_ib = A[self.interior][:, self.bnd].tocsr()
        star = []
        for k, (comp, bcs) in enumerate(((self.u, self.u_bc), (self.v, self.v_bc),
                                         (self.w, self.w_bc))):
            phi_n = self._flat(comp)
            cur = {b: comp[b].copy() for b in range(nb)}
            if bdf2:
                # J (2 u^n - u^{n-1}/2) / dt, against the 3J/2dt on the diagonal
                trans = (2.0 * phi_n - 0.5 * self._flat(self.u_prev[k])) / self.dt
            else:
                trans = phi_n / self.dt
            # incremental/rotational carry the OLD pressure gradient into the predictor
            # (PICT's applyPressureGradient=True); chorin does not.
            src = 0.0
            if self.velocity_source is not None:
                sk = self.velocity_source[k]
                src = sk if np.isscalar(sk) else self._flat(sk)
            # Momentum cross-diffusion, carried explicitly and iterated -- the deferred
            # correction the single-block solver applies. On a Cartesian grid it is identically
            # zero and costs one extra assembly; on a warped grid omitting it leaves the
            # momentum solving the orthogonal operator only.
            base = Jg * (trans - (gp[k] if gp is not None else 0.0) + src)
            rhs = base
            x = phi_n
            for _dc in range(self.momentum_dc_iters):
                if self.momentum_dc_iters > 1 or self.nu != 0.0:
                    cd = {b: d.cross_diffusion(b, cur) for b in range(nb)}
                    rhs = base + Jg * (self.nu * self._flat(cd))
                if has_wall:
                    # Dirichlet elimination, as the single-block solver does: solve only for
                    # the interior and move the known wall values across to the RHS.
                    phi_b = self._flat(bcs)[self.bnd]
                    xi, info = spla.bicgstab(A_ii, rhs[self.interior] - A_ib @ phi_b,
                                             x0=x[self.interior], rtol=self.tol, maxiter=20000)
                    x = np.zeros(A.shape[0]); x[self.interior] = xi; x[self.bnd] = phi_b
                else:
                    x, info = spla.bicgstab(A, rhs, x0=x, rtol=self.tol, maxiter=20000)
                cur = self._unflat(x)
            star.append(self._unflat(x))
        us, vs, ws = star

        # --- Gamma from the GLOBAL row sums, not per block
        rowsum = np.asarray(A.sum(axis=1)).ravel()
        coef_g = Jg / rowsum
        coef = self._unflat(coef_g)

        M = d.build_diffusion_matrix(self.Js, self.ms, coefs=[coef[b] for b in range(nb)])
        pD, pD_val = self._dong_nodes()
        if pD.size:
            # Dong outlet nodes carry a prescribed pressure, so they leave the unknown set and
            # the reduced matrix is NON-SINGULAR: no global pin and no compatibility projection.
            # Their continuity equation is dropped, which is what lets mass leave.
            mask = np.ones(M.shape[0], dtype=bool); mask[pD] = False
            free = np.arange(M.shape[0])[mask]
            M_fD = M[free][:, pD].tocsr()
        else:
            free = np.arange(M.shape[0])[1:]             # ONE pin for the whole domain
            M_fD = None
        M_ff = M[free][:, free].tocsr()

        phi_tot = {b: np.zeros(bl.shape) for b, bl in enumerate(d.blocks)}
        div_star = None
        # Fluxes are RECOMPUTED from the current velocities at the start of every corrector,
        # matching the single-block loop exactly. Carrying the corrected flux forward instead
        # changes the algorithm: it still converges and still reports a divergence of 1e-16,
        # but the velocity drifts ~1% from the single-block trajectory. Only the FINAL
        # divergence is reported from the corrected flux, which is what makes the single-block
        # solver print 1e-16 while the velocity field itself is not solenoidal on a collocated
        # grid.
        Fb = {}
        for _ in range(self.corrector_steps):
            divF = {}
            for b in range(nb):
                Fb[b] = d.face_fluxes(b, us, vs, ws)
                divF[b] = d.divergence(b, Fb[b], self.Js[b])
            if div_star is None:
                div_star = {b: divF[b].copy() for b in range(nb)}   # predictor divergence
            rhs = -(Jg * self._flat(divF))
            if M_fD is None:
                rhs = rhs - rhs.mean()                    # compatibility (M is singular)
            else:
                # NON-singular: do NOT project the mean out. That would remove the genuine
                # inflow/outflow imbalance the outlet exists to carry.
                rhs = rhs - 0.0
            b_free = rhs[free] - (M_fD @ pD_val if M_fD is not None else 0.0)
            if self.implicit_cross:
                sol = self._solve_cross(M, M_ff, free, rhs, coef, Jg)
            elif M_fD is not None:
                sol, info = spla.bicgstab(M_ff, b_free, rtol=self.tol, maxiter=20000)
            else:
                sol, info = spla.cg(M_ff, b_free, rtol=self.tol, maxiter=20000)
            pv = np.zeros(M.shape[0]); pv[free] = sol
            if M_fD is not None:
                pv[pD] = pD_val
            pp = self._unflat(pv)
            for b in range(nb):
                gx, gy, gz = d.gradient(b, pp)
                us[b] = us[b] - coef[b] * gx
                vs[b] = vs[b] - coef[b] * gy
                ws[b] = ws[b] - coef[b] * gz
            if has_wall:                                   # re-impose the wall values
                for arr, bc in ((us, self.u_bc), (vs, self.v_bc), (ws, self.w_bc)):
                    fl = self._flat(arr); fl[self.bnd] = self._flat(bc)[self.bnd]
                    upd = self._unflat(fl)
                    for b in range(nb):
                        arr[b] = upd[b]
            for b in range(nb):
                # The flux correction must use the SAME operator the pressure was solved
                # with. Correcting with the orthogonal part only, while solving the full
                # operator, leaves the corrected flux non-solenoidal -- measured divergence
                # 3.2e-02 against 1.5e-13 for the single-block solver.
                Phi = d.pressure_face_fluxes(b, pp, coef[b], coef,
                                             include_cross=self.implicit_cross)
                Fb[b] = [Fb[b][a] - Phi[a] for a in range(3)]
                phi_tot[b] = phi_tot[b] + pp[b]

        if self.scheme == 'chorin':
            self.p = phi_tot                              # recomputed, never accumulated
        elif self.scheme == 'incremental':
            self.p = {b: self.p[b] + phi_tot[b] for b in range(nb)}
        elif self.scheme == 'rotational':
            # p <- p + phi - nu*div(u*), cancelling the spurious dp/dn = 0 the projection
            # otherwise imposes
            self.p = {b: self.p[b] + phi_tot[b] - self.nu * div_star[b] for b in range(nb)}
        else:
            raise ValueError(f"unknown scheme {self.scheme!r}")
        self.u_prev = (dict(self.u), dict(self.v), dict(self.w))
        self.u, self.v, self.w = us, vs, ws
        div = 0.0
        for b in range(nb):
            div = max(div, np.abs(d.divergence(b, Fb[b], self.Js[b])).max())
        return div

    def _dong_nodes(self):
        """
        (global indices, prescribed pressure) for every Dong outflow node.

            p = nu d(u.n)/dn - 1/2 |u|^2 Theta,   Theta = 1/2 (1 - tanh(u.n / (U0 delta)))

        delta = 0.01, not the 0.05 first used single-block: Theta does not vanish where u_n -> 0,
        so at a no-slip wall junction it leaves a spurious near-wall traction of size
        O(U0^2 delta^2). Measured single-block, that cost a factor of ~100 in accuracy.
        """
        from multiblock import face_axis_side, face_slice
        d = self.d
        idx, val = [], []
        for spec in self.outflow:
            if (spec[3] if len(spec) > 3 else "convective") != "dong":
                continue
            b, fid = spec[0], spec[1]
            axis, side = face_axis_side(fid)
            blk = d.blocks[b]
            bs = face_slice(fid)
            isl = [slice(None)] * 3
            isl[axis] = 1 if side == 0 else -2
            isl = tuple(isl)
            _, mb = d.block_metrics_cached(b)
            key = ("xi", "eta", "zeta")[axis]
            nx_, ny_, nz_ = mb[f"{key}_x"][bs], mb[f"{key}_y"][bs], mb[f"{key}_z"][bs]
            nrm = np.sqrt(nx_ ** 2 + ny_ ** 2 + nz_ ** 2)
            sg = -1.0 if side == 0 else 1.0
            nx_, ny_, nz_ = sg * nx_ / nrm, sg * ny_ / nrm, sg * nz_ / nrm
            ub, vb, wb = self.u[b][bs], self.v[b][bs], self.w[b][bs]
            un = ub * nx_ + vb * ny_ + wb * nz_
            un_i = self.u[b][isl] * nx_ + self.v[b][isl] * ny_ + self.w[b][isl] * nz_
            dn = np.sqrt((blk.x[bs] - blk.x[isl]) ** 2 + (blk.y[bs] - blk.y[isl]) ** 2
                         + (blk.z[bs] - blk.z[isl]) ** 2)
            U0 = max(float(np.max(np.abs(un))), 1e-12)
            th = 0.5 * (1.0 - np.tanh(un / (U0 * self.dong_delta)))
            pv = self.nu * (un - un_i) / dn - 0.5 * (ub ** 2 + vb ** 2 + wb ** 2) * th
            idx.append(d.global_ids(b)[bs].ravel())
            val.append(pv.ravel())
        if not idx:
            return np.empty(0, dtype=int), np.empty(0)
        i = np.concatenate(idx); v = np.concatenate(val)
        u_, first = np.unique(i, return_index=True)     # a corner may sit on two faces
        return u_, v[first]

    def _solve_cross(self, M, M_ff, free, rhs, coef, Jg):
        """
        Solve the FULL operator  M p - J div(Phi_cross(p)) = rhs, matrix-free.

        The cross part is applied per block on padded fields, so it is exactly the operator the
        deferred correction would converge to. Preconditioned by an INCOMPLETE factorisation of
        the orthogonal part M: M^-1 A = I - M^-1 J div(Phi_cross), whose spectrum clusters at 1,
        so Krylov resolves it in a few iterations. spilu rather than splu because exact LU on a
        3D 7-point matrix suffers catastrophic fill-in; the factorisation is cached and reused,
        since a preconditioner must be spectrally close, never current.
        """
        d, nb = self.d, len(self.d.blocks)
        nf = len(free)

        def apply(xf):
            v = np.zeros(M.shape[0]); v[free] = xf
            pb = self._unflat(v)
            dc = {}
            for b in range(nb):
                Phi = d.pressure_face_fluxes(b, pb, coef[b], coef,
                                             include_orth=False, include_cross=True)
                dc[b] = d.divergence(b, Phi, self.Js[b])
            return ((M @ v) - Jg * self._flat(dc))[free]

        op = spla.LinearOperator((nf, nf), matvec=apply, dtype=float)
        if self._prec is None or self._prec[0] != nf:
            lu = spla.spilu(M_ff.tocsc(), drop_tol=1e-3, fill_factor=10)
            self._prec = (nf, lu)
        prec = spla.LinearOperator((nf, nf), matvec=self._prec[1].solve, dtype=float)
        sol, info = spla.bicgstab(op, rhs[free], M=prec, rtol=self.tol, maxiter=20000)
        if info != 0:
            sol, info = spla.lgmres(op, rhs[free], M=prec, rtol=self.tol, maxiter=5000)
        return sol
