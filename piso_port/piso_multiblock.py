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

SCOPE: fully periodic, Cartesian, central convection, chorin; BDF2 in time (BE on the
first step, which has no previous level). That is deliberate -- it is the
configuration where the multi-block result can be compared against the single-block solver
EXACTLY, with no cross terms (Cartesian) and no boundary conditions (periodic) to confound it.
Warped multi-block additionally needs the implicit cross operator across seams; walls need the
face-type registry. Both are separate increments.
"""
import numpy as np
import scipy.sparse.linalg as spla


class MultiBlockPISO:
    def __init__(self, domain, nu, dt, corrector_steps=2, tol=1e-13, time_scheme='bdf2'):
        self.d = domain
        self.nu, self.dt = nu, dt
        # BDF2 by default, matching the single-block solver. The single-block Stokes study
        # measured order 2.00 with no-slip walls once the Richardson triple was taken inside
        # the asymptotic range, so second-order time is a real property worth carrying across
        # blocks rather than quietly dropping to Backward Euler.
        self.time_scheme = time_scheme
        self.u_prev = None
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
        d, nb = self.d, len(self.d.blocks)
        # BDF2 needs two levels, so the FIRST step must fall back to Backward Euler -- there
        # is no u^{n-1} yet. Using a zero previous level instead would silently corrupt the
        # first step and, with it, every convergence study built on short runs.
        bdf2 = (self.time_scheme == 'bdf2' and self.u_prev is not None)
        A = d.build_momentum_matrix(self.Js, self.ms, self.u, self.v, self.w,
                                    self.nu, self.dt, bdf2=bdf2)
        Jg = self._flat({b: self.Js[b] for b in range(nb)})

        # --- momentum predictor, all three components on the GLOBAL matrix
        star = []
        for k, comp in enumerate((self.u, self.v, self.w)):
            phi_n = self._flat(comp)
            if bdf2:
                # J (2 u^n - u^{n-1}/2) / dt, against the 3J/2dt on the diagonal
                trans = (2.0 * phi_n - 0.5 * self._flat(self.u_prev[k])) / self.dt
            else:
                trans = phi_n / self.dt
            rhs = Jg * trans
            x, info = spla.bicgstab(A, rhs, x0=phi_n, rtol=self.tol, maxiter=20000)
            star.append(self._unflat(x))
        us, vs, ws = star

        # --- Gamma from the GLOBAL row sums, not per block
        rowsum = np.asarray(A.sum(axis=1)).ravel()
        coef_g = Jg / rowsum
        coef = self._unflat(coef_g)

        M = d.build_diffusion_matrix(self.Js, self.ms, coefs=[coef[b] for b in range(nb)])
        free = np.arange(M.shape[0])[1:]                 # ONE pin for the whole domain
        M_ff = M[free][:, free].tocsr()

        phi_tot = {b: np.zeros(bl.shape) for b, bl in enumerate(d.blocks)}
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
            rhs = -(Jg * self._flat(divF))
            rhs = rhs - rhs.mean()                        # compatibility (M is singular)
            sol, info = spla.cg(M_ff, rhs[free], rtol=self.tol, maxiter=20000)
            pv = np.zeros(M.shape[0]); pv[free] = sol
            pp = self._unflat(pv)
            for b in range(nb):
                gx, gy, gz = d.gradient(b, pp)
                us[b] = us[b] - coef[b] * gx
                vs[b] = vs[b] - coef[b] * gy
                ws[b] = ws[b] - coef[b] * gz
                Phi = d.pressure_face_fluxes(b, pp, coef[b], coef)
                Fb[b] = [Fb[b][a] - Phi[a] for a in range(3)]
                phi_tot[b] = phi_tot[b] + pp[b]

        self.p = phi_tot                                  # chorin: recomputed, not accumulated
        self.u_prev = (dict(self.u), dict(self.v), dict(self.w))
        self.u, self.v, self.w = us, vs, ws
        div = 0.0
        for b in range(nb):
            div = max(div, np.abs(d.divergence(b, Fb[b], self.Js[b])).max())
        return div
