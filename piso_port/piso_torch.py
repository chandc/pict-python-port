"""
A fully differentiable PISO step, assembled entirely from sparse matrices and linear solves.

Every operator in a PISO step is linear in its input except the momentum matrix's dependence
on the convecting velocity (frozen here, Picard-style). So the whole step can be written as

    u* = A^{-1} (J u^n / dt + S)          non-symmetric solve
    p  = M^{-1} (-J D u*)                 symmetric SINGULAR solve
    u  = u* - G p                         sparse mat-vec

with D the flux-divergence operator and G the pressure-correction operator, both built once as
sparse matrices. Writing them as matrices rather than as NumPy stencil code is what makes the
adjoint exact and checkable: the transpose is D.T, not a hand-derived operator that has to be
re-derived (and re-debugged) every time the stencil changes.

Cartesian periodic only. On a uniform grid the off-diagonal metric terms g^ij (i != j) vanish
identically, so the non-orthogonal deferred correction is not needed and the projection is a
single solve -- no fixed point to differentiate through. That is deliberate: this module is for
validating the network-to-solver gradient path, not the deferred correction (see
reference/nn_piso_plan.md section 5).
"""
import numpy as np
import scipy.sparse as sparse
import torch

from phase1_grid_metrics import make_grid, compute_numerical_metrics
from phase3_momentum import build_momentum_matrix_7point, build_conservative_diffusion_matrix
from adjoint_piso import LinearSolve, csr_pattern
from momentum_torch import MomentumAssembler

_KEYS = [("xi_x", "xi_y", "xi_z"), ("eta_x", "eta_y", "eta_z"), ("zeta_x", "zeta_y", "zeta_z")]


def build_divergence_matrix(shape, J, metrics, h):
    """
    D: (u, v, w) stacked -> flux divergence, one row per cell, 3N columns.

    Face flux is F = 0.5*(JU_C + JU_N), so the cell divergence telescopes to a central
    difference of the volume-weighted contravariant components:

        div_P = (1/J_P) * sum_ax [ (JU)_{P+e} - (JU)_{P-e} ] / (2 h_ax)
    """
    N = int(np.prod(shape))
    idx = np.arange(N).reshape(shape)
    rows, cols, vals = [], [], []
    for ax in range(3):
        for c in range(3):
            Jm = J * metrics[_KEYS[ax][c]]
            for sgn in (+1, -1):
                nb = np.roll(idx, -sgn, axis=ax)          # neighbour P + sgn*e_ax
                coef = sgn * np.roll(Jm, -sgn, axis=ax) / (2 * h[ax] * J)
                rows.append(idx.ravel())
                cols.append(nb.ravel() + c * N)
                vals.append(coef.ravel())
    return sparse.coo_matrix((np.concatenate(vals),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(N, 3 * N)).tocsr()


def build_gradient_matrices(shape, metrics, h, gamma):
    """G_c: p -> gamma * (grad p)_c, one sparse matrix per velocity component."""
    N = int(np.prod(shape))
    idx = np.arange(N).reshape(shape)
    mats = []
    for c in range(3):
        rows, cols, vals = [], [], []
        for ax in range(3):
            m_c = metrics[_KEYS[ax][c]]
            for sgn in (+1, -1):
                nb = np.roll(idx, -sgn, axis=ax)
                coef = sgn * gamma * m_c / (2 * h[ax])
                rows.append(idx.ravel())
                cols.append(nb.ravel())
                vals.append(coef.ravel())
        mats.append(sparse.coo_matrix((np.concatenate(vals),
                                       (np.concatenate(rows), np.concatenate(cols))),
                                      shape=(N, N)).tocsr())
    return mats


def _sp2torch(A):
    A = A.tocoo()
    return torch.sparse_coo_tensor(np.vstack([A.row, A.col]), A.data, A.shape).coalesce()


class DifferentiablePISO:
    """One PISO step, differentiable in an additive momentum source S."""

    def __init__(self, n=16, nu=0.05, dt=0.05, exact_A=False):
        """
        exact_A=True assembles the momentum matrix differentiably, so dL/dA propagates back to
        the convecting velocity (PICT's SetupAdvectionMatrixEulerImplicit_GRAD). Gamma =
        J/A_diag, and hence M and G, are still taken detached -- so this captures A's
        dependence but not Gamma's. How much of the exact gradient that recovers is measured
        rather than assumed: see nn_stage4_bias.py.
        """
        self.exact_A = exact_A
        self.shape = (n, n, n)
        self.x, self.y, self.z, *h = make_grid(n, warp=1e-9, periodic=True)
        self.h = tuple(h)
        self.J, self.m = compute_numerical_metrics(self.x, self.y, self.z, *self.h,
                                                   periodic=True)
        self.nu, self.dt, self.N = nu, dt, n ** 3

    def build(self, u, v, w):
        """Assemble everything that depends on the convecting velocity."""
        n = self.shape[0]
        if self.exact_A:
            if not hasattr(self, "_asm"):
                self._asm = MomentumAssembler(self.shape, self.J, self.m, self.h,
                                              self.nu, self.dt)
            ut = [f if torch.is_tensor(f) else torch.as_tensor(np.asarray(f).ravel())
                  for f in (u, v, w)]
            self.A_val_t = self._asm.values(*[t.reshape(-1) for t in ut])
            A = self._asm.csr(self.A_val_t.detach().numpy())
        else:
            un = [f.detach().numpy() if torch.is_tensor(f) else np.asarray(f) for f in (u, v, w)]
            un = [f.reshape(self.shape) for f in un]
            A = build_momentum_matrix_7point(n, n, n, self.J, self.m, *self.h, *un,
                                             self.nu, periodic=True)
            A = (A + sparse.diags(self.J.ravel() / self.dt)).tocsr()
            self.A_val_t = None
        gamma = self.J / A.diagonal().reshape(self.J.shape)

        M = build_conservative_diffusion_matrix(n, n, n, *self.h, self.J, self.m,
                                                coef=gamma, periodic=True)
        if self.exact_A:
            self.A_pat = (self._asm.rc, (self.N, self.N), self.A_val_t)
        else:
            self.A_pat = csr_pattern(A)
        self.M_pat = csr_pattern(M.tocsr())
        self.D = _sp2torch(build_divergence_matrix(self.shape, self.J, self.m, self.h))
        self.G = [_sp2torch(g) for g in
                  build_gradient_matrices(self.shape, self.m, self.h, gamma)]
        self.Jt = torch.as_tensor(self.J.ravel())
        return self

    def step(self, u, v, w, S):
        """S: (3, N) additive momentum source. Returns (u, v, w) after one step."""
        Aidx, Ashape, Aval = self.A_pat
        Midx, Mshape, Mval = self.M_pat

        stars = []
        for c, f in enumerate((u, v, w)):
            b = self.Jt * torch.as_tensor(f.ravel()) / self.dt + S[c]
            stars.append(LinearSolve.apply(Aval, b, (Aidx, Ashape), False, False))

        div = torch.sparse.mm(self.D, torch.cat(stars).unsqueeze(1)).squeeze(1)
        rhs = -self.Jt * div
        p = LinearSolve.apply(Mval, rhs, (Midx, Mshape), True, True)

        return tuple(stars[c] - torch.sparse.mm(self.G[c], p.unsqueeze(1)).squeeze(1)
                     for c in range(3))
