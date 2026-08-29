"""
Differentiable assembly of the momentum matrix.

The momentum matrix A depends on the convecting velocity through the contravariant components
and the SOU coefficients. Freezing that dependence (Picard) gives a cheaper but BIASED
gradient; propagating it gives the exact one. This module builds A's values as a torch tensor
so autograd carries dL/dA back to u^n -- the analogue of PICT's
SetupAdvectionMatrixEulerImplicit_GRAD.

Cartesian periodic only, matching piso_torch.py.

Note on the upwind switch: SOU selects its stencil on sign(a), which is not differentiable at
a = 0. The mask is taken detached, so the result carries the correct almost-everywhere
derivative and the measure-zero switching set is ignored -- standard practice, and worth
stating rather than leaving implicit.
"""
import numpy as np
import scipy.sparse as sparse
import torch

from src.phase3_momentum import build_conservative_diffusion_matrix

_KEYS = [("xi_x", "xi_y", "xi_z"), ("eta_x", "eta_y", "eta_z"), ("zeta_x", "zeta_y", "zeta_z")]


class MomentumAssembler:
    """Precomputes everything independent of u; call values() per step."""

    def __init__(self, shape, J, metrics, h, nu, dt):
        self.shape, self.h, self.dt = shape, h, dt
        N = int(np.prod(shape))
        self.N = N
        idx = np.arange(N).reshape(shape)
        self.idx = idx
        self.Jt = torch.as_tensor(J.ravel())
        self.met = [[torch.as_tensor((J * metrics[_KEYS[ax][c]]).ravel()) for c in range(3)]
                    for ax in range(3)]

        # constant part: diffusion + transient, taken straight from the validated assembler
        D = (nu * build_conservative_diffusion_matrix(*shape, *h, J, metrics, periodic=True)
             + sparse.diags(J.ravel() / dt)).tocoo()
        self.const_rc = (D.row.copy(), D.col.copy())
        self.const_val = torch.as_tensor(D.data.copy())

        # advection sparsity: every axis contributes offsets -2,-1,+1,+2 plus the diagonal
        rows, cols, self.slots = [], [], []
        for ax in range(3):
            for off in (-1, -2, 1, 2):
                rows.append(idx.ravel())
                cols.append(np.roll(idx, -off, axis=ax).ravel())
                self.slots.append((ax, off))
        rows.append(idx.ravel()); cols.append(idx.ravel())          # advection diagonal
        self.adv_rc = (np.concatenate(rows), np.concatenate(cols))
        self.rc = (np.concatenate([self.const_rc[0], self.adv_rc[0]]),
                   np.concatenate([self.const_rc[1], self.adv_rc[1]]))

    def values(self, u, v, w):
        """A's values on the fixed pattern, differentiable in (u, v, w)."""
        parts, diag = [], torch.zeros(self.N, dtype=torch.float64)
        per_axis = {}
        for ax in range(3):
            U = self.met[ax][0]*u + self.met[ax][1]*v + self.met[ax][2]*w
            a = U / self.h[ax]
            pos = (a > 0).to(a.dtype).detach()       # a.e. derivative; switch set has measure 0
            neg = 1.0 - pos
            per_axis[ax] = {
                -1: -2.0 * a * pos,
                -2:  0.5 * a * pos,
                 1:  2.0 * a * neg,
                 2: -0.5 * a * neg,
            }
            diag = diag + 1.5 * a * pos - 1.5 * a * neg
        for ax, off in self.slots:
            parts.append(per_axis[ax][off])
        parts.append(diag)
        return torch.cat([self.const_val, torch.cat(parts)])

    def csr(self, vals):
        return sparse.csr_matrix((vals, self.rc), shape=(self.N, self.N))
