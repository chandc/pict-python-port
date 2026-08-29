"""
Stage 1 of reference/nn_piso_plan.md: one PISO step, one learnable scalar.

The smallest thing that can be trained through the solver. A single scalar c scales a fixed
forcing shape, S = c * Phi(x), injected into the momentum RHS -- mirroring PICT's
`velocitySource` hook and its SetupAdvectionVelocityEulerImplicitRHS_GRAD.

Two gates, and the second is the one that matters:

  1. dL/dc matches central finite differences.
  2. RECOVERY -- with c_true known, gradient descent from c = 0 must find it.

Gate 2 exists because a gradient can have the right magnitude and the wrong sign, and a
magnitude-only check passes that happily. With one parameter, a sign error is unmissable.
"""
import sys
import numpy as np
import torch

from src.adjoint_piso import MiniPISO, LinearSolve, csr_pattern
import scipy.sparse as sparse
from src.phase3_momentum import build_momentum_matrix_7point, build_conservative_diffusion_matrix


class ForcedPISO(MiniPISO):
    """One PISO step with S = c * Phi, returning the resulting velocity field."""

    def __init__(self, n=8, **kw):
        super().__init__(n=n, **kw)
        # a fixed, arbitrary forcing shape over the interior cells
        self.phi_shape = np.sin(2 * np.pi * self.x) * np.cos(np.pi * self.y)
        self.phi_shape = self.phi_shape.ravel()[self.ib]

    def velocity(self, c):
        n, J, ib = self.n, self.J, self.ib
        u = self.u0
        A = build_momentum_matrix_7point(n, n, n, J, self.m, *self.h, u, u, u, self.nu)
        A = (A + sparse.diags(J.ravel() / self.dt)).tocsr()
        Aidx, Ashape, Aval = csr_pattern(A[ib][:, ib].tocsr())

        S = c * torch.as_tensor(self.phi_shape)                     # the learnable term
        b = torch.as_tensor((J * u / self.dt).ravel()[ib]) + S
        u_star = LinearSolve.apply(Aval, b, (Aidx, Ashape), False, False)

        M = build_conservative_diffusion_matrix(n, n, n, *self.h, J, self.m)
        Midx, Mshape, Mval = csr_pattern(M.tocsr())
        r = torch.zeros(M.shape[0], dtype=torch.float64)
        r[ib] = u_star
        phi = LinearSolve.apply(Mval, r, (Midx, Mshape), True, True)

        Gamma = torch.as_tensor((J / A.diagonal().reshape(J.shape)).ravel()[ib])
        return u_star - Gamma * phi[ib]      # corrected interior velocity


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    ok = True

    def check(name, got, want, tol):
        global ok
        good = abs(got - want) <= tol * max(1.0, abs(want))
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: {got:.10e} vs {want:.10e}")

    sim = ForcedPISO(n=8)
    C_TRUE = 0.7
    with torch.no_grad():
        target = sim.velocity(torch.tensor(C_TRUE))

    def loss(c):
        return ((sim.velocity(c) - target) ** 2).sum()

    # ---- Gate 1: gradient vs central finite differences -----------------------
    print("\nGate 1: dL/dc vs central finite differences")
    c = torch.tensor(0.15, requires_grad=True)
    L = loss(c); L.backward()
    g_adj = c.grad.item()
    eps = 1e-6
    with torch.no_grad():
        g_fd = (loss(torch.tensor(0.15 + eps)).item()
                - loss(torch.tensor(0.15 - eps)).item()) / (2 * eps)
    check("dL/dc", g_adj, g_fd, 1e-6)
    print(f"   (sign: adjoint {'negative' if g_adj < 0 else 'positive'}, and c=0.15 < "
          f"c_true={C_TRUE}, so the gradient MUST be negative)")
    ok &= (g_adj < 0)

    # ---- Gate 2: recovery ----------------------------------------------------
    print(f"\nGate 2: recover c_true = {C_TRUE} from c = 0 by gradient descent")
    c = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.Adam([c], lr=0.05)
    for it in range(400):
        opt.zero_grad(); L = loss(c); L.backward(); opt.step()
        if it % 100 == 0:
            print(f"   iter {it:3d}  c = {c.item():.6f}   loss = {L.item():.3e}")
    print(f"   final     c = {c.item():.6f}   (true {C_TRUE})")
    check("recovered c", c.item(), C_TRUE, 1e-4)

    print("\n" + ("Stage 1 gates passed" if ok else "STAGE 1 FAILED"))
    sys.exit(0 if ok else 1)
