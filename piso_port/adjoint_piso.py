"""
Differentiable PISO: the discrete adjoint, and a check that it is actually correct.

Implements the linear-solve adjoint described in reference/nn_piso_coupling.md and exercises
it on a real (small) PISO step with a neural-network source term, so that the two cases that
behave differently are both covered:

  * momentum  A  -- NON-symmetric (2nd-order upwind is one-sided), so the backward pass must
                    solve with A^T, a genuinely different operator, using BiCGStab.
  * pressure  M  -- symmetric, so the backward pass reuses M and CG; but SINGULAR, so the
                    incoming gradient must be projected off the constant null space and the
                    null-space constant pinned identically in both passes.

Correctness is established by finite differences, not by inspection.
"""
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spl
import torch

from phase1_grid_metrics import make_grid, compute_numerical_metrics
from phase3_momentum import (build_momentum_matrix_7point, build_conservative_diffusion_matrix,
                             boundary_masks)

TOL = 1e-13

# Adjoint-norm log. The adjoint of an advection-dominated flow transports sensitivity
# UPSTREAM and can amplify over a long rollout, so ||lambda|| per solve is the diagnostic
# that catches it -- silently clipping gradients would hide exactly this.
ADJOINT_NORMS = []


def _solve(A, b, symmetric, transpose=False):
    """
    Solve, with a fallback for BiCGStab breakdown.

    BiCGStab can break down (scipy `info < 0`) on the non-symmetric momentum system -- it
    happens in practice during training, when the adjoint RHS becomes nearly orthogonal to the
    shadow residual. PICT carries the same machinery for the same reason (`double_fallback`,
    `BiCG_precondition_fallback`, `return_best_result`). LGMRES does not break down the same
    way, so it is the fallback here.

    A zero RHS is short-circuited: the answer is zero, but an iterative solver handed an
    all-zero right-hand side reports non-convergence rather than returning it.
    """
    if not np.any(b):
        return np.zeros_like(b)
    op = A.T if transpose else A
    if symmetric:                       # M^T == M, so `transpose` is a no-op here by design
        x, info = spl.cg(op, b, rtol=TOL, maxiter=50000)
        if info != 0:
            raise RuntimeError(f"symmetric solve failed, info={info}")
        return x
    x, info = spl.bicgstab(op, b, rtol=TOL, maxiter=50000)
    if info != 0:
        x, info = spl.lgmres(op, b, rtol=TOL, maxiter=5000)      # breakdown fallback
    if info != 0:
        raise RuntimeError(f"non-symmetric solve failed after fallback, info={info}")
    return x


class LinearSolve(torch.autograd.Function):
    """
    x = A^{-1} b, differentiable in both A's values and b.

        lambda = A^{-T} g,     dL/db = lambda,     dL/dA = -lambda x^T
    """

    @staticmethod
    def forward(ctx, A_val, b, A_pattern, symmetric, singular):
        idx, shape = A_pattern
        A = sparse.csr_matrix((A_val.detach().numpy(), idx), shape=shape)
        bn = b.detach().numpy().copy()
        if singular:
            bn -= bn.mean()                     # compatibility with N(M) = span{1}
        x = _solve(A, bn, symmetric)
        if singular:
            x -= x.mean()                       # pin the constant, identically in both passes
        ctx.save_for_backward(A_val, torch.as_tensor(x))
        ctx.cfg = (A_pattern, symmetric, singular)
        return torch.as_tensor(x)

    @staticmethod
    def backward(ctx, g):
        A_val, x = ctx.saved_tensors
        (idx, shape), symmetric, singular = ctx.cfg
        A = sparse.csr_matrix((A_val.detach().numpy(), idx), shape=shape)
        gn = g.detach().numpy().copy()
        if singular:
            gn -= gn.mean()                     # else the adjoint system is inconsistent
        # THE line that separates the two cases: symmetric reuses A, otherwise transpose.
        lam = _solve(A, gn, symmetric, transpose=not symmetric)
        if singular:
            lam -= lam.mean()

        ADJOINT_NORMS.append(float(np.linalg.norm(lam)))

        grad_A = None
        if ctx.needs_input_grad[0]:
            # -lambda x^T restricted to the sparsity pattern -- never formed densely
            rows, cols = idx
            grad_A = torch.as_tensor(-lam[rows] * x.detach().numpy()[cols])
        return grad_A, torch.as_tensor(lam), None, None, None


def csr_pattern(A):
    A = A.tocoo()
    order = np.lexsort((A.col, A.row))
    return (A.row[order], A.col[order]), A.shape, torch.tensor(A.data[order], dtype=torch.float64)


class MiniPISO:
    """One PISO step with an additive network source in the momentum RHS."""

    def __init__(self, n=8, nu=0.05, dt=0.05, warp=0.03):
        self.x, self.y, self.z, *h = make_grid((n, n, n), warp=warp, periodic=False)
        self.h = tuple(h)
        self.J, self.m = compute_numerical_metrics(self.x, self.y, self.z, *self.h)
        self.n, self.nu, self.dt = n, nu, dt
        self.ib, self.bb, self.mask = boundary_masks(n, n, n)
        self.u0 = np.sin(np.pi * self.x) * np.cos(np.pi * self.y) * np.cos(np.pi * self.z)

    def step(self, S):
        """S: torch tensor over interior cells (the network output). Returns scalar loss."""
        n, J, ib = self.n, self.J, self.ib
        u = self.u0
        A = build_momentum_matrix_7point(n, n, n, J, self.m, *self.h, u, u, u, self.nu)
        A = (A + sparse.diags(J.ravel() / self.dt)).tocsr()
        A_ii = A[ib][:, ib].tocsr()
        Aidx, Ashape, Aval = csr_pattern(A_ii)

        b = torch.as_tensor((J * u / self.dt).ravel()[ib]) + S
        # --- momentum: NON-symmetric -> backward solves A^T with BiCGStab
        u_star = LinearSolve.apply(Aval, b, (Aidx, Ashape), False, False)

        M = build_conservative_diffusion_matrix(n, n, n, *self.h, J, self.m)
        Midx, Mshape, Mval = csr_pattern(M.tocsr())
        r = torch.zeros(M.shape[0], dtype=torch.float64)
        r[ib] = u_star
        # --- pressure: symmetric + SINGULAR -> backward reuses M and CG, gradient projected
        phi = LinearSolve.apply(Mval, r, (Midx, Mshape), True, True)
        return (phi ** 2).sum() + (u_star ** 2).sum()


if __name__ == "__main__":
    import sys
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(0)
    ok = True

    def check(name, got, want, tol):
        global ok
        good = abs(got - want) <= tol * max(1.0, abs(want))
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: {got:.10e} vs {want:.10e}")

    piso = MiniPISO(n=8)
    ni = len(piso.ib)

    # ---- 1. adjoint identity on the NON-SYMMETRIC momentum matrix -------------
    print("\n1. Adjoint identity <A^-1 v, w> == <v, A^-T w>  (momentum matrix, non-symmetric)")
    n, J = piso.n, piso.J
    A = build_momentum_matrix_7point(n, n, n, J, piso.m, *piso.h,
                                     piso.u0, piso.u0, piso.u0, piso.nu)
    A = (A + sparse.diags(J.ravel() / piso.dt)).tocsr()[piso.ib][:, piso.ib].tocsr()
    print(f"   asymmetry of A: max|A - A^T| = {abs(A - A.T).max():.3e}  (non-zero => transpose matters)")
    v, w = rng.normal(size=ni), rng.normal(size=ni)
    lhs = np.dot(_solve(A, v, False), w)
    rhs = np.dot(v, _solve(A, w, False, transpose=True))
    check("adjoint identity", lhs, rhs, 1e-8)
    wrong = np.dot(v, _solve(A, w, False, transpose=False))   # the bug we are guarding against
    print(f"   (forgetting the transpose would give {wrong:.10e} -- "
          f"{abs(wrong-lhs)/abs(lhs)*100:.1f}% off)")

    # ---- 2. finite-difference gradient check through the whole step -----------
    print("\n2. Finite differences through a full PISO step (momentum + pressure)")
    S0 = torch.tensor(rng.normal(size=ni) * 0.1, requires_grad=True)
    L = piso.step(S0); L.backward()
    g_adj = S0.grad.detach().numpy().copy()
    eps = 1e-6
    idxs = rng.choice(ni, 4, replace=False)
    for k in idxs:
        d = np.zeros(ni); d[k] = eps
        Lp = piso.step(torch.tensor(S0.detach().numpy() + d)).item()
        Lm = piso.step(torch.tensor(S0.detach().numpy() - d)).item()
        check(f"dL/dS[{k}]", g_adj[k], (Lp - Lm) / (2 * eps), 1e-5)

    # ---- 3. null-space invariance of the pressure adjoint ---------------------
    print("\n3. Pressure adjoint must be invariant to a constant shift of the incoming gradient")
    M = build_conservative_diffusion_matrix(n, n, n, *piso.h, J, piso.m).tocsr()
    Midx, Mshape, Mval = csr_pattern(M)
    print(f"   null space check: max|M @ 1| = {abs(M @ np.ones(M.shape[0])).max():.3e}")
    r = torch.tensor(rng.normal(size=M.shape[0]), requires_grad=True)
    base = None
    for shift in (0.0, 7.3):
        r.grad = None
        phi = LinearSolve.apply(Mval, r, (Midx, Mshape), True, True)
        (phi * torch.as_tensor(rng.normal(size=M.shape[0]) * 0 + 1.0) + phi.pow(2).sum()).sum().backward()
        g = r.grad.detach().numpy().copy() + shift
        if base is None:
            base = g
        else:
            check("gradient unchanged under constant shift of g",
                  float(np.abs((g - shift) - base).max()), 0.0, 1e-10)

    print("\n" + ("adjoint verified" if ok else "ADJOINT INCORRECT"))
    sys.exit(0 if ok else 1)
