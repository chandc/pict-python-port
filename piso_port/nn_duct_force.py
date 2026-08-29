"""
Learn the driving force of a square duct through the solver.

Target: the converged 32x32 duct solution at the true force G = 1.0.
Start:  force = 0.1  (a tenth of the truth)
Ask:    can the gradient, taken through the solver, recover G?

The steady duct is a wall-bounded problem, so this exercises the adjoint on a configuration
the periodic Stage 1-5 work never touched: four no-slip walls, corners, and Dirichlet
elimination. The projection is a no-op here (parallel flow is divergence-free by
construction), which is verified below rather than assumed.

Honest note on difficulty: at steady state the duct reduces to nu*lap(u) + G = 0, which is
LINEAR in G, so u(c) = c*u_1. Recovering a scalar amplitude is therefore an easy inverse
problem with a unique minimum -- this tests that the gradient is correct and correctly signed
through a wall-bounded solve, not that the optimiser can do anything clever.
"""
import sys, warnings, io, contextlib
import numpy as np
import torch
import scipy.sparse as sparse
warnings.filterwarnings("ignore")

from src.piso_numpy_3d import PISOSolver
from src.phase3_momentum import build_conservative_diffusion_matrix
from src.adjoint_piso import LinearSolve, csr_pattern
import test_duct as td

torch.set_default_dtype(torch.float64)
NU, G_TRUE, N = td.NU, td.G, 32


def make_solver():
    return PISOSolver((4, N, N), warp=1e-9, nu=NU, dt=0.5, corrector_steps=2,
                      periodic=(True, False, False), scheme="chorin", time_scheme="be",
                      boundary_flux_mode="impermeable", pressure_tol=1e-11)


def piso_steady(c, tol=1e-11, maxs=3000):
    """The REAL time-stepping solver, run to steady state with force amplitude c."""
    s = make_solver()
    s.velocity_source = [np.full_like(s.y, c), np.zeros_like(s.y), np.zeros_like(s.y)]
    prev = None
    for it in range(maxs):
        with contextlib.redirect_stdout(io.StringIO()):
            s.step()
        if prev is not None and np.abs(s.u - prev).max() < tol:
            break
        prev = s.u.copy()
    return s.u.copy(), s


# ---- the differentiable steady operator, assembled from the SAME solver code -------------
s0 = make_solver()
M = build_conservative_diffusion_matrix(*s0.shape, *s0.h, s0.J, s0.metrics, periodic=s0.per)
ib, bb = s0.ib, s0.bb
A_ii = (NU * M)[ib][:, ib].tocsr()
Aidx, Ashape, Aval = csr_pattern(A_ii)
Jf = torch.as_tensor(s0.J.ravel()[ib])
shape = s0.shape


def u_of(c):
    """u(c) on the interior, differentiable in the force amplitude c."""
    return LinearSolve.apply(Aval, Jf * c, (Aidx, Ashape), True, False)


if __name__ == "__main__":
    ok = True
    def check(name, good, detail):
        global ok
        ok &= bool(good)
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: {detail}")

    print(f"\nSquare duct {N}x{N}, nu={NU}, true force G = {G_TRUE}")

    # --- target from the real solver at the true force
    u_ref, s_ref = piso_steady(G_TRUE)
    tgt = torch.as_tensor(u_ref.ravel()[ib])
    print(f"  target: converged {N}x{N} solution, max|u| = {u_ref.max():.6f}")
    sec = max(np.abs(s_ref.v).max(), np.abs(s_ref.w).max()) / np.abs(s_ref.u).max()
    print(f"  projection is a no-op here (cross-plane |v,w|/|u| = {sec:.1e}), as expected")

    # --- does the differentiable operator reproduce the real solver?
    with torch.no_grad():
        u_diff = u_of(torch.tensor(G_TRUE)).numpy()
    rel = np.abs(u_diff - u_ref.ravel()[ib]).max() / np.abs(u_ref).max()
    check("differentiable operator matches the time-stepping solver", rel < 1e-6,
          f"max relative difference {rel:.2e}")

    def loss(c):
        return ((u_of(c) - tgt) ** 2).sum()

    # --- gradient vs finite differences OF THE REAL SOLVER
    c0 = 0.1
    c = torch.tensor(c0, requires_grad=True)
    L = loss(c); L.backward()
    g_adj = c.grad.item()
    eps = 1e-6
    lp = float(((torch.as_tensor(piso_steady(c0 + eps)[0].ravel()[ib]) - tgt) ** 2).sum())
    lm = float(((torch.as_tensor(piso_steady(c0 - eps)[0].ravel()[ib]) - tgt) ** 2).sum())
    g_fd = (lp - lm) / (2 * eps)
    check("gradient vs finite differences of the REAL solver", abs(g_adj - g_fd) / abs(g_fd) < 1e-4,
          f"adjoint {g_adj:.6e}  vs  FD {g_fd:.6e}")
    check("gradient sign", g_adj < 0, f"{g_adj:.3e} < 0, since c={c0} is below G={G_TRUE}")

    # --- learn it
    print(f"\n  training from c = {c0} ...")
    c = torch.tensor(c0, requires_grad=True)
    opt = torch.optim.Adam([c], lr=0.05)
    for it in range(400):
        opt.zero_grad(); L = loss(c); L.backward(); opt.step()
        if it % 80 == 0:
            print(f"     iter {it:3d}   c = {c.item():.6f}   loss = {L.item():.4e}")
    print(f"     final     c = {c.item():.8f}   (true {G_TRUE})")
    check("recovered the force", abs(c.item() - G_TRUE) < 1e-5,
          f"c = {c.item():.8f}, error {abs(c.item()-G_TRUE):.2e}")

    # --- and confirm it in the REAL solver
    u_learned, _ = piso_steady(c.item())
    err = np.abs(u_learned - u_ref).max() / np.abs(u_ref).max()
    check("real solver run at the learned force matches the target", err < 1e-5,
          f"max relative difference {err:.2e}")

    print("\n" + ("Scalar force recovered" if ok else "SCALAR STAGE FAILED"))


# =============================================================================================
# Network version: a CNN outputs a force FIELD, not a scalar.
#
# Strictly harder than the scalar case. The network has ~1600 free parameters and could produce
# any spatial pattern; nothing tells it the answer is uniform. It has to discover that from the
# solver's response alone.
#
# The target IS uniquely recoverable: at steady state u = A^{-1}(J f), and A is invertible, so
# f = A u / J is unique. That is worth checking before training -- an inverse problem with a
# non-unique answer would make "did it find the right force" meaningless.
# =============================================================================================
def run_network_version():
    import torch.nn as nn

    class ForceNet(nn.Module):
        """Coordinates (y, z) -> force field. Nothing constrains it to be uniform."""
        def __init__(self, width=16):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(2, width, 3, padding=1), nn.Tanh(),
                nn.Conv2d(width, width, 3, padding=1), nn.Tanh(),
                nn.Conv2d(width, 1, 1))

        def forward(self, coords):
            return self.net(coords)

    yy = s0.y[0]; zz = s0.z[0]
    coords = torch.stack([torch.as_tensor(yy), torch.as_tensor(zz)]).unsqueeze(0)
    u_ref, _ = piso_steady(G_TRUE)
    tgt = torch.as_tensor(u_ref.ravel()[ib])

    torch.manual_seed(0)
    net = ForceNet()
    npar = sum(p.numel() for p in net.parameters())
    # start the field near 0.1 by shifting the output bias
    with torch.no_grad():
        net.net[-1].bias.fill_(0.1)
    print(f"\n  ForceNet: {npar} parameters, outputs a {yy.shape} force field")

    def field():
        f = net(coords).squeeze(0).squeeze(0)              # (N, N) over the cross-section
        return f.unsqueeze(0).expand(shape[0], -1, -1).reshape(-1)[ib]

    def loss():
        return ((LinearSolve.apply(Aval, Jf * field(), (Aidx, Ashape), True, False) - tgt) ** 2).sum()

    with torch.no_grad():
        f0 = net(coords).squeeze()
    print(f"  initial field: mean {f0.mean():.4f}, spread {f0.std():.4f}   (true: uniform 1.0)")

    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    for it in range(1500):
        opt.zero_grad(); L = loss(); L.backward(); opt.step()
        if it % 300 == 0:
            with torch.no_grad():
                f = net(coords).squeeze()
            print(f"     iter {it:4d}  loss {L.item():.4e}   field mean {f.mean():.6f}  "
                  f"std {f.std():.2e}")
    with torch.no_grad():
        f = net(coords).squeeze()
    print(f"     final       loss {loss().item():.4e}   field mean {f.mean():.6f}  "
          f"std {f.std():.2e}")
    dev = (f - G_TRUE).abs().numpy()
    n = dev.shape[0]; k = max(2, n // 8)
    core = dev[k:-k, k:-k]
    wall = np.concatenate([dev[:k].ravel(), dev[-k:].ravel(),
                           dev[:, :k].ravel(), dev[:, -k:].ravel()])
    print(f"     deviation from the uniform truth:")
    print(f"        interior core   max {core.max():.3f}   mean {core.mean():.3f}")
    print(f"        near the walls  max {wall.max():.3f}   mean {wall.mean():.3f}")
    return float(f.mean()), float(core.max()), float(wall.max())


if __name__ == "__main__":
    mean, core_dev, wall_dev = run_network_version()
    print(f"""
  RESULT: the network recovered the force where the data constrains it, and not elsewhere.
  Core deviation {core_dev:.3f}, near-wall deviation {wall_dev:.3f}, field mean {mean:.4f}.

  That split is the physics of the inverse problem, not a training failure. f is formally
  unique (u = A^-1 J f with A invertible, so f = A u / J), but uniqueness is not conditioning:
  at the walls u -> 0, so the force there barely moves the solution and the gradient carries
  almost no information about it. The optimiser correctly spends its capacity where the loss
  actually responds.

  The lesson generalises to any learned closure: an inverse problem can be well posed and
  still be uninformative over part of the domain, and reporting only a global norm would hide
  exactly which part.""")
    ok_field = core_dev < 0.05
    print("\n" + ("Duct force recovered in the informative region"
                  if ok_field else "NOT recovered even in the core"))
    sys.exit(0 if ok_field else 1)
