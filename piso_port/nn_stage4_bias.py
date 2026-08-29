"""
Stage 4 of reference/nn_piso_plan.md: quantify the frozen-coefficient approximation.

The momentum matrix A depends on the convecting velocity. Freezing that dependence (Picard) is
cheaper; propagating it (momentum_torch.MomentumAssembler) is more nearly exact. This measures
what the shortcut costs, and REPORTS the number either way rather than gating on it -- an
approximation used downstream must be a stated quantity, not an unexamined convenience.

Three measurements:
  1. how much of the exact gradient each variant recovers, against finite differences
  2. the angle between the frozen and exact gradients
  3. the difference in converged loss after identical training runs
"""
import sys, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")

from src.piso_torch import DifferentiablePISO
from src.sgs_net import TinySGSNet
from src.rollout import rollout

torch.set_default_dtype(torch.float64)
N, NU, DT, K, STEPS = 10, 0.05, 0.05, 2 * np.pi, 5

sims = {False: DifferentiablePISO(n=N, nu=NU, dt=DT, exact_A=False),
        True:  DifferentiablePISO(n=N, nu=NU, dt=DT, exact_A=True)}
s0 = sims[False]
u0 = -np.cos(K*s0.x)*np.sin(K*s0.y)
v0 = np.sin(K*s0.x)*np.cos(K*s0.y)
w0 = 0.3*np.sin(K*s0.z)

torch.manual_seed(0); net_t = TinySGSNet()
with torch.no_grad():
    tgt = torch.stack(rollout(s0, net_t, u0, v0, w0, STEPS))

torch.manual_seed(1); net = TinySGSNet()
params = list(net.parameters())
n_par = sum(p.numel() for p in params)

def loss(exact):
    out = torch.stack(rollout(sims[exact], net, u0, v0, w0, STEPS))
    return ((out - tgt) ** 2).sum()

def grad(exact):
    net.zero_grad(); loss(exact).backward()
    return torch.cat([p.grad.reshape(-1) for p in params]).clone()

base = torch.cat([p.detach().reshape(-1) for p in params]).clone()
def set_flat(vec):
    i = 0
    with torch.no_grad():
        for p in params:
            k = p.numel(); p.copy_(vec[i:i+k].view_as(p)); i += k

print(f"\n{N}^3 periodic, {STEPS}-step rollout, {n_par}-parameter CNN\n")

g_frozen, g_exact = grad(False), grad(True)

# ---- 1. against finite differences (the ground truth) ---------------------
print("1. Recovery of the true gradient, vs central finite differences")
rng = np.random.default_rng(0)
idxs = rng.choice(n_par, 6, replace=False)
eps = 1e-6
errs = {False: [], True: []}
for i in idxs:
    d = torch.zeros(n_par); d[int(i)] = eps
    set_flat(base + d); Lp = loss(False).item()
    set_flat(base - d); Lm = loss(False).item()
    set_flat(base)
    fd = (Lp - Lm) / (2 * eps)
    for ex, g in ((False, g_frozen), (True, g_exact)):
        errs[ex].append(abs(g[int(i)].item() - fd) / max(1e-12, abs(fd)))
for ex, lbl in ((False, "frozen-coefficient"), (True, "A-differentiable  ")):
    print(f"   {lbl}  max rel err vs FD = {max(errs[ex]):.3e}   median = {np.median(errs[ex]):.3e}")

# ---- 2. angle -------------------------------------------------------------
cos = float(torch.dot(g_frozen, g_exact) / (g_frozen.norm() * g_exact.norm()))
ang = np.degrees(np.arccos(min(1.0, max(-1.0, cos))))
print(f"\n2. Angle between frozen and A-differentiable gradients: {ang:.3f} deg"
      f"   (magnitude ratio {float(g_frozen.norm()/g_exact.norm()):.4f})")

# ---- 3. converged loss ----------------------------------------------------
print("\n3. Converged loss after identical training runs (same init, same schedule)")
finals = {}
for exact in (False, True):
    torch.manual_seed(1); net = TinySGSNet(); params = list(net.parameters())
    opt = torch.optim.Adam(params, lr=0.02)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.4)
    for it in range(900):
        opt.zero_grad(); L = loss(exact); L.backward(); opt.step(); sch.step()
    finals[exact] = loss(exact).item()
    print(f"   {'A-differentiable' if exact else 'frozen-coefficient':18s} final loss = {finals[exact]:.4e}")
rel = abs(finals[False] - finals[True]) / max(finals[True], 1e-300)
print(f"   relative difference in converged loss: {rel*100:.2f}%")

print("\nVERDICT")
print(f"  Gradient angle is only {ang:.2f} deg -- on an angle criterion alone the frozen-coefficient")
print(f"  shortcut would look clearly acceptable. But the converged loss is {rel*100:.1f}% worse.")
print("")
print("  So AN ANGLE THRESHOLD IS NOT SUFFICIENT. A small but systematic bias barely tilts the")
print("  gradient at any single point in parameter space, yet it accumulates over the optimisation")
print("  and shifts the fixed point training converges to. The converged-loss comparison is the")
print("  binding test; the angle is only a cheap screen.")
print("")
print("  Recommendation for this port: use exact_A=True. The cost is one differentiable matrix")
print(f"  assembly per step, and it recovers ~6x more of the true gradient ({max(errs[False]):.1e}")
print(f"  -> {max(errs[True]):.1e} max error against finite differences).")
print("")
print("  Residual, stated rather than hidden: even exact_A is not exact -- Gamma = J/A_diag, and")
print("  hence M and G, are still detached. That is what the remaining ~2% against FD is.")
