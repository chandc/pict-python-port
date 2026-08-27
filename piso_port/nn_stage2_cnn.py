"""
Stage 2 of reference/nn_piso_plan.md: a tiny CNN predicting a momentum source, one PISO step.

Gates:
  (a) finite differences on EVERY weight, max relative error < 1e-5
  (b) recovery -- train from a different initialisation and reproduce the target
  (c) shift-equivariance -- translating the input by one cell translates the output by one cell
"""
import sys, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")

from piso_torch import DifferentiablePISO
from sgs_net import TinySGSNet

torch.set_default_dtype(torch.float64)
ok = True
def check(name, good, detail):
    global ok
    ok &= bool(good)
    print(f"  [{'PASS' if good else 'FAIL'}] {name}: {detail}")

N, NU, DT = 16, 0.05, 0.05
K = 2 * np.pi
sim = DifferentiablePISO(n=N, nu=NU, dt=DT)
u0 = -np.cos(K*sim.x)*np.sin(K*sim.y)
v0 = np.sin(K*sim.x)*np.cos(K*sim.y)
w0 = np.zeros_like(u0)
sim.build(u0, v0, w0)

# The target is the same architecture with DIFFERENT random weights, so it is non-trivial
# but representable. Note: an earlier version scaled these weights by 2 "to make the target
# non-trivial", which pushed tanh into saturation -- a known-hard optimisation regime that has
# nothing to do with the solver coupling under test. It capped the fit at 2.9e-03 while the
# unsaturated target reaches 8.9e-04. The lesson is about test design, not about the gradient:
# gate (a) was passing at 5e-08 throughout.
torch.manual_seed(0)
net_true = TinySGSNet()
with torch.no_grad():
    S_true = net_true.field(u0, v0, w0, sim.shape)
    tgt = torch.stack(sim.step(u0, v0, w0, S_true))

torch.manual_seed(1)
net = TinySGSNet()
n_par = sum(p.numel() for p in net.parameters())
print(f"\nTinySGSNet: {n_par} parameters, {N}^3 Cartesian periodic grid")

def loss_fn():
    S = net.field(u0, v0, w0, sim.shape)
    return ((torch.stack(sim.step(u0, v0, w0, S)) - tgt) ** 2).sum()

# ---------------------------------------------------------------- gate (a)
print("\nGate (a): finite differences on EVERY weight")
net.zero_grad(); loss_fn().backward()
flat = torch.cat([p.grad.reshape(-1) for p in net.parameters()]).clone()
params = list(net.parameters())

def set_flat(vec):
    i = 0
    with torch.no_grad():
        for p in params:
            k = p.numel(); p.copy_(vec[i:i+k].view_as(p)); i += k

base = torch.cat([p.detach().reshape(-1) for p in params]).clone()
eps, worst, worst_i = 1e-6, 0.0, -1
for i in range(n_par):
    d = torch.zeros(n_par); d[i] = eps
    set_flat(base + d); Lp = loss_fn().item()
    set_flat(base - d); Lm = loss_fn().item()
    fd = (Lp - Lm) / (2 * eps)
    rel = abs(fd - flat[i].item()) / max(1e-8, abs(fd))
    if rel > worst: worst, worst_i = rel, i
set_flat(base)
check(f"all {n_par} weights", worst < 1e-5,
      f"max relative error {worst:.3e} (weight #{worst_i})")

# ---------------------------------------------------------------- gate (c)
print("\nGate (c): shift-equivariance (circular padding)")
x = torch.stack([torch.as_tensor(f) for f in (u0, v0, w0)]).unsqueeze(0)
with torch.no_grad():
    a = net(x)
    b = net(torch.roll(x, 1, dims=2))
    err = (torch.roll(a, 1, dims=2) - b).abs().max().item()
check("shift input by one cell", err < 1e-10, f"max deviation {err:.2e}")

# ---------------------------------------------------------------- gate (b)
print("\nGate (b): recovery by training")
opt = torch.optim.Adam(net.parameters(), lr=0.02)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=800, gamma=0.3)
L0 = loss_fn().item()
for it in range(2400):
    opt.zero_grad(); L = loss_fn(); L.backward(); opt.step(); sched.step()
    if it % 600 == 0:
        print(f"   iter {it:4d}  loss = {L.item():.4e}")
with torch.no_grad():
    S = net.field(u0, v0, w0, sim.shape)
    out = torch.stack(sim.step(u0, v0, w0, S))
    rel_u = ((out - tgt).norm() / tgt.norm()).item()
    rel_S = ((S - S_true).norm() / S_true.norm()).item()
print(f"   final loss = {loss_fn().item():.4e}  (started {L0:.4e})")
check("velocity reproduced", rel_u < 1e-3, f"||u - u_target|| / ||u_target|| = {rel_u:.3e}")
print(f"   for reference, source-field error ||S - S*|| / ||S*|| = {rel_S:.3e}")
print("   NOTE: only the SOLENOIDAL part of S is identifiable from velocity data -- the")
print("   projection removes any gradient component, so S itself is not uniquely recoverable.")
print("   The velocity match is therefore the meaningful gate, not the source match.")

print("\n" + ("Stage 2 gates passed" if ok else "STAGE 2 FAILED"))
sys.exit(0 if ok else 1)
