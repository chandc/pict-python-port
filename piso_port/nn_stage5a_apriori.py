"""
Stage 5a of reference/nn_piso_plan.md: a-priori SGS regression, no solver in the loop.

Train the network to predict -div(tau) directly from the filtered velocity. This isolates
NETWORK CAPACITY from SOLVER COUPLING: if a closure cannot be learned here, no amount of
differentiable-solver machinery in 5b will rescue it.

Gate: correlation with the true SGS term > 0.8 on HELD-OUT snapshots.
"""
import sys, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")
from sgs_net import SGSNet

torch.set_default_dtype(torch.float32)

d = np.load("sgs_data.npz")
X = torch.as_tensor(d["inputs"], dtype=torch.float32)      # (S, 3, n, n, n)
Y = torch.as_tensor(d["targets"], dtype=torch.float32)
S = X.shape[0]
# Split RANDOMLY, not temporally. The flow decays, so the SGS magnitude falls ~35x across the
# run; a first-70%/last-30% split would train on strong signal and test on near-zero, which is
# a distribution shift rather than a held-out test.
perm = np.random.default_rng(0).permutation(S)
n_tr = int(0.7 * S)
tr, te = perm[:n_tr], perm[n_tr:]
Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]

# normalise by the TRAINING statistics only -- using test statistics would leak
xs, ys = Xtr.std(), Ytr.std()
Xtr, Xte = Xtr / xs, Xte / xs
Ytr, Yte = Ytr / ys, Yte / ys

print(f"\n{S} snapshots of {tuple(X.shape[2:])}, {n_tr} train / {S-n_tr} held out (random split)")
mags = [float(Y[i].std()) for i in range(S)]
print(f"target magnitude varies {min(mags):.2e} -> {max(mags):.2e} across the run (decaying flow)")
print(f"input std {xs:.4f}   target std {ys:.4f}")

torch.manual_seed(0)
net = SGSNet()
n_par = sum(p.numel() for p in net.parameters())
print(f"SGSNet: {n_par} parameters")

def corr(a, b):
    """Correlation averaged PER SNAPSHOT.

    Pooling everything into one correlation would let the few high-amplitude early snapshots
    dominate and hide poor prediction on the decayed ones -- the SGS magnitude falls ~35x
    over the run.
    """
    cs = []
    for i in range(a.shape[0]):
        p = a[i].reshape(-1) - a[i].mean()
        q = b[i].reshape(-1) - b[i].mean()
        cs.append(float((p * q).sum() / (p.norm() * q.norm() + 1e-30)))
    return float(np.mean(cs))

opt = torch.optim.Adam(net.parameters(), lr=2e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=600)
for it in range(600):
    opt.zero_grad()
    loss = ((net(Xtr) - Ytr) ** 2).mean()
    loss.backward(); opt.step(); sch.step()
    if it % 150 == 0:
        with torch.no_grad():
            print(f"   iter {it:3d}  train mse {loss.item():.4f}   "
                  f"held-out corr {corr(net(Xte), Yte):.4f}")

with torch.no_grad():
    c_te = corr(net(Xte), Yte)
    c_tr = corr(net(Xtr), Ytr)
    # a baseline any closure must beat: the Smagorinsky-like assumption that the SGS force is
    # simply proportional to the resolved field
    c_triv = corr(Xte, Yte)

print(f"\n   training correlation  {c_tr:.4f}")
print(f"   HELD-OUT correlation  {c_te:.4f}   (gate: > 0.80)")
print(f"   trivial baseline (SGS proportional to resolved u): {c_triv:.4f}")
ok = c_te > 0.80
print(f"\n  [{'PASS' if ok else 'FAIL'}] Stage 5a: held-out correlation {c_te:.4f}")
if c_tr - c_te > 0.15:
    print(f"   WARNING: train-test gap {c_tr - c_te:.3f} suggests memorisation on {n_tr} snapshots")
print("\n   CAVEAT: the snapshots come from a single decaying run and are temporally correlated,")
print("   so the held-out set is not fully independent. This gate tests network CAPACITY --")
print("   whether the mapping is learnable at all -- not generalisation across flows.")
torch.save({"state": net.state_dict(), "x_std": float(xs), "y_std": float(ys)},
           "sgs_apriori.pt")   # normalisation travels with the weights
sys.exit(0 if ok else 1)
