"""TGV grid-resolution study: curve collapse and the convergence of numerical dissipation."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)          # figures/ and results/ paths are relative to the root

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e2e0da"
ORD = {32: "#86b6ef", 48: "#2a78d6", 64: "#104281"}     # resolution is ORDERED -> ordinal ramp

d = np.load("results/tgv_resolution.npz"); nu = float(d["nu"]); NS = [32, 48, 64]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK,
                     "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
                     "xtick.color": INK_2, "ytick.color": INK_2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.9))

diss = []
for N in NS:
    t, E, Z = d[f"n{N}_t"], d[f"n{N}_E"], d[f"n{N}_Z"]
    m = (t > 0.05) & (t < t[-1] - 1e-9)
    a1.plot(t[m], 100 * np.abs(-np.gradient(E, t) - 2*nu*Z)[m] / (2*nu*Z)[m],
            color=ORD[N], lw=2.0, label=f"{N}³")
    mm = (t > 0.1) & (t < 1.5)
    diss.append((np.abs(-np.gradient(E, t) - 2*nu*Z)[mm] / (2*nu*Z)[mm]).mean())

a1.set_xlabel("t"); a1.set_ylabel("numerical / physical dissipation  [%]")
a1.set_title("Numerical dissipation falls with refinement", color=INK, fontsize=11.5,
             fontweight="bold", loc="left", pad=8)

h = np.array([1.0 / N for N in NS])
a2.loglog(NS, np.array(diss) * 100, "o-", color=ORD[48], lw=2.0, ms=8, label="measured")
ref = diss[0] * 100 * (h / h[0])**2
a2.loglog(NS, ref, "--", color=INK_2, lw=1.6, label="2nd-order reference")
for i, (N, v) in enumerate(zip(NS, diss)):
    dx = -8 if i == len(NS) - 1 else 8
    a2.annotate(f"{v*100:.2f}%", xy=(N, v*100), xytext=(dx, 8),
                textcoords="offset points", color=INK_2, fontsize=9,
                ha="right" if dx < 0 else "left")
a2.set_xlabel("grid points per direction  N"); a2.set_ylabel("mean numerical dissipation  [%]")
from matplotlib.ticker import NullFormatter, NullLocator
a2.xaxis.set_minor_locator(NullLocator()); a2.xaxis.set_minor_formatter(NullFormatter())
a2.set_xticks(NS); a2.set_xticklabels([str(n) for n in NS])
a2.set_xlim(NS[0]*0.92, NS[-1]*1.10)
a2.set_title("Converges at 2nd order (rates 2.01, 2.00)", color=INK, fontsize=11.5,
             fontweight="bold", loc="left", pad=8)

for ax in (a1, a2):
    ax.grid(True, color=GRID, lw=0.8, which="major"); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    leg = ax.legend(frameon=False, fontsize=9, loc="best")
    for tx in leg.get_texts(): tx.set_color(INK_2)

fig.text(0.045, 0.965, "3D Taylor-Green — grid-resolution study", color=INK,
         fontsize=13.5, fontweight="bold", ha="left")
fig.text(0.045, 0.923,
         f"central convection, ν = {nu}, fully periodic.  Against 64³, the 48³ energy curve "
         f"differs by ≤ 0.07% and enstrophy by ≤ 0.26%.",
         color=INK_2, fontsize=9.5, ha="left")
fig.subplots_adjust(left=0.075, right=0.98, top=0.845, bottom=0.11, wspace=0.25)
fig.savefig("figures/tgv3d_resolution.png", dpi=170, facecolor=SURFACE)
print("wrote tgv3d_resolution.png")
