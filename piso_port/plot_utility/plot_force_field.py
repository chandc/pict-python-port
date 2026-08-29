"""Learned force field on the duct cross-section, and why its error sits where it does."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)          # figures/ and results/ paths are relative to the root

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e2e0da"
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("blue_seq", BLUE)
DIV = LinearSegmentedColormap.from_list(
    "blue_red", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a99a", "#e34948", "#8f1f1e"])

d = np.load("results/duct_force_learned.npz")
Y, Z, F, G = d["y"], d["z"], d["f"], float(d["G"])
U = d["u_ref"]
DEV = F - G

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK,
                     "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
                     "xtick.color": INK_2, "ytick.color": INK_2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14.4, 5.3))

m1 = a1.contourf(Y, Z, F, levels=14, cmap=SEQ)
a1.contour(Y, Z, F, levels=14, colors=SURFACE, linewidths=0.5, alpha=0.7)
cb = fig.colorbar(m1, ax=a1, fraction=0.046, pad=0.03)
cb.set_label("learned force", color=INK_2, fontsize=9)
cb.outline.set_edgecolor(GRID); cb.ax.tick_params(colors=INK_2, labelsize=8.5)
a1.set_title("Learned force field", color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=8)

lim = float(np.abs(DEV).max())
m2 = a2.contourf(Y, Z, DEV, levels=np.linspace(-lim, lim, 15), cmap=DIV,
                 norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim))
cb2 = fig.colorbar(m2, ax=a2, fraction=0.046, pad=0.03)
cb2.set_label("error", color=INK_2, fontsize=9)
cb2.outline.set_edgecolor(GRID); cb2.ax.tick_params(colors=INK_2, labelsize=8.5)
a2.set_title(f"Error vs the true uniform G = {G}", color=INK, fontsize=11.5,
             fontweight="bold", loc="left", pad=8)

# the payoff panel: the error lives exactly where the velocity -- and so the sensitivity -- is small
a3.plot(U.ravel(), np.abs(DEV).ravel(), "o", ms=3, color="#2a78d6", alpha=0.35,
        markeredgewidth=0)
a3.set_xlabel("local axial velocity  u"); a3.set_ylabel("|force error|")
a3.set_title("Error concentrates where u is small", color=INK, fontsize=11.5,
             fontweight="bold", loc="left", pad=8)
a3.grid(True, color=GRID, lw=0.8); a3.set_axisbelow(True)
a3.annotate("walls: u → 0, so the force there\nbarely moves the solution and the\ngradient cannot see it",
            xy=(0.02, np.abs(DEV).max()*0.92), xytext=(0.18, np.abs(DEV).max()*0.86),
            color=INK_2, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=INK_2, lw=1.1))

for ax in (a1, a2):
    ax.set_xlabel("y"); ax.set_ylabel("z"); ax.set_aspect("equal")
for ax in (a1, a2, a3):
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)

n = F.shape[0]; k = max(2, n//8)
core = np.abs(DEV)[k:-k, k:-k].max()
wall = max(np.abs(DEV)[:k].max(), np.abs(DEV)[-k:].max(),
           np.abs(DEV)[:, :k].max(), np.abs(DEV)[:, -k:].max())
fig.text(0.035, 0.975, "Learned force field — CNN formulation", color=INK,
         fontsize=13.5, fontweight="bold", ha="left")
fig.text(0.035, 0.912,
         f"~1600-parameter CNN, coordinates → force, trained through the solver. Truth is uniform "
         f"G = {G}; nothing tells the network that.\nRecovered to {core:.3f} in the interior core "
         f"but only {wall:.2f} near the walls — and the resulting velocity still matches the target "
         f"to 2.2e-03.",
         color=INK_2, fontsize=9.5, ha="left")
fig.subplots_adjust(left=0.05, right=0.985, top=0.775, bottom=0.10, wspace=0.42)
fig.savefig("figures/duct_force_field.png", dpi=170, facecolor=SURFACE)
print(f"wrote duct_force_field.png   core {core:.3f}  wall {wall:.3f}")
