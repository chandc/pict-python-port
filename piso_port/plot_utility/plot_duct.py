"""Square-duct cross-section: axial velocity, error against the series, and the secondary flow."""
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
import test_duct as td

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e2e0da"
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("blue_seq", BLUE)
# diverging: blue <-> gray <-> red, neutral gray midpoint (never a hue at the midpoint)
DIV = LinearSegmentedColormap.from_list(
    "blue_red", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a99a", "#e34948", "#8f1f1e"])

d = np.load("results/duct_field.npz")
j = 0                                   # any streamwise slice: the flow is x-invariant
Y, Z = d["y"][j], d["z"][j]
U = d["u"][j]
V, W = d["v"][j], d["w"][j]
EX = td.duct_exact(Y, Z, G=float(d["G"]), mu=float(d["nu"]), n_terms=400)
ERR = U - EX

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK,
                     "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
                     "xtick.color": INK_2, "ytick.color": INK_2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14.4, 5.3))

m = a1.contourf(Y, Z, U, levels=14, cmap=SEQ)
a1.contour(Y, Z, U, levels=14, colors=SURFACE, linewidths=0.5, alpha=0.75)
cb = fig.colorbar(m, ax=a1, fraction=0.046, pad=0.03)
cb.set_label("u", color=INK_2, fontsize=9)
cb.outline.set_edgecolor(GRID); cb.ax.tick_params(colors=INK_2, labelsize=8.5)
a1.set_title("Axial velocity", color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=8)

lim = float(np.abs(ERR).max())
m2 = a2.contourf(Y, Z, ERR, levels=np.linspace(-lim, lim, 15), cmap=DIV,
                 norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim))
cb2 = fig.colorbar(m2, ax=a2, fraction=0.046, pad=0.03)
cb2.set_label("error", color=INK_2, fontsize=9)
cb2.outline.set_edgecolor(GRID); cb2.ax.tick_params(colors=INK_2, labelsize=8.5)
a2.set_title("Error vs the Fourier series", color=INK, fontsize=11.5,
             fontweight="bold", loc="left", pad=8)

# centreline and diagonal profiles against the analytic solution
n = U.shape[0]; mid = n // 2
a3.plot(Y[:, mid], U[:, mid], "-", color="#2a78d6", lw=2.0, label="computed, centreline")
a3.plot(Y[::2, mid], EX[::2, mid], "o", ms=5.5, mfc="none", mew=1.5, color=INK,
        label="series, centreline")
diag = np.arange(n)
a3.plot(Y[diag, diag], U[diag, diag], "--", color="#eb6834", lw=2.0, label="computed, diagonal")
a3.plot(Y[diag, diag][::2], EX[diag, diag][::2], "s", ms=5, mfc="none", mew=1.5,
        color=INK_2, label="series, diagonal")
a3.set_xlabel("y"); a3.set_ylabel("u")
a3.set_title("Profiles vs the analytic series", color=INK, fontsize=11.5,
             fontweight="bold", loc="left", pad=8)
a3.grid(True, color=GRID, lw=0.8); a3.set_axisbelow(True)
leg = a3.legend(frameon=False, fontsize=8.5, loc="lower center")
for t in leg.get_texts(): t.set_color(INK_2)

for ax in (a1, a2):
    ax.set_xlabel("y"); ax.set_ylabel("z"); ax.set_aspect("equal")
for ax in (a1, a2, a3):
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)

sec = max(float(np.abs(V).max()), float(np.abs(W).max())) / float(np.abs(U).max())
fig.text(0.035, 0.975, "Square duct, cross-section", color=INK, fontsize=13.5,
         fontweight="bold", ha="left")
fig.text(0.035, 0.912,
         f"{U.shape[0]}×{U.shape[1]} cross-section, ν = {float(d['nu'])}, body force G = {float(d['G'])}.  "
         f"Cross-plane velocity |v,w|/|u| = {sec:.1e} — laminar duct flow has NO secondary flow, so there\n"
         f"are no cross-sectional streamlines to draw; secondary motion in a duct is a turbulent effect.",
         color=INK_2, fontsize=9.5, ha="left")
fig.subplots_adjust(left=0.05, right=0.985, top=0.775, bottom=0.10, wspace=0.42)
fig.savefig("figures/duct_cross_section.png", dpi=170, facecolor=SURFACE)
print(f"wrote duct_cross_section.png   (secondary flow ratio {sec:.2e})")
