"""Centreline profiles vs Ghia, Ghia & Shin (1982) Re=100 benchmark."""
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
# Resolution is an ORDERED quantity, so the three Cartesian grids use an ordinal ramp
# (blue, light->dark, no step lighter than 250 on a light surface). The warped grid is a
# different KIND of case, not a finer one, so it takes categorical slot 2 (orange).
ORD = {24: "#86b6ef", 32: "#2a78d6", 40: "#104281"}
WARPED = "#eb6834"

# Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982) 387-411, Tables I & II, Re = 100.
# 2D benchmark: u along the vertical centreline, v along the horizontal centreline.
GHIA_Y = [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
          0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000]
GHIA_U = [0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090,
          -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.00000]
GHIA_X = [0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
          0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000]
GHIA_V = [0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527,
          0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.00000]

def unit(a):
    """Normalise a physical coordinate line to [0,1] (the warped domain is shifted)."""
    return (a - a.min()) / (a.max() - a.min())

runs = []
d = np.load("results/cavity_results.npz")
N = int(d["N"]); j = N // 2
runs.append(("warped 24³ (warp 0.05)", WARPED, "-",
             unit(d["z"][N//2, j, :]), d["Re100_u"][N//2, j, :],
             unit(d["x"][:, j, N//2]), d["Re100_w"][:, j, N//2]))
try:
    c = dict(np.load("results/cavity_cartesian.npz"))
    try:
        c.update(dict(np.load("results/cavity_cart40.npz")))
    except FileNotFoundError:
        pass
    avail = [t for t in ("cart24", "cart32", "cart40") if f"{t}_N" in c]
    for tag, lbl, col in [(t, f"Cartesian {int(c[f'{t}_N'])}³", ORD[int(c[f"{t}_N"])])
                          for t in avail]:
        n = int(c[f"{tag}_N"]); jj = n // 2
        runs.append((lbl, col, "-",
                     unit(c[f"{tag}_z"][n//2, jj, :]), c[f"{tag}_u"][n//2, jj, :],
                     unit(c[f"{tag}_x"][:, jj, n//2]), c[f"{tag}_w"][:, jj, n//2]))
except FileNotFoundError:
    pass

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                     "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
                     "xtick.color": INK_2, "ytick.color": INK_2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))

for ax, which in zip(axes, ("u", "w")):
    ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
    if which == "u":
        ax.plot(GHIA_U, GHIA_Y, "o", ms=6.5, mfc="none", mew=1.6, color=INK,
                label="Ghia et al. 1982 (2D)", zorder=5)
        for lbl, col, ls, zc, uu, _, _ in runs:
            ax.plot(uu, zc, ls, color=col, lw=2.0, label=lbl, zorder=3)
        ax.set_xlabel("u"); ax.set_ylabel("z  (normalised)")
        ax.set_title("u along the vertical centreline", color=INK, fontsize=11,
                     fontweight="bold", loc="left", pad=8)
    else:
        ax.plot(GHIA_X, GHIA_V, "o", ms=6.5, mfc="none", mew=1.6, color=INK,
                label="Ghia et al. 1982 (2D)", zorder=5)
        for lbl, col, ls, _, _, xc, ww in runs:
            ax.plot(xc, ww, ls, color=col, lw=2.0, label=lbl, zorder=3)
        ax.set_xlabel("x  (normalised)"); ax.set_ylabel("w")
        ax.set_title("w along the horizontal centreline", color=INK, fontsize=11,
                     fontweight="bold", loc="left", pad=8)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="best")
    for t in leg.get_texts(): t.set_color(INK_2)

fig.text(0.045, 0.965, "Centreline profiles vs the Ghia et al. Re = 100 benchmark",
         color=INK, fontsize=13.5, fontweight="bold", ha="left")
fig.text(0.045, 0.925,
         "Ghia is a 2D cavity; these runs are 3D with no-slip end walls, sampled on the "
         "mid-plane — they are not expected to coincide.",
         color=INK_2, fontsize=9.5, ha="left")
fig.subplots_adjust(left=0.075, right=0.975, top=0.85, bottom=0.10, wspace=0.24)
fig.savefig("figures/cavity_vs_ghia.png", dpi=170, facecolor=SURFACE)

# quantitative summary
print(f"{'case':24s} {'u_min':>9s} {'z@u_min':>9s} {'w_max':>9s} {'w_min':>9s}")
print(f"{'Ghia 1982 (2D)':24s} {min(GHIA_U):9.4f} {GHIA_Y[int(np.argmin(GHIA_U))]:9.4f} "
      f"{max(GHIA_V):9.4f} {min(GHIA_V):9.4f}")
for lbl, col, ls, zc, uu, xc, ww in runs:
    print(f"{lbl:24s} {uu.min():9.4f} {zc[int(np.argmin(uu))]:9.4f} "
          f"{ww.max():9.4f} {ww.min():9.4f}")
print("\nwrote cavity_vs_ghia.png")
