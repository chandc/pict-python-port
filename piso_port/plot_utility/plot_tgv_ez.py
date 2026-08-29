"""Kinetic energy and enstrophy vs time for the 3D Taylor-Green vortex."""
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
S1, S2 = "#2a78d6", "#eb6834"

d = np.load("results/tgv3d.npz"); nu = float(d["nu"]); N = int(d["N"]); dt = float(d["dt"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK,
                     "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
                     "xtick.color": INK_2, "ytick.color": INK_2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 5.0))
cases = [("sou", "2nd-order upwind", S1, "-"), ("central", "central", S2, "--")]

for tag, lbl, col, ls in cases:
    t, E, Z = d[f"{tag}_t"], d[f"{tag}_E"], d[f"{tag}_Z"]
    a1.semilogy(t, E, ls, color=col, lw=2.0, label=lbl)
    a2.semilogy(t, Z, ls, color=col, lw=2.0, label=lbl)

t = d["sou_t"]; E = d["sou_E"]; Z = d["sou_Z"]
a1.plot(0, E[0], "o", color=INK, ms=5.5, zorder=5)
a1.annotate(f"E(0) = {E[0]:.4f}", xy=(0, E[0]), xytext=(10, -4),
            textcoords="offset points", color=INK_2, fontsize=9)
a2.plot(0, Z[0], "o", color=INK, ms=5.5, zorder=5)
a2.annotate(f"Z(0) = {Z[0]:.2f}  ← peak is at t = 0,\nso there is no cascade",
            xy=(0, Z[0]), xytext=(14, -18), textcoords="offset points",
            color=INK_2, fontsize=9)

a1.set_xlabel("t"); a1.set_ylabel("kinetic energy   E = ½⟨|u|²⟩")
a1.set_title("Kinetic energy", color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=8)
a2.set_xlabel("t"); a2.set_ylabel("enstrophy   Z = ½⟨|ω|²⟩")
a2.set_title("Enstrophy", color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=8)
for ax in (a1, a2):
    ax.grid(True, color=GRID, lw=0.8, which="both"); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    leg = ax.legend(frameon=False, fontsize=9, loc="lower left")
    for tx in leg.get_texts(): tx.set_color(INK_2)

fig.text(0.045, 0.965, "3D Taylor-Green vortex — energy and enstrophy decay",
         color=INK, fontsize=13.5, fontweight="bold", ha="left")
fig.text(0.045, 0.925,
         f"Grid: {N}×{N}×{N}, fully periodic  ·  ν = {nu} (Re = {int(1/nu)})  ·  "
         f"Δt = {dt}  ·  rotational projection + BDF2",
         color=INK_2, fontsize=9.5, ha="left")
fig.subplots_adjust(left=0.075, right=0.98, top=0.85, bottom=0.10, wspace=0.24)
fig.savefig("figures/tgv3d_E_Z.png", dpi=170, facecolor=SURFACE)
print("wrote tgv3d_E_Z.png")
