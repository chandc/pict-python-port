"""
The far-field cylinder O-grid, coloured by block.

Three views because no single one is honest about this mesh: the full domain is 80 diameters
across and the cylinder is a dot in it, while the near field is where every cell that matters
lives. The radial distribution is plotted on its own so the 450x growth from wall cell to outer
cell is visible as a number rather than implied by a picture.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from cylinder_grid import cylinder_domain, outer_role, D, R_CYL

NBLK = 8
COL = ["#e4572e", "#17bebb", "#ffc914", "#5b6c8f", "#76b041",
       "#a26769", "#8e6bbf", "#d1495b"]

d, r, ratio = cylinder_domain(nblk=NBLK)
roles = outer_role(d, NBLK)


def draw(ax, rstep, tstep, lw=0.3, rmax=None):
    for b, blk in enumerate(d.blocks):
        X, Y = blk.x[:, :, 0], blk.y[:, :, 0]
        rr = np.hypot(X, Y)
        keep = slice(None) if rmax is None else slice(0, int((rr[:, 0] <= rmax).sum()) + 1)
        Xc, Yc = X[keep], Y[keep]
        c = COL[b % len(COL)]
        for i in range(0, Xc.shape[0], rstep):
            ax.plot(Xc[i, :], Yc[i, :], color=c, lw=lw, alpha=.9)
        for j in range(0, Xc.shape[1], tstep):
            ax.plot(Xc[:, j], Yc[:, j], color=c, lw=lw, alpha=.9)
    ax.add_patch(plt.Circle((0, 0), R_CYL, color="k", zorder=9))


fig = plt.figure(figsize=(16.5, 8.6))

ax = fig.add_axes([0.035, 0.10, 0.30, 0.78])
draw(ax, rstep=4, tstep=4)
ax.set_xlim(-42, 42); ax.set_ylim(-42, 42); ax.set_aspect("equal")
ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
ax.set_title(f"full domain — far field at {r[-1]/D:.0f} D\n(every 4th mesh line)", fontsize=11)

ax = fig.add_axes([0.365, 0.10, 0.30, 0.78])
draw(ax, rstep=1, tstep=1, lw=0.35, rmax=3.0)
ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
ax.set_title("near field — every mesh line\n(wall cell 0.006 D)", fontsize=11)

ax = fig.add_axes([0.72, 0.56, 0.26, 0.32])
ax.semilogy(np.arange(len(r) - 1), np.diff(r) / D, "o-", ms=2.5, lw=1.1, color="#5b6c8f")
ax.set_xlabel("radial index"); ax.set_ylabel("cell size / D")
ax.grid(alpha=.3, which="both")
ax.set_title(f"radial spacing — growth ratio {ratio:.4f}", fontsize=10)

ax = fig.add_axes([0.72, 0.10, 0.26, 0.32])
th = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(th), np.sin(th), color="0.75", lw=1)
for b in range(NBLK):
    a0, a1 = 2 * np.pi * b / NBLK, 2 * np.pi * (b + 1) / NBLK
    t = np.linspace(a0, a1, 40)
    ls = "-" if roles[b] == "outflow" else "--"
    ax.plot(np.cos(t), np.sin(t), color=COL[b % len(COL)], lw=5, solid_capstyle="butt",
            ls=ls)
    tm = 0.5 * (a0 + a1)
    ax.text(1.28 * np.cos(tm), 1.28 * np.sin(tm), str(b), ha="center", va="center",
            fontsize=9, color=COL[b % len(COL)], fontweight="bold")
ax.add_patch(plt.Circle((0, 0), 0.30, color="k"))
ax.annotate("", xy=(-1.55, 0), xytext=(-2.15, 0),
            arrowprops=dict(arrowstyle="-|>", lw=2, color="k"))
ax.text(-1.85, 0.22, "flow", ha="center", fontsize=9)
ax.set_xlim(-2.4, 1.7); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("far-field face by block\nsolid = outflow, dashed = free stream", fontsize=10)

fig.legend(handles=[Line2D([], [], color=COL[b], lw=3,
                           label=f"{b} {roles[b][:3]}") for b in range(NBLK)],
           fontsize=9, ncol=8, loc="lower center", bbox_to_anchor=(0.5, 0.005),
           frameon=False)
fig.suptitle(f"Cylinder vortex street — O-grid, {NBLK} blocks, {d.n_cells:,} cells, "
             f"far field {r[-1]/D:.0f} D, span {4.0:.0f} D periodic", fontsize=13)
fig.savefig("figures/cylinder_farfield_grid.png", dpi=145, bbox_inches="tight")
print(f"  {len(d.blocks)} blocks, {d.n_cells:,} cells, validate() {len(d.validate())} problems")
print(f"  min(J) {min(d.block_metrics_cached(b)[0].min() for b in range(NBLK)):.3e}")
print(f"  wall cell {r[1]-r[0]:.4f} D, outer cell {r[-1]-r[-2]:.2f} D, ratio {ratio:.4f}")
print("wrote figures/cylinder_farfield_grid.png")
