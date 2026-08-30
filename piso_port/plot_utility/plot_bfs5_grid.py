"""
The five-domain Armaly BFS grid, one colour per domain.

Shown before any computation, because a multi-block grid is where the silent errors live: a
wrong period shift, an inverted clustering, a partial-face connection. Two of those had already
been caught on this geometry by looking at numbers; this is the picture that goes with them.

Mesh lines are decimated for legibility -- every Nth line is drawn, so the apparent cell count
is lower than the real one. The zoom panel draws every line.
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

from armaly_bfs5_grid import bfs5_domain, NAMES, S, H_IN, L_IN, X1, L_OUT, L_SPAN

COL = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]


def draw(ax, d, every=1, lw=0.35):
    for b in range(len(d.blocks)):
        blk = d.blocks[b]
        X, Y = blk.x[:, :, 0], blk.y[:, :, 0]
        c = COL[b]
        for i in range(0, X.shape[0], every):
            ax.plot(X[i, :], Y[i, :], color=c, lw=lw, alpha=.85)
        for j in range(0, X.shape[1], every):
            ax.plot(X[:, j], Y[:, j], color=c, lw=lw, alpha=.85)
        # outline
        ax.plot(np.r_[X[0, :], X[:, -1], X[-1, ::-1], X[::-1, 0]],
                np.r_[Y[0, :], Y[:, -1], Y[-1, ::-1], Y[::-1, 0]],
                color=c, lw=1.4)


d = bfs5_domain()
fig = plt.figure(figsize=(16, 8.4))

ax = fig.add_axes([0.05, 0.70, 0.92, 0.18])
draw(ax, d, every=2)
ax.plot([0, 0], [-S, 0], color="k", lw=4, zorder=9, solid_capstyle="butt")
ax.set_xlim(-L_IN, L_OUT); ax.set_ylim(-S * 1.05, H_IN * 1.05); ax.set_aspect("equal")
ax.set_xlabel("x / S"); ax.set_ylabel("y / S")
ax.set_title("five domains — inlet channel, recirculation, recovery "
             f"(every 2nd mesh line; {d.n_cells:,} cells total)", fontsize=11)
fig.legend(handles=[Line2D([], [], color=COL[b], lw=2.5, label=f"{b}  {NAMES[b]}")
                    for b in range(5)], fontsize=10, ncol=5, loc="upper center",
           bbox_to_anchor=(0.5, 0.935), frameon=False)

ax = fig.add_axes([0.05, 0.07, 0.42, 0.52])
draw(ax, d, every=1, lw=0.45)
ax.plot([0, 0], [-S, 0], color="k", lw=5, zorder=9, solid_capstyle="butt")
ax.set_xlim(-1.6, 2.6); ax.set_ylim(-S * 1.05, H_IN * 1.05); ax.set_aspect("equal")
ax.set_xlabel("x / S"); ax.set_ylabel("y / S")
ax.set_title("zoom at the step: where three domains meet (every mesh line)", fontsize=11)

ax = fig.add_axes([0.53, 0.05, 0.45, 0.56], projection="3d")
for b in range(len(d.blocks)):
    blk = d.blocks[b]
    for kk in (0, blk.shape[2] - 1):
        X, Y = blk.x[:, :, kk], blk.y[:, :, kk]
        Z = np.full_like(X, blk.z[0, 0, kk])
        for i in range(0, X.shape[0], 6):
            ax.plot(X[i, :], Y[i, :], Z[i, :], color=COL[b], lw=.35, alpha=.8)
        for j in range(0, X.shape[1], 6):
            ax.plot(X[:, j], Y[:, j], Z[:, j], color=COL[b], lw=.35, alpha=.8)
ax.set_box_aspect((3.0, 0.55, 0.75), zoom=1.15)
ax.grid(False)
for a in (ax.xaxis, ax.yaxis, ax.zaxis):
    a.pane.set_visible(False); a._axinfo["grid"]["color"] = (1, 1, 1, 0)
ax.set_xticks([-5, 0, 10, 20, 30]); ax.set_yticks([-1, 0, 1]); ax.set_zticks([0, 2, 4])
ax.tick_params(labelsize=7, pad=0)
ax.set_xlabel("x / S", labelpad=0); ax.set_ylabel("y / S", labelpad=-4)
ax.set_zlabel("z / S", labelpad=-4)
ax.set_title(f"spanwise extent {L_SPAN:.0f}S, periodic (end planes only)", fontsize=11)
ax.view_init(elev=22, azim=-62)

fig.suptitle(f"Armaly backward-facing step, five domains — "
             f"ER = {(S+H_IN)/H_IN:.4f}, inlet {L_IN:.0f}S, split at {X1:.0f}S, "
             f"outlet {L_OUT:.0f}S", fontsize=13, y=0.99)
fig.savefig("figures/bfs5_grid.png", dpi=145, bbox_inches="tight")
print(f"  {len(d.blocks)} blocks, {d.n_cells:,} cells, {len(d.connections)} connections")
print(f"  validate(): {len(d.validate())} problem(s)")
print(f"  min J = {min(J.min() for J, _ in (d.block_metrics_cached(b) for b in range(5))):.3e}")
print("wrote figures/bfs5_grid.png")
