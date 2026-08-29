"""The Armaly backward-facing step grid: two domains, periodic spanwise, coloured by block."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)          # figures/ and results/ paths are relative to the root

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from armaly_bfs_grid import bfs_domain, S, H_IN, H_TOT, L_OUT, L_SPAN

d, LOW, UP = bfs_domain(nx=90, ny_lo=20, ny_up=22, nz=12)
COL = {LOW: "#e4572e", UP: "#17bebb"}
NAME = {LOW: "lower (recirculation)", UP: "upper (inlet stream)"}
Jmin = min(d.block_metrics_cached(b)[0].min() for b in range(2))

fig = plt.figure(figsize=(17, 9.5))

# ------------------------------------------------------------------ 3D
ax = fig.add_subplot(2, 1, 1, projection="3d")
for b in (LOW, UP):
    blk, c = d.blocks[b], COL[b]
    x, y, z = blk.x, blk.y, blk.z
    for k in (0, blk.shape[2] - 1):
        for j in range(0, blk.shape[1], 3):
            ax.plot(x[:, j, k], y[:, j, k], z[:, j, k], color=c, lw=0.45, alpha=0.85)
        for i in range(0, blk.shape[0], 4):
            ax.plot(x[i, :, k], y[i, :, k], z[i, :, k], color=c, lw=0.45, alpha=0.85)
    for i in range(0, blk.shape[0], 10):                    # spanwise lines
        for j in (0, blk.shape[1] - 1):
            ax.plot(x[i, j, :], y[i, j, :], z[i, j, :], color=c, lw=0.7, alpha=0.9)
ax.plot([0, 0], [-S, 0], [0, 0], color="k", lw=3)           # the step face
ax.plot([0, 0], [-S, 0], [L_SPAN, L_SPAN], color="k", lw=3)
ax.set_xlim(0, L_OUT); ax.set_ylim(-S, H_IN); ax.set_zlim(0, L_SPAN)
ax.set_box_aspect((4.2, 0.55, 0.75))
ax.grid(False)
for a in (ax.xaxis, ax.yaxis, ax.zaxis):
    a.pane.set_visible(False); a._axinfo["grid"]["color"] = (1, 1, 1, 0)
ax.set_xticks([0, 10, 20, 30]); ax.set_yticks([-1, 0, 1]); ax.set_zticks([0, 2, 4])
ax.tick_params(labelsize=8, pad=0)
ax.set_xlabel("x / S", labelpad=2); ax.set_ylabel("y / S", labelpad=-2)
ax.set_zlabel("z / S", labelpad=-2)
ax.set_title("3D grid — two domains, spanwise periodic (black = step face)", fontsize=10)
ax.view_init(elev=22, azim=-64)

# ------------------------------------------------------------------ near-step section
ax = fig.add_subplot(2, 2, 3)
for b in (LOW, UP):
    blk, c = d.blocks[b], COL[b]
    x, y = blk.x[:, :, 0], blk.y[:, :, 0]
    m = x[:, 0] <= 6.0
    for j in range(blk.shape[1]):
        ax.plot(x[m, j], y[m, j], color=c, lw=0.55)
    for i in np.where(m)[0]:
        ax.plot(x[i], y[i], color=c, lw=0.55)
ax.plot([0, 0], [-S, 0], color="k", lw=3.5, zorder=6, label="step face (wall)")
ax.axhline(0, color="0.35", ls="--", lw=1.4, zorder=5)
ax.text(3.0, 0.06, "connected interface (interior face, NOT a wall)", fontsize=8.5,
        color="0.25", ha="center")
ax.set_xlim(-0.6, 6.0); ax.set_ylim(-S * 1.06, H_IN * 1.06)
ax.set_aspect("equal"); ax.set_xlabel("x / S"); ax.set_ylabel("y / S")
ax.legend(fontsize=8, loc="upper right")
ax.set_title("near the step: clustering at both walls and the shear layer", fontsize=10)

# ------------------------------------------------------------------ full section
ax = fig.add_subplot(2, 2, 4)
for b in (LOW, UP):
    blk, c = d.blocks[b], COL[b]
    x, y = blk.x[:, :, 0], blk.y[:, :, 0]
    for j in range(0, blk.shape[1], 2):
        ax.plot(x[:, j], y[:, j], color=c, lw=0.5)
    for i in range(0, blk.shape[0], 2):
        ax.plot(x[i], y[i], color=c, lw=0.5)
    ax.text(L_OUT * 0.55, blk.y.mean(), NAME[b], color=c, fontsize=10, weight="bold",
            ha="center", bbox=dict(fc="w", ec=c, alpha=0.9, boxstyle="round,pad=0.25"))
ax.plot([0, 0], [-S, 0], color="k", lw=3.5, zorder=6)
ax.axhline(0, color="0.35", ls="--", lw=1.0)
ax.set_xlim(-0.5, L_OUT); ax.set_ylim(-S * 1.06, H_IN * 1.06)
ax.set_xlabel("x / S"); ax.set_ylabel("y / S")
ax.set_title(f"full domain, L = {L_OUT:.0f}S   (Armaly reattachment is 6–7S at Re≈400)",
             fontsize=10)

fig.suptitle(f"Armaly backward-facing step: ER = {H_TOT/H_IN:.4f}, span {L_SPAN:.0f}S periodic, "
             f"{d.n_cells} cells in 2 domains, min J = {Jmin:.2f}", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("figures/armaly_bfs_grid.png", dpi=145)
print(f"  {d.n_cells} cells, min J {Jmin:.3f}, validate {len(d.validate())} problems")
print("wrote armaly_bfs_grid.png")
