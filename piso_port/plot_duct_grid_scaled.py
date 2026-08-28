"""
The warped square duct grid rendered TO SCALE.

mpl_toolkits' 3D axes stretch each axis to fill the box by default, which silently exaggerates
whichever direction is shortest -- the duct is a unit cube and must look like one. The 3D view
uses set_box_aspect((1,1,1)) over equal data ranges; the 2D cuts use set_aspect("equal"). So
the warp you see is the warp that is there.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase1_grid_metrics import compute_numerical_metrics
from test_duct_implicit import duct_grid

def bare_3d(ax):
    """Strip matplotlib's coordinate scaffolding so only the computational mesh is drawn."""
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)          # the grey background panes
        axis.line.set_color((1, 1, 1, 0))     # the axis spines
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0)

A, N, NX = 0.10, 17, 24
PER = (True, False, False)
x, y, z, d1, d2, d3 = duct_grid(NX, N, A)
J, _ = compute_numerical_metrics(x, y, z, d1, d2, d3, periodic=PER)

# close the periodic seam so the drawn block is the full unit cube, not one cell short
xc = np.concatenate([x, x[:1] + 1.0], axis=0)
yc = np.concatenate([y, y[:1]], axis=0)
zc = np.concatenate([z, z[:1]], axis=0)
NXC = xc.shape[0]

fig = plt.figure(figsize=(16.5, 6.2))

# ------------------------------------------------------------ 3D, cutaway, to scale
ax = fig.add_subplot(1, 3, 1, projection="3d")
# Only the two NEAR walls (y=0, z=0) are drawn. Drawing all four overprints into a grey mass
# and hides exactly the thing worth seeing -- how the interior lines bend.
for i in range(0, NXC, 2):
    ax.plot(xc[i, 0, :], yc[i, 0, :], zc[i, 0, :], color="0.35", lw=0.5, alpha=0.9)
    ax.plot(xc[i, :, 0], yc[i, :, 0], zc[i, :, 0], color="0.35", lw=0.5, alpha=0.9)
for k in range(0, N, 2):
    ax.plot(xc[:, 0, k], yc[:, 0, k], zc[:, 0, k], color="tab:blue", lw=0.7, alpha=0.9)
for j in range(0, N, 2):
    ax.plot(xc[:, j, 0], yc[:, j, 0], zc[:, j, 0], color="tab:blue", lw=0.7, alpha=0.9)
for e in ([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)],):                      # duct outline
    for (a, b), (c, d) in zip(e[:-1], e[1:]):
        ax.plot([0, 0], [a, c], [b, d], color="r", lw=1.6)
        ax.plot([1, 1], [a, c], [b, d], color="r", lw=1.6)
for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
    ax.plot([0, 1], [a, a], [b, b], color="r", lw=1.6)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
ax.set_box_aspect((1, 1, 1))
bare_3d(ax)
ax.set_xlabel("x  (streamwise, periodic)"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("3D, two near walls only (1:1:1)", fontsize=10)
ax.view_init(elev=20, azim=-62)

# ------------------------------------------------------------ 2D cuts, equal aspect
kq = N // 4
ax = fig.add_subplot(1, 3, 2)
for j in range(N):
    ax.plot(xc[:, j, kq], yc[:, j, kq], color="tab:blue", lw=0.6)
for i in range(NXC):
    ax.plot(xc[i, :, kq], yc[i, :, kq], color="0.45", lw=0.6)
ax.axhline(0, color="r", lw=2); ax.axhline(1, color="r", lw=2)
ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(f"x-y plane at z = 1/4  (equal aspect)\nwarp amplitude A = {A}", fontsize=10)

iq = NX // 4
ax = fig.add_subplot(1, 3, 3)
for j in range(N):
    ax.plot(y[iq, j, :], z[iq, j, :], color="tab:blue", lw=0.6)
for k in range(N):
    ax.plot(y[iq, :, k], z[iq, :, k], color="0.45", lw=0.6)
ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "r-", lw=2)
ax.set_aspect("equal"); ax.set_xlabel("y"); ax.set_ylabel("z")
ax.set_title("y-z cross-section at x = 1/4\nseparable map -> rectangular, unevenly spaced",
             fontsize=10)

fig.suptitle(f"Warped square duct grid, to scale   "
             f"(A = {A}, {NX}x{N}x{N}, min J = {J.min():.3f}, worst cell angle 58 deg)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("duct_grid_scaled.png", dpi=150)
print(f"extent  x {xc.min():.3f}-{xc.max():.3f}   y {y.min():.3f}-{y.max():.3f}   "
      f"z {z.min():.3f}-{z.max():.3f}   (unit cube)")
print("wrote duct_grid_scaled.png")
