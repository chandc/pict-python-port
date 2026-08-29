"""
The wall-preserving warped duct grid used by test_duct_implicit.py.

The point of this grid is that it is strongly non-orthogonal WITHOUT moving the walls, so the
exact Fourier-series solution for a square duct remains valid on it. The stock make_grid warp
does move the walls, which is why it cannot be used for an accuracy test against that series.
Both are drawn here side by side so the difference is visible rather than asserted.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)          # figures/ and results/ paths are relative to the root

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.phase1_grid_metrics import make_grid, compute_numerical_metrics
from test_duct_implicit import duct_grid

# A = 0.10 is the strongest warp the ACCURACY tests actually use, and it is a defensible mesh:
# worst cell angle 58 deg, min(J) 0.465. A = 0.20 (31 deg cells) is a solver stress case, not a
# grid anyone would build for a duct -- it is shown in the amplitude panel, not as the headline.
def bare_3d(ax):
    """Strip matplotlib's coordinate scaffolding so only the computational mesh is drawn."""
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)          # the grey background panes
        axis.line.set_color((1, 1, 1, 0))     # the axis spines
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0)

A, N, NX = 0.10, 17, 24
PER = (True, False, False)

x, y, z, dxi, deta, dzeta = duct_grid(NX, N, A)
J, _ = compute_numerical_metrics(x, y, z, dxi, deta, dzeta, periodic=PER)

fig = plt.figure(figsize=(17, 9.5))

# ---------------------------------------------------------------- 3D view
ax = fig.add_subplot(2, 3, 1, projection="3d")
# the four duct walls, drawn as surface grid lines
kw = dict(color="0.25", lw=0.5, alpha=0.75)
for j in (0, N - 1):
    for i in range(NX):
        ax.plot(x[i, j, :], y[i, j, :], z[i, j, :], **kw)
    for k in range(0, N, 2):
        ax.plot(x[:, j, k], y[:, j, k], z[:, j, k], **kw)
for k in (0, N - 1):
    for i in range(NX):
        ax.plot(x[i, :, k], y[i, :, k], z[i, :, k], **kw)
    for j in range(0, N, 2):
        ax.plot(x[:, j, k], y[:, j, k], z[:, j, k], **kw)
ax.set_xlim(x.min(), x.max()); ax.set_ylim(-0.02, 1.02); ax.set_zlim(-0.02, 1.02)
ax.set_xlabel("x (streamwise, periodic)"); ax.set_ylabel("y"); ax.set_zlabel("z")
bare_3d(ax)
ax.set_title(f"Duct walls, A={A}\n(walls stay flat: y,z $\\in$ {{0,1}} exactly)", fontsize=10)
ax.view_init(elev=22, azim=-58)

# ------------------------------------------------- interior: a constant-j sheet
ax = fig.add_subplot(2, 3, 2, projection="3d")
for j in range(0, N, 3):
    for k in range(0, N, 3):
        ax.plot(x[:, j, k], y[:, j, k], z[:, j, k], color="tab:blue", lw=0.7, alpha=0.8)
ax.set_xlim(x.min(), x.max()); ax.set_ylim(-0.02, 1.02); ax.set_zlim(-0.02, 1.02)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
bare_3d(ax)
ax.set_title("Streamwise grid lines\n(they meander: $\\partial y/\\partial\\xi \\neq 0$)", fontsize=10)
ax.view_init(elev=22, azim=-58)

# ------------------------------------------------- x-y cut: the money shot
# At fixed xi the map is SEPARABLE (y depends only on eta, z only on zeta), so a cross-section
# stays rectangular -- just unevenly spaced. The non-orthogonality is streamwise, so the cut
# that shows it is a constant-z plane: interior lines meander while the walls stay dead flat.
# Cut at zeta = 1/4, NOT the mid-plane. The stock warp carries sin(2 pi zeta), which vanishes
# at zeta = 1/2 -- cutting there would show its walls perfectly flat and make the comparison
# in the next panel a lie. zeta = 1/4 is where sin(2 pi zeta) = 1, i.e. its worst case. The
# wall-preserving warp has no zeta dependence in y, so this cut is representative for it.
kmid = N // 4
ax = fig.add_subplot(2, 3, 3)
for j in range(N):
    ax.plot(x[:, j, kmid], y[:, j, kmid], "-", color="tab:blue", lw=0.6, alpha=0.8)
for i in range(NX):
    ax.plot(x[i, :, kmid], y[i, :, kmid], "-", color="0.4", lw=0.6, alpha=0.8)
ax.axhline(0, color="r", lw=2.0); ax.axhline(1, color="r", lw=2.0, label="duct wall")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=8, loc="center right")
ax.set_ylim(-0.25, 1.25)
ax.set_title(f"x-y cut at z=1/4, A={A}\ninterior meanders, walls dead flat", fontsize=10)

# ------------------------------------------------- the contrast: stock warp, same cut
xs, ys, zs, dxs, des, dzs = make_grid((NX, N, N), warp=A, periodic=PER)
ax = fig.add_subplot(2, 3, 4)
for j in range(N):
    ax.plot(xs[:, j, kmid], ys[:, j, kmid], "-", color="tab:green", lw=0.6, alpha=0.8)
for i in range(NX):
    ax.plot(xs[i, :, kmid], ys[i, :, kmid], "-", color="0.4", lw=0.6, alpha=0.8)
ax.axhline(0, color="r", lw=2.0)
ax.axhline(1, color="r", lw=2.0, label="where the wall should be")
wall_s = max(np.abs(ys[:, 0, :]).max(), np.abs(ys[:, -1, :] - 1).max())
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=8, loc="center right")
ax.set_ylim(-0.25, 1.25)
ax.set_title(f"Stock make_grid warp, same A and cut\nwalls MOVE by up to {wall_s:.2f} "
             "-> series invalid", fontsize=10)

# ------------------------------------------------- Jacobian
ax = fig.add_subplot(2, 3, 5)
# Station NX//4, not 0: the warp carries sin(2 pi xi), which is zero at xi = 0, so J is
# identically 1 there and the panel would show nothing. xi = 1/4 is where the warp peaks.
imax = NX // 4
im = ax.contourf(y[imax], z[imax], J[imax], 30, cmap="viridis")
plt.colorbar(im, ax=ax, label="J")
ax.set_aspect("equal"); ax.set_xlabel("y"); ax.set_ylabel("z")
ax.set_title(f"Cell volume J at x = 1/4 (warp peak)\nmin(J) = {J.min():.3f} > 0 (untangled)",
             fontsize=10)

# ------------------------------------------------- validity vs A
ax = fig.add_subplot(2, 3, 6)
As = np.linspace(0, 0.30, 25)
mjw, mjs = [], []
for a in As:
    xa, ya, za, d1, d2, d3 = duct_grid(NX, N, a)
    mjw.append(compute_numerical_metrics(xa, ya, za, d1, d2, d3, periodic=PER)[0].min())
    xb, yb, zb, e1, e2, e3 = make_grid((NX, N, N), warp=a, periodic=PER)
    mjs.append(compute_numerical_metrics(xb, yb, zb, e1, e2, e3, periodic=PER)[0].min())
ax.plot(As, mjw, "o-", ms=3, label="wall-preserving (this grid)")
ax.plot(As, mjs, "s-", ms=3, color="tab:red", label="stock make_grid")
ax.axhline(0, color="k", lw=1)
ax.axvline(A, color="0.5", ls=":", lw=1)
ax.text(A, ax.get_ylim()[1] * 0.9, f" A={A}", fontsize=8, color="0.35")
ax.set_xlabel("warp amplitude A"); ax.set_ylabel("min(J)")
ax.set_title("Grid validity: min(J) must stay > 0", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle(f"Wall-preserving warped square duct  (A = {A}, {NX}x{N}x{N})", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("figures/duct_grid_3d.png", dpi=150)
print(f"min(J) = {J.min():.4f}   wall displacement = "
      f"{max(np.abs(y[:,0,:]).max(), np.abs(y[:,-1,:]-1).max(), np.abs(z[:,:,0]).max(), np.abs(z[:,:,-1]-1).max()):.1e}")
print("wrote duct_grid_3d.png")
