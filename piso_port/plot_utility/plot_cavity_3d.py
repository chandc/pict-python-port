"""3D views: the warped curvilinear mesh, and 3D streamlines of the cavity flow."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)          # figures/ and results/ paths are relative to the root

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import RegularGridInterpolator
from src.phase1_grid_metrics import compute_numerical_metrics

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#c9c7c0"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)

d = np.load("results/cavity_results.npz")
N, warp = int(d["N"]), float(d["warp"])
x, y, z = d["x"], d["y"], d["z"]
g = np.linspace(0, 1, N)                      # uniform computational grid
h = 1.0 / (N - 1)
# metrics are a pure function of the coordinates, so recompute them here rather
# than re-running the solver just to save them out
_J, MET = compute_numerical_metrics(x, y, z, h, h, h)

def face_wire(ax, X, Y, Z, step, color, lw, alpha):
    """Draw the mesh lines of one boundary face."""
    for a in range(0, X.shape[0], step):
        ax.plot(X[a, :], Y[a, :], Z[a, :], color=color, lw=lw, alpha=alpha)
    for b in range(0, X.shape[1], step):
        ax.plot(X[:, b], Y[:, b], Z[:, b], color=color, lw=lw, alpha=alpha)

def streamlines(tag, seeds, n_steps=1400, ds=0.004):
    """
    Integrate streamlines in COMPUTATIONAL space, where the grid is uniform and the
    interpolation is trivial: dxi^i/ds = U^i/|U|, with U^i the contravariant components.
    The path is then mapped back to physical space through the coordinate arrays.
    """
    u, v, w = d[f"{tag}_u"], d[f"{tag}_v"], d[f"{tag}_w"]
    keys = [("xi_x", "xi_y", "xi_z"), ("eta_x", "eta_y", "eta_z"), ("zeta_x", "zeta_y", "zeta_z")]
    U = [MET[k[0]]*u + MET[k[1]]*v + MET[k[2]]*w for k in keys]

    grid = (g, g, g)
    fU = [RegularGridInterpolator(grid, c, bounds_error=False, fill_value=None) for c in U]
    fX = [RegularGridInterpolator(grid, c, bounds_error=False, fill_value=None) for c in (x, y, z)]
    fS = RegularGridInterpolator(grid, np.sqrt(u**2 + v**2 + w**2),
                                 bounds_error=False, fill_value=None)

    def vel(p):
        return np.stack([f(p) for f in fU], axis=-1)

    paths = []
    for direction in (1.0, -1.0):
        P = seeds.copy()
        alive = np.ones(len(P), bool)
        trail = [P.copy()]
        for _ in range(n_steps):
            k1 = vel(P); n1 = np.linalg.norm(k1, axis=1, keepdims=True)
            k1 = direction * k1 / np.maximum(n1, 1e-12)
            k2 = vel(np.clip(P + 0.5*ds*k1, 0, 1))
            k2 = direction * k2 / np.maximum(np.linalg.norm(k2, axis=1, keepdims=True), 1e-12)
            P = P + ds * k2
            alive &= n1[:, 0] > 1e-4
            alive &= np.all((P > -1e-9) & (P < 1 + 1e-9), axis=1)
            P = np.clip(P, 0, 1)
            P[~alive] = np.nan
            trail.append(P.copy())
        paths.append(np.array(trail))

    segs, vals = [], []
    for trail in paths:
        for i in range(trail.shape[1]):
            c = trail[:, i, :]
            c = c[~np.isnan(c).any(axis=1)]
            if len(c) < 4:
                continue
            phys = np.stack([f(c) for f in fX], axis=-1)
            s = fS(c)
            segs.extend(list(np.stack([phys[:-1], phys[1:]], axis=1)))
            vals.extend(list(0.5*(s[:-1] + s[1:])))
    return np.array(segs), np.array(vals)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                     "axes.labelcolor": INK_2, "figure.facecolor": SURFACE})
fig = plt.figure(figsize=(13.2, 6.6))

# ---------------------------------------------------------------- grid
ax = fig.add_subplot(1, 2, 1, projection="3d")
step = 2
faces = [(x[0], y[0], z[0]), (x[-1], y[-1], z[-1]),
         (x[:, 0], y[:, 0], z[:, 0]), (x[:, -1], y[:, -1], z[:, -1]),
         (x[:, :, 0], y[:, :, 0], z[:, :, 0]), (x[:, :, -1], y[:, :, -1], z[:, :, -1])]
for X, Y, Z in faces:
    face_wire(ax, X, Y, Z, N - 1, GRID, 0.8, 0.9)      # box outline only
# The mapping x = xi + A sin(pi eta) sin(pi zeta) has ZERO warp on the boundary faces
# (a sine vanishes there), so the boundary mesh would misleadingly look Cartesian.
# The warp is largest in the interior -- show three orthogonal mid-plane cuts instead.
m = N // 2
face_wire(ax, x[m], y[m], z[m], step, "#86b6ef", 0.8, 0.9)
face_wire(ax, x[:, m], y[:, m], z[:, m], step, "#3987e5", 0.8, 0.9)
face_wire(ax, x[:, :, m], y[:, :, m], z[:, :, m], step, "#1c5cab", 0.8, 0.9)
ax.text2D(0.0, 0.95, "Curvilinear grid", transform=ax.transAxes, color=INK,
          fontsize=12, fontweight="bold", ha="left", va="bottom")
ax.text2D(0.0, -0.055, f"{N}³ cells, warp {warp} — three orthogonal mid-plane cuts; the warp vanishes\n"
          f"on the outer faces by construction, so the interior is shown",
          transform=ax.transAxes, color=INK_2, fontsize=9)

# ---------------------------------------------------------------- flow
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
# Seed across the span (y) so the 3D structure shows: the end walls drive a secondary
# flow, so streamlines at different y are NOT copies of the same 2D spiral.
sx, sy, sz = np.meshgrid([0.30, 0.62], np.linspace(0.10, 0.90, 7), [0.45, 0.72],
                         indexing="ij")
seeds = np.stack([sx.ravel(), sy.ravel(), sz.ravel()], axis=1)
segs, vals = streamlines("Re100", seeds, n_steps=620, ds=0.006)
norm = Normalize(0, np.nanmax(vals))
lc = Line3DCollection(segs, cmap=SEQ, norm=norm, linewidths=1.25, alpha=0.85)
lc.set_array(vals)
ax2.add_collection3d(lc)
for X, Y, Z in faces:
    face_wire(ax2, X, Y, Z, N - 1, GRID, 0.8, 0.9)   # box outline only
ax2.text2D(0.0, 0.95, "3D flow, Re = 100", transform=ax2.transAxes, color=INK,
           fontsize=12, fontweight="bold", ha="left", va="bottom")
ax2.text2D(0.0, -0.055, "streamlines coloured by velocity magnitude; lid slides in +x at z = 1",
           transform=ax2.transAxes, color=INK_2, fontsize=9)

for a_ in (ax, ax2):
    a_.set_xlabel("x"); a_.set_ylabel("y"); a_.set_zlabel("z")
    a_.set_xlim(x.min(), x.max()); a_.set_ylim(y.min(), y.max()); a_.set_zlim(z.min(), z.max())
    a_.set_box_aspect((1, 1, 1)); a_.view_init(elev=22, azim=-52)
    a_.xaxis.pane.set_facecolor(SURFACE); a_.yaxis.pane.set_facecolor(SURFACE)
    a_.zaxis.pane.set_facecolor(SURFACE)
    for axis in (a_.xaxis, a_.yaxis, a_.zaxis):
        axis.pane.set_edgecolor(GRID); axis._axinfo["grid"]["color"] = GRID
    a_.tick_params(colors=INK_2, labelsize=8)

cax = fig.add_axes([0.92, 0.24, 0.013, 0.46])
cb = fig.colorbar(lc, cax=cax); cb.set_label("|u|", color=INK_2, fontsize=9)
cb.outline.set_edgecolor(GRID); cb.ax.tick_params(colors=INK_2, labelsize=8)

fig.text(0.028, 0.955, "3D lid-driven cavity — grid and flow", color=INK,
         fontsize=14, fontweight="bold", ha="left")
fig.subplots_adjust(left=0.01, right=0.90, top=0.94, bottom=0.09, wspace=0.02)
fig.savefig("figures/cavity_flow_3d.png", dpi=170, facecolor=SURFACE)
print("wrote cavity_flow_3d.png")
