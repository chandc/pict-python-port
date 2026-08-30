"""
Five-domain BFS: streamlines and u, v, p profiles, including the resolved inlet channel.

ASSEMBLY. The five blocks form an L, not a rectangle: the upper row spans the whole channel
(inlet + recirculation + recovery) while the lower row exists only downstream of the step. They
are stitched onto one rectangular array with the solid quadrant (x < 0, y < 0) MASKED, which is
what lets a single streamplot cross every seam. The seams themselves are drawn, so any
discontinuity in the streamlines would be visible rather than hidden by interpolation.

The upstream station is the point of this grid: the profile reaching the step has developed over
5S of channel instead of being imposed at the step plane.
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
from scipy.interpolate import RegularGridInterpolator

from src import checkpoint as ck
from armaly_bfs5_grid import (S, H_IN, L_IN, X1, L_OUT, INLET, RECIRC_U, RECIRC_L,
                              RECOV_U, RECOV_L)

RUN = "results/fields/bfs5_Re100_dong.npz"     # written by run_armaly_bfs5.py
STATIONS = [-2.0, 0.5, 2.0, 3.0, 6.0, 12.0]


def field():
    """Read a SAVED run. This script never solves -- a completed run is post-processed from
    its checkpoint, so plotting costs seconds and never repeats an hour of compute."""
    import os
    if not os.path.exists(RUN):
        raise SystemExit(f"no saved run at {RUN} -- run `uv run run_armaly_bfs5.py` first")
    fl, meta = ck.load_fields(RUN)
    out = {"xr": meta["extra"]["xr"], "divi": meta["extra"]["div_interior"]}
    for b, nm in ((INLET, "in"), (RECIRC_U, "ru"), (RECIRC_L, "rl"),
                  (RECOV_U, "vu"), (RECOV_L, "vl")):
        g = np.load(RUN.replace(".npz", f"_geom{b}.npz"))
        k = g["x"].shape[2] // 2
        out[f"x_{nm}"] = g["x"][:, 0, k]
        out[f"y_{nm}"] = g["y"][0, :, k]
        for q in ("u", "v", "p"):
            out[f"{q}_{nm}"] = fl[q][b][:, :, k]
    print(f"  read {RUN}: Re {float(meta['extra']['Re']):.0f}, "
          f"{meta['nstep']} steps, x_r/S {float(out['xr']):.3f}")
    return out


f = field()
# --- upper row spans the full channel; lower row only downstream of the step
xu = np.concatenate([f["x_in"], f["x_ru"], f["x_vu"]])
xl = np.concatenate([f["x_rl"], f["x_vl"]])
yu, yl = f["y_ru"], f["y_rl"]
UPP = {q: np.concatenate([f[f"{q}_in"], f[f"{q}_ru"], f[f"{q}_vu"]], axis=0)
       for q in ("u", "v", "p")}
LOW = {q: np.concatenate([f[f"{q}_rl"], f[f"{q}_vl"]], axis=0) for q in ("u", "v", "p")}

# --- one rectangular grid, solid quadrant masked
XR = np.linspace(-L_IN, 22.0, 460)
YR = np.linspace(-S, H_IN, 190)
GX, GY = np.meshgrid(XR, YR, indexing="ij")
solid = (GX < 0) & (GY < 0)
out = {}
for q in ("u", "v", "p"):
    iu = RegularGridInterpolator((xu, yu), UPP[q], bounds_error=False, fill_value=None)
    il = RegularGridInterpolator((xl, yl), LOW[q], bounds_error=False, fill_value=None)
    a = np.where(GY >= 0, iu(np.stack([GX, GY], -1)), il(np.stack([np.maximum(GX, 0), GY], -1)))
    out[q] = np.where(solid, np.nan, a)

fig = plt.figure(figsize=(16, 10.5))

ax = fig.add_axes([0.06, 0.60, 0.90, 0.32])
sp = np.sqrt(out["u"] ** 2 + out["v"] ** 2)
ax.contourf(XR, YR, np.nan_to_num(sp).T, 26, cmap="viridis")
U2, V2 = np.nan_to_num(out["u"]).T, np.nan_to_num(out["v"]).T
ax.streamplot(XR, YR, U2, V2, color="w", density=2.4, linewidth=.7, arrowsize=.7)
ax.add_patch(plt.Rectangle((-L_IN, -S), L_IN, S, color="0.92", zorder=5))
ax.plot([0, 0], [-S, 0], color="k", lw=4, zorder=6)
ax.plot([-L_IN, 0], [0, 0], color="k", lw=3, zorder=6)
for xs, c in ((0.0, "w"), (X1, "w")):
    ax.axvline(xs, color=c, ls=":", lw=1.1, alpha=.8, zorder=7)
xr = float(f["xr"])
if np.isfinite(xr):
    ax.plot([xr], [-S], "r^", ms=13, zorder=8, label=f"reattachment  $x_r/S$ = {xr:.2f}")
    ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(-L_IN, 22); ax.set_ylim(-S, H_IN); ax.set_aspect("equal")
ax.set_xlabel("x / S"); ax.set_ylabel("y / S")
ax.set_title("streamlines across all five domains (dotted: block seams at x = 0 and x = 8S; "
             "grey: solid step)", fontsize=11)

cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(STATIONS)))
for n, (q, lab) in enumerate((("u", "$u/U_{bulk}$"), ("v", "$v/U_{bulk}$"),
                              ("p", "$p-\\bar p$"))):
    ax = fig.add_axes([0.06 + n * 0.325, 0.07, 0.26, 0.42])
    pm = np.nanmean(out["p"])
    for c, xs in zip(cols, STATIONS):
        i = int(np.argmin(np.abs(XR - xs)))
        val = out[q][i, :] - (pm if q == "p" else 0.0)
        ax.plot(val, YR, color=c, lw=1.9, label=f"x/S = {xs:+.1f}")
    ax.axvline(0, color="0.6", lw=.8)
    ax.axhline(0, color="tab:blue", ls=":", lw=1.1)
    ax.axhline(-S, color="k", lw=2.5); ax.axhline(H_IN, color="k", lw=2.5)
    ax.set_ylim(-S * 1.03, H_IN * 1.03); ax.set_xlabel(lab); ax.set_ylabel("y / S")
    ax.grid(alpha=.25)
    if n == 0:
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title("streamwise velocity (x<0 is the inlet channel)", fontsize=10)
    ax.set_title(["streamwise velocity", "wall-normal velocity",
                  "pressure"][n], fontsize=10)

fig.suptitle(f"Armaly BFS, five domains, Dong outflow — Re = 100, "
             f"$x_r/S$ = {xr:.2f}, interior div {float(f['divi']):.1e}", fontsize=13)
fig.savefig("figures/bfs5_fields.png", dpi=145, bbox_inches="tight")
print(f"  x_r/S = {xr:.3f}   interior divergence {float(f['divi']):.2e}")
print("wrote figures/bfs5_fields.png")
