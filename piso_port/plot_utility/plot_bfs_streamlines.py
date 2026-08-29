"""
3D streamlines through the Armaly backward-facing step.

The two blocks share their x and z grids and PARTITION y without duplicating the interface, so
they reassemble into one structured array -- which is what makes a global streamline integrator
possible without any multi-block interpolation machinery. Streamlines are integrated with RK4
through a trilinear interpolant and coloured by speed.

The flow is two-dimensional at these Reynolds numbers, so the pathlines are planar; seeding
several spanwise stations shows that directly rather than asserting it.
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
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import RegularGridInterpolator

from run_armaly_bfs import run, H_IN, U_BULK
from armaly_bfs_grid import S, L_SPAN


def assemble(m, d, LOW=0, UP=1):
    """One structured grid from the two blocks (they partition y, sharing x and z)."""
    lo, up = d.blocks[LOW], d.blocks[UP]
    x = lo.x[:, 0, 0]
    y = np.concatenate([lo.y[0, :, 0], up.y[0, :, 0]])
    z = lo.z[0, 0, :]
    U = np.concatenate([m.u[LOW], m.u[UP]], axis=1)
    V = np.concatenate([m.v[LOW], m.v[UP]], axis=1)
    W = np.concatenate([m.w[LOW], m.w[UP]], axis=1)
    assert np.all(np.diff(y) > 0), "y must be monotonic across the interface"
    return x, y, z, U, V, W


def streamline(interp, p0, ds=0.02, n=4000, bounds=None):
    """RK4 through the velocity field, stopping at the domain edge or a stagnation point."""
    pts = [np.asarray(p0, float)]
    for _ in range(n):
        p = pts[-1]

        def f(q):
            q = np.clip(q, bounds[0], bounds[1])
            v = np.array([interp[k](q[None, :])[0] for k in range(3)])
            s = np.linalg.norm(v)
            return v / s if s > 1e-12 else np.zeros(3)

        k1 = f(p); k2 = f(p + 0.5 * ds * k1); k3 = f(p + 0.5 * ds * k2); k4 = f(p + ds * k3)
        step = ds * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        if np.linalg.norm(step) < 1e-10:
            break
        q = p + step
        if np.any(q < bounds[0]) or np.any(q > bounds[1]):
            break
        pts.append(q)
    return np.array(pts)


if __name__ == "__main__":
    RE, CACHE = 100.0, "results/bfs_field3d_Re100.npz"
    try:
        c = np.load(CACHE)
        x, y, z, U, V, W, xr = (c["x"], c["y"], c["z"], c["U"], c["V"], c["W"],
                                float(c["xr"]))
        print(f"  using cached {CACHE}  (x_r/S = {xr:.3f})")
    except (FileNotFoundError, OSError):
        xr, err, m, d = run(Re=RE, nx=80, ny_lo=18, ny_up=20, nz=6, dt=0.02, nsteps=1500,
                            report=True)
        if err:
            raise SystemExit(err)
        x, y, z, U, V, W = assemble(m, d)
        P = np.concatenate([m.p[0], m.p[1]], axis=1)
        np.savez(CACHE, x=x, y=y, z=z, U=U, V=V, W=W, P=P, xr=np.array(xr))
        print(f"  wrote {CACHE}")
    lo3 = np.array([x[0], y[0], z[0]])
    hi3 = np.array([x[-1], y[-1], z[-1]])
    interp = [RegularGridInterpolator((x, y, z), C, bounds_error=False, fill_value=None)
              for C in (U, V, W)]
    speed = np.sqrt(U ** 2 + V ** 2 + W ** 2)

    seeds = []
    for zz in np.linspace(z[0] + 0.2, z[-1] - 0.2, 4):          # several spanwise stations
        for yy in np.linspace(0.06, H_IN * 0.97, 7):            # across the inlet stream
            seeds.append((x[0] + 1e-3, yy, zz))
        for xx, yy in ((0.9, -0.30), (1.3, -0.55), (1.9, -0.35), (0.6, -0.70),
                       (2.4, -0.65)):                           # inside the recirculation
            seeds.append((xx, yy, zz))

    lines = [streamline(interp, p, bounds=(lo3, hi3)) for p in seeds]

    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_axes([0.02, 0.42, 0.96, 0.52], projection="3d")
    for L in lines:
        if len(L) < 3:
            continue
        sp = np.array([np.linalg.norm([interp[k](q[None, :])[0] for k in range(3)]) for q in L])
        segs = np.stack([L[:-1], L[1:]], axis=1)
        lc = Line3DCollection(segs, cmap="viridis",
                              norm=plt.Normalize(0, speed.max()), lw=1.0)
        lc.set_array(sp[:-1])
        ax.add_collection3d(lc)
    for zz in (z[0], z[-1]):                                     # the step face
        ax.plot([0, 0], [-S, 0], [zz, zz], color="k", lw=3)
    ax.set_xlim(0, 8); ax.set_ylim(-S, H_IN); ax.set_zlim(z[0], z[-1])
    ax.set_box_aspect((2.3, 0.78, 0.78), zoom=1.25)
    ax.grid(False)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_visible(False); a._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.set_xticks([0, 2, 4, 6, 8]); ax.set_yticks([-1, 0, 1])
    ax.set_zticks([0, 2, 4])
    ax.tick_params(labelsize=8, pad=0)
    ax.set_xlabel("x / S", labelpad=2); ax.set_ylabel("y / S", labelpad=-2)
    ax.set_zlabel("z / S", labelpad=-2)
    ax.set_title(f"3D streamlines, Re={RE:.0f}  (coloured by speed; black = step face)",
                 fontsize=10)
    ax.view_init(elev=26, azim=-58)

    # --- 2D section, where the bubble is actually legible
    ax = fig.add_axes([0.07, 0.06, 0.88, 0.30])
    k = len(z) // 2
    sp2 = speed[:, :, k].T
    ax.contourf(x, y, sp2, 24, cmap="viridis")
    xu = np.linspace(x[0], 12.0, 240)
    yu = np.linspace(y[0], y[-1], 180)
    Qu, Qv = (RegularGridInterpolator((x, y), F[:, :, k], bounds_error=False,
                                      fill_value=None)(
                  np.stack(np.meshgrid(xu, yu, indexing="ij"), -1)).T
              for F in (U, V))
    ax.streamplot(xu, yu, Qu, Qv, color="w", density=2.0, linewidth=0.7, arrowsize=0.7)
    ax.plot([0, 0], [-S, 0], color="k", lw=4, zorder=6)
    if np.isfinite(xr):
        ax.plot([xr], [-S], "r^", ms=13, zorder=7,
                label=f"reattachment  $x_r/S$ = {xr:.2f}")
        ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0, 12); ax.set_ylim(-S, H_IN); ax.set_aspect("equal")
    ax.set_xlabel("x / S"); ax.set_ylabel("y / S")
    ax.set_title("mid-span section: the recirculation bubble and the reattachment point",
                 fontsize=10)

    fig.suptitle(f"Armaly backward-facing step, Re = {RE:.0f}, two domains  "
                 f"(Armaly laminar reference: $x_r/S \\approx 3$ at Re = 100)",
                 fontsize=13, y=0.985)
    fig.savefig("figures/bfs_streamlines.png", dpi=145)
    print(f"  x_r/S = {xr:.3f}")
    print("wrote bfs_streamlines.png")
