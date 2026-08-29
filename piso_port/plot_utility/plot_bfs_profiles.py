"""
u, v and p profiles at five axial stations through the Armaly backward-facing step.

The stations bracket the recirculation: just downstream of the step, inside the bubble, at
reattachment, and twice in the recovering channel flow. Reading the three quantities together
is what makes the bubble legible -- u reverses near the bottom wall, v marks the entrainment
into and ejection out of the bubble, and p recovers as the expansion decelerates the stream.

The two blocks partition y while sharing x and z, so they reassemble into one structured array
and a profile can be drawn straight through the interface. That the curves show no kink at
y = 0 is itself a check on the multi-block coupling.

Pressure is defined only up to a constant here (convective outflow gives Neumann pressure all
round), so p is plotted relative to the domain mean.
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

from run_armaly_bfs import run, H_IN, U_BULK
from armaly_bfs_grid import S

RE = 100.0
CACHE = "results/bfs_field_Re100.npz"
STATIONS = [0.5, 2.0, 3.2, 6.0, 10.0]      # x/S; 3.2 is the measured reattachment


def field():
    try:
        d = np.load(CACHE)
        print(f"  using cached {CACHE}  (x_r/S = {d['xr']:.3f})")
        return {k: d[k] for k in d.files}
    except (FileNotFoundError, OSError):
        pass
    xr, err, m, dom = run(Re=RE, nx=80, ny_lo=18, ny_up=20, nz=6, dt=0.02, nsteps=1500)
    if err:
        raise SystemExit(err)
    LOW, UP = 0, 1
    lo, up = dom.blocks[LOW], dom.blocks[UP]
    out = dict(
        x=lo.x[:, 0, 0],
        y=np.concatenate([lo.y[0, :, 0], up.y[0, :, 0]]),
        z=lo.z[0, 0, :],
        U=np.concatenate([m.u[LOW], m.u[UP]], axis=1),
        V=np.concatenate([m.v[LOW], m.v[UP]], axis=1),
        P=np.concatenate([m.p[LOW], m.p[UP]], axis=1),
        ny_lo=np.array(lo.y.shape[1]), xr=np.array(xr))
    assert np.all(np.diff(out["y"]) > 0), "y must be monotonic across the block interface"
    np.savez(CACHE, **out)
    print(f"  wrote {CACHE}   x_r/S = {xr:.3f}")
    return out


if __name__ == "__main__":
    f = field()
    x, y, z = f["x"], f["y"], f["z"]
    k = len(z) // 2
    U, V = f["U"][:, :, k], f["V"][:, :, k]
    P = f["P"][:, :, k] - f["P"].mean()
    xr, ny_lo = float(f["xr"]), int(f["ny_lo"])
    idx = [int(np.argmin(np.abs(x - s))) for s in STATIONS]
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(idx)))

    fig = plt.figure(figsize=(15, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 2.6], hspace=0.32, wspace=0.26)

    # --- locator: where the five stations sit relative to the bubble
    ax = fig.add_subplot(gs[0, :])
    sp = np.sqrt(U ** 2 + V ** 2)
    ax.contourf(x, y, sp.T, 24, cmap="Greys", alpha=0.75)
    ax.contour(x, y, U.T, levels=[0.0], colors="crimson", linewidths=1.6)
    ax.plot([0, 0], [-S, 0], color="k", lw=4)
    ax.axhline(0.0, color="tab:blue", ls=":", lw=1.2)
    for c, i in zip(cols, idx):
        ax.axvline(x[i], color=c, lw=2.4)
        ax.text(x[i], H_IN * 1.06, f"{x[i]/S:.1f}", color=c, ha="center", fontsize=9,
                fontweight="bold")
    ax.plot([xr], [-S], "r^", ms=12, zorder=6)
    ax.set_xlim(0, 12); ax.set_ylim(-S, H_IN * 1.02); ax.set_aspect("equal")
    ax.set_xlabel("x / S"); ax.set_ylabel("y / S")
    ax.set_title(f"station locations   (red line: u = 0;  red triangle: reattachment "
                 f"$x_r/S$ = {xr:.2f};  dotted blue: block interface)", fontsize=10)

    for ax, (F, lab) in zip([fig.add_subplot(gs[1, c]) for c in range(3)],
                            [(U, "$u\\,/\\,U_{bulk}$"), (V, "$v\\,/\\,U_{bulk}$"),
                             (P, "$p - \\bar{p}$")]):
        for c, i in zip(cols, idx):
            ax.plot(F[i, :], y, color=c, lw=1.9, label=f"x/S = {x[i]/S:.1f}")
        ax.axvline(0.0, color="0.6", lw=0.8)
        ax.axhline(0.0, color="tab:blue", ls=":", lw=1.2)      # the block interface
        ax.axhline(-S, color="k", lw=2.5)
        ax.axhline(H_IN, color="k", lw=2.5)
        ax.set_ylim(-S * 1.03, H_IN * 1.03)
        ax.set_xlabel(lab); ax.set_ylabel("y / S")
        ax.grid(alpha=0.25)
    fig.axes[-3].legend(fontsize=9, loc="lower right")
    fig.axes[-3].set_title("streamwise velocity: reversed flow below the dashed line",
                           fontsize=10)
    fig.axes[-2].set_title("wall-normal velocity", fontsize=10)
    fig.axes[-1].set_title("pressure (relative to the domain mean)", fontsize=10)

    fig.suptitle(f"Armaly backward-facing step, Re = {RE:.0f}, two domains — "
                 f"profiles at five axial stations", fontsize=13)
    fig.savefig("figures/bfs_profiles.png", dpi=145, bbox_inches="tight")

    print("\n  station     min u      max |v|     p - p_mean      reversed flow?")
    for i in idx:
        rev = U[i, :].min() < 0
        print(f"   x/S={x[i]/S:5.1f}  {U[i,:].min():8.4f}  {np.abs(V[i,:]).max():9.4f}  "
              f"{P[i,:].mean():11.4f}      {'yes' if rev else 'no'}")
    print("\nwrote bfs_profiles.png")
