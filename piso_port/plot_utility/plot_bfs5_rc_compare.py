"""
Re = 300 five-domain BFS: u, v and p profiles with and without the Rhie-Chow correction.

Velocities are OVERLAID because the physics must not move -- reattachment and the profiles are
what the solver is trusted for, and a pressure fix that shifted them would be a regression, not
an improvement. Pressure gets one panel each, because that is where the difference should live
and a single overlaid panel would just show one curve buried in the other's sawtooth.

Both runs are read from saved checkpoints. Nothing is re-solved.
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

from src import checkpoint as ck
from diag_checkerboard import checkerboard
from armaly_bfs5_grid import S, H_IN, L_IN, INLET, RECIRC_U, RECIRC_L, RECOV_U, RECOV_L

RE = 300
OFF = "results/fields/bfs5_Re300_dong.npz"
ON = "results/fields/bfs5_Re300_dong_rc_n26720.npz"
STATIONS = [-2.0, 1.0, 4.0, 6.7, 10.0, 16.0]        # 6.7 is the measured reattachment


def load(path):
    fl, meta = ck.load_fields(path)
    g = {b: np.load(path.replace(".npz", f"_geom{b}.npz")) for b in range(5)}
    k = g[0]["x"].shape[2] // 2
    xu = np.concatenate([g[b]["x"][:, 0, k] for b in (INLET, RECIRC_U, RECOV_U)])
    xl = np.concatenate([g[b]["x"][:, 0, k] for b in (RECIRC_L, RECOV_L)])
    yu = g[RECIRC_U]["y"][0, :, k]
    yl = g[RECIRC_L]["y"][0, :, k]
    up = {q: np.concatenate([fl[q][b][:, :, k] for b in (INLET, RECIRC_U, RECOV_U)], axis=0)
          for q in ("u", "v", "p")}
    lo = {q: np.concatenate([fl[q][b][:, :, k] for b in (RECIRC_L, RECOV_L)], axis=0)
          for q in ("u", "v", "p")}
    return dict(xu=xu, xl=xl, yu=yu, yl=yl, up=up, lo=lo,
                xr=float(meta["extra"]["xr"]),
                div=float(meta["extra"]["div_interior"]))


def profile(d, q, xs):
    """(values, y) down the full channel height at station xs; upstream of the step the
    lower row does not exist, so only the inlet-channel part is returned."""
    iu = int(np.argmin(np.abs(d["xu"] - xs)))
    vu, yu = d["up"][q][iu, :], d["yu"]
    if xs < 0:
        return vu, yu
    il = int(np.argmin(np.abs(d["xl"] - xs)))
    return np.concatenate([d["lo"][q][il, :], vu]), np.concatenate([d["yl"], yu])


A, B = load(OFF), load(ON)
cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(STATIONS)))

fig, axes = plt.subplots(1, 4, figsize=(17.5, 6.4))
for ax, q, lab, ttl in ((axes[0], "u", "$u\\,/\\,U_{bulk}$", "streamwise velocity"),
                        (axes[1], "v", "$v\\,/\\,U_{bulk}$", "wall-normal velocity")):
    for c, xs in zip(cols, STATIONS):
        va, ya = profile(A, q, xs)
        vb, yb = profile(B, q, xs)
        ax.plot(va, ya, color=c, lw=2.6, alpha=.35)
        ax.plot(vb, yb, color=c, lw=1.4, ls="--", label=f"x/S = {xs:+.1f}")
    ax.set_xlabel(lab); ax.set_title(ttl, fontsize=10)

pmA = np.concatenate([A["up"]["p"].ravel(), A["lo"]["p"].ravel()]).mean()
pmB = np.concatenate([B["up"]["p"].ravel(), B["lo"]["p"].ravel()]).mean()
for ax, d, pm, ttl in ((axes[2], A, pmA, "pressure — Rhie-Chow OFF"),
                       (axes[3], B, pmB, "pressure — Rhie-Chow ON")):
    P = np.concatenate([d["lo"]["p"][:, :], d["up"]["p"][:len(d["xl"]), :]], axis=1)
    amp, flip = checkerboard(P, axis=1)
    for c, xs in zip(cols, STATIONS):
        v, y = profile(d, "p", xs)
        ax.plot(v - pm, y, color=c, lw=1.9)
    ax.set_xlabel("$p - \\bar{p}$")
    ax.set_title(f"{ttl}\namp {amp:.2e},  flips {flip:.2f}", fontsize=10)

for ax in axes:
    ax.axhline(0.0, color="tab:blue", ls=":", lw=1.1)
    ax.axhline(-S, color="k", lw=2.5); ax.axhline(H_IN, color="k", lw=2.5)
    ax.set_ylim(-S * 1.04, H_IN * 1.04); ax.set_ylabel("y / S"); ax.grid(alpha=.25)
axes[0].plot([], [], color="0.4", lw=2.6, alpha=.35, label="Rhie-Chow off")
axes[0].plot([], [], color="0.4", lw=1.4, ls="--", label="Rhie-Chow on")
axes[0].legend(fontsize=7.5, loc="lower right", ncol=1)
fig.suptitle(f"Armaly BFS, five domains, Dong outflow, Re = {RE} — "
             f"$x_r/S$ = {A['xr']:.3f} (off) vs {B['xr']:.3f} (on), "
             f"interior div {A['div']:.1e} / {B['div']:.1e}", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("figures/bfs5_rc_compare_Re300.png", dpi=145, bbox_inches="tight")

print(f"  x_r/S  off {A['xr']:.3f}   on {B['xr']:.3f}   "
      f"({abs(B['xr']-A['xr'])/A['xr']*100:.2f}% apart)")
for nm, d, pm in (("off", A, pmA), ("on", B, pmB)):
    P = np.concatenate([d["lo"]["p"], d["up"]["p"][:len(d["xl"])]], axis=1)
    U = np.concatenate([d["lo"]["u"], d["up"]["u"][:len(d["xl"])]], axis=1)
    ap, fp = checkerboard(P, axis=1); au, fu = checkerboard(U, axis=1)
    print(f"  {nm:>3}: p amp {ap:.3e} flips {fp:.2f} | u amp {au:.3e} flips {fu:.2f}")
print("wrote figures/bfs5_rc_compare_Re300.png")
