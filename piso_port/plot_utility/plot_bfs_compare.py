"""
Armaly BFS: u, v and pressure profiles, with and without the Rhie-Chow correction.

The velocity panels carry both curves because the physics must NOT change -- reattachment and
the profiles are the result the solver is trusted for, and a pressure fix that moved them would
be a regression, not an improvement. The pressure panel is where the difference should live.
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
from diag_checkerboard import checkerboard
from armaly_bfs_grid import S

H_IN = 1.0612
STATIONS = [0.5, 2.0, 3.2, 6.0, 10.0]
OFF, ON = "results/bfs_field_Re100.npz", "results/bfs_field_rc_Re100.npz"

d0, d1 = np.load(OFF), np.load(ON)
x, y, z = d0["x"], d0["y"], d0["z"]
k = len(z) // 2
idx = [int(np.argmin(np.abs(x - s))) for s in STATIONS]
cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(idx)))

fig, axes = plt.subplots(1, 4, figsize=(17, 6.2))
panels = [("$u\\,/\\,U_{bulk}$", "U", "streamwise velocity"),
          ("$v\\,/\\,U_{bulk}$", "V", "wall-normal velocity")]
for ax, (lab, key, ttl) in zip(axes[:2], panels):
    for c, i in zip(cols, idx):
        ax.plot(d0[key][:, :, k][i, :], y, color=c, lw=2.4, alpha=.35)
        ax.plot(d1[key][:, :, k][i, :], y, color=c, lw=1.5, ls="--")
    ax.set_xlabel(lab); ax.set_title(ttl, fontsize=10)

for ax, dd, ttl in ((axes[2], d0, "pressure — Rhie-Chow OFF"),
                    (axes[3], d1, "pressure — Rhie-Chow ON")):
    P = dd["P"][:, :, k] - dd["P"].mean()
    amp, flip = checkerboard(P, axis=1)
    for c, i in zip(cols, idx):
        ax.plot(P[i, :], y, color=c, lw=1.9, label=f"x/S = {x[i]/S:.1f}")
    ax.set_xlabel("$p - \\bar{p}$")
    ax.set_title(f"{ttl}\namp {amp:.2e}, flips {flip:.2f}", fontsize=10)

for ax in axes:
    ax.axhline(0.0, color="tab:blue", ls=":", lw=1.2)
    ax.axhline(-S, color="k", lw=2.5); ax.axhline(H_IN, color="k", lw=2.5)
    ax.set_ylim(-S * 1.03, H_IN * 1.03); ax.set_ylabel("y / S"); ax.grid(alpha=.25)
axes[0].plot([], [], color="0.4", lw=2.4, alpha=.35, label="Rhie-Chow off")
axes[0].plot([], [], color="0.4", lw=1.5, ls="--", label="Rhie-Chow on")
axes[0].legend(fontsize=8, loc="lower right")
axes[3].legend(fontsize=8, loc="upper right")
fig.suptitle("Armaly backward-facing step, Re = 100, two domains — "
             "velocities unchanged, pressure checkerboard removed", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("figures/bfs_profiles_compare.png", dpi=145, bbox_inches="tight")

for nm, dd in (("off", d0), ("on", d1)):
    P = dd["P"][:, :, k] - dd["P"].mean()
    a, f = checkerboard(P, axis=1)
    au, fu = checkerboard(dd["U"][:, :, k], axis=1)
    print(f"  rhie_chow {nm:>3}:  p amp {a:.3e} flips {f:.2f}   "
          f"u amp {au:.3e} flips {fu:.2f}   x_r/S {float(dd['xr']):.3f}")
print(f"  max|u_on - u_off| = {np.abs(d1['U']-d0['U']).max():.3e}")
print("wrote figures/bfs_profiles_compare.png")
