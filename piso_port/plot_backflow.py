"""
Streamlines for the backflow outflow test: a vortex driven out through the outlet at Re=200.

Both outflow treatments are shown at the same instants so the comparison is like-for-like.
Red marks on the right edge are outlet nodes with u < 0 -- fluid entering the domain through
the OUTLET, which is the situation a convective outflow condition is known to fail on and the
one Dong's -1/2|u|^2*Theta term exists to handle.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, contextlib, warnings
warnings.filterwarnings("ignore")
from test_outflow import build, UMAX, L

STEPS = (0, 100, 200, 300, 500)
KINDS = ("convective", "dong")
DT = 0.005

snaps = {}
for kind in KINDS:
    s, bc, prof = build(49, 33, kind, dt=DT, nu=UMAX * 1.0 / 200.0,
                        vortex=(0.55, 0.80 * L, 0.5, 0.22))
    snaps[(kind, 0)] = (s.u[:, :, 0].copy(), s.v[:, :, 0].copy())
    done = 0
    with contextlib.redirect_stdout(io.StringIO()):
        for target in STEPS[1:]:
            for _ in range(target - done):
                s.step()
            done = target
            snaps[(kind, target)] = (s.u[:, :, 0].copy(), s.v[:, :, 0].copy())
    X1, Y1 = s.x[:, 0, 0], s.y[0, :, 0]

fig, axes = plt.subplots(len(KINDS), len(STEPS), figsize=(4.0 * len(STEPS), 5.4),
                         sharex=True, sharey=True)
vmax = max(np.hypot(*snaps[k]).max() for k in snaps)
for r, kind in enumerate(KINDS):
    for c, n in enumerate(STEPS):
        ax = axes[r, c]
        U, V = snaps[(kind, n)]
        spd = np.hypot(U, V)
        ax.contourf(X1, Y1, spd.T, np.linspace(0, vmax, 25), cmap="viridis")
        ax.streamplot(X1, Y1, U.T, V.T, color="w", density=1.25, linewidth=0.6,
                      arrowsize=0.6)
        back = U[-1] < 0
        if back.any():
            ax.plot(np.full(back.sum(), X1[-1]), Y1[back], "r_", ms=9, mew=2.2)
        ax.set_xlim(X1[0], X1[-1]); ax.set_ylim(0, 1)
        frac = back.mean()
        ax.set_title(f"{kind}  t={n*DT:.2f}" + (f"   backflow {frac:.0%}" if frac else ""),
                     fontsize=9)
        if r == len(KINDS) - 1: ax.set_xlabel("x")
        if c == 0: ax.set_ylabel(f"{kind}\ny", fontsize=9)

fig.suptitle("Vortex leaving through the outlet, Re=200 — streamlines over speed "
             "(red = backflow into the domain)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("backflow_streamlines.png", dpi=140)
print("wrote backflow_streamlines.png")

# difference between the two treatments, at the last instant
Uc, Vc = snaps[("convective", STEPS[-1])]
Ud, Vd = snaps[("dong", STEPS[-1])]
d = np.hypot(Uc - Ud, Vc - Vd)
print(f"max |u_convective - u_dong| at t={STEPS[-1]*DT:.2f}: {d.max():.3e} "
      f"({d.max()/vmax*100:.2f}% of peak speed);  largest near the outlet: "
      f"{d[-6:].max():.3e}")
