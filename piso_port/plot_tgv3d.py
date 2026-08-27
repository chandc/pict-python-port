"""3D Taylor-Green: energy, enstrophy, and the numerical-dissipation budget."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e2e0da"
S1, S2 = "#2a78d6", "#eb6834"

d = np.load("tgv3d.npz"); nu = float(d["nu"]); N = int(d["N"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                     "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
                     "xtick.color": INK_2, "ytick.color": INK_2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5))
cases = [("sou", "2nd-order upwind", S1, "-"), ("central", "central", S2, "--")]

for ax, kind in zip(axes, ("E", "Z", "budget")):
    for tag, lbl, col, ls in cases:
        t, E, Z = d[f"{tag}_t"], d[f"{tag}_E"], d[f"{tag}_Z"]
        if kind == "E":
            ax.semilogy(t, E, ls, color=col, lw=2.0, label=lbl)
        elif kind == "Z":
            ax.plot(t, Z, ls, color=col, lw=2.0, label=lbl)
        else:
            num = -np.gradient(E, t) - 2 * nu * Z
            m = (t > 0.05) & (t < t[-1] - 1e-9)   # drop the endpoint: np.gradient is one-sided there
            ax.plot(t[m], 100 * np.abs(num[m]) / np.abs(2 * nu * Z[m]), ls,
                    color=col, lw=2.0, label=lbl)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.set_xlabel("t")
    leg = ax.legend(frameon=False, fontsize=8.5, loc="best")
    for tx in leg.get_texts(): tx.set_color(INK_2)

axes[0].set_ylabel("kinetic energy  E"); axes[0].set_title(
    "Energy decay", color=INK, fontsize=11, fontweight="bold", loc="left", pad=8)
axes[1].set_ylabel("enstrophy  Z"); axes[1].set_title(
    "Enstrophy — peaks at t=0, no cascade", color=INK, fontsize=11,
    fontweight="bold", loc="left", pad=8)
axes[2].set_ylabel("numerical / physical dissipation  [%]"); axes[2].set_title(
    "Numerical dissipation", color=INK, fontsize=11,
    fontweight="bold", loc="left", pad=8)

fig.text(0.035, 0.965, "3D Taylor-Green vortex — energy budget", color=INK,
         fontsize=13.5, fontweight="bold", ha="left")
fig.text(0.035, 0.925,
         f"{N}³ fully periodic, ν = {nu}, rotational + BDF2.  The exact periodic identity "
         f"−dE/dt = 2νZ makes any gap the scheme's own numerical dissipation.",
         color=INK_2, fontsize=9.5, ha="left")
fig.subplots_adjust(left=0.055, right=0.99, top=0.80, bottom=0.11, wspace=0.28)
fig.savefig("tgv3d_energy.png", dpi=170, facecolor=SURFACE)
print("wrote tgv3d_energy.png")
