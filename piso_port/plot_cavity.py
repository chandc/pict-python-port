"""Figure for the 3D lid-driven cavity on a warped curvilinear grid."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata

# --- design tokens (validated reference palette) -----------------------------
SURFACE    = "#fcfcfb"
INK        = "#0b0b0b"
INK_2      = "#52514e"
GRID       = "#e2e0da"
SERIES     = ["#2a78d6", "#eb6834"]           # categorical slots 1, 2 - fixed order
BLUE_RAMP  = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)   # one hue, light -> dark

d = np.load("cavity_results.npz")
N, warp = int(d["N"]), float(d["warp"])
x, z = d["x"], d["z"]
j = N // 2                                     # mid-plane in y; recirculation is in x-z

cases = [("Re100", "Re = 100"), ("Re20", "Re = 20")]
speeds = {}
for tag, _ in cases:
    U, W = d[f"{tag}_u"][:, j, :], d[f"{tag}_w"][:, j, :]
    speeds[tag] = np.sqrt(U**2 + W**2)
vmax = max(s.max() for s in speeds.values())

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})
fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.4))
fig.patch.set_facecolor(SURFACE)

X, Z = x[:, j, :], z[:, j, :]
Xi, Zi = np.meshgrid(np.linspace(X.min(), X.max(), 90), np.linspace(Z.min(), Z.max(), 90))
pts = np.column_stack([X.ravel(), Z.ravel()])

# --- top row: velocity magnitude + streamlines (small multiples, shared scale)
for col, (tag, label) in enumerate(cases):
    ax = axes[0, col]
    m = ax.pcolormesh(X, Z, speeds[tag], cmap=SEQ, vmin=0, vmax=vmax,
                      shading="gouraud", rasterized=True)
    # the warped mesh itself, every 4th line, recessive
    for i in range(0, N, 4):
        ax.plot(X[i, :], Z[i, :], color=SURFACE, lw=0.4, alpha=0.35)
        ax.plot(X[:, i], Z[:, i], color=SURFACE, lw=0.4, alpha=0.35)
    Ui = griddata(pts, d[f"{tag}_u"][:, j, :].ravel(), (Xi, Zi), method="linear")
    Wi = griddata(pts, d[f"{tag}_w"][:, j, :].ravel(), (Xi, Zi), method="linear")
    ax.streamplot(Xi, Zi, Ui, Wi, color=INK_2, linewidth=0.8, density=1.15,
                  arrowsize=0.8, arrowstyle="->")
    ax.set_title(label, color=INK, fontsize=11, fontweight="bold", pad=8, loc="left")
    ax.set_xlabel("x"); ax.set_ylabel("z" if col == 0 else "")
    ax.set_aspect("equal"); ax.set_xlim(X.min(), X.max()); ax.set_ylim(Z.min(), Z.max())
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.annotate("lid  →", xy=(0.5, 1.005), xycoords="axes fraction", ha="center",
                va="bottom", color=INK_2, fontsize=8.5)


# --- bottom row: centreline profiles (2 series -> legend + direct labels)
i_c, k_c = N // 2, N // 2
prof = [
    (axes[1, 0], "u  along the vertical centreline", "u", "z",
     lambda t: (d[f"{t}_u"][i_c, j, :], z[i_c, j, :])),
    (axes[1, 1], "w  along the horizontal centreline", "x", "w",
     lambda t: (x[:, j, k_c], d[f"{t}_w"][:, j, k_c])),
]
for ax, title, xl, yl, get in prof:
    ax.axhline(0, color=GRID, lw=1, zorder=1)
    ax.axvline(0, color=GRID, lw=1, zorder=1)
    curves = [get(tag) for tag, _ in cases]
    for c, (tag, label) in enumerate(cases):
        a, b = curves[c]
        ax.plot(a, b, color=SERIES[c], lw=2.0, label=label, zorder=3,
                solid_capstyle="round")
    # Direct-label where the curves are furthest apart, not at the endpoints: both series
    # start and end on the same boundary values, so endpoint labels sit on top of each other.
    val = 0 if title.startswith("u") else 1          # which component varies between series
    sep = np.abs(curves[0][val] - curves[1][val])
    m_i = int(np.argmax(sep))
    for c, (tag, label) in enumerate(cases):
        a, b = curves[c]
        ax.plot(a[m_i], b[m_i], "o", color=SERIES[c], ms=5, zorder=4)
        dx = 7 if (a[m_i] >= curves[1-c][0][m_i]) else -7
        ax.annotate(label, xy=(a[m_i], b[m_i]), xytext=(dx, 0),
                    textcoords="offset points", color=INK_2, fontsize=8.5,
                    va="center", ha="left" if dx > 0 else "right", zorder=5)
    ax.set_title(title, color=INK, fontsize=11, fontweight="bold", pad=8, loc="left")
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.9); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="lower right"
                    if title.startswith("u") else "upper right")
    for t in leg.get_texts(): t.set_color(INK_2)

div = max(np.abs(d[f"{t}_div"]).max() for t, _ in cases)
y_mid = float(d["y"][0, j, 0])
fig.suptitle("3D lid-driven cavity on a warped curvilinear grid", color=INK,
             fontsize=14, fontweight="bold", x=0.062, ha="left", y=0.982)
fig.text(0.062, 0.947,
         f"PISO  ·  {N}³ cells  ·  grid warp {warp}  ·  mid-plane y = {y_mid:.2f}"
         f"  ·  steady state  ·  max flux divergence {div:.0e}",
         color=INK_2, fontsize=9.5, ha="left")

# layout first, then park the colorbar in an explicit slot so it cannot be shifted
fig.subplots_adjust(left=0.075, right=0.875, top=0.885, bottom=0.065,
                    hspace=0.32, wspace=0.22)
cax = fig.add_axes([0.895, 0.505, 0.016, 0.38])
cb = fig.colorbar(m, cax=cax)
cb.set_label("velocity magnitude  |u|", color=INK_2, fontsize=9)
cb.outline.set_edgecolor(GRID); cb.ax.tick_params(colors=INK_2, labelsize=8.5)

fig.savefig("cavity_flow.png", dpi=170, facecolor=SURFACE)
print("wrote cavity_flow.png")
