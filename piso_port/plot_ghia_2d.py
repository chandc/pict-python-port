"""2D cavity (span-periodic) vs Ghia, Ghia & Shin (1982), Re = 100 and Re = 400."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e2e0da"
S1, S2 = "#2a78d6", "#eb6834"          # categorical slots 1 and 2

# Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982) 387-411, Tables I & II.
G = {
 100: dict(
  y=[0.0,0.0547,0.0625,0.0703,0.1016,0.1719,0.2813,0.4531,0.5,0.6172,0.7344,0.8516,0.9531,0.9609,0.9688,0.9766,1.0],
  u=[0.0,-0.03717,-0.04192,-0.04775,-0.06434,-0.1015,-0.15662,-0.2109,-0.20581,-0.13641,0.00332,0.23151,0.68717,0.73722,0.78871,0.84123,1.0],
  x=[0.0,0.0625,0.0703,0.0781,0.0938,0.1563,0.2266,0.2344,0.5,0.8047,0.8594,0.9063,0.9453,0.9531,0.9609,0.9688,1.0],
  v=[0.0,0.07027,0.08344,0.09233,0.10091,0.16077,0.17507,0.17527,0.05454,-0.24533,-0.22445,-0.16914,-0.10313,-0.08864,-0.07391,-0.05906,0.0]),
 400: dict(
  y=[0.0,0.0547,0.0625,0.0703,0.1016,0.1719,0.2813,0.4531,0.5,0.6172,0.7344,0.8516,0.9531,0.9609,0.9688,0.9766,1.0],
  u=[0.0,-0.08186,-0.09266,-0.10338,-0.14612,-0.24299,-0.32726,-0.17119,-0.11477,0.02135,0.16256,0.29093,0.55892,0.61756,0.68439,0.75837,1.0],
  x=[0.0,0.0625,0.0703,0.0781,0.0938,0.1563,0.2266,0.2344,0.5,0.8047,0.8594,0.9063,0.9453,0.9531,0.9609,0.9688,1.0],
  v=[0.0,0.18360,0.19713,0.20920,0.22965,0.28124,0.30174,0.30203,0.05186,-0.38598,-0.44993,-0.23827,-0.22847,-0.19254,-0.15663,-0.12146,0.0]),
}

d = np.load("cavity_2d.npz")
def line(tag):
    n = int(d["N"]); m = n // 2
    z = d[f"{tag}_z"][m, :]; u = d[f"{tag}_u"][m, :]
    x = d[f"{tag}_x"][:, m]; w = d[f"{tag}_w"][:, m]
    return z, u, x, w

CASES = [("Re100_sou", 100, "Re 100, SOU", S1, "-"),
         ("Re100_central", 100, "Re 100, central", S2, "--"),
         ("Re400_sou", 400, "Re 400, SOU", S1, "-")]
have = [c for c in CASES if f"{c[0]}_u" in d]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                     "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
                     "xtick.color": INK_2, "ytick.color": INK_2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.4))

for row, Re in enumerate((100, 400)):
    cs = [c for c in have if c[1] == Re]
    if not cs:
        for ax in axes[row]: ax.axis("off")
        continue
    for col, which in enumerate(("u", "w")):
        ax = axes[row][col]
        ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
        g = G[Re]
        if which == "u":
            ax.plot(g["u"], g["y"], "o", ms=6.5, mfc="none", mew=1.6, color=INK,
                    label="Ghia et al. 1982", zorder=5)
        else:
            ax.plot(g["x"], g["v"], "o", ms=6.5, mfc="none", mew=1.6, color=INK,
                    label="Ghia et al. 1982", zorder=5)
        for tag, _, lbl, col_, ls in cs:
            z, u, x, w = line(tag)
            if which == "u": ax.plot(u, z, ls, color=col_, lw=2.0, label=lbl, zorder=3)
            else:            ax.plot(x, w, ls, color=col_, lw=2.0, label=lbl, zorder=3)
        ax.set_title(f"Re = {Re}:  " + ("u on the vertical centreline" if which == "u"
                     else "v on the horizontal centreline"),
                     color=INK, fontsize=10.5, fontweight="bold", loc="left", pad=8)
        ax.set_xlabel("u" if which == "u" else "x")
        ax.set_ylabel("y" if which == "u" else "v")
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        leg = ax.legend(frameon=False, fontsize=8.5, loc="best")
        for t in leg.get_texts(): t.set_color(INK_2)

fig.text(0.045, 0.968, "2D lid-driven cavity vs the Ghia et al. benchmark",
         color=INK, fontsize=13.5, fontweight="bold", ha="left")
fig.text(0.045, 0.937,
         f"{int(d['N'])}x{int(d['N'])} in-plane, spanwise direction PERIODIC — no end walls, "
         "so this is a genuine 2D cavity, matching Ghia's configuration.",
         color=INK_2, fontsize=9.5, ha="left")
fig.subplots_adjust(left=0.075, right=0.975, top=0.90, bottom=0.06, hspace=0.30, wspace=0.22)
fig.savefig("cavity_2d_vs_ghia.png", dpi=170, facecolor=SURFACE)

print(f"{'case':18s} {'u_min':>9s} {'Ghia':>9s} {'v_max':>9s} {'Ghia':>9s} {'v_min':>9s} {'Ghia':>9s}")
for tag, Re, lbl, _, _ in have:
    z, u, x, w = line(tag); g = G[Re]
    print(f"{lbl:18s} {u.min():9.4f} {min(g['u']):9.4f} {w.max():9.4f} {max(g['v']):9.4f} "
          f"{w.min():9.4f} {min(g['v']):9.4f}")
    du = np.interp(g["y"], z, u) - np.array(g["u"])
    dv = np.interp(g["x"], x, w) - np.array(g["v"])
    print(f"{'':18s}   RMS deviation at Ghia's stations:  u {np.sqrt((du**2).mean()):.4f}"
          f"   v {np.sqrt((dv**2).mean()):.4f}")
print("\nwrote cavity_2d_vs_ghia.png")
