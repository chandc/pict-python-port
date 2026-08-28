"""Decay curves and running decay rate for the wall-bounded Stokes mode (100x amplitude drop)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("stokes_decay_study.npz")
SIG, NYS, DTS, DT_G, NY_T = d["sig"], d["nys"], d["dts"], d["dt_g"], d["ny_t"]

def get(pfx, i):
    return d[f"{pfx}{i}_t"], d[f"{pfx}{i}_E"]

def rate(t, E, w=6):
    lnA = 0.5 * np.log(E)
    i = np.arange(w, len(t) - w)
    return t[i], (lnA[i + w] - lnA[i - w]) / (t[i + w] - t[i - w])

fig, ax = plt.subplots(2, 2, figsize=(14.5, 9))
cg = plt.cm.viridis(np.linspace(0, .85, len(NYS)))
ct = plt.cm.plasma(np.linspace(0, .85, len(DTS)))

for row, (pfx, labels, cols, ttl) in enumerate((
        ("grid", [f"ny={n}" for n in NYS], cg, f"grid sweep at dt={DT_G:g}"),
        ("dt",   [f"dt={x:g}" for x in DTS], ct, f"dt sweep at ny={NY_T}"))):
    for i, (lab, c) in enumerate(zip(labels, cols)):
        t, E = get(pfx, i)
        ax[row, 0].semilogy(t, np.sqrt(E / E[0]), color=c, lw=1.4, label=lab)
        tr, rr = rate(t, E)
        ax[row, 1].plot(tr, rr, color=c, lw=1.4, label=lab)
    t0, E0 = get(pfx, 0)
    ax[row, 0].semilogy(t0, np.exp(SIG * t0), "k--", lw=1.2, label="exact $e^{\\sigma t}$")
    ax[row, 0].axhline(0.01, color="r", ls=":", lw=1)
    ax[row, 0].text(0.01, 0.0115, " amplitude / 100", color="r", fontsize=8)
    ax[row, 0].set_xlabel("t"); ax[row, 0].set_ylabel("$A(t)/A(0)$")
    ax[row, 0].set_title(f"Decay — {ttl}", fontsize=10)
    ax[row, 0].legend(fontsize=8); ax[row, 0].grid(alpha=.3)

    ax[row, 1].axhline(SIG, color="k", ls="--", lw=1.2, label="exact $\\sigma=-9.313739$")
    ax[row, 1].set_xlabel("t"); ax[row, 1].set_ylabel("running $\\sigma = d\\ln A/dt$")
    ax[row, 1].set_title(f"Decay rate — {ttl}", fontsize=10)
    ax[row, 1].legend(fontsize=8); ax[row, 1].grid(alpha=.3)
    ax[row, 1].set_ylim(SIG - 0.10, SIG + 0.06)

fig.suptitle("Wall-bounded Stokes mode: decay and decay rate over a 100x amplitude drop "
             "($\\alpha=1$, $\\nu=1$, no-slip)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("stokes_decay.png", dpi=145)
print("wrote stokes_decay.png")
