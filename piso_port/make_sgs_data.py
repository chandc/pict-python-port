"""
Generate a-priori SGS training data: filter a fine simulation onto a coarse grid and compute
the exact sub-grid term the coarse grid cannot represent.

Filtering the incompressible momentum equation gives

    d(u_bar)/dt + div(u_bar u_bar) = -grad(p_bar) + nu lap(u_bar) - div(tau)
    tau_ij = filter(u_i u_j) - u_bar_i u_bar_j

so -div(tau) is exactly the force a coarse solver is missing. That is the regression target.

The initial condition is a RANDOM divergence-free field with broadband content, not a
Taylor-Green vortex. TGV at the Reynolds numbers this solver can reach is laminar and smooth,
so its sub-grid term is a nearly trivial function of the resolved field -- a closure could
"learn" it without learning anything about closure. A broadband field makes the filtered-scale
interaction non-trivial.
"""
import numpy as np, time, warnings, sys
warnings.filterwarnings("ignore")
from src.piso_numpy_3d import PISOSolver

FINE, COARSE, RATIO = 48, 16, 3
NU, DT = 0.004, 0.004
N_SNAP, SPACING, SPINUP = 24, 12, 60


def random_solenoidal(n, seed=0, kmax=6, amp=1.0):
    """Random divergence-free field, band-limited, via a projection in Fourier space."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(n, d=1.0 / n)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0
    mask = (np.sqrt(K2) <= kmax)
    fields = []
    for _ in range(3):
        a = (rng.normal(size=(n, n, n)) + 1j*rng.normal(size=(n, n, n))) * mask
        fields.append(a)
    # project out the compressive part:  u_hat <- u_hat - k (k.u_hat)/|k|^2
    dot = KX*fields[0] + KY*fields[1] + KZ*fields[2]
    fields = [f - K*dot/K2 for f, K in zip(fields, (KX, KY, KZ))]
    out = [np.real(np.fft.ifftn(f)) for f in fields]
    s = amp / max(np.abs(o).max() for o in out)
    return [o * s for o in out]


def box_filter(f, r=RATIO):
    n = f.shape[0] // r
    return f.reshape(n, r, n, r, n, r).mean(axis=(1, 3, 5))


def d_dx(f, h, axis):
    return (np.roll(f, -1, axis) - np.roll(f, 1, axis)) / (2 * h)


def sgs_force(uf, vf, wf, h_coarse):
    """-div(tau), the force the coarse grid is missing."""
    fine = [uf, vf, wf]
    bar = [box_filter(f) for f in fine]
    force = [np.zeros_like(bar[0]) for _ in range(3)]
    for i in range(3):
        for j in range(3):
            tau = box_filter(fine[i] * fine[j]) - bar[i] * bar[j]
            force[i] -= d_dx(tau, h_coarse, j)
    return bar, force


if __name__ == "__main__":
    s = PISOSolver(FINE, warp=1e-9, nu=NU, dt=DT, corrector_steps=2, periodic=True,
                   scheme="rotational", time_scheme="bdf2", pressure_tol=1e-11)
    u, v, w = random_solenoidal(FINE, seed=1, kmax=6, amp=1.0)
    s.u, s.v, s.w = u, v, w
    hc = 1.0 / COARSE
    print(f"fine {FINE}^3 -> coarse {COARSE}^3 (box filter {RATIO}^3), nu={NU}")
    t0 = time.time()
    for it in range(SPINUP):
        s.step()
    print(f"  spin-up {SPINUP} steps done ({time.time()-t0:.0f}s), "
          f"max|u|={np.abs(s.u).max():.3f}", flush=True)
    B, F = [], []
    for k in range(N_SNAP):
        for _ in range(SPACING):
            s.step()
        bar, force = sgs_force(s.u, s.v, s.w, hc)
        B.append(np.stack(bar)); F.append(np.stack(force))
        if k % 6 == 0:
            print(f"  snapshot {k:2d}  |u_bar|max={np.abs(bar[0]).max():.3f}  "
                  f"|sgs|rms={np.sqrt((force[0]**2).mean()):.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)
    np.savez("results/sgs_data.npz", inputs=np.array(B), targets=np.array(F),
             fine=FINE, coarse=COARSE, nu=NU)
    print(f"saved sgs_data.npz  {len(B)} snapshots of shape {B[0].shape}")
