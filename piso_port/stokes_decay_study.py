"""
Wall-bounded Stokes: what happens to the decay, and to the decay RATE, over a 100x drop.

Runs the alpha=1, nu=1, no-slip channel (exact sigma = -9.313739) until the disturbance
AMPLITUDE has fallen by 100x, i.e. to  t = ln(100)/|sigma| = 0.4945, and records the running
decay rate throughout. Two sweeps:

  * 5 grid resolutions at fixed dt
  * 5 time steps at fixed grid

The question is not just "is the final rate right" but "does the rate stay put". A scheme can
land on the correct eigenvalue early and then drift as the signal decays four orders of
magnitude in energy -- that drift is what a long integration exposes and a short one hides.
"""
import io, contextlib, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from src.piso_numpy_3d import PISOSolver
from src.phase1_grid_metrics import compute_numerical_metrics
from test_stokes_channel import stokes_mode, ALPHA, NU, AMP, SIGMA_REF

DROP = 100.0                                   # amplitude drop
T_END = np.log(DROP) / abs(SIGMA_REF)          # 0.4945
SIG_EX, PHI, DPHI = stokes_mode()


def trajectory(ny, dt, scheme="rotational", time_scheme="bdf2", nsample=200):
    Lx = 2 * np.pi / ALPHA
    nx = max(12, int(round((ny - 1) * 0.75)))
    nz = 4
    s = PISOSolver((nx, ny, nz), warp=1e-9, nu=NU, dt=dt, corrector_steps=2,
                   periodic=(True, False, True), scheme=scheme, time_scheme=time_scheme,
                   boundary_flux_mode="impermeable", pressure_coef="rowsum",
                   pressure_tol=1e-12)
    xi = np.arange(nx) / nx
    eta = np.linspace(0, 1, ny)
    zeta = np.arange(nz) / nz
    XI, ETA, ZETA = np.meshgrid(xi, eta, zeta, indexing="ij")
    s.x, s.y, s.z = Lx * XI, -1.0 + 2.0 * ETA, ZETA
    s.h = (1.0 / nx, 1.0 / (ny - 1), 1.0 / nz)
    s.J, s.metrics = compute_numerical_metrics(s.x, s.y, s.z, *s.h, periodic=s.per,
                                               period=(Lx, 1.0, 1.0))
    s.u = AMP * DPHI(s.y) * np.cos(ALPHA * s.x)
    s.v = AMP * ALPHA * PHI(s.y) * np.sin(ALPHA * s.x)
    s.w[:] = 0.0

    nsteps = int(round(T_END / dt))
    every = max(1, nsteps // nsample)
    ts, Es = [0.0], [0.5 * np.mean(s.u ** 2 + s.v ** 2)]
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(nsteps):
            s.step()
            if (i + 1) % every == 0 or i + 1 == nsteps:
                ts.append((i + 1) * dt)
                Es.append(0.5 * np.mean(s.u ** 2 + s.v ** 2))
    return dict(t=np.array(ts), E=np.array(Es), nx=nx, ny=ny, dt=dt,
                wall=time.time() - t0, nsteps=nsteps)


def rates(d, w=6):
    """Running decay rate d(ln A)/dt = (1/2) d(ln E)/dt, centred over +/- w samples."""
    t, E = d["t"], d["E"]
    lnA = 0.5 * np.log(E)
    i = np.arange(w, len(t) - w)
    return t[i], (lnA[i + w] - lnA[i - w]) / (t[i + w] - t[i - w])


def settled(d, lo=0.25):
    t, E = d["t"], d["E"]
    i0 = np.searchsorted(t, lo)
    return 0.5 * np.log(E[-1] / E[i0]) / (t[-1] - t[i0])


if __name__ == "__main__":
    print(f"exact sigma = {SIG_EX:.6f};  run to t = {T_END:.4f} "
          f"(amplitude x1/{DROP:.0f}, energy x1/{DROP**2:.0f})\n")

    NYS, DT_G = (25, 33, 49, 65, 97), 2e-4
    DTS, NY_T = (4e-4, 2e-4, 1e-4, 5e-5, 2.5e-5), 49

    grid, tstep = [], []
    print(f"A. grid sweep at dt={DT_G:g}")
    print(f"   {'ny':>4} {'nx':>4} {'steps':>6} {'A_end/A_0':>10} {'sigma_settled':>14}"
          f" {'rel err':>9} {'drift':>9} {'wall':>8}")
    for ny in NYS:
        d = trajectory(ny, DT_G); grid.append(d)
        tr, rr = rates(d)
        sg = settled(d)
        drift = rr[-1] - rr[np.searchsorted(tr, 0.15)]
        print(f"   {ny:4d} {d['nx']:4d} {d['nsteps']:6d} {np.sqrt(d['E'][-1]/d['E'][0]):10.5f}"
              f" {sg:14.6f} {abs(sg/SIG_EX-1):9.2e} {drift:+9.2e} {d['wall']:7.1f}s")

    print(f"\nB. dt sweep at ny={NY_T}")
    print(f"   {'dt':>9} {'steps':>6} {'A_end/A_0':>10} {'sigma_settled':>14}"
          f" {'rel err':>9} {'drift':>9} {'wall':>8}")
    for dt in DTS:
        d = trajectory(NY_T, dt); tstep.append(d)
        tr, rr = rates(d)
        sg = settled(d)
        drift = rr[-1] - rr[np.searchsorted(tr, 0.15)]
        print(f"   {dt:9.2e} {d['nsteps']:6d} {np.sqrt(d['E'][-1]/d['E'][0]):10.5f}"
              f" {sg:14.6f} {abs(sg/SIG_EX-1):9.2e} {drift:+9.2e} {d['wall']:7.1f}s")

    np.savez("results/stokes_decay_study.npz",
             **{f"grid{i}_{k}": v for i, d in enumerate(grid) for k, v in d.items()},
             **{f"dt{i}_{k}": v for i, d in enumerate(tstep) for k, v in d.items()},
             nys=np.array(NYS), dts=np.array(DTS), sig=SIG_EX, dt_g=DT_G, ny_t=NY_T)
    print("\nwrote stokes_decay_study.npz")
