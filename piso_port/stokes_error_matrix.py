"""
Decay-rate error of the wall-bounded Stokes mode as a function of BOTH resolution and dt.

The two one-at-a-time sweeps in stokes_decay_study.py each saturated: refining the grid alone
stalled at 7.5e-4 (temporal-limited) and refining dt alone stalled at 9.3e-4 (spatial-limited).
Neither reaches the exact eigenvalue, and neither sweep can show why on its own. The full
matrix separates the two contributions.

sigma is measured over the SETTLED window [0.05, 0.10], never from t=0: the initial field is an
eigenmode of the continuous operator, not the discrete one, so an early window reports the
startup transient instead of the eigenvalue (that mistake produced an apparent temporal order
of 0.04 for a scheme that is ~1.7 here).
"""
import io, contextlib, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from src.piso_numpy_3d import PISOSolver
from src.phase1_grid_metrics import compute_numerical_metrics
from test_stokes_channel import stokes_mode, ALPHA, NU, AMP

SIG_EX, PHI, DPHI = stokes_mode()
LO, HI = 0.05, 0.10
NYS = (25, 33, 49, 65, 97)
DTS = (4e-4, 2e-4, 1e-4, 5e-5, 2.5e-5)


def sigma(ny, dt, scheme="rotational", time_scheme="bdf2"):
    Lx = 2 * np.pi / ALPHA
    nx = max(12, int(round((ny - 1) * 0.75)))
    nz = 4
    s = PISOSolver((nx, ny, nz), warp=1e-9, nu=NU, dt=dt, corrector_steps=2,
                   periodic=(True, False, True), scheme=scheme, time_scheme=time_scheme,
                   boundary_flux_mode="impermeable", pressure_coef="rowsum",
                   pressure_tol=1e-12)
    xi = np.arange(nx) / nx; eta = np.linspace(0, 1, ny); ze = np.arange(nz) / nz
    XI, ETA, ZE = np.meshgrid(xi, eta, ze, indexing="ij")
    s.x, s.y, s.z = Lx * XI, -1.0 + 2.0 * ETA, ZE
    s.h = (1.0 / nx, 1.0 / (ny - 1), 1.0 / nz)
    s.J, s.metrics = compute_numerical_metrics(s.x, s.y, s.z, *s.h, periodic=s.per,
                                               period=(Lx, 1.0, 1.0))
    s.u = AMP * DPHI(s.y) * np.cos(ALPHA * s.x)
    s.v = AMP * ALPHA * PHI(s.y) * np.sin(ALPHA * s.x)
    s.w[:] = 0.0
    i0, n = int(round(LO / dt)), int(round(HI / dt))
    E0 = None
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(n):
            s.step()
            if i + 1 == i0:
                E0 = 0.5 * np.mean(s.u ** 2 + s.v ** 2)
    E1 = 0.5 * np.mean(s.u ** 2 + s.v ** 2)
    return 0.5 * np.log(E1 / E0) / (HI - LO), nx


if __name__ == "__main__":
    print(f"exact sigma = {SIG_EX:.8f};  rate measured over the settled window "
          f"[{LO}, {HI}]\n")
    S = np.zeros((len(NYS), len(DTS)))
    hdr = "  ny\\dt  " + "".join(f"{d:>12.2e}" for d in DTS)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    t0 = time.time()
    for i, ny in enumerate(NYS):
        row = []
        for j, dt in enumerate(DTS):
            S[i, j], nx = sigma(ny, dt)
            row.append(f"{abs(S[i,j]-SIG_EX):12.3e}")
        print(f"  {ny:4d}   " + "".join(row) + f"   [{time.time()-t0:6.0f}s]", flush=True)
    np.savez("results/stokes_error_matrix.npz", S=S, nys=np.array(NYS), dts=np.array(DTS), sig=SIG_EX)

    err = np.abs(S - SIG_EX)
    print("\n  relative error |sigma_num/sigma_exact - 1|")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for i, ny in enumerate(NYS):
        print(f"  {ny:4d}   " + "".join(f"{e/abs(SIG_EX):12.3e}" for e in err[i]))

    print("\n  raw sigma")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for i, ny in enumerate(NYS):
        print(f"  {ny:4d}   " + "".join(f"{v:12.6f}" for v in S[i]))

    # Separate the two contributions: along the finest-dt column the error is spatial;
    # along the finest-grid row it is temporal.
    print("\n  spatial-limited (finest dt column) and temporal-limited (finest grid row):")
    h = 2.0 / (np.array(NYS) - 1)
    es = err[:, -1]
    rs = [np.log(es[i] / es[i + 1]) / np.log(h[i] / h[i + 1]) for i in range(len(NYS) - 1)]
    print("     spatial order vs h : " + ", ".join(f"{r:.2f}" for r in rs))
    et = err[-1, :]
    rt = [np.log(et[j] / et[j + 1]) / np.log(DTS[j] / DTS[j + 1]) for j in range(len(DTS) - 1)]
    print("     temporal order vs dt: " + ", ".join(f"{r:.2f}" for r in rt))
    print("\nwrote stokes_error_matrix.npz")
