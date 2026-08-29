"""
Wall-bounded 2D Stokes: decay rate of the least-damped mode in a no-slip channel.

Reference configuration: channel y in [-1,1] (half-width 1), no-slip both walls, streamwise
wavenumber alpha = 1 (so the box is 2*pi long and periodic), nu = 1. The least-damped
eigenvalue of the Stokes operator there is

    sigma = -9.313739

which is the number this test has to reproduce. Unlike the doubly-periodic case, there is no
closed form: the walls make it the root of a transcendental condition, so the reference comes
from a Chebyshev eigensolve of

    sigma (D^2 - alpha^2) phi = nu (D^2 - alpha^2)^2 phi,   phi = phi' = 0 at y = +/-1

using Trefethen's clamped D4 construction. phi = phi' = 0 IS no-slip: with the streamfunction
psi = phi(y) cos(alpha x), u = phi' cos and v = alpha phi sin, so both velocity components
vanish at the walls exactly when phi and phi' do. (A discretisation that imposes only phi = 0
returns alpha^2 + n^2 pi^2 instead -- the simply-supported spectrum, a different problem.)

This is a stiffer test than the periodic one: it exercises the wall boundary treatment, and the
eigenvalue is set by the wall conditions rather than by a Fourier symmetry the scheme satisfies
trivially.
"""
import sys, warnings, io, contextlib
import numpy as np
warnings.filterwarnings("ignore")
from scipy.linalg import eig
from scipy.interpolate import BarycentricInterpolator
from src.piso_numpy_3d import PISOSolver
from src.phase1_grid_metrics import compute_numerical_metrics

ALPHA, NU, AMP = 1.0, 1.0, 1e-3
SIGMA_REF = -9.313739

results = []
def check(name, good, detail):
    results.append(good)
    print(f"  [{'PASS' if good else 'FAIL'}] {name}: {detail}")


def cheb(N):
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2., np.ones(N - 1), 2.]) * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    D = np.outer(c, 1. / c) / ((X - X.T) + np.eye(N + 1))
    return D - np.diag(D.sum(axis=1)), x


def stokes_mode(alpha=ALPHA, nu=NU, N=80):
    """Least-damped clamped Stokes mode: returns sigma and interpolators for phi and phi'."""
    D, xc = cheb(N)
    D2 = D @ D
    S = np.diag(np.hstack([0., 1. / (1. - xc[1:N] ** 2), 0.]))
    D4 = (np.diag(1. - xc ** 2) @ np.linalg.matrix_power(D, 4)
          - 8 * np.diag(xc) @ np.linalg.matrix_power(D, 3) - 12 * D2) @ S
    I = np.eye(N - 1)
    A = nu * (D4[1:N, 1:N] - 2 * alpha ** 2 * D2[1:N, 1:N] + alpha ** 4 * I)
    B = D2[1:N, 1:N] - alpha ** 2 * I
    w, V = eig(A, B)
    w = np.real(w)
    # The generalised eigenvalue IS sigma (A phi = sigma B phi), and every finite one is
    # negative -- Stokes flow only decays. The least-damped mode is the one closest to zero.
    ok = np.isfinite(w) & (w < 0)
    idx = np.where(ok)[0]
    j = idx[np.argmin(np.abs(w[idx]))]
    sigma = w[j]
    phi = np.zeros(N + 1)
    phi[1:N] = np.real(V[:, j])
    phi /= np.abs(phi).max()
    dphi = D @ phi
    return sigma, BarycentricInterpolator(xc, phi), BarycentricInterpolator(xc, dphi)


def run(ny, dt, nsteps, alpha=ALPHA, nu=NU, nx=None, scheme="rotational",
        time_scheme="bdf2", window=None):
    Lx = 2 * np.pi / alpha
    # Refine x with y so the order study measures grid convergence rather than a fixed
    # streamwise error. One full wavelength spans the box, so 3/4 of the wall-normal count
    # resolves it comparably.
    nx = nx if nx is not None else max(12, int(round((ny - 1) * 0.75)))
    nz = 4
    s = PISOSolver((nx, ny, nz), warp=1e-9, nu=nu, dt=dt, corrector_steps=2,
                   periodic=(True, False, True), scheme=scheme, time_scheme=time_scheme,
                   boundary_flux_mode="impermeable", pressure_coef="rowsum",
                   pressure_tol=1e-12)
    # physical box: x in [0,Lx) periodic, y in [-1,1] no-slip, z thin periodic
    xi = np.arange(nx) / nx
    eta = np.linspace(0, 1, ny)
    zeta = np.arange(nz) / nz
    XI, ETA, ZETA = np.meshgrid(xi, eta, zeta, indexing="ij")
    s.x, s.y, s.z = Lx * XI, -1.0 + 2.0 * ETA, ZETA
    s.h = (1.0 / nx, 1.0 / (ny - 1), 1.0 / nz)
    s.J, s.metrics = compute_numerical_metrics(s.x, s.y, s.z, *s.h, periodic=s.per,
                                               period=(Lx, 1.0, 1.0))
    sigma_ex, phi, dphi = stokes_mode(alpha, nu)
    s.u = AMP * dphi(s.y) * np.cos(alpha * s.x)
    s.v = AMP * alpha * phi(s.y) * np.sin(alpha * s.x)
    s.w[:] = 0.0
    # `window` measures the rate over [lo, hi] instead of from t=0. That matters: the initial
    # field is an eigenmode of the CONTINUOUS operator, not of the discrete one, so there is a
    # startup transient. Measuring across it reports the transient, not the eigenvalue --
    # measured local rates at dt=1e-4 were -9.32155 over [0,0.02] settling to a flat -9.32266
    # from t=0.02 onward. A Richardson study run inside the transient returned order 0.04 for
    # a scheme that is genuinely ~1.7 here.
    i0 = 0 if window is None else int(round(window[0] / dt))
    n = nsteps if window is None else int(round(window[1] / dt))
    E0 = 0.5 * np.mean(s.u ** 2 + s.v ** 2)
    T = 0.0 if window is None else (window[1] - window[0])
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(n):
            s.step()
            if window is None:
                T += s.dt
            elif i + 1 == i0:
                E0 = 0.5 * np.mean(s.u ** 2 + s.v ** 2)
    E1 = 0.5 * np.mean(s.u ** 2 + s.v ** 2)
    return np.log(E1 / E0) / (2 * T), sigma_ex, s


if __name__ == "__main__":
    print(__doc__.strip().split("\n\n")[0])

    # ---------------------------------------------------------- 1. the reference itself
    print(f"\n1. Reference eigenvalue (Chebyshev, clamped) vs the quoted {SIGMA_REF}")
    for N in (40, 60, 80, 120):
        sg, _, _ = stokes_mode(N=N)
        print(f"   N={N:4d}   sigma = {sg:.8f}   |diff| = {abs(sg - SIGMA_REF):.2e}")
    sg80 = stokes_mode(N=80)[0]
    check("Chebyshev eigensolve reproduces the quoted decay rate",
          abs(sg80 - SIGMA_REF) < 1e-5, f"sigma = {sg80:.6f} vs {SIGMA_REF}")

    # ---------------------------------------------------------- 2. PISO vs the eigenvalue
    print("\n2. PISO decay rate (alpha=1, nu=1, no-slip walls at y=+/-1)")
    print(f"   {'ny':>5} {'nx':>5} {'dt':>8} {'sigma':>12} {'rel err':>10}")
    errs, NYS, DT = [], (25, 33, 49, 65), 1e-4
    for ny in NYS:
        sg, sx, sv = run(ny, dt=DT, nsteps=500)
        e = abs(sg / sx - 1)
        errs.append(e)
        print(f"   {ny:5d} {sv.shape[0]:5d} {DT:8.0e} {sg:12.6f} {e:10.2e}")
    check("PISO reproduces the wall-bounded Stokes decay rate",
          errs[-1] < 0.01, f"finest-grid relative error {errs[-1]:.2e} vs sigma = {SIGMA_REF}")

    rates = [np.log(errs[i] / errs[i + 1]) / np.log(NYS[i + 1] / NYS[i]) for i in range(len(NYS) - 1)]
    print("   spatial order: " + ", ".join(f"{r:.2f}" for r in rates))
    check("decay-rate error is 2nd order in space", rates[-1] > 1.7,
          f"rates {', '.join(f'{r:.2f}' for r in rates)}")

    # ---------------------------------------------------------- 3. temporal order
    print("\n3. Temporal order of the decay rate (ny=49, settled window [0.05, 0.10])")
    for scheme, ts, want, floor in (("chorin", "be", 1.0, 0.7), ("rotational", "bdf2", 2.0, 1.5)):
        sg = [run(49, dt=d, nsteps=0, scheme=scheme, time_scheme=ts, window=(0.05, 0.10))[0]
              for d in (2e-4, 1e-4, 5e-5)]
        d1, d2 = abs(sg[0] - sg[1]), abs(sg[1] - sg[2])
        order = np.log2(d1 / d2) if d2 > 0 else np.inf
        print(f"   {scheme:10s}/{ts:5s}  sigma = {sg[0]:.6f}, {sg[1]:.6f}, {sg[2]:.6f}"
              f"   order {order:.2f}")
        check(f"{scheme}/{ts} decay rate converges above order {floor}",
              order > floor, f"observed {order:.2f} (design order {want:.0f})")

    # BDF2 measures ~1.68 on the triple used above and 2.01 in the doubly-periodic case. That
    # gap is NOT an order loss: extending the sweep to finer dt gives ratios 1.74 / 3.20 / 4.00
    # for the triples (4e-4,2e-4,1e-4) / (2e-4,1e-4,5e-5) / (1e-4,5e-5,2.5e-5), i.e. orders
    # 0.80 / 1.68 / 2.00. The scheme IS second order in time with walls; the triple here is
    # simply pre-asymptotic, which is why the gate is a floor rather than an equality.
    # An earlier version of this comment attributed the shortfall to the O(dt^3/2) near-wall
    # splitting error of rotational projection. That explanation was plausible and wrong -- see
    # test_chan_channel.py, where an independent spectral-element solver reports 1.94-1.99 on
    # this exact problem and prompted the re-check.

    n_pass = sum(results)
    print(f"\n{'='*74}\n  {n_pass}/{len(results)} checks passed\n{'='*74}")
    sys.exit(0 if n_pass == len(results) else 1)
