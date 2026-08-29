"""
Orr-Sommerfeld growth at Re=7500 -- replicating Problem 2 of chandc/Python_SEM.

Plane Poiseuille, U = 1 - y^2 on y in [-1,1], periodic in x over [0, 2pi] (alpha = 1), no-slip
walls, sustained by the body force f_x = 2 nu that the base flow requires. A tiny perturbation
(1e-4) is seeded with the least-stable Orr-Sommerfeld eigenmode and must GROW at

    alpha * Im(c) = 0.00223497     (Streett, via Chan; reproduced independently in
                                    orr_sommerfeld.py to 7-9 digits)

WHY THIS IS HARD, and harder for this solver than for a spectral one. The growth rate is 2.2e-3
per unit time. Any numerical damping of comparable size does not merely add error -- it can flip
the sign of the answer and report a stable flow. Their own study shows N=8 giving 227% error
from under-resolution, with a spurious mode taking over around t ~ 20; only N=14 converges
(0.014%). A 2nd-order finite-difference scheme has to reach the same resolution the hard way.

`convection='central'` is mandatory here and is not a preference: test_energy_conservation.py
measures SOU removing ~10% of kinetic energy per turnover on a broadband field, which would
swamp a 2.2e-3 growth rate by orders of magnitude. Central's convective operator conserves
energy to round-off, so what remains is the time scheme and the projection.

MEASUREMENT. The base flow is x-independent and the perturbation is ~exp(i alpha x), so the two
separate exactly under a streamwise Fourier transform -- no second "base-flow-only" run is
needed and base-flow drift cannot contaminate the answer. The complex amplitude

    a(t) = <A(t), A(0)> / <A(0), A(0)>,     A = first streamwise Fourier mode of u'

gives the growth rate as d ln|a|/dt and the phase speed as -d arg(a)/dt / alpha.

RESOLUTION STUDY. Runs beyond ny=201 are NOT in the default suite -- 48x401 takes ~80 minutes --
so the numbers are recorded here.

    grid       dt      growth      err     phase      err
    48x97      0.05    0.001141   49.0%   0.246358   1.4%
    48x129     0.05    0.001595   28.6%   0.246243   1.5%
    48x201     0.05    0.001983   11.3%   0.246169   1.5%
    48x201     0.025   0.001956   12.5%   0.247535   0.94%
    48x401     0.05    0.002209    1.2%   0.246138   1.50%
    48x401     0.1     0.002195    1.8%   0.243419   2.59%
    reference          0.002235           0.249892

THE TWO ERROR SOURCES SPLIT CLEANLY BY VARIABLE. The GROWTH RATE is spatial-limited: refining
ny 201->401 gained a factor 9.4 while halving dt at ny=201 gained nothing. The PHASE SPEED is
temporal-limited: it sat at 1.4-1.5% across ny = 97->401 regardless of grid, and moved only with
dt (2.59% / 1.50% / 0.94% at dt = 0.1 / 0.05 / 0.025). That is why the phase error looked
stubbornly flat under grid refinement -- it was never the grid.

WHAT IS NOT SETTLED. The implied convergence order from 201->401 is 3.28, and stays ~3.4 using
the spatial-limited value at each grid. The scheme's design order is 2, so an implied 3.3 is NOT
superconvergence: partial cancellation between the spatial error (over-damping) and the temporal
error (under-damping) is still flattering the finest point. The sign of the dt-sensitivity FLIPS
between the grids -- at ny=201 refining dt makes the answer WORSE (11.3% -> 12.5%), at ny=401 it
makes it better (1.8% -> 1.2%) -- which is the signature of two errors of opposite sign crossing
somewhere between them. Settling it needs ny=801, which is not affordable here.

So the defensible claims are: the disturbance GROWS, which is the correct sign of stability and
the substantive result; the growth rate converges monotonically toward Streett's value; and
48x401 reaches 1.2% -- with that last error smaller than a purely second-order extrapolation
would predict, for a reason that is understood but not eliminated.
"""
import sys, io, contextlib, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from piso_numpy_3d import PISOSolver
from phase1_grid_metrics import compute_numerical_metrics
from orr_sommerfeld import least_stable

RE, ALPHA, AMP = 7500.0, 1.0, 1e-4
NU = 1.0 / RE
G_REF, C_REF = 0.00223497, 0.24989154
results = []
def check(name, good, detail):
    results.append(good)
    print(f"  [{'PASS' if good else 'FAIL'}] {name}: {detail}")


def build(nx, ny, dt, N_eig=120):
    Lx = 2 * np.pi / ALPHA
    nz = 4
    s = PISOSolver((nx, ny, nz), warp=1e-9, nu=NU, dt=dt, corrector_steps=2,
                   periodic=(True, False, True), scheme="rotational", time_scheme="bdf2",
                   convection="central", boundary_flux_mode="impermeable",
                   pressure_coef="rowsum", pressure_tol=1e-13)
    xi = np.arange(nx) / nx
    eta = np.linspace(0, 1, ny)
    ze = np.arange(nz) / nz
    XI, ETA, ZE = np.meshgrid(xi, eta, ze, indexing="ij")
    s.x, s.y, s.z = Lx * XI, -1.0 + 2.0 * ETA, ZE
    s.h = (1.0 / nx, 1.0 / (ny - 1), 1.0 / nz)
    s.J, s.metrics = compute_numerical_metrics(s.x, s.y, s.z, *s.h, periodic=s.per,
                                               period=(Lx, 1.0, 1.0))

    # base flow, and the body force that sustains it: nu U'' = -2 nu, so f_x = 2 nu
    U = 1.0 - s.y ** 2
    s.velocity_source = [np.full_like(s.y, 2.0 * NU), np.zeros_like(s.y), np.zeros_like(s.y)]

    # seed the least-stable OS eigenmode
    c, phi, yc = least_stable(RE, ALPHA, N_eig)
    from scipy.interpolate import BarycentricInterpolator
    dphi = np.gradient(phi, yc)            # yc is Chebyshev-spaced; refined below by the solver
    Dp = BarycentricInterpolator(yc, phi)
    # analytic derivative via the Chebyshev matrix is cleaner than np.gradient on uneven nodes
    from orr_sommerfeld import cheb
    D, _ = cheb(len(yc) - 1)
    Dphi = BarycentricInterpolator(yc, D @ phi)

    ph = np.exp(1j * ALPHA * s.x)
    up = np.real(Dphi(s.y) * ph)
    vp = np.real(-1j * ALPHA * Dp(s.y) * ph)
    scale = AMP / max(np.abs(up).max(), np.abs(vp).max())
    s.u = U + scale * up
    s.v = scale * vp
    s.w[:] = 0.0
    s.u[:, 0, :] = 0.0; s.u[:, -1, :] = 0.0
    s.v[:, 0, :] = 0.0; s.v[:, -1, :] = 0.0
    s.u_bc[:] = 0.0; s.v_bc[:] = 0.0; s.w_bc[:] = 0.0
    return s, c


def fourier_amp(s):
    """First streamwise Fourier mode of the perturbation, as a complex field over (y,z)."""
    up = s.u - s.u.mean(axis=0, keepdims=True)
    vp = s.v - s.v.mean(axis=0, keepdims=True)
    e = np.exp(-1j * ALPHA * s.x)
    return (up * e).mean(axis=0), (vp * e).mean(axis=0)


def run(nx, ny, dt, T=100.0, sample=40):
    s, c_exact = build(nx, ny, dt)
    A0 = fourier_amp(s)
    n = int(round(T / dt))
    every = max(1, n // sample)
    ts, aa = [0.0], [1.0 + 0j]
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(n):
            s.step()
            if (i + 1) % every == 0:
                A = fourier_amp(s)
                num = sum(np.vdot(A0[k], A[k]) for k in (0, 1))
                den = sum(np.vdot(A0[k], A0[k]) for k in (0, 1))
                ts.append((i + 1) * dt); aa.append(num / den)
    return np.array(ts), np.array(aa), time.time() - t0, c_exact


def fit(ts, aa, lo, hi):
    m = (ts >= lo) & (ts <= hi)
    g = np.polyfit(ts[m], np.log(np.abs(aa[m])), 1)[0]
    ph = np.unwrap(np.angle(aa[m]))
    cr = -np.polyfit(ts[m], ph, 1)[0] / ALPHA
    return g, cr


if __name__ == "__main__":
    print(__doc__.strip().split("\n\n")[0])
    c_ex, _, _ = least_stable(RE, ALPHA, 120)
    print(f"\n  eigensolver reference: growth {c_ex.imag*ALPHA:.9f}  phase {c_ex.real:.9f}")
    print(f"  Streett / Chan:        growth {G_REF:.9f}  phase {C_REF:.9f}")

    print(f"\n  {'nx x ny':>10} {'dt':>6} {'growth':>12} {'err %':>9} "
          f"{'phase c_r':>11} {'err %':>8} {'wall':>8}")
    got = {}
    for nx, ny, dt in ((48, 129, 0.05), (48, 201, 0.05)):
        ts, aa, wall, _ = run(nx, ny, dt)
        g, cr = fit(ts, aa, 20.0, 100.0)
        got[(nx, ny)] = (g, cr)
        print(f"  {f'{nx}x{ny}':>10} {dt:6.3f} {g:12.6f} {abs(g/G_REF-1)*100:8.1f}% "
              f"{cr:11.6f} {abs(cr/C_REF-1)*100:7.2f}% {wall:7.0f}s", flush=True)

    g, cr = got[(48, 201)]
    check("the disturbance GROWS (correct sign of stability)", g > 0,
          f"growth {g:+.6f} vs reference {G_REF:+.6f}")
    check("phase speed within 5%", abs(cr / C_REF - 1) < 0.05,
          f"c_r = {cr:.6f} vs {C_REF:.6f} ({abs(cr/C_REF-1)*100:.2f}%)")

    n_pass = sum(results)
    print(f"\n{'='*78}\n  {n_pass}/{len(results)} checks passed\n{'='*78}")
    sys.exit(0 if n_pass == len(results) else 1)
