"""
Square duct, implicit vs deferred cross terms: ACCURACY against the exact series, and cost.

Why this test needed a new grid. test_duct.py runs at warp 1e-9 -- effectively Cartesian, where
the cross terms vanish identically and implicit_cross is a no-op. But the solver's stock warp
(make_grid) displaces the WALLS: for periodic=(True,False,False) it sets

    y = eta + A sin(2 pi xi) sin(2 pi zeta)

which is nonzero at eta = 0, so the duct stops being square and the Fourier series stops being
the right answer. Warping that way would compare both solvers against a solution neither is
solving for.

So the warp used here is built to vanish ON the walls while still being non-orthogonal:

    x = xi   + A sin(pi eta) sin(pi zeta)
    y = eta  + A sin(pi eta) sin(2 pi xi)
    z = zeta + A sin(pi zeta) sin(2 pi xi)

The y-displacement carries sin(pi eta), which is zero at eta = 0 and 1, so both y-walls stay
exactly at y = 0 and y = 1 (likewise z). The physical cross-section is therefore still the unit
square at every streamwise station -- only the node distribution inside it moves -- so u(y,z)
from the series remains the exact solution. Meanwhile dy/dxi is nonzero (up to 0.8 at A = 0.2),
so g^12 and g^13 are genuinely nonzero and the cross terms are real work.

Verified before use: walls at 0 and 1 to 0.0e+00, GCL ~1e-13, min(J) > 0 through A = 0.20
(this family does NOT tangle at 0.18 the way the stock warp does).
"""
import sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from src.piso_numpy_3d import PISOSolver
from src.phase1_grid_metrics import compute_numerical_metrics
from test_duct import duct_exact, NU, G, NX

results = []
def check(name, good, detail):
    results.append(good)
    print(f"  [{'PASS' if good else 'FAIL'}] {name}: {detail}")


def duct_grid(nx, n, A):
    xi = np.arange(nx) / nx                       # periodic streamwise
    eta = np.linspace(0, 1, n)
    zeta = np.linspace(0, 1, n)
    XI, ETA, ZETA = np.meshgrid(xi, eta, zeta, indexing="ij")
    # No x-displacement. An earlier version had x = xi + A sin(pi eta) sin(pi zeta), which
    # sheared the whole block and made the mesh look nothing like a duct. Measured, it bought
    # essentially no non-orthogonality (|g12|/|g11| 0.334 vs 0.311 at A=0.05) while degrading
    # the worst cell angle (31.0 deg vs 38.8 deg at A=0.20) and tangling at A=0.25 where this
    # form is still valid. Dropping it is strictly better on every count.
    x = XI
    y = ETA + A * np.sin(np.pi * ETA) * np.sin(2 * np.pi * XI)
    z = ZETA + A * np.sin(np.pi * ZETA) * np.sin(2 * np.pi * XI)
    return x, y, z, 1.0 / nx, 1.0 / (n - 1), 1.0 / (n - 1)


def run(n, A, implicit, dt=0.5, max_steps=3000, tol=1e-10, nx=None):
    # Refine the STREAMWISE direction along with the cross-section. The warp varies as
    # sin(2 pi xi), so pinning nx at 4 (as the stock duct test does, where the grid is
    # Cartesian and x-resolution is irrelevant) leaves the streamwise metric variation badly
    # under-resolved and puts a floor under the error that refining y,z cannot lift. Measured
    # at n=16, A=0.05: L2 = 6.35e-03 / 2.48e-03 / 1.62e-03 / 1.51e-03 for nx = 4 / 8 / 16 / 32,
    # converging onto the Cartesian n=16 value of 1.43e-03. Refining nx with n is what makes
    # this a grid-convergence test rather than a measurement of the nx=4 error.
    nx = n if nx is None else nx
    s = PISOSolver((nx, n, n), warp=1e-9, nu=NU, dt=dt, corrector_steps=2,
                   periodic=(True, False, False), scheme="chorin", time_scheme="be",
                   boundary_flux_mode="impermeable", pressure_tol=1e-11,
                   implicit_cross=implicit)
    # swap in the wall-preserving grid and rebuild the metrics from it
    x, y, z, dxi, deta, dzeta = duct_grid(nx, n, A)
    s.x, s.y, s.z = x, y, z
    s.h = (dxi, deta, dzeta)
    s.J, s.metrics = compute_numerical_metrics(x, y, z, dxi, deta, dzeta, periodic=s.per)
    assert s.J.min() > 0, f"tangled grid at A={A}, n={n}"

    exact = duct_exact(s.y, s.z)
    s.velocity_source = [np.full_like(s.y, G), np.zeros_like(s.y), np.zeros_like(s.y)]

    its, prev = 0, None
    t0 = time.time()
    for step in range(max_steps):
        s.step()
        its += s._implicit_its if implicit else s._dc_sweeps
        if prev is not None and np.abs(s.u - prev).max() < tol:
            break
        prev = s.u.copy()
    wall = time.time() - t0
    l2 = float(np.sqrt(((s.u - exact) ** 2).mean()))
    return dict(l2=l2, t=wall, its=its, steps=step + 1, u=s.u.copy(), umax=float(s.u.max()),
                uex=float(exact.max()))


if __name__ == "__main__":
    print(__doc__.strip().split("\n\n")[0])
    print("\nGrid sanity (the series is only the right answer if the walls do not move)")
    for A in (0.05, 0.10, 0.15):
        x, y, z, dxi, deta, dzeta = duct_grid(NX, 17, A)
        J, _ = compute_numerical_metrics(x, y, z, dxi, deta, dzeta, periodic=(True, False, False))
        wall_err = max(np.abs(y[:, 0, :]).max(), np.abs(y[:, -1, :] - 1).max(),
                       np.abs(z[:, :, 0]).max(), np.abs(z[:, :, -1] - 1).max())
        print(f"   A={A:4.2f}   min(J) {J.min():.3e}   max wall displacement {wall_err:.1e}")
        check(f"A={A:4.2f}: duct stays square and untangled", wall_err < 1e-14 and J.min() > 0,
              f"wall displacement {wall_err:.1e}, min(J) {J.min():.3e}")

    NS = (8, 16, 32)
    for A in (0.0, 0.05, 0.10):
        print(f"\nWarp A={A:.2f}   (L2 is against the exact Fourier series)")
        print(f"   {'n':>3}  {'deferred: L2':>14} {'time':>8} {'sweeps':>7}   "
              f"{'implicit: L2':>14} {'time':>8} {'Krylov':>7}   {'speed-up':>8}")
        errs = {False: [], True: []}
        for n in NS:
            r = {imp: run(n, A, imp) for imp in (False, True)}
            for imp in (False, True):
                errs[imp].append(r[imp]["l2"])
            rel = np.abs(r[True]["u"] - r[False]["u"]).max() / max(np.abs(r[False]["u"]).max(), 1e-30)
            print(f"   {n:3d}  {r[False]['l2']:14.4e} {r[False]['t']:7.2f}s {r[False]['its']:7d}   "
                  f"{r[True]['l2']:14.4e} {r[True]['t']:7.2f}s {r[True]['its']:7d}   "
                  f"{r[False]['t']/r[True]['t']:7.2f}x")
            check(f"A={A:.2f} n={n}: same answer", rel < 1e-6, f"max relative difference {rel:.2e}")
        for imp in (False, True):
            rates = [np.log2(errs[imp][i] / errs[imp][i + 1]) for i in range(len(NS) - 1)]
            label = "implicit" if imp else "deferred"
            print(f"      {label:8s} convergence rates: " + ", ".join(f"{r:.2f}" for r in rates))
            check(f"A={A:.2f} {label}: 2nd-order vs the series", min(rates) > 1.5,
                  f"rates {', '.join(f'{r:.2f}' for r in rates)}")

    n_pass = sum(results)
    print(f"\n{'='*78}\n  {n_pass}/{len(results)} checks passed\n{'='*78}")
    sys.exit(0 if n_pass == len(results) else 1)
