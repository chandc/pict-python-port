"""
Fully developed laminar flow in a square duct -- the Phase 5 validation the original plan
called for and that was deferred until periodic boundaries existed.

Four no-slip walls (y and z), periodic streamwise (x), constant body force G. At steady state
the advective term vanishes for parallel flow, so this reduces to a 2D Poisson problem on the
cross-section with an exact Fourier-series solution:

    u(y,z) = sum_{n odd} 4 G h^2 / (mu n^3 pi^3)
             * [ 1 - cosh(n pi (z - w/2)/h) / cosh(n pi w / (2h)) ] * sin(n pi y / h)

This is a considerably stronger wall test than plane Poiseuille: the cross-section is genuinely
2D, there are FOUR walls rather than two, and it has CORNERS, where the boundary treatment of
two walls meets. Plane Poiseuille cannot exercise any of that.
"""
import sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from piso_numpy_3d import PISOSolver

NU, G, NX, DT = 0.1, 1.0, 4, 0.5


def duct_exact(y, z, G=G, mu=NU, h=1.0, w=1.0, n_terms=200):
    """Series solution; truncated where the n^-3 tail is far below discretisation error."""
    u = np.zeros_like(y)
    for n in range(1, n_terms, 2):
        kn = n * np.pi / h
        a, b = kn * np.abs(z - w / 2), kn * w / 2
        # cosh(a)/cosh(b) written so nothing overflows: both cosh terms exceed 1e308 for
        # n of a few hundred, and the naive form silently returns nan.
        ratio = (np.exp(a - b) + np.exp(-a - b)) / (1.0 + np.exp(-2 * b))
        u += (4 * G * h**2 / (mu * n**3 * np.pi**3)) * (1.0 - ratio) * np.sin(kn * y)
    return u


def run(n, max_steps=3000, tol=1e-10):
    s = PISOSolver((NX, n, n), warp=1e-9, nu=NU, dt=DT, corrector_steps=2,
                   periodic=(True, False, False), scheme="chorin", time_scheme="be",
                   boundary_flux_mode="impermeable", pressure_tol=1e-11)
    exact = duct_exact(s.y, s.z)
    s.velocity_source = [np.full_like(s.y, G), np.zeros_like(s.y), np.zeros_like(s.y)]
    prev = None
    for it in range(max_steps):
        s.step()
        if prev is not None and np.abs(s.u - prev).max() < tol:
            break
        prev = s.u.copy()
    return np.sqrt(((s.u - exact)**2).mean()), float(s.u.max()), float(exact.max()), it + 1


if __name__ == "__main__":
    # sanity-check the series against the PDE it solves, before trusting it as a reference
    n = 200
    yy, zz = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    ue = duct_exact(yy, zz)
    hh = 1.0 / (n - 1)
    lap = ((np.roll(ue, -1, 0) - 2*ue + np.roll(ue, 1, 0))
           + (np.roll(ue, -1, 1) - 2*ue + np.roll(ue, 1, 1))) / hh**2
    I = (slice(3, -3),) * 2
    core = (slice(40, -40),) * 2      # away from the corners, where the series converges slowly
    print(f"\nseries check:  max |nu*lap(u) + G| / G = {np.abs(NU*lap[core] + G).max()/G:.2e} "
          f"in the interior, {np.abs(NU*lap[I] + G).max()/G:.2e} near the corners")
    print("               (the corner value is the series' own slow convergence there, plus the")
    print("                finite-difference error used to check it -- not an error in the solver)")
    print(f"               u_max = {ue.max():.6f},  centre value {duct_exact(np.array(0.5), np.array(0.5)):.6f}")

    print(f"\nSquare duct: 4 no-slip walls, periodic streamwise, nu={NU}, G={G}")
    errs, ns = [], (8, 16, 32)
    for nn in ns:
        e, um, uex, its = run(nn)
        errs.append(e)
        print(f"   {nn:2d}x{nn:2d} cross-section   L2 error {e:.3e}   "
              f"u_max {um:.6f} (exact {uex:.6f})   {its} steps")
    rates = [np.log2(errs[i] / errs[i+1]) for i in range(len(errs) - 1)]
    print(f"\n   convergence rates: " + ", ".join(f"{r:.2f}" for r in rates))
    ok = min(rates) > 1.5
    print(f"\n  [{'PASS' if ok else 'FAIL'}] square duct vs the analytic series: "
          f"rates {', '.join(f'{r:.2f}' for r in rates)}")
    sys.exit(0 if ok else 1)
