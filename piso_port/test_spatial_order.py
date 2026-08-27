"""
Spatial order of the FULL PISO solver.

Everything measured so far establishes the spatial order of the individual OPERATORS
(phases 1-4) and the TEMPORAL order of the assembled solver (test_phase5_order.py). What was
never isolated is the spatial order of the complete solver, because the walled Taylor-Green
test refined dt together with h and so measured the two errors mixed.

This runs fully PERIODIC (no walls, so no projection boundary layer) with rotational + BDF2
(2nd order in time) and a dt small enough that the temporal error sits well below the spatial
error at every resolution -- so the rate that comes out is the spatial one.
"""
import sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from piso_numpy_3d import PISOSolver

K = 2 * np.pi
def tg(x, y, z, t, nu):
    F = np.exp(-2 * K * K * nu * t)
    return (-np.cos(K*x)*np.sin(K*y)*F, np.sin(K*x)*np.cos(K*y)*F, np.zeros_like(x))

def run(n, warp, dt, t_end, nu):
    s = PISOSolver(n, warp=max(warp, 1e-9), nu=nu, dt=dt, corrector_steps=2, periodic=True,
                   scheme="rotational", time_scheme="bdf2", pressure_tol=1e-12)
    u0, v0, w0 = tg(s.x, s.y, s.z, 0.0, nu)
    s.u, s.v, s.w = u0.copy(), v0.copy(), w0.copy()
    for _ in range(int(round(t_end / dt))):
        s.step()
    ue, ve, we = tg(s.x, s.y, s.z, t_end, nu)
    return np.sqrt(((s.u - ue)**2).mean())

NU, T_END, DT = 0.01, 0.02, 0.0003125     # 64 steps; temporal error is far below spatial here
NS = [16, 24, 32, 48]
results = []
for warp in (0.0, 0.05):
    errs = [run(n, warp, DT, T_END, NU) for n in NS]
    r = [np.log(errs[i]/errs[i+1]) / np.log(NS[i+1]/NS[i]) for i in range(len(errs)-1)]
    # Bar: at least 2nd order, monotone, and not erratic. The UPPER bound is generous (4.5)
    # because the Cartesian case genuinely converges FASTER than 2nd order (~3.1-3.3) -- all
    # metrics are exactly constant on a uniform grid, so the metric-induced error terms vanish
    # and a subdominant term dies out. Verified not to be a grid-alignment artifact: shifting
    # the (still exact) Taylor-Green solution by half a cell, or by an irrational offset,
    # leaves the rates unchanged at 3.13 / 3.12. Converging faster than required is not a
    # failure, but erratic or non-monotone rates still are.
    ok = np.all(np.diff(errs) < 0) and all(1.7 < v < 4.5 for v in r)
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] warp={warp:.2f}: errs " +
          "  ".join(f"{e:.3e}" for e in errs) + "   rates " + ", ".join(f"{v:.2f}" for v in r))

print(f"\n{'='*60}\n  {sum(results)}/{len(results)} passed  (2nd-order spatial bar: 1.7-4.5)\n{'='*60}")
sys.exit(0 if all(results) else 1)
