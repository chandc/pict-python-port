"""
Plane Poiseuille flow -- the wall-bounded validation this port was missing.

Ported from PICT's own validation (tests/validations.py:474), which drives a channel with a
constant body force and compares against the analytic profile

    u(y) = G / (2 nu) * y (L - y)

Walls in y, periodic in x and z. For fully developed parallel flow the advective term vanishes
identically, so at steady state this reduces to  nu u'' + G = 0 -- a pure wall-bounded problem.
That is exactly where this port is weakest: the prescribed boundary-face flux uses a half-cell
(1st-order) stencil, so wall-bounded cases do not reach 2nd order. This puts a number on it.

TWO cases, because the classic one cannot measure an order:

  Runs `chorin`: the incremental and rotational projections diverge on wall-bounded domains
  in this port. Both cases are steady, so the 1st-order-in-time splitting costs nothing here.

  A. constant forcing -> the exact solution is a PARABOLA, and a central second difference is
     exact on quadratics. The interior error is therefore zero by construction and anything
     left is purely the wall treatment. It is a sharp defect detector, not a rate measurement.

  B. forcing  nu pi^2 sin(pi y)  -> exact solution sin(pi y), which is NOT polynomial, so the
     interior discretisation contributes its own O(h^2) and the observed rate is meaningful.
"""
import sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from src.piso_numpy_3d import PISOSolver

NU, NSPAN, DT = 0.1, 4, 0.5
results = []
def check(name, good, detail):
    results.append(good)
    print(f"  [{'PASS' if good else 'FAIL'}] {name}: {detail}")


def run(ny, case, max_steps=3000, tol=1e-10):
    s = PISOSolver((NSPAN, ny, NSPAN), warp=1e-9, nu=NU, dt=DT, corrector_steps=2,
                   periodic=(True, False, True), scheme="chorin", time_scheme="be",
                   boundary_flux_mode="impermeable", pressure_tol=1e-11)
    y = s.y
    if case == "parabola":
        G = 8 * NU                       # so that u_max = G L^2 / (8 nu) = 1
        force = np.full_like(y, G)
        exact = G / (2 * NU) * y * (1.0 - y)
    else:
        force = NU * np.pi**2 * np.sin(np.pi * y)
        exact = np.sin(np.pi * y)
    s.velocity_source = [force, np.zeros_like(y), np.zeros_like(y)]
    prev = None
    for it in range(max_steps):
        s.step()
        if prev is not None and np.abs(s.u - prev).max() < tol:
            break
        prev = s.u.copy()
    err = np.sqrt(((s.u - exact) ** 2).mean())
    return err, float(np.abs(s.u).max()), float(np.abs(exact).max()), it + 1


print(f"\nPlane Poiseuille flow: walls in y, periodic in x/z, nu={NU}, {NSPAN} spanwise cells")

print("\nA. Constant forcing -- exact solution is a parabola, so the interior scheme is EXACT")
print("   and any error is purely the wall treatment.")
for ny in (8, 16, 32):
    e, umax, uex, its = run(ny, "parabola")
    print(f"   ny={ny:3d}   L2 error {e:.3e}   u_max {umax:.6f} (exact {uex:.6f})   {its} steps")
    if ny == 32:
        check("A: parabola reproduced to near machine precision", e < 1e-8,
              f"L2 error {e:.2e} at ny=64")

print("\nB. Forcing nu*pi^2*sin(pi y) -- exact solution sin(pi y), NOT polynomial, so this")
print("   measures the genuine wall-bounded spatial order.")
errs, ns = [], (8, 16, 32)
for ny in ns:
    e, umax, uex, its = run(ny, "sine")
    errs.append(e)
    print(f"   ny={ny:3d}   L2 error {e:.3e}   u_max {umax:.6f} (exact {uex:.6f})   {its} steps")
rates = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
print(f"\n   convergence rates: " + ", ".join(f"{r:.2f}" for r in rates))
check("B: wall-bounded order", min(rates) > 1.7,
      f"rates {', '.join(f'{r:.2f}' for r in rates)} — 2nd order at walls"
      if min(rates) > 1.7 else
      f"rates {', '.join(f'{r:.2f}' for r in rates)} — below 2nd order, as the half-cell "
      f"boundary stencil predicts")

print("\n" + ("Poiseuille validation passed" if all(results) else "see results above"))
sys.exit(0 if all(results) else 1)
