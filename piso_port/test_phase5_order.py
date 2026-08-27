"""
Temporal order of the projection schemes.

Two settings, because they probe different things:

  PERIODIC  -- no walls, so the spurious dp/dn = 0 that limits a projection scheme never
               arises. This isolates the TIME INTEGRATOR: Backward Euler should give 1st
               order and BDF2 2nd order, for every scheme variant. The rotational term is
               expected to make no difference here, and that is the point -- it is a
               boundary-condition fix, not a time-integration fix.

  WALLS     -- Dirichlet velocity on all six faces. Here the projection's pressure boundary
               condition bites, and this is where 'rotational' should beat 'chorin'.

dt is swept at FIXED grid, so the spatial error is a constant floor; the sweep is chosen to
stay above it.
"""
import sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from piso_numpy_3d import PISOSolver

K = 2 * np.pi
def tg(x, y, z, t, nu):
    F = np.exp(-2 * K * K * nu * t)
    return (-np.cos(K*x)*np.sin(K*y)*F, np.sin(K*x)*np.cos(K*y)*F, np.zeros_like(x))

def run(n, dt, t_end, nu, scheme, time_scheme, periodic, ref=None):
    s = PISOSolver(n, warp=1e-9, nu=nu, dt=dt, corrector_steps=2, periodic=periodic,
                   scheme=scheme, time_scheme=time_scheme,
                   boundary_flux_mode="from_velocity", pressure_tol=1e-11)
    u0, v0, w0 = tg(s.x, s.y, s.z, 0.0, nu)
    s.u, s.v, s.w = u0.copy(), v0.copy(), w0.copy()
    nsteps = int(round(t_end / dt))
    for it in range(nsteps):
        ue, ve, we = tg(s.x, s.y, s.z, (it + 1) * dt, nu)
        s.u_bc, s.v_bc, s.w_bc = ue, ve, we
        s.step()
    if ref is None:
        return s.u.copy()
    I = (slice(None),)*3 if periodic else (slice(2, -2),)*3
    return np.sqrt(((s.u - ref)[I]**2).mean())

def order(errs, dts):
    return [np.log(errs[i]/errs[i+1]) / np.log(dts[i]/dts[i+1]) for i in range(len(errs)-1)]

results = []
def check(name, ok, detail):
    results.append((name, ok)); print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

# Error is measured against a NUMERICAL reference run on the SAME grid with a very small
# dt, not against the analytic solution. That matters: the spatial discretisation error is
# identical in both runs and cancels exactly, leaving only the temporal error. Measuring
# against the analytic solution instead mixes the two, and the spatial floor (1.6e-04 at
# n=32) swamps the temporal term long before the order can be read off.
N, NU, T_END = 24, 0.01, 0.04
DTS = [T_END/8, T_END/16, T_END/32, T_END/64]
DT_REF = T_END / 512

print("\nPERIODIC (all three directions) -- isolates the time integrator")
# Expected orders. Note chorin is 1st order with BDF2 TOO -- the non-incremental splitting
# error is O(dt) and no time integrator repairs it. (I initially expected 2 here; that was
# wrong, and the measurement is what corrected it.)
CASES = [("chorin", "be", 1.0), ("chorin", "bdf2", 1.0),
         ("rotational", "be", None), ("rotational", "bdf2", 2.0)]
for scheme, ts, want in CASES:
    ref = run(N, DT_REF, T_END, NU, scheme, ts, True)
    e = [run(N, dt, T_END, NU, scheme, ts, True, ref) for dt in DTS]
    p = order(e, DTS)
    # judge on the COARSEST interval: the finest dt sits only 8x from the reference dt, so
    # the reference's own temporal error contaminates the last estimate and inflates it.
    if want is None:
        ok, label = p[0] > 0.9, "expect >1"
    else:
        ok, label = abs(p[0] - want) < 0.35, f"expect ~{want:.0f}"
    check(f"{scheme:11s} {ts:4s} ({label})", ok,
          "errs " + "  ".join(f"{v:.2e}" for v in e) +
          "  orders " + ", ".join(f"{v:.2f}" for v in p))

print("\nWALLS (Dirichlet on all six faces) -- where the rotational term should matter")
wall = {}
for scheme in ("chorin", "incremental", "rotational"):
    ref = run(N, DT_REF, T_END, NU, scheme, "bdf2", False)
    e = [run(N, dt, T_END, NU, scheme, "bdf2", False, ref) for dt in DTS]
    p = order(e, DTS)
    wall[scheme] = (e, p)
    print(f"    {scheme:12s} bdf2  errs " + "  ".join(f"{v:.2e}" for v in e) +
          "  orders " + ", ".join(f"{v:.2f}" for v in p))
check("rotational beats chorin at the finest dt",
      wall["rotational"][0][-1] < wall["chorin"][0][-1],
      f"rotational {wall['rotational'][0][-1]:.2e} vs chorin {wall['chorin'][0][-1]:.2e}")

n_pass = sum(1 for _, ok in results if ok)
print(f"\n{'='*62}\n  {n_pass}/{len(results)} checks passed")
if n_pass != len(results):
    print("  FAILED: " + ", ".join(nm for nm, ok in results if not ok))
print('='*62)
sys.exit(0 if n_pass == len(results) else 1)
