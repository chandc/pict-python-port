"""
2D Stokes: does PISO reproduce the growth rate of a disturbance?

The unsteady Stokes equations

    du/dt = -grad p + nu lap u,     div u = 0

have an exact eigenvalue. Any divergence-free Fourier mode with wavevector k evolves as
exp(sigma t) with

    sigma = -nu |k|^2

so the "growth rate" is a hard analytic number, not a fitted curve. That makes this a sharper
test than a decay-curve comparison: a scheme can track a decaying solution roughly right while
getting its eigenvalue wrong, and the eigenvalue is what governs whether a disturbance in a real
calculation grows or dies.

Modes are built from a streamfunction so that div u = 0 holds to machine precision by
construction, and so oblique modes (k1 != k2) are available -- a diagonal-only test could pass
on a symmetry that an oblique mode would break:

    psi = sin(k1 x) sin(k2 y),   u = dpsi/dy,  v = -dpsi/dx,   sigma = -nu (k1^2 + k2^2)

STOKES, not Navier-Stokes: the amplitude is small (1e-4) so the convective term, which is
O(A^2) against the O(A) linear terms, is negligible. Checked explicitly below by re-running at
a 100x smaller amplitude -- if convection were contaminating the rate, sigma would move.
"""
import sys, warnings, io, contextlib
import numpy as np
warnings.filterwarnings("ignore")
from src.piso_numpy_3d import PISOSolver

NU, AMP = 0.01, 1e-4
results = []
def check(name, good, detail):
    results.append(good)
    print(f"  [{'PASS' if good else 'FAIL'}] {name}: {detail}")


def mode(x, y, m1, m2, amp=AMP):
    """
    Divergence-free 2D mode from psi = cos(k1 x) cos(k2 y), so u = dpsi/dy, v = -dpsi/dx.

    cos*cos, not sin*sin: the latter vanishes identically when m2 = 0, which silently hands the
    solver a zero field and makes the measured rate log(0/0). With cos*cos the m2 = 0 case is a
    genuine mode -- a pure shear layer, u = 0, v = k1 sin(k1 x) -- which is worth testing
    precisely because it is not diagonal.
    """
    k1, k2 = 2 * np.pi * m1, 2 * np.pi * m2
    u = -amp * k2 * np.cos(k1 * x) * np.sin(k2 * y)
    v = amp * k1 * np.sin(k1 * x) * np.cos(k2 * y)
    return u, v


def sigma_exact(m1, m2, nu=NU):
    return -nu * ((2 * np.pi * m1) ** 2 + (2 * np.pi * m2) ** 2)


def run(m1, m2, n=32, dt=0.002, nsteps=100, scheme="rotational", time_scheme="bdf2",
        amp=AMP, extra_modes=()):
    s = PISOSolver((n, n, 4), warp=1e-9, nu=NU, dt=dt, corrector_steps=2,
                   periodic=(True, True, True), scheme=scheme, time_scheme=time_scheme,
                   pressure_coef="rowsum", pressure_tol=1e-12)
    u, v = mode(s.x, s.y, m1, m2, amp)
    for (a, b) in extra_modes:
        du, dv = mode(s.x, s.y, a, b, amp)
        u, v = u + du, v + dv
    s.u, s.v = u, v
    s.w[:] = 0.0
    u0, v0 = s.u.copy(), s.v.copy()

    E0 = 0.5 * np.mean(s.u ** 2 + s.v ** 2)
    T = 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(nsteps):
            d = s.step()
            T += s.dt
    E1 = 0.5 * np.mean(s.u ** 2 + s.v ** 2)
    return dict(sigma=np.log(E1 / E0) / (2 * T), T=T, divF=d, s=s, u0=u0, v0=v0)


def project(s, u, v, m1, m2):
    """Amplitude of one mode in the field (u,v), by projection onto its analytic shape."""
    um, vm = mode(s.x, s.y, m1, m2, 1.0)
    return np.mean(u * um + v * vm) / np.mean(um * um + vm * vm)


if __name__ == "__main__":
    # ---------------------------------------------------------------- 1. dispersion relation
    print("1. Dispersion relation  sigma = -nu |k|^2   (n=48, dt=0.001, 100 steps)")
    print(f"   {'mode (m1,m2)':>14} {'|k|^2':>10} {'exact':>10} {'measured':>10} {'rel err':>10}")
    disp = []
    for m1, m2 in ((1, 0), (1, 1), (2, 1), (2, 2), (3, 1)):
        r = run(m1, m2, n=48, dt=0.001, nsteps=100)
        ex = sigma_exact(m1, m2)
        k2 = (2 * np.pi * m1) ** 2 + (2 * np.pi * m2) ** 2
        err = abs(r["sigma"] / ex - 1)
        disp.append((m1, m2, ex, r["sigma"], err))
        print(f"   {f'({m1},{m2})':>14} {k2:10.1f} {ex:10.4f} {r['sigma']:10.4f} {err:10.2e}")
    worst = max(d[4] for d in disp)
    check("growth rate matches -nu|k|^2 for every mode, oblique included",
          worst < 0.02, f"worst relative error {worst:.2e} over 5 modes incl. (1,0) and (3,1)")

    # ---------------------------------------------------------------- 2. it really is Stokes
    print("\n2. Amplitude independence (confirms the convective term is not contaminating sigma)")
    s_hi = run(2, 1, n=48, dt=0.001, nsteps=100, amp=1e-4)["sigma"]
    s_lo = run(2, 1, n=48, dt=0.001, nsteps=100, amp=1e-6)["sigma"]
    print(f"   amp 1e-4 -> sigma {s_hi:.6f}      amp 1e-6 -> sigma {s_lo:.6f}")
    # The bar is set against the discretisation error, not an arbitrary constant: convection is
    # negligible if changing the amplitude 100x moves sigma by far less than the error we are
    # trying to measure (~5e-3 at this resolution).
    d_amp = abs(s_hi / s_lo - 1)
    check("sigma is amplitude-independent (Stokes regime)", d_amp < 1e-4,
          f"relative change {d_amp:.2e} for a 100x amplitude change -- "
          f"{5e-3/max(d_amp,1e-30):.0f}x smaller than the discretisation error it sits inside")

    # ---------------------------------------------------------------- 3. spatial order
    print("\n3. Spatial convergence of the growth rate (dt=2e-4, temporal error suppressed)")
    # Refined to n=96. The rate approaches 2 from BELOW as the mode becomes well resolved
    # (measured 1.76 / 1.89 / 1.95 / 1.97 for n = 24->32->48->64->96), so a coarse study reads
    # as "not quite second order" when it is only pre-asymptotic. The (2,1) mode has just 12
    # points per wavelength in x at n=24. Gate on the FINEST rate plus the monotone trend
    # rather than on an average that the coarse end drags down.
    NS = (32, 48, 64, 96)
    errs = []
    for n in NS:
        r = run(2, 1, n=n, dt=2e-4, nsteps=100)
        e = abs(r["sigma"] - sigma_exact(2, 1))
        errs.append(e)
        print(f"   n={n:3d}   sigma {r['sigma']:9.5f}   |error| {e:.3e}")
    rates = [np.log(errs[i] / errs[i + 1]) / np.log(NS[i + 1] / NS[i]) for i in range(len(NS) - 1)]
    print("   observed order: " + ", ".join(f"{r:.2f}" for r in rates))
    monotone = all(rates[i] <= rates[i + 1] + 0.02 for i in range(len(rates) - 1))
    check("growth-rate error is 2nd order in space", rates[-1] > 1.9 and monotone,
          f"rates {', '.join(f'{r:.2f}' for r in rates)} -- "
          f"{'rising monotonically toward 2' if monotone else 'NOT monotone'}, finest {rates[-1]:.2f}")

    # ---------------------------------------------------------------- 4. temporal order
    # Compared against the dt->0 limit ON THE SAME GRID (Richardson), not against sigma_exact:
    # the spatial error is a fixed offset that would otherwise swamp the temporal one.
    print("\n4. Temporal convergence (n=32 fixed; Richardson, so the spatial error cancels)")
    for scheme, ts, want in (("chorin", "be", 1.0), ("rotational", "bdf2", 2.0)):
        sg = [run(2, 1, n=32, dt=d, nsteps=int(round(0.2 / d)), scheme=scheme,
                  time_scheme=ts)["sigma"] for d in (8e-3, 4e-3, 2e-3)]
        d1, d2 = abs(sg[0] - sg[1]), abs(sg[1] - sg[2])
        order = np.log2(d1 / d2) if d2 > 0 else np.inf
        print(f"   {scheme:10s}/{ts:5s}  sigma = {sg[0]:.5f}, {sg[1]:.5f}, {sg[2]:.5f}"
              f"   observed order {order:.2f} (expect ~{want:.0f})")
        check(f"{scheme}/{ts} growth rate converges at order ~{want:.0f}",
              abs(order - want) < 0.5, f"observed {order:.2f}")

    # ---------------------------------------------------------------- 5. a real disturbance
    # A disturbance is not one eigenmode. Superpose three and check each decays at ITS OWN rate --
    # i.e. the solver reproduces the Stokes spectrum, not just one eigenvalue. The high mode decays
    # ~13x faster here, so any mode-to-mode leakage would show up immediately.
    print("\n5. Multi-mode disturbance: each mode must decay at its own rate (n=48, dt=5e-4)")
    MODES = ((1, 1), (2, 1), (3, 2))
    r = run(*MODES[0], n=48, dt=5e-4, nsteps=200, extra_modes=MODES[1:])
    print(f"   {'mode':>8} {'exact':>10} {'measured':>10} {'rel err':>10}")
    mm = []
    for (m1, m2) in MODES:
        # RATIO of final to initial projection. Each mode enters with weight AMP, not 1, so
        # log|a_final|/T would return sigma + log(AMP)/T -- off by a large constant.
        a_i = project(r["s"], r["u0"], r["v0"], m1, m2)
        a_f = project(r["s"], r["s"].u, r["s"].v, m1, m2)
        sg = np.log(abs(a_f / a_i)) / r["T"]
        ex = sigma_exact(m1, m2)
        mm.append(abs(sg / ex - 1))
        print(f"   {f'({m1},{m2})':>8} {ex:10.4f} {sg:10.4f} {mm[-1]:10.2e}")
    check("every mode in a superposed disturbance decays at its own rate",
          max(mm) < 0.03, f"worst relative error {max(mm):.2e} across 3 simultaneous modes")

    n_pass = sum(results)
    print(f"\n{'='*74}\n  {n_pass}/{len(results)} checks passed\n{'='*74}")
    sys.exit(0 if n_pass == len(results) else 1)
