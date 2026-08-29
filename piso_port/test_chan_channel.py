"""
Replication of the Chan (1996) channel validation, Problem 1 (Stokes decay).

Source: github.com/chandc/Python_SEM/blob/main/CHANNEL_VALIDATION.md -- a SPECTRAL ELEMENT
solver validating against Chan (1996). Same physical problem as test_stokes_channel.py:

    plane channel, y in [-1,1], no-slip; periodic in x over [0, 2*pi] with alpha = 1; nu = 1
    least-damped Stokes eigenvalue  sigma = 9.3137399   (Chan's published value: 9.313316)

WHAT CAN AND CANNOT BE REPLICATED. Their solver is spectral in space (polynomial order N = 6-14,
exponential convergence), so its spatial error is ~1e-9 and its dt table is essentially PURE
temporal error, bottoming out at 1.7e-6. This solver is 2nd-order finite difference: matching
1.7e-6 in space would need ny ~ 1080, which is not affordable. So the comparison is:

    directly comparable   the eigenvalue itself; E(T)/E_0; the TEMPORAL convergence slope;
                          amplitude-independence; divergence level
    not comparable        absolute accuracy at a given cost -- spectral wins by construction

WHAT THIS REPLICATION FOUND. Their fitted temporal slope is 1.940 (N=6) to 1.993 (N=14) -- clean
second order. This repo had published a measured 1.68 on the same problem and attributed the gap
to the O(dt^3/2) near-wall splitting error of rotational incremental projection. Their result
prompted a re-check, and THE PUBLISHED CLAIM WAS WRONG: the 1.68 came from one Richardson triple
measured outside the asymptotic range. Extending the sweep at ny=49 gives

    triple (dt)        4e-4,2e-4,1e-4   2e-4,1e-4,5e-5   1e-4,5e-5,2.5e-5
    ratio                        1.74             3.20               4.00
    order                        0.80             1.68               2.00

so the scheme IS second order in time with no-slip walls, in agreement with their solver. The
mistake was one already made and fixed for the SPATIAL order in test_stokes_growth.py, where
rates of 1.76/1.89/1.95 turned out to be pre-asymptotic -- and having caught it once, it should
have been checked here instead of being explained away with a plausible mechanism.
"""
import sys, io, contextlib, warnings
import numpy as np
warnings.filterwarnings("ignore")
from test_stokes_channel import stokes_mode, run, ALPHA, NU, SIGMA_REF

SIGMA_ANALYTIC = 9.3137399          # their eigenproblem value
SIGMA_CHAN = 9.313316               # Chan (1996) published
results = []
def check(name, good, detail):
    results.append(good)
    print(f"  [{'PASS' if good else 'FAIL'}] {name}: {detail}")


if __name__ == "__main__":
    print(__doc__.strip().split("\n\n")[0])

    # ---------------------------------------------------------------- 1. the eigenvalue
    print(f"\n1. The eigenvalue itself (their analytic {SIGMA_ANALYTIC}, Chan {SIGMA_CHAN})")
    sg = stokes_mode(N=80)[0]
    print(f"   this repo, Chebyshev clamped-D4, N=80:  sigma = {sg:.7f}")
    print(f"   their eigenproblem:                     sigma = -{SIGMA_ANALYTIC}")
    print(f"   Chan (1996) published:                  sigma = -{SIGMA_CHAN}")
    d_an = abs(abs(sg) - SIGMA_ANALYTIC) / SIGMA_ANALYTIC
    d_ch = abs(abs(sg) - SIGMA_CHAN) / SIGMA_CHAN
    check("independent eigensolvers agree", d_an < 1e-7,
          f"relative difference {d_an:.2e} vs their analytic value "
          f"(both differ from Chan's published number by {d_ch:.1e}, as their doc also notes)")

    # ---------------------------------------------------------------- 2. E(T)/E_0 at T=0.1
    # Their table reports E(T)/E_0 = 0.1553 after 40 steps of dt=0.0025, i.e. T = 0.1.
    print("\n2. Energy ratio at T=0.1 (their reported E(T)/E_0 = 0.1553)")
    print(f"   {'dt':>9} {'steps':>6} {'E(T)/E_0':>10} {'vs 0.1553':>11}")
    exact_ratio = np.exp(2 * SIGMA_REF * 0.1)
    for dt in (0.0025, 0.00125, 0.000625):
        sg_m, _, s_ = run(65, dt=dt, nsteps=int(round(0.1 / dt)))
        E = np.exp(2 * sg_m * 0.1)
        print(f"   {dt:9.6f} {int(round(0.1/dt)):6d} {E:10.4f} {abs(E - 0.1553):11.2e}")
    check("energy ratio matches their 0.1553", abs(E - 0.1553) < 5e-3,
          f"{E:.4f} vs 0.1553 (exact {exact_ratio:.4f})")

    # ---------------------------------------------------------------- 3. their dt table
    print("\n3. Their dt sweep, same protocol (rate measured from t=0, as they do)")
    print("   NOTE their table is nearly pure TEMPORAL error; ours is floored by the 2nd-order")
    print("   SPATIAL error, which is why the columns stop improving.")
    DTS = (0.02, 0.01, 0.005, 0.0025, 0.00125)
    print(f"   {'dt':>9} {'this repo':>12} {'their N=6':>11} {'their N=14':>11}")
    theirs6 = {0.02: 9.864e-3, 0.01: 3.018e-3, 0.005: 7.630e-4, 0.0025: 1.966e-4,
               0.00125: 4.652e-5}
    theirs14 = {0.02: 9.861e-3, 0.01: 3.046e-3, 0.005: 7.620e-4, 0.0025: 1.824e-4,
                0.00125: 4.031e-5}
    ours = {}
    for dt in DTS:
        sg_m, _, _ = run(65, dt=dt, nsteps=int(round(0.1 / dt)))
        ours[dt] = abs(abs(sg_m) - SIGMA_ANALYTIC) / SIGMA_ANALYTIC
        print(f"   {dt:9.5f} {ours[dt]:12.3e} {theirs6[dt]:11.3e} {theirs14[dt]:11.3e}")
    # their fitted slope over 0.02 -> 0.00125
    lx = np.log(np.array(DTS)); ly = np.log(np.array([ours[d] for d in DTS]))
    slope_raw = np.polyfit(lx, ly, 1)[0]
    print(f"   fitted slope (0.02-0.00125): ours {slope_raw:.3f}   "
          f"theirs 1.940 (N=6) to 1.993 (N=14)")

    # ---------------------------------------------------------------- 4. temporal slope, fairly
    print("\n4. TEMPORAL slope isolated by Richardson (cancels our fixed spatial offset)")
    print("   measured over a settled window, since our initial field is an eigenmode of the")
    print("   CONTINUOUS operator and relaxes onto the discrete one first")
    sgs = [run(49, dt=d, nsteps=0, window=(0.05, 0.10))[0] for d in (2e-4, 1e-4, 5e-5)]
    d1, d2 = abs(sgs[0] - sgs[1]), abs(sgs[1] - sgs[2])
    slope = np.log2(d1 / d2)
    print(f"   sigma = {sgs[0]:.6f}, {sgs[1]:.6f}, {sgs[2]:.6f}   ->  order {slope:.2f}")
    print("   this triple is PRE-ASYMPTOTIC; extending it:")
    sg2 = [run(49, dt=d, nsteps=0, window=(0.05, 0.10))[0] for d in (1e-4, 5e-5, 2.5e-5)]
    slope2 = np.log2(abs(sg2[0] - sg2[1]) / abs(sg2[1] - sg2[2]))
    print(f"   dt = 1e-4, 5e-5, 2.5e-5  ->  order {slope2:.2f}")
    check("temporal order reaches 2, matching their spectral-element result",
          slope2 > 1.9,
          f"order {slope2:.2f} on the finest triple (was {slope:.2f} on a coarser one) "
          f"vs theirs 1.94-1.99 -- agreement, and the earlier published 1.68 was pre-asymptotic")

    # ---------------------------------------------------------------- 5. amplitude independence
    print("\n5. Amplitude independence (their check: halving the IC changes sigma in the 5th dp)")
    import test_stokes_channel as T
    a0 = T.AMP
    vals = {}
    for amp in (1e-3, 5e-4):
        T.AMP = amp
        vals[amp] = run(49, dt=1e-4, nsteps=0, window=(0.05, 0.10))[0]
    T.AMP = a0
    dd = abs(vals[1e-3] - vals[5e-4])
    print(f"   amp 1e-3 -> {vals[1e-3]:.6f}    amp 5e-4 -> {vals[5e-4]:.6f}    diff {dd:.1e}")
    check("halving the amplitude leaves sigma unchanged to ~5 decimals", dd < 1e-4,
          f"difference {dd:.2e} -- genuinely in the Stokes limit, as their check requires")

    n_pass = sum(results)
    print(f"\n{'='*78}\n  {n_pass}/{len(results)} checks passed\n{'='*78}")
    sys.exit(0 if n_pass == len(results) else 1)
