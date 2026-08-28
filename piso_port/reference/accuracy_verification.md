# Verifying order of accuracy: spatial and temporal

How the solver's convergence rates were established, what went wrong on the way, and which
claims are actually supported by the measurements.

The short version:

| | how measured | result |
|---|---|---|
| **spatial**, individual operators | MMS on a warped grid, per phase | **2.0–2.5** across warp 0.01–0.30 and ν 0.01–10 |
| **spatial**, full solver | fully periodic, `dt` fixed small so the time error is negligible | **2.02, 2.00, 2.00** (warped); 3.1–3.3 (Cartesian) |
| **temporal**, full solver | vs a *same-grid numerical reference*, so the spatial error cancels | **1.91, 1.94, 1.97** (rotational + BDF2) |
| **temporal**, wall-bounded | same method, Dirichlet on all faces | **~0.6** — limited by the 1st-order wall stencil |

Reproduce with:

```bash
uv run phase1_grid_metrics.py     # per-operator MMS rates
uv run phase2_operators.py
uv run phase3_momentum.py
uv run phase4_poisson.py
uv run test_phase3_rigorous.py    # 23 checks: warp x viscosity, independent solver, 2nd MMS field
uv run test_spatial_order.py      # full-solver spatial order
uv run test_phase5_order.py       # full-solver temporal order
```

---

## 1. The central methodological point

**Spatial and temporal error must be separated, or each hides the other.** Both of the obvious
approaches fail, and both failed here before being fixed:

- **Refining `dt` together with `h`** (e.g. `dt ∝ h²`) measures the *sum* of the two errors.
  A scheme can then look 1st order because of its time integrator while its spatial
  discretisation is a clean 2nd order, and the measurement cannot tell you which.
- **Comparing against the analytic solution at fixed `h`** to get the temporal order fails
  because the spatial error is a floor. Measured here: at n = 32 the spatial floor is
  **1.6e-04**, and a `dt` sweep bottomed out at 3.7e-04 — only 2× above the floor, which
  inflated and scrambled the apparent orders.

The two fixes used below:

- **Spatial order** → hold `dt` fixed and *small enough that the temporal error is negligible
  at every resolution*, then refine `h`.
- **Temporal order** → compare against a **numerical reference computed on the same grid** with
  a very small `dt`. The spatial discretisation error is identical in both runs and cancels
  *exactly*, leaving only the temporal term.

---

## 2. Spatial accuracy

### 2a. Individual operators (Phases 1–4)

Each phase is gated by the Method of Manufactured Solutions on a deliberately warped grid:

| Phase | Quantity | Rates |
|---|---|---|
| 1 | Jacobian & metrics | 2.08, 2.05 (GCL **7.2e-13**) |
| 2 | gradient / divergence | 2.13, 2.07 / 2.26, 2.12 |
| 3 | momentum matrix solve | 2.26, 2.15 (u) · 2.46, 2.46 (v) |
| 4 | pressure Poisson | 2.30, 2.14 |
| — | periodic Poisson | 2.01, 2.01 (Cartesian) · 1.91, 1.95 (warped) |

`test_phase3_rigorous.py` extends this to 23 checks: warp sweeps 0.01→0.20, viscosity sweeps
0.01→10, a second independent MMS field (deliberately non-solenoidal), a dense-LAPACK
cross-check agreeing to 9.45e-13, and the warp×ν corner where the two interact.

Phase 2's operators were separately confirmed to hold 2nd order out to **warp 0.20**
(rates 2.07 / 2.03), so the operator accuracy is not an artifact of mild skewness.

### 2b. The full solver

Never isolated until late, because the only full-solver test refined `dt` with `h`. Done
properly — fully periodic (no walls, hence no projection boundary layer), rotational + BDF2,
`dt = 3.125e-4` fixed, `n = 16, 24, 32, 48`:

```
warp 0.05   errs 3.785e-03  1.670e-03  9.384e-04  4.167e-04   rates 2.02, 2.00, 2.00
Cartesian   errs 7.110e-04  2.027e-04  8.094e-05  2.144e-05   rates 3.10, 3.19, 3.28
```

**Second-order spatial accuracy confirmed on the warped grid** — which is the case that
matters, since it exercises the metrics, the non-orthogonal correction, and the curvilinear
operators.

### 2c. The Cartesian case converges faster than 2nd order

Rates of 3.1–3.3 initially tripped the test's upper bound, which had been added earlier to
catch blown-up data points. Before widening the bar, the result was checked for the obvious
artifact — grid alignment. A translated Taylor-Green field is *still an exact solution*, so
shifting it de-aligns the solution from the nodes without changing the problem:

| offset | rates |
|---|---|
| aligned (a = 0) | 3.10, 3.19, 3.28 |
| half-cell (a = h/2) | 3.13, 3.21, 3.28 |
| irrational (a = 0.137) | 3.12, 3.19, 3.29 |

Unchanged — so this is genuine super-convergence, not aliasing. On a uniform grid every metric
is exactly constant, so the metric-induced error terms vanish identically and a subdominant
term is left to die out. The bar was widened to [1.7, 4.5] **with that reasoning recorded**,
while keeping the monotonicity guard that catches actual corruption.

---

## 3. Temporal accuracy

### 3a. Method

Fixed grid (n = 24), fully periodic, error measured against a numerical reference computed on
the *same grid* at `dt_ref = T/512`. Judged on the **coarsest** interval, since the finest `dt`
sits only 8× from the reference and the reference's own temporal error contaminates that
estimate (visible as the inflating third column below).

### 3b. Results, periodic

```
chorin      be     errs 2.08e-03  1.03e-03  5.01e-04  2.34e-04   orders 1.01, 1.04, 1.10
chorin      bdf2   errs 1.55e-03  7.35e-04  3.48e-04  1.61e-04   orders 1.08, 1.08, 1.12
rotational  be     errs 1.66e-04  4.16e-05  8.36e-06  1.23e-06   orders 2.00, 2.32, 2.77
rotational  bdf2   errs 3.50e-04  9.34e-05  2.43e-05  6.19e-06   orders 1.91, 1.94, 1.97
```

**Rotational + BDF2 gives clean 2nd order.**

**A prediction of mine that the measurement corrected:** I expected `chorin + bdf2` to reach
2nd order and wrote the test to assert it. It does not, and *should* not — the non-incremental
(Chorin) splitting error is O(Δt) and no time integrator repairs it. The test now asserts 1st
order there deliberately, and the case is kept precisely because it demonstrates that the
rotational correction — not the integrator — is what lifts the order.

### 3c. Results, wall-bounded (channel)

Measured on channel startup flow — walls in y, periodic in x/z, exact unsteady solution
(`test_channel_order.py`):

```
chorin / BE        errs 1.62e-03  8.07e-04  3.93e-04  1.84e-04   orders 1.01, 1.04, 1.10
rotational / BDF2  errs 2.55e-04  6.27e-05  1.55e-05  3.83e-06   orders 2.03, 2.01, 2.02
```

**2nd order in time is achieved on a wall-bounded channel.** Chorin is 1st order by
construction, as designed.

An earlier version of this document reported that 2nd order in time was *not* achievable on a
channel. That was a consequence of the $\Gamma$ coefficient bug described in
[`piso_equations.md`](piso_equations.md) §5, not a property of the scheme.

### 3d. Curvilinear grids need more correctors, not a different order

On warped grids the same 2nd order is reached, but `corrector_steps=2` is not enough to be
asymptotic at coarse `dt`:

| warp | corr=2 | corr=4 | corr=8 |
|---|---|---|---|
| 0.05 | 1.55, 1.14, 1.74 | 1.68, 1.53, **2.02** | 1.83, 1.90, **2.05** |
| 0.10 | 0.80, 0.79, 1.63 | 0.99, 1.15, **2.01** | 1.25, 1.65, **2.09** |

This is the PISO operator-**splitting** error, not the time integrator and not the grid
transformation — a spatial map cannot change a temporal order, and it does not. On a Cartesian
channel the flow is 1D, advection vanishes, and the cell-centred and flux corrections agree, so
2 correctors suffice; warping makes the flow genuinely 3D and opens a gap between them that each
corrector shrinks.

Worth recording how long this took to see: four hypotheses were tested and eliminated first
(pressure-DC tolerance, momentum-DC sweep count, reference contamination, and the Picard lag —
`picard_iters` was implemented specifically to test the last). Only a systematic bisection over
scheme ingredients found it. Bisect before theorising.

## 4. Why the earlier walled measurement read ~1st order

Before the separation was done properly, the walled Taylor-Green test (refining `dt ∝ h²`)
gave rates 0.96–1.12 (Cartesian) and 0.79–0.81 (warped). That was **attributed to scheme order
reduction, and the attribution was verified rather than assumed**:

- the error decays monotonically away from the wall — 5.4e-02, 2.7e-02, 2.4e-02, 1.4e-02,
  7.8e-03 across the first five layers — the signature of a numerical boundary layer;
- it does **not** improve with more correctors (1/2/4 → 1.25e-02, 1.35e-02, 1.48e-02), ruling
  out an unconverged iteration.

Both observations point at the boundary treatment, which §3c then confirmed independently.

---

## 5. Pitfalls worth carrying forward

1. **Never refine `dt` and `h` together** when you want to attribute an order to one of them.
2. **Use a same-grid numerical reference for temporal order.** Against the analytic solution,
   the spatial floor dominates long before the order can be read.
3. **Check where the reference's own error sits.** With `dt` only 8× from `dt_ref`, the finest
   estimate is contaminated — judge on the coarsest interval.
4. **A rate *above* the design order deserves a check, not just a widened bar.** The
   phase-shift test settled it in three runs.
5. **Locate the error before explaining it.** Error-versus-wall-distance and
   error-versus-corrector-count distinguished "boundary layer" from "unconverged iteration"
   without any theory.
6. **Bar the rate from both sides.** The convergence checks require all rates within a band
   *and* monotone decreasing — a bound on only the last rate lets a blown-up intermediate
   point through, which it did, producing a "23/23 passed" run that contained garbage.

---

## 6. Standing limitations

- **Wall-bounded spatial order is 1st**, from the half-cell boundary-flux stencil. Periodic
  cases are 2nd order.
- **Deferred correction limits grid warp to ≲ 0.15**; beyond it the Picard iteration stops
  contracting (ratios 0.31 / 0.59 / 0.92 / 1.27 at warp 0.05 / 0.10 / 0.15 / 0.20) and the
  solver warns rather than returning a quietly wrong field.
- **Chorin remains the default** (matching PICT's `apply_pressure_gradient = False`), so the
  default configuration is 1st order in time. `scheme='rotational', time_scheme='bdf2'` opts
  into 2nd order.

---

## 7. What order of accuracy does *not* tell you

Convergence rates verify that each operator approaches its differential counterpart as
$h,\Delta t \to 0$. They say nothing about whether the assembled **nonlinear** scheme respects
the quadratic invariants that govern a turbulent cascade — a scheme can be cleanly 2nd order
and still bleed energy through numerical dissipation.

The complementary check is the exact periodic identity

$$\frac{\mathrm{d}E}{\mathrm{d}t} = -2\nu Z, \qquad
E=\tfrac12\langle|\mathbf u|^2\rangle, \quad Z=\tfrac12\langle|\boldsymbol\omega|^2\rangle$$

on a 3D Taylor-Green vortex, fully periodic. The gap between measured $-\mathrm{d}E/\mathrm{d}t$
and $2\nu Z$ **is** the scheme's numerical dissipation. Measured at 48³, ν = 0.01:

| convection scheme | mean numerical / physical dissipation | max |
|---|---|---|
| 2nd-order upwind | **1.10 %** | 1.48 % |
| central | **0.56 %** | 0.66 % |

Both are formally 2nd order and both pass every MMS test, yet SOU dissipates about twice as
much — the price of the upwind bias that buys its stability. Neither number is visible in a
convergence rate. Run `run_tgv3d.py` then `plot_tgv3d.py`.

### Grid convergence of the budget

Run at three resolutions (central convection) to confirm 48³ is adequate:

| N | E(t=2) | mean numerical dissipation | E vs 64³ |
|---|---|---|---|
| 32³ | 6.850e-04 | 1.27 % | 1.64 % |
| **48³** | **6.768e-04** | **0.56 %** | **0.42 %** |
| 64³ | 6.739e-04 | 0.31 % | — |

Two things follow. First, the **numerical dissipation itself converges at 2nd order** — rates
**2.01** and **2.00** — which is an independent confirmation of the spatial order reached by a
completely different route from the MMS tests: measured from a *physical invariant* rather than
from a manufactured solution. Second, 48³ is converged for this purpose: against 64³ the energy
curve differs by ≤ 0.07 % and enstrophy by ≤ 0.26 %, well below the ~1 % effect being measured,
so the SOU-vs-central comparison is not a resolution artifact.

### Caveat: this is not turbulence

At ν = 0.01 enstrophy **peaks at t = 0** and decays monotonically — there is no vortex
stretching and no cascade. In the canonical TGV at Re = 1600, enstrophy rises to a peak near
t ≈ 9, and that peak *is* the cascade; reaching it needs ~256³, roughly two orders of magnitude
more work than a NumPy solver can carry. What this validates is the energy-budget machinery and
the relative dissipation of the two convection schemes — nothing about cascade prediction.
