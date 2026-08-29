# The pressure checkerboard: symptom, cause, and fix

## How it surfaced

Plotting `u`, `v` and `p` at five axial stations through the Armaly backward-facing step
(`figures/bfs_profiles.png`) produced clean velocity profiles and a **visibly sawtoothed
pressure**. The velocities were trustworthy — they reproduce Armaly's reattachment lengths to
within a few percent — so the first question was whether the sawtooth was a plotting artefact
of reassembling two blocks, or something real in the solver.

It is real, and it is not a multi-block artefact.

Measured on the production BFS field (Re = 100, nx = 80), splitting each wall-normal profile
into a smooth part and a node-to-node alternating part:

| field | range | odd–even amplitude | ratio |
|---|---|---|---|
| p | 0.105 | 0.0237 | **22.6 %** |
| u | 1.052 | 0.0097 | 0.9 % |

The pressure sign flips at **36 of 36** consecutive nodes — a pure alternating mode, not noise —
and the amplitude is the same in both blocks (0.0235 lower, 0.0244 upper), so the seam is not
involved.

## Isolating the mechanism

`diag_checkerboard.py` measures the alternating content by scheme. The split is sharp:

| case | scheme | p | u | φ |
|---|---|---|---|---|
| single block, warped duct | chorin | **9.2 %** | 1.6 % | |
| | incremental | 25.5 % | 1.5 % | |
| | rotational | 25.5 % | 1.5 % | |
| multi-block BFS, 2 domains | chorin | **4.8 %** | 2.7 % | 4.8 % |
| | incremental | 23.7 % | 2.7 % | **37.5 %** |
| | rotational | 24.7 % | 2.7 % | **38.6 %** |

Two things stand out. First, the defect is **present in both solvers** — it is a property of the
shared collocated discretisation, not of the multi-block layer. Second, `chorin` is clean while
the accumulating schemes are not, and in those schemes **φ itself** carries 38 % — so the mode
is regenerated every step, not merely inherited.

> Percentages in this table are ratios; see the measurement-error section below
> for why a ratio alone is not safe to read.

> The lid-driven cavity is deliberately **not** the probe case. Its corners carry a genuine
> pressure singularity, so a node-to-node metric there reads ~29 % even for `chorin` and mixes
> real physics with the artefact. The smooth warped duct isolates it.

## The cause

Three facts about the discretisation, each individually reasonable:

1. **The Poisson operator uses compact face differences.** `pressure_face_fluxes` builds every
   face coefficient from `(p[i+1] − p[i])/h`. A compact Laplacian has no checkerboard null
   mode, which is why `chorin` — where `p = φ` straight from the solve — comes out clean.
2. **The momentum predictor uses the wide gradient.** `Domain.gradient` calls `np.gradient`,
   i.e. `(p[i+1] − p[i−1])/2h`, which is *identically zero* for an alternating field.
3. **The face flux is a plain average.** `Domain.face_fluxes` interpolates the contravariant
   velocity as `0.5*(JU[i] + JU[i+1])` with no Rhie–Chow pressure term.

Together, (2) and (3) mean a checkerboard in `p` **exerts no force and moves no mass**. Nothing
excites it deliberately; nothing damps it either. But the projection still credits the predictor
with having felt the full `p`, so every step φ must absorb the discrepancy between the pressure
that was actually applied and the pressure on the books. That discrepancy *is* the checkerboard,
so φ acquires it, `p += φ` accumulates it, and the loop closes with no dissipation anywhere in
it. Hence 4.8 % → 38 % in φ the moment accumulation is switched on.

The velocity is unharmed because it only ever sees the smooth part — which is why the BFS
reattachment lengths are still right.

## The fix: Rhie–Chow momentum interpolation

Add to the face flux the **compact-minus-wide** pressure difference:

```
F_face  ←  F_face  −  Γ_f · [ (p_N − p_P)/h  −  ½((∇p)_P + (∇p)_N)·n ]
```

The bracket is the whole point:

* for a **smooth** field the two stencils agree to O(h²), so the term is O(h³ ∂⁴p/∂x⁴) — it
  vanishes with the grid and **does not touch second-order accuracy**;
* for an **alternating** field the wide term is exactly zero while the compact term is maximal,
  so the term equals the full checkerboard amplitude.

It is therefore a targeted ~h³∇⁴p dissipation that acts *only* on the mode that is currently
undamped. Both halves already existed in the code — the compact term in `pressure_face_fluxes`,
the wide one in `gradient`/`deriv` — so the implementation reuses the **same face enumeration**
as the pressure operator rather than duplicating it, which is what keeps it correct at seams,
periodic wraps and walls.

## A measurement error worth recording

The first version of `odd_even` reported the alternating amplitude as a **percentage of each
profile's range**, and skipped profiles that were flat. When every profile was flat it returned
`0.0` as a fallback. A force-driven straight duct has uniform pressure (|p|max = 2.4e-15), so
all twelve profiles were skipped and the metric reported "0.0% -- clean" for a field that did
not exist. That produced a confident and completely wrong claim that grid warp had an enormous
effect on the checkerboard.

Measured absolutely, the ratio is roughly **constant at 17-21% across warps**: warp does not
amplify the mode, it merely creates a pressure field for the mode to live in. `odd_even_abs`
now returns the absolute amplitude and the range separately, and the percentage form returns
NaN when there is nothing to normalise against.

Two further false alarms came from the verification itself, not the solver. Comparing
`div(Phi(p))` against `+M p` gives a relative error of exactly 2.00 (sign convention: M
discretises -laplacian); fixing the sign but dropping the J factor gives a 1.7-9% mismatch that
grows with warp and looks exactly like a real operator inconsistency. The correct invariant is
`J*div(Phi(p)) == -M p`, and it holds to **2-4e-16** at warp 0.00 / 0.05 / 0.10 and n = 8 / 12 /
16. The flux operator and the Poisson matrix are the same operator.

## What upstream PICT does about it

Nothing. Reading the CUDA kernels directly:

* `getPressureGradient` is `(valP - valN)*0.5` -- the wide stencil.
* `getPressureGradientFVM` is a Gauss gradient whose own comment says it is "identical to
  finite difference gradient for orthogonal grids"; all four `gradientInterpolation` variants
  are half-point interpolations between P and N, which telescope back to the wide stencil.
* `computeFluxesNDLoop` is `fluxes[bound] = (velN + velC) * 0.5f` -- a plain average with no
  pressure term.

The only oscillation-related code in the whole kernel file is a boundary extrapolation choice
(`//p = pP; //using center pressure ... leads to oscillations at the boundary`) and a smoothing
kernel marked "DO NOT USE". So the susceptibility is **inherited from the reference
implementation**, not introduced by this port, and everything below is a deliberate deviation
from PICT.

## Implementation

Three options on `PISOSolver`, all defaulting to `False`:

| option | what it does |
|---|---|
| `persistent_flux` | build the face flux once per step and correct it in place, instead of rebuilding it from the cell velocity every corrector |
| `rhie_chow` | subtract `D = Phi_compact(p) - I(Phi_wide(p))` from the flux, once per step |
| `ddt_corr` | add `(Gamma_f/dt)(F_prev - I(F(u^n)))`, the transient term |

`rhie_chow` is fed **`p_flux`, not `p`**. This is load-bearing. The rotational scheme sets
`p += phi - nu*div(u*)`, and that last term was never carried by the face flux; feeding it back
made Rhie-Chow remove flux that was never added. The resulting loop diverged with a sharp
threshold -- `|RC|/|F|` went 0.02 at step 50 to **1.36** at step 60 to 58 at step 70, NaN at
step 83 -- while `divF` stayed at 1e-12 throughout, which is what ruled out the linear solver
and located the fault in the feedback path. `p_flux` accumulates the projection pressure only
and is identical to `p` under every scheme except `rotational`.

## Results

Warped duct, A = 0.05, `rotational`. Odd-even amplitude as a fraction of the pressure range:

| n | RC off | RC on |
|---|---|---|
| 12 | 20.7% | 15.7% |
| 16 | 25.4% | 14.4% |
| 24 | **32.1%** | **10.8%** |

Without the term the mode **grows** under refinement; with it the mode **converges away**. That
reversal, not the single-grid reduction, is the result that matters.

dt sweep at n = 12 (`ddt_corr` is not optional):

| dt | RC only | RC + ddt_corr |
|---|---|---|
| 0.0200 | 14.2% | 2.6% |
| 0.0100 | 15.6% | 2.8% |
| 0.0050 | 17.0% | 3.0% |
| 0.0025 | 18.0% | 3.7% |

Since `Gamma ~ dt`, plain Rhie-Chow damping is O(dt) and **weakens as the step is refined** --
14.2% to 18.0% over an 8x refinement. Shipping without the transient term would have passed at
whatever dt happened to be tested and failed quietly at another. With it, the absolute amplitude
drops ~45x (1.93e-02 to 4.31e-04) and holds roughly flat.

Order of accuracy, against the exact duct Fourier series:

| warp | n=8 | n=16 | rates (off / on) |
|---|---|---|---|
| 0.00 | 0.00% | 0.00% | 2.06, 2.04 / 2.06, 2.04 |
| 0.05 | 0.48% | 0.42% | -- |

On an orthogonal grid the term is **bitwise inert**, which is the O(h^3) claim made good.

## Multi-block

The same three options exist on `MultiBlockPISO`. Two things had to hold and both do:

* **Block-count independence survives.** With the full fix on, 2 blocks vs 1 agree to
  2.07e-10 and 4 blocks vs 1 to 3.78e-10 -- the same order as the existing 55-check suite.
* **Seam fluxes need no ownership rule.** Both sides of a connection already compute bitwise
  identical face fluxes AND bitwise identical pressure corrections (0.00e+00, with and without
  cross terms), and periodic wrap faces are single-valued. The plan had assumed persistence
  would let the two copies drift and that an A-side owner would be needed; measurement says
  otherwise. That assumption came from `build_diffusion_matrix`, where double-adding a
  connection genuinely does halve the diffusion -- but fluxes are not summed that way, so the
  hazard does not transfer.

On the forced periodic strip -- whose exact pressure is uniform, so all pressure variation is
spurious -- the improvement is much larger than on the duct:

| blocks | baseline | persistent | persistent + RC | divF |
|---|---|---|---|---|
| 1 | 1.26e-02 | 1.33e-05 | **1.43e-11** | 1.5e-13 -> 2.7e-15 |
| 2 | 9.50e-03 | 3.36e-05 | **1.34e-12** | 1.1e-13 -> 3.1e-15 |
| 4 | 3.04e-03 | 1.92e-07 | **8.46e-12** | 3.6e-14 -> 2.7e-15 |

Note that `persistent_flux` alone does most of the work here (1e-2 -> 1e-5) while on the warped
duct it did nothing measurable. The strip is Cartesian, so there are no cross terms; different
mechanisms dominate in the two cases, and neither case alone characterises the fix.

dt sweep across blocks makes the `Gamma ~ dt` argument starker than the single-block duct did:

| blocks | dt | RC only | RC + ddt_corr |
|---|---|---|---|
| 2 | 0.0200 | 6.41e-13 | 1.66e-13 |
| 2 | 0.0050 | **5.68e-10** | **8.18e-13** |
| 4 | 0.0200 | 4.90e-13 | 2.24e-13 |
| 4 | 0.0050 | **1.15e-09** | **7.31e-13** |

A 4x refinement in dt makes plain Rhie-Chow ~900x worse; with the transient term it holds flat.

## Skew grids, and the seam bug they exposed

The multi-block gates above all ran on `strip`, which is **Cartesian** -- no cross terms at all.
Re-running them on a warped channel (walls in y, periodic z, connections in x) immediately broke
mass conservation: flux divergence went from 3.8e-15 to **1.5e-04**, stalled at that value for 2,
3 and 5 correctors (so it was a formulation error, not a convergence shortfall), and failed at
one block too (so it was not a seam *coupling* problem).

The residual was exactly proportional to 1/J (`corr = 1.0000`) -- the signature of a
compatibility constant being discarded -- and `div(RC)` had volume integral -0.247 instead of
zero, meaning the term was moving net mass through the boundary.

The cause: the Rhie-Chow **wide** half is `np.gradient` evaluated on the block's own padded
field, and on a width-1 pad that stencil is **one-sided exactly at the ghost cell a seam face
needs**. Each block pads from its own neighbour, so the two sides of a seam disagreed --
measured mismatch **1.084e+02**, against 0.000e+00 for the plain pressure flux. Padding to
width 2 makes the stencil central at every cell a face touches; the seam mismatch returns to
**0.000e+00** and divergence to 1.4e-15, better than the baseline.

This was written into the plan as a known caveat and dismissed as "O(h^3), a small
inconsistency worth remembering". It was not small: it broke mass conservation outright. A
seam consistency check on the RC term itself would have caught it immediately, and the earlier
seam test covered `include_cross` but never `rhie_chow=True`.

Warped channel, `rotational`, amplitude and flip fraction (see below on why both are needed):

| blocks | rhie_chow | amp(p) | flip | amp(u) | flip | divF |
|---|---|---|---|---|---|---|
| 1 | off | 3.77e-03 | 0.74 | 2.38e-02 | 0.06 | 3.7e-15 |
| 1 | **on** | **2.32e-04** | **0.28** | 2.41e-02 | 0.09 | 2.7e-15 |
| 2 | **on** | **2.32e-04** | **0.28** | 2.41e-02 | 0.08 | 2.9e-15 |
| 4 | **on** | **2.32e-04** | **0.28** | 2.41e-02 | 0.07 | 2.7e-15 |

Identical to four digits across block counts, and |u|max agrees to 2.2e-16.

## Measuring oscillation: neither single number works

Two metrics were tried and both were wrong, in opposite directions.

* The **1-2-1 smoother deviation** also sees smooth CURVATURE. On a parabolic channel profile it
  reports 1.9e-02 of "oscillation" where only 7% of node pairs actually reverse slope. Every
  velocity number quoted before this was measuring the profile, not a mode.
* A **global (-1)^j Fourier projection** is blind to curvature but cancels for a mode whose
  envelope is antisymmetric about the channel centreline. It reported 1.6e-16 for a field whose
  slope reverses at 88% of node pairs -- a genuine checkerboard called clean.

`diag_checkerboard.checkerboard()` now returns **(amplitude, flip fraction)**. The flip fraction
is the detector -- a pure mode flips at every pair, a smooth profile almost never -- and the
amplitude is only a checkerboard magnitude when the flip fraction is high. On that reading the
velocity never carried a checkerboard in these cases (flips 0.04-0.09), which is why Rhie-Chow
leaves it untouched, and the pressure did (flips 0.74-0.98).

## What this does NOT fix, and what to watch for

* **The plan's central hypothesis was wrong.** Making the flux persistent was argued to be what
  would stabilise Rhie-Chow. It is not: persistent+RC still diverged at n=16, and rebuild+RC and
  persistent+RC give identical answers (2.150e-02 vs 2.149e-02) because the rebuild path already
  tracked the accumulated pressure. Persistence is still worth having -- one flux build instead
  of `corrector_steps`, and divergence improved 1.6e-12 to 5.1e-13 -- but it was not the
  mechanism. The `p_flux` fix was.
* **The term is orthogonal-only.** The cross part of the pressure flux is computed cell-centred
  and interpolated, so it carries the same wide-stencil blind spot. Measured cross/orthogonal
  magnitude: 10.6% at warp 0.02, 29.4% at 0.05, 61.6% at 0.10, 75.6% at 0.15. Expect the fix to
  degrade as skewness grows; a cross-term extension may be needed past warp ~0.10.
* **Pair with `implicit_cross` on skew grids.** With deferred correction, a non-converged pass
  now writes into a flux that persists rather than being wiped, so non-convergence becomes
  cumulative.
* **New restart state.** `p_flux` and `F_prev` are running state and are saved in checkpoint
  format 3. Neither is reconstructible from u and p.
