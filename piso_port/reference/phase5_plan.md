# Phase 5: PISO Orchestration — design, matched to the PICT C++ implementation

## Correction to the earlier assessment

I previously recommended a **staggered** refactor (velocities on faces). Reading the C++
shows that is **not** what PICT does, and we should not do it either.

PICT stores **both pressure and velocity at cell centres** — it is a *collocated* solver:

- `Block::CreatePressure()` -> `CreateDataTensor(1)`            (`domain_structs.cpp:1456`)
- `Block::CreateVelocity()` -> `CreateDataTensor(getSpatialDims())` (`domain_structs.cpp:1460`)

Both call `CreateTensor(1, channels, getSizes(), ...)` — the *same* grid size. Velocity is
simply a `dims`-channel cell-centred field.

The thing that is staggered in PICT is only the **metric transforms** (`m_faceTransform`,
"NCDHWT, staggered layout"), not the solution variables. My earlier "PICT is staggered"
conclusion came from grepping for the word *staggered* and finding it in the grid/transform
code — that was the wrong inference.

**Face fluxes are a derived quantity**, interpolated from cell-centred contravariant
components (`computeFluxesNDLoop`, `PISO_multiblock_cuda_kernel.cu:1553`):

```
fluxes[bound] = (velN + velC) * 0.5f;     // plain average, no Rhie-Chow
```

A grep for `rhie|chow` across the repo returns nothing — PICT does not use Rhie-Chow
interpolation. It instead keeps divergence and the pressure operator **both expressed on
faces**, which is what makes the projection consistent.

## Why this matters: measured

The 188x divergence floor I reported was an artefact of composing the **wide-stencil**
Phase-2 `divergence`/`gradient` operators with the **compact** Phase-4 Laplacian. Those are
different operators (9.9e-2 apart), so the correction cannot cancel the divergence.

Doing it PICT's way — divergence from face fluxes, and correcting those same fluxes with the
same compact face operator the pressure matrix is built from — the projection is exact:

| projection route | ||div|| after correction | reduction |
|---|---|---|
| wide-stencil div/grad (what I tested first) | 1.5e-02 | 188x |
| **PICT-style face fluxes** | **6.97e-13** | **4.4e12x** |

So Phases 1-4 do **not** need rebuilding. They already match PICT structurally.

## What Phases 1-4 already got right

`PISO_build_pressure_matrix` (`:4798`) turns out to be the same construction we built:

- `index_t indices[7]` — compact **7-point** stencil, as in `build_conservative_diffusion_matrix`
- `getLaplaceCoefficientOrthogonalDimSwitch` per dim — our `J*g11, J*g22, J*g33`
- face coefficient combines `alphaP` and `alphaN` — our face interpolation `0.5*(Jg[i]+Jg[i+1])`
- `nonOrthoFlags` — non-orthogonal terms handled separately = our **deferred correction**
- `raP = 1/Adiag` — the operator is `div( (1/A) grad p )`, which we must now add

The deferred-correction machinery, the conservative metrics, and the momentum assembly all
carry over unchanged.

## Work items

### 1. Face-flux module (new, small) — `phase5_fluxes.py`
- `compute_face_fluxes(u,v,w,J,metrics)` -> per-axis face arrays,
  `F = 0.5*(JU_C + JU_N)`, mirroring `computeFluxesNDLoop`.
- `divergence_from_fluxes(F,J)` -> mirrors `k_computePressureRHSdivergenceFromFlux` (`:5375`).
- These replace `compute_divergence` **inside the PISO loop only**. Phase 2's operators stay
  as they are — they are correct as differential operators and remain the MMS-validated
  reference; they are simply not the right thing to build a projection from.

### 2. Transient term in the momentum matrix
`SetupAdvectionMatrixEulerImplicit` (`:4483`) is Backward Euler, matching the plan's §2.
Add `rho/dt` to the diagonal of `build_momentum_matrix_7point` and the corresponding
`rho/dt * u^n` to the RHS. This *improves* diagonal dominance, so it should also cut the
deferred-correction iteration counts measured in Phase 3.

### 3. `1/A` weighting in the pressure matrix
`build_conservative_diffusion_matrix` currently uses coefficient `J*g`. PICT weights the face
coefficient by `1/Adiag`. Add an optional per-cell coefficient argument, face-interpolated the
same way, so the operator becomes `div((1/A) grad p)`. Phase 4's Poisson test keeps passing
by leaving the coefficient at 1.

### 4. Neumann / singular pressure
Walls are zero-gradient for pressure. Our assembled matrix **is already the Neumann operator**
(verified: row sums zero to 1.6e-12) and is therefore singular with a constant null space.
Needs: RHS compatibility projection (subtract the weighted mean) and either pinning one cell
or projecting the constant out each solve.

### 5. The PISO loop — `piso_numpy_3d.py`
Sequence and defaults taken from `PISOtorch_simulation.py:398-451`:
```
predictor:            solve A u* = rho/dt u^n + S        (Euler implicit)
for _ in range(corrector_steps):        # PICT default corrector_steps = 2
    fluxes   <- from u*
    solve    M p = div(fluxes)          # M = div((1/A) grad), Neumann
    correct  fluxes and u
```
Velocity correction: `u = hbyA - (1/Adiag) * grad p` (`PISO_update_velocity`, `:5948`), which
is PICT's default `velocity_corrector="FD"` (version 1). Versions 5/6 (`FVM_CENTER`,
`FVM_FACE`) use an FVM pressure gradient; version 4 exists in the kernel but is **not exposed**
in the Python driver. Start with FD to match the default, keep the flux correction compact.

### 6. Validation
- Divergence after each corrector — target machine precision on the fluxes (we measured 7e-13).
  Note the plan's "< 1e-7" criterion applies to the **flux** divergence; the cell-centred
  wide-stencil divergence will *not* reach it, and that is inherent to a collocated scheme,
  PICT included.
- 3D lid-driven cavity on a warped grid; 3D duct flow vs the analytical series solution.
- Reuse the `second_order` / meta-tested harness from `test_phase3_rigorous.py`.

## Open question for the user
The momentum predictor currently freezes the convecting velocity at the exact field (MMS
setup). PISO uses the previous iterate, so the Picard nonlinearity gets exercised for the
first time in Phase 5 — worth watching the iteration counts there.

---

# Outcome (implemented)

`phase5_fluxes.py`, `piso_numpy_3d.py`, `test_phase5_piso.py` — **10/10 checks pass.**
Phases 1-4 unchanged in behaviour (rates 2.05 / 2.07-2.12 / 2.15-2.46 / 2.14-2.30;
`test_phase3_rigorous.py` still 23/23).

| result | value |
|---|---|
| flux divergence after projection | **1e-11 – 2e-10** (plan criterion 1e-7) |
| divergence over a 40-step run | max 1.0e-10, no drift |
| Taylor-Green vs exact Navier-Stokes | converges, **~1st order** (see below) |
| lid-driven cavity | steady state, du 1.2e-01 -> 3.2e-04, no overshoot |
| usable grid warp | **up to ~0.15**, warns beyond |

## Three findings worth recording

**1. Boundary fluxes must come from the boundary velocity, not zero.** Defaulting them to
zero (impermeable) gave a discrete flux divergence of **1.8e+01** on an *exactly*
divergence-free Taylor-Green field; the projection then "corrected" that phantom divergence
and wrecked the interior (pressure 358 vs exact 0.24). `compute_face_fluxes` now takes
`boundary='from_velocity' | 'impermeable' | explicit`, matching PICT, where a Dirichlet
boundary enforces the flux from the boundary velocity. A closed cavity uses `'impermeable'`,
which also keeps the singular Neumann system exactly compatible.

**2. The scheme is 1st order, and that is expected.** With `apply_pressure_gradient=False`
(PICT's default) this is a non-incremental projection method, whose velocity Dirichlet BCs
generate a numerical boundary layer. Verified as the cause rather than assumed: the error
decays monotonically away from the wall (5.4e-2 -> 7.8e-3 over five layers) and does **not**
improve with more correctors (1/2/4 -> 1.25e-2/1.35e-2/1.48e-2). Recovering 2nd order needs an
incremental/rotational projection — PICT exposes the flag for it, we have not enabled it.

**3. Deferred correction has a hard warp limit near 0.18.** Measured contraction ratios at
n=16: 0.31 (warp 0.05), 0.59 (0.10), 0.92 (0.15), **1.27 (0.20 — divergent)**. Under-relaxation
does not lift it. Neither does boosting the implicit coefficient with an exactly compensating
explicit term: that leaves the fixed point unchanged and merely drives the iteration toward the
identity, converging *more slowly* — I implemented it, measured it, and removed it rather than
ship a remedy that does not work. Lifting the limit properly means making the cross terms
implicit (19- or 27-point matrix). Past the limit the solver warns instead of returning a
quietly wrong field (asserted by T5).

## Not done
- Incremental/rotational projection (would restore 2nd order).
- Duct flow vs the analytical series solution: needs periodic BCs, which the port does not
  have. Taylor-Green against the exact unsteady Navier-Stokes solution covers the same ground
  and does not require them.
