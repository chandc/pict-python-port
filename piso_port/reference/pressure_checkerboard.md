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

## Implementation

Enabled by `rhie_chow=True` on both solvers (default `False`, so existing results are
reproducible):

* `PISOSolver(..., rhie_chow=True)` — `src/piso_numpy_3d.py`, applied where
  `compute_face_fluxes` forms `F`.
* `MultiBlockPISO(..., rhie_chow=True)` — `src/piso_multiblock.py`, applied per block after
  `Domain.face_fluxes`.

Both call `pressure_face_fluxes(..., rhie_chow=True)`, which returns the dissipation term
instead of the pressure flux, using the **current** pressure and the **same** Γ the pressure
equation uses.

Note the correction is applied once to the base flux built from velocity — it is **not** applied
to the corrector's φ flux, which is an increment, not a pressure.

## What to watch for

* The correction is O(h³) and so must **not** change a smooth solution beyond truncation error.
  That is the gate: order-of-accuracy tests must be unchanged with it on.
* It uses Γ from the momentum matrix, so it inherits Γ's behaviour; with `pressure_coef='diag'`
  versus `'rowsum'` the magnitude differs, though the null-mode targeting does not.
* At a seam the wide gradient is evaluated on the padded field, so the face adjacent to the pad
  edge uses a one-sided `np.gradient` stencil. The term is O(h³) there regardless, but it is a
  small inconsistency worth remembering if seam-local pressure behaviour is ever in question.
