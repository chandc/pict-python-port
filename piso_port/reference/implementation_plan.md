# Implementation plan — as originally written, and as actually followed

The original plan is preserved in §1–§2 because the gap between it and what happened is the most
useful thing in this document. §3 records the route actually taken; §4 is the operational part —
default settings and what to watch for to get second order on **non-linear** flows.

---

## 1. The original plan (five phases, each gated by MMS)

| Phase | Content | Gate |
|---|---|---|
| 1 | 3D mesh and metric generation | MMS on a wavy grid; GCL |
| 2 | Gradient and divergence operators | MMS, trigo-exponential fields |
| 3 | Momentum matrix assembly | MMS, 3D Taylor-Green |
| 4 | Poisson matrix and solver | MMS, trigo-exponential Laplacian |
| 5 | PISO orchestration and validation | lid-driven cavity vs Ghia |

Single domain, collocated, curvilinear. The plan was sound and all five phases were completed.

## 2. What the plan got wrong about itself

**It assumed MMS was sufficient.** Every phase gate is a manufactured solution or a steady
comparison. Those verify that each operator approximates its differential counterpart, and they
are blind to:

- whether the convective operator **conserves kinetic energy** (§4.3) — nothing in the plan tests it
- whether the scheme is second order **once convection is active** (§4.2) — every gate is
  effectively linear
- whether the *spectrum* of the linear operator is right (`stokes_verification.md`)

**It assumed passing meant correct.** Phase 3 passed at grid warp 0.01 and collapsed to
convergence rate **0.42** at warp 0.10 — the split between the implicit 7-point operator and the
explicit cross terms was inconsistent, and a weak test hid it. The fix was the volume-integrated
conservative form.

**"Staggered" in the title was wrong.** The plan specified a staggered grid; reading PICT's C++
showed it is **collocated**. Corrected in Phase 5.

---

## 3. The route actually followed

Phases 1–5 as planned, then everything below, none of which was foreseen:

| # | Work | Why it happened |
|---|---|---|
| 6 | Periodic BCs on all axes; rotational correction | needed before any spectral test |
| 7 | Pressure coefficient fix (`rowsum` vs `diag`) | a Re=10 channel diverged; see §4.5 |
| 8 | Implicit cross terms + ILU preconditioning | deferred correction cost 121–2584 sweeps |
| 9 | Stokes eigenvalue verification (periodic and walled) | MMS cannot check a spectrum |
| 10 | Inviscid energy-conservation check | the one property turbulence actually needs |
| 11 | Inflow / outflow, convective and Dong | no open boundaries existed |
| 12 | Multi-block: geometry → operators → PISO step | PICT's defining feature |

**Four published claims were retracted along the way**, each after a measurement contradicted an
earlier one. They are documented in place rather than quietly edited, because the *reason* each
was wrong is more useful than the corrected number:

- "not second order in time with walls" — measured outside the asymptotic range
- "the deferred-correction warp limit is a solver property" — the grid tangles at the same warp
- "the full operator stays symmetric" — true only for periodic BCs
- an Orr-Sommerfeld extrapolation reported as 0.00% — circular, honest answer 4.5%

---

## 4. Getting second order on non-linear flows

This is the operational section. Full detail in
[`second_order_convective.md`](second_order_convective.md).

### 4.1 Default settings

```python
PISOSolver(...,
    convection='central',      # SOU removes ~10% of KE per turnover on broadband fields
    scheme='rotational',       # chorin is O(dt) regardless of the predictor
    time_scheme='bdf2',
    picard_iters=2,            # removes the O(dt) lag in the convecting velocity
    pressure_coef='rowsum',    # diag under-corrects by 1 + 2 nu dt sum(1/h^2)
    implicit_cross=True,       # warped grids only; it LOSES on near-orthogonal ones
    momentum_dc_iters=2,       # momentum cross-diffusion, warped grids
)
```

`MultiBlockPISO` defaults to `scheme='rotational', picard_iters=2` for the same reasons.

### 4.2 The two O(Δt) errors that sit in front of BDF2

| configuration | measured order |
|---|---|
| `chorin` + BDF2 | 0.93, 0.89, 0.95 |
| `rotational` + BDF2, `picard_iters=1` | 1.32, 1.20, 1.08 |
| **`rotational` + BDF2, `picard_iters=2`** | **2.19, 2.16, 2.09** |

BDF2 is in front of neither. **Chorin's splitting error** is first order whatever the predictor,
and the momentum matrix assembled from the lagged $u^n$ is a second first-order error. A third
Picard iteration buys nothing — the signature of a converged fixed point rather than a knob.

> This is invisible in a near-linear test. `stokes_verification.md` measured **2.00** at
> `picard_iters=1` because the perturbation amplitude was $10^{-4}$: convection was negligible,
> so the lag cost nothing. **Second order in a linear test does not imply second order with
> convection.**

### 4.3 `central` is required, not preferred

Inviscid energy production $P = \sum_c \mathbf{u}_c\cdot(C\mathbf{u}_c)\,\mathrm{d}V$ at $n=48$:

| | Taylor-Green | random solenoidal |
|---|---|---|
| `central` | −5.8e-18 | +1.2e-16 |
| `sou` | 5.36e-3 (0.54%/turnover) | 9.60e-2 (**10.35%**/turnover) |

`central`'s convective operator is discretely skew-symmetric. `sou` removes 10% of the kinetic
energy per eddy turnover on a broadband field — and any real convective flow is broadband. For an
LES or a stability calculation that dissipation swamps the physics: the Orr-Sommerfeld growth
rate at $Re=7500$ is $2.2\times10^{-3}$ per unit time, and SOU could flip its sign.

### 4.4 Spatial order needs a consistent operator split

- **Volume-integrated conservative form.** Using cell-centred $g^{11}$ without the Jacobian drops
  an $O(1)$-in-warp term and makes the implicit/explicit split *inconsistent*: the error plateaus
  instead of converging (rate 0.42 at warp 0.10).
- **Cross terms**, on warped grids, in **both** operators — pressure *and* momentum. Getting only
  the pressure one right produced a perfectly divergence-free field (6.7e-14) that was wrong by
  5.5e-2 in velocity.

### 4.5 The pressure coefficient

$\Gamma = J/A_{\rm diag}$ under-corrects by $1 + 2\nu\Delta t\sum 1/h^2$, because the conservative
diffusion and SOU advection operators have **exactly zero row sum** (7e-14 on warped grids).
Chorin survives it; the accumulating schemes feed the deficit back each step and diverge — which
is how a Re=10 channel blew up. `pressure_coef='rowsum'` gives $\Gamma = \Delta t$ exactly.

### 4.6 Things to watch for when measuring the order

Five traps, each of which produced a wrong published conclusion here first:

1. **Richardson on differences, not total error** — the spatial error is a fixed offset that
   swamps the temporal one.
2. **Verify the asymptotic range.** One triple is not a measurement: orders read
   0.80 → 1.68 → **2.00** across successively finer triples of the same study.
3. **Spatial and temporal errors can have opposite signs.** In a 5×5 $(n,\Delta t)$ matrix the
   best entry was the *coarsest* $\Delta t$ at the finest grid; refining $\Delta t$ made it worse.
   Refine both together.
4. **Do not measure inside a startup transient** — a continuous eigenmode is not a discrete one.
   Reported order 0.04 for a scheme that is ~2.
5. **Never derive the order from the reference value.** Defining the error as
   $G_{\rm ref}-G(n)$ and extrapolating with *its* order cannot miss $G_{\rm ref}$.

**And the meta-lesson:** a plausible physical explanation for a shortfall is what stops you
checking whether the measurement converged. The retracted 1.68 had a real mechanism attached
($O(\Delta t^{3/2})$ near-wall splitting error) and was still just a pre-asymptotic triple.

### 4.7 What second order does *not* buy

It is a statement about the **rate** at which error vanishes, not its size at affordable
resolution. The Orr-Sommerfeld residual numerical damping is 1.09e-3 / 6.4e-4 / 2.5e-4 at
$n_y = 97/129/201$ — converging at the design rate, and still comparable to the 2.2e-3 growth
rate being measured.

---

## 5. Planned work, and what it costs

Two extensions are scoped but not built. Both interact with the Rhie–Chow work in
[pressure_checkerboard.md](pressure_checkerboard.md), so the interactions are recorded here
rather than discovered later.

### 5.1 The adjoint, after Rhie–Chow

**Nothing is broken today, for a reason that is itself a hazard.** The adjoint path
(`src/piso_torch.py`, `src/adjoint_piso.py`) is a *separate implementation*, not a wrapper over
the NumPy solvers: it builds its own divergence and gradient matrices and never calls
`face_fluxes` or `pressure_face_fluxes`. So adding Rhie–Chow left it untouched.

The hazard is that the two paths now **discretise differently when `rhie_chow=True`**. Before,
they agreed in structure — plain-average face flux, wide pressure gradient, both. Train an SGS
model through the torch path and validate on the NumPy path with the option on, and the two are
solving different equations. That is exactly the kind of mismatch that surfaces as an
unexplained a-posteriori gap with no obvious cause. RC defaults off, so nothing is wrong now;
this must be checked before any learning run turns it on.

Bringing RC into the adjoint is three new gradient paths, and they are not equally cheap:

| path | what it is | cost |
|---|---|---|
| `p -> F` | new dependency: F previously depended only on u. `D` is linear in p, so the backward is that operator transposed | cheap |
| `p_flux` recurrence | `p_flux^{n+1} = p_flux^n + phi` is a new accumulator across steps; the adjoint gains a matching backward accumulator | cheap |
| `F_prev` recurrence | `ddt_corr` makes step *n* depend on step *n-1*'s converged FLUX, so gradients must propagate backward through time via F as well as u and p | **expensive** |

The third roughly doubles the per-step tape state: three face-flux arrays of ~N each, plus an
extra recurrence chain in the backward pass. Cost that one first before committing.

**A multi-block consequence of the width-2 fix.** The Rhie–Chow wide gradient now pads to
width 2 (it had to — a width-1 pad is one-sided exactly at the ghost a seam face needs, which
broke mass conservation). So the stencil reaches **two cells deep across a seam**, and the
adjoint scatter at a connection is wider than the existing width-1 machinery assumes. The
multi-domain adjoint is unbuilt, so this is a design input rather than a repair.

### 5.2 Variable effective viscosity (LES)

The diffusion operator is most of the way there already, by accident of good design.
`Jg_of` in `build_momentum_matrix` returns `nu * Js[b] * g` — a per-cell array — and every face
uses `0.5*(Jg[lo] + Jg[hi])`. Make `nu` a per-block array and that line broadcasts, the face
average becomes the symmetric average of `nu_eff * J * g`, symmetry is preserved **by
construction**, and it works across seams unchanged. `build_diffusion_matrix` already accepts
per-block `coefs` arrays and is exercised in that mode by the pressure solve, so the
variable-coefficient machinery is not merely present but verified.

What remains splits into plumbing and one real physics gap:

* **Plumbing** (~half a day). The deferred cross term `nu * cross`, the rotational correction
  `- nu * div(u*)`, and the Dong outflow's `nu` are all scalar multiplies that become
  elementwise.
* **Physics — the transpose term.** Our operator is `div(nu grad u)`. The full stress is
  `div(nu_eff (grad u + grad u^T))`. For constant `nu` the transpose part vanishes by
  continuity; with variable `nu_t` it leaves `grad(nu_t) . (grad u)^T`, which is **not**
  negligible and has to be added. This is the only part that is new physics rather than
  threading an array through.
* **The model itself.** Smagorinsky needs `|S|` from the full velocity-gradient tensor; the
  curvilinear metrics are all present and `pad_field` handles the seams, so it is assemblable.
  `Delta = J^(1/3)` is available. The harder pieces are wall damping (needs a wall-distance
  field, not currently computed) and the dynamic procedure — a test filter crossing seams is
  feasible with the existing padding, but the homogeneous-direction averaging that makes it
  stable is case-specific.

**Interaction with Rhie–Chow.** `Gamma = J / rowsum(A)` and A carries the diffusion, so a
spatially varying `nu_eff` makes `Gamma` vary more sharply than it does today. The Rhie–Chow
dissipation scales with `Gamma`, so its strength becomes a function of the local eddy
viscosity — largest where the SGS model is most active. That is probably benign and possibly
helpful, but it is untested and should be measured, not assumed.

**Precedent.** Upstream PICT does exactly this for its TCF SGS-learning example, so there is a
reference implementation for the variable-`nu` path if we want one.
