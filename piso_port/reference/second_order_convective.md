# Keeping second order when convection is active

Most of this repo's accuracy work was done on problems where the convective term is negligible —
MMS fields, Stokes eigenmodes at amplitude $10^{-4}$, steady Poiseuille and duct flow. Those
establish that the operators and the time integrator are individually second order, and they are
*not sufficient*: several first-order errors are invisible until convection is doing real work.

This document collects what actually has to be true, in the order it bites, with the measurement
that established each. Every number here is measured in this repo.

---

## The headline

| configuration | measured temporal order |
|---|---|
| `chorin` + BDF2 | 0.93, 0.89, 0.95 |
| `rotational` + BDF2, `picard_iters=1` | 1.32, 1.20, 1.08 |
| **`rotational` + BDF2, `picard_iters=2`** | **2.19, 2.16, 2.09** |
| `rotational` + BDF2, `picard_iters=3` | 2.19, 2.16, 2.09 (no further gain) |

**BDF2 is in front of neither error.** Two separate $O(\Delta t)$ terms cap the scheme, and both
must be removed. Measured on a periodic Taylor-Green-like flow with $\nu = 0.05$, central
convection, and confirmed identical single-block and multi-block.

---

## 1. Time: the projection must be incremental

`chorin` recomputes the pressure from scratch each step and never carries $\nabla p^n$ into the
predictor. Its splitting error is $O(\Delta t)$ **regardless of the predictor's order**, so BDF2
buys nothing on top of it.

The same defect shows up in a completely different measurement — a steady one. Fully developed
Poiseuille through an inflow/outflow channel, where the exact answer is the parabola everywhere:

| scheme | dt=0.04 | dt=0.02 | dt=0.01 |
|---|---|---|---|
| `chorin` | 5.47e-3 | 2.78e-3 | 1.40e-3 |
| `incremental` | 3.63e-7 | 3.07e-7 | 2.25e-7 |
| `rotational` | 3.36e-7 | 2.71e-7 | **1.84e-7** |

The error is **grid-independent and proportional to $\Delta t$** — it does not vanish as the flow
converges, because the converged state solves
$A\mathbf{u} = \mathrm{rhs} - \Delta t\,A\nabla p$ instead of
$A\mathbf{u} = \mathrm{rhs} - J\nabla p$. Four orders of magnitude, from one flag.

> This masquerades as a boundary-condition error in any steady test. It cost real time during the
> outflow work before being recognised as Chorin's, not the outlet's.

`rotational` additionally applies $p \leftarrow p + \phi - \nu\,\nabla\!\cdot\mathbf{u}^*$, which
cancels the spurious $\partial p/\partial n = 0$ the projection otherwise imposes at walls.

## 2. Time: the convecting velocity must not lag

The momentum matrix $A$ is assembled from the convecting velocity. Using $\mathbf{u}^n$ is an
$O(\Delta t)$ lag, and it caps the scheme at first order once convection matters — the
`picard_iters=1` row above.

The fix is to repeat the step with $A$ rebuilt from the latest $\mathbf{u}^*$, **restoring the
starting state each time so time advances only once**. One extra iteration removes it; a third
buys nothing, which is the signature of a converged fixed point rather than a tuning knob.

### Why this was missed for so long

`stokes_verification.md` measured order **2.00 with `picard_iters=1`** and stated the scheme was
second order in time. That measurement used a perturbation amplitude of $10^{-4}$, so the
convective term was $O(A^2)$ against $O(A)$ linear terms — **the lag cost nothing because there
was nothing to lag**. Amplitude-independence was even verified there (halving the amplitude moved
$\sigma$ by 4.5e-07), which confirms the Stokes limit and *simultaneously* confirms that the test
could not see this error.

The second-order claim therefore holds for near-linear flows and **not** for convectively driven
ones unless `picard_iters=2`. A qualification the original claim did not carry.

## 3. Space: the convective scheme decides whether energy is conserved

For $\nu = 0$ and periodic boundaries, convection must redistribute kinetic energy without
creating or destroying it. Discretely that requires the convective operator to be
**skew-symmetric**. Measured directly as
$P = \sum_c \mathbf{u}_c\cdot(C\,\mathbf{u}_c)\,\mathrm{d}V$ with $C$ the $\nu=0$ operator:

| $n=48$ | Taylor-Green | random solenoidal |
|---|---|---|
| `central` | **-5.8e-18** | **+1.2e-16** |
| `sou` | 5.36e-3 (0.54%/turnover) | 9.60e-2 (**10.35%**/turnover) |
| `sou` convergence | $h^{2.99}$ | $h^{2.86}$ |

`central` conserves to round-off. `sou` removes **10.35% of the kinetic energy per eddy turnover**
on a broadband field against 0.54% on a smooth one — upwind dissipation scales with small-scale
content, and any real convective flow is broadband.

**Consequence:** `convection='central'` is required for anything where dissipation matters. It is
not a preference. The Orr-Sommerfeld growth rate at $Re=7500$ is $2.2\times10^{-3}$ per unit
time; SOU's dissipation would swamp it outright and could flip the reported sign of stability.

> The multi-block momentum assembler **raises** for `convection='sou'` rather than falling back to
> central, because a silent fallback changes the physics, not merely the accuracy.

## 4. Space: the operator split must be consistent

Two things that silently destroy second order on curvilinear grids, both found the hard way:

**The volume-integrated form is load-bearing.** A discretisation using the cell-centred
coefficient $g^{11}$ with no Jacobian weighting drops the term from differentiating $(Jg^{11})$.
That term is $O(1)$ in the grid warp, so combining it with a $J$-weighted cross-diffusion gives an
*inconsistent* operator: the error plateaus under refinement instead of falling. Phase 3 passed at
warp 0.01 for this reason and collapsed to rate **0.42** at warp 0.10.

**The pressure coefficient must use the row sum.** $\Gamma = J/A_{\rm diag}$ under-corrects by
$1 + 2\nu\Delta t\sum 1/h^2$, because the conservative diffusion and SOU advection operators have
*exactly zero row sum* (verified to 7e-14 on warped grids). Chorin survives it — it replaces the
pressure rather than accumulating — but the incremental schemes feed the deficit back every step
and diverge. `pressure_coef='rowsum'` gives $\Gamma = \Delta t$ exactly.

## 5. Space: non-orthogonal terms

On a warped grid the cross terms carry real error. Deferred correction is a Picard iteration whose
contraction ratio grows with warp (0.31 / 0.59 / 0.92 at warp 0.05 / 0.10 / 0.15), so it costs
sweeps but converges to the right answer. `implicit_cross=True` folds them into the operator and
is 3.7×/6.8×/16.2× faster at those warps — same answer to 1e-13, verified.

Both preserve second order; the choice is cost, not accuracy. On a *near-orthogonal* grid the
implicit path loses (0.12×–0.93×), correctly, since there are no cross terms to make implicit.

---

## How to measure the order without fooling yourself

Five traps, every one of which produced a wrong published conclusion in this repo before being
caught.

**1. Use Richardson on differences, not total error.** The spatial error is a fixed offset that
swamps the temporal one. Fit the total error against $\Delta t$ and you measure the wrong thing.

**2. Verify you are in the asymptotic range.** A single Richardson triple is not a measurement.
The temporal order was published as **1.68** with a plausible physical explanation attached (the
$O(\Delta t^{3/2})$ near-wall splitting error of rotational projection). Extending the sweep:

| triple ($\Delta t$) | 4e-4, 2e-4, 1e-4 | 2e-4, 1e-4, 5e-5 | 1e-4, 5e-5, 2.5e-5 |
|---|---|---|---|
| order | 0.80 | 1.68 | **2.00** |

It was pre-asymptotic. **A physical explanation for a shortfall is worth nothing until the
measurement is shown to be converged** — the explanation is what stopped the checking.

**3. Spatial and temporal errors can have opposite signs.** In the 5×5 $(n, \Delta t)$ matrix for
the wall-bounded Stokes case, the spatial error over-damps and the temporal under-damps. The
single best entry is the **coarsest** time step at the finest grid, and refining $\Delta t$ from
there nearly doubles the error. Refine both together; a single-parameter sweep stalls or reverses.

**4. Do not measure inside a startup transient.** An initial field built from a continuous
eigenmode is not an eigenmode of the *discrete* operator and relaxes onto it first. Measuring over
$[0, 0.02]$ reported order **0.04** for a scheme that is genuinely ~2. Use a settled window.

**5. Never derive the convergence order from the reference value.** Defining the error as
$G_{\rm ref} - G(n)$, measuring *its* order, then extrapolating with that order is circular — it
cannot miss $G_{\rm ref}$ whatever the data says. It produced a flattering **0.00%** on the
Orr-Sommerfeld growth rate where a genuine three-point extrapolation gives order 1.66 and
**4.5%**.

---

## Checklist for a convective run

```python
PISOSolver(...,
    convection='central',      # SOU removes ~10% of KE per turnover on broadband fields
    scheme='rotational',       # chorin is O(dt) regardless of the predictor
    time_scheme='bdf2',
    picard_iters=2,            # removes the O(dt) lag in the convecting velocity
    pressure_coef='rowsum',    # diag under-corrects by 1 + 2 nu dt sum(1/h^2)
    implicit_cross=True,       # only on warped grids; it loses on near-orthogonal ones
)
```

`MultiBlockPISO` defaults to `scheme='rotational', picard_iters=2` for this reason.

**What this does not fix.** These settings deliver the design order; they do not make the scheme
non-dissipative. The Orr-Sommerfeld test measures the residual numerical damping directly:

| $n_y$ | 97 | 129 | 201 | 401 |
|---|---|---|---|---|
| growth rate | 0.001141 | 0.001595 | 0.001983 | **0.002209** |
| error vs 0.002235 | 49.0% | 28.6% | 11.3% | **1.2%** |

Converging, but comparable to the 2.2e-3 growth rate itself until the grid is fine. Second order
is a statement about the *rate* at which error vanishes, not about whether it is small at the
resolution you can afford.

That study also shows the two error sources splitting cleanly **by variable**: the growth rate is
spatial-limited (201→401 gained 9.4×, halving $\Delta t$ gained nothing) while the phase speed is
temporal-limited (flat at 1.4–1.5% across $n_y$ = 97→401, moving only with $\Delta t$: 2.59% /
1.50% / 0.94%). And it shows the opposite-sign trap in the wild — the sign of the
$\Delta t$-sensitivity **flips** between $n_y$=201 (refining $\Delta t$ makes it worse) and
$n_y$=401 (refining makes it better), which is two errors of opposite sign crossing between the
two grids. The implied order 201→401 is 3.28, above the design order, so partial cancellation is
still flattering the finest point.
