# Inflow and outflow boundary conditions

## 1. What the solver could not do

Before this work the solver handled closed and periodic domains only, and three things blocked
an outflow — they compound, so none could be fixed alone:

1. **The pressure system is pure Neumann and singular.** `_solve_pressure` builds no face at any
   domain boundary, so row sums vanish, the matrix has a constant null space, and one cell is
   pinned to remove it.
2. **RHS compatibility is assumed.** `b = b - b.mean()` silently projects out any imbalance. With
   a genuine outflow the net boundary flux is *not* zero and should not be, so that projection
   would absorb the physics and return a plausible wrong field.
3. **Velocity BCs were Dirichlet-only** — no zero-gradient or convective path.

## 2. What PICT actually does — and what it does *not* do

Reading `PISOtorch_simulation.py` and the shipped samples changed the design. PICT does **not**
add a Dirichlet-pressure branch, drop the pin, or drop the mean projection. It keeps the pressure
system pure-Neumann and singular exactly as ours is, and handles outflow entirely **outside** the
solve, in a `prep_fn` callback each step. Outflow boundaries stay ordinary **time-varying
Dirichlet velocity** boundaries.

Two functions carry it:

**`update_advective_boundaries`** solves $\partial u_b/\partial t + U_c\,\partial u/\partial n = 0$
by an implicit blend toward the adjacent interior cell:

$$\alpha = \frac{2\,\Delta t\,U_c}{\Delta n},\qquad t = 1 - \frac{1}{1+\alpha},
\qquad u_b \leftarrow u_b - t\,(u_b - u_{\rm int})$$

The implicit form is the clever part: $t \in [0,1)$ for **any** $\alpha \ge 0$, so the update
cannot go unstable however large the CFL.

**`balance_boundary_fluxes`** sums the flux over fixed boundaries (inlet, walls) and over the
free (outflow) ones, and if they do not cancel to 0.01× solver tolerance, scales the outflow
velocities by $-\Phi_{\rm fixed}/\Phi_{\rm free}$. This is what keeps the **singular Neumann
system compatible** — enforcing the compatibility condition by construction instead of changing
the pressure boundary condition. It runs at setup *and* after every advective update.

### The factor of 2 that does not transfer

PICT writes $\alpha = 2\Delta t\,U_c/h$ because it is **cell-centred**: the centre-to-face
distance is $h/2$. **Our nodes sit ON the boundary**, so the distance to the first interior node
is $h_n$ and the factor 2 is absent:

$$\alpha = \frac{\Delta t\,U_c}{h_n}$$

Copying their formula verbatim would advect the outlet twice too fast. This is the kind of
detail that transfers silently and wrongly between a cell-centred and a node-on-boundary code.

## 3. What was implemented

`outflow.py`, with two treatments.

### convective (PICT's)

Outlet stays a Dirichlet velocity boundary, advected as above, with `balance_boundary_fluxes`
keeping the singular system compatible. Faithful to PICT apart from the factor 2.

### dong (Dong, Karniadakis & Chryssostomidis, JCP 261 (2014) 83–105)

An **energy-stable** open boundary. The traction condition

$$\nu\frac{\partial \mathbf{u}}{\partial n} - p\,\mathbf{n}
- \tfrac{1}{2}|\mathbf{u}|^2\,\Theta(\mathbf{n},\mathbf{u})\,\mathbf{n} = 0,
\qquad \Theta = \tfrac{1}{2}\Big(1 - \tanh\frac{\mathbf{n}\cdot\mathbf{u}}{U_0\delta}\Big)$$

with $\Theta \to 1$ in backflow and $\to 0$ in outflow. Dotting with $\mathbf{n}$ splits it, for a
projection scheme, into a **Dirichlet pressure** plus zero normal-gradient velocity:

$$p_{\rm out} = \nu\,\frac{\partial(\mathbf{u}\cdot\mathbf{n})}{\partial n}
- \tfrac{1}{2}|\mathbf{u}|^2\,\Theta$$

The $-\tfrac12|\mathbf{u}|^2\Theta$ term is the whole point: it makes the boundary contribution to
$\mathrm{d}E/\mathrm{d}t$ non-positive even where fluid flows back **into** the domain, so
vortices crossing the outlet cannot pump energy in.

**Why this needs no new pressure assembler.** Our grid puts nodes *on* the boundary, so a
Dirichlet pressure is node **elimination** — exactly what `phase4_poisson.py` already does for its
Dirichlet walls. Eliminating the outlet nodes also makes the reduced matrix **non-singular**, so
there is no pinned cell, no compatibility projection, and no flux balancing. Their continuity
equation is dropped, which is precisely what lets mass leave as the solution dictates.

The mean-subtraction in `_solve_pressure` is now skipped whenever Dirichlet pressure nodes are
present — with an outflow it would remove the genuine imbalance the outlet exists to carry.

### One detail that bites: wall corners

An outlet's end nodes are also no-slip **wall** nodes. Without the `hold` mask the advective
update overwrites them with interior velocity, turning the wall into a slip surface exactly where
the shear is largest, and the Poiseuille profile then fails to hold.

## 4. Results

Fully developed Poiseuille with prescribed parabolic inflow — the exact solution is the parabola
everywhere, so a correct outlet must not distort it.

| BC | 17×13 | 33×25 | 65×49 | mass imbalance |
|---|---|---|---|---|
| convective | **2.29e-08** | **2.71e-07** | **3.34e-08** | 1.7e-16 |
| dong | 1.10e-06 | 5.35e-05 | 6.32e-05 | 1.7e-08 |

**convective is correct.** dong is not, and the error *grows* under refinement.

### A finding that is not about the outflow at all

With `scheme='chorin'` the steady state carries an $O(\Delta t)$ error that does **not** vanish as
the flow converges: the predictor drops $\nabla p$, so the converged state solves
$A\mathbf{u} = \mathrm{rhs} - \Delta t\,A\nabla p$ instead of
$A\mathbf{u} = \mathrm{rhs} - J\nabla p$. Measured: 5.47e-3 / 2.78e-3 / 1.40e-3 at
$\Delta t$ = 0.04 / 0.02 / 0.01, grid-independent. The accumulating schemes keep $\nabla p^n$ in
the predictor and drop it to ~3e-7:

| scheme | dt=0.04 | dt=0.02 | dt=0.01 |
|---|---|---|---|
| chorin | 5.47e-3 | 2.78e-3 | 1.40e-3 |
| incremental | 3.63e-7 | 3.07e-7 | 2.25e-7 |
| rotational | 3.36e-7 | 2.71e-7 | **1.84e-7** |

This is a property of Chorin, not of the boundary, but it masquerades as a boundary-condition
error in any steady test. These tests use `rotational`.

## 5. The Dong defect, diagnosed

**The first hypothesis was wrong.** The `KNOWN DEFECT` note originally blamed the explicit
$\nu\,\partial u_n/\partial n$ term, reasoning that it closes a feedback loop of gain
$\nu\Delta t/\Delta n^2$ = 0.016 / 0.064 / 0.256 across the three grids — growing four-fold per
refinement and heading for instability. Two pieces of evidence killed it.

**The error shape.** 1.10e-6 → 5.35e-5 → 6.32e-5 grows and then *flattens*. That is convergence
to a fixed non-zero limit — a **consistency** error. A feedback instability with gain approaching
1 would keep growing.

**Direct isolation**, running each term alone:

| variant | 17×13 | 33×25 | 65×49 | |
|---|---|---|---|---|
| full (visc + Θ) | 8.51e-7 | 3.00e-5 | 4.16e-5 | grows |
| **viscous only** | 2.22e-7 | 5.26e-8 | 3.51e-7 | no growth |
| **Θ only** | 8.51e-7 | 3.00e-5 | 4.16e-5 | **identical to full** |

The Θ term accounts for the entire defect. The viscous term contributes at the 1e-7 level and
causes no growth — as it must for Poiseuille, where $\partial u/\partial x = 0$ exactly.

### The mechanism

$\Theta = \tfrac12(1-\tanh(u_n/U_0\delta))$ tends to $\tfrac12$ wherever $u_n \to 0$ — which is
**every wall-adjacent node**, since the outflow meets no-slip walls. For exact Poiseuille the
correct outlet pressure is uniform, but the term imposes a spurious near-wall pressure:

| $\delta$ | 0.20 | 0.10 | 0.05 | 0.02 | 0.01 | 0.005 |
|---|---|---|---|---|---|---|
| peak spurious $p$ | 2.41e-3 | 6.04e-4 | 1.51e-4 | 2.41e-5 | 6.04e-6 | 1.50e-6 |

It scales as $\delta^2$. At the default $\delta = 0.05$ the peak is 1.51e-4, or **9.4e-5 relative**
to the channel's 1.6 pressure drop — the same size as the observed 6.3e-5 error.

So this is **not an implementation bug** but an inherent property of Dong's regularised switch
where an open boundary meets a no-slip wall: $\Theta$ does not vanish where $u_n \to 0$, leaving a
thin near-wall layer with a spurious traction of size $O(U_0^2\delta^2)$. $\delta$ trades
energy-stability robustness against a near-wall consistency error.

## 6. What is NOT established

**Dong's advantage over convective is undemonstrated here.** The backflow test — a single Gaussian
vortex advected out at Re=200 — did not discriminate: both stayed stable, energy ratios 0.6962 vs
0.6963, and both produced backflow (33% and 18% of the outlet at different instants). The fields
*do* differ by 11% of peak speed near the outlet, but with no reference solution neither can be
called more correct.

The test was too benign: one vortex crosses once and is gone. A discriminating test needs
**sustained** shedding from a bluff body, so vortices arrive at the outlet at full strength
indefinitely — which needs multi-block geometry. There is also no long-domain reference run, which
is the standard way to judge an outflow condition: rerun with the outlet far downstream and
compare each short-domain result against that solution truncated to the same region.

## 7. Status

| | state |
|---|---|
| convective outflow | **working** — exact Poiseuille to 3e-8, mass conserved to 1e-16 |
| Dong outflow | **runs and is stable, but not consistent at the default $\delta$** |
| Dong root cause | **identified** — the $\Theta$ term, $O(U_0^2\delta^2)$ near a no-slip wall |
| Dong remedy | reduce $\delta$; verification in progress |
| backflow discrimination | **not achieved** — needs sustained shedding |
