# Connecting a CNN to PISO — staged implementation and verification plan

Mirrors what PICT does (`velocitySource` / `m_viscosity` hooks, discrete adjoint through every
solver stage, training against reference data). Built in six stages, smallest first, each with
a **gate** that must pass before the next stage starts.

The governing principle: **a wrong gradient still trains, just to the wrong place.** Every
stage's gate is therefore a gradient check, not a falling loss curve.

Theory in [`nn_piso_coupling.md`](nn_piso_coupling.md). Stage 0 is already done.

---

## Stage 0 — the linear-solve adjoint ✅ done

**Build.** `LinearSolve(torch.autograd.Function)`: forward solves $A\mathbf{x}=\mathbf{b}$ and
saves $(A_{\text{val}}, \mathbf{x})$; backward solves $A^{\mathsf T}\lambda = \bar{\mathbf g}$
and returns $\partial L/\partial\mathbf b = \lambda$, $\partial L/\partial A = -\lambda\mathbf{x}^{\mathsf T}$
on the sparsity pattern only.

**Gate — passed** ([`adjoint_piso.py`](../adjoint_piso.py)):

| check | result |
|---|---|
| adjoint identity on the **non-symmetric** momentum matrix | exact to 1e-10 (and forgetting the transpose is 24.5% off) |
| finite differences through a full PISO step | agree to ~7 digits |
| pressure gradient invariant to a constant shift of $\bar{\mathbf g}$ | 8.9e-16 |

---

## Stage 1 — one step, one scalar parameter

The smallest thing that can possibly be trained: a single learnable scalar $c$ scaling a fixed
forcing shape, $S = c\,\Phi(\mathbf{x})$, injected into the momentum RHS. No network yet.

**Build.** `nn_hooks.py`: a `MomentumSource` wrapper that adds $J\,S$ to the predictor RHS and
routes the gradient back through `LinearSolve`. Mirrors PICT's `velocitySource` +
`SetupAdvectionVelocityEulerImplicitRHS_GRAD`.

**Gate.**
1. $\partial L/\partial c$ matches central finite differences to 6 digits (float64).
2. **Recovery test** — set $c_\text{true}=0.7$, generate a target field, start from $c=0$, and
   confirm gradient descent recovers $0.7$ to 4 digits. This is the first end-to-end proof that
   the sign and scale of the gradient are right, not just its magnitude.

*Why this first:* if the sign is flipped, a 1-parameter recovery makes it obvious immediately,
whereas a CNN would just train to something plausible-looking.

---

## Stage 2 — one step, a small CNN

Replace the scalar with a genuinely small network: 2 conv layers, ~200 weights, taking
$(\mathbf{u}^n)$ and predicting a 3-component source field.

**Build.** `sgs_net.py` — a 3D CNN with periodic padding (matching the solver's periodic BCs;
use replicate padding on wall axes). Keep it tiny so finite differences over *all* weights is
affordable.

**Gate — passed** ([`nn_stage2_cnn.py`](../nn_stage2_cnn.py)): FD on all 173 weights, max
rel. err **4.6e-08**; shift-equivariance **1.1e-16**; target velocity reproduced to **8.9e-04**.

**Gate (b) is stated on VELOCITY, not on the source field, and that is a physics point, not a
weakened bar.** Only the *solenoidal* part of $S$ is identifiable from velocity data — the
projection removes any gradient component, so $S$ itself is not uniquely recoverable. Measured:
the velocity matches to 8.9e-04 while the source field still differs by 1.8e-01, exactly as that
non-identifiability predicts. Asking for source recovery would be asking for something the
physics does not determine.

**A test-design trap worth recording.** The first version scaled the target network's weights by
2 "to make the target non-trivial". That pushed `tanh` into saturation — a known-hard
optimisation regime — and capped the achievable fit at 2.9e-03 against a 1e-3 bar. The gradient
was correct the whole time (gate (a) passed at 5e-08). The fix was to the *target*, not the bar:
an unsaturated target of the same architecture reaches 8.9e-04. When a gradient check passes and
a training check fails, suspect the optimisation setup before the coupling.

---

## Stage 3 — multi-step rollout with checkpointing

Training signal comes from a trajectory, not one step. Roll out $N$ steps, loss on the final
state (or accumulated).

**Build.** `rollout.py`: forward over $N$ steps; backward accumulating adjoints in reverse.
Add gradient checkpointing — store the state every $k$ steps and recompute the forward within
a segment, since each step must otherwise retain $A$, $M$, $\mathbf{u}^*$, $\phi$.

**Gate — passed** ([`nn_stage3_rollout.py`](../nn_stage3_rollout.py)): FD **6.1e-09**;
checkpointed gradient identical to **exactly 0** difference; 16-step peak memory **0.8 MB**
checkpointed vs **13.8 MB** not (a 17× reduction); adjoint norm ratio **0.91**.

**Gate (a) had to be redesigned, and the reason matters.** As first written it compared the
rollout gradient against finite differences — but the rollout uses the frozen-coefficient
approximation, while FD measures the *exact* derivative. The gate was therefore testing
something the implementation deliberately does not compute, and it failed at 16% relative
error. The fix was **not** a looser tolerance: the gate now runs with `rebuild=False`, holding
$A$ genuinely constant so the adjoint gradient *is* exact and FD must match — isolating the
time chain (Stage 3's actual subject) from the frozen-coefficient bias (Stage 4's).

**Watch for.** Adjoint instability over long rollouts: the adjoint of an advection-dominated
flow transports sensitivity *upstream* and can amplify. Plot $\lVert\lambda\rVert$ per step;
if it grows exponentially, shorten the window rather than clipping gradients silently.

---

## Stage 4 — the frozen-coefficient decision

Currently $A$ depends on $\mathbf{u}^n$ through the SOU coefficients, so the exact gradient
needs $\partial L/\partial A \to \partial L/\partial\mathbf{u}^n$ (PICT's
`SetupAdvectionMatrixEulerImplicit_GRAD`). Dropping it is cheaper and often still a usable
descent direction — but it is **not** the true gradient.

**Build.** Implement the term; make it a flag.

**Gate.** Measure, don't assume: report the angle between the exact and frozen-coefficient
gradients, and the difference in converged loss after a short training run. **Publish the number
either way** — if the shortcut is used in later stages, its bias must be a stated quantity, not
an unexamined convenience.

**Angle already measured** (Stage 3, 12³, 5-step rollout): **1.75°**, magnitude ratio **1.003**
— inside the 5° threshold, so the shortcut is acceptable at this configuration.

Worth noting *why the angle is the right metric*: per-component relative error between the two
gradients reaches **16%**, which looks alarming, yet the gradient *direction* — the only thing
descent uses — differs by under 2°. Small components can be badly wrong in relative terms while
contributing nothing to the direction. Had the criterion been per-component error, this shortcut
would have been rejected on a misleading number.

*Remaining for Stage 4:* the Δ converged-loss comparison after training with each gradient.

---

## Stage 5 — a real closure task

Only now attempt something with physical content. Two sub-steps, in order:

**5a — a-priori.** Filter a fine-grid solution onto a coarse grid, compute the exact
sub-grid term, and train the CNN to predict it directly (no solver in the loop). This is a
plain regression problem and isolates *network capacity* from *solver coupling*.

**5b — a-posteriori.** Put the solver back in the loop: train so that the coarse-grid
trajectory matches the filtered fine-grid trajectory over $N$ steps. This is the PICT
configuration and the whole point of differentiability — the network sees the solver's actual
response rather than a one-step surrogate.

**Gate.**
- 5a: correlation with the true SGS term > 0.8 on held-out data.
- 5b: coarse-grid trajectory error beats both (i) no model and (ii) the a-priori-trained model
  used a-posteriori. If 5b does not beat 5a's model, the differentiability is buying nothing
  and that should be reported plainly rather than buried.

**Test case.** Decaying Taylor-Green on a fully periodic grid — we already have exact
solutions, periodic BCs, and verified 2nd-order spatial accuracy there, so the coarse-grid
error is attributable to the closure rather than to boundary treatment.

---

---

## Test problems and acceptance criteria at a glance

Every stage names one concrete problem and a numeric bar. Stages 1–3 are deliberately tiny —
they are debugging instruments, not experiments, and should run in seconds to minutes.

| Stage | Test problem | Config | Acceptance criteria |
|---|---|---|---|
| **0** ✅ | adjoint of one linear solve | 8³ warped, one PISO step | (a) adjoint identity on the **non-symmetric** momentum matrix, rel. err < 1e-8; (b) FD through a full step, ≥ 6 digits; (c) pressure gradient invariant to a constant shift of $\bar g$, < 1e-12 |
| **1** ✅ | recover a scalar forcing amplitude | 8³ warped, 1 step, $S = c\,\Phi(\mathbf x)$, $c_{\text{true}}=0.7$ | (a) $\partial L/\partial c$ vs central FD, rel. err < 1e-6; (b) **sign** correct (negative when $c < c_{\text{true}}$); (c) descent from $c=0$ recovers $0.7$ to < 1e-4 |
| **2** ✅ | tiny CNN predicts a source field | 16³ **periodic** Cartesian, 1 step, 173 weights, target from the same architecture with different weights | (a) FD on **every** weight, max rel. err < 1e-5 → **4.6e-08**; (b) reproduce the target **velocity** to < 1e-3 → **8.9e-04**; (c) shift-equivariance < 1e-10 → **1.1e-16** |
| **3** ✅ | same CNN, 5-step rollout | 12³ periodic, loss on final state | (a) FD on 5 sampled weights, rel. err < 1e-5 → **6.1e-09**; (b) checkpointed == non-checkpointed → **exactly 0**; (c) peak memory at 16 steps **0.8 MB vs 13.8 MB**; (d) $\lVert\lambda\rVert$ ratio early→late **0.91** (bounded) |
| **3.5** | **3D Taylor-Green energy budget** | 48³ periodic, $\nu=0.01$, $t\in[0,2]$ | (a) $-\mathrm{d}E/\mathrm{d}t$ vs $2\nu Z$ agree within **5%** for central; (b) numerical dissipation *quantified* for SOU; (c) $E$ monotone decreasing; (d) flux divergence < 1e-9 throughout |
| **4** | frozen-coefficient bias | 16³ periodic, 10-step rollout | Report, do not gate: angle between exact and frozen gradients (deg), and $\Delta$ converged loss after 200 training steps. **Publish the number**; only use the shortcut if the angle < 5° |
| **5a** | a-priori SGS regression | filter 64³ → 16³ TGV, no solver in the loop | correlation with the true SGS term > **0.8** on held-out snapshots |
| **5b** | a-posteriori closure training | 16³ coarse vs filtered 64³, 20-step rollout | (a) trajectory error beats **no model** by > 30%; (b) beats the 5a model used a-posteriori; (c) stable over 5× the training horizon. If (b) fails, differentiability bought nothing — report that plainly |

**Why Stage 3.5 sits where it does.** It is not a gradient check, it is a *prerequisite* for
Stage 5 to mean anything. MMS verifies that each operator approximates its differential
counterpart; it says nothing about whether the assembled nonlinear scheme conserves the
quadratic invariants that govern a cascade. The periodic identity

$$\frac{\mathrm{d}E}{\mathrm{d}t} = -2\nu Z, \qquad E=\tfrac12\langle|\mathbf u|^2\rangle,\quad Z=\tfrac12\langle|\boldsymbol\omega|^2\rangle$$

holds exactly in the continuum, so the gap between measured $-\mathrm{d}E/\mathrm{d}t$ and
$2\nu Z$ **is** the scheme's numerical dissipation. That matters directly for Stage 5: SOU is
dissipative by construction, so training a closure on an SOU baseline asks the network to
correct numerics as well as physics, and the two become inseparable. Measure the numerical
dissipation first, then decide which convection scheme the closure work should use.

**Measured** (`run_tgv3d.py`, 48³, ν = 0.01, rotational + BDF2, t ∈ [0,2]):

| scheme | mean numerical / physical dissipation | max | energy at t=2 |
|---|---|---|---|
| 2nd-order upwind | **1.10 %** | 1.48 % | 6.66e-04 |
| central | **0.56 %** | 0.66 % | 6.77e-04 |

Energy is monotone decreasing for both, and flux divergence stays at 7.8e-15 (SOU) /
4.3e-14 (central). SOU carries **roughly twice** central's numerical dissipation and removes
1.56 % more total energy by t = 2 — the dissipative error is real, quantified, and exactly the
sort of thing a closure would otherwise be asked to absorb.

**Resolution honesty — this is not a turbulence benchmark.** The canonical TGV case is
$Re=1600$, needing ~$256^3$ for DNS, far beyond a NumPy solver. At ν = 0.01 the flow is fully
resolved and laminar: **enstrophy peaks at t = 0 and decays monotonically**, so there is no
vortex stretching and no cascade (the real TGV peaks near t ≈ 9). What this validates is the
*energy-budget machinery* and the relative dissipation of the two convection schemes — not
turbulence. Any write-up must say so.


## Verification tooling to build alongside

| Tool | Purpose |
|---|---|
| `check_gradient(fn, params, idxs)` | central FD vs adjoint, float64, with tolerance tied to solver tolerance |
| `check_adjoint_identity(A)` | $\langle A^{-1}v, w\rangle = \langle v, A^{-\mathsf T}w\rangle$ — run it on the **momentum** matrix, where a missing transpose is fatal and detectable |
| adjoint-norm logger | $\lVert\lambda\rVert$ per step, to catch adjoint blow-up early |

---

## Standing risks

- **Solver tolerance caps gradient accuracy.** The adjoint inherits the forward residual, so a
  forward tolerance of $10^{-6}$ limits gradient accuracy to roughly the same level. Tighten to
  $10^{-12}$ during FD checks or the residual dominates the comparison.
- **Non-convergence must raise, not warn.** A silently unconverged adjoint solve produces a
  plausible but wrong gradient. The deferred-correction contraction check already in the port
  must be applied to the adjoint loop too.
- **Grid warp ≲ 0.15.** The deferred correction stops contracting beyond that, forward and
  adjoint alike. Keep training cases inside it, or make the cross terms implicit first.
- **Wall-bounded cases are 1st-order in space.** Prefer periodic training cases until the
  half-cell boundary-flux stencil is upgraded, so closure error is not confounded with
  boundary error.

---

## Order of work

```
Stage 0  done
Stage 1  done  scalar recovery        <- proves sign and scale
Stage 2  done  tiny CNN, all weights  <- proves the field mapping
Stage 3  done  rollout + checkpoint   <- proves the time chain
Stage 3.5 TGV energy budget     <- proves the baseline is not numerically polluted
Stage 4  part   frozen-coeff bias     <- angle measured (1.75 deg); loss delta pending
Stage 5a a-priori regression    <- isolates network capacity
Stage 5b a-posteriori training  <- the actual PICT configuration
```

Each gate is cheap relative to the stage it protects; stages 1–3 should run in seconds to
minutes on a 16³ grid, which is deliberate — they are debugging instruments, not experiments.
