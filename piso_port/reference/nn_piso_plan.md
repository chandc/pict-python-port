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

**Gate.**
1. Finite-difference check on **every** weight (~200 of them), max relative error < 1e-5.
2. Gradient is unchanged when the network output is shifted by a constant in a periodic
   direction where the physics is shift-invariant — a cheap symmetry check that catches
   indexing errors in the CNN↔solver field mapping.

---

## Stage 3 — multi-step rollout with checkpointing

Training signal comes from a trajectory, not one step. Roll out $N$ steps, loss on the final
state (or accumulated).

**Build.** `rollout.py`: forward over $N$ steps; backward accumulating adjoints in reverse.
Add gradient checkpointing — store the state every $k$ steps and recompute the forward within
a segment, since each step must otherwise retain $A$, $M$, $\mathbf{u}^*$, $\phi$.

**Gate.**
1. FD check on a 5-step rollout (a few sampled weights) to 5 digits.
2. Checkpointed gradient equals the non-checkpointed gradient to machine precision — this
   isolates checkpointing bugs from adjoint bugs.
3. Memory scales as $O(N/k + k)$, measured, not assumed.

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
gradients over a 10-step rollout, and the difference in converged loss after a short training
run. **Publish the number either way** — if the shortcut is used in later stages, its bias must
be a stated quantity, not an unexamined convenience.

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
Stage 1  scalar recovery        <- proves sign and scale
Stage 2  tiny CNN, all weights  <- proves the field mapping
Stage 3  rollout + checkpoint   <- proves the time chain
Stage 4  frozen-coeff bias      <- quantifies a known approximation
Stage 5a a-priori regression    <- isolates network capacity
Stage 5b a-posteriori training  <- the actual PICT configuration
```

Each gate is cheap relative to the stage it protects; stages 1–3 should run in seconds to
minutes on a 16³ grid, which is deliberate — they are debugging instruments, not experiments.
