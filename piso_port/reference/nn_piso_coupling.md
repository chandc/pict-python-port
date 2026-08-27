# Coupling a neural network to PISO: the discrete adjoint

How to make a PISO step differentiable so a network embedded in it can be trained against the
true solver response, and — the part that actually decides whether the gradients are correct —
how the adjoint linear solves differ for **velocity** and **pressure**.

Companion to [`piso_equations.md`](piso_equations.md) and
[`spatial_discretization.md`](spatial_discretization.md).

---

## 1. Where the network plugs in

A PISO step is a sequence of assemblies and linear solves. A network can enter anywhere a
*field* is consumed; three useful hooks, in increasing order of adjoint difficulty:

| Hook | Enters as | Adjoint cost |
|---|---|---|
| **Body force / SGS stress** $\;S_\theta = \mathrm{NN}(\mathbf{u}^n)$ | additive term in the momentum **RHS** | trivial — linear, so $\partial L/\partial S = J\lambda_u$ |
| **Eddy viscosity** $\;\nu_t = \mathrm{NN}(\mathbf{u}^n)$ | inside the momentum **matrix** $A$ | needs $\partial L/\partial A$ contracted with $\partial A/\partial\nu_t$ |
| **Boundary values** | Dirichlet data, eliminated into the RHS | via the elimination term $-A_{ib}\phi_b$ |

PICT exposes all three (`velocitySource`, `m_viscosity`, boundary tensors). The SGS-stress hook
is what turbulence-model learning uses, and is the cleanest place to start.

One step's forward graph, with $\theta$ the network weights:

$$\theta \to S_\theta \to \underbrace{A\,\mathbf{u}^{*} = \mathbf{b}(S_\theta,\mathbf{u}^n)}_{\text{non-symmetric solve}}
\to \underbrace{M\,\phi = \mathbf{r}(\mathbf{u}^{*})}_{\text{symmetric singular solve}}
\to \mathbf{u}^{n+1} = \mathbf{u}^{*} - \Gamma\nabla\phi \to L$$

---

## 2. The one identity everything rests on

For $\mathbf{x} = A^{-1}\mathbf{b}$ with incoming gradient $\bar{\mathbf{g}} = \partial L/\partial\mathbf{x}$:

$$\mathrm{d}\mathbf{x} = A^{-1}\mathrm{d}\mathbf{b} - A^{-1}\,\mathrm{d}A\;\mathbf{x}
\;\Longrightarrow\;
\mathrm{d}L = \bar{\mathbf{g}}^{\mathsf T}\mathrm{d}\mathbf{x}
= \underbrace{(A^{-\mathsf T}\bar{\mathbf{g}})}_{\textstyle \lambda}{}^{\mathsf T}\mathrm{d}\mathbf{b}
- \lambda^{\mathsf T}\,\mathrm{d}A\;\mathbf{x}$$

$$\boxed{\;\lambda = A^{-\mathsf T}\bar{\mathbf{g}}, \qquad
\frac{\partial L}{\partial \mathbf{b}} = \lambda, \qquad
\frac{\partial L}{\partial A} = -\,\lambda\,\mathbf{x}^{\mathsf T}\;}$$

Three consequences that drive the implementation:

1. **The backward pass is another linear solve, with $A^{\mathsf T}$** — comparable in cost to
   the forward solve.
2. **$-\lambda\mathbf{x}^{\mathsf T}$ is never formed densely.** Only entries inside $A$'s
   sparsity pattern are needed: a *sparse* outer product, $O(\mathrm{nnz})$ not $O(N^2)$. PICT
   has a dedicated `SparseOuterProduct` kernel for it.
3. **Save $\mathbf{x}$, not $\mathbf{b}$** — the backward pass needs the solution. PICT does
   `ctx.save_for_backward(A_val, x)`.

This is exactly PICT's implementation, [`PISOtorch_diff.py:307-374`](../../PISOtorch_diff.py):
forward calls the solver with a `transpose` flag; backward calls the *same* solver with
`not transpose`, then `SparseOuterProduct(grad_b, x, grad_A)` followed by `grad_A_val *= -1`.

---

## 3. Velocity: the non-symmetric adjoint

$$A \;=\; \frac{J}{\Delta t} \;+\; J\,\mathcal{A}_{\mathrm{SOU}}(\mathbf{u}^n) \;-\; \nu\,\mathcal{L}_\perp$$

**$A$ is not symmetric, and advection is the culprit.** The diffusion block is symmetric by
construction (face-interpolated coefficients — `spatial_discretization.md` §2a) and $J/\Delta t$
is diagonal. But second-order upwind is *one-sided*: for $a>0$ it writes $a_{i-1}=-2a$ and
$a_{i-2}=+\tfrac12 a$ while writing **nothing** into $a_{i+1}, a_{i+2}$. So
$A_{i,i-1} \neq A_{i-1,i}$, and the asymmetry is not a perturbation — it is the entire
convective operator.

Four consequences:

**(a) The adjoint solve uses a different operator.** $A^{\mathsf T}\neq A$, so
$\lambda = A^{-\mathsf T}\bar{\mathbf g}$ is a genuinely separate solve. BiCGStab both ways; CG
is invalid on either.

**(b) $A^{\mathsf T}$ is the adjoint *transport* operator.** Transposing an upwind stencil
reverses the direction of information flow: the forward operator carries information
downstream, the adjoint carries sensitivity **upstream**. Physically right, but it means the
adjoint solve's conditioning and convergence differ from the forward one — a solver tuned on
the forward problem is not automatically tuned on the adjoint.

**(c) Preconditioners do not carry over unchanged — but they can be reused.** An ILU
factorisation $A\approx LU$ is *not* a preconditioner for $A^{\mathsf T}$. But
$(LU)^{\mathsf T} = U^{\mathsf T}L^{\mathsf T}$ **is**, so the same factorisation serves both
directions if the triangular solves are applied in transposed order. Recompute nothing; just
transpose the application. Routing both directions through one solver with a `transpose` flag,
as PICT does, gets this for free.

**(d) $A$ depends on $\mathbf{u}^n$, so $\partial L/\partial A$ must be propagated further.**
The convecting velocity is frozen at the previous step (Picard linearisation), so *within* a
step $A$ is constant — but across steps it is not. The chain
$\partial L/\partial A \to \partial L/\partial\mathbf{u}^n$ runs through the SOU coefficients
and the metrics; PICT implements it as `SetupAdvectionMatrixEulerImplicit_GRAD`.
**Dropping it gives a "frozen-coefficient" gradient** — cheaper, often still a usable descent
direction, but *not* the true gradient. If you drop it, say so: it silently biases long
training rollouts.

---

## 4. Pressure: symmetric, but singular

$$M \;=\; -\nabla\!\cdot\!\left(\Gamma\,\nabla\;\right), \qquad \Gamma = J/A_{\mathrm{diag}}$$

**$M$ is symmetric exactly** — the same face-interpolation argument as the diffusion block,
asserted in code as `abs(M - M.T).max() == 0`. So the adjoint is markedly cheaper than the
velocity side:

$$M^{\mathsf T} = M \quad\Longrightarrow\quad \lambda_p = M^{-1}\bar{\mathbf{g}}_p$$

- **Same matrix** — no transpose assembly.
- **Same solver** — CG, valid in both directions.
- **Same preconditioner / factorisation** — an IC or AMG hierarchy built for the forward solve
  is reusable verbatim. Cache it per step and the adjoint pressure solve is nearly free.

**But $M$ is singular**, and this is where pressure adjoints usually go wrong. With Neumann
walls or full periodicity $M\mathbf{1}=0$; the null space is the constants (measured
`max|M @ 1| = 1.6e-12`). Three rules:

1. **Project the incoming gradient.** $M\lambda_p = \bar{\mathbf g}_p$ is solvable only if
   $\bar{\mathbf g}_p \perp \mathcal N(M^{\mathsf T}) = \mathrm{span}\{\mathbf 1\}$, so subtract
   its mean first. Skip this and CG is handed an inconsistent system: it will not converge, and
   whatever it returns is noise.
2. **Fix the constant identically in both passes.** If the forward pins cell 0, the forward map
   is $p = P M_{ff}^{-1} R\,\mathbf b$ with restriction $R$, prolongation $P$; its adjoint is
   $R^{\mathsf T} M_{ff}^{-1} P^{\mathsf T}$ — the *same* reduced solve. Pin identically, or the
   gradient picks up a spurious rank-one component.
3. **Remove the mean from $p$ in the forward pass too**, so the loss cannot depend on the
   arbitrary constant. PICT does this explicitly:
   `pressureResult = pressureResult - torch.mean(pressureResult)  # for numerical and backwards
   stability` ([`PISOtorch_simulation.py:1029`](../../PISOtorch_simulation.py)) — "backwards
   stability" is the tell.

### Side by side

| | velocity ($A$) | pressure ($M$) |
|---|---|---|
| symmetric | **no** — SOU is one-sided | **yes**, exactly |
| adjoint operator | $A^{\mathsf T}$, a different matrix | $M$ itself |
| Krylov method | BiCGStab both ways | CG both ways |
| preconditioner | reuse $LU$ **transposed** | reuse verbatim |
| singular | no ($J/\Delta t$ on the diagonal) | **yes** — constant null space |
| extra care | propagate $\partial L/\partial A \to \partial L/\partial\mathbf u^n$ | project $\bar{\mathbf g}$, pin consistently |

---

## 5. The deferred-correction loop

Both solves sit inside a Picard iteration carrying the non-orthogonal terms explicitly
(`piso_equations.md` §4). Three ways to differentiate it:

1. **Unroll.** Exact for the truncated iteration, but memory is
   $O(\text{sweeps}\times\text{field})$ — and at warp 0.15 that is ~320 sweeps. Impractical.
2. **Freeze the correction.** Treat the converged cross-term contribution as constant. Cheap,
   usually a usable descent direction, wrong on skewed grids where the correction carries real
   sensitivity.
3. **Adjoint fixed-point iteration** — the right answer. At convergence $p^\star = G(p^\star)$,
   so by the implicit function theorem
   $$\frac{\partial p^\star}{\partial\theta} = \Big(I - \tfrac{\partial G}{\partial p}\Big)^{-1}\frac{\partial G}{\partial\theta}$$
   and $\big(I - \partial G/\partial p\big)^{\mathsf T}\lambda = \bar{\mathbf g}$ is solved by
   **running the same deferred-correction loop on the adjoint variable** with the transposed
   cross-term operator. Cost mirrors the forward loop; memory is $O(1)$ in sweeps. Reuse the
   same under-relaxation ladder — the contraction ratio is identical, so it inherits the same
   warp ≈ 0.18 limit.

---

## 6. Implementation sketch for this port

```python
class LinearSolve(torch.autograd.Function):
    """x = A^{-1} b.  Mirrors PICT's LinearSolveFunction."""

    @staticmethod
    def forward(ctx, A_val, b, A_idx, shape, symmetric, singular):
        A = csr(A_val, A_idx, shape)
        if singular:
            b = b - b.mean()                    # compatibility with N(M) = span{1}
        x = cg(A, b) if symmetric else bicgstab(A, b)
        ctx.save_for_backward(A_val, x)         # the SOLUTION, not the RHS
        ctx.meta = (A_idx, shape, symmetric, singular)
        return x

    @staticmethod
    def backward(ctx, g):
        A_val, x = ctx.saved_tensors
        A_idx, shape, symmetric, singular = ctx.meta
        A = csr(A_val, A_idx, shape)
        if singular:
            g = g - g.mean()                    # else the adjoint system is inconsistent
        # symmetric -> reuse A and CG;  non-symmetric -> transpose, BiCGStab
        lam = cg(A, g) if symmetric else bicgstab(A.T, g)
        grad_A = -sparse_outer(lam, x, pattern=A_idx)   # only the nnz entries
        return grad_A, lam, None, None, None, None
```

Call sites:

```python
u_star = LinearSolve.apply(A_val, b, A_idx, shp, symmetric=False, singular=False)  # BiCGStab, A^T
phi    = LinearSolve.apply(M_val, r, M_idx, shp, symmetric=True,  singular=True)   # CG, M, projected
```

They differ only in those two flags — and those flags are the whole content of §3 versus §4.

---

## 7. Verifying the adjoint — before training anything

A wrong adjoint still trains, just to the wrong place. Verify directly; do not infer
correctness from a falling loss.

1. **Dot-product (adjoint identity) test.** For random $\mathbf v,\mathbf w$:
   $\langle A^{-1}\mathbf v,\mathbf w\rangle = \langle \mathbf v, A^{-\mathsf T}\mathbf w\rangle$
   to solver tolerance. This catches a missing transpose instantly — but note it passes on a
   *symmetric* matrix even if you forgot the transpose, so **run it on the momentum matrix**,
   where forgetting is both fatal and detectable.
2. **Finite differences on the full step.** Perturb one weight and compare
   $(L(\theta+\epsilon)-L(\theta-\epsilon))/2\epsilon$ against the adjoint gradient. Use
   float64 with $\epsilon\sim10^{-6}$, and tighten the linear-solve tolerance well below the
   expected gradient error or the solver residual dominates the comparison.
3. **Check the singular case separately.** Adding a constant to $\bar{\mathbf g}_p$ must leave
   the final gradient unchanged. If it does not, the projection or the pinning is inconsistent
   between passes.
4. **Quantify the frozen-coefficient shortcut, if used.** Compare gradients with and without
   the $\partial A/\partial\mathbf u^n$ term over a multi-step rollout. That gap is the bias you
   are accepting — measure it rather than assuming it is small.

---

## 8. Practical notes

- **Checkpointing.** Every step retains $A$, $M$, $\mathbf u^*$, $\phi$ for its backward pass;
  over a long rollout this dominates memory. Checkpoint every $k$ steps and recompute forward
  within a segment.
- **Solver tolerance caps gradient accuracy.** The adjoint inherits the forward residual, so a
  forward tolerance of $10^{-6}$ limits achievable gradient accuracy to about the same level —
  which matters when comparing against finite differences.
- **Non-convergence must be an error, not a warning.** A silently unconverged adjoint solve
  yields a plausible-looking but wrong gradient. The deferred-correction contraction check in
  this port applies to the adjoint loop as well.
