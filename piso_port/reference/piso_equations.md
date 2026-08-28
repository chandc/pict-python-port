# PISO on a 3D curvilinear collocated grid — equations as implemented

Companion to `piso_numpy_3d.py`, `phase5_fluxes.py`, `phase3_momentum.py`.
Cross-references to the PICT C++ kernels are given so each discrete choice can be traced
back to the reference implementation.

---

## 1. Governing equations and the curvilinear map

Incompressible Navier–Stokes, unit density:

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u}
= -\nabla p + \nu \nabla^2 \mathbf{u}, \qquad \nabla\cdot\mathbf{u} = 0$$

Physical space $(x,y,z)$ maps to a uniform computational space $(\xi^1,\xi^2,\xi^3)=(\xi,\eta,\zeta)$
with $\Delta\xi^i = h_i$. The Jacobian (cell volume) and the **contravariant metric tensor** are

$$J = \det\!\left(\frac{\partial x_m}{\partial \xi^i}\right), \qquad
g^{ij} = \nabla\xi^i\cdot\nabla\xi^j = \sum_m \frac{\partial\xi^i}{\partial x_m}\frac{\partial\xi^j}{\partial x_m}$$

$g^{ij}$ is the object that matters for everything below: **$g^{ij}=0$ for $i\neq j$ exactly when
the grid is orthogonal.** Every non-orthogonal difficulty in this solver is the story of the
off-diagonal $g^{ij}$.

Metrics use the conservative Thomas & Lombard (1979) form, which makes the discrete Geometric
Conservation Law vanish to machine precision (measured 7.2e-13; a naive form gives O(1)).

**Volume-weighted contravariant velocity** — the flux-carrying quantity:

$$JU^i = J\,\nabla\xi^i\cdot\mathbf{u} = J\left(\frac{\partial\xi^i}{\partial x}u
+ \frac{\partial\xi^i}{\partial y}v + \frac{\partial\xi^i}{\partial z}w\right)$$

Continuity in conservative curvilinear form is then simply $\partial_{\xi^i}(JU^i) = 0$.

### Storage layout
Pressure **and** velocity live at cell centres — PICT is *collocated*
(`Block::CreatePressure -> CreateDataTensor(1)`, `Block::CreateVelocity -> CreateDataTensor(dims)`,
same grid). Only the metric transforms are face-staggered (`m_faceTransform`). Face fluxes are a
**derived** quantity.

---

## 2. Momentum predictor — Backward Euler

Multiply the momentum equation by $J$ (volume integration — this is what keeps the implicit
operator and the lagged correction on the same footing) and treat time with Backward Euler,
matching `SetupAdvectionMatrixEulerImplicit` (`PISO_multiblock_cuda_kernel.cu:4483`):

$$\underbrace{\left[\frac{J}{\Delta t} + J\,\mathcal{A}(\mathbf{u}^n) - \nu\,\mathcal{L}_\perp\right]}_{A}
\mathbf{u}^{*} \;=\; \frac{J}{\Delta t}\mathbf{u}^n \;+\; \nu\,J\,\mathcal{C}(\mathbf{u}^{*,\,\text{old}})$$

with

* **Advection** $J\mathcal{A}\phi = J\,U^i\,\partial_{\xi^i}\phi$, discretised 2nd-order upwind,
  coefficients $\tfrac{1}{2h}(3,-4,1)$ upwind-biased, degrading to 1st-order upwind within two
  cells of a boundary. The convecting $U^i$ is lagged one step (Picard linearisation).
* **Orthogonal diffusion** $\mathcal{L}_\perp$ — the $i=j$ part, implicit, 7-point.
* **Cross diffusion** $\mathcal{C}$ — the $i\neq j$ part, **lagged** (§4).

$A$ is non-symmetric once advection is present, so the solve uses BiCGStab (unpreconditioned).

---

## 3. Face fluxes, the pressure equation, and the correction

### 3.1 Face flux
Interpolated from cell-centred contravariant components — mirroring `computeFluxesNDLoop`
(`:1553`), where the line is literally `fluxes[bound] = (velN + velC) * 0.5f`:

$$F_f = \tfrac{1}{2}\left[(JU^i)_P + (JU^i)_N\right]$$

**No Rhie–Chow interpolation** — a grep for `rhie|chow` across PICT returns nothing. Consistency
comes instead from expressing divergence *and* the pressure operator on the same faces.

Domain-boundary faces are **prescribed, never interpolated**: from the boundary velocity
(PICT's Dirichlet branch), or hard zero for a closed wall. Getting this wrong is expensive —
zeroing them on a flow with through-boundary velocity gave a discrete divergence of
**1.8e+01** on an *exactly* divergence-free field, and the projection then "corrected" that
phantom divergence and destroyed the interior solution.

### 3.2 Discrete divergence
Mirrors `k_computePressureRHSdivergenceFromFlux` (`:5375`):

$$(\nabla\cdot\mathbf{u})_P = \frac{1}{J_P}\sum_{i}\frac{F_{i,+} - F_{i,-}}{h_i}$$

### 3.3 Pressure equation
Requiring the *corrected* fluxes to be divergence-free gives, with $\Gamma = J/A_{\text{diag}}$
(PICT's `raP = 1/Adiag`, `PISO_build_pressure_matrix:4798`):

$$\nabla\cdot\left(\Gamma\,\nabla p\right) = \nabla\cdot\mathbf{u}^{*}$$

Discretely, the pressure flux through a face and its matrix are built from the **same**
face-interpolated coefficient — this is the whole trick:

$$\Phi_f = \underbrace{c_f\frac{p_N-p_P}{h_i}}_{\text{orthogonal, implicit}}
\;+\; \underbrace{\Psi_f}_{\text{cross, lagged}},
\qquad c_f = \tfrac{1}{2}\left[(\Gamma J g^{ii})_P + (\Gamma J g^{ii})_N\right]$$

Because $\;\mathcal{D}(\Phi_\perp) = -M p / J\;$ *exactly*, subtracting $\Phi$ from $F$ drives the
flux divergence to machine precision rather than to a floor. Measured: **1e-11 – 2e-10**, versus
a **188×**-only reduction (floor 1.5e-2) if one instead composes the wide-stencil differential
operators with the compact Laplacian. They are different operators — 9.9e-2 apart.

$M$ builds no face at the domain boundary, so its row sums vanish: it **is** the zero-flux
(Neumann) operator, singular with a constant null space. The RHS is compatible by construction
(the flux divergence telescopes to the prescribed boundary fluxes, zero for a closed domain);
we project the mean out anyway and pin one cell. Symmetric positive definite after pinning, so CG.

### 3.4 Velocity correction
`PISO_update_velocity` (`:5948`), PICT's default `velocity_corrector="FD"`:

$$\mathbf{u}^{n+1} = \underbrace{\mathbf{u}^{*}}_{\text{hbyA}} - \Gamma\,\nabla p$$

The predictor carries **no** pressure gradient — `PISOtorch_simulation.py:1173` sets
`apply_pressure_gradient = False` — so this is a *non-incremental* (Chorin) projection. §5 is
about why that costs an order, and how to get it back.

### 3.5 The loop
```
u*  <- momentum predictor                       # hbyA, no pressure gradient
repeat corrector_steps (PICT default: 2):
    F  <- face fluxes of u*
    solve  M p = J(D(Psi(p)) - div F)           # lagged cross terms, §4
    F  <- F - Phi(p)                            # fluxes now divergence-free
    u* <- u* - Gamma grad p
```

---

## 4. The non-orthogonal lag

### 4.1 Where the lag comes from
The full curvilinear Laplacian is a **27-point** operator:

$$\nabla^2 p = \frac{1}{J}\frac{\partial}{\partial \xi^i}\left(J g^{ij}\frac{\partial p}{\partial \xi^j}\right)
= \underbrace{\frac{1}{J}\partial_{\xi^i}\!\left(Jg^{ii}\partial_{\xi^i}p\right)}_{\text{7-point, } i=j}
+ \underbrace{\frac{1}{J}\partial_{\xi^i}\!\left(Jg^{ij}\partial_{\xi^j}p\right)_{i\neq j}}_{\text{20 cross terms}}$$

Making all 27 points implicit would destroy the cheap symmetric 7-point structure. So the cross
terms are moved to the right-hand side and evaluated at the **previous iterate** — PICT's
`nonOrthoFlags`. That one-iterate staleness *is* the lag:

$$M\,p^{(k+1)} = J\Big(\mathcal{D}\big(\Psi(p^{(k)})\big) - \nabla\cdot\mathbf{u}^{*}\Big)$$

This is a Picard iteration $p^{(k+1)} = M^{-1}C\,p^{(k)} + b$, so it converges only if
$\rho\!\left(M^{-1}C\right) < 1$, where $C$ is the cross-term operator. Since
$\|C\|/\|M\| \sim |g^{ij}|/g^{ii}$, the lag is harmless on an orthogonal grid ($g^{ij}=0$,
$\rho=0$, converges in one sweep) and fatal on a strongly skewed one.

### 4.2 Measured cost of the lag

Contraction ratio $\rho$ and sweeps to converge, $n=16$ (pressure solve):

| grid warp | $\rho$ | sweeps | outcome |
|---|---|---|---|
| 0.05 | 0.31 | 21 | fine |
| 0.10 | 0.59 | 49 | fine |
| 0.15 | 0.92 | 321 | converges, slow |
| **0.20** | **1.27** | — | **diverges — but see below** |

**Correction to that last row.** Measuring the Jacobian afterwards shows this grid family
*tangles* at warp 0.18: $\min(J)$ goes negative (0.15 → $+1.14\times10^{-2}$, 0.18 →
$-4.24\times10^{-1}$, 0.25 → $-1.75$). The $\rho = 1.27$ was therefore measured on a mesh with
negative cell volumes, and cannot be read as a solver limit alone — the deferred-correction
limit and the grid-validity limit coincide here. Both solve paths blow up at warp 0.25 at every
$\Delta t$ tried (0.02 / 0.005 / 0.001), which is the signature of a tangled grid, not of a CFL
limit. The practical consequence is in §4.4.

The same mechanism governs the momentum solve, where it additionally depends on $\nu$: neither
warp nor $\nu$ alone predicts failure, only their combination (a warp-only sweep and a
$\nu$-only sweep both miss it).

### 4.3 Treatment

1. **Iterate to convergence, not a fixed count.** Stop on $\max|p^{(k+1)}-p^{(k)}| < \text{tol}$.
2. **Under-relaxation ladder.** $p \leftarrow (1-\omega)p + \omega\,p^{(k+1)}$, trying
   $\omega = 1.0$ first (≈2× faster when it works) and stepping down only on genuine
   non-contraction. Restores convergence where $\rho$ is marginal; converges to the *same* fixed
   point (verified: identical errors across $\omega$).
3. **Contraction test, not a magnitude guard.** A ratio just above 1 grows too slowly to trip a
   "has it blown up" threshold within the iteration cap, and would otherwise be mistaken for slow
   convergence and returned as a valid answer. Detection uses the measured ratio over a window.
4. **Transient term helps.** $J/\Delta t$ on the momentum diagonal strengthens $M$ relative to
   $C$, which is why the unsteady momentum solve needs far fewer sweeps than the steady test.

**One treatment that does *not* work** — recorded so it is not re-attempted. Scaling the implicit
coefficient by $\beta$ and subtracting the same amount back out explicitly,

$$\Phi = \underbrace{\beta c_f \tfrac{\Delta p}{h}}_{\text{implicit}}
+ \underbrace{\Psi - (\beta-1)c_f\tfrac{\Delta p}{h}}_{\text{explicit}}$$

leaves $\Phi$ — and therefore the fixed point — identical, and drives the iteration toward the
identity map, converging *more slowly*. It was implemented, measured, and removed. The genuine
remedy is to make the cross terms **implicit**, which is now implemented — see §4.4.

### 4.4 Implicit cross terms (`implicit_cross=True`)

The lag exists only because the cross flux $\Psi(p)$ sits on the RHS. Moving it to the left
gives the system the deferred correction was converging *to*:

$$\underbrace{M p}_{\text{7-point, orthogonal}} \;-\; J\,\nabla\!\cdot\Psi(p) \;=\; -J\,\nabla\!\cdot F$$

Both paths therefore have the same solution, and the equivalence test measures exactly that.

**Applied matrix-free.** $\Psi$ is re-used from `pressure_face_fluxes(include_orth=False,
include_cross=True)` with the *same* `coef`, so the operator is bit-for-bit the fixed point of
the loop it replaces. (Assembling a 19-point matrix would also work; matrix-free keeps the two
paths provably identical, which is what makes the equivalence gate meaningful.)

**The preconditioner is the whole ballgame.** One application of the full operator costs about
**32× a sparse matvec** (191.5 µs vs 5.9 µs at 1024 cells; the cross term alone is 185.6 µs), so
unpreconditioned this is a net *loss* — measured at 0.13×–0.94× on the cavity. Preconditioning with the orthogonal part $M$
alone — precisely the operator each deferred sweep already inverts — gives

$$M^{-1}A \;=\; I - M^{-1}J\,\nabla\!\cdot\Psi(\cdot)$$

whose second term has spectral radius equal to the contraction ratio $\rho$ of §4.2. The
preconditioned spectrum thus clusters at 1 with spread $\rho$, which Krylov resolves in a few
iterations instead of the $\rho^k$ decay of a fixed-point sweep.

**Measured** (`test_implicit_cross.py`, $n=16$, 20 steps, iteration counts for the final solve):

| problem | warp | deferred | implicit | speed-up | same answer to |
|---|---|---|---|---|---|
| 2D cavity | 0.00 | 0.25 s, 3 sweeps | 0.19 s, 1 Krylov | 1.33× | 3.8e-14 |
| 2D cavity | 0.05 | 1.64 s, 22 sweeps | 0.34 s, 8 Krylov | 4.76× | 5.4e-13 |
| 2D cavity | 0.10 | 3.61 s, 48 sweeps | 0.43 s, 12 Krylov | 8.36× | 4.9e-13 |
| 2D cavity | 0.15 | 9.67 s, 121 sweeps | 0.54 s, 17 Krylov | **17.84×** | 7.1e-13 |
| channel | 0.00 | 0.12 s, 2 sweeps | 0.10 s, 1 Krylov | 1.23× | 0.0 |
| channel | 0.05 | 0.39 s, 11 sweeps | 0.19 s, 5 Krylov | 2.05× | 1.6e-14 |
| channel | 0.10 | 0.53 s, 16 sweeps | 0.22 s, 7 Krylov | 2.41× | 6.5e-14 |
| channel | 0.15 | 0.72 s, 22 sweeps | 0.25 s, 9 Krylov | 2.84× | 3.5e-14 |

The gain grows with warp, which is the point: it is largest exactly where the lag hurts most.
The channel gains less because its walls lie on one axis only, so it carries fewer cross terms
to begin with.

**Two honest caveats.**

1. **Symmetry is boundary-condition dependent.** Measuring $\langle Av,w\rangle$ vs
   $\langle v,Aw\rangle$ on a $12^3$ warp-0.10 grid: fully periodic gives **1.35e-16**
   (symmetric, CG valid); walls in $y$ give **2.96e-02** and the cavity's walls in $x,z$ give
   **1.12e-01** — the one-sided edge differences are not self-adjoint. This
   retracts the earlier "still SPD and CG-able" claim above, which had only been checked on the
   periodic case. The code uses CG when all axes are periodic and BiCGStab (LGMRES fallback)
   otherwise.
2. **It does not buy extra warp range.** The tempting claim — that removing the Picard iteration
   lifts the warp-0.18 limit — is not demonstrable, because that limit is also where the mesh
   tangles. There is no valid grid beyond it on which to show the benefit. The demonstrated
   value is speed at warps that actually mesh. `test_implicit_cross.py` asserts $\min(J) > 0$ at
   every warp it exercises so this cannot silently regress into testing on a broken mesh.

---

## 5. Getting to second order: the rotational correction

### 5.1 Why the present scheme is 1st order
With `apply_pressure_gradient = False` this is the **non-incremental (Chorin) projection**. The
projection step implicitly imposes

$$\left.\frac{\partial p}{\partial n}\right|_{\text{wall}} = 0$$

which the true Navier–Stokes pressure does **not** satisfy. The inconsistency is confined to a
numerical boundary layer of thickness $O(\sqrt{\nu\Delta t})$ and caps the velocity at
$O(\Delta t)$.

Measured, and confirmed as the cause rather than assumed: the error decays monotonically away
from the wall (5.4e-2 → 7.8e-3 over five layers) and does **not** improve with more correctors
(1/2/4 → 1.25e-2/1.35e-2/1.48e-2) — scheme order reduction, not an unconverged iteration.
Taylor-Green against the exact unsteady solution gives rates **0.96–1.12** (Cartesian),
0.79–0.81 (warped).

### 5.2 Standard incremental form — necessary but not sufficient
Carry $p^n$ in the predictor and solve for the increment $\phi$:

$$\frac{\mathbf{u}^*-\mathbf{u}^n}{\Delta t} + \mathcal{A}(\mathbf{u}^n)
= -\nabla p^{n} + \nu\nabla^2\mathbf{u}^*,\qquad
\nabla\cdot(\Gamma\nabla\phi)=\nabla\cdot\mathbf{u}^*,\qquad
\mathbf{u}^{n+1}=\mathbf{u}^*-\Gamma\nabla\phi$$

$$p^{n+1} = p^n + \phi$$

This improves the pressure but **still enforces** $\partial\phi/\partial n = 0$, hence
$\partial p/\partial n = 0$, so the boundary layer survives: velocity saturates near
$O(\Delta t^{3/2})$.

### 5.3 The rotational term — the actual fix
Use the identity

$$\nu\nabla^2\mathbf{u} = \nu\nabla(\nabla\cdot\mathbf{u}) - \nu\nabla\times(\nabla\times\mathbf{u})$$

On the *predictor* field $\mathbf{u}^*$ the term $\nabla\cdot\mathbf{u}^*$ is **not** zero, and it
is exactly the piece the standard update discards. Restoring it
(Timmermans, Minev & van de Vosse 1996; Guermond & Shen) changes **only the pressure update**:

$$\boxed{\;p^{n+1} = p^{n} + \phi - \nu\,\big(\nabla\cdot\mathbf{u}^{*}\big)\;}$$

The added term cancels the spurious $\partial p/\partial n = 0$ constraint at leading order,
removing the boundary layer and recovering

$$\|\mathbf{u}-\mathbf{u}_h\|_{L^2} = O(\Delta t^{2}), \qquad \|p-p_h\| = O(\Delta t^{3/2})$$

### 5.4 What this costs us in code
Small, because the pieces already exist:

1. `_solve_momentum` — add $-J\,\nabla p^n$ to the RHS (PICT supports this:
   `SetupAdvectionVelocityEulerImplicitRHS(..., applyPressureGradient)`; the flag is simply
   `False` in the driver).
2. Treat the solved field as the increment $\phi$, not the pressure.
3. Add the rotational update. **$\nabla\cdot\mathbf{u}^*$ is already computed** — it is `div_F`,
   the flux divergence that forms the pressure RHS. So this is one line:
   `p_new = p_old + phi - nu * div_F`.
4. Pair with a 2nd-order time scheme (BDF2 or Crank–Nicolson) — otherwise Backward Euler's own
   $O(\Delta t)$ truncation error becomes the binding constraint and the rotational term buys
   nothing.

**Caveat — tested twice, and the second test overturned the first.**

An early attempt at the incremental form showed the pressure drifting without bound. A first
round of testing found both incremental and rotational blowing up on a wall-bounded channel
while Chorin stayed stable, and this document previously concluded that *"incremental and
rotational are usable only on fully periodic domains."*

**That was wrong.** The divergence was a bug in the pressure-correction coefficient, not a
property of the schemes:

$\Gamma$ approximates how $A^{-1}$ responds to a pressure gradient, and was taken as
$J/A_\text{diag}$ (PICT's `raP = 1/Adiag`). But **the conservative diffusion and SOU advection
operators both have exactly zero row sum** — verified to 7e-14 on warped grids, periodic and
walled — so for a smooth gradient the correct measure is the *row sum*, not the diagonal.
Using the diagonal under-corrects by

$$\frac{A_\text{diag}}{\text{rowsum}(A)} = 1 + 2\nu\Delta t\sum_i \frac{1}{h_i^2}$$

which grows without bound as the grid refines: measured 1.60 at $\nu\Delta t/h^2 = 0.28$, 3.37
at 1.20, 10.4 at 4.81. Chorin is immune because it *replaces* $p$ each step; the incremental
schemes *accumulate* it, feeding that deficit back until the loop gain exceeds 1. The
divergence threshold tracks the ratio exactly (stable below ~3, unstable above ~3.4).

Since $\text{rowsum}(A) = J/\Delta t$ identically, $\Gamma_\text{rowsum} = \Delta t$ — grid-
and $\Delta t$-independent, and the classic SIMPLEC choice. With it, every previously divergent
case is stable, on Cartesian and warped grids alike.

`pressure_coef` now defaults to `'diag'` for `chorin` (matching PICT exactly, where it is
harmless) and `'rowsum'` for the accumulating schemes, with a setup check that **raises** if the
unsound combination is requested above the ratio-3 threshold.

A note on what this episode should teach: the symptom was a stability limit on
$\nu\Delta t/h^2$ — the *explicit* diffusion number — in a scheme that solves diffusion
implicitly. That is physically backwards (more diffusion must stabilise an implicit scheme) and
should have been read immediately as evidence of a bug rather than recorded as a scheme
limitation.
