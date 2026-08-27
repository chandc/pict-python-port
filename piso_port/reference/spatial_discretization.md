# Spatial discretisation — how each operator is built

Every operator below lives on the same footing: the equations are **volume-integrated**
(multiplied through by the cell volume $J$), and every flux is evaluated at a **face** using
a coefficient interpolated from the two cells that share it. That single convention is what
makes the discrete divergence of the pressure flux equal the pressure matrix exactly, which
is in turn what lets the projection drive continuity to machine precision.

Notation: computational coordinates $\xi^i = (\xi,\eta,\zeta)$ with uniform spacing $h_i$;
$J$ the Jacobian (cell volume); $g^{ij} = \nabla\xi^i\!\cdot\!\nabla\xi^j$ the contravariant
metric tensor. Cell $P$ has index $i$ along the axis in question, with neighbours
$i\pm1, i\pm2$.

---

## 1. Convective term

The advective operator in curvilinear form is

$$J\,(\mathbf{u}\cdot\nabla)\phi \;=\; J\,U^i\,\frac{\partial \phi}{\partial \xi^i},
\qquad U^i = \nabla\xi^i\cdot\mathbf{u}$$

so along one axis it reduces to a 1D problem with signed speed $a = J U^i / h$. Two schemes
are available (`convection='sou'` or `'central'` in `build_momentum_matrix_7point`).

### 1a. Second-order upwind (SOU) — the default

Bias the stencil to the **upstream** side, using two cells there:

$$a>0:\quad a\,\frac{\partial\phi}{\partial\xi} \approx a\,\frac{3\phi_i - 4\phi_{i-1} + \phi_{i-2}}{2h}
\qquad
a<0:\quad a\,\frac{-3\phi_i + 4\phi_{i+1} - \phi_{i+2}}{2h}$$

Matrix coefficients (`_convection_coefs`, `scheme='sou'`), with $a$ already divided by $h$:

| | $a_{i-2}$ | $a_{i-1}$ | $a_{P}$ | $a_{i+1}$ | $a_{i+2}$ |
|---|---|---|---|---|---|
| $a > 0$ | $+\tfrac{1}{2}a$ | $-2a$ | $+\tfrac{3}{2}a$ | 0 | 0 |
| $a < 0$ | 0 | 0 | $-\tfrac{3}{2}a$ | $+2a$ | $-\tfrac{1}{2}a$ |

Note the diagonal term $\tfrac{3}{2}|a|$ is **positive in both cases**, which is what buys
diagonal dominance and hence stability.

**Worked example.** Uniform grid $h = 0.1$, $J = 1$, $U = +2$ (so $a = JU/h = 20$), and a
field $\phi = (\dots, \phi_{i-2}, \phi_{i-1}, \phi_i) = (\dots, 1.0, 1.4, 2.2)$:

$$a\,\partial_\xi\phi \approx 20\cdot\frac{3(2.2) - 4(1.4) + 1.0}{2} = 20\cdot\frac{6.6-5.6+1.0}{2} = 20\,(1.0) = 20.0$$

Row entries for cell $i$: $a_{i-2} = +10$, $a_{i-1} = -40$, $a_P = +30$.
Check: $10(1.0) + (-40)(1.4) + 30(2.2) = 10 - 56 + 66 = 20.0$ ✓

**Boundary degradation.** Within two cells of a *wall* the upstream stencil would leave the
domain, so the scheme falls back to 1st-order upwind ($a_P = a$, $a_{i-1} = -a$). On a
**periodic** axis it never degrades — the stencil wraps, so SOU applies everywhere.

**In the code** — `_convection_coefs`, [`phase3_momentum.py`](../phase3_momentum.py). SOU is
built by masking on flow direction and on whether the upstream stencil fits:

```python
# Flow in +xi: 2nd-order upwind reaches back to i-1, i-2
m = pos & can_back
aP[m] += 1.5 * a[m]; c[-1][m] += -2.0 * a[m]; c[-2][m] += 0.5 * a[m]
m = pos & ~can_back                                  # 1st-order fallback
aP[m] += a[m]; c[-1][m] += -a[m]

# Flow in -xi: 2nd-order upwind reaches forward to i+1, i+2
m = (~pos) & can_fwd
aP[m] += -1.5 * a[m]; c[1][m] += 2.0 * a[m]; c[2][m] += -0.5 * a[m]
m = (~pos) & ~can_fwd                                # 1st-order fallback
aP[m] += -a[m]; c[1][m] += a[m]
```

`pos = a > 0` selects the flow direction; `can_back` / `can_fwd` are the two-cell room
checks that trigger the 1st-order fallback near a wall — and are forced `True` on a
periodic axis, where the stencil wraps instead.

### 1b. Second-order central

$$a\,\frac{\partial\phi}{\partial\xi} \approx a\,\frac{\phi_{i+1} - \phi_{i-1}}{2h}$$

| | $a_{i-1}$ | $a_{P}$ | $a_{i+1}$ |
|---|---|---|---|
| any sign of $a$ | $-\tfrac{1}{2}a$ | $\;0\;$ | $+\tfrac{1}{2}a$ |

**Worked example.** Same $a = 20$, with $\phi_{i-1} = 1.4$, $\phi_{i+1} = 3.0$:

$$20\cdot\frac{3.0 - 1.4}{2} = 20\,(0.8) = 16.0$$

Row entries: $a_{i-1} = -10$, $a_P = 0$, $a_{i+1} = +10$. Check: $-10(1.4) + 10(3.0) = 16.0$ ✓

**In the code** — central is an early return, before any of the upwind masking:

```python
if scheme == "central":
    # d(phi)/dxi ~ (phi_{i+1} - phi_{i-1}) / 2h  -- symmetric, no diagonal term
    c[1] += 0.5 * a
    c[-1] += -0.5 * a
    return aP, c
```

`aP` is left untouched: central contributes nothing to the diagonal, which is exactly why
it loses diagonal dominance once the cell Péclet number exceeds 2.

### 1c. Choosing between them

| | SOU | Central |
|---|---|---|
| formal order | 2 | 2 |
| diagonal contribution | $+\tfrac32\lvert a\rvert$ | $0$ |
| leading error | dissipative ($\propto \partial^3\phi$) | dispersive ($\propto \partial^3\phi$) |
| stability | robust at any cell Péclet number | oscillates once $\mathrm{Pe} = \lvert U\rvert h/\nu > 2$ |
| accuracy on smooth flow | slightly damped | sharper |

The **cell Péclet number** decides it. In the 2D cavity at $Re=400$ with $129$ cells,
$\mathrm{Pe} = 1 \times (1/128) / 0.0025 \approx 3.1 > 2$, so central is expected to wobble
there while SOU stays clean. At $Re=100$, $\mathrm{Pe}\approx0.8$ and central should be the
more accurate of the two.

---

## 2. Diffusive term

The curvilinear Laplacian is a **27-point** operator,

$$\nabla^2\phi = \frac{1}{J}\frac{\partial}{\partial\xi^i}\!\left(J g^{ij}\frac{\partial\phi}{\partial\xi^j}\right)$$

which splits into a diagonal ($i=j$) part and 20 cross terms ($i\neq j$).

### 2a. Orthogonal part — implicit, 7-point

Discretised in **conservative, volume-integrated** form with **face-interpolated**
coefficients:

$$\frac{\partial}{\partial\xi}\!\left(Jg^{11}\frac{\partial\phi}{\partial\xi}\right)_P
\approx \frac{c_{i+\frac12}\,(\phi_{i+1}-\phi_i) \;-\; c_{i-\frac12}\,(\phi_i - \phi_{i-1})}{h^2},
\qquad c_{i+\frac12} = \tfrac12\!\left[(Jg^{11})_i + (Jg^{11})_{i+1}\right]$$

Two properties follow, and both are load-bearing:

- **Symmetry by construction.** The face $i+\tfrac12$ writes the *same* value $-c_{i+1/2}/h^2$
  into row $i$ (column $i{+}1$) and row $i{+}1$ (column $i$). So $M = M^{\mathsf T}$ exactly,
  which is what makes CG legal. Using the cell-centred $g^{11}_i$ for both neighbours instead
  breaks this — and CG on a non-symmetric matrix does not converge slowly, it *fails*.
- **Keep $J$ on the operator.** Dividing row $P$ by $J_P$ would scale the two halves of a
  shared face differently and destroy the symmetry. So $J$ stays on the left and moves to the
  RHS of the equation instead.

**Worked example.** $h = 0.1$; $(Jg^{11})$ at cells $i{-}1,i,i{+}1$ = $1.00, 1.20, 1.40$;
$\phi = 2.0, 2.5, 3.5$.

$$c_{i-\frac12} = \tfrac12(1.00+1.20) = 1.10, \qquad c_{i+\frac12} = \tfrac12(1.20+1.40) = 1.30$$
$$\frac{1.30(3.5-2.5) - 1.10(2.5-2.0)}{0.01} = \frac{1.30 - 0.55}{0.01} = 75.0$$

Matrix row (as $M = -[\,\cdot\,]$, positive diagonal): $M_{i,i-1} = -110$,
$M_{i,i+1} = -130$, $M_{i,i} = +240$.
Check: $-110(2.0) + 240(2.5) - 130(3.5) = -220 + 600 - 455 = -75.0 = -(75.0)$ ✓

**In the code** — `build_conservative_diffusion_matrix`, [`phase3_momentum.py`](../phase3_momentum.py).
The symmetry argument is visible in a handful of lines: one `cf` per face, written into
`rows`/`cols` in both orders with the same value, and accumulated into the diagonal of both
adjacent cells:

```python
sl_lo = [slice(None)] * 3; sl_lo[axis] = slice(0, -1)
sl_hi = [slice(None)] * 3; sl_hi[axis] = slice(1, None)

# Face-interpolated coefficient -> one shared value per interior face
cf = 0.5 * (Jg[tuple(sl_lo)] + Jg[tuple(sl_hi)]) / h[axis]**2

lo = idx[tuple(sl_lo)].ravel()
hi = idx[tuple(sl_hi)].ravel()
c = cf.ravel()

rows += [lo, hi]
cols += [hi, lo]
vals += [-c, -c]

diag[tuple(sl_lo)] += cf
diag[tuple(sl_hi)] += cf
```

`rows += [lo, hi]` paired with `cols += [hi, lo]` and the *same* `-c` is what makes
`M == M.T` hold exactly rather than approximately.

### 2b. Cross terms — explicit, deferred

The 20 cross terms would destroy the cheap symmetric 7-point structure, so they are
evaluated at the previous iterate and carried on the RHS (PICT's `nonOrthoFlags`):

$$M\,\phi^{(k+1)} = J\left(\mathcal{D}\big(\Psi(\phi^{(k)})\big) - \text{source}\right)$$

That one-iterate staleness is the **non-orthogonal lag**. It is free on an orthogonal grid
($g^{ij}=0$ for $i\neq j$) and increasingly expensive as skewness grows — the iteration is a
Picard map whose contraction ratio measures 0.31 / 0.59 / 0.92 / **1.27** at grid warp
0.05 / 0.10 / 0.15 / 0.20, i.e. it stops converging near warp 0.18. See
[`piso_equations.md`](piso_equations.md) §4.

---

## 3. Divergence operator

Two divergences appear in this code and they are **not interchangeable**.

### 3a. Differential divergence (Phase 2) — for validating operators

$$\nabla\cdot\mathbf{u} = \frac{1}{J}\left[\frac{\partial (JU)}{\partial\xi}
+ \frac{\partial (JV)}{\partial\eta} + \frac{\partial (JW)}{\partial\zeta}\right]$$

evaluated with wide central differences (`np.gradient`). This is the MMS-validated operator
(2nd order) and is the right thing for *measuring* divergence — but it is the wrong thing to
build a projection from.

### 3b. Flux divergence (Phase 5) — for the projection

Face fluxes are interpolated from cell-centred contravariant components — mirroring PICT's
`computeFluxesNDLoop`, where the line is literally `fluxes[bound] = (velN + velC) * 0.5f`:

$$F_{i+\frac12} = \tfrac12\left[(JU)_i + (JU)_{i+1}\right]$$

and the divergence is the signed sum of the faces of the cell:

$$(\nabla\cdot\mathbf{u})_P = \frac{1}{J_P}\sum_{i}\frac{F_{i+\frac12} - F_{i-\frac12}}{h_i}$$

**Worked example.** $h = 0.1$, $J_P = 1$, $(JU)$ at $i{-}1,i,i{+}1 = 0.8, 1.0, 1.6$, and no
$\eta,\zeta$ variation:

$$F_{i-\frac12} = \tfrac12(0.8+1.0) = 0.90, \qquad F_{i+\frac12} = \tfrac12(1.0+1.6) = 1.30$$
$$(\nabla\cdot\mathbf{u})_P = \frac{1.30 - 0.90}{0.1} = 4.0$$

Note this collapses to $\big[(JU)_{i+1} - (JU)_{i-1}\big]/2h$ — a central difference over
$2h$, so it is 2nd-order accurate but **wide**.

**In the code** — `compute_face_fluxes` and `divergence_from_fluxes`,
[`phase5_fluxes.py`](../phase5_fluxes.py). The interpolation is one line:

```python
f[tuple(interior)] = 0.5 * (JU[axis][lo] + JU[axis][hi])
```

and the divergence is the signed sum over each cell's own faces:

```python
def divergence_from_fluxes(F, J, h):
    """Signed sum of face fluxes over the cell. Mirrors k_computePressureRHSdivergenceFromFlux."""
    d = np.zeros_like(J)
    for axis in range(3):
        lo, hi = _lo_hi(axis)
        d += (F[axis][hi] - F[axis][lo]) / h[axis]
    return d / J

```

`F[axis]` carries one extra entry along its own axis, so `F[hi] - F[lo]` picks out exactly
the pair of faces bounding each cell with no index juggling.

### 3c. Why the distinction matters — measured

The pressure matrix is **compact** (7-point). The Phase-2 divergence and gradient are
**wide**. Composing a wide $\nabla\cdot$ and $\nabla$ with a compact Laplacian means the
correction cannot cancel the divergence it was computed from — the two are different
operators, 9.9e-02 apart:

| projection route | $\lVert\nabla\cdot\mathbf{u}\rVert$ after correction |
|---|---|
| wide-stencil `divergence` + `gradient` | 1.5e-02 &nbsp;*(188× reduction — a floor)* |
| **flux-based, same faces as the matrix** | **7e-13** &nbsp;*(4.4e12× — machine precision)* |

So the projection corrects the **fluxes**, using the same face coefficients the matrix
carries. Continuity is then satisfied to round-off by construction rather than by iteration.

### 3d. Boundary faces

Domain-boundary faces are **prescribed, never interpolated**:

- `from_velocity` — the contravariant component of the boundary velocity (PICT's Dirichlet
  branch). Zero for an impermeable wall, since a velocity tangent to the wall has no
  component along the face normal $J\nabla\xi^i$.
- `impermeable` — hard zero, for a closed domain. Also guarantees the net boundary flux
  vanishes, keeping the singular Neumann pressure system exactly compatible.
- periodic — the seam is an ordinary interior face; the wrap value is stored at both end
  slots so the divergence formula needs no special case.

Defaulting this to zero is a trap worth naming: on a flow with through-boundary velocity it
injects a large spurious divergence at the boundary cells, which the projection then
faithfully "corrects" by wrecking the interior. Measured on an *exactly* divergence-free
Taylor-Green field, zeroing the boundary faces gave a discrete flux divergence of **1.8e+01**.

---

## 4. Summary of stencils

| operator | stencil | treatment | symmetric? |
|---|---|---|---|
| convection, SOU | 5-point per axis ($i\pm1, i\pm2$) | implicit | no |
| convection, central | 3-point per axis | implicit | no (skew-symmetric) |
| diffusion, orthogonal | 7-point | implicit | **yes** |
| diffusion, cross | 20 remaining points of 27 | deferred to RHS | — |
| divergence (projection) | face-based, wide in $JU$ | explicit | — |
| pressure Laplacian | 7-point + deferred cross | implicit | **yes** (CG) |

---

## 5. Reproducing the worked examples

Every number in sections 1–3 is checked against the assembly code itself by
[`verify_discretization_examples.py`](../verify_discretization_examples.py), so this
document cannot drift away from the implementation:

```bash
uv run verify_discretization_examples.py     # 9 checks
```
