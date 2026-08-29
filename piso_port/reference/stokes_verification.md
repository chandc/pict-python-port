# Stokes-flow verification: eigenvalue tests of the PISO solver

## 1. Why Stokes, when MMS already passes

The MMS tests verify that each discrete operator approximates its differential counterpart, and
the Ghia / Poiseuille / duct tests verify that the assembled solver reproduces known *solutions*.
Neither checks the property that governs whether a disturbance in a real calculation grows or
dies: the **spectrum of the linear operator**.

The unsteady Stokes equations

$$\frac{\partial \mathbf{u}}{\partial t} = -\nabla p + \nu \nabla^2 \mathbf{u}, \qquad
\nabla\cdot\mathbf{u} = 0$$

have exact eigenvalues. A scheme can track a decaying solution to plotting accuracy while
getting its eigenvalue wrong by percent, and that error is invisible in a solution-comparison
test but decisive for stability and transition problems. These tests measure the eigenvalue
directly.

Two cases, deliberately different in what they stress:

| | Case A — doubly periodic | Case B — wall-bounded channel |
|---|---|---|
| exact $\sigma$ | closed form $-\nu\|k\|^2$ | no closed form; root of a transcendental condition |
| what sets it | Fourier symmetry the scheme satisfies trivially | the **wall** boundary treatment |
| script | `test_stokes_growth.py` (6/6) | `test_stokes_channel.py` (5/5) |

Case B is the sharper test, and it found things Case A structurally cannot.

---

## 2. Case A — doubly-periodic decay

### 2.1 Problem

Unit square, periodic in $x$ and $y$, thin periodic $z$ (the solver is 3D; a 4-cell periodic
span makes the problem 2D). $\nu = 0.01$.

Modes are built from a streamfunction so $\nabla\cdot\mathbf{u} = 0$ holds to machine precision
by construction, and so oblique modes ($k_1 \ne k_2$) are available:

$$\psi = \cos(k_1 x)\cos(k_2 y),\qquad u = \partial_y\psi,\quad v = -\partial_x\psi,
\qquad k_i = 2\pi m_i$$

$$\boxed{\ \sigma = -\nu\,(k_1^2 + k_2^2)\ }$$

**Not $\sin\cdot\sin$.** That form vanishes identically when $m_2 = 0$, which silently hands the
solver a zero field and makes the measured rate $\log(0/0)$. With $\cos\cdot\cos$ the $m_2 = 0$
case is a genuine mode — a pure shear layer, $u = 0$, $v = k_1\sin(k_1x)$ — worth testing
precisely because it is not diagonal.

### 2.2 Methodology

- Amplitude $A = 10^{-4}$, so the convective term ($O(A^2)$) is negligible against the linear
  terms ($O(A)$). This is verified, not assumed — see §2.3.
- $\sigma$ is measured from the kinetic energy, $\sigma = \tfrac{1}{2}\,\mathrm{d}\ln E/\mathrm{d}t$,
  since $E \propto e^{2\sigma t}$.
- Temporal order uses **Richardson on $\sigma$ differences**, not on the error against the exact
  value: the spatial error is a fixed offset that would otherwise swamp the temporal one.

### 2.3 Results

**Dispersion relation** ($n=48$, $\mathrm{d}t=10^{-3}$, 100 steps):

| mode $(m_1,m_2)$ | $\|k\|^2$ | exact $\sigma$ | measured | rel. err |
|---|---|---|---|---|
| (1,0) shear | 39.5 | −0.3948 | −0.3942 | 1.43e-3 |
| (1,1) | 79.0 | −0.7896 | −0.7884 | 1.43e-3 |
| (2,1) oblique | 197.4 | −1.9739 | −1.9644 | 4.82e-3 |
| (2,2) | 315.8 | −3.1583 | −3.1403 | 5.70e-3 |
| (3,1) oblique | 394.8 | −3.9478 | −3.9020 | 1.16e-2 |

**It really is Stokes.** A 100× amplitude change (1e-4 → 1e-6) moved $\sigma$ by **5.87e-6** —
852× smaller than the discretisation error it sits inside. Convection is not contaminating the
rate, which also means SOU upwinding contributes no measurable dissipation here.

**Spatial order** of the growth rate, $\mathrm{d}t = 2\times10^{-4}$:

| $n$ | 32 | 48 | 64 | 96 |
|---|---|---|---|---|
| \|error\| | 1.99e-2 | 9.26e-3 | 5.29e-3 | 2.38e-3 |
| order | | 1.89 | 1.95 | 1.97 |

Rising monotonically toward 2 — the coarse grids are pre-asymptotic (the (2,1) mode has only 12
points per wavelength at $n=24$), not defective.

**Temporal order** (Richardson, $n=32$):

| scheme | measured | design |
|---|---|---|
| chorin / BE | **0.99** | 1 |
| rotational / BDF2 | **2.01** | 2 |

**Multi-mode disturbance.** A real disturbance is not one eigenmode. Superposing (1,1), (2,1) and
(3,2) and projecting each out afterwards, every mode decays at *its own* rate to ≈1%, though the
fastest decays 6.5× quicker than the slowest — so the solver reproduces the Stokes *spectrum*,
not just one eigenvalue.

---

## 3. Case B — wall-bounded channel

### 3.1 Problem

$y \in [-1,1]$ with **no-slip at both walls**, periodic in $x$ with wavenumber $\alpha = 1$ (so
the box is $2\pi$ long), $\nu = 1$. The least-damped Stokes eigenvalue is

$$\boxed{\ \sigma = -9.313739\ }$$

There is no closed form. Writing $\psi(y)e^{i\alpha x}$, the eigenproblem is

$$\sigma\,(D^2-\alpha^2)\phi = \nu\,(D^2-\alpha^2)^2\phi,
\qquad \phi = \phi' = 0 \ \text{at}\ y=\pm 1$$

**$\phi = \phi' = 0$ *is* no-slip**: with $\psi = \phi(y)\cos(\alpha x)$ we get
$u = \phi'\cos(\alpha x)$ and $v = \alpha\phi\sin(\alpha x)$, so both velocity components vanish
at the wall exactly when $\phi$ and $\phi'$ do.

### 3.2 The reference, and two ways I got it wrong

The reference is a **Chebyshev eigensolve** using Trefethen's clamped $D_4$ construction,
converged across $N$:

| $N$ | 40 | 60 | 80 | 120 |
|---|---|---|---|---|
| $\sigma$ | −9.31373986 | −9.31373985 | −9.31373976 | −9.31373139 |

Two earlier attempts were wrong, both worth recording:

1. **A finite-difference solve that imposed only $\phi = 0$**, not $\phi' = 0$. It returned
   $\alpha^2 + n^2\pi^2$ — the *simply-supported* spectrum, a different physical problem. It
   looked plausible (−39.48, −49.38, −79.09) and disagreed with the analytic route by 2.7×.
2. **Transcendental root-finding** on $\lambda\tan\lambda = -a\tanh a$ (even) and
   $\lambda\cot\lambda = a\coth a$ (odd). The conditions are right but the branch bracketing was
   wrong and it returned nothing usable.

The Chebyshev solve is the trustworthy route and is what the test uses.

### 3.3 Methodology

- The domain is $2\pi$ long, which required threading a `period` argument through
  `compute_numerical_metrics` — see §6.
- Initial condition: the Chebyshev eigenvector interpolated to the solver grid.
- $\sigma$ measured over the **settled window** $[0.05, 0.10]$, never from $t=0$ — see §6.

### 3.4 Results

| $n_y \times n_x$ | measured $\sigma$ | rel. err |
|---|---|---|
| 25 × 18 | −9.348407 | 3.72e-3 |
| 33 × 24 | −9.333265 | 2.10e-3 |
| 49 × 36 | −9.322216 | 9.10e-4 |
| 65 × 48 | −9.318097 | **4.68e-4** |

Spatial order **2.07, 2.11, 2.35**.

**Temporal order:**

| scheme | walled | periodic (Case A) |
|---|---|---|
| chorin / BE | 0.93 | 0.99 |
| rotational / BDF2 | **2.00** | **2.01** |

**CORRECTION.** This table previously reported **1.68** for the walled case and claimed the
scheme "is not second order in time for wall-bounded flow", attributing the shortfall to the
$O(\Delta t^{3/2})$ near-wall splitting error of rotational incremental projection
(Guermond & Shen). **That was wrong.** The 1.68 came from a single Richardson triple measured
outside the asymptotic range. Extending the sweep shows a clean approach to 2:

| triple ($\Delta t$) | 4e-4, 2e-4, 1e-4 | 2e-4, 1e-4, 5e-5 | 1e-4, 5e-5, 2.5e-5 |
|---|---|---|---|
| ratio | 1.74 | 3.20 | **4.00** |
| order | 0.80 | 1.68 | **2.00** |

The scheme **is** second order in time with no-slip walls. The error was the same one already
made and fixed for the *spatial* order in §2.3 (rates 1.76/1.89/1.95 were pre-asymptotic there
too) — and having caught it once, it should have been checked here rather than explained away
with a plausible mechanism. A physical explanation for a shortfall is worth nothing until the
measurement is shown to be converged.

Found by replicating an independent spectral-element solver on this exact problem
(`test_chan_channel.py`), which reported a fitted temporal slope of 1.94–1.99 and prompted the
re-check.

---

## 4. Long-time behaviour: a 100× amplitude drop

`stokes_decay_study.py` integrates to $t = \ln(100)/|\sigma| = 0.4944$, where the amplitude has
fallen 100× (energy $10^4$×), recording the *running* rate $\mathrm{d}\ln A/\mathrm{d}t$.

**The decay itself is clean everywhere**: all ten runs land within 1.6% of the exact 1/100
($A_{\rm end}/A_0 = 0.00984$–$0.00998$ against 0.01), no run develops a floor or goes unstable.
The spread tracks resolution — the 0.00984 is the coarsest grid — so it is discretisation error,
not drift in the integration.

**The rate is the informative part** (grid sweep at $\mathrm{d}t = 2\times10^{-4}$):

| $n_y$ | $\sigma_{\rm settled}$ | rel. err | drift over the run |
|---|---|---|---|
| 25 | −9.347173 | 3.59e-3 | +3.5e-6 |
| 33 | −9.333337 | 2.10e-3 | +1.5e-6 |
| 49 | −9.323409 | 1.04e-3 | −3.7e-6 |
| 65 | −9.320994 | 7.79e-4 | −1.0e-3 |
| 97 | −9.320714 | 7.49e-4 | **−8.4e-3** |

Refining the grid at fixed $\mathrm{d}t$ **stops helping and starts hurting**: the error stalls
at 7.5e-4 while the drift grows by three orders of magnitude. Refining $\mathrm{d}t$ at fixed
grid converges to a floor of 9.34e-4 with drift 4e-7. Neither sweep reaches the exact value —
one is temporal-limited, the other spatial-limited.

---

## 5. The 5×5 error matrix — and why single-parameter studies mislead

`stokes_error_matrix.py`, relative error in $\sigma$:

| $n_y$＼$\mathrm{d}t$ | 4e-4 | 2e-4 | 1e-4 | 5e-5 | 2.5e-5 |
|---|---|---|---|---|---|
| 25 | 3.67e-3 | 3.59e-3 | 3.57e-3 | 3.57e-3 | 3.57e-3 |
| 33 | 2.23e-3 | 2.11e-3 | 2.07e-3 | 2.06e-3 | 2.06e-3 |
| 49 | 9.14e-4 | 1.02e-3 | 9.58e-4 | 9.39e-4 | 9.35e-4 |
| 65 | 4.02e-4 | 5.38e-4 | 5.69e-4 | 5.39e-4 | 5.31e-4 |
| 97 | **1.34e-4** | 1.69e-4 | 2.30e-4 | 2.56e-4 | 2.42e-4 |

**Spatial order** along the $\mathrm{d}t\to 0$ column: **1.91, 1.95, 1.97, 1.94** — clean second
order converging to zero.

**The two error contributions have opposite signs.** The spatial error **over**-damps; the
temporal error **under**-damps. Consequences:

- Along the $n_y = 97$ row, **refining $\mathrm{d}t$ makes the answer worse** (1.34e-4 → 2.42e-4)
  because it removes a fortuitous cancellation. The best single entry in the whole matrix is the
  *coarsest* time step at the finest grid, and that is luck, not accuracy.
- A "temporal order" computed from the total error is meaningless here; doing so yields
  −0.34, −0.44, −0.16, +0.08. The correct measurement is Richardson on $\sigma$ *differences* at
  fixed grid, which cancels the spatial offset and gives **1.67**, matching §3.4.
- **Refine both together.** Down the diagonal the error falls monotonically; along any single row
  or column it stalls or reverses.

### Is this numerical dissipation?

No — not in the artificial-viscosity sense, and the sign comparison settles it:

- **Periodic modes are consistently UNDER-damped** (−0.7884 vs −0.7896 exact, etc.), which is
  textbook central differencing: the discrete Laplacian eigenvalue $(4/h^2)\sin^2(kh/2) < k^2$.
- **Walled modes are consistently OVER-damped** (−9.3472 … −9.3207 vs −9.3137 exact).

A dissipative mechanism — upwinding, artificial viscosity, a filter — acts the same way under
either boundary condition. It cannot flip sign. Combined with the amplitude-independence result
(§2.3) and the second-order convergence of the $\mathrm{d}t\to0$ column **to zero**, there is no
persistent dissipation floor: the over-damping is ordinary wall-treatment and time-integration
error converging at the design rate.

---

## 6. Defects these tests exposed

**In the solver:**

- `compute_numerical_metrics` never passed `period` through to `wrap_pad_coords`, hardcoding the
  ghost shift to 1. Any periodic domain of length $\ne 1$ — such as the $2\pi$ box Case B needs —
  would have had a corrupted seam and a collapsed Jacobian. The parameter existed downstream but
  was unreachable. Fixed and verified: $J = 12.566 = 2\pi\times2$, GCL 4e-14, unit-period
  behaviour unchanged.

**In the tests themselves** (recorded so they are not repeated):

- $\psi = \sin\cdot\sin$ vanishes identically at $m_2 = 0$; the (1,0) mode ran on a zero field and
  returned `nan`.
- The temporal-order window was placed at $[0, 0.02]$, **inside the startup transient**. The
  initial field is an eigenmode of the *continuous* operator, not the discrete one, so an early
  window measures relaxation rather than the eigenvalue. It reported order **0.04** for a scheme
  that is genuinely ~1.7. Local rates at $\mathrm{d}t=10^{-4}$ read −9.32155 over $[0,0.02]$ and
  settle to a flat −9.32266 from $t = 0.02$ onward.
- An amplitude-independence threshold of 1e-6 was arbitrary; the meaningful bar is whether the
  amplitude effect is small *relative to the discretisation error being measured*.
- Coarse-grid convergence rates (1.76, 1.89) were judged against a 1.8 bar when the study simply
  had not reached the asymptotic range. The fix was to refine, not to lower the bar.

---

## 7. What this establishes, and what it does not

**Established.** The solver reproduces the Stokes spectrum: closed-form eigenvalues to 0.1–1% in
a periodic box, and a wall-bounded eigenvalue with no closed form to 4.7e-4. Spatial accuracy is
second order against an exact eigenvalue (1.91–2.35 across four independent studies). Temporal
accuracy is second order both periodic and walled (2.01 and 2.00). There is no numerical
dissipation floor.

**Not established.** These are all *linear* tests — the convective term is deliberately
suppressed. They say nothing about energy transfer, aliasing, or whether the convective
discretisation conserves kinetic energy in the inviscid limit, which is the property that governs
turbulence. See `run_tgv3d.py` for the nonlinear counterpart, and note that the inviscid
energy-conservation check remains unwritten.
