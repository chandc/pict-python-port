# Educational Python PICT: Single-Domain 3D Curvilinear FVM PISO Port

**Goal:** Translate the complex C++/CUDA PICT solver into a pure Python/NumPy library for educational purposes, focusing on a single domain but fully preserving the complex physics of **3D Curvilinear (Non-Orthogonal) Grids**.

**Workspace:** All work for this educational port will be isolated in a dedicated new directory: `/Users/danielchan/Dropbox/PICT/educational_python_port/`.

**Execution Rule:** Always use `uv run <script.py>` to run any Python jobs for this project, instead of invoking the python interpreter directly.

---

## 1. Mathematical Formulation (3D Curvilinear Staggered Grid)

To solve equations on a warped/skewed 3D mesh, we map the physical coordinates $(x,y,z)$ to a perfectly uniform 3D computational space $(\xi, \eta, \zeta)$ where $\Delta \xi = \Delta \eta = \Delta \zeta = 1$.

### Metric Tensors & Contravariant Velocity
The mapping requires computing the massive 3D Jacobian $J$ (cell volume) and a full 3x3 matrix of metric derivatives (e.g., $\xi_x, \xi_y, \dots, \zeta_z$). 
Instead of standard physical velocities $(u, v, w)$, fluid flow across the warped 3D cell faces is tracked using **Contravariant Velocities** $(U, V, W)$:
$$ U = \xi_x u + \xi_y v + \xi_z w $$
$$ V = \eta_x u + \eta_y v + \eta_z w $$
$$ W = \zeta_x u + \zeta_y v + \zeta_z w $$

### Transformed 3D Navier-Stokes Equations & Time Marching
1.  **Continuity (Mass Conservation):** The divergence operator incorporates the 3D Jacobian.
    $$ \frac{\partial}{\partial \xi}(J U) + \frac{\partial}{\partial \eta}(J V) + \frac{\partial}{\partial \zeta}(J W) = 0 $$
2.  **Momentum Predictor (1st-Order Euler Implicit):** To match the core PICT C++ implementation (`SetupAdvectionMatrixEulerImplicit`), we will use the unconditionally stable **Backward Euler** time marching scheme. The transient, advection, and diffusion terms are evaluated implicitly at time $n+1$ and assembled into the sparse matrix $A$.
    $$ A \mathbf{u}^* = \frac{\rho}{\Delta t} \mathbf{u}^n - \nabla_{\xi,\eta,\zeta} p^n $$
3.  **Pressure Poisson Equation:** The Laplacian $M$ expands into a massive 27-point 3D stencil due to the cross-derivative terms introduced by 3D grid skewness.
    $$ \nabla \cdot (A^{-1} \nabla p^{n+1}) = \nabla \cdot \mathbf{U}^* $$
4.  **Velocity Corrector:** The pressure gradients are transformed back into physical 3D space to correct the velocity.

---

## 2. Step-by-Step Conversion & Testing Plan

To ensure the complex 3D curvilinear math is implemented perfectly, we will use the **Method of Manufactured Solutions (MMS)**. This involves feeding a known analytical equation into the solver and proving the numerical code recovers it.

### Phase 1: 3D Mesh & Metric Generation (MMS: 3D Wavy Grid)
*   **Implementation:** Create a `Domain3D` Python class inside the new dedicated folder. Calculate the Jacobian $J$ and all 9 metric tensors at both cell centers and cell faces using NumPy gradients.
*   **The MMS Setup:** To strictly validate the grid metrics, we will manufacture a known **3D Wavy Grid** mapping:
    $$ x(\xi, \eta, \zeta) = \xi + A_x \sin(\pi \eta) \sin(\pi \zeta) $$
    $$ y(\xi, \eta, \zeta) = \eta + A_y \sin(\pi \xi) \sin(\pi \zeta) $$
    $$ z(\xi, \eta, \zeta) = \zeta + A_z \sin(\pi \xi) \sin(\pi \eta) $$
    Using calculus, we will compute the *exact* analytical equations for all 9 metric tensors ($\xi_x, \xi_y$, etc.) and the exact analytical equation for the Jacobian $J$.
*   **Pass Criteria:** 
    1.  **Metric Accuracy:** The numerical metrics calculated by our Python code must match the exact analytical wavy grid equations. The $L_2$ error norm must decrease at exactly $O(\Delta^2)$ (second-order convergence) when the grid is refined.
    2.  **3D Geometric Conservation Law (GCL):** The metrics must perfectly preserve uniform flow. The sum of the face metrics for a closed cell (e.g., $\frac{\partial}{\partial \xi}(J \xi_x) + \frac{\partial}{\partial \eta}(J \eta_x) + \frac{\partial}{\partial \zeta}(J \zeta_x)$) must equal zero down to machine precision ($< 10^{-14}$).

### Phase 2: 3D Differential Operators (MMS: Trigo-Exponential Fields)
*   **Implementation:** Write vectorized NumPy functions for 3D curvilinear `divergence` and `gradient` that consume the massive metric tensors computed in Phase 1.
*   **The MMS Setup:** We will invent highly non-linear 3D analytical fields that exercise all spatial derivatives on our Wavy Grid:
    *   **Pressure Field:** $p(x,y,z) = \sin(2\pi x) \cos(2\pi y) e^z$
    *   **Velocity Field:** $\mathbf{V}(x,y,z) = [ \cos(\pi x)\sin(\pi y)\sin(\pi z), \;\sin(\pi x)\cos(\pi y)\sin(\pi z), \;\sin(\pi x)\sin(\pi y)\cos(\pi z) ]$
    Using calculus, we will compute the exact analytical gradient ($\nabla p$) and exact analytical divergence ($\nabla \cdot \mathbf{V} = -3\pi \sin(\pi x) \sin(\pi y) \sin(\pi z)$).
*   **Testing:** We will evaluate the analytical $p$ and $\mathbf{V}$ on our Wavy Grid nodes, transform the velocity into contravariant components using the metrics, and run them through our numerical `divergence()` and `gradient()` Python functions.
*   **Pass Criteria:** The output of our Python operators must match the exact analytical calculus derivations, with the $L_2$ norm of the numerical error decreasing at a strict 2nd-order convergence rate when the 3D grid resolution is doubled.

### Phase 3: Momentum Matrix Assembly (MMS: 3D Taylor-Green Vortex)
*   **Implementation:** Build the massive SciPy sparse matrix $A$ for 3D curvilinear advection and diffusion. Solve $A \mathbf{u}^* = \text{RHS}$ using `scipy.sparse.linalg.bicgstab`.
*   **The MMS Setup:** We will use the classic **3D Taylor-Green Vortex** as our analytical velocity field:
    *   $u(x,y,z) = \sin(x) \cos(y) \cos(z)$
    *   $v(x,y,z) = -\cos(x) \sin(y) \cos(z)$
    *   $w(x,y,z) = 0$
    We will mathematically derive the exact advection-diffusion source term $S = (\mathbf{u} \cdot \nabla)\mathbf{u} - \nu \nabla^2 \mathbf{u}$ required to perfectly sustain this vortex in physical space.
*   **Testing:** We will feed that analytical source term $S$ into our numerical matrix solver as the $\text{RHS}$. 
*   **Pass Criteria:** The `bicgstab` solver must successfully converge, and the output velocity field must match the Taylor-Green Vortex equations within a $10^{-4}$ tolerance despite the highly skewed 3D computational grid.

### Phase 4: Poisson Matrix Assembly & Solver (MMS: Trigo-Exponential Laplacian)
*   **Implementation:** Construct the complex 27-point discrete 3D curvilinear Laplacian matrix $M$. Solve $M p^{n+1} = D$ using `scipy.sparse.linalg.cg`.
*   **The MMS Setup:** We will recycle the exact same analytical pressure field from Phase 2: $p(x,y,z) = \sin(2\pi x) \cos(2\pi y) e^z$.
    Using calculus, we derive the exact analytical Laplacian source term $f = \nabla^2 p$ for this field:
    $$ f(x,y,z) = (1 - 8\pi^2) \sin(2\pi x) \cos(2\pi y) e^z $$
*   **Testing:** We will feed this exact analytical $f$ into our massive 27-point matrix solver as the $\text{RHS}$.
*   **Pass Criteria:** The Conjugate Gradient solver must converge, and the predicted pressure field must match the exact analytical solution $p(x,y,z)$ regardless of 3D grid skewness.

### Phase 5: PISO Orchestration & Final Validation
*   **Implementation:** Tie Predictor $\rightarrow$ Poisson $\rightarrow$ Corrector together into the final `piso_numpy_3d.py` script inside the new folder.
*   **Testing 1 (3D Lid-Driven Cavity):** Simulate the classic 3D Lid-Driven Cavity on a heavily warped, non-orthogonal 3D grid. 
*   **Testing 2 (3D Square Duct Flow):** Simulate fully developed 3D flow in a square pipe (rectangular duct) driven by a constant pressure gradient.
*   **Pass Criteria:** 
    1. The 3D Lid-Driven Cavity velocity field reaches steady-state matching standard literature benchmarks (e.g., Albensoeder & Kuhlmann for 3D cavities).
    2. The fully developed 3D duct flow perfectly matches the analytical series solution for laminar flow in a rectangular cross-section.
    3. Global 3D mass conservation (divergence) must rigidly remain $< 10^{-7}$ at every single time-step.

---
> [!IMPORTANT]
> **User Review Required**
> The plan now explicitly names **1st-Order Euler Implicit** as the time marching scheme for the Momentum Predictor, which perfectly matches the underlying CUDA implementation in PICT.
> 
> If you are ready to begin, please click **Proceed** and I will create the new folder and start coding Phase 1!
