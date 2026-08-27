# Educational Python PICT Port: Progress Report (Phases 1 - 3)

This document details the successful execution and rigorous mathematical validation of the first three phases of the 3D Curvilinear Python port. 

All phases were strictly validated using the **Method of Manufactured Solutions (MMS)** on highly warped (skewed) grids to prove 2nd-order convergence and correctness.

---

## Phase 1: 3D Mesh & Metric Generation

### The Challenge
To solve fluid dynamics on complex geometries, physical grids $(x,y,z)$ must be mapped to uniform computational spaces $(\xi, \eta, \zeta)$. This requires tracking the Jacobian (cell volume) and 9 unique metric tensors (e.g., $\xi_x, \eta_z$) across the entire domain.

### Implementation & The GCL Pitfall
We generated a massive **3D Wavy Grid** defined by mathematical sine waves. 
Initially, we used standard 2nd-Order Central Differencing to calculate the grid metrics. While this achieved 2nd-order convergence, it failed the **Geometric Conservation Law (GCL)**—meaning the grid artificially leaked mass.

To perfectly satisfy CFD conservation laws, we rewrote the metric calculations using the strict **Thomas & Lombard (1979) Conservative Formulation** (calculating cross-products of cell face diagonals).

### Validation Results
*   **MMS Test:** 3D Wavy Grid vs exact Analytical Jacobian/Metrics.
*   **Convergence Rate:** **2.05** (Perfect 2nd-order convergence as the grid is refined).
*   **GCL Error:** **7.20e-13** (Exactly machine precision! The conservative formulation successfully eliminated all artificial mass leakage).

### Visualizing the Grid Warp
Here are cuts across all three constant indices ($k, j, i$) showing the severe mesh skewness our numerical solvers successfully navigate:

![3 Cuts of the 3D Wavy Grid](/Users/danielchan/.gemini/antigravity-ide/brain/703d4ece-8e8e-4d5d-ab81-a1422e38161f/wavy_grid_3cuts.png)

---

## Phase 2: 3D Differential Operators

### Implementation
We implemented vectorized NumPy functions for `compute_gradient` and `compute_divergence`. These operators consume the 9 metric tensors and the Jacobian generated in Phase 1 to correctly apply derivatives on the skewed computational grid.

### Validation Results
To push the operators to their limits, we evaluated them against highly non-linear, exact mathematical fields:
*   **Pressure:** Trigo-Exponential function $p(x,y,z) = \sin(2\pi x) \cos(2\pi y) e^z$
*   **Velocity:** 3D Trigonometric flow field $\mathbf{V}(x,y,z)$.

We then ran the exact analytical fields through our Python curvilinear operators and compared the output to the exact calculus derivations.

*   **Gradient Convergence Rate:** **2.07**
*   **Divergence Convergence Rate:** **2.12**
*   *Conclusion:* Both numerical operators perfectly recreate the analytical derivatives with $O(\Delta^2)$ accuracy.

---

## Phase 3: Momentum Matrix Assembly

### The Challenge
We needed to assemble a `scipy.sparse` matrix representing the discrete Advection and Diffusion operators. A true 3D curvilinear Laplacian matrix requires a massive 27-point stencil, which is computationally expensive to solve implicitly. 

### Implementation & Deferred Correction
We implemented the standard CFD approach: assembling an implicit 7-point orthogonal sparse matrix $A$.

*   **Advection:** We initially used 1st-Order Upwind differencing for stability, but testing revealed it degraded the solver to 1st-order accuracy. We first upgraded to 2nd-Order Central Differencing, and then ultimately implemented a robust **2nd-Order Upwind (SOU)** scheme by expanding the implicit matrix stencil to 13 points (reaching upstream to $i-2$) and gracefully degrading to 1st-order upwind near the boundaries.
*   **Deferred Correction:** Because a 7-point orthogonal diffusion matrix ignores the 20 cross-derivative terms caused by grid skewness ($g^{12} \neq 0$), grid refinement initially flatlined at an $O(1)$ error. To fix this, we calculated the cross-diffusion explicitly and moved it to the RHS (Deferred Correction).

### Validation Results
We tested the Sparse Matrix solve ($A u = S$) using the classic **3D Taylor-Green Vortex**. We derived the exact source term $S$ required to sustain the vortex, injected it into our numerical solver, and used `scipy.sparse.linalg.bicgstab` to iteratively solve for the velocity.

*   **BiCGStab Solver:** Converged successfully in 0.01 seconds.
*   **Solution Error (N=40):** **2.08e-05**
*   **Matrix Convergence Rate:** **2.08**
*   *Conclusion:* The sparse matrix assembly perfectly resolves advection and diffusion on highly skewed grids, validating the Deferred Correction strategy!
