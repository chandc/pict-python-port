# CNN-Coupled FEM Solver Documentation

This document explains the mathematical formulation and provides a step-by-step breakdown of the code inside `cnn_fem_poisson.py`.

## 1. Mathematical Formulation

### The Physics: 2D Poisson Equation
The underlying physics we are solving is the 2D Poisson equation with a constant source term $f$:
$$ \nabla^2 u = f $$
With zero Dirichlet boundary conditions ($u = 0$) on the edges of a unit square domain.

### The Galerkin Finite Element Method (FEM)
To solve this on a computer, we use the Galerkin FEM. We multiply the equation by a test function $v$, integrate over the domain, and apply the divergence theorem to reach the **Weak Form**:
$$ \int_\Omega \nabla u \cdot \nabla v \, dx = - \int_\Omega f v \, dx $$

By discretizing the domain into a triangular mesh and using linear basis functions, this continuous integral turns into a massive system of linear algebraic equations:
$$ K u = F $$
Where:
*   $K$ is the global stiffness matrix.
*   $F$ is the load vector (forcing term).
*   $u$ is the unknown field (e.g., velocity or pressure) at the mesh nodes.

### The Machine Learning Coupling (Sub-Grid Scale Correction)
When we run this solver on a **coarse grid** (e.g., 10x10), it lacks the resolution to capture the exact physics, leading to discretization error. 

To fix this, we couple a Convolutional Neural Network (CNN) to the solver. The CNN looks at the coarse grid and predicts a localized force correction ($\Delta F$) for every node:
$$ \Delta F = \text{CNN}(\text{Grid Coordinates}) $$

We inject this correction into the physics solver:
$$ K_{coarse} u_{coarse} = F_{coarse} + \Delta F $$

### The Discrete Adjoint Method (Training)
We define our Loss function as the Mean Squared Error (MSE) between our coarse solution and a down-sampled high-resolution "Target" solution ($u_{target}$):
$$ \text{Loss} = \text{MSE}(u_{coarse}, \, u_{target}) $$

To train the CNN, we must find the gradient of the Loss with respect to the Neural Network's internal weights ($W$). By applying the Chain Rule, we get:
$$ \frac{\partial \text{Loss}}{\partial W} = \frac{\partial \text{Loss}}{\partial u_{coarse}} \cdot \frac{\partial u_{coarse}}{\partial F_{total}} \cdot \frac{\partial F_{total}}{\partial W} $$

Here is how this is computed step-by-step during the backward pass:

**1. The Loss Derivative**
PyTorch first calculates how the error changes with respect to the output velocity:
$$ \frac{\partial \text{Loss}}{\partial u_{coarse}} = 2(u_{coarse} - u_{target}) $$
*(Let's call this vector $g$)*

**2. The Discrete Adjoint Solve (Bypassing the Physics)**
Next, PyTorch must backpropagate through the implicit physics solver ($K_{coarse} u_{coarse} = F_{total}$). 

To find $\frac{\partial \text{Loss}}{\partial F_{total}}$, we apply the chain rule:
$$ \frac{\partial \text{Loss}}{\partial F_{total}} = \frac{\partial \text{Loss}}{\partial u_{coarse}} \cdot \frac{\partial u_{coarse}}{\partial F_{total}} $$

If we take the derivative of our forward equation ($K u = F$) with respect to $F$, we get $K \frac{\partial u}{\partial F} = I$, which means $\frac{\partial u}{\partial F} = K^{-1}$.
Plugging this and our error vector $g^T = \frac{\partial \text{Loss}}{\partial u}$ into the chain rule gives:
$$ \frac{\partial \text{Loss}}{\partial F_{total}} = g^T K^{-1} $$

Computing an inverse matrix ($K^{-1}$) is computationally disastrous for massive physics problems. 
To bypass this, the **Discrete Adjoint Method** uses matrix transposition $(AB)^T = B^TA^T$ on both sides:
$$ \left( \frac{\partial \text{Loss}}{\partial F_{total}} \right)^T = (g^T K^{-1})^T = K^{-T} g $$

We invent a dummy variable called the adjoint variable ($\lambda$) and set it equal to our target derivative $\lambda = \left( \frac{\partial \text{Loss}}{\partial F_{total}} \right)^T$.
$$ \lambda = K^{-T} g $$
Multiply both sides by $K^T$ to remove the inverse, leaving us with the Adjoint Equation:
$$ K^T \lambda = g $$

By solving this brand new linear equation, PyTorch instantly finds $\lambda$ (which is our exact derivative) without ever inverting the matrix!

This step relies on two powerful computational tricks:

*   **Trick A: Exploiting Symmetry ($K = K^T$)**
    The mathematical definition of the Adjoint Equation requires solving the *transpose* of the physics matrix ($K^T$). However, because our specific physics problem (Galerkin FEM for the Poisson equation) results in a Symmetric Stiffness Matrix, the matrix is a perfect mirror image of itself diagonally ($K = K^T$). This means PyTorch doesn't have to waste time or memory computing a transposed matrix; it directly reuses the exact same physics matrix from the forward pass, solving $K_{coarse} \lambda = g$.
*   **Trick B: Bypassing Loop Unrolling**
    When PyTorch solves $K u = F_{total}$ in the forward pass, the linear solver takes hundreds of tiny iterative steps in a "while-loop" to converge on the answer. Normally, standard backpropagation acts like a tape recorder—it records *every single math operation* in that loop to apply the chain rule in reverse. Recording hundreds of solver loops (called "unrolling the loop") creates a monstrously long chain rule that quickly crashes computer memory. The Adjoint Method acts as a cheat code: it tells PyTorch to throw away the tape recorder, ignore the iterative loops, and simply fire up the linear solver a second time to solve $K \lambda = g$. The final answer ($\lambda$) is mathematically proven to be the exact derivative we need.

Thus, the solver finds $\lambda$ cleanly and efficiently. The rules of the adjoint method prove that $\lambda$ perfectly equals $g^T K_{coarse}^{-1}$, successfully crossing the physics barrier:
$$ \frac{\partial \text{Loss}}{\partial F_{total}} = \lambda $$

**3. The Neural Network Update (Autograd & Optimizer)**
Once $\lambda$ successfully crosses the physics barrier, PyTorch's automatic differentiation engine (Autograd) takes over to compute the final term: $\frac{\partial F_{total}}{\partial W}$.

Since $F_{coarse}$ is a constant physics vector, its derivative is zero, meaning $\frac{\partial F_{total}}{\partial W}$ is exactly equal to $\frac{\partial \Delta F}{\partial W}$. Because $\Delta F$ is the direct output of our CNN, Autograd applies the standard chain rule layer-by-layer backwards through the network's `Conv2d` and `Tanh` layers:
$$ \frac{\partial \text{Loss}}{\partial W} = \lambda \cdot \frac{\partial \Delta F}{\partial W} $$
By the time `loss.backward()` finishes, PyTorch has calculated the exact gradient for every single weight and bias in the CNN and stored them in the `.grad` attribute of each weight tensor.

Finally, the **Adam Optimizer** looks at those `.grad` values and updates the weights when we call `optimizer.step()`. At its core, it adjusts the weights "downhill" in the opposite direction of the gradient to minimize the loss:
$$ W_{new} = W_{old} - (\text{learning\_rate} \times \frac{\partial \text{Loss}}{\partial W}) $$
*(Note: Adam improves upon basic Stochastic Gradient Descent by adding momentum and adaptive learning rates for each specific weight, speeding up convergence).*

---

## 2. Step-by-Step Code Breakdown (`cnn_fem_poisson.py`)

### Step 1: The FEM Utilities
*   **`generate_mesh(nx, ny)`**: Uses `scipy.spatial.Delaunay` to create a structured triangular mesh.
*   **`assemble_system()`**: Iterates over every triangle, calculates its area, computes the analytical gradients of the linear shape functions, and builds the local stiffness matrix $K_{local}$. It then accumulates these into the massive global NumPy arrays $K$ and $F$.
*   **`apply_boundary_conditions()`**: Identifies nodes on the edges of the box and forces $u=0$ by altering the $K$ matrix (putting a 1 on the diagonal) and setting $F=0$.

### Step 2: The CNN Model (`SGS_CNN`)
*   We define a simple PyTorch `nn.Module` consisting of three `Conv2d` layers separated by `Tanh` activations.
*   **Input**: The (x, y) coordinates of the 10x10 grid, reshaped into an "image" of size `(1, 2, 10, 10)`.
*   **Output**: The forcing correction $\Delta F$ for each node, flattened back into a 1D vector of length 100 to plug into the FEM solver.

### Step 3: Generating the High-Res Target
*   We generate a 30x30 mesh and assemble the physics ($K_{hr}$, $F_{hr}$).
*   We solve it normally to get our highly accurate "ground truth" ($u_{hr}$).
*   We generate our coarse 10x10 mesh, and use `scipy.interpolate.griddata` to map the 30x30 ground truth down onto the 10x10 nodes. This creates our `u_target_cr_tensor`.

### Step 4: The Differentiable Training Loop
We run a standard PyTorch training loop for 1,000 epochs:
1.  **Forward Pass**: The CNN looks at the coarse nodes and predicts `dF`.
2.  **Physics Injection**: We add the prediction to the baseline forcing: `F_total = F_cr_tensor + dF`.
3.  **Boundary Conditions**: We ensure the CNN hasn't altered the forcing on the boundary edges by manually zeroing out `F_total_bc` along the edges.
4.  **Implicit Solve**: We pass the modified forcing into PyTorch's linear solver: `u_pred = torch.linalg.solve(...)`.
5.  **Loss & Backprop**: We calculate the MSE between `u_pred` and `u_target`. We call `loss.backward()`, which triggers the Adjoint Method inside PyTorch to pass gradients back through `linalg.solve` and update the CNN via `optimizer.step()`.

### Step 5: Error Analysis & Visualization
*   The script calculates the exact Root Mean Square (RMS) error of both the uncorrected and CNN-corrected coarse grids against the target.
*   Finally, it utilizes `matplotlib`'s `tripcolor` to render a 3-panel visual comparison showing the dramatic improvement caused by the CNN.
