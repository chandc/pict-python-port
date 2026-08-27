# PICT PISO Scheme: Forward and Backward Pass with CNN Coupling

This document provides a highly detailed mathematical breakdown of how the PICT solver executes the PISO (Pressure Implicit with Splitting of Operators) algorithm in the forward pass, and how it uses the Discrete Adjoint Method to perfectly backpropagate errors into a coupled Convolutional Neural Network (CNN) during training.

---

## 1. The Forward Pass (Solving the Physics)

The Navier-Stokes equations for incompressible fluid flow require solving for both velocity and pressure simultaneously. Because this is computationally difficult, the PISO algorithm splits the process into a "Predictor-Corrector" sequence. 

Assume our CNN looks at the current fluid state and predicts a Sub-Grid Scale correction force: $\Delta F_{SGS} = \text{CNN}(\text{state})$.

### Step 1. The Momentum Predictor
We construct a momentum matrix $A$ (which contains both advection and diffusion terms) and solve an implicit linear system for an intermediate "predictor" velocity $u^*$:
$$ A \, u^* = H(u^n) - \nabla p^n + \Delta F_{SGS} $$
*(Note: $u^*$ accounts for momentum and viscosity, but it is generally not divergence-free. It violates the physical law of fluid incompressibility).*

### Step 2. The Pressure Poisson Solve
Because $u^*$ is unphysical, we must correct it. We assume the true physical velocity $u^{n+1}$ can be constructed by taking our predictor $u^*$ and subtracting the effect of the pressure gradient:
$$ u^{n+1} = u^* - A^{-1} \nabla p^{n+1} $$

We have a strict physical requirement: the final velocity must be incompressible (divergence-free), meaning $\nabla \cdot u^{n+1} = 0$.
If we take the divergence of both sides of our assumed equation, we get:
$$ 0 = \nabla \cdot u^* - \nabla \cdot (A^{-1} \nabla p^{n+1}) $$
Rearranging this gives us the **Pressure Poisson Equation**:
$$ \nabla \cdot (A^{-1} \nabla p^{n+1}) = \nabla \cdot u^* $$

In discrete matrix form, let $M$ be the Laplacian-like operator $\nabla \cdot (A^{-1} \nabla)$, and $D$ be the discrete Divergence operator. We set up this massive linear system to find the exact pressure $p^{n+1}$ that will push back against the compression:
$$ M \, p^{n+1} = D \, u^* $$

**Bridging Step 2 and Step 3**
Once the linear solver finds the new pressure field $p^{n+1}$, we know exactly how much force is required to eliminate the unphysical divergence. We substitute this pressure back into our assumed velocity equation.

### Step 3. The Velocity Corrector
We use the gradient of the newly found pressure to correct the intermediate velocity. Let $G$ be the discrete Gradient operator ($G = -D^T$). The final, physical, divergence-free velocity is:
$$ u^{n+1} = u^* - A^{-1} G \, p^{n+1} $$

---

## 2. The Backward Pass (Adjoint Backpropagation)

During training, we define a Loss function (like MSE against a high-res target fluid state). PyTorch calculates how the error changes with respect to the final velocity: $\frac{\partial \text{Loss}}{\partial u^{n+1}}$. 

To train the CNN, we must apply the chain rule in reverse all the way back to $\Delta F_{SGS}$.

### Step 1. Reverse Corrector (Splitting the Gradients)
First, we trace the gradient back through the velocity corrector step ($u^{n+1} = u^* - A^{-1} G p^{n+1}$). By applying standard matrix calculus, the gradient splits into two distinct paths:
$$ \frac{\partial \text{Loss}}{\partial u^*} = \frac{\partial \text{Loss}}{\partial u^{n+1}} $$
$$ \frac{\partial \text{Loss}}{\partial p^{n+1}} = -G^T A^{-T} \frac{\partial \text{Loss}}{\partial u^{n+1}} $$

### Step 2. Reverse Poisson (The First Adjoint Solve)
Next, we must backpropagate through the massive Poisson solve ($M p^{n+1} = D u^*$). 
Applying the chain rule gives $\frac{\partial \text{Loss}}{\partial (D u^*)} = \left(\frac{\partial \text{Loss}}{\partial p^{n+1}}\right)^T M^{-1}$. 
Because inverting $M$ is computationally disastrous, we use the **Discrete Adjoint Method**. We apply matrix transposition to both sides and invent an adjoint variable $\lambda_p$:
$$ M^T \lambda_p = \frac{\partial \text{Loss}}{\partial p^{n+1}} $$

*   **Symmetry Trick:** The discrete Laplacian $M$ is symmetric ($M = M^T$). PICT simply re-uses the Poisson matrix from the forward pass ($M \lambda_p = \dots$) to instantly find the exact derivative without unrolling any iterative loops!

Once the linear solver finds $\lambda_p$, we have successfully found the gradient with respect to the entire right-hand side of the Poisson equation: $\frac{\partial \text{Loss}}{\partial (D u^*)} = \lambda_p$.

**Bridging Step 2 and Step 3: Accumulating the Gradients into $u^*$**
To move backward into the Predictor step, we must translate this gradient from $(D u^*)$ into just $u^*$. 
We do this by applying the chain rule through the linear Divergence operator $D$. The derivative of $D u^*$ with respect to $u^*$ is simply the matrix $D$. Thus:
$$ \left( \frac{\partial \text{Loss}}{\partial u^*} \right)_{from\_pressure} = D^T \lambda_p $$

Now, recall that in the forward PISO algorithm, the intermediate velocity $u^*$ influences the final output through **two distinct paths**:
1.  **Path A (Direct):** It is directly added in the Velocity Corrector step ($u^{n+1} = u^* - \dots$).
2.  **Path B (Indirect):** It forces the pressure field via the Poisson equation ($M p^{n+1} = D u^*$), and that pressure field is then used in the Corrector step.

Because it splits in the forward pass, we must **sum** the gradients from both paths in the backward pass! 
We take the gradient from Path A (which we found in Reverse Step 1) and add the gradient from Path B (which we just found via $D^T \lambda_p$):
$$ \frac{\partial \text{Loss}}{\partial u^*}_{total} = \frac{\partial \text{Loss}}{\partial u^*} + D^T \lambda_p $$

This `total` gradient now holds all the error information required to backpropagate into the Momentum solver!

### Step 3. Reverse Predictor (The Second Adjoint Solve)
Finally, we must backpropagate through the initial Momentum solve ($A u^* = \text{RHS}$). 
Applying the chain rule gives $\frac{\partial \text{Loss}}{\partial \text{RHS}} = \left(\frac{\partial \text{Loss}}{\partial u^*_{total}}\right)^T A^{-1}$.
Again, we use the **Discrete Adjoint Method** and invent a second adjoint variable $\lambda_u$:
$$ A^T \lambda_u = \frac{\partial \text{Loss}}{\partial u^*}_{total} $$

*   **No Symmetry Here:** Unlike the Poisson step, the momentum matrix $A$ contains advection terms, which makes it **asymmetric** ($A \neq A^T$). PICT cannot simply re-use the forward matrix. It must explicitly compute the transpose $A^T$ and run a brand new linear solve for $\lambda_u$. However, this is still infinitely faster than unrolling loops!

Once the solver finds $\lambda_u$, we know it is exactly the gradient with respect to the right-hand side of the predictor equation. Since $\Delta F_{SGS}$ was part of that RHS, we have successfully crossed the physics barrier!
$$ \frac{\partial \text{Loss}}{\partial \Delta F_{SGS}} = \lambda_u $$

### Step 4. The Neural Network Update (Autograd & Optimizer)
Once $\lambda_u$ emerges from the physics solver, PyTorch's automatic differentiation engine (Autograd) takes over.

Because $\Delta F_{SGS}$ is the direct output of our CNN, Autograd acts like a tape recorder in reverse. It applies the standard chain rule layer-by-layer backwards through the network's internal Convolutions and Activations:
$$ \frac{\partial \text{Loss}}{\partial W_{cnn}} = \lambda_u \cdot \frac{\partial \Delta F_{SGS}}{\partial W_{cnn}} $$

By the time `loss.backward()` finishes, PyTorch has calculated the exact gradient for every single weight and bias in the CNN and stored them in the `.grad` attributes. 

Finally, the **Adam Optimizer** looks at those `.grad` values and updates the CNN weights downhill:
$$ W_{new} = W_{old} - (\text{learning\_rate} \times \frac{\partial \text{Loss}}{\partial W_{cnn}}) $$
The CNN is now smarter for the next time-step!
