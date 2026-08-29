# Implementation Plan: 2D Poisson Equation Solver

The goal is to implement an example solving a 2D Poisson equation ($\nabla^2 u = f$). 

## ⚠️ Important Architectural Clarifications
There are a few fundamental conflicts in combining **PICT**, **Galerkin Finite Element Method (FEM)**, and **NumPy** into a single script:

1. **PICT is a Finite Volume (FVM) Solver:** PICT is specifically designed around the Finite Volume Method on structured grids (used for the PISO algorithm in fluid dynamics). It does not have built-in capabilities or matrix assembly routines for the Galerkin Finite Element Method (which uses unstructured meshes and basis functions).
2. **PICT Requires PyTorch CUDA Tensors:** PICT's core physics operators are written in custom CUDA code that strictly expects `torch.cuda` tensors. Using NumPy "as much as possible" will break PICT, as it cannot interface with NumPy arrays directly without converting back and forth from the GPU.

Because of this, we have two distinct options for how to proceed with the implementation. 

---

## Option A: Native PICT Solver (Finite Volume)
We use the PICT framework exactly as it was designed to solve the Poisson equation (which is equivalent to solving the pressure equation in fluid dynamics).
*   **Method:** Finite Volume Method (FVM) on a structured 2D grid.
*   **Backend:** Pure PyTorch & PICT CUDA extensions (No NumPy).
*   **Approach:** We will define a 2D `PISOtorch.Domain`, set up a source term (the right-hand side of the Poisson equation), and use PICT's implicit linear solver to find the field that satisfies the Poisson equation.

## Option B: Standalone PyTorch/NumPy Solver (Galerkin FEM)
We bypass the PICT framework entirely and write a custom script from scratch.
*   **Method:** Galerkin Linear Finite Element Method (FEM) on a 2D mesh (using triangular or quadrilateral elements).
*   **Backend:** PyTorch (for the final linear solve and differentiability) and NumPy (for mesh generation and local stiffness matrix assembly).
*   **Approach:** We will manually write the code to generate a simple mesh, calculate the local element stiffness matrices using numerical integration, assemble the global sparse matrix in NumPy/SciPy, and solve it using PyTorch.

---

> [!IMPORTANT]
> **User Review Required**
> Please let me know which path you would prefer to take! 
> 
> *   Reply **"Option A"** if you want to see how PICT solves this using its highly optimized Finite Volume architecture.
> *   Reply **"Option B"** if you want a custom Galerkin FEM script built from scratch using NumPy and PyTorch (ignoring the PICT codebase).
