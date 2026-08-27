import numpy as np
import torch
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scipy.spatial

def generate_mesh(nx, ny):
    """Generate a structured triangular mesh for a 2D domain [0, 1] x [0, 1]."""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)
    
    # Flatten coordinates to list of nodes
    nodes = np.column_stack([xv.ravel(), yv.ravel()])
    
    # Triangulate
    tri = scipy.spatial.Delaunay(nodes)
    elements = tri.simplices
    
    return nodes, elements

def assemble_system(nodes, elements, f_val=-1.0):
    """Assemble the global stiffness matrix and load vector using NumPy."""
    n_nodes = len(nodes)
    K = np.zeros((n_nodes, n_nodes))
    F = np.zeros(n_nodes)
    
    for el in elements:
        n1, n2, n3 = el
        coords = nodes[el]
        
        # Calculate element area
        # Area = 0.5 * det([1 x1 y1; 1 x2 y2; 1 x3 y3])
        matrix = np.ones((3, 3))
        matrix[:, 1:] = coords
        area = 0.5 * np.abs(np.linalg.det(matrix))
        
        # Calculate basis function gradients (b_i, c_i)
        # N_i(x,y) = (a_i + b_i*x + c_i*y) / (2*Area)
        b = np.array([
            coords[1, 1] - coords[2, 1], # y2 - y3
            coords[2, 1] - coords[0, 1], # y3 - y1
            coords[0, 1] - coords[1, 1]  # y1 - y2
        ])
        
        c = np.array([
            coords[2, 0] - coords[1, 0], # x3 - x2
            coords[0, 0] - coords[2, 0], # x1 - x3
            coords[1, 0] - coords[0, 0]  # x2 - x1
        ])
        
        # Local stiffness matrix
        # K_ij = (1 / (4*Area)) * (b_i*b_j + c_i*c_j)
        K_local = (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)
        
        # Local load vector (weak form: int(grad u * grad v) = -int(f * v))
        F_local = np.ones(3) * (-f_val) * area / 3.0
        
        # Assemble into global matrices
        for i in range(3):
            F[el[i]] += F_local[i]
            for j in range(3):
                K[el[i], el[j]] += K_local[i, j]
                
    return K, F

def apply_boundary_conditions(K, F, nodes):
    """Apply zero Dirichlet boundary conditions on all domain edges."""
    # Find boundary nodes (x=0, x=1, y=0, y=1)
    boundary_mask = (nodes[:, 0] == 0) | (nodes[:, 0] == 1) | \
                    (nodes[:, 1] == 0) | (nodes[:, 1] == 1)
    
    boundary_indices = np.where(boundary_mask)[0]
    
    # Apply Dirichlet BC (u=0)
    for idx in boundary_indices:
        K[idx, :] = 0
        K[idx, idx] = 1.0
        F[idx] = 0.0
        
    return K, F

def solve_with_pytorch(K, F):
    """Convert NumPy arrays to PyTorch tensors and solve the linear system."""
    # Convert to PyTorch tensors
    K_tensor = torch.tensor(K, dtype=torch.float32)
    F_tensor = torch.tensor(F, dtype=torch.float32).unsqueeze(1)
    
    print("Solving linear system using PyTorch...")
    # Solve Ku = F
    # For small problems, we can use dense solvers. 
    # For larger problems, sparse iterative solvers would be preferred, 
    # but torch.linalg.solve handles dense systems efficiently on GPU/CPU.
    if torch.cuda.is_available():
        print("Using CUDA.")
        K_tensor = K_tensor.cuda()
        F_tensor = F_tensor.cuda()
        
    u_tensor = torch.linalg.solve(K_tensor, F_tensor)
    
    # Bring back to CPU/NumPy
    u = u_tensor.cpu().numpy().flatten()
    return u

def plot_results(nodes, elements, u):
    """Visualize the solved field."""
    plt.figure(figsize=(8, 6))
    
    # Cast to standard numpy types to avoid matplotlib array parsing errors
    elements = np.asarray(elements, dtype=np.int32)
    u = np.asarray(u, dtype=np.float64)
    
    plt.tripcolor(nodes[:, 0], nodes[:, 1], elements, u, shading='gouraud', cmap='viridis')
    plt.colorbar(label='Solution (u)')
    plt.title('2D Poisson Equation Solution using Galerkin FEM')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Plot mesh lines
    plt.triplot(nodes[:, 0], nodes[:, 1], elements, color='white', alpha=0.2, linewidth=0.5)
    
    output_file = 'poisson_fem_solution.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    
def analytical_solution(nodes, f_val=-10.0, terms=50):
    """Compute the analytical solution using a Fourier sine series."""
    x = nodes[:, 0]
    y = nodes[:, 1]
    u_exact = np.zeros_like(x)
    
    # f_val is the constant forcing term. nabla^2 u = f_val.
    for n in range(1, terms, 2):
        for m in range(1, terms, 2):
            coeff = (-16.0 * f_val) / (np.pi**4 * n * m * (n**2 + m**2))
            u_exact += coeff * np.sin(n * np.pi * x) * np.sin(m * np.pi * y)
            
    return u_exact

def main():
    # Setup resolution
    nx, ny = 30, 30
    print(f"Generating {nx}x{ny} mesh...")
    nodes, elements = generate_mesh(nx, ny)
    print(f"Generated {len(nodes)} nodes and {len(elements)} elements.")
    
    f_val = -10.0
    print("Assembling global stiffness matrix in NumPy...")
    K, F = assemble_system(nodes, elements, f_val=f_val) # Poisson eq: nabla^2 u = -10
    
    print("Applying Dirichlet boundary conditions...")
    K, F = apply_boundary_conditions(K, F, nodes)
    
    # Solve with PyTorch
    u = solve_with_pytorch(K, F)
    
    # Compute RMS Error
    u_exact = analytical_solution(nodes, f_val=f_val)
    rms_error = np.sqrt(np.mean((u - u_exact)**2))
    print(f"\\n--- Error Analysis ---")
    print(f"RMS Error compared to analytical solution: {rms_error:.6e}")
    print(f"Max absolute error: {np.max(np.abs(u - u_exact)):.6e}\\n")
    
    print("Plotting results...")
    plot_results(nodes, elements, u)
    
    print("Done!")

if __name__ == "__main__":
    main()
