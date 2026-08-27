import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scipy.spatial
from scipy.interpolate import griddata
import os

# --- FEM Utilities ---
def generate_mesh(nx, ny):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)
    nodes = np.column_stack([xv.ravel(), yv.ravel()])
    tri = scipy.spatial.Delaunay(nodes)
    return nodes, tri.simplices

def assemble_system(nodes, elements, f_val=-10.0):
    n_nodes = len(nodes)
    K = np.zeros((n_nodes, n_nodes))
    F = np.zeros(n_nodes)
    
    for el in elements:
        coords = nodes[el]
        matrix = np.ones((3, 3))
        matrix[:, 1:] = coords
        area = 0.5 * np.abs(np.linalg.det(matrix))
        
        b = np.array([coords[1, 1] - coords[2, 1], coords[2, 1] - coords[0, 1], coords[0, 1] - coords[1, 1]])
        c = np.array([coords[2, 0] - coords[1, 0], coords[0, 0] - coords[2, 0], coords[1, 0] - coords[0, 0]])
        
        K_local = (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)
        F_local = np.ones(3) * (-f_val) * area / 3.0
        
        for i in range(3):
            F[el[i]] += F_local[i]
            for j in range(3):
                K[el[i], el[j]] += K_local[i, j]
    return K, F

def get_boundary_mask(nodes):
    return (nodes[:, 0] == 0) | (nodes[:, 0] == 1) | (nodes[:, 1] == 0) | (nodes[:, 1] == 1)

def apply_boundary_conditions(K, F, boundary_mask):
    K_bc = K.copy()
    F_bc = F.clone() if torch.is_tensor(F) else F.copy()
    
    boundary_indices = np.where(boundary_mask)[0]
    for idx in boundary_indices:
        K_bc[idx, :] = 0
        K_bc[idx, idx] = 1.0
        F_bc[idx] = 0.0
    return K_bc, F_bc

# --- CNN Model ---
class SGS_CNN(nn.Module):
    def __init__(self, nx, ny):
        super().__init__()
        self.nx = nx
        self.ny = ny
        # Simple 3-layer CNN
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1), # Input: [x, y] coordinates
            nn.Tanh(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)  # Output: forcing correction
        )
        
    def forward(self, coords):
        # coords shape: (N, 2). Reshape to (1, 2, ny, nx)
        grid_coords = coords.view(self.ny, self.nx, 2).permute(2, 0, 1).unsqueeze(0)
        
        dF = self.net(grid_coords) # shape: (1, 1, ny, nx)
        return dF.view(-1) # flatten to (N,)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. GENERATE HIGH-RES TARGET
    nx_hr, ny_hr = 30, 30
    nodes_hr, elems_hr = generate_mesh(nx_hr, ny_hr)
    K_hr, F_hr = assemble_system(nodes_hr, elems_hr, f_val=-10.0)
    bc_mask_hr = get_boundary_mask(nodes_hr)
    K_hr_bc, F_hr_bc = apply_boundary_conditions(K_hr, F_hr, bc_mask_hr)
    
    u_hr_tensor = torch.linalg.solve(
        torch.tensor(K_hr_bc, dtype=torch.float32, device=device),
        torch.tensor(F_hr_bc, dtype=torch.float32, device=device).unsqueeze(1)
    )
    u_hr = u_hr_tensor.cpu().numpy().flatten()
    
    # 2. GENERATE COARSE GRID & INTERPOLATE TARGET
    nx_cr, ny_cr = 3, 3
    nodes_cr, elems_cr = generate_mesh(nx_cr, ny_cr)
    
    # Interpolate high-res solution to coarse grid to get target
    u_target_cr = griddata(nodes_hr, u_hr, nodes_cr, method='cubic')
    u_target_cr_tensor = torch.tensor(u_target_cr, dtype=torch.float32, device=device).unsqueeze(1)
    
    # Assemble coarse system (Baseline physics)
    K_cr, F_cr = assemble_system(nodes_cr, elems_cr, f_val=-10.0)
    bc_mask_cr = get_boundary_mask(nodes_cr)
    
    # Solve uncorrected coarse system for baseline comparison
    K_cr_bc, F_cr_bc = apply_boundary_conditions(K_cr, F_cr, bc_mask_cr)
    u_uncorrected = torch.linalg.solve(
        torch.tensor(K_cr_bc, dtype=torch.float32, device=device),
        torch.tensor(F_cr_bc, dtype=torch.float32, device=device).unsqueeze(1)
    ).cpu().numpy().flatten()
    
    # 3. SET UP CNN & DIFFERENTIABLE PHYSICS SOLVER
    model = SGS_CNN(nx_cr, ny_cr).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    
    K_cr_tensor = torch.tensor(K_cr, dtype=torch.float32, device=device)
    F_cr_tensor = torch.tensor(F_cr, dtype=torch.float32, device=device)
    nodes_cr_tensor = torch.tensor(nodes_cr, dtype=torch.float32, device=device)
    
    print("\\n--- Starting Training Loop ---")
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Predict forcing correction Delta F
        dF = model(nodes_cr_tensor)
        
        # Add correction to base forcing
        F_total = F_cr_tensor + dF
        
        # Apply Dirichlet BCs
        # We must zero out the equations at the boundaries
        F_total_bc = F_total.clone()
        F_total_bc[bc_mask_cr] = 0.0
        
        # K_cr_bc is already constructed correctly in numpy, we just use it
        K_cr_bc_tensor = torch.tensor(K_cr_bc, dtype=torch.float32, device=device)
        
        # DIFFERENTIABLE IMPLICIT SOLVE! 
        # PyTorch uses the Discrete Adjoint Method automatically here during .backward()
        u_pred = torch.linalg.solve(K_cr_bc_tensor, F_total_bc.unsqueeze(1))
        
        # Compute Loss
        loss = nn.MSELoss()(u_pred, u_target_cr_tensor)
        
        # Backpropagate through the implicit solver
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} | MSE Loss: {loss.item():.6e}")
            
    print("--- Training Complete ---\\n")
    
    # 4. ERROR ANALYSIS & VISUALIZATION
    u_corrected = u_pred.detach().cpu().numpy().flatten()
    u_target = u_target_cr_tensor.cpu().numpy().flatten()
    
    rms_uncorrected = np.sqrt(np.mean((u_uncorrected - u_target)**2))
    rms_corrected = np.sqrt(np.mean((u_corrected - u_target)**2))
    
    print("\\n--- Error Analysis (Compared to downsampled High-Res Target) ---")
    print(f"Uncorrected Coarse RMS Error : {rms_uncorrected:.6e}")
    print(f"CNN-Corrected Coarse RMS Error: {rms_corrected:.6e}")
    print(f"Improvement Factor           : {rms_uncorrected / rms_corrected:.2f}x\\n")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Uncorrected Coarse
    ax = axes[0]
    tc = ax.tripcolor(nodes_cr[:, 0], nodes_cr[:, 1], elems_cr, u_uncorrected, shading='gouraud', cmap='viridis', vmin=0, vmax=0.8)
    ax.triplot(nodes_cr[:, 0], nodes_cr[:, 1], elems_cr, color='white', alpha=0.3, lw=0.5)
    ax.set_title("Uncorrected Coarse FEM")
    plt.colorbar(tc, ax=ax)
    
    # CNN Corrected Coarse
    ax = axes[1]
    tc = ax.tripcolor(nodes_cr[:, 0], nodes_cr[:, 1], elems_cr, u_corrected, shading='gouraud', cmap='viridis', vmin=0, vmax=0.8)
    ax.triplot(nodes_cr[:, 0], nodes_cr[:, 1], elems_cr, color='white', alpha=0.3, lw=0.5)
    ax.set_title("CNN Corrected Coarse FEM")
    plt.colorbar(tc, ax=ax)
    
    # High-Res Target
    ax = axes[2]
    tc = ax.tripcolor(nodes_hr[:, 0], nodes_hr[:, 1], elems_hr, u_hr, shading='gouraud', cmap='viridis', vmin=0, vmax=0.8)
    ax.triplot(nodes_hr[:, 0], nodes_hr[:, 1], elems_hr, color='white', alpha=0.1, lw=0.5)
    ax.set_title("High-Res Target (30x30)")
    plt.colorbar(tc, ax=ax)
    
    plt.tight_layout()
    output_file = 'cnn_fem_comparison.png'
    plt.savefig(output_file)
    print(f"Comparison plot saved to {output_file}")

if __name__ == "__main__":
    main()
