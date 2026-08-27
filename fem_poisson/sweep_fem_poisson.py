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
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )
    def forward(self, coords):
        grid_coords = coords.view(self.ny, self.nx, 2).permute(2, 0, 1).unsqueeze(0)
        dF = self.net(grid_coords)
        return dF.view(-1)

def run_grid(nx_cr, ny_cr, device, u_hr, nodes_hr):
    print(f"\\n--- Running Grid {nx_cr}x{ny_cr} ---")
    nodes_cr, elems_cr = generate_mesh(nx_cr, ny_cr)
    
    # Target
    u_target_cr = griddata(nodes_hr, u_hr, nodes_cr, method='cubic')
    u_target_cr_tensor = torch.tensor(u_target_cr, dtype=torch.float32, device=device).unsqueeze(1)
    
    # Assemble Base System
    K_cr, F_cr = assemble_system(nodes_cr, elems_cr, f_val=-10.0)
    bc_mask_cr = get_boundary_mask(nodes_cr)
    
    # Uncorrected solve
    K_cr_bc, F_cr_bc = apply_boundary_conditions(K_cr, F_cr, bc_mask_cr)
    u_uncorrected = torch.linalg.solve(
        torch.tensor(K_cr_bc, dtype=torch.float32, device=device),
        torch.tensor(F_cr_bc, dtype=torch.float32, device=device).unsqueeze(1)
    ).cpu().numpy().flatten()
    
    # CNN Training
    model = SGS_CNN(nx_cr, ny_cr).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    K_cr_tensor = torch.tensor(K_cr, dtype=torch.float32, device=device)
    F_cr_tensor = torch.tensor(F_cr, dtype=torch.float32, device=device)
    nodes_cr_tensor = torch.tensor(nodes_cr, dtype=torch.float32, device=device)
    K_cr_bc_tensor = torch.tensor(K_cr_bc, dtype=torch.float32, device=device)
    
    epochs = 400
    for epoch in range(epochs):
        optimizer.zero_grad()
        dF = model(nodes_cr_tensor)
        F_total = F_cr_tensor + dF
        F_total_bc = F_total.clone()
        F_total_bc[bc_mask_cr] = 0.0
        u_pred = torch.linalg.solve(K_cr_bc_tensor, F_total_bc.unsqueeze(1))
        loss = nn.MSELoss()(u_pred, u_target_cr_tensor)
        loss.backward()
        optimizer.step()
        
    u_corrected = u_pred.detach().cpu().numpy().flatten()
    
    # Errors
    rms_uncorrected = np.sqrt(np.mean((u_uncorrected - u_target_cr)**2))
    rms_corrected = np.sqrt(np.mean((u_corrected - u_target_cr)**2))
    dof = (nx_cr - 2) * (ny_cr - 2)
    
    return dof, rms_uncorrected, rms_corrected

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Generating High-Res Target...")
    nodes_hr, elems_hr = generate_mesh(30, 30)
    K_hr, F_hr = assemble_system(nodes_hr, elems_hr, f_val=-10.0)
    bc_mask_hr = get_boundary_mask(nodes_hr)
    K_hr_bc, F_hr_bc = apply_boundary_conditions(K_hr, F_hr, bc_mask_hr)
    u_hr = torch.linalg.solve(
        torch.tensor(K_hr_bc, dtype=torch.float32, device=device),
        torch.tensor(F_hr_bc, dtype=torch.float32, device=device).unsqueeze(1)
    ).cpu().numpy().flatten()
    
    resolutions = [(3,3), (5,3), (3,5), (5,5), (7,3), (3,7), (7,7), (10,10)]
    results = []
    
    for nx, ny in resolutions:
        dof, rms_u, rms_c = run_grid(nx, ny, device, u_hr, nodes_hr)
        results.append((nx, ny, dof, rms_u, rms_c))
        
    # Write Markdown Table
    with open('sweep_results.md', 'w') as f:
        f.write("| Grid Resolution | DoF | Uncorrected RMS | Corrected RMS |\\n")
        f.write("| :--- | :--- | :--- | :--- |\\n")
        for nx, ny, dof, rms_u, rms_c in results:
            f.write(f"| {nx}x{ny} | {dof} | {rms_u:.6e} | {rms_c:.6e} |\\n")
            
    # Plotting
    # Sort results by DoF for plotting line charts cleanly
    results_sorted = sorted(results, key=lambda x: x[2])
    
    dofs = [r[2] for r in results_sorted]
    rms_u_list = [r[3] for r in results_sorted]
    rms_c_list = [r[4] for r in results_sorted]
    
    plt.figure(figsize=(8, 6))
    plt.plot(dofs, rms_u_list, 'o--', color='red', label='Uncorrected Coarse')
    plt.plot(dofs, rms_c_list, 'o-', color='blue', label='CNN Corrected Coarse')
    
    # 30x30 Target Line
    plt.axhline(y=5.88e-04, color='green', linestyle=':', label='30x30 Natural Error')
    
    plt.yscale('log')
    plt.xlabel('Degrees of Freedom (DoF)')
    plt.ylabel('RMS Error (Log Scale)')
    plt.title('Error vs Degrees of Freedom')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig('error_vs_dof.png')
    
if __name__ == "__main__":
    main()
