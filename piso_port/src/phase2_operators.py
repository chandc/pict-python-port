import numpy as np
import sympy as sp
import time
from src.phase1_grid_metrics import (analytical_wavy_grid_mms, compute_numerical_metrics,
                                 as_periodic, deriv)

def compute_gradient(p, metrics, dxi, deta, dzeta, periodic=None):
    """
    Computes the gradient (dp/dx, dp/dy, dp/dz) of a scalar field p 
    in curvilinear coordinates.
    """
    per = as_periodic(periodic)
    dp_dxi = deriv(p, dxi, 0, per[0])
    dp_deta = deriv(p, deta, 1, per[1])
    dp_dzeta = deriv(p, dzeta, 2, per[2])
    
    dp_dx = metrics['xi_x'] * dp_dxi + metrics['eta_x'] * dp_deta + metrics['zeta_x'] * dp_dzeta
    dp_dy = metrics['xi_y'] * dp_dxi + metrics['eta_y'] * dp_deta + metrics['zeta_y'] * dp_dzeta
    dp_dz = metrics['xi_z'] * dp_dxi + metrics['eta_z'] * dp_deta + metrics['zeta_z'] * dp_dzeta
    
    return dp_dx, dp_dy, dp_dz

def compute_divergence(u, v, w, J, metrics, dxi, deta, dzeta, periodic=None):
    """
    Computes the divergence of a vector field (u, v, w) in curvilinear coordinates.
    Uses the conservative formulation (area vectors).
    """
    # Calculate contravariant fluxes (J * U, J * V, J * W)
    JU = J * (metrics['xi_x']*u + metrics['xi_y']*v + metrics['xi_z']*w)
    JV = J * (metrics['eta_x']*u + metrics['eta_y']*v + metrics['eta_z']*w)
    JW = J * (metrics['zeta_x']*u + metrics['zeta_y']*v + metrics['zeta_z']*w)
    
    per = as_periodic(periodic)
    dJU_dxi = deriv(JU, dxi, 0, per[0])
    dJV_deta = deriv(JV, deta, 1, per[1])
    dJW_dzeta = deriv(JW, dzeta, 2, per[2])
    
    div = (dJU_dxi + dJV_deta + dJW_dzeta) / J
    return div

def get_mms_trigo_exponential(x, y, z):
    """
    Evaluates the Trigo-Exponential MMS fields and returns the exact
    analytical values for pressure, velocity, gradient, and divergence.
    """
    # Pressure Field: p = sin(2*pi*x) * cos(2*pi*y) * exp(z)
    p = np.sin(2*np.pi*x) * np.cos(2*np.pi*y) * np.exp(z)
    
    # Velocity Field
    u = np.cos(np.pi*x) * np.sin(np.pi*y) * np.sin(np.pi*z)
    v = np.sin(np.pi*x) * np.cos(np.pi*y) * np.sin(np.pi*z)
    w = np.sin(np.pi*x) * np.sin(np.pi*y) * np.cos(np.pi*z)
    
    # Exact Analytical Gradient of p
    dp_dx_exact = 2*np.pi * np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.exp(z)
    dp_dy_exact = -2*np.pi * np.sin(2*np.pi*x) * np.sin(2*np.pi*y) * np.exp(z)
    dp_dz_exact = p # since d/dz exp(z) = exp(z)
    
    # Exact Analytical Divergence of V
    div_exact = -3*np.pi * np.sin(np.pi*x) * np.sin(np.pi*y) * np.sin(np.pi*z)
    
    return p, u, v, w, dp_dx_exact, dp_dy_exact, dp_dz_exact, div_exact

if __name__ == "__main__":
    print("--- Phase 2: 3D Differential Operators (MMS: Trigo-Exponential) ---")
    
    resolutions = [10, 20, 40]
    errors_grad = []
    errors_div = []
    
    for n in resolutions:
        print(f"\\nTesting Resolution: {n}x{n}x{n}")
        
        # 1. Setup Grid & Metrics
        x, y, z, dxi, deta, dzeta, _, _ = analytical_wavy_grid_mms(n, n, n, Ax=0.05, Ay=0.05, Az=0.05)
        J, metrics = compute_numerical_metrics(x, y, z, dxi, deta, dzeta)
        
        # 2. Get Analytical MMS Fields
        p, u, v, w, dpdx_ex, dpdy_ex, dpdz_ex, div_ex = get_mms_trigo_exponential(x, y, z)
        
        # 3. Compute Numerical Operators
        dpdx_num, dpdy_num, dpdz_num = compute_gradient(p, metrics, dxi, deta, dzeta)
        div_num = compute_divergence(u, v, w, J, metrics, dxi, deta, dzeta)
        
        # 4. Compute L2 Norm Error (exclude boundary halo to avoid central difference edge effects)
        vol_factor = (n-2)**1.5
        
        err_grad_x = np.linalg.norm((dpdx_num - dpdx_ex)[1:-1, 1:-1, 1:-1]) / vol_factor
        err_grad_y = np.linalg.norm((dpdy_num - dpdy_ex)[1:-1, 1:-1, 1:-1]) / vol_factor
        err_grad_z = np.linalg.norm((dpdz_num - dpdz_ex)[1:-1, 1:-1, 1:-1]) / vol_factor
        err_grad = np.sqrt(err_grad_x**2 + err_grad_y**2 + err_grad_z**2)
        errors_grad.append(err_grad)
        
        err_div = np.linalg.norm((div_num - div_ex)[1:-1, 1:-1, 1:-1]) / vol_factor
        errors_div.append(err_div)
        
        print(f"Gradient L2 Error: {err_grad:.2e}")
        print(f"Divergence L2 Error: {err_div:.2e}")

    # Verify Convergence Rate
    print("\\n--- Convergence Check ---")
    for i in range(len(resolutions)-1):
        rate_grad = np.log2(errors_grad[i] / errors_grad[i+1])
        rate_div = np.log2(errors_div[i] / errors_div[i+1])
        print(f"[{resolutions[i]} -> {resolutions[i+1]}] Gradient Convergence Rate: {rate_grad:.2f} (Expected: ~2.0)")
        print(f"[{resolutions[i]} -> {resolutions[i+1]}] Divergence Convergence Rate: {rate_div:.2f} (Expected: ~2.0)")
    
    print("\\nPhase 2 complete!")
