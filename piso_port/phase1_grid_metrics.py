import numpy as np
import sympy as sp
import time

def as_periodic(periodic):
    """Normalise a periodic flag to a 3-tuple of bools."""
    if periodic is None:
        return (False, False, False)
    if isinstance(periodic, bool):
        return (periodic, periodic, periodic)
    return tuple(bool(p) for p in periodic)

def deriv(f, h, axis, periodic=False, order=1):
    """
    Central difference along `axis`.

    Non-periodic: np.gradient with edge_order=2 (one-sided at the ends).
    Periodic: wrap-pad by one cell first, so the stencil is a true central difference
    everywhere INCLUDING across the seam, then strip the padding. Using np.gradient
    directly on a periodic field would silently apply one-sided differences at the seam
    and destroy both the metric accuracy and the GCL there.
    """
    if not periodic:
        g = np.gradient(f, h, axis=axis, edge_order=2)
        return g if order == 1 else np.gradient(g, h, axis=axis, edge_order=2)
    pad = [(0, 0)] * f.ndim
    pad[axis] = (1, 1)
    fp = np.pad(f, pad, mode="wrap")
    g = np.gradient(fp, h, axis=axis, edge_order=2)
    sl = [slice(None)] * f.ndim
    sl[axis] = slice(1, -1)
    return g[tuple(sl)]


class Domain3D:
    def __init__(self, nx, ny, nz, dxi=1.0, deta=1.0, dzeta=1.0):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dxi = dxi
        self.deta = deta
        self.dzeta = dzeta
        
        # Physical coordinates at cell centers
        self.x = np.zeros((nx, ny, nz))
        self.y = np.zeros((nx, ny, nz))
        self.z = np.zeros((nx, ny, nz))
        
        # Metric tensors (9 components)
        self.metrics = {} # e.g. metrics['xi_x']
        self.J = np.zeros((nx, ny, nz)) # Scalar Jacobian (volume)


def wrap_pad_coords(x, y, z, periodic, width=2, period=(1.0, 1.0, 1.0)):
    """
    Ghost-pad the COORDINATE arrays across periodic seams.

    Coordinates are not themselves periodic: x ramps 0 -> 1 along xi and then jumps back.
    Padding them with a plain wrap would inject a spurious derivative of -1/h at the seam
    and collapse the Jacobian (measured: mean J = 0 instead of 1). What is periodic is the
    DEVIATION from the linear ramp, so the wrapped ghost layers must be shifted by one
    period: moving one period along computational axis `a` displaces the physical point by
    period[a] in physical component `a`.

    Two ghost layers, because the conservative metric formula nests two derivatives and we
    want an exact central difference at every real node.
    """
    fields = [x, y, z]
    for axis in range(3):
        if not periodic[axis]:
            continue
        out = []
        for m, f in enumerate(fields):
            jump = period[axis] if m == axis else 0.0
            sl_lo = [slice(None)] * 3; sl_lo[axis] = slice(-width, None)
            sl_hi = [slice(None)] * 3; sl_hi[axis] = slice(0, width)
            out.append(np.concatenate([f[tuple(sl_lo)] - jump, f, f[tuple(sl_hi)] + jump],
                                      axis=axis))
        fields = out
    return fields

def _metrics_core(x, y, z, dxi, deta, dzeta):
    """
    Computes the 9 metric tensors and the Jacobian J using the strict 
    Thomas & Lombard (1979) conservative formulation.
    This mathematically guarantees that the discrete GCL = 0.0
    """
    def D(f, spacing, axis):
        return np.gradient(f, spacing, axis=axis, edge_order=2)
        
    def C(p, q, s1, s2, axis1, axis2):
        term1 = D(p * D(q, s2, axis2) - q * D(p, s2, axis2), s1, axis1)
        term2 = D(q * D(p, s1, axis1) - p * D(q, s1, axis1), s2, axis2)
        return 0.5 * (term1 + term2)

    metrics = {}
    
    # Calculate J * metrics (Area Vectors)
    J_xi_x = C(y, z, deta, dzeta, 1, 2)
    J_xi_y = C(z, x, deta, dzeta, 1, 2)
    J_xi_z = C(x, y, deta, dzeta, 1, 2)
    
    J_eta_x = C(y, z, dzeta, dxi, 2, 0)
    J_eta_y = C(z, x, dzeta, dxi, 2, 0)
    J_eta_z = C(x, y, dzeta, dxi, 2, 0)
    
    J_zeta_x = C(y, z, dxi, deta, 0, 1)
    J_zeta_y = C(z, x, dxi, deta, 0, 1)
    J_zeta_z = C(x, y, dxi, deta, 0, 1)
    
    # Calculate J using the dot product of physical coordinates with Area Vectors
    # J = x * J_xi_x + y * J_xi_y + z * J_xi_z (evaluated using standard forward diff to avoid averaging issues)
    # Actually, standard forward diff is fine for J scalar volume
    dx_dxi, dx_deta, dx_dzeta = D(x, dxi, 0), D(x, deta, 1), D(x, dzeta, 2)
    dy_dxi, dy_deta, dy_dzeta = D(y, dxi, 0), D(y, deta, 1), D(y, dzeta, 2)
    dz_dxi, dz_deta, dz_dzeta = D(z, dxi, 0), D(z, deta, 1), D(z, dzeta, 2)
    
    J = (dx_dxi * (dy_deta * dz_dzeta - dy_dzeta * dz_deta) -
         dx_deta * (dy_dxi * dz_dzeta - dy_dzeta * dz_dxi) +
         dx_dzeta * (dy_dxi * dz_deta - dy_deta * dz_dxi))
         
    # Extract raw metrics by dividing by J
    metrics['xi_x'] = J_xi_x / J
    metrics['xi_y'] = J_xi_y / J
    metrics['xi_z'] = J_xi_z / J
    
    metrics['eta_x'] = J_eta_x / J
    metrics['eta_y'] = J_eta_y / J
    metrics['eta_z'] = J_eta_z / J
    
    metrics['zeta_x'] = J_zeta_x / J
    metrics['zeta_y'] = J_zeta_y / J
    metrics['zeta_z'] = J_zeta_z / J
    
    return J, metrics


def make_grid(n, warp=0.0, periodic=None):
    """
    Build a 3D wavy grid, optionally periodic per axis.

    Node placement differs by axis type and this matters:
      * wall axis     -- linspace(0,1,n), BOTH endpoints included, spacing 1/(n-1);
                         the boundary nodes sit exactly on the walls (Dirichlet).
      * periodic axis -- arange(n)/n, spacing 1/n; the far endpoint is NOT stored, since
                         it is the same point as node 0. Including it would duplicate a
                         node and halve the effective resolution across the seam.

    The warp uses sin(2*pi*.) rather than sin(pi*.) so that the mapping AND all of its
    derivatives are smoothly periodic; sin(pi*.) vanishes at both ends but its derivative
    flips sign across the seam, which would make the metrics discontinuous there.

    `n` may be an int (cubic) or a (nx, ny, nz) tuple. A thin periodic direction is how a
    genuinely 2D problem is run on this 3D solver: make the spanwise axis periodic with a
    handful of cells and the solution is span-invariant, i.e. 2D, with no end walls.
    """
    per = as_periodic(periodic)
    ns = (n, n, n) if isinstance(n, (int, np.integer)) else tuple(n)
    axes, hs = [], []
    for p, ni in zip(per, ns):
        if p:
            axes.append(np.arange(ni) / ni); hs.append(1.0 / ni)
        else:
            axes.append(np.linspace(0, 1, ni)); hs.append(1.0 / (ni - 1))
    XI, ETA, ZETA = np.meshgrid(*axes, indexing="ij")
    if any(per):
        s2 = lambda t: np.sin(2 * np.pi * t)
    else:
        s2 = lambda t: np.sin(np.pi * t)
    x = XI + warp * s2(ETA) * s2(ZETA)
    y = ETA + warp * s2(XI) * s2(ZETA)
    z = ZETA + warp * s2(XI) * s2(ETA)
    return x, y, z, hs[0], hs[1], hs[2]

def compute_numerical_metrics(x, y, z, dxi, deta, dzeta, periodic=None, width=2,
                              period=(1.0, 1.0, 1.0)):
    """
    Metrics and Jacobian. For periodic axes the coordinates are ghost-padded with a
    one-period shift first (see wrap_pad_coords), the validated conservative formula is run
    unchanged on the padded arrays, and the ghosts are stripped. That keeps a single
    implementation of the metric algebra and gives exact central differences at the seam.
    """
    per = as_periodic(periodic)
    if not any(per):
        return _metrics_core(x, y, z, dxi, deta, dzeta)
    # `period` must reach wrap_pad_coords: the ghost shift is the PHYSICAL length of one
    # period, and defaulting it to 1 on a domain of length 2*pi injects a wrong seam
    # derivative and collapses the Jacobian there. It was unreachable before.
    xp, yp, zp = wrap_pad_coords(x, y, z, per, width, period=period)
    J, m = _metrics_core(xp, yp, zp, dxi, deta, dzeta)
    sl = tuple(slice(width, -width) if per[a] else slice(None) for a in range(3))
    return J[sl], {k: v[sl] for k, v in m.items()}

def analytical_wavy_grid_mms(nx, ny, nz, Ax=0.1, Ay=0.1, Az=0.1):
    """
    Generates a 3D wavy grid and uses SymPy to calculate the exact 
    analytical Jacobian and metric tensors for MMS testing.
    """
    # Computational coordinates (0 to 1)
    xi_arr = np.linspace(0, 1, nx)
    eta_arr = np.linspace(0, 1, ny)
    zeta_arr = np.linspace(0, 1, nz)
    dxi = 1.0 / (nx - 1)
    deta = 1.0 / (ny - 1)
    dzeta = 1.0 / (nz - 1)
    
    XI, ETA, ZETA = np.meshgrid(xi_arr, eta_arr, zeta_arr, indexing='ij')
    
    # 1. Define SymPy symbols and analytical mapping
    xi, eta, zeta = sp.symbols('xi eta zeta')
    x_sym = xi + Ax * sp.sin(sp.pi * eta) * sp.sin(sp.pi * zeta)
    y_sym = eta + Ay * sp.sin(sp.pi * xi) * sp.sin(sp.pi * zeta)
    z_sym = zeta + Az * sp.sin(sp.pi * xi) * sp.sin(sp.pi * eta)
    
    # Evaluate physical coordinates
    x_func = sp.lambdify((xi, eta, zeta), x_sym, 'numpy')
    y_func = sp.lambdify((xi, eta, zeta), y_sym, 'numpy')
    z_func = sp.lambdify((xi, eta, zeta), z_sym, 'numpy')
    
    x = x_func(XI, ETA, ZETA)
    y = y_func(XI, ETA, ZETA)
    z = z_func(XI, ETA, ZETA)
    
    # 2. Derive analytical forward Jacobian elements
    J_fwd_sym = sp.Matrix([
        [sp.diff(x_sym, xi), sp.diff(x_sym, eta), sp.diff(x_sym, zeta)],
        [sp.diff(y_sym, xi), sp.diff(y_sym, eta), sp.diff(y_sym, zeta)],
        [sp.diff(z_sym, xi), sp.diff(z_sym, eta), sp.diff(z_sym, zeta)]
    ])
    
    J_scalar_sym = J_fwd_sym.det()
    J_inv_sym = J_fwd_sym.inv()
    
    # 3. Evaluate exact analytical metrics on the grid
    J_func = sp.lambdify((xi, eta, zeta), J_scalar_sym, 'numpy')
    J_exact = J_func(XI, ETA, ZETA)
    
    metrics_exact = {}
    keys = [['xi_x', 'xi_y', 'xi_z'], 
            ['eta_x', 'eta_y', 'eta_z'], 
            ['zeta_x', 'zeta_y', 'zeta_z']]
            
    for i in range(3):
        for j in range(3):
            metric_func = sp.lambdify((xi, eta, zeta), J_inv_sym[i, j], 'numpy')
            metrics_exact[keys[i][j]] = metric_func(XI, ETA, ZETA)
            
    return x, y, z, dxi, deta, dzeta, J_exact, metrics_exact

def verify_gcl(J, metrics, dxi, deta, dzeta, periodic=None):
    """
    Verifies the 3D Geometric Conservation Law (GCL).
    sum_i ( d/dxi_i (J * dxi_i / dx_j) ) = 0 for j=1,2,3
    """
    per = as_periodic(periodic)

    # GCL for x-component
    d_dxi_J_xi_x = deriv(J * metrics['xi_x'], dxi, 0, per[0])
    d_deta_J_eta_x = deriv(J * metrics['eta_x'], deta, 1, per[1])
    d_dzeta_J_zeta_x = deriv(J * metrics['zeta_x'], dzeta, 2, per[2])
    gcl_x = d_dxi_J_xi_x + d_deta_J_eta_x + d_dzeta_J_zeta_x
    
    # GCL for y-component
    d_dxi_J_xi_y = deriv(J * metrics['xi_y'], dxi, 0, per[0])
    d_deta_J_eta_y = deriv(J * metrics['eta_y'], deta, 1, per[1])
    d_dzeta_J_zeta_y = deriv(J * metrics['zeta_y'], dzeta, 2, per[2])
    gcl_y = d_dxi_J_xi_y + d_deta_J_eta_y + d_dzeta_J_zeta_y
    
    # GCL for z-component
    d_dxi_J_xi_z = deriv(J * metrics['xi_z'], dxi, 0, per[0])
    d_deta_J_eta_z = deriv(J * metrics['eta_z'], deta, 1, per[1])
    d_dzeta_J_zeta_z = deriv(J * metrics['zeta_z'], dzeta, 2, per[2])
    gcl_z = d_dxi_J_xi_z + d_deta_J_eta_z + d_dzeta_J_zeta_z
    
    return np.max(np.abs(gcl_x)), np.max(np.abs(gcl_y)), np.max(np.abs(gcl_z))

if __name__ == "__main__":
    print("--- Phase 1: 3D Mesh & Metric Generation (MMS: Wavy Grid) ---")
    
    resolutions = [10, 20, 40]
    errors_J = []
    
    for n in resolutions:
        print(f"\\nTesting Resolution: {n}x{n}x{n}")
        
        # 1. Generate exact analytical fields
        t0 = time.time()
        x, y, z, dxi, deta, dzeta, J_exact, metrics_exact = analytical_wavy_grid_mms(n, n, n)
        t1 = time.time()
        print(f"Analytical generation time: {t1-t0:.4f}s")
        
        # 2. Compute numerical metrics using our Python functions
        J_num, metrics_num = compute_numerical_metrics(x, y, z, dxi, deta, dzeta)
        
        # 3. Compute L2 Norm Error
        # Cut off the boundary cells (1-cell halo) because central difference edge order 
        # is sometimes slightly less accurate right on the boundary.
        err_J = np.linalg.norm((J_num - J_exact)[1:-1, 1:-1, 1:-1]) / ((n-2)**1.5)
        errors_J.append(err_J)
        print(f"Jacobian L2 Error: {err_J:.2e}")
        
        # Check one of the metric tensors
        err_xi_x = np.linalg.norm((metrics_num['xi_x'] - metrics_exact['xi_x'])[1:-1, 1:-1, 1:-1]) / ((n-2)**1.5)
        print(f"Metric xi_x L2 Error: {err_xi_x:.2e}")
        
        # 4. Verify 3D GCL
        gcl_x, gcl_y, gcl_z = verify_gcl(J_num, metrics_num, dxi, deta, dzeta)
        print(f"GCL Max Error (x, y, z): {gcl_x:.2e}, {gcl_y:.2e}, {gcl_z:.2e}")

    # Verify Convergence Rate
    print("\\n--- Convergence Check ---")
    for i in range(len(resolutions)-1):
        rate = np.log2(errors_J[i] / errors_J[i+1])
        print(f"Convergence Rate ({resolutions[i]} -> {resolutions[i+1]}): {rate:.2f} (Expected: ~2.0)")
    
    print("\\nPhase 1 complete!")
