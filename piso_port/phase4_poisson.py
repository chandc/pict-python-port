import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import time

from phase1_grid_metrics import analytical_wavy_grid_mms, compute_numerical_metrics
from phase3_momentum import (compute_cross_diffusion, boundary_masks,
                             build_conservative_diffusion_matrix,
                             deferred_correction)

def get_mms_poisson(x, y, z):
    """
    Evaluates the Trigo-Exponential MMS field for the Pressure Poisson equation.
    Returns the exact pressure p and the exact analytical Laplacian source term f.
    """
    # Exact Pressure Field
    p = np.sin(2*np.pi*x) * np.cos(2*np.pi*y) * np.exp(z)

    # Exact Analytical Laplacian f = nabla^2 p
    # d^2p/dx^2 = -4*pi^2 * p
    # d^2p/dy^2 = -4*pi^2 * p
    # d^2p/dz^2 = p
    f = (1.0 - 8.0 * np.pi**2) * p

    return p, f

def build_poisson_matrix_7point(nx, ny, nz, dxi, deta, dzeta, J, metrics):
    """
    Assembles the 7-point sparse matrix M for the Poisson Equation.

    This is exactly the volume-integrated conservative diagonal diffusion operator
    shared with the momentum solve, so both phases discretise the Laplacian the same
    way. See build_conservative_diffusion_matrix() in phase3_momentum.py for the
    derivation and for why the Jacobian weighting is load-bearing.

    M discretises -[ d/dxi( J g11 d/dxi ) + ... ], i.e. -(J * nabla^2) restricted to its
    orthogonal part, so it is symmetric positive definite and callers must solve
        M p = J * (cross_diffusion - laplacian_source).
    """
    return build_conservative_diffusion_matrix(nx, ny, nz, dxi, deta, dzeta, J, metrics)

def solve_poisson(n, warp, max_dc_iters=200, dc_tol=1e-10, verbose=True):
    """
    Solves the curvilinear Poisson equation on a wavy grid via deferred correction,
    and returns (p_num, p_exact, diagnostics).
    """
    # 1. Setup Grid & Metrics (Skewed Wavy Grid)
    x, y, z, dxi, deta, dzeta, _, _ = analytical_wavy_grid_mms(n, n, n, Ax=warp, Ay=warp, Az=warp)
    J, metrics = compute_numerical_metrics(x, y, z, dxi, deta, dzeta)

    # 2. Get Analytical MMS Fields
    p_ex, f_ex = get_mms_poisson(x, y, z)

    # 3. Assemble the Symmetric 7-point Sparse Matrix M
    M = build_poisson_matrix_7point(n, n, n, dxi, deta, dzeta, J, metrics)

    # Symmetry is what makes the CG solve below legal. Assert it rather than trust it:
    # this is precisely the invariant that used to break silently.
    asym = abs(M - M.T).max()
    assert asym == 0.0, f"Poisson matrix is not symmetric (max |M - M^T| = {asym:.3e})"

    # 4. Apply Dirichlet BCs by ELIMINATION, not by stamping identity rows.
    # Stamping row P with the identity while leaving column P populated would destroy
    # the symmetry we just built. Instead we solve only for the interior unknowns and
    # move the known boundary contributions across to the RHS.
    ib, bb, _ = boundary_masks(n, n, n)
    M_ii = M[ib][:, ib].tocsr()   # interior-interior block (still symmetric, now SPD)
    M_ib = M[ib][:, bb].tocsr()   # interior-boundary coupling
    p_b = p_ex.ravel()[bb]        # exact Dirichlet values

    p_num = np.zeros_like(p_ex)
    # Use .flat, not .ravel(), for writes: ravel() only returns a *view* for contiguous
    # arrays, so on a non-contiguous field the assignment would be silently discarded.
    p_num.flat[bb] = p_b          # boundary values are known exactly

    # 5. Iterative Deferred Correction Loop
    # Because we dropped the 20 cross-diffusion terms from the implicit matrix M,
    # we must iterate to explicitly resolve them on the RHS to achieve full 27-point
    # accuracy on the skewed grid. Like the momentum solve, this Picard iteration stops
    # contracting on strongly warped grids, so it shares the same under-relaxation ladder.
    t0 = time.time()
    cg_iters = [0]

    def rhs_fn(p):
        # Non-orthogonal cross-diffusion from the current guess (already divided by J)
        cross_diff = compute_cross_diffusion(p, J, metrics, dxi, deta, dzeta)
        # The discrete system is  M p = -J * (laplacian source), and the orthogonal part
        # of the source is (f_ex - cross_diff), so the RHS is J*(cross_diff - f_ex).
        # Getting this sign backwards converges to -p_exact.
        return (J * (cross_diff - f_ex)).flat[ib] - M_ib @ p_b

    def solve_fn(rhs, x0):
        it = [0]
        def count(_xk):
            it[0] += 1
        sol, info = splinalg.cg(M_ii, rhs, x0=x0, rtol=1e-10, maxiter=5000, callback=count)
        cg_iters[0] += it[0]
        if info != 0:
            print(f"  Warning: CG solve failed with info={info}")
        return sol

    p_num, d = deferred_correction(p_num, ib, rhs_fn, solve_fn,
                                   max_iters=max_dc_iters, tol=dc_tol)
    if not d['converged']:
        print(f"  Warning: deferred correction did not converge "
              f"({d['iters']} iters at omega={d['omega']})")

    diag = {'dc_iters': d['iters'], 'omega': d['omega'], 'converged': d['converged'],
            'cg_iters': cg_iters[0], 'time': time.time() - t0}
    if verbose:
        print(f"  DC iters: {diag['dc_iters']:3d} (omega={diag['omega']}) | "
              f"CG iters: {cg_iters[0]:5d} | solve: {diag['time']:.2f}s")

    return p_num, p_ex, diag

if __name__ == "__main__":
    print("--- Phase 4: Poisson Matrix Assembly (MMS: Trigo-Exponential) ---")

    resolutions = [10, 20, 40]
    warp = 0.05  # matches the grid skewness used in Phase 2
    errors_p = []

    for n in resolutions:
        print(f"\nTesting Resolution: {n}x{n}x{n}  (grid warp A={warp})")

        p_num, p_ex, _ = solve_poisson(n, warp)

        # Verify against exact analytical solution
        err_p = np.linalg.norm((p_num - p_ex)[1:-1, 1:-1, 1:-1]) / ((n-2)**1.5)
        errors_p.append(err_p)
        print(f"  Pressure P L2 Error: {err_p:.2e}")

    # Verify Convergence Rate
    print("\n--- Convergence Check ---")
    for i in range(len(resolutions)-1):
        rate_p = np.log2(errors_p[i] / errors_p[i+1])
        print(f"[{resolutions[i]} -> {resolutions[i+1]}] Poisson Solution Convergence Rate: {rate_p:.2f} (Expected: ~2.0)")

    print("\nPhase 4 complete!")
