import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import time

from phase1_grid_metrics import (analytical_wavy_grid_mms, compute_numerical_metrics,
                                 as_periodic, deriv, make_grid)
from phase2_operators import compute_gradient

def get_mms_taylor_green(x, y, z, nu):
    """
    Evaluates the 3D Taylor-Green Vortex MMS fields and returns the exact
    analytical values for velocity and the advection-diffusion source term S.
    """
    # Velocity Field
    u = np.sin(x) * np.cos(y) * np.cos(z)
    v = -np.cos(x) * np.sin(y) * np.cos(z)
    w = np.zeros_like(x)

    # Analytical derivatives (hand-derived for Taylor Green)
    dudx, dudy, dudz = np.cos(x)*np.cos(y)*np.cos(z), -np.sin(x)*np.sin(y)*np.cos(z), -np.sin(x)*np.cos(y)*np.sin(z)
    dvdx, dvdy, dvdz = np.sin(x)*np.sin(y)*np.cos(z), -np.cos(x)*np.cos(y)*np.cos(z), np.cos(x)*np.sin(y)*np.sin(z)

    # Advection (u.grad)u
    adv_u = u*dudx + v*dudy + w*dudz
    adv_v = u*dvdx + v*dvdy + w*dvdz
    adv_w = np.zeros_like(x)

    # Laplacian nabla^2 u
    lap_u = -3 * u
    lap_v = -3 * v
    lap_w = np.zeros_like(x)

    # Source term S = Advection - nu * Laplacian
    S_u = adv_u - nu * lap_u
    S_v = adv_v - nu * lap_v
    S_w = adv_w - nu * lap_w

    return u, v, w, S_u, S_v, S_w

def boundary_masks(nx, ny, nz, periodic=None):
    """
    Returns (interior_indices, boundary_indices, mask) into the flattened field.
    Faces on a non-periodic axis are Dirichlet boundaries; a periodic axis has none.
    """
    per = as_periodic(periodic)
    mask = np.zeros((nx, ny, nz), dtype=bool)
    for axis, p in enumerate(per):
        if p:
            continue
        lo = [slice(None)] * 3; lo[axis] = 0
        hi = [slice(None)] * 3; hi[axis] = -1
        mask[tuple(lo)] = True
        mask[tuple(hi)] = True

    interior = np.flatnonzero(~mask.ravel())
    boundary = np.flatnonzero(mask.ravel())
    return interior, boundary, mask

def build_conservative_diffusion_matrix(nx, ny, nz, dxi, deta, dzeta, J, metrics,
                                        coef=None, periodic=None):
    """
    Assembles the VOLUME-INTEGRATED conservative diagonal diffusion operator

        -[ d/dxi( J g11 d/dxi ) + d/deta( J g22 d/deta ) + d/dzeta( J g33 d/dzeta ) ]

    i.e. the orthogonal (7-point) part of J * nabla^2, negated so the matrix has a
    positive diagonal.

    WHY THE VOLUME-INTEGRATED FORM MATTERS. The true curvilinear Laplacian is

        nabla^2 phi = (1/J) [ d/dxi( J g11 phi_xi ) + ... + cross terms ]

    A discretisation that instead uses the *cell-centred* coefficient g11 with no
    Jacobian weighting (i.e. g11 * phi_xixi) silently drops the term arising from
    differentiating (J g11). That dropped term is O(1) in the grid warp, so combining
    it with the J-weighted compute_cross_diffusion() below produces an INCONSISTENT
    operator: the split does not converge to the Laplacian at all, and the solution
    error plateaus under grid refinement instead of falling at O(h^2). Keeping J on
    the operator (and moving it to the RHS of the equation) is what makes the implicit
    and deferred-corrected halves reconstruct the same Laplacian.

    The face coefficients are INTERPOLATED to faces, 0.5*(Jg[i] + Jg[i+1]), so each face
    writes the same value into both rows it touches and the matrix is symmetric by
    construction. (Symmetry is not needed for the momentum solve, which is non-symmetric
    once advection is added, but it is required for the Poisson solve in Phase 4, which
    reuses this function and solves with Conjugate Gradient.)

    Fully vectorized; no per-cell Python loop.
    """
    g = [metrics['xi_x']**2 + metrics['xi_y']**2 + metrics['xi_z']**2,
         metrics['eta_x']**2 + metrics['eta_y']**2 + metrics['eta_z']**2,
         metrics['zeta_x']**2 + metrics['zeta_y']**2 + metrics['zeta_z']**2]
    h = [dxi, deta, dzeta]

    # Optional per-cell coefficient. PISO's pressure equation is div( (1/A_diag) grad p ),
    # matching PICT's `raP = 1/Adiag` in PISO_build_pressure_matrix. Left at 1 this is the
    # plain Laplacian used by the Phase 4 Poisson test.
    cw = 1.0 if coef is None else coef

    N = nx * ny * nz
    idx = np.arange(N).reshape(nx, ny, nz)

    rows, cols, vals = [], [], []
    diag = np.zeros((nx, ny, nz))

    per = as_periodic(periodic)

    for axis in range(3):
        Jg = cw * J * g[axis]

        if per[axis]:
            # every cell owns one face to its +axis neighbour, wrapping at the seam
            Jg_hi = np.roll(Jg, -1, axis=axis)
            cf = 0.5 * (Jg + Jg_hi) / h[axis]**2
            lo = idx.ravel()
            hi = np.roll(idx, -1, axis=axis).ravel()
            c = cf.ravel()
            rows += [lo, hi]
            cols += [hi, lo]
            vals += [-c, -c]
            # each face adds to both of its cells; rolling by +1 lands on the upper cell
            diag += cf + np.roll(cf, 1, axis=axis)
        else:
            sl_lo = [slice(None)] * 3; sl_lo[axis] = slice(0, -1)
            sl_hi = [slice(None)] * 3; sl_hi[axis] = slice(1, None)

            # Face-interpolated coefficient -> one shared value per interior face
            cf = 0.5 * (Jg[tuple(sl_lo)] + Jg[tuple(sl_hi)]) / h[axis]**2

            lo = idx[tuple(sl_lo)].ravel()
            hi = idx[tuple(sl_hi)].ravel()
            c = cf.ravel()

            rows += [lo, hi]
            cols += [hi, lo]
            vals += [-c, -c]

            diag[tuple(sl_lo)] += cf
            diag[tuple(sl_hi)] += cf

    rows.append(idx.ravel())
    cols.append(idx.ravel())
    vals.append(diag.ravel())

    return sparse.coo_matrix((np.concatenate(vals),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(N, N)).tocsr()

def _convection_coefs(a, axis, n_axis, periodic=False, scheme='sou'):
    """
    Convection coefficients for the signed speed array `a` (already scaled by 1/h).

    scheme='sou'      2nd-Order Upwind. Upwind-biased stencil (3, -4, 1)/2h, taking the two
                      cells on the UPSTREAM side. Degrades to 1st-order upwind within two
                      cells of a wall, where the upstream stencil would leave the domain;
                      on a periodic axis it never degrades, because the stencil wraps.
                      Adds numerical dissipation, which is what keeps it stable at high
                      cell Peclet number.

    scheme='central'  2nd-Order Central, (phi_{i+1} - phi_{i-1})/2h. No dissipation and a
                      zero diagonal contribution, so it is more accurate on smooth flows
                      but loses diagonal dominance once the cell Peclet number
                      Pe = |U| h / nu exceeds 2, where it produces oscillations.

    Returns (diagonal_coef, {offset: coef_array}).
    """
    shape = a.shape
    i_along = np.arange(n_axis).reshape([-1 if d == axis else 1 for d in range(3)])
    i_along = np.broadcast_to(i_along, shape)

    pos = a > 0
    if periodic:
        # the upstream stencil always exists -- it wraps -- so no 1st-order fallback
        can_back = np.ones_like(pos)
        can_fwd = np.ones_like(pos)
    else:
        can_back = i_along >= 2          # room for the i-2 backward stencil
        can_fwd = i_along <= n_axis - 3  # room for the i+2 forward stencil

    aP = np.zeros_like(a)
    c = {-1: np.zeros_like(a), -2: np.zeros_like(a),
          1: np.zeros_like(a),  2: np.zeros_like(a)}

    if scheme == "central":
        # d(phi)/dxi ~ (phi_{i+1} - phi_{i-1}) / 2h  -- symmetric, no diagonal term
        c[1] += 0.5 * a
        c[-1] += -0.5 * a
        return aP, c
    if scheme != "sou":
        raise ValueError(f"unknown convection scheme {scheme!r}")

    # Flow in +xi: 2nd-order upwind reaches back to i-1, i-2
    m = pos & can_back
    aP[m] += 1.5 * a[m]; c[-1][m] += -2.0 * a[m]; c[-2][m] += 0.5 * a[m]
    m = pos & ~can_back                                  # 1st-order fallback
    aP[m] += a[m]; c[-1][m] += -a[m]

    # Flow in -xi: 2nd-order upwind reaches forward to i+1, i+2
    m = (~pos) & can_fwd
    aP[m] += -1.5 * a[m]; c[1][m] += 2.0 * a[m]; c[2][m] += -0.5 * a[m]
    m = (~pos) & ~can_fwd                                # 1st-order fallback
    aP[m] += -a[m]; c[1][m] += a[m]

    return aP, c

def build_momentum_matrix_7point(nx, ny, nz, J, metrics, dxi, deta, dzeta,
                                 u_conv, v_conv, w_conv, nu, periodic=None,
                                 convection='sou'):
    """
    Assembles the implicit sparse matrix A for curvilinear advection-diffusion, in the
    VOLUME-INTEGRATED form that is consistent with compute_cross_diffusion():

        A = J * advection(SOU)  +  nu * (conservative diagonal diffusion)

    so that the equation actually being solved is

        A phi = J * S  +  nu * J * cross_diffusion(phi)

    Deferred correction supplies the 20 cross-derivative terms on the RHS.

    Note the advection term is J-weighted too: every term in the equation is multiplied
    through by the cell volume J, which is what keeps the implicit operator and the
    explicit correction on the same footing.

    Fully vectorized; no per-cell Python loop.
    """
    # Contravariant convecting velocities (U, V, W), volume-weighted
    U = J * (metrics['xi_x']*u_conv + metrics['xi_y']*v_conv + metrics['xi_z']*w_conv)
    V = J * (metrics['eta_x']*u_conv + metrics['eta_y']*v_conv + metrics['eta_z']*w_conv)
    W = J * (metrics['zeta_x']*u_conv + metrics['zeta_y']*v_conv + metrics['zeta_z']*w_conv)

    N = nx * ny * nz
    n_ax = [nx, ny, nz]
    idx = np.arange(N).reshape(nx, ny, nz)

    rows, cols, vals = [], [], []
    diag = np.zeros((nx, ny, nz))

    per = as_periodic(periodic)

    for axis, (speed, h) in enumerate([(U, dxi), (V, deta), (W, dzeta)]):
        aP, coefs = _convection_coefs(speed / h, axis, n_ax[axis], per[axis],
                                      scheme=convection)
        diag += aP

        for off, carr in coefs.items():
            if per[axis]:
                # neighbour P+off always exists, wrapping at the seam
                rows.append(idx.ravel())
                cols.append(np.roll(idx, -off, axis=axis).ravel())
                vals.append(carr.ravel())
                continue
            # cells P whose neighbour P+off lies inside the domain
            if off < 0:
                src = slice(-off, None); dst = slice(0, n_ax[axis] + off)
            else:
                src = slice(0, n_ax[axis] - off); dst = slice(off, None)
            sl_src = [slice(None)]*3; sl_src[axis] = src
            sl_dst = [slice(None)]*3; sl_dst[axis] = dst

            rows.append(idx[tuple(sl_src)].ravel())
            cols.append(idx[tuple(sl_dst)].ravel())
            vals.append(carr[tuple(sl_src)].ravel())

    rows.append(idx.ravel())
    cols.append(idx.ravel())
    vals.append(diag.ravel())

    A_adv = sparse.coo_matrix((np.concatenate(vals),
                               (np.concatenate(rows), np.concatenate(cols))),
                              shape=(N, N)).tocsr()

    A_diff = build_conservative_diffusion_matrix(nx, ny, nz, dxi, deta, dzeta, J, metrics,
                                                 periodic=periodic)

    return A_adv + nu * A_diff

def compute_cross_diffusion(phi, J, metrics, dxi, deta, dzeta, periodic=None):
    """
    Computes the cross-derivative terms of the Laplacian (the other 20 points of the 27-point stencil)
    which are omitted from the 7-point implicit matrix. Used for Deferred Correction on skewed grids.
    Returns (1/J) * div(cross fluxes), i.e. the cross part of nabla^2 phi.
    """
    # Contravariant metric tensor off-diagonals
    g12 = metrics['xi_x']*metrics['eta_x'] + metrics['xi_y']*metrics['eta_y'] + metrics['xi_z']*metrics['eta_z']
    g13 = metrics['xi_x']*metrics['zeta_x'] + metrics['xi_y']*metrics['zeta_y'] + metrics['xi_z']*metrics['zeta_z']
    g23 = metrics['eta_x']*metrics['zeta_x'] + metrics['eta_y']*metrics['zeta_y'] + metrics['eta_z']*metrics['zeta_z']

    per = as_periodic(periodic)

    # First derivatives
    dphi_dxi = deriv(phi, dxi, 0, per[0])
    dphi_deta = deriv(phi, deta, 1, per[1])
    dphi_dzeta = deriv(phi, dzeta, 2, per[2])

    # Cross fluxes
    flux_xi = J * (g12 * dphi_deta + g13 * dphi_dzeta)
    flux_eta = J * (g12 * dphi_dxi + g23 * dphi_dzeta)
    flux_zeta = J * (g13 * dphi_dxi + g23 * dphi_deta)

    # Divergence of cross fluxes
    cross_diff = (deriv(flux_xi, dxi, 0, per[0]) +
                  deriv(flux_eta, deta, 1, per[1]) +
                  deriv(flux_zeta, dzeta, 2, per[2])) / J

    return cross_diff

def deferred_correction(phi0, ib, rhs_fn, solve_fn,
                        max_iters=200, tol=1e-11,
                        omega_ladder=(1.0, 0.7, 0.4, 0.2)):
    """
    Fixed-point deferred-correction loop with automatic under-relaxation fallback.

    Deferred correction is a Picard iteration whose contraction factor grows with grid
    skewness AND with the diffusion coefficient. Measured on the wavy grid at n=20,
    nu=1.0, the contraction ratio is ~0.46 at warp 0.10, ~0.72 at 0.15 and ~1.20 at 0.25
    -- i.e. it stops being a contraction and the plain (omega=1) iteration DIVERGES.
    Neither warp nor nu alone predicts this; only their combination does.

    Under-relaxation, phi <- (1-omega) phi + omega phi_solved, restores the contraction
    and converges to the SAME fixed point (verified: identical L2 errors across omega,
    only the iteration count differs). Since omega=1 is roughly twice as fast on mild
    grids, we try it first and step down the ladder only if the iteration actually blows
    up -- so the common case pays nothing for the robustness.

    A run that is merely slow (not diverging) does not trigger a retry, because more
    relaxation would only make it slower.

    rhs_fn(phi)          -> RHS vector for the interior system given the current field
    solve_fn(rhs, x0)    -> solution of the linear system on the interior
    """
    phi = phi0.copy()
    for attempt, omega in enumerate(omega_ladder):
        phi = phi0.copy()
        diverged = False
        deltas = []
        for it in range(max_iters):
            sol = solve_fn(rhs_fn(phi), phi.flat[ib])
            prev = phi.copy()
            phi.flat[ib] = (1.0 - omega) * phi.flat[ib] + omega * sol
            delta = np.abs(phi - prev).max()
            deltas.append(delta)
            if not np.isfinite(delta) or delta > 1e8:
                diverged = True
                break
            if delta < tol:
                return phi, {'omega': omega, 'iters': it + 1, 'converged': True,
                             'attempts': attempt + 1}

        # Ran out of iterations. Distinguish "slow but contracting" from "not contracting":
        # a ratio marginally above 1 diverges too slowly to trip the guard above, and would
        # otherwise be mistaken for slow convergence and returned as a valid answer.
        if not diverged and len(deltas) >= 10:
            window = deltas[-10:]
            ratio = (window[-1] / window[0]) ** (1.0 / (len(window) - 1)) if window[0] > 0 else 0.0
            if ratio < 0.98:
                break   # genuinely contracting, just slow -- more relaxation would not help
            diverged = True

        if not diverged:
            break
        print(f"  deferred correction not contracting at omega={omega}, retrying under-relaxed")
    return phi, {'omega': omega, 'iters': it + 1, 'converged': False,
                 'attempts': attempt + 1}

def solve_momentum(n, warp, nu, mms=get_mms_taylor_green,
                   max_dc_iters=200, dc_tol=1e-11, verbose=True):
    """
    Solves the curvilinear advection-diffusion momentum equation for u and v via
    deferred correction, and returns (u_num, v_num, u_exact, v_exact, diagnostics).

    The deferred correction iterates on the NUMERICAL solution (starting from a zero
    interior), never on the exact one -- so this measures what a real solver would do.
    """
    x, y, z, dxi, deta, dzeta, _, _ = analytical_wavy_grid_mms(n, n, n, Ax=warp, Ay=warp, Az=warp)
    J, metrics = compute_numerical_metrics(x, y, z, dxi, deta, dzeta)

    u_ex, v_ex, w_ex, S_u, S_v, S_w = mms(x, y, z, nu)

    # Frozen-coefficient linearisation: the convecting velocity is the exact field,
    # which is the standard MMS setup for validating the discrete operator itself.
    A = build_momentum_matrix_7point(n, n, n, J, metrics, dxi, deta, dzeta,
                                     u_ex, v_ex, w_ex, nu)

    # Dirichlet BCs by ELIMINATION: solve the interior block and move the known
    # boundary contributions to the RHS. Avoids stamping identity rows into A.
    ib, bb, _ = boundary_masks(n, n, n)
    A_ii = A[ib][:, ib].tocsr()
    A_ib = A[ib][:, bb].tocsr()

    results, diags = [], []
    for phi_ex, S in ((u_ex, S_u), (v_ex, S_v)):
        phi_b = phi_ex.flat[bb]
        phi = np.zeros_like(phi_ex)
        phi.flat[bb] = phi_b

        def rhs_fn(p, S=S, phi_b=phi_b):
            cross = compute_cross_diffusion(p, J, metrics, dxi, deta, dzeta)
            # A phi = J*S + nu*J*cross   (everything volume-integrated)
            return (J * (S + nu * cross)).flat[ib] - A_ib @ phi_b

        def solve_fn(rhs, x0):
            sol, info = splinalg.bicgstab(A_ii, rhs, x0=x0, rtol=1e-12, maxiter=5000)
            if info != 0:
                print(f"  Warning: BiCGStab returned info={info}")
            return sol

        phi, d = deferred_correction(phi, ib, rhs_fn, solve_fn,
                                     max_iters=max_dc_iters, tol=dc_tol)
        if not d['converged']:
            print(f"  Warning: deferred correction did not converge "
                  f"({d['iters']} iters at omega={d['omega']})")
        results.append(phi)
        diags.append(d)

    if verbose:
        print(f"  DC iters: u={diags[0]['iters']} (omega={diags[0]['omega']}), "
              f"v={diags[1]['iters']} (omega={diags[1]['omega']})")

    return results[0], results[1], u_ex, v_ex, {'u': diags[0], 'v': diags[1]}

if __name__ == "__main__":
    print("--- Phase 3: Momentum Matrix Assembly (MMS: Taylor-Green Vortex) ---")

    resolutions = [10, 20, 40]
    nu = 0.01     # Kinematic viscosity
    warp = 0.05   # grid skewness, matching Phase 2 and Phase 4
    errors_u, errors_v = [], []

    for n in resolutions:
        print(f"\nTesting Resolution: {n}x{n}x{n}  (grid warp A={warp}, nu={nu})")

        u_num, v_num, u_ex, v_ex, _ = solve_momentum(n, warp, nu)

        I = (slice(1, -1),) * 3
        f = (n - 2)**1.5
        err_u = np.linalg.norm((u_num - u_ex)[I]) / f
        err_v = np.linalg.norm((v_num - v_ex)[I]) / f
        errors_u.append(err_u); errors_v.append(err_v)
        print(f"  Velocity U L2 Error: {err_u:.2e}")
        print(f"  Velocity V L2 Error: {err_v:.2e}")

    print("\n--- Convergence Check ---")
    for i in range(len(resolutions)-1):
        rate_u = np.log2(errors_u[i] / errors_u[i+1])
        rate_v = np.log2(errors_v[i] / errors_v[i+1])
        print(f"[{resolutions[i]} -> {resolutions[i+1]}] U Convergence Rate: {rate_u:.2f} | V Convergence Rate: {rate_v:.2f} (Expected: ~2.0)")

    print("\nPhase 3 complete!")
