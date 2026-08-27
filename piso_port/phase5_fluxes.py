"""
Phase 5a: Face fluxes and the flux-consistent projection.

This module is what makes the PISO projection exact. Phases 1-4 validated the *differential*
operators (compute_gradient / compute_divergence) against analytic answers, and those remain
correct as differential operators. But they are WIDE-stencil (central differences), while the
pressure matrix is COMPACT (face-based). Composing the two leaves an O(1) inconsistency: the
correction cannot cancel the divergence it was computed from. Measured on a warped grid, the
projection stalls at a 188x reduction, five orders short of the target.

The fix -- and what PICT does -- is to express BOTH the divergence and the pressure operator
on the same faces:

  * face flux is interpolated from cell-centred contravariant components,
        F_f = 0.5*(JU_C + JU_N)
    exactly mirroring computeFluxesNDLoop() in PISO_multiblock_cuda_kernel.cu:1553
    (`fluxes[bound] = (velN + velC) * 0.5f`) -- note PICT uses no Rhie-Chow interpolation;
  * divergence is the signed sum of those face fluxes, mirroring
    k_computePressureRHSdivergenceFromFlux (:5375);
  * the pressure flux through a face uses the SAME face-interpolated coefficient that
    build_conservative_diffusion_matrix() puts into the matrix.

Because the discrete divergence of the pressure flux is then exactly the matrix action,
subtracting it drives the flux divergence to machine precision rather than to a floor.

Fluxes are stored per axis with one extra entry along that axis, so element [i] on axis `ax`
is the face below cell i and element [i+1] the face above it. Index 0 and -1 are the domain
boundary faces, which carry PRESCRIBED fluxes (zero for a solid wall -- including a
tangentially moving lid, since a tangential velocity has no component along the face normal
J*grad(xi_ax)) and are never corrected. That is exactly consistent with the pressure matrix,
which builds no face at the domain boundary and is therefore the zero-flux (Neumann) operator.
"""
import numpy as np
from phase1_grid_metrics import as_periodic, deriv

_KEYS = [('xi_x', 'xi_y', 'xi_z'),
         ('eta_x', 'eta_y', 'eta_z'),
         ('zeta_x', 'zeta_y', 'zeta_z')]

def _lo_hi(axis):
    lo = [slice(None)] * 3; lo[axis] = slice(0, -1)
    hi = [slice(None)] * 3; hi[axis] = slice(1, None)
    return tuple(lo), tuple(hi)

def contravariant_components(u, v, w, J, metrics):
    """Volume-weighted contravariant velocity components (J*U, J*V, J*W) at cell centres."""
    return [J * (metrics[k[0]]*u + metrics[k[1]]*v + metrics[k[2]]*w) for k in _KEYS]

def compute_face_fluxes(u, v, w, J, metrics, boundary='from_velocity', periodic=None):
    """
    Face fluxes by linear interpolation of the cell-centred contravariant components.
    Mirrors PICT's computeFluxesNDLoop.

    `boundary` controls the domain boundary faces, which are PRESCRIBED, never interpolated:

      'from_velocity' (default) -- flux is the contravariant component of the boundary cell,
            matching PICT, where a Dirichlet boundary enforces
            `fluxes[bound] = <contravariant component of the boundary velocity>`.
            For an impermeable wall this is already zero, since a velocity tangent to the wall
            has no component along the face normal J*grad(xi_ax).
      'impermeable' -- hard zero. Use for a closed domain (e.g. a lid-driven cavity), where it
            also guarantees the net boundary flux vanishes and so keeps the singular Neumann
            pressure system exactly compatible.
      a list of 3 (lower, upper) arrays -- explicit prescribed fluxes, e.g. an inlet.

    Defaulting this to zero is a trap: on a flow with through-boundary velocity it injects a
    large spurious divergence at the boundary cells, which the projection then "corrects" by
    wrecking the interior solution. Measured on an exactly divergence-free Taylor-Green field,
    zeroing the boundary faces gave a discrete flux divergence of 1.8e+01.
    """
    per = as_periodic(periodic)
    JU = contravariant_components(u, v, w, J, metrics)
    F = []
    for axis in range(3):
        shape = list(J.shape); shape[axis] += 1
        f = np.zeros(shape)
        lo, hi = _lo_hi(axis)
        interior = [slice(None)] * 3; interior[axis] = slice(1, -1)
        f[tuple(interior)] = 0.5 * (JU[axis][lo] + JU[axis][hi])

        sl_lo = [slice(None)] * 3; sl_lo[axis] = 0
        sl_hi = [slice(None)] * 3; sl_hi[axis] = -1
        if per[axis]:
            # The seam is an ordinary interior face. Storing it at BOTH end slots keeps the
            # array shape (n+1) identical to the wall case, so divergence_from_fluxes needs
            # no special case: the same (F[hi] - F[lo]) difference is then correct for the
            # first and last cell too.
            wrap = 0.5 * (JU[axis][tuple(sl_hi)] + JU[axis][tuple(sl_lo)])
            f[tuple(sl_lo)] = wrap
            f[tuple(sl_hi)] = wrap
        elif boundary == 'impermeable':
            pass                                     # already zero
        elif boundary == 'periodic':
            pass                                     # every axis periodic; handled above
        elif boundary == 'from_velocity':
            f[tuple(sl_lo)] = JU[axis][tuple(sl_lo)]
            f[tuple(sl_hi)] = JU[axis][tuple(sl_hi)]
        else:
            b_lo, b_hi = boundary[axis]
            f[tuple(sl_lo)] = b_lo
            f[tuple(sl_hi)] = b_hi
        F.append(f)
    return F

def divergence_from_fluxes(F, J, h):
    """Signed sum of face fluxes over the cell. Mirrors k_computePressureRHSdivergenceFromFlux."""
    d = np.zeros_like(J)
    for axis in range(3):
        lo, hi = _lo_hi(axis)
        d += (F[axis][hi] - F[axis][lo]) / h[axis]
    return d / J

def pressure_face_fluxes(p, J, metrics, h, coef=None, include_orth=True,
                         include_cross=True, periodic=None):
    """
    Pressure flux through each INTERIOR face,

        Phi_f = c_f * (p_N - p_P)/h   +   cross_f

    with c_f the same face-interpolated coefficient the matrix carries, so that the discrete
    divergence of Phi is exactly the matrix action. Domain boundary faces stay zero (Neumann).

    The cross term is the non-orthogonal part (PICT's `nonOrthoFlags`). It must be applied to
    the FLUX as well as to the matrix RHS -- correcting the flux with only the orthogonal part
    while carrying cross terms on the RHS would leave the cross contribution in the divergence.
    """
    per = as_periodic(periodic)
    cw = 1.0 if coef is None else coef
    g_diag = [metrics[k[0]]**2 + metrics[k[1]]**2 + metrics[k[2]]**2 for k in _KEYS]

    dp = [deriv(p, h[a], a, per[a]) for a in range(3)]

    def g_off(a, b):
        ka, kb = _KEYS[a], _KEYS[b]
        return (metrics[ka[0]]*metrics[kb[0]] + metrics[ka[1]]*metrics[kb[1]]
                + metrics[ka[2]]*metrics[kb[2]])

    Phi = []
    for axis in range(3):
        shape = list(J.shape); shape[axis] += 1
        phi = np.zeros(shape)
        lo, hi = _lo_hi(axis)
        interior = [slice(None)] * 3; interior[axis] = slice(1, -1)

        c = cw * J * g_diag[axis]
        if include_orth:
            c_face = 0.5 * (c[lo] + c[hi])
            phi[tuple(interior)] = c_face * (p[hi] - p[lo]) / h[axis]

        if include_cross:
            others = [a for a in range(3) if a != axis]
            cross_cell = sum(cw * J * g_off(axis, b) * dp[b] for b in others)
            phi[tuple(interior)] += 0.5 * (cross_cell[lo] + cross_cell[hi])

        if per[axis]:
            sl_a = [slice(None)] * 3; sl_a[axis] = 0
            sl_b = [slice(None)] * 3; sl_b[axis] = -1
            wrap = np.zeros_like(p[tuple(sl_a)])
            if include_orth:
                cf = 0.5 * (c[tuple(sl_b)] + c[tuple(sl_a)])
                wrap = wrap + cf * (p[tuple(sl_a)] - p[tuple(sl_b)]) / h[axis]
            if include_cross:
                others = [a for a in range(3) if a != axis]
                cc = sum(cw * J * g_off(axis, b) * dp[b] for b in others)
                wrap = wrap + 0.5 * (cc[tuple(sl_b)] + cc[tuple(sl_a)])
            phi[tuple(sl_a)] = wrap
            phi[tuple(sl_b)] = wrap

        Phi.append(phi)
    return Phi

def correct_fluxes(F, Phi):
    """F <- F - Phi, leaving prescribed boundary faces untouched (Phi is zero there)."""
    return [F[a] - Phi[a] for a in range(3)]
