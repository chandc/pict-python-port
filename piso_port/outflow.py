"""
Inflow / outflow boundary conditions for the collocated PISO solver.

Two outflow treatments, because they fail in different places:

CONVECTIVE (PICT's approach, `update_advective_boundaries` in PISOtorch_simulation.py).
    The outlet stays an ordinary Dirichlet velocity boundary whose value is advected out each
    step by  du/dt + U_c du/dn = 0, discretised implicitly as

        alpha = dt * U_c / h_n        t = 1 - 1/(1 + alpha)        u_b <- u_b - t (u_b - u_int)

    t lies in [0,1) for every alpha >= 0, so the update cannot go unstable at any CFL. PICT
    writes alpha = 2*dt*U_c/h because it is CELL-CENTRED and the centre-to-face distance is
    h/2; our nodes sit ON the boundary, so the distance to the first interior node is h_n and
    the factor 2 is absent. Copying their formula verbatim would advect twice too fast.

    The pressure system stays pure Neumann and SINGULAR, so the prescribed boundary fluxes must
    sum to zero or the solve is inconsistent. `balance_boundary_fluxes` enforces that by scaling
    the outflow, mirroring PICT. This is the weakness of the scheme: it imposes a global
    constraint the flow did not ask for, and it carries no mechanism to stop energy entering
    through regions of BACKFLOW (n.u < 0), which is where convective outflow famously blows up.

DONG (Dong, Karniadakis & Chryssostomidis, JCP 261 (2014) 83-105).
    An energy-stable open boundary. The traction condition

        nu du/dn - p n - 1/2 |u|^2 Theta(n,u) n = 0,
        Theta(n,u) = 1/2 (1 - tanh( (n.u) / (U0 delta) ))

    with Theta -> 1 in backflow and -> 0 in outflow. Dotting with n splits it, for a projection
    scheme, into a DIRICHLET pressure at the outlet plus zero normal-gradient velocity:

        p_out = nu d(u.n)/dn - 1/2 |u|^2 Theta(n,u)

    The -1/2|u|^2 Theta term is the whole point: it makes the boundary contribution to dE/dt
    non-positive even when fluid flows back INTO the domain, so vortices crossing the outlet
    cannot pump energy in. Because the outlet nodes carry a Dirichlet pressure they drop out of
    the pressure system, which is then NON-SINGULAR -- no pinned cell, no compatibility
    projection, and no global flux balancing. Mass leaves through the outlet as the solution
    dictates rather than being rescaled to a target.

KNOWN DEFECT (dong, unresolved as of this commit)
    The Dong path does NOT yet pass the Poiseuille exactness test: the error GROWS under
    refinement -- L2/Umax 1.1e-6 / 5.3e-5 / 6.3e-5 on 17x13 / 33x25 / 65x49 -- which is the
    signature of an inconsistency, not of truncation error. The convective path on the same
    case gives 2.3e-8 / 2.7e-7 / 3.3e-8.

    Leading hypothesis: the  nu d(u.n)/dn  term is evaluated EXPLICITLY from the lagged
    velocity as (u_b - u_i)/dn, and that closes a feedback loop with gain ~ nu*dt/dn^2, which
    is 0.016 / 0.064 / 0.256 on those three grids -- growing four-fold per refinement and
    heading for 1 (outright instability) around nx ~ 129. The standard remedy is to replace it
    using continuity at the boundary, d(u.n)/dn = -div_t(u_t), so the term is built from
    TANGENTIAL derivatives of boundary values and no longer feeds back through the normal
    pressure correction. Not yet implemented or verified.

    Until that is fixed, treat `kind="dong"` as work in progress: it runs and is stable, but its
    steady state is not correct to discretisation order. Use `kind="convective"` for production.
"""
import numpy as np

from phase5_fluxes import contravariant_components


class Outflow:
    """One outflow face. axis 0/1/2 = xi/eta/zeta; side 0 = lower, 1 = upper."""

    def __init__(self, axis, side, kind="convective", U_c=1.0, U0=None, delta=0.05,
                 hold=None):
        if kind not in ("convective", "zero_gradient", "dong"):
            raise ValueError(f"unknown outflow kind {kind!r}")
        self.axis, self.side, self.kind = axis, side, kind
        self.U_c, self.U0, self.delta = U_c, U0, delta
        # `hold`: boolean mask over the face marking nodes that must KEEP their prescribed
        # value rather than being advected -- the corner nodes an outlet shares with a no-slip
        # wall. Without it the outflow update overwrites no-slip corners with interior
        # velocity, quietly turning the wall into a slip surface right where the shear is
        # largest, and the Poiseuille profile then fails to hold.
        self.hold = hold

    # ---- index helpers -------------------------------------------------------------
    def bnd_slice(self):
        s = [slice(None)] * 3
        s[self.axis] = 0 if self.side == 0 else -1
        return tuple(s)

    def int_slice(self):
        """The first node INSIDE the domain, adjacent to the boundary node."""
        s = [slice(None)] * 3
        s[self.axis] = 1 if self.side == 0 else -2
        return tuple(s)

    def normal_sign(self):
        """Outward normal in computational space: -1 at the lower face, +1 at the upper."""
        return -1.0 if self.side == 0 else +1.0

    def normal_spacing(self, solver):
        """Physical distance from the boundary node to the first interior node."""
        b, i = self.bnd_slice(), self.int_slice()
        return np.sqrt((solver.x[b] - solver.x[i]) ** 2 +
                       (solver.y[b] - solver.y[i]) ** 2 +
                       (solver.z[b] - solver.z[i]) ** 2)

    def node_indices(self, shape):
        return np.arange(np.prod(shape)).reshape(shape)[self.bnd_slice()].ravel()


def update_outflow_velocity(solver, bcs, dt):
    """Advance every outflow face's Dirichlet velocity by its own rule."""
    for bc in bcs:
        b, i = bc.bnd_slice(), bc.int_slice()
        keep = None if bc.hold is None else np.asarray(bc.hold, dtype=bool)
        if bc.kind == "zero_gradient" or bc.kind == "dong":
            # Dong's tangential condition is du/dn = 0; the traction is carried by the pressure.
            def rule(fbc_b, f_i):
                return f_i
        else:  # convective
            alpha = dt * abs(bc.U_c) / bc.normal_spacing(solver)
            t = 1.0 - 1.0 / (1.0 + alpha)
            def rule(fbc_b, f_i, t=t):
                return fbc_b - t * (fbc_b - f_i)
        for f, fbc in ((solver.u, solver.u_bc), (solver.v, solver.v_bc), (solver.w, solver.w_bc)):
            upd = rule(fbc[b], f[i])
            fbc[b] = fbc[b] if keep is None else np.where(keep, fbc[b], upd)
            if keep is None:
                fbc[b] = upd
    for f, fbc in ((solver.u, solver.u_bc), (solver.v, solver.v_bc), (solver.w, solver.w_bc)):
        for bc in bcs:
            f[bc.bnd_slice()] = fbc[bc.bnd_slice()]


def _face_area_weight(solver, axis):
    """Product of the two computational spacings tangent to `axis`."""
    h = solver.h
    return np.prod([h[a] for a in range(3) if a != axis])


def boundary_flux_totals(solver, bcs):
    """
    Net volume flux out of the domain, split into the FIXED part (inlet, walls) and the
    part carried by the outflow faces. Uses the contravariant components, so it is the same
    quantity the pressure equation sees -- not a re-derived approximation of it.
    """
    JU = contravariant_components(solver.u, solver.v, solver.w, solver.J, solver.metrics)
    out_keys = {(bc.axis, bc.side) for bc in bcs}
    fixed = free = 0.0
    for axis in range(3):
        if solver.per[axis]:
            continue                                  # a periodic seam moves no net mass
        w = _face_area_weight(solver, axis)
        for side in (0, 1):
            s = [slice(None)] * 3
            s[axis] = 0 if side == 0 else -1
            sgn = -1.0 if side == 0 else +1.0
            flux = sgn * float(np.sum(JU[axis][tuple(s)])) * w
            if (axis, side) in out_keys:
                free += flux
            else:
                fixed += flux
    return fixed, free


def balance_boundary_fluxes(solver, bcs, tol=1e-12):
    """
    Scale the outflow velocities so total flux in == total flux out.

    Required ONLY when the pressure system is singular (all-Neumann), which is the case for
    the convective outflow: an incompatible RHS there has no solution, and the pinned solve
    would return a plausible-looking wrong field. Dong outflows make the system non-singular
    and are skipped.
    """
    active = [bc for bc in bcs if bc.kind != "dong"]
    if not active:
        return 1.0
    fixed, free = boundary_flux_totals(solver, active)
    if abs(fixed + free) < tol:
        return 1.0
    if abs(free) < 1e-30:
        raise RuntimeError(
            "cannot balance boundary fluxes: the outflow carries no flux to scale "
            f"(net imbalance {fixed:.3e}). Give the outlet a non-zero initial velocity.")
    scale = -fixed / free
    for bc in active:
        b = bc.bnd_slice()
        keep = None if bc.hold is None else np.asarray(bc.hold, dtype=bool)
        for f, fbc in ((solver.u, solver.u_bc), (solver.v, solver.v_bc), (solver.w, solver.w_bc)):
            scaled = fbc[b] * scale
            fbc[b] = scaled if keep is None else np.where(keep, fbc[b], scaled)
            f[b] = fbc[b]
    return scale


def dong_pressure(solver, bc):
    """
    Dirichlet pressure on a Dong outflow face:

        p = nu d(u.n)/dn - 1/2 |u|^2 Theta(n,u),    Theta = 1/2 (1 - tanh( (n.u)/(U0 delta) ))

    Theta switches on only where fluid enters (n.u < 0); in clean outflow it vanishes and the
    condition reduces to the ordinary traction-free p = nu d(u.n)/dn.
    """
    b, i = bc.bnd_slice(), bc.int_slice()
    m, ax = solver.metrics, bc.axis
    key = ("xi", "eta", "zeta")[ax]
    # unit outward normal in physical space, from the contravariant basis vector grad(xi_ax)
    nx, ny, nz = m[f"{key}_x"][b], m[f"{key}_y"][b], m[f"{key}_z"][b]
    nrm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    sgn = bc.normal_sign()
    nx, ny, nz = sgn * nx / nrm, sgn * ny / nrm, sgn * nz / nrm

    ub, vb, wb = solver.u[b], solver.v[b], solver.w[b]
    un = ub * nx + vb * ny + wb * nz
    un_i = solver.u[i] * nx + solver.v[i] * ny + solver.w[i] * nz
    dn = bc.normal_spacing(solver)
    dundn = (un - un_i) / dn * sgn * sgn          # outward-directed difference

    U0 = bc.U0 if bc.U0 is not None else max(float(np.max(np.abs(un))), 1e-12)
    theta = 0.5 * (1.0 - np.tanh(un / (U0 * bc.delta)))
    usq = ub ** 2 + vb ** 2 + wb ** 2
    return solver.nu * dundn - 0.5 * usq * theta


def dirichlet_pressure_nodes(solver, bcs):
    """(indices, values) for every Dong outflow node; empty arrays if there are none."""
    idx, val = [], []
    for bc in bcs:
        if bc.kind != "dong":
            continue
        idx.append(bc.node_indices(solver.J.shape))
        val.append(dong_pressure(solver, bc).ravel())
    if not idx:
        return np.empty(0, dtype=int), np.empty(0)
    idx = np.concatenate(idx)
    val = np.concatenate(val)
    uniq, first = np.unique(idx, return_index=True)    # a corner node may sit on two faces
    return uniq, val[first]
