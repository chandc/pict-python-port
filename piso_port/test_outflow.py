"""
Inflow / outflow boundary conditions: exactness on Poiseuille, and robustness under BACKFLOW.

Two outflow treatments (see outflow.py):

  convective  -- PICT's. The outlet is a Dirichlet velocity boundary advected out by
                 du/dt + U_c du/dn = 0. The pressure system stays singular, so the boundary
                 fluxes must be rescaled to balance globally.
  dong        -- Dong, Karniadakis & Chryssostomidis, JCP 261 (2014) 83-105. A Dirichlet
                 pressure  p = nu d(u.n)/dn - 1/2 |u|^2 Theta(n,u)  with
                 Theta = 1/2 (1 - tanh(u.n/(U0 delta))). The extra term makes the boundary
                 contribution to dE/dt non-positive even where fluid flows back IN, which is
                 the situation convective outflow is known to fail on. It also makes the
                 pressure matrix non-singular, so no pinned cell and no flux balancing.

A note on the scheme, learned the hard way here: with scheme='chorin' the steady state carries
an O(dt) error that does NOT vanish as the flow converges -- the predictor drops grad p, so the
converged state solves  A u = rhs - dt A grad p  instead of  A u = rhs - J grad p. Measured on
this exact case: 5.47e-3 / 2.78e-3 / 1.40e-3 at dt = 0.04 / 0.02 / 0.01, grid-independent.
The accumulating schemes keep grad p^n in the predictor and drop it to ~3e-7. That is a
property of Chorin, not of the outflow, but it will masquerade as a boundary-condition error
in any steady test, so these tests use 'rotational'.
"""
import sys, warnings, io, contextlib
import numpy as np
warnings.filterwarnings("ignore")
from piso_numpy_3d import PISOSolver
from phase1_grid_metrics import compute_numerical_metrics
from outflow import Outflow, boundary_flux_totals

NU, UMAX, L = 0.05, 1.0, 4.0
results = []
def check(name, good, detail):
    results.append(good)
    print(f"  [{'PASS' if good else 'FAIL'}] {name}: {detail}")


def build(nx, ny, kind, dt=0.02, nu=NU, scheme="rotational", vortex=None):
    nz = 4
    bc = Outflow(axis=0, side=1, kind=kind, U_c=2.0 / 3.0 * UMAX)
    s = PISOSolver((nx, ny, nz), warp=1e-9, nu=nu, dt=dt, corrector_steps=2,
                   periodic=(False, False, True), scheme=scheme, time_scheme="be",
                   boundary_flux_mode="from_velocity", pressure_coef="rowsum",
                   pressure_tol=1e-11, outflow=[bc])
    xi = np.linspace(0, 1, nx); eta = np.linspace(0, 1, ny); ze = np.arange(nz) / nz
    XI, ETA, ZE = np.meshgrid(xi, eta, ze, indexing="ij")
    s.x, s.y, s.z = L * XI, ETA, ZE
    s.h = (1.0 / (nx - 1), 1.0 / (ny - 1), 1.0 / nz)
    s.J, s.metrics = compute_numerical_metrics(s.x, s.y, s.z, *s.h, periodic=s.per)

    prof = 4 * UMAX * s.y * (1 - s.y)                      # exact Poiseuille
    s.u_bc[0, :, :] = prof[0, :, :]                        # inflow
    s.u_bc[:, 0, :] = 0.0; s.u_bc[:, -1, :] = 0.0          # no-slip walls
    # The outlet's end nodes are ALSO wall nodes. Without holding them the advective update
    # overwrites no-slip corners with interior velocity, turning the wall into a slip surface
    # exactly where the shear is largest.
    hold = np.zeros_like(prof[-1, :, :], dtype=bool); hold[0, :] = True; hold[-1, :] = True
    bc.hold = hold
    s.u_bc[-1, :, :] = prof[-1, :, :]
    s.u[:] = prof; s.v[:] = 0.0; s.w[:] = 0.0
    s.u[:, 0, :] = 0.0; s.u[:, -1, :] = 0.0

    if vortex is not None:
        amp, x0, y0, r = vortex
        X, Y = s.x, s.y
        psi = amp * np.exp(-(((X - x0) ** 2 + (Y - y0) ** 2) / r ** 2))
        s.u += psi * (-2 * (Y - y0) / r ** 2)
        s.v -= psi * (-2 * (X - x0) / r ** 2)
        s.u[:, 0, :] = 0.0; s.u[:, -1, :] = 0.0
        s.v[:, 0, :] = 0.0; s.v[:, -1, :] = 0.0
        s.u[0, :, :] = prof[0, :, :]; s.v[0, :, :] = 0.0
    return s, bc, prof


def to_steady(s, tol=1e-11, maxit=6000):
    prev = None
    with contextlib.redirect_stdout(io.StringIO()):
        for it in range(maxit):
            s.step()
            if prev is not None and np.abs(s.u - prev).max() < tol:
                break
            prev = s.u.copy()
    return it + 1


if __name__ == "__main__":
    print("1. Fully developed Poiseuille through an inflow/outflow channel")
    print("   (the exact solution is the parabola everywhere, so a good outlet must not distort it)")
    print(f"   {'BC':>12} {'grid':>9} {'L2/Umax':>11} {'outlet vs exact':>16} {'flux imbalance':>15}")
    for kind in ("convective", "dong"):
        errs = []
        for nx, ny in ((17, 13), (33, 25), (65, 49)):
            s, bc, prof = build(nx, ny, kind)
            to_steady(s)
            e = np.sqrt(np.mean((s.u - prof) ** 2)) / UMAX
            errs.append(e)
            out = np.abs(s.u[-1] - prof[-1]).max() / UMAX
            fixed, free = boundary_flux_totals(s, [bc])
            print(f"   {kind:>12} {f'{nx}x{ny}':>9} {e:11.3e} {out:16.3e} {fixed+free:+15.2e}")
        check(f"{kind}: reproduces the exact parabola", max(errs) < 5e-6,
              f"worst L2/Umax {max(errs):.2e} over three grids")

    print("\n2. Mass conservation (inflow flux must equal outflow flux)")
    for kind in ("convective", "dong"):
        s, bc, prof = build(33, 25, kind)
        to_steady(s)
        fixed, free = boundary_flux_totals(s, [bc])
        rel = abs(fixed + free) / abs(fixed)
        print(f"   {kind:>12}  in {-fixed:.6f}  out {free:.6f}  relative imbalance {rel:.2e}")
        check(f"{kind}: conserves mass", rel < 1e-6, f"relative imbalance {rel:.2e}")

    print("\n3. BACKFLOW robustness — a vortex driven out through the outlet at Re=200")
    print("   (this is the regime convective outflow is known to fail in; Dong's -1/2|u|^2*Theta")
    print("    term exists precisely to stop energy entering where n.u < 0)")
    print(f"   {'BC':>12} {'steps':>6} {'max|u| end':>11} {'E_end/E_0':>10} {'backflow frac':>14} {'status':>10}")
    verdict = {}
    for kind in ("convective", "dong"):
        s, bc, prof = build(49, 33, kind, dt=0.005, nu=UMAX * 1.0 / 200.0,
                            vortex=(0.55, 0.80 * L, 0.5, 0.22))
        E0 = 0.5 * np.mean(s.u ** 2 + s.v ** 2)
        blown, nsteps, bfrac = False, 0, 0.0
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(1200):
                s.step(); nsteps = i + 1
                bfrac = max(bfrac, float(np.mean(s.u[-1] < 0)))
                if not np.isfinite(s.u).all() or np.abs(s.u).max() > 50 * UMAX:
                    blown = True; break
        E1 = 0.5 * np.mean(s.u ** 2 + s.v ** 2)
        verdict[kind] = (blown, np.abs(s.u).max(), E1 / E0)
        print(f"   {kind:>12} {nsteps:6d} {np.abs(s.u).max():11.3e} {E1/E0:10.4f} "
              f"{bfrac:14.3f} {'DIVERGED' if blown else 'stable':>10}")
    check("dong survives backflow", not verdict["dong"][0],
          f"max|u| {verdict['dong'][1]:.3e}, E_end/E_0 {verdict['dong'][2]:.4f}")
    check("dong does not gain energy through the outlet", verdict["dong"][2] <= 1.02,
          f"E_end/E_0 = {verdict['dong'][2]:.4f} (energy-stable BC must not pump energy in)")

    n_pass = sum(results)
    print(f"\n{'='*76}\n  {n_pass}/{len(results)} checks passed\n{'='*76}")
    sys.exit(0 if n_pass == len(results) else 1)
