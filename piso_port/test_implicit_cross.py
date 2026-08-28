"""
Implicit treatment of the non-orthogonal cross terms: equivalence and performance.

The default path carries the cross terms by DEFERRED CORRECTION -- a Picard iteration whose
contraction ratio grows with grid warp (0.31 / 0.59 / 0.92 at warp 0.05 / 0.10 / 0.15).
Setting implicit_cross=True removes the loop:
the operator solved there IS the fixed point that loop converges to, so the two paths must
agree to solver tolerance.

Making that pay off needs the right preconditioner. The full operator is applied matrix-free
and one application costs ~28x a sparse matvec, so the implicit path only wins if the
iteration count falls hard. Preconditioning with the orthogonal part M gives

    M^-1 A  =  I - M^-1 J D(Phi_cross(.))

whose second term has spectral radius equal to the deferred-correction contraction ratio --
so the preconditioned spectrum clusters at 1 with spread rho, and Krylov resolves it in a
few iterations rather than the ~rho^k of a fixed-point sweep.

Gates:
  1. EQUIVALENCE -- both paths give the same answer (they solve the same system)
  2. PERFORMANCE -- wall time and iteration counts, reported side by side
  3. two real problems: the 2D (span-periodic) cavity, and the force-driven channel
"""
import sys, time, warnings, io, contextlib
import numpy as np
warnings.filterwarnings("ignore")
from piso_numpy_3d import PISOSolver

results = []
def check(name, good, detail):
    results.append(good)
    print(f"  [{'PASS' if good else 'FAIL'}] {name}: {detail}")


def cavity(n, warp, implicit, nsteps, dt=0.02):
    """2D lid-driven cavity: span-periodic, walls in x and z."""
    s = PISOSolver((n, 4, n), warp=max(warp, 1e-9), nu=0.01, dt=dt, corrector_steps=2,
                   periodic=(False, True, False), scheme="chorin",
                   pressure_tol=1e-11, implicit_cross=implicit)
    s.set_lid_driven_cavity(1.0)
    return _run(s, nsteps, implicit)


def channel(n, warp, implicit, nsteps, dt=0.05):
    """Force-driven channel: walls in y, periodic in x and z."""
    s = PISOSolver((4, n, 4), warp=max(warp, 1e-9), nu=0.1, dt=dt, corrector_steps=2,
                   periodic=(True, False, True), scheme="chorin",
                   boundary_flux_mode="impermeable", pressure_tol=1e-11,
                   implicit_cross=implicit)
    s.velocity_source = [np.full_like(s.y, 0.8), np.zeros_like(s.y), np.zeros_like(s.y)]
    return _run(s, nsteps, implicit)


def _run(s, nsteps, implicit):
    t0 = time.time()
    for _ in range(nsteps):
        with contextlib.redirect_stdout(io.StringIO()):
            d = s.step()
    its = s._implicit_its if implicit else s._dc_sweeps
    return s.u.copy(), time.time() - t0, d, its


print("Equivalence and performance (20 steps; iteration count is for the final pressure solve)")
for name, fn, n in (("2D cavity", cavity, 16), ("channel", channel, 16)):
    print(f"\n{name}  n={n}")
    print(f"   {'warp':>5}  {'deferred':>21}   {'implicit':>21}   speed-up")
    for warp in (0.0, 0.05, 0.10, 0.15):
        u_dc, t_dc, d_dc, i_dc = fn(n, warp, False, 20)
        u_ic, t_ic, d_ic, i_ic = fn(n, warp, True, 20)
        rel = np.abs(u_ic - u_dc).max() / max(np.abs(u_dc).max(), 1e-30)
        w = 0.0 if warp < 1e-6 else warp
        print(f"   {w:5.2f}  {t_dc:7.2f}s {i_dc:4d} sweeps   {t_ic:7.2f}s {i_ic:4d} Krylov   "
              f"{t_dc/t_ic:5.2f}x")
        check(f"{name} warp={w:.2f}: same answer", rel < 1e-6,
              f"max relative difference {rel:.2e}")

# --- grid validity, and what it means for the "warp limit"
#
# The deferred correction was characterised earlier as stalling near warp 0.18 (contraction
# ratio 0.31 / 0.59 / 0.92 / 1.27 at warp 0.05 / 0.10 / 0.15 / 0.20). Measuring the Jacobian
# shows that limit is NOT a property of deferred correction: this grid family TANGLES at the
# same place -- min(J) goes negative at warp 0.18 -- so the ratio > 1 was measured on a mesh
# with negative cell volumes. Both solve paths blow up there, at every dt tried (0.02 / 0.005
# / 0.001), which is what a tangled grid looks like and not what a CFL limit looks like.
#
# Consequence, stated plainly: the implicit option CANNOT be shown to extend the usable warp
# range, because there is no valid grid beyond the deferred-correction limit to show it on.
# Its demonstrated value is the speed-up at warps that are actually meshable. This assertion
# exists so that "test at higher warp" cannot silently become "test on a broken mesh" again.
print("\nGrid validity (min Jacobian must stay positive)")
for w in (0.05, 0.10, 0.15, 0.18, 0.25):
    s = PISOSolver((16, 4, 16), warp=w, nu=0.01, dt=0.02, periodic=(False, True, False))
    print(f"   warp {w:4.2f}:  min(J) = {s.J.min():10.3e}   "
          f"{'valid' if s.J.min() > 0 else 'TANGLED -- not a usable test grid'}")
check("tested warps use untangled grids", all(
    PISOSolver((16, 4, 16), warp=w, nu=0.01, dt=0.02, periodic=(False, True, False)).J.min() > 0
    for w in (0.05, 0.10, 0.15)), "min(J) > 0 at every warp exercised above")

# --- symmetry is boundary-condition dependent, and that choice is load-bearing
#
# CG is only legal on a symmetric operator. Measuring <v,Aw> vs <Av,w> directly: the full
# operator is symmetric under all-periodic BCs but NOT once a wall axis is present, because
# the one-sided edge differences there are not self-adjoint. That is why the solver picks CG
# only when every axis is periodic and BiCGStab otherwise. Gated here rather than left as a
# comment, since silently feeding a non-symmetric operator to CG is exactly the kind of bug
# that produces a plausible-looking wrong answer.
print("\nOperator symmetry by boundary condition (12^3, warp 0.10)")
from phase5_fluxes import pressure_face_fluxes, divergence_from_fluxes
from phase3_momentum import build_conservative_diffusion_matrix
rng = np.random.default_rng(0)
sym = {}
for label, per in (("all periodic", (True, True, True)),
                   ("walls in y (channel)", (True, False, True)),
                   ("walls in x,z (cavity)", (False, True, False))):
    sv = PISOSolver((12, 12, 12), warp=0.10, nu=0.01, dt=0.02, periodic=per)
    J, coef = sv.J, np.full_like(sv.J, sv.dt)
    M = build_conservative_diffusion_matrix(*sv.shape, *sv.h, J, sv.metrics,
                                            coef=coef, periodic=sv.per)
    def A(x):
        Pc = pressure_face_fluxes(x.reshape(sv.shape), J, sv.metrics, sv.h, coef=coef,
                                  include_orth=False, include_cross=True, periodic=sv.per)
        return (M @ x) - (J * divergence_from_fluxes(Pc, J, sv.h)).ravel()
    v, w = rng.standard_normal(J.size), rng.standard_normal(J.size)
    a, b = v @ A(w), w @ A(v)
    sym[label] = abs(a - b) / max(abs(a), abs(b), 1e-30)
    print(f"   {label:24s} asymmetry {sym[label]:.2e}   "
          f"{'symmetric -> CG' if sym[label] < 1e-10 else 'NOT symmetric -> BiCGStab'}")
check("CG used only where the operator is symmetric",
      sym["all periodic"] < 1e-10 and sym["walls in y (channel)"] > 1e-10
      and sym["walls in x,z (cavity)"] > 1e-10,
      "periodic symmetric; both walled cases are not, matching the solver's CG/BiCGStab switch")

n_pass = sum(results)
print(f"\n{'='*72}\n  {n_pass}/{len(results)} checks passed\n{'='*72}")
sys.exit(0 if n_pass == len(results) else 1)
