"""
2D lid-driven cavity, run as a span-periodic 3D case.

Periodic in y with a handful of cells removes the end walls entirely, so the solution is
span-invariant -- a genuine 2D cavity, which is exactly Ghia et al.'s configuration. That
makes the comparison apples-to-apples, unlike sampling the mid-plane of a walled 3D cavity.
Resolution in x-z matches Ghia's 129x129.
"""
import numpy as np, time, warnings
warnings.filterwarnings("ignore")
from piso_numpy_3d import PISOSolver

N, NY, DT = 129, 4, 0.02
cases = [("Re100_sou", 0.01, "sou"), ("Re100_central", 0.01, "central"), ("Re400_sou", 0.0025, "sou")]
out = {}
for tag, nu, conv in cases:
    s = PISOSolver((N, NY, N), warp=1e-9, nu=nu, dt=DT, corrector_steps=2,
                   periodic=(False, True, False), convection=conv, pressure_tol=1e-10)
    s.set_lid_driven_cavity(1.0)
    prev, t0 = None, time.time()
    for it in range(3000):
        d = s.step()
        if prev is not None:
            du = np.abs(s.u - prev).max()
            if it % 200 == 0:
                print(f"  {tag} step {it:4d} du={du:.2e} divF={d:.1e} ({time.time()-t0:.0f}s)", flush=True)
            if du < 1e-6:
                print(f"  {tag} converged at {it}, du={du:.2e}", flush=True); break
        prev = s.u.copy()
    span = np.abs(s.u - s.u[:, :1, :]).max()
    print(f"  {tag} DONE du={du:.2e} divF={d:.2e} span-var={span:.1e} {time.time()-t0:.0f}s", flush=True)
    j = 0
    out[f"{tag}_u"] = s.u[:, j, :]; out[f"{tag}_w"] = s.w[:, j, :]
    out[f"{tag}_x"] = s.x[:, j, :]; out[f"{tag}_z"] = s.z[:, j, :]
np.savez("cavity_2d.npz", N=N, **out)
print("saved cavity_2d.npz")
