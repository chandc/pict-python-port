"""Cartesian (unwarped) cavity runs at Re=100 to isolate warp vs 3D effects."""
import numpy as np, time, warnings
warnings.filterwarnings("ignore")
from piso_numpy_3d import PISOSolver

out = {}
for tag, N, warp in [("cart24", 24, 1e-9), ("cart32", 32, 1e-9)]:
    s = PISOSolver(N, warp=warp, nu=0.01, dt=0.02, corrector_steps=2, pressure_tol=1e-10)
    s.set_lid_driven_cavity(1.0)
    prev, t0 = None, time.time()
    for it in range(600):
        d = s.step()
        if prev is not None:
            du = np.abs(s.u - prev).max()
            if it % 50 == 0:
                print(f"  {tag} step {it:3d} du={du:.2e} divF={d:.1e} ({time.time()-t0:.0f}s)", flush=True)
            if du < 2e-6:
                print(f"  {tag} converged at {it}, du={du:.2e}", flush=True); break
        prev = s.u.copy()
    print(f"  {tag} DONE du={du:.2e} divF={d:.2e} {time.time()-t0:.0f}s", flush=True)
    out[f"{tag}_u"], out[f"{tag}_w"] = s.u, s.w
    out[f"{tag}_x"], out[f"{tag}_z"] = s.x, s.z
    out[f"{tag}_N"] = N
np.savez("cavity_cartesian.npz", **out)
print("saved cavity_cartesian.npz")
