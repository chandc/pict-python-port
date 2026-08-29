"""Run the 3D lid-driven cavity to steady state and save fields for plotting."""
import numpy as np, time, warnings
warnings.filterwarnings("ignore")
from src.piso_numpy_3d import PISOSolver

N, WARP, DT = 24, 0.05, 0.02
cases = [("Re100", 0.01), ("Re20", 0.05)]
out = {}
for tag, nu in cases:
    s = PISOSolver(N, warp=WARP, nu=nu, dt=DT, corrector_steps=2, pressure_tol=1e-10)
    s.set_lid_driven_cavity(1.0)
    prev, t0 = None, time.time()
    for it in range(400):
        d = s.step()
        if prev is not None:
            du = np.abs(s.u - prev).max()
            if it % 25 == 0:
                print(f"  {tag} step {it:3d}  du={du:.2e}  divF={d:.1e}  ({time.time()-t0:.0f}s)", flush=True)
            if du < 2e-6:
                print(f"  {tag} converged at step {it}, du={du:.2e}", flush=True)
                break
        prev = s.u.copy()
    print(f"  {tag} DONE  du={du:.2e}  max|divF|={d:.2e}  {time.time()-t0:.0f}s", flush=True)
    out[f"{tag}_u"], out[f"{tag}_v"], out[f"{tag}_w"] = s.u, s.v, s.w
    out[f"{tag}_p"], out[f"{tag}_div"] = s.p, s.last_flux_divergence
    out[f"{tag}_nu"], out[f"{tag}_du"] = nu, du
out["x"], out["y"], out["z"] = s.x, s.y, s.z
out["N"], out["warp"], out["dt"] = N, WARP, DT
np.savez("results/cavity_results.npz", **out)
print("saved cavity_results.npz")
