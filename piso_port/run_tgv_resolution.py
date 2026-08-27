"""Grid-resolution study for the 3D TGV energy budget: is 48^3 converged?"""
import numpy as np, time, warnings
warnings.filterwarnings("ignore")
from piso_numpy_3d import PISOSolver
from run_tgv3d import tgv_init, diagnostics

NU, DT, T_END = 0.01, 0.005, 2.0
out = {}
for N in (32, 48, 64):
    s = PISOSolver(N, warp=1e-9, nu=NU, dt=DT, corrector_steps=2, periodic=True,
                   scheme="rotational", time_scheme="bdf2", convection="central",
                   pressure_tol=1e-11)
    u0, v0, w0 = tgv_init(s.x, s.y, s.z)
    s.u, s.v, s.w = u0.copy(), v0.copy(), w0.copy()
    E, Z = diagnostics(s); ts, Es, Zs = [0.0], [E], [Z]
    t0 = time.time()
    for it in range(int(round(T_END / DT))):
        s.step(); E, Z = diagnostics(s)
        ts.append((it+1)*DT); Es.append(E); Zs.append(Z)
    out[f"n{N}_t"] = np.array(ts); out[f"n{N}_E"] = np.array(Es); out[f"n{N}_Z"] = np.array(Zs)
    dEdt = np.gradient(np.array(Es), np.array(ts)); phys = 2*NU*np.array(Zs)
    m = (np.array(ts) > 0.1) & (np.array(ts) < 1.5)
    rel = np.abs(-dEdt - phys)[m] / np.abs(phys)[m]
    print(f"  N={N}: E(2)={Es[-1]:.6e}  numerical dissipation {rel.mean()*100:.2f}%  "
          f"({time.time()-t0:.0f}s)", flush=True)
np.savez("tgv_resolution.npz", nu=NU, **out)
print("saved tgv_resolution.npz")
