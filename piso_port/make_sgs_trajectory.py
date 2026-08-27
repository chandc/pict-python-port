"""
Filtered fine-grid TRAJECTORY for a-posteriori training (Stage 5b).

Stage 5a needed scattered snapshots; 5b needs consecutive states, because the loss compares a
coarse ROLLOUT against the filtered fine trajectory step for step. Fine and coarse therefore
share the same dt so the two align.
"""
import numpy as np, time, warnings
warnings.filterwarnings("ignore")
from piso_numpy_3d import PISOSolver
from make_sgs_data import random_solenoidal, box_filter, sgs_force, FINE, COARSE, NU, DT

STEPS, SPINUP = 30, 60
s = PISOSolver(FINE, warp=1e-9, nu=NU, dt=DT, corrector_steps=2, periodic=True,
               scheme="rotational", time_scheme="bdf2", pressure_tol=1e-11)
u, v, w = random_solenoidal(FINE, seed=7, kmax=6, amp=1.0)
s.u, s.v, s.w = u, v, w
t0 = time.time()
for _ in range(SPINUP):
    s.step()
print(f"spin-up done ({time.time()-t0:.0f}s), max|u|={np.abs(s.u).max():.3f}", flush=True)

hc = 1.0 / COARSE
bar, force = sgs_force(s.u, s.v, s.w, hc)
traj, sgs = [np.stack(bar)], [np.stack(force)]
for k in range(STEPS):
    s.step()
    bar, force = sgs_force(s.u, s.v, s.w, hc)
    traj.append(np.stack(bar)); sgs.append(np.stack(force))
    if k % 10 == 0:
        print(f"  step {k:2d}  |u_bar|max={np.abs(traj[-1][0]).max():.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
np.savez("sgs_traj.npz", traj=np.array(traj), sgs=np.array(sgs),
         nu=NU, dt=DT, coarse=COARSE)   # exact SGS force -> the achievable floor
print(f"saved sgs_traj.npz  {len(traj)} consecutive filtered states")
