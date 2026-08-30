"""Five-domain Armaly BFS with Dong outflow AND the Rhie-Chow correction.

The companion to run_armaly_bfs5.py. Kept as a separate entry point rather than a flag on the
sweep so both sets of checkpoints survive side by side and can be compared without re-running
either: the pressure field is the whole reason the correction exists, and the only way to show
what it bought is to have both.
"""
import numpy as np
from run_armaly_bfs5 import run

if __name__ == "__main__":
    print(__doc__.strip().split("\n\n")[0])
    print("\n  Reattachment reference: x_r/S ~ 3 / 5 / 6-7\n")
    med = dict(nx_in=20, nx_re=40, nx_rc=40, ny_lo=18, ny_up=19, nz=8)
    out = {}
    for Re in (100.0, 200.0, 300.0):
        xr, err, m, d = run(Re=Re, dt=0.02, nsteps=3000, dong=True, rhie_chow=True,
                            grid=med)
        if err:
            print(f"   Re={Re:.0f}: {err}", flush=True)
        out[Re] = xr
    ok = all(np.isfinite(v) for v in out.values()) and out[100.0] < out[200.0] < out[300.0]
    print(f"\n  [{'PASS' if ok else 'FAIL'}] recirculation lengthens with Re: " +
          ", ".join(f"Re={k:.0f} -> {v:.2f}S" for k, v in out.items()))
