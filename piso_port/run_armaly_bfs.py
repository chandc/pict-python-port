"""
Run the Armaly backward-facing step and measure the reattachment length.

Re = U_bulk * D_h / nu with D_h = 2 h_in, following Armaly. The inlet is the fully developed
parabolic profile at the step plane (the upstream channel is not resolved -- see
armaly_bfs_grid.py for why that is what the two-block decomposition gives).

Reattachment is located where the wall shear on the BOTTOM wall changes sign: du/dy|_wall goes
negative in the recirculation bubble and positive downstream of it. Armaly's laminar data give
x_r/S rising from ~3 at Re=100 to ~6-7 near Re=400.

The span is periodic and the flow is two-dimensional at these Reynolds numbers, so nz is kept
small: a wide span would multiply the cost without changing the answer.
"""
import sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from src.multiblock import face_id
from src.piso_multiblock import MultiBlockPISO
from armaly_bfs_grid import bfs_domain, S, H_IN

U_BULK = 1.0
D_H = 2.0 * H_IN


def run(Re=100.0, nx=100, ny_lo=22, ny_up=24, nz=4, dt=0.02, nsteps=400, report=True):
    nu = U_BULK * D_H / Re
    d, LOW, UP = bfs_domain(nx=nx, ny_lo=ny_lo, ny_up=ny_up, nz=nz)
    m = MultiBlockPISO(d, nu, dt, 2, 1e-11, time_scheme="be", scheme="rotational",
                       picard_iters=1)
    up, lo = d.blocks[UP], d.blocks[LOW]

    # fully developed parabola over the inlet height, bulk U_BULK
    yin = up.y[0, :, :]
    prof = 6.0 * U_BULK * (yin / H_IN) * (1.0 - yin / H_IN)
    m.u_bc[UP][0, :, :] = prof
    m.u[UP][:] = prof[None, :, :]                    # start with the inlet stream everywhere
    m.u[LOW][:] = 0.0
    for b, blk in ((UP, up), (LOW, lo)):             # no-slip on every wall face
        for f, kind in enumerate(blk.faces):
            if kind in ("periodic", "connected") or f == face_id(0, 1):
                continue
            if b == UP and f == face_id(0, 0):
                continue                              # that is the inlet, not a wall
            ax, sd = f // 2, f % 2
            sl = [slice(None)] * 3; sl[ax] = 0 if sd == 0 else -1
            m.u_bc[b][tuple(sl)] = 0.0
            m.u[b][tuple(sl)] = 0.0
    m.outflow = [(UP, face_id(0, 1), U_BULK, "convective"),
                 (LOW, face_id(0, 1), U_BULK, "convective")]
    m.u_bc[UP][-1, :, :] = prof
    m.u_bc[LOW][-1, :, :] = 0.0

    t0 = time.time()
    prev = None
    for it in range(nsteps):
        div = m.step()
        if not np.isfinite(m.u[UP]).all() or not np.isfinite(m.u[LOW]).all():
            return None, f"diverged at step {it+1}", None, None
        cur = np.concatenate([m.u[LOW].ravel(), m.u[UP].ravel()])
        if prev is not None and np.abs(cur - prev).max() < 1e-9:
            break
        prev = cur
    wall = time.time() - t0

    # bottom-wall shear: du/dy at the first cell above the wall
    xb = lo.x[:, 0, nz // 2]
    dudy = (m.u[LOW][:, 1, nz // 2] - m.u[LOW][:, 0, nz // 2]) / (lo.y[:, 1, nz // 2] -
                                                                  lo.y[:, 0, nz // 2])
    xr = np.nan
    neg = dudy < 0
    if neg.any():
        i = np.where(neg)[0][-1]
        if i + 1 < len(dudy) and dudy[i + 1] > 0:     # interpolate the sign change
            f = -dudy[i] / (dudy[i + 1] - dudy[i])
            xr = xb[i] + f * (xb[i + 1] - xb[i])
    if report:
        print(f"   Re={Re:6.0f}  nu={nu:.5f}  {it+1:4d} steps  divF {div:.1e}  "
              f"x_r/S = {xr:6.3f}   [{wall:.0f}s]")
    return xr, None, m, d


if __name__ == "__main__":
    print(__doc__.strip().split("\n\n")[0])
    print(f"\n  geometry: S=1, h_in={H_IN:.4f}, ER={(S+H_IN)/H_IN:.4f}, D_h={D_H:.4f}")
    print(f"  {'':>3}Armaly laminar reference: x_r/S ~ 3 at Re=100, ~6-7 near Re=400\n")
    out = {}
    for Re in (100.0, 200.0, 300.0):
        xr, err, _, _ = run(Re=Re, nsteps=3000)
        if err:
            print(f"   Re={Re:6.0f}  {err}")
        out[Re] = xr
    ok = all(np.isfinite(v) for v in out.values()) and \
        all(out[a] < out[b] for a, b in ((100.0, 200.0), (200.0, 300.0)))
    print(f"\n  [{'PASS' if ok else 'FAIL'}] a recirculation forms and lengthens with Re: " +
          ", ".join(f"Re={k:.0f} -> {v:.2f}S" for k, v in out.items()))
    sys.exit(0 if ok else 1)
