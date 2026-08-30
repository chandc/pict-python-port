"""
Armaly backward-facing step on the FIVE-domain grid, with Dong outflow.

The inlet profile is prescribed at x = -5S and develops over the upstream channel before it
reaches the step, rather than being imposed at the step plane as the two-domain case must.

WHY DONG RATHER THAN CONVECTIVE. Both are implemented; Dong is the energy-stable one. It
prescribes a pressure at the outlet, which makes the pressure system NON-SINGULAR -- so the
compatibility projection (`rhs -= rhs.mean()`) is skipped and the outlet flux must NOT be
rescaled by `balance_boundary_fluxes`, since mass is free to leave as the solution dictates.
That is a genuinely different regime from the convective outlet, not a cosmetic swap.

Reattachment is located from the sign change of the bottom-wall shear, which lives entirely
in the recirculation-lower and recovery-lower blocks.
"""
import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")
from src.multiblock import face_id
from src.piso_multiblock import MultiBlockPISO
from src import checkpoint as ck
from armaly_bfs5_grid import (bfs5_domain, S, H_IN, L_IN, X1,
                              INLET, RECIRC_U, RECIRC_L, RECOV_U, RECOV_L)

U_BULK = 1.0
D_H = 2.0 * H_IN


def setup(d, Re, dt, rhie_chow=False, dong=True):
    nu = U_BULK * D_H / Re
    m = MultiBlockPISO(d, nu, dt, 2, 1e-11, time_scheme="be", scheme="rotational",
                       picard_iters=1, rhie_chow=rhie_chow,
                       persistent_flux=rhie_chow, ddt_corr=rhie_chow)

    # fully developed parabola over the inlet height, prescribed at x = -L_IN
    yin = d.blocks[INLET].y[0, :, :]
    prof = 6.0 * U_BULK * (yin / H_IN) * (1.0 - yin / H_IN)
    m.u_bc[INLET][0, :, :] = prof
    for b in (INLET, RECIRC_U, RECOV_U):                 # start with the inlet stream
        m.u[b][:] = prof[None, :, :] if b == INLET else prof[None, :, :]
    m.u[RECIRC_L][:] = 0.0
    m.u[RECOV_L][:] = 0.0

    # no-slip on every wall face except the inlet itself
    for b in range(len(d.blocks)):
        blk = d.blocks[b]
        for f, kind in enumerate(blk.faces):
            if kind in ("periodic", "connected"):
                continue
            if b == INLET and f == face_id(0, 0):        # inlet, not a wall
                continue
            if b in (RECOV_U, RECOV_L) and f == face_id(0, 1):   # outlet, not a wall
                continue
            ax, sd = f // 2, f % 2
            sl = [slice(None)] * 3
            sl[ax] = 0 if sd == 0 else -1
            m.u_bc[b][tuple(sl)] = 0.0
            m.u[b][tuple(sl)] = 0.0

    kind = "dong" if dong else "convective"
    m.outflow = [(RECOV_U, face_id(0, 1), U_BULK, kind),
                 (RECOV_L, face_id(0, 1), U_BULK, kind)]
    return m


def reattachment(m, d, nz):
    """x where the bottom-wall shear changes sign, searched across both lower blocks."""
    xs, tau = [], []
    for b in (RECIRC_L, RECOV_L):
        blk = d.blocks[b]
        k = nz // 2
        xs.append(blk.x[:, 0, k])
        dy = blk.y[:, 1, k] - blk.y[:, 0, k]
        tau.append((m.u[b][:, 1, k] - m.u[b][:, 0, k]) / dy)
    x = np.concatenate(xs); t = np.concatenate(tau)
    o = np.argsort(x); x, t = x[o], t[o]
    neg = t < 0
    if not neg.any():
        return np.nan
    i = np.where(neg)[0][-1]
    if i + 1 >= len(t) or t[i + 1] <= 0:
        return np.nan
    f = -t[i] / (t[i + 1] - t[i])
    return x[i] + f * (x[i + 1] - x[i])


def interior_divergence(m, d):
    """max |div F| EXCLUDING Dong's Dirichlet nodes.

    step() reports a max over ALL cells, and a Dong outlet node carries a PRESCRIBED pressure:
    it leaves the unknown set, mass leaves as the solution dictates, and its divergence is not
    supposed to vanish. Reporting that max makes a healthy run look broken -- measured 3.8e-04
    on the Dirichlet nodes against 1.4e-14 everywhere else.
    """
    nodes, _ = m._dong_nodes()
    glob = np.zeros(d.n_cells, bool)
    if nodes.size:
        glob[nodes] = True
    worst = 0.0
    for b in range(len(d.blocks)):
        div = np.abs(d.divergence(b, m.F_prev[b], m.Js[b]))
        mask = glob[d.global_ids(b)]
        if (~mask).any():
            worst = max(worst, div[~mask].max())
    return worst


def run(Re=100.0, dt=0.02, nsteps=3000, rhie_chow=False, dong=True, grid=None,
        report=True, save=True, restart_from=None):
    d = bfs5_domain(**(grid or {}))
    nz = d.blocks[0].shape[2]
    m = setup(d, Re, dt, rhie_chow=rhie_chow, dong=dong)
    if restart_from:
        # strict=False on purpose: the Reynolds number (hence nu) differs from the saved run,
        # which is the whole point of continuing from it. Everything else -- grid, scheme,
        # dt, corrector count -- must still match, and the shape checks are NOT relaxed.
        meta = ck.load(m, restart_from, strict=False)
        if report:
            print(f"      restarted from {restart_from} "
                  f"(Re {float(meta['extra']['Re']):.0f} -> {Re:.0f}, "
                  f"{meta['nstep']} steps done)", flush=True)
    t0 = time.time()
    prev = None
    for it in range(nsteps):
        div = m.step()
        if not all(np.isfinite(m.u[b]).all() for b in range(len(d.blocks))):
            return None, f"diverged at step {it + 1}", m, d
        cur = np.concatenate([m.u[b].ravel() for b in range(len(d.blocks))])
        if prev is not None and np.abs(cur - prev).max() < 1e-9:
            break
        prev = cur
    xr = reattachment(m, d, nz)
    divi = interior_divergence(m, d)
    if save:
        # EVERY run is saved. A completed run that leaves only a printed number has to be
        # repeated to be post-processed, which is exactly the waste this avoids. The
        # checkpoint carries full restart state (u_prev, p_flux, F_prev), so a run can also
        # be CONTINUED rather than restarted, and load_fields() reads it for plotting
        # without constructing a solver at all.
        os.makedirs("results/fields", exist_ok=True)
        tag = (f"bfs5_Re{int(Re)}_{'dong' if dong else 'conv'}"
               f"{'_rc' if rhie_chow else ''}_n{d.n_cells}")
        path = f"results/fields/{tag}.npz"
        ck.save(m, path, Re=np.array(Re), dt=np.array(dt), xr=np.array(xr),
                div_interior=np.array(divi), steps=np.array(it + 1),
                # plain ints, NOT an object array: checkpoint.load_fields reads with
                # allow_pickle=False on purpose, and an object array cannot be loaded there.
                block_shapes=np.array([blk.shape for blk in d.blocks], dtype=np.int64))
        for b in range(len(d.blocks)):
            np.savez(f"results/fields/{tag}_geom{b}.npz",
                     x=d.blocks[b].x, y=d.blocks[b].y, z=d.blocks[b].z)
        if report:
            print(f"      saved {path}", flush=True)
    if report:
        print(f"   Re={Re:6.0f}  {'dong' if dong else 'conv':>4}  rc={str(rhie_chow):<5} "
              f"{it + 1:5d} steps  divF(interior) {divi:.1e}  x_r/S = {xr:6.3f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
    return xr, None, m, d


if __name__ == "__main__":
    print(__doc__.strip().split("\n\n")[0])
    print("\n  Dong outflow, five domains. Reattachment reference: x_r/S ~ 3 / 5 / 6-7\n")
    med = dict(nx_in=20, nx_re=40, nx_rc=40, ny_lo=18, ny_up=19, nz=8)
    out = {}
    for Re in (100.0, 200.0, 300.0):
        xr, err, m, d = run(Re=Re, dt=0.02, nsteps=3000, dong=True, grid=med)
        if err:
            print(f"   Re={Re:.0f}: {err}")
        out[Re] = xr
    ok = all(np.isfinite(v) for v in out.values()) and \
        out[100.0] < out[200.0] < out[300.0]
    print(f"\n  [{'PASS' if ok else 'FAIL'}] recirculation lengthens with Re: " +
          ", ".join(f"Re={k:.0f} -> {v:.2f}S" for k, v in out.items()))
