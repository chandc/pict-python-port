"""
Rhie-Chow gates: oscillations in BOTH velocity and pressure, on Cartesian AND skew grids,
single-block and multi-block.

Two things this exists to cover that the development gates did not:

  * VELOCITY. Every earlier measurement looked at pressure only. The checkerboard was shown
    to feed back into u (filtering p changed u by ~1%), so a fix that cleaned up p while
    leaving u ragged would be a false result.

  * SKEW GRIDS IN MULTI-BLOCK. The multi-block gates all ran on `strip`, which is Cartesian --
    no cross terms at all. Since the Rhie-Chow term is orthogonal-only and the cross part of
    the pressure flux carries the same wide-stencil blind spot, skewness is exactly where the
    fix is expected to weaken. Measuring it on Cartesian grids only would have hidden that.

Amplitudes are reported ABSOLUTELY. A ratio is meaningless when the field is flat, and reading
one anyway is how a uniform-pressure duct once got recorded as "0% checkerboard, clean".
"""
import sys
import numpy as np

from src.piso_numpy_3d import PISOSolver
from src.piso_multiblock import MultiBlockPISO
from src.multiblock import Block, Connection, Domain, face_id
from src.phase1_grid_metrics import make_grid
from diag_checkerboard import checkerboard

PASS = []


def check(cond, msg):
    PASS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")


def oscillation(F):
    """(amplitude, flip fraction). The flip fraction says WHETHER it is a checkerboard;
    the amplitude is only meaningful when it is. See diag_checkerboard.checkerboard."""
    return checkerboard(F, axis=1)


def warped_blocks(n_split, ntot=16, ny=12, nz=4, warp=0.06):
    """A warped channel split in x: walls in y, periodic in z, connections in x.

    Unlike `strip` this has genuine non-orthogonality, so the cross terms are active and the
    orthogonal-only Rhie-Chow term is being asked to work where it is weakest.
    """
    per = (True, False, True)
    xs, ys, zs, hx, hy, hz = make_grid((ntot, ny, nz), warp=warp, periodic=per)
    nxb = ntot // n_split
    blks = []
    for b in range(n_split):
        sl = slice(b * nxb, (b + 1) * nxb)
        blk = Block((nxb, ny, nz), xs[sl], ys[sl], zs[sl], (hx, hy, hz))
        blk.faces[face_id(1, 0)] = blk.faces[face_id(1, 1)] = "wall"
        blk.faces[face_id(2, 0)] = blk.faces[face_id(2, 1)] = "periodic"
        blks.append(blk)
    cs = []
    for b in range(n_split):
        nb = (b + 1) % n_split
        # The block that wraps back to block 0 sits one PERIOD away in x. That includes the
        # n_split == 1 case, where the single block wraps to ITSELF -- guarding this with
        # `n_split > 1` (as the fully-periodic helper in test_multiblock does, which is only
        # ever called with 2 or 4) leaves the self-wrap with no shift and TANGLES the grid:
        # measured min(J) = -7.15. Same defect class as the BFS period bug.
        sh = (1.0, 0.0, 0.0) if nb == 0 else (0.0, 0.0, 0.0)
        cs.append(Connection(b, face_id(0, 1), nb, face_id(0, 0), shift=sh))
    dom = Domain(blks, cs)
    Jmin = min(J.min() for J, _ in (dom.block_metrics_cached(b) for b in range(n_split)))
    assert Jmin > 0, f"tangled grid at n_split={n_split}: min(J) = {Jmin:.3f}"
    return dom


def run_mb(dom, nb, rc, nsteps=150, nz=4):
    m = MultiBlockPISO(dom, 0.05, 0.01, 2, 1e-12, time_scheme='be', scheme='rotational',
                       picard_iters=1, rhie_chow=rc, persistent_flux=rc, ddt_corr=rc)
    m.velocity_source = (1.0, 0.0, 0.0)
    for _ in range(nsteps):
        div = m.step()
    P = np.concatenate([m.p[b] for b in range(nb)], axis=0)[:, :, nz // 2]
    U = np.concatenate([m.u[b] for b in range(nb)], axis=0)[:, :, nz // 2]
    fin = all(np.isfinite(m.p[b]).all() and np.isfinite(m.u[b]).all() for b in range(nb))
    return oscillation(P), oscillation(U), div, fin, np.abs(U).max()


def run_sb(warp, rc, n=16, nsteps=150):
    s = PISOSolver(n, warp=warp, nu=0.05, dt=0.01, scheme='rotational', time_scheme='be',
                   convection='central', periodic=(True, False, True),
                   rhie_chow=rc, persistent_flux=rc, ddt_corr=rc)
    s.velocity_source = (1.0, 0.0, 0.0)
    for _ in range(nsteps):
        div = s.step()
    k = n // 2
    fin = np.isfinite(s.p).all() and np.isfinite(s.u).all()
    return oscillation(s.p[:, :, k]), oscillation(s.u[:, :, k]), div, fin, np.abs(s.u).max()


if __name__ == "__main__":
    print(__doc__.strip().split("\n\n")[0])

    print("\nSINGLE BLOCK -- oscillation amplitude in p and u, Cartesian vs skew\n")
    print(f"  {'grid':>12}{'rhie_chow':>11}{'amp(p)':>11}{'flip':>6}"
          f"{'amp(u)':>11}{'flip':>6}{'divF':>10}")
    for label, warp in (("cartesian", 1e-9), ("skew A=0.06", 0.06)):
        vals = {}
        for rc in (False, True):
            op, ou, dv, fin, um = run_sb(warp, rc)
            vals[rc] = (op[0], ou[0])
            print(f"  {label:>12}{str(rc):>11}{op[0]:11.3e}{op[1]:6.2f}"
                  f"{ou[0]:11.3e}{ou[1]:6.2f}{dv:10.1e}")
            check(fin, f"{label}, rhie_chow={rc}: stayed finite")
        for i, nm in ((0, "pressure"), (1, "velocity")):
            check(vals[True][i] <= vals[False][i] * 1.05,
                  f"{label}: {nm} oscillation not made worse "
                  f"({vals[False][i]:.2e} -> {vals[True][i]:.2e})")

    print("\nMULTI-BLOCK ON A SKEW GRID -- the case the earlier gates never covered\n")
    print(f"  {'blocks':>7}{'rhie_chow':>11}{'amp(p)':>11}{'flip':>6}"
          f"{'amp(u)':>11}{'flip':>6}{'divF':>10}")
    ref = {}
    for nb in (1, 2, 4):
        dom = warped_blocks(nb)
        vals = {}
        for rc in (False, True):
            op, ou, dv, fin, um = run_mb(dom, nb, rc)
            vals[rc] = (op[0], ou[0])
            ref[(nb, rc)] = um
            print(f"  {nb:7d}{str(rc):>11}{op[0]:11.3e}{op[1]:6.2f}"
                  f"{ou[0]:11.3e}{ou[1]:6.2f}{dv:10.1e}")
            check(fin, f"skew, {nb} block(s), rhie_chow={rc}: stayed finite")
            check(dv < 1e-10, f"skew, {nb} block(s), rhie_chow={rc}: flux divergence "
                              f"{dv:.1e} < 1e-10")
        for i, nm in ((0, "pressure"), (1, "velocity")):
            check(vals[True][i] <= vals[False][i] * 1.05,
                  f"skew, {nb} block(s): {nm} oscillation not made worse "
                  f"({vals[False][i]:.2e} -> {vals[True][i]:.2e})")

    print("\n  block-count independence on the SKEW grid, with the fix on:")
    for nb in (2, 4):
        d = abs(ref[(nb, True)] - ref[(1, True)])
        check(d < 1e-6, f"    {nb} blocks vs 1: |u|max differs by {d:.2e}")

    n = len(PASS)
    print("\n" + "=" * 74)
    print(f"  {sum(PASS)}/{n} checks passed")
    print("=" * 74)
    sys.exit(0 if all(PASS) else 1)
