"""
Locate the source of the pressure checkerboard, by measuring which term carries it.

Hypothesis: the Poisson operator uses COMPACT face differences (p[i+1]-p[i]), which admit no
checkerboard, but the momentum predictor subtracts a WIDE gradient (np.gradient, i.e.
(p[i+1]-p[i-1])/2h) and the face flux is a PLAIN average of cell velocities with no Rhie-Chow
pressure term. So an alternating mode in p exerts no force and moves no mass: nothing excites
it deliberately, and nothing damps it either.

That predicts a sharp split by scheme:
  * chorin sets p = phi each step, straight from the compact solve -> CLEAN
  * incremental/rotational ACCUMULATE p += phi -> the projection must absorb, every step, the
    discrepancy between the pressure the predictor actually felt and the p it is credited with.
    The mode is regenerated in phi and accumulates in p.

The cavity is deliberately NOT the probe case: its lid corners carry a genuine pressure
singularity, so a node-to-node metric there mixes real physics with the artefact and reads
~29% even for chorin. The smooth warped duct isolates the artefact.
"""
import sys
import numpy as np
from src.piso_numpy_3d import PISOSolver
from src.piso_multiblock import MultiBlockPISO
from armaly_bfs_grid import bfs_domain, H_IN
from src.multiblock import face_id

SCHEMES = ("chorin", "incremental", "rotational")


def odd_even_abs(F):
    """(absolute alternating amplitude, mean profile range, fraction of profiles skipped).

    Reported ABSOLUTELY, not just as a ratio. A ratio is meaningless when the field is flat,
    and reporting one anyway is how a force-driven straight duct -- whose pressure is uniform
    to 2e-15, so every profile is skipped -- was once read as "0% checkerboard, therefore
    clean" rather than "nothing here to measure".
    """
    tot, odd = [], []
    for i in range(F.shape[0]):
        f = F[i, :]
        if np.ptp(f) < 1e-12:
            continue
        sm = np.convolve(f, [0.25, 0.5, 0.25], mode="same")[1:-1]
        tot.append(np.ptp(f)); odd.append(np.abs(f[1:-1] - sm).max())
    if not tot:
        return 0.0, 0.0, 1.0
    return np.mean(odd), np.mean(tot), 1.0 - len(tot) / F.shape[0]


def odd_even(F):
    """Percentage form; NaN when there is no field to normalise against."""
    o, t, _ = odd_even_abs(F)
    return 100.0 * o / t if t > 0 else float("nan")


def single_duct(scheme, n=16, nsteps=400):
    """Force-driven warped duct: smooth solution, walls in y and z, no corner singularity."""
    s = PISOSolver(n, warp=0.05, nu=0.05, dt=0.01, scheme=scheme, time_scheme='be',
                   convection='central', periodic=(True, False, False))
    s.velocity_source = (1.0, 0.0, 0.0)
    for _ in range(nsteps):
        s.step()
    k = n // 2
    return odd_even(s.p[:, :, k]), odd_even(s.u[:, :, k])


def single_cavity(scheme, n=16, nsteps=300):
    """Kept only to show WHY it is the wrong probe: the lid corners are singular."""
    s = PISOSolver(n, warp=0.0, nu=0.02, dt=0.01, scheme=scheme, time_scheme='be',
                   convection='central', periodic=(False, False, True))
    s.u_bc[:, -1, :] = 1.0
    for _ in range(nsteps):
        s.step()
    k = n // 2
    return odd_even(s.p[:, :, k]), odd_even(s.u[:, :, k])


def multi_bfs(scheme, nsteps=400, Re=100.0):
    nu = 2.0 * H_IN / Re
    d, LOW, UP = bfs_domain(nx=40, ny_lo=10, ny_up=12, nz=4)
    m = MultiBlockPISO(d, nu, 0.02, 2, 1e-11, time_scheme="be", scheme=scheme,
                       picard_iters=1)
    up = d.blocks[UP]
    prof = 6.0 * (up.y[0, :, :] / H_IN) * (1.0 - up.y[0, :, :] / H_IN)
    m.u_bc[UP][0, :, :] = prof
    m.u[UP][:] = prof[None, :, :]
    for b, blk in ((UP, up), (LOW, d.blocks[LOW])):
        for f, kind in enumerate(blk.faces):
            if kind in ("periodic", "connected") or f == face_id(0, 1):
                continue
            if b == UP and f == face_id(0, 0):
                continue
            ax, sd = f // 2, f % 2
            sl = [slice(None)] * 3; sl[ax] = 0 if sd == 0 else -1
            m.u_bc[b][tuple(sl)] = 0.0; m.u[b][tuple(sl)] = 0.0
    m.outflow = [(UP, face_id(0, 1), 1.0, "convective"),
                 (LOW, face_id(0, 1), 1.0, "convective")]
    m.u_bc[UP][-1, :, :] = prof
    for _ in range(nsteps):
        m.step()
    k = 2
    P = np.concatenate([m.p[LOW], m.p[UP]], axis=1)[:, :, k]
    U = np.concatenate([m.u[LOW], m.u[UP]], axis=1)[:, :, k]
    PHI = np.concatenate([m._diag["phi"][LOW], m._diag["phi"][UP]], axis=1)[:, :, k]
    return odd_even(P), odd_even(U), odd_even(PHI)


if __name__ == "__main__":
    print(__doc__.strip().split("\n\n")[0])
    rows, res = [], {}
    print("\n  odd-even content (% of each field's own range)\n")
    print(f"  {'case':<34}{'scheme':<14}{'p':>9}{'u':>9}")
    for nm, fn in (("single block: warped duct", single_duct),
                   ("single block: lid cavity (singular)", single_cavity)):
        for sc in SCHEMES:
            pp, uu = fn(sc)
            res[(nm, sc)] = pp
            print(f"  {nm:<34}{sc:<14}{pp:8.1f}%{uu:8.1f}%")
        print()
    print(f"  {'multi-block: BFS, 2 domains':<34}{'scheme':<14}{'p':>9}{'u':>9}{'phi':>9}")
    for sc in SCHEMES:
        pp, uu, ph = multi_bfs(sc)
        res[("mb", sc)] = pp
        print(f"  {'':<34}{sc:<14}{pp:8.1f}%{uu:8.1f}%{ph:8.1f}%")

    duct = "single block: warped duct"
    ok_s = res[(duct, "rotational")] > 3 * max(res[(duct, "chorin")], 1e-9)
    ok_m = res[("mb", "rotational")] > 3 * max(res[("mb", "chorin")], 1e-9)
    print(f"\n  single block: chorin {res[(duct,'chorin')]:.1f}% -> rotational "
          f"{res[(duct,'rotational')]:.1f}%   [{'DEFECT PRESENT' if ok_s else 'clean'}]")
    print(f"  multi-block : chorin {res[('mb','chorin')]:.1f}% -> rotational "
          f"{res[('mb','rotational')]:.1f}%   [{'DEFECT PRESENT' if ok_m else 'clean'}]")
    sys.exit(0)
