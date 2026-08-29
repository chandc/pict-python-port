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


def nyquist_amp(F, axis=1):
    """TRUE amplitude of the node-to-node (-1)^j mode: the exact Fourier coefficient at
    the Nyquist wavenumber, averaged over profiles.

    odd_even_abs uses a 1-2-1 smoother, whose deviation from a SMOOTH curved profile is
    ~(h^2/4) f'' -- real curvature, not oscillation. That is tolerable for pressure, where
    the checkerboard dominates, and badly misleading for velocity, where a parabolic channel
    profile's curvature swamps any actual mode. This projector sees only the alternating
    component and is blind to smooth structure of any order.
    """
    G = np.moveaxis(F, axis, 0)
    n = G.shape[0]
    sign = (-1.0) ** np.arange(n)
    amp = np.abs(np.tensordot(sign, G, axes=(0, 0))) / n
    return float(np.mean(amp))


def checkerboard(F, axis=1):
    """(amplitude, flip_fraction) -- the honest pair. Neither number is safe alone.

    flip_fraction is the DETECTOR: the fraction of consecutive node pairs where the slope
    reverses. A pure node-to-node mode flips at every pair (1.00); a smooth profile almost
    never does. amplitude is the 1-2-1 deviation, which is only a checkerboard magnitude
    when the flip fraction is high -- on a smooth curved profile it reports ~(h^2/4) f''.

    Both single-number metrics tried before this were wrong in opposite directions. The
    1-2-1 deviation alone called a parabolic velocity profile's curvature a 1.9e-02
    oscillation (7% flips). A global (-1)^j Fourier projection alone called a genuine
    checkerboard 1.6e-16 (88% flips) because the mode's envelope is antisymmetric about the
    channel centreline, so the sum cancels.
    """
    G = np.moveaxis(F, axis, 0)
    sm = 0.25 * G[:-2] + 0.5 * G[1:-1] + 0.25 * G[2:]
    amp = float(np.abs(G[1:-1] - sm).mean())
    d = np.diff(G, axis=0); sg = np.sign(d)
    ch = (sg[1:] != sg[:-1]) & (sg[1:] != 0) & (sg[:-1] != 0)
    return amp, float(ch.mean())


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
