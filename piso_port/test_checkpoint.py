"""
Checkpoint / restart gates.

The claim under test is EXACTNESS: a run split by a save/restore must reproduce, to round-off,
the run that was never interrupted. Anything weaker is not a restart, it is a warm start.

Three of these are negative controls. A restart test passes trivially if the discarded state
did not matter, so each piece of saved state is also dropped deliberately to confirm the test
can actually see its absence. Without those, "restart works" would be unfalsifiable.
"""
import os
import sys
import tempfile
import numpy as np

from src.piso_numpy_3d import PISOSolver
from src.piso_multiblock import MultiBlockPISO
from test_multiblock import strip
from src import checkpoint as ck

PASS = []


def check(cond, msg):
    PASS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")


def force(s):
    """A steady body force, so the flow actually evolves and BDF2 history is non-trivial."""
    s.velocity_source = (1.0, 0.0, 0.0)      # both solvers index this by component


def single(**kw):
    s = PISOSolver(8, warp=0.05, nu=0.05, dt=0.02, periodic=(True, False, True),
                   scheme='rotational', time_scheme='bdf2', convection='central',
                   picard_iters=2, **kw)
    force(s)
    return s


def single_rc(**kw):
    """Rhie-Chow on: adds p_flux and F_prev to the running state."""
    return single(rhie_chow=True, persistent_flux=True, ddt_corr=True, **kw)


def multi_rc(nb=2, **kw):
    """Multi-block with Rhie-Chow: p_flux and F_prev become per-block running state."""
    return multi(nb, rhie_chow=True, persistent_flux=True, ddt_corr=True, **kw)


def multi(nb=2, **kw):
    s = MultiBlockPISO(strip(nb), 0.05, 0.02, 2, 1e-12, time_scheme='bdf2',
                       scheme='rotational', picard_iters=2, **kw)
    force(s)
    return s


def state(s):
    """Every evolving field, flattened, for a single scalar comparison."""
    if isinstance(s.u, dict):
        return np.concatenate([np.asarray(v).ravel() for f in ("u", "v", "w", "p")
                               for v in getattr(s, f).values()])
    return np.concatenate([getattr(s, f).ravel() for f in ("u", "v", "w", "p")])


def split_run(make, nsteps=12, cut=6, mangle=None):
    """Run straight through, then run again with a save/restore at `cut`; compare."""
    a = make()
    for _ in range(nsteps):
        a.step()
    ref = state(a)

    b = make()
    for _ in range(cut):
        b.step()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "chk.npz")
        ck.save(b, path, note=np.array(1.0))
        c = make()
        meta = ck.load(c, path)
        if mangle:
            mangle(c)
        for _ in range(nsteps - cut):
            c.step()
    return np.abs(state(c) - ref).max(), meta, c


if __name__ == "__main__":
    print(__doc__.strip().split("\n\n")[0])

    print("\nExact restart (a save/restore must be invisible in the answer)")
    for name, make in (("single block", single),
                       ("single block, rhie_chow", single_rc),
                       ("multi-block, 2 domains", multi),
                       ("multi-block, 4 domains", lambda: multi(4)),
                       ("multi-block, rhie_chow", multi_rc)):
        err, meta, c = split_run(make)
        check(err < 1e-12, f"{name}: restart reproduces the uninterrupted run "
                           f"(max|diff| = {err:.2e})")
        check(c.nstep == 12 and abs(c.time - 12 * 0.02) < 1e-12,
              f"{name}: clock survives the restart (nstep {c.nstep}, t {c.time:.3f})")

    print("\nNegative controls (the test must be able to SEE state going missing)")
    err_np, _, _ = split_run(single, mangle=lambda s: setattr(s, "u_prev", None))
    check(err_np > 1e-8, f"dropping u_prev demotes the restart step to backward Euler and "
                         f"the test detects it (max|diff| = {err_np:.2e})")
    err_p, _, _ = split_run(single, mangle=lambda s: s.p.fill(0.0))
    check(err_p > 1e-8, f"dropping p perturbs the run and the test detects it "
                        f"(max|diff| = {err_p:.2e})")
    err_mb, _, _ = split_run(multi, mangle=lambda s: setattr(s, "u_prev", None))
    check(err_mb > 1e-8, f"multi-block: dropping u_prev is detected too "
                         f"(max|diff| = {err_mb:.2e})")
    err_fp, _, _ = split_run(single_rc, mangle=lambda s: setattr(s, "F_prev", None))
    check(err_fp > 1e-12, f"rhie_chow: dropping F_prev loses a step of ddt_corr damping "
                          f"and the test detects it (max|diff| = {err_fp:.2e})")
    # A correct restart is bitwise exact (0.00e+00), so ANY nonzero difference proves the
    # dropped field mattered. The margin is small here only because the strip is a forced
    # periodic channel whose exact pressure is uniform -- the spurious pressure the
    # transient term suppresses is ~1e-11, so there is little left for it to change. That
    # makes this a weak probe, not a failing one; the single-block control above carries
    # the strong signal (8.15e-03).
    err_mbf, _, _ = split_run(multi_rc, mangle=lambda s: setattr(s, "F_prev", None))
    check(err_mbf > 1e-15, f"multi-block rhie_chow: dropping F_prev is detected "
                           f"(max|diff| = {err_mbf:.2e}, vs 0.00e+00 for an exact restart)")
    err_pf, _, _ = split_run(single_rc, mangle=lambda s: s.p_flux.fill(0.0))
    check(err_pf > 1e-12, f"rhie_chow: dropping p_flux is detected "
                          f"(max|diff| = {err_pf:.2e})")

    print("\nRefusing a restart that is not a continuation")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "c.npz")
        s = single(); s.step(); ck.save(s, path)

        try:
            bad = PISOSolver(8, warp=0.05, nu=0.10, dt=0.02, periodic=(True, False, True),
                             scheme='rotational', time_scheme='bdf2', convection='central',
                             picard_iters=2)
            ck.load(bad, path)
            check(False, "a different nu is refused")
        except ValueError as e:
            check("nu" in str(e), f"a different nu is refused: {str(e)[:58]}...")

        try:
            ck.load(multi(), path)
            check(False, "loading a single-block file into a multi-block solver is refused")
        except ValueError as e:
            check("multi-block" in str(e),
                  "loading a single-block file into a multi-block solver is refused")

        try:
            ck.load(PISOSolver(12, warp=0.05, nu=0.05, dt=0.02,
                               periodic=(True, False, True), scheme='rotational',
                               time_scheme='bdf2', convection='central', picard_iters=2),
                    path)
            check(False, "a grid-size mismatch is refused")
        except ValueError as e:
            check("checkpoint" in str(e), f"a grid-size mismatch is refused: {str(e)[:52]}...")

        bad = single(); bad.nu = 0.10
        meta = ck.load(bad, path, strict=False)
        check(meta["nstep"] == 1, "strict=False allows a deliberate config change")

    print("\nPost-processing: reading a checkpoint without building a solver")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m.npz")
        s = multi()
        for _ in range(3):
            s.step()
        ck.save(s, path, reynolds=np.array(100.0))
        f, meta = ck.load_fields(path)
        check(set(f) == {"u", "v", "w", "p"} and len(f["u"]) == 2 and meta["nstep"] == 3,
              f"load_fields returns all four fields for both blocks (nstep {meta['nstep']})")
        check(float(meta["extra"]["reynolds"]) == 100.0,
              "user metadata round-trips for post-processing")
        check(np.abs(f["u"][0] - s.u[0]).max() == 0.0,
              "fields read back bit-identical to the solver state")

    n = len(PASS)
    print("\n" + "=" * 70)
    print(f"  {sum(PASS)}/{n} checks passed")
    print("=" * 70)
    sys.exit(0 if all(PASS) else 1)
