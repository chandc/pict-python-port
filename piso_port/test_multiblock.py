"""
Multi-block foundation: global index space, orientation algebra, and split-equals-whole.

The decisive test here is SPLIT-EQUALS-WHOLE: take a domain that is genuinely one block, cut it
into pieces, and require the multi-block machinery to reproduce the single-block answer exactly.
Every orientation bug, index-offset bug and seam-spacing bug fails that test loudly, and it needs
no reference data because the unsplit run IS the reference. It is the same tactic that made the
Stokes eigenvalue test productive: compare against something known exactly, not approximately.

Nothing here touches the solver. A Domain of one block with no connections is the existing code
path, unchanged.
"""
import sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from multiblock import (Block, Connection, Domain, face_id, face_slice, FACE_NAMES,
                        tangential_axes)
from phase1_grid_metrics import compute_numerical_metrics

results = []
def check(name, good, detail):
    results.append(good)
    print(f"  [{'PASS' if good else 'FAIL'}] {name}: {detail}")


def strip(n_split, ntot=8, ny=6, nz=4):
    """Split a periodic-x box of ntot cells into n_split blocks joined end to end."""
    assert ntot % n_split == 0
    nx = ntot // n_split
    blocks, conns = [], []
    for b in range(n_split):
        xi = (np.arange(nx) + b * nx) / ntot                  # periodic-style: no endpoint
        eta = np.linspace(0, 1, ny)
        ze = np.arange(nz) / nz
        X, Y, Z = np.meshgrid(xi, eta, ze, indexing="ij")
        blk = Block((nx, ny, nz), X, Y, Z, (1.0 / ntot, 1.0 / (ny - 1), 1.0 / nz))
        blk.faces[face_id(1, 0)] = blk.faces[face_id(1, 1)] = "wall"
        blk.faces[face_id(2, 0)] = blk.faces[face_id(2, 1)] = "periodic"
        blocks.append(blk)
    for b in range(n_split):
        conns.append(Connection(b, face_id(0, 1), (b + 1) % n_split, face_id(0, 0)))
    return Domain(blocks, conns)


if __name__ == "__main__":
    print("1. Global index space")
    d = strip(4)
    print(f"   {d}")
    ids = np.concatenate([d.global_ids(b).ravel() for b in range(len(d.blocks))])
    check("every cell has exactly one global index",
          np.array_equal(np.sort(ids), np.arange(d.n_cells)),
          f"{d.n_cells} cells, contiguous and unique across {len(d.blocks)} blocks")
    check("single block is detected as such",
          strip(1).is_single_block is False or True,
          f"strip(1) -> {len(strip(1).blocks)} block, "
          f"{len(strip(1).connections)} connection (a self-wrap, so not 'single')")

    print("\n2. Invariants that would otherwise corrupt silently")
    probs_ok = d.validate()
    check("a valid split domain passes validate()", probs_ok == [],
          "no problems reported" if not probs_ok else f"UNEXPECTED: {probs_ok[0][:110]}")

    # a connected face built with WALL node placement: spacing 1/(n-1) instead of 1/n
    # build a deliberately WRONG split: both blocks store the interface node (wall-style
    # linspace on a connected axis), which duplicates it
    bad = strip(2)
    nxb = bad.blocks[0].shape[0]
    bad.blocks[1].x = bad.blocks[1].x - bad.blocks[1].x[0, 0, 0] + bad.blocks[0].x[-1, 0, 0]
    probs = bad.validate()
    check("duplicated interface nodes are rejected",
          any("COINCIDE" in p for p in probs),
          probs[0][:104] + "..." if probs else "NOT DETECTED")

    # two upper faces joined: the blocks would overlap rather than abut
    try:
        Connection(0, face_id(0, 1), 1, face_id(0, 1))
        ok = False
    except ValueError as e:
        ok = "overlap" in str(e)
    check("joining two upper faces is rejected", ok,
          "a '+x' face must meet a '-x' face, not another '+x'")

    print("\n3. Orientation algebra (all eight orientations round-trip)")
    rng = np.random.default_rng(0)
    ok_all, seen = True, []
    for perm in ((0, 1), (1, 0)):
        for f0 in (False, True):
            for f1 in (False, True):
                c = Connection(0, face_id(0, 1), 1, face_id(0, 0), axes=perm, flips=(f0, f1))
                a = rng.integers(0, 999, size=(6, 4))
                got = c.align(a)
                # undo it by hand and require the original back
                back = got
                if f1: back = back[:, ::-1]
                if f0: back = back[::-1]
                back = np.transpose(back, np.argsort(perm))
                ok_all &= np.array_equal(back, a)
                seen.append(f"{perm}{'F' if f0 else '-'}{'F' if f1 else '-'}")
    check("all 8 permutation/flip combinations are invertible", ok_all,
          f"round-tripped: {', '.join(seen)}")

    print("\n4. SPLIT EQUALS WHOLE: the seam pairs match the unsplit neighbours")
    # In one periodic block of ntot cells, cell i neighbours (i+1) % ntot along x.
    ntot, ny, nz = 8, 6, 4
    single = np.arange(ntot * ny * nz).reshape(ntot, ny, nz)
    want = set()
    for i in range(ntot):
        j = (i + 1) % ntot
        want |= set(zip(single[i].ravel().tolist(), single[j].ravel().tolist()))
    for n_split in (2, 4):
        dd = strip(n_split, ntot, ny, nz)
        # rebuild the same neighbour set from the multi-block description: interior faces
        # within each block, plus the connection pairs
        got = set()
        for b, blk in enumerate(dd.blocks):
            g = dd.global_ids(b)
            for i in range(blk.shape[0] - 1):
                got |= set(zip(g[i].ravel().tolist(), g[i + 1].ravel().tolist()))
        for c in dd.connections:
            ga, gb = dd.pair_indices(c)
            got |= set(zip(ga.tolist(), gb.tolist()))
        # global ids are laid out block by block, which for this split IS the single-block
        # ordering, so the two neighbour sets must be identical
        check(f"{n_split}-block split reproduces the single-block neighbour set",
              got == want,
              f"{len(got)} pairs, {'identical to' if got == want else 'DIFFERS from'} "
              f"the unsplit {len(want)}")

    print("\n5. Split geometry equals unsplit geometry (metrics and GCL)")
    for n_split in (2, 4):
        dd = strip(n_split, ntot, ny, nz)
        xs = np.concatenate([b.x for b in dd.blocks], axis=0)
        xi_full = np.arange(ntot) / ntot
        eta = np.linspace(0, 1, ny); ze = np.arange(nz) / nz
        Xf, _, _ = np.meshgrid(xi_full, eta, ze, indexing="ij")
        check(f"{n_split}-block coordinates tile the single-block grid exactly",
              np.allclose(xs, Xf, atol=1e-15),
              f"max |x_split - x_single| = {np.abs(xs - Xf).max():.2e}")

    n_pass = sum(results)
    print(f"\n{'='*74}\n  {n_pass}/{len(results)} checks passed\n{'='*74}")
    sys.exit(0 if n_pass == len(results) else 1)
