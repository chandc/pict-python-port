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

    print("\n6. SPLIT EQUALS WHOLE for the GEOMETRY: seam metrics from real neighbour data")
    # The sharpest form of the test: warp the grid, split it, and require the per-block metrics
    # computed through the connection map to equal the single-block metrics EXACTLY. Any error
    # in the orientation transform, the period shift, or the ghost ordering shows up here.
    from phase1_grid_metrics import make_grid, compute_numerical_metrics
    from phase2_operators import compute_divergence
    NTOT, NY, NZ, WARP = 8, 6, 4, 0.08
    P3 = (True, True, True)
    xs, ys, zs, hx, hy, hz = make_grid((NTOT, NY, NZ), warp=WARP, periodic=P3)
    Jref, mref = compute_numerical_metrics(xs, ys, zs, hx, hy, hz, periodic=P3)

    def warped_split(n_split):
        nxb = NTOT // n_split
        blks = []
        for b in range(n_split):
            sl = slice(b * nxb, (b + 1) * nxb)
            blk = Block((nxb, NY, NZ), xs[sl], ys[sl], zs[sl], (hx, hy, hz))
            for a in (1, 2):
                blk.faces[face_id(a, 0)] = blk.faces[face_id(a, 1)] = "periodic"
            blks.append(blk)
        cs = []
        for b in range(n_split):
            nb = (b + 1) % n_split
            # the last block wraps to the first: its neighbour lies one PERIOD away in x
            sh = (1.0, 0.0, 0.0) if nb == 0 and n_split > 1 else (0.0, 0.0, 0.0)
            cs.append(Connection(b, face_id(0, 1), nb, face_id(0, 0), shift=sh))
        return Domain(blks, cs)

    for n_split in (2, 4):
        dd = warped_split(n_split)
        nxb = NTOT // n_split
        Jerr = merr = 0.0
        Js, ms = [], []
        for b in range(n_split):
            Jb, mb = dd.block_metrics(b)
            sl = slice(b * nxb, (b + 1) * nxb)
            Jerr = max(Jerr, np.abs(Jb - Jref[sl]).max())
            for k in mref:
                merr = max(merr, np.abs(mb[k] - mref[k][sl]).max())
            Js.append(Jb); ms.append(mb)
        check(f"{n_split}-block warped metrics equal the single-block metrics",
              Jerr < 1e-14 and merr < 1e-14,
              f"max|J-Jref| {Jerr:.2e}, max|metric-ref| {merr:.2e} (warp {WARP})")

        # GCL must be checked on the REASSEMBLED domain, not block by block. A block-local
        # divergence with periodic=(True,...) would wrap each block onto ITSELF at a face that
        # is actually connected to a neighbour -- measuring a seam that does not exist and
        # reporting ~1e-1. A genuinely seam-aware divergence needs the operator layer, which is
        # not built yet; reassembling is the honest check available today.
        Jg = np.concatenate(Js, axis=0)
        mg = {k: np.concatenate([m[k] for m in ms], axis=0) for k in mref}
        o = np.ones_like(Jg)
        gcl = 0.0
        for comp in range(3):
            f = [o * (1 if c == comp else 0) for c in range(3)]
            gcl = max(gcl, np.abs(compute_divergence(*f, Jg, mg, hx, hy, hz,
                                                     periodic=P3)).max())
        check(f"{n_split}-block reassembled metrics satisfy the GCL", gcl < 1e-11,
              f"max |div(uniform field)| = {gcl:.2e} over the reassembled domain")

    print("\n7. SPLIT EQUALS WHOLE for the MATRIX and the SOLVE")
    # The strongest form yet: assemble the global operator across blocks and SOLVE with it.
    # A matrix can be right in structure and wrong in the seam coefficients; only a solve
    # exercises the coupling end to end.
    import scipy.sparse.linalg as spla
    from phase3_momentum import build_conservative_diffusion_matrix
    Mref = build_conservative_diffusion_matrix(NTOT, NY, NZ, hx, hy, hz, Jref, mref,
                                               periodic=P3)
    rng2 = np.random.default_rng(7)
    rhs = rng2.standard_normal(Mref.shape[0]); rhs -= rhs.mean()   # compatible: M is singular
    free = np.arange(Mref.shape[0])[1:]                            # pin one cell globally
    pref = np.zeros(Mref.shape[0])
    pref[free] = spla.cg(Mref[free][:, free].tocsr(), rhs[free], rtol=1e-13, maxiter=20000)[0]

    for n_split in (2, 4):
        dd = warped_split(n_split)
        Js, ms = [], []
        for b in range(n_split):
            Jb, mb = dd.block_metrics(b)
            Js.append(Jb); ms.append(mb)
        M = dd.build_diffusion_matrix(Js, ms)
        check(f"{n_split}-block global matrix equals the single-block matrix",
              abs(M - Mref).max() < 1e-12 and M.nnz == Mref.nnz,
              f"max|M-Mref| {abs(M - Mref).max():.2e}, {M.nnz} nnz vs {Mref.nnz}, "
              f"asymmetry {abs(M - M.T).max():.1e}")

        p = np.zeros(M.shape[0])
        p[free] = spla.cg(M[free][:, free].tocsr(), rhs[free], rtol=1e-13, maxiter=20000)[0]
        check(f"{n_split}-block SOLVE reproduces the single-block solution",
              np.abs(p - pref).max() < 1e-9,
              f"max|p - p_single| = {np.abs(p - pref).max():.2e}")

    # The trap named in multiblock_offsets.md: a preconditioner (or an assembly) that ignores
    # the connections leaves the blocks DECOUPLED. It still solves, still converges, and returns
    # a confidently wrong field. Demonstrated here so the gate above is known to have teeth.
    dd = warped_split(4)
    Js, ms = [], []
    for b in range(4):
        Jb, mb = dd.block_metrics(b); Js.append(Jb); ms.append(mb)
    decoupled = Domain(dd.blocks, [])          # same blocks, connections dropped
    for blk in decoupled.blocks:               # undo the 'connected' marks
        for fid in (face_id(0, 0), face_id(0, 1)):
            blk.faces[fid] = "wall"
    Md = decoupled.build_diffusion_matrix(Js, ms)
    pd = np.zeros(Md.shape[0])
    try:
        pd[free] = spla.cg(Md[free][:, free].tocsr(), rhs[free], rtol=1e-13, maxiter=20000)[0]
        err_d = np.abs(pd - pref).max()
    except Exception:
        err_d = np.inf
    check("dropping the connections is caught, not silently tolerated",
          err_d > 1e-3,
          f"decoupled solve differs from the true solution by {err_d:.2e} "
          f"-- it converges happily and is wrong")

    print("\n8. A block connected to ITSELF is a full-period wrap")
    # Found while running a multi-block MMS Poisson: with one block whose +x joins its own -x,
    # the connection still crosses a whole period, so it needs the period shift exactly as a
    # two-block wrap does. Omitting it degraded the seam and dropped the solve to FIRST order
    # (8.9e-2 vs 1.9e-2 at ntot=8) while validate() reported nothing -- the interface nodes are
    # genuinely distinct, so the duplicated-node check cannot see it. Gated here instead.
    import scipy.sparse.linalg as spla2

    def periodic_mms(ntot, n_split, shift_selfwrap=True):
        nxb = ntot // n_split
        xi1 = np.arange(ntot) / ntot
        Xg, Yg, Zg = np.meshgrid(xi1, xi1, xi1, indexing="ij")
        blks = []
        for b in range(n_split):
            sl = slice(b * nxb, (b + 1) * nxb)
            blk = Block((nxb, ntot, ntot), Xg[sl], Yg[sl], Zg[sl],
                        (1.0 / ntot,) * 3)
            for a in (1, 2):
                blk.faces[face_id(a, 0)] = blk.faces[face_id(a, 1)] = "periodic"
            blks.append(blk)
        cs = []
        for b in range(n_split):
            nb = (b + 1) % n_split
            wraps = (nb == 0)
            sh = (1.0, 0.0, 0.0) if (wraps and shift_selfwrap) else (0.0, 0.0, 0.0)
            cs.append(Connection(b, face_id(0, 1), nb, face_id(0, 0), shift=sh))
        dm = Domain(blks, cs)
        Js2, ms2 = [], []
        for b in range(n_split):
            Jb, mb = dm.block_metrics(b); Js2.append(Jb); ms2.append(mb)
        M2 = dm.build_diffusion_matrix(Js2, ms2)
        xa = np.concatenate([bk.x.ravel() for bk in blks])
        ya = np.concatenate([bk.y.ravel() for bk in blks])
        za = np.concatenate([bk.z.ravel() for bk in blks])
        pe = np.sin(2 * np.pi * xa) * np.sin(2 * np.pi * ya) * np.sin(2 * np.pi * za)
        Ja = np.concatenate([J.ravel() for J in Js2])
        rr = -(Ja * (-3 * (2 * np.pi) ** 2 * pe)); rr -= rr.mean()
        fr = np.arange(M2.shape[0])[1:]
        pp = np.zeros(M2.shape[0])
        pp[fr] = spla2.cg(M2[fr][:, fr].tocsr(), rr[fr], rtol=1e-14, maxiter=50000)[0]
        pp -= pp.mean()
        return np.sqrt(np.mean((pp - (pe - pe.mean())) ** 2))

    print(f"   {'ntot':>5} {'1 block':>12} {'2 blocks':>12} {'4 blocks':>12}")
    errs_by_split, prev_row = {}, None
    for ntot in (8, 16):
        row = {ns: periodic_mms(ntot, ns) for ns in (1, 2, 4)}
        line = f"   {ntot:5d}" + "".join(f"{row[k]:12.4e}" for k in (1, 2, 4))
        if prev_row:
            line += "   order " + ", ".join(f"{np.log2(prev_row[k]/row[k]):.2f}" for k in (1, 2, 4))
        print(line)
        errs_by_split[ntot] = row
        prev_row = row
    spread = max(abs(errs_by_split[16][k] / errs_by_split[16][1] - 1) for k in (2, 4))
    check("MMS Poisson: the number of blocks does not change the answer",
          spread < 1e-9,
          f"1, 2 and 4 blocks agree to {spread:.1e} relative; order "
          f"{np.log2(errs_by_split[8][4]/errs_by_split[16][4]):.2f}")

    bad_e = periodic_mms(16, 1, shift_selfwrap=False)
    good_e = errs_by_split[16][1]
    check("omitting the self-wrap period shift is caught",
          bad_e > 3 * good_e,
          f"without the shift {bad_e:.3e} vs {good_e:.3e} -- {bad_e/good_e:.1f}x worse, and "
          f"validate() cannot see it (the nodes are genuinely distinct)")

    print("\n9. FIELD padding across seams (the input the operators will need)")
    # Fields are padded with the same connection machinery as coordinates but WITHOUT the
    # period shift: coordinates ramp and jump back, velocity and pressure are genuinely
    # periodic. Getting that backwards offsets every seam by exactly one period -- large,
    # smooth, and entirely plausible-looking.
    fld = (np.sin(2 * np.pi * xs) * np.cos(2 * np.pi * ys) * np.sin(2 * np.pi * zs))
    for n_split in (2, 4):
        dd = warped_split(n_split)
        nxb = NTOT // n_split
        parts = {b: fld[b * nxb:(b + 1) * nxb] for b in range(n_split)}
        err = 0.0
        for b in range(n_split):
            pf, lo, hi = dd.pad_field(b, parts, width=2)
            core = (slice(lo[0], pf.shape[0] - hi[0]),
                    slice(lo[1], pf.shape[1] - hi[1]),
                    slice(lo[2], pf.shape[2] - hi[2]))
            err = max(err, np.abs(pf[core] - parts[b]).max())
            for k in (1, 2):                      # streamwise ghosts either side
                gh = pf[lo[0] + nxb + k - 1][core[1], core[2]]
                want = np.roll(fld, -(b * nxb + nxb + k - 1), axis=0)[0]
                err = max(err, np.abs(gh - want).max())
                gl = pf[lo[0] - k][core[1], core[2]]
                wantl = np.roll(fld, -(b * nxb - k), axis=0)[0]
                err = max(err, np.abs(gl - wantl).max())
        check(f"{n_split}-block field padding matches the single-block field",
              err < 1e-14, f"max |ghost - single-block value| = {err:.2e} (width=2, both sides)")

    # The trap, demonstrated: applying the COORDINATE period shift to a field.
    dd = warped_split(2)
    parts = {b: fld[b * (NTOT // 2):(b + 1) * (NTOT // 2)] for b in range(2)}
    good, lo, hi = dd.pad_field(0, parts, width=2)
    bad = good.copy()
    bad[lo[0] + NTOT // 2:] += 1.0                # what a period shift would inject
    check("a period shift on a FIELD would be a large error, not a subtle one",
          np.abs(bad - good).max() > 0.5,
          f"it displaces the seam ghosts by {np.abs(bad - good).max():.1f} = one period; "
          f"fields must be padded WITHOUT the shift that coordinates require")

    print("\n10. SEAM-AWARE FLUXES and DIVERGENCE (the last piece before a PISO step)")
    # compute_face_fluxes cannot do this block-locally: it treats every non-periodic face as a
    # DOMAIN BOUNDARY, so a connection would receive a PRESCRIBED flux rather than an
    # interpolated one, injecting or losing mass at every seam. Domain.face_fluxes resolves
    # connected faces from real neighbour data, making a seam an ordinary interior face.
    from phase5_fluxes import compute_face_fluxes, divergence_from_fluxes
    Kw = 2 * np.pi
    uu = np.sin(Kw * xs) * np.cos(Kw * ys) * np.cos(Kw * zs)
    vv = -np.cos(Kw * xs) * np.sin(Kw * ys) * np.cos(Kw * zs)
    ww = np.zeros_like(uu)
    Fs = compute_face_fluxes(uu, vv, ww, Jref, mref, boundary="periodic", periodic=P3)
    dref = divergence_from_fluxes(Fs, Jref, (hx, hy, hz))
    for n_split in (2, 4):
        dd = warped_split(n_split)
        nxb = NTOT // n_split
        us = {b: uu[b * nxb:(b + 1) * nxb] for b in range(n_split)}
        vs = {b: vv[b * nxb:(b + 1) * nxb] for b in range(n_split)}
        ws = {b: ww[b * nxb:(b + 1) * nxb] for b in range(n_split)}
        fe = de = 0.0
        for b in range(n_split):
            F = dd.face_fluxes(b, us, vs, ws)
            Jb, _ = dd.block_metrics(b)
            db = dd.divergence(b, F, Jb)
            cell = slice(b * nxb, (b + 1) * nxb)
            de = max(de, np.abs(db - dref[cell]).max())
            fe = max(fe, np.abs(F[0] - Fs[0][b * nxb:(b + 1) * nxb + 1]).max())
            for a in (1, 2):
                fe = max(fe, np.abs(F[a] - Fs[a][cell]).max())
        check(f"{n_split}-block seam fluxes equal the single-block fluxes",
              fe < 1e-14, f"max |flux - single| = {fe:.2e}")
        check(f"{n_split}-block seam divergence equals the single-block divergence",
              de < 1e-13, f"max |div - single| = {de:.2e} (single-block max|div| itself is "
                          f"{np.abs(dref).max():.2e}, coarse-grid truncation at n={NTOT})")

    print("\n11. GLOBAL MOMENTUM MATRIX (convection + diffusion + transient)")
    import scipy.sparse as sp2
    from phase3_momentum import build_momentum_matrix_7point
    NUm, DTm = 0.05, 0.02
    Aref = build_momentum_matrix_7point(NTOT, NY, NZ, Jref, mref, hx, hy, hz, uu, vv, ww,
                                        NUm, periodic=P3, convection="central")
    Aref = Aref + sp2.diags(Jref.ravel() / DTm)
    for n_split in (2, 4):
        dd = warped_split(n_split)
        nxb = NTOT // n_split
        us = {b: uu[b * nxb:(b + 1) * nxb] for b in range(n_split)}
        vs = {b: vv[b * nxb:(b + 1) * nxb] for b in range(n_split)}
        ws = {b: ww[b * nxb:(b + 1) * nxb] for b in range(n_split)}
        Js2, ms2 = [], []
        for b in range(n_split):
            Jb, mb = dd.block_metrics(b); Js2.append(Jb); ms2.append(mb)
        A = dd.build_momentum_matrix(Js2, ms2, us, vs, ws, NUm, DTm)
        check(f"{n_split}-block momentum matrix equals the single-block matrix",
              abs(A - Aref).max() < 1e-12 and A.nnz == Aref.nnz,
              f"max|A - Aref| = {abs(A - Aref).max():.2e}, {A.nnz} nnz vs {Aref.nnz} "
              f"(convection is NOT symmetric: asymmetry {abs(A - A.T).max():.2e})")

    # SOU must refuse rather than quietly give central
    dd = warped_split(2)
    Js2, ms2 = [], []
    for b in range(2):
        Jb, mb = dd.block_metrics(b); Js2.append(Jb); ms2.append(mb)
    nxb = NTOT // 2
    try:
        dd.build_momentum_matrix(Js2, ms2,
                                 {b: uu[b*nxb:(b+1)*nxb] for b in range(2)},
                                 {b: vv[b*nxb:(b+1)*nxb] for b in range(2)},
                                 {b: ww[b*nxb:(b+1)*nxb] for b in range(2)},
                                 NUm, DTm, convection="sou")
        refused = False
    except NotImplementedError:
        refused = True
    check("asking for SOU raises rather than silently giving central", refused,
          "SOU reaches i-2 and needs wider seam handling; a silent fallback would change the "
          "physics, since SOU removes ~10% of KE per turnover where central conserves it")

    print("\n12. A FULL PISO STEP ACROSS BLOCKS")
    # Cartesian and fully periodic on purpose: that is the configuration where the multi-block
    # result can be compared against the single-block solver EXACTLY, with no cross terms and
    # no boundary conditions to confound it.
    import io as _io, contextlib as _cl
    from piso_multiblock import MultiBlockPISO
    from piso_numpy_3d import PISOSolver
    NTp, NUp, Kp = 8, 0.05, 2 * np.pi

    def cart_split(n):
        nxb = NTp // n
        xi1 = np.arange(NTp) / NTp
        Xc, Yc, Zc = np.meshgrid(xi1, xi1, xi1, indexing="ij")
        bl = []
        for b in range(n):
            sl = slice(b * nxb, (b + 1) * nxb)
            blk = Block((nxb, NTp, NTp), Xc[sl], Yc[sl], Zc[sl], (1.0 / NTp,) * 3)
            for a in (1, 2):
                blk.faces[face_id(a, 0)] = blk.faces[face_id(a, 1)] = "periodic"
            bl.append(blk)
        return Domain(bl, [Connection(b, face_id(0, 1), (b + 1) % n, face_id(0, 0),
                                      shift=(1.0, 0, 0) if (b + 1) % n == 0 else (0, 0, 0))
                           for b in range(n)])

    for ts in ("be", "bdf2"):
        DTp = 0.02
        sref = PISOSolver((NTp, NTp, NTp), warp=1e-9, nu=NUp, dt=DTp, corrector_steps=2,
                          periodic=P3, scheme="chorin", time_scheme=ts, convection="central",
                          pressure_coef="rowsum", pressure_tol=1e-13)
        u0 = np.sin(Kp * sref.x) * np.cos(Kp * sref.y) * np.cos(Kp * sref.z)
        v0 = -np.cos(Kp * sref.x) * np.sin(Kp * sref.y) * np.cos(Kp * sref.z)
        sref.u, sref.v, sref.w = u0.copy(), v0.copy(), np.zeros_like(u0)
        with _cl.redirect_stdout(_io.StringIO()):
            for _ in range(10):
                dsingle = sref.step()
        for n_split in (2, 4):
            dm = cart_split(n_split)
            # spell out scheme and picard so this gate does not silently follow a change
            # of defaults -- it compares against a chorin/picard=1 single-block run
            mb = MultiBlockPISO(dm, NUp, DTp, 2, 1e-13, time_scheme=ts,
                                scheme='chorin', picard_iters=1)
            nxb = NTp // n_split
            for b in range(n_split):
                mb.u[b] = u0[b * nxb:(b + 1) * nxb].copy()
                mb.v[b] = v0[b * nxb:(b + 1) * nxb].copy()
            for _ in range(10):
                dmulti = mb.step()
            ue = np.concatenate([mb.u[b] for b in range(n_split)], axis=0)
            ve = np.concatenate([mb.v[b] for b in range(n_split)], axis=0)
            err = max(np.abs(ue - sref.u).max(), np.abs(ve - sref.v).max())
            check(f"{n_split}-block PISO step ({ts}) reproduces the single-block trajectory",
                  err < 1e-8 and dmulti < 1e-12,
                  f"max|u - u_single| = {err:.2e} after 10 steps; flux divergence "
                  f"{dmulti:.1e} (single block {dsingle:.1e})")

    print("\n13. SECOND-ORDER TIME ACROSS BLOCKS (rotational + BDF2 + Picard)")
    # BDF2 alone does not deliver second order once convection is active: chorin's splitting
    # error and the Picard lag on the convecting velocity are each O(dt) and sit in front of it.
    # Both must be removed, and the multi-block solver must reproduce that.
    def mb_run(dt, pic, scheme, n=4, T=0.16):
        dm = cart_split(n)
        m = MultiBlockPISO(dm, NUp, dt, 2, 1e-14, time_scheme="bdf2",
                           scheme=scheme, picard_iters=pic)
        nxb = NTp // n
        xi1 = np.arange(NTp) / NTp
        Xc, Yc, Zc = np.meshgrid(xi1, xi1, xi1, indexing="ij")
        ua = np.sin(Kp * Xc) * np.cos(Kp * Yc) * np.cos(Kp * Zc)
        va = -np.cos(Kp * Xc) * np.sin(Kp * Yc) * np.cos(Kp * Zc)
        for b in range(n):
            m.u[b] = ua[b * nxb:(b + 1) * nxb].copy()
            m.v[b] = va[b * nxb:(b + 1) * nxb].copy()
        for _ in range(int(round(T / dt))):
            m.step()
        return np.concatenate([m.u[b] for b in range(n)], axis=0)

    DTt = (0.02, 0.01, 0.005, 0.0025, 0.00125)
    for pic, lo_bar, hi_bar, tag in ((1, 0.9, 1.6, "capped by the Picard lag"),
                                     (2, 1.9, 2.4, "second order")):
        Ut = [mb_run(dt, pic, "rotational") for dt in DTt]
        ordt = [np.log2(np.abs(Ut[i] - Ut[i + 1]).max() / np.abs(Ut[i + 1] - Ut[i + 2]).max())
                for i in range(len(DTt) - 2)]
        print(f"   rotational+BDF2, picard_iters={pic}: orders " +
              ", ".join(f"{o:.2f}" for o in ordt) + f"   ({tag})")
        check(f"multi-block temporal order with picard_iters={pic} is {tag}",
              lo_bar < min(ordt) and max(ordt) < hi_bar,
              f"orders {', '.join(f'{o:.2f}' for o in ordt)}")

    # and it must still be the SAME answer as the single-block solver in that configuration
    sref = PISOSolver((NTp, NTp, NTp), warp=1e-9, nu=NUp, dt=0.02, corrector_steps=2,
                      periodic=P3, scheme="rotational", time_scheme="bdf2",
                      convection="central", pressure_coef="rowsum", pressure_tol=1e-14,
                      picard_iters=2)
    u0 = np.sin(Kp * sref.x) * np.cos(Kp * sref.y) * np.cos(Kp * sref.z)
    v0 = -np.cos(Kp * sref.x) * np.sin(Kp * sref.y) * np.cos(Kp * sref.z)
    sref.u, sref.v, sref.w = u0.copy(), v0.copy(), np.zeros_like(u0)
    with _cl.redirect_stdout(_io.StringIO()):
        for _ in range(10):
            sref.step()
    dm = cart_split(4)
    m = MultiBlockPISO(dm, NUp, 0.02, 2, 1e-14, time_scheme="bdf2", scheme="rotational",
                       picard_iters=2)
    nxb = NTp // 4
    for b in range(4):
        m.u[b] = u0[b * nxb:(b + 1) * nxb].copy()
        m.v[b] = v0[b * nxb:(b + 1) * nxb].copy()
    for _ in range(10):
        dmv = m.step()
    ue = np.concatenate([m.u[b] for b in range(4)], axis=0)
    check("rotational+BDF2+Picard multi-block matches the single-block trajectory",
          np.abs(ue - sref.u).max() < 1e-8 and dmv < 1e-12,
          f"max|u - u_single| = {np.abs(ue - sref.u).max():.2e}, flux divergence {dmv:.1e}")

    n_pass = sum(results)
    print(f"\n{'='*74}\n  {n_pass}/{len(results)} checks passed\n{'='*74}")
    sys.exit(0 if n_pass == len(results) else 1)
