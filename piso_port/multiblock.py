"""
Multi-block domains: global index space, face types, and the connection map.

OPTIONAL BY CONSTRUCTION. Nothing here is imported by the single-block solve path; a domain of
one block with no connections is exactly the existing solver, and `Domain.is_single_block` lets
callers keep the old fast path. The single-block behaviour is not "mostly preserved" -- it is
untouched.

Following PICT (see reference/multiblock_offsets.md): blocks are joined through ONE GLOBAL
MATRIX with NO GHOST CELLS. Each block owns a contiguous slab of the global index space starting
at its `globalOffset`, and a connection contributes off-diagonal entries directly between the
two blocks' boundary cells, exactly as an interior face does within a block.

THE NODE-PLACEMENT CONSTRAINT, which our grid layout forces and PICT's does not:

    wall / inflow face -- linspace(0,1,n): BOTH endpoints stored, nodes sit ON the boundary
    periodic axis      -- arange(n)/n: far endpoint NOT stored, it is the same point as node 0
    CONNECTED face     -- must follow the PERIODIC rule

PICT is cell-centred, so its cells are distinct either way and a connection is just a face. Our
nodes sit on boundaries, so if both blocks stored their interface nodes those nodes would be
DUPLICATED and the resolution would halve across the seam -- the same failure `make_grid`'s
docstring warns about for periodic axes. A connected axis therefore behaves like a periodic one
for node placement: block A stores up to but not including the interface, and block B's first
node IS the next node. `Domain.validate()` enforces this rather than leaving it to the caller.
"""
import numpy as np
import scipy.sparse as sparse

FACE_NAMES = ("-x", "+x", "-y", "+y", "-z", "+z")


def face_id(axis, side):
    """axis 0/1/2, side 0=lower 1=upper  ->  0..5, matching FACE_NAMES."""
    return 2 * axis + side


def face_axis_side(fid):
    return fid // 2, fid % 2


def face_slice(fid):
    """Index tuple selecting that face's layer of cells from a (nx,ny,nz) array."""
    axis, side = face_axis_side(fid)
    s = [slice(None)] * 3
    s[axis] = 0 if side == 0 else -1
    return tuple(s)


def tangential_axes(axis):
    return tuple(a for a in range(3) if a != axis)


class Connection:
    """
    Joins face `fa` of block index `ba` to face `fb` of block index `bb`.

    `axes` maps face fa's two TANGENTIAL axes onto fb's: (0,1) identity, (1,0) swapped.
    `flips` says whether each of those directions is reversed.

    Orientation is kept as a permutation PLUS an explicit flip pair rather than PICT's signed
    axis indices, because a sign cannot express "flip axis 0" in Python: -0 == 0, so a signed
    encoding silently loses exactly one of the eight orientations. This is the piece where
    multi-block bugs live -- two blocks can meet with permuted or flipped axes, so a face cell
    at (i,j) on one side may be (j, n-1-i) on the other, and getting it wrong yields a
    plausible-looking field with a scrambled seam.
    """

    def __init__(self, ba, fa, bb, fb, axes=(0, 1), flips=(False, False),
                 shift=(0.0, 0.0, 0.0)):
        # Physical displacement to ADD to block B's coordinates when viewed from
        # A. Zero for blocks that simply abut; for a WRAP-AROUND connection (the
        # last block of a periodic strip joining back to the first) it is the
        # domain period, exactly as wrap_pad_coords shifts a periodic seam. Omit
        # it and the ghost coordinates jump backwards across the seam, collapsing
        # the Jacobian there -- the same failure the `period` bug produced.
        self.shift = np.asarray(shift, dtype=float)
        self.ba, self.fa, self.bb, self.fb = ba, fa, bb, fb
        self.axes = tuple(axes)
        self.flips = tuple(bool(f) for f in flips)
        if sorted(self.axes) != [0, 1]:
            raise ValueError(f"axes must be a permutation of (0,1), got {axes}")
        aa, sa = face_axis_side(fa)
        ab, sb = face_axis_side(fb)
        if sa == sb:
            # A '+x' face meets a '-x' face, never another '+x': one block's outgoing normal
            # must be the other's incoming one, or the blocks overlap instead of abutting.
            raise ValueError(
                f"connection joins {FACE_NAMES[fa]} to {FACE_NAMES[fb]}: both are "
                f"{'upper' if sa else 'lower'} faces, so the blocks would overlap")

    def align(self, arr_b):
        """Reorder block B's face array so element [i,j] is the neighbour of A's face [i,j]."""
        out = np.transpose(arr_b, self.axes)
        if self.flips[0]:
            out = out[::-1]
        if self.flips[1]:
            out = out[:, ::-1]
        return out

    def unalign(self, arr_a):
        """Inverse of align(): reorder an A-face array into B's face ordering."""
        out = arr_a
        if self.flips[1]:
            out = out[:, ::-1]
        if self.flips[0]:
            out = out[::-1]
        return np.transpose(out, np.argsort(self.axes))

    def __repr__(self):
        return (f"Connection(block {self.ba} {FACE_NAMES[self.fa]} <-> "
                f"block {self.bb} {FACE_NAMES[self.fb]}, axes={self.axes}, flips={self.flips})")


class Block:
    """One structured block: a shape, physical coordinates, and six face types."""

    def __init__(self, shape, x, y, z, h, faces=None, period=(1.0, 1.0, 1.0)):
        self.shape = tuple(shape)
        self.x, self.y, self.z = x, y, z
        self.h = tuple(h)
        # Physical length of one period along each axis, for PERIODIC faces. Coordinates are
        # not themselves periodic -- x ramps and then jumps back -- so a wrapped ghost must be
        # shifted by one period or it injects a spurious derivative and collapses the Jacobian
        # at the seam. This is the same defect that `compute_numerical_metrics` had when it
        # hardcoded period=1; here it is per-block data, not an assumption.
        self.period = np.asarray(period, dtype=float)
        # face type per face id: 'wall' | 'periodic' | 'connected' | 'inflow' | 'outflow'
        self.faces = list(faces) if faces is not None else ["wall"] * 6

    @property
    def size(self):
        return int(np.prod(self.shape))


class Domain:
    """Blocks plus connections, with the global index space laid over them."""

    def __init__(self, blocks, connections=()):
        self.blocks = list(blocks)
        self.connections = list(connections)
        self.offsets = np.cumsum([0] + [b.size for b in self.blocks])[:-1]
        self.n_cells = int(sum(b.size for b in self.blocks))
        for c in self.connections:
            self.blocks[c.ba].faces[c.fa] = "connected"
            self.blocks[c.bb].faces[c.fb] = "connected"

    @property
    def is_single_block(self):
        return len(self.blocks) == 1 and not self.connections

    def global_ids(self, b):
        """Global cell indices for block b, shaped like the block."""
        blk = self.blocks[b]
        return (self.offsets[b] + np.arange(blk.size)).reshape(blk.shape)

    def pair_indices(self, conn):
        """
        (ids_a, ids_b): matching global cell ids either side of a connection, aligned so that
        ids_a[k] and ids_b[k] are the two cells that share the interface face.
        """
        ga = self.global_ids(conn.ba)[face_slice(conn.fa)]
        gb = self.global_ids(conn.bb)[face_slice(conn.fb)]
        gb = conn.align(gb)
        if ga.shape != gb.shape:
            raise ValueError(
                f"{conn}: face shapes {ga.shape} and {gb.shape} do not match after alignment "
                f"-- check the `axes` permutation")
        return ga.ravel(), gb.ravel()

    def validate(self):
        """
        Check the invariants that produce silent, plausible-looking corruption if violated.
        Returns a list of problems; empty means clean.
        """
        problems = []
        for c in self.connections:
            try:
                ga, gb = self.pair_indices(c)
            except ValueError as e:
                problems.append(str(e)); continue
            if len(set(ga.tolist()) & set(gb.tolist())):
                problems.append(f"{c}: a cell is connected to itself")
        for bi, blk in enumerate(self.blocks):
            na = sum(1 for a in range(3)
                     if any(blk.faces[face_id(a, s)] == "connected" for s in (0, 1)))
            if na > 1:
                problems.append(
                    f"block {bi} has connections on {na} axes. pad_coords currently supports at "
                    f"most one connected axis per block: the corner ghosts would need the "
                    f"neighbour's coordinates padded along the other connected axis, which is "
                    f"not implemented. Reported rather than silently mis-padded.")
        # No node may be stored by both blocks. Checking the SPACING against 1/n is wrong --
        # a connected axis's spacing is set by the GLOBAL cell count across all blocks in that
        # direction, not by one block's count -- so the invariant is stated directly on the
        # coordinates instead: the two blocks' interface nodes must be distinct.
        for c in self.connections:
            A, B = self.blocks[c.ba], self.blocks[c.bb]
            try:
                pa = [f[face_slice(c.fa)] for f in (A.x, A.y, A.z)]
                pb = [c.align(f[face_slice(c.fb)]) for f in (B.x, B.y, B.z)]
            except ValueError:
                continue                                   # already reported above
            if pa[0].shape != pb[0].shape:
                continue
            gap = np.sqrt(sum((a - b) ** 2 for a, b in zip(pa, pb)))
            if np.any(gap < 1e-12):
                problems.append(
                    f"{c}: interface nodes COINCIDE (min separation {gap.min():.2e}). Both "
                    f"blocks are storing the same node, so it is counted twice and the "
                    f"resolution halves across the seam. A connected face must use "
                    f"periodic-style node placement: store up to but NOT including the "
                    f"interface, and let the neighbour supply the next node.")
        return problems

    def __repr__(self):
        return (f"Domain({len(self.blocks)} blocks, {self.n_cells} cells, "
                f"{len(self.connections)} connections)")

    # ------------------------------------------------------------------ geometry across seams

    def _neighbour_of(self, b, fid):
        """(other_block, other_face, to_my_ordering, shift) for a connected face, else None."""
        for c in self.connections:
            if c.ba == b and c.fa == fid:
                return c.bb, c.fb, c.align, +c.shift
            if c.bb == b and c.fb == fid:
                return c.ba, c.fa, c.unalign, -c.shift
        return None

    def _ghost_layers(self, b, fid, width, src=None):
        """
        `width` coordinate layers just beyond face `fid` of block b, in b's own index ordering
        and physical frame. Returns a list of 3 arrays shaped like b's face with a leading
        layer axis, ordered NEAREST-first, or None if the face is not connected/periodic.
        """
        blk = self.blocks[b]
        axis, side = face_axis_side(fid)
        kind = blk.faces[fid]

        if kind == "periodic":
            # Self-wrap, taken from `src` -- the CURRENT partially padded field, not the raw
            # block. That is what makes the corner ghosts right: by the time a periodic axis is
            # padded the connected axis already carries its neighbour's data, and wrapping the
            # padded field carries that into the corner.
            out = []
            src_f = src if src is not None else (blk.x, blk.y, blk.z)
            for comp, f in enumerate(src_f):
                sl = [slice(None)] * 3
                sl[axis] = slice(0, width) if side == 1 else slice(-width, None)
                lay = np.moveaxis(f[tuple(sl)], axis, 0)
                lay = lay if side == 1 else lay[::-1]          # nearest-first
                # one-period shift: moving one period along axis `axis` displaces the physical
                # point by period[axis] in physical component `axis`, and nothing else
                jump = blk.period[axis] if comp == axis else 0.0
                out.append(lay + (jump if side == 1 else -jump))
            return out

        nb = self._neighbour_of(b, fid)
        if nb is None:
            return None
        ob, ofid, to_mine, shift = nb
        other = self.blocks[ob]
        oaxis, oside = face_axis_side(ofid)
        out = []
        for comp, f in enumerate((other.x, other.y, other.z)):
            sl = [slice(None)] * 3
            sl[oaxis] = slice(0, width) if oside == 0 else slice(-width, None)
            lay = np.moveaxis(f[tuple(sl)], oaxis, 0)          # (width, t0, t1) in B's ordering
            if oside == 1:
                lay = lay[::-1]                                # nearest-first
            lay = np.stack([to_mine(l) for l in lay])          # into my face ordering
            out.append(lay + shift[comp])
        return out

    def pad_coords(self, b, width=2):
        """
        Coordinates of block b ghost-padded across every periodic/connected face.

        Returns (xp, yp, zp, lo, hi) with lo/hi the per-axis pad widths actually applied. Wall
        faces are NOT padded -- there is nothing beyond them and the metric formula falls back
        to one-sided differences there, exactly as the single-block code does.

        Both sides of an axis are read BEFORE either is attached. Padding side 0 and then
        reading side 1 from the modified array makes the upper ghosts a copy of the lower ones,
        which is silent and produces a plausible-looking but wrong Jacobian.

        LIMITATION: at most ONE connected axis per block. Two would need the neighbour's
        coordinates themselves padded along the other connected axis (recursive padding), which
        is not implemented; validate() reports it rather than mis-padding the corners.
        """
        blk = self.blocks[b]
        fields = [blk.x.copy(), blk.y.copy(), blk.z.copy()]
        lo, hi = [0, 0, 0], [0, 0, 0]

        # Connected axes first, while every block still has its original extents so a
        # neighbour's face layer matches this block's face. Periodic axes then wrap the
        # already-padded field, which is what fills the corners correctly.
        conn_axes = [a for a in range(3)
                     if any(blk.faces[face_id(a, s)] == "connected" for s in (0, 1))]
        for axis in conn_axes + [a for a in range(3) if a not in conn_axes]:
            g = {}
            for side in (0, 1):
                g[side] = self._ghost_layers(b, face_id(axis, side), width, src=fields)
            if g[0] is None and g[1] is None:
                continue
            new_fields = []
            for comp in range(3):
                parts = []
                if g[0] is not None:
                    parts.append(np.moveaxis(g[0][comp][::-1], 0, axis))   # farthest-first
                parts.append(fields[comp])
                if g[1] is not None:
                    parts.append(np.moveaxis(g[1][comp], 0, axis))
                new_fields.append(np.concatenate(parts, axis=axis))
            fields = new_fields
            if g[0] is not None:
                lo[axis] = width
            if g[1] is not None:
                hi[axis] = width
        return fields[0], fields[1], fields[2], lo, hi

    def block_metrics(self, b, width=2):
        """Jacobian and metrics for block b, with seams resolved by real neighbour data."""
        from phase1_grid_metrics import _metrics_core
        blk = self.blocks[b]
        xp, yp, zp, lo, hi = self.pad_coords(b, width)
        J, m = _metrics_core(xp, yp, zp, *blk.h)
        sl = tuple(slice(lo[a] or None, -hi[a] if hi[a] else None) for a in range(3))
        return J[sl], {k: v[sl] for k, v in m.items()}

    # ------------------------------------------------------------------ global assembly

    def build_diffusion_matrix(self, Js, metrics_list, coefs=None):
        """
        The volume-integrated conservative diffusion operator over the WHOLE domain, as ONE
        sparse matrix -- PICT's design, and the reason it needs no ghost cells: a connection
        contributes off-diagonal entries between the two blocks' boundary cells exactly as an
        interior face does within a block, so the coupling is implicit in the linear solve
        rather than exchanged between steps.

        Faces are enumerated once each:
          * interior faces of every block,
          * one wrap face per PERIODIC axis of a block (joining its own two ends),
          * one face per CONNECTION, added from the A side only -- adding it from both would
            double the coupling and quietly halve the effective diffusion across every seam.

        Each face writes the SAME interpolated coefficient into both rows it touches, so the
        matrix is symmetric by construction, as in the single-block assembler.
        """
        N = self.n_cells
        rows, cols, vals = [], [], []
        diag = [np.zeros(b.shape) for b in self.blocks]

        def Jg_of(b, axis):
            m = metrics_list[b]
            key = ("xi", "eta", "zeta")[axis]
            g = m[f"{key}_x"] ** 2 + m[f"{key}_y"] ** 2 + m[f"{key}_z"] ** 2
            c = 1.0 if coefs is None else coefs[b]
            return c * Js[b] * g

        for b, blk in enumerate(self.blocks):
            gid = self.global_ids(b)
            for axis in range(3):
                h = blk.h[axis]
                Jg = Jg_of(b, axis)
                lo_s = [slice(None)] * 3; lo_s[axis] = slice(0, -1)
                hi_s = [slice(None)] * 3; hi_s[axis] = slice(1, None)
                cf = 0.5 * (Jg[tuple(lo_s)] + Jg[tuple(hi_s)]) / h ** 2
                rows += [gid[tuple(lo_s)].ravel(), gid[tuple(hi_s)].ravel()]
                cols += [gid[tuple(hi_s)].ravel(), gid[tuple(lo_s)].ravel()]
                vals += [-cf.ravel(), -cf.ravel()]
                diag[b][tuple(lo_s)] += cf
                diag[b][tuple(hi_s)] += cf

                if blk.faces[face_id(axis, 1)] == "periodic":
                    f0 = [slice(None)] * 3; f0[axis] = 0
                    fn = [slice(None)] * 3; fn[axis] = -1
                    cw = 0.5 * (Jg[tuple(fn)] + Jg[tuple(f0)]) / h ** 2
                    a, c_ = gid[tuple(fn)].ravel(), gid[tuple(f0)].ravel()
                    rows += [a, c_]; cols += [c_, a]
                    vals += [-cw.ravel(), -cw.ravel()]
                    diag[b][tuple(fn)] += cw
                    diag[b][tuple(f0)] += cw

        for c in self.connections:
            axis, _ = face_axis_side(c.fa)
            ha = self.blocks[c.ba].h[axis]
            oaxis, _ = face_axis_side(c.fb)
            hb = self.blocks[c.bb].h[oaxis]
            if not np.isclose(ha, hb, rtol=1e-12):
                raise ValueError(
                    f"{c}: computational spacing differs across the seam ({ha:.6g} vs "
                    f"{hb:.6g}). The face coefficient would be ambiguous.")
            JgA = Jg_of(c.ba, axis)[face_slice(c.fa)]
            JgB = c.align(Jg_of(c.bb, oaxis)[face_slice(c.fb)])
            cf = 0.5 * (JgA + JgB) / ha ** 2
            ga, gb = self.pair_indices(c)
            rows += [ga, gb]; cols += [gb, ga]
            vals += [-cf.ravel(), -cf.ravel()]
            diag[c.ba][face_slice(c.fa)] += cf
            diag[c.bb][face_slice(c.fb)] += c.unalign(cf)

        for b in range(len(self.blocks)):
            rows.append(self.global_ids(b).ravel())
            cols.append(self.global_ids(b).ravel())
            vals.append(diag[b].ravel())

        return sparse.coo_matrix((np.concatenate(vals),
                                  (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(N, N)).tocsr()
