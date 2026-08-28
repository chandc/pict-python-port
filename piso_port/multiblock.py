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

    def __init__(self, ba, fa, bb, fb, axes=(0, 1), flips=(False, False)):
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

    def __repr__(self):
        return (f"Connection(block {self.ba} {FACE_NAMES[self.fa]} <-> "
                f"block {self.bb} {FACE_NAMES[self.fb]}, axes={self.axes}, flips={self.flips})")


class Block:
    """One structured block: a shape, physical coordinates, and six face types."""

    def __init__(self, shape, x, y, z, h, faces=None):
        self.shape = tuple(shape)
        self.x, self.y, self.z = x, y, z
        self.h = tuple(h)
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
