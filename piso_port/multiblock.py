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

    def _ghost_layers(self, b, fid, width, src=None, shift=True):
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
                jump = (blk.period[axis] if comp == axis else 0.0) if shift else 0.0
                out.append(lay + (jump if side == 1 else -jump))
            return out

        nb = self._neighbour_of(b, fid)
        if nb is None:
            return None
        ob, ofid, to_mine, sh = nb
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
            out.append(lay + (sh[comp] if shift else 0.0))
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

    def pad_field(self, b, fields, width=2):
        """
        Ghost-pad a SCALAR FIELD of block b across its periodic/connected faces.

        Same machinery as pad_coords with ONE critical difference: NO PERIOD SHIFT. Coordinates
        ramp and jump back, so a wrapped coordinate ghost must be displaced by one period.
        Velocity and pressure are genuinely periodic and must NOT be. Copying the coordinate
        path verbatim would offset every seam by exactly one period -- a large, smooth, entirely
        plausible-looking error.

        `width` must cover the widest stencil that will be applied. Central differencing needs
        1; SOU reaches i-2 and needs 2. Padding with 1 and then applying SOU silently degrades
        the upwind stencil to first order AT SEAMS ONLY, which no smooth test would reveal.

        Returns (padded, lo, hi) so the caller can strip the ghosts afterwards.
        """
        blk = self.blocks[b]
        if isinstance(fields, np.ndarray):
            raise TypeError(
                "pad_field needs the field for EVERY block, as a dict or list keyed by block "
                "index -- a connected face reads across the seam. Passing one array cannot "
                "work, and an earlier cached-state version of this API silently used whichever "
                "field had been registered last for all three velocity components.")
        cur = np.asarray(fields[b])
        if cur.shape != blk.shape:
            raise ValueError(f"field shape {cur.shape} does not match block {b} {blk.shape}")
        lo, hi = [0, 0, 0], [0, 0, 0]
        conn_axes = [a for a in range(3)
                     if any(blk.faces[face_id(a, s)] == "connected" for s in (0, 1))]
        for axis in conn_axes + [a for a in range(3) if a not in conn_axes]:
            g = {}
            for side in (0, 1):
                lay = self._ghost_layers_field(b, face_id(axis, side), width, cur,
                                               fields)
                g[side] = lay
            if g[0] is None and g[1] is None:
                continue
            parts = []
            if g[0] is not None:
                parts.append(np.moveaxis(g[0][::-1], 0, axis))
            parts.append(cur)
            if g[1] is not None:
                parts.append(np.moveaxis(g[1], 0, axis))
            cur = np.concatenate(parts, axis=axis)
            if g[0] is not None:
                lo[axis] = width
            if g[1] is not None:
                hi[axis] = width
        return cur, lo, hi

    def _ghost_layers_field(self, b, fid, width, cur, fields):
        """Layers of a scalar field beyond face `fid`, in b's ordering. No period shift."""
        blk = self.blocks[b]
        axis, side = face_axis_side(fid)
        kind = blk.faces[fid]
        if kind == "periodic":
            sl = [slice(None)] * 3
            sl[axis] = slice(0, width) if side == 1 else slice(-width, None)
            lay = np.moveaxis(cur[tuple(sl)], axis, 0)
            return lay if side == 1 else lay[::-1]
        nb = self._neighbour_of(b, fid)
        if nb is None:
            return None
        ob, ofid, to_mine, _ = nb
        oaxis, oside = face_axis_side(ofid)
        other = np.asarray(fields[ob])
        sl = [slice(None)] * 3
        sl[oaxis] = slice(0, width) if oside == 0 else slice(-width, None)
        lay = np.moveaxis(other[tuple(sl)], oaxis, 0)
        if oside == 1:
            lay = lay[::-1]
        return np.stack([to_mine(l) for l in lay])


    # ------------------------------------------------------------------ seam-aware operators

    def padded_geometry(self, b, width=1):
        """Jacobian and metrics on the PADDED block, ghosts retained."""
        from phase1_grid_metrics import _metrics_core
        blk = self.blocks[b]
        xp, yp, zp, lo, hi = self.pad_coords(b, max(width, 2))
        J, m = _metrics_core(xp, yp, zp, *blk.h)
        # metrics need width>=2 for the nested derivatives, but the caller may only want 1
        trim = [max(0, (max(width, 2) if lo[a] else 0) - width) for a in range(3)]
        trimh = [max(0, (max(width, 2) if hi[a] else 0) - width) for a in range(3)]
        sl = tuple(slice(trim[a] or None, -trimh[a] if trimh[a] else None) for a in range(3))
        lo2 = [lo[a] - trim[a] for a in range(3)]
        hi2 = [hi[a] - trimh[a] for a in range(3)]
        return J[sl], {k: v[sl] for k, v in m.items()}, lo2, hi2

    def face_fluxes(self, b, us, vs, ws):
        """
        Face fluxes for block b with CONNECTED and PERIODIC faces resolved from real neighbour
        data, so a seam face is an ordinary interior face rather than a prescribed boundary.

        This is what `compute_face_fluxes` cannot do block-locally: it treats every
        non-periodic face as a domain boundary, so a connection would receive a PRESCRIBED flux
        instead of an interpolated one -- mass would be injected or lost at every seam.

        Wall faces keep the single-block behaviour (flux from the boundary cell's contravariant
        component, zero for an impermeable wall).
        """
        from phase5_fluxes import contravariant_components
        blk = self.blocks[b]
        Jp, mp, lo, hi = self.padded_geometry(b, 1)
        up = self.pad_field(b, us, 1)[0]
        vp = self.pad_field(b, vs, 1)[0]
        wp = self.pad_field(b, ws, 1)[0]
        JU = contravariant_components(up, vp, wp, Jp, mp)
        F = []
        for axis in range(3):
            n = blk.shape[axis]
            shape = list(blk.shape); shape[axis] += 1
            f = np.zeros(shape)
            for k in range(n + 1):
                a_lo = lo[axis] + k - 1
                a_hi = lo[axis] + k
                sl_out = [slice(None)] * 3; sl_out[axis] = k
                core = [slice(lo[a], lo[a] + blk.shape[a]) for a in range(3)]
                if a_lo < 0 or a_hi >= JU[axis].shape[axis]:
                    # a true domain boundary: fall back to the boundary cell's own component
                    idx = max(a_lo, 0) if a_lo < 0 else a_hi - 1
                    sl_in = list(core); sl_in[axis] = idx
                    f[tuple(sl_out)] = JU[axis][tuple(sl_in)]
                    continue
                s1 = list(core); s1[axis] = a_lo
                s2 = list(core); s2[axis] = a_hi
                f[tuple(sl_out)] = 0.5 * (JU[axis][tuple(s1)] + JU[axis][tuple(s2)])
            F.append(f)
        return F

    def divergence(self, b, F, J):
        """Flux divergence for block b -- identical form to divergence_from_fluxes."""
        blk = self.blocks[b]
        d = np.zeros(blk.shape)
        for axis in range(3):
            lo_s = [slice(None)] * 3; lo_s[axis] = slice(0, -1)
            hi_s = [slice(None)] * 3; hi_s[axis] = slice(1, None)
            d += (F[axis][tuple(hi_s)] - F[axis][tuple(lo_s)]) / blk.h[axis]
        return d / J

    def build_momentum_matrix(self, Js, metrics_list, us, vs, ws, nu, dt, bdf2=False,
                              convection='central'):
        """
        Global momentum operator  A = J/dt (or 3J/2dt) + J*convection + nu*diffusion, assembled
        across blocks as one matrix.

        CENTRAL convection only. That is a deliberate scope choice, not an oversight: central is
        the scheme required for anything where dissipation matters -- test_energy_conservation.py
        shows its convective operator conserves kinetic energy to round-off while SOU removes
        ~10% per turnover on a broadband field -- and its 7-point stencil matches the connection
        machinery already verified here. SOU reaches i-2, so it needs two ghost layers at a seam
        and a wider assembly; that is a separate increment, and build_momentum_matrix raises
        rather than silently degrading the upwind stencil at seams.

        The convection term is NOT symmetric, unlike diffusion, so each face writes DIFFERENT
        values into the two rows it touches.
        """
        if convection != 'central':
            raise NotImplementedError(
                f"multi-block momentum supports convection='central' only, not {convection!r}. "
                f"SOU reaches i-2, so it needs two ghost layers at a seam and a wider assembly "
                f"than the 7-point connection machinery verified here. Falling back to central "
                f"silently would change the physics -- SOU removes ~10% of kinetic energy per "
                f"turnover on a broadband field where central conserves it to round-off -- so "
                f"this raises instead.")
        N = self.n_cells
        rows, cols, vals = [], [], []
        diag = [np.zeros(b.shape) for b in self.blocks]

        # contravariant convecting velocity, per block, on PADDED arrays so a seam face sees the
        # neighbour's velocity rather than a one-sided guess
        from phase5_fluxes import contravariant_components
        UVW = {}
        for b, blk in enumerate(self.blocks):
            up = self.pad_field(b, us, 1)[0]
            vp = self.pad_field(b, vs, 1)[0]
            wp = self.pad_field(b, ws, 1)[0]
            Jp, mp, lo, hi = self.padded_geometry(b, 1)
            UVW[b] = (contravariant_components(up, vp, wp, Jp, mp), lo, hi)

        def conv_coef(b, axis):
            """J-weighted contravariant component on this block's own cells."""
            (JU, lo, hi) = UVW[b]
            core = tuple(slice(lo[a], lo[a] + self.blocks[b].shape[a]) for a in range(3))
            return JU[axis][core]

        def Jg_of(b, axis):
            m = metrics_list[b]
            key = ("xi", "eta", "zeta")[axis]
            g = m[f"{key}_x"] ** 2 + m[f"{key}_y"] ** 2 + m[f"{key}_z"] ** 2
            return nu * Js[b] * g

        def add_face(gP, gN, cf_diff, aP, aN, hh):
            """One face: symmetric diffusion, antisymmetric central convection."""
            rows.append(gP); cols.append(gN); vals.append(-cf_diff + aP / (2 * hh))
            rows.append(gN); cols.append(gP); vals.append(-cf_diff - aN / (2 * hh))

        for b, blk in enumerate(self.blocks):
            gid = self.global_ids(b)
            for axis in range(3):
                h = blk.h[axis]
                Jg = Jg_of(b, axis)
                a_c = conv_coef(b, axis)
                lo_s = [slice(None)] * 3; lo_s[axis] = slice(0, -1)
                hi_s = [slice(None)] * 3; hi_s[axis] = slice(1, None)
                cf = 0.5 * (Jg[tuple(lo_s)] + Jg[tuple(hi_s)]) / h ** 2
                add_face(gid[tuple(lo_s)].ravel(), gid[tuple(hi_s)].ravel(), cf.ravel(),
                         a_c[tuple(lo_s)].ravel(), a_c[tuple(hi_s)].ravel(), h)
                diag[b][tuple(lo_s)] += cf
                diag[b][tuple(hi_s)] += cf
                if blk.faces[face_id(axis, 1)] == "periodic":
                    f0 = [slice(None)] * 3; f0[axis] = 0
                    fn = [slice(None)] * 3; fn[axis] = -1
                    cw = 0.5 * (Jg[tuple(fn)] + Jg[tuple(f0)]) / h ** 2
                    add_face(gid[tuple(fn)].ravel(), gid[tuple(f0)].ravel(), cw.ravel(),
                             a_c[tuple(fn)].ravel(), a_c[tuple(f0)].ravel(), h)
                    diag[b][tuple(fn)] += cw
                    diag[b][tuple(f0)] += cw

        for c in self.connections:
            axis, _ = face_axis_side(c.fa)
            oaxis, _ = face_axis_side(c.fb)
            h = self.blocks[c.ba].h[axis]
            JgA = Jg_of(c.ba, axis)[face_slice(c.fa)]
            JgB = c.align(Jg_of(c.bb, oaxis)[face_slice(c.fb)])
            cf = 0.5 * (JgA + JgB) / h ** 2
            aA = conv_coef(c.ba, axis)[face_slice(c.fa)]
            aB = c.align(conv_coef(c.bb, oaxis)[face_slice(c.fb)])
            ga, gb = self.pair_indices(c)
            add_face(ga, gb, cf.ravel(), aA.ravel(), aB.ravel(), h)
            diag[c.ba][face_slice(c.fa)] += cf
            diag[c.bb][face_slice(c.fb)] += c.unalign(cf)

        c0 = 1.5 / dt if bdf2 else 1.0 / dt
        for b in range(len(self.blocks)):
            rows.append(self.global_ids(b).ravel())
            cols.append(self.global_ids(b).ravel())
            vals.append((diag[b] + Js[b] * c0).ravel())

        return sparse.coo_matrix((np.concatenate(vals),
                                  (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(N, N)).tocsr()
