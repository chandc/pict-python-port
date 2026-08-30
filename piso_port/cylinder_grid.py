"""
Vortex street behind a circular cylinder: a multi-block O-grid extended to the FAR FIELD.

WHY AN O-GRID. An H-grid around a body has REENTRANT CORNERS, where the diagonal ghost lies
inside the solid and the two blocks either side of a seam extrapolate it differently. That cost
real work to fix in `face_fluxes` (see reference/pressure_checkerboard.md). An O-grid has none:
every block joins its two azimuthal neighbours around a closed ring, and the only boundaries are
the cylinder surface and the far field.

WHY THE FAR FIELD HAS TO BE FAR. The earlier O-grid stopped at R = 6 D_cyl -- 12 diameters
across. Confinement that tight changes the shedding frequency and can suppress the instability
outright; the usual guidance for a clean Strouhal number is a domain radius of 20-50 diameters.
Reaching it without an absurd cell count is what the geometric radial stretching is for: the
wall cell stays fine enough to resolve the boundary layer while the outer cells grow to O(1)
diameter.

BOUNDARY ASSIGNMENT IS PER BLOCK, which is why the azimuthal block count matters beyond load
balance. A face carries ONE condition, so the outer ring can only be split into inflow and
outflow at block boundaries. With the flow along +x, blocks straddling theta = pi face upstream
(free stream) and those straddling theta = 0 face downstream (outflow); the count sets how
faithfully that division follows the true stagnation streamline.

NODE PLACEMENT. The azimuthal direction closes on itself, so it is partitioned WITHOUT
duplicating the seam nodes, exactly like a periodic axis. Radially there are real boundaries at
both ends -- cylinder surface and far field -- so both endpoints are kept. The span is periodic.
"""
import numpy as np

from src.multiblock import Block, Connection, Domain, face_id

D = 1.0                       # cylinder diameter -- the reference length
R_CYL = 0.5 * D


def _radial(nr, r0, r1, first):
    """Geometric radial distribution: `first` is the wall cell height, growing to reach r1."""
    lo, hi = 1.0, 1.6
    for _ in range(200):                      # bisect the growth ratio to land exactly on r1
        s = 0.5 * (lo + hi)
        tot = first * (nr - 1) if abs(s - 1) < 1e-12 else first * (s ** (nr - 1) - 1) / (s - 1)
        if tot < (r1 - r0):
            lo = s
        else:
            hi = s
    s = 0.5 * (lo + hi)
    dr = first * s ** np.arange(nr - 1)
    r = np.concatenate([[r0], r0 + np.cumsum(dr)])
    return r * (r1 - r0) / (r[-1] - r0) - r0 * ((r1 - r0) / (r[-1] - r0) - 1.0), s


def cylinder_domain(nblk=8, nr=88, nth_tot=192, nz=8, r_out=40.0 * D,
                    first=0.006 * D, span=4.0 * D):
    """O-grid ring of `nblk` blocks, cylinder at the centre, far field at `r_out`."""
    if nth_tot % nblk:
        raise ValueError(f"nth_tot={nth_tot} must divide by nblk={nblk}")
    r, ratio = _radial(nr, R_CYL, r_out, first)
    th_all = np.linspace(0.0, 2 * np.pi, nth_tot, endpoint=False)   # ring: no duplicate seam
    z = np.arange(nz) / nz * span                                   # periodic: no far endpoint
    nth = nth_tot // nblk
    h = (1.0 / (nr - 1), 1.0 / nth_tot, 1.0 / nz)

    blocks = []
    for b in range(nblk):
        th = th_all[b * nth:(b + 1) * nth]
        Rg, Tg, Zg = np.meshgrid(r, th, z, indexing="ij")
        blk = Block((nr, nth, nz), Rg * np.cos(Tg), Rg * np.sin(Tg), Zg, h,
                    period=(1.0, 1.0, span))     # span is physical, NOT the default 1
        blk.faces[face_id(0, 0)] = "wall"        # cylinder surface, no-slip
        blk.faces[face_id(0, 1)] = "wall"        # far field: free stream or outflow, by block
        blk.faces[face_id(2, 0)] = blk.faces[face_id(2, 1)] = "periodic"
        blocks.append(blk)

    # the ring closes in PHYSICAL space, so no period shift -- unlike a periodic box, where the
    # coordinate itself jumps by one box length across the wrap
    conns = [Connection(b, face_id(1, 1), (b + 1) % nblk, face_id(1, 0)) for b in range(nblk)]
    return Domain(blocks, conns), r, ratio


def outer_role(d, nblk, flow_x=True):
    """Classify each block's far-field face as 'inflow' or 'outflow' by its mean azimuth.

    A face carries ONE condition, so the split can only fall on block boundaries. With the
    stream along +x the downstream half-plane is |theta| < pi/2.
    """
    roles = {}
    for b in range(nblk):
        th = np.arctan2(d.blocks[b].y[-1].mean(), d.blocks[b].x[-1].mean())
        roles[b] = "outflow" if (np.cos(th) > 0) == flow_x else "inflow"
    return roles


if __name__ == "__main__":
    NBLK = 8
    d, r, ratio = cylinder_domain(nblk=NBLK)
    print(f"  Cylinder vortex street, O-grid to the far field.  D = {D}")
    print(f"  {len(d.blocks)} blocks, {d.n_cells:,} cells, {len(d.connections)} connections\n")
    probs = d.validate()
    print(f"  validate(): {len(probs)} problem(s)")
    for p in probs[:4]:
        print(f"      {p[:110]}")
    Jmin = min(d.block_metrics_cached(b)[0].min() for b in range(len(d.blocks)))
    print(f"  min(J) = {Jmin:.4e}   {'valid' if Jmin > 0 else 'TANGLED'}")
    print(f"\n  radial: {len(r)} points, r = {r[0]:.3f} -> {r[-1]:.1f}  "
          f"({r[-1]/D:.0f} diameters)")
    print(f"    wall cell  {r[1]-r[0]:.4f} D   outer cell {r[-1]-r[-2]:.3f} D   "
          f"growth ratio {ratio:.4f}")
    roles = outer_role(d, NBLK)
    ins = [b for b in roles if roles[b] == "inflow"]
    outs = [b for b in roles if roles[b] == "outflow"]
    print(f"\n  far-field faces: inflow on blocks {ins}, outflow on blocks {outs}")
