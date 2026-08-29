"""
Armaly backward-facing step: a 3D two-domain grid, periodic in the spanwise direction.

GEOMETRY (Armaly, Durst, Pereira & Schonung 1983), non-dimensionalised by the step height S:

    S      = 1                 step height
    h_in   = 1.0612            inlet channel height   (5.2 / 4.9 mm)
    H      = 2.0612            downstream height      (10.1 / 4.9 mm)
    ER     = H / h_in = 1.9423 expansion ratio -- Armaly's defining number
    Re     = U_bulk * D_h / nu,  D_h = 2 h_in  (hydraulic diameter of the inlet duct)

WHY THIS DECOMPOSITION. A `Connection` joins two FULL faces, so the obvious split -- inlet block
upstream of the step, expansion block downstream -- does not work: the inlet face is shorter than
the downstream face, which is a PARTIAL-face interface and is not supported. Splitting instead
about the step level y = 0 gives two blocks that share their entire interface:

    y                        upper block: inflow at -x, wall at +y, CONNECTED at -y
    h_in +-------------------------------------------+
         |                UPPER (inlet stream)        |
      0  +=========== connected interface ============+   <- an interior face, not a wall
         |                LOWER (recirculation)       |
    -S   +-------------------------------------------+
         x=0                                      x=L
         ^ step face (WALL on the lower block only)

The step is then the lower block's -x wall, and the inlet profile is prescribed on the upper
block's -x face. That is the standard BFS setup when the upstream channel is not resolved: the
fully developed profile is imposed at the step plane.

NODE PLACEMENT. The two blocks partition y WITHOUT duplicating the interface node, as a
connected face requires -- so no node sits exactly on y = 0. Both outer walls DO carry nodes.
x keeps both endpoints (inflow and outflow are real boundaries); z is periodic, so its far
endpoint is not stored.
"""
import numpy as np

from multiblock import Block, Connection, Domain, face_id

S, H_IN = 1.0, 1.0612
H_TOT = S + H_IN                      # 2.0612  -> ER = 1.9423
L_OUT, L_SPAN = 30.0 * S, 4.0 * S


def _cluster(n, lo, hi, beta=1.6, both=True):
    """Symmetric (or one-sided) tanh clustering -- fine at the walls, coarse in the middle."""
    t = np.linspace(-1.0, 1.0, n) if both else np.linspace(0.0, 1.0, n)
    if both:
        u = np.tanh(beta * t) / np.tanh(beta)
        s = 0.5 * (u + 1.0)
    else:
        s = np.tanh(beta * t) / np.tanh(beta)
    return lo + (hi - lo) * s


def bfs_domain(nx=120, ny_lo=24, ny_up=26, nz=16):
    """Two blocks about the step level, periodic spanwise."""
    # x: clustered toward the step, both endpoints kept (inflow and outflow are real boundaries)
    # Clustered toward the STEP. Note the direction: tanh(a*t) is steep at t=0, so it gives
    # COARSE cells at x=0 and fine ones at the outlet -- the exact opposite of what a BFS needs,
    # where the shear layer and reattachment are near the step. The reflected form is required.
    t = np.linspace(0, 1, nx)
    x = L_OUT * (1.0 - np.tanh(2.2 * (1.0 - t)) / np.tanh(2.2))
    # y: ONE distribution over the full height, then partitioned. Clustered at both walls and at
    # the step level, where the shear layer sits.
    y_lo = _cluster(ny_lo + 1, -S, 0.0, beta=1.5)[:-1]        # drop the interface node
    y_up = _cluster(ny_up + 1, 0.0, H_IN, beta=1.5)[1:]       # and again on the other side
    z = np.arange(nz) / nz * L_SPAN                           # periodic: far endpoint not stored

    # The y-axis is ONE axis shared by both blocks, so its computational spacing is GLOBAL:
    # 1/(Ny_total - 1), not 1/ny per block. Passing a per-block spacing gives each block a
    # different metric scaling for the same physical direction, which corrupts the Jacobian --
    # it came out NEGATIVE (-2.2e+03) before this was fixed. build_diffusion_matrix does check
    # that the spacing matches across a seam, but only at assembly, long after the grid looks
    # plausible.
    ny_tot = ny_lo + ny_up
    hx, hy, hz = 1.0 / (nx - 1), 1.0 / (ny_tot - 1), 1.0 / nz
    blocks = []
    for ys, ny in ((y_lo, ny_lo), (y_up, ny_up)):
        X, Y, Z = np.meshgrid(x, ys, z, indexing="ij")
        # period MUST be the physical spanwise extent. It defaults to (1,1,1), and the span
        # here is 4S: a wrap shifted by 1.0 instead of 4.0 corrupts the seam and flips the
        # Jacobian negative (measured -4.5e+03). This is the same defect that once had
        # compute_numerical_metrics hardcoding period=1 -- which is precisely why Block carries
        # it as data rather than assuming it.
        blk = Block((nx, ny, nz), X, Y, Z, (hx, hy, hz), period=(1.0, 1.0, L_SPAN))
        blk.faces[face_id(2, 0)] = blk.faces[face_id(2, 1)] = "periodic"
        blocks.append(blk)
    LOW, UP = 0, 1
    blocks[LOW].faces[face_id(0, 0)] = "wall"      # the STEP face
    blocks[LOW].faces[face_id(1, 0)] = "wall"      # bottom wall
    blocks[LOW].faces[face_id(0, 1)] = "wall"      # outflow (velocity BC set by the caller)
    blocks[UP].faces[face_id(0, 0)] = "wall"       # inflow (profile set by the caller)
    blocks[UP].faces[face_id(1, 1)] = "wall"       # top wall
    blocks[UP].faces[face_id(0, 1)] = "wall"       # outflow
    conn = [Connection(LOW, face_id(1, 1), UP, face_id(1, 0))]
    return Domain(blocks, conn), LOW, UP


if __name__ == "__main__":
    d, LOW, UP = bfs_domain()
    print(f"  Armaly BFS: ER = {H_TOT / H_IN:.4f} (Armaly 1.9423), "
          f"L = {L_OUT:.0f}S, span = {L_SPAN:.0f}S periodic")
    print(f"  {len(d.blocks)} blocks, {d.n_cells} cells, {len(d.connections)} connection")
    probs = d.validate()
    print(f"  validate(): {len(probs)} problem(s)" + (f"   {probs[0][:100]}" if probs else ""))
    for nm, b in (("lower", LOW), ("upper", UP)):
        blk = d.blocks[b]
        J, _ = d.block_metrics_cached(b)
        print(f"    {nm}: {blk.shape}  y in [{blk.y.min():+.4f}, {blk.y.max():+.4f}]  "
              f"min J {J.min():.3e}")
    ga, gb = d.pair_indices(d.connections[0])
    print(f"  interface: {len(ga)} matched node pairs; "
          f"gap {abs(d.blocks[UP].y.min() - d.blocks[LOW].y.max()):.5f} "
          f"(non-zero => the interface node is NOT duplicated)")
