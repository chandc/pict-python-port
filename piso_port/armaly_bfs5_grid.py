"""
Armaly backward-facing step: a FIVE-domain grid with a resolved upstream inlet channel.

WHY FIVE. The two-domain grid prescribes the inlet profile at the step plane, because a
`Connection` joins two FULL faces and an upstream block is shorter than the downstream one.
Splitting the DOWNSTREAM region about y = 0 removes that obstruction: the upstream channel then
connects to the upper downstream block face-to-face, at equal height.

    y                inlet channel                recirculation        recovery
  h_in +--------------------------++------------------++--------------------------+
       |         0  INLET         ||   1  RECIRC-U    ||     3  RECOVERY-U        |
   0   +==========================++==================++==========================+
       ^ step top wall            ||   2  RECIRC-L    ||     4  RECOVERY-L        |
  -S                              ++------------------++--------------------------+
       x=-L_IN                   x=0                x=X1                     x=L_OUT
                                  ^ step face (block 2's -x wall)

    == connected      || connected      -- wall

Five full-face connections: 0-1 and 1-3 and 2-4 in x, 1-2 and 3-4 in y. The inlet block has no
counterpart below it, which is exactly the geometry of a step.

NODE PLACEMENT, and why it is not arbitrary. A connected face must not duplicate nodes, so a
partitioned axis leaves a one-cell gap at the interface; a WALL, by contrast, carries nodes. The
two requirements collide wherever a wall and a connection meet at the same station -- block 2's
step face sits at x = 0 while block 1's face there is connected. Resolved by making the
DOWNSTREAM segment start exactly at 0 and the inlet segment end one cell short:

    x_in ends at -dx      x_re starts at  0   -> step face lands exactly on x = 0
    y_lo ends at -dy      y_up starts at  0   -> step top wall lands exactly on y = 0

Every block sharing an axis therefore shares its node distribution, which is what keeps the
metrics consistent across a seam.

COMPUTATIONAL SPACING IS GLOBAL PER AXIS. x is one axis partitioned three ways and y one axis
partitioned two ways, so h = 1/(N_total - 1) for each, NOT 1/n per block. A per-block spacing
gives each block a different metric scaling for the same physical direction and flips the
Jacobian negative -- measured on the two-domain grid before it was fixed.
"""
import numpy as np

from src.multiblock import Block, Connection, Domain, face_id

S, H_IN = 1.0, 1.0612
H_TOT = S + H_IN                       # 2.0612 -> ER = 1.9423
L_IN, X1, L_OUT, L_SPAN = 5.0 * S, 8.0 * S, 30.0 * S, 4.0 * S

INLET, RECIRC_U, RECIRC_L, RECOV_U, RECOV_L = 0, 1, 2, 3, 4
NAMES = ("inlet", "recirc-upper", "recirc-lower", "recovery-upper", "recovery-lower")


def _tanh_to(n, lo, hi, beta, toward_hi=True):
    """One-sided tanh clustering, fine at `hi` when toward_hi else fine at `lo`.

    Watch the direction. tanh(beta*t) is steep near t = 0 and saturates near t = 1, so the
    PLAIN form gives coarse cells at lo and fine ones at hi; the reflected form does the
    opposite. Getting this backwards is silent -- the grid still looks like a grid -- and puts
    the coarse cells exactly where the shear layer is. Measured on the first version of this
    file: dx = 3.7e-01 at the step and 2.8e-02 at the far inlet, precisely inverted.
    """
    t = np.linspace(0.0, 1.0, n)
    u = np.tanh(beta * t) / np.tanh(beta)
    return lo + (hi - lo) * u if toward_hi else lo + (hi - lo) * (1.0 - u[::-1])


def _cluster_both(n, lo, hi, beta=1.5):
    t = np.linspace(-1.0, 1.0, n)
    u = np.tanh(beta * t) / np.tanh(beta)
    return lo + (hi - lo) * 0.5 * (u + 1.0)


def bfs5_domain(nx_in=28, nx_re=56, nx_rc=56, ny_lo=24, ny_up=26, nz=16):
    # --- x: three segments of ONE axis. Inlet stops one cell short of the step; the
    #     recirculation segment starts exactly on it so the step face lands on x = 0.
    x_in_full = _tanh_to(nx_in + 1, -L_IN, 0.0, 2.0, toward_hi=True)
    x_in = x_in_full[:-1]                                   # ends at -dx
    x_re = _tanh_to(nx_re, 0.0, X1, 1.8, toward_hi=False)    # fine at the step
    x_rc = _tanh_to(nx_rc + 1, X1, L_OUT, 1.2, toward_hi=False)[1:]

    # --- y: two segments of ONE axis. Lower stops one cell short; upper starts on y = 0 so
    #     the step's top wall (the inlet block's -y face) lands exactly there.
    y_lo = _cluster_both(ny_lo + 1, -S, 0.0)[:-1]
    y_up = _cluster_both(ny_up, 0.0, H_IN)

    z = np.arange(nz) / nz * L_SPAN                          # periodic: far endpoint not stored

    nx_tot, ny_tot = nx_in + nx_re + nx_rc, ny_lo + ny_up
    hx, hy, hz = 1.0 / (nx_tot - 1), 1.0 / (ny_tot - 1), 1.0 / nz

    def mk(xs, ys):
        X, Y, Z = np.meshgrid(xs, ys, z, indexing="ij")
        b = Block((len(xs), len(ys), nz), X, Y, Z, (hx, hy, hz),
                  period=(1.0, 1.0, L_SPAN))       # span is 4S, NOT the default 1
        b.faces[face_id(2, 0)] = b.faces[face_id(2, 1)] = "periodic"
        return b

    blocks = [mk(x_in, y_up), mk(x_re, y_up), mk(x_re, y_lo),
              mk(x_rc, y_up), mk(x_rc, y_lo)]

    W = "wall"
    blocks[INLET].faces[face_id(0, 0)] = W        # inflow (profile prescribed by the caller)
    blocks[INLET].faces[face_id(1, 0)] = W        # step TOP wall, upstream of the step
    blocks[INLET].faces[face_id(1, 1)] = W        # top wall
    blocks[RECIRC_U].faces[face_id(1, 1)] = W     # top wall
    blocks[RECIRC_L].faces[face_id(0, 0)] = W     # the STEP FACE
    blocks[RECIRC_L].faces[face_id(1, 0)] = W     # bottom wall
    blocks[RECOV_U].faces[face_id(1, 1)] = W      # top wall
    blocks[RECOV_U].faces[face_id(0, 1)] = W      # outflow
    blocks[RECOV_L].faces[face_id(1, 0)] = W      # bottom wall
    blocks[RECOV_L].faces[face_id(0, 1)] = W      # outflow

    conns = [
        Connection(INLET,    face_id(0, 1), RECIRC_U, face_id(0, 0)),   # inlet  -> recirc (x)
        Connection(RECIRC_U, face_id(0, 1), RECOV_U,  face_id(0, 0)),   # recirc -> recovery (x)
        Connection(RECIRC_L, face_id(0, 1), RECOV_L,  face_id(0, 0)),   # ditto, lower row
        Connection(RECIRC_L, face_id(1, 1), RECIRC_U, face_id(1, 0)),   # lower -> upper (y)
        Connection(RECOV_L,  face_id(1, 1), RECOV_U,  face_id(1, 0)),   # ditto, downstream
    ]
    return Domain(blocks, conns)


if __name__ == "__main__":
    d = bfs5_domain()
    print(f"  Armaly BFS, FIVE domains.  ER = {H_TOT / H_IN:.4f} (Armaly 1.9423)")
    print(f"  inlet {L_IN:.0f}S | recirculation to {X1:.0f}S | recovery to {L_OUT:.0f}S | "
          f"span {L_SPAN:.0f}S periodic")
    print(f"  {len(d.blocks)} blocks, {d.n_cells} cells, {len(d.connections)} connections\n")
    probs = d.validate()
    print(f"  validate(): {len(probs)} problem(s)")
    for p in probs[:6]:
        print(f"      {p[:110]}")
    print()
    for b, nm in enumerate(NAMES):
        blk, (J, _) = d.blocks[b], d.block_metrics_cached(b)
        print(f"    {nm:>15}  {str(blk.shape):>14}  "
              f"x [{blk.x.min():+7.3f},{blk.x.max():+7.3f}]  "
              f"y [{blk.y.min():+.4f},{blk.y.max():+.4f}]  min J {J.min():.3e}")
    print(f"\n  step face at x = {d.blocks[RECIRC_L].x.min():+.4f} (wants exactly 0)")
    print(f"  step top  at y = {d.blocks[INLET].y.min():+.4f} (wants exactly 0)")
