"""
The multi-block O-grid around a circular cylinder -- the vortex-street mesh, coloured by block.

An O-GRID rather than the H-grid used in test_obstacle_topology.py, and deliberately: an H-grid
around a square body has REENTRANT CORNERS, where the two blocks either side of a connection
carry different tangential padding and the diagonal ghost lies inside the solid. An O-grid has
none -- every block is connected to its two azimuthal neighbours around a closed ring, and the
only boundaries are the cylinder wall and the far field.

Node placement follows the rule the connections require: the azimuthal direction closes on
itself, so it is partitioned WITHOUT duplicating the seam nodes (like a periodic axis). The
radial direction has real boundaries at both ends -- the cylinder surface and the far field --
so it keeps both endpoints.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)          # figures/ and results/ paths are relative to the root

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.multiblock import Block, Connection, Domain, face_id

R_CYL, R_OUT, NBLK = 0.5, 6.0, 4
NR, NTH_TOT, NZ = 24, 64, 8
STRETCH = 1.13                      # geometric radial stretching away from the wall


def cylinder_domain(nblk=NBLK):
    dr = np.array([STRETCH ** i for i in range(NR - 1)])
    r = np.concatenate([[0.0], np.cumsum(dr)])
    r = R_CYL + (R_OUT - R_CYL) * r / r[-1]
    th_all = np.linspace(0.0, 2 * np.pi, NTH_TOT, endpoint=False)   # ring: no duplicate seam
    z = np.linspace(0.0, 2.0, NZ)
    nth = NTH_TOT // nblk
    blocks = []
    for b in range(nblk):
        th = th_all[b * nth:(b + 1) * nth]
        Rg, Tg, Zg = np.meshgrid(r, th, z, indexing="ij")
        blk = Block((NR, nth, NZ), Rg * np.cos(Tg), Rg * np.sin(Tg), Zg,
                    (1.0 / (NR - 1), 1.0 / NTH_TOT, 1.0 / (NZ - 1)))
        blk.faces[face_id(0, 0)] = "wall"        # cylinder surface
        blk.faces[face_id(0, 1)] = "wall"        # far field (inflow/outflow set on it)
        blk.faces[face_id(2, 0)] = blk.faces[face_id(2, 1)] = "wall"
        blocks.append(blk)
    # azimuthal ring: no period shift -- the ring closes in PHYSICAL space, unlike a periodic
    # box where the coordinate jumps by one box length
    conns = [Connection(b, face_id(1, 1), (b + 1) % nblk, face_id(1, 0)) for b in range(nblk)]
    return Domain(blocks, conns)


if __name__ == "__main__":
    d = cylinder_domain()
    probs = d.validate()
    print(f"  {len(d.blocks)} blocks, {d.n_cells} cells, {len(d.connections)} connections")
    print(f"  validate(): {len(probs)} problem(s)" + (f"  {probs[0][:90]}" if probs else ""))
    print(f"  wall nodes {d.wall_mask().sum()} of {d.n_cells}")
    Jmin = min(d.block_metrics_cached(b)[0].min() for b in range(len(d.blocks)))
    print(f"  min(J) over all blocks: {Jmin:.4e}  {'valid' if Jmin > 0 else 'TANGLED'}")

    cols = ["#e4572e", "#17bebb", "#ffc914", "#2e282a", "#76b041", "#a26769"]
    fig = plt.figure(figsize=(16.5, 7.5))

    # ---------------------------------------------------------------- 3D, coloured by block
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    # Near field only. The full domain is 12 across and 2 deep, so drawing all of it renders as
    # a flat smear with the mesh invisible; the interesting structure is at the cylinder.
    RVIEW = 2.2
    for b, blk in enumerate(d.blocks):
        c = cols[b % len(cols)]
        x, y, z = blk.x, blk.y, blk.z
        rr = np.sqrt(x[:, 0, 0] ** 2 + y[:, 0, 0] ** 2)
        imax = int(np.searchsorted(rr, RVIEW))
        for k in (0, blk.shape[2] - 1):
            for i in range(0, imax, 2):
                ax.plot(x[i, :, k], y[i, :, k], z[i, :, k], color=c, lw=0.7, alpha=0.9)
            for j in range(blk.shape[1]):
                ax.plot(x[:imax, j, k], y[:imax, j, k], z[:imax, j, k], color=c, lw=0.7,
                        alpha=0.9)
        for j in range(0, blk.shape[1], 3):          # spanwise lines on the cylinder surface
            ax.plot(x[0, j, :], y[0, j, :], z[0, j, :], color=c, lw=1.1)
    th = np.linspace(0, 2 * np.pi, 200)
    for zz in (0.0, 2.0):
        ax.plot(R_CYL * np.cos(th), R_CYL * np.sin(th), zz * np.ones_like(th),
                color="k", lw=2.4)
    ax.set_xlim(-RVIEW, RVIEW); ax.set_ylim(-RVIEW, RVIEW); ax.set_zlim(0, 2)
    ax.set_box_aspect((1, 1, 0.55))
    ax.grid(False)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_visible(False)
        a._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2]); ax.set_zticks([0, 1, 2])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"near field (r < {RVIEW}), coloured by block\n"
                 f"black = cylinder surface; z is the spanwise direction", fontsize=10)
    ax.view_init(elev=24, azim=-62)

    # ---------------------------------------------------------------- cross-section
    ax = fig.add_subplot(1, 2, 2)
    for b, blk in enumerate(d.blocks):
        c = cols[b % len(cols)]
        x, y = blk.x[:, :, 0], blk.y[:, :, 0]
        for i in range(blk.shape[0]):
            ax.plot(x[i], y[i], color=c, lw=0.5, alpha=0.9)
        for j in range(blk.shape[1]):
            ax.plot(x[:, j], y[:, j], color=c, lw=0.5, alpha=0.9)
        # label each block at its centroid
        ax.text(x.mean(), y.mean(), f"block {b}", color=c, fontsize=11, weight="bold",
                ha="center", va="center",
                bbox=dict(fc="w", ec=c, alpha=0.85, boxstyle="round,pad=0.25"))
    ax.add_patch(plt.Circle((0, 0), R_CYL, fc="0.85", ec="k", lw=2.2, zorder=5))
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("z-section: azimuthal seams are ordinary interior faces\n"
                 "(no reentrant corners -- the reason for an O-grid)", fontsize=10)

    fig.suptitle(f"Multi-block mesh for flow past a cylinder  "
                 f"({d.n_cells} cells, {len(d.connections)} connections, min J = {Jmin:.3f})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("figures/cylinder_blocks.png", dpi=145)
    print("wrote cylinder_blocks.png")
