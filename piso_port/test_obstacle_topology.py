"""
The 8-block obstacle topology -- the vortex-street layout, and the first mesh here with a HOLE.

Everything before this is a strip, a 2x2, or a channel: topologies where every block has the
same padding along every tangential axis. An obstacle breaks that. Block LB pads its x-axis into
MB, but its y-neighbour LM cannot, because LM's +x IS the obstacle surface -- so the two blocks
either side of that connection carry different tangential padding, and the ghost LB needs
diagonally beyond both axes would have to come from inside the solid body.

Three bugs surfaced only because this topology was tried, each of which produced a
plausible-looking field rather than an obvious failure:

  1. Ghost layers must have only their MISMATCH in padding reconciled. Trimming to the core and
     re-padding with edge replication discards the neighbour's real ghost values wherever both
     blocks were padded -- silently corrupting every seam in a uniform topology (2x2 metrics
     went to 1.4e-01).
  2. `_fit_face` re-entered `upto()` for the block currently being built.
  3. THE FACE COEFFICIENT MUST COME FROM THE SAME SOURCE THE MATRIX USES. build_diffusion_matrix
     forms it from each block's OWN metrics; pressure_face_fluxes was recomputing it from padded
     COORDINATES, which are extrapolated at a reentrant corner. The CG solve converged happily
     to 9e-11 while the corrected flux divergence sat at 1.2e-01, because no pressure field can
     make an inconsistent pair of operators agree.
"""
import numpy as np, sys, warnings; warnings.filterwarnings("ignore")
from src.multiblock import Block, Connection, Domain, face_id, FACE_NAMES
from src.piso_multiblock import MultiBlockPISO

# 8 blocks around a square hole -- the vortex-street topology.
#
#     TL | TM | TR          columns L,M,R with DIFFERENT widths (non-uniform blocks)
#     ---+----+---          rows    B,M,T with different heights
#     ML |HOLE| MR
#     ---+----+---
#     BL | BM | BR
#
# Outer: -x inflow, +x outflow, -y/+y walls, z periodic.
# Hole:  ML's +x, MR's -x, BM's +y, TM's -y are all WALLS (the obstacle surface).
NXc = (6, 4, 8)          # column node counts -- deliberately unequal
NYr = (5, 4, 5)          # row node counts
NZ  = 4
L, H = 3.0, 2.0
Nx, Ny = sum(NXc), sum(NYr)
xs = np.linspace(0, L, Nx)
ys = np.linspace(0, H, Ny)
zs = np.arange(NZ) / NZ
cx = [slice(0, NXc[0]), slice(NXc[0], NXc[0]+NXc[1]), slice(NXc[0]+NXc[1], Nx)]
cy = [slice(0, NYr[0]), slice(NYr[0], NYr[0]+NYr[1]), slice(NYr[0]+NYr[1], Ny)]
h = (L/(Nx-1), H/(Ny-1), 1.0/NZ)

names = {}
blocks = []
for j, ry in enumerate("BMT"):
    for i, rx in enumerate("LMR"):
        if i == 1 and j == 1:
            continue                      # the hole
        X, Y, Z = np.meshgrid(xs[cx[i]], ys[cy[j]], zs, indexing="ij")
        blk = Block((NXc[i], NYr[j], NZ), X, Y, Z, h)
        blk.faces[face_id(2,0)] = blk.faces[face_id(2,1)] = "periodic"
        names[rx+ry] = len(blocks); blocks.append(blk)

C = []
# x-direction connections, per row
for j, ry in enumerate("BMT"):
    for i in range(2):
        a, b = "LMR"[i]+ry, "LMR"[i+1]+ry
        if a in names and b in names:
            C.append(Connection(names[a], face_id(0,1), names[b], face_id(0,0)))
# y-direction connections, per column
for i, rx in enumerate("LMR"):
    for j in range(2):
        a, b = rx+"BMT"[j], rx+"BMT"[j+1]
        if a in names and b in names:
            C.append(Connection(names[a], face_id(1,1), names[b], face_id(1,0)))
d = Domain(blocks, C)

print(f"  {len(blocks)} blocks around a hole, {d.n_cells} cells, {len(C)} connections")
print(f"  column widths {NXc}, row heights {NYr}  <- NON-UNIFORM block sizes")
probs = d.validate()
print(f"  validate(): {len(probs)} problem(s)" + (f"\n    {probs[0][:130]}" if probs else ""))
print(f"  wall nodes: {d.wall_mask().sum()} of {d.n_cells}")
# which faces ended up as walls (the obstacle surface + outer boundary)
for nm, b in sorted(names.items()):
    w = [FACE_NAMES[f] for f, k in enumerate(blocks[b].faces) if k not in ("periodic","connected")]
    print(f"    {nm}: walls {w}")

# --- can it actually step? inflow on the left column, convective outflow on the right
import time
U0, NU, DT = 1.0, 1.0/100.0, 0.01
m = MultiBlockPISO(d, NU, DT, 2, 1e-10, time_scheme='bdf2', scheme='rotational', picard_iters=2)
for nm, b in names.items():
    m.u[b][:] = U0                                  # uniform stream as the initial condition
    if nm.startswith("L"):                          # inflow face
        m.u_bc[b][0, :, :] = U0
    for f, k in enumerate(blocks[b].faces):
        if k in ("periodic", "connected"):
            continue
        ax, sd = f // 2, f % 2
        sl = [slice(None)]*3; sl[ax] = 0 if sd == 0 else -1
        if not (nm.startswith("L") and f == face_id(0,0)):
            m.u_bc[b][tuple(sl)] = 0.0              # no-slip on walls and the obstacle
            m.u[b][tuple(sl)] = 0.0
        else:
            m.u[b][tuple(sl)] = U0
m.outflow = [(names[n], face_id(0,1), U0) for n in ("RB","RM","RT")]
for n in ("RB","RM","RT"):
    m.u_bc[names[n]][-1, :, :] = U0
print()
t0 = time.time()
try:
    for it in range(20):
        div = m.step()
        if not all(np.isfinite(m.u[b]).all() for b in range(len(blocks))):
            print(f"  step {it+1}: NON-FINITE"); break
    umax = max(np.abs(m.u[b]).max() for b in range(len(blocks)))
    print(f"  20 PISO steps on the obstacle topology: OK  [{time.time()-t0:.1f}s]")
    print(f"    max|u| {umax:.4f} (inflow {U0})   final flux divergence {div:.2e}")
    fixed, free = d.boundary_flux_totals(m.u, m.v, m.w,
                                         [(names[n], face_id(0,1)) for n in ("RB","RM","RT")])
    print(f"    boundary flux: in {-fixed:+.5f}  out {free:+.5f}  imbalance {fixed+free:+.2e}")
except Exception as e:
    import traceback; traceback.print_exc()

ok = (len(probs) == 0 and div < 1e-9 and abs(fixed + free) < 1e-12
      and all(np.isfinite(m.u[b]).all() for b in range(len(blocks))))
print()
print(f"  [{'PASS' if ok else 'FAIL'}] obstacle topology runs, conserves mass, and projects cleanly")
print(f"          validate {len(probs)} problems | flux divergence {div:.2e} "
      f"| mass imbalance {fixed+free:+.1e}")
sys.exit(0 if ok else 1)
