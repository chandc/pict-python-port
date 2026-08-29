# Multi-block coupling: offsets, indexing, and the connection map

How PICT joins several blocks into one solver. Written from the C++/CUDA source as
preparation for extending this single-block port; every claim cites the file and line it
came from.

**The headline, because it determines everything else: PICT has no ghost cells.** Blocks are
joined by (a) a *global index space* so that all blocks share one sparse matrix, and (b) a
*connection map* that lets a cell read its neighbour directly out of the adjacent block's own
array. Inter-block coupling is therefore **implicit** — it lives in the matrix, not in an
outer iteration.

---

## 1. Why not ghost cells

| | ghost cells (explicit) | one global matrix (PICT) |
|---|---|---|
| coupling | lagged one iteration | exact, inside the solve |
| pressure Poisson | Schwarz iteration; convergence degrades as blocks are added | single CG solve, block-count independent |
| per-block code | simpler | needs global indexing |
| adjoint | reintroduces a fixed point to differentiate | one `LinearSolve`, existing adjoint applies |

Pressure is globally elliptic: a disturbance anywhere is felt everywhere immediately. Lagging
the interface turns one elliptic solve into an iteration between subdomains, and the iteration
count grows with the number of blocks. That is the case against ghost cells for the pressure
equation, and it is presumably why PICT went implicit.

---

## 2. The two offsets

Assigned once, in `Domain::...` (`domain_structs.cpp:2562-2569`):

```cpp
index_t csrSize = 0;  totalSize = 0;
for (auto block : blocks) {
    block->csrOffset    = csrSize;      // start of this block's entries in the CSR value array
    block->globalOffset = totalSize;    // start of this block's cells in the global cell numbering
    csrSize   += block->ComputeCSRSize();
    totalSize += block->getStrides().w; // = nx*ny*nz for this block
}
```

Two different offsets because a CSR matrix has two different arrays to index into.

### 2a. `globalOffset` — the cell (row) numbering

Blocks are numbered consecutively. If block $b$ has $N_b = n_x n_y n_z$ cells:

$$\texttt{globalOffset}_b = \sum_{b' < b} N_{b'}, \qquad
\text{global row of local cell } f \;=\; \texttt{globalOffset}_b + f$$

```
        block 0 (N0 = 64)      block 1 (N1 = 48)     block 2 (N2 = 96)
      ┌───────────────────┬──────────────────┬────────────────────────┐
rows  │ 0 ............ 63 │ 64 .......... 111│ 112 ............... 207│
      └───────────────────┴──────────────────┴────────────────────────┘
        globalOffset = 0     globalOffset = 64   globalOffset = 112
```

Used directly when writing rows — `PISO_multiblock_cuda_kernel.cu:3657`:

```cpp
s_domain.C.row[s_block.globalOffset + flatPos + 1] = row.endOffset + s_block.csrOffset;
```

The local flat index itself is the ordinary lexicographic one
(`grid_definitions.h`, `flattenIndex`):

$$f = x + s_y\,y + s_z\,z, \qquad s_y = n_x,\; s_z = n_x n_y$$

### 2b. `csrOffset` — the value/column-index array

Rows have **different lengths**, so the offset into the value array is *not*
`globalOffset × rowsize`. It accumulates the actual entry counts
(`domain_structs.cpp:2151`):

$$\texttt{ComputeCSRSize}(b) \;=\; \underbrace{N_b\,(2d+1)}_{\text{full stencil everywhere}}
\;-\; \sum_{\text{faces }F \text{ unconnected}} A_F$$

where $d$ is the spatial dimension and $A_F$ is the face's cell count. Each cell would have
$2d+1$ entries (itself plus two neighbours per axis); a cell sitting on an **unconnected**
boundary has no neighbour that way, so one entry is subtracted per such cell.

```
  a 4x3 block (d = 2, full stencil 2d+1 = 5),  -x and +y CLOSED,  +x and -y CONNECTED

                 entries per cell
                ┌───┬───┬───┬───┐
   +y closed →  │ 3 │ 4 │ 4 │ 4 │   +y row loses one; the corner loses two
                ├───┼───┼───┼───┤
                │ 4 │ 5 │ 5 │ 5 │   interior cells keep all 5
                ├───┼───┼───┼───┤
                │ 4 │ 5 │ 5 │ 5 │
                └───┴───┴───┴───┘
                  ↑
                -x closed (this column loses one)

   sum = 53   =   N(2d+1) - A(-x) - A(+y)   =   12*5 - 3 - 4   =   53
```

**Connected faces are not subtracted** — that is the whole point. A cell on a connected face
keeps its full stencil, and the missing neighbour is supplied by the adjacent block. The
implicit coupling is exactly those retained entries.

### 2c. Row start and length

`getCSRMatrixRowEndOffsetFromBlockBoundaries3D`
(`PISO_multiblock_cuda_kernel.cu:252-312`) returns, for one cell, its row length and the
running end-offset. It starts from the dense assumption and subtracts:

$$\texttt{rowEndOffset} = (f+1)(2d+1) - \!\!\sum_{\text{closed faces}}\!\! (\text{cells before and including } f \text{ that touch that face})$$

The per-axis subtractions are closed-form counts, not loops — e.g. for the $-x$ face, the
number of cells up to and including $f$ that lie on it is $\lfloor f/n_x\rfloor + 1$:

```cpp
if (isEmptyBound(0, block.boundaries)) {              // -x closed
    rowEndOffset -= (flatPos / block.size.x) + 1;
    if (pos.x == 0) --rowSize;
}
```

and the row start is then

$$\texttt{rowStart} = \texttt{rowEndOffset} - \texttt{rowSize} + \texttt{csrOffset}$$

Note `isEmptyBound` (`:188`) counts **Dirichlet, varying-Dirichlet and gradient** boundaries as
"empty". `CONNECTED_GRID` and `PERIODIC` are *not* empty — they keep their entry.

---

## 3. The connection map: `axes`

A `ConnectedBoundary` stores only two things
(`domain_structs_gpu.h`, `ConnectedBoundaryGPU`):

```cpp
struct ConnectedBoundaryGPU {
    index_t connectedGridIndex;   // which block
    U4      axes;                 // how the two blocks are oriented relative to each other
};
```

### 3a. The bit encoding

Faces and axes share one encoding (`PISO_multiblock_cuda_kernel.cu:199-221`):

$$\texttt{bound} = 2\,a + u, \qquad
a = \texttt{bound} \gg 1 \;(\text{axis}), \qquad
u = \texttt{bound} \,\&\, 1 \;(0 = \text{lower},\, 1 = \text{upper})$$

```
   bound:   0     1     2     3     4     5
   meaning -x    +x    -y    +y    -z    +z
   axis     0     0     1     1     2     2
   upper    0     1     0     1     0     1
```

`axes.a[k]` uses the same layout, but there the low bit means **the connection is reversed
along that direction**, not "upper face" (`:322`).

`axes.a[0]` describes the *connection axis* — which of the neighbour's axes is normal to the
shared face, and at which end. `axes.a[1]` and `axes.a[2]` describe the two transverse
directions, **relative to the boundary axis**, via
$\;\texttt{getAxisRelativeToOther}(a, a_\text{bnd}, d) = (a - a_\text{bnd}) \bmod d\;$ (`:243`).

That relative numbering is what lets one encoding serve all six faces.

### 3b. Reading the neighbour

`computeConnectedPos` (`:327-346`) maps a local cell position to the corresponding position in
the neighbour's own array:

```cpp
index_t connectedAxis = p_cb->axes.a[0] >> 1;
connectedPos.a[connectedAxis] = (p_cb->axes.a[0] & 1)
      ? p_connectedBlock->size.a[connectedAxis] - 1 - borderOffset    // attach at the upper end
      : borderOffset;                                                 // attach at the lower end

for (k = 1, 2) {                                    // the two transverse directions
    axis          = (boundaryDim + k) % numDims;    // local transverse axis
    connectedAxis = p_cb->axes.a[k] >> 1;           // which neighbour axis it maps to
    connectedPos.a[connectedAxis] = (p_cb->axes.a[k] & 1)
          ? p_connectedBlock->size.a[connectedAxis] - 1 - pos.a[axis] // reversed
          : pos.a[axis];                                             // aligned
}
```

In words, with $\pi$ the axis permutation and $r_k \in \{0,1\}$ the reversal flags:

$$q_{\pi(0)} = \begin{cases} 0 & r_0 = 0\\ m_{\pi(0)}-1 & r_0 = 1\end{cases}
\qquad
q_{\pi(k)} = \begin{cases} p_{a_k} & r_k = 0\\ m_{\pi(k)} - 1 - p_{a_k} & r_k = 1\end{cases}$$

for $k = 1,2$, where $p$ is the local position, $q$ the neighbour position, $m$ the neighbour's
sizes, and $a_k = (a_\text{bnd}+k) \bmod d$.

```
  aligned (r1 = 0)                    reversed (r1 = 1)

  block A        block B              block A        block B
  j=0 ─────────── j'=0                j=0 ───────┐   j'=3
  j=1 ─────────── j'=1                j=1 ─────┐ └── j'=2
  j=2 ─────────── j'=2                j=2 ───┐ └──── j'=1
  j=3 ─────────── j'=3                j=3 ─┐ └────── j'=0
        shared face                            shared face
```

The permutation handles blocks meeting with different axis orders (an L-shaped or O-shaped
arrangement); the flips handle blocks meeting with opposite orientation.

### 3c. Sign flips on vector quantities

Positions permute, but **vector components must also be signed**. In `computeFluxesNDLoop`
(`:1553`) a flux read from a connected block is negated when both sides attach at the same
end:

```cpp
const bool otherIsUpper = boundIsUpper(block.boundaries[bound].cb.axes.a[0]);
if (otherIsUpper == isUpper) velN = -velN;   // upper-to-upper or lower-to-lower: invert
fluxes[bound] = (velN + velC) * 0.5f;
```

If A's *upper* face meets B's *upper* face, the two outward normals point at each other, so
the contravariant component measured in B has the opposite sign in A's frame. Getting this
wrong does not crash — it quietly leaks mass at the interface.

---

## 4. Where it lands in the matrix

`PISO_build_pressure_matrix` (`:4841-4852`) treats a connected neighbour exactly like an
interior one, only fetching its coefficient from the other block:

```cpp
if (atBound && s_block.boundaries[bound].type == BoundaryType::CONNECTED_GRID) {
    p_block = s_domain.blocks + s_block.boundaries[bound].cb.connectedGridIndex;
    tempPos = computeConnectedPosWithChannel(tempPos, dim, &s_block.boundaries[bound].cb, s_domain);
}
const scalar_t alphaN = getLaplaceCoefficientOrthogonalDimSwitch(tempPos, p_block, s_domain.numDims);
```

and the column index is that neighbour's **global** index. So the assembled matrix looks like

$$M \;=\;
\begin{pmatrix}
M_{00} & C_{01} & \\
C_{01}^{\mathsf T} & M_{11} & C_{12}\\
& C_{12}^{\mathsf T} & M_{22}
\end{pmatrix}$$

— block-diagonal per block, plus off-diagonal coupling blocks $C$ carrying exactly the
retained face entries from §2b. **Symmetry survives** provided both sides agree on the shared
face coefficient, which is the one physical requirement the implementation must guarantee.

---

## 5. What this port would need

| Piece | Effort | Risk |
|---|---|---|
| `globalOffset` threading through the assemblers | mechanical, pervasive | low — touches every builder |
| `csrOffset` / variable row lengths | moderate | low, and testable in isolation |
| `axes` permutation + flips | small in code | **highest** — silent wrong answers |
| Interface metric agreement | small | **high** — silent mass leak |
| Flux sign inversion | small | **high** — silent mass leak |

Two things to test first, before any physics:

1. **Connect a block to itself** across a periodic pair. The result must reproduce the existing
   single-block periodic case *exactly*, to machine precision. That validates offsets,
   `axes`, and sign handling in one shot against a known answer.
2. **Assert interface symmetry**: build the global matrix and check `abs(M - M.T).max() == 0`
   as the single-block path already does. An interface metric mismatch shows up there
   immediately, rather than as a divergence floor twenty steps into a run.

### Interaction with the deferred correction

Worth flagging before starting: on warped grids the pressure deferred correction already needs
10–320 sweeps, and each is a **global** solve across all blocks. Multi-block multiplies the
existing performance cliff. Making the cross terms implicit (a 19-point stencil, measured 27×
faster and 24× lighter for the adjoint) would be worth sequencing *before* multi-block rather
than after.

---

## Implementation status (foundation landed)

`multiblock.py` provides the index and connection layer; `test_multiblock.py` gates it (10/10).
Nothing in it is imported by the single-block path — a `Domain` of one block with no connections
*is* the existing solver, unchanged.

### The constraint our grid adds that PICT's does not

PICT is cell-centred, so cells are distinct however blocks meet and a connection is simply a
face. **Our nodes sit ON boundaries**, so if both blocks stored their interface nodes those nodes
would be *duplicated* and the resolution would halve across the seam — the same failure
`make_grid`'s docstring warns about for periodic axes. A connected face must therefore follow the
**periodic** node rule: store up to but not including the interface, and let the neighbour supply
the next node.

`Domain.validate()` enforces this by checking the coordinates directly — the two blocks'
interface nodes must not coincide. An earlier version checked the *spacing* against $1/n$ and was
wrong: a connected axis's spacing is set by the **global** cell count across all blocks in that
direction, not by one block's count, so a correct 4-way split was rejected.

### Orientation is a permutation plus explicit flips, not signed axes

PICT encodes the connection map with signed axis indices, the sign meaning "inverted". That
cannot express *flip axis 0* in Python, because `-0 == 0`, so a signed encoding silently loses
one of the eight orientations. `Connection` therefore carries `axes=(0,1)` and
`flips=(False,False)` separately. All eight combinations are round-trip tested.

### The test that earns its keep: split-equals-whole

Take a domain that is genuinely one block, cut it into 2 and 4 pieces, and require the
multi-block machinery to reproduce the single-block answer **exactly**. The unsplit run *is* the
reference, so no external data is needed, and every orientation, index-offset and seam-spacing
bug fails it loudly. Currently verified: the neighbour set (192 pairs, identical) and the
coordinates (0.00e+00 difference).

### Seam geometry: landed and exact

`Domain.pad_coords` ghost-pads a block from its neighbours across the connection map, and
`Domain.block_metrics` computes that block's Jacobian and metrics from the padded coordinates.
On a **warped** grid split 2 and 4 ways, the per-block metrics equal the single-block metrics to
**0.00e+00** — every one of the nine metric tensors and the Jacobian — and the reassembled
domain satisfies the GCL at **1.8e-15**.

Two bugs it caught, both of which produce a plausible-looking field rather than an obvious
failure:

- **The periodic self-wrap needs a one-period shift.** Coordinates are not periodic — $x$ ramps
  and jumps back — so a wrapped ghost must be displaced by one period or it injects a spurious
  derivative. This is the *same* defect `compute_numerical_metrics` had when it hardcoded
  `period=1`; here the period is per-block data rather than an assumption.
- **Both sides of an axis must be read before either is attached.** Padding the lower side and
  then reading the upper side from the modified array makes the upper ghosts a copy of the lower
  ones. Silent, and it corrupts the Jacobian at both ends.

A third was in the *test*, not the code: checking the GCL block-by-block with
`periodic=(True,...)` wraps each block onto **itself** at a face that is really connected to a
neighbour, measuring a seam that does not exist and reporting ~1e-1. GCL must be checked on the
reassembled domain until a genuinely seam-aware divergence exists.

**Current limitation:** at most one connected axis per block. Two would need the neighbour's
coordinates themselves padded along the other connected axis (recursive padding).
`validate()` reports this rather than silently mis-padding the corners.

### Global assembly and solve: landed and exact

`Domain.build_diffusion_matrix` assembles the conservative diffusion operator over the whole
domain as **one sparse matrix** — PICT's design, and the reason no ghost cells are needed: a
connection contributes off-diagonal entries between the two blocks' boundary cells exactly as an
interior face does within a block, so the coupling is implicit in the linear solve rather than
exchanged between steps.

| split | vs single-block matrix | nnz | asymmetry | vs single-block **solve** |
|---|---|---|---|---|
| 2 blocks | 5.68e-14 | 1344 = 1344 | 0.0 | **3.12e-17** |
| 4 blocks | 5.68e-14 | 1344 = 1344 | 0.0 | **1.73e-17** |

Faces are enumerated **once each**: interior faces, one wrap per periodic axis, and one face per
connection added **from the A side only**. Adding it from both ends would double the coupling and
quietly halve the effective diffusion across every seam.

**The decoupling trap, demonstrated rather than warned about.** Dropping the connections — which
is what a block-diagonal preconditioner effectively does — leaves the blocks independent. The
solve still converges, reports no error, and is wrong by **1.3e+16**. `test_multiblock.py`
includes that case, so the split-equals-whole gate is known to have teeth rather than assumed to.

### MMS Poisson across seams: the number of blocks is irrelevant

A manufactured-solution Poisson problem solved on 1, 2 and 4 blocks gives **byte-identical**
errors — 1.8749e-02 at $n=8$ and 4.5788e-03 at $n=16$, order **2.03** — so the decomposition has
no effect whatever on the answer, which is what a correct multi-block implementation owes.

**A block joined to ITSELF is a full-period wrap.** With one block whose `+x` meets its own
`-x`, the connection still crosses a whole period and needs the period shift exactly as a
two-block wrap does. Omitting it made the solve **9.9× worse** and dropped it to first order —
and `validate()` could not see it, because the interface nodes are genuinely distinct, so the
duplicated-node check has nothing to catch. It is gated by the MMS test instead. Worth noting as
a class: some seam errors are invisible to structural checks and only a *solve* reveals them.

### Field padding, and the decision to use IMPLICIT cross terms

`Domain.pad_field` ghost-pads a scalar field across seams using the same connection machinery as
`pad_coords`, with one critical difference: **no period shift**. Coordinates ramp and jump back,
so a wrapped coordinate ghost must be displaced by one period; velocity and pressure are
genuinely periodic and must not be. Copying the coordinate path verbatim would offset every seam
by exactly one period — large, smooth and entirely plausible-looking. Verified exact (0.00e+00)
against the single-block field, and the trap is gated.

`width` must cover the widest stencil to be applied: central needs 1, **SOU reaches $i-2$ and
needs 2**. Padding with 1 and then applying SOU degrades the upwind stencil to first order *at
seams only*, which no smooth test would reveal.

**Cross terms will be treated IMPLICITLY in multi-block, not by deferred correction.** The
deferred path would need cross-term fluxes exchanged across every seam on every Picard sweep —
the fiddliest piece of the whole build. The implicit path instead applies the cross operator
per-block on padded fields using the already-verified `pressure_face_fluxes`, with the global
7-point matrix (assembled exactly, above) as the preconditioner. That reuses verified code
rather than requiring a new seam-aware 19-point assembler.

### Seam-aware fluxes and divergence: landed and exact

`Domain.face_fluxes` resolves connected faces from real neighbour data, so a seam is an ordinary
interior face. This is what `compute_face_fluxes` cannot do block-locally: it treats every
non-periodic face as a **domain boundary**, so a connection would receive a *prescribed* flux
rather than an interpolated one — injecting or losing mass at every seam.

Against the single-block result on a warped grid, split 2 and 4 ways:

| | 2 blocks | 4 blocks |
|---|---|---|
| max \|flux − single\| | 1.11e-16 | 1.11e-16 |
| max \|div − single\| | 9.44e-16 | 9.44e-16 |

**An API flaw fixed on the way.** `pad_field` originally read the neighbour's data from a cached
`set_fields({block: array})`. That holds ONE field per block, so padding $u$, $v$ and $w$ in turn
would read whichever component had been registered last across every seam — a silent, smooth
corruption. The field for every block is now passed explicitly and passing a bare array raises
rather than guessing.

### Still to build

1. Block-aware **face-type registry** — one block's `+x` can be wall, inlet, outflow *or*
   connection. `outflow.py`'s `Outflow(axis, side, ...)` is a first step but covers outflow only.
2. **Metrics across a seam** — ghost padding must come from the neighbour's coordinates with the
   orientation transform applied, or the Jacobian collapses at the interface. The GCL check
   catches this.
3. **Global solvers** — one pinned cell for the whole domain, not one per block; cross-term
   fluxes must cross interfaces; a *block-diagonal* preconditioner would silently decouple the
   blocks and converge to the wrong answer.
4. **Adjoint** — `LinearSolve` transposes one global matrix so it should carry over, but the
   connection map contributes to `dL/dA` and that path is unexercised.
