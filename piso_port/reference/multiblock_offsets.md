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

### Global momentum matrix: landed and exact

`Domain.build_momentum_matrix` assembles $A = J/\Delta t + J\,\mathrm{conv} + \nu\,\mathrm{diff}$
across blocks as one matrix, matching the single-block assembler to **1.42e-14** with identical
nnz for 2- and 4-way splits. Unlike diffusion, convection is **not symmetric** (measured
asymmetry 7.55), so each face writes *different* values into the two rows it touches.

**Central convection only, and it raises rather than falling back.** SOU reaches $i-2$, so it
needs two ghost layers at a seam and a wider assembly than the verified 7-point machinery. A
silent fallback to central would change the physics, not merely the accuracy — SOU removes ~10%
of kinetic energy per turnover on a broadband field where central conserves it to round-off — so
`convection='sou'` raises `NotImplementedError`. That is also the right default for this work:
central is the scheme required for anything where dissipation matters.

### A full PISO step across blocks: landed

`piso_multiblock.MultiBlockPISO` runs a complete step across blocks and reproduces the
single-block trajectory to **7.4e-10** (BE) and **7.9e-10** (BDF2) after 10 steps, with flux
divergence at machine precision. Scope is fully periodic and Cartesian on purpose: that is the
configuration where an *exact* single-block comparison exists, with no cross terms and no
boundary conditions to confound it.

Three orchestration mistakes it had to get past, all global-vs-local:

- **The pressure pin is global** — one cell for the whole domain. One per block lets each block's
  pressure level float independently, and it still converges.
- **Γ comes from the global matrix.** `coef = J/rowsum(A)` must use the assembled global A, or
  seam rows get a coefficient computed as if their neighbours did not exist.
- **A units error in the pressure flux.** Deriving from the matrix,
  $\Phi = \tfrac12(Jg_P+Jg_N)(p_N-p_P)/h$ needs **one** division by $h$; two makes the
  correction $h^{-1}$ too large — a factor of 8 at $h=1/8$ — and the solution blows up.

And one self-inflicted detour worth recording: seeing a divergence of 9.5e-2 against the single
block's 8.9e-16, the corrector was rewritten to carry corrected fluxes forward. The single-block
loop actually **recomputes** the flux from the current velocities every corrector and only
*reports* the corrected flux's divergence at the end. The original code was right; the "fix"
made the velocity drift 1% while the divergence diagnostic looked perfect.

### Temporal order: BDF2 is carried, but two O(Δt) errors sit in front of it

Measured on the multi-block solver, and identical to single-block:

| configuration | measured order |
|---|---|
| chorin + BDF2 | 0.93, 0.89, 0.95 |
| rotational + BDF2, `picard_iters=1` | 1.32, 1.20, 1.08 |
| **rotational + BDF2, `picard_iters=2`** | **2.19, 2.16, 2.09** |

Two separate first-order errors cap the scheme before BDF2 can matter: **Chorin's splitting
error** (the non-incremental projection is first order regardless of the predictor) and the
**Picard lag** on the convecting velocity, since the momentum matrix is assembled from $u^n$.
One extra Picard iteration removes the second; a third buys nothing.

This also resolves an apparent conflict with `stokes_verification.md`, which measured order 2.00
at `picard_iters=1`. That test used a perturbation amplitude of 1e-4, so **convection was
negligible and the lag cost nothing**. With convection active it dominates. The second-order
claim therefore holds for near-linear flows and *not* for convectively driven ones unless
`picard_iters=2`.

**Both are now implemented across blocks**, and the multi-block solver recovers second order:

| multi-block, rotational + BDF2 | measured order |
|---|---|
| `picard_iters=1` | 1.32, 1.20, 1.08 |
| **`picard_iters=2`** | **2.19, 2.16, 2.09** |

identical to single-block, and the 4-block trajectory still matches the single-block one to
7.9e-10 with flux divergence 5.6e-16. `MultiBlockPISO` therefore defaults to
`scheme='rotational', picard_iters=2` — the configuration that actually delivers the design
order once convection is active.

Changing those defaults broke four existing gates, which had been comparing against a
chorin/`picard_iters=1` single-block run while relying on the constructor's defaults for the
multi-block side. They now state their configuration explicitly. A test that inherits a default
is a test that silently changes meaning when the default does.

### Multi-axis connections: recursive padding

`pad_coords` and `pad_field` are now **recursive**, so a block may be connected on any number of
axes. Padding axis 1 requires the neighbour's data **already padded along axis 0**, or the corner
ghosts are missing — and the array extents do not even match, so it fails loudly rather than
silently. `upto(bb, k)` returns block `bb` padded along the first `k` axes, memoised so the cost
is O(blocks × axes) rather than branching.

Verified on a **2×2 topology** — split in both $x$ and $y$, so every block is connected on two
axes, which a strip of blocks never exercises:

| | vs single block |
|---|---|
| metrics and Jacobian (warped) | **0.00e+00** |
| field padding incl. **corner** ghosts | **0.00e+00** |
| seam divergence | 9.44e-16 |

The corner check is the one that matters: it reads one step beyond in $x$ **and** $y$ at once,
which is precisely what the old one-axis implementation could not produce.

This was the hard blocker for real geometry. PICT's vortex-street sample uses 8 blocks around
the obstacle and every edge block is connected on two axes; the previous implementation would
have refused the topology outright.

> Written gate-first, as planned: the 2×2 test was written *before* the recursive padding and
> confirmed failing (4 validate problems, and `block_metrics` raising on the extent mismatch).
> A gate written after the fix only proves the fix runs.

### Walls: the face-type registry consumed

`Domain.wall_mask()` marks every node on a face that is **neither periodic nor connected**, and
`MultiBlockPISO` applies Dirichlet elimination on those rows of the global matrix — the same
treatment `phase4_poisson.py` uses single-block. Wall values are re-imposed after each pressure
correction.

Per-**face** typing is the point: one block's `+x` may be a wall while its neighbour's `+x` is a
connection, which per-*axis* periodicity flags cannot express. That is the whole reason real
geometry needs the registry.

Verified on a channel — no-slip walls in $y$, periodic in $z$, **connections in $x$**, so walls
and seams are exercised together:

| split | vs single block | wall nodes |
|---|---|---|
| 2 blocks | **6.84e-10** | 64 of 288 |
| 4 blocks | **6.84e-10** | 64 of 288 |

**A body force had to be added at the same time.** Without `velocity_source` a periodic channel
has nothing driving it and the velocity stays identically zero. The first wall test reported
`max|u - u_single| = 0.60`, exactly `max|u|` — the multi-block field was zero, and the omission
looked like a solver failure. There is now a gate asserting that a *forceless* channel stays at
rest, so the zero field is pinned as correct physics rather than mistaken for a bug again.

### Warped multi-block: the Cartesian-only limit removed

Two cross operators were needed, not one, and finding that took two rounds of debugging:

- **Pressure** — `Domain.pressure_face_fluxes(include_cross=True)`, solved implicitly with the
  exact global 7-point matrix as an ILU preconditioner. Matches the single-block cross flux to
  **2.7e-17**.
- **Momentum** — `Domain.cross_diffusion`, carried explicitly and iterated, exactly the deferred
  correction the single-block solver applies. It needs **two nested derivatives**, so the field
  is padded with `width=2`; padding with 1 leaves the outer derivative one-sided at every seam.

| split, warp 0.08 | vs single block | flux divergence |
|---|---|---|
| 2 blocks | **3.17e-11** | 1.4e-13 |
| 4 blocks | **3.17e-11** | 1.8e-13 |

**Two bugs, each of which produced a plausible field:**

1. The corrector subtracted an **orthogonal-only** flux while the pressure had been solved with
   the *full* operator. Inconsistent, so the corrected flux was not solenoidal — divergence
   3.2e-02 against the single-block 1.5e-13.
2. With that fixed the divergence was perfect (6.7e-14) and the velocity was still wrong by
   5.5e-02, because the **momentum** cross-diffusion was missing entirely. A divergence-free but
   incorrect velocity is exactly the failure mode a flux-divergence diagnostic cannot see.

### Coverage: does a skew grid work in both single and multi domain?

Yes, and this is measured rather than inferred from the parts. Multi-block reproduces the
single-block trajectory across warps, topologies, wall types and the production scheme
(rotational + BDF2 + Picard 2 + implicit cross):

| case | warp | vs single block | flux divergence |
|---|---|---|---|
| strip, 4 blocks | 0.08 / 0.12 / 0.15 | 4.9e-11 / 1.6e-10 / **2.4e-10** | ≤7.5e-13 |
| 2×2 topology (two connected axes) | 0.08 / 0.12 | 4.9e-11 / 1.6e-10 | ≤1.0e-12 |
| **warped + WALLS + seams** | 0.06 / 0.10 | 8.9e-11 / **6.4e-11** | ≤8.7e-16 |

The 2×2 numbers are *identical* to the strip at the same warp: the decomposition genuinely does
not affect the answer.

Three things worth carrying:

- **The warp ceiling is the grid, not the method.** `make_grid`'s family tangles at 0.18
  (min $J$ < 0), so 0.15 is near its limit. A wall-preserving warp stays valid past 0.20.
- **Error grows with warp as it should** — 4.9e-11 → 2.4e-10 from 0.08 to 0.15 — tracking the
  cross-term contraction ratios (0.31 / 0.59 / 0.92) measured single-block, not a seam defect.
- **The wall test uses a wall-PRESERVING warp**: the $y$-displacement carries $\sin\pi\eta$,
  zero at both walls, so the channel stays flat-walled while $\partial y/\partial\xi \neq 0$
  keeps it non-orthogonal. A warp that moved the walls would change the geometry rather than the
  mesh — the trap `test_duct_implicit.py` documents.

### Inflow / outflow across blocks

Outflow faces stay Dirichlet **velocity** boundaries, advected out each step, then rescaled so
the **domain-wide** flux balances. Balancing block by block would force each block to be
individually conservative, which is wrong: mass legitimately crosses a seam, and only the total
has to vanish for the singular Neumann system to be compatible. Connected and periodic faces are
skipped by `boundary_flux_totals` for that reason — they move mass *between* blocks, not out.

Poiseuille with a prescribed inflow and a convective outlet, split along the flow direction:

| split | L2/Umax vs exact parabola | vs single block |
|---|---|---|
| single | 1.604e-06 | — |
| 2 blocks | **1.604e-06** | 6.0e-10 |
| 4 blocks | **1.604e-06** | 6.0e-10 |

**A node-placement trap worth naming.** With inflow/outflow the streamwise axis is *not*
periodic, so the blocks must **partition** the nodes without duplicating the interface — 16 nodes
into 4 blocks of 4, not 5. Splitting a `linspace` by reflex gives overlapping blocks, which
`validate()` rejects; the first attempt did exactly that.

### Obstacle topology, and a 4x speed-up

The 8-block vortex-street layout — a mesh with a **hole** — now runs: 20 PISO steps, flux
divergence **4.66e-11**, mass balanced to 2.2e-16, on deliberately **non-uniform** blocks
(columns 6/4/8, rows 5/4/5).

**Three bugs surfaced only because this topology was tried**, each producing a plausible field:

1. **Only the MISMATCH in tangential padding may be reconciled.** Trimming a ghost to the core
   and re-padding with edge replication discards the neighbour's *real* ghost values wherever
   both blocks were padded — which silently corrupted every seam in the uniform topologies (2×2
   metrics went to 1.4e-01).
2. `_fit_face` re-entered `upto()` for the block currently being built.
3. **The face coefficient must come from the same source the matrix uses.**
   `build_diffusion_matrix` forms it from each block's own metrics;
   `pressure_face_fluxes` was recomputing it from *padded coordinates*, which are extrapolated
   at a reentrant corner. The CG solve converged happily to 9e-11 while the corrected flux
   divergence sat at **1.2e-01** — no pressure field can reconcile two inconsistent operators.
   Padding the coefficient as a **field** fixes it and sidesteps corner ghosts entirely, since a
   face coefficient only needs the core tangential range.

**Performance.** Caching block metrics and padded geometry — both static, both previously
recomputed on every operator call of every step:

| | before | after |
|---|---|---|
| 2 blocks | 3.4× single-block | **0.9×** |
| 4 blocks | 5.9× single-block | **1.4×** |

Multi-block at 2 blocks is now marginally *faster* than single-block, and the obstacle smoke test
went 6.1 s → 1.4 s.

### Dong outflow across blocks

Dong's outlet nodes carry a **prescribed pressure**, so they leave the global unknown set and the
reduced matrix is non-singular: **no global pin, no compatibility projection, and no flux
rescaling** — mass leaves as the solution dictates rather than being forced to a target. The
convective outlet needs all three, because its system stays singular.

On the Poiseuille reference, split 4 ways:

| outflow | L2/Umax (4 blocks) | single block | vs single block |
|---|---|---|---|
| convective | 1.974e-06 | 1.974e-06 | 7.8e-10 |
| **dong** | **1.583e-07** | 1.582e-07 | 1.3e-09 |

Dong is **12× more accurate** here — the reverse of where it started, when a δ of 0.05 made it
1570× *worse* than convective. `dong_delta` defaults to 0.01 for the reason established
single-block: Θ does not vanish where $u_n \to 0$, so at a wall junction it leaves a spurious
near-wall traction of size $O(U_0^2\delta^2)$.

### The right topology beats a workaround: use an O-grid

`plot_cylinder_blocks.py` builds the vortex-street mesh as a **4-block O-grid** around a circular
cylinder — every block connected to its two azimuthal neighbours around a closed ring, with only
the cylinder wall and the far field as boundaries.

That choice is not cosmetic. The H-grid around a square body (`test_obstacle_topology.py`) has
**reentrant corners**, where the blocks either side of a connection carry different tangential
padding and the diagonal ghost lies inside the solid. Handling that needed edge replication, and
getting the face coefficient from the wrong source there produced a corrected flux divergence of
1.2e-01 while the CG residual looked healthy at 9e-11. **An O-grid has no reentrant corners at
all**, so the problem does not arise rather than being worked around.

Node placement follows the connection rule: the azimuthal direction closes on itself and is
partitioned **without** duplicating the seam nodes, like a periodic axis; the radial direction has
real boundaries at both ends and keeps both endpoints.

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
