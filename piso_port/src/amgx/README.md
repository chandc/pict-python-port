# AmgX backend (in progress)

NVIDIA's algebraic multigrid, used as the pressure-Poisson preconditioner. Requires an NVIDIA
GPU; on a machine without one the solver falls back to `jacobi` and nothing here is loaded.

## Why

Measured on the real five-domain BFS pressure operator (86,040 unknowns, CG to rtol=1e-10),
CPU = Apple M3 Max, GPU = NVIDIA GB10:

| solver | iterations | time |
|---|---|---|
| CPU CG, no preconditioner | 3,122 | 1.540 s |
| CPU CG, Jacobi | 1,418 | 0.784 s |
| GPU CG, Jacobi (CuPy) | 1,418 | 0.236 s |
| AmgX, fresh setup each solve | 53 | 0.077 s |
| **AmgX, hierarchy reused** | **53** | **0.028 s** |

## The property that makes reuse legal

The pressure matrix keeps **identical sparsity structure** every step — the grid does not move —
and only its values change, by ~1e-3 relative per step, as Gamma = J/rowsum(A) drifts with the
flow. `AMGX_matrix_replace_coefficients` updates the values in place and the existing hierarchy
stays a good preconditioner: measured 53 iterations per step either way, with setup falling from
0.043 s to 0.0005 s.

Drift accumulates, so production use needs a **periodic rebuild** — the same policy the existing
`spilu` cache already applies, refactoring when ||coef|| moves more than 5%. The refresh interval
is not yet calibrated.

## Status

- [x] AmgX 2.5.0 builds on GB10 (`cmake -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_NO_MPI=1`)
- [x] CSR uploads directly via `AMGX_matrix_upload_all`, bypassing the Matrix Market reader,
      which rejected our files
- [x] Hierarchy reuse measured (`bench_reuse.c`)
- [ ] Python binding (ctypes shim vs pyamgx — undecided)
- [ ] Wire into `src/precond.py` as a fourth `kind`
- [ ] Refresh policy calibrated against a full 3,000-step run
- [x] Device residency: **not a problem on this hardware** — see below.

## Unified memory removes the need for a CuPy port

GB10 shares memory between the Grace CPU and the Blackwell GPU over NVLink-C2C, so "uploading"
a matrix is not a PCIe transfer. Measured against the 28 ms/step solve:

| operation | time | rate |
|---|---|---|
| `matrix_upload_all` (full CSR, 4.8 MB) | 0.223 ms | 33.6 GB/s |
| `replace_coefficients` (values only) | 0.088 ms | 54.4 GB/s |
| `vector_upload` (rhs) | 0.019 ms | 36.3 GB/s |
| `vector_download` (solution) | 0.017 ms | 41.4 GB/s |
| **per step: replace + rhs up + solution down** | **0.123 ms** | **0.44% of solve** |

The earlier plan assumed the matrix would have to live on the device, which implied porting the
whole array layer to CuPy first. It does not. Assembly stays in SciPy on the host, the matrix is
handed to AmgX each step for well under 1% overhead, and the solution comes back. That reduces
the work from a rewrite of ~3,100 lines to one preconditioner behind the interface
`src/precond.py` already provides.

(The transfer driver prints 0.04% because its denominator is a 318 ms non-converging solve --
the modes-A/B bug below. 0.44% against the healthy 28 ms solve is the honest figure.)

## Known unknown

`bench_reuse.c` modes A and B (full setup / resetup per step) report 5,148 iterations per step —
a non-converging solver, not a slow one. Calling `AMGX_solver_setup` or `AMGX_solver_resetup`
after `replace_coefficients` leaves the solver in a state this driver mishandles, and the cause
is **not yet understood**. Mode C (reuse) is unaffected and is the path being pursued; the
fresh-setup baseline quoted above comes from the single-matrix driver instead.

## Config

`PCG_CLASSICAL_V_JACOBI` at tolerance 1e-10. Config choice dominates: `AGGREGATION_JACOBI`, run
as a standalone solver rather than a CG preconditioner, hit its iteration cap at a convergence
rate of 0.90 — the same library, tuned wrongly, looks useless. Re-tune per problem.
