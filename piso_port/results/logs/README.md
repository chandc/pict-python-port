# Saved diagnostic output

Raw stdout from the gates and diagnostics behind `reference/pressure_checkerboard.md`, kept so
the numbers quoted there can be checked rather than taken on trust. Deferred-correction warning
spam is stripped; everything else is verbatim.

| log | what it measures |
|---|---|
| `gateA` | persistent flux alone: divergence and velocity drift |
| `gateB` | Rhie-Chow on the persistent flux; refinement study |
| `gateC` | dt independence, with and without `ddt_corr` |
| `gateD`, `gateD2` | multi-block, Cartesian strip; dt sweep across blocks |
| `gateE` | oscillation in u AND p, Cartesian vs skew, 1/2/4 blocks |
| `order` | order of accuracy vs the exact duct series |
| `metric3` | the three oscillation metrics compared -- why two of them lie |
| `refine` | checkerboard vs grid refinement, no fix |
| `feedback` | does the mode reach the velocity (filter p, watch u) |
| `compat`, `wallrc`, `seamrc` | tracking the seam bug that broke mass conservation |
| `isolate`, `conv` | which flag breaks what, and whether more correctors help |
| `bfs_iso*`, `bfs_loc`, `bfs80` | backward-facing step: which flag, and where it blows up |
| `regress*` | regression suites after each change |

## results/fields/

Every solver run now writes a checkpoint here, not just a printed number. Each is a full
restart state (u, v, w, p, u_prev, p_flux, F_prev) plus metadata (Re, dt, x_r, interior
divergence, step count), so a run can be CONTINUED as well as post-processed, and
`checkpoint.load_fields()` reads one for plotting without constructing a solver.

## Performance investigation (preconditioning, GPU, AmgX)

| log | what it measures |
|---|---|
| `blas_prof` | where a step's time goes: 93% SciPy sparse, 48.8% `csr_matvec` |
| `iters` | Krylov iterations per step -- 3,193 on 12.6k cells, unpreconditioned |
| `precond`, `precond2` | preconditioner comparison on the real pressure operator |
| `scale` | AMG vs Jacobi vs none as problem size grows |
| `vcycle` | pyamg V-cycle cost in SpMV units, and the break-even point |
| `prec_check` | end-to-end effect of the preconditioner on a real run |
| `gpu_bench` | SpMV and CG, CPU vs GB10 |
| `dong_sparsity` | whether the Dong reduced system keeps constant sparsity (it does) |
