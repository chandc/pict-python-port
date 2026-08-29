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
