"""
Preconditioners for the Krylov solves, and the measurements behind the default.

The solver used to pass no preconditioner at all. On the five-domain BFS pressure operator that
cost 1,577 CG iterations for 12,576 cells -- and since ~94% of a step is spent inside Krylov
iterations, iteration count IS the runtime.

Measured on that operator, CG to rtol=1e-10:

    preconditioner        iterations     total
    none (the old default)     1,577     0.115 s
    jacobi                       785     0.059 s     <- 2x, and free to build
    ILU drop_tol=1e-3         20,000     11.9 s      <- FAILED (hit maxiter)
    amg (pyamg V-cycle)           78     0.157 s

WHY JACOBI IS THE DEFAULT rather than AMG. AMG cuts iterations ~20x and its count barely moves
with problem size -- 70 / 78 / 78 / 76 across 3,888 -> 52,780 cells, which is the O(1)
convergence multigrid is for. But each AMG iteration is a full V-cycle, far dearer than one
sparse mat-vec, so on wall time it still LOSES at every size tested:

    cells      none      jacobi     amg
     3,888    0.035 s    0.020 s   0.035 s
    12,576    0.115 s    0.060 s   0.160 s
    26,720    0.248 s    0.130 s   0.303 s
    52,780    0.590 s    0.290 s   0.550 s

The AMG/Jacobi gap does narrow with size (2.7x -> 1.9x), so AMG should overtake somewhere above
these resolutions -- hence it is an option rather than dead code. Re-measure before trusting it
on a new problem; do not assume the crossover.

WHY NOT ILU. `spilu` produces a NON-SYMMETRIC preconditioner, which breaks CG's assumptions
outright -- it does not merely converge slowly, it fails to converge at all (20,000 iterations,
100x slower than no preconditioner). It remains useful for the non-symmetric implicit-cross
operator, which is solved with BiCGStab; that path builds its own and is untouched here.
"""
import numpy as np
import scipy.sparse.linalg as spla

KINDS = ("none", "jacobi", "amg", "amgx")


def make(A, kind="jacobi"):
    """A LinearOperator to pass as `M=` to cg/bicgstab, or None.

    Unknown or unavailable kinds fall back to Jacobi rather than raising: a preconditioner is
    an acceleration, and refusing to run because pyamg is missing would be the wrong trade.
    """
    if kind in (None, "none"):
        return None

    if kind == "amg":
        try:
            import pyamg
        except ImportError:
            kind = "jacobi"          # fall through, below
        else:
            ml = pyamg.smoothed_aggregation_solver(A.tocsr(), max_coarse=200)
            return ml.aspreconditioner(cycle="V")

    if kind == "amgx":
        # NVIDIA AmgX -- see src/amgx/README.md. Measured 28x faster than Jacobi on the
        # 86k-unknown pressure operator once the hierarchy is reused across steps. Needs an
        # NVIDIA GPU and the built library; falls back rather than failing, because a missing
        # GPU should not stop a run.
        try:
            from src.amgx.binding import make_amgx_preconditioner
        except ImportError:
            kind = "jacobi"
        else:
            return make_amgx_preconditioner(A)

    if kind != "jacobi":
        raise ValueError(f"unknown preconditioner {kind!r}; expected one of {KINDS}")

    d = A.diagonal().astype(float)
    # A zero diagonal would make this a division by zero. It should not happen for the
    # operators here -- the conservative diffusion matrix carries a positive diagonal by
    # construction -- but a silent inf would corrupt the solve rather than fail it.
    bad = d == 0.0
    if bad.any():
        d = d.copy()
        d[bad] = 1.0
    dinv = 1.0 / d
    n = A.shape[0]
    return spla.LinearOperator((n, n), matvec=lambda v: dinv * v, dtype=float)
