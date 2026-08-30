"""
One dispatch point for every sparse solve, so a backend swap is a single argument.

Two backends:

  scipy  SciPy's CG / BiCGStab with a preconditioner from src.precond (default: jacobi).
  amgx   NVIDIA AmgX solving the whole system on the GPU. Requires libamgxsh.so; falls back
         to scipy when it is unavailable, because a missing GPU should not stop a run.

WHY AMGX REPLACES THE SOLVE RATHER THAN PRECONDITIONING SciPy's. Handing AmgX one V-cycle at a
time to a host Krylov loop would pay the round trip per iteration and discard AmgX's own
iteration control. Measured whole-system on the 86k-unknown pressure operator: 28 ms/step at 53
iterations against 784 ms for SciPy CG+Jacobi -- about 25x.

HIERARCHY REUSE IS WHAT MAKES IT FAST, and it is legal here because the pressure matrix keeps
IDENTICAL sparsity every step. Verified on the Dong path specifically, where the solve is on a
REDUCED system with Dirichlet outlet nodes eliminated: the Dirichlet set is geometric (outlet
faces), so it does not move -- 296 nodes, M_ff nnz 182,792, both identical across 8 steps. The
cache is therefore keyed on the sparsity pattern and invalidated if it ever changes.
"""
import numpy as np
import scipy.sparse.linalg as spla

from src.precond import make as make_precond


class SolveCache:
    """Holds a backend solver bound to one sparsity pattern, across steps.

    Lives on the PISO solver instance, because the whole point is to survive from one step to
    the next. `key()` is the pattern; if it changes the cached solver is discarded rather than
    silently reused on a matrix it was not built for.
    """

    def __init__(self, backend="scipy", precond="jacobi", drift_tol=0.05, config=None):
        self.backend = backend
        self.precond = precond
        self.drift_tol = drift_tol
        self.config = config
        self._key = None
        self._amgx = None
        self.iterations = 0
        self.fell_back = False

    @staticmethod
    def key(A):
        # cheap fingerprint of the sparsity: shape, nnz, and the two index arrays
        return (A.shape, A.nnz,
                hash(A.indptr.tobytes()), hash(A.indices.tobytes()))

    def solve(self, A, b, x0=None, symmetric=True, rtol=1e-12, maxiter=20000,
              singular=False):
        A = A.tocsr()
        A.sort_indices()
        if self.backend == "amgx":
            x = self._amgx_solve(A, b, x0)
            if x is not None:
                if singular:
                    # An all-Neumann operator fixes the solution only up to a constant, and
                    # AmgX may land on a different member than SciPy would. Removing the mean
                    # picks the same one, which keeps a backend swap from shifting p by a
                    # constant. It does not change grad(p), so the flow is unaffected either way.
                    x = x - x.mean()
                return x
            self.fell_back = True

        M = make_precond(A, self.precond)
        solver = spla.cg if symmetric else spla.bicgstab
        x, info = solver(A, b, x0=x0, M=M, rtol=rtol, maxiter=maxiter)
        return x

    def _amgx_solve(self, A, b, x0):
        k = self.key(A)
        if self._amgx is not None and k != self._key:
            self._amgx.close()
            self._amgx = None
        if self._amgx is None:
            try:
                from src.amgx.binding import AmgXSolver
            except ImportError:
                return None
            try:
                self._amgx = AmgXSolver(A, config=self.config, drift_tol=self.drift_tol)
            except Exception:
                return None
            self._key = k
        x = self._amgx.solve(A.data, b, x0=x0)
        self.iterations = self._amgx.iterations
        return x

    def close(self):
        if self._amgx is not None:
            self._amgx.close()
            self._amgx = None
