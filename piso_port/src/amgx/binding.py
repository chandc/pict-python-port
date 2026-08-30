"""
ctypes binding to NVIDIA AmgX, used as a COMPLETE solver for the pressure Poisson system.

NOT a preconditioner handed to SciPy's CG. The measured 28 ms/step comes from AmgX running its
own AMG-preconditioned CG entirely on the device; feeding one V-cycle at a time back into a host
Krylov loop would pay the round trip 53 times per solve and discard AmgX's own iteration control.
So this replaces the solve, and `linear_solve()` in src/linsolve.py is the dispatch point.

REUSE IS THE WHOLE POINT. The pressure matrix keeps identical sparsity every step -- the grid
does not move -- and only its values drift, ~1e-3 relative per step, as Gamma = J/rowsum(A)
follows the flow. So the hierarchy is built ONCE and thereafter only the coefficients are
replaced: measured 53 iterations per step either way, with setup falling from 43 ms to 0.5 ms.
Drift accumulates, so `drift_tol` triggers a rebuild -- the same policy the spilu cache in
piso_numpy_3d already uses.

Requires libamgxsh.so; import fails cleanly on any machine without it, and src/precond.py falls
back to Jacobi rather than refusing to run.
"""
import ctypes
import os

import numpy as np

AMGX_MODE_dDDI = 8193          # device, double vec, double mat, int index
AMGX_RC_OK = 0

_LIB_ENV = "AMGX_LIB"          # explicit path wins
_LIB_CANDIDATES = ("libamgxsh.so", "/opt/amgx/lib/libamgxsh.so",
                   "/tmp/AMGX/build/libamgxsh.so")
_CFG_ENV = "AMGX_CONFIG"


def _load():
    path = os.environ.get(_LIB_ENV)
    errs = []
    for cand in ([path] if path else []) + list(_LIB_CANDIDATES):
        if not cand:
            continue
        try:
            return ctypes.CDLL(cand)
        except OSError as e:
            # Report the REAL dlopen error. "not found" is usually wrong: the file is
            # normally present and it is a missing CUDA runtime dependency that fails,
            # which a bare "not found" hides and sends you looking in the wrong place.
            errs.append(f"{cand}: {e}")
    raise ImportError(
        "could not load libamgxsh.so. Set AMGX_LIB, and note the library needs the CUDA "
        "runtime present in the same container. Tried:\n  " + "\n  ".join(errs))


_lib = _load()


def _chk(rc, what):
    if rc != AMGX_RC_OK:
        raise RuntimeError(f"AmgX {what} failed with code {rc}")


# Declare argtypes explicitly. Without them ctypes infers from the Python value, which
# rejects a numpy scalar outright ("Don't know how to convert parameter 2") and, worse, would
# silently truncate a 64-bit pointer passed as an int on some platforms.
_P = ctypes.c_void_p
_I = ctypes.c_int
for _f, _a in (
    ("AMGX_initialize", []),
    ("AMGX_finalize", []),
    ("AMGX_config_create_from_file", [ctypes.POINTER(_P), ctypes.c_char_p]),
    ("AMGX_config_destroy", [_P]),
    ("AMGX_resources_create_simple", [ctypes.POINTER(_P), _P]),
    ("AMGX_resources_destroy", [_P]),
    ("AMGX_matrix_create", [ctypes.POINTER(_P), _P, _I]),
    ("AMGX_matrix_destroy", [_P]),
    ("AMGX_vector_create", [ctypes.POINTER(_P), _P, _I]),
    ("AMGX_vector_destroy", [_P]),
    ("AMGX_solver_create", [ctypes.POINTER(_P), _P, _I, _P]),
    ("AMGX_solver_destroy", [_P]),
    ("AMGX_matrix_upload_all", [_P, _I, _I, _I, _I, _P, _P, _P, _P]),
    ("AMGX_matrix_replace_coefficients", [_P, _I, _I, _P, _P]),
    ("AMGX_vector_upload", [_P, _I, _I, _P]),
    ("AMGX_vector_download", [_P, _P]),
    ("AMGX_solver_setup", [_P, _P]),
    ("AMGX_solver_solve", [_P, _P, _P]),
    ("AMGX_solver_get_iterations_number", [_P, ctypes.POINTER(_I)]),
):
    _fn = getattr(_lib, _f)
    _fn.argtypes = _a
    _fn.restype = ctypes.c_int

_initialised = False


def _init_once():
    global _initialised
    if not _initialised:
        _chk(_lib.AMGX_initialize(), "initialize")
        _initialised = True


class AmgXSolver:
    """One AmgX solver bound to one sparsity pattern.

    Call `solve(values, b)` per step with the current matrix VALUES; the structure passed at
    construction is reused. The hierarchy is rebuilt only when the values have drifted past
    `drift_tol` relative to those it was built from.
    """

    def __init__(self, A, config=None, drift_tol=0.05):
        _init_once()
        A = A.tocsr()
        A.sort_indices()
        self.n = int(A.shape[0])
        self.nnz = int(A.nnz)
        self._ptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        self._col = np.ascontiguousarray(A.indices, dtype=np.int32)
        self.drift_tol = drift_tol

        cfg_path = config or os.environ.get(_CFG_ENV)
        if not cfg_path or not os.path.exists(cfg_path):
            raise RuntimeError(
                "no AmgX config; set AMGX_CONFIG to e.g. PCG_CLASSICAL_V_JACOBI.json. "
                "Config choice dominates performance -- AGGREGATION_JACOBI run as a standalone "
                "solver hit its iteration cap at convergence rate 0.90 on this operator.")

        self._cfg = ctypes.c_void_p()
        _chk(_lib.AMGX_config_create_from_file(ctypes.byref(self._cfg),
                                               cfg_path.encode()), "config_create")
        self._rsrc = ctypes.c_void_p()
        _chk(_lib.AMGX_resources_create_simple(ctypes.byref(self._rsrc), self._cfg),
             "resources_create")
        self._A = ctypes.c_void_p(); self._b = ctypes.c_void_p(); self._x = ctypes.c_void_p()
        self._slv = ctypes.c_void_p()
        m = ctypes.c_int(AMGX_MODE_dDDI)
        _chk(_lib.AMGX_matrix_create(ctypes.byref(self._A), self._rsrc, m), "matrix_create")
        _chk(_lib.AMGX_vector_create(ctypes.byref(self._b), self._rsrc, m), "vector_create b")
        _chk(_lib.AMGX_vector_create(ctypes.byref(self._x), self._rsrc, m), "vector_create x")
        _chk(_lib.AMGX_solver_create(ctypes.byref(self._slv), self._rsrc, m, self._cfg),
             "solver_create")

        vals = np.ascontiguousarray(A.data, dtype=np.float64)
        _chk(_lib.AMGX_matrix_upload_all(
            self._A, self.n, self.nnz, 1, 1,
            self._ptr.ctypes.data_as(ctypes.c_void_p),
            self._col.ctypes.data_as(ctypes.c_void_p),
            vals.ctypes.data_as(ctypes.c_void_p), None), "matrix_upload_all")
        _chk(_lib.AMGX_solver_setup(self._slv, self._A), "solver_setup")
        self._ref = vals.copy()          # values the hierarchy was built from
        self.rebuilds = 1
        self.iterations = 0

    def solve(self, values, b, x0=None):
        vals = np.ascontiguousarray(values, dtype=np.float64)
        drift = (np.abs(vals - self._ref).max()
                 / max(np.abs(self._ref).max(), 1e-300))
        _chk(_lib.AMGX_matrix_replace_coefficients(
            self._A, self.n, self.nnz,
            vals.ctypes.data_as(ctypes.c_void_p), None), "replace_coefficients")
        if drift > self.drift_tol:
            # The hierarchy is stale. Rebuilding costs ~43 ms against a ~28 ms solve, so this
            # must stay rare -- which it is at ~1e-3 drift per step.
            _chk(_lib.AMGX_solver_setup(self._slv, self._A), "solver_setup (rebuild)")
            self._ref = vals.copy()
            self.rebuilds += 1

        rhs = np.ascontiguousarray(b, dtype=np.float64)
        x = np.ascontiguousarray(np.zeros_like(rhs) if x0 is None else x0, dtype=np.float64)
        one = 1
        _chk(_lib.AMGX_vector_upload(self._b, self.n, one,
                                     rhs.ctypes.data_as(ctypes.c_void_p)), "vector_upload b")
        _chk(_lib.AMGX_vector_upload(self._x, self.n, one,
                                     x.ctypes.data_as(ctypes.c_void_p)), "vector_upload x")
        _chk(_lib.AMGX_solver_solve(self._slv, self._b, self._x), "solver_solve")
        _chk(_lib.AMGX_vector_download(self._x,
                                       x.ctypes.data_as(ctypes.c_void_p)), "vector_download")
        it = ctypes.c_int(0)
        _lib.AMGX_solver_get_iterations_number(self._slv, ctypes.byref(it))
        self.iterations = it.value
        return x

    def close(self):
        for fn, h in ((_lib.AMGX_solver_destroy, self._slv),
                      (_lib.AMGX_vector_destroy, self._x),
                      (_lib.AMGX_vector_destroy, self._b),
                      (_lib.AMGX_matrix_destroy, self._A),
                      (_lib.AMGX_resources_destroy, self._rsrc),
                      (_lib.AMGX_config_destroy, self._cfg)):
            try:
                fn(h)
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
