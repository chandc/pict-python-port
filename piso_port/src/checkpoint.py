"""
Checkpoint / restart for the PISO solvers, single-block and multi-block alike.

A restart is only worth having if it is EXACT: continuing from a checkpoint must reproduce, to
round-off, what an uninterrupted run would have produced. Two things make that true, and both
are easy to get wrong:

  * u_prev, the BDF2 history, must be saved. Without it the first step after a restart falls
    back to backward Euler -- the run continues, looks stable, and is quietly first-order
    accurate across the seam. Nothing in the output announces this.
  * The pressure must be saved. The incremental and rotational schemes carry p forward as a
    running total; a restart from p = 0 re-derives it over several steps and perturbs the
    velocity meanwhile.
  * With Rhie-Chow enabled there are two MORE pieces of running state: p_flux, the projection
    pressure the face flux actually carries (it differs from p under the rotational scheme),
    and F_prev, the previous step's face flux that ddt_corr re-injects. Neither is
    reconstructible from u and p, so a restart without them silently loses the checkerboard
    damping for a step and then limps back.

Restarting into a solver configured differently from the one that wrote the file (different nu,
dt, time scheme, or grid) is refused rather than silently accepted, because the resulting run
would be neither the old case nor a clean new one. Pass strict=False to override deliberately.

Fields are stored flat with a per-block prefix, so one .npz serves both solver types and can be
read for post-processing without constructing a solver at all -- see load_fields().
"""
import numpy as np

FORMAT = 3                      # bump when the on-disk layout changes incompatibly
FIELDS = ("u", "v", "w", "p")
# Config that changes the meaning of the state. A restart that disagrees on any of these is
# not a continuation of the same simulation.
CONFIG = ("nu", "dt", "time_scheme", "scheme", "picard_iters", "corrector_steps",
          "implicit_cross", "rhie_chow", "persistent_flux", "ddt_corr")


def _is_multiblock(s):
    return isinstance(getattr(s, "u", None), dict)


def _blocks(s):
    """Field state as {name: {block: array}}, single-block presented as block 0."""
    if _is_multiblock(s):
        return {f: dict(getattr(s, f)) for f in FIELDS}
    return {f: {0: getattr(s, f)} for f in FIELDS}


def save(solver, path, **extra):
    """Write the full restart state. `extra` is stored alongside for post-processing."""
    out = {"__format__": np.array(FORMAT),
           "nstep": np.array(getattr(solver, "nstep", 0)),
           "time": np.array(getattr(solver, "time", 0.0)),
           "multiblock": np.array(_is_multiblock(solver))}
    for k in CONFIG:
        if hasattr(solver, k):
            out[f"cfg_{k}"] = np.array(getattr(solver, k))

    st = _blocks(solver)
    nb = len(st["u"])
    out["nblocks"] = np.array(nb)
    for f in FIELDS:
        for b, arr in st[f].items():
            out[f"{f}_{b}"] = np.asarray(arr)

    # BDF2 history -- absent only before the first step has been taken
    prev = getattr(solver, "u_prev", None)
    out["has_prev"] = np.array(prev is not None)
    if prev is not None:
        for f, part in zip(("u", "v", "w"), prev):
            part = part if isinstance(part, dict) else {0: part}
            for b, arr in part.items():
                out[f"prev{f}_{b}"] = np.asarray(arr)

    # Rhie-Chow running state (absent unless the option is on)
    pf = getattr(solver, "p_flux", None)
    if pf is not None:
        part = pf if isinstance(pf, dict) else {0: pf}
        for b, arr in part.items():
            out[f"pflux_{b}"] = np.asarray(arr)
    Fp = getattr(solver, "F_prev", None)
    out["has_Fprev"] = np.array(Fp is not None)
    if Fp is not None:
        per_block = Fp if isinstance(Fp, dict) else {0: Fp}
        for b, axes in per_block.items():
            for a, arr in enumerate(axes):
                out[f"Fprev_{b}_{a}"] = np.asarray(arr)

    for k, v in extra.items():
        out[f"x_{k}"] = np.asarray(v)
    np.savez_compressed(path, **out)
    return path


def load_fields(path):
    """Read a checkpoint for post-processing, without needing a solver.

    Returns (fields, meta) where fields is {name: {block: array}} and meta carries nstep,
    time, the saved config and any extras.
    """
    d = np.load(path, allow_pickle=False)
    fmt = int(d["__format__"])
    if fmt != FORMAT:
        raise ValueError(f"checkpoint format {fmt}, this build reads {FORMAT}")
    nb = int(d["nblocks"])
    fields = {f: {b: d[f"{f}_{b}"] for b in range(nb)} for f in FIELDS}
    meta = {"nstep": int(d["nstep"]), "time": float(d["time"]),
            "multiblock": bool(d["multiblock"]), "nblocks": nb,
            "config": {k: d[f"cfg_{k}"].item() for k in CONFIG if f"cfg_{k}" in d},
            "extra": {k[2:]: d[k] for k in d.files if k.startswith("x_")}}
    return fields, meta


def load(solver, path, strict=True):
    """Restore a checkpoint into `solver`, which must already be built on the same grid."""
    fields, meta = load_fields(path)

    if meta["multiblock"] != _is_multiblock(solver):
        raise ValueError("checkpoint/solver disagree on multi-block vs single-block")
    nb_solver = len(_blocks(solver)["u"])
    if meta["nblocks"] != nb_solver:
        raise ValueError(f"checkpoint has {meta['nblocks']} block(s), "
                         f"solver has {nb_solver}")
    for f in FIELDS:
        for b, arr in fields[f].items():
            want = _blocks(solver)[f][b].shape
            if arr.shape != want:
                raise ValueError(f"block {b} field {f}: checkpoint {arr.shape} "
                                 f"vs solver {want}")
    if strict:
        bad = {k: (v, getattr(solver, k)) for k, v in meta["config"].items()
               if hasattr(solver, k) and getattr(solver, k) != v}
        if bad:
            detail = ", ".join(f"{k}: file={a!r} solver={b!r}" for k, (a, b) in bad.items())
            raise ValueError(
                f"solver is configured differently from the checkpoint ({detail}). "
                "This would be neither a continuation nor a clean new run; "
                "pass strict=False if the change is intended.")

    mb = meta["multiblock"]
    for f in FIELDS:
        if mb:
            getattr(solver, f).update({b: a.copy() for b, a in fields[f].items()})
        else:
            setattr(solver, f, fields[f][0].copy())

    d = np.load(path, allow_pickle=False)
    if bool(d["has_prev"]):
        parts = []
        for f in ("u", "v", "w"):
            per = {b: d[f"prev{f}_{b}"].copy() for b in range(meta["nblocks"])}
            parts.append(per if mb else per[0])
        solver.u_prev = tuple(parts)
    else:
        solver.u_prev = None

    if f"pflux_0" in d.files:
        per = {b: d[f"pflux_{b}"].copy() for b in range(meta["nblocks"])
               if f"pflux_{b}" in d.files}
        solver.p_flux = per if mb else per[0]
    if bool(d["has_Fprev"]) if "has_Fprev" in d.files else False:
        per = {}
        for b in range(meta["nblocks"]):
            axes = [d[f"Fprev_{b}_{a}"].copy() for a in range(3)
                    if f"Fprev_{b}_{a}" in d.files]
            if axes:
                per[b] = axes
        solver.F_prev = per if mb else per.get(0)
    else:
        solver.F_prev = None

    solver.nstep, solver.time = meta["nstep"], meta["time"]
    return meta
