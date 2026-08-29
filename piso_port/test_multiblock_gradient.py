"""
Finite-difference gradients through a MULTI-BLOCK PISO step.

Written BEFORE any multi-block adjoint, deliberately. It is the reference the adjoint will have
to match, and it answers two questions that decide whether building the adjoint is even sensible:

  1. IS THE FORWARD SOLVER SMOOTH ENOUGH TO DIFFERENTIATE? Every step contains iterative solves
     (BiCGStab, CG) truncated at a tolerance. If that truncation noise is larger than the finite
     -difference signal, the gradient is meaningless and an adjoint would inherit the problem.
  2. IS THE GRADIENT BLOCK-COUNT-INDEPENDENT? Split-equals-whole applied to dL/dtheta rather
     than to the solution. This is the gate that would catch the specific bug the multi-block
     adjoint is most likely to have: a connection face coefficient is 0.5(Jg_A + Jg_B), so
     dL/dA there must scatter back to BOTH blocks. Accumulate into only one and the gradient is
     silently HALVED -- the loss still decreases, training still appears to work, and nothing
     complains.

theta is the body force driving a periodic channel; the loss is the kinetic energy after N
steps. Central differences, with a step-size sweep so the truncation/round-off balance is
measured rather than assumed.
"""
import sys, io, contextlib, warnings
import numpy as np
warnings.filterwarnings("ignore")
from multiblock import Block, Connection, Domain, face_id
from piso_multiblock import MultiBlockPISO
from piso_numpy_3d import PISOSolver

NX, NY, NZ = 8, 9, 4
NU, DT, NSTEP = 0.1, 0.05, 6
results = []
def check(name, good, detail):
    results.append(good)
    print(f"  [{'PASS' if good else 'FAIL'}] {name}: {detail}")


def loss_single(G, tol=1e-13):
    s = PISOSolver((NX, NY, NZ), warp=1e-9, nu=NU, dt=DT, corrector_steps=2,
                   periodic=(True, False, True), scheme="rotational", time_scheme="be",
                   convection="central", boundary_flux_mode="impermeable",
                   pressure_coef="rowsum", pressure_tol=tol, picard_iters=1)
    s.velocity_source = [np.full_like(s.y, G), np.zeros_like(s.y), np.zeros_like(s.y)]
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(NSTEP):
            s.step()
    return float(np.sum(s.u ** 2 + s.v ** 2 + s.w ** 2))


def channel_domain(n):
    nxb = NX // n
    xi = np.arange(NX) / NX
    eta = np.linspace(0, 1, NY)
    ze = np.arange(NZ) / NZ
    X, Y, Z = np.meshgrid(xi, eta, ze, indexing="ij")
    bl = []
    for b in range(n):
        sl = slice(b * nxb, (b + 1) * nxb)
        blk = Block((nxb, NY, NZ), X[sl], Y[sl], Z[sl], (1.0 / NX, 1.0 / (NY - 1), 1.0 / NZ))
        blk.faces[face_id(1, 0)] = blk.faces[face_id(1, 1)] = "wall"
        blk.faces[face_id(2, 0)] = blk.faces[face_id(2, 1)] = "periodic"
        bl.append(blk)
    return Domain(bl, [Connection(b, face_id(0, 1), (b + 1) % n, face_id(0, 0),
                                  shift=(1.0, 0, 0) if (b + 1) % n == 0 else (0, 0, 0))
                       for b in range(n)])


def loss_multi(G, n, tol=1e-13):
    d = channel_domain(n)
    m = MultiBlockPISO(d, NU, DT, 2, tol, time_scheme="be", scheme="rotational",
                       picard_iters=1)
    m.velocity_source = [G, 0.0, 0.0]
    for _ in range(NSTEP):
        m.step()
    return float(sum(np.sum(m.u[b] ** 2 + m.v[b] ** 2 + m.w[b] ** 2)
                     for b in range(len(d.blocks))))


def fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


if __name__ == "__main__":
    print(__doc__.strip().split("\n\n")[0])
    G0 = 0.8

    print("\n1. Is the forward solver smooth enough to differentiate?")
    print("   central-difference dL/dG vs step size (a plateau means signal beats solver noise)")
    print(f"   {'h':>10} {'dL/dG (1 block)':>18}")
    vals = {}
    for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        vals[h] = fd(loss_single, G0, h)
        print(f"   {h:10.0e} {vals[h]:18.10f}")
    mid = [vals[h] for h in (1e-3, 1e-4, 1e-5)]
    spread = (max(mid) - min(mid)) / abs(np.mean(mid))
    check("the gradient has a stable plateau (solver noise is below the FD signal)",
          spread < 1e-6,
          f"relative spread over h = 1e-3..1e-5 is {spread:.2e}; a noisy solver would show no "
          f"plateau and an adjoint would inherit that")

    print("\n2. SPLIT EQUALS WHOLE, applied to the GRADIENT")
    h = 1e-4
    g1 = fd(loss_single, G0, h)
    print(f"   {'blocks':>7} {'dL/dG':>18} {'rel. diff vs single':>21}")
    print(f"   {'single':>7} {g1:18.10f} {'--':>21}")
    ok = True
    for n in (2, 4):
        gn = fd(lambda G: loss_multi(G, n), G0, h)
        rel = abs(gn / g1 - 1)
        ok &= rel < 1e-6
        print(f"   {n:7d} {gn:18.10f} {rel:21.2e}")
    check("the gradient is block-count-independent", ok,
          "a halved seam contribution -- the likeliest multi-block adjoint bug -- would show "
          "here as a systematic offset, and nowhere else")

    print("\n3. The loss itself must also be block-count-independent (sanity)")
    L1 = loss_single(G0)
    Ls = {n: loss_multi(G0, n) for n in (2, 4)}
    worst = max(abs(Ls[n] / L1 - 1) for n in (2, 4))
    print(f"   single {L1:.12f}   2 blocks {Ls[2]:.12f}   4 blocks {Ls[4]:.12f}")
    check("the loss agrees across decompositions", worst < 1e-9,
          f"worst relative difference {worst:.2e}")

    n_pass = sum(results)
    print(f"\n{'='*74}\n  {n_pass}/{len(results)} checks passed\n{'='*74}")
    sys.exit(0 if n_pass == len(results) else 1)
