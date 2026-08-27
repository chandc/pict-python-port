"""
Multi-step rollout with gradient checkpointing.

Training signal comes from a trajectory, not one step. Two things this must get right:

  * the network input is the DIFFERENTIABLE velocity, not a detached copy, so sensitivity
    flows back through the whole chain rather than one step at a time;
  * the momentum matrix A is rebuilt each step from the current (detached) state -- the
    Picard/frozen-coefficient choice. dL/dA is therefore NOT propagated to u^n, so the
    resulting gradient is an APPROXIMATION and will not match finite differences. That bias
    is Stage 4's subject; measured here it is large (~16%), so it is not a free shortcut.

    Pass rebuild=False to hold A genuinely constant (built once from the initial state). The
    gradient is then EXACT, which is what lets Stage 3 verify the time chain against finite
    differences without the frozen-coefficient bias confounding the measurement.

Checkpointing stores the state every k steps and recomputes the forward within a segment, so
peak memory is O(N/k + k) instead of O(N).
"""
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


def _one_step(sim, net, u, v, w, rebuild=True):
    if rebuild:
        un = [t.detach().numpy().reshape(sim.shape) for t in (u, v, w)]
        sim.build(*un)                              # frozen-coefficient linearisation
    S = net.field(u, v, w, sim.shape)               # torch in -> stays in the graph
    return sim.step(u, v, w, S)


def rollout(sim, net, u0, v0, w0, nsteps, checkpoint_every=None, rebuild=True):
    u, v, w = (torch.as_tensor(np.asarray(f).ravel()) for f in (u0, v0, w0))
    if not rebuild:
        sim.build(*(np.asarray(f).reshape(sim.shape) for f in (u0, v0, w0)))
    if checkpoint_every is None:
        for _ in range(nsteps):
            u, v, w = _one_step(sim, net, u, v, w, rebuild)
        return u, v, w

    k = checkpoint_every
    done = 0
    while done < nsteps:
        m = min(k, nsteps - done)

        def seg(a, b, c, m=m):
            for _ in range(m):
                a, b, c = _one_step(sim, net, a, b, c, rebuild)
            return a, b, c

        u, v, w = checkpoint(seg, u, v, w, use_reentrant=False)
        done += m
    return u, v, w
