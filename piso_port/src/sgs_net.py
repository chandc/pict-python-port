"""A deliberately tiny 3D CNN, small enough that finite differences on EVERY weight is cheap."""
import torch
import torch.nn as nn


class TinySGSNet(nn.Module):
    """
    velocity (1,3,n,n,n) -> momentum source (1,3,n,n,n).

    Circular padding, because the solver cases these are trained on are periodic: a conv with
    zero padding would inject a spurious boundary and break the shift-equivariance that gate (c)
    checks for.

    Two layers, 173 parameters:  conv3d(3->2, k=3) = 164,  conv3d(2->3, k=1) = 9.
    """

    def __init__(self, hidden=2):
        super().__init__()
        self.c1 = nn.Conv3d(3, hidden, 3, padding=1, padding_mode="circular")
        self.c2 = nn.Conv3d(hidden, 3, 1)

    def forward(self, x):
        return self.c2(torch.tanh(self.c1(x)))

    def field(self, u, v, w, shape):
        """
        Fields in, a (3, N) source tensor out. Accepts numpy arrays or torch tensors --
        passing TORCH tensors keeps the network input inside the autograd graph, which is
        what makes a multi-step rollout gradient faithful rather than per-step.
        """
        cols = []
        for f in (u, v, w):
            t = f if torch.is_tensor(f) else torch.as_tensor(f)
            cols.append(t.reshape(shape))
        return self(torch.stack(cols).unsqueeze(0)).squeeze(0).reshape(3, -1)


class SGSNet(nn.Module):
    """
    Larger network for the closure task. Stage 2's 173-parameter net was sized so that finite
    differences on EVERY weight stayed affordable; that constraint does not apply here, where
    the gate is statistical (correlation on held-out data) rather than exact.

    Still deliberately small -- ~10k parameters. A closure that only works with a large network
    on 24 snapshots would be memorising, and the held-out correlation is what would catch it.
    """

    def __init__(self, width=24, depth=3):
        super().__init__()
        layers, c_in = [], 3
        for _ in range(depth):
            layers += [nn.Conv3d(c_in, width, 3, padding=1, padding_mode="circular"),
                       nn.GELU()]
            c_in = width
        layers += [nn.Conv3d(width, 3, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
