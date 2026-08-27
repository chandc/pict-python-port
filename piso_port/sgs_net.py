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
