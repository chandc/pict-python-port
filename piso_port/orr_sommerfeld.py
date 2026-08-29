"""
Orr-Sommerfeld eigenproblem for plane Poiseuille flow -- the reference for the growth-rate test.

    (U - c)(D^2 - a^2) phi - U'' phi = (1/(i a Re)) (D^2 - a^2)^2 phi,
    phi = phi' = 0 at y = +/- 1,      U = 1 - y^2

Written as a generalised eigenproblem A phi = c B phi with

    A = U (D^2 - a^2) - U''  -  (1/(i a Re)) (D^2 - a^2)^2 ,     B = (D^2 - a^2)

Chebyshev collocation with Trefethen's clamped D4, the same construction the Stokes case uses --
phi = phi' = 0 IS no-slip for the perturbation, and imposing only phi = 0 gives a different
(simply-supported) problem, a mistake already made once in this repo.

At Re = 7500, a = 1 the least-stable mode is UNSTABLE: c = 0.24989154 + 0.00223497i, so a
disturbance GROWS at alpha*Im(c) = 0.00223497. That number is tiny, which is what makes it a
severe test of a solver's numerical dissipation: any scheme that damps at a comparable rate will
report the wrong sign of stability.
"""
import numpy as np
from scipy.linalg import eig


def cheb(N):
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2., np.ones(N - 1), 2.]) * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    D = np.outer(c, 1. / c) / ((X - X.T) + np.eye(N + 1))
    return D - np.diag(D.sum(axis=1)), x


def os_spectrum(Re=7500.0, alpha=1.0, N=120):
    """Returns (c_all, phi_all, y). c = phase speed + i growth/alpha."""
    D, y = cheb(N)
    D2 = D @ D
    S = np.diag(np.hstack([0., 1. / (1. - y[1:N] ** 2), 0.]))
    D4 = (np.diag(1. - y ** 2) @ np.linalg.matrix_power(D, 4)
          - 8 * np.diag(y) @ np.linalg.matrix_power(D, 3) - 12 * D2) @ S
    i = slice(1, N)
    I = np.eye(N - 1)
    U = np.diag(1 - y[i] ** 2)
    Upp = -2.0 * I
    L = D2[i, i] - alpha ** 2 * I                      # (D^2 - a^2)
    L2 = D4[i, i] - 2 * alpha ** 2 * D2[i, i] + alpha ** 4 * I
    A = U @ L - Upp - L2 / (1j * alpha * Re)
    B = L
    c, V = eig(A, B)
    return c, V, y


def least_stable(Re=7500.0, alpha=1.0, N=120):
    c, V, y = os_spectrum(Re, alpha, N)
    ok = np.isfinite(c) & (np.abs(c) < 10)             # drop the spurious/infinite branch
    j = np.argmax(c[ok].imag)
    idx = np.where(ok)[0][j]
    phi = np.zeros(len(y), dtype=complex)
    phi[1:len(y) - 1] = V[:, idx]
    return c[idx], phi, y


if __name__ == "__main__":
    print("Orr-Sommerfeld, plane Poiseuille, Re=7500, alpha=1")
    print(f"   {'N':>5} {'Re(c) phase speed':>20} {'alpha*Im(c) growth':>21}")
    for N in (60, 80, 120, 200, 250):
        c, _, _ = least_stable(N=N)
        print(f"   {N:5d} {c.real:20.9f} {c.imag:21.9f}")
    print(f"   {'Streett/Chan':>5} {0.24989154:20.9f} {0.00223497:21.9f}")
    c, _, _ = least_stable(N=200)
    print(f"\n   relative error at N=200:  phase {abs(c.real/0.24989154-1):.2e}   "
          f"growth {abs(c.imag/0.00223497-1):.2e}")
