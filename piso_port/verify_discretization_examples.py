"""
Checks every worked example in reference/spatial_discretization.md against the actual
assembly code, so the document cannot drift away from the implementation.
"""
import sys
import numpy as np
from phase3_momentum import _convection_coefs, build_conservative_diffusion_matrix
from phase5_fluxes import divergence_from_fluxes

ok = True
def check(name, got, want, tol=1e-12):
    global ok
    good = np.allclose(got, want, atol=tol)
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {name}: got {got}  want {want}")

print("Worked examples from reference/spatial_discretization.md\n")

# 1. SOU, a = +20 (already divided by h)
a = np.full((5, 1, 1), 20.0)
aP, c = _convection_coefs(a, 0, 5, periodic=False, scheme="sou")
i = 3
check("SOU a>0 coefficients (i-2, i-1, P)",
      [c[-2][i,0,0], c[-1][i,0,0], aP[i,0,0]], [10.0, -40.0, 30.0])
check("SOU a>0 applied to (1.0, 1.4, 2.2)",
      c[-2][i,0,0]*1.0 + c[-1][i,0,0]*1.4 + aP[i,0,0]*2.2, 20.0)

aN = np.full((5, 1, 1), -20.0)
aPn, cn = _convection_coefs(aN, 0, 5, periodic=False, scheme="sou")
check("SOU a<0 coefficients (P, i+1, i+2)",
      [aPn[1,0,0], cn[1][1,0,0], cn[2][1,0,0]], [30.0, -40.0, 10.0])

# 2. Central
aPc, cc = _convection_coefs(a, 0, 5, periodic=False, scheme="central")
check("central coefficients (i-1, P, i+1)",
      [cc[-1][i,0,0], aPc[i,0,0], cc[1][i,0,0]], [-10.0, 0.0, 10.0])
check("central applied to phi_{i-1}=1.4, phi_{i+1}=3.0",
      cc[-1][i,0,0]*1.4 + cc[1][i,0,0]*3.0, 16.0)

# 3. Conservative diffusion row, J*g11 = 1.00, 1.20, 1.40 at i-1, i, i+1, h = 0.1
n, h = 5, 0.1
J = np.ones((n, 1, 1))
met = {k: np.zeros((n, 1, 1)) for k in
       ["xi_x","xi_y","xi_z","eta_x","eta_y","eta_z","zeta_x","zeta_y","zeta_z"]}
met["xi_x"] = np.sqrt(np.array([0.8, 1.00, 1.20, 1.40, 1.6]).reshape(n, 1, 1))
M = build_conservative_diffusion_matrix(n, 1, 1, h, 1.0, 1.0, J, met)
row = M.toarray()[2]
check("diffusion row (i-1, i, i+1)", [row[1], row[2], row[3]], [-110.0, 240.0, -130.0], 1e-9)
check("diffusion applied to phi = (2.0, 2.5, 3.5)",
      -(row @ np.array([2.0, 2.0, 2.5, 3.5, 3.5])), 75.0, 1e-9)

# 4. Flux divergence, (JU) = 0.8, 1.0, 1.6 at i-1, i, i+1, h = 0.1
F = [np.zeros((4, 1, 1)), np.zeros((3, 2, 1)), np.zeros((3, 1, 2))]
F[0][1, 0, 0] = 0.5 * (0.8 + 1.0)
F[0][2, 0, 0] = 0.5 * (1.0 + 1.6)
d = divergence_from_fluxes(F, np.ones((3, 1, 1)), (0.1, 1.0, 1.0))
check("face fluxes (lower, upper)", [F[0][1,0,0], F[0][2,0,0]], [0.90, 1.30])
check("flux divergence at P", d[1, 0, 0], 4.0)

print("\n" + ("all examples match the implementation" if ok else "MISMATCH"))
sys.exit(0 if ok else 1)
