"""
Rigorous validation of the Phase 3 curvilinear momentum operator.

The previous implementation passed its shipped test only because that test used an
unusually mild grid warp (0.01) and a small viscosity (nu=0.01), which together pushed
an inconsistent-discretisation error below the solution error. These tests are designed
to defeat exactly that kind of false pass:

  T1  operator consistency   -- does implicit + deferred correction actually reconstruct
                                the Laplacian, under refinement, at every warp?
  T2  grid-warp sweep        -- 2nd order must survive strong skewness, not just mild
  T3  viscosity sweep        -- diffusion-dominated flow, where a bad diffusion operator
                                can no longer hide behind a small nu
  T4  independent MMS field  -- guards against tuning to Taylor-Green
  T5  independent solver     -- dense LAPACK vs BiCGStab, sharing no code path
  T6  regression vs old form -- the non-J-weighted split must visibly fail
  T7  warp x nu corner       -- the joint instability that warp-only and nu-only
                                sweeps both miss; must recover via under-relaxation
"""
import sys
import numpy as np
import scipy.sparse.linalg as splinalg

from src.phase1_grid_metrics import analytical_wavy_grid_mms, compute_numerical_metrics
from src.phase3_momentum import (get_mms_taylor_green, build_momentum_matrix_7point,
                             build_conservative_diffusion_matrix, compute_cross_diffusion,
                             boundary_masks, solve_momentum)

PASS, FAIL = "PASS", "FAIL"
results = []
def check(name, ok, detail):
    results.append((name, ok))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")

def rates(errs):
    return [np.log2(errs[i]/errs[i+1]) for i in range(len(errs)-1)]

def second_order(errs, bar=1.8, ceiling=4.0):
    """Every refinement step must converge at ~2nd order.

    Three guards, each earned by a false pass this suite actually produced:
      - all rates, not just the last: a blown-up middle point is otherwise hidden by the
        next point recovering;
      - monotone decrease: catches stalled/flatlining refinement;
      - an UPPER bound on the rate: a blown-up *first* point yields a huge apparent rate
        (e.g. 15) while still looking monotone, so unphysical superconvergence must fail
        too. Real 2nd-order data sits comfortably inside [1.8, 4.0].
    """
    e = np.asarray(errs)
    r = rates(errs)
    return bool(np.all(np.isfinite(e)) and np.all(np.diff(e) < 0)
                and min(r) > bar and max(r) < ceiling)

def fmt(errs):
    return "  ".join(f"{e:.2e}" for e in errs)

# ---------------------------------------------------------------- T1
print("\nT1: operator consistency -- does (implicit + cross) reconstruct nabla^2 ?")
G = np.gradient
for warp in (0.01, 0.05, 0.10, 0.20):
    errs = []
    for n in (20, 40, 80):
        x,y,z,dxi,deta,dzeta,_,_ = analytical_wavy_grid_mms(n,n,n,Ax=warp,Ay=warp,Az=warp)
        J,m = compute_numerical_metrics(x,y,z,dxi,deta,dzeta)
        u = np.sin(x)*np.cos(y)*np.cos(z); lap_ex = -3*u
        # implicit part = -(M @ u)/J, since M discretises -(J * diagonal Laplacian)
        M = build_conservative_diffusion_matrix(n,n,n,dxi,deta,dzeta,J,m)
        implicit = -(M @ u.ravel()).reshape(n,n,n)/J
        total = implicit + compute_cross_diffusion(u,J,m,dxi,deta,dzeta)
        I = (slice(2,-2),)*3
        errs.append(np.linalg.norm((total-lap_ex)[I])/np.linalg.norm(lap_ex[I]))
    r = rates(errs)
    check(f"warp={warp:.2f}", second_order(errs), f"errs {fmt(errs)}  rates {r[0]:.2f}, {r[1]:.2f}")

# ---------------------------------------------------------------- T2
print("\nT2: grid-warp sweep (nu=0.01, Taylor-Green) -- 2nd order must survive skewness")
for warp in (0.01, 0.05, 0.10, 0.20):
    eu, ev = [], []
    for n in (10, 20, 40):
        un,vn,ue,ve,_ = solve_momentum(n, warp, 0.01, verbose=False)
        I=(slice(1,-1),)*3; f=(n-2)**1.5
        eu.append(np.linalg.norm((un-ue)[I])/f); ev.append(np.linalg.norm((vn-ve)[I])/f)
    ru, rv = rates(eu), rates(ev)
    ok = second_order(eu) and second_order(ev)
    check(f"warp={warp:.2f}", ok, f"U rates {ru[0]:.2f},{ru[1]:.2f} | V rates {rv[0]:.2f},{rv[1]:.2f}")

# ---------------------------------------------------------------- T3
print("\nT3: viscosity sweep (warp=0.10) -- diffusion-dominated, nu can no longer mask")
for nu in (0.01, 0.1, 1.0, 10.0):
    eu = []
    for n in (10, 20, 40):
        un,vn,ue,ve,_ = solve_momentum(n, 0.10, nu, verbose=False)
        I=(slice(1,-1),)*3
        eu.append(np.linalg.norm((un-ue)[I])/((n-2)**1.5))
    r = rates(eu)
    check(f"nu={nu:<5}", second_order(eu), f"errs {fmt(eu)}  rates {r[0]:.2f}, {r[1]:.2f}")

# ---------------------------------------------------------------- T4
def mms_polytrig(x, y, z, nu):
    """Independent MMS field, deliberately NOT divergence-free and not Taylor-Green."""
    u = x**2*y + np.sin(z)
    v = y**2*z + np.sin(x)
    w = np.zeros_like(x)
    dudx, dudy, dudz = 2*x*y, x**2, np.cos(z)
    dvdx, dvdy, dvdz = np.cos(x), 2*y*z, y**2
    lap_u = 2*y - np.sin(z)
    lap_v = 2*z - np.sin(x)
    S_u = u*dudx + v*dudy + w*dudz - nu*lap_u
    S_v = u*dvdx + v*dvdy + w*dvdz - nu*lap_v
    return u, v, w, S_u, S_v, np.zeros_like(x)

print("\nT4: independent MMS field (poly-trig, non-solenoidal)")
for warp in (0.05, 0.15):
    for nu in (0.01, 1.0):
        eu = []
        for n in (10, 20, 40):
            un,vn,ue,ve,_ = solve_momentum(n, warp, nu, mms=mms_polytrig, verbose=False)
            I=(slice(1,-1),)*3
            eu.append(np.linalg.norm((un-ue)[I])/((n-2)**1.5))
        r = rates(eu)
        check(f"warp={warp:.2f} nu={nu}", second_order(eu), f"errs {fmt(eu)}  rates {r[0]:.2f}, {r[1]:.2f}")

# ---------------------------------------------------------------- T5
print("\nT5: independent solver cross-check (dense LAPACK vs BiCGStab)")
n, warp, nu = 12, 0.10, 0.1
x,y,z,dxi,deta,dzeta,_,_ = analytical_wavy_grid_mms(n,n,n,Ax=warp,Ay=warp,Az=warp)
J,m = compute_numerical_metrics(x,y,z,dxi,deta,dzeta)
ue,ve,we,Su,Sv,Sw = get_mms_taylor_green(x,y,z,nu)
A = build_momentum_matrix_7point(n,n,n,J,m,dxi,deta,dzeta,ue,ve,we,nu)
ib,bb,_ = boundary_masks(n,n,n)
Aii, Aib = A[ib][:,ib], A[ib][:,bb]
ub = ue.flat[bb]
u_d = np.zeros_like(ue); u_d.flat[bb] = ub
for _ in range(80):
    cr = compute_cross_diffusion(u_d,J,m,dxi,deta,dzeta)
    rhs = (J*(Su+nu*cr)).flat[ib] - Aib@ub
    sol = np.linalg.solve(Aii.toarray(), rhs)          # dense, independent of BiCGStab
    prev = u_d.copy(); u_d.flat[bb]=ub; u_d.flat[ib]=sol
    if np.abs(u_d-prev).max() < 1e-12: break
u_i,_,_,_,_ = solve_momentum(n, warp, nu, verbose=False)
d = np.abs(u_d-u_i).max()
check("dense vs iterative", d < 1e-8, f"max|u_dense - u_bicgstab| = {d:.2e}")

# ---------------------------------------------------------------- T6
print("\nT6: regression -- the OLD non-J-weighted split must visibly fail this bar")
for warp in (0.05, 0.10):
    errs = []
    for n in (20, 40, 80):
        x,y,z,dxi,deta,dzeta,_,_ = analytical_wavy_grid_mms(n,n,n,Ax=warp,Ay=warp,Az=warp)
        J,m = compute_numerical_metrics(x,y,z,dxi,deta,dzeta)
        u = np.sin(x)*np.cos(y)*np.cos(z); lap_ex = -3*u
        g11=m['xi_x']**2+m['xi_y']**2+m['xi_z']**2
        g22=m['eta_x']**2+m['eta_y']**2+m['eta_z']**2
        g33=m['zeta_x']**2+m['zeta_y']**2+m['zeta_z']**2
        d2 = lambda f,s,a: G(G(f,s,axis=a,edge_order=2),s,axis=a,edge_order=2)
        old = g11*d2(u,dxi,0)+g22*d2(u,deta,1)+g33*d2(u,dzeta,2)   # no J weighting
        tot = old + compute_cross_diffusion(u,J,m,dxi,deta,dzeta)
        I=(slice(2,-2),)*3
        errs.append(np.linalg.norm((tot-lap_ex)[I])/np.linalg.norm(lap_ex[I]))
    r = rates(errs)
    check(f"old form warp={warp:.2f} stalls", max(r) < 0.5,
          f"errs {fmt(errs)}  rates {r[0]:.2f}, {r[1]:.2f}  (stalled, as expected)")


# ---------------------------------------------------------------- T7
print("\nT7: warp x nu CORNER -- the joint failure mode; neither axis alone predicts it")
for warp, nu in [(0.20,1.0), (0.25,1.0), (0.30,1.0), (0.25,10.0)]:
    eu = []
    for n in (10, 20, 40):
        un,vn,ue,ve,dg = solve_momentum(n, warp, nu, verbose=False)
        I=(slice(1,-1),)*3
        eu.append(np.linalg.norm((un-ue)[I])/((n-2)**1.5))
    r = rates(eu)
    ok = second_order(eu)
    check(f"warp={warp:.2f} nu={nu}", ok,
          f"errs {fmt(eu)}  rates {r[0]:.2f}, {r[1]:.2f}  omega={dg['u']['omega']}")

# ---------------------------------------------------------------- summary
n_pass = sum(1 for _,ok in results if ok)
print(f"\n{'='*62}\n  {n_pass}/{len(results)} checks passed")
if n_pass != len(results):
    print("  FAILED: " + ", ".join(nm for nm,ok in results if not ok))
print('='*62)
sys.exit(0 if n_pass == len(results) else 1)
