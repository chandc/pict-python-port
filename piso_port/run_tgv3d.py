"""
3D Taylor-Green vortex, fully periodic: energy and enstrophy budget.

This tests something no MMS test can: whether the NONLINEAR convective term transfers energy
correctly. MMS verifies that each operator approximates its differential counterpart; it says
nothing about whether the assembled scheme conserves the quadratic invariants that govern a
turbulent cascade.

The sharp check is the exact periodic identity

    dE/dt = -2 nu * Z,      E = <|u|^2>/2,   Z = <|omega|^2>/2

Any gap between the measured -dE/dt and 2*nu*Z is NUMERICAL dissipation -- energy removed by
the discretisation rather than by physical viscosity. That is exactly the quantity an SGS
closure is supposed to supply, so a scheme with large numerical dissipation is a polluted
baseline to learn against.
"""
import numpy as np, time, warnings, sys
warnings.filterwarnings("ignore")
from piso_numpy_3d import PISOSolver
from phase1_grid_metrics import deriv

K = 2 * np.pi

def tgv_init(x, y, z):
    return (np.sin(K*x)*np.cos(K*y)*np.cos(K*z),
            -np.cos(K*x)*np.sin(K*y)*np.cos(K*z),
            np.zeros_like(x))

def diagnostics(s):
    """Kinetic energy and enstrophy, both volume-averaged."""
    u, v, w, h, per = s.u, s.v, s.w, s.h, s.per
    E = 0.5 * np.mean(u**2 + v**2 + w**2)
    # vorticity in physical space via the chain rule through the metrics
    def dd(f):
        fx = (s.metrics['xi_x']*deriv(f, h[0], 0, per[0]) + s.metrics['eta_x']*deriv(f, h[1], 1, per[1])
              + s.metrics['zeta_x']*deriv(f, h[2], 2, per[2]))
        fy = (s.metrics['xi_y']*deriv(f, h[0], 0, per[0]) + s.metrics['eta_y']*deriv(f, h[1], 1, per[1])
              + s.metrics['zeta_y']*deriv(f, h[2], 2, per[2]))
        fz = (s.metrics['xi_z']*deriv(f, h[0], 0, per[0]) + s.metrics['eta_z']*deriv(f, h[1], 1, per[1])
              + s.metrics['zeta_z']*deriv(f, h[2], 2, per[2]))
        return fx, fy, fz
    ux, uy, uz = dd(u); vx, vy, vz = dd(v); wx, wy, wz = dd(w)
    ox, oy, oz = wy - vz, uz - wx, vx - uy
    Z = 0.5 * np.mean(ox**2 + oy**2 + oz**2)
    return E, Z

if __name__ == "__main__":
    N, NU, DT, T_END = 48, 0.01, 0.005, 2.0
    out = {}
    for conv in ("sou", "central"):
        s = PISOSolver(N, warp=1e-9, nu=NU, dt=DT, corrector_steps=2, periodic=True,
                       scheme="rotational", time_scheme="bdf2", convection=conv,
                       pressure_tol=1e-11)
        u0, v0, w0 = tgv_init(s.x, s.y, s.z)
        s.u, s.v, s.w = u0.copy(), v0.copy(), w0.copy()
        ts, Es, Zs, divs = [0.0], [], [], []
        E, Z = diagnostics(s); Es.append(E); Zs.append(Z)
        t0 = time.time()
        for it in range(int(round(T_END / DT))):
            d = s.step()
            E, Z = diagnostics(s)
            ts.append((it+1)*DT); Es.append(E); Zs.append(Z); divs.append(d)
            if it % 100 == 0:
                print(f"  {conv} t={ts[-1]:.2f}  E={E:.6f}  Z={Z:.4f}  divF={d:.1e} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        out[f"{conv}_t"] = np.array(ts); out[f"{conv}_E"] = np.array(Es)
        out[f"{conv}_Z"] = np.array(Zs); out[f"{conv}_div"] = np.array(divs)
        print(f"  {conv} DONE  E: {Es[0]:.6f} -> {Es[-1]:.6f}   max divF={max(divs):.2e} "
              f"{time.time()-t0:.0f}s", flush=True)
    np.savez("tgv3d.npz", N=N, nu=NU, dt=DT, **out)
    print("saved tgv3d.npz")
