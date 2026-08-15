# pipelines/task3_extension_gate.py
"""
Validation gate for the uhu_symbolic extension. Every new code path is
checked against an independent reference before anything is built on it.

  G1  PRatio dx / dy closure vs direct differentiation on concrete psi
  G2  pn_p_numeric vs lambdified symbolic p's (single defect)
  G3  gh_average_matrix: (a) scalar-sigma agreement with the old
      exact_average, (b) rotated elliptical vs brute Riemann convolution,
      (c) pn_parity plumbing vs an explicitly reflected integrand
  G4  taylor_average order 1 == the H S H identity, symbolically
  G5  model_averaged_tensor vs GH quadrature of the model integrand
  G6  Riesz plane-wave model vs true FFT Riesz on the exact knee field
      (measures the size of the plane-wave correction, which is the
      model error term (iii))

Run: python task3_extension_gate.py
"""

import sys
from pathlib import Path

import numpy as np
import sympy as sp

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.uhu_symbolic import (KneeAnsatz, PNAnsatz, PRatio, is_zero,
                          lambdify_np, pn_p_numeric, gh_average_matrix,
                          taylor_average, model_averaged_tensor,
                          sstar_knee_needle)

FAILED = []


def report(msg=""):
    print(msg, flush=True)


def check(name, ok):
    tag = "PASS" if ok else "FAIL"
    report(f"  [{tag}] {name}")
    if not ok:
        FAILED.append(name)


def part_G1():
    report("=" * 70)
    report("G1  PRatio closure vs direct differentiation")
    report("=" * 70)
    pr = PRatio(max_order=8)
    pn = PNAnsatz(numeric_weights=False)
    x, y = pn.x, pn.y
    smap = pr.subs_map(pn.psi, x)

    cases = [
        ("d/dx of theta_x  = theta_xx", pr.dx(pr.theta_x()),
         sp.diff(pn.theta, x, 2)),
        ("d/dy of theta_x  = theta_xy", pr.dy(pr.theta_x()),
         sp.diff(pn.theta, x, y)),
        ("d/dy of theta_y  = theta_yy", pr.dy(pr.theta_y()),
         sp.diff(pn.theta, y, 2)),
        ("d/dy of H_xy      (third order)", pr.dy(pr.hess()[0, 1]),
         sp.diff(pn.theta, x, y, y)),
        ("d/dx of J         (third order)", pr.dx(pr.J()),
         sp.diff(sp.Matrix([[sp.diff(pn.theta, x, 2),
                             sp.diff(pn.theta, x, y)],
                            [sp.diff(pn.theta, x, y),
                             sp.diff(pn.theta, y, 2)]]).det(), x)),
    ]
    for name, pexpr, direct in cases:
        check(f"{name}", is_zero(sp.simplify(pexpr.subs(smap) - direct)))
    report("")


def part_G2():
    report("=" * 70)
    report("G2  pn_p_numeric vs symbolic p's (single defect, 300 points)")
    report("=" * 70)
    pn = PNAnsatz(numeric_weights=True)
    x, y = pn.x, pn.y
    fs = [lambdify_np((x, y), sp.diff(pn.psi, x, m) / pn.psi)
          for m in range(1, 5)]
    rng = np.random.default_rng(1)
    X = rng.uniform(-4, 4, 300)
    Y = rng.uniform(0.3, 6.0, 300)
    P = pn_p_numeric(X, Y, nd=1, k1=np.sqrt(1 - 0.81), n_p=4)
    # nd=1 library chain centers its defect at 0 like PNAnsatz
    worst = max(np.max(np.abs(P[m](X, Y) if callable(P[m]) else P[m]
                              - fs[m](X, Y))) for m in range(4))
    report(f"    max |library - symbolic| over p1..p4: {worst:.3e}")
    check("G2 agreement < 1e-10", worst < 1e-10)
    report("")


def part_G3():
    report("=" * 70)
    report("G3  gh_average_matrix")
    report("=" * 70)
    a = KneeAnsatz()
    mu = 0.9
    k1v = float(np.sqrt(1 - mu**2))
    T = a.uhu_tensor("full")
    fTxx = lambdify_np((a.x, a.y, a.k1, a.k2), T[0, 0])
    fn = lambda X, Y: fTxx(X, Y, k1v, mu)

    # (a) scalar sigma vs the old axis-aligned quadrature
    from exact_convolution_check import gh_nodes, exact_average
    X0, Y0 = np.meshgrid(np.linspace(0, 5, 7), np.linspace(0.3, 3, 5))
    sg = np.pi / 3
    s, w = gh_nodes(40)
    old = exact_average(fn, X0, Y0, sg, sg, s, w)
    new = gh_average_matrix(fn, X0, Y0, sg**2 * np.eye(2), n_gh=40)
    d = np.max(np.abs(old - new))
    report(f"    (a) scalar sigma: max |old - new| = {d:.3e}")
    check("G3a scalar-sigma agreement < 1e-12", d < 1e-12)

    # (b) rotated elliptical vs brute Riemann convolution
    ang = 0.6
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    S = R @ np.diag([1.3**2, 0.4**2]) @ R.T
    x0, y0 = 1.7, 1.1
    gv = np.linspace(-6, 6, 801)
    GX, GY = np.meshgrid(gv, gv)
    Si = np.linalg.inv(S)
    q = Si[0, 0] * GX**2 + 2 * Si[0, 1] * GX * GY + Si[1, 1] * GY**2
    ker = np.exp(-q / 2)
    ker /= ker.sum()
    brute = np.sum(ker * fn(x0 + GX, y0 + GY))
    new = float(gh_average_matrix(fn, np.array(x0), np.array(y0), S,
                                  n_gh=60))
    report(f"    (b) rotated ellipse: |brute - GH| = {abs(brute-new):.3e}")
    check("G3b rotated-elliptical agreement < 1e-8",
          abs(brute - new) < 1e-8)

    # (c) parity plumbing: SMOOTH even extension (all odd y-derivatives
    # vanish at 0, as for the true PN tensor components). A kinked
    # extension is measured to stall near 1e-3 and is documented as out
    # of scope in the docstring.
    def f_half(X, Y):
        return np.cos(1.3 * X) * np.exp(-0.5 * Y**2) * (1 + Y**2)

    def f_ext(X, Y):
        return f_half(X, np.abs(Y))

    S2 = np.diag([0.8**2, 1.1**2])
    p_even = float(gh_average_matrix(f_half, np.array(0.4), np.array(0.5),
                                     S2, n_gh=48, pn_parity="even"))
    ref = np.sum(_brute_kernel(S2, 0.4, 0.5, f_ext))
    report(f"    (c) parity 'even', smooth extension: "
           f"|wrapped - reflected brute| = {abs(p_even - ref):.3e}")
    check("G3c parity plumbing (smooth extension) < 1e-6",
          abs(p_even - ref) < 1e-6)
    report("")


def _brute_kernel(S, x0, y0, f):
    gv = np.linspace(-7, 7, 901)
    GX, GY = np.meshgrid(gv, gv)
    Si = np.linalg.inv(S)
    q = Si[0, 0] * GX**2 + 2 * Si[0, 1] * GX * GY + Si[1, 1] * GY**2
    ker = np.exp(-q / 2)
    ker /= ker.sum()
    return ker * f(x0 + GX, y0 + GY)


def part_G4():
    report("=" * 70)
    report("G4  taylor_average order 1 == H S H identity, symbolically")
    report("=" * 70)
    x, y = sp.symbols("x y", real=True)
    th = sp.Function("theta")(x, y)
    S11, S22 = sp.symbols("S11 S22", positive=True)
    S12 = sp.symbols("S12", real=True)
    H = sp.Matrix([[sp.diff(th, x, 2), sp.diff(th, x, y)],
                   [sp.diff(th, x, y), sp.diff(th, y, 2)]])
    Sm = sp.Matrix([[S11, S12], [S12, S22]])
    HSH = H * Sm * H
    c = {0: x, 1: y}
    ok = True
    for i in range(2):
        for j in range(2):
            Q = sp.diff(th, c[i]) * sp.diff(th, c[j])
            got = taylor_average(Q, x, y, (S11, S12, S22), order=1) - Q
            third = sum(Sm[a, b] * (sp.diff(th, c[i], c[a], c[b])
                                    * sp.diff(th, c[j])
                                    + sp.diff(th, c[i])
                                    * sp.diff(th, c[j], c[a], c[b]))
                        for a in range(2) for b in range(2)) / 2
            if sp.simplify(got - HSH[i, j] - third) != 0:
                ok = False
    check("G4 order-1 layer reproduces (H S H)_ij + transport terms", ok)
    report("    (order >= 2 available; labeled [taylor], numbers still")
    report("     to be taken from exact quadrature)")
    report("")


def part_G5():
    report("=" * 70)
    report("G5  model_averaged_tensor vs GH quadrature of the model")
    report("=" * 70)
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(4):
        ang = rng.uniform(0, 2 * np.pi)
        k = rng.uniform(0.6, 1.1) * np.array([np.cos(ang), np.sin(ang)])
        B = rng.uniform(-0.5, 0.5, (2, 2))
        S = B @ B.T + 0.2 * np.eye(2)
        Hm = rng.uniform(-0.4, 0.4, (2, 2))
        Hm = 0.5 * (Hm + Hm.T)
        th0 = rng.uniform(0, 2 * np.pi)
        pred = model_averaged_tensor(k, Hm, S, th0)

        L = np.linalg.cholesky(S)
        s, w = np.polynomial.hermite_e.hermegauss(70)
        w = w / w.sum()
        Z1, Z2 = np.meshgrid(s, s, indexing="ij")
        xi = L @ np.vstack([Z1.ravel(), Z2.ravel()])
        Wq = np.outer(w, w).ravel()
        g1 = k[0] + Hm[0, 0] * xi[0] + Hm[0, 1] * xi[1]
        g2 = k[1] + Hm[0, 1] * xi[0] + Hm[1, 1] * xi[1]
        th = th0 + k[0] * xi[0] + k[1] * xi[1] + 0.5 * (
            Hm[0, 0] * xi[0]**2 + 2 * Hm[0, 1] * xi[0] * xi[1]
            + Hm[1, 1] * xi[1]**2)
        for (i, j), gi, gj in (((0, 0), g1, g1), ((0, 1), g1, g2),
                               ((1, 1), g2, g2)):
            v = np.sum(Wq * (-0.5 * gi * gj * (1 + np.cos(2 * th))
                             - 0.5 * np.sin(2 * th) * Hm[i, j]))
            worst = max(worst, abs(v - pred[i, j]))
    report(f"    max |formula - quadrature| over draws/components: "
           f"{worst:.3e}")
    check("G5 model tensor formula < 1e-12", worst < 1e-12)
    report("")


def part_G6():
    report("=" * 70)
    report("G6  Riesz plane-wave model vs true FFT Riesz on the knee")
    report("=" * 70)
    mu = 0.9
    k1v, k2v = np.sqrt(1 - mu**2), mu
    N = 768
    Lx = 6 * 2 * np.pi / k1v
    Ly = 40.0
    xs = np.linspace(0, Lx, N, endpoint=False)
    ys = np.linspace(-Ly / 2, Ly / 2, N, endpoint=False)
    X, Y = np.meshgrid(xs, ys)
    theta = k1v * X + np.log(2 * np.cosh(k2v * Y))
    u = np.cos(theta)

    from numpy.fft import fft2, ifft2, fftfreq
    kxf = fftfreq(N, xs[1] - xs[0])
    kyf = fftfreq(N, ys[1] - ys[0])
    KX, KY = np.meshgrid(kxf, kyf)
    Kn = np.sqrt(KX**2 + KY**2) + 1e-12
    U = fft2(u)
    R1 = np.real(ifft2(-1j * (KX / Kn) * U))
    R2 = np.real(ifft2(-1j * (KY / Kn) * U))

    thx = np.full_like(X, k1v)
    thy = k2v * np.tanh(k2v * Y)
    kn = np.hypot(thx, thy)
    model1 = np.sin(theta) * thx / kn
    model2 = np.sin(theta) * thy / kn

    core = np.abs(Y) < Ly / 5
    e1 = np.abs(R1 - model1)[core]
    e2 = np.abs(R2 - model2)[core]
    report(f"    median |R1 - model| = {np.median(e1):.4f}, "
           f"|R2 - model| = {np.median(e2):.4f}   (u has amplitude 1)")
    report("    this residual IS the plane-wave correction, the model's")
    report("    only H-bearing term; it is measured, not estimated.")
    check("G6a plane-wave model tracks true Riesz to a few percent",
          np.median(e1) < 0.05 and np.median(e2) < 0.05)

    # structure of the raw orientation error: it must concentrate where
    # the energy weight vanishes (claim (ii) of the model class)
    phi = np.arctan2(R2, R1)
    phi_k = np.arctan2(thy, thx)
    derr = np.abs(np.angle(np.exp(2j * (phi - phi_k)))) / 2
    band = (np.abs(Y) > 1.0) & (np.abs(Y) < Ly / 5)
    w = R1**2 + R2**2
    hi = np.degrees(np.median(derr[band & (w > 0.5)]))
    lo = np.degrees(np.median(derr[band & (w < 0.1)]))
    report(f"    raw orientation error: median {hi:.2f} deg at high "
           f"weight, {lo:.2f} deg at low weight")
    check("G6b errors concentrate where the weight vanishes "
          "(low-weight > 3x high-weight, high-weight < 3 deg)",
          lo > 3 * hi and hi < 3.0)

    # the working-point regression: sigma_dir = 0.4 holds the knee core,
    # pi/2 loses it (the user-established rule, now measured here)
    from scipy.ndimage import gaussian_filter
    dxg, dyg = xs[1] - xs[0], ys[1] - ys[0]
    rows = {}
    for sig in (0.4, np.pi / 2):
        sm = lambda a, s=sig: gaussian_filter(a, [s / dyg, s / dxg],
                                              mode="wrap")
        c2, s2 = sm(w * np.cos(2 * phi)), sm(w * np.sin(2 * phi))
        ang = 0.5 * np.arctan2(s2, c2)
        err = np.abs(np.angle(np.exp(2j * (ang - phi_k)))) / 2
        rows[sig] = np.degrees(np.median(err[np.abs(Y) < 0.3]))
        report(f"    smoothed director, sigma_dir = {sig:.3f}: knee core "
               f"median error {rows[sig]:.2f} deg")
    check("G6c knee core: sigma_dir 0.4 works (< 8 deg), pi/2 does not "
          "(> 20 deg), reproducing the established rule",
          rows[0.4] < 8.0 and rows[np.pi / 2] > 20.0)
    report("")


if __name__ == "__main__":
    part_G1()
    part_G2()
    part_G3()
    part_G4()
    part_G5()
    part_G6()
    report("=" * 70)
    if FAILED:
        report(f"EXTENSION GATE FAILED: {len(FAILED)}")
        for f in FAILED:
            report(f"  - {f}")
        sys.exit(1)
    report("EXTENSION GATE PASSED: all new code paths verified.")
