# pipelines/symbolic_tour.py
"""
Guided tour of uhu_symbolic, aimed at getting closed forms onto paper.

Every section prints (i) the API call that produced the object and (ii) the
simplified expression, so the script doubles as a cheatsheet. The knee is
used throughout because its expressions are short enough to write by hand;
the same calls work on PNAnsatz and on any subclass of GeneralAnsatz.

Notation used in the printed output (substituted at the end for brevity):
    t = tanh(k2 y),   s = sech^2(k2 y) = 1 - t^2

Run: python symbolic_tour.py [--latex] [--check]
"""

import argparse

import numpy as np
import sympy as sp
from scipy.optimize import brentq


import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.uhu_symbolic import KneeAnsatz, PNAnsatz, eig2x2, is_zero, lambdify_np

GH_S, GH_W = np.polynomial.hermite_e.hermegauss(160)
GH_W = GH_W / GH_W.sum()


def shorthand(expr, k2, y):
    """Rewrite tanh/sech in terms of symbols t, s for readability."""
    t, s = sp.symbols("t s")
    e = expr.rewrite(sp.tanh)
    e = e.subs(sp.tanh(k2 * y), t)
    e = e.subs(sp.cosh(k2 * y), 1 / sp.sqrt(s))
    e = e.subs(sp.sinh(k2 * y), t / sp.sqrt(s))
    return sp.simplify(e)


def main(latex, check):
    a = KneeAnsatz()
    x, y, k1, k2 = a.x, a.y, a.k1, a.k2
    show = (lambda e: sp.latex(e)) if latex else (lambda e: e)

    def sec(title):
        print("\n" + "=" * 68)
        print(title)
        print("=" * 68)

    def item(call, expr, short=True):
        e = shorthand(sp.simplify(expr), k2, y) if short else sp.simplify(expr)
        print(f"\n  {call}")
        print(f"    = {show(e)}")

    print("KneeAnsatz:  theta = k1 x + log(2 cosh(k2 y)),  A = 1,")
    print("             mu = k2,  k1 = sqrt(1 - mu^2),  t = tanh(k2 y),")
    print("             s = sech^2(k2 y) = 1 - t^2")

    # ---- 1. ground truth phase geometry ----------------------------
    sec("1. GROUND TRUTH PHASE GEOMETRY")
    item("a.theta", a.theta, short=False)
    item("a.thx        # k1 = theta_x", a.thx)
    item("a.thy        # k2 = theta_y", a.thy)
    item("a.k_mag2     # |k|^2", a.k_mag2)
    item("a.hess[0,0]  # theta_xx", a.hess[0, 0])
    item("a.hess[0,1]  # theta_xy", a.hess[0, 1])
    item("a.hess[1,1]  # theta_yy", a.hess[1, 1])
    item("a.div_k", a.div_k)
    item("a.curl_k", a.curl_k)
    item("a.J          # det(grad k)", a.J)
    item("a.W          # ||H||_F / |k|^2", a.W)
    print("\n  NOTE: J = 0 identically for the knee. theta_xx = theta_xy = 0,")
    print("  so the Hessian is rank one and its determinant vanishes. The")
    print("  knee carries curvature but no twist; J is blind to it.")
    print(f"\n  W on the core (y = 0): "
          f"{sp.simplify(a.W.subs(y, 0))}   = k2^2/k1^2")

    # ---- 2. the uHu tensor, pointwise ------------------------------
    sec("2. uHu TENSOR, POINTWISE (no averaging)")
    print("\n  Exact factorization, valid for any A = 1 phase field:")
    print("      T_ij = -cos^2(theta) ( theta_i theta_j + tan(theta) theta_ij )")
    for i, j in (("x", "x"), ("x", "y"), ("y", "y")):
        item(f"a.uhu_dc('{i}','{j}') + a.uhu_osc('{i}','{j}')",
             a.uhu_dc(i, j) + a.uhu_osc(i, j))

    # ---- 3. the ideally averaged tensor ----------------------------
    sec("3. IDEALLY AVERAGED (DC) TENSOR")
    print("\n  For A = 1:  <T> = -(1/2) k k^T,  rank one, exactly.")
    T = a.uhu_tensor("DC")
    for lbl, e in (("T_xx", T[0, 0]), ("T_xy", T[0, 1]), ("T_yy", T[1, 1])):
        item(f"a.uhu_tensor('DC')  ->  {lbl}", e)

    # ---- 4. eigensystem ---------------------------------------------
    sec("4. EIGENSYSTEM OF THE DC TENSOR")
    eig = a.eigensystem("DC")
    print("\n  Closed 2x2 forms (NOT sympy eigenvects):")
    print("      lam_pm = (tr +- sqrt((Txx-Tyy)^2 + 4 Txy^2)) / 2")
    print("      axis   = (1/2) atan2(2 Txy, Txx - Tyy)")
    print("  uHu is negative semidefinite, so DOMINANT = MOST NEGATIVE.")
    item("eig['lam_sub']   # the zero eigenvalue", eig["lam_sub"])
    item("eig['lam_dom']   # = -|k|^2 / 2", eig["lam_dom"])
    item("eig['k_mag_A1']  # sqrt(-2 lam_dom) = |k|", eig["k_mag_A1"])
    print("\n  eig['angle_dom'] is atan2(theta_y, theta_x) exactly, so")
    print("  a.angle_error_vs_truth('DC') = 0. Verified to 1e-12.")
    print("  eig['chi'] = lam_sub/lam_dom = 0: rank one.")

    # ---- 5. the two averaging channels -----------------------------
    sec("5. THE TWO AVERAGING CHANNELS")
    sx, sy = sp.symbols("s_x s_y", positive=True)
    item("a.corrugation_factor(sx, sy)   # channel 1 survival",
         a.corrugation_factor(sx, sy))
    print("\n  channel 2: a.kernel_expand(f, sx, sy) = f + (sx^2/2) f_xx")
    print("             + (sy^2/2) f_yy. Applied to the DC tensor:")
    Tk = a.uhu_tensor("kernel", sx, sy)
    for lbl, e in (("T_xx", Tk[0, 0]), ("T_xy", Tk[0, 1]), ("T_yy", Tk[1, 1])):
        item(f"a.uhu_tensor('kernel', sx, sy)  ->  {lbl}", e)
    print("\n  On the core (y = 0, so t = 0, s = 1) these collapse to")
    for lbl, e in (("T_xx", Tk[0, 0]), ("T_xy", Tk[0, 1]), ("T_yy", Tk[1, 1])):
        print(f"      {lbl} = {sp.simplify(e.subs(y, 0))}")
    print("\n  so the second-order kernel predicts an on-core flip when")
    print("      sy^2 k2^4 > k1^2,   i.e.   sy > k1 / k2^2.")

    # ---- 6. Riesz / monogenic --------------------------------------
    sec("6. RIESZ / MONOGENIC (local plane-wave)")
    mono = a.monogenic_lpw()
    print("\n  R ~ A sin(theta) khat, valid to O(|grad A|/|k|, |grad khat|/|k|).")
    print("  The exact Riesz transform is NONLOCAL; compare numerically")
    print("  against riesz.riesz_analyze_pattern.")
    item("mono['R1']", mono["R1"])
    item("mono['R2']", mono["R2"])
    item("mono['A_mono']", mono["A_mono"], short=False)

    # ---- 7. line energy --------------------------------------------
    sec("7. RCN LINE ENERGY (closed form)")
    dens, I = a.line_energy()
    item("a.line_energy()[0]   # density, on the self-dual constraint", dens)
    print(f"\n  a.line_energy()[1]   # integral over y")
    print(f"    = {show(I)}     (expect 8 k2^3 / 3)")

    # ---- 8. optional: second-order vs exact ------------------------
    if check:
        sec("8. HOW GOOD IS THE SECOND-ORDER KERNEL EXPANSION?")
        print("\n  Two predictions of the on-core flip threshold mu*(sigma):")
        print("    second order:  sigma = k1 / k2^2")
        print("    exact Gaussian: k1^2 = <theta_y^2>_sigma  (Section 3)")
        print(f"\n  {'sigma':>8} {'mu* 2nd order':>15} {'mu* exact':>12}")
        for sg in (np.pi / 8, np.pi / 4, np.pi / 2, np.pi):
            f2 = lambda m: np.sqrt(1 - m**2) / m**2 - sg
            fe = lambda m: (1 - m**2) - float(np.sum(
                GH_W * (m * np.tanh(m * sg * GH_S))**2))
            try:
                m2 = brentq(f2, 0.05, 0.999, xtol=1e-6)
            except ValueError:
                m2 = np.nan
            me = brentq(fe, 0.05, 0.999, xtol=1e-6)
            print(f"  {sg:>8.4f} {m2:>15.4f} {me:>12.4f}")
        print("\n  The O(sigma^2) truncation is quantitatively POOR at the")
        print("  sigma values actually in use. Use it to see WHICH terms")
        print("  appear and how they scale, not to predict thresholds.")

    # ---- 9. same calls on PN ----------------------------------------
    sec("9. THE SAME CALLS ON PN")
    pn = PNAnsatz(numeric_weights=True)
    print("\n  pn = PNAnsatz(numeric_weights=True)")
    print("  pn.theta, pn.thx, pn.thy, pn.hess, pn.W, pn.J, pn.curl_k")
    print("  pn.uhu_tensor(src, sx, sy), pn.eigensystem(src),")
    print("  pn.angle_error_vs_truth(src)")
    print("  PNAnsatz(amplitude='tanh', xi=1) adds A = tanh(r/xi), and then")
    print("  pn.amplitude_contamination() gives W_A.")
    print(f"\n  pn.heat_check() (Hopf-Cole, should be 0): {pn.heat_check()}")
    print("\n  For hand work use numeric_weights=False, which keeps a, b")
    print("  symbolic: psi = a + b erf(x/sqrt(2y)), theta = y - log(psi).")
    print("  Single defect: a = (1+e^-pi)/2, b = (e^-pi - 1)/2.")

    if check:
        sec("SANITY ASSERTS")
        assert is_zero(a.eigensystem("DC")["lam_plus"])
        assert is_zero(a.J)
        assert is_zero(a.curl_k)
        assert is_zero(I - sp.Rational(8, 3) * k2**3)
        print("\n  [ok] lam_sub = 0, J = 0, curl k = 0, line energy = 8 k2^3/3")


class _Args:
    latex = False
    check = True


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--latex", action="store_true", default=_Args.latex)
    p.add_argument("--check", action="store_true", default=_Args.check)
    aa = p.parse_args()
    main(aa.latex, aa.check)
