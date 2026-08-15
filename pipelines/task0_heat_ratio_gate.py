# pipelines/task0_heat_ratio_gate.py
"""
Task 0 gate. Check, without repair, the heat-equation ratio identities for
the PN phase theta = y - log psi, psi_y = psi_xx / 2.

With p1 = psi_x/psi, p2 = psi_xx/psi, p3 = psi_xxx/psi, p4 = psi_xxxx/psi:

    (I1) theta_x  = -p1
    (I2) theta_y  = 1 - p2/2
    (I3) theta_xx = -(p2 - p1^2)
    (I4) theta_xy = -(p3 - p1 p2)/2
    (I5) theta_yy = -(p4 - p2^2)/4

Three legs:
  L0  lemmas: psi_xy = psi_xxx/2 and psi_yy = psi_xxxx/4 (the heat
      equation differentiated), symbolic, on the concrete PN psi
  L1  the five identities, symbolic, against PNAnsatz with symbolic a, b
  L2  numeric: (a) identity route vs direct symbolic derivatives,
      lambdified, at scattered off-core points -> machine precision test
      of the algebra; (b) identity route vs t1_common pn_exact grid
      differences at nd = 6 -> convention test, FD-sized error expected

Every failure prints and the script exits nonzero. No repairs.

Run: python task0_heat_ratio_gate.py
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

from utils.uhu_symbolic import PNAnsatz, is_zero, lambdify_np
from op_extract.t1_common import pn_exact

FAILED = []


def report(msg):
    print(msg, flush=True)


def check(name, ok):
    tag = "PASS" if ok else "FAIL"
    report(f"  [{tag}] {name}")
    if not ok:
        FAILED.append(name)


# =====================================================================
# L0 + L1: symbolic
# =====================================================================

def symbolic_leg():
    report("=" * 70)
    report("L0/L1  SYMBOLIC, PNAnsatz with symbolic weights a, b")
    report("=" * 70)
    pn = PNAnsatz(numeric_weights=False)
    x, y, psi = pn.x, pn.y, pn.psi

    dpsi = {("x",): sp.diff(psi, x),
            ("y",): sp.diff(psi, y),
            ("x", "x"): sp.diff(psi, x, 2),
            ("x", "y"): sp.diff(psi, x, y),
            ("y", "y"): sp.diff(psi, y, 2),
            ("x", "x", "x"): sp.diff(psi, x, 3),
            ("x", "x", "x", "x"): sp.diff(psi, x, 4)}

    # L0 lemmas: heat equation and its derivatives
    check("L0a  psi_y  = psi_xx / 2",
          is_zero(dpsi[("y",)] - dpsi[("x", "x")] / 2))
    check("L0b  psi_xy = psi_xxx / 2",
          is_zero(dpsi[("x", "y")] - dpsi[("x", "x", "x")] / 2))
    check("L0c  psi_yy = psi_xxxx / 4",
          is_zero(dpsi[("y", "y")] - dpsi[("x", "x", "x", "x")] / 4))

    # ratios
    p1 = dpsi[("x",)] / psi
    p2 = dpsi[("x", "x")] / psi
    p3 = dpsi[("x", "x", "x")] / psi
    p4 = dpsi[("x", "x", "x", "x")] / psi

    # direct derivatives of theta, no heat equation used
    th = pn.theta
    lhs = {"I1 theta_x": sp.diff(th, x),
           "I2 theta_y": sp.diff(th, y),
           "I3 theta_xx": sp.diff(th, x, 2),
           "I4 theta_xy": sp.diff(th, x, y),
           "I5 theta_yy": sp.diff(th, y, 2)}
    rhs = {"I1 theta_x": -p1,
           "I2 theta_y": 1 - p2 / 2,
           "I3 theta_xx": -(p2 - p1**2),
           "I4 theta_xy": -(p3 - p1 * p2) / 2,
           "I5 theta_yy": -(p4 - p2**2) / 4}

    for name in lhs:
        check(f"{name:<12s} identity", is_zero(lhs[name] - rhs[name]))
    report("")
    return pn, lhs, rhs


# =====================================================================
# L2a: numeric, identity route vs direct symbolic derivatives
# =====================================================================

def numeric_machine_leg():
    report("=" * 70)
    report("L2a  NUMERIC, identity route vs direct derivatives (single")
    report("     defect, numeric weights) at scattered off-core points")
    report("=" * 70)
    pn = PNAnsatz(numeric_weights=True)
    x, y, psi = pn.x, pn.y, pn.psi

    args = (x, y)
    f_psi = lambdify_np(args, psi)
    f_dx = {m: lambdify_np(args, sp.diff(psi, x, m)) for m in (1, 2, 3, 4)}

    th = pn.theta
    direct = {"I1": lambdify_np(args, sp.diff(th, x)),
              "I2": lambdify_np(args, sp.diff(th, y)),
              "I3": lambdify_np(args, sp.diff(th, x, 2)),
              "I4": lambdify_np(args, sp.diff(th, x, y)),
              "I5": lambdify_np(args, sp.diff(th, y, 2))}

    rng = np.random.default_rng(0)
    X = rng.uniform(-6.0, 6.0, 400)
    Y = rng.uniform(0.3, 8.0, 400)          # y > 0 branch, off core

    P = f_psi(X, Y)
    p = {m: f_dx[m](X, Y) / P for m in (1, 2, 3, 4)}
    ident = {"I1": -p[1],
             "I2": 1 - p[2] / 2,
             "I3": -(p[2] - p[1]**2),
             "I4": -(p[3] - p[1] * p[2]) / 2,
             "I5": -(p[4] - p[2]**2) / 4}

    report(f"  {'identity':<10s} {'max |diff|':>14s} {'max |rel diff|':>16s}")
    for k in ("I1", "I2", "I3", "I4", "I5"):
        d = direct[k](X, Y)
        i = ident[k]
        ad = np.max(np.abs(d - i))
        scale = np.maximum(np.abs(d), 1e-12)
        rd = np.max(np.abs(d - i) / scale)
        report(f"  {k:<10s} {ad:>14.3e} {rd:>16.3e}")
        check(f"L2a {k} machine-precision agreement", ad < 1e-10)
    report("")


# =====================================================================
# L2b: numeric, identity route vs t1_common grid differences
# =====================================================================

def numeric_grid_leg():
    report("=" * 70)
    report("L2b  NUMERIC, identity route vs t1_common pn_exact grid")
    report("     differences, nd = 6 so the naive psi assembly is safe")
    report("=" * 70)
    mu = 0.9
    fld = pn_exact(mu, nd=6, N=768)
    xg, yg = fld["x"], fld["y"]
    xd = fld["xd"]

    # naive multi-defect psi and x-derivatives in closed form. Weights
    # match t1_common / uhu_symbolic: a = (1 + e^{-nd pi})/2,
    # delta_j = (e^{-(j+1) pi} - e^{-j pi})/2. Safe here because nd = 6
    # keeps the telescoped floor at e^{-6 pi} ~ 6.5e-9 >> 1e-16.
    nd = len(xd)
    a = 0.5 * (1.0 + np.exp(-np.pi * nd))
    deltas = 0.5 * (np.exp(-np.pi * np.arange(1, nd + 1))
                    - np.exp(-np.pi * np.arange(nd)))

    # sample rows well off core, columns inside the defect row
    iy = [np.argmin(np.abs(yg - v)) for v in (1.5, 3.0, 6.0)]
    ix = np.where(np.abs(xg) < 2.5 * fld["Lsp"])[0][::4]
    Xs, Ys = np.meshgrid(xg[ix], yg[iy])

    from scipy.special import erf as _erf
    w = np.sqrt(2.0 * Ys)
    c0 = 2.0 / np.sqrt(np.pi)
    psi = np.full_like(Xs, a)
    d1 = np.zeros_like(Xs)
    d2 = np.zeros_like(Xs)
    d3 = np.zeros_like(Xs)
    d4 = np.zeros_like(Xs)
    for j, cj in enumerate(xd):
        z = (Xs - cj) / w
        e = np.exp(-z**2)
        psi += deltas[j] * _erf(z)
        d1 += deltas[j] * c0 * e / w
        d2 += deltas[j] * c0 * e * (-2.0 * z) / w**2
        d3 += deltas[j] * c0 * e * (4.0 * z**2 - 2.0) / w**3
        d4 += deltas[j] * c0 * e * (-8.0 * z**3 + 12.0 * z) / w**4

    p1, p2 = d1 / psi, d2 / psi
    kx_ident = -p1
    ky_ident = 1.0 - p2 / 2.0

    kx_grid = fld["k1"][np.ix_(iy, ix)]
    ky_grid = fld["k2"][np.ix_(iy, ix)]

    ex = np.max(np.abs(kx_ident - kx_grid))
    ey = np.max(np.abs(ky_ident - ky_grid))
    hx = xg[1] - xg[0]
    report(f"  grid spacing hx = {hx:.4f}; central differences are O(h^2)")
    report(f"  max |theta_x identity - grid| = {ex:.3e}")
    report(f"  max |theta_y identity - grid| = {ey:.3e}")
    # FD-sized, not order one: generous ceiling of 100 h^2
    ceil = 100.0 * hx**2
    check(f"L2b theta_x agreement (< {ceil:.1e})", ex < ceil)
    check(f"L2b theta_y agreement (< {ceil:.1e})", ey < ceil)

    # theta itself, same formula both sides, should be near machine
    th_ident = Ys - np.log(psi)
    th_grid = fld["theta"][np.ix_(iy, ix)]
    et = np.max(np.abs(th_ident - th_grid))
    report(f"  max |theta identity - grid|   = {et:.3e}")
    check("L2b theta agreement (< 1e-8)", et < 1e-8)
    report("")


# =====================================================================

if __name__ == "__main__":
    symbolic_leg()
    numeric_machine_leg()
    numeric_grid_leg()
    report("=" * 70)
    if FAILED:
        report(f"GATE FAILED: {len(FAILED)} check(s) failed:")
        for f in FAILED:
            report(f"  - {f}")
        sys.exit(1)
    report("GATE PASSED: all identities hold, symbolically and numerically.")
