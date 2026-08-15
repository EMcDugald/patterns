# pipelines/exact_convolution_matrix.py
"""
The uHu tensor under EXACT Gaussian averaging with a full covariance
matrix S, on both ansatze. Successor to exact_convolution_check.py:

  - scalar sigma generalized to S = L L^T, Gauss-Hermite on the whitened
    variable, one code path for circular / elliptical / rotated /
    position dependent (gh_average_matrix, gated in task3_extension_gate)
  - PN extended across y = 0 by the measured parities: T_xx, T_yy, u
    even, T_xy odd
  - the old part 4 (off-core frozen-phase table) is deleted per the
    consolidation instructions

What it answers, all on EXACT fields, no modeling layer:
  0  quadrature convergence, knee and PN-with-parity
  1  KNEE: flank angle-error ledger and the on-core flip threshold,
     isotropic pi/2 vs the S* needle. The needle prediction from the
     session: flip pushed from mu* = 0.82 to roughly mu ~ 0.975 at
     tol = 0.05 (osc leakage at the tol level is what remains).
  2  PN (nd = 6, mu = 0.9): angle error vs the exact wavevector on the
     near-core rows where the impersonation lives, isotropic pi/2 vs
     the pointwise S* field from the p-coordinates.

Run: python exact_convolution_matrix.py [--outdir DIR] [--n-gh 40]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.uhu_symbolic import (KneeAnsatz, lambdify_np, eig2x2_numeric,
                          wrap_mod_pi, gh_average_matrix, pn_chain_lnpsi,
                          pn_chain_defects, pn_p_numeric,
                          sstar_closed_form, sstar_knee_needle)

# ---------------------------------------------------------------------
# closed-form component factories
# ---------------------------------------------------------------------

_A = KneeAnsatz()
_T = _A.uhu_tensor("full")
_FK = {nm: lambdify_np((_A.x, _A.y, _A.k1, _A.k2), e) for nm, e in
       (("Txx", _T[0, 0]), ("Txy", _T[0, 1]), ("Tyy", _T[1, 1]),
        ("thx", _A.thx), ("thy", _A.thy))}


def knee_fns(mu):
    k1v = float(np.sqrt(1 - mu**2))
    f = {nm: (lambda X, Y, _g=g: _g(X, Y, k1v, mu))
         for nm, g in _FK.items()}
    f["k1v"] = k1v
    return f


def pn_fns(nd, k1):
    """PN tensor components on y > 0 from theta (stabilized) and the
    p-coordinates. u = cos theta;
      T_xx = -cos^2 th th_x^2 - cos th sin th th_xx, etc."""
    def fields(X, Y):
        out = pn_chain_lnpsi(np.asarray(X, float).ravel(),
                             np.asarray(Y, float).ravel(), nd, k1)
        th = out["theta"].reshape(np.shape(X))
        p = pn_p_numeric(np.asarray(X, float), np.asarray(Y, float),
                         nd, k1, n_p=4)
        thx = -p[0]
        thy = 1 - p[1] / 2
        thxx = -(p[1] - p[0]**2)
        thxy = -(p[2] - p[0] * p[1]) / 2
        thyy = -(p[3] - p[1]**2) / 4
        c, s = np.cos(th), np.sin(th)
        return {"Txx": -c * c * thx * thx - c * s * thxx,
                "Txy": -c * c * thx * thy - c * s * thxy,
                "Tyy": -c * c * thy * thy - c * s * thyy,
                "thx": thx, "thy": thy}

    def comp(name):
        return lambda X, Y: fields(X, Y)[name]
    return comp, fields


# ---------------------------------------------------------------------
# angle error under exact averaging with matrix S
# ---------------------------------------------------------------------

def averaged_angle_error(comp, truth, X, Y, S, n_gh, parity=None):
    """Degrees mod pi between the dominant eigenvector of the exactly
    averaged tensor and the exact wavevector at the same points.
    parity: None (knee) or dict name->'even'/'odd' (PN across the wall)."""
    par = parity or {}
    Txx = gh_average_matrix(comp("Txx"), X, Y, S, n_gh,
                            pn_parity=par.get("Txx"))
    Txy = gh_average_matrix(comp("Txy"), X, Y, S, n_gh,
                            pn_parity=par.get("Txy"))
    Tyy = gh_average_matrix(comp("Tyy"), X, Y, S, n_gh,
                            pn_parity=par.get("Tyy"))
    eg = eig2x2_numeric(Txx, Txy, Tyy)
    tr = truth(X, Y)
    ang_t = np.arctan2(tr["thy"], tr["thx"])
    return (np.degrees(np.abs(wrap_mod_pi(eg["angle_dom"] - ang_t,
                                          numeric=True))), eg)


PN_PARITY = {"Txx": "even", "Tyy": "even", "Txy": "odd"}


# ---------------------------------------------------------------------

def part_convergence(report, n_gh):
    report("=" * 70)
    report("0. QUADRATURE CONVERGENCE, matrix S, both fields")
    report("=" * 70)
    mu = 0.9
    f = knee_fns(mu)
    Sk = np.array([[1.5**2, 0.3], [0.3, 0.7**2]])
    X, Y = np.meshgrid(np.linspace(0, 2 * np.pi / f["k1v"], 7),
                       np.linspace(0.3, 3.0, 5))
    ref = None
    report(f"  knee, rotated ellipse: {'n_gh':>6} {'max |delta|':>14}")
    for n in (16, 24, 32, 40):
        v = gh_average_matrix(f["Txx"], X, Y, Sk, n)
        if ref is not None:
            report(f"  {'':>23}{n:>6} {np.max(np.abs(v - ref)):>14.3e}")
        ref = v

    k1v = float(np.sqrt(1 - mu**2))
    comp, _ = pn_fns(6, k1v)
    _, L = pn_chain_defects(6, k1v)
    Xp = np.linspace(-L, L, 5)
    Yp = np.full_like(Xp, 0.25 * L)
    Sp = np.array([[1.2**2, 0.0], [0.0, 1.6**2]])   # crosses y = 0
    ref = None
    report(f"  PN, parity across wall: {'n_gh':>5} {'max |delta|':>14}")
    for n in (16, 24, 32, 40):
        v = gh_average_matrix(comp("Txy"), Xp, Yp, Sp, n,
                              pn_parity="odd")
        if ref is not None:
            report(f"  {'':>23}{n:>6} {np.max(np.abs(v - ref)):>14.3e}")
        ref = v
    report(f"  using n_gh = {n_gh}")
    report("")


def part_knee(report, n_gh, outdir, tol):
    report("=" * 70)
    report("1. KNEE, EXACT FIELD: isotropic pi/2 vs the S* needle")
    report("=" * 70)
    report("  flank patch 0.2 < y < 4, one period in x; median |angle "
           "error| (deg)")
    report(f"  {'mu':>5} {'iso pi/2':>10} {'needle S*':>10}")
    for mu in (0.5, 0.9):
        f = knee_fns(mu)
        truth = lambda X, Y, f=f: {"thx": f["thx"](X, Y),
                                   "thy": f["thy"](X, Y)}
        comp = lambda name, f=f: f[name]
        X, Y = np.meshgrid(np.linspace(0, 2 * np.pi / f["k1v"], 36),
                           np.linspace(0.2, 4.0, 28))
        e_iso, _ = averaged_angle_error(comp, truth, X, Y,
                                        (np.pi / 2)**2 * np.eye(2), n_gh)
        Sn = sstar_knee_needle(mu, tol=tol)
        e_st, _ = averaged_angle_error(comp, truth, X, Y, Sn, n_gh)
        report(f"  {mu:>5.2f} {np.nanmedian(e_iso):>10.3f} "
               f"{np.nanmedian(e_st):>10.3f}")

    report("")
    report("  on-core flip threshold mu*: flipped fraction > 1/2 over the")
    report("  core row (criterion identical to the old machine)")

    def flipped_fraction(mu, S):
        f = knee_fns(mu)
        xs = np.linspace(0, 2 * np.pi / f["k1v"], 48, endpoint=False)
        ys = np.zeros_like(xs)
        comp = lambda name, f=f: f[name]
        truth = lambda X, Y, f=f: {"thx": f["thx"](X, Y),
                                   "thy": f["thy"](X, Y)}
        _, eg = averaged_angle_error(comp, truth, xs, ys, S, n_gh)
        ang = np.abs(wrap_mod_pi(eg["angle_dom"], numeric=True))
        return float(np.mean(ang > np.pi / 4))

    S_iso = (np.pi / 2)**2 * np.eye(2)
    g = lambda m: flipped_fraction(m, S_iso) - 0.5
    mstar_iso = brentq(g, 0.3, 0.995, xtol=2e-4)
    report(f"  isotropic pi/2 : mu* = {mstar_iso:.4f}   "
           f"(old machine: 0.8203, continuity check)")

    mus = np.linspace(0.90, 0.999, 12)
    fr = []
    for m in mus:
        Sn = sstar_knee_needle(m, tol=tol)
        fr.append(flipped_fraction(m, Sn))
    fr = np.array(fr)
    if np.any(fr > 0.5):
        g2 = lambda m: flipped_fraction(m, sstar_knee_needle(m, tol)) - 0.5
        lo = mus[np.argmax(fr > 0.5) - 1]
        hi = mus[np.argmax(fr > 0.5)]
        mstar_needle = brentq(g2, lo, hi, xtol=2e-4)
        report(f"  S* needle      : mu* = {mstar_needle:.4f}   "
               f"(session prediction from osc leakage: ~0.975 at "
               f"tol = {tol})")
    else:
        mstar_needle = np.nan
        report(f"  S* needle      : NO flip found up to mu = {mus[-1]:.3f}"
               f"   (prediction was ~0.975 at tol = {tol}; measured "
               "leakage is evidently below the crossing level)")
    report("")
    return mstar_iso, mstar_needle


def part_pn(report, n_gh, outdir):
    report("=" * 70)
    report("2. PN, EXACT FIELD, ACROSS THE WALL: iso pi/2 vs S* field")
    report("=" * 70)
    mu = 0.9
    k1v = float(np.sqrt(1 - mu**2))
    nd = 6
    comp, fields = pn_fns(nd, k1v)
    _, L = pn_chain_defects(nd, k1v)
    xd, _ = pn_chain_defects(nd, k1v)

    def truth(X, Y):
        p = pn_p_numeric(np.asarray(X, float), np.asarray(Y, float),
                         nd, k1v, n_p=2)
        return {"thx": -p[0], "thy": 1 - p[1] / 2}

    def sstar_field(X, Y, cap=4 * L, tol=0.05):
        p = pn_p_numeric(np.asarray(X, float), np.asarray(Y, float),
                         nd, k1v, n_p=4)
        kx, ky = -p[0], 1 - p[1] / 2
        H11 = -(p[1] - p[0]**2)
        H12 = -(p[2] - p[0] * p[1]) / 2
        H22 = -(p[3] - p[1]**2) / 4
        return sstar_closed_form(kx, ky, H11, H12, H22, tol=tol,
                                 cap=cap)["S"]

    report("  rows at heights y/L; 40 x-points across the two central")
    report("  spacings; angle error vs the exact wavevector (deg)")
    report(f"  {'y/L':>6} {'iso median':>11} {'iso max':>9} "
           f"{'S* median':>10} {'S* max':>8}")
    S_iso = (np.pi / 2)**2 * np.eye(2)
    for yfrac in (0.20, 0.30, 0.60):
        X = np.linspace(xd[2], xd[3], 40)
        Y = np.full_like(X, yfrac * L)
        e_iso, _ = averaged_angle_error(comp, truth, X, Y, S_iso, n_gh,
                                        parity=PN_PARITY)
        e_st, _ = averaged_angle_error(comp, truth, X, Y,
                                       sstar_field(X, Y), n_gh,
                                       parity=PN_PARITY)
        report(f"  {yfrac:>6.2f} {np.nanmedian(e_iso):>11.3f} "
               f"{np.nanmax(e_iso):>9.2f} {np.nanmedian(e_st):>10.3f} "
               f"{np.nanmax(e_st):>8.2f}")
    report("")
    report("  The 0.2L to 0.3L rows are the near-core zone where flipped")
    report("  knees impersonate PN; the S* column is the exact-field")
    report("  verdict on kernel design there.")
    report("")


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    lines = []

    def report(s=""):
        print(s, flush=True)
        lines.append(s)

    part_convergence(report, args.n_gh)
    part_knee(report, args.n_gh, args.outdir, args.tol)
    part_pn(report, args.n_gh, args.outdir)

    with open(os.path.join(args.outdir, "exact_convolution_matrix.txt"),
              "w") as fh:
        fh.write("\n".join(lines) + "\n")


class _Args:
    outdir = "out_exact_matrix"
    n_gh = 40
    tol = 0.05


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=_Args.outdir)
    p.add_argument("--n-gh", type=int, default=_Args.n_gh)
    p.add_argument("--tol", type=float, default=_Args.tol)
    main(p.parse_args())
