# pipelines/exact_convolution_check.py
"""
The uHu tensor under EXACT Gaussian averaging, with no series expansion and
no frozen-phase model, compared against the exact analytic wavevector.

Why this exists. eigen_averaging_track.py has four averaging models, three
of them exact ("full", "DC", and "corrugated" on the knee core) and one of
them a second-order truncation ("kernel"). The truncation is quantitatively
poor at the sigma actually in use: on core it predicts a flip threshold
mu* = 0.68 at sigma = pi/2 against the exact Gaussian's 0.82. Every number
in the "kernel" column therefore shows a real trend but is not a real
number. This script removes the modeling layer entirely.

Method. The knee's tensor components T_ij = u d_i d_j u are available in
closed form, so the ONLY numerical operation needed is the average itself:

    <T_ij>(x0, y0) = sum_ab w_a w_b T_ij(x0 + sx s_a, y0 + sy s_b)

by tensor-product Gauss-Hermite quadrature. This is better than an FFT
convolution here for two reasons: the knee is not periodic in y, so an FFT
would need a taper or a padded box, and the quadrature converges
spectrally on these smooth integrands, so 40 nodes per dimension is already
at machine precision. Convergence is checked, not assumed.

What it answers:
  1  the Part C ledger with a TRUE column beside the modeled ones
  2  where the real optimum in sigma sits on the flank
  3  the on-core mu*(sigma) on a fine mu grid, which is the measurement the
     Section 3 attribution question needs
  4  whether the frozen-phase corrugation model holds OFF core, which
     eigen_averaging_track.py assumes without evidence

Scope: knee only. PN needs the y > 0 branch mirrored before the quadrature
can sample across y = 0, which is a separate piece of work.

Run: python exact_convolution_check.py [--outdir DIR] [--n-gh 40]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.optimize import brentq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.uhu_symbolic import (KneeAnsatz, eig2x2_numeric, wrap_mod_pi,
                                lambdify_np)


# =====================================================================
# exact averaging by Gauss-Hermite quadrature
# =====================================================================

def gh_nodes(n):
    """Probabilists' Gauss-Hermite nodes/weights, normalized so that the
    weights sum to 1. Then sum_a w_a f(x + sigma s_a) is exactly the
    Gaussian average of f with standard deviation sigma."""
    s, w = np.polynomial.hermite_e.hermegauss(n)
    return s, w / w.sum()


def exact_average(fn, X0, Y0, sx, sy, s, w):
    """
    <fn>(X0, Y0) under an axis-aligned Gaussian of widths (sx, sy), by
    tensor-product Gauss-Hermite quadrature.

    fn takes arrays. X0, Y0 broadcast together to the point shape; the
    return has that shape. The two shifts are independent, so the nodes go
    on two leading axes and the weights contract them at the end.
    """
    X0 = np.asarray(X0, dtype=float)
    Y0 = np.asarray(Y0, dtype=float)
    nd = len(np.broadcast_shapes(X0.shape, Y0.shape))
    sa = s.reshape((-1, 1) + (1,) * nd)
    sb = s.reshape((1, -1) + (1,) * nd)
    vals = fn(X0[None, None, ...] + sx * sa,
              Y0[None, None, ...] + sy * sb)
    return np.einsum("a,b,ab...->...", w, w, vals)


# Build and lambdify the knee symbolics ONCE, with (k1, k2) left free, so
# that root-finding over mu does not pay the sympy cost on every call.
_A = KneeAnsatz()
_T = _A.uhu_tensor("full")
_F = {
    "Txx": lambdify_np((_A.x, _A.y, _A.k1, _A.k2), _T[0, 0]),
    "Txy": lambdify_np((_A.x, _A.y, _A.k1, _A.k2), _T[0, 1]),
    "Tyy": lambdify_np((_A.x, _A.y, _A.k1, _A.k2), _T[1, 1]),
    "thx": lambdify_np((_A.x, _A.y, _A.k1, _A.k2), _A.thx),
    "thy": lambdify_np((_A.x, _A.y, _A.k1, _A.k2), _A.thy),
}


def knee_components(mu):
    """Closed-form T_ij = u d_i d_j u and the exact wavevector, as numpy
    callables of (x, y) at fixed mu. Thin partials over the cached
    lambdified functions."""
    k1v = float(np.sqrt(1 - mu**2))
    f = {name: (lambda X, Y, _g=g: _g(X, Y, k1v, mu))
         for name, g in _F.items()}
    f["k1v"] = k1v
    return f


def exact_angle_error(f, X, Y, sx, sy, s, w):
    """Degrees, wrapped mod pi, of the exactly-averaged director against the
    exact wavevector at the SAME points."""
    Txx = exact_average(f["Txx"], X, Y, sx, sy, s, w)
    Txy = exact_average(f["Txy"], X, Y, sx, sy, s, w)
    Tyy = exact_average(f["Tyy"], X, Y, sx, sy, s, w)
    eg = eig2x2_numeric(Txx, Txy, Tyy)
    truth = np.arctan2(f["thy"](X, Y), f["thx"](X, Y))
    return np.degrees(np.abs(wrap_mod_pi(eg["angle_dom"] - truth,
                                         numeric=True))), eg


# =====================================================================

def part_convergence(report, n_gh):
    report("=" * 70)
    report("0. QUADRATURE CONVERGENCE (assumption check, not assumed)")
    report("=" * 70)
    f = knee_components(0.9)
    X, Y = np.meshgrid(np.linspace(0, 2 * np.pi / f["k1v"], 9),
                       np.linspace(0.3, 3.0, 7))
    ref = None
    report(f"  {'n_gh':>6} {'max |d<Txx>|':>16}")
    for n in (10, 20, 30, 40, 60):
        s, w = gh_nodes(n)
        v = exact_average(f["Txx"], X, Y, np.pi / 2, np.pi / 2, s, w)
        if ref is not None:
            report(f"  {n:>6} {np.max(np.abs(v - ref)):>16.3e}")
        ref = v
    report(f"  using n_gh = {n_gh}")
    report("")


def part_ledger(report, n_gh, sigmas):
    """The Part C ledger, with an exact column beside the modeled ones."""
    report("=" * 70)
    report("1. THE LEDGER WITH A TRUE COLUMN")
    report("=" * 70)
    s, w = gh_nodes(n_gh)
    a = KneeAnsatz()
    sxs, sys_ = sp.symbols("s_x s_y", positive=True)
    f_kern = lambdify_np((a.x, a.y, a.k1, a.k2, sxs, sys_),
                         a.angle_error_vs_truth("kernel", sxs, sys_))
    f_corr = lambdify_np((a.x, a.y, a.k1, a.k2, sxs, sys_),
                         a.angle_error_vs_truth("corrugated", sxs, sys_))

    report("  median |angle error| in degrees, flank patch 0.2 < y < 4,")
    report("  one period in x")
    report(f"  {'mu':>5} {'sigma':>7} {'kernel(2nd)':>12} {'corrugated':>11} "
           f"{'EXACT':>9}")
    out = {}
    for mu in (0.5, 0.9):
        f = knee_components(mu)
        X, Y = np.meshgrid(np.linspace(0, 2 * np.pi / f["k1v"], 40),
                           np.linspace(0.2, 4.0, 30))
        for sg in sigmas:
            e_ex, _ = exact_angle_error(f, X, Y, sg, sg, s, w)
            e_k = np.degrees(np.abs(wrap_mod_pi(
                f_kern(X, Y, f["k1v"], mu, sg, sg), numeric=True)))
            e_c = np.degrees(np.abs(wrap_mod_pi(
                f_corr(X, Y, f["k1v"], mu, sg, sg), numeric=True)))
            out[(mu, sg)] = (float(np.nanmedian(e_k)),
                             float(np.nanmedian(e_c)),
                             float(np.nanmedian(e_ex)))
            report(f"  {mu:>5.2f} {sg:>7.4f} {out[(mu, sg)][0]:>12.3f} "
                   f"{out[(mu, sg)][1]:>11.3f} {out[(mu, sg)][2]:>9.3f}")
    report("")
    report("  The exact column is the only one to trust. Where it departs")
    report("  from kernel(2nd), the truncation is the thing that is wrong.")
    report("")
    return out


def part_optimum(report, n_gh, outdir):
    """Where the real optimum in sigma sits, on the flank and on the core."""
    report("=" * 70)
    report("2. WHERE THE REAL OPTIMUM IN SIGMA SITS")
    report("=" * 70)
    s, w = gh_nodes(n_gh)
    ss = np.linspace(0.15, 3.0, 26)
    curves = {}
    report(f"  {'mu':>5} {'best sigma':>11} {'error there':>12} "
           f"{'error at pi/2':>14}")
    for mu in (0.5, 0.9):
        f = knee_components(mu)
        X, Y = np.meshgrid(np.linspace(0, 2 * np.pi / f["k1v"], 36),
                           np.linspace(0.2, 4.0, 28))
        med = []
        for sg in ss:
            e, _ = exact_angle_error(f, X, Y, sg, sg, s, w)
            med.append(float(np.nanmedian(e)))
        med = np.array(med)
        curves[mu] = med
        i = int(np.argmin(med))
        e_half = float(np.interp(np.pi / 2, ss, med))
        report(f"  {mu:>5.2f} {ss[i]:>11.3f} {med[i]:>12.4f} "
               f"{e_half:>14.4f}")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for mu, cl in ((0.5, "C0"), (0.9, "C3")):
        ax.plot(ss, curves[mu], cl, lw=1.7, marker="o", ms=3,
                label=rf"$\mu={mu}$ exact")
    ax.axvline(np.pi / 2, color="k", ls=":", lw=1.0, label=r"$\sigma=\pi/2$")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel("median |angle error| (deg)")
    ax.set_yscale("log")
    ax.set_title("exact Gaussian averaging, knee flank")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "exact_sigma_sweep.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    report(f"  figure: {p}")
    report("")
    return ss, curves


def part_mustar(report, n_gh):
    """On-core mu*(sigma) from exact convolution, on a fine mu grid. This is
    the measurement the Section 3 attribution question needs."""
    report("=" * 70)
    report("3. ON-CORE mu*(sigma) FROM EXACT CONVOLUTION")
    report("=" * 70)
    s, w = gh_nodes(n_gh)

    def flipped_fraction(mu, sg):
        f = knee_components(mu)
        xs = np.linspace(0, 2 * np.pi / f["k1v"], 48, endpoint=False)
        ys = np.zeros_like(xs)
        _, eg = exact_angle_error(f, xs, ys, sg, sg, s, w)
        ang = np.abs(wrap_mod_pi(eg["angle_dom"], numeric=True))
        return float(np.mean(ang > np.pi / 4))

    report("  flip criterion: the dominant eigenvector is closer to yhat")
    report("  than to xhat, at a MEDIAN point of the core row")
    report(f"  {'sigma':>8} {'mu* exact':>11} {'Sec 3 DC':>10} "
           f"{'Sec 3 full':>11} {'|exact-DC|':>11}")
    refs = [(np.pi / 8, 0.949, 0.950), (np.pi / 4, 0.888, 0.890),
            (np.pi / 2, 0.820, 0.810), (np.pi, 0.769, 0.770)]
    rows = []
    for sg, rd, rf in refs:
        g = lambda m: flipped_fraction(m, sg) - 0.5
        try:
            mstar = brentq(g, 0.30, 0.995, xtol=2e-4)
        except ValueError:
            mstar = np.nan
        rows.append((sg, mstar, rd, rf))
        d = abs(mstar - rd) if np.isfinite(mstar) else np.nan
        report(f"  {sg:>8.4f} {mstar:>11.4f} {rd:>10.3f} {rf:>11.3f} "
               f"{d:>11.4f}")
    report("")
    report("  This column carries no modeling at all. If it lands on the")
    report("  Sec 3 DC values, the DC mechanism is the whole story on core")
    report("  and the 0.810-vs-0.820 difference was grid resolution. If it")
    report("  lands on the Sec 3 FULL values instead, corrugation does")
    report("  contribute and the Part B argument needs revisiting.")
    report("")
    return rows


def part_frozen_phase(report, n_gh):
    """Does the frozen-phase corrugation model hold OFF core? Part C assumes
    it does. Compare the exact average against DC + g * osc."""
    report("=" * 70)
    report("4. IS THE FROZEN-PHASE MODEL VALID OFF CORE?")
    report("=" * 70)
    s, w = gh_nodes(n_gh)
    a = KneeAnsatz()
    sxs, sys_ = sp.symbols("s_x s_y", positive=True)
    mu = 0.9
    f = knee_components(mu)
    fc = {ij: lambdify_np((a.x, a.y, a.k1, a.k2, sxs, sys_),
                          a.uhu_tensor("corrugated", sxs, sys_)[i, j])
          for ij, (i, j) in (("xx", (0, 0)), ("xy", (0, 1)), ("yy", (1, 1)))}

    report(f"  mu = {mu}; relative discrepancy in the averaged tensor,")
    report("  ||exact - frozen phase||_F / ||exact||_F, median over x")
    report(f"  {'y':>6} {'sigma=0.39':>12} {'sigma=0.79':>12} "
           f"{'sigma=1.57':>12}")
    for yv in (0.0, 0.5, 1.0, 2.0, 4.0):
        xs = np.linspace(0, 2 * np.pi / f["k1v"], 24, endpoint=False)
        ys = np.full_like(xs, yv)
        cells = []
        for sg in (np.pi / 8, np.pi / 4, np.pi / 2):
            ex = np.stack([exact_average(f[c], xs, ys, sg, sg, s, w)
                           for c in ("Txx", "Txy", "Tyy")])
            fr = np.stack([fc[c](xs, ys, f["k1v"], mu, sg, sg)
                           for c in ("xx", "xy", "yy")])
            num = np.sqrt(((ex - fr)**2 * np.array([1, 2, 1])[:, None]).sum(0))
            den = np.sqrt((ex**2 * np.array([1, 2, 1])[:, None]).sum(0))
            cells.append(float(np.nanmedian(num / np.maximum(den, 1e-30))))
        report(f"  {yv:>6.1f} {cells[0]:>12.3e} {cells[1]:>12.3e} "
               f"{cells[2]:>12.3e}")
    report("")
    report("  On core the frozen-phase model is exact by construction")
    report("  (theta = k1 x + slow(y), so the x-average is exactly")
    report("  exp(-2 sx^2 k1^2)). Off core it is an approximation, and this")
    report("  table is how good it is.")
    report("")


# =====================================================================

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    lines = []

    def report(s=""):
        print(s, flush=True)
        lines.append(s)

    sigmas = (np.pi / 8, np.pi / 4, np.pi / 2, np.pi)
    part_convergence(report, args.n_gh)
    part_ledger(report, args.n_gh, sigmas)
    part_optimum(report, args.n_gh, args.outdir)
    part_mustar(report, args.n_gh)
    part_frozen_phase(report, args.n_gh)

    with open(os.path.join(args.outdir, "exact_convolution_check.txt"),
              "w") as fh:
        fh.write("\n".join(lines) + "\n")


class _Args:
    outdir = "out_exact_conv"
    n_gh = 40


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=_Args.outdir)
    p.add_argument("--n-gh", type=int, default=_Args.n_gh)
    main(p.parse_args())
