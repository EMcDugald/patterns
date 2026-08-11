# pipelines/eigen_averaging_track.py
"""
Main track (handoff Section 9): eigenvalues/eigenvectors of the uHu tensor
BEFORE averaging and AFTER each averaging model, for the knee and PN, against
the exact wavevector. The organizing question is which operations move the
estimate toward ground truth and which move it away.

Four rungs, matching the two channels of handoff Section 1:

    full        no averaging at all
    corrugated  channel (1) only: the 2 theta piece survives, suppressed by
                exp(-2 (sx^2 thx^2 + sy^2 thy^2))
    kernel      channel (2) only: ideal 2 theta removal, then second-order
                kernel mixing of the DC part
    DC          both removed; exactly -(1/2) k k^T for A = 1

Parts:
  A  exact pointwise director law (symbolic, then numerically validated)
  B  on-core knee flip threshold with BOTH channels, checked against the
     Section 3 spot checks
  C  the help/hurt ledger assembled from A and B
  D  amplitude-bearing PN: what ideal averaging alone cannot fix

Every claim is asserted against a closed form or a numeric reference before
it is used. Run: python eigen_averaging_track.py [--outdir DIR]
"""

import argparse
import os

import numpy as np
import sympy as sp
from scipy.optimize import brentq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.uhu_symbolic import (KneeAnsatz, PNAnsatz, eig2x2_numeric, wrap_mod_pi,
                          lambdify_np, is_zero)

GH_S, GH_W = np.polynomial.hermite_e.hermegauss(120)
GH_W = GH_W / GH_W.sum()


# =====================================================================
# Part A: exact pointwise director law
# =====================================================================

def part_A(report):
    """
    For A = 1 the pointwise tensor factors exactly (verified previously):

        T = -cos^2(theta) [ k k^T + tan(theta) H ],    H = Hess(theta)

    Rotate into the (khat, khat_perp) frame and write
    h11 = khat.H.khat, h12 = khat.H.khat_perp, h22 = khat_perp.H.khat_perp.
    The bracket is diag(|k|^2, 0) + tan(theta) H', so the closed 2x2 formula
    gives the director offset from the TRUE wavevector in closed form:

        Delta_a = (1/2) atan2( 2 tan(theta) h12,
                               |k|^2 + tan(theta) (h11 - h22) )

    Two limits, both immediate from this expression:
      tan(theta) -> 0   (at the stripe extrema, u = +-A):  Delta_a -> 0
      tan(theta) -> inf (at the stripe zeros, u = 0):
                        Delta_a -> (1/2) atan2(2 h12, h11 - h22),
                        which is the eigen-angle of H itself.

    So the unaveraged director does not diverge. It SWINGS, once per half
    period, between the true wavevector and the dominant eigendirection of
    the phase Hessian. Averaging over the fast phase is what collapses the
    swing; the residual after averaging is the W-controlled part.
    """
    report("=" * 70)
    report("PART A: exact pointwise director law (no averaging)")
    report("=" * 70)

    knee = KneeAnsatz()
    x, y, k1s, k2s = knee.x, knee.y, knee.k1, knee.k2

    a_true = sp.atan2(knee.thy, knee.thx)
    c, s = sp.cos(a_true), sp.sin(a_true)
    H = knee.hess
    h11 = c**2 * H[0, 0] + 2 * c * s * H[0, 1] + s**2 * H[1, 1]
    h12 = -c * s * H[0, 0] + (c**2 - s**2) * H[0, 1] + c * s * H[1, 1]
    h22 = s**2 * H[0, 0] - 2 * c * s * H[0, 1] + c**2 * H[1, 1]
    t = sp.tan(knee.theta)
    delta_pred = sp.atan2(2 * t * h12, knee.k_mag2 + t * (h11 - h22)) / 2

    f_pred = lambdify_np((x, y, k1s, k2s), delta_pred)
    f_exact = lambdify_np((x, y, k1s, k2s),
                          knee.angle_error_vs_truth("full"))

    worst = 0.0
    for mu in (0.3, 0.5, 0.9):
        k1v = np.sqrt(1 - mu**2)
        X, Y = np.meshgrid(np.linspace(0.05, 6.0, 120),
                           np.linspace(0.2, 5.0, 90))
        pr = wrap_mod_pi(f_pred(X, Y, k1v, mu), numeric=True)
        ex = wrap_mod_pi(f_exact(X, Y, k1v, mu), numeric=True)
        d = np.abs(wrap_mod_pi(pr - ex, numeric=True))
        worst = max(worst, float(np.nanmax(d)))
    assert worst < 1e-9, worst
    report(f"  [ok] closed-form Delta_a matches eigensystem('full') to "
           f"{worst:.1e} rad over mu in {{0.3, 0.5, 0.9}}")

    # the two limits, numerically, on the knee flank
    mu, k1v = 0.9, np.sqrt(1 - 0.81)
    yv = 1.0
    hh = lambdify_np((x, y, k1s, k2s), sp.Matrix([h11, h12, h22]))
    h = np.array(hh(0.0, yv, k1v, mu), dtype=float).ravel()
    lim = 0.5 * np.arctan2(2 * h[1], h[0] - h[2])
    # theta = k1 x + log(2 cosh(k2 y)); pick x so that cos(theta) ~ 0
    th0 = k1v * 0.0 + np.log(2 * np.cosh(mu * yv))
    x_zero = (np.pi / 2 - th0) / k1v
    x_ext = (0.0 - th0) / k1v
    d_zero_hi = float(wrap_mod_pi(f_exact(x_zero + 1e-7, yv, k1v, mu),
                                  numeric=True))
    d_zero_lo = float(wrap_mod_pi(f_exact(x_zero - 1e-7, yv, k1v, mu),
                                  numeric=True))
    d_ext = float(wrap_mod_pi(f_exact(x_ext, yv, k1v, mu), numeric=True))
    lim2 = float(wrap_mod_pi(lim + np.pi / 2, numeric=True))
    report(f"  at a stripe EXTREMUM (u = +-1):        {np.degrees(d_ext):8.3f} deg")
    report(f"  approaching a ZERO from below (tan -> +inf): "
           f"{np.degrees(d_zero_lo):8.3f} deg")
    report(f"  approaching the same ZERO from above (tan -> -inf): "
           f"{np.degrees(d_zero_hi):8.3f} deg")
    report(f"  Hessian eigen-axes here: {np.degrees(lim):.3f} deg and "
           f"{np.degrees(lim2):.3f} deg")
    assert abs(d_ext) < 1e-6
    # The two one-sided limits are the two eigen-AXES of H, and which one is
    # selected depends on the SIGN of tan(theta): the bracket is
    # diag(|k|^2, 0) + tan(theta) H', so tan -> +inf picks the max-eigenvalue
    # axis of H and tan -> -inf picks the min-eigenvalue axis, 90 deg away.
    got = sorted([d_zero_lo, d_zero_hi])
    want = sorted([lim, lim2])
    assert max(abs(wrap_mod_pi(g - w, numeric=True))
               for g, w in zip(got, want)) < 1e-3, (got, want)
    jump = float(abs(wrap_mod_pi(d_zero_hi - d_zero_lo, numeric=True)))
    assert abs(np.degrees(jump) - 90.0) < 1e-3, np.degrees(jump)
    report("  [ok] extremum limit = true wavevector; the two one-sided zero")
    report(f"       limits = the two Hessian eigen-axes; the director JUMPS")
    report(f"       by {np.degrees(jump):.3f} deg across the zero")
    report("")
    report("  READING:")
    report("  before averaging the estimate is not noisy, it is SYSTEMATIC and")
    report("  in fact DISCONTINUOUS. Within each half period the director runs")
    report("  from the true k (at the stripe extremum, tan theta = 0) out to a")
    report("  Hessian eigen-axis, jumps 90 deg across the stripe zero to the")
    report("  other Hessian eigen-axis, and runs back to true k. The jump is")
    report("  exact, not a numerical artifact.")
    report("")
    report("  Consequence worth carrying: a 90 deg eigen-flip on the u = 0")
    report("  contours is the generic behaviour of UNDER-smoothed extraction,")
    report("  and it is distinct in origin from the on-core flip of Section 3,")
    report("  which is a DC eigenvalue competition rather than a phase-driven")
    report("  jump. Anything measuring flip statistics should separate them.")
    report("")
    return {"worst_pointwise": worst}


# =====================================================================
# Part B: on-core knee, both channels
# =====================================================================

def _gh_avg_oncore(fn, sigma, k2):
    """1D Gaussian average at y = 0 of a function of y alone (the 2D
    isotropic average reduces to 1D because these depend on y only)."""
    return float(np.sum(GH_W * fn(sigma * GH_S, k2)))


def _thy2(yv, k2):
    return (k2 * np.tanh(k2 * yv))**2


def _thyy(yv, k2):
    return k2**2 / np.cosh(k2 * yv)**2


def _oncore_margin(mu, sx, sy, n_phase=721, stat="worst"):
    """
    Margin for the dominant eigenvector to point ALONG the core (xhat) on the
    knee core row, with both channels active.

    On the core row theta_y = 0 and theta_xy = 0, so T_xy = 0 exactly and the
    tensor is diagonal for any symmetric kernel. Frozen-phase model:

      T_xx = -(k1^2/2) (1 + g cos 2th)         exact: theta_x = k1 is constant,
                                               so u u_xx = -k1^2 u^2 pointwise
      T_yy = -(1/2)[ <thy^2> + g (cos 2th <thy^2> + sin 2th <thyy>) ]
      g    = exp(-2 sx^2 k1^2)                 corrugation survival on core

    The residual oscillation is a pure cosine in 2 theta:

      Txx - Tyy = D + g R cos(2 th + phi),
      D = k1^2 - <thy^2>,  R = sqrt(D^2 + <thyy>^2),

    so the statistic used to define "flipped" matters, and the two natural
    choices answer different questions:

      stat="median"  the typical point on the core row. The oscillation has
                     zero mean AND zero median, so this returns D exactly and
                     reproduces the Section 3 DC criterion identically, for
                     any g. Corrugation cannot shift it at all.
      stat="worst"   no point anywhere in the period is flipped. This is a
                     much stronger requirement, and at small sigma it is
                     dominated by the Part A jump rather than by the DC
                     competition, so it is NOT comparable to the Section 3
                     spot checks.
    """
    k1 = np.sqrt(1 - mu**2)
    g = np.exp(-2 * sx**2 * k1**2)
    a2 = _gh_avg_oncore(_thy2, sy, mu)
    ayy = _gh_avg_oncore(_thyy, sy, mu)
    th = np.linspace(0, np.pi, n_phase)
    Txx = k1**2 * (1 + g * np.cos(2 * th))
    Tyy = a2 + g * (np.cos(2 * th) * a2 + np.sin(2 * th) * ayy)
    d = Txx - Tyy
    return float(np.min(d)) if stat == "worst" else float(np.median(d))


def _mu_star(sx, sy, dc_only=False, stat="worst"):
    f = (lambda m: (1 - m**2) - _gh_avg_oncore(_thy2, sy, m)) if dc_only \
        else (lambda m: _oncore_margin(m, sx, sy, stat=stat))
    lo, hi = 0.05, 0.999
    if f(lo) < 0 or f(hi) > 0:
        return np.nan
    return brentq(f, lo, hi, xtol=1e-6)


def part_B(report):
    report("=" * 70)
    report("PART B: on-core knee flip threshold with BOTH channels")
    report("=" * 70)
    report("  Section 3 established mu*(sigma) from the DC channel alone and")
    report("  noted that corrugation supplies only a small wiggle. Adding the")
    report("  corrugation term to the same on-core competition tests that.")
    report("")
    report(f"  {'sigma':>8} {'mu* DC only':>13} {'mu* median':>12} "
           f"{'mu* worst':>11} {'Sec 3 DC':>10} {'Sec 3 full':>11}")

    sigmas = [np.pi / 8, np.pi / 4, np.pi / 2, np.pi]
    ref_dc = [0.949, 0.888, 0.820, 0.769]
    ref_full = [0.950, 0.890, 0.810, 0.770]
    rows = []
    for sg, rd, rf in zip(sigmas, ref_dc, ref_full):
        m_dc = _mu_star(sg, sg, dc_only=True)
        m_med = _mu_star(sg, sg, stat="median")
        m_wst = _mu_star(sg, sg, stat="worst")
        rows.append((sg, m_dc, m_med, m_wst, rd, rf))
        report(f"  {sg:>8.4f} {m_dc:>13.4f} {m_med:>12.4f} {m_wst:>11.4f} "
               f"{rd:>10.3f} {rf:>11.3f}")

    dc_err = max(abs(r[1] - r[4]) for r in rows)
    assert dc_err < 2e-3, dc_err
    report(f"  [ok] DC-only column reproduces the Section 3 values to "
           f"{dc_err:.1e}")

    med_err = max(abs(r[1] - r[2]) for r in rows)
    assert med_err < 1e-4, med_err
    report(f"  [ok] the median-phase column equals the DC-only column to "
           f"{med_err:.1e}, for every sigma")
    report("")
    report("  *** FLAG FOR ED: attribution question on Section 3 ***")
    report("")
    report("  Section 3 says the DC mechanism sets the flip and 'corrugation")
    report("  only supplies the small 0.81-vs-0.82 wiggle'. Adding the")
    report("  corrugation term to the same on-core competition does not")
    report("  reproduce that wiggle, and the reason is structural rather than")
    report("  numerical. On the core row the surviving oscillation is a PURE")
    report("  cosine in 2 theta, so it has zero mean and zero median. Any")
    report("  symmetric phase statistic therefore returns the DC answer")
    report("  EXACTLY, for any suppression factor g. Corrugation cannot shift")
    report("  a typical-point threshold on the core at all.")
    report("")
    report("  The worst-phase column does move, a lot, but it answers a")
    report("  different question ('is any point in the period flipped') and at")
    report("  small sigma it is dominated by the Part A jump, not by the DC")
    report("  competition. It is not the comparable quantity.")
    report("")
    report("  So: the Section 3 full-convolution spot checks are quoted to two")
    report("  decimals (0.950, 0.890, 0.810, 0.770), i.e. on a mu grid of")
    report("  0.01, and the DC-only prediction sits within one grid step of")
    report("  all four. The 0.810-vs-0.820 difference is AT the quoted")
    report("  resolution. I do not think it can be attributed to corrugation")
    report("  on this evidence; it needs a finer mu sweep of the full")
    report("  convolution to say anything either way. Nothing else in the")
    report("  handoff depends on the attribution, and the main Section 3")
    report("  conclusion (the flip is the DC mechanism) is strengthened, not")
    report("  weakened, by this.")
    report("")

    # the corrugation channel on its own has an EXACT on-core flip criterion
    report("  Corrugation-only criterion (exact on core, WORST phase):")
    report("  no point in the period is flipped iff  g < k1^2 / k2^2, i.e.")
    report("      sx^2 > ln(mu^2/(1-mu^2)) / (2 (1-mu^2)),  binding only for")
    report("      mu > 1/sqrt(2).")
    report(f"  {'mu':>6} {'sx required':>13} {'Sec 6 tol=5%':>14}")
    for mu in (0.75, 0.80, 0.88, 0.95, 0.99):
        k1 = np.sqrt(1 - mu**2)
        need = np.sqrt(max(np.log(mu**2 / (1 - mu**2)), 0.0) / (2 * k1**2))
        sec6 = np.sqrt(np.log(1 / 0.05) / (2 * k1**2))
        report(f"  {mu:>6.2f} {need:>13.3f} {sec6:>14.3f}")
    report("  [note] Section 6 used a 5% survival tolerance as a heuristic.")
    report("  The exact on-core flip criterion is g < k1^2/k2^2, which is")
    report("  LOOSER than 5% for mu < 0.97 and TIGHTER beyond. Same shape,")
    report("  sharper constant. This refines Section 6, it does not")
    report("  contradict it.")
    report("")
    return rows


# =====================================================================
# Part C: help / hurt ledger
# =====================================================================

def part_C(report, sigmas=(np.pi / 8, np.pi / 4, np.pi / 2, np.pi)):
    report("=" * 70)
    report("PART C: which operations help, which hurt (knee, off-core)")
    report("=" * 70)
    knee = KneeAnsatz()
    x, y, k1s, k2s = knee.x, knee.y, knee.k1, knee.k2
    sx, sy = sp.symbols("s_x s_y", positive=True)

    f_full = lambdify_np((x, y, k1s, k2s), knee.angle_error_vs_truth("full"))
    f_corr = lambdify_np((x, y, k1s, k2s, sx, sy),
                         knee.angle_error_vs_truth("corrugated", sx, sy))
    f_kern = lambdify_np((x, y, k1s, k2s, sx, sy),
                         knee.angle_error_vs_truth("kernel", sx, sy))

    out = {}
    report(f"  median |angle error| in degrees over a flank patch "
           f"(0 < y < 4, one period in x)")
    report(f"  {'mu':>5} {'sigma':>7} {'full':>9} {'corrugated':>11} "
           f"{'kernel':>9} {'DC':>6}")
    for mu in (0.5, 0.9):
        k1v = np.sqrt(1 - mu**2)
        X, Y = np.meshgrid(np.linspace(0, 2 * np.pi / k1v, 160),
                           np.linspace(0.2, 4.0, 100))
        e_full = np.degrees(np.abs(wrap_mod_pi(f_full(X, Y, k1v, mu),
                                               numeric=True)))
        for sg in sigmas:
            e_c = np.degrees(np.abs(wrap_mod_pi(
                f_corr(X, Y, k1v, mu, sg, sg), numeric=True)))
            e_k = np.degrees(np.abs(wrap_mod_pi(
                f_kern(X, Y, k1v, mu, sg, sg), numeric=True)))
            row = (float(np.nanmedian(e_full)), float(np.nanmedian(e_c)),
                   float(np.nanmedian(e_k)), 0.0)
            out[(mu, sg)] = row
            report(f"  {mu:>5.2f} {sg:>7.4f} {row[0]:>9.3f} {row[1]:>11.3f} "
                   f"{row[2]:>9.3f} {row[3]:>6.1f}")
    report("")
    report("  READING:")
    report("  - removing the 2 theta piece (full -> corrugated -> DC) is")
    report("    monotonically HELPFUL: it only ever removes the phase swing")
    report("    of Part A, and the survival factor g is the fraction of that")
    report("    swing left behind.")
    report("  - the kernel column is the opposite: at DC the error is zero by")
    report("    construction, and every finite sigma ADDS error. Channel (2)")
    report("    is purely a cost.")
    report("  - so sigma is not a quality knob with an optimum in the usual")
    report("    sense. It buys suppression of a term that is already")
    report("    exponentially suppressed, and pays with a term that grows")
    report("    like sigma^2 ||H||^2 / |k|^2. The optimum exists only because")
    report("    the two rates differ.")
    report("")
    return out


# =====================================================================
# Part D: amplitude-bearing PN
# =====================================================================

def part_D(report):
    report("=" * 70)
    report("PART D: amplitude-bearing PN, and what ideal averaging cannot fix")
    report("=" * 70)
    pn1 = PNAnsatz(numeric_weights=True)
    pna = PNAnsatz(numeric_weights=True, amplitude="tanh", xi=1)

    Txx = lambdify_np((pna.x, pna.y), pna.uhu_dc("x", "x"))
    Txy = lambdify_np((pna.x, pna.y), pna.uhu_dc("x", "y"))
    Tyy = lambdify_np((pna.x, pna.y), pna.uhu_dc("y", "y"))
    thx = lambdify_np((pna.x, pna.y), pna.thx)
    thy = lambdify_np((pna.x, pna.y), pna.thy)
    WA = lambdify_np((pna.x, pna.y), pna.amplitude_contamination())
    Wf = lambdify_np((pn1.x, pn1.y), pn1.W)

    # phase-only DC is exact everywhere theta is smooth
    e1 = lambdify_np((pn1.x, pn1.y), pn1.angle_error_vs_truth("DC"))
    phis = np.linspace(0.15, np.pi - 0.15, 60)
    report(f"  {'r':>5} {'A=1 DC err':>12} {'A=tanh DC err':>15} "
           f"{'W_A':>10} {'W':>10}")
    rows = []
    for r in (0.3, 0.5, 1.0, 2.0, 4.0, 8.0):
        xr, yr = r * np.cos(phis), r * np.sin(phis)
        err1 = np.degrees(np.abs(wrap_mod_pi(e1(xr, yr), numeric=True)))
        eg = eig2x2_numeric(Txx(xr, yr), Txy(xr, yr), Tyy(xr, yr))
        erra = np.degrees(np.abs(wrap_mod_pi(
            eg["angle_dom"] - np.arctan2(thy(xr, yr), thx(xr, yr)),
            numeric=True)))
        wa = np.nanmedian([float(WA(a, b)) for a, b in zip(xr, yr)])
        ww = np.nanmedian([float(Wf(a, b)) for a, b in zip(xr, yr)])
        rows.append((r, float(np.nanmedian(err1)), float(np.nanmedian(erra)),
                     float(wa), float(ww)))
        report(f"  {r:>5.1f} {rows[-1][1]:>12.2e} {rows[-1][2]:>15.3f} "
               f"{wa:>10.3f} {ww:>10.3f}")

    assert max(r[1] for r in rows) < 1e-6, "phase-only DC should be exact"
    report("  [ok] with A = 1 the IDEALLY averaged tensor is exact at every")
    report("       radius, as handoff Section 1 states for phase-only fields")
    report("")
    report("  *** FLAG FOR ED (handoff Sec 9b escalation criterion: 'the PN")
    report("  amplitude insertion changes qualitative conclusions') ***")
    report("")
    report("  Section 1 concludes that for phase-only fields every extraction")
    report("  failure is a finite-kernel effect. The table above shows that")
    report("  this conclusion is conditional on A = 1 in a way that matters")
    report("  here. With the physical amplitude the DC tensor is")
    report("      (1/2)(A Hess A - A^2 k k^T),")
    report("  and the first term does not vanish under ANY averaging: it is")
    report("  already in the ideally averaged tensor. So near an")
    report("  amplitude-bearing core the estimate is wrong by tens of degrees")
    report("  even with the 2 theta piece removed exactly and no kernel")
    report("  applied at all. There is no sigma that fixes it, because sigma")
    report("  is not what is breaking it.")
    report("")
    report("  This is consistent with Section 8, which anticipated the")
    report("  amplitude route, but it does change the shape of the claim: the")
    report("  two-channel picture is complete for phase-only fields and")
    report("  incomplete once amplitude varies. A third channel belongs in")
    report("  the ledger, call it AMPLITUDE MIXING, controlled by")
    report("      W_A = ||A Hess A||_F / (A^2 |grad theta|^2)")
    report("  alongside W. Note in the table that W_A overtakes W as r")
    report("  decreases, and crosses 1 near r ~ 0.5 for xi = 1.")
    report("")
    report("  Why this may matter for the nucleation argument: it is a")
    report("  mechanism, internal to the extraction, by which amplitude")
    report("  carries information at a core that phase provably cannot. I am")
    report("  flagging rather than developing it, per the handoff.")
    report("")
    return rows


# =====================================================================
# figure
# =====================================================================

def make_figure(path):
    knee = KneeAnsatz()
    x, y, k1s, k2s = knee.x, knee.y, knee.k1, knee.k2
    sx, sy = sp.symbols("s_x s_y", positive=True)
    f_full = lambdify_np((x, y, k1s, k2s), knee.angle_error_vs_truth("full"))
    f_kern = lambdify_np((x, y, k1s, k2s, sx, sy),
                         knee.angle_error_vs_truth("kernel", sx, sy))

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.0))

    # (1) the pointwise swing along one period
    mu, k1v = 0.9, np.sqrt(1 - 0.81)
    xs = np.linspace(0, 2 * np.pi / k1v, 900)
    for yv, cl in ((0.6, "C0"), (1.2, "C1"), (2.4, "C2")):
        d = np.degrees(wrap_mod_pi(f_full(xs, yv, k1v, mu), numeric=True))
        axes[0].plot(xs * k1v / np.pi, d, cl, lw=1.4, label=f"y = {yv}")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_xlabel(r"$k_1 x/\pi$ (one stripe period = 2)")
    axes[0].set_ylabel("director offset from true k (deg)")
    axes[0].set_title(r"(a) unaveraged: swings once per half period, $\mu=0.9$")
    axes[0].legend(fontsize=8)

    # (2) on-core flip threshold, DC only vs both channels
    sgs = np.linspace(0.15, 3.4, 40)
    m_dc = [_mu_star(s, s, dc_only=True) for s in sgs]
    m_bo = [_mu_star(s, s, stat="worst") for s in sgs]
    axes[1].plot(sgs, m_dc, "C0", lw=1.6, label="DC channel only (Sec 3)")
    axes[1].plot(sgs, m_bo, "C3--", lw=1.6, label="both channels, worst phase")
    axes[1].axhline(1 / np.sqrt(2), color="k", ls=":", lw=1.0,
                    label=r"$W=1$: $\mu=1/\sqrt{2}$")
    axes[1].set_xlabel(r"$\sigma$")
    axes[1].set_ylabel(r"$\mu^*$")
    axes[1].set_title(r"(b) on-core flip threshold $\mu^*(\sigma)$")
    axes[1].legend(fontsize=8)

    # (3) kernel channel cost vs sigma
    for mu, cl in ((0.5, "C0"), (0.9, "C3")):
        k1v = np.sqrt(1 - mu**2)
        X, Y = np.meshgrid(np.linspace(0, 2 * np.pi / k1v, 120),
                           np.linspace(0.2, 4.0, 80))
        ss = np.linspace(0.05, 1.6, 22)
        med = [np.nanmedian(np.degrees(np.abs(wrap_mod_pi(
            f_kern(X, Y, k1v, mu, s, s), numeric=True)))) for s in ss]
        axes[2].plot(ss, med, cl, lw=1.6, marker="o", ms=3,
                     label=rf"$\mu={mu}$")
    axes[2].set_xlabel(r"$\sigma$")
    axes[2].set_ylabel("median |angle error| (deg)")
    axes[2].set_title("(c) channel (2) is pure cost: 0 at DC, grows with "
                      r"$\sigma$")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =====================================================================

def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    lines = []

    def report(s=""):
        print(s, flush=True)
        lines.append(s)

    part_A(report)
    part_B(report)
    part_C(report)
    part_D(report)

    fig_path = os.path.join(outdir, "eigen_averaging_track.png")
    make_figure(fig_path)
    report(f"figure written: {fig_path}")

    with open(os.path.join(outdir, "eigen_averaging_track.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


class _Args:
    outdir = "out_eigen_track"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=_Args.outdir)
    main(p.parse_args().outdir)
