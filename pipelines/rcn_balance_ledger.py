# pipelines/rcn_balance_ledger.py
"""
Stationary balances and energy ledger for the SH zigzag data (Writeup 4,
task 4): no rates are fitted, so none of the collinearity of the rate
fits enters; each quantity is a single ratio or a single integral per
frame and per domain.

Per frame and per domain (row |y - y_row| < row_to, band < band_to, shell
band_to..far_from, far > far_from, all; core discs of core_excl spacings
removed; distances in spacings L = pi/k1 = pi/cos alpha):

 (a) self-dual balance  div k = -beta w   (Writeup 1):
     weak least squares of Int[div k phi] on Int[w phi] over the domain's
     query centres -> beta_fit and its R^2, plus the binned pointwise
     ratio -div k / w against d_row (|w| > w_min only).
 (b) RCN stationary balance  Lap div k = lambda (w div k + k.grad w):
     weak least squares -> lambda and R^2. RCN with alpha = eta gives
     lambda = 2; on a self-dual knee with div k = -beta w, beta =
     sqrt(alpha/eta), lambda = 2 alpha/eta = 2 beta^2. The bending
     coefficient of the RATE fits is measured against this number: on a
     state at balance the stretching and bending groups are collinear
     and only their departure from lambda is a rate. Also the balance
     defect  ||2S - B|| / (||2S|| + ||B||)  at alpha = eta (0 at balance).
 (c) energies per unit x-length:
     E_RCN = Int (eta/2)(div k)^2 + (alpha/2) w^2  and with the quartic
     (alpha/2)(w^2 - w^4/(2R)), alpha = eta = 2R/3, R from the file;
     E_SH  = Int (e - e_roll) with e the solver's SH energy density and
     e_roll its far-field row average (excess over rolls, the
     energy_vs_angle_zigzag convention). Both vs t, by domain, and the
     number of frames in which each increases (monotonicity).
 (d) stationarity: RMS theta_t by domain (centred differences of z) so a
     "stationary" tail can be certified before its balances are quoted.

Validation (--mode knee): the analytic self-dual knee
theta = k1 x + (1/beta) log cosh(beta mu y), u = cos theta, through the
estimator (route estimator) and with the analytic k (route exact):
beta_fit must read beta and lambda must read 2 beta^2 on both routes,
which calibrates the crease reading of the ledger.
(--mode pn): the Pismen row (pn_field.pismen_on_sh_grid) with a row of
dislocations at the zigzag spacing, both routes; its balances are
reported as a reference, not a truth.

Outputs (relative): results/rcn_balance_ledger/<stem>/<stamp>/
  report.txt, ledger.csv (one row per frame x domain), energies_vs_t.png,
  balances_vs_t.png, profiles_last_frame.png

Changelog
---------
v1.5  (1) SIGN FIX. v1.4 chose the reference component by mean |k_j| and
      applied np.sign(ref) POINTWISE. On every zigzag file with mu > 0.71
      the dominant component is k2 = +-mu, so the lower half-plane was
      flipped, turning the estimator's gradient branch (k1 = cos alpha > 0
      on both sides, verified on the 0.85 tail: k1 = +0.54 above and
      below, 50 percent of pixels flipped by v1.4) into a field with a
      jump in k1 along the row; the knee validation's estimator route
      read beta 0.009 under v1.4 (0.82 of record under v1.0). v1.5 picks
      the component with the larger |mean| (k1 for zigzag and knee,
      k2 for the Pismen rolls) and applies ONE scalar sign per frame.
      pn-mode numbers of record (k2 > 0 everywhere) are unaffected.
      (2) Relation B with a J column: Lap div k = lambda_J S + m J (OLS),
      beside the one-column lambda, per domain and per core-distance bin.
      Motivation: for any gradient field, with chi_b = div k + b w the
      residual of relation A,
        Lap div k - 2 b^2 S = Lap chi_b - 2 b div(chi_b k) + 4 b J    (*)
      (SymPy, chi_identity_check.py), so an exactly self-dual J-free
      field satisfies B with lambda = 2 b^2 > 0 whatever the sign of b,
      and the RCN stationary equation is the statement that the right
      side of (*) vanishes with b = sqrt(alpha/eta). The J column is the
      part of (*) the papers keep (defects_are_weak 2.12, universal 3.16).
      (3) Identity decomposition per domain: for b in {beta_fit, 1,
      --beta_scan}, the projection on S of each term of (*) (their sum is
      lambda - 2 b^2 up to the weak-form discretisation of the nonlinear
      terms, reported as the identity check), and rms chi_b / rms div k.
      Reading: the term that carries lambda - 2 b^2 is where B's departure
      from A's implication lives. On the 0.85 tail (band, b = 2.7): -21
      from -2b div(chi k), +3 from 4bJ, -0.2 from Lap chi; chi_1 = -1.7 w
      is the crease, not a core potential.
      (4) Profiles by core distance (0.25 L bins to 4 L, last frame, both
      routes): medians of w, div k, -div k / w, A, J, curl k / |grad k|,
      written to profiles.csv and profiles_by_core_distance.png, for the
      SH-tail-versus-Pismen comparison (T2.4c).
      (5) beta_scan.csv: chi ratio and decomposition per b per domain.
      (6) --pn_amp 2: smooth synthetic core dip A = A_eq tanh(r/r_c)
      (--pn_core, x units) for the estimator control. --pn_amp 1 clips
      A to zero in a disc about each core (the exact PN wavenumber
      diverges there), the estimator's z vanishes in the disc and its k
      explodes (weak w column 2e3 against 8e-2 exact), so every kernel
      within 1.5 L of a core is contaminated: the T2.4a estimator
      numbers (beta -0.78, lambda -3.4 at 0.5-1 L; single dislocation
      lambda -9.4) test the disc, not the estimator. With the smooth dip
      (r_c 1.5 and 3.0) the estimator reads beta -1.38 / -1.28 (R2 0.43 /
      0.30) at 0.5-1 L against the exact -1.49 (0.91), nothing beyond
      1 L, and NO coherent relation B (R2 0.00-0.25, either sign).
      (7) T2.4b, the estimator's floor (knee mode, beta 1, crease depth
      w_max = mu^2 = 0.005 .. 0.9; estimator band, exact reads 1.000 /
      2.00 throughout): beta_fit 0.90 (R2 -1.5), 0.99 (0.94), 0.99
      (0.99), 0.97, 0.95, 0.90, 0.87, 0.82, 0.79 (R2 >= 0.999) at w_max
      0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.56, 0.72, 0.9; lambda -0.50
      (0.80), -0.50 (0.26), +0.01 (-0.05), +1.45 (0.63), +1.57 (0.80),
      +1.54 (0.94), +1.38 (0.98), +1.17 (0.99), +1.08 (0.97). Relation A
      is readable to w_max 0.02; relation B needs w_max >= 0.1 and reads
      a wrong-signed lambda of -0.5 below 0.05. Consequence for the SH
      row: its lambda = -2 (R2 0.94-0.97, robust to discs to 1 L) rests
      on band columns carried by the near-core signal (|w| 0.05-0.12),
      where the knee floor reads the right sign; every negative lambda
      produced by the estimator on an exact-RCN field has R2 <= 0.26
      except the w_max 0.005 knee (-0.5, R2 0.80), whose signal level
      the SH band columns exceed by an order of magnitude.
v1.4  the global sign of the estimator's k is fixed by its dominant
      component (k2 for rolls along y, as in pn mode; k1 for the zigzag),
      where sign(k1) alone randomised k on the Pismen reference.
      Least-squares beta and lambda binned by distance to the nearest
      core (0.5-1, 1-2, 2-4, 4-8, >8 L), pointwise fields, both routes:
      the readout that reproduces the Pismen half-plane numbers of
      record (single dislocation: --mode pn --pn_spacing 200 --n_xper 1
      gives one core in the box). config.json now excludes the
      internal _xd field.
v1.3  pn mode forces a half-plane (side upper unless given): the Pismen
      relation div k = +w is odd in Y and a two-sided fit cancels to
      zero by symmetry (v1.2 reported beta 0.00 for that reason). On the
      upper half-plane the exact single dislocation reads slope
      div k / w = 1.06-1.19 (R2 0.97-0.99) at 4-16 core radii, 1.75 at
      2-4 (the paraxial psi_YY term), 0.98 beyond 16; the row of 8 at
      the zigzag spacing keeps it only within one spacing of the cores
      (1.43, R2 0.89) and loses it between them (0.8, 0.6). In the
      ledger's convention div k = -beta w this reads beta = -1.
v1.2  pn mode: dislocation spacing defaults to the zigzag core spacing
      pi/k1 (--pn_spacing), --pn_amp 1 attaches the SH amplitude
      A1^2 = (4/3)(R - w^2) to the Pismen phase so the reference field is
      the one an amplitude-slaved phase theory describes; --side
      upper/lower restricts every domain to a half-plane (the stationary
      self-dual test on a PN half-plane).
v1.1  TOTAL energies on the fixed crop (rim removed, no core discs, no
      baseline) for the Lyapunov test: the per-domain excess over the
      far roll density was not monotone on the T25 files because e_roll
      itself relaxes (the far amplitude goes from 1 to 0.816 in the
      first time unit) and the core discs move; the SH total is the
      functional that must decrease. find_cores merge (v1.5 of
      mg_zigzag_theta_t) inherited.
v1.0  first version. Validated on the knee (beta 1.0 and 1.3) and
      smoke-tested on the 0.85 tail excerpt (frame 0 dropped).

Run:
  python rcn_balance_ledger.py --mode knee --beta 1.3
  python rcn_balance_ledger.py --npz <tail file> --drop_first 1
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT / "src", _ROOT / "src" / "op_extract", _ROOT / "src" / "utils", _HERE):
    sys.path.insert(0, str(p))

from mg_zigzag_theta_t import load_npz, estimate, find_cores, dist_to_cores   # noqa: E402
from wsindy_theta_t import kernel_d3, corr1d, inner_deriv, theta_t_from_z     # noqa: E402
from op_extract import t1_common as t1c                                        # noqa: E402
try:
    from op_extract.pn_field import pismen_on_sh_grid                          # noqa: E402
except ImportError:                                                            # pragma: no cover
    from pn_field import pismen_on_sh_grid                                     # noqa: E402

import matplotlib                                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                # noqa: E402

DOMAINS = ["row", "band", "shell", "far", "all"]
BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, np.inf)]


def corr2(F, kx, ky, dx, dy):
    return corr1d(corr1d(F, ky, 0, dy), kx, 1, dx)


def r2(y, yh):
    return 1.0 - np.sum((y - yh) ** 2) / (np.sum((y - np.mean(y)) ** 2) + 1e-300)


def domain_masks(Y, y_row, Lsp, args):
    d = np.abs(Y - y_row) / Lsp
    side = np.ones_like(d, bool)
    if args.side == "upper":
        side = Y > y_row
    elif args.side == "lower":
        side = Y < y_row
    return dict(row=(d < args.row_to) & side, band=(d < args.band_to) & side,
                shell=(d >= args.band_to) & (d <= args.far_from) & side,
                far=(d > args.far_from) & side, all=side), d


def frame_balances(k1, k2, x, y, Lsp, masks, d_row, excl, args, d_core=None, A_field=None):
    """weak columns on one frame, least-squares balances per domain,
    pointwise binned ratio and energy densities"""
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    a = args.radius * Lsp
    fx, fx1, fx2, fx3, mx = kernel_d3(a, dx, args.poly_p)
    fy, fy1, fy2, fy3, my = kernel_d3(a, dy, args.poly_p)
    eta = args.eta * Lsp
    D = lambda F, ax: inner_deriv(F, eta, dx, dy, ax, "poly", args.poly_p)
    w = k1 ** 2 + k2 ** 2 - 1.0
    k1x, k1y, k2x, k2y = D(k1, "x"), D(k1, "y"), D(k2, "x"), D(k2, "y")
    divk = k1x + k2y
    J = k1x * k2y - k1y * k2x                              # det grad k = Gaussian curvature of the phase surface
    curlk = k2x - k1y
    wx, wy = D(w, "x"), D(w, "y")
    S = w * divk + k1 * wx + k2 * wy                       # stretching sum, pointwise
    K = lambda F, kxx, kyy: corr2(F, kxx, kyy, dx, dy)
    Wdiv = -(K(k1, fx1, fy) + K(k2, fx, fy1))              # Int div k phi
    Ww = K(w, fx, fy)                                      # Int w phi
    WB = -(K(k1, fx3, fy) + K(k1, fx1, fy2) + K(k2, fx2, fy1) + K(k2, fx, fy3))  # Int Lap div k phi
    WS = K(S, fx, fy)                                      # Int S phi
    WJ = K(J, fx, fy)                                      # Int J phi
    WLw = K(w, fx2, fy) + K(w, fx, fy2)                    # Int Lap w phi  (fully weak)
    ny, nx = w.shape
    ix = np.arange(mx + args.rim, nx - mx - args.rim, args.stride)
    iy = np.arange(my + args.rim, ny - my - args.rim, args.stride)
    IY, IX = np.meshgrid(iy, ix, indexing="ij")
    ok = ~excl[IY, IX]
    # v1.5: the div(chi_b k) column for each b of the decomposition needs the
    # pointwise chi_b; build the weak column lazily per b (cheap: two corr2)
    _divchik_cache = {}

    def Wdivchik(b):
        if b not in _divchik_cache:
            chi = divk + b * w
            _divchik_cache[b] = -(K(chi * k1, fx1, fy) + K(chi * k2, fx, fy1))   # Int div(chi k) phi
        return _divchik_cache[b]

    def decomposition(sel, b, Bv, Sv, dv, wv):
        """projection on S of each term of (*), lambda - 2b^2, identity check, chi ratio"""
        C1 = Bv + b * WLw[IY, IX][sel]                     # Int Lap chi_b phi
        C2 = -2.0 * b * Wdivchik(b)[IY, IX][sel]           # -2b Int div(chi_b k) phi
        C3 = 4.0 * b * WJ[IY, IX][sel]                     # 4b Int J phi
        SS = max(np.dot(Sv, Sv), 1e-300)
        c1, c2, c3 = (float(np.dot(Sv, C) / SS) for C in (C1, C2, C3))
        lhs = Bv - 2 * b * b * Sv
        chk = float(np.sqrt(np.mean((lhs - C1 - C2 - C3) ** 2)) / max(np.sqrt(np.mean(Bv ** 2)), 1e-300))   # relative to rms B
        chi_ratio = float(np.sqrt(np.mean((dv + b * wv) ** 2)) / max(np.sqrt(np.mean(dv ** 2)), 1e-300))
        return dict(b=b, c_lapchi=c1, c_divchik=c2, c_J=c3, lam_minus_2b2=float(np.dot(Sv, Bv) / SS - 2 * b * b),
                    identity_check=chk, chi_ratio=chi_ratio)

    out = {}
    scans = {}
    for name, m in masks.items():
        sel = ok & m[IY, IX]
        n = int(sel.sum())
        if n < 30:
            out[name] = dict(n=n, beta=np.nan, beta_r2=np.nan, lam=np.nan, lam_r2=np.nan,
                             defect=np.nan, lam_J=np.nan, m_J=np.nan, lamJ_r2=np.nan,
                             chi_ratio_bfit=np.nan, chi_ratio_b1=np.nan,
                             c_lapchi_bfit=np.nan, c_divchik_bfit=np.nan, c_J_bfit=np.nan, idcheck_bfit=np.nan,
                             c_lapchi_b1=np.nan, c_divchik_b1=np.nan, c_J_b1=np.nan, idcheck_b1=np.nan)
            scans[name] = []
            continue
        dv, wv, Bv, Sv, Jv = (F[IY, IX][sel] for F in (Wdiv, Ww, WB, WS, WJ))
        # div k = -beta w
        beta = -float(np.dot(wv, dv) / max(np.dot(wv, wv), 1e-300))
        # Lap div k = lambda S
        lam = float(np.dot(Sv, Bv) / max(np.dot(Sv, Sv), 1e-300))
        defect = float(np.sqrt(np.mean((2 * Sv - Bv) ** 2)) /
                       max(np.sqrt(np.mean((2 * Sv) ** 2)) + np.sqrt(np.mean(Bv ** 2)), 1e-300))
        # v1.5: Lap div k = lambda_J S + m J
        M = np.vstack([Sv, Jv]).T
        cJ = np.linalg.lstsq(M, Bv, rcond=None)[0]
        lamJ_r2 = float(r2(Bv, M @ cJ))
        # v1.5: decomposition at beta_fit, at the theory's 1, and over the scan
        dec_fit = decomposition(sel, float(np.round(beta, 3)), Bv, Sv, dv, wv)
        dec_1 = decomposition(sel, 1.0, Bv, Sv, dv, wv)
        scans[name] = [decomposition(sel, float(b), Bv, Sv, dv, wv) for b in args.beta_scan_list]
        out[name] = dict(n=n, beta=beta, beta_r2=float(r2(dv, -beta * wv)),
                         lam=lam, lam_r2=float(r2(Bv, lam * Sv)), defect=defect,
                         lam_J=float(cJ[0]), m_J=float(cJ[1]), lamJ_r2=lamJ_r2,
                         chi_ratio_bfit=dec_fit["chi_ratio"], chi_ratio_b1=dec_1["chi_ratio"],
                         c_lapchi_bfit=dec_fit["c_lapchi"], c_divchik_bfit=dec_fit["c_divchik"],
                         c_J_bfit=dec_fit["c_J"], idcheck_bfit=dec_fit["identity_check"],
                         c_lapchi_b1=dec_1["c_lapchi"], c_divchik_b1=dec_1["c_divchik"],
                         c_J_b1=dec_1["c_J"], idcheck_b1=dec_1["identity_check"])
    # pointwise binned ratio -div k / w against the row distance
    ratio = {}
    good = (np.abs(w) > args.w_min) & ~excl
    for lo, hi in BINS:
        sel = good & (d_row >= lo) & (d_row < hi)
        ratio["%.1f-%s" % (lo, ("%.1f" % hi) if np.isfinite(hi) else "inf")] = \
            float(np.median(-divk[sel] / w[sel])) if sel.sum() > 30 else np.nan
    # v1.4: least-squares slopes by distance to the nearest core (in
    # spacings), both relations, pointwise fields, rim excluded: the
    # readout that reproduces the Pismen half-plane numbers of record
    bycore = {}
    profiles = []
    if d_core is not None:
        lapdivk = D(D(divk, "x"), "x") + D(D(divk, "y"), "y")
        for lo, hi in ((0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, np.inf)):
            sel = (d_core >= lo) & (d_core < hi) & ~excl & (np.abs(w) > 1e-6) & masks["all"]   # masks carry --side
            key = "%.1f-%s" % (lo, ("%.1f" % hi) if np.isfinite(hi) else "inf")
            if sel.sum() < 100:
                bycore[key] = dict(n=int(sel.sum()), beta=np.nan, beta_r2=np.nan, lam=np.nan, lam_r2=np.nan,
                                   lam_J=np.nan, m_J=np.nan, lamJ_r2=np.nan)
                continue
            dv, wv = divk[sel], w[sel]
            b = -float(np.dot(wv, dv) / max(np.dot(wv, wv), 1e-300))
            Bv, Sv, Jv = lapdivk[sel], S[sel], J[sel]
            l = float(np.dot(Sv, Bv) / max(np.dot(Sv, Sv), 1e-300))
            M = np.vstack([Sv, Jv]).T                          # v1.5: Lap div k = lambda_J S + m J, pointwise
            cJ = np.linalg.lstsq(M, Bv, rcond=None)[0]
            bycore[key] = dict(n=int(sel.sum()), beta=b, beta_r2=float(r2(dv, -b * wv)),
                               lam=l, lam_r2=float(r2(Bv, l * Sv)),
                               lam_J=float(cJ[0]), m_J=float(cJ[1]), lamJ_r2=float(r2(Bv, M @ cJ)))
        # v1.5: profiles by core distance, 0.25 L bins to 4 L (medians of pointwise fields)
        gk = np.sqrt(k1x ** 2 + k1y ** 2 + k2x ** 2 + k2y ** 2)
        for lo in np.arange(0.0, 4.0, 0.25):
            sel = (d_core >= lo) & (d_core < lo + 0.25) & ~excl & masks["all"]
            if sel.sum() < 30:
                continue
            ws = w[sel]
            good = np.abs(ws) > args.w_min
            profiles.append(dict(d_lo=float(lo), n=int(sel.sum()), w=float(np.median(ws)),
                                 divk=float(np.median(divk[sel])),
                                 neg_divk_over_w=(float(np.median(-divk[sel][good] / ws[good])) if good.sum() > 30 else np.nan),
                                 J=float(np.median(J[sel])), absJ=float(np.median(np.abs(J[sel]))),
                                 curl_over_gradk=float(np.median(np.abs(curlk[sel]) / np.maximum(gk[sel], 1e-12))),
                                 A=(float(np.median(A_field[sel])) if A_field is not None else np.nan)))
    return out, ratio, w, divk, bycore, scans, profiles


def energies(w, divk, e_sh, masks, excl, dx, dy, Lx, alpha, eta, R, far_mask, rim=None):
    """per-domain energies (core discs and rim removed, excess over the
    frame's far-field roll density) and, under key "total", the integrals
    over the FIXED domain (rim removed only, no core discs, no baseline):
    the SH total is the Lyapunov functional of the gradient flow on the
    crop and must decrease; the per-domain excess need not, because the
    roll baseline e_roll(t) itself relaxes and the core discs move."""
    dens_rcn = 0.5 * eta * divk ** 2 + 0.5 * alpha * w ** 2
    dens_q = 0.5 * eta * divk ** 2 + 0.5 * alpha * (w ** 2 - w ** 4 / (2 * R))
    out = {}
    e_roll = float(np.mean(e_sh[far_mask])) if (e_sh is not None and far_mask.any()) else np.nan
    fixed = ~rim if rim is not None else np.ones_like(w, bool)
    out["total"] = dict(E_rcn=float(np.sum(dens_rcn[fixed]) * dx * dy / Lx),
                        E_rcn_quartic=float(np.sum(dens_q[fixed]) * dx * dy / Lx),
                        E_sh=(float(np.sum(e_sh[fixed]) * dx * dy / Lx) if e_sh is not None else np.nan),
                        e_roll=e_roll)
    for name, m in masks.items():
        sel = m & ~excl
        out[name] = dict(E_rcn=float(np.sum(dens_rcn[sel]) * dx * dy / Lx),
                         E_rcn_quartic=float(np.sum(dens_q[sel]) * dx * dy / Lx),
                         E_sh=(float(np.sum((e_sh[sel] - e_roll)) * dx * dy / Lx)
                               if e_sh is not None else np.nan))
    return out, e_roll


def run_fields(u, x, y, tt, e_sh, mu, R, y_row, args, out, lines, label, k_exact=None, A_override=None):
    """u: [ny, nx, nt]; e_sh: [ny, nx, nt] or None; k_exact: (k1, k2) stacks
    for an exact route. Returns the ledger rows."""
    k1c = float(np.sqrt(1 - mu ** 2)); Lsp = np.pi / k1c
    alpha = eta_b = 2.0 * R / 3.0
    ny, nx, nt = u.shape
    dx = float(x[1] - x[0]); dy = float(y[1] - y[0]); Lx = nx * dx; Ly = ny * dy
    X, Y = np.meshgrid(x, y)
    masks, d_row = domain_masks(Y, y_row, Lsp, args)
    rim = ((np.abs(X - x.mean()) > (0.5 - args.rim_frac) * Lx) |
           (np.abs(Y - y.mean()) > (0.5 - args.rim_frac) * Ly))   # taper zone and any x seam
    win = (np.abs(X - x.mean()) < 0.4 * Lx) & (np.abs(Y - y.mean()) < 0.4 * Ly)
    taper = t1c.edge_taper(x, y, frac=args.taper_frac)
    routes = {}
    ests, n_prev = [], None
    for i in range(nt):
        e = estimate(u[..., i], Lx, Ly, args.sigma_dir, n_prev, taper)
        n_prev = e["n"]; ests.append(e)
    k1m = np.stack([e["k1"] for e in ests], -1); k2m = np.stack([e["k2"] for e in ests], -1)
    Am = np.stack([e["A"] for e in ests], -1) if A_override is None else A_override
    # sign of k: continuous in time by the estimator, fixed globally here by
    # the dominant component (k1 for the zigzag far field k1 = cos alpha > 0;
    # k2 for a field of rolls k = (0, 1) such as the Pismen reference)
    # v1.5: ONE scalar sign per frame from the component with the larger
    # |mean| (k1 = cos alpha > 0 for the zigzag and the knee, k2 for rolls);
    # v1.4's pointwise sign(k2) flipped the lower half-plane of every zigzag
    # file with mu > 0.71 (see changelog)
    use_k1 = np.abs(np.mean(k1m)) >= np.abs(np.mean(k2m))
    s = np.sign(np.mean(k1m if use_k1 else k2m, axis=(0, 1)))[None, None, :]
    s[s == 0] = 1
    k1m, k2m = k1m * s, k2m * s
    lines.append("  sign of k fixed per frame by the mean of %s (v1.5, scalar per frame)" % ("k1" if use_k1 else "k2"))
    th_t = None
    if nt > 1:
        zs = np.stack([e["z"] for e in ests], -1)
        th_t = theta_t_from_z(zs, float(np.median(np.diff(tt)))) * s
        del zs
    del ests
    if k_exact is not None:
        routes["exact"] = (k_exact[0], k_exact[1], Am, None)   # cores from the estimator's amplitude
    routes["estimator"] = (k1m, k2m, Am, th_t)
    lines.append("%s: mu %.3f (L = %.2f), R %.2f, alpha = eta = %.3f, %d frames t %.3f..%.3f, crop %dx%d = +-%.1f L;"
                 " kernel radius %.2g L; domains row<%.2g band<%.2g shell..far>%.2g; core discs %.2g L; rim %.0f%% excluded" %
                 (label, mu, Lsp, R, alpha, nt, tt[0], tt[-1], nx, ny, 0.5 * Ly / Lsp, args.radius,
                  args.row_to, args.band_to, args.far_from, args.core_excl, 100 * args.rim_frac))
    rows = []
    profile_rows, scan_rows = [], []
    for route, (K1, K2, AA, TT) in routes.items():
        lines.append("  [route %s]" % route)
        inc = {d: dict(E_rcn=0, E_sh=0) for d in DOMAINS}
        prev = None
        totals = []
        for i in range(nt):
            excl = rim.copy()
            ncore = 0
            d_core = None
            if AA is not None:
                xc, yc = find_cores(AA[..., i], x, y, win, args.core_frac, merge_dist=0.5 * Lsp)
                ncore = len(xc)
                if ncore:
                    d_core = dist_to_cores(X, Y, xc, yc) / Lsp
                    if args.core_excl > 0:
                        excl |= d_core < args.core_excl
            if args.mode == "pn" and getattr(args, "_xd", None) is not None:
                d_core = np.min(np.hypot(X[..., None] - args._xd[None, None, :], Y[..., None]), axis=-1) / Lsp
            bal, ratio, w, divk, bycore, scans, profiles = frame_balances(
                K1[..., i], K2[..., i], x, y, Lsp, masks, d_row, excl, args, d_core,
                A_field=(AA[..., i] if AA is not None else None))
            if i == nt - 1:
                for pr in profiles:
                    profile_rows.append(dict(route=route, t=float(tt[i]), **pr))
                for d in DOMAINS:
                    for sc in scans.get(d, []):
                        scan_rows.append(dict(route=route, t=float(tt[i]), domain=d, **sc))
            e_i = e_sh[..., i] if e_sh is not None else None
            en, e_roll = energies(w, divk, e_i, masks, excl, dx, dy, Lx, alpha, eta_b, R, masks["far"] & ~excl, rim=rim)
            tot = en.pop("total")
            totals.append(dict(t=float(tt[i]), **tot))
            for d in DOMAINS:
                rms_tt = (float(np.sqrt(np.nanmean(TT[..., i][masks[d] & ~excl] ** 2)))
                          if TT is not None else np.nan)
                row = dict(route=route, t=float(tt[i]), frame=i, domain=d, n_cores=ncore,
                           rms_theta_t=rms_tt, **bal[d], **en[d])
                rows.append(row)
                if prev is not None:
                    inc[d]["E_rcn"] += int(en[d]["E_rcn"] > prev[d]["E_rcn"] * (1 + 1e-6))
                    inc[d]["E_sh"] += int(np.isfinite(en[d]["E_sh"]) and en[d]["E_sh"] > prev[d]["E_sh"] + 1e-9)
            prev = en
            if i in (0, nt - 1) or (nt > 2 and i == nt // 2):
                lines.append("  t = %.3f  cores %d  e_roll %.5f" % (tt[i], ncore, e_roll if np.isfinite(e_roll) else np.nan))
                for d in DOMAINS:
                    b = bal[d]; E = en[d]
                    lines.append("    %-5s n %6d  beta_fit %7.3f (R2 %6.3f)  lambda %7.3f (R2 %6.3f)  balance defect %.3f"
                                 "  E_rcn %.4f  E_rcn_q %.4f  E_sh_excess %.4f  rms theta_t %s" %
                                 (d, b["n"], b["beta"], b["beta_r2"], b["lam"], b["lam_r2"], b["defect"],
                                  E["E_rcn"], E["E_rcn_quartic"], E["E_sh"],
                                  ("%.2e" % rows[-len(DOMAINS) + DOMAINS.index(d)]["rms_theta_t"])
                                  if TT is not None else "n/a"))
                for d in ("band", "row"):
                    b_ = bal[d]
                    if np.isfinite(b_.get("lam_J", np.nan)):
                        lines.append("    %-5s v1.5  B = lambda_J S + m J: lambda_J %6.2f, m %6.2f (R2 %.3f vs %.3f without J);"
                                     "  chi ratio rms(div k + b w)/rms(div k): b=beta_fit %.2f, b=1 %.2f" %
                                     (d, b_["lam_J"], b_["m_J"], b_["lamJ_r2"], b_["lam_r2"], b_["chi_ratio_bfit"], b_["chi_ratio_b1"]))
                        lines.append("    %-5s       identity (*) at b = beta_fit %.2f: lambda - 2b^2 = %6.2f = %6.2f (Lap chi) + %6.2f (-2b div chi k) + %6.2f (4bJ)  [check %.2f]" %
                                     (d, b_["beta"], b_["lam"] - 2 * b_["beta"] ** 2, b_["c_lapchi_bfit"], b_["c_divchik_bfit"], b_["c_J_bfit"], b_["idcheck_bfit"]))
                        lines.append("    %-5s       identity (*) at b = 1:            lambda - 2   = %6.2f = %6.2f (Lap chi) + %6.2f (-2b div chi k) + %6.2f (4bJ)  [check %.2f]" %
                                     (d, b_["lam"] - 2, b_["c_lapchi_b1"], b_["c_divchik_b1"], b_["c_J_b1"], b_["idcheck_b1"]))
                lines.append("    pointwise median -div k / w by row distance (L): " +
                             ", ".join("%s: %s" % (k, ("%.3f" % v) if np.isfinite(v) else "nan") for k, v in ratio.items()))
                if bycore:
                    lines.append("    by distance to the nearest core (L), least squares: " + "; ".join(
                        "%s: beta %.2f (R2 %.2f) lambda %.2f (R2 %.2f) n %d" %
                        (k, v["beta"], v["beta_r2"], v["lam"], v["lam_r2"], v["n"]) for k, v in bycore.items()))
                    lines.append("    by core distance, v1.5 B with J: " + "; ".join(
                        "%s: lambda_J %.2f m %.2f (R2 %.2f)" % (k, v["lam_J"], v["m_J"], v["lamJ_r2"]) for k, v in bycore.items()))
        if nt > 1:
            lines.append("  frames in which the per-domain EXCESS energy increased (of %d steps; not a Lyapunov"
                         " test, see totals): " % (nt - 1) +
                         "; ".join("%s: E_rcn %d, E_sh %d" % (d, inc[d]["E_rcn"], inc[d]["E_sh"]) for d in DOMAINS))
            Esh = np.array([q["E_sh"] for q in totals]); Ercn = np.array([q["E_rcn"] for q in totals])
            er = np.array([q["e_roll"] for q in totals])
            lines.append("  TOTALS on the fixed crop (rim removed, no core discs): SH energy per unit length"
                         " %.5f -> %.5f (%d of %d steps increased; Lyapunov: expect 0); RCN energy of the"
                         " estimator's k %.5f -> %.5f (%d increased); far roll density e_roll %.5f -> %.5f" %
                         (Esh[0], Esh[-1], int(np.sum(np.diff(Esh) > 1e-6 * np.abs(Esh[0]))), nt - 1,
                          Ercn[0], Ercn[-1], int(np.sum(np.diff(Ercn) > 1e-6 * np.abs(Ercn[0]))), er[0], er[-1]))
        for q in totals:
            rows.append(dict(route=route, t=q["t"], frame=-1, domain="TOTAL_fixed", n_cores=-1,
                             rms_theta_t=np.nan, n=-1, beta=np.nan, beta_r2=np.nan, lam=np.nan, lam_r2=np.nan,
                             defect=np.nan, E_rcn=q["E_rcn"], E_rcn_quartic=q["E_rcn_quartic"], E_sh=q["E_sh"]))
    _write_csv(out / "profiles.csv", profile_rows)
    _write_csv(out / "beta_scan.csv", scan_rows)
    profile_figure(profile_rows, out)
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(rows)


def profile_figure(prow, out):
    """v1.5: medians by core distance, last frame, both routes"""
    if not prow:
        return
    routes = sorted({r["route"] for r in prow})
    fields = [("w", "w"), ("divk", "div k"), ("neg_divk_over_w", "-div k / w"), ("A", "A"), ("J", "J"), ("curl_over_gradk", "|curl k| / |grad k|")]
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (key, lab) in zip(axs.ravel(), fields):
        for route in routes:
            rr = [r for r in prow if r["route"] == route]
            ax.plot([r["d_lo"] + 0.125 for r in rr], [r[key] for r in rr], marker="o", ms=3, label=route)
        ax.set_title(lab, fontsize=9); ax.set_xlabel("distance to nearest core (L)"); ax.axhline(0, color="k", lw=0.5)
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "profiles_by_core_distance.png", dpi=130); plt.close(fig)


def figures(rows, out, title):
    routes = sorted({r["route"] for r in rows})
    for route in routes:
        R = [r for r in rows if r["route"] == route]
        t = sorted({r["t"] for r in R})
        if len(t) < 2:
            continue
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
        for d in ("row", "band", "shell", "far"):
            rr = [r for r in R if r["domain"] == d]
            tt_ = [r["t"] for r in rr]
            axs[0].plot(tt_, [r["E_rcn"] for r in rr], label=d)
            axs[1].plot(tt_, [r["E_sh"] for r in rr], label=d)
            axs[2].plot(tt_, [r["beta"] for r in rr], label="beta " + d)
        axs[0].set_title("E_RCN per unit length, by domain", fontsize=9)
        axs[1].set_title("SH energy excess over rolls, by domain", fontsize=9)
        axs[2].set_title("self-dual beta_fit, by domain", fontsize=9)
        for ax in axs:
            ax.set_xlabel("t"); ax.legend(fontsize=6)
        fig.suptitle("%s, route %s" % (title, route), fontsize=9)
        fig.tight_layout(); fig.savefig(out / ("energies_vs_t_%s.png" % route), dpi=130); plt.close(fig)
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
        for d in ("row", "band", "shell"):
            rr = [r for r in R if r["domain"] == d]
            axs[0].plot([r["t"] for r in rr], [r["lam"] for r in rr], label=d)
            axs[1].plot([r["t"] for r in rr], [r["defect"] for r in rr], label=d)
        axs[0].axhline(2.0, color="k", lw=0.6, ls="--"); axs[0].set_title("lambda = Lap div k / S (RCN: 2)", fontsize=9)
        axs[1].set_title("balance defect ||2S - B||/(||2S||+||B||)", fontsize=9)
        for ax in axs:
            ax.set_xlabel("t"); ax.legend(fontsize=6)
        fig.suptitle("%s, route %s" % (title, route), fontsize=9)
        fig.tight_layout(); fig.savefig(out / ("balances_vs_t_%s.png" % route), dpi=130); plt.close(fig)


def main(args):
    out = Path(args.outdir)
    lines = ["rcn_balance_ledger v1.5", ""]
    if not hasattr(args, "beta_scan_list"):
        args.beta_scan_list = [float(v) for v in str(args.beta_scan).split(",") if v.strip()]
    t0 = time.time()
    if args.mode in ("knee", "pn"):
        mu = args.mu; k1 = float(np.sqrt(1 - mu ** 2)); Lsp = np.pi / k1
        dx = args.Ly / (args.ny_control - 1)
        y = np.linspace(-args.Ly / 2, args.Ly / 2, args.ny_control)
        nx = int(round(args.n_xper * 2 * np.pi / k1 / dx)); Lx = nx * dx
        x = (np.arange(nx) - nx // 2) * dx
        X, Y = np.meshgrid(x, y)
        if args.mode == "knee":
            b = args.beta
            th = k1 * X + np.log(np.cosh(b * mu * Y)) / b
            k1e = np.full_like(th, k1); k2e = mu * np.tanh(b * mu * Y)
            label = "analytic self-dual knee, beta %.2f (expect beta_fit %.2f, lambda %.2f)" % (b, b, 2 * b * b)
            stem = "knee_beta%.2f" % b
        else:
            # Pismen row: dislocations pn_spacing apart (default the zigzag core spacing pi/k1)
            sp = args.pn_spacing if args.pn_spacing > 0 else np.pi / k1
            nd = max(1, int(round(Lx / sp)))
            xd = (np.arange(nd) - nd / 2 + 0.5) * sp
            th, _ = pismen_on_sh_grid(X, Y, xd)
            th = th * 1.0
            args._xd = xd
            # k = grad theta through the cut-free form cos th grad sin th - sin th grad cos th
            c, sn = np.cos(th), np.sin(th)
            k1e = c * np.gradient(sn, dx, axis=1) - sn * np.gradient(c, dx, axis=1)
            k2e = c * np.gradient(sn, dx, axis=0) - sn * np.gradient(c, dx, axis=0)
            rr = np.min(np.hypot(X[..., None] - xd[None, None, :], Y[..., None]), axis=-1)
            if args.pn_amp == 2:
                # v1.5: smooth synthetic core dip A = A_eq tanh(r / r_c), r_c = --pn_core (units of
                # x), A_eq = sqrt(2R/3)... the SH equilibrium sqrt(4R/3)/... at R = 0.5: 0.816.
                # Motivation: --pn_amp 1 clips A to ZERO in a disc about every core (the exact
                # wavenumber diverges there), the estimator's z vanishes in the disc and its k
                # explodes (weak w column 2e3 against 8e-2 exact, 18 Sep); every kernel within
                # 1.5 L of a core is contaminated and the T2.4a estimator numbers test the disc,
                # not the estimator. An SH core is a smooth dip (A 0.77 at 0.6 L on the 0.85 tail).
                A_pn = np.sqrt(4.0 * args.R / 3.0) * np.tanh(rr / max(args.pn_core, 1e-6))
                amp_note = "u = A_eq tanh(r/%.2f) cos theta (smooth synthetic core dip, A_eq %.3f)" % (args.pn_core, np.sqrt(4 * args.R / 3))
            elif args.pn_amp:
                # SH amplitude slaved to the exact wavenumber: A1^2 = (4/3)(R - w^2), clipped at 0,
                # and u = A1 cos theta (the field an amplitude-slaved phase theory describes)
                w_e = k1e ** 2 + k2e ** 2 - 1.0
                A_pn = np.sqrt(np.clip(4.0 / 3.0 * (args.R - w_e ** 2), 0.0, None))
                amp_note = "u = A1(w) cos theta, A1^2 = (4/3)(R - w^2)"
            else:
                A_pn = np.tanh(rr / 1.0)      # synthetic dip so the core discs are excluded
                amp_note = "u = cos theta"
            label = ("Pismen row, %d dislocations %.2f apart, k = (0, 1) rolls, %s, cores excluded (reference)"
                     % (len(xd), sp, amp_note))
            stem = "pn_row_amp%d" % int(args.pn_amp)
            mu = 1.0 - 1e-9   # rolls along y in that construction; row distance uses Lsp = pi/k1 -> use period
            Lsp = 2 * np.pi
        if args.mode == "pn" and args.side == "both":
            # the Pismen field's self-dual relation is ODD in Y (div k = +w above the
            # row, -w below, theta = Y - sign(Y) log psi) while w is even, so a
            # two-sided fit cancels to zero by symmetry; fit one half-plane
            lines.append("  FLAG: pn mode with side both would cancel by symmetry; using side upper")
            args.side = "upper"
        u = (A_pn * np.cos(th) if (args.mode == "pn" and args.pn_amp) else np.cos(th))[..., None]
        R = args.R
        stamp = "rad%.2g_%s" % (args.radius, time.strftime("%Y%m%d_%H%M%S"))
        out = out / stem / stamp; out.mkdir(parents=True, exist_ok=True)
        rows = run_fields(u, x, y, np.array([0.0]), None, mu if args.mode == "knee" else args.mu, R, 0.0,
                          args, out, lines, label, k_exact=(k1e[..., None], k2e[..., None]),
                          A_override=(A_pn[..., None] if args.mode == "pn" else None))
    else:
        files = [f.strip() for f in args.npz.split(",") if f.strip()]
        rows = []
        stem = Path(files[0]).stem if len(files) == 1 else "multi_%d" % len(files)
        stamp = "rad%.2g_ce%.2g_t%.2f-%.2f" % (args.radius, args.core_excl, args.t0, args.t1)
        out = out / stem / stamp; out.mkdir(parents=True, exist_ok=True)
        for f in files:
            D = load_npz(f)
            u, x, y, tt, md = D["u"], D["x"], D["y"], D["t"], D["md"]
            raw = np.load(f, allow_pickle=True)
            e_sh = raw["e"] if "e" in raw.files else None
            if e_sh is not None and e_sh.shape[-1] != u.shape[-1]:
                lines.append("  NOTE %s: energy stack has %d frames, u has %d; aligning on the last frames" %
                             (Path(f).stem, e_sh.shape[-1], u.shape[-1]))
                e_sh = e_sh[..., -u.shape[-1]:]
            sel = (tt >= args.t0) & (tt <= args.t1)
            sel[:args.drop_first] = False
            u, tt = u[..., sel], tt[sel]
            if e_sh is not None:
                e_sh = e_sh[..., sel]
            mu = float(md.get("mu", args.mu)); R = float(md.get("Rscale", args.R))
            jc0 = int(md.get("gb_row_in_crop", u.shape[0] // 2))
            rows += run_fields(u, x, y, tt, e_sh, mu, R, float(y[jc0]), args, out, lines, Path(f).stem)
            lines.append("")
    figures(rows, out, stem)
    with open(out / "ledger.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    with open(out / "config.json", "w") as fh:
        json.dump({k: v for k, v in vars(args).items() if not k.startswith("_")}, fh, indent=2, default=str)
    lines.append("elapsed %.1f s" % (time.time() - t0))
    rep = "\n".join(lines)
    (out / "report.txt").write_text(rep)
    print(rep); print("outputs ->", out)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    mode = "data"                # data | knee | pn
    npz = ""
    outdir = str(Path("results") / "rcn_balance_ledger")
    t0 = -1.0
    t1 = 1e9
    drop_first = 0               # 1 for the T100 tails (t = 0 frame before the 95-unit gap)
    mu = 0.85
    R = 0.5
    beta = 1.0                   # knee mode: planted self-dual coefficient
    Ly = 60.0
    ny_control = 300
    n_xper = 4
    sigma_dir = 0.4
    taper_frac = 0.06
    core_frac = 0.70
    core_excl = 0.5
    radius = 1.5                 # spatial test-function half width, spacings
    poly_p = 6
    eta = 0.0
    stride = 4
    rim = 4
    w_min = 0.05                 # pointwise ratio only where |w| exceeds this
    row_to = 0.5
    band_to = 2.0
    far_from = 4.0
    rim_frac = 0.10              # fraction of the box edge (taper zone, x seam) excluded everywhere
    side = "both"                # both | upper | lower  (half-plane above / below the row)
    pn_amp = 0                   # pn mode: 1 = attach the SH amplitude A1(w) (clipped: zero discs); 2 = smooth tanh core dip (v1.5, the fair estimator control)
    pn_core = 1.5                # pn mode, pn_amp 2: core dip radius r_c in x units (0.25 L at mu 0.85 ~ 1.5)
    pn_spacing = 0.0             # pn mode: dislocation spacing (0 = the zigzag core spacing pi/k1)
    beta_scan = "0.5,1,1.5,2,2.5,3,3.5"   # v1.5: b values for the identity decomposition scan (beta_scan.csv)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in dir(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    cli = p.parse_args()
    cli._xd = None
    cli.beta_scan_list = [float(v) for v in str(cli.beta_scan).split(",") if v.strip()]
    if cli.npz:
        cli.mode = "data"
    sys.exit(main(cli))
