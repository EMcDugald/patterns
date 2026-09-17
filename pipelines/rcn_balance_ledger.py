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


def frame_balances(k1, k2, x, y, Lsp, masks, d_row, excl, args):
    """weak columns on one frame, least-squares balances per domain,
    pointwise binned ratio and energy densities"""
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    a = args.radius * Lsp
    fx, fx1, fx2, fx3, mx = kernel_d3(a, dx, args.poly_p)
    fy, fy1, fy2, fy3, my = kernel_d3(a, dy, args.poly_p)
    eta = args.eta * Lsp
    D = lambda F, ax: inner_deriv(F, eta, dx, dy, ax, "poly", args.poly_p)
    w = k1 ** 2 + k2 ** 2 - 1.0
    divk = D(k1, "x") + D(k2, "y")
    wx, wy = D(w, "x"), D(w, "y")
    S = w * divk + k1 * wx + k2 * wy                       # stretching sum, pointwise
    K = lambda F, kxx, kyy: corr2(F, kxx, kyy, dx, dy)
    Wdiv = -(K(k1, fx1, fy) + K(k2, fx, fy1))              # Int div k phi
    Ww = K(w, fx, fy)                                      # Int w phi
    WB = -(K(k1, fx3, fy) + K(k1, fx1, fy2) + K(k2, fx2, fy1) + K(k2, fx, fy3))  # Int Lap div k phi
    WS = K(S, fx, fy)                                      # Int S phi
    ny, nx = w.shape
    ix = np.arange(mx + args.rim, nx - mx - args.rim, args.stride)
    iy = np.arange(my + args.rim, ny - my - args.rim, args.stride)
    IY, IX = np.meshgrid(iy, ix, indexing="ij")
    ok = ~excl[IY, IX]
    out = {}
    for name, m in masks.items():
        sel = ok & m[IY, IX]
        n = int(sel.sum())
        if n < 30:
            out[name] = dict(n=n, beta=np.nan, beta_r2=np.nan, lam=np.nan, lam_r2=np.nan,
                             defect=np.nan)
            continue
        dv, wv, Bv, Sv = Wdiv[IY, IX][sel], Ww[IY, IX][sel], WB[IY, IX][sel], WS[IY, IX][sel]
        # div k = -beta w
        beta = -float(np.dot(wv, dv) / max(np.dot(wv, wv), 1e-300))
        # Lap div k = lambda S
        lam = float(np.dot(Sv, Bv) / max(np.dot(Sv, Sv), 1e-300))
        defect = float(np.sqrt(np.mean((2 * Sv - Bv) ** 2)) /
                       max(np.sqrt(np.mean((2 * Sv) ** 2)) + np.sqrt(np.mean(Bv ** 2)), 1e-300))
        out[name] = dict(n=n, beta=beta, beta_r2=float(r2(dv, -beta * wv)),
                         lam=lam, lam_r2=float(r2(Bv, lam * Sv)), defect=defect)
    # pointwise binned ratio -div k / w against the row distance
    ratio = {}
    good = (np.abs(w) > args.w_min) & ~excl
    for lo, hi in BINS:
        sel = good & (d_row >= lo) & (d_row < hi)
        ratio["%.1f-%s" % (lo, ("%.1f" % hi) if np.isfinite(hi) else "inf")] = \
            float(np.median(-divk[sel] / w[sel])) if sel.sum() > 30 else np.nan
    return out, ratio, w, divk


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
    s = np.sign(k1m); s[s == 0] = 1
    k1m, k2m = k1m * s, k2m * s
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
    for route, (K1, K2, AA, TT) in routes.items():
        lines.append("  [route %s]" % route)
        inc = {d: dict(E_rcn=0, E_sh=0) for d in DOMAINS}
        prev = None
        totals = []
        for i in range(nt):
            excl = rim.copy()
            ncore = 0
            if AA is not None and args.core_excl > 0:
                xc, yc = find_cores(AA[..., i], x, y, win, args.core_frac, merge_dist=0.5 * Lsp)
                ncore = len(xc)
                if ncore:
                    excl |= dist_to_cores(X, Y, xc, yc) / Lsp < args.core_excl
            bal, ratio, w, divk = frame_balances(K1[..., i], K2[..., i], x, y, Lsp, masks, d_row, excl, args)
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
                lines.append("    pointwise median -div k / w by row distance (L): " +
                             ", ".join("%s: %s" % (k, ("%.3f" % v) if np.isfinite(v) else "nan") for k, v in ratio.items()))
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
    return rows


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
    lines = ["rcn_balance_ledger v1.2", ""]
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
            # k = grad theta through the cut-free form cos th grad sin th - sin th grad cos th
            c, sn = np.cos(th), np.sin(th)
            k1e = c * np.gradient(sn, dx, axis=1) - sn * np.gradient(c, dx, axis=1)
            k2e = c * np.gradient(sn, dx, axis=0) - sn * np.gradient(c, dx, axis=0)
            rr = np.min(np.hypot(X[..., None] - xd[None, None, :], Y[..., None]), axis=-1)
            if args.pn_amp:
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
            stem = "pn_row_amp%d" % int(bool(args.pn_amp))
            mu = 1.0 - 1e-9   # rolls along y in that construction; row distance uses Lsp = pi/k1 -> use period
            Lsp = 2 * np.pi
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
        json.dump(vars(args), fh, indent=2, default=str)
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
    pn_amp = 0                   # pn mode: 1 = attach the SH amplitude A1(w) to the Pismen phase
    pn_spacing = 0.0             # pn mode: dislocation spacing (0 = the zigzag core spacing pi/k1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in dir(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    cli = p.parse_args()
    if cli.npz:
        cli.mode = "data"
    sys.exit(main(cli))
