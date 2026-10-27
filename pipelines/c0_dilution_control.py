# pipelines/c0_dilution_control.py
"""
Workstream C0 (10 Oct 2026): is the attenuation of the mg forced RCN
triple on the knee control mostly regression dilution near the
stationary balance, or mostly a deterministic distortion (blur) of the
weak-form columns by the estimator?

Everything is measured on the RCN knee control of wsindy_theta_t_v19
(--mode control), which is imported and not edited: the same knee
relaxation, the same mg estimator frame by frame, the same bump test
function at 1.5 L and t_half 3, the same query centres, the same
weak-form columns. The script takes the RCN triple columns
{w div k, k.grad w, Lap div k} and the target Int[theta_t phi] at the
query centres on BOTH routes (exact k and theta_t from the PDE; the
estimator's k and theta_t) and asks what turns the exact (2, 2, -1)
into the estimator's (1.41, 1.43, -1.37).

Geometry (plain terms). On the control the truth is
    theta_t = 2 S - B,   S = w div k + k.grad w = div(w k),  B = Lap div k.
Near the self-dual balance 2S and B nearly cancel, so the target is a
small residual of two large, nearly proportional columns. In the plane
of coefficients (c_S, c_B) the data then pin down one combination very
well (the direction along which the columns ADD, roughly (1, 2): the
"strong" direction, the large eigenvector of the Gram matrix X^T X) and
the other only weakly (the direction along which they CANCEL, roughly
(2, -1): the "weak" direction, the small eigenvector). The truth
(2, -1) lies entirely along the weak direction. Errors-in-variables
theory says: noise in the columns of covariance Sigma shrinks the
least-squares answer by (G + Sigma)^-1 G, which bites only where G is
small, so a weak-direction coefficient is shrunk by about
lambda_w / (lambda_w + sigma_w^2) and the strong direction is left
alone. A deterministic distortion of the columns (X -> X (I + Gamma))
instead rotates and rescales the answer by (I + Gamma)^-1 whatever the
signal size. The two mechanisms are told apart by how the attenuation
scales with the signal (the distance from balance) at fixed estimator.

Tests
  1. Error decomposition. E = X_m - X_ex and e_b = b_m - b_ex are
     regressed on the exact columns: E = X_ex Gamma + eps (a gain part,
     blur-like, and a residual, noise-like), e_b = X_ex gamma_b + eps_b.
     Synthetic refits of the forced law with each piece switched on
     alone and in combination say which piece reproduces the record.
     The analytic dilution prediction (G + Sigma_eps)^-1 (G c +
     Sigma_eps_b) is printed beside the refit it should equal.
  2. Errors-in-variables corrections on the estimator route: OLS with
     the Gram matrix and cross-moment corrected by the residual-error
     covariance (own beta0, and transferred from beta0 1.8), and total
     least squares. Under dilution the corrected triple returns to
     (2, 2, -1) at every beta0; under blur it returns the gain-only
     answer.
  3. Signal-size test inside one run (beta0 1.8): the query centres are
     split into early, middle and late frame windows (the knee far from
     balance, then nearer), and the shrink of the weak direction is
     read per window beside the window's own lambda_w / (lambda_w +
     sigma_w^2) and its crease width. Dilution: the shrink follows the
     signal (worse late). Blur: it follows the crease width (worse
     early) or stays flat.

Readouts per beta0 and per window: the forced triple and the tied pair
{S, B} on both routes; the Gram eigen-directions; the weak-direction
shrink and the strong-direction shift of every fit; holdout R^2 (train
x < median, test the rest, as forced_fit does).

Outputs (relative to this script): results/c0_dilution_control/<stamp>/
  report.txt, c0_results.json, columns_beta<b>.npz (the sampled
  columns, both routes, for refits without the estimator),
  fig_shrink_vs_beta0.png, fig_decomposition.png, fig_frame_windows.png

Changelog
---------
v1.0  First version (10 Oct 2026). --control_amp slaved renders the knee
      as u = A cos theta with the slaved amplitude of v1.8 (A floor as
      there), so the decomposition can be read on an SH-like crease
      with an amplitude dip; the phase law is unchanged. Sandbox step 0 reproduced the control
      of record to the printed digits (exact (2.008, 2.008, -1.01), R2
      1.0000; estimator (1.408, 1.427, -1.366), R2 0.9545; theta_t rel
      RMS 0.3525) before this script was written.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import wsindy_theta_t_v19 as W                                        # noqa: E402
from wsindy_theta_t_v19 import (rcn_knee_relaxation, estimate, theta_t_from_z,  # noqa: E402
                                kernel_sizes, query_centres, build_columns,
                                forced_fit, signal_line, t1c)
from self_dual_sindy import r_squared                                 # noqa: E402

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

TRIPLE = ["w div k", "k.grad w", "Lap div k"]
TRUTH3 = np.array([2.0, 2.0, -1.0])
TRUTH2 = np.array([2.0, -1.0])            # tied pair {S, B}


# ---------------------------------------------------------------------
# the control, both routes, columns at the query centres
# ---------------------------------------------------------------------

def control_columns(args, beta0, lines):
    """replicates run_control's assembly (v1.9) and returns the sampled
    RHS columns and theta_t target on both routes, with the frame index
    and x of every centre and the crease width per frame"""
    mu = args.mu
    k1 = float(np.sqrt(1 - mu ** 2))
    Lsp = np.pi / k1
    alpha, tau, eta_b, R = args.alpha, args.tau, args.eta_b, args.R
    ny_1d = args.ny_control
    y, tt, p, Phi, pt = rcn_knee_relaxation(mu, alpha, eta_b, tau, R, beta0,
                                            args.Ly, ny_1d, args.t_end, args.nt)
    nper = args.n_xper
    Lx = nper * 2 * np.pi / k1
    dx = y[1] - y[0]
    nx = int(round(Lx / dx)); Lx = nx * dx
    x = (np.arange(nx) - nx // 2) * dx
    nt = len(tt)
    th = k1 * x[None, :, None] + Phi[:, None, :]
    Phit = np.cumsum(0.5 * (pt[1:] + pt[:-1]) * dx, axis=0)
    Phit = np.vstack([np.zeros((1, nt)), Phit])
    th_t_ex = np.broadcast_to(Phit[:, None, :], th.shape).copy()
    k1_ex = np.full(th.shape, k1); k2_ex = np.broadcast_to(p[:, None, :], th.shape).copy()
    crease = np.array([np.gradient(p[:, j], y)[ny_1d // 2] / mu ** 2 for j in range(nt)])
    lines.append("RCN control: mu %.2f, alpha %.3f, tau %.3f, eta_b %.3f, beta0 %.2f, Ly %.1f,"
                 " t in [0, %.2f] (%d frames, dt %.4f), grid %dx%d" %
                 (mu, alpha, tau, eta_b, beta0, args.Ly, args.t_end, nt, tt[1] - tt[0], ny_1d, nx))
    lines.append("  knee width ratio p_y(0)/mu^2: %.3f at t = 0 -> %.3f at t_end (1 = self-dual balance)" %
                 (crease[0], crease[-1]))
    camp = str(getattr(args, "control_amp", "none"))
    if camp == "slaved":                                   # v1.8 slaved amplitude, as run_control
        A_far = float(np.sqrt(4.0 / 3.0 * R))
        floor = float(args.amp_floor) * A_far
        w1 = k1 ** 2 + p ** 2 - 1.0
        A1 = np.sqrt(np.maximum(4.0 / 3.0 * (R - w1 ** 2), floor ** 2))
        u = A1[:, None, :] * np.cos(th)
        lines.append("  amplitude-coupled control: slaved A = sqrt(max(4/3 (R - w^2), floor^2)), floor %.2g x A_far;"
                     " on the crease line A = %.3f at t = 0 and %.3f at t_end" % (args.amp_floor, A1[ny_1d // 2, 0], A1[ny_1d // 2, -1]))
    else:
        u = np.cos(th)
    taper = t1c.edge_taper(x, y, frac=args.taper_frac)
    Lyy = ny_1d * dx
    ests, n_prev = [], None
    for i in range(nt):
        e = estimate(u[..., i], Lx, Lyy, args.sigma_dir, n_prev, taper)
        n_prev = e["n"]; ests.append(e)
    dt = tt[1] - tt[0]
    k1m = np.stack([e["k1"] for e in ests], -1); k2m = np.stack([e["k2"] for e in ests], -1)
    zs = np.stack([e["z"] for e in ests], -1)
    del ests
    th_t_m = theta_t_from_z(zs, dt)
    s = np.sign(np.mean(k1m, axis=(0, 1)))[None, None, :]; s[s == 0] = 1
    routes = {"exact": (k1_ex, k2_ex, th_t_ex), "estimator": (k1m * s, k2m * s, th_t_m * s)}
    mask_tt = np.abs(th_t_m - th_t_ex)[..., 2:-2]
    lines.append("  estimator theta_t vs exact, rel RMS over the box (frames 2..nt-3): %.4f" %
                 (np.sqrt(np.nanmean(mask_tt ** 2)) / np.sqrt(np.mean(th_t_ex[..., 2:-2] ** 2))))
    m = kernel_sizes(args, dx, dx, dt, Lsp)
    idx = query_centres(th.shape, m, args, excl=None)
    lines.append("  query centres %d" % len(idx[0]))
    lines.append("  [exact route]" + signal_line(th_t_ex, idx)[1:])
    out = dict(xs=x[idx[1]], it=idx[2], tt=tt, crease=crease, beta0=beta0, Lsp=Lsp)
    for route, (K1, K2, TT) in routes.items():
        cols, targets, _ = build_columns(K1, K2, TT, x, y, tt, args, Lsp, idx, A=None, Abar=None, snap=None)
        C = cols["rhs"]
        out[route] = dict(cols={nm: np.asarray(C[nm]) for nm in W.RHS_LIB}, b=np.asarray(targets["theta_t"]))
    return out


# ---------------------------------------------------------------------
# small linear algebra, in plain terms
# ---------------------------------------------------------------------

def ols(X, b):
    c, *_ = np.linalg.lstsq(X, b, rcond=None)
    return c


def split_fit(X, b, xs):
    """forced_fit's rule: train on x < median, holdout R2 on the rest"""
    tr = xs < np.median(xs)
    c = ols(X[tr], b[tr])
    return c, float(r_squared(b[~tr], X[~tr] @ c))


def gram_geometry(X):
    """eigen-directions of G = X^T X: (lambda_w, u_w) the weak direction
    (small eigenvalue, where the columns cancel), (lambda_s, u_s) the
    strong one; for 3 columns the middle direction is reported too"""
    G = X.T @ X
    lam, U = np.linalg.eigh(G)
    order = np.argsort(lam)
    lam, U = lam[order], U[:, order]
    # orient: weak direction with the sign of the truth
    return G, lam, U


def components(c, U):
    return U.T @ c


def shrink_and_shift(c_hat, c_true, U):
    """weak-direction shrink (ratio of components along U[:, 0]) and the
    shift along the other directions (estimate minus truth)"""
    a_hat, a_true = components(c_hat, U), components(c_true, U)
    shrink = float(a_hat[0] / a_true[0]) if abs(a_true[0]) > 0 else np.nan
    shift = a_hat[1:] - a_true[1:]
    return shrink, shift


def pair_from_triple(cols):
    """tied pair columns: S = w div k + k.grad w, B = Lap div k"""
    return np.column_stack([cols["w div k"] + cols["k.grad w"], cols["Lap div k"]])


def triple(cols):
    return np.column_stack([cols[nm] for nm in TRIPLE])


def tls(X, b):
    """total least squares: smallest right singular vector of [X b]"""
    M = np.column_stack([X, b])
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    v = Vt[-1]
    return -v[:-1] / v[-1]


def fmt(c):
    return "(" + ", ".join("%+.3f" % v for v in np.atleast_1d(c)) + ")"


# ---------------------------------------------------------------------
# test 1: error decomposition and synthetic refits
# ---------------------------------------------------------------------

def decomposition(Xe, be, Xm, bm, xs, truth, label, lines, geom_U=None):
    """Xe, be: exact columns and target; Xm, bm: estimator's. Returns a
    dict of fits for every synthetic combination, read in the weak and
    strong directions of the exact Gram matrix (train rows)."""
    tr = xs < np.median(xs)
    G, lam, U = gram_geometry(Xe[tr])
    if geom_U is not None:
        U = geom_U
    # make the weak direction point the way the truth does
    if components(truth, U)[0] < 0:
        U = U.copy(); U[:, 0] *= -1
    # error pieces, regressed on the exact columns (train rows define Gamma)
    E = Xm - Xe
    eb = bm - be
    Gamma = ols(Xe[tr], E[tr])                 # [p, p]: column j of E ~ Xe Gamma[:, j]
    gam_b = ols(Xe[tr], eb[tr])                # [p]
    eps = E - Xe @ Gamma
    eps_b = eb - Xe @ gam_b
    Sig = eps[tr].T @ eps[tr]
    Sig_b = eps[tr].T @ eps_b[tr]
    cases = {
        "exact columns, exact target": (Xe, be),
        "residual column error only": (Xe + eps, be),
        "column gain only": (Xe @ (np.eye(Xe.shape[1]) + Gamma), be),
        "measured target, exact columns": (Xe, bm),
        "residual target error only": (Xe, be + eps_b),
        "gains on columns and target": (Xe @ (np.eye(Xe.shape[1]) + Gamma), be + Xe @ gam_b),
        "residual errors on both": (Xe + eps, be + eps_b),
        "measured columns, measured target": (Xm, bm),
    }
    res = {}
    lines.append("  %s: Gram eigenvalues (train rows) %s; weak direction u_w %s, strong u_s %s" %
                 (label, ", ".join("%.3g" % v for v in lam), fmt(U[:, 0]), fmt(U[:, -1])))
    lines.append("    truth %s has components %s on (u_w, ..., u_s)" % (fmt(truth), fmt(components(truth, U))))
    for nm, (X, b) in cases.items():
        c, r2 = split_fit(X, b, xs)
        sh, shift = shrink_and_shift(c, truth, U)
        res[nm] = dict(coefs=c.tolist(), r2=r2, shrink=sh, shift=shift.tolist())
        lines.append("    %-36s %s  R2 %.4f  weak-direction shrink %.3f  strong shift %s" %
                     (nm, fmt(c), r2, sh, fmt(shift)))
    # analytic dilution predictions
    c_dil = np.linalg.solve(G + Sig, G @ truth)
    c_dil_b = np.linalg.solve(G + Sig, G @ truth + Sig_b)
    sw2 = float(U[:, 0] @ Sig @ U[:, 0])
    scalar = float(lam[0] / (lam[0] + sw2))
    res["analytic"] = dict(dilution_columns_only=c_dil.tolist(), dilution_both=c_dil_b.tolist(),
                           lambda_w=float(lam[0]), sigma_w2=sw2, scalar_shrink=scalar,
                           lambda_all=lam.tolist(), U=U.tolist(), Gamma=Gamma.tolist(),
                           gamma_b=gam_b.tolist(), Sigma=Sig.tolist(), Sigma_b=Sig_b.tolist(),
                           col_rms_exact=np.sqrt(np.mean(Xe[tr] ** 2, axis=0)).tolist(),
                           err_rms_total=np.sqrt(np.mean(E[tr] ** 2, axis=0)).tolist(),
                           err_rms_residual=np.sqrt(np.mean(eps[tr] ** 2, axis=0)).tolist(),
                           target_rms_exact=float(np.sqrt(np.mean(be[tr] ** 2))),
                           target_err_rms_total=float(np.sqrt(np.mean(eb[tr] ** 2))),
                           target_err_rms_residual=float(np.sqrt(np.mean(eps_b[tr] ** 2))))
    lines.append("    analytic (G + Sigma)^-1 G c = %s ; with the target cross-term %s ;"
                 " scalar lambda_w/(lambda_w + sigma_w^2) = %.3g/(%.3g + %.3g) = %.3f" %
                 (fmt(c_dil), fmt(c_dil_b), lam[0], lam[0], sw2, scalar))
    lines.append("    column rms (exact) %s ; column error rms total %s, residual after the gain %s" %
                 (fmt(res["analytic"]["col_rms_exact"]), fmt(res["analytic"]["err_rms_total"]),
                  fmt(res["analytic"]["err_rms_residual"])))
    lines.append("    target rms (exact) %.4g ; target error rms total %.4g, residual after the gain %.4g ;"
                 " gain matrix I + Gamma rows %s ; target gain gamma_b %s" %
                 (res["analytic"]["target_rms_exact"], res["analytic"]["target_err_rms_total"],
                  res["analytic"]["target_err_rms_residual"],
                  " / ".join(fmt(r) for r in (np.eye(Xe.shape[1]) + Gamma)), fmt(gam_b)))
    return res, dict(G=G, Sig=Sig, Sig_b=Sig_b, U=U, lam=lam)


# ---------------------------------------------------------------------
# test 2: errors-in-variables corrections on the estimator route
# ---------------------------------------------------------------------

def eiv_corrections(Xm, bm, xs, truth, Sig_own, Sigb_own, Sig_ref, Sigb_ref, U, lines, label):
    tr = xs < np.median(xs)
    Gm = Xm[tr].T @ Xm[tr]; hm = Xm[tr].T @ bm[tr]
    out = {}
    c_ols = np.linalg.solve(Gm, hm)
    for nm, (S, Sb) in {"corrected OLS, own Sigma": (Sig_own, Sigb_own),
                        "corrected OLS, Sigma from reference beta0": (Sig_ref, Sigb_ref)}.items():
        try:
            c = np.linalg.solve(Gm - S, hm - Sb)
        except np.linalg.LinAlgError:
            c = np.full(len(truth), np.nan)
        r2 = float(r_squared(bm[~tr], Xm[~tr] @ c)) if np.all(np.isfinite(c)) else np.nan
        sh, shift = shrink_and_shift(c, truth, U)
        out[nm] = dict(coefs=c.tolist(), r2=r2, shrink=sh, shift=shift.tolist())
        lines.append("    %-42s %s  R2(holdout, OLS sense) %.4f  shrink %.3f  shift %s" %
                     (nm, fmt(c), r2, sh, fmt(shift)))
    c = tls(Xm[tr], bm[tr])
    sh, shift = shrink_and_shift(c, truth, U)
    out["total least squares"] = dict(coefs=c.tolist(), r2=float(r_squared(bm[~tr], Xm[~tr] @ c)),
                                      shrink=sh, shift=shift.tolist())
    lines.append("    %-42s %s  R2 %.4f  shrink %.3f  shift %s" %
                 ("total least squares", fmt(c), out["total least squares"]["r2"], sh, fmt(shift)))
    out["ols"] = dict(coefs=c_ols.tolist())
    return out


# ---------------------------------------------------------------------
# test 3: frame windows inside one run
# ---------------------------------------------------------------------

def frame_windows(D, windows, lines, U_fixed=None):
    xs, it, crease = D["xs"], D["it"], D["crease"]
    Xe3, be = triple(D["exact"]["cols"]), D["exact"]["b"]
    Xm3, bm = triple(D["estimator"]["cols"]), D["estimator"]["b"]
    Xe2, Xm2 = pair_from_triple(D["exact"]["cols"]), pair_from_triple(D["estimator"]["cols"])
    res = []
    for wname, (i0, i1) in windows.items():
        sel = (it >= i0) & (it <= i1)
        if sel.sum() < 50:
            continue
        xs_w = xs[sel]
        tr = xs_w < np.median(xs_w)
        G, lam, U = gram_geometry(Xe2[sel][tr])
        if components(TRUTH2, U)[0] < 0:
            U = U.copy(); U[:, 0] *= -1
        if U_fixed is not None:
            U = U_fixed
        E = (Xm2 - Xe2)[sel]
        Gamma = ols(Xe2[sel][tr], E[tr]); eps = E - Xe2[sel] @ Gamma
        Sig = eps[tr].T @ eps[tr]
        sw2 = float(U[:, 0] @ Sig @ U[:, 0])
        lam_w = float(U[:, 0] @ G @ U[:, 0])
        c_ex3, r2e3 = split_fit(Xe3[sel], be[sel], xs_w)
        c_m3, r2m3 = split_fit(Xm3[sel], bm[sel], xs_w)
        c_ex2, r2e2 = split_fit(Xe2[sel], be[sel], xs_w)
        c_m2, r2m2 = split_fit(Xm2[sel], bm[sel], xs_w)
        sh_m, shift_m = shrink_and_shift(c_m2, TRUTH2, U)
        sh_e, _ = shrink_and_shift(c_ex2, TRUTH2, U)
        c_dil = np.linalg.solve(G + Sig, G @ TRUTH2)
        sh_dil, _ = shrink_and_shift(c_dil, TRUTH2, U)
        sig = float(np.sqrt(np.mean(be[sel] ** 2)))
        cw = float(np.mean(crease[i0:i1 + 1]))
        r = dict(window=wname, frames=[int(i0), int(i1)], n=int(sel.sum()), crease_width_ratio=cw,
                 target_rms_exact=sig, lambda_w=lam_w, sigma_w2=sw2,
                 scalar_shrink=lam_w / (lam_w + sw2), dilution_pred_shrink=sh_dil,
                 exact_triple=c_ex3.tolist(), exact_r2=r2e3, estimator_triple=c_m3.tolist(), estimator_r2=r2m3,
                 exact_pair=c_ex2.tolist(), estimator_pair=c_m2.tolist(), estimator_pair_r2=r2m2,
                 measured_shrink=sh_m, exact_shrink=sh_e, strong_shift=shift_m.tolist(),
                 col_err_rms_residual=np.sqrt(np.mean(eps[tr] ** 2, axis=0)).tolist())
        res.append(r)
        lines.append("    %-8s frames %2d..%2d  n %6d  crease p_y/mu^2 %.3f  target rms %.2e  exact triple %s (R2 %.4f)"
                     "  estimator triple %s (R2 %.4f)  pair %s  shrink measured %.3f, dilution-predicted %.3f,"
                     " scalar %.3f  strong shift %+.3f" %
                     (wname, i0, i1, sel.sum(), cw, sig, fmt(c_ex3), r2e3, fmt(c_m3), r2m3, fmt(c_m2),
                      sh_m, sh_dil, lam_w / (lam_w + sw2), shift_m[0]))
    return res


# ---------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------

def fig_shrink(summary, path):
    b0 = [s["beta0"] for s in summary]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axs[0]
    keys = [("measured columns, measured target", "record (estimator route)", "k", "o"),
            ("residual column error only", "residual column error only (dilution)", "C0", "s"),
            ("residual errors on both", "residual errors on both", "C0", "^"),
            ("column gain only", "column gain only (blur)", "C3", "s"),
            ("gains on columns and target", "gains on both (blur)", "C3", "^")]
    for k, lab, col, mk in keys:
        ax.plot(b0, [s["pair"][k]["shrink"] for s in summary], marker=mk, color=col, lw=1, label=lab)
    ax.plot(b0, [s["pair"]["analytic"]["scalar_shrink"] for s in summary], ":", color="C0",
            label="scalar lambda_w/(lambda_w + sigma_w^2)")
    ax.axhline(1, color="0.6", lw=0.8)
    ax.set_xlabel("planted beta0 (1 = balance)"); ax.set_ylabel("weak-direction shrink (1 = exact)")
    ax.set_title("tied pair {S, B}: what shrinks the (2, -1) direction", fontsize=9)
    ax.legend(fontsize=6)
    ax = axs[1]
    for k, lab, col, mk in keys:
        ax.plot(b0, [s["pair"][k]["shift"][0] for s in summary], marker=mk, color=col, lw=1, label=lab)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("planted beta0"); ax.set_ylabel("strong-direction shift (0 = exact)")
    ax.set_title("the well-identified bias ('stretching low, bending high')", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_decomp(summary, path):
    names = ["exact columns, exact target", "residual column error only", "column gain only",
             "measured target, exact columns", "residual target error only", "gains on columns and target",
             "residual errors on both", "measured columns, measured target"]
    n = len(summary)
    fig, axs = plt.subplots(1, n, figsize=(3.2 * n, 3.8), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, s in zip(axs, summary):
        xpos = np.arange(len(names))
        wd = 0.26
        for j, (nm, col) in enumerate(zip(TRIPLE, ("C0", "C1", "C3"))):
            vals = [s["triple"][k]["coefs"][j] for k in names]
            ax.bar(xpos + (j - 1) * wd, vals, wd, color=col, label=nm if ax is axs[0] else None)
            ax.axhline(TRUTH3[j], color=col, lw=0.7, ls="--")
        ax.set_xticks(xpos); ax.set_xticklabels(names, rotation=70, ha="right", fontsize=6)
        ax.set_title("beta0 %.2f" % s["beta0"], fontsize=9)
        ax.axhline(0, color="k", lw=0.5)
    axs[0].set_ylabel("forced triple coefficient (dashed: truth)")
    axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_windows(res, beta0, path):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    lab = [r["window"] for r in res]
    xp = np.arange(len(res))
    ax = axs[0]
    ax.plot(xp, [r["measured_shrink"] for r in res], "ko-", label="measured (estimator route)")
    ax.plot(xp, [r["dilution_pred_shrink"] for r in res], "s--", color="C0", label="dilution-predicted")
    ax.plot(xp, [r["scalar_shrink"] for r in res], ":", color="C0", label="scalar formula")
    ax.plot(xp, [r["exact_shrink"] for r in res], "x-", color="0.5", label="exact route")
    ax.set_xticks(xp); ax.set_xticklabels(lab); ax.set_ylabel("weak-direction shrink"); ax.legend(fontsize=7)
    ax.set_title("beta0 %.2f, frame windows" % beta0, fontsize=9)
    ax = axs[1]
    ax.semilogy(xp, [r["target_rms_exact"] for r in res], "o-", color="C2", label="target rms (signal)")
    ax.semilogy(xp, [np.sqrt(r["sigma_w2"] / r["n"] * 2) for r in res], "s-", color="C3",
                label="residual column error, weak direction (rms)")
    ax.set_xticks(xp); ax.set_xticklabels(lab); ax.legend(fontsize=7); ax.set_title("signal against column error", fontsize=9)
    ax = axs[2]
    ax.plot(xp, [r["crease_width_ratio"] for r in res], "o-", color="C1")
    ax.set_xticks(xp); ax.set_xticklabels(lab); ax.set_ylabel("knee width ratio p_y(0)/mu^2 (mean over window)")
    ax.set_title("crease sharpness per window", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main(args):
    t_start = time.time()
    base = W._Args()
    for k in [a for a in dir(base) if not a.startswith("_")]:
        if hasattr(args, k) and k not in ("outdir", "tag"):
            setattr(base, k, getattr(args, k))
    base.score_h = 0
    beta0s = [float(v) for v in str(args.beta0s).split(",")]
    stamp = time.strftime("%m%d") + "_c0_dilution_rad%g_th%d_b0%s%s%s" % (
        base.radius, base.t_half, "-".join("%g" % b for b in beta0s),
        ("_amp" + str(base.control_amp) + "_R%g" % base.R) if str(base.control_amp) != "none" else "",
        ("_" + args.tag) if args.tag else "")
    out = (_HERE / args.outdir / stamp) if not Path(args.outdir).is_absolute() else Path(args.outdir) / stamp
    out.mkdir(parents=True, exist_ok=True)
    lines = ["c0_dilution_control v1.0 (%s)" % time.strftime("%Y-%m-%d %H:%M"),
             "radius %g L, t_half %d, stride_xy %d, mu %.2f, beta0 set %s" %
             (base.radius, base.t_half, base.stride_xy, base.mu, beta0s), ""]
    summary = []
    ref = None
    data = {}
    for b0 in beta0s:
        D = control_columns(base, b0, lines)
        data[b0] = D
        np.savez_compressed(out / ("columns_beta%g.npz" % b0), xs=D["xs"], it=D["it"], tt=D["tt"], crease=D["crease"],
                            **{"exact_" + nm.replace(" ", "_").replace(".", ""): v for nm, v in D["exact"]["cols"].items()},
                            **{"est_" + nm.replace(" ", "_").replace(".", ""): v for nm, v in D["estimator"]["cols"].items()},
                            exact_b=D["exact"]["b"], est_b=D["estimator"]["b"])
        xs = D["xs"]
        Xe3, be = triple(D["exact"]["cols"]), D["exact"]["b"]
        Xm3, bm = triple(D["estimator"]["cols"]), D["estimator"]["b"]
        Xe2, Xm2 = pair_from_triple(D["exact"]["cols"]), pair_from_triple(D["estimator"]["cols"])
        # the record, by forced_fit itself
        fe = forced_fit(TRIPLE, Xe3, be, xs, TRIPLE); fm = forced_fit(TRIPLE, Xm3, bm, xs, TRIPLE)
        lines.append("  forced triple by forced_fit: exact %s R2 %.4f ; estimator %s R2 %.4f" %
                     (fmt([fe["coefs"][k] for k in TRIPLE]), fe["r2_holdout"],
                      fmt([fm["coefs"][k] for k in TRIPLE]), fm["r2_holdout"]))
        lines.append("  --- test 1, tied pair {S, B} (truth (2, -1)) ---")
        res2, geo2 = decomposition(Xe2, be, Xm2, bm, xs, TRUTH2, "pair", lines)
        lines.append("  --- test 1, forced triple (truth (2, 2, -1)) ---")
        res3, geo3 = decomposition(Xe3, be, Xm3, bm, xs, TRUTH3, "triple", lines)
        if ref is None:
            ref = dict(pair=geo2, triple=geo3, beta0=b0)
        lines.append("  --- test 2, errors-in-variables corrections, estimator route (reference Sigma from beta0 %g) ---" % ref["beta0"])
        lines.append("    pair:")
        eiv2 = eiv_corrections(Xm2, bm, xs, TRUTH2, geo2["Sig"], geo2["Sig_b"], ref["pair"]["Sig"], ref["pair"]["Sig_b"],
                               geo2["U"], lines, "pair")
        lines.append("    triple:")
        eiv3 = eiv_corrections(Xm3, bm, xs, TRUTH3, geo3["Sig"], geo3["Sig_b"], ref["triple"]["Sig"], ref["triple"]["Sig_b"],
                               geo3["U"], lines, "triple")
        summary.append(dict(beta0=b0, pair=res2, triple=res3, eiv_pair=eiv2, eiv_triple=eiv3,
                            forced_exact=fe, forced_estimator=fm,
                            crease=[float(D["crease"][0]), float(D["crease"][-1])]))
        lines.append("")
    # test 3 on the first beta0 of the set (the record's 1.8 when present)
    b_win = 1.8 if 1.8 in beta0s else beta0s[0]
    D = data[b_win]
    it = D["it"]; i0, i1 = int(it.min()), int(it.max())
    thirds = np.linspace(i0, i1 + 1, 4).astype(int)
    windows = {"early": (thirds[0], thirds[1] - 1), "middle": (thirds[1], thirds[2] - 1),
               "late": (thirds[2], thirds[3] - 1), "all": (i0, i1)}
    lines.append("--- test 3, frame windows at beta0 %g (centres at frames %d..%d) ---" % (b_win, i0, i1))
    win = frame_windows(D, windows, lines)
    lines.append("")
    lines.append("elapsed %.1f s" % (time.time() - t_start))
    lines.append("outputs -> %s" % out)
    fig_shrink(summary, out / "fig_shrink_vs_beta0.png")
    fig_decomp(summary, out / "fig_decomposition.png")
    fig_windows(win, b_win, out / "fig_frame_windows.png")
    with open(out / "c0_results.json", "w") as fh:
        json.dump(dict(summary=summary, windows=win, beta0_windows=b_win,
                       config={k: getattr(base, k) for k in dir(base) if not k.startswith("_")}), fh, indent=1, default=float)
    (out / "report.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below. Control and
    estimator settings not listed here take wsindy_theta_t_v19._Args."""
    beta0s = "1.8,2.5,1.4,1.2,1.05"   # first entry supplies the reference Sigma for the transfer test
    outdir = str(Path("results") / "c0_dilution_control")
    tag = ""
    # control (defaults = the record)
    mu = 0.85
    radius = 1.5
    t_half = 3
    stride_xy = 4
    ny_control = 300
    Ly = 60.0
    t_end = 2.0
    nt = 41
    n_xper = 4
    sigma_dir = 0.4
    control_amp = "none"      # none | slaved (v1.8 amplitude-coupled control; the phase law stays pure RCN)
    amp_floor = 0.05
    R = 0.5                   # control R (alpha, tau, eta_b stay at the v1.9 defaults; R sets the slaved amplitude only)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in dir(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    sys.exit(main(p.parse_args()))
