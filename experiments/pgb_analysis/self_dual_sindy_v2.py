"""
self_dual_sindy.py
------------------
Sparse identification (STLSQ) of the self-dual balance on a knee-bend PGB,
from a SINGLE frame of uHu-extracted wavevector data.

Idea
====
The regularized Cross-Newell self-dual equation reads

    eps * div k  =  +/- G(|k|^2),      k = grad(theta),

with G^2(s) = -s^4 + 4 s^3 - 5 s^2 + 2 s  (exact SH->RCN form).
In the shifted variable  w = |k|^2 - 1  this is EXACTLY

    G^2 = w^2 - w^4,

so for |w| << 1 the balance is a low-order polynomial statement:

    (div k)^2  ~=  c * w^2          (energy form,   c ?= alpha/eta or 1)
     div k     ~=  -c * w           (self-dual form, sign fixed by GB branch)

This script regresses each target against a LIBRARY of candidate pointwise
fields (mismatch monomials w, w^2, w^3; curvature/rotational distractors
curl k, J = det(grad k); component distractors k2, k1*k2; gradient
distractor |grad w|) using sequentially-thresholded least squares, and
reports which terms survive, with what coefficients, against theory.

A single frame suffices: the balance is spatial, and one frame supplies
O(1e4-1e5) valid samples. Spatial holdout (train x<0, test x>0) and
bootstrap resampling give honest generalization / coefficient-error
estimates.

Data conventions match the pgb_analysis repo:
- npz with keys x, y, u, ramp, k1_sym, k2_sym (knee/uhu files), OR
  k1_orig / k2_orig (phase files) -- key names configurable below.
- ramp-threshold + erosion masks, BFS orientation, pi-jump masking,
  identical in spirit to knee_bend_fits_and_residuals.py.

Run modes:
    python self_dual_sindy.py                # uses _Args below (PyCharm style)
    python self_dual_sindy.py --op_path ...  # CLI
    python self_dual_sindy.py --synthetic    # analytic Hopf-Cole knee test
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, gaussian_filter

# ---------------------------------------------------------------------
# Paper style (self-contained; mirror of paper_style.py)
# ---------------------------------------------------------------------

PAPER_RC = {
    "font.size": 11,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "image.origin": "lower",
}

PANEL_LABEL_KW = dict(fontsize=11, fontweight="bold", va="top", ha="left")


def panel_label(ax, s, dx=0.02, dy=0.98, color="k"):
    ax.text(dx, dy, s, transform=ax.transAxes, color=color, **PANEL_LABEL_KW)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class _Args:
    # --- input ---
    op_path: str = (
        "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/uhu/"
        "shallow_mus/sig_pio2/raw/"
        "0312_v3_sh_pgb_zigzag_cropped_v3_knee_mu0.4_T100_N50_nx512_Ny512_"
        "uhu_sigma1.57.npz"
    )
    k1_key: str = "k1_sym"        # "k1_orig" for phase files
    k2_key: str = "k2_sym"
    t_index: int = -1
    mu: float = 0.4               # far-field mu = sin(alpha); used for theory lines
    synthetic: bool = False       # ignore op_path; build analytic knee
    synthetic_lambda: float = 1.0 # ground-truth balance coeff for synthetic test
    synthetic_noise: float = 0.005
    synthetic_sigma: float = np.pi / 2.0  # mimic uHu Gaussian averaging

    # --- masking (matches knee_bend_fits_and_residuals.py) ---
    ramp_thresh_bfs: float = 0.98
    ramp_thresh_strict: float = 0.999
    ramp_erosion_iters: int = 48
    phi_jump_tol: float = np.pi / 10.0

    # --- RCN weights (theory reference values only; not used in the fit) ---
    alpha: float = 2.0 / 3.0
    eta: float = 8.0 / 9.0

    # --- regression ---
    thresholds: tuple = tuple(np.geomspace(1e-4, 1.0, 25))
    n_boot: int = 200
    boot_frac: float = 0.5
    rel_tol_pareto: float = 0.01   # accept sparsest model within 1% of best R^2
    weight_by_target: bool = False # optionally weight rows by |target|
    seed: int = 0

    # --- output ---
    out_root: str = ""             # default: results/self_dual_sindy/<stem>
    make_field_maps: bool = True   # 2D diagnostics ported from knee-bend script


# ---------------------------------------------------------------------
# Helpers reused from the repo (vectorized where the originals looped)
# ---------------------------------------------------------------------

def orient_vector_field(f, g, mask):
    """BFS sign-alignment of a director field on mask (same as repo)."""
    ny, nx = f.shape
    kx = np.zeros_like(f)
    ky = np.zeros_like(g)
    visited = np.zeros_like(mask, dtype=bool)

    start = np.argwhere(mask)
    if start.size == 0:
        return kx, ky

    iy0, ix0 = start[0]
    kx[iy0, ix0] = f[iy0, ix0]
    ky[iy0, ix0] = g[iy0, ix0]
    visited[iy0, ix0] = True
    queue = deque([(iy0, ix0)])
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while queue:
        iy, ix = queue.popleft()
        for dy, dx in offsets:
            j, i = iy + dy, ix + dx
            if 0 <= j < ny and 0 <= i < nx and mask[j, i] and not visited[j, i]:
                if kx[iy, ix] * f[j, i] + ky[iy, ix] * g[j, i] < 0:
                    kx[j, i], ky[j, i] = -f[j, i], -g[j, i]
                else:
                    kx[j, i], ky[j, i] = f[j, i], g[j, i]
                visited[j, i] = True
                queue.append((j, i))
    return kx, ky


def phi_jump_mask(phi, tol=np.pi / 10.0):
    mask_jump = np.zeros_like(phi, dtype=bool)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        dphi = np.roll(np.roll(phi, dy, axis=0), dx, axis=1) - phi
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        mask_jump |= np.abs(dphi) > (np.pi - tol)
    return mask_jump


def masked_central_derivs(s, dx, dy, mask_ok):
    """Vectorized central differences; NaN wherever a needed neighbor is bad."""
    s = np.asarray(s, dtype=float)
    sx = np.full_like(s, np.nan)
    sy = np.full_like(s, np.nan)

    ok_x = mask_ok.copy()
    ok_x[:, 1:-1] &= mask_ok[:, :-2] & mask_ok[:, 2:]
    ok_x[:, [0, -1]] = False
    ok_y = mask_ok.copy()
    ok_y[1:-1, :] &= mask_ok[:-2, :] & mask_ok[2:, :]
    ok_y[[0, -1], :] = False

    gx = np.empty_like(s)
    gx[:, 1:-1] = (s[:, 2:] - s[:, :-2]) / (2.0 * dx)
    gx[:, [0, -1]] = np.nan
    gy = np.empty_like(s)
    gy[1:-1, :] = (s[2:, :] - s[:-2, :]) / (2.0 * dy)
    gy[[0, -1], :] = np.nan

    sx[ok_x] = gx[ok_x]
    sy[ok_y] = gy[ok_y]
    return sx, sy


# ---------------------------------------------------------------------
# Library construction
# ---------------------------------------------------------------------

def build_library(k1, k2, dx, dy, mask, mu):
    """
    Returns (names, columns, sample_mask, aux) where columns is a dict of 2D
    fields evaluated everywhere finite on sample_mask, plus targets.
    """
    kmag2 = k1 ** 2 + k2 ** 2
    w = kmag2 - 1.0

    k1x, k1y = masked_central_derivs(k1, dx, dy, mask)
    k2x, k2y = masked_central_derivs(k2, dx, dy, mask)

    div_k = k1x + k2y
    curl_k = k2x - k1y
    J = k1x * k2y - k1y * k2x

    wx, wy = masked_central_derivs(w, dx, dy, mask)
    grad_w = np.sqrt(wx ** 2 + wy ** 2)

    lib = {
        "1": np.ones_like(w),
        "w": w,
        "w^2": w ** 2,
        "w^3": w ** 3,
        "curl k": curl_k,
        "(curl k)^2": curl_k ** 2,
        "J": J,
        "k_2": k2,
        "k_1 k_2": k1 * k2,
        "|grad w|": grad_w,
    }

    targets = {
        "divk": div_k,                       # self-dual form
        "divk_sq": div_k ** 2,               # energy form
    }

    fields = list(lib.values()) + list(targets.values())
    sample_mask = mask.copy()
    for f in fields:
        sample_mask &= np.isfinite(f)

    aux = dict(div_k=div_k, curl_k=curl_k, J=J, w=w, kmag2=kmag2)
    return lib, targets, sample_mask, aux


# ---------------------------------------------------------------------
# STLSQ
# ---------------------------------------------------------------------

def stlsq(Theta, y, threshold, max_iter=20):
    """
    Sequentially thresholded least squares on column-normalized Theta.
    Returns coefficients in PHYSICAL units (denormalized), support mask.
    """
    norms = np.linalg.norm(Theta, axis=0)
    norms[norms == 0] = 1.0
    Th = Theta / norms

    xi = np.linalg.lstsq(Th, y, rcond=None)[0]
    for _ in range(max_iter):
        small = np.abs(xi) < threshold
        xi_new = np.zeros_like(xi)
        big = ~small
        if big.sum() == 0:
            xi = xi_new
            break
        xi_new[big] = np.linalg.lstsq(Th[:, big], y, rcond=None)[0]
        if np.array_equal(small, np.abs(xi_new) < threshold) and np.allclose(xi, xi_new):
            xi = xi_new
            break
        xi = xi_new
    return xi / norms, np.abs(xi) > 0


def r_squared(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-300)


def fit_target(names, Theta, y, args, rng, X_train_mask):
    """
    Threshold sweep -> Pareto model selection -> bootstrap on chosen support.
    Spatial holdout: fit on X_train_mask rows, score on the complement.
    """
    tr = X_train_mask
    te = ~X_train_mask

    sweep = []
    for th in args.thresholds:
        coef, supp = stlsq(Theta[tr], y[tr], th)
        r2_tr = r_squared(y[tr], Theta[tr] @ coef)
        r2_te = r_squared(y[te], Theta[te] @ coef)
        sweep.append(dict(threshold=th, coef=coef, support=supp,
                          n_terms=int(supp.sum()), r2_train=r2_tr, r2_test=r2_te))

    # Guarantee every single-term model is in the candidate pool (the
    # threshold sweep can skip over sparse supports).
    p = Theta.shape[1]
    for j in range(p):
        supp = np.zeros(p, dtype=bool)
        supp[j] = True
        coef = np.zeros(p)
        col = Theta[tr][:, j]
        denom = np.dot(col, col)
        if denom == 0:
            continue
        coef[j] = np.dot(col, y[tr]) / denom
        sweep.append(dict(threshold=np.nan, coef=coef, support=supp,
                          n_terms=1,
                          r2_train=r_squared(y[tr], Theta[tr] @ coef),
                          r2_test=r_squared(y[te], Theta[te] @ coef)))

    # Pareto selection: sparsest model whose TEST R^2 is within rel_tol of best
    best_r2 = max(s["r2_test"] for s in sweep)
    admissible = [s for s in sweep
                  if s["n_terms"] > 0
                  and s["r2_test"] >= best_r2 - args.rel_tol_pareto * abs(best_r2)]
    admissible.sort(key=lambda s: (s["n_terms"], -s["r2_test"]))
    chosen = admissible[0]

    # Refit chosen support on ALL data (unbiased coefficients), then bootstrap
    supp = chosen["support"]
    coef_full = np.zeros(Theta.shape[1])
    coef_full[supp] = np.linalg.lstsq(Theta[:, supp], y, rcond=None)[0]

    n = Theta.shape[0]
    m = max(1, int(args.boot_frac * n))
    boots = np.zeros((args.n_boot, int(supp.sum())))
    for b in range(args.n_boot):
        idx = rng.integers(0, n, size=m)
        boots[b] = np.linalg.lstsq(Theta[idx][:, supp], y[idx], rcond=None)[0]

    return dict(sweep=sweep, chosen=chosen, support=supp,
                coef_full=coef_full, boots=boots,
                r2_full=r_squared(y, Theta @ coef_full))


# ---------------------------------------------------------------------
# Synthetic knee (validation mode)
# ---------------------------------------------------------------------

def make_synthetic(args):
    """
    Analytic Hopf-Cole knee with balance coefficient lambda:
        theta = (1/lam) log( exp(lam k+.x) + exp(lam k-.x) )
    satisfies  div k = lam (1 - |k|^2)  exactly.
    k-field is Gaussian-smoothed (sigma ~ uHu window) + white noise, to mimic
    the extraction pipeline. ramp is a tanh window.
    """
    mu, lam = args.mu, args.synthetic_lambda
    k1c, k2c = np.sqrt(1 - mu ** 2), mu
    n = 512
    L = 60.0
    x = np.linspace(-L, L, n)
    y = np.linspace(-L, L, n)
    X, Y = np.meshgrid(x, y)
    dx = x[1] - x[0]

    dp = k1c * X + k2c * Y
    dm = k1c * X - k2c * Y
    mx = np.maximum(dp, dm)
    theta = lam * mx + np.log(np.exp(lam * (dp - mx)) + np.exp(lam * (dm - mx)))
    theta /= lam

    k1 = np.gradient(theta, x, axis=1)
    k2 = np.gradient(theta, y, axis=0)

    sig_grid = args.synthetic_sigma / dx
    rng = np.random.default_rng(args.seed)
    k1 = gaussian_filter(k1, sig_grid) + args.synthetic_noise * rng.standard_normal(k1.shape)
    k2 = gaussian_filter(k2, sig_grid) + args.synthetic_noise * rng.standard_normal(k2.shape)

    u = np.cos(theta)
    ramp = 0.25 * ((1 + np.tanh((X + 0.85 * L) / 3)) * (1 - np.tanh((X - 0.85 * L) / 3)))
    ramp *= 0.25 * ((1 + np.tanh((Y + 0.85 * L) / 3)) * (1 - np.tanh((Y - 0.85 * L) / 3)))
    return x, y, u, ramp, k1, k2


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def theory_lines_for(target_key, args):
    """Reference values to draw against discovered coefficients."""
    r = args.alpha / args.eta
    if target_key == "divk":
        # div k = -c w  (lower-GB sign); Hopf-Cole: c=1; RCN-weighted: sqrt(alpha/eta)
        return {"w": [(-1.0, "Hopf–Cole ($-1$)"),
                      (-np.sqrt(r), r"$-\sqrt{\alpha/\eta}$")]}
    else:
        return {"w^2": [(1.0, "Hopf–Cole ($+1$)"),
                        (r, r"$\alpha/\eta$")]}


def attenuation_reference(mu, sigma, seed=0):
    """
    Smoothing-attenuated coefficients from an ideal (lambda=1) Hopf-Cole
    knee passed through a Gaussian window of width sigma:
        div k = c_att * w,   (div k)^2 = c_att_sq * w^2.
    Gives an expectation for how far below |c|=1 the measured coefficient
    should sit purely because of macro-averaging in the extraction.
    """
    a = _Args(synthetic=True, mu=mu, synthetic_lambda=1.0,
              synthetic_noise=0.0, synthetic_sigma=sigma, seed=seed)
    x, y, u, ramp, k1, k2 = make_synthetic(a)
    rn = (ramp - ramp.min()) / (ramp.max() - ramp.min() + 1e-12)
    st = np.ones((3, 3), bool)
    m = binary_erosion(rn >= a.ramp_thresh_strict, st,
                       iterations=a.ramp_erosion_iters)
    dx = x[1] - x[0]
    lib, tg, sm, _ = build_library(k1, k2, dx, dx, m, mu)
    w = lib["w"][sm]
    w2 = lib["w^2"][sm]
    d = tg["divk"][sm]
    d2 = tg["divk_sq"][sm]
    c_att = float(np.dot(w, d) / np.dot(w, w))
    c_att_sq = float(np.dot(w2, d2) / np.dot(w2, w2))
    return c_att, c_att_sq


def fig_library_correlation(names, Theta, out):
    keep = np.std(Theta, axis=0) > 0          # drop constant columns
    names = [n for n, k in zip(names, keep) if k]
    Theta = Theta[:, keep]
    C = np.corrcoef(Theta, rowvar=False)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    labels = [f"${n}$" if n not in ("1",) else "1" for n in names]
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            if abs(C[i, j]) > 0.5 and i != j:
                ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="w" if abs(C[i, j]) > 0.8 else "k")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")
    ax.set_title("Library collinearity structure")
    fig.savefig(out)
    plt.close(fig)


def fig_pareto(res, target_key, out):
    sw = [s for s in res["sweep"] if np.isfinite(s["threshold"])]
    th = [s["threshold"] for s in sw]
    nt = [s["n_terms"] for s in sw]
    r2t = [s["r2_test"] for s in sw]

    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.2))
    ax = axs[0]
    ax.semilogx(th, r2t, "o-", ms=4, color="C0", label="test $R^2$")
    ax.set_xlabel("STLSQ threshold")
    ax.set_ylabel("holdout $R^2$")
    ax2 = ax.twinx()
    ax2.semilogx(th, nt, "s--", ms=4, color="C3", alpha=0.7)
    ax2.set_ylabel("terms retained", color="C3")
    ax2.tick_params(axis="y", colors="C3")
    if np.isfinite(res["chosen"]["threshold"]):
        ax.axvline(res["chosen"]["threshold"], color="k", lw=0.8, ls=":")
    panel_label(ax, "(a)")

    ax = axs[1]
    seen = {}
    for s in res["sweep"]:
        seen.setdefault(s["n_terms"], []).append(s["r2_test"])
    ks = sorted(seen)
    ax.plot(ks, [max(seen[k]) for k in ks], "o-", color="C0")
    ax.scatter([res["chosen"]["n_terms"]], [res["chosen"]["r2_test"]],
               s=120, facecolors="none", edgecolors="C3", lw=2, zorder=5,
               label="selected model")
    ax.set_xlabel("number of terms")
    ax.set_ylabel("best holdout $R^2$")
    ax.legend(loc="lower right")
    panel_label(ax, "(b)")

    fig.suptitle(f"Model selection, target: {target_key}", y=1.02)
    fig.savefig(out)
    plt.close(fig)


def fig_coefficients(names, res, theory, target_key, out):
    supp = res["support"]
    coefs = res["coef_full"][supp]
    labels = [names[i] for i in np.where(supp)[0]]
    boots = res["boots"]

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    xpos = np.arange(len(labels))
    err = boots.std(axis=0)
    ax.bar(xpos, coefs, yerr=err, capsize=4, color="C0", alpha=0.85, width=0.55)
    for i, lab in enumerate(labels):
        if lab in theory:
            for val, tname in theory[lab]:
                ax.axhline(val, color="C3", lw=1.0, ls="--", alpha=0.8)
                ax.text(len(labels) - 0.45, val, tname, fontsize=8,
                        va="bottom", ha="right", color="C3")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"${l}$" if l != "1" else "1" for l in labels])
    ax.set_ylabel("coefficient")
    ax.set_title(f"Identified model, target: {target_key} "
                 f"($R^2={res['r2_full']:.3f}$)")
    ax.axhline(0, color="k", lw=0.6)
    fig.savefig(out)
    plt.close(fig)


def fig_fit_and_residual(y2d, yhat2d, mask, extent, target_key, out):
    yv = y2d[mask]
    yh = yhat2d[mask]
    resid = np.where(mask, y2d - yhat2d, np.nan)

    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4))
    ax = axs[0]
    lim = np.nanpercentile(np.abs(yv), 99.5)
    ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.8)
    ax.scatter(yv[::7], yh[::7], s=2, alpha=0.25, color="C0", rasterized=True)
    ax.set_xlim(-lim * 0.05 if target_key == "divk_sq" else -lim, lim)
    ax.set_ylim(-lim * 0.05 if target_key == "divk_sq" else -lim, lim)
    ax.set_xlabel("measured")
    ax.set_ylabel("model")
    ax.set_aspect("equal")
    panel_label(ax, "(a)")

    ax = axs[1]
    rl = np.nanpercentile(np.abs(resid), 99)
    im = ax.imshow(resid, extent=extent, cmap="seismic", vmin=-rl, vmax=rl,
                   aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.85, label="residual")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    panel_label(ax, "(b)")

    fig.suptitle(f"Fit quality, target: {target_key}", y=1.02)
    fig.savefig(out)
    plt.close(fig)


def fig_bootstrap(names, res, theory, target_key, out):
    supp = res["support"]
    labels = [names[i] for i in np.where(supp)[0]]
    boots = res["boots"]

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.violinplot([boots[:, j] for j in range(boots.shape[1])],
                  positions=np.arange(len(labels)), showmedians=True)
    for i, lab in enumerate(labels):
        if lab in theory:
            for val, tname in theory[lab]:
                ax.axhline(val, color="C3", lw=1.0, ls="--", alpha=0.8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([f"${l}$" if l != "1" else "1" for l in labels])
    ax.set_ylabel("bootstrap coefficient")
    ax.set_title(f"Coefficient stability ({boots.shape[0]} resamples)")
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------
# Field maps (ported from knee_bend_fits_and_residuals.py, paper style)
# ---------------------------------------------------------------------

def _add_cbar(fig, ax, im, label=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.06)
    cb = fig.colorbar(im, cax=cax)
    if label:
        cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return cb


def _robust_lims(f, mask, signed, pct=99.0):
    v = np.asarray(f)[mask]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (-1, 1) if signed else (0, 1)
    if signed:
        a = np.nanpercentile(np.abs(v), pct)
        return -a, a
    return np.nanpercentile(v, 100 - pct), np.nanpercentile(v, pct)


def fig_fields_overview(out, u, k1, k2, aux, mask, extent, quiver_skip=16):
    """2x3 overview: u+k quiver, |k|, w, div k, curl k, J."""
    kmag = np.sqrt(aux["kmag2"])
    panels = [
        ("$u$ with $\\mathbf{k}$", u, "gray", True),
        ("$|\\mathbf{k}|$", kmag, "viridis", False),
        ("$w = |\\mathbf{k}|^2 - 1$", aux["w"], "seismic", True),
        ("$\\nabla\\!\\cdot\\!\\mathbf{k}$", aux["div_k"], "seismic", True),
        ("$\\nabla\\!\\times\\!\\mathbf{k}$", aux["curl_k"], "seismic", True),
        ("$J = \\det(\\nabla\\mathbf{k})$", aux["J"], "seismic", True),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 6.2), constrained_layout=True)
    for pl, ax, (title, f, cmap, signed) in zip(
            "abcdef", axs.flat, panels):
        if title.startswith("$u$"):
            im = ax.imshow(np.ma.masked_where(~mask, f), extent=extent,
                           cmap=cmap, vmin=-1, vmax=1, aspect="auto",
                           rasterized=True)
            ny, nx = u.shape
            xs = np.linspace(extent[0], extent[1], nx)
            ys = np.linspace(extent[2], extent[3], ny)
            Xg, Yg = np.meshgrid(xs, ys)
            s = quiver_skip
            sub = mask[::s, ::s]
            ax.quiver(Xg[::s, ::s][sub], Yg[::s, ::s][sub],
                      np.asarray(k1)[::s, ::s][sub],
                      np.asarray(k2)[::s, ::s][sub],
                      color="cyan", scale=35, width=0.003)
        else:
            v0, v1 = _robust_lims(f, mask, signed)
            im = ax.imshow(np.ma.masked_where(~mask, f), extent=extent,
                           cmap=cmap, vmin=v0, vmax=v1, aspect="auto",
                           rasterized=True)
        _add_cbar(fig, ax, im)
        ax.set_title(title)
        panel_label(ax, f"({pl})",
                    color="w" if cmap in ("gray",) else "k")
        if pl in "def":
            ax.set_xlabel("$x$")
        if pl in "ad":
            ax.set_ylabel("$y$")
    fig.savefig(out)
    plt.close(fig)


def fig_energy_balance(out, aux, mask, extent, alpha, eta):
    """
    eta (div k)^2 | alpha G^2 | global-scaled relative error.
    G^2 = w^2 - w^4 exactly (SH->RCN constitutive function in shifted form).
    Returns RMS of the global-scaled relative error (quotable number).
    """
    w = aux["w"]
    div_k = aux["div_k"]
    G2 = np.maximum(w ** 2 - w ** 4, 0.0)
    E_comp = eta * div_k ** 2
    E_bend = alpha * G2

    C = max(np.nanmean(E_comp[mask]), np.nanmean(E_bend[mask]))
    eps = 1e-12
    E_rel = np.abs(E_comp - E_bend) / (E_comp + E_bend + C + eps)
    rms_rel = float(np.sqrt(np.nanmean(E_rel[mask] ** 2)))

    v1 = max(np.nanpercentile(E_comp[mask], 99.5),
             np.nanpercentile(E_bend[mask], 99.5))
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.2), constrained_layout=True)
    specs = [
        (r"$\eta\,(\nabla\!\cdot\!\mathbf{k})^2$", E_comp, "magma", (0, v1)),
        (r"$\alpha\,G^2$", E_bend, "magma", (0, v1)),
        (r"scaled rel.\ error (RMS $= %.3f$)" % rms_rel, E_rel,
         "viridis", (0, np.nanpercentile(E_rel[mask], 99.5))),
    ]
    for pl, ax, (title, f, cmap, (v0, vmax)) in zip("abc", axs, specs):
        im = ax.imshow(np.ma.masked_where(~mask, f), extent=extent,
                       cmap=cmap, vmin=v0, vmax=vmax, aspect="auto",
                       rasterized=True)
        _add_cbar(fig, ax, im)
        ax.set_title(title)
        ax.set_xlabel("$x$")
        panel_label(ax, f"({pl})", color="w")
    axs[0].set_ylabel("$y$")
    fig.savefig(out)
    plt.close(fig)
    return rms_rel


def fig_sd_residual_map(out, aux, mask, extent, coef_w):
    """Map of div k - c_fit * w : the identified-model residual in 2D."""
    resid = aux["div_k"] - coef_w * aux["w"]
    v = np.nanpercentile(np.abs(resid[mask]), 99.0)
    fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
    im = ax.imshow(np.ma.masked_where(~mask, resid), extent=extent,
                   cmap="seismic", vmin=-v, vmax=v, aspect="auto",
                   rasterized=True)
    _add_cbar(fig, ax, im, label="residual")
    ax.set_title(r"$\nabla\!\cdot\!\mathbf{k} - c\,w$, $c=%.3f$" % coef_w)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    fig.savefig(out)
    plt.close(fig)


def fig_constitutive(out, aux, mask, args, c_divk, c_divk_sq, c_att, c_att_sq):
    """
    Identified RHS vs theoretical RHS in function space.
    (a) div k vs w : samples + identified line c*w + Hopf-Cole (-w)
        + RCN curve sgn*sqrt(alpha/eta)*G(w) + smoothing-attenuated line.
    (b) (div k)^2 vs w^2 : same comparison for the energy form.
    """
    w = aux["w"][mask]
    d = aux["div_k"][mask]
    r = args.alpha / args.eta
    sgn = np.sign(np.nanmean(d)) or 1.0

    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.6), constrained_layout=True)

    # ---- (a) self-dual form ----
    ax = axs[0]
    step = max(1, w.size // 4000)
    ax.scatter(w[::step], d[::step], s=3, alpha=0.25, color="0.55",
               rasterized=True, label="data")
    wg = np.linspace(min(w.min(), 0), max(w.max(), 0), 400)
    G = np.sqrt(np.maximum(wg ** 2 - wg ** 4, 0.0))
    ax.plot(wg, c_divk * wg, "-", color="C0", lw=1.8,
            label=rf"identified: ${c_divk:.3f}\,w$")
    ax.plot(wg, -wg, "--", color="C3", lw=1.2, label=r"Hopf–Cole: $-w$")
    ax.plot(wg, sgn * np.sqrt(r) * G, "-.", color="C2", lw=1.2,
            label=r"RCN: $\pm\sqrt{\alpha/\eta}\,G(w)$")
    if c_att is not None:
        ax.plot(wg, c_att * wg, ":", color="C1", lw=1.4,
                label=rf"smoothed HC: ${c_att:.2f}\,w$")
    ax.set_xlabel(r"$w = |\mathbf{k}|^2 - 1$")
    ax.set_ylabel(r"$\nabla\!\cdot\!\mathbf{k}$")
    ax.legend(loc="best", fontsize=7)
    panel_label(ax, "(a)")

    # ---- (b) energy form ----
    ax = axs[1]
    w2 = w ** 2
    d2 = d ** 2
    ax.scatter(w2[::step], d2[::step], s=3, alpha=0.25, color="0.55",
               rasterized=True, label="data")
    sg = np.linspace(0, w2.max(), 400)
    ax.plot(sg, c_divk_sq * sg, "-", color="C0", lw=1.8,
            label=rf"identified: ${c_divk_sq:.3f}\,w^2$")
    ax.plot(sg, sg, "--", color="C3", lw=1.2, label=r"Hopf–Cole: $w^2$")
    ax.plot(sg, r * (sg - sg ** 2), "-.", color="C2", lw=1.2,
            label=r"RCN: $(\alpha/\eta)(w^2 - w^4)$")
    if c_att_sq is not None:
        ax.plot(sg, c_att_sq * sg, ":", color="C1", lw=1.4,
                label=rf"smoothed HC: ${c_att_sq:.2f}\,w^2$")
    ax.set_xlabel(r"$w^2$")
    ax.set_ylabel(r"$(\nabla\!\cdot\!\mathbf{k})^2$")
    ax.legend(loc="best", fontsize=7)
    panel_label(ax, "(b)")

    fig.savefig(out)
    plt.close(fig)


def fig_rhs_profile(out, aux, mask, y_c, args, c_divk, c_att,
                    min_row_frac=0.25, pad_rows=12):
    """
    Transverse profile through the crease: x-averaged measured LHS
    div k(y) against the identified RHS c*w(y) and theoretical RHS
    curves, restricted to rows where the crease carries signal.
    """
    r = args.alpha / args.eta

    def row_mean(f):
        fm = np.where(mask, f, np.nan)
        cnt = mask.sum(axis=1)
        with np.errstate(invalid="ignore"):
            m = np.nanmean(fm, axis=1)
        m[cnt < min_row_frac * cnt.max()] = np.nan
        return m

    d_row = row_mean(aux["div_k"])
    w_row = row_mean(aux["w"])
    sgn = np.sign(np.nanmean(aux["div_k"][mask])) or 1.0
    G_row = sgn * np.sqrt(r) * np.sqrt(
        np.maximum(w_row ** 2 - w_row ** 4, 0.0))

    good = np.isfinite(d_row)
    sig = np.abs(d_row) > 0.02 * np.nanmax(np.abs(d_row))
    idx = np.where(good & sig)[0]
    if idx.size == 0:
        return
    j0 = max(idx.min() - pad_rows, 0)
    j1 = min(idx.max() + pad_rows, len(y_c) - 1)
    sl = slice(j0, j1 + 1)

    fig, axs = plt.subplots(
        2, 1, figsize=(6.4, 4.6), sharex=True, constrained_layout=True,
        height_ratios=[3, 1])

    ax = axs[0]
    ax.plot(y_c[sl], d_row[sl], "o", ms=3, color="0.35",
            label=r"measured $\langle\nabla\!\cdot\!\mathbf{k}\rangle_x$")
    ax.plot(y_c[sl], c_divk * w_row[sl], "-", color="C0", lw=1.8,
            label=rf"identified ${c_divk:.3f}\,w$")
    ax.plot(y_c[sl], -w_row[sl], "--", color="C3", lw=1.2,
            label=r"Hopf–Cole $-w$")
    ax.plot(y_c[sl], G_row[sl], "-.", color="C2", lw=1.2,
            label=r"RCN $\pm\sqrt{\alpha/\eta}\,G$")
    if c_att is not None:
        ax.plot(y_c[sl], c_att * w_row[sl], ":", color="C1", lw=1.4,
                label=rf"smoothed HC ${c_att:.2f}\,w$")
    ax.set_ylabel(r"$\nabla\!\cdot\!\mathbf{k}$")
    ax.legend(loc="best", fontsize=7, ncol=2)
    panel_label(ax, "(a)")

    ax = axs[1]
    ax.axhline(0, color="k", lw=0.6)
    ax.plot(y_c[sl], d_row[sl] - c_divk * w_row[sl], "-", color="C0",
            lw=1.2, label="identified")
    ax.plot(y_c[sl], d_row[sl] - G_row[sl], "-.", color="C2", lw=1.0,
            label="RCN")
    ax.set_xlabel("$y$")
    ax.set_ylabel("residual")
    ax.legend(loc="best", fontsize=7)
    panel_label(ax, "(b)")

    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(args: _Args):
    mpl.rcParams.update(PAPER_RC)
    rng = np.random.default_rng(args.seed)

    # ---- load or synthesize ----
    if args.synthetic:
        x, y, u, ramp, k1_raw, k2_raw = make_synthetic(args)
        stem = f"synthetic_mu{args.mu}_lam{args.synthetic_lambda}"
        already_oriented = True
    else:
        op = np.load(args.op_path)
        x, y = op["x"], op["y"]
        u = op["u"][..., args.t_index]
        ramp = op["ramp"]
        k1_raw = op[args.k1_key][..., args.t_index]
        k2_raw = op[args.k2_key][..., args.t_index]
        stem = Path(args.op_path).stem
        already_oriented = False

    out_root = Path(args.out_root) if args.out_root else (
        Path(__file__).resolve().parent / "results" / "self_dual_sindy_v2" / stem)
    out_root.mkdir(parents=True, exist_ok=True)
    print("Output:", out_root)

    x0, y0 = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
    x_c, y_c = x - x0, y - y0
    dx, dy = x_c[1] - x_c[0], y_c[1] - y_c[0]
    extent = [x_c.min(), x_c.max(), y_c.min(), y_c.max()]
    Xc, Yc = np.meshgrid(x_c, y_c)

    # ---- masks (repo conventions) ----
    rn = (ramp - np.nanmin(ramp)) / (np.nanmax(ramp) - np.nanmin(ramp) + 1e-12)
    st = np.ones((3, 3), bool)
    m_bfs = binary_erosion(rn >= args.ramp_thresh_bfs, st,
                           iterations=args.ramp_erosion_iters)
    m_strict = binary_erosion(rn >= args.ramp_thresh_strict, st,
                              iterations=args.ramp_erosion_iters)

    if already_oriented:
        k1o, k2o = k1_raw, k2_raw
    else:
        k1o, k2o = orient_vector_field(k1_raw, k2_raw, m_bfs)

    phi = np.arctan2(k2o, k1o)
    valid = m_strict & (~phi_jump_mask(phi, args.phi_jump_tol))
    print("Valid points:", int(valid.sum()))

    # ---- library + targets ----
    lib, targets, sample_mask, aux = build_library(k1o, k2o, dx, dy, valid, args.mu)
    names = list(lib.keys())
    N = int(sample_mask.sum())
    print("Regression samples:", N)

    Theta = np.column_stack([lib[nm][sample_mask] for nm in names])
    X_train = Xc[sample_mask] < 0.0   # spatial holdout: left half train

    weights = None
    summary = {"n_samples": N, "targets": {}}

    fig_library_correlation(names, Theta, out_root / "library_correlation.png")

    # smoothing-attenuation reference at this (mu, sigma)
    c_att, c_att_sq = attenuation_reference(args.mu, args.synthetic_sigma,
                                            seed=args.seed)
    print(f"Attenuation reference (mu={args.mu}, sigma={args.synthetic_sigma:.3f}): "
          f"divk {c_att:+.4f}, divk_sq {c_att_sq:+.4f}")
    summary["attenuation_reference"] = dict(
        mu=args.mu, sigma=float(args.synthetic_sigma),
        c_divk=c_att, c_divk_sq=c_att_sq)

    for tkey, tfield in targets.items():
        yvec = tfield[sample_mask]
        if args.weight_by_target:
            wts = np.sqrt(np.abs(yvec) + 1e-12)
            Th_w, y_w = Theta * wts[:, None], yvec * wts
        else:
            Th_w, y_w = Theta, yvec

        res = fit_target(names, Th_w, y_w, args, rng, X_train)
        theory = theory_lines_for(tkey, args)
        if tkey == "divk":
            theory.setdefault("w", []).append(
                (c_att, r"smoothed HC ($%+.2f$)" % c_att))
        else:
            theory.setdefault("w^2", []).append(
                (c_att_sq, r"smoothed HC ($%+.2f$)" % c_att_sq))

        # figures
        fig_pareto(res, tkey, out_root / f"pareto_{tkey}.png")
        fig_coefficients(names, res, theory, tkey,
                         out_root / f"coefficients_{tkey}.png")
        fig_bootstrap(names, res, theory, tkey,
                      out_root / f"bootstrap_{tkey}.png")

        yhat2d = np.full_like(tfield, np.nan)
        yhat2d[sample_mask] = Theta @ res["coef_full"]
        fig_fit_and_residual(tfield, yhat2d, sample_mask, extent, tkey,
                             out_root / f"fit_residual_{tkey}.png")

        sel = {names[i]: float(res["coef_full"][i])
               for i in np.where(res["support"])[0]}
        errs = {names[i]: float(res["boots"][:, j].std())
                for j, i in enumerate(np.where(res["support"])[0])}
        print(f"\n[{tkey}] selected: " +
              ", ".join(f"{k}: {v:+.4f} ± {errs[k]:.4f}" for k, v in sel.items()))
        print(f"[{tkey}] R^2 (full) = {res['r2_full']:.4f}, "
              f"holdout R^2 = {res['chosen']['r2_test']:.4f}")

        summary["targets"][tkey] = dict(
            selected=sel, boot_std=errs,
            r2_full=float(res["r2_full"]),
            r2_holdout=float(res["chosen"]["r2_test"]),
            threshold=float(res["chosen"]["threshold"]),
        )

    if args.make_field_maps:
        fig_fields_overview(out_root / "fields_overview.png",
                            u, k1o, k2o, aux, sample_mask, extent)
        rms_rel = fig_energy_balance(out_root / "energy_balance.png",
                                     aux, sample_mask, extent,
                                     args.alpha, args.eta)
        summary["energy_balance_rms_rel_error"] = rms_rel
        cw = summary["targets"]["divk"]["selected"].get("w")
        cw2 = summary["targets"]["divk_sq"]["selected"].get("w^2")
        if cw is not None:
            fig_sd_residual_map(out_root / "sd_residual_map.png",
                                aux, sample_mask, extent, cw)
        if cw is not None and cw2 is not None:
            fig_constitutive(out_root / "constitutive_rhs.png",
                             aux, sample_mask, args, cw, cw2,
                             c_att, c_att_sq)
        if cw is not None:
            fig_rhs_profile(out_root / "rhs_profile.png",
                            aux, sample_mask, y_c, args, cw, c_att)

    with open(out_root / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nWrote", out_root / "summary.json")
    return summary


def parse_cli() -> _Args:
    d = _Args()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--op_path", default=d.op_path)
    p.add_argument("--k1_key", default=d.k1_key)
    p.add_argument("--k2_key", default=d.k2_key)
    p.add_argument("--t_index", type=int, default=d.t_index)
    p.add_argument("--mu", type=float, default=d.mu)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--synthetic_lambda", type=float, default=d.synthetic_lambda)
    p.add_argument("--synthetic_noise", type=float, default=d.synthetic_noise)
    p.add_argument("--synthetic_sigma", type=float, default=d.synthetic_sigma)
    p.add_argument("--out_root", default=d.out_root)
    p.add_argument("--weight_by_target", action="store_true")
    p.add_argument("--n_boot", type=int, default=d.n_boot)
    a = p.parse_args()
    out = _Args(**{**d.__dict__, **{k: v for k, v in vars(a).items()}})
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(parse_cli())
    else:
        main(_Args())