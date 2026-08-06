"""
self_dual_pismen_probes.py
--------------------------
Probes for the structural issues in the mu = 0.95 multi-dislocation
(PN) SINDy fits of self_dual_pismen.py. Four probes, each designed to
VALIDATE or RULE OUT a specific mechanism, plus the fit the mechanisms
say should work.

Probe 1: IDENTITY NULL TEST.
    For any field, pointwise and exactly (also on discrete central
    differences, since the identity is algebraic in the entries of
    grad k):

        (div k)^2 = |grad k|^2 - (curl k)^2 + 2 J.

    The production library contains J and (curl k)^2 but NOT
    |grad k|^2, so the regression can partially reconstruct its own
    target kinematically. This probe (a) fits (div k)^2 by plain OLS
    on exactly {|grad k|^2, (curl k)^2, J} and checks coefficients
    (+1, -1, +2) with R^2 = 1 to roundoff, and (b) adds |grad k|^2 to
    the full library and runs STLSQ, documenting that the regression
    then returns the tautology. If (a) holds, the contamination
    mechanism is proven and J / (curl k)^2 cannot be read as physics
    in any (div k)^2 fit.

Probe 2: IDENTITY-SAFE BAND SWEEP.
    Rerun the core-exclusion band sweep with the identity terms
    {J, (curl k)^2, curl k} removed (library: 1, w, w^2, w^3, k_2,
    k_1 k_2, |grad w|), alongside the full library on identical
    samples. Outputs c_{w^2}(b_in) and holdout R^2(b_in) for both.
    The safe R^2 is the honest explanatory power of a w-family
    balance; the gap to the full-library R^2 is the tautological
    share. If the safe c plateaus with clean selection past some
    b_in, THAT plateau (not the full-library one) is the candidate
    physical coefficient.

Probe 3: BRANCH / QUADRATURE DIAGNOSTIC for the signed form.
    The signed self-dual law div k = +/- sqrt(alpha/eta) G(w) carries
    a branch sign. Hypotheses for the observed R^2 ~ 0.05 of every
    linear-form fit on the chain geometry:
      (H1) antiphase: div k and w are locally proportional with an
           alternating sign that a single global coefficient cannot
           carry -> a per-cell sign (or |div k| vs |w|) fit works;
      (H2) quadrature: div k and w oscillate a quarter cell apart
           along the boundary -> NO pointwise sign convention can
           work; the linear form is structurally dead and only the
           branch-blind energy form is identifiable.
    The probe demodulates div k and w row by row at the boundary
    wavenumber q (estimated from the div k spectrum) and reports the
    relative phase and amplitude ratio vs y, plus a rectified OLS
    |div k| ~ {|w|, |w|^3}. Relative phase near 0 or pi supports H1;
    near +/- pi/2 proves H2.

Probe 4: ANALYTIC PN BASELINE (tier 0; handoff inclusion criterion i).
    Build the analytic PN multi-dislocation phase (erf superposition,
    same construction as pismen_fit_and_residuals_v2.pismen_on_sh_grid)
    with uniformly spaced defects, take k = grad theta by the same
    central differences as the data path (no extraction), and run the
    identical safe-library band sweep. The resulting c(b_in) plateau
    is the value the EXTRACTED sweep should be compared against; it
    isolates estimator-plus-geometry behavior from extraction and
    from SH physics. NOTE the normalization caveat: the analytic PN
    solution lives in the unit-coefficient phase-equation frame, so
    its plateau need not equal the SH knee value alpha/eta; the
    comparison of interest is extracted-vs-analytic, not either-vs-
    alpha/eta.

Run modes:
    python self_dual_pismen_probes.py                  # _Args (data mode)
    python self_dual_pismen_probes.py --analytic_only  # probe 4 + null tests
                                                       #   on the analytic
                                                       #   field; no npz needed
    python self_dual_pismen_probes.py --op_path ...    # CLI, data mode

Imports the shared machinery from self_dual_sindy (build_library,
fit_target, masking, paper style), mirroring self_dual_pismen.py.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import special
from scipy.ndimage import binary_erosion

from self_dual_sindy import (
    PAPER_RC, panel_label,
    orient_vector_field, phi_jump_mask, masked_central_derivs,
    build_library, fit_target,
)

IDENTITY_DROP = ("J", "(curl k)^2", "curl k")


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class _Args:
    # --- input (data mode) ---
    op_path: str = (
        "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/phase/"
        "mu_sweeps_full_Nx512_hp2_T25_NyF5_np18_Nsave125/sig_pio2/raw/"
        "sh_pgb_zigzag_mu0.950_T25_N125_nx512_Ny256_lower_uhu_sigma1.571_"
        "xm0.03_ym0.03_ts120.0_phase_xg0.10_yg0.10_ns128_shlower_ds0.125_"
        "prmrebuild_prc0.100_prs1.00_prt0.050.npz"
    )
    k1_key: str = "k1_orig"
    k2_key: str = "k2_orig"
    t_index: int = -1
    mu: float = 0.95

    # --- band geometry (data mode; centered coordinates) ---
    y_core: float = 0.0
    band_out: float = 4.0 * np.pi
    band_in_sweep: tuple = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
    x_trim: float = 8.0

    # --- masking (phase-file conventions, as in self_dual_pismen) ---
    ramp_thresh_bfs: float = 0.995
    ramp_thresh_strict: float = 0.999
    ramp_erosion_iters: int = 10
    phi_jump_tol: float = np.pi / 10.0

    # --- analytic baseline (probe 4; also --analytic_only mode) ---
    analytic_only: bool = False
    run_analytic: bool = True       # run probe 4 alongside data mode
    an_nx: int = 768
    an_ny: int = 384
    an_n_defects: int = 7           # uniformly spaced, centered
    an_spacing: float = 0.0         # 0 -> pi / sqrt(1 - mu^2)  (data spacing)
    an_pad_x: float = 1.0           # extra half-spacings beyond outer defects
    an_ly_factor: float = 2.2       # Ly = factor * band_out

    # --- quadrature probe ---
    quad_y_min: float = 1.0         # rows |y| in [quad_y_min, band_out]
    quad_min_frac: float = 0.9      # min valid fraction of the x-span per row

    # --- theory references ---
    alpha: float = 2.0 / 3.0
    eta: float = 8.0 / 9.0

    # --- regression (as in self_dual_pismen) ---
    thresholds: tuple = tuple(np.geomspace(1e-4, 1.0, 25))
    n_boot: int = 100
    boot_frac: float = 0.5
    rel_tol_pareto: float = 0.01
    weight_by_target: bool = False
    seed: int = 0

    out_root: str = ""


# ---------------------------------------------------------------------
# Library helpers
# ---------------------------------------------------------------------

def grad_k_sq(k1, k2, dx, dy, mask):
    """|grad k|^2 from the same masked central differences as the
    library, so the discrete identity holds exactly."""
    k1x, k1y = masked_central_derivs(k1, dx, dy, mask)
    k2x, k2y = masked_central_derivs(k2, dx, dy, mask)
    return k1x ** 2 + k1y ** 2 + k2x ** 2 + k2y ** 2


def split_library(lib):
    """(full_names, safe_names) with the identity terms removed from
    the safe set."""
    full = list(lib.keys())
    safe = [nm for nm in full if nm not in IDENTITY_DROP]
    return full, safe


# ---------------------------------------------------------------------
# Probe 1: identity null test
# ---------------------------------------------------------------------

def probe_identity(lib, targets, gksq, sample_mask, args, rng, Xc, out_png):
    """(a) OLS of (div k)^2 on exactly {|grad k|^2, (curl k)^2, J};
    (b) STLSQ on the full library augmented with |grad k|^2."""
    sm = sample_mask & np.isfinite(gksq)
    y = targets["divk_sq"][sm]

    B = np.column_stack([gksq[sm], lib["(curl k)^2"][sm], lib["J"][sm]])
    coef, *_ = np.linalg.lstsq(B, y, rcond=None)
    yhat = B @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    max_res = float(np.max(np.abs(y - yhat)))
    report = dict(
        ols_coeffs={"|grad k|^2": float(coef[0]),
                    "(curl k)^2": float(coef[1]),
                    "J": float(coef[2])},
        ols_theory={"|grad k|^2": 1.0, "(curl k)^2": -1.0, "J": 2.0},
        ols_r2=float(r2), ols_max_residual=max_res,
        target_rms=float(np.sqrt(np.mean(y ** 2))),
        n_samples=int(sm.sum()),
    )
    print("\n[probe 1] identity null test on {|grad k|^2, (curl k)^2, J}:")
    print(f"  coefficients = ({coef[0]:+.6f}, {coef[1]:+.6f}, {coef[2]:+.6f})"
          "   theory (+1, -1, +2)")
    print(f"  R^2 = {r2:.12f},  max |residual| = {max_res:.3e}"
          f"  (target rms {report['target_rms']:.3e})")

    # (b) full library + |grad k|^2 through the production STLSQ
    names_aug = list(lib.keys()) + ["|grad k|^2"]
    Theta = np.column_stack([lib[nm][sm] for nm in lib] + [gksq[sm]])
    res = fit_target(names_aug, Theta, y, args, rng, Xc[sm] < 0.0)
    sel = {names_aug[i]: float(res["coef_full"][i])
           for i in np.where(res["support"])[0]}
    report["stlsq_augmented"] = dict(
        selected=sel, r2_full=float(res["r2_full"]),
        r2_holdout=float(res["chosen"]["r2_test"]))
    print("  STLSQ with |grad k|^2 added to the full library:")
    print(f"    R^2 = {res['r2_full']:.6f} (holdout "
          f"{res['chosen']['r2_test']:.6f}); selected: "
          + ", ".join(f"{k}: {v:+.3f}" for k, v in sel.items()))

    # figure
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.4), constrained_layout=True)
    ax = axs[0]
    step = max(1, y.size // 4000)
    ax.scatter(y[::step], yhat[::step], s=3, alpha=0.3, color="0.5",
               rasterized=True)
    lo, hi = float(y.min()), float(y.max())
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.9)
    ax.set_xlabel(r"measured $(\nabla\!\cdot\!\mathbf{k})^2$")
    ax.set_ylabel(r"$|\nabla\mathbf{k}|^2 - (\nabla\!\times\!\mathbf{k})^2"
                  r" + 2J$ (OLS)")
    ax.text(0.03, 0.95, rf"$R^2 = {r2:.10f}$", transform=ax.transAxes,
            va="top", fontsize=8,
            bbox=dict(fc="white", ec="0.7", alpha=0.9, pad=2.5))
    panel_label(ax, "(a)")

    ax = axs[1]
    labs = [r"$|\nabla\mathbf{k}|^2$", r"$(\nabla\!\times\!\mathbf{k})^2$",
            "$J$"]
    xpos = np.arange(3)
    ax.bar(xpos, coef, width=0.5, color="C0", label="OLS")
    ax.plot(xpos, [1.0, -1.0, 2.0], "_", ms=26, mew=2.2, color="C3",
            label="identity (+1, -1, +2)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("coefficient")
    ax.legend(fontsize=7, loc="upper left")
    panel_label(ax, "(b)")
    fig.suptitle(r"Identity null test: $(\nabla\!\cdot\!\mathbf{k})^2 ="
                 r" |\nabla\mathbf{k}|^2 - (\nabla\!\times\!\mathbf{k})^2"
                 r" + 2J$ pointwise", y=1.04, fontsize=10)
    fig.savefig(out_png)
    plt.close(fig)
    return report


# ---------------------------------------------------------------------
# Probe 2: band sweep with full vs identity-safe library
# ---------------------------------------------------------------------

def band_sweep(lib, names, targets, dist, base_mask, bands, args, rng, Xc,
               tag):
    rows = []
    for b_in in bands:
        sm = base_mask & (dist > b_in)
        n_s = int(sm.sum())
        if n_s < 500:
            print(f"  [{tag}] band_in={b_in}: too few samples ({n_s}), skip")
            continue
        Theta = np.column_stack([lib[nm][sm] for nm in names])
        yvec = targets["divk_sq"][sm]
        res = fit_target(names, Theta, yvec, args, rng, Xc[sm] < 0.0)
        sel_idx = np.where(res["support"])[0]
        coeffs = {names[i]: float(res["coef_full"][i]) for i in sel_idx}
        ystd = float(yvec.std()) + 1e-15
        shares = {names[i]: float(res["coef_full"][i]
                                  * Theta[:, i].std() / ystd)
                  for i in sel_idx}
        if "w^2" in coeffs:
            jb = [names[i] for i in sel_idx].index("w^2")
            cw2_std = float(res["boots"][:, jb].std())
        else:
            cw2_std = np.nan
        rows.append(dict(band_in=float(b_in), n_samples=n_s, coeffs=coeffs,
                         shares=shares, coef_w2_std=cw2_std,
                         n_terms=int(res["support"].sum()),
                         r2_holdout=float(res["chosen"]["r2_test"]),
                         r2_full=float(res["r2_full"])))
        cw2 = coeffs.get("w^2", np.nan)
        print(f"  [{tag}] band_in={b_in:4.1f}: N={n_s:6d}  "
              f"R2={res['r2_full']:.4f}  c_w2={cw2:+.3f}  "
              f"terms={int(res['support'].sum())}")
    return rows


def fig_library_comparison(out, rows_full, rows_safe, safe_names, args,
                           spacing, title_tag):
    fig = plt.figure(figsize=(10.8, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25])

    ax = fig.add_subplot(gs[0, 0])
    for rows, lab, col, mk in ((rows_full, "full library", "0.45", "s"),
                               (rows_safe, "identity-safe", "C0", "o")):
        b = [r["band_in"] for r in rows]
        c = [r["coeffs"].get("w^2", np.nan) for r in rows]
        e = [r["coef_w2_std"] for r in rows]
        ax.errorbar(b, c, yerr=e, fmt=mk + "-", ms=4, color=col, capsize=3,
                    label=lab)
    ax.axhline(args.alpha / args.eta, color="C3", lw=1.0, ls=":",
               label=r"knee $\alpha/\eta$ (reference)")
    ax.set_ylabel(r"$c$ in $c\,w^2$")
    ax.legend(fontsize=7, loc="best")
    ax.tick_params(labelbottom=False)
    panel_label(ax, "(a)")

    ax = fig.add_subplot(gs[1, 0])
    for rows, lab, col, mk in ((rows_full, "full", "0.45", "s"),
                               (rows_safe, "safe", "C0", "o")):
        b = [r["band_in"] for r in rows]
        ax.plot(b, [r["r2_holdout"] for r in rows], mk + "-", ms=4,
                color=col, label=lab)
    ax.set_xlabel(r"core exclusion half-width $b_{\rm in}$"
                  + rf"   (defect spacing $L = {spacing:.2f}$)")
    ax.set_ylabel(r"holdout $R^2$")
    ax.legend(fontsize=7, loc="lower right")
    panel_label(ax, "(b)")

    # safe-library activation matrix
    ax = fig.add_subplot(gs[:, 1])
    b = [r["band_in"] for r in rows_safe]
    order = [t for t in ("w^2", "w", "w^3", "k_2", "k_1 k_2", "|grad w|",
                         "1") if t in safe_names]
    M = np.full((len(order), len(b)), np.nan)
    for jc, r in enumerate(rows_safe):
        for t, v in r["shares"].items():
            if t in order:
                M[order.index(t), jc] = v
    C = np.ma.masked_invalid(np.minimum(np.abs(M), 1.2))
    ax.pcolormesh(np.arange(len(b) + 1), np.arange(len(order) + 1), C,
                  cmap="Blues", vmin=0, vmax=1.8, edgecolors="0.85", lw=0.4)
    for i in range(len(order)):
        for jc in range(len(b)):
            if np.isfinite(M[i, jc]):
                ax.text(jc + 0.5, i + 0.5, f"{M[i, jc]:+.2f}", ha="center",
                        va="center", fontsize=6,
                        color="white" if abs(M[i, jc]) > 0.75 else "black")
    ax.set_xticks(np.arange(len(b)) + 0.5)
    ax.set_xticklabels([f"{bi:g}" for bi in b], fontsize=7)
    ax.set_yticks(np.arange(len(order)) + 0.5)
    tex = {"w^2": "$w^2$", "w": "$w$", "w^3": "$w^3$", "k_2": "$k_2$",
           "k_1 k_2": "$k_1 k_2$", "|grad w|": r"$|\nabla w|$", "1": "$1$"}
    ax.set_yticklabels([tex[t] for t in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"$b_{\rm in}$")
    ax.set_title("identity-safe library: signed contribution shares",
                 fontsize=9)
    panel_label(ax, "(c)")

    fig.suptitle(title_tag, y=1.03)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------
# Probe 3: branch / quadrature diagnostic
# ---------------------------------------------------------------------

def probe_quadrature(aux, sample_mask, x_c, y_c, dist, args, out_png):
    div_k, w = aux["div_k"], aux["w"]
    ny, nx = div_k.shape
    dx = x_c[1] - x_c[0]

    xin = (x_c > x_c.min() + args.x_trim) & (x_c < x_c.max() - args.x_trim)
    span = np.where(xin)[0]
    xs = x_c[span]
    n_span = span.size

    # rows in the annulus with sufficient valid coverage of the span
    rows = []
    for iy in range(ny):
        if not (args.quad_y_min <= abs(y_c[iy] - 0.0) <= args.band_out):
            continue
        ok = sample_mask[iy, span] & np.isfinite(div_k[iy, span]) \
            & np.isfinite(w[iy, span])
        if ok.mean() >= args.quad_min_frac:
            rows.append((iy, ok))
    if not rows:
        print("[probe 3] no usable rows; skipping")
        return None

    # boundary wavenumber q from the mean div k spectrum over these rows
    spec = np.zeros(n_span // 2 + 1)
    for iy, ok in rows:
        f = np.where(ok, div_k[iy, span], 0.0)
        f = f - f[ok].mean()
        spec += np.abs(np.fft.rfft(f))
    freqs = np.fft.rfftfreq(n_span, d=dx) * 2 * np.pi
    ipk = 1 + int(np.argmax(spec[1:]))
    q = float(freqs[ipk])

    # per-row complex demodulation at q
    ys, ratio, relphase, wamp = [], [], [], []
    for iy, ok in rows:
        e = np.exp(-1j * q * xs[ok])
        fd = div_k[iy, span][ok]
        fw = w[iy, span][ok]
        D = np.mean((fd - fd.mean()) * e)
        W = np.mean((fw - fw.mean()) * e)
        if abs(W) < 1e-14 or abs(D) < 1e-14:
            continue
        ys.append(y_c[iy])
        ratio.append(abs(D) / abs(W))
        relphase.append(np.angle(D * np.conj(W)))
        wamp.append(abs(W))
    ys = np.array(ys)
    ratio = np.array(ratio)
    relphase = np.array(relphase)
    wamp = np.array(wamp)

    # amplitude-weighted circular mean of the relative phase; computed
    # per half-plane, since a symmetric boundary flips the relative
    # phase by pi across y = 0 and the global mean is then meaningless
    def circ(sel):
        if not np.any(sel):
            return np.nan, 0.0
        z = np.sum(wamp[sel] * np.exp(1j * relphase[sel])) / max(
            np.sum(wamp[sel]), 1e-300)
        return float(np.angle(z)), float(np.abs(z))

    ph_pos, cc_pos = circ(ys > 0)
    ph_neg, cc_neg = circ(ys < 0)
    # verdict from the better-concentrated half
    phase_mean, phase_conc = ((ph_pos, cc_pos) if cc_pos >= cc_neg
                              else (ph_neg, cc_neg))

    # rectified linear-level fit: |div k| ~ c1 |w| + c3 |w|^3
    sm = sample_mask & (dist > args.quad_y_min) & (dist < args.band_out)
    ad = np.abs(div_k[sm])
    aw = np.abs(w[sm])
    B = np.column_stack([aw, aw ** 3])
    coef, *_ = np.linalg.lstsq(B, ad, rcond=None)
    r2_rect = 1.0 - np.sum((ad - B @ coef) ** 2) / np.sum(
        (ad - ad.mean()) ** 2)

    report = dict(q=q, wavelength=float(2 * np.pi / q),
                  phase_mean=phase_mean, phase_concentration=phase_conc,
                  phase_upper=ph_pos, conc_upper=cc_pos,
                  phase_lower=ph_neg, conc_lower=cc_neg,
                  ratio_median=float(np.median(ratio)),
                  rectified=dict(c_absw=float(coef[0]),
                                 c_absw3=float(coef[1]),
                                 r2=float(r2_rect)),
                  n_rows=int(ys.size))
    verdict = ("QUADRATURE (H2): pointwise signed law structurally dead"
               if min(abs(abs(phase_mean) - np.pi / 2),
                      abs(abs(phase_mean) - 3 * np.pi / 2))
               < min(abs(phase_mean), abs(abs(phase_mean) - np.pi))
               else "ANTIPHASE/IN-PHASE (H1): sign structure, magnitude law"
               " may survive")
    report["verdict"] = verdict
    print(f"\n[probe 3] demodulation at q = {q:.4f} "
          f"(wavelength {2 * np.pi / q:.2f}):")
    print(f"  relative phase, upper half: {ph_pos / np.pi:+.3f} pi "
          f"(conc {cc_pos:.2f});  lower half: {ph_neg / np.pi:+.3f} pi "
          f"(conc {cc_neg:.2f})")
    print(f"  amplitude ratio |D|/|W| median = {np.median(ratio):.3f}")
    print(f"  rectified |div k| ~ {coef[0]:+.3f}|w| {coef[1]:+.3f}|w|^3,"
          f"  R^2 = {r2_rect:.3f}")
    print(f"  verdict: {verdict}")

    fig, axs = plt.subplots(1, 3, figsize=(11.2, 3.3),
                            constrained_layout=True)
    ax = axs[0]
    ax.plot(ys, relphase / np.pi, ".", ms=3, color="C0")
    for v, lab in ((0.5, r"$+\pi/2$"), (-0.5, r"$-\pi/2$"),
                   (0.0, "0"), (1.0, r"$\pi$"), (-1.0, r"$-\pi$")):
        ax.axhline(v, color="0.8", lw=0.7, zorder=0)
    for ph, cc, lab in ((ph_pos, cc_pos, "upper"),
                        (ph_neg, cc_neg, "lower")):
        if np.isfinite(ph):
            ax.axhline(ph / np.pi, color="C3", lw=1.1, ls="--",
                       label=f"{lab} mean {ph / np.pi:+.2f}"
                             r"$\,\pi$" + f" (c={cc:.2f})")
    ax.set_xlabel("$y$")
    ax.set_ylabel(r"$\arg(D\,\overline{W})/\pi$")
    ax.set_ylim(-1.15, 1.15)
    ax.legend(fontsize=7)
    panel_label(ax, "(a)")

    ax = axs[1]
    ax.plot(ys, ratio, ".", ms=3, color="C0")
    ax.axhline(np.median(ratio), color="C3", lw=1.0, ls="--",
               label=f"median {np.median(ratio):.2f}")
    ax.set_xlabel("$y$")
    ax.set_ylabel(r"$|D|/|W|$ at $q$")
    ax.legend(fontsize=7)
    panel_label(ax, "(b)")

    ax = axs[2]
    d = div_k[sm]
    wv = aux["w"][sm]
    step = max(1, d.size // 6000)
    ax.scatter(wv[::step], d[::step], s=3, alpha=0.25, c=np.sign(
        np.meshgrid(x_c, y_c)[1][sm][::step]), cmap="coolwarm",
        rasterized=True)
    ax.set_xlabel("$w$")
    ax.set_ylabel(r"$\nabla\!\cdot\!\mathbf{k}$")
    ax.set_title("colored by sign($y$)", fontsize=8)
    panel_label(ax, "(c)")
    fig.suptitle("Branch / quadrature diagnostic at the boundary "
                 "wavenumber", y=1.05, fontsize=10)
    fig.savefig(out_png)
    plt.close(fig)
    return report


# ---------------------------------------------------------------------
# Probe 4: analytic PN baseline
# ---------------------------------------------------------------------

def pismen_theta(X, Y, x_defects):
    """Analytic PN multi-dislocation phase (erf superposition), same
    construction as pismen_fit_and_residuals_v2.pismen_on_sh_grid."""
    linear_phase = Y
    x_defects = np.asarray(x_defects, dtype=float)
    npts = len(x_defects)
    if npts == 0:
        return linear_phase.copy()
    thetas = np.arange(npts + 1) * np.pi
    psi = 0.5 * (1 + np.exp(-thetas[-1]))
    for j in range(npts):
        delta = 0.5 * (np.exp(-thetas[j + 1]) - np.exp(-thetas[j]))
        arg = (X - x_defects[j]) / np.sqrt(2 * np.abs(Y) + 1e-8)
        psi += delta * special.erf(arg)
    return linear_phase - np.sign(Y) * np.log(psi)


def make_analytic_chain(args):
    """Grid, k = grad theta by central differences, all-valid ramp."""
    d = args.an_spacing if args.an_spacing > 0 else (
        np.pi / np.sqrt(1.0 - args.mu ** 2))
    n = args.an_n_defects
    x_defects = (np.arange(n) - (n - 1) / 2.0) * d
    Lx = (n - 1 + 2 * args.an_pad_x) * d
    Ly = args.an_ly_factor * args.band_out * 2
    x = np.linspace(-Lx / 2, Lx / 2, args.an_nx)
    y = np.linspace(-Ly / 2, Ly / 2, args.an_ny)
    X, Y = np.meshgrid(x, y)
    theta = pismen_theta(X, Y, x_defects)
    k1 = np.gradient(theta, x, axis=1)
    k2 = np.gradient(theta, y, axis=0)
    return x, y, k1, k2, d, x_defects


# ---------------------------------------------------------------------
# Shared per-field pipeline
# ---------------------------------------------------------------------

def run_probes_on_field(tag, x, y, k1o, k2o, valid, args, rng, out_root,
                        spacing, summary):
    """Library, probes 1-3, and the full-vs-safe band sweep on one
    oriented field with its validity mask."""
    x0, y0 = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
    x_c, y_c = x - x0, y - y0
    dx, dy = x_c[1] - x_c[0], y_c[1] - y_c[0]
    Xc, Yc = np.meshgrid(x_c, y_c)

    lib, targets, sample_mask, aux = build_library(
        k1o, k2o, dx, dy, valid, args.mu)
    full_names, safe_names = split_library(lib)
    dist = np.abs(Yc - args.y_core)
    annulus = sample_mask & (dist < args.band_out)
    print(f"\n===== {tag}: {int(annulus.sum())} annulus samples, "
          f"spacing L = {spacing:.3f} =====")

    gksq = grad_k_sq(k1o, k2o, dx, dy, valid)
    summary[tag] = {}
    summary[tag]["probe1_identity"] = probe_identity(
        lib, targets, gksq, annulus, args, rng, Xc,
        out_root / f"identity_null_{tag}.png")

    print(f"\n[probe 2] band sweep, full library ({tag}):")
    rows_full = band_sweep(lib, full_names, targets, dist, annulus,
                           args.band_in_sweep, args, rng, Xc, "full")
    print(f"[probe 2] band sweep, identity-safe library ({tag}):")
    rows_safe = band_sweep(lib, safe_names, targets, dist, annulus,
                           args.band_in_sweep, args, rng, Xc, "safe")
    fig_library_comparison(
        out_root / f"library_comparison_{tag}.png", rows_full, rows_safe,
        safe_names, args, spacing,
        rf"$\mu = {args.mu}$ ({tag}): energy form, full vs identity-safe"
        " library")
    summary[tag]["sweep_full"] = rows_full
    summary[tag]["sweep_safe"] = rows_safe

    summary[tag]["probe3_quadrature"] = probe_quadrature(
        aux, annulus, x_c, y_c, dist, args,
        out_root / f"branch_quadrature_{tag}.png")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(args: _Args):
    mpl.rcParams.update(PAPER_RC)
    rng = np.random.default_rng(args.seed)

    out_root = Path(args.out_root) if args.out_root else (
        Path(__file__).resolve().parent / "results"
        / "self_dual_pismen_probes"
        / ("analytic_only" if args.analytic_only
           else Path(args.op_path).stem))
    out_root.mkdir(parents=True, exist_ok=True)
    print("Output:", out_root)

    summary = {}

    # ---------------- data mode ----------------
    if not args.analytic_only:
        op = np.load(args.op_path)
        x, y = op["x"], op["y"]
        ramp = op["ramp"]
        k1_raw = op[args.k1_key][..., args.t_index]
        k2_raw = op[args.k2_key][..., args.t_index]

        x0, y0 = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
        Xc, Yc = np.meshgrid(x - x0, y - y0)
        rn = (ramp - np.nanmin(ramp)) / (
            np.nanmax(ramp) - np.nanmin(ramp) + 1e-12)
        st = np.ones((3, 3), bool)
        m_bfs = binary_erosion(rn >= args.ramp_thresh_bfs, st,
                               iterations=args.ramp_erosion_iters)
        m_strict = binary_erosion(rn >= args.ramp_thresh_strict, st,
                                  iterations=args.ramp_erosion_iters)
        k1o, k2o = orient_vector_field(k1_raw, k2_raw, m_bfs)
        k1o, k2o = np.asarray(k1o), np.asarray(k2o)
        phi = np.arctan2(k2o, k1o)
        valid = m_strict & (~phi_jump_mask(phi, args.phi_jump_tol))
        valid &= (Xc > Xc.min() + args.x_trim) & (Xc < Xc.max() - args.x_trim)

        L_data = np.pi / np.sqrt(1.0 - args.mu ** 2)
        run_probes_on_field("data", x, y, k1o, k2o, valid, args, rng,
                            out_root, L_data, summary)

    # ---------------- analytic baseline (probe 4) ----------------
    if args.analytic_only or args.run_analytic:
        xa, ya, k1a, k2a, d, x_defects = make_analytic_chain(args)
        Xa, Ya = np.meshgrid(xa - 0.5 * (xa.min() + xa.max()),
                             ya - 0.5 * (ya.min() + ya.max()))
        phi_a = np.arctan2(k2a, k1a)
        valid_a = ~phi_jump_mask(phi_a, args.phi_jump_tol)
        valid_a &= (Xa > Xa.min() + args.x_trim) \
            & (Xa < Xa.max() - args.x_trim)
        valid_a &= np.isfinite(k1a) & np.isfinite(k2a)
        summary["analytic_meta"] = dict(
            spacing=float(d), n_defects=len(x_defects),
            note="unit-coefficient PN normalization; compare plateau"
                 " against the DATA plateau, not against alpha/eta")
        run_probes_on_field("analytic", xa, ya, k1a, k2a, valid_a, args,
                            rng, out_root, d, summary)

    with open(out_root / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
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
    p.add_argument("--analytic_only", action="store_true")
    p.add_argument("--no_analytic", action="store_true")
    p.add_argument("--an_spacing", type=float, default=d.an_spacing)
    p.add_argument("--an_n_defects", type=int, default=d.an_n_defects)
    p.add_argument("--n_boot", type=int, default=d.n_boot)
    p.add_argument("--out_root", default=d.out_root)
    a = p.parse_args()
    kw = {k: v for k, v in vars(a).items() if k != "no_analytic"}
    out = _Args(**{**d.__dict__, **kw})
    if a.no_analytic:
        out.run_analytic = False
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(parse_cli())
    else:
        main(_Args())