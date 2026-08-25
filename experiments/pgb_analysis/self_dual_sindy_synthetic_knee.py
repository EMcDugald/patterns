
"""
self_dual_sindy_synthetic_knee.py
---------------------------------
End-to-end synthetic control for the self-dual identification (the
control anticipated in main_v6's \\todo in subsec:knee_sindy, extended
to sharp knees and the auto-gated adaptive average).

Build an EXACT knee pattern with a PLANTED balance coefficient, push it
through the actual pattern-level uHu extraction, and rerun the STLSQ
identification, comparing three tiers:

  T0 analytic  library evaluated on the closed-form wavevector field
               (no extraction): the identification floor set by finite
               differences, masking, and the library alone.
  T1 iso       ordinary isotropic uHu extraction at sigma = pi/2.
  T2 auto      auto-gated adaptive extraction: the same
               compute_uhu_ops_smoother call with the smoother replaced
               by the patchwise auto-gated anisotropic average
               (all guidance fields from one isotropic seed pass;
               single pass, non-circular; mirrors
               adaptive_auto_gate_patched.run_case stage for stage).

Planted field
-------------
    theta_lam = (1/lam) log( exp(lam(k1 x + k2 y)) + exp(lam(k1 x - k2 y)) )
              = k1 x + (1/lam) log( 2 cosh(lam k2 y) )
with k1 = sqrt(1 - mu^2), k2 = mu, satisfies

    div k = lam (1 - |k|^2) = -lam w        (w = |k|^2 - 1)

EXACTLY, for every mu, with |k| -> 1 in the far field and
K_core = 1 - mu^2. NOTE: the plain Hopf-Cole knee (lam = 1) plants
coefficient 1, NOT sqrt(alpha/eta); the 1/lam rescaling above is what
plants a general lam.

The pattern is  u = A1(x, y) cos(theta_lam)  with the SH-consistent
amplitude  A1^2 = (4/3)(R - (1 - K)^2)  at the local K. Existence
requires R > mu^4 at the core; the default R = 1.0 keeps a positive
margin at mu = 0.9 (margin 0.34). A1 varies on the crease scale, so its
neglected derivatives enter the extraction exactly as on real SH data;
that is the point of the control. Third-harmonic content is off by
default (a separate axis, tested in knee_pgb_extraction_demo.py).

Everything downstream (orientation, pi-jump masking, library, STLSQ,
spatial holdout, bootstrap, figures) is imported unchanged from
self_dual_sindy.py, so the identification machinery is literally the
one used on SH data.

Reported per (mu, lam) config
-----------------------------
  * identified c_w (divk target) and c_w2 (divk_sq target) per tier,
    with bootstrap errors, against the planted -lam and lam^2
  * holdout and full R^2 per tier
  * censoring diagnostic: fraction of the core band (|y| < 2 sigma)
    removed by the pi-jump mask, per tier. The isotropic eigen-flip at
    mu > mu* shows up here as heavy core censoring; the auto tier
    should not censor the core of a defect-free knee.
  * tier_comparison.png, per-tier fit/residual and coefficient figures,
    summary.json

Usage
-----
    python self_dual_sindy_synthetic_knee.py                  # _Args below
    python self_dual_sindy_synthetic_knee.py --mus 0.9 --lams 0.8660254
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

# --- path shims: siblings in experiments/pgb_analysis, plus repo
# --- pipelines/ (auto-gate machinery) and src/ (utils, op_extract)
_HERE = Path(__file__).resolve().parent
for _p in (_HERE,
           _HERE.parents[1] / "pipelines",
           _HERE.parents[1] / "src",
           _HERE / "src"):
    if _p.exists():
        sys.path.insert(0, str(_p))

from self_dual_sindy import (
    PAPER_RC, panel_label,
    orient_vector_field, phi_jump_mask, build_library, fit_target,
    fig_library_correlation, fig_pareto, fig_coefficients,
    fig_bootstrap, fig_fit_and_residual,
)

try:
    from utils.spectral import macro, SpectralDerivs
except ImportError:
    from spectral import macro, SpectralDerivs
try:
    from utils.geometry import build_rectangular_ramp_smooth
except ImportError:
    from geometry import build_rectangular_ramp_smooth
try:
    from op_extract.uhu_smoother import compute_uhu_ops_smoother
except ImportError:
    from uhu_smoother import compute_uhu_ops_smoother
try:
    from utils.anisotropic_macro import coherence_from_eigs
except ImportError:
    from anisotropic_macro import coherence_from_eigs
try:
    from op_extract.uhu_adaptive import (slowness_director,
                                         derived_kernel_params)
except ImportError:
    from uhu_adaptive import slowness_director, derived_kernel_params

from adaptive_auto_gate_patched import (
    core_path, circulation_profile, line_score_from_circulation,
    riesz_gate_field, macro_aniso_field_auto,
)


# ---------------------------------------------------------------------
# Planted-coefficient knee field
# ---------------------------------------------------------------------

def field_knee_planted(X, Y, mu, lam, R, amp_mode="unit"):
    """
    theta = (1/lam) logaddexp(lam(k1 x + k2 y), lam(k1 x - k2 y)).
    Closed-form truth: k = (k1, k2 tanh(lam k2 y)),
    div k = lam k2^2 sech^2(lam k2 y) = -lam w exactly.

    amp_mode = "unit": u = cos(theta). The corrugation and the
        eigen-flip are phase-structure effects, so this isolates the
        averaging question with no amplitude confound. R is unused.
    amp_mode = "sh": u = A1 cos(theta) with the SH-consistent closure
        A1^2 = (4/3)(R - (1 - K)^2) at the local K. Requires
        R > mu^4 for a positive core amplitude (R = 1.0 gives margin
        0.34 at mu = 0.9); at the simulation value R = 0.5 no
        stationary amplitude exists on a mu = 0.9 core and the
        constructor clips to zero there.
    """
    k1c = np.sqrt(1.0 - mu ** 2)
    k2c = mu
    theta = np.logaddexp(lam * (k1c * X + k2c * Y),
                         lam * (k1c * X - k2c * Y)) / lam
    k1_true = np.full_like(X, k1c)
    k2_true = k2c * np.tanh(lam * k2c * Y)
    K = k1c ** 2 + k2_true ** 2

    if amp_mode == "sh":
        A1_sq = (4.0 / 3.0) * (R - (1.0 - K) ** 2)
        frac_invalid = float(np.mean(A1_sq < 0))
        A1 = np.sqrt(np.clip(A1_sq, 0.0, None))
    else:
        A1 = np.ones_like(K)
        frac_invalid = 0.0
    u = A1 * np.cos(theta)
    return dict(u=u, theta=theta, k1_true=k1_true, k2_true=k2_true,
                K=K, A1=A1, frac_invalid=frac_invalid)


# ---------------------------------------------------------------------
# Auto-gated adaptive smoother (mirrors adaptive_auto_gate_patched.
# run_case stages: seed pass -> gate -> kernel params -> patch average)
# ---------------------------------------------------------------------

def build_auto_smoother(u, x, y, ramp, mu, cfg):
    sigma = cfg.sigma
    iso_sm = lambda a: macro(a, sigma, cfg.Lx, cfg.Ly)
    seed = compute_uhu_ops_smoother(u, x, y, iso_sm, SpectralDerivs, ramp)
    coh = coherence_from_eigs(seed["lam1"], seed["lam2"])
    d1, d2 = slowness_director(seed["k1_orig"], seed["k2_orig"], sigma,
                               cfg.Lx, cfg.Ly, SpectralDerivs, macro)
    interior = ramp >= 0.99

    k1c = np.sqrt(1.0 - mu ** 2)
    L = np.pi / k1c
    dx = cfg.Lx / cfg.Nx
    half_w = max(4, int(round(cfg.gate_window_L * L / dx / 2)))

    rows_c, _ = core_path(coh, y, band_frac=0.35)
    rows_s = np.round(np.convolve(
        rows_c, np.ones(2 * half_w + 1) / (2 * half_w + 1),
        mode="same")).astype(int)

    use_riesz = L > cfg.L_over_sigdir_min * cfg.sigma_dir
    if use_riesz:
        g1, g2 = riesz_gate_field(u, cfg.Lx, cfg.Ly, cfg.sigma_dir, macro)
        ur = u * ramp
        Txx = iso_sm(u * SpectralDerivs(ur, cfg.Lx, cfg.Ly, "xx"))
        Txy = iso_sm(u * SpectralDerivs(ur, cfg.Lx, cfg.Ly, "xy"))
        Tyy = iso_sm(u * SpectralDerivs(ur, cfg.Lx, cfg.Ly, "yy"))
        uu = iso_sm(u * u * ramp)
        uu = np.where(np.abs(uu) < 1e-12, 1.0, uu)
        q = np.sqrt(np.maximum(
            -(g1 * g1 * Txx + 2 * g1 * g2 * Txy + g2 * g2 * Tyy) / uu, 0.0))
        C = circulation_profile(q * g1, q * g2, x, y, rows_s, L)
        gate_used = "riesz_director_proj"
    else:
        gate_sm = lambda a: macro(a, cfg.sigma_gate, cfg.Lx, cfg.Ly)
        gseed = compute_uhu_ops_smoother(u, x, y, gate_sm,
                                         SpectralDerivs, ramp)
        C = circulation_profile(gseed["k1_orig"], gseed["k2_orig"],
                                x, y, rows_s, L)
        gate_used = "tensor"
    ls = line_score_from_circulation(C, cfg.circ_lo, cfg.circ_hi)

    cmin = float(coh[interior].min())
    cmed = float(np.median(coh[interior]))
    cth = cmin + cfg.core_coh_frac * (cmed - cmin)
    Ygrid = np.meshgrid(x, y)[1]
    y_core = y[rows_s][None, :]
    band = np.abs(Ygrid - y_core) < cfg.core_band_sigma * sigma
    core_mask = (coh < cth) & band
    ls2d = np.where(band, ls[None, :], 0.0)

    aspect, sharpen = derived_kernel_params(k1c, mu, sigma,
                                            delta=cfg.leak_delta)

    def auto_sm(arr):
        return macro_aniso_field_auto(
            arr, d1, d2, seed["k1_orig"], seed["k2_orig"], coh,
            core_mask, ls2d, cfg.Lx, cfg.Ly, sigma, aspect, sharpen,
            patch=cfg.patch, ramp_mask=interior,
            sharpen_weight=cfg.sharpen_weight)

    ctr = np.abs(x - 0.5 * (x[0] + x[-1])) < 0.3 * cfg.Lx
    diag = dict(gate_used=gate_used,
                aspect=float(aspect), sharpen=float(sharpen),
                C_med=float(np.median(np.abs(C[ctr]))),
                ls_med=float(np.median(ls[ctr])))
    return auto_sm, diag


# ---------------------------------------------------------------------
# Per-tier identification
# ---------------------------------------------------------------------

def identify_tier(tier_name, k1_raw, k2_raw, oriented, masks, grids,
                  mu, lam, cfg, rng, out_dir, save_full_figs):
    m_bfs, m_strict, trim, band = masks
    X, dx, dy = grids

    if oriented:
        k1o, k2o = np.asarray(k1_raw, float), np.asarray(k2_raw, float)
        jump = np.zeros_like(m_strict)
    else:
        k1o, k2o = orient_vector_field(k1_raw, k2_raw, m_bfs)
        k1o, k2o = np.asarray(k1o), np.asarray(k2o)
        # Canonical global sign: the BFS seed pixel is arbitrary, and
        # under k -> -k the divergence flips while w is invariant, so
        # the fitted coefficient flips sign with the orientation draw.
        # Fix the representative with k1 > 0 (knee geometry: the true
        # k1 is a positive constant).
        if np.nanmedian(k1o[m_bfs]) < 0:
            k1o, k2o = -k1o, -k2o
        phi = np.arctan2(k2o, k1o)
        jump = phi_jump_mask(phi, cfg.phi_jump_tol)

    valid = m_strict & trim & (~jump)
    denom = m_strict & trim & band
    cens_band = float(np.mean(jump[denom])) if denom.any() else np.nan

    lib, targets, sm, aux = build_library(k1o, k2o, dx, dy, valid, mu)
    names = list(lib.keys())
    n_s = int(sm.sum())

    # midline cross-sections (median over the central x window) for the
    # attenuation diagnostic figure; NaN-safe without warnings
    ctr = np.abs(X) < 0.25 * cfg.Lx

    def _prof(f2d):
        v = np.where(ctr & sm, f2d, np.nan)
        out_p = np.full(v.shape[0], np.nan)
        ok = np.isfinite(v).any(axis=1)
        out_p[ok] = np.nanmedian(v[ok], axis=1)
        return out_p

    profiles = dict(divk=_prof(aux["div_k"]), w=_prof(aux["w"]))
    Theta = np.column_stack([lib[nm][sm] for nm in names])
    X_train = X[sm] < 0.0

    theory = {
        "divk": {"w": [(-lam, r"$-\lambda_{\rm planted}$")]},
        "divk_sq": {"w^2": [(lam ** 2, r"$\lambda_{\rm planted}^2$")]},
    }
    lead = {"divk": "w", "divk_sq": "w^2"}

    row = dict(tier=tier_name, n_samples=n_s,
               core_band_censored=cens_band, targets={})
    row["_profiles"] = profiles
    for tkey in ("divk", "divk_sq"):
        yvec = targets[tkey][sm]
        res = fit_target(names, Theta, yvec, cfg, rng, X_train)
        sel_idx = np.where(res["support"])[0]
        sel = {names[i]: float(res["coef_full"][i]) for i in sel_idx}
        errs = {names[i]: float(res["boots"][:, j].std())
                for j, i in enumerate(sel_idx)}
        ld = lead[tkey]
        row["targets"][tkey] = dict(
            selected=sel, boot_std=errs,
            c_lead=sel.get(ld, np.nan),
            c_lead_std=errs.get(ld, np.nan),
            extra_terms=[nm for nm in sel if nm != ld],
            r2_full=float(res["r2_full"]),
            r2_holdout=float(res["chosen"]["r2_test"]))

        if save_full_figs:
            fig_pareto(res, tkey, out_dir / f"pareto_{tkey}_{tier_name}.png")
            fig_coefficients(names, res, theory[tkey], tkey,
                             out_dir / f"coefficients_{tkey}_{tier_name}.png")
            fig_bootstrap(names, res, theory[tkey], tkey,
                          out_dir / f"bootstrap_{tkey}_{tier_name}.png")
        if tkey == "divk":
            yhat2d = np.full_like(targets[tkey], np.nan)
            yhat2d[sm] = Theta @ res["coef_full"]
            row["_divk_pair"] = (targets[tkey], yhat2d, sm)

        t = row["targets"][tkey]
        print(f"  [{tier_name:8s}] {tkey:8s} N={n_s:6d} "
              f"cens_band={cens_band:5.1%}  "
              f"c={t['c_lead']:+.4f} ± {t['c_lead_std']:.4f}  "
              f"(planted {(-lam if tkey == 'divk' else lam**2):+.4f})  "
              f"R2={t['r2_full']:.4f}/{t['r2_holdout']:.4f}  "
              f"extra={t['extra_terms']}")
    return row, (k1o, k2o)


def fig_tier_comparison(out, rows, lam, mu):
    tiers = [r["tier"] for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2), constrained_layout=True)
    for ax, tkey, ref, lab in (
            (axs[0], "divk", -lam, r"planted $-\lambda$"),
            (axs[1], "divk_sq", lam ** 2, r"planted $\lambda^2$")):
        c = [r["targets"][tkey]["c_lead"] for r in rows]
        e = [r["targets"][tkey]["c_lead_std"] for r in rows]
        xpos = np.arange(len(tiers))
        ax.errorbar(xpos, c, yerr=e, fmt="o", ms=6, capsize=4, color="C0")
        for xp, r in zip(xpos, rows):
            if r["targets"][tkey]["extra_terms"]:
                ax.plot([xp], [c[list(xpos).index(xp)]], "s", ms=12,
                        mfc="none", mec="C3", mew=1.5)
        ax.axhline(ref, color="C3", ls="--", lw=1.1, label=lab)
        ax.set_xticks(xpos)
        ax.set_xticklabels(tiers)
        ax.set_ylabel("leading coefficient")
        ax.legend(loc="best")
    axs[0].set_title(r"$\nabla\!\cdot\!\mathbf{k} = c\,w$")
    axs[1].set_title(r"$(\nabla\!\cdot\!\mathbf{k})^2 = c\,w^2$")
    fig.suptitle(rf"$\mu={mu}$, $\lambda_{{\rm planted}}={lam:.4f}$"
                 " (open squares: extra terms selected)", y=1.06)
    fig.savefig(out)
    plt.close(fig)


def fig_midline_profiles(out, y, profs, mu, lam, sigma, sharpen):
    """
    Midline cross-sections of div k and w per tier against the exact
    profiles and Gaussian-smoothed reference curves, plus the pointwise
    coefficient -div k / w. The reference curves are INDICATIVE only:
    the extraction is not literally a Gaussian applied to the k-field
    (it averages quadratic pattern-level quantities), so agreement or
    disagreement with them is the diagnostic, not an identity.
    Panel (c) shows whether a pointwise law survives smoothing with a
    renormalized coefficient (flat curve) or is broken (y-dependent).
    """
    from scipy.ndimage import gaussian_filter1d
    k2c = mu
    dy = y[1] - y[0]
    div_true = lam * k2c ** 2 / np.cosh(lam * k2c * y) ** 2
    w_true = -div_true / lam
    refs = [(sigma, "--", r"$G_\sigma\!*$truth")]
    if sharpen > 1.05:
        refs.append((sigma / sharpen, ":", r"$G_{\sigma/\rm sh}\!*$truth"))

    colors = {"analytic": "0.3", "iso": "C0", "auto": "C1"}
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.2),
                            constrained_layout=True)

    for ax, key, lab, pl in ((axs[0], "divk",
                              r"$\nabla\!\cdot\!\mathbf{k}$", "(a)"),
                             (axs[1], "w",
                              r"$w=|\mathbf{k}|^2-1$", "(b)")):
        tru = div_true if key == "divk" else w_true
        ax.plot(y, tru, "k-", lw=1.4, label="truth")
        for s_ref, ls, rlab in refs:
            ax.plot(y, gaussian_filter1d(tru, s_ref / dy), color="0.55",
                    ls=ls, lw=1.1, label=rlab)
        for tname, pf in profs.items():
            if tname == "analytic":
                continue
            ax.plot(y, pf[key], color=colors.get(tname, "C2"),
                    lw=1.2, label=tname)
        ax.set_xlim(-8.0, 8.0)
        ax.set_xlabel("$y$")
        ax.set_ylabel(lab)
        ax.legend(fontsize=7, loc="best")
        panel_label(ax, pl)

    ax = axs[2]
    for tname, pf in profs.items():
        wv = pf["w"]
        ok = np.abs(wv) > 0.02
        ratio = np.where(ok, -pf["divk"] / np.where(ok, wv, 1.0), np.nan)
        ax.plot(y, ratio, color=colors.get(tname, "C2"), lw=1.2,
                label=tname)
    ax.axhline(lam, color="C3", ls="--", lw=1.1,
               label=r"$\lambda_{\rm planted}$")
    ax.set_xlim(-8.0, 8.0)
    ax.set_ylim(0.0, 1.6 * lam)
    ax.set_xlabel("$y$")
    ax.set_ylabel(r"$-\nabla\!\cdot\!\mathbf{k}\,/\,w$")
    ax.legend(fontsize=7, loc="best")
    panel_label(ax, "(c)")

    fig.suptitle(rf"$\mu={mu}$, $\lambda_{{\rm planted}}={lam:.3f}$:"
                 " midline cross-sections")
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------
# Per-config driver
# ---------------------------------------------------------------------

def run_config(cfg, mu, lam, rng, out_root):
    x = np.linspace(-cfg.Lx / 2, cfg.Lx / 2, cfg.Nx, endpoint=False)
    y = np.linspace(-cfg.Ly / 2, cfg.Ly / 2, cfg.Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)
    dx, dy = x[1] - x[0], y[1] - y[0]
    extent = [x.min(), x.max(), y.min(), y.max()]

    field = field_knee_planted(X, Y, mu, lam, cfg.R, amp_mode=cfg.amp_mode)
    u = field["u"]
    print(f"\n=== mu={mu}  lam_planted={lam:.4f}  amp_mode={cfg.amp_mode}"
          + (f"  R={cfg.R}" if cfg.amp_mode == "sh" else "") + " ===")
    if field["frac_invalid"] > 0:
        print(f"  WARNING: A1^2 < 0 on {field['frac_invalid']:.2%} of the "
              f"domain (R too small for this mu); amplitudes clipped.")

    ramp = build_rectangular_ramp_smooth(
        x, y, xmargin=cfg.xmargin, ymargin=cfg.ymargin,
        tanhscale=cfg.tanhscale)
    rn = (ramp - ramp.min()) / (ramp.max() - ramp.min() + 1e-12)
    st = np.ones((3, 3), bool)
    m_bfs = binary_erosion(rn >= cfg.ramp_thresh_bfs, st,
                           iterations=cfg.ramp_erosion_iters)
    m_strict = binary_erosion(rn >= cfg.ramp_thresh_strict, st,
                              iterations=cfg.ramp_erosion_iters)
    trim = (X > x.min() + cfg.x_trim) & (X < x.max() - cfg.x_trim)
    band = np.abs(Y) < 2.0 * cfg.sigma
    masks = (m_bfs, m_strict, trim, band)
    grids = (X, dx, dy)

    out_dir = out_root / f"mu{mu:.2f}_lam{lam:.4f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # tier fields
    iso_sm = lambda a: macro(a, cfg.sigma, cfg.Lx, cfg.Ly)
    res_iso = compute_uhu_ops_smoother(u, x, y, iso_sm,
                                       SpectralDerivs, ramp)
    auto_sm, diag = build_auto_smoother(u, x, y, ramp, mu, cfg)
    print(f"  gate: {diag['gate_used']}  |C| med={diag['C_med']:.3f}  "
          f"ls med={diag['ls_med']:.3f}  aspect={diag['aspect']:.2f}  "
          f"sharpen={diag['sharpen']:.2f}")
    res_auto = compute_uhu_ops_smoother(u, x, y, auto_sm,
                                        SpectralDerivs, ramp)

    tiers = [
        ("analytic", field["k1_true"], field["k2_true"], True),
        ("iso", res_iso["k1_orig"], res_iso["k2_orig"], False),
        ("auto", res_auto["k1_orig"], res_auto["k2_orig"], False),
    ]

    rows = []
    profs = {}
    for tname, k1_raw, k2_raw, oriented in tiers:
        row, _ = identify_tier(
            tname, k1_raw, k2_raw, oriented, masks, grids,
            mu, lam, cfg, rng, out_dir,
            save_full_figs=cfg.save_full_figs)
        y2d, yhat2d, sm = row.pop("_divk_pair")
        fig_fit_and_residual(y2d, yhat2d, sm, extent, "divk",
                             out_dir / f"fit_residual_divk_{tname}.png")
        profs[tname] = row.pop("_profiles")
        rows.append(row)

    fig_tier_comparison(out_dir / "tier_comparison.png", rows, lam, mu)
    fig_midline_profiles(out_dir / "midline_profiles.png", y, profs,
                         mu, lam, cfg.sigma, diag["sharpen"])
    summary = dict(mu=mu, lam_planted=lam, R=cfg.R,
                   frac_A1_invalid=field["frac_invalid"],
                   gate=diag, tiers=rows)
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main(cfg):
    mpl.rcParams.update(PAPER_RC)
    rng = np.random.default_rng(cfg.seed)
    out_root = Path(cfg.out) if cfg.out else (
        _HERE / "results" / "self_dual_sindy_synthetic_knee_823")
    out_root.mkdir(parents=True, exist_ok=True)
    print("Output:", out_root)

    all_summaries = []
    for mu in cfg.mus:
        for lam in cfg.lams:
            all_summaries.append(run_config(cfg, mu, lam, rng, out_root))
    with open(out_root / "summary_all.json", "w") as fh:
        json.dump(all_summaries, fh, indent=2)
    print("\nWrote", out_root / "summary_all.json")


def parse_cli(d):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=d.out)
    p.add_argument("--mus", type=float, nargs="+", default=list(d.mus))
    p.add_argument("--lams", type=float, nargs="+", default=list(d.lams))
    p.add_argument("--R", type=float, default=d.R)
    p.add_argument("--amp_mode", default=d.amp_mode, choices=["unit", "sh"])
    p.add_argument("--Nx", type=int, default=d.Nx)
    p.add_argument("--Ny", type=int, default=d.Ny)
    p.add_argument("--sigma", type=float, default=d.sigma)
    p.add_argument("--leak_delta", type=float, default=d.leak_delta)
    p.add_argument("--sharpen_weight", default=d.sharpen_weight,
                   choices=["coh", "ls"])
    a = p.parse_args()
    for k, v in vars(a).items():
        setattr(d, k, v)
    d.mus, d.lams = tuple(d.mus), tuple(d.lams)
    return d


if __name__ == "__main__":
    class _Args:
        out = ""
        # config grid: shallow control + sharp case, planted RCN
        # coefficient + Hopf-Cole contrast
        mus = (0.5, 0.9)
        lams = (np.sqrt(3.0) / 2.0, 1.0)   # sqrt(alpha/eta) = sqrt(3)/2
        amp_mode = "unit"  # "unit": u = cos(theta); "sh": SH amplitude
        R = 1.0            # sh mode only; existence needs R > mu^4
                           # (margin 0.34 at mu = 0.9); the simulation
                           # R = 0.5 cannot dress a stationary mu = 0.9
                           # core and clips to zero there

        # grid / extraction (matches adaptive_auto_gate_patched defaults)
        Nx, Ny = 512, 256
        Lx, Ly = 40 * np.pi, 20 * np.pi
        sigma = np.pi / 2

        # auto-gate knobs (mirror adaptive_auto_gate_patched)
        patch = 96
        leak_delta = 0.05        # CLI default of the auto-gate script;
                                 # its PyCharm _Args uses 0.02 -- pick ONE
                                 # config and quote from that run only
        gate_window_L = 2.0
        sigma_gate = np.pi / 8
        sigma_dir = np.pi / 4
        L_over_sigdir_min = 8.5
        circ_lo, circ_hi = 0.25, 0.5
        core_coh_frac = 0.5
        core_band_sigma = 2.0
        sharpen_weight = "coh"   # "ls" = topological gate alone
                                 # authorizes cross-core sharpening

        # ramp / masking (self_dual_sindy family conventions; erosion
        # reduced from 48 since this grid is coarser than the SH npz)
        xmargin, ymargin = 0.05, 0.10
        tanhscale = 60.0
        ramp_thresh_bfs = 0.98
        ramp_thresh_strict = 0.999
        ramp_erosion_iters = 24
        x_trim = 8.0
        phi_jump_tol = np.pi / 10.0

        # STLSQ (identical to self_dual_sindy defaults)
        thresholds = tuple(np.geomspace(1e-4, 1.0, 25))
        n_boot = 200
        boot_frac = 0.5
        rel_tol_pareto = 0.01
        weight_by_target = False
        seed = 0

        save_full_figs = True

    if len(sys.argv) > 1:
        main(parse_cli(_Args()))
    else:
        main(_Args())