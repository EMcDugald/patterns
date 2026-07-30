"""
self_dual_sindy_pismen_v2.py
----------------------------
Self-dual identification on a DISLOCATION-CHAIN (PN/zipper) state, with a
core-exclusion band sweep and a THREE-REGIME presentation.

v2 changes (vs. self_dual_sindy_pismen.py):
- Imports from self_dual_sindy_v3 (RCN-only reference lines).
- New primary figure `three_regime_fits.png`: at three exclusion bands
  (default: cores included / intermediate / far), shows (top) parity plots
  of measured vs. identified-model (div k)^2 with the recovered equation
  annotated, and (bottom) transverse x-averaged profiles of measured vs.
  model, with the excluded slab shaded. This answers directly: "what PDE
  is recovered in each regime, how well does it hold, and where?"
- New figure `term_activation.png` replacing band_sweep.png: (a) a
  term-activation matrix (library term vs. b_in, annotated with selected
  coefficients) making visible which terms enter as the cores are
  approached; (b) the leading w^2 coefficient vs. b_in; (c) holdout R^2
  vs. b_in.
- Both targets are fit at each regime; the sign-definite linear form
  div k ~ library is reported (console + summary) to document that it is
  structurally invalid on the chain, but figures use the energy form.
- Recovered equations are printed to console and stored as strings in
  summary.json.

Geometry: the exclusion is a horizontal slab |y - y_core| < b_in spanning
the full chain; fits are performed on the annulus
b_in < |y - y_core| < band_out (x-trimmed, ramp/jump masked).

Usage:
    python self_dual_sindy_pismen_v2.py                 # _Args below
    python self_dual_sindy_pismen_v2.py --op_path ...   # CLI
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

from scipy.ndimage import binary_erosion

from self_dual_sindy_v3 import (
    PAPER_RC, panel_label,
    orient_vector_field, phi_jump_mask, build_library,
    stlsq, r_squared, fit_target, theory_lines_for,
    fig_library_correlation, fig_pareto, fig_coefficients,
    fig_bootstrap, fig_fit_and_residual,
    fig_fields_overview, fig_energy_balance, fig_sd_residual_map,
)


@dataclass
class _Args:
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

    # core geometry (centered coordinates; override if metadata says else)
    y_core: float = 0.0
    band_out: float = 4.0 * np.pi         # outer edge of analysis annulus
    band_in_sweep: tuple = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
    band_regimes: tuple = (0.0, 3.0, 5.0)  # cores-in / intermediate / far
    regime_target: str = "divk_sq"         # energy form (sign-agnostic)
    x_trim: float = 8.0

    # masking (phase-file conventions)
    ramp_thresh_bfs: float = 0.995
    ramp_thresh_strict: float = 0.999
    ramp_erosion_iters: int = 10
    phi_jump_tol: float = np.pi / 10.0

    alpha: float = 2.0 / 3.0
    eta: float = 8.0 / 9.0

    thresholds: tuple = tuple(np.geomspace(1e-4, 1.0, 25))
    n_boot: int = 200
    boot_frac: float = 0.5
    rel_tol_pareto: float = 0.01
    weight_by_target: bool = False
    seed: int = 0

    out_root: str = ""
    make_field_maps: bool = True
    # kept for interface compatibility with the knee _Args (unused here)
    synthetic_sigma: float = np.pi / 2.0


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

TARGET_LABEL = {
    "divk": r"\nabla\!\cdot\!\mathbf{k}",
    "divk_sq": r"(\nabla\!\cdot\!\mathbf{k})^2",
}

# display order for the activation matrix (leading term first)
TERM_ORDER = ["w^2", "w", "w^3", "(curl k)^2", "curl k", "J",
              "k_1 k_2", "k_2", "|grad w|", "1"]

TERM_TEX = {
    "w^2": r"$w^2$", "w": r"$w$", "w^3": r"$w^3$",
    "(curl k)^2": r"$(\nabla\!\times\!\mathbf{k})^2$",
    "curl k": r"$\nabla\!\times\!\mathbf{k}$",
    "J": r"$J$", "k_1 k_2": r"$k_1 k_2$", "k_2": r"$k_2$",
    "|grad w|": r"$|\nabla w|$", "1": r"$1$",
}


def model_equation(names, coef_full, support, target_key, max_terms=6):
    """Human-readable equation for the selected model, |coef|-sorted."""
    idx = [i for i in np.where(support)[0]]
    idx.sort(key=lambda i: -abs(coef_full[i]))
    shown = idx[:max_terms]
    terms = " ".join(
        f"{coef_full[i]:+.3g} {names[i]}" for i in shown)
    tail = f" + ({len(idx) - max_terms} more)" if len(idx) > max_terms else ""
    lhs = "(div k)^2" if target_key == "divk_sq" else "div k"
    return f"{lhs} = {terms}{tail}"


def model_equation_tex(names, coef_full, support, target_key, max_terms=4):
    idx = [i for i in np.where(support)[0]]
    idx.sort(key=lambda i: -abs(coef_full[i]))
    shown = idx[:max_terms]
    parts = []
    for i in shown:
        t = TERM_TEX.get(names[i], names[i]).strip("$")
        parts.append(f"{coef_full[i]:+.2f}\\,{t}")
    tail = (f"\\,+\\cdots({len(idx) - max_terms})"
            if len(idx) > max_terms else "")
    return "$" + TARGET_LABEL[target_key] + " = " + " ".join(parts) + tail + "$"


def _row_profiles(field2d, model2d, mask, y_c, min_count=30):
    """x-averaged measured and model profiles over the sample mask."""
    cnt = mask.sum(axis=1)
    fm = np.where(mask, field2d, np.nan)
    mm = np.where(mask, model2d, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fmean = np.nanmean(fm, axis=1)
        mmean = np.nanmean(mm, axis=1)
    ok = cnt >= min_count
    fmean = np.where(ok, fmean, np.nan)
    mmean = np.where(ok, mmean, np.nan)
    return y_c, fmean, mmean


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def fig_three_regimes(out, regime_rows, target_key, band_out):
    """2x3: parity plots (top) and transverse profiles (bottom), one
    column per exclusion band, equations annotated."""
    n = len(regime_rows)
    fig, axs = plt.subplots(2, n, figsize=(3.9 * n, 6.4),
                            constrained_layout=True)
    if n == 1:
        axs = axs.reshape(2, 1)
    lab = TARGET_LABEL[target_key]

    for j, rr in enumerate(regime_rows):
        # ---- top: parity ----
        ax = axs[0, j]
        ym, yh = rr["y_meas"], rr["y_model"]
        step = max(1, ym.size // 4000)
        ax.scatter(ym[::step], yh[::step], s=3, alpha=0.25, color="0.55",
                   rasterized=True)
        lo = min(ym.min(), yh.min())
        hi = max(ym.max(), yh.max())
        ax.plot([lo, hi], [lo, hi], "-", color="k", lw=0.9)
        ax.set_xlabel(rf"measured ${lab}$")
        if j == 0:
            ax.set_ylabel(rf"model ${lab}$")
        ax.set_title(rf"$b_{{\rm in}} = {rr['band_in']:g}$"
                     + ("  (cores included)" if rr["band_in"] == 0 else ""),
                     fontsize=9)
        ax.text(0.03, 0.86,
                rr["eq_tex"] + "\n"
                + rf"$R^2_{{\rm holdout}} = {rr['r2_holdout']:.3f}$"
                + rf",  {rr['n_terms']} terms",
                transform=ax.transAxes, va="top", ha="left", fontsize=6.5,
                bbox=dict(fc="white", ec="0.7", alpha=0.85, pad=2.5))
        panel_label(ax, f"({chr(97 + j)})")

        # ---- bottom: transverse profile ----
        ax = axs[1, j]
        yy, fmean, mmean = rr["profile"]
        ax.plot(yy, fmean, "o", ms=2.6, color="0.35", label="measured")
        ax.plot(yy, mmean, "-", color="C0", lw=1.6, label="model")
        if rr["band_in"] > 0:
            ax.axvspan(-rr["band_in"], rr["band_in"], color="0.85",
                       zorder=0, label="excluded slab")
        ax.set_xlim(-band_out, band_out)
        ax.set_xlabel("$y$")
        if j == 0:
            ax.set_ylabel(rf"$\langle {lab} \rangle_x$")
            ax.legend(loc="best", fontsize=6.5)
        panel_label(ax, f"({chr(97 + n + j)})")

    fig.savefig(out)
    plt.close(fig)


def fig_term_activation(out, sweep_rows, names, mu, alpha, eta,
                        band_regimes=()):
    """(a) term-activation matrix over the band sweep, annotated with
    selected coefficients; (b) leading w^2 coefficient vs b_in;
    (c) holdout R^2 vs b_in."""
    b = [r["band_in"] for r in sweep_rows]
    order = [t for t in TERM_ORDER if t in names]
    M = np.full((len(order), len(b)), np.nan)
    for jc, r in enumerate(sweep_rows):
        for t, v in r.get("shares", r["coeffs"]).items():
            if t in order:
                M[order.index(t), jc] = v

    fig = plt.figure(figsize=(10.8, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0])
    ax = fig.add_subplot(gs[:, 0])

    C = np.ma.masked_invalid(np.minimum(np.abs(M), 1.2))
    ax.pcolormesh(np.arange(len(b) + 1), np.arange(len(order) + 1),
                  C, cmap="Blues", vmin=0, vmax=1.8,
                  edgecolors="0.85", lw=0.4)
    for i in range(len(order)):
        for jc in range(len(b)):
            if np.isfinite(M[i, jc]):
                ax.text(jc + 0.5, i + 0.5, f"{M[i, jc]:+.2f}",
                        ha="center", va="center", fontsize=5.8,
                        color="white" if abs(M[i, jc]) > 0.75 else "black")
    ax.set_xticks(np.arange(len(b)) + 0.5)
    ax.set_xticklabels([f"{bi:g}" for bi in b], fontsize=7)
    ax.set_yticks(np.arange(len(order)) + 0.5)
    ax.set_yticklabels([TERM_TEX[t] for t in order], fontsize=8)
    ax.invert_yaxis()
    for br in band_regimes:
        if br in b:
            jc = b.index(br)
            ax.add_patch(plt.Rectangle((jc, 0), 1, len(order), fill=False,
                                       ec="C3", lw=1.4))
    ax.set_xlabel(r"core exclusion half-width $b_{\rm in}$")
    ax.set_title(r"signed contribution share "
                 r"$c_i\,\mathrm{std}(\theta_i)/\mathrm{std}(\mathrm{target})$"
                 "  (red boxes: showcased regimes)", fontsize=9)
    ax.text(-0.11, 1.02, "(a)", transform=ax.transAxes,
            fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    c = [r["coeffs"].get("w^2", np.nan) for r in sweep_rows]
    e = [r["coef_w2_std"] for r in sweep_rows]
    ax.errorbar(b, c, yerr=e, fmt="o-", ms=4, color="C0", capsize=3)
    ax.axhline(alpha / eta, color="C3", lw=1.0, ls=":",
               label=r"knee value $\alpha/\eta$ (reference)")
    ax.set_ylabel(r"$c$ in $c\,w^2$")
    ax.legend(loc="lower right", fontsize=7)
    ax.tick_params(labelbottom=False)
    panel_label(ax, "(b)")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(b, [r["r2_holdout"] for r in sweep_rows], "o-", ms=4, color="C0")
    ax.set_xlabel(r"$b_{\rm in}$")
    ax.set_ylabel(r"holdout $R^2$")
    panel_label(ax, "(c)")

    fig.suptitle(rf"$\mu = {mu}$: energy-form identification vs.\ distance "
                 "from the chain", y=1.03)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(args: _Args):
    mpl.rcParams.update(PAPER_RC)
    rng = np.random.default_rng(args.seed)

    op = np.load(args.op_path)
    x, y = op["x"], op["y"]
    u = op["u"][..., args.t_index]
    ramp = op["ramp"]
    k1_raw = op[args.k1_key][..., args.t_index]
    k2_raw = op[args.k2_key][..., args.t_index]
    stem = Path(args.op_path).stem

    out_root = Path(args.out_root) if args.out_root else (
        Path(__file__).resolve().parent / "results"
        / "self_dual_sindy_pismen_v2" / stem)
    out_root.mkdir(parents=True, exist_ok=True)
    print("Output:", out_root)

    x0, y0 = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
    x_c, y_c = x - x0, y - y0
    dx, dy = x_c[1] - x_c[0], y_c[1] - y_c[0]
    extent = [x_c.min(), x_c.max(), y_c.min(), y_c.max()]
    Xc, Yc = np.meshgrid(x_c, y_c)

    rn = (ramp - np.nanmin(ramp)) / (np.nanmax(ramp) - np.nanmin(ramp) + 1e-12)
    st = np.ones((3, 3), bool)
    m_bfs = binary_erosion(rn >= args.ramp_thresh_bfs, st,
                           iterations=args.ramp_erosion_iters)
    m_strict = binary_erosion(rn >= args.ramp_thresh_strict, st,
                              iterations=args.ramp_erosion_iters)

    k1o, k2o = orient_vector_field(k1_raw, k2_raw, m_bfs)
    k1o, k2o = np.asarray(k1o), np.asarray(k2o)
    phi = np.arctan2(k2o, k1o)
    valid = m_strict & (~phi_jump_mask(phi, args.phi_jump_tol))
    valid &= (Xc > x_c.min() + args.x_trim) & (Xc < x_c.max() - args.x_trim)
    print("Valid points (pre-band):", int(valid.sum()))

    lib, targets, sample_mask_full, aux = build_library(
        k1o, k2o, dx, dy, valid, args.mu)
    names = list(lib.keys())

    dist = np.abs(Yc - args.y_core)
    annulus_out = sample_mask_full & (dist < args.band_out)

    def fit_on(sm, tkey):
        Theta = np.column_stack([lib[nm][sm] for nm in names])
        yvec = targets[tkey][sm]
        X_train = Xc[sm] < 0.0
        return Theta, yvec, fit_target(names, Theta, yvec, args, rng, X_train)

    # ---------------- band sweep (energy form) ----------------
    sweep_rows = []
    for b_in in args.band_in_sweep:
        sm = annulus_out & (dist > b_in)
        n_s = int(sm.sum())
        if n_s < 500:
            print(f"band_in={b_in}: too few samples ({n_s}), skipping")
            continue
        Theta_b, yvec_b, res = fit_on(sm, "divk_sq")
        sel_idx = np.where(res["support"])[0]
        coeffs = {names[i]: float(res["coef_full"][i]) for i in sel_idx}
        ystd = float(yvec_b.std()) + 1e-15
        shares = {names[i]: float(res["coef_full"][i]
                                  * Theta_b[:, i].std() / ystd)
                  for i in sel_idx}
        if "w^2" in coeffs:
            jb = [names[i] for i in sel_idx].index("w^2")
            cw2_std = float(res["boots"][:, jb].std())
        else:
            cw2_std = np.nan
        row = dict(band_in=float(b_in), n_samples=n_s, coeffs=coeffs,
                   shares=shares, coef_w2_std=cw2_std,
                   r2_holdout=float(res["chosen"]["r2_test"]),
                   r2_full=float(res["r2_full"]),
                   equation=model_equation(names, res["coef_full"],
                                           res["support"], "divk_sq"))
        sweep_rows.append(row)
        print(f"band_in={b_in:4.1f}: N={n_s:6d}  R2={res['r2_full']:.4f}  "
              f"{row['equation']}")

    fig_term_activation(out_root / "term_activation.png", sweep_rows, names,
                        args.mu, args.alpha, args.eta,
                        band_regimes=args.band_regimes)

    # ---------------- three-regime showcase ----------------
    tkey = args.regime_target
    regime_rows = []
    print(f"\n--- three-regime showcase, target: {tkey} ---")
    for b_in in args.band_regimes:
        sm = annulus_out & (dist > b_in)
        if int(sm.sum()) < 500:
            print(f"regime band_in={b_in}: too few samples, skipping")
            continue
        Theta, yvec, res = fit_on(sm, tkey)
        yhat = Theta @ res["coef_full"]
        yhat2d = np.full_like(targets[tkey], np.nan)
        yhat2d[sm] = yhat
        prof = _row_profiles(targets[tkey], yhat2d, sm, y_c)

        # linear form on the same annulus, for the record
        _, _, res_lin = fit_on(sm, "divk")

        eq = model_equation(names, res["coef_full"], res["support"], tkey)
        rr = dict(band_in=float(b_in), n_samples=int(sm.sum()),
                  y_meas=yvec, y_model=yhat, profile=prof,
                  eq=eq,
                  eq_tex=model_equation_tex(names, res["coef_full"],
                                            res["support"], tkey),
                  n_terms=int(res["support"].sum()),
                  r2_full=float(res["r2_full"]),
                  r2_holdout=float(res["chosen"]["r2_test"]),
                  coeffs={names[i]: float(res["coef_full"][i])
                          for i in np.where(res["support"])[0]},
                  boot_std={names[i]: float(res["boots"][:, j].std())
                            for j, i in
                            enumerate(np.where(res["support"])[0])},
                  linear_form_r2_holdout=float(res_lin["chosen"]["r2_test"]))
        regime_rows.append(rr)
        print(f"band_in={b_in:g}: {eq}")
        print(f"           R2 full={rr['r2_full']:.4f}, "
              f"holdout={rr['r2_holdout']:.4f};  "
              f"linear-form holdout R2={rr['linear_form_r2_holdout']:.3f}")

    fig_three_regimes(out_root / "three_regime_fits.png", regime_rows,
                      tkey, args.band_out)

    # ---------------- full figure set at the middle regime ----------------
    band_show = args.band_regimes[min(1, len(args.band_regimes) - 1)]
    sm = annulus_out & (dist > band_show)
    Theta = np.column_stack([lib[nm][sm] for nm in names])
    fig_library_correlation(names, Theta,
                            out_root / "library_correlation.png")

    summary = {"band_out": float(args.band_out),
               "band_showcase": float(band_show),
               "band_sweep": [
                   {k: v for k, v in r.items()} for r in sweep_rows],
               "regimes": [
                   {k: v for k, v in r.items()
                    if k not in ("y_meas", "y_model", "profile")}
                   for r in regime_rows],
               "targets": {}}

    for t in ("divk_sq", "divk"):
        _, yvec, res = fit_on(sm, t)
        theory = theory_lines_for(t, args)
        fig_pareto(res, t, out_root / f"pareto_{t}.png")
        fig_coefficients(names, res, theory, t,
                         out_root / f"coefficients_{t}.png")
        fig_bootstrap(names, res, theory, t,
                      out_root / f"bootstrap_{t}.png")
        yhat2d = np.full_like(targets[t], np.nan)
        yhat2d[sm] = Theta @ res["coef_full"]
        fig_fit_and_residual(targets[t], yhat2d, sm, extent, t,
                             out_root / f"fit_residual_{t}.png")
        sel = {names[i]: float(res["coef_full"][i])
               for i in np.where(res["support"])[0]}
        errs = {names[i]: float(res["boots"][:, j].std())
                for j, i in enumerate(np.where(res["support"])[0])}
        print(f"\n[{t}] band_in={band_show}: " +
              ", ".join(f"{k}: {v:+.4f} ± {errs[k]:.4f}"
                        for k, v in sel.items()))
        print(f"[{t}] R^2 = {res['r2_full']:.4f}, "
              f"holdout = {res['chosen']['r2_test']:.4f}")
        summary["targets"][t] = dict(
            selected=sel, boot_std=errs,
            r2_full=float(res["r2_full"]),
            r2_holdout=float(res["chosen"]["r2_test"]))

    if args.make_field_maps:
        fig_fields_overview(out_root / "fields_overview.png",
                            u, k1o, k2o, aux, sm, extent)
        rms_rel = fig_energy_balance(out_root / "energy_balance.png",
                                     aux, sm, extent, args.alpha, args.eta)
        summary["energy_balance_rms_rel_error"] = rms_rel
        cw = summary["targets"]["divk"]["selected"].get("w")
        if cw is not None:
            fig_sd_residual_map(out_root / "sd_residual_map.png",
                                aux, sm, extent, cw)

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
    p.add_argument("--y_core", type=float, default=d.y_core)
    p.add_argument("--band_out", type=float, default=d.band_out)
    p.add_argument("--band_regimes", type=float, nargs="+",
                   default=list(d.band_regimes))
    p.add_argument("--regime_target", default=d.regime_target)
    p.add_argument("--x_trim", type=float, default=d.x_trim)
    p.add_argument("--out_root", default=d.out_root)
    a = p.parse_args()
    kw = {**d.__dict__, **vars(a)}
    kw["band_regimes"] = tuple(kw["band_regimes"])
    return _Args(**kw)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(parse_cli())
    else:
        main(_Args())