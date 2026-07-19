"""
self_dual_sindy_pismen.py
-------------------------
Self-dual identification on a DISLOCATION-CHAIN (Pismen/zipper) state,
with a core-exclusion band sweep.

Physics being tested: RCN theory (Ercolani et al.) says self-dual balance
holds almost everywhere in the neighborhood of dislocations, failing only
at the cores where the Hessian obstruction is supported. So we regress
    (div k)^2 ~ library      and      div k ~ library
on the annular band  band_in < |y - y_core| < band_out, sweeping band_in
from 0 (cores included) outward. Expectation: the single-term mismatch
model (w^2, resp. w) is selected and its R^2 / coefficient stabilize once
the cores are excluded; with the cores included, either the fit degrades
or defect-density terms (J) enter.

Notes vs. the knee runner:
- Phase-file npz keys default to k1_orig / k2_orig.
- The energy form (div k)^2 ~ w^2 is the primary target here: it is
  sign-agnostic, which matters because div k need not have uniform sign
  around a defect chain. The linear form is reported too, with that caveat.
- x_trim removes edge columns so the band is not contaminated by the ramp.

Usage:
    python self_dual_sindy_pismen.py                 # _Args below
    python self_dual_sindy_pismen.py --op_path ...   # CLI
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

from self_dual_sindy_v2 import (
    PAPER_RC, panel_label,
    orient_vector_field, phi_jump_mask, build_library,
    stlsq, r_squared, fit_target, theory_lines_for,
    fig_library_correlation, fig_pareto, fig_coefficients,
    fig_bootstrap, fig_fit_and_residual,
    fig_fields_overview, fig_energy_balance, fig_sd_residual_map,
    _Args as _KneeArgs,
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

    # core geometry (in centered coordinates; override if metadata says else)
    y_core: float = 0.0
    band_out: float = 4.0 * np.pi     # outer edge of analysis annulus
    band_in_sweep: tuple = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
    band_in_showcase: float = 3.0     # full figure set produced at this band
    x_trim: float = 8.0               # drop columns within x_trim of x-extremes

    # masking (phase-file conventions: pismen script used 0.995/0.999 + 10 er)
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
    # attenuation reference (same synthetic knee machinery; indicative only
    # for the chain, since the local geometry differs from a knee)
    synthetic_sigma: float = np.pi / 2.0


def band_sweep_figure(out, sweep_rows, mu, band_showcase):
    """Coefficient and R^2 of the energy-form fit vs core-exclusion radius."""
    b = [r["band_in"] for r in sweep_rows]
    c = [r["coef_w2"] for r in sweep_rows]
    e = [r["coef_w2_std"] for r in sweep_rows]
    r2 = [r["r2_holdout"] for r in sweep_rows]
    n = [r["n_samples"] for r in sweep_rows]
    extra = [r["extra_terms"] for r in sweep_rows]

    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.3), constrained_layout=True)

    ax = axs[0]
    ax.errorbar(b, c, yerr=e, fmt="o-", ms=5, color="C0", capsize=3,
                label=r"$c$ in $(\nabla\!\cdot\!\mathbf{k})^2 = c\,w^2$")
    for bi, ex in zip(b, extra):
        if ex:
            ax.plot([bi], [c[b.index(bi)]], "s", ms=10, mfc="none",
                    mec="C3", mew=1.5)
    ax.axvline(band_showcase, color="k", lw=0.8, ls=":")
    ax.set_xlabel(r"core exclusion radius $b_{\rm in}$")
    ax.set_ylabel("coefficient")
    ax.legend(loc="best")
    panel_label(ax, "(a)")

    ax = axs[1]
    ax.plot(b, r2, "o-", ms=5, color="C0")
    ax.set_xlabel(r"core exclusion radius $b_{\rm in}$")
    ax.set_ylabel(r"holdout $R^2$")
    ax2 = ax.twinx()
    ax2.plot(b, n, "s--", ms=4, color="0.6", alpha=0.8)
    ax2.set_ylabel("samples", color="0.5")
    ax2.tick_params(axis="y", colors="0.5")
    panel_label(ax, "(b)")

    fig.suptitle(rf"$\mu = {mu}$: self-dual balance vs.\ distance from chain"
                 " (open squares: extra terms selected)", y=1.04)
    fig.savefig(out)
    plt.close(fig)


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
        / "self_dual_sindy_pismen" / stem)
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

    xa = np.abs(Xc)
    valid &= (Xc > x_c.min() + args.x_trim) & (Xc < x_c.max() - args.x_trim)
    print("Valid points (pre-band):", int(valid.sum()))

    lib, targets, sample_mask_full, aux = build_library(
        k1o, k2o, dx, dy, valid, args.mu)
    names = list(lib.keys())

    dist = np.abs(Yc - args.y_core)
    annulus_out = sample_mask_full & (dist < args.band_out)

    # ---------------- band sweep (energy form) ----------------
    sweep_rows = []
    for b_in in args.band_in_sweep:
        sm = annulus_out & (dist > b_in)
        n_s = int(sm.sum())
        if n_s < 500:
            print(f"band_in={b_in}: too few samples ({n_s}), skipping")
            continue
        Theta = np.column_stack([lib[nm][sm] for nm in names])
        yvec = targets["divk_sq"][sm]
        X_train = Xc[sm] < 0.0
        res = fit_target(names, Theta, yvec, args, rng, X_train)

        sel_idx = np.where(res["support"])[0]
        sel_names = [names[i] for i in sel_idx]
        cw2 = res["coef_full"][names.index("w^2")] \
            if "w^2" in sel_names else np.nan
        if "w^2" in sel_names:
            j = sel_names.index("w^2")
            cw2_std = float(res["boots"][:, j].std())
        else:
            cw2_std = np.nan
        extra = [nm for nm in sel_names if nm != "w^2"]

        row = dict(band_in=float(b_in), n_samples=n_s,
                   coef_w2=float(cw2), coef_w2_std=cw2_std,
                   r2_holdout=float(res["chosen"]["r2_test"]),
                   r2_full=float(res["r2_full"]),
                   selected=sel_names, extra_terms=extra)
        sweep_rows.append(row)
        print(f"band_in={b_in:4.1f}: N={n_s:6d}  selected={sel_names}  "
              f"c_w2={cw2:+.4f}  R2={res['r2_full']:.4f}")

    band_sweep_figure(out_root / "band_sweep.png", sweep_rows,
                      args.mu, args.band_in_showcase)

    # ---------------- full figure set at showcase band ----------------
    sm = annulus_out & (dist > args.band_in_showcase)
    Theta = np.column_stack([lib[nm][sm] for nm in names])
    X_train = Xc[sm] < 0.0
    fig_library_correlation(names, Theta,
                            out_root / "library_correlation.png")

    summary = {"n_samples_showcase": int(sm.sum()),
               "band_in_showcase": args.band_in_showcase,
               "band_out": float(args.band_out),
               "band_sweep": sweep_rows, "targets": {}}

    for tkey in ("divk_sq", "divk"):
        yvec = targets[tkey][sm]
        res = fit_target(names, Theta, yvec, args, rng, X_train)
        theory = theory_lines_for(tkey, args)

        fig_pareto(res, tkey, out_root / f"pareto_{tkey}.png")
        fig_coefficients(names, res, theory, tkey,
                         out_root / f"coefficients_{tkey}.png")
        fig_bootstrap(names, res, theory, tkey,
                      out_root / f"bootstrap_{tkey}.png")
        yhat2d = np.full_like(targets[tkey], np.nan)
        yhat2d[sm] = Theta @ res["coef_full"]
        fig_fit_and_residual(targets[tkey], yhat2d, sm, extent, tkey,
                             out_root / f"fit_residual_{tkey}.png")

        sel = {names[i]: float(res["coef_full"][i])
               for i in np.where(res["support"])[0]}
        errs = {names[i]: float(res["boots"][:, j].std())
                for j, i in enumerate(np.where(res["support"])[0])}
        print(f"\n[{tkey}] band_in={args.band_in_showcase}: " +
              ", ".join(f"{k}: {v:+.4f} ± {errs[k]:.4f}"
                        for k, v in sel.items()))
        print(f"[{tkey}] R^2 = {res['r2_full']:.4f}, "
              f"holdout = {res['chosen']['r2_test']:.4f}")
        summary["targets"][tkey] = dict(
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
    p.add_argument("--band_in_showcase", type=float,
                   default=d.band_in_showcase)
    p.add_argument("--x_trim", type=float, default=d.x_trim)
    p.add_argument("--out_root", default=d.out_root)
    a = p.parse_args()
    return _Args(**{**d.__dict__, **vars(a)})


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(parse_cli())
    else:
        main(_Args())
