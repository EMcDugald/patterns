# pipelines/self_dual_sindy_riesz.py
"""
Stationary SINDy on the Riesz OP files (zigzag PGB), one frame at a time
(default: the final frame, the most relaxed chain).

Regresses the self-dual targets
    divk_sq = (div k)^2   (energy form, primary: sign-agnostic on a chain)
    divk    =  div k      (self-dual form, reported with the sign caveat)
against the knee-session library (w, w^2, w^3, curl k, J, k2, k1 k2,
|grad w|) AUGMENTED with the defect observables this pipeline provides:
    rho_d  doubled-angle disclination density (seam-free, core coverage)
    dA     A_mono / A_far - 1  (relaxation-immune amplitude deviation)
    dA^2
using the STLSQ -> Pareto -> bootstrap machinery (stlsq already
column-normalizes internally, so selection is scale-free across the
4+ orders of magnitude the augmented columns span) imported verbatim from
self_dual_sindy.py (which must sit in the same directory, per convention).

Differences from the uHu-era scripts, deliberate:
  - k comes from the Riesz OP file (k1_direct/k2_direct) oriented by the
    DETERMINISTIC gauge rule of run_riesz_metrics_zigzag.orient_direct_k
    (k . grad(theta_smooth) > 0, odd-in-y component fallback). No BFS.
  - No ramp in the OP file: the interior mask is physical-margin trims
    (x_trim, y_trim) plus finiteness of every library column.
  - Domain switch: --domain full (default, better pictures) or upper
    (y >= core). Both run under the same core-censoring band sweep:
    samples satisfy band_in < |y - y_core| < band_out, band_in swept
    from 0 (cores included) outward; full figure set at the showcase
    band. Expectation, as in the pismen script: the single-term mismatch
    model stabilizes once cores are excluded.
  - Optional coherence weighting (--coherence_weight): sample weights
    from the structure tensor coherence (= the pipeline's alignment,
    verified identical in verify_director_structure_tensor.py), computed
    from the source SH pattern at sigma_coh if the SH file can be
    located. Rows of (Theta, y) are scaled by sqrt(coherence): fit where
    the pattern is honestly rank one instead of masking by hand. Off by
    default; R^2 values are then coherence-weighted, stated in summary.
  - Modes: single file; --all over an input_dir with optional --mu
    filter; --pooled additionally stacks the samples of every selected
    file into one regression (library terms are mu-free by construction:
    w = |k|^2 - 1).

Outputs per file (and per pooled fit) under out_root:
  band_sweep_<target>.png, library_correlation.png, pareto_<t>.png,
  coefficients_<t>.png, bootstrap_<t>.png, fit_residual_<t>.png,
  summary.json.

Usage:
    python self_dual_sindy_riesz.py <op_path>
    python self_dual_sindy_riesz.py --all --input_dir <raw dir> --mu 0.9
Defaults in _Args at the bottom, mirrored by the CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from self_dual_sindy import (  # noqa: E402
    PAPER_RC, panel_label, build_library, fit_target, theory_lines_for,
    fig_library_correlation, fig_pareto, fig_coefficients, fig_bootstrap,
    fig_fit_and_residual, masked_central_derivs,
)
from run_riesz_metrics_zigzag import (  # noqa: E402
    load_op, ensure_theta_smooth, locate_sh_file, orient_direct_k,
    rho_d_from_director, nan_grad,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class _Args:
    op_path: str = ""
    all: bool = False
    input_dir: str = ("data/sh_pgb_zigzag/riesz_v8_aware/"
                      "mu_sweeps_full_Nx512_hp2_T25_NyF5_np18_Nsave125/raw")
    mu: float = -1.0            # -1: no filter; else select files by mu
    pooled: bool = False        # stack samples of all selected files
    t_index: int = -1

    domain: str = "full"        # full | upper
    band_out: float = 4.0 * np.pi
    band_in_sweep: tuple = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
    band_in_showcase: float = 2.0
    x_trim: float = 5.0         # physical units off each x edge
    y_trim: float = 2.0         # physical units off each y edge

    coherence_weight: bool = False
    sigma_coh: float = np.pi / 2
    sh_path: str = ""           # optional SH override for coherence

    theta_smooth_w: float = 0.75
    drop_terms: str = ""     # comma list, e.g. "k_2,k_1 k_2"
    alpha: float = 2.0 / 3.0
    eta: float = 8.0 / 9.0
    thresholds: tuple = tuple(np.geomspace(1e-4, 1.0, 25))
    n_boot: int = 200
    boot_frac: float = 0.5
    rel_tol_pareto: float = 0.01
    seed: int = 0
    out_root: str = ""


# ---------------------------------------------------------------------------
# Per-file sample assembly
# ---------------------------------------------------------------------------

def assemble_samples(op_path, args):
    """Load one OP file, orient k, build the augmented library on the
    domain/trim mask (band censoring applied later). Returns dict."""
    op = load_op(op_path)
    op, _, computed = ensure_theta_smooth(op, args.theta_smooth_w)
    if computed:
        print(f"    theta_smooth computed on the fly (w={args.theta_smooth_w})")

    x, y = op["x"], op["y"]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    core = int(op["core_row"])
    y_core = float(y[core])
    mu = float(op["mu"])
    t = args.t_index
    Xc, Yc = np.meshgrid(x, y)

    k1d = op["k1_direct"][:, :, t].astype(float)
    k2d = op["k2_direct"][:, :, t].astype(float)
    th_s = op["theta_smooth"][:, :, t].astype(float)
    k1o, k2o, fb_frac = orient_direct_k(k1d, k2d, th_s, dx, dy, core)

    mask = np.ones_like(k1o, dtype=bool)
    mask &= (Xc > x[0] + args.x_trim) & (Xc < x[-1] - args.x_trim)
    mask &= (Yc > y[0] + args.y_trim) & (Yc < y[-1] - args.y_trim)
    if args.domain == "upper":
        mask &= Yc >= y_core

    lib, targets, sample_mask, aux = build_library(k1o, k2o, dx, dy, mask, mu)

    # ---- augmented columns ----
    rho = rho_d_from_director(k1d, k2d, dx, dy)
    A = op["A_mono"][:, :, t].astype(float)
    af = float(op["A_far_median"][t]) if "A_far_median" in op else \
        float(np.nanmedian(A[mask]))
    dA = A / max(af, 1e-12) - 1.0
    lib["rho_d"] = rho
    lib["dA"] = dA
    lib["dA^2"] = dA ** 2
    for f in (rho, dA):
        sample_mask &= np.isfinite(f)

    # ---- optional term dropping (e.g. k_2, k_1 k_2 on the upper half,
    # where the knee relation w = k_2^2 - mu^2 makes them w proxies) ----
    if args.drop_terms:
        for nm in [t.strip() for t in args.drop_terms.split(",") if t.strip()]:
            if nm in lib:
                del lib[nm]
                print(f"    dropped library term: {nm}")
            else:
                print(f"    drop_terms: no such term '{nm}' "
                      f"(have: {list(lib.keys())})")

    # ---- optional coherence weights ----
    coh = None
    if args.coherence_weight:
        sh_file = locate_sh_file(op, op_path,
                                 args.sh_path if args.sh_path else None)
        if sh_file is None:
            print("    coherence_weight requested but SH file not found; "
                  "proceeding UNWEIGHTED")
        else:
            from utils.field_diagnostics_viz import riesz_director
            sh = np.load(sh_file, allow_pickle=True)
            u = sh["u"][:, :, t] if sh["u"].ndim == 3 else sh["u"]
            Lx, Ly = x[-1] - x[0], y[-1] - y[0]
            rd = riesz_director(u.astype(float), Lx, Ly, args.sigma_coh)
            coh = np.clip(rd["alignment"], 0.0, 1.0)
            print(f"    coherence weights from {sh_file.name} "
                  f"(sigma={args.sigma_coh:.3f})")

    return dict(op_path=Path(op_path), mu=mu, x=x, y=y, Xc=Xc, Yc=Yc,
                y_core=y_core, lib=lib, targets=targets,
                sample_mask=sample_mask, aux=aux, coh=coh,
                fb_frac=fb_frac,
                extent=[x[0], x[-1], y[0], y[-1]])


def stack_rows(datasets, names, band_in, band_out, target_key):
    """Row-stack (Theta, y, weights, train_mask) across datasets for the
    given censoring band."""
    Ths, ys, ws, trs = [], [], [], []
    for d in datasets:
        dist = np.abs(d["Yc"] - d["y_core"])
        sm = d["sample_mask"] & (dist > band_in) & (dist < band_out)
        if not sm.any():
            continue
        Ths.append(np.column_stack([d["lib"][nm][sm] for nm in names]))
        ys.append(d["targets"][target_key][sm])
        trs.append(d["Xc"][sm] < 0.0)
        if d["coh"] is not None:
            ws.append(d["coh"][sm])
        else:
            ws.append(np.ones(int(sm.sum())))
    if not Ths:
        return None
    Theta = np.vstack(Ths)
    yv = np.concatenate(ys)
    wv = np.concatenate(ws)
    tr = np.concatenate(trs)
    sw = np.sqrt(np.maximum(wv, 0.0))
    return Theta * sw[:, None], yv * sw, tr, int(len(yv))


# ---------------------------------------------------------------------------
# Band sweep figure (both targets)
# ---------------------------------------------------------------------------

def fig_band_sweep(out, rows, mu_label, showcase, target_key, prim_name):
    b = [r["band_in"] for r in rows]
    c = [r["coef_prim"] for r in rows]
    e = [r["coef_prim_std"] for r in rows]
    r2 = [r["r2_holdout"] for r in rows]
    n = [r["n_samples"] for r in rows]
    extra = [r["extra_terms"] for r in rows]

    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.3), constrained_layout=True)
    ax = axs[0]
    ax.errorbar(b, c, yerr=e, fmt="o-", ms=5, color="C0", capsize=3,
                label=rf"coef of {prim_name} in {target_key}")
    for i, ex in enumerate(extra):
        if ex:
            ax.plot([b[i]], [c[i]], "s", ms=10, mfc="none", mec="C3", mew=1.5)
    ax.axvline(showcase, color="k", lw=0.8, ls=":")
    ax.set_xlabel(r"core exclusion radius $b_{\rm in}$")
    ax.set_ylabel("coefficient")
    ax.legend(loc="best", fontsize=8)
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

    fig.suptitle(f"{mu_label}: {target_key} vs distance from chain "
                 "(open squares: extra terms selected)", y=1.04)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# One fit unit (single file or pooled)
# ---------------------------------------------------------------------------


def fig_contributions(out, names, res, Theta_std, target_key):
    """Standardized contributions |coef| * column std for the selected
    terms: the variance-relevant size of each term, unlike the raw
    physical-unit coefficients, which make localized small-magnitude
    columns (J, rho_d) look dominant."""
    idx = np.where(res["support"])[0]
    labs = [names[i] for i in idx]
    contrib = [abs(res["coef_full"][i]) * Theta_std[i] for i in idx]
    order = np.argsort(contrib)[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
    ax.bar([labs[o] for o in order], [contrib[o] for o in order],
           color="C0")
    ax.set_ylabel(r"$|c_i|\,\mathrm{std}(\Theta_i)$")
    ax.set_title(f"Standardized term contributions, target: {target_key}")
    ax.tick_params(axis="x", rotation=45)
    fig.savefig(out)
    plt.close(fig)


PRIM = {"divk_sq": "w^2", "divk": "w"}


def run_fits(datasets, args, out_root, label):
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    names = list(datasets[0]["lib"].keys())
    summary = {"label": label, "domain": args.domain,
               "coherence_weight": bool(args.coherence_weight
                                        and datasets[0]["coh"] is not None),
               "files": [str(d["op_path"]) for d in datasets],
               "orient_fallback_frac": [d["fb_frac"] for d in datasets],
               "band_out": float(args.band_out),
               "band_in_showcase": float(args.band_in_showcase),
               "targets": {}}

    for tkey in ("divk_sq", "divk"):
        prim = PRIM[tkey]
        rows = []
        for b_in in args.band_in_sweep:
            packed = stack_rows(datasets, names, b_in, args.band_out, tkey)
            if packed is None:
                continue
            Theta, yv, tr, n_s = packed
            if n_s < 500:
                print(f"    band_in={b_in}: too few samples ({n_s}), skip")
                continue
            res = fit_target(names, Theta, yv, args, rng, tr)
            sel = [names[i] for i in np.where(res["support"])[0]]
            if prim in sel:
                jp = sel.index(prim)
                cp = float(res["coef_full"][names.index(prim)])
                cps = float(res["boots"][:, jp].std())
            else:
                cp, cps = np.nan, np.nan
            rows.append(dict(band_in=float(b_in), n_samples=n_s,
                             coef_prim=cp, coef_prim_std=cps,
                             r2_holdout=float(res["chosen"]["r2_test"]),
                             r2_full=float(res["r2_full"]),
                             selected=sel,
                             extra_terms=[s for s in sel if s != prim]))
            print(f"    [{tkey}] band_in={b_in:4.1f}: N={n_s:7d}  "
                  f"selected={sel}  c_{prim}={cp:+.4f}  "
                  f"R2={res['r2_full']:.4f}")
        fig_band_sweep(out_root / f"band_sweep_{tkey}.png", rows, label,
                       args.band_in_showcase, tkey, prim)
        summary["targets"][tkey] = {"band_sweep": rows}

    # ---- full figure set at the showcase band, per target ----
    packed = stack_rows(datasets, names, args.band_in_showcase,
                        args.band_out, "divk_sq")
    Theta, _, tr, n_s = packed
    summary["n_samples_showcase"] = n_s
    fig_library_correlation(names, Theta,
                            out_root / "library_correlation.png")
    for tkey in ("divk_sq", "divk"):
        Theta, yv, tr, n_s = stack_rows(datasets, names,
                                        args.band_in_showcase,
                                        args.band_out, tkey)
        res = fit_target(names, Theta, yv, args, rng, tr)
        theory = theory_lines_for(tkey, args)
        fig_pareto(res, tkey, out_root / f"pareto_{tkey}.png")
        fig_coefficients(names, res, theory, tkey,
                         out_root / f"coefficients_{tkey}.png")
        fig_bootstrap(names, res, theory, tkey,
                      out_root / f"bootstrap_{tkey}.png")
        fig_contributions(out_root / f"contributions_{tkey}.png",
                          names, res, Theta.std(axis=0), tkey)
        if len(datasets) == 1:
            d = datasets[0]
            dist = np.abs(d["Yc"] - d["y_core"])
            sm = d["sample_mask"] & (dist > args.band_in_showcase) \
                & (dist < args.band_out)
            yhat2d = np.full(d["Yc"].shape, np.nan)
            Th2 = np.column_stack([d["lib"][nm][sm] for nm in names])
            yhat2d[sm] = Th2 @ res["coef_full"]
            fig_fit_and_residual(d["targets"][tkey], yhat2d, sm,
                                 d["extent"], tkey,
                                 out_root / f"fit_residual_{tkey}.png")
        sel = {names[i]: float(res["coef_full"][i])
               for i in np.where(res["support"])[0]}
        errs = {names[i]: float(res["boots"][:, j].std())
                for j, i in enumerate(np.where(res["support"])[0])}
        print(f"  [{tkey}] showcase band {args.band_in_showcase}: " +
              ", ".join(f"{k}: {v:+.4f} +/- {errs[k]:.4f}"
                        for k, v in sel.items()) +
              f"  R2={res['r2_full']:.4f} holdout="
              f"{res['chosen']['r2_test']:.4f}")
        summary["targets"][tkey]["showcase"] = dict(
            selected=sel, boot_std=errs, r2_full=float(res["r2_full"]),
            r2_holdout=float(res["chosen"]["r2_test"]))

    with open(out_root / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  wrote {out_root / 'summary.json'}")
    return summary


# ---------------------------------------------------------------------------
# Orchestration + CLI
# ---------------------------------------------------------------------------

def select_files(args):
    if args.op_path:
        return [Path(args.op_path)]
    files = sorted(Path(args.input_dir).glob("*.npz"))
    if args.mu > 0:
        tag = f"mu{args.mu:.3f}".replace(".", "_")
        alt = f"mu{args.mu:.3f}"
        files = [f for f in files if tag in f.name.replace(".", "_")
                 or alt in f.name]
    if not files:
        raise FileNotFoundError("No OP files selected.")
    return files


def main(args: _Args):
    matplotlib.rcParams.update(PAPER_RC)
    files = select_files(args)
    print(f"{len(files)} file(s), domain={args.domain}, "
          f"t_index={args.t_index}, pooled={args.pooled}")

    base = Path(args.out_root) if args.out_root else (
        _HERE / "results" / "self_dual_sindy_riesz")

    datasets = []
    for f in files:
        print(f"  loading {f.name}")
        datasets.append(assemble_samples(f, args))

    if args.pooled and len(datasets) > 1:
        label = f"pooled ({len(datasets)} files), domain={args.domain}"
        run_fits(datasets, args, base / f"pooled_{args.domain}", label)
    else:
        for d in datasets:
            label = rf"$\mu={d['mu']}$, domain={args.domain}"
            run_fits([d], args,
                     base / f"{d['op_path'].stem}_{args.domain}", label)
    print("Done.")


def parse_cli() -> _Args:
    d = _Args()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("op_path", nargs="?", default=d.op_path)
    p.add_argument("--all", action="store_true", default=d.all)
    p.add_argument("--input_dir", default=d.input_dir)
    p.add_argument("--mu", type=float, default=d.mu)
    p.add_argument("--pooled", action="store_true", default=d.pooled)
    p.add_argument("--t_index", type=int, default=d.t_index)
    p.add_argument("--domain", choices=("full", "upper"), default=d.domain)
    p.add_argument("--band_out", type=float, default=d.band_out)
    p.add_argument("--band_in_showcase", type=float,
                   default=d.band_in_showcase)
    p.add_argument("--x_trim", type=float, default=d.x_trim)
    p.add_argument("--y_trim", type=float, default=d.y_trim)
    p.add_argument("--coherence_weight", action="store_true",
                   default=d.coherence_weight)
    p.add_argument("--sigma_coh", type=float, default=d.sigma_coh)
    p.add_argument("--sh_path", default=d.sh_path)
    p.add_argument("--theta_smooth_w", type=float, default=d.theta_smooth_w)
    p.add_argument("--drop_terms", default=d.drop_terms,
                   help="Comma list of library terms to remove, "
                        "e.g. --drop_terms 'k_2,k_1 k_2' for "
                        "upper-half fits.")
    p.add_argument("--out_root", default=d.out_root)
    a = p.parse_args()
    return _Args(**{**d.__dict__, **vars(a)})


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(parse_cli())
    else:
        main(_Args())