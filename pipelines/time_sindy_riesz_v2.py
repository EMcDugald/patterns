# pipelines/time_sindy_riesz_v2.py
"""
Time-SINDy runner v2. Same fits as time_sindy_riesz.py, but the sample domain is
the large reliable interior instead of a thin near-core strip, swept over a set
of core holes, with the diagnostics folded in and the recovered PDE printed.

No file is patched. It reuses time_sindy_riesz.build_time_samples and .fit_window
and the time_sindy_diagnostics figures unchanged, by expressing the domain as
(band_y, x_trim, band_in) on a copied args, so every downstream piece produces
exactly this domain without edits.

Domain
------
The reliable region is the shared finite-theta_smooth footprint over the interior
frames (the reconstruction hull). We stay a margin inside it (default 1 x-wave in
x, 1 y-wave in y) and fit the largest symmetric interior rectangle, never the
ragged NaN edge. Core holes censor |y - y_core| < hole for a set of hole sizes in
y-wavelengths (default 0, 1, 2, 3), giving the with-core vs core-removed contrast:
if a law survives as the core is carved out, it is a field property; if it
collapses, it was a core property.

Read the domain sweep, not the headline R^2: opening to the far field adds many
near-constant easy samples that inflate R^2 without testing the law. The signal
is whether the SELECTED TERMS and coefficients stay put as the hole grows.

Collinearity is reported (VIF + correlation figure), never auto-handled: which
member of a collinear cluster to keep is a physics choice.

Outputs (relative):
  <out>/pde_table.txt              every subcase's recovered PDE + R2 + holdout
  <out>/summary.json               machine-readable
  <out>/*.png                      reconstruction / overlap / query / correlation
                                   at the full domain (hole 0)

Run:
  python time_sindy_riesz_v2.py <op_path> --u_from /path/to/raw_sh_dir
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import copy
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from time_sindy_riesz import _Args, build_time_samples, detect_freeze_frame, fit_window
from run_riesz_metrics_zigzag import load_op, ensure_theta_smooth
from self_dual_sindy import PAPER_RC, fig_library_correlation
from time_sindy_diagnostics import (
    fig_reconstruction, fig_feature_overlap, fig_query_points,
    _resolve_u, _pick_frame,
)


# ---------------------------------------------------------------------------
# reliable-interior rectangle from the theta hull
# ---------------------------------------------------------------------------

from scipy.ndimage import binary_erosion


def compute_hull_domain(op, margin, ycap_frac=0.5):
    """Bowtie-aware domain from the shared finite-theta_smooth hull.

      drop : finite cells within `margin` (physical) of the NaN boundary, to be
             NaN'd out so build_time_samples' finite filter excludes them; this
             is the erosion that keeps the fit off the ragged wedge edge.
      y_cap: |y - core| at which the per-row finite fraction falls below
             ycap_frac, capping the thin wedge tips in y.
      x_trim: a small margin off the crop x-edges.

    We do not force a rectangle: the sampled region is the eroded hull itself,
    intersected with |y-core| <= y_cap and the core hole, all handled downstream
    by the existing rectangle mask AND the finite filter together.
    """
    th = op["theta_smooth"]
    x, y = op["x"], op["y"]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    core = int(op["core_row"])
    finite = np.all(np.isfinite(th), axis=2)

    it = max(1, int(round(margin / min(dx, dy))))
    eroded = binary_erosion(finite, iterations=it)
    drop = finite & ~eroded

    rowfrac = finite.mean(axis=1)
    up = core
    while up + 1 < len(y) and rowfrac[up + 1] >= ycap_frac:
        up += 1
    dn = core
    while dn - 1 >= 0 and rowfrac[dn - 1] >= ycap_frac:
        dn -= 1
    y_cap = min(float(y[up] - y[core]), float(y[core] - y[dn]))

    info = dict(y_cap=y_cap, x_trim=float(margin), erosion_px=int(it),
                hull_finite_frac=float(finite.mean()),
                y_up=float(y[up] - y[core]), y_dn=float(y[core] - y[dn]))
    return drop, y_cap, float(margin), info


# ---------------------------------------------------------------------------
# PDE printout
# ---------------------------------------------------------------------------

_TSYM = {"dA_dt": "\u2202\u209c A", "dw_dt": "\u2202\u209c w",
         "dth_dt": "\u2202\u209c \u03b8"}
_PRETTY = {"1": "1", "w": "w", "w^2": "w\u00b2", "dA": "dA", "dA^2": "dA\u00b2",
           "J_or": "J_or", "J_th": "J_\u03b8", "rho_d": "\u03c1_d",
           "curl k": "curl k", "A": "A", "A^3": "A\u00b3", "Lap A": "\u2207\u00b2A",
           "Lap w": "\u2207\u00b2w", "div k": "div k", "|grad w|": "|\u2207w|",
           "th_xx": "\u03b8_xx", "th_yy": "\u03b8_yy"}


def format_pde(target, coefs, r2, holdout):
    tsym = _TSYM.get(target, target)
    if not coefs:
        return f"{tsym} = 0    [R\u00b2={r2:.3f}, holdout={holdout:.3f}]"
    terms = []
    for nm, c in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
        sign = "+" if c >= 0 else "\u2212"
        pretty = _PRETTY.get(nm, nm)
        body = f"{abs(c):.4g}" if nm == "1" else f"{abs(c):.4g}\u00b7{pretty}"
        terms.append(f"{sign} {body}")
    rhs = " ".join(terms)
    if rhs.startswith("+ "):
        rhs = rhs[2:]
    elif rhs.startswith("\u2212 "):
        rhs = "\u2212" + rhs[2:]
    return f"{tsym} = {rhs}    [R\u00b2={r2:.3f}, holdout={holdout:.3f}]"


# ---------------------------------------------------------------------------
# collinearity report (reporting only)
# ---------------------------------------------------------------------------

def vif_report(names, Theta, out_lines, rng, cap=50000):
    sub = Theta[rng.choice(Theta.shape[0], min(cap, Theta.shape[0]),
                           replace=False)]
    out_lines.append("    VIF (variance inflation; >10 strong collinearity):")
    for j, nm in enumerate(names):
        if nm == "1":
            continue
        others = [k for k in range(Theta.shape[1]) if k != j]
        Xo, yj = sub[:, others], sub[:, j]
        g = np.all(np.isfinite(Xo), axis=1) & np.isfinite(yj)
        if g.sum() < 100:
            continue
        beta, *_ = np.linalg.lstsq(Xo[g], yj[g], rcond=None)
        ss = np.sum((yj[g] - Xo[g] @ beta) ** 2)
        tot = max(np.sum((yj[g] - yj[g].mean()) ** 2), 1e-12)
        vif = 1.0 / max(1.0 - (1.0 - ss / tot), 1e-6)
        flag = "  <-- collinear" if vif > 10 else ""
        out_lines.append(f"      {nm:>8}: VIF={vif:8.1f}{flag}")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _windows(data, key, fz):
    w = {"full": np.ones(len(data[key]["y"]), dtype=bool)}
    if fz is not None and fz >= 3:
        w["pre-freeze"] = data[key]["t_index"] < fz
        w["post-freeze"] = data[key]["t_index"] >= fz
    return w


def run(args):
    matplotlib.rcParams.update(PAPER_RC)
    rng = np.random.default_rng(args.seed)
    op = load_op(Path(args.op_path))
    op, _, _ = ensure_theta_smooth(op, args.theta_smooth_w)
    fz = detect_freeze_frame(op.get("n_dislocations"))
    mu = float(op["mu"])
    k1 = float(np.sqrt(max(1.0 - mu ** 2, 1e-9))); k2 = max(mu, 1e-9)
    lam_x, lam_y = 2 * np.pi / k1, 2 * np.pi / k2
    hpx = np.pi / k2                       # pi / sin(alpha) = pi / mu, the base unit
    print(f"loaded {Path(args.op_path).name}  mu={mu}  freeze={fz}  "
          f"pi/mu={hpx:.2f} (lam_y={lam_y:.2f})")

    out = Path(args.out_root) if args.out_root else (
        _HERE / "results" / "time_sindy_riesz_v2" / "mode_persistence" / Path(args.op_path).stem)
    out.mkdir(parents=True, exist_ok=True)

    drop, y_cap, x_trim, info = compute_hull_domain(
        op, margin=args.margin_units * hpx)
    print(f"  bowtie hull: |y-core| <= {y_cap:.1f} "
          f"({y_cap / hpx:.1f} units of pi/mu), erosion={info['erosion_px']}px, "
          f"x_trim={x_trim:.1f}")

    # eroded working copy: NaN the margin band so build_time_samples' finite
    # filter excludes the ragged wedge edge (all theta-derived columns go NaN
    # there and their rows are dropped).
    op_work = dict(op)
    for fld in ("theta_smooth", "theta"):
        if fld in op:
            arr = np.array(op[fld], dtype=float)
            arr[drop] = np.nan
            op_work[fld] = arr

    u3d = _resolve_u(op, args.op_path, args.u_from)
    targets = ["dA_dt", "dw_dt", "dth_dt"]

    lines = [f"# {Path(args.op_path).name}   mu={mu}   freeze={fz}",
             f"# bowtie hull |y-core|<={y_cap:.2f}; core holes in units of "
             f"pi/mu={hpx:.3f}", ""]
    summary = {"op_path": str(args.op_path), "mu": mu, "freeze_frame": fz,
               "hull": info, "pi_over_mu": hpx, "lam_x": lam_x, "lam_y": lam_y,
               "domains": {}}

    for hole_u in args.hole_units:
        hole = hole_u * hpx
        if hole >= y_cap:
            print(f"  [hole {hole_u:g} pi/mu = {hole:.1f}] >= y_cap "
                  f"{y_cap:.1f}; domain empty, skipped")
            lines.append(f"== core hole {hole_u:g} (pi/mu): SKIPPED "
                         f"(empties the domain) ==\n")
            continue
        a = copy(args)
        a.band_y = float(y_cap); a.x_trim = float(x_trim); a.band_in = float(hole)
        data = build_time_samples(op_work, a)
        tag_h = f"hole{hole_u:g}"
        header = (f"== core hole {hole_u:g} pi/mu (|y-core| >= {hole:.2f}), "
                  f"eroded bowtie otherwise ==")
        print("  " + header)
        lines.append(header)
        summary["domains"][tag_h] = {}

        for key in targets:
            if key not in data:
                continue
            d = data[key]
            for wl, sel in _windows(data, key, fz).items():
                r = fit_window(d, sel, a, rng, f"{key} {wl} [{tag_h}]")
                if r is None:
                    continue
                pde = format_pde(key, r["coefs"], r["r2_full"], r["r2_holdout"])
                print(f"      {pde}")
                lines.append(f"  [{key} | {wl}]  N={r['n']}")
                lines.append(f"    {pde}")
                summary["domains"][tag_h].setdefault(key, {})[wl] = dict(
                    pde=pde, selected=r["coefs"], boot_std=r["errs"],
                    r2_full=r["r2_full"], r2_holdout=r["r2_holdout"], n=r["n"])

                if hole_u == 0:
                    frame = _pick_frame(data, key, sel)
                    coef = r["res"]["coef_full"]; names = d["names"]
                    ftag = f"{key}_{wl}"
                    fig_reconstruction(op_work, data, key, sel, coef, names,
                                       frame, out / f"reconstruction_{ftag}.png", a)
                    fig_feature_overlap(op_work, data, key, frame,
                                        out / f"feature_overlap_{ftag}.png", a)
                    fig_query_points(op_work, key, frame,
                                     out / f"query_points_{ftag}.png", a, u3d=u3d)
                    if wl == "full":
                        sub = rng.choice(d["Theta"].shape[0],
                                         min(50000, d["Theta"].shape[0]),
                                         replace=False)
                        fig_library_correlation(
                            names, d["Theta"][sub],
                            out / f"library_correlation_{key}.png")
                        vif_report(names, d["Theta"], lines, rng)
        lines.append("")

    (out / "pde_table.txt").write_text("\n".join(lines))
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  wrote {out / 'pde_table.txt'}")
    print(f"  wrote {out / 'summary.json'}")
    print(f"done -> {out}")


def parse_cli():
    d = _Args(op_path="")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("op_path")
    p.add_argument("--hole_units", type=float, nargs="+", default=[0, 1, 2, 3],
                   help="core-hole sizes in units of pi/mu (= pi/sin alpha)")
    p.add_argument("--margin_units", type=float, default=1.0,
                   help="margin held inside the theta hull, in units of pi/mu")
    p.add_argument("--u_from", default=None,
                   help="file/dir of raw SH .npz for the query-points underlay")
    p.add_argument("--theta_smooth_w", type=float, default=d.theta_smooth_w)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--out_root", default=None)
    a = p.parse_args()
    import dataclasses
    for f in dataclasses.fields(_Args):
        if not hasattr(a, f.name):
            setattr(a, f.name, getattr(d, f.name))
    # band_y / x_trim / band_in are set per-domain in run(); seed the rest
    return a


if __name__ == "__main__":
    run(parse_cli())