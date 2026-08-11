# pipelines/t1_director_test.py
"""
T1: which director does the phase pipeline need, on both analytic fields.

Give it a list of mu and it runs the knee and the PN multi-defect solution
through the same pipeline with the same three directors, scores them with
the same metrics, and writes the figures.

    python t1_director_test.py
    python t1_director_test.py --mu 0.85 0.90 0.95 --sigmas 1.5708 3.1416
    python t1_director_test.py --fields pn --mu 0.90 --diagnose --surfaces
    python t1_director_test.py --fields knee --mu 0.76 0.79 --nx-periods 14

The scoring is described in src/op_extract/t1_common.py. The short version:
every metric is compared against the exact field evaluated on the SAME row,
so a reconstruction is never scored against a target the truth itself does
not meet. The headline is the sawtooth peak to peak against the truth's own
value, where an excess is invented structure and a deficit is suppressed
structure, which covers the knee failure and the PN failure with one number.

Rows. On PN the staircase only exists in the y to 0 limit, and the true step
is sqrt(2 y) wide, so the field is scored on a ladder of rows and the
resolution trend is visible in every run. The knee is defect free and gets
its core row only.

Sigma. The same sigma feeds the uHu averaging and the Riesz smoothing, so
looping over --sigmas sweeps both at once.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from op_extract.t1_common_v2 import (
    FIELDS, SOURCES, SOURCE_COLORS, align_to_truth_upper, build_seed_contour,
    diagnostic_figure, edge_taper, landing_figure, planewave_error,
    profiles_figure, riesz_panels_figure, row_ladder_figure, run_one,
    score_core, sizing_figure, surface_figures)


def _fmt(v, w=8, p=4):
    """Print a dash rather than nan, so a missing measurement does not look
    like a failed one."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return f"{'-':>{w}}"
    return f"{v:>{w}.{p}f}"


def _tag(field, mu, sigma):
    return f"{field}_mu{mu:.3f}_sig{sigma:.3f}".replace(".", "p")


def run_field(field, mu, sigma, args, report, outdir):
    """One (field, mu, sigma). Returns the ladder dict for the summary."""
    builder = FIELDS[field]
    if field == "knee":
        fld = builder(mu, n_periods=args.n_periods,
                      n_spacings=args.n_spacings, N=args.N, ds=args.ds,
                      nx_periods=args.nx_periods, ly_factor=args.ly_factor)
        rows = [0.0]
    else:
        fld = builder(mu, n_periods=args.n_periods, nd=args.nd, N=args.N,
                      ly_factor=args.ly_factor, n_spacings=args.n_spacings)
        rows = list(args.core_rows)

    dx, dy = fld["x"][1] - fld["x"][0], fld["y"][1] - fld["y"][0]
    ex, ey, yj = planewave_error(fld)
    report("")
    report(f"  {field}  mu={mu:.3f}  sigma_uhu={sigma:.4f}  "
           f"sigma_riesz={args.sigma_riesz:.4f}")
    report(f"     k1={fld['k1_ff']:.4f}  k2_ff={fld['k2_ff']:.4f}  "
           f"pi/k1={fld['Lsp']:.3f}  nd={fld['nd']}  y_top={fld['y_top']:.1f}")
    report(f"     box Lx={fld['Lx']:.1f} ({fld['Lx'] / (2 * np.pi / fld['k1_ff']):.1f} "
           f"horizontal periods)  Ly={fld['Ly']:.1f}  dx={dx:.3f} dy={dy:.3f}"
           f"  core row y={fld['y'][fld['iy_core']]:+.4f}")
    report(f"     contour span {fld['contour_span']:.1f}, drift "
           f"{fld['drift']:.1f}, requested coverage {fld['cover']:.1f}")
    report(f"     far-field error at y={yj:.1f}: {100 * ex:.2f}% in theta_x, "
           f"{100 * ey:.2f}% in theta_y"
           + ("   [warn] nd may be too small" if max(ex, ey) > 0.02 else ""))
    report(f"     exact k from: {fld['k_source']}")

    sizing_figure(fld, os.path.join(outdir, f"sizing_{_tag(field, mu, sigma)}.png"))
    if args.riesz_panels:
        p = os.path.join(outdir, f"riesz_panels_{_tag(field, mu, sigma)}.png")
        riesz_panels_figure(fld, p, sigma_dir=args.sigma_riesz)
        report(f"     riesz panels: {p}")

    ramp = edge_taper(fld["x"], fld["y"], frac=args.taper)
    seed_bdry = build_seed_contour(fld, args.seed, args.ds,
                                   cover_frac=args.cover_frac, report=report)

    report("")
    report(f"  {'source':>8} {'y row':>7} {'cells':>6} {'cover':>6} "
           f"{'reach':>6} {'a':>7} {'rms':>8} {'max|r|':>8} {'saw/pi':>7} "
           f"{'(truth)':>8} {'local':>6} {'(truth)':>8} {'edges':>6} "
           f"{'spacing':>8} {'pi/k1':>7} {'d2':>7} {'gapmax':>7}")

    ladder, landings = {}, {}
    for src in args.sources:
        try:
            phi, res = run_one(fld, src, seed_bdry, args.n_seeds, sigma,
                               args.ds, ramp, use_sym=args.symmetric,
                               gauge=args.gauge, half=args.half,
                               sigma_riesz=args.sigma_riesz)
        except Exception as exc:
            report(f"  {src:>8}   FAILED: {exc}")
            continue
        landings[src] = res["core_landings"]
        frac = res.get("frac_reaching_core", np.nan)
        if frac < 0.9:
            report(f"  {src:>8}   [warn] only {100 * frac:.0f} percent of "
                   f"lines reach the core")
        al = align_to_truth_upper(phi, fld)

        scs = []
        for cy in rows:
            sc = score_core(phi, fld, x_trim=args.x_trim,
                            thresh_frac=args.thresh_frac,
                            core_y=(None if cy <= 0 else cy), phi_aligned=al)
            scs.append(sc)
            if "rms" not in sc:
                report(f"  {src:>8} {sc['y_core']:>7.2f}   too few valid "
                       f"points on this row ({sc['n_ok']})")
                continue
            ref = sc["ref"]
            report(f"  {src:>8} {sc['y_core']:>7.2f} "
                   f"{sc['step_width'] / dx:>6.1f} {sc['coverage']:>6.2f} "
                   f"{frac:>6.2f} {_fmt(sc['a_align'], 7, 3)} "
                   f"{_fmt(sc['rms'])} {_fmt(sc['max_abs'])} "
                   f"{_fmt(sc.get('sawtooth_ptp', np.nan) / np.pi, 7, 2)} "
                   f"{_fmt(ref.get('sawtooth_ptp', np.nan) / np.pi, 8, 2)} "
                   f"{_fmt(sc.get('localization'), 6, 2)} "
                   f"{_fmt(ref.get('localization'), 8, 2)} "
                   f"{sc['n_edges']:>6} "
                   f"{_fmt(sc.get('spacing_mean'), 8, 3)} "
                   f"{fld['Lsp']:>7.3f} {_fmt(sc.get('d2_rms'), 7, 3)} "
                   f"{_fmt(res.get('landing_gap_max'), 7, 2)}")
            pl = sc["plateaus"]
            if pl["n_gaps"]:
                report(f"  {' ':>8}   [note] {pl['n_gaps']} sampling gaps on "
                       f"this row, {100 * pl['gap_frac']:.0f}% not covered")
        ladder[src] = scs

        if args.diagnose and scs and "rms" in scs[0]:
            p = os.path.join(outdir,
                             f"diag_{_tag(field, mu, sigma)}_{src}.png")
            diagnostic_figure(fld, src, res, phi, p, sc=scs[0])
            report(f"  {' ':>8}   diagnostic: {p}")
        if args.surfaces and src in args.surface_sources:
            p3 = os.path.join(outdir,
                              f"surface3d_{_tag(field, mu, sigma)}_{src}.png")
            p2 = os.path.join(outdir,
                              f"surface2d_{_tag(field, mu, sigma)}_{src}.png")
            surface_figures(fld, src, phi, p3, p2, target=args.surface_target)
            report(f"  {' ':>8}   surfaces: {p3}")
            report(f"  {' ':>8}             {p2}")

    if args.riesz_sweep and sigma == args.sigmas[0]:
        riesz_width_sweep(fld, seed_bdry, ramp, args, report, outdir, sigma)

    if ladder:
        t = _tag(field, mu, sigma)
        profiles_figure(fld, ladder, os.path.join(outdir, f"profiles_{t}.png"))
        zc = float(fld["xd"][len(fld["xd"]) // 2]) if fld["nd"] else 0.0
        profiles_figure(fld, ladder,
                        os.path.join(outdir, f"profiles_zoom_{t}.png"),
                        zoom=zc, n_spacings=args.zoom_spacings)
        landing_figure(fld, landings, os.path.join(outdir, f"landings_{t}.png"))
        if field == "pn" and len(rows) > 1:
            row_ladder_figure(fld, ladder,
                              os.path.join(outdir, f"row_ladder_{t}.png"))
        report(f"     figures written with tag {t}")
    return fld, ladder


def riesz_width_sweep(fld, seed_bdry, ramp, args, report, outdir, sigma):
    """
    Sweep the Riesz smoothing width at fixed uHu sigma, so the fixed value
    used everywhere else is chosen from data rather than asserted. The knee
    sweep in the earlier harness put the usable shoulder near 0.4 to 0.8 and
    showed degradation by pi/2, which is where the default comes from; this
    re-measures it on whichever field is being run.
    """
    report("")
    report(f"  Riesz width sweep at sigma_uhu={sigma:.4f}")
    report(f"  {'sigma_dir':>10} {'cover':>6} {'a':>7} {'rms':>8} "
           f"{'max|r|':>8} {'saw/pi':>7} {'(truth)':>8} {'edges':>6} "
           f"{'d2':>7}")
    curves = []
    for sg in args.riesz_sweep:
        try:
            phi, res = run_one(fld, "riesz", seed_bdry, args.n_seeds, sigma,
                               args.ds, ramp, use_sym=args.symmetric,
                               gauge=args.gauge, half=args.half,
                               sigma_riesz=sg)
        except Exception as exc:
            report(f"  {sg:>10.3f}   FAILED: {exc}")
            continue
        sc = score_core(phi, fld, x_trim=args.x_trim,
                        thresh_frac=args.thresh_frac)
        if "rms" not in sc:
            report(f"  {sg:>10.3f}   too few valid points")
            continue
        ref = sc["ref"]
        curves.append((sg, sc))
        report(f"  {sg:>10.3f} {sc['coverage']:>6.2f} "
               f"{_fmt(sc['a_align'], 7, 3)} {_fmt(sc['rms'])} "
               f"{_fmt(sc['max_abs'])} "
               f"{_fmt(sc.get('sawtooth_ptp', np.nan) / np.pi, 7, 2)} "
               f"{_fmt(ref.get('sawtooth_ptp', np.nan) / np.pi, 8, 2)} "
               f"{sc['n_edges']:>6} {_fmt(sc.get('d2_rms'), 7, 3)}")
    if not curves:
        return
    fig, ax = plt.subplots(2, 1, figsize=(10.0, 7.5))
    for sg, sc in curves:
        ax[0].plot(sc["x"], sc["phi"] / np.pi, lw=1.1,
                   label=rf"$\sigma_{{dir}}={sg:.2f}$")
    ax[0].plot(fld["x"], fld["theta"][fld["iy_core"]] / np.pi, "k-", lw=1.8,
               label="truth")
    for c in fld["xd"]:
        ax[0].axvline(c, color="0.88", lw=0.6, zorder=0)
    ax[0].set_ylabel(r"core phase / $\pi$")
    ax[0].set_xlabel("x")
    ax[0].legend(fontsize=7, ncol=3)
    sgs = [c[0] for c in curves]
    ax[1].plot(sgs, [c[1]["rms"] for c in curves], "o-", label="rms vs truth")
    ax[1].plot(sgs, [c[1].get("sawtooth_ptp", np.nan) / np.pi
                     for c in curves], "s-", label=r"sawtooth / $\pi$")
    ax[1].axhline(curves[0][1]["ref"].get("sawtooth_ptp", np.nan) / np.pi,
                  color="k", ls="--", lw=1.2, label="truth sawtooth")
    ax[1].set_xlabel(r"$\sigma_{dir}$")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.25)
    fig.suptitle(rf"{fld['field']} $\mu={fld['mu']}$: Riesz width sweep at "
                 rf"$\sigma_{{uhu}}={sigma:.3f}$")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pth = os.path.join(outdir,
                       f"riesz_sweep_{_tag(fld['field'], fld['mu'], sigma)}.png")
    fig.savefig(pth, dpi=150)
    plt.close(fig)
    report(f"     riesz sweep figure: {pth}")


def summary_figure(store, args, path):
    """One panel per (field, mu): the sawtooth excess over the truth, which
    is the number that reads the same way on both fields. Zero is correct,
    positive is invented structure, negative is suppressed structure."""
    keys = [k for k in store if store[k][1]]
    if not keys:
        return
    fig, ax = plt.subplots(1, len(args.sigmas), figsize=(7.0 * len(args.sigmas),
                                                         4.6), squeeze=False)
    for si, sigma in enumerate(args.sigmas):
        a_ = ax[0, si]
        labels, offsets = [], {s: [] for s in args.sources}
        for (field, mu, sg) in sorted(k for k in keys if k[2] == sigma):
            fld, ladder = store[(field, mu, sg)]
            labels.append(f"{field}\n{mu:g}")
            for src in args.sources:
                scs = ladder.get(src, [])
                v = np.nan
                if scs and "rms" in scs[0]:
                    v = (scs[0].get("sawtooth_ptp", np.nan)
                         - scs[0]["ref"].get("sawtooth_ptp", np.nan)) / np.pi
                offsets[src].append(v)
        idx = np.arange(len(labels))
        w = 0.8 / max(len(args.sources), 1)
        for i, src in enumerate(args.sources):
            a_.bar(idx + i * w - 0.4 + w / 2, offsets[src], width=w,
                   color=SOURCE_COLORS.get(src, "C4"), label=src)
        a_.axhline(0.0, color="k", lw=1.0)
        a_.set_xticks(idx)
        a_.set_xticklabels(labels, fontsize=8)
        a_.set_ylabel(r"(sawtooth $-$ truth) / $\pi$")
        a_.set_title(rf"$\sigma={sigma:.3f}$", fontsize=10)
        a_.legend(fontsize=8)
        a_.grid(alpha=0.25, axis="y")
    fig.suptitle("core row: excess structure above the truth. "
                 "positive = invented, negative = suppressed", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    lines = []

    def report(s=""):
        print(s, flush=True)
        lines.append(s)

    report("=" * 100)
    report("T1: reconstructed phase vs exact, by director source, on the "
           "knee and PN")
    report("=" * 100)
    report("  every metric is compared against the exact field on the SAME "
           "row, shown in the (truth) columns")
    report("  sawtooth above the truth means invented structure, below means "
           "suppressed")
    report("  sigma sweeps the uHu averaging only. The Riesz width is a "
           "different kind of parameter and is held fixed;")
    report("  see run_one in t1_common for why.")
    report(f"  fields: {' '.join(args.fields)}   mu: "
           f"{' '.join(f'{m:g}' for m in args.mu)}   sigmas: "
           f"{' '.join(f'{s:.4f}' for s in args.sigmas)} (uHu only; "
           f"Riesz fixed at {args.sigma_riesz:.4f})")
    report(f"  seed contour: {args.seed}   gauge: {args.gauge}   "
           f"seeds: {args.n_seeds}   ds: {args.ds}   N: {args.N}")

    store = {}
    for sigma in args.sigmas:
        for field in args.fields:
            for mu in args.mu:
                try:
                    store[(field, mu, sigma)] = run_field(
                        field, mu, sigma, args, report, args.outdir)
                except Exception as exc:
                    report(f"  {field} mu={mu} sigma={sigma} FAILED: {exc}")

    p = os.path.join(args.outdir, "summary_sawtooth_excess.png")
    summary_figure(store, args, p)
    report("")
    report(f"summary figure: {p}")

    with open(os.path.join(args.outdir, "t1_report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


class _Args:
    outdir = "out_t1_v2"
    fields = ["knee", "pn"]
    mu = [0.85, 0.90, 0.95]
    sigmas = [np.pi / 2, np.pi]    # uHu averaging scale, swept
    sigma_riesz = 0.8              # Riesz width, held fixed
    riesz_sweep = None             # e.g. 0.2 0.4 0.8 1.5708 2.5 3.1416
    sources = ["exact", "uhu", "riesz"]
    seed = "exact"                  # or "trapezoid"
    ds = 0.15
    n_seeds = 256
    N = 768
    n_periods = 8.0                 # vertical periods, sets y_top
    n_spacings = None               # core coverage in units of pi/k1;
                                    # knee defaults to 8, PN to its defect row
    nx_periods = None               # override Lx as a count of 2 pi / k1
    nd = None                       # PN defect count; None uses the rule
    ly_factor = 2.8
    cover_frac = 0.9
    core_rows = [0.0, 2.0, 4.0]     # PN only; the knee gets its core row
    taper = 0.04
    x_trim = 0.15
    thresh_frac = 0.15
    zoom_spacings = 2.5
    symmetric = False
    gauge = "seed_phase"
    half = "orig"
    diagnose = True
    surfaces = True
    surface_sources = ["riesz"]
    surface_target = 180
    riesz_panels = True


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=_Args.outdir)
    p.add_argument("--fields", nargs="+", default=_Args.fields,
                   choices=["knee", "pn"])
    p.add_argument("--mu", type=float, nargs="+", default=_Args.mu)
    p.add_argument("--sigmas", type=float, nargs="+", default=_Args.sigmas,
                   help="uHu averaging scale only; swept")
    p.add_argument("--sigma-riesz", type=float, default=_Args.sigma_riesz,
                   help="Riesz smoothing width, held fixed across --sigmas")
    p.add_argument("--riesz-sweep", type=float, nargs="+",
                   default=_Args.riesz_sweep,
                   help="sweep the Riesz width at the first --sigmas value")
    p.add_argument("--sources", nargs="+", default=_Args.sources,
                   choices=list(SOURCES))
    p.add_argument("--seed", default=_Args.seed,
                   choices=["trapezoid", "exact"])
    p.add_argument("--ds", type=float, default=_Args.ds)
    p.add_argument("--n-seeds", type=int, default=_Args.n_seeds)
    p.add_argument("--N", type=int, default=_Args.N)
    p.add_argument("--n-periods", type=float, default=_Args.n_periods)
    p.add_argument("--n-spacings", type=float, default=_Args.n_spacings)
    p.add_argument("--nx-periods", type=float, default=_Args.nx_periods)
    p.add_argument("--nd", type=int, default=_Args.nd)
    p.add_argument("--ly-factor", type=float, default=_Args.ly_factor)
    p.add_argument("--cover-frac", type=float, default=_Args.cover_frac)
    p.add_argument("--core-rows", type=float, nargs="+",
                   default=_Args.core_rows,
                   help="PN only; 0 means the first row above the core")
    p.add_argument("--taper", type=float, default=_Args.taper)
    p.add_argument("--x-trim", type=float, default=_Args.x_trim)
    p.add_argument("--thresh-frac", type=float, default=_Args.thresh_frac)
    p.add_argument("--zoom-spacings", type=float, default=_Args.zoom_spacings)
    p.add_argument("--symmetric", action="store_true", default=_Args.symmetric)
    p.add_argument("--gauge", default=_Args.gauge,
                   choices=["raw", "zero_origin", "seed_phase"])
    p.add_argument("--half", default=_Args.half, choices=["orig", "stitched"])
    p.add_argument("--diagnose", action="store_true", default=_Args.diagnose)
    p.add_argument("--surfaces", action="store_true", default=_Args.surfaces)
    p.add_argument("--surface-sources", nargs="+",
                   default=_Args.surface_sources, choices=list(SOURCES))
    p.add_argument("--surface-target", type=int, default=_Args.surface_target)
    p.add_argument("--riesz-panels", action="store_true",
                   default=_Args.riesz_panels)
    main(p.parse_args())