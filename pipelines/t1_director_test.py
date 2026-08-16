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

from op_extract.t1_common import (
    FIELDS, SOURCES, SOURCE_COLORS, align_to_truth_upper, build_seed_contour,
    defect_index_figure, diagnostic_figure, edge_taper, k_from_theta,
    kfield_figure, landing_figure, planewave_error, profiles_figure,
    riesz_k_field, riesz_panels_figure, row_ladder_figure, run_one,
    score_core, sizing_figure, stage, surface_figures, timings_report,
    timings_reset)


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
    timings_reset()
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

    with stage("fig: sizing"):
        sizing_figure(fld, os.path.join(
            outdir, f"sizing_{_tag(field, mu, sigma)}.png"))
    if args.riesz_panels:
        p = os.path.join(outdir, f"riesz_panels_{_tag(field, mu, sigma)}.png")
        with stage("fig: riesz panels"):
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

    ladder, landings, thetas = {}, {}, {}
    for src in args.sources:
        try:
            phi, res = run_one(fld, src, seed_bdry, args.n_seeds, sigma,
                               args.ds, ramp, use_sym=args.symmetric,
                               gauge=args.gauge, half=args.half,
                               sigma_riesz=args.sigma_riesz,
                               auto_cfg={"sigma_det": args.sigma_det,
                                         "sigma_small": args.sigma_small,
                                         "sigma_large": args.sigma_large})
        except Exception as exc:
            report(f"  {src:>8}   FAILED: {exc}")
            continue
        landings[src] = res["core_landings"]
        aw = res.get("auto_width")
        if aw is not None:
            report(f"  {src:>8}   defect index at sigma_det="
                   f"{aw['sigma_det']:.2f}: sum|q|={aw['sum_abs']}, "
                   f"{aw['n_dislocations']} dislocations"
                   + (f" (true nd={fld['nd']})" if fld["nd"] else
                      " (defect free field)")
                   + f" -> sigma_dir={aw['sigma_chosen']:.4f}")
            if args.defect_figs:
                pdf_ = os.path.join(
                    outdir, f"defects_{_tag(field, mu, sigma)}.png")
                with stage("fig: defect index"):
                    defect_index_figure(fld, aw, pdf_)
                report(f"  {' ':>8}   defect figure: {pdf_}")
        report(f"  {src:>8}   {res.get('n_line_points', 0)} points across all "
               f"coordinate lines (this is what rebuild_grid triangulates)")
        frac = res.get("frac_reaching_core", np.nan)
        if frac < 0.9:
            report(f"  {src:>8}   [warn] only {100 * frac:.0f} percent of "
                   f"lines reach the core")
        al = align_to_truth_upper(phi, fld)
        # the k-field block differentiates the ALIGNED phase, so the
        # gauge scale a (near -1) does not bias |k|
        thetas[src] = al[0]

        scs = []
        for cy in rows:
            with stage("scoring"):
                sc = score_core(phi, fld, x_trim=args.x_trim,
                                thresh_frac=args.thresh_frac,
                                core_y=(None if cy <= 0 else cy),
                                phi_aligned=al)
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
            with stage("fig: diagnostic"):
                diagnostic_figure(fld, src, res, phi, p, sc=scs[0])
            report(f"  {' ':>8}   diagnostic: {p}")
        if args.surfaces and src in args.surface_sources:
            p3 = os.path.join(outdir,
                              f"surface3d_{_tag(field, mu, sigma)}_{src}.png")
            p2 = os.path.join(outdir,
                              f"surface2d_{_tag(field, mu, sigma)}_{src}.png")
            with stage("fig: surfaces (3d + 2d)"):
                surface_figures(fld, src, phi, p3, p2,
                                target=args.surface_target)
            report(f"  {' ':>8}   surfaces: {p3}")
            report(f"  {' ':>8}             {p2}")

    if args.kfields:
        kfield_block(fld, ladder, thetas, args, report, outdir,
                     _tag(field, mu, sigma))

    if args.riesz_sweep and sigma == args.sigmas[0]:
        riesz_width_sweep(fld, seed_bdry, ramp, args, report, outdir, sigma)

    if ladder:
        t = _tag(field, mu, sigma)
        with stage("fig: profiles + zoom + landings"):
            profiles_figure(fld, ladder,
                            os.path.join(outdir, f"profiles_{t}.png"))
            zc = float(fld["xd"][len(fld["xd"]) // 2]) if fld["nd"] else 0.0
            profiles_figure(fld, ladder,
                            os.path.join(outdir, f"profiles_zoom_{t}.png"),
                            zoom=zc, n_spacings=args.zoom_spacings)
            landing_figure(fld, landings,
                           os.path.join(outdir, f"landings_{t}.png"))
            if field == "pn" and len(rows) > 1:
                row_ladder_figure(fld, ladder,
                                  os.path.join(outdir, f"row_ladder_{t}.png"))
        report(f"     figures written with tag {t}")

    timings_report(report)
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


def kfield_block(fld, ladder, thetas, args, report, outdir, tag):
    """
    Two k fields per requested source, scored against exact and against
    each other. The direct one uses the Riesz direction and the averaged
    magnitude ratio; the second differentiates the reconstructed phase.
    Their difference map is the trust check that survives to real data,
    where the exact columns of this table do not exist.
    """
    report("")
    report(f"  k fields (band: upper half; angle errors are mod pi)")
    report(f"  {'source':>10} {'route':>10} {'ang rms':>8} {'ang p98':>8} "
           f"{'|k| rms%':>9} {'|k| p98%':>9} {'mutual med':>11}")
    for src in args.kfield_sources:
        if src not in thetas or thetas[src] is None:
            continue
        scs = ladder.get(src, [])
        if not scs or "rms" not in scs[0]:
            continue
        sig_dir = args.sigma_riesz
        if src == "riesz_auto":
            # reuse the width the auto selector chose for this field
            sig_dir, _ = __import__(
                "op_extract.t1_common_v5", fromlist=["select_sigma_riesz"]
            ).select_sigma_riesz(fld, sigma_det=args.sigma_det,
                                 sigma_small=args.sigma_small,
                                 sigma_large=args.sigma_large)
        with stage("k fields"):
            kd = riesz_k_field(fld["u"], fld["x"], fld["y"], fld["Lx"],
                               fld["Ly"], sigma_dir=sig_dir)
            kg = k_from_theta(thetas[src], fld["x"], fld["y"], fld["Lx"],
                              fld["Ly"], smooth=args.kgrad_smooth)
        p = os.path.join(outdir, f"kfields_{tag}_{src}.png")
        with stage("fig: k fields"):
            stats = kfield_figure(fld, kd, kg, p, source=src)
        sd, sg = stats["score_direct"], stats["score_grad"]
        sdi = stats["score_direct_interior"]
        cons = stats["consistency"]
        report(f"  {src:>10} {'direct':>10} "
               f"{_fmt(sd.get('ang_rms_deg'), 8, 2)} "
               f"{_fmt(sd.get('ang_p98_deg'), 8, 2)} "
               f"{_fmt(100 * sd.get('kmag_rel_rms', np.nan), 9, 2)} "
               f"{_fmt(100 * sd.get('kmag_rel_p98', np.nan), 9, 2)} "
               f"{_fmt(cons.get('dang_med_abs'), 11, 2)}")
        report(f"  {src:>10} {'d.interior':>10} "
               f"{_fmt(sdi.get('ang_rms_deg'), 8, 2)} "
               f"{_fmt(sdi.get('ang_p98_deg'), 8, 2)} "
               f"{_fmt(100 * sdi.get('kmag_rel_rms', np.nan), 9, 2)} "
               f"{_fmt(100 * sdi.get('kmag_rel_p98', np.nan), 9, 2)}")
        report(f"  {src:>10} {'grad(theta)':>10} "
               f"{_fmt(sg.get('ang_rms_deg'), 8, 2)} "
               f"{_fmt(sg.get('ang_p98_deg'), 8, 2)} "
               f"{_fmt(100 * sg.get('kmag_rel_rms', np.nan), 9, 2)} "
               f"{_fmt(100 * sg.get('kmag_rel_p98', np.nan), 9, 2)}")
        report(f"  {' ':>10}   figure: {p}")


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
    outdir = "out_t1"
    fields = ["knee", "pn"]
    #mu = [0.85, 0.90, 0.95]
    #mu = [0.75, 0.80]
    mu = [.90]
    sigmas = [np.pi / 2]    # uHu averaging scale, swept
    sigma_riesz = 1.0             # Riesz width, held fixed
    riesz_sweep = None             # e.g. 0.2 0.4 0.8 1.5708 2.5 3.1416
    sources = ["exact", "uhu", "riesz", "riesz_auto"]
    # riesz_auto: the defect index at sigma_det picks the width. The
    # endpoints are the measured optima, not tuned values: small was
    # clean on the knee at every mu, large was best on PN.
    sigma_det = 0.4
    sigma_small = 0.4
    sigma_large = float(np.pi)
    defect_figs = True
    # k fields: direct Riesz route and grad of the reconstructed phase
    kfields = True
    kfield_sources = ["riesz_auto"]
    kgrad_smooth = 1.0
    seed = "exact"                  # or "trapezoid"
    ds = 0.30
    n_seeds = 128
    N = 512
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
    p.add_argument("--sigma-det", type=float, default=_Args.sigma_det,
                   help="width of the defect-detection director")
    p.add_argument("--sigma-small", type=float, default=_Args.sigma_small,
                   help="riesz_auto width when no defects are found")
    p.add_argument("--sigma-large", type=float, default=_Args.sigma_large,
                   help="riesz_auto width when defects are found")
    p.add_argument("--defect-figs", dest="defect_figs",
                   action="store_true", default=_Args.defect_figs)
    p.add_argument("--no-defect-figs", dest="defect_figs",
                   action="store_false")
    p.add_argument("--kfields", dest="kfields", action="store_true",
                   default=_Args.kfields)
    p.add_argument("--no-kfields", dest="kfields", action="store_false")
    p.add_argument("--kfield-sources", nargs="+",
                   default=_Args.kfield_sources, choices=list(SOURCES))
    p.add_argument("--kgrad-smooth", type=float, default=_Args.kgrad_smooth,
                   help="NaN-aware smoothing of phi before grad")
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
    p.add_argument("--diagnose", dest="diagnose", action="store_true",
                   default=_Args.diagnose)
    p.add_argument("--no-diagnose", dest="diagnose", action="store_false")
    p.add_argument("--surfaces", dest="surfaces", action="store_true",
                   default=_Args.surfaces)
    p.add_argument("--no-surfaces", dest="surfaces", action="store_false")
    p.add_argument("--surface-sources", nargs="+",
                   default=_Args.surface_sources, choices=list(SOURCES))
    p.add_argument("--surface-target", type=int, default=_Args.surface_target)
    p.add_argument("--riesz-panels", dest="riesz_panels",
                   action="store_true", default=_Args.riesz_panels)
    p.add_argument("--no-riesz-panels", dest="riesz_panels", action="store_false")
    main(p.parse_args())