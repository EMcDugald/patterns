"""large_sweep_charge_circ_v12.py  v1.2  (26 Oct 2026)   director charge, circulation and amplitude dips
on Ed's large sweep

Plain terms. One command, one dataset: the zigzag sweep of 24 Oct
(data/sh_pgb_zigzag/1024_large_mu_sweep/raw; mu 0.40 to 0.98 in steps of
0.01, relaxed to t = 200, frames saved at t = 0, 100 and 200). Every file is
surveyed at t = 200 (the result) and at t = 100 (the stationarity check) by
pgb_charge_circ_survey_v12.survey_file, imported, so the measures, the options
and the regime rule (charge, then the circulation at the on-line dips) are
exactly those of pgb_charge_circ_survey_v12.py v1.2. This script measures
nothing of its own; it runs the survey twice, puts the two times side by
side, and draws the set's figures from the t = 200 pass.

Outputs under --outdir (default results/large_sweep_charge_circ/, anchored
to this script's folder when relative):
  t200/ and t100/: per file <stem>_f<frame>/ (markers.json, columns.csv,
    circ_map.npz, fig_domain.png, fig_markers.png) and survey_table.csv,
    survey_table.md for that time;
  t200/: the set figures (fig_montage.png, fig_montage_all.pdf,
    fig_positions_vs_mu.png, fig_magnitudes_vs_mu.png, fig_energy_vs_mu.png);
  stationarity.csv: per mu, the regime and the main measures at t = 100 and
    t = 200 and their change, flagged where the regime changed, a count
    changed, a measure moved by more than --move_tol (relative), a distance by
    more than --move_tol_L, or a circulation by more than --move_tol_turn;
  report.txt (both passes and the stationarity summary), config.json.

The t = 100 pass skips the per-file figures by default (--figs100 1 draws
them). --loop_lam, --with_winding, --with_rowdemod and --show_beat_node are
passed to the survey (all off by default).

Python 3.8 compatible. Imports pgb_charge_circ_survey_v12 (v1.2) from this
folder, which imports pgb_marker_survey (v1.2); the frozen method layer is
imported through them and never edited.

Changelog
  v1.2  (new file name, v1.1 kept) on pgb_charge_circ_survey_v12: the dips'
        lowest points and the circulation-map summary pass through; the
        stationarity comparison adds map_circ_max_closed and the off-line
        lowest points' distance from the line.
  v1.1  (new file name, v1.0 kept) on pgb_charge_circ_survey_v11: the width
        scan, the circulation closure and the band minimum of mg A pass
        through; the stationarity comparison adds A_band_min, its distance
        from the line, the open-loop fractions and the charge at sigma_det
        0.2.
  v1.0  from large_sweep_markers v1.0: the survey is pgb_charge_circ_survey;
        the stationarity comparison follows the circulation (median and max at
        the dips, tolerance --move_tol_turn), the dip offsets and depths, and
        the dip-to-V distance, in place of the winding counts. Sandbox check
        on stand-in SH frames in the sweep's layout (not of record).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgb_charge_circ_survey_v12 as PCS                                        # noqa: E402

VERSION = "large_sweep_charge_circ v1.2 (on %s)" % PCS.VERSION
COMPARE = [("charge_abs_per_L", "rel"), ("n_dips_on", "count"), ("n_dips_off", "count"), ("n_dips_on_circ_hi", "count"),
           ("V_offline_med_L", "L"), ("X_offline_med_L", "L"), ("XX_sep_med_L", "L"), ("VX_sep_med_L", "L"),
           ("dips_off_med_L", "L"), ("dip_off_to_V_med_L", "L"), ("dip_depth_off_med", "rel"), ("dip_depth_on_med", "rel"),
           ("dip_circ_on_med", "turn"), ("dip_circ_off_med", "turn"), ("dip_circ_off_max", "turn"),
           ("A_at_V_med", "rel"), ("A_line_min", "rel"), ("A_band_min", "rel"), ("A_band_min_off_L", "L"),
           ("dip_open_frac_on", "turn"), ("dip_open_frac_off", "turn"), ("charge_abs_per_L_sd2", "rel"), ("map_circ_max_closed", "turn"),
           ("dip_min_off_off_med_L", "L"), ("E_GB", "rel")]


def survey_pass(files, frame, args, out, figs):
    sargs = PCS.default_args()
    for k, v in vars(args).items():
        if hasattr(sargs, k):
            setattr(sargs, k, v)
    sargs.frame = frame
    sargs.tag_frame = 1
    out.mkdir(parents=True, exist_ok=True)
    lines = [PCS.header_line(sargs), ""]
    rows, Gs = [], []
    for f in files:
        try:
            row, G = PCS.survey_file(f, sargs, out, lines, quiet_figs=not figs)
            rows.append(row); Gs.append(G)
        except Exception as exc:
            lines.append("FLAG: %s failed: %r" % (f, exc))
        print(lines[-1])
    order = np.argsort([r["mu"] for r in rows])
    rows = [rows[i] for i in order]; Gs = [Gs[i] for i in order]
    if rows:
        PCS.write_tables(rows, out)
    return rows, Gs, lines, sargs


def stationarity(r200, r100, args):
    by100 = {round(r["mu"], 4): r for r in r100}
    table, flagged = [], []
    for r in r200:
        q = by100.get(round(r["mu"], 4))
        rec = dict(mu=r["mu"], t_late=r["t"], regime_late=r["regime"])
        if q is None:
            rec.update(t_early=float("nan"), regime_early="missing", moved="no t = 100 pass"); table.append(rec); flagged.append(rec); continue
        rec.update(t_early=q["t"], regime_early=q["regime"])
        moved = []
        if q["regime"] != r["regime"]:
            moved.append("regime %s -> %s" % (q["regime"], r["regime"]))
        for k, kind in COMPARE:
            a, b = q.get(k, float("nan")), r.get(k, float("nan"))
            rec[k + "_early"] = a; rec[k + "_late"] = b
            if not (np.isfinite(a) and np.isfinite(b)):
                if np.isfinite(a) != np.isfinite(b):
                    moved.append("%s %s -> %s" % (k, a, b))
                continue
            if kind == "count" and abs(b - a) >= 1:
                moved.append("%s %g -> %g" % (k, a, b))
            elif kind == "L" and abs(b - a) > args.move_tol_L:
                moved.append("%s %.3f -> %.3f L" % (k, a, b))
            elif kind == "turn" and abs(b - a) > args.move_tol_turn:
                moved.append("%s %.3f -> %.3f turn" % (k, a, b))
            elif kind == "rel" and abs(b - a) > args.move_tol * max(abs(a), 1e-12):
                moved.append("%s %.4g -> %.4g" % (k, a, b))
        rec["moved"] = "; ".join(moved)
        table.append(rec)
        if moved:
            flagged.append(rec)
    return table, flagged


def main(args):
    out = Path(args.outdir)
    if not out.is_absolute():
        out = HERE / out
    out.mkdir(parents=True, exist_ok=True)
    raw = Path(args.raw)
    if not raw.is_absolute():
        raw = HERE / raw
    files = sorted(str(p) for p in raw.glob("*.npz"))
    if not files:
        print("no .npz files in %s" % raw); return 1
    with open(out / "config.json", "w") as fh:
        json.dump(dict(vars(args), version=VERSION, n_files=len(files)), fh, indent=2, default=str)
    t0 = time.time()
    r200, G200, l200, sargs = survey_pass(files, args.frame_late, args, out / "t200", figs=True)
    r100, _, l100, _ = survey_pass(files, args.frame_early, args, out / "t100", figs=bool(args.figs100))
    lines = [VERSION, "data %s: %d files" % (raw, len(files)), ""]
    if r200:
        has_E, sel = PCS.set_figures(r200, G200, sargs, out / "t200")
        lines.append("t = %s: regimes by mu: %s" % (r200[0]["t"], ", ".join("%.2f %s" % (r["mu"], r["regime"]) for r in r200)))
        if not has_E:
            lines.append("FLAG: no energy found; the energy figure was not drawn")
    if r100:
        lines.append("t = %s: regimes by mu: %s" % (r100[0]["t"], ", ".join("%.2f %s" % (r["mu"], r["regime"]) for r in r100)))
    table, flagged = stationarity(r200, r100, args)
    if table:
        keys = []
        for rec in table:
            for k in rec:
                if k not in keys:
                    keys.append(k)
        with open(out / "stationarity.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(table)
    lines.append("")
    lines.append("stationarity (t = 100 against t = 200; tolerances %.0f%% relative, %.2g L, %.2g turn): %d of %d angles moved"
                 % (100 * args.move_tol, args.move_tol_L, args.move_tol_turn, len(flagged), len(table)))
    for rec in flagged:
        lines.append("   mu %.2f: %s" % (rec["mu"], rec["moved"]))
    lines.append("")
    lines.append("elapsed %.1f s" % (time.time() - t0))
    lines += ["", "==== t = 200 pass ===="] + l200 + ["", "==== t = 100 pass ===="] + l100
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:8 + len(flagged)]))
    print("outputs ->", out)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    raw = str(Path("data") / "sh_pgb_zigzag" / "1024_large_mu_sweep" / "raw")
    outdir = str(Path("results") / "large_sweep_charge_circ_v12")
    frame_late = -1                # the last frame (t = 200)
    frame_early = 1                # the middle frame (t = 100)
    figs100 = 0                    # 1: per-file figures for the t = 100 pass too
    loop_lam = 0.0                 # > 0: dip loops of this many stripe wavelengths (the loop-size control)
    wind_hi = 0.9                  # turns: the regime threshold for on-line dips
    move_tol = 0.10                # stationarity: relative change flagged above this
    move_tol_L = 0.05              # stationarity: change of a distance flagged above this, L
    move_tol_turn = 0.10           # stationarity: change of a circulation flagged above this, turns
    montage_step = 0.02            # the montage: every montage_step in mu, plus the angles beside a regime change
    sigma_scan = "0.15,0.2,0.25,0.3,0.35,0.4"   # director-charge widths for the width scan
    with_winding = 0               # options passed to the survey, off by default
    with_rowdemod = 0
    show_beat_node = 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in vars(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    sys.exit(main(p.parse_args()))
