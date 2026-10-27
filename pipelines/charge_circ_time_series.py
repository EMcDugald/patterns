"""charge_circ_time_series.py  v1.0  (25 Oct 2026)   director charge, circulation and amplitude dips
frame by frame through the movies

Plain terms. The stationary survey (pgb_charge_circ_survey_v11) labels a
boundary from its last frame. This script runs the same survey on every
frame of each movie (or every --frame_step frames), so the three measures
can be watched while a knee turns into a dislocation. It answers the open
question from the winding-times figure: when the circulation about the dips
first reaches a quarter turn, is that a step toward one full turn, a passing
excursion that falls back, or a loop that has just taken in an odd number of
half-turn cells (its sign of k does not close)?

Per movie, per frame: the director charge per L at sigma_det and
sigma_det2; the dips on and off the line, their offsets, depths, and distance
to the nearest V cell; the deepest mg A in the band and its distance from
the line; |Gamma| about every dip (median, mean and largest, on and off the
line) and whether the sign of k closes on the loop with the largest |Gamma|;
the open-loop fractions; the regime the stationary rule would give that
frame.

Per movie, the event times (first frame at which each holds; the resolution
is one frame, dt of the file):
  t_charge, t_charge_sd2  director charge appears at sigma_det, at sigma_det2;
  t_dip, t_dip_on         a dip below core_frac appears; one appears on the line;
  t_quarter_mean          the mean |Gamma| over the dips reaches circ_on (0.25),
                          the records' convention of the winding-times figure;
  t_quarter_max           the largest |Gamma| reaches circ_on, with whether that
                          loop's sign of k closes at that frame;
  t_wound                 a dip on the line reaches wind_hi (0.9 turn), the
                          figure's "first core wound";
and the final values: the largest |Gamma|, the regime, and fell_back (the
largest |Gamma| reached circ_on and ends below it).

Outputs under --outdir (default results/charge_circ_time_series/, anchored to
this script's folder when relative): per movie <stem>/series.csv and
<stem>/fig_series.png; for the set events.csv, fig_event_times.png (the
winding-times figure rebuilt from these measures, with the quarter-turn
crossings marked by closure and by whether they fell back), report.txt,
config.json.

Each movie is loaded once; each frame goes through
pgb_charge_circ_survey_v11.survey_file with the loop map, the width scan,
the energy and the per-frame files switched off. Python 3.8 compatible;
imports pgb_charge_circ_survey_v11 (v1.1) and through it pgb_marker_survey
(v1.2) and the frozen method layer, none of them edited.

Changelog
  v1.0  first version. Sandbox check on SH stand-in movies (three frames
        each, t = 0, 50, 100; not of record): every frame surveys, the event
        times and the figures are written, and the last frame's row matches
        the stationary survey of the same file.
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
import pgb_charge_circ_survey_v11 as PCS                                    # noqa: E402
import pgb_marker_survey as PMS                                             # noqa: E402
import matplotlib                                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402
from matplotlib.lines import Line2D                                         # noqa: E402

VERSION = "charge_circ_time_series v1.0 (on %s)" % PCS.VERSION
SERIES_KEYS = ["t", "frame", "regime", "n_V", "n_X", "charge_abs_per_L", "charge_abs_per_L_sd2", "n_dips_on", "n_dips_off",
               "dips_off_med_L", "dip_off_to_V_med_L", "dip_depth_on_med", "dip_depth_off_med", "A_band_min", "A_band_min_off_L",
               "A_line_min", "dip_circ_on_med", "dip_circ_on_max", "dip_circ_off_med", "dip_circ_off_max", "circ_mean_all",
               "circ_max_all", "circ_max_closed", "dip_open_frac_on", "dip_open_frac_off", "n_dips_on_circ_hi"]


def movie_frames(path, args):
    """load a movie once and yield the frame records survey_file takes (energy left out)"""
    D = PMS.load_npz(path)
    u = D["u"]
    if u.ndim == 2:
        u = u[..., None]
    nt = u.shape[-1]
    t = np.atleast_1d(D["t"])
    idx = list(range(0, nt, max(1, args.frame_step)))
    if args.skip_first and idx and idx[0] == 0:
        idx = idx[1:]
    if idx[-1] != nt - 1:
        idx.append(nt - 1)
    for i in idx:
        yield dict(stem=Path(path).stem, path=str(path), E_GB_stored=float("nan"), e2=None, energy_source="none",
                   u2=np.asarray(u[..., i], float), x=np.asarray(D["x"], float), y=np.asarray(D["y"], float),
                   t=float(t[min(i, t.size - 1)]), frame=int(i), nt=int(nt), md=dict(D["md"]), layout="movie")


def first_time(ts, cond):
    hit = np.where(cond)[0]
    return (float(ts[hit[0]]), int(hit[0])) if len(hit) else (float("nan"), -1)


def run_movie(path, sargs, args, out):
    rows = []
    t0 = time.time()
    for R in movie_frames(path, args):
        row, G = PCS.survey_file(path, sargs, out, [], quiet_figs=True, R=R, write=False)
        circ = np.array([dp["circ"] for dp in G["dips"]]) if G["dips"] else np.zeros(0)
        row["circ_mean_all"] = float(np.mean(circ)) if len(circ) else float("nan")
        if len(circ):
            j = int(np.argmax(circ))
            row["circ_max_all"] = float(circ[j]); row["circ_max_closed"] = int(G["dips"][j]["closed"])
        else:
            row["circ_max_all"] = float("nan"); row["circ_max_closed"] = -1
        rows.append({k: row.get(k, float("nan")) for k in SERIES_KEYS + ["mu", "L"]})
    mu = rows[0]["mu"]
    stem = Path(path).stem
    od = out / stem; od.mkdir(parents=True, exist_ok=True)
    with open(od / "series.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=SERIES_KEYS); wr.writeheader()
        wr.writerows([{k: r[k] for k in SERIES_KEYS} for r in rows])
    g = lambda k: np.array([r[k] for r in rows], float)
    ts = g("t")
    ev = dict(stem=stem, mu=mu, n_frames=len(rows), t_first=float(ts[0]), t_last=float(ts[-1]),
              dt=float(np.median(np.diff(ts))) if len(ts) > 1 else float("nan"))
    ev["t_charge"], _ = first_time(ts, g("charge_abs_per_L") > 0)
    ev["t_charge_sd2"], _ = first_time(ts, g("charge_abs_per_L_sd2") > 0)
    ev["t_dip"], _ = first_time(ts, (g("n_dips_on") + g("n_dips_off")) > 0)
    ev["t_dip_on"], _ = first_time(ts, g("n_dips_on") > 0)
    cm_, cx_ = np.nan_to_num(g("circ_mean_all")), np.nan_to_num(g("circ_max_all"))
    ev["t_quarter_mean"], _ = first_time(ts, cm_ >= args.circ_on)
    ev["t_quarter_max"], iq = first_time(ts, cx_ >= args.circ_on)
    ev["quarter_max_closed"] = int(rows[iq]["circ_max_closed"]) if iq >= 0 else -1
    ev["t_wound"], _ = first_time(ts, g("n_dips_on_circ_hi") > 0)
    ev["circ_max_final"] = float(cx_[-1]); ev["circ_mean_final"] = float(cm_[-1])
    ev["circ_max_peak"] = float(cx_.max()) if len(cx_) else float("nan")
    ev["fell_back"] = int(iq >= 0 and cx_[-1] < args.circ_on)
    ev["regime_final"] = rows[-1]["regime"]
    ev["elapsed_s"] = float(time.time() - t0)
    fig_series(rows, ev, od / "fig_series.png", args, sargs)
    return ev


def fig_series(rows, ev, path, args, sargs):
    g = lambda k: np.array([r[k] for r in rows], float)
    ts = g("t")
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 6.6), sharex=True)
    ax = axs[0, 0]
    ax.plot(ts, g("charge_abs_per_L"), "-", color=PCS.C_POS, lw=1.3, label="sum |T| per L, sigma_det %.2g" % sargs.sigma_det)
    ax.plot(ts, g("charge_abs_per_L_sd2"), "--", color=PCS.C_POS, lw=1.0, label="sigma_det %.2g" % sargs.sigma_det2)
    ax.set_ylabel("director charge per L"); ax.legend(fontsize=7)
    ax = axs[0, 1]
    ax.plot(ts, g("circ_max_all"), "-", color="#3b5ba5", lw=1.3, label="largest |Gamma| over the dips")
    ax.plot(ts, g("circ_mean_all"), "-", color=PCS.MUTED, lw=1.0, label="mean |Gamma| over the dips")
    op = g("circ_max_closed") == 0
    ax.plot(ts[op], g("circ_max_all")[op], "o", mfc="none", mec="r", ms=4, label="largest loop: sign of k does not close")
    ax.axhline(args.circ_on, color=PCS.MUTED, lw=0.6, ls=":"); ax.axhline(sargs.wind_hi, color=PCS.MUTED, lw=0.6, ls="--")
    ax.set_ylabel("circulation (turns)"); ax.legend(fontsize=7)
    ax = axs[1, 0]
    ax.plot(ts, g("A_band_min"), "-", color="k", lw=1.2, label="deepest mg A in the band")
    ax.plot(ts, g("dip_depth_off_med"), "o", color=PCS.MUTED, mfc="none", ms=3, label="dip depth, off the line")
    ax.plot(ts, g("dip_depth_on_med"), "o", color=PCS.MUTED, ms=3, label="dip depth, on the line")
    ax.axhline(sargs.core_frac, color=PCS.MUTED, lw=0.6, ls=":")
    ax.set_ylabel("amplitude / far"); ax.set_xlabel("t"); ax.legend(fontsize=7)
    ax = axs[1, 1]
    ax.plot(ts, g("A_band_min_off_L"), ".", color=PCS.C_CUT2, ms=4, label="deepest mg A: distance from the line")
    ax.plot(ts, g("dips_off_med_L"), "o", color=PCS.MUTED, mfc="none", ms=3, label="dips off the line")
    ax.plot(ts, g("dip_off_to_V_med_L"), "v", color=PCS.C_NEG, mfc="none", ms=3, label="off-line dip to nearest V")
    ax.set_ylabel("distance (L)"); ax.set_xlabel("t"); ax.legend(fontsize=7)
    for ax in axs.ravel():
        for k, c in (("t_charge", PCS.C_POS), ("t_quarter_max", "#3b5ba5"), ("t_wound", "#5b3fa8")):
            if np.isfinite(ev[k]):
                ax.axvline(ev[k], color=c, lw=0.6, ls=":")
    fig.suptitle("mu = %.3f (%.1f deg): final regime %s; dotted verticals: charge, quarter turn (largest loop), wound"
                 % (ev["mu"], np.degrees(np.arcsin(ev["mu"])), ev["regime_final"]), fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_event_times(evs, path, args):
    mu = np.array([e["mu"] for e in evs])

    def g(k):                       # times at or before the first surveyed frame are not drawn on the log axis
        v = np.array([e[k] for e in evs], float)
        if k.startswith("t_"):
            v = np.where(v > 0, v, np.nan)
        return v
    fig, axs = plt.subplots(2, 1, figsize=(8, 8.6), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1], hspace=0.55))
    ax = axs[0]
    ax.plot(mu, g("t_charge"), "o-", color="#1f6fb4", ms=4, label="director charge appears (sigma_det 0.3)")
    ax.plot(mu, g("t_charge_sd2"), "o--", color="#1f6fb4", mfc="none", ms=4, lw=0.8, label="director charge appears (sigma_det 0.2)")
    tq, cl, fb = g("t_quarter_max"), g("quarter_max_closed"), g("fell_back")
    ax.plot(mu, g("t_quarter_mean"), "-", color="#2a9d4b", lw=0.8, label="mean circulation reaches 1/4 turn (the records' convention)")
    for m_, t_, c_, f_ in zip(mu, tq, cl, fb):
        if np.isfinite(t_):
            ax.plot(m_, t_, "^", ms=6, color="#2a9d4b", mfc="#2a9d4b" if c_ == 1 else "w", mec="#2a9d4b" if not f_ else "r", mew=1.2)
    ax.plot(mu, g("t_wound"), "D-", color="#c4501b", ms=4, label="first on-line dip at 0.9 turn (wound)")
    ax.set_yscale("log"); ax.set_ylabel("first time it appears (t, log scale)")
    hs, ls = ax.get_legend_handles_labels()
    hs += [Line2D([], [], marker="^", ls="", color="#2a9d4b", mfc="#2a9d4b", ms=6),
           Line2D([], [], marker="^", ls="", color="#2a9d4b", mfc="w", ms=6),
           Line2D([], [], marker="^", ls="", mfc="w", mec="r", ms=6)]
    ls += ["largest loop reaches 1/4 turn, sign of k closes", "largest loop reaches 1/4 turn, sign of k does not close",
           "red edge: ends below 1/4 turn (fell back)"]
    ax.legend(hs, ls, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False)
    ax = axs[1]
    ax.plot(mu, g("circ_max_peak"), "^", color="#3b5ba5", mfc="none", ms=5, label="largest |Gamma| at any frame")
    ax.plot(mu, g("circ_max_final"), "o", color="#3b5ba5", ms=4, label="largest |Gamma| at the last frame")
    ax.plot(mu, g("circ_mean_final"), "x", color=PCS.MUTED, ms=5, label="mean |Gamma| at the last frame")
    ax.axhline(args.circ_on, color=PCS.MUTED, lw=0.6, ls=":"); ax.axhline(0.9, color=PCS.MUTED, lw=0.6, ls="--")
    ax.set_ylabel("circulation (turns)"); ax.set_xlabel("mu = sin(far-field angle)"); ax.legend(fontsize=7)
    PCS._deg_axis(axs[0])
    fig.suptitle("runs end at t = %s; events already present at the first surveyed frame are in events.csv, not drawn"
                 % ", ".join(sorted(set("%.4g" % e["t_last"] for e in evs))), fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def main(args):
    out = Path(args.outdir)
    if not out.is_absolute():
        out = HERE / out
    out.mkdir(parents=True, exist_ok=True)
    files = PMS.expand_inputs(args.npz)
    if not files:
        print("no input files (--npz)"); return 1
    sargs = PCS.default_args()
    for k, v in vars(args).items():
        if hasattr(sargs, k):
            setattr(sargs, k, v)
    sargs.map = 0; sargs.sigma_scan = ""; sargs.tag_frame = 0
    with open(out / "config.json", "w") as fh:
        json.dump(dict(vars(args), version=VERSION, files=files), fh, indent=2, default=str)
    lines = [VERSION, PCS.header_line(sargs), ""]
    evs = []
    t0 = time.time()
    for f in files:
        try:
            ev = run_movie(f, sargs, args, out)
            evs.append(ev)
            lines.append("%s: mu %.3f, %d frames (t %.4g to %.4g, dt %.4g); charge at t %.4g (sigma_det 0.2: %.4g); first dip %.4g,"
                         " on the line %.4g; quarter turn (mean) %.4g, (largest loop) %.4g, sign closes %s; wound %.4g;"
                         " largest |Gamma| peak %.3f, final %.3f%s; final regime %s; %.0f s" % (
                             ev["stem"], ev["mu"], ev["n_frames"], ev["t_first"], ev["t_last"], ev["dt"], ev["t_charge"],
                             ev["t_charge_sd2"], ev["t_dip"], ev["t_dip_on"], ev["t_quarter_mean"], ev["t_quarter_max"],
                             {1: "yes", 0: "no", -1: "n/a"}[ev["quarter_max_closed"]], ev["t_wound"], ev["circ_max_peak"],
                             ev["circ_max_final"], " (fell back below the quarter turn)" if ev["fell_back"] else "",
                             ev["regime_final"], ev["elapsed_s"]))
        except Exception as exc:                                    # loud flag, keep going
            lines.append("FLAG: %s failed: %r" % (f, exc))
        print(lines[-1])
    if evs:
        evs = sorted(evs, key=lambda e: e["mu"])
        with open(out / "events.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(evs[0].keys())); wr.writeheader(); wr.writerows(evs)
        fig_event_times(evs, out / "fig_event_times.png", args)
    lines.append("")
    lines.append("elapsed %.1f s" % (time.time() - t0))
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("outputs ->", out)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    npz = ""                       # comma-separated movie files and/or folders
    outdir = str(Path("results") / "charge_circ_time_series")
    frame_step = 1                 # survey every frame_step frames (the last frame always)
    skip_first = 0                 # 1: leave out frame 0 (the planted initial condition)
    circ_on = 0.25                 # turns: the quarter-turn threshold of the winding-times figure
    wind_hi = 0.9                  # turns: wound
    loop_lam = 0.0                 # > 0: dip loops of this many stripe wavelengths (passed to the survey)
    sigma_det = 0.3
    sigma_det2 = 0.2


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in vars(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    sys.exit(main(p.parse_args()))
