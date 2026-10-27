"""pgb_regime_summary_figure_v11.py  v1.1  (26 Oct 2026)   one clean summary figure of the PGB regimes against angle

Plain terms. Reads a survey table already written by the charge-circulation
survey (pgb_charge_circ_survey.py v1.0, _v11.py v1.1, _v12.py v1.2, or
the t200/ and t100/ tables of the large-sweep runners) and draws six
panels against mu = sin(far-field angle), shaded knee / intermediate /
dislocation. Nothing is measured here; it only replots the table.

  (a) director charge per L at one detection width (--width; one width only);
  (b) circulation |Gamma| about the dips, in turns: dips on the line
      (filled) and dips off the line (hollow), each where the angle has them;
      the median as a circle, the largest as a faint triangle; and, with a
      v1.2 table, the largest |Gamma| on any closed loop of the circulation
      map (grey squares), which is defined at every angle, knees included;
  (c) dips on the line: the amplitude at the dip over far field (lower is
      deeper), drawn only where such dips exist;
  (d) dips off the line: the same, drawn only where they exist;
  (e) where the dips sit, in L from the PGB line, at every angle with
      charge: the V cells (blue triangles); the off-line dips' patch centres
      (hollow circles, where their circulation loops are drawn); and, with a
      v1.2 table, the dips' lowest points, off the line in the intermediates
      (teal crosses) and beside the line in the dislocations (filled teal
      crosses; the patch centre there is on the line);
  (f) dislocation cores: the V-X separation, in L, for the dislocation
      angles only.

The shading is recomputed from the plotted width so the two always agree:
knee where the charge at that width is zero; otherwise dislocation where at
least one on-line dip reaches wind_hi (0.9 turn), else intermediate. At the
survey's own width (0.3) this is the survey's regime column; any angle where
it differs is listed in the report.

Width: 0.3 by default (the records' width; its curve is a clean step at
0.69). Widths other than 0.3 and 0.2 need a v1.1 table (the width scan).

Outputs under --outdir (default results/pgb_regime_summary_figure/, anchored
to this script's folder when relative): fig_regime_summary.png and .pdf,
summary_table.csv (exactly the plotted numbers), report.txt.

Python 3.8 compatible; standard library, numpy and matplotlib only.

Changelog
  v1.1  (26 Oct 2026; new file name, v1.0 kept; panel (e) extended later the
        same day, after the first v1.2 run showed that the dips' lowest
        points form one continuous curve across both regimes: 0.42 L at
        mu 0.70, 0.30 at 0.82, 0.29 at 0.83, 0.06 at 0.99 in the energy
        sweep, tracking the V cells; the V cells moved from (f) to (e); a
        footnote on the map's loop size) the circulation-map maximum in
        panel (b), so the knees carry a measured value instead of a gap, and
        the dips' lowest points in panel (e); both need a v1.2 survey table
        and are skipped silently for older tables.
  v1.0  first version, after Ed's reading of the v1.0 magnitudes figure
        (25 Oct): one width, no cell-arrangement curves other than the core
        tightness of the dislocations, a gap where there is nothing to
        measure. Checked on Ed's v1.0 table of the 24 Oct sweep at t = 200.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402
from matplotlib.patches import Patch                                        # noqa: E402

HERE = Path(__file__).resolve().parent
VERSION = "pgb_regime_summary_figure v1.1"
REGIME_COL = {"knee": "#e9e7e2", "intermediate": "#f6e3b4", "dislocation": "#cfe0f5"}
C_ON, C_OFF, C_V, C_VX = "#2f4b9a", "#6b6b6b", "#2a6fd6", "#7a3fa0"


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _wtag(w):
    return ("%.3f" % w).rstrip("0").rstrip(".").replace(".", "p")


def charge_column(header, width):
    """the table column holding the charge per L at this width"""
    for name in ("charge_per_L_sd%s" % _wtag(width),):
        if name in header:
            return name
    if abs(width - 0.3) < 1e-9 and "charge_abs_per_L" in header:
        return "charge_abs_per_L"
    if abs(width - 0.2) < 1e-9 and "charge_abs_per_L_sd2" in header:
        return "charge_abs_per_L_sd2"
    raise SystemExit("no charge column for width %.3g in this table (widths other than 0.3 and 0.2 need a v1.1 table)" % width)


def pick(r, *names):
    """the first of these columns present in the row (v1.0/v1.1 names, then the v1.2 marker-survey names)"""
    for n in names:
        if n in r:
            return _f(r[n])
    return float("nan")


def load(path, args):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit("empty table: %s" % path)
    ccol = charge_column(rows[0].keys(), args.width)
    out = []
    for r in rows:
        q = dict(mu=_f(r["mu"]), regime_table=r.get("regime", ""), charge=_f(r[ccol]),
                 n_on=pick(r, "n_dips_on"), n_off=pick(r, "n_dips_off"),
                 n_on_hi=pick(r, "n_dips_on_circ_hi", "rec_n_wound"),
                 circ_on_med=pick(r, "dip_circ_on_med", "dip_circk_on_med"), circ_on_max=pick(r, "dip_circ_on_max"),
                 circ_off_med=pick(r, "dip_circ_off_med", "dip_circk_off_med"), circ_off_max=pick(r, "dip_circ_off_max"),
                 depth_on=pick(r, "dip_depth_on_med"), depth_off=pick(r, "dip_depth_off_med"),
                 off_dist=pick(r, "dips_off_med_L"), vx=pick(r, "VX_sep_med_L"), v_off=pick(r, "V_offline_med_L"),
                 map_circ=pick(r, "map_circ_max_closed"), off_min=pick(r, "dip_min_off_off_med_L"),
                 on_min=pick(r, "dip_min_off_on_med_L"))
        n_on, n_off = np.nan_to_num(q["n_on"]), np.nan_to_num(q["n_off"])
        if not np.isfinite(q["charge"]) or q["charge"] <= 0:
            q["regime"] = "knee"
        elif np.nan_to_num(q["n_on_hi"]) > 0:
            q["regime"] = "dislocation"
        else:
            q["regime"] = "intermediate"
        q["has_on"], q["has_off"] = n_on > 0, n_off > 0
        out.append(q)
    out.sort(key=lambda q: q["mu"])
    return out, ccol


def shade(ax, Q):
    mus = np.array([q["mu"] for q in Q])
    edges = np.r_[mus[0] - 0.005, 0.5 * (mus[1:] + mus[:-1]), mus[-1] + 0.005]
    for i, q in enumerate(Q):
        ax.axvspan(edges[i], edges[i + 1], color=REGIME_COL[q["regime"]], lw=0, zorder=0)
    ax.set_xlim(edges[0], edges[-1])
    ax.grid(axis="y", color="w", lw=0.8, zorder=1)


def deg_axis(ax):
    try:
        sec = ax.secondary_xaxis("top", functions=(lambda m: np.degrees(np.arcsin(np.clip(m, -1, 1))),
                                                   lambda d: np.sin(np.radians(d))))
        sec.set_xlabel("far-field angle (deg)", fontsize=8)
        sec.tick_params(labelsize=8)
    except Exception:
        pass


def first_mu(Q, regime):
    m = [q["mu"] for q in Q if q["regime"] == regime]
    return min(m) if m else float("nan")


def figure(Q, args, out, title):
    mu = np.array([q["mu"] for q in Q])
    g = lambda k: np.array([q[k] for q in Q], float)
    has_on, has_off = g("has_on") > 0, g("has_off") > 0
    disl = np.array([q["regime"] == "dislocation" for q in Q])
    fig, axs = plt.subplots(3, 2, figsize=(11, 9.6), sharex=True)
    for ax in axs.ravel():
        shade(ax, Q)
        ax.tick_params(labelsize=8)

    ax = axs[0, 0]
    ax.plot(mu, g("charge"), "-", color="#c0392b", lw=1.6, zorder=3)
    ax.plot(mu, g("charge"), "o", color="#c0392b", ms=3, zorder=4)
    ax.set_ylabel("director charge per L\n(sum of |T|, width %.2g)" % args.width, fontsize=9)
    ax.set_ylim(bottom=-0.3)
    ax.set_title("(a) director charge", fontsize=10, loc="left")

    ax = axs[0, 1]
    con, coff = g("circ_on_med"), g("circ_off_med")
    m_on = has_on & np.isfinite(con)
    m_off = has_off & np.isfinite(coff)
    ax.plot(mu[m_on], g("circ_on_max")[m_on], "^", color=C_ON, ms=3.5, alpha=0.45, zorder=3)
    ax.plot(mu[m_off], g("circ_off_max")[m_off], "^", mfc="w", mec=C_OFF, ms=3.5, alpha=0.6, zorder=3)
    ax.plot(mu[m_on], con[m_on], "o", color=C_ON, ms=4, zorder=4, label="dips on the line: median (triangle: largest)")
    ax.plot(mu[m_off], coff[m_off], "o", mfc="w", mec=C_OFF, ms=4, zorder=4, label="dips off the line: median (triangle: largest)")
    mc = g("map_circ")
    has_map = np.isfinite(mc).any()
    if has_map:
        ax.plot(mu, mc, "s", color="#9a9a9a", ms=2.6, zorder=3, label="largest on any closed loop in the band (map; see note)")
    ax.axhline(1.0, color="k", lw=0.6, ls=":", zorder=2)
    ax.axhline(args.wind_hi, color="k", lw=0.6, ls="--", zorder=2)
    k_mu = [q["mu"] for q in Q if q["regime"] == "knee"]
    if k_mu and not has_map:
        ax.text(0.5 * (min(k_mu) + max(k_mu)), 0.3, "no dips:\nnothing to measure", ha="center", va="center", fontsize=8, color="#555")
    ax.set_ylim(-0.05, 1.12)
    ax.set_ylabel("circulation |Gamma| about the dip\n(turns)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.set_title("(b) circulation (dashed: %.2g turn, the wound threshold)" % args.wind_hi, fontsize=10, loc="left")

    ax = axs[1, 0]
    ax.plot(mu[has_on], g("depth_on")[has_on], "o", color=C_ON, ms=4, zorder=4)
    ax.set_ylabel("amplitude at the dip / far field\n(lower is deeper)", fontsize=9)
    ax.set_title("(c) dips on the PGB line", fontsize=10, loc="left")

    ax = axs[1, 1]
    ax.plot(mu[has_off], g("depth_off")[has_off], "o", mfc="w", mec=C_OFF, ms=4, zorder=4)
    ax.set_ylabel("amplitude at the dip / far field\n(lower is deeper)", fontsize=9)
    ax.set_title("(d) dips off the PGB line: depth", fontsize=10, loc="left")
    lo = np.nanmin(np.r_[g("depth_on")[has_on], g("depth_off")[has_off], 1.0]) - 0.03
    hi = np.nanmax(np.r_[g("depth_on")[has_on], g("depth_off")[has_off], 0.0]) + 0.03
    for a in (axs[1, 0], axs[1, 1]):
        a.set_ylim(lo, hi)

    ax = axs[2, 0]
    charged = np.array([q["regime"] != "knee" for q in Q])
    ax.plot(mu[charged], g("v_off")[charged], "v", color=C_V, ms=4, zorder=4, label="V cells")
    ax.plot(mu[has_off], g("off_dist")[has_off], "o", mfc="w", mec=C_OFF, ms=4, zorder=4, label="off-line dips: patch centre (loop centre)")
    om, onm = g("off_min"), g("on_min")
    if np.isfinite(om[has_off]).any():
        ax.plot(mu[has_off], om[has_off], "x", color="#1b7f8c", ms=4.5, zorder=5, label="off-line dips: lowest point")
    if np.isfinite(onm[has_on]).any():
        ax.plot(mu[has_on], onm[has_on], "P", color="#1b7f8c", ms=4.5, zorder=5, label="on-line dips: lowest point (centre on the line)")
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
    ax.set_ylabel("distance from the line (L)", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.set_title("(e) where the dips and V cells sit", fontsize=10, loc="left")
    ax.set_xlabel("mu = sin(far-field angle)", fontsize=9)

    ax = axs[2, 1]
    ax.plot(mu[disl], g("vx")[disl], "D-", color=C_VX, ms=3.5, lw=1.0, zorder=4, label="V-X separation")
    ax.set_ylabel("distance (L)", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
    ax.set_title("(f) dislocation cores: how tight", fontsize=10, loc="left")
    ax.set_xlabel("mu = sin(far-field angle)", fontsize=9)

    deg_axis(axs[0, 0]); deg_axis(axs[0, 1])
    fig.legend(handles=[Patch(color=v, label=k) for k, v in REGIME_COL.items()], loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=11)
    if np.isfinite(g("map_circ")).any():
        fig.text(0.5, 0.035, "Note (b): map loops have radius 0.5 L and sit on a grid; below mu about 0.87 such a loop is narrower than one stripe "
                 "wavelength, and in the upper intermediates the full-turn readings come from a minority of placements.",
                 ha="center", fontsize=7, color="#444", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out / "fig_regime_summary.png", dpi=170)
    fig.savefig(out / "fig_regime_summary.pdf")
    plt.close(fig)


def main(args):
    out = Path(args.outdir)
    if not out.is_absolute():
        out = HERE / out
    out.mkdir(parents=True, exist_ok=True)
    tab = Path(args.table)
    if not tab.is_absolute():
        tab = HERE / tab
    Q, ccol = load(tab, args)
    lines = [VERSION, "table: %s" % tab, "charge column: %s (width %.3g); regime threshold %.2g turn" % (ccol, args.width, args.wind_hi), ""]
    diff = [q for q in Q if q["regime_table"] and q["regime"] != q["regime_table"]]
    lines.append("regimes: knee %s; intermediate from %s; dislocation from %s" % (
        "to %.3g" % max([q["mu"] for q in Q if q["regime"] == "knee"]) if any(q["regime"] == "knee" for q in Q) else "none",
        "%.3g" % first_mu(Q, "intermediate"), "%.3g" % first_mu(Q, "dislocation")))
    lines.append("angles where the plotted regime differs from the table's regime column: %s" % (
        ", ".join("%.3f (%s -> %s)" % (q["mu"], q["regime_table"], q["regime"]) for q in diff) if diff else "none"))
    keys = ["mu", "regime", "regime_table", "charge", "n_on", "n_off", "circ_on_med", "circ_on_max", "circ_off_med", "circ_off_max",
            "map_circ", "depth_on", "depth_off", "off_dist", "off_min", "on_min", "vx", "v_off"]
    with open(out / "summary_table.csv", "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(keys)
        for q in Q:
            wr.writerow([q[k] if isinstance(q[k], str) else ("%.5g" % q[k]) for k in keys])
    title = args.title or ("PGB regimes against angle (%d angles; director charge at width %.2g)" % (len(Q), args.width))
    figure(Q, args, out, title)
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print("outputs ->", out)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    table = str(Path("results") / "1025_charge_circ_large_sweep" / "t200" / "survey_table.csv")
    outdir = str(Path("results") / "pgb_regime_summary_figure_v11")
    width = 0.3                    # director-charge detection width; one width only
    wind_hi = 0.9                  # turns: an on-line dip at or above this makes the angle a dislocation
    title = ""                     # figure title ("" for the default)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in vars(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    sys.exit(main(p.parse_args()))
