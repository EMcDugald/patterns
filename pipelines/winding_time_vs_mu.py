# pipelines/winding_time_vs_mu.py
"""
Winding time vs mu from mg_zigzag_theta_t v1.3 runs.

Walks a results root, reads every series.csv (one per npz run), and reports
per file:
  t_charge    first frame whose director charge sum|q| exceeds its initial
              value (the V-X pair onset; frozen detect_defects, sigma_det)
  t_circ_on   first frame with |circ| mean at r = 0.5 >= circ_on (a partial
              winding: transition frame)
  t_wind_mean first frame with |circ| mean at r = 0.5 >= circ_full
  t_wind_min  first frame with |circ| min  at r = 0.5 >= circ_full (every
              detected core wound; the strict reading)
  held        whether circ_min stays >= circ_full from t_wind_min to the end
plus the final-frame circulation, |z| min/median, and core count. Frames
with no detected core carry nan circulation and are skipped. r = 0.5 is the
validated loop (exact PN through the same code: 1.00 turns). The time
resolution of every onset is one frame (dt of the file); quote it.

Outputs (relative): results/winding_time_vs_mu/<stamp>/table.txt, table.csv

Changelog
---------
v1.1  staggering block from the v1.4 per-core columns: t_first_core,
      t_last_core (first frame at which one, then all final-wound cores
      are at or above circ_full) and n_wound_final.
v1.0  first version; tested on the 0.85 tail excerpt (frame 0 dropped).

Run:
  python winding_time_vs_mu.py --root results/mg_zigzag_theta_t_v3
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np


def read_series(path):
    rows = list(csv.DictReader(open(path)))
    f = lambda k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])
    S = dict(t=f("t"), n_cores=f("n_cores"), q=f("director_sum_abs_q"),
             zmin=f("A_min_over_median"),
             cmean=f("circ_mean_r0.5"), cmin=f("circ_min_r0.5"))
    if "circ_r0.5_per_core" in rows[0]:            # v1.4 columns
        S["per_core"] = [np.array([float(v) for v in r["circ_r0.5_per_core"].split(";")])
                         if r["circ_r0.5_per_core"] else np.zeros(0) for r in rows]
        S["n_wound"] = f("n_wound_r0.5")
    return S


def mu_of(run_dir):
    rep = run_dir / "report.txt"
    if rep.exists():
        for ln in open(rep):
            m = re.match(r"mu\s+([0-9.]+)\s+Lsp", ln)
            if m:
                return float(m.group(1))
    m = re.search(r"mu0[._](\d+)", run_dir.parent.name)
    return float("0." + m.group(1)) if m else np.nan


def first_t(t, cond):
    idx = np.flatnonzero(cond)
    return (float(t[idx[0]]), int(idx[0])) if len(idx) else (np.nan, -1)


def analyse(run_dir, circ_on, circ_full):
    S = read_series(run_dir / "series.csv")
    t, q = S["t"], S["q"]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else np.nan
    has = S["n_cores"] > 0
    ok = has & np.isfinite(S["cmean"])
    q0 = q[0]
    t_charge, _ = first_t(t, q > q0)
    t_on, _ = first_t(t, ok & (S["cmean"] >= circ_on))
    t_wm, _ = first_t(t, ok & (S["cmean"] >= circ_full))
    t_wn, i_wn = first_t(t, ok & (S["cmin"] >= circ_full))
    held = bool(np.all((S["cmin"][i_wn:] >= circ_full) | ~ok[i_wn:])) if i_wn >= 0 else False
    # staggering (v1.4 per-core columns): first and last frame at which a
    # core crosses circ_full, and the number of frames over which the
    # row's cores wind one by one
    t_first_core, t_last_core, n_final = np.nan, np.nan, -1
    if "per_core" in S:
        nw = S["n_wound"]
        n_final = int(nw[-1])
        t_first_core, _ = first_t(t, nw >= 1)
        t_last_core, _ = first_t(t, (nw >= n_final) & (n_final > 0))
    return dict(file=run_dir.parent.name, mu=mu_of(run_dir), dt=dt,
                t_first=float(t[0]), t_last=float(t[-1]), nframes=len(t),
                q_initial=int(q0), q_final=int(q[-1]), q_max=int(np.nanmax(q)),
                t_charge=t_charge, t_circ_on=t_on, t_wind_mean=t_wm,
                t_wind_min=t_wn, held=held,
                t_first_core=t_first_core, t_last_core=t_last_core, n_wound_final=n_final,
                circ_mean_final=float(S["cmean"][-1]), circ_min_final=float(S["cmin"][-1]),
                zmin_final=float(S["zmin"][-1]), n_cores_final=int(S["n_cores"][-1]),
                n_cores_max=int(np.nanmax(S["n_cores"])))


def run(args):
    root = Path(args.root)
    runs = sorted(p.parent for p in root.rglob("series.csv"))
    if not runs:
        print("no series.csv under", root)
        return 1
    rows = [analyse(r, args.circ_on, args.circ_full) for r in runs]
    rows.sort(key=lambda r: (r["mu"], r["file"]))
    out = Path(args.outdir) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as fh:
        json.dump(vars(args), fh, indent=2, default=str)
    hdr = ("%-6s %-6s %-9s %-9s %-11s %-11s %-6s %-9s %-9s %-7s %-5s %-4s %-4s %s" %
           ("mu", "dt", "t_charge", "t_circ_on", "t_wind_mean", "t_wind_min",
            "held", "circ_fin", "cmin_fin", "zmin", "cores", "q0", "qend", "file"))
    lines = ["winding_time_vs_mu v1.1   root %s" % root,
             "r = 0.5 loop; circ_on %.2f, circ_full %.2f; onset resolution one frame (dt)" %
             (args.circ_on, args.circ_full), "", hdr]
    for r in rows:
        lines.append("%-6.3f %-6.3f %-9.3f %-9.3f %-11.3f %-11.3f %-6s %-9.3f %-9.3f %-7.3f %-5d %-4d %-4d %s" %
                     (r["mu"], r["dt"], r["t_charge"], r["t_circ_on"], r["t_wind_mean"],
                      r["t_wind_min"], r["held"], r["circ_mean_final"], r["circ_min_final"],
                      r["zmin_final"], r["n_cores_final"], r["q_initial"], r["q_final"], r["file"]))
    lines += ["", "staggering from the v1.4 per-core columns (nan where the run predates v1.4):",
              "  %-6s %-13s %-12s %-14s %s" % ("mu", "t_first_core", "t_last_core", "n_wound_final", "file")]
    for r in rows:
        lines.append("  %-6.3f %-13.3f %-12.3f %-14d %s" % (r["mu"], r["t_first_core"], r["t_last_core"],
                                                          r["n_wound_final"], r["file"]))
    lines += ["", "nan = never reached in the file's window; t_first .. t_last per file:"]
    for r in rows:
        lines.append("  %.3f: %.3f .. %.3f (%d frames)" % (r["mu"], r["t_first"], r["t_last"], r["nframes"]))
    txt = "\n".join(lines)
    print(txt)
    (out / "table.txt").write_text(txt + "\n")
    with open(out / "table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("outputs ->", out)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    root = str(Path("results") / "mg_zigzag_theta_t_v3")
    outdir = str(Path("results") / "winding_time_vs_mu")
    circ_on = 0.25       # partial winding threshold (transition frame)
    circ_full = 0.90     # full winding threshold, units of 2 pi


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=_Args.root)
    p.add_argument("--outdir", default=_Args.outdir)
    p.add_argument("--circ-on", type=float, dest="circ_on", default=_Args.circ_on)
    p.add_argument("--circ-full", type=float, dest="circ_full", default=_Args.circ_full)
    sys.exit(run(p.parse_args()))
