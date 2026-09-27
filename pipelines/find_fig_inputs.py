# pipelines/find_fig_inputs.py
"""
Walks pipelines/results (or --root) and prints, for each of the six
fig_*.py scripts, the input files it can use and a ready-to-paste command
line. Nothing is computed; it only lists what exists.

  fig_phase_law_windows / fig_competitors_r2   sysid_scan runs.csv (or the
        wsindy run directories to feed sysid_scan first)
  fig_lambdaJ_time                             rcn_balance_ledger ledger.csv
        per file, with mu parsed from the file stem and the route names
        found in the csv
  fig_rollout_scores                           rcn_rollout scores.json dirs,
        with law and mu from config.json
  fig_winding_time                             winding_time_vs_mu table.csv,
        with the mu values each holds (R is not in the table: pass it)
  fig_theta_t_maps                             the T25 npz files (dt 0.2)

Run:
  python find_fig_inputs.py                    (from pipelines/, root = results)
  python find_fig_inputs.py --root results --data data
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

MU_RE = re.compile(r"mu(0\.\d+|0_\d+|0\d{2,3})")


def mu_of(name):
    m = MU_RE.search(str(name))
    if not m:
        return None
    s = m.group(1).replace("_", ".")
    return float(s if "." in s else "0." + s[1:])


def main(args):
    root = Path(args.root)
    print("root:", root.resolve())
    # ---- sysid_scan outputs and wsindy runs
    runs = sorted(root.rglob("runs.csv"))
    print("\n== sysid_scan runs.csv (for fig_phase_law_windows, fig_competitors_r2) ==")
    for r in runs:
        rows = list(csv.DictReader(open(r)))
        blocks = sorted({x.get("block", "") for x in rows})
        wins = sorted({(x.get("file", ""), x.get("mu", ""), x.get("t0", ""), x.get("t1", ""), x.get("wclass", ""), x.get("domain", ""))
                       for x in rows if x.get("file") not in ("control", "pooled")})
        has_SJ = any("forced {w div k, k.grad w, J}" in b for b in blocks)
        has_ctrl = any(x.get("wclass") == "control" for x in rows)
        print("  %s   rows %d, windows %d, {S,J} competitor %s, control rows %s" % (r, len(rows), len(wins), has_SJ, has_ctrl))
        for w in wins[:20]:
            print("      file %s  mu %s  t %s-%s  %s  %s" % w)
        print("    python fig_phase_law_windows.py --runs %s" % r)
        print("    python fig_competitors_r2.py --runs %s" % r)
    if not runs:
        reps = [p.parent for p in root.rglob("report.txt") if open(p, errors="ignore").read(20).startswith("wsindy_theta_t")]
        print("  none. wsindy_theta_t run directories found: %d; make runs.csv with" % len(reps))
        roots = sorted({str(p.parents[1]) for p in reps})[:8]
        print("    python sysid_scan.py --roots %s --outdir results/sysid_scan" % ",".join(roots))
    # ---- ledgers
    print("\n== rcn_balance_ledger ledger.csv (for fig_lambdaJ_time) ==")
    specs = []
    for L in sorted(root.rglob("ledger.csv")):
        rows = list(csv.DictReader(open(L)))
        if not rows:
            continue
        routes = sorted({x.get("route", "") for x in rows})
        ts = [float(x["t"]) for x in rows if x.get("t")]
        has_lamJ = "lam_J" in rows[0]
        mu = mu_of(L.parent.parent.name)
        stem = L.parent.parent.name
        print("  %s\n      stem %s  mu %s  t %.2f..%.2f  frames %d  routes %s  lam_J column %s" %
              (L, stem[:60], mu, min(ts), max(ts), len(set(ts)), routes, has_lamJ))
        if has_lamJ and len(set(ts)) > 3 and mu is not None:
            specs.append((mu, "%s:%s" % (L, mu), routes))
    if specs:
        best = {}
        for mu, spec, routes in specs:       # keep the longest ledger per mu (the T25 runs)
            best.setdefault(mu, (spec, routes))
        route = "estimator" if any("estimator" in r for _, r in best.values()) else best[next(iter(best))][1][0]
        print("    python fig_lambdaJ_time.py --route %s --ledgers \"%s\"" % (route, ",".join(s for s, _ in best.values())))
    # ---- rollouts
    print("\n== rcn_rollout scores.json (for fig_rollout_scores) ==")
    dirs = []
    for S in sorted(root.rglob("scores.json")):
        cfg = json.load(open(S.parent / "config.json")) if (S.parent / "config.json").exists() else {}
        law = cfg.get("law", "?")
        mu = mu_of(cfg.get("npz", "")) or mu_of(S.parent.parent.name)
        print("  %s   law %s  mu %s  t0 %s  t_fit %s" % (S.parent, law, mu, cfg.get("t0"), cfg.get("t_fit")))
        dirs.append(str(S.parent))
    if dirs:
        print("    python fig_rollout_scores.py --runs \"%s\"" % ",".join(dirs))
    # ---- winding tables
    print("\n== winding_time_vs_mu table.csv (for fig_winding_time; R is NOT in the table, pass path:R) ==")
    for T in sorted(root.rglob("table.csv")):
        rows = list(csv.DictReader(open(T)))
        if not rows or "t_wind_mean" not in rows[0]:
            continue
        mus = sorted({float(x["mu"]) for x in rows})
        dts = sorted({round(float(x["dt"]), 4) for x in rows})
        print("  %s   mu %s  dt %s  files %d" % (T, mus, dts, len(rows)))
        print("    python fig_winding_time.py --tables \"%s:0.5\"      (edit R; join several with commas)" % T)
    # ---- data files for the maps
    print("\n== T25 npz files (for fig_theta_t_maps) ==")
    data = Path(args.data)
    n = 0
    for f in sorted(data.rglob("*T25*lower*.npz")) if data.exists() else []:
        n += 1
        print("  %s" % f)
        if n == 1:
            print("    python fig_theta_t_maps.py --npz %s --t 4.0 --coef 0.80,0.85,-1.63 --smooth 0.5" % f)
    if not n:
        print("  none under %s (pass --data <folder with the sh_pgb_zigzag npz files>)" % data)
    return 0


class _Args:
    root = "results"
    data = "data"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=_Args.root)
    p.add_argument("--data", default=_Args.data)
    sys.exit(main(p.parse_args()))
