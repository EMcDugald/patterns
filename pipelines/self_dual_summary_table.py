# pipelines/self_dual_summary_table.py
"""
One table across the self_dual_discovery v1.1 runs (controls and SH files
by mu): per run and route, the headline, whether div k ~ w is in the tie
family and its margin, the A2 selections for relation A and relation B
(at both tolerances), the signed beta and its R2, the sign fraction, the
signed / absolute / squared R2 of relation A, gamma/|beta|, lambda and
(lambda_J, m), the model-corrected beta where the curve allows, and the
cores. Reads summary_row.json from every run folder under --root.

Why: the research thread reads laws across the run set (0.79 to 0.95 and
later the fine mu sweep), so a qualitative change between two angles
should show as a row change in one table rather than be read file by file.

Outputs: <root>/summary_table_<stamp>.csv and .md

Changelog
---------
v1.0  10 Oct 2026, companion to self_dual_discovery v1.1.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent


def short(stem):
    if stem.startswith("sh_pgb_zigzag_mu"):
        return "SH mu %s" % stem.split("mu")[1][:5]
    return stem


def main(args):
    root = Path(args.root) if Path(args.root).is_absolute() else _HERE / args.root
    rows = []
    for f in sorted(root.glob("*/*/summary_row.json")):
        J = json.load(open(f))
        for route, r in J["routes"].items():
            a2 = r.get("a2", {})
            def sel(key, tol):
                blk = a2.get(key, {}).get("by_tol", {}).get(tol)
                return ("{%s} %.3f" % (", ".join("%+.2f %s" % (c, n) for c, n in zip(blk["coefs"], blk["support"])), blk["r2_holdout"])) if blk else ""
            sg = r.get("signs", {})
            fits = sg.get("fits", {})
            fam = r.get("family", {})
            rows.append(dict(
                run=short(J["stem"]), mu=J["mu"], route=route, n=r["n"], folder=f.parent.name,
                headline="%s (%.3f)" % (r["headline"], r["headline_r2"]),
                sd_in_family=fam.get("sd_in_family"), sd_r2_gap=(fam.get("best_r2", np.nan) - fam.get("sd_r2", np.nan)),
                tie_margin=fam.get("tie_margin"), n_same_order=fam.get("n_same_order"),
                A2_divk_tol01=sel("div k | full library", "0.01"), A2_divk_tol02=sel("div k | full library", "0.02"),
                A2_divk_noself_tol02=sel("div k | without columns that contain the target", "0.02"),
                A2_B_tol01=sel("Lap div k | full library", "0.01"), A2_B_tol02=sel("Lap div k | full library", "0.02"),
                beta_signed=r["beta_signed"], beta_r2=r["beta_r2"],
                beta_model_corrected=r.get("beta_model_corrected", np.nan),
                sign_fraction=sg.get("sign_fraction"),
                r2_signed=fits.get("signed: div k on w", {}).get("r2_holdout"),
                r2_abs=fits.get("absolute: |div k| on |w|", {}).get("r2_holdout"),
                r2_squared=fits.get("squared: (div k)^2 on w^2", {}).get("r2_holdout"),
                gamma_over_absbeta=r["gamma_over_absbeta"], lam=r["lam"], lam_r2=r["lam_r2"], lambdaJ=r["lambdaJ"], mJ=r["mJ"], SJ_r2=r["SJ_r2"],
                sign_verdict=sg.get("verdict", "")))
    if not rows:
        print("no summary_row.json under", root); return 1
    stamp = time.strftime("%m%d_%H%M")
    keys = list(rows[0])
    with open(root / ("summary_table_%s.csv" % stamp), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
    md = ["| run | route | n | headline (R2) | div k~w in family (gap) | A2 A tol .01 | A2 A tol .02 | A2 B tol .02 | beta (R2) | beta corr | sign frac | R2 signed/abs/sq | g/|b| | lambda (R2) | lambda_J, m (R2) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| %s | %s | %d | %s | %s (%.4f) | %s | %s | %s | %+.3f (%.3f) | %s | %.2f | %.3f/%.3f/%.3f | %.2f | %+.2f (%.2f) | %+.2f, %+.2f (%.2f) |" % (
            r["run"], r["route"], r["n"], r["headline"], "yes" if r["sd_in_family"] else "NO", r["sd_r2_gap"] if r["sd_r2_gap"] == r["sd_r2_gap"] else np.nan,
            r["A2_divk_tol01"], r["A2_divk_tol02"], r["A2_B_tol02"], r["beta_signed"], r["beta_r2"],
            ("%.3f" % r["beta_model_corrected"]) if r["beta_model_corrected"] == r["beta_model_corrected"] else "n/a",
            r["sign_fraction"] if r["sign_fraction"] is not None else np.nan,
            r["r2_signed"] or np.nan, r["r2_abs"] or np.nan, r["r2_squared"] or np.nan,
            r["gamma_over_absbeta"], r["lam"], r["lam_r2"], r["lambdaJ"], r["mJ"], r["SJ_r2"]))
    (root / ("summary_table_%s.md" % stamp)).write_text("\n".join(md))
    print("\n".join(md)); print("wrote", root / ("summary_table_%s.md" % stamp))
    return 0


class _Args:
    root = "results/1010_self_dual_discovery_v11"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in dir(_Args) if not a.startswith("_")]:
        p.add_argument("--" + k, type=type(getattr(_Args, k)), default=getattr(_Args, k))
    sys.exit(main(p.parse_args()))
