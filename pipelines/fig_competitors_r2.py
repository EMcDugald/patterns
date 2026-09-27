# pipelines/fig_competitors_r2.py
"""
Figure for Section 3C: holdout R2 of the three-term competitors at equal
size, by window: forced RCN {S, Lap div k}, {S, J}, {S, A_dev div k},
{S, chi_J}, with S the stretching pair {w div k, k.grad w}; beside them
the free RHS+ext selection's R2 and its support (annotated). This is
sysid_scan's Table (iii-b) as a figure.

Input: runs.csv from sysid_scan.py (columns used: file, route, stamp, mu,
t0, t1, wclass, domain, block, coefs, r2_holdout). Blocks are matched by
target theta_t, kind forced, and support set (library name ignored):

  {w div k, k.grad w, Lap div k}       RCN triple
  {w div k, k.grad w, J}
  {w div k, k.grad w, A_dev div k}
  {w div k, k.grad w, chi_J}

and the free block "theta_t | RHS+ext library | free" for the annotation.
Nothing is fitted.

Run:
  python fig_competitors_r2.py --runs results/sysid_scan/runs.csv
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import (read_csv, fnum, parse_support, window_label, outdir_for,  # noqa: E402
                       write_config, write_table, save, cli, plt)

COMPETITORS = [("RCN triple ($\\nabla^2\\nabla\\cdot k$)", {"w div k", "k.grad w", "Lap div k"}),
               ("$J$", {"w div k", "k.grad w", "J"}),
               ("$(A - A_{far})\\nabla\\cdot k$", {"w div k", "k.grad w", "A_dev div k"}),
               ("$\\chi_J$", {"w div k", "k.grad w", "chi_J"})]


def run(args):
    rows = read_csv(args.runs)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    files = [f.strip() for f in args.files.split(",") if f.strip()]
    stamps = [s.strip() for s in args.stamps.split(",") if s.strip()]
    groups = {}
    for r in rows:
        ps = parse_support(r["block"])
        if ps is None or ps[0] != "theta_t":
            continue
        key = (r.get("file", ""), r.get("route", ""), r.get("t0", ""), r.get("t1", ""), r.get("stamp", ""))
        if ps[2] == "forced":
            sup = set(ps[3])
            for i, (_, s) in enumerate(COMPETITORS):
                if sup == s:
                    groups.setdefault(key, {})[i] = r
        elif ps[2] == "free" and ps[1] == "RHS+ext":
            groups.setdefault(key, {})["free"] = r
    wins = []
    for k, d in groups.items():
        if 1 not in d or k[0] in ("control", "pooled"):
            continue
        r = d[1]
        if r.get("domain", "") != args.domain or r.get("wclass", "") not in classes:
            continue
        if files and not any(f in k[0] for f in files):
            continue
        if stamps and not any(s in k[4] for s in stamps):
            continue
        wins.append((fnum(r["mu"]), fnum(r["t0"]), k, d))
    wins.sort(key=lambda w: (w[0], w[1]))
    if not wins:
        print("no windows with the {S, J} competitor in", args.runs)
        return 1
    out = outdir_for(__file__, args.outdir)
    write_config(out, args)

    labels = [window_label(d[1]) for _, _, _, d in wins]
    n, m = len(wins), len(COMPETITORS)
    width = 0.8 / (m + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    tab = []
    for i, (name, _) in enumerate(COMPETITORS):
        v = [fnum(d[i]["r2_holdout"]) if i in d else np.nan for _, _, _, d in wins]
        ax.bar(np.arange(n) + (i - (m - 1) / 2) * width, v, width, label="$\\{S,$ " + name + "$\\}$" if i else name)
    vf = [fnum(d["free"]["r2_holdout"]) if "free" in d else np.nan for _, _, _, d in wins]
    ax.plot(np.arange(n), vf, "k*", ms=9, label="free selection (RHS+ext)")
    for j, (_, _, _, d) in enumerate(wins):
        if "free" in d:
            nsup = len(json.loads(d["free"]["coefs"]))
            ax.annotate("free: %d terms" % nsup, (j, vf[j]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6)
    lo = np.nanmin([fnum(d[i]["r2_holdout"]) for _, _, _, d in wins for i in d if i != "free"] + [v for v in vf if np.isfinite(v)])
    ax.set_ylim(max(0.0, lo - 0.05), 1.0)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("holdout $R^2$")
    ax.set_title("three-term competitors at equal size, $\\theta_t$ on the band ($S = \\{w\\nabla\\cdot k, k\\cdot\\nabla w\\}$); "
                 "the free selection's support is in data.csv", fontsize=8)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    save(fig, out)
    for (mu, t0, k, d), lab in zip(wins, labels):
        row = dict(file=k[0], route=k[1], stamp=k[4], mu=mu, t0=t0, t1=fnum(d[1]["t1"]), wclass=d[1].get("wclass", ""))
        for i, (name, s) in enumerate(COMPETITORS):
            row["r2_" + "_".join(sorted(s - {"w div k", "k.grad w"}))] = fnum(d[i]["r2_holdout"]) if i in d else np.nan
        row["r2_free"] = fnum(d["free"]["r2_holdout"]) if "free" in d else np.nan
        row["free_support"] = d["free"]["coefs"] if "free" in d else ""
        tab.append(row)
    write_table(out, tab)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    runs = str(Path("results") / "sysid_scan" / "runs.csv")
    outdir = "results"
    domain = "band"
    classes = "pre,straddle,post"
    files = ""
    stamps = ""


if __name__ == "__main__":
    sys.exit(cli(_Args, __doc__, run))
