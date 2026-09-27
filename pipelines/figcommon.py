# pipelines/figcommon.py
"""
Shared pieces for the paper's csv-to-figure scripts (fig_*.py).

Every fig_* script: reads outputs of the scripts of record (sysid_scan,
rcn_balance_ledger, rcn_rollout, winding_time_vs_mu, mg_zigzag_theta_t),
fits nothing, writes <outdir>/<script_name>/<stamp>/{fig.png, fig.pdf,
data.csv, config.json}. The data.csv is the exact table the figure was
drawn from, so a reader of the paper can be handed the numbers.

Python 3.8 compatible (no match, no X | Y types).
"""
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:                                   # Ed's tree has a paper style module
    from paper_style import use_paper_style
    use_paper_style()
except Exception:
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                         "figure.dpi": 150, "savefig.dpi": 300})

# event times of record (RESULTS_mg_campaign_0916 Section 1; sysid_scan T_WIND):
# winding first .. last core, and the charge onset (H1 1.2), R = 1/2, coarse
# cadence dt 0.2 (one-frame resolution). Override from the CLI when the fine
# cadence files have been analysed (todo in Section 3C).
T_WIND = {0.85: (3.8, 5.4), 0.86: (3.0, 3.6), 0.87: (2.4, 2.8), 0.88: (2.0, 2.2),
          0.89: (1.6, 1.8), 0.90: (1.2, 1.6), 0.95: (0.6, 0.6), 0.975: (0.2, 0.2),
          0.99: (0.2, 0.2)}
T_CHARGE = {0.85: 0.6, 0.90: 0.4, 0.95: 0.2}   # H1 1.2: 0.4-0.6 at 0.85-0.90; before the first frame at 0.95

MU_RE = re.compile(r"mu(0\.\d+|0_\d+|0\d{2,3})")


def mu_from_name(name):
    """'..._mu0.85_...', '..._mu0_85_...' or '..._mu085_...' -> 0.85; nan if absent"""
    m = MU_RE.search(str(name))
    if not m:
        return float("nan")
    s = m.group(1).replace("_", ".")
    if "." not in s:
        s = "0." + s[1:]
    return float(s)


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def outdir_for(script_file, outdir):
    name = Path(script_file).stem
    out = Path(outdir) / name / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_config(out, args):
    with open(out / "config.json", "w") as fh:
        json.dump({k: getattr(args, k) for k in dir(args) if not k.startswith("_") and
                   not callable(getattr(args, k))}, fh, indent=2, default=str)


def write_table(out, rows, name="data.csv"):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save(fig, out, name="fig"):
    fig.savefig(out / (name + ".png"), bbox_inches="tight")
    fig.savefig(out / (name + ".pdf"), bbox_inches="tight")
    print("figure ->", out / (name + ".png"))


def cli(args_cls, doc, run):
    """argparse mirror of an _Args class (the sysid_scan pattern); bool
    attributes become --flag / --no-flag, everything else typed by default"""
    import argparse
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in dir(args_cls) if not a.startswith("_")]:
        v = getattr(args_cls, k)
        if isinstance(v, bool):
            p.add_argument("--" + k, dest=k, action="store_true", default=v)
            p.add_argument("--no_" + k, dest=k, action="store_false")
        else:
            p.add_argument("--" + k, type=type(v), default=v)
    return run(p.parse_args())


def parse_support(block):
    """'theta_t | RHS+ext library | forced {w div k, k.grad w, J}' ->
    ('theta_t', 'RHS+ext', 'forced', ['w div k', 'k.grad w', 'J']);
    free blocks return kind 'free' and an empty support"""
    parts = [s.strip() for s in block.split("|")]
    if len(parts) < 3:
        return None
    target, lib, rest = parts[0], parts[1].replace(" library", ""), "|".join(parts[2:])
    m = re.match(r"forced\s*\{([^}]*)\}", rest)
    if m:
        return target, lib, "forced", [s.strip() for s in m.group(1).split(",") if s.strip()]
    return target, lib, ("free" if rest.startswith("free") else rest.split()[0]), []


def window_label(r):
    mu, t0, t1, wc = fnum(r.get("mu")), fnum(r.get("t0")), fnum(r.get("t1")), r.get("wclass", "")
    if not np.isfinite(mu):
        return "%s %s" % (r.get("file", ""), wc)
    return "%.2f %s\n%.1f-%.1f" % (mu, wc, t0, t1)
