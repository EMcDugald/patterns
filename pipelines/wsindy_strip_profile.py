# pipelines/wsindy_strip_profile.py
"""
Distance-from-the-row profile of the theta_t fits of wsindy_theta_t v1.8.

A set of wsindy_theta_t runs on the domain "shell" with (band_to,
far_from) = (lo, hi) cuts the query centres into strips by distance to
the defect row, lo <= d_row <= hi in spacings L. This script collects
the strips of each file (config.json + fits.json under a results root)
and draws, against the strip midpoint:

  (a) bending over stretching (axis clipped to [-3, 1.5]; off-scale
      values are printed beside the edge marker, they mark strips where
      the stretching column is too small to identify): c(Lap div k) / mean(c(w div k), c(k.grad w))
      for the RCN triple and for RCN + J, and c(Lap div k) / c(S) for the
      tied {S, Lap div k}; reference lines at the RCN value -1/2
      (alpha = eta = tau) and at the estimator-route value of the RCN
      control (-1.36 / 1.415 = -0.96, radius 1.5, beta0 1.8);
  (b) unexplained holdout variance 1 - R2 (log axis) of the RCN triple,
      tied {S, Lap div k}, {S pair, J}, {S pair, k.grad A}, the
      Cross-Newell {S_A} and {S_A, Lap div k}, and the best of the 560
      triples;
  (c) the target scale RMS Int[theta_t phi].

Reading: a strip's test functions reach radius x L beyond its edges, so a
strip of width 0.5 L at radius 1.5 L locates the CENTRES, not the data;
compare a radius-1.0 set before reading structure finer than about one
spacing. Coefficients in strips where a column is small (w div k far from
the row) are not identifiable; panel (a) uses ratios of the tied law for
that reason and panel (b) is the primary reading (holdout R2, never
collinear coefficient values).

Outputs (relative): results/wsindy_strip_profile/<tag>/strip_profile.csv,
  strip_profile.png

Changelog
---------
v1.0  first version (4 Oct). Checked on synthetic shell runs of
      wsindy_theta_t v1.8 (strips 0-0.5 ... 4-6 L on the synthetic knee
      file): every strip read, ordering by midpoint, csv and figure.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent

LAWS = [  # (fits key, label) in fixed colour order
    ("theta_rhs_forced", "RCN triple"),
    ("theta_tied_SB", "tied {S, Lap div k}"),
    ("theta_rhs_forced_SJ", "{S pair, J}"),
    ("theta_rhs_forced_SkA", "{S pair, k.grad A}"),
    ("theta_cn_SA", "CN {S_A}"),
    ("theta_cn_SAB", "CN {S_A, Lap div k}"),
    ("best3", "best of all triples"),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#52514e"]
RATIOS = [("rcn", "RCN triple"), ("rcnJ", "RCN + J"), ("tiedSB", "tied {S, Lap div k}")]


def strip_of(cfg):
    if cfg.get("domain") == "shell":
        return float(cfg["band_to"]), float(cfg["far_from"])
    if cfg.get("domain") == "band":
        return 0.0, float(cfg["band_to"])
    if cfg.get("domain") == "row":
        return 0.0, float(cfg["row_to"])
    return None


def collect(root):
    rows = []
    for cfgp in sorted(Path(root).rglob("config.json")):
        fp = cfgp.parent / "fits.json"
        if not fp.exists():
            continue
        cfg = json.load(open(cfgp))
        st = strip_of(cfg)
        if st is None or cfg.get("mode") != "data":
            continue
        F = json.load(open(fp))
        for stem, f in F.items():
            if stem == "pooled" or not isinstance(f, dict) or "theta_rhs_forced" not in f:
                continue
            r = dict(file=stem, lo=st[0], hi=st[1], mid=0.5 * (st[0] + st[1]), radius=float(cfg["radius"]),
                     t0=float(cfg["t0"]), t1=float(cfg["t1"]), path=str(cfgp.parent))
            r["mu"] = next((float(tok[2:]) for tok in stem.split("_") if tok.startswith("mu")), np.nan)
            r["target_rms"] = f.get("target_rms", np.nan)
            r["n_rows"] = f.get("n_rows", np.nan)
            for key, _ in LAWS:
                if key == "best3":
                    ex = f.get("theta_exhaustive", {})
                    e3 = ex.get("3", ex.get(3, {}))
                    r["r2_best3"] = e3.get("best_r2", np.nan) if isinstance(e3, dict) else np.nan
                    continue
                g = f.get(key) or {}
                r["r2_" + key] = g.get("r2_holdout", np.nan) if not g.get("skipped") else np.nan
            c = (f.get("theta_rhs_forced") or {}).get("coefs", {})
            st_ = 0.5 * (c.get("w div k", np.nan) + c.get("k.grad w", np.nan))
            r["ratio_rcn"] = c.get("Lap div k", np.nan) / st_ if st_ else np.nan
            c = (f.get("theta_rhs_forced_rcn_J") or {}).get("coefs", {})
            st_ = 0.5 * (c.get("w div k", np.nan) + c.get("k.grad w", np.nan))
            r["ratio_rcnJ"] = c.get("Lap div k", np.nan) / st_ if st_ else np.nan
            c = (f.get("theta_tied_SB") or {}).get("coefs", {})
            r["ratio_tiedSB"] = c.get("Lap div k", np.nan) / c.get("S", np.nan) if c.get("S") else np.nan
            r["bend_rcn"] = (f.get("theta_rhs_forced") or {}).get("coefs", {}).get("Lap div k", np.nan)
            rows.append(r)
    return rows


def figure(rows, path, title):
    groups = {}
    for r in rows:
        groups.setdefault((r["mu"], r["radius"]), []).append(r)
    keys = sorted(groups)
    nc = len(keys)
    fig, axs = plt.subplots(3, nc, figsize=(4.2 * nc, 9.0), squeeze=False, sharex=True)
    for j, key in enumerate(keys):
        G = sorted(groups[key], key=lambda r: r["mid"])
        x = np.array([r["mid"] for r in G])
        ax = axs[0, j]
        lo_, hi_ = -3.0, 1.5
        for ik, ((k, lab), col) in enumerate(zip(RATIOS, COLORS)):
            v = np.array([r["ratio_" + k] for r in G], float)
            ax.plot(x, np.clip(v, lo_, hi_), "-o", color=col, lw=2, ms=6, label=lab)
            for xi, vi in zip(x, v):                       # off-scale ratios: edge marker + value
                if np.isfinite(vi) and (vi < lo_ or vi > hi_):
                    ax.annotate("%.3g" % vi, (xi, np.clip(vi, lo_, hi_)), textcoords="offset points",
                                xytext=(4, (4 + 9 * ik) if vi > hi_ else (-10 - 9 * ik)), fontsize=7, color=col)
        ax.set_ylim(lo_ - 0.1, hi_ + 0.1)
        ax.axhline(-0.5, color="0.3", lw=0.8, ls="--")
        ax.axhline(-0.96, color="0.3", lw=0.8, ls=":")
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_title("mu %.2f, radius %.3g L, t %.2f-%.2f" % (key[0], key[1], G[0]["t0"], G[0]["t1"]), fontsize=9)
        if j == 0:
            ax.set_ylabel("bending / stretching\n(-0.5 RCN; -0.96 control, estimator)", fontsize=8)
        ax = axs[1, j]
        for (k, lab), col in zip(LAWS, COLORS):
            v = np.array([1.0 - r["r2_" + k] for r in G])
            ax.plot(x, np.clip(v, 1e-5, None), "-o", color=col, lw=2, ms=6, label=lab)
        ax.set_yscale("log")
        if j == 0:
            ax.set_ylabel("unexplained holdout variance 1 - R2", fontsize=8)
        ax = axs[2, j]
        ax.plot(x, [r["target_rms"] for r in G], "-o", color="#52514e", lw=2, ms=6)
        for r in G:
            ax.plot([r["lo"], r["hi"]], [r["target_rms"]] * 2, "-", color="0.6", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("strip midpoint, distance from the row (L)")
        if j == 0:
            ax.set_ylabel("RMS Int[theta_t phi]", fontsize=8)
    axs[0, -1].legend(fontsize=7, loc="best")
    axs[1, -1].legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(args):
    rows = collect(args.root)
    out = Path(args.outdir) / (args.tag or Path(args.root).name)
    out.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("no strip runs found under", args.root)
        return 1
    keys = list(rows[0].keys())
    with open(out / "strip_profile.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in sorted(rows, key=lambda r: (r["mu"], r["radius"], r["mid"])):
            wr.writerow(r)
    figure(rows, out / "strip_profile.png", "theta_t fits by distance from the row (%s)" % args.root)
    for r in sorted(rows, key=lambda r: (r["mu"], r["radius"], r["mid"])):
        print("mu %.2f rad %.2g strip %.2g-%.2g  n %5s  rms %.2e  ratio RCN %+.2f tiedSB %+.2f | 1-R2 RCN %.4f SJ %.4f SkA %.4f SA %.4f best3 %.4f" % (
            r["mu"], r["radius"], r["lo"], r["hi"], r["n_rows"], r["target_rms"], r["ratio_rcn"], r["ratio_tiedSB"],
            1 - r["r2_theta_rhs_forced"], 1 - r["r2_theta_rhs_forced_SJ"], 1 - r["r2_theta_rhs_forced_SkA"],
            1 - r["r2_theta_cn_SA"], 1 - r["r2_best3"]))
    print("outputs ->", out)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    root = str(Path("results") / "1004c_wsindy_strips")
    outdir = str(Path("results") / "wsindy_strip_profile")
    tag = ""


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in dir(_Args) if not a.startswith("_")]:
        p.add_argument("--" + k, type=type(getattr(_Args, k)), default=getattr(_Args, k))
    sys.exit(main(p.parse_args()))
