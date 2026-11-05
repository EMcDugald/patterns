"""
hysteresis_compare.py  v1.0  (2 Nov 2026)   the knee-started and the dislocation-started branches side by side

Plain terms. Reads two survey tables written by large_sweep_charge_circ_v13.py (t200/survey_table.csv): the large
sweep started from knees (--up) and the ramp started from dislocations (--down, from dislocation_ramp_down.py). For
every angle in both it lines up the regime, the director charge per L, the circulation at the on-line dips, the
fraction of on-line dips at or above one turn, and the line energy E_GB, and marks the angles where the two branches
end in different regimes. Same regimes everywhere: no hysteresis within the run length. Different regimes over a
range of angles: a hysteresis window, and there the branch with the lower E_GB is the lower-energy state.

Outputs under --outdir (anchored to this script's folder when relative): compare.csv, compare.md, fig_branches.png,
report.txt. Measures nothing itself. Python 3.8 compatible.

Changelog
  v1.0  (2 Nov 2026) first version.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VERSION = "hysteresis_compare v1.0"
KEYS = ["regime", "charge_abs_per_L", "dip_circ_on_med", "frac_on_circ_hi", "n_dips_on", "n_dips_off", "E_GB"]
COL = {"knee": "#e9e7e2", "intermediate": "#f6e3b4", "dislocation": "#cfe0f5"}


def read(path):
    rows = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows[round(float(r["mu"]), 3)] = r
    return rows


def num(r, k):
    try:
        return float(r.get(k, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def main(a):
    out = Path(a.outdir)
    if not out.is_absolute():
        out = HERE / out
    out.mkdir(parents=True, exist_ok=True)
    up_p, dn_p = Path(a.up), Path(a.down)
    up_p = up_p if up_p.is_absolute() else HERE / up_p
    dn_p = dn_p if dn_p.is_absolute() else HERE / dn_p
    up, dn = read(up_p), read(dn_p)
    mus = sorted(set(up) & set(dn))
    lines = [VERSION, "up (from knees): %s" % up_p, "down (from dislocations): %s" % dn_p, "angles in both: %d" % len(mus), ""]
    table = []
    for m in mus:
        u, d = up[m], dn[m]
        rec = dict(mu=m, alpha_deg=float(np.degrees(np.arcsin(m))))
        for k in KEYS:
            rec[k + "_up"] = u.get(k, "") if k == "regime" else num(u, k)
            rec[k + "_down"] = d.get(k, "") if k == "regime" else num(d, k)
        rec["same_regime"] = rec["regime_up"] == rec["regime_down"]
        rec["dE_down_minus_up"] = rec["E_GB_down"] - rec["E_GB_up"]
        table.append(rec)
    if not table:
        lines.append("no angle in both tables"); (out / "report.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines)); return 1
    keys = list(table[0].keys())
    with open(out / "compare.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(table)
    show = ["mu", "alpha_deg", "regime_up", "regime_down", "charge_abs_per_L_up", "charge_abs_per_L_down",
            "dip_circ_on_med_up", "dip_circ_on_med_down", "E_GB_up", "E_GB_down", "dE_down_minus_up"]
    with open(out / "compare.md", "w") as fh:
        fh.write("| " + " | ".join(show) + " |\n|" + "---|" * len(show) + "\n")
        for r in table:
            fh.write("| " + " | ".join(("%.4g" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in show) + " |\n")
    diff = [r for r in table if not r["same_regime"]]
    if diff:
        lines.append("regimes differ at %d angles: %s" % (len(diff), ", ".join(
            "%.2f (%.1f deg: up %s, down %s, E down - up %+.4g)" % (r["mu"], r["alpha_deg"], r["regime_up"], r["regime_down"],
                                                                 r["dE_down_minus_up"]) for r in diff)))
        lines.append("window: mu %.2f to %.2f (%.1f to %.1f deg)" % (min(r["mu"] for r in diff), max(r["mu"] for r in diff),
                                                                    min(r["alpha_deg"] for r in diff), max(r["alpha_deg"] for r in diff)))
    else:
        lines.append("the two branches end in the same regime at every angle in both tables: no hysteresis within the run length")
    for name, b in (("up", "_up"), ("down", "_down")):
        dis = [r["mu"] for r in table if r["regime" + b] == "dislocation"]
        ch = [r["mu"] for r in table if r["regime" + b] != "knee"]
        lines.append("%s: lowest dislocation angle %s; lowest charged angle %s" % (
            name, ("%.2f" % min(dis)) if dis else "none", ("%.2f" % min(ch)) if ch else "none"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mu = np.array([r["mu"] for r in table])
    fig, axs = plt.subplots(4, 1, figsize=(8, 9.5), sharex=True, gridspec_kw=dict(height_ratios=[0.5, 1, 1, 1]))
    ax = axs[0]
    for j, b in enumerate(("_up", "_down")):
        for i, r in enumerate(table):
            ax.add_patch(plt.Rectangle((mu[i] - 0.005, 1 - j), 0.01, 0.9, color=COL.get(r["regime" + b], "w"), lw=0))
    ax.set_ylim(0, 2); ax.set_yticks([0.45, 1.45]); ax.set_yticklabels(["from dislocations", "from knees"], fontsize=8)
    ax.set_title("regime at t = 200 (knee: grey, intermediate: amber, dislocation: blue)", fontsize=9)
    for ax, k, lab in ((axs[1], "charge_abs_per_L", "half-turn cells per L"), (axs[2], "dip_circ_on_med", "|Gamma| at on-line dips (turns)"),
                       (axs[3], "E_GB", "line energy E_GB")):
        ax.plot(mu, [r[k + "_up"] for r in table], "o-", color="#1f4f8f", ms=3, label="from knees")
        ax.plot(mu, [r[k + "_down"] for r in table], "s--", color="#b5531f", ms=3, mfc="none", label="from dislocations")
        ax.set_ylabel(lab, fontsize=8); ax.legend(fontsize=7)
    axs[2].axhline(0.9, color="grey", lw=0.6, ls=":")
    axs[3].set_xlabel("mu = sin(far-field angle)")
    fig.tight_layout(); fig.savefig(out / "fig_branches.png", dpi=150); plt.close(fig)
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print("outputs ->", out)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    up = str(Path("results") / "1030_w04_charge_circ_large_sweep" / "t200" / "survey_table.csv")
    down = str(Path("results") / "dislocation_ramp_down" / "survey" / "t200" / "survey_table.csv")
    outdir = str(Path("results") / "hysteresis_compare")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [k for k in vars(_Args) if not k.startswith("_")]:
        p.add_argument("--" + k, type=type(getattr(_Args, k)), default=getattr(_Args, k))
    sys.exit(main(p.parse_args()))
