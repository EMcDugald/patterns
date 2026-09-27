# pipelines/fig_phase_law_windows.py
"""
Figure for Section 3C (Table tab:phase_law_windows): the coefficients of
the forced RCN triple and of the forced law theta_t = a S - d s J, by
window, with the RCN control's response drawn as a band.

Input: runs.csv written by sysid_scan.py (one row per run / file / block;
columns used: file, mu, t0, t1, wclass, domain, route, block, coefs (JSON
term -> coefficient), r2_holdout). The blocks read are

  theta_t | RHS library     | forced {w div k, k.grad w, Lap div k}      (RCN triple)
  theta_t | RHS+ext library | forced {w div k, k.grad w, J}              (the law)
  theta_t | RHS+ext library | forced {w div k, k.grad w, Lap div k, J}   (bending with J admitted)

matched by target, kind and support set, so the library name may differ.
Control rows are those with wclass == "control" (the 1d RCN knee, both
routes); each is drawn as a horizontal line per term, solid for the
estimator route (the response the SH coefficients are compared with)
and dotted for the exact route (RCN's own value). Nothing is fitted.

Panels: (a) RCN triple coefficients per window (stretching pair and the
bending) with the control lines; (b) the law's coefficients a1, a2 (of
w div k, k.grad w) and the raw coefficient of J (= -s d); (c) the
bending coefficient without and with J admitted; (d) holdout R2 of the
two three-term fits.

Windows: every row with domain == --domain, wclass in --classes, and
file not pooled / control, ordered by (mu, t0). --files restricts to file
stems containing the substring (comma list). The four windows of the
paper are the T25 band runs at 0.85 pre, 0.85 straddle, 0.90 post, 0.95
post; if runs.csv holds more (fine cadence, core_excl variants), pass
--stamps with a comma list of run stamps to pick.

Run:
  python fig_phase_law_windows.py --runs results/sysid_scan/runs.csv
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import (read_csv, fnum, parse_support, window_label, outdir_for,  # noqa: E402
                       write_config, write_table, save, cli, plt)

TRIPLE = {"w div k", "k.grad w", "Lap div k"}
LAW = {"w div k", "k.grad w", "J"}
QUAD = {"w div k", "k.grad w", "Lap div k", "J"}


def pick_blocks(rows):
    """key (file, route, t0, t1, stamp) -> dict tag -> row"""
    out = {}
    for r in rows:
        ps = parse_support(r["block"])
        if ps is None or ps[0] != "theta_t" or ps[2] != "forced":
            continue
        sup = set(ps[3])
        if sup == TRIPLE:
            tag = "triple"
        elif sup == LAW:
            tag = "law"
        elif TRIPLE < sup and "J" in sup and len(sup) <= 5:
            tag = "quad"          # RCN triple with J admitted: the four-term block, else RCN + {curl k, J}
        else:
            continue
        key = (r.get("file", ""), r.get("route", ""), r.get("t0", ""), r.get("t1", ""), r.get("stamp", ""))
        d = out.setdefault(key, {})
        if tag in d:
            # the same support can appear under two libraries (RHS and RHS+ext) or two sizes;
            # keep the plain RHS-library triple and the smaller J-admitted block, and say so
            prev = d[tag]
            keep_prev = (tag == "triple" and parse_support(prev["block"])[1] == "RHS") or \
                        (tag == "quad" and len(parse_support(prev["block"])[3]) <= len(sup))
            if prev.get("r2_holdout") != r.get("r2_holdout"):
                print("  note: two '%s' blocks in %s (%s R2 %s | %s R2 %s); keeping %s" %
                      (tag, key[4], prev["block"], prev.get("r2_holdout"), r["block"], r.get("r2_holdout"),
                       "the first" if keep_prev else "the second"))
            if keep_prev:
                continue
        d[tag] = r
    return out


def run(args):
    rows = read_csv(args.runs)
    groups = pick_blocks(rows)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    files = [f.strip() for f in args.files.split(",") if f.strip()]
    stamps = [s.strip() for s in args.stamps.split(",") if s.strip()]

    ctrl = [d for k, d in groups.items() if k[0] == "control"]
    wins = []
    for k, d in groups.items():
        r = d.get("triple") or d.get("law")
        if r is None or k[0] in ("control", "pooled"):
            continue
        if args.require_law and "law" not in d:
            continue
        if r.get("domain", "") != args.domain or r.get("wclass", "") not in classes:
            continue
        if files and not any(f in k[0] for f in files):
            continue
        if stamps and not any(s in k[4] for s in stamps):
            continue
        wins.append((fnum(r["mu"]), fnum(r["t0"]), k, d))
    wins.sort(key=lambda w: (w[0], w[1]))
    if not wins:
        print("no windows matched in", args.runs)
        return 1

    out = outdir_for(__file__, args.outdir)
    write_config(out, args)
    labels = [window_label(d.get("triple") or d.get("law")) for _, _, _, d in wins]
    xs = np.arange(len(wins))
    tab = []

    def coef(d, tag, term):
        if tag not in d:
            return float("nan")
        return float(json.loads(d[tag]["coefs"]).get(term, float("nan")))

    def r2(d, tag):
        return fnum(d[tag]["r2_holdout"]) if tag in d else float("nan")

    def ctrl_lines(a, term, first):
        """the RCN control's value of a term, one line per route: the
        estimator route (solid, what SH coefficients are compared with)
        and the exact route (dotted, RCN's own value through the weak form)"""
        for d in ctrl:
            v = coef(d, "triple", term)
            if not np.isfinite(v):
                continue
            route = d["triple"].get("route", "")
            exact = route.startswith("exact")
            a.axhline(v, color="0.4", ls=":" if exact else "-", lw=0.9, alpha=0.8,
                      label=(("RCN control, %s route" % (route or "?")) if first else None))

    fig, ax = plt.subplots(2, 2, figsize=(8.4, 6.0))
    ax = ax.ravel()
    # (a) RCN triple with the control band
    for term, mk in (("w div k", "o"), ("k.grad w", "s"), ("Lap div k", "^")):
        v = [coef(d, "triple", term) for _, _, _, d in wins]
        ax[0].plot(xs, v, mk + "-", label=term)
        ctrl_lines(ax[0], term, first=(term == "w div k"))
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_title("(a) forced RCN triple $\\{w\\nabla\\cdot k,\\ k\\cdot\\nabla w,\\ \\nabla^2\\nabla\\cdot k\\}$", fontsize=8)
    ax[0].set_ylabel("coefficient (estimator response)")
    ax[0].legend(fontsize=6, loc="lower left")
    # (b) the law
    for term, mk, lab in (("w div k", "o", "$a_1$ ($w\\nabla\\cdot k$)"), ("k.grad w", "s", "$a_2$ ($k\\cdot\\nabla w$)"),
                          ("J", "D", "coefficient of $J$ ($= -s\\,d$)")):
        v = [coef(d, "law", term) for _, _, _, d in wins]
        ax[1].plot(xs, v, mk + "-", label=lab)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_title("(b) forced law $\\theta_t = a S - d s J$ at equal size", fontsize=8)
    ax[1].legend(fontsize=7)
    # (c) bending with and without J
    b0 = [coef(d, "triple", "Lap div k") for _, _, _, d in wins]
    b1 = [coef(d, "quad", "Lap div k") for _, _, _, d in wins]
    ax[2].plot(xs, b0, "^-", label="bending, $J$ absent (RCN triple)")
    ax[2].plot(xs, b1, "v-", label="bending, $J$ admitted (four terms)")
    ctrl_lines(ax[2], "Lap div k", first=True)
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_title("(c) the bending coefficient", fontsize=8)
    ax[2].legend(fontsize=6, loc="center left")
    # (d) R2
    ax[3].plot(xs, [r2(d, "triple") for _, _, _, d in wins], "^-", label="RCN triple")
    ax[3].plot(xs, [r2(d, "law") for _, _, _, d in wins], "D-", label="law with $J$")
    ax[3].set_title("(d) holdout $R^2$ (half of the boundary in $x$)", fontsize=8)
    ax[3].legend(fontsize=7)
    for a in ax:
        a.set_xticks(xs)
        a.set_xticklabels(labels, fontsize=7)
    fig.suptitle("phase rate on the band, R = 1/2, by window (mg estimator, weak regression)", fontsize=9)
    fig.tight_layout()
    save(fig, out)

    for (mu, t0, k, d), lab in zip(wins, labels):
        tab.append(dict(file=k[0], route=k[1], stamp=k[4], mu=mu, t0=t0, t1=fnum((d.get("triple") or d["law"])["t1"]),
                        wclass=(d.get("triple") or d["law"]).get("wclass", ""),
                        c1=coef(d, "triple", "w div k"), c2=coef(d, "triple", "k.grad w"), c3=coef(d, "triple", "Lap div k"),
                        r2_triple=r2(d, "triple"), a1=coef(d, "law", "w div k"), a2=coef(d, "law", "k.grad w"),
                        cJ=coef(d, "law", "J"), r2_law=r2(d, "law"), bending_with_J=coef(d, "quad", "Lap div k"),
                        r2_quad=r2(d, "quad")))
    for d in ctrl:
        tab.append(dict(file="control", route=d.get("triple", d.get("law", {})).get("route", ""),
                        c1=coef(d, "triple", "w div k"), c2=coef(d, "triple", "k.grad w"), c3=coef(d, "triple", "Lap div k"),
                        r2_triple=r2(d, "triple")))
    write_table(out, tab)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    runs = str(Path("results") / "sysid_scan" / "runs.csv")
    outdir = "results"
    domain = "band"
    classes = "pre,straddle,post"     # window classes to show
    files = ""                        # comma list of file-stem substrings, empty = all
    stamps = ""                       # comma list of run-stamp substrings, empty = all
    require_law = True                # drop windows on which the {S, J} law was not fitted


if __name__ == "__main__":
    sys.exit(cli(_Args, __doc__, run))
