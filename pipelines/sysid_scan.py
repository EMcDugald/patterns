# pipelines/sysid_scan.py
"""
Persistence scanner for the system-identification campaign (Writeup 4).

Walks one or more results roots, collects every wsindy_theta_t run
(fits.json when present, else the report.txt of v1.2+) and every
rcn_balance_ledger run (ledger.csv), tags each block with its context
(mu, window t0-t1, domain, core_excl, radius, t_half, lib_ext, route or
file, pooled) and writes:

  runs.csv          one row per (run, file/route, block): context, block
                    label, kind (free/forced/residual), holdout R2, n,
                    support, coefficients (JSON), collinearity (cond,
                    most correlated pair) when the report carries it.
  persistence.txt   per block label, per term: number of runs in which
                    the term is selected (free) or present (forced), the
                    sign consistency (fraction of runs with the majority
                    sign), and min / median / max of the coefficient,
                    across all contexts and again split by mu and by
                    window class (pre / straddle / post / whole / tail).
  tables.md         Writeup 4 table stubs filled from the scanned runs:
                    (ii) theta_t forced RCN by window and domain with the
                    control beside; (iii) the extended-library free law
                    across windows; (iv) w_t and A_t forced laws; (v)
                    ledger last-frame beta, lambda, lambda_J, m_J per mu.
                    Every cell carries the run stamp as provenance; cells
                    with no run are "\\todo{}".
  scores.csv        the v1.6 prediction scores (law, horizon, columns)
                    when present in fits.json.

Window classes are assigned from the event times of record (RESULTS
Section 1): pre = t1 <= t_wind_first, post = t0 >= t_wind_last,
straddle = otherwise inside, whole = t1 - t0 > 10, tail = t0 >= 90.

Nothing here fits anything; it reads. Numbers in the tables are copied
from the run outputs with their stamps.

Changelog
---------
v1.0  first version (18 Sep). Tested on the 0915 report-only zips (v1.4
      reports), the v1.6 control and tail runs, and the v1.5 ledger runs
      on the 11 tails.

Run:
  python sysid_scan.py --roots results/0915_wsindy_theta_t_v14,results/0918_wsindy_v16,results/rcn_balance_ledger --outdir results/sysid_scan
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# event times of record (RESULTS_mg_campaign_0916 Section 1), winding first..last
T_WIND = {0.85: (3.8, 5.4), 0.86: (3.0, 3.6), 0.87: (2.4, 2.8), 0.88: (2.0, 2.2), 0.89: (1.6, 1.8),
          0.90: (1.2, 1.6), 0.95: (0.6, 0.6), 0.975: (0.2, 0.2), 0.99: (0.2, 0.2)}

BLOCK_RE = re.compile(r"^\s{2}(?P<label>[^:]+?):\s+(?P<rest>.+)$")
COEF_RE = re.compile(r"^([+-]?[0-9.]+(?:e[+-]?\d+)?)\s+(.+)$")
R2_RE = re.compile(r"holdout R2\s+(-?[0-9.]+(?:e[+-]?\d+)?)")
N_RE = re.compile(r"n\s+(\d+)\)")
COLL_RE = re.compile(r"cond\s+([0-9.e+-]+),\s+(?:most correlated pair|max) \|r\|\s+([0-9.]+)\s+\{([^}]*)\}")
STAMP_RE = re.compile(r"rad(?P<rad>[0-9.]+)_eta(?P<eta>[0-9.]+)_th(?P<th>\d+)_mc(?P<mc>[0-9.]+)_(?P<dom>[a-z]+)_ce(?P<ce>[0-9.]+)_ext(?P<ext>\d)_t(?P<t0>-?[0-9.]+)-(?P<t1>[0-9.e+]+)")
MU_RE = re.compile(r"mu(?P<mu>[0-9.]+)")
DATA_RE = re.compile(r"^data:\s+(?P<stem>\S+)\s+mu\s+(?P<mu>[0-9.]+)\s+t in \[(?P<t0>[0-9.]+),\s+(?P<t1>[0-9.]+)\]\s+\((?P<nt>\d+) frames, dt (?P<dt>[0-9.]+)\).*domain\s+(?P<dom>\w+)")


def parse_coefs(text):
    """'+0.86 w div k + +0.63 k.grad w + -0.185 w   holdout R2 ...' -> dict"""
    head = text.split("   holdout R2")[0]
    if "(empty)" in head or "SKIPPED" in head:
        return {}
    out = {}
    for piece in head.split(" + "):
        m = COEF_RE.match(piece.strip())
        if m:
            out[m.group(2).strip()] = float(m.group(1))
    return out


def window_class(mu, t0, t1):
    if t0 is None or t1 is None:
        return ""
    if t0 >= 90:
        return "tail"
    if t1 - t0 > 10:
        return "whole"
    tw = T_WIND.get(round(mu, 3))
    if tw is None:
        return "window"
    if t1 <= tw[0]:
        return "pre"
    if t0 >= tw[1]:
        return "post"
    return "straddle"


def scan_wsindy_dir(run_dir):
    """one run directory (contains report.txt); returns list of block rows"""
    rows, scores = [], []
    rep = (run_dir / "report.txt").read_text().splitlines()
    cfg = json.load(open(run_dir / "config.json")) if (run_dir / "config.json").exists() else {}
    fits = json.load(open(run_dir / "fits.json")) if (run_dir / "fits.json").exists() else None
    stamp = run_dir.name
    ms = STAMP_RE.search(stamp)
    ctx0 = dict(run=str(run_dir), stamp=stamp, version=rep[0].split()[1] if rep and rep[0].startswith("wsindy") else "",
                radius=float(ms["rad"]) if ms else cfg.get("radius", np.nan),
                t_half=int(ms["th"]) if ms else cfg.get("t_half", np.nan),
                domain=ms["dom"] if ms else cfg.get("domain", ""),
                core_excl=float(ms["ce"]) if ms else cfg.get("core_excl", np.nan),
                lib_ext=int(ms["ext"]) if ms else cfg.get("lib_ext", np.nan),
                mode=cfg.get("mode", "control" if "control_rcn" in str(run_dir) else "data"))
    # walk the report: sections start at "data: <stem>" / "pooled:" / "[route x]"
    ctx = dict(ctx0, file="", mu=np.nan, t0=np.nan, t1=np.nan, route="", wclass="")
    for ln in rep:
        md = DATA_RE.match(ln)
        if md:
            mu = float(md["mu"])
            ctx = dict(ctx0, file=md["stem"], mu=mu, t0=float(md["t0"]), t1=float(md["t1"]), route="",
                       wclass=window_class(mu, float(md["t0"]), float(md["t1"])), domain=md["dom"])
            continue
        if ln.startswith("pooled:"):
            ctx = dict(ctx0, file="pooled", mu=np.nan, t0=ctx["t0"], t1=ctx["t1"], route="pooled",
                       wclass=ctx["wclass"])
            continue
        if ln.startswith("RCN control:"):
            m = re.search(r"mu ([0-9.]+).*beta0 ([0-9.]+)", ln)
            ctx = dict(ctx0, file="control", mu=float(m.group(1)) if m else np.nan, t0=0.0, t1=np.nan,
                       route="", wclass="control", beta0=float(m.group(2)) if m else np.nan)
            continue
        m = re.match(r"^\s+\[route (\w+)\]", ln)
        if m:
            ctx = dict(ctx, route=m.group(1))
            continue
        mb = BLOCK_RE.match(ln)
        if mb and "holdout R2" in ln:
            label = mb["label"].strip()
            parts = [p.strip() for p in label.split("|")]
            if len(parts) < 3:
                continue
            target, lib, kind = parts[0], parts[1].replace(" library", ""), "|".join(parts[2:])
            r2 = R2_RE.search(ln); n = N_RE.search(ln); cl = COLL_RE.search(ln)
            rows.append(dict(ctx, target=target, library=lib, kind=("free" if kind.startswith("free") else
                                                                    "residual" if "residual" in lib else "forced"),
                             block=label, r2_holdout=float(r2.group(1)) if r2 else np.nan,
                             n=int(n.group(1)) if n else -1,
                             coefs=json.dumps(parse_coefs(mb["rest"])),
                             cond=float(cl.group(1)) if cl else np.nan,
                             max_corr=float(cl.group(2)) if cl else np.nan,
                             pair=cl.group(3) if cl else ""))
        mcl = re.search(r"library conditioning: cond\s+([0-9.e+-]+), most correlated pair \|r\|\s+([0-9.]+)\s+\{([^}]*)\}", ln)
        if mcl and rows and rows[-1]["run"] == str(run_dir) and not np.isfinite(rows[-1]["cond"]):
            rows[-1]["cond"] = float(mcl.group(1)); rows[-1]["max_corr"] = float(mcl.group(2)); rows[-1]["pair"] = mcl.group(3)
            continue
        ms_ = re.match(r"^\s{4}(?P<law>.+?):\s+h (?P<cells>.+)$", ln)
        if ms_ and "||" in ln or (ms_ and re.search(r"\| [0-9.]+, [0-9.]+ \|", ln)):
            for cell in ("h " + ms_["cells"]).split("  ||  "):
                mc = re.match(r"h (\d+) \| ([0-9.]+), ([0-9.]+) \| ([0-9.]+), ([0-9.]+)", cell.strip())
                if mc:
                    scores.append(dict(run=str(run_dir), stamp=stamp, file=ctx["file"], route=ctx["route"], mu=ctx["mu"],
                                       t0=ctx["t0"], t1=ctx["t1"], wclass=ctx["wclass"], domain=ctx["domain"],
                                       law=ms_["law"].strip(), h=int(mc.group(1)), euler_pw=float(mc.group(2)),
                                       multi_pw=float(mc.group(3)), euler_wk=float(mc.group(4)), multi_wk=float(mc.group(5))))
    return rows, scores


def scan_ledger_dir(run_dir):
    L = list(csv.DictReader(open(run_dir / "ledger.csv")))
    if not L:
        return []
    stem = run_dir.parent.name
    m = MU_RE.search(stem)
    mu = float(m["mu"]) if m else np.nan
    out = []
    for route in sorted({r["route"] for r in L}):
        R = [r for r in L if r["route"] == route and r["domain"] in ("band", "row")]
        if not R:
            continue
        tmax = max(float(r["t"]) for r in R)
        for r in R:
            if float(r["t"]) != tmax:
                continue
            g = lambda k: float(r[k]) if k in r and r[k] not in ("", "nan") else np.nan
            out.append(dict(run=str(run_dir), stamp=run_dir.name, stem=stem, mu=mu, route=route, domain=r["domain"],
                            t=tmax, n_cores=int(float(r.get("n_cores", -1))), rms_theta_t=g("rms_theta_t"),
                            beta=g("beta"), beta_r2=g("beta_r2"), lam=g("lam"), lam_r2=g("lam_r2"),
                            lam_J=g("lam_J"), m_J=g("m_J"), lamJ_r2=g("lamJ_r2"),
                            chi_ratio_b1=g("chi_ratio_b1"), c_lapchi_b1=g("c_lapchi_b1"),
                            c_divchik_b1=g("c_divchik_b1"), c_J_b1=g("c_J_b1")))
    return out


def persistence(rows):
    """per block label, per term: presence count, sign consistency, coef range; overall and by mu / wclass"""
    lines = []
    by_block = defaultdict(list)
    for r in rows:
        by_block[r["block"]].append(r)
    for block in sorted(by_block):
        R = by_block[block]
        lines.append("%s   (%d runs)" % (block, len(R)))
        for split_name, keyf in (("all", lambda r: "all"), ("by wclass", lambda r: r["wclass"] or "?"),
                                 ("by mu", lambda r: ("mu %.3f" % r["mu"]) if np.isfinite(r["mu"]) else "pooled/ctrl")):
            groups = defaultdict(list)
            for r in R:
                groups[keyf(r)].append(r)
            for g in sorted(groups):
                G = groups[g]
                terms = defaultdict(list)
                for r in G:
                    for k, v in json.loads(r["coefs"]).items():
                        terms[k].append(v)
                if split_name != "all":
                    lines.append("    [%s = %s, %d runs, R2 %s]" % (split_name, g, len(G),
                                 " ".join("%.3f" % r["r2_holdout"] for r in G)))
                else:
                    lines.append("    [all %d runs]" % len(G))
                for k in sorted(terms, key=lambda t: -len(terms[t])):
                    v = np.array(terms[k])
                    sgn = np.sign(v); maj = 1.0 if (sgn > 0).sum() >= (sgn < 0).sum() else -1.0
                    lines.append("      %-16s in %2d/%2d  sign %s %3.0f%%  coef min %+.3g median %+.3g max %+.3g" %
                                 (k, len(v), len(G), "+" if maj > 0 else "-", 100 * (sgn == maj).mean(), v.min(), np.median(v), v.max()))
        lines.append("")
    return "\n".join(lines)


def fmt_coefs(c, order):
    return ", ".join("%s %+.2f" % (k, c[k]) for k in order if k in c) + \
           ("".join(", %s %+.2f" % (k, v) for k, v in c.items() if k not in order) if any(k not in order for k in c) else "")


def tables(rows, ledger_rows, scores):
    T = []
    forced_rcn = [r for r in rows if r["block"].startswith("theta_t | RHS library | forced {w div k, k.grad w, Lap div k}")]
    ctrl = [r for r in forced_rcn if r["wclass"] == "control"]
    T.append("## Table (ii): theta_t forced RCN {w div k, k.grad w, Lap div k} by window and domain, control beside\n")
    T.append("| mu | window | class | domain | core_excl | w div k | k.grad w | Lap div k | R2 | n | stamp |")
    T.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(ctrl, key=lambda r: r["route"]):
        c = json.loads(r["coefs"])
        T.append("| control %s | beta0 %.2g | control | all | - | %+.3f | %+.3f | %+.3f | %.4f | %d | %s |" %
                 (r["route"], r.get("beta0", np.nan), c.get("w div k", np.nan), c.get("k.grad w", np.nan), c.get("Lap div k", np.nan), r["r2_holdout"], r["n"], r["stamp"]))
    for r in sorted([r for r in forced_rcn if r["wclass"] != "control"], key=lambda r: (r["mu"] if np.isfinite(r["mu"]) else 9, r["t0"])):
        c = json.loads(r["coefs"])
        T.append("| %s | %.1f-%.1f | %s | %s | %.2g | %+.2f | %+.2f | %+.2f | %.3f | %d | %s |" %
                 (("%.2f" % r["mu"]) if np.isfinite(r["mu"]) else "pooled", r["t0"], r["t1"], r["wclass"], r["domain"], r["core_excl"],
                  c.get("w div k", np.nan), c.get("k.grad w", np.nan), c.get("Lap div k", np.nan), r["r2_holdout"], r["n"], r["stamp"]))
    if not forced_rcn:
        T.append("| \\todo{no runs scanned} | | | | | | | | | | |")
    T.append("\n## Table (iii): extended-library free law across windows (theta_t | RHS+ext | free)\n")
    T.append("| mu | window | class | support and coefficients | R2 | most correlated pair | stamp |")
    T.append("|---|---|---|---|---|---|---|")
    ext = [r for r in rows if r["block"].startswith("theta_t | RHS+ext library | free")]
    for r in sorted(ext, key=lambda r: (r["mu"] if np.isfinite(r["mu"]) else 9, r["t0"])):
        c = json.loads(r["coefs"])
        T.append("| %s | %.1f-%.1f | %s | %s | %.3f | %s |r| %.2f | %s |" %
                 (("%.2f" % r["mu"]) if np.isfinite(r["mu"]) else "pooled", r["t0"], r["t1"], r["wclass"],
                  fmt_coefs(c, ["w div k", "k.grad w", "Lap div k", "w", "A_dev", "A_dev div k", "k.grad A", "curl k", "J", "chi_J"]),
                  r["r2_holdout"], r["pair"], r["max_corr"] if np.isfinite(r["max_corr"]) else np.nan, r["stamp"]))
    if not ext:
        T.append("| \\todo{no runs scanned} | | | | | | |")
    T.append("\n## Table (iii-b): three-term competitors at equal size (v1.6), holdout R2 by window\n")
    T.append("| mu | window | class | forced RCN {S, Lap div k} | {S, J} | {S, A_dev div k} | {S, chi_J} | stamp |")
    T.append("|---|---|---|---|---|---|---|---|")
    keyset = defaultdict(dict)
    for r in rows:
        for tag, pref in (("rcn", "theta_t | RHS library | forced {w div k, k.grad w, Lap div k}"),
                          ("SJ", "theta_t | RHS+ext library | forced {w div k, k.grad w, J}"),
                          ("SA", "theta_t | RHS+ext library | forced {w div k, k.grad w, A_dev div k}"),
                          ("SchiJ", "theta_t | RHS+ext library | forced {w div k, k.grad w, chi_J}")):
            if r["block"].startswith(pref):
                keyset[(r["run"], r["file"], r["route"])][tag] = r
    any_comp = False
    for key, d in sorted(keyset.items()):
        if "SJ" not in d:
            continue
        any_comp = True
        r = d["rcn"] if "rcn" in d else d["SJ"]
        cell = lambda t: ("%.4f" % d[t]["r2_holdout"]) if t in d else "\\todo{}"
        T.append("| %s | %.1f-%.1f | %s | %s | %s | %s | %s | %s |" %
                 (("%.2f" % r["mu"]) if np.isfinite(r["mu"]) else r["file"], r["t0"], r["t1"], r["wclass"],
                  cell("rcn"), cell("SJ"), cell("SA"), cell("SchiJ"), r["stamp"]))
    if not any_comp:
        T.append("| \\todo{no v1.6 runs scanned} | | | | | | | |")
    T.append("\n## Table (iv): w_t and A_t forced laws\n")
    T.append("| mu | window | class | w_t {Lap w, w} | R2 | w_t {kk:ddw, w_nn, w} | R2 | A_t SH {A, A^3, A w^2} | R2 | A_t band {A, A^3, w, A w} | R2 |")
    T.append("|---|---|---|---|---|---|---|---|---|---|---|")
    grp = defaultdict(dict)
    for r in rows:
        for tag, pref in (("lapw", "w_t | W+Lap w library | forced {Lap w, w}"), ("aniso", "w_t | W library | forced {kk:ddw, w_nn, w}"),
                          ("sh", "A_t | A library | forced {A, A^3, A w^2}"), ("band", "A_t | A library | forced {A, A^3, w, A w}")):
            if r["block"].startswith(pref):
                grp[(r["run"], r["file"], r["route"])][tag] = r
    for key, d in sorted(grp.items()):
        r = next(iter(d.values()))
        cell = lambda t, order: (fmt_coefs(json.loads(d[t]["coefs"]), order) + " | %.3f" % d[t]["r2_holdout"]) if t in d else "\\todo{} | "
        T.append("| %s | %.1f-%.1f | %s | %s | %s | %s | %s |" %
                 (("%.2f" % r["mu"]) if np.isfinite(r["mu"]) else r["file"], r["t0"], r["t1"], r["wclass"],
                  cell("lapw", ["Lap w", "w"]), cell("aniso", ["kk:ddw", "w_nn", "w"]), cell("sh", ["A", "A^3", "A w^2"]), cell("band", ["A", "A^3", "w", "A w"])))
    T.append("\n## Table (v): stationary ledger, last frame, band (rcn_balance_ledger v1.5 columns where present)\n")
    T.append("| mu | route | cores | rms theta_t | beta_A (R2) | lambda (R2) | lambda_J, m_J (R2) | chi_1/div k | lambda-2 = Lap chi + (-2b div chi k) + 4bJ at b=1 | stamp |")
    T.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted([r for r in ledger_rows if r["domain"] == "band"], key=lambda r: (r["mu"], r["route"])):
        T.append("| %.3f | %s | %d | %.1e | %.2f (%.2f) | %+.2f (%.2f) | %s | %s | %s | %s |" %
                 (r["mu"], r["route"], r["n_cores"], r["rms_theta_t"], r["beta"], r["beta_r2"], r["lam"], r["lam_r2"],
                  ("%+.2f, %+.2f (%.2f)" % (r["lam_J"], r["m_J"], r["lamJ_r2"])) if np.isfinite(r["lam_J"]) else "\\todo{v1.5 run}",
                  ("%.2f" % r["chi_ratio_b1"]) if np.isfinite(r["chi_ratio_b1"]) else "-",
                  ("%+.2f = %+.2f + %+.2f + %+.2f" % (r["lam"] - 2, r["c_lapchi_b1"], r["c_divchik_b1"], r["c_J_b1"])) if np.isfinite(r["c_lapchi_b1"]) else "-",
                  r["stem"][:40]))
    if not ledger_rows:
        T.append("| \\todo{no ledger runs scanned} | | | | | | | | | |")
    if scores:
        T.append("\n## Table (vi): prediction scores (v1.6), weak multi-step error ratio to the null, by law and window\n")
        laws = sorted({s["law"] for s in scores})
        hs = sorted({s["h"] for s in scores})
        T.append("| file | route | window | class | " + " | ".join("%s h%d" % (l, h) for l in laws for h in hs) + " |")
        T.append("|---|---|---|---|" + "---|" * (len(laws) * len(hs)))
        keys = sorted({(s["file"], s["route"], s["t0"], s["t1"], s["wclass"]) for s in scores})
        for k in keys:
            cells = []
            for l in laws:
                for h in hs:
                    v = [s for s in scores if (s["file"], s["route"], s["t0"], s["t1"], s["wclass"]) == k and s["law"] == l and s["h"] == h]
                    cells.append("%.3f" % v[0]["multi_wk"] if v else "-")
            T.append("| %s | %s | %.1f-%.1f | %s | %s |" % (k[0][:30], k[1], k[2], k[3], k[4], " | ".join(cells)))
    return "\n".join(T)


def main(args):
    roots = [Path(r.strip()) for r in args.roots.split(",") if r.strip()]
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    rows, scores, ledger_rows = [], [], []
    n_w = n_l = 0
    for root in roots:
        for rep in sorted(root.rglob("report.txt")):
            d = rep.parent
            if "__MACOSX" in str(d):
                continue
            head = rep.read_text(errors="ignore")[:40]
            if head.startswith("wsindy_theta_t"):
                r, s = scan_wsindy_dir(d); rows += r; scores += s; n_w += 1
            elif head.startswith("rcn_balance_ledger") and (d / "ledger.csv").exists():
                ledger_rows += scan_ledger_dir(d); n_l += 1
    if rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "run", k))
        with open(out / "runs.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
    if scores:
        with open(out / "scores.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(scores[0])); wr.writeheader(); wr.writerows(scores)
    if ledger_rows:
        with open(out / "ledger_last_frames.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(ledger_rows[0])); wr.writeheader(); wr.writerows(ledger_rows)
    (out / "persistence.txt").write_text("sysid_scan v1.0: %d wsindy runs, %d blocks, %d ledger runs\n\n" % (n_w, len(rows), n_l) + persistence(rows))
    (out / "tables.md").write_text("# Writeup 4 table stubs (sysid_scan v1.0)\n\n%d wsindy runs, %d ledger runs scanned from %s\n\n" %
                                   (n_w, n_l, args.roots) + tables(rows, ledger_rows, scores))
    print("wsindy runs %d (blocks %d), ledger runs %d, scores %d -> %s" % (n_w, len(rows), n_l, len(scores), out))
    return 0


class _Args:
    roots = "results"
    outdir = str(Path("results") / "sysid_scan")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in dir(_Args) if not a.startswith("_")]:
        p.add_argument("--" + k, type=type(getattr(_Args, k)), default=getattr(_Args, k))
    sys.exit(main(p.parse_args()))
