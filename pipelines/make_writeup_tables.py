"""
make_writeup_tables.py  (v1.0)
------------------------------
Turn the CSVs of self_dual_sindy_sweep.py and knee_balance_sweep.py into
the LaTeX tables a writeup wants, without hand copying numbers.

    python make_writeup_tables.py --sindy-csv <path> [--balance-csv <path>]
                                  [--out-dir DIR] [--routes ...]

Tables written (one .tex file each, booktabs, \\input-able):
  table_form_r2.tex        per route x (mu, R): holdout R^2 of the linear
                           self-dual law, and the STLSQ verdict (minimal
                           form or the extra terms it added)
  table_coefficients.tex   per route x (mu, R): c raw with bootstrap
                           std, c calibrated (smoothed routes only, and
                           only where the calibration is valid), |b|/peak
  table_response.tex       per route x (mu, R): planted beta 0.9/1.0/1.1
                           -> recovered c (the instrument response)
  table_record.tex         demod: tanh estimator of record (rerun) vs the
                           record table vs the regression slope
  table_balance.tex        (if --balance-csv) per route x (mu, R):
                           beta_fit, RMS relative error nominal and
                           effective, synthetic beta = 1 reference
Raw and calibrated coefficients never share a column. Numbers are
formatted, never rounded in place: the CSV stays the record.
Python 3.8 compatible.
"""
import csv
import argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent

ROUTE_TEX = {"uhu": r"uHu $\sigma=1.571$", "riesz": r"Riesz $0.4/\pi$",
             "riesz_pi2": r"Riesz $0.4/(\pi/2)$", "demod": "demod",
             "demod_sm": r"demod $\ast G_{1.571}$", "riesz_04": "Riesz 0.4/0.4",
             "riesz_1": "Riesz 0.4/1.0"}


def _f(v, fmt="%.3f"):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "--"
    if x != x:
        return "--"
    return fmt % x


def _read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _keys(rows):
    return sorted({(float(r["mu"]), float(r["R"])) for r in rows}, key=lambda t: (t[1], t[0]))


def _routes(rows, wanted):
    seen = []
    for r in rows:
        if r["route"] not in seen and (not wanted or r["route"] in wanted):
            seen.append(r["route"])
    return seen


def _row(rows, route, mu, R):
    for r in rows:
        if r["route"] == route and abs(float(r["mu"]) - mu) < 1e-9 and abs(float(r["R"]) - R) < 1e-9:
            return r
    return None


def _verdict(r):
    if r is None:
        return "--"
    if r["form_ok_divk"] in ("True", "1", "true"):
        return "minimal"
    terms = [t.split(":")[0] for t in r["terms_divk"].split("+") if t]
    extra = [t for t in terms if t not in ("1", "w")]
    return "+" + ",".join("$%s$" % t.replace("^", "^") for t in extra) if extra else "minimal"


def table_form_r2(rows, routes, out):
    keys = _keys(rows)
    lines = [r"\begin{tabular}{cc" + "cc" * len(routes) + "}", r"\toprule",
             r"$\mu$ & $R$ & " + " & ".join(r"\multicolumn{2}{c}{%s}" % ROUTE_TEX.get(rt, rt) for rt in routes) + r" \\",
             " & & " + " & ".join(r"$R^2_{\rm hold}$ & STLSQ" for _ in routes) + r" \\", r"\midrule"]
    for mu, R in keys:
        cells = []
        for rt in routes:
            r = _row(rows, rt, mu, R)
            cells += [_f(r["r2_lin_holdout"], "%.4f") if r else "--", _verdict(r)]
        lines.append("%.3f & %.1f & " % (mu, R) + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def table_coefficients(rows, routes, out):
    keys = _keys(rows)
    lines = [r"\begin{tabular}{cc" + "ccc" * len(routes) + "}", r"\toprule",
             r"$\mu$ & $R$ & " + " & ".join(r"\multicolumn{3}{c}{%s}" % ROUTE_TEX.get(rt, rt) for rt in routes) + r" \\",
             " & & " + " & ".join(r"$c_{\rm raw}$ & $c_{\rm cal}$ & $|b|/{\rm peak}$" for _ in routes) + r" \\",
             r"\midrule"]
    for mu, R in keys:
        cells = []
        for rt in routes:
            r = _row(rows, rt, mu, R)
            if r is None:
                cells += ["--", "--", "--"]; continue
            craw = "%s (%s)" % (_f(r["c_raw"]), _f(r["c_raw_std"], "%.3f"))
            cal = _f(r["c_calibrated"]) if r.get("calibration_valid", "False") in ("True", "1", "true") else "--"
            bp = _f(100 * abs(float(r["const_over_peak"])), "%.1f") + r"\%" if "const_over_peak" in r else "--"
            cells += [craw, cal, bp]
        lines.append("%.3f & %.1f & " % (mu, R) + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def table_response(rows, routes, out):
    keys = _keys(rows)
    lines = [r"\begin{tabular}{cc" + "ccc" * len(routes) + "}", r"\toprule",
             r"$\mu$ & $R$ & " + " & ".join(r"\multicolumn{3}{c}{%s}" % ROUTE_TEX.get(rt, rt) for rt in routes) + r" \\",
             " & & " + " & ".join(r"$\beta=0.9$ & $1.0$ & $1.1$" for _ in routes) + r" \\", r"\midrule"]
    for mu, R in keys:
        cells = []
        for rt in routes:
            r = _row(rows, rt, mu, R)
            resp = {}
            if r and r.get("synth_response"):
                for item in r["synth_response"].split(","):
                    if ":" in item:
                        b, c = item.split(":"); resp[float(b)] = float(c)
            cells += [_f(resp.get(b, float("nan"))) for b in (0.9, 1.0, 1.1)]
        lines.append("%.3f & %.1f & " % (mu, R) + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def table_record(rows, out):
    keys = _keys(rows)
    lines = [r"\begin{tabular}{cccccc}", r"\toprule",
             r"$\mu$ & $R$ & record (tanh) & tanh rerun & regression $c$ & $R^2_{\rm hold}$ \\",
             r"\midrule"]
    for mu, R in keys:
        r = _row(rows, "demod", mu, R)
        if r is None:
            continue
        lines.append("%.3f & %.1f & %s & %s & %s & %s \\\\" % (
            mu, R, _f(r["record_value"]), _f(r["tanh_beta_record_rerun"]),
            _f(r["c_raw"]), _f(r["r2_lin_holdout"], "%.4f")))
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def table_balance(rows, routes, out):
    data = [r for r in rows if r["synthetic"] in ("False", "0", "false")]
    syn = [r for r in rows if r["synthetic"] in ("True", "1", "true")]
    keys = _keys(data)
    lines = [r"\begin{tabular}{cc" + "cccc" * len(routes) + "}", r"\toprule",
             r"$\mu$ & $R$ & " + " & ".join(r"\multicolumn{4}{c}{%s}" % ROUTE_TEX.get(rt, rt) for rt in routes) + r" \\",
             " & & " + " & ".join(r"$\beta_{\rm fit}$ & rms nom. & rms eff. & synth. nom." for _ in routes) + r" \\",
             r"\midrule"]
    for mu, R in keys:
        cells = []
        for rt in routes:
            r = _row(data, rt, mu, R); s = _row(syn, rt, mu, R)
            cells += [_f(r["beta_fit"]) if r else "--",
                      _f(r["rms_rel_global_nominal"]) if r else "--",
                      _f(r["rms_rel_global_effective"]) if r else "--",
                      _f(s["rms_rel_global_nominal"]) if s else "--"]
        lines.append("%.3f & %.1f & " % (mu, R) + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def main(sindy_csv, balance_csv=None, out_dir=None, routes=None):
    rows = _read(sindy_csv)
    out_dir = Path(out_dir) if out_dir else Path(sindy_csv).parent / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    rts = _routes(rows, routes)
    table_form_r2(rows, rts, out_dir / "table_form_r2.tex")
    table_coefficients(rows, rts, out_dir / "table_coefficients.tex")
    table_response(rows, rts, out_dir / "table_response.tex")
    if any(r["route"] == "demod" for r in rows):
        table_record(rows, out_dir / "table_record.tex")
    if balance_csv:
        brows = _read(balance_csv)
        table_balance(brows, _routes(brows, routes), out_dir / "table_balance.tex")
    print("tables ->", out_dir)
    return out_dir


class _Args:
    sindy_csv = "/Users/edwardmcdugald/patterns/pipelines/results/self_dual_sindy_sweep/sdsindy_R0p5-1_mu0.325-0.475_routes-uhu-riesz-riesz_pi2-demod-demod_sm_dom-profile/self_dual_sindy_sweep.csv"
    balance_csv = ""
    out_dir = ""
    routes = []


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sindy-csv", default=_Args.sindy_csv, required=not _Args.sindy_csv)
    p.add_argument("--balance-csv", default=_Args.balance_csv)
    p.add_argument("--out-dir", default=_Args.out_dir)
    p.add_argument("--routes", nargs="*", default=_Args.routes)
    a = p.parse_args()
    main(a.sindy_csv, a.balance_csv or None, a.out_dir or None, a.routes or None)
