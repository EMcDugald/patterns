# pipelines/fig_winding_time.py
"""
Figure for Section 3C (the events): the charge onset and the winding
time against mu at R = 1/2 and R = 1, from the table.csv written by
winding_time_vs_mu.py (v1.1).

Input: --tables "path:R,path:R,..." (one table.csv per stress parameter;
R is required because the table does not carry it). Columns used: mu, dt,
t_first, t_last, t_charge, t_circ_on, t_wind_mean, t_wind_min, held,
t_first_core, t_last_core, file. Files that never wind (t_wind nan) are
drawn as open markers at the run's end t_last with an upward arrow ("no
winding within the run"). Error bars are one frame (dt), the resolution
of every onset. When the per-core columns exist, the staggering
t_first_core .. t_last_core is drawn as a vertical bar.

--which picks t_wind_mean (default, the |circ| mean over the cores) or
t_wind_min (every core wound, the strict reading). Nothing is fitted.

R per file: the R after the colon in --tables is the default for that
table; --R_tags "Rp5=0.5,_R1=1" overrides it by substring of the file
stem (first match wins), which handles a table that mixes the R = 1/2
and R = 1 sweeps. The same file stem listed in several tables is kept
once, from the last table listed (so list the fine-cadence table last).
Fine-cadence files (dt < 0.1) are drawn with a different marker. The
file -> R assignment is printed and written to data.csv: check it.

Section 3C todo (Ed, 23 Sep): the 0.90 and 0.95 windings coincide at
coarse cadence (dt 0.2) and may differ at fine cadence (dt 0.025); run
winding_time_vs_mu on the fine files and pass their table here.

Run:
  python fig_winding_time.py --tables "results/winding_time_vs_mu/<R0.5 stamp>/table.csv:0.5,results/winding_time_vs_mu/<R1 stamp>/table.csv:1"
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import read_csv, fnum, outdir_for, write_config, write_table, save, cli, plt  # noqa: E402


def run(args):
    specs = []
    for spec in [s.strip() for s in args.tables.split(",") if s.strip()]:
        path, _, R = spec.rpartition(":")
        if not path:                      # no ":R" given
            path, R = R, "nan"
        specs.append((fnum(R), Path(path)))
    if not specs:
        print("no tables given")
        return 1
    out = outdir_for(__file__, args.outdir)
    write_config(out, args)
    tags = []
    for item in [t for t in args.R_tags.split(",") if t.strip()]:
        k, _, v = item.partition("=")
        tags.append((k.strip(), fnum(v)))
    # collect rows from every table; a file stem seen again replaces the earlier one
    byfile = {}
    for R_default, path in specs:
        for r in read_csv(path):
            R = R_default
            for k, v in tags:
                if k in r["file"]:
                    R = v
                    break
            byfile[r["file"]] = (R, path, r)
    tab = []
    for f, (R, path, r) in sorted(byfile.items(), key=lambda kv: (kv[1][0], fnum(kv[1][2]["mu"]))):
        # t_charge is the first frame at which the charge exceeds its initial value; when the
        # first frame already has charge (q_initial > 0) the onset is before the file starts and
        # a later "exceedance" is a fluctuation, not the onset: drawn as absent, noted in data.csv
        q0 = fnum(r.get("q_initial", "nan"))
        t_charge = fnum(r["t_charge"])
        charge_note = ""
        if np.isfinite(q0) and q0 > 0:
            charge_note = "charge present at first frame (q_initial %d); t_charge %s not an onset" % (int(q0), t_charge)
            t_charge = float("nan")
        tab.append(dict(R=R, table=str(path), file=f, mu=fnum(r["mu"]), dt=fnum(r["dt"]),
                        t_charge=t_charge, charge_note=charge_note, t_wind=fnum(r[args.which]), which=args.which,
                        t_wind_min=fnum(r.get("t_wind_min", "nan")), held=r.get("held"), t_last=fnum(r["t_last"]),
                        t_first_core=fnum(r.get("t_first_core", "nan")), t_last_core=fnum(r.get("t_last_core", "nan"))))
        print("  R %-5s mu %.3f dt %.3f  t_charge %s  t_wind %s  %s  %s" % (R, tab[-1]["mu"], tab[-1]["dt"], tab[-1]["t_charge"], tab[-1]["t_wind"], f, charge_note))
    Rs = sorted({t["R"] for t in tab}, key=lambda v: (not np.isfinite(v), v))
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for i, R in enumerate(Rs):
        color = "C%d" % i
        Rlab = ("R = %g" % R) if np.isfinite(R) else "R unknown"
        T = [t for t in tab if t["R"] == R or (not np.isfinite(R) and not np.isfinite(t["R"]))]
        T.sort(key=lambda t: t["mu"])
        for fine, mk, suffix in ((False, "o", ""), (True, "D", ", fine cadence")):
            U = [t for t in T if (t["dt"] < 0.1) == fine]
            if not U:
                continue
            mu = np.array([t["mu"] for t in U]); dt = np.array([t["dt"] for t in U])
            tw = np.array([t["t_wind"] for t in U]); tc = np.array([t["t_charge"] for t in U])
            tl = np.array([t["t_last"] for t in U])
            tfc = np.array([t["t_first_core"] for t in U]); tlc = np.array([t["t_last_core"] for t in U])
            ok = np.isfinite(tw)
            lo = np.minimum(dt[ok], 0.5 * tw[ok])          # a log axis cannot show an error bar through zero
            ax.errorbar(mu[ok], tw[ok], yerr=[lo, dt[ok]], fmt=mk + ("-" if not fine else ""), color=color, ms=5,
                        capsize=2, label=Rlab + ": winding" + suffix)
            okc = np.isfinite(tc)
            loc_ = np.minimum(dt[okc], 0.5 * tc[okc])
            ax.errorbar(mu[okc], tc[okc], yerr=[loc_, dt[okc]], fmt=("s--" if not fine else "s"), color=color, ms=4,
                        capsize=2, mfc="none", label=Rlab + ": charge onset" + suffix)
            stag = np.isfinite(tfc) & np.isfinite(tlc) & (tlc > tfc)
            for m, a, b in zip(mu[stag], tfc[stag], tlc[stag]):
                ax.plot([m, m], [a, b], "-", color=color, lw=3, alpha=0.35)
            for m, t_end in zip(mu[~ok], tl[~ok]):
                ax.annotate("", xy=(m, t_end * 1.25), xytext=(m, t_end), arrowprops=dict(arrowstyle="->", color=color))
                ax.plot(m, t_end, mk, mfc="none", color=color, ms=6)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 300)
    ax.set_xlabel("$\\mu = \\sin\\alpha$")
    ax.set_ylabel("time")
    ax.set_title("charge onset and winding time (open marker + arrow: no winding within the run; bar: first .. last core)",
                 fontsize=7)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.tight_layout()
    save(fig, out)
    write_table(out, tab)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    tables = ""             # "path:R,path:R" table.csv files of winding_time_vs_mu; R is the default per table
    R_tags = "Rp5=0.5,_R1=1"  # substring of the file stem -> R, overrides the table default
    which = "t_wind_mean"   # or t_wind_min (every core wound)
    outdir = "results"


if __name__ == "__main__":
    sys.exit(cli(_Args, __doc__, run))
