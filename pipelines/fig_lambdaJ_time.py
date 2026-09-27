# pipelines/fig_lambdaJ_time.py
"""
Figure for Section 3C (the stationary balance): lambda_J(t) on the band
of the three nucleation runs, with the charge onset and the winding
marked, from the ledger.csv of rcn_balance_ledger.py (v1.5).

Input: one ledger.csv per file, passed as --ledgers "path:mu,path:mu"
(mu is parsed from the path when omitted, e.g. ..._mu0.85_...). Columns
used: route, t, frame, domain, n_cores, lam, lam_r2, lam_J, m_J, lamJ_r2
(the v1.5 columns; runs before v1.5 have no lam_J and are drawn as lambda
only). Rows with domain == --domain and route == --route are used.

Panels: (a) lambda_J(t) per mu, with lambda (the fit without J) dashed,
the RCN knee value drawn as a reference line when --lam_ref is given;
(b) the holdout R2 of the two fits; (c) the number of detected cores.
Event marks: charge onset (dotted) and the winding interval first .. last
core (shaded), from figcommon.T_WIND / T_CHARGE unless --events overrides
("mu:t_charge:t_wind_first:t_wind_last;..."). Nothing is fitted.

Run:
  python fig_lambdaJ_time.py --ledgers "results/rcn_balance_ledger/<mu0.85 T25>/<stamp>/ledger.csv,results/.../ledger.csv"
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import (read_csv, fnum, mu_from_name, outdir_for, write_config,  # noqa: E402
                       write_table, save, cli, plt, T_WIND, T_CHARGE)


def parse_events(s):
    ev = {}
    for item in [p for p in s.split(";") if p.strip()]:
        parts = [fnum(v) for v in item.split(":")]
        ev[round(parts[0], 3)] = tuple(parts[1:4])
    return ev


def run(args):
    items = []
    for spec in [s.strip() for s in args.ledgers.split(",") if s.strip()]:
        path, _, mu = spec.partition(":")
        path = Path(path)
        mu = fnum(mu) if mu else mu_from_name(path.parent.parent.name + "_" + path.parent.name + "_" + path.name)
        items.append((mu, path))
    items.sort(key=lambda it: it[0])
    if not items:
        print("no ledgers given")
        return 1
    events = parse_events(args.events)
    out = outdir_for(__file__, args.outdir)
    write_config(out, args)

    fig, ax = plt.subplots(3, 1, figsize=(6.4, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4, 1]))
    tab = []
    for mu, path in items:
        rows = [r for r in read_csv(path) if r.get("domain") == args.domain and r.get("route") == args.route]
        if not rows:
            print("no rows with domain %s route %s in %s" % (args.domain, args.route, path))
            continue
        t = np.array([fnum(r["t"]) for r in rows])
        lam = np.array([fnum(r.get("lam")) for r in rows])
        lam_r2 = np.array([fnum(r.get("lam_r2")) for r in rows])
        lamJ = np.array([fnum(r.get("lam_J")) for r in rows])
        lamJ_r2 = np.array([fnum(r.get("lamJ_r2")) for r in rows])
        mJ = np.array([fnum(r.get("m_J")) for r in rows])
        nc = np.array([fnum(r.get("n_cores")) for r in rows])
        lab = "$\\mu = %.2f$" % mu
        line, = ax[0].plot(t, lamJ, "-", label=lab + " $\\lambda_J$")
        ax[0].plot(t, lam, "--", color=line.get_color(), lw=0.8, label=lab + " $\\lambda$ (no $J$)")
        ax[1].plot(t, lamJ_r2, "-", color=line.get_color())
        ax[1].plot(t, lam_r2, "--", color=line.get_color(), lw=0.8)
        ax[2].plot(t, nc, "-", color=line.get_color())
        ev = events.get(round(mu, 3))
        if ev is None:
            tw = T_WIND.get(round(mu, 3))
            ev = (T_CHARGE.get(round(mu, 3), np.nan),) + (tw if tw else (np.nan, np.nan))
        for a in ax:
            if np.isfinite(ev[0]):
                a.axvline(ev[0], color=line.get_color(), ls=":", lw=0.8)
            if np.isfinite(ev[1]) and np.isfinite(ev[2]):
                a.axvspan(ev[1], max(ev[2], ev[1] + 0.05), color=line.get_color(), alpha=0.12, lw=0)
        for r, a1, a2, a3, a4, a5 in zip(rows, lam, lam_r2, lamJ, lamJ_r2, mJ):
            tab.append(dict(mu=mu, ledger=str(path), t=fnum(r["t"]), n_cores=fnum(r.get("n_cores")),
                            lam=a1, lam_r2=a2, lam_J=a3, lamJ_r2=a4, m_J=a5,
                            t_charge=ev[0], t_wind_first=ev[1], t_wind_last=ev[2]))
    if np.isfinite(args.lam_ref):
        ax[0].axhline(args.lam_ref, color="k", lw=0.8, ls="-.", label="reference $\\lambda$ = %.2f" % args.lam_ref)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_ylabel("$\\lambda_J$ in $\\nabla^2\\nabla\\cdot k = \\lambda_J S + m J$")
    ax[0].legend(fontsize=7, ncol=2)
    ax[0].set_title("stationary $S$-coefficient on the band by frame; dotted: charge onset; shaded: winding (first .. last core)", fontsize=8)
    ax[1].set_ylabel("$R^2$")
    ax[1].set_ylim(0, 1)
    ax[2].set_ylabel("cores")
    ax[2].set_xlabel("$t$")
    fig.tight_layout()
    save(fig, out)
    write_table(out, tab)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    ledgers = ""            # "path[:mu],path[:mu],..." ledger.csv files of rcn_balance_ledger v1.5
    outdir = "results"
    domain = "band"
    route = "estimator"     # the ledger's route label ("estimator" on data; "exact" on the knee / pn modes)
    events = ""             # "mu:t_charge:t_wind_first:t_wind_last;..." overrides figcommon.T_WIND
    lam_ref = float("nan")  # e.g. the RCN knee's stationary value from the knee mode, if wanted


if __name__ == "__main__":
    sys.exit(cli(_Args, __doc__, run))
