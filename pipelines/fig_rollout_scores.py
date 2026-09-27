# pipelines/fig_rollout_scores.py
"""
Figure for Section 3C: rollout score against horizon for the far-field
amplitude law (A_far), the band amplitude law (A_band, the negative) and
the isotropic wavenumber law (w_band), from the scores.json of
rcn_rollout.py (v1.1).

Input: --runs "dir,dir,..." where each dir holds scores.json and
config.json (as rcn_rollout writes them). From config.json: law, npz
(mu parsed from the file name unless --labels overrides), t0, t_fit; from
scores.json: coef, scores = [{horizon, frame, score, rel_err, rel_null,
R2_incr}, ...].

  score    = rms(rolled - measured) / rms(measured - initial); the null
             rollout (no change) scores 1
  R2_incr  = R2 of the rolled increment against the measured increment
             over the domain (spatial structure of the change)
  rel_null = rms(measured - initial) / rms(measured): how much the field
             changed over the horizon (context for the score)

Panels: (a) score vs horizon, one line per run, law by marker, mu by
colour; the fit window's end is marked on each line (horizons beyond it
are holdout); (b) R2_incr vs horizon; (c) rel_null vs horizon. Nothing
is fitted.

Run:
  python fig_rollout_scores.py --runs "results/rcn_rollout/<T3 0.85>/<stamp>,results/rcn_rollout/<T25 0.90>/<stamp>,..."
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import (fnum, mu_from_name, outdir_for, write_config, write_table,  # noqa: E402
                       save, cli, plt)

MARK = {"A_far": "o", "A_band": "s", "w_band": "D"}


def run(args):
    dirs = [Path(d.strip()) for d in args.runs.split(",") if d.strip()]
    labels = [s.strip() for s in args.labels.split(";")] if args.labels.strip() else []
    if not dirs:
        print("no run directories given")
        return 1
    out = outdir_for(__file__, args.outdir)
    write_config(out, args)
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.0))
    tab = []
    mus = []
    for i, d in enumerate(dirs):
        cfg = json.load(open(d / "config.json")) if (d / "config.json").exists() else {}
        sc = json.load(open(d / "scores.json"))
        law = cfg.get("law", "?")
        mu = mu_from_name(cfg.get("npz", d.parent.name)) if np.isfinite(mu_from_name(cfg.get("npz", ""))) else mu_from_name(d.parent.name)
        lab = labels[i] if i < len(labels) else "%s, $\\mu$ = %.2f" % (law.replace("_", " "), mu)
        h = np.array([fnum(s["horizon"]) for s in sc["scores"]])
        score = np.array([fnum(s["score"]) for s in sc["scores"]])
        r2i = np.array([fnum(s["R2_incr"]) for s in sc["scores"]])
        rn = np.array([fnum(s["rel_null"]) for s in sc["scores"]])
        if mu not in mus:
            mus.append(mu)
        color = "C%d" % (mus.index(mu) % 10)
        mk = MARK.get(law, "x")
        ax[0].plot(h, score, mk + "-", color=color, label=lab)
        ax[1].plot(h, r2i, mk + "-", color=color)
        ax[2].plot(h, rn, mk + "-", color=color)
        t_fit = fnum(cfg.get("t_fit"))
        if np.isfinite(t_fit):
            for a in ax:
                a.axvline(t_fit, color=color, ls=":", lw=0.7)
        for s in sc["scores"]:
            tab.append(dict(run=str(d), law=law, mu=mu, t0=cfg.get("t0"), t_fit=cfg.get("t_fit"),
                            coef=json.dumps(sc.get("coef")), horizon=s["horizon"], frame=s["frame"],
                            score=s["score"], rel_err=s["rel_err"], rel_null=s["rel_null"], R2_incr=s["R2_incr"]))
    ax[0].axhline(1, color="k", lw=0.6, ls="--")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("score = rms(rolled $-$ measured) / rms(change)")
    ax[0].set_title("(a) rollout score (null = 1; dotted: end of fit window)", fontsize=8)
    ax[0].legend(fontsize=6)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_ylabel("$R^2$ of the rolled increment")
    ax[1].set_title("(b) spatial structure of the change", fontsize=8)
    ax[1].set_ylim(min(-1.0, float(np.nanmin([r["R2_incr"] for r in tab])) - 0.5) if tab else -1, 1.05)
    ax[2].set_ylabel("rms(change) / rms(field)")
    ax[2].set_title("(c) how much the field changed", fontsize=8)
    for a in ax:
        a.set_xlabel("horizon (time units)")
    fig.tight_layout()
    save(fig, out)
    write_table(out, tab)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    runs = ""          # comma list of rcn_rollout output directories (each with scores.json, config.json)
    labels = ""        # optional legend labels separated by ";" (commas are allowed inside), same order as runs
    outdir = "results"


if __name__ == "__main__":
    sys.exit(cli(_Args, __doc__, run))
