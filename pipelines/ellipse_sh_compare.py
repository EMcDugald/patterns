"""
ellipse_sh_compare.py

Compare the RCN zipper construction (results/ellipse_sweep/<tag>_construction.npz)
with Swift-Hohenberg (results/sh_ellipse_standin/<tag>.npz).

Defect detector (applied identically to both fields): the axis row is split
into sign-constant runs; each run is classified by the curvature across the
axis, K = mean(-u_yy/u) over its middle half, which is about 0 for rolls
crossing the axis (even phase) and about q^2 for a roll lying along it (odd
phase about n pi). Each Dirichlet run beyond the central one is one
dislocation: junction at its inner end, convex tip at its outer end. Tried
first and rejected: a structure-tensor orientation test (fails above about
60 deg, where crossing rolls also have k mostly across the axis just off the
core), an amplitude-plateau test and a run-flatness test (both confounded by
SH's amplitude dips and bulges and by grid sampling of short runs).

Outputs: pipelines/results/ellipse_sh_compare/

Changelog
---------
v1 (2026-10-08). First version.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_NAME = Path(__file__).stem
sys.path.insert(0, str(SCRIPT_DIR))
import ellipse_zipper_assemble as ez  # noqa: E402


def axis_segments(u, x, y, ell, K_min=0.25, amp_frac=0.15):
    """Dirichlet segments along y = 0 for a field u(y, x) (mirror symmetric in y).
    Split the axis row into sign-constant runs (between zero crossings). Between
    crossings the core phase advances by pi in a knee or a zipper alike, so run
    length cannot tell them apart. The curvature across the axis can:
      rolls crossing the axis (knee, Neumann): theta is even in y,
          theta ~ eps x + q^2 y^2 / 2, so -u_yy/u = q^2 tan(theta): antisymmetric
          about the crest centre, averaging to about 0 over a run;
      a roll lying along the axis (Dirichlet): theta is odd about n pi,
          theta ~ theta_y y, so -u_yy/u ~ theta_y^2 ~ q^2 > 0 along the whole run.
    K = mean of -u_yy/u over the middle half of the run (u_yy from the mirror
    symmetry: 2 (u(dy) - u(0)) / dy^2). A run is Dirichlet when K > K_min.
    Returns (segments [(x0, x1, K)], x, |u| / amplitude, runs [(x0, x1, sign, K)])."""
    j0 = int(np.argmin(np.abs(y)))
    dy = y[1] - y[0]
    row = u[j0].copy()
    uyy = 2.0 * (0.5 * (u[j0 + 1] + u[j0 - 1]) - row) / dy ** 2
    inside = np.abs(x) < ell.A
    amp = np.percentile(np.abs(row[inside]), 90)
    sgn = np.sign(row)
    sgn[np.abs(row) < amp_frac * amp] = 0
    runs = []
    i = 0
    n = x.size
    while i < n:
        if sgn[i] == 0 or not inside[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and inside[j + 1] and sgn[j + 1] in (0, sgn[i]):
            j += 1
        k = j
        while k > i and sgn[k] == 0:
            k -= 1
        L = k - i + 1
        a0, a1 = i + L // 4, i + max(L // 4 + 1, (3 * L) // 4)
        K = float(np.mean(-uyy[a0:a1] / row[a0:a1])) if L >= 2 else 0.0
        runs.append((float(x[i]), float(x[k]), int(sgn[i]), K))
        i = k + 1
    segs = [(a0, a1, K) for (a0, a1, sg, K) in runs if K > K_min]
    return segs, x, np.abs(row) / amp, runs


def _running_max(v, x, half_width):
    """Max of v over a window of +/- half_width (local amplitude envelope)."""
    from scipy.ndimage import maximum_filter1d
    n = max(1, int(round(2 * half_width / (x[1] - x[0]))))
    return maximum_filter1d(v, size=n, mode="nearest")


def summarize(segs, A):
    """Central segment = the one containing x = 0; dislocations per side =
    Dirichlet segments strictly on that side (inside the ellipse)."""
    central = [s for s in segs if s[0] <= 0 <= s[1]]
    right = [s for s in segs if s[0] > 0 and s[1] < A]
    left = [s for s in segs if s[1] < 0 and s[0] > -A]
    return central, left, right


def main(args):
    out = Path(args.out_dir) if args.out_dir else SCRIPT_DIR / "results" / SCRIPT_NAME
    out.mkdir(parents=True, exist_ok=True)
    cdir = Path(args.construction_dir) if args.construction_dir else SCRIPT_DIR / "results" / "ellipse_sweep"
    sdir = Path(args.sh_dir) if args.sh_dir else SCRIPT_DIR / "results" / "sh_ellipse_standin"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = {}
    lines = []

    def say(s):
        print(s, flush=True)
        lines.append(s)

    for tag in args.tags:
        cp, sp_ = cdir / ("%s_construction.npz" % tag), sdir / ("%s.npz" % tag)
        if not (cp.exists() and sp_.exists()):
            say("!! LOUD FLAG: missing %s or %s; skipping %s" % (cp, sp_, tag))
            continue
        C = np.load(cp); S = np.load(sp_)
        A, B = float(C["A"]), float(C["B"])
        # construction on the full ellipse
        xq, yq = C["x"], C["y"]
        xf = np.concatenate([-xq[:0:-1], xq]); yf = np.concatenate([-yq[:0:-1], yq])
        Uc = np.nan_to_num(ez.mirror_full(np.cos(C["theta_fine"])), nan=0.0)
        ell = ez.Ellipse(A, B)
        segs_c, xa_c, m_c, runs_c = axis_segments(Uc, xf, yf, ell, K_min=args.K_min)
        cen_c, left_c, right_c = summarize(segs_c, A)
        # SH frames
        xx, yy, tt, uu = S["xx"], S["yy"], S["tt"], S["uu"]
        rows = []
        for it in range(uu.shape[-1]):
            segs_s, xa_s, m_s, runs_s = axis_segments(uu[..., it], xx, yy, ell, K_min=args.K_min)
            cen_s, left_s, right_s = summarize(segs_s, A)
            rows.append(dict(t=float(tt[it]), n_left=len(left_s), n_right=len(right_s),
                             central=cen_s, right=right_s, m=m_s, runs=runs_s))
        say("%s (A=%.0f, B=%.0f): construction %d|%d dislocations (left|right), central roll %s"
            % (tag, A, B, len(left_c), len(right_c), ["%.1f..%.1f" % s[:2] for s in cen_c]))
        for r_ in rows:
            say("   SH t=%6.0f: %d|%d dislocations, central %s" %
                (r_["t"], r_["n_left"], r_["n_right"], ["%.1f..%.1f" % s[:2] for s in r_["central"]]))
        last = rows[-1]
        say("   right-side Dirichlet segments (inner end = junction, outer end = tip, K = mean -u_yy/u):")
        say("     construction: %s" % ["%.1f..%.1f (K=%.2f)" % s for s in right_c])
        say("     SH (t=%.0f):   %s" % (last["t"], ["%.1f..%.1f (K=%.2f)" % s for s in last["right"]]))
        say("   all right-side runs, SH final (x0..x1 sign K):")
        say("     " + "  ".join("%.0f..%.0f%s%.2f" % (r0, r1, "+" if sg > 0 else "-", F)
                                for (r0, r1, sg, F) in last["runs"] if r0 >= 0))
        say("   all right-side runs, construction:")
        say("     " + "  ".join("%.0f..%.0f%s%.2f" % (r0, r1, "+" if sg > 0 else "-", F)
                                for (r0, r1, sg, F) in runs_c if r0 >= 0))
        report[tag] = dict(A=A, B=B, construction=dict(left=len(left_c), right=len(right_c), right_segs=right_c,
                                                       central=cen_c),
                           sh=[dict(t=r_["t"], left=r_["n_left"], right=r_["n_right"], right_segs=r_["right"],
                                    central=r_["central"]) for r_ in rows])

        # figure: SH final, construction, axis orientation measure
        fig = plt.figure(figsize=(11, 9))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.55])
        xr = max(0.75 * A, float(C["X_switch"]) + 30)
        yr = min(B, 0.22 * xr)
        for k, (U, xg, yg, ttl, segs) in enumerate([
                (uu[..., -1], xx, yy, "Swift-Hohenberg (stand-in runner), t = %.0f" % tt[-1], last["right"]),
                (Uc, xf, yf, "RCN zipper construction (fine)", right_c)]):
            ax = fig.add_subplot(gs[k])
            ax.imshow(U, extent=[xg[0], xg[-1], yg[0], yg[-1]], origin="lower", cmap="gray",
                      interpolation="bilinear")
            for (a0, a1, _F) in segs:
                ax.plot([a0, -a0], [0, 0], "^", ms=4, mfc="none", mec="#2a78d6", mew=1.0)
                ax.plot([a1, -a1], [0, 0], "o", ms=4, mfc="none", mec="#eb6834", mew=1.0)
            ax.set_xlim(-xr, xr); ax.set_ylim(-yr, yr); ax.set_aspect("equal")
            ax.set_title(ttl + "  (markers: detected junctions ^ and tips o)", fontsize=9)
        ax = fig.add_subplot(gs[2])
        ax.plot(xa_c, m_c, color="#2a78d6", lw=1.0, label="construction")
        ax.plot(xx, last["m"], color="#eb6834", lw=1.0, label="SH t = %.0f" % tt[-1])
        ax.axhline(0.0, color="#8a8984", lw=0.8)
        ax.axvline(float(C["X_switch"]), color="#8a8984", lw=0.8, ls="--")
        ax.axvline(-float(C["X_switch"]), color="#8a8984", lw=0.8, ls="--")
        ax.set_xlim(-xr, xr)
        ax.set_xlabel("X along the major axis")
        ax.set_ylabel("|u(X,0)| / amplitude")
        ax.set_title("Pattern along the axis: flat plateaus = roll along the axis (Dirichlet), cosine crests = "
                     "rolls crossing; dashed: RCN switch", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / ("%s_sh_vs_rcn.png" % tag), dpi=170)
        plt.close(fig)

        # SH time series of the full ellipse
        nt = uu.shape[-1]
        fig, axs = plt.subplots(nt, 1, figsize=(9, 2.2 * nt * (B / A) * 2.2 + 0.5))
        for k in range(nt):
            axs[k].imshow(uu[..., k], extent=[xx[0], xx[-1], yy[0], yy[-1]], origin="lower", cmap="gray",
                          interpolation="bilinear")
            axs[k].set_xlim(-1.03 * A, 1.03 * A); axs[k].set_ylim(-1.03 * B, 1.03 * B)
            axs[k].set_aspect("equal"); axs[k].set_xticks([]); axs[k].set_yticks([])
            axs[k].set_title("SH t = %.0f: %d|%d dislocations" % (tt[k], rows[k]["n_left"], rows[k]["n_right"]),
                             fontsize=8)
        fig.tight_layout()
        fig.savefig(out / ("%s_sh_time_series.png" % tag), dpi=150)
        plt.close(fig)

    (out / "compare_report.json").write_text(json.dumps(report, indent=2, default=float))
    (out / "compare_log.txt").write_text("\n".join(lines) + "\n")


class _Args(object):
    tags = ["ab3_B70", "ab2_B135", "ab15_B270"]
    K_min = 0.25                # curvature-across-the-axis threshold (crossing rolls ~0, axis roll ~q^2)
    construction_dir = None
    sh_dir = None
    out_dir = None


def _parse_cli():
    d = _Args()
    p = argparse.ArgumentParser(description="SH against RCN construction on ellipses")
    p.add_argument("--tags", nargs="+", default=d.tags)
    p.add_argument("--K_min", type=float, default=d.K_min)
    for k in ["construction_dir", "sh_dir", "out_dir"]:
        p.add_argument("--" + k, default=getattr(d, k))
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_cli() if len(sys.argv) > 1 else _Args())
