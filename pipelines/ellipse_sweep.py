"""
ellipse_sweep.py

Aspect-ratio sweep of the RCN zipper construction (ellipse_zipper_assemble.py),
coarse (thick strips) against fine (continuum), plus an ellipse-level energy
check of the knee/zipper switch angle, plus (when the SH stand-in runs exist)
a side-by-side with Swift-Hohenberg.

For each geometry (A/B, B):
  1. fine and coarse (n_strips) constructions at one wall phase, with total
     RCN energy and the max |cos theta| difference between them;
  2. total RCN energy of the assembled ellipse against the switch angle
     (66, RCN, 68, 71, 75, 80, 90 = no zipper), on a coarser grid;
  3. if results/sh_ellipse_standin/<tag>.npz exists: SH final frame next to the
     fine construction, with the predicted defect positions overlaid, and the
     SH pattern along the major axis.

Outputs: pipelines/results/ellipse_sweep/

Changelog
---------
v1 (2026-10-08). First version. Summary json is merged across runs (a second
    run with --only originally overwrote the first run's entries).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_NAME = Path(__file__).stem
sys.path.insert(0, str(SCRIPT_DIR))
import ellipse_zipper_assemble as ez  # noqa: E402

GEOMS = [  # tag, A/B, B (B chosen for 6 to 7 dislocations per side under the RCN switch)
    ("ab15_B270", 1.5, 270.0),
    ("ab2_B135", 2.0, 135.0),
    ("ab3_B70", 3.0, 70.0),
]


def total_energy(th, inside, dx):
    e = ez.energy_density(th, dx)
    wgt = np.ones_like(e); wgt[0, :] *= 0.5; wgt[:, 0] *= 0.5
    ok = inside & np.isfinite(e)
    return 4.0 * dx ** 2 * float(np.sum((e * wgt)[ok]))


def build(E, Z, far, switch, s_center, xq, yq, n_strips):
    ez.STRIPS["n"] = n_strips if n_strips else None
    ez.STRIPS["switch"] = switch
    lay = ez.layout(E, Z, switch, s_center, far=far)
    A_ = ez.assemble(E, Z, lay, xq, yq)
    ez.STRIPS["n"] = None
    return lay, A_


def main(args):
    out = Path(args.out_dir) if args.out_dir else SCRIPT_DIR / "results" / SCRIPT_NAME
    out.mkdir(parents=True, exist_ok=True)
    Z = ez.ZipperLookup(SCRIPT_DIR / "results" / "zipper_table")
    rcn = Z.rcn_switch()
    a_c = float(Z.a_star(90.0))
    lines = []

    def say(s):
        print(s, flush=True)
        lines.append(s)

    sp = out / "sweep_summary.json"
    summary = json.loads(sp.read_text()) if sp.exists() else {}   # merge, never overwrite other tags
    for tag, r, B in GEOMS:
        if args.only and tag not in args.only:
            continue
        t0 = time.time()
        A = r * B
        E = ez.Ellipse(A, B)
        E.selfdual_weights(M=args.M, n_iter=40)
        far = ez.axis_far_field(E)
        s_center = args.s_center_frac * a_c * np.pi
        xq = np.arange(0.0, A * 1.01 + args.dx, args.dx)
        yq = np.arange(0.0, B * 1.01 + args.dx, args.dx)
        lay_f, A_f = build(E, Z, far, rcn, s_center, xq, yq, 0)
        lay_c, A_c = build(E, Z, far, rcn, s_center, xq, yq, args.n_strips)
        th_sd = A_f["theta_sd"]
        E_f = total_energy(A_f["theta"], A_f["inside"], args.dx)
        E_c = total_energy(A_c["theta"], A_c["inside"], args.dx)
        E_k = total_energy(th_sd, A_f["inside"], args.dx)
        diff = np.nanmax(np.abs(np.cos(A_f["theta"]) - np.cos(A_c["theta"])))
        say("%s: A=%.1f B=%.1f  X_switch=%.1f  dislocations/side fine %d, coarse %d  |  energy: knee everywhere %.1f, "
            "fine %.1f, coarse(%d strips) %.1f  |  max|cos diff| coarse vs fine %.2f  [%.0fs]"
            % (tag, A, B, lay_f["Xsw"], len(lay_f["tips"]) - 1, len(lay_c["tips"]) - 1, E_k, E_f,
               args.n_strips, E_c, diff, time.time() - t0))
        say("   fine tips %s" % ["%.1f" % v for v in lay_f["tips"]])
        say("   fine junctions %s" % ["%.1f" % v for v in lay_f["starts"]])
        say("   coarse tips %s" % ["%.1f" % v for v in lay_c["tips"]])
        np.savez_compressed(out / ("%s_construction.npz" % tag), x=xq, y=yq, theta_fine=A_f["theta"],
                            theta_coarse=A_c["theta"], theta_sd=th_sd, inside=A_f["inside"],
                            tips=np.array(lay_f["tips"]), starts=np.array(lay_f["starts"]),
                            tips_coarse=np.array(lay_c["tips"]), starts_coarse=np.array(lay_c["starts"]),
                            X_switch=lay_f["Xsw"], A=A, B=B, Theta0=lay_f["Theta0"])
        figs_geometry(tag, r, E, lay_f, lay_c, A_f, A_c, xq, yq, out, args)

        # ellipse-level switch-angle check, on a coarser grid
        dx2 = args.dx_energy
        xq2 = np.arange(0.0, A * 1.01 + dx2, dx2)
        yq2 = np.arange(0.0, B * 1.01 + dx2, dx2)
        sw_list = sorted(set([Z.amin, round(rcn, 2), 68.0, 71.0, 75.0, 80.0, 90.0]))
        Es = []
        for sw in sw_list:
            lay2, A2 = build(E, Z, far, sw, s_center, xq2, yq2, 0)
            Es.append(total_energy(A2["theta"], A2["inside"], dx2))
            say("   switch %.2f deg: dislocations/side %d, total energy %.1f"
                % (sw, max(len(lay2["tips"]) - 1, 0), Es[-1]))
        summary[tag] = dict(A=A, B=B, ratio=r, X_switch=lay_f["Xsw"], n_disl_fine=len(lay_f["tips"]) - 1,
                            n_disl_coarse=len(lay_c["tips"]) - 1, E_knee=E_k, E_fine=E_f, E_coarse=E_c,
                            maxdiff=float(diff), switch_list=sw_list, E_switch=Es,
                            tips=lay_f["tips"], starts=lay_f["starts"])
        (out / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    (out / ("sweep_log%s.txt" % ("_" + "_".join(args.only) if args.only else ""))).write_text("\n".join(lines) + "\n")
    if summary:
        fig_switch(summary, rcn, out)


def figs_geometry(tag, r, E, lay_f, lay_c, A_f, A_c, xq, yq, out, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xf = np.concatenate([-xq[:0:-1], xq]); yf = np.concatenate([-yq[:0:-1], yq])
    ext = [xf[0], xf[-1], yf[0], yf[-1]]

    def marks(ax, lay, ms=3.5):
        tx, sx = np.array(lay["tips"]), np.array(lay["starts"])
        ax.plot(np.r_[tx, -tx], np.zeros(2 * tx.size), "o", ms=ms, mfc="none", mec="#eb6834", mew=1.0)
        if sx.size:
            ax.plot(np.r_[sx, -sx], np.zeros(2 * sx.size), "^", ms=ms, mfc="none", mec="#2a78d6", mew=1.0)

    # full ellipse, fine, gray like the SH figures
    fig, ax = plt.subplots(figsize=(11, 11 / r * 0.95 + 0.6))
    U = ez.mirror_full(np.cos(A_f["theta"]))
    ax.imshow(np.nan_to_num(U, nan=0.0), extent=ext, origin="lower", cmap="gray", vmin=-1, vmax=1,
              interpolation="bilinear")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("RCN zipper construction, fine, A/B = %.1f, B = %.0f (%d dislocations per side)"
                 % (r, E.B, len(lay_f["tips"]) - 1), fontsize=9)
    fig.tight_layout(); fig.savefig(out / ("%s_fine_full.png" % tag), dpi=200); plt.close(fig)

    # coarse vs fine zoom on the right half of the zipper zone
    xr = lay_f["Xsw"] + 25
    fig, axs = plt.subplots(3, 1, figsize=(10, 9.5))
    for ax, th, lay, ttl in [(axs[0], A_c["theta"], lay_c, "coarse: %d strips by normal angle" % args.n_strips),
                             (axs[1], A_f["theta"], lay_f, "fine: continuum (arbitrarily thin strips)")]:
        U = ez.mirror_full(np.cos(th))
        ax.imshow(np.nan_to_num(U, nan=0.0), extent=ext, origin="lower", cmap="gray", vmin=-1, vmax=1,
                  interpolation="bilinear")
        marks(ax, lay)
        ax.set_xlim(-5, xr); ax.set_ylim(-0.12 * xr, 0.12 * xr); ax.set_aspect("equal")
        ax.axvline(lay["Xsw"], color="#8a8984", lw=0.8, ls="--")
        ax.set_title(ttl, fontsize=9)
    D = ez.mirror_full(np.abs(np.cos(A_f["theta"]) - np.cos(A_c["theta"])))
    im = axs[2].imshow(np.nan_to_num(D, nan=0.0), extent=ext, origin="lower", cmap="magma", vmin=0, vmax=1)
    axs[2].set_xlim(-5, xr); axs[2].set_ylim(-0.12 * xr, 0.12 * xr); axs[2].set_aspect("equal")
    axs[2].set_title("|cos theta (fine) - cos theta (coarse)|", fontsize=9)
    fig.colorbar(im, ax=axs[2], shrink=0.6)
    fig.suptitle("A/B = %.1f: right half of the zipper zone (dashed: RCN switch); o tips, ^ junctions" % r,
                 fontsize=9)
    fig.tight_layout(); fig.savefig(out / ("%s_coarse_vs_fine.png" % tag), dpi=170); plt.close(fig)


def fig_switch(summary, rcn, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for c, (tag, S) in zip(cols, summary.items()):
        e = np.array(S["E_switch"]) / S["E_switch"][-1]
        ax.plot(S["switch_list"], e, "-o", color=c, ms=4, lw=1.5, label="A/B = %.1f" % S["ratio"])
    ax.axvline(rcn, color="#0b0b0b", lw=1, label="RCN switch from the zipper table (%.1f deg)" % rcn)
    ax.set_xlabel("switch angle used in the assembly (deg); 90 = no zipper (knee everywhere)")
    ax.set_ylabel("total RCN energy / knee-everywhere energy")
    ax.set_title("Ellipse-level check: which switch angle does the RCN energy prefer?", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out / "fig_switch_energy.png", dpi=170); plt.close(fig)


class _Args(object):
    dx = 0.35                   # assembly grid for figures and fine/coarse energies
    dx_energy = 0.5             # assembly grid for the switch-angle check
    n_strips = 4                # coarse mode
    s_center_frac = 0.3         # wall phase (energy-preferred range on the large ellipses)
    M = 4096
    only = []                   # restrict to these tags
    out_dir = None


def _parse_cli():
    d = _Args()
    p = argparse.ArgumentParser(description="Aspect-ratio sweep of the RCN zipper ellipse construction")
    for k in ["dx", "dx_energy", "s_center_frac"]:
        p.add_argument("--" + k, type=float, default=getattr(d, k))
    for k in ["n_strips", "M"]:
        p.add_argument("--" + k, type=int, default=getattr(d, k))
    p.add_argument("--only", nargs="*", default=d.only)
    p.add_argument("--out_dir", default=d.out_dir)
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_cli() if len(sys.argv) > 1 else _Args())
