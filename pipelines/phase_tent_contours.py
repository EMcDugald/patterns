#!/usr/bin/env python
"""
phase_tent_contours.py  v1.0 (28 Oct 2026)

Two pictures that sit beside the migration study (nucleation_migration.py v1.2, MIGRATION_LOCAL_RESULTS_1028.md):

(1) THE PHASE SURFACE ("the upside-down tent"), files tent_*. At the migration study's exemplar core, at its stage
    frames (first frame, stages 1-2, first crossing, second crossing, last frame), the smooth single-valued phase
    phi_LS (least-squares integration of mg's k at width 0.4; mg_estimator_validation.integrate_k, the record's
    smooth phase) with the far-field ramp removed:
        psi(x, y) = phi_LS(x, y) - k1 (x - x_core) - phi_LS(x_core, y_line),   k1 = sqrt(1 - mu^2)
    so a knee is a trough along the line (psi rises away from the line on both sides) whose floor is level along x,
    and a fold shows up as a stretch where the floor no longer keeps pace with the ramp. Drawn as a surface, as a
    map with the stripe edges (u = 0) and the cells of u (width 0.4), and as cuts psi(y) at three positions along the
    line (through the amplitude dip, x_core; and at 0.3 L and 0.5 L, where the X cells sit). An ILLUSTRATION: phi_LS
    is the best smooth phase consistent with mg's k, not mg's own field.

(2) THE CONTOUR TOPOLOGY ("do the stripe edges cross the line?"), files con_*. Model-free: the zero contours of u
    (the stripe edges; u and cos(theta) share them exactly). Each contour is cut into the pieces that lie within
    cls_h_L L of the line; a piece that comes within near_L L of the line is
        crossing    if it has points on both sides of the line beyond sep_L L
        turning     if it stays on one side (it runs along the line, or turns back)
    and pieces that never come that close are not counted. f_cross = crossing / (crossing + turning): 1 when every
    stripe edge passes through the line (a knee), falling as the edges stay in their halves (the contour change
    asked about in the 28 Oct e-mail). Also the rate at which u changes sign along the line itself, per L
    (1 for a knee: two zero crossings per stripe period 2 L). Pieces are counted only in the central window
    |x - x_mid| < win_frac Lx. Run across angle on the stationary large sweep (--sweep), and in time on movies
    (--npz), where the migration study's median stage onsets (--mig_dir) are marked.

Outputs (folder names 1028_phasecon_*; files begin tent_ or con_):
  --npz movies with --mig_dir:  <stem>/tent_panels.png/.pdf, <stem>/con_time.png/.pdf (f_cross and the line rate
                                against time, stage onsets marked), <stem>/con_frames.png/.pdf (contours at the stage
                                frames, colored by class), <stem>/con_time.csv
  --sweep (stationary frames):  con_survey.png/.pdf (f_cross and the line rate against mu), con_gallery.png/.pdf
                                (contours at --gallery_mus), con_survey.csv
  phasecon_notes.txt

Changelog
v1.0 (28 Oct 2026)
  - First version, run end to end in the sandbox on stand-ins only (no SH data there). An analytic self-dual knee at
    mu 0.60 (R 0.5) reads f_cross = 1 (6 of 6 near-line pieces cross) and a line rate of 0.94 per L (1 expected; the
    window holds a non-integer number of periods), as a knee requires. The planted vortex pair on the knee phase
    keeps f_cross = 1 at every separation: a vortex pair does not by itself make the stripe edges stay in their
    halves. An analytic knee at mu 0.88 with R 0.5 has zero slaved amplitude on the line (mu^4 > R), so u = 0 there
    and neither measure is defined; both need A > 0 on the line, which holds in every SH frame measured so far
    (amplitude minimum 0.48 to 0.74 of far field). Numbers of record come from Ed's run.

Imports nucleation_migration (and through it the earlier scripts); edits none. Python 3.8 compatible.
Run from ~/patterns/pipelines (PHASECON_CLI_1028.txt).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nucleation_migration as M                                          # noqa: E402
F, P = M.F, M.P
import matplotlib                                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.lines import Line2D                                       # noqa: E402
from mpl_toolkits.mplot3d import Axes3D                                   # noqa: E402,F401

VERSION = "phase_tent_contours v1.0 (28 Oct 2026, on %s)" % M.VERSION
C_CROSS, C_TURN = "#1e8449", "#c0392b"


# ---------------------------------------------------------------------
# contours of u = 0 and their classes
# ---------------------------------------------------------------------

def zero_contours(x, y, u):
    """list of (n, 2) arrays (x, y) along u = 0"""
    try:
        import contourpy
        gen = contourpy.contour_generator(x, y, u, line_type=contourpy.LineType.Separate)
        return [np.asarray(s) for s in gen.lines(0.0) if len(s) > 1]
    except ImportError:
        fig, ax = plt.subplots()
        cs = ax.contour(x, y, u, levels=[0.0])
        segs = [np.asarray(s) for s in cs.allsegs[0] if len(s) > 1]
        plt.close(fig)
        return segs


def classify(segs, y_row, x_mid, Lx, L, args):
    """pieces of the contours within cls_h_L L of the line; returns lists of pieces by class"""
    h, near, sep = args.cls_h_L * L, args.near_L * L, args.sep_L * L
    out = {"cross": [], "turn": [], "away": []}
    for s in segs:
        d = s[:, 1] - y_row
        inside = np.abs(d) < h
        if not inside.any():
            continue
        idx = np.where(inside)[0]
        brk = np.where(np.diff(idx) > 1)[0]
        for run in np.split(idx, brk + 1):
            if len(run) < 2:
                continue
            p = s[run]
            if abs(np.mean(p[:, 0]) - x_mid) > args.win_frac * Lx:
                continue
            dd = p[:, 1] - y_row
            if np.min(np.abs(dd)) > near:
                out["away"].append(p)
            elif (dd > sep).any() and (dd < -sep).any():
                out["cross"].append(p)
            else:
                out["turn"].append(p)
    return out


def line_rate(u2, x, jr, x_mid, Lx, L, args):
    sel = np.abs(x - x_mid) < args.win_frac * Lx
    v = u2[jr, sel]
    n = int(np.sum(np.sign(v[1:]) * np.sign(v[:-1]) < 0))
    span = float(np.sum(sel) * (x[1] - x[0]))
    return n / (span / L) if span > 0 else float("nan")


def frame_contours(u2, x, y, y_row, mu, args):
    k1 = float(np.sqrt(1 - mu ** 2)); L = np.pi / k1
    Lx = len(x) * (x[1] - x[0]); x_mid = float(x.mean())
    jr = int(np.argmin(np.abs(y - y_row)))
    segs = zero_contours(x, y, u2)
    cl = classify(segs, y_row, x_mid, Lx, L, args)
    nc, nt = len(cl["cross"]), len(cl["turn"])
    return dict(cl=cl, segs=segs, L=L, Lx=Lx, x_mid=x_mid, n_cross=nc, n_turn=nt,
                f_cross=(nc / (nc + nt)) if (nc + nt) else float("nan"),
                rate=line_rate(u2, x, jr, x_mid, Lx, L, args))


def draw_contours(ax, fc, x, y, y_row, xc, args, half_x_L, half_y_L, cells=None, small=False):
    L = fc["L"]
    for s in fc["segs"]:
        ax.plot((s[:, 0] - xc) / L, (s[:, 1] - y_row) / L, "-", color="0.75", lw=0.6)
    for key, col in (("cross", C_CROSS), ("turn", C_TURN)):
        for p in fc["cl"][key]:
            ax.plot((p[:, 0] - xc) / L, (p[:, 1] - y_row) / L, "-", color=col, lw=1.6 if not small else 1.2)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    if cells is not None:
        px, py, pq = cells
        for sel, mk, col in ((pq > 0, "^", M.CX), (pq < 0, "v", M.CV)):
            ax.plot((px[sel] - xc) / L, (py[sel] - y_row) / L, mk, ms=5, mfc=col, mec="w", mew=0.6, ls="none")
    ax.set_xlim(-half_x_L, half_x_L); ax.set_ylim(-half_y_L, half_y_L)
    ax.set_aspect("equal"); ax.tick_params(labelsize=6.5)


def con_key():
    return [Line2D([], [], color=C_CROSS, lw=2, label="stripe edge crosses the line"),
            Line2D([], [], color=C_TURN, lw=2, label="stripe edge stays on one side"),
            Line2D([], [], color="0.75", lw=1, label="other stripe edges (u = 0)"),
            Line2D([], [], color="k", ls="--", lw=0.8, label="the line")]


# ---------------------------------------------------------------------
# the phase surface
# ---------------------------------------------------------------------

def psi_crop(u2, x, y, y_row, mu, xc, args):
    D = M.frame_fields(u2, x, y, y_row, mu, args)
    a = argparse.Namespace(**vars(args)); a.ls_mode = "plain"
    phi, _ = P.smooth_phase(D["e"], x, y, D["G"], a)
    k1 = float(np.sqrt(1 - mu ** 2)); L = np.pi / k1
    X, Y = np.meshgrid(x, y)
    psi = phi - k1 * (X - xc)
    ic = int(np.argmin(np.abs(x - xc))); jr = int(np.argmin(np.abs(y - y_row)))
    psi = psi - psi[jr, ic]
    ix = np.abs(x - xc) <= args.tent_half_x_L * L
    jy = np.abs(y - y_row) <= args.tent_half_y_L * L
    sub = lambda f: f[np.ix_(jy, ix)]                                     # noqa: E731
    return dict(psi=sub(psi), u=sub(u2), xs=(x[ix] - xc) / L, ys=(y[jy] - y_row) / L, L=L, cells=D["cu"],
                fold=(x[D["fku"]] - xc) / L)


def fig_tent(cols, title, path, args):
    n = len(cols)
    fig = plt.figure(figsize=(3.0 * n + 0.6, 9.6))
    gs = fig.add_gridspec(3, n, height_ratios=[1.25, 1.0, 0.9], left=0.05, right=0.98, top=0.88, bottom=0.06,
                          hspace=0.28, wspace=0.18)
    allpsi = np.concatenate([c[3]["psi"].ravel() for c in cols])
    lo, hi = np.percentile(allpsi, 1), np.percentile(allpsi, 99)
    cuts = [0.0, 0.3, 0.5]
    ccol = ["#2c3e50", "#d35400", "#8e44ad"]
    for j, (lab, col, t, cr) in enumerate(cols):
        Xs, Ys = np.meshgrid(cr["xs"], cr["ys"])
        ax = fig.add_subplot(gs[0, j], projection="3d")
        st = max(1, len(cr["xs"]) // 45)
        ax.plot_surface(Xs[::st, ::st], Ys[::st, ::st], cr["psi"][::st, ::st], cmap="viridis", vmin=lo, vmax=hi,
                        linewidth=0, antialiased=True, alpha=0.95)
        ax.plot(cr["xs"], np.zeros_like(cr["xs"]), cr["psi"][np.argmin(np.abs(cr["ys"])), :], "k-", lw=1.2)
        ax.set_zlim(lo, hi); ax.view_init(elev=args.tent_elev, azim=args.tent_azim)
        ax.set_xlabel("x (L)", fontsize=6.5, labelpad=-6); ax.set_ylabel("y (L)", fontsize=6.5, labelpad=-6)
        ax.tick_params(labelsize=5.5, pad=-2)
        ax.set_title("%s\nt = %.4g" % (lab, t), fontsize=8, color=col)
        ax = fig.add_subplot(gs[1, j])
        im = ax.contourf(Xs, Ys, cr["psi"], levels=np.linspace(lo, hi, 21), cmap="viridis", extend="both")
        ax.contour(Xs, Ys, cr["u"], levels=[0.0], colors="w", linewidths=0.7)
        ax.axhline(0, color="k", lw=0.6, ls="--")
        f = cr["fold"][np.abs(cr["fold"]) <= args.tent_half_x_L]
        if f.size:
            ax.plot(f, np.zeros_like(f), "s", ms=2.2, color="#e74c3c", ls="none")
        px, py, pq = cr["cells"]
        L = cr["L"]
        for c_x, c_c in zip(cuts, ccol):
            ax.axvline(c_x, color=c_c, lw=1.0, ls=":")
        ax.set_xlim(cr["xs"][0], cr["xs"][-1]); ax.set_ylim(cr["ys"][0], cr["ys"][-1]); ax.set_aspect("equal")
        ax.tick_params(labelsize=6.5)
        if j == 0:
            ax.set_ylabel("y (L)", fontsize=7)
        ax.set_xlabel("x (L)", fontsize=7)
        ax = fig.add_subplot(gs[2, j])
        for c_x, c_c in zip(cuts, ccol):
            ic = int(np.argmin(np.abs(cr["xs"] - c_x)))
            ax.plot(cr["ys"], cr["psi"][:, ic], "-", color=c_c, lw=1.4, label="x = %.1f L" % c_x)
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.set_ylim(lo, hi); ax.tick_params(labelsize=6.5)
        ax.set_xlabel("y (L), across the line", fontsize=7)
        if j == 0:
            ax.set_ylabel("psi = phi_LS - k1 (x - x_core)", fontsize=7)
            ax.legend(fontsize=6.5)
    fig.suptitle(title + "\nthe smooth single-valued phase with the far-field ramp removed (an illustration: phi_LS is the "
                 "best smooth phase consistent with mg's k)\nmiddle: map of psi, stripe edges u = 0 in white, fold (mg's k) "
                 "red; bottom: cuts across the line at the dotted positions", fontsize=8.5)
    fig.savefig(path.with_suffix(".png"), dpi=170); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)


# ---------------------------------------------------------------------
# one movie (tent and contours in time)
# ---------------------------------------------------------------------

def read_mig(mig_dir, stem):
    p = Path(mig_dir) / "mig_cores.csv"
    if not p.is_file():
        return []
    with open(p) as fh:
        return [r for r in csv.DictReader(fh) if r["stem"] == stem]


def fnum(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def run_movie(path, args, out, lines):
    u, x, y, t, md = F.movie(path)
    mu = float(md.get("mu", F.mu_of(path)))
    jr = int(md.get("gb_row_in_crop", u.shape[0] // 2)); y_row = float(y[jr])
    nt = u.shape[-1]
    ts = np.array([float(t[min(i, t.size - 1)]) for i in range(nt)])
    stem = Path(path).stem
    d = out / stem; d.mkdir(parents=True, exist_ok=True)
    rows = read_mig(args.mig_dir, stem) if args.mig_dir else []
    inner = [r for r in rows if r.get("edge_core", "0") in ("0", "0.0")]
    med = {}
    for k in ("t_s1", "t_s2", "t_s3", "t_s4", "t_cos"):
        v = [fnum(r[k]) for r in inner if np.isfinite(fnum(r[k]))]
        med[k] = float(np.median(v)) if v else float("nan")
    # contours in time
    recs = []
    for i in range(0, nt, max(1, args.time_stride)):
        fc = frame_contours(np.asarray(u[..., i], float), x, y, y_row, mu, args)
        recs.append(dict(t=ts[i], frame=i, f_cross=fc["f_cross"], n_cross=fc["n_cross"], n_turn=fc["n_turn"],
                         line_rate_per_L=fc["rate"]))
    with open(d / "con_time.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)
    tt = np.array([r["t"] for r in recs])
    fig, ax = plt.subplots(figsize=(7.0, 3.6), constrained_layout=True)
    ax.plot(tt, [r["f_cross"] for r in recs], "-", color=C_CROSS, lw=1.5, label="share of near-line stripe edges that cross")
    ax.plot(tt, [r["line_rate_per_L"] for r in recs], "-", color="0.3", lw=1.2,
            label="sign changes of u along the line, per L (knee: 1)")
    for k, lab in (("t_s1", "cells of u (1-2)"), ("t_s2", "fold (1-2)"), ("t_cos", "cells of cos(theta)"),
                   ("t_s3", "first crossing (3)"), ("t_s4", "second crossing (4)")):
        if np.isfinite(med[k]):
            ax.axvline(med[k], color=M.SCOL.get(k[2:], "0.4") if k != "t_cos" else "#2471a3", ls=":", lw=1.0)
            ax.text(med[k], 1.03, lab, rotation=90, fontsize=6, va="bottom", ha="center",
                    transform=ax.get_xaxis_transform())
    if args.time_scale == "sqrt":
        ax.set_xscale("function", functions=(lambda v: np.sqrt(np.clip(v, 0, None)), lambda v: np.asarray(v) ** 2))
    ax.set_ylim(-0.05, 1.3); ax.set_xlabel("time"); ax.legend(fontsize=7, loc="lower left")
    ax.set_title("%s  mu %.3f: do the stripe edges cross the line? (median stage onsets of the interior cores)"
                 % (stem, mu), fontsize=8, pad=48)
    fig.savefig(d / "con_time.png", dpi=170); fig.savefig(d / "con_time.pdf"); plt.close(fig)
    # stage frames at the exemplar core
    ex = None
    if inner:
        hits = [(fnum(r["t_s4"]), r) for r in inner if np.isfinite(fnum(r["t_s4"]))]
        ex = (sorted(hits, key=lambda h: h[0])[0][1] if hits else inner[len(inner) // 2])
        if args.core >= 0:
            ex = [r for r in rows if int(float(r["core"])) == args.core][0]
    G0 = P.geometry(np.asarray(u[..., -1], float), x, y, y_row, mu, M.default_args())
    L = G0["L"]
    xc = G0["xm"] + fnum(ex["x_core_L"]) * L if ex is not None else G0["xm"]
    def idx(tv):
        return int(np.argmin(np.abs(ts - tv)))
    marks = [("first frame", "#7f8c8d", 0)]
    if ex is not None:
        t12 = np.nanmin([fnum(ex["t_s1"]), fnum(ex["t_s2"])])
        if np.isfinite(t12):
            marks.append(("stages 1-2", M.SCOL["s1"], idx(max(fnum(ex["t_s1"]), fnum(ex["t_s2"])) if
                                                         np.isfinite(fnum(ex["t_s1"])) and np.isfinite(fnum(ex["t_s2"]))
                                                         else t12)))
        for k, lab in (("t_s3", "first crossing"), ("t_s4", "second crossing")):
            if np.isfinite(fnum(ex[k])):
                marks.append((lab, M.SCOL["s" + k[-1]], idx(fnum(ex[k]))))
    marks.append(("last frame", "#6c3483", nt - 1))
    title = "%s   mu %.3f (%.0f deg), core %s at x = %+.2f L" % (stem, mu, np.degrees(np.arcsin(mu)),
                                                                   ex["core"] if ex is not None else "-",
                                                                   (xc - G0["xm"]) / L)
    # contour frames
    n = len(marks)
    fig, axs = plt.subplots(1, n, figsize=(2.9 * n + 0.4, 3.2), constrained_layout=True, squeeze=False)
    for ax, (lab, col, i0) in zip(axs[0], marks):
        u2 = np.asarray(u[..., i0], float)
        fc = frame_contours(u2, x, y, y_row, mu, args)
        D = M.frame_fields(u2, x, y, y_row, mu, args)
        draw_contours(ax, fc, x, y, y_row, xc, args, args.con_half_x_L, args.con_half_y_L, cells=D["cu"], small=True)
        ax.set_title("%s, t = %.4g\ncrossing share %.2f" % (lab, ts[i0], fc["f_cross"]), fontsize=7.5, color=col)
    fig.legend(handles=con_key(), loc="upper center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.suptitle(title + "  (x, y in L about the core; cells of u at width 0.4)", fontsize=8, y=1.2)
    fig.savefig(d / "con_frames.png", dpi=170, bbox_inches="tight"); fig.savefig(d / "con_frames.pdf", bbox_inches="tight")
    plt.close(fig)
    # the phase surface
    if args.tent:
        cols = []
        for lab, col, i0 in marks:
            cols.append((lab, col, ts[i0], psi_crop(np.asarray(u[..., i0], float), x, y, y_row, mu, xc, args)))
        fig_tent(cols, title, d / "tent_panels", args)
    msg = ("%s mu %.3f: crossing share %.2f at t = %.4g, %.2f at the last frame; line rate %.2f to %.2f per L; "
           "exemplar core %s" % (stem, mu, recs[0]["f_cross"], recs[0]["t"], recs[-1]["f_cross"],
                                 recs[0]["line_rate_per_L"], recs[-1]["line_rate_per_L"],
                                 ex["core"] if ex is not None else "-"))
    print(msg); lines.append(msg)


# ---------------------------------------------------------------------
# across angle (stationary frames)
# ---------------------------------------------------------------------

def run_sweep(args, out, lines):
    files = F.select_files(args.sweep, args.sweep_mus)
    if not files:
        lines.append("LOUD: no stationary frames under --sweep %s" % args.sweep); return
    rows, keep = [], {}
    gal = [float(v) for v in args.gallery_mus.split(",") if v] if args.gallery_mus else []
    for p in files:
        u, x, y, t, md = F.movie(p)
        mu = float(md.get("mu", F.mu_of(p)))
        jr = int(md.get("gb_row_in_crop", u.shape[0] // 2)); y_row = float(y[jr])
        u2 = np.asarray(u[..., args.sweep_frame], float)
        fc = frame_contours(u2, x, y, y_row, mu, args)
        rows.append(dict(mu=mu, stem=Path(p).stem, f_cross=fc["f_cross"], n_cross=fc["n_cross"], n_turn=fc["n_turn"],
                         line_rate_per_L=fc["rate"]))
        for g in gal:
            if abs(mu - g) < 0.0051:
                keep[g] = (mu, u2, x, y, y_row, fc)
    rows.sort(key=lambda r: r["mu"])
    with open(out / "con_survey.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    mus = np.array([r["mu"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    lo, hi = args.regime_edges
    ax.axvspan(mus.min() - 0.01, lo, color="#e9e7e2", lw=0)
    ax.axvspan(lo, hi, color="#f6e3b4", lw=0)
    ax.axvspan(hi, mus.max() + 0.01, color="#cfe0f5", lw=0)
    ax.plot(mus, [r["f_cross"] for r in rows], "o-", color=C_CROSS, ms=3.5, lw=1.2,
            label="share of near-line stripe edges that cross the line")
    ax.plot(mus, [r["line_rate_per_L"] for r in rows], "s-", color="0.3", ms=3, lw=1.0,
            label="sign changes of u along the line, per L (knee: 1)")
    ax.set_xlabel("mu"); ax.set_ylim(-0.05, 1.25); ax.legend(fontsize=7, loc="lower left")
    ax.set_title("Stationary boundaries (t = %s): do the stripe edges cross the line?  shading: knee, intermediate, "
                 "dislocation (edges %.3f, %.3f)" % (args.sweep_t_label, lo, hi), fontsize=8)
    fig.savefig(out / "con_survey.png", dpi=170); fig.savefig(out / "con_survey.pdf"); plt.close(fig)
    if keep:
        gs_ = sorted(keep)
        fig, axs = plt.subplots(1, len(gs_), figsize=(3.0 * len(gs_) + 0.4, 3.0), constrained_layout=True, squeeze=False)
        for ax, g in zip(axs[0], gs_):
            mu, u2, x, y, y_row, fc = keep[g]
            draw_contours(ax, fc, x, y, y_row, float(x.mean()), args, args.gal_half_x_L, args.con_half_y_L, small=True)
            ax.set_title("mu %.2f (%.0f deg)\ncrossing share %.2f" % (mu, np.degrees(np.arcsin(mu)), fc["f_cross"]),
                         fontsize=7.5)
        fig.legend(handles=con_key(), loc="upper center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.15))
        fig.savefig(out / "con_gallery.png", dpi=170, bbox_inches="tight")
        fig.savefig(out / "con_gallery.pdf", bbox_inches="tight"); plt.close(fig)
    knee = [r for r in rows if r["mu"] < lo]
    first = [r["mu"] for r in rows if r["f_cross"] < args.report_below]
    lines.append("sweep: %d frames; crossing share %.2f to %.2f in the knees (mu < %.3f); first mu with crossing share "
                 "below %.2f: %s" % (len(rows), min(r["f_cross"] for r in knee) if knee else float("nan"),
                                     max(r["f_cross"] for r in knee) if knee else float("nan"), lo, args.report_below,
                                     ("%.3f" % first[0]) if first else "none"))
    for r in rows:
        lines.append("   mu %.3f  crossing share %.3f (%d cross, %d stay)  line rate %.3f per L"
                     % (r["mu"], r["f_cross"], r["n_cross"], r["n_turn"], r["line_rate_per_L"]))


def main(args):
    t0 = time.time()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    lines = [VERSION, "args: " + json.dumps(vars(args)), ""]
    if args.npz:
        files = F.select_files(args.npz, args.mus)
        excl = [s for s in args.exclude.split(",") if s]
        files = [f for f in files if not any(s in Path(f).name for s in excl)]
        for p in files:
            run_movie(p, args, out, lines)
    if args.sweep:
        run_sweep(args, out, lines)
    lines += ["", "done in %.0f s" % (time.time() - t0)]
    (out / "phasecon_notes.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[3:]))
    return 0


class _Args:
    npz = ""                       # movies (folders or files)
    mus = ""
    exclude = "last10,f4-84"
    mig_dir = ""                   # the migration run's output folder (mig_cores.csv) for the exemplar and onsets
    core = -1
    sweep = ""                     # stationary frames (the 24 Oct large sweep raw folder)
    sweep_mus = ""
    sweep_frame = -1
    sweep_t_label = "200"
    gallery_mus = "0.60,0.70,0.76,0.80,0.84,0.90"
    regime_edges = [0.685, 0.815]
    report_below = 0.9
    outdir = str(Path("results") / "phase_tent_contours")
    tent = 1
    tent_half_x_L = 1.0
    tent_half_y_L = 0.8
    tent_elev = 28.0
    tent_azim = -60.0
    con_half_x_L = 1.0
    con_half_y_L = 0.8
    gal_half_x_L = 2.0
    cls_h_L = 0.6                  # pieces of the u = 0 contours within this distance of the line, L
    near_L = 0.1                   # a piece is counted when it comes this close to the line, L
    sep_L = 0.02                   # crossing: points on both sides beyond this, L
    win_frac = 0.4                 # central window |x - x_mid| < win_frac Lx
    time_stride = 1
    time_scale = "sqrt"
    # shared with nucleation_migration (same values)
    width = 0.4
    with_phi = 0
    turn_hi = 0.9
    win_x_L = 0.5
    win_y_L = 0.75
    mu = 0.85
    sigma_dir = 0.4
    taper_frac = 0.06
    sigma_det = 0.4
    band_L = 2.0
    core_band_L = 1.0
    far_from = 4.0
    core_frac = 0.70
    loop_L = 0.5
    fold_off_L = 0.05
    ls_mode = "plain"
    seam_dilate = 1
    nang = 181
    on_line_L = 0.1


def default_args():
    return argparse.Namespace(**{k: getattr(_Args, k) for k in vars(_Args) if not k.startswith("_")})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in vars(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        if isinstance(v, list):
            p.add_argument("--" + k, type=float, nargs="+", default=v)
        else:
            p.add_argument("--" + k, type=type(v), default=v)
    sys.exit(main(p.parse_args()))
