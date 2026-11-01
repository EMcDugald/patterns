#!/usr/bin/env python
"""
nucleation_stages_figures_v11.py  v1.1 (27 Oct 2026)

v1.1 changes (v1.0 kept unchanged as nucleation_stages_figures.py; no measured number changes):
  (1) A key drawn into the animation and into stages_panels: every marker explained in words, and a color wheel
      showing which color means which direction of k, so the figures read without a separate guide.
  (2) The animation pauses at each stage onset (--hold_frames repeats of that frame, default 6) and runs slower
      (--fps default 5, was 8).
  (3) The stage-timing diagram and jump grid are unchanged; the v1.1 command list does not redraw them.
  (4) Labels use US spelling throughout (label text only; 27 Oct, after Ed's first v1.1 run).

nucleation_stages_figures.py  v1.0 (27 Oct 2026)

Figures and animations of the stages of nucleation at one core, as seen through the mg estimator, plus the
stage-timing diagram over every core of every movie. Builds on core_track_time_series.py
(CORE_TRACK_LOCAL_RESULTS_1027.md), whose per-core results define the stages:

  stage 0  the knee: no half-turn cells near the core, no fold
  stage 1  amplitude texture: half-turn cells of u (detector width 0.3) near the core, none in cos(theta)
  stage 2  the fold: the wavevector just above the line points against the one just below over more than
           5 percent of the line near the core (at mu >= 0.88 this comes before stage 1 or is present from t = 0)
  stage 3  the phase rearranges: the first reversal of mg's k on the 0.5 L loop about the core (the loop is open;
           the circulation has no reading); cells of cos(theta) appear near this time
  stage 4  one turn of circulation: the second reversal, the loop closes and reads one turn

What the reversals are. The parity of the number of reversals of mg's k on a loop equals the parity of the number
of half-turn cells of mg's own director (detection width 0.4, the estimator's sigma_dir) inside the loop: 135 of 135
loops on SH stand-in frames, and 100 percent of frames for the drawn core in Ed's run of 27 Oct (coarse mu 0.83,
fine mu 0.90); 76 percent on the planted vortex-pair movie, where the amplitude falls to zero. In that run the
count of width-0.4 cells inside the loop went, at both cores, 0, then 2 (one V and one X, 0.05 to 0.07 L apart),
then 6, then 4 (two V and two X) with no reversal and no circulation; at stage 3 one X cell on the line leaves the
loop across its crossing point with the line (3 inside, one reversal, open); at stage 4 an X cell from the
neighbouring core's side enters across the other crossing point (4 inside, two reversals, one turn). So the
circulation step is the moment the X cells on the line, drifting along it, have crossed both points where the loop
meets the line.

Outputs (every file begins with stages_; the command list puts them in folders named 1027_stages_*):
  per movie, in <outdir>/<stem>/:
    stages_panels.png / .pdf     one column per stage onset at the chosen core (plus the first and last frame);
                                 rows: (a) the pattern u with its cells (width 0.3), (b) the amplitude A / A_far,
                                 (c) the phase alone, cos(theta), with its cells, (d) the direction of mg's k
                                 (hue: angle on a cyclic map, so a reversal is a sharp change between blue and
                                 red; arrows: k), the 0.5 L loop with its reversal points, the cells of mg's own
                                 director (width 0.4) and the folded stretch of the line
    stages_closeup.png / .pdf    rows (a) and (d) over the open stretch of stage 3: from the frame before the
                                 first reversal to the frame after one turn (at most --closeup_max columns)
    stages_series.png / .pdf     the chosen core against time with the stages shaded: amplitude, fold, cells
                                 (width 0.3 of u and of cos(theta); width 0.4 inside the loop), reversals,
                                 circulation
    stages_anim.gif (and stages_anim.mp4 when ffmpeg is available)
                                 the same four panels as stages_panels through every --anim_stride-th frame,
                                 with a banner naming the current stage and a time bar
    stages_core_series.csv       the chosen core, one row per frame
  with --core_track_dirs (the core_track results folders):
    stages_timing.png / .pdf     every core of every movie: the time each stage begins against mu (symmetric log
                                 time axis), one vertical line per core from its first to its last stage; below,
                                 the open stretch of stage 3 (one turn minus first reversal)
    stages_jump_grid.png / .pdf  the core_track jump-score grid redrawn for reading: scores for the local
                                 quantities; the reversals on the loop as "1 to 2 at n of N cores" (their score was
                                 capped by the floor and read 50 everywhere); the plain line integral as its
                                 absolute change in turns (its score overstated a 0.04 to 0.11 turn change)
  stages_notes.txt

Imports core_track_time_series (and through it fold_time_series and phase_vs_amplitude_charge_v11); edits none.
Python 3.8 compatible. Run from ~/patterns/pipelines (STAGES_FIG_CLI_1027.txt).
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
import core_track_time_series as C                                         # noqa: E402
F = C.F
P = C.P
t1c = P.t1c
import matplotlib                                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib import animation                                          # noqa: E402
from matplotlib.patches import Circle                                     # noqa: E402

VERSION = "nucleation_stages_figures v1.1 (27 Oct 2026, on %s)" % C.VERSION

STAGES = [  # key, short name, banner text, color
    ("s0", "knee", "Stage 0: the knee. No half-turn cells near the core, no fold", "#7f8c8d"),
    ("s1", "amplitude texture", "Stage 1: amplitude texture. Half-turn cells in u, none in the phase alone", "#1e8449"),
    ("s2", "fold", "Stage 2: the fold. k above the line points against k below", "#c0392b"),
    ("s3", "phase rearranges", "Stage 3: the phase rearranges. One reversal of k on the loop; the loop is open", "#2471a3"),
    ("s4", "one turn", "Stage 4: one turn of circulation. Second reversal; the loop closes", "#6c3483"),
]
SCOL = {k: c for k, _, _, c in STAGES}
SNAME = {k: n for k, n, _, _ in STAGES}
SBANNER = {k: b for k, _, b, _ in STAGES}


# ---------------------------------------------------------------------
# per-frame record of one core, with crops for drawing
# ---------------------------------------------------------------------

def mg_cells(u2, x, y, G, args):
    """half-turn cells of the director at mg's own width (sigma_dir), in the survey's band and window"""
    L, y_row = G["L"], G["y_row"]
    det = t1c.detect_defects(u2 * G["taper"], x, y - y_row, G["Lx"], G["Ly"], sigma_det=args.sigma_dir,
                             band_y=args.band_L * L)
    q = det["q"]
    J, I = np.nonzero(q)
    return det["xc"][I], det["yc"][J] + y_row, q[J, I].astype(int)


def loop_reversals(k1, k2, x, y, xc, yc, r, nang):
    dx = x[1] - x[0]; dy = y[1] - y[0]
    ang = np.linspace(0, 2 * np.pi, nang)
    px = xc + r * np.cos(ang); py = yc + r * np.sin(ang)
    ix = np.clip(np.round((px - x[0]) / dx).astype(int), 0, len(x) - 1)
    iy = np.clip(np.round((py - y[0]) / dy).astype(int), 0, len(y) - 1)
    kx, ky = k1[iy, ix], k2[iy, ix]
    rev = np.where(kx[1:] * kx[:-1] + ky[1:] * ky[:-1] < 0)[0]
    am = 0.5 * (ang[rev] + ang[rev + 1])
    return xc + r * np.cos(am), yc + r * np.sin(am)


def frame_core(u2, x, y, y_row, mu, xc, args):
    D = C.frame_fields(u2, x, y, y_row, mu, args)
    rec = C.core_record(D, x, y, xc, args)
    G, e = D["G"], D["e"]
    L = G["L"]
    r = args.loop_L * L
    mx, my, mq = mg_cells(u2, x, y, G, args)
    ins = np.hypot(mx - xc, my - y_row) < r
    rec["n04_in"] = float(np.sum(ins)); rec["V04_in"] = float(np.sum(ins & (mq < 0))); rec["X04_in"] = float(np.sum(ins & (mq > 0)))
    near = np.abs(mx - xc) < args.win_x_L * L
    V, X = near & (mq < 0), near & (mq > 0)
    if V.any() and X.any():
        rec["VX04_dist_L"] = float(np.min(np.hypot(mx[V][:, None] - mx[X][None, :], my[V][:, None] - my[X][None, :])) / L)
    else:
        rec["VX04_dist_L"] = float("nan")
    rx, ry = loop_reversals(e["k1"], e["k2"], x, y, xc, y_row, r, args.nang)
    ix = np.abs(x - xc) <= args.half_x_L * L
    jy = np.abs(y - y_row) <= args.half_y_L * L
    sub = lambda a: a[np.ix_(jy, ix)]                                      # noqa: E731
    crop = dict(u=sub(u2), A=sub(D["A"] / D["A_far"]), cos=sub(np.cos(e["theta"])), k1=sub(e["k1"]), k2=sub(e["k2"]),
                xs=(x[ix] - xc) / L, ys=(y[jy] - y_row) / L,
                cu=_rel(D["cu"], xc, y_row, L), cc=_rel(D["cc"], xc, y_row, L), cm=_rel((mx, my, mq), xc, y_row, L),
                rev=((rx - xc) / L, (ry - y_row) / L),
                fold=(x[D["fphi"]] - xc) / L)
    return rec, crop, L


def _rel(c, xc, yc, L):
    px, py, pq = c
    return (np.asarray(px) - xc) / L, (np.asarray(py) - yc) / L, np.asarray(pq)


def stage_onsets(ts, S, args):
    def first(cond):
        h = np.where(np.asarray(cond, bool))[0]
        return int(h[0]) if len(h) else -1
    on = dict(s0=0,
              s1=first((S["nV_u"] + S["nX_u"]) > 0),
              s2=first(S["fold_loc_phi"] > args.fold_on),
              s3=first(S["flips"] >= 1),
              s4=first((S["closed"] > 0) & (S["G_carried"] >= args.turn_hi)),
              cos=first(S["n_cells_cos"] > 0),
              in04=first(S["n04_in"] > 0))
    return on


def stage_at(i, on):
    """the latest stage reached by frame i (stages may begin out of order; the banner names the latest by number)"""
    cur = "s0"
    for k in ["s1", "s2", "s3", "s4"]:
        if on[k] >= 0 and i >= on[k]:
            cur = k
    return cur


# ---------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------

def _ext(cr):
    return [cr["xs"][0], cr["xs"][-1], cr["ys"][0], cr["ys"][-1]]


def _cells(ax, c, size=5.5, diamond=False):
    px, py, pq = c
    if diamond:
        ax.plot(px[pq > 0], py[pq > 0], "D", ms=size, mfc="#f5b041", mec="w", mew=0.8, ls="none")
        ax.plot(px[pq < 0], py[pq < 0], "D", ms=size, mfc="#5dade2", mec="w", mew=0.8, ls="none")
    else:
        ax.plot(px[pq > 0], py[pq > 0], "^", ms=size, mfc="#d62728", mec="w", mew=0.7, ls="none")
        ax.plot(px[pq < 0], py[pq < 0], "v", ms=size, mfc="#1f77b4", mec="w", mew=0.7, ls="none")


def draw_row(ax, cr, row, args, small=False):
    ext = _ext(cr)
    ms = 4.5 if small else 5.5
    if row == "u":
        v = np.max(np.abs(cr["u"]))
        ax.imshow(cr["u"], origin="lower", extent=ext, cmap="Greys", vmin=-v, vmax=v, aspect="equal", interpolation="bilinear")
        _cells(ax, cr["cu"], ms)
    elif row == "A":
        ax.imshow(cr["A"], origin="lower", extent=ext, cmap="viridis", vmin=args.A_lo, vmax=args.A_hi, aspect="equal",
                  interpolation="bilinear")
        _cells(ax, cr["cu"], ms)
    elif row == "cos":
        ax.imshow(cr["cos"], origin="lower", extent=ext, cmap="Greys", vmin=-1, vmax=1, aspect="equal", interpolation="bilinear")
        _cells(ax, cr["cc"], ms)
    elif row == "k":
        ang = np.arctan2(cr["k2"], cr["k1"])
        ax.imshow(ang, origin="lower", extent=ext, cmap="twilight", vmin=-np.pi, vmax=np.pi, aspect="equal",
                  interpolation="nearest")
        ny, nx = cr["k1"].shape
        sx = max(1, nx // args.n_arrows)
        sy = max(1, int(round(sx * (cr["xs"][1] - cr["xs"][0]) / max(cr["ys"][1] - cr["ys"][0], 1e-12))))
        kk = np.hypot(cr["k1"], cr["k2"]) + 1e-12
        X, Y = np.meshgrid(cr["xs"], cr["ys"])
        ax.quiver(X[::sy, ::sx], Y[::sy, ::sx], (cr["k1"] / kk)[::sy, ::sx], (cr["k2"] / kk)[::sy, ::sx],
                  color="k", alpha=0.75, scale=args.arrow_scale, width=0.006, headwidth=3.5, headlength=4, pivot="mid")
        f = cr["fold"]
        f = f[np.abs(f) <= args.half_x_L]
        ax.plot(f, np.zeros_like(f), "s", ms=2.0, color="#c0392b", ls="none")
        _cells(ax, cr["cm"], ms - 0.5, diamond=True)
        rx, ry = cr["rev"]
        ax.plot(rx, ry, "X", ms=8 if not small else 6.5, mfc="w", mec="k", mew=1.2, ls="none")
    ax.add_patch(Circle((0, 0), args.loop_L, fill=False, ec="#f39c12", lw=1.3))
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.tick_params(labelsize=6.5)


ROWS = [("u", "(a) pattern u, cells (width 0.3)"), ("A", "(b) amplitude A / A_far"),
        ("cos", "(c) phase alone cos(theta), its cells"),
        ("k", "(d) direction of k; X: reversals on the loop;\ndiamonds: cells of mg's director (width 0.4)")]


def legend_text():
    return ("up-triangles: X cells, down-triangles: V cells (width 0.3); diamonds: cells at width 0.4 (orange X, blue V)\n"
            "orange circle: the 0.5 L loop; red squares on the line: folded stretch; white X: reversals of k on the loop")


def draw_key(ax, compact=False, ncol=2):
    """the key: markers in words, and a color wheel for the direction of k"""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fs = 7 if compact else 7.5
    items = [
        (("^", "#d62728", "w"), "X cell of u, or of cos(theta) in panel (c); width 0.3"),
        (("v", "#1f77b4", "w"), "V cell of u, or of cos(theta) in panel (c); width 0.3"),
        (("D", "#f5b041", "w"), "X cell of mg's own director, width 0.4"),
        (("D", "#5dade2", "w"), "V cell of mg's own director, width 0.4"),
        (("X", "w", "k"), "reversal of k on the loop"),
        (("o_ring", "#f39c12", None), "the 0.5 L loop (circulation)"),
        (("s", "#c0392b", "#c0392b"), "folded stretch of the line"),
    ]
    nrow = int(np.ceil(len(items) / float(ncol)))
    for n, ((m, fc, ec), txt) in enumerate(items):
        c, r = n // nrow, n % nrow
        x0 = 0.01 + c * 0.44
        y0 = 0.86 - r * (0.80 / max(nrow - 1, 1))
        if m == "o_ring":
            ax.plot([x0 + 0.012], [y0], "o", ms=9, mfc="none", mec=fc, mew=1.6, ls="none", transform=ax.transAxes)
        else:
            ax.plot([x0 + 0.012], [y0], m, ms=7 if m != "s" else 4, mfc=fc, mec=ec if ec else fc, mew=1.0,
                    ls="none", transform=ax.transAxes)
        ax.text(x0 + 0.035, y0, txt, fontsize=fs, va="center", ha="left", transform=ax.transAxes)
    # color wheel: the color of panel (d) for each direction of k (arrows show the direction)
    w = ax.inset_axes([0.86, 0.0, 0.14, 1.0])
    n = 201
    gx, gy = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    rr = np.hypot(gx, gy)
    ang = np.ma.masked_where((rr > 1) | (rr < 0.45), np.arctan2(gy, gx))
    w.imshow(ang, origin="lower", extent=[-1, 1, -1, 1], cmap="twilight", vmin=-np.pi, vmax=np.pi)
    for a_ in np.arange(0, 2 * np.pi, np.pi / 4):
        w.annotate("", xy=(0.95 * np.cos(a_), 0.95 * np.sin(a_)), xytext=(0.55 * np.cos(a_), 0.55 * np.sin(a_)),
                   arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
    w.set_aspect("equal"); w.axis("off")
    w.set_title("color of k\nby direction", fontsize=fs - 0.5, pad=1)


def fig_panels(cols, title, path, args):
    n = len(cols)
    kc = 1 if n <= 3 else 2                                               # key columns: one in a narrow figure
    fig = plt.figure(figsize=(2.55 * n + 1.2, 11.2 if kc == 2 else 12.4), constrained_layout=True)
    gs = fig.add_gridspec(5, n, height_ratios=[1, 1, 1, 1, 0.42 if kc == 2 else 0.8])
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(n)] for i in range(4)])
    for j, (lab, scol, t, cr, rec) in enumerate(cols):
        for i, (row, rlab) in enumerate(ROWS):
            ax = axs[i, j]
            draw_row(ax, cr, row, args)
            if j == 0:
                ax.set_ylabel(rlab + "\ny / L", fontsize=7)
            if i == 3:
                ax.set_xlabel("(x - x_core) / L", fontsize=7)
        axs[0, j].set_title("%s\nt = %.4g; circulation %s" % (lab, t, ("%.2f turn" % rec["G_carried"]) if rec["closed"]
                                                                   else "no reading (open)"),
                            fontsize=7.5, color=scol, fontweight="bold")
    draw_key(fig.add_subplot(gs[4, :]), ncol=kc)
    fig.suptitle(title, fontsize=8.5)
    fig.savefig(path, dpi=170); fig.savefig(str(path).replace(".png", ".pdf")); plt.close(fig)


def fig_closeup(frames_idx, ts, crops, recs, on, title, path, args):
    n = len(frames_idx)
    fig, axs = plt.subplots(2, n, figsize=(2.45 * n + 1.0, 5.2), constrained_layout=True, squeeze=False)
    for j, i in enumerate(frames_idx):
        st = stage_at(i, on)
        for r_, row in enumerate(["u", "k"]):
            draw_row(axs[r_, j], crops[i], row, args, small=True)
        rec = recs[i]
        axs[0, j].set_title("t = %.4g\n%d reversal%s, %s" % (ts[i], int(rec["flips"]), "" if rec["flips"] == 1 else "s",
                                                              ("%.2f turn" % rec["G_carried"]) if rec["closed"] else "open"),
                            fontsize=7.5, color=SCOL[st])
        axs[1, j].set_xlabel("width-0.4 cells inside: %d" % int(rec["n04_in"]), fontsize=7)
    axs[0, 0].set_ylabel("pattern u\ny / L", fontsize=7); axs[1, 0].set_ylabel("direction of k\ny / L", fontsize=7)
    fig.suptitle(title + "\nthe open stretch of stage 3, frame by frame\n" + legend_text(), fontsize=7.5)
    fig.savefig(path, dpi=170); fig.savefig(str(path).replace(".png", ".pdf")); plt.close(fig)


def fig_series(ts, S, on, title, path, args, tlabel="t"):
    fig, axs = plt.subplots(5, 1, figsize=(7.5, 9.5), sharex=True, constrained_layout=True)
    bounds = sorted([(on[k], k) for k in ["s1", "s2", "s3", "s4"] if on[k] >= 0])
    for ax in axs:
        for (i0, k) in bounds:
            ax.axvline(ts[i0], color=SCOL[k], lw=1.1, alpha=0.9)
        cur_t = ts[0]
        segs = [(0, "s0")] + bounds
        for a_, (i0, k) in enumerate(segs):
            t0 = ts[i0]; t1 = ts[segs[a_ + 1][0]] if a_ + 1 < len(segs) else ts[-1]
            ax.axvspan(t0, t1, color=SCOL[stage_at(i0, on)], alpha=0.07, lw=0)
    ax = axs[0]
    ax.plot(ts, S["A_min_loc"], "-", color="#1e8449", label="amplitude minimum near the core")
    ax.plot(ts, S["A_centre"], "--", color="#1e8449", label="amplitude on the line at the core")
    ax.set_ylabel("A / A_far", fontsize=8); ax.legend(fontsize=7, loc="best")
    ax = axs[1]
    ax.plot(ts, S["fold_loc_phi"], "-", color="#c0392b", label="folded share of the line near the core")
    ax.set_ylabel("fraction", fontsize=8); ax.legend(fontsize=7, loc="best")
    ax = axs[2]
    ax.step(ts, S["nV_u"] + S["nX_u"], where="mid", color="#1e8449", label="cells of u near the core (width 0.3)")
    ax.step(ts, S["n_cells_cos"], where="mid", color="#2471a3", label="cells of cos(theta) near the core (width 0.3)")
    ax.step(ts, S["n04_in"], where="mid", color="#6c3483", ls="--", label="cells of mg's director inside the loop (width 0.4)")
    ax.set_ylabel("count", fontsize=8); ax.legend(fontsize=7, loc="best")
    ax = axs[3]
    ax.step(ts, S["flips"], where="mid", color="k", label="reversals of k on the loop")
    ax.set_ylabel("count", fontsize=8); ax.legend(fontsize=7, loc="best")
    ax = axs[4]
    g = S["G_carried"]; cl = S["closed"] > 0
    ax.plot(ts, np.where(cl, g, np.nan), "-", color="#6c3483", lw=1.5, label="circulation, loop closed")
    ax.plot(ts, np.where(~cl, g, np.nan), ":", color="#6c3483", lw=1, alpha=0.6, label="loop open (no reading)")
    ax.plot(ts, S["G_plain"], "-", color="0.55", lw=0.9, label="plain line integral")
    ax.set_ylabel("turns", fontsize=8); ax.set_xlabel(tlabel, fontsize=8); ax.legend(fontsize=7, loc="best")
    for a in axs:
        a.tick_params(labelsize=7)
    handles = [plt.Line2D([], [], color=SCOL[k], lw=6, alpha=0.5) for k, _, _, _ in STAGES]
    axs[1].legend(handles, ["stage %s: %s" % (k[1], SNAME[k]) for k, _, _, _ in STAGES], fontsize=6.5, loc="center left",
                  bbox_to_anchor=(1.01, 0.5), frameon=False, title="shading and lines", title_fontsize=6.5)
    fig.suptitle(title, fontsize=8.5)
    fig.savefig(path, dpi=170); fig.savefig(str(path).replace(".png", ".pdf")); plt.close(fig)


def make_anim(ts, crops, recs, on, title, base, args):
    base_idx = list(range(0, len(ts), max(1, args.anim_stride)))
    onsets = set(on[k] for k in ["s1", "s2", "s3", "s4"] if on[k] >= 0)
    base_idx = sorted(set(base_idx) | onsets | {len(ts) - 1})
    idx = []
    for i in base_idx:
        idx += [i] * (1 + (args.hold_frames if i in onsets else 0))
    fig = plt.figure(figsize=(8.6, 9.0), dpi=args.anim_dpi)
    gs = fig.add_gridspec(4, 2, height_ratios=[0.13, 1, 1, 0.42], hspace=0.30, wspace=0.12, left=0.07, right=0.98,
                          top=0.94, bottom=0.03)
    axb = fig.add_subplot(gs[0, :])
    axs = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    draw_key(fig.add_subplot(gs[3, :]), compact=True)
    rows = ["u", "A", "cos", "k"]
    rtitle = ["pattern u, cells (width 0.3)", "amplitude A / A_far", "phase alone cos(theta), its cells",
              "direction of k, reversals (X), width-0.4 cells (diamonds)"]
    fig.suptitle(title, fontsize=8.5)

    def draw(n):
        i = idx[n]
        st = stage_at(i, on)
        for ax, row, rt in zip(axs, rows, rtitle):
            ax.clear()
            draw_row(ax, crops[i], row, args, small=True)
            ax.set_title(rt, fontsize=7.5)
        axb.clear()
        axb.set_xlim(ts[0], ts[-1]); axb.set_ylim(0, 1)
        for k in ["s1", "s2", "s3", "s4"]:
            if on[k] >= 0:
                axb.axvline(ts[on[k]], color=SCOL[k], lw=2)
        axb.axvspan(ts[0], ts[i], color=SCOL[st], alpha=0.35, lw=0)
        rec = recs[i]
        axb.text(0.005, 0.5, "t = %.4g   %s   (circulation: %s)" % (
            ts[i], SBANNER[st], ("%.2f turn" % rec["G_carried"]) if rec["closed"] else "loop open, no reading"),
            transform=axb.transAxes, va="center", ha="left", fontsize=8, color=SCOL[st], fontweight="bold")
        axb.set_yticks([]); axb.tick_params(labelsize=6.5)
        return axs
    an = animation.FuncAnimation(fig, draw, frames=len(idx), interval=1000.0 / args.fps, blit=False)
    out = []
    try:
        an.save(str(base) + ".gif", writer=animation.PillowWriter(fps=args.fps))
        out.append(str(base) + ".gif")
    except Exception as ex:                                              # loud, not fatal
        out.append("LOUD: gif not written (%s)" % ex)
    if args.mp4:
        try:
            if animation.writers.is_available("ffmpeg"):
                an.save(str(base) + ".mp4", writer=animation.FFMpegWriter(fps=args.fps, bitrate=2400))
                out.append(str(base) + ".mp4")
            else:
                out.append("mp4 skipped: ffmpeg not available")
        except Exception as ex:
            out.append("LOUD: mp4 not written (%s)" % ex)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------
# one movie
# ---------------------------------------------------------------------

def run_movie(path, args, out, lines):
    u, x, y, t, md = F.movie(path)
    mu = float(md.get("mu", F.mu_of(path)))
    jr = int(md.get("gb_row_in_crop", u.shape[0] // 2)); y_row = float(y[jr])
    nt = u.shape[-1]
    ts = np.array([float(t[min(i, t.size - 1)]) for i in range(nt)])
    stem = Path(path).stem
    d = out / stem; d.mkdir(parents=True, exist_ok=True)
    uL = np.asarray(u[..., -1], float)
    G0 = P.geometry(uL, x, y, y_row, mu, args)
    eL = P.estimate(uL, G0["Lx"], G0["Ly"], args.sigma_dir, None, G0["taper"])
    xd, yd, _, _, _ = P.find_dip_patches(eL["A"], x, y, G0["win"], args.core_frac, 0.5 * G0["L"])
    centres = F.centres_from(xd, G0["L"], G0["xm"], G0["Lx"])
    if not len(centres):
        lines.append("LOUD: %s: no cores in the last frame" % stem); return
    # pick the core: --core, else the first core to reach one turn (one pass over every frame for every core's loop)
    j = min(args.core, len(centres) - 1) if args.core >= 0 else None
    recs, crops = [], []
    if j is None:
        # one pass over all frames for all cores' loops, then pick
        allrec = []
        for i in range(nt):
            D = C.frame_fields(np.asarray(u[..., i], float), x, y, y_row, mu, args)
            allrec.append([C.core_record(D, x, y, xc, args) for xc in centres])
        hits = []
        for jj in range(len(centres)):
            ok = [i for i in range(nt) if allrec[i][jj]["closed"] and allrec[i][jj]["G_carried"] >= args.turn_hi]
            hits.append(ok[0] if ok else 10 ** 9)
        j = int(np.argmin(hits))
        if hits[j] == 10 ** 9:
            lines.append("%s: no core reaches one turn; drawing core 0" % stem); j = 0
    xc = float(centres[j])
    for i in range(nt):
        rec, cr, L = frame_core(np.asarray(u[..., i], float), x, y, y_row, mu, xc, args)
        recs.append(rec); crops.append(cr)
    del u
    keys = list(recs[0].keys())
    S = {k: np.array([r[k] for r in recs], float) for k in keys}
    on = stage_onsets(ts, S, args)
    rows = [dict(t=float(ts[i]), frame=i, stage=stage_at(i, on), **recs[i]) for i in range(nt)]
    P.write_rows(rows, d / "stages_core_series")
    title = "%s   mu %.3f (%.0f deg), core %d at x = %.2f L" % (stem, mu, np.degrees(np.arcsin(mu)), j,
                                                                  (xc - G0["xm"]) / G0["L"])
    # stage columns
    cols, seen = [], {}
    marks = [(0, "s0")] + [(on[k], k) for k in ["s1", "s2", "s3", "s4"] if on[k] >= 0] + [(nt - 1, "end")]
    for i0, k in sorted(marks):
        seen.setdefault(i0, []).append(k)
    for i0 in sorted(seen):
        ks = seen[i0]
        names = [("onset of stage %s: %s" % (k[1], SNAME[k])) if k != "end" else "last frame" for k in ks]
        if ks == ["s0"]:
            names = ["first frame"]
        st = stage_at(i0, on)
        cols.append(("\n".join(names), SCOL[st], ts[i0], crops[i0], recs[i0]))
    fig_panels(cols, title, d / "stages_panels.png", args)
    if on["s3"] >= 0 and on["s4"] >= 0:
        a, b = max(on["s3"] - 1, 0), min(on["s4"] + 1, nt - 1)
        idx = list(range(a, b + 1))
        if len(idx) > args.closeup_max:
            sel = np.unique(np.round(np.linspace(0, len(idx) - 1, args.closeup_max)).astype(int))
            keep = set([idx[s] for s in sel] + [on["s3"], on["s4"], a, b])
            idx = sorted(keep)[:args.closeup_max + 2]
        fig_closeup(idx, ts, crops, recs, on, title, d / "stages_closeup.png", args)
    fig_series(ts, S, on, title, d / "stages_series.png", args)
    made = make_anim(ts, crops, recs, on, title, d / "stages_anim", args) if args.anim else ["animation off"]
    f = lambda i: ("%.4g" % ts[i]) if i >= 0 else "never"                 # noqa: E731
    msg = ("%s mu %.3f core %d: stage 1 %s, stage 2 %s, cos(theta) cells %s, width-0.4 cell inside the loop %s, stage 3 %s, "
           "stage 4 %s; %s" % (stem, mu, j, f(on["s1"]), f(on["s2"]), f(on["cos"]), f(on["in04"]), f(on["s3"]), f(on["s4"]),
                               "; ".join(made)))
    # the parity check on the real movie: reversals mod 2 against width-0.4 cells inside mod 2
    par = float(np.mean((S["flips"] % 2) == (S["n04_in"] % 2)))
    msg += "; reversal parity equals the parity of width-0.4 cells inside the loop in %.0f%% of frames" % (100 * par)
    print(msg); lines.append(msg)


# ---------------------------------------------------------------------
# across movies: the timing diagram and the redrawn jump grid
# ---------------------------------------------------------------------

def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fl(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def timing_and_grid(dirs, out, args, lines):
    allcores, events = [], []
    for dname in dirs:
        dd = Path(dname)
        evp = dd / "core_events.csv"
        if not evp.exists():
            lines.append("LOUD: %s missing" % evp); continue
        ev = read_csv(evp)
        events += ev
        meta = {}
        for e in ev:
            meta[e["stem"]] = (fl(e["mu"]), fl(e["dt"]))
        for stem, (mu, dt) in meta.items():
            sp = dd / stem / "core_series.csv"
            if not sp.exists():
                continue
            rows = read_csv(sp)
            for c in sorted(set(int(r["core"]) for r in rows)):
                rr = [r for r in rows if int(r["core"]) == c]
                t = np.array([fl(r["t"]) for r in rr])
                g = lambda k: np.array([fl(r[k]) for r in rr])            # noqa: E731

                def first(cond):
                    h = np.where(cond)[0]
                    return float(t[h[0]]) if len(h) else float("nan")
                allcores.append(dict(stem=stem, mu=mu, dt=dt, core=c,
                                     s1=first(g("nV_u") + g("nX_u") > 0), s2=first(g("fold_loc_phi") > args.fold_on),
                                     cos=first(g("n_cells_cos") > 0), s3=first(g("flips") >= 1),
                                     s4=first((g("closed") > 0) & (g("G_carried") >= args.turn_hi))))
    if not allcores:
        return
    P.write_rows(allcores, out / "stages_timing_table")
    # timing diagram
    fig, axs = plt.subplots(2, 1, figsize=(8.0, 8.6), constrained_layout=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    ax = axs[0]
    mus = sorted(set((r["mu"], r["dt"]) for r in allcores))
    for (mu, dt) in mus:
        R = [r for r in allcores if r["mu"] == mu and r["dt"] == dt]
        fine = dt < 0.1
        for n, r in enumerate(R):
            xj = mu + (n - (len(R) - 1) / 2.0) * 0.0018 + (0.004 if fine else 0.0)
            vals = [r[k] for k in ["s1", "s2", "s3", "s4"] if np.isfinite(r[k])]
            if vals:
                ax.plot([xj, xj], [min(vals), max(vals)], "-", color="0.75", lw=0.8, zorder=1)
            for k, m in [("s1", "o"), ("s2", "s"), ("s3", "v"), ("s4", "*")]:
                if np.isfinite(r[k]):
                    ax.plot(xj, r[k], m, ms=7 if k == "s4" else 4.5, color=SCOL[k], mfc=SCOL[k] if not fine else "w",
                            mew=1.0, zorder=3)
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_ylabel("time the stage begins at each core (symmetric log axis)", fontsize=8)
    hs = [plt.Line2D([], [], marker=m, color=SCOL[k], ls="none", ms=7 if k == "s4" else 5)
          for k, m in [("s1", "o"), ("s2", "s"), ("s3", "v"), ("s4", "*")]]
    hs += [plt.Line2D([], [], marker="o", color="0.3", mfc="0.3", ls="none"), plt.Line2D([], [], marker="o", color="0.3", mfc="w", ls="none")]
    ax.legend(hs, ["stage 1: amplitude texture", "stage 2: fold", "stage 3: first reversal (phase rearranges)",
                   "stage 4: one turn of circulation", "coarse cadence (dt 0.2)", "fine cadence (dt 0.025), shifted right"],
              fontsize=7, loc="upper right")
    ax.set_title("When each stage begins, core by core (grey line: from a core's first stage to its last)", fontsize=9, loc="left")
    ax = axs[1]
    for (mu, dt) in mus:
        R = [r for r in allcores if r["mu"] == mu and r["dt"] == dt and np.isfinite(r["s3"]) and np.isfinite(r["s4"])]
        fine = dt < 0.1
        for n, r in enumerate(R):
            xj = mu + (n - (len(R) - 1) / 2.0) * 0.0018 + (0.004 if fine else 0.0)
            ax.plot(xj, max(r["s4"] - r["s3"], 1e-3), "o", ms=4.5, color=SCOL["s3"], mfc=SCOL["s3"] if not fine else "w")
    ax.set_yscale("log")
    ax.set_xlabel("mu = sin(alpha)", fontsize=8)
    ax.set_ylabel("open stretch of stage 3\n(one turn minus first reversal)", fontsize=8)
    for a in axs:
        a.tick_params(labelsize=7.5)
    fig.savefig(out / "stages_timing.png", dpi=170); fig.savefig(out / "stages_timing.pdf"); plt.close(fig)
    # redrawn jump grid
    q = [k for k, _, _ in C.QUANT if k not in ("G_plain", "G_carried", "flips")]
    lab = {k: l for k, l, _ in C.QUANT}
    grp = {k: g for k, _, g in C.QUANT}
    stems = sorted(set(e["stem"] for e in events if np.isfinite(fl(e.get("t_event", "nan")))),
                   key=lambda s: [fl(e["mu"]) + (0.001 if fl(e["dt"]) < 0.1 else 0) for e in events if e["stem"] == s][0])
    M = np.full((len(q), len(stems)), np.nan)
    plain = np.full(len(stems), np.nan)
    revtxt = []
    hdr = []
    for j, s in enumerate(stems):
        E = [e for e in events if e["stem"] == s and np.isfinite(fl(e.get("t_event", "nan")))]
        hdr.append("mu %.2f\ndt %.3g\n%d cores" % (fl(E[0]["mu"]), fl(E[0]["dt"]), len(E)))
        for i, k in enumerate(q):
            v = [fl(e.get(k + "_score", "nan")) for e in E]
            v = [a for a in v if np.isfinite(a)]
            if v:
                M[i, j] = np.median(v)
        dv = [abs(fl(e["G_plain_after"]) - fl(e["G_plain_before"])) for e in E]
        plain[j] = np.median(dv)
        fb = [(int(round(fl(e["flips_before"]))), int(round(fl(e["flips_after"])))) for e in E]
        n12 = sum(1 for a, b in fb if (a, b) == (1, 2))
        revtxt.append("1 to 2\nat %d of %d" % (n12, len(fb)))
    fig, ax = plt.subplots(figsize=(2.2 + 1.1 * len(stems), 0.42 * (len(q) + 2) + 2.0), constrained_layout=True)
    im = ax.imshow(np.log10(np.clip(M, 0.1, 100)), cmap="magma_r", vmin=-1, vmax=2, aspect="auto")
    for i in range(len(q)):
        for j in range(len(stems)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, ("%.0f" % M[i, j]) if M[i, j] >= 10 else ("%.1f" % M[i, j]), ha="center", va="center",
                        fontsize=7, color="w" if M[i, j] > 8 else "k")
    for j in range(len(stems)):
        ax.text(j, len(q), revtxt[j], ha="center", va="center", fontsize=7, color="k", fontweight="bold")
        ax.text(j, len(q) + 1, "%.2f" % plain[j], ha="center", va="center", fontsize=7, color="0.35")
    ax.set_ylim(len(q) + 1.5, -0.5)
    ax.set_yticks(list(range(len(q))) + [len(q), len(q) + 1])
    ax.set_yticklabels([lab[k] for k in q] + ["reversals of k on the loop:\nbefore to after the event",
                                              "plain line integral of k:\nabsolute change at the event (turns)"], fontsize=7)
    for tl, k in zip(ax.get_yticklabels(), q + ["flips", "plain"]):
        tl.set_color(C.GROUP_COLOR[grp[k]] if k in grp and k != "flips" else ("k" if k == "flips" else "0.35"))
    ax.set_xticks(range(len(stems))); ax.set_xticklabels(hdr, fontsize=7)
    cb = plt.colorbar(im, ax=ax, shrink=0.6)
    cb.set_ticks([-1, 0, 1, 2]); cb.set_ticklabels(["0.1", "1", "10", "100"])
    cb.set_label("median jump score at the core's event\n(change in the event frame / typical change per frame)", fontsize=7)
    ax.set_title("What changes, core by core, in the frame where the circulation reaches one turn", fontsize=8.5)
    fig.savefig(out / "stages_jump_grid.png", dpi=170); fig.savefig(out / "stages_jump_grid.pdf"); plt.close(fig)
    lines.append("timing diagram: %d cores from %s" % (len(allcores), ", ".join(dirs)))


def main(args):
    t0 = time.time()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    lines = [VERSION, "args: " + json.dumps(vars(args)), ""]
    files = F.select_files(args.npz, args.mus) if args.npz else []
    excl = [s for s in args.exclude.split(",") if s]
    files = [f for f in files if not any(s in Path(f).name for s in excl)]
    if args.npz and not files:
        lines.append("LOUD: no movie matched --npz %s --mus %s" % (args.npz, args.mus)); print(lines[-1])
    for p in files:
        run_movie(p, args, out, lines)
    if args.core_track_dirs:
        timing_and_grid([s for s in args.core_track_dirs.split(",") if s], out, args, lines)
    lines += ["", "done in %.0f s" % (time.time() - t0)]
    (out / "stages_notes.txt").write_text("\n".join(lines) + "\n")
    print("done in %.0f s; outputs in %s" % (time.time() - t0, out))
    return 0


class _Args:
    """defaults; mirrored by the CLI below. Keys shared with core_track_time_series keep its values."""
    npz = ""                       # movie files and/or folders
    mus = ""                       # mu values to keep ("" keeps every file)
    exclude = "last10,f4-84"
    outdir = str(Path("results") / "nucleation_stages_figures_v11")
    core = -1                      # core to draw (-1: the first to reach one turn)
    core_track_dirs = ""           # comma-separated core_track results folders, for the timing diagram and jump grid
    fold_on = 0.05                 # stage 2: folded share of the line near the core above this
    half_x_L = 1.0                 # crop half-width along the line, L
    half_y_L = 0.8                 # crop half-height, L
    A_lo = 0.3
    A_hi = 1.15
    n_arrows = 16                  # arrows across the crop in the k panel
    arrow_scale = 26.0
    closeup_max = 7                # at most this many columns in the close-up
    anim = 1
    anim_stride = 1
    anim_dpi = 90
    fps = 5
    hold_frames = 6                # the animation repeats each stage-onset frame this many extra times
    mp4 = 1                        # also write an mp4 when ffmpeg is available
    # shared with core_track_time_series (same values)
    turn_hi = 0.9
    win_x_L = 0.5
    win_y_L = 0.75
    low_frac = 0.7
    mu = 0.85
    sigma_dir = 0.4
    taper_frac = 0.06
    sigma_det = 0.3
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
        p.add_argument("--" + k, type=type(v), default=v)
    sys.exit(main(p.parse_args()))
