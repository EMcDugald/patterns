#!/usr/bin/env python
"""
cells_motion_anim.py  v1.3 (29 Oct 2026)

v1.3: the three-panel movie (u, cos(theta), colour of k with the fold bars, flips and arrows; no space-time plot)
as the default animation (--anim_three); the amplitude as a third set of stills (cm_stills_A, cells of u drawn on it,
counter on u only); the one-field and side-by-side animations behind --anim_single and --side.
v1.2: the red fold bar is drawn per contiguous folded run (v1.1 bridged the gap between two folded stretches with
one bar, which made the unfolded core between them look folded); yellow arrows along every run.
v1.1: the counter is drawn on the u panels only (the loop readout is of k of u).

The motion of the V and X cells while a knee becomes a dislocation, drawn as simply as possible, for the talk:
the pattern faint in grey, large V and X markers with short fading trails, the L/2 loop about one core, and a
counter that reads the number of cells inside the loop (even or odd) and the circulation (a number when the sign
of k closes round the loop, "undefined" when it does not). One animation for the pattern u and one for the phase
alone cos(theta), the same frames and the same layout, so the two can be compared; a side-by-side version; stills
at the key frames; and, for the same frames, the colour map of the direction of k with arrows drawn at the places
that matter (a sparse grid, both sides of each flip on the loop, and above and below the folded stretch of the
line), so the colours can be read against actual arrows.

Everything is computed by nucleation_migration v1.3 (imported, unchanged): mg at width 0.4, the cells of u and of
cos(theta) with the same detector at the same width, the loop readout, the fold. Cell positions therefore agree
with mig_cells.csv of the migration runs. Default exemplar: the first interior core that reaches one turn (the
rule of nucleation_migration), or --core.

Key frames (stills and the arrowed colour maps): first frame; birth (first cells of u near the core); mid-walk
(the walking X first past the midpoint of its travel); the frame the loop's count first goes odd (circulation
undefined); the frame the circulation first reads one turn; last frame.

Outputs in <outdir>/<stem>/ (every name begins with cm_):
  cm_anim_three.mp4 / .gif                    u | cos(theta) | colour of k with arrows, with the counter (the talk's movie)
  cm_anim_u, cm_anim_cos, cm_anim_side        with --anim_single 1 / --side 1   (mp4 needs ffmpeg; gif always)
  cm_stills_u.png, cm_stills_A.png, cm_stills_cos.png   the key frames in a row, one field each
  cm_still_<field>_f<frame>.png               each key frame alone (for a slide of one's own)
  cm_karrows_f<frame>.png                     colour of k with arrows, at the key frames
  cm_counts.csv                               per frame: cells inside the loop, flips, circulation, defined, X position
  cm_notes.txt
Run from ~/patterns/pipelines (CELLS_MOTION_CLI_1029.txt). Python 3.8 compatible.
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
import nucleation_migration as M                                            # noqa: E402
F, P = M.F, M.P
import matplotlib                                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402
from matplotlib import animation                                           # noqa: E402
from matplotlib.patches import Circle                                      # noqa: E402
from matplotlib.lines import Line2D                                        # noqa: E402

VERSION = "cells_motion_anim v1.3 (29 Oct 2026, on %s)" % M.VERSION
CX, CV = "#d62728", "#1f77b4"


# ---------------------------------------------------------------------
# per-frame data for the exemplar core
# ---------------------------------------------------------------------

def frame_pack(D, x, y, xc, args):
    """crop about the core plus the loop readout and the cells, in units of L about (xc, y_row)"""
    G, e, L = D["G"], D["e"], D["G"]["L"]
    y_row = G["y_row"]
    rec = M.core_record(D, x, y, xc, args)
    cr = M.crop_frame(D, x, y, xc, args)
    out = dict(rec=rec, u=cr["u"], A=cr["A"], cos=cr["cos"], ang=cr["ang"], xs=cr["xs"], ys=cr["ys"], cu=cr["cu"],
               cc=cr["cc"], rev=cr["rev"], fold=cr["fold"], k1=None, k2=None)
    ix = np.abs(x - xc) <= args.half_x_L * L
    jy = np.abs(y - y_row) <= args.half_y_L * L
    out["k1"] = e["k1"][np.ix_(jy, ix)]
    out["k2"] = e["k2"][np.ix_(jy, ix)]
    # the walking X: on-line X of u furthest along +x within (xt_lo, xt_hi)
    px, py, pq = cr["cu"]
    m = (pq > 0) & (np.abs(py) <= args.on_line_L) & (px >= args.xt_lo) & (px < args.xt_hi)
    out["xwalk"] = float(px[m].max()) if m.any() else float("nan")
    return out


def key_frames(packs, ts, args):
    n = len(packs)
    nin = np.array([p["rec"]["n_in"] for p in packs])
    defd = np.array([p["rec"]["defined"] for p in packs])
    G = np.array([p["rec"]["G_carried"] for p in packs])
    ncell = np.array([p["rec"]["nX_u"] + p["rec"]["nV_u"] for p in packs])
    xw = np.array([p["xwalk"] for p in packs])

    def first(c):
        h = np.where(np.asarray(c, bool))[0]
        return int(h[0]) if len(h) else -1
    birth = first(ncell > 0)
    odd = first((nin % 2) == 1)
    one = first((defd > 0) & (G >= args.turn_hi))
    mid = -1
    ok = np.isfinite(xw)
    if ok.any():
        x0, x1 = xw[ok][0], float(np.nanmedian(xw[ok][-5:]))
        h = np.where(ok & (xw >= 0.5 * (x0 + x1)) & (np.arange(n) > first(ok)))[0]
        mid = int(h[0]) if len(h) else -1
    keys = [(0, "first frame")]
    for i, name in ((birth, "birth of the pair"), (mid, "the X walks"), (odd, "odd count: undefined"),
                    (one, "one turn"), (n - 1, "last frame")):
        if i >= 0 and i not in [k[0] for k in keys]:
            keys.append((i, name))
    return sorted(keys), dict(birth=birth, mid=mid, odd=odd, one=one)


# ---------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------

def counter_text(rec):
    n = int(rec["n_in"])
    par = "even" if n % 2 == 0 else "odd"
    if rec["defined"] > 0:
        circ = "%.2f" % rec["G_carried"]
    else:
        circ = "undefined"
    return "cells inside the loop: %d (%s)\ncirculation: %s" % (n, par, circ)


def draw_field(ax, packs, i, field, args, trail=True, title=None, show_counter=None):
    """show_counter: the loop readout comes from k of u, so by default the counter is drawn on the u panels only
    (v1.1; v1.0 drew it on the cos(theta) panels too, where it did not describe the cells shown)"""
    if show_counter is None:
        show_counter = (field == "u")
    p = packs[i]
    ext = [p["xs"][0], p["xs"][-1], p["ys"][0], p["ys"][-1]]
    if field == "k":                                                      # the colour of k with the arrows (v1.3)
        draw_karrows(ax, p, args, small=True)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1]); ax.set_yticks([-0.5, 0, 0.5]); ax.tick_params(labelsize=8)
        if title:
            ax.set_title(title, fontsize=10)
        return
    if field == "A":                                                      # the amplitude, A / A_far (v1.3)
        ax.imshow(p["A"], origin="lower", extent=ext, cmap="viridis", vmin=args.A_lo, vmax=args.A_hi, aspect="equal",
                  interpolation="bilinear")
    else:
        ax.imshow(p[field], origin="lower", extent=ext, cmap="gray", vmin=-args.vmax, vmax=args.vmax, alpha=args.bg_alpha,
                  aspect="equal", interpolation="bilinear")
    ax.axhline(0, color="0.55", lw=0.8, ls="--")
    ax.add_patch(Circle((0, 0), args.loop_L, fill=False, color="#f39c12", lw=2.0))
    key = "cc" if field == "cos" else "cu"                                # the amplitude panel carries u's cells
    if trail:
        for j in range(max(0, i - args.trail), i):
            a = 0.12 + 0.5 * (j - (i - args.trail)) / max(args.trail, 1)
            px, py, pq = packs[j][key]
            for sel, mk, col in ((pq > 0, "^", CX), (pq < 0, "v", CV)):
                ax.plot(px[sel], py[sel], mk, ms=args.ms * 0.7, mfc=col, mec="none", alpha=a, ls="none")
    px, py, pq = p[key]
    for sel, mk, col in ((pq > 0, "^", CX), (pq < 0, "v", CV)):
        ax.plot(px[sel], py[sel], mk, ms=args.ms, mfc=col, mec="white", mew=1.0, ls="none", zorder=5)
    if show_counter:
        ax.text(0.02, 0.97, counter_text(p["rec"]), transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9), zorder=6)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([-1, -0.5, 0, 0.5, 1]); ax.set_yticks([-0.5, 0, 0.5])
    ax.tick_params(labelsize=8)
    if title:
        ax.set_title(title, fontsize=10)


FIELD_TITLE = {"u": "the pattern $u$ and its cells", "cos": r"the phase alone $\cos\theta$ and its cells",
               "A": r"the amplitude $A/A_{\rm far}$ (cells of $u$)", "k": "direction of $k$, fold, flips"}


def legend_handles():
    return [Line2D([], [], marker="^", ls="none", mfc=CX, mec="white", ms=9, label="X cell (+1/2 turn)"),
            Line2D([], [], marker="v", ls="none", mfc=CV, mec="white", ms=9, label="V cell (-1/2 turn)"),
            Line2D([], [], color="#f39c12", lw=2, label="loop of radius L/2"),
            Line2D([], [], color="0.55", ls="--", label="the boundary line")]


def make_anim(packs, ts, fields, title, base, args):
    nf = len(fields)
    with_k = "k" in fields
    fig = plt.figure(figsize=(5.4 * nf + (1.6 if with_k else 0), 4.9))
    right = 0.84 if with_k else 0.98
    gs = fig.add_gridspec(1, nf, left=0.05, right=right, top=0.84, bottom=0.17, wspace=0.12)
    axs = [fig.add_subplot(gs[0, j]) for j in range(nf)]
    hand = legend_handles() + (karrows_handles()[2:] if with_k else [])
    fig.legend(handles=hand, loc="lower center", ncol=4 if not with_k else 5, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5 * right, 0.0))
    if with_k:
        M.draw_wheel(fig, [0.86, 0.40, 0.12, 0.28])
    sup = fig.suptitle(title, fontsize=10)

    def draw(i):
        for ax, f in zip(axs, fields):
            ax.cla()
            draw_field(ax, packs, i, f, args, title=FIELD_TITLE[f])
            ax.set_xlabel("along the line, from the core (L)", fontsize=8)
        axs[0].set_ylabel("across the line (L)", fontsize=8)
        for ax in axs[1:]:
            ax.set_yticklabels([])
        sup.set_text("%s    t = %.4g" % (title, ts[i]))
        return []

    seq = list(range(0, len(ts), max(1, args.stride)))
    made = []
    ani = animation.FuncAnimation(fig, draw, frames=seq, blit=False)
    if args.mp4:
        try:
            ani.save(str(base.with_suffix(".mp4")), writer=animation.FFMpegWriter(fps=args.fps, bitrate=2400), dpi=args.dpi)
            made.append(base.with_suffix(".mp4").name)
        except Exception as exc:
            made.append("LOUD: mp4 failed (%s); gif only" % exc)
    try:
        ani.save(str(base.with_suffix(".gif")), writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
        made.append(base.with_suffix(".gif").name)
    except Exception as exc:
        made.append("LOUD: gif failed (%s)" % exc)
    plt.close(fig)
    return made


def fig_stills(packs, ts, keys, field, title, path, args):
    n = len(keys)
    fig, axs = plt.subplots(1, n, figsize=(3.3 * n, 3.6), squeeze=False, constrained_layout=True)
    for ax, (i, name) in zip(axs[0], keys):
        draw_field(ax, packs, i, field, args, trail=False, title="%s\nt = %.4g" % (name, ts[i]))
        ax.set_xlabel("along the line (L)", fontsize=8)
    axs[0, 0].set_ylabel("across (L)", fontsize=8)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("%s: %s" % (title, FIELD_TITLE[field]), fontsize=10)
    fig.savefig(path.with_suffix(".png"), dpi=170, bbox_inches="tight"); fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    for i, name in keys:
        f2, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
        draw_field(ax, packs, i, field, args, trail=False, title="%s, t = %.4g" % (name, ts[i]))
        ax.set_xlabel("along the line, from the core (L)", fontsize=9); ax.set_ylabel("across the line (L)", fontsize=9)
        f2.savefig(path.parent / ("cm_still_%s_f%03d.png" % (field, i)), dpi=170); plt.close(f2)


def draw_karrows(ax, p, args, small=False):
    """the colour of k, with arrows: a sparse grid, both sides of each flip on the loop, above and below each
    folded run (one red bar per contiguous run); the cells of u on top"""
    xs, ys = p["xs"], p["ys"]
    ext = [xs[0], xs[-1], ys[0], ys[-1]]
    ax.imshow(p["ang"], origin="lower", extent=ext, cmap="twilight", vmin=-np.pi, vmax=np.pi, aspect="equal",
              interpolation="nearest")
    ax.axhline(0, color="0.3", lw=0.8, ls="--")
    ax.add_patch(Circle((0, 0), args.loop_L, fill=False, color="#f39c12", lw=2.0))
    k1, k2 = p["k1"], p["k2"]
    X, Y = np.meshgrid(xs, ys)
    dxs = xs[1] - xs[0]
    step = max(1, int(round(args.arrow_grid_L / dxs)))
    sl = (slice(step // 2, None, step), slice(step // 2, None, step))
    nrm = np.hypot(k1, k2) + 1e-12
    sc = 1.25 if small else 1.0
    ax.quiver(X[sl], Y[sl], (k1 / nrm)[sl], (k2 / nrm)[sl], color="white", alpha=0.55, scale=22 * sc, width=0.004,
              headwidth=4, pivot="mid")
    L_off = args.fold_off_L
    fx = np.sort(p["fold"])
    if len(fx):
        brk = np.where(np.diff(fx) > 1.5 * dxs)[0]
        starts = np.r_[fx[0], fx[brk + 1]]; ends = np.r_[fx[brk], fx[-1]]
        for s0, s1 in zip(starts, ends):
            ax.plot([s0 - 0.5 * dxs, s1 + 0.5 * dxs], [0, 0], color="#e74c3c", lw=5, alpha=0.9, solid_capstyle="butt", zorder=7)
        every = max(1, int(round(args.arrow_grid_L / dxs)))
        for s0, s1 in zip(starts, ends):
            run = fx[(fx >= s0) & (fx <= s1)]
            for xf in run[::every]:
                for yy in (+L_off, -L_off):
                    i = int(np.argmin(np.abs(xs - xf))); j = int(np.argmin(np.abs(ys - yy)))
                    ax.quiver([xf], [yy], [k1[j, i] / nrm[j, i]], [k2[j, i] / nrm[j, i]], color="#ffd166", scale=20 * sc,
                              width=0.006, headwidth=4, pivot="tail", zorder=6)
    rx, ry = p["rev"]
    for xr, yr in zip(rx, ry):
        a0 = np.arctan2(yr, xr)
        for da in (-args.flip_sep, +args.flip_sep):
            xq, yq = args.loop_L * np.cos(a0 + da), args.loop_L * np.sin(a0 + da)
            i = int(np.argmin(np.abs(xs - xq))); j = int(np.argmin(np.abs(ys - yq)))
            ax.quiver([xq], [yq], [k1[j, i] / nrm[j, i]], [k2[j, i] / nrm[j, i]], color="black", scale=14 * sc,
                      width=0.008, headwidth=4, pivot="tail", zorder=7)
        ax.plot(xr, yr, "X", color="white", mec="black", ms=11 if not small else 9, mew=1.2, zorder=8)
    M._cells(ax, p["cu"], size=8 if not small else 7)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])


def karrows_handles():
    return [Line2D([], [], marker="^", ls="none", mfc=CX, mec="white", ms=8, label="X cell of u"),
            Line2D([], [], marker="v", ls="none", mfc=CV, mec="white", ms=8, label="V cell of u"),
            Line2D([], [], marker="X", ls="none", mfc="white", mec="black", ms=9, label="flip of k on the loop"),
            Line2D([], [], color="#e74c3c", lw=4, label="folded stretch of the line"),
            Line2D([], [], color="#ffd166", lw=3, label="yellow arrows: k just above and below the fold"),
            Line2D([], [], color="black", lw=3, label="black arrows: k either side of a flip")]


def fig_karrows(p, t, name, path, args):
    fig = plt.figure(figsize=(8.6, 4.6))
    ax = fig.add_axes([0.07, 0.12, 0.66, 0.78])
    draw_karrows(ax, p, args)
    ax.set_xlabel("along the line, from the core (L)", fontsize=9); ax.set_ylabel("across the line (L)", fontsize=9)
    ax.set_title("%s, t = %.4g.  Colour: direction of k. Black arrows: either side of a flip on the loop.\n"
                 "Yellow arrows: just above and just below the folded stretch. Grey: a sparse sample of k." % (name, t),
                 fontsize=8.5)
    M.draw_wheel(fig, [0.78, 0.42, 0.18, 0.34])
    fig.legend(handles=karrows_handles()[:4], loc="lower right", fontsize=7, frameon=False, bbox_to_anchor=(0.99, 0.08))
    fig.savefig(path, dpi=170); plt.close(fig)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def run_movie(path, args, out, lines):
    u, x, y, t, md = F.movie(path)
    mu = float(md.get("mu", F.mu_of(path)))
    jr = int(md.get("gb_row_in_crop", u.shape[0] // 2)); y_row = float(y[jr])
    nt = u.shape[-1]
    ts = np.array([float(t[min(i, t.size - 1)]) for i in range(nt)])
    stem = Path(path).stem
    d = out / stem; d.mkdir(parents=True, exist_ok=True)
    DL = M.frame_fields(np.asarray(u[..., -1], float), x, y, y_row, mu, args)
    G0 = DL["G"]; L = G0["L"]
    xd, _, _, _, _ = P.find_dip_patches(DL["A"], x, y, G0["win"], args.core_frac, 0.5 * L)
    centres = F.centres_from(xd, L, G0["xm"], G0["Lx"])
    if not len(centres):
        lines.append("LOUD: %s: no cores in the last frame" % stem); return
    edges = [bool(abs(c - G0["xm"]) + args.half_x_L * L > args.edge_frac * G0["Lx"]) for c in centres]
    # exemplar: --core, else the first interior core to reach one turn (needs the loop readout at every core)
    if args.core >= 0:
        jx = min(args.core, len(centres) - 1)
        packs = None
    else:
        allp = [[] for _ in centres]
        for i in range(nt):
            D = M.frame_fields(np.asarray(u[..., i], float), x, y, y_row, mu, args)
            D["u2"] = np.asarray(u[..., i], float)
            for j, xc in enumerate(centres):
                allp[j].append(frame_pack(D, x, y, xc, args))
        hits = []
        for j in range(len(centres)):
            _, on = key_frames(allp[j], ts, args)
            hits.append(ts[on["one"]] if (on["one"] >= 0 and not edges[j]) else np.inf)
        jx = int(np.argmin(hits))
        if not np.isfinite(hits[jx]):
            inner = [j for j in range(len(centres)) if not edges[j]]
            jx = inner[len(inner) // 2] if inner else 0
            lines.append("%s: no interior core reaches one turn; exemplar is core %d" % (stem, jx))
        packs = allp[jx]
    if packs is None:
        packs = []
        for i in range(nt):
            D = M.frame_fields(np.asarray(u[..., i], float), x, y, y_row, mu, args)
            D["u2"] = np.asarray(u[..., i], float)
            packs.append(frame_pack(D, x, y, centres[jx], args))
    del u
    keys, on = key_frames(packs, ts, args)
    title = "%s   mu %.3f (%.0f deg), core %d" % (stem, mu, np.degrees(np.arcsin(mu)), jx)
    made = []
    if args.anim_single:
        made += make_anim(packs, ts, ["u"], title, d / "cm_anim_u", args)
        made += make_anim(packs, ts, ["cos"], title, d / "cm_anim_cos", args)
    if args.side:
        made += make_anim(packs, ts, ["u", "cos"], title, d / "cm_anim_side", args)
    if args.anim_three:
        made += make_anim(packs, ts, ["u", "cos", "k"], title, d / "cm_anim_three", args)
    fig_stills(packs, ts, keys, "u", title, d / "cm_stills_u", args)
    fig_stills(packs, ts, keys, "A", title, d / "cm_stills_A", args)
    fig_stills(packs, ts, keys, "cos", title, d / "cm_stills_cos", args)
    for i, name in keys:
        fig_karrows(packs[i], ts[i], name, d / ("cm_karrows_f%03d.png" % i), args)
    with open(d / "cm_counts.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frame", "t", "n_in", "flips", "defined", "G_carried", "nX_u", "nV_u", "nX_cos", "nV_cos", "x_walk_L"])
        for i, p in enumerate(packs):
            r = p["rec"]
            w.writerow([i, ts[i], r["n_in"], r["flips"], r["defined"], r["G_carried"], r["nX_u"], r["nV_u"], r["nX_cos"], r["nV_cos"], p["xwalk"]])
    fmt = lambda i: ("%.4g" % ts[i]) if i >= 0 else "never"                 # noqa: E731
    msg = ("%s mu %.3f core %d (edge: %s): birth %s, mid-walk %s, odd count %s, one turn %s; key frames %s; %s"
           % (stem, mu, jx, edges[jx], fmt(on["birth"]), fmt(on["mid"]), fmt(on["odd"]), fmt(on["one"]),
              ", ".join("%d (%s)" % k for k in keys), "; ".join(made)))
    print(msg); lines.append(msg)


def main(args):
    t0 = time.time()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    lines = [VERSION, "args: " + json.dumps(vars(args)), ""]
    files = F.select_files(args.npz, args.mus) if args.npz else []
    excl = [s for s in args.exclude.split(",") if s]
    files = [f for f in files if not any(s in Path(f).name for s in excl)]
    if not files:
        lines.append("LOUD: no movie matched --npz %s --mus %s" % (args.npz, args.mus)); print(lines[-1])
    for p in files:
        run_movie(p, args, out, lines)
    lines += ["", "done in %.0f s" % (time.time() - t0)]
    (out / "cm_notes.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-6:])); print("outputs in %s" % out)
    return 0


class _Args:
    """defaults; mirrored by the CLI below. Keys shared with nucleation_migration keep its values."""
    npz = ""
    mus = ""
    exclude = "last10,f4-84"
    outdir = str(Path("results") / "cells_motion_anim")
    core = -1                      # exemplar core (-1: the first interior core to reach one turn)
    half_x_L = 1.0                 # crop half-width along the line (L); shows the neighbouring cores' Xs arriving
    half_y_L = 0.6
    vmax = 1.1                     # grey scale of the pattern
    bg_alpha = 0.55
    ms = 11.0                      # marker size of the cells
    trail = 12                     # frames of fading trail
    stride = 1
    fps = 8
    dpi = 140
    mp4 = 1
    side = 0                       # the u | cos(theta) side-by-side animation
    anim_single = 0                # the one-field animations
    anim_three = 1                 # u | cos(theta) | colour of k with arrows (v1.3): the talk's movie
    A_lo = 0.3                     # amplitude colour scale (A / A_far)
    A_hi = 1.15
    arrow_grid_L = 0.12            # spacing of the sparse arrow grid on the colour map (L)
    flip_sep = 0.12                # angular offset (rad) of the two arrows drawn either side of a flip
    xt_lo = 0.0
    xt_hi = 0.95
    # shared with nucleation_migration (same values)
    width = 0.4
    with_phi = 0
    fold_on = 0.05
    band_y_L = 0.5
    edge_frac = 0.4
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
        p.add_argument("--" + k, type=type(v), default=v)
    sys.exit(main(p.parse_args()))
