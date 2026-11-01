#!/usr/bin/env python
"""
nucleation_migration.py  v1.3 (28 Oct 2026)

What happens when a knee destabilizes into a dislocation, told with ONE stripe-direction field: mg's own director
at width 0.4 (the estimator's sigma_dir). Successor to nucleation_stages_figures_v11.py (kept unchanged), built on
the same per-frame machinery (core_track_time_series.py, fold_time_series.py, phase_vs_amplitude_charge_v11.py,
imported, edited nowhere). Scope: dislocation angles only, mu >= 0.82 (Ed's decision of 28 Oct).

Plain terms. A "half-turn cell" is a tiny square of four grid points around which the stripe direction (no
arrowhead) turns by half a turn: X (+1/2) or V (-1/2). Here every cell, in u, in cos(theta) and inside the loop,
is found with the same detector at the same width, 0.4, which is the director mg builds k from. So the cells of u
drawn in every panel are exactly the cells that set the flips of k on the loop (the parity rule, checked at every
core). The fold is read from mg's k: a column of the line is folded where k 0.05 L above the line points against k
0.05 L below (the stripes run along the line there instead of crossing it). The smooth phase phi_LS is not used in
any figure (--with_phi 1 adds its fold as a comparison column in the CSV only).

The stages at a core (the same five as before, at one width):
  0  the knee: no cells of u near the core
  1  weak spots in the stripes' brightness: cells of u appear near the core (|x - x_core| < win_x_L L,
     |y - y_line| < win_y_L L); amplitude-first holds at a core when cells of cos(theta) appear later or never
  2  the fold: the folded share of the line within win_x_L L of the core exceeds fold_on (mg's k)
  3  the first crossing: one flip of k on the loop of radius L/2 about the core (circulation undefined)
  4  the second crossing: two flips and |circulation| >= turn_hi (one turn)
Stages 3 and 4 are readouts of that loop (the frames when X cells of u reach the two points where the loop meets
the line; the flips follow those cells, not the fold), not events in the pattern; the space-time diagram shows where
the cells are. The order of stages 1 and 2 depends on the detector width and on the field the fold is read from, so
the figures label them together ("stages 1-2").

Measured per frame and core (mig_core_series.csv): circulation (carried) and whether it is defined, flips, plain
line integral, cells of u and of cos(theta) near the core (counts by sign), cells of u inside the loop, nearest V-X
distance among u's cells near the core, the folded share near the core and at the two loop crossings, the
amplitude minimum near the core and A on the line at the core (both / A_far).
Per frame along the line (mig_line.npz): the fold mask, A on the line, every cell of u and of cos(theta) within
band_y_L L of the line.

Outputs, per movie in <outdir>/<stem>/ (every name begins with mig_):
  mig_spacetime.png/.pdf          the exemplar core: position along the line (L, about the core) across, time
                                  down; A on the line in grey, the folded stretch in red, X cells of u as red dots
                                  and V cells as blue dots (open circles: cells of cos(theta)), the loop's two
                                  crossings as dashed lines, the stage onsets as ticks
  mig_spacetime_grid.png/.pdf     the same for every core of the movie, small multiples
  mig_panels_contours.png/.pdf    one column per stage onset (plus first and last frame); rows: u with its cells and
                                  amplitude contours; cos(theta) with its cells; the direction of k (color), the
                                  fold, the flips, the loop
  mig_panels_rows.png/.pdf        the same with the amplitude as its own row (four rows)
  mig_xtrack.png/.pdf             every interior core (exemplar thick): where the X cell of u that travels along
                                  the line is (top: u solid, cos(theta) dashed) and the amplitude minimum near the
                                  core (bottom), against time; first crossings marked
  mig_separation.png/.pdf         (only with --sep_fig 1) the v1.2 figure: nearest V-X distance and amplitude minimum
  mig_anim.gif                    three panels and the space-time diagram with a moving time cursor, a
                                  plain-language banner, pauses at each stage onset
  mig_core_series.csv, mig_line.npz
  mig_cells.csv                   every cell of u and of cos(theta) within half_x_L L along and band_y_L L across the
                                  line, every frame, every core (x, y in L about the core)
  mig_xtrack.csv                  per frame and core: position of the travelling X cell of u and of cos(theta)
Across movies, in <outdir>/: mig_cores.csv (one row per core: onsets, amplitude-first, parity, first V-X pair and
its separation, the travelling X: birth, final position, speed, time it passes the loop radius; exemplar flag),
mig_timing.png/.pdf (stage onsets of every core against mu), mig_notes.txt.

Combine mode (--combine "dirA,dirB", no movie read): reads mig_cores.csv and every mig_xtrack.csv in those output
folders (coarse and fine runs together) and writes, in --outdir:
  mig_timing_all.png/.pdf         stage onsets against mu, one bar per movie and stage (range over interior cores,
                                  median marked), cells of cos(theta) included; coarse filled, fine open; the number
                                  of distinct cases per movie printed (coarse cores repeat every third core, a grid
                                  effect: dx / L = 3/64)
  mig_xtrack_all.png/.pdf         the travelling X against time (log axis), one color per angle, u and cos(theta)
  mig_glide_all.csv, mig_combine_notes.txt

The travelling X. On-line X cells (|y| <= on_line_L L) between xt_lo and xt_hi L from the core; the one furthest
along +x is tracked (Ed's runs of 28 Oct: at every dislocation angle the X born with the first pair moves in +x,
inside the folded stretch, while the V cells stay put; the pattern is periodic in L, so any X in that window
belongs to the same family). Birth: first frame with a tracked X. Final: median of the last five tracked frames.
Speed: distance to the midpoint between birth and final position over the time taken to reach it (L per time unit).
Passing the loop: first frame the X is within half a grid spacing of loop_L or beyond, to compare with stage 3.

Changelog
v1.3 (28 Oct 2026, evening, after Ed's v1.2 run; every v1.2 number unchanged)
  - The travelling X: cell positions saved (mig_cells.csv), the X of u and of cos(theta) tracked along the line
    (mig_xtrack.csv), its birth, final position, speed and loop-passing time added to mig_cores.csv. Read off Ed's
    v1.2 space-time diagrams: 0.83 from about 0.22 L at t = 2 to 0.5 L by t = 8; 0.90 fine from 0.25 L to about 0.6 L
    at t = 3.1; 0.92 to about 0.67 L by t = 3. The first loop crossing is the X passing x = loop_L.
  - mig_xtrack figure replaces mig_separation (kept behind --sep_fig 1): the nearest V-X distance jumps because the
    nearest pair changes identity, while the cells themselves move steadily.
  - Distinct cases: coarse cadence has dx / L = 3/64 exactly, so a core's position on the grid repeats every third
    core, and in Ed's v1.2 table cores 0, 3, 6, 9 (and 1, 4, 7; 2, 5, 8) agree in every column to the last digit.
    The summary now counts distinct cases per angle, and the combined timing figure draws each stage as a range.
  - Combine mode (--combine) for one timing figure and one travelling-X figure over the coarse and fine runs.
  - Per-run timing legend lists only the cadences present.

v1.2 (28 Oct 2026, cosmetic; no measured number changes)
  - Stages 1 and 2 labeled together ("stages 1-2") in banners, column titles and the space-time diagram: their order
    depends on the detector width and on the field the fold is read from (MIGRATION_LOCAL_RESULTS_1028.md).
  - Space-time legend moved above the plot (it covered cells); shorter k-row label (it was cut off); animation at
    110 dpi with larger titles.

v1.1 (28 Oct 2026, after Ed's first run)
  - Cells over the whole line (cells_band) instead of phase_vs_amplitude_charge_v11.cells, which also keeps only
    |x - x_mid| < 0.4 Lx: the outermost cores lost cells there and their flip parity read 6 to 79 percent against
    100 percent at every interior core (50 of 50). Cores whose crop reaches past edge_frac Lx are flagged (edge_core)
    and left out of the summary, the separation figure, the timing figure and the exemplar choice (v1.0 drew edge
    cores as exemplars in four of the six movies).
  - Space-time diagrams on a square-root time axis by default (--time_scale linear|sqrt|symlog, --t_max): at coarse
    cadence the events fall between t = 1 and 10 of a 25-unit run.
  - A count strip beside the space-time diagram and in the animation: cells of u inside the loop, flips of k on the
    loop, amplitude minimum near the core. Measured in v1.0 at interior core 3, mu 0.83: the flips follow the count
    of cells inside the loop (4 to 3 at t = 7.4, back to 4 at 9.4), while both loop crossings already read as folded
    from t = 3.0, so the figure no longer says the flips follow the fold.
  - Results of record from v1.0 at interior cores are unchanged by this version except where an edge core entered.

v1.0 (28 Oct 2026)
  - One width (0.4) for every cell detection and for mg, replacing the 0.3 (cells of u, cos(theta)) and 0.4 (mg's
    director, diamonds) pair of nucleation_stages_figures_v11.py, so the cells drawn are the cells that set the flips.
  - The fold, and stage 2, from mg's k. v1.1 of the stages figures drew the fold of grad(phi_LS) (its red squares
    and its stage 2 onset came from fold_loc_phi); phi_LS now enters only as an optional CSV column (--with_phi 1).
  - New: the space-time diagram (migration of the cells and the fold along the line), its grid over every core, the
    V-X separation figure, and an animation built around the space-time diagram, which fills in as time runs.
  - Sandbox check only (no SH movie in the sandbox): the planted vortex-pair movie (mu 0.88, 36 frames) end to end,
    every figure drawn; flip parity matched the cell parity in 81 percent of frames there, where the amplitude falls
    to zero (the stages v1.0 notes report 76 percent on the same control). Numbers of record come from Ed's run.


Python 3.8 compatible. Run from ~/patterns/pipelines (MIGRATION_V13_CLI_1028.txt).
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
loop_grid = C.loop_grid
import matplotlib                                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib import animation                                          # noqa: E402
from matplotlib.patches import Circle                                     # noqa: E402
from matplotlib.lines import Line2D                                       # noqa: E402

VERSION = "nucleation_migration v1.3 (28 Oct 2026, on %s)" % C.VERSION

STAGES = [  # key, short name, plain-language banner, color
    ("s0", "knee", "The knee: the stripes bend smoothly through the line", "#7f8c8d"),
    ("s1", "weak spots", "Stages 1-2: weak spots in the brightness give cells in u (none in cos(theta)); the stripes start to run along the line (fold)", "#1e8449"),
    ("s2", "fold", "Stages 1-2: weak spots in the brightness give cells in u (none in cos(theta)); the stripes start to run along the line (fold)", "#c0392b"),
    ("s3", "first crossing", "An X cell crosses the loop: one flip of k, the circulation is undefined", "#2471a3"),
    ("s4", "second crossing", "A second X cell crosses: two flips, the circulation reads one turn (a dislocation)", "#6c3483"),
]
SCOL = {k: c for k, _, _, c in STAGES}
SNAME = {k: n for k, n, _, _ in STAGES}
SBANNER = {k: b for k, _, b, _ in STAGES}
CX, CV = "#d62728", "#1f77b4"          # X cells red, V cells blue, everywhere


# ---------------------------------------------------------------------
# per-frame fields, one width
# ---------------------------------------------------------------------

def cells_band(Fld, x, y, G, args):
    """half-turn cells of Fld's director (detect_defects at width args.sigma_det on the tapered field), kept within
    band_L L of the line along the WHOLE line. v1.0 used phase_vs_amplitude_charge_v11.cells, which also keeps only
    the central window |x - x_mid| < 0.4 Lx: cells of the outermost cores were dropped there and their flip parity read
    6 to 79 percent against 100 percent at every other core (Ed's run of 28 Oct). Cores too close to the taper are now
    flagged instead (edge_core) and left out of the summary and of the exemplar choice."""
    L, y_row = G["L"], G["y_row"]
    det = P.t1c.detect_defects(Fld * G["taper"], x, y - y_row, G["Lx"], G["Ly"], sigma_det=args.sigma_det,
                               band_y=args.band_L * L)
    q = det["q"]
    J, I = np.nonzero(q)
    px, py, pq = det["xc"][I], det["yc"][J] + y_row, q[J, I].astype(int)
    keep = np.abs(py - y_row) < args.band_L * L
    return px[keep], py[keep], pq[keep]


def frame_fields(u2, x, y, y_row, mu, args):
    """mg at sigma_dir; cells of u and of cos(theta) with the same detector at the same width; the fold of mg's k"""
    a = argparse.Namespace(**vars(args))
    a.sigma_det = args.width                                    # one width for every cell detection
    G = P.geometry(u2, x, y, y_row, mu, a)
    e = P.estimate(u2, G["Lx"], G["Ly"], args.width, None, G["taper"])
    A = e["A"]
    farm = G["win"] & (np.abs(G["Y"] - y_row) > args.far_from * G["L"])
    A_far = float(np.median(A[farm])) if farm.sum() > 50 else float(np.median(A[G["win"]]))
    jr, ia, ib = F.line_rows(y, G, a)
    allc = np.ones(len(x), bool)
    fku = F.fold_line(e["k1"], e["k2"], ia, ib, allc)
    fphi = None
    if args.with_phi:
        phi, _ = P.smooth_phase(e, x, y, G, a)
        gy, gx = np.gradient(phi, G["dy"], G["dx"])
        fphi = F.fold_line(gx, gy, ia, ib, allc)
    cu = cells_band(u2, x, y, G, a)
    cc = cells_band(np.cos(e["theta"]), x, y, G, a)
    return dict(G=G, e=e, A=A, A_far=A_far, jr=jr, fku=fku, fphi=fphi, cu=cu, cc=cc)


def core_record(D, x, y, xc, args):
    G, e, L = D["G"], D["e"], D["G"]["L"]
    y_row = G["y_row"]
    r = args.loop_L * L
    c, cl = loop_grid(e["k1"], e["k2"], x, y, np.array([xc]), np.array([y_row]), r, args.nang)
    cp, fl = P.loop_plain(e["k1"], e["k2"], x, y, np.array([xc]), np.array([y_row]), r, args.nang)
    W = (np.abs(G["X"] - xc) < args.win_x_L * L) & (np.abs(G["Y"] - y_row) < args.win_y_L * L)
    A, A_far = D["A"], D["A_far"]
    ic = int(np.argmin(np.abs(x - xc)))
    near = np.abs(x - xc) < args.win_x_L * L
    il, ir = int(np.argmin(np.abs(x - (xc - r)))), int(np.argmin(np.abs(x - (xc + r))))
    rec = dict(G_carried=float(abs(c[0])), defined=int(bool(cl[0])), G_plain=float(abs(cp[0])), flips=float(fl[0]),
               A_min_loc=float(A[W].min() / A_far), A_centre=float(A[D["jr"], ic] / A_far),
               fold_loc=float(np.mean(D["fku"][near])),
               cross_fold=float(int(D["fku"][il]) + int(D["fku"][ir])))
    if D["fphi"] is not None:
        rec["fold_loc_phi"] = float(np.mean(D["fphi"][near]))
    for tag, cells in (("u", D["cu"]), ("cos", D["cc"])):
        px, py, pq = cells
        m = (np.abs(px - xc) < args.win_x_L * L) & (np.abs(py - y_row) < args.win_y_L * L)
        rec["nX_" + tag] = float(np.sum(m & (pq > 0)))
        rec["nV_" + tag] = float(np.sum(m & (pq < 0)))
        if tag == "u":
            ins = np.hypot(px - xc, py - y_row) < r
            rec["n_in"] = float(np.sum(ins))
            V, X = m & (pq < 0), m & (pq > 0)
            if V.any() and X.any():
                d = np.hypot(px[V][:, None] - px[X][None, :], py[V][:, None] - py[X][None, :])
                rec["VX_dist_L"] = float(d.min() / L)
            else:
                rec["VX_dist_L"] = float("nan")
    return rec


def stage_onsets(S, args):
    def first(cond):
        h = np.where(np.asarray(cond, bool))[0]
        return int(h[0]) if len(h) else -1
    nu = S["nX_u"] + S["nV_u"]
    nc = S["nX_cos"] + S["nV_cos"]
    return dict(s0=0, s1=first(nu > 0), s2=first(S["fold_loc"] > args.fold_on), s3=first(S["flips"] >= 1),
                s4=first((S["defined"] > 0) & (S["G_carried"] >= args.turn_hi)), cos=first(nc > 0),
                pair=first(np.isfinite(S["VX_dist_L"])))


def stage_at(i, on):
    cur = "s0"
    for k in ["s1", "s2", "s3", "s4"]:
        if on[k] >= 0 and i >= on[k]:
            cur = k
    return cur


# ---------------------------------------------------------------------
# crops for drawing
# ---------------------------------------------------------------------

def crop_frame(D, x, y, xc, args):
    G, e, L = D["G"], D["e"], D["G"]["L"]
    y_row = G["y_row"]
    ix = np.abs(x - xc) <= args.half_x_L * L
    jy = np.abs(y - y_row) <= args.half_y_L * L
    sub = lambda a: a[np.ix_(jy, ix)]                                      # noqa: E731
    r = args.loop_L * L
    ang = np.linspace(0, 2 * np.pi, args.nang)
    px = xc + r * np.cos(ang); py = y_row + r * np.sin(ang)
    dx, dy = x[1] - x[0], y[1] - y[0]
    kx = e["k1"][np.clip(np.round((py - y[0]) / dy).astype(int), 0, len(y) - 1),
                 np.clip(np.round((px - x[0]) / dx).astype(int), 0, len(x) - 1)]
    ky = e["k2"][np.clip(np.round((py - y[0]) / dy).astype(int), 0, len(y) - 1),
                 np.clip(np.round((px - x[0]) / dx).astype(int), 0, len(x) - 1)]
    rev = np.where(kx[1:] * kx[:-1] + ky[1:] * ky[:-1] < 0)[0]
    am = 0.5 * (ang[rev] + ang[rev + 1])
    rel = lambda c: ((np.asarray(c[0]) - xc) / L, (np.asarray(c[1]) - y_row) / L, np.asarray(c[2]))  # noqa: E731
    fx = (x[D["fku"]] - xc) / L
    return dict(u=sub(D["u2"]), A=sub(D["A"] / D["A_far"]), cos=sub(np.cos(e["theta"])),
                ang=sub(np.arctan2(e["k2"], e["k1"])), xs=(x[ix] - xc) / L, ys=(y[jy] - y_row) / L,
                cu=rel(D["cu"]), cc=rel(D["cc"]), rev=(r * np.cos(am) / L, r * np.sin(am) / L),
                fold=fx[np.abs(fx) <= args.half_x_L])


def line_record(D, x, xc, args):
    """along the line, relative to the core: fold mask, A on the line, cells of u and of cos(theta) near the line"""
    G, L = D["G"], D["G"]["L"]
    y_row = G["y_row"]
    sel = np.abs(x - xc) <= args.half_x_L * L
    out = dict(xs=(x[sel] - xc) / L, fold=D["fku"][sel].astype(np.int8), A=D["A"][D["jr"], sel] / D["A_far"])
    for tag, cells in (("u", D["cu"]), ("cos", D["cc"])):
        px, py, pq = cells
        m = (np.abs(px - xc) <= args.half_x_L * L) & (np.abs(py - y_row) <= args.band_y_L * L)
        out["c_" + tag] = np.stack([(px[m] - xc) / L, (py[m] - y_row) / L, pq[m]], axis=1) if m.any() \
            else np.zeros((0, 3))
    return out


# ---------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------

def _ext(cr):
    return [cr["xs"][0], cr["xs"][-1], cr["ys"][0], cr["ys"][-1]]


def _cells(ax, c, size=5.5, hollow=False):
    px, py, pq = c
    for sel, mk, col in ((pq > 0, "^", CX), (pq < 0, "v", CV)):
        if hollow:
            ax.plot(px[sel], py[sel], mk, ms=size, mfc="none", mec=col, mew=1.2, ls="none")
        else:
            ax.plot(px[sel], py[sel], mk, ms=size, mfc=col, mec="w", mew=0.7, ls="none")


def draw_panel(ax, cr, row, args, contours=True, small=False):
    ext = _ext(cr)
    ms = 4.5 if small else 5.5
    if row == "u":
        v = np.max(np.abs(cr["u"]))
        ax.imshow(cr["u"], origin="lower", extent=ext, cmap="Greys", vmin=-v, vmax=v, aspect="equal",
                  interpolation="bilinear")
        if contours:
            X, Y = np.meshgrid(cr["xs"], cr["ys"])
            cs = ax.contour(X, Y, cr["A"], levels=args.A_levels, colors="#16a085", linewidths=0.9)
            if not small:
                ax.clabel(cs, fontsize=5.5, fmt="%.1f")
        _cells(ax, cr["cu"], ms)
    elif row == "A":
        ax.imshow(cr["A"], origin="lower", extent=ext, cmap="viridis", vmin=args.A_lo, vmax=args.A_hi,
                  aspect="equal", interpolation="bilinear")
        _cells(ax, cr["cu"], ms)
    elif row == "cos":
        ax.imshow(cr["cos"], origin="lower", extent=ext, cmap="Greys", vmin=-1, vmax=1, aspect="equal",
                  interpolation="bilinear")
        _cells(ax, cr["cc"], ms)
    elif row == "k":
        ax.imshow(cr["ang"], origin="lower", extent=ext, cmap="twilight", vmin=-np.pi, vmax=np.pi, aspect="equal",
                  interpolation="nearest")
        f = np.sort(cr["fold"])
        if f.size:
            # draw each folded run as a thick segment on the line
            step = np.median(np.diff(cr["xs"])) if len(cr["xs"]) > 1 else 0.05
            brk = np.where(np.diff(f) > 1.5 * step)[0]
            starts = np.r_[f[0], f[brk + 1]]; ends = np.r_[f[brk], f[-1]]
            for s0, s1 in zip(starts, ends):
                ax.plot([s0 - 0.5 * step, s1 + 0.5 * step], [0, 0], "-", color="#e74c3c", lw=4.0 if not small else 3.0,
                        solid_capstyle="butt", alpha=0.95)
        _cells(ax, cr["cu"], ms)
        rx, ry = cr["rev"]
        ax.plot(rx, ry, "X", ms=8.5 if not small else 7, mfc="w", mec="k", mew=1.3, ls="none")
    ax.add_patch(Circle((0, 0), args.loop_L, fill=False, ec="#f39c12", lw=1.4))
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.tick_params(labelsize=6.5)


ROWNAME = {"u": "pattern u, its cells", "A": "amplitude A / A_far", "cos": "phase alone cos(theta), its cells",
           "k": "direction of k, fold, flips"}


def key_handles(contours):
    h = [Line2D([], [], marker="^", color="w", mfc=CX, mec="w", ms=7, label="X cell (+1/2 turn)"),
         Line2D([], [], marker="v", color="w", mfc=CV, mec="w", ms=7, label="V cell (-1/2 turn)"),
         Line2D([], [], color="#f39c12", lw=1.5, label="loop of radius L/2"),
         Line2D([], [], color="#e74c3c", lw=4, label="folded stretch of the line"),
         Line2D([], [], marker="X", color="w", mfc="w", mec="k", ms=8, label="flip of k on the loop")]
    if contours:
        h.append(Line2D([], [], color="#16a085", lw=1, label="amplitude contours (A / A_far)"))
    return h


def draw_wheel(fig, rect):
    ax = fig.add_axes(rect, projection="polar")
    th = np.linspace(-np.pi, np.pi, 256)
    rr = np.linspace(0.55, 1.0, 2)
    T, R = np.meshgrid(th, rr)
    ax.pcolormesh(T, R, T, cmap="twilight", vmin=-np.pi, vmax=np.pi, shading="auto")
    ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["right", "up", "left", "down"], fontsize=6)
    ax.set_title("color of k", fontsize=7, pad=8)
    ax.tick_params(pad=1)
    return ax


def fig_panels(cols, title, path, args, contours):
    rows = ["u", "cos", "k"] if contours else ["u", "A", "cos", "k"]
    nc = len(cols)
    fig = plt.figure(figsize=(2.35 * nc + 2.6, 1.95 * len(rows) + 1.3))
    gs = fig.add_gridspec(len(rows), nc, left=0.06, right=0.80, top=0.86, bottom=0.05, wspace=0.08, hspace=0.12)
    for j, (lab, col, t, cr) in enumerate(cols):
        for i, row in enumerate(rows):
            ax = fig.add_subplot(gs[i, j])
            draw_panel(ax, cr, row, args, contours=contours, small=True)
            if i == 0:
                ax.set_title("%s\nt = %.4g" % (lab, t), fontsize=7.5, color=col)
            if j == 0:
                ax.set_ylabel(ROWNAME[row], fontsize=7)
            if i < len(rows) - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
    fig.legend(handles=key_handles(contours), loc="upper left", bbox_to_anchor=(0.81, 0.85), fontsize=7, frameon=False)
    draw_wheel(fig, [0.835, 0.10, 0.10, 0.16])
    fig.suptitle(title + "\naxes: distance from the core in units of the dislocation spacing L; one director width (0.4) "
                 "for every cell", fontsize=8.5)
    fig.savefig(path.with_suffix(".png"), dpi=170, bbox_inches="tight"); fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def time_window(ts, args):
    """indices kept on the time axis (t <= t_max when t_max > 0)"""
    return np.where(ts <= args.t_max + 1e-12)[0] if args.t_max > 0 else np.arange(len(ts))


def set_time_axis(ax, t_lo, t_hi, args):
    """time runs down; linear, square-root or symlog spacing"""
    if args.time_scale == "sqrt":
        ax.set_yscale("function", functions=(lambda t: np.sqrt(np.clip(t, 0, None)), lambda s_: np.asarray(s_) ** 2))
    elif args.time_scale == "symlog":
        ax.set_yscale("symlog", linthresh=args.symlog_lin)
    ax.set_ylim(t_hi, max(t_lo, 0.0) if args.time_scale == "sqrt" else t_lo)


def _t_edges(ts):
    if len(ts) == 1:
        return np.array([ts[0] - 0.5, ts[0] + 0.5])
    mid = 0.5 * (ts[1:] + ts[:-1])
    e = np.r_[ts[0] - (mid[0] - ts[0]), mid, ts[-1] + (ts[-1] - mid[-1])]
    e[0] = max(e[0], 0.0) if ts[0] >= 0 else e[0]
    return e


def draw_spacetime(ax, LR, ts, on, args, small=False, cursor=None):
    """position along the line across, time down"""
    keep = time_window(ts, args)
    LR = [LR[i] for i in keep]; ts = ts[keep]
    xs = LR[0]["xs"]
    Aimg = np.array([lr["A"] if len(lr["A"]) == len(xs) else np.interp(xs, lr["xs"], lr["A"]) for lr in LR])
    Fimg = np.array([lr["fold"] if len(lr["fold"]) == len(xs) else np.interp(xs, lr["xs"], lr["fold"]) for lr in LR],
                    float)
    if cursor is not None and args.reveal:
        hide = ts > cursor + 1e-12
        Aimg = np.where(hide[:, None], np.nan, Aimg)
        Fimg = np.where(hide[:, None], 0.0, Fimg)
    dxs = np.median(np.diff(xs)) if len(xs) > 1 else 0.05
    xe = np.r_[xs - 0.5 * dxs, xs[-1] + 0.5 * dxs]
    te = _t_edges(ts)
    ax.pcolormesh(xe, te, np.ma.masked_invalid(Aimg), cmap="Greys_r", vmin=args.A_lo, vmax=args.A_hi, alpha=0.55,
                  shading="flat", rasterized=True)
    ax.pcolormesh(xe, te, np.ma.masked_where(Fimg < 0.5, Fimg), cmap="Reds", vmin=0, vmax=1.2, alpha=0.75,
                  shading="flat", rasterized=True)
    ms = 2.2 if small else 3.2
    for tag, hollow in (("u", False), ("cos", True)):
        for sgn, col in ((1, CX), (-1, CV)):
            T, Xp = [], []
            for i, lr in enumerate(LR):
                if cursor is not None and args.reveal and ts[i] > cursor + 1e-12:
                    continue
                c = lr["c_" + tag]
                if c.size:
                    sel = c[:, 2] == sgn
                    Xp.extend(c[sel, 0].tolist()); T.extend([ts[i]] * int(sel.sum()))
            if T:
                if hollow:
                    ax.plot(Xp, T, "o", ms=ms + 1.2, mfc="none", mec=col, mew=0.7, ls="none", alpha=0.9)
                else:
                    ax.plot(Xp, T, "o", ms=ms, mfc=col, mec="none", ls="none", alpha=0.9)
    for v in (-args.loop_L, args.loop_L):
        ax.axvline(v, color="#f39c12", ls="--", lw=1.1)
    at = {}
    for k in ["s1", "s2", "s3", "s4"]:
        tk = None
        if on[k] >= 0:
            tk = on["t_" + k] if ("t_" + k) in on else None
        if tk is None or tk > ts[-1] + 1e-12:
            continue
        if cursor is not None and args.reveal and tk > cursor + 1e-12:
            continue
        ax.axhline(tk, color=SCOL[k], lw=0.9, ls=":")
        at.setdefault(tk, []).append(k)
    if not small:
        for tk, ks in at.items():
            ax.text(xs[-1] + 0.5 * dxs, tk, " " + ", ".join((("1-2: %s" % SNAME[k]) if k in ("s1", "s2") else
                                                              "%s: %s" % (k[1], SNAME[k])) for k in ks), color=SCOL[ks[-1]],
                    fontsize=6.5, va="center", ha="left", clip_on=False)
    if cursor is not None:
        ax.axhline(cursor, color="k", lw=1.6)
    ax.set_xlim(xe[0], xe[-1])
    set_time_axis(ax, te[0], te[-1], args)
    ax.tick_params(labelsize=6.5)


def draw_counts(ax, S, ts, on, args, cursor=None):
    """the count strip beside the space-time diagram: cells of u inside the loop, flips of k on the loop (time down);
    the amplitude minimum near the core on a twin axis"""
    keep = time_window(ts, args)
    t = ts[keep]
    if cursor is not None and args.reveal:
        keep = keep[t <= cursor + 1e-12]; t = ts[keep]
    ax.step(S["n_in"][keep], t, where="mid", color="k", lw=1.2, label="cells of u inside the loop")
    ax.step(S["flips"][keep], t, where="mid", color="#6c3483", lw=1.2, ls="--", label="flips of k on the loop")
    ax.set_xlim(-0.3, max(6.5, float(np.nanmax(S["n_in"])) + 0.5))
    ax.set_xlabel("count", fontsize=7)
    ax2 = ax.twiny()
    ax2.plot(S["A_min_loc"][keep], t, "-", color="#16a085", lw=1.2, label="amplitude minimum near the core")
    ax2.set_xlim(args.A_lo, 1.0)
    ax2.set_xlabel("A_min / A_far", fontsize=7, color="#16a085")
    ax2.tick_params(labelsize=6, colors="#16a085")
    te = _t_edges(ts[time_window(ts, args)])
    for a_ in (ax, ax2):
        set_time_axis(a_, te[0], te[-1], args)
    if cursor is not None:
        ax.axhline(cursor, color="k", lw=1.6)
    ax.tick_params(labelsize=6.5)
    ax.legend(handles=[Line2D([], [], color="k", label="cells of u inside the loop"),
                       Line2D([], [], color="#6c3483", ls="--", label="flips of k on the loop"),
                       Line2D([], [], color="#16a085", label="amplitude minimum")], fontsize=5.5, loc="lower right")


def st_key():
    return [Line2D([], [], marker="o", color="w", mfc=CX, mec="none", ms=5, label="X cell of u"),
            Line2D([], [], marker="o", color="w", mfc=CV, mec="none", ms=5, label="V cell of u"),
            Line2D([], [], marker="o", color="w", mfc="none", mec="k", ms=5, label="open: cell of cos(theta)"),
            Line2D([], [], color="#e74c3c", lw=5, alpha=0.6, label="folded stretch of the line"),
            Line2D([], [], color="#f39c12", ls="--", label="where the loop meets the line"),
            Line2D([], [], color="0.5", lw=5, alpha=0.5, label="grey: A on the line (dark = weak)")]


def fig_spacetime(LR, S, ts, on, title, path, args):
    fig = plt.figure(figsize=(10.2, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.0, 1.35], left=0.08, right=0.97, top=0.80, bottom=0.09, wspace=0.62)
    ax = fig.add_subplot(gs[0, 0]); axc = fig.add_subplot(gs[0, 1])
    draw_spacetime(ax, LR, ts, on, args)
    draw_counts(axc, S, ts, on, args)
    axc.set_yticklabels([])
    ax.set_xlabel("position along the line, from the core (L)", fontsize=8)
    ax.set_ylabel("time (%s scale)" % args.time_scale, fontsize=8)
    fig.legend(handles=st_key(), loc="upper center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.45, 0.93))
    fig.suptitle(title + "\ncells within %.2g L of the line, one director width (0.4); the flips follow the cells crossing "
                 "the loop (dashed lines)" % args.band_y_L, fontsize=8)
    fig.savefig(path.with_suffix(".png"), dpi=170, bbox_inches="tight"); fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_spacetime_grid(allLR, ts, ons, labels, title, path, args):
    n = len(allLR)
    nc = min(n, args.grid_cols); nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.3 * nc + 0.4, 2.6 * nr + 0.9), sharex=True, sharey=True,
                            constrained_layout=True, squeeze=False)
    for j in range(nr * nc):
        ax = axs[j // nc, j % nc]
        if j >= n:
            ax.axis("off"); continue
        draw_spacetime(ax, allLR[j], ts, ons[j], args, small=True)
        ax.set_title(labels[j], fontsize=7)
    for ax in axs[-1, :]:
        ax.set_xlabel("along the line (L)", fontsize=7)
    for ax in axs[:, 0]:
        ax.set_ylabel("time", fontsize=7)
    fig.legend(handles=st_key()[:5], loc="upper center", ncol=5, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, fontsize=8.5, y=1.06)
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight"); fig.savefig(path.with_suffix(".pdf"),
                                                                                   bbox_inches="tight")
    plt.close(fig)


def fig_separation(ts, Ss, ons, labels, title, path, args):
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True, constrained_layout=True)
    cm = plt.get_cmap("viridis")
    n = len(Ss)
    for j, (S, on) in enumerate(zip(Ss, ons)):
        col = cm(0.1 + 0.8 * j / max(n - 1, 1))
        axs[0].plot(ts, S["VX_dist_L"], "-", color=col, lw=1.0, label=labels[j])
        axs[1].plot(ts, S["A_min_loc"], "-", color=col, lw=1.0)
        for k in ["s3", "s4"]:
            if on[k] >= 0:
                axs[0].plot(ts[on[k]], S["VX_dist_L"][on[k]] if np.isfinite(S["VX_dist_L"][on[k]]) else np.nan,
                            "o" if k == "s3" else "s", color=SCOL[k], ms=4)
    axs[0].set_ylabel("nearest V-X distance\namong u's cells near the core (L)", fontsize=7.5)
    axs[1].set_ylabel("amplitude minimum near the core\n(A / A_far)", fontsize=7.5)
    axs[1].set_xlabel("time", fontsize=8)
    if args.sep_log_t and ts[0] >= 0:
        for ax in axs:
            ax.set_xscale("symlog", linthresh=max(ts[1] - ts[0], 1e-3))
    axs[0].legend(fontsize=6, ncol=2)
    axs[0].legend(handles=[Line2D([], [], marker="o", ls="none", color=SCOL["s3"], label="first crossing (stage 3)"),
                           Line2D([], [], marker="s", ls="none", color=SCOL["s4"], label="second crossing (stage 4)")]
                  + [Line2D([], [], color=cm(0.1 + 0.8 * j / max(n - 1, 1)), label=labels[j]) for j in range(n)],
                  fontsize=6, ncol=2)
    fig.suptitle(title, fontsize=8.5)
    fig.savefig(path.with_suffix(".png"), dpi=170); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)


def make_anim(ts, crops, LR, S, on, title, base, args):
    nt = len(ts)
    fig = plt.figure(figsize=(11.5, 8.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.78, 1.0], left=0.06, right=0.80, top=0.88, bottom=0.07,
                          wspace=0.14, hspace=0.16)
    axu, axc, axk = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    gsb = gs[1, :].subgridspec(1, 2, width_ratios=[3.3, 1.0], wspace=0.30)
    axs = fig.add_subplot(gsb[0, 0]); axn = fig.add_subplot(gsb[0, 1])
    fig.legend(handles=key_handles(True), loc="upper left", bbox_to_anchor=(0.805, 0.88), fontsize=7, frameon=False)
    wax = draw_wheel(fig, [0.845, 0.50, 0.085, 0.12])
    wax._mig_wheel = True
    fig.legend(handles=st_key(), loc="upper left", bbox_to_anchor=(0.805, 0.42), fontsize=6.5, frameon=False,
               title="space-time diagram (below)", title_fontsize=7)
    banner = fig.text(0.06, 0.935, "", fontsize=11, fontweight="bold", va="center", wrap=True)
    fig.suptitle(title, fontsize=8.5, y=0.985)
    seq = []
    for i in range(0, nt, max(1, args.anim_stride)):
        seq.append(i)
        if any(on[k] == i for k in ["s1", "s2", "s3", "s4"]):
            seq.extend([i] * args.hold_frames)

    def draw(i):
        for ax in list(fig.axes):
            if ax not in (axu, axc, axk, axs, axn) and getattr(ax, "_mig_twin", False):
                ax.remove()
        for ax in (axu, axc, axk, axs, axn):
            ax.cla()
        cr = crops[i]
        draw_panel(axu, cr, "u", args, contours=True)
        draw_panel(axc, cr, "cos", args)
        draw_panel(axk, cr, "k", args)
        axu.set_title(ROWNAME["u"] + " (contours: A)", fontsize=9)
        axc.set_title(ROWNAME["cos"], fontsize=9)
        axk.set_title(ROWNAME["k"], fontsize=9)
        draw_spacetime(axs, LR, ts, on, args, cursor=ts[i])
        draw_counts(axn, S, ts, on, args, cursor=ts[i])
        for ax in fig.axes:
            if ax not in (axu, axc, axk, axs, axn) and not hasattr(ax, "_mig_wheel"):
                ax._mig_twin = True
        axn.set_yticklabels([])
        axs.set_xlabel("position along the line, from the core (L)", fontsize=8)
        axs.set_ylabel("time (%s scale)" % args.time_scale, fontsize=8)
        st = stage_at(i, on)
        banner.set_text("t = %.4g    %s" % (ts[i], SBANNER[st]))
        banner.set_color(SCOL[st])
        return []

    made = []
    try:
        ani = animation.FuncAnimation(fig, draw, frames=seq, blit=False)
        ani.save(str(base.with_suffix(".gif")), writer=animation.PillowWriter(fps=args.fps), dpi=args.anim_dpi)
        made.append(base.with_suffix(".gif").name)
    except Exception as exc:                                               # loud, not blocking
        made.append("LOUD: gif failed (%s)" % exc)
    plt.close(fig)
    return made



# ---------------------------------------------------------------------
# the travelling X (v1.3)
# ---------------------------------------------------------------------

def x_track(LR, tag, args):
    """per frame: position (L, along the line from the core) of the on-line X cell of field tag furthest along +x
    between xt_lo and xt_hi; nan where there is none"""
    out = np.full(len(LR), np.nan)
    for i, lr in enumerate(LR):
        c = lr["c_" + tag]
        if c.size:
            m = (c[:, 2] > 0) & (np.abs(c[:, 1]) <= args.on_line_L) & (c[:, 0] >= args.xt_lo) & (c[:, 0] < args.xt_hi)
            if m.any():
                out[i] = float(c[m, 0].max())
    return out


def glide_stats(ts, xu, xcos, dxL, args):
    """birth, final position, speed to the midpoint, loop-passing time of the travelling X of u; final lag of
    cos(theta)'s X behind it"""
    nan = float("nan")
    r = dict(t_xt_birth=nan, xt_birth_L=nan, xt_final_L=nan, t_xt_half=nan, v_xt_half=nan, t_xt_pass_loop=nan,
             xt_cos_final_L=nan, xt_lag_L=nan)
    ok = np.where(np.isfinite(xu))[0]
    if len(ok):
        ib = int(ok[0])
        r["t_xt_birth"], r["xt_birth_L"] = float(ts[ib]), float(xu[ib])
        r["xt_final_L"] = float(np.median(xu[ok[-5:]]))
        if r["xt_final_L"] - r["xt_birth_L"] >= 2 * dxL:
            mid = 0.5 * (r["xt_birth_L"] + r["xt_final_L"])
            h = [i for i in ok if i > ib and xu[i] >= mid - 1e-9]
            if h and ts[h[0]] > ts[ib]:
                r["t_xt_half"] = float(ts[h[0]])
                r["v_xt_half"] = float((xu[h[0]] - xu[ib]) / (ts[h[0]] - ts[ib]))
        ps = [i for i in ok if xu[i] >= args.loop_L - 0.5 * dxL]
        if ps:
            r["t_xt_pass_loop"] = float(ts[ps[0]])
    okc = np.where(np.isfinite(xcos))[0]
    if len(okc):
        r["xt_cos_final_L"] = float(np.median(xcos[okc[-5:]]))
        if np.isfinite(r["xt_final_L"]):
            r["xt_lag_L"] = r["xt_final_L"] - r["xt_cos_final_L"]
    return r


def set_time_x(ax, args, scale=None):
    """time along x: linear, square-root, symlog or log"""
    sc = scale or args.time_scale
    if sc == "sqrt":
        ax.set_xscale("function", functions=(lambda t: np.sqrt(np.clip(t, 0, None)), lambda s_: np.asarray(s_) ** 2))
    elif sc == "symlog":
        ax.set_xscale("symlog", linthresh=args.symlog_lin)
    elif sc == "log":
        ax.set_xscale("log")


def fig_xtrack(ts, XU, XC, Ss, ons, labels, jx_pos, title, path, args):
    """top: the travelling X of u (solid) and of cos(theta) (dashed), every interior core, the exemplar thick;
    bottom: the amplitude minimum near the core"""
    fig, axs = plt.subplots(2, 1, figsize=(9.4, 6.0), sharex=True, constrained_layout=True,
                            gridspec_kw=dict(height_ratios=[1.5, 1.0]))
    cm = plt.get_cmap("viridis")
    n = len(XU)
    for j in range(n):
        ex = (j == jx_pos)
        col = "k" if ex else cm(0.15 + 0.7 * j / max(n - 1, 1))
        lw, al = (2.0, 1.0) if ex else (0.9, 0.55)
        axs[0].plot(ts, XU[j], "-", color=col, lw=lw, alpha=al, drawstyle="steps-mid")
        axs[0].plot(ts, XC[j], "--", color=col, lw=lw * 0.8, alpha=al, drawstyle="steps-mid")
        axs[1].plot(ts, Ss[j]["A_min_loc"], "-", color=col, lw=lw * 0.8, alpha=al)
        k3 = ons[j]["s3"]
        if k3 >= 0 and np.isfinite(XU[j][k3]):
            axs[0].plot(ts[k3], XU[j][k3], "o", color=SCOL["s3"], ms=5 if ex else 3.5, zorder=5)
    axs[0].axhline(args.loop_L, color="#f39c12", ls="--", lw=1.0)
    axs[0].set_ylabel("position of the travelling X cell\nalong the line, from the core (L)", fontsize=7.5)
    axs[1].set_ylabel("amplitude minimum near the core\n(A / A_far)", fontsize=7.5)
    axs[1].set_xlabel("time (%s scale)" % args.time_scale, fontsize=8)
    set_time_x(axs[1], args)
    axs[1].set_xlim(ts[0], ts[-1])
    axs[0].legend(handles=[Line2D([], [], color="k", lw=2.0, label="exemplar core, X cell of u"),
                           Line2D([], [], color="k", lw=1.6, ls="--", label="exemplar core, X cell of cos(theta)"),
                           Line2D([], [], color=cm(0.5), lw=0.9, label="other interior cores"),
                           Line2D([], [], color="#f39c12", ls="--", label="loop radius (where the loop meets the line)"),
                           Line2D([], [], marker="o", ls="none", color=SCOL["s3"], label="first crossing (stage 3)")],
                  fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    for ax in axs:
        ax.tick_params(labelsize=7)
    fig.suptitle(title + "\non-line X cells (|y| <= %.2g L), the one furthest along +x between %.2g and %.2g L; "
                 "coarse cores repeat every third core (grid)" % (args.on_line_L, args.xt_lo, args.xt_hi), fontsize=8)
    fig.savefig(path.with_suffix(".png"), dpi=170); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)


def fig_timing(rows, path):
    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    off = {"coarse": -0.0025, "fine": 0.0025}
    for r in rows:
        x0 = r["mu"] + off.get(r["cadence"], 0.0)
        tt = [r["t_" + k] for k in ["s1", "s2", "s3", "s4"]]
        ok = [v for v in tt if np.isfinite(v)]
        if ok:
            ax.plot([x0, x0], [min(ok), max(ok)], "-", color="0.75", lw=0.8, zorder=1)
        for k, v in zip(["s1", "s2", "s3", "s4"], tt):
            if np.isfinite(v):
                ax.plot(x0, v, "o" if r["cadence"] == "coarse" else "D", ms=4, color=SCOL[k],
                        mfc=SCOL[k] if r["cadence"] == "coarse" else "none", zorder=2)
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_xlabel("mu"); ax.set_ylabel("time the stage begins at a core")
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", color=SCOL[k], label="stage %s: %s" % (k[1], SNAME[k]))
                       for k in ["s1", "s2", "s3", "s4"]]
              + ([Line2D([], [], marker="o", ls="none", color="k", label="filled: dt = 0.2")]
                 if any(r["cadence"] == "coarse" for r in rows) else [])
              + ([Line2D([], [], marker="D", ls="none", color="k", mfc="none", label="open: dt = 0.025")]
                 if any(r["cadence"] == "fine" for r in rows) else []), fontsize=6.5)
    ax.set_title("every core: when each stage begins (one director width, 0.4)", fontsize=8.5)
    fig.savefig(path.with_suffix(".png"), dpi=170); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)


# ---------------------------------------------------------------------
# one movie
# ---------------------------------------------------------------------

def write_csv(rows, path):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


def run_movie(path, args, out, lines, cores_out):
    u, x, y, t, md = F.movie(path)
    mu = float(md.get("mu", F.mu_of(path)))
    jr = int(md.get("gb_row_in_crop", u.shape[0] // 2)); y_row = float(y[jr])
    nt = u.shape[-1]
    ts = np.array([float(t[min(i, t.size - 1)]) for i in range(nt)])
    stem = Path(path).stem
    cad = "fine" if (nt > 1 and ts[1] - ts[0] < 0.1) else "coarse"
    if mu < args.mu_min - 1e-9:
        lines.append("%s: mu %.3f below mu_min %.2f (scope: dislocation angles); skipped" % (stem, mu, args.mu_min))
        return
    d = out / stem; d.mkdir(parents=True, exist_ok=True)
    DL = frame_fields(np.asarray(u[..., -1], float), x, y, y_row, mu, args)
    G0 = DL["G"]; L = G0["L"]
    xd, _, _, _, _ = P.find_dip_patches(DL["A"], x, y, G0["win"], args.core_frac, 0.5 * L)
    centres = F.centres_from(xd, L, G0["xm"], G0["Lx"])
    if not len(centres):
        lines.append("LOUD: %s: no cores in the last frame" % stem); return
    nc = len(centres)
    recs = [[] for _ in range(nc)]
    LRs = [[] for _ in range(nc)]
    crops = [[] for _ in range(nc)]
    for i in range(nt):
        D = frame_fields(np.asarray(u[..., i], float), x, y, y_row, mu, args)
        D["u2"] = np.asarray(u[..., i], float)
        for j, xc in enumerate(centres):
            recs[j].append(core_record(D, x, y, xc, args))
            LRs[j].append(line_record(D, x, xc, args))
            crops[j].append(crop_frame(D, x, y, xc, args) if (args.core < 0 or j == args.core) else None)
    del u
    Ss, ons, labels, edges = [], [], [], []
    series_rows = []
    XU, XC = [], []
    xs0 = LRs[0][0]["xs"]
    dxL = float(np.median(np.diff(xs0))) if len(xs0) > 1 else 0.05
    for j in range(nc):
        S = {k: np.array([r[k] for r in recs[j]], float) for k in recs[j][0]}
        on = stage_onsets(S, args)
        for k in ["s1", "s2", "s3", "s4"]:
            on["t_" + k] = float(ts[on[k]]) if on[k] >= 0 else None
        edge = bool(abs(centres[j] - G0["xm"]) + args.half_x_L * L > args.edge_frac * G0["Lx"])
        edges.append(edge)
        Ss.append(S); ons.append(on)
        labels.append("core %d (x = %+.2f L)%s" % (j, (centres[j] - G0["xm"]) / L, ", edge" if edge else ""))
        for i in range(nt):
            series_rows.append(dict(core=j, t=ts[i], frame=i, stage=stage_at(i, on), **recs[j][i]))
        f = lambda k: float(ts[on[k]]) if on[k] >= 0 else float("nan")      # noqa: E731
        amp_first = (on["s1"] >= 0) and (on["cos"] < 0 or on["cos"] > on["s1"])
        parity = float(np.mean((S["flips"] % 2) == (S["n_in"] % 2)))
        pair_sep = float(S["VX_dist_L"][on["pair"]]) if on["pair"] >= 0 else float("nan")
        row = dict(stem=stem, mu=mu, cadence=cad, core=j, x_core_L=(centres[j] - G0["xm"]) / L, edge_core=int(edge),
                   t_s1=f("s1"), t_s2=f("s2"), t_s3=f("s3"), t_s4=f("s4"), t_cos=f("cos"),
                   amplitude_first=int(amp_first), lead_cells_over_cos=(f("cos") - f("s1")) if on["cos"] >= 0 else float("nan"),
                   parity_frac=parity, t_first_pair=f("pair"), first_pair_sep_L=pair_sep,
                   min_pair_sep_L=float(np.nanmin(S["VX_dist_L"])) if np.isfinite(S["VX_dist_L"]).any() else float("nan"),
                   undefined_stretch=(f("s4") - f("s3")) if (on["s3"] >= 0 and on["s4"] >= 0) else float("nan"),
                   reaches_one_turn=int(on["s4"] >= 0))
        xu, xcos = x_track(LRs[j], "u", args), x_track(LRs[j], "cos", args)
        XU.append(xu); XC.append(xcos)
        row.update(glide_stats(ts, xu, xcos, dxL, args))
        row["exemplar"] = 0
        cores_out.append(row)
    write_csv(series_rows, d / "mig_core_series.csv")
    write_csv([dict(core=j, t=ts[i], frame=i, x_X_u_L=XU[j][i], x_X_cos_L=XC[j][i])
               for j in range(nc) for i in range(nt)], d / "mig_xtrack.csv")
    cell_rows = []
    for j in range(nc):
        for i in range(nt):
            for tag in ("u", "cos"):
                c = LRs[j][i]["c_" + tag]
                for cx_, cy_, cq_ in c:
                    cell_rows.append(dict(core=j, t=ts[i], frame=i, field=tag, kind="X" if cq_ > 0 else "V",
                                          x_L=round(float(cx_), 5), y_L=round(float(cy_), 5)))
    write_csv(cell_rows, d / "mig_cells.csv")
    # exemplar: --core, else the first core to reach one turn
    if args.core >= 0:
        jx = min(args.core, nc - 1)
    else:
        hits = [ts[o["s4"]] if (o["s4"] >= 0 and not e_) else np.inf for o, e_ in zip(ons, edges)]
        jx = int(np.argmin(hits))
        if not np.isfinite(hits[jx]):
            inner = [j for j in range(nc) if not edges[j]]
            jx = inner[len(inner) // 2] if inner else 0
            lines.append("%s: no interior core reaches one turn; exemplar is core %d" % (stem, jx))
    cores_out[-nc + jx]["exemplar"] = 1
    if edges[jx]:
        lines.append("LOUD: %s: the exemplar core %d is an edge core (cells near the taper are unreliable)" % (stem, jx))
    on = ons[jx]
    title = "%s   mu %.3f (%.0f deg), core %d at x = %+.2f L" % (stem, mu, np.degrees(np.arcsin(mu)), jx,
                                                                   (centres[jx] - G0["xm"]) / L)
    np.savez_compressed(d / "mig_line.npz", ts=ts, xs=LRs[jx][0]["xs"],
                        fold=np.array([lr["fold"] for lr in LRs[jx]]), A=np.array([lr["A"] for lr in LRs[jx]]),
                        core=jx, mu=mu)
    fig_spacetime(LRs[jx], Ss[jx], ts, on, title, d / "mig_spacetime", args)
    fig_spacetime_grid(LRs, ts, ons, labels, "%s   mu %.3f: every core" % (stem, mu), d / "mig_spacetime_grid", args)
    inner = [j for j in range(nc) if not edges[j]] or list(range(nc))
    if args.sep_fig:
        fig_separation(ts, [Ss[j] for j in inner], [ons[j] for j in inner], [labels[j] for j in inner],
                       "%s   mu %.3f: the V-X pairs of u's cells and the amplitude, every core" % (stem, mu),
                       d / "mig_separation", args)
    fig_xtrack(ts, [XU[j] for j in inner], [XC[j] for j in inner], [Ss[j] for j in inner], [ons[j] for j in inner],
               [labels[j] for j in inner], inner.index(jx) if jx in inner else -1,
               "%s   mu %.3f: the travelling X cell and the amplitude, every interior core" % (stem, mu),
               d / "mig_xtrack", args)
    cr = crops[jx]
    cols, seen = [], {}
    marks = [(0, "s0")] + [(on[k], k) for k in ["s1", "s2", "s3", "s4"] if on[k] >= 0] + [(nt - 1, "end")]
    for i0, k in sorted(marks):
        seen.setdefault(i0, []).append(k)
    for i0 in sorted(seen):
        ks = seen[i0]
        names = [("stages 1-2: %s" % SNAME[k]) if k in ("s1", "s2") else
                 ("stage %s: %s" % (k[1], SNAME[k]) if k != "end" else "last frame") for k in ks]
        if ks == ["s0"]:
            names = ["first frame"]
        cols.append(("\n".join(names), SCOL[stage_at(i0, on)], ts[i0], cr[i0]))
    fig_panels(cols, title, d / "mig_panels_contours", args, contours=True)
    fig_panels(cols, title, d / "mig_panels_rows", args, contours=False)
    made = make_anim(ts, cr, LRs[jx], Ss[jx], on, title, d / "mig_anim", args) if args.anim else ["animation off"]
    S = Ss[jx]
    fmt = lambda k: ("%.4g" % ts[on[k]]) if on[k] >= 0 else "never"       # noqa: E731
    msg = ("%s mu %.3f (%s, %d cores; exemplar core %d): stage 1 %s, stage 2 %s, cos(theta) cells %s, stage 3 %s, "
           "stage 4 %s; first V-X pair of u's cells at %s, %.3f L apart; parity %.0f%%; travelling X from %.3f L "
           "at t = %.4g to %.3f L (speed to midpoint %.3g L per time unit, passes the loop at %.4g; cos(theta)'s X "
           "%.3f L behind at the end); %s"
           % (stem, mu, cad, nc, jx, fmt("s1"), fmt("s2"), fmt("cos"), fmt("s3"), fmt("s4"), fmt("pair"),
              cores_out[-nc + jx]["first_pair_sep_L"], 100 * cores_out[-nc + jx]["parity_frac"],
              cores_out[-nc + jx]["xt_birth_L"], cores_out[-nc + jx]["t_xt_birth"], cores_out[-nc + jx]["xt_final_L"],
              cores_out[-nc + jx]["v_xt_half"], cores_out[-nc + jx]["t_xt_pass_loop"], cores_out[-nc + jx]["xt_lag_L"],
              "; ".join(made)))
    print(msg); lines.append(msg)


ONSET_KEYS = ["t_s1", "t_s2", "t_s3", "t_s4", "t_cos"]


def n_distinct(rs):
    """distinct onset tuples among rows (coarse cores repeat every third core: dx / L = 3/64)"""
    return len(set(tuple(("nan" if not np.isfinite(float(r[k])) else round(float(r[k]), 6)) for k in ONSET_KEYS)
                   for r in rs))


def _rng(v, fmt="%.3g"):
    v = [float(a) for a in v if np.isfinite(float(a))]
    if not v:
        return "n/a"
    lo, hi = fmt % min(v), fmt % max(v)
    return lo if lo == hi else "%s to %s" % (lo, hi)


def glide_line(rs):
    return ("born %s L at t = %s; final %s L; speed to midpoint %s L per time unit; passes the loop at t = %s "
            "(stage 3 at %s); cos(theta)'s X %s L behind at the end"
            % (_rng([r["xt_birth_L"] for r in rs]), _rng([r["t_xt_birth"] for r in rs], "%.4g"),
               _rng([r["xt_final_L"] for r in rs]), _rng([r["v_xt_half"] for r in rs]),
               _rng([r["t_xt_pass_loop"] for r in rs], "%.4g"), _rng([r["t_s3"] for r in rs], "%.4g"),
               _rng([r["xt_lag_L"] for r in rs])))


def summarize(rows, lines):
    if not rows:
        return
    lines.append("")
    lines.append("== across cores (one director width, %s) ==" % "0.4")
    ne = sum(r["edge_core"] for r in rows)
    lines.append("edge cores (within %s of the taper; left out below): %d" % ("the window", ne))
    rows = [r for r in rows if not r["edge_core"]]
    hit = [r for r in rows if r["reaches_one_turn"]]
    lines.append("interior cores: %d, of which %d reach one turn" % (len(rows), len(hit)))
    for r_set, name in ((rows, "all cores"), (hit, "cores reaching one turn")):
        if not r_set:
            continue
        amp = sum(r["amplitude_first"] for r in r_set)
        par = [r["parity_frac"] for r in r_set]
        lines.append("%s: amplitude-first (cells of u before any cell of cos(theta), or cos never) at %d of %d; "
                     "flip parity equals the parity of u's cells inside the loop in %.1f%% to %.1f%% of frames (median %.1f%%)"
                     % (name, amp, len(r_set), 100 * min(par), 100 * max(par), 100 * float(np.median(par))))
    order = [r for r in hit if all(np.isfinite(r["t_" + k]) for k in ["s1", "s2", "s3", "s4"])]
    ok34 = sum(1 for r in order if r["t_s3"] >= max(r["t_s1"], r["t_s2"]) and r["t_s4"] >= r["t_s3"])
    lines.append("stages 3 and 4 after stages 1 and 2: %d of %d cores with all four onsets" % (ok34, len(order)))
    for mu in sorted(set(r["mu"] for r in rows)):
        rs = [r for r in rows if r["mu"] == mu]
        h = [r for r in rs if r["reaches_one_turn"]]
        ps = [r["first_pair_sep_L"] for r in rs if np.isfinite(r["first_pair_sep_L"])]
        us = [r["undefined_stretch"] for r in h if np.isfinite(r["undefined_stretch"])]
        nd = n_distinct(rs)
        lines.append("mu %.3f: %d of %d cores reach one turn (%d distinct cases among the %d interior cores); first "
                     "V-X pair separation %s L; stretch with the circulation undefined %s"
                     % (mu, len(h), len(rs), nd, len(rs),
                        ("%.3f to %.3f" % (min(ps), max(ps))) if ps else "n/a",
                        ("%.3g to %.3g" % (min(us), max(us))) if us else "n/a"))
        lines.append("   travelling X: %s" % glide_line(rs))



# ---------------------------------------------------------------------
# combine mode (v1.3): coarse and fine runs together
# ---------------------------------------------------------------------

def _read_csv(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        o = {}
        for k, v in r.items():
            try:
                o[k] = float(v)
            except (TypeError, ValueError):
                o[k] = v
        out.append(o)
    return out


def fig_timing_all(movies, path):
    """one bar per movie and stage: range over interior cores, median marked"""
    fig, ax = plt.subplots(figsize=(9.6, 4.8), constrained_layout=True)
    keys = [("t_s1", "s1", "stages 1-2: cells of u", -0.0036), ("t_s2", "s2", "stages 1-2: fold", -0.0018),
            ("t_cos", "cos", "cells of cos(theta)", 0.0), ("t_s3", "s3", "stage 3: first crossing", 0.0018),
            ("t_s4", "s4", "stage 4: second crossing", 0.0036)]
    col = dict(SCOL); col["cos"] = "#555555"
    lo_all = []
    for mv in movies:
        rs = mv["rows"]
        fine = mv["cadence"] == "fine"
        for k, sk, _, dx_ in keys:
            v = np.array([float(r[k]) for r in rs], float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            x0 = mv["mu"] + dx_
            ax.plot([x0, x0], [v.min(), v.max()], "-", color=col[sk], lw=2.2, alpha=0.55, solid_capstyle="butt")
            ax.plot(x0, np.median(v), "D" if fine else "o", ms=5, color=col[sk], mfc="white" if fine else col[sk],
                    mew=1.3, zorder=3)
            lo_all.append(v.min())
        ax.text(mv["mu"], 0.985, "%d" % mv["n_distinct"], transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=7, color="0.35")
    ax.set_yscale("symlog", linthresh=0.02)
    ax.set_ylim(-0.002, None)
    ax.set_xlabel("mu"); ax.set_ylabel("time the stage begins at a core")
    ax.legend(handles=[Line2D([], [], marker="o", ls="-", lw=2, color=col[sk], label=lab) for _, sk, lab, _ in keys]
              + [Line2D([], [], marker="o", ls="none", color="k", label="dt = 0.2 (median; bar: range over cores)"),
                 Line2D([], [], marker="D", ls="none", color="k", mfc="white", label="dt = 0.025")],
              fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.set_title("when each stage begins, every movie (one director width, 0.4); numbers on top: distinct cases per "
                 "movie\n(coarse cores repeat every third core because dx / L = 3/64; fold timing is built into the "
                 "test from about mu 0.88)", fontsize=8)
    fig.savefig(path.with_suffix(".png"), dpi=170); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)


def fig_xtrack_all(movies, path, args):
    """the travelling X against time, one color per angle; u left, cos(theta) right; median over interior cores
    thick, each core thin"""
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.4), sharey=True, constrained_layout=True)
    cm = plt.get_cmap("plasma")
    mus = sorted(set(mv["mu"] for mv in movies))
    cof = lambda m: cm(0.08 + 0.8 * (mus.index(m) / max(len(mus) - 1, 1)))      # noqa: E731
    hand = []
    tmin, tmax = np.inf, 0.0
    for mv in movies:
        if mv["track"] is None:
            continue
        ls = "--" if mv["cadence"] == "fine" else "-"
        for ax, key in ((axs[0], "x_X_u_L"), (axs[1], "x_X_cos_L")):
            T = mv["track"]
            Ms = []
            for c in mv["cores"]:
                rr = [r for r in T if int(r["core"]) == c]
                t = np.array([r["t"] for r in rr], float); xv = np.array([r[key] for r in rr], float)
                if np.isfinite(xv).any():
                    tmin = min(tmin, float(np.min(t[np.isfinite(xv) & (t > 0)])) if (np.isfinite(xv) & (t > 0)).any()
                               else tmin)
                    tmax = max(tmax, float(t.max()))
                ax.plot(t, xv, ls, color=cof(mv["mu"]), lw=0.6, alpha=0.35, drawstyle="steps-mid")
                Ms.append(xv)
            if Ms:
                med = np.nanmedian(np.array(Ms), axis=0) if np.isfinite(np.array(Ms)).any() else None
                if med is not None:
                    ax.plot(t, med, ls, color=cof(mv["mu"]), lw=2.0, drawstyle="steps-mid")
        hand.append(Line2D([], [], color=cof(mv["mu"]), ls=ls, lw=2,
                           label="mu %.2f (%s)" % (mv["mu"], "dt 0.025" if mv["cadence"] == "fine" else "dt 0.2")))
    for ax, name in zip(axs, ["X cell of u", "X cell of cos(theta)"]):
        ax.axhline(args.loop_L, color="#f39c12", ls="--", lw=1.0)
        ax.set_xscale("log")
        if np.isfinite(tmin) and tmax > tmin:
            ax.set_xlim(0.8 * tmin, 1.1 * tmax)
        ax.set_xlabel("time (log scale)", fontsize=8)
        ax.set_title(name, fontsize=8.5)
        ax.tick_params(labelsize=7)
    axs[0].set_ylabel("position along the line, from the core (L)", fontsize=8)
    axs[0].legend(handles=hand + [Line2D([], [], color="#f39c12", ls="--", label="loop radius")], fontsize=6.5,
                  loc="upper left")
    fig.suptitle("the travelling X: on-line X cells (|y| <= %.2g L) furthest along +x between %.2g and %.2g L from "
                 "the core; thick: median over interior cores, thin: each core" % (args.on_line_L, args.xt_lo,
                                                                                   args.xt_hi), fontsize=8)
    fig.savefig(path.with_suffix(".png"), dpi=170); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)


def combine(args):
    t0 = time.time()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    lines = [VERSION, "combine: " + args.combine, ""]
    movies = []
    for dname in [s for s in args.combine.split(",") if s]:
        dpath = Path(dname)
        cp = dpath / "mig_cores.csv"
        if not cp.exists():
            lines.append("LOUD: no mig_cores.csv in %s" % dpath); continue
        rows = [r for r in _read_csv(cp) if not int(r["edge_core"])]
        for stem in sorted(set(r["stem"] for r in rows)):
            rs = [r for r in rows if r["stem"] == stem]
            tp = dpath / stem / "mig_xtrack.csv"
            track = _read_csv(tp) if tp.exists() else None
            if track is None:
                lines.append("LOUD: %s has no mig_xtrack.csv (run v1.3 on it first); timing only" % stem)
            movies.append(dict(stem=stem, mu=float(rs[0]["mu"]), cadence=rs[0]["cadence"], rows=rs,
                               cores=[int(r["core"]) for r in rs], track=track, n_distinct=n_distinct(rs)))
    movies.sort(key=lambda m: (m["mu"], m["cadence"]))
    if not movies:
        lines.append("LOUD: nothing to combine")
    else:
        fig_timing_all(movies, out / "mig_timing_all")
        if any(mv["track"] is not None for mv in movies):
            fig_xtrack_all(movies, out / "mig_xtrack_all", args)
        glide = []
        for mv in movies:
            rs = mv["rows"]
            med = lambda k: float(np.nanmedian([float(r[k]) for r in rs])) if any(np.isfinite(float(r[k])) for r in rs) \
                else float("nan")                                            # noqa: E731
            g = dict(stem=mv["stem"], mu=mv["mu"], cadence=mv["cadence"], interior_cores=len(rs),
                     distinct_cases=mv["n_distinct"])
            for k in ONSET_KEYS + (["t_xt_birth", "xt_birth_L", "xt_final_L", "v_xt_half", "t_xt_pass_loop", "xt_lag_L"]
                                   if "xt_final_L" in rs[0] else []):
                g[k + "_median"] = med(k)
                v = [float(r[k]) for r in rs if np.isfinite(float(r[k]))]
                g[k + "_min"] = min(v) if v else float("nan")
                g[k + "_max"] = max(v) if v else float("nan")
            glide.append(g)
            lines.append("mu %.3f %s: %d interior cores, %d distinct cases; onsets (median, range over cores) "
                         "cells of u %s, fold %s, cos(theta) %s, stage 3 %s, stage 4 %s"
                         % (mv["mu"], mv["cadence"], len(rs), mv["n_distinct"], _rng([r["t_s1"] for r in rs], "%.4g"),
                            _rng([r["t_s2"] for r in rs], "%.4g"), _rng([r["t_cos"] for r in rs], "%.4g"),
                            _rng([r["t_s3"] for r in rs], "%.4g"), _rng([r["t_s4"] for r in rs], "%.4g")))
            if "xt_final_L" in rs[0]:
                lines.append("   travelling X: " + glide_line(rs))
        write_csv(glide, out / "mig_glide_all.csv")
    lines += ["", "done in %.0f s" % (time.time() - t0)]
    (out / "mig_combine_notes.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("outputs in %s" % out)
    return 0


def main(args):
    if args.combine:
        return combine(args)
    t0 = time.time()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    lines = [VERSION, "args: " + json.dumps(vars(args)), ""]
    files = F.select_files(args.npz, args.mus) if args.npz else []
    excl = [s for s in args.exclude.split(",") if s]
    files = [f for f in files if not any(s in Path(f).name for s in excl)]
    if not files:
        lines.append("LOUD: no movie matched --npz %s --mus %s" % (args.npz, args.mus)); print(lines[-1])
    cores = []
    for p in files:
        run_movie(p, args, out, lines, cores)
    write_csv(cores, out / "mig_cores.csv")
    if cores:
        fig_timing([r for r in cores if not r["edge_core"]] or cores, out / "mig_timing")
    summarize(cores, lines)
    lines += ["", "done in %.0f s" % (time.time() - t0)]
    (out / "mig_notes.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-12:]))
    print("outputs in %s" % out)
    return 0


class _Args:
    """defaults; mirrored by the CLI below. Keys shared with core_track_time_series keep its values."""
    npz = ""                       # movie files and/or folders
    mus = ""                       # mu values to keep ("" keeps every file)
    exclude = "last10,f4-84"
    outdir = str(Path("results") / "nucleation_migration")
    mu_min = 0.82                  # scope: dislocation angles only (Ed, 28 Oct)
    width = 0.4                    # THE director width: mg's sigma_dir, used for k and for every cell detection
    core = -1                      # exemplar core (-1: the first to reach one turn); -1 also keeps crops of every core
    with_phi = 0                   # 1: also the fold of grad(phi_LS), as a CSV column only
    fold_on = 0.05
    half_x_L = 1.0                 # crops and space-time diagram: half-width along the line, L
    half_y_L = 0.8
    band_y_L = 0.5                 # space-time diagram: cells within this distance of the line, L
    A_lo = 0.3
    A_hi = 1.15
    A_levels = [0.5, 0.6, 0.7, 0.8]
    grid_cols = 5
    sep_log_t = 1
    anim = 1
    anim_stride = 1
    anim_dpi = 110
    fps = 4
    hold_frames = 8
    reveal = 1                     # animation: the space-time diagram fills in up to the current time
    time_scale = "sqrt"            # space-time time axis: linear | sqrt | symlog
    symlog_lin = 0.5               # symlog: linear below this time
    t_max = 0.0                    # space-time diagrams: last time drawn (0: all)
    sep_fig = 0                    # 1: also the v1.2 separation figure (nearest V-X distance)
    xt_lo = 0.0                    # travelling X: on-line X cells between xt_lo and xt_hi L from the core (+x side)
    xt_hi = 0.95
    combine = ""                   # "dirA,dirB": combine mode, reads earlier v1.3 outputs, reads no movie
    edge_frac = 0.4                # a core is an edge core when |x_core - x_mid| + half_x_L L exceeds edge_frac Lx
    # shared with core_track_time_series (same values; sigma_dir and sigma_det are replaced by width)
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
