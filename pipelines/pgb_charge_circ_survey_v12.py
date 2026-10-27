"""pgb_charge_circ_survey_v12.py  v1.2  (26 Oct 2026)   director charge, circulation and amplitude dips
along a zigzag phase grain boundary

Plain terms. For each boundary (one frame per file, the last by default) the
survey finds three things and nothing else by default: the director charge
(the half-turn cells of the stripe orientation, with their sign), the
amplitude dips of the mg amplitude, and the circulation of the measured
wavevector about each dip. It draws them on the pattern and on the amplitude
and labels the boundary by the rule agreed with Ed on 25 Oct:

  knee          no director charge;
  intermediate  director charge, and no dip on the line reaches a circulation
                of wind_hi turns (0.9, the "wound" threshold of the
                winding-times figure);
  dislocation   director charge, and at least one dip on the line reaches it.

Notation (the advisor's): T is the twist, the director charge, in half-turns
(a V cell T = -1, an X cell T = +1, found plaquette by plaquette from the
doubled director angle). Gamma = closed integral of k . dx about a loop,
reported in turns (Gamma / 2 pi); it is a real number. The survey reports
|Gamma|, because mg's k carries a sign only up to the director.

Distances are in L = pi / k1, k1 = sqrt(1 - mu^2), from the line y = y_row;
the band is |y - y_row| < band_L L; the window is the inner 80 percent of the
box.

Measures per file:
  (1) director charge: the frozen t1c.detect_defects (Riesz director at
      sigma_det, plaquette winding of the doubled angle) on u x taper; every
      nonzero cell in the band and window with its sign; also counted at
      sigma_det2. Cells per L, the median distance of the V cells and of the
      X cells from the line, the median V-X and X-X separations.
  (2) amplitude dips: the records' find_cores (regions where mg's A falls
      below core_frac of its window median, merged within 0.5 L); per dip its
      distance from the line (on the line within on_line_L), its depth (the
      minimum of A / A_far within 0.25 L), and its distance to the nearest V
      cell and the nearest X cell.
  (3) circulation |Gamma| on a circle about each dip: radius loop_L L (0.5 L,
      the records'), or with --loop_lam > 0 a fixed number of far-field stripe
      wavelengths (the loop-size control, which now moves the circulation
      loop itself). At the default radius this is exactly the records'
      loop_circulation, checked in every file (circ_check_maxdiff).
  (4) amplitude: mg's A at the V cells and along the line (median and
      minimum, over A_far).
  (5) a descriptive map for the domain figure: |Gamma| for loops centred on a
      grid over the band. Not summarised: below mu of about 0.87 a 0.5 L loop
      is narrower than a stripe wavelength and its reading depends on where
      it sits.
  (6) line energy E_GB when the file carries an energy density or is an
      energy_vs_angle_zigzag cache record (read or integrated from the stored
      density; nothing about the energy is tested).
  (7) closure of each circulation loop: whether the sign of k, carried
      around the loop by continuity, returns to its start. An open loop
      encloses an odd number of half-turn cells and its |Gamma| is not a
      clean integer.
  (8) the director-charge width scan: cells per L at each width in
      --sigma_scan.
  (9) a dip magnitude at every angle: per column of the band, the minimum of
      mg A / A_far and its distance from the line; A_band_min is the deepest
      (with its distance), A_colmin_med the median over columns.
  (10) the lowest point of each dip: where in its patch A is smallest, its
      value over A_far, its distance from the line and from the patch centre.
  (11) the circulation map summarised at every angle: the largest |Gamma|
      over the grid loops whose sign closes, its distance from the line, and
      the median over those loops.

Optional, off by default (each restores exactly the pgb_marker_survey v1.2
measure, imported from it):
  --with_winding 1    the loop twist and the integer phase winding n on each
                      dip loop (the defect gallery's rule), the odd-twist marks,
                      and the columns dip_odd_twist_frac, n_dips_on_n1.
  --with_rowdemod 1   the x-local row demodulation |c|: its panel, montage row
                      and cuts, and its beat-node trough columns.
  --show_beat_node 1  the beat-node lines y = +- pi / (2 mu) on the figures.

Outputs under --outdir (default results/pgb_charge_circ_survey/, anchored to
this script's folder when relative): per file <stem>/markers.json,
columns.csv, circ_map.npz, fig_domain.png, fig_markers.png; for the set
survey_table.csv, survey_table.md, fig_montage.png, fig_montage_all.pdf,
fig_positions_vs_mu.png, fig_magnitudes_vs_mu.png, fig_energy_vs_mu.png,
report.txt, config.json. survey_file and the figure functions are importable
(large_sweep_charge_circ_v12.py uses them).

Imports the shared loaders and loop arithmetic from pgb_marker_survey.py
(v1.2, kept beside it) and the frozen method layer through it; nothing frozen
is edited. Python 3.8 compatible.

Changelog
  v1.2  (26 Oct 2026; new file name, v1.1 kept) after Ed's reading of the
        v1.1 figures and tables. (1) The lowest point of each dip: the dips
        are found as before (connected patches below core_frac of the window
        median, reduced to the patch centroid and merged within 0.5 L, the
        records' rule, so the circulation loops stay where the records put
        them), and now each dip also records where its patch is lowest and
        how low, from the same patches (find_dip_patches; it reproduces the
        records' find_cores centroids, checked in every file and flagged if
        not). Columns dip_min_* and the per-dip fields min_x, min_y,
        min_off_L, min_depth, min_to_centroid_L; panel (b) of the marker
        figure marks each lowest point with a cyan cross. (2) The circulation
        map summarised at every angle, knees included: map_circ_max_closed
        (the largest |Gamma| over the grid loops whose sign closes),
        map_circ_max_closed_off_L (that loop's distance from the line) and
        map_circ_med_closed. In v1.1 the circulation was summarised only
        about dips, so knees (no dips) had no value; on the v1.1 large sweep
        every map loop closed at every knee angle. (3) Axis labels say what
        they measure: position along the line and distance from the line,
        both in L. Later on 26 Oct, after the first v1.2 run: the axis
        labels of the cut panels (c) and (d) of fig_markers were shortened
        (they ran into each other); no number changes.
  v1.1  (25 Oct 2026; new file name, v1.0 kept) after Ed's v1.0 run of 25 Oct
        (CHARGE_CIRC_LOCAL_RESULTS_1025.md). (1) Closure of the circulation:
        loop_grid carries the sign of k around each loop by continuity; the
        flag says whether that sign comes back to its start. Where it does
        not, the loop encloses an odd number of half-turn cells and |Gamma|
        is not a clean integer (on the v1.0 run's 0.6-wavelength loops, 1.05
        to 1.5 turn where v1.2 found odd twist, one turn from 0.88 up where it
        found none). Columns dip_open_frac_on/off and map_open_frac; open
        loops are drawn dotted. (2) Width scan of the director charge
        (--sigma_scan, default 0.15 to 0.4): v1.0 found the charge onset at
        0.69 at sigma_det 0.3 and at 0.62 at 0.2; the scan shows how the onset
        moves with width. Columns charge_per_L_sd<width>. (3) A continuous
        amplitude-dip magnitude at every angle, knees included: per column
        of the band the minimum of mg A / A_far (as the green dots of the
        figures), summarised as A_band_min (deepest), its distance from the
        line, and the column median; the thresholded dips (below 0.70) start
        only at 0.70. (4) Readable marker figure: the pattern is muted under
        the markers and the legend sits below the figure; the band panel of
        the domain figure keeps circles round. (5) survey_file accepts a
        preloaded frame and --map 0 skips the loop map (the time-series
        script uses both). Later the same day, before any run of record:
        the per-column minima of A (green dots) are off by default
        (--show_colmin 1 draws them) and figure labels say "amplitude A"
        throughout; no number changes.
  v1.0  (25 Oct 2026) from pgb_marker_survey v1.2, after Ed's v1.2 run of 25
        Oct (PGB_MARKER_SURVEY_FINDINGS_1025.md). The regime rule moves from
        the integer winding to the circulation: on that run's t = 200 tables
        both rules give the same regime at every one of the 59 angles (the
        borderline 0.82 has one on-line dip at 0.98 turn). Winding, row
        demodulation and beat node become options, off by default. New
        columns: the dips' distances to the nearest V and X cell (by hand on
        25 Oct: 0.26 L at mu 0.70 closing to 0.10 L at 0.80 to 0.82), the
        largest |Gamma| on and off the line, the count and fraction of
        on-line dips at or above wind_hi. --loop_lam now moves the
        circulation loop (in v1.2 it moved only the twist and winding loop).
        Sandbox check on SH stand-ins and the planted knee and PN rows (not
        of record): default and all-options runs complete; with every option
        on, the v1.2 columns are reproduced.
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
import pgb_marker_survey as PMS                                             # noqa: E402
from pgb_marker_survey import (load_frame, _line_energy, loop_grid, loop_phase_winding, column_minima,  # noqa: E402
                               expand_inputs, montage_select, _crop, _lx, _ly, _regime_spans, _deg_axis)
from pgb_marker_survey import estimate, find_cores, t1c, loop_circulation  # noqa: E402
from scipy.ndimage import label as nd_label, center_of_mass, minimum, minimum_position  # noqa: E402
import matplotlib                                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages                        # noqa: E402
from matplotlib.patches import Circle, Rectangle, Patch                    # noqa: E402
from matplotlib import cm, colors as mcolors                                # noqa: E402

VERSION = "pgb_charge_circ_survey v1.2 (on %s)" % PMS.VERSION
C_POS, C_NEG, C_CUT1, C_CUT2, MUTED = PMS.C_POS, PMS.C_NEG, PMS.C_CUT1, PMS.C_CUT2, PMS.MUTED
REGIME_COL = {"knee": "#e9e7e2", "intermediate": "#f6e3b4", "dislocation": "#cfe0f5"}
CMAP_G = matplotlib.colormaps["viridis"] if hasattr(matplotlib, "colormaps") else cm.get_cmap("viridis")
NORM_G = mcolors.Normalize(vmin=0.0, vmax=1.0)


def scan_widths(args):
    """the director-charge widths of the scan (--sigma_scan, comma-separated; empty for none)"""
    return [float(v) for v in str(getattr(args, "sigma_scan", "")).split(",") if v.strip()]


def _wtag(w):
    return ("%.3f" % w).rstrip("0").rstrip(".").replace(".", "p")


def _med(v):
    return float(np.median(v)) if len(v) else float("nan")


def _max(v):
    return float(np.max(v)) if len(v) else float("nan")


def _nearest(ax_, ay_, px, py):
    """distance from each (ax_, ay_) to the nearest of (px, py)"""
    if len(ax_) == 0 or len(px) == 0:
        return np.full(len(ax_), np.nan)
    return np.min(np.hypot(np.asarray(ax_)[:, None] - px[None, :], np.asarray(ay_)[:, None] - py[None, :]), axis=1)


def find_dip_patches(A, x, y, win, frac, merge_dist):
    """the records' find_cores (mg_zigzag_theta_t v1.5), step for step, also returning where each dip's patch is
    lowest: connected regions of A < frac * median(A[win]), labelled on the whole array, kept when the centroid lies in
    the window, merged when centroids are closer than merge_dist (the merged centroid is the mean of the members', as
    in find_cores; the merged lowest point is the lowest of the members'). Returns xc, yc, xmin, ymin, Amin."""
    ref = np.median(A[win])
    lab, n = nd_label(A < frac * ref)
    e = np.zeros(0)
    if n == 0:
        return e, e, e, e, e
    idx = list(range(1, n + 1))
    cms = center_of_mass(np.ones_like(A), lab, idx)
    iy = np.array([c[0] for c in cms]); ix = np.array([c[1] for c in cms])
    mins = np.array(minimum(A, lab, idx), float)
    mpos = minimum_position(A, lab, idx)
    my_ = np.array([p[0] for p in mpos], float); mx_ = np.array([p[1] for p in mpos], float)
    keep = win[np.clip(np.round(iy).astype(int), 0, A.shape[0] - 1), np.clip(np.round(ix).astype(int), 0, A.shape[1] - 1)]
    ix, iy, mins, mx_, my_ = ix[keep], iy[keep], mins[keep], mx_[keep], my_[keep]
    xc, yc = np.interp(ix, np.arange(len(x)), x), np.interp(iy, np.arange(len(y)), y)
    xm, ym = np.interp(mx_, np.arange(len(x)), x), np.interp(my_, np.arange(len(y)), y)
    if len(xc) > 1 and merge_dist > 0:
        order = np.argsort(xc)
        xc, yc, xm, ym, mins = xc[order], yc[order], xm[order], ym[order], mins[order]
        groups, cur = [], [0]
        for j in range(1, len(xc)):
            if np.hypot(xc[j] - xc[cur[0]], yc[j] - yc[cur[0]]) < merge_dist:
                cur.append(j)
            else:
                groups.append(cur); cur = [j]
        groups.append(cur)
        low = [g[int(np.argmin(mins[g]))] for g in groups]
        xc = np.array([xc[g].mean() for g in groups]); yc = np.array([yc[g].mean() for g in groups])
        xm, ym, mins = xm[low], ym[low], mins[low]
    return xc, yc, xm, ym, mins


def survey_file(path, args, outroot, lines, quiet_figs=False, R=None, write=True):
    t0 = time.time()
    R = load_frame(path, args) if R is None else R
    u2, x, y, md = R["u2"], R["x"], R["y"], R["md"]
    stem = R["stem"] + (("_f%d" % R["frame"]) if getattr(args, "tag_frame", 0) else "")
    mu = float(md.get("mu", args.mu))
    k1t = float(np.sqrt(1 - mu ** 2)); L = np.pi / k1t
    node = k1t / (2 * mu)                                   # beat node pi/(2 mu) in units of L (options only)
    ny, nx = u2.shape
    dx = float(x[1] - x[0]); dy = float(y[1] - y[0])
    Lx, Ly = nx * dx, ny * dy
    X, Y = np.meshgrid(x, y)
    jr = int(md.get("gb_row_in_crop", ny // 2)); y_row = float(y[jr])
    xm, ym_ = float(x.mean()), float(y.mean())
    win = (np.abs(X - xm) < 0.4 * Lx) & (np.abs(Y - ym_) < 0.4 * Ly)
    flags = []
    if (y_row - args.band_L * L < ym_ - 0.4 * Ly) or (y_row + args.band_L * L > ym_ + 0.4 * Ly):
        flags.append("band extends beyond the window in y")
    taper = t1c.edge_taper(x, y, frac=args.taper_frac)

    # mg estimator
    e = estimate(u2, Lx, Ly, args.sigma_dir, None, taper)
    A = e["A"]
    K1r, K2r = e["k1"], e["k2"]
    d1, d2 = e["n"][0], e["n"][1]
    far = (np.abs(Y - y_row) > args.far_from * L) & win
    if far.sum() > 50:
        A_far = float(np.median(A[far]))
    else:
        A_far = float(np.median(A[win])); flags.append("mg A_far from the window median (no far field)")

    # (1) director charge, signed, at two widths
    det = t1c.detect_defects(u2 * taper, x, y - y_row, Lx, Ly, sigma_det=args.sigma_det, band_y=args.band_L * L)
    qmap = det["q"]; xcp, ycp = det["xc"], det["yc"]
    Jq, Iq = np.nonzero(qmap)
    px, py, pq = xcp[Iq], ycp[Jq] + y_row, qmap[Jq, Iq].astype(int)
    inw = (np.abs(px - xm) < 0.4 * Lx) & (np.abs(py - ym_) < 0.4 * Ly)
    px, py, pq = px[inw], py[inw], pq[inw]
    inb = np.abs(py - y_row) < args.band_L * L
    bx, by, bq = px[inb], py[inb], pq[inb]
    line_len_L = 0.8 * Lx / L
    det2 = t1c.detect_defects(u2 * taper, x, y - y_row, Lx, Ly, sigma_det=args.sigma_det2, band_y=args.band_L * L)
    q2 = [(a, b + y_row, c) for a, b, c in det2["charges"] if abs(a - xm) < 0.4 * Lx and abs(b + y_row - ym_) < 0.4 * Ly]
    q2_abs = int(sum(abs(c) for _, _, c in q2)); q2_net = int(sum(c for _, _, c in q2))
    scan = {}
    for sw in scan_widths(args):
        if abs(sw - args.sigma_det) < 1e-9:
            ds = det
        elif abs(sw - args.sigma_det2) < 1e-9:
            ds = det2
        else:
            ds = t1c.detect_defects(u2 * taper, x, y - y_row, Lx, Ly, sigma_det=sw, band_y=args.band_L * L)
        scan[sw] = int(sum(abs(c) for a, b, c in ds["charges"] if abs(a - xm) < 0.4 * Lx and abs(b + y_row - ym_) < 0.4 * Ly))
    Vm, Xm = bq < 0, bq > 0
    v_off = np.abs(by[Vm] - y_row) / L; x_off = np.abs(by[Xm] - y_row) / L
    vx_sep = _nearest(bx[Vm], by[Vm], bx[Xm], by[Xm]) / L if Xm.any() else np.zeros(0)
    xx_sep = np.array([np.min(np.hypot(np.delete(bx[Xm], i) - a, np.delete(by[Xm], i) - b))
                       for i, (a, b) in enumerate(zip(bx[Xm], by[Xm]))]) / L if Xm.sum() > 1 else np.zeros(0)

    # (2) amplitude dips and (3) the circulation about them
    wavelength = 2 * np.pi / max(float(np.median(np.hypot(K1r, K2r)[far])) if far.sum() > 50 else 1.0, 1e-6)
    r = args.loop_lam * wavelength if args.loop_lam > 0 else args.loop_L * L
    xd_, yd_, xmn_, ymn_, Amn_ = find_dip_patches(A, x, y, win, args.core_frac, 0.5 * L)
    xr_, yr_ = find_cores(A, x, y, win, args.core_frac, merge_dist=0.5 * L)
    xr_, yr_ = np.asarray(xr_, float), np.asarray(yr_, float)
    if len(xr_) != len(xd_) or (len(xd_) and max(np.max(np.abs(xr_ - xd_)), np.max(np.abs(yr_ - yd_))) > 1e-9):
        flags.append("LOUD: dip centroids differ from the records' find_cores (%d against %d dips)" % (len(xd_), len(xr_)))
    min_off_ = np.abs(ymn_ - y_row) / L
    min_dep_ = Amn_ / A_far
    min_to_c_ = np.hypot(xmn_ - xd_, ymn_ - yd_) / L
    if len(xd_):
        circ_d, cl_d = loop_grid(K1r, K2r, x, y, xd_, yd_, r, args.nang)
        circ_d = np.abs(circ_d); cl_d = np.asarray(cl_d, bool)
        rec_circ = np.array([abs(loop_circulation(K1r, K2r, x, y, a, b, 0.5 * L)) for a, b in zip(xd_, yd_)])
        mine05, _ = loop_grid(K1r, K2r, x, y, xd_, yd_, 0.5 * L, 181)
        chk = float(np.max(np.abs(np.abs(mine05) - rec_circ)))
    else:
        circ_d = np.zeros(0); cl_d = np.zeros(0, bool); rec_circ = np.zeros(0); chk = 0.0
    d_off = np.abs(yd_ - y_row) / L
    on_d = d_off <= args.on_line_L
    hi_d = circ_d >= args.wind_hi
    rq = max(1, int(round(0.25 * L / dx)))

    def depth_at(a, b):
        i = int(np.argmin(np.abs(x - a))); j = int(np.argmin(np.abs(y - b)))
        return float(A[max(0, j - rq):j + rq + 1, max(0, i - rq):i + rq + 1].min() / A_far)
    dep_d = np.array([depth_at(a, b) for a, b in zip(xd_, yd_)])
    toV = _nearest(xd_, yd_, bx[Vm], by[Vm]) / L
    toX = _nearest(xd_, yd_, bx[Xm], by[Xm]) / L

    # optional: the integer winding n and the loop twist (v1.2)
    if args.with_winding and len(xd_):
        tw_d, n_d = loop_phase_winding(e["z"], d1, d2, x, y, xd_, yd_, r, args.nang_phase)
    else:
        tw_d, n_d = np.zeros(len(xd_), int), np.full(len(xd_), np.nan)
    n1_d = (tw_d % 2 == 0) & (np.abs(np.nan_to_num(n_d)) >= 0.5) if args.with_winding else np.zeros(len(xd_), bool)
    odd_d = (tw_d % 2 != 0) if args.with_winding else np.zeros(len(xd_), bool)

    # (4) amplitude at the cells and along the line; column minima of mg A (figure only)
    A_at_V = np.array([A[int(np.argmin(np.abs(y - b))), int(np.argmin(np.abs(x - a)))] for a, b in zip(bx[Vm], by[Vm])]) / A_far
    ix = np.where(np.abs(x - xm) < 0.4 * Lx)[0]
    A_line = A[jr, ix] / A_far
    bmin = args.band_min if args.band_min > 0 else max(1.0, 1.5 * node)
    jb = np.where(np.abs(y - y_row) < bmin * L)[0]
    yg_min, dg = column_minima(A[:, ix], y, jb, y_row, A_far, args.min_contrast)
    ib = int(np.argmin(dg))
    A_band_min = float(dg[ib])
    A_band_min_off = float(abs(yg_min[ib]) / L) if np.isfinite(yg_min[ib]) else float("nan")
    okg = np.isfinite(yg_min)
    A_colmin_off_med = float(np.median(np.abs(yg_min[okg]))) / L if okg.any() else float("nan")
    xcol = x[ix]

    # optional: the row demodulation |c| and its beat-node trough (v1.2)
    Cmap, yc_min, dc, c_node, cdepth, A_far_dm, k1f = None, None, None, float("nan"), float("nan"), float("nan"), float("nan")
    if args.with_rowdemod:
        from knee_row_demod_survey import fit_k1, row_demod, profile_numbers
        from knee_amplitude_sweep import local_row_demod
        k1f = fit_k1(u2, x, k1t)
        c_rows, _xinv = row_demod(u2, x, k1f)
        P = profile_numbers(y, jr, c_rows, k1f, mu, (args.far_from * L, (args.far_from + 2.0) * L))
        A_far_dm = P["A_far"]
        if not np.isfinite(A_far_dm):
            sel = np.abs(y - y_row) > 2.0 * L
            A_far_dm = float(np.median(np.abs(c_rows[sel]))) if sel.any() else float(np.median(np.abs(c_rows)))
            flags.append("demod A_far from rows beyond 2 L (no rows at far_from)")
        Cmap = local_row_demod(u2, x, k1f, args.loc_L * L) / A_far_dm
        yc_min, dc = column_minima(Cmap[:, ix] * A_far_dm, y, jb, y_row, A_far_dm, args.min_contrast)
        okc = np.isfinite(yc_min)
        c_node = float(np.nanmedian(np.abs(yc_min[okc]))) / L if okc.any() else float("nan")
        cdepth = float(np.median(dc))

    # (5) the descriptive loop map (domain figure only)
    step = args.grid_L * L
    cxs = np.arange(xm - 0.4 * Lx + r, xm + 0.4 * Lx - r + 1e-9, step)
    ylo = max(y_row - args.band_L * L, ym_ - 0.4 * Ly + r); yhi = min(y_row + args.band_L * L, ym_ + 0.4 * Ly - r)
    cys = np.arange(ylo, yhi + 1e-9, step)
    CX, CY = np.meshgrid(cxs, cys)
    if not args.map:                                         # the time-series script skips the map
        CX, CY = np.zeros((0, 0)), np.zeros((0, 0))
    if CX.size:
        circ_g, kc_g = loop_grid(K1r, K2r, x, y, CX.ravel(), CY.ravel(), r, args.nang)
        circ_g = np.abs(circ_g).reshape(CX.shape); kc_g = np.asarray(kc_g, bool).reshape(CX.shape)
    else:
        circ_g, kc_g = np.zeros(CX.shape), np.ones(CX.shape, bool)
    if args.with_winding and CX.size:
        tw_g, n_g = loop_phase_winding(e["z"], d1, d2, x, y, CX.ravel(), CY.ravel(), r, args.nang_phase)
        tw_g, n_g = tw_g.reshape(CX.shape), n_g.reshape(CX.shape)
    else:
        tw_g, n_g = np.zeros(CX.shape, int), np.full(CX.shape, np.nan)

    # regime by charge and circulation
    has_charge = len(bq) > 0
    n_on, n_off = int(on_d.sum()), int((~on_d).sum())
    n_on_hi, n_off_hi = int((on_d & hi_d).sum()), int((~on_d & hi_d).sum())
    if not has_charge:
        regime = "knee"
        if n_on_hi or n_off_hi:
            flags.append("no charge, yet %d dips reach %.2g turn" % (n_on_hi + n_off_hi, args.wind_hi))
    elif n_on_hi == 0:
        regime = "intermediate"
        if n_off == 0 and n_on == 0:
            flags.append("no dips (charge without a %.0f percent dip)" % (100 * args.core_frac))
    else:
        regime = "dislocation"
        if n_on_hi < 0.9 * n_on:
            flags.append("partly: %d of %d dips on the line reach %.2g turn" % (n_on_hi, n_on, args.wind_hi))
    if n_off_hi:
        flags.append("%d of %d dips off the line reach %.2g turn" % (n_off_hi, n_off, args.wind_hi))
    if (q2_abs > 0) != has_charge:
        flags.append("charge at sigma_det %.2g: %d cells, at %.2g: %d (the widths disagree on whether charge is present)"
                     % (args.sigma_det, len(bq), args.sigma_det2, len(q2)))
    if 2 * r < wavelength:
        flags.append("dip loops (diameter %.2f) narrower than a stripe wavelength (%.2f)" % (2 * r, wavelength))

    # (6) energy
    E_GB, flat, esrc = float("nan"), float("nan"), R["energy_source"]
    if R["e2"] is not None:
        E_GB, flat, how = _line_energy(R["e2"], y, jr, args.edge_frac)
        esrc = "%s via %s" % (esrc, how)
    elif np.isfinite(R["E_GB_stored"]):
        E_GB, esrc = R["E_GB_stored"], "E_GB stored in the cache record"

    nan = float("nan")
    row = dict(stem=stem, mu=mu, alpha_deg=float(np.degrees(np.arcsin(mu))), t=R["t"], frame=R["frame"], L=L, nx=nx, ny=ny,
               layout=R["layout"], regime=regime, flags="; ".join(flags),
               n_V=int(Vm.sum()), n_X=int(Xm.sum()), charge_abs_per_L=float(np.abs(bq).sum() / line_len_L),
               charge_net=int(bq.sum()) if len(bq) else 0, charge_abs_per_L_sd2=q2_abs / line_len_L, charge_net_sd2=q2_net,
               V_offline_med_L=_med(v_off), V_offline_min_L=float(v_off.min()) if len(v_off) else nan,
               V_offline_max_L=float(v_off.max()) if len(v_off) else nan, X_offline_med_L=_med(x_off),
               VX_sep_med_L=_med(vx_sep), XX_sep_med_L=_med(xx_sep),
               n_dips=len(xd_), n_dips_on=n_on, n_dips_off=n_off,
               dips_off_med_L=_med(d_off[~on_d]),
               dip_depth_on_med=_med(dep_d[on_d]), dip_depth_off_med=_med(dep_d[~on_d]),
               dip_off_to_V_med_L=_med(toV[~on_d]), dip_off_to_X_med_L=_med(toX[~on_d]),
               dip_on_to_V_med_L=_med(toV[on_d]), dip_on_to_X_med_L=_med(toX[on_d]),
               dip_circ_on_med=_med(circ_d[on_d]), dip_circ_on_max=_max(circ_d[on_d]),
               dip_circ_off_med=_med(circ_d[~on_d]), dip_circ_off_max=_max(circ_d[~on_d]),
               n_dips_on_circ_hi=n_on_hi, n_dips_off_circ_hi=n_off_hi,
               frac_on_circ_hi=(n_on_hi / n_on) if n_on else nan,
               loop_r_L=r / L, loop_r_lambda=r / wavelength, loop_diameter_over_wavelength=2 * r / wavelength,
               A_at_V_med=_med(A_at_V), A_line_med=float(np.median(A_line)), A_line_min=float(np.min(A_line)),
               A_far_mg=A_far,
               rec_n_dips=int(len(xd_)), rec_n_wound=int(np.sum(rec_circ >= args.wind_hi)) if len(rec_circ) else 0,
               rec_circ_max=float(rec_circ.max()) if len(rec_circ) else 0.0, rec_charge_sum_abs=int(det["sum_abs"]),
               circ_check_maxdiff=chk, E_GB=E_GB, E_flatness=flat, energy_source=esrc)
    row.update(dip_open_frac_on=float(np.mean(~cl_d[on_d])) if n_on else nan,
               dip_open_frac_off=float(np.mean(~cl_d[~on_d])) if n_off else nan,
               map_open_frac=float(np.mean(~kc_g)) if kc_g.size else nan,
               A_band_min=A_band_min, A_band_min_off_L=A_band_min_off, A_colmin_med=float(np.median(dg)),
               A_colmin_off_med_L=A_colmin_off_med)
    row.update(dip_min_depth_on_med=_med(min_dep_[on_d]), dip_min_depth_off_med=_med(min_dep_[~on_d]),
               dip_min_off_on_med_L=_med(min_off_[on_d]), dip_min_off_off_med_L=_med(min_off_[~on_d]),
               dip_min_to_centroid_med_L=_med(min_to_c_), dip_min_to_centroid_max_L=_max(min_to_c_))
    if kc_g.size and kc_g.any():
        cg = np.where(kc_g, circ_g, -1.0)
        jm = np.unravel_index(int(np.argmax(cg)), cg.shape)
        row.update(map_circ_max_closed=float(cg[jm]), map_circ_max_closed_off_L=float(abs(CY[jm] - y_row) / L),
                   map_circ_med_closed=float(np.median(circ_g[kc_g])))
    else:
        row.update(map_circ_max_closed=nan, map_circ_max_closed_off_L=nan, map_circ_med_closed=nan)
    for sw in sorted(scan):
        row["charge_per_L_sd%s" % _wtag(sw)] = scan[sw] / line_len_L
    if args.with_winding:
        row.update(dip_odd_twist_frac=float(np.mean(odd_d)) if len(odd_d) else nan,
                   n_dips_on_n1=int((on_d & n1_d).sum()), n_dips_off_n1=int((~on_d & n1_d).sum()),
                   map_frac_odd_twist=float(np.mean(tw_g % 2 != 0)) if tw_g.size else nan)
    if args.with_rowdemod:
        row.update(node_L=node, cmin_offline_med_L=c_node,
                   cmin_over_node_med=c_node / node if np.isfinite(c_node) else nan, cmin_depth_med=cdepth,
                   A_far_demod=A_far_dm, k1_fit=k1f)
    row["elapsed_s"] = float(time.time() - t0)

    G = dict(stem=stem, mu=mu, L=L, node=node, x=x, y=y, xm=xm, y_row=y_row, u=u2, A=A / A_far, C=Cmap,
             win=(xm - 0.4 * Lx, xm + 0.4 * Lx, ym_ - 0.4 * Ly, ym_ + 0.4 * Ly), band_L=args.band_L,
             CX=CX, CY=CY, circ=circ_g, closed=kc_g, even=(tw_g % 2 == 0), r=r, loc=args.loc_L * L,
             bx=bx, by=by, bq=bq,
             dips=[dict(x=xd_[i], y=yd_[i], circ=circ_d[i], hi=bool(hi_d[i]), on=bool(on_d[i]), n1=bool(n1_d[i]), odd=bool(odd_d[i]),
                        closed=bool(cl_d[i]), depth=dep_d[i], min_x=xmn_[i], min_y=ymn_[i]) for i in range(len(xd_))],
             xcol=xcol, ix=ix, dc=dc, yc_min=yc_min, yg_min=yg_min, dg=dg, regime=regime, t=R["t"], jr=jr,
             n_on=n_on, n_on_hi=n_on_hi)
    if not write:
        return row, G
    od = outroot / stem; od.mkdir(parents=True, exist_ok=True)
    dips_json = []
    for i in range(len(xd_)):
        dj = dict(x=float(xd_[i]), y=float(yd_[i]), off_L=float(d_off[i]), on_line=bool(on_d[i]), depth=float(dep_d[i]),
                  circ=float(circ_d[i]), circ_hi=bool(hi_d[i]), circ_sign_closes=bool(cl_d[i]), circ_records_05L=float(rec_circ[i]),
                  to_V_L=float(toV[i]), to_X_L=float(toX[i]), min_x=float(xmn_[i]), min_y=float(ymn_[i]),
                  min_off_L=float(min_off_[i]), min_depth=float(min_dep_[i]), min_to_centroid_L=float(min_to_c_[i]))
        if args.with_winding:
            dj.update(twist=int(tw_d[i]), n=None if not np.isfinite(n_d[i]) else float(n_d[i]))
        dips_json.append(dj)
    with open(od / "markers.json", "w") as fh:
        json.dump(dict(row=row,
                       cells=[dict(x=float(a), y=float(b), q=int(c)) for a, b, c in zip(bx, by, bq)],
                       cells_sigma_det2=[dict(x=float(a), y=float(b), q=int(c)) for a, b, c in q2],
                       dips=dips_json, line_y=y_row, L=L), fh, indent=1)
    with open(od / "columns.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        head = ["x_over_L", "mgmin_y_over_L", "mgmin_depth", "A_line"] + (["cmin_y_over_L", "cmin_depth"] if args.with_rowdemod else [])
        wr.writerow(head)
        for i in range(len(ix)):
            vals = [(xcol[i] - xm) / L, yg_min[i] / L, dg[i], A_line[i]]
            if args.with_rowdemod:
                vals += [yc_min[i] / L, dc[i]]
            wr.writerow(["%.5f" % v for v in vals])
    np.savez_compressed(od / "circ_map.npz", cx=CX, cy=CY, circ=circ_g, sign_closes=kc_g, twist=tw_g, n=n_g, loop_r=r, line_y=y_row, L=L)

    if not quiet_figs:
        fig_domain(G, od / "fig_domain.png", args)
        fig_markers(G, od / "fig_markers.png", args)

    lines.append("%s: mu %.3f (%.1f deg, L %.2f), t %.2f, %s; regime %s%s" % (
        stem, mu, row["alpha_deg"], L, R["t"], R["layout"], regime, (" [FLAGS: %s]" % row["flags"]) if flags else ""))
    lines.append("   charge (sigma_det %.2g): %d V cells (T = -1), %d X cells (T = +1) in the band (net %+d, %.2f |T| per L); V %.2f L off"
                 " the line (range %.2f to %.2f), X %.2f L off; V-X %.2f L, X-X %.2f L; at sigma_det %.2g: %d cells (net %+d)" % (
                     args.sigma_det, row["n_V"], row["n_X"], row["charge_net"], row["charge_abs_per_L"], row["V_offline_med_L"],
                     row["V_offline_min_L"], row["V_offline_max_L"], row["X_offline_med_L"], row["VX_sep_med_L"], row["XX_sep_med_L"],
                     args.sigma_det2, len(q2), q2_net))
    lines.append("   dips: %d on the line (depth %.3f; |Gamma| median %.3f, max %.3f turn; %d at or above %.2g), %d off the line"
                 " (%.2f L off; %.2f L to the nearest V, %.2f L to the nearest X; depth %.3f; |Gamma| median %.3f, max %.3f turn);"
                 " loops r %.2f L = %.2f stripe wavelengths" % (
                     n_on, row["dip_depth_on_med"], row["dip_circ_on_med"], row["dip_circ_on_max"], n_on_hi, args.wind_hi,
                     n_off, row["dips_off_med_L"], row["dip_off_to_V_med_L"], row["dip_off_to_X_med_L"], row["dip_depth_off_med"],
                     row["dip_circ_off_med"], row["dip_circ_off_max"], r / L, r / wavelength))
    lines.append("   amplitude (over far): at the V cells %.3f; on the line median %.3f, min %.3f; band (column minima) deepest %.3f"
                 " at %.2f L off the line, median %.3f at %.2f L" % (
                     row["A_at_V_med"], row["A_line_med"], row["A_line_min"], A_band_min, A_band_min_off, row["A_colmin_med"],
                     A_colmin_off_med))
    lines.append("   lowest points of the dips: off the line %.2f L from it (depth %.3f), on the line %.2f L (depth %.3f); lowest point to"
                 " patch centre median %.2f L, max %.2f L; circulation map: largest |Gamma| on a closed loop %.3f turn, %.2f L from"
                 " the line" % (row["dip_min_off_off_med_L"], row["dip_min_depth_off_med"], row["dip_min_off_on_med_L"],
                                row["dip_min_depth_on_med"], row["dip_min_to_centroid_med_L"], row["dip_min_to_centroid_max_L"],
                                row["map_circ_max_closed"], row["map_circ_max_closed_off_L"]))
    lines.append("   circulation sign does not close on %s of the on-line and %s of the off-line dip loops; director charge per L by"
                 " width: %s" % (
                     ("%.0f%%" % (100 * row["dip_open_frac_on"])) if n_on else "none", ("%.0f%%" % (100 * row["dip_open_frac_off"])) if n_off else "none",
                     ", ".join("%.2g: %.2f" % (sw, scan[sw] / line_len_L) for sw in sorted(scan))))
    if args.with_winding:
        lines.append("   [winding] dip loops with odd twist %.0f%%; dips with n = 1: %d on the line, %d off" % (
            100 * row["dip_odd_twist_frac"] if np.isfinite(row["dip_odd_twist_frac"]) else nan, row["n_dips_on_n1"], row["n_dips_off_n1"]))
    if args.with_rowdemod:
        lines.append("   [row demodulation] |c| beat-node trough %.2f L off the line (%.2f of the node), depth %.3f" % (
            c_node, row["cmin_over_node_med"], cdepth))
    lines.append("   records' convention (0.5 L loops): %d dips, %d at or above %.2g turn, max %.3f; loop code against loop_circulation:"
                 " max difference %.1e; E_GB %s (flatness %s; %s); %.1f s" % (
                     len(xd_), row["rec_n_wound"], args.wind_hi, row["rec_circ_max"], chk,
                     ("%.5g" % E_GB) if np.isfinite(E_GB) else "n/a", ("%.3f" % flat) if np.isfinite(flat) else "n/a", esrc,
                     row["elapsed_s"]))
    return row, G


# ---------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------

def _loop_style(dp, args):
    """edge colour by |Gamma|; line style: solid at or above wind_hi, dashed below, dotted where the sign of k does not
    close around the loop (or, with --with_winding, by n and the twist)"""
    col = CMAP_G(NORM_G(min(dp["circ"], 1.0)))
    if args.with_winding:
        ls = "-" if dp["n1"] else (":" if dp["odd"] else "--")
    else:
        ls = ":" if not dp.get("closed", True) else ("-" if dp["hi"] else "--")
    return col, ls


def draw_markers(ax, G, args, curve="mg", legend=False, on_light=False, show_min=False):
    """V and X cells, the dips with their loops (colour: |Gamma|; solid at or above wind_hi), the column minima,
    the line, a 1 L bar; the beat nodes with --show_beat_node; axes in units of L about (xm, y_row)"""
    hx = args.crop_x_L / 2
    ink = "k" if on_light else "w"
    ax.axhline(0, color=ink, lw=0.8, ls="--", label="the PGB line")
    if args.show_beat_node:
        for s in (1, -1):
            ax.axhline(s * G["node"], color="#6fd3e8", lw=0.8, ls=":")
    pos, neg = G["bq"] > 0, G["bq"] < 0
    ax.scatter(_lx(G, G["bx"][pos]), _ly(G, G["by"][pos]), marker="^", s=24, c=C_POS, edgecolors="k", linewidths=0.3, zorder=6,
               label="X cell (T = +1)")
    ax.scatter(_lx(G, G["bx"][neg]), _ly(G, G["by"][neg]), marker="v", s=24, c=C_NEG, edgecolors="k", linewidths=0.3, zorder=6,
               label="V cell (T = -1)")
    for dp in G["dips"]:
        col, ls = _loop_style(dp, args)
        ax.add_patch(Circle((_lx(G, dp["x"]), _ly(G, dp["y"])), G["r"] / G["L"], fill=False, zorder=5, ec=col, ls=ls, lw=1.3))
        ax.plot(_lx(G, dp["x"]), _ly(G, dp["y"]), "o", ms=4, mfc=col, mec="k", mew=0.3, zorder=7)
    if show_min:
        for dp in G["dips"]:
            ax.plot(_lx(G, dp["min_x"]), _ly(G, dp["min_y"]), "+", color="#35e0f0", ms=8, mew=1.6, zorder=8)
    ax.plot([], [], "o", ms=4, mfc=CMAP_G(0.6), mec="k", mew=0.3, label="amplitude dip (patch centre); its loop coloured by |Gamma|")
    ax.plot([], [], "-", color="k", lw=1.3, label="loop with |Gamma| at or above %.2g turn" % args.wind_hi)
    ax.plot([], [], "--", color="k", lw=1.3, label="loop below %.2g turn" % args.wind_hi)
    if not args.with_winding:
        ax.plot([], [], ":", color="k", lw=1.3, label="loop where the sign of k does not close")
    if curve in ("both", "mg") and args.show_colmin:
        ax.plot(_lx(G, G["xcol"]), G["yg_min"] / G["L"], ".", color=C_CUT2, ms=2.0, zorder=4, label="column minimum of amplitude A")
    if curve in ("both", "c") and G["yc_min"] is not None:
        ax.plot(_lx(G, G["xcol"]), G["yc_min"] / G["L"], ".", color=C_CUT1, ms=2.0, zorder=4, label="column minimum of |c|")
    ax.plot([hx - 1.15, hx - 0.15], [-args.crop_y_L + 0.15] * 2, "-", color=ink, lw=2.5)
    ax.text(hx - 0.65, -args.crop_y_L + 0.22, "1 L", color=ink, ha="center", va="bottom", fontsize=7)
    ax.set_xlim(-hx, hx); ax.set_ylim(-args.crop_y_L, args.crop_y_L)
    if legend:
        ax.legend(fontsize=6, loc="upper left", framealpha=0.85)


def show_pattern(ax, G, my, mx, ext):
    """the pattern muted to light greys, so the markers and labels drawn over it stay readable"""
    ax.set_facecolor("w")
    ax.imshow(G["u"][np.ix_(my, mx)], origin="lower", extent=ext, cmap="Greys", alpha=0.40, aspect="equal")


def _gamma_colorbar(fig, ax, label="|Gamma| of the dip loop (turns)"):
    sm = cm.ScalarMappable(norm=NORM_G, cmap=CMAP_G); sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.8, label=label)


def fig_markers(G, path, args):
    """(a) pattern with the markers, (b) mg A with the markers, (c) mg A along the line, (d) mg A across the line through
    the deepest dip; with --with_rowdemod |c| is added to (c) and (d) and drawn as panel (e)"""
    mx, my, ext = _crop(G, args)
    rd = bool(args.with_rowdemod) and G["C"] is not None
    fig = plt.figure(figsize=(11, 7.6 if not rd else 11.0))
    gs = fig.add_gridspec(2 if not rd else 3, 2, hspace=0.32, wspace=0.18)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axc, axd = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    show_pattern(axa, G, my, mx, ext)
    draw_markers(axa, G, args, "mg", on_light=True); _gamma_colorbar(fig, axa)
    axa.plot([], [], "+", color="#35e0f0", ms=8, mew=1.6, label="lowest point of the dip (panel b)")
    axa.set_title("(a) pattern u with the markers", fontsize=9)
    im = axb.imshow(G["A"][np.ix_(my, mx)], origin="lower", extent=ext, cmap="magma", vmin=0.3, vmax=1.2, aspect="equal")
    draw_markers(axb, G, args, "mg", show_min=True); fig.colorbar(im, ax=axb, shrink=0.8, label="A / A_far")
    axb.set_title("(b) amplitude A", fontsize=9)
    for ax in (axa, axb):
        ax.set_xlabel("position along the PGB line, x / L"); ax.set_ylabel("distance from the PGB line / L")
    xs = (G["x"][mx] - G["xm"]) / G["L"]
    axc.plot(xs, G["A"][G["jr"], mx], color=C_CUT2, lw=1, label="amplitude A")
    if rd:
        axc.plot(xs, G["C"][G["jr"], mx], color=C_CUT1, lw=1, label="|c|")
    for dp in G["dips"]:
        if dp["on"] and abs(_lx(G, dp["x"])) < args.crop_x_L / 2:
            col, ls = _loop_style(dp, args)
            axc.axvline(_lx(G, dp["x"]), color=col, lw=1.0, ls=ls)
    axc.set_xlabel("position along the PGB line, x / L")
    axc.set_ylabel("amplitude A / far field")
    axc.legend(fontsize=7); axc.set_title("(c) along the PGB line (vertical lines: on-line dips)", fontsize=9)
    inc = [dp for dp in G["dips"] if abs(_lx(G, dp["x"])) < args.crop_x_L / 2]
    if inc:
        deep = min(inc, key=lambda dp: dp["depth"])
        i_full = int(np.argmin(np.abs(G["x"] - deep["x"])))
    else:
        i_full = int(np.argmin(np.abs(G["x"] - G["xm"])))
    ysel = np.abs(G["y"] - G["y_row"]) <= args.crop_y_L * G["L"]
    ys = (G["y"][ysel] - G["y_row"]) / G["L"]
    axd.plot(ys, G["A"][ysel, i_full], color=C_CUT2, lw=1, label="amplitude A")
    if rd:
        axd.plot(ys, G["C"][ysel, i_full], color=C_CUT1, lw=1, label="|c|")
    if args.show_beat_node:
        for s in (1, -1):
            axd.axvline(s * G["node"], color="#2aa7c0", lw=0.8, ls=":")
    axd.axvline(0, color=MUTED, lw=0.6, ls="--")
    axd.set_xlabel("distance from the PGB line / L (below is negative)")
    axd.set_ylabel("amplitude A / far field"); axd.legend(fontsize=7)
    axd.set_title("(d) across the line at x = %.2f L, through the deepest dip" % _lx(G, G["x"][i_full]), fontsize=9)
    if rd:
        axe = fig.add_subplot(gs[2, 0])
        im = axe.imshow(G["C"][np.ix_(my, mx)], origin="lower", extent=ext, cmap="magma", vmin=0.3, vmax=1.2, aspect="equal")
        draw_markers(axe, G, args, "c"); fig.colorbar(im, ax=axe, shrink=0.8, label="|c| / A_far (row demodulation)")
        axe.set_title("(e) local row demodulation |c| (option)", fontsize=9)
        axe.set_xlabel("x / L"); axe.set_ylabel("(y - line) / L")
    h, lab = axa.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=7, frameon=False)
    fig.suptitle("mu = %.3f (%.1f deg), t = %.1f: %s;  dips on the line at or above %.2g turn: %d of %d (loop r = %.2g L)" % (
        G["mu"], np.degrees(np.arcsin(G["mu"])), G["t"], G["regime"], args.wind_hi, G["n_on_hi"], G["n_on"], G["r"] / G["L"]),
        fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_domain(G, path, args):
    """the band with every cell and dip loop, and the |Gamma| map of the crop with the loop drawn to scale"""
    x, y, L = G["x"], G["y"], G["L"]
    x0, x1, y0, y1 = G["win"]
    my = np.abs(y - G["y_row"]) <= (G["band_L"] + 0.5) * L
    ext = [(x[0] - G["xm"]) / L, (x[-1] - G["xm"]) / L, (y[my][0] - G["y_row"]) / L, (y[my][-1] - G["y_row"]) / L]
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.6), gridspec_kw=dict(height_ratios=[1, 1.4]))
    ax = axs[0]
    ax.imshow(G["u"][my], origin="lower", extent=ext, cmap="Greys", alpha=0.40, aspect="equal")
    ax.add_patch(Rectangle(((x0 - G["xm"]) / L, -G["band_L"]), (x1 - x0) / L, 2 * G["band_L"], fill=False, ec="k", lw=0.8, ls=":"))
    pos, neg = G["bq"] > 0, G["bq"] < 0
    ax.scatter(_lx(G, G["bx"][pos]), _ly(G, G["by"][pos]), marker="^", s=10, c=C_POS, lw=0)
    ax.scatter(_lx(G, G["bx"][neg]), _ly(G, G["by"][neg]), marker="v", s=10, c=C_NEG, lw=0)
    for dp in G["dips"]:
        col, ls = _loop_style(dp, args)
        ax.add_patch(Circle((_lx(G, dp["x"]), _ly(G, dp["y"])), G["r"] / L, fill=False, ec=col, ls=ls, lw=0.9))
    ax.set_title("band (dotted), V (blue) and X (red) cells, dip loops coloured by |Gamma| (solid: at or above %.2g turn;"
                 " dotted: sign of k does not close)" % args.wind_hi, fontsize=9)
    ax.set_xlabel("position along the PGB line, x / L"); ax.set_ylabel("distance from the PGB line / L")
    ax = axs[1]
    mx, myc, ext2 = _crop(G, args)
    ax.imshow(G["u"][np.ix_(myc, mx)], origin="lower", extent=ext2, cmap="gray", aspect="equal")
    sel = (np.abs(_lx(G, G["CX"])) <= args.crop_x_L / 2) & (np.abs(_ly(G, G["CY"])) <= args.crop_y_L)
    shown = sel & G["even"] if args.with_winding else sel
    sc = ax.scatter(_lx(G, G["CX"][shown]), _ly(G, G["CY"][shown]), c=G["circ"][shown], s=5, cmap=CMAP_G, norm=NORM_G,
                    alpha=0.8, lw=0, marker="s")
    if args.with_winding:
        uc = sel & ~G["even"]
        ax.scatter(_lx(G, G["CX"][uc]), _ly(G, G["CY"][uc]), c="r", s=6, marker="x", lw=0.5)
    else:
        op = sel & ~G["closed"]
        ax.scatter(_lx(G, G["CX"][op]), _ly(G, G["CY"][op]), facecolors="none", edgecolors="r", s=9, marker="o", lw=0.5)
    fig.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.6, pad=0.24, label="|Gamma| of the loop centred there (turns)")
    cx0, cy0 = -args.crop_x_L / 4, 0.5
    ax.add_patch(Circle((cx0, cy0), G["r"] / L, fill=False, ec="c", lw=1.5))
    note = "to scale:\nloop r = %.2g L (cyan)" % (G["r"] / L)
    if args.with_rowdemod:
        ax.plot([cx0 - G["loc"] / L, cx0 + G["loc"] / L], [cy0 - 0.75] * 2, "-", color="orange", lw=2)
        note += "\nrow-demodulation Gaussian\n+- 1 sd = %.2g L (orange bar)" % (G["loc"] / L)
    if args.with_winding:
        note += "\n\nred x: odd twist"
    else:
        note += "\n\nred ring: the sign of k\ndoes not close"
    note += "\n\nbelow mu ~0.87 a 0.5 L\nloop is narrower than\na stripe wavelength"
    ax.text(1.02, 0.98, note, transform=ax.transAxes, color="k", fontsize=7, va="top", ha="left")
    ax.set_xlim(-args.crop_x_L / 2, args.crop_x_L / 2); ax.set_ylim(-args.crop_y_L, args.crop_y_L)
    ax.set_title("|Gamma| for loops centred on a %.3g L grid" % args.grid_L, fontsize=9)
    ax.set_xlabel("position along the PGB line, x / L"); ax.set_ylabel("distance from the PGB line / L")
    fig.suptitle("mu = %.3f: domain of the measures" % G["mu"], fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _regime_legend(ax):
    ax.legend(handles=[Patch(color=v, label=k) for k, v in REGIME_COL.items()], fontsize=6, loc="best")


def _spans(ax, rows):
    mus = np.array([r["mu"] for r in rows])
    if len(mus) == 0:
        return
    edges = np.r_[mus[0] - 0.005, 0.5 * (mus[1:] + mus[:-1]), mus[-1] + 0.005]
    for i, r in enumerate(rows):
        ax.axvspan(edges[i], edges[i + 1], color=REGIME_COL.get(r["regime"], "w"), lw=0, zorder=0)


def _col(rows, k):
    return np.array([r.get(k, float("nan")) for r in rows], float)


def fig_positions(rows, path, args):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    _spans(ax, rows)
    mu = _col(rows, "mu")
    if args.show_beat_node:
        mm = np.linspace(mu.min(), mu.max(), 200)
        ax.plot(mm, np.sqrt(1 - mm ** 2) / (2 * mm), "k--", lw=1, label="beat node k1/(2 mu) (option)")
    v, lo, hi = _col(rows, "V_offline_med_L"), _col(rows, "V_offline_min_L"), _col(rows, "V_offline_max_L")
    ax.errorbar(mu, v, yerr=[v - lo, hi - v], fmt="v", color=C_NEG, ms=4, lw=0.8, capsize=2, label="V cells (median, range)")
    ax.plot(mu, _col(rows, "X_offline_med_L"), "^", color=C_POS, ms=4, label="X cells (median)")
    ax.plot(mu, _col(rows, "dips_off_med_L"), "o", mfc="none", color=MUTED, ms=5, label="dips off the line (median)")
    ax.plot(mu, _col(rows, "A_band_min_off_L"), ".", color=C_CUT2, ms=4, label="deepest amplitude A in the band (every angle)")
    on = _col(rows, "n_dips_on") > 0
    ax.plot(mu[on], np.zeros(on.sum()), "o", color=MUTED, ms=4, label="dips on the line")
    if args.with_rowdemod:
        ax.plot(mu, _col(rows, "cmin_offline_med_L"), ".", color=C_CUT1, ms=5, label="|c| beat-node trough (option)")
    ax.set_xlabel("mu = sin(far-field angle)"); ax.set_ylabel("distance from the line (L)")
    ax.set_ylim(bottom=-0.03); ax.legend(fontsize=7); _deg_axis(ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_magnitudes(rows, path, args):
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.8), sharex=True)
    mu = _col(rows, "mu")
    for ax in axs.ravel():
        _spans(ax, rows)
    ax = axs[0, 0]
    ws = scan_widths(args)
    if ws:
        cmap_w = plt.get_cmap("plasma")
        for i, w in enumerate(ws):
            ax.plot(mu, _col(rows, "charge_per_L_sd%s" % _wtag(w)), "-", color=cmap_w(0.1 + 0.75 * i / max(len(ws) - 1, 1)),
                    lw=1.6 if abs(w - args.sigma_det) < 1e-9 else 0.9, label="sum |T| per L, sigma_det %.2g" % w)
    else:
        ax.plot(mu, _col(rows, "charge_abs_per_L"), "^-", color=C_POS, ms=3, label="sum |T| per L, sigma_det %.2g" % args.sigma_det)
        ax.plot(mu, _col(rows, "charge_abs_per_L_sd2"), "^--", color=C_POS, ms=3, mfc="none", label="sigma_det %.2g" % args.sigma_det2)
    ax.set_ylabel("director charge per L"); ax.legend(fontsize=7)
    ax = axs[0, 1]
    ax.plot(mu, _col(rows, "VX_sep_med_L"), "D-", color=C_NEG, ms=3, label="V-X separation")
    ax.plot(mu, _col(rows, "XX_sep_med_L"), "^-", color=C_POS, ms=3, mfc="none", label="X-X separation")
    ax.plot(mu, _col(rows, "V_offline_med_L"), "v:", color=C_NEG, ms=3, mfc="none", label="V distance from the line")
    ax.plot(mu, _col(rows, "dip_off_to_V_med_L"), "o-", color=MUTED, ms=3, mfc="none", label="off-line dip to nearest V")
    ax.set_ylabel("distance (L)"); ax.legend(fontsize=7)
    ax = axs[1, 0]
    ax.plot(mu, _col(rows, "dip_circ_on_med"), "o-", color="#3b5ba5", ms=3, label="|Gamma| at on-line dips (median)")
    ax.plot(mu, _col(rows, "dip_circ_on_max"), "^", color="#3b5ba5", ms=3, mfc="none", label="on-line dips (max)")
    ax.plot(mu, _col(rows, "dip_circ_off_med"), "x-", color=MUTED, ms=4, lw=0.8, label="|Gamma| at off-line dips (median)")
    ax.plot(mu, _col(rows, "dip_circ_off_max"), "+", color=MUTED, ms=5, label="off-line dips (max)")
    ax.plot(mu, _col(rows, "frac_on_circ_hi"), ":", color="k", lw=1, label="fraction of on-line dips at or above %.2g" % args.wind_hi)
    if args.with_winding:
        ax.plot(mu, _col(rows, "dip_odd_twist_frac"), ":", color="r", lw=1, label="fraction of dip loops with odd twist (option)")
    ax.axhline(1, color=MUTED, lw=0.6, ls=":"); ax.axhline(args.wind_hi, color=MUTED, lw=0.6, ls="--")
    ax.set_ylabel("circulation (turns)"); ax.set_xlabel("mu"); ax.legend(fontsize=6)
    ax = axs[1, 1]
    ax.plot(mu, _col(rows, "A_at_V_med"), "v-", color=C_NEG, ms=3, label="amplitude A at the V cells")
    ax.plot(mu, _col(rows, "A_line_min"), "-", color=C_CUT2, lw=1, label="amplitude A on the line, min")
    ax.plot(mu, _col(rows, "A_line_med"), "--", color=C_CUT2, lw=1, label="amplitude A on the line, median")
    ax.plot(mu, _col(rows, "dip_depth_off_med"), "o", color=MUTED, ms=3, mfc="none", label="dip depth, off the line")
    ax.plot(mu, _col(rows, "dip_depth_on_med"), "o", color=MUTED, ms=3, label="dip depth, on the line")
    ax.plot(mu, _col(rows, "A_band_min"), "-", color="k", lw=1.2, label="deepest amplitude A in the band (every angle)")
    if args.with_rowdemod:
        ax.plot(mu, _col(rows, "cmin_depth_med"), ".", color=C_CUT1, ms=4, label="|c| at the beat node (option)")
    ax.set_ylabel("amplitude / far"); ax.set_xlabel("mu"); ax.legend(fontsize=6)
    axs[0, 0].legend(fontsize=6, loc="upper left"); _deg_axis(axs[0, 0]); _deg_axis(axs[0, 1])
    fig.legend(handles=[Patch(color=v, label=k) for k, v in REGIME_COL.items()], loc="lower center", ncol=3, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_energy(rows, path):
    rr = [r for r in rows if np.isfinite(r["E_GB"])]
    if not rr:
        return False
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    _spans(ax, rr)
    ax.plot([r["mu"] for r in rr], [r["E_GB"] for r in rr], "o-", color="k", ms=3)
    ax.set_xlabel("mu"); ax.set_ylabel("line energy E_GB (excess per unit length)")
    _regime_legend(ax); _deg_axis(ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return True


def fig_montage(Gs, idx, args, path=None, pdf=None, per_page=8):
    rd = bool(args.with_rowdemod)
    nrow = 3 if rd else 2
    pages = [idx[s:s + per_page] for s in range(0, len(idx), per_page)] if pdf is not None else [idx]
    for page in pages:
        n = len(page)
        if n == 0:
            continue
        fig, axs = plt.subplots(nrow, n, figsize=(2.3 * n + 0.6, 2.15 * nrow), squeeze=False)
        for j, i in enumerate(page):
            G = Gs[i]; mx, my, ext = _crop(G, args)
            show_pattern(axs[0, j], G, my, mx, ext)
            draw_markers(axs[0, j], G, args, "none", on_light=True)
            axs[0, j].set_title("mu %.2f (%.0f deg)\n%s" % (G["mu"], np.degrees(np.arcsin(G["mu"])), G["regime"]), fontsize=7)
            axs[1, j].imshow(G["A"][np.ix_(my, mx)], origin="lower", extent=ext, cmap="magma", vmin=0.3, vmax=1.2, aspect="equal")
            draw_markers(axs[1, j], G, args, "none")
            if rd and G["C"] is not None:
                axs[2, j].imshow(G["C"][np.ix_(my, mx)], origin="lower", extent=ext, cmap="magma", vmin=0.3, vmax=1.2, aspect="equal")
            for ax in axs[:, j]:
                ax.set_xticks([]); ax.set_yticks([])
        axs[0, 0].set_ylabel("pattern", fontsize=8); axs[1, 0].set_ylabel("A / far", fontsize=8)
        if rd:
            axs[2, 0].set_ylabel("|c| / far", fontsize=8)
        fig.tight_layout()
        if pdf is not None:
            pdf.savefig(fig)
        else:
            fig.savefig(path, dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------

MD_KEYS = ["mu", "alpha_deg", "t", "regime", "charge_abs_per_L", "charge_abs_per_L_sd2", "A_band_min", "A_band_min_off_L",
           "map_circ_max_closed", "dip_min_off_off_med_L", "dip_min_to_centroid_med_L",
           "dip_open_frac_on", "dip_open_frac_off", "V_offline_med_L", "X_offline_med_L", "VX_sep_med_L",
           "n_dips_on", "n_dips_off", "dips_off_med_L", "dip_off_to_V_med_L", "dip_depth_off_med", "dip_depth_on_med",
           "dip_circ_off_med", "dip_circ_off_max", "dip_circ_on_med", "n_dips_on_circ_hi", "A_at_V_med", "A_line_min", "E_GB", "flags"]


def write_tables(rows, out):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out / "survey_table.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
    with open(out / "survey_table.md", "w") as fh:
        fh.write("| " + " | ".join(MD_KEYS) + " |\n|" + "---|" * len(MD_KEYS) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(("%.3f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in MD_KEYS) + " |\n")


def set_figures(rows, Gs, args, out):
    fig_positions(rows, out / "fig_positions_vs_mu.png", args)
    fig_magnitudes(rows, out / "fig_magnitudes_vs_mu.png", args)
    has_E = fig_energy(rows, out / "fig_energy_vs_mu.png")
    sel = montage_select(rows, args.montage_step)
    fig_montage(Gs, sel, args, path=out / "fig_montage.png")
    with PdfPages(out / "fig_montage_all.pdf") as pdf:
        fig_montage(Gs, list(range(len(Gs))), args, pdf=pdf)
    return has_E, sel


def header_line(args):
    opts = [k for k in ("with_winding", "with_rowdemod", "show_beat_node") if getattr(args, k)]
    return ("%s; sigma_dir %.2g, sigma_det %.2g (and %.2g), band %.2g L, dip loops %s, on the line <= %.2g L, dips below %.2g"
            " of the median A, regime threshold %.2g turn, frame %d; options on: %s" % (
                VERSION, args.sigma_dir, args.sigma_det, args.sigma_det2, args.band_L,
                ("r %.2g stripe wavelengths" % args.loop_lam) if args.loop_lam > 0 else ("r %.2g L" % args.loop_L),
                args.on_line_L, args.core_frac, args.wind_hi, args.frame, ", ".join(opts) if opts else "none"))


def main(args):
    out = Path(args.outdir)
    if not out.is_absolute():
        out = HERE / out
    out.mkdir(parents=True, exist_ok=True)
    files = expand_inputs(args.npz)
    if not files:
        print("no input files (--npz)"); return 1
    with open(out / "config.json", "w") as fh:
        json.dump(dict(vars(args), version=VERSION, files=files), fh, indent=2, default=str)
    lines = [header_line(args), ""]
    rows, Gs = [], []
    t0 = time.time()
    for f in files:
        try:
            row, G = survey_file(f, args, out, lines)
            rows.append(row); Gs.append(G)
        except Exception as exc:                                    # loud flag, keep going
            lines.append("FLAG: %s failed: %r" % (f, exc))
        print(lines[-1] if lines else "")
    if not rows:
        (out / "report.txt").write_text("\n".join(lines)); return 1
    order = np.argsort([r["mu"] for r in rows])
    rows = [rows[i] for i in order]; Gs = [Gs[i] for i in order]
    write_tables(rows, out)
    has_E, sel = set_figures(rows, Gs, args, out)
    lines.append("")
    lines.append("regimes by mu: " + ", ".join("%.3f %s" % (r["mu"], r["regime"]) for r in rows))
    if not has_E:
        lines.append("FLAG: no energy found in any file (no 'e' array or cache E_GB); the energy figure was not drawn")
    lines.append("montage angles: " + ", ".join("%.3f" % rows[i]["mu"] for i in sel))
    lines.append("elapsed %.1f s" % (time.time() - t0))
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-4:]))
    print("outputs ->", out)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    npz = ""                       # comma-separated files and/or folders (every *.npz in a folder)
    outdir = str(Path("results") / "pgb_charge_circ_survey_v12")
    frame = -1                     # frame index; -1 = last
    tag_frame = 0                  # 1: append _f<frame> to each file's folder name
    mu = 0.85                      # fallback when a file carries no mu
    sigma_dir = 0.4                # mg director width (production value)
    taper_frac = 0.06
    sigma_det = 0.3                # director-charge width (the records' value; the regime uses it)
    sigma_det2 = 0.2               # second detection width (the gallery reads 0.3 and 0.2)
    sigma_scan = "0.15,0.2,0.25,0.3,0.35,0.4"   # director-charge widths for the width scan ("" for none)
    show_colmin = 0                # 1: draw the per-column minimum of A (green dots) on the marker figures
    map = 1                        # 1: the loop map of the domain figure; 0 skips it (the time-series script)
    band_L = 2.0                   # band half-height, L
    loop_L = 0.5                   # dip loop radius, L (the records' radius)
    loop_lam = 0.0                 # > 0: dip loop radius in far-field stripe wavelengths instead of loop_L (the loop-size control)
    nang = 181                     # samples per loop for the circulation (loop_circulation's value)
    wind_hi = 0.9                  # turns: an on-line dip at or above this makes the boundary a dislocation
    grid_L = 0.125                 # loop-map spacing, L (domain figure only)
    band_min = -1.0                # search half-width for the column minima of mg A, L; <= 0: max(1, 1.5 pi/(2 mu) / L)
    min_contrast = 0.03            # a column needs this contrast (over far field) to have a minimum position
    far_from = 4.0                 # far field for A_far, L from the line
    core_frac = 0.70               # dips: mg A below this fraction of its window median (the records' value)
    on_line_L = 0.1                # a dip within this distance of the line is on the line, L
    crop_x_L = 4.0                 # figure crop along the line, L
    crop_y_L = 1.5                 # figure crop half-height, L
    montage_step = 0.02            # montage: every montage_step in mu, plus the angles beside a regime change
    energy_key = ""                # energy-density array in the movie files ("" tries e, energy_density, E_density)
    edge_frac = 0.12               # line energy: rows at each edge for the roll baseline (the sweep's value)
    # options, off by default
    with_winding = 0               # 1: loop twist and integer phase winding n at the dips (pgb_marker_survey v1.2)
    nang_phase = 720               # samples per loop for the twist and n (with_winding only)
    with_rowdemod = 0              # 1: the x-local row demodulation |c| and its beat-node trough
    loc_L = 0.5                    # Gaussian standard deviation of the row demodulation, L (with_rowdemod only)
    show_beat_node = 0             # 1: draw the beat-node lines y = +- pi/(2 mu)


def default_args():
    """an argparse-like namespace with every default, for importing scripts"""
    return argparse.Namespace(**{k: getattr(_Args, k) for k in vars(_Args) if not k.startswith("_")})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in vars(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    sys.exit(main(p.parse_args()))
