"""pgb_marker_survey.py  v1.2  (25 Oct 2026)   the stationary marker survey (N1 of
RESEARCH_REVAMP_EXPERIMENTS_HANDOFF_1024.md; design in MARKER_SURVEY_DESIGN_NOTES_1024.md)

Plain terms. For each zigzag phase grain boundary (one frame per file, the
last by default) the survey finds the director charges (half-turn defects of
the stripe orientation, with their sign), the amplitude dips, and the phase
winding, draws them on the pattern, and labels the boundary knee,
intermediate or dislocation by the rule agreed with Ed on 25 Oct:

  knee          no director charge;
  intermediate  director charge, and no dip winds; the dips, when present,
                sit off the line;
  dislocation   director charge, and the dips on the line wind (even twist,
                integer winding 1 on the dip loop); no dip off the line winds;
  unresolved    anything else (a dip off the line winds), with the flags.

Distances are in L = pi / k1, k1 = sqrt(1 - mu^2), from the line y = y_row;
the band is |y - y_row| < band_L L; the window is the inner 80 percent of
the box.

Measures per file:
  (1) director charges. The frozen t1c.detect_defects (Riesz director at
      sigma_det, plaquette winding of the doubled angle) on u x taper; every
      nonzero cell in the band and window with its sign: q = -1 cells (V
      type) and q = +1 cells (X type); also counted at sigma_det2, the
      gallery's second width.
  (2) dips and the winding at them. The records' dips (find_cores: regions
      where mg's A falls below core_frac of its window median, merged within
      0.5 L), each with: distance from the line (on the line within
      on_line_L), depth (the minimum of A / A_far within 0.25 L), and on the
      0.5 L loop about it the continuous k circulation (loop_circulation,
      the records' number), the twist and the integer phase winding Gamma
      (the arithmetic of defect_gallery.circuit_twist and
      circuit_circulation, written for a rectangular grid from mg's own z
      and director). A dip winds when its loop has even twist and
      |Gamma| = 1.
  (3) the dip loops' size. The radius is loop_L L (0.5 L, the records'), or,
      with --loop_lam > 0, a fixed number of far-field stripe wavelengths:
      the control for whether the switch to wound dips is a property of
      the pattern or of the loop's size against the core's (the V cell and
      the X cell half a stripe away sit 0.58 to 0.64 wavelengths apart at
      every angle on the stand-ins, so in units of L the core shrinks only
      because L grows).
  (4) geometry of the charges, continuous in mu: the median distance from
      the line of the V cells and of the X cells, the median V-X separation
      (each V cell to its nearest X cell), the median X-X separation, and
      the cells per L of line.
  (5) amplitude: mg's A at the V cells and along the line (median and
      minimum, over A_far); the depths of the dips on and off the line; and
      the beat-node trough of the x-local row demodulation |c| (its
      per-column minimum, position over pi/(2 mu) and depth). |c| is a
      beat-node measure only: its 0.5 L Gaussian along x spans neighbouring
      cores, so it does not see a core (v1.1 run of 25 Oct).
  (6) a descriptive map for the domain figure: the 0.5 L loop on a grid of
      centres over the band, coloured by the continuous k circulation, with
      the odd-twist centres marked. It is not summarised: below mu of about
      0.87 a 0.5 L loop is narrower than a stripe wavelength and its reading
      depends on where it sits within a core (v1.1 run of 25 Oct).
  (7) line energy E_GB when the file carries an energy density ("e", or
      --energy_key) or is an energy_vs_angle_zigzag cache record.
  (8) the records' convention for continuity: dip count, wound dips
      (k circulation >= wind_hi), largest circulation, charge sum; on the
      same files these reproduce the October records, and the survey's loop
      code reproduces loop_circulation exactly.

Inputs: --npz takes files and/or folders (every *.npz in a folder). Layouts:
the solver movies (u, x, y, tt or t, metadata_json with mu and
gb_row_in_crop) and the energy_vs_angle_zigzag cache records (u_last, y, dx,
gb_row, mu, e_last, E_GB, t_final). --frame picks the frame (-1 the last).

Outputs under --outdir (default results/pgb_marker_survey/, anchored to
this script's folder when relative): per file <stem>/markers.json,
columns.csv, circ_map.npz, fig_domain.png (F0), fig_markers.png (F1); for
the set survey_table.csv, survey_table.md, fig_montage.png and
fig_montage_all.pdf (F2), fig_positions_vs_mu.png (F3),
fig_magnitudes_vs_mu.png (F4), fig_energy_vs_mu.png (F6), report.txt,
config.json. The functions are importable (survey_file, the figure
functions), as large_sweep_markers.py does.

The frozen method layer (src/op_extract, src/utils, the mg estimator) is
imported and never edited. Python 3.8 compatible.

Changelog
  v1.2  (25 Oct 2026) after Ed's v1.1 run of 25 Oct
        (MARKER_SURVEY_LOCAL_RESULTS_1025.md) and sandbox prototypes on
        SH stand-in frames (sh_zigzag_standin, mu 0.60 to 0.95; not of
        record). (1) Regime by Ed's rule (above): charge, and the winding
        at the dips with Gamma on the same loops. v1.1's rule needed the |c|
        minimum on the line at wound centres; |c| never sees a core, so
        every file from 0.76 up came out unresolved. (2) No count of winding
        along the line: on the stand-ins mg's z has no phase vortex anywhere
        in the band (plaquette by plaquette, 0.78, 0.88, 0.95); the winding
        is carried by the half-turn cells, around which the phase changes
        sign (theta -> -theta), so windings around groups of cells combine
        like reflections and depend on how a circuit threads between them
        (the non-commuting defect group of stripe order). A one-spacing
        rectangle read Gamma 1 or 0 at 0.88 and 0.95 depending only on
        where its sides crossed the line (1 between the X cells that bound
        neighbouring cores, 0 elsewhere), and 1 at 2 of 20 placements at
        0.78; a check of phase against k along the sides (drift 0.1 rad
        everywhere) did not separate good placements from bad. (3) No
        fractional winding from the V-X phase offset: b = |phase difference
        between a V cell and an X cell| / pi, from the integral of k along
        the segment, is 0 or 1 to within 0.15 at 0.72, 0.78, 0.88, 0.95 (the
        cells sit on stripe lines, as the reflection picture requires); the
        V cell and its next-line X cell sit 0.64, 0.62, 0.59, 0.58 stripe
        wavelengths apart, hence the loop-size control (3). (4) Dropped:
        the continuous circulation about the deepest minimum per spacing and
        about the charge clusters, the winding density curl k / 2 pi (at
        0.95, Gamma 1 per spacing, it integrated to 0.02: mg's k has a sign
        only up to the director and any local orientation rule puts a seam
        through each core), and loops about single V-X pairs (a third of a
        wavelength across). (5) Added the charge geometry (V and X
        distances from the line, V-X and X-X separations) and the amplitude
        at the V cells and along the line as the continuous descriptors.
        Reading for the write-up: in a stationary frame the winding is an
        integer wherever it is defined; the intermediate structure is
        half-turn cells (V at the off-line dips, X on or near the line), and
        the continuous quantities are geometric.
  v1.1  (25 Oct 2026) twist parity and integer winding Gamma on every loop
        by the gallery's rule; continuous k circulation; charges at two
        widths; loop-size flag; gallery terms (even and odd twist).
  v1.0  (24 Oct 2026) first version: signed charges, the 0.5 L loop grid
        with sign closure, amplitude minima by mg and |c|, four-panel figure.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import label

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT / "src", ROOT / "src" / "op_extract", ROOT / "src" / "utils", HERE):
    sys.path.insert(0, str(_p))
from mg_zigzag_theta_t import load_npz, estimate, find_cores                # noqa: E402
from op_extract import t1_common as t1c                                     # noqa: E402
from knee_row_demod_survey import fit_k1, row_demod, profile_numbers        # noqa: E402
from knee_amplitude_sweep import local_row_demod, sub_min                   # noqa: E402
from theta_t_mu_sweep import loop_circulation                               # noqa: E402
import matplotlib                                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages                        # noqa: E402
from matplotlib.patches import Circle, Rectangle                           # noqa: E402

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
VERSION = "pgb_marker_survey v1.2"
C_POS, C_NEG, C_CUT1, C_CUT2, MUTED = "#d1352b", "#2a6fd6", "#eb8a34", "#1baf7a", "#77756f"
C_WIND = "#5b3fa8"
REGIME_COL = {"knee": "#e9e7e2", "intermediate": "#f6e3b4", "dislocation": "#cfe0f5", "unresolved": "#f2c6c6"}


# ---------------------------------------------------------------------
# input and the measures shared with v1.1
# ---------------------------------------------------------------------

def expand_inputs(spec):
    out = []
    for item in [s.strip() for s in str(spec).split(",") if s.strip()]:
        p = Path(item)
        if p.is_dir():
            out.extend(sorted(str(q) for q in p.glob("*.npz")))
        else:
            out.append(item)
    return out


def _yx_nt(a):
    """the [ny, nx, nt] layout of load_npz"""
    if a.ndim == 3 and not (a.shape[-1] < a.shape[0]):
        a = np.moveaxis(a, 0, -1)
    return a


def load_frame(path, args):
    """one frame and its geometry from either layout"""
    d = np.load(path, allow_pickle=True)
    files = list(d.files)
    stem = Path(path).stem
    rec = dict(stem=stem, path=str(path), E_GB_stored=float("nan"), e2=None, energy_source="none")
    if "u" in files:
        D = load_npz(path)
        u = D["u"]
        if u.ndim == 2:                                            # a single frame stored as [ny, nx]
            u = u[..., None]
        nt = u.shape[-1]
        i = args.frame if args.frame >= 0 else nt + args.frame
        rec.update(u2=np.asarray(u[..., i], float), x=np.asarray(D["x"], float), y=np.asarray(D["y"], float),
                   t=float(np.atleast_1d(D["t"])[min(i, np.atleast_1d(D["t"]).size - 1)]), frame=int(i), nt=int(nt), md=dict(D["md"]), layout="movie")
        del u, D
        keys = [args.energy_key] if args.energy_key else ["e", "energy_density", "E_density"]
        for kk in keys:
            if kk and kk in files:
                e = _yx_nt(np.asarray(d[kk]))
                if e.ndim == 3 and e.shape[-1] == nt:
                    rec["e2"] = np.asarray(e[..., i], float); rec["energy_source"] = "%s[..., %d]" % (kk, i)
                elif e.ndim == 2:
                    rec["e2"] = np.asarray(e, float); rec["energy_source"] = kk
                break
    elif "u_last" in files:
        u2 = np.asarray(d["u_last"], float)
        dx = float(d["dx"])
        rec.update(u2=u2, x=np.arange(u2.shape[1]) * dx, y=np.asarray(d["y"], float),
                   t=float(d["t_final"]) if "t_final" in files else float("nan"), frame=-1, nt=1,
                   md=dict(mu=float(d["mu"]), gb_row_in_crop=int(d["gb_row"])), layout="energy_cache")
        if "e_last" in files:
            rec["e2"] = np.asarray(d["e_last"], float); rec["energy_source"] = "e_last"
        if "E_GB" in files:
            rec["E_GB_stored"] = float(d["E_GB"])
    else:
        raise ValueError("%s: no u or u_last array (keys %s)" % (path, files))
    return rec


def _line_energy(e2, y, jr, edge_frac):
    """E_GB of one energy-density frame: energy_vs_angle_zigzag_v2.line_energy_from_frame,
    or the same arithmetic when that module cannot be imported (it imports the solver)"""
    try:
        from energy_vs_angle_zigzag_v2 import line_energy_from_frame
        r = line_energy_from_frame(e2, y, jr, edge_frac=edge_frac)
        return float(r["E_GB"]), float(r["flatness"]), "energy_vs_angle_zigzag_v2.line_energy_from_frame"
    except Exception:
        Ny = e2.shape[0]; dy = float(y[1] - y[0])
        e_bar = e2.mean(axis=1)
        n_edge = max(2, int(round(edge_frac * Ny)))
        bot, top = float(np.mean(e_bar[:n_edge])), float(np.mean(e_bar[-n_edge:]))
        e_roll = 0.5 * (bot + top)
        flat = abs(top - bot) / max(abs(e_roll), 1e-12)
        return float(np.sum(e_bar - e_roll) * dy), float(flat), "fallback copy of line_energy_from_frame"


# ---------------------------------------------------------------------
# measures
# ---------------------------------------------------------------------

def loop_grid(k1, k2, x, y, xc, yc, r, nang=181):
    """circulation (turns) on circles of radius r about every (xc, yc), the
    director sign carried by continuity as in theta_t_mu_sweep.loop_circulation
    (identical sampling and quadrature), plus whether the carried sign closes"""
    dx = x[1] - x[0]; dy = y[1] - y[0]
    ang = np.linspace(0, 2 * np.pi, nang)
    px = xc[:, None] + r * np.cos(ang)[None, :]
    py = yc[:, None] + r * np.sin(ang)[None, :]
    ix = np.clip(np.round((px - x[0]) / dx).astype(int), 0, len(x) - 1)
    iy = np.clip(np.round((py - y[0]) / dy).astype(int), 0, len(y) - 1)
    kx, ky = k1[iy, ix], k2[iy, ix]
    sg = np.ones_like(kx)
    for q in range(1, nang):
        keep = kx[:, q] * kx[:, q - 1] + ky[:, q] * ky[:, q - 1] >= 0
        sg[:, q] = np.where(keep, sg[:, q - 1], -sg[:, q - 1])
    tx, ty = -np.sin(ang)[None, :], np.cos(ang)[None, :]
    circ = _trapz((kx * tx + ky * ty) * sg, ang, axis=1) * r / (2 * np.pi)
    closed = sg[:, -1] == sg[:, 0]
    return circ, closed


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def loop_phase_winding(z, d1, d2, x, y, xc, yc, r, nang=720, chunk=2000):
    """twist parity and integer phase winding on circles of radius r about every (xc, yc), by the
    rule and arithmetic of defect_gallery.circuit_twist and circuit_circulation (certified on the
    relaxed SH gallery): the director's doubled angle is unwrapped along the circuit; the twist is
    its total over 2 pi (units of pi); when the twist is even the director is flip-corrected along
    the circuit and Gamma = sum of wrapped increments of psi = atan2(d_loop . r, u), in units of
    2 pi, an integer; when the twist is odd Gamma is undefined (NaN; Passot and Newell). Written
    for a rectangular grid (the gallery's sampler assumes a square one) and from mg's own z and
    director: since d_loop = s d_grid with s = +-1, d_loop . r = s Im z and u x taper = Re z."""
    dx = x[1] - x[0]; dy = y[1] - y[0]
    ang = np.linspace(0, 2 * np.pi, nang, endpoint=False)
    a2g = 2.0 * np.arctan2(d2, d1)
    twist = np.zeros(len(xc), int); gamma = np.full(len(xc), np.nan)
    for s0 in range(0, len(xc), chunk):
        cx, cy = xc[s0:s0 + chunk], yc[s0:s0 + chunk]
        px = cx[:, None] + r * np.cos(ang)[None, :]
        py = cy[:, None] + r * np.sin(ang)[None, :]
        ix = np.clip(np.round((px - x[0]) / dx).astype(int), 0, len(x) - 1)
        iy = np.clip(np.round((py - y[0]) / dy).astype(int), 0, len(y) - 1)
        a2 = a2g[iy, ix]
        steps = _wrap(np.diff(a2, axis=1))
        a2u = np.concatenate([a2[:, :1], a2[:, :1] + np.cumsum(steps, axis=1)], axis=1)
        total = a2u[:, -1] + _wrap(a2[:, 0] - a2u[:, -1]) - a2u[:, 0]
        tw = np.round(total / (2 * np.pi)).astype(int)                 # twist in units of pi
        twist[s0:s0 + chunk] = tw
        s = np.sign(np.cos(0.5 * a2u - 0.5 * a2))                       # d_loop = s d_grid
        s[s == 0] = 1.0
        zz = z[iy, ix]
        psi = np.arctan2(s * zz.imag, zz.real)
        g = np.sum(_wrap(np.diff(np.concatenate([psi, psi[:, :1]], axis=1), axis=1)), axis=1) / (2 * np.pi)
        g[tw % 2 != 0] = np.nan
        gamma[s0:s0 + chunk] = g
    return twist, gamma


def enclosed_charge(cx, cy, px, py, pq, r, chunk=4000):
    out = np.zeros(len(cx), int)
    if len(px) == 0:
        return out
    for s in range(0, len(cx), chunk):
        d = np.hypot(cx[s:s + chunk, None] - px[None, :], cy[s:s + chunk, None] - py[None, :])
        out[s:s + chunk] = np.sum((d < r) * pq[None, :], axis=1)
    return out


def signed_clusters(px, py, pq, merge):
    order = np.argsort(px)
    px, py, pq = px[order], py[order], pq[order]
    used = np.zeros(len(px), bool)
    cl = []
    for i in range(len(px)):
        if used[i]:
            continue
        g = (~used) & (np.hypot(px - px[i], py - py[i]) < merge)
        used |= g
        cl.append(dict(x=float(px[g].mean()), y=float(py[g].mean()), net=int(pq[g].sum()),
                       abs=int(np.abs(pq[g]).sum()), n=int(g.sum())))
    return cl


def column_minima(F, y, jb, y_row, Fref, min_contrast):
    """per column: y of the minimum of F[jb, :] (sub-grid) and depth F_min / Fref; NaN where the
    column's contrast in the band is below min_contrast of Fref"""
    sub = F[jb, :]
    jloc = np.argmin(sub, axis=0)
    dyy = float(y[1] - y[0])
    ym = np.array([y[jb[j]] + sub_min(sub[:, i], j) * dyy - y_row for i, j in enumerate(jloc)])
    depth = sub.min(axis=0) / Fref
    contrast = (sub.max(axis=0) - sub.min(axis=0)) / Fref
    ym[contrast < min_contrast] = np.nan
    return ym, depth


def components(mask, CX, CY, vals, step, L):
    lab, n = label(mask, structure=np.ones((3, 3)))
    out = []
    for k in range(1, n + 1):
        m = lab == k
        out.append(dict(x=float(CX[m].mean()), y=float(CY[m].mean()), area_L2=float(m.sum() * step ** 2 / L ** 2),
                        circ_med=float(np.median(np.abs(vals[m]))), n=int(m.sum())))
    return out



# ---------------------------------------------------------------------
# small figure helpers (v1.1)
# ---------------------------------------------------------------------

def _crop(G, args):
    x, y = G["x"], G["y"]
    mx = np.abs(x - G["xm"]) <= args.crop_x_L / 2 * G["L"]
    my = np.abs(y - G["y_row"]) <= args.crop_y_L * G["L"]
    ext = [(x[mx][0] - G["xm"]) / G["L"], (x[mx][-1] - G["xm"]) / G["L"],
           (y[my][0] - G["y_row"]) / G["L"], (y[my][-1] - G["y_row"]) / G["L"]]
    return mx, my, ext


def _lx(G, v):
    return (np.asarray(v) - G["xm"]) / G["L"]


def _ly(G, v):
    return (np.asarray(v) - G["y_row"]) / G["L"]


def _regime_spans(ax, rows):
    mus = np.array([r["mu"] for r in rows])
    if len(mus) == 0:
        return
    edges = np.r_[mus[0] - 0.005, 0.5 * (mus[1:] + mus[:-1]), mus[-1] + 0.005]
    for i, r in enumerate(rows):
        ax.axvspan(edges[i], edges[i + 1], color=REGIME_COL.get(r["regime"], "w"), lw=0, zorder=0)


def _deg_axis(ax):
    try:
        sec = ax.secondary_xaxis("top", functions=(lambda m: np.degrees(np.arcsin(np.clip(m, -1, 1))),
                                                   lambda d: np.sin(np.radians(d))))
        sec.set_xlabel("far-field angle (deg)", fontsize=8)
    except Exception:
        pass


def _regime_legend(ax):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=v, label=k) for k, v in REGIME_COL.items()], fontsize=6, loc="best")


def montage_select(rows, step=0.02):
    keep = set()
    for i, r in enumerate(rows):
        if abs(r["mu"] / step - round(r["mu"] / step)) < 1e-3:
            keep.add(i)
        if i > 0 and rows[i - 1]["regime"] != r["regime"]:
            keep.update([i - 1, i])
    return sorted(keep)



def survey_file(path, args, outroot, lines, quiet_figs=False):
    t0 = time.time()
    R = load_frame(path, args)
    u2, x, y, md = R["u2"], R["x"], R["y"], R["md"]
    stem = R["stem"] + (("_f%d" % R["frame"]) if getattr(args, "tag_frame", 0) else "")
    mu = float(md.get("mu", args.mu))
    k1t = float(np.sqrt(1 - mu ** 2)); L = np.pi / k1t
    node = k1t / (2 * mu)                                   # beat node pi/(2 mu) in units of L
    ny, nx = u2.shape
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
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
    use_k1 = abs(np.mean(K1r)) >= abs(np.mean(K2r))
    sgn = np.sign(np.mean(K1r if use_k1 else K2r)) or 1.0
    K1, K2 = K1r * sgn, K2r * sgn
    d1, d2 = e["n"][0], e["n"][1]
    far = (np.abs(Y - y_row) > args.far_from * L) & win
    if far.sum() > 50:
        A_far = float(np.median(A[far]))
    else:
        A_far = float(np.median(A[win])); flags.append("mg A_far from the window median (no far field)")

    # (1) director charges, signed, at two widths
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
    Vm, Xm = bq < 0, bq > 0

    # (2) dips and the winding at them
    wavelength = 2 * np.pi / max(float(np.median(np.hypot(K1, K2)[far])) if far.sum() > 50 else 1.0, 1e-6)
    r = args.loop_lam * wavelength if args.loop_lam > 0 else args.loop_L * L
    xd_, yd_ = find_cores(A, x, y, win, args.core_frac, merge_dist=0.5 * L)
    xd_, yd_ = np.asarray(xd_, float), np.asarray(yd_, float)
    rec_circ = np.array([abs(loop_circulation(K1r, K2r, x, y, a, b, 0.5 * L)) for a, b in zip(xd_, yd_)])
    if len(xd_):
        mine, _ = loop_grid(K1r, K2r, x, y, xd_, yd_, 0.5 * L, 181)
        chk = float(np.max(np.abs(np.abs(mine) - rec_circ)))
        tw_d, g_d = loop_phase_winding(e["z"], d1, d2, x, y, xd_, yd_, r, args.nang_phase)
    else:
        chk = 0.0; tw_d, g_d = np.zeros(0, int), np.zeros(0)
    d_off = np.abs(yd_ - y_row) / L
    on_d = d_off <= args.on_line_L
    wound_d = (tw_d % 2 == 0) & (np.abs(np.nan_to_num(g_d)) >= 0.5)
    rq = max(1, int(round(0.25 * L / dx)))
    def depth_at(a, b):
        i = int(np.argmin(np.abs(x - a))); j = int(np.argmin(np.abs(y - b)))
        return float(A[max(0, j - rq):j + rq + 1, max(0, i - rq):i + rq + 1].min() / A_far)
    dep_d = np.array([depth_at(a, b) for a, b in zip(xd_, yd_)])

    # (4) charge geometry
    def med(v):
        return float(np.median(v)) if len(v) else float("nan")
    v_off = np.abs(by[Vm] - y_row) / L; x_off = np.abs(by[Xm] - y_row) / L
    vx_sep = np.array([np.min(np.hypot(bx[Xm] - a, by[Xm] - b)) for a, b in zip(bx[Vm], by[Vm])]) / L if Xm.any() else np.zeros(0)
    xx_sep = np.array([np.min(np.hypot(np.delete(bx[Xm], i) - a, np.delete(by[Xm], i) - b))
                       for i, (a, b) in enumerate(zip(bx[Xm], by[Xm]))]) / L if Xm.sum() > 1 else np.zeros(0)

    # (5) amplitude: at the V cells, along the line, and the beat node of |c|
    A_at_V = np.array([A[int(np.argmin(np.abs(y - b))), int(np.argmin(np.abs(x - a)))] for a, b in zip(bx[Vm], by[Vm])]) / A_far
    ix = np.where(np.abs(x - xm) < 0.4 * Lx)[0]
    A_line = A[jr, ix] / A_far
    k1f = fit_k1(u2, x, k1t)
    c_rows, _xinv = row_demod(u2, x, k1f)
    P = profile_numbers(y, jr, c_rows, k1f, mu, (args.far_from * L, (args.far_from + 2.0) * L))
    A_far_dm = P["A_far"]
    if not np.isfinite(A_far_dm):
        sel = np.abs(y - y_row) > 2.0 * L
        A_far_dm = float(np.median(np.abs(c_rows[sel]))) if sel.any() else float(np.median(np.abs(c_rows)))
        flags.append("demod A_far from rows beyond 2 L (no rows at far_from)")
    Cmap = local_row_demod(u2, x, k1f, args.loc_L * L)
    bmin = args.band_min if args.band_min > 0 else max(1.0, 1.5 * node)
    jb = np.where(np.abs(y - y_row) < bmin * L)[0]
    yc_min, dc = column_minima(Cmap[:, ix], y, jb, y_row, A_far_dm, args.min_contrast)
    yg_min, dg = column_minima(A[:, ix], y, jb, y_row, A_far, args.min_contrast)
    xcol = x[ix]
    okc = np.isfinite(yc_min)
    c_node = float(np.nanmedian(np.abs(yc_min[okc]))) / L if okc.any() else float("nan")

    # (6) the descriptive loop map (domain figure only)
    step = args.grid_L * L
    cxs = np.arange(xm - 0.4 * Lx + r, xm + 0.4 * Lx - r + 1e-9, step)
    ylo = max(y_row - args.band_L * L, ym_ - 0.4 * Ly + r); yhi = min(y_row + args.band_L * L, ym_ + 0.4 * Ly - r)
    cys = np.arange(ylo, yhi + 1e-9, step)
    CX, CY = np.meshgrid(cxs, cys)
    circ_g, _kc = loop_grid(K1, K2, x, y, CX.ravel(), CY.ravel(), r, args.nang)
    tw_g, g_g = loop_phase_winding(e["z"], d1, d2, x, y, CX.ravel(), CY.ravel(), r, args.nang_phase)
    circ_g, tw_g, g_g = circ_g.reshape(CX.shape), tw_g.reshape(CX.shape), g_g.reshape(CX.shape)

    # (7) regime by Ed's rule
    has_charge = len(bq) > 0
    n_on, n_off = int(on_d.sum()), int((~on_d).sum())
    n_on_w = int((on_d & wound_d).sum()); n_off_w = int((~on_d & wound_d).sum())
    if not has_charge:
        regime = "knee"
        if n_on_w or n_off_w:
            flags.append("no charge, yet %d dips wind" % (n_on_w + n_off_w))
    elif n_on_w == 0 and n_off_w == 0:
        regime = "intermediate"
        if n_on:
            flags.append("%d dips on the line, none wound" % n_on)
        if n_off == 0:
            flags.append("no dips (charge without a %.0f percent dip)" % (100 * args.core_frac))
    elif n_on_w > 0 and n_off_w == 0:
        regime = "dislocation"
        if n_on_w < 0.9 * n_on:
            flags.append("partly wound: %d of %d dips on the line wind" % (n_on_w, n_on))
    else:
        regime = "unresolved"
        flags.append("dips off the line wind (%d of %d); on the line %d of %d" % (n_off_w, n_off, n_on_w, n_on))
    if (q2_abs > 0) != has_charge:
        flags.append("charge at sigma_det %.2g: %d cells, at %.2g: %d (the widths disagree on whether charge is present)"
                     % (args.sigma_det, len(bq), args.sigma_det2, len(q2)))
    if 2 * r < wavelength:
        flags.append("dip loops (diameter %.2f) narrower than a stripe wavelength (%.2f)" % (2 * r, wavelength))

    # (8) energy
    E_GB, flat, esrc = float("nan"), float("nan"), R["energy_source"]
    if R["e2"] is not None:
        E_GB, flat, how = _line_energy(R["e2"], y, jr, args.edge_frac)
        esrc = "%s via %s" % (esrc, how)
    elif np.isfinite(R["E_GB_stored"]):
        E_GB, esrc = R["E_GB_stored"], "E_GB stored in the cache record"

    row = dict(stem=stem, mu=mu, alpha_deg=float(np.degrees(np.arcsin(mu))), t=R["t"], frame=R["frame"], L=L, nx=nx, ny=ny,
               layout=R["layout"], regime=regime, flags="; ".join(flags),
               n_V=int(Vm.sum()), n_X=int(Xm.sum()), charge_abs_per_L=float(np.abs(bq).sum() / line_len_L),
               charge_net=int(bq.sum()) if len(bq) else 0, charge_abs_per_L_sd2=q2_abs / line_len_L, charge_net_sd2=q2_net,
               V_offline_med_L=med(v_off), V_offline_min_L=float(v_off.min()) if len(v_off) else float("nan"),
               V_offline_max_L=float(v_off.max()) if len(v_off) else float("nan"), X_offline_med_L=med(x_off),
               VX_sep_med_L=med(vx_sep), XX_sep_med_L=med(xx_sep),
               n_dips=len(xd_), n_dips_on=n_on, n_dips_off=n_off, n_dips_on_wound=n_on_w, n_dips_off_wound=n_off_w,
               dips_off_med_L=med(d_off[~on_d]) if n_off else float("nan"),
               dip_depth_on_med=med(dep_d[on_d]) if n_on else float("nan"),
               dip_depth_off_med=med(dep_d[~on_d]) if n_off else float("nan"),
               dip_circk_on_med=med(rec_circ[on_d]) if n_on else float("nan"),
               dip_circk_off_med=med(rec_circ[~on_d]) if n_off else float("nan"),
               dip_odd_twist_frac=float(np.mean(tw_d % 2 != 0)) if len(tw_d) else float("nan"),
               loop_r_L=r / L, loop_r_lambda=r / wavelength,
               A_at_V_med=med(A_at_V), A_line_med=float(np.median(A_line)), A_line_min=float(np.min(A_line)),
               node_L=node, cmin_offline_med_L=c_node, cmin_over_node_med=c_node / node if np.isfinite(c_node) else float("nan"),
               cmin_depth_med=float(np.median(dc)), mgcol_offline_med_L=float(np.nanmedian(np.abs(yg_min))) / L if np.isfinite(yg_min).any() else float("nan"),
               loop_diameter_over_wavelength=2 * r / wavelength,
               map_frac_odd_twist=float(np.mean(tw_g % 2 != 0)),
               A_far_mg=A_far, A_far_demod=A_far_dm, k1_fit=k1f,
               rec_n_dips=int(len(xd_)), rec_n_wound=int(np.sum(rec_circ >= args.wind_hi)) if len(rec_circ) else 0,
               rec_circ_max=float(rec_circ.max()) if len(rec_circ) else 0.0, rec_charge_sum_abs=int(det["sum_abs"]),
               circ_check_maxdiff=chk, E_GB=E_GB, E_flatness=flat, energy_source=esrc, elapsed_s=float(time.time() - t0))

    od = outroot / stem; od.mkdir(parents=True, exist_ok=True)
    with open(od / "markers.json", "w") as fh:
        json.dump(dict(row=row,
                       cells=[dict(x=float(a), y=float(b), q=int(c)) for a, b, c in zip(bx, by, bq)],
                       cells_sigma_det2=[dict(x=float(a), y=float(b), q=int(c)) for a, b, c in q2],
                       dips=[dict(x=float(a), y=float(b), off_L=float(o), on_line=bool(on), depth=float(dp), circ_k=float(c),
                                  twist=int(t), gamma=None if not np.isfinite(g) else float(g), wound=bool(w))
                             for a, b, o, on, dp, c, t, g, w in zip(xd_, yd_, d_off, on_d, dep_d, rec_circ, tw_d, g_d, wound_d)],
                       line_y=y_row, L=L), fh, indent=1)
    with open(od / "columns.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["x_over_L", "cmin_y_over_L", "cmin_depth", "mgmin_y_over_L", "mgmin_depth", "A_line"])
        for a, b, c, d_, e_, f_ in zip((xcol - xm) / L, yc_min / L, dc, yg_min / L, dg, A_line):
            wr.writerow(["%.5f" % a, "%.5f" % b, "%.5f" % c, "%.5f" % d_, "%.5f" % e_, "%.5f" % f_])
    np.savez_compressed(od / "circ_map.npz", cx=CX, cy=CY, circ_k=circ_g, twist=tw_g, gamma=g_g, loop_r=r, line_y=y_row, L=L)

    G = dict(stem=stem, mu=mu, L=L, node=node, x=x, y=y, xm=xm, y_row=y_row, u=u2, A=A / A_far, C=Cmap / A_far_dm,
             win=(xm - 0.4 * Lx, xm + 0.4 * Lx, ym_ - 0.4 * Ly, ym_ + 0.4 * Ly), band_L=args.band_L,
             CX=CX, CY=CY, circ=np.abs(circ_g), even=(tw_g % 2 == 0), r=r, loc=args.loc_L * L,
             bx=bx, by=by, bq=bq, dips=list(zip(xd_, yd_, wound_d, tw_d % 2 != 0, rec_circ)),
             xcol=xcol, ix=ix, dc=dc, yc_min=yc_min, yg_min=yg_min, regime=regime, t=R["t"], jr=jr, n_on=n_on, n_on_w=n_on_w)
    if not quiet_figs:
        fig_domain(G, od / "fig_domain.png", args)
        fig_markers(G, od / "fig_markers.png", args)
    lines.append("%s: mu %.3f (%.1f deg, L %.2f), t %.2f, %s; regime %s%s" % (
        stem, mu, row["alpha_deg"], L, R["t"], R["layout"], regime, (" [FLAGS: %s]" % row["flags"]) if flags else ""))
    lines.append("   charge (sigma_det %.2g): %d V cells (-1), %d X cells (+1) in the band (net %+d, %.2f |q| per L); V %.2f L off the line"
                 " (range %.2f to %.2f), X %.2f L off; V-X %.2f L, X-X %.2f L; at sigma_det %.2g: %d cells (net %+d)" % (
                     args.sigma_det, row["n_V"], row["n_X"], row["charge_net"], row["charge_abs_per_L"], row["V_offline_med_L"],
                     row["V_offline_min_L"], row["V_offline_max_L"], row["X_offline_med_L"], row["VX_sep_med_L"], row["XX_sep_med_L"],
                     args.sigma_det2, len(q2), q2_net))
    lines.append("   dip loops: radius %.2f L = %.2f stripe wavelengths" % (r / L, r / wavelength))
    lines.append("   dips: %d on the line (%d wind; depth %.3f; k circ %.3f), %d off the line (median %.2f L; %d wind; depth %.3f;"
                 " k circ %.3f); odd twist on %.0f%% of the dip loops" % (
                     n_on, n_on_w, row["dip_depth_on_med"], row["dip_circk_on_med"], n_off, row["dips_off_med_L"], n_off_w,
                     row["dip_depth_off_med"], row["dip_circk_off_med"],
                     100 * row["dip_odd_twist_frac"] if np.isfinite(row["dip_odd_twist_frac"]) else float("nan")))
    lines.append("   amplitude (over far): at the V cells %.3f; on the line median %.3f, min %.3f; |c| beat-node trough %.2f L off the line"
                 " (%.2f of the node), depth %.3f" % (row["A_at_V_med"], row["A_line_med"], row["A_line_min"], c_node,
                                                      row["cmin_over_node_med"], row["cmin_depth_med"]))
    lines.append("   records' convention: %d dips, %d wound, circ max %.3f, charge sum %d (loop code against loop_circulation: max"
                 " difference %.1e); energy E_GB %s (flatness %s; %s); %.1f s" % (
                     len(xd_), row["rec_n_wound"], row["rec_circ_max"], row["rec_charge_sum_abs"], chk,
                     ("%.5g" % E_GB) if np.isfinite(E_GB) else "n/a", ("%.3f" % flat) if np.isfinite(flat) else "n/a", esrc,
                     row["elapsed_s"]))
    return row, G


# ---------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------

def draw_markers(ax, G, args, curve="both", legend=False):
    """V and X cells, the dips with their loops (solid: wound; dashed: even twist, not wound; dotted: odd twist), the
    column minima, the line and the beat nodes, a 1 L bar; axes in units of L about (xm, y_row)"""
    hx = args.crop_x_L / 2
    ax.axhline(0, color="w", lw=0.8, ls="--")
    for s in (1, -1):
        ax.axhline(s * G["node"], color="#6fd3e8", lw=0.8, ls=":")
    pos, neg = G["bq"] > 0, G["bq"] < 0
    ax.scatter(_lx(G, G["bx"][pos]), _ly(G, G["by"][pos]), marker="^", s=24, c=C_POS, edgecolors="k", linewidths=0.3, zorder=6,
               label="X cell (+1 half turn)")
    ax.scatter(_lx(G, G["bx"][neg]), _ly(G, G["by"][neg]), marker="v", s=24, c=C_NEG, edgecolors="k", linewidths=0.3, zorder=6,
               label="V cell (-1 half turn)")
    first = {"w": True, "e": True, "o": True}
    for a, b, wnd, odd, c in G["dips"]:
        if wnd:
            kw = dict(ec=C_WIND, ls="-", lw=1.4); key, lab = "w", "dip loop, wound (|Gamma| = 1)"
        elif odd:
            kw = dict(ec="r", ls=":", lw=1.1); key, lab = "o", "dip loop, odd twist (undefined)"
        else:
            kw = dict(ec="#9a9a9a", ls="--", lw=1.0); key, lab = "e", "dip loop, not wound"
        ax.add_patch(Circle((_lx(G, a), _ly(G, b)), G["r"] / G["L"], fill=False, zorder=5, **kw))
        ax.plot(_lx(G, a), _ly(G, b), "o", ms=3.5, color=kw["ec"], zorder=5, label=lab if first[key] else None)
        first[key] = False
    if curve in ("both", "c"):
        ax.plot(_lx(G, G["xcol"]), G["yc_min"] / G["L"], ".", color=C_CUT1, ms=2.0, zorder=4, label="column minimum of |c|")
    if curve in ("both", "mg"):
        ax.plot(_lx(G, G["xcol"]), G["yg_min"] / G["L"], ".", color=C_CUT2, ms=2.0, zorder=4, label="column minimum of mg A")
    ax.plot([hx - 1.15, hx - 0.15], [-args.crop_y_L + 0.15] * 2, "w-", lw=2.5)
    ax.text(hx - 0.65, -args.crop_y_L + 0.22, "1 L", color="w", ha="center", va="bottom", fontsize=7)
    ax.set_xlim(-hx, hx); ax.set_ylim(-args.crop_y_L, args.crop_y_L)
    if legend:
        ax.legend(fontsize=6, loc="upper left", framealpha=0.85)


def fig_markers(G, path, args):
    mx, my, ext = _crop(G, args)
    fig = plt.figure(figsize=(11, 7.6))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.18)
    axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])
    gsd = gs[1, 1].subgridspec(2, 1, hspace=0.55)
    axd1, axd2 = fig.add_subplot(gsd[0]), fig.add_subplot(gsd[1])
    axa.imshow(G["u"][np.ix_(my, mx)], origin="lower", extent=ext, cmap="gray", aspect="equal")
    draw_markers(axa, G, args, "both", legend=True)
    axa.set_title("(a) pattern u with the markers", fontsize=9)
    im = axb.imshow(G["A"][np.ix_(my, mx)], origin="lower", extent=ext, cmap="magma", vmin=0.3, vmax=1.2, aspect="equal")
    draw_markers(axb, G, args, "mg"); fig.colorbar(im, ax=axb, shrink=0.8, label="A / A_far (mg)")
    axb.set_title("(b) mg amplitude", fontsize=9)
    im = axc.imshow(G["C"][np.ix_(my, mx)], origin="lower", extent=ext, cmap="magma", vmin=0.3, vmax=1.2, aspect="equal")
    draw_markers(axc, G, args, "c"); fig.colorbar(im, ax=axc, shrink=0.8, label="|c| / A_far (row demodulation)")
    axc.set_title("(c) local row demodulation |c| (beat-node measure)", fontsize=9)
    for ax in (axa, axb, axc):
        ax.set_xlabel("x / L"); ax.set_ylabel("(y - line) / L")
    xs = (G["x"][mx] - G["xm"]) / G["L"]
    axd1.plot(xs, G["A"][G["jr"], mx], color=C_CUT2, lw=1, label="mg A")
    axd1.plot(xs, G["C"][G["jr"], mx], color=C_CUT1, lw=1, label="|c|")
    for a, b, wnd, odd, c in G["dips"]:
        if abs(_lx(G, a)) < args.crop_x_L / 2:
            axd1.axvline(_lx(G, a), color=C_WIND if wnd else MUTED, lw=0.7, ls="-" if wnd else ":")
    axd1.set_xlabel("x / L (along the line; vertical lines: dips, solid if wound)"); axd1.set_ylabel("amplitude / far")
    axd1.legend(fontsize=7)
    axd1.set_title("(d) cuts: along the line, and across it through the deepest |c| column", fontsize=9)
    cand = np.where(np.abs(G["xcol"] - G["xm"]) <= args.crop_x_L / 2 * G["L"])[0]
    ic = int(cand[np.argmin(G["dc"][cand])])
    i_full = int(G["ix"][ic])
    ysel = np.abs(G["y"] - G["y_row"]) <= args.crop_y_L * G["L"]
    ys = (G["y"][ysel] - G["y_row"]) / G["L"]
    axd2.plot(ys, G["A"][ysel, i_full], color=C_CUT2, lw=1, label="mg A")
    axd2.plot(ys, G["C"][ysel, i_full], color=C_CUT1, lw=1, label="|c|")
    for s in (1, -1):
        axd2.axvline(s * G["node"], color="#2aa7c0", lw=0.8, ls=":")
    axd2.axvline(0, color=MUTED, lw=0.6, ls="--")
    axd2.set_xlabel("(y - line) / L at x = %.2f L (dotted: beat nodes)" % _lx(G, G["xcol"][ic])); axd2.set_ylabel("amplitude / far")
    fig.suptitle("mu = %.3f (%.1f deg), t = %.1f: %s;  dips on the line wound %d of %d (loop r = %.2g L)" % (
        G["mu"], np.degrees(np.arcsin(G["mu"])), G["t"], G["regime"], G["n_on_w"], G["n_on"], G["r"] / G["L"]), fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_domain(G, path, args):
    """F0: the band and window, both counting rectangles, the dip loops and cells, and the descriptive loop map"""
    x, y, L = G["x"], G["y"], G["L"]
    x0, x1, y0, y1 = G["win"]
    my = np.abs(y - G["y_row"]) <= (G["band_L"] + 0.5) * L
    ext = [(x[0] - G["xm"]) / L, (x[-1] - G["xm"]) / L, (y[my][0] - G["y_row"]) / L, (y[my][-1] - G["y_row"]) / L]
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.6), gridspec_kw=dict(height_ratios=[1, 1.4]))
    ax = axs[0]
    ax.imshow(G["u"][my], origin="lower", extent=ext, cmap="gray", aspect="auto")
    ax.add_patch(Rectangle(((x0 - G["xm"]) / L, -G["band_L"]), (x1 - x0) / L, 2 * G["band_L"], fill=False, ec="w", lw=0.8, ls=":"))
    pos, neg = G["bq"] > 0, G["bq"] < 0
    ax.scatter(_lx(G, G["bx"][pos]), _ly(G, G["by"][pos]), marker="^", s=10, c=C_POS, lw=0)
    ax.scatter(_lx(G, G["bx"][neg]), _ly(G, G["by"][neg]), marker="v", s=10, c=C_NEG, lw=0)
    for a, b, wnd, odd, c in G["dips"]:
        ax.add_patch(Circle((_lx(G, a), _ly(G, b)), G["r"] / L, fill=False, ec=C_WIND if wnd else ("r" if odd else "#9a9a9a"), lw=0.8))
    ax.set_title("band (dotted), dip loops (purple: wound; red: odd twist; grey: not wound), V and X cells", fontsize=9)
    ax.set_xlabel("x / L"); ax.set_ylabel("(y - line) / L")
    ax = axs[1]
    mx, myc, ext2 = _crop(G, args)
    ax.imshow(G["u"][np.ix_(myc, mx)], origin="lower", extent=ext2, cmap="gray", aspect="equal")
    sel = (np.abs(_lx(G, G["CX"])) <= args.crop_x_L / 2) & (np.abs(_ly(G, G["CY"])) <= args.crop_y_L)
    cl, uc = sel & G["even"], sel & ~G["even"]
    sc = ax.scatter(_lx(G, G["CX"][cl]), _ly(G, G["CY"][cl]), c=G["circ"][cl], s=5, cmap="viridis", vmin=0, vmax=1, alpha=0.8, lw=0, marker="s")
    ax.scatter(_lx(G, G["CX"][uc]), _ly(G, G["CY"][uc]), c="r", s=6, marker="x", lw=0.5)
    fig.colorbar(sc, ax=ax, shrink=0.8, label="|k circulation| of the 0.5 L loop centred there (descriptive)")
    cx0, cy0 = -args.crop_x_L / 4, 0.5
    ax.add_patch(Circle((cx0, cy0), G["r"] / L, fill=False, ec="c", lw=1.5))
    ax.plot([cx0 - G["loc"] / L, cx0 + G["loc"] / L], [cy0 - 0.75] * 2, "-", color="orange", lw=2)
    ax.text(1.02, 0.98, "to scale:\nloop r = %.2g L (cyan)\nrow-demodulation Gaussian\n+- 1 sd = %.2g L (orange bar)\n\nred x: odd twist\n(circulation undefined)"
            % (G["r"] / L, G["loc"] / L), transform=ax.transAxes, color="k", fontsize=7, va="top", ha="left")
    ax.set_xlim(-args.crop_x_L / 2, args.crop_x_L / 2); ax.set_ylim(-args.crop_y_L, args.crop_y_L)
    ax.set_title("the F1 crop: the 0.5 L loop map (grid %.3g L); below mu ~0.87 a loop is narrower than a wavelength" % args.grid_L, fontsize=9)
    ax.set_xlabel("x / L"); ax.set_ylabel("(y - line) / L")
    fig.suptitle("mu = %.3f: domain of the measures" % G["mu"], fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_positions(rows, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    _regime_spans(ax, rows)
    mu = np.array([r["mu"] for r in rows]); g = lambda k: np.array([r[k] for r in rows], float)
    mm = np.linspace(mu.min(), mu.max(), 200)
    ax.plot(mm, np.sqrt(1 - mm ** 2) / (2 * mm), "k--", lw=1, label="beat node k1/(2 mu)")
    v, lo, hi = g("V_offline_med_L"), g("V_offline_min_L"), g("V_offline_max_L")
    ax.errorbar(mu, v, yerr=[v - lo, hi - v], fmt="v", color=C_NEG, ms=4, lw=0.8, capsize=2, label="V cells (median, range)")
    ax.plot(mu, g("X_offline_med_L"), "^", color=C_POS, ms=4, label="X cells (median)")
    ax.plot(mu, g("dips_off_med_L"), "o", mfc="none", color=MUTED, ms=5, label="dips off the line (median)")
    ax.plot(mu, g("cmin_offline_med_L"), ".", color=C_CUT1, ms=5, label="|c| beat-node trough")
    ax.set_xlabel("mu = sin(far-field angle)"); ax.set_ylabel("distance from the line (L)")
    ax.set_ylim(bottom=-0.02); ax.legend(fontsize=7); _deg_axis(ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_magnitudes(rows, path):
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.8), sharex=True)
    mu = np.array([r["mu"] for r in rows]); g = lambda k: np.array([r[k] for r in rows], float)
    for ax in axs.ravel():
        _regime_spans(ax, rows)
    ax = axs[0, 0]
    ax.plot(mu, g("charge_abs_per_L"), "^-", color=C_POS, ms=3, label="sum |q| per L, sigma_det 0.3")
    ax.plot(mu, g("charge_abs_per_L_sd2"), "^--", color=C_POS, ms=3, mfc="none", label="sigma_det 0.2")
    ax.set_ylabel("charge per L"); ax.legend(fontsize=7)
    ax = axs[0, 1]
    ax.plot(mu, g("VX_sep_med_L"), "D-", color=C_NEG, ms=3, label="V-X separation")
    ax.plot(mu, g("XX_sep_med_L"), "^-", color=C_POS, ms=3, mfc="none", label="X-X separation")
    ax.plot(mu, g("V_offline_med_L"), "v:", color=C_NEG, ms=3, mfc="none", label="V distance from the line")
    ax.set_ylabel("distance (L)"); ax.legend(fontsize=7)
    ax = axs[1, 0]
    fr = g("n_dips_on_wound") / np.maximum(g("n_dips_on"), 1)
    ax.plot(mu, np.where(g("n_dips_on") > 0, fr, np.nan), "o-", color=C_WIND, ms=3, label="fraction of on-line dips wound")
    ax.plot(mu, g("dip_circk_on_med"), "D", color=C_WIND, ms=3, mfc="none", label="k circulation at on-line dips (median)")
    ax.plot(mu, g("dip_circk_off_med"), "x", color=MUTED, ms=4, label="k circulation at off-line dips (median)")
    ax.plot(mu, g("dip_odd_twist_frac"), ":", color="r", lw=1, label="fraction of dip loops with odd twist")
    ax.axhline(1, color=MUTED, lw=0.6, ls=":")
    ax.set_ylabel("winding"); ax.set_xlabel("mu"); ax.legend(fontsize=6)
    ax = axs[1, 1]
    ax.plot(mu, g("A_at_V_med"), "v-", color=C_NEG, ms=3, label="mg A at the V cells")
    ax.plot(mu, g("A_line_min"), "-", color=C_CUT2, lw=1, label="mg A on the line, min")
    ax.plot(mu, g("A_line_med"), "--", color=C_CUT2, lw=1, label="mg A on the line, median")
    ax.plot(mu, g("cmin_depth_med"), ".", color=C_CUT1, ms=4, label="|c| at the beat node")
    ax.set_ylabel("amplitude / far"); ax.set_xlabel("mu"); ax.legend(fontsize=6)
    _regime_legend(axs[0, 0]); _deg_axis(axs[0, 0]); _deg_axis(axs[0, 1])
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_energy(rows, path):
    rr = [r for r in rows if np.isfinite(r["E_GB"])]
    if not rr:
        return False
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    _regime_spans(ax, rr)
    ax.plot([r["mu"] for r in rr], [r["E_GB"] for r in rr], "o-", color="k", ms=3)
    ax.set_xlabel("mu"); ax.set_ylabel("line energy E_GB (excess per unit length)")
    _regime_legend(ax); _deg_axis(ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return True


def fig_montage(Gs, idx, args, path=None, pdf=None, per_page=8):
    pages = [idx[s:s + per_page] for s in range(0, len(idx), per_page)] if pdf is not None else [idx]
    for page in pages:
        n = len(page)
        if n == 0:
            continue
        fig, axs = plt.subplots(3, n, figsize=(2.3 * n + 0.6, 6.4), squeeze=False)
        for j, i in enumerate(page):
            G = Gs[i]; mx, my, ext = _crop(G, args)
            axs[0, j].imshow(G["u"][np.ix_(my, mx)], origin="lower", extent=ext, cmap="gray", aspect="equal")
            draw_markers(axs[0, j], G, args, "none")
            axs[0, j].set_title("mu %.2f (%.0f deg)\n%s" % (G["mu"], np.degrees(np.arcsin(G["mu"])), G["regime"]), fontsize=7)
            axs[1, j].imshow(G["A"][np.ix_(my, mx)], origin="lower", extent=ext, cmap="magma", vmin=0.3, vmax=1.2, aspect="equal")
            draw_markers(axs[1, j], G, args, "none")
            axs[2, j].imshow(G["C"][np.ix_(my, mx)], origin="lower", extent=ext, cmap="magma", vmin=0.3, vmax=1.2, aspect="equal")
            for ax in axs[:, j]:
                ax.set_xticks([]); ax.set_yticks([])
        axs[0, 0].set_ylabel("pattern", fontsize=8); axs[1, 0].set_ylabel("mg A / far", fontsize=8); axs[2, 0].set_ylabel("|c| / far", fontsize=8)
        fig.tight_layout()
        if pdf is not None:
            pdf.savefig(fig)
        else:
            fig.savefig(path, dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------

MD_KEYS = ["mu", "alpha_deg", "t", "regime", "charge_abs_per_L", "charge_abs_per_L_sd2", "V_offline_med_L", "X_offline_med_L",
           "VX_sep_med_L", "XX_sep_med_L", "n_dips_on", "n_dips_on_wound", "n_dips_off", "dips_off_med_L", "dip_circk_on_med",
           "dip_odd_twist_frac", "A_at_V_med", "A_line_min", "cmin_over_node_med", "rec_n_dips", "rec_n_wound", "rec_circ_max", "E_GB", "flags"]


def write_tables(rows, out):
    keys = list(rows[0].keys())
    with open(out / "survey_table.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
    with open(out / "survey_table.md", "w") as fh:
        fh.write("| " + " | ".join(MD_KEYS) + " |\n|" + "---|" * len(MD_KEYS) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(("%.3f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in MD_KEYS) + " |\n")


def set_figures(rows, Gs, args, out):
    fig_positions(rows, out / "fig_positions_vs_mu.png")
    fig_magnitudes(rows, out / "fig_magnitudes_vs_mu.png")
    has_E = fig_energy(rows, out / "fig_energy_vs_mu.png")
    sel = montage_select(rows, args.montage_step)
    fig_montage(Gs, sel, args, path=out / "fig_montage.png")
    with PdfPages(out / "fig_montage_all.pdf") as pdf:
        fig_montage(Gs, list(range(len(Gs))), args, pdf=pdf)
    return has_E, sel


def header_line(args):
    return ("%s; sigma_dir %.2g, sigma_det %.2g (and %.2g), band %.2g L, dip loops %s (%d samples for the twist and Gamma),"
            " on the line <= %.2g L, dips below %.2g of the median A, frame %d" % (
                VERSION, args.sigma_dir, args.sigma_det, args.sigma_det2, args.band_L,
                ("r %.2g stripe wavelengths" % args.loop_lam) if args.loop_lam > 0 else ("r %.2g L" % args.loop_L), args.nang_phase,
                args.on_line_L, args.core_frac, args.frame))


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
        lines.append("FLAG: no energy found in any file (no 'e' array or cache E_GB); F6 not drawn")
    lines.append("montage (F2) angles: " + ", ".join("%.3f" % rows[i]["mu"] for i in sel))
    lines.append("elapsed %.1f s" % (time.time() - t0))
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-4:]))
    print("outputs ->", out)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    npz = ""                       # comma-separated files and/or folders (every *.npz in a folder)
    outdir = str(Path("results") / "pgb_marker_survey")
    frame = -1                     # frame index; -1 = last
    tag_frame = 0                  # 1: append _f<frame> to each file's folder name
    mu = 0.85                      # fallback when a file carries no mu
    sigma_dir = 0.4                # mg director width (production value)
    taper_frac = 0.06
    sigma_det = 0.3                # director-charge width (the records' value; the regime uses it)
    sigma_det2 = 0.2               # second detection width (the gallery reads 0.3 and 0.2)
    band_L = 2.0                   # band half-height, L
    loop_L = 0.5                   # dip loop radius, L (the validated radius)
    nang = 181                     # samples per loop for the k circulation (loop_circulation's value)
    nang_phase = 720               # samples per circuit for the twist and the phase winding (the gallery's n_circuit)
    grid_L = 0.125                 # loop-map spacing, L (domain figure only)
    loop_lam = 0.0                 # > 0: dip-loop radius in far-field stripe wavelengths instead of loop_L (the loop-size control)
    wind_hi = 0.9                  # turns: the records' convention only (rec_n_wound)
    loc_L = 0.5                    # Gaussian standard deviation of the local row demodulation, L
    band_min = -1.0                # search half-width for the column minima, L; <= 0: max(1, 1.5 node)
    min_contrast = 0.03            # a column needs this contrast (over far field) to have a minimum position
    far_from = 4.0                 # far field for A_far, L from the line
    core_frac = 0.70               # dips: mg A below this fraction of its window median (the records' value)
    on_line_L = 0.1                # a dip within this distance of the line is on the line, L
    crop_x_L = 4.0                 # figure crop along the line, L
    crop_y_L = 1.5                 # figure crop half-height, L
    montage_step = 0.02            # F2: every montage_step in mu, plus the angles beside a regime change
    energy_key = ""                # energy-density array in the movie files ("" tries e, energy_density, E_density)
    edge_frac = 0.12               # line energy: rows at each edge for the roll baseline (the sweep's value)


def default_args():
    """an argparse-like namespace with every default, for importing scripts"""
    return argparse.Namespace(**{k: getattr(_Args, k) for k in vars(_Args) if not k.startswith("_")})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in vars(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    sys.exit(main(p.parse_args()))
