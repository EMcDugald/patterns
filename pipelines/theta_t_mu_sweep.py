"""theta_t_mu_sweep.py  v1.3.1  (13 Oct 2026)   workstreams B0, G11, G12, G14 of the research thread

Rate-law sweep driver: for every movie given, (1) a per-frame record of the
dislocation formation read from the extracted fields (amplitude dips, the
circulation of the extracted k on the validated 0.5 L loop about each dip,
the trough depth, the off-row distance and spacing of the dips).  Two things
were tried and dropped: the plaquette charge of arg z (the mg phase does not
wind about a core; charges appear only at the taper edge) and a second loop
radius (through mg the circulation reads 0 inside 0.4 L and is not clean at
0.6 L and beyond where the loop nears the neighbouring cores; 0.5 L is the
radius validated in mg_zigzag_theta_t v1.5 and is the only one used), (2) windows
cut from that record (pre-winding, straddle, post-winding, whole), (3) the
weak-form rate fits of wsindy_theta_t v1.9 on each window (free RHS+ext,
free RHS, forced RCN triple, forced {S pair, J}, forced RCN + J, tied {S, J},
{S, B}, {S, B, J}; the identity w_t = 2 k.grad theta_t; the Gram geometry
of the RCN columns), (4) fit maps of theta_t at the window's middle frame
with the expected RCN right-hand side and both residuals, (5) one
sweep_table.csv and a coefficients-against-mu figure over all files.

The frozen method layer (src/op_extract) and wsindy_theta_t_v19.py are
imported, never edited.  Every weak column, target and fit routine is
v1.9's; this script adds the record, the windows, the maps and the table.

Windows (frames; t_half is in frames, pad = t_half + 2):
  first   = first frame whose largest loop circulation reaches wind_lo turns
  wound   = first frame on which the number of dips with circulation >= wind_hi
            turns reaches 90 percent of its value on the last frame
  pre      = [0, first - 1]          straddle = [first - pad, wound + pad]
  post     = [wound + 1, last]       whole    = [0, last]
  to_wind  = [0, first wound]        (v1.3; first frame with a dip of >= wind_hi turns)
A window shorter than 2 t_half + 3 frames is skipped.  A movie with no
circulation has whole only (v1.3; v1.2 also fitted pre, the same frames);
a movie with no winding has no to_wind (it would be whole).  --windows overrides with explicit times,
e.g. "pre:0-2.8,straddle:3-7,post:7.2-25".

Outputs (under --outdir, one folder per file stem):
  record_<stem>.csv, fig_record_<stem>.png
  <window>/report.txt, fits.json, fit_maps_<window>.png, domains figure
  sweep_table.csv, sweep_table.md, fig_sweep_coefficients.png (top level)

Changelog
  v1.3.1 forced {w div k, k.grad w, w} and tied {S, w}, with their
        prediction scores and table columns (SPw_*, score_SPw). Reason:
        on Ed's local run of 12 Oct the free selection returns exactly
        {w div k, k.grad w, w} (about 1.0, 0.6, -0.15; R2 0.99; the best
        prediction score) in every window before the first circulation
        from 0.79 to 0.87, where the line row is still x-invariant. The
        forced fit puts that law beside the RCN triple on every window
        and on the control, where no such term exists.
  v1.3  (G12) the to_wind window [0, first wound dip], so the global law
        theta_t = a S + d J (tied {S, J}) is fitted from the start to the
        winding (or over the whole run when nothing winds);
        fig_global_law.png plots it against mu (to_wind where the run
        winds, whole where it does not). The duplicate pre window of a
        run without circulation is no longer fitted.
        (G14) prediction scores of v1.9 (prediction_scores, imported) on
        by default, score_h 6: for the free, forced RCN, {S pair, J},
        RCN + J, tied {S, J}, tied {S, B} and w-bending laws, the weak
        multi-step error ratio of the phase increment over 3 and 6
        frames on the holdout half (null 1); the 6-frame value goes to
        the table (score_*).
        (G11) w-weighted bending (Shankar's suggestion, the bending
        coefficient as a function of w): two extra weak columns built in
        this script from the same kernels, Lap(w div k) =
        Int w div k Lap(phi), and div(w grad div k) =
        Int div k [grad w . grad phi + w Lap phi], both with first
        derivatives of k only; forced fits RCN + Lap(w div k),
        RCN + div(w grad div k), tied {S, Lap div k, Lap(w div k)}.
        Validated on the RCN control (no w-bending in the truth): see the
        explanation file of 12 Oct.
  v1.2  --core_def charge: discs about the clusters (within --merge_dist L)
        of the director disclinations of the frozen t1c.detect_defects at
        sigma_det 0.3, the early index of the two-onset record. The record
        gains charge_sum_abs, charge_clusters and their off-row distance per
        frame. Measured on the t = 25 last frames: sum|q| 72 at 0.79 and 0.85
        (tall), 48 at 0.70 and 0.75 (fine sweep), so the charge discs also
        fall on the 0.65 to 0.70 knees that have no amplitude dip.
  v1.1  (b) expected panel carried through the C0 gain: the truth (2, 2, -1)
        applied to the extracted columns reads holdout R2 -0.32 on the mg
        control (near balance 2S - B is a small difference and the measured
        S carries 0.31 B); solved through the C0 gain it becomes (1.84, 1.84,
        -1.67) and reads 0.945 on the same control against the forced fit's
        0.955.  The raw-truth line stays in the report; the panel shows the
        gain-carried law on the target's colour scale.  --c0_gain sets the
        matrix (default the 0.85 control's; flagged at other mu).
  v1.1  (a) --core_def circulation (default): the core discs are drawn only about
        amplitude dips whose 0.5 L loop circulation reads >= 0.9 turns.  On the
        28 last frames of the fine sweep the dips at 0.71 to 0.81 (20 per crop,
        in pairs 0.30 to 0.42 L above and below the line) read 0.00 turns, and
        so do the 28 dips of the 0.79 and 0.81 files of record at t = 25: they
        are a split amplitude trough of an unwound knee, not dislocations.  v1.0
        put discs on them (and on the knee-line dips of every pre-winding
        frame); --core_def dips restores that.
  v1.0  record, windows, fits, fit maps, sweep table and figure; validated on
        the RCN knee control movie (no winding: pre = whole; forced triple
        through mg) and on the stand-in SH zigzag movie of
        sh_zigzag_standin.py (winding at mu 0.85 detected, windows cut).
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wsindy_theta_t_v19 as V                                           # noqa: E402
from wsindy_theta_t_v19 import (load_npz, estimate, find_cores, dist_to_cores, t1c,   # noqa: E402
                                kernel_sizes, query_centres, build_columns, theta_t_from_z,
                                signal_line, assemble, fit_pareto, forced_fit, collinearity,
                                cstr, RHS_LIB, RHS_EXT, RCN_RHS, r_squared, radii, _slug)
import matplotlib                                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

SP = ["w div k", "k.grad w"]
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
TRUTH = {"w div k": 2.0, "k.grad w": 2.0, "Lap div k": -1.0}


def director_charge_points(u2d, x, y, y_row, taper, Lsp, sigma_det=0.3, band_L=2.0, merge_L=0.5, win=None):
    """v1.2 / v1.3.3: director disclinations of the frozen t1c.detect_defects
    (Riesz director at sigma_det, plaquette winding of its angle), the
    index of the two-onset record; plaquettes with q != 0 within band_L
    spacings of the line, clustered within merge_L spacings. Returns the
    cluster positions (x, absolute y), the |q| summed per cluster, and
    the band's sum |q|."""
    ny, nx = u2d.shape
    dx = float(x[1] - x[0])
    det = t1c.detect_defects(u2d * taper, x, y - y_row, nx * dx, ny * dx,
                             sigma_det=sigma_det, band_y=band_L * Lsp)
    ch = det["charges"]
    if win is not None and len(ch):
        Lx = nx * dx; Ly = ny * dx
        ch = [c for c in ch if abs(c[0] - x.mean()) < 0.4 * Lx and abs(c[1] + y_row - y.mean()) < 0.4 * Ly]
    if not ch:
        return np.zeros(0), np.zeros(0), np.zeros(0), int(det["sum_abs"])
    px = np.array([c[0] for c in ch]); py = np.array([c[1] for c in ch]) + y_row
    pq = np.abs(np.array([c[2] for c in ch], float))
    order = np.argsort(px); px, py, pq = px[order], py[order], pq[order]
    used = np.zeros(len(px), bool); cx, cy, cq = [], [], []
    for i in range(len(px)):
        if used[i]:
            continue
        grp = (~used) & (np.hypot(px - px[i], py - py[i]) < merge_L * Lsp)
        used |= grp
        cx.append(px[grp].mean()); cy.append(py[grp].mean()); cq.append(pq[grp].sum())
    return np.array(cx), np.array(cy), np.array(cq), int(det["sum_abs"])


# ---------------------------------------------------------------------
# (1) the per-frame record
# ---------------------------------------------------------------------

def loop_circulation(k1, k2, x, y, xc, yc, r):
    """circulation of the extracted k on a circle of radius r about (xc, yc),
    in turns; the director sign is carried along the loop by continuity
    (mg_zigzag_theta_t v1.5, the validated r = 0.5 L loop)"""
    dx = x[1] - x[0]; dy = y[1] - y[0]
    ang = np.linspace(0, 2 * np.pi, 181)
    px = xc + r * np.cos(ang); py = yc + r * np.sin(ang)
    ix = np.clip(np.round((px - x[0]) / dx).astype(int), 0, len(x) - 1)
    iy = np.clip(np.round((py - y[0]) / dy).astype(int), 0, len(y) - 1)
    kx, ky = k1[iy, ix], k2[iy, ix]
    sg = np.ones_like(ang)
    for q in range(1, len(ang)):
        sg[q] = sg[q - 1] if kx[q] * kx[q - 1] + ky[q] * ky[q - 1] >= 0 else -sg[q - 1]
    tx, ty = -np.sin(ang), np.cos(ang)
    return float(_trapz((kx * tx + ky * ty) * sg, ang) * r / (2 * np.pi))


def movie_record(path, args, out, lines):
    D = load_npz(path)
    u, x, y, t, md = D["u"], D["x"], D["y"], D["t"], D["md"]
    mu = float(md.get("mu", args.mu)); k1 = np.sqrt(1 - mu ** 2); Lsp = np.pi / k1
    ny, nx, nt = u.shape
    dx = float(x[1] - x[0]); Lx, Ly = nx * dx, ny * dx
    X, Y = np.meshgrid(x, y)
    win = (np.abs(X - x.mean()) < 0.4 * Lx) & (np.abs(Y - y.mean()) < 0.4 * Ly)
    jc0 = int(md.get("gb_row_in_crop", ny // 2)); y_row = float(y[jc0])
    band = (np.abs(Y - y_row) < args.band_to * Lsp) & win
    taper = t1c.edge_taper(x, y, frac=args.taper_frac)
    rows = []
    for i in range(nt):
        e = estimate(u[..., i], Lx, Ly, args.sigma_dir, None, taper)     # no frame-to-frame director sign: the loop needs none
        A = e["A"]
        xc, yc = find_cores(A, x, y, win, args.core_frac, merge_dist=args.merge_dist * Lsp)
        circ = np.array([abs(loop_circulation(e["k1"], e["k2"], x, y, a, b, args.loop_r * Lsp)) for a, b in zip(xc, yc)])
        try:                                                              # v1.2: director charge (two-onset index)
            qx, qy, qq, qsum = director_charge_points(u[..., i], x, y, y_row, taper, Lsp, args.sigma_det_charge,
                                                      args.band_to, args.merge_dist, win=win)
        except Exception:
            qx, qy, qq, qsum = np.zeros(0), np.zeros(0), np.zeros(0), -1
        ref = float(np.median(A[win]))
        r = dict(frame=i, t=float(t[i]), n_dips=len(xc),
                 n_wound=int(np.sum(circ >= args.wind_hi)) if len(circ) else 0,
                 circ_max=float(circ.max()) if len(circ) else 0.0,
                 circ_median=float(np.median(circ)) if len(circ) else 0.0,
                 charge_sum_abs=qsum, charge_clusters=len(qx),
                 charge_offrow_median=float(np.median(np.abs(qy - y_row)) / Lsp) if len(qy) else np.nan,
                 trough=float(A[band].min() / ref) if band.any() else np.nan,
                 dip_offrow_median=float(np.median(np.abs(yc - y_row)) / Lsp) if len(yc) else np.nan,
                 dip_offrow_max=float(np.max(np.abs(yc - y_row)) / Lsp) if len(yc) else np.nan,
                 dip_spacing_median=float(np.median(np.diff(np.sort(xc))) / Lsp) if len(xc) > 1 else np.nan,
                 dip_nearest_median=float(np.median([np.min(np.hypot(xc - a, yc - b)[np.hypot(xc - a, yc - b) > 0]) for a, b in zip(xc, yc)]) / Lsp) if len(xc) > 1 else np.nan)
        rows.append(r)
    stem = Path(path).stem
    with open(out / ("record_%s.csv" % stem), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    lines.append("record %s: mu %.3f (L %.2f), %d frames t %.3f..%.3f (dt %.4g); dips %d..%d; wound (>= %.2f turns at r %.2g L) %d..%d;"
                 " trough %.3f..%.3f; dip off-row median %.2f..%.2f L" % (
                     stem, mu, Lsp, nt, t[0], t[-1], float(np.median(np.diff(t))) if nt > 1 else 0,
                     min(r["n_dips"] for r in rows), max(r["n_dips"] for r in rows), args.wind_hi, args.loop_r,
                     min(r["n_wound"] for r in rows), max(r["n_wound"] for r in rows),
                     np.nanmin([r["trough"] for r in rows]), np.nanmax([r["trough"] for r in rows]),
                     np.nanmin([r["dip_offrow_median"] for r in rows]), np.nanmax([r["dip_offrow_median"] for r in rows])))
    return rows, dict(mu=mu, Lsp=Lsp, t=t, nt=nt, stem=stem)


def fig_record(rows, info, wins, path, args):
    t = np.array([r["t"] for r in rows])
    g = lambda k: np.array([r[k] for r in rows], float)
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    ax = axs[0]
    ax.plot(t, g("n_dips"), "k-", label="amplitude dips (%.0f%% of median A, merged within %.2g L)" % (100 * args.core_frac, args.merge_dist))
    ax.plot(t, g("n_wound"), "-", color="C3", lw=2, label="dips with loop circulation >= %.2g turn (r = %.2g L)" % (args.wind_hi, args.loop_r))
    ax.plot(t, g("charge_sum_abs") / 2.0, "--", color="C0", lw=1.5, label="director charge sum|q| / 2 on the band (sigma_det %.2g)" % args.sigma_det_charge)
    ax.plot(t, g("charge_clusters"), ":", color="C0", lw=1.5, label="director charge clusters (merged within %.2g L)" % args.merge_dist)
    ax.set_ylabel("count"); ax.legend(fontsize=7, loc="best")
    ax = axs[1]
    ax.plot(t, g("circ_max"), "-", color="C3", label="largest |circulation| (turns)")
    ax.plot(t, g("circ_median"), "--", color="C3", label="median |circulation|")
    ax.axhline(args.wind_hi, color="0.6", lw=0.8); ax.axhline(args.wind_lo, color="0.8", lw=0.8)
    ax.set_ylabel("turns"); ax.set_ylim(-0.05, 1.3); ax.legend(fontsize=7, loc="best")
    ax = axs[2]
    ax.plot(t, g("trough"), "k-", label="trough: min A on the band / median A")
    ax.set_ylabel("A_min / A_med"); ax.legend(fontsize=7, loc="best")
    ax = axs[3]
    ax.plot(t, g("dip_offrow_median"), "-", color="C2", label="median |y_dip - y_row| / L")
    ax.plot(t, g("dip_offrow_max"), ":", color="C2", label="largest")
    ax.plot(t, g("dip_nearest_median"), "-", color="C4", label="median distance to the nearest other dip / L")
    ax.set_ylabel("L"); ax.set_xlabel("t"); ax.legend(fontsize=7, loc="best")
    cols = {"pre": "C0", "straddle": "C1", "post": "C2", "whole": "0.5"}
    for ax in axs:
        for nm, (t0, t1) in wins.items():
            if nm == "whole":
                continue
            ax.axvspan(t0, t1, color=cols.get(nm, "0.7"), alpha=0.12)
            ax.text(t0, ax.get_ylim()[1], nm, fontsize=7, va="top", color=cols.get(nm, "0.3"))
    fig.suptitle("%s, mu %.3f: formation record from the extracted fields (windows shaded)" % (info["stem"], info["mu"]), fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ---------------------------------------------------------------------
# (2) windows
# ---------------------------------------------------------------------

def windows_from_record(rows, args, lines):
    t = np.array([r["t"] for r in rows]); nt = len(t)
    minlen = 2 * args.t_half + 3
    pad = args.t_half + 2
    wins = {}
    if args.windows:
        for item in args.windows.split(","):
            nm, rng = item.split(":"); a, b = rng.split("-")
            wins[nm.strip()] = (float(a), float(b))
        lines.append("windows given on the command line: %s" % wins)
        return wins, dict(first=None, wound=None, w1=None)
    cmax = np.array([r["circ_max"] for r in rows]); nw = np.array([r["n_wound"] for r in rows])
    first = int(np.argmax(cmax >= args.wind_lo)) if np.any(cmax >= args.wind_lo) else None
    w1 = int(np.argmax(nw >= 1)) if np.any(nw >= 1) else None             # v1.3: first wound dip
    nfinal = nw[-1]
    wound = None
    if nfinal > 0:
        hit = np.where(nw >= 0.9 * nfinal)[0]
        wound = int(hit[0]) if len(hit) else None
    wins["whole"] = (float(t[0]), float(t[-1]))
    if first is None:
        lines.append("windows: no circulation above %.2f turns on any frame; whole only (pre would be the same frames)" % args.wind_lo)
        return wins, dict(first=None, wound=None, w1=None)
    if w1 is not None:
        if w1 + 1 >= minlen:
            wins["to_wind"] = (float(t[0]), float(t[w1]))
        else:
            lines.append("windows: only %d frames before the first wound dip (need %d); to_wind skipped" % (w1 + 1, minlen))
    if first - 1 >= minlen - 1:
        wins["pre"] = (float(t[0]), float(t[first - 1]))
    else:
        lines.append("windows: only %d frames before the first circulation (need %d); pre skipped" % (first, minlen))
    if wound is not None:
        a, b = max(0, first - pad), min(nt - 1, wound + pad)
        if b - a + 1 >= minlen:
            wins["straddle"] = (float(t[a]), float(t[b]))
        if nt - 1 - (wound + 1) + 1 >= minlen:
            wins["post"] = (float(t[wound + 1]), float(t[-1]))
        else:
            lines.append("windows: only %d frames after winding (need %d); post skipped" % (nt - 1 - wound, minlen))
    else:
        lines.append("windows: circulation starts (frame %d, t %.3f) but no dip reaches %.2f turns by the last frame; straddle = [first - pad, last]" % (first, t[first], args.wind_hi))
        a = max(0, first - pad)
        if nt - a >= minlen:
            wins["straddle"] = (float(t[a]), float(t[-1]))
    lines.append("windows (frames: first circulation %s, first wound dip %s, wound %s): %s" % (
        first, w1, wound, ", ".join("%s %.3f-%.3f" % (k, v[0], v[1]) for k, v in wins.items())))
    return wins, dict(first=first, wound=wound, w1=w1)


# ---------------------------------------------------------------------
# (3) rows of one window (mirror of v1.9 load_file_rows, with the centres kept)
# ---------------------------------------------------------------------

def window_rows(path, t0, t1, args, out, lines):
    D = load_npz(path)
    u, x, y, t, md = D["u"], D["x"], D["y"], D["t"], D["md"]
    exact = None
    if int(args.exact):
        d_ = np.load(path, allow_pickle=True)
        if all(k in d_.files for k in ("k1_exact", "k2_exact", "theta_t_exact")):
            exact = (d_["k1_exact"], d_["k2_exact"], d_["theta_t_exact"])
        else:
            lines.append("  --exact 1 but the file carries no exact fields; estimator route used")
    mu = float(md.get("mu", args.mu)); k1 = np.sqrt(1 - mu ** 2); Lsp = np.pi / k1
    sel = (t >= t0 - 1e-9) & (t <= t1 + 1e-9)
    u, t = u[..., sel], t[sel]
    if exact is not None:
        exact = tuple(np.asarray(a, float)[..., sel] for a in exact)
    fs = 1
    if int(args.max_frames) > 0 and u.shape[-1] > int(args.max_frames):
        fs = int(np.ceil(u.shape[-1] / float(args.max_frames)))
        u, t = u[..., ::fs], t[::fs]
        if exact is not None:
            exact = tuple(a[..., ::fs] for a in exact)
        lines.append("  window longer than max_frames %d: every %d-th frame kept (dt becomes %.4g; t_half %d frames now spans %.3g)" % (
            int(args.max_frames), fs, float(np.median(np.diff(t))), args.t_half, args.t_half * float(np.median(np.diff(t)))))
    ny, nx, nt = u.shape
    stem = Path(path).stem
    if nt < 2 * args.t_half + 3:
        lines.append("  only %d frames in [%.3f, %.3f]; need >= %d; skipped" % (nt, t0, t1, 2 * args.t_half + 3))
        return None
    dx = float(x[1] - x[0]); dt = float(np.median(np.diff(t)))
    Lx, Ly = nx * dx, ny * dx
    taper = t1c.edge_taper(x, y, frac=args.taper_frac)
    X, Y = np.meshgrid(x, y)
    win = (np.abs(X - x.mean()) < 0.4 * Lx) & (np.abs(Y - y.mean()) < 0.4 * Ly)
    jc0 = int(md.get("gb_row_in_crop", ny // 2))
    d_row = np.abs(Y - y[jc0]) / Lsp
    ests, n_prev = [], None
    for i in range(nt):
        e = estimate(u[..., i], Lx, Ly, args.sigma_dir, n_prev, taper)
        n_prev = e["n"]; ests.append(e)
    shape = u.shape
    u_mid = u[..., nt // 2].astype(np.float32)
    u_all = u if args.core_def == "charge" else None                    # v1.2: the charge index needs u per frame
    del u, D
    k1m = np.stack([e["k1"] for e in ests], -1); k2m = np.stack([e["k2"] for e in ests], -1)
    zs = np.stack([e["z"] for e in ests], -1)
    Am = np.stack([e["A"] for e in ests], -1)
    th_t = theta_t_from_z(zs, dt)
    del zs
    s = np.sign(np.mean(k1m, axis=(0, 1)))[None, None, :]; s[s == 0] = 1
    k1m, k2m, th_t = k1m * s, k2m * s, th_t * s
    if exact is not None:
        k1m, k2m, th_t = exact
        lines.append("  exact route: the file's k1, k2, theta_t replace the estimator's (amplitude and dips still from mg)")
    excl = np.zeros(shape, bool)
    ncores, cores_xy, ndips = [], [], []
    for i in range(nt):
        xc, yc = find_cores(ests[i]["A"], x, y, win, args.core_frac, merge_dist=args.merge_dist * Lsp)
        ndips.append(len(xc))
        if args.core_def == "circulation" and len(xc):                  # v1.1: discs only about wound dips
            circ = np.array([abs(loop_circulation(ests[i]["k1"], ests[i]["k2"], x, y, a, b, args.loop_r * Lsp)) for a, b in zip(xc, yc)])
            keep_c = circ >= args.wind_hi
            xc, yc = xc[keep_c], yc[keep_c]
        elif args.core_def == "charge":                                   # v1.2: discs about director-charge clusters
            xc, yc, _, _ = director_charge_points(u_all[..., i], x, y, float(y[jc0]), taper, Lsp, args.sigma_det_charge,
                                                  args.band_to, args.merge_dist, win=win)
        ncores.append(len(xc)); cores_xy.append((xc, yc))
        if args.core_excl > 0 and len(xc):
            excl[..., i] = dist_to_cores(X, Y, xc, yc) / Lsp < args.core_excl
    del ests, u_all
    lines.append("  cores (%s): amplitude dips per frame %d..%d, discs about %d..%d of them" % (
        ("dips whose %.2g L loop reads >= %.2g turns" % (args.loop_r, args.wind_hi)) if args.core_def == "circulation" else
        (("director-charge clusters (sigma_det %.2g; counts in this line are clusters)" % args.sigma_det_charge) if args.core_def == "charge" else "every amplitude dip"),
        min(ndips), max(ndips), min(ncores), max(ncores)))
    cores = excl.copy()
    if args.domain == "band":
        dom = d_row < args.band_to
    elif args.domain == "row":
        dom = d_row < args.row_to
    elif args.domain == "far":
        dom = d_row > args.far_from
    else:
        dom = np.ones_like(d_row, bool)
    if args.side == "upper":
        dom &= Y > y[jc0]
    elif args.side == "lower":
        dom &= Y < y[jc0]
    excl |= ~dom[..., None]
    if int(args.no_cross):
        my_nc = kernel_sizes(args, dx, dx, dt, Lsp)[1]
        excl |= (np.abs(Y - y[jc0]) < my_nc * dx - 1e-9)[..., None]
    far = d_row > args.far_from
    Abar = float(np.nanmedian(Am[far]) if far.any() else np.nanmedian(Am[win]))
    m = kernel_sizes(args, dx, dx, dt, Lsp)
    idx = query_centres(shape, m, args, excl=excl, cores=cores)
    if idx is None:
        lines.append("  no query centres survive; skipped")
        return None
    lines.append("  window t %.3f..%.3f: %d frames (dt %.4g), cores per frame %d..%d, %d query centres (stride %d, %d), domain %s, discs %.2g L" % (
        t[0], t[-1], nt, dt, min(ncores), max(ncores), len(idx[0]), args.stride_xy, args.stride_t, args.domain, args.core_excl))
    lines.append(signal_line(th_t, idx, rows=win.any(axis=1)))
    cols, targets, fields = build_columns(k1m, k2m, th_t, x, y, t, args, Lsp, idx, A=Am, Abar=Abar, snap=nt // 2)
    wbend_columns(cols, k1m, k2m, x, y, t, args, Lsp, idx)               # v1.3 (G11)
    R = dict(rhs=cols["rhs"], var=cols["var"], w=cols["w"], A=cols["A"], score=cols.get("score"), targets=targets, xs=x[idx[1]],
             idx=idx, x=x, y=y, t=t, Lsp=Lsp, mu=mu, y_row=float(y[jc0]), ncores=ncores, cores_xy=cores_xy,
             u_mid=u_mid, fields=fields, Abar=Abar, nt=nt, shape=shape)
    return R


WB = ["Lap(w div k)", "div(w grad div k)"]


def wbend_columns(cols, k1, k2, x, y, tt, args, Lsp, idx):
    """v1.3 (G11): w-weighted bending columns, weak with first derivatives
    of k only (same kernels and inner derivative as v1.9 build_columns):
      Lap(w div k)       Int Lap(w D) phi = Int w D Lap(phi)
      div(w grad div k)  Int div(w grad D) phi = Int D div(w grad phi)
                                             = Int D (grad w . grad phi + w Lap phi)
    with D = div k, w = |k|^2 - 1. Pointwise versions (np.gradient) and the
    spatially weak versions go into the prediction-score data when present."""
    IY, IX, IT = idx
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    dt = float(np.median(np.diff(tt)))
    rx, ry = radii(args)
    fx, fx1, fx2, _, _ = V.kernel_d3(rx * Lsp, dx, args.poly_p)
    fy, fy1, fy2, _, _ = V.kernel_d3(ry * Lsp, dy, args.poly_p)
    ht, _, _, _, _ = V.kernel_d3(args.t_half * dt, dt, args.poly_p)
    eta = args.eta * Lsp
    Dd = lambda F, ax: V.inner_deriv(F, eta, dx, dy, ax, "poly", args.poly_p)
    w = k1 ** 2 + k2 ** 2 - 1.0
    D = Dd(k1, "x") + Dd(k2, "y")
    wx, wy = Dd(w, "x"), Dd(w, "y")
    K = lambda F, a, b: V.corr3(F, a, b, ht, dx, dy, dt)[IY, IX, IT]
    wD = w * D
    C = cols["rhs"]
    C[WB[0]] = K(wD, fx2, fy) + K(wD, fx, fy2)
    C[WB[1]] = K(D * wx, fx1, fy) + K(D * wy, fx, fy1) + C[WB[0]]
    P = cols.get("score")
    if P is not None:
        g = lambda F, ax: np.gradient(F, dx if ax == 1 else dy, axis=ax)
        Dx, Dy = g(D, 1), g(D, 0)
        pw = {WB[0]: g(g(wD, 1), 1) + g(g(wD, 0), 0), WB[1]: g(w * Dx, 1) + g(w * Dy, 0)}
        H = P["H"]; nt = k1.shape[2]
        ITc = np.minimum(IT[:, None] + np.arange(H + 1)[None, :], nt - 1)
        K2 = lambda F, a, b: V.corr1d(V.corr1d(F, b, 0, dy), a, 1, dx)
        for nm in WB:
            P["pw"][nm] = pw[nm][IY[:, None], IX[:, None], ITc]
            P["wk"][nm] = np.zeros(ITc.shape)
        for f in sorted(set(ITc.ravel().tolist())):
            rows_, cc = np.where(ITc == f)
            G0 = K2(wD[..., f], fx2, fy) + K2(wD[..., f], fx, fy2)
            G1 = K2(D[..., f] * wx[..., f], fx1, fy) + K2(D[..., f] * wy[..., f], fx, fy1) + G0
            P["wk"][WB[0]][rows_, cc] = G0[IY[rows_], IX[rows_]]
            P["wk"][WB[1]][rows_, cc] = G1[IY[rows_], IX[rows_]]


# ---------------------------------------------------------------------
# (4) fits of one window
# ---------------------------------------------------------------------

def gram_geometry(C, names, b):
    """Gram of the named weak columns on the kept rows: eigenvalues, the
    weak direction, and the target's projection on it (C0 vocabulary)"""
    Th, bv, _ = assemble(C, names, b, np.zeros_like(b))
    G = Th.T @ Th / max(len(bv), 1)
    ev, U = np.linalg.eigh(G)
    iw, is_ = int(np.argmin(ev)), int(np.argmax(ev))
    return dict(names=names, eig=[float(v) for v in ev], lam_w=float(ev[iw]), lam_s=float(ev[is_]),
                weak_dir={nm: float(U[j, iw]) for j, nm in enumerate(names)},
                cond=float(np.sqrt(ev[is_] / max(ev[iw], 1e-300))))


def window_fits(R, args, rng, lines, out, label):
    C, T, xs = R["rhs"], R["targets"], R["xs"]
    b = T["theta_t"]
    fits = dict(n_rows=int(len(b)), target_rms=float(np.sqrt(np.mean(b ** 2))))
    lines.append("  target scale: RMS Int[theta_t phi] = %.3e over %d rows" % (fits["target_rms"], len(b)))
    LIB = RHS_LIB + (RHS_EXT if args.lib_ext else [])
    LT = LIB + ["S"]

    def free(names, key, tag):
        Th, bv, xv = assemble(C, names, b, xs)
        r = fit_pareto(names, Th, bv, xv, rng, args.rel_tol, args.min_contrib, abs_tol=args.abs_tol)
        r["collinearity"] = collinearity(names, Th)
        fits[key] = r
        lines.append("  %-34s %s" % (tag + ":", V.fmt(r)))
        return r

    def forced(names, want, key, tag):
        Th, bv, xv = assemble(C, names, b, xs)
        if len(bv) < 20:
            fits[key] = dict(skipped=True, reason="too few rows"); return None
        r = forced_fit(names, Th, bv, xv, want)
        r["collinearity"] = collinearity(want, Th[:, [names.index(n) for n in want]])
        fits[key] = r
        lines.append("  %-34s %s   holdout R2 %.4f (n %d)  [cond %.3g, max |r| %.3f]" % (
            tag + ":", cstr(r["coefs"]), r["r2_holdout"], r["n"], r["collinearity"]["cond"], abs(r["collinearity"]["max_corr"])))
        return r

    free(LIB, "free_ext", "free RHS+ext")
    free(RHS_LIB, "free_rhs", "free RHS (phase columns)")
    forced(RHS_LIB, RCN_RHS, "forced_rcn", "forced RCN {w div k, k.grad w, Lap div k}")
    if args.lib_ext:
        forced(LIB, SP + ["J"], "forced_SJ", "forced {w div k, k.grad w, J}")
        forced(LIB, RCN_RHS + ["J"], "forced_rcn_J", "forced RCN + J")
        forced(LT, ["S", "J"], "tied_SJ", "tied {S, J}")
        forced(LT, ["S", "Lap div k", "J"], "tied_SBJ", "tied {S, Lap div k, J}")
    forced(LT, ["S", "Lap div k"], "tied_SB", "tied {S, Lap div k}")
    # v1.3.1: the three-term law free selection returns before winding on the data (12 Oct local run)
    forced(RHS_LIB, SP + ["w"], "forced_SPw", "forced {w div k, k.grad w, w}")
    forced(LT, ["S", "w"], "tied_Sw", "tied {S, w}")
    # v1.3 (G11): w-weighted bending
    LW = RHS_LIB + WB
    forced(LW, RCN_RHS + [WB[0]], "forced_rcn_wB", "forced RCN + Lap(w div k)")
    forced(LW, RCN_RHS + [WB[1]], "forced_rcn_wB2", "forced RCN + div(w grad div k)")
    forced(LW + ["S"], ["S", "Lap div k", WB[0]], "tied_SBwB", "tied {S, Lap div k, Lap(w div k)}")
    lines.append("  FLAG (w-bending): on the RCN control (no w-bending in the truth) the Lap(w div k) coefficient is -0.045 by the exact route"
                 " and -1.77 through mg at 0.85 (12 Oct); read the values above against the mg control, not against 0")
    # expected RHS: the RCN truth (2, 2, -1) on the data's own weak columns
    Th, bv, xv = assemble(C, RCN_RHS, b, xs)
    exp_ = Th @ np.array([TRUTH[n] for n in RCN_RHS])
    tr = xv < np.median(xv)
    sc = float(np.dot(exp_[~tr], bv[~tr]) / max(np.dot(exp_[~tr], exp_[~tr]), 1e-300))
    fits["expected_rcn"] = dict(r2_holdout=float(r_squared(bv[~tr], exp_[~tr])), scale_to_target=sc,
                                r2_scaled=float(r_squared(bv[~tr], sc * exp_[~tr])))
    lines.append("  %-34s holdout R2 %.4f as is; one scale factor %.3f brings it to R2 %.4f" % (
        "expected RCN (2, 2, -1) on these columns:", fits["expected_rcn"]["r2_holdout"], sc, fits["expected_rcn"]["r2_scaled"]))
    # v1.1: the truth carried through the C0 cross-talk gain. With measured
    # columns S_m = g11 S + g12 B, B_m = g21 S + g22 B, the law 2S - B on the
    # true columns is c_S S_m + c_B B_m with G^T (c_S, c_B) = (2, -1).
    g = [float(v) for v in str(args.c0_gain).split(",")]
    G = np.array([[g[0], g[1]], [g[2], g[3]]])
    cS, cB = np.linalg.solve(G.T, np.array([2.0, -1.0]))
    cg = {"w div k": float(cS), "k.grad w": float(cS), "Lap div k": float(cB)}
    expg = Th @ np.array([cg[n] for n in RCN_RHS])
    scg = float(np.dot(expg[~tr], bv[~tr]) / max(np.dot(expg[~tr], expg[~tr]), 1e-300))
    fits["expected_gain"] = dict(coefs=cg, gain=g, r2_holdout=float(r_squared(bv[~tr], expg[~tr])), scale_to_target=scg,
                                 r2_scaled=float(r_squared(bv[~tr], scg * expg[~tr])))
    lines.append("  %-34s (%.3f, %.3f, %.3f) from the gain [S_m = %.2f S + %.2f B, B_m = %.2f S + %.2f B]: holdout R2 %.4f as is; scale %.3f gives %.4f%s" % (
        "expected RCN through the C0 gain:", cS, cS, cB, g[0], g[1], g[2], g[3], fits["expected_gain"]["r2_holdout"], scg, fits["expected_gain"]["r2_scaled"],
        "" if abs(R["mu"] - 0.85) < 0.005 else "   FLAG: gain measured on the 0.85 control (beta0 >= 1.8), applied at mu %.2f" % R["mu"]))
    # identity
    bw, wid = T["w_t"], T["w_t_identity"]
    fin = np.isfinite(wid) & np.isfinite(bw)
    fits["identity"] = dict(r2=float(r_squared(bw[fin], wid[fin])),
                            rms_ratio=float(np.sqrt(np.mean(wid[fin] ** 2)) / max(np.sqrt(np.mean(bw[fin] ** 2)), 1e-300)))
    lines.append("  %-34s R2 %.4f, rms ratio (identity / measured) %.3f" % ("w_t = 2 k.grad theta_t (measured):", fits["identity"]["r2"], fits["identity"]["rms_ratio"]))
    # Gram geometry
    fits["gram_rcn"] = gram_geometry(C, RCN_RHS, b)
    fits["gram_SB"] = gram_geometry(C, ["S", "Lap div k"], b)
    g = fits["gram_SB"]
    lines.append("  Gram of {S, Lap div k}: lam_w %.3e, lam_s %.3e (cond %.3g); weak direction S %.3f, B %.3f   (sigma_w^2 of the C0 control at this mu gives the dilution; not computed here)" % (
        g["lam_w"], g["lam_s"], g["cond"], g["weak_dir"]["S"], g["weak_dir"]["Lap div k"]))
    # v1.3 (G14): prediction scores of v1.9 on the fitted laws
    if R.get("score") is not None:
        laws = {}
        for key, nm in (("free_ext", "free RHS+ext"), ("forced_rcn", "forced RCN"), ("forced_SJ", "forced {S pair, J}"),
                        ("forced_rcn_J", "forced RCN + J"), ("tied_SJ", "tied {S, J}"), ("tied_SB", "tied {S, B}"),
                        ("forced_rcn_wB", "forced RCN + Lap(w div k)"), ("forced_SPw", "forced {S pair, w}")):
            r = fits.get(key) or {}
            if r.get("coefs") and not r.get("skipped"):
                laws[nm] = dict(r["coefs"])
        laws["expected through the gain"] = dict(fits["expected_gain"]["coefs"])
        try:
            fits["prediction_scores"] = {k: {str(h): v for h, v in d.items()} for k, d in
                                         V.prediction_scores(R["score"], laws, xs, args, lines).items()}
        except Exception as exc:                                         # loud flag, keep going
            lines.append("  prediction scores FAILED: %r" % exc)
    return fits


# ---------------------------------------------------------------------
# (5) fit maps at one frame
# ---------------------------------------------------------------------

def fit_maps(R, fits, args, path, title):
    IY, IX, IT = R["idx"]
    frames = np.unique(IT)
    f = frames[len(frames) // 2]
    sel = IT == f
    if sel.sum() < 10:
        return None
    C, b = R["rhs"], R["targets"]["theta_t"]
    x, y, Lsp, y_row = R["x"], R["y"], R["Lsp"], R["y_row"]
    iy_u, ix_u = np.unique(IY[sel]), np.unique(IX[sel])
    grid = lambda v: _grid(v, IY[sel], IX[sel], iy_u, ix_u)
    def pred(coefs):
        return sum(c * C[nm] for nm, c in coefs.items()) if coefs else np.zeros_like(b)
    fr = fits.get("free_ext") or fits.get("free_rhs")
    p_free = pred(fr.get("coefs", {})) if fr and not fr.get("skipped") else np.zeros_like(b)
    fc = fits.get("forced_rcn", {})
    p_rcn = pred(fc.get("coefs", {})) if fc and not fc.get("skipped") else np.zeros_like(b)
    eg = fits.get("expected_gain", {})
    p_exp = pred(eg.get("coefs", TRUTH))
    sc = eg.get("scale_to_target", 1.0)
    nfree = len(fr.get("coefs", {})) if fr else 0
    panels = [("target: Int[theta_t phi]", b), ("free fit, %d terms (law in the report)" % nfree, p_free),
              ("forced RCN fit (%s)" % cstr(fc.get("coefs", {})), p_rcn),
              ("expected: truth (2, 2, -1) through the C0 gain = (%.2f, %.2f, %.2f), unscaled\nholdout R2 %.2f (raw truth on these columns: R2 %.2f)" % (
                  eg.get("coefs", TRUTH)["w div k"], eg.get("coefs", TRUTH)["k.grad w"], eg.get("coefs", TRUTH)["Lap div k"],
                  eg.get("r2_holdout", np.nan), fits.get("expected_rcn", {}).get("r2_holdout", np.nan)), p_exp),
              ("residual: target - free fit", b - p_free), ("residual: target - forced RCN", b - p_rcn),
              ("residual: target - expected", b - p_exp)]
    lim = float(np.nanpercentile(np.abs(b[sel]), 99)) or 1.0
    lim_e = lim                                                          # v1.1: the gain-carried expectation is on the target's scale
    ncol = 4; nrow = 2
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow))
    axs = axs.ravel()
    ext = [x[ix_u[0]] / Lsp, x[ix_u[-1]] / Lsp, (y[iy_u[0]] - y_row) / Lsp, (y[iy_u[-1]] - y_row) / Lsp]
    ims = {}
    for j, (ax, (nm, v)) in enumerate(zip(axs, panels)):
        own = False
        ims[own] = ax.imshow(grid(v[sel]), origin="lower", extent=ext, cmap="RdBu_r", vmin=-(lim_e if own else lim), vmax=(lim_e if own else lim), aspect="auto", interpolation="nearest")
        xc, yc = R["cores_xy"][f]
        if len(xc):
            ax.plot(xc / Lsp, (yc - y_row) / Lsp, "k.", ms=3)
        rr = float(np.sqrt(np.nanmean(v[sel] ** 2)))
        ax.set_title("%s\nrms %.2e (target rms %.2e)%s" % (nm, rr, float(np.sqrt(np.mean(b[sel] ** 2))), "; own colour scale" if own else ""), fontsize=7)
        ax.set_xlabel("x / L", fontsize=7); ax.set_ylabel("(y - y_row) / L", fontsize=7); ax.tick_params(labelsize=6)
    for ax in axs[len(panels):]:
        ax.axis("off")
    fig.colorbar(ims[False], ax=axs[6], fraction=0.05, pad=0.03, label="weak theta_t (all panels)")
    fig.suptitle("%s\nframe t = %.3f: weak columns at the query centres (same colour scale; dots: amplitude dips)" % (title, R["t"][f]), fontsize=9)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return float(R["t"][f])


def _grid(v, IY, IX, iy_u, ix_u):
    G = np.full((len(iy_u), len(ix_u)), np.nan)
    jy = np.searchsorted(iy_u, IY); jx = np.searchsorted(ix_u, IX)
    G[jy, jx] = v
    return G


def fig_domain(R, args, path, title):
    """the pattern at the window's middle frame with the kept centres"""
    IY, IX, IT = R["idx"]
    f = R["nt"] // 2
    sel = IT == f
    x, y, Lsp, y_row = R["x"], R["y"], R["Lsp"], R["y_row"]
    sl = np.abs(y - y_row) < (args.band_to + 2.0) * Lsp
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.imshow(R["u_mid"][sl], origin="lower", extent=[x[0] / Lsp, x[-1] / Lsp, (y[sl][0] - y_row) / Lsp, (y[sl][-1] - y_row) / Lsp],
              cmap="Greys", aspect="auto")
    ax.plot(x[IX[sel]] / Lsp, (y[IY[sel]] - y_row) / Lsp, ".", color="C2", ms=2, label="kept centres")
    xc, yc = R["cores_xy"][f]
    if len(xc):
        ax.plot(xc / Lsp, (yc - y_row) / Lsp, "o", mfc="none", mec="C3", ms=8, label="amplitude dips")
    rx, ry = radii(args)
    ax.add_patch(plt.Rectangle((x[0] / Lsp + 0.3, -ry), 2 * rx, 2 * ry, fill=False, ec="C0", lw=1.5))
    ax.set_title("%s, t = %.3f: domain '%s', discs %.2g L, support %.3g x %.3g L (box)" % (title, R["t"][f], args.domain, args.core_excl, rx, ry), fontsize=8)
    ax.set_xlabel("x / L"); ax.set_ylabel("(y - y_row) / L"); ax.legend(fontsize=7, loc="upper right")
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------
# (6) sweep table and figure
# ---------------------------------------------------------------------

def table_row(stem, mu, wname, win, R, fits, rec_rows, marks):
    g = lambda k, nm: (fits.get(k) or {}).get("coefs", {}).get(nm, np.nan)
    r2 = lambda k: (fits.get(k) or {}).get("r2_holdout", np.nan)
    fr = fits.get("free_ext") or {}
    sel = [r for r in rec_rows if win[0] - 1e-9 <= r["t"] <= win[1] + 1e-9]
    return dict(file=stem, mu=mu, window=wname, t0=win[0], t1=win[1], n_rows=fits.get("n_rows", 0),
                cores_min=min(R["ncores"]), cores_max=max(R["ncores"]),
                n_wound_end=sel[-1]["n_wound"] if sel else np.nan, trough_min=float(np.nanmin([r["trough"] for r in sel])) if sel else np.nan,
                dip_offrow_median=float(np.nanmedian([r["dip_offrow_median"] for r in sel])) if sel else np.nan,
                t_first=marks.get("t_first", np.nan), t_wound=marks.get("t_wound", np.nan),
                rcn_wdivk=g("forced_rcn", "w div k"), rcn_kgw=g("forced_rcn", "k.grad w"), rcn_B=g("forced_rcn", "Lap div k"), rcn_r2=r2("forced_rcn"),
                SJ_wdivk=g("forced_SJ", "w div k"), SJ_kgw=g("forced_SJ", "k.grad w"), SJ_J=g("forced_SJ", "J"), SJ_r2=r2("forced_SJ"),
                rcnJ_B=g("forced_rcn_J", "Lap div k"), rcnJ_J=g("forced_rcn_J", "J"), rcnJ_r2=r2("forced_rcn_J"),
                tied_S=g("tied_SB", "S"), tied_B=g("tied_SB", "Lap div k"), tied_SB_r2=r2("tied_SB"),
                tiedSJ_S=g("tied_SJ", "S"), tiedSJ_J=g("tied_SJ", "J"), tied_SJ_r2=r2("tied_SJ"),
                SPw_wdivk=g("forced_SPw", "w div k"), SPw_kgw=g("forced_SPw", "k.grad w"), SPw_w=g("forced_SPw", "w"), SPw_r2=r2("forced_SPw"),
                score_SPw=_score(fits, "forced {S pair, w}"),
                wB_B=g("forced_rcn_wB", "Lap div k"), wB_wB=g("forced_rcn_wB", WB[0]), wB_r2=r2("forced_rcn_wB"),
                wB2_B=g("forced_rcn_wB2", "Lap div k"), wB2_wB=g("forced_rcn_wB2", WB[1]), wB2_r2=r2("forced_rcn_wB2"),
                tiedwB_S=g("tied_SBwB", "S"), tiedwB_B=g("tied_SBwB", "Lap div k"), tiedwB_wB=g("tied_SBwB", WB[0]), tiedwB_r2=r2("tied_SBwB"),
                score_rcn=_score(fits, "forced RCN"), score_SJ=_score(fits, "tied {S, J}"), score_free=_score(fits, "free RHS+ext"),
                score_wB=_score(fits, "forced RCN + Lap(w div k)"), score_expected=_score(fits, "expected through the gain"),
                free_support=cstr(fr.get("coefs", {})) if fr.get("coefs") else "", free_r2=fr.get("r2_holdout", np.nan),
                free_rhs_support=cstr((fits.get("free_rhs") or {}).get("coefs", {})), free_rhs_r2=(fits.get("free_rhs") or {}).get("r2_holdout", np.nan),
                expected_r2=(fits.get("expected_rcn") or {}).get("r2_holdout", np.nan), expected_scale=(fits.get("expected_rcn") or {}).get("scale_to_target", np.nan),
                expected_gain_r2=(fits.get("expected_gain") or {}).get("r2_holdout", np.nan), expected_gain_scale=(fits.get("expected_gain") or {}).get("scale_to_target", np.nan),
                identity_r2=(fits.get("identity") or {}).get("r2", np.nan), identity_ratio=(fits.get("identity") or {}).get("rms_ratio", np.nan),
                gram_lam_w=(fits.get("gram_SB") or {}).get("lam_w", np.nan), gram_cond=(fits.get("gram_SB") or {}).get("cond", np.nan),
                target_rms=fits.get("target_rms", np.nan))


def _score(fits, name):
    """the weak multi-step error ratio at the longest horizon (v1.3)"""
    d = (fits.get("prediction_scores") or {}).get(name) or {}
    if not d:
        return np.nan
    h = max(d, key=lambda k: int(k))
    return float(d[h].get("multi_wk", np.nan))


def fig_sweep(rows, path):
    classes = [c for c in ("pre", "to_wind", "straddle", "post", "whole") if any(r["window"] == c for r in rows)]
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5))
    mk = {"pre": "o", "to_wind": "v", "straddle": "s", "post": "^", "whole": "x"}
    for ax, (key, lab, truth) in zip(axs.ravel(), (("rcn_wdivk", "forced RCN: w div k", 2.0), ("rcn_kgw", "forced RCN: k.grad w", 2.0),
                                                     ("rcn_B", "forced RCN: Lap div k", -1.0), ("SJ_J", "forced {S pair, J}: J", None),
                                                     ("rcn_r2", "holdout R2: forced RCN (filled), {S pair, J} (open)", None), ("identity_ratio", "identity rms ratio (1 = consistent)", 1.0))):
        for c in classes:
            rr = [r for r in rows if r["window"] == c]
            mu = [r["mu"] for r in rr]; v = [r[key] for r in rr]
            ax.plot(mu, v, mk[c], ls="-", label=c, ms=6)
            if key == "rcn_r2":
                ax.plot(mu, [r["SJ_r2"] for r in rr], mk[c], mfc="none", ls=":", color=ax.lines[-1].get_color())
        if truth is not None:
            ax.axhline(truth, color="0.6", lw=0.8, ls="--")
        ax.set_title(lab, fontsize=9); ax.set_xlabel("mu"); ax.legend(fontsize=7)
    fig.suptitle("theta_t rate fits by window class across the sweep (dashed: RCN truth at alpha = eta_b = tau)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_global(rows, path):
    """v1.3 (G12): tied {S, J} from the start to the first winding (to_wind)
    where the run winds, over the whole run where it does not; the forced
    RCN triple on the same windows; the w-bending coefficient; scores"""
    pick = {}
    for r in rows:
        if r["window"] == "to_wind" or (r["window"] == "whole" and r["file"] not in pick):
            if r["window"] == "to_wind" or not any(q["file"] == r["file"] and q["window"] == "to_wind" for q in rows):
                pick[r["file"]] = r
    rr = sorted(pick.values(), key=lambda r: r["mu"])
    if not rr:
        return
    mu = np.array([r["mu"] for r in rr]); tw = np.array([r["window"] == "to_wind" for r in rr])
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, keys, lab in ((axs[0, 0], [("tiedSJ_S", "a (S)"), ("tiedSJ_J", "d (J)")], "global law theta_t = a S + d J (tied {S, J})"),
                          (axs[0, 1], [("tied_SJ_r2", "tied {S, J}"), ("rcn_r2", "forced RCN"), ("wB_r2", "RCN + Lap(w div k)")], "holdout R2 on the same windows"),
                          (axs[0, 2], [("score_SJ", "tied {S, J}"), ("score_rcn", "forced RCN"), ("score_wB", "RCN + Lap(w div k)"), ("score_expected", "RCN truth through the gain")],
                           "prediction error ratio, weak multi-step, longest horizon (null 1)"),
                          (axs[1, 0], [("rcn_B", "RCN bending"), ("wB_B", "bending with Lap(w div k) admitted"), ("tiedwB_B", "tied {S, B, Lap(w div k)}: B")], "bending coefficient"),
                          (axs[1, 1], [("wB_wB", "RCN + Lap(w div k)"), ("wB2_wB", "RCN + div(w grad div k)"), ("tiedwB_wB", "tied {S, B, Lap(w div k)}")],
                           "w-bending coefficient (RCN control through mg: -1.77, exact -0.045)"),
                          (axs[1, 2], [("identity_ratio", "identity rms ratio")], "w_t = 2 k.grad theta_t (1 = consistent)")):
        for j, (k, nm) in enumerate(keys):
            v = np.array([r.get(k, np.nan) for r in rr], float)
            c = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")[j]
            ax.plot(mu, v, "-", color=c, lw=1.2, label=nm)
            ax.plot(mu[tw], v[tw], "v", color=c, ms=5)
            ax.plot(mu[~tw], v[~tw], "o", mfc="none", color=c, ms=5)
        ax.set_title(lab, fontsize=9); ax.set_xlabel("mu"); ax.legend(fontsize=7)
    fig.suptitle("Global rate law from the start to the first winding (triangles: to_wind) or over the whole run (open circles: no winding)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def write_md(rows, path):
    keys = ["file", "mu", "window", "t0", "t1", "n_rows", "cores_min", "cores_max", "rcn_wdivk", "rcn_kgw", "rcn_B", "rcn_r2",
            "SJ_wdivk", "SJ_kgw", "SJ_J", "SJ_r2", "rcnJ_B", "rcnJ_J", "rcnJ_r2", "tiedSJ_S", "tiedSJ_J", "tied_SJ_r2",
            "wB_B", "wB_wB", "wB_r2", "wB2_wB", "wB2_r2", "score_rcn", "score_SJ", "score_wB", "score_free", "score_expected",
            "identity_ratio", "expected_r2", "expected_gain_r2", "expected_gain_scale", "free_support", "free_r2"]
    fmt = lambda v: ("%.3f" % v if isinstance(v, float) and np.isfinite(v) else str(v))
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r[k]) for k in keys) + " |")
    Path(path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------

def write_control_movie(args, out):
    """the RCN knee relaxation of wsindy_theta_t v1.9 (mu, alpha = eta_b =
    tau = 1/3, R 1/2, beta0 1.8, Ly 60, 300 rows, t 0..2 in 41 frames,
    4 periods) written as a data-layout movie; truth (2, 2, -1); no cores"""
    mu = args.mu; k1 = float(np.sqrt(1 - mu ** 2))
    y, tt, p, Phi, pt = V.rcn_knee_relaxation(mu, 1 / 3, 1 / 3, 1 / 3, 0.5, 1.8, 60.0, 300, 2.0, 41)
    dx = y[1] - y[0]
    Lx = 4 * 2 * np.pi / k1; nx = int(round(Lx / dx)); Lx = nx * dx
    x = (np.arange(nx) - nx // 2) * dx
    th = k1 * x[None, :, None] + Phi[:, None, :]
    u = np.cos(th).astype(np.float32)
    nt = len(tt)
    Phit = np.vstack([np.zeros((1, nt)), np.cumsum(0.5 * (pt[1:] + pt[:-1]) * dx, axis=0)])
    k1_ex = np.full(th.shape, k1, np.float32)
    k2_ex = np.broadcast_to(p[:, None, :], th.shape).astype(np.float32)
    th_t_ex = np.broadcast_to(Phit[:, None, :], th.shape).astype(np.float32)
    md = dict(mu=mu, gb_row_in_crop=int(len(y) // 2), control=True, truth=[2.0, 2.0, -1.0],
              note="RCN knee relaxation (wsindy_theta_t v1.9 rcn_knee_relaxation), u = cos theta, alpha = eta_b = tau = 1/3, beta0 1.8")
    path = out / ("control_rcn_mu%.2f.npz" % mu)
    np.savez_compressed(path, u=u, x=x, y=y, tt=tt, metadata_json=json.dumps(md),
                        k1_exact=k1_ex, k2_exact=k2_ex, theta_t_exact=th_t_ex)
    return str(path)


def main(args):
    rng = np.random.default_rng(args.seed)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    files = [f.strip() for f in args.npz.split(",") if f.strip()]
    if int(args.control):
        files = [write_control_movie(args, out)] + files
    lines = ["theta_t_mu_sweep v1.3.1 on wsindy_theta_t v1.9 columns; half widths %.3g (x) %.3g (y) L, t_half %d frames, poly p %d, rel_tol %.2g, abs_tol %.2g, min_contrib %.2g; domain %s, discs %.2g L about cores defined by %s, lib_ext %d" % (
        radii(args)[0], radii(args)[1], args.t_half, args.poly_p, args.rel_tol, args.abs_tol, args.min_contrib, args.domain, args.core_excl, args.core_def, args.lib_ext), ""]
    table = []
    t_start = time.time()
    for f in files:
        stem = Path(f).stem
        fo = out / stem; fo.mkdir(parents=True, exist_ok=True)
        rec, info = movie_record(f, args, fo, lines)
        wins, marks_i = windows_from_record(rec, args, lines)
        t_of = lambda i: float(info["t"][i]) if i is not None else np.nan
        marks = dict(t_first=t_of(marks_i.get("first")), t_wound=t_of(marks_i.get("wound")))
        fig_record(rec, info, wins, fo / ("fig_record_%s.png" % stem), args)
        if args.record_only:
            lines.append(""); continue
        for wname, win in wins.items():
            if args.only_windows and wname not in args.only_windows.split(","):
                continue
            lines.append("[%s | %s | t %.3f-%.3f]" % (stem, wname, win[0], win[1]))
            wo = fo / wname; wo.mkdir(exist_ok=True)
            R = window_rows(f, win[0], win[1], args, wo, lines)
            if R is None:
                continue
            fits = window_fits(R, args, rng, lines, wo, "%s %s" % (stem, wname))
            fits["window"] = dict(name=wname, t0=win[0], t1=win[1]); fits["marks"] = marks
            try:
                tf = fit_maps(R, fits, args, wo / ("fit_maps_%s.png" % wname), "%s, mu %.3f, window %s (t %.2f-%.2f)" % (stem, info["mu"], wname, win[0], win[1]))
                lines.append("  fit maps at t = %s -> %s" % (("%.3f" % tf) if tf else "n/a", wo / ("fit_maps_%s.png" % wname)))
                fig_domain(R, args, wo / ("domain_%s.png" % wname), "%s, %s" % (stem, wname))
            except Exception as exc:                                     # pragma: no cover
                lines.append("  fit maps skipped: %s" % exc)
            with open(wo / "fits.json", "w") as fh:
                json.dump(fits, fh, indent=1, default=str)
            table.append(table_row(stem, info["mu"], wname, win, R, fits, rec, marks))
            del R
        lines.append("")
        (out / "report.txt").write_text("\n".join(lines) + "\n")
    if table:
        with open(out / "sweep_table.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(table[0].keys())); wr.writeheader(); wr.writerows(table)
        write_md(table, out / "sweep_table.md")
        fig_sweep(table, out / "fig_sweep_coefficients.png")
        fig_global(table, out / "fig_global_law.png")
    lines.append("elapsed %.1f s" % (time.time() - t_start))
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    with open(out / "config.json", "w") as fh:
        json.dump({k: getattr(args, k) for k in vars(args) if not k.startswith("_")}, fh, indent=2, default=str)
    print("\n".join(lines)); print("outputs ->", out)


class _Args:
    npz = ""                      # one movie or a comma-separated list
    outdir = "results/theta_t_mu_sweep"
    mu = 0.85                     # fallback when the file carries no mu
    control = 0                   # 1: prepend the RCN knee control movie (written to outdir; truth 2, 2, -1)
    exact = 0                     # 1: use the file's exact k1, k2, theta_t when present (control validation)
    # record and windows
    windows = ""                  # explicit windows "name:t0-t1,..." (overrides the record)
    only_windows = ""             # fit only these window names (comma-separated)
    record_only = 0               # 1: record and windows, no fits
    max_frames = 60               # a window with more frames is decimated in time to at most this many (memory)
    wind_lo = 0.25                # turns: first circulation (start of the straddle)
    wind_hi = 0.9                 # turns: a dip counts as wound
    loop_r = 0.5                  # loop radius about each dip, L
    merge_dist = 0.5              # dips merged within this distance, L
    c0_gain = "0.86,0.31,-0.25,0.94"   # v1.1: S_m = g11 S + g12 B, B_m = g21 S + g22 B (C0, 0.85 control, beta0 >= 1.8)
    sigma_det_charge = 0.3        # v1.2: director index width for the charge (mg_zigzag_theta_t production value)
    core_def = "circulation"      # circulation (discs about wound dips) | charge (about director-charge clusters, v1.2) | dips (every amplitude dip, v1.0)
    # estimator
    sigma_dir = 0.4
    taper_frac = 0.06
    core_frac = 0.70
    # weak form (wsindy_theta_t v1.9 names)
    radius = 1.5
    radius_x = -1.0
    radius_y = -1.0
    no_cross = 0
    t_half = 3
    eta = 0.0
    poly_p = 6
    stride_xy = 4
    stride_t = 1
    rim = 4
    rel_tol = 0.01
    abs_tol = 0.0
    min_contrib = 0.05
    seed = 0
    lib_ext = 1
    chi_m = 1.0
    score_h = 6                   # v1.3: prediction horizon in frames (v1.9 default; 0 disables the scores)
    # domains
    domain = "band"
    side = "both"
    core_excl = 0.5
    row_to = 0.5
    band_to = 2.0
    far_from = 4.0
    excl_dilate = 0
    dilate_cores_only = 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in vars(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    main(p.parse_args())
