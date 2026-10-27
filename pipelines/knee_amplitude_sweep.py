"""knee_amplitude_sweep.py  v1.0  (12 Oct 2026)   workstreams G9 and F1 of the research thread

The amplitude along and across the knee core line, frame by frame, beside
the four formation markers of theta_t_mu_sweep (director charge,
amplitude dips, partial circulation, winding), so every angle is read on
its own clock.

Per frame (one mg estimator pass, the frozen layer through
mg_zigzag_theta_t.estimate; one row projection):
  (1) row demodulation (knee_row_demod_survey v1.0, imported): each row
      projected onto exp(i k1 x) with k1 fitted per file; A(y), k2(y);
      the trough depth Amin/Afar and its distance from the line, r_A on
      the line, beta_row, the k2 overshoot, and the x-invariance of the
      line row (share of its variance in the k1 mode; it falls when the
      pattern stops being a uniform knee, and then (1) is an x-average);
  (2) the off-line split without a threshold, from an x-local row
      demodulation (each row times exp(-i k1 x), Gaussian window of
      half-width loc_L L along x, so the image at 2 k1 is suppressed by
      exp(-(2 pi loc_L)^2 / 2) and the y profile keeps the grid
      resolution): in every column of the inner window the y of the
      minimum of |c| within band_min L of the line (parabolic sub-grid
      refinement; a column whose contrast in the band is below
      min_contrast of the far field has no trough and no position);
      median and spread of its distance from the line, the
      median trough depth and |c| on the line over the far field. The mg
      amplitude on the line and its column minimum are kept beside it as
      the estimator's view (mg_*), but the mg estimator cannot resolve
      the split (see the changelog);
      Reading (measured on the 28 last frames, 12 Oct): the trough off
      the line is present on every knee, defect-free ones included, at
      0.94 to 1.0 of y_node = pi/(2 mu) (k1/(2 mu) in units of L), the
      first node of the beat between the two crossing roll sets
      (k1, +mu) and (k1, -mu) that overlap at the knee. With the two
      envelopes a+(y) and a-(y) (a- the mirror image of a+), the k1
      projection is a+ e^{i mu y} + a- e^{-i mu y}: on the line it reads
      a+ + a- (above A_far, 1.08 to 1.17), at the node |a+ - a-|. So the
      position of the split is geometry, set by the angle; its depth
      measures how far each roll set penetrates the other (a trough of
      0 would be complete overlap at the node). split_over_node is the
      measured position over y_node; mirror_asym is the rms of
      u(y_line + s) - u(y_line - s) over the band, over rms u (the runs
      are mirror symmetric about the line to round-off);
  (3) along-line spectra at the dislocation harmonic 2 k1 (spacing
      L = pi/k1), Hann-weighted, power over mean square: of the mg A on
      the line, of the Hilbert envelope of the line row u(x, y_line)
      (estimator-free), and of J = det(grad k) on the line; rms J on the
      line. Per file the same numbers on a smooth knee at the file's mu
      and grid (k2 = mu tanh(mu y), no defect) are reported as the floor,
      in their own columns of events.csv and in report.txt;
      off_J_line and off_A_line are the shares of the along-line variance
      of J and of the mg A that lie between the harmonics 2 n k1 (outside
      half a k1 of each): the estimator's own ripple on a knee sits on
      the harmonics, so this share has a floor near 1e-5 on a smooth
      knee and rises when defects appear (stand-in movie: 1e-5 to 1e-2
      at the first director charge); offrms_J_line is the same as an rms
      of J L^2 (share times rms, square-rooted). The share depends on the
      grid and on how small J is on the knee: its floor is 8e-4 on the
      256 x 384 sweep, 2.5e-5 on the 512-wide six, 0.09 to 0.38 on the
      shallow knees, so it is read against the floor of its own file;
  (4) amplitude curvature: Lap A / A on the line and at the trough
      points, and max |Lap A| / A_far on the band;
  (5) the markers: amplitude dips (70 percent, merged within 0.5 L),
      wound dips (0.5 L loop >= 0.9 turn), largest circulation, director
      charge sum |q| on the band (t1c.detect_defects, sigma_det 0.3).

Outputs under --outdir: per file amp_<stem>.csv (one row per frame),
fig_amp_<stem>.png (time series with the marker times), kymo_<stem>.png
and kymo_<stem>.npz (A and J on the line, x against t), fig_spec_<stem>.png
(along-line power spectra of A and J at the first frame, at each marker
time and at the end, against q/k1, floor in grey); for the set,
events.csv (each quantity at first charge, first dip, first circulation,
first winding and the end), fig_events_vs_mu.png, fig_collapse.png (the
trough depth and the off-line distance against t - t_charge, every
angle), report.txt, config.json.

--mode synthetic plants a movie with a known trough (depth, width, an
off-line split at a planted time and offset, and a growing 2 k1
modulation along the line) and reports recovered against planted.

Changelog
  v1.0  first version. Validation in the sandbox: planted movie (section
        printed in report.txt), the stand-in SH movie, the sliced last
        frames of the fine sweep and the six of record (end-of-run values
        against the formation records of Ed's local run).
        Design choices measured on the planted movie before release:
        the first draft took the split from the column minimum of the mg
        amplitude; on a planted trough split to 0.35 L (troughs 0.7 L
        apart) the mg amplitude at sigma_dir 0.4 has a single minimum on
        the line (0.78 on the line, 0.82 at 0.34 L, against planted 0.89
        and 0.75), so it returned split 0.00 at every frame. The split now
        comes from the x-local row demodulation. The first draft's 2 k1
        power of the mg A on the line read 0.021 to 0.024 at every frame
        whatever the planted modulation (0.02 to 0.27 of the trough),
        so the estimator has a 2 k1 ripple on a knee of its own; hence
        the Hilbert-envelope column and the smooth-knee floor.
"""
import argparse, csv, json, re, sys, time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT / "src", ROOT / "src" / "op_extract", ROOT / "src" / "utils", HERE):
    sys.path.insert(0, str(_p))
from mg_zigzag_theta_t import load_npz, estimate, find_cores           # noqa: E402
from op_extract import t1_common as t1c                                # noqa: E402
from knee_row_demod_survey import fit_k1, row_demod, profile_numbers   # noqa: E402
from theta_t_mu_sweep import loop_circulation, director_charge_points  # noqa: E402
import matplotlib                                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MUTED = "#77756f"


# ---------------------------------------------------------------------
# per-frame quantities
# ---------------------------------------------------------------------

def hann_power(sig, dx, q):
    """Hann-weighted power of sig at wavenumber q, over the mean square"""
    w = np.hanning(len(sig))
    xx = np.arange(len(sig)) * dx
    s = sig - np.sum(w * sig) / w.sum()
    c = 2.0 * np.sum(w * s * np.exp(-1j * q * xx)) / w.sum()
    ms = np.sum(w * sig ** 2) / w.sum()
    return float(np.abs(c) ** 2 / 2.0 / max(ms, 1e-300))


def local_row_demod(u_rows, x, k1, sig):
    """|c(y, x0)|, c = 2 G*(u exp(-i k1 x)) / G*1 along x, G Gaussian of
    standard deviation sig (same normalisation as row_demod)"""
    dx = float(x[1] - x[0])
    E = np.exp(-1j * k1 * x)[None, :]
    z = u_rows * E
    s = sig / dx
    re = gaussian_filter1d(z.real, s, axis=1, mode="constant")
    im = gaussian_filter1d(z.imag, s, axis=1, mode="constant")
    one = gaussian_filter1d(np.ones(len(x)), s, mode="constant")
    return 2.0 * np.abs(re + 1j * im) / one[None, :]


def sub_min(prof, j):
    """parabolic sub-grid offset of a minimum at index j"""
    if 0 < j < len(prof) - 1:
        a, b, c = prof[j - 1], prof[j], prof[j + 1]
        den = a - 2 * b + c
        if den > 0:
            return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return 0.0


def smooth_knee(x, y, y_row, mu, k1):
    X, Y = np.meshgrid(x, y)
    z = mu * (Y - y_row)
    return np.cos(k1 * X + np.logaddexp(z, -z))


def offharmonic_fraction(sig, dx, k1, half=0.5):
    """share of the Hann-weighted variance of sig along x that lies away
    from the harmonics 2 n k1 (outside |q - 2 n k1| < half k1, n >= 1,
    and above q = half k1); a knee has almost none, a moving or isolated
    defect spreads power between the harmonics"""
    w = np.hanning(len(sig))
    s = (sig - np.sum(w * sig) / w.sum()) * w
    P = np.abs(np.fft.rfft(s)) ** 2
    q = 2 * np.pi * np.fft.rfftfreq(len(sig), dx) / k1
    near = np.min(np.abs(q[:, None] - 2.0 * np.arange(1, int(q.max() / 2) + 2)[None, :]), axis=1) < half
    off = (~near) & (q > half)
    return float(P[off].sum() / max(P[1:].sum(), 1e-300))


def frame_quantities(u2, x, y, jr, mu, k1f, args, state):
    ny, nx = u2.shape
    dx = float(x[1] - x[0])
    Lsp = np.pi / np.sqrt(1 - mu ** 2)
    taper = state["taper"]
    X, Y = state["X"], state["Y"]
    win = state["win"]
    y_row = float(y[jr])
    out = {}
    # (1) row demodulation
    c, xinv = row_demod(u2, x, k1f)
    P = profile_numbers(y, jr, c, k1f, mu, (args.far_from * Lsp, (args.far_from + 2.0) * Lsp))
    out.update(xinv_row=float(xinv[jr]), dm_A_row=P["A_row"] / P["A_far"], dm_trough=P["Amin"] / P["A_far"],
               dm_trough_y=P["y_Amin"] / Lsp, r_A=P["r_A"], beta_row=P["beta_row"], k2_overshoot=P["k2max"] / mu,
               dm_A_far=P["A_far"])
    # mg estimator
    e = estimate(u2, nx * dx, ny * dx, args.sigma_dir, None, taper)
    A = e["A"]
    s = np.sign(np.mean(e["k1"])) or 1.0
    k1m, k2m = e["k1"] * s, e["k2"] * s
    far = (np.abs(Y - y_row) > args.far_from * Lsp) & win
    Afar = float(np.median(A[far])) if far.any() else float(np.median(A[win]))
    # (2) the off-line split, per column, from the x-local row demodulation
    ix = np.where(np.abs(x - x.mean()) < 0.4 * nx * dx)[0]
    node = np.sqrt(1 - mu ** 2) / (2 * mu)          # pi / (2 mu) in units of L: beat node of the two crossing roll sets
    bmin = args.band_min if args.band_min > 0 else max(1.0, 1.5 * node)
    jb = np.where(np.abs(y - y_row) < bmin * Lsp)[0]
    Cl = local_row_demod(u2[jb, :], x, k1f, args.loc_L * Lsp)[:, ix]
    jloc = np.argmin(Cl, axis=0)
    dyy = float(y[1] - y[0])
    ymin = np.array([(y[jb[j]] + sub_min(Cl[:, i], j) * dyy - y_row) / Lsp for i, j in enumerate(jloc)])
    AfarD = P["A_far"]
    # a column without a trough has no split: contrast in the band below min_contrast of A_far -> no position
    contrast = (Cl.max(axis=0) - Cl.min(axis=0)) / AfarD
    ymin[contrast < args.min_contrast] = np.nan
    ok = np.isfinite(ymin)
    tro = Cl.min(axis=0) / AfarD
    jrl = int(np.argmin(np.abs(y[jb] - y_row)))
    jmin = jb[jloc]
    # the estimator's view: mg A on the line and its column minimum in the same band
    subm = A[np.ix_(jb, ix)]
    # mirror asymmetry about the line (the runs are mirror symmetric to round-off; a growing value is symmetry breaking)
    nb = min(int(bmin * Lsp / dyy), jr, ny - jr - 1)
    up_, dn_ = u2[jr + 1:jr + nb + 1][:, ix], u2[jr - 1:jr - nb - 1:-1][:, ix]
    out["mirror_asym"] = float(np.sqrt(np.mean((up_ - dn_) ** 2)) / max(np.sqrt(np.mean(u2[:, ix] ** 2)), 1e-300))
    out["split_over_node"] = float(np.median(np.abs(ymin[ok]))) / node if ok.any() else float("nan")
    out.update(ld_A_line=float(np.median(Cl[jrl]) / AfarD), ld_trough=float(np.median(tro)),
               ld_trough_min=float(tro.min()), split_med=float(np.median(np.abs(ymin[ok]))) if ok.any() else float("nan"),
               split_spread=float(np.std(np.abs(ymin[ok]))) if ok.any() else float("nan"),
               split_frac=float(np.mean(np.abs(ymin[ok]) > 0.1)) if ok.any() else float("nan"), trough_cols=float(np.mean(ok)),
               mg_A_line=float(np.median(A[jr, ix]) / Afar), mg_trough=float(np.median(subm.min(axis=0)) / Afar),
               mg_A_far=Afar)
    # (4) amplitude curvature
    lapA = (np.gradient(np.gradient(A, dx, axis=1), dx, axis=1) + np.gradient(np.gradient(A, dx, axis=0), dx, axis=0))
    band = (np.abs(Y - y_row) < bmin * Lsp) & win
    out.update(lapA_line=float(np.median(lapA[jr, ix] / A[jr, ix])) * Lsp ** 2,
               lapA_trough=float(np.median(lapA[jmin, ix] / A[jmin, ix])) * Lsp ** 2,
               lapA_max=float(np.max(np.abs(lapA[band])) / Afar) * Lsp ** 2)
    # (3) along-line spectra and J
    J = (np.gradient(k1m, dx, axis=1) * np.gradient(k2m, dx, axis=0) - np.gradient(k1m, dx, axis=0) * np.gradient(k2m, dx, axis=1))
    q = 2.0 * k1f
    jt = int(np.round(np.median(jmin)))
    env = np.abs(hilbert(u2[jr, :] - np.mean(u2[jr, :])))[ix]
    out.update(spec_A_line=hann_power(A[jr, ix], dx, q), spec_A_trough=hann_power(A[jt, ix], dx, q),
               spec_env_line=hann_power(env, dx, q),
               spec_J_line=hann_power(J[jr, ix], dx, q), off_J_line=offharmonic_fraction(J[jr, ix], dx, k1f),
               off_A_line=offharmonic_fraction(A[jr, ix], dx, k1f), rms_J_line=float(np.sqrt(np.mean(J[jr, ix] ** 2))) * Lsp ** 2)
    # (5) markers
    out["offrms_J_line"] = out["rms_J_line"] * float(np.sqrt(out["off_J_line"]))
    xc, yc = find_cores(A, x, y, win, args.core_frac, merge_dist=0.5 * Lsp)
    circ = np.array([abs(loop_circulation(k1m, k2m, x, y, a, b, 0.5 * Lsp)) for a, b in zip(xc, yc)])
    try:
        _, _, _, qsum = director_charge_points(u2, x, y, y_row, taper, Lsp, args.sigma_det_charge, 2.0, 0.5, win=win)
    except Exception:
        qsum = -1
    out.update(n_dips=int(len(xc)), n_wound=int(np.sum(circ >= 0.9)) if len(circ) else 0,
               circ_max=float(circ.max()) if len(circ) else 0.0, charge_sum_abs=int(qsum))
    kym = dict(A=(A[jr, ix] / Afar).astype(np.float32), J=(J[jr, ix] * Lsp ** 2).astype(np.float32),
               L=(Cl[jrl] / AfarD).astype(np.float32))
    return out, kym


# ---------------------------------------------------------------------
# a file
# ---------------------------------------------------------------------

def run_file(path, args, outdir, lines, u_override=None):
    if u_override is None:
        D = load_npz(path)
        u, x, y, t, md = D["u"], D["x"], D["y"], D["t"], D["md"]
        stem = Path(path).stem
    else:
        u, x, y, t, md, stem = u_override
    mu = float(md.get("mu", args.mu))
    ny, nx, nt = u.shape
    frames = list(range(0, nt, max(1, args.frame_stride)))
    if frames[-1] != nt - 1:
        frames.append(nt - 1)
    jr = int(md.get("gb_row_in_crop", ny // 2))
    dx = float(x[1] - x[0])
    X, Y = np.meshgrid(x, y)
    state = dict(taper=t1c.edge_taper(x, y, frac=args.taper_frac), X=X, Y=Y,
                 win=(np.abs(X - x.mean()) < 0.4 * nx * dx) & (np.abs(Y - y.mean()) < 0.4 * ny * dx))
    k1f = fit_k1(u[..., frames[0]].astype(float), x, np.sqrt(1 - mu ** 2))
    rows, kA, kJ, kL = [], [], [], []
    t0 = time.time()
    floor, kf = frame_quantities(smooth_knee(x, y, float(y[jr]), mu, k1f), x, y, jr, mu, k1f, args, state)
    for i in frames:
        q, k = frame_quantities(u[..., i].astype(float), x, y, jr, mu, k1f, args, state)
        q.update(frame=i, t=float(t[i]), mu=mu)
        rows.append(q); kA.append(k["A"]); kJ.append(k["J"]); kL.append(k["L"])
    keys = ["frame", "t", "mu"] + [kk for kk in rows[0] if kk not in ("frame", "t", "mu")]
    with open(outdir / ("amp_%s.csv" % stem), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
    xi = x[np.abs(x - x.mean()) < 0.4 * nx * dx]
    tt = np.array([r["t"] for r in rows])
    np.savez_compressed(outdir / ("kymo_%s.npz" % stem), A=np.array(kA), J=np.array(kJ), Aloc=np.array(kL), x=xi, t=tt, mu=mu,
                        floor_A=kf["A"], floor_J=kf["J"])
    ev = events(rows)
    lines.append("%s: mu %.3f, k1 fit %.4f (sqrt(1-mu^2) %.4f), %d frames (stride %d) in %.1f s; events: %s" % (
        stem, mu, k1f, np.sqrt(1 - mu ** 2), len(frames), args.frame_stride, time.time() - t0,
        ", ".join("%s %s" % (kk, ("%.2f" % v) if np.isfinite(v) else "none") for kk, v in ev.items())))
    last = rows[-1]
    lines.append("   end: demod trough %.3f at %.2f L; local demod trough %.3f, on line %.3f, split %.2f L (spread %.2f, %.0f%% of columns off the line);"
                 " split / y_node %.2f, mirror asymmetry %.1e; mg trough %.3f, mg A on line %.3f; r_A %.2f, x-inv %.3f; LapA/A line %.2f trough %.2f (per L^2);"
                 " 2k1 power mg A line %.4f envelope %.4f J line %.4f, off-harmonic share J %.1e A %.1e, rms J line %.3f; dips %d wound %d charge %d" % (
                     last["dm_trough"], last["dm_trough_y"], last["ld_trough"], last["ld_A_line"], last["split_med"], last["split_spread"],
                     100 * last["split_frac"], last["split_over_node"], last["mirror_asym"], last["mg_trough"], last["mg_A_line"], last["r_A"], last["xinv_row"], last["lapA_line"],
                     last["lapA_trough"], last["spec_A_line"], last["spec_env_line"], last["spec_J_line"], last["off_J_line"], last["off_A_line"], last["rms_J_line"],
                     last["n_dips"], last["n_wound"], last["charge_sum_abs"]))
    lines.append("   floor (smooth knee, same grid): mg trough %.3f, mg A on line %.3f, ld trough %.3f, split %.2f L, LapA/A line %.2f,"
                 " 2k1 power mg A line %.4f envelope %.4f J line %.4f, off-harmonic share J %.1e A %.1e, rms J line %.4f, dips %d charge %d" % (
                     floor["mg_trough"], floor["mg_A_line"], floor["ld_trough"], floor["split_med"], floor["lapA_line"],
                     floor["spec_A_line"], floor["spec_env_line"], floor["spec_J_line"], floor["off_J_line"], floor["off_A_line"],
                     floor["rms_J_line"], floor["n_dips"], floor["charge_sum_abs"]))
    fig_file(rows, ev, stem, mu, outdir / ("fig_amp_%s.png" % stem), floor)
    Lsp = np.pi / np.sqrt(1 - mu ** 2)
    fig_kymo(np.array(kA), np.array(kJ), np.array(kL), xi, tt, ev, stem, mu, Lsp, outdir / ("kymo_%s.png" % stem))
    fig_spec(np.array(kA), np.array(kJ), kf, xi, tt, ev, stem, mu, k1f, outdir / ("fig_spec_%s.png" % stem))
    return dict(stem=stem, mu=mu, rows=rows, events=ev, floor=floor)


def events(rows):
    def first(p):
        for r in rows:
            if p(r):
                return float(r["t"])
        return float("nan")
    return dict(t_charge=first(lambda r: r["charge_sum_abs"] > 0), t_dips=first(lambda r: r["n_dips"] > 0),
                t_circ=first(lambda r: r["circ_max"] >= 0.25), t_wound=first(lambda r: r["n_wound"] >= 1))


def at_time(rows, tq, key):
    if not np.isfinite(tq):
        return float("nan")
    t = np.array([r["t"] for r in rows]); j = int(np.argmin(np.abs(t - tq)))
    return float(rows[j][key])


# ---------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------

def marks(ax, ev):
    for (kk, c) in (("t_charge", C1), ("t_dips", C2), ("t_circ", C3), ("t_wound", C4)):
        if np.isfinite(ev[kk]):
            ax.axvline(ev[kk], color=c, lw=1.0, ls=":")


def fig_file(rows, ev, stem, mu, path, floor):
    t = np.array([r["t"] for r in rows]); g = lambda k: np.array([r[k] for r in rows], float)
    fig, axs = plt.subplots(5, 1, figsize=(10, 13), sharex=True)
    ax = axs[0]
    ax.plot(t, g("dm_trough"), color=C1, label="row demodulation (x-average): trough A_min / A_far")
    ax.plot(t, g("ld_trough"), color=C2, label="local row demodulation: median column trough / A_far")
    ax.plot(t, g("ld_A_line"), color=C3, label="local row demodulation: on the line / A_far")
    ax.plot(t, g("mg_trough"), color=C2, lw=1.0, ls="--", label="mg estimator: median column minimum / A_far")
    ax.axhline(floor["mg_trough"], color=MUTED, lw=0.8, ls="--", label="mg on a smooth knee (floor)")
    ax.plot(t, g("xinv_row"), color=MUTED, lw=1.2, ls="-.", label="x-invariance of the line row")
    ax.set_ylabel("amplitude"); ax.legend(fontsize=7)
    ax = axs[1]
    ax.plot(t, g("split_med"), color=C2, label="median |y of the column minimum| / L (local demodulation)")
    ax.plot(t, g("split_spread"), color=C2, lw=1.2, ls="--", label="its spread along x")
    ax.plot(t, g("split_frac"), color=C4, lw=1.2, label="share of columns with the minimum > 0.1 L off the line")
    ax.plot(t, np.abs(g("dm_trough_y")), color=C1, label="|y of the x-averaged trough| / L")
    ax.axhline(np.sqrt(1 - mu ** 2) / (2 * mu), color=MUTED, lw=0.8, ls="--", label="beat node of the crossing rolls, pi/(2 mu)")
    ax.set_ylabel("L"); ax.legend(fontsize=7)
    ax = axs[2]
    ax.plot(t, g("lapA_line"), color=C3, label="L^2 Lap A / A on the line (mg)")
    ax.plot(t, g("lapA_trough"), color=C2, label="L^2 Lap A / A at the trough points (mg)")
    ax.plot(t, g("r_A"), color=C1, lw=1.2, label="r_A = -(A''/A)/k1^2 on the line (row demodulation)")
    ax.axhline(floor["lapA_line"], color=MUTED, lw=0.8, ls="--", label="mg on a smooth knee (floor)")
    ax.axhline(0, color=MUTED, lw=0.6); ax.legend(fontsize=7)
    ax = axs[3]
    ax.semilogy(t, np.maximum(g("spec_A_line"), 1e-8), color=C1, label="mg A on the line: power at 2 k1 / mean square")
    ax.semilogy(t, np.maximum(g("spec_env_line"), 1e-8), color=C2, label="Hilbert envelope of the line row: same")
    ax.semilogy(t, np.maximum(g("spec_J_line"), 1e-8), color=C3, label="J on the line: same")
    ax.semilogy(t, np.maximum(g("off_J_line"), 1e-8), color=C4, label="J on the line: share of variance between the harmonics 2 n k1")
    ax.axhline(max(floor["off_J_line"], 1e-8), color=C4, lw=0.8, ls="--")
    for k, c in (("spec_A_line", C1), ("spec_env_line", C2)):
        ax.axhline(max(floor[k], 1e-8), color=c, lw=0.8, ls="--")
    ax.legend(fontsize=7); ax.set_title("dashed: the same on a smooth knee (floor)", fontsize=8)
    ax = axs[4]
    ax.plot(t, g("n_dips"), color=C2, label="amplitude dips")
    ax.plot(t, g("n_wound"), color=C4, label="wound dips")
    ax.plot(t, g("charge_sum_abs") / 2.0, color=C1, label="director charge sum|q| / 2")
    ax.plot(t, 10 * g("circ_max"), color=C3, ls="--", label="10 x largest circulation (turns)")
    ax.set_xlabel("t"); ax.legend(fontsize=7)
    for ax in axs:
        marks(ax, ev)
    fig.suptitle("%s, mu %.3f: amplitude on the knee core line (dotted: first charge, dip, circulation, winding)" % (stem, mu), fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def fig_kymo(kA, kJ, kL, xi, t, ev, stem, mu, Lsp, path):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    ext = [xi[0] / Lsp, xi[-1] / Lsp, t[0], t[-1]]
    im = axs[0].imshow(kA, origin="lower", aspect="auto", extent=ext, cmap="Blues_r", vmin=np.nanpercentile(kA, 1), vmax=1.0)
    fig.colorbar(im, ax=axs[0], label="mg A on the line / A_far")
    im = axs[1].imshow(kL, origin="lower", aspect="auto", extent=ext, cmap="Blues_r", vmin=np.nanpercentile(kL, 1), vmax=np.nanpercentile(kL, 99))
    fig.colorbar(im, ax=axs[1], label="local row demodulation on the line / A_far")
    v = np.nanpercentile(np.abs(kJ), 99) or 1.0
    im = axs[2].imshow(kJ, origin="lower", aspect="auto", extent=ext, cmap="RdBu_r", vmin=-v, vmax=v)
    fig.colorbar(im, ax=axs[2], label="J L^2 on the line (mg)")
    for ax in axs:
        ax.set_xlabel("x / L")
        for (kk, c) in (("t_charge", C1), ("t_dips", C2), ("t_circ", C3), ("t_wound", C4)):
            if np.isfinite(ev[kk]):
                ax.axhline(ev[kk], color=c, lw=1.0, ls=":")
    axs[0].set_ylabel("t")
    fig.suptitle("%s, mu %.3f: the knee core line against time (dotted: first charge, dip, circulation, winding)" % (stem, mu), fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def line_spectrum(sig, dx):
    w = np.hanning(len(sig))
    s = (sig - np.sum(w * sig) / w.sum()) * w
    F = np.fft.rfft(s)
    q = 2 * np.pi * np.fft.rfftfreq(len(sig), dx)
    ms = np.sum(w * sig ** 2) / w.sum()
    return q, (np.abs(F) * 2.0 / w.sum()) ** 2 / 2.0 / max(ms, 1e-300)


def fig_spec(kA, kJ, kf, xi, t, ev, stem, mu, k1, path):
    dx = float(xi[1] - xi[0])
    picks = [("first frame", t[0], MUTED)]
    for kk, lab, c in (("t_charge", "first charge", C1), ("t_dips", "first dip", C2), ("t_circ", "first circulation", C3), ("t_wound", "first winding", C4)):
        if np.isfinite(ev[kk]):
            picks.append((lab, ev[kk], c))
    picks.append(("end", t[-1], "#222222"))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, K, Kf, name in ((axs[0], kA, kf["A"], "mg A on the line"), (axs[1], kJ, kf["J"], "J on the line")):
        q, P = line_spectrum(Kf.astype(float), dx)
        ax.semilogy(q / k1, np.maximum(P, 1e-10), color="#bbbbbb", lw=2.5, label="smooth knee (floor)")
        for lab, tq, c in picks:
            j = int(np.argmin(np.abs(t - tq)))
            q, P = line_spectrum(K[j].astype(float), dx)
            ax.semilogy(q / k1, np.maximum(P, 1e-10), color=c, lw=1.3, label="%s (t %.2f)" % (lab, t[j]))
        for m in (2, 4):
            ax.axvline(m, color=MUTED, lw=0.6, ls=":")
        ax.set_xlim(0, 6); ax.set_xlabel("q / k1 (dislocation harmonic at 2)"); ax.set_title(name + ": power / mean square", fontsize=9)
    axs[0].legend(fontsize=7)
    fig.suptitle("%s, mu %.3f: along-line spectra at the formation events" % (stem, mu), fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def fig_set(results, outdir):
    results = sorted(results, key=lambda r: r["mu"])
    mus = np.array([r["mu"] for r in results])
    cmap = plt.get_cmap("viridis"); norm = plt.Normalize(mus.min(), mus.max() if mus.max() > mus.min() else mus.min() + 1e-3)
    # events table and figure
    keys = ["dm_trough", "ld_trough", "ld_A_line", "mg_trough", "split_med", "split_over_node", "split_frac", "mirror_asym", "lapA_trough", "lapA_line", "r_A",
            "spec_A_line", "spec_env_line", "spec_J_line", "off_J_line", "offrms_J_line", "off_A_line", "rms_J_line", "xinv_row"]
    evrows = []
    for r in results:
        row = dict(stem=r["stem"], mu=r["mu"], **r["events"])
        for k in ("mg_trough", "mg_A_line", "lapA_line", "spec_A_line", "spec_env_line", "spec_J_line", "off_J_line", "offrms_J_line", "off_A_line", "rms_J_line", "n_dips", "charge_sum_abs"):
            row["floor_" + k] = r["floor"][k]
        for k in keys:
            for en, tq in (("charge", r["events"]["t_charge"]), ("dips", r["events"]["t_dips"]), ("wound", r["events"]["t_wound"]), ("end", r["rows"][-1]["t"])):
                row["%s_at_%s" % (k, en)] = at_time(r["rows"], tq, k)
        evrows.append(row)
    with open(outdir / "events.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(evrows[0].keys())); wr.writeheader(); wr.writerows(evrows)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, k, lab in zip(axs.ravel(), ["ld_trough", "ld_A_line", "split_over_node", "r_A", "offrms_J_line", "rms_J_line"],
                          ["k1 projection at the node: |a+ - a-| / A_far", "k1 projection on the line: (a+ + a-) / A_far",
                           "off-line distance of the minimum / beat node pi/(2 mu)", "r_A on the line",
                           "J L^2 on the line: rms between the harmonics", "rms J L^2 on the line"]):
        for en, c, mk in (("charge", C1, "o"), ("dips", C2, "s"), ("wound", C4, "D"), ("end", MUTED, "x")):
            v = np.array([row["%s_at_%s" % (k, en)] for row in evrows])
            ok = np.isfinite(v)
            ax.plot(mus[ok], v[ok], "-", marker=mk, color=c, ms=5, lw=1.5, label="at first %s" % en if en != "end" else "at the end")
        if "floor_" + k in evrows[0]:
            ax.plot(mus, [row["floor_" + k] for row in evrows], color="#bbbbbb", lw=2.0, label="smooth knee (floor)")
        if k == "offrms_J_line":
            ax.set_yscale("log")
        ax.set_title(lab, fontsize=9); ax.set_xlabel("mu")
        for m in (0.685, 0.705, 0.815):
            ax.axvline(m, color=MUTED, lw=0.6, ls=":")
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Amplitude quantities at each formation event, against mu (dotted: charge, dip and winding thresholds of the 12 Oct run)", fontsize=10)
    fig.tight_layout(); fig.savefig(outdir / "fig_events_vs_mu.png", dpi=130); plt.close(fig)
    # collapse
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for r in results:
        t = np.array([q["t"] for q in r["rows"]]); tc = r["events"]["t_charge"]
        col = cmap(norm(r["mu"]))
        axs[0].plot(t, [q["dm_trough"] for q in r["rows"]], color=col, lw=1.2)
        if np.isfinite(tc):
            axs[1].plot(t - tc, [q["dm_trough"] for q in r["rows"]], color=col, lw=1.2)
            axs[2].plot(t - tc, [q["split_med"] for q in r["rows"]], color=col, lw=1.2)
    axs[0].set_xlabel("t"); axs[0].set_title("demodulated trough against t", fontsize=9)
    axs[1].set_xlabel("t - t_charge"); axs[1].set_title("the same against time since the first charge", fontsize=9)
    axs[2].set_xlabel("t - t_charge"); axs[2].set_title("off-line distance of the minimum / L", fontsize=9)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); fig.colorbar(sm, ax=axs.tolist(), label="mu")
    fig.savefig(outdir / "fig_collapse.png", dpi=130, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------
# synthetic movie
# ---------------------------------------------------------------------

def synthetic_movie(args):
    """a knee (beta 1) whose amplitude trough deepens, splits off the line
    after t_split to a planted offset, and gains a 2 k1 modulation along
    the line; returns the movie and the planted curves"""
    mu = args.mu; k1 = np.sqrt(1 - mu ** 2); Lsp = np.pi / k1
    dx = 0.25
    nx = int(round(6 * 2 * np.pi / k1 / dx)); x = (np.arange(nx) - nx // 2) * dx
    y = np.arange(-int(8 * Lsp / dx), int(8 * Lsp / dx) + 1) * dx
    X, Y = np.meshgrid(x, y)
    t = np.linspace(0, 10, 41)
    depth = 0.15 + 0.35 * t / t[-1]
    t_split = 5.0
    off = np.where(t > t_split, 0.35 * np.minimum(1.0, (t - t_split) / 2.0), 0.0)
    modu = 0.02 + 0.25 * np.clip((t - 3.0) / 7.0, 0, 1)
    wid = 0.25 * Lsp
    z = mu * Y; phi = np.logaddexp(z, -z)
    u = np.empty((len(y), nx, len(t)), np.float32)
    for i in range(len(t)):
        tr = 0.5 * (1 / np.cosh((Y - off[i] * Lsp) / wid) ** 2 + 1 / np.cosh((Y + off[i] * Lsp) / wid) ** 2)
        A = 1.0 - depth[i] * tr * (1 + modu[i] * np.cos(2 * k1 * X))
        u[..., i] = A * np.cos(k1 * X + phi)
    md = dict(mu=mu, gb_row_in_crop=int(np.argmin(np.abs(y))))
    yy = np.linspace(-1.5 * Lsp, 1.5 * Lsp, 3001)
    amin, aline, env = [], [], []
    for i in range(len(t)):
        tr = 0.5 * (1 / np.cosh((yy - off[i] * Lsp) / wid) ** 2 + 1 / np.cosh((yy + off[i] * Lsp) / wid) ** 2)
        Am = 1.0 - depth[i] * tr
        amin.append(Am.min()); aline.append(Am[len(yy) // 2])
        a0 = 1.0 - depth[i] * tr[len(yy) // 2]; a2 = depth[i] * tr[len(yy) // 2] * modu[i]
        env.append(a2 ** 2 / 2.0 / (a0 ** 2 + a2 ** 2 / 2.0))
    return (u, x, y, t, md, "synthetic_mu%.2f" % mu), dict(t=t, depth=depth, off=off, modu=modu, amin=np.array(amin),
                                                         aline=np.array(aline), env2k1=np.array(env))


# ---------------------------------------------------------------------

def main(args):
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    lines = ["knee_amplitude_sweep v1.0; sigma_dir %.2g, band_min %.2g L (<= 0: 1.5 x the beat node, at least 1 L), loc_L %.2g L, far_from %.2g L, frame_stride %d, sigma_det %.2g" % (
        args.sigma_dir, args.band_min, args.loc_L, args.far_from, args.frame_stride, args.sigma_det_charge), ""]
    results = []
    if args.mode == "synthetic":
        mov, planted = synthetic_movie(args)
        r = run_file(None, args, out, lines, u_override=mov)
        t = np.array([q["t"] for q in r["rows"]])
        pd = np.interp(t, planted["t"], planted["depth"]); po = np.interp(t, planted["t"], planted["off"])
        dd = np.array([q["dm_trough"] for q in r["rows"]]); ld = np.array([q["ld_trough"] for q in r["rows"]])
        so = np.array([q["split_med"] for q in r["rows"]]); md_ = np.array([q["mg_trough"] for q in r["rows"]])
        se = np.array([q["spec_env_line"] for q in r["rows"]]); sa = np.array([q["spec_A_line"] for q in r["rows"]])
        pm = np.interp(t, planted["t"], planted["modu"])
        pmin = np.interp(t, planted["t"], planted["amin"]); pline = np.interp(t, planted["t"], planted["aline"])
        penv = np.interp(t, planted["t"], planted["env2k1"])
        lines.append("SYNTHETIC: planted x-mean minimum of A, its offset, and the 2 k1 power of the planted envelope on the line, against recovered:")
        for i in range(0, len(t), max(1, len(t) // 10)):
            lines.append("   t %5.2f planted min %.3f at %.3f L (line %.3f), 2k1 env power %.5f | demod trough %.3f, local trough %.3f, split %.3f L,"
                         " mg trough %.3f | envelope power %.5f, mg A power %.5f" % (
                             t[i], pmin[i], po[i], pline[i], penv[i], dd[i], ld[i], so[i], md_[i], se[i], sa[i]))
        aft = t > 6.5
        lines.append("   after the split has settled (t > 6.5): max |split - planted offset| %.3f L; max |local trough - planted min| %.3f;"
                     " envelope power / planted: %.2f to %.2f" % (
                         float(np.max(np.abs(so - po)[aft])), float(np.max(np.abs(ld - pmin)[aft])),
                         float(np.min(se[aft] / penv[aft])), float(np.max(se[aft] / penv[aft]))))
        lines.append("   before the split (t <= 5): max split %.3f L; max |local trough - planted min| %.3f" % (
            float(np.max(so[t <= 5.0])), float(np.max(np.abs(ld - pmin)[t <= 5.0]))))
        results.append(r)
    else:
        for f in [s.strip() for s in args.npz.split(",") if s.strip()]:
            try:
                results.append(run_file(f, args, out, lines))
            except Exception as exc:                                     # loud flag, keep going
                lines.append("FAILED %s: %r" % (f, exc))
            (out / "report.txt").write_text("\n".join(lines) + "\n")
    if len(results) > 1:
        fig_set(results, out)
        lines.append(""); lines.append("set figures: fig_events_vs_mu.png, fig_collapse.png; table events.csv")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    with open(out / "config.json", "w") as fh:
        json.dump({k: getattr(args, k) for k in vars(args)}, fh, indent=2)
    print("\n".join(lines)); print("outputs ->", out)


class _Args:
    mode = "data"            # data | synthetic
    npz = ""                 # one movie or a comma-separated list
    outdir = "results/knee_amplitude_sweep"
    mu = 0.85                # synthetic mode, and fallback when the file has no mu
    sigma_dir = 0.4
    taper_frac = 0.06
    core_frac = 0.70
    sigma_det_charge = 0.3
    band_min = -1.0          # search half-width for the column minimum, in L; <= 0: max(1, 1.5 pi/(2 mu) in L), so the beat node is inside at shallow angles
    loc_L = 0.5              # Gaussian half-width (standard deviation) of the local row demodulation, in L
    min_contrast = 0.03      # a column needs this much contrast of |c| (over A_far) in the band to have a trough position
    far_from = 4.0           # far field for A_far, in L from the line
    frame_stride = 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in vars(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    main(p.parse_args())
