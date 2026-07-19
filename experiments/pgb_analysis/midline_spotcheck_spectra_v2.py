# experiments/pgb_analysis/midline_spotcheck_spectra.py
"""
Theory-blind midline spot checks, spectral evolution, correlations, and
coordinate-line curvature for PGB runs.

Loads a matched pair of files per mu:
  * uhu OP npz   (x, y, u, tt, ramp, A, k1, k2, lam1, lam2, metadata)
  * phase OP npz (phase_grid_[symmetric_]wrapped/unwrapped,
                  analytic_amplitude_grid[_symmetric], coordinate_lines)

Outputs per run (out_dir/<stem>/):
  figs/midlines_initial_final.png   overlaid initial vs final midlines of
                                    u, J, A_uhu, A_analytic, theta_uw,
                                    theta_w, lam1, lam2
  figs/fieldmap_<name>.png          2D initial|final maps of each field
  figs/spectrum_<name>.png          per-field: spectrogram, t0-vs-T spectra
                                    overlay, emergent-mode time series M(t)
                                    and dM/dt, n_modes(t)
  figs/line_curvature.png           near-core curvature of k-coordinate
                                    lines: per-line fingerprint (initial vs
                                    final) + curvature time series
  figs/correlations.png             Pearson correlation heatmaps of the
                                    scalar time series and their d/dt
  figs/top_pairs.png                scatter plots of the strongest pairs
  gifs/<name>_midline.gif           midline evolution animations
  data/*.csv, data/summary.json     everything numeric

No reference to any theoretical wavenumber is made anywhere: the "emergent
mode" is detected empirically as a spectral peak present in the late-time
spectrum that is absent (or far weaker) in the early-time spectrum.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

_HERE = Path(__file__).resolve().parent


# -----------------------------------------------------------------------
# small utilities
# -----------------------------------------------------------------------

def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def interp_nans_1d(v):
    v = np.asarray(v, dtype=float)
    bad = ~np.isfinite(v)
    if bad.all():
        return np.zeros_like(v)
    if bad.any():
        idx = np.arange(v.size)
        v = v.copy()
        v[bad] = np.interp(idx[bad], idx[~bad], v[~bad])
    return v


def get_meta(npz, key):
    for mk in ("sh_meta_json", "uhu_meta_json", "phase_meta_json"):
        if mk in npz:
            try:
                raw = npz[mk]
                meta = json.loads(str(raw.item() if hasattr(raw, "item")
                                      else raw))
                if key in meta and meta[key] is not None:
                    return meta[key]
            except Exception:
                pass
    return None


def mu_token(stem):
    m = re.search(r"mu(\d+\.\d+)", stem)
    return m.group(1) if m else None


def pick_core_row(op, y, valid):
    j = get_meta(op, "gb_row_in_crop")
    if j is not None:
        j = int(j)
        if 0 <= j < len(y):
            return j, "metadata"
    A = op["A"]
    varA = np.where(valid, np.nanvar(A, axis=-1), np.nan)
    row_ok = np.sum(valid, axis=1) >= 0.5 * valid.shape[1]
    score = np.full(len(y), np.nan)
    if row_ok.any():
        with np.errstate(invalid="ignore"):
            score[row_ok] = np.nanmean(varA[row_ok, :], axis=1)
    return int(np.nanargmax(score)), "max-var"


def compute_J(f, g, dx, dy, valid):
    """det(grad k) with pi-jump masking (self-contained)."""
    phi = np.arctan2(g, f)
    dpx = np.abs((np.diff(phi, axis=1, append=phi[:, -1:]) + np.pi)
                 % (2 * np.pi) - np.pi)
    dpy = np.abs((np.diff(phi, axis=0, append=phi[-1:, :]) + np.pi)
                 % (2 * np.pi) - np.pi)
    pj = (dpx > 0.9 * np.pi) | (dpy > 0.9 * np.pi)
    fx = np.gradient(f, dx, axis=1); fy = np.gradient(f, dy, axis=0)
    gx = np.gradient(g, dx, axis=1); gy = np.gradient(g, dy, axis=0)
    J = fx * gy - fy * gx
    return np.where(valid & ~pj, J, np.nan)


def detrend_row(v, x, linear=False):
    v = interp_nans_1d(v)
    if linear:
        c = np.polyfit(x, v, 1)
        return v - np.polyval(c, x)
    return v - v.mean()


# -----------------------------------------------------------------------
# spectra and emergent-mode detection (fully data-driven)
# -----------------------------------------------------------------------

def row_spectrum(v, x, linear_detrend=False, pad_factor=8):
    """Zero-padded (sinc-interpolated) spectrum of one midline row.
    Padding evaluates the same spectrum on a pad_factor-times finer
    q grid: smooth curves + sub-bin peak localization. It does NOT add
    true resolution, which is fixed at dq = 2*pi/L by the domain length."""
    dx = float(x[1] - x[0])
    w = detrend_row(v, x, linear=linear_detrend)
    W = np.hanning(w.size)
    n_fft = int(pad_factor) * w.size
    F = np.abs(np.fft.rfft(w * W, n=n_fft))
    q = 2 * np.pi * np.fft.rfftfreq(n_fft, dx)
    return q, F


def field_spectrogram(F_xt, x, linear_detrend=False, pad_factor=8):
    rows = [row_spectrum(F_xt[i], x, linear_detrend, pad_factor) for i in
            range(F_xt.shape[0])]
    q = rows[0][0]
    S = np.array([r[1] for r in rows])          # (Nt, Nq)
    return q, S


def spectral_peaks(q, s, rel=0.05, q_lo=0.05, min_sep_q=0.1):
    sel = q > q_lo
    if not sel.any():
        return np.array([], int)
    smax = s[sel].max()
    dq = q[1] - q[0]
    dist = max(1, int(round(min_sep_q / dq)))
    pk, _ = find_peaks(s, height=rel * smax,
                       prominence=0.5 * rel * smax, distance=dist)
    return pk[q[pk] > q_lo]


def track_modes(q, S, t, n_avg=5, q_lo=0.05, win_q=0.05,
                merge_tol=0.12):
    """Identify all spectral modes present early OR late, and fit each
    one's exponential growth/decay rate over the full series.

    Returns a list of dicts sorted by q:
      q     : mode center (from padded/interpolated spectrum)
      M     : amplitude time series (max of S over q +- win_q)
      rho   : d/dt log M  (least-squares slope; >0 growth, <0 decay)
      r2    : fit quality
      ratio : late/early amplitude ratio
    """
    S0 = S[:n_avg].mean(axis=0)
    S1 = S[-n_avg:].mean(axis=0)
    cand = sorted(set(
        [float(q[i]) for i in spectral_peaks(q, S0, q_lo=q_lo)] +
        [float(q[i]) for i in spectral_peaks(q, S1, q_lo=q_lo)]))
    centers = []
    for qv in cand:
        if not centers or qv - centers[-1] > merge_tol:
            centers.append(qv)
    modes = []
    for qv in centers:
        lo = int(np.searchsorted(q, qv - win_q))
        hi = int(np.searchsorted(q, qv + win_q)) + 1
        M = S[:, lo:hi].max(axis=1)
        yv = np.log(np.maximum(M, 1e-30))
        A_ = np.vstack([t, np.ones_like(t)]).T
        coef, *_ = np.linalg.lstsq(A_, yv, rcond=None)
        yhat = A_ @ coef
        r2 = 1.0 - ((yv - yhat) ** 2).sum() / (
            ((yv - yv.mean()) ** 2).sum() + 1e-30)
        modes.append(dict(
            q=float(qv), M=M, rho=float(coef[0]), r2=float(r2),
            ratio=float(M[-n_avg:].mean() / (M[:n_avg].mean() + 1e-30))))
    return modes


def detect_emergent_mode(q, S, n_avg=5, q_lo=0.05, match_tol_q=0.15):
    """Compare early vs late mean spectra; return info on the mode that is
    present late but absent (or much weaker) early. Falls back to the
    largest-growth bin if every late peak already existed early."""
    S0 = S[:n_avg].mean(axis=0)
    S1 = S[-n_avg:].mean(axis=0)
    dq = q[1] - q[0]
    tol_bins = max(1, int(round(match_tol_q / dq)))
    pk0 = spectral_peaks(q, S0, q_lo=q_lo)
    pk1 = spectral_peaks(q, S1, q_lo=q_lo)
    new = [p for p in pk1
           if not any(abs(p - p0) <= tol_bins for p0 in pk0)]
    if new:
        iq = int(new[int(np.argmax(S1[new]))])
        kind = "new_peak"
    else:
        sel = q > q_lo
        growth = np.where(sel, S1 - S0, -np.inf)
        iq = int(np.argmax(growth))
        kind = "grown_bin"
    hw = max(2, int(round(0.05 / dq)))   # +-0.05 physical half-window
    lo, hi = max(iq - hw, 0), min(iq + hw + 1, len(q))
    M = S[:, lo:hi].max(axis=1)
    ratio = float(S1[iq] / (S0[iq] + 1e-30))
    return {"iq": iq, "q": float(q[iq]), "kind": kind, "M": M,
            "growth_ratio": ratio, "n_peaks_t0": int(len(pk0)),
            "n_peaks_T": int(len(pk1))}


def n_modes_series(q, S, rel=0.1, q_lo=0.05):
    return np.array([len(spectral_peaks(q, S[i], rel=rel, q_lo=q_lo))
                     for i in range(S.shape[0])])


# -----------------------------------------------------------------------
# coordinate-line curvature near the core
# -----------------------------------------------------------------------

def line_curvature(pts, smooth_sigma=2.0):
    pts = np.asarray(pts, float)
    if pts.ndim != 2:
        return None
    if pts.shape[0] == 2 and pts.shape[1] != 2:
        pts = pts.T
    if pts.shape[0] < 7:
        return None
    xs = gaussian_filter1d(pts[:, 0], smooth_sigma)
    ys = gaussian_filter1d(pts[:, 1], smooth_sigma)
    x1 = np.gradient(xs); y1 = np.gradient(ys)
    x2 = np.gradient(x1); y2 = np.gradient(y1)
    speed2 = x1 ** 2 + y1 ** 2
    kap = np.abs(x1 * y2 - y1 * x2) / (speed2 ** 1.5 + 1e-30)
    # mask stalled/reversal samples (tracer pathologies, not geometry):
    # where the local speed collapses, the curvature formula divides by
    # ~0 and produces astronomically large spurious values
    med = np.median(speed2)
    kap = np.where(speed2 > 0.09 * med, kap, np.nan)  # speed<30% of median
    return kap, xs, ys


def core_curvature_per_line(lines_t, y_core, band_out=4.0, band_in=1.0):
    """For each coordinate line at one frame: max curvature within the
    ANNULAR band  band_in < |y - y_core| < band_out, and the line's seed
    x. The inner exclusion keeps the (near-singular) core crossing point
    itself out of the statistic, so the number measures bending NEAR the
    core rather than the vertex kink AT it. Returns arrays sorted by
    seed x."""
    seeds, kmax = [], []
    for ln in lines_t:
        out = line_curvature(ln)
        if out is None:
            continue
        kap, xs, ys = out
        d = np.abs(ys - y_core)
        near = (d < band_out) & (d > band_in)
        if not near.any():
            continue
        seeds.append(xs[0])
        kmax.append(float(np.nanmax(kap[near])))
    if not seeds:
        return np.array([]), np.array([])
    o = np.argsort(seeds)
    return np.array(seeds)[o], np.array(kmax)[o]


# -----------------------------------------------------------------------
# per-run processing
# -----------------------------------------------------------------------

def process_pair(uhu_path, phase_path, base_out, cfg):
    uhu_path, phase_path = Path(uhu_path), Path(phase_path)
    stem = uhu_path.stem
    out = ensure_dir(Path(base_out) / stem)
    fig_dir = ensure_dir(out / "figs")
    gif_dir = ensure_dir(out / "gifs")
    data_dir = ensure_dir(out / "data")

    op = np.load(uhu_path, allow_pickle=True)
    pf = np.load(phase_path, allow_pickle=True)

    x = op["x"]; y = op["y"]
    dx = float(x[1] - x[0]); dy = float(y[1] - y[0])
    t = op["tt"] if "tt" in op and op["tt"].size else \
        np.arange(op["A"].shape[-1])

    ramp = op["ramp"] if "ramp" in op else np.ones((len(y), len(x)))
    rn = (ramp - np.nanmin(ramp)) / (np.nanmax(ramp) - np.nanmin(ramp)
                                     + 1e-12)
    valid = rn >= cfg["domain_thresh"]
    j_core, how = pick_core_row(op, y, valid)
    y_core = float(y[j_core])

    # --- gather fields as (Ny, Nx, Nt), truncating to common Nt ---
    def pget(*names):
        for nm in names:
            if nm in pf and pf[nm] is not None and np.asarray(pf[nm]).size:
                arr = np.asarray(pf[nm], float)
                if arr.ndim == 3:
                    return arr
        return None

    prefer = cfg["prefer_sym"]
    th_uw = pget(*(("phase_grid_symmetric_unwrapped",
                    "phase_grid_unwrapped") if prefer else
                   ("phase_grid_unwrapped",
                    "phase_grid_symmetric_unwrapped")))
    th_w = pget(*(("phase_grid_symmetric_wrapped",
                   "phase_grid_wrapped") if prefer else
                  ("phase_grid_wrapped", "phase_grid_symmetric_wrapped")))
    A_an = pget(*(("analytic_amplitude_grid_symmetric",
                   "analytic_amplitude_grid") if prefer else
                  ("analytic_amplitude_grid",
                   "analytic_amplitude_grid_symmetric")))

    Nt = min(op["A"].shape[-1],
             *[a.shape[-1] for a in (th_uw, th_w, A_an) if a is not None])
    if Nt < op["A"].shape[-1]:
        print(f"  NOTE: truncating to common Nt={Nt} "
              f"(uhu has {op['A'].shape[-1]}) -- check that uhu and phase "
              f"dirs come from the SAME sweep")
    t = t[:Nt]

    J3 = np.empty((len(y), len(x), Nt))
    for it in range(Nt):
        J3[..., it] = compute_J(op["k1"][..., it], op["k2"][..., it],
                                dx, dy, valid)

    fields2d = {
        "J": J3,
        "A_uhu": op["A"][..., :Nt],
        "A_analytic": A_an[..., :Nt] if A_an is not None else None,
        "theta_uw": th_uw[..., :Nt] if th_uw is not None else None,
        "theta_w": th_w[..., :Nt] if th_w is not None else None,
        "lam1": op["lam1"][..., :Nt],
        "lam2": op["lam2"][..., :Nt],
        "u": op["u"][..., :Nt],
    }
    for nm, a in list(fields2d.items()):
        if a is None:
            print(f"  WARNING: field {nm} not found in phase file; skipped")
            fields2d.pop(nm)

    # --- valid columns on the core row, with x_trim ---
    cols = valid[j_core, :].copy()
    idx = np.where(cols)[0]
    if cfg["x_trim"] > 0 and idx.size:
        n_trim = int(round(cfg["x_trim"] / dx))
        cols[:] = False
        lo, hi = idx[0] + n_trim, idx[-1] - n_trim
        if hi > lo:
            cols[lo:hi + 1] = True
    xs = x[cols]

    midlines = {nm: np.array([np.where(cols, a[j_core, :, it], np.nan)[cols]
                              for it in range(Nt)])
                for nm, a in fields2d.items()}

    print(f"  {stem}: core row y={y_core:.3f} [{how}], "
          f"{int(cols.sum())}/{len(x)} cols, Nt={Nt}")

    # =========================== SPOT CHECKS ===========================
    names = list(midlines.keys())
    ncols_fig = 2
    nrows_fig = int(np.ceil(len(names) / 2))
    fig, axs = plt.subplots(nrows_fig, ncols_fig,
                            figsize=(12, 2.6 * nrows_fig), sharex=True)
    for ax, nm in zip(np.atleast_1d(axs).ravel(), names):
        ax.plot(xs, midlines[nm][0], lw=1.2, label="initial")
        ax.plot(xs, midlines[nm][-1], lw=1.2, label="final")
        ax.set_title(nm, fontsize=10)
        ax.legend(fontsize=7)
    for ax in np.atleast_1d(axs).ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(f"{stem}\nmidlines at t0 and T (y_core={y_core:.2f})",
                 fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "midlines_initial_final.png", dpi=180)
    plt.close(fig)

    # 2D field maps, initial | final
    for nm, a in fields2d.items():
        # same validity mask used for J: suppress ramp-edge artifacts
        f0 = np.where(valid, a[..., 0], np.nan)
        f1 = np.where(valid, a[..., -1], np.nan)
        vals = np.concatenate([f0[np.isfinite(f0)], f1[np.isfinite(f1)]])
        if vals.size == 0:
            continue
        vmin, vmax = np.percentile(vals, [2, 98])
        fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, f, lab in zip(axs, (f0, f1), ("initial", "final")):
            im = ax.pcolormesh(x, y, f, shading="auto", vmin=vmin,
                               vmax=vmax, cmap="viridis")
            ax.axhline(y_core, color="w", lw=0.5, ls=":")
            ax.set_title(f"{nm} ({lab})", fontsize=10)
        fig.colorbar(im, ax=axs, shrink=0.9)
        plt.savefig(fig_dir / f"fieldmap_{nm}.png", dpi=180,
                    bbox_inches="tight")
        plt.close(fig)

    # midline evolution gifs
    if cfg["make_gifs"]:
        for nm in names:
            M = midlines[nm]
            finite = M[np.isfinite(M)]
            if finite.size == 0:
                continue
            ylo, yhi = np.percentile(finite, [1, 99])
            pad = 0.05 * (yhi - ylo + 1e-12)
            fig, ax = plt.subplots(figsize=(7, 3))
            ln, = ax.plot(xs, M[0], lw=1.3)
            ax.set_ylim(ylo - pad, yhi + pad)
            ax.set_xlabel("x"); ax.set_title(f"{nm} midline")
            txt = ax.text(0.02, 0.92, "", transform=ax.transAxes,
                          fontsize=9)

            def _upd(i, ln=ln, txt=txt, M=M):
                ln.set_ydata(M[i])
                txt.set_text(f"t = {t[i]:.3f}")
                return ln, txt

            ani = animation.FuncAnimation(fig, _upd, frames=Nt,
                                          blit=True)
            ani.save(gif_dir / f"{nm}_midline.gif",
                     writer=animation.PillowWriter(fps=cfg["gif_fps"]))
            plt.close(fig)

    # ========================= SPECTRAL LAYER ==========================
    spec_fields = [nm for nm in
                   ("J", "A_uhu", "A_analytic", "theta_uw", "u",
                    "lam1", "lam2") if nm in midlines]
    emergent = {}
    scalars = {}
    for nm in spec_fields:
        lin = (nm == "theta_uw")   # remove mean slope, keep modulation
        q, S = field_spectrogram(midlines[nm], xs, linear_detrend=lin)
        modes = track_modes(q, S, t, n_avg=cfg["n_avg"])
        nm_modes = n_modes_series(q, S)
        emergent[nm] = modes

        fig, axs = plt.subplots(1, 4, figsize=(19, 3.6))
        sel = q <= min(4.0, q[-1])
        im = axs[0].pcolormesh(q[sel], t, np.log10(S[:, sel] + 1e-12),
                               shading="auto", cmap="magma")
        for md in modes:
            axs[0].axvline(md["q"], color="cyan", ls=":", lw=0.7)
        axs[0].set_xlabel("q"); axs[0].set_ylabel("t")
        axs[0].set_title(f"log10 |F_x {nm}|", fontsize=10)
        fig.colorbar(im, ax=axs[0], shrink=0.9)

        axs[1].semilogy(q[sel], S[:cfg["n_avg"]].mean(0)[sel],
                        label="early", lw=1.2)
        axs[1].semilogy(q[sel], S[-cfg["n_avg"]:].mean(0)[sel],
                        label="late", lw=1.2)
        for md in modes:
            axs[1].axvline(md["q"], color="gray", ls=":", lw=0.7)
        s_top = max(S[:cfg["n_avg"]].mean(0)[sel].max(),
                    S[-cfg["n_avg"]:].mean(0)[sel].max())
        axs[1].set_ylim(s_top / 1e4, 2 * s_top)   # hide window sidelobes
        axs[1].set_xlabel("q")
        axs[1].set_title(f"{nm}: spectra early vs late "
                         f"({len(modes)} tracked modes)", fontsize=9)
        axs[1].legend(fontsize=7)

        ax = axs[2]
        if modes:
            qs = [md["q"] for md in modes]
            rr = [md["rho"] for md in modes]
            cc = ["tab:green" if r > 0 else "tab:red" for r in rr]
            ax.bar(qs, rr, width=0.18, color=cc, edgecolor="k",
                   linewidth=0.4)
            span = max(rr) - min(rr) if len(rr) > 1 else abs(rr[0]) + 0.1
            for qv, rv in zip(qs, rr):
                ax.annotate(f"q={qv:.2f}\n{rv:+.2f}",
                            (qv, rv + (0.04 if rv >= 0 else -0.04) * span),
                            ha="center",
                            va="bottom" if rv >= 0 else "top",
                            fontsize=7)
            ax.axhline(0, color="k", lw=0.6)
            ax.margins(y=0.25)
        ax.grid(alpha=0.3)
        ax.set_xlabel("mode q"); ax.set_ylabel("rho = d/dt log M")
        ax.set_title(f"{nm}: per-mode exponential rate "
                     "(LSQ fit over full run)", fontsize=9)

        ax = axs[3]
        for md in modes:
            ax.semilogy(t, md["M"], lw=1.1,
                        label=f"q={md['q']:.2f} "
                              f"(rho={md['rho']:+.2f})")
        ax.set_xlabel("t"); ax.set_ylabel("mode amplitude")
        ax.set_title(f"{nm}: mode amplitudes M(t)", fontsize=9)
        if modes:
            ax.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        plt.savefig(fig_dir / f"spectrum_{nm}.png", dpi=180)
        plt.close(fig)

        if modes:
            best = max(modes, key=lambda m: m["rho"])
            scalars[f"M_{nm}"] = best["M"]
        scalars[f"max_{nm}"] = np.nanmax(midlines[nm], axis=1)
        scalars[f"min_{nm}"] = np.nanmin(midlines[nm], axis=1)
        np.savez_compressed(
            data_dir / f"spectrum_{nm}.npz", q=q, S=S, t=t,
            n_modes=nm_modes,
            mode_q=np.array([md["q"] for md in modes]),
            mode_rho=np.array([md["rho"] for md in modes]),
            mode_r2=np.array([md["r2"] for md in modes]),
            mode_ratio=np.array([md["ratio"] for md in modes]),
            mode_M=np.array([md["M"] for md in modes]))

    # ==================== COORDINATE-LINE CURVATURE ====================
    kappa_ts = None
    mu_val = get_meta(op, "mu")
    if mu_val is None:
        tok = mu_token(stem)
        mu_val = float(tok) if tok else None
    band_out = cfg["curv_band"] if cfg["curv_band"] else \
        (2 * np.pi / mu_val if mu_val else 6.0)
    band_in = cfg["curv_exclude"]
    print(f"    curvature band: {band_in:.2f} < |y-y_core| < "
          f"{band_out:.2f}")
    if "coordinate_lines" in pf:
        try:
            CL = pf["coordinate_lines"]
            n_frames_cl = len(CL)
            n_use = min(n_frames_cl, Nt)
            kap_mean = np.full(n_use, np.nan)
            kap_max = np.full(n_use, np.nan)
            kap_std = np.full(n_use, np.nan)
            for it in range(n_use):
                _, km = core_curvature_per_line(
                    CL[it], y_core, band_out=band_out, band_in=band_in)
                if km.size:
                    kap_mean[it] = km.mean()
                    kap_max[it] = km.max()
                    kap_std[it] = km.std()
            kappa_ts = (kap_mean, kap_max, kap_std)

            s0, k0 = core_curvature_per_line(
                CL[0], y_core, band_out=band_out, band_in=band_in)
            s1, k1_ = core_curvature_per_line(
                CL[n_use - 1], y_core, band_out=band_out,
                band_in=band_in)
            fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
            axs[0].plot(s0, k0, "o-", ms=2.5, lw=1, label="initial")
            axs[0].plot(s1, k1_, "o-", ms=2.5, lw=1, label="final")
            axs[0].set_xlabel("line seed x")
            axs[0].set_ylabel(
                f"max curvature, {band_in:.1f}<|y-yc|<{band_out:.1f}")
            axs[0].set_title("per-line near-core curvature fingerprint",
                             fontsize=9)
            axs[0].legend(fontsize=8)
            axs[1].plot(t[:n_use], kap_mean, label="mean over lines")
            axs[1].plot(t[:n_use], kap_max, label="max over lines")
            axs[1].plot(t[:n_use], kap_std, label="std across lines")
            axs[1].set_xlabel("t"); axs[1].set_title(
                "near-core line curvature vs time", fontsize=9)
            axs[1].legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(fig_dir / "line_curvature.png", dpi=180)
            plt.close(fig)

            # ---- coordinate lines overlaid on the pattern ----
            def _pts(ln):
                p = np.asarray(ln, float)
                if p.ndim != 2:
                    return None
                if p.shape[0] == 2 and p.shape[1] != 2:
                    p = p.T
                return p if p.shape[0] >= 2 else None

            def _seed_x(ln):
                p = _pts(ln)
                return np.nan if p is None else float(p[0, 0])

            u0 = fields2d["u"][..., 0]
            u1 = fields2d["u"][..., n_use - 1]
            lines0 = [p for p in (_pts(l) for l in CL[0])
                      if p is not None]
            lines1 = [p for p in (_pts(l) for l in CL[n_use - 1])
                      if p is not None]
            stride = max(1, len(lines0) // cfg["overlay_n_lines"])

            fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
            for ax, uf, lns, lab, c in ((axs[0], u0, lines0,
                                         "initial", "tab:orange"),
                                        (axs[1], u1, lines1,
                                         "final", "tab:cyan")):
                ax.pcolormesh(x, y, uf, shading="auto", cmap="gray")
                for p in lns[::stride]:
                    ax.plot(p[:, 0], p[:, 1], color=c, lw=0.8, alpha=0.9)
                ax.axhline(y_core, color="r", lw=0.5, ls=":")
                ax.set_title(f"pattern + coordinate lines ({lab})",
                             fontsize=10)
                ax.set_xlim(x[0], x[-1]); ax.set_ylim(y[0], y[-1])

            # zoom: lines seeded near the RIGHT edge of the trimmed
            # validity range, initial vs final overlaid
            ax = axs[2]
            seeds0 = np.array([_seed_x(l) for l in CL[0]])
            x_hi = xs.max() - 2.0
            x_lo = xs.min()
            in_dom = np.where((seeds0 <= x_hi) & (seeds0 >= x_lo))[0]
            if in_dom.size == 0:
                in_dom = np.arange(len(seeds0))
            order = in_dom[np.argsort(seeds0[in_dom])[::-1]]
            pick = sorted(order[:cfg["zoom_n_lines"]])
            ax.pcolormesh(x, y, u1, shading="auto", cmap="gray",
                          alpha=0.5)
            for i in pick:
                p0, p1 = _pts(CL[0][i]), _pts(CL[n_use - 1][i])
                if p0 is not None:
                    ax.plot(p0[:, 0], p0[:, 1], color="tab:orange",
                            lw=1.6, label="initial" if i == pick[0]
                            else None)
                if p1 is not None:
                    ax.plot(p1[:, 0], p1[:, 1], color="tab:cyan",
                            lw=1.6, label="final" if i == pick[0]
                            else None)
            zoom_y = 1.5 * band_out
            xs_pick = [_seed_x(CL[0][i]) for i in pick]
            xc_lo = np.nanmin(xs_pick) - 5.0
            xc_hi = np.nanmax(xs_pick) + 5.0
            ax.set_xlim(xc_lo, xc_hi)
            ax.set_ylim(y_core - zoom_y, y_core + zoom_y)
            ax.axhline(y_core, color="r", lw=0.5, ls=":")
            ax.legend(fontsize=8)
            ax.set_title("central lines: initial vs final (zoom)",
                         fontsize=10)
            plt.tight_layout()
            plt.savefig(fig_dir / "line_overlay.png", dpi=180)
            plt.close(fig)

            n_pad = Nt - n_use
            for nmk, arr in zip(("kappa_mean", "kappa_max", "kappa_std"),
                                kappa_ts):
                scalars[nmk] = np.concatenate(
                    [arr, np.full(n_pad, np.nan)]) if n_pad else arr
        except Exception as e:
            print(f"  WARNING: coordinate-line curvature failed: {e}")
    else:
        print("  NOTE: no coordinate_lines in phase file; "
              "curvature layer skipped")

    # ========================= CORRELATIONS ============================
    keys = list(scalars.keys())
    Xmat = np.vstack([scalars[k] for k in keys])
    dXmat = np.vstack([np.gradient(gaussian_filter1d(
        interp_nans_1d(scalars[k]), 2.0), t) for k in keys])

    def corr_matrix(Z):
        n = Z.shape[0]
        C = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                a, b = Z[i], Z[j]
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() > 4 and a[ok].std() > 0 and b[ok].std() > 0:
                    C[i, j] = np.corrcoef(a[ok], b[ok])[0, 1]
        return C

    C = corr_matrix(Xmat)
    dC = corr_matrix(dXmat)

    fig, axs = plt.subplots(1, 2, figsize=(2 + 0.6 * len(keys) * 2,
                                           1.5 + 0.55 * len(keys)))
    for ax, M_, title in zip(axs, (C, dC),
                             ("corr(series)", "corr(d/dt series)")):
        im = ax.imshow(M_, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=90, fontsize=6)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(keys, fontsize=6)
        ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=axs, shrink=0.8)
    plt.savefig(fig_dir / "correlations.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    # strongest cross-field pairs (exclude same-field combinations)
    def field_of(k):
        if k.startswith("kappa"):
            return "kappa"
        return k.split("_", 1)[1] if "_" in k else k
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if field_of(keys[i]) == field_of(keys[j]):
                continue
            if np.isfinite(C[i, j]):
                pairs.append((abs(C[i, j]), C[i, j], keys[i], keys[j]))
    pairs.sort(reverse=True)
    top = pairs[:4]
    if top:
        fig, axs = plt.subplots(1, len(top), figsize=(4 * len(top), 3.4))
        for ax, (_, r, ka, kb) in zip(np.atleast_1d(axs), top):
            ax.scatter(scalars[ka], scalars[kb], s=8,
                       c=t, cmap="viridis")
            ax.set_xlabel(ka, fontsize=8); ax.set_ylabel(kb, fontsize=8)
            ax.set_title(f"r = {r:+.3f} (color = t)", fontsize=9)
        plt.tight_layout()
        plt.savefig(fig_dir / "top_pairs.png", dpi=180)
        plt.close(fig)

    # ============================ OUTPUTS =============================
    import csv
    with open(data_dir / "scalars.csv", "w", newline="") as fc:
        w = csv.writer(fc)
        w.writerow(["t"] + keys)
        for i in range(Nt):
            w.writerow([t[i]] + [scalars[k][i] for k in keys])

    summary = {
        "file": stem, "mu": get_meta(op, "mu") or mu_token(stem),
        "y_core": y_core,
        **{f"modes_{nm}": [(round(md["q"], 3), round(md["rho"], 3),
                             round(md["ratio"], 2))
                            for md in emergent[nm]] for nm in emergent},
        "top_pairs": [(k[2], k[3], round(k[1], 3)) for k in top],
    }
    with open(data_dir / "summary.json", "w") as fj:
        json.dump(summary, fj, indent=2, default=str)
    for nm in emergent:
        parts = ", ".join(f"q={md['q']:.2f}:rho={md['rho']:+.2f}"
                          for md in emergent[nm])
        print(f"    {nm}: {len(emergent[nm])} modes  [{parts}]")
    if top:
        print("    top pairs: " + "; ".join(
            f"{a}~{b} r={r:+.2f}" for _, r, a, b in top))
    return summary


# -----------------------------------------------------------------------
# pairing + entry points
# -----------------------------------------------------------------------

def run_with_args(ns):
    base_out = ensure_dir(ns.out_dir if ns.out_dir else
                          _HERE / "results" / "midline_spotcheck_spectra")
    cfg = dict(domain_thresh=ns.domain_thresh, x_trim=ns.x_trim,
               prefer_sym=ns.prefer_sym, make_gifs=ns.make_gifs,
               gif_fps=ns.gif_fps, n_avg=ns.n_avg,
               curv_band=ns.curv_band,
               curv_exclude=getattr(ns, "curv_exclude", 1.0),
               overlay_n_lines=getattr(ns, 'overlay_n_lines', 24),
               zoom_n_lines=getattr(ns, 'zoom_n_lines', 5))

    if ns.uhu_file and ns.phase_file:
        pairs = [(Path(ns.uhu_file), Path(ns.phase_file))]
    else:
        ufiles = sorted(Path(ns.uhu_dir).glob(ns.pattern))
        pfiles = sorted(Path(ns.phase_dir).glob(ns.pattern))
        pmap = {}
        for p in pfiles:
            tok = mu_token(p.stem)
            if tok:
                pmap.setdefault(tok, p)
        pairs = []
        for uf in ufiles:
            tok = mu_token(uf.stem)
            if tok and tok in pmap:
                pairs.append((uf, pmap[tok]))
            else:
                print(f"  no phase match for {uf.name}; skipped")
    if not pairs:
        raise SystemExit("No uhu/phase pairs found.")

    summaries = []
    for uf, pfl in pairs:
        print(f"pair: {uf.name}  <->  {pfl.name}")
        try:
            summaries.append(process_pair(uf, pfl, base_out, cfg))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAILED {uf.name}: {e}")
    print("Done.")


def main(args=None):
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--uhu_file", default=None)
    p.add_argument("--phase_file", default=None)
    p.add_argument("--uhu_dir", default=None)
    p.add_argument("--phase_dir", default=None)
    p.add_argument("--pattern", default="*.npz")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--domain_thresh", type=float, default=0.995)
    p.add_argument("--x_trim", type=float, default=8.0)
    p.add_argument("--prefer_sym", type=int, default=1)
    p.add_argument("--make_gifs", type=int, default=1)
    p.add_argument("--gif_fps", type=int, default=15)
    p.add_argument("--n_avg", type=int, default=5,
                   help="frames averaged for early/late spectra")
    p.add_argument("--overlay_n_lines", type=int, default=24,
                   help="number of coordinate lines drawn in the overlay")
    p.add_argument("--zoom_n_lines", type=int, default=10,
                   help="number of central lines in the zoom comparison")
    p.add_argument("--curv_band", type=float, default=None,
                   help="outer half-width (PHYSICAL y units) of the "
                        "near-core curvature band; default None = "
                        "2*pi/mu per file")
    p.add_argument("--curv_exclude", type=float, default=1.0,
                   help="inner exclusion half-width around the core "
                        "crossing (physical units)")
    ns = p.parse_args(args)
    run_with_args(ns)


if __name__ == "__main__":
    if len(sys.argv) == 1:

        class _Args:
            # --- single pair (takes precedence if both set) ---
            uhu_file = None
            phase_file = None

            # --- or matched directories (paired by muX.XXX token) ---
            uhu_dir = (
                "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/uhu/mu_sweeps_full_Nx512_hp2_T25_NyF5_np18_Nsave125/sig_pio2/raw")
            phase_dir = (
                "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/phase/mu_sweeps_full_Nx512_hp2_T25_NyF5_np18_Nsave125/sig_pio2/raw")
            pattern = "*.npz"

            out_dir = ("/Users/edwardmcdugald/patterns/experiments/pgb_analysis/results/midline_spotcheck_spectra_v2/mu_sweeps_full_Nx512_hp2_T25_NyF5_np18_Nsave125/sig_pio2")

            # --- knobs ---
            domain_thresh = 0.995
            x_trim = 8.0        # trimmed from each end of valid x-range
            prefer_sym = True   # prefer *_symmetric phase/amplitude grids
            make_gifs = True    # set False to skip gif rendering (faster)
            gif_fps = 15
            n_avg = 5           # frames averaged for early/late spectra
            curv_band = None    # OUTER curvature band half-width in
                                # PHYSICAL y units; None = 2*pi/mu
            curv_exclude = 1.0  # INNER exclusion: skip the near-singular
                                # core crossing itself (physical units)
            overlay_n_lines = 24  # lines drawn on pattern overlays
            zoom_n_lines = 10     # lines in the zoom comparison,
                                  # taken from the right edge of the
                                  # trimmed validity range

        run_with_args(_Args())
    else:
        main()