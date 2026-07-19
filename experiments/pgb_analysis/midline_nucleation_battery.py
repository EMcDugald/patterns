# experiments/pgb_analysis/midline_nucleation_battery.py
"""
Midline nucleation battery for saved zigzag/diamond OP runs.

For each OP .npz (with keys x, y, u, tt, ramp, k, A, k1, k2, lam1, lam2,
uhu_meta_json), this script extracts space-time midline profiles and produces:

  1) Midline space-time (x, t) heatmaps of:
       - A_uhu (uHu amplitude), raw u, |J|, chi = lam2/lam1 (cross-roll fraction)
  2) Spectrograms |F_x{field}|(q, t) for A, |J|, chi, u along the core,
     with the PN prediction q* = 2*sqrt(1 - mu^2) overlaid
  3) Per-mode exponential growth-rate fits rho(q) over an early time window,
     with rho(q*) reported
  4) Scalar battery time series:
       A_min, A_contrast, chi_max, chi_contrast, maxabsJ, intabsJ,
       spectral peak height at q* for A/J/chi, phase staircase metric,
       u-midline envelope contrast
  5) Threshold-crossing "nucleation times" per diagnostic (event ordering)
  6) Lead-lag cross-correlations between key scalars
  7) A cross-run summary CSV (per mu): q_peak, rho(q*), t_nuc per diagnostic

Follows the file-or-directory conventions of zigzag_defect_J_analysis.py.

NOTE on baselines: the sigma-window puts a q*-scale ripple into A at t=0,
so mode amplitudes are reported both raw and relative to the t=0 frame
(frozen-IC baseline). Contrast metrics are baseline-free.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "src"))

try:
    from utils.kfield_calcs import phi_jump_mask, safe_central_derivs
    _HAVE_UTILS = True
except Exception:
    _HAVE_UTILS = False


# -----------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------

def load_op_npz(path):
    return np.load(path, allow_pickle=True)


def get_mu(op, stem):
    for key in ("uhu_meta_json", "sh_meta_json"):
        if key in op:
            try:
                meta = json.loads(str(op[key].item() if hasattr(op[key], "item")
                                      else op[key]))
                if "mu" in meta and meta["mu"] is not None:
                    return float(meta["mu"])
            except Exception:
                pass
    try:
        part = stem.split("mu", 1)[1]
        for sep in ["_", "T", "t"]:
            if sep in part:
                return float(part.split(sep)[0])
        return float(part)
    except Exception:
        return np.nan


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------------------------------------------------
# Field construction
# -----------------------------------------------------------------------

def compute_J_frame(f, g, dx, dy, valid):
    """J = det(grad k) with pi-jump masking; NaN where invalid."""
    if _HAVE_UTILS:
        phi = np.arctan2(g, f)
        pj = phi_jump_mask(phi, tol=np.pi / 10)
        mask = valid & (~pj) & np.isfinite(f) & np.isfinite(g)
        fx, fy = safe_central_derivs(f, dx, dy, mask)
        gx, gy = safe_central_derivs(g, dx, dy, mask)
        J = fx * gy - fy * gx
        return np.where(mask, J, np.nan)
    # fallback: plain gradients + jump mask via angle differences
    phi = np.arctan2(g, f)
    dphix = np.abs((np.diff(phi, axis=1, append=phi[:, -1:]) + np.pi)
                   % (2 * np.pi) - np.pi)
    dphiy = np.abs((np.diff(phi, axis=0, append=phi[-1:, :]) + np.pi)
                   % (2 * np.pi) - np.pi)
    pj = (dphix > 0.9 * np.pi) | (dphiy > 0.9 * np.pi)
    fx = np.gradient(f, dx, axis=1); fy = np.gradient(f, dy, axis=0)
    gx = np.gradient(g, dx, axis=1); gy = np.gradient(g, dy, axis=0)
    J = fx * gy - fy * gx
    return np.where(valid & ~pj, J, np.nan)


def pick_core_row(op, y, valid, y_core_override=None):
    """Choose the GB core row index.

    Priority:
      1) y_core_override (physical y value supplied by the user)
      2) gb_row_in_crop from the SH metadata carried in the OP file
      3) row with largest time-variance of A among rows with enough
         valid columns (no y=0 assumption: crops keep full-domain y).
    Returns (j_core, how).
    """
    if y_core_override is not None:
        return int(np.argmin(np.abs(y - float(y_core_override)))), "override"

    if "sh_meta_json" in op:
        try:
            raw = op["sh_meta_json"]
            meta = json.loads(str(raw.item() if hasattr(raw, "item") else raw))
            j = meta.get("gb_row_in_crop", None)
            if j is not None:
                j = int(j)
                if 0 <= j < len(y):
                    return j, "metadata"
        except Exception:
            pass

    A = op["A"]
    varA = np.nanvar(A, axis=-1)              # (Ny, Nx)
    varA = np.where(valid, varA, np.nan)
    row_ok = np.sum(valid, axis=1) >= 0.5 * valid.shape[1]
    row_score = np.full(len(y), np.nan)
    if row_ok.any():
        with np.errstate(invalid="ignore"):
            row_score[row_ok] = np.nanmean(varA[row_ok, :], axis=1)
    j_var = int(np.nanargmax(row_score))
    return j_var, "max-var"


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


def staircase_metric(phi_row):
    """Total-variation excess of the (unwrapped) midline phase:
    TV(theta) / |theta(end)-theta(0)| - 1. Zero for monotone/linear phase,
    grows as plateaus+jumps develop. phi_row is the wrapped angle of
    (k1 + i k2); we unwrap along x."""
    phi = interp_nans_1d(phi_row)
    th = np.unwrap(phi)
    tv = np.sum(np.abs(np.diff(th)))
    net = np.abs(th[-1] - th[0])
    if net < 1e-8:
        return np.nan
    return tv / net - 1.0


def contrast(v):
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    vmax, vmin = np.nanmax(v), np.nanmin(v)
    if vmax + vmin == 0:
        return np.nan
    return (vmax - vmin) / (vmax + vmin)


def _nanmin(v):
    v = np.asarray(v, float)
    return np.nan if not np.isfinite(v).any() else float(np.nanmin(v))


def _nanmax(v):
    v = np.asarray(v, float)
    return np.nan if not np.isfinite(v).any() else float(np.nanmax(v))


# -----------------------------------------------------------------------
# J waveform shape metrics (symmetric oscillation -> asymmetric spike train)
# -----------------------------------------------------------------------

def waveform_metrics(J_row, dx, q_star):
    """Shape descriptors of the signed J profile along the core.

    A knee crease shows a fairly symmetric smooth oscillation; a mature PN
    string shows an asymmetric spike train (tall narrow positive peaks,
    shallow broad negative undershoot). Three scalars capture the change:

      skew      : skewness of J values (0 symmetric; grows with spikiness)
      asym      : max(J) / |min(J)|  (1 for symmetric; >>1 for spike train)
      harm_ratio: spectral distortion = sum_{n=2,3} |F(n q*)|^2 / |F(q*)|^2
                  (a pure sinusoid at q* gives ~0; a spike train pushes
                  power into harmonics)
    """
    v = interp_nans_1d(J_row)
    if np.ptp(v) < 1e-14:
        return np.nan, np.nan, np.nan
    vc = v - v.mean()
    sd = vc.std()
    skew = float(np.mean(vc ** 3) / (sd ** 3 + 1e-30))
    vmin = v.min()
    asym = float(v.max() / (abs(vmin) + 1e-30))
    W = np.hanning(v.size)
    F = np.abs(np.fft.rfft(vc * W))
    q = 2 * np.pi * np.fft.rfftfreq(v.size, dx)
    if not (np.isfinite(q_star) and q_star > 0) or q_star > q[-1]:
        return skew, asym, np.nan
    def pw(qq):
        iq = int(np.argmin(np.abs(q - qq)))
        return float(F[iq] ** 2)
    p1 = pw(q_star)
    ph = sum(pw(n * q_star) for n in (2, 3) if n * q_star <= q[-1])
    harm = ph / (p1 + 1e-30)
    return skew, asym, harm


# -----------------------------------------------------------------------
# Nematic winding census: contour topology without any phase reconstruction
# -----------------------------------------------------------------------

def nematic_winding(interp1, interp2, xc, yc, r=2.5, npts=144):
    """Winding index of the k-director around a circle centered at (xc, yc).

    The k field is a director (defined mod pi), so angle increments are
    wrapped into (-pi/2, pi/2] before summing; the total divided by 2*pi is
    the disclination index s. A defect-free knee gives s ~ 0; convex/concave
    disclinations of the PN string give s = +-1/2. No Hilbert transform, no
    unwrapping, no phase files: only k1, k2.
    """
    th = np.linspace(0.0, 2 * np.pi, npts + 1)
    xs = xc + r * np.cos(th)
    ys = yc + r * np.sin(th)
    v1 = interp1(np.column_stack([ys, xs]))
    v2 = interp2(np.column_stack([ys, xs]))
    if not (np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):
        return np.nan
    phi = np.arctan2(v2, v1)
    dphi = np.diff(phi)
    dphi = (dphi + np.pi / 2) % np.pi - np.pi / 2   # director wrap (mod pi)
    return float(np.sum(dphi) / (2 * np.pi))


def winding_census(k1f, k2f, x, y, y_core_val, J_row, dx, q_star,
                   rel_height=0.3, r_loop=None):
    """Detect |J| peaks on the core row and measure the winding at each.

    Returns (mean |s|, fraction of peaks with |s| >= 0.25, n_peaks).
    """
    from scipy.signal import find_peaks
    from scipy.interpolate import RegularGridInterpolator
    a = interp_nans_1d(np.abs(J_row))
    if np.ptp(a) < 1e-14:
        return np.nan, np.nan, 0
    spacing = 2 * np.pi / q_star if (np.isfinite(q_star) and q_star > 0) \
        else 10.0
    if r_loop is None:
        r_loop = min(2.5, 0.3 * spacing)
    dist = max(3, int(0.5 * spacing / dx))
    pk, _ = find_peaks(a, height=rel_height * a.max(), distance=dist)
    if pk.size == 0:
        return np.nan, np.nan, 0
    I1 = RegularGridInterpolator((y, x), k1f, bounds_error=False,
                                 fill_value=np.nan)
    I2 = RegularGridInterpolator((y, x), k2f, bounds_error=False,
                                 fill_value=np.nan)
    ss = []
    for ip in pk:
        s = nematic_winding(I1, I2, x[ip], y_core_val, r=r_loop)
        if np.isfinite(s):
            ss.append(s)
    if not ss:
        return np.nan, np.nan, int(pk.size)
    ss = np.array(ss)
    return float(np.mean(np.abs(ss))), \
        float(np.mean(np.abs(ss) >= 0.25)), int(pk.size)


def envelope_contrast(u_row, dx):
    """Contrast of the Hilbert envelope of the raw midline signal."""
    from scipy.signal import hilbert
    ur = interp_nans_1d(u_row)
    env = np.abs(hilbert(ur - ur.mean()))
    return contrast(env)


# -----------------------------------------------------------------------
# Spectra & growth rates
# -----------------------------------------------------------------------

def midline_spectrum(rows_xt, dx, detrend=True):
    """rows_xt: (Nt, Nx). Returns q (positive freqs) and |F|(Nt, Nq)."""
    Nt, Nx = rows_xt.shape
    W = np.hanning(Nx)[None, :]
    data = np.array([interp_nans_1d(r) for r in rows_xt])
    if detrend:
        data = data - np.nanmean(data, axis=1, keepdims=True)
    F = np.fft.rfft(data * W, axis=1)
    q = 2 * np.pi * np.fft.rfftfreq(Nx, dx)
    return q, np.abs(F)


def fit_growth_rates(t, S, fit_frac=(0.05, 0.5), floor=1e-12):
    """Fit log S(q,t) ~ rho(q) * t over an early window (by index fraction).
    S: (Nt, Nq). Returns rho (Nq,) and r2 (Nq,)."""
    Nt = S.shape[0]
    i0 = max(1, int(fit_frac[0] * Nt))
    i1 = max(i0 + 3, int(fit_frac[1] * Nt))
    tt = t[i0:i1]
    rho = np.full(S.shape[1], np.nan)
    r2 = np.full(S.shape[1], np.nan)
    for iq in range(S.shape[1]):
        y = np.log(np.maximum(S[i0:i1, iq], floor))
        if not np.all(np.isfinite(y)):
            continue
        A_ = np.vstack([tt, np.ones_like(tt)]).T
        coef, res, *_ = np.linalg.lstsq(A_, y, rcond=None)
        rho[iq] = coef[0]
        yhat = A_ @ coef
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2) + 1e-30
        r2[iq] = 1.0 - ss_res / ss_tot
    return rho, r2


def threshold_crossing_time(t, s, frac=0.5, increasing=None):
    """Time at which s crosses frac of its initial->final excursion."""
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    if ok.sum() < 3:
        return np.nan
    s0 = np.nanmean(s[ok][:3])
    s1 = np.nanmean(s[ok][-3:])
    if increasing is None:
        increasing = s1 > s0
    target = s0 + frac * (s1 - s0)
    for i in range(1, len(s)):
        if not (np.isfinite(s[i]) and np.isfinite(s[i - 1])):
            continue
        crossed = (s[i] >= target) if increasing else (s[i] <= target)
        if crossed:
            # linear interp
            if s[i] != s[i - 1]:
                w = (target - s[i - 1]) / (s[i] - s[i - 1])
            else:
                w = 0.0
            return t[i - 1] + w * (t[i] - t[i - 1])
    return np.nan


def lead_lag(t, a, b, max_lag_frames=30):
    """Lag (in time units) at which corr(a(t), b(t+lag)) is maximized.
    Positive lag => a leads b."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a = a[ok]; b = b[ok]; tt = t[ok]
    if a.size < 8:
        return np.nan, np.nan
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    dt = np.median(np.diff(tt))
    best_c, best_lag = -np.inf, 0
    L = min(max_lag_frames, a.size // 2)
    for lag in range(-L, L + 1):
        if lag >= 0:
            c = np.mean(a[: a.size - lag] * b[lag:]) if lag < a.size else np.nan
        else:
            c = np.mean(a[-lag:] * b[: b.size + lag])
        if np.isfinite(c) and c > best_c:
            best_c, best_lag = c, lag
    return best_lag * dt, best_c


# -----------------------------------------------------------------------
# Per-file processing
# -----------------------------------------------------------------------

def process_one(path, base_out, fit_frac_lo, fit_frac_hi, nuc_frac,
                domain_thresh=0.995, y_core=None, x_trim=0.0,
                rel_spectra=True):
    path = Path(path)
    stem = path.stem
    out = ensure_dir(base_out / stem)
    fig_dir = ensure_dir(out / "figs")
    data_dir = ensure_dir(out / "data")

    op = load_op_npz(path)
    x = op["x"]; y = op["y"]
    t = op["tt"] if "tt" in op and op["tt"].size else np.arange(op["A"].shape[-1])
    dx = float(x[1] - x[0]); dy = float(y[1] - y[0])
    mu = get_mu(op, stem)
    k1_ff = np.sqrt(max(1.0 - mu ** 2, 0.0)) if np.isfinite(mu) else np.nan
    q_star = 2.0 * k1_ff

    ramp = op["ramp"] if "ramp" in op else np.ones((len(y), len(x)))
    rn = (ramp - np.nanmin(ramp)) / (np.nanmax(ramp) - np.nanmin(ramp) + 1e-12)
    valid = rn >= domain_thresh

    j_core, how = pick_core_row(op, y, valid, y_core_override=y_core)
    n_valid_cols = int(valid[j_core, :].sum())
    print(f"  {stem}: mu={mu}, core row y={y[j_core]:.3f} "
          f"[{how}], valid cols={n_valid_cols}/{len(x)}, q*={q_star:.3f}")
    if n_valid_cols < 0.25 * len(x):
        raise RuntimeError(
            f"core row y={y[j_core]:.3f} has too few valid columns "
            f"({n_valid_cols}); set y_core manually in _Args")

    Nt = op["A"].shape[-1]
    cols = valid[j_core, :].copy()
    # trim x_trim physical units from each end of the valid interval, to
    # keep ramp-edge artifacts out of every diagnostic
    if x_trim and x_trim > 0:
        idx = np.where(cols)[0]
        if idx.size:
            n_trim = int(round(x_trim / dx))
            lo, hi = idx[0] + n_trim, idx[-1] - n_trim
            cols[:] = False
            if hi > lo:
                cols[lo:hi + 1] = True
    n_used = int(cols.sum())
    print(f"    using {n_used}/{len(x)} columns after x_trim={x_trim}")

    # ---- midline space-time arrays ----
    A_xt = np.array([np.where(cols, op["A"][j_core, :, it], np.nan)
                     for it in range(Nt)])
    u_xt = np.array([np.where(cols, op["u"][j_core, :, it], np.nan)
                     for it in range(Nt)])
    lam1_xt = np.array([op["lam1"][j_core, :, it] for it in range(Nt)])
    lam2_xt = np.array([op["lam2"][j_core, :, it] for it in range(Nt)])
    with np.errstate(divide="ignore", invalid="ignore"):
        chi_xt = np.where(np.abs(lam1_xt) > 1e-12, lam2_xt / lam1_xt, np.nan)
    chi_xt = np.where(cols[None, :], np.clip(chi_xt, 0.0, 1.5), np.nan)

    J_xt = np.full((Nt, len(x)), np.nan)
    phi_stair = np.full(Nt, np.nan)
    J_skew = np.full(Nt, np.nan)
    J_asym = np.full(Nt, np.nan)
    J_harm = np.full(Nt, np.nan)
    wind_mean = np.full(Nt, np.nan)
    wind_frac = np.full(Nt, np.nan)
    from scipy.signal import hilbert as _hilbert
    for it in range(Nt):
        f = op["k1"][..., it]; g = op["k2"][..., it]
        J = compute_J_frame(f, g, dx, dy, valid)
        J_xt[it] = np.where(cols, J[j_core, :], np.nan)
        # waveform shape of signed J along the core
        J_skew[it], J_asym[it], J_harm[it] = waveform_metrics(
            J_xt[it], dx, q_star)
        # nematic winding census at |J| peaks (director winding of k;
        # knee ~ 0, disclinations -> +-1/2). Uses k1_sym/k2_sym if saved
        # (director-lifted), else raw k1/k2 -- the mod-pi wrap makes the
        # census insensitive to the residual sign ambiguity either way.
        fs = op["k1_sym"][..., it] if "k1_sym" in op else f
        gs = op["k2_sym"][..., it] if "k2_sym" in op else g
        wind_mean[it], wind_frac[it], _ = winding_census(
            fs, gs, x, y, y[j_core], J_xt[it], dx, q_star)
        # staircase metric on the Hilbert phase of the raw midline signal
        # (arg(k) along the core is nearly constant -> net change ~ 0 and the
        # old arg(k)-based metric returned NaN; the Hilbert phase of u along
        # the core advances at ~k1 and is the direct analog of the phase
        # pipeline's midline profile)
        ur = interp_nans_1d(u_xt[it])
        if np.ptp(ur) > 1e-10:
            th = np.unwrap(np.angle(_hilbert(ur - ur.mean())))
            tv = np.sum(np.abs(np.diff(th)))
            net = np.abs(th[-1] - th[0])
            phi_stair[it] = tv / net - 1.0 if net > 1e-8 else np.nan

    np.savez_compressed(data_dir / "midline_xt.npz",
                        x=x, t=t, A_xt=A_xt, u_xt=u_xt, J_xt=J_xt,
                        chi_xt=chi_xt, mu=mu, y_core=y[j_core])

    # ---- heatmaps ----
    fields = {"A": A_xt, "|J|": np.abs(J_xt), "chi": chi_xt, "u": u_xt}
    fig, axs = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for ax, (name, F) in zip(axs, fields.items()):
        im = ax.pcolormesh(x, t, F, shading="auto", cmap="viridis")
        ax.set_title(f"{name}(x, t) midline")
        ax.set_xlabel("x")
        fig.colorbar(im, ax=ax, shrink=0.85)
    axs[0].set_ylabel("t")
    fig.suptitle(f"{stem}  (mu={mu}, y_core={y[j_core]:.2f})", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "midline_heatmaps.png", dpi=180)
    plt.close(fig)

    # ---- spectrograms + growth rates ----
    growth = {}
    fig, axs = plt.subplots(2, 4, figsize=(18, 7))
    for col, (name, F) in enumerate(fields.items()):
        # For A and chi, divide each frame by its spatial mean before the
        # FFT so that the global IC-relaxation trend (overall amplitude
        # settling) does not masquerade as decay of every Fourier mode;
        # what remains measures growth of *relative* modulation.
        if rel_spectra and name in ("A", "chi"):
            Fn = np.array([r / m if np.isfinite(m := np.nanmean(r)) and
                           abs(m) > 1e-12 else r for r in F])
        else:
            Fn = F
        q, S = midline_spectrum(Fn, dx)
        qmax_plot = min(4.0, q[-1])
        sel = q <= qmax_plot
        ax = axs[0, col]
        im = ax.pcolormesh(q[sel], t, np.log10(S[:, sel] + 1e-12),
                           shading="auto", cmap="magma")
        if np.isfinite(q_star):
            ax.axvline(q_star, color="cyan", ls="--", lw=1)
        ax.set_title(f"log10 |F_x {name}|")
        ax.set_xlabel("q"); fig.colorbar(im, ax=ax, shrink=0.85)

        rho, r2 = fit_growth_rates(t, S,
                                   fit_frac=(fit_frac_lo, fit_frac_hi))
        growth[name] = (q, rho, r2, S)
        ax2 = axs[1, col]
        ax2.plot(q[sel], rho[sel], lw=1)
        if np.isfinite(q_star):
            ax2.axvline(q_star, color="r", ls="--", lw=1, label="q*=2cos(a)")
            iq = int(np.argmin(np.abs(q - q_star)))
            ax2.plot(q[iq], rho[iq], "ro", ms=5)
        ax2.set_xlabel("q"); ax2.set_ylabel("rho(q)")
        ax2.set_title(f"growth rate ({name})")
        ax2.axhline(0, color="gray", lw=0.5)
        ax2.legend(fontsize=7)
    axs[0, 0].set_ylabel("t")
    fig.suptitle(f"{stem}  midline spectrograms and growth rates", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "spectrograms_growth.png", dpi=180)
    plt.close(fig)

    # ---- scalar battery ----
    def q_peak_series(name):
        q, _, _, S = growth[name]
        if not np.isfinite(q_star):
            return np.full(Nt, np.nan)
        iq = int(np.argmin(np.abs(q - q_star)))
        s = S[:, iq]
        return s / (s[0] + 1e-12)   # frozen-IC baseline normalization

    battery = {
        "A_min":        np.array([_nanmin(A_xt[i]) for i in range(Nt)]),
        "A_contrast":   np.array([contrast(A_xt[i]) for i in range(Nt)]),
        "chi_max":      np.array([_nanmax(chi_xt[i]) for i in range(Nt)]),
        "chi_contrast": np.array([contrast(chi_xt[i]) for i in range(Nt)]),
        "maxabsJ":      np.array([_nanmax(np.abs(J_xt[i])) for i in range(Nt)]),
        "intabsJ":      np.array([np.nansum(np.abs(J_xt[i])) * dx
                                  for i in range(Nt)]),
        "J_skew":       J_skew,
        "J_asym":       J_asym,
        "J_harm_ratio": J_harm,
        "winding_meanabs": wind_mean,
        "winding_frac_quant": wind_frac,
        "phase_staircase": phi_stair,
        "u_env_contrast": np.array([envelope_contrast(u_xt[i], dx)
                                    for i in range(Nt)]),
        "Sq*_A":   q_peak_series("A"),
        "Sq*_J":   q_peak_series("|J|"),
        "Sq*_chi": q_peak_series("chi"),
    }
    import csv
    with open(data_dir / "scalar_battery.csv", "w", newline="") as fcsv:
        wtr = csv.writer(fcsv)
        wtr.writerow(["t"] + list(battery.keys()))
        for i in range(Nt):
            wtr.writerow([t[i]] + [battery[k][i] for k in battery])

    nrows = int(np.ceil(len(battery) / 4))
    fig, axs = plt.subplots(nrows, 4, figsize=(16, 3 * nrows), sharex=True)
    axs = np.atleast_2d(axs)
    for ax in axs.flat[len(battery):]:
        ax.axis("off")
    for ax, (name, s) in zip(axs.flat, battery.items()):
        ax.plot(t, s, "o-", ms=2, lw=1)
        ax.set_title(name, fontsize=9)
    for ax in axs[-1]:
        ax.set_xlabel("t")
    fig.suptitle(f"{stem}  scalar battery", fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_dir / "scalar_battery.png", dpi=180)
    plt.close(fig)

    # ---- nucleation times (event ordering) ----
    t_nuc = {name: threshold_crossing_time(t, s, frac=nuc_frac)
             for name, s in battery.items()}

    # ---- lead-lag between key drivers ----
    pairs = [("A_min", "maxabsJ"), ("chi_max", "maxabsJ"),
             ("A_min", "Sq*_J"), ("phase_staircase", "A_min"),
             ("A_min", "winding_meanabs"), ("J_harm_ratio", "winding_meanabs")]
    ll = {}
    for a, b in pairs:
        lag, c = lead_lag(t, battery[a], battery[b])
        ll[f"{a}->{b}"] = (lag, c)

    # ---- growth rate at q* ----
    rho_at_qstar = {}
    for name in ("A", "|J|", "chi", "u"):
        q, rho, r2, _ = growth[name]
        if np.isfinite(q_star):
            iq = int(np.argmin(np.abs(q - q_star)))
            rho_at_qstar[name] = (float(rho[iq]), float(r2[iq]))
        else:
            rho_at_qstar[name] = (np.nan, np.nan)

    # dominant growing wavenumber (over q in (0.2, 1.8), avoid DC & stripe scale)
    q, rhoA, _, _ = growth["A"]
    selq = (q > 0.2) & (q < 1.8)
    q_fastest = float(q[selq][np.nanargmax(rhoA[selq])]) if selq.any() else np.nan

    summary = {
        "file": stem, "mu": mu, "q_star": q_star, "q_fastest_A": q_fastest,
        **{f"rho_qstar_{k}": v[0] for k, v in rho_at_qstar.items()},
        **{f"tnuc_{k}": v for k, v in t_nuc.items()},
        **{f"lag_{k}": v[0] for k, v in ll.items()},
    }
    with open(data_dir / "summary.json", "w") as fj:
        json.dump(summary, fj, indent=2, default=float)
    print(f"    q* = {q_star:.3f}, fastest-growing q(A) = {q_fastest:.3f}")
    print(f"    t_nuc ordering: " + ", ".join(
        f"{k}={v:.3g}" for k, v in sorted(t_nuc.items(),
                                          key=lambda kv: (np.inf if not
                                          np.isfinite(kv[1]) else kv[1]))))
    return summary


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def run_with_args(ns):
    """Core driver. `ns` is anything with the attributes:
    op_file, op_dir, pattern, out_dir, fit_frac_lo, fit_frac_hi, nuc_frac.
    Works with an argparse Namespace or a plain _Args class."""
    base_out = Path(ns.out_dir) if ns.out_dir else \
        _HERE / "results" / "midline_nucleation_battery"
    ensure_dir(base_out)

    if ns.op_file:
        files = [Path(ns.op_file)]
    else:
        files = sorted(Path(ns.op_dir or ".").glob(ns.pattern))
    if not files:
        raise SystemExit(
            f"No input files found "
            f"(op_file={ns.op_file}, op_dir={ns.op_dir}, pattern={ns.pattern})")

    summaries = []
    for f in files:
        try:
            summaries.append(process_one(
                f, base_out, ns.fit_frac_lo, ns.fit_frac_hi, ns.nuc_frac,
                y_core=getattr(ns, "y_core", None),
                x_trim=getattr(ns, "x_trim", 0.0),
                rel_spectra=getattr(ns, "rel_spectra", True)))
        except Exception as e:
            print(f"  FAILED {f.name}: {e}")

    # cross-run summary (per mu)
    if summaries:
        import csv
        keys = sorted({k for s in summaries for k in s})
        with open(base_out / "cross_run_summary.csv", "w", newline="") as fcsv:
            wtr = csv.DictWriter(fcsv, fieldnames=keys)
            wtr.writeheader()
            for s in sorted(summaries, key=lambda s: s.get("mu", 0)):
                wtr.writerow(s)
        print(f"Cross-run summary -> {base_out / 'cross_run_summary.csv'}")

        # rho(q*) and t_nuc vs mu quick-look
        mus = [s["mu"] for s in summaries]
        if len(set(mus)) > 1:
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            for name in ("A", "|J|", "chi"):
                axs[0].plot(mus, [s.get(f"rho_qstar_{name}", np.nan)
                                  for s in summaries], "o-", label=name)
            axs[0].set_xlabel("mu = sin(alpha)"); axs[0].set_ylabel("rho(q*)")
            axs[0].legend(); axs[0].set_title("growth rate at PN wavenumber")
            for key in ("tnuc_A_min", "tnuc_maxabsJ", "tnuc_chi_max",
                        "tnuc_phase_staircase"):
                axs[1].plot(mus, [s.get(key, np.nan) for s in summaries],
                            "o-", label=key.replace("tnuc_", ""))
            axs[1].set_xlabel("mu"); axs[1].set_ylabel("t_nuc")
            axs[1].legend(fontsize=7); axs[1].set_title("nucleation times")
            plt.tight_layout()
            plt.savefig(base_out / "mu_dependence.png", dpi=180)
            plt.close(fig)

    print("Done.")


def main(args=None):
    """CLI entry point (same behavior as before)."""
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--op_file", type=str, default=None,
                   help="Single OP .npz file.")
    p.add_argument("--op_dir", type=str, default=None,
                   help="Directory of OP .npz files.")
    p.add_argument("--pattern", type=str, default="*.npz",
                   help="Glob pattern inside op_dir.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Root output directory.")
    p.add_argument("--fit_frac_lo", type=float, default=0.05,
                   help="start of growth-rate fit window (fraction of frames)")
    p.add_argument("--fit_frac_hi", type=float, default=0.5,
                   help="end of growth-rate fit window (fraction of frames)")
    p.add_argument("--nuc_frac", type=float, default=0.5,
                   help="excursion fraction defining t_nuc")
    p.add_argument("--y_core", type=float, default=None,
                   help="physical y of the GB core (overrides auto-detect); "
                        "only sensible for single-file runs")
    p.add_argument("--x_trim", type=float, default=8.0,
                   help="physical units trimmed from each end of the valid "
                        "x-interval (ramp-edge exclusion)")
    p.add_argument("--rel_spectra", type=int, default=1,
                   help="1: normalize A/chi frames by spatial mean pre-FFT")
    ns = p.parse_args(args)
    run_with_args(ns)


# -----------------------------------------------------------------------
# Debug block (PyCharm: just click Run — edit _Args below)
# -----------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) == 1:

        class _Args:
            # --- input: set op_file to run ONE file, or leave it None and
            #     set op_dir (+ pattern) to run a whole directory ---
            op_file = None
            op_dir = ("/Users/edwardmcdugald/patterns/pipelines/data/"
                      "sh_pgb_zigzag/uhu/"
                      "mu_sweeps_full_Nx512_hp2_T25_NyF5_np18_Nsave125/"
                      "sig_pio2/raw")
            pattern = "*.npz"

            # --- output root; per-file subdirs are created inside ---
            out_dir = ("/Users/edwardmcdugald/patterns/experiments/"
                       "pgb_analysis/results/midline_nucleation_battery/"
                       "mu_sweeps_full_Nx512_hp2_T25_NyF5_np18_Nsave125/"
                       "sig_pio2")

            # --- analysis knobs ---
            fit_frac_lo = 0.05   # growth-rate fit window start (frame frac)
            fit_frac_hi = 0.5    # growth-rate fit window end (frame frac)
            nuc_frac = 0.5       # excursion fraction defining t_nuc
            y_core = None        # physical y of GB core; None = auto
                                 # (metadata gb_row_in_crop, else max-var row).
                                 # Only use with a single op_file, since each
                                 # mu has a different core y.
            x_trim = 8.0         # physical units trimmed from EACH end of
                                 # the valid x-interval (ramp-edge exclusion);
                                 # ~sigma + one stripe wavelength is a good
                                 # default. Set 0.0 to disable.
            rel_spectra = True   # divide A and chi frames by their spatial
                                 # mean before FFT, so global IC relaxation
                                 # doesn't read as decay of every mode

        run_with_args(_Args())
    else:
        main()