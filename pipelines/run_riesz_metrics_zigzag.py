# pipelines/run_riesz_metrics_zigzag.py
"""
Downstream metrics for Riesz OP time series (zigzag PGB), decoupled from
extraction. Reads the OP .npz written by run_riesz_zigzag.py and produces
time-series figures of the phase, the k fields, the defect index, J, and
curl, plus a SINDy-ready metrics .npz and .csv per file.

Design rules carried over from the extraction sessions:
  - Winding and current densities NEVER come componentwise from the mod-pi
    director fields (k1_direct, k2_direct carry a convention seam). Three
    labeled routes are computed instead and compared on data:
      (a) rho_d      doubled-angle winding density of the direct director
                     (seam-free by construction, core coverage)
      (b) oriented   J, curl, div from the direct k field AFTER a
                     deterministic orientation: sign chosen so that
                     k . grad(theta_smooth) > 0 wherever the reconstruction
                     exists; where it does not (the core band), component
                     signs are matched to the same frame's far-field
                     k_grad signs on the upper half (y rule first, x rule
                     where the director is nearly horizontal). No BFS
                     flood fill anywhere. The fraction of pixels decided
                     by the fallback is reported per frame.
      (c) theta      J and div from grad(theta_smooth), oriented by
                     construction, interior coverage only.
  - theta_smooth is display/metrics smoothing (nan_smooth, default width
    0.75). If the OP file predates the theta_smooth stack, it is computed
    here on the fly with the same function and masked to the raw support.
  - The source SH file (for pattern panels) is auto-located through
    riesz_meta_json.source_file; the runner degrades gracefully without it.

Outputs per OP file, under <output_dir>/{figures,metrics}:
  figures:
    <stem>_scalars_timeseries.png   phase residual, defect index, amplitude
                                    dip, J band maxima, far-field |k|
    <stem>_J_routes.png             rho_d vs oriented J vs theta J on the
                                    midline band, three frames + time series
    <stem>_maps_f{t}.png            field maps (A_mono, rho_d, J, curl, div)
                                    at frames {0, Nt//10, Nt-1}
    <stem>_profiles_heights.png     theta and theta_smooth at y = +0.25,
                                    +0.5, +1.0, frames 0/mid/last
    <stem>_midline_evolution.png    (x, t) heatmaps: A_mono, J, |k| near core
    <stem>_spacing_timeseries.png   charge spacings vs pi/k1 over time
  metrics:
    <stem>_metrics.npz              per-frame scalars, (Nx, Nt) midline
                                    profiles, spacing stats, x, tt, mu
    <stem>_metrics.csv              per-frame scalars only (SINDy-ready)

Usage:
    python run_riesz_metrics_zigzag.py <op_path>
    python run_riesz_metrics_zigzag.py --all --input_dir <dir with OP npz>
Defaults live in _Args at the bottom, mirrored by the CLI.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

from op_extract.t1_common import nan_smooth  # noqa: E402
from utils.paper_style import (PAPER_RC, panel_label, add_cbar,  # noqa: E402
                               sym_limits)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_op(op_path):
    d = np.load(op_path, allow_pickle=True)
    out = {k: d[k] for k in d.files if not k.endswith("_json")}
    out["riesz_meta"] = json.loads(str(d["riesz_meta_json"])) \
        if "riesz_meta_json" in d.files else {}
    out["charges"] = json.loads(str(d["charges_json"])) \
        if "charges_json" in d.files else None
    out["sh_meta"] = json.loads(str(d["sh_meta_json"])) \
        if "sh_meta_json" in d.files else {}
    return out


def locate_sh_file(op, op_path, sh_path_override=None):
    """Find the source SH npz for pattern panels. Tries the override, the
    absolute/relative source_file from riesz_meta, and the same basename
    next to the OP file's raw/ tree. Returns a Path or None (graceful)."""
    cands = []
    if sh_path_override:
        cands.append(Path(sh_path_override))
    src = op["riesz_meta"].get("source_file")
    if src:
        p = Path(src)
        cands.append(p)
        if not p.is_absolute():
            cands.append((Path(op_path).resolve().parent / p).resolve())
            cands.append((_HERE / p).resolve())
        cands.append(Path(op_path).resolve().parent / p.name)
        cands.append(Path(op_path).resolve().parents[1] / "raw" / p.name)
    for c in cands:
        try:
            if c.exists():
                return c
        except OSError:
            pass
    return None


def ensure_theta_smooth(op, width_default=0.75):
    """Fallback for OP files that predate the theta_smooth stack: compute
    it here with the same nan_smooth, masked to the raw support."""
    if "theta_smooth" in op:
        return op, float(op.get("theta_smooth_w", width_default)), False
    x, y = op["x"], op["y"]
    Lx, Ly = x[-1] - x[0], y[-1] - y[0]
    th = op["theta"]
    ts = np.empty_like(th, dtype=np.float32)
    for t in range(th.shape[-1]):
        s = nan_smooth(th[:, :, t].astype(float), width_default, Lx, Ly)
        s[~np.isfinite(th[:, :, t])] = np.nan
        ts[:, :, t] = s.astype(np.float32)
    op["theta_smooth"] = ts
    return op, width_default, True


# ---------------------------------------------------------------------------
# Derived fields, one frame at a time
# ---------------------------------------------------------------------------

def nan_grad(arr, dx, dy):
    """np.gradient with NaN propagation (NaN margin widens by one cell;
    that is the honest behavior near the hull boundary)."""
    gy, gx = np.gradient(arr, dy, dx)
    return gx, gy


def rho_d_from_director(k1d, k2d, dx, dy):
    """Doubled-angle winding density (seam-free). Mirrors riesz.rho_d but
    takes the direct director components instead of the raw orientation."""
    phi = np.arctan2(k2d, k1d)          # mod-pi seam irrelevant at 2*phi
    c2, s2 = np.cos(2 * phi), np.sin(2 * phi)
    c2x, c2y = nan_grad(c2, dx, dy)
    s2x, s2y = nan_grad(s2, dx, dy)
    return (c2x * s2y - c2y * s2x) / (4 * np.pi)


def orient_direct_k(k1d, k2d, th_s, dx, dy, core_row):
    """Deterministic orientation of the mod-pi direct k field.

    Primary rule: sign so that k . grad(theta_smooth) > 0 (align with the
    reconstruction's own gauge). Fallback where grad(theta_smooth) is
    missing or ambiguous: component signs matched to the frame's own
    far-field k_grad signs on the upper half; y rule first, x rule where
    the director is nearly horizontal. Returns (k1o, k2o, fallback_frac).
    """
    tx, ty = nan_grad(th_s, dx, dy)
    dot = k1d * tx + k2d * ty
    kmag2 = k1d ** 2 + k2d ** 2
    gmag2 = tx ** 2 + ty ** 2
    # confident where the projection is a decent fraction of both norms
    conf = np.isfinite(dot) & (np.abs(dot) >
                               0.2 * np.sqrt(kmag2 * np.maximum(gmag2, 1e-30)))

    # far-field reference signs from the reconstruction itself: upper rows
    ny = k1d.shape[0]
    r0 = min(core_row + int(0.25 * (ny - core_row)), ny - 2)
    sx = np.nanmedian(np.sign(tx[r0:, :])[np.isfinite(tx[r0:, :])
                                          & (np.abs(tx[r0:, :]) > 1e-6)])
    sy = np.nanmedian(np.sign(ty[r0:, :])[np.isfinite(ty[r0:, :])
                                          & (np.abs(ty[r0:, :]) > 1e-6)])
    sx = 1.0 if not np.isfinite(sx) or sx == 0 else sx
    sy = 1.0 if not np.isfinite(sy) or sy == 0 else sy

    sign = np.where(conf, np.sign(dot), 0.0)
    # fallback: y rule where the director has vertical content, x rule else.
    # theta is even about the core, so theta_y is ODD: the y-rule sign must
    # flip below the core row or the fallback disagrees with the mirrored
    # gauge and k flips sign across the hull seam (measured: spurious sharp
    # lines in J and curl exactly along the hull boundary before this fix).
    ny_ = k1d.shape[0]
    sgn_row = np.where(np.arange(ny_) >= core_row, 1.0, -1.0)[:, None]
    vert = np.abs(k2d) > 0.2 * np.sqrt(np.maximum(kmag2, 1e-30))
    fb = np.where(vert, np.sign(k2d) * sy * sgn_row, np.sign(k1d) * sx)
    need_fb = (sign == 0.0)
    sign = np.where(need_fb, fb, sign)
    sign = np.where(sign == 0.0, 1.0, sign)
    frac = float(np.mean(need_fb))
    return k1d * sign, k2d * sign, frac


def frame_fields(op, t, cfg):
    """All derived fields for frame t. Returns dict of 2d arrays and
    per-frame scalars."""
    x, y = op["x"], op["y"]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    core = int(op["core_row"])

    k1d = op["k1_direct"][:, :, t].astype(float)
    k2d = op["k2_direct"][:, :, t].astype(float)
    th_s = op["theta_smooth"][:, :, t].astype(float)

    rho = rho_d_from_director(k1d, k2d, dx, dy)

    k1o, k2o, fb = orient_direct_k(k1d, k2d, th_s, dx, dy, core)
    k1x, k1y = nan_grad(k1o, dx, dy)
    k2x, k2y = nan_grad(k2o, dx, dy)
    J_or = k1x * k2y - k1y * k2x
    curl_or = k2x - k1y
    div_or = k1x + k2y

    tx, ty = nan_grad(th_s, dx, dy)
    txx, txy = nan_grad(tx, dx, dy)
    _, tyy = nan_grad(ty, dx, dy)
    J_th = txx * tyy - txy ** 2
    div_th = txx + tyy

    return {"rho_d": rho, "J_or": J_or, "curl_or": curl_or,
            "div_or": div_or, "J_th": J_th, "div_th": div_th,
            "orient_fallback_frac": fb}


# ---------------------------------------------------------------------------
# Band / midline reductions
# ---------------------------------------------------------------------------

def band_rows(op, band_y):
    y = op["y"]
    core = int(op["core_row"])
    rows = np.where(np.abs(y - y[core]) <= band_y)[0]
    return rows


def x_interior(op, band_x_frac):
    Nx = len(op["x"])
    i = int(round(band_x_frac * Nx))
    return slice(i, Nx - i)


def band_stat(arr2d, rows, xs, stat):
    sub = arr2d[rows][:, xs]
    fin = np.isfinite(sub)
    if not fin.any():
        return np.nan
    if stat == "maxabs":
        return float(np.nanmax(np.abs(sub)))
    if stat == "intabs":
        return float(np.nansum(np.abs(sub)))
    if stat == "median":
        return float(np.nanmedian(sub))
    raise ValueError(stat)


def spacing_stats(charges_t, band_y, k1_theory, gap_thr_frac=0.5):
    """Mean/std of DISLOCATION spacings inside |y| <= band_y. Each physical
    dislocation registers as a compact multi-charge cluster (often a
    quadrupole with NET CHARGE ZERO on production data, which broke the
    earlier same-sign-family statistic: net-zero clusters dropped out of
    both sign families and produced spurious 2x and 8x plateaus during
    transients). Signs are therefore ignored entirely: charges are
    clustered in x with a gap threshold of gap_thr_frac * pi/k1 (default
    0.5, wide enough to merge the +/- disclination sub-clusters of one
    dislocation, measured 0.5*pi/k1 apart, and narrow enough not to chain
    adjacent dislocations at spacing pi/k1), and the statistic is the
    consecutive gap between cluster centers. n returned = cluster count =
    dislocation count in the band."""
    if not charges_t:
        return np.nan, np.nan, 0
    xs = sorted(c[0] for c in charges_t if abs(c[1]) <= band_y)
    if len(xs) < 2:
        return np.nan, np.nan, len(xs)
    gap_thr = gap_thr_frac * (np.pi / k1_theory) if k1_theory > 0 else np.inf
    centers = []
    cur = [xs[0]]
    for xv in xs[1:]:
        if xv - cur[-1] <= gap_thr:
            cur.append(xv)
        else:
            centers.append(float(np.mean(cur)))
            cur = [xv]
    centers.append(float(np.mean(cur)))
    if len(centers) < 2:
        return np.nan, np.nan, len(centers)
    gaps = np.diff(centers)
    return float(np.mean(gaps)), float(np.std(gaps)), len(centers)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_scalars(out, m, mu):
    t = m["tt"]
    fig, axs = plt.subplots(2, 3, figsize=(12.5, 6.2), constrained_layout=True)

    ax = axs[0, 0]
    ax.plot(t, m["shift_residual_int"], "C0-o", ms=3, label="interior")
    if np.isfinite(m["shift_residual"]).any():
        ax.plot(t, m["shift_residual"], "C1--", lw=0.9, label="full row")
    ax.set_ylabel(r"median $|\Delta\theta - \pi|$ (rad)")
    ax.set_title("shift-periodicity residual")
    ax.legend()
    panel_label(ax, "(a)")

    ax = axs[0, 1]
    ax.plot(t, m["n_dislocations"], "C0-o", ms=3, label="pairs")
    ax.plot(t, m["sum_abs"], "C2-s", ms=3, alpha=0.7, label=r"$\sum|q|$")
    ax.set_ylabel("count")
    ax.set_title("defect index")
    ax.legend()
    panel_label(ax, "(b)")

    ax = axs[0, 2]
    ax.plot(t, m["A_dip"], "C0-o", ms=3)
    ax.set_ylabel(r"$A_{\min}/A_{\rm far}$")
    ax.set_title("normalized amplitude dip (band)")
    panel_label(ax, "(c)")

    ax = axs[1, 0]
    ax.plot(t, m["J_or_maxabs"], "C0-o", ms=3, label=r"oriented $J$")
    ax.plot(t, m["J_th_maxabs"], "C1-s", ms=3, label=r"$\theta$-based $J$")
    ax.plot(t, m["rho_d_maxabs"], "C2-^", ms=3, label=r"$\rho_d$")
    ax.set_yscale("log")
    ax.set_ylabel(r"band $\max|\cdot|$")
    ax.set_xlabel("t")
    ax.set_title("defect current, three routes")
    ax.legend(fontsize=8)
    panel_label(ax, "(d)")

    ax = axs[1, 1]
    ax.plot(t, m["curl_or_maxabs"], "C0-o", ms=3, label=r"$|{\rm curl}\,k|$")
    ax.plot(t, np.abs(m["div_th_median"]), "C1-s", ms=3,
            label=r"$|{\rm med~div}\,k_\theta|$")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_title("curl and div in the band")
    ax.legend(fontsize=8)
    panel_label(ax, "(e)")

    ax = axs[1, 2]
    ax.plot(t, m["kmag_far_median"], "C0-o", ms=3)
    ax.set_xlabel("t")
    ax.set_ylabel(r"far-field median $|k|$")
    ax.set_title("wavenumber relaxation")
    panel_label(ax, "(f)")

    fig.suptitle(rf"$\mu={mu}$: per-frame scalar metrics", y=1.03)
    fig.savefig(out)
    plt.close(fig)


def fig_J_routes(out, op, F_by_frame, rows_band, xs, mu, m):
    """Midline-band x profiles of the three current routes at three frames,
    plus their band-max time series (repeated small for context)."""
    x = op["x"][xs]
    Nt = op["theta"].shape[-1]
    fr = sorted({0, Nt // 2, Nt - 1})
    fig, axs = plt.subplots(len(fr), 3, figsize=(12.5, 2.9 * len(fr)),
                            constrained_layout=True, sharex=True)
    axs = np.atleast_2d(axs)
    names = [("rho_d", r"$\rho_d$"), ("J_or", r"oriented $J$"),
             ("J_th", r"$\theta$-based $J$")]
    for r, t in enumerate(fr):
        F = F_by_frame[t]
        for c, (key, lab) in enumerate(names):
            prof = np.nanmean(np.abs(F[key][rows_band][:, xs]), axis=0)
            axs[r, c].plot(x, prof, "C0-", lw=1.0)
            if r == 0:
                axs[r, c].set_title(lab)
            if c == 0:
                axs[r, c].set_ylabel(f"frame {t}\nband mean $|\\cdot|$")
    for c in range(3):
        axs[-1, c].set_xlabel("$x$")
    fig.suptitle(rf"$\mu={mu}$: current-density routes along the midline band"
                 " (which $J$ is cleanest for SINDy)", y=1.02)
    fig.savefig(out)
    plt.close(fig)


def fig_maps(out, op, F, t, mu):
    x, y = op["x"], op["y"]
    ext = [x[0], x[-1], y[0], y[-1]]
    A = op["A_mono"][:, :, t]
    panels = [("A_mono", A, "magma", False),
              ("rho_d", F["rho_d"], "coolwarm", True),
              ("J_or", F["J_or"], "coolwarm", True),
              ("curl_or", F["curl_or"], "coolwarm", True),
              ("div_th", F["div_th"], "coolwarm", True)]
    fig, axs = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.0),
                            constrained_layout=True)
    for ax, (name, f, cm, signed) in zip(axs, panels):
        if signed:
            v = np.nanpercentile(np.abs(f[np.isfinite(f)]), 99.0) \
                if np.isfinite(f).any() else 1.0
            im = ax.imshow(np.ma.masked_invalid(f), extent=ext, cmap=cm,
                           vmin=-v, vmax=v, aspect="auto", origin="lower")
        else:
            im = ax.imshow(np.ma.masked_invalid(f), extent=ext, cmap=cm,
                           aspect="auto", origin="lower")
        add_cbar(fig, ax, im)
        ax.set_title(name)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(rf"$\mu={mu}$, frame {t}", y=1.05)
    fig.savefig(out)
    plt.close(fig)


def fig_profiles_heights(out, op, mu, heights=(0.25, 0.5, 1.0)):
    x, y = op["x"], op["y"]
    dy = float(y[1] - y[0])
    core = int(op["core_row"])
    Nt = op["theta"].shape[-1]
    fr = sorted({0, Nt // 2, Nt - 1})
    # distinct grid rows only: at coarse dx several requested heights can
    # round to the same row (measured at mu 0.95, dx 0.35: 0.25 and 0.5
    # both landed on core+1 and the figure showed a duplicate column)
    rows, seen = [], set()
    for h in heights:
        row = int(np.clip(core + max(1, int(round(h / dy))), 0, len(y) - 1))
        if row not in seen:
            rows.append(row)
            seen.add(row)
    fig, axs = plt.subplots(len(fr), len(rows),
                            figsize=(3.9 * len(rows), 2.7 * len(fr)),
                            constrained_layout=True, sharex=True)
    axs = np.atleast_2d(axs)
    if axs.shape != (len(fr), len(rows)):
        axs = axs.reshape(len(fr), len(rows))
    for r, t in enumerate(fr):
        for c, row in enumerate(rows):
            axs[r, c].plot(x, op["theta"][row, :, t], "C0-", lw=0.7,
                           alpha=0.6, label="raw")
            axs[r, c].plot(x, op["theta_smooth"][row, :, t], "C3-", lw=1.1,
                           label="smooth")
            if r == 0:
                axs[r, c].set_title(rf"$y=+{y[row]:.2f}$")
            if c == 0:
                axs[r, c].set_ylabel(f"frame {t}\n" + r"$\theta$")
            if r == 0 and c == 0:
                axs[r, c].legend(fontsize=7)
    for c in range(len(rows)):
        axs[-1, c].set_xlabel("$x$")
    fig.suptitle(rf"$\mu={mu}$: phase profiles approaching the core"
                 " (the core boundary layer, seen directly)", y=1.02)
    fig.savefig(out)
    plt.close(fig)


def fig_midline_evolution(out, op, mid, mu):
    x, tt = op["x"], op["tt"]
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.4), constrained_layout=True)
    entries = [("A_mono_core", "A_mono at core", "magma", False),
               ("J_or_nearcore", "oriented J near core", "coolwarm", True),
               ("kmag_core", "|k| at core", "viridis", False)]
    ext = [x[0], x[-1], tt[0], tt[-1]]
    for ax, (key, lab, cm, signed) in zip(axs, entries):
        Z = mid[key].T  # (Nt, Nx) for imshow with t on the vertical axis
        if signed:
            v = np.nanpercentile(np.abs(Z[np.isfinite(Z)]), 99.0) \
                if np.isfinite(Z).any() else 1.0
            im = ax.imshow(Z, extent=ext, aspect="auto", origin="lower",
                           cmap=cm, vmin=-v, vmax=v)
        else:
            im = ax.imshow(Z, extent=ext, aspect="auto", origin="lower",
                           cmap=cm)
        add_cbar(fig, ax, im)
        ax.set_title(lab)
        ax.set_xlabel("$x$")
    axs[0].set_ylabel("$t$")
    fig.suptitle(rf"$\mu={mu}$: midline evolution (nucleation observables)",
                 y=1.04)
    fig.savefig(out)
    plt.close(fig)


def fig_spacing(out, m, mu, d_theory):
    t = m["tt"]
    fig, ax = plt.subplots(figsize=(5.6, 3.3), constrained_layout=True)
    ax.axhline(d_theory, color="C3", ls="--", lw=1.2,
               label=rf"$\pi/k_1 = {d_theory:.3f}$")
    ax.errorbar(t, m["spacing_mean"], yerr=m["spacing_std"], fmt="o-",
                ms=3.5, color="C0", capsize=2, lw=0.9, label="measured")
    ax.set_xlabel("$t$")
    ax.set_ylabel("charge spacing")
    ax.set_title(rf"$\mu={mu}$: spacing vs the PN law over time")
    ax.legend()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_one(op_path, cfg):
    op_path = Path(op_path)
    print(f"  {op_path.name}")
    op = load_op(op_path)
    op, w_sm, computed = ensure_theta_smooth(op, cfg["theta_smooth_w"])
    if computed:
        print(f"    theta_smooth absent in file; computed on the fly "
              f"(w={w_sm})")

    x, y = op["x"], op["y"]
    tt = op["tt"] if op["tt"].size else np.arange(op["theta"].shape[-1],
                                                  dtype=float)
    mu = float(op["mu"])
    k1_th = float(np.sqrt(max(0.0, 1.0 - mu ** 2)))
    Nt = op["theta"].shape[-1]
    core = int(op["core_row"])
    dy = float(y[1] - y[0])

    rows_band = band_rows(op, cfg["band_y"])
    xs = x_interior(op, cfg["band_x_frac"])
    r_near = int(np.clip(core + max(1, int(round(0.5 / dy))), 0, len(y) - 1))
    # far-field rows for |k| relaxation: upper quarter of the upper half
    r_far0 = min(core + int(0.5 * (len(y) - core)), len(y) - 2)

    scalars = {k: np.full(Nt, np.nan) for k in
               ("rho_d_maxabs", "J_or_maxabs", "J_th_maxabs",
                "curl_or_maxabs", "div_or_median", "div_th_median",
                "J_or_intabs", "rho_d_intabs", "orient_fallback_frac",
                "kmag_far_median", "A_dip", "spacing_mean", "spacing_std")}
    n_ch = np.zeros(Nt, dtype=int)
    mid = {k: np.full((len(x), Nt), np.nan, dtype=np.float32) for k in
           ("A_mono_core", "J_or_nearcore", "kmag_core", "rho_d_nearcore",
            "theta_smooth_profile")}
    F_keep = {}
    fig_frames = sorted({0, Nt // 10, Nt - 1, Nt // 2})

    for t in range(Nt):
        F = frame_fields(op, t, cfg)
        scalars["rho_d_maxabs"][t] = band_stat(F["rho_d"], rows_band, xs,
                                               "maxabs")
        scalars["J_or_maxabs"][t] = band_stat(F["J_or"], rows_band, xs,
                                              "maxabs")
        scalars["J_th_maxabs"][t] = band_stat(F["J_th"], rows_band, xs,
                                              "maxabs")
        scalars["curl_or_maxabs"][t] = band_stat(F["curl_or"], rows_band, xs,
                                                 "maxabs")
        scalars["div_or_median"][t] = band_stat(F["div_or"], rows_band, xs,
                                                "median")
        scalars["div_th_median"][t] = band_stat(F["div_th"], rows_band, xs,
                                                "median")
        scalars["J_or_intabs"][t] = band_stat(F["J_or"], rows_band, xs,
                                              "intabs")
        scalars["rho_d_intabs"][t] = band_stat(F["rho_d"], rows_band, xs,
                                               "intabs")
        scalars["orient_fallback_frac"][t] = F["orient_fallback_frac"]
        scalars["kmag_far_median"][t] = float(np.nanmedian(
            op["kmag_direct"][r_far0:, xs, t]))
        af = float(op["A_far_median"][t]) if "A_far_median" in op else np.nan
        am = float(op["A_min_band"][t]) if "A_min_band" in op else np.nan
        scalars["A_dip"][t] = am / af if np.isfinite(am) and af else np.nan
        ch_t = op["charges"][t] if op["charges"] is not None else []
        sm_, ss_, n_ = spacing_stats(ch_t, cfg["band_y"], k1_th)
        scalars["spacing_mean"][t] = sm_
        scalars["spacing_std"][t] = ss_
        n_ch[t] = n_

        mid["A_mono_core"][:, t] = op["A_mono"][core, :, t]
        mid["J_or_nearcore"][:, t] = F["J_or"][r_near, :]
        mid["rho_d_nearcore"][:, t] = F["rho_d"][r_near, :]
        mid["kmag_core"][:, t] = op["kmag_direct"][core, :, t]
        prof_off = max(1, int(round(cfg["profile_y"] / dy)))
        mid["theta_smooth_profile"][:, t] = \
            op["theta_smooth"][core + prof_off, :, t]

        if t in fig_frames:
            F_keep[t] = F

    m = dict(scalars)
    m["tt"] = tt
    m["shift_residual"] = op.get("shift_residual", np.full(Nt, np.nan))
    m["shift_residual_int"] = op.get("shift_residual_int",
                                     m["shift_residual"])
    m["n_dislocations"] = op.get("n_dislocations", np.full(Nt, np.nan))
    m["sum_abs"] = op.get("sum_abs", np.full(Nt, np.nan))
    m["n_charges_band"] = n_ch

    stem = op_path.stem
    fig_dir = Path(cfg["output_dir"]) / "figures"
    met_dir = Path(cfg["output_dir"]) / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    with matplotlib.rc_context(PAPER_RC):
        if not cfg["no_plot"]:
            fig_scalars(fig_dir / f"{stem}_scalars_timeseries.png", m, mu)
            fig_J_routes(fig_dir / f"{stem}_J_routes.png", op, F_keep,
                         rows_band, xs, mu, m)
            for t in sorted({0, Nt // 10, Nt - 1}):
                fig_maps(fig_dir / f"{stem}_maps_f{t:03d}.png", op,
                         F_keep[t], t, mu)
            fig_profiles_heights(fig_dir / f"{stem}_profiles_heights.png",
                                 op, mu)
            fig_midline_evolution(fig_dir / f"{stem}_midline_evolution.png",
                                  op, mid, mu)
            fig_spacing(fig_dir / f"{stem}_spacing_timeseries.png", m, mu,
                        np.pi / k1_th if k1_th > 0 else np.nan)
            print(f"    figures -> {fig_dir}")

    # ---- metrics npz + csv ----
    save = {"x": x, "y_band": y[rows_band], "tt": tt, "mu": mu,
            "theta_smooth_w": w_sm,
            "profile_y": cfg["profile_y"], "band_y": cfg["band_y"],
            "band_x_frac": cfg["band_x_frac"]}
    for k, v in m.items():
        save[k] = v
    for k, v in mid.items():
        save[f"mid_{k}"] = v
    np.savez_compressed(met_dir / f"{stem}_metrics.npz", **save)

    scalar_keys = [k for k in m if np.ndim(m[k]) == 1 and k != "tt"]
    with open(met_dir / f"{stem}_metrics.csv", "w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["t"] + scalar_keys)
        for t in range(Nt):
            wcsv.writerow([tt[t]] + [m[k][t] for k in scalar_keys])
    print(f"    metrics -> {met_dir / (stem + '_metrics.npz')} (+ .csv)")
    print(f"    orientation fallback fraction: "
          f"f0 {scalars['orient_fallback_frac'][0]:.3f}  "
          f"fLast {scalars['orient_fallback_frac'][-1]:.3f}")


# ---------------------------------------------------------------------------
# Orchestration + CLI (mirrors _Args)
# ---------------------------------------------------------------------------

def run(cfg):
    if cfg.get("op_path"):
        process_one(cfg["op_path"], cfg)
    elif cfg.get("all") and cfg.get("input_dir"):
        files = sorted(Path(cfg["input_dir"]).glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz in {cfg['input_dir']}")
        print(f"Processing {len(files)} OP file(s)")
        for p in files:
            process_one(p, cfg)
    else:
        raise SystemExit("Provide an op_path or --all with --input_dir.")
    print("Done.")


def main():
    p = argparse.ArgumentParser(
        description="Downstream metrics for Riesz OP zigzag time series.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("op_path", nargs="?", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--input_dir", type=str, default=_Args.input_dir)
    p.add_argument("--output_dir", type=str, default=_Args.output_dir)
    p.add_argument("--sh_path", type=str, default=_Args.sh_path,
                   help="Optional source SH npz override for pattern "
                        "panels; auto-located from riesz_meta otherwise.")
    p.add_argument("--band_y", type=float, default=_Args.band_y)
    p.add_argument("--band_x_frac", type=float, default=_Args.band_x_frac)
    p.add_argument("--profile_y", type=float, default=_Args.profile_y)
    p.add_argument("--theta_smooth_w", type=float,
                   default=_Args.theta_smooth_w,
                   help="Used only when the OP file predates the "
                        "theta_smooth stack (computed on the fly).")
    p.add_argument("--no_plot", action="store_true")
    a = p.parse_args()
    cfg = {k: getattr(a, k) for k in ("op_path", "all", "input_dir",
                                      "output_dir", "sh_path", "band_y",
                                      "band_x_frac", "profile_y",
                                      "theta_smooth_w", "no_plot")}
    run(cfg)


class _Args:
    op_path = None
    all = False
    input_dir = "data/sh_pgb_zigzag/riesz_v8_aware/" \
                "mu_sweeps_full_Nx512_hp025_T3p125_NyF5_np18_Nsave125/raw"
    output_dir = "data/sh_pgb_zigzag/riesz_v8_aware/" \
                 "mu_sweeps_full_Nx512_hp025_T3p125_NyF5_np18_Nsave125/metrics"
    sh_path = None
    band_y = 2.0
    band_x_frac = 0.10
    profile_y = 1.0
    theta_smooth_w = 0.75
    no_plot = False


if __name__ == "__main__":
    if len(sys.argv) == 1:
        cfg = {k: getattr(_Args, k) for k in
               ("op_path", "all", "input_dir", "output_dir", "sh_path",
                "band_y", "band_x_frac", "profile_y", "theta_smooth_w",
                "no_plot")}
        cfg["all"] = True
        run(cfg)
    else:
        main()