# experiments/pgb_analysis/zigzag_multiview_analysis.py
"""
Zigzag multiview analysis for saved OP runs.

Adapts the older zigzag multiview script into the new codebase, with:
  - file-or-directory input
  - per-file subdirectories for figs/gifs/data/logs
  - oriented k-field, old-style J, and energy diagnostics
  - time-dependent core-line GIFs and 2D field GIFs
  - spacing-vs-time and energy-vs-time summaries

Uses old-style safe central derivatives via utils.kfield_calcs.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from utils.kfield_calcs import (
    orient_vector_field,
    orient_vector_field_v2,
    phi_jump_mask,
    safe_central_derivs,
    compute_J_old_style,
)
from utils.geometry import build_rectangular_ramp_smooth


# -----------------------------------------------------------------------
# I/O + ramps
# -----------------------------------------------------------------------

def load_run_npz(path):
    d = np.load(path, allow_pickle=True)
    return d


def _get_ramp(x, y, ramp_saved, xmargin, ymargin, tanhscale):
    if xmargin is not None and ymargin is not None and tanhscale is not None:
        return build_rectangular_ramp_smooth(x, y,
                                             xmargin=xmargin,
                                             ymargin=ymargin,
                                             tanhscale=tanhscale)
    if ramp_saved is not None:
        return ramp_saved
    raise ValueError("No ramp in OP file and no ramp params supplied.")


def _parse_mu(stem):
    try:
        part = stem.split("mu", 1)[1]
        for sep in ["_", "T", "t"]:
            if sep in part:
                return float(part.split(sep)[0])
        return float(part)
    except Exception:
        return np.nan


# -----------------------------------------------------------------------
# Masks and energy helpers
# -----------------------------------------------------------------------

def build_domain_masks_from_ramp(ramp, domain_thresh, core_thresh):
    """
    Build domain and inner/core masks from ramp only.
    """
    rmin, rmax = np.nanmin(ramp), np.nanmax(ramp)
    ramp_n = (ramp - rmin) / (rmax - rmin + 1e-12)
    domain_mask = ramp_n >= domain_thresh
    inner_mask = ramp_n >= core_thresh
    return domain_mask, inner_mask, ramp_n


def compute_energy_k(f, g, dx, dy, full_mask):
    """
    Old-style k-energy: E_k = (div k)^2 + (1 - |k|^2)^2.
    Uses safe central differences under full_mask.
    """
    fx, fy = safe_central_derivs(f, dx, dy, full_mask)
    gx, gy = safe_central_derivs(g, dx, dy, full_mask)
    div_k = fx + gy
    k2 = f ** 2 + g ** 2
    E = div_k ** 2 + (1.0 - k2) ** 2
    return E, div_k, k2


def second_derivs_safe(phi, dx, dy, mask_ok):
    """
    Safe 2nd derivatives for phase (used in E_theta).
    """
    ny, nx = phi.shape
    phixx = np.full_like(phi, np.nan, dtype=float)
    phiyy = np.full_like(phi, np.nan, dtype=float)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            if not mask_ok[j, i]:
                continue
            if mask_ok[j, i-1] and mask_ok[j, i+1]:
                phixx[j, i] = (phi[j, i+1] - 2*phi[j, i] + phi[j, i-1]) / (dx**2)
            if mask_ok[j-1, i] and mask_ok[j+1, i]:
                phiyy[j, i] = (phi[j+1, i] - 2*phi[j, i] + phi[j-1, i]) / (dy**2)
    return phixx, phiyy


def compute_energy_theta(phi, dx, dy, mask_ok, k2):
    """
    Old-style theta-energy using masked phase and |k|^2.
    """
    phix, phiy = safe_central_derivs(phi, dx, dy, mask_ok)
    phixx, phiyy = second_derivs_safe(phi, dx, dy, mask_ok)
    grad_phi2 = phix**2 + phiy**2
    lap_phi = phixx + phiyy
    E_theta = grad_phi2 + (1.0 - k2) ** 2 + lap_phi**2
    E_theta = np.where(mask_ok, E_theta, np.nan)
    return E_theta


def stats_on_mask(arr, mask):
    """
    Simple mean/std on a given mask, ignoring NaN.
    """
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    return float(np.nanmean(vals)), float(np.nanstd(vals))


def build_core_strip_mask(Y, y_core, half_width, domain_mask):
    """
    Horizontal strip centered at y_core with a given half-width.
    """
    strip = (np.abs(Y - y_core) <= half_width) & domain_mask
    return strip


def find_local_extrema_1d(x, y, min_sep, threshold_rel):
    """
    1D local maxima/minima for spacing along core line.
    """
    y = np.asarray(y)
    finite = np.isfinite(y)
    if not finite.any():
        return np.array([]), np.array([])

    # crude: use sign of discrete derivative
    dy = np.diff(y)
    sign = np.sign(dy)
    sign[~np.isfinite(sign)] = 0.0
    zero_cross = np.diff(sign)

    maxima = []
    minima = []
    max_val = np.nanmax(np.abs(y[finite]))
    thr = threshold_rel * max_val

    for i in range(1, len(y) - 1):
        if not finite[i]:
            continue
        if abs(y[i]) < thr:
            continue
        if zero_cross[i-1] < 0:  # + to -
            maxima.append(i)
        elif zero_cross[i-1] > 0:  # - to +
            minima.append(i)

    maxima = np.array(maxima, dtype=int)
    minima = np.array(minima, dtype=int)

    # min_sep pruning (in x)
    def prune(indices):
        if indices.size == 0:
            return indices
        keep = [indices[0]]
        for idx in indices[1:]:
            if abs(x[idx] - x[keep[-1]]) >= min_sep:
                keep.append(idx)
        return np.array(keep, dtype=int)

    maxima = prune(maxima)
    minima = prune(minima)
    return maxima, minima


def representative_vx_spacing(x_core, J_core_line, min_sep, threshold_rel):
    """
    Old-style representative V/X spacing from 1D core-line J.

    Returns spacing, x_V, x_X, idx_V, idx_X.
    """
    maxima, minima = find_local_extrema_1d(
        x_core, J_core_line, min_sep=min_sep, threshold_rel=threshold_rel
    )
    if maxima.size == 0 or minima.size == 0:
        return np.nan, np.nan, np.nan, None, None

    x_max = x_core[maxima]
    x_min = x_core[minima]

    # pick central maximum (closest to domain center)
    x_center = 0.5 * (x_core[0] + x_core[-1])
    i_cen = int(np.argmin(np.abs(x_max - x_center)))
    idx_V = maxima[i_cen]
    x_V = x_max[i_cen]

    # nearest minimum to the right
    mask_right = x_min > x_V
    if np.any(mask_right):
        xr = x_min[mask_right]
        idxs_right = np.where(mask_right)[0]
        j_min = idxs_right[int(np.argmin(xr - x_V))]
    else:
        j_min = int(np.argmin(np.abs(x_min - x_V)))
    idx_X = minima[j_min]
    x_X = x_min[j_min]

    spacing = abs(x_X - x_V)
    return spacing, x_V, x_X, idx_V, idx_X


def global_limits(arrays):
    """
    Global min/max over a list of arrays, ignoring NaN.
    """
    vals = []
    for a in arrays:
        a = np.asarray(a)
        vals.append(a[np.isfinite(a)].ravel())
    vals = np.concatenate(vals) if vals else np.array([])
    if vals.size == 0:
        return 0.0, 1.0
    return float(vals.min()), float(vals.max())


# -----------------------------------------------------------------------
# Orientation selection
# -----------------------------------------------------------------------

def choose_orientation_fields(op, it, domain_mask,
                              orientation, orient_method, pi_tol):
    """
    Pick f,g,phi,pi-jump mask, and full_mask at time index it
    according to orientation and orient_method.
    """
    # pick raw/sym/oriented fields
    if "k1_orig" in op and "k2_orig" in op:
        f_raw = op["k1_orig"][..., it]
        g_raw = op["k2_orig"][..., it]
    else:
        f_raw = op["k1"][..., it]
        g_raw = op["k2"][..., it]

    f_sym = op["k1_sym"][..., it] if "k1_sym" in op else f_raw
    g_sym = op["k2_sym"][..., it] if "k2_sym" in op else g_raw

    if orientation == "raw":
        f_use, g_use = f_raw, g_raw
    elif orientation == "sym":
        f_use, g_use = f_sym, g_sym
    elif orientation == "oriented":
        if orient_method == "bfs2":
            f_or, g_or = orient_vector_field_v2(f_raw, g_raw,
                                                mask=domain_mask,
                                                pi_tol=pi_tol)
        else:
            f_or, g_or = orient_vector_field(f_raw, g_raw, mask=domain_mask)
        f_use = np.asarray(f_or)
        g_use = np.asarray(g_or)
    else:
        raise ValueError(f"Unknown orientation '{orientation}'")

    phi = np.arctan2(g_use, f_use)
    pj = phi_jump_mask(phi, tol=pi_tol)
    full_mask = domain_mask & (~pj) & np.isfinite(f_use) & np.isfinite(g_use)
    return f_use, g_use, phi, pj, full_mask


# -----------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------

def pattern_with_J_peaks_plot(fig_dir, x, y, u_t, X, Y,
                              domain_mask, J_t, J_maxima, J_minima,
                              x_V, x_X, mu):
    """
    Final-time pattern + J peaks plot, adapted from legacy script.
    """
    extent = [x.min(), x.max(), y.min(), y.max()]
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        np.ma.masked_where(~domain_mask, u_t),
        extent=extent,
        origin="lower",
        cmap="gray",
    )
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    if J_maxima.size:
        ax.scatter(
            X[J_maxima[:, 0], J_maxima[:, 1]],
            Y[J_maxima[:, 0], J_maxima[:, 1]],
            c="red", s=15, label="J maxima",
        )
    if J_minima.size:
        ax.scatter(
            X[J_minima[:, 0], J_minima[:, 1]],
            Y[J_minima[:, 0], J_minima[:, 1]],
            c="blue", s=15, label="J minima",
        )

    if np.isfinite(x_V):
        ax.axvline(x_V, color="yellow", linewidth=1.5, label="V_core")
    if np.isfinite(x_X):
        ax.axvline(x_X, color="orange", linewidth=1.5, label="X_core")

    ax.set_title(f"Pattern with J peaks (mu={mu:.3f}, t=t_final)")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "pattern_with_J_peaks.png", dpi=200)
    plt.close(fig)


def make_line_gif(gif_path, x, t, arr_xt, ylabel, title, color="k", fps=10):
    """
    GIF of a 1D curve y(x) evolving in time.
    arr_xt has shape (Nx_core, Nt).
    """
    frames = []
    finite = np.isfinite(arr_xt)
    if finite.any():
        ymin = float(np.nanmin(arr_xt))
        ymax = float(np.nanmax(arr_xt))
        if np.isclose(ymin, ymax):
            pad = 1e-6 if ymin == 0 else 0.05 * abs(ymin)
            ymin -= pad
            ymax += pad
    else:
        ymin, ymax = -1.0, 1.0

    for it in range(arr_xt.shape[1]):
        fig, ax = plt.subplots(figsize=(6, 3))
        yvals = arr_xt[:, it]
        ax.plot(x, yvals, color=color, linewidth=1.5)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}  (t={t[it]:.3g})", fontsize=9)
        ax.axhline(0.0, color="0.5", linewidth=0.7, linestyle=":")
        plt.tight_layout()

        buf = gif_path.parent / f"__tmp_{gif_path.stem}_{it}.png"
        fig.savefig(buf, dpi=200)
        plt.close(fig)

        frames.append(imageio.imread(buf))
        buf.unlink(missing_ok=True)

    imageio.mimsave(gif_path, frames, fps=fps)


def make_field_gif(gif_path, X, Y, arr_t, domain_mask, extent,
                   vmin=None, vmax=None, cmap="gray", fps=10):
    """
    2D field GIF over time on domain_mask.
    """
    frames = []
    for it in range(arr_t.shape[-1]):
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(
            np.ma.masked_where(~domain_mask, arr_t[..., it]),
            extent=extent,
            origin="lower",
            cmap=cmap,
            vmin=vmin, vmax=vmax,
        )
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{gif_path.stem} (t={it})", fontsize=9)
        ax.set_aspect("equal")
        plt.tight_layout()
        buf = gif_path.parent / f"__tmp_{gif_path.stem}_{it}.png"
        plt.savefig(buf, dpi=200)
        plt.close(fig)
        frames.append(imageio.imread(buf))
        buf.unlink(missing_ok=True)
    imageio.mimsave(gif_path, frames, fps=fps)


# -----------------------------------------------------------------------
# Per-file driver
# -----------------------------------------------------------------------

def process_one(path, base_out,
                ramp_domain_thresh, ramp_core_thresh,
                xmargin, ymargin, tanhscale,
                orientation, orient_method, pi_tol,
                core_strip_frac,
                J_core_peak_min_sep, J_field_peak_min_distance, J_peak_min_rel,
                fps):
    path = Path(path)
    stem = path.stem
    print(f"  → {stem}")

    op = load_run_npz(path)
    x = op["x"]
    y = op["y"]
    u = op["u"]
    ramp_saved = op["ramp"] if "ramp" in op else None

    ramp = _get_ramp(x, y, ramp_saved, xmargin, ymargin, tanhscale)
    mu = _parse_mu(stem)

    X, Y = np.meshgrid(x, y)
    Ny, Nx, Nt = u.shape
    t = op["t"] if "t" in op else np.arange(Nt)
    dx = float(X[0, 1] - X[0, 0])
    dy = float(Y[1, 0] - Y[0, 0])
    extent = [x[0], x[-1], y[0], y[-1]]

    domain_mask, inner_mask, ramp_n = build_domain_masks_from_ramp(
        ramp, domain_thresh=ramp_domain_thresh, core_thresh=ramp_core_thresh
    )
    j_core = Ny // 2
    core_mask_x = ramp_n[j_core, :] >= ramp_core_thresh
    x_core = x[core_mask_x]
    y_core = 0.5 * (y[0] + y[-1])
    half_w = core_strip_frac * (y[-1] - y[0])
    strip_mask = build_core_strip_mask(Y, y_core, half_w, domain_mask)

    # per-file dirs
    out_dir = base_out / stem
    fig_dir = out_dir / "figures"
    gif_dir = out_dir / "gifs"
    data_dir = out_dir / "data"
    log_dir = out_dir / "logs"
    for d in [fig_dir, gif_dir, data_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # time-series storage
    u_core_xt = np.full((x_core.size, Nt), np.nan)
    J_core_xt = np.full((x_core.size, Nt), np.nan)
    Ek_core_xt = np.full((x_core.size, Nt), np.nan)

    spacing_t = np.full(Nt, np.nan)
    xV_t = np.full(Nt, np.nan)
    xX_t = np.full(Nt, np.nan)

    Ek_global_mean = np.full(Nt, np.nan)
    Ek_global_std = np.full(Nt, np.nan)
    Ek_strip_mean = np.full(Nt, np.nan)
    Ek_strip_std = np.full(Nt, np.nan)

    Etheta_global_mean = np.full(Nt, np.nan)
    Etheta_global_std = np.full(Nt, np.nan)
    Etheta_strip_mean = np.full(Nt, np.nan)
    Etheta_strip_std = np.full(Nt, np.nan)

    Ek_global_int = np.full(Nt, np.nan)
    Ek_strip_int = np.full(Nt, np.nan)
    Etheta_global_int = np.full(Nt, np.nan)
    Etheta_strip_int = np.full(Nt, np.nan)

    # main time loop
    for it in range(Nt):
        f_use, g_use, phi_t, pj_t, full_mask_t = choose_orientation_fields(
            op, it, domain_mask, orientation, orient_method, pi_tol
        )

        # old-style J using helper; orientation already applied
        J_t, _, _, _, _, _ = compute_J_old_style(
            f_use, g_use, x, y, domain_mask, pi_tol=pi_tol, orient=False
        )
        J_t = np.where(full_mask_t, J_t, np.nan)

        # k-energy
        Ek_t, div_k_t, k2_t = compute_energy_k(f_use, g_use, dx, dy, full_mask_t)

        # theta-energy if phase-unwrapped present
        Etheta_t = np.full_like(Ek_t, np.nan)
        if "phase_grid_symmetric_unwrapped" in op:
            phi_sym_unwrapped_t = op["phase_grid_symmetric_unwrapped"][..., it]
            mask_phase_ok = full_mask_t & np.isfinite(phi_sym_unwrapped_t)
            Etheta_t = compute_energy_theta(
                phi_sym_unwrapped_t, dx, dy, mask_phase_ok, k2_t
            )

        # core-line slices
        u_core_line = u[j_core, :, it]
        J_core_line_full = J_t[j_core, :]
        Ek_core_line_full = Ek_t[j_core, :]

        u_core_xt[:, it] = u_core_line[core_mask_x]
        J_core_xt[:, it] = J_core_line_full[core_mask_x]
        Ek_core_xt[:, it] = Ek_core_line_full[core_mask_x]

        # spacing from core-line J
        spacing, x_V, x_X, idx_V, idx_X = representative_vx_spacing(
            x_core, J_core_xt[:, it],
            min_sep=J_core_peak_min_sep,
            threshold_rel=J_peak_min_rel,
        )
        spacing_t[it] = spacing
        xV_t[it] = x_V
        xX_t[it] = x_X

        # energy stats
        Ek_global_mean[it], Ek_global_std[it] = stats_on_mask(Ek_t, full_mask_t)
        Ek_strip_mean[it], Ek_strip_std[it] = stats_on_mask(Ek_t, strip_mask & full_mask_t)
        Ek_global_int[it] = np.nansum(np.where(full_mask_t, Ek_t, np.nan)) * dx * dy
        Ek_strip_int[it] = np.nansum(np.where(strip_mask & full_mask_t, Ek_t, np.nan)) * dx * dy

        if np.isfinite(Etheta_t).any():
            Etheta_global_mean[it], Etheta_global_std[it] = stats_on_mask(Etheta_t, full_mask_t)
            Etheta_strip_mean[it], Etheta_strip_std[it] = stats_on_mask(Etheta_t, strip_mask & full_mask_t)
            Etheta_global_int[it] = np.nansum(np.where(full_mask_t, Etheta_t, np.nan)) * dx * dy
            Etheta_strip_int[it] = np.nansum(np.where(strip_mask & full_mask_t, Etheta_t, np.nan)) * dx * dy

    # final-time diagnostics
    t0 = Nt - 1
    f0, g0, phi0, pj0, full_mask0 = choose_orientation_fields(
        op, t0, domain_mask, orientation, orient_method, pi_tol
    )
    J0, _, _, _, _, _ = compute_J_old_style(
        f0, g0, x, y, domain_mask, pi_tol=pi_tol, orient=False
    )
    J0 = np.where(full_mask0, J0, np.nan)

    # 2D J peaks for final-time pattern plot
    J_valid = np.where(full_mask0, J0, -np.inf)
    from utils.kfield_calcs import get_local_peaks

    peak_min_distance_px = max(1, int(J_field_peak_min_distance))

    maxima_indices, minima_indices = get_local_peaks(
        J_valid,
        mask=full_mask0,
        min_distance=peak_min_distance_px,
        threshold_rel=J_peak_min_rel,
    )

    pattern_with_J_peaks_plot(
        fig_dir,
        x, y,
        u[..., t0],
        X, Y,
        domain_mask,
        J0,
        maxima_indices, minima_indices,
        xV_t[t0], xX_t[t0],
        mu,
    )

    # GIFs: core-line u, J, E_k
    vmin_u, vmax_u = global_limits([u_core_xt])
    vmin_J, vmax_J = global_limits([J_core_xt])
    vmin_Ek, vmax_Ek = global_limits([Ek_core_xt])

    make_line_gif(gif_dir / "u_core_line.gif", x_core, t, u_core_xt,
                  ylabel="u", title=f"u along core (mu={mu:.3f})",
                  color="k", fps=fps)
    make_line_gif(gif_dir / "J_core_line.gif", x_core, t, J_core_xt,
                  ylabel="J", title=f"J along core (mu={mu:.3f})",
                  color="tab:blue", fps=fps)
    make_line_gif(gif_dir / "Ek_core_line.gif", x_core, t, Ek_core_xt,
                  ylabel="E_k", title=f"E_k along core (mu={mu:.3f})",
                  color="tab:green", fps=fps)

    # CSV summary
    ts_csv = data_dir / "spacing_and_energy_time_series.csv"
    header = (
        "t,spacing,xV,xX,"
        "Ek_global_mean,Ek_global_std,Ek_strip_mean,Ek_strip_std,"
        "Etheta_global_mean,Etheta_global_std,Etheta_strip_mean,Etheta_strip_std,"
        "Ek_global_int,Ek_strip_int,Etheta_global_int,Etheta_strip_int"
    )
    rows = np.column_stack([
        t,
        spacing_t,
        xV_t,
        xX_t,
        Ek_global_mean,
        Ek_global_std,
        Ek_strip_mean,
        Ek_strip_std,
        Etheta_global_mean,
        Etheta_global_std,
        Etheta_strip_mean,
        Etheta_strip_std,
        Ek_global_int,
        Ek_strip_int,
        Etheta_global_int,
        Etheta_strip_int,
    ])
    np.savetxt(ts_csv, rows, delimiter=",", header=header, comments="")

    # simple spacing plot
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t, spacing_t, "o-", color="steelblue")
    ax.set_xlabel("t")
    ax.set_ylabel("spacing (V→X, core line)")
    ax.set_title(f"V–X spacing vs t (mu={mu:.3f})", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "spacing_vs_time.png", dpi=200)
    plt.close(fig)
    print(f"    spacing(t_final)={spacing_t[t0]:.4f}  mu={mu:.3f}")

    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)

    axs[0, 0].plot(t, Ek_global_mean, "o-", label="global mean")
    axs[0, 0].plot(t, Ek_strip_mean, "s--", label="strip mean")
    axs[0, 0].set_ylabel("E_k mean")
    axs[0, 0].set_title("E_k means", fontsize=9)
    axs[0, 0].legend(fontsize=8)

    axs[0, 1].plot(t, Ek_global_std, "o-", label="global std")
    axs[0, 1].plot(t, Ek_strip_std, "s--", label="strip std")
    axs[0, 1].set_ylabel("E_k std")
    axs[0, 1].set_title("E_k stds", fontsize=9)
    axs[0, 1].legend(fontsize=8)

    axs[1, 0].plot(t, Etheta_global_mean, "o-", label="global mean")
    axs[1, 0].plot(t, Etheta_strip_mean, "s--", label="strip mean")
    axs[1, 0].set_xlabel("t")
    axs[1, 0].set_ylabel("E_theta mean")
    axs[1, 0].set_title("E_theta means", fontsize=9)
    axs[1, 0].legend(fontsize=8)

    axs[1, 1].plot(t, Etheta_global_std, "o-", label="global std")
    axs[1, 1].plot(t, Etheta_strip_std, "s--", label="strip std")
    axs[1, 1].set_xlabel("t")
    axs[1, 1].set_ylabel("E_theta std")
    axs[1, 1].set_title("E_theta stds", fontsize=9)
    axs[1, 1].legend(fontsize=8)

    fig.suptitle(f"Energy summary stats vs time (mu={mu:.3f})", fontsize=10)
    plt.tight_layout()
    fig.savefig(fig_dir / "energy_summary_stats_vs_time.png", dpi=200)
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    axs[0].plot(t, Ek_global_int, "o-", label="global ∫E_k")
    axs[0].plot(t, Ek_strip_int, "s--", label="strip ∫E_k")
    axs[0].set_ylabel("integrated E_k")
    axs[0].set_title("Integrated E_k vs time", fontsize=9)
    axs[0].legend(fontsize=8)

    axs[1].plot(t, Etheta_global_int, "o-", label="global ∫E_theta")
    axs[1].plot(t, Etheta_strip_int, "s--", label="strip ∫E_theta")
    axs[1].set_xlabel("t")
    axs[1].set_ylabel("integrated E_theta")
    axs[1].set_title("Integrated E_theta vs time", fontsize=9)
    axs[1].legend(fontsize=8)

    fig.suptitle(f"Integrated energy vs time (mu={mu:.3f})", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_dir / "integrated_energy_vs_time.png", dpi=200)
    plt.close(fig)

    return mu, spacing_t[t0]


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--op_file",  type=str, default=None,
                        help="Single OP .npz file.")
    parser.add_argument("--op_dir",   type=str, default=None,
                        help="Directory of OP .npz files.")
    parser.add_argument("--pattern",  type=str, default="*.npz",
                        help="Glob pattern inside op_dir.")
    parser.add_argument("--out_dir",  type=str, default=None,
                        help="Root output directory. "
                             "Default: experiments/pgb_analysis/results/zigzag_multiview_analysis/")
    parser.add_argument("--ramp_domain_thresh", type=float, default=0.995)
    parser.add_argument("--ramp_core_thresh",   type=float, default=0.999)
    parser.add_argument("--xmargin",        type=float, default=None)
    parser.add_argument("--ymargin",        type=float, default=None)
    parser.add_argument("--tanhscale",      type=float, default=None)
    parser.add_argument("--orientation",    type=str,   default="raw",
                        choices=["raw", "sym", "oriented"])
    parser.add_argument("--orient_method",  type=str,   default="bfs",
                        choices=["bfs", "bfs2"])
    parser.add_argument("--pi_tol",         type=float, default=np.pi/10)
    parser.add_argument("--core_strip_frac", type=float, default=0.10,
                        help="Half-width of midline core strip (fraction of Ly).")
    parser.add_argument("--J_core_peak_min_sep", type=float, default=3.0,
                        help="Min separation in x for 1D core peaks (physical units).")
    parser.add_argument("--J_field_peak_min_distance", type=int, default=5,
                        help="Min pixel separation for 2D peak finding on J field.")
    parser.add_argument("--J_peak_min_rel", type=float, default=0.10,
                        help="Relative threshold for core-line peaks.")
    parser.add_argument("--fps", type=int, default=10,
                        help="FPS for GIFs.")
    parser.add_argument("--mu_min", type=float, default=None,
                        help="Only process files with mu > this value.")

    ns = parser.parse_args(args)

    if ns.out_dir is not None:
        base_out = Path(ns.out_dir)
    else:
        base_out = _HERE / "results" / "zigzag_multiview_analysis"
    base_out.mkdir(parents=True, exist_ok=True)

    ramp_kw = dict(xmargin=ns.xmargin, ymargin=ns.ymargin, tanhscale=ns.tanhscale)
    common = dict(
        ramp_domain_thresh=ns.ramp_domain_thresh,
        ramp_core_thresh=ns.ramp_core_thresh,
        orientation=ns.orientation,
        orient_method=ns.orient_method,
        pi_tol=ns.pi_tol,
        core_strip_frac=ns.core_strip_frac,
        J_core_peak_min_sep=ns.J_core_peak_min_sep,
        J_field_peak_min_distance=ns.J_field_peak_min_distance,
        J_peak_min_rel=ns.J_peak_min_rel,
        fps=ns.fps,
        **ramp_kw,
    )

    results = []

    if ns.op_file:
        r = process_one(Path(ns.op_file), base_out, **common)
        results.append(r)
    else:
        op_dir = Path(ns.op_dir or ".")
        files = sorted(op_dir.glob(ns.pattern))
        if not files:
            raise SystemExit(f"No files matching '{ns.pattern}' in {op_dir}")
        if ns.mu_min is not None:
            files = [f for f in files if _parse_mu(f.stem) > ns.mu_min]
        print(f"Processing {len(files)} file(s) from {op_dir}")
        for f in files:
            r = process_one(f, base_out, **common)
            results.append(r)

    # optional summary over files: spacing vs mu
    valid = [(m, s) for m, s in results if np.isfinite(m)]
    if len(valid) > 1:
        valid.sort(key=lambda t: t[0])
        mus = np.array([r[0] for r in valid])
        spacings = np.array([r[1] for r in valid])

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(mus, spacings, "o-", color="steelblue")
        ax.set_xlabel(r"$\mu$")
        ax.set_ylabel("V–X spacing at t_final")
        ax.set_title("V–X spacing vs μ (zigzag multiview)", fontsize=9)
        plt.tight_layout()
        fig.savefig(base_out / "spacing_vs_mu.png", dpi=200)
        plt.close(fig)
        print(f"  spacing curve → {base_out / 'spacing_vs_mu.png'}")

    print("Done.")


# -----------------------------------------------------------------------
# Debug block
# -----------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) == 1:

        class _Args:
            op_file        = None
            #op_dir         = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/uhu/0711_np16_nx1024/sig_pio2/raw"
            op_dir         = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/uhu/mu_sweeps_full_Nx512_hp025_T3p125_NyF5_np18_Nsave125/sig_pio2/raw"
            pattern        = "*.npz"
            #out_dir        = "/Users/edwardmcdugald/patterns/experiments/pgb_analysis/results/zigzag_multiview_analysis/uhu/0711_np16_nx1024/sig_pio2"
            out_dir        = "/Users/edwardmcdugald/patterns/experiments/pgb_analysis/results/zigzag_multiview_analysis/mu_sweeps_full_Nx512_hp025_T3p125_NyF5_np18_Nsave125/sig_pio2"
            ramp_domain_thresh = 0.995
            ramp_core_thresh   = 0.999
            xmargin        = 0.025
            ymargin        = 0.025
            tanhscale      = 120.0
            orientation    = "raw"
            orient_method  = "bfs"
            pi_tol         = np.pi / 10
            core_strip_frac = 0.05
            J_core_peak_min_sep = 5.0
            J_field_peak_min_distance = 5
            J_peak_min_rel = 0.05
            fps             = 10
            mu_min          = 0.63

        a = _Args()
        main([
            "--op_dir",         a.op_dir,
            "--out_dir",        a.out_dir,
            "--pattern",        a.pattern,
            "--ramp_domain_thresh", str(a.ramp_domain_thresh),
            "--ramp_core_thresh",   str(a.ramp_core_thresh),
            "--pi_tol",         str(a.pi_tol),
            "--orientation",    a.orientation,
            "--orient_method",  a.orient_method,
            "--core_strip_frac", str(a.core_strip_frac),
            "--J_core_peak_min_sep", str(a.J_core_peak_min_sep),
            "--J_field_peak_min_distance", str(a.J_field_peak_min_distance),
            "--J_peak_min_rel", str(a.J_peak_min_rel),
            "--J_peak_min_rel",  str(a.J_peak_min_rel),
            "--fps",             str(a.fps),
            *(["--xmargin",    str(a.xmargin),
               "--ymargin",    str(a.ymargin),
               "--tanhscale",  str(a.tanhscale)]
              if a.xmargin is not None else []),
            *(["--mu_min", str(a.mu_min)] if a.mu_min is not None else []),
        ])
    else:
        main()