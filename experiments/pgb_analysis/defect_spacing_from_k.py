# experiments/pgb_analysis/defect_spacing.py
"""
Defect spacing analysis using J-density maxima/minima near the PGB midline.

No integrals. Uses raw J = det(∇k) from the oriented k-field.

Defect detection:
  - find maxima and minima of J restricted to a horizontal core band
    around the y-midline
  - select the maximum closest to the domain center in x  →  "central max"
  - max–max spacing : distance between central max and its nearest max neighbour
  - max–min spacing : distance between central max and its nearest minimum

Per-file outputs (in results/defect_spacing/<stem>/):
  - pattern contour plot
  - pattern (copper) + defect markers
  - J field with markers
  - J midline profile + projected defect x-positions
  - k quiver (oriented)

Summary output (in results/defect_spacing/):
  - defect_spacing_vs_mu.png  (both metrics + theoretical curve)

Usage
-----
# IDE: press Run — uses debug block below

python defect_spacing.py --op_dir /path/to/raw
python defect_spacing.py --op_dir /path/to/raw --out_dir /path/to/output
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from utils.kfield_calcs import (
    orient_vector_field,
    orient_vector_field_v2,
    phi_jump_mask,
    kfield_diagnostics,
)
from utils.geometry import build_rectangular_ramp_smooth


# -----------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------

def load_op_npz(path):
    d = np.load(path, allow_pickle=True)
    x    = d["x"]
    y    = d["y"]
    u    = d["u"][..., -1]  if d["u"].ndim  == 3 else d["u"]
    k1   = d["k1"][..., -1] if d["k1"].ndim == 3 else d["k1"]
    k2   = d["k2"][..., -1] if d["k2"].ndim == 3 else d["k2"]
    ramp = d["ramp"] if "ramp" in d else None
    return x, y, u, k1, k2, ramp


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
# J-density peak detection
# -----------------------------------------------------------------------

def detect_J_peaks(J_field, X, Y, mask_ok,
                   core_band_frac, min_distance, threshold_rel):
    """
    Detect maxima and minima of J within the core band.

    Returns
    -------
    maxima : (N,2) int array of (row, col) indices
    minima : (M,2) int array of (row, col) indices
    band_mask : bool array showing the core band region used
    """
    y_vals = Y[:, 0]
    y_mid  = 0.5 * (y_vals[0] + y_vals[-1])
    half_w = core_band_frac * (y_vals[-1] - y_vals[0])
    band_mask = (np.abs(Y - y_mid) <= half_w) & mask_ok & np.isfinite(J_field)

    # maxima
    field_max = np.where(band_mask, J_field, -np.inf)
    maxima = peak_local_max(field_max,
                            min_distance=min_distance,
                            threshold_rel=threshold_rel,
                            exclude_border=False)

    # minima  (flip sign)
    field_min = np.where(band_mask, -J_field, -np.inf)
    minima = peak_local_max(field_min,
                            min_distance=min_distance,
                            threshold_rel=threshold_rel,
                            exclude_border=False)

    return maxima, minima, band_mask


def _central_pair(indices, X, Y, x_center):
    """
    From a set of peak indices, return the one closest to x_center
    and its nearest neighbour among the same set (by x-distance).

    Returns
    -------
    central_idx  : (row, col) of the central peak, or None
    neighbour_idx: (row, col) of the nearest neighbour, or None
    x_central    : physical x of central peak
    x_neighbour  : physical x of neighbour peak
    """
    if indices.size == 0:
        return None, None, np.nan, np.nan

    xs = X[indices[:, 0], indices[:, 1]]

    # central: closest to x_center
    i_cen = int(np.argmin(np.abs(xs - x_center)))
    central_idx = tuple(indices[i_cen])
    x_cen = xs[i_cen]

    if len(xs) < 2:
        return central_idx, None, x_cen, np.nan

    # nearest neighbour in x (excluding itself)
    dists = np.abs(xs - x_cen)
    dists[i_cen] = np.inf
    i_nn = int(np.argmin(dists))
    neighbour_idx = tuple(indices[i_nn])
    x_nn = xs[i_nn]

    return central_idx, neighbour_idx, x_cen, x_nn


def compute_spacings(maxima, minima, X, Y, x_center):
    """
    max–max spacing : |x_central_max − x_nearest_max_neighbour|
    max–min spacing : |x_central_max − x_nearest_minimum|

    Returns (spacing_max_max, spacing_max_min,
             central_max_idx, nn_max_idx, nn_min_idx)
    """
    cen_max, nn_max, x_cen_max, x_nn_max = _central_pair(maxima, X, Y, x_center)
    spacing_mm = abs(x_cen_max - x_nn_max) if np.isfinite(x_nn_max) else np.nan

    # nearest minimum to central max (by x distance)
    nn_min_idx = None
    spacing_pm = np.nan
    if cen_max is not None and minima.size > 0:
        xs_min = X[minima[:, 0], minima[:, 1]]
        i_nm   = int(np.argmin(np.abs(xs_min - x_cen_max)))
        nn_min_idx = tuple(minima[i_nm])
        spacing_pm = abs(xs_min[i_nm] - x_cen_max)

    return spacing_mm, spacing_pm, cen_max, nn_max, nn_min_idx


# -----------------------------------------------------------------------
# Per-file plots
# -----------------------------------------------------------------------

def _imshow(ax, data, mask, extent, cmap="viridis", **kw):
    im = ax.imshow(np.ma.masked_where(~mask, data),
                   extent=extent, origin="lower", cmap=cmap, **kw)
    ax.set_aspect("equal")   # removed set_xticks/set_yticks lines
    return im


def _scatter_markers(ax, indices, X, Y, color, marker, label, s=60):
    if indices is not None and not (isinstance(indices, np.ndarray) and indices.size == 0):
        if isinstance(indices, tuple):
            indices_list = [indices]
        else:
            indices_list = [tuple(r) for r in indices]
        xs = [X[r] for r in indices_list]
        ys = [Y[r] for r in indices_list]
        ax.scatter(xs, ys, c=color, marker=marker, s=s,
                   linewidths=1.5, zorder=5, label=label)


def save_per_file_figs(stem, out_dir,
                       x, y, u, X, Y, extent,
                       k1_or, k2_or,
                       J_field, mask_vis, mask_ok, band_mask,
                       maxima, minima,
                       cen_max, nn_max, nn_min,
                       mu):
    out_dir.mkdir(parents=True, exist_ok=True)
    cbkw  = dict(shrink=0.8)
    iy_mid = len(y) // 2

    # collect all maxima/minima x-positions for midline plot
    x_maxima = X[maxima[:, 0], maxima[:, 1]] if maxima.size else np.array([])
    x_minima = X[minima[:, 0], minima[:, 1]] if minima.size else np.array([])

    # --- Fig 1: pattern contours ---
    fig, ax = plt.subplots(figsize=(7, 3.5))
    levels = np.linspace(np.nanmin(u), np.nanmax(u), 40)
    cs = ax.contour(X, Y, u, levels=levels, cmap="gray", linewidths=0.6)
    if cen_max is not None:
        ax.scatter([X[cen_max]], [Y[cen_max]], c="cyan",  marker="+",
                   s=100, linewidths=2, zorder=5, label="central max")
    if nn_max is not None:
        ax.scatter([X[nn_max]], [Y[nn_max]], c="lime",  marker="+",
                   s=100, linewidths=2, zorder=5, label="nn max")
    if nn_min is not None:
        ax.scatter([X[nn_min]], [Y[nn_min]], c="red",   marker="x",
                   s=100, linewidths=2, zorder=5, label="nn min")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"contours + selected defects  (mu={mu:.3f})", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / f"{stem}_contours_defects.png", dpi=200)
    plt.close(fig)

    # --- Fig 2: pattern (copper) + ALL detected maxima and minima ---
    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(np.ma.masked_where(~mask_vis, u),
                   extent=extent, origin="lower", cmap="copper")
    plt.colorbar(im, ax=ax, **cbkw)
    if maxima.size:
        ax.scatter(X[maxima[:, 0], maxima[:, 1]],
                   Y[maxima[:, 0], maxima[:, 1]],
                   c="cyan", marker="+", s=80, linewidths=1.5,
                   zorder=5, label="J maxima")
    if minima.size:
        ax.scatter(X[minima[:, 0], minima[:, 1]],
                   Y[minima[:, 0], minima[:, 1]],
                   c="red", marker="x", s=80, linewidths=1.5,
                   zorder=5, label="J minima")
    # highlight selected pair
    if cen_max is not None:
        ax.scatter([X[cen_max]], [Y[cen_max]], c="yellow", marker="*",
                   s=160, zorder=6, label="central max")
    if nn_min is not None:
        ax.scatter([X[nn_min]], [Y[nn_min]], c="orange", marker="*",
                   s=160, zorder=6, label="nn min")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.5)
    ax.set_title(f"u + J defect markers  (mu={mu:.3f})", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / f"{stem}_pattern_defects.png", dpi=200)
    plt.close(fig)

    # --- Fig 3: J field with markers + band overlay ---
    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = _imshow(ax, J_field, mask_vis, extent, cmap="coolwarm")
    plt.colorbar(im, ax=ax, **cbkw)
    # shade core band
    band_display = np.ma.masked_where(~(band_mask & mask_vis),
                                      np.ones_like(J_field))
    ax.imshow(band_display, extent=extent, origin="lower",
              cmap="Greens", alpha=0.15, vmin=0, vmax=1)
    if maxima.size:
        ax.scatter(X[maxima[:, 0], maxima[:, 1]],
                   Y[maxima[:, 0], maxima[:, 1]],
                   c="cyan", marker="+", s=80, linewidths=1.5, zorder=5)
    if minima.size:
        ax.scatter(X[minima[:, 0], minima[:, 1]],
                   Y[minima[:, 0], minima[:, 1]],
                   c="red", marker="x", s=80, linewidths=1.5, zorder=5)
    if cen_max is not None:
        ax.scatter([X[cen_max]], [Y[cen_max]], c="yellow", marker="*",
                   s=160, zorder=6)
    if nn_min is not None:
        ax.scatter([X[nn_min]], [Y[nn_min]], c="orange", marker="*",
                   s=160, zorder=6)
    ax.set_title(f"J field + defect markers  (mu={mu:.3f})", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / f"{stem}_J_defects.png", dpi=200)
    plt.close(fig)

    # --- Fig 4: J midline profile + projected x-positions ---
    fig, ax = plt.subplots(figsize=(7, 3))
    J_mid = np.where(mask_ok[iy_mid, :], J_field[iy_mid, :], np.nan)
    ax.plot(x, J_mid, "k-", linewidth=1.0, label="J midline")
    for xm in x_maxima:
        ax.axvline(xm, color="cyan", linestyle="--", alpha=0.7, linewidth=0.8)
    for xm in x_minima:
        ax.axvline(xm, color="red",  linestyle=":",  alpha=0.7, linewidth=0.8)
    if cen_max is not None:
        ax.axvline(X[cen_max], color="yellow", linestyle="-",
                   linewidth=1.5, label="central max")
    if nn_min is not None:
        ax.axvline(X[nn_min], color="orange", linestyle="-",
                   linewidth=1.5, label="nn min")
    ax.axhline(0.0, color="k", linewidth=0.5, linestyle=":")
    ax.set_xlabel("x"); ax.set_ylabel("J")
    ax.set_title(f"J midline  (mu={mu:.3f})", fontsize=9)
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(out_dir / f"{stem}_J_midline.png", dpi=200)
    plt.close(fig)

    # --- Fig 4b: upper half of J (rows iy_mid:) ---
    fig, ax = plt.subplots(figsize=(7, 2.5))
    extent_upper = [x[0], x[-1], y[iy_mid], y[-1]]
    J_upper = J_field[iy_mid:, :]
    mask_upper = mask_ok[iy_mid:, :]
    im = ax.imshow(np.ma.masked_where(~mask_upper, J_upper),
                   extent=extent_upper, origin="lower", cmap="coolwarm", aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"J upper half (rows iy_mid:)  (mu={mu:.3f})", fontsize=9)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(out_dir / f"{stem}_J_upper_half.png", dpi=200)
    plt.close(fig)

    # --- Fig 4c: lower half of J (rows 0:iy_mid) ---
    fig, ax = plt.subplots(figsize=(7, 2.5))
    extent_lower = [x[0], x[-1], y[0], y[iy_mid]]
    J_lower = J_field[:iy_mid, :]
    mask_lower = mask_ok[:iy_mid, :]
    im = ax.imshow(np.ma.masked_where(~mask_lower, J_lower),
                   extent=extent_lower, origin="lower", cmap="coolwarm", aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"J lower half (rows 0:iy_mid)  (mu={mu:.3f})", fontsize=9)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(out_dir / f"{stem}_J_lower_half.png", dpi=200)
    plt.close(fig)

    # --- Reflected J: upper half defined by mirror of lower half ---
    J_refl = J_field.copy()
    n = iy_mid
    # flip lower half and copy into upper half row-by-row
    J_refl[n:n + n, :] = J_field[n - 1::-1, :]  # row iy_mid-1 → row iy_mid, etc.
    #J_refl = np.where(mask_ok, J_refl, np.nan)

    # --- Reflected J field (no mask) ---
    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(J_refl,
                   extent=extent, origin="lower", cmap="coolwarm", aspect="equal")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.axhline(y[iy_mid], color="white", linestyle="--", linewidth=0.8, alpha=0.7,
               label="midline")
    ax.set_title(f"J reflected (lower→upper)  (mu={mu:.3f})", fontsize=9)
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(out_dir / f"{stem}_J_reflected.png", dpi=200)
    plt.close(fig)

    # --- Reflected J midline (no original) ---
    fig, ax = plt.subplots(figsize=(7, 3))
    J_mid_refl = J_refl[iy_mid, :]  # no mask_ok
    ax.plot(x, J_mid_refl, "r-", linewidth=1.0, label="J midline (reflected)")
    ax.axhline(0.0, color="k", linewidth=0.4, linestyle=":")
    ax.set_xlabel("x");
    ax.set_ylabel("J")
    ax.set_title(f"J midline (reflected only)  (mu={mu:.3f})", fontsize=9)
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(out_dir / f"{stem}_J_reflected_midline.png", dpi=200)
    plt.close(fig)

    # --- Fig 5: k quiver ---
    step = max(1, min(len(x), len(y)) // 40)
    mask_q = mask_vis[::step, ::step]
    Xq = X[::step, ::step]; Yq = Y[::step, ::step]
    k1q = np.asarray(k1_or)[::step, ::step]
    k2q = np.asarray(k2_or)[::step, ::step]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.imshow(np.ma.masked_where(~mask_vis, u),
              extent=extent, origin="lower", cmap="copper")
    ax.quiver(Xq[mask_q], Yq[mask_q], k1q[mask_q], k2q[mask_q],
              color="white", scale=None, scale_units="xy", angles="xy")
    ax.set_title(f"k quiver (oriented)  (mu={mu:.3f})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(out_dir / f"{stem}_k_quiver.png", dpi=200)
    plt.close(fig)

    print(f"    saved 5 figs → {out_dir.name}/")


# -----------------------------------------------------------------------
# Summary spacing plot
# -----------------------------------------------------------------------

def save_spacing_curve(results, out_dir):
    """
    results : list of (mu, spacing_mm, spacing_pm)
    Two panels: max–max spacing and max–min spacing vs mu,
    both with theoretical curve L = pi / cos(alpha), sin(alpha) = mu.
    """
    valid = [(m, smm, spm) for m, smm, spm in results if np.isfinite(m)]
    if not valid:
        print("  no valid results — skipping spacing curve.")
        return

    valid.sort(key=lambda t: t[0])
    mus  = np.array([r[0] for r in valid])
    s_mm = np.array([r[1] for r in valid])
    s_pm = np.array([r[2] for r in valid])

    ok = mus < 1.0
    mu_th = mus[ok]
    L_th  = np.pi / np.sqrt(np.maximum(1.0 - mu_th**2, 1e-12))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))

    for ax, spacings, title, color in [
        (axs[0], s_mm, "max–max spacing  (nearest J maxima pair)", "steelblue"),
        (axs[1], s_pm, "max–min spacing  (central max → nearest min)", "tomato"),
    ]:
        ax.plot(mus, spacings, "o-", color=color, label="measured")
        if ok.any():
            ax.plot(mu_th, L_th, "k--", linewidth=1.2,
                    label=r"$\pi/\cos\alpha$,  $\sin\alpha=\mu$")
        ax.set_xlabel(r"$\mu$")
        ax.set_ylabel("spacing  (physical units)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle("Defect spacing vs μ  (J-density peaks, core band)", fontsize=10)
    plt.tight_layout()
    fpath = out_dir / "defect_spacing_vs_mu.png"
    fig.savefig(fpath, dpi=200)
    plt.close(fig)
    print(f"  spacing curve → {fpath}")


# -----------------------------------------------------------------------
# Per-file driver
# -----------------------------------------------------------------------

def process_one(path, base_out,
                ramp_thresh, pi_tol,
                xmargin, ymargin, tanhscale,
                orient_method,
                core_band_frac,
                min_distance, threshold_rel):

    path = Path(path)
    stem = path.stem
    out_dir = base_out / stem
    print(f"  → {stem}")

    x, y, u, k1_raw, k2_raw, ramp_saved = load_op_npz(path)
    ramp = _get_ramp(x, y, ramp_saved, xmargin, ymargin, tanhscale)
    mu   = _parse_mu(stem)

    X, Y   = np.meshgrid(x, y)
    extent = [x[0], x[-1], y[0], y[-1]]
    x_center = 0.5 * (x[0] + x[-1])

    mask_vis = ramp >= ramp_thresh
    mask_fd  = ramp >= ramp_thresh

    # orientation
    if orient_method == "bfs2":
        k1_or, k2_or = orient_vector_field_v2(k1_raw, k2_raw,
                                               mask=mask_fd, pi_tol=pi_tol)
    else:
        k1_or, k2_or = orient_vector_field(k1_raw, k2_raw, mask=mask_fd)

    phi_or     = np.arctan2(np.asarray(k2_or), np.asarray(k1_or))
    mask_pi_or = phi_jump_mask(phi_or, tol=pi_tol)
    mask_ok    = mask_fd & ~mask_pi_or

    # J field
    diag    = kfield_diagnostics(k1_or, k2_or, x, y, mask_ok)
    J_field = np.where(mask_ok, diag["J"], np.nan)

    # detect maxima and minima in core band
    maxima, minima, band_mask = detect_J_peaks(
        J_field, X, Y, mask_ok,
        core_band_frac=core_band_frac,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
    )

    # spacing calculations
    spacing_mm, spacing_pm, cen_max, nn_max, nn_min = compute_spacings(
        maxima, minima, X, Y, x_center)

    print(f"    mu={mu:.3f}  "
          f"n_max={len(maxima)}  n_min={len(minima)}  "
          f"spacing_mm={spacing_mm:.4f}  spacing_pm={spacing_pm:.4f}")

    save_per_file_figs(
        stem, out_dir,
        x, y, u, X, Y, extent,
        k1_or, k2_or,
        J_field, mask_vis, mask_ok, band_mask,
        maxima, minima,
        cen_max, nn_max, nn_min,
        mu,
    )

    return mu, spacing_mm, spacing_pm


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
                             "Default: experiments/pgb_analysis/results/defect_spacing_from_k_v2/")
    parser.add_argument("--ramp_thresh",    type=float, default=0.99)
    parser.add_argument("--pi_tol",         type=float, default=np.pi/10)
    parser.add_argument("--xmargin",        type=float, default=None)
    parser.add_argument("--ymargin",        type=float, default=None)
    parser.add_argument("--tanhscale",      type=float, default=None)
    parser.add_argument("--orient_method",  type=str,   default="bfs",
                        choices=["bfs", "bfs2"])
    parser.add_argument("--core_band_frac", type=float, default=0.10,
                        help="Half-width of midline core band (fraction of Ly).")
    parser.add_argument("--min_distance",   type=int,   default=5,
                        help="Min pixel separation between peaks.")
    parser.add_argument("--threshold_rel",  type=float, default=0.10,
                        help="Relative threshold for peak_local_max.")
    parser.add_argument("--mu_min", type=float, default=None,
                        help="Only process files with mu > this value.")

    ns = parser.parse_args(args)

    if ns.out_dir is not None:
        base_out = Path(ns.out_dir)
    else:
        base_out = _HERE / "results" / "defect_spacing_from_k_v2"
    base_out.mkdir(parents=True, exist_ok=True)

    ramp_kw = dict(xmargin=ns.xmargin, ymargin=ns.ymargin, tanhscale=ns.tanhscale)
    common = dict(
        ramp_thresh=ns.ramp_thresh, pi_tol=ns.pi_tol,
        orient_method=ns.orient_method,
        core_band_frac=ns.core_band_frac,
        min_distance=ns.min_distance,
        threshold_rel=ns.threshold_rel,
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

    if len(results) > 1:
        save_spacing_curve(results, base_out)

    print("Done.")


# -----------------------------------------------------------------------
# Debug block
# -----------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) == 1:

        class _Args:
            op_file        = None
            #op_dir         = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweep_uhu_3_sig_pio2/raw"
            #op_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/uhu/0711_np16_nx1024/sig_1/raw"
            #op_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/uhu/0711_np8_nx1024/sig_pio2/raw"
            op_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/uhu/0711_np16_nx1024/sig_pio2/raw"
            pattern        = "*.npz"
            #out_dir        = "/Users/edwardmcdugald/patterns/experiments/pgb_analysis/results/defect_spacing_from_k_v2/mu_sweep_uhu_3_sig_pio2"
            #out_dir = "/Users/edwardmcdugald/patterns/experiments/pgb_analysis/results/defect_spacing_from_k/uhu/0711_np16_nx1024/sig_1"
            #out_dir = "/Users/edwardmcdugald/patterns/experiments/pgb_analysis/results/defect_spacing_from_k/uhu/0711_np8_nx1024/sig_pio2"
            out_dir = "/Users/edwardmcdugald/patterns/experiments/pgb_analysis/results/defect_spacing_from_k/uhu/0711_np16_nx1024/sig_pio2"
            ramp_thresh    = 1 - 1e-12
            pi_tol         = np.pi / 10
            xmargin        = 0.025
            ymargin        = 0.025
            tanhscale      = 120.0
            orient_method  = "bfs"
            core_band_frac = 0.05
            min_distance   = 5
            threshold_rel  = 0.05
            mu_min = 0.63

        a = _Args()
        main([
            "--op_dir",         a.op_dir,
            "--out_dir",        a.out_dir,
            "--pattern",        a.pattern,
            "--ramp_thresh",    str(a.ramp_thresh),
            "--pi_tol",         str(a.pi_tol),
            "--orient_method",  a.orient_method,
            "--core_band_frac", str(a.core_band_frac),
            "--min_distance",   str(a.min_distance),
            "--threshold_rel",  str(a.threshold_rel),
            *(["--xmargin",    str(a.xmargin),
               "--ymargin",    str(a.ymargin),
               "--tanhscale",  str(a.tanhscale)]
              if a.xmargin is not None else []),
            *(["--mu_min", str(a.mu_min)] if a.mu_min is not None else []),
        ])
    else:
        main()