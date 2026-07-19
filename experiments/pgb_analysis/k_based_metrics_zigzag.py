# experiments/pgb_analysis/k_based_metrics.py
"""
K-based diagnostic plots for PGB zigzag SH order-parameter files.

Default (press Run / F5): processes all OP .npz files found in
    /Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweep_uhu_2/raw

Output figures land in <op_dir>/../figures/k_based_metrics/

Usage
-----
# press Run in IDE — uses the debug block below

# --- directory of OP files ---
python k_based_metrics.py --op_dir /path/to/ops/raw

# --- single file ---
python k_based_metrics.py --op_file /path/to/my_op.npz

# --- rebuild ramp with custom params (override saved ramp) ---
python k_based_metrics.py --op_dir /path/to/ops/raw \\
    --xmargin 0.05 --ymargin 0.05 --tanhscale 60.0

# --- tighten interior threshold ---
python k_based_metrics.py --op_dir /path/to/ops/raw --ramp_thresh 0.999
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- make src/ importable regardless of cwd ---
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]          # project root  (experiments/ → root)
sys.path.insert(0, str(_ROOT / "src"))

from utils.kfield_calcs import (
    phi_jump_mask,
    disk_twist_integrals,
    circle_circulation_integrals,
)

from skimage.feature import peak_local_max
from utils.geometry import build_rectangular_ramp_smooth   # same as OP runner


# -----------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------

def load_op_npz(path):
    d = np.load(path, allow_pickle=True)
    x = d["x"]
    y = d["y"]

    def get_frame(name, frame):
        arr = d[name]
        if arr.ndim == 3:
            return arr[..., frame]
        return arr

    u_all = d["u"]
    nt = u_all.shape[-1] if u_all.ndim == 3 else 1

    u0 = get_frame("u", 0)
    uf = get_frame("u", -1)

    k10 = get_frame("k1", 0)
    k1f = get_frame("k1", -1)

    k20 = get_frame("k2", 0)
    k2f = get_frame("k2", -1)

    A0 = get_frame("A", 0)
    Af = get_frame("A", -1)

    ramp = d["ramp"] if "ramp" in d else None
    return x, y, u0, uf, k10, k1f, k20, k2f, A0, Af, ramp, nt


def _get_ramp(x, y, ramp_saved, xmargin, ymargin, tanhscale):
    """
    Return ramp array.
    If xmargin/ymargin/tanhscale are provided (not None), rebuild from scratch.
    Otherwise fall back to the ramp saved in the OP file.
    """
    if xmargin is not None and ymargin is not None and tanhscale is not None:
        print(f"    rebuilding ramp: xmargin={xmargin}, ymargin={ymargin}, tanhscale={tanhscale}")
        return build_rectangular_ramp_smooth(x, y,
                                             xmargin=xmargin,
                                             ymargin=ymargin,
                                             tanhscale=tanhscale)
    if ramp_saved is not None:
        return ramp_saved
    raise ValueError("No ramp in OP file and no ramp params supplied. "
                     "Pass --xmargin/--ymargin/--tanhscale.")


# -----------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------

def _imshow(ax, data, mask, extent, cmap="viridis", **kw):
    im = ax.imshow(np.ma.masked_where(~mask, data),
                   extent=extent, origin="lower", cmap=cmap, **kw)
    ax.set_aspect("equal")
    return im


def make_amplitude_fig(A0, Af, mask_vis, extent, stem):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    cbkw = dict(shrink=0.8)

    panels = [
        (A0, "amplitude surface A (initial)", "magma"),
        (Af, "amplitude surface A (final)", "magma"),
    ]

    for ax, (arr, title, cmap) in zip(axs.flat, panels):
        im = _imshow(ax, arr, mask_vis, extent, cmap=cmap)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, **cbkw)

    fig.suptitle(stem, fontsize=10)
    plt.tight_layout()
    return fig


def make_midline_fig(x, y, u0, uf, k10, k1f, k20, k2f, A0, Af, ramp, mask_vis, stem):
    k0 = np.sqrt(k10**2 + k20**2)
    kf = np.sqrt(k1f**2 + k2f**2)
    phi0 = np.arctan2(k20, k10)
    phif = np.arctan2(k2f, k1f)

    iy_mid = len(y) // 2
    ix_mid = len(x) // 2

    u0x = u0[iy_mid, :]
    ufx = uf[iy_mid, :]
    u0y = u0[:, ix_mid]
    ufy = uf[:, ix_mid]

    k0x = k0[iy_mid, :]
    kfx = kf[iy_mid, :]
    k0y = k0[:, ix_mid]
    kfy = kf[:, ix_mid]

    A0x = A0[iy_mid, :]
    Afx = Af[iy_mid, :]
    A0y = A0[:, ix_mid]
    Afy = Af[:, ix_mid]

    p0x = phi0[iy_mid, :]
    pfx = phif[iy_mid, :]
    p0y = phi0[:, ix_mid]
    pfy = phif[:, ix_mid]

    rx = ramp[iy_mid, :]
    ry = ramp[:, ix_mid]

    fig, axs = plt.subplots(5, 2, figsize=(14, 17), constrained_layout=True)

    axs[0, 0].plot(x, u0x, lw=2, label="u initial")
    axs[0, 0].plot(x, ufx, lw=2, label="u final")
    axs[0, 0].set_title(f"u along x-midline (y={y[iy_mid]:.3f})")
    axs[0, 0].legend()

    axs[0, 1].plot(y, u0y, lw=2, label="u initial")
    axs[0, 1].plot(y, ufy, lw=2, label="u final")
    axs[0, 1].set_title(f"u along y-midline (x={x[ix_mid]:.3f})")
    axs[0, 1].legend()

    axs[1, 0].plot(x, k0x, lw=2, label="|k| initial")
    axs[1, 0].plot(x, kfx, lw=2, label="|k| final")
    axs[1, 0].set_title("|k| along x-midline")
    axs[1, 0].legend()

    axs[1, 1].plot(y, k0y, lw=2, label="|k| initial")
    axs[1, 1].plot(y, kfy, lw=2, label="|k| final")
    axs[1, 1].set_title("|k| along y-midline")
    axs[1, 1].legend()

    axs[2, 0].plot(x, A0x, lw=2, label="A initial")
    axs[2, 0].plot(x, Afx, lw=2, label="A final")
    axs[2, 0].set_title("A along x-midline")
    axs[2, 0].legend()

    axs[2, 1].plot(y, A0y, lw=2, label="A initial")
    axs[2, 1].plot(y, Afy, lw=2, label="A final")
    axs[2, 1].set_title("A along y-midline")
    axs[2, 1].legend()

    axs[3, 0].plot(x, p0x, lw=1.8, label="arg(k) init")
    axs[3, 0].plot(x, pfx, lw=1.8, label="arg(k) final")
    axs[3, 0].set_title("arg(k) along x-midline")
    axs[3, 0].legend()

    axs[3, 1].plot(y, p0y, lw=1.8, label="arg(k) init")
    axs[3, 1].plot(y, pfy, lw=1.8, label="arg(k) final")
    axs[3, 1].set_title("arg(k) along y-midline")
    axs[3, 1].legend()

    axs[4, 0].plot(x, rx, lw=2, color="black")
    axs[4, 0].set_title("ramp along x-midline")

    axs[4, 1].plot(y, ry, lw=2, color="black")
    axs[4, 1].set_title("ramp along y-midline")

    for ax in axs.flat:
        ax.grid(alpha=0.25)

    fig.suptitle(stem, fontsize=10)
    return fig


def kfield_diagnostics_lower_and_reflected(k1, k2, x, y, mask_fd, mask_pi_raw):
    """
    Compute k-based diagnostics from RAW k only.

    - Derivatives are computed on the LOWER half-plane (rows 0..iy_mid-1),
      using np.gradient (central where possible, one-sided at row 0 and row iy_mid-1).
    - Upper half-plane diagnostics are obtained by vertical reflection of the lower half
      across the midline.
    - mask_fd is the ramp-based interior mask; mask_pi_raw excludes π-jumps.

    Returns
    -------
    diag_lower : dict with 'div_k', 'curl_k', 'J', 'E' (upper rows NaN)
    diag_full  : dict with same keys, upper rows filled by reflection.
    """
    k1 = np.asarray(k1)
    k2 = np.asarray(k2)
    Ny, Nx = k1.shape

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    iy_mid = Ny // 2  # consistent with your other scripts

    # Good region mask: interior AND not π-jump AND lower half only
    mask_ok_lower = mask_fd & (~mask_pi_raw)
    mask_ok_lower[iy_mid:, :] = False

    # Work on lower-half slices only
    k1_lower = k1[:iy_mid, :]
    k2_lower = k2[:iy_mid, :]

    # Derivatives on lower half
    k1_y_lower = np.gradient(k1_lower, dy, axis=0)
    k2_y_lower = np.gradient(k2_lower, dy, axis=0)
    k1_x_lower = np.gradient(k1_lower, dx, axis=1)
    k2_x_lower = np.gradient(k2_lower, dx, axis=1)

    # Embed derivatives into full-size arrays; upper half = NaN
    k1_x = np.full_like(k1, np.nan)
    k1_y = np.full_like(k1, np.nan)
    k2_x = np.full_like(k2, np.nan)
    k2_y = np.full_like(k2, np.nan)

    k1_x[:iy_mid, :] = k1_x_lower
    k1_y[:iy_mid, :] = k1_y_lower
    k2_x[:iy_mid, :] = k2_x_lower
    k2_y[:iy_mid, :] = k2_y_lower

    # Lower-half diagnostics
    div_k_lower  = k1_x + k2_y
    curl_k_lower = k2_x - k1_y
    J_lower      = k1_x * k2_y - k1_y * k2_x
    E_lower      = (k1_x**2 + k1_y**2 + k2_x**2 + k2_y**2)

    # Apply lower mask: outside lower-half good region → NaN
    div_k_lower  = np.where(mask_ok_lower, div_k_lower,  np.nan)
    curl_k_lower = np.where(mask_ok_lower, curl_k_lower, np.nan)
    J_lower      = np.where(mask_ok_lower, J_lower,      np.nan)
    E_lower      = np.where(mask_ok_lower, E_lower,      np.nan)

    # Build reflected full-field diagnostics
    # Assume vertical symmetry: upper half mirrors lower half about midline.
    div_k_full  = np.full_like(div_k_lower, np.nan)
    curl_k_full = np.full_like(curl_k_lower, np.nan)
    J_full      = np.full_like(J_lower, np.nan)
    E_full      = np.full_like(E_lower, np.nan)

    # Copy lower half as-is
    div_k_full[:iy_mid, :]  = div_k_lower[:iy_mid, :]
    curl_k_full[:iy_mid, :] = curl_k_lower[:iy_mid, :]
    J_full[:iy_mid, :]      = J_lower[:iy_mid, :]
    E_full[:iy_mid, :]      = E_lower[:iy_mid, :]

    # Reflect lower half into upper half rows iy_mid..(2*iy_mid-1) if possible
    n_upper = min(Ny - iy_mid, iy_mid)
    if n_upper > 0:
        source = div_k_lower[:iy_mid, :][::-1, :]  # flipped lower half
        div_k_full[iy_mid:iy_mid + n_upper, :]  = source[:n_upper, :]
        curl_k_full[iy_mid:iy_mid + n_upper, :] = curl_k_lower[:iy_mid, :][::-1, :][:n_upper, :]
        J_full[iy_mid:iy_mid + n_upper, :]      = J_lower[:iy_mid, :][::-1, :][:n_upper, :]
        E_full[iy_mid:iy_mid + n_upper, :]      = E_lower[:iy_mid, :][::-1, :][:n_upper, :]

    diag_lower = {
        "div_k":  div_k_lower,
        "curl_k": curl_k_lower,
        "J":      J_lower,
        "E":      E_lower,
    }

    diag_full = {
        "div_k":  div_k_full,
        "curl_k": curl_k_full,
        "J":      J_full,
        "E":      E_full,
    }

    return diag_lower, diag_full, mask_ok_lower


def make_geometry_fig(x, y,
                      u0, uf,
                      k10, k1f,
                      k20, k2f,
                      A0, Af,
                      ramp, mask_vis,
                      mask_pi0, mask_pif,
                      extent, stem):
    """
    4 rows × 4 cols:
      Row 0: u init, u final, A init, A final
      Row 1: |k| init, |k| final, arg(k) init, arg(k) final
      Row 2: k1 init, k1 final, k2 init, k2 final
      Row 3: ramp, pi-jump init, pi-jump final, empty
    """
    k0   = np.sqrt(k10 ** 2 + k20 ** 2)
    kf   = np.sqrt(k1f ** 2 + k2f ** 2)
    phi0 = np.arctan2(k20, k10)
    phif = np.arctan2(k2f, k1f)

    fig, axs = plt.subplots(4, 4, figsize=(20, 16))
    cbkw = dict(shrink=0.8)

    panels = [
        (u0, "u (initial)", "copper", {}),
        (uf, "u (final)", "copper", {}),
        (A0, "A (initial)", "magma", {}),
        (Af, "A (final)", "magma", {}),

        (k0, "|k| (initial)", "viridis", {}),
        (kf, "|k| (final)", "viridis", {}),
        (phi0, "arg(k) (initial)", "twilight", {}),
        (phif, "arg(k) (final)", "twilight", {}),

        (k10, "k1 (initial)", "coolwarm", {}),
        (k1f, "k1 (final)", "coolwarm", {}),
        (k20, "k2 (initial)", "coolwarm", {}),
        (k2f, "k2 (final)", "coolwarm", {}),

        (ramp, "ramp", "gray", dict(vmin=0, vmax=1)),
        (mask_pi0.astype(float), "π-jump (initial)", "gray_r", dict(vmin=0, vmax=1)),
        (mask_pif.astype(float), "π-jump (final)", "gray_r", dict(vmin=0, vmax=1)),
        (np.zeros_like(ramp), "", "gray", {}),
    ]

    for ax, (arr, title, cmap, kw) in zip(axs.flat, panels):
        if title:
            im = _imshow(ax, arr, mask_vis, extent, cmap=cmap, **kw)
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, **cbkw)
        else:
            ax.axis("off")

    fig.suptitle(stem, fontsize=10)
    plt.tight_layout()
    return fig


def make_diagnostics_fig(diag0_lower, diag0_full,
                         diagf_lower, diagf_full,
                         mask_ok0_lower, mask_okf_lower,
                         mask_vis, extent, stem):
    """
    4 rows:
      initial lower-half
      initial reflected full
      final lower-half
      final reflected full
    Cols: curl k | div k | J | E
    """
    fields = ["curl_k", "div_k", "J", "E"]
    labels = ["curl k", "div k", "J", "E"]
    cmaps = ["coolwarm", "coolwarm", "coolwarm", "hot"]

    fig, axs = plt.subplots(4, 4, figsize=(18, 16))
    cbkw = dict(shrink=0.8)

    rows = [
        (diag0_lower, "initial lower-half", mask_ok0_lower),
        (diag0_full,  "initial reflected full", mask_vis),
        (diagf_lower, "final lower-half", mask_okf_lower),
        (diagf_full,  "final reflected full", mask_vis),
    ]

    for col, (key, label, cmap) in enumerate(zip(fields, labels, cmaps)):
        for row_idx, (diag, tag, mask_row) in enumerate(rows):
            arr = diag[key]
            im = _imshow(axs[row_idx, col], arr, mask_row, extent, cmap=cmap)
            axs[row_idx, col].set_title(f"{label} ({tag})", fontsize=9)
            fig.colorbar(im, ax=axs[row_idx, col], **cbkw)

    fig.suptitle(stem, fontsize=10)
    plt.tight_layout()
    return fig


def make_quiver_fig(x, y, u,
                    k1_raw, k2_raw,
                    mask_vis, extent, stem,
                    step=12):
    """
    Wave-vector quiver for RAW k only, over u, masked by mask_vis.
    """
    X, Y = np.meshgrid(x, y)
    Xq = X[::step, ::step]
    Yq = Y[::step, ::step]
    mask_q = mask_vis[::step, ::step]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    k1q = np.asarray(k1_raw)[::step, ::step]
    k2q = np.asarray(k2_raw)[::step, ::step]

    ax.imshow(np.ma.masked_where(~mask_vis, u),
              extent=extent, origin="lower", cmap="copper")
    ax.quiver(
        Xq[mask_q],
        Yq[mask_q],
        k1q[mask_q],
        k2q[mask_q],
        color="white",
        scale=None,
        scale_units="xy",
        angles="xy",
    )
    ax.set_title("wave-vector quiver (raw)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    fig.suptitle(stem, fontsize=10)
    plt.tight_layout()
    return fig

def _overlay_peaks(ax, arr, mask, X, Y, min_distance, threshold_rel, mode="both"):
    """Detect and scatter peaks of arr onto ax. mode: 'both'=|arr|, 'pos', 'neg'."""
    data = np.nan_to_num(np.where(mask, arr, np.nan), nan=0.0)
    if mode == "both":
        field = np.abs(data)
    elif mode == "pos":
        field = data
    else:
        field = -data
    peaks = peak_local_max(field, min_distance=min_distance,
                           threshold_rel=threshold_rel)
    if peaks.size:
        ax.scatter(X[peaks[:, 0], peaks[:, 1]],
                   Y[peaks[:, 0], peaks[:, 1]],
                   s=18, c="cyan", marker="+", linewidths=1.5, zorder=5)


def make_defect_fig(x, y, u, k1_raw, k2_raw, diag_full, mask_vis,
                    extent, stem,
                    radius, threshold_rel, min_distance):
    """
    3 rows × 4 cols:
      Row 0: J density | twist integral | curl_k density | circ integral  (field only)
      Row 1: same fields with peaks overlaid on diagnostic field
      Row 2: u (pattern) as background, peaks from each field overlaid
    """
    X, Y = np.meshgrid(x, y)

    J_field    = diag_full["J"]
    curl_field = diag_full["curl_k"]

    # For integrals, use mask_vis as both data and integration mask
    twist_int = disk_twist_integrals(
        J_field, X, Y, mask_vis, mask_vis, radius=radius)
    circ_int  = circle_circulation_integrals(
        np.asarray(k1_raw), np.asarray(k2_raw),
        X, Y, mask_vis, mask_vis, radius=radius)

    fields = [
        (J_field,    "J density (reflected)",          "coolwarm"),
        (twist_int,  f"disk twist (r={radius:.2f})",   "coolwarm"),
        (curl_field, "curl k density (reflected)",     "coolwarm"),
        (circ_int,   f"circle circ (r={radius:.2f})",  "coolwarm"),
    ]

    fig, axs = plt.subplots(3, 4, figsize=(22, 15))
    cbkw = dict(shrink=0.8)

    for col, (arr, title, cmap) in enumerate(fields):
        # row 0: field only
        im = _imshow(axs[0, col], arr, mask_vis, extent, cmap=cmap)
        axs[0, col].set_title(title, fontsize=9)
        fig.colorbar(im, ax=axs[0, col], **cbkw)

        # row 1: field + peaks overlaid on diagnostic field
        im2 = _imshow(axs[1, col], arr, mask_vis, extent, cmap=cmap)
        _overlay_peaks(axs[1, col], arr, mask_vis, X, Y,
                       min_distance=min_distance,
                       threshold_rel=threshold_rel,
                       mode="both")
        axs[1, col].set_title(f"{title} + peaks", fontsize=8)
        fig.colorbar(im2, ax=axs[1, col], **cbkw)

        # row 2: u as background + peaks overlaid on pattern
        im3 = _imshow(axs[2, col], u, mask_vis, extent, cmap="copper")
        _overlay_peaks(axs[2, col], arr, mask_vis, X, Y,
                       min_distance=min_distance,
                       threshold_rel=threshold_rel,
                       mode="both")
        axs[2, col].set_title(f"u + peaks from {title}", fontsize=8)
        fig.colorbar(im3, ax=axs[2, col], **cbkw)

    fig.suptitle(stem, fontsize=10)
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------
# Per-file processing
# -----------------------------------------------------------------------

def process_one(path, out_dir, ramp_thresh, pi_tol,
                xmargin, ymargin, tanhscale,
                defect_radius=np.pi/2,
                defect_thresh=0.05,
                defect_min_dist=5):
    path    = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem    = path.stem
    print(f"  → {stem}")

    x, y, u0, uf, k10, k1f, k20, k2f, A0, Af, ramp_saved, nt = load_op_npz(path)
    ramp = _get_ramp(x, y, ramp_saved, xmargin, ymargin, tanhscale)

    extent   = [x[0], x[-1], y[0], y[-1]]
    mask_vis = ramp >= ramp_thresh          # for plotting / diagnostics
    mask_fd  = ramp >= ramp_thresh

    phi0_raw = np.arctan2(k20, k10)
    phif_raw = np.arctan2(k2f, k1f)

    mask_pi0_raw = phi_jump_mask(phi0_raw, tol=pi_tol)
    mask_pif_raw = phi_jump_mask(phif_raw, tol=pi_tol)

    diag0_lower, diag0_full, mask_ok0_lower = kfield_diagnostics_lower_and_reflected(
        k10, k20, x, y, mask_fd, mask_pi0_raw
    )

    diagf_lower, diagf_full, mask_okf_lower = kfield_diagnostics_lower_and_reflected(
        k1f, k2f, x, y, mask_fd, mask_pif_raw
    )

    # Geometry (final raw k only + π jumps)
    fig1 = make_geometry_fig(
        x, y,
        u0, uf,
        k10, k1f,
        k20, k2f,
        A0, Af,
        ramp, mask_vis,
        mask_pi0_raw, mask_pif_raw,
        extent,
        f"{stem} | Nt={nt}",
    )
    fig1.savefig(out_dir / f"{stem}_geometry.png", dpi=150)
    plt.close(fig1)

    # Amplitude surfaces: initial + final
    figA = make_amplitude_fig(
        A0, Af,
        mask_vis,
        extent,
        f"{stem} | Nt={nt}",
    )
    figA.savefig(out_dir / f"{stem}_amplitude.png", dpi=150)
    plt.close(figA)

    # Midline diagnostics: initial + final
    figM = make_midline_fig(
        x, y,
        u0, uf,
        k10, k1f,
        k20, k2f,
        A0, Af,
        ramp,
        mask_vis,
        f"{stem} | Nt={nt}",
    )
    figM.savefig(out_dir / f"{stem}_midlines.png", dpi=150)
    plt.close(figM)

    # Diagnostics: lower vs reflected full-field
    fig2 = make_diagnostics_fig(
        diag0_lower, diag0_full,
        diagf_lower, diagf_full,
        mask_ok0_lower, mask_okf_lower,
        mask_vis, extent,
        f"{stem} | Nt={nt}"
    )
    fig2.savefig(out_dir / f"{stem}_diagnostics.png", dpi=150)
    plt.close(fig2)

    # Quiver: final raw k only
    fig3 = make_quiver_fig(
        x, y, uf,
        k1f, k2f,
        mask_vis,
        extent,
        f"{stem} | Nt={nt}",
    )
    fig3.savefig(out_dir / f"{stem}_quiver.png", dpi=150)
    plt.close(fig3)

    # Defects: based on reflected J/curl and final raw k
    fig4 = make_defect_fig(
        x, y, uf,
        k1f, k2f, diagf_full,
        mask_vis,
        extent, f"{stem} | Nt={nt}",
        radius=defect_radius,
        threshold_rel=defect_thresh,
        min_distance=defect_min_dist,
    )
    fig4.savefig(out_dir / f"{stem}_defects.png", dpi=150)
    plt.close(fig4)

    print(
        f"    saved: {stem}_geometry.png  {stem}_amplitude.png  "
        f"{stem}_midlines.png  {stem}_diagnostics.png  "
        f"{stem}_quiver.png  {stem}_defects.png"
    )


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(
        description="K-based diagnostic plots for PGB OP files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--op_file",     type=str, default=None,
                        help="Single OP .npz file.")
    parser.add_argument("--op_dir",      type=str, default=None,
                        help="Directory of OP .npz files.")
    parser.add_argument("--pattern",     type=str, default="*.npz",
                        help="Glob pattern inside op_dir.")
    parser.add_argument("--out_dir",     type=str, default=None,
                        help="Output directory for figures. "
                             "Default: <op_dir>/../figures/k_based_metrics_v2")
    parser.add_argument("--ramp_thresh", type=float, default=0.99,
                        help="Mask threshold: keep where ramp >= this.")
    parser.add_argument("--pi_tol",      type=float, default=np.pi / 10,
                        help="π-jump tolerance.")
    # optional ramp override
    parser.add_argument("--xmargin",   type=float, default=None,
                        help="Rebuild ramp: x-margin fraction.")
    parser.add_argument("--ymargin",   type=float, default=None,
                        help="Rebuild ramp: y-margin fraction.")
    parser.add_argument("--tanhscale", type=float, default=None,
                        help="Rebuild ramp: tanh steepness.")
    parser.add_argument("--defect_radius", type=float, default=np.pi / 2,
                        help="Radius for disk/circle integral defect detection.")
    parser.add_argument("--defect_thresh", type=float, default=0.05,
                        help="threshold_rel for peak_local_max.")
    parser.add_argument("--defect_min_dist", type=int, default=5,
                        help="min_distance (pixels) for peak_local_max.")

    ns = parser.parse_args(args)

    # Base results directory under experiments/pgb_analysis
    if ns.out_dir is not None:
        base_out = Path(ns.out_dir)
    else:
        root_results = _HERE / "results" / "k_based_metrics_v2"
        if ns.op_file is not None:
            tag = Path(ns.op_file).stem
        elif ns.op_dir is not None:
            tag = Path(ns.op_dir).name
        else:
            tag = "default"
        base_out = root_results / tag

    if ns.op_file is not None:
        op_path = Path(ns.op_file)
        out_dir = base_out
        process_one(op_path, out_dir,
                    ns.ramp_thresh, ns.pi_tol,
                    ns.xmargin, ns.ymargin, ns.tanhscale,
                    defect_radius=ns.defect_radius,
                    defect_thresh=ns.defect_thresh,
                    defect_min_dist=ns.defect_min_dist)

    else:
        op_dir = Path(ns.op_dir or ".")
        out_dir = base_out
        files = sorted(op_dir.glob(ns.pattern))
        if not files:
            raise SystemExit(f"No files matching '{ns.pattern}' in {op_dir}")
        print(f"Processing {len(files)} file(s) from {op_dir}")
        for f in files:
            process_one(f, out_dir,
                        ns.ramp_thresh, ns.pi_tol,
                        ns.xmargin, ns.ymargin, ns.tanhscale,
                        defect_radius=ns.defect_radius,
                        defect_thresh=ns.defect_thresh,
                        defect_min_dist=ns.defect_min_dist)

    print("Done.")

# -----------------------------------------------------------------------
# Debug / IDE "press Run" block — edit paths here, then Run/Debug
# -----------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) == 1:

        class _Args:
            op_file     = None
            #op_dir      = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweep_uhu_3_sig1/raw"
            #op_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/uhu/full_run_np16_Ny5_longrun/sig_pio2/raw"
            op_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/uhu/mu_sweeps_full_Nx512_hp2_T25_NyF5_np18_Nsave125/sig_pio2/raw"
            pattern     = "*.npz"
            #out_dir     = "/Users/edwardmcdugald/patterns/experiments/pgb_analysis/results/k_based_metrics_v2/mu_sweep_uhu_3_sig1/"
            #out_dir = "/Users/edwardmcdugald/patterns/experiments/pgb_analysis/results/k_based_metrics/updated/full_run_np16_Ny5_longrun/sig_pio2"
            out_dir = "/Users/edwardmcdugald/patterns/experiments/pgb_analysis/results/k_based_metrics/mu_sweeps_full_Nx512_hp2_T25_NyF5_np18_Nsave125/sig_pio2"
            ramp_thresh = 1-1e-12
            pi_tol      = np.pi / 10

            # set all three to rebuild ramp; leave as None to use saved ramp
            # xmargin     = None
            # ymargin     = None
            # tanhscale   = None

            # example override to match your debug OP run:
            xmargin   = 0.025
            ymargin   = 0.025
            tanhscale = 120.0
            defect_radius = np.pi / 2
            defect_thresh = 0.10
            defect_min_dist = 10

        a = _Args()
        main([
            "--op_dir",      a.op_dir,
            "--out_dir",     a.out_dir,
            "--pattern",     a.pattern,
            "--ramp_thresh", str(a.ramp_thresh),
            "--pi_tol",      str(a.pi_tol),
            *(["--xmargin",   str(a.xmargin),
               "--ymargin",   str(a.ymargin),
               "--tanhscale", str(a.tanhscale)]
              if a.xmargin is not None else []),
        ])
    else:
        main()