# experiments/pgb_analysis/k_based_metrics_diamond.py
"""
K-based diagnostic plots for PGB diamond SH order-parameter files.

Default (press Run / F5): processes all OP .npz files found in
    /Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_diamond/uhu/raw

Output figures land in <op_dir>/../figures/k_based_metrics_diamond/

Usage
-----
# press Run in IDE — uses the debug block below

# --- directory of OP files ---
python k_based_metrics_diamond.py --op_dir /path/to/ops/raw

# --- single file ---
python k_based_metrics_diamond.py --op_file /path/to/my_op.npz

# --- rebuild diamond ramp with custom params (override saved ramp) ---
python k_based_metrics_diamond.py --op_dir /path/to/ops/raw \\
    --margin 0.30 --tanh_scale 80.0 --smooth_sigma 1.0

# --- tighten interior threshold ---
python k_based_metrics_diamond.py --op_dir /path/to/ops/raw --ramp_thresh 0.999
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
from utils.geometry import build_diamond_ramp_general


# -----------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------

def _maybe_item_string(raw):
    if raw is None:
        return None
    try:
        return raw.item()
    except Exception:
        return str(raw)


def _get_json_mu(*json_blobs):
    for blob in json_blobs:
        if not blob:
            continue
        try:
            obj = json.loads(blob)
            mu = obj.get("mu")
            if mu is not None:
                return float(mu)
        except Exception:
            pass
    return None


def load_op_npz_diamond(path):
    """
    Load diamond OP .npz file.

    Expected fields (flexible):
      - x, y
      - u (Ny, Nx, Nt) or (Ny, Nx)
      - k1_sym/k2_sym or k1/k2
      - A
      - ramp (optional)
      - mu or mu in *_meta_json
    """
    path = Path(path)
    d = np.load(path, allow_pickle=True)

    x = d["x"]
    y = d["y"]

    # Ensure 3D u for consistent indexing
    u = d["u"]
    if u.ndim == 2:
        u = u[:, :, None]

    nt = u.shape[-1]

    def get_frame(name, frame):
        if name not in d:
            return None
        arr = d[name]
        if arr.ndim == 3:
            return arr[..., frame]
        return arr

    u0 = u[..., 0]
    uf = u[..., -1]

    # Prefer symmetric k if present
    k1_key = "k1_sym" if "k1_sym" in d else ("k1" if "k1" in d else "k1_orig")
    k2_key = "k2_sym" if "k2_sym" in d else ("k2" if "k2" in d else "k2_orig")

    if k1_key not in d or k2_key not in d:
        raise ValueError(f"{path.name}: missing usable k1/k2 fields")

    k10 = get_frame(k1_key, 0)
    k1f = get_frame(k1_key, -1)
    k20 = get_frame(k2_key, 0)
    k2f = get_frame(k2_key, -1)

    A0 = get_frame("A", 0) if "A" in d else None
    Af = get_frame("A", -1) if "A" in d else None

    ramp = d["ramp"] if "ramp" in d else None

    uhu_meta_json = _maybe_item_string(d["uhu_meta_json"]) if "uhu_meta_json" in d else None
    sh_meta_json = _maybe_item_string(d["sh_meta_json"]) if "sh_meta_json" in d else None

    mu = _get_json_mu(uhu_meta_json, sh_meta_json)
    if mu is None and "mu" in d:
        mu = float(np.asarray(d["mu"]).ravel()[0])

    return {
        "path": path,
        "x": x,
        "y": y,
        "u0": u0,
        "uf": uf,
        "k10": k10,
        "k1f": k1f,
        "k20": k20,
        "k2f": k2f,
        "A0": A0,
        "Af": Af,
        "ramp_saved": ramp,
        "mu": mu,
        "uhu_meta_json": uhu_meta_json,
        "sh_meta_json": sh_meta_json,
        "nt": nt,
    }


def _get_ramp_diamond(x, y, mu, ramp_saved, margin, tanh_scale, smooth_sigma):
    """
    Return ramp array for diamond geometry.

    If margin/tanh_scale/smooth_sigma are provided (not None), rebuild from scratch
    via build_diamond_ramp_general. Otherwise fall back to the ramp saved in the OP file.
    """
    if margin is not None and tanh_scale is not None and smooth_sigma is not None:
        if mu is None:
            raise ValueError("Rebuild ramp requested but mu is missing.")
        print(
            f"    rebuilding diamond ramp: margin={margin}, tanh_scale={tanh_scale}, "
            f"smooth_sigma={smooth_sigma}, mu={mu:.4f}"
        )
        return build_diamond_ramp_general(
            x, y, mu,
            margin=margin,
            tanh_scale=tanh_scale,
            smooth_sigma=smooth_sigma,
        )

    if ramp_saved is not None:
        return ramp_saved

    raise ValueError(
        "No ramp in OP file and no ramp params supplied.\n"
        "Provide --margin/--tanh_scale/--smooth_sigma, or ensure ramp is saved."
    )


# -----------------------------------------------------------------------
# Plot helpers (mostly geometry-agnostic)
# -----------------------------------------------------------------------

def _imshow(ax, data, mask, extent, cmap="viridis", **kw):
    arr = np.asarray(data)
    m = np.asarray(mask, dtype=bool)
    arrm = np.ma.masked_where(~m, arr)
    im = ax.imshow(arrm, extent=extent, origin="lower", cmap=cmap, **kw)
    return im

def _style_image_ax(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _add_cbar(fig, ax, im, size="4%", pad=0.05):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=8, length=3)
    return cb


def _finalize_image_grid(fig, suptitle=None):
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
        fig.subplots_adjust(top=0.93)

def make_amplitude_fig(A0, Af, mask_vis, extent, stem):
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.4), constrained_layout=True)

    panels = []
    if A0 is not None:
        panels.append((A0, "amplitude surface A (initial)", "magma"))
    else:
        panels.append((np.zeros_like(mask_vis, dtype=float), "no A field (initial)", "gray"))

    if Af is not None:
        panels.append((Af, "amplitude surface A (final)", "magma"))
    else:
        panels.append((np.zeros_like(mask_vis, dtype=float), "no A field (final)", "gray"))

    for ax, (arr, title, cmap) in zip(axs.flat, panels):
        im = _imshow(ax, arr, mask_vis, extent, cmap=cmap)
        _style_image_ax(ax)
        ax.set_title(title, fontsize=9, pad=6)
        _add_cbar(fig, ax, im)

    _finalize_image_grid(fig, stem)
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

    if A0 is not None:
        A0x = A0[iy_mid, :]
        A0y = A0[:, ix_mid]
    else:
        A0x = np.zeros_like(u0x)
        A0y = np.zeros_like(u0y)

    if Af is not None:
        Afx = Af[iy_mid, :]
        Afy = Af[:, ix_mid]
    else:
        Afx = np.zeros_like(ufx)
        Afy = np.zeros_like(ufy)

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


def make_geometry_fig(
    x, y,
    u0, uf,
    k10, k1f,
    k20, k2f,
    A0, Af,
    ramp, mask_vis,
    mask_pi0, mask_pif,
    extent, stem,
):
    k0   = np.sqrt(k10 ** 2 + k20 ** 2)
    kf   = np.sqrt(k1f ** 2 + k2f ** 2)
    phi0 = np.arctan2(k20, k10)
    phif = np.arctan2(k2f, k1f)

    fig, axs = plt.subplots(4, 4, figsize=(16.5, 14.5), constrained_layout=True)

    if A0 is None:
        A0 = np.zeros_like(u0)
    if Af is None:
        Af = np.zeros_like(uf)

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
            _style_image_ax(ax)
            ax.set_title(title, fontsize=9, pad=5)
            _add_cbar(fig, ax, im, size="3.5%", pad=0.04)
        else:
            ax.axis("off")

    _finalize_image_grid(fig, stem)
    return fig


# -----------------------------------------------------------------------
# K-field diagnostics (no reflection; use mask)
# -----------------------------------------------------------------------

def kfield_diagnostics_masked(k1, k2, x, y, mask_fd, mask_pi_raw):
    """
    Compute k-based diagnostics on the full grid, then mask.

    - Derivatives via np.gradient over the whole field.
    - Good region = mask_fd & (~mask_pi_raw).
    - Outside good region → NaN.
    """
    k1 = np.asarray(k1)
    k2 = np.asarray(k2)
    Ny, Nx = k1.shape

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    mask_ok = mask_fd & (~mask_pi_raw)

    k1_y = np.gradient(k1, dy, axis=0)
    k1_x = np.gradient(k1, dx, axis=1)
    k2_y = np.gradient(k2, dy, axis=0)
    k2_x = np.gradient(k2, dx, axis=1)

    div_k  = k1_x + k2_y
    curl_k = k2_x - k1_y
    J      = k1_x * k2_y - k1_y * k2_x
    E      = (k1_x**2 + k1_y**2 + k2_x**2 + k2_y**2)

    div_k  = np.where(mask_ok, div_k,  np.nan)
    curl_k = np.where(mask_ok, curl_k, np.nan)
    J      = np.where(mask_ok, J,      np.nan)
    E      = np.where(mask_ok, E,      np.nan)

    diag = {
        "div_k":  div_k,
        "curl_k": curl_k,
        "J":      J,
        "E":      E,
    }
    return diag, mask_ok


def make_diagnostics_fig_diamond(diag0, diagf, mask_vis, extent, stem):
    fields = ["curl_k", "div_k", "J", "E"]
    labels = ["curl k", "div k", "J", "E"]
    cmaps = ["coolwarm", "coolwarm", "coolwarm", "hot"]

    fig, axs = plt.subplots(2, 4, figsize=(15.5, 7.8), constrained_layout=True)

    rows = [
        (diag0, "initial"),
        (diagf, "final"),
    ]

    for col, (key, label, cmap) in enumerate(zip(fields, labels, cmaps)):
        for row_idx, (diag, tag) in enumerate(rows):
            arr = diag[key]
            im = _imshow(axs[row_idx, col], arr, mask_vis, extent, cmap=cmap)
            _style_image_ax(axs[row_idx, col])
            axs[row_idx, col].set_title(f"{label} ({tag})", fontsize=9, pad=5)
            _add_cbar(fig, axs[row_idx, col], im, size="4%", pad=0.04)

    _finalize_image_grid(fig, stem)
    return fig


def make_quiver_fig(
    x, y, u,
    k1_raw, k2_raw,
    mask_vis, extent, stem,
    step=12,
):
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


def make_defect_fig(
    x, y, u, k1_raw, k2_raw, diag_full,
    mask_vis,
    extent, stem,
    radius, threshold_rel, min_distance,
):
    X, Y = np.meshgrid(x, y)

    J_field = diag_full["J"]
    curl_field = diag_full["curl_k"]

    twist_int = disk_twist_integrals(
        J_field, X, Y, mask_vis, mask_vis, radius=radius
    )
    circ_int = circle_circulation_integrals(
        np.asarray(k1_raw), np.asarray(k2_raw),
        X, Y, mask_vis, mask_vis, radius=radius
    )

    fields = [
        (J_field,   "J density",                     "coolwarm"),
        (twist_int, f"disk twist (r={radius:.2f})", "coolwarm"),
        (curl_field, "curl k density",              "coolwarm"),
        (circ_int,  f"circle circ (r={radius:.2f})","coolwarm"),
    ]

    fig, axs = plt.subplots(3, 4, figsize=(17.5, 12.5), constrained_layout=True)

    for col, (arr, title, cmap) in enumerate(fields):
        im = _imshow(axs[0, col], arr, mask_vis, extent, cmap=cmap)
        _style_image_ax(axs[0, col])
        axs[0, col].set_title(title, fontsize=9, pad=5)
        _add_cbar(fig, axs[0, col], im, size="3.5%", pad=0.04)

        im2 = _imshow(axs[1, col], arr, mask_vis, extent, cmap=cmap)
        _overlay_peaks(axs[1, col], arr, mask_vis, X, Y,
                       min_distance=min_distance,
                       threshold_rel=threshold_rel,
                       mode="both")
        _style_image_ax(axs[1, col])
        axs[1, col].set_title(f"{title} + peaks", fontsize=8, pad=5)
        _add_cbar(fig, axs[1, col], im2, size="3.5%", pad=0.04)

        im3 = _imshow(axs[2, col], u, mask_vis, extent, cmap="copper")
        _overlay_peaks(axs[2, col], arr, mask_vis, X, Y,
                       min_distance=min_distance,
                       threshold_rel=threshold_rel,
                       mode="both")
        _style_image_ax(axs[2, col])
        axs[2, col].set_title(f"u + peaks from {title}", fontsize=8, pad=5)
        _add_cbar(fig, axs[2, col], im3, size="3.5%", pad=0.04)

    _finalize_image_grid(fig, stem)
    return fig


# -----------------------------------------------------------------------
# Per-file processing
# -----------------------------------------------------------------------

def process_one(
    path, out_dir,
    ramp_thresh, pi_tol,
    margin, tanh_scale, smooth_sigma,
    defect_radius=np.pi/2,
    defect_thresh=0.05,
    defect_min_dist=5,
):
    path    = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  → {path.stem}")

    data = load_op_npz_diamond(path)
    x = data["x"]
    y = data["y"]
    u0 = data["u0"]
    uf = data["uf"]
    k10 = data["k10"]
    k1f = data["k1f"]
    k20 = data["k20"]
    k2f = data["k2f"]
    A0 = data["A0"]
    Af = data["Af"]
    ramp_saved = data["ramp_saved"]
    mu = data["mu"]
    nt = data["nt"]

    ramp = _get_ramp_diamond(
        x, y, mu,
        ramp_saved,
        margin, tanh_scale, smooth_sigma,
    )

    extent   = [x[0], x[-1], y[0], y[-1]]
    mask_vis = ramp >= ramp_thresh
    mask_fd  = ramp >= ramp_thresh

    phi0_raw = np.arctan2(k20, k10)
    phif_raw = np.arctan2(k2f, k1f)

    mask_pi0_raw = phi_jump_mask(phi0_raw, tol=pi_tol)
    mask_pif_raw = phi_jump_mask(phif_raw, tol=pi_tol)

    diag0, mask_ok0 = kfield_diagnostics_masked(
        k10, k20, x, y, mask_fd, mask_pi0_raw
    )
    diagf, mask_okf = kfield_diagnostics_masked(
        k1f, k2f, x, y, mask_fd, mask_pif_raw
    )

    stem = f"{path.stem} | Nt={nt}"

    # Geometry: fields + k, ramp, π-jumps
    fig1 = make_geometry_fig(
        x, y,
        u0, uf,
        k10, k1f,
        k20, k2f,
        A0, Af,
        ramp, mask_vis,
        mask_pi0_raw, mask_pif_raw,
        extent,
        stem,
    )
    fig1.savefig(out_dir / f"{path.stem}_geometry.png", dpi=150)
    plt.close(fig1)

    # Amplitude surfaces: initial + final
    figA = make_amplitude_fig(
        A0, Af,
        mask_vis,
        extent,
        stem,
    )
    figA.savefig(out_dir / f"{path.stem}_amplitude.png", dpi=150)
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
        stem,
    )
    figM.savefig(out_dir / f"{path.stem}_midlines.png", dpi=150)
    plt.close(figM)

    # Diagnostics: initial vs final (masked, no reflection)
    fig2 = make_diagnostics_fig_diamond(
        diag0, diagf,
        mask_vis,
        extent,
        stem,
    )
    fig2.savefig(out_dir / f"{path.stem}_diagnostics.png", dpi=150)
    plt.close(fig2)

    # Quiver: final raw k only
    fig3 = make_quiver_fig(
        x, y, uf,
        k1f, k2f,
        mask_vis,
        extent,
        stem,
    )
    fig3.savefig(out_dir / f"{path.stem}_quiver.png", dpi=150)
    plt.close(fig3)

    # Defects: based on J/curl and final raw k
    fig4 = make_defect_fig(
        x, y, uf,
        k1f, k2f, diagf,
        mask_vis,
        extent, stem,
        radius=defect_radius,
        threshold_rel=defect_thresh,
        min_distance=defect_min_dist,
    )
    fig4.savefig(out_dir / f"{path.stem}_defects.png", dpi=150)
    plt.close(fig4)

    print(
        f"    saved: {path.stem}_geometry.png  {path.stem}_amplitude.png  "
        f"{path.stem}_midlines.png  {path.stem}_diagnostics.png  "
        f"{path.stem}_quiver.png  {path.stem}_defects.png"
    )


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(
        description="K-based diagnostic plots for PGB diamond OP files.",
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
                             "Default: <op_dir>/../figures/k_based_metrics_diamond")
    parser.add_argument("--ramp_thresh", type=float, default=0.99,
                        help="Mask threshold: keep where ramp >= this.")
    parser.add_argument("--pi_tol",      type=float, default=np.pi / 10,
                        help="π-jump tolerance.")
    # optional ramp override: diamond parameters
    parser.add_argument("--margin",      type=float, default=None,
                        help="Rebuild diamond ramp: dimensionless margin fraction.")
    parser.add_argument("--tanh_scale",  type=float, default=None,
                        help="Rebuild diamond ramp: tanh steepness.")
    parser.add_argument("--smooth_sigma", type=float, default=None,
                        help="Rebuild diamond ramp: Gaussian smoothing sigma.")
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
        root_results = _HERE / "results" / "k_based_metrics_diamond"
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
        process_one(
            op_path, out_dir,
            ns.ramp_thresh, ns.pi_tol,
            ns.margin, ns.tanh_scale, ns.smooth_sigma,
            defect_radius=ns.defect_radius,
            defect_thresh=ns.defect_thresh,
            defect_min_dist=ns.defect_min_dist,
        )

    else:
        op_dir = Path(ns.op_dir or ".")
        out_dir = base_out
        files = sorted(op_dir.glob(ns.pattern))
        if not files:
            raise SystemExit(f"No files matching '{ns.pattern}' in {op_dir}")
        print(f"Processing {len(files)} file(s) from {op_dir}")
        for f in files:
            process_one(
                f, out_dir,
                ns.ramp_thresh, ns.pi_tol,
                ns.margin, ns.tanh_scale, ns.smooth_sigma,
                defect_radius=ns.defect_radius,
                defect_thresh=ns.defect_thresh,
                defect_min_dist=ns.defect_min_dist,
            )

    print("Done.")


# -----------------------------------------------------------------------
# Debug / IDE \"press Run\" block — edit paths here, then Run/Debug
# -----------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) == 1:

        class _Args:
            op_file     = None
            op_dir      = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_diamond/uhu/mu_sweeps_full_Ly30pi_Ny384_hp025_tmax3p125_nsave125/sig_pio2/raw"
            pattern     = "*.npz"
            out_dir     = "/Users/edwardmcdugald/patterns/experiments/pgb_analysis/results/k_based_metrics_diamond/mu_sweeps_full_Ly30pi_Ny384_hp025_tmax3p125_nsave125/sig_pio2"
            ramp_thresh = 1 - 1e-12
            pi_tol      = np.pi / 10

            # set all three to rebuild diamond ramp; leave as None to use saved ramp
            margin       = 0.30
            tanh_scale   = 80.0
            smooth_sigma = 1.0

            defect_radius    = np.pi / 2
            defect_thresh    = 0.10
            defect_min_dist  = 10

        a = _Args()
        main([
            "--op_dir",      a.op_dir,
            "--out_dir",     a.out_dir,
            "--pattern",     a.pattern,
            "--ramp_thresh", str(a.ramp_thresh),
            "--pi_tol",      str(a.pi_tol),
            *(["--margin",       str(a.margin),
               "--tanh_scale",   str(a.tanh_scale),
               "--smooth_sigma", str(a.smooth_sigma)]
              if a.margin is not None else []),
        ])
    else:
        main()