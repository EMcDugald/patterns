# experiments/pgb_analysis/phase_midline_and_field_plots_diamond.py
"""
Diamond-only plotting script for existing saved phase/OP results.

This script reads already-saved diamond .npz outputs and generates:
  1) 2D pattern plots (initial/final)
  2) 2D wrapped and unwrapped phase plots (initial/final)
  3) 2D amplitude plots: A and analytic-signal amplitude (initial/final)
  4) midline phase + J overlays for x/y midlines (initial/final)
  5) midline amplitude comparisons: A vs analytic amplitude (initial/final)

Only PNG figures and an auxiliary *_derived_fields.npz are saved.

Supports:
  - single input file
  - whole directory of runs
  - optional recursive search
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- make src/ importable regardless of cwd ---
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from utils.kfield_calcs import orient_vector_field_v2, phi_jump_mask, _central_derivs


# ---------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _maybe_frame(arr, frame_index):
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr[..., frame_index]
    return arr


def _pick_key(d, candidates, required=True):
    for key in candidates:
        if key in d:
            return key
    if required:
        raise KeyError(f"Missing any of keys: {candidates}")
    return None


def _load_optional(d, candidates, frame_index=None):
    key = _pick_key(d, candidates, required=False)
    if key is None:
        return None
    arr = d[key]
    if frame_index is None:
        return arr
    return _maybe_frame(arr, frame_index)


def _masked(arr, mask):
    arr = np.asarray(arr)
    return np.ma.masked_where(~mask, arr)


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


def _imshow(ax, arr, mask, extent, cmap="viridis", **kwargs):
    im = ax.imshow(
        _masked(arr, mask),
        extent=extent,
        origin="lower",
        cmap=cmap,
        **kwargs,
    )
    _style_image_ax(ax)
    return im


def _infer_masks(data, frame_shape, fd_tol=1e-3, vis_tol=0.20):
    """
    Infer a finite-difference mask (mask_fd) and a visualization mask (mask_vis)
    from any available ramp / phase-ramp field in the .npz.
    """
    for key in ["phase_ramp", "ramp_inner", "ramp"]:
        if key in data:
            arr = data[key]
            arr = _maybe_frame(arr, -1)
            arr = np.asarray(arr, dtype=float)
            return arr >= fd_tol, arr >= vis_tol
    ones = np.ones(frame_shape, dtype=bool)
    return ones, ones


def _compute_J_old_style(k1, k2, x, y, mask_fd, pi_tol=np.pi / 10, orient=True):
    """
    Replicate the old pattern_analysis J definition:

      1. Optionally orient the vector field with orient_vector_field_v2
         under mask_fd with a given pi_tol.
      2. Compute phi = atan2(k2, k1) of the oriented field.
      3. Build mask_ok = mask_fd & ~phi_jump_mask(phi, tol=pi_tol)
         & finite(k1) & finite(k2).
      4. Use safe central derivatives (_central_derivs) on k1,k2 with mask_ok.
      5. J = (∂k1/∂x)(∂k2/∂y) − (∂k1/∂y)(∂k2/∂x), masked outside mask_ok.

    Returns
    -------
    J, mask_ok, mask_pi, k1_use, k2_use, phi
    """
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    k1 = np.asarray(k1, dtype=float)
    k2 = np.asarray(k2, dtype=float)
    mask_fd = np.asarray(mask_fd, dtype=bool)

    if orient:
        k1_or, k2_or = orient_vector_field_v2(k1, k2, mask=mask_fd, pi_tol=pi_tol)
        k1_use = np.ma.asarray(k1_or).filled(np.nan)
        k2_use = np.ma.asarray(k2_or).filled(np.nan)
    else:
        k1_use = np.array(k1, dtype=float, copy=True)
        k2_use = np.array(k2, dtype=float, copy=True)

    phi = np.arctan2(k2_use, k1_use)
    mask_pi = phi_jump_mask(phi, tol=pi_tol)
    mask_ok = mask_fd & (~mask_pi) & np.isfinite(k1_use) & np.isfinite(k2_use)

    k1_x, k1_y = _central_derivs(k1_use, dx, dy, mask_ok)
    k2_x, k2_y = _central_derivs(k2_use, dx, dy, mask_ok)

    J = k1_x * k2_y - k1_y * k2_x
    J = np.where(mask_ok, J, np.nan)

    return J, mask_ok, mask_pi, k1_use, k2_use, phi


def _midline_indices(x, y):
    ix0 = int(np.argmin(np.abs(x - 0.0)))
    iy0 = int(np.argmin(np.abs(y - 0.0)))
    return ix0, iy0


def _central_window_mask(coord, frac=0.5):
    coord = np.asarray(coord)
    if not (0.0 < frac <= 1.0):
        raise ValueError("frac must be in (0, 1].")
    cmin = float(coord[0])
    cmax = float(coord[-1])
    L = cmax - cmin
    pad = 0.5 * (1.0 - frac) * L
    lo = cmin + pad
    hi = cmax - pad
    return (coord >= lo) & (coord <= hi)


def _restrict_line(coord, vals, mask_1d, side=None, center_frac=0.5, min_points=8):
    """
    Restrict a 1D cut to the central window and a given side, but fall back
    to the full trusted midline if the central window is too sparse.
    """
    coord = np.asarray(coord)
    vals = np.asarray(vals)
    mask_1d = np.asarray(mask_1d, dtype=bool)

    keep_full = mask_1d.copy()
    keep = keep_full & _central_window_mask(coord, frac=center_frac)

    if side == "right":
        keep &= coord >= 0.0
        keep_full &= coord >= 0.0
    elif side == "left":
        keep &= coord <= 0.0
        keep_full &= coord <= 0.0
    elif side == "upper":
        keep &= coord >= 0.0
        keep_full &= coord >= 0.0
    elif side == "lower":
        keep &= coord <= 0.0
        keep_full &= coord <= 0.0

    if np.count_nonzero(keep) < min_points:
        keep = keep_full

    return coord[keep], vals[keep]


def _set_comparable_J_scale(ax_phase, ax_J, phi_line, J_line, pad_frac=0.08):
    phi_line = np.asarray(phi_line)
    J_line = np.asarray(J_line)

    phi_finite = phi_line[np.isfinite(phi_line)]
    J_finite = J_line[np.isfinite(J_line)]

    if phi_finite.size == 0 or J_finite.size == 0:
        return

    phi_min = float(np.nanmin(phi_finite))
    phi_max = float(np.nanmax(phi_finite))
    J_min = float(np.nanmin(J_finite))
    J_max = float(np.nanmax(J_finite))

    phi_span = phi_max - phi_min
    J_span = J_max - J_min

    if phi_span <= 0:
        phi_span = 1.0
    if J_span <= 0:
        j0 = 1.0 if J_max == 0 else abs(J_max)
        J_min = -j0
        J_max = j0
        J_span = J_max - J_min

    phi_pad = pad_frac * phi_span
    J_pad = pad_frac * J_span

    ax_phase.set_ylim(phi_min - phi_pad, phi_max + phi_pad)
    ax_J.set_ylim(J_min - J_pad, J_max + J_pad)


# ---------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------

def load_diamond_run(path, prefer_sym=True, mask_tol=0.20):
    path = Path(path)
    d = np.load(path, allow_pickle=True)

    x = np.asarray(d["x"])
    y = np.asarray(d["y"])
    u = np.asarray(d["u"])

    if u.ndim == 2:
        u = u[:, :, None]

    u0 = u[..., 0]
    uf = u[..., -1]

    A0 = _load_optional(d, ["A"], frame_index=0)
    Af = _load_optional(d, ["A"], frame_index=-1)

    if prefer_sym:
        k1_candidates = ["k1_sym", "k1"]
        k2_candidates = ["k2_sym", "k2"]
        phiw_candidates = ["phase_grid_symmetric_wrapped", "phase_grid_wrapped"]
        phiu_candidates = ["phase_grid_symmetric_unwrapped", "phase_grid_unwrapped"]
        amp_candidates = ["analytic_amplitude_grid_symmetric", "analytic_amplitude_grid"]
    else:
        k1_candidates = ["k1_orig", "k1", "k1_sym"]
        k2_candidates = ["k2_orig", "k2", "k2_sym"]
        phiw_candidates = ["phase_grid_wrapped", "phase_grid_symmetric_wrapped"]
        phiu_candidates = ["phase_grid_unwrapped", "phase_grid_symmetric_unwrapped"]
        amp_candidates = ["analytic_amplitude_grid", "analytic_amplitude_grid_symmetric"]

    k10 = _load_optional(d, k1_candidates, frame_index=0)
    k1f = _load_optional(d, k1_candidates, frame_index=-1)
    k20 = _load_optional(d, k2_candidates, frame_index=0)
    k2f = _load_optional(d, k2_candidates, frame_index=-1)

    phiw0 = _load_optional(d, phiw_candidates, frame_index=0)
    phiwf = _load_optional(d, phiw_candidates, frame_index=-1)
    phiu0 = _load_optional(d, phiu_candidates, frame_index=0)
    phiuf = _load_optional(d, phiu_candidates, frame_index=-1)

    amp_h0 = _load_optional(d, amp_candidates, frame_index=0)
    amp_hf = _load_optional(d, amp_candidates, frame_index=-1)

    mask_fd, mask_vis = _infer_masks(d, u0.shape, fd_tol=1e-3, vis_tol=mask_tol)

    J0 = Jf = None
    maskJ0 = maskJf = None
    mask_pi0 = mask_pif = None
    k10_use = k20_use = None
    k1f_use = k2f_use = None
    phi_k0 = phi_kf = None

    if k10 is not None and k20 is not None:
        J0, maskJ0, mask_pi0, k10_use, k20_use, phi_k0 = _compute_J_old_style(
            k10, k20, x, y, mask_fd, pi_tol=np.pi / 10, orient=True
        )

    if k1f is not None and k2f is not None:
        Jf, maskJf, mask_pif, k1f_use, k2f_use, phi_kf = _compute_J_old_style(
            k1f, k2f, x, y, mask_fd, pi_tol=np.pi / 10, orient=True
        )

    return {
        "path": path,
        "x": x,
        "y": y,
        "u0": u0,
        "uf": uf,
        "A0": A0,
        "Af": Af,
        "phiw0": phiw0,
        "phiwf": phiwf,
        "phiu0": phiu0,
        "phiuf": phiuf,
        "amp_h0": amp_h0,
        "amp_hf": amp_hf,
        "J0": J0,
        "Jf": Jf,
        "mask_fd": mask_fd,
        "mask_vis": mask_vis,
        "maskJ0": maskJ0,
        "maskJf": maskJf,
        "mask_pi0": mask_pi0,
        "mask_pif": mask_pif,
        "k10_use": k10_use,
        "k20_use": k20_use,
        "k1f_use": k1f_use,
        "k2f_use": k2f_use,
        "phi_k0": phi_k0,
        "phi_kf": phi_kf,
        "extent": [x[0], x[-1], y[0], y[-1]],
    }


def save_derived_fields_npz(run, out_path):
    out_path = Path(out_path)

    J0 = run["J0"] if run["J0"] is not None else np.full_like(run["u0"], np.nan, dtype=float)
    Jf = run["Jf"] if run["Jf"] is not None else np.full_like(run["uf"], np.nan, dtype=float)

    maskJ0 = run["maskJ0"] if run["maskJ0"] is not None else np.zeros_like(run["mask_vis"], dtype=bool)
    maskJf = run["maskJf"] if run["maskJf"] is not None else np.zeros_like(run["mask_vis"], dtype=bool)
    mask_pi0 = run["mask_pi0"] if run["mask_pi0"] is not None else np.zeros_like(run["mask_vis"], dtype=bool)
    mask_pif = run["mask_pif"] if run["mask_pif"] is not None else np.zeros_like(run["mask_vis"], dtype=bool)

    np.savez_compressed(
        out_path,
        x=run["x"],
        y=run["y"],
        mask_fd=run["mask_fd"],
        mask_vis=run["mask_vis"],
        maskJ0=maskJ0,
        maskJf=maskJf,
        mask_pi0=mask_pi0,
        mask_pif=mask_pif,
        J0=J0,
        Jf=Jf,
        phi_k0=run["phi_k0"] if run["phi_k0"] is not None else np.full_like(run["u0"], np.nan, dtype=float),
        phi_kf=run["phi_kf"] if run["phi_kf"] is not None else np.full_like(run["uf"], np.nan, dtype=float),
        k10_oriented=run["k10_use"] if run["k10_use"] is not None else np.full_like(run["u0"], np.nan, dtype=float),
        k20_oriented=run["k20_use"] if run["k20_use"] is not None else np.full_like(run["u0"], np.nan, dtype=float),
        k1f_oriented=run["k1f_use"] if run["k1f_use"] is not None else np.full_like(run["uf"], np.nan, dtype=float),
        k2f_oriented=run["k2f_use"] if run["k2f_use"] is not None else np.full_like(run["uf"], np.nan, dtype=float),
        J_method_primary=np.array("old_style_safe_central_oriented", dtype=object),
        J_desc=np.array(
            "J = det(grad k) computed from oriented k-field using phi_jump_mask "
            "and safe central derivatives, matching old pattern_analysis logic.",
            dtype=object,
        ),
    )


# ---------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------

def make_pattern_2d_fig(run, stem):
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.4), constrained_layout=True)

    u0 = run["u0"]
    uf = run["uf"]
    mask = run["mask_vis"]
    extent = run["extent"]

    vmax = np.nanmax(np.abs([u0, uf]))
    vmax = 1.0 if not np.isfinite(vmax) or vmax == 0 else vmax

    for ax, arr, title in zip(
        axs,
        [u0, uf],
        ["pattern u (initial)", "pattern u (final)"],
    ):
        im = _imshow(ax, arr, mask, extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=9, pad=6)
        _add_cbar(fig, ax, im)

    fig.suptitle(stem, fontsize=10)
    return fig


def make_phase_2d_fig(run, stem):
    fig, axs = plt.subplots(2, 2, figsize=(10.2, 8.2), constrained_layout=True)

    mask = run["mask_vis"]
    extent = run["extent"]

    panels = [
        (run["phiw0"], "wrapped phase (initial)", "twilight", dict(vmin=-np.pi, vmax=np.pi)),
        (run["phiwf"], "wrapped phase (final)", "twilight", dict(vmin=-np.pi, vmax=np.pi)),
        (run["phiu0"], "unwrapped phase (initial)", "twilight", {}),
        (run["phiuf"], "unwrapped phase (final)", "twilight", {}),
    ]

    for ax, (arr, title, cmap, kw) in zip(axs.flat, panels):
        if arr is None:
            ax.axis("off")
            ax.set_title(f"{title} [missing]", fontsize=9)
            continue
        im = _imshow(ax, arr, mask, extent, cmap=cmap, **kw)
        ax.set_title(title, fontsize=9, pad=5)
        _add_cbar(fig, ax, im, size="3.5%", pad=0.04)

    fig.suptitle(stem, fontsize=10)
    return fig


def make_amplitude_2d_fig(run, stem):
    fig, axs = plt.subplots(2, 2, figsize=(10.2, 8.2), constrained_layout=True)

    mask = run["mask_vis"]
    extent = run["extent"]

    panels = [
        (run["A0"], "A amplitude (initial)", "magma"),
        (run["Af"], "A amplitude (final)", "magma"),
        (run["amp_h0"], "analytic-signal amplitude (initial)", "magma"),
        (run["amp_hf"], "analytic-signal amplitude (final)", "magma"),
    ]

    finite_vals = []
    for arr, _, _ in panels:
        if arr is not None:
            finite_vals.append(np.asarray(arr)[mask])
    if finite_vals:
        pieces = []
        for v in finite_vals:
            v = np.asarray(v)
            if v.size == 0:
                continue
            vf = v[np.isfinite(v)]
            if vf.size > 0:
                pieces.append(vf)
        allv = np.concatenate(pieces) if pieces else np.array([], dtype=float)
        vmax = np.nanmax(allv) if allv.size else 1.0
    else:
        vmax = 1.0
    vmax = 1.0 if not np.isfinite(vmax) or vmax == 0 else vmax

    for ax, (arr, title, cmap) in zip(axs.flat, panels):
        if arr is None:
            ax.axis("off")
            ax.set_title(f"{title} [missing]", fontsize=9)
            continue
        im = _imshow(ax, arr, mask, extent, cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(title, fontsize=9, pad=5)
        _add_cbar(fig, ax, im, size="3.5%", pad=0.04)

    fig.suptitle(stem, fontsize=10)
    return fig


def make_midlines_phase_J_fig(run, stem):
    x = run["x"]
    y = run["y"]
    ix0, iy0 = _midline_indices(x, y)

    maskJ0 = run["maskJ0"] if run["maskJ0"] is not None else run["mask_vis"]
    maskJf = run["maskJf"] if run["maskJf"] is not None else run["mask_vis"]

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)

    jobs = [
        ("x-midline initial", x, run["phiu0"], run["J0"], maskJ0[iy0, :], ("row", iy0)),
        ("x-midline final",   x, run["phiuf"], run["Jf"], maskJf[iy0, :], ("row", iy0)),
        ("y-midline initial", y, run["phiu0"], run["J0"], maskJ0[:, ix0], ("col", ix0)),
        ("y-midline final",   y, run["phiuf"], run["Jf"], maskJf[:, ix0], ("col", ix0)),
    ]

    for ax, (title, coord, phi2d, J2d, mask1d, loc) in zip(axs.flat, jobs):
        if phi2d is None or J2d is None:
            ax.text(0.5, 0.5, "missing phase or J", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=9)
            continue

        if loc[0] == "row":
            idx = loc[1]
            phi1d = phi2d[idx, :]
            J1d = J2d[idx, :]
            xlabel = "x"
        else:
            idx = loc[1]
            phi1d = phi2d[:, idx]
            J1d = J2d[:, idx]
            xlabel = "y"

        c, phi_line = _restrict_line(coord, phi1d, mask1d, side=None, center_frac=0.5)
        _, J_line = _restrict_line(coord, J1d, mask1d, side=None, center_frac=0.5)

        phi_finite = np.isfinite(phi_line)
        J_finite = np.isfinite(J_line)

        if c.size == 0 or phi_finite.sum() == 0:
            ax.text(0.5, 0.5, "no trusted midline points", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=9)
            continue

        ax.plot(c, phi_line, color="black", lw=1.8, label="phase (unwrapped)")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("phase", color="black")
        ax.tick_params(axis="y", labelcolor="black")
        ax.grid(alpha=0.25)

        ax2 = ax.twinx()
        if J_finite.sum() > 0:
            ax2.plot(c, J_line, color="C3", lw=1.6, label="J")
            _set_comparable_J_scale(ax, ax2, phi_line, J_line)
        else:
            ax2.text(0.5, 0.08, "no valid J on this cut", ha="center", va="bottom",
                     transform=ax2.transAxes, color="C3", fontsize=8)

        ax2.set_ylabel("J", color="C3")
        ax2.tick_params(axis="y", labelcolor="C3")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    fig.suptitle(stem, fontsize=10)
    return fig


def make_midlines_amplitude_fig(run, stem):
    x = run["x"]
    y = run["y"]
    ix0, iy0 = _midline_indices(x, y)
    mask = run["mask_vis"]

    fig, axs = plt.subplots(2, 2, figsize=(11, 7.8), constrained_layout=True)

    jobs = [
        ("x-midline initial", x, run["A0"], run["amp_h0"], mask[iy0, :], ("row", iy0)),
        ("x-midline final",   x, run["Af"], run["amp_hf"], mask[iy0, :], ("row", iy0)),
        ("y-midline initial", y, run["A0"], run["amp_h0"], mask[:, ix0], ("col", ix0)),
        ("y-midline final",   y, run["Af"], run["amp_hf"], mask[:, ix0], ("col", ix0)),
    ]

    for ax, (title, coord, A2d, H2d, mask1d, loc) in zip(axs.flat, jobs):
        if A2d is None and H2d is None:
            ax.text(0.5, 0.5, "missing amplitude fields", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=9)
            continue

        if loc[0] == "row":
            idx = loc[1]
            A1d = None if A2d is None else A2d[idx, :]
            H1d = None if H2d is None else H2d[idx, :]
            xlabel = "x"
        else:
            idx = loc[1]
            A1d = None if A2d is None else A2d[:, idx]
            H1d = None if H2d is None else H2d[:, idx]
            xlabel = "y"

        if A1d is not None:
            c, Aline = _restrict_line(coord, A1d, mask1d, side=None, center_frac=0.5)
            ax.plot(c, Aline, color="C0", lw=1.8, label="A")
        else:
            c = coord[mask1d]

        if H1d is not None:
            c2, Hline = _restrict_line(coord, H1d, mask1d, side=None, center_frac=0.5)
            ax.plot(c2, Hline, color="C1", lw=1.8, label="analytic amplitude")

        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("amplitude")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(stem, fontsize=10)
    return fig


# ---------------------------------------------------------------------
# per-file processing
# ---------------------------------------------------------------------

def process_one(path, out_root, prefer_sym=True, mask_tol=0.20):
    path = Path(path)
    run = load_diamond_run(path, prefer_sym=prefer_sym, mask_tol=mask_tol)

    run_dir = ensure_dir(Path(out_root) / path.stem)
    stem = path.stem

    derived_npz = run_dir / f"{path.stem}_derived_fields.npz"
    save_derived_fields_npz(run, derived_npz)

    fig = make_pattern_2d_fig(run, stem)
    fig.savefig(run_dir / f"{path.stem}_pattern_2d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig = make_phase_2d_fig(run, stem)
    fig.savefig(run_dir / f"{path.stem}_phase_2d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig = make_amplitude_2d_fig(run, stem)
    fig.savefig(run_dir / f"{path.stem}_amplitude_2d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig = make_midlines_phase_J_fig(run, stem)
    fig.savefig(run_dir / f"{path.stem}_midlines_phase_J.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig = make_midlines_amplitude_fig(run, stem)
    fig.savefig(run_dir / f"{path.stem}_midlines_amplitude.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"  saved figures for {path.name} -> {run_dir}")


# ---------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(
        description="Diamond-only batch plotting for saved phase/OP runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_file", type=str, default=None,
                        help="One saved diamond .npz file.")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing saved diamond .npz files.")
    parser.add_argument("--pattern", type=str, default="*.npz",
                        help="Filename glob pattern inside input_dir.")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively search input_dir with rglob().")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for PNG figures.")
    parser.add_argument("--prefer_sym", action="store_true", default=True,
                        help="Prefer symmetric phase/k/amplitude fields when available.")
    parser.add_argument("--mask_tol", type=float, default=0.20,
                        help="Threshold for ramp/phase-ramp visibility mask.")

    ns = parser.parse_args(args)

    if ns.out_dir is not None:
        out_root = Path(ns.out_dir)
    else:
        out_root = _HERE / "results" / "phase_midline_and_field_plots_diamond"

    if ns.input_file is not None:
        files = [Path(ns.input_file)]
    elif ns.input_dir is not None:
        root = Path(ns.input_dir)
        files = sorted(root.rglob(ns.pattern) if ns.recursive else root.glob(ns.pattern))
    else:
        raise SystemExit("Provide --input_file or --input_dir.")

    if not files:
        raise SystemExit("No matching .npz files found.")

    print(f"Processing {len(files)} file(s)")
    for path in files:
        try:
            process_one(
                path,
                out_root=out_root,
                prefer_sym=ns.prefer_sym,
                mask_tol=ns.mask_tol,
            )
        except Exception as exc:
            print(f"  FAILED: {path}\n    {type(exc).__name__}: {exc}")

    print("Done.")


# ---------------------------------------------------------------------
# debug / IDE block
# ---------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) == 1:
        class _Args:
            input_file = None
            input_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_diamond/phase/mu_sweeps_full_Ly30pi_Ny384_hp025_tmax3p125_nsave125/sig_pio2/raw"
            pattern = "*.npz"
            recursive = False
            out_dir = "/Users/edwardmcdugald/patterns/experiments/pgb_analysis/results/phase_midline_and_field_plots_diamond/mu_sweeps_full_Ly30pi_Ny384_hp025_tmax3p125_nsave125/sig_pio2"
            prefer_sym = True
            mask_tol = 0.20

        a = _Args()
        cli = [
            "--input_dir", a.input_dir,
            "--pattern", a.pattern,
            "--out_dir", a.out_dir,
            "--mask_tol", str(a.mask_tol),
        ]
        if a.recursive:
            cli.append("--recursive")
        if a.prefer_sym:
            cli.append("--prefer_sym")
        main(cli)
    else:
        main()