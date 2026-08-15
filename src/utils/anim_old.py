# src/utils/anim.py
import os
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

def make_scalar_gif(data, out_dir, name, fps=10, cmap="gray", mask=None):
    """
    data: (ny, nx, nt)
    mask: (ny, nx) boolean or float array; True/1 means keep, False/0 means mask.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if mask is not None:
        # broadcast to 3D and mask out
        if mask.ndim == 2:
            mask3 = mask[:, :, None]
        else:
            mask3 = mask
        data = np.where(mask3, data, np.nan)

    frames = []
    for idx in range(data.shape[2]):
        print("making frame: ", idx)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(data[:, :, idx], cmap=cmap, origin="lower")
        ax.set_axis_off()
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    out_path = out_dir / f"{name}.gif"
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Saved scalar GIF → {out_path}")



def make_director_gif(orientation, u, out_dir, name, fps=10, step=12, cmap="gray"):
    """
    orientation: (ny, nx, nt) angle field
    u:          (ny, nx, nt) intensity field
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for idx in range(orientation.shape[2]):
        print("making frame: ", idx)
        orient = orientation[:, :, idx]
        u_frame = u[:, :, idx]

        Y, X = np.mgrid[0:u_frame.shape[0], 0:u_frame.shape[1]]
        nx = np.cos(orient)
        ny = np.sin(orient)

        X_s, Y_s = X[::step, ::step], Y[::step, ::step]
        nx_s, ny_s = nx[::step, ::step], ny[::step, ::step]
        mag = np.sqrt(nx_s**2 + ny_s**2) + 1e-12
        nx_s /= mag
        ny_s /= mag

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(u_frame, cmap=cmap, origin="lower")
        ax.quiver(X_s, Y_s, nx_s, ny_s, color="cyan", pivot="middle", scale=30)
        ax.quiver(X_s, Y_s, -nx_s, -ny_s, color="cyan", pivot="middle", scale=30)
        ax.set_axis_off()
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    out_path = out_dir / f"{name}.gif"
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Saved director GIF → {out_path}")


def make_director_quiver_frame(
    orientation, u_frame, x, y, fig_path,
    step=12, cmap="gray", mask=None, title=None
):
    """
    Static director quiver plot.

    orientation : (ny, nx) angle field
    u_frame     : (ny, nx) intensity/pattern field
    x, y        : 1D arrays of physical coordinates (length nx, ny)
    mask        : (ny, nx) boolean; True → keep arrow, False → skip
    """
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    # grid in physical coordinates
    X, Y = np.meshgrid(x, y)

    nx = np.cos(orientation)
    ny = np.sin(orientation)

    # subsample
    X_s = X[::step, ::step]
    Y_s = Y[::step, ::step]
    nx_s = nx[::step, ::step]
    ny_s = ny[::step, ::step]

    if mask is not None:
        mask_s = mask[::step, ::step]
        keep = mask_s.astype(bool)
        X_s  = X_s[keep]
        Y_s  = Y_s[keep]
        nx_s = nx_s[keep]
        ny_s = ny_s[keep]

    # normalize for unit-length arrows (director)
    mag = np.sqrt(nx_s**2 + ny_s**2) + 1e-12
    nx_s /= mag
    ny_s /= mag

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        u_frame,
        cmap=cmap,
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
    )

    # double-headed director arrows
    ax.quiver(X_s, Y_s, nx_s, ny_s,  color="cyan", pivot="middle", scale=30)
    ax.quiver(X_s, Y_s, -nx_s, -ny_s, color="cyan", pivot="middle", scale=30)

    ax.set_axis_off()
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved director quiver → {fig_path}")


def make_wavevector_quiver_frame(
    k1_frame, k2_frame, u_frame, x, y, fig_path,
    step=12, cmap="gray", mask=None, title=None, scale=None
):
    """
    Static wave-vector quiver plot.

    k1_frame, k2_frame : (ny, nx) wave-vector components
    u_frame            : (ny, nx) intensity/pattern field
    x, y               : 1D arrays of physical coordinates (length nx, ny)
    mask               : (ny, nx) boolean; True → keep arrow, False → skip
    scale              : quiver scale; None uses raw lengths (|k| in data units)

    Arrow length reflects |k|, so deviations from norm ~1 are visible.
    """
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    X, Y = np.meshgrid(x, y)

    # subsample
    X_s = X[::step, ::step]
    Y_s = Y[::step, ::step]
    k1_s = k1_frame[::step, ::step]
    k2_s = k2_frame[::step, ::step]

    if mask is not None:
        mask_s = mask[::step, ::step]
        keep = mask_s.astype(bool)
        X_s  = X_s[keep]
        Y_s  = Y_s[keep]
        k1_s = k1_s[keep]
        k2_s = k2_s[keep]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        u_frame,
        cmap=cmap,
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
    )

    ax.quiver(
        X_s, Y_s, k1_s, k2_s,
        color="cyan",
        pivot="tail",       # arrow tail at (x, y), head along k
        scale=scale,        # None → raw lengths; can tune if arrows too long
        scale_units="xy",
        angles="xy",
    )

    ax.set_axis_off()
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved wave-vector quiver → {fig_path}")


def _pick_frame_indices(nt, n_panels):
    """
    Choose n_panels frame indices between 0 and nt-1 (inclusive),
    spaced as evenly as possible.
    """
    n_panels = max(1, min(n_panels, nt))
    return list(np.linspace(0, nt - 1, n_panels, dtype=int))


def make_director_quiver_panels(
    orientation, u, x, y, fig_path,
    n_panels=2, step=12, cmap="gray", mask=None, suptitle=None
):
    """
    Multi-panel director quiver plot on a single figure.

    orientation : (ny, nx, nt) angle field
    u           : (ny, nx, nt) intensity/pattern field
    x, y        : 1D arrays of physical coordinates (length nx, ny)
    mask        : (ny, nx) boolean; True → keep arrow, False → skip
    n_panels    : number of time snapshots to plot (default 2: first + last)
    """
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    ny, nx, nt = orientation.shape
    frame_indices = _pick_frame_indices(nt, n_panels)

    fig, axs = plt.subplots(
        1, len(frame_indices),
        figsize=(6 * len(frame_indices), 6),
        squeeze=False,
    )
    axs = axs[0]  # 1 row

    # grid in physical coordinates (shared across panels)
    X, Y = np.meshgrid(x, y)

    for ax, fi in zip(axs, frame_indices):
        orient_f = orientation[:, :, fi]
        u_f      = u[:, :, fi]

        nx = np.cos(orient_f)
        ny = np.sin(orient_f)

        # subsample
        X_s = X[::step, ::step]
        Y_s = Y[::step, ::step]
        nx_s = nx[::step, ::step]
        ny_s = ny[::step, ::step]

        if mask is not None:
            mask_s = mask[::step, ::step]
            keep = mask_s.astype(bool)
            X_s  = X_s[keep]
            Y_s  = Y_s[keep]
            nx_s = nx_s[keep]
            ny_s = ny_s[keep]

        # normalize for unit-length arrows (director)
        mag = np.sqrt(nx_s**2 + ny_s**2) + 1e-12
        nx_s /= mag
        ny_s /= mag

        im = ax.imshow(
            u_f,
            cmap=cmap,
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
        )

        ax.quiver(X_s, Y_s, nx_s, ny_s,  color="cyan", pivot="middle", scale=30)
        ax.quiver(X_s, Y_s, -nx_s, -ny_s, color="cyan", pivot="middle", scale=30)

        ax.set_axis_off()
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(f"t index = {fi}")

    if suptitle is not None:
        fig.suptitle(suptitle, y=0.98)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved director quiver panels → {fig_path}")


def make_wavevector_quiver_panels(
    k1, k2, u, x, y, fig_path,
    n_panels=2, step=12, cmap="gray", mask=None, suptitle=None, scale=None
):
    """
    Multi-panel wave-vector quiver plot on a single figure.

    k1, k2 : (ny, nx, nt) wave-vector components
    u      : (ny, nx, nt) intensity/pattern field
    x, y   : 1D arrays of physical coordinates (length nx, ny)
    mask   : (ny, nx) boolean; True → keep arrow, False → skip
    scale  : quiver scale; None uses raw lengths (|k| in data units)
    """
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    ny, nx, nt = k1.shape
    frame_indices = _pick_frame_indices(nt, n_panels)

    fig, axs = plt.subplots(
        1, len(frame_indices),
        figsize=(6 * len(frame_indices), 6),
        squeeze=False,
    )
    axs = axs[0]

    X, Y = np.meshgrid(x, y)

    for ax, fi in zip(axs, frame_indices):
        k1_f = k1[:, :, fi]
        k2_f = k2[:, :, fi]
        u_f  = u[:, :, fi]

        # subsample
        X_s = X[::step, ::step]
        Y_s = Y[::step, ::step]
        k1_s = k1_f[::step, ::step]
        k2_s = k2_f[::step, ::step]

        if mask is not None:
            mask_s = mask[::step, ::step]
            keep = mask_s.astype(bool)
            X_s  = X_s[keep]
            Y_s  = Y_s[keep]
            k1_s = k1_s[keep]
            k2_s = k2_s[keep]

        im = ax.imshow(
            u_f,
            cmap=cmap,
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
        )

        ax.quiver(
            X_s, Y_s, k1_s, k2_s,
            color="cyan",
            pivot="tail",
            scale=scale,        # None → raw lengths
            scale_units="xy",
            angles="xy",
        )

        ax.set_axis_off()
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"t index = {fi}")

    if suptitle is not None:
        fig.suptitle(suptitle, y=0.98)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved wave-vector quiver panels → {fig_path}")