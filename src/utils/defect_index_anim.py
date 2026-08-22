# src/utils/defect_index_anim.py
"""
GIF makers for the defect index (t1_common.detect_defects), the animated
counterpart to t1_common.defect_index_figure.

Three functions:
  make_defect_overlay_gif  pattern + red/blue defect markers, one sigma_det
  make_charge_map_gif      the raw charge map alone, no pattern underneath
  make_defect_sigma_sweep_gif  multi-panel, several sigma_det values at once

All three take an `sh` dict as returned by run_riesz_zigzag.load_sh_npz
(u, x, y already centered so the core sits at y = 0), so nothing here
re-derives grid centering or crop extents; that stays in one place.

No new defect-detection logic lives here. Every frame just calls
t1_common.detect_defects and draws its output; the counting, the winding
number, the band restriction, all of that is unchanged from what the
static per-frame figures already use.
"""

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from op_extract.t1_common import detect_defects


def _frame_from_fig(fig):
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def _overlay_markers(ax, det, ms=7):
    ch = det["charges"]
    xp = [c[0] for c in ch if c[2] > 0]
    yp = [c[1] for c in ch if c[2] > 0]
    xm = [c[0] for c in ch if c[2] < 0]
    ym = [c[1] for c in ch if c[2] < 0]
    ax.plot(xp, yp, "o", ms=ms, mfc="none", mec="red", mew=1.6)
    ax.plot(xm, ym, "s", ms=ms, mfc="none", mec="dodgerblue", mew=1.6)


def make_defect_overlay_gif(sh, out_dir, name, sigma_det=0.4, band_y=2.0,
                            band_x=None, fps=10, cmap="gray", vmin=-1,
                            vmax=1, figsize=(6, 6)):
    """
    Pattern with +1 (red circle) / -1 (blue square) defect markers overlaid,
    one frame per timestep. Same visual as defect_index_figure's top panel,
    animated. Fixed color scale on u (vmin/vmax) so amplitude drift over
    time stays visible rather than being normalized away each frame.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    u, x, y, Lx, Ly = sh["u"], sh["x"], sh["y"], sh["Lx"], sh["Ly"]
    tt = sh.get("tt")
    ext = [x[0], x[-1], y[0], y[-1]]
    nt = u.shape[2]

    frames = []
    for idx in range(nt):
        print("making frame:", idx)
        det = detect_defects(u[:, :, idx], x, y, Lx, Ly,
                             sigma_det=sigma_det, band_y=band_y,
                             band_x=band_x)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(u[:, :, idx], extent=ext, origin="lower", cmap=cmap,
                  vmin=vmin, vmax=vmax, aspect="auto")
        _overlay_markers(ax, det)
        ax.set_axis_off()
        label = f"t = {tt[idx]:.3f}" if tt is not None else f"frame {idx}"
        ax.text(0.02, 0.98, label, transform=ax.transAxes, color="yellow",
                fontsize=12, va="top", ha="left",
                bbox=dict(facecolor="black", alpha=0.5, pad=2, lw=0))
        ax.text(0.02, 0.90,
                f"dislocations = {det['n_dislocations']}  "
                f"(sigma_det={sigma_det:.2f})",
                transform=ax.transAxes, color="lime", fontsize=9,
                va="top", ha="left",
                bbox=dict(facecolor="black", alpha=0.5, pad=3, lw=0))
        frames.append(_frame_from_fig(fig))

    out_path = out_dir / f"{name}.gif"
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Saved defect overlay GIF -> {out_path}")
    return out_path


def make_charge_map_gif(sh, out_dir, name, sigma_det=0.4, band_y=2.0,
                        band_x=None, fps=10, figsize=(6, 6),
                        show_full_grid=False):
    """
    The raw charge map alone: colored dots at nonzero plaquettes on a
    blank background, no pattern underneath. show_full_grid=True also
    plots every zero-charge plaquette as a faint gray dot, for a sense
    of grid coverage; off by default since it is mostly empty space.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    u, x, y, Lx, Ly = sh["u"], sh["x"], sh["y"], sh["Lx"], sh["Ly"]
    tt = sh.get("tt")
    ext = [x[0], x[-1], y[0], y[-1]]
    nt = u.shape[2]

    frames = []
    for idx in range(nt):
        print("making frame:", idx)
        det = detect_defects(u[:, :, idx], x, y, Lx, Ly,
                             sigma_det=sigma_det, band_y=band_y,
                             band_x=band_x)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("white")
        if show_full_grid:
            Xc, Yc = np.meshgrid(det["xc"], det["yc"])
            ax.plot(Xc.ravel(), Yc.ravel(), ".", ms=1, color="0.85",
                    zorder=0)
        _overlay_markers(ax, det, ms=8)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_axis_off()
        label = f"t = {tt[idx]:.3f}" if tt is not None else f"frame {idx}"
        ax.text(0.02, 0.98, label, transform=ax.transAxes, color="black",
                fontsize=12, va="top", ha="left")
        ax.text(0.02, 0.92,
                f"sum|q| = {det['sum_abs']}   dislocations = "
                f"{det['n_dislocations']}",
                transform=ax.transAxes, color="black", fontsize=10,
                va="top", ha="left")
        frames.append(_frame_from_fig(fig))

    out_path = out_dir / f"{name}.gif"
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Saved charge map GIF -> {out_path}")
    return out_path


def make_full_domain_charge_map_gif(sh, out_dir, name, sigma_det=0.4,
                                    fps=10, cmap="RdBu_r", figsize=(6, 6),
                                    show_band=True, band_y=2.0,
                                    band_x=None):
    """
    The raw plaquette winding q over the WHOLE domain, one sigma_det,
    as an image rather than a scatter of band-restricted points. q is 0
    almost everywhere and +-1 at a disclination, no thresholding: this
    is det["q"] itself, direct from director_winding, nothing dropped.

    show_band draws a dotted rectangle at (band_y, band_x) so you can
    see, on the same picture, how much of the raw field the dislocation
    COUNT (as opposed to this map) actually looks at. It is just a
    reference line; it does not hide or restrict anything drawn here.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    u, x, y, Lx, Ly = sh["u"], sh["x"], sh["y"], sh["Lx"], sh["Ly"]
    tt = sh.get("tt")
    nt = u.shape[2]

    frames = []
    for idx in range(nt):
        print("making frame:", idx)
        det = detect_defects(u[:, :, idx], x, y, Lx, Ly,
                             sigma_det=sigma_det, band_y=band_y,
                             band_x=band_x)
        xc, yc, q = det["xc"], det["yc"], det["q"]
        ext = [xc[0], xc[-1], yc[0], yc[-1]]

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(q, extent=ext, origin="lower", cmap=cmap, vmin=-1,
                  vmax=1, aspect="auto", interpolation="nearest")
        if show_band:
            for sy in (band_y, -band_y):
                ax.axhline(sy, color="orange", lw=0.8, ls=":")
            if band_x is not None:
                for sx in (band_x, -band_x):
                    ax.axvline(sx, color="orange", lw=0.8, ls=":")
        ax.set_axis_off()
        label = f"t = {tt[idx]:.3f}" if tt is not None else f"frame {idx}"
        ax.text(0.02, 0.98, label, transform=ax.transAxes, color="black",
                fontsize=12, va="top", ha="left")
        ax.text(0.02, 0.93,
                f"sigma_det={sigma_det:.2f}   "
                f"full-domain sum|q|={int(np.abs(q).sum())}",
                transform=ax.transAxes, color="black", fontsize=9,
                va="top", ha="left")
        frames.append(_frame_from_fig(fig))

    out_path = out_dir / f"{name}.gif"
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Saved full-domain charge map GIF -> {out_path}")
    return out_path


def make_full_domain_charge_sweep_gif(sh, out_dir, name, sigma_det_list,
                                      fps=10, cmap="RdBu_r",
                                      panel_size=(4.2, 4.2),
                                      show_band=True, band_y=2.0,
                                      band_x=None):
    """
    Multi-panel version of make_full_domain_charge_map_gif: one panel per
    sigma_det, full-domain q as an image, side by side, animated together.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    u, x, y, Lx, Ly = sh["u"], sh["x"], sh["y"], sh["Lx"], sh["Ly"]
    tt = sh.get("tt")
    nt = u.shape[2]
    n_panels = len(sigma_det_list)

    frames = []
    for idx in range(nt):
        print("making frame:", idx)
        fig, axs = plt.subplots(
            1, n_panels,
            figsize=(panel_size[0] * n_panels, panel_size[1]),
            squeeze=False,
        )
        axs = axs[0]
        for ax, sg in zip(axs, sigma_det_list):
            det = detect_defects(u[:, :, idx], x, y, Lx, Ly,
                                 sigma_det=sg, band_y=band_y,
                                 band_x=band_x)
            xc, yc, q = det["xc"], det["yc"], det["q"]
            ext = [xc[0], xc[-1], yc[0], yc[-1]]
            ax.imshow(q, extent=ext, origin="lower", cmap=cmap, vmin=-1,
                      vmax=1, aspect="auto", interpolation="nearest")
            if show_band:
                for sy in (band_y, -band_y):
                    ax.axhline(sy, color="orange", lw=0.6, ls=":")
                if band_x is not None:
                    for sx in (band_x, -band_x):
                        ax.axvline(sx, color="orange", lw=0.6, ls=":")
            ax.set_axis_off()
            ax.set_title(
                f"sigma_det={sg:.2f}  full sum|q|={int(np.abs(q).sum())}",
                fontsize=9, color="black")
        label = f"t = {tt[idx]:.3f}" if tt is not None else f"frame {idx}"
        fig.suptitle(label, fontsize=12, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        frames.append(_frame_from_fig(fig))

    out_path = out_dir / f"{name}.gif"
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Saved full-domain charge sweep GIF -> {out_path}")
    return out_path


def make_defect_sigma_sweep_gif(sh, out_dir, name, sigma_det_list,
                                band_y=2.0, band_x=None, fps=10,
                                cmap="gray", vmin=-1, vmax=1,
                                panel_size=(4.2, 4.2)):
    """
    Multi-panel: one panel per sigma_det value, pattern + defect markers,
    side by side, animated together. Lets you see directly how the
    detection width changes what counts as a defect on the same frame.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    u, x, y, Lx, Ly = sh["u"], sh["x"], sh["y"], sh["Lx"], sh["Ly"]
    tt = sh.get("tt")
    ext = [x[0], x[-1], y[0], y[-1]]
    nt = u.shape[2]
    n_panels = len(sigma_det_list)

    frames = []
    for idx in range(nt):
        print("making frame:", idx)
        fig, axs = plt.subplots(
            1, n_panels,
            figsize=(panel_size[0] * n_panels, panel_size[1]),
            squeeze=False,
        )
        axs = axs[0]
        for ax, sg in zip(axs, sigma_det_list):
            det = detect_defects(u[:, :, idx], x, y, Lx, Ly,
                                 sigma_det=sg, band_y=band_y,
                                 band_x=band_x)
            ax.imshow(u[:, :, idx], extent=ext, origin="lower", cmap=cmap,
                      vmin=vmin, vmax=vmax, aspect="auto")
            _overlay_markers(ax, det, ms=6)
            ax.set_axis_off()
            ax.set_title(
                f"sigma_det={sg:.2f}  nd={det['n_dislocations']}",
                fontsize=10, color="black")
        label = f"t = {tt[idx]:.3f}" if tt is not None else f"frame {idx}"
        fig.suptitle(label, fontsize=12, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        frames.append(_frame_from_fig(fig))

    out_path = out_dir / f"{name}.gif"
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Saved sigma sweep GIF -> {out_path}")
    return out_path