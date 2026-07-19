"""
paper_style.py -- shared matplotlib style + figure helpers for the PGB paper.
Import and call `use_paper_style()` at the top of any analysis script.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

PAPER_RC = {
    "font.size": 11,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "image.origin": "lower",
}


def use_paper_style():
    mpl.rcParams.update(PAPER_RC)


def panel_label(ax, s, dx=0.02, dy=0.98, color="k"):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left", color=color)


def add_cbar(fig, ax, im, label=None):
    """Colorbar matched to axis height (no shrink guessing)."""
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.06)
    cb = fig.colorbar(im, cax=cax)
    if label:
        cb.set_label(label)
    cb.ax.tick_params(labelsize=8)
    return cb


def sym_limits(*fields, mask=None, pct=99.5):
    """Symmetric color limits from pooled robust percentile."""
    vals = []
    for f in fields:
        a = np.asarray(f, dtype=float)
        if mask is not None:
            a = a[mask]
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(a)
    if not vals:
        return -1.0, 1.0
    v = np.nanpercentile(np.abs(np.concatenate(vals)), pct)
    return -v, v


def masked(field, mask):
    return np.ma.masked_where(~mask, field)


def comparison_triptych(fig_path, measured, analytic, mask, extent,
                        labels=("measured", "analytic"),
                        cmap="RdBu_r", err_cmap="magma",
                        xlab="$x$", ylab="$y$", suptitle=None,
                        signed=True, err_label="abs.\\ error"):
    """
    Three-panel: measured | analytic | |difference|, shared signed limits.
    Returns RMS error over mask.
    """
    err = np.abs(np.asarray(measured) - np.asarray(analytic))
    if signed:
        vmin, vmax = sym_limits(measured, analytic, mask=mask)
    else:
        pool = np.concatenate([np.asarray(measured)[mask],
                               np.asarray(analytic)[mask]])
        vmin, vmax = 0.0, np.nanpercentile(pool, 99.5)

    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    for ax, f, lab, cm, (v0, v1) in zip(
            axs,
            (measured, analytic, err),
            (labels[0], labels[1], f"$|${labels[0]}$ - ${labels[1]}$|$"),
            (cmap, cmap, err_cmap),
            ((vmin, vmax), (vmin, vmax),
             (0.0, np.nanpercentile(err[mask], 99.5)))):
        im = ax.imshow(masked(f, mask), extent=extent, cmap=cm,
                       vmin=v0, vmax=v1, aspect="auto", rasterized=True)
        ax.set_title(lab)
        ax.set_xlabel(xlab)
        add_cbar(fig, ax, im)
    axs[0].set_ylabel(ylab)
    for ax, pl in zip(axs, ("(a)", "(b)", "(c)")):
        panel_label(ax, pl, color="k")
    if suptitle:
        fig.suptitle(suptitle)
    fig.savefig(fig_path)
    plt.close(fig)

    rms = float(np.sqrt(np.nanmean(err[mask] ** 2)))
    return rms


def pattern_pair(fig_path, u_left, u_right, extent,
                 titles=("SH pattern", "analytic pattern"),
                 vlines=None, mask_left=None, mask_right=None):
    """Two grayscale patterns side by side with optional defect verticals."""
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)
    for ax, f, m, t, pl in zip(axs, (u_left, u_right),
                               (mask_left, mask_right), titles,
                               ("(a)", "(b)")):
        ff = masked(f, m) if m is not None else f
        ax.imshow(ff, extent=extent, cmap="gray", vmin=-1, vmax=1,
                  aspect="auto", rasterized=True)
        ax.set_title(t)
        ax.set_xlabel("$x$")
        if vlines is not None:
            for xv in vlines:
                ax.axvline(xv, color="#ffcc00", ls=":", lw=0.9, alpha=0.9)
        panel_label(ax, pl, color="w")
    axs[0].set_ylabel("$y$")
    fig.savefig(fig_path)
    plt.close(fig)
