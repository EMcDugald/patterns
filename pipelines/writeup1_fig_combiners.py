"""
writeup1_fig_combiners.py

Assembles three figures for Writeup 1 from existing outputs and the
frozen IC builder, so no Inkscape or slide software is needed:

  1. k2_profile_fits_2x2.png : 2x2 grid of the per-mu knee-profile
     panels (mu = 0.325, 0.375, 0.425, 0.475 at R = 0.5) produced by
     extraction_comparison.py.
  2. selfdual_scatter_mu0475.png : the self-dual balance scatter,
     cropped from the lower-right quadrant of the mu = 0.475
     balance_and_scatter panel.
  3. zigzag_ic_panels.png : the zigzag initial condition re-rendered
     horizontally (domain rotated so the tall periodic direction runs
     left to right), with cores, stitch lines, and the analysis crop
     marked, plus a zoom of the lower-boundary knee. Replaces the
     tall-and-skinny overlay, which wasted page area.

v1.2 changelog (measured justification):
  - scatter_png now defaults to the mu = 0.475 run-of-record path
    under results/extraction_comparison_Rp5/ (the location printed by
    the extraction_comparison console run), with a graceful skip and
    a hint if the file is not found; no flag needed for the standard
    layout.

v1.1 changelog:
  - Default paths are now anchored to the script's own directory
    (pipelines/), not the process working directory; running from
    PyCharm with any working directory now finds
    pipelines/results/... correctly.

v1.0 changelog:
  - New standalone figure script; frozen method layer untouched
    (the IC is built by importing src/sh_sims/pgb_zigzag.py, not by
    reimplementing it).
  - Panels 1 and 2 are image-level operations on archived pngs, so
    the displayed numbers are exactly those of the runs of record.
  - Crop fractions for panel 2 are CLI-tunable because they are
    pixel-space and depend on the source figure's layout.

Output: <outdir>/{k2_profile_fits_2x2, selfdual_scatter_mu0475,
zigzag_ic_panels}.png   (default outdir
pipelines/results/writeup1_figures/)
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

try:
    from paper_style import use_paper_style
except ImportError:
    def use_paper_style():
        pass

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# ----------------------------------------------------------------------
def make_k2_grid(png_paths, out):
    """2x2 grid of archived knee-profile panels, no re-rendering."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, p in zip(axes.ravel(), png_paths):
        ax.imshow(mpimg.imread(p))
        ax.set_axis_off()
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,
                        wspace=0.02, hspace=0.02)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("saved", out)


# ----------------------------------------------------------------------
def make_scatter_crop(png_path, out, x0=0.50, x1=1.00, y0=0.50, y1=1.00):
    """Crop the self-dual scatter quadrant out of the archived
    balance_and_scatter panel (fractions of width/height)."""
    img = mpimg.imread(png_path)
    H, W = img.shape[:2]
    sub = img[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.imshow(sub)
    ax.set_axis_off()
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("saved", out)


# ----------------------------------------------------------------------
def make_ic_panels(mu, Nx, crop_Ny, out):
    """Re-render the zigzag IC horizontally with annotations."""
    from sh_sims.pgb_zigzag import build_pgb_zigzag_ic
    d = build_pgb_zigzag_ic(Nx=Nx, mu=mu)
    u0, xx, yy = d["u0"], d["xx"], d["yy"]
    Lx, Ly, yl, yr, dy = d["Lx"], d["Ly"], d["yl"], d["yr"], d["dy"]

    use_paper_style()
    fig = plt.figure(figsize=(10.5, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.1, 1.0], wspace=0.14)

    # left: rotated full domain (y runs horizontally)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(u0.T, origin="lower", cmap="RdBu_r", aspect="auto",
              extent=[yy[0], yy[-1] + dy, xx[0], xx[-1] + d["dx"]])
    for yc, lab in [(yl, "lower core"), (yr, "upper core")]:
        ax.axvline(yc, color="k", lw=1.0)
        ax.text(yc, xx[-1] * 1.06, lab, ha="center", fontsize=9)
    for ys in [0.0, yy[0], yy[-1] + dy]:
        ax.axvline(ys, color="k", lw=1.0, ls=":")
    ax.text(0.0, xx[-1] * 1.06, "stitch", ha="center", fontsize=9)
    half = 0.5 * crop_Ny * dy
    ax.axvspan(yl - half, yl + half, color="gold", alpha=0.18, lw=0)
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$x$")

    # right: zoom of the lower knee (upright orientation)
    ax2 = fig.add_subplot(gs[0, 1])
    m = (yy >= yl - half) & (yy < yl + half)
    ax2.imshow(u0[m, :], origin="lower", cmap="RdBu_r",
               extent=[xx[0], xx[-1] + d["dx"], yy[m][0], yy[m][-1] + dy],
               aspect="equal")
    ax2.axhline(yl, color="k", lw=0.8, ls="--")
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$y$")
    ax2.set_title("analysis crop", fontsize=10)

    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# ----------------------------------------------------------------------
_RESULTS = os.path.join(_HERE, "results")


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below.
    Paths are anchored to this script's directory (pipelines/), so
    the working directory does not matter."""
    outdir = os.path.join(_RESULTS, "writeup1_fig_combiners")
    k2_dir = os.path.join(_RESULTS, "extraction_comparison_Rp5")
    k2_pngs = None          # explicit list overrides k2_dir discovery
    scatter_png = os.path.join(
        _RESULTS, "extraction_comparison_Rp5",
        "Rp5_sh_pgb_zigzag_mu0.475_T500_N2_nx512_Ny256_lower",
        "balance_and_scatter.png")
    scatter_frac = (0.50, 1.00, 0.53, 1.00)
    mu = 0.475
    Nx = 512
    crop_Ny = 256


def main(a):
    os.makedirs(a.outdir, exist_ok=True)

    if a.k2_pngs:
        pngs = a.k2_pngs
    else:
        stems = ["Rp5_sh_pgb_zigzag_mu%.3f_T500_N2_nx512_Ny256_lower" % m
                 for m in (0.325, 0.375, 0.425, 0.475)]
        pngs = [os.path.join(a.k2_dir, s, "k2_profile_fits.png")
                for s in stems]
    make_k2_grid(pngs, os.path.join(a.outdir, "k2_profile_fits_2x2.png"))

    if a.scatter_png and os.path.exists(a.scatter_png):
        x0, x1, y0, y1 = a.scatter_frac
        make_scatter_crop(a.scatter_png,
                          os.path.join(a.outdir,
                                       "selfdual_scatter_mu0475.png"),
                          x0, x1, y0, y1)
    else:
        print("scatter source not found at %r; pass --scatter_png "
              "pointing at the mu=0.475 balance_and_scatter.png"
              % a.scatter_png)

    make_ic_panels(a.mu, a.Nx, a.crop_Ny,
                   os.path.join(a.outdir, "zigzag_ic_panels.png"))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--outdir", default=_Args.outdir)
    p.add_argument("--k2_dir", default=_Args.k2_dir)
    p.add_argument("--k2_pngs", nargs=4, default=_Args.k2_pngs,
                   help="explicit four panel paths, mu ascending")
    p.add_argument("--scatter_png", default=_Args.scatter_png)
    p.add_argument("--scatter_frac", nargs=4, type=float,
                   default=list(_Args.scatter_frac),
                   metavar=("X0", "X1", "Y0", "Y1"))
    p.add_argument("--mu", type=float, default=_Args.mu)
    p.add_argument("--Nx", type=int, default=_Args.Nx)
    p.add_argument("--crop_Ny", type=int, default=_Args.crop_Ny)
    sys.exit(main(p.parse_args()))