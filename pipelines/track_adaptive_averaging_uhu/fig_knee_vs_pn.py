"""
fig_knee_vs_pn.py  the main-text comparison figure.

Two rows (defect-free knee, PN multi-dislocation), three columns, one
configuration and no manual switch:

  (1) seed coherence with the detected core path: LOW on both cores,
      so coherence localizes boundaries but cannot tell a knee from a
      dislocation-bearing boundary;
  (2) defect index |C|/2pi over the pattern (opacity proportional to
      the index): the topological discriminator, ~0 everywhere on the
      knee, one quantum on the PN core;
  (3) the auto-gated averaged energy: the physical outcome, a uniform
      LINE energy along the knee core versus POINT energies at the
      dislocations of the PN boundary.

Reads left to right as: where the boundary is, what it is, and what
its macroscopic energy therefore looks like.

Run with no arguments for mu = 0.90.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from defect_index_figures import prepare_case, eroded_interior, crop_extent
from patch_composition_figs import averaged_energy
from index_overlay import overlay_index
try:
    from paper_style import use_paper_style
    use_paper_style()
except ImportError:
    pass


def fig_knee_vs_pn(cfg, mu, out_dir, patch=96):
    fig, axs = plt.subplots(2, 3, figsize=(13.6, 5.4),
                            constrained_layout=True)
    for row, field in enumerate(("knee", "pismen")):
        case = prepare_case(cfg, field, mu)
        x, y = case["x"], case["y"]
        disp = eroded_interior(case, cfg)
        ext, sl = crop_extent(disp, x, y)
        label = ("defect-free knee" if field == "knee"
                 else "PN multi-dislocation")

        # (1) seed coherence + core path
        ax = axs[row, 0]
        im = ax.imshow(np.ma.masked_where(~disp, case["coh"])[sl],
                       extent=ext, origin="lower", aspect="auto",
                       cmap="viridis", vmin=0, vmax=1, rasterized=True)
        ax.plot(x, y[case["rows_s"]], "r-", lw=0.7)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_title(f"{label}: seed coherence", fontsize=9)
        ax.set_ylabel("$y$")
        fig.colorbar(im, ax=ax, shrink=0.85)

        # (2) defect index over the pattern
        ax = axs[row, 1]
        sm = overlay_index(ax, case, cfg, show_core_path=False)
        ax.set_title("defect index $|C|/2\\pi$", fontsize=9)
        fig.colorbar(sm, ax=ax, shrink=0.85)

        # (3) auto-gated averaged energy
        ax = axs[row, 2]
        record = []
        e_micro, e_iso, e_auto, aspect, sharpen = averaged_energy(
            case, cfg, patch, record)
        vmax = np.percentile(e_auto[disp], 99.5)
        im = ax.imshow(np.ma.masked_where(~disp, e_auto)[sl],
                       extent=ext, origin="lower", aspect="auto",
                       cmap="magma", vmin=0, vmax=vmax, rasterized=True)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_title("auto-gated $\\langle e\\rangle$: "
                     + ("line energy" if field == "knee"
                        else "point energies"), fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85)

        if row == 1:
            for ax in axs[row]:
                ax.set_xlabel("$x$")
    fig.savefig(Path(out_dir) / f"knee_vs_pn_mu{mu:.2f}.png",
                bbox_inches="tight", dpi=200)
    plt.close(fig)


def run(cfg):
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for mu in cfg.mus:
        fig_knee_vs_pn(cfg, mu, out_dir, patch=cfg.patch)
        print(f"mu={mu:.2f}: knee_vs_pn written")


def main():
    p = argparse.ArgumentParser(
        description="Main-text knee versus PN comparison figure")
    p.add_argument("--out", type=str, default="results/knee_vs_pn")
    p.add_argument("--mus", type=float, nargs="+", default=[0.90])
    p.add_argument("--patch", type=int, default=96)
    p.add_argument("--R", type=float, default=0.5)
    p.add_argument("--R_out", type=float, default=0.2)
    p.add_argument("--Nx", type=int, default=512)
    p.add_argument("--Ny", type=int, default=256)
    p.add_argument("--Lx", type=float, default=40 * np.pi)
    p.add_argument("--Ly", type=float, default=20 * np.pi)
    p.add_argument("--sigma", type=float, default=np.pi / 2)
    p.add_argument("--leak_delta", type=float, default=0.02)
    p.add_argument("--n_defects", type=int, default=8)
    p.add_argument("--edge_margin_periods", type=float, default=2.0)
    p.add_argument("--relax_T_pismen", type=float, default=3.0)
    p.add_argument("--relax_T_knee", type=float, default=0.0)
    p.add_argument("--relax_h", type=float, default=0.05)
    p.add_argument("--gate_window_L", type=float, default=2.0)
    p.add_argument("--sigma_gate", type=float, default=np.pi / 8)
    p.add_argument("--sigma_dir", type=float, default=np.pi / 4)
    p.add_argument("--L_over_sigdir_min", type=float, default=8.5)
    p.add_argument("--circ_lo", type=float, default=0.25)
    p.add_argument("--circ_hi", type=float, default=0.5)
    p.add_argument("--cf_lo", type=float, default=1.10)
    p.add_argument("--cf_hi", type=float, default=1.35)
    p.add_argument("--core_coh_frac", type=float, default=0.5)
    p.add_argument("--core_band_sigma", type=float, default=2.0)
    p.add_argument("--xmargin", type=float, default=0.05)
    p.add_argument("--ymargin", type=float, default=0.10)
    p.add_argument("--tanhscale", type=float, default=60.0)
    run(p.parse_args())


if __name__ == "__main__":
    if len(sys.argv) == 1:
        class _Args:
            out = "results/knee_vs_pn"
            mus = [.85, 0.90, 0.95]
            patch = 96
            R, R_out = 0.5, 0.2
            Nx, Ny = 512, 256
            Lx, Ly = 40 * np.pi, 20 * np.pi
            sigma = np.pi / 2
            leak_delta = 0.02
            n_defects = 8
            edge_margin_periods = 2.0
            relax_T_pismen = 3.0
            relax_T_knee = 0.0
            relax_h = 0.05
            gate_window_L = 2.0
            sigma_gate = np.pi / 8
            sigma_dir = np.pi / 4
            L_over_sigdir_min = 8.5
            circ_lo, circ_hi = 0.25, 0.5
            cf_lo, cf_hi = 1.10, 1.35
            core_coh_frac = 0.5
            core_band_sigma = 2.0
            xmargin, ymargin = 0.05, 0.10
            tanhscale = 60.0
        run(_Args)
    else:
        main()
