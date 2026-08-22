"""
fig_energy_firstpass.py  the compact main-text test figure.

One row, two panels, same configuration and no manual switch:

  left   defect-free knee: along-core profile of the averaged
         microscopic energy, isotropic versus auto-gated kernel. The
         isotropic average carries the 2 k1 corrugation at exactly the
         PN spacing L = pi/k1 (gridlines); the auto-gated average is
         flat, as the macroscopic energy of a knee core must be.
  right  PN multi-dislocation boundary: the same comparison. Here the
         oscillation IS the signal (one energy concentration per
         dislocation, gridlines at the true defect positions) and the
         auto-gated average reproduces the isotropic one, because the
         quantized defect index reads one charge per period and the
         kernel stays isotropic.

Together the two panels are the first-pass acceptance of the method:
it removes the artifact where the theory says the profile is flat and
changes nothing where the structure is physical.

Run with no arguments for knee + pismen at mu = 0.90.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adaptive_auto_gate_patched import sample_along_path
from defect_index_figures import prepare_case
from patch_composition_figs import averaged_energy
try:
    from paper_style import use_paper_style
    use_paper_style()
except ImportError:
    pass


def fig_energy_firstpass(cfg, mu, out_dir, patch=96):
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 3.4),
                            constrained_layout=True)
    for ax, field in zip(axs, ("knee", "pismen")):
        case = prepare_case(cfg, field, mu)
        record = []
        e_micro, e_iso, e_auto, aspect, sharpen = averaged_energy(
            case, cfg, patch, record)
        x, rows_s, L = case["x"], case["rows_s"], case["L"]
        xin = np.abs(x) < (2.0 * L * 2 if field == "pismen"
                           else 0.35 * case["Lx"])
        ax.plot(x[xin], sample_along_path(e_iso, rows_s)[xin], "r-",
                lw=1.4, label="isotropic")
        ax.plot(x[xin], sample_along_path(e_auto, rows_s)[xin], "b--",
                lw=1.2, label="auto-gated")
        marks = (case["x_defects"] if field == "pismen"
                 else np.arange(-20, 21) * L)
        for xm in marks:
            if x[xin].min() < xm < x[xin].max():
                ax.axvline(xm, color="k", lw=0.4, alpha=0.25)
        ax.set_xlabel("$x$")
        if field == "knee":
            ax.set_ylabel("$\\langle e\\rangle$ on the core")
            ax.set_title("defect-free knee: corrugation at $2k_1$ "
                         "removed\n(gridlines at multiples of "
                         "$L=\\pi/k_1$)", fontsize=9)
        else:
            ax.set_title("PN multi-dislocation: defect signal "
                         "preserved\n(gridlines at the true defect "
                         "positions)", fontsize=9)
        ax.legend(fontsize=8)
    fig.savefig(Path(out_dir) / f"energy_firstpass_mu{mu:.2f}.png",
                bbox_inches="tight", dpi=200)
    plt.close(fig)


def run(cfg):
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for mu in cfg.mus:
        fig_energy_firstpass(cfg, mu, out_dir, patch=cfg.patch)
        print(f"mu={mu:.2f}: energy_firstpass written")


def main():
    p = argparse.ArgumentParser(
        description="Compact first-pass energy comparison figure")
    p.add_argument("--out", type=str, default="results/firstpass")
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
            out = "results/firstpass"
            mus = [0.90]
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
