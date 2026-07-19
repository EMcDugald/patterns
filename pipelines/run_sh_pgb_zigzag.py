# pipelines/run_sh_pgb_zigzag.py
"""
Runner for Swift-Hohenberg PGB zigzag simulations.

Energy density is always saved (energy=True).

Usage examples
--------------
# --- Single mu, many snapshots (time-resolved run) ---
python run_sh_pgb_zigzag.py --mu 0.5 --nsave 100 --tmax 5000

# --- Single mu, tail-window snapshots ---
python run_sh_pgb_zigzag.py --mu 0.97 --nsave 50 --tmax 10000 --t_save_window 500

# --- Mu sweep, first + last only ---
python run_sh_pgb_zigzag.py --mu_list 0.4 0.5 0.6 0.7 0.8 0.9 --nsave 1 --tmax 500

# --- From a JSON config file ---
python run_sh_pgb_zigzag.py --config my_run.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

from sh_sims.pgb_zigzag import solve_sh_pgb_zigzag, build_pgb_zigzag_ic


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_stem(mu, tmax, nsave, Nx, crop_Ny, which_gb, t_save_window):
    stem = (
        f"sh_pgb_zigzag"
        f"_mu{mu:.3f}"
        f"_T{int(tmax)}"
        f"_N{nsave}"
        f"_nx{Nx}"
        f"_Ny{crop_Ny if crop_Ny is not None else 'full'}"
        f"_{which_gb}"
    )
    if t_save_window is not None:
        stem += f"_tail{int(t_save_window)}"
    return stem


def save_result(result, out_path):
    save_dict = {
        "tt":    result["tt"],
        "x":     result["x"],
        "y":     result["y"],
        "x_full":result["x_full"],
        "y_full":result["y_full"],
        "u":     result["u"],
        "metadata_json": result["metadata_json"],
    }
    # energy is always computed; store if present
    if result["e"] is not None:
        save_dict["e"] = result["e"]
    if result["theta_initial"] is not None:
        save_dict["theta_initial"]     = result["theta_initial"]
        save_dict["cos_theta_initial"] = result["cos_theta_initial"]

    np.savez_compressed(out_path, **save_dict)
    print(f"    saved -> {out_path}")


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------

def make_summary_plot(result, fig_path, mu):
    meta  = json.loads(result["metadata_json"])
    u     = result["u"]   # (Ny, Nx, Nt)
    e     = result["e"]
    tt    = result["tt"]
    Nt    = u.shape[-1]

    has_e = e is not None

    # Decide which frames to plot:
    # - Nt == 1: only initial
    # - Nt >= 2: initial and final
    if Nt == 1:
        fi     = [0]
    else:
        fi     = [0, Nt - 1]
    labels = [f"t = {tt[i]:.1f}" for i in fi]

    n_rows = 2 if has_e else 1
    n_cols = len(fi)

    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.8 * n_rows),
        squeeze=False
    )

    for col, (idx, lbl) in enumerate(zip(fi, labels)):
        im0 = axs[0, col].imshow(u[..., idx], cmap="bwr", origin="lower")
        axs[0, col].set_title(f"u  {lbl}")
        fig.colorbar(im0, ax=axs[0, col], shrink=0.8)
        axs[0, col].set_xticks([])
        axs[0, col].set_yticks([])

        if has_e:
            im1 = axs[1, col].imshow(e[..., idx], cmap="viridis", origin="lower")
            axs[1, col].set_title(f"energy  {lbl}")
            fig.colorbar(im1, ax=axs[1, col], shrink=0.8)
            axs[1, col].set_xticks([])
            axs[1, col].set_yticks([])

    fig.suptitle(
        f"mu={mu:.3f}  alpha={meta['alpha']:.4f}  "
        f"Nx={meta['Nx']}  Lx_periods={meta['Lx_periods']:.0f}  "
        f"y_centering={meta['y_centering']}  "
        f"GB row in crop: {meta['gb_row_in_crop']} ({meta['crop_Ny_parity']})",
        y=0.995, fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"    plot  -> {fig_path}")


def make_ic_overlay_plot(result, fig_path_full):
    """
    Plot the full-domain initial condition with a rectangle showing the crop
    region used in the saved data.
    """
    meta = json.loads(result["metadata_json"])

    # Reconstruct full IC on the computational domain
    Nx         = meta["Nx"]
    Ny         = meta["Ny"]
    mu         = meta["mu"]
    n_periods  = meta["n_periods"]
    # Reconstruct Ny_factor from Ny and Nx (Ny ≈ Ny_factor * Nx)
    Ny_factor  = float(Ny) / float(Nx)
    Rscale     = meta["Rscale"]
    amp        = meta["amp"]
    y_centering = meta["y_centering"]

    geom = build_pgb_zigzag_ic(
        Nx          = Nx,
        mu          = mu,
        n_periods   = n_periods,
        Ny_factor   = Ny_factor,
        Rscale      = Rscale,
        amp         = amp,
        y_centering = y_centering,
    )
    u0_full = geom["u0"]
    xx_full = geom["xx"]
    yy_full = geom["yy"]

    # Crop slices in index space, as recorded by the solver
    x_slice = meta["crop_x_slice"]   # (x_start_idx, x_end_idx)
    y_slice = meta["crop_y_slice"]   # (y_start_idx, y_end_idx)

    # Convert index slices to physical extents
    x_min = xx_full[x_slice[0]]
    x_max = xx_full[x_slice[1] - 1]
    y_min = yy_full[y_slice[0]]
    y_max = yy_full[y_slice[1] - 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    im = ax.imshow(u0_full, cmap="bwr", origin="lower",
                   extent=[xx_full[0], xx_full[-1], yy_full[0], yy_full[-1]])
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Full-domain IC with crop overlay")
    ax.set_xticks([])
    ax.set_yticks([])

    # Overlay rectangle for cropped region
    rect = plt.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        edgecolor="yellow",
        facecolor="none",
        linewidth=1.5,
    )
    ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(fig_path_full, dpi=150)
    plt.close(fig)
    print(f"    IC overlay -> {fig_path_full}")


# ---------------------------------------------------------------------------
# Core: process one mu value
# ---------------------------------------------------------------------------

def process_mu(mu, cfg, raw_dir, fig_dir):
    stem = build_stem(
        mu            = mu,
        tmax          = cfg["tmax"],
        nsave         = cfg["nsave"],
        Nx            = cfg["Nx"],
        crop_Ny       = cfg.get("crop_Ny"),
        which_gb      = cfg["which_gb"],
        t_save_window = cfg.get("t_save_window"),
    )

    out_path = raw_dir / f"{stem}.npz"
    fig_path = fig_dir / f"{stem}.png"
    fig_path_full = fig_dir / f"{stem}_ic_overlay.png"

    if out_path.exists() and not cfg.get("overwrite", False):
        print(f"  skip (exists): {stem}")
        return

    print(f"  mu={mu:.3f} ...")
    result = solve_sh_pgb_zigzag(
        Nx                = cfg["Nx"],
        mu                = float(mu),
        h                 = cfg["h"],
        tmax              = cfg["tmax"],
        nsave             = cfg["nsave"],
        n_periods         = cfg.get("n_periods",   12),
        Ny_factor         = cfg.get("Ny_factor",    6),
        Rscale            = cfg.get("Rscale",      0.5),
        amp               = cfg.get("amp",         0.5),
        y_centering       = cfg.get("y_centering", "node"),
        energy            = True,              # always on
        t_save_window     = cfg.get("t_save_window"),
        which_gb          = cfg["which_gb"],
        crop_Nx           = cfg.get("crop_Nx"),
        crop_Ny           = cfg.get("crop_Ny"),
        save_initial_phase= cfg.get("save_initial_phase", True),
    )

    save_result(result, out_path)

    if not cfg.get("no_plot", False):
        make_summary_plot(result, fig_path, mu)
        make_ic_overlay_plot(result, fig_path_full)


# ---------------------------------------------------------------------------
# Config from CLI args
# ---------------------------------------------------------------------------

def cfg_from_args(args):
    if args.mu_list:
        mus = [float(m) for m in args.mu_list]
    elif args.mu is not None:
        mus = [float(args.mu)]
    else:
        raise ValueError("Provide --mu <value> or --mu_list <v1 v2 ...>")

    return {
        "mus":             mus,
        "Nx":              args.Nx,
        "h":               args.h,
        "tmax":            args.tmax,
        "nsave":           args.nsave,
        "t_save_window":   args.t_save_window,
        "n_periods":       args.n_periods,
        "Ny_factor":       args.Ny_factor,
        "Rscale":          args.Rscale,
        "amp":             args.amp,
        "y_centering":     args.y_centering,
        "which_gb":        args.which_gb,
        "crop_Nx":         args.crop_Nx,
        "crop_Ny":         args.crop_Ny,
        "save_initial_phase": not args.no_initial_phase,
        "output_dir":      args.output_dir,
        "no_plot":         args.no_plot,
        "overwrite":       args.overwrite,
    }

def run_with_cfg(cfg):
    out_root = Path(cfg["output_dir"])
    raw_dir  = ensure_dir(out_root / "raw")
    fig_dir  = ensure_dir(out_root / "figures")

    mus = cfg["mus"]
    print(f"Running {len(mus)} mu value(s): {mus}")
    for mu in mus:
        process_mu(mu, cfg, raw_dir, fig_dir)
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Swift-Hohenberg PGB zigzag simulations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # mu
    mu_grp = parser.add_mutually_exclusive_group()
    mu_grp.add_argument("--mu", type=float, default=None,
                        help="Single mu value.")
    mu_grp.add_argument("--mu_list", type=float, nargs="+", default=None,
                        help="List of mu values for a sweep.")

    # time integration
    parser.add_argument("--h",             type=float, default=0.25)
    parser.add_argument("--tmax",          type=float, default=500.0)
    parser.add_argument("--nsave",         type=int,   default=1,
                        help="Snapshots after t=0. nsave=1 -> first+last only.")
    parser.add_argument("--t_save_window", type=float, default=None,
                        help="Save nsave frames from [tmax-window, tmax].")

    # grid / IC
    parser.add_argument("--Nx",          type=int,   default=512)
    parser.add_argument("--n_periods",   type=int,   default=12,
                        help="Integer number of x stripe periods (sets Lx).")
    parser.add_argument("--Ny_factor",   type=float, default=6.0,
                        help="Ny = round(Ny_factor * Nx). Use value s.t. Ny%%4==0.")
    parser.add_argument("--Rscale",      type=float, default=0.5)
    parser.add_argument("--amp",         type=float, default=0.5)
    parser.add_argument("--y_centering", type=str,   default="node",
                        choices=["node", "cell"])

    # crop
    parser.add_argument("--which_gb", type=str, default="lower",
                        choices=["lower", "upper"])
    parser.add_argument("--crop_Nx",  type=int, default=None)
    parser.add_argument("--crop_Ny",  type=int, default=None,
                        help="Odd -> GB on centre row. Even -> GB straddles.")

    # output
    parser.add_argument("--output_dir",        type=str,
                        default="results/sh_pgb_zigzag")
    parser.add_argument("--no_plot",           action="store_true")
    parser.add_argument("--no_initial_phase",  action="store_true")
    parser.add_argument("--overwrite",         action="store_true")

    # config file
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config file (overrides all CLI args).")

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            cfg = json.load(f)
    else:
        cfg = cfg_from_args(args)

    # instead of duplicating the run logic:
    run_with_cfg(cfg)


if __name__ == "__main__":
    # If no CLI args (just script name), use an in-code debug config.
    if len(sys.argv) == 1:
        cfg = {
            "mus":             [0.85, .95],     # tweak these in PyCharm
            "Nx":              512,
            # "h":               0.025,
            # "tmax":            3.125,
            "h": 0.025,
            "tmax": 3.125,
            "nsave":           125,
            "t_save_window":   None,
            "n_periods":       18,
            "Ny_factor":       5.0,
            "Rscale":          0.5,
            "amp":             1.0,
            "y_centering":     "node",
            "which_gb":        "lower",
            "crop_Nx":         256,
            "crop_Ny":         256,
            "save_initial_phase": True,
            "output_dir":      "data/sh_pgb_zigzag/mu_sweeps_full_Nx512_hp025_T3p125_NyF5_np18_Nsave125",
            "no_plot":         False,
            "overwrite":       True,
        }
        run_with_cfg(cfg)
    else:
        main()


# =============================================================================
# Example configs (for reference / copy-paste into JSON files)
# =============================================================================
#
# --- 1) Single mu, many snapshots (time-resolved study) ---
#
# python run_sh_pgb_zigzag.py \
#     --mu 0.5 \
#     --Nx 512 --n_periods 12 --Ny_factor 6 \
#     --h 0.25 --tmax 5000 --nsave 100 \
#     --crop_Ny 257 \          # odd -> GB on centre row
#     --output_dir data/sh_pgb_zigzag/time_resolved
#
# Equivalent JSON (pgb_time_resolved.json):
# {
#   "mus": [0.5],
#   "Nx": 512, "n_periods": 12, "Ny_factor": 6,
#   "h": 0.25, "tmax": 5000, "nsave": 100,
#   "t_save_window": null,
#   "crop_Ny": 257,
#   "which_gb": "lower", "y_centering": "node",
#   "Rscale": 0.5, "amp": 0.5,
#   "save_initial_phase": true,
#   "output_dir": "results/sh_pgb_zigzag/time_resolved",
#   "no_plot": false, "overwrite": false
# }
#
# --- 2) Mu sweep, first + last only ---
#
# python run_sh_pgb_zigzag.py \
#     --mu_list 0.40 0.50 0.60 0.70 0.80 0.90 \
#     --Nx 512 --n_periods 12 --Ny_factor 6 \
#     --h 0.25 --tmax 500 --nsave 1 \
#     --crop_Ny 257 \
#     --output_dir results/sh_pgb_zigzag/mu_sweep
#
# Equivalent JSON (pgb_mu_sweep.json):
# {
#   "mus": [0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
#   "Nx": 512, "n_periods": 12, "Ny_factor": 6,
#   "h": 0.25, "tmax": 500, "nsave": 1,
#   "t_save_window": null,
#   "crop_Ny": 257,
#   "which_gb": "lower", "y_centering": "node",
#   "Rscale": 0.5, "amp": 0.5,
#   "save_initial_phase": true,
#   "output_dir": "results/sh_pgb_zigzag/mu_sweep",
#   "no_plot": false, "overwrite": false
# }
#
# --- 3) Single mu, tail-window snapshots (near-equilibrium study) ---
#
# python run_sh_pgb_zigzag.py \
#     --mu 0.97 --nsave 50 --tmax 10000 --t_save_window 500 \
#     --Nx 512 --crop_Ny 257 \
#     --output_dir results/sh_pgb_zigzag/tail_window