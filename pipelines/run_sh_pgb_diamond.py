"""
Runner for Swift-Hohenberg PGB diamond simulations.

Usage examples
--------------
# Single mu, first + last only
python run_sh_pgb_diamond.py --mu 0.80 --Ly 157.0796 --Ny 512 --tmax 500 --nsave 1

# Single mu, many frames
python run_sh_pgb_diamond.py --mu 0.975 --Ly 376.9911 --Ny 768 --tmax 20 --nsave 100

# Tail-window snapshots
python run_sh_pgb_diamond.py --mu 0.97 --Ly 157.0796 --Ny 512 --tmax 1000 --nsave 50 --t_save_window 200

# Mu sweep
python run_sh_pgb_diamond.py --mu_list 0.4 0.5 0.6 0.7 --Ly 157.0796 --Ny 512 --tmax 500 --nsave 1
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

from sh_sims.pgb_diamond import solve_sh_pgb_diamond


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_stem(mu, tmax, nsave, Ny, Ly, margin, ic_method, t_save_window):
    stem = (
        f"sh_pgb_diamond"
        f"_mu{mu:.3f}"
        f"_T{int(tmax)}"
        f"_N{nsave}"
        f"_Ny{Ny}"
        f"_Ly{Ly:.3f}"
        f"_margin{margin:.2f}"
        f"_{ic_method}"
    )
    if t_save_window is not None:
        stem += f"_tail{int(t_save_window)}"
    return stem


def save_result(result, out_path):
    save_dict = {
        "tt": result["tt"],
        "x": result["x"],
        "y": result["y"],
        "x_full": result["x_full"],
        "y_full": result["y_full"],
        "u": result["u"],
        "metadata_json": result["metadata_json"],
        "ramp": result["ramp"],
        "inner_ramp": result["inner_ramp"],
        "bdry": result["bdry"],
        "bdry_inner": result["bdry_inner"],
    }

    if result["e"] is not None:
        save_dict["e"] = result["e"]
    if result["theta_initial"] is not None:
        save_dict["theta_initial"] = result["theta_initial"]
    if result["cos_theta_initial"] is not None:
        save_dict["cos_theta_initial"] = result["cos_theta_initial"]
    if result.get("theta_left_initial") is not None:
        save_dict["theta_left_initial"] = result["theta_left_initial"]
    if result.get("theta_right_initial") is not None:
        save_dict["theta_right_initial"] = result["theta_right_initial"]
    if result.get("hat_left_initial") is not None:
        save_dict["hat_left_initial"] = result["hat_left_initial"]
    if result.get("hat_right_initial") is not None:
        save_dict["hat_right_initial"] = result["hat_right_initial"]

    np.savez_compressed(out_path, **save_dict)
    print(f"    saved -> {out_path}")


def make_summary_plot(result, fig_path, mu):
    meta = json.loads(result["metadata_json"])
    u = result["u"]
    e = result["e"]
    inner_ramp = result["inner_ramp"]
    tt = result["tt"]
    Nt = u.shape[-1]

    has_e = e is not None
    fi = [0] if Nt == 1 else [0, Nt - 1]
    labels = [f"t = {tt[i]:.1f}" for i in fi]

    n_rows = 2 if has_e else 1
    n_cols = len(fi)

    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(4.8 * n_cols, 3.8 * n_rows),
        squeeze=False,
    )

    for col, (idx, lbl) in enumerate(zip(fi, labels)):
        im0 = axs[0, col].imshow(u[..., idx] * inner_ramp, cmap="bwr", origin="lower")
        axs[0, col].set_title(f"u  {lbl}")
        fig.colorbar(im0, ax=axs[0, col], shrink=0.8)
        axs[0, col].set_xticks([])
        axs[0, col].set_yticks([])

        if has_e:
            im1 = axs[1, col].imshow(e[..., idx] * inner_ramp, cmap="viridis", origin="lower")
            axs[1, col].set_title(f"energy  {lbl}")
            fig.colorbar(im1, ax=axs[1, col], shrink=0.8)
            axs[1, col].set_xticks([])
            axs[1, col].set_yticks([])

    fig.suptitle(
        f"mu={mu:.3f}  ic={meta['ic_method']}  "
        f"Nx={meta['Nx']} Ny={meta['Ny']}  "
        f"Lx={meta['Lx']:.3f} Ly={meta['Ly']:.3f}  "
        f"dx/dy={meta['dx_over_dy']:.3f}",
        y=0.995, fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"    plot  -> {fig_path}")


def make_ic_plot(result, fig_path):
    theta = result.get("theta_initial")
    cos_theta = result.get("cos_theta_initial")
    theta_left = result.get("theta_left_initial")
    theta_right = result.get("theta_right_initial")
    hat_left = result.get("hat_left_initial")
    hat_right = result.get("hat_right_initial")
    ramp = result["ramp"]

    items = []
    if theta_left is not None:
        items.append(("theta_left * ramp", theta_left * ramp, "twilight"))
    if theta_right is not None:
        items.append(("theta_right * ramp", theta_right * ramp, "twilight"))
    if hat_left is not None:
        items.append(("hat_left", hat_left, "viridis"))
    if hat_right is not None:
        items.append(("hat_right", hat_right, "viridis"))
    if theta is not None:
        items.append(("theta_initial * ramp", theta * ramp, "twilight"))
    if cos_theta is not None:
        items.append(("cos(theta_initial) * ramp", cos_theta * ramp, "bwr"))

    if not items:
        return

    ncols = 2
    nrows = int(np.ceil(len(items) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), squeeze=False)
    axs = axs.ravel()

    for ax, (title, arr, cmap) in zip(axs, items):
        im = ax.imshow(arr, cmap=cmap, origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)

    for ax in axs[len(items):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"    IC plot -> {fig_path}")


def process_mu(mu, cfg, raw_dir, fig_dir):
    stem = build_stem(
        mu=mu,
        tmax=cfg["tmax"],
        nsave=cfg["nsave"],
        Ny=cfg["Ny"],
        Ly=cfg["Ly"],
        margin=cfg["margin"],
        ic_method=cfg["ic_method"],
        t_save_window=cfg.get("t_save_window"),
    )

    out_path = raw_dir / f"{stem}.npz"
    fig_path = fig_dir / f"{stem}.png"
    ic_fig_path = fig_dir / f"{stem}_ic.png"

    if out_path.exists() and not cfg.get("overwrite", False):
        print(f"  skip (exists): {stem}")
        return

    print(f"  mu={mu:.3f} ...")
    result = solve_sh_pgb_diamond(
        Ly=cfg["Ly"],
        Ny=cfg["Ny"],
        mu=float(mu),
        h=cfg["h"],
        tmax=cfg["tmax"],
        nsave=cfg["nsave"],
        margin=cfg["margin"],
        margin_inner=cfg["margin_inner"],
        ic_method=cfg["ic_method"],
        Rscale=cfg["Rscale"],
        xlim_scale=cfg["xlim_scale"],
        tanh_scale=cfg["tanh_scale"],
        amp=cfg["amp"],
        sigma_R=cfg["sigma_R"],
        sigma_k=cfg["sigma_k"],
        knee_center_frac=cfg["knee_center_frac"],
        knee_stitch_width=cfg["knee_stitch_width"],
        energy=True,
        t_save_window=cfg.get("t_save_window"),
        save_initial_phase=cfg.get("save_initial_phase", True),
    )

    save_result(result, out_path)

    if not cfg.get("no_plot", False):
        make_summary_plot(result, fig_path, mu)
        make_ic_plot(result, ic_fig_path)


def cfg_from_args(args):
    if args.mu_list:
        mus = [float(m) for m in args.mu_list]
    elif args.mu is not None:
        mus = [float(args.mu)]
    else:
        raise ValueError("Provide --mu <value> or --mu_list <v1 v2 ...>")

    return {
        "mus": mus,
        "Ly": args.Ly,
        "Ny": args.Ny,
        "h": args.h,
        "tmax": args.tmax,
        "nsave": args.nsave,
        "t_save_window": args.t_save_window,
        "margin": args.margin,
        "margin_inner": args.margin_inner,
        "ic_method": args.ic_method,
        "Rscale": args.Rscale,
        "xlim_scale": args.xlim_scale,
        "tanh_scale": args.tanh_scale,
        "amp": args.amp,
        "sigma_R": args.sigma_R,
        "sigma_k": args.sigma_k,
        "knee_center_frac": args.knee_center_frac,
        "knee_stitch_width": args.knee_stitch_width,
        "save_initial_phase": not args.no_initial_phase,
        "output_dir": args.output_dir,
        "no_plot": args.no_plot,
        "overwrite": args.overwrite,
    }


def run_with_cfg(cfg):
    out_root = Path(cfg["output_dir"])
    raw_dir = ensure_dir(out_root / "raw")
    fig_dir = ensure_dir(out_root / "figures")

    mus = cfg["mus"]
    print(f"Running {len(mus)} mu value(s): {mus}")
    for mu in mus:
        process_mu(mu, cfg, raw_dir, fig_dir)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Run Swift-Hohenberg PGB diamond simulations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mu_grp = parser.add_mutually_exclusive_group()
    mu_grp.add_argument("--mu", type=float, default=None, help="Single mu value.")
    mu_grp.add_argument("--mu_list", type=float, nargs="+", default=None,
                        help="List of mu values for a sweep.")

    parser.add_argument("--Ly", type=float, default=50 * np.pi)
    parser.add_argument("--Ny", type=int, default=512)

    parser.add_argument("--h", type=float, default=0.5)
    parser.add_argument("--tmax", type=float, default=500.0)
    parser.add_argument("--nsave", type=int, default=1,
                        help="Snapshots after t=0. nsave=1 -> first+last only.")
    parser.add_argument("--t_save_window", type=float, default=None)

    parser.add_argument("--margin", type=float, default=0.45)
    parser.add_argument("--margin_inner", type=float, default=0.35)
    parser.add_argument("--ic_method", type=str, default="knee",
                        choices=["distance", "knee"])
    parser.add_argument("--Rscale", type=float, default=0.5)
    parser.add_argument("--xlim_scale", type=float, default=1.0)
    parser.add_argument("--tanh_scale", type=float, default=5.0)
    parser.add_argument("--amp", type=float, default=0.5)
    parser.add_argument("--sigma_R", type=float, default=1.0)
    parser.add_argument("--sigma_k", type=float, default=1.0)
    parser.add_argument("--knee_center_frac", type=float, default=0.5)
    parser.add_argument("--knee_stitch_width", type=float, default=None)

    parser.add_argument("--output_dir", type=str, default="results/sh_pgb_diamond")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--no_initial_phase", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--config", type=str, default=None,
                        help="JSON config file (overrides all CLI args).")

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            cfg = json.load(f)
    else:
        cfg = cfg_from_args(args)

    run_with_cfg(cfg)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        cfg = {
            "mus": [0.45],
            "Ly": 60 * np.pi,
            "Ny": 1536,
            "h": 0.5,
            "tmax": 50.0,
            "nsave": 2,
            "t_save_window": None,
            "margin": 0.45,
            "margin_inner": 0.35,
            "ic_method": "knee",
            "Rscale": 0.5,
            "xlim_scale": 1.0,
            "tanh_scale": 3.0,
            "amp": 0.5,
            "sigma_R": 1.0,
            "sigma_k": 1.0,
            "knee_center_frac": 0.5,
            "knee_stitch_width": None,
            "save_initial_phase": True,
            "output_dir": "data/sh_pgb_diamond/debug",
            "no_plot": False,
            "overwrite": True,
        }
        run_with_cfg(cfg)
    else:
        main()