# pipelines/run_uhu_diamond.py

"""
Order-parameter extraction for diamond SH data (k-only).

Follows the standardized runner style:
- load one standard SH .npz or batch a directory,
- build diamond ramp internally in the extractor,
- run uHu extraction frame-by-frame,
- save standardized OP .npz outputs and diagnostic figures.
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

from op_extract.uhu_diamond import compute_uhu_ops_diamond
from utils.spectral import SpectralDerivs
from utils.anim import make_director_quiver_panels, make_wavevector_quiver_panels


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_sh_npz(sh_path):
    """
    Load a standardised SH .npz and return consistent arrays.

    Expected keys: x, y, u (Ny, Nx, Nt), tt, and optionally metadata_json, mu.
    """
    sh_path = Path(sh_path)
    data = np.load(sh_path, allow_pickle=True)

    x = data["x"]
    y = data["y"]
    u = data["u"]

    if u.ndim != 3:
        raise ValueError(f"{sh_path.name}: expected u with ndim=3 (Ny, Nx, Nt), got {u.shape}")

    # ensure (Ny, Nx, Nt)
    if u.shape[0] == len(y) and u.shape[1] == len(x):
        pass
    elif u.shape[1] == len(y) and u.shape[2] == len(x):
        u = np.transpose(u, (1, 2, 0))  # (Nt, Ny, Nx) -> (Ny, Nx, Nt)
    else:
        raise ValueError(
            f"{sh_path.name}: cannot reconcile u shape {u.shape} "
            f"with x ({len(x)},) y ({len(y)},)"
        )

    tt = data["tt"] if "tt" in data else None

    metadata_json = None
    if "metadata_json" in data:
        raw = data["metadata_json"]
        metadata_json = raw.item() if hasattr(raw, "item") else str(raw)

    mu = None
    if metadata_json:
        try:
            mu = json.loads(metadata_json).get("mu")
        except Exception:
            pass
    if mu is None and "mu" in data:
        mu = float(data["mu"])

    return {
        "x":             x,
        "y":             y,
        "u":             u,
        "tt":            tt,
        "mu":            mu,
        "metadata_json": metadata_json,
        "source_path":   str(sh_path),
    }


def build_out_stem(sh_path, sigma, margin, tanh_scale, smooth_sigma):
    sh_path = Path(sh_path)
    smooth_str = "none" if smooth_sigma is None else f"{smooth_sigma:.2f}"
    return (
        f"{sh_path.stem}"
        f"_uhu_sigma{sigma:.3f}"
        f"_dm{margin:.3f}"
        f"_ts{tanh_scale:.2f}"
        f"_gs{smooth_str}"
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def masked_for_plot(field, ramp, tol=1e-10):
    # Focus on interior region: mask where ramp < ~1
    return np.ma.masked_where(ramp < (1.0 - tol), field)


def make_summary_plot(fields, ramp, sh, fig_path, cfg):
    """
    Diagnostic plot for OP fields.
    Top row: k(t0), k(t_final), phase(t_final), ramp.
    Bottom row: k1(t0), k1(t_final), k2(t0), k2(t_final).
    """
    u = sh["u"]
    x = sh["x"]
    y = sh["y"]
    Nt = u.shape[-1]

    extent = [x[0], x[-1], y[0], y[-1]]

    if Nt == 1:
        fi0 = fi1 = 0
    else:
        fi0 = 0
        fi1 = Nt - 1

    k = fields["k"]
    k1 = fields["k1"]
    k2 = fields["k2"]

    k_initial_plot = masked_for_plot(k[:, :, fi0], ramp)
    k_final_plot = masked_for_plot(k[:, :, fi1], ramp)
    k1_initial_plot = masked_for_plot(k1[:, :, fi0], ramp)
    k1_final_plot = masked_for_plot(k1[:, :, fi1], ramp)
    k2_initial_plot = masked_for_plot(k2[:, :, fi0], ramp)
    k2_final_plot = masked_for_plot(k2[:, :, fi1], ramp)

    phase_final = np.angle(k1[:, :, fi1] + 1j * k2[:, :, fi1])
    phase_final_plot = masked_for_plot(phase_final, ramp)

    def add_short_colorbar(fig, im, ax):
        return fig.colorbar(
            im,
            ax=ax,
            shrink=0.58,
            pad=0.02,
            fraction=0.05,
            aspect=20,
        )

    fig, axs = plt.subplots(2, 4, figsize=(18, 6))

    cmap_k = plt.cm.viridis.copy()
    cmap_k.set_bad(color="white")

    cmap_vec = plt.cm.coolwarm.copy()
    cmap_vec.set_bad(color="white")

    # top row
    im0 = axs[0, 0].imshow(k_initial_plot, cmap=cmap_k, origin="lower", extent=extent)
    axs[0, 0].set_title("k (initial)")
    add_short_colorbar(fig, im0, axs[0, 0])

    im1 = axs[0, 1].imshow(k_final_plot, cmap=cmap_k, origin="lower", extent=extent)
    axs[0, 1].set_title("k (final)")
    add_short_colorbar(fig, im1, axs[0, 1])

    im2 = axs[0, 2].imshow(phase_final_plot, cmap="twilight", origin="lower", extent=extent)
    axs[0, 2].set_title("arg(k1 + i k2) (final)")
    add_short_colorbar(fig, im2, axs[0, 2])

    im3 = axs[0, 3].imshow(ramp, cmap="gray", origin="lower", extent=extent)
    axs[0, 3].set_title("diamond ramp")
    add_short_colorbar(fig, im3, axs[0, 3])

    # bottom row
    im4 = axs[1, 0].imshow(k1_initial_plot, cmap=cmap_vec, origin="lower", extent=extent)
    axs[1, 0].set_title("k1 (initial)")
    add_short_colorbar(fig, im4, axs[1, 0])

    im5 = axs[1, 1].imshow(k1_final_plot, cmap=cmap_vec, origin="lower", extent=extent)
    axs[1, 1].set_title("k1 (final)")
    add_short_colorbar(fig, im5, axs[1, 1])

    im6 = axs[1, 2].imshow(k2_initial_plot, cmap=cmap_vec, origin="lower", extent=extent)
    axs[1, 2].set_title("k2 (initial)")
    add_short_colorbar(fig, im6, axs[1, 2])

    im7 = axs[1, 3].imshow(k2_final_plot, cmap=cmap_vec, origin="lower", extent=extent)
    axs[1, 3].set_title("k2 (final)")
    add_short_colorbar(fig, im7, axs[1, 3])

    for ax in axs.flat:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    mu_str = f"mu={sh['mu']:.3f}" if sh["mu"] is not None else ""
    fig.suptitle(
        f"{mu_str}  sigma={cfg['sigma']:.3f}  "
        f"margin={cfg['margin']:.3f}  tanh_scale={cfg['tanh_scale']:.2f}",
        y=0.97,
    )

    plt.subplots_adjust(wspace=0.35, hspace=0.15, top=0.87)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  summary plot -> {fig_path}")


def make_ramp_profile_plot(ramp, x, y, fig_path, cfg):
    """
    Ramp profile diagnostics for the diamond ramp:
    full horizontal and vertical profiles at the midlines.
    """
    Ny, Nx = ramp.shape
    mid_y = Ny // 2
    mid_x = Nx // 2

    horizontal_profile = ramp[mid_y, :]   # vary x at middle y
    vertical_profile   = ramp[:, mid_x]   # vary y at middle x

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Horizontal profile: middle y, vary x
    axs[0].plot(x, horizontal_profile, lw=2, color="black")
    axs[0].axhline(0.5, color="gray", ls=":", lw=1)
    axs[0].set_title("horizontal profile at middle y")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("ramp")
    axs[0].set_ylim(-0.05, 1.05)
    axs[0].grid(alpha=0.3)

    # Vertical profile: middle x, vary y
    axs[1].plot(y, vertical_profile, lw=2, color="black")
    axs[1].axhline(0.5, color="gray", ls=":", lw=1)
    axs[1].set_title("vertical profile at middle x")
    axs[1].set_xlabel("y")
    axs[1].set_ylabel("ramp")
    axs[1].set_ylim(-0.05, 1.05)
    axs[1].grid(alpha=0.3)

    title = (
        f"Diamond ramp profiles: margin={cfg['margin']:.3f}, "
        f"tanh_scale={cfg['tanh_scale']:.3f}, "
        f"smooth_sigma={cfg.get('smooth_sigma', None)}"
    )
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ramp profile plot -> {fig_path}")


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def run_uhu_diamond(sh_path, out_path, cfg):
    """
    Extract uHu order parameters (k-only) from one SH .npz file.
    """
    sh_path  = Path(sh_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sh = load_sh_npz(sh_path)
    if sh["mu"] is None:
        raise ValueError(f"{sh_path.name}: could not determine mu from metadata or npz keys.")

    x  = sh["x"]
    y  = sh["y"]
    u  = sh["u"]
    mu = sh["mu"]

    ops = compute_uhu_ops_diamond(
        u_ts=u,
        x=x,
        y=y,
        mu=mu,
        sigma=cfg["sigma"],
        deriv_fun=SpectralDerivs,
        margin=cfg["margin"],
        tanh_scale=cfg.get("tanh_scale", 1.0),
        smooth_sigma=cfg.get("smooth_sigma", None),
    )

    # build metadata
    uhu_meta = {
        "sigma":        cfg["sigma"],
        "margin":       cfg["margin"],
        "tanh_scale":   cfg.get("tanh_scale", 1.0),
        "smooth_sigma": cfg.get("smooth_sigma", None),
        "source_file":  str(sh_path),
        "mu":           mu,
    }

    save_dict = {
        "x":             x,
        "y":             y,
        "u":             u,
        "tt":            sh["tt"] if sh["tt"] is not None else np.array([]),
        "ramp":          ops["ramp"],
        "uhu_meta_json": json.dumps(uhu_meta),

        # basic OP fields (use orig for standardized k1/k2)
        "k":        ops["k"],
        "A":        ops["A"],
        "k1":       ops["k1_orig"],
        "k2":       ops["k2_orig"],
        "k1_sym":   ops["k1_sym"],
        "k2_sym":   ops["k2_sym"],
        "div_k":    ops["div_k"],
        "curl_k":   ops["curl_k"],
        "detJk":    ops["detJk"],
        "lam1":     ops["lam1"],
        "lam2":     ops["lam2"],
    }

    if sh["metadata_json"] is not None:
        save_dict["sh_meta_json"] = sh["metadata_json"]

    np.savez_compressed(out_path, **save_dict)
    print(f"  saved -> {out_path}")

    # also return a fields dict like zigzag/rectangular runner does
    fields = {
        "k":    ops["k"],
        "A":    ops["A"],
        "k1":   ops["k1_orig"],
        "k2":   ops["k2_orig"],
        "k1_sym": ops["k1_sym"],
        "k2_sym": ops["k2_sym"],
        "div_k": ops["div_k"],
        "curl_k": ops["curl_k"],
        "detJk": ops["detJk"],
        "lam1":  ops["lam1"],
        "lam2":  ops["lam2"],
        "ramp":  ops["ramp"],
    }
    return fields, sh


def process_one(sh_path, out_dir, fig_dir, cfg):
    sh_path = Path(sh_path)
    stem    = build_out_stem(
        sh_path,
        cfg["sigma"],
        cfg["margin"],
        cfg.get("tanh_scale", 1.0),
        cfg.get("smooth_sigma", None),
    )

    out_path        = out_dir / f"{stem}.npz"
    fig_path        = fig_dir / f"{stem}.png"
    fig_path_rp     = fig_dir / f"{stem}_ramp_profiles.png"
    fig_path_direct = fig_dir / f"{stem}_director_panels.png"
    fig_path_wave   = fig_dir / f"{stem}_wavevec_panels.png"

    if out_path.exists() and not cfg.get("overwrite", False):
        print(f"  skip (exists): {out_path.name}")
        return

    fields, sh = run_uhu_diamond(sh_path, out_path, cfg)

    ramp = fields["ramp"]
    x    = sh["x"]
    y    = sh["y"]
    u    = sh["u"]
    k1   = fields["k1"]
    k2   = fields["k2"]
    mask = ramp >= 0.99  # interior

    if not cfg.get("no_plot", False):
        make_summary_plot(fields, ramp, sh, fig_path, cfg)
        make_ramp_profile_plot(ramp, x, y, fig_path_rp, cfg)

        # director snapshots: default 2 panels (first + last)
        make_director_quiver_panels(
            orientation=np.angle(k1 + 1j * k2),
            u=u,
            x=x,
            y=y,
            fig_path=fig_path_direct,
            n_panels=2,
            step=36,
            mask=mask,
            suptitle="diamond director snapshots",
        )

        # wave-vector snapshots: default 2 panels (first + last)
        make_wavevector_quiver_panels(
            k1=k1,
            k2=k2,
            u=u,
            x=x,
            y=y,
            fig_path=fig_path_wave,
            n_panels=2,
            step=36,
            mask=mask,
            suptitle="diamond wave-vector snapshots",
            scale=None,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_with_cfg(cfg, args):
    out_root = Path(cfg["output_dir"])
    raw_dir  = ensure_dir(out_root / "raw")
    fig_dir  = ensure_dir(out_root / "figures")

    if args.sh_path is not None:
        process_one(args.sh_path, raw_dir, fig_dir, cfg)
    elif args.all or cfg.get("input_dir"):
        input_dir = Path(cfg.get("input_dir", args.input_dir))
        sh_files  = sorted(input_dir.glob("*.npz"))
        if not sh_files:
            raise FileNotFoundError(f"No .npz files found in {input_dir}")
        print(f"Processing {len(sh_files)} file(s) from {input_dir}")
        for sh_path in sh_files:
            process_one(sh_path, raw_dir, fig_dir, cfg)
    else:
        raise SystemExit(
            "\nProvide a sh_path argument or use --all with --input_dir."
        )

    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute uHu order parameters for diamond SH data (k-only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("sh_path", nargs="?", default=None,
                        help="Path to one SH .npz file (omit for --all mode).")
    parser.add_argument("--all",       action="store_true",
                        help="Process all .npz files in --input_dir.")
    parser.add_argument("--input_dir", type=str,
                        default="results/sh_diamond/raw",
                        help="Directory scanned with --all.")
    parser.add_argument("--output_dir", type=str,
                        default="results/sh_diamond/ops",
                        help="Root output directory for OP .npz files.")

    # uHu knobs
    parser.add_argument("--sigma",        type=float, default=float("nan"),
                        help="uHu smoothing scale. Default: pi/2.")
    parser.add_argument("--margin",       type=float, default=0.22)
    parser.add_argument("--tanh_scale",   type=float, default=1.0)
    parser.add_argument("--smooth_sigma", type=float, default=float("nan"))

    parser.add_argument("--no_plot",   action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config",    type=str, default=None,
                        help="JSON config file (overrides all CLI args).")

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            cfg = json.load(f)
    else:
        sigma = args.sigma if not (args.sigma != args.sigma) else np.pi / 2.0
        smooth_sigma = None if (args.smooth_sigma != args.smooth_sigma) else args.smooth_sigma

        cfg = {
            "sigma":        sigma,
            "margin":       args.margin,
            "tanh_scale":   args.tanh_scale,
            "smooth_sigma": smooth_sigma,
            "output_dir":   args.output_dir,
            "no_plot":      args.no_plot,
            "overwrite":    args.overwrite,
            "input_dir":    args.input_dir,
        }

    run_with_cfg(cfg, args)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Debug-friendly default config: tweak these in PyCharm, then Run/Debug
        class Args:
            sh_path     = None
            all         = True

            # input_dir  = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_diamond/debug/raw"
            # output_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_diamond/debug_uhu/sig_pio2/"
            
            input_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_diamond/debug/raw"
            output_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_diamond/debug_uhu/sig_2pi/"

            #sigma = np.pi / 16.0
            #sigma = np.pi / 2.0
            #sigma = 1.0
            sigma = 2*np.pi

            margin       = 0.4
            tanh_scale   = 0.4
            smooth_sigma = None

            no_plot    = False
            overwrite  = True
            config     = None

        cfg = {
            "sigma":        Args.sigma,
            "margin":       Args.margin,
            "tanh_scale":   Args.tanh_scale,
            "smooth_sigma": Args.smooth_sigma,
            "output_dir":   Args.output_dir,
            "no_plot":      Args.no_plot,
            "overwrite":    Args.overwrite,
            "input_dir":    Args.input_dir,
        }

        run_with_cfg(cfg, Args)
    else:
        main()