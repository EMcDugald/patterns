# pipelines/run_uhu_pgb.py
"""
Order-parameter extraction for PGB zigzag SH data.

Reads a standardised SH .npz file (produced by run_sh_pgb_zigzag.py),
builds a rectangular ramp, runs uHu extraction frame-by-frame, and
writes a standardised OP .npz file.

Usage examples
--------------
# --- Single SH file ---
python run_uhu_pgb.py results/sh_pgb_zigzag/raw/sh_pgb_zigzag_mu0.500_T500_N1_nx512_Nyfull_lower.npz

# --- Batch over a directory (all .npz files) ---
python run_uhu_pgb.py --all --input_dir results/sh_pgb_zigzag/raw

# --- From a JSON config file ---
python run_uhu_pgb.py --config my_uhu.json
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

from op_extract.uhu import compute_uhu_ops_ramp
from utils.geometry import build_rectangular_ramp_smooth
from utils.spectral import SpectralDerivs

from utils.anim import (
    make_director_quiver_panels,
    make_wavevector_quiver_panels,
)

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
        raise ValueError(
            f"{sh_path.name}: expected u with ndim=3 (Ny, Nx, Nt), got {u.shape}"
        )
    # guard: ensure (Ny, Nx, Nt) — if loaded as (Nt, Ny, Nx) transpose
    if u.shape[0] == len(y) and u.shape[1] == len(x):
        pass  # already (Ny, Nx, Nt)
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
        "u":             u,           # (Ny, Nx, Nt)
        "tt":            tt,
        "mu":            mu,
        "metadata_json": metadata_json,
        "source_path":   str(sh_path),
    }


def build_out_stem(sh_path, sigma, xmargin, ymargin, tanhscale):
    sh_path = Path(sh_path)
    return (
        f"{sh_path.stem}"
        f"_uhu_sigma{sigma:.3f}"
        f"_xm{xmargin:.2f}_ym{ymargin:.2f}_ts{tanhscale:.1f}"
    )


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def run_uhu_pgb(sh_path, out_path, cfg):
    """
    Extract uHu order parameters from one SH .npz file.

    Parameters
    ----------
    sh_path : str or Path
    out_path : str or Path
    cfg : dict   must contain sigma, xmargin, ymargin, tanhscale
    """
    sh_path  = Path(sh_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sigma     = cfg["sigma"]
    xmargin   = cfg.get("xmargin",   0.10)
    ymargin   = cfg.get("ymargin",   0.10)
    tanhscale = cfg.get("tanhscale", 7.5)

    sh = load_sh_npz(sh_path)
    x  = sh["x"]
    y  = sh["y"]
    u  = sh["u"]          # (Ny, Nx, Nt)
    Ny, Nx, Nt = u.shape

    # --- ramp ---
    ramp = build_rectangular_ramp_smooth(
        x, y,
        xmargin   = xmargin,
        ymargin   = ymargin,
        tanhscale = tanhscale,
    )

    # --- per-frame extraction ---
    fields = {
        "k":    np.empty((Ny, Nx, Nt)),
        "A":    np.empty((Ny, Nx, Nt)),
        "k1":   np.empty((Ny, Nx, Nt)),
        "k2":   np.empty((Ny, Nx, Nt)),
        "lam1": np.empty((Ny, Nx, Nt)),
        "lam2": np.empty((Ny, Nx, Nt)),
    }

    for t in range(Nt):
        print(f"  uHu frame {t+1}/{Nt}  ({sh_path.name})")
        ops = compute_uhu_ops_ramp(
            u[:, :, t], x, y, sigma, SpectralDerivs, ramp=ramp
        )
        fields["k"]   [:, :, t] = ops["k"]
        fields["A"]   [:, :, t] = ops["A"]
        fields["k1"]  [:, :, t] = ops["k1_orig"]
        fields["k2"]  [:, :, t] = ops["k2_orig"]
        fields["lam1"][:, :, t] = ops["lam1"]
        fields["lam2"][:, :, t] = ops["lam2"]

    # --- build uhu metadata ---
    uhu_meta = {
        "sigma":       sigma,
        "xmargin":     xmargin,
        "ymargin":     ymargin,
        "tanhscale":   tanhscale,
        "source_file": str(sh_path),
    }
    if sh["mu"] is not None:
        uhu_meta["mu"] = sh["mu"]

    save_dict = {
        "x":             x,
        "y":             y,
        "u":             u,
        "tt":            sh["tt"] if sh["tt"] is not None else np.array([]),
        "ramp":          ramp,
        "uhu_meta_json": json.dumps(uhu_meta),
    }
    if sh["metadata_json"] is not None:
        save_dict["sh_meta_json"] = sh["metadata_json"]

    save_dict.update(fields)
    np.savez_compressed(out_path, **save_dict)
    print(f"  saved -> {out_path}")
    return fields, ramp, sh


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def masked_for_plot(field, ramp, tol=1e-12):
    # Focus on interior region: mask near boundaries where ramp < ~1
    return np.ma.masked_where(ramp < (1.0 - tol), field)


def make_summary_plot(fields, ramp, sh, fig_path, cfg):
    """
    Diagnostic plot for OP fields.
    Top row: k(t0), k(t_final), phase(t_final), ramp.
    Bottom row: k1(t0), k1(t_final), k2(t0), k2(t_final).
    """
    u  = sh["u"]
    x  = sh["x"]
    y  = sh["y"]
    Nt = u.shape[-1]

    extent = [x[0], x[-1], y[0], y[-1]]

    # choose frames
    if Nt == 1:
        fi0 = fi1 = 0
    else:
        fi0 = 0
        fi1 = Nt - 1

    k   = fields["k"]
    k1  = fields["k1"]
    k2  = fields["k2"]

    k_initial_plot = masked_for_plot(k[:, :, fi0], ramp)
    k_final_plot   = masked_for_plot(k[:, :, fi1], ramp)
    k1_initial_plot = masked_for_plot(k1[:, :, fi0], ramp)
    k1_final_plot   = masked_for_plot(k1[:, :, fi1], ramp)
    k2_initial_plot = masked_for_plot(k2[:, :, fi0], ramp)
    k2_final_plot   = masked_for_plot(k2[:, :, fi1], ramp)

    phase_final = np.angle(k1[:, :, fi1] + 1j * k2[:, :, fi1])
    phase_final_plot = masked_for_plot(phase_final, ramp)

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

    cmap_k = plt.cm.viridis.copy()
    cmap_k.set_bad(color="white")

    cmap_vec = plt.cm.coolwarm.copy()
    cmap_vec.set_bad(color="white")

    # top row: k(t0), k(t1), phase(t1), ramp
    im0 = axs[0, 0].imshow(k_initial_plot, cmap=cmap_k, origin="lower", extent=extent)
    axs[0, 0].set_title("k  (initial)")
    fig.colorbar(im0, ax=axs[0, 0], shrink=0.8)

    im1 = axs[0, 1].imshow(k_final_plot, cmap=cmap_k, origin="lower", extent=extent)
    axs[0, 1].set_title("k  (final)")
    fig.colorbar(im1, ax=axs[0, 1], shrink=0.8)

    im2 = axs[0, 2].imshow(phase_final_plot, cmap="twilight", origin="lower", extent=extent)
    axs[0, 2].set_title("arg(k1 + i k2)  (final)")
    fig.colorbar(im2, ax=axs[0, 2], shrink=0.8)

    im3 = axs[0, 3].imshow(ramp, cmap="gray", origin="lower", extent=extent)
    axs[0, 3].set_title("rectangular ramp")
    fig.colorbar(im3, ax=axs[0, 3], shrink=0.8)

    # bottom row: k1/k2 first/last
    im4 = axs[1, 0].imshow(k1_initial_plot, cmap=cmap_vec, origin="lower", extent=extent)
    axs[1, 0].set_title("k1  (initial)")
    fig.colorbar(im4, ax=axs[1, 0], shrink=0.8)

    im5 = axs[1, 1].imshow(k1_final_plot, cmap=cmap_vec, origin="lower", extent=extent)
    axs[1, 1].set_title("k1  (final)")
    fig.colorbar(im5, ax=axs[1, 1], shrink=0.8)

    im6 = axs[1, 2].imshow(k2_initial_plot, cmap=cmap_vec, origin="lower", extent=extent)
    axs[1, 2].set_title("k2  (initial)")
    fig.colorbar(im6, ax=axs[1, 2], shrink=0.8)

    im7 = axs[1, 3].imshow(k2_final_plot, cmap=cmap_vec, origin="lower", extent=extent)
    axs[1, 3].set_title("k2  (final)")
    fig.colorbar(im7, ax=axs[1, 3], shrink=0.8)

    for ax in axs.flat:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    mu_str = f"mu={sh['mu']:.3f}" if sh["mu"] is not None else ""
    fig.suptitle(
        f"{mu_str}  sigma={cfg['sigma']:.3f}  "
        f"xmargin={cfg.get('xmargin', 0.1):.2f}  ymargin={cfg.get('ymargin', 0.1):.2f}",
        y=0.99,
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  summary plot -> {fig_path}")


def make_ramp_profile_plot(ramp, x, y, fig_path, cfg):
    """
    Ramp profile diagnostics: full horizontal/vertical profiles and zoomed
    views around the transition midpoints.
    """
    Ny, Nx = ramp.shape
    mid_y = Ny // 2
    mid_x = Nx // 2

    horizontal_profile = ramp[mid_y, :]   # vary x at middle y
    vertical_profile   = ramp[:, mid_x]   # vary y at middle x

    x0, x1 = x[0], x[-1]
    y0, y1 = y[0], y[-1]
    Lx = x1 - x0
    Ly = y1 - y0

    xmargin = cfg.get("xmargin", 0.1)
    ymargin = cfg.get("ymargin", 0.1)

    # Transition-band midpoints
    x_left_mid   = x0 + 0.5 * xmargin * Lx
    x_right_mid  = x1 - 0.5 * xmargin * Lx
    y_bottom_mid = y0 + 0.5 * ymargin * Ly
    y_top_mid    = y1 - 0.5 * ymargin * Ly

    def nearest_index(arr, value):
        return int(np.argmin(np.abs(arr - value)))

    def fixed_window(n, center_idx, half_width=15):
        start = center_idx - half_width
        stop  = center_idx + half_width + 1

        if start < 0:
            stop += -start
            start = 0
        if stop > n:
            start -= stop - n
            stop = n

        start = max(start, 0)
        stop  = min(stop, n)
        return slice(start, stop)

    ix_left_mid   = nearest_index(x, x_left_mid)
    iy_bottom_mid = nearest_index(y, y_bottom_mid)

    sx = fixed_window(len(x), ix_left_mid, half_width=15)
    sy = fixed_window(len(y), iy_bottom_mid, half_width=15)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Full horizontal profile: middle y, vary x
    axs[0, 0].plot(x, horizontal_profile, lw=2, color="black")
    axs[0, 0].axvline(x_left_mid,  color="tab:red",  ls="--", lw=1, label="left transition midpoint")
    axs[0, 0].axvline(x_right_mid, color="tab:blue", ls="--", lw=1, label="right transition midpoint")
    axs[0, 0].axhline(0.5, color="gray", ls=":", lw=1)
    axs[0, 0].set_title("horizontal profile at middle y")
    axs[0, 0].set_xlabel("x")
    axs[0, 0].set_ylabel("ramp")
    axs[0, 0].set_ylim(-0.05, 1.05)
    axs[0, 0].grid(alpha=0.3)
    axs[0, 0].legend()

    # Full vertical profile: middle x, vary y
    axs[0, 1].plot(y, vertical_profile, lw=2, color="black")
    axs[0, 1].axvline(y_bottom_mid, color="tab:red",  ls="--", lw=1, label="bottom transition midpoint")
    axs[0, 1].axvline(y_top_mid,    color="tab:blue", ls="--", lw=1, label="top transition midpoint")
    axs[0, 1].axhline(0.5, color="gray", ls=":", lw=1)
    axs[0, 1].set_title("vertical profile at middle x")
    axs[0, 1].set_xlabel("y")
    axs[0, 1].set_ylabel("ramp")
    axs[0, 1].set_ylim(-0.05, 1.05)
    axs[0, 1].grid(alpha=0.3)
    axs[0, 1].legend()

    # Zoomed horizontal profile around LEFT transition midpoint
    axs[1, 0].plot(x[sx], horizontal_profile[sx], "o-", ms=4, lw=1.5, color="black")
    axs[1, 0].axvline(x_left_mid, color="tab:red", ls="--", lw=1, label="left transition midpoint")
    axs[1, 0].axhline(0.5, color="gray", ls=":", lw=1)
    axs[1, 0].set_title("horizontal zoom (31 pts, left transition midpoint)")
    axs[1, 0].set_xlabel("x")
    axs[1, 0].set_ylabel("ramp")
    axs[1, 0].set_ylim(-0.05, 1.05)
    axs[1, 0].grid(alpha=0.3)
    axs[1, 0].legend()

    # Zoomed vertical profile around BOTTOM transition midpoint
    axs[1, 1].plot(y[sy], vertical_profile[sy], "o-", ms=4, lw=1.5, color="black")
    axs[1, 1].axvline(y_bottom_mid, color="tab:red", ls="--", lw=1, label="bottom transition midpoint")
    axs[1, 1].axhline(0.5, color="gray", ls=":", lw=1)
    axs[1, 1].set_title("vertical zoom (31 pts, bottom transition midpoint)")
    axs[1, 1].set_xlabel("y")
    axs[1, 1].set_ylabel("ramp")
    axs[1, 1].set_ylim(-0.05, 1.05)
    axs[1, 1].grid(alpha=0.3)
    axs[1, 1].legend()

    title = (
        f"Ramp profiles: xmargin={xmargin:.3f}, ymargin={ymargin:.3f}, "
        f"tanhscale={cfg.get('tanhscale', 7.5):.3f}"
    )
    fig.suptitle(title, y=0.98)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"  ramp profile plot -> {fig_path}")


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_one(sh_path, out_dir, fig_dir, cfg):
    sh_path = Path(sh_path)
    stem    = build_out_stem(
        sh_path,
        cfg["sigma"],
        cfg.get("xmargin",   0.10),
        cfg.get("ymargin",   0.10),
        cfg.get("tanhscale", 7.5),
    )
    out_path    = out_dir / f"{stem}.npz"
    fig_path    = fig_dir  / f"{stem}.png"
    fig_path_rp = fig_dir  / f"{stem}_ramp_profiles.png"

    if out_path.exists() and not cfg.get("overwrite", False):
        print(f"  skip (exists): {out_path.name}")
        return

    fields, ramp, sh = run_uhu_pgb(sh_path, out_path, cfg)

    # basic arrays
    x = sh["x"]
    y = sh["y"]
    u = sh["u"]
    k1 = fields["k1"]
    k2 = fields["k2"]
    Nt = u.shape[-1]

    # interior mask from ramp
    mask = ramp >= 0.99  # interior region

    # filenames for multi-panel quiver plots
    fig_path_director_panels = fig_dir / f"{stem}_director_panels.png"
    fig_path_wavevec_panels = fig_dir / f"{stem}_wavevec_panels.png"

    if not cfg.get("no_plot", False):
        # summary + ramp diagnostics
        make_summary_plot(fields, ramp, sh, fig_path, cfg)
        make_ramp_profile_plot(ramp, x, y, fig_path_rp, cfg)

        # director: default 2 panels (first + last)
        make_director_quiver_panels(
            orientation=np.angle(k1 + 1j * k2),
            u=u,
            x=x, y=y,
            fig_path=fig_path_director_panels,
            n_panels=2,  # change to 3, 4, ... if desired
            step=12,
            mask=mask,
            suptitle="director snapshots",
        )

        # wave-vector: default 2 panels (first + last)
        make_wavevector_quiver_panels(
            k1=k1,
            k2=k2,
            u=u,
            x=x, y=y,
            fig_path=fig_path_wavevec_panels,
            n_panels=2,  # change to 3, 4, ... if desired
            step=12,
            mask=mask,
            suptitle="wave-vector snapshots",
            scale=None,  # set e.g. 40 if arrows are too long
        )


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
        description="Compute uHu order parameters for PGB zigzag SH data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("sh_path", nargs="?", default=None,
                        help="Path to one SH .npz file (omit for --all mode).")
    parser.add_argument("--all",       action="store_true",
                        help="Process all .npz files in --input_dir.")
    parser.add_argument("--input_dir", type=str,
                        default="results/sh_pgb_zigzag/raw",
                        help="Directory scanned with --all.")
    parser.add_argument("--output_dir", type=str,
                        default="results/sh_pgb_zigzag/ops",
                        help="Root output directory for OP .npz files.")

    # uHu knobs
    parser.add_argument("--sigma",     type=float, default=float("nan"),
                        help="uHu smoothing scale. Default: pi/2.")
    parser.add_argument("--xmargin",   type=float, default=0.10)
    parser.add_argument("--ymargin",   type=float, default=0.10)
    parser.add_argument("--tanhscale", type=float, default=7.5)

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
        cfg = {
            "sigma":     sigma,
            "xmargin":   args.xmargin,
            "ymargin":   args.ymargin,
            "tanhscale": args.tanhscale,
            "output_dir":args.output_dir,
            "no_plot":   args.no_plot,
            "overwrite": args.overwrite,
            "input_dir": args.input_dir,
        }

    run_with_cfg(cfg, args)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Debug-friendly default config: tweak these in PyCharm, then Run/Debug
        class Args:
            sh_path     = None        # or set to a specific file path
            all         = True        # process all files in input_dir
            # input_dir   = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweep/raw"
            # output_dir  = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweep_uhu/"
            # input_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweep_2/raw"
            # output_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweep_uhu_2_sig1/"
            # input_dir = "//Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/debug/raw"
            # output_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/debug_uhu/"
            # input_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/debug_2/raw"
            # output_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/debug_uhu_2/"
            # input_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweep_3/raw"
            # output_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweep_uhu_3_sig_1p5pio2/"
            # input_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweep_5/raw"
            # output_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweep_uhu_5_sig_pio4/"
            input_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/full_run_2/raw"
            output_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/uhu_full_run_2_sig1/"
            #sigma       = np.pi / 4.0
            sigma = 1.0
            #sigma = .50
            #sigma = 1.1
            xmargin     = 0.025
            ymargin     = 0.025
            tanhscale = 120.0
            no_plot     = False
            overwrite   = True
            config      = None

        cfg = {
            "sigma":     Args.sigma,
            "xmargin":   Args.xmargin,
            "ymargin":   Args.ymargin,
            "tanhscale": Args.tanhscale,
            "output_dir":Args.output_dir,
            "no_plot":   Args.no_plot,
            "overwrite": Args.overwrite,
            "input_dir": Args.input_dir,
        }
        run_with_cfg(cfg, Args)
    else:
        main()


# =============================================================================
# Example configs (for reference / copy-paste into JSON files)
# =============================================================================
#
# --- 1) Single file, many snapshots (time-resolved OP study) ---
#
# python run_uhu_pgb.py \
#     results/sh_pgb_zigzag/raw/sh_pgb_zigzag_mu0.500_T5000_N100_nx512_Ny257_lower.npz \
#     --sigma   1.5708 \
#     --xmargin 0.10 --ymargin 0.10 --tanhscale 7.5 \
#     --output_dir results/sh_pgb_zigzag/ops_time_resolved
#
# --- 2) Batch over directory (mu sweep, first+last or tail window) ---
#
# python run_uhu_pgb.py \
#     --all --input_dir results/sh_pgb_zigzag/raw \
#     --sigma   1.5708 \
#     --xmargin 0.10 --ymargin 0.10 --tanhscale 7.5 \
#     --output_dir results/sh_pgb_zigzag/ops_mu_sweep
#
# --- 3) From JSON config file ---
#
# python run_uhu_pgb.py --config my_uhu.json