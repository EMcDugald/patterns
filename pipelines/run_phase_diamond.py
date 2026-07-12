import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

from op_extract.phase_diamond import (
    compute_diamond_phase_from_uhu,
    load_uhu_npz_diamond,
    make_coordinate_line_diagnostic_diamond,
    make_geometry_diagnostic_plot_diamond,
    make_phase_profile_plot_diamond,
    make_phase_summary_plot_diamond,
    postprocess_phase_amplitude,
)


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p



def build_out_stem(
    uhu_path,
    n_phase_seeds,
    ds,
    phase_ramp_mode,
    phase_margin,
    phase_tanh_scale,
    phase_smooth_sigma,
    ramp_sample_thresh,
    boundary_source,
    prefer_sym,
):
    uhu_path = Path(uhu_path)
    stem = (
        f"{uhu_path.stem}"
        f"_phase_diamond"
        f"_ns{int(n_phase_seeds)}"
        f"_ds{ds:.3f}"
        f"_bd{boundary_source}"
        f"_k{'sym' if prefer_sym else 'orig'}"
        f"_prm{phase_ramp_mode}"
    )
    if phase_ramp_mode == "rebuild":
        stem += (
            f"_pm{phase_margin:.3f}"
            f"_pt{phase_tanh_scale:.3f}"
            f"_ps{phase_smooth_sigma:.2f}"
            f"_rst{ramp_sample_thresh:.3f}"
        )
    elif phase_ramp_mode == "saved":
        stem += f"_rst{ramp_sample_thresh:.3f}"
    return stem



def run_diamond_phase(uhu_path, out_path, cfg):
    uhu_path = Path(uhu_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    uhu = load_uhu_npz_diamond(uhu_path)

    print(
        "[diamond_phase_runner] "
        f"boundary_source={cfg.get('boundary_source', 'inner')} "
        f"prefer_sym={cfg.get('prefer_sym', True)} "
        f"phase_ramp_mode={cfg.get('phase_ramp_mode', 'saved')}"
    )

    result = compute_diamond_phase_from_uhu(
        uhu_data=uhu,
        mu=cfg.get("mu"),
        n_phase_seeds=cfg.get("n_phase_seeds", 128),
        ds=cfg.get("ds", 0.2),
        max_steps=cfg.get("max_steps", 10000),
        prefer_sym=cfg.get("prefer_sym", True),
        phase_ramp_mode=cfg.get("phase_ramp_mode", "saved"),
        phase_margin=cfg.get("phase_margin"),
        phase_tanh_scale=cfg.get("phase_tanh_scale", 1.0),
        phase_smooth_sigma=cfg.get("phase_smooth_sigma", 1.0),
        ramp_sample_thresh=cfg.get("ramp_sample_thresh", 0.0),
        boundary_source=cfg.get("boundary_source", "inner"),
    )
    return uhu, result



def save_diamond_phase_result(result, uhu, out_path):
    phase_meta_json = json.dumps(result["phase_meta"])
    save_dict = {
        "x": result["x"],
        "y": result["y"],
        "u": result["u"],
        "tt": result["tt"] if result.get("tt") is not None else np.array([]),
        "mu": result["mu"],
        "ramp": result.get("ramp"),
        "phase_ramp": result.get("phase_ramp"),
        "k": result.get("k"),
        "A": result.get("A"),
        "k1_sym": result.get("k1_sym"),
        "k2_sym": result.get("k2_sym"),
        "k1_orig": result.get("k1_orig"),
        "k2_orig": result.get("k2_orig"),
        "bdry": result.get("bdry"),
        "bdry_inner": result.get("bdry_inner"),
        "upper_bdry_phase": result.get("upper_bdry_phase"),
        "phase_meta_json": phase_meta_json,
        "coordinate_lines": result["coordinate_lines"],
        "phase_lines_wrapped": result["phase_lines_wrapped"],
        "phase_lines_unwrapped": result["phase_lines_unwrapped"],
        "phase_grid_wrapped": result["phase_grid_wrapped"],
        "phase_grid_unwrapped": result["phase_grid_unwrapped"],
        "phase_lines_symmetric_wrapped": result["phase_lines_symmetric_wrapped"],
        "phase_lines_symmetric_unwrapped": result["phase_lines_symmetric_unwrapped"],
        "phase_grid_symmetric_wrapped": result["phase_grid_symmetric_wrapped"],
        "phase_grid_symmetric_unwrapped": result["phase_grid_symmetric_unwrapped"],
        "analytic_amplitude_lines": result["analytic_amplitude_lines"],
        "analytic_amplitude_grid": result["analytic_amplitude_grid"],
        "analytic_amplitude_lines_symmetric": result["analytic_amplitude_lines_symmetric"],
        "analytic_amplitude_grid_symmetric": result["analytic_amplitude_grid_symmetric"],
    }

    if uhu.get("uhu_meta_json") is not None:
        save_dict["uhu_meta_json"] = uhu["uhu_meta_json"]
    if uhu.get("sh_meta_json") is not None:
        save_dict["sh_meta_json"] = uhu["sh_meta_json"]

    if result.get("postprocess_meta") is not None:
        save_dict["postprocess_meta_json"] = json.dumps(result["postprocess_meta"])

    for key in (
        "phase_grid_wrapped_smooth",
        "phase_grid_unwrapped_smooth",
        "analytic_amplitude_grid_smooth",
        "phase_grid_symmetric_wrapped_smooth",
        "phase_grid_symmetric_unwrapped_smooth",
        "analytic_amplitude_grid_symmetric_smooth",
    ):
        if key in result:
            save_dict[key] = result[key]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **save_dict)
    print(f" saved phase -> {out_path}")



def infer_mu_from_path_or_data(uhu_path):
    uhu = load_uhu_npz_diamond(uhu_path)
    if uhu.get("mu") is not None:
        return float(uhu["mu"])

    m = re.search(r"mu([0-9]*\.?[0-9]+)", Path(uhu_path).stem)
    if m:
        return float(m.group(1))
    return None



def make_boundary_geometry_plot(result, fig_path):
    x = result["x"]
    y = result["y"]
    u = result["u"]
    upper_bdry = result.get("upper_bdry_phase")
    phase_ramp = result.get("phase_ramp", None)
    extent = [x[0], x[-1], y[0], y[-1]]

    fi = u.shape[-1] - 1 if u.ndim == 3 else 0
    u_plot = u[:, :, fi] if u.ndim == 3 else u

    if phase_ramp is None:
        phase_ramp = np.ones_like(u_plot)

    u_ramped = u_plot * phase_ramp

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    vmax0 = np.nanmax(np.abs(u_plot))
    vmax1 = np.nanmax(np.abs(u_ramped))
    vmax0 = 1.0 if not np.isfinite(vmax0) or vmax0 == 0 else vmax0
    vmax1 = 1.0 if not np.isfinite(vmax1) or vmax1 == 0 else vmax1

    im0 = axs[0].imshow(
        u_plot,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-vmax0,
        vmax=vmax0,
    )
    if upper_bdry is not None:
        axs[0].plot(upper_bdry[0], upper_bdry[1], "k.", ms=2.5, alpha=0.9)
    axs[0].set_title("pattern with phase-seeding boundary")
    fig.colorbar(im0, ax=axs[0], shrink=0.85)

    im1 = axs[1].imshow(
        u_ramped,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-vmax1,
        vmax=vmax1,
    )
    if upper_bdry is not None:
        axs[1].plot(upper_bdry[0], upper_bdry[1], "k.", ms=2.5, alpha=0.9)
    axs[1].set_title("pattern × phase ramp")
    fig.colorbar(im1, ax=axs[1], shrink=0.85)

    for ax in axs:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    meta = result.get("phase_meta", {})
    mu = result.get("mu", None)
    mu_str = f"mu={mu:.3f}" if mu is not None else "mu=?"
    ramp_mode = meta.get("phase_ramp_mode", "saved")
    fig.suptitle(f"{mu_str}   phase_ramp_mode={ramp_mode}", y=0.98)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def process_one(uhu_path, out_dir, fig_dir, cfg):
    uhu_path = Path(uhu_path)

    min_mu = cfg.get("min_mu", None)
    if min_mu is not None:
        mu_this = infer_mu_from_path_or_data(uhu_path)
        if mu_this is None:
            print(f"  skip (could not infer mu): {uhu_path.name}")
            return
        if mu_this < min_mu:
            print(f"  skip (mu={mu_this:.3f} < min_mu={min_mu:.3f}): {uhu_path.name}")
            return

    stem = build_out_stem(
        uhu_path,
        cfg.get("n_phase_seeds", 128),
        cfg.get("ds", 0.2),
        cfg.get("phase_ramp_mode", "saved"),
        cfg.get("phase_margin", 0.0) if cfg.get("phase_margin") is not None else 0.0,
        cfg.get("phase_tanh_scale", 1.0),
        cfg.get("phase_smooth_sigma", 1.0),
        cfg.get("ramp_sample_thresh", 0.0),
        cfg.get("boundary_source", "inner"),
        cfg.get("prefer_sym", True),
    )

    out_path = out_dir / f"{stem}.npz"
    fig_path_initial = fig_dir / f"{stem}_initial.png"
    fig_path_final = fig_dir / f"{stem}_final.png"
    fig_path_coord = fig_dir / f"{stem}_coord_lines.png"
    fig_path_geom = fig_dir / f"{stem}_geometry.png"
    fig_path_profiles_raw_initial = fig_dir / f"{stem}_profiles_raw_initial.png"
    fig_path_profiles_raw_final = fig_dir / f"{stem}_profiles_raw_final.png"
    fig_path_profiles_smooth_initial = fig_dir / f"{stem}_profiles_smooth_initial.png"
    fig_path_profiles_smooth_final = fig_dir / f"{stem}_profiles_smooth_final.png"

    if out_path.exists() and not cfg.get("overwrite", False):
        print(f" skip (exists): {out_path.name}")
        return

    uhu, result = run_diamond_phase(uhu_path, out_path, cfg)

    if cfg.get("postprocess", False):
        sigma = cfg.get("postprocess_sigma", 1.0)
        result = postprocess_phase_amplitude(result, sigma=sigma)
        result["postprocess_meta"] = {
            "sigma": float(sigma),
            "method": "gaussian_filter",
        }

    save_diamond_phase_result(result, uhu, out_path)

    if not cfg.get("no_plot", False):
        make_phase_summary_plot_diamond(
            result,
            fig_path_initial,
            prefer_sym=cfg.get("prefer_sym", True),
            frame_index=0,
            use_smoothed=False,
            mask_tol=cfg.get("profile_mask_tol", 0.20),
        )
        print(f" initial phase plot -> {fig_path_initial}")

        make_phase_summary_plot_diamond(
            result,
            fig_path_final,
            prefer_sym=cfg.get("prefer_sym", True),
            frame_index=-1,
            use_smoothed=False,
            mask_tol=cfg.get("profile_mask_tol", 0.20),
        )
        print(f" final phase plot -> {fig_path_final}")

        make_coordinate_line_diagnostic_diamond(
            result,
            fig_path_coord,
            frame_index=-1,
            n_show=cfg.get("coord_n_lines", 12),
        )
        print(f" coordinate-line plot -> {fig_path_coord}")

        make_boundary_geometry_plot(result, fig_path_geom)
        print(f" geometry plot -> {fig_path_geom}")

        profile_jobs = [
            ("raw", False, "initial", 0, fig_path_profiles_raw_initial),
            ("raw", False, "final", -1, fig_path_profiles_raw_final),
        ]

        if cfg.get("postprocess", False):
            profile_jobs += [
                ("smoothed", True, "initial", 0, fig_path_profiles_smooth_initial),
                ("smoothed", True, "final", -1, fig_path_profiles_smooth_final),
            ]

        for tag, use_smoothed, frame_tag, frame_index, profile_path in profile_jobs:
            make_phase_profile_plot_diamond(
                result,
                profile_path,
                prefer_sym=cfg.get("prefer_sym", True),
                frame_index=frame_index,
                use_smoothed=use_smoothed,
                mask_tol=cfg.get("profile_mask_tol", 0.20),
            )
            print(f" {tag} {frame_tag} profile plot -> {profile_path}")



def run_with_cfg(cfg, args):
    out_root = Path(cfg["output_dir"])
    raw_dir = ensure_dir(out_root / "raw")
    fig_dir = ensure_dir(out_root / "figures")

    if args.uhu_path is not None:
        process_one(args.uhu_path, raw_dir, fig_dir, cfg)
    elif args.all or cfg.get("input_dir"):
        input_dir = Path(cfg.get("input_dir", getattr(args, "input_dir", None)))
        uhu_files = sorted(input_dir.glob("*.npz"))
        if not uhu_files:
            raise FileNotFoundError(f"No .npz files found in {input_dir}")
        print(f"Processing {len(uhu_files)} file(s) from {input_dir}")
        for uhu_path in uhu_files:
            process_one(uhu_path, raw_dir, fig_dir, cfg)
    else:
        raise SystemExit("Provide a uhu_path argument or use --all with --input_dir.")
    print("Done.")



def main():
    parser = argparse.ArgumentParser(
        description="Compute diamond phase fields from existing uHu files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("uhu_path", nargs="?", default=None, help="Path to one uHu .npz file.")
    parser.add_argument("--all", action="store_true", help="Process all .npz files in --input_dir.")
    parser.add_argument("--input_dir", type=str, default="results/sh_pgb_diamond/uhu/raw")
    parser.add_argument("--output_dir", type=str, default="results/sh_pgb_diamond/phase")
    parser.add_argument("--mu", type=float, default=None, help="Override mu if not present in metadata.")
    parser.add_argument("--n_phase_seeds", type=int, default=128)
    parser.add_argument("--ds", type=float, default=0.2)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--prefer_sym", action="store_true", default=True)
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--min_mu", type=float, default=None, help="Skip files whose mu is below this threshold.")
    parser.add_argument(
        "--phase_ramp_mode",
        type=str,
        default="saved",
        choices=["none", "rebuild", "saved"],
        help="Ramp policy used during phase extraction.",
    )
    parser.add_argument("--phase_margin", type=float, default=None, help="Margin for rebuilt diamond ramp.")
    parser.add_argument("--phase_tanh_scale", type=float, default=1.0, help="tanh steepness for rebuilt diamond ramp.")
    parser.add_argument("--phase_smooth_sigma", type=float, default=1.0, help="Gaussian smoothing sigma for rebuilt diamond ramp.")
    parser.add_argument("--ramp_sample_thresh", type=float, default=0.0, help="Discard traced samples with ramp below this threshold before gridding.")
    parser.add_argument("--boundary_source", type=str, default="inner", choices=["inner", "outer"], help="Which stored diamond boundary to use for seeding.")
    parser.add_argument("--coord_n_lines", type=int, default=12, help="Number of coordinate lines to overlay in the diagnostic plot.")
    parser.add_argument("--postprocess", action="store_true", help="Apply Gaussian smoothing to phase/amplitude after raw extraction.")
    parser.add_argument("--postprocess_sigma", type=float, default=1.0, help="Gaussian sigma for postprocessing.")
    parser.add_argument("--profile_mask_tol", type=float, default=0.20, help="Mask threshold used for profile diagnostics.")
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            cfg = json.load(f)
    else:
        cfg = {
            "output_dir": args.output_dir,
            "input_dir": args.input_dir,
            "mu": args.mu,
            "n_phase_seeds": args.n_phase_seeds,
            "ds": args.ds,
            "max_steps": args.max_steps,
            "prefer_sym": args.prefer_sym,
            "no_plot": args.no_plot,
            "overwrite": args.overwrite,
            "min_mu": args.min_mu,
            "phase_ramp_mode": args.phase_ramp_mode,
            "phase_margin": args.phase_margin,
            "phase_tanh_scale": args.phase_tanh_scale,
            "phase_smooth_sigma": args.phase_smooth_sigma,
            "ramp_sample_thresh": args.ramp_sample_thresh,
            "boundary_source": args.boundary_source,
            "coord_n_lines": args.coord_n_lines,
            "postprocess": args.postprocess,
            "postprocess_sigma": args.postprocess_sigma,
            "profile_mask_tol": args.profile_mask_tol,
        }
    run_with_cfg(cfg, args)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        class Args:
            uhu_path = None
            all = True
            input_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_diamond/debug_uhu/sig_pio2/raw"
            output_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_diamond/debug_phase/sig_pio2"
            mu = None
            n_phase_seeds = 256
            ds = 0.10
            max_steps = 10000
            prefer_sym = True
            phase_ramp_mode = "saved"
            phase_margin = None
            phase_tanh_scale = 1.0
            phase_smooth_sigma = 1.0
            ramp_sample_thresh = 0.0
            no_plot = False
            overwrite = True
            config = None
            min_mu = None
            boundary_source = "inner"
            coord_n_lines = 12
            postprocess = True
            postprocess_sigma = np.pi/2
            profile_mask_tol = 0.20

        cfg = {
            "output_dir": Args.output_dir,
            "input_dir": Args.input_dir,
            "mu": Args.mu,
            "n_phase_seeds": Args.n_phase_seeds,
            "ds": Args.ds,
            "max_steps": Args.max_steps,
            "prefer_sym": Args.prefer_sym,
            "phase_ramp_mode": Args.phase_ramp_mode,
            "phase_margin": Args.phase_margin,
            "phase_tanh_scale": Args.phase_tanh_scale,
            "phase_smooth_sigma": Args.phase_smooth_sigma,
            "ramp_sample_thresh": Args.ramp_sample_thresh,
            "no_plot": Args.no_plot,
            "overwrite": Args.overwrite,
            "min_mu": Args.min_mu,
            "boundary_source": Args.boundary_source,
            "coord_n_lines": Args.coord_n_lines,
            "postprocess": Args.postprocess,
            "postprocess_sigma": Args.postprocess_sigma,
            "profile_mask_tol": Args.profile_mask_tol,
        }
        run_with_cfg(cfg, Args)
    else:
        main()