import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

from op_extract.phase_zigzag import (
    compute_pgb_phase_from_uhu,
    load_uhu_npz,
    make_phase_summary_plot_four_panels,
    make_coordinate_line_diagnostic,
    postprocess_phase_amplitude,
    make_phase_profile_plot,
    make_amplitude_diagnostic_plot,
)


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_out_stem(
    uhu_path,
    x_gap_frac,
    y_gap_frac,
    n_phase_seeds,
    seed_half,
    ds,
    phase_ramp_mode,
    phase_ramp_c,
    phase_ramp_smooth_sigma,
    ramp_sample_thresh,
):
    uhu_path = Path(uhu_path)
    stem = (
        f"{uhu_path.stem}"
        f"_phase_xg{x_gap_frac:.2f}"
        f"_yg{y_gap_frac:.2f}"
        f"_ns{int(n_phase_seeds)}"
        f"_sh{seed_half}"
        f"_ds{ds:.3f}"
        f"_prm{phase_ramp_mode}"
    )
    if phase_ramp_mode == "rebuild":
        stem += (
            f"_prc{phase_ramp_c:.3f}"
            f"_prs{phase_ramp_smooth_sigma:.2f}"
            f"_prt{ramp_sample_thresh:.3f}"
        )
    elif phase_ramp_mode == "saved":
        stem += f"_prt{ramp_sample_thresh:.3f}"
    return stem


def run_pgb_phase(uhu_path, out_path, cfg):
    uhu_path = Path(uhu_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    uhu = load_uhu_npz(uhu_path)

    print(
        "[pgb_phase_runner] "
        f"seed_half={cfg.get('seed_half', 'upper')} "
        f"prefer_sym={cfg.get('prefer_sym', True)} "
        f"phase_ramp_mode={cfg.get('phase_ramp_mode', 'none')}"
    )
    result = compute_pgb_phase_from_uhu(
        uhu_data=uhu,
        mu=cfg.get("mu"),
        x_gap_frac=cfg.get("x_gap_frac", 0.10),
        y_gap_frac=cfg.get("y_gap_frac", 0.10),
        n_phase_seeds=cfg.get("n_phase_seeds", 256),
        seed_half=cfg.get("seed_half", "upper"),
        ds=cfg.get("ds", 0.15),
        max_steps=cfg.get("max_steps", 10000),
        prefer_sym=cfg.get("prefer_sym", True),
        phase_ramp_mode=cfg.get("phase_ramp_mode", "none"),
        phase_ramp_c=cfg.get("phase_ramp_c", 0.03),
        phase_ramp_smooth_sigma=cfg.get("phase_ramp_smooth_sigma", 1.0),
        ramp_sample_thresh=cfg.get("ramp_sample_thresh", 0.05),
    )

    return uhu, result

def save_pgb_phase_result(result, uhu, out_path):
    phase_meta_json = json.dumps(result["phase_meta"])
    save_dict = {
        "x": result["x"],
        "y": result["y"],
        "u": result["u"],
        "tt": result["tt"] if result.get("tt") is not None else np.array([]),
        "mu": result["mu"],
        "ramp": result.get("ramp"),
        "k": result.get("k"),
        "A": result.get("A"),
        "k1_sym": result.get("k1_sym"),
        "k2_sym": result.get("k2_sym"),
        "k1_orig": result.get("k1_orig"),
        "k2_orig": result.get("k2_orig"),
        "knee_bdry": result["knee_bdry"],
        "knee_bdry_phase": result["knee_bdry_phase"],
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
        "phase_ramp": result.get("phase_ramp"),
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
        "amplitude_cos",
        "amplitude_cos_smooth",
        "phase_grid_symmetric_wrapped_smooth",
        "phase_grid_symmetric_unwrapped_smooth",
        "analytic_amplitude_grid_symmetric_smooth",
        "amplitude_cos_symmetric",
        "amplitude_cos_symmetric_smooth",
    ):
        if key in result:
            save_dict[key] = result[key]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **save_dict)
    print(f" saved phase -> {out_path}")

import re

def infer_mu_from_path_or_data(uhu_path):
    uhu = load_uhu_npz(uhu_path)
    if uhu.get("mu") is not None:
        return float(uhu["mu"])

    m = re.search(r"mu([0-9]*\.?[0-9]+)", Path(uhu_path).stem)
    if m:
        return float(m.group(1))

    return None


def make_knee_geometry_plot(result, fig_path):
    x = result["x"]
    y = result["y"]
    u = result["u"]
    knee_bdry = result["knee_bdry"]
    phase_ramp = result.get("phase_ramp", None)
    extent = [x[0], x[-1], y[0], y[-1]]

    if u.ndim == 3:
        fi = u.shape[-1] - 1
        u_plot = u[:, :, fi]
    else:
        fi = 0
        u_plot = u

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
    axs[0].plot(knee_bdry[0], knee_bdry[1], "k.", ms=2.5, alpha=0.9)
    axs[0].set_title("pattern with knee boundary")
    fig.colorbar(im0, ax=axs[0], shrink=0.85)

    im1 = axs[1].imshow(
        u_ramped,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-vmax1,
        vmax=vmax1,
    )
    axs[1].plot(knee_bdry[0], knee_bdry[1], "k.", ms=2.5, alpha=0.9)
    axs[1].set_title("pattern × phase ramp")
    fig.colorbar(im1, ax=axs[1], shrink=0.85)

    for ax in axs:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    meta = result.get("phase_meta", {})
    mu = result.get("mu", None)
    mu_str = f"mu={mu:.3f}" if mu is not None else "mu=?"
    ramp_mode = meta.get("phase_ramp_mode", "none")
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
        cfg.get("x_gap_frac", 0.10),
        cfg.get("y_gap_frac", 0.10),
        cfg.get("n_phase_seeds", 256),
        cfg.get("seed_half", "upper"),
        cfg.get("ds", 0.15),
        cfg.get("phase_ramp_mode", "none"),
        cfg.get("phase_ramp_c", 0.03),
        cfg.get("phase_ramp_smooth_sigma", 1.0),
        cfg.get("ramp_sample_thresh", 0.05),
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

    fig_path_amp_raw_initial = fig_dir / f"{stem}_amp_raw_initial.png"
    fig_path_amp_raw_final = fig_dir / f"{stem}_amp_raw_final.png"
    fig_path_amp_smooth_initial = fig_dir / f"{stem}_amp_smooth_initial.png"
    fig_path_amp_smooth_final = fig_dir / f"{stem}_amp_smooth_final.png"

    if out_path.exists() and not cfg.get("overwrite", False):
        print(f" skip (exists): {out_path.name}")
        return

    uhu, result = run_pgb_phase(uhu_path, out_path, cfg)

    if cfg.get("postprocess", False):
        post = postprocess_phase_amplitude(
            result,
            prefer_sym=cfg.get("prefer_sym", True),
            sigma=None,
            sigma_prefactor=cfg.get("post_sigma_prefactor", 2.0),
            mask_tol=cfg.get("profile_mask_tol", 0.99),
        )
        result.update(post)

    save_pgb_phase_result(result, uhu, out_path)

    if not cfg.get("no_plot", False):
        make_phase_summary_plot_four_panels(
            result,
            fig_path_initial,
            prefer_sym=cfg.get("prefer_sym", True),
            frame_index=0,
            mask_with_ramp=True,
        )
        print(f" initial phase plot -> {fig_path_initial}")

        make_phase_summary_plot_four_panels(
            result,
            fig_path_final,
            prefer_sym=cfg.get("prefer_sym", True),
            frame_index=-1,
            mask_with_ramp=True,
        )
        print(f" final phase plot -> {fig_path_final}")

        make_coordinate_line_diagnostic(
            result,
            fig_path_coord,
            frame_index=-1,
            n_lines_to_show=cfg.get("coord_n_lines", 12),
            points_per_line=cfg.get("coord_points_per_line", 12),
        )
        print(f" coordinate-line plot -> {fig_path_coord}")

        make_knee_geometry_plot(result, fig_path_geom)
        print(f" geometry plot -> {fig_path_geom}")

        profile_jobs = [
            ("raw", False, "initial", 0, fig_path_profiles_raw_initial, fig_path_amp_raw_initial),
            ("raw", False, "final", -1, fig_path_profiles_raw_final, fig_path_amp_raw_final),
        ]

        if cfg.get("postprocess", False):
            profile_jobs += [
                ("smoothed", True, "initial", 0, fig_path_profiles_smooth_initial, fig_path_amp_smooth_initial),
                ("smoothed", True, "final", -1, fig_path_profiles_smooth_final, fig_path_amp_smooth_final),
            ]

        for tag, use_smoothed, frame_tag, frame_index, profile_path, amp_path in profile_jobs:
            make_phase_profile_plot(
                result,
                profile_path,
                prefer_sym=cfg.get("prefer_sym", True),
                frame_index=frame_index,
                use_smoothed=use_smoothed,
                mask_tol=cfg.get("profile_mask_tol", 0.99),
            )
            print(f" {tag} {frame_tag} profile plot -> {profile_path}")

            make_amplitude_diagnostic_plot(
                result,
                amp_path,
                prefer_sym=cfg.get("prefer_sym", True),
                frame_index=frame_index,
                use_smoothed=use_smoothed,
                mask_with_ramp=True,
                mask_tol=cfg.get("profile_mask_tol", 0.99),
            )
            print(f" {tag} {frame_tag} amplitude plot -> {amp_path}")


def run_with_cfg(cfg, args):
    out_root = Path(cfg["output_dir"])
    raw_dir = ensure_dir(out_root / "raw")
    fig_dir = ensure_dir(out_root / "figures")

    if args.uhu_path is not None:
        process_one(args.uhu_path, raw_dir, fig_dir, cfg)
    elif args.all or cfg.get("input_dir"):
        input_dir = Path(cfg.get("input_dir", args.input_dir))
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
        description="Compute knee-geometry phase fields from existing PGB uHu files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("uhu_path", nargs="?", default=None, help="Path to one uHu .npz file.")
    parser.add_argument("--all", action="store_true", help="Process all .npz files in --input_dir.")
    parser.add_argument("--input_dir", type=str, default="results/sh_pgb_zigzag/ops/raw")
    parser.add_argument("--output_dir", type=str, default="results/sh_pgb_zigzag/phase")
    parser.add_argument("--mu", type=float, default=None, help="Override mu if not present in metadata.")
    parser.add_argument("--x_gap_frac", type=float, default=0.10)
    parser.add_argument("--y_gap_frac", type=float, default=0.10)
    parser.add_argument("--n_phase_seeds", type=int, default=256)
    parser.add_argument("--ds", type=float, default=0.15)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--prefer_sym", action="store_true", default=True)
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--min_mu", type=float, default=None,
                        help="Skip files whose mu is below this threshold.")
    parser.add_argument(
        "--phase_ramp_mode",
        type=str,
        default="rebuild",
        choices=["none", "rebuild", "saved"],
        help="Ramp policy used during phase extraction.",
    )
    parser.add_argument(
        "--phase_ramp_c",
        type=float,
        default=0.03,
        help="tanh steepness for rebuilt knee ramp.",
    )
    parser.add_argument(
        "--phase_ramp_smooth_sigma",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma for rebuilt knee ramp.",
    )
    parser.add_argument(
        "--ramp_sample_thresh",
        type=float,
        default=0.05,
        help="Discard traced samples with ramp below this threshold before gridding.",
    )
    parser.add_argument(
        "--seed_half",
        type=str,
        default="upper",
        choices=["upper", "lower"],
        help="Which half of the knee boundary to use for phase seeding.",
    )
    parser.add_argument(
        "--coord_n_lines",
        type=int,
        default=12,
        help="Number of coordinate lines to overlay in the diagnostic plot.",
    )
    parser.add_argument(
        "--coord_points_per_line",
        type=int,
        default=12,
        help="Number of marked sample points per shown coordinate line.",
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Apply Gaussian smoothing to phase/amplitude after raw extraction.",
    )
    parser.add_argument(
        "--post_sigma_prefactor",
        type=float,
        default=2.0,
        help="Postprocessing Gaussian sigma = prefactor * sqrt(1 - mu^2).",
    )
    parser.add_argument(
        "--profile_mask_tol",
        type=float,
        default=0.99,
        help="Mask threshold used for profile/amplitude diagnostics.",
    )
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            cfg = json.load(f)
    else:
        cfg = {
            "output_dir": args.output_dir,
            "input_dir": args.input_dir,
            "mu": args.mu,
            "x_gap_frac": args.x_gap_frac,
            "y_gap_frac": args.y_gap_frac,
            "n_phase_seeds": args.n_phase_seeds,
            "ds": args.ds,
            "max_steps": args.max_steps,
            "prefer_sym": args.prefer_sym,
            "no_plot": args.no_plot,
            "overwrite": args.overwrite,
            "min_mu": args.min_mu,
            "phase_ramp_mode": args.phase_ramp_mode,
            "phase_ramp_c": args.phase_ramp_c,
            "phase_ramp_smooth_sigma": args.phase_ramp_smooth_sigma,
            "ramp_sample_thresh": args.ramp_sample_thresh,
            "seed_half": args.seed_half,
            "coord_n_lines": args.coord_n_lines,
            "coord_points_per_line": args.coord_points_per_line,
            "postprocess": args.postprocess,
            "post_sigma_prefactor": args.post_sigma_prefactor,
            "profile_mask_tol": args.profile_mask_tol,
        }
    run_with_cfg(cfg, args)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        class Args:
            uhu_path = None
            all = True
            # input_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/uhu_full_run_2_sig1/raw"
            # output_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/phase_full_run_2_sig1/"
            # input_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweep_uhu_5_sig_pio2/raw"
            # output_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweep_phase_v2_5_sig_pio2_2/"
            # input_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweep_uhu_3_sig_pio2/raw"
            # output_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweep_phase_3_sig_pio2/"
            input_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/uhu/0711_np16_nx1024/sig_pio4/raw"
            output_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/phase/0711_np16_nx1024/sig_pio4"
            mu = None
            x_gap_frac = 0.10
            y_gap_frac = 0.10
            n_phase_seeds = 64
            ds = 0.25
            max_steps = 10000
            prefer_sym = True
            phase_ramp_mode = "rebuild"
            phase_ramp_c = 0.10
            phase_ramp_smooth_sigma = 1.0
            ramp_sample_thresh = 0.05
            no_plot = False
            overwrite = True
            config = None
            min_mu = 0.70
            seed_half = "lower"
            coord_n_lines = 12
            coord_points_per_line = 12
            postprocess = True
            post_sigma_prefactor = 2.5
            profile_mask_tol = 0.99

        cfg = {
            "output_dir": Args.output_dir,
            "input_dir": Args.input_dir,
            "mu": Args.mu,
            "x_gap_frac": Args.x_gap_frac,
            "y_gap_frac": Args.y_gap_frac,
            "n_phase_seeds": Args.n_phase_seeds,
            "ds": Args.ds,
            "max_steps": Args.max_steps,
            "prefer_sym": Args.prefer_sym,
            "phase_ramp_mode": Args.phase_ramp_mode,
            "phase_ramp_c": Args.phase_ramp_c,
            "phase_ramp_smooth_sigma": Args.phase_ramp_smooth_sigma,
            "ramp_sample_thresh": Args.ramp_sample_thresh,
            "no_plot": Args.no_plot,
            "overwrite": Args.overwrite,
            "min_mu": Args.min_mu,
            "seed_half": Args.seed_half,
            "coord_n_lines": Args.coord_n_lines,
            "coord_points_per_line": Args.coord_points_per_line,
            "postprocess": Args.postprocess,
            "post_sigma_prefactor": Args.post_sigma_prefactor,
            "profile_mask_tol": Args.profile_mask_tol,
        }
        run_with_cfg(cfg, Args)
    else:
        main()