# pipelines/run_defect_index_gifs.py
"""
Defect-index GIFs, the animated counterpart to t1_common.defect_index_figure.

Three GIFs per data file:
  1. <stem>_defect_overlay.gif   pattern + red/blue defect markers, one sigma_det
  2. <stem>_defect_chargemap.gif the raw charge map alone, no pattern
  3. <stem>_defect_sigma_sweep.gif  multi-panel, several sigma_det side by side

Built on utils.defect_index_anim (which calls op_extract.t1_common.detect_defects,
the same function the static per-frame figures already use) and
run_riesz_zigzag.load_sh_npz for loading and centering the raw SH data.
Nothing about defect detection is reinvented here.

Usage
-----
python run_defect_index_gifs.py

Runs on the six default zigzag paths below. Edit _Args or pass --paths to
change that.
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from run_riesz_zigzag import load_sh_npz
from utils.defect_index_anim import (
    make_defect_overlay_gif,
    make_charge_map_gif,
    make_defect_sigma_sweep_gif,
    make_full_domain_charge_map_gif,
    make_full_domain_charge_sweep_gif,
)


DEFAULT_PATHS = [
    # coarse, dt = 0.2, T = 25
    "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/"
    "mu_sweeps_full_Nx512_hp2_T25_NyF5_np18_Nsave125/raw/"
    "sh_pgb_zigzag_mu0.850_T25_N125_nx512_Ny256_lower.npz",
    "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/"
    "mu_sweeps_full_Nx512_hp2_T25_NyF5_np18_Nsave125/raw/"
    "sh_pgb_zigzag_mu0.900_T25_N125_nx512_Ny256_lower.npz",
    "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/"
    "mu_sweeps_full_Nx512_hp2_T25_NyF5_np18_Nsave125/raw/"
    "sh_pgb_zigzag_mu0.950_T25_N125_nx512_Ny256_lower.npz",
    # fine, dt = 0.025, T = 3.125
    "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/"
    "mu_sweeps_full_Nx512_hp025_T3p125_NyF5_np18_Nsave125/raw/"
    "sh_pgb_zigzag_mu0.850_T3_N125_nx512_Ny256_lower.npz",
    "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/"
    "mu_sweeps_full_Nx512_hp025_T3p125_NyF5_np18_Nsave125/raw/"
    "sh_pgb_zigzag_mu0.900_T3_N125_nx512_Ny256_lower.npz",
    "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/"
    "mu_sweeps_full_Nx512_hp025_T3p125_NyF5_np18_Nsave125/raw/"
    "sh_pgb_zigzag_mu0.950_T3_N125_nx512_Ny256_lower.npz",
]


def process_one(path, cfg):
    path = Path(path)
    print(f"loading {path.name} ...")
    sh = load_sh_npz(path)

    out_dir = path.parent.parent / cfg["out_subdir"]
    stem = path.stem

    make_defect_overlay_gif(
        sh, out_dir, f"{stem}_defect_overlay",
        sigma_det=cfg["sigma_det"], band_y=cfg["band_y"],
        band_x=cfg["band_x"], fps=cfg["fps"],
    )
    make_charge_map_gif(
        sh, out_dir, f"{stem}_defect_chargemap",
        sigma_det=cfg["sigma_det"], band_y=cfg["band_y"],
        band_x=cfg["band_x"], fps=cfg["fps"],
    )
    make_defect_sigma_sweep_gif(
        sh, out_dir, f"{stem}_defect_sigma_sweep",
        sigma_det_list=cfg["sigma_det_sweep"], band_y=cfg["band_y"],
        band_x=cfg["band_x"], fps=cfg["fps"],
    )
    make_full_domain_charge_map_gif(
        sh, out_dir, f"{stem}_defect_fulldomain",
        sigma_det=cfg["sigma_det"], band_y=cfg["band_y"],
        band_x=cfg["band_x"], fps=cfg["fps"],
    )
    make_full_domain_charge_sweep_gif(
        sh, out_dir, f"{stem}_defect_fulldomain_sweep",
        sigma_det_list=cfg["sigma_det_sweep"], band_y=cfg["band_y"],
        band_x=cfg["band_x"], fps=cfg["fps"],
    )
    print(f"  done: {stem}  mu={sh['mu']}")


def run_with_cfg(cfg):
    for path in cfg["paths"]:
        process_one(path, cfg)
    print("all done")


def cfg_from_args(args):
    return {
        "paths":          args.paths if args.paths else DEFAULT_PATHS,
        "sigma_det":      args.sigma_det,
        "sigma_det_sweep": args.sigma_det_sweep,
        "band_y":         args.band_y,
        "band_x":         args.band_x,
        "fps":            args.fps,
        "out_subdir":     args.out_subdir,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Defect-index GIFs: overlay, charge map, sigma sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--paths", type=str, nargs="+", default=None,
                        help="Override the six default npz paths.")
    parser.add_argument("--sigma_det", type=float, default=0.4,
                        help="Detection width for the overlay and charge "
                             "map GIFs. Validated clean up to mu=0.95; "
                             "lower toward 0.2 for sharper folds.")
    parser.add_argument("--sigma_det_sweep", type=float, nargs="+",
                        default=[0.2, 0.3, 0.4, 0.5, 0.8],
                        help="Detection widths shown side by side in the "
                             "sweep GIF.")
    parser.add_argument("--band_y", type=float, default=2.0,
                        help="Half-height of the band around the core "
                             "row that counts toward the dislocation "
                             "count (charges outside it are still drawn).")
    parser.add_argument("--band_x", type=float, default=None,
                        help="Half-width of the band, or omit for the "
                             "full row width.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--out_subdir", type=str, default="animations",
                        help="Subfolder created next to raw/, sibling to "
                             "it, holding the GIFs. Same folder task 1's "
                             "pattern and contour GIFs already use.")
    args = parser.parse_args()
    run_with_cfg(cfg_from_args(args))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        class _Args:
            paths = None          # None -> use DEFAULT_PATHS
            sigma_det = 0.4
            sigma_det_sweep = [0.2, 0.3, 0.4, 0.5, 0.8]
            band_y = 2.0
            band_x = None
            fps = 10
            out_subdir = "animations_updated"
        run_with_cfg(cfg_from_args(_Args))
    else:
        main()