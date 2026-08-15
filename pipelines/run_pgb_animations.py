# pipelines/run_pgb_animations.py
"""
Task 1 from the animations/bugfix handoff: two GIFs per data file.

  1. the raw pattern u over time
  2. u with two contour lines near its own per-frame max and min

Built on src/utils/anim.py (make_scalar_gif, make_minmax_contour_gif).
Nothing about frame construction is reinvented here; this script just loads
each npz, pulls out u and tt, and calls the two anim.py functions.

Usage
-----
python run_pgb_animations.py

Runs on the six default zigzag paths below. Edit _Args or pass --paths to
change that.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

from utils.anim import make_scalar_gif, make_minmax_contour_gif


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
    d = np.load(path, allow_pickle=True)

    u = d["u"]     # (Ny, Nx, Nt)
    tt = d["tt"]   # (Nt,)

    meta = json.loads(str(d["metadata_json"]))
    mu = meta.get("mu")

    out_dir = path.parent.parent / cfg["out_subdir"]
    stem = path.stem  # reuse the existing file naming, no new naming scheme

    if cfg["colorscale"] == "global":
        vmin = float(np.nanmin(u))
        vmax = float(np.nanmax(u))
    else:
        vmin = None
        vmax = None

    make_scalar_gif(
        u, out_dir, f"{stem}_pattern",
        fps=cfg["fps"], cmap="gray", vmin=vmin, vmax=vmax, tt=tt,
    )

    make_minmax_contour_gif(
        u, out_dir, f"{stem}_contours",
        fps=cfg["fps"], f=cfg["contour_f"],
        vmin=vmin, vmax=vmax, tt=tt,
    )

    print(f"  mu = {mu}, done: {stem}")


def run_with_cfg(cfg):
    for path in cfg["paths"]:
        process_one(path, cfg)
    print("all done")


def cfg_from_args(args):
    return {
        "paths":        args.paths if args.paths else DEFAULT_PATHS,
        "colorscale":   args.colorscale,
        "contour_f":    args.contour_f,
        "fps":          args.fps,
        "out_subdir":   args.out_subdir,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Task 1: raw pattern and min/max contour GIFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--paths", type=str, nargs="+", default=None,
                        help="Override the six default npz paths.")
    parser.add_argument("--colorscale", type=str, default="global",
                        choices=["perframe", "global"],
                        help="global: one fixed vmin/vmax computed from the "
                             "full time series, so amplitude drift over "
                             "time is visible rather than normalized away. "
                             "perframe: each frame autoscales its own "
                             "color range instead.")
    parser.add_argument("--contour_f", type=float, default=0.02,
                        help="How far the two contour levels sit from the "
                             "true per-frame max/min, as a fraction of that "
                             "frame's range.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--out_subdir", type=str, default="animations",
                        help="Subfolder name created next to raw/, "
                             "sibling to it, holding the GIFs.")
    args = parser.parse_args()
    run_with_cfg(cfg_from_args(args))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        class _Args:
            paths = None          # None -> use DEFAULT_PATHS
            colorscale = "global"
            contour_f = 0.20
            fps = 10
            out_subdir = f"animations_{str(contour_f)}"
        run_with_cfg(cfg_from_args(_Args))
    else:
        main()
