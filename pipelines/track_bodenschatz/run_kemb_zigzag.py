# pipelines/run_kemb_zigzag.py
"""
Order-parameter extraction for PGB zigzag SH data with the
Egolf-Melnikov-Bodenschatz (EMB) derivative-ratio method
(op_extract.kfield_emb), as a comparison and alternative to the uHu
pipeline.

Same contract as run_kuhu_zigzag.py: reads standardised SH .npz files,
writes standardised OP .npz files with keys x, y, u, tt, ramp, k, A, k1,
k2, lam1, lam2, uhu_meta_json, sh_meta_json, so downstream J diagnostics
and the phase pipeline consume it unchanged. Caveats recorded in the
metadata: lam1/lam2 hold the negated eigenvalues of the box-averaged
orientation tensor Q = k k^T (NOT uHu-tensor eigenvalues); chi_Q and
valid_frac are additionally saved.

Usage
-----
python run_kemb_zigzag.py path/to/sh.npz
python run_kemb_zigzag.py --all --input_dir .../raw
(or run in PyCharm; edit Args at the bottom)
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

from op_extract.kfield_emb import compute_kfield_emb
from utils.geometry import build_rectangular_ramp_smooth
from utils.spectral import SpectralDerivs

# legacy plot parity: reuse the original uHu runner's figures directly
sys.path.insert(0, str(_HERE))
from run_kuhu_zigzag import (
    make_summary_plot as make_legacy_summary_plot,
    make_ramp_profile_plot,
)


# ---------------------------------------------------------------------------
# I/O helpers (same conventions as run_kuhu_zigzag.py)
# ---------------------------------------------------------------------------

def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_sh_npz(sh_path):
    sh_path = Path(sh_path)
    data = np.load(sh_path, allow_pickle=True)
    x = data["x"]
    y = data["y"]
    u = data["u"]
    if u.ndim != 3:
        raise ValueError(f"{sh_path.name}: expected u ndim=3, got {u.shape}")
    if u.shape[0] == len(y) and u.shape[1] == len(x):
        pass
    elif u.shape[1] == len(y) and u.shape[2] == len(x):
        u = np.transpose(u, (1, 2, 0))
    else:
        raise ValueError(f"{sh_path.name}: cannot reconcile u shape {u.shape}")
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
    return {"x": x, "y": y, "u": u, "tt": tt, "mu": mu,
            "metadata_json": metadata_json, "source_path": str(sh_path)}


def build_out_stem(sh_path, cfg):
    sh_path = Path(sh_path)
    return (
        f"{sh_path.stem}"
        f"_emb_box{cfg['box_wavelengths']:.2f}"
        f"_n{cfg['n_smooth']}"
        f"_xm{cfg.get('xmargin', 0.10):.2f}"
        f"_ym{cfg.get('ymargin', 0.10):.2f}"
        f"_ts{cfg.get('tanhscale', 7.5):.1f}"
    )


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def run_emb(sh_path, out_path, cfg):
    sh_path = Path(sh_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sh = load_sh_npz(sh_path)
    x, y, u = sh["x"], sh["y"], sh["u"]
    Ny, Nx, Nt = u.shape

    ramp = build_rectangular_ramp_smooth(
        x, y,
        xmargin=cfg.get("xmargin", 0.10),
        ymargin=cfg.get("ymargin", 0.10),
        tanhscale=cfg.get("tanhscale", 7.5),
    )

    fields = {name: np.empty((Ny, Nx, Nt)) for name in
              ("k", "A", "k1", "k2", "lam1", "lam2", "chi_Q", "valid_frac")}

    for t in range(Nt):
        print(f"  EMB frame {t+1}/{Nt}  ({sh_path.name})")
        ops = compute_kfield_emb(
            u[:, :, t], x, y, SpectralDerivs, ramp,
            box_wavelengths=cfg["box_wavelengths"],
            n_smooth=cfg["n_smooth"],
            kmin=cfg["kmin"],
            kmax=cfg["kmax"],
        )
        fields["k"][:, :, t] = ops["k"]
        fields["A"][:, :, t] = ops["A"]
        fields["k1"][:, :, t] = ops["k1_orig"]
        fields["k2"][:, :, t] = ops["k2_orig"]
        fields["lam1"][:, :, t] = ops["lam1"]
        fields["lam2"][:, :, t] = ops["lam2"]
        fields["chi_Q"][:, :, t] = ops["chi_Q"]
        fields["valid_frac"][:, :, t] = ops["valid_frac"]

    meta = {
        "method": "emb_derivative_ratio",
        "box_wavelengths": cfg["box_wavelengths"],
        "n_smooth": cfg["n_smooth"],
        "kmin": cfg["kmin"],
        "kmax": cfg["kmax"],
        "lam_note": "lam1/lam2 are negated Q-tensor eigenvalues, not uHu",
        "xmargin": cfg.get("xmargin", 0.10),
        "ymargin": cfg.get("ymargin", 0.10),
        "tanhscale": cfg.get("tanhscale", 7.5),
        "source_file": str(sh_path),
    }
    if sh["mu"] is not None:
        meta["mu"] = sh["mu"]

    save_dict = {
        "x": x, "y": y, "u": u,
        "tt": sh["tt"] if sh["tt"] is not None else np.array([]),
        "ramp": ramp,
        "uhu_meta_json": json.dumps(meta),
    }
    if sh["metadata_json"] is not None:
        save_dict["sh_meta_json"] = sh["metadata_json"]
    save_dict.update(fields)
    np.savez_compressed(out_path, **save_dict)
    print(f"  saved -> {out_path}")
    return fields, ramp, sh


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def masked_for_plot(field, ramp, tol=1e-12):
    return np.ma.masked_where(ramp < (1.0 - tol), field)


def make_summary_plot(fields, ramp, sh, fig_path, cfg):
    x, y = sh["x"], sh["y"]
    extent = [x[0], x[-1], y[0], y[-1]]
    fig, axs = plt.subplots(2, 3, figsize=(12.6, 6.0), constrained_layout=True)
    panels = [
        (axs[0, 0], fields["k"][:, :, 0], "$|k|$, first frame"),
        (axs[0, 1], fields["k"][:, :, -1], "$|k|$, final frame"),
        (axs[0, 2], fields["chi_Q"][:, :, -1], "$\\chi_Q$, final frame"),
        (axs[1, 0], fields["k2"][:, :, 0], "$k_2$, first frame"),
        (axs[1, 1], fields["k2"][:, :, -1], "$k_2$, final frame"),
        (axs[1, 2], fields["valid_frac"][:, :, -1],
         "valid fraction, final frame"),
    ]
    for ax, f, title in panels:
        im = ax.imshow(masked_for_plot(f, ramp), extent=extent,
                       origin="lower", aspect="auto", cmap="viridis",
                       rasterized=True)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(
        "EMB derivative-ratio extraction summary; chi_Q measures the local "
        "orientation mixture the box average had to discard, and the valid "
        "fraction shows where raw pointwise estimates were excluded.",
        fontsize=9)
    fig.savefig(fig_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def process_one(sh_path, out_dir, fig_dir, cfg):
    sh_path = Path(sh_path)
    stem = build_out_stem(sh_path, cfg)
    out_path = out_dir / f"{stem}.npz"
    fig_path = fig_dir / f"{stem}.png"
    if out_path.exists() and not cfg.get("overwrite", False):
        print(f"  skip (exists): {out_path.name}")
        return
    fields, ramp, sh = run_emb(sh_path, out_path, cfg)
    if not cfg.get("no_plot", False):
        make_summary_plot(fields, ramp, sh, fig_path, cfg)
        legacy_cfg = dict(cfg)
        # legacy summary title hard-reads cfg['sigma']; EMB has no Gaussian
        # width, so print NaN rather than inventing an equivalent
        legacy_cfg.setdefault("sigma", float("nan"))
        make_legacy_summary_plot(
            fields, ramp, sh,
            fig_dir / f"{stem}_legacy_summary.png", legacy_cfg)
        make_ramp_profile_plot(
            ramp, sh["x"], sh["y"],
            fig_dir / f"{stem}_ramp_profile.png", legacy_cfg)


def run_with_cfg(cfg, args):
    out_root = Path(cfg["output_dir"])
    raw_dir = ensure_dir(out_root / "raw")
    fig_dir = ensure_dir(out_root / "figures")
    if getattr(args, "sh_path", None) is not None:
        process_one(args.sh_path, raw_dir, fig_dir, cfg)
    elif getattr(args, "all", False) or cfg.get("input_dir"):
        input_dir = Path(cfg.get("input_dir", getattr(args, "input_dir", "")))
        sh_files = sorted(input_dir.glob("*.npz"))
        if not sh_files:
            raise FileNotFoundError(f"No .npz files found in {input_dir}")
        print(f"Processing {len(sh_files)} file(s) from {input_dir}")
        for sh_path in sh_files:
            process_one(sh_path, raw_dir, fig_dir, cfg)
    else:
        raise SystemExit("Provide a sh_path argument or --all with --input_dir.")
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _cfg_from(ns):
    return {
        "box_wavelengths": ns.box_wavelengths,
        "n_smooth": ns.n_smooth,
        "kmin": ns.kmin,
        "kmax": ns.kmax,
        "xmargin": ns.xmargin,
        "ymargin": ns.ymargin,
        "tanhscale": ns.tanhscale,
        "output_dir": ns.output_dir,
        "input_dir": ns.input_dir,
        "no_plot": ns.no_plot,
        "overwrite": ns.overwrite,
    }


def main():
    p = argparse.ArgumentParser(
        description="EMB derivative-ratio k extraction for PGB zigzag SH data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("sh_path", nargs="?", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--input_dir", type=str, default="results/sh_pgb_zigzag/raw")
    p.add_argument("--output_dir", type=str, default="results/sh_pgb_zigzag/ops_emb")
    p.add_argument("--box_wavelengths", type=float, default=1.0)
    p.add_argument("--n_smooth", type=int, default=2)
    p.add_argument("--kmin", type=float, default=0.2)
    p.add_argument("--kmax", type=float, default=2.0)
    p.add_argument("--xmargin", type=float, default=0.10)
    p.add_argument("--ymargin", type=float, default=0.10)
    p.add_argument("--tanhscale", type=float, default=7.5)
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args()
    if args.config is not None:
        with open(args.config) as f:
            cfg = json.load(f)
    else:
        cfg = _cfg_from(args)
    run_with_cfg(cfg, args)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        class Args:
            sh_path = None
            all = True
            input_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweeps_full_Nx512_hp025_T3p125_NyF5_np18_Nsave125/raw"
            output_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/emb/mu_sweeps_full_Nx512_hp025_T3p125_NyF5_np18_Nsave125/box1_n2/"
            box_wavelengths = 1.0
            n_smooth = 2
            kmin = 0.2
            kmax = 2.0
            xmargin = 0.025
            ymargin = 0.025
            tanhscale = 120.0
            no_plot = False
            overwrite = True
            config = None
        run_with_cfg(_cfg_from(Args), Args)
    else:
        main()
