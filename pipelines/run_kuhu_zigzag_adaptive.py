# pipelines/run_kuhu_zigzag_adaptive.py
"""
Order-parameter extraction for PGB zigzag SH data with the ADAPTIVE
anisotropic uHu filter (op_extract.uhu_adaptive).

Same contract as run_kuhu_zigzag.py: reads standardised SH .npz files,
writes standardised OP .npz files with keys x, y, u, tt, ramp, k, A, k1,
k2, lam1, lam2, uhu_meta_json, sh_meta_json, so every downstream consumer
(J diagnostics, phase pipeline, midline batteries) works unchanged. Extra
saved keys: coh (final coherence), n_iter_run, iter_delta.

Per-frame warm start: frame t is seeded with the converged fields of frame
t-1 (skipping the isotropic seed pass), which is faster and keeps the
kernel field temporally coherent; disable with warm_start = False.

A midline diagnostic figure integrates the extracted k1 along the core row
to give theta_core(x) for the first and last frames; for a knee this
phase-from-k profile should be nearly linear if the adaptive filter has
removed the core mis-extraction (note this is a different construction
from the Hilbert phase pipeline, see the figure caption).

Usage
-----
python run_kuhu_zigzag_adaptive.py path/to/sh.npz
python run_kuhu_zigzag_adaptive.py --all --input_dir .../raw
python run_kuhu_zigzag_adaptive.py --config my_adaptive.json
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

from op_extract.uhu_adaptive import (
    compute_uhu_ops_adaptive,
    derived_kernel_params_mu,
)
from utils.geometry import build_rectangular_ramp_smooth
from utils.spectral import SpectralDerivs

# legacy plot parity: reuse the original runner's figures directly
sys.path.insert(0, str(_HERE))
from run_kuhu_zigzag import (
    make_summary_plot as make_legacy_summary_plot,
    make_ramp_profile_plot,
)

try:
    from utils.anim import (
        make_director_quiver_panels,
        make_wavevector_quiver_panels,
    )
    _HAVE_ANIM = True
except ImportError:
    _HAVE_ANIM = False


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


def _fmt(v):
    return "auto" if v == "auto" else f"{float(v):.1f}"


def build_out_stem(sh_path, cfg):
    sh_path = Path(sh_path)
    return (
        f"{sh_path.stem}"
        f"_uhuA_sigma{cfg['sigma']:.3f}"
        f"_it{cfg['n_iter']}_a{_fmt(cfg['aspect_max'])}"
        f"_s{_fmt(cfg['sharpen'])}_d{cfg['damping']:.2f}"
        f"_xm{cfg.get('xmargin', 0.10):.2f}"
        f"_ym{cfg.get('ymargin', 0.10):.2f}"
        f"_ts{cfg.get('tanhscale', 7.5):.1f}"
    )


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def run_uhu_adaptive(sh_path, out_path, cfg):
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

    # "auto": derive aspect_max and sharpen from the far-field angle via
    # the leakage calculation (see uhu_adaptive.derived_kernel_params)
    cfg = dict(cfg)
    if cfg.get("aspect_max") == "auto":
        if sh["mu"] is None:
            raise ValueError(f"{sh_path.name}: aspect_max='auto' needs mu "
                             "in the SH metadata")
        a, s = derived_kernel_params_mu(sh["mu"], cfg["sigma"],
                                        delta=cfg.get("leak_delta", 0.05))
        cfg["aspect_max"], cfg["sharpen"] = a, s
        print(f"  auto kernel params from mu={sh['mu']}: "
              f"aspect_max={a:.2f}, sharpen={s:.2f}")

    fields = {name: np.empty((Ny, Nx, Nt)) for name in
              ("k", "A", "k1", "k2", "lam1", "lam2", "coh")}
    n_iter_run = np.zeros(Nt, int)
    iter_delta = []

    seed = None
    for t in range(Nt):
        print(f"  adaptive uHu frame {t+1}/{Nt}  ({sh_path.name})")
        ops = compute_uhu_ops_adaptive(
            u[:, :, t], x, y, cfg["sigma"], SpectralDerivs, ramp,
            n_iter=cfg["n_iter"],
            aspect_max=cfg["aspect_max"],
            sharpen=cfg["sharpen"],
            damping=cfg["damping"],
            patch=cfg["patch"],
            orient=cfg.get("orient", "slowness"),
            seed=seed,
            tol=cfg.get("tol"),
        )
        fields["k"][:, :, t] = ops["k"]
        fields["A"][:, :, t] = ops["A"]
        fields["k1"][:, :, t] = ops["k1_orig"]
        fields["k2"][:, :, t] = ops["k2_orig"]
        fields["lam1"][:, :, t] = ops["lam1"]
        fields["lam2"][:, :, t] = ops["lam2"]
        fields["coh"][:, :, t] = ops["coh"]
        n_iter_run[t] = ops["n_iter_run"]
        iter_delta.append(ops["iter_delta"].tolist())
        if cfg.get("warm_start", True):
            seed = {"k1_orig": ops["k1_orig"], "k2_orig": ops["k2_orig"],
                    "lam1": ops["lam1"], "lam2": ops["lam2"]}

    uhu_meta = {
        "method": "uhu_adaptive",
        "sigma": cfg["sigma"],
        "n_iter": cfg["n_iter"],
        "aspect_max": cfg["aspect_max"],
        "sharpen": cfg["sharpen"],
        "damping": cfg["damping"],
        "patch": cfg["patch"],
        "orient": cfg.get("orient", "slowness"),
        "warm_start": bool(cfg.get("warm_start", True)),
        "tol": cfg.get("tol"),
        "xmargin": cfg.get("xmargin", 0.10),
        "ymargin": cfg.get("ymargin", 0.10),
        "tanhscale": cfg.get("tanhscale", 7.5),
        "source_file": str(sh_path),
        "n_iter_run": n_iter_run.tolist(),
        "iter_delta": iter_delta,
    }
    if sh["mu"] is not None:
        uhu_meta["mu"] = sh["mu"]

    save_dict = {
        "x": x, "y": y, "u": u,
        "tt": sh["tt"] if sh["tt"] is not None else np.array([]),
        "ramp": ramp,
        "uhu_meta_json": json.dumps(uhu_meta),
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

def find_core_row(k_frame, ramp, y, sh_meta_json=None, core_row=None):
    """
    Core row index: explicit cfg value, then gb_row_in_crop metadata, then
    the lower-half row minimizing the x-averaged |k| profile (the core has
    |k| = k1 < 1 for a knee).
    """
    if core_row is not None:
        return int(core_row)
    if sh_meta_json:
        try:
            gb = json.loads(sh_meta_json).get("gb_row_in_crop")
            if gb is not None:
                return int(gb)
        except Exception:
            pass
    mask = ramp >= 0.99999
    prof = np.where(mask, k_frame, np.nan)
    prof = np.nanmean(prof, axis=1)
    lower = slice(0, len(y) // 2)
    idx = np.nanargmin(prof[lower])
    return int(idx)


def masked_for_plot(field, ramp, tol=1e-12):
    return np.ma.masked_where(ramp < (1.0 - tol), field)


def make_summary_plot(fields, ramp, sh, fig_path, cfg, core_row):
    x, y = sh["x"], sh["y"]
    k = fields["k"]
    coh = fields["coh"]
    Nt = k.shape[-1]
    extent = [x[0], x[-1], y[0], y[-1]]

    fig, axs = plt.subplots(2, 3, figsize=(12.6, 6.0), constrained_layout=True)
    panels = [
        (axs[0, 0], masked_for_plot(k[:, :, 0], ramp), "$|k|$, first frame"),
        (axs[0, 1], masked_for_plot(k[:, :, -1], ramp), "$|k|$, final frame"),
        (axs[0, 2], masked_for_plot(coh[:, :, -1], ramp),
         "coherence, final frame"),
        (axs[1, 0], masked_for_plot(fields["k2"][:, :, 0], ramp),
         "$k_2$, first frame"),
        (axs[1, 1], masked_for_plot(fields["k2"][:, :, -1], ramp),
         "$k_2$, final frame"),
    ]
    for ax, f, title in panels:
        im = ax.imshow(f, extent=extent, origin="lower", aspect="auto",
                       cmap="viridis", rasterized=True)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.axhline(y[core_row], color="r", lw=0.6, ls=":")

    # midline phase-from-k check: theta(x) = integral of k1 along the core
    ax = axs[1, 2]
    mask_row = ramp[core_row, :] >= 0.99999
    xs = x[mask_row]
    for t, lab, style in ((0, "first frame", "b-"), (Nt - 1, "final frame", "r-")):
        k1_row = np.abs(fields["k1"][core_row, :, t])[mask_row]
        theta = np.concatenate([[0.0], np.cumsum(
            0.5 * (k1_row[1:] + k1_row[:-1]) * np.diff(xs))])
        theta_lin = np.polyval(np.polyfit(xs, theta, 1), xs)
        ax.plot(xs, theta - theta_lin, style, lw=1.0, label=lab)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.legend(fontsize=8)
    ax.set_xlabel("$x$")
    ax.set_ylabel("residual")
    ax.set_title("core phase-from-k, residual from linear fit", fontsize=9)

    fig.suptitle(
        "Adaptive uHu extraction summary; the bottom-right panel shows the "
        "deviation of the along-core integrated phase "
        "$\\theta(x)=\\int |k_1|\\,dx$ from its best linear fit, which should "
        "be small for a knee if the core wavevector is correctly extracted.",
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

    fields, ramp, sh = run_uhu_adaptive(sh_path, out_path, cfg)

    core_row = find_core_row(fields["k"][:, :, 0], ramp, sh["y"],
                             sh_meta_json=sh["metadata_json"],
                             core_row=cfg.get("core_row"))

    if not cfg.get("no_plot", False):
        make_summary_plot(fields, ramp, sh, fig_path, cfg, core_row)
        if _HAVE_ANIM:
            mask = ramp >= 0.99999
            make_wavevector_quiver_panels(
                k1=fields["k1"], k2=fields["k2"], u=sh["u"],
                x=sh["x"], y=sh["y"],
                fig_path=fig_dir / f"{stem}_wavevec_panels.png",
                n_panels=2, step=12, mask=mask,
                suptitle="adaptive wave-vector snapshots", scale=None)

        make_legacy_summary_plot(
            fields, ramp, sh, fig_dir / f"{stem}_legacy_summary.png", cfg)
        make_ramp_profile_plot(
            ramp, sh["x"], sh["y"],
            fig_dir / f"{stem}_ramp_profile.png", cfg)


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
    sigma = getattr(ns, "sigma", float("nan"))
    if sigma != sigma:
        sigma = np.pi / 2.0
    return {
        "sigma": sigma,
        "n_iter": ns.n_iter,
        "aspect_max": ns.aspect_max,
        "sharpen": ns.sharpen,
        "damping": ns.damping,
        "patch": ns.patch,
        "orient": ns.orient,
        "warm_start": ns.warm_start,
        "tol": ns.tol,
        "core_row": ns.core_row,
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
        description="Adaptive anisotropic uHu extraction for PGB zigzag SH data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("sh_path", nargs="?", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--input_dir", type=str, default="results/sh_pgb_zigzag/raw")
    p.add_argument("--output_dir", type=str, default="results/sh_pgb_zigzag/ops_adaptive")
    p.add_argument("--sigma", type=float, default=float("nan"))
    p.add_argument("--n_iter", type=int, default=3)
    _auto = lambda s: s if s == "auto" else float(s)
    p.add_argument("--aspect_max", type=_auto, default="auto")
    p.add_argument("--sharpen", type=_auto, default="auto")
    p.add_argument("--damping", type=float, default=0.4)
    p.add_argument("--patch", type=int, default=64)
    p.add_argument("--orient", choices=["slowness", "stripe"], default="slowness")
    p.add_argument("--no_warm_start", dest="warm_start", action="store_false")
    p.add_argument("--tol", type=float, default=None)
    p.add_argument("--core_row", type=int, default=None)
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
            output_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/uhu_adaptive/mu_sweeps_full_Nx512_hp025_T3p125_NyF5_np18_Nsave125/sig_pio2/"
            sigma = 1.0
            n_iter = 5
            aspect_max = "auto"
            sharpen = "auto"
            damping = 0.5
            patch = 96
            orient = "slowness"
            warm_start = True
            tol = None
            core_row = None
            xmargin = 0.05
            ymargin = 0.05
            tanhscale = 120.0
            no_plot = False
            overwrite = True
            config = None
        run_with_cfg(_cfg_from(Args), Args)
    else:
        main()
