# experiments/pgb_analysis/k_based_metrics.py
"""
K-based diagnostic plots for PGB zigzag SH order-parameter files.

Default (press Run / F5): processes all OP .npz files found in
    /Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweep_uhu_2/raw

Output figures land in <op_dir>/../figures/k_based_metrics/

Usage
-----
# press Run in IDE — uses the debug block below

# --- directory of OP files ---
python k_based_metrics.py --op_dir /path/to/ops/raw

# --- single file ---
python k_based_metrics.py --op_file /path/to/my_op.npz

# --- rebuild ramp with custom params (override saved ramp) ---
python k_based_metrics.py --op_dir /path/to/ops/raw \\
    --xmargin 0.05 --ymargin 0.05 --tanhscale 60.0

# --- tighten interior threshold ---
python k_based_metrics.py --op_dir /path/to/ops/raw --ramp_thresh 0.999
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- make src/ importable regardless of cwd ---
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]          # project root  (experiments/ → root)
sys.path.insert(0, str(_ROOT / "src"))

from utils.kfield_calcs import orient_vector_field, phi_jump_mask, kfield_diagnostics
from utils.geometry import build_rectangular_ramp_smooth   # same as OP runner


# -----------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------

def load_op_npz(path):
    """Load a standardised OP .npz produced by run_uhu_pgb.py."""
    d = np.load(path, allow_pickle=True)
    x    = d["x"]
    y    = d["y"]
    u    = d["u"][..., -1]   if d["u"].ndim  == 3 else d["u"]
    k1   = d["k1"][..., -1]  if d["k1"].ndim == 3 else d["k1"]
    k2   = d["k2"][..., -1]  if d["k2"].ndim == 3 else d["k2"]
    ramp = d["ramp"]          if "ramp" in d else None
    return x, y, u, k1, k2, ramp


def _get_ramp(x, y, ramp_saved, xmargin, ymargin, tanhscale):
    """
    Return ramp array.
    If xmargin/ymargin/tanhscale are provided (not None), rebuild from scratch.
    Otherwise fall back to the ramp saved in the OP file.
    """
    if xmargin is not None and ymargin is not None and tanhscale is not None:
        print(f"    rebuilding ramp: xmargin={xmargin}, ymargin={ymargin}, tanhscale={tanhscale}")
        return build_rectangular_ramp_smooth(x, y,
                                             xmargin=xmargin,
                                             ymargin=ymargin,
                                             tanhscale=tanhscale)
    if ramp_saved is not None:
        return ramp_saved
    raise ValueError("No ramp in OP file and no ramp params supplied. "
                     "Pass --xmargin/--ymargin/--tanhscale.")


# -----------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------

def _imshow(ax, data, mask, extent, cmap="viridis", **kw):
    im = ax.imshow(np.ma.masked_where(~mask, data),
                   extent=extent, origin="lower", cmap=cmap, **kw)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    return im


def make_geometry_fig(x, y, u, k1_raw, k2_raw, k1_or, k2_or,
                      ramp, mask_vis, mask_pi_raw, mask_pi_or,
                      extent, stem):
    """
    Row 0: u, |k| (raw), |k| (oriented), arg(k) raw, arg(k) oriented, ramp
    Row 1: k1 raw, k2 raw, k1 oriented, k2 oriented, π-jump raw, π-jump oriented
    """
    k_raw   = np.sqrt(k1_raw ** 2 + k2_raw ** 2)
    k_or    = np.sqrt(k1_or ** 2 + k2_or ** 2)
    phi_raw = np.arctan2(k2_raw, k1_raw)
    phi_or  = np.arctan2(k2_or, k1_or)

    # 2 rows × 6 columns
    fig, axs = plt.subplots(2, 6, figsize=(26, 9))
    cbkw = dict(shrink=0.8)

    # Top row: scalar fields (last panel reserved for quiver)
    panels_top = [
        (u, "u (final)", "copper", {}),
        (k_raw, "|k| raw", "viridis", {}),
        (k_or, "|k| oriented", "viridis", {}),
        (phi_raw, "arg(k) raw", "twilight", {}),
        (phi_or, "arg(k) oriented", "twilight", {}),
        (ramp, "ramp", "gray", dict(vmin=0, vmax=1)),
    ]

    # Bottom row: vector components + both π-jump masks
    panels_bot = [
        (k1_raw, "k1 raw", "coolwarm", {}),
        (k2_raw, "k2 raw", "coolwarm", {}),
        (k1_or, "k1 oriented", "coolwarm", {}),
        (k2_or, "k2 oriented", "coolwarm", {}),
        (mask_pi_raw.astype(float), "π-jump raw", "gray_r", dict(vmin=0, vmax=1)),
        (mask_pi_or.astype(float), "π-jump oriented", "gray_r", dict(vmin=0, vmax=1)),
    ]

    # Top row panels (0–4)
    # Top row panels (all 6 columns)
    for col, (arr, title, cmap, kw) in enumerate(panels_top):
        im = _imshow(axs[0, col], arr, mask_vis, extent, cmap=cmap, **kw)
        axs[0, col].set_title(title, fontsize=9)
        fig.colorbar(im, ax=axs[0, col], **cbkw)

    # Bottom row panels (0–5)
    for col, (arr, title, cmap, kw) in enumerate(panels_bot):
        im = _imshow(axs[1, col], arr, mask_vis, extent, cmap=cmap, **kw)
        axs[1, col].set_title(title, fontsize=9)
        fig.colorbar(im, ax=axs[1, col], **cbkw)

    fig.suptitle(stem, fontsize=10)
    plt.tight_layout()
    return fig


def make_diagnostics_fig(raw_d, or_d, mask_ok_raw, mask_ok_or, extent, stem):
    """
    Rows: raw (top) / oriented (bottom)
    Cols: curl k | div k | J | E
    """
    fields = ["curl_k", "div_k", "J", "E"]
    labels = ["curl k", "div k", "J", "E"]
    cmaps = ["coolwarm", "coolwarm", "coolwarm", "hot"]

    fig, axs = plt.subplots(2, 4, figsize=(18, 9))
    cbkw = dict(shrink=0.8)

    rows = [
        (raw_d, "raw", mask_ok_raw),
        (or_d, "oriented", mask_ok_or),
    ]

    for col, (key, label, cmap) in enumerate(zip(fields, labels, cmaps)):
        for row_idx, (diag, tag, mask_row) in enumerate(rows):
            arr = diag[key]
            im = _imshow(axs[row_idx, col], arr, mask_row, extent, cmap=cmap)
            axs[row_idx, col].set_title(f"{label}  ({tag})", fontsize=9)
            fig.colorbar(im, ax=axs[row_idx, col], **cbkw)

    fig.suptitle(stem, fontsize=10)
    plt.tight_layout()
    return fig


def make_quiver_fig(x, y, u,
                    k1_raw, k2_raw,
                    k1_or,  k2_or,
                    mask_vis, extent, stem,
                    step=12):
    """
    Side-by-side wave-vector quivers for raw and oriented k fields.
    Both plotted over u, masked by mask_vis.
    """
    X, Y = np.meshgrid(x, y)
    Xq = X[::step, ::step]
    Yq = Y[::step, ::step]
    mask_q = mask_vis[::step, ::step]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    configs = [
        (k1_raw, k2_raw, "wave-vector quiver (raw)"),
        (k1_or,  k2_or,  "wave-vector quiver (oriented)"),
    ]

    for ax, (k1, k2, title) in zip(axs, configs):
        k1q = np.asarray(k1)[::step, ::step]
        k2q = np.asarray(k2)[::step, ::step]

        ax.imshow(np.ma.masked_where(~mask_vis, u),
                  extent=extent, origin="lower", cmap="copper")
        ax.quiver(
            Xq[mask_q],
            Yq[mask_q],
            k1q[mask_q],
            k2q[mask_q],
            color="white",
            scale=None,
            scale_units="xy",
            angles="xy",
        )
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    fig.suptitle(stem, fontsize=10)
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------
# Per-file processing
# -----------------------------------------------------------------------

def process_one(path, out_dir, ramp_thresh, pi_tol,
                xmargin, ymargin, tanhscale):
    path    = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem    = path.stem
    print(f"  → {stem}")

    x, y, u, k1_raw, k2_raw, ramp_saved = load_op_npz(path)
    ramp = _get_ramp(x, y, ramp_saved, xmargin, ymargin, tanhscale)

    extent   = [x[0], x[-1], y[0], y[-1]]
    mask_vis = ramp >= ramp_thresh          # for plotting / diagnostics
    mask_fd  = ramp >= ramp_thresh          # same here; can be separated

    # orientation
    k1_or, k2_or = orient_vector_field(k1_raw, k2_raw, mask=mask_fd)

    # raw and oriented phases + π-jump masks
    phi_raw = np.arctan2(k2_raw, k1_raw)
    mask_pi_raw = phi_jump_mask(phi_raw, tol=pi_tol)

    phi_or = np.arctan2(np.asarray(k2_or), np.asarray(k1_or))
    mask_pi_or = phi_jump_mask(phi_or, tol=pi_tol)

    # separate masks for raw vs oriented diagnostics
    mask_ok_raw = mask_fd & (~mask_pi_raw)
    mask_ok_or = mask_fd & (~mask_pi_or)

    # diagnostics
    raw_d = kfield_diagnostics(k1_raw, k2_raw, x, y, mask_ok_raw)
    or_d = kfield_diagnostics(k1_or, k2_or, x, y, mask_ok_or)

    fig1 = make_geometry_fig(
        x, y, u,
        k1_raw, k2_raw,
        k1_or, k2_or,
        ramp,
        mask_vis,
        mask_pi_raw,
        mask_pi_or,
        extent,
        stem,
    )
    fig1.savefig(out_dir / f"{stem}_geometry.png", dpi=150)
    plt.close(fig1)

    fig2 = make_diagnostics_fig(raw_d, or_d, mask_ok_raw, mask_ok_or, extent, stem)
    fig2.savefig(out_dir / f"{stem}_diagnostics.png", dpi=150)
    plt.close(fig2)

    fig3 = make_quiver_fig(
        x, y, u,
        k1_raw, k2_raw,
        k1_or, k2_or,
        mask_vis,
        extent,
        stem,
    )
    fig3.savefig(out_dir / f"{stem}_quiver.png", dpi=150)
    plt.close(fig3)

    print(f"    saved: {stem}_geometry.png  {stem}_diagnostics.png  {stem}_quiver.png")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(
        description="K-based diagnostic plots for PGB OP files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--op_file",     type=str, default=None,
                        help="Single OP .npz file.")
    parser.add_argument("--op_dir",      type=str, default=None,
                        help="Directory of OP .npz files.")
    parser.add_argument("--pattern",     type=str, default="*.npz",
                        help="Glob pattern inside op_dir.")
    parser.add_argument("--out_dir",     type=str, default=None,
                        help="Output directory for figures. "
                             "Default: <op_dir>/../figures/k_based_metrics")
    parser.add_argument("--ramp_thresh", type=float, default=0.99,
                        help="Mask threshold: keep where ramp >= this.")
    parser.add_argument("--pi_tol",      type=float, default=np.pi / 10,
                        help="π-jump tolerance.")
    # optional ramp override
    parser.add_argument("--xmargin",   type=float, default=None,
                        help="Rebuild ramp: x-margin fraction.")
    parser.add_argument("--ymargin",   type=float, default=None,
                        help="Rebuild ramp: y-margin fraction.")
    parser.add_argument("--tanhscale", type=float, default=None,
                        help="Rebuild ramp: tanh steepness.")

    ns = parser.parse_args(args)

    # Base results directory under experiments/pgb_analysis
    if ns.out_dir is not None:
        base_out = Path(ns.out_dir)
    else:
        # e.g. experiments/pgb_analysis/results/k_based_metrics/<tag>/
        root_results = _HERE / "results" / "k_based_metrics"
        if ns.op_file is not None:
            tag = Path(ns.op_file).stem
        elif ns.op_dir is not None:
            tag = Path(ns.op_dir).name
        else:
            tag = "default"
        base_out = root_results / tag

    if ns.op_file is not None:
        op_path = Path(ns.op_file)
        out_dir = base_out
        process_one(op_path, out_dir,
                    ns.ramp_thresh, ns.pi_tol,
                    ns.xmargin, ns.ymargin, ns.tanhscale)

    else:
        op_dir = Path(ns.op_dir or ".")
        out_dir = base_out
        files = sorted(op_dir.glob(ns.pattern))
        if not files:
            raise SystemExit(f"No files matching '{ns.pattern}' in {op_dir}")
        print(f"Processing {len(files)} file(s) from {op_dir}")
        for f in files:
            process_one(f, out_dir,
                        ns.ramp_thresh, ns.pi_tol,
                        ns.xmargin, ns.ymargin, ns.tanhscale)

    print("Done.")

    print("Done.")


# -----------------------------------------------------------------------
# Debug / IDE "press Run" block — edit paths here, then Run/Debug
# -----------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) == 1:

        class _Args:
            op_file     = None
            op_dir      = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweep_uhu_2/raw"
            pattern     = "*.npz"
            out_dir     = "/Users/edwardmcdugald/patterns/experiments/pgb_analysis/results/k_based_metrics/mu_sweep_uhu_2_2/"
            ramp_thresh = 1-1e-2
            pi_tol      = np.pi / 10

            # set all three to rebuild ramp; leave as None to use saved ramp
            # xmargin     = None
            # ymargin     = None
            # tanhscale   = None

            # example override to match your debug OP run:
            xmargin   = 0.025
            ymargin   = 0.025
            tanhscale = 120.0

        a = _Args()
        main([
            "--op_dir",      a.op_dir,
            "--out_dir",     a.out_dir,
            "--pattern",     a.pattern,
            "--ramp_thresh", str(a.ramp_thresh),
            "--pi_tol",      str(a.pi_tol),
            *(["--xmargin",   str(a.xmargin),
               "--ymargin",   str(a.ymargin),
               "--tanhscale", str(a.tanhscale)]
              if a.xmargin is not None else []),
        ])
    else:
        main()