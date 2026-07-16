import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from utils.geometry import build_diamond_ramp_general


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------

def load_npz(path):
    return np.load(path, allow_pickle=True)


def _parse_mu_from_stem(stem):
    try:
        part = stem.split("mu", 1)[1]
        for sep in ["_", "T", "t"]:
            if sep in part:
                return float(part.split(sep)[0])
        return float(part)
    except Exception:
        return np.nan


def _to_2d(arr, frame_idx=-1):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[..., frame_idx]
    raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")


def _choose_key(d, candidates):
    for k in candidates:
        if k in d:
            return k
    return None


# ---------------------------------------------------------------------
# Field selection
# ---------------------------------------------------------------------

def choose_pattern_field(d, frame_idx=-1):
    key = _choose_key(d, ["u", "uu"])
    if key is None:
        raise KeyError("No pattern field found. Expected one of: 'u', 'uu'.")
    return key, _to_2d(d[key], frame_idx=frame_idx)


def choose_amplitude_field(d, frame_idx=-1):
    key = _choose_key(d, ["A", "amp", "amplitude"])
    if key is None:
        return None, None
    return key, _to_2d(d[key], frame_idx=frame_idx)


def choose_phase_field(d, frame_idx=-1):
    candidates = [
        "phase_grid_symmetric_unwrapped",
        "phase_grid_unwrapped",
        "phase_symmetric_unwrapped",
        "phase_unwrapped",
        "theta_unwrapped",
        "theta",
        "phase_grid_symmetric",
        "phase_grid",
        "phase",
    ]
    key = _choose_key(d, candidates)
    if key is None:
        return None, None
    return key, _to_2d(d[key], frame_idx=frame_idx)


# ---------------------------------------------------------------------
# Ramp / mask logic
# ---------------------------------------------------------------------

def _norm01(arr):
    arr = np.asarray(arr, dtype=float)
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    return (arr - amin) / (amax - amin + 1e-12)


def _looks_like_mask(arr):
    arr = np.asarray(arr)
    if arr.ndim != 2:
        return False
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False
    u = np.unique(np.round(finite, 12))
    return u.size <= 4


def choose_phase_ramp(d, x, y, mu=None,
                      prefer_rebuild=False,
                      ramp_mode="auto",
                      xmargin=None, ymargin=None, tanhscale=None):
    """
    Returns
    -------
    ramp_key : str
    ramp_2d  : ndarray
    source   : str   ('loaded' or 'rebuilt')
    notes    : list[str]
    """
    notes = []

    saved_candidates = [
        "phase_ramp",
        "ramp_inner",
        "ramp",
    ]

    if ramp_mode == "rebuild" or prefer_rebuild:
        if xmargin is None or ymargin is None or tanhscale is None:
            raise ValueError("Rebuild requested but xmargin/ymargin/tanhscale not fully provided.")
        if mu is None or not np.isfinite(mu):
            notes.append("mu unavailable for rebuild; falling back to loaded ramp if possible")
        else:
            try:
                ramp = build_diamond_ramp_general(
                    x, y, mu=mu,
                    xmargin=xmargin,
                    ymargin=ymargin,
                    tanhscale=tanhscale,
                )
                notes.append("rebuilt ramp using build_diamond_ramp_general")
                return "rebuilt_diamond_ramp", np.asarray(ramp), "rebuilt", notes
            except Exception as e:
                notes.append(f"rebuild failed: {e}")

    if ramp_mode in ["auto", "loaded", "rebuild"]:
        for key in saved_candidates:
            if key in d:
                ramp = np.asarray(d[key])
                if ramp.ndim == 3:
                    ramp = ramp[..., -1]
                if ramp.ndim != 2:
                    notes.append(f"skipping {key}: expected 2D or 3D, got shape {ramp.shape}")
                    continue
                if _looks_like_mask(ramp):
                    notes.append(f"using {key} (appears mask-like / binary-ish)")
                else:
                    notes.append(f"using {key}")
                return key, ramp, "loaded", notes

    raise ValueError("No usable ramp found or rebuilt.")


def build_valid_mask(ramp, thresh):
    ramp_n = _norm01(ramp)
    mask = np.isfinite(ramp_n) & (ramp_n >= thresh)
    return ramp_n, mask


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def _imshow(ax, arr, extent, cmap="viridis", title=None, mask=None):
    if mask is not None:
        arr = np.ma.masked_where(~mask, arr)
    im = ax.imshow(arr, extent=extent, origin="lower", cmap=cmap)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return im


def save_probe_figure(out_path, x, y, u, phase, A, ramp_n, mask, info_title):
    extent = [x[0], x[-1], y[0], y[-1]]

    cos_phase = np.cos(phase) if phase is not None else None
    u_r = u * ramp_n
    phase_r = phase * ramp_n if phase is not None else None
    cos_r = cos_phase * ramp_n if cos_phase is not None else None
    A_r = A * ramp_n if A is not None else None

    fig, axs = plt.subplots(3, 3, figsize=(13, 11))
    axs = axs.ravel()

    im = _imshow(axs[0], u, extent, cmap="copper", title="u")
    plt.colorbar(im, ax=axs[0], shrink=0.8)

    im = _imshow(axs[1], u_r, extent, cmap="copper", title="u * ramp_n")
    plt.colorbar(im, ax=axs[1], shrink=0.8)

    im = _imshow(axs[2], mask.astype(float), extent, cmap="gray", title="valid mask")
    plt.colorbar(im, ax=axs[2], shrink=0.8)

    if phase is not None:
        im = _imshow(axs[3], phase, extent, cmap="twilight", title="phase")
        plt.colorbar(im, ax=axs[3], shrink=0.8)

        im = _imshow(axs[4], phase_r, extent, cmap="twilight", title="phase * ramp_n")
        plt.colorbar(im, ax=axs[4], shrink=0.8)

        im = _imshow(axs[5], cos_r if cos_r is not None else np.zeros_like(u),
                     extent, cmap="gray", title="cos(phase) * ramp_n")
        plt.colorbar(im, ax=axs[5], shrink=0.8)
    else:
        for i, ttl in zip([3, 4, 5], ["phase missing", "phase * ramp_n", "cos(phase) * ramp_n"]):
            axs[i].axis("off")
            axs[i].set_title(ttl, fontsize=9)

    if A is not None:
        im = _imshow(axs[6], A, extent, cmap="viridis", title="A")
        plt.colorbar(im, ax=axs[6], shrink=0.8)

        im = _imshow(axs[7], A_r, extent, cmap="viridis", title="A * ramp_n")
        plt.colorbar(im, ax=axs[7], shrink=0.8)
    else:
        axs[6].axis("off")
        axs[6].set_title("A missing", fontsize=9)
        axs[7].axis("off")
        axs[7].set_title("A * ramp_n", fontsize=9)

    im = _imshow(axs[8], ramp_n, extent, cmap="magma", title="normalized ramp")
    plt.colorbar(im, ax=axs[8], shrink=0.8)

    fig.suptitle(info_title, fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def process_one(path, base_out,
                frame_idx,
                ramp_thresh,
                ramp_mode,
                prefer_rebuild,
                xmargin, ymargin, tanhscale):
    path = Path(path)
    stem = path.stem
    print(f"  → {stem}")

    d = load_npz(path)
    keys = sorted(list(d.keys()))

    if "x" not in d or "y" not in d:
        raise KeyError("Expected x and y in file.")
    x = np.asarray(d["x"]).ravel()
    y = np.asarray(d["y"]).ravel()

    mu = None
    if "mu" in d:
        try:
            mu = float(np.asarray(d["mu"]).ravel()[0])
        except Exception:
            mu = _parse_mu_from_stem(stem)
    else:
        mu = _parse_mu_from_stem(stem)

    u_key, u = choose_pattern_field(d, frame_idx=frame_idx)
    A_key, A = choose_amplitude_field(d, frame_idx=frame_idx)
    phase_key, phase = choose_phase_field(d, frame_idx=frame_idx)

    ramp_key, ramp, ramp_source, ramp_notes = choose_phase_ramp(
        d, x, y, mu=mu,
        prefer_rebuild=prefer_rebuild,
        ramp_mode=ramp_mode,
        xmargin=xmargin, ymargin=ymargin, tanhscale=tanhscale,
    )
    ramp_n, valid_mask = build_valid_mask(ramp, ramp_thresh)

    out_dir = base_out / stem
    fig_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    info_title = (
        f"{stem} | mu={mu if np.isfinite(mu) else np.nan:.3f} | "
        f"u:{u_key} | phase:{phase_key} | A:{A_key} | ramp:{ramp_key} ({ramp_source})"
    )

    save_probe_figure(
        fig_dir / "diamond_phase_probe.png",
        x, y, u, phase, A, ramp_n, valid_mask, info_title,
    )

    summary = {
        "file": str(path),
        "stem": stem,
        "mu": None if not np.isfinite(mu) else float(mu),
        "frame_idx": int(frame_idx),
        "available_keys": keys,
        "selected_fields": {
            "u": u_key,
            "phase": phase_key,
            "A": A_key,
            "ramp": ramp_key,
        },
        "ramp_source": ramp_source,
        "ramp_thresh": float(ramp_thresh),
        "valid_fraction": float(np.mean(valid_mask)),
        "shape": {
            "ny": int(u.shape[0]),
            "nx": int(u.shape[1]),
        },
        "notes": ramp_notes,
    }

    with open(data_dir / "probe_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        data_dir / "probe_fields.npz",
        x=x,
        y=y,
        u=u,
        A=A if A is not None else np.full_like(u, np.nan),
        phase=phase if phase is not None else np.full_like(u, np.nan),
        ramp=ramp,
        ramp_n=ramp_n,
        valid_mask=valid_mask.astype(np.uint8),
    )

    print(f"    phase_key={phase_key}  A_key={A_key}  ramp_key={ramp_key}  valid_frac={np.mean(valid_mask):.4f}")
    return summary


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--op_file", type=str, default=None,
                        help="Single diamond OP .npz file.")
    parser.add_argument("--op_dir", type=str, default=None,
                        help="Directory of diamond OP .npz files.")
    parser.add_argument("--pattern", type=str, default="*.npz",
                        help="Glob pattern inside op_dir.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Root output directory. Default: experiments/pgb_phase_networks/results/diamond_phase_probe/")
    parser.add_argument("--frame_idx", type=int, default=-1,
                        help="Frame index for 3D arrays; -1 means final frame.")
    parser.add_argument("--ramp_thresh", type=float, default=0.99,
                        help="Threshold on normalized ramp to define valid mask.")
    parser.add_argument("--ramp_mode", type=str, default="auto",
                        choices=["auto", "loaded", "rebuild"],
                        help="How to choose the phase-valid ramp.")
    parser.add_argument("--prefer_rebuild", action="store_true",
                        help="In auto mode, try rebuilt ramp before loaded ramps.")
    parser.add_argument("--xmargin", type=float, default=None)
    parser.add_argument("--ymargin", type=float, default=None)
    parser.add_argument("--tanhscale", type=float, default=None)
    parser.add_argument("--mu_min", type=float, default=None,
                        help="Only process files with mu > this value.")

    ns = parser.parse_args(args)

    if ns.out_dir is not None:
        base_out = Path(ns.out_dir)
    else:
        base_out = _HERE / "results" / "diamond_phase_probe"
    base_out.mkdir(parents=True, exist_ok=True)

    common = dict(
        frame_idx=ns.frame_idx,
        ramp_thresh=ns.ramp_thresh,
        ramp_mode=ns.ramp_mode,
        prefer_rebuild=ns.prefer_rebuild,
        xmargin=ns.xmargin,
        ymargin=ns.ymargin,
        tanhscale=ns.tanhscale,
    )

    results = []

    if ns.op_file:
        results.append(process_one(Path(ns.op_file), base_out, **common))
    else:
        op_dir = Path(ns.op_dir or ".")
        files = sorted(op_dir.glob(ns.pattern))
        if not files:
            raise SystemExit(f"No files matching '{ns.pattern}' in {op_dir}")
        if ns.mu_min is not None:
            files = [f for f in files if _parse_mu_from_stem(f.stem) > ns.mu_min]
        print(f"Processing {len(files)} file(s) from {op_dir}")
        for f in files:
            results.append(process_one(f, base_out, **common))

    with open(base_out / "run_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Done.")


# ---------------------------------------------------------------------
# Debug block
# ---------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) == 1:

        class _Args:
            op_file = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_diamond/debug_phase/sig_pio2/raw/sh_pgb_diamond_mu0.450_T50_N2_Ny1536_Ly188.496_margin0.45_knee_uhu_sigma1.571_dm0.400_ts0.40_gsnone_phase_diamond_ns192_ds0.125_bdinner_ksym_prmsaved_sif0.200_srm0.990_rst0.000.npz"
            op_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_diamond/raw"
            pattern = "*.npz"
            out_dir = "/Users/edwardmcdugald/patterns/experiments/pgb_phase_networks/results/diamond_phase_probe"
            frame_idx = -1
            ramp_thresh = 0.99
            ramp_mode = "auto"
            prefer_rebuild = False
            xmargin = None
            ymargin = None
            tanhscale = None
            mu_min = None


        a = _Args()
        argv = [
            "--out_dir", a.out_dir,
            "--pattern", a.pattern,
            "--frame_idx", str(a.frame_idx),
            "--ramp_thresh", str(a.ramp_thresh),
            "--ramp_mode", a.ramp_mode,
        ]

        if a.op_file is not None:
            argv += ["--op_file", a.op_file]
        elif a.op_dir is not None:
            argv += ["--op_dir", a.op_dir]

        if a.prefer_rebuild:
            argv.append("--prefer_rebuild")

        if a.xmargin is not None and a.ymargin is not None and a.tanhscale is not None:
            argv += [
                "--xmargin", str(a.xmargin),
                "--ymargin", str(a.ymargin),
                "--tanhscale", str(a.tanhscale),
            ]

        if a.mu_min is not None:
            argv += ["--mu_min", str(a.mu_min)]

        main(argv)
    else:
        main()