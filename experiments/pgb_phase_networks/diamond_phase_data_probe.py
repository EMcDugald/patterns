import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1] if len(_HERE.parents) > 1 else _HERE
sys.path.insert(0, str(_ROOT / "src"))

from utils.geometry import build_diamond_ramp_general
from utils.spectral import macro


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


def _choose_key(d, candidates):
    for k in candidates:
        if k in d:
            return k
    return None


def _to_time_last_stack(arr, name):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"{name}: expected 2D or 3D array, got shape {arr.shape}")


def _broadcast_time_last(arr, T, name):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"{name}: expected 2D or 3D array, got shape {arr.shape}")
    if arr.shape[2] == T:
        return arr
    if arr.shape[2] == 1:
        return np.repeat(arr, T, axis=2)
    raise ValueError(f"{name}: time dimension {arr.shape[2]} incompatible with target T={T}")


def _get_frame_indices(T, frame_idx=None):
    if frame_idx is not None:
        if frame_idx < 0:
            frame_idx = T + frame_idx
        if frame_idx < 0 or frame_idx >= T:
            raise IndexError(f"frame_idx={frame_idx} out of range for T={T}")
        return [frame_idx]
    idx = [0, T // 2, T - 1]
    out = []
    for i in idx:
        if i not in out:
            out.append(i)
    return out


def _frame_label(k, T):
    if k == 0:
        return "first"
    if k == T - 1:
        return "last"
    if k == T // 2:
        return "middle"
    return f"frame{k:04d}"


def choose_pattern_field(d):
    key = _choose_key(d, ["u", "uu"])
    if key is None:
        raise KeyError("No pattern field found. Expected one of: 'u', 'uu'.")
    return key, _to_time_last_stack(d[key], key)


def choose_amplitude_field(d):
    key = _choose_key(d, ["A", "amp", "amplitude"])
    if key is None:
        return None, None
    return key, _to_time_last_stack(d[key], key)


def choose_phase_field(d):
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
    return key, _to_time_last_stack(d[key], key)


def choose_micro_energy_field(d):
    if "e" not in d:
        return None, None
    return "e", _to_time_last_stack(d["e"], "e")


def symmetrize_right_to_left(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[..., None]
    ny, nx, nt = arr.shape
    out = arr.copy()
    left_width = nx // 2
    right = arr[:, nx - left_width:, :]
    out[:, :left_width, :] = np.flip(right, axis=1)
    return out


def build_macro_energy(micro_energy, x, y, sigma):
    micro_energy = np.asarray(micro_energy)
    if micro_energy.ndim == 2:
        micro_energy = micro_energy[..., None]

    ny, nx, nt = micro_energy.shape
    Lx = float(x[-1] - x[0])
    Ly = float(y[-1] - y[0])

    out = np.empty((ny, nx, nt), dtype=float)
    for k in range(nt):
        out[..., k] = np.asarray(macro(micro_energy[..., k], sigma, Lx, Ly))
    return out


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
                      xmargin=None, ymargin=None, tanhscale=None,
                      target_T=None):
    notes = []
    saved_candidates = ["phase_ramp", "ramp_inner", "ramp"]

    if ramp_mode == "rebuild" or prefer_rebuild:
        if xmargin is None or ymargin is None or tanhscale is None:
            raise ValueError("Rebuild requested but xmargin/ymargin/tanhscale not fully provided.")
        if mu is None or not np.isfinite(mu):
            notes.append("mu unavailable for rebuild; falling back to loaded ramp if possible")
        else:
            try:
                ramp2d = build_diamond_ramp_general(
                    x, y, mu=mu,
                    xmargin=xmargin,
                    ymargin=ymargin,
                    tanhscale=tanhscale,
                )
                ramp = np.asarray(ramp2d)[..., None]
                if target_T is not None:
                    ramp = _broadcast_time_last(ramp, target_T, "rebuilt_diamond_ramp")
                notes.append("rebuilt ramp using build_diamond_ramp_general")
                return "rebuilt_diamond_ramp", ramp, "rebuilt", notes
            except Exception as e:
                notes.append(f"rebuild failed: {e}")

    for key in saved_candidates:
        if key in d:
            ramp = _to_time_last_stack(d[key], key)
            if target_T is not None:
                ramp = _broadcast_time_last(ramp, target_T, key)
            if _looks_like_mask(ramp[..., 0]):
                notes.append(f"using {key} (appears mask-like / binary-ish)")
            else:
                notes.append(f"using {key}")
            return key, ramp, "loaded", notes

    raise ValueError("No usable ramp found or rebuilt.")


def build_valid_mask(ramp, thresh):
    ramp = np.asarray(ramp, dtype=float)
    if ramp.ndim == 2:
        ramp = ramp[..., None]
    ny, nx, nt = ramp.shape
    ramp_n = np.empty((ny, nx, nt), dtype=float)
    mask = np.zeros((ny, nx, nt), dtype=bool)
    for k in range(nt):
        rk = _norm01(ramp[..., k])
        ramp_n[..., k] = rk
        mask[..., k] = np.isfinite(rk) & (rk >= thresh)
    return ramp_n, mask


def _imshow(ax, arr, extent, cmap="viridis", title=None, mask=None):
    arr = np.asarray(arr)
    if mask is not None:
        arr = np.ma.masked_where(~mask, arr)
    im = ax.imshow(arr, extent=extent, origin="lower", cmap=cmap)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return im


def save_probe_figure(out_path, x, y, u, phase, phase_sym, A,
                      micro_energy, macro_energy, ramp_n, mask, info_title):
    extent = [x[0], x[-1], y[0], y[-1]]
    fig, axs = plt.subplots(4, 3, figsize=(14, 14))
    axs = axs.ravel()

    im = _imshow(axs[0], u, extent, cmap="copper", title="u")
    plt.colorbar(im, ax=axs[0], shrink=0.8)

    im = _imshow(axs[1], u * ramp_n, extent, cmap="copper", title="u * ramp_n")
    plt.colorbar(im, ax=axs[1], shrink=0.8)

    im = _imshow(axs[2], mask.astype(float), extent, cmap="gray", title="valid mask")
    plt.colorbar(im, ax=axs[2], shrink=0.8)

    if phase is not None:
        im = _imshow(axs[3], phase, extent, cmap="twilight", title="phase")
        plt.colorbar(im, ax=axs[3], shrink=0.8)
        im = _imshow(axs[4], phase_sym, extent, cmap="twilight", title="phase_sym")
        plt.colorbar(im, ax=axs[4], shrink=0.8)
        im = _imshow(axs[5], np.cos(phase_sym) * ramp_n, extent, cmap="gray", title="cos(phase_sym) * ramp_n")
        plt.colorbar(im, ax=axs[5], shrink=0.8)
    else:
        for i, ttl in zip([3, 4, 5], ["phase missing", "phase_sym missing", "cos(phase_sym) * ramp_n"]):
            axs[i].axis("off")
            axs[i].set_title(ttl, fontsize=9)

    if A is not None:
        im = _imshow(axs[6], A, extent, cmap="viridis", title="A")
        plt.colorbar(im, ax=axs[6], shrink=0.8)
    else:
        axs[6].axis("off")
        axs[6].set_title("A missing", fontsize=9)

    im = _imshow(axs[7], micro_energy, extent, cmap="inferno", title="micro_energy",mask=mask)
    plt.colorbar(im, ax=axs[7], shrink=0.8)

    im = _imshow(axs[8], macro_energy, extent, cmap="inferno", title="macro_energy",mask=mask)
    plt.colorbar(im, ax=axs[8], shrink=0.8)

    im = _imshow(axs[9], ramp_n, extent, cmap="magma", title="normalized ramp")
    plt.colorbar(im, ax=axs[9], shrink=0.8)

    im = _imshow(axs[10], micro_energy * mask, extent, cmap="inferno", title="micro_energy * mask",mask=mask)
    plt.colorbar(im, ax=axs[10], shrink=0.8)

    im = _imshow(axs[11], macro_energy * mask, extent, cmap="inferno", title="macro_energy * mask",mask=mask)
    plt.colorbar(im, ax=axs[11], shrink=0.8)

    fig.suptitle(info_title, fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def process_one(path, base_out,
                frame_idx,
                ramp_thresh,
                ramp_mode,
                prefer_rebuild,
                xmargin, ymargin, tanhscale,
                save_sym_phase,
                raw_sh_file=None,
                macro_sigma=None):
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

    u_key, u = choose_pattern_field(d)
    A_key, A = choose_amplitude_field(d)
    phase_key, phase = choose_phase_field(d)
    micro_energy_key, micro_energy = choose_micro_energy_field(d)

    if micro_energy is None and raw_sh_file is not None:
        d_raw = load_npz(raw_sh_file)
        micro_energy_key, micro_energy = choose_micro_energy_field(d_raw)

    T = u.shape[2]
    if A is not None:
        A = _broadcast_time_last(A, T, A_key)
    if phase is not None:
        phase = _broadcast_time_last(phase, T, phase_key)
    if micro_energy is None:
        raise KeyError("No micro energy field found. Expected key: 'e'.")
    micro_energy = _broadcast_time_last(micro_energy, T, micro_energy_key)

    phase_sym = symmetrize_right_to_left(phase) if (phase is not None and save_sym_phase) else phase
    macro_energy = build_macro_energy(micro_energy, x, y, sigma=macro_sigma)

    ramp_key, ramp, ramp_source, ramp_notes = choose_phase_ramp(
        d, x, y, mu=mu,
        prefer_rebuild=prefer_rebuild,
        ramp_mode=ramp_mode,
        xmargin=xmargin, ymargin=ymargin, tanhscale=tanhscale,
        target_T=T,
    )
    ramp_n, valid_mask = build_valid_mask(ramp, ramp_thresh)

    out_dir = base_out / stem
    fig_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    plot_frames = _get_frame_indices(T, frame_idx=frame_idx)
    for k in plot_frames:
        label = _frame_label(k, T)
        info_title = (
            f"{stem} | frame={k}/{T-1} ({label}) | "
            f"mu={mu if np.isfinite(mu) else np.nan:.3f} | "
            f"u:{u_key} | phase:{phase_key} | A:{A_key} | "
            f"micro:{micro_energy_key} | macro:macro(e) | ramp:{ramp_key} ({ramp_source})"
        )
        save_probe_figure(
            fig_dir / f"diamond_phase_probe_{label}.png",
            x, y,
            u[..., k],
            phase[..., k] if phase is not None else None,
            phase_sym[..., k] if phase_sym is not None else None,
            A[..., k] if A is not None else None,
            micro_energy[..., k],
            macro_energy[..., k],
            ramp_n[..., k],
            valid_mask[..., k],
            info_title,
        )

    summary = {
        "file": str(path),
        "stem": stem,
        "mu": None if not np.isfinite(mu) else float(mu),
        "frame_idx_argument": None if frame_idx is None else int(frame_idx),
        "saved_plot_frames": [int(k) for k in plot_frames],
        "available_keys": keys,
        "selected_fields": {
            "u": u_key,
            "phase": phase_key,
            "phase_sym": "right_to_left_reflection" if (phase is not None and save_sym_phase) else None,
            "A": A_key,
            "micro_energy": micro_energy_key,
            "macro_energy": "macro(e)",
            "macro_sigma": None if macro_sigma is None else float(macro_sigma),
            "ramp": ramp_key,
        },
        "ramp_source": ramp_source,
        "ramp_thresh": float(ramp_thresh),
        "valid_fraction_mean": float(np.mean(valid_mask)),
        "shape": {
            "ny": int(u.shape[0]),
            "nx": int(u.shape[1]),
            "nt": int(u.shape[2]),
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
        phase_sym=phase_sym if phase_sym is not None else np.full_like(u, np.nan),
        micro_energy=micro_energy,
        macro_energy=macro_energy,
        ramp=ramp,
        ramp_n=ramp_n,
        valid_mask=valid_mask.astype(np.uint8),
    )

    print(
        f"    phase_key={phase_key}  A_key={A_key}  micro_energy_key={micro_energy_key}  "
        f"ramp_key={ramp_key}  shape={u.shape}  valid_frac_mean={np.mean(valid_mask):.4f}"
    )
    return summary


def main(args=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--op_file", type=str, default=None, help="Single diamond OP .npz file.")
    parser.add_argument("--op_dir", type=str, default=None, help="Directory of diamond OP .npz files.")
    parser.add_argument("--pattern", type=str, default="*.npz", help="Glob pattern inside op_dir.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Root output directory. Default: experiments/pgb_phase_networks/results/diamond_phase_probe/")
    parser.add_argument("--frame_idx", type=int, default=None,
                        help="Optional frame index to plot. Default: save first/middle/last plots while preserving all frames in probe_fields.npz.")
    parser.add_argument("--ramp_thresh", type=float, default=0.99,
                        help="Threshold on normalized ramp to define valid mask.")
    parser.add_argument("--ramp_mode", type=str, default="auto", choices=["auto", "loaded", "rebuild"],
                        help="How to choose the phase-valid ramp.")
    parser.add_argument("--prefer_rebuild", action="store_true",
                        help="In auto mode, try rebuilt ramp before loaded ramps.")
    parser.add_argument("--xmargin", type=float, default=None)
    parser.add_argument("--ymargin", type=float, default=None)
    parser.add_argument("--tanhscale", type=float, default=None)
    parser.add_argument("--mu_min", type=float, default=None, help="Only process files with mu > this value.")
    parser.add_argument("--no_sym_phase", action="store_true",
                        help="Disable right-to-left reflected symmetrized phase saving.")
    parser.add_argument("--raw_sh_file", type=str, default=None,
                        help="Companion raw SH .npz file containing micro energy key 'e'.")
    parser.add_argument(
        "--macro_sigma",
        type=float,
        required=True,
        help="Sigma for macro energy smoothing."
    )

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
        save_sym_phase=not ns.no_sym_phase,
        raw_sh_file=ns.raw_sh_file,
        macro_sigma=ns.macro_sigma,
    )

    results = []

    if ns.op_file is not None:
        results.append(process_one(Path(ns.op_file), base_out, **common))

    elif ns.op_dir is not None:
        op_dir = Path(ns.op_dir)
        files = sorted(op_dir.glob(ns.pattern))
        if not files:
            raise SystemExit(f"No files matching '{ns.pattern}' in {op_dir}")
        if ns.mu_min is not None:
            files = [f for f in files if _parse_mu_from_stem(f.stem) > ns.mu_min]
        print(f"Processing {len(files)} file(s) from {op_dir}")
        for f in files:
            results.append(process_one(f, base_out, **common))

    else:
        raise SystemExit("Provide either --op_file <file.npz> or --op_dir <directory>.")

    with open(base_out / "run_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        class _Args:
            op_file = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_diamond/debug_phase/sig_pio2/raw/sh_pgb_diamond_mu0.450_T50_N2_Ny1536_Ly188.496_margin0.45_knee_uhu_sigma1.571_dm0.400_ts0.40_gsnone_phase_diamond_ns192_ds0.125_bdinner_ksym_prmsaved_sif0.200_srm0.990_rst0.000.npz"
            raw_sh_file = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_diamond/debug/raw/sh_pgb_diamond_mu0.450_T50_N2_Ny1536_Ly188.496_margin0.45_knee.npz"
            op_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_diamond/debug_phase/sig_pio2/raw"
            pattern = "*.npz"
            out_dir = "/Users/edwardmcdugald/patterns/experiments/pgb_phase_networks/results/diamond_phase_probe"
            frame_idx = None
            ramp_thresh = 0.99
            ramp_mode = "auto"
            prefer_rebuild = False
            xmargin = None
            ymargin = None
            tanhscale = None
            mu_min = None
            macro_sigma = np.pi/2

        a = _Args()
        argv = [
            "--out_dir", a.out_dir,
            "--pattern", a.pattern,
            "--ramp_thresh", str(a.ramp_thresh),
            "--ramp_mode", a.ramp_mode,
        ]

        if a.raw_sh_file is not None:
            argv += ["--raw_sh_file", a.raw_sh_file]

        if a.frame_idx is not None:
            argv += ["--frame_idx", str(a.frame_idx)]

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

        argv += ["--macro_sigma", str(a.macro_sigma)]

        main(argv)
    else:
        main()