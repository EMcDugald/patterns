import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.signal import hilbert


def ensure_u_shape(u, x, y):
    if u.ndim != 3:
        raise ValueError(f"expected u with ndim=3, got shape {u.shape}")
    if u.shape[0] == len(y) and u.shape[1] == len(x):
        return u
    if u.shape[1] == len(y) and u.shape[2] == len(x):
        return np.transpose(u, (1, 2, 0))
    raise ValueError(
        f"cannot reconcile u shape {u.shape} with x ({len(x)},) and y ({len(y)},)"
    )


def load_uhu_npz(uhu_path):
    uhu_path = Path(uhu_path)
    data = np.load(uhu_path, allow_pickle=True)

    x = data["x"]
    y = data["y"]

    if "u" not in data:
        raise ValueError(f"{uhu_path.name}: missing u field")
    u = ensure_u_shape(data["u"], x, y)

    tt = data["tt"] if "tt" in data else None
    ramp = data["ramp"] if "ramp" in data else None

    uhu_meta_json = None
    if "uhu_meta_json" in data:
        raw = data["uhu_meta_json"]
        uhu_meta_json = raw.item() if hasattr(raw, "item") else str(raw)

    sh_meta_json = None
    if "sh_meta_json" in data:
        raw = data["sh_meta_json"]
        sh_meta_json = raw.item() if hasattr(raw, "item") else str(raw)

    mu = None
    for blob in (uhu_meta_json, sh_meta_json):
        if blob:
            try:
                mu = json.loads(blob).get("mu")
            except Exception:
                pass
        if mu is not None:
            break
    if mu is None and "mu" in data:
        mu = float(data["mu"])

    def get_field(*names):
        for name in names:
            if name in data:
                return ensure_u_shape(data[name], x, y)
        return None

    k = get_field("k")
    A = get_field("A")
    k1_sym = get_field("k1_sym")
    k2_sym = get_field("k2_sym")
    k1_orig = get_field("k1_orig", "k1")
    k2_orig = get_field("k2_orig", "k2")

    if k1_sym is None and k1_orig is not None:
        k1_sym = k1_orig.copy()
    if k2_sym is None and k2_orig is not None:
        k2_sym = k2_orig.copy()

    if k1_sym is None or k2_sym is None:
        raise ValueError(f"{uhu_path.name}: could not find usable k1/k2 fields")

    return {
        "x": x,
        "y": y,
        "u": u,
        "tt": tt,
        "ramp": ramp,
        "k": k,
        "A": A,
        "k1_sym": k1_sym,
        "k2_sym": k2_sym,
        "k1_orig": k1_orig,
        "k2_orig": k2_orig,
        "mu": mu,
        "uhu_meta_json": uhu_meta_json,
        "sh_meta_json": sh_meta_json,
        "source_path": str(uhu_path),
    }


def build_knee_boundary_from_mu(x, y, mu, x_gap_frac=0.10, y_gap_frac=0.10, n_samples_per_edge=256):
    x = np.asarray(x)
    y = np.asarray(y)
    x_min, x_max = x[0], x[-1]
    y_min, y_max = y[0], y[-1]
    Lx = x_max - x_min
    Ly = y_max - y_min
    x_shift = 0.5 * (x_min + x_max)
    y_shift = 0.5 * (y_min + y_max)

    if not (0.0 < mu < 1.0):
        raise ValueError("mu must be in (0,1) for trapezoid construction.")

    x_gap = x_gap_frac * (Lx / 2.0)
    y_gap = y_gap_frac * (Ly / 2.0)
    k1 = np.sqrt(1.0 - mu**2)
    k2 = mu
    slope = k1 / k2

    p1x = -Lx / 2.0 + x_gap
    p2x = -Lx / 2.0 + x_gap
    p1y = Ly / 2.0 - y_gap
    p2y = -Ly / 2.0 + y_gap
    p3x = Lx / 2.0 - x_gap
    p4x = Lx / 2.0 - x_gap
    dx = p3x - p1x
    p3y = p1y - slope * dx
    p4y = p2y + slope * dx

    if not (-Ly / 2.0 < p3y < Ly / 2.0) or not (-Ly / 2.0 < p4y < Ly / 2.0):
        raise ValueError(
            "Trapezoid right corners out of centered vertical bounds. "
            "Try reducing x_gap_frac or y_gap_frac."
        )

    t = np.linspace(0.0, 1.0, n_samples_per_edge)
    upper_x = (1.0 - t) * p1x + t * p3x
    upper_y = (1.0 - t) * p1y + t * p3y
    lower_x = (1.0 - t) * p4x + t * p2x
    lower_y = (1.0 - t) * p4y + t * p2y

    knee_x = np.concatenate([upper_x, lower_x]) + x_shift
    knee_y = np.concatenate([upper_y, lower_y]) + y_shift
    return np.vstack([knee_x, knee_y])


def resample_knee_bdry(knee_bdry, n_seeds):
    x_b, y_b = knee_bdry
    pts = np.vstack([x_b, y_b]).T
    diffs = np.diff(pts, axis=0)
    seglen = np.sqrt((diffs**2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    s /= s[-1]
    s_target = np.linspace(0.0, 1.0, n_seeds)
    x_seeds = np.interp(s_target, s, x_b)
    y_seeds = np.interp(s_target, s, y_b)
    return np.vstack([x_seeds, y_seeds])


def _mirror_line_values(arr):
    npts = len(arr)
    mid = npts // 2
    upper = arr[: mid + 1] if npts % 2 else arr[:mid]
    lower = upper[:-1][::-1] if npts % 2 else upper[::-1]
    return np.concatenate([upper, lower])


def compute_phase_from_k_knee(u_ts, k1_ts, k2_ts, x, y, knee_bdry, max_steps=10000, ds=0.15):
    Ny, Nx, nt = u_ts.shape
    y_shift = 0.5 * (y[0] + y[-1])

    x_b, y_b = knee_bdry
    y_b_centered = y_b - y_shift
    mask_up = y_b_centered > 0.0
    start_x = x_b[mask_up]
    start_y = y_b[mask_up]
    nlines = start_x.size
    if nlines == 0:
        raise RuntimeError("No upper-arm knee boundary points found (y>0).")

    all_stitched_lines = []
    all_phases_on_lines_wrapped = []
    all_phases_on_lines_unwrapped = []
    all_grid_phases_wrapped = []
    all_grid_phases_unwrapped = []
    all_phases_on_lines_symmetric_wrapped = []
    all_phases_on_lines_symmetric_unwrapped = []
    all_grid_phases_symmetric_wrapped = []
    all_grid_phases_symmetric_unwrapped = []
    all_analytic_amplitudes_on_lines = []
    all_grid_analytic_amplitudes = []
    all_analytic_amplitudes_on_lines_symmetric = []
    all_grid_analytic_amplitudes_symmetric = []

    Xg, Yg = np.meshgrid(x, y)

    for t in range(nt):
        print(f"> [pgb_phase] Processing timestep {t+1}/{nt}")
        k1 = k1_ts[:, :, t]
        k2 = k2_ts[:, :, t]
        u = u_ts[:, :, t]

        interp_k1 = RegularGridInterpolator((y, x), k1, bounds_error=False, fill_value=0.0)
        interp_k2 = RegularGridInterpolator((y, x), k2, bounds_error=False, fill_value=0.0)
        interp_u = RegularGridInterpolator((y, x), u, bounds_error=False, fill_value=0.0)

        orig_lines = []
        refl_lines = []

        for i in range(nlines):
            x0 = start_x[i]
            y0 = start_y[i]
            line = [(x0, y0)]
            for _ in range(max_steps):
                kx = interp_k1((y0, x0))[()]
                ky = interp_k2((y0, x0))[()]
                mag = np.hypot(kx, ky)
                if mag == 0:
                    break
                y0c = y0 - y_shift
                dir_sign = -1 if y0c > 0 else 1
                dx = dir_sign * kx / mag * ds
                dy = dir_sign * ky / mag * ds
                x0 = x0 + dx
                y0 = y0 + dy
                line.append((x0, y0))
                if (y0 - y_shift) < 0:
                    break
            orig_lines.append(np.array(line))

        for line in orig_lines:
            y_line_centered = line[:, 1] - y_shift
            axis_mask = np.isclose(y_line_centered, 0.0, atol=1e-8)
            non_axis = ~axis_mask
            reflected = np.vstack([line[non_axis, 0], 2 * y_shift - line[non_axis, 1]]).T
            refl_lines.append(reflected[::-1])

        stitched_lines = []
        for orig, refl in zip(orig_lines, refl_lines):
            if refl.shape[0] == 0:
                stitched = orig
            elif np.allclose(orig[-1], refl[0], atol=1e-8):
                stitched = np.vstack([orig, refl[1:]])
            else:
                stitched = np.vstack([orig, refl])
            stitched_lines.append(stitched)
        all_stitched_lines.append(stitched_lines)

        phases_wrapped_t = []
        phases_unwrapped_t = []
        amplitudes_t = []
        sampled_points = []
        phase_samples_wrapped = []
        phase_samples_unwrapped = []
        amplitude_samples = []
        phase_samples_sym_wrapped = []
        phase_samples_sym_unwrapped = []
        amplitude_samples_sym = []
        phases_symmetric_wrapped_t = []
        phases_symmetric_unwrapped_t = []
        amplitudes_symmetric_t = []

        for line in stitched_lines:
            vals = np.array([interp_u((y_, x_))[()] for x_, y_ in line])
            analytic = hilbert(vals)
            phase_wrapped = np.angle(analytic)
            phase_unwrapped = np.unwrap(phase_wrapped)
            amplitude = np.abs(analytic)

            phases_wrapped_t.append(phase_wrapped)
            phases_unwrapped_t.append(phase_unwrapped)
            amplitudes_t.append(amplitude)

            sym_wrapped = _mirror_line_values(phase_wrapped)
            sym_unwrapped = _mirror_line_values(phase_unwrapped)
            sym_amp = _mirror_line_values(amplitude)

            phases_symmetric_wrapped_t.append(sym_wrapped)
            phases_symmetric_unwrapped_t.append(sym_unwrapped)
            amplitudes_symmetric_t.append(sym_amp)

            for (x_, y_), ph_w, ph_u, amp, sph_w, sph_u, samp in zip(
                line, phase_wrapped, phase_unwrapped, amplitude, sym_wrapped, sym_unwrapped, sym_amp
            ):
                sampled_points.append([x_, y_])
                phase_samples_wrapped.append(ph_w)
                phase_samples_unwrapped.append(ph_u)
                amplitude_samples.append(amp)
                phase_samples_sym_wrapped.append(sph_w)
                phase_samples_sym_unwrapped.append(sph_u)
                amplitude_samples_sym.append(samp)

        pts = np.array(sampled_points)
        if len(pts) >= 3:
            grid_phase_wrapped = griddata(pts, np.array(phase_samples_wrapped), (Xg, Yg), method="linear")
            grid_phase_unwrapped = griddata(pts, np.array(phase_samples_unwrapped), (Xg, Yg), method="linear")
            grid_amp = griddata(pts, np.array(amplitude_samples), (Xg, Yg), method="linear")
            grid_phase_sym_wrapped = griddata(pts, np.array(phase_samples_sym_wrapped), (Xg, Yg), method="linear")
            grid_phase_sym_unwrapped = griddata(pts, np.array(phase_samples_sym_unwrapped), (Xg, Yg), method="linear")
            grid_amp_sym = griddata(pts, np.array(amplitude_samples_sym), (Xg, Yg), method="linear")
        else:
            shape = Xg.shape
            grid_phase_wrapped = np.full(shape, np.nan)
            grid_phase_unwrapped = np.full(shape, np.nan)
            grid_amp = np.full(shape, np.nan)
            grid_phase_sym_wrapped = np.full(shape, np.nan)
            grid_phase_sym_unwrapped = np.full(shape, np.nan)
            grid_amp_sym = np.full(shape, np.nan)

        all_phases_on_lines_wrapped.append(phases_wrapped_t)
        all_phases_on_lines_unwrapped.append(phases_unwrapped_t)
        all_grid_phases_wrapped.append(grid_phase_wrapped)
        all_grid_phases_unwrapped.append(grid_phase_unwrapped)
        all_phases_on_lines_symmetric_wrapped.append(phases_symmetric_wrapped_t)
        all_phases_on_lines_symmetric_unwrapped.append(phases_symmetric_unwrapped_t)
        all_grid_phases_symmetric_wrapped.append(grid_phase_sym_wrapped)
        all_grid_phases_symmetric_unwrapped.append(grid_phase_sym_unwrapped)
        all_analytic_amplitudes_on_lines.append(amplitudes_t)
        all_grid_analytic_amplitudes.append(grid_amp)
        all_analytic_amplitudes_on_lines_symmetric.append(amplitudes_symmetric_t)
        all_grid_analytic_amplitudes_symmetric.append(grid_amp_sym)

    return {
        "coordinate_lines": np.array(all_stitched_lines, dtype=object),
        "phase_lines_wrapped": np.array(all_phases_on_lines_wrapped, dtype=object),
        "phase_lines_unwrapped": np.array(all_phases_on_lines_unwrapped, dtype=object),
        "phase_grid_wrapped": np.stack(all_grid_phases_wrapped, axis=-1),
        "phase_grid_unwrapped": np.stack(all_grid_phases_unwrapped, axis=-1),
        "phase_lines_symmetric_wrapped": np.array(all_phases_on_lines_symmetric_wrapped, dtype=object),
        "phase_lines_symmetric_unwrapped": np.array(all_phases_on_lines_symmetric_unwrapped, dtype=object),
        "phase_grid_symmetric_wrapped": np.stack(all_grid_phases_symmetric_wrapped, axis=-1),
        "phase_grid_symmetric_unwrapped": np.stack(all_grid_phases_symmetric_unwrapped, axis=-1),
        "analytic_amplitude_lines": np.array(all_analytic_amplitudes_on_lines, dtype=object),
        "analytic_amplitude_grid": np.stack(all_grid_analytic_amplitudes, axis=-1),
        "analytic_amplitude_lines_symmetric": np.array(all_analytic_amplitudes_on_lines_symmetric, dtype=object),
        "analytic_amplitude_grid_symmetric": np.stack(all_grid_analytic_amplitudes_symmetric, axis=-1),
    }


def compute_pgb_phase_from_uhu(uhu_data, mu=None, x_gap_frac=0.10, y_gap_frac=0.10, n_phase_seeds=256, ds=0.15, max_steps=10000, prefer_sym=True):
    x = uhu_data["x"]
    y = uhu_data["y"]
    u = uhu_data["u"]

    mu_eff = mu if mu is not None else uhu_data.get("mu", None)
    if mu_eff is None:
        raise ValueError("mu was not provided and could not be inferred from the uHu file metadata")

    knee_bdry_raw = build_knee_boundary_from_mu(
        x, y, mu_eff,
        x_gap_frac=x_gap_frac,
        y_gap_frac=y_gap_frac,
        n_samples_per_edge=max(256, n_phase_seeds),
    )

    x_b, y_b = knee_bdry_raw
    y_shift = 0.5 * (y[0] + y[-1])
    mask_up = (y_b - y_shift) > 0.0
    knee_bdry_upper = np.vstack([x_b[mask_up], y_b[mask_up]])
    knee_bdry_phase = resample_knee_bdry(knee_bdry_upper, n_seeds=n_phase_seeds)

    if prefer_sym and uhu_data.get("k1_sym") is not None and uhu_data.get("k2_sym") is not None:
        k1_ts = uhu_data["k1_sym"]
        k2_ts = uhu_data["k2_sym"]
        k_field_source = "sym"
    else:
        k1_ts = uhu_data["k1_orig"]
        k2_ts = uhu_data["k2_orig"]
        k_field_source = "orig"

    phase = compute_phase_from_k_knee(
        u_ts=u,
        k1_ts=k1_ts,
        k2_ts=k2_ts,
        x=x,
        y=y,
        knee_bdry=knee_bdry_phase,
        max_steps=max_steps,
        ds=ds,
    )

    return {
        "x": x,
        "y": y,
        "u": u,
        "tt": uhu_data.get("tt"),
        "mu": mu_eff,
        "ramp": uhu_data.get("ramp"),
        "k": uhu_data.get("k"),
        "A": uhu_data.get("A"),
        "k1_sym": uhu_data.get("k1_sym"),
        "k2_sym": uhu_data.get("k2_sym"),
        "k1_orig": uhu_data.get("k1_orig"),
        "k2_orig": uhu_data.get("k2_orig"),
        "knee_bdry": knee_bdry_raw,
        "knee_bdry_phase": knee_bdry_phase,
        "phase_meta": {
            "mu": float(mu_eff),
            "x_gap_frac": float(x_gap_frac),
            "y_gap_frac": float(y_gap_frac),
            "n_phase_seeds": int(n_phase_seeds),
            "ds": float(ds),
            "max_steps": int(max_steps),
            "k_field_source": k_field_source,
            "source_file": uhu_data.get("source_path"),
        },
        **phase,
    }


def masked_for_plot(field, ramp, tol=1e-12):
    if ramp is None:
        return field
    return np.ma.masked_where(ramp < (1.0 - tol), field)


def make_phase_summary_plot(result, fig_path, n_overlay_lines=24):
    x = result["x"]
    y = result["y"]
    u = result["u"]
    ramp = result.get("ramp")
    k = result.get("k")
    phase_sym = result["phase_grid_symmetric_wrapped"]
    coordinate_lines = result["coordinate_lines"]
    knee_bdry = result["knee_bdry"]
    extent = [x[0], x[-1], y[0], y[-1]]
    fi0 = 0
    fi1 = u.shape[-1] - 1

    fig, axs = plt.subplots(2, 3, figsize=(15, 9))

    im00 = axs[0, 0].imshow(u[:, :, fi0], cmap="bwr", origin="lower", extent=extent)
    axs[0, 0].set_title("pattern (initial)")
    fig.colorbar(im00, ax=axs[0, 0], shrink=0.8)

    phase0 = masked_for_plot(phase_sym[:, :, fi0], ramp)
    im01 = axs[0, 1].imshow(phase0, cmap="twilight", origin="lower", extent=extent)
    axs[0, 1].set_title("symmetric phase (initial)")
    fig.colorbar(im01, ax=axs[0, 1], shrink=0.8)

    im02 = axs[0, 2].imshow(ramp if ramp is not None else np.ones((len(y), len(x))), cmap="gray", origin="lower", extent=extent)
    axs[0, 2].set_title("ramp / support")
    fig.colorbar(im02, ax=axs[0, 2], shrink=0.8)

    im10 = axs[1, 0].imshow(u[:, :, fi1], cmap="bwr", origin="lower", extent=extent)
    axs[1, 0].set_title("pattern (final)")
    fig.colorbar(im10, ax=axs[1, 0], shrink=0.8)

    phase1 = masked_for_plot(phase_sym[:, :, fi1], ramp)
    im11 = axs[1, 1].imshow(phase1, cmap="twilight", origin="lower", extent=extent)
    axs[1, 1].set_title("symmetric phase (final)")
    fig.colorbar(im11, ax=axs[1, 1], shrink=0.8)

    if k is not None:
        im12 = axs[1, 2].imshow(masked_for_plot(k[:, :, fi1], ramp), cmap="viridis", origin="lower", extent=extent)
        axs[1, 2].set_title("k (final)")
        fig.colorbar(im12, ax=axs[1, 2], shrink=0.8)
    else:
        im12 = axs[1, 2].imshow(u[:, :, fi1], cmap="bwr", origin="lower", extent=extent)
        axs[1, 2].set_title("final with traced lines")
        fig.colorbar(im12, ax=axs[1, 2], shrink=0.8)

    for ax in axs.flat:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(knee_bdry[0], knee_bdry[1], color="black", lw=1.0, alpha=0.8)

    lines_final = coordinate_lines[fi1]
    if len(lines_final) > 0:
        stride = max(1, len(lines_final) // max(1, n_overlay_lines))
        for line in lines_final[::stride]:
            line = np.asarray(line)
            axs[1, 2].plot(line[:, 0], line[:, 1], color="white", lw=0.7, alpha=0.75)

    meta = result["phase_meta"]
    fig.suptitle(
        f"mu={meta['mu']:.3f} ds={meta['ds']:.3f} max_steps={meta['max_steps']} source={meta['k_field_source']}",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)