import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import gaussian_filter
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
    knee_bdry = data["knee_bdry"] if "knee_bdry" in data else None

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
        "knee_bdry": knee_bdry,
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

    if not (0.0 < x_gap_frac < 1.0):
        raise ValueError(f"x_gap_frac must be in (0,1), got {x_gap_frac}")
    if not (0.0 < y_gap_frac < 1.0):
        raise ValueError(f"y_gap_frac must be in (0,1), got {y_gap_frac}")

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


def build_ramp_from_knee_bdry(x, y, knee_bdry, c=0.03, smooth_sigma=1.0):
    x = np.asarray(x)
    y = np.asarray(y)
    X, Y = np.meshgrid(x, y)

    xb = np.asarray(knee_bdry[0])
    yb = np.asarray(knee_bdry[1])

    y_mid_global = 0.5 * (yb.min() + yb.max())

    mask_upper = yb <= y_mid_global
    mask_lower = yb >= y_mid_global

    xb_upper = xb[mask_upper]
    yb_upper = yb[mask_upper]
    xb_lower = xb[mask_lower]
    yb_lower = yb[mask_lower]

    order_u = np.argsort(xb_upper)
    xb_upper = xb_upper[order_u]
    yb_upper = yb_upper[order_u]

    order_l = np.argsort(xb_lower)
    xb_lower = xb_lower[order_l]
    yb_lower = yb_lower[order_l]

    y_top = np.interp(x, xb_upper, yb_upper)
    y_bot = np.interp(x, xb_lower, yb_lower)

    h = 0.5 * (y_top - y_bot)
    y_mid = 0.5 * (y_top + y_bot)

    H = np.tile(h, (y.size, 1))
    Ymid = np.tile(y_mid, (y.size, 1))

    S = H**2 - (Y - Ymid)**2

    x_min_bdry = xb.min()
    x_max_bdry = xb.max()
    outside_x = (X < x_min_bdry) | (X > x_max_bdry)
    S[outside_x] = -1e6

    ramp = 0.5 * (1.0 + np.tanh(c * S))
    ramp = gaussian_filter(ramp, sigma=smooth_sigma)

    rmin = ramp.min()
    rmax = ramp.max()
    ramp = (ramp - rmin) / (rmax - rmin + 1e-12)
    return ramp



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


def compute_phase_from_k_knee(
    u_ts,
    k1_ts,
    k2_ts,
    x,
    y,
    knee_bdry,
    max_steps=10000,
    ds=0.15,
    ramp=None,
    ramp_sample_thresh=0.05,
):
    Ny, Nx, nt = u_ts.shape
    y_shift = 0.5 * (y[0] + y[-1])

    x_b, y_b = knee_bdry
    start_x = np.asarray(x_b)
    start_y = np.asarray(y_b)
    nlines = start_x.size
    if nlines == 0:
        raise RuntimeError("No seed points found in knee_bdry.")

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
        u_for_phase = u if ramp is None else (u * ramp)

        interp_k1 = RegularGridInterpolator((y, x), k1, bounds_error=False, fill_value=0.0)
        interp_k2 = RegularGridInterpolator((y, x), k2, bounds_error=False, fill_value=0.0)
        interp_u = RegularGridInterpolator((y, x), u_for_phase, bounds_error=False, fill_value=0.0)

        interp_ramp = None
        if ramp is not None:
            interp_ramp = RegularGridInterpolator((y, x), ramp, bounds_error=False, fill_value=0.0)

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

                x0_new = x0 + dx
                y0_new = y0 + dy
                line.append((x0_new, y0_new))

                y0c_new = y0_new - y_shift
                if y0c == 0.0 or y0c * y0c_new <= 0.0:
                    x0, y0 = x0_new, y0_new
                    break

                x0, y0 = x0_new, y0_new

            orig_lines.append(np.array(line))

        for line in orig_lines:
            y_line_centered = line[:, 1] - y_shift
            axis_mask = np.isclose(y_line_centered, 0.0, atol=1e-8)
            non_axis = ~axis_mask
            reflected = np.vstack(
                [line[non_axis, 0], 2 * y_shift - line[non_axis, 1]]
            ).T
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

        # ###
        #
        # for line in stitched_lines:
        #     vals = np.array([interp_u((y_, x_))[()] for x_, y_ in line])
        #     analytic = hilbert(vals)
        #     phase_wrapped = np.angle(analytic)
        #     phase_unwrapped = np.unwrap(phase_wrapped)
        #     amplitude = np.abs(analytic)
        #
        #     # Enforce phi = 0 at the knee boundary (first point on the line)
        #     if phase_unwrapped.size > 0:
        #         phase_unwrapped = phase_unwrapped - phase_unwrapped[0]
        #         phase_wrapped = wrap_to_pi(phase_unwrapped)
        #
        #     phases_wrapped_t.append(phase_wrapped)
        #     phases_unwrapped_t.append(phase_unwrapped)
        #     amplitudes_t.append(amplitude)
        #
        #     ###

            sym_wrapped = _mirror_line_values(phase_wrapped)
            sym_unwrapped = _mirror_line_values(phase_unwrapped)
            sym_amp = _mirror_line_values(amplitude)

            phases_symmetric_wrapped_t.append(sym_wrapped)
            phases_symmetric_unwrapped_t.append(sym_unwrapped)
            amplitudes_symmetric_t.append(sym_amp)

            for (x_, y_), ph_w, ph_u, amp, sph_w, sph_u, samp in zip(
                line,
                phase_wrapped,
                phase_unwrapped,
                amplitude,
                sym_wrapped,
                sym_unwrapped,
                sym_amp,
            ):
                if interp_ramp is not None:
                    rv = interp_ramp((y_, x_))[()]
                    if rv < ramp_sample_thresh:
                        continue

                sampled_points.append([x_, y_])
                phase_samples_wrapped.append(ph_w)
                phase_samples_unwrapped.append(ph_u)
                amplitude_samples.append(amp)
                phase_samples_sym_wrapped.append(sph_w)
                phase_samples_sym_unwrapped.append(sph_u)
                amplitude_samples_sym.append(samp)

        pts = np.array(sampled_points)
        if len(pts) >= 3:
            grid_phase_wrapped = griddata(
                pts, np.array(phase_samples_wrapped), (Xg, Yg), method="linear"
            )
            grid_phase_unwrapped = griddata(
                pts, np.array(phase_samples_unwrapped), (Xg, Yg), method="linear"
            )
            grid_amp = griddata(
                pts, np.array(amplitude_samples), (Xg, Yg), method="linear"
            )
            grid_phase_sym_wrapped = griddata(
                pts, np.array(phase_samples_sym_wrapped), (Xg, Yg), method="linear"
            )
            grid_phase_sym_unwrapped = griddata(
                pts, np.array(phase_samples_sym_unwrapped), (Xg, Yg), method="linear"
            )
            grid_amp_sym = griddata(
                pts, np.array(amplitude_samples_sym), (Xg, Yg), method="linear"
            )
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


def compute_pgb_phase_from_uhu(
    uhu_data,
    mu=None,
    x_gap_frac=0.10,
    y_gap_frac=0.10,
    n_phase_seeds=256,
    seed_half="upper",
    ds=0.15,
    max_steps=10000,
    prefer_sym=True,
    phase_ramp_mode="none",
    phase_ramp_c=0.03,
    phase_ramp_smooth_sigma=1.0,
    ramp_sample_thresh=0.05,
):
    x = uhu_data["x"]
    y = uhu_data["y"]
    u = uhu_data["u"]

    mu_eff = mu if mu is not None else uhu_data.get("mu", None)
    if seed_half not in ("upper", "lower"):
        raise ValueError(f"seed_half must be 'upper' or 'lower', got {seed_half}")

    knee_bdry_raw = uhu_data.get("knee_bdry")
    knee_bdry_source = "saved"
    if knee_bdry_raw is None:
        if mu_eff is None:
            raise ValueError("mu was not provided and no saved knee_bdry was found")
        knee_bdry_raw = build_knee_boundary_from_mu(
            x, y, mu_eff,
            x_gap_frac=x_gap_frac,
            y_gap_frac=y_gap_frac,
            n_samples_per_edge=max(256, n_phase_seeds),
        )
        knee_bdry_source = "reconstructed_from_mu"

    knee_bdry_raw = np.asarray(knee_bdry_raw)
    if knee_bdry_raw.shape[0] != 2:
        raise ValueError(f"knee_bdry must have shape (2, N), got {knee_bdry_raw.shape}")

    x_b, y_b = knee_bdry_raw
    y_shift = 0.5 * (y[0] + y[-1])

    if seed_half == "upper":
        mask_seed = (y_b - y_shift) > 0.0
    elif seed_half == "lower":
        mask_seed = (y_b - y_shift) < 0.0
    else:
        raise ValueError(f"seed_half must be 'upper' or 'lower', got {seed_half}")

    if not np.any(mask_seed):
        raise RuntimeError(f"No {seed_half}-arm knee boundary points found for phase seeding.")

    knee_bdry_seed = np.vstack([x_b[mask_seed], y_b[mask_seed]])
    knee_bdry_phase = resample_knee_bdry(knee_bdry_seed, n_seeds=n_phase_seeds)

    if phase_ramp_mode == "saved":
        ramp_phase = uhu_data.get("ramp")
        if ramp_phase is None:
            raise ValueError("phase_ramp_mode='saved' but no ramp was found in the uHu file")
    elif phase_ramp_mode == "rebuild":
        ramp_phase = build_ramp_from_knee_bdry(
            x, y, knee_bdry_raw,
            c=phase_ramp_c,
            smooth_sigma=phase_ramp_smooth_sigma,
        )
    elif phase_ramp_mode == "none":
        ramp_phase = None
    else:
        raise ValueError(f"Unknown phase_ramp_mode={phase_ramp_mode}")

    if ramp_phase is not None:
        ramp_phase = np.asarray(ramp_phase)
        if ramp_phase.shape != (len(y), len(x)):
            raise ValueError(f"phase ramp has shape {ramp_phase.shape}, expected {(len(y), len(x))}")

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
        ramp=ramp_phase,
        ramp_sample_thresh=ramp_sample_thresh,
    )

    return {
        "x": x,
        "y": y,
        "u": u,
        "tt": uhu_data.get("tt"),
        "mu": mu_eff,
        "ramp": uhu_data.get("ramp"),
        "phase_ramp": ramp_phase,
        "k": uhu_data.get("k"),
        "A": uhu_data.get("A"),
        "k1_sym": uhu_data.get("k1_sym"),
        "k2_sym": uhu_data.get("k2_sym"),
        "k1_orig": uhu_data.get("k1_orig"),
        "k2_orig": uhu_data.get("k2_orig"),
        "knee_bdry": knee_bdry_raw,
        "knee_bdry_phase": knee_bdry_phase,
        "phase_meta": {
            "mu": None if mu_eff is None else float(mu_eff),
            "x_gap_frac": float(x_gap_frac),
            "y_gap_frac": float(y_gap_frac),
            "n_phase_seeds": int(n_phase_seeds),
            "seed_half": seed_half,
            "ds": float(ds),
            "max_steps": int(max_steps),
            "k_field_source": k_field_source,
            "knee_bdry_source": knee_bdry_source,
            "phase_ramp_mode": phase_ramp_mode,
            "phase_ramp_c": float(phase_ramp_c),
            "phase_ramp_smooth_sigma": float(phase_ramp_smooth_sigma),
            "ramp_sample_thresh": float(ramp_sample_thresh),
            "source_file": uhu_data.get("source_path"),
        },
        **phase,
    }


def masked_for_plot(field, ramp, tol=1e-12):
    if ramp is None:
        return field
    return np.ma.masked_where(ramp < (1.0 - tol), field)


def wrap_to_pi(phi):
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def _default_post_sigma(mu):
    if mu is None:
        raise ValueError("mu is required when post_sigma is not provided.")
    return 5.0 * np.sqrt(max(0.0, 1.0 - float(mu) ** 2))


def _select_phase_amp_fields(result, prefer_sym=True, use_smoothed=False):
    if use_smoothed:
        phase_wrapped = result.get("phase_grid_symmetric_wrapped_smooth" if prefer_sym else "phase_grid_wrapped_smooth")
        phase_unwrapped = result.get("phase_grid_symmetric_unwrapped_smooth" if prefer_sym else "phase_grid_unwrapped_smooth")
        amplitude = result.get("analytic_amplitude_grid_symmetric_smooth" if prefer_sym else "analytic_amplitude_grid_smooth")
    else:
        phase_wrapped = result.get("phase_grid_symmetric_wrapped" if prefer_sym else "phase_grid_wrapped")
        phase_unwrapped = result.get("phase_grid_symmetric_unwrapped" if prefer_sym else "phase_grid_unwrapped")
        amplitude = result.get("analytic_amplitude_grid_symmetric" if prefer_sym else "analytic_amplitude_grid")

    if phase_wrapped is None or phase_unwrapped is None or amplitude is None:
        kind = "smoothed" if use_smoothed else "raw"
        raise ValueError(f"Missing {kind} phase/amplitude fields for prefer_sym={prefer_sym}")

    return phase_wrapped, phase_unwrapped, amplitude


def postprocess_phase_amplitude(
    result,
    prefer_sym=True,
    sigma=None,
    sigma_prefactor=2.0,
    mask_tol=0.99,
    mode="nearest",
):
    mu = result.get("mu", None)
    if sigma is None:
        if mu is None:
            raise ValueError("mu is required to derive postprocessing sigma.")
        sigma = float(sigma_prefactor) * np.sqrt(max(0.0, 1.0 - float(mu) ** 2))

    phase_wrapped, phase_unwrapped, amplitude = _select_phase_amp_fields(
        result, prefer_sym=prefer_sym, use_smoothed=False
    )

    ramp = result.get("phase_ramp", None)
    if ramp is None:
        ramp = result.get("ramp", None)

    phase_unwrapped_smooth = np.empty_like(phase_unwrapped)
    phase_wrapped_smooth = np.empty_like(phase_wrapped)
    amplitude_smooth = np.empty_like(amplitude)
    amplitude_cos = np.empty_like(amplitude)
    amplitude_cos_smooth = np.empty_like(amplitude)

    nt = phase_unwrapped.shape[-1]
    for fi in range(nt):
        phu = phase_unwrapped[:, :, fi]
        amp = amplitude[:, :, fi]

        phu_work = np.array(phu, copy=True)
        amp_work = np.array(amp, copy=True)

        if ramp is not None:
            mask = ramp < mask_tol
            phu_work = phu_work.copy()
            amp_work = amp_work.copy()
            phu_work[mask] = 0.0
            amp_work[mask] = 0.0

        phu_s = gaussian_filter(phu_work, sigma=sigma, mode=mode)
        amp_s = gaussian_filter(amp_work, sigma=sigma, mode=mode)

        phase_unwrapped_smooth[:, :, fi] = phu_s
        phase_wrapped_smooth[:, :, fi] = wrap_to_pi(phu_s)
        amplitude_smooth[:, :, fi] = amp_s
        amplitude_cos[:, :, fi] = amp * np.cos(phu)
        amplitude_cos_smooth[:, :, fi] = amp_s * np.cos(phu_s)

    if prefer_sym:
        return {
            "postprocess_meta": {
                "sigma": float(sigma),
                "sigma_prefactor": float(sigma_prefactor),
                "mask_tol": float(mask_tol),
                "mode": mode,
                "prefer_sym": bool(prefer_sym),
            },
            "phase_grid_symmetric_unwrapped_smooth": phase_unwrapped_smooth,
            "phase_grid_symmetric_wrapped_smooth": phase_wrapped_smooth,
            "analytic_amplitude_grid_symmetric_smooth": amplitude_smooth,
            "amplitude_cos_symmetric": amplitude_cos,
            "amplitude_cos_symmetric_smooth": amplitude_cos_smooth,
        }
    else:
        return {
            "postprocess_meta": {
                "sigma": float(sigma),
                "sigma_prefactor": float(sigma_prefactor),
                "mask_tol": float(mask_tol),
                "mode": mode,
                "prefer_sym": bool(prefer_sym),
            },
            "phase_grid_unwrapped_smooth": phase_unwrapped_smooth,
            "phase_grid_wrapped_smooth": phase_wrapped_smooth,
            "analytic_amplitude_grid_smooth": amplitude_smooth,
            "amplitude_cos": amplitude_cos,
            "amplitude_cos_smooth": amplitude_cos_smooth,
        }


def make_phase_summary_plot_four_panels(
    result,
    fig_path,
    prefer_sym=True,
    frame_index=-1,
    mask_with_ramp=True,
):
    x = result["x"]
    y = result["y"]
    u = result["u"]
    ramp = result.get("phase_ramp", None)
    if ramp is None:
        ramp = result.get("ramp", None)

    extent = [x[0], x[-1], y[0], y[-1]]

    if u.ndim == 3:
        fi = frame_index if frame_index >= 0 else u.shape[-1] - 1
        u_plot = u[:, :, fi]
    else:
        fi = 0
        u_plot = u

    if prefer_sym:
        phase_wrapped = result.get("phase_grid_symmetric_wrapped")
        phase_unwrapped = result.get("phase_grid_symmetric_unwrapped")
    else:
        phase_wrapped = result.get("phase_grid_wrapped")
        phase_unwrapped = result.get("phase_grid_unwrapped")

    if phase_wrapped is None or phase_unwrapped is None:
        raise ValueError("Wrapped/unwrapped phase grids not found in result.")

    if phase_wrapped.ndim == 3:
        phase_wrapped = phase_wrapped[:, :, fi]
    if phase_unwrapped.ndim == 3:
        phase_unwrapped = phase_unwrapped[:, :, fi]

    cos_unwrapped = np.cos(phase_unwrapped)

    if mask_with_ramp and ramp is not None:
        u_plot = np.ma.masked_where(ramp < 0.99, u_plot)
        phase_wrapped = np.ma.masked_where(ramp < 0.99, phase_wrapped)
        phase_unwrapped = np.ma.masked_where(ramp < 0.99, phase_unwrapped)
        cos_unwrapped = np.ma.masked_where(ramp < 0.99, cos_unwrapped)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    im0 = axs[0].imshow(u_plot, origin="lower", extent=extent, cmap="gray")
    axs[0].set_title("pattern u")
    fig.colorbar(im0, ax=axs[0], shrink=0.85)

    im1 = axs[1].imshow(
        phase_wrapped,
        origin="lower",
        extent=extent,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axs[1].set_title("wrapped phase")
    fig.colorbar(im1, ax=axs[1], shrink=0.85)

    im2 = axs[2].imshow(
        phase_unwrapped,
        origin="lower",
        extent=extent,
        cmap="viridis",
    )
    axs[2].set_title("unwrapped phase")
    fig.colorbar(im2, ax=axs[2], shrink=0.85)

    im3 = axs[3].imshow(
        cos_unwrapped,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )
    axs[3].set_title("cos(unwrapped phase)")
    fig.colorbar(im3, ax=axs[3], shrink=0.85)

    for ax in axs:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    meta = result.get("phase_meta", {})
    mu = result.get("mu", None)
    mu_str = f"mu={mu:.3f}" if mu is not None else "mu=?"
    seed_half = meta.get("seed_half", "upper")
    ramp_mode = meta.get("phase_ramp_mode", "none")
    frame_label = "initial" if fi == 0 else ("final" if (u.ndim == 3 and fi == u.shape[-1] - 1) else f"frame {fi}")

    fig.suptitle(
        f"{mu_str}   {frame_label}   seed_half={seed_half}   phase_ramp_mode={ramp_mode}",
        y=0.98,
    )

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_coordinate_line_diagnostic(
    result,
    fig_path,
    frame_index=-1,
    n_lines_to_show=12,
    points_per_line=12,
):
    x = result["x"]
    y = result["y"]
    u = result["u"]
    phase_ramp = result.get("phase_ramp", None)
    knee_bdry = result.get("knee_bdry_phase", result.get("knee_bdry"))
    coordinate_lines = result.get("coordinate_lines", None)

    if coordinate_lines is None:
        raise ValueError("coordinate_lines not found in result.")

    extent = [x[0], x[-1], y[0], y[-1]]

    if u.ndim == 3:
        fi = frame_index if frame_index >= 0 else u.shape[-1] - 1
        u_plot = u[:, :, fi]
    else:
        fi = 0
        u_plot = u

    if phase_ramp is None:
        phase_ramp = np.ones_like(u_plot)

    u_ramped = u_plot * phase_ramp

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    vmax = np.nanmax(np.abs(u_ramped))
    vmax = 1.0 if (not np.isfinite(vmax) or vmax == 0.0) else vmax

    im = ax.imshow(
        u_ramped,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, shrink=0.85)

    if knee_bdry is not None:
        ax.plot(knee_bdry[0], knee_bdry[1], "k-", lw=1.0, alpha=0.8)

    lines_t = coordinate_lines[fi]
    n_total = len(lines_t)

    if n_total > 0:
        line_ids = np.linspace(0, n_total - 1, min(n_lines_to_show, n_total), dtype=int)

        for j in line_ids:
            line = np.asarray(lines_t[j])
            if line.ndim != 2 or line.shape[0] < 2:
                continue

            ax.plot(line[:, 0], line[:, 1], color="k", lw=0.8, alpha=0.4)

            m = line.shape[0]
            sample_ids = np.linspace(0, m - 1, min(points_per_line, m), dtype=int)
            ax.scatter(
                line[sample_ids, 0],
                line[sample_ids, 1],
                s=16,
                c="yellow",
                edgecolors="k",
                linewidths=0.3,
                alpha=0.9,
            )

    ax.set_title("pattern × phase ramp with coordinate lines")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    meta = result.get("phase_meta", {})
    mu = result.get("mu", None)
    mu_str = f"mu={mu:.3f}" if mu is not None else "mu=?"
    seed_half = meta.get("seed_half", "upper")
    ramp_mode = meta.get("phase_ramp_mode", "none")
    frame_label = "initial" if fi == 0 else ("final" if (u.ndim == 3 and fi == u.shape[-1] - 1) else f"frame {fi}")

    fig.suptitle(
        f"{mu_str}   {frame_label}   seed_half={seed_half}   phase_ramp_mode={ramp_mode}",
        y=0.98,
    )

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_phase_profile_plot(
    result,
    fig_path,
    prefer_sym=True,
    frame_index=-1,
    use_smoothed=False,
    mask_tol=0.99,
):
    x = result["x"]
    y = result["y"]
    ramp = result.get("phase_ramp", None)
    if ramp is None:
        ramp = result.get("ramp", None)

    phase_wrapped, phase_unwrapped, amplitude = _select_phase_amp_fields(
        result, prefer_sym=prefer_sym, use_smoothed=use_smoothed
    )

    fi = frame_index if frame_index >= 0 else phase_unwrapped.shape[-1] - 1
    phw = phase_wrapped[:, :, fi]
    phu = phase_unwrapped[:, :, fi]
    amp = amplitude[:, :, fi]
    amp_cos = amp * np.cos(phu)

    if ramp is None:
        ramp = np.ones((len(y), len(x)))

    phw_m = masked_for_plot(phw, ramp, tol=1.0 - mask_tol)
    phu_m = masked_for_plot(phu, ramp, tol=1.0 - mask_tol)
    amp_m = masked_for_plot(amp, ramp, tol=1.0 - mask_tol)
    amp_cos_m = masked_for_plot(amp_cos, ramp, tol=1.0 - mask_tol)

    ymid_idx = len(y) // 2
    ymid_lo = max(0, ymid_idx - 1)
    ymid_hi = min(len(y) - 1, ymid_idx + 1)

    valid_mask = ramp >= mask_tol
    valid_frac_by_x = np.mean(valid_mask, axis=0)

    min_valid_frac = 0.90
    candidate_x = np.where(valid_frac_by_x >= min_valid_frac)[0]

    if len(candidate_x) > 0:
        ix_left = int(candidate_x[0])
    else:
        ix_left = 0

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].plot(y, phw_m[:, ix_left], lw=2, label="wrapped")
    axs[0, 0].plot(y, phu_m[:, ix_left], lw=2, label="unwrapped")
    axs[0, 0].set_title(
        f"phase at left-valid x = {x[ix_left]:.3f} (col {ix_left})"
    )
    axs[0, 0].set_xlabel("y")
    axs[0, 0].legend()

    axs[0, 1].plot(x, phu_m[ymid_lo, :], lw=2, label=f"y[mid-1]={y[ymid_lo]:.3f}")
    axs[0, 1].plot(x, phu_m[ymid_idx, :], lw=2, label=f"y[mid]={y[ymid_idx]:.3f}")
    axs[0, 1].plot(x, phu_m[ymid_hi, :], lw=2, label=f"y[mid+1]={y[ymid_hi]:.3f}")
    axs[0, 1].set_title("unwrapped phase near y_mid")
    axs[0, 1].set_xlabel("x")
    axs[0, 1].legend()

    axs[1, 0].plot(x, amp_m[ymid_lo, :], lw=2, label=f"y[mid-1]={y[ymid_lo]:.3f}")
    axs[1, 0].plot(x, amp_m[ymid_idx, :], lw=2, label=f"y[mid]={y[ymid_idx]:.3f}")
    axs[1, 0].plot(x, amp_m[ymid_hi, :], lw=2, label=f"y[mid+1]={y[ymid_hi]:.3f}")
    axs[1, 0].set_title("amplitude near y_mid")
    axs[1, 0].set_xlabel("x")
    axs[1, 0].legend()

    axs[1, 1].plot(x, amp_cos_m[ymid_lo, :], lw=2, label=f"y[mid-1]={y[ymid_lo]:.3f}")
    axs[1, 1].plot(x, amp_cos_m[ymid_idx, :], lw=2, label=f"y[mid]={y[ymid_idx]:.3f}")
    axs[1, 1].plot(x, amp_cos_m[ymid_hi, :], lw=2, label=f"y[mid+1]={y[ymid_hi]:.3f}")
    axs[1, 1].set_title("amplitude * cos(unwrapped phase) near y_mid")
    axs[1, 1].set_xlabel("x")
    axs[1, 1].legend()

    for ax in axs.flat:
        ax.grid(True, alpha=0.25)

    meta = result.get("phase_meta", {})
    mu = result.get("mu", None)
    mu_str = f"mu={mu:.3f}" if mu is not None else "mu=?"
    seed_half = meta.get("seed_half", "upper")
    tag = "smoothed" if use_smoothed else "raw"
    frame_label = "initial" if fi == 0 else ("final" if fi == phase_unwrapped.shape[-1] - 1 else f"frame {fi}")

    fig.suptitle(
        f"{mu_str}   {frame_label}   {tag} profiles   seed_half={seed_half}",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def make_amplitude_diagnostic_plot(
    result,
    fig_path,
    prefer_sym=True,
    frame_index=-1,
    use_smoothed=False,
    mask_with_ramp=True,
    mask_tol=0.99,
):
    x = result["x"]
    y = result["y"]
    extent = [x[0], x[-1], y[0], y[-1]]
    ramp = result.get("phase_ramp", None)
    if ramp is None:
        ramp = result.get("ramp", None)

    _, phase_unwrapped, amplitude = _select_phase_amp_fields(
        result, prefer_sym=prefer_sym, use_smoothed=use_smoothed
    )

    fi = frame_index if frame_index >= 0 else phase_unwrapped.shape[-1] - 1
    phu = phase_unwrapped[:, :, fi]
    amp = amplitude[:, :, fi]
    amp_cos = amp * np.cos(phu)

    if mask_with_ramp and ramp is not None:
        amp = np.ma.masked_where(ramp < mask_tol, amp)
        amp_cos = np.ma.masked_where(ramp < mask_tol, amp_cos)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axs[0].imshow(amp, origin="lower", extent=extent, cmap="magma")
    axs[0].set_title("amplitude")
    fig.colorbar(im0, ax=axs[0], shrink=0.85)

    vmax = np.nanmax(np.abs(np.asarray(amp_cos)))
    vmax = 1.0 if (not np.isfinite(vmax) or vmax == 0.0) else vmax
    im1 = axs[1].imshow(
        amp_cos,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    axs[1].set_title("amplitude * cos(unwrapped phase)")
    fig.colorbar(im1, ax=axs[1], shrink=0.85)

    for ax in axs:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    meta = result.get("phase_meta", {})
    mu = result.get("mu", None)
    mu_str = f"mu={mu:.3f}" if mu is not None else "mu=?"
    seed_half = meta.get("seed_half", "upper")
    tag = "smoothed" if use_smoothed else "raw"
    frame_label = "initial" if fi == 0 else ("final" if fi == phase_unwrapped.shape[-1] - 1 else f"frame {fi}")

    fig.suptitle(
        f"{mu_str}   {frame_label}   {tag} amplitude diagnostics   seed_half={seed_half}",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)




def make_phase_summary_plot(result, fig_path, n_overlay_lines=24):
    x = result["x"]
    y = result["y"]
    u = result["u"]
    ramp = result.get("phase_ramp")
    if ramp is None:
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
    mu_val = meta.get("mu", None)
    mu_str = f"mu={mu_val:.3f}" if mu_val is not None else "mu=?"
    seed_half = meta.get("seed_half", "upper")
    fig.suptitle(
        f"{mu_str} ds={meta['ds']:.3f} max_steps={meta['max_steps']} "
        f"source={meta['k_field_source']} seed_half={seed_half}",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)