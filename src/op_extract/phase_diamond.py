import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

# Ensure src/ on path, mirroring other modules
import sys
sys.path.insert(0, str(_ROOT / "src"))

from utils.geometry import build_diamond_ramp_general


# -----------------------------------------------------------------------------
# Loading / shape helpers
# -----------------------------------------------------------------------------

def ensure_u_shape(u, x, y):
    u = np.asarray(u)
    if u.ndim != 3:
        raise ValueError(f"expected 3D array, got shape {u.shape}")

    if u.shape[0] == len(y) and u.shape[1] == len(x):
        return u

    if u.shape[1] == len(y) and u.shape[2] == len(x):
        return np.transpose(u, (1, 2, 0))

    if u.shape[0] == len(x) and u.shape[1] == len(y):
        return np.transpose(u, (1, 0, 2))

    raise ValueError(
        f"cannot reconcile array shape {u.shape} with x ({len(x)},) and y ({len(y)},)"
    )



def _maybe_item_string(raw):
    if raw is None:
        return None
    try:
        return raw.item()
    except Exception:
        return str(raw)



def _get_json_mu(*json_blobs):
    for blob in json_blobs:
        if not blob:
            continue
        try:
            obj = json.loads(blob)
            mu = obj.get("mu")
            if mu is not None:
                return float(mu)
        except Exception:
            pass
    return None



def load_uhu_npz_diamond(uhu_path):
    uhu_path = Path(uhu_path)
    data = np.load(uhu_path, allow_pickle=True)

    x = data["x"] if "x" in data else data["xx"]
    y = data["y"] if "y" in data else data["yy"]

    if "u" in data:
        u = ensure_u_shape(data["u"], x, y)
    elif "uu" in data:
        u = ensure_u_shape(data["uu"], x, y)
    else:
        raise ValueError(f"{uhu_path.name}: missing u/uu field")

    tt = data["tt"] if "tt" in data else None

    uhu_meta_json = _maybe_item_string(data["uhu_meta_json"]) if "uhu_meta_json" in data else None
    sh_meta_json = _maybe_item_string(data["sh_meta_json"]) if "sh_meta_json" in data else None

    mu = _get_json_mu(uhu_meta_json, sh_meta_json)
    if mu is None and "mu" in data:
        mu = float(np.asarray(data["mu"]).ravel()[0])

    def get_field(*names):
        for name in names:
            if name in data:
                arr = data[name]
                if getattr(arr, "ndim", 0) == 3:
                    return ensure_u_shape(arr, x, y)
                return arr
        return None

    ramp = get_field("ramp", "ramp_inner")
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

    bdry = get_field("bdry")
    bdry_inner = get_field("bdry_inner")
    upper_bdry = get_field("upper_boundary_points", "upper_bdry")

    return {
        "x": x,
        "y": y,
        "u": u,
        "tt": tt,
        "mu": mu,
        "ramp": ramp,
        "k": k,
        "A": A,
        "k1_sym": k1_sym,
        "k2_sym": k2_sym,
        "k1_orig": k1_orig,
        "k2_orig": k2_orig,
        "bdry": bdry,
        "bdry_inner": bdry_inner,
        "upper_bdry": upper_bdry,
        "uhu_meta_json": uhu_meta_json,
        "sh_meta_json": sh_meta_json,
        "source_path": str(uhu_path),
    }


# -----------------------------------------------------------------------------
# Boundary / ramp helpers
# -----------------------------------------------------------------------------

def extract_upper_boundary(bdry, y_center=None):
    bdry = np.asarray(bdry)
    if bdry.ndim != 2 or bdry.shape[0] != 2:
        raise ValueError(f"boundary must have shape (2, N), got {bdry.shape}")

    xb = bdry[0]
    yb = bdry[1]

    if y_center is None:
        y_center = 0.5 * (np.nanmin(yb) + np.nanmax(yb))

    mask = yb > y_center
    if not np.any(mask):
        raise RuntimeError("No upper boundary points found.")

    upper = np.vstack([xb[mask], yb[mask]])
    order = np.argsort(upper[0])
    return upper[:, order]



def build_upper_diamond_boundary_from_mu(x, y, mu, margin, n_samples=512):
    x = np.asarray(x)
    y = np.asarray(y)

    if margin is None:
        raise ValueError("margin is required to rebuild diamond boundary")
    if mu is None:
        raise ValueError("mu is required to rebuild diamond boundary")
    if abs(mu) < 1e-12:
        raise ValueError("mu is too small to build diamond boundary")

    x0 = 0.5 * (x[0] + x[-1])
    y0 = 0.5 * (y[0] + y[-1])
    Ly = y[-1] - y[0]

    k1_far = np.sqrt(max(0.0, 1.0 - float(mu) ** 2))
    slope = k1_far / float(mu)

    diamond_height = 2.0 * float(margin) * Ly
    y_top = y0 + 0.5 * diamond_height
    y_mid = y0
    x_half = abs(slope) * (diamond_height / 2.0)

    x_left = x0 - x_half
    x_right = x0 + x_half

    n_left = max(2, int(n_samples) // 2)
    n_right = max(2, int(n_samples) - n_left)

    xb_left = np.linspace(x_left, x0, n_left, endpoint=False)
    yb_left = np.linspace(y_mid, y_top, n_left, endpoint=False)

    xb_right = np.linspace(x0, x_right, n_right)
    yb_right = np.linspace(y_top, y_mid, n_right)

    xb = np.concatenate([xb_left, xb_right])
    yb = np.concatenate([yb_left, yb_right])
    return np.vstack([xb, yb])



def resample_boundary_points(bdry, n_seeds):
    xb, yb = bdry
    pts = np.vstack([xb, yb]).T
    diffs = np.diff(pts, axis=0)
    seglen = np.sqrt((diffs ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    if s[-1] <= 0:
        raise RuntimeError("Degenerate boundary supplied for resampling.")
    s /= s[-1]
    st = np.linspace(0.0, 1.0, n_seeds)
    xs = np.interp(st, s, xb)
    ys = np.interp(st, s, yb)
    return np.vstack([xs, ys])



def inset_boundary_toward_center(bdry, x_center, y_center, inset_frac):
    bdry = np.asarray(bdry)
    if bdry.ndim != 2 or bdry.shape[0] != 2:
        raise ValueError(f"boundary must have shape (2, N), got {bdry.shape}")

    if inset_frac is None or inset_frac <= 0.0:
        return bdry.copy()

    xb = bdry[0]
    yb = bdry[1]
    xb_new = x_center + (1.0 - inset_frac) * (xb - x_center)
    yb_new = y_center + (1.0 - inset_frac) * (yb - y_center)
    return np.vstack([xb_new, yb_new])


def boundary_min_ramp(bdry, ramp, x, y):
    if ramp is None:
        return np.inf

    interp_ramp = RegularGridInterpolator(
        (y, x), ramp, bounds_error=False, fill_value=0.0
    )
    vals = np.array([interp_ramp((yy, xx))[()] for xx, yy in bdry.T])
    return float(np.min(vals))


def inset_boundary_until_ramp_ok(
    bdry,
    ramp,
    x,
    y,
    x_center,
    y_center,
    inset_frac0=0.0,
    seed_ramp_min=0.0,
    inset_step=0.01,
    inset_max=0.35,
):
    if ramp is None or seed_ramp_min <= 0.0:
        return inset_boundary_toward_center(bdry, x_center, y_center, inset_frac0), float(inset_frac0)

    inset_frac = max(0.0, float(inset_frac0))
    best = inset_boundary_toward_center(bdry, x_center, y_center, inset_frac)

    while inset_frac <= inset_max:
        cand = inset_boundary_toward_center(bdry, x_center, y_center, inset_frac)
        rmin = boundary_min_ramp(cand, ramp, x, y)
        if rmin >= seed_ramp_min:
            return cand, float(inset_frac)
        best = cand
        inset_frac += inset_step

    return best, float(min(inset_frac, inset_max))



def choose_phase_ramp_diamond(
    uhu_data,
    phase_ramp_mode="saved",
    phase_margin=None,
    phase_tanh_scale=1.0,
    phase_smooth_sigma=1.0,
):
    if phase_ramp_mode == "none":
        return None

    if phase_ramp_mode == "saved":
        ramp = uhu_data.get("ramp", None)
        if ramp is None:
            raise ValueError("phase_ramp_mode='saved' but no ramp found in uHu data")
        return np.asarray(ramp)

    if phase_ramp_mode == "rebuild":
        x = uhu_data["x"]
        y = uhu_data["y"]
        mu = uhu_data.get("mu", None)
        if mu is None:
            raise ValueError("mu required to rebuild diamond ramp")
        if phase_margin is None:
            raise ValueError("phase_margin required when phase_ramp_mode='rebuild'")
        return build_diamond_ramp_general(
            x,
            y,
            mu,
            margin=phase_margin,
            tanh_scale=phase_tanh_scale,
            smooth_sigma=phase_smooth_sigma,
        )

    raise ValueError(f"unknown phase_ramp_mode={phase_ramp_mode}")


# -----------------------------------------------------------------------------
# Shared phase helpers
# -----------------------------------------------------------------------------

def wrap_to_pi(phi):
    return (phi + np.pi) % (2.0 * np.pi) - np.pi



def _mirror_line_values(arr):
    npts = len(arr)
    mid = npts // 2
    upper = arr[: mid + 1] if npts % 2 else arr[:mid]
    lower = upper[:-1][::-1] if npts % 2 else upper[::-1]
    return np.concatenate([upper, lower])



def _object_time_line_array(per_time_lists):
    nt = len(per_time_lists)
    nlines = len(per_time_lists[0]) if nt > 0 else 0
    out = np.empty((nt, nlines), dtype=object)
    for t in range(nt):
        for l in range(nlines):
            out[t, l] = per_time_lists[t][l]
    return out



def _select_phase_amp_fields(result, prefer_sym=True, use_smoothed=False):
    if use_smoothed:
        phase_wrapped = result.get(
            "phase_grid_symmetric_wrapped_smooth" if prefer_sym else "phase_grid_wrapped_smooth"
        )
        phase_unwrapped = result.get(
            "phase_grid_symmetric_unwrapped_smooth" if prefer_sym else "phase_grid_unwrapped_smooth"
        )
        amplitude = result.get(
            "analytic_amplitude_grid_symmetric_smooth" if prefer_sym else "analytic_amplitude_grid_smooth"
        )
    else:
        phase_wrapped = result.get(
            "phase_grid_symmetric_wrapped" if prefer_sym else "phase_grid_wrapped"
        )
        phase_unwrapped = result.get(
            "phase_grid_symmetric_unwrapped" if prefer_sym else "phase_grid_unwrapped"
        )
        amplitude = result.get(
            "analytic_amplitude_grid_symmetric" if prefer_sym else "analytic_amplitude_grid"
        )

    if phase_wrapped is None or phase_unwrapped is None or amplitude is None:
        kind = "smoothed" if use_smoothed else "raw"
        raise ValueError(f"missing {kind} phase/amplitude fields for prefer_sym={prefer_sym}")

    return phase_wrapped, phase_unwrapped, amplitude


# -----------------------------------------------------------------------------
# Phase extraction core
# -----------------------------------------------------------------------------

def compute_phase_from_k_diamond(
    u_ts,
    k1_ts,
    k2_ts,
    x,
    y,
    upper_boundary_points,
    max_steps=10000,
    ds=0.2,
    ramp=None,
    ramp_sample_thresh=0.0,
):
    Ny, Nx, nt = u_ts.shape
    Xg, Yg = np.meshgrid(x, y)
    nlines = upper_boundary_points.shape[1]

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

    for t in range(nt):
        print(f"> [diamond_phase] Processing timestep {t + 1}/{nt}")
        k1 = k1_ts[:, :, t]
        k2 = k2_ts[:, :, t]
        u = u_ts[:, :, t]
        u_for_phase = u if ramp is None else u * ramp

        interp_k1 = RegularGridInterpolator((y, x), k1, bounds_error=False, fill_value=0.0)
        interp_k2 = RegularGridInterpolator((y, x), k2, bounds_error=False, fill_value=0.0)
        interp_u = RegularGridInterpolator((y, x), u_for_phase, bounds_error=False, fill_value=0.0)
        interp_ramp = None
        if ramp is not None:
            interp_ramp = RegularGridInterpolator((y, x), ramp, bounds_error=False, fill_value=0.0)

        orig_lines = []
        for i in range(nlines):
            x0 = upper_boundary_points[0, i]
            y0 = upper_boundary_points[1, i]
            line = [(x0, y0)]

            for _ in range(max_steps):
                kx = interp_k1((y0, x0))[()]
                ky = interp_k2((y0, x0))[()]
                mag = np.hypot(kx, ky)
                if mag == 0:
                    break

                dir_sign = -1 if y0 > 0 else 1
                dx = dir_sign * kx / mag * ds
                dy = dir_sign * ky / mag * ds
                x0_new = x0 + dx
                y0_new = y0 + dy
                line.append((x0_new, y0_new))
                x0, y0 = x0_new, y0_new
                if y0 < .25:
                    break

            orig_lines.append(np.array(line))

        refl_lines = []
        for line in orig_lines:
            axis_mask = np.isclose(line[:, 1], 0.0, atol=1e-8)
            non_axis = ~axis_mask
            reflected = np.vstack([line[non_axis, 0], -line[non_axis, 1]]).T
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
        phases_symmetric_wrapped_t = []
        phases_symmetric_unwrapped_t = []
        amplitudes_symmetric_t = []

        sampled_points = []
        phase_samples_wrapped = []
        phase_samples_unwrapped = []
        amplitude_samples = []
        phase_samples_sym_wrapped = []
        phase_samples_sym_unwrapped = []
        amplitude_samples_sym = []

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
                line, phase_wrapped, phase_unwrapped, amplitude,
                sym_wrapped, sym_unwrapped, sym_amp,
            ):
                if interp_ramp is not None and ramp_sample_thresh > 0.0:
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

        if len(sampled_points) >= 3:
            pts = np.array(sampled_points)
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
        "coordinate_lines": _object_time_line_array(all_stitched_lines),
        "phase_lines_wrapped": _object_time_line_array(all_phases_on_lines_wrapped),
        "phase_lines_unwrapped": _object_time_line_array(all_phases_on_lines_unwrapped),
        "phase_grid_wrapped": np.stack(all_grid_phases_wrapped, axis=-1),
        "phase_grid_unwrapped": np.stack(all_grid_phases_unwrapped, axis=-1),
        "phase_lines_symmetric_wrapped": _object_time_line_array(all_phases_on_lines_symmetric_wrapped),
        "phase_lines_symmetric_unwrapped": _object_time_line_array(all_phases_on_lines_symmetric_unwrapped),
        "phase_grid_symmetric_wrapped": np.stack(all_grid_phases_symmetric_wrapped, axis=-1),
        "phase_grid_symmetric_unwrapped": np.stack(all_grid_phases_symmetric_unwrapped, axis=-1),
        "analytic_amplitude_lines": _object_time_line_array(all_analytic_amplitudes_on_lines),
        "analytic_amplitude_grid": np.stack(all_grid_analytic_amplitudes, axis=-1),
        "analytic_amplitude_lines_symmetric": _object_time_line_array(all_analytic_amplitudes_on_lines_symmetric),
        "analytic_amplitude_grid_symmetric": np.stack(all_grid_analytic_amplitudes_symmetric, axis=-1),
    }


# -----------------------------------------------------------------------------
# Wrapper / postprocess
# -----------------------------------------------------------------------------

def compute_diamond_phase_from_uhu(
    uhu_data,
    mu=None,
    n_phase_seeds=128,
    ds=0.2,
    max_steps=10000,
    prefer_sym=True,
    phase_ramp_mode="saved",
    phase_margin=None,
    phase_tanh_scale=1.0,
    phase_smooth_sigma=1.0,
    ramp_sample_thresh=0.0,
    boundary_source="inner",
    seed_boundary_inset_frac=0.0,
    seed_ramp_min=0.0,
    seed_inset_step=0.01,
    seed_inset_max=0.35,
):
    x = uhu_data["x"]
    y = uhu_data["y"]
    u = uhu_data["u"]
    mu_eff = mu if mu is not None else uhu_data.get("mu", None)

    bdry_inner = uhu_data.get("bdry_inner", None)
    bdry_outer = uhu_data.get("bdry", None)
    upper_bdry_saved = uhu_data.get("upper_bdry", None)
    if upper_bdry_saved is not None:
        upper_bdry = np.asarray(upper_bdry_saved)
        bdry_source_used = "upper_bdry"
    else:
        if boundary_source == "inner":
            bdry_raw = bdry_inner if bdry_inner is not None else bdry_outer
            bdry_source_used = "bdry_inner" if bdry_inner is not None else "bdry"
        elif boundary_source == "outer":
            bdry_raw = bdry_outer
            bdry_source_used = "bdry"
        else:
            raise ValueError(f"Unknown boundary_source={boundary_source!r}")

        if bdry_raw is not None:
            upper_bdry = extract_upper_boundary(bdry_raw)
        else:
            mu_for_geom = mu_eff if mu_eff is not None else uhu_data.get("mu", None)
            if mu_for_geom is None:
                raise ValueError(
                    "No usable diamond boundary found in uHu data, and could not infer mu "
                    "to rebuild the upper diamond boundary."
                )

            margin_for_bdry = phase_margin
            if margin_for_bdry is None:
                meta_json = uhu_data.get("uhu_meta_json", None)
                if meta_json is not None:
                    try:
                        import json
                        meta = json.loads(meta_json.item() if hasattr(meta_json, "item") else meta_json)
                        margin_for_bdry = meta.get("margin", None)
                    except Exception:
                        margin_for_bdry = None

            if margin_for_bdry is None:
                raise ValueError(
                    "No usable diamond boundary found in uHu data. "
                    "Please supply phase_margin in the runner config, or save margin/boundary metadata in the uHu file."
                )

            upper_bdry = build_upper_diamond_boundary_from_mu(
                x,
                y,
                mu_for_geom,
                margin=margin_for_bdry,
                n_samples=max(4 * int(n_phase_seeds), 256),
            )
            bdry_source_used = "reconstructed_upper"

    ramp_phase = choose_phase_ramp_diamond(
        uhu_data,
        phase_ramp_mode=phase_ramp_mode,
        phase_margin=phase_margin,
        phase_tanh_scale=phase_tanh_scale,
        phase_smooth_sigma=phase_smooth_sigma,
    )

    x_center = 0.5 * (x[0] + x[-1])
    y_center = 0.5 * (y[0] + y[-1])

    upper_bdry_seed, seed_inset_frac_used = inset_boundary_until_ramp_ok(
        upper_bdry,
        ramp=ramp_phase,
        x=x,
        y=y,
        x_center=x_center,
        y_center=y_center,
        inset_frac0=seed_boundary_inset_frac,
        seed_ramp_min=seed_ramp_min,
        inset_step=seed_inset_step,
        inset_max=seed_inset_max,
    )

    upper_bdry_phase = resample_boundary_points(upper_bdry_seed, n_phase_seeds)

    if prefer_sym and uhu_data.get("k1_sym") is not None and uhu_data.get("k2_sym") is not None:
        k1_ts = uhu_data["k1_sym"]
        k2_ts = uhu_data["k2_sym"]
        k_field_source = "sym"
    else:
        k1_ts = uhu_data["k1_orig"]
        k2_ts = uhu_data["k2_orig"]
        k_field_source = "orig"

    phase = compute_phase_from_k_diamond(
        u_ts=u,
        k1_ts=k1_ts,
        k2_ts=k2_ts,
        x=x,
        y=y,
        upper_boundary_points=upper_bdry_phase,
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
        "bdry": bdry_outer,
        "bdry_inner": bdry_inner,
        "upper_bdry": upper_bdry,
        "upper_bdry_seed": upper_bdry_seed,
        "upper_bdry_phase": upper_bdry_phase,
        "phase_meta": {
            "mu": None if mu_eff is None else float(mu_eff),
            "n_phase_seeds": int(n_phase_seeds),
            "ds": float(ds),
            "max_steps": int(max_steps),
            "k_field_source": k_field_source,
            "boundary_source": boundary_source,
            "boundary_source_used": bdry_source_used,
            "phase_ramp_mode": phase_ramp_mode,
            "phase_margin": None if phase_margin is None else float(phase_margin),
            "phase_tanh_scale": float(phase_tanh_scale),
            "phase_smooth_sigma": float(phase_smooth_sigma),
            "ramp_sample_thresh": float(ramp_sample_thresh),
            "seed_boundary_inset_frac": float(seed_boundary_inset_frac),
            "seed_inset_frac_used": float(seed_inset_frac_used),
            "seed_ramp_min": float(seed_ramp_min),
            "seed_inset_step": float(seed_inset_step),
            "seed_inset_max": float(seed_inset_max),
            "source_file": uhu_data.get("source_path"),
        },
        **phase,
    }



def postprocess_phase_amplitude(result, sigma=1.0, wrap_after_smoothing=True):
    out = dict(result)
    raw_pairs = [
        ("phase_grid_wrapped", "phase_grid_wrapped_smooth", True),
        ("phase_grid_unwrapped", "phase_grid_unwrapped_smooth", False),
        ("phase_grid_symmetric_wrapped", "phase_grid_symmetric_wrapped_smooth", True),
        ("phase_grid_symmetric_unwrapped", "phase_grid_symmetric_unwrapped_smooth", False),
        ("analytic_amplitude_grid", "analytic_amplitude_grid_smooth", False),
        ("analytic_amplitude_grid_symmetric", "analytic_amplitude_grid_symmetric_smooth", False),
    ]

    for key_in, key_out, is_wrapped in raw_pairs:
        arr = out.get(key_in)
        if arr is None:
            continue
        arr_s = np.empty_like(arr)
        for t in range(arr.shape[-1]):
            frame = arr[:, :, t]
            if is_wrapped and not wrap_after_smoothing:
                arr_s[:, :, t] = gaussian_filter(frame, sigma=sigma)
            elif is_wrapped and wrap_after_smoothing:
                arr_s[:, :, t] = wrap_to_pi(gaussian_filter(frame, sigma=sigma))
            else:
                arr_s[:, :, t] = gaussian_filter(frame, sigma=sigma)
        out[key_out] = arr_s

    out["phase_postprocess_sigma"] = float(sigma)
    return out


# -----------------------------------------------------------------------------
# Saving helpers
# -----------------------------------------------------------------------------

def save_phase_result_npz(result, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {}
    for k, v in result.items():
        if isinstance(v, dict):
            save_dict[k + "_json"] = np.array(json.dumps(v), dtype=object)
        else:
            save_dict[k] = v

    np.savez_compressed(out_path, **save_dict)
    return out_path


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def _masked(arr, ramp, tol):
    if ramp is None:
        return arr
    return np.ma.masked_where(ramp < tol, arr)



def make_phase_summary_plot_diamond(
    result,
    fig_path,
    prefer_sym=True,
    frame_index=-1,
    use_smoothed=False,
    mask_tol=0.2,
):
    x = result["x"]
    y = result["y"]
    u = result["u"]
    ramp = result.get("phase_ramp", None)
    phase_wrapped, phase_unwrapped, amplitude = _select_phase_amp_fields(
        result, prefer_sym=prefer_sym, use_smoothed=use_smoothed
    )

    fi = frame_index if frame_index >= 0 else u.shape[-1] - 1
    uu = u[:, :, fi]
    phw = _masked(phase_wrapped[:, :, fi], ramp, mask_tol)
    phu = _masked(phase_unwrapped[:, :, fi], ramp, mask_tol)
    cph = _masked(np.cos(phase_unwrapped[:, :, fi]), ramp, mask_tol)

    extent = [x[0], x[-1], y[0], y[-1]]
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    im0 = axs[0, 0].imshow(uu, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r")
    axs[0, 0].set_title("pattern")
    plt.colorbar(im0, ax=axs[0, 0], shrink=0.85)

    im1 = axs[0, 1].imshow(phw, origin="lower", extent=extent, aspect="auto", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axs[0, 1].set_title("wrapped phase")
    plt.colorbar(im1, ax=axs[0, 1], shrink=0.85)

    im2 = axs[1, 0].imshow(phu, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    axs[1, 0].set_title("unwrapped phase")
    plt.colorbar(im2, ax=axs[1, 0], shrink=0.85)

    im3 = axs[1, 1].imshow(cph, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axs[1, 1].set_title("cos(unwrapped phase)")
    plt.colorbar(im3, ax=axs[1, 1], shrink=0.85)

    upper = result.get("upper_bdry_phase", None)
    if upper is not None:
        axs[0, 0].scatter(upper[0], upper[1], s=8, c="k", alpha=0.6)

    for ax in axs.flat:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")

    tag = "smoothed" if use_smoothed else "raw"
    frame_label = "initial" if fi == 0 else ("final" if fi == u.shape[-1] - 1 else f"frame {fi}")
    fig.suptitle(f"diamond phase summary: {frame_label}, {tag}")
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)



def make_geometry_diagnostic_plot_diamond(result, fig_path, frame_index=-1, mask_tol=0.2):
    x = result["x"]
    y = result["y"]
    u = result["u"]
    ramp = result.get("phase_ramp", None)
    if ramp is None:
        ramp = result.get("ramp", None)

    fi = frame_index if frame_index >= 0 else u.shape[-1] - 1
    uu = u[:, :, fi]
    ur = uu if ramp is None else uu * ramp
    extent = [x[0], x[-1], y[0], y[-1]]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    im0 = axs[0].imshow(uu, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r")
    axs[0].set_title("pattern")
    plt.colorbar(im0, ax=axs[0], shrink=0.82)

    im1 = axs[1].imshow(ramp if ramp is not None else np.ones_like(uu), origin="lower", extent=extent, aspect="auto", cmap="magma", vmin=0, vmax=1)
    axs[1].set_title("phase ramp")
    plt.colorbar(im1, ax=axs[1], shrink=0.82)

    im2 = axs[2].imshow(ur, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r")
    axs[2].set_title("pattern * phase ramp")
    plt.colorbar(im2, ax=axs[2], shrink=0.82)

    upper_seed = result.get("upper_bdry_seed", None)
    upper_pts = result.get("upper_bdry_phase", None)

    if upper_seed is not None:
        axs[0].plot(upper_seed[0], upper_seed[1], "k-", lw=1.0, alpha=0.8)
        axs[2].plot(upper_seed[0], upper_seed[1], "k-", lw=1.0, alpha=0.8)

    if upper_pts is not None:
        axs[0].scatter(upper_pts[0], upper_pts[1], s=8, c="k", alpha=0.6)
        axs[2].scatter(upper_pts[0], upper_pts[1], s=8, c="k", alpha=0.6)

    for ax in axs:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")

    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)



def make_coordinate_line_diagnostic_diamond(result, fig_path, frame_index=-1, n_show=15):
    x = result["x"]
    y = result["y"]
    u = result["u"]
    ramp = result.get("phase_ramp", None)
    fi = frame_index if frame_index >= 0 else u.shape[-1] - 1
    uu = u[:, :, fi]
    ur = uu if ramp is None else uu * ramp
    lines = result["coordinate_lines"][fi]
    extent = [x[0], x[-1], y[0], y[-1]]

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(ur, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r")
    plt.colorbar(im, ax=ax, shrink=0.85)

    if len(lines) > 0:
        idx = np.linspace(0, len(lines) - 1, min(n_show, len(lines))).astype(int)
        for j in idx:
            line = lines[j]
            ax.plot(line[:, 0], line[:, 1], lw=1.0, alpha=0.85)
            ax.scatter(line[0, 0], line[0, 1], s=10)

    ax.set_title("coordinate lines on pattern * ramp")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)



def make_phase_profile_plot_diamond(
    result,
    fig_path,
    prefer_sym=True,
    frame_index=-1,
    use_smoothed=False,
    mask_tol=0.2,
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

    phw_m = np.ma.masked_where(ramp < mask_tol, phw)
    phu_m = np.ma.masked_where(ramp < mask_tol, phu)
    amp_m = np.ma.masked_where(ramp < mask_tol, amp)
    amp_cos_m = np.ma.masked_where(ramp < mask_tol, amp_cos)

    ix0 = np.argmin(np.abs(x))
    iy0 = np.argmin(np.abs(y))

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    axs[0, 0].plot(y, phw_m[:, ix0], lw=2, label="wrapped")
    axs[0, 0].plot(y, phu_m[:, ix0], lw=2, label="unwrapped")
    axs[0, 0].set_title(f"phase along y-axis (x={x[ix0]:.3f})")
    axs[0, 0].set_xlabel("y")
    axs[0, 0].legend()

    axs[0, 1].plot(x, phw_m[iy0, :], lw=2, label="wrapped")
    axs[0, 1].plot(x, phu_m[iy0, :], lw=2, label="unwrapped")
    axs[0, 1].set_title(f"phase along x-axis (y={y[iy0]:.3f})")
    axs[0, 1].set_xlabel("x")
    axs[0, 1].legend()

    axs[1, 0].plot(y, amp_m[:, ix0], lw=2, label="y-axis")
    axs[1, 0].plot(x, amp_m[iy0, :], lw=2, label="x-axis")
    axs[1, 0].set_title("amplitude on coordinate axes")
    axs[1, 0].legend()

    axs[1, 1].plot(y, amp_cos_m[:, ix0], lw=2, label="y-axis")
    axs[1, 1].plot(x, amp_cos_m[iy0, :], lw=2, label="x-axis")
    axs[1, 1].set_title("amplitude * cos(unwrapped phase) on coordinate axes")
    axs[1, 1].legend()

    for ax in axs.flat:
        ax.grid(True, alpha=0.25)

    tag = "smoothed" if use_smoothed else "raw"
    frame_label = "initial" if fi == 0 else ("final" if fi == phase_unwrapped.shape[-1] - 1 else f"frame {fi}")
    mu = result.get("mu", None)
    mu_str = f"mu={mu:.4f}" if mu is not None else "mu=?"
    fig.suptitle(f"{mu_str} | {frame_label} | {tag} axis profiles")
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)