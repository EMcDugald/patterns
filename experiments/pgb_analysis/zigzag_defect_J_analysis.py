# experiments/pgb_analysis/zigzag_defect_J_analysis.py
"""
Final-time defect-centered J/k analysis for saved zigzag OP runs.

Adapts the older multi-dislocation J analysis into the new codebase, with:
  - file-or-directory input
  - per-file subdirectories for figures/data/logs
  - raw/sym/oriented k-field choice
  - twist_J peak detection at final time
  - representative V/X defect selection
  - radius sweeps of Q_J(r), circ_k(r), and boundary |k|-1 mismatch
  - core J metrics and simple time series around chosen defects
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from utils.kfield_calcs import (
    orient_vector_field,
    orient_vector_field_v2,
    phi_jump_mask,
    safe_central_derivs,
    compute_J_old_style,
    disk_area_integrals_safe,
    get_local_peaks,
    circle_line_integrals_safe,
)
from utils.geometry import build_rectangular_ramp_smooth


# -----------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------

def load_run_npz(path):
    return np.load(path, allow_pickle=True)


def _get_ramp(x, y, ramp_saved, xmargin, ymargin, tanhscale):
    if xmargin is not None and ymargin is not None and tanhscale is not None:
        return build_rectangular_ramp_smooth(
            x, y, xmargin=xmargin, ymargin=ymargin, tanhscale=tanhscale
        )
    if ramp_saved is not None:
        return ramp_saved
    raise ValueError("No ramp in OP file and no ramp params supplied.")


def _parse_mu(stem):
    try:
        part = stem.split("mu", 1)[1]
        for sep in ["_", "T", "t"]:
            if sep in part:
                return float(part.split(sep)[0])
        return float(part)
    except Exception:
        return np.nan


# -----------------------------------------------------------------------
# Masks + helpers
# -----------------------------------------------------------------------

def build_domain_masks_from_ramp(ramp, domain_thresh=0.995, core_thresh=0.999):
    rmin, rmax = np.nanmin(ramp), np.nanmax(ramp)
    ramp_n = (ramp - rmin) / (rmax - rmin + 1e-12)
    domain_mask = ramp_n >= domain_thresh
    inner_mask = ramp_n >= core_thresh
    return domain_mask, inner_mask, ramp_n


def choose_orientation_fields(op, it, domain_mask, orientation, orient_method, pi_tol):
    if "k1_orig" in op and "k2_orig" in op:
        f_raw = op["k1_orig"][..., it]
        g_raw = op["k2_orig"][..., it]
    else:
        f_raw = op["k1"][..., it]
        g_raw = op["k2"][..., it]

    f_sym = op["k1_sym"][..., it] if "k1_sym" in op else f_raw
    g_sym = op["k2_sym"][..., it] if "k2_sym" in op else g_raw

    if orientation == "raw":
        f_use, g_use = f_raw, g_raw
    elif orientation == "sym":
        f_use, g_use = f_sym, g_sym
    elif orientation == "oriented":
        if orient_method == "bfs2":
            f_or, g_or = orient_vector_field_v2(
                f_raw, g_raw, mask=domain_mask, pi_tol=pi_tol
            )
        else:
            f_or, g_or = orient_vector_field(f_raw, g_raw, mask=domain_mask)
        f_use = np.asarray(f_or)
        g_use = np.asarray(g_or)
    else:
        raise ValueError(f"Unknown orientation '{orientation}'")

    phi = np.arctan2(g_use, f_use)
    pj = phi_jump_mask(phi, tol=pi_tol)
    full_mask = domain_mask & (~pj) & np.isfinite(f_use) & np.isfinite(g_use)
    return f_use, g_use, phi, pj, full_mask


def radial_grid(X, Y, x_center, y_center):
    return np.sqrt((X - x_center) ** 2 + (Y - y_center) ** 2)


def compute_J_from_fg(f, g, dx, dy, full_mask):
    fx, fy = safe_central_derivs(f, dx, dy, full_mask)
    gx, gy = safe_central_derivs(g, dx, dy, full_mask)
    J = fx * gy - fy * gx
    return J, fx, fy, gx, gy


def boundary_k_stats(k_field, X, Y, x_center, y_center, radius, dx, dy,
                     mask_ok=None, n_theta=256):
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    x_c = x_center + radius * np.cos(theta)
    y_c = y_center + radius * np.sin(theta)

    j_c = (x_c - X[0, 0]) / dx
    i_c = (y_c - Y[0, 0]) / dy

    if mask_ok is not None:
        j_idx = np.round(j_c).astype(int)
        i_idx = np.round(i_c).astype(int)
        inside = (
            (i_idx >= 0) & (i_idx < k_field.shape[0]) &
            (j_idx >= 0) & (j_idx < k_field.shape[1])
        )
        bad = inside & (~mask_ok[i_idx, j_idx])
        if np.any(bad):
            return np.nan, np.nan

    coords = np.vstack([i_c, j_c])
    k_c = map_coordinates(k_field, coords, order=1, mode="nearest")
    diff = np.abs(k_c) - 1.0
    return np.nanmax(np.abs(diff)), np.nanstd(diff)


def k_on_circle(k_field, X, Y, x_center, y_center, radius, dx, dy, n_theta=256):
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    x_c = x_center + radius * np.cos(theta)
    y_c = y_center + radius * np.sin(theta)

    j_c = (x_c - X[0, 0]) / dx
    i_c = (y_c - Y[0, 0]) / dy
    coords = np.vstack([i_c, j_c])
    k_c = map_coordinates(k_field, coords, order=1, mode="nearest")
    return theta, np.abs(k_c)


def core_J_metrics(J_field, R_grid, R_small, domain_mask):
    core_mask = (R_grid <= R_small) & domain_mask
    J_core = J_field[core_mask]
    r_core = R_grid[core_mask]

    if J_core.size == 0 or np.all(~np.isfinite(J_core)):
        return np.nan, np.nan

    J_abs = np.abs(J_core)
    total_w = np.nansum(J_abs)
    if total_w == 0 or not np.isfinite(total_w):
        return np.nan, np.nan

    maxJ = np.nanmax(J_abs)
    spread = np.sqrt(np.nansum(J_abs * r_core**2) / total_w)
    return maxJ, spread


def choose_VX_from_twist_peaks(maxima_indices, minima_indices, x, y):
    if maxima_indices.size == 0 or minima_indices.size == 0:
        raise RuntimeError("Need at least one twist_J max and one min.")

    ys_max = maxima_indices[:, 0]
    xs_max = maxima_indices[:, 1]
    x_max = x[xs_max]
    order_max = np.argsort(x_max)

    ys_max = ys_max[order_max]
    xs_max = xs_max[order_max]
    x_max = x_max[order_max]

    ys_min = minima_indices[:, 0]
    xs_min = minima_indices[:, 1]
    x_min = x[xs_min]
    order_min = np.argsort(x_min)

    ys_min = ys_min[order_min]
    xs_min = xs_min[order_min]
    x_min = x_min[order_min]

    mid_idx = (len(xs_max) - 1) // 2
    ix_V = xs_max[mid_idx]
    iy_V = ys_max[mid_idx]
    x_V = x[ix_V]
    y_V = y[iy_V]

    mask_right = x_min > x_V
    if np.any(mask_right):
        idxs_right = np.where(mask_right)[0]
        j_min = idxs_right[np.argmin(x_min[idxs_right] - x_V)]
    else:
        j_min = int(np.argmin(np.abs(x_min - x_V)))

    ix_X = xs_min[j_min]
    iy_X = ys_min[j_min]
    x_X = x[ix_X]
    y_X = y[iy_X]

    return (ix_V, iy_V, x_V, y_V), (ix_X, iy_X, x_X, y_X)


# -----------------------------------------------------------------------
# Per-file driver
# -----------------------------------------------------------------------

def process_one(path, base_out,
                ramp_domain_thresh, ramp_core_thresh,
                xmargin, ymargin, tanhscale,
                orientation, orient_method, pi_tol,
                r_min, r_max, r_count, r_small, twist_radius,
                J_peak_min_distance, J_peak_threshold_rel,
                phase_frac_keep_x):
    path = Path(path)
    stem = path.stem
    print(f"  → {stem}")

    op = load_run_npz(path)
    x = op["x"]
    y = op["y"]
    u = op["u"]
    ramp_saved = op["ramp"] if "ramp" in op else None
    ramp = _get_ramp(x, y, ramp_saved, xmargin, ymargin, tanhscale)

    mu = _parse_mu(stem)
    X, Y = np.meshgrid(x, y)
    Ny, Nx, Nt = u.shape
    t = op["t"] if "t" in op else np.arange(Nt)
    t0 = Nt - 1
    dx = float(X[0, 1] - X[0, 0])
    dy = float(Y[1, 0] - Y[0, 0])
    extent = [x.min(), x.max(), y.min(), y.max()]
    R_LIST = np.linspace(r_min, r_max, r_count)

    domain_mask, inner_mask, ramp_n = build_domain_masks_from_ramp(
        ramp, domain_thresh=ramp_domain_thresh, core_thresh=ramp_core_thresh
    )

    out_dir = base_out / stem
    fig_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    log_dir = out_dir / "logs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # final-time orientation diagnostics
    f_raw, g_raw, phi_raw, pj_raw, full_mask_raw = choose_orientation_fields(
        op, t0, domain_mask, "raw", orient_method, pi_tol
    )
    f_sym, g_sym, phi_sym, pj_sym, full_mask_sym = choose_orientation_fields(
        op, t0, domain_mask, "sym", orient_method, pi_tol
    )
    f_or, g_or, phi_or, pj_or, full_mask_or = choose_orientation_fields(
        op, t0, domain_mask, "oriented", orient_method, pi_tol
    )

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, mask, title in zip(
        axs, [pj_raw, pj_sym, pj_or], ["raw", "sym", "oriented"]
    ):
        im = ax.imshow(
            np.ma.masked_where(~domain_mask, mask),
            extent=extent, origin="lower", cmap="gray"
        )
        ax.set_title(f"pi-jump mask ({title})")
        plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(fig_dir / "pi_jump_masks.png", dpi=200)
    plt.close(fig)

    f_use, g_use, phi_use, pj_use, full_mask_use = choose_orientation_fields(
        op, t0, domain_mask, orientation, orient_method, pi_tol
    )

    J0, _, _, _, _ = compute_J_from_fg(f_use, g_use, dx, dy, full_mask_use)
    J0 = np.where(full_mask_use, J0, np.nan)
    k0 = np.sqrt(f_use**2 + g_use**2)

    mask_centers_all = domain_mask.copy()
    twist_J = disk_area_integrals_safe(
        J0, X, Y,
        mask_centers=mask_centers_all,
        mask_ok=full_mask_use,
        radius=twist_radius,
    )

    valid_mask_twist = domain_mask & np.isfinite(twist_J)
    maxima_indices, minima_indices = get_local_peaks(
        twist_J,
        mask=valid_mask_twist,
        min_distance=J_peak_min_distance,
        threshold_rel=J_peak_threshold_rel,
    )

    if maxima_indices.size == 0 or minima_indices.size == 0:
        raise RuntimeError(f"No usable twist_J max/min found for {stem}")

    (ix_V, iy_V, x_V, y_V), (ix_X, iy_X, x_X, y_X) = choose_VX_from_twist_peaks(
        maxima_indices, minima_indices, x, y
    )

    max_x = x[maxima_indices[:, 1]]
    max_y = y[maxima_indices[:, 0]]
    min_x = x[minima_indices[:, 1]]
    min_y = y[minima_indices[:, 0]]

    # pattern + chosen points
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        np.ma.masked_where(~domain_mask, u[..., t0]),
        extent=extent, origin="lower", cmap="gray"
    )
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    ax.scatter(max_x, max_y, c="red", s=15, label="twist_J maxima")
    ax.scatter(min_x, min_y, c="blue", s=15, label="twist_J minima")
    ax.scatter([x_V], [y_V], s=80, facecolors="none", edgecolors="yellow",
               linewidths=2, label="chosen V")
    ax.scatter([x_X], [y_X], s=80, facecolors="none", edgecolors="cyan",
               linewidths=2, label="chosen X")
    ax.set_title(f"Pattern with twist_J peaks (mu={mu:.3f}, t=t_final)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "pattern_with_twistJ_peaks.png", dpi=200)
    plt.close(fig)

    # J field
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        np.ma.masked_where(~domain_mask, J0),
        extent=extent, origin="lower", cmap="seismic"
    )
    ax.set_title(f"J field at t_final (mu={mu:.3f})")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(fig_dir / "J_field_tfinal.png", dpi=200)
    plt.close(fig)

    # phase field
    if "phase_grid_symmetric_unwrapped" in op:
        phase_t0 = op["phase_grid_symmetric_unwrapped"][..., t0]
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(
            np.ma.masked_where(~domain_mask, phase_t0),
            extent=extent, origin="lower", cmap="twilight"
        )
        ax.set_title(f"phase_sym_unwrapped at t_final (mu={mu:.3f})")
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.savefig(fig_dir / "phase_sym_unwrapped_tfinal.png", dpi=200)
        plt.close(fig)

    defect_specs = [
        (ix_V, iy_V, x_V, y_V, "V_from_max"),
        (ix_X, iy_X, x_X, y_X, "X_from_min"),
    ]

    for ix0, iy0, x_c, y_c, defect_label in defect_specs:
        R_grid = radial_grid(X, Y, x_c, y_c)

        def_fig_dir = fig_dir / defect_label
        def_data_dir = data_dir / defect_label
        def_log_dir = log_dir / defect_label
        def_fig_dir.mkdir(parents=True, exist_ok=True)
        def_data_dir.mkdir(parents=True, exist_ok=True)
        def_log_dir.mkdir(parents=True, exist_ok=True)

        r_fig_dir = def_fig_dir / "r_sweeps"
        ktheta_dir = def_fig_dir / "k_on_circle"
        r_fig_dir.mkdir(parents=True, exist_ok=True)
        ktheta_dir.mkdir(parents=True, exist_ok=True)

        QJ_list = []
        dK_max_list = []
        dK_std_list = []
        circ_k_list = []

        mask_centers = np.zeros_like(domain_mask, dtype=bool)
        mask_centers[iy0, ix0] = True

        for r_val in R_LIST:
            J_disk = disk_area_integrals_safe(
                J0, X, Y,
                mask_centers=mask_centers,
                mask_ok=full_mask_use,
                radius=r_val,
            )
            QJ_list.append(J_disk[iy0, ix0])

            disk_mask = (R_grid <= r_val) & full_mask_use
            J_disk_view = np.ma.masked_where(~disk_mask, J0)

            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(
                J_disk_view,
                extent=extent, origin="lower", cmap="seismic"
            )
            ax.set_title(f"J over disk r={r_val:.3f} ({defect_label}, mu={mu:.3f})")
            plt.colorbar(im, ax=ax, shrink=0.8)
            plt.tight_layout()
            plt.savefig(r_fig_dir / f"J_disk_r{r_val:.3f}.png", dpi=200)
            plt.close(fig)

            dmax, dstd = boundary_k_stats(
                k0, X, Y, x_c, y_c, r_val, dx, dy,
                mask_ok=full_mask_use, n_theta=256
            )
            dK_max_list.append(dmax)
            dK_std_list.append(dstd)

            circ_k_field = circle_line_integrals_safe(
                f_use, g_use, X, Y,
                mask_centers=mask_centers,
                mask_ok=full_mask_use,
                radius=r_val,
                n_theta=256,
            )
            circ_k_list.append(circ_k_field[iy0, ix0])

            theta_s, kabs_s = k_on_circle(
                k0, X, Y, x_c, y_c, r_val, dx, dy, n_theta=256
            )
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(theta_s, kabs_s, "k-")
            ax.set_xlabel("theta")
            ax.set_ylabel("|k|(theta)")
            ax.set_title(f"|k| on circle r={r_val:.3f} ({defect_label}, mu={mu:.3f})")
            plt.tight_layout()
            plt.savefig(ktheta_dir / f"k_on_circle_r{r_val:.3f}.png", dpi=200)
            plt.close(fig)

        QJ_list = np.asarray(QJ_list)
        dK_max_list = np.asarray(dK_max_list)
        dK_std_list = np.asarray(dK_std_list)
        circ_k_list = np.asarray(circ_k_list)

        maxJ_core, spreadJ_core = core_J_metrics(J0, R_grid, r_small, domain_mask)

        np.savetxt(
            def_data_dir / "J_charge_and_boundary_stats.csv",
            np.column_stack([
                R_LIST,
                QJ_list,
                circ_k_list,
                dK_max_list,
                dK_std_list,
                np.full_like(R_LIST, maxJ_core),
                np.full_like(R_LIST, spreadJ_core),
            ]),
            delimiter=",",
            header="r,QJ,circ_k,delta_k_max,delta_k_std,maxJ_core_t0,spreadJ_core_t0",
            comments="",
        )

        fig, axs = plt.subplots(1, 3, figsize=(14, 4))

        axs[0].plot(R_LIST, QJ_list, "o-")
        axs[0].axhline(np.pi, color="k", linestyle="--", linewidth=0.8)
        axs[0].axhline(-np.pi, color="k", linestyle="--", linewidth=0.8)
        axs[0].set_xlabel("r")
        axs[0].set_ylabel("Q_J(r)")
        axs[0].set_title(f"J charge vs r ({defect_label}, mu={mu:.3f})")

        axs[1].plot(R_LIST, circ_k_list, "o-")
        axs[1].set_xlabel("r")
        axs[1].set_ylabel("circ_k(r)")
        axs[1].set_title(f"∮k·dl vs r ({defect_label}, mu={mu:.3f})")

        axs[2].plot(R_LIST, dK_max_list, "o-", label="max ||k|-1|")
        axs[2].plot(R_LIST, dK_std_list, "s--", label="std(|k|-1)")
        axs[2].set_xlabel("r")
        axs[2].set_ylabel("boundary mismatch")
        axs[2].set_title(f"|k| boundary stats vs r ({defect_label}, mu={mu:.3f})")
        axs[2].legend()

        plt.tight_layout()
        plt.savefig(def_fig_dir / "QJ_circK_and_boundary_k_vs_r.png", dpi=200)
        plt.close(fig)

        with open(def_log_dir / "analyze_single_defect_log.txt", "w") as f_log:
            f_log.write(f"OP path: {path}\n")
            f_log.write(f"mu: {mu}\n")
            f_log.write(f"defect_label: {defect_label}\n")
            f_log.write(f"orientation: {orientation}\n")
            f_log.write(f"orient_method: {orient_method}\n")
            f_log.write(
                f"defect_center_from_twistJ_peak_grid: "
                f"(ix={ix0}, iy={iy0}), x_center={x_c:.6g}, y_center={y_c:.6g}\n"
            )
            f_log.write(f"R_LIST: {R_LIST}\n")
            f_log.write(f"r_small: {r_small}\n")
            f_log.write(f"maxJ_core_t0: {maxJ_core}\n")
            f_log.write(f"spreadJ_core_t0: {spreadJ_core}\n")

    # final-time core profile
    j_core = Ny // 2
    core_mask_x = ramp_n[j_core, :] >= ramp_core_thresh
    x_core = x[core_mask_x]
    J_core_profile = J0[j_core, core_mask_x]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x_core, J_core_profile, "k-")
    ax.set_xlabel("x")
    ax.set_ylabel("J(x, y_core)")
    ax.set_title(f"J along PGB core (mu={mu:.3f}, t=t_final)")
    plt.tight_layout()
    plt.savefig(fig_dir / "J_core_profile.png", dpi=200)
    plt.close(fig)

    if "phase_grid_symmetric_unwrapped" in op:
        phase_sym_unwrapped = op["phase_grid_symmetric_unwrapped"][..., t0]
        valid_phase_mask = (ramp_n[j_core, :] >= ramp_core_thresh) & np.isfinite(
            phase_sym_unwrapped[j_core, :]
        )
        valid_indices = np.where(valid_phase_mask)[0]
        if valid_indices.size > 0:
            i_min = valid_indices[0]
            i_max = valid_indices[-1]
            keep_len = max(1, int(np.floor(phase_frac_keep_x * (i_max - i_min + 1))))
            core_slice = slice(i_min, i_min + keep_len)

            x_phase = x[core_slice]
            J_phase = J0[j_core, core_slice]
            phi_phase = phase_sym_unwrapped[j_core, core_slice]

            fig, ax1 = plt.subplots(figsize=(6, 3))
            ax1.plot(x_phase, J_phase, color="tab:blue")
            ax1.set_xlabel("x")
            ax1.set_ylabel("J", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")

            ax2 = ax1.twinx()
            ax2.plot(x_phase, phi_phase, color="tab:red", linestyle="--")
            ax2.set_ylabel("phase (symmetric unwrapped)", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")

            fig.suptitle(f"J and phase along PGB core (mu={mu:.3f}, t=t_final)")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(fig_dir / "J_and_phase_core_profile_left.png", dpi=200)
            plt.close(fig)

    # time-dependent J metrics around chosen V defect
    R_grid_V = radial_grid(X, Y, x_V, y_V)
    maxJ_t_list = []
    spreadJ_t_list = []
    QJ_global_list = []

    for it in range(Nt):
        f_t, g_t, _, _, full_mask_t = choose_orientation_fields(
            op, it, domain_mask, orientation, orient_method, pi_tol
        )
        J_t, *_ = compute_J_from_fg(f_t, g_t, dx, dy, full_mask_t)
        J_t = np.where(full_mask_t, J_t, np.nan)

        maxJ_t, spreadJ_t = core_J_metrics(J_t, R_grid_V, r_small, domain_mask)
        maxJ_t_list.append(maxJ_t)
        spreadJ_t_list.append(spreadJ_t)

        QJ_global = np.nansum(np.where(full_mask_t, J_t, np.nan)) * dx * dy
        QJ_global_list.append(QJ_global)

    maxJ_t_list = np.asarray(maxJ_t_list)
    spreadJ_t_list = np.asarray(spreadJ_t_list)
    QJ_global_list = np.asarray(QJ_global_list)

    np.savetxt(
        data_dir / "J_core_time_series.csv",
        np.column_stack([t, maxJ_t_list, spreadJ_t_list, QJ_global_list]),
        delimiter=",",
        header="t,maxJ_core,spreadJ_core,QJ_global",
        comments="",
    )

    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    axs[0].plot(t, maxJ_t_list, "o-")
    axs[0].set_ylabel("max |J| (core)")
    axs[0].set_title(f"Core J + global ∫J dxdy (mu={mu:.3f})")

    axs[1].plot(t, spreadJ_t_list, "o-")
    axs[1].set_ylabel("spread(J) (core)")

    axs[2].plot(t, QJ_global_list, "o-")
    axs[2].set_xlabel("t")
    axs[2].set_ylabel("∫J dxdy")

    plt.tight_layout()
    plt.savefig(fig_dir / "J_core_and_global_time_series.png", dpi=200)
    plt.close(fig)

    with open(log_dir / "run_summary_log.txt", "w") as f_log:
        f_log.write(f"OP path: {path}\n")
        f_log.write(f"mu: {mu}\n")
        f_log.write(f"orientation: {orientation}\n")
        f_log.write(f"orient_method: {orient_method}\n")
        f_log.write(f"twist_radius: {twist_radius}\n")
        f_log.write(f"J_peak_min_distance: {J_peak_min_distance}\n")
        f_log.write(f"J_peak_threshold_rel: {J_peak_threshold_rel}\n")
        f_log.write(f"chosen_V: (ix={ix_V}, iy={iy_V}, x={x_V:.6g}, y={y_V:.6g})\n")
        f_log.write(f"chosen_X: (ix={ix_X}, iy={iy_X}, x={x_X:.6g}, y={y_X:.6g})\n")

    print(f"    done mu={mu:.3f}")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--op_file", type=str, default=None,
                        help="Single OP .npz file.")
    parser.add_argument("--op_dir", type=str, default=None,
                        help="Directory of OP .npz files.")
    parser.add_argument("--pattern", type=str, default="*.npz",
                        help="Glob pattern inside op_dir.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Root output directory.")
    parser.add_argument("--ramp_domain_thresh", type=float, default=0.995)
    parser.add_argument("--ramp_core_thresh", type=float, default=0.999)
    parser.add_argument("--xmargin", type=float, default=None)
    parser.add_argument("--ymargin", type=float, default=None)
    parser.add_argument("--tanhscale", type=float, default=None)
    parser.add_argument("--orientation", type=str, default="raw",
                        choices=["raw", "sym", "oriented"])
    parser.add_argument("--orient_method", type=str, default="bfs",
                        choices=["bfs", "bfs2"])
    parser.add_argument("--pi_tol", type=float, default=np.pi / 10)
    parser.add_argument("--r_min", type=float, default=0.125)
    parser.add_argument("--r_max", type=float, default=6.5)
    parser.add_argument("--r_count", type=int, default=52)
    parser.add_argument("--r_small", type=float, default=0.5)
    parser.add_argument("--twist_radius", type=float, default=0.5)
    parser.add_argument("--J_peak_min_distance", type=int, default=3)
    parser.add_argument("--J_peak_threshold_rel", type=float, default=0.10)
    parser.add_argument("--phase_frac_keep_x", type=float, default=0.5)
    parser.add_argument("--mu_min", type=float, default=None)

    ns = parser.parse_args(args)

    if ns.out_dir is not None:
        base_out = Path(ns.out_dir)
    else:
        base_out = _HERE / "results" / "zigzag_defect_J_analysis"
    base_out.mkdir(parents=True, exist_ok=True)

    common = dict(
        ramp_domain_thresh=ns.ramp_domain_thresh,
        ramp_core_thresh=ns.ramp_core_thresh,
        xmargin=ns.xmargin,
        ymargin=ns.ymargin,
        tanhscale=ns.tanhscale,
        orientation=ns.orientation,
        orient_method=ns.orient_method,
        pi_tol=ns.pi_tol,
        r_min=ns.r_min,
        r_max=ns.r_max,
        r_count=ns.r_count,
        r_small=ns.r_small,
        twist_radius=ns.twist_radius,
        J_peak_min_distance=ns.J_peak_min_distance,
        J_peak_threshold_rel=ns.J_peak_threshold_rel,
        phase_frac_keep_x=ns.phase_frac_keep_x,
    )

    if ns.op_file:
        process_one(Path(ns.op_file), base_out, **common)
    else:
        op_dir = Path(ns.op_dir or ".")
        files = sorted(op_dir.glob(ns.pattern))
        if not files:
            raise SystemExit(f"No files matching '{ns.pattern}' in {op_dir}")
        if ns.mu_min is not None:
            files = [f for f in files if _parse_mu(f.stem) > ns.mu_min]
        print(f"Processing {len(files)} file(s) from {op_dir}")
        for f in files:
            process_one(f, base_out, **common)

    print("Done.")


# -----------------------------------------------------------------------
# Debug block
# -----------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) == 1:

        class _Args:
            op_file = None
            op_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/uhu/mu_sweeps_full_Nx512_hp025_T3p125_NyF5_np18_Nsave125/sig_pio2/raw"
            pattern = "*.npz"
            out_dir = "/Users/edwardmcdugald/patterns/experiments/pgb_analysis/results/zigzag_defect_J_analysis/mu_sweeps_full_Nx512_hp025_T3p125_NyF5_np18_Nsave125/sig_pio2"
            ramp_domain_thresh = 0.995
            ramp_core_thresh = 0.999
            xmargin = None
            ymargin = None
            tanhscale = None
            orientation = "raw"
            orient_method = "bfs"
            pi_tol = np.pi / 10
            r_min = 0.125
            r_max = 6.5
            r_count = 52
            r_small = 0.5
            twist_radius = 0.5
            J_peak_min_distance = 3
            J_peak_threshold_rel = 0.10
            phase_frac_keep_x = 0.5
            mu_min = None

        a = _Args()
        main([
            "--op_dir", a.op_dir,
            "--pattern", a.pattern,
            *(["--out_dir", a.out_dir] if a.out_dir is not None else []),
            "--ramp_domain_thresh", str(a.ramp_domain_thresh),
            "--ramp_core_thresh", str(a.ramp_core_thresh),
            "--orientation", a.orientation,
            "--orient_method", a.orient_method,
            "--pi_tol", str(a.pi_tol),
            "--r_min", str(a.r_min),
            "--r_max", str(a.r_max),
            "--r_count", str(a.r_count),
            "--r_small", str(a.r_small),
            "--twist_radius", str(a.twist_radius),
            "--J_peak_min_distance", str(a.J_peak_min_distance),
            "--J_peak_threshold_rel", str(a.J_peak_threshold_rel),
            "--phase_frac_keep_x", str(a.phase_frac_keep_x),
            *(["--xmargin", str(a.xmargin),
               "--ymargin", str(a.ymargin),
               "--tanhscale", str(a.tanhscale)]
              if a.xmargin is not None else []),
            *(["--mu_min", str(a.mu_min)] if a.mu_min is not None else []),
        ])
    else:
        main()