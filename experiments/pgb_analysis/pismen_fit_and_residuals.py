from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy import special
from scipy.ndimage import map_coordinates, binary_erosion
from skimage.feature import peak_local_max


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
OP_PATH = Path(
    "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/phase/mu_sweeps_full_Nx512_hp2_T25_NyF5_np18_Nsave125/sig_pio2/raw/sh_pgb_zigzag_mu0.950_T25_N125_nx512_Ny256_lower_uhu_sigma1.571_xm0.03_ym0.03_ts120.0_phase_xg0.10_yg0.10_ns128_shlower_ds0.125_prmrebuild_prc0.100_prs1.00_prt0.050.npz"
)


MU = 0.95
TWIST_RADIUS = np.pi / 2.0       # disk radius for twist integrals
PEAK_MIN_DISTANCE = 5            # in grid points
PEAK_THRESHOLD_REL = 0.15        # relative threshold for twist peaks
MIDLINE_BAND_FRACTION = 0.15     # fraction of vertical span kept around midline

INCLUDE_VIRTUAL_DEFECTS = True   # master switch
N_VIRTUAL_PER_SIDE = 20          # number of extra spacings left/right



# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def orient_vector_field(f, g, mask):
    ny, nx = f.shape
    kx = np.zeros_like(f)
    ky = np.zeros_like(g)
    visited = np.zeros_like(mask, dtype=bool)

    start = np.argwhere(mask)
    if start.size == 0:
        return np.ma.masked_where(~mask, kx), np.ma.masked_where(~mask, ky)

    iy0, ix0 = start[0]
    kx[iy0, ix0] = f[iy0, ix0]
    ky[iy0, ix0] = g[iy0, ix0]
    visited[iy0, ix0] = True

    from collections import deque
    queue = deque()
    queue.append((iy0, ix0))

    offsets = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    while queue:
        iy, ix = queue.popleft()
        for dy, dx in offsets:
            nyi, nxi = iy + dy, ix + dx
            if (
                0 <= nyi < ny
                and 0 <= nxi < nx
                and mask[nyi, nxi]
                and not visited[nyi, nxi]
            ):
                k_curr = np.array([kx[iy, ix], ky[iy, ix]])
                k_cand = np.array([f[nyi, nxi], g[nyi, nxi]])
                if np.dot(k_curr, k_cand) < 0:
                    k_cand = -k_cand
                kx[nyi, nxi], ky[nyi, nxi] = k_cand
                visited[nyi, nxi] = True
                queue.append((nyi, nxi))

    kx = np.ma.masked_where(~mask, kx)
    ky = np.ma.masked_where(~mask, ky)
    return kx, ky


def phi_jump_mask(phi, tol=np.pi / 10):
    mask_jump = np.zeros_like(phi, dtype=bool)
    offsets = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    for dy, dx in offsets:
        phi_shift = np.roll(np.roll(phi, dy, axis=0), dx, axis=1)
        dphi = phi_shift - phi
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        mask_jump |= (np.abs(dphi) > (np.pi - tol))

    return mask_jump


def safe_central_derivs(s, dx, dy, mask_ok):
    """
    Central differences on plain ndarray s, using mask_ok to skip bad points.
    If s is masked, pass np.asarray(s) first.
    """
    s = np.asarray(s)
    ny, nx = s.shape
    sx = np.full_like(s, np.nan, dtype=float)
    sy = np.full_like(s, np.nan, dtype=float)

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            if not mask_ok[j, i]:
                continue
            if mask_ok[j, i - 1] and mask_ok[j, i + 1]:
                sx[j, i] = (s[j, i + 1] - s[j, i - 1]) / (2.0 * dx)
            if mask_ok[j - 1, i] and mask_ok[j + 1, i]:
                sy[j, i] = (s[j + 1, i] - s[j - 1, i]) / (2.0 * dy)

    return sx, sy


def disk_area_integrals_safe(s, X, Y, mask_centers, mask_ok,
                             radius, n_r=32, n_theta=64):
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    r = np.linspace(0, radius, n_r, endpoint=True)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    dr = r[1] - r[0] if n_r > 1 else radius
    dtheta = 2 * np.pi / n_theta

    A = np.full_like(s, np.nan, dtype=float)
    ys, xs = np.where(mask_centers)

    for iy, ix in zip(ys, xs):
        x0 = X[iy, ix]
        y0 = Y[iy, ix]

        rr, tt = np.meshgrid(r, theta, indexing="ij")
        x_d = x0 + rr * np.cos(tt)
        y_d = y0 + rr * np.sin(tt)

        j_d = (x_d - X[0, 0]) / dx
        i_d = (y_d - Y[0, 0]) / dy

        j_idx = np.round(j_d).astype(int)
        i_idx = np.round(i_d).astype(int)

        inside = (
            (i_idx >= 0) & (i_idx < s.shape[0]) &
            (j_idx >= 0) & (j_idx < s.shape[1])
        )

        bad = inside & (~mask_ok[i_idx, j_idx])
        if np.any(bad):
            continue

        coords = np.vstack([i_d.ravel(), j_d.ravel()])
        s_d = map_coordinates(s, coords, order=1, mode="nearest").reshape(rr.shape)
        if np.isnan(s_d).any():
            continue

        integrand = s_d * rr
        A[iy, ix] = integrand.sum() * dr * dtheta

    return A


def get_local_peaks(arr, mask, min_distance=3, threshold_rel=0.75):
    valid_mask = mask & np.isfinite(arr)
    arr_masked = np.ma.masked_where(~valid_mask, arr)
    arr_for_peaks = arr_masked.filled(-np.inf)

    maxima_indices = peak_local_max(
        arr_for_peaks,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        exclude_border=False,
        num_peaks=np.inf,
    )

    return maxima_indices


def pismen_on_sh_grid(X, Y, x_defects, amp=1.0):
    """
    Pismen multi-dislocation phase on the (possibly centred) SH zipper grid.

    Defects live on the x-axis (y=0) at x = x_defects.
    This is just the 'right-half' construction: no x->-x mirroring
    on this grid — the domain itself plays the role of the right half.
    """
    linear_phase = Y
    #linear_phase = np.sqrt(1-MU**2)*X + MU*Y

    x_defects = np.asarray(x_defects, dtype=float)
    npts = len(x_defects)
    if npts == 0:
        theta = linear_phase.copy()
        pattern = amp * np.cos(theta)
        return theta, pattern

    thetas = np.arange(npts + 1) * np.pi
    psi = 0.5 * (1 + np.exp(-thetas[-1]))

    for j in range(npts):
        delta = 0.5 * (np.exp(-thetas[j + 1]) - np.exp(-thetas[j]))
        arg = (X - x_defects[j]) / np.sqrt(2 * np.abs(Y) + 1e-8)
        psi += delta * special.erf(arg)

    theta = linear_phase - np.sign(Y) * np.log(psi)
    pattern = amp * np.cos(theta)
    return theta, pattern


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def compare_zigzag_pismen():
    proj_root = Path(__file__).resolve().parents[2]
    fig_root = proj_root / "experiments" / "pgb_analysis" / "results" / "pismen_fit_and_residuals"
    fig_root.mkdir(parents=True, exist_ok=True)

    # ---- load data and build masks ----
    op = np.load(OP_PATH)
    x = op["x"]
    y = op["y"]

    # recenter to symmetric box about (0,0)
    Lx = x.max() - x.min()
    Ly = y.max() - y.min()
    x0 = 0.5 * (x.min() + x.max())
    y0 = 0.5 * (y.min() + y.max())

    x_c = x - x0          # ≈ [-Lx/2, Lx/2]
    y_c = y - y0          # ≈ [-Ly/2, Ly/2]

    Xc, Yc = np.meshgrid(x_c, y_c)
    dx = x_c[1] - x_c[0]
    dy = y_c[1] - y_c[0]
    extent_c = [x_c.min(), x_c.max(), y_c.min(), y_c.max()]

    u = op["u"][..., -1]
    ramp_raw = op["ramp"]

    rmin = np.nanmin(ramp_raw)
    rmax = np.nanmax(ramp_raw)
    ramp = (ramp_raw - rmin) / (rmax - rmin + 1e-12)

    domain_mask = ramp >= 0.995
    inner_mask = ramp >= 0.999

    structure = np.ones((3, 3), dtype=bool)
    domain_mask = binary_erosion(domain_mask, structure=structure, iterations=10)
    inner_mask = binary_erosion(inner_mask, structure=structure, iterations=10)
    domain_mask = inner_mask

    def phi_jump_mask_inner(phi, tol=np.pi / 10, inner_mask=None):
        if inner_mask is None:
            inner_mask = np.ones_like(phi, dtype=bool)

        mask_jump = np.zeros_like(phi, dtype=bool)
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dy0, dx0 in offsets:
            phi_shift = np.roll(np.roll(phi, dy0, axis=0), dx0, axis=1)
            dphi = phi_shift - phi
            dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
            local = inner_mask & np.roll(inner_mask, dy0, axis=0)
            local &= np.roll(inner_mask, dx0, axis=1)
            mask_jump |= local & (np.abs(dphi) > (np.pi - tol))
        return mask_jump

    # ---- oriented k and diagnostics ----
    f_raw = op["k1_orig"][..., -1]
    g_raw = op["k2_orig"][..., -1]

    f_or_m, g_or_m = orient_vector_field(f_raw, g_raw, domain_mask)
    f_or = np.asarray(f_or_m)
    g_or = np.asarray(g_or_m)

    k_or = np.sqrt(f_or**2 + g_or**2)
    phi_or = np.arctan2(g_or, f_or)

    pi_mask_or = phi_jump_mask_inner(phi_or, tol=np.pi/10, inner_mask=inner_mask)
    mask_ok_or = domain_mask & (~pi_mask_or)

    # standalone π-jump mask plot
    plt.figure(figsize=(6, 5))
    plt.imshow(pi_mask_or, extent=extent_c, origin="lower", cmap="gray")
    plt.title("π-jump mask (oriented k)")
    plt.tight_layout()
    plt.savefig(fig_root / "pi_jump_mask_oriented.png", dpi=300)
    plt.close()

    fx_or, fy_or = safe_central_derivs(f_or, dx, dy, mask_ok_or)
    gx_or, gy_or = safe_central_derivs(g_or, dx, dy, mask_ok_or)
    J_or = fx_or * gy_or - fy_or * gx_or

    # ---- twist integrals and defect x-locations ----
    twist_or = disk_area_integrals_safe(
        J_or, Xc, Yc, domain_mask, mask_ok_or, radius=TWIST_RADIUS
    )

    peaks = get_local_peaks(
        twist_or,
        mask=mask_ok_or,
        min_distance=PEAK_MIN_DISTANCE,
        threshold_rel=PEAK_THRESHOLD_REL,
    )

    x_peaks = Xc[peaks[:, 0], peaks[:, 1]]
    y_peaks = Yc[peaks[:, 0], peaks[:, 1]]

    y_mid = 0.0
    y_span = y_c.max() - y_c.min()
    band_halfwidth = 0.5 * MIDLINE_BAND_FRACTION * y_span
    band_mask = np.abs(y_peaks - y_mid) <= band_halfwidth

    x_def = x_peaks[band_mask]
    y_def = y_peaks[band_mask]

    if x_def.size == 0:
        print("No twist peaks in midline band; using all peaks.")
        x_def = x_peaks
        y_def = y_peaks

    idx_sort = np.argsort(x_def)
    x_def_sorted = x_def[idx_sort]
    y_def_sorted = y_def[idx_sort]

    alpha = np.arcsin(MU)
    c = np.cos(alpha)
    d_theory = np.pi / c

    # ---- optional padding by virtual defects ----
    x_min_dom = x_c.min()
    x_max_dom = x_c.max()
    padded = list(x_def_sorted)

    if INCLUDE_VIRTUAL_DEFECTS and len(padded) > 0:
        # left side: add up to N_VIRTUAL_PER_SIDE spacings
        for _ in range(N_VIRTUAL_PER_SIDE):
            left_candidate = padded[0] - d_theory
            if left_candidate < x_min_dom:
                break
            padded.insert(0, left_candidate)

        # right side: add up to N_VIRTUAL_PER_SIDE spacings
        for _ in range(N_VIRTUAL_PER_SIDE):
            right_candidate = padded[-1] + d_theory
            if right_candidate > x_max_dom:
                break
            padded.append(right_candidate)

    x_def_padded = np.array(padded, dtype=float)
    x_def_padded.sort()


    if x_def_padded.size >= 2:
        dx_def = np.diff(x_def_padded)
        print("Defect x-locations (padded):", x_def_padded)
        print("Measured spacings:", dx_def)
        print("Mean spacing:", dx_def.mean())
        print("Theoretical spacing π/cos α =", d_theory)
    else:
        print("Only one (or zero) defect after padding; spacing not available.")

    # twist peaks visualization
    plt.figure(figsize=(6, 5))
    plt.imshow(u, extent=extent_c, origin="lower", cmap="gray")
    plt.scatter(x_peaks, y_peaks, c="red", s=10, label="twist peaks")
    plt.axhline(y_mid, color="cyan", linestyle="--", linewidth=0.8,
                label="midline band center")
    for xd in x_def_padded:
        plt.axvline(xd, color="yellow", linestyle=":", linewidth=0.8)
    plt.title("Twist peaks (oriented k) with padded defect lines")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_root / "twist_peaks_oriented.png", dpi=300)
    plt.close()

    # ---- Pismen phase on centred SH grid ----
    theta_p, pattern_p = pismen_on_sh_grid(Xc, Yc, x_def_padded, amp=1.0)

    plt.figure(figsize=(6, 5))
    plt.imshow(pattern_p, extent=extent_c, origin="lower", cmap="gray")
    for xd in x_def_padded:
        plt.axvline(xd, color="yellow", linestyle=":", linewidth=0.8)
    plt.title(f"Pismen pattern on centred SH grid (N={len(x_def_padded)} defects)")
    plt.tight_layout()
    plt.savefig(fig_root / "pismen_pattern_on_sh_grid.png", dpi=300)
    plt.close()

    # ---- θ_x, θ_y vs k1, k2 comparisons ----
    theta_px, theta_py = safe_central_derivs(theta_p, dx, dy, domain_mask)
    k1_pism = theta_px
    k2_pism = theta_py

    # k1 vs θ_x + absolute error
    v1 = np.nanmax(np.abs(k1_pism[domain_mask]))
    v2 = np.nanmax(np.abs(f_or[domain_mask]))
    v = max(v1, v2)
    abs_err_k1 = np.abs(f_or - k1_pism)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axs[0].imshow(
        np.ma.masked_where(~domain_mask, f_or),
        extent=extent_c, origin="lower",
        cmap="bwr", vmin=-v, vmax=v
    )
    axs[0].set_title("k1 (oriented SH)")
    plt.colorbar(im0, ax=axs[0], shrink=0.7, pad=0.02)

    im1 = axs[1].imshow(
        np.ma.masked_where(~domain_mask, k1_pism),
        extent=extent_c, origin="lower",
        cmap="bwr", vmin=-v, vmax=v
    )
    axs[1].set_title("θ_x (Pismen)")
    plt.colorbar(im1, ax=axs[1], shrink=0.7, pad=0.02)

    im2 = axs[2].imshow(
        np.ma.masked_where(~domain_mask, abs_err_k1),
        extent=extent_c, origin="lower",
        cmap="viridis"
    )
    axs[2].set_title("|k1 − θ_x|")
    plt.colorbar(im2, ax=axs[2], shrink=0.7, pad=0.02)

    plt.tight_layout()
    plt.savefig(fig_root / "k1_vs_theta_x.png", dpi=300)
    plt.close()

    # k2 vs θ_y + absolute error
    v1 = np.nanmax(np.abs(k2_pism[domain_mask]))
    v2 = np.nanmax(np.abs(g_or[domain_mask]))
    v = max(v1, v2)
    abs_err_k2 = np.abs(g_or - k2_pism)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axs[0].imshow(
        np.ma.masked_where(~domain_mask, g_or),
        extent=extent_c, origin="lower",
        cmap="bwr", vmin=-v, vmax=v
    )
    axs[0].set_title("k2 (oriented SH)")
    plt.colorbar(im0, ax=axs[0], shrink=0.7, pad=0.02)

    im1 = axs[1].imshow(
        np.ma.masked_where(~domain_mask, k2_pism),
        extent=extent_c, origin="lower",
        cmap="bwr", vmin=-v, vmax=v
    )
    axs[1].set_title("θ_y (Pismen)")
    plt.colorbar(im1, ax=axs[1], shrink=0.7, pad=0.02)

    im2 = axs[2].imshow(
        np.ma.masked_where(~domain_mask, abs_err_k2),
        extent=extent_c, origin="lower",
        cmap="viridis"
    )
    axs[2].set_title("|k2 − θ_y|")
    plt.colorbar(im2, ax=axs[2], shrink=0.7, pad=0.02)

    plt.tight_layout()
    plt.savefig(fig_root / "k2_vs_theta_y.png", dpi=300)
    plt.close()

    # ---- side-by-side SH vs Pismen patterns ----
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axs[0].imshow(
        u,
        extent=extent_c,
        origin="lower",
        cmap="gray"
    )
    axs[0].set_title("SH zipper pattern")
    plt.colorbar(im0, ax=axs[0], shrink=0.7, pad=0.02)

    im1 = axs[1].imshow(
        pattern_p,
        extent=extent_c,
        origin="lower",
        cmap="gray"
    )
    axs[1].set_title(f"Pismen pattern (N={len(x_def_padded)} defects)")
    plt.colorbar(im1, ax=axs[1], shrink=0.7, pad=0.02)

    for ax in axs:
        for xd in x_def_padded:
            ax.axvline(xd, color="yellow", linestyle=":", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(fig_root / "sh_vs_pismen_patterns.png", dpi=300)
    plt.close()


    print(f"Saved figures to {fig_root}")


if __name__ == "__main__":
    compare_zigzag_pismen()
