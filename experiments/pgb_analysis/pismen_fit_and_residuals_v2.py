"""
pismen_fit_and_residuals_v2.py
------------------------------
Same computation as pismen_fit_and_residuals.py (twist-peak defect location,
virtual-defect padding, Pismen multi-dislocation phase comparison), with a
publication-quality plotting layer and two additions:

  * pismen_spacing.png : measured defect spacings vs the L = pi/cos(alpha)
    law (previously only printed to stdout).
  * summary.json       : RMS errors and spacing statistics for quoting.

Figures produced:
  pismen_pattern_and_peaks.png   (a) SH u + twist peaks + defect lines
                                 (b) Pismen pattern + defect lines
  pismen_k1_comparison.png       k1 | theta_x | error  (shared limits)
  pismen_k2_comparison.png       k2 | theta_y | error
  pismen_spacing.png             per-gap spacing vs pi/cos(alpha)
"""

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

from scipy import special
from scipy.ndimage import map_coordinates, binary_erosion
from skimage.feature import peak_local_max

from paper_style import (use_paper_style, panel_label, add_cbar,
                         comparison_triptych, pattern_pair, masked)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
OP_PATH = Path(
    "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/phase/"
    "mu_sweeps_full_Nx512_hp2_T25_NyF5_np18_Nsave125/sig_pio2/raw/"
    "sh_pgb_zigzag_mu0.950_T25_N125_nx512_Ny256_lower_uhu_sigma1.571_"
    "xm0.03_ym0.03_ts120.0_phase_xg0.10_yg0.10_ns128_shlower_ds0.125_"
    "prmrebuild_prc0.100_prs1.00_prt0.050.npz"
)

MU = 0.95
TWIST_RADIUS = np.pi / 2.0
PEAK_MIN_DISTANCE = 5
PEAK_THRESHOLD_REL = 0.15
MIDLINE_BAND_FRACTION = 0.15
INCLUDE_VIRTUAL_DEFECTS = True
N_VIRTUAL_PER_SIDE = 20
OUT_ROOT = None   # default: ./results/pismen_fit_and_residuals_v2/<stem>


# ---------------------------------------------------------------------
# Helpers (verbatim from v1)
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
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while queue:
        iy, ix = queue.popleft()
        for dy, dx in offsets:
            nyi, nxi = iy + dy, ix + dx
            if (0 <= nyi < ny and 0 <= nxi < nx
                    and mask[nyi, nxi] and not visited[nyi, nxi]):
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


def safe_central_derivs(s, dx, dy, mask_ok):
    """Vectorized central differences; NaN where any needed neighbor bad."""
    s = np.asarray(s, dtype=float)
    sx = np.full_like(s, np.nan)
    sy = np.full_like(s, np.nan)

    ok_x = mask_ok.copy()
    ok_x[:, 1:-1] &= mask_ok[:, :-2] & mask_ok[:, 2:]
    ok_x[:, [0, -1]] = False
    ok_y = mask_ok.copy()
    ok_y[1:-1, :] &= mask_ok[:-2, :] & mask_ok[2:, :]
    ok_y[[0, -1], :] = False

    gx = np.empty_like(s)
    gx[:, 1:-1] = (s[:, 2:] - s[:, :-2]) / (2.0 * dx)
    gx[:, [0, -1]] = np.nan
    gy = np.empty_like(s)
    gy[1:-1, :] = (s[2:, :] - s[:-2, :]) / (2.0 * dy)
    gy[[0, -1], :] = np.nan

    sx[ok_x] = gx[ok_x]
    sy[ok_y] = gy[ok_y]
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
        inside = ((i_idx >= 0) & (i_idx < s.shape[0]) &
                  (j_idx >= 0) & (j_idx < s.shape[1]))
        bad = inside & (~mask_ok[i_idx, j_idx])
        if np.any(bad):
            continue
        coords = np.vstack([i_d.ravel(), j_d.ravel()])
        s_d = map_coordinates(s, coords, order=1,
                              mode="nearest").reshape(rr.shape)
        if np.isnan(s_d).any():
            continue
        A[iy, ix] = (s_d * rr).sum() * dr * dtheta
    return A


def get_local_peaks(arr, mask, min_distance=3, threshold_rel=0.75):
    valid_mask = mask & np.isfinite(arr)
    arr_for_peaks = np.ma.masked_where(~valid_mask, arr).filled(-np.inf)
    return peak_local_max(arr_for_peaks, min_distance=min_distance,
                          threshold_rel=threshold_rel,
                          exclude_border=False, num_peaks=np.inf)


def pismen_on_sh_grid(X, Y, x_defects, amp=1.0):
    linear_phase = Y
    x_defects = np.asarray(x_defects, dtype=float)
    npts = len(x_defects)
    if npts == 0:
        theta = linear_phase.copy()
        return theta, amp * np.cos(theta)

    thetas = np.arange(npts + 1) * np.pi
    psi = 0.5 * (1 + np.exp(-thetas[-1]))
    for j in range(npts):
        delta = 0.5 * (np.exp(-thetas[j + 1]) - np.exp(-thetas[j]))
        arg = (X - x_defects[j]) / np.sqrt(2 * np.abs(Y) + 1e-8)
        psi += delta * special.erf(arg)
    theta = linear_phase - np.sign(Y) * np.log(psi)
    return theta, amp * np.cos(theta)


def phi_jump_mask_inner(phi, tol=np.pi / 10, inner_mask=None):
    if inner_mask is None:
        inner_mask = np.ones_like(phi, dtype=bool)
    mask_jump = np.zeros_like(phi, dtype=bool)
    for dy0, dx0 in [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        phi_shift = np.roll(np.roll(phi, dy0, axis=0), dx0, axis=1)
        dphi = phi_shift - phi
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        local = inner_mask & np.roll(inner_mask, dy0, axis=0)
        local &= np.roll(inner_mask, dx0, axis=1)
        mask_jump |= local & (np.abs(dphi) > (np.pi - tol))
    return mask_jump


# ---------------------------------------------------------------------
# New presentation figures
# ---------------------------------------------------------------------

def fig_pattern_and_peaks(out, u, pattern_p, extent, x_peaks, y_peaks,
                          x_def_padded, n_measured):
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6), constrained_layout=True)

    ax = axs[0]
    ax.imshow(u, extent=extent, cmap="gray", vmin=-1, vmax=1,
              aspect="auto", rasterized=True)
    ax.scatter(x_peaks, y_peaks, s=14, facecolors="none",
               edgecolors="red", lw=1.0, label="twist peaks")
    ax.axhline(0.0, color="cyan", ls="--", lw=0.8, alpha=0.8)
    for xd in x_def_padded:
        ax.axvline(xd, color="#ffcc00", ls=":", lw=0.9, alpha=0.9)
    ax.set_title("SH pattern, twist maxima")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc="upper right")
    panel_label(ax, "(a)", color="w")

    ax = axs[1]
    ax.imshow(pattern_p, extent=extent, cmap="gray", vmin=-1, vmax=1,
              aspect="auto", rasterized=True)
    for xd in x_def_padded:
        ax.axvline(xd, color="#ffcc00", ls=":", lw=0.9, alpha=0.9)
    ax.set_title(rf"Pismen ladder ($N={len(x_def_padded)}$,"
                 rf" {n_measured} measured)")
    ax.set_xlabel("$x$")
    panel_label(ax, "(b)", color="w")

    fig.savefig(out)
    plt.close(fig)


def fig_spacing(out, x_def_sorted, d_theory, mu):
    """Per-gap measured spacings against L = pi/cos(alpha)."""
    gaps = np.diff(np.sort(x_def_sorted))
    centers = 0.5 * (np.sort(x_def_sorted)[:-1] + np.sort(x_def_sorted)[1:])

    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    ax.axhline(d_theory, color="C3", ls="--", lw=1.2,
               label=r"$L=\pi/\cos\alpha = %.3f$" % d_theory)
    ax.plot(centers, gaps, "o", ms=5, color="C0", label="measured gaps")
    m, s = gaps.mean(), gaps.std()
    ax.axhspan(m - s, m + s, color="C0", alpha=0.12)
    ax.axhline(m, color="C0", lw=1.0,
               label=r"mean $= %.3f \pm %.3f$" % (m, s))
    ax.set_xlabel("gap center $x$")
    ax.set_ylabel("defect spacing")
    ax.set_title(rf"$\mu = {mu}$: spacing vs.\ PN law")
    ax.legend(loc="best")
    fig.savefig(out)
    plt.close(fig)
    return float(m), float(s)


# ---------------------------------------------------------------------
# Main (computation verbatim from v1; plotting replaced)
# ---------------------------------------------------------------------

def compare_zigzag_pismen():
    use_paper_style()
    stem = OP_PATH.stem
    fig_root = Path(OUT_ROOT) if OUT_ROOT else (
        Path(__file__).resolve().parent / "results"
        / "pismen_fit_and_residuals_v2" / stem)
    fig_root.mkdir(parents=True, exist_ok=True)

    op = np.load(OP_PATH)
    x = op["x"]
    y = op["y"]
    x0 = 0.5 * (x.min() + x.max())
    y0 = 0.5 * (y.min() + y.max())
    x_c = x - x0
    y_c = y - y0
    Xc, Yc = np.meshgrid(x_c, y_c)
    dx = x_c[1] - x_c[0]
    dy = y_c[1] - y_c[0]
    extent_c = [x_c.min(), x_c.max(), y_c.min(), y_c.max()]

    u = op["u"][..., -1]
    ramp_raw = op["ramp"]
    ramp = (ramp_raw - np.nanmin(ramp_raw)) / (
        np.nanmax(ramp_raw) - np.nanmin(ramp_raw) + 1e-12)

    domain_mask = ramp >= 0.995
    inner_mask = ramp >= 0.999
    structure = np.ones((3, 3), dtype=bool)
    domain_mask = binary_erosion(domain_mask, structure=structure, iterations=10)
    inner_mask = binary_erosion(inner_mask, structure=structure, iterations=10)
    domain_mask = inner_mask

    f_raw = op["k1_orig"][..., -1]
    g_raw = op["k2_orig"][..., -1]
    f_or_m, g_or_m = orient_vector_field(f_raw, g_raw, domain_mask)
    f_or = np.asarray(f_or_m)
    g_or = np.asarray(g_or_m)
    phi_or = np.arctan2(g_or, f_or)
    pi_mask_or = phi_jump_mask_inner(phi_or, tol=np.pi / 10,
                                     inner_mask=inner_mask)
    mask_ok_or = domain_mask & (~pi_mask_or)

    fx_or, fy_or = safe_central_derivs(f_or, dx, dy, mask_ok_or)
    gx_or, gy_or = safe_central_derivs(g_or, dx, dy, mask_ok_or)
    J_or = fx_or * gy_or - fy_or * gx_or

    twist_or = disk_area_integrals_safe(
        J_or, Xc, Yc, domain_mask, mask_ok_or, radius=TWIST_RADIUS)
    peaks = get_local_peaks(twist_or, mask=mask_ok_or,
                            min_distance=PEAK_MIN_DISTANCE,
                            threshold_rel=PEAK_THRESHOLD_REL)
    x_peaks = Xc[peaks[:, 0], peaks[:, 1]]
    y_peaks = Yc[peaks[:, 0], peaks[:, 1]]

    y_span = y_c.max() - y_c.min()
    band_halfwidth = 0.5 * MIDLINE_BAND_FRACTION * y_span
    band_mask = np.abs(y_peaks) <= band_halfwidth
    x_def = x_peaks[band_mask]
    if x_def.size == 0:
        print("No twist peaks in midline band; using all peaks.")
        x_def = x_peaks
    x_def_sorted = np.sort(x_def)

    alpha = np.arcsin(MU)
    d_theory = np.pi / np.cos(alpha)

    padded = list(x_def_sorted)
    if INCLUDE_VIRTUAL_DEFECTS and len(padded) > 0:
        for _ in range(N_VIRTUAL_PER_SIDE):
            cand = padded[0] - d_theory
            if cand < x_c.min():
                break
            padded.insert(0, cand)
        for _ in range(N_VIRTUAL_PER_SIDE):
            cand = padded[-1] + d_theory
            if cand > x_c.max():
                break
            padded.append(cand)
    x_def_padded = np.sort(np.array(padded, dtype=float))

    theta_p, pattern_p = pismen_on_sh_grid(Xc, Yc, x_def_padded, amp=1.0)
    theta_px, theta_py = safe_central_derivs(theta_p, dx, dy, domain_mask)

    # ------------------ figures ------------------
    fig_pattern_and_peaks(fig_root / "pismen_pattern_and_peaks.png",
                          u, pattern_p, extent_c, x_peaks, y_peaks,
                          x_def_padded, n_measured=len(x_def_sorted))

    cmp_mask = domain_mask & np.isfinite(theta_px) & np.isfinite(theta_py) \
        & np.isfinite(f_or) & np.isfinite(g_or)

    rms_k1 = comparison_triptych(
        fig_root / "pismen_k1_comparison.png",
        f_or, theta_px, cmp_mask, extent_c,
        labels=(r"$k_1$ (uHu)", r"$\partial_x\theta_{\rm P}$"),
        suptitle=rf"$\mu={MU}$: longitudinal wavevector vs.\ Pismen ladder")

    rms_k2 = comparison_triptych(
        fig_root / "pismen_k2_comparison.png",
        g_or, theta_py, cmp_mask, extent_c,
        labels=(r"$k_2$ (uHu)", r"$\partial_y\theta_{\rm P}$"),
        suptitle=rf"$\mu={MU}$: transverse wavevector vs.\ Pismen ladder")

    if x_def_sorted.size >= 2:
        mean_gap, std_gap = fig_spacing(fig_root / "pismen_spacing.png",
                                        x_def_sorted, d_theory, MU)
    else:
        mean_gap, std_gap = np.nan, np.nan

    summary = dict(
        mu=MU, d_theory=float(d_theory),
        n_defects_measured=int(len(x_def_sorted)),
        n_defects_padded=int(len(x_def_padded)),
        mean_spacing=mean_gap, std_spacing=std_gap,
        rel_spacing_error=(abs(mean_gap - d_theory) / d_theory
                           if np.isfinite(mean_gap) else None),
        rms_k1_error=rms_k1, rms_k2_error=rms_k2,
    )
    with open(fig_root / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print("Saved figures to", fig_root)


if __name__ == "__main__":
    compare_zigzag_pismen()
