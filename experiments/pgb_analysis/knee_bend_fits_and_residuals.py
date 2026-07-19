from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

OP_PATH = Path(
    "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/uhu/shallow_mus/sig_pio2/raw/"
    "0312_v3_sh_pgb_zigzag_cropped_v3_knee_mu0.4_T100_N50_nx512_Ny512_uhu_sigma1.57.npz"
)

T_INDEX = -1  # final frame

RAMP_THRESH_BFS = 0.98
RAMP_THRESH_STRICT = 0.999
RAMP_EROSION_ITERS = 48

PHI_JUMP_TOL = np.pi / 10.0
QUIVER_SKIP = 16

# RCN / balance parameters
ALPHA = 2.0 / 3.0
ETA   = 8.0 / 9.0

# Analytic knee-bend parameters
USE_ANALYTIC_THETA = True
MU_ANALYTIC = 0.4  # must match the simulation
# ---------------------------------------------------------------------
# Analytic knee-bend phase
# ---------------------------------------------------------------------


def build_analytic_kneebend_theta(X, Y, lambdaval, k_plus, k_minus):
    """
    Analytic knee-bend phase:
      theta = log( exp(lambda k_plus · x) + exp(lambda k_minus · x) )
    with x = (X,Y). Returns theta only.
    """
    dot_plus  = k_plus[0] * X + k_plus[1] * Y
    dot_minus = k_minus[0] * X + k_minus[1] * Y

    t1 = np.exp(lambdaval * dot_plus)
    t2 = np.exp(lambdaval * dot_minus)

    theta = np.log(t1 + t2)
    return theta



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

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while queue:
        iy, ix = queue.popleft()
        for dy, dx in offsets:
            nyi, nxi = iy + dy, ix + dx
            if (0 <= nyi < ny) and (0 <= nxi < nx) and mask[nyi, nxi] and not visited[nyi, nxi]:
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
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dy, dx in offsets:
        phi_shift = np.roll(np.roll(phi, dy, axis=0), dx, axis=1)
        dphi = phi_shift - phi
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        mask_jump |= (np.abs(dphi) > (np.pi - tol))
    return mask_jump


def safe_central_derivs(s, dx, dy, mask_ok):
    """
    Central differences on s, using mask_ok to avoid points
    near invalid / boundary regions. Returns (s_x, s_y).
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


def plot_field(field, mask, X, Y, title, fname, cmap="viridis", vmin=None, vmax=None):
    """
    Generic masked scalar field plot on (X,Y), saved to fname.
    """
    extent = [X.min(), X.max(), Y.min(), Y.max()]
    Fm = np.ma.masked_where(~mask, field)

    if vmin is None or vmax is None:
        data = Fm.compressed()
        if data.size == 0:
            vmin_eff, vmax_eff = 0.0, 1.0
        else:
            vmin_eff, vmax_eff = np.nanmin(data), np.nanmax(data)
    else:
        vmin_eff, vmax_eff = vmin, vmax

    plt.figure(figsize=(6, 5))
    im = plt.imshow(Fm, extent=extent, origin="lower",
                    cmap=cmap, vmin=vmin_eff, vmax=vmax_eff)
    plt.title(title)
    plt.colorbar(im, shrink=0.7)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


def rms_on_mask(arr, mask):
    vals = arr[mask]
    return np.sqrt(np.nanmean(vals**2)) if vals.size > 0 else np.nan


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    # Output directory based on OP filename
    proj_root = Path(__file__).resolve().parents[2]
    op_stem = OP_PATH.stem  # e.g. "0312_v3_sh_pgb_..."
    fig_root = (
        proj_root
        / "experiments"
        / "pgb_analysis"
        / "results"
        / "knee_bend_fits_and_residuals"
        / op_stem
    )
    fig_root.mkdir(parents=True, exist_ok=True)
    print("Saving figures to:", fig_root)

    op = np.load(OP_PATH)

    x = op["x"]
    y = op["y"]
    u_all = op["u"]
    ramp_raw = op["ramp"]
    k1_sym_all = op["k1_sym"]
    k2_sym_all = op["k2_sym"]

    Ny, Nx, Nt = k1_sym_all.shape
    print("Grid:", Ny, Nx, "Nt:", Nt)

    # recenter coordinates
    x0 = 0.5 * (x.min() + x.max())
    y0 = 0.5 * (y.min() + y.max())
    x_c = x - x0
    y_c = y - y0
    Xc, Yc = np.meshgrid(x_c, y_c)
    dx = x_c[1] - x_c[0]
    dy = y_c[1] - y_c[0]
    extent_c = [x_c.min(), x_c.max(), y_c.min(), y_c.max()]

    # time slice
    it = T_INDEX if T_INDEX >= 0 else Nt + T_INDEX
    u = u_all[..., it]
    k1_sym = k1_sym_all[..., it]
    k2_sym = k2_sym_all[..., it]
    ramp = ramp_raw

    # masks from ramp
    rmin, rmax = np.nanmin(ramp), np.nanmax(ramp)
    ramp_n = (ramp - rmin) / (rmax - rmin + 1e-12)
    structure = np.ones((3, 3), dtype=bool)

    domain_mask_bfs = ramp_n >= RAMP_THRESH_BFS
    domain_mask_bfs = binary_erosion(
        domain_mask_bfs, structure=structure, iterations=RAMP_EROSION_ITERS
    )

    domain_mask_strict = ramp_n >= RAMP_THRESH_STRICT
    domain_mask_strict = binary_erosion(
        domain_mask_strict, structure=structure, iterations=RAMP_EROSION_ITERS
    )

    print("BFS mask points:", domain_mask_bfs.sum())
    print("Strict mask points:", domain_mask_strict.sum())

    # 2D ramp plot
    plt.figure(figsize=(6, 5))
    im = plt.imshow(ramp_n, extent=extent_c, origin="lower", cmap="viridis")
    plt.title("Rectangular ramp (normalized)")
    plt.colorbar(im, shrink=0.7, label="ramp_n")
    plt.tight_layout()
    plt.savefig(fig_root / "ramp_2d.png", dpi=300)
    plt.close()

    # 1D cross-section at mid y
    j_mid = Ny // 2
    plt.figure(figsize=(6, 4))
    plt.plot(x_c, ramp_n[j_mid, :], "-k")
    plt.axhline(RAMP_THRESH_BFS, color="orange", linestyle="--",
                label=f"BFS thresh = {RAMP_THRESH_BFS}")
    plt.axhline(RAMP_THRESH_STRICT, color="red", linestyle="--",
                label=f"strict thresh = {RAMP_THRESH_STRICT}")
    plt.xlabel("x (centered)")
    plt.ylabel("ramp_n at y ≈ 0")
    plt.legend()
    plt.title("Rectangular ramp 1D cross-section (mid y)")
    plt.tight_layout()
    plt.savefig(fig_root / "ramp_1d_mid_y.png", dpi=300)
    plt.close()

    # symmetric native and oriented
    k1_sym_native = np.ma.masked_where(~domain_mask_strict, k1_sym)
    k2_sym_native = np.ma.masked_where(~domain_mask_strict, k2_sym)

    k1_sym_or_bfs, k2_sym_or_bfs = orient_vector_field(k1_sym, k2_sym, domain_mask_bfs)
    k1_sym_oriented = np.ma.masked_where(~domain_mask_strict, k1_sym_or_bfs)
    k2_sym_oriented = np.ma.masked_where(~domain_mask_strict, k2_sym_or_bfs)

    # SH pattern in strict mask
    plot_field(u, domain_mask_strict, Xc, Yc,
               "u (strict mask)", fig_root / "u_strict.png", cmap="gray")

    # π-jump mask for oriented symmetric field
    phi_sym_or = np.arctan2(np.asarray(k2_sym_oriented), np.asarray(k1_sym_oriented))
    pi_mask_sym_or = phi_jump_mask(phi_sym_or, tol=PHI_JUMP_TOL)

    valid_mask = domain_mask_strict & (~pi_mask_sym_or)
    print("Valid mask points (strict minus π-jumps):", valid_mask.sum())

    # Plot π-jump mask restricted to strict mask
    pi_mask_display = np.zeros_like(pi_mask_sym_or, dtype=float)
    pi_mask_display[pi_mask_sym_or & domain_mask_strict] = 1.0
    pi_mask_display[~domain_mask_strict] = np.nan
    plot_field(pi_mask_display, np.isfinite(pi_mask_display),
               Xc, Yc, "π-jumps (sym oriented, strict mask)",
               fig_root / "pi_jumps_sym_oriented_strict.png", cmap="gray")

    # Quiver of k_sym_oriented on valid_mask
    skip = max(1, QUIVER_SKIP)
    Xq = Xc[::skip, ::skip]
    Yq = Yc[::skip, ::skip]
    K1q = np.asarray(k1_sym_oriented)[::skip, ::skip]
    K2q = np.asarray(k2_sym_oriented)[::skip, ::skip]
    valid_sub = valid_mask[::skip, ::skip]

    plt.figure(figsize=(6, 5))
    plt.imshow(np.ma.masked_where(~valid_mask, u),
               extent=extent_c,
               origin="lower", cmap="gray")
    plt.quiver(Xq[valid_sub], Yq[valid_sub],
               K1q[valid_sub], K2q[valid_sub],
               color="cyan", scale=30)
    plt.title("u with k_sym_oriented quiver (valid mask)")
    plt.tight_layout()
    plt.savefig(fig_root / "u_with_k_sym_oriented_quiver_valid.png", dpi=300)
    plt.close()

    # -----------------------------------------
    # Finite-difference derivatives on valid_mask
    # -----------------------------------------
    k1 = np.asarray(k1_sym_oriented)
    k2 = np.asarray(k2_sym_oriented)
    k_mag = np.sqrt(k1**2 + k2**2)

    kx_x, kx_y = safe_central_derivs(k1, dx, dy, valid_mask)
    ky_x, ky_y = safe_central_derivs(k2, dx, dy, valid_mask)

    div_k = kx_x + ky_y
    curl_k = ky_x - kx_y
    J = kx_x * ky_y - kx_y * ky_x

    plot_field(k_mag, valid_mask, Xc, Yc,
               "|k| (wave number, sym oriented)", fig_root / "k_mag_valid.png")

    plot_field(k1, valid_mask, Xc, Yc,
               "k1 (sym oriented)", fig_root / "k1_valid.png", cmap="seismic")

    plot_field(k2, valid_mask, Xc, Yc,
               "k2 (sym oriented)", fig_root / "k2_valid.png", cmap="seismic")

    plot_field(J, valid_mask, Xc, Yc,
               "J = det(∇k) (FD)", fig_root / "J_valid.png", cmap="seismic")

    plot_field(div_k, valid_mask, Xc, Yc,
               "div k (FD)", fig_root / "divk_valid.png", cmap="seismic")

    plot_field(div_k**2, valid_mask, Xc, Yc,
               "(div k)^2", fig_root / "divk_sq_valid.png", cmap="viridis")

    # -----------------------------------------
    # G and G_approx
    # -----------------------------------------
    k2_ = k_mag**2
    G2_exact = -k2_**4 + 4.0 * k2_**3 - 5.0 * k2_**2 + 2.0 * k2_
    G2_exact_clip = np.maximum(G2_exact, 0.0)
    G_exact = np.sqrt(G2_exact_clip)

    G2_approx = (k2_ - 1.0)**2
    G_approx = np.sqrt(G2_approx)

    plot_field(G2_exact, valid_mask, Xc, Yc,
               "G^2 exact = -k^8 + 4k^6 -5k^4 + 2k^2",
               fig_root / "G2_exact_valid.png", cmap="magma")

    plot_field(G_exact, valid_mask, Xc, Yc,
               "G exact (sqrt clipped)", fig_root / "G_exact_valid.png", cmap="viridis")

    plot_field(G2_approx, valid_mask, Xc, Yc,
               "G_approx^2 = (k^2 - 1)^2",
               fig_root / "G2_approx_valid.png", cmap="magma")

    plot_field(G_approx, valid_mask, Xc, Yc,
               "G_approx", fig_root / "G_approx_valid.png", cmap="viridis")

    print("\nSaved k, J, div(k), and G diagnostics to", fig_root)

    # -----------------------------------------
    # Self-dual balance residuals and Edens (exact vs approx G)
    # -----------------------------------------
    coeff = np.sqrt(ALPHA / ETA)

    for label, G_use in [("exact", G_exact), ("approx", G_approx)]:
        E_comp = ETA * (div_k**2)
        E_bend = ALPHA * (G_use**2)
        E_diff = E_comp - E_bend

        rms_E = rms_on_mask(E_diff, valid_mask)
        print(f"[{label}] RMS[eta|div k|^2 - alpha G^2] on valid_mask = {rms_E:.3e}")

        R_plus = div_k - coeff * G_use
        R_minus = div_k + coeff * G_use

        rms_plus = rms_on_mask(R_plus, valid_mask)
        rms_minus = rms_on_mask(R_minus, valid_mask)
        print(f"[{label}] RMS[div k -  sqrt(alpha/eta) G]  = {rms_plus:.3e}")
        print(f"[{label}] RMS[div k +  sqrt(alpha/eta) G]  = {rms_minus:.3e}")

        R_best = R_plus if rms_plus <= rms_minus else R_minus
        plot_field(R_best, valid_mask, Xc, Yc,
                   f"Residual div k - sgn*sqrt(alpha/eta) G ({label})",
                   fig_root / f"residual_divk_vs_G_best_{label}.png",
                   cmap="seismic")

        # Energy density
        Edens = 0.5 * (ETA * div_k**2 + ALPHA * G_use**2)
        Edens_m = np.where(valid_mask, Edens, np.nan)
        plot_field(Edens_m, np.isfinite(Edens_m), Xc, Yc,
                   f"Edens 0.5[eta div k^2 + alpha G^2] ({label})",
                   fig_root / f"Edens_2d_valid_{label}.png", cmap="magma")

    # -----------------------------------------
    # Alpha–eta balance: 2D energy densities + errors (exact vs approx)
    # -----------------------------------------
    eps = 1e-12
    E_comp = ETA * (div_k**2)

    # exact
    E_bend_e = ALPHA * (G_exact**2)
    E_diff_e = np.abs(E_comp - E_bend_e)
    E_rel_e = E_diff_e / (E_comp + E_bend_e + eps)

    E_comp_m = np.where(valid_mask, E_comp, np.nan)
    E_bend_em = np.where(valid_mask, E_bend_e, np.nan)
    E_diff_em = np.where(valid_mask, E_diff_e, np.nan)
    E_rel_em = np.where(valid_mask, E_rel_e, np.nan)

    plot_field(E_comp_m, np.isfinite(E_comp_m), Xc, Yc,
               r"$\eta\,|\mathrm{div}\,k|^2$",
               fig_root / "E_comp_2d_valid_exact.png", cmap="magma")

    plot_field(E_bend_em, np.isfinite(E_bend_em), Xc, Yc,
               r"$\alpha\,G^2$ (exact)",
               fig_root / "E_bend_2d_valid_exact.png", cmap="magma")

    plot_field(E_diff_em, np.isfinite(E_diff_em), Xc, Yc,
               r"$|\eta\,\mathrm{div}k^2 - \alpha\,G^2|$ (exact)",
               fig_root / "E_diff_2d_valid_exact.png", cmap="magma")

    plot_field(E_rel_em, np.isfinite(E_rel_em), Xc, Yc,
               "relative error (exact G)",
               fig_root / "E_rel_2d_valid_exact.png", cmap="viridis")

    # approx
    E_bend_a = ALPHA * (G_approx**2)
    E_diff_a = np.abs(E_comp - E_bend_a)
    E_rel_a = E_diff_a / (E_comp + E_bend_a + eps)

    E_bend_am = np.where(valid_mask, E_bend_a, np.nan)
    E_diff_am = np.where(valid_mask, E_diff_a, np.nan)
    E_rel_am = np.where(valid_mask, E_rel_a, np.nan)

    plot_field(E_bend_am, np.isfinite(E_bend_am), Xc, Yc,
               r"$\alpha\,G_{\mathrm{approx}}^2$",
               fig_root / "E_bend_2d_valid_approx.png", cmap="magma")

    plot_field(E_diff_am, np.isfinite(E_diff_am), Xc, Yc,
               r"|eta div k^2 - alpha G_approx^2|",
               fig_root / "E_diff_2d_valid_approx.png", cmap="magma")

    plot_field(E_rel_am, np.isfinite(E_rel_am), Xc, Yc,
               "relative error (approx G)",
               fig_root / "E_rel_2d_valid_approx.png", cmap="viridis")

    # --- Global-scaled relative error (exact G) ---
    # Reuse E_comp, E_bend_e, E_diff_e, eps from above
    C = max(np.nanmean(E_comp[valid_mask]),
            np.nanmean(E_bend_e[valid_mask]))

    E_rel_global = E_diff_e / (E_comp + E_bend_e + C + eps)
    E_rel_global_m = np.where(valid_mask, E_rel_global, np.nan)

    plot_field(E_rel_global_m, np.isfinite(E_rel_global_m), Xc, Yc,
               "global-scaled relative error (exact G)",
               fig_root / "E_rel_global_2d_valid_exact.png", cmap="viridis")

    rms_rel_global = rms_on_mask(E_rel_global, valid_mask)
    print("RMS[global-scaled relative error (exact G)] =",
          f"{rms_rel_global:.3e}")


    # -----------------------------------------
    # Analytic knee-bend comparison
    # -----------------------------------------
    if USE_ANALYTIC_THETA:
        mu = MU_ANALYTIC
        k1_const = np.sqrt(1.0 - mu**2)
        k2_const = mu
        k_plus  = np.array([k1_const,  k2_const])
        k_minus = np.array([k1_const, -k2_const])

        #lambdaval = np.sqrt(ALPHA / ETA)
        lambdaval = 1.0

        # theta_analytic via log-sum-exp
        theta_a = build_analytic_kneebend_theta(
            Xc, Yc, lambdaval, k_plus, k_minus
        )

        # Finite-difference gradient of analytic theta on valid_mask
        theta_ax, theta_ay = safe_central_derivs(theta_a, dx, dy, valid_mask)
        k1_anal = theta_ax
        k2_anal = theta_ay

        mask = valid_mask

        # ---------------------------------
        # k1 vs theta_x
        # ---------------------------------
        v1 = np.nanmax(np.abs(k1[mask]))
        v2 = np.nanmax(np.abs(k1_anal[mask]))
        vmax = max(v1, v2)

        abs_err_k1 = np.abs(k1 - k1_anal)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        im0 = plt.imshow(np.ma.masked_where(~mask, k1),
                         extent=extent_c, origin="lower",
                         cmap="seismic", vmin=-vmax, vmax=vmax)
        plt.title("k1 (oriented OP)")
        plt.colorbar(im0, shrink=0.7)

        plt.subplot(1, 3, 2)
        im1 = plt.imshow(np.ma.masked_where(~mask, k1_anal),
                         extent=extent_c, origin="lower",
                         cmap="seismic", vmin=-vmax, vmax=vmax)
        plt.title("theta_x (analytic)")
        plt.colorbar(im1, shrink=0.7)

        plt.subplot(1, 3, 3)
        im2 = plt.imshow(np.ma.masked_where(~mask, abs_err_k1),
                         extent=extent_c, origin="lower",
                         cmap="viridis")
        plt.title("|k1 - theta_x|")
        plt.colorbar(im2, shrink=0.7)

        plt.tight_layout()
        plt.savefig(fig_root / "k1_vs_theta_x_analytic.png", dpi=300)
        plt.close()

        # ---------------------------------
        # k2 vs theta_y
        # ---------------------------------
        v1 = np.nanmax(np.abs(k2[mask]))
        v2 = np.nanmax(np.abs(k2_anal[mask]))
        vmax = max(v1, v2)

        abs_err_k2 = np.abs(k2 - k2_anal)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        im0 = plt.imshow(np.ma.masked_where(~mask, k2),
                         extent=extent_c, origin="lower",
                         cmap="seismic", vmin=-vmax, vmax=vmax)
        plt.title("k2 (oriented OP)")
        plt.colorbar(im0, shrink=0.7)

        plt.subplot(1, 3, 2)
        im1 = plt.imshow(np.ma.masked_where(~mask, k2_anal),
                         extent=extent_c, origin="lower",
                         cmap="seismic", vmin=-vmax, vmax=vmax)
        plt.title("theta_y (analytic)")
        plt.colorbar(im1, shrink=0.7)

        plt.subplot(1, 3, 3)
        im2 = plt.imshow(np.ma.masked_where(~mask, abs_err_k2),
                         extent=extent_c, origin="lower",
                         cmap="viridis")
        plt.title("|k2 - theta_y|")
        plt.colorbar(im2, shrink=0.7)

        plt.tight_layout()
        plt.savefig(fig_root / "k2_vs_theta_y_analytic.png", dpi=300)
        plt.close()

        print("Analytic knee-bend comparison (valid_mask):")
        print("  RMS[k1 - theta_x] =", rms_on_mask(abs_err_k1, mask))
        print("  RMS[k2 - theta_y] =", rms_on_mask(abs_err_k2, mask))

        # ---------------------------------
        # Wavenumber |k| vs |grad theta_analytic|
        # ---------------------------------
        k_mag_anal = np.sqrt(k1_anal**2 + k2_anal**2)
        abs_err_kmag = np.abs(k_mag - k_mag_anal)

        v1 = np.nanmax(np.abs(k_mag[mask]))
        v2 = np.nanmax(np.abs(k_mag_anal[mask]))
        vmax = max(v1, v2)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        im0 = plt.imshow(np.ma.masked_where(~mask, k_mag),
                         extent=extent_c, origin="lower",
                         cmap="viridis", vmin=0.0, vmax=vmax)
        plt.title("|k| (OP)")
        plt.colorbar(im0, shrink=0.7)

        plt.subplot(1, 3, 2)
        im1 = plt.imshow(np.ma.masked_where(~mask, k_mag_anal),
                         extent=extent_c, origin="lower",
                         cmap="viridis", vmin=0.0, vmax=vmax)
        plt.title("|grad theta_analytic|")
        plt.colorbar(im1, shrink=0.7)

        plt.subplot(1, 3, 3)
        im2 = plt.imshow(np.ma.masked_where(~mask, abs_err_kmag),
                         extent=extent_c, origin="lower",
                         cmap="magma")
        plt.title("||k| - |grad theta||")
        plt.colorbar(im2, shrink=0.7)

        plt.tight_layout()
        plt.savefig(fig_root / "kmag_vs_grad_theta_analytic.png", dpi=300)
        plt.close()

        print("  RMS[|k| - |grad theta|] =",
              rms_on_mask(abs_err_kmag, mask))

        # ---------------------------------
        # Pattern comparison u vs cos(theta_analytic) (visual only)
        # ---------------------------------
        pattern_anal = np.cos(theta_a)

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        im0 = plt.imshow(np.ma.masked_where(~mask, u),
                         extent=extent_c, origin="lower",
                         cmap="gray", vmin=-1, vmax=1)
        plt.title("SH pattern u (valid mask)")
        plt.colorbar(im0, shrink=0.7)

        plt.subplot(1, 2, 2)
        im1 = plt.imshow(np.ma.masked_where(~mask, pattern_anal),
                         extent=extent_c, origin="lower",
                         cmap="gray", vmin=-1, vmax=1)
        plt.title("cos(theta_analytic)")
        plt.colorbar(im1, shrink=0.7)

        plt.tight_layout()
        plt.savefig(fig_root / "pattern_vs_cos_theta_analytic.png", dpi=300)
        plt.close()

    print("\nDone. Figures are in", fig_root)


if __name__ == "__main__":
    main()
