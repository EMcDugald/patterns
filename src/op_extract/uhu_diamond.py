# src/op_extract/uhu_diamond.py

import numpy as np
from pathlib import Path
from scipy.fft import fft2, ifft2, fftfreq

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

# Ensure src/ on path, mirroring other modules
import sys
sys.path.insert(0, str(_ROOT / "src"))

from utils.geometry import build_diamond_ramp_general


def macro(arr, sigma, Lx, Ly):
    """
    Gaussian spectral smoothing on a periodic rectangle.
    Matches the older uHu convention (same as rectangular/zigzag).
    """
    M, N = np.shape(arr)
    a = 0.5 * Lx
    b = 0.5 * Ly
    kx = (np.pi / a) * fftfreq(N, d=1 / N)
    ky = (np.pi / b) * fftfreq(M, d=1 / M)
    xi, eta = np.meshgrid(kx, ky)
    kernel = np.exp(-0.5 * sigma**2 * (xi**2 + eta**2))
    return np.real(ifft2(kernel * fft2(arr)))


def compute_k_diagnostics(k1, k2, Lx, Ly, deriv_fun):
    """
    Given k1, k2 and a derivative function deriv_fun(u, Lx, Ly, type),
    compute div k, curl k, det Jk.
    """
    f_x = deriv_fun(k1, Lx, Ly, "x")
    f_y = deriv_fun(k1, Lx, Ly, "y")
    g_x = deriv_fun(k2, Lx, Ly, "x")
    g_y = deriv_fun(k2, Lx, Ly, "y")
    div_k = f_x + g_y
    curl_k = g_x - f_y
    detJk = f_x * g_y - f_y * g_x
    return div_k, curl_k, detJk


def compute_uhu_ops_diamond_frame(u, x, y, sigma, deriv_fun, ramp):
    """
    Per-frame uHu extraction on a diamond support.

    Derivatives are taken on u * ramp, tensor pieces are macro(u * u_{ij}),
    and the local amplitude uses macro(u * (u * ramp)).
    """
    Lx = x[-1] - x[0]
    Ly = y[-1] - y[0]
    X, Y = np.meshgrid(x, y)

    # ramped field
    u_r = u * ramp

    # second derivatives of ramped field
    u_xx = deriv_fun(u_r, Lx, Ly, "xx")
    u_xy = deriv_fun(u_r, Lx, Ly, "xy")
    u_yy = deriv_fun(u_r, Lx, Ly, "yy")

    # tensor components
    u_uxx = macro(u * u_xx, sigma, Lx, Ly)
    u_uxy = macro(u * u_xy, sigma, Lx, Ly)
    u_uyy = macro(u * u_yy, sigma, Lx, Ly)

    uHu = np.stack([[u_uxx, u_uxy], [u_uxy, u_uyy]], axis=0)  # (2,2,Ny,Nx)
    uHu = np.moveaxis(uHu, (0, 1), (-2, -1))                  # (Ny,Nx,2,2)

    evals, evecs = np.linalg.eigh(uHu)

    # amplitude scale
    uu_macro = macro(u * u_r, sigma, Lx, Ly)
    uu_macro = np.where(np.isfinite(uu_macro), uu_macro, 0.0)
    uu_macro = np.where(uu_macro < 0, 0.0, uu_macro)

    lam1 = evals[..., 0]
    lam2 = evals[..., 1]
    if lam1.max() > 0:
        lam1 = lam1 - lam1.max()

    tmp_uu = uu_macro.copy()
    mask_bad = (ramp < 1e-6) | (tmp_uu < 1e-12)
    tmp_uu[mask_bad] = 1.0

    k_sq = -lam1 / tmp_uu
    k_sq = np.where(np.isfinite(k_sq), k_sq, 0.0)
    k_sq = np.maximum(k_sq, 0.0)
    k = np.sqrt(k_sq)

    A_sq = 2.0 * uu_macro
    A_sq = np.maximum(A_sq, 0.0)
    A = np.sqrt(A_sq)

    # principal eigenvector
    e11 = evecs[..., 0, 0]
    e12 = evecs[..., 1, 0]
    norm_e1 = np.sqrt(e11**2 + e12**2)
    e11 = e11 / (norm_e1 + 1e-8)
    e12 = e12 / (norm_e1 + 1e-8)

    # raw components
    k1_orig = k * e11 * ramp
    k2_orig = k * e12 * ramp
    k1_orig = np.where(np.isfinite(k1_orig), k1_orig, 0.0)
    k2_orig = np.where(np.isfinite(k2_orig), k2_orig, 0.0)

    # symmetric orientation: x-reflection left/right, y-reflection top/bottom
    e11_sym = e11.copy()
    e12_sym = e12.copy()

    x_shift = 0.5 * (x[0] + x[-1])
    y_shift = 0.5 * (y[0] + y[-1])
    Xc = X - x_shift
    Yc = Y - y_shift

    e11_sym[Xc < 0] = -np.abs(e11_sym[Xc < 0])
    e11_sym[Xc >= 0] = np.abs(e11_sym[Xc >= 0])
    e12_sym[Yc < 0] = -np.abs(e12_sym[Yc < 0])
    e12_sym[Yc >= 0] = np.abs(e12_sym[Yc >= 0])

    k1_sym = k * e11_sym * ramp
    k2_sym = k * e12_sym * ramp

    # diagnostics from symmetrized field (more stable orientation)
    div_k, curl_k, detJk = compute_k_diagnostics(k1_sym, k2_sym, Lx, Ly, deriv_fun)

    return {
        "k": k,
        "A": A,
        "k1_sym": k1_sym,
        "k2_sym": k2_sym,
        "k1_orig": k1_orig,
        "k2_orig": k2_orig,
        "div_k": div_k,
        "curl_k": curl_k,
        "detJk": detJk,
        "e11": evecs[..., 0, 0],
        "e12": evecs[..., 1, 0],
        "e21": evecs[..., 0, 1],
        "e22": evecs[..., 1, 1],
        "lam1": lam1,
        "lam2": lam2,
    }


def compute_uhu_ops_diamond(
    u_ts,
    x,
    y,
    mu,
    sigma,
    deriv_fun,
    margin,
    tanh_scale=1.0,
    smooth_sigma=None,
):
    """
    Time-series wrapper for diamond uHu extraction (k only).

    Parameters
    ----------
    u_ts : (Ny, Nx, Nt)
        Time-series pattern field.
    x, y : 1D arrays
    mu : float
        Far-field vertical component, passed through to ramp builder.
    sigma : float
        Smoothing scale passed to macro.
    deriv_fun : SpectralDerivs-like
    margin, tanh_scale, smooth_sigma : diamond ramp parameters.

    Returns
    -------
    ops : dict
        Contains k, A, k1/k2 (orig and sym), diagnostics, lam1/lam2, ramp, x, y, mu.
    """
    Ny, Nx, nt = u_ts.shape

    ramp = build_diamond_ramp_general(
        x,
        y,
        mu=mu,
        margin=margin,
        tanh_scale=tanh_scale,
        smooth_sigma=smooth_sigma,
    )

    k_ts = []
    A_ts = []
    k1_sym_ts = []
    k2_sym_ts = []
    k1_orig_ts = []
    k2_orig_ts = []
    div_ts = []
    curl_ts = []
    detJ_ts = []
    lam1_ts = []
    lam2_ts = []

    for t in range(nt):
        print(f"  diamond uHu frame {t+1}/{nt}")
        ops_t = compute_uhu_ops_diamond_frame(
            u_ts[:, :, t], x, y, sigma, deriv_fun, ramp=ramp
        )
        k_ts.append(ops_t["k"])
        A_ts.append(ops_t["A"])
        k1_sym_ts.append(ops_t["k1_sym"])
        k2_sym_ts.append(ops_t["k2_sym"])
        k1_orig_ts.append(ops_t["k1_orig"])
        k2_orig_ts.append(ops_t["k2_orig"])
        div_ts.append(ops_t["div_k"])
        curl_ts.append(ops_t["curl_k"])
        detJ_ts.append(ops_t["detJk"])
        lam1_ts.append(ops_t["lam1"])
        lam2_ts.append(ops_t["lam2"])

    ops = {
        "k":       np.stack(k_ts,       axis=-1),
        "A":       np.stack(A_ts,       axis=-1),
        "k1_sym":  np.stack(k1_sym_ts,  axis=-1),
        "k2_sym":  np.stack(k2_sym_ts,  axis=-1),
        "k1_orig": np.stack(k1_orig_ts, axis=-1),
        "k2_orig": np.stack(k2_orig_ts, axis=-1),
        "div_k":   np.stack(div_ts,     axis=-1),
        "curl_k":  np.stack(curl_ts,    axis=-1),
        "detJk":   np.stack(detJ_ts,    axis=-1),
        "lam1":    np.stack(lam1_ts,    axis=-1),
        "lam2":    np.stack(lam2_ts,    axis=-1),
        "ramp":    ramp,
        "x":       x,
        "y":       y,
        "mu":      mu,
    }

    return ops