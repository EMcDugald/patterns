# src/op_extract/uhu.py

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
from utils.spectral import macro


def compute_uhu_ops_ramp(u, x, y, sigma, deriv_fun, ramp):
    """
    Generic uHu-based order parameters on a periodic rectangle,
    given an explicit ramp.

    Returns k, A, k1_orig, k2_orig, eigenvalues, eigenvectors, ramp.
    """
    Lx = x[-1] - x[0]
    Ly = y[-1] - y[0]

    # derivatives on u * ramp
    u_xx = deriv_fun(u * ramp, Lx, Ly, "xx")
    u_xy = deriv_fun(u * ramp, Lx, Ly, "xy")
    u_yy = deriv_fun(u * ramp, Lx, Ly, "yy")

    u_uxx = macro(u * u_xx, sigma, Lx, Ly)
    u_uxy = macro(u * u_xy, sigma, Lx, Ly)
    u_uyy = macro(u * u_yy, sigma, Lx, Ly)

    uHu = np.stack(
        [
            [u_uxx, u_uxy],
            [u_uxy, u_uyy],
        ],
        axis=0,
    )
    uHu = np.moveaxis(uHu, (0, 1), (-2, -1))
    evals, evecs = np.linalg.eigh(uHu)

    uu_macro = macro(u * u * ramp, sigma, Lx, Ly)
    lam1 = evals[..., 0]
    lam2 = evals[..., 1]
    if lam1.max() > 0:
        lam1 = lam1 - lam1.max()

    tmp_uu = uu_macro.copy()
    tmp_uu[ramp < 1e-8] = 1.0

    k_sq = -lam1 / tmp_uu
    k_sq = np.maximum(k_sq, 0.0)
    k = np.sqrt(k_sq)

    A_sq = 2 * tmp_uu
    A_sq = np.maximum(A_sq, 0.0)
    A = np.sqrt(A_sq)

    e11 = evecs[..., 0, 0]
    e12 = evecs[..., 1, 0]
    norm_e1 = np.sqrt(e11**2 + e12**2)
    e11 = e11 / (norm_e1 + 1e-8)
    e12 = e12 / (norm_e1 + 1e-8)

    k1_orig = k * e11 * ramp
    k2_orig = k * e12 * ramp

    return {
        "k": k,
        "A": A,
        "k1_orig": k1_orig,
        "k2_orig": k2_orig,
        "e11": evecs[..., 0, 0],
        "e12": evecs[..., 1, 0],
        "e21": evecs[..., 0, 1],
        "e22": evecs[..., 1, 1],
        "lam1": lam1,
        "lam2": lam2,
        "ramp": ramp,
    }