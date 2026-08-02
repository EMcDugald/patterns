# src/op_extract/uhu_smoother.py
"""
Smoother-agnostic uHu extraction + moment-ladder correlators.

`compute_uhu_ops_smoother` reproduces `compute_uhu_ops_ramp` exactly when
`smoother` is the global isotropic Gaussian, but takes the averaging
operator as a callable so it can be driven by:

  - the standard global `macro()` (isotropic baseline),
  - `anisotropic_macro.macro_aniso_field` (patchwise oriented ellipse),
  - `anisotropic_diffusion.macro_aniso_diffusion` (tensor-diffusion PDE),

and therefore plugs directly into `anisotropic_macro.iterate_anisotropic_uhu`
as its `extract_fn` (signature: (u, x, y, smoother, deriv_fun, ramp) -> dict).

With `with_moments=True` it additionally returns the scalar moment ladder

    M0 = <u^2>,   M1 = <u Lap u>  (= lam1 + lam2 = trace T),
    M2 = <(Lap u)^2>,

and the {1,3} de-mixed fields (K13, A1_13, A3_13) via the quadratic
9 M0 K^2 + 10 M1 K + M2 = 0 (see harmonic_moments_sympy.solve_moments_13),
plus the rank-one validity diagnostic chi = lam2 / lam1. The de-mixing is
only meaningful where |chi| is small; downstream scripts should mask on it.
"""

import numpy as np

try:
    from utils.harmonic_moments_sympy import solve_moments_13
except ImportError:  # standalone use: file sits next to harmonic_moments_sympy
    from harmonic_moments_sympy import solve_moments_13


def make_gaussian_smoother(sigma, Lx, Ly, macro_fn):
    """Wrap the standard global macro() as a smoother callable."""
    return lambda arr: macro_fn(arr, sigma, Lx, Ly)


def compute_uhu_ops_smoother(u, x, y, smoother, deriv_fun, ramp,
                             with_moments=False):
    """
    uHu order parameters on a periodic rectangle with a caller-supplied
    averaging operator `smoother(arr) -> arr`.

    Mirrors compute_uhu_ops_ramp (uhu_zigzag.py) step for step; the only
    changes are (i) macro(...) -> smoother(...) and (ii) the optional
    moment-ladder block.
    """
    Lx = x[-1] - x[0]
    Ly = y[-1] - y[0]

    ur = u * ramp
    u_xx = deriv_fun(ur, Lx, Ly, "xx")
    u_xy = deriv_fun(ur, Lx, Ly, "xy")
    u_yy = deriv_fun(ur, Lx, Ly, "yy")

    u_uxx = smoother(u * u_xx)
    u_uxy = smoother(u * u_xy)
    u_uyy = smoother(u * u_yy)

    uHu = np.stack([[u_uxx, u_uxy], [u_uxy, u_uyy]], axis=0)
    uHu = np.moveaxis(uHu, (0, 1), (-2, -1))
    evals, evecs = np.linalg.eigh(uHu)

    uu_macro = smoother(u * u * ramp)
    lam1 = evals[..., 0]
    lam2 = evals[..., 1]
    if lam1.max() > 0:
        lam1 = lam1 - lam1.max()

    tmp_uu = uu_macro.copy()
    tmp_uu[ramp < 1e-8] = 1.0
    tmp_uu[np.abs(tmp_uu) < 1e-12] = 1.0

    k_sq = np.maximum(-lam1 / tmp_uu, 0.0)
    k = np.sqrt(k_sq)
    A = np.sqrt(np.maximum(2 * tmp_uu, 0.0))

    e11 = evecs[..., 0, 0]
    e12 = evecs[..., 1, 0]
    norm_e1 = np.sqrt(e11**2 + e12**2)
    e11n = e11 / (norm_e1 + 1e-8)
    e12n = e12 / (norm_e1 + 1e-8)

    out = {
        "k": k,
        "A": A,
        "k1_orig": k * e11n * ramp,
        "k2_orig": k * e12n * ramp,
        "e11": evecs[..., 0, 0],
        "e12": evecs[..., 1, 0],
        "e21": evecs[..., 0, 1],
        "e22": evecs[..., 1, 1],
        "lam1": lam1,
        "lam2": lam2,
        "ramp": ramp,
    }

    if with_moments:
        lap_u = u_xx + u_yy                      # Lap(u * ramp)
        M0 = uu_macro                            # <u^2>
        M1 = smoother(u * lap_u)                 # <u Lap u> = trace(T)
        M2 = smoother(lap_u * lap_u)             # <(Lap u)^2>, 4th-deriv corr.

        M0s = M0.copy()
        M0s[ramp < 1e-8] = 1.0
        K13, A1_13, A3_13 = solve_moments_13(M0s, M1, M2)

        with np.errstate(invalid="ignore", divide="ignore"):
            chi = lam2 / np.where(np.abs(lam1) < 1e-14, np.nan, lam1)

        out.update({
            "M0": M0, "M1": M1, "M2": M2,
            "K13": K13, "A1_13": A1_13, "A3_13": A3_13,
            "chi": chi,
        })

    return out
