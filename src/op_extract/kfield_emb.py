# src/op_extract/kfield_emb.py
"""
Local wavevector extraction by the Egolf-Melnikov-Bodenschatz (EMB)
derivative-ratio method (arXiv:comp-gas/9702001, "A new fast method for
determining local properties of striped patterns"), adapted to this
project's conventions and output schema.

Method
------
For u ~ A cos(theta) with slowly varying A, pointwise ratios of partial
derivatives give the wavevector components directly, e.g.
kx^2 = -u_xx / u. EMB select among four such ratio pairs at every point to
avoid dividing by a near-zero quantity:

  1. |u| > max(|u_x|,|u_y|), |u_xx| >  |u_yy| : kx = sqrt|u_xx/u|,
                                                ky = kx * u_xy/u_xx
  2. |u| > max(|u_x|,|u_y|), |u_xx| <= |u_yy| : ky = sqrt|u_yy/u|,
                                                kx = ky * u_xy/u_yy
  3. otherwise, |u_x| >  |u_y|               : kx = sqrt|u_xxx/u_x|,
                                                ky = kx * u_y/u_x
  4. otherwise                                : ky = sqrt|u_yyy/u_y|,
                                                kx = ky * u_x/u_y

The sign of kx is fixed non-negative (director convention), and the
pointwise field is then smoothed over a box of side ~ one wavelength.

Two deliberate departures from the paper, both sign-robustness upgrades:

  * Instead of EMB's neighbor-by-neighbor sign-aligned averaging of
    (kx, ky), we average the orientation TENSOR Q = k k^T (components
    kx^2, kx*ky, ky^2), which is invariant under k -> -k, then recover
    |k| and the director from the principal eigenpair of the averaged Q.
    This is the exact branch-free generalization of their aligned
    average, is a plain (fast, separable) box convolution, and hands
    back a free fidelity diagnostic chi_Q = lam_minor / lam_major, the
    Q-tensor analogue of the uHu chi = lam2/lam1: it measures how much
    genuine orientation mixture the box average had to discard.
  * EMB's exclusion of spurious values near defects (k^2 far outside the
    expected band) is implemented as a validity weight in a normalized
    convolution, rather than a hard neighborhood test.

Relation to the uHu pipeline (worth keeping in mind when comparing):
the uHu method averages the tensor u * Hess(u) FIRST and eigendecomposes
after, so at a knee core the average is genuinely rank-two and the
eigen-step must discard the sub-dominant family. EMB is pointwise-first:
for the symmetric standing superposition u = 2A cos(k1 x) cos(k2 y) the
raw ratios return |kx| = k1 and |ky| = k2 exactly at every point.
CAVEAT (established on the exact Hopf-Cole knee): the TRUE knee core is
a phase-curvature line, theta_yy = k2^2 sech^2(k2 y) = O(1), so there
|u_yy| = |theta_yy sin(theta)| dominates |u_xx| and the case selection
takes the y-branch, whose ratio measures the curvature term rather than
theta_y^2; the raw core estimate is then wrong (measured kx ~ 0.07,
ky ~ 0.69 at mu = 0.95 where truth is (0.31, 0)). EMB is exact in the
far field and unbiased through moderate-angle cores, but it does NOT fix
the sharp-core problem at large mu; treat it as an independent
cross-check away from cores, not a core-region authority.

Output schema matches compute_uhu_ops_ramp so runners, J diagnostics, and
the phase pipeline consume it unchanged. NOTE: lam1/lam2 here are the
NEGATED principal/minor eigenvalues of the averaged Q tensor (so lam1 <= 0
and chi = lam2/lam1 stays meaningful as a mixing diagnostic); they are not
uHu-tensor eigenvalues.
"""

import numpy as np

try:
    from scipy.ndimage import uniform_filter
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


def _box_smooth(arr, size_yx, n_pass=1, mode="wrap"):
    """Repeated uniform box filter; separable and cheap."""
    out = arr
    if _HAVE_SCIPY:
        for _ in range(n_pass):
            out = uniform_filter(out, size=size_yx, mode=mode)
        return out
    # FFT fallback: n_pass-fold box kernel via transfer function powers
    from numpy.fft import fft2, ifft2
    ny, nx = arr.shape
    by, bx = size_yx
    ky = np.ones(ny)
    kx = np.ones(nx)
    kery = np.zeros(ny); kery[:by] = 1.0 / by; kery = np.roll(kery, -(by // 2))
    kerx = np.zeros(nx); kerx[:bx] = 1.0 / bx; kerx = np.roll(kerx, -(bx // 2))
    ker = np.outer(kery, kerx)
    H = fft2(ker) ** n_pass
    return np.real(ifft2(H * fft2(arr)))


def compute_kfield_emb(u, x, y, deriv_fun, ramp,
                       box_wavelengths=1.0,
                       n_smooth=2,
                       kmin=0.2,
                       kmax=2.0,
                       wavelength=2.0 * np.pi,
                       eps=1e-12,
                       boundary_mode="wrap"):
    """
    EMB derivative-ratio wavevector extraction for a single frame.

    Parameters
    ----------
    u, x, y : field and grid, as in compute_uhu_ops_ramp.
    deriv_fun : SpectralDerivs-compatible callable.
    ramp : boundary taper; u is multiplied by it before differentiation
        and the validity weight is zero outside the interior.
    box_wavelengths : smoothing box side, in units of `wavelength`
        (EMB recommend ~ one wavelength).
    n_smooth : number of repeated box passes ("further smoothing can be
        accomplished by repeating the procedure").
    kmin, kmax : validity band for the raw |k|; points outside are
        excluded from the average, per EMB's defect exclusion.
    wavelength : nominal pattern wavelength (2 pi for SH with |k| = 1).

    Returns
    -------
    dict with keys k, A, k1_orig, k2_orig, lam1, lam2, ramp, chi_Q,
    valid_frac (local fraction of valid raw points inside the box),
    kx_raw, ky_raw (the unsmoothed pointwise field, for diagnostics).
    """
    Lx, Ly = x[-1] - x[0], y[-1] - y[0]
    ny, nx = u.shape
    dx = Lx / nx
    dy = Ly / ny

    ur = u * ramp
    scale = np.max(np.abs(ur))
    if scale < eps:
        raise ValueError("field is identically zero after ramping")
    un = (ur - np.mean(ur)) / scale

    ux = deriv_fun(un, Lx, Ly, "x")
    uy = deriv_fun(un, Lx, Ly, "y")
    uxx = deriv_fun(un, Lx, Ly, "xx")
    uyy = deriv_fun(un, Lx, Ly, "yy")
    uxy = deriv_fun(un, Lx, Ly, "xy")
    uxxx = deriv_fun(uxx, Lx, Ly, "x")
    uyyy = deriv_fun(uyy, Lx, Ly, "y")

    au, aux, auy = np.abs(un), np.abs(ux), np.abs(uy)
    use_u = au > np.maximum(aux, auy)          # cases 1-2 vs 3-4

    with np.errstate(divide="ignore", invalid="ignore"):
        # cases 1 / 2
        kx_1 = np.sqrt(np.abs(uxx / un))
        ky_1 = kx_1 * uxy / uxx
        ky_2 = np.sqrt(np.abs(uyy / un))
        kx_2 = ky_2 * uxy / uyy
        # cases 3 / 4
        kx_3 = np.sqrt(np.abs(uxxx / ux))
        ky_3 = kx_3 * uy / ux
        ky_4 = np.sqrt(np.abs(uyyy / uy))
        kx_4 = ky_4 * ux / uy

    case12 = np.abs(uxx) > np.abs(uyy)
    case34 = aux > auy
    kx = np.where(use_u,
                  np.where(case12, kx_1, kx_2),
                  np.where(case34, kx_3, kx_4))
    ky = np.where(use_u,
                  np.where(case12, ky_1, ky_2),
                  np.where(case34, ky_3, ky_4))

    # director convention: kx >= 0, sign information lives in ky
    flip = np.where(kx < 0, -1.0, 1.0)
    kx, ky = kx * flip, ky * flip

    kmag2 = kx * kx + ky * ky
    valid = (np.isfinite(kmag2)
             & (kmag2 >= kmin ** 2)
             & (kmag2 <= kmax ** 2)
             & (ramp >= 0.99))
    w = valid.astype(float)

    # sign-invariant tensor components, validity-weighted normalized box
    # odd sizes keep the box filter centered; an even box shifts the
    # averaged field by half a pixel and skews profiles through the core
    box = max(3, int(round(box_wavelengths * wavelength / dx)) | 1)
    boy = max(3, int(round(box_wavelengths * wavelength / dy)) | 1)
    size = (boy, box)

    kx_c = np.where(valid, kx, 0.0)
    ky_c = np.where(valid, ky, 0.0)
    q11 = _box_smooth(w * kx_c * kx_c, size, n_smooth, boundary_mode)
    q12 = _box_smooth(w * kx_c * ky_c, size, n_smooth, boundary_mode)
    q22 = _box_smooth(w * ky_c * ky_c, size, n_smooth, boundary_mode)
    wsm = _box_smooth(w, size, n_smooth, boundary_mode)
    wsafe = np.where(wsm < 1e-8, 1.0, wsm)
    q11, q12, q22 = q11 / wsafe, q12 / wsafe, q22 / wsafe

    # principal eigenpair of the 2x2 PSD tensor, closed form
    tr = q11 + q22
    dif = q11 - q22
    disc = np.sqrt(np.maximum(dif * dif + 4.0 * q12 * q12, 0.0))
    q_major = 0.5 * (tr + disc)
    q_minor = 0.5 * (tr - disc)

    # eigenvector of q_major: (q12, q_major - q11), guarded when q12 ~ 0
    e1x = np.where(np.abs(q12) > 1e-14, q12, np.where(dif >= 0, 1.0, 0.0))
    e1y = np.where(np.abs(q12) > 1e-14, q_major - q11,
                   np.where(dif >= 0, 0.0, 1.0))
    enorm = np.hypot(e1x, e1y)
    enorm = np.where(enorm < 1e-14, 1.0, enorm)
    e1x, e1y = e1x / enorm, e1y / enorm
    # keep the kx >= 0 director convention on the averaged field too
    eflip = np.where(e1x < 0, -1.0, 1.0)
    e1x, e1y = e1x * eflip, e1y * eflip

    k = np.sqrt(np.maximum(q_major, 0.0))
    interior_w = wsm > 1e-8
    k = np.where(interior_w, k, 0.0)

    # amplitude for schema compatibility: RMS envelope over the same box
    u2 = _box_smooth(un * un, size, n_smooth, boundary_mode)
    A = scale * np.sqrt(np.maximum(2.0 * u2, 0.0))

    with np.errstate(invalid="ignore", divide="ignore"):
        chi_q = np.where(q_major > 1e-14, q_minor / q_major, np.nan)

    return {
        "k": k,
        "A": A,
        "k1_orig": k * e1x * ramp,
        "k2_orig": k * e1y * ramp,
        "lam1": -q_major,      # Q-tensor eigenvalues, NOT uHu eigenvalues
        "lam2": -q_minor,
        "ramp": ramp,
        "chi_Q": chi_q,
        "valid_frac": wsm,
        "kx_raw": np.where(valid, kx, np.nan),
        "ky_raw": np.where(valid, ky, np.nan),
    }
