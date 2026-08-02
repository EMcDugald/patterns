# src/utils/anisotropic_macro.py
"""
Coherence-guided anisotropic generalization of utils.spectral.macro().

Standard macro() applies a single isotropic Gaussian kernel via one global
FFT multiply: kernel = exp(-0.5 * sigma^2 * |xi|^2). That's exact for a
spatially-uniform kernel. Here we want the kernel's orientation and aspect
ratio to vary in space, tracking the local director field (k1, k2) and a
local coherence measure (e.g. from lam1/lam2). A spatially-varying kernel
can't be applied as a single FFT multiply, so we do it patch-wise: within
each (overlapping, Hann-windowed) patch the kernel is treated as locally
frozen -- the same separation-of-scales assumption already underlying the
uHu derivation (kernel width ~ 1 wavelength  <<  macroscopic variation
scale). Patches are stitched back together via overlap-add.

This is meant as a drop-in companion to macro() in spectral.py, and is
written to plug into compute_uhu_ops_ramp / compute_uhu_ops_diamond_frame
in place of the isotropic macro() calls.
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq


# ---------------------------------------------------------------------
# Local field descriptors: orientation (director-correct) and coherence
# ---------------------------------------------------------------------

def director_orientation(k1, k2, weight=None):
    """
    Circular mean orientation of a director field (k ~ -k ambiguity),
    via the doubling trick: average e^{i*2*theta}, then halve the angle.
    Returns theta in (-pi/2, pi/2].
    """
    theta = np.arctan2(k2, k1)
    z = np.exp(1j * 2.0 * theta)
    if weight is not None:
        z = z * weight
    return 0.5 * np.angle(np.mean(z))


def coherence_from_eigs(lam1, lam2, eps=1e-12):
    """
    c in [0,1]: ~0 near defects (rank-one structure breaks down, both
    eigenvalues comparable), ~1 in well-formed single-wavevector stripe
    regions. lam1 <= lam2 (lam1 most negative), matching your uHu sign
    convention.
    """
    num = lam2 - lam1
    den = np.abs(lam2) + np.abs(lam1) + eps
    c = num / den
    return np.clip(c, 0.0, 1.0)


# ---------------------------------------------------------------------
# Uniform (spatially constant) rotated-elliptical kernel, exact FFT form
# ---------------------------------------------------------------------

def _kernel_transfer(shape, theta, s_par, s_perp, Lx, Ly, kind="gaussian"):
    """
    Fourier-domain rotated kernel transfer function, in the same
    convention as macro(): theta=0 means the parallel axis (s_par) acts
    along x, the perpendicular axis (s_perp) along y. (xi_par, xi_perp)
    are frequency coordinates rotated by theta.

    kind="gaussian": exp(-0.5*(s_par^2 xi_par^2 + s_perp^2 xi_perp^2)),
        i.e. real-space Gaussian of standard deviation (s_par, s_perp).

    kind="rect": sinc(s_par*xi_par/pi) * sinc(s_perp*xi_perp/pi), i.e.
        real-space UNIFORM averaging over a rotated rectangle of
        half-widths (s_par, s_perp) -- a hard cutoff, unlike the Gaussian's
        smooth falloff. This is the "rectangular support" kernel: a
        genuine box average over a rotated strip, useful for testing
        whether the Gaussian's soft tails (which always sample a little
        of both sides of a sharp core) are themselves contributing to
        the core-oscillation leakage, independent of the aspect/sharpen
        shape question.
    """
    M, N = shape
    a, b = 0.5 * Lx, 0.5 * Ly
    kx = (np.pi / a) * fftfreq(N, d=1 / N)
    ky = (np.pi / b) * fftfreq(M, d=1 / M)
    xi, eta = np.meshgrid(kx, ky)

    c, s = np.cos(theta), np.sin(theta)
    xi_par = c * xi + s * eta
    xi_perp = -s * xi + c * eta

    if kind == "gaussian":
        return np.exp(-0.5 * (s_par**2 * xi_par**2 + s_perp**2 * xi_perp**2))
    elif kind == "rect":
        return (np.sinc(s_par * xi_par / np.pi)
               * np.sinc(s_perp * xi_perp / np.pi))
    else:
        raise ValueError(f"unknown kernel kind {kind!r}")


# kept for backward compatibility with any caller importing the old name
def _elliptical_kernel(shape, theta, sigma_par, sigma_perp, Lx, Ly):
    return _kernel_transfer(shape, theta, sigma_par, sigma_perp, Lx, Ly,
                            kind="gaussian")


def macro_aniso_uniform(arr, theta, sigma_par, sigma_perp, Lx, Ly,
                        kind="gaussian"):
    """Spatially-uniform anisotropic analogue of macro(). Exact (single FFT).
    kind="gaussian" (default, backward compatible) or "rect" (rotated box
    average of half-widths sigma_par x sigma_perp); see _kernel_transfer.
    """
    kernel = _kernel_transfer(arr.shape, theta, sigma_par, sigma_perp,
                              Lx, Ly, kind=kind)
    return np.real(ifft2(kernel * fft2(arr)))


# ---------------------------------------------------------------------
# Spatially-varying version: patch-wise, Hann-windowed overlap-add
# ---------------------------------------------------------------------

def _hann2d(ny, nx):
    wy = np.hanning(ny) if ny > 1 else np.ones(1)
    wx = np.hanning(nx) if nx > 1 else np.ones(1)
    return np.outer(wy, wx)


def macro_aniso_field(
    arr,
    k1,
    k2,
    coherence,
    Lx,
    Ly,
    sigma_perp_floor,
    aspect_max=2.5,
    patch=64,
    stride=None,
):
    """
    Spatially-varying anisotropic smoothing of `arr`, guided by the local
    director field (k1, k2) and a coherence field in [0, 1] (same shape).

    sigma_perp_floor : minimum width across stripes (~ your current
        isotropic sigma, e.g. pi/2). Never shrunk below this.
    aspect_max : cap on sigma_par / sigma_perp. Keep modest (2-3) --
        this is deliberately conservative near defects/knees, where the
        orientation estimate itself is least trustworthy.
    patch, stride : patch size and step (pixels). stride defaults to
        patch // 2 for 50% overlap.

    Returns the smoothed field, same shape as arr.
    """
    ny, nx = arr.shape
    stride = stride or patch // 2
    win = _hann2d(patch, patch)

    out = np.zeros_like(arr, dtype=float)
    wsum = np.zeros_like(arr, dtype=float)

    ys = list(range(0, ny - patch + 1, stride)) or [0]
    xs = list(range(0, nx - patch + 1, stride)) or [0]
    if ys[-1] != ny - patch:
        ys.append(max(ny - patch, 0))
    if xs[-1] != nx - patch:
        xs.append(max(nx - patch, 0))

    for y0 in ys:
        for x0 in xs:
            sl = (slice(y0, y0 + patch), slice(x0, x0 + patch))
            a_patch = arr[sl]
            if a_patch.shape != (patch, patch):
                continue

            k1p, k2p, cp = k1[sl], k2[sl], coherence[sl]
            weight = np.hypot(k1p, k2p)
            theta_k = director_orientation(k1p, k2p, weight=weight)
            theta_stripe = theta_k + np.pi / 2.0  # long axis perp to k

            c_local = float(np.median(cp))
            aspect = 1.0 + (aspect_max - 1.0) * c_local
            s_perp = sigma_perp_floor
            s_par = sigma_perp_floor * aspect

            patch_Lx = Lx * patch / nx
            patch_Ly = Ly * patch / ny
            smoothed = macro_aniso_uniform(
                a_patch, theta_stripe, s_par, s_perp, patch_Lx, patch_Ly
            )

            out[sl] += smoothed * win
            wsum[sl] += win

    wsum = np.where(wsum < 1e-12, 1.0, wsum)
    return out / wsum


# ---------------------------------------------------------------------
# Iteration driver skeleton -- plug your compute_uhu_ops_* in for
# `extract_fn`, which should accept a `smoother` callable in place of
# the hardcoded macro() calls.
# ---------------------------------------------------------------------

def iterate_anisotropic_uhu(
    u, x, y, deriv_fun, ramp,
    extract_fn,               # signature: (u, x, y, smoother, deriv_fun, ramp) -> dict
    sigma_isotropic,          # your current fixed sigma, e.g. pi/2 -- used as floor
    n_iter=3,
    aspect_max=2.5,
    damping=0.4,
    patch=64,
):
    """
    1) Isotropic pass (aspect=1 everywhere) to get an initial k1,k2,lam1,lam2.
    2) Loop: build coherence + orientation from current fields, construct
       anisotropic smoother, damp against previous kernel, re-extract.
    3) Return the full history so you can inspect where/whether it converges,
       and -- just as importantly -- where it *doesn't* (that's diagnostic).
    """
    Lx, Ly = x[-1] - x[0], y[-1] - y[0]

    iso_smoother = lambda arr, s=sigma_isotropic: macro_aniso_field(
        arr,
        k1=np.zeros_like(u), k2=np.ones_like(u),   # dummy: isotropic => aspect forced to 1
        coherence=np.zeros_like(u),
        Lx=Lx, Ly=Ly, sigma_perp_floor=s, aspect_max=1.0, patch=patch,
    )

    result = extract_fn(u, x, y, iso_smoother, deriv_fun, ramp)
    history = [result]

    prev_k1, prev_k2 = result["k1_orig"], result["k2_orig"]

    for it in range(n_iter):
        coherence = coherence_from_eigs(result["lam1"], result["lam2"])

        def aniso_smoother(arr, k1=prev_k1, k2=prev_k2, c=coherence):
            return macro_aniso_field(
                arr, k1, k2, c, Lx, Ly,
                sigma_perp_floor=sigma_isotropic,
                aspect_max=aspect_max, patch=patch,
            )

        new_result = extract_fn(u, x, y, aniso_smoother, deriv_fun, ramp)

        # damp k-field update to avoid feedback sharpening near defects
        new_result["k1_orig"] = (
            (1 - damping) * prev_k1 + damping * new_result["k1_orig"]
        )
        new_result["k2_orig"] = (
            (1 - damping) * prev_k2 + damping * new_result["k2_orig"]
        )

        prev_k1, prev_k2 = new_result["k1_orig"], new_result["k2_orig"]
        result = new_result
        history.append(result)

    return history
