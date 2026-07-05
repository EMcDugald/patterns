# src/op_extract/riesz.py

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import zoom, gaussian_filter
from scipy.signal import fftconvolve

def riesz_analyze_pattern(u, dx, dy):
    """
    Compute Riesz-based order parameters for one frame u(x,y).

    Returns dict with:
      'R1','R2','A','theta','orientation','rho_d','k'
    """
    ny, nx = u.shape
    kx = fftfreq(nx, dx)
    ky = fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    K_norm = np.sqrt(KX**2 + KY**2) + 1e-12

    U = fft2(u)
    R1 = np.real(ifft2(-1j * (KX / K_norm) * U))
    R2 = np.real(ifft2(-1j * (KY / K_norm) * U))

    A = np.sqrt(u**2 + R1**2 + R2**2)
    theta = np.arctan2(np.sqrt(R1**2 + R2**2), u)
    orientation = np.arctan2(R2, R1)

    dnx_dx = np.gradient(np.cos(2 * orientation), dx, axis=1)
    dnx_dy = np.gradient(np.cos(2 * orientation), dy, axis=0)
    dny_dx = np.gradient(np.sin(2 * orientation), dx, axis=1)
    dny_dy = np.gradient(np.sin(2 * orientation), dy, axis=0)
    rho_d = (dnx_dx * dny_dy - dnx_dy * dny_dx) / (4 * np.pi)

    theta_x = np.gradient(theta, dx, axis=1)
    theta_y = np.gradient(theta, dy, axis=0)
    k = np.sqrt(theta_x**2 + theta_y**2)

    return {
        "R1": R1,
        "R2": R2,
        "A": A,
        "theta": theta,
        "orientation": orientation,
        "rho_d": rho_d,
        "k": k,
    }

def local_winding_number(
    rho_d,
    dx,
    dy,
    mode="disk",
    radius=None,
    wavelength=None,
    interpolate=False,
    interp_factor=2.0,
    interp_order=3,
):
    """
    Coarse-grain rho_d into a local winding field.

    Parameters
    ----------
    rho_d : 2D array
    dx, dy : float
    mode : 'gaussian' or 'disk'
    radius : float (for disk)
    wavelength : float (for gaussian)
    interpolate : bool
    interp_factor : float
    interp_order : int

    Returns
    -------
    W : 2D array
    """
    field = rho_d.copy()

    if interpolate and interp_factor != 1.0:
        field = zoom(field, interp_factor, order=interp_order, mode="reflect", grid_mode=True)
        dx /= interp_factor
        dy /= interp_factor

    if mode == "gaussian":
        if wavelength is None:
            raise ValueError("Provide wavelength for gaussian kernel.")
        sigma = (wavelength / 2.0) / dx
        W = gaussian_filter(field, sigma=sigma, mode="wrap") * dx * dy
    elif mode == "disk":
        if radius is None:
            raise ValueError("Provide radius for disk kernel.")
        ny, nx = field.shape
        yy, xx = np.ogrid[-ny // 2 : ny // 2, -nx // 2 : nx // 2]
        mask = (xx * dx) ** 2 + (yy * dy) ** 2 <= radius**2
        kernel = mask.astype(float) / np.sum(mask)
        W = fftconvolve(field, kernel, mode="same") * dx * dy
    else:
        raise ValueError("Invalid mode. Choose 'gaussian' or 'disk'.")

    return W
