# src/utils/spectral.py
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

def SpectralDerivs(u, Lx, Ly, type):
    """
    Periodic spectral derivatives for 2D fields.
    u : 2D array (ny, nx)
    Lx, Ly : domain lengths
    type : 'x','y','xx','yy','xy'
    """
    ny, nx = u.shape
    kx = 2 * np.pi * fftfreq(nx, d=Lx / nx)
    ky = 2 * np.pi * fftfreq(ny, d=Ly / ny)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")

    Uhat = fft2(u)

    if type == "x":
        dU = 1j * KX * Uhat
    elif type == "y":
        dU = 1j * KY * Uhat
    elif type == "xx":
        dU = -(KX**2) * Uhat
    elif type == "yy":
        dU = -(KY**2) * Uhat
    elif type == "xy":
        dU = -(KX * KY) * Uhat
    else:
        raise ValueError(f"Unknown derivative type '{type}'")

    return np.real(ifft2(dU))


def macro(arr, sigma, Lx, Ly):
    """
    Gaussian spectral smoothing on a periodic rectangle.
    Matches original calc_order_parameter_and_phase_sh_rectangle.py.
    """
    M, N = np.shape(arr)
    a = 0.5 * Lx
    b = 0.5 * Ly
    kx = (np.pi / a) * fftfreq(N, d=1 / N)
    ky = (np.pi / b) * fftfreq(M, d=1 / M)
    xi, eta = np.meshgrid(kx, ky)
    kernel = np.exp(-0.5 * sigma**2 * (xi**2 + eta**2))
    return np.real(ifft2(kernel * fft2(arr)))
