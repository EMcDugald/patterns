# src/sh_sims/core.py
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq


def etdrk4_coeffs(Lx, Ly, Nx, Ny, h):
    kx = (2.0 * np.pi / Lx) * fftfreq(Nx, 1.0 / Nx)
    ky = (2.0 * np.pi / Ly) * fftfreq(Ny, 1.0 / Ny)
    xi, eta = np.meshgrid(kx, ky)

    L = -(1 - xi**2 - eta**2)**2
    E = np.exp(h * L)
    E2 = np.exp(h * L / 2.0)

    M = 16
    r = np.exp(1j * np.pi * ((np.arange(1, M+1) - 0.5) / M))
    L2 = L.flatten()
    LR = h * np.vstack([L2] * M).T + np.vstack([r] * (Nx * Ny))

    Q = h * np.real(np.mean((np.exp(LR/2.0) - 1.0) / LR, axis=1))
    f1 = h * np.real(np.mean(
        (-4 - LR + np.exp(LR) * (4 - 3*LR + LR**2)) / LR**3, axis=1))
    f2 = h * np.real(np.mean(
        (2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
    f3 = h * np.real(np.mean(
        (-4 - 3*LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))

    Q = Q.reshape(Ny, Nx)
    f1 = f1.reshape(Ny, Nx)
    f2 = f2.reshape(Ny, Nx)
    f3 = f3.reshape(Ny, Nx)

    return xi, eta, E, E2, Q, f1, f2, f3


def dealias_mask(Nx, Ny):
    Fx = np.zeros((Nx, 1), dtype=bool)
    Fy = np.zeros((Ny, 1), dtype=bool)
    Fx[int(Nx/2 - np.round(Nx/4)):int(1 + Nx/2 + np.round(Nx/4))] = True
    Fy[int(Ny/2 - np.round(Ny/4)):int(1 + Ny/2 + np.round(Ny/4))] = True
    alxi, aleta = np.meshgrid(Fx, Fy)
    return alxi | aleta


def edensity(xi, eta, u0, ind, R):
    eloc = (1 - xi**2 - eta**2) * fft2(u0)
    eloc[ind] = 0
    eloc = np.real(ifft2(eloc)**2)

    u0sq = fft2(u0**2)
    u0sq[ind] = 0
    u0sq = np.real(ifft2(u0sq))

    u04th = fft2(u0sq**2)
    u04th[ind] = 0
    u04th = np.real(ifft2(u04th))

    return 0.5 * (eloc - R * u0sq + 0.5 * u04th)


def integrate_sh(u0, R, Lx, Ly, h, tmax, nsave, energy=True, t_save_window=None):
    Ny, Nx = u0.shape
    xi, eta, E, E2, Q, f1, f2, f3 = etdrk4_coeffs(Lx, Ly, Nx, Ny, h)
    ind = dealias_mask(Nx, Ny)

    # filter
    Rhat = fft2(R); Rhat[ind] = 0; R = np.real(ifft2(Rhat))
    vv = fft2(u0); vv[ind] = 0; u0 = np.real(ifft2(vv))
    Q[ind] = 0

    tt = np.zeros(nsave + 1)
    uu = np.zeros((Ny, Nx, nsave + 1))
    ee = np.zeros((Ny, Nx, nsave + 1)) if energy else None

    uu[..., 0] = u0
    if energy:
        ee[..., 0] = edensity(xi, eta, u0, ind, R)
    tt[0] = 0.0

    nmax = int(round(tmax / h))

    # --------------------------------------------
    # Decide which times we want to save
    # --------------------------------------------
    if t_save_window is None:
        # original behavior: evenly spaced from 0 to tmax
        idx_shift = int(np.floor(nmax / nsave))
        save_n = [k * idx_shift for k in range(1, nsave + 1)]
    else:
        # new behavior: evenly spaced in [tmax - t_save_window, tmax]
        t_start = tmax - float(t_save_window)
        if t_start < 0.0:
            t_start = 0.0
        # desired save times (excluding t=0, which we already stored)
        ts = np.linspace(t_start, tmax, nsave + 1)[1:]  # length nsave
        # convert to nearest integer step indices
        save_n = [max(1, min(nmax, int(round(t / h)))) for t in ts]

    # we will step n=1..nmax and fill snapshots when n hits these indices
    save_n_set = set(save_n)

    j = 0  # index into tt/uu for saved frames (we already used j=0 for IC)

    for n in range(1, nmax + 1):
        if n % 10 == 0 or n == nmax:
            print(f"  step {n}/{nmax}  t={t:.1f}")
        t = n * h

        Nv = fft2(R * u0 - u0**3)
        a = E2 * vv + Q * Nv
        ua = np.real(ifft2(a))
        Na = fft2(R * ua - ua**3)

        b = E2 * vv + Q * Na
        ub = np.real(ifft2(b))
        Nb = fft2(R * ub - ub**3)

        c = E2 * a + Q * (2*Nb - Nv)
        uc = np.real(ifft2(c))
        Nc = fft2(R * uc - uc**3)

        vv = E*vv + Nv*f1 + 2*(Na + Nb)*f2 + Nc*f3
        u0 = np.real(ifft2(vv))

        # Save logic: either original evenly-spaced indices, or late-window indices
        if n in save_n_set:
            j += 1
            if j > nsave:
                # safety guard; should not happen if save_n has length nsave
                break
            uu[..., j] = u0
            tt[j] = t
            if energy:
                ee[..., j] = edensity(xi, eta, u0, ind, R)

    return tt, uu, ee, xi, eta, ind, R
