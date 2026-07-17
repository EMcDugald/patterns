"""
Swift-Hohenberg PGB diamond simulation module.

Design notes
------------
- Geometry is driven by (Ly, Ny, mu, margin), then Lx and Nx are chosen so dx ~ dy.
- The active diamond is approximately
      |x| + slope * |y| <= xlim,
  where slope = sqrt(1 - mu^2) / mu.
- Supports two IC choices:
    * ic_method="distance" : old distance-to-boundary style IC
    * ic_method="knee"     : two side-by-side knee bends stitched in x
- Saves full-domain fields; no crop by default.
"""

import json
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import gaussian_filter

from .core import integrate_sh
import math


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def build_diamond_geometry(Ly, Ny, mu, margin=0.45):
    if not (0.0 < mu < 1.0):
        raise ValueError(f"mu must be in (0,1), got {mu}")
    if not (0.0 < margin < 1.0):
        raise ValueError(f"margin must be in (0,1), got {margin}")

    Ly = float(Ly)
    Ny = int(Ny)
    dy = Ly / Ny

    k1 = np.sqrt(1.0 - mu**2)
    k2 = mu
    slope = k1 / k2

    H_diamond = 2.0 * margin * Ly
    xlim_geom = abs(slope) * (H_diamond / 2.0)
    Lx = 2.0 * xlim_geom

    Nx_float = Lx / dy
    Nx = int(round(Nx_float))
    if Nx % 2 != 0:
        Nx += 1
    Nx = max(Nx, 2)

    dx = Lx / Nx

    return {
        "k1": k1,
        "k2": k2,
        "slope": slope,
        "Ly": Ly,
        "Ny": Ny,
        "dy": dy,
        "H_diamond": H_diamond,
        "xlim_geom": xlim_geom,
        "Lx": Lx,
        "Nx": Nx,
        "dx": dx,
    }


def make_centered_grid(Lx, Ly, Nx, Ny):
    xx = (Lx / Nx) * np.linspace(-Nx / 2 + 1, Nx / 2, Nx)
    yy = (Ly / Ny) * np.linspace(-Ny / 2 + 1, Ny / 2, Ny)
    X, Y = np.meshgrid(xx, yy)
    return xx, yy, X, Y


# ---------------------------------------------------------------------------
# Ramp / boundary helpers
# ---------------------------------------------------------------------------

def build_diamond_ramp(X, Y, slope, xlim, Rscale=0.5, tanh_scale=5.0, sigma_R=1.0):
    pyramid = -(np.abs(X) + np.abs(slope * Y)) + xlim
    R_raw = Rscale * np.tanh(tanh_scale * pyramid)
    R = gaussian_filter(R_raw, sigma=sigma_R)
    return R


def build_diamond_inner_ramp(xx, yy, slope, Ly, margin_inner=0.35, tanh_scale=5.0):
    if not (0.0 < margin_inner < 1.0):
        raise ValueError(f"margin_inner must be in (0,1), got {margin_inner}")

    H_inner = 2.0 * margin_inner * Ly
    xlim_inner = abs(slope) * (H_inner / 2.0)

    Xg, Yg = np.meshgrid(xx, yy)
    pyramid_inner = -(np.abs(Xg) + np.abs(slope * Yg)) + xlim_inner

    R_inner_raw = 0.5 * np.tanh(tanh_scale * pyramid_inner)
    R_inner = (R_inner_raw - R_inner_raw.min()) / (
        R_inner_raw.max() - R_inner_raw.min() + 1e-12
    )
    return R_inner


def build_diamond_boundary(Ly, slope, margin=0.45, npts=512):
    H_diamond = 2.0 * margin * Ly
    xlim = abs(slope) * (H_diamond / 2.0)

    y_top = np.linspace(0.0, H_diamond / 2.0, int(npts))
    x_top_right = xlim - slope * y_top
    x_top_left = -xlim + slope * y_top

    x_bdry = np.concatenate([x_top_left, x_top_right])
    y_bdry = np.concatenate([y_top, y_top])
    return np.vstack([x_bdry, y_bdry])


def build_diamond_inner_boundary(Ly, slope, margin_inner=0.35, npts=512):
    H_inner = 2.0 * margin_inner * Ly
    xlim_inner = abs(slope) * (H_inner / 2.0)

    y_top_inner = np.linspace(0.0, H_inner / 2.0, int(npts))
    x_top_inner_right = xlim_inner - slope * y_top_inner
    x_top_inner_left = -xlim_inner + slope * y_top_inner

    x_bdry_inner = np.concatenate([x_top_inner_left, x_top_inner_right])
    y_bdry_inner = np.concatenate([y_top_inner, y_top_inner])
    return np.vstack([x_bdry_inner, y_bdry_inner])


# ---------------------------------------------------------------------------
# IC builders
# ---------------------------------------------------------------------------

def _distance_to_sampled_boundary(X, Y, bdry):
    rho = np.zeros_like(X)
    bx = bdry[0, :]
    by = bdry[1, :]

    Ny, Nx = X.shape
    for ii in range(Ny):
        for jj in range(Nx):
            dx2 = (X[ii, jj] - bx) ** 2
            dy2 = (Y[ii, jj] - by) ** 2
            rho[ii, jj] = np.min(dx2 + dy2)
    return rho


def build_diamond_distance_ic(
    X,
    Y,
    Lx,
    Ly,
    Nx,
    Ny,
    mu,
    xlim,
    amp=0.1,
    sigma_k=1.0,
):
    slope = np.sqrt(1.0 - mu**2) / mu

    nmx = 64
    xplus = xlim * np.arange(1, nmx + 1) / nmx
    xminus = -xlim * np.arange(1, nmx + 1) / nmx

    tr_bdry = np.vstack((xplus, -xplus / slope + xlim / slope))
    tl_bdry = np.vstack((xminus, xminus / slope + xlim / slope))
    ll_bdry = np.vstack((xminus, -xminus / slope - xlim / slope))
    lr_bdry = np.vstack((xplus, xplus / slope - xlim / slope))
    bdry_full = np.hstack((tr_bdry, tl_bdry, ll_bdry, lr_bdry))

    rho = _distance_to_sampled_boundary(X, Y, bdry_full)

    kx = (2.0 * np.pi / Lx) * fftfreq(Nx, 1.0 / Nx)
    ky = (2.0 * np.pi / Ly) * fftfreq(Ny, 1.0 / Ny)
    xi, eta = np.meshgrid(kx, ky)

    rho_hat = fft2(rho)
    rho_hat *= np.exp(-sigma_k * (xi**2 + eta**2))
    rho_smooth = np.real(ifft2(rho_hat))

    u0 = amp * np.sin(np.sqrt(np.maximum(rho_smooth, 0.0)))
    return u0, rho_smooth


def _logsumexp_pair(a, b):
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))

def build_diamond_knee_ic(X, Y, mu, amp=0.5, stitch_width=None):
    """
    Two knee bends with phase grain boundary along x-axis.

    For x < 0: use a knee whose far-field normals are (-cos(alpha), ± sin(alpha)).
    For x > 0: use a knee whose far-field normals are ( cos(alpha), ± sin(alpha)).

    The two phases coincide along x = 0, and we blend them smoothly in x
    using hat_left / hat_right indicator fields.
    """
    # Far-field wavevector components
    k1 = np.sqrt(1.0 - mu**2)  # cos(alpha)
    k2 = mu                    # sin(alpha)

    # Left knee: normal ~ (-k1, ±k2)
    # Choose a pair of branches whose dominant far field has kx ≈ -k1.
    theta_left = _logsumexp_pair(
        -k2 * X - k1 * Y,   # branch 1: k = (-k2,  k1)
        -k2 * X + k1 * Y,   # branch 2: k = ( k2,  k1)
    )

    # Right knee: normal ~ (k1, ±k2)
    # Mirror the left knee by flipping X -> -X in the arguments.
    theta_right = _logsumexp_pair(
        k2 * X - k1 * Y,  # branch 1: k = (-k2,  k1)
        k2 * X + k1 * Y,  # branch 2: k = ( k2,  k1)
    )

    # Smooth indicators in x: left active for x<0, right for x>0
    if stitch_width is None:
        stitch_width = max(math.floor(X[0,1]-X[0,0])*30, 1e-12)

    hat_right = 0.5 * (1.0 + np.tanh(X / stitch_width))   # ~1 for x >> 0
    hat_left = 1.0 - hat_right                            # ~1 for x << 0

    # Combined phase: left knee active on x<0, right on x>0, gentle blend at x=0
    theta = hat_left * theta_left + hat_right * theta_right

    # Base IC before ramp
    u0 = amp * np.cos(theta)

    return {
        "u0": u0,
        "theta": theta,
        "theta_left": theta_left,
        "theta_right": theta_right,
        "hat_left": hat_left,
        "hat_right": hat_right,
        "stitch_width": stitch_width,
    }


# ---------------------------------------------------------------------------
# Top-level solver
# ---------------------------------------------------------------------------

def solve_sh_pgb_diamond(
    Ly,
    Ny,
    mu,
    h,
    tmax,
    nsave=1,
    margin=0.45,
    margin_inner=0.35,
    ic_method="knee",
    Rscale=0.5,
    xlim_scale=1.0,
    tanh_scale=5.0,
    amp=0.5,
    sigma_R=1.0,
    sigma_k=1.0,
    knee_center_frac=0.5,
    knee_stitch_width=None,
    energy=True,
    t_save_window=None,
    save_initial_phase=True,
):
    if ic_method not in ("distance", "knee"):
        raise ValueError(f"ic_method must be 'distance' or 'knee', got {ic_method!r}")

    geom = build_diamond_geometry(Ly=Ly, Ny=Ny, mu=mu, margin=margin)

    Lx = geom["Lx"]
    Nx = geom["Nx"]
    dx = geom["dx"]
    dy = geom["dy"]
    slope = geom["slope"]
    xlim = xlim_scale * geom["xlim_geom"]

    xx, yy, X, Y = make_centered_grid(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)

    R = build_diamond_ramp(
        X, Y,
        slope=slope,
        xlim=xlim,
        Rscale=Rscale,
        tanh_scale=tanh_scale,
        sigma_R=sigma_R,
    )

    theta_initial = None
    cos_theta_initial = None
    theta_left = None
    theta_right = None
    hat_left = None
    hat_right = None

    if ic_method == "distance":
        u0_base, _ = build_diamond_distance_ic(
            X, Y, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, mu=mu, xlim=xlim,
            amp=amp, sigma_k=sigma_k,
        )
        u0 = u0_base * R
    else:
        knee = build_diamond_knee_ic(
            X, Y, mu=mu, amp=amp,
            stitch_width=knee_stitch_width,
        )
        u0_base = knee["u0"]
        theta_initial = knee["theta"]
        theta_left = knee["theta_left"]
        theta_right = knee["theta_right"]
        hat_left = knee["hat_left"]
        hat_right = knee["hat_right"]
        u0 = u0_base * R

        if save_initial_phase:
            cos_theta_initial = np.cos(theta_initial)

    tt, uu, ee, _, _, _, _ = integrate_sh(
        u0, R, Lx, Ly,
        h=h,
        tmax=tmax,
        nsave=nsave,
        energy=energy,
        t_save_window=t_save_window,
    )

    bdry = build_diamond_boundary(Ly=Ly, slope=slope, margin=margin, npts=512)
    bdry_inner = build_diamond_inner_boundary(
        Ly=Ly, slope=slope, margin_inner=margin_inner, npts=512
    )
    inner_ramp = build_diamond_inner_ramp(
        xx, yy, slope=slope, Ly=Ly, margin_inner=margin_inner, tanh_scale=tanh_scale
    )

    meta = {
        "geometry": "diamond",
        "mu": float(mu),
        "alpha": float(np.arcsin(mu)),
        "k1": float(geom["k1"]),
        "k2": float(geom["k2"]),
        "slope": float(slope),
        "Ly": float(Ly),
        "Ny": int(Ny),
        "dy": float(dy),
        "Lx": float(Lx),
        "Nx": int(Nx),
        "dx": float(dx),
        "dx_over_dy": float(dx / dy),
        "margin": float(margin),
        "margin_inner": float(margin_inner),
        "H_diamond": float(geom["H_diamond"]),
        "xlim_geom": float(geom["xlim_geom"]),
        "xlim": float(xlim),
        "ic_method": ic_method,
        "Rscale": float(Rscale),
        "xlim_scale": float(xlim_scale),
        "tanh_scale": float(tanh_scale),
        "amp": float(amp),
        "sigma_R": float(sigma_R),
        "sigma_k": float(sigma_k),
        "knee_center_frac": float(knee_center_frac),
        "knee_stitch_width": None if knee_stitch_width is None else float(knee_stitch_width),
        "h": float(h),
        "tmax": float(tmax),
        "nsave": int(nsave),
        "t_save_window": None if t_save_window is None else float(t_save_window),
        "save_initial_phase": bool(save_initial_phase),
    }

    return {
        "tt": tt,
        "x_full": xx,
        "y_full": yy,
        "x": xx,
        "y": yy,
        "u": uu,
        "e": ee,
        "ramp": R,
        "inner_ramp": inner_ramp,
        "bdry": bdry,
        "bdry_inner": bdry_inner,
        "theta_initial": theta_initial if save_initial_phase else None,
        "cos_theta_initial": cos_theta_initial if save_initial_phase else None,
        "theta_left_initial": theta_left if save_initial_phase else None,
        "theta_right_initial": theta_right if save_initial_phase else None,
        "hat_left_initial": hat_left if save_initial_phase else None,
        "hat_right_initial": hat_right if save_initial_phase else None,
        "metadata_json": json.dumps(meta),
    }