# src/sh_sims/pgb_zigzag.py
"""
Swift-Hohenberg PGB zipper simulation module.

Grid commensurability contract
-------------------------------
- Lx = n_periods * (pi / k1)  exactly (integer number of x stripe periods).
- dx = Lx / Nx  (square cells: dy = dx).
- y_centering="node"  places y=0 on the grid for any Ny.
  For the GB cores at y = ±Ly/4 to also land on grid points, Ny must be
  divisible by 4.
- crop_Ny parity:
    odd  crop_Ny -> centre row (index crop_Ny//2) is exactly the GB grid point.
    even crop_Ny -> GB falls between the two centre rows (straddle mode).
  The crop is always centred on the nearest grid point to yl or yr.

Provides
--------
  make_x_grid         : commensurate x grid
  make_y_grid         : y grid with explicit node/cell centering
  build_pgb_zigzag_ic : stitched two-knee IC + geometry dict
  crop_around_gb      : crop 2D/3D array around a chosen GB
  solve_sh_pgb_zigzag : top-level solver; returns standardised output dict
"""

import json
import warnings
import numpy as np
from .core import integrate_sh


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def make_x_grid(Nx, k1, n_periods):
    """
    Build an x grid whose domain is exactly n_periods * lambda_x long.

    Parameters
    ----------
    Nx        : int
    k1        : float   cos(alpha)
    n_periods : int     integer number of stripe periods in x

    Returns
    -------
    xx : (Nx,) array    points in [-Lx/2, Lx/2)
    Lx : float
    dx : float
    """
    lambda_x = np.pi / k1
    Lx = int(n_periods) * lambda_x
    dx = Lx / Nx
    xx = dx * np.arange(-Nx // 2, Nx // 2)
    return xx, Lx, dx


def make_y_grid(Ny, dy, centering="node"):
    """
    Build a y grid with explicit midline centering.

    Parameters
    ----------
    Ny        : int
    dy        : float    grid spacing
    centering : str
        "node"  -> y=0 is always a grid point (any Ny)
        "cell"  -> grid straddles y=0; nearest points are at ±dy/2

    Returns
    -------
    yy : (Ny,) array
    Ly : float
    """
    Ly = Ny * dy
    if centering == "node":
        yy = dy * np.arange(-Ny // 2, Ny // 2)
    elif centering == "cell":
        yy = dy * (np.arange(-Ny // 2, Ny // 2) + 0.5)
    else:
        raise ValueError(f"centering must be 'node' or 'cell', got {centering!r}")
    return yy, Ly


def _check_gb_commensurability(Ny, y_centering):
    """Warn if the GB cores at y=±Ly/4 will not land on grid points."""
    if y_centering == "node" and Ny % 4 != 0:
        warnings.warn(
            f"y_centering='node' but Ny={Ny} is not divisible by 4. "
            "The GB cores at y=±Ly/4 will not land exactly on grid points. "
            "Choose Ny_factor so that Ny % 4 == 0 for exact node-on-GB alignment.",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# IC builder
# ---------------------------------------------------------------------------

def build_pgb_zigzag_ic(
    Nx,
    mu,
    n_periods   = 12,
    Ny_factor   = 6,
    Rscale      = 0.5,
    amp         = 0.5,
    y_centering = "node",
):
    """
    Stitched two-knee zigzag initial condition.

    mu = sin(alpha),  k1 = cos(alpha) = sqrt(1 - mu^2),  k2 = mu.

    Returns dict with keys:
        u0, theta_lower, theta_upper,
        xx, yy, X, Y, R, Lx, Ly, dx, dy,
        yl, yr, k1, k2, hat_top, hat_bottom
    """
    k1 = np.sqrt(1.0 - mu ** 2)
    k2 = mu

    xx, Lx, dx = make_x_grid(Nx, k1, n_periods)
    dy = dx
    Ny = int(round(Ny_factor * Nx))
    yy, Ly = make_y_grid(Ny, dy, centering=y_centering)

    X, Y = np.meshgrid(xx, yy)
    R    = Rscale * np.ones((Ny, Nx))

    yl = -Ly / 4.0
    yr = +Ly / 4.0

    # --- lower GB knee (core at y = yl) ---
    Yl          = Y - yl
    phi_plus_l  = k1 * X + k2 * Yl
    phi_minus_l = k1 * X - k2 * Yl
    phi_max_l   = np.maximum(phi_plus_l, phi_minus_l)
    theta_l     = phi_max_l + np.log(
        np.exp(phi_plus_l - phi_max_l) + np.exp(phi_minus_l - phi_max_l)
    )
    u_knee_lower = amp * np.cos(theta_l)

    # --- upper GB knee (core at y = yr, orientation flipped) ---
    Yr          = Y - yr
    phi_plus_r  = -k1 * X + k2 * Yr
    phi_minus_r = -k1 * X - k2 * Yr
    phi_max_r   = np.maximum(phi_plus_r, phi_minus_r)
    theta_r     = phi_max_r + np.log(
        np.exp(phi_plus_r - phi_max_r) + np.exp(phi_minus_r - phi_max_r)
    )
    u_knee_upper = amp * np.cos(theta_r)

    # --- smooth stitch about y = 0 ---
    w          = Ly / 40.0
    hat_top    = 0.5 * (1.0 + np.tanh(Y / w))
    hat_bottom = 1.0 - hat_top
    u0         = u_knee_lower * hat_bottom + u_knee_upper * hat_top

    return {
        "u0":           u0,
        "theta_lower":  theta_l,
        "theta_upper":  theta_r,
        "xx":           xx,
        "yy":           yy,
        "X":            X,
        "Y":            Y,
        "R":            R,
        "Lx":           Lx,
        "Ly":           Ly,
        "dx":           dx,
        "dy":           dy,
        "yl":           yl,
        "yr":           yr,
        "k1":           k1,
        "k2":           k2,
        "hat_top":      hat_top,
        "hat_bottom":   hat_bottom,
    }


# ---------------------------------------------------------------------------
# Crop helper
# ---------------------------------------------------------------------------

def crop_around_gb(xx, yy, arr, yl, yr, which_gb="lower", crop_Nx=None, crop_Ny=None):
    """
    Crop a 2D (Ny, Nx) or 3D (Ny, Nx, Nt) array around the chosen GB centre.

    The crop is centred on the grid index nearest to yl (which_gb="lower")
    or yr (which_gb="upper").

    crop_Ny parity
    --------------
    odd  -> centre row (index crop_Ny//2) lands on the GB grid point
    even -> GB falls between rows crop_Ny//2 - 1 and crop_Ny//2 (straddle)

    Returns dict: data, xx, yy, x_slice, y_slice.
    """
    Ny_full = len(yy)
    Nx_full = len(xx)

    Ny_crop = Ny_full if crop_Ny is None else min(int(crop_Ny), Ny_full)
    Nx_crop = Nx_full if crop_Nx is None else min(int(crop_Nx), Nx_full)

    idx_yl = int(np.argmin(np.abs(yy - yl)))
    idx_yr = int(np.argmin(np.abs(yy - yr)))

    if which_gb.lower() == "lower":
        gb_center = idx_yl
    elif which_gb.lower() == "upper":
        gb_center = idx_yr
    else:
        raise ValueError("which_gb must be 'lower' or 'upper'")

    y_st  = gb_center - Ny_crop // 2
    y_end = y_st + Ny_crop
    if y_st < 0:
        y_st, y_end = 0, Ny_crop
    if y_end > Ny_full:
        y_end, y_st = Ny_full, Ny_full - Ny_crop

    x_center = Nx_full // 2
    x_st  = x_center - Nx_crop // 2
    x_end = x_st + Nx_crop
    if x_st < 0:
        x_st, x_end = 0, Nx_crop
    if x_end > Nx_full:
        x_end, x_st = Nx_full, Nx_full - Nx_crop

    if arr.ndim == 2:
        data = arr[y_st:y_end, x_st:x_end]
    elif arr.ndim == 3:
        if arr.shape[0] == Ny_full:           # (Ny, Nx, Nt)
            data = arr[y_st:y_end, x_st:x_end, :]
        else:                                  # legacy (Nt, Ny, Nx)
            data = np.transpose(arr, (1, 2, 0))[y_st:y_end, x_st:x_end, :]
    else:
        raise ValueError("arr must be 2D or 3D")

    return {
        "data":    data,
        "xx":      xx[x_st:x_end],
        "yy":      yy[y_st:y_end],
        "x_slice": (int(x_st),  int(x_end)),
        "y_slice": (int(y_st),  int(y_end)),
    }


# ---------------------------------------------------------------------------
# Top-level solver
# ---------------------------------------------------------------------------

def solve_sh_pgb_zigzag(
    Nx,
    mu,
    h,
    tmax,
    nsave              = 1,
    n_periods          = 12,
    Ny_factor          = 6,
    Rscale             = 0.5,
    amp                = 0.5,
    y_centering        = "node",
    energy             = True,
    t_save_window      = None,
    which_gb           = "lower",
    crop_Nx            = None,
    crop_Ny            = None,
    save_initial_phase = True,
):
    """
    Full PGB zigzag run: build IC -> integrate SH -> crop -> return.

    Parameters
    ----------
    Nx : int
    mu : float          sin(alpha) in (0, 1)
    h : float           timestep
    tmax : float        final time
    nsave : int
        Snapshots saved *after* t=0.
        nsave=1  -> store t=0 and t=tmax only (first/last mode).
        nsave=N  -> store t=0 + N evenly-spaced snapshots up to tmax.
    n_periods : int     integer number of x stripe periods (sets Lx exactly)
    Ny_factor : float   Ny = round(Ny_factor * Nx); use Ny_factor s.t. Ny%4==0
                        for exact GB-node alignment with y_centering="node"
    Rscale : float      uniform bifurcation parameter
    amp : float         IC amplitude
    y_centering : str
        "node"  -> y=0 is a grid point (any Ny); GB cores land on grid when Ny%4==0
        "cell"  -> grid straddles y=0; nearest points at ±dy/2
    energy : bool       compute and store energy density
    t_save_window : float or None
        If given, the nsave snapshots are taken from [tmax-t_save_window, tmax]
        instead of evenly across [0, tmax].
    which_gb : str      "lower" | "upper"
    crop_Nx, crop_Ny : int or None
        Crop window size in grid points around the chosen GB.
        odd crop_Ny  -> centre row is the GB grid point
        even crop_Ny -> GB straddles the two centre rows
        None         -> no crop
    save_initial_phase : bool
        If True, store the analytic phase field theta at t=0 (cropped).

    Returns
    -------
    dict with keys:
        tt                             (nsave+1,)
        x, y                           1D cropped coordinate arrays
        x_full, y_full                 1D full-domain coordinate arrays
        u                              (Ny_crop, Nx_crop, nsave+1)
        e                              (Ny_crop, Nx_crop, nsave+1) or None
        theta_initial                  (Ny_crop, Nx_crop) or None
        cos_theta_initial              (Ny_crop, Nx_crop) or None
        metadata_json                  JSON string of run parameters
    """
    Ny = int(round(Ny_factor * Nx))
    _check_gb_commensurability(Ny, y_centering)

    geom = build_pgb_zigzag_ic(
        Nx          = Nx,
        mu          = mu,
        n_periods   = n_periods,
        Ny_factor   = Ny_factor,
        Rscale      = Rscale,
        amp         = amp,
        y_centering = y_centering,
    )

    tt, uu, ee, _, _, _, _ = integrate_sh(
        geom["u0"], geom["R"], geom["Lx"], geom["Ly"],
        h             = h,
        tmax          = tmax,
        nsave         = nsave,
        energy        = energy,
        t_save_window = t_save_window,
    )
    # integrate_sh_v2 returns arrays shaped (Ny, Nx, nsave+1)

    crop_u = crop_around_gb(
        geom["xx"], geom["yy"], uu,
        geom["yl"], geom["yr"], which_gb, crop_Nx, crop_Ny,
    )
    crop_e = None if ee is None else crop_around_gb(
        geom["xx"], geom["yy"], ee,
        geom["yl"], geom["yr"], which_gb, crop_Nx, crop_Ny,
    )

    theta_initial = cos_theta_initial = None
    if save_initial_phase:
        theta_field = (
            geom["theta_lower"] if which_gb.lower() == "lower"
            else geom["theta_upper"]
        )
        c = crop_around_gb(
            geom["xx"], geom["yy"], theta_field,
            geom["yl"], geom["yr"], which_gb, crop_Nx, crop_Ny,
        )
        theta_initial     = c["data"]
        cos_theta_initial = np.cos(theta_initial)

    lambda_x = np.pi / geom["k1"]
    lambda_y = np.pi / geom["k2"] if geom["k2"] > 0 else np.inf

    y_crop         = crop_u["yy"]
    gb_y           = geom["yl"] if which_gb.lower() == "lower" else geom["yr"]
    gb_row_in_crop = int(np.argmin(np.abs(y_crop - gb_y)))

    meta = {
        "mu":             float(mu),
        "alpha":          float(np.arcsin(mu)),
        "k1":             float(geom["k1"]),
        "k2":             float(geom["k2"]),
        "lambda_x":       float(lambda_x),
        "lambda_y":       float(lambda_y),
        "Lx_periods":     float(geom["Lx"] / lambda_x),  # == n_periods exactly
        "Ly_periods":     float(geom["Ly"] / lambda_y),
        "n_periods":      int(n_periods),
        "Nx":             int(Nx),
        "Ny":             int(Ny),
        "Lx":             float(geom["Lx"]),
        "Ly":             float(geom["Ly"]),
        "dx":             float(geom["dx"]),
        "dy":             float(geom["dy"]),
        "y_centering":    y_centering,
        "h":              float(h),
        "tmax":           float(tmax),
        "nsave":          int(nsave),
        "t_save_window":  None if t_save_window is None else float(t_save_window),
        "Rscale":         float(Rscale),
        "amp":            float(amp),
        "which_gb":       which_gb,
        "crop_Nx":        None if crop_Nx is None else int(crop_Nx),
        "crop_Ny":        None if crop_Ny is None else int(crop_Ny),
        "crop_x_slice":   crop_u["x_slice"],
        "crop_y_slice":   crop_u["y_slice"],
        "gb_row_in_crop": gb_row_in_crop,
        "crop_Ny_parity": ("odd"     if crop_Ny is not None and crop_Ny % 2 == 1
                           else "even"    if crop_Ny is not None
                           else "no_crop"),
    }

    return {
        "tt":                tt,
        "x_full":            geom["xx"],
        "y_full":            geom["yy"],
        "x":                 crop_u["xx"],
        "y":                 y_crop,
        "u":                 crop_u["data"],
        "e":                 None if crop_e is None else crop_e["data"],
        "theta_initial":     theta_initial,
        "cos_theta_initial": cos_theta_initial,
        "metadata_json":     json.dumps(meta),
    }