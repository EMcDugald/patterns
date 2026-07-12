# op_extract/geometry.py
"""
Geometry helpers for uHu-based order parameters.

Provides ramp functions on periodic rectangles:
  - Rectangular ramps for PGB zippers.
  - Ellipse ramps for zigzag/ellipse phase work.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Rectangular ramp (current PGB usage)
# ---------------------------------------------------------------------------

def build_rectangular_ramp_smooth(x, y, xmargin=0.10, ymargin=0.10, tanhscale=7.5):
    """
    Smooth rectangular ramp that is ~1 in the interior and decays to 0 near
    the boundaries in x and y.

    This is the same function you currently use from uhu.py, just moved here.
    """
    X, Y = np.meshgrid(x, y)
    Lx = x[-1] - x[0]
    Ly = y[-1] - y[0]

    xm = xmargin * Lx
    ym = ymargin * Ly

    # distances to boundaries
    dx_left  = X - x[0]
    dx_right = x[-1] - X
    dy_bot   = Y - y[0]
    dy_top   = y[-1] - Y

    # smooth ramps in each direction
    rx = 0.5 * (1.0 + np.tanh((np.minimum(dx_left, dx_right) - xm) * tanhscale / Lx))
    ry = 0.5 * (1.0 + np.tanh((np.minimum(dy_bot, dy_top) - ym) * tanhscale / Ly))

    return rx * ry


# ---------------------------------------------------------------------------
# Ellipse ramp (for later phase/zigzag work)
# ---------------------------------------------------------------------------

def build_ellipse_ramp(x, y, xmargin=0.10, ymargin=0.10, tanhscale=7.5):
    """
    Smooth ellipse ramp on the same rectangular domain.

    Roughly: 1 inside an ellipse centred at the origin, decays to 0 outside.

    Implementation: moved from your existing build_ellipse_ramps in uhu.py,
    simplified to return a single ramp.
    """
    X, Y = np.meshgrid(x, y)
    Lx = x[-1] - x[0]
    Ly = y[-1] - y[0]

    # semi-axes (shrink by margins)
    a = 0.5 * Lx * (1.0 - xmargin)
    b = 0.5 * Ly * (1.0 - ymargin)

    # ellipse level set
    lvl = (X / a)**2 + (Y / b)**2

    # tanh-based soft cut: lvl < 1 interior, lvl > 1 exterior
    ramp = 0.5 * (1.0 + np.tanh((1.0 - lvl) * tanhscale))

    return ramp




from scipy.ndimage import gaussian_filter


def build_diamond_ramp_general(
    x,
    y,
    mu,
    margin,
    tanh_scale=1.0,
    smooth_sigma=None,
):
    """
    Smooth diamond-shaped ramp on a rectangle grid.

    This keeps the older diamond geometry logic: centered coordinates,
    far-field slope sqrt(1-mu^2)/mu, tanh support, optional corner rounding.

    Parameters
    ----------
    x, y : 1D arrays
        Grid coordinates.
    mu : float
        Far-field vertical wavenumber component in (0,1].
    margin : float
        Vertical fraction of full Ly/2 used to define the diamond half-height.
    tanh_scale : float
        Steepness of the tanh transition.
    smooth_sigma : float or None
        Optional post-smoothing applied to the ramp to soften corners.

    Returns
    -------
    ramp : 2D array, shape (Ny, Nx)
        Soft support in [0, 1].
    """
    x = np.asarray(x)
    y = np.asarray(y)
    X, Y = np.meshgrid(x, y)

    Lx = x[-1] - x[0]
    Ly = y[-1] - y[0]

    x_shift = 0.5 * (x[0] + x[-1])
    y_shift = 0.5 * (y[0] + y[-1])
    Xc = X - x_shift
    Yc = Y - y_shift

    if not (0.0 < mu <= 1.0):
        raise ValueError(f"mu must lie in (0,1]; got {mu}")
    if not (0.0 < margin < 0.5):
        raise ValueError(f"margin must lie in (0,0.5); got {margin}")

    k1 = np.sqrt(max(0.0, 1.0 - mu**2))
    slope = k1 / max(mu, 1e-12)

    H_diamond = 2.0 * margin * Ly
    xlim = abs(slope) * (H_diamond / 2.0)
    pyramid = -(np.abs(Xc) + np.abs(slope * Yc)) + xlim

    raw = 0.5 * np.tanh(tanh_scale * pyramid)
    raw = raw - raw.min()
    ramp = raw / (raw.max() + 1e-12)

    if smooth_sigma is not None and smooth_sigma > 0:
        ramp = gaussian_filter(ramp, sigma=smooth_sigma)
        ramp = ramp - ramp.min()
        ramp = ramp / (ramp.max() + 1e-12)

    return ramp


def build_diamond_boundary_from_mu(
    x,
    y,
    mu,
    margin,
    n_samples_per_edge=256,
):
    """
    Construct a piecewise-linear diamond boundary in centered coordinates,
    then shift back to the original grid coordinates.

    The shape is consistent with build_diamond_ramp_general.

    Returns
    -------
    boundary : array, shape (2, N)
        Closed polyline ordered counterclockwise without duplicate endpoint.
    upper_boundary : array, shape (2, Nedge)
        Upper half from left vertex to right vertex.
    lower_boundary : array, shape (2, Nedge)
        Lower half from right vertex to left vertex.
    vertices : dict
        Named vertex coordinates in original coordinates.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    Lx = x[-1] - x[0]
    Ly = y[-1] - y[0]
    x_shift = 0.5 * (x[0] + x[-1])
    y_shift = 0.5 * (y[0] + y[-1])

    if not (0.0 < mu <= 1.0):
        raise ValueError(f"mu must lie in (0,1]; got {mu}")
    if not (0.0 < margin < 0.5):
        raise ValueError(f"margin must lie in (0,0.5); got {margin}")

    k1 = np.sqrt(max(0.0, 1.0 - mu**2))
    slope = k1 / max(mu, 1e-12)
    H_diamond = 2.0 * margin * Ly
    y_top_c = H_diamond / 2.0
    y_bot_c = -H_diamond / 2.0
    x_side_c = abs(slope) * (H_diamond / 2.0)

    left_c = np.array([-x_side_c, 0.0])
    top_c = np.array([0.0, y_top_c])
    right_c = np.array([x_side_c, 0.0])
    bottom_c = np.array([0.0, y_bot_c])

    t = np.linspace(0.0, 1.0, n_samples_per_edge)
    seg_lt = (1.0 - t)[:, None] * left_c + t[:, None] * top_c
    seg_tr = (1.0 - t)[:, None] * top_c + t[:, None] * right_c
    seg_rb = (1.0 - t)[:, None] * right_c + t[:, None] * bottom_c
    seg_bl = (1.0 - t)[:, None] * bottom_c + t[:, None] * left_c

    upper_c = np.vstack([seg_lt[:-1], seg_tr])
    lower_c = np.vstack([seg_rb[:-1], seg_bl])
    boundary_c = np.vstack([upper_c[:-1], lower_c[:-1]])

    def shift_back(arr):
        out = arr.copy()
        out[:, 0] += x_shift
        out[:, 1] += y_shift
        return out

    upper = shift_back(upper_c)
    lower = shift_back(lower_c)
    boundary = shift_back(boundary_c)

    vertices = {
        "left": np.array([left_c[0] + x_shift, left_c[1] + y_shift]),
        "top": np.array([top_c[0] + x_shift, top_c[1] + y_shift]),
        "right": np.array([right_c[0] + x_shift, right_c[1] + y_shift]),
        "bottom": np.array([bottom_c[0] + x_shift, bottom_c[1] + y_shift]),
    }

    return {
        "boundary": boundary.T,
        "upper_boundary": upper.T,
        "lower_boundary": lower.T,
        "vertices": vertices,
    }


def resample_boundary_arc_length(boundary_points, n_points):
    """
    Arc-length resampling of a polyline given as shape (2, N).
    """
    xb, yb = boundary_points
    pts = np.vstack([xb, yb]).T
    diffs = np.diff(pts, axis=0)
    seglen = np.sqrt((diffs**2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    total = s[-1]
    if total <= 0:
        return np.repeat(pts[:1].T, n_points, axis=1)
    s /= total
    s_target = np.linspace(0.0, 1.0, n_points)
    x_new = np.interp(s_target, s, xb)
    y_new = np.interp(s_target, s, yb)
    return np.vstack([x_new, y_new])


def build_diamond_phase_seeds(
    x,
    y,
    mu,
    margin,
    n_seeds=41,
    inset=0.0,
):
    """
    Build phase seed points along the upper diamond boundary.

    Parameters
    ----------
    inset : float
        Optional fractional trimming from each end of the upper boundary in
        arc-length coordinates, useful for avoiding sharp corners.
    """
    geom = build_diamond_boundary_from_mu(x, y, mu, margin)
    upper = geom["upper_boundary"]

    if inset <= 0.0:
        return resample_boundary_arc_length(upper, n_seeds)

    xb, yb = upper
    pts = np.vstack([xb, yb]).T
    diffs = np.diff(pts, axis=0)
    seglen = np.sqrt((diffs**2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    s /= s[-1]

    s0 = min(max(inset, 0.0), 0.49)
    s1 = max(min(1.0 - inset, 1.0), 0.51)
    s_target = np.linspace(s0, s1, n_seeds)
    x_new = np.interp(s_target, s, xb)
    y_new = np.interp(s_target, s, yb)
    return np.vstack([x_new, y_new])