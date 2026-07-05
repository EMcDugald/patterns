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