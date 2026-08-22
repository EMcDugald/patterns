# src/op_extract/loop_persistence.py
"""
Continuous defect field (loop-persistence) and a spatial width modulator for
the Riesz director.

Loop-persistence generalises the binary defect gate in t1_common.select_sigma_
riesz. That gate reads one width for the whole crop: sigma_large if any
disclination charge sits in the band, else sigma_small. Loop-persistence is the
same plaquette winding, but instead of a presence flag it counts, per point, how
many square-loop sizes still enclose net winding. Validated on the analytic
knee (0 everywhere, every mu) and PN (bright compact bumps at cores), and shown
to rise monotonically as a dislocation pair separates, so it reads out defect
development continuously rather than as a step.

The width modulator uses it to make the gate spatial. The current gate, being
global, applies the large width to the whole field whenever any defect is
present, including the clean-knee stretches where t1_common's notes say a wide
kernel invents structure. Here the width follows the local persistence: large
only where the field is locally defected, small on the clean stretches, with a
smooth ramp between. It collapses to the current behaviour at both extremes.

Direction (matches select_sigma_riesz): persistence high -> large width,
persistence zero -> small width.

The plaquette winding mirrors t1_common.director_winding exactly so the numbers
match the production detector.
"""

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter


def _wrap_pi(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def director_winding(ang):
    """Integer winding per plaquette of a director angle field (mod pi).
    Identical to t1_common.director_winding. Shape (Ny-1, Nx-1)."""
    a2 = 2.0 * np.asarray(ang, dtype=float)
    w = (_wrap_pi(a2[:-1, 1:] - a2[:-1, :-1])
         + _wrap_pi(a2[1:, 1:] - a2[:-1, 1:])
         + _wrap_pi(a2[1:, :-1] - a2[1:, 1:])
         + _wrap_pi(a2[:-1, :-1] - a2[1:, :-1]))
    return np.round(w / (2.0 * np.pi)).astype(int)


def loop_persistence_field(ang, half_sizes=(1, 2, 3, 4, 6, 8), thresh=0.5):
    """
    Per-plaquette count of loop sizes at which the enclosed net winding stays
    nonzero. 0 on a smooth field, rising toward len(half_sizes) at a resolved
    core, and growing with pair separation.

    Returns (count, q) with count in [0, len(half_sizes)] on the plaquette grid
    (Ny-1, Nx-1); q is the unit-plaquette winding.
    """
    q = director_winding(ang).astype(float)
    count = np.zeros_like(q)
    for m in half_sizes:
        L = 2 * m + 1
        boxsum = uniform_filter(q, size=L, mode="constant") * (L * L)
        count += (np.abs(boxsum) >= thresh).astype(float)
    return count, q


def _to_node_grid(field_plaq, node_shape):
    """Pad a plaquette-grid field (Ny-1, Nx-1) up to the node grid (Ny, Nx)."""
    return np.pad(field_plaq, ((0, 1), (0, 1)), mode="edge")[:node_shape[0],
                                                             :node_shape[1]]


def persistence_to_weight(count, n_scales, smooth_px=1.5, gamma=1.0,
                          node_shape=None):
    """
    Map the loop-persistence count to a smooth blend weight w in [0, 1].
    w = 0 -> use the small width; w = 1 -> use the large width.

    smooth_px Gaussian-smooths the weight so the width varies gently in space;
    gamma > 1 sharpens the ramp toward the cores. Returned on the node grid if
    node_shape is given.
    """
    w = np.clip(count / max(int(n_scales), 1), 0.0, 1.0) ** gamma
    if node_shape is not None:
        w = _to_node_grid(w, node_shape)
    if smooth_px > 0:
        w = gaussian_filter(w, smooth_px, mode="nearest")
    return np.clip(w, 0.0, 1.0)


def adaptive_director(u, Lx, Ly, riesz_director_fn,
                      sigma_small=0.4, sigma_large=np.pi,
                      half_sizes=(1, 2, 3, 4, 6, 8), smooth_px=1.5, gamma=1.0):
    """
    Two-pass spatial width modulator for the Riesz director.

    Pass 1: director at sigma_small (the knee-clean width). Its winding drives
            the loop-persistence field and the blend weight w(x, y).
    Pass 2: director at sigma_large. The doubled-angle vectors of the two widths
            are blended per point by w, so the reconstruction uses the small
            width on clean stretches and the large width where the field is
            locally defected.

    riesz_director_fn(u, Lx, Ly, sigma) -> dict with at least "angle" (the same
    signature as t1_common's riesz_director). Blending is done on the
    doubled-angle unit vectors, not the raw angles, so it is flip-safe.

    Returns a dict with the blended director plus every intermediate field, so a
    caller can report and validate the modulator rather than trust it.
    """
    rd_s = riesz_director_fn(u, Lx, Ly, sigma_small)
    ang_s = np.asarray(rd_s["angle"])

    count, q = loop_persistence_field(ang_s, half_sizes)
    w = persistence_to_weight(count, len(half_sizes), smooth_px=smooth_px,
                              gamma=gamma, node_shape=ang_s.shape)

    rd_l = riesz_director_fn(u, Lx, Ly, sigma_large)
    ang_l = np.asarray(rd_l["angle"])

    c2 = (1.0 - w) * np.cos(2 * ang_s) + w * np.cos(2 * ang_l)
    s2 = (1.0 - w) * np.sin(2 * ang_s) + w * np.sin(2 * ang_l)
    ang = 0.5 * np.arctan2(s2, c2)

    return {
        "angle": ang, "d1": np.cos(ang), "d2": np.sin(ang),
        "weight": w, "count": count, "q": q,
        "angle_small": ang_s, "angle_large": ang_l,
        "sigma_small": float(sigma_small), "sigma_large": float(sigma_large),
    }
