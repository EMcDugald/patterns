# src/utils/kfield_calcs.py

import numpy as np
from collections import deque


# -----------------------------------------------------------------------
# Orientation / phase helpers
# -----------------------------------------------------------------------

def orient_vector_field(k1, k2, mask=None):
    """
    Enforce local sign-consistency on a 2D vector field (k1, k2)
    by BFS flood-fill: each newly visited neighbour is flipped if its
    dot-product with the current cell is negative.

    Parameters
    ----------
    k1, k2 : (Ny, Nx) array-like
    mask   : bool (Ny, Nx), True where field is valid. None → all valid.

    Returns
    -------
    k1_or, k2_or : masked arrays (masked where mask is False).
    """
    f = np.asarray(k1, dtype=float)
    g = np.asarray(k2, dtype=float)
    ny, nx = f.shape

    if mask is None:
        mask = np.ones((ny, nx), dtype=bool)
    mask = np.asarray(mask, dtype=bool)

    kx = np.zeros_like(f)
    ky = np.zeros_like(g)
    visited = np.zeros((ny, nx), dtype=bool)

    starts = np.argwhere(mask)
    if starts.size == 0:
        return np.ma.masked_where(~mask, kx), np.ma.masked_where(~mask, ky)

    iy0, ix0 = starts[0]
    kx[iy0, ix0] = f[iy0, ix0]
    ky[iy0, ix0] = g[iy0, ix0]
    visited[iy0, ix0] = True

    queue = deque([(iy0, ix0)])
    offsets = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    while queue:
        iy, ix = queue.popleft()
        for dy, dx in offsets:
            nyi, nxi = iy + dy, ix + dx
            if 0 <= nyi < ny and 0 <= nxi < nx and mask[nyi, nxi] and not visited[nyi, nxi]:
                curr = np.array([kx[iy, ix], ky[iy, ix]])
                cand = np.array([f[nyi, nxi], g[nyi, nxi]])
                if np.dot(curr, cand) < 0:
                    cand = -cand
                kx[nyi, nxi] = cand[0]
                ky[nyi, nxi] = cand[1]
                visited[nyi, nxi] = True
                queue.append((nyi, nxi))

    return np.ma.masked_where(~mask, kx), np.ma.masked_where(~mask, ky)


def phi_jump_mask(phi, tol=np.pi / 10):
    """
    Detect π-jumps in phase field phi (Ny, Nx).
    Returns bool array True where any 8-neighbour differs by > π - tol.
    """
    mask_jump = np.zeros_like(phi, dtype=bool)
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        phi_shift = np.roll(np.roll(phi, dy, axis=0), dx, axis=1)
        dphi = (phi_shift - phi + np.pi) % (2 * np.pi) - np.pi
        mask_jump |= np.abs(dphi) > (np.pi - tol)
    return mask_jump


# -----------------------------------------------------------------------
# Finite-difference diagnostics
# -----------------------------------------------------------------------

def _central_derivs(s, dx, dy, mask_ok):
    """
    Safe 2nd-order central differences on scalar s, only where neighbours
    are valid according to mask_ok.
    """
    ny, nx = s.shape
    sx = np.full((ny, nx), np.nan)
    sy = np.full((ny, nx), np.nan)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            if not mask_ok[j, i]:
                continue
            if mask_ok[j, i-1] and mask_ok[j, i+1]:
                sx[j, i] = (s[j, i+1] - s[j, i-1]) / (2.0 * dx)
            if mask_ok[j-1, i] and mask_ok[j+1, i]:
                sy[j, i] = (s[j+1, i] - s[j-1, i]) / (2.0 * dy)
    return sx, sy


def kfield_diagnostics(k1, k2, x, y, mask_ok):
    """
    Compute scalar diagnostics from wavevector field (k1, k2).

    Returns dict with:
      curl_k  = ∂k2/∂x − ∂k1/∂y
      div_k   = ∂k1/∂x + ∂k2/∂y
      J       = det(∇k) = (∂k1/∂x)(∂k2/∂y) − (∂k1/∂y)(∂k2/∂x)
      E       = (div k)² + (1 − |k|²)²
      k_mag   = |k|
    """
    X, Y = np.meshgrid(x, y)
    dx = float(X[0, 1] - X[0, 0])
    dy = float(Y[1, 0] - Y[0, 0])

    f = np.asarray(k1, dtype=float)
    g = np.asarray(k2, dtype=float)
    k_mag = np.sqrt(f**2 + g**2)

    fx, fy = _central_derivs(f, dx, dy, mask_ok)
    gx, gy = _central_derivs(g, dx, dy, mask_ok)

    curl_k = gx - fy
    div_k  = fx + gy
    J      = fx * gy - fy * gx
    E      = div_k**2 + (1.0 - k_mag**2)**2

    return dict(curl_k=curl_k, div_k=div_k, J=J, E=E, k_mag=k_mag)