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

# -----------------------------------------------------------------------
# Improved orientation (two-pass BFS with best-seed)
# -----------------------------------------------------------------------

def _best_seed(f, g, mask):
    """Pick the valid cell with smallest angular variance among 8-neighbours."""
    starts = np.argwhere(mask)
    best, best_var = tuple(starts[0]), np.inf
    offsets = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    ny, nx = f.shape
    for iy, ix in starts:
        v0 = np.array([f[iy, ix], g[iy, ix]])
        norm0 = np.linalg.norm(v0)
        if norm0 < 1e-10:
            continue
        dots = []
        for dy, dx in offsets:
            nyi, nxi = iy + dy, ix + dx
            if 0 <= nyi < ny and 0 <= nxi < nx and mask[nyi, nxi]:
                vn = np.array([f[nyi, nxi], g[nyi, nxi]])
                nn = np.linalg.norm(vn)
                if nn > 1e-10:
                    dots.append(np.dot(v0 / norm0, vn / nn))
        if dots and np.var(dots) < best_var:
            best_var = np.var(dots)
            best = (iy, ix)
    return best


def _bfs_from_seed(f, g, mask, seed):
    """Single BFS orientation pass from a given (iy, ix) seed."""
    ny, nx = f.shape
    kx = np.zeros_like(f)
    ky = np.zeros_like(g)
    visited = np.zeros((ny, nx), dtype=bool)

    iy0, ix0 = seed
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
                cand = np.array([f[nyi, nxi], g[nyi, nxi]])
                if np.dot(np.array([kx[iy, ix], ky[iy, ix]]), cand) < 0:
                    cand = -cand
                kx[nyi, nxi] = cand[0]
                ky[nyi, nxi] = cand[1]
                visited[nyi, nxi] = True
                queue.append((nyi, nxi))

    return kx, ky


def _mask_boundary(mask):
    """True at valid cells that touch an invalid cell (8-connected boundary)."""
    from scipy.ndimage import binary_dilation
    dilated = binary_dilation(~mask, structure=np.ones((3,3), dtype=bool))
    return mask & dilated


def orient_vector_field_v2(k1, k2, mask=None, pi_tol=np.pi / 10):
    """
    Improved sign-consistency via two-pass BFS with best-seed selection.

    Pass 1: BFS from the smoothest seed.
    Pass 2: find residual internal π-jumps; for each connected component
            separated by a jump, flip the minority-sign component.

    Parameters
    ----------
    k1, k2   : (Ny, Nx) array-like
    mask     : bool (Ny, Nx), True where valid. None → all valid.
    pi_tol   : tolerance for π-jump detection (passed to phi_jump_mask).

    Returns
    -------
    k1_or, k2_or : masked arrays
    """
    from scipy.ndimage import label as nd_label

    f = np.asarray(k1, dtype=float)
    g = np.asarray(k2, dtype=float)
    ny, nx = f.shape

    if mask is None:
        mask = np.ones((ny, nx), dtype=bool)
    mask = np.asarray(mask, dtype=bool)

    starts = np.argwhere(mask)
    if starts.size == 0:
        return np.ma.masked_where(~mask, np.zeros_like(f)), \
               np.ma.masked_where(~mask, np.zeros_like(g))

    # --- Pass 1: BFS from best seed ---
    seed = _best_seed(f, g, mask)
    kx, ky = _bfs_from_seed(f, g, mask, seed)

    # --- Pass 2: fix residual internal π-jumps ---
    phi = np.arctan2(ky, kx)
    jump = phi_jump_mask(phi, tol=pi_tol) & mask
    internal_jump = jump & ~_mask_boundary(mask)

    if internal_jump.any():
        # label connected components of the valid non-jump region
        clean_mask = mask & ~jump
        labeled, n_comp = nd_label(clean_mask)

        # for each component, check sign vs seed component
        seed_label = labeled[seed[0], seed[1]]

        for comp_id in range(1, n_comp + 1):
            if comp_id == seed_label:
                continue
            comp = labeled == comp_id
            if not comp.any():
                continue
            # sample a point in this component
            pts = np.argwhere(comp)
            iy_s, ix_s = pts[len(pts)//2]
            # find a neighbour across a jump boundary to compare sign
            offsets4 = [(-1,0),(1,0),(0,-1),(0,1)]
            flipped = False
            for dy, dx in offsets4:
                nyi, nxi = iy_s + dy, ix_s + dx
                if 0 <= nyi < ny and 0 <= nxi < nx and mask[nyi, nxi]:
                    dot = kx[iy_s, ix_s]*kx[nyi, nxi] + ky[iy_s, ix_s]*ky[nyi, nxi]
                    if dot < 0:
                        kx[comp] *= -1
                        ky[comp] *= -1
                        flipped = True
                        break
            _ = flipped  # suppress unused warning

    return np.ma.masked_where(~mask, kx), np.ma.masked_where(~mask, ky)


# -----------------------------------------------------------------------
# Integral diagnostics: disk twist + circle circulation
# -----------------------------------------------------------------------

from scipy.ndimage import map_coordinates

def disk_twist_integrals(J, X, Y, mask_centers, mask_ok,
                         radius, n_r=32, n_theta=64):
    """
    For each center in mask_centers, integrate J over a disk of given radius.
    Skips centers whose disk overlaps invalid (mask_ok=False) pixels.
    Returns (Ny,Nx) array, NaN where not computed.
    """
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    r      = np.linspace(0, radius, n_r, endpoint=True)
    theta  = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    dr     = r[1] - r[0] if n_r > 1 else radius
    dtheta = 2*np.pi / n_theta

    A  = np.full_like(J, np.nan, dtype=float)
    ys, xs = np.where(mask_centers)

    for iy, ix in zip(ys, xs):
        x0, y0 = X[iy, ix], Y[iy, ix]
        rr, tt = np.meshgrid(r, theta, indexing="ij")
        x_d = x0 + rr * np.cos(tt)
        y_d = y0 + rr * np.sin(tt)
        j_d = (x_d - X[0, 0]) / dx
        i_d = (y_d - Y[0, 0]) / dy
        j_idx = np.round(j_d).astype(int)
        i_idx = np.round(i_d).astype(int)
        inside = ((i_idx >= 0) & (i_idx < J.shape[0]) &
                  (j_idx >= 0) & (j_idx < J.shape[1]))
        if np.any(inside & (~mask_ok[i_idx * inside, j_idx * inside])):
            continue
        coords = np.vstack([i_d.ravel(), j_d.ravel()])
        s_d = map_coordinates(J, coords, order=1, mode="nearest").reshape(rr.shape)
        if np.isnan(s_d).any():
            continue
        A[iy, ix] = (s_d * rr).sum() * dr * dtheta

    return A


def circle_circulation_integrals(k1, k2, X, Y, mask_centers, mask_ok,
                                  radius, n_theta=256):
    """
    For each center in mask_centers, compute the line integral
    ∮ k · dl around a circle of given radius (circulation).
    Skips centers whose circle overlaps invalid pixels.
    Returns (Ny,Nx) array, NaN where not computed.
    """
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    f = np.asarray(k1, dtype=float)
    g = np.asarray(k2, dtype=float)
    I = np.full_like(f, np.nan, dtype=float)
    ys, xs = np.where(mask_centers)

    for iy, ix in zip(ys, xs):
        x0, y0 = X[iy, ix], Y[iy, ix]
        x_c = x0 + radius * cos_t
        y_c = y0 + radius * sin_t
        j_c = (x_c - X[0, 0]) / dx
        i_c = (y_c - Y[0, 0]) / dy
        j_idx = np.round(j_c).astype(int)
        i_idx = np.round(i_c).astype(int)
        inside = ((i_idx >= 0) & (i_idx < f.shape[0]) &
                  (j_idx >= 0) & (j_idx < f.shape[1]))
        if np.any(inside & (~mask_ok[i_idx * inside, j_idx * inside])):
            continue
        coords = np.vstack([i_c, j_c])
        f_c = map_coordinates(f, coords, order=1, mode="nearest")
        g_c = map_coordinates(g, coords, order=1, mode="nearest")
        if np.isnan(f_c).any() or np.isnan(g_c).any():
            continue
        dlx = -radius * sin_t
        dly =  radius * cos_t
        I[iy, ix] = (f_c * dlx + g_c * dly).sum() * (2*np.pi / n_theta)

    return I