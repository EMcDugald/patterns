# src/sh_sims/ellipse.py
"""
Swift-Hohenberg ellipse simulation module (modernized from the 2025
order_parameter_models runners eikonal_ellipse.py and
eikonal_ellipse_hollow.py).

Model:  u_t = R(x) u - (1 + Lap)^2 u - u^3 on a periodic box, stepped by
core.integrate_sh (ETDRK4, the same integrator as the zigzag and diamond
runs; this module never edits core.py). R(x) is a ramp that is +R0 inside
the ellipse X^2/A^2 + Y^2/B^2 = 1 and -R0 outside.

Grid commensurability contract
------------------------------
- grid="square": square cells (dy = dx exactly), nx and ny even and
  FFT-friendly, box = ellipse plus a margin on every side, x = 0 and y = 0
  are grid nodes, so the major axis is the row ny // 2 (gb_row_in_crop).
  The analysis stack assumes square cells (wsindy_theta_t sets Ly = ny dx
  and uses dx for both kernel axes), which the 2025 box did not satisfy
  (dx 0.456 against dy 0.442 at 1024 x 352).
- grid="legacy": the 2025 box and grid exactly (x = (Lx/Nx) linspace(-Nx/2
  + 1, Nx/2, Nx), semi-axes beta Lx, beta Ly), for reproducing old runs.

Ramp
----
- ramp="distance" (default): R = R0 tanh(s / w), s the signed distance to
  the wall (positive inside), from the exact foot point. The transition
  has the same width w everywhere on the wall.
- ramp="legacy": the 2025 formula R0 tanh(3 sqrt(Lx^2+Ly^2) (beta - r)/2),
  r = sqrt((X/Lx)^2 + (Y/Ly)^2). Written without the box it is
  R0 tanh(1.5 sqrt(A^2+B^2) (1 - rhat)), rhat = sqrt((X/A)^2 + (Y/B)^2), so
  beta cancels and the same function is available on any grid. Its width
  in physical distance varies around the wall as the local |grad rhat|:
  about 0.63 at the ends of the major axis and 0.21 at the ends of the
  minor axis for A = 210, B = 70.

Initial conditions
------------------
- ic="eikonal": u0 = amp sin(s) W(s), s the exact signed distance to the
  wall (phase 0 on the wall) and W(s) = (1 + erf(s / (sqrt 2 wall_sigma)))/2
  a smooth wall envelope (wall_sigma = 0: the hard cut amp sin(d) inside, 0
  outside, as in the stand-in).
- ic="hollow": the eikonal start times a shell profile
  S(d) = erfc((d - d_shell) / (sqrt 2 edge_sigma)) / 2, plus an explicit
  interior seed amp_seed sin(s) W(s) (1 - S) (default 0), so whether a run
  tests front invasion or the growth of a graded seed is a choice, not a
  side effect. The integrator always applies its dealiasing filter to u0,
  so "empty" is limited by how much of u0 lies above the cutoff; smooth
  (Gaussian-edged) profiles are used for that reason (measured below).
- ic="eikonal_legacy", "hollow_legacy": the 2025 initial conditions
  verbatim (squared distance to 256 sampled wall points, Gaussian
  smoothing in Fourier space, sin of its square root; the hollow shell from
  tanh tails). These require grid="legacy" because their smoothing is
  written in grid-index wavenumbers.
- restart: the caller passes u_init (a saved frame) and t_offset.

Note on core.integrate_sh (observed, not changed): it zeroes the
dealiasing band of u0 and of R before stepping (the 1/2 rule: modes with
|k_x| or |k_y| above (N/4) 2 pi / L are removed, about 3.5 at dx 0.45).
Q is zeroed on that band but f1, f2, f3 are not, so a running state holds
one step's worth of tiny masked-band content; a restart from a saved frame
zeroes it again. The chunked mode below therefore differs from a single
run by round-off-sized amounts, which the runner's validation mode
measures. integrate_sh also prints t before assigning it, which raises
UnboundLocalError when nmax = 1; this module never asks for fewer than 2
steps.

Provides
--------
  EllipseGeom          geometry (copied with provenance from
                       pipelines/ellipse_zipper_assemble.py v1.1, the
                       Ellipse class: foot point, distance, axis formulas)
  make_grid            square or legacy grid
  wall_distance        d, signed s, inside, distance to the medial segment
  region_masks         bulk / axis band / wall band
  build_ramp           distance or legacy ramp
  build_ic             eikonal, hollow, eikonal_legacy, hollow_legacy
                       (wall_envelope, shell_profile)
  solve_sh_ellipse     build, integrate (single call or chunked with an
                       optional stationarity stop), return a standard dict

Changelog
---------
v1 (2026-10-27). First version. Reasons for the departures from the 2025
    code, each measured or computed in the porting session: square cells
    (see the contract above); exact wall distance instead of the 256-point
    sampled distance (wall spacing about 3.7 on the 210 x 70 ellipse, so
    the sampled distance is off by up to about half that near the wall);
    no Gaussian smoothing of the squared distance (the 2025 factor
    exp(-1.23 q^2) acts as a heat step of duration 1.23 and adds about
    2.5 to d^2, so the 2025 start has phase sqrt(d^2 + 2.5): 1.6 at the
    wall instead of 0); signed-distance ramp as the default (the legacy
    ramp is about half a cell wide on the long flanks at dx 0.45, inside
    the band the dealiasing filter removes: filter change 1.7e-1 at the wall
    and 1.7e-2 four units inside it, against 1.1e-5 for the distance ramp of
    width 2). Hollow start: the integrator's filter decides how empty the
    hole is. Measured at A = 210, B = 70, dx 0.45, d_shell 20 (max |u0| in
    the hole after the filter, amp 0.1): hard wall cut and a cos^2 shell edge
    of width 2, 6e-4 one wavelength inside the hole (a draft of this version;
    the 2025 start had 2 to 5e-4); cos^2 edge of width 2 lambda, 2e-4; erfc
    edge with any edge_sigma 2 to 4 but wall envelope 1, 1.5e-5 (the floor
    then comes from the wall); wall envelope 1.5, 2.9e-7; wall envelope 2 or
    3 with edge_sigma 3, 1.35e-7, located near the medial-axis ends, where
    the crease of the distance function lies about one edge width beyond
    the shell. Defaults wall_sigma 2, edge_sigma 3. At growth rate R0 = 0.5
    a seed of 1e-7 reaches order one in about 32 time units, so this floor,
    not zero, is the hollow run's effective interior seed; a deliberate
    amp_seed well above it controls the experiment.
"""

import contextlib
import io
import json
import time

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq, next_fast_len
from scipy.special import erf, erfc

from .core import integrate_sh, dealias_mask

MODULE_VERSION = "ellipse.py v1 (2026-10-27)"
LAMBDA = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Geometry (copied from pipelines/ellipse_zipper_assemble.py v1.1, class
# Ellipse; only the geometry methods, none of the zipper or self-dual parts)
# ---------------------------------------------------------------------------

class EllipseGeom(object):
    def __init__(self, A, B):
        self.A, self.B = float(A), float(B)
        if self.A < self.B:
            raise ValueError("EllipseGeom expects A >= B (major axis along x)")
        self.c2 = self.A ** 2 - self.B ** 2
        self.Xm = self.c2 / self.A           # half-length of the medial segment

    def d_axis(self, X):
        X = np.minimum(np.abs(X), self.Xm)
        return self.B * np.sqrt(np.maximum(1.0 - X ** 2 / max(self.c2, 1e-300), 0.0))

    def alpha_axis(self, X):
        """Angle of the wall normal through (X, 0), measured from the axis."""
        X = np.minimum(np.abs(X), self.Xm)
        ct = np.clip(self.A * X / max(self.c2, 1e-300), -1.0, 1.0)
        st = np.sqrt(1.0 - ct ** 2)
        return np.arctan2(st / self.B, ct / self.A)

    def foot(self, X, Y, n_seed=2048, n_newton=8):
        """Parametric angle t of the nearest wall point for (X, Y) with
        X, Y >= 0 (inside or outside). Returns t, distance d, axis foot
        X_foot = c^2 cos t / A, normal angle."""
        A, B, c2 = self.A, self.B, self.c2
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        ts = np.linspace(0.0, 0.5 * np.pi, n_seed)
        cs, sn = np.cos(ts), np.sin(ts)
        flatX, flatY = X.ravel(), Y.ravel()
        flatt = np.empty(flatX.size)
        for i0 in range(0, flatX.size, 4096):
            sl = slice(i0, i0 + 4096)
            D = (flatX[sl, None] - A * cs[None, :]) ** 2 + (flatY[sl, None] - B * sn[None, :]) ** 2
            flatt[sl] = ts[np.argmin(D, axis=1)]
        t = flatt.reshape(X.shape)
        for _ in range(n_newton):
            g = A * X * np.sin(t) - B * Y * np.cos(t) - c2 * np.sin(t) * np.cos(t)
            gp = A * X * np.cos(t) + B * Y * np.sin(t) - c2 * np.cos(2 * t)
            ok = np.abs(gp) > 1e-12
            step = np.where(ok, g / np.where(ok, gp, 1.0), 0.0)
            t = np.clip(t - step, 0.0, 0.5 * np.pi)
        d = np.sqrt((X - A * np.cos(t)) ** 2 + (Y - B * np.sin(t)) ** 2)
        Xf = c2 * np.cos(t) / A
        al = np.arctan2(np.sin(t) / B, np.cos(t) / A)
        return t, d, Xf, al

    def inside(self, X, Y):
        return (X / self.A) ** 2 + (Y / self.B) ** 2 <= 1.0


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

def _fast_even(n):
    m = int(next_fast_len(int(n)))
    while m % 2:
        m = int(next_fast_len(m + 1))
    return m


def make_grid(grid="square", A=210.0, B=70.0, dx=0.45, margin=15.0,
              Lx=None, Ly=None, Nx=None, Ny=None, beta=0.45):
    """grid="square": box (A + margin, B + margin) on each side, square cells
    of side dx, even FFT-friendly sizes (the box grows slightly to fit).
    grid="legacy": the 2025 box (Lx, Ly, Nx, Ny), semi-axes beta Lx, beta Ly."""
    if grid == "square":
        nx = _fast_even(np.ceil(2.0 * (A + margin) / dx))
        ny = _fast_even(np.ceil(2.0 * (B + margin) / dx))
        x = dx * (np.arange(nx) - nx // 2)
        y = dx * (np.arange(ny) - ny // 2)
        return dict(grid="square", x=x, y=y, Lx=nx * dx, Ly=ny * dx, nx=nx, ny=ny,
                    dx=float(dx), dy=float(dx), A=float(A), B=float(B), margin=float(margin))
    if grid == "legacy":
        if None in (Lx, Ly, Nx, Ny):
            raise ValueError("grid='legacy' needs Lx, Ly, Nx, Ny")
        Nx, Ny = int(Nx), int(Ny)
        xx = (Lx / Nx) * np.linspace(-Nx / 2 + 1, Nx / 2, Nx)
        yy = (Ly / Ny) * np.linspace(-Ny / 2 + 1, Ny / 2, Ny)
        return dict(grid="legacy", x=xx, y=yy, Lx=float(Lx), Ly=float(Ly), nx=Nx, ny=Ny,
                    dx=float(Lx / Nx), dy=float(Ly / Ny), A=float(beta * Lx), B=float(beta * Ly),
                    beta=float(beta))
    raise ValueError("grid must be 'square' or 'legacy', got %r" % (grid,))


# ---------------------------------------------------------------------------
# Distances and regions
# ---------------------------------------------------------------------------

def wall_distance(E, X, Y):
    """Exact distance to the wall for every grid point (the foot point is
    computed in the first quadrant and mirrored). Returns d >= 0, the signed
    distance s (positive inside), the inside mask, the distance dm to the
    medial segment |X| <= c^2/A, Y = 0, the signed axis foot and the local
    normal angle alpha of the foot (from the major axis)."""
    t, d, Xf, al = E.foot(np.abs(X), np.abs(Y))
    ins = E.inside(X, Y)
    s = np.where(ins, d, -d)
    dm = np.hypot(np.maximum(np.abs(X) - E.Xm, 0.0), Y)
    return dict(d=d, s=s, inside=ins, dm=dm, Xfoot=np.sign(X) * Xf, alpha=al)


def region_masks(dist, wall_margin=2.0, axis_band=1.5):
    """bulk: inside, more than wall_margin wavelengths from the wall and more
    than axis_band wavelengths from the medial segment; axis: the same wall
    condition but within axis_band of the segment; wall: inside and within
    wall_margin of the wall."""
    ins, d, dm = dist["inside"], dist["d"], dist["dm"]
    far_wall = ins & (d > wall_margin * LAMBDA)
    return dict(bulk=far_wall & (dm > axis_band * LAMBDA),
                axis=far_wall & (dm <= axis_band * LAMBDA),
                wall=ins & (d <= wall_margin * LAMBDA))


# ---------------------------------------------------------------------------
# Ramps
# ---------------------------------------------------------------------------

def ramp_distance(s, R0, w):
    return R0 * np.tanh(s / w)


def ramp_legacy(X, Y, A, B, R0, Lx=None, Ly=None, beta=None):
    """The 2025 ramp. With Lx, Ly, beta given, the 2025 expression verbatim
    (bitwise reproduction on the legacy grid); otherwise the equivalent
    box-free form R0 tanh(1.5 sqrt(A^2 + B^2) (1 - rhat))."""
    if Lx is not None and Ly is not None and beta is not None:
        return R0 * np.tanh(3 * np.sqrt(Lx ** 2 + Ly ** 2)
                            * (beta - np.sqrt((X / Lx) ** 2 + (Y / Ly) ** 2)) / 2.0)
    return R0 * np.tanh(1.5 * np.sqrt(A ** 2 + B ** 2) * (1.0 - np.sqrt((X / A) ** 2 + (Y / B) ** 2)))


def build_ramp(kind, X, Y, dist, A, B, R0=0.5, w=2.0, legacy_box=None):
    if kind == "distance":
        return ramp_distance(dist["s"], R0, w)
    if kind == "legacy":
        if legacy_box is not None:
            return ramp_legacy(X, Y, A, B, R0, *legacy_box)
        return ramp_legacy(X, Y, A, B, R0)
    raise ValueError("ramp must be 'distance' or 'legacy', got %r" % (kind,))


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def wall_envelope(sd, wall_sigma):
    """(1 + erf(s / (sqrt 2 wall_sigma))) / 2; wall_sigma <= 0 gives the hard cut (s > 0)"""
    if wall_sigma <= 0:
        return (sd > 0).astype(float)
    return 0.5 * (1.0 + erf(sd / (np.sqrt(2.0) * wall_sigma)))


def ic_eikonal(dist, amp, wall_sigma=2.0):
    if wall_sigma <= 0:
        return np.where(dist["inside"], amp * np.sin(dist["d"]), 0.0)
    return amp * np.sin(dist["s"]) * wall_envelope(dist["s"], wall_sigma)


def shell_profile(d, d_shell, edge_sigma):
    """erfc((d - d_shell) / (sqrt 2 edge_sigma)) / 2: 1 near the wall, a
    Gaussian-edged step to 0 at d_shell (edge_sigma <= 0: hard step)"""
    if edge_sigma <= 0:
        return (d <= d_shell).astype(float)
    return 0.5 * erfc((d - d_shell) / (np.sqrt(2.0) * edge_sigma))


def ic_hollow(dist, amp, d_shell, edge_sigma=3.0, amp_seed=0.0, wall_sigma=2.0):
    S = shell_profile(dist["d"], d_shell, edge_sigma)
    base = np.sin(dist["s"]) * wall_envelope(dist["s"], wall_sigma)
    if wall_sigma <= 0:
        base = np.where(dist["inside"], np.sin(dist["d"]), 0.0)
    return amp * base * S + amp_seed * base * (1.0 - S), S


def _legacy_sq_dist(X, Y, a, b, nmx=256):
    """2025 squared distance to nmx sampled wall points, vectorized by rows
    (identical arithmetic to the 2025 per-pixel loop: the same elementwise
    expression, then the minimum over the same samples)."""
    q = 2 * np.pi * np.arange(1, nmx + 1, 1) / nmx
    bdry = np.vstack((a * np.cos(q), b * np.sin(q)))
    imx, jmx = np.shape(X)
    rho = np.zeros((imx, jmx))
    for ii in range(imx):
        D = (X[ii, :, None] - bdry[0, None, :]) ** 2 + (Y[ii, :, None] - bdry[1, None, :]) ** 2
        rho[ii, :] = np.min(D, axis=1)
    return rho, bdry


def ellipse_init_legacy(X, Y, a, b, amp):
    """2025 eikonal_ellipse.ellipse_init, verbatim except the vectorized
    distance loop. Valid on the legacy grid only (index-scaled smoothing)."""
    rho, bdry = _legacy_sq_dist(X, Y, a, b)
    imx, jmx = np.shape(X)
    kx = (np.pi / a) * fftfreq(jmx, 1.0 / jmx)
    ky = (np.pi / b) * fftfreq(imx, 1.0 / imx)
    xi, eta = np.meshgrid(kx, ky)
    rho = ifft2(np.exp(-(xi ** 2 + eta ** 2)) * fft2(rho))
    return np.real(amp * np.sin(np.sqrt(rho))), bdry


def ellipse_shell_init_legacy(X, Y, Lx, Ly, beta, amp=0.1, start_eik=None, hat_scale=1.0, hat_gap=0.5):
    """2025 eikonal_ellipse_hollow.ellipse_shell_init, verbatim except the
    vectorized distance loop. Valid on the legacy grid only."""
    a = beta * Lx
    b = beta * Ly
    rho, bdry = _legacy_sq_dist(X, Y, a, b)
    imx, jmx = np.shape(X)
    kx = (np.pi / a) * fftfreq(jmx, 1.0 / jmx)
    ky = (np.pi / b) * fftfreq(imx, 1.0 / imx)
    xi, eta = np.meshgrid(kx, ky)
    ellipse_ramp = 0.5 * (1 + np.tanh(
        np.sqrt(Lx ** 2 + Ly ** 2) * (beta - np.sqrt((X / Lx) ** 2 + (Y / Ly) ** 2)) / 2.0))
    rho = ifft2(np.exp(-(xi ** 2 + eta ** 2)) * fft2(rho))
    rhohat = np.real(np.sqrt(rho))
    if start_eik is None:
        ind_Y = np.argmin(np.abs(Y[:, 0]))
        ind_X = np.argmin(np.abs(X[0, :] - a * 0.9))
    else:
        ind_X = np.argmin(np.abs(X[0, :] - start_eik))
        ind_Y = np.argmin(np.abs(Y[:, 0]))
    rho_contour_level = rhohat[ind_Y, ind_X]
    hat = 0.5 * (1 + np.tanh((2.0 / hat_scale) * (rhohat * ellipse_ramp - (rho_contour_level - hat_gap))))
    u_eik = np.real(amp * np.sin(np.sqrt(np.maximum(rho, 0.0))))
    shell_indicator_ = 1.0 - hat
    return u_eik * shell_indicator_, bdry


def build_ic(kind, X, Y, dist, grid, amp=0.1, d_shell=20.0, edge_sigma=3.0, amp_seed=0.0, wall_sigma=2.0):
    """Returns (u0, info dict). For the hollow start, info["hole"] marks the
    points more than 6 edge widths inside the shell edge."""
    if kind == "eikonal":
        return ic_eikonal(dist, amp, wall_sigma), {}
    if kind == "hollow":
        u0, S = ic_hollow(dist, amp, d_shell, edge_sigma, amp_seed, wall_sigma)
        return u0, dict(hole=dist["inside"] & (dist["d"] > d_shell + 6.0 * max(edge_sigma, 0.0)))
    if kind in ("eikonal_legacy", "hollow_legacy"):
        if grid["grid"] != "legacy":
            raise ValueError("ic=%r needs grid='legacy' (its smoothing is written in grid-index "
                             "wavenumbers)" % kind)
        if kind == "eikonal_legacy":
            u0, _ = ellipse_init_legacy(X, Y, grid["A"], grid["B"], amp)
            return u0, {}
        u0, _ = ellipse_shell_init_legacy(X, Y, grid["Lx"], grid["Ly"], grid["beta"], amp=amp)
        return u0, {}
    raise ValueError("unknown ic %r" % (kind,))


# ---------------------------------------------------------------------------
# Spectral filter (the band core.integrate_sh removes), for reporting
# ---------------------------------------------------------------------------

def dealias_filter(F):
    ny, nx = F.shape
    ind = dealias_mask(nx, ny)
    Fh = fft2(F)
    Fh[ind] = 0
    return np.real(ifft2(Fh))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

DEFAULTS = dict(grid="square", A=210.0, B=70.0, dx=0.45, margin=15.0,
                Lx=None, Ly=None, Nx=None, Ny=None, beta=0.45,
                R0=0.5, ramp="distance", ramp_w=2.0,
                ic="eikonal", amp=0.1, wall_sigma=2.0, d_shell=20.0, edge_sigma=3.0, amp_seed=0.0,
                h=0.5, tmax=100.0, nsave=10, t_save_window=None,
                stop_tol=0.0, chunk_T=0.0, wall_margin=2.0, axis_band=1.5,
                energy=True)


def _bulk_rate(u0, u1, dt, mask):
    if mask.sum() < 10:
        return np.nan
    a = np.sqrt(np.mean((u1[mask] - u0[mask]) ** 2))
    b = np.sqrt(np.mean(u0[mask] ** 2))
    return float(a / max(b, 1e-300) / dt)


def _integrate(u0, R, Lx, Ly, h, tmax, nsave, energy, t_save_window, quiet):
    nmax = int(round(tmax / h))
    if nmax < 2:
        raise ValueError("tmax / h = %d steps; at least 2 are needed (core.integrate_sh prints t "
                         "before assigning it when nmax = 1)" % nmax)
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            return integrate_sh(u0, R, Lx, Ly, h, tmax, nsave, energy=energy, t_save_window=t_save_window)
    return integrate_sh(u0, R, Lx, Ly, h, tmax, nsave, energy=energy, t_save_window=t_save_window)


def solve_sh_ellipse(cfg, u_init=None, t_offset=0.0, quiet=True, log=print):
    """Build grid, ramp and initial condition from cfg (see DEFAULTS), step
    with core.integrate_sh, and return a dict:
      tt, x, y, u (ny, nx, nframes) float64, E_total (nframes,) or None,
      e (ny, nx, nframes) or None, R (unfiltered), R_filtered, u0 (before the
      integrator's filter), dist, regions, grid, flags (list of str),
      stop (str), metadata (dict, JSON-able).
    u_init: start from this field (restart); t_offset is added to tt.
    Chunked mode (cfg chunk_T > 0): integrate in chunks of chunk_T with
    nsave frames per chunk, restarting from the last frame, until t >= tmax
    or (stop_tol > 0) the bulk rate between the last two frames falls below
    stop_tol per unit time."""
    c = dict(DEFAULTS)
    c.update({k: v for k, v in cfg.items() if v is not None or k not in DEFAULTS})
    flags = []
    t_build = time.time()
    G = make_grid(c["grid"], A=c["A"], B=c["B"], dx=c["dx"], margin=c["margin"],
                  Lx=c["Lx"], Ly=c["Ly"], Nx=c["Nx"], Ny=c["Ny"], beta=c["beta"])
    x, y = G["x"], G["y"]
    X, Y = np.meshgrid(x, y)
    E = EllipseGeom(G["A"], G["B"])
    dist = wall_distance(E, X, Y)
    regions = region_masks(dist, c["wall_margin"], c["axis_band"])
    legacy_box = (G["Lx"], G["Ly"], G["beta"]) if G["grid"] == "legacy" else None
    R = build_ramp(c["ramp"], X, Y, dist, G["A"], G["B"], c["R0"], c["ramp_w"], legacy_box=legacy_box)
    ic_info = {}
    if u_init is not None:
        u0 = np.asarray(u_init, dtype=float)
        if u0.shape != X.shape:
            raise ValueError("u_init shape %s does not match the grid %s" % (u0.shape, X.shape))
        ic_used = "restart"
    else:
        u0, ic_info = build_ic(c["ic"], X, Y, dist, G, amp=c["amp"], d_shell=c["d_shell"],
                               edge_sigma=c["edge_sigma"], amp_seed=c["amp_seed"], wall_sigma=c["wall_sigma"])
        ic_used = c["ic"]
    R_f = dealias_filter(R)
    u0_f = dealias_filter(u0)
    t_build = time.time() - t_build

    # frame bookkeeping checks (loud flags, not blocks)
    h, tmax, nsave = float(c["h"]), float(c["tmax"]), int(c["nsave"])
    chunk_T = float(c["chunk_T"] or 0.0)
    span = chunk_T if chunk_T > 0 else tmax
    nmax = int(round(span / h))
    if abs(nmax * h - span) > 1e-9 * max(1.0, span):
        flags.append("LOUD FLAG: %s / h = %.6g is not an integer; the last step lands at %.6g"
                     % ("chunk_T" if chunk_T > 0 else "tmax", span / h, nmax * h))
    if c["t_save_window"] is None and nmax % nsave != 0:
        flags.append("LOUD FLAG: steps per %s (%d) not divisible by nsave (%d); frames are every %d steps "
                     "and the last frame is at %.6g, not %.6g" % ("chunk" if chunk_T > 0 else "run", nmax,
                                                                 nsave, nmax // nsave,
                                                                 (nmax // nsave) * nsave * h, span))

    t_run = time.time()
    stop = "tmax"
    if chunk_T <= 0:
        tt, uu, ee, xi, eta, ind, Rf_core = _integrate(u0, R, G["Lx"], G["Ly"], h, tmax, nsave,
                                                       c["energy"], c["t_save_window"], quiet)
        tt = tt + t_offset
    else:
        frames, times, energies = [], [], []
        u_cur = u0
        t_done = 0.0
        k = 0
        while t_done < tmax - 1e-9:
            T_here = min(chunk_T, tmax - t_done)
            tt_c, uu_c, ee_c, xi, eta, ind, Rf_core = _integrate(u_cur, R, G["Lx"], G["Ly"], h, T_here,
                                                                 nsave, c["energy"], None, quiet)
            j0 = 0 if k == 0 else 1          # frame 0 of later chunks repeats the restart frame
            for j in range(j0, uu_c.shape[-1]):
                frames.append(uu_c[..., j].copy())
                times.append(t_offset + t_done + tt_c[j])
                energies.append(None if ee_c is None else ee_c[..., j].copy())
            u_cur = uu_c[..., -1].copy()
            t_done += tt_c[-1]
            k += 1
            log("  chunk %d done: t = %.4g (%.1f s so far)" % (k, t_offset + t_done, time.time() - t_run))
            if c["stop_tol"] > 0 and len(frames) >= 2:
                r = _bulk_rate(frames[-2], frames[-1], times[-1] - times[-2], regions["bulk"])
                if np.isfinite(r) and r < c["stop_tol"]:
                    stop = "stop_tol (bulk rate %.3g < %.3g per unit time at t = %.4g)" % (r, c["stop_tol"],
                                                                                      times[-1])
                    break
            if tt_c[-1] <= 0:
                flags.append("LOUD FLAG: a chunk advanced zero time; stopping")
                stop = "error"
                break
        tt = np.array(times)
        uu = np.stack(frames, axis=-1)
        ee = None if (not c["energy"] or energies[0] is None) else np.stack(energies, axis=-1)
    t_run = time.time() - t_run

    dxdy = G["dx"] * G["dy"]
    E_total = None if ee is None else ee.sum(axis=(0, 1)) * dxdy
    gb_row = int(np.argmin(np.abs(y)))
    md = dict(module=MODULE_VERSION, grid=G["grid"], A=G["A"], B=G["B"], Lx=G["Lx"], Ly=G["Ly"],
              Nx=G["nx"], Ny=G["ny"], dx=G["dx"], dy=G["dy"], margin=G.get("margin"), beta=G.get("beta"),
              Xm=E.Xm, R0=c["R0"], ramp=c["ramp"], ramp_w=c["ramp_w"], ic=ic_used, amp=c["amp"],
              wall_sigma=c["wall_sigma"], d_shell=c["d_shell"], edge_sigma=c["edge_sigma"],
              amp_seed=c["amp_seed"], h=h, tmax=tmax, nsave=nsave,
              t_save_window=c["t_save_window"], chunk_T=chunk_T, stop_tol=c["stop_tol"], t_offset=float(t_offset),
              stop=stop, gb_row_in_crop=gb_row, wall_margin=c["wall_margin"], axis_band=c["axis_band"],
              seconds_build=round(t_build, 2), seconds_run=round(t_run, 2))
    return dict(tt=np.asarray(tt, float), x=x, y=y, u=uu, e=ee, E_total=E_total, R=R, R_filtered=R_f,
                R_core=Rf_core, u0=u0, u0_filtered=u0_f, dist=dist, regions=regions, grid=G,
                ic_info=ic_info, flags=flags, stop=stop, metadata=md)
