"""
knee_sweep_common.py  (v1.2)
----------------------------
Shared plumbing for self_dual_sindy_sweep.py and knee_balance_sweep.py:
run discovery, stem/metadata parsing, final-frame loading, the three
extraction routes (uhu, riesz, demod) behind one interface, the
x-averaged knee profile, the synthetic-knee planter on the data grid,
and the tables of record. The frozen method layer (t1_common,
uhu_smoother, field_diagnostics_viz) is imported, never edited.

Changelog (measured justifications, all on the sandbox copies of the
shallow-sweep files Rp5/R1p0 mu 0.325..0.475 and the 0312 knee files):

v1.2
  - x-window policy 'trim' (window_frame / trim_to_periods): the crop
    is cut to an integer number of stripe periods and no taper is
    applied. The leakage of every FFT-based route is set by the phase
    jump at the periodic seam, pi for the 4.5-period crop; 228 of the
    256 columns hold 4.008 periods, a jump of 0.049 rad. Measured on
    the mu 0.375 R 0.5 crop, planted beta = 1: uHu 0.807 (periodic
    512-column reference 0.810, Tukey 0.5 0.809), demod 0.9999 with R^2
    1.0000 (Tukey 0.5: 0.995), Riesz pi/2 1.073; data frame uHu 0.866,
    demod 1.103, both equal to the Tukey run. 180 usable columns
    against 128, and no smoothing halo to blank in x. The crop stays
    aperiodic in y (lower-knee crop: k2 = +mu on one edge, -mu on the
    other), so the y taper and clear-of-taper distance are unchanged.
    Default remains 'tukey' (run of record); the stamp carries
    xwin-trim when set.

v1.1
  - demod_sm route (demod k fields convolved with the uHu Gaussian):
    reproduces the uHu slope within 0.5 to 6 percent and its R^2 to the
    third decimal at shallow angles on all eight frames; the uHu result
    is the kernel-broadened demod result.
  - opt-in riesz_04 and riesz_1 routes for the magnitude-width question
    (0.4 coupled: R^2 0.03 to 0.07; 1.0: 0.77 to 0.82; pi/2 best tested).
  - profile derivatives (w_y, w_yy) and 2D extras (|grad w|^2, Lap w,
    curl k, J) exported for the extended library.
  - far-field rows for the renormalization taken between the band edge
    and the clear-of-taper distance when at least 8 such rows exist.

v1.0
  - x taper (Tukey, alpha 0.5) applied to u before EVERY route, and only
    the flat central columns enter the x-average. Reason: the sweep
    crops hold 4.5 stripe periods (256 of 512 columns of an 18-lambda_x
    box, lambda_x = half a period), so they are not periodic in x, and
    every FFT-based operator rings. Measured on a planted beta = 1 knee
    on the data grid, uHu sigma = 1.571: y taper only gives far-field
    w = +0.18 and profile R^2 = 0.33; with the x taper, c = 0.795 and
    R^2 = 1.000, identical to the same knee on a 512^2 periodic box
    with the old-codebase ramp. On the 0312 files themselves (12.00
    periods, periodic) a 4.5-period window without the taper drops the
    pointwise R^2 to 0.11 (mu 0.4); with the taper c = 0.857, R^2 = 0.994,
    the periodic-box value. Demod: k1 ripple 0.035 -> 0.008, regression
    slope on the planted knee 0.997 -> 0.999.
  - Fit band |y - yc| < band_w / mu with band_w = 3.0 (sech^2 down to
    2.5e-3), capped by the route's clear-of-taper distance
    Ly/2 - y_frac Ly - 3 sigma. The record's tanh band 4.5/(0.9 mu)
    reaches 15.4 at mu = 0.325 on a 16-unit half box, inside the taper
    zone of the smoothed routes; with the wide band uHu at mu = 0.375
    read R^2 = 0.71, with this band 0.995.
  - Far-field renormalization of w (subtract the mean of wbar over the
    outer quarter of the band, both sides) so that w = |k|^2 - |k|_ff^2.
    The offset is recorded (w_ff) and the constant library column then
    reports any residual offset in div k rather than the instrument's
    |k| gain. Switchable (ff_renorm=False).
  - The tanh estimator of record (beta_knee_fit.profile_estimators) is
    imported and run on the UNTAPERED u so the record table reproduces
    to the digit (checked: all eight values).
"""
import os
import re
import sys
import json
import glob
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import hilbert
from scipy.signal.windows import tukey

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------
# method layer resolution (frozen; import only)
# ---------------------------------------------------------------------

DEFAULT_METHOD_DIRS = [
    str(_HERE.parent / "src" / "op_extract"),
    str(_HERE.parent / "src" / "utils"),
    str(_HERE.parent / "src"),
    str(_HERE),
    "/Users/edwardmcdugald/patterns/src/op_extract",
    "/Users/edwardmcdugald/patterns/src/utils",
    "/Users/edwardmcdugald/patterns/src",
    "/Users/edwardmcdugald/patterns/pipelines",
]

_METHOD = {}


def setup_method_layer(extra_dirs=()):
    """Insert the method-layer directories on sys.path and import the
    frozen modules once. Returns a dict of modules."""
    if _METHOD:
        return _METHOD
    for d in list(extra_dirs) + DEFAULT_METHOD_DIRS:
        if d and os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
    try:
        import t1_common as tc
        import uhu_smoother as us
        import field_diagnostics_viz as fdv
        import beta_knee_fit as bkf
    except ModuleNotFoundError as exc:
        raise ImportError(
            "could not import the frozen method layer (%s). Searched:\n  %s\n"
            "Fix: pass --method-dirs pointing at src/op_extract, src/utils, "
            "src, and the directory holding beta_knee_fit.py."
            % (exc, "\n  ".join(list(extra_dirs) + DEFAULT_METHOD_DIRS)))
    _METHOD.update(dict(tc=tc, us=us, fdv=fdv, bkf=bkf))
    try:
        import paper_style
        _METHOD["paper_style"] = paper_style
    except ImportError:
        _METHOD["paper_style"] = None
    return _METHOD


def use_house_style():
    ps = setup_method_layer().get("paper_style")
    if ps is not None:
        ps.use_paper_style()
    else:
        import matplotlib as mpl
        mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                             "font.size": 11, "legend.frameon": False,
                             "savefig.dpi": 300, "savefig.bbox": "tight",
                             "image.origin": "lower"})


def panel_label(ax, s, color="k"):
    ps = setup_method_layer().get("paper_style")
    if ps is not None:
        ps.panel_label(ax, s, color=color)
    else:
        ax.text(0.02, 0.98, s, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="left", color=color)


# ---------------------------------------------------------------------
# tables of record
# ---------------------------------------------------------------------

# demod tanh-fit beta on the final frames (Writeup 1, Table beta_measured;
# session-6 handoff item 2). Same estimator, same frames: must reproduce.
RECORD_BETA = {
    (0.325, 0.5): 1.094, (0.375, 0.5): 1.123, (0.425, 0.5): 1.169, (0.475, 0.5): 1.219,
    (0.325, 1.0): 1.056, (0.375, 1.0): 1.071, (0.425, 1.0): 1.103, (0.475, 1.0): 1.141,
}
RECORD_LAW_C = {0.5: 0.9, 1.0: 0.55}      # beta ~ 1 + c(R) mu^2

# Console calibration block (extraction_comparison.py, Rp5/R1p0 runs of
# record): TANH-instrument response of each route to a planted knee.
# Reference only; the SINDy scripts calibrate through their own
# regression response, which is a different reading of the same fields.
CONSOLE_TANH_RESPONSE = {
    # (mu, b_true): {route: reading}
    (0.375, 0.9): {"riesz": 0.6489, "riesz_pi2": 0.7767, "uhu": 0.6801, "demod": 0.8849},
    (0.375, 1.0): {"riesz": 0.7113, "riesz_pi2": 0.8470, "uhu": 0.7425, "demod": 0.9813},
    (0.375, 1.1): {"riesz": 0.7724, "riesz_pi2": 0.9153, "uhu": 0.7989, "demod": 1.0787},
    (0.475, 0.9): {"riesz": 0.7052, "riesz_pi2": 0.7660, "uhu": 0.7457, "demod": 0.8955},
    (0.475, 1.0): {"riesz": 0.7753, "riesz_pi2": 0.8384, "uhu": 0.7964, "demod": 0.9986},
    (0.475, 1.1): {"riesz": 0.8402, "riesz_pi2": 0.9068, "uhu": 0.8419, "demod": 1.1021},
}


def record_lookup(mu, R, tol=1e-3):
    for (m, r), v in RECORD_BETA.items():
        if abs(m - mu) < tol and abs(r - R) < tol:
            return v
    return float("nan")


# ---------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------

ROUTE_ORDER = ["uhu", "riesz", "demod", "riesz_pi2"]
ROUTES = {
    "uhu":       dict(kind="uhu", sigma=1.571, label="uHu $\\sigma=1.571$"),
    "riesz":     dict(kind="riesz", sigma_dir=0.4, sigma_avg=np.pi,
                      label="Riesz 0.4/$\\pi$"),
    "riesz_pi2": dict(kind="riesz", sigma_dir=0.4, sigma_avg=np.pi / 2,
                      label="Riesz 0.4/$\\pi/2$"),
    "riesz_04":  dict(kind="riesz", sigma_dir=0.4, sigma_avg=0.4,
                      label="Riesz 0.4/0.4 (coupled widths)"),
    "riesz_1":   dict(kind="riesz", sigma_dir=0.4, sigma_avg=1.0,
                      label="Riesz 0.4/1.0"),
    "demod":     dict(kind="demod", label="demod"),
    # kernel forensics for the regression: the demod k fields convolved
    # with uHu's Gaussian, then read with uHu's band and seam. Measured
    # (v1.1): reproduces the uHu slope within 0.5 to 6 percent, its
    # holdout R^2 to the third decimal at shallow angles, and its
    # intercept, on all eight frames.
    "demod_sm":  dict(kind="demod_sm", sigma=1.571,
                      label="demod smoothed $\\sigma=1.571$"),
}
ROUTE_COLORS = {"uhu": "C2", "riesz": "C3", "riesz_pi2": "C1", "riesz_04": "C4",
                "riesz_1": "C5", "demod": "C0", "demod_sm": "C9"}


def route_sigma(route):
    """Smoothing width that sets the clear-of-taper distance."""
    r = ROUTES[route]
    if r["kind"] in ("uhu", "demod_sm"):
        return float(r["sigma"])
    if r["kind"] == "riesz":
        return float(max(r["sigma_dir"], r["sigma_avg"]))
    return 0.0


def route_seam(route):
    """Rows excluded around the core: the sign-flipped |k2| of the
    director routes never reaches zero there (extraction_comparison)."""
    r = ROUTES[route]
    if r["kind"] in ("uhu", "demod_sm"):
        return max(0.8, 0.6 * r["sigma"])
    if r["kind"] == "riesz":
        return max(0.8, 0.6 * r["sigma_dir"])
    return 0.0


# ---------------------------------------------------------------------
# discovery and loading
# ---------------------------------------------------------------------

_MU_RE = re.compile(r"mu(\d+(?:\.\d+)?)")
_R_RE = re.compile(r"(?:^|_)R(\d+)p(\d+)(?:_|$)")


def _parse_stem(stem):
    mu = R = None
    m = _MU_RE.search(stem)
    if m:
        mu = float(m.group(1))
    m = _R_RE.search(stem)
    if m:
        R = float("%s.%s" % (m.group(1), m.group(2)))
    return mu, R


def read_meta(path):
    d = np.load(path, allow_pickle=True)
    meta = {}
    for key in ("metadata_json", "sh_meta_json"):
        if key in d.files:
            try:
                meta.update(json.loads(str(d[key])))
            except Exception:
                pass
    stem = Path(path).stem
    mu_s, R_s = _parse_stem(stem)
    mu = float(meta.get("mu", mu_s if mu_s is not None else np.nan))
    R = float(meta.get("Rscale", R_s if R_s is not None else np.nan))
    ny = d["u"].shape[0]
    jc = int(meta.get("gb_row_in_crop", ny // 2))
    nt = d["u"].shape[2] if d["u"].ndim == 3 else 1
    return dict(path=str(path), stem=stem, mu=mu, R=R, jc=jc, n_frames=nt,
                has_ramp=("ramp" in d.files))


def discover_runs(roots, pattern="*sh_pgb_zigzag*mu*.npz", mu_min=None,
                  mu_max=None, R_filter=None, exclude=("_uhu_", "_phase_",
                                                        "_riesz_")):
    """Walk each root for raw-u knee files. Order-parameter products of
    the frozen pipelines are excluded by stem substring."""
    found = []
    for root in roots:
        for p in sorted(glob.glob(os.path.join(root, "**", pattern),
                                  recursive=True)):
            if any(tag in Path(p).stem for tag in exclude):
                continue
            try:
                info = read_meta(p)
            except Exception as exc:
                print("   [discover] skipping %s (%s)" % (p, exc))
                continue
            if mu_min is not None and info["mu"] < mu_min - 1e-9:
                continue
            if mu_max is not None and info["mu"] > mu_max + 1e-9:
                continue
            if R_filter and not any(abs(info["R"] - r) < 1e-6 for r in R_filter):
                continue
            found.append(info)
    found.sort(key=lambda i: (i["R"], i["mu"]))
    return found


def load_frame(path, frame=-1):
    d = np.load(path, allow_pickle=True)
    u = d["u"]
    if u.ndim == 3:
        idx = frame if frame >= 0 else u.shape[2] + frame
        u = u[..., idx]
    else:
        idx = 0
    return d["x"].astype(float), d["y"].astype(float), np.asarray(u, float), int(idx)


# ---------------------------------------------------------------------
# preprocessing: tapers and column window
# ---------------------------------------------------------------------

def x_taper(nx, alpha):
    return tukey(nx, alpha) if alpha > 0 else np.ones(nx)


def flat_columns(nx, alpha, margin=24):
    """Columns where the Tukey window is identically 1 (alpha > 0), or
    the record's edge_cols margin when there is no taper."""
    if alpha <= 0:
        return slice(margin, nx - margin)
    half = int(np.ceil(0.5 * alpha * nx))
    return slice(half, nx - half)


def carrier_k1(mu):
    """Imposed far-field x wavenumber of the zigzag box, sqrt(1 - mu^2):
    pgb_zigzag builds Lx from it, so the relaxed periodic box keeps it
    exactly (measured: k1 dx = 18 pi / 512 on every sweep file; the
    demod reading is 0.33 percent lower, the central-difference factor
    sin(k1 dx)/(k1 dx) plus the row mean over the knee)."""
    return float(np.sqrt(1.0 - mu ** 2))


def trim_to_periods(u, x, k1, max_jump=0.1):
    """Trim the crop along x to the largest run of columns holding an
    integer number of stripe periods, so the periodic extension closes
    with a seam jump below max_jump radians and no window is needed.

    The period is 2 pi / (k1 dx) columns (56.89 on the sweep crops), so
    no trim closes exactly; for n = 1, 2, ... the candidate width is
    N = round(2 pi n / (k1 dx)) with seam jump k1 N dx - 2 pi n, and the
    widest N <= nx with |jump| <= max_jump wins (the crop's own jump at
    4.5 periods is pi). The kept columns are centred in the crop. A
    frame already holding an integer number of periods (the 0312 files,
    12.00 periods in 512 columns) is returned unchanged.
    Returns u_t, x_t and a dict (n_periods, n_cols, jump_rad, start)."""
    nx = len(x); dx = x[1] - x[0]
    per = 2.0 * np.pi / (k1 * dx)
    best = None
    for n in range(1, int(np.floor(nx / per)) + 1):
        N = int(round(n * per))
        if N > nx:
            continue
        jump = k1 * N * dx - 2.0 * np.pi * n
        if abs(jump) <= max_jump and (best is None or N > best[1]):
            best = (n, N, jump)
    if best is None:
        # fall back to the closest-closing candidate regardless of the cap
        cands = [(abs(k1 * int(round(n * per)) * dx - 2 * np.pi * n), n)
                 for n in range(1, int(np.floor(nx / per)) + 1)]
        j, n = min(cands); N = int(round(n * per))
        best = (n, N, k1 * N * dx - 2 * np.pi * n)
        warnings.warn("trim_to_periods: no trim closes within %.3f rad; using %d periods "
                      "(jump %.3f rad)" % (max_jump, n, best[2]))
    n, N, jump = best
    start = (nx - N) // 2
    info = dict(n_periods=int(n), n_cols=int(N), jump_rad=float(jump), start=int(start),
                period_cols=float(per), n_cols_in=int(nx))
    return u[:, start:start + N], x[start:start + N], info


def window_frame(u, x, mu, x_window="tukey", alpha_x=0.5, max_jump=0.1):
    """Apply the x-window policy to a loaded frame. 'tukey': the frame is
    untouched here and the Tukey taper of width alpha_x is applied
    inside extract_k (run of record). 'trim': the frame is cut to an
    integer number of periods (trim_to_periods) and the effective taper
    is 0, i.e. a box, with the record's 24-column margin as the flat
    region. Returns u, x, alpha_x_effective, trim_info (None for tukey).
    Measured on the mu 0.375 R 0.5 crop, planted beta = 1: trim to 228
    columns (4.008 periods, jump 0.049 rad) gives uHu 0.807 against the
    periodic 512-column reference 0.810 and Tukey 0.5 0.809; demod
    0.9999 with R^2 1.0000 (Tukey 0.5: 0.995); 180 usable columns
    against 128."""
    if x_window == "trim":
        u_t, x_t, info = trim_to_periods(u, x, carrier_k1(mu), max_jump)
        return u_t, x_t, 0.0, info
    if x_window != "tukey":
        raise ValueError("x_window must be 'tukey' or 'trim', got %r" % (x_window,))
    return u, x, float(alpha_x), None


def box_lengths(x, y):
    return (x[-1] - x[0] + (x[1] - x[0]), y[-1] - y[0] + (y[1] - y[0]))


def y_clear_distance(y, y_frac, route):
    """Rows farther than this from the box edge are untouched by the
    y taper and its smoothing halo (3 sigma)."""
    Ly = box_lengths(np.zeros(2), y)[1]
    return 0.5 * Ly - y_frac * Ly - 3.0 * route_sigma(route)


# ---------------------------------------------------------------------
# extraction (one interface for the three routes)
# ---------------------------------------------------------------------

def extract_k(route, u, x, y, jc, alpha_x=0.5, y_frac=0.06):
    """Return oriented (k1, k2) on the full grid: k1 >= 0 and k2 odd
    about the core row, matching the lower-boundary sign convention of
    the record. The x taper is applied to every route; the y edge taper
    only to the spectral routes (extraction_comparison convention)."""
    M = setup_method_layer()
    tc, us, fdv = M["tc"], M["us"], M["fdv"]
    r = ROUTES[route]
    nx = len(x)
    ux = u * x_taper(nx, alpha_x)[None, :]
    Lx, Ly = box_lengths(x, y)
    s = np.sign(y[:, None] - y[jc])
    s[s == 0] = 1.0
    if r["kind"] in ("demod", "demod_sm"):
        dx, dy = x[1] - x[0], y[1] - y[0]
        ua = hilbert(ux, axis=1)
        den = np.abs(ua) ** 2 + 1e-12
        k1 = np.imag(np.conj(ua) * np.gradient(ua, dx, axis=1)) / den
        k2 = np.imag(np.conj(ua) * np.gradient(ua, dy, axis=0)) / den
        if r["kind"] == "demod_sm":
            yt = tc.edge_taper(x, y, frac=y_frac)
            k1 = fdv.macro_iso(k1 * yt, r["sigma"], Lx, Ly)
            k2 = fdv.macro_iso(k2 * yt, r["sigma"], Lx, Ly)
        return k1, k2
    uy = ux * tc.edge_taper(x, y, frac=y_frac)
    if r["kind"] == "uhu":
        sm = us.make_gaussian_smoother(r["sigma"], Lx, Ly, fdv.macro_iso)
        ops = us.compute_uhu_ops_smoother(uy, x, y, sm, fdv.spectral_deriv,
                                          np.ones_like(u))
        return np.abs(ops["k1_orig"]), np.abs(ops["k2_orig"]) * s
    if r["kind"] == "riesz":
        rk = tc.riesz_k_field(uy, x, y, Lx, Ly, sigma_dir=r["sigma_dir"],
                              sigma_avg=r["sigma_avg"])
        return np.abs(rk["k1"]), np.abs(rk["k2"]) * s
    raise ValueError(route)


# ---------------------------------------------------------------------
# profile and pointwise sample sets
# ---------------------------------------------------------------------

def build_profile(k1, k2, x, y, jc, mu, route, alpha_x=0.5, y_frac=0.06,
                  band_w=3.0, ff_renorm=True, ff_frac=0.25):
    """x-averaged transverse profile of the knee on the flat columns.

    Returns a dict with y (centered on the core), k1bar, k2bar, wbar
    (far-field renormalized when ff_renorm), dk2 = d k2bar / dy, the fit
    band (rows), the far-field offset w_ff, and the band bookkeeping."""
    dy = y[1] - y[0]
    cols = flat_columns(len(x), alpha_x)
    yc = y - y[jc]
    k1b = k1[:, cols].mean(1)
    k2b = k2[:, cols].mean(1)
    wb = (k1[:, cols] ** 2 + k2[:, cols] ** 2).mean(1) - 1.0
    dk2 = np.gradient(k2b, dy)
    seam = route_seam(route)
    y_clear = y_clear_distance(y, y_frac, route)
    band_half = min(band_w / mu, y_clear)
    band = (np.abs(yc) < band_half) & (np.abs(yc) > seam)
    band[:2] = band[-2:] = False
    # far field: rows between the band edge and the clear-of-taper
    # distance when that gap holds at least 4 rows per side, otherwise
    # the outer ff_frac of the band itself
    ff = (np.abs(yc) >= band_half) & (np.abs(yc) < y_clear)
    ff[:2] = ff[-2:] = False
    if ff.sum() < 8:
        ff = band & (np.abs(yc) > (1.0 - ff_frac) * band_half)
    w_ff = float(wb[ff].mean()) if ff.any() else 0.0
    if ff_renorm:
        wb = wb - w_ff
    # profile derivatives for the extended library (central differences on
    # the x-averaged profile; even about the core like div k and w)
    wy = np.gradient(wb, dy)
    wyy = np.gradient(wy, dy)
    return dict(y=yc, k1bar=k1b, k2bar=k2b, wbar=wb, dk2=dk2, band=band,
                w_y=wy, w_yy=wyy,
                ff_rows=ff, w_ff=w_ff, band_half=float(band_half),
                y_clear=float(y_clear), seam=float(seam), cols=cols,
                n_rows=int(band.sum()), k1_ff=float(k1b[ff].mean()) if ff.any() else np.nan,
                m_ff=float(np.abs(k2b[ff]).mean()) if ff.any() else np.nan)


def build_pointwise(k1, k2, x, y, jc, mu, route, alpha_x=0.5, y_frac=0.06,
                    band_w=3.0, ff_renorm=True, ff_frac=0.25):
    """Pointwise 2D samples on band rows x flat columns; w renormalized
    by the same far-field offset as the profile (consistent both sides).
    Secondary domain: on demod the 2-theta ripple is not averaged out
    (planted-knee R^2 ~ 0.2), so read this domain on the smoothed routes."""
    dx, dy = x[1] - x[0], y[1] - y[0]
    prof = build_profile(k1, k2, x, y, jc, mu, route, alpha_x, y_frac,
                         band_w, ff_renorm, ff_frac)
    cols = prof["cols"]
    w = k1 ** 2 + k2 ** 2 - 1.0 - (prof["w_ff"] if ff_renorm else 0.0)
    divk = np.gradient(k1, dx, axis=1) + np.gradient(k2, dy, axis=0)
    M = np.zeros_like(w, dtype=bool)
    M[prof["band"], cols] = True
    Y = np.tile(prof["y"][:, None], (1, len(x)))
    # 2D extras for the extended library (self_dual_sindy_patch conventions)
    k1x, k1y = np.gradient(k1, dx, axis=1), np.gradient(k1, dy, axis=0)
    k2x, k2y = np.gradient(k2, dx, axis=1), np.gradient(k2, dy, axis=0)
    wx, wy = np.gradient(w, dx, axis=1), np.gradient(w, dy, axis=0)
    lap_w = np.gradient(wx, dx, axis=1) + np.gradient(wy, dy, axis=0)
    extras = {"|grad w|^2": (wx ** 2 + wy ** 2)[M], "Lap w": lap_w[M],
              "curl k": (k2x - k1y)[M], "J": (k1x * k2y - k1y * k2x)[M]}
    return dict(w=w[M], divk=divk[M], y=Y[M], mask=M, prof=prof, extras=extras,
                divk_2d=divk, w_2d=w)


# ---------------------------------------------------------------------
# synthetic planter and the estimator of record
# ---------------------------------------------------------------------

def synth_knee_on_grid(x, y, mu, R, beta, jc):
    """beta_knee_fit.synth_knee (analytic RCN knee phase, slaved A1 and
    A3) evaluated on the data grid with the core on row jc."""
    bkf = setup_method_layer()["bkf"]
    return bkf.synth_knee(x, y, mu, R, beta, yc=y[jc])


def tanh_beta_of_record(u, x, y, jc, mu):
    """The demod tanh estimator exactly as archived (untapered u)."""
    bkf = setup_method_layer()["bkf"]
    try:
        b, m_ff, b_reg, r2 = bkf.profile_estimators(u, x, y, jc, mu)
        return float(b), float(m_ff), float(b_reg), float(r2)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


# ---------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------

def linear_fit(X, Y):
    """Y = b + c X by least squares; returns c, b, R^2."""
    A = np.vstack([np.ones(X.size), X]).T
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    b, c = coef
    ss = np.sum((Y - Y.mean()) ** 2)
    r2 = 1.0 - np.sum((Y - A @ coef) ** 2) / (ss + 1e-300)
    return float(c), float(b), float(r2)


def calibrate(reading, pts):
    """Invert a route's response curve: pts = [(b_true, reading), ...]
    -> implied true beta by linear interpolation, linear extrapolation
    outside the planted range. NaN if fewer than two points."""
    pts = [(b, r) for b, r in pts if np.isfinite(r)]
    if len(pts) < 2 or not np.isfinite(reading):
        return float("nan")
    pts.sort(key=lambda t: t[1])
    bs = np.array([b for b, r in pts]); rs = np.array([r for b, r in pts])
    if reading < rs[0]:
        return float(bs[0] + (reading - rs[0]) * (bs[1] - bs[0]) / (rs[1] - rs[0]))
    if reading > rs[-1]:
        return float(bs[-1] + (reading - rs[-1]) * (bs[-1] - bs[-2]) / (rs[-1] - rs[-2]))
    return float(np.interp(reading, rs, bs))


def config_stamp(cfg, script_tag):
    Rs = "R" + "-".join("%gp%s" % (int(r), str(r).split(".")[1]) if r != int(r)
                        else "%d" % r for r in cfg["R_values"]) if cfg.get("R_values") else "Rall"
    mus = "mu%s-%s" % (cfg.get("mu_min", "min"), cfg.get("mu_max", "max"))
    routes = "routes-" + "-".join(cfg["routes"])
    dom = "dom-" + cfg.get("domain", "profile")
    parts = [script_tag, Rs, mus, routes, dom]
    if cfg.get("library", "basic") != "basic":
        parts.append("lib-" + cfg["library"])
    if cfg.get("coef_cap") is not None:
        parts.append("cap%g" % cfg["coef_cap"])
    if cfg.get("x_window", "tukey") != "tukey":
        parts.append("xwin-" + cfg["x_window"])
    return "_".join(parts)


def save_config(out_dir, cfg):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = {}
    for k, v in cfg.items():
        if isinstance(v, (np.floating, np.integer)):
            v = float(v)
        elif isinstance(v, (tuple, list)):
            v = [float(t) if isinstance(t, (np.floating, np.integer)) else t for t in v]
        elif isinstance(v, Path):
            v = str(v)
        clean[k] = v
    with open(out_dir / "config.json", "w") as fh:
        json.dump(clean, fh, indent=2)
