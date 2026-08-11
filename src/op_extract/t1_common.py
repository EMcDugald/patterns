# src/op_extract/t1_common.py
"""
Shared machinery for the T1 director-source test on both analytic fields.

The test holds every stage of the phase pipeline fixed and swaps only the
director:

    exact   k = grad theta, in closed form on the knee and differenced on
            the grid on PN
    uhu     the current extraction, isotropic Gaussian at sigma
    riesz   the doubled-angle smoothed Riesz director at sigma_dir

Both fields are scored with ONE metric set, because the same number answers
both questions. The knee's exact core phase is linear, so the truth's
sawtooth is zero; PN's is a staircase, so at the core row the truth's
sawtooth is 0.75 to 0.86. Comparing a reconstruction's sawtooth against the
truth's own value at the same row therefore reads:

    excess over truth    invented structure   (the knee failure)
    deficit below truth  suppressed structure (the PN failure)
    matching the truth   correct, on either field

Two metrics do not carry over cleanly and are handled explicitly rather than
quietly. Localization needs known defect columns, so it is PN only. The
second-difference ratio divides by the truth's roughness, which is
essentially zero on the knee's straight core, so the absolute value is
reported for both and the ratio only where the denominator is meaningful.

Conventions inherited from the pipeline:
  - upper-half seeds step along -khat, so the Hilbert phase accumulates as
    -theta and the alignment coefficient should come out near -1
  - only the seed-to-core branch of each line is used; along the full
    stitched line the reflected half accumulates with the wrong sign
"""

import importlib
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.special import log_ndtr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _import_first(names, attr):
    """Import `attr` from the first module path that provides it. The repo
    layout has moved between sessions; this avoids hard-coding one."""
    tried = []
    for nm in names:
        try:
            mod = importlib.import_module(nm)
        except Exception as exc:
            tried.append(f"{nm}: {exc}")
            continue
        if hasattr(mod, attr):
            return getattr(mod, attr)
        tried.append(f"{nm}: no attribute {attr}")
    raise ImportError(f"could not import {attr}; tried:\n  " + "\n  ".join(tried))


compute_pgb_phase_from_uhu = _import_first(
    ["op_extract.phase_zigzag", "utils.phase_zigzag", "phase_zigzag"],
    "compute_pgb_phase_from_uhu")
compute_uhu_ops_smoother = _import_first(
    ["op_extract.uhu_smoother", "utils.uhu_smoother", "uhu_smoother"],
    "compute_uhu_ops_smoother")
riesz_director = _import_first(
    ["utils.field_diagnostics_viz", "field_diagnostics_viz"], "riesz_director")
monogenic = _import_first(
    ["utils.field_diagnostics_viz", "field_diagnostics_viz"], "monogenic")
spectral_deriv = _import_first(
    ["utils.field_diagnostics_viz", "field_diagnostics_viz"], "spectral_deriv")
macro_iso = _import_first(
    ["utils.field_diagnostics_viz", "field_diagnostics_viz"], "macro_iso")


# =====================================================================
# small numerics
# =====================================================================

def _log_sinh(z):
    """log sinh z for z > 0 without overflowing at large z."""
    z = np.asarray(z, dtype=float)
    return z + np.log1p(-np.exp(-2.0 * z)) - np.log(2.0)


def edge_taper(x, y, frac=0.04):
    """Cosine taper in y only, over the outer `frac` of the box, so the
    spectral derivatives inside the extraction do not ring off the y seam.
    Identically 1 over the interior, so it never touches the core row."""
    Ly = y[-1] - y[0]
    d = np.minimum(y - y[0], y[-1] - y) / Ly
    w = np.ones_like(y)
    m = d < frac
    w[m] = 0.5 * (1 - np.cos(np.pi * d[m] / frac))
    return np.tile(w[:, None], (1, len(x)))


def exact_k_from_theta(theta, x, y, odd_in_y):
    """
    grad theta by differencing theta on the grid.

    Used for PN, where the closed form carries a 1/sqrt(y) factor on the
    defect columns, singular exactly where the coordinate lines terminate,
    while theta itself is bounded everywhere. This makes the exact arm
    resolution limited rather than analytic, which the report states.

    When theta is odd in y the y-difference is taken separately on the two
    branches, so nothing straddles the 2 m pi jump at the core.
    """
    k1 = np.gradient(theta, x, axis=1)
    k2 = np.empty_like(theta)
    if odd_in_y:
        up, lo = y > 0, y < 0
        if up.sum() > 1:
            k2[up] = np.gradient(theta[up], y[up], axis=0)
        if lo.sum() > 1:
            k2[lo] = np.gradient(theta[lo], y[lo], axis=0)
    else:
        k2 = np.gradient(theta, y, axis=0)
    return k1, k2


# =====================================================================
# the knee
# =====================================================================

def knee_exact(mu, n_periods=8.0, n_spacings=None, N=512, ds=0.15,
               nx_periods=None, ly_factor=2.8, x_margin_periods=1.0, **kw):
    """
    theta = k1 x + log(2 cosh(k2 y)),  u = cos(theta),  k = grad theta in
    closed form (smooth and bounded everywhere, so no differencing needed).

    Sizing is by fundamental periods and closed forms, with no margin
    factor. Previously Lx was 1.45 times an estimate, and at shallow mu the
    estimate was low enough that the lines ran out of the box.

      y_top = 2 n_periods pi / k2       an integer number of vertical
                                        periods of the far-field wave
      contour span = log(2 cosh(k2 y_top)) / k1
      drift(eps)   = (k1/k2^2) log( sinh(k2 y_top) / sinh(k2 eps) )

    The drift is the horizontal displacement of a coordinate line falling
    from y_top to the core. It diverges logarithmically as eps goes to zero,
    because k rotates to horizontal at the core and the line creeps
    sideways; eps is set by the step size and the divergence is slow, about
    5 percent per factor of 16 in ds.

    Lx holds the seeded arc plus the drift plus a margin, since the arc is
    clipped to the requested coverage rather than run end to end. Pass
    nx_periods to set Lx directly as a count of horizontal periods instead.
    """
    k1, k2 = float(np.sqrt(1 - mu**2)), float(mu)
    y_top = 2.0 * n_periods * np.pi / k2
    Ly = ly_factor * y_top
    Lsp = np.pi / k1                      # PN defect spacing at the same mu
    cover = (n_spacings if n_spacings else 8.0) * Lsp
    drift = float((k1 / k2**2) * (_log_sinh(k2 * y_top) - _log_sinh(k2 * ds)))
    contour_span = float(np.logaddexp(k2 * y_top, -k2 * y_top) / k1)

    # Lx must hold the whole level set, not just the arc that survives
    # clipping: the contour is built over the full height first, and a box
    # that only fits the useful arc drops most of the rows and leaves the
    # centring with nothing to measure.
    if nx_periods is None:
        Lx = contour_span + drift + 2.0 * x_margin_periods * (2 * np.pi / k1)
    else:
        Lx = nx_periods * (2 * np.pi / k1)

    x = np.linspace(-Lx / 2, Lx / 2, N, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, N, endpoint=False)   # includes y = 0
    X, Y = np.meshgrid(x, y)

    theta = k1 * X + np.logaddexp(k2 * Y, -k2 * Y)
    return {"field": "knee", "odd_in_y": False, "x": x, "y": y, "X": X,
            "Y": Y, "u": np.cos(theta), "theta": theta,
            "k1": np.full_like(X, k1), "k2": k2 * np.tanh(k2 * Y),
            "k_source": "closed form",
            "k1_ff": k1, "k2_ff": k2, "Lx": Lx, "Ly": Ly, "mu": mu,
            "y_top": y_top, "drift": drift, "contour_span": contour_span,
            "cover": cover, "Lsp": Lsp, "xd": np.array([]), "nd": 0,
            "n_periods": n_periods,
            "iy_core": int(np.argmin(np.abs(y)))}


# =====================================================================
# Pismen-Nepomnyashchy
# =====================================================================

def log_psi_pn(X, Y, xd, y_floor=1e-6):
    """
    log psi with no cancellation, exact at any number of defects.

    The literal form is psi = a + sum_j delta_j erf((x - c_j)/sqrt(2|y|))
    with a = (1 + e^{-nd pi})/2, whose smallest value is e^{-nd pi} but
    which is assembled from terms of size 1/2. Past about eleven defects the
    right end of the staircase is roundoff.

    Substituting erf(z) = 1 - 2p with p = erfc(z)/2 and telescoping the
    weights gives a sum of positive terms,

        psi = e^{-nd pi} + (1 - e^{-pi}) sum_j e^{-j pi} p_j

    accumulated here with logaddexp, taking log p_j from log_ndtr so the far
    tail does not underflow. As y goes to 0+, p_j is 1 exactly when x < c_j,
    the sum telescopes and psi collapses to e^{-m pi}, the staircase.
    """
    nd = len(xd)
    w = np.sqrt(2.0 * np.maximum(np.abs(Y), y_floor))
    lg = float(np.log1p(-np.exp(-np.pi)))
    acc = np.full(X.shape, -nd * np.pi, dtype=float)
    for j, c in enumerate(xd):
        acc = np.logaddexp(acc, lg - j * np.pi
                           + log_ndtr(-np.sqrt(2.0) * (X - c) / w))
    return acc


def pn_exact(mu, n_periods=8.0, nd=None, N=768, ly_factor=2.8,
             n_spacings=None, x_margin_spacings=2.0, **kw):
    """
    Exact PN field. theta = y - sign(y) log psi, so

        theta(x, 0+) = m(x) pi        theta(x, 0-) = -m(x) pi

    an exact staircase with a step of pi at each defect. theta is ODD in y
    and jumps by 2 m pi across the core, so every comparison here is done on
    the upper half and the y grid carries a half-cell offset to keep any row
    off y = 0.

    Sizing:
      k2_ff  = 1 - k1^2/2, PN's merged far field, which agrees with the
               knee's sqrt(1 - k1^2) to second order in k1
      y_top  = 2 n_periods pi / k2_ff
      nd     = 10 k1 sqrt(2 y_top) / pi, rounded up to an even number

    The plane-wave zone is a window in y, not an asymptote: below it the erf
    steps have not merged, above it the finite defect row collapses into a
    single step. Both edges come from the smoothing width sqrt(2 y) against
    the spacing and the row half-width, which is where the nd rule comes
    from. The resulting error at y_top is measured by planewave_error, not
    assumed.

    nd is forced even so x = 0 sits midway between defects and the seed
    contour is centred on a plateau rather than a singular column.
    """
    k1 = float(np.sqrt(1.0 - mu**2))
    k2_ff = 1.0 - 0.5 * k1**2
    Lsp = np.pi / k1
    y_top = 2.0 * n_periods * np.pi / k2_ff

    if nd is None:
        nd = 10.0 * k1 * np.sqrt(2.0 * y_top) / np.pi
        nd = int(2 * np.ceil(nd / 2.0))
    nd = max(int(2 * ((int(nd) + 1) // 2)), 2)

    drift = y_top * k1 / k2_ff
    Lx = nd * Lsp + 2.0 * drift + 2.0 * x_margin_spacings * Lsp
    Ly = ly_factor * y_top

    x = np.linspace(-Lx / 2, Lx / 2, N, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, N, endpoint=False) + 0.5 * Ly / N
    if np.any(np.abs(y) < 1e-12):
        raise RuntimeError("a grid row landed on y = 0; theta is odd there")
    X, Y = np.meshgrid(x, y)

    xd = Lsp * (np.arange(nd) - (nd - 1) / 2.0)
    theta = Y - np.sign(Y) * log_psi_pn(X, Y, xd)
    k1g, k2g = exact_k_from_theta(theta, x, y, odd_in_y=True)

    cover = (n_spacings * Lsp) if n_spacings else (0.95 * nd * Lsp)
    return {"field": "pn", "odd_in_y": True, "x": x, "y": y, "X": X, "Y": Y,
            "u": np.cos(theta), "theta": theta, "k1": k1g, "k2": k2g,
            "k_source": "differenced on the grid",
            "k1_ff": k1, "k2_ff": k2_ff, "Lx": Lx, "Ly": Ly, "mu": mu,
            "y_top": y_top, "drift": drift, "contour_span": k2_ff * y_top / k1,
            "cover": cover, "Lsp": Lsp, "xd": xd, "nd": nd,
            "n_periods": n_periods,
            "iy_core": int(np.argmin(np.abs(y - 0.5 * Ly / N)))}


def planewave_error(fld, n_spacings=2.0):
    """Max relative deviation of grad theta from (k1_ff, k2_ff) on the row
    nearest y_top, over a window of n_spacings spacings in x. This is the
    number that says whether nd is large enough on PN; on the knee it is
    exact by construction and reported for symmetry."""
    x, y = fld["x"], fld["y"]
    j = int(np.argmin(np.abs(y - fld["y_top"])))
    m = np.abs(x) <= n_spacings * fld["Lsp"]
    if m.sum() < 8 or j in (0, len(y) - 1):
        return np.nan, np.nan, float(y[j])
    tx = np.gradient(fld["theta"][j], x)[m]
    ty = ((fld["theta"][j + 1] - fld["theta"][j - 1])
          / (y[j + 1] - y[j - 1]))[m]
    return (float(np.max(np.abs(tx / fld["k1_ff"] - 1.0))),
            float(np.max(np.abs(ty / fld["k2_ff"] - 1.0))), float(y[j]))


FIELDS = {"knee": knee_exact, "pn": pn_exact}


# =====================================================================
# seed contour, identical machinery on both fields
# =====================================================================

def theta_interp(fld):
    return RegularGridInterpolator((fld["y"], fld["x"]), fld["theta"],
                                   bounds_error=False, fill_value=np.nan)


def level_set_contour(fld, C, n_rows=600):
    """
    A true constant-phase contour, upper arm only.

    theta is strictly monotone in x at fixed y on both fields: on the knee
    because theta = k1 x + f(y), on PN because psi decreases in x and every
    telescoped weight is positive. So one interpolation per row gives a
    single-valued curve.

    Upper arm only because only upper-half seeds are used. On PN the lower
    arm of the same V sits at level -C, not +C, since theta is odd.
    """
    x, y = fld["x"], fld["y"]
    rows = np.linspace(fld["y_top"], 0.0, n_rows)
    xs, ys, dropped = [], [], 0
    for yy in rows:
        j = int(np.argmin(np.abs(y - yy)))
        th = fld["theta"][j]
        xp, fp = (th, x) if th[0] < th[-1] else (th[::-1], x[::-1])
        if not (xp[0] <= C <= xp[-1]):
            dropped += 1
            continue
        xs.append(float(np.interp(C, xp, fp)))
        ys.append(float(y[j]))
    if not xs:
        raise RuntimeError("level set contour is empty; C is off the field")
    return np.vstack([np.array(xs), np.array(ys)]), dropped


def shoot_line(x0, y0, ik1, ik2, ds, max_steps=40000):
    """One coordinate line with exactly the pipeline's stepping convention
    (dir_sign = -1 in the upper half). Used only to place and clip the
    contour, never to build phase."""
    for _ in range(max_steps):
        kx, ky = float(ik1((y0, x0))), float(ik2((y0, x0)))
        m = np.hypot(kx, ky)
        if m == 0.0:
            return np.nan
        s = -1.0 if y0 > 0 else 1.0
        xn, yn = x0 + s * kx / m * ds, y0 + s * ky / m * ds
        if y0 * yn <= 0.0:
            return xn
        x0, y0 = xn, yn
    return np.nan


def build_seed_contour(fld, mode, ds, cover_frac=0.9, n_probe=25, n_iter=2,
                       report=print):
    """
    Place the contour so the seeded core interval is centred, then clip it
    to the arc that lands inside the requested coverage.

    Centring: shifting C by dC translates the contour by dC/k1 in the far
    field and translates the landings by the same amount, so one correction
    per pass converges immediately.

    Clipping matters on both fields. The full contour spans far more in x
    than the interval worth scoring, so without it most lines are spent
    outside the region of interest and the box has to be correspondingly
    larger.
    """
    if mode == "trapezoid":
        return None
    x, y = fld["x"], fld["y"]
    ik1 = RegularGridInterpolator((y, x), fld["k1"], bounds_error=False,
                                  fill_value=0.0)
    ik2 = RegularGridInterpolator((y, x), fld["k2"], bounds_error=False,
                                  fill_value=0.0)

    j_top = int(np.argmin(np.abs(y - fld["y_top"])))
    C = float(fld["theta"][j_top, int(np.argmin(np.abs(x)))])
    for _ in range(n_iter):
        con, _ = level_set_contour(fld, C, n_rows=240)
        up = con[1] > 0
        if up.sum() < 3:
            break
        idx = np.linspace(0, up.sum() - 1, min(n_probe, up.sum())).astype(int)
        land = np.array([shoot_line(con[0, up][i], con[1, up][i], ik1, ik2, ds)
                         for i in idx])
        land = land[np.isfinite(land)]
        if land.size == 0:
            break
        C = C - fld["k1_ff"] * 0.5 * (land.min() + land.max())

    con, dropped = level_set_contour(fld, C)
    if dropped:
        report(f"     [contour] level set outside theta range on {dropped} "
               f"rows; dropped")

    n = con.shape[1]
    idx = np.linspace(0, n - 1, min(n_probe, n)).astype(int)
    land = np.array([shoot_line(con[0, i], con[1, i], ik1, ik2, ds)
                     for i in idx])
    good = np.isfinite(land)
    if good.sum() < 3:
        return con
    ii, ll = idx[good].astype(float), land[good]
    order = np.argsort(ll)
    target = cover_frac * 0.5 * fld["cover"]
    i_lo = float(np.interp(-target, ll[order], ii[order]))
    i_hi = float(np.interp(+target, ll[order], ii[order]))
    a, b = int(max(0, min(i_lo, i_hi))), int(min(n - 1, max(i_lo, i_hi)))
    report(f"     [contour] probe landings span [{ll.min():.1f}, "
           f"{ll.max():.1f}], target [{-target:.1f}, {target:.1f}], "
           f"keeping arc {a} to {b} of {n}")
    return con if b - a < 8 else con[:, a:b + 1]


def trapezoid_mu(fld):
    """The mu whose knee far-field angle equals this field's, so the
    pipeline's own trapezoid is asked for at the right slope. Identity on
    the knee; on PN it corrects for k2_ff = 1 - k1^2/2."""
    k1, k2 = fld["k1_ff"], fld["k2_ff"]
    return float(k2 / np.hypot(k1, k2))


# =====================================================================
# the three directors
# =====================================================================

def director_exact(fld, **kw):
    return fld["k1"].copy(), fld["k2"].copy()


def director_uhu(fld, sigma=np.pi / 2, ramp=None, **kw):
    if ramp is None:
        ramp = np.ones_like(fld["u"])
    sm = lambda arr: macro_iso(arr, sigma, fld["Lx"], fld["Ly"])
    ops = compute_uhu_ops_smoother(fld["u"], fld["x"], fld["y"], sm,
                                   spectral_deriv, ramp)
    return ops["k1_orig"], ops["k2_orig"]


def director_riesz(fld, sigma_dir=np.pi / 2, **kw):
    rd = riesz_director(fld["u"], fld["Lx"], fld["Ly"], sigma_dir)
    return rd["d1"], rd["d2"]


SOURCES = {"exact": director_exact, "uhu": director_uhu,
           "riesz": director_riesz}
SOURCE_COLORS = {"exact": "C0", "uhu": "C2", "riesz": "C3"}


# =====================================================================
# run one configuration
# =====================================================================

def rebuild_grid(res, fld, gauge="seed_phase", half="orig"):
    """
    Re-grid the phase from the pipeline's own coordinate lines and Hilbert
    phases, changing only the per-line unwrapping origin.

      raw          the pipeline as it stands
      zero_origin  subtract each line's first value
      seed_phase   subtract the first value and add the exact theta at the
                   seed point; available only because this is analytic

    The seed-to-core branch is used, not the stitched line: along the full
    stitched line the true phase is V-shaped while the Hilbert phase of an
    oscillatory record is monotone, so the reflected half accumulates with
    the wrong sign and griddata blends the two.
    """
    if gauge == "raw" and half == "stitched":
        return res["phase_grid_unwrapped"][:, :, 0]

    x, y = fld["x"], fld["y"]
    ith = theta_interp(fld) if gauge == "seed_phase" else None
    pts, vals = [], []
    for ln, ph in zip(res["coordinate_lines"][0],
                      res["phase_lines_unwrapped"][0]):
        ln = np.asarray(ln, dtype=float)
        ph = np.asarray(ph, dtype=float)
        if ln.shape[0] != ph.shape[0] or ph.size < 2:
            continue
        if half == "orig":
            ic = int(np.argmin(np.abs(ln[:, 1])))
            if ic < 8:
                continue
            ln, ph = ln[:ic + 1], ph[:ic + 1]
        off = ph[0]
        if gauge == "seed_phase":
            # the Hilbert phase accumulates as -theta, so the seed phase
            # enters with a plus and the result is aligned to -theta
            t0 = float(ith((ln[0, 1], ln[0, 0])))
            if not np.isfinite(t0):
                continue
            off = off + t0
        pts.append(ln)
        vals.append(ph - off)
    if not pts:
        return res["phase_grid_unwrapped"][:, :, 0]
    Xg, Yg = np.meshgrid(x, y)
    return griddata(np.vstack(pts), np.concatenate(vals), (Xg, Yg),
                    method="linear")


def run_one(fld, source, seed_bdry, n_seeds, sigma, ds, ramp,
            use_sym=False, gauge="seed_phase", half="orig"):
    """Build the director, run the pipeline, re-grid, and report coverage
    diagnostics. Returns (phi_grid, res)."""
    k1d, k2d = SOURCES[source](fld, sigma=sigma, ramp=ramp, sigma_dir=sigma)

    uhu_data = {
        "x": fld["x"], "y": fld["y"], "u": fld["u"][:, :, None],
        "k1_sym": np.asarray(k1d)[:, :, None],
        "k2_sym": np.asarray(k2d)[:, :, None],
        "k1_orig": np.asarray(k1d)[:, :, None],
        "k2_orig": np.asarray(k2d)[:, :, None],
        "mu": fld["mu"], "tt": None, "ramp": None, "k": None, "A": None,
        "knee_bdry": seed_bdry,
        "source_path": (f"analytic {fld['field']} mu={fld['mu']} "
                        f"director={source} sigma={sigma:.4f}"),
    }
    res = compute_pgb_phase_from_uhu(
        uhu_data, mu=trapezoid_mu(fld), n_phase_seeds=n_seeds, ds=ds,
        seed_half="upper", phase_ramp_mode="none", prefer_sym=True)

    reach, landing = [], []
    for ln in res["coordinate_lines"][0]:
        ln = np.asarray(ln, float)
        if not len(ln):
            continue
        ic = int(np.argmin(np.abs(ln[:, 1])))
        reach.append(abs(ln[ic, 1]))
        landing.append(ln[ic, 0])
    res["frac_reaching_core"] = (float(np.mean(np.array(reach) < 0.5))
                                 if reach else 0.0)
    land = np.sort(np.asarray(landing))
    land = land[np.isfinite(land)]
    res["core_landings"] = land
    res["landing_gap_max"] = float(np.diff(land).max()) if land.size > 2 \
        else np.nan
    res["director"] = (np.asarray(k1d), np.asarray(k2d))
    return rebuild_grid(res, fld, gauge, half), res


# =====================================================================
# scoring
# =====================================================================

def align_to_truth_upper(phi, fld):
    """
    Put the reconstruction in the gauge of the exact theta by least squares
    on the two free conventions, phi ~ a theta + c, fitted on the upper half.
    a should come out near -1 because upper-half seeds step along -khat.
    Anything else means the reconstruction is not a reparametrized copy of
    the truth at all, which is itself the diagnosis. Restricting to the
    upper half is required on PN, where theta is odd, and harmless on the
    knee, where it is even.
    """
    up = np.tile((fld["y"] > 0)[:, None], (1, len(fld["x"])))
    ok = np.isfinite(phi) & np.isfinite(fld["theta"]) & up
    if ok.sum() < 32:
        return np.full_like(phi, np.nan), np.nan, np.nan
    M = np.vstack([fld["theta"][ok], np.ones(ok.sum())]).T
    (a, c), *_ = np.linalg.lstsq(M, phi[ok], rcond=None)
    if abs(a) < 1e-6:
        return np.full_like(phi, np.nan), float(a), float(c)
    return (phi - c) / a, float(a), float(c)


def detect_plateaus(x, r, min_sep, thresh_frac=0.15, level_frac=0.6,
                    hi_pct=99.0):
    """
    Locate the edges of a staircase and measure it.

    Edges are peaks of |dr/dx| above a threshold set as the larger of
    thresh_frac of a HIGH PERCENTILE of |dr/dx| and a robust floor at the
    median plus five MADs. The percentile rather than the maximum matters:
    one large excursion anywhere in the window would otherwise lift the
    threshold above every real step.

    Plateau levels are medians over the central level_frac of each interval
    between edges, which is far steadier than reading the rise across an
    edge. Step heights are differences of consecutive levels.
    """
    ok = np.isfinite(r)
    out = {"n_gaps": 0, "gap_frac": float(1.0 - ok.mean()) if r.size else 1.0,
           "x_edges": np.array([]), "spacings": np.array([]),
           "heights": np.array([]), "widths": np.array([]),
           "levels": np.array([])}
    if ok.sum() < 16:
        return out

    idx = np.flatnonzero(ok)
    brk = np.flatnonzero(np.diff(idx) > 1)
    starts, ends = np.r_[idx[0], idx[brk + 1]], np.r_[idx[brk], idx[-1]]
    b = int(np.argmax(ends - starts))
    out["n_gaps"] = int(len(starts) - 1)
    sl = slice(starts[b], ends[b] + 1)
    xs, rs = x[sl], r[sl]
    keep = np.isfinite(rs)
    xs, rs = xs[keep], rs[keep]
    if xs.size < 16:
        return out

    d = np.abs(np.gradient(rs, xs))
    med = np.median(d)
    mad = np.median(np.abs(d - med)) + 1e-12
    thr = max(thresh_frac * float(np.percentile(d, hi_pct)), med + 5.0 * mad)

    cand = np.flatnonzero((d > thr) & (d >= np.r_[d[0], d[:-1]])
                          & (d >= np.r_[d[1:], d[-1]]))
    peaks = []
    for i in cand:
        if peaks and xs[i] - xs[peaks[-1]] < min_sep:
            if d[i] > d[peaks[-1]]:
                peaks[-1] = i
            continue
        peaks.append(i)
    peaks = np.array(peaks, dtype=int)
    if peaks.size == 0:
        return out

    widths = []
    for i in peaks:
        half = 0.5 * d[i]
        lo, hi = i, i
        while lo > 0 and d[lo] > half:
            lo -= 1
        while hi < len(d) - 1 and d[hi] > half:
            hi += 1
        widths.append(xs[hi] - xs[lo])

    edges = np.r_[-np.inf, xs[peaks], np.inf]
    levels = []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        m = (xs > a_) & (xs < b_)
        if m.sum() < 3:
            levels.append(np.nan)
            continue
        xi = xs[m]
        pad = 0.5 * (1.0 - level_frac) * (xi[-1] - xi[0])
        mm = m & (xs >= xi[0] + pad) & (xs <= xi[-1] - pad)
        levels.append(float(np.median(rs[mm if mm.sum() >= 3 else m])))

    out["x_edges"] = xs[peaks]
    out["spacings"] = np.diff(xs[peaks]) if peaks.size > 1 else np.array([])
    out["heights"] = np.diff(np.array(levels))
    out["widths"] = np.array(widths)
    out["levels"] = np.array(levels)
    return out


def stair_metrics(x, vals, ok, fld, thresh_frac=0.15, delta_frac=0.15):
    """
    Metrics for one row, applied identically to the reconstruction and to
    the exact theta on the same row so the target is measured, not assumed.

    That matters on both fields for different reasons. On PN the true step
    is sqrt(2 y) wide, so a row four units up is genuinely close to a ramp
    and its own sawtooth is small; scoring against a fixed target would call
    a correct answer a failure. On the knee the truth's sawtooth is zero, so
    any excess is invented structure.
    """
    out = {}
    xw, v = x[ok], vals[ok]
    if xw.size < 16:
        return out
    out["mean_slope"] = float((v[-1] - v[0]) / (xw[-1] - xw[0]))
    A = np.vstack([xw, np.ones(xw.size)]).T
    (sl, ic), *_ = np.linalg.lstsq(A, v, rcond=None)
    saw = np.full(len(x), np.nan)
    saw[ok] = v - (sl * xw + ic)
    out["lsq_slope"] = float(sl)
    out["sawtooth"] = saw
    out["sawtooth_ptp"] = float(np.ptp(saw[ok]))

    # localization needs known defect columns, so it is PN only
    if fld["xd"].size:
        delta = delta_frac * fld["Lsp"]
        xd_in = fld["xd"][(fld["xd"] > xw[0] + delta)
                          & (fld["xd"] < xw[-1] - delta)]
        out["xd_in_window"] = xd_in
        out["n_defects_in_window"] = int(xd_in.size)
        if xd_in.size >= 2:
            f = lambda q: np.interp(q, xw, v)
            steps = f(xd_in + delta) - f(xd_in - delta)
            betw = f(xd_in[1:] - delta) - f(xd_in[:-1] + delta)
            tot = float(np.sum(steps[:-1]) + np.sum(betw))
            out["step_rise_mean"] = float(np.mean(steps))
            out["step_rise_spread"] = float(np.std(steps))
            if abs(tot) > 1e-9:
                out["localization"] = float(np.sum(steps[:-1]) / tot)

    pl = detect_plateaus(x, np.where(ok, vals, np.nan),
                         min_sep=0.35 * fld["Lsp"], thresh_frac=thresh_frac)
    out["plateaus"] = pl
    out["n_edges"] = int(pl["x_edges"].size)
    if pl["spacings"].size:
        out["spacing_mean"] = float(np.mean(pl["spacings"]))
        out["spacing_spread"] = float(np.std(pl["spacings"]))
    return out


def score_core(phi_grid, fld, x_trim=0.15, thresh_frac=0.15, core_y=None,
               phi_aligned=None):
    """
    Score one row against the exact phase on the same row.

    core_y = None selects the core row: y = 0 on the knee, the first row
    above it on PN, which is where the coordinate lines terminate. A
    positive value selects the nearest row above it, which is how the PN
    resolution ladder is built.
    """
    x, y = fld["x"], fld["y"]
    if core_y is None or core_y <= 0:
        iy = fld["iy_core"]
    else:
        cand = np.flatnonzero(y > 0)
        iy = int(cand[np.argmin(np.abs(y[cand] - core_y))])

    phi_al, a, c = (align_to_truth_upper(phi_grid, fld)
                    if phi_aligned is None else phi_aligned)
    phi = np.asarray(phi_al[iy, :], dtype=float)
    th = np.asarray(fld["theta"][iy, :], dtype=float)

    res = {"a_align": a, "c_align": c, "iy": iy, "y_core": float(y[iy]),
           "coverage": float(np.isfinite(phi).mean()),
           "n_ok": int(np.isfinite(phi).sum()),
           "step_width": float(np.sqrt(2.0 * max(y[iy], 0.0))),
           "x": x, "phi": phi, "theta_core": th}
    fin = np.isfinite(phi)
    if fin.sum() < 16:
        return res

    idx = np.flatnonzero(fin)
    pad = int(x_trim * (idx[-1] - idx[0] + 1))
    ok = np.zeros(len(x), dtype=bool)
    ok[idx[0] + pad: idx[-1] - pad + 1] = True
    ok &= fin
    if ok.sum() < 16:
        return res
    res["ok"] = ok
    res["x_window"] = (float(x[ok][0]), float(x[ok][-1]))

    r = np.full(len(x), np.nan)
    r[ok] = phi[ok] - th[ok]
    r[ok] -= np.median(r[ok])
    res["resid_vs_exact"] = r
    res["rms"] = float(np.sqrt(np.nanmean(r[ok]**2)))
    res["max_abs"] = float(np.nanmax(np.abs(r[ok])))

    res.update(stair_metrics(x, phi, ok, fld, thresh_frac))
    res["ref"] = stair_metrics(x, th, ok, fld, thresh_frac)

    # roughness. The ratio is only meaningful where the truth has some; on
    # the knee the exact core is a straight line and the denominator is
    # essentially zero, so the absolute value is what to read there.
    d2p = np.diff(phi[ok], 2)
    d2t = np.diff(th[ok], 2)
    res["d2_rms"] = float(np.sqrt(np.nanmean(d2p**2)))
    res["d2_rms_exact"] = float(np.sqrt(np.nanmean(d2t**2)))
    if res["d2_rms_exact"] > 1e-6:
        res["d2_ratio"] = float(res["d2_rms"] / res["d2_rms_exact"])
    return res


# =====================================================================
# figures
# =====================================================================

def _show_u(axis, fld, ext):
    axis.imshow(fld["u"], extent=ext, origin="lower", cmap="gray",
                aspect="auto", alpha=0.85)


def _mark_defects(axis, fld, color="0.88", lw=0.6):
    for c in fld["xd"]:
        axis.axvline(c, color=color, lw=lw, zorder=0)


def sizing_figure(fld, path):
    """grad theta along the row at y_top against the far field, so the
    choice of nd (PN) or the box (knee) is inspectable rather than trusted."""
    x, y = fld["x"], fld["y"]
    j = int(np.argmin(np.abs(y - fld["y_top"])))
    tx = np.gradient(fld["theta"][j], x)
    ty = (fld["theta"][j + 1] - fld["theta"][j - 1]) / (y[j + 1] - y[j - 1])
    ex, ey, yj = planewave_error(fld)

    fig, ax = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True)
    ax[0].plot(x, tx, "C0-", lw=1.2)
    ax[0].axhline(fld["k1_ff"], color="k", ls="--", lw=1.2,
                  label=rf"$k_1={fld['k1_ff']:.4f}$")
    ax[0].set_ylabel(r"$\theta_x$ at $y_{top}$")
    ax[1].plot(x, ty, "C1-", lw=1.2)
    ax[1].axhline(fld["k2_ff"], color="k", ls="--", lw=1.2,
                  label=rf"$k_2={fld['k2_ff']:.4f}$")
    ax[1].set_ylabel(r"$\theta_y$ at $y_{top}$")
    ax[1].set_xlabel("x")
    for a_ in ax:
        a_.legend(fontsize=8)
        _mark_defects(a_, fld, color="0.9")
        a_.axvspan(-2 * fld["Lsp"], 2 * fld["Lsp"], color="C2", alpha=0.08,
                   zorder=0)
    fig.suptitle(rf"{fld['field']} $\mu={fld['mu']}$, $n_d={fld['nd']}$, "
                 rf"$y_{{top}}={yj:.1f}$: far-field error {100 * ex:.2f}% in "
                 rf"$\theta_x$, {100 * ey:.2f}% in $\theta_y$ (shaded window)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def diagnostic_figure(fld, source, res, phi, path, sc=None, quiver_step=24,
                      line_step=6):
    """Nine panels. Row 1 is what went into the reconstruction, row 2 is the
    reconstruction against the truth on the upper half, row 3 is the core."""
    x, y = fld["x"], fld["y"]
    X, Y, theta = fld["X"], fld["Y"], fld["theta"]
    k1, Lsp = fld["k1_ff"], fld["Lsp"]
    ext = [x[0], x[-1], y[0], y[-1]]
    up = y > 0
    ext_up = [x[0], x[-1], y[up][0], y[up][-1]]

    if sc is None:
        sc = score_core(phi, fld)
    phi_al, a, _ = align_to_truth_upper(phi, fld)

    fig, ax = plt.subplots(3, 3, figsize=(17.0, 13.5))

    _show_u(ax[0, 0], fld, ext)
    sb = res.get("knee_bdry_phase")
    if sb is not None:
        sb = np.asarray(sb)
        ax[0, 0].plot(sb[0], sb[1], ".", color="deepskyblue", ms=3.0)
    for c in fld["xd"]:
        ax[0, 0].axvline(c, color="lime", lw=0.7, alpha=0.65)
    ax[0, 0].set_title(f"pattern, seeds"
                       + (f", {fld['nd']} defect columns" if fld["nd"] else ""))

    _show_u(ax[0, 1], fld, ext)
    d1, d2 = res.get("director", (None, None))
    if d1 is not None:
        m = np.hypot(d1, d2)
        s_ = quiver_step
        ax[0, 1].quiver(X[::s_, ::s_], Y[::s_, ::s_],
                        (d1 / np.maximum(m, 1e-12))[::s_, ::s_],
                        (d2 / np.maximum(m, 1e-12))[::s_, ::s_],
                        color="crimson", scale=28, headwidth=2, pivot="mid")
    ax[0, 1].set_title(f"pattern + {source} director (unit)")

    _show_u(ax[0, 2], fld, ext)
    for j, ln in enumerate(res["coordinate_lines"][0]):
        if j % line_step:
            continue
        ln = np.asarray(ln, float)
        ax[0, 2].plot(ln[:, 0], ln[:, 1], "-", color="gold", lw=0.9)
    ax[0, 2].set_title("pattern + coordinate lines "
                       f"({100 * res.get('frac_reaching_core', np.nan):.0f}%"
                       " reach core)")

    lo_, hi_ = np.nanpercentile(theta[up], [1, 99])
    for axis, f_, ttl in ((ax[1, 0], theta, r"exact $\theta$"),
                          (ax[1, 1], phi_al, f"recovered (aligned, a={a:.3f})")):
        im = axis.imshow(f_[up], extent=ext_up, origin="lower",
                         cmap="twilight", vmin=lo_, vmax=hi_, aspect="auto")
        fig.colorbar(im, ax=axis)
        axis.set_title(ttl + " (upper half)")

    e_up = (phi_al - theta)[up]
    if np.isfinite(e_up).any():
        e_up = e_up - np.nanmedian(e_up)
        v = np.nanpercentile(np.abs(e_up), 98)
    else:
        v = 1.0
    im = ax[1, 2].imshow(e_up, extent=ext_up, origin="lower", cmap="RdBu_r",
                         vmin=-v, vmax=v, aspect="auto")
    fig.colorbar(im, ax=ax[1, 2])
    ax[1, 2].set_title(f"error, upper half (98th pct = {v:.2f} rad)")
    for axis in ax[1]:
        for c in fld["xd"]:
            axis.axvline(c, color="0.4", lw=0.4, alpha=0.5)

    iy = sc["iy"]
    ph_c = sc.get("phi", np.full_like(x, np.nan))
    ax[2, 0].plot(x, theta[iy] / np.pi, "k-", lw=1.8, label=r"exact $\theta$")
    ax[2, 0].plot(x, ph_c / np.pi, "C3-", lw=1.2, label="recovered")
    _mark_defects(ax[2, 0], fld)
    ax[2, 0].set_title(rf"core row $y={sc['y_core']:+.3f}$")
    ax[2, 0].set_ylabel(r"phase / $\pi$")
    ax[2, 0].legend(fontsize=8)
    ax[2, 0].set_xlabel("x")

    r = sc.get("resid_vs_exact")
    if r is not None:
        ax[2, 1].plot(x, r, "C0-", lw=1.2, label="vs exact")
    saw = sc.get("sawtooth")
    if saw is not None:
        ref = sc.get("ref", {}).get("sawtooth_ptp", np.nan)
        ax[2, 1].plot(x, saw, "C5-", lw=1.0, alpha=0.8,
                      label=rf"about best-fit line "
                            rf"(ptp/$\pi$={sc.get('sawtooth_ptp', np.nan) / np.pi:.2f}"
                            rf", truth {ref / np.pi:.2f})")
    for lv in (-np.pi / 2, 0.0, np.pi / 2):
        ax[2, 1].axhline(lv, color="0.6", ls=":", lw=0.9)
    _mark_defects(ax[2, 1], fld)
    for xe in sc.get("plateaus", {}).get("x_edges", []):
        ax[2, 1].axvline(xe, color="C3", lw=0.9, ls="--", alpha=0.8)
    ax[2, 1].set_title("core residuals; dashed = detected edges")
    ax[2, 1].legend(fontsize=7)
    ax[2, 1].set_xlabel("x")

    okc = np.isfinite(ph_c)
    if okc.sum() > 4:
        dphi = np.abs(np.gradient(ph_c[okc], x[okc]))
        ax[2, 2].plot(x[okc], dphi, "C2-", lw=1.0, label=r"$|d\phi/dx|$")
        w = max(int(Lsp / (x[1] - x[0])), 3)
        ax[2, 2].plot(x[okc], np.convolve(dphi, np.ones(w) / w, mode="same"),
                      "C4-", lw=1.4, label=r"running mean over $\pi/k_1$")
    ax[2, 2].axhline(k1, color="k", ls="--", lw=1.2, label=f"$k_1$={k1:.3f}")
    ax[2, 2].axhline(0.0, color="k", ls="-", lw=0.8)
    ax[2, 2].set_ylim(-0.05, 3.0 * max(k1, 0.1))
    ax[2, 2].set_title("core slope" + (r": 0 between defects, mean $k_1$"
                                       if fld["nd"] else r": uniform $k_1$"))
    ax[2, 2].legend(fontsize=8)
    ax[2, 2].set_xlabel("x")

    fig.suptitle(rf"{fld['field']}  $\mu={fld['mu']}$  director: {source}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def row_ladder_figure(fld, ladder, path):
    """How each director does as a function of how close to the core it is
    scored. PN only: on the knee the field is defect free and the core row
    is the only one that means anything."""
    fig, ax = plt.subplots(1, 3, figsize=(16.0, 4.6))
    any_scs = None
    for src, scs in ladder.items():
        good = [s for s in scs if "rms" in s]
        if not good:
            continue
        any_scs = good
        ys = [s["y_core"] for s in good]
        cl = SOURCE_COLORS.get(src, "C4")
        ax[0].plot(ys, [s["rms"] for s in good], "o-", color=cl, lw=1.4,
                   label=f"{src} director")
        ax[1].plot(ys, [s.get("sawtooth_ptp", np.nan) / np.pi for s in good],
                   "o-", color=cl, lw=1.4, label=f"{src} director")
        ax[2].plot(ys, [s.get("localization", np.nan) for s in good], "o-",
                   color=cl, lw=1.4, label=f"{src} director")
    if any_scs:
        ys = [s["y_core"] for s in any_scs]
        ax[1].plot(ys, [s["ref"].get("sawtooth_ptp", np.nan) / np.pi
                        for s in any_scs], "k--", lw=1.6,
                   label="truth at this row")
        ax[2].plot(ys, [s["ref"].get("localization", np.nan)
                        for s in any_scs], "k--", lw=1.6,
                   label="truth at this row")
        sw = [s["step_width"] / (fld["x"][1] - fld["x"][0]) for s in any_scs]
        ax[0].set_title("step width at these rows: "
                        + ", ".join(f"{v:.1f}" for v in sw) + " cells",
                        fontsize=9)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("rms vs exact (rad)")
    ax[1].set_ylabel(r"sawtooth ptp / $\pi$")
    ax[2].set_ylabel("localization")
    for a_ in ax:
        a_.set_xscale("log")
        a_.set_xlabel("y of the scored row")
        a_.legend(fontsize=8)
        a_.grid(alpha=0.25)
    fig.suptitle(rf"PN $\mu={fld['mu']}$: director performance against "
                 "proximity to the core", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def profiles_figure(fld, ladder, path, zoom=None, n_spacings=2.5):
    """Core profiles at every scored row, all sources against the truth.
    zoom centres a window of n_spacings spacings, which is what shows
    whether the recovered curve is jagged at the grid scale."""
    any_scs = next(iter(ladder.values()))
    rows = [i for i, s in enumerate(any_scs) if "rms" in s]
    if not rows:
        return
    fig, ax = plt.subplots(len(rows), 2, figsize=(15.0, 3.1 * len(rows)),
                           squeeze=False)
    for r, i in enumerate(rows):
        a0, a1 = ax[r, 0], ax[r, 1]
        sc0 = any_scs[i]
        x = sc0["x"]
        a0.plot(x, sc0["theta_core"] / np.pi, "k-", lw=2.0,
                label=r"truth: exact $\theta$")
        for src, scs in ladder.items():
            s_ = scs[i]
            if "rms" not in s_:
                continue
            cl = SOURCE_COLORS.get(src, "C4")
            a0.plot(x, s_["phi"] / np.pi, color=cl, lw=1.1,
                    label=f"{src} director")
            a1.plot(x, s_["resid_vs_exact"], color=cl, lw=1.1,
                    label=f"{src} (rms {s_['rms']:.3f})")
        _mark_defects(a0, fld)
        _mark_defects(a1, fld)
        a1.axhline(0.0, color="0.4", lw=0.8)
        if zoom is not None:
            w = n_spacings * fld["Lsp"] / 2.0
            a0.set_xlim(zoom - w, zoom + w)
            a1.set_xlim(zoom - w, zoom + w)
            m = (x > zoom - w) & (x < zoom + w)
            if m.any():
                lo = np.nanmin(sc0["theta_core"][m]) / np.pi
                hi = np.nanmax(sc0["theta_core"][m]) / np.pi
                a0.set_ylim(lo - 0.6, hi + 0.6)
                rr = np.concatenate([s[i]["resid_vs_exact"][m]
                                     for s in ladder.values()
                                     if "rms" in s[i]])
                if np.isfinite(rr).any():
                    v = np.nanmax(np.abs(rr)) * 1.2 + 1e-3
                    a1.set_ylim(-v, v)
        a0.set_ylabel(rf"$y={sc0['y_core']:.2f}$" "\n" r"phase / $\pi$")
        a0.set_title(f"step width {sc0['step_width']:.2f} = "
                     f"{sc0['step_width'] / (x[1] - x[0]):.1f} cells"
                     if fld["nd"] else "core row", fontsize=9)
        a1.set_title("recovered minus exact", fontsize=9)
        if r == 0:
            a0.legend(fontsize=7, ncol=2)
            a1.legend(fontsize=7)
    ax[-1, 0].set_xlabel("x")
    ax[-1, 1].set_xlabel("x")
    fig.suptitle(rf"{fld['field']} $\mu={fld['mu']}$: core profiles"
                 + ("" if zoom is None
                    else rf", zoom on $x={zoom:.1f}$ over {n_spacings} spacings"),
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def landing_figure(fld, landings, path):
    """Where the coordinate lines terminate on the core. A stretch of core
    with no landings is a sampling gap, not a director failure, and the two
    have to be told apart."""
    fig, ax = plt.subplots(figsize=(10.0, 1.3 + 0.9 * len(landings)))
    for i, (src, ld) in enumerate(landings.items()):
        ld = np.asarray(ld)
        ld = ld[np.isfinite(ld)]
        ax.plot(ld, np.full_like(ld, i), "|", ms=12,
                color=SOURCE_COLORS.get(src, "C4"))
    _mark_defects(ax, fld, color="0.85", lw=0.8)
    ax.set_yticks(range(len(landings)))
    ax.set_yticklabels(list(landings.keys()), fontsize=8)
    ax.set_xlabel("x on the core")
    ax.set_title(rf"{fld['field']} $\mu={fld['mu']}$: coordinate-line "
                 "landings", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# =====================================================================
# surfaces and Riesz panels
# =====================================================================

def _downsample(a, target=180):
    step = max(1, int(np.ceil(max(a.shape) / target)))
    return a[::step, ::step], step


def surface_figures(fld, source, phi, path_3d, path_2d, target=180):
    """
    The recovered phase as a surface, against the exact phase and the error.

    Two versions of the same three panels. The 3D one reads better in a
    talk, the 2D height map with contours reads more accurately at column
    width, so both are emitted and you can pick from the figures rather than
    from a description.

    Upper half only, because on PN theta is odd and jumps by 2 m pi across
    the core, and because that is the half the seeds live in on both fields.
    """
    x, y = fld["x"], fld["y"]
    up = y > 0
    theta = fld["theta"][up]
    phi_al, a, _ = align_to_truth_upper(phi, fld)
    rec = phi_al[up]
    err = rec - theta
    if np.isfinite(err).any():
        err = err - np.nanmedian(err)

    Th, st = _downsample(theta, target)
    Re, _ = _downsample(rec, target)
    Er, _ = _downsample(err, target)
    Xs, Ys = np.meshgrid(x[::st], y[up][::st])

    lo, hi = np.nanpercentile(Th, [1, 99])
    ev = np.nanpercentile(np.abs(Er), 98) if np.isfinite(Er).any() else 1.0
    ev = max(ev, 1e-6)

    panels = ((Th, r"exact $\theta$", "twilight", lo, hi),
              (Re, f"recovered ({source})", "twilight", lo, hi),
              (Er, f"error (98th pct {ev:.3f} rad)", "RdBu_r", -ev, ev))

    # --- 3D --------------------------------------------------------
    fig = plt.figure(figsize=(18.0, 5.6))
    for i, (Z, ttl, cm, v0, v1) in enumerate(panels):
        axis = fig.add_subplot(1, 3, i + 1, projection="3d")
        Zm = np.where(np.isfinite(Z), Z, np.nan)
        axis.plot_surface(Xs, Ys, Zm, cmap=cm, vmin=v0, vmax=v1,
                          linewidth=0, antialiased=True, rstride=1,
                          cstride=1, shade=True)
        axis.set_title(ttl, fontsize=10)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.view_init(elev=32, azim=-124)
    fig.suptitle(rf"{fld['field']} $\mu={fld['mu']}$: recovered phase "
                 f"surface, {source} director", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path_3d, dpi=140)
    plt.close(fig)

    # --- 2D height map with contours -------------------------------
    ext = [x[0], x[-1], y[up][0], y[up][-1]]
    fig, ax = plt.subplots(1, 3, figsize=(17.0, 4.8))
    for i, (Z, ttl, cm, v0, v1) in enumerate(panels):
        im = ax[i].imshow(Z, extent=ext, origin="lower", cmap=cm, vmin=v0,
                          vmax=v1, aspect="auto")
        fig.colorbar(im, ax=ax[i])
        with np.errstate(invalid="ignore"):
            if np.isfinite(Z).sum() > 100:
                ax[i].contour(Xs, Ys, Z, levels=14, colors="k",
                              linewidths=0.45, alpha=0.55)
        for c in fld["xd"]:
            ax[i].axvline(c, color="0.3", lw=0.4, alpha=0.5)
        ax[i].set_title(ttl, fontsize=10)
        ax[i].set_xlabel("x")
    ax[0].set_ylabel("y")
    fig.suptitle(rf"{fld['field']} $\mu={fld['mu']}$: recovered phase, "
                 f"{source} director (contours every 14 levels)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path_2d, dpi=140)
    plt.close(fig)


def riesz_panels_figure(fld, path, sigma_dir=np.pi / 2, quiver_step=26):
    """
    The Riesz construction laid out in full: the two Riesz components, the
    raw pointwise orientation, the doubled-angle smoothed director, its
    alignment magnitude, and the angle error against the exact grad theta
    direction.

    The last panel is the one that says where Riesz is trustworthy. R is
    linear in u, so its orientation is already the local wavevector
    direction pointwise, and the smoothing only resolves the mod-pi sign
    ambiguity. The error map shows what that costs near a defect.
    """
    x, y = fld["x"], fld["y"]
    ext = [x[0], x[-1], y[0], y[-1]]
    mono = monogenic(fld["u"], fld["Lx"], fld["Ly"])
    rd = riesz_director(fld["u"], fld["Lx"], fld["Ly"], sigma_dir)

    ang_exact = np.arctan2(fld["k2"], fld["k1"])
    dang = np.degrees(np.angle(np.exp(2j * (rd["angle"] - ang_exact)))) / 2.0

    fig, ax = plt.subplots(2, 3, figsize=(17.0, 9.0))
    v = np.nanpercentile(np.abs(mono["R1"]), 99)
    for axis, f_, ttl, cm, lo, hi in (
            (ax[0, 0], mono["R1"], r"$R_1$", "RdBu_r", -v, v),
            (ax[0, 1], mono["R2"], r"$R_2$", "RdBu_r", -v, v),
            (ax[0, 2], np.degrees(mono["orientation"]),
             "raw Riesz orientation (deg)", "twilight", -180, 180),
            (ax[1, 0], np.degrees(rd["angle"]),
             rf"smoothed director angle, $\sigma_{{dir}}={sigma_dir:.2f}$ (deg)",
             "twilight", -90, 90),
            (ax[1, 1], rd["alignment"], "alignment magnitude", "viridis", 0, 1),
            (ax[1, 2], dang, "director angle error vs exact (deg)",
             "coolwarm", -20, 20)):
        im = axis.imshow(f_, extent=ext, origin="lower", cmap=cm, vmin=lo,
                         vmax=hi, aspect="auto")
        fig.colorbar(im, ax=axis)
        axis.set_title(ttl, fontsize=10)
        for c in fld["xd"]:
            axis.axvline(c, color="0.25", lw=0.4, alpha=0.5)
        axis.axhline(0.0, color="k", lw=0.6, ls="--", alpha=0.7)

    s_ = quiver_step
    ax[1, 0].quiver(fld["X"][::s_, ::s_], fld["Y"][::s_, ::s_],
                    rd["d1"][::s_, ::s_], rd["d2"][::s_, ::s_],
                    color="k", scale=32, headwidth=2, pivot="mid", alpha=0.6)
    fig.suptitle(rf"{fld['field']} $\mu={fld['mu']}$: Riesz construction, "
                 rf"$\sigma_{{dir}}={sigma_dir:.3f}$", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"monogenic": mono, "director": rd, "angle_error_deg": dang}
