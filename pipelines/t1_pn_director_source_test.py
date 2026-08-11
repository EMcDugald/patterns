# pipelines/t1_pn_director_source_test.py
"""
T1 for Pismen-Nepomnyashchy: does the phase pipeline REPRODUCE a staircase?

The knee version of this test (pipelines/t1_director_source_test.py) asks
whether the pipeline invents a staircase on a field whose exact core phase
is linear. That test cannot see a method which suppresses structure, since
suppressing structure on the knee looks like success.

Here the exact core phase IS a staircase, so the same four director sources
are scored on whether they reproduce one.

    exact           k = grad(theta), differenced on the grid
    exact_oriented  the same, pushed through orient_vector_field_v2
    uhu             the current extraction, isotropic Gaussian
    riesz           the doubled-angle smoothed Riesz director

Ground truth. With theta = y - sign(y) log psi and psi solving the heat
equation in |y|, the erf steps telescope as y -> 0, giving psi = e^{-m pi}
with m(x) the number of defects to the left of x. Hence

    theta(x, 0+) = m(x) pi        (upper branch)
    theta(x, 0-) = -m(x) pi       (lower branch)

an exact staircase with a step of pi at each defect column. The sign(y)
matters: theta is ODD in y, so the field jumps by 2 m pi across y = 0.
Every centred y-difference straddling the core is therefore meaningless,
and every 2D comparison here is done on the UPPER half only. Verified
numerically against the direct erf sum, see _selftest.

Two consequences worth having in mind when reading the core panels:

  - a step of pi is half a stripe wavelength, so u alternates between +1
    and -1 on successive plateaus along the core row;
  - the core phase has ZERO local slope between defects but mean slope
    exactly k1, since a step of pi per spacing pi/k1 is k1. The knee
    reaches the same k1 with a uniform slope. So the "local |dphi/dx|"
    panel is referenced to 0 here, not to k1, and the running mean slope
    is shown alongside it.

Prediction being tested. Near the PN core between defects the true k is
vertical (theta_x = 0 there exactly). Near the knee core it is horizontal.
The eigen-flip is 90 degrees. If the same mechanism operates, uHu above
mu* should invent a staircase on the knee (already measured) and FLATTEN
one here: step heights below pi and fewer edges than defects, not extra
edges.

Box sizing. y_top = 2 n pi / k2_ff with k2_ff = 1 - k1^2/2, the actual PN
merged far field, which agrees with the knee's sqrt(1 - k1^2) to second
order in k1. The plane-wave zone is a WINDOW in y, not an asymptote:
below it the erf steps have not merged, above it the finite defect row
collapses into a single step. Both edges are set by the smoothing width
sqrt(2 y) against the spacing pi/k1 and the row half-width nd pi / (2 k1),
which gives

    nd ~ 10 sqrt(2 y_top) / Lsp = 10 k1 sqrt(2 y_top) / pi

(row half-width about five smoothing widths). That is the default, and the
plane-wave error at y_top is measured and printed every run rather than
assumed.

Run:
    python t1_pn_director_source_test.py
    python t1_pn_director_source_test.py --mu 0.90 --sources exact riesz --diagnose
    python t1_pn_director_source_test.py --riesz-sweep 0.90
    python t1_pn_director_source_test.py --knee-staircase
    python t1_pn_director_source_test.py --selftest
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.special import log_ndtr, erf as nerf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
for _p in (str(_SRC), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


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
orient_vector_field_v2 = _import_first(
    ["utils.kfield_calcs", "op_extract.kfield_calcs", "kfield_calcs"],
    "orient_vector_field_v2")
riesz_director = _import_first(
    ["utils.field_diagnostics_viz", "field_diagnostics_viz"], "riesz_director")
spectral_deriv = _import_first(
    ["utils.field_diagnostics_viz", "field_diagnostics_viz"], "spectral_deriv")
macro_iso = _import_first(
    ["utils.field_diagnostics_viz", "field_diagnostics_viz"], "macro_iso")

# borrowed from the knee harness so the two tests cannot drift apart
_KNEE = ["t1_director_source_test", "pipelines.t1_director_source_test"]
edge_taper = _import_first(_KNEE, "edge_taper")


# =====================================================================
# the exact PN field
# =====================================================================

def log_psi_pn(X, Y, xd, y_floor=1e-6):
    """
    log psi for the PN multi-defect solution, in a form with no
    cancellation, so it stays exact for any number of defects.

    The literal expression in pn_chain_field is

        psi = a + sum_j delta_j erf((x - c_j) / sqrt(2 |y|)),
        a = (1 + e^{-nd pi}) / 2,  delta_j = (e^{-(j+1)pi} - e^{-j pi}) / 2

    whose smallest value is e^{-nd pi} but which is assembled from terms of
    size 1/2. Past about eleven defects e^{-nd pi} drops below the roundoff
    of that sum and the right end of the staircase becomes noise.

    Substituting erf(z) = 1 - 2 p, p = erfc(z)/2, and telescoping the
    weights turns it into a sum of POSITIVE terms:

        psi = e^{-nd pi} + (1 - e^{-pi}) sum_j e^{-j pi} p_j

    with p_j = erfc((x - c_j)/sqrt(2|y|)) / 2 in (0, 1). Accumulating this
    with logaddexp, and taking log p_j from log_ndtr so the far tail does
    not underflow, is exact at any nd.

    Check of the limit: as y -> 0+, p_j -> 1 exactly when x < c_j, so the
    sum is e^{-m pi}(1 - e^{-(nd-m) pi}) / (1 - e^{-pi}) and psi collapses
    to e^{-m pi}, which is the staircase.
    """
    nd = len(xd)
    w = np.sqrt(2.0 * np.maximum(np.abs(Y), y_floor))
    lg = float(np.log1p(-np.exp(-np.pi)))
    acc = np.full(X.shape, -nd * np.pi, dtype=float)
    for j, c in enumerate(xd):
        z = (X - c) / w
        acc = np.logaddexp(acc, lg - j * np.pi + log_ndtr(-np.sqrt(2.0) * z))
    return acc


def pn_exact(mu, n_periods=8.0, nd=None, N=768, ly_factor=2.8,
             x_margin_spacings=2.0, **kw):
    """
    Exact PN field with the same dict interface as knee_exact, plus the
    quantities the PN scorer and the seed contour need.

    Sizing, all of it reported rather than assumed:
      k2_ff  = 1 - k1^2/2, the merged far field (not sqrt(1 - k1^2))
      y_top  = 2 n_periods pi / k2_ff, an integer number of vertical
               periods of the far-field wave
      nd     = 10 k1 sqrt(2 y_top) / pi rounded up to an even number, so
               the defect row is about five smoothing widths wide at y_top
      Lx     = nd Lsp + 2 drift + margin, where drift = y_top k1 / k2_ff is
               the horizontal displacement of a coordinate line falling
               from y_top to the core in the far field
      Ly     = ly_factor y_top

    nd is forced EVEN so that x = 0 sits midway between two defects and the
    seed contour is centred on a plateau rather than on a singular column.

    The y grid carries a half-cell offset so that no row sits exactly on
    y = 0. That matters here in a way it does not on the knee: theta is odd
    in y and jumps by 2 m pi across the core, so a row at y = 0 would be on
    neither branch.
    """
    k1 = float(np.sqrt(1.0 - mu**2))
    k2_ff = 1.0 - 0.5 * k1**2
    Lsp = np.pi / k1
    y_top = 2.0 * n_periods * np.pi / k2_ff

    if nd is None:
        nd = 10.0 * k1 * np.sqrt(2.0 * y_top) / np.pi
        nd = int(2 * np.ceil(nd / 2.0))
    nd = int(2 * ((int(nd) + 1) // 2))          # force even
    nd = max(nd, 2)

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
    u = np.cos(theta)

    k1g, k2g = exact_k_from_theta(theta, x, y)

    return {"x": x, "y": y, "X": X, "Y": Y, "u": u, "theta": theta,
            "k1": k1g, "k2": k2g,
            "k1_ff": k1, "k2_ff": k2_ff, "Lx": Lx, "Ly": Ly, "mu": mu,
            "y_top": y_top, "drift": drift, "nd": nd, "xd": xd, "Lsp": Lsp,
            "n_periods": n_periods,
            "iy_core": int(np.argmin(np.abs(y - 0.5 * Ly / N))),
            "field": "pn"}


def exact_k_from_theta(theta, x, y):
    """
    k = grad theta by differencing theta on the grid, NOT from the closed
    form. The analytic theta_x carries a 1/sqrt(y) factor on the defect
    columns, singular exactly where the coordinate lines terminate, so the
    closed form is unusable on the core row. theta itself is bounded
    everywhere.

    The y-difference is taken separately on the two branches, so nothing
    straddles the 2 m pi jump at y = 0. That makes this arm
    RESOLUTION-LIMITED rather than analytic, which is stated in the report:
    the step in theta at a defect has width sqrt(2 y), so on the first row
    above the core it is about sqrt(dy) wide.
    """
    k1 = np.gradient(theta, x, axis=1)
    k2 = np.empty_like(theta)
    up, lo = y > 0, y < 0
    if up.sum() > 1:
        k2[up] = np.gradient(theta[up], y[up], axis=0)
    if lo.sum() > 1:
        k2[lo] = np.gradient(theta[lo], y[lo], axis=0)
    return k1, k2


def planewave_error(fld, n_spacings=2.0):
    """max relative deviation of grad theta from (k1_ff, k2_ff) along the
    row nearest y_top, over a window of n_spacings defect spacings in x
    centred on the middle of the defect row. This is the number that says
    whether nd is large enough."""
    x, y = fld["x"], fld["y"]
    j = int(np.argmin(np.abs(y - fld["y_top"])))
    m = np.abs(x) <= n_spacings * fld["Lsp"]
    if m.sum() < 8 or j in (0, len(y) - 1):
        return np.nan, np.nan, y[j]
    tx = np.gradient(fld["theta"][j], x)[m]
    ty = ((fld["theta"][j + 1] - fld["theta"][j - 1])
          / (y[j + 1] - y[j - 1]))[m]
    return (float(np.max(np.abs(tx / fld["k1_ff"] - 1.0))),
            float(np.max(np.abs(ty / fld["k2_ff"] - 1.0))), float(y[j]))


# =====================================================================
# seed contour
# =====================================================================

def _theta_interp(fld):
    return RegularGridInterpolator((fld["y"], fld["x"]), fld["theta"],
                                   bounds_error=False, fill_value=np.nan)


def level_set_contour(fld, C, n_rows=600):
    """
    A true constant-phase contour of the exact PN field. There is no closed
    form in x, but theta is strictly monotone in x at fixed y (psi is
    decreasing in x and all the telescoped weights are positive), increasing
    for y > 0 and decreasing for y < 0. So one interpolation per row gives a
    single-valued curve with no ambiguity.

    UPPER ARM ONLY. theta is odd in y, so the lower arm of the same
    V-shaped contour sits at level -C, not +C, and asking for +C below the
    core throws away almost every row. Only upper-half seeds are used
    anyway (seed_half = "upper"), so the lower arm is not built.

    Rows where C falls outside the range of theta are dropped and counted.
    """
    x, y = fld["x"], fld["y"]
    y_top = fld["y_top"]
    rows = np.linspace(y_top, 0.0, n_rows)
    xs, ys, dropped = [], [], 0
    for yy in rows:
        j = int(np.argmin(np.abs(y - yy)))
        th = fld["theta"][j]
        if yy > 0:
            xp, fp = th, x
        else:
            xp, fp = th[::-1], x[::-1]
        if not (xp[0] <= C <= xp[-1]):
            dropped += 1
            continue
        xs.append(float(np.interp(C, xp, fp)))
        ys.append(float(y[j]))
    if dropped:
        print(f"  [warn] level set outside theta range on {dropped} / "
              f"{n_rows} rows; dropped")
    if not xs:
        raise RuntimeError("level set contour is empty; C is off the field")
    return np.vstack([np.array(xs), np.array(ys)])


def shoot_line(x0, y0, ik1, ik2, ds, max_steps=20000):
    """One coordinate line, stepping with exactly the pipeline's convention
    (dir_sign = -1 in the upper half), used only to place the contour."""
    for _ in range(max_steps):
        kx = float(ik1((y0, x0)))
        ky = float(ik2((y0, x0)))
        m = np.hypot(kx, ky)
        if m == 0.0:
            return np.nan
        s = -1.0 if y0 > 0 else 1.0
        xn, yn = x0 + s * kx / m * ds, y0 + s * ky / m * ds
        if y0 * yn <= 0.0:
            return xn
        x0, y0 = xn, yn
    return np.nan


def centered_level(fld, ds, n_probe=9, n_iter=2, verbose=True):
    """
    Choose the contour level C so the core interval that actually receives
    coordinate lines is centred on the defect row.

    Shifting C by dC translates the contour in x by dC / k1 in the far
    field, and translates the landing points by the same amount, so one
    correction per pass converges immediately. Two passes by default.
    """
    x, y = fld["x"], fld["y"]
    j_top = int(np.argmin(np.abs(y - fld["y_top"])))
    C = float(fld["theta"][j_top, int(np.argmin(np.abs(x)))])
    ik1 = RegularGridInterpolator((y, x), fld["k1"], bounds_error=False,
                                  fill_value=0.0)
    ik2 = RegularGridInterpolator((y, x), fld["k2"], bounds_error=False,
                                  fill_value=0.0)
    for it in range(n_iter):
        con = level_set_contour(fld, C, n_rows=240)
        up = con[1] > 0
        idx = np.linspace(0, up.sum() - 1, min(n_probe, up.sum())).astype(int)
        land = [shoot_line(con[0, up][i], con[1, up][i], ik1, ik2, ds)
                for i in idx]
        land = np.array([v for v in land if np.isfinite(v)])
        if land.size == 0:
            break
        mid = 0.5 * (land.min() + land.max())
        if verbose:
            print(f"  [contour] pass {it}: landings span "
                  f"[{land.min():.1f}, {land.max():.1f}], centre {mid:+.2f}")
        C = C - fld["k1_ff"] * mid
    return C


def clip_contour_to_row(fld, con, ds, cover_frac=0.9, n_probe=25,
                        verbose=True):
    """
    Keep only the arc of the contour whose coordinate lines land on the
    defect row.

    The contour spans about k2_ff y_top / k1 in x and the lines drift a
    further y_top k1 / k2_ff, so the core interval that receives samples is
    typically wider than the defect row itself. Seeding the whole contour
    spends most of the lines outside the staircase. Landing position is
    monotone along the contour, so one probe pass and an interpolation
    gives the arc worth keeping.
    """
    x, y = fld["x"], fld["y"]
    ik1 = RegularGridInterpolator((y, x), fld["k1"], bounds_error=False,
                                  fill_value=0.0)
    ik2 = RegularGridInterpolator((y, x), fld["k2"], bounds_error=False,
                                  fill_value=0.0)
    n = con.shape[1]
    idx = np.linspace(0, n - 1, min(n_probe, n)).astype(int)
    land = np.array([shoot_line(con[0, i], con[1, i], ik1, ik2, ds)
                     for i in idx])
    good = np.isfinite(land)
    if good.sum() < 3:
        return con
    ii, ll = idx[good].astype(float), land[good]
    order = np.argsort(ll)
    target = cover_frac * 0.5 * fld["nd"] * fld["Lsp"]
    i_lo = float(np.interp(-target, ll[order], ii[order]))
    i_hi = float(np.interp(+target, ll[order], ii[order]))
    a, b = int(max(0, min(i_lo, i_hi))), int(min(n - 1, max(i_lo, i_hi)))
    if b - a < 8:
        if verbose:
            print("  [contour] clip would leave too little arc; kept whole")
        return con
    if verbose:
        print(f"  [contour] probe landings span "
              f"[{ll.min():.1f}, {ll.max():.1f}], defect row is "
              f"[{-0.5 * fld['nd'] * fld['Lsp']:.1f}, "
              f"{0.5 * fld['nd'] * fld['Lsp']:.1f}]; keeping arc "
              f"{a} to {b} of {n}")
    return con[:, a:b + 1]


def build_seed_contour(fld, mode, ds, cover_frac=0.9, verbose=True):
    """`exact` returns a level set of theta, `trapezoid` returns None so the
    pipeline builds its own from an effective mu.

    build_knee_boundary_from_mu makes a trapezoid of slope k1/k2 with
    k2 = mu, which is the KNEE far field. PN's is (k1, 1 - k1^2/2). The two
    agree to second order in k1 and split at fourth. Rather than accept the
    mismatch, the trapezoid is asked for with the mu whose knee angle equals
    PN's actual one, so the only difference left between the two seeding
    modes is straight line against true level set."""
    if mode == "trapezoid":
        return None
    C = centered_level(fld, ds, verbose=verbose)
    con = level_set_contour(fld, C)
    return clip_contour_to_row(fld, con, ds, cover_frac=cover_frac,
                               verbose=verbose)


def trapezoid_mu(fld):
    k1, k2 = fld["k1_ff"], fld["k2_ff"]
    return float(k2 / np.hypot(k1, k2))


# =====================================================================
# the four director sources
# =====================================================================

def director_exact(fld, **kw):
    return fld["k1"].copy(), fld["k2"].copy()


def director_exact_oriented(fld, pi_tol=np.pi / 10, **kw):
    a, b = orient_vector_field_v2(fld["k1"], fld["k2"], pi_tol=pi_tol)
    return (np.ma.filled(a, 0.0).astype(float),
            np.ma.filled(b, 0.0).astype(float))


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


SOURCES = {
    "exact": director_exact,
    "exact_oriented": director_exact_oriented,
    "uhu": director_uhu,
    "riesz": director_riesz,
}


# =====================================================================
# run one (mu, source)
# =====================================================================

def rebuild_grid_pn(res, fld, gauge, half="orig"):
    """
    Re-grid the phase from the pipeline's own coordinate lines and Hilbert
    phases, changing only the per-line unwrapping origin. Local copy of the
    knee version; the single difference is that the seed_phase gauge reads
    theta from the PN field by interpolation instead of from the knee
    closed form.

      raw          the pipeline as it stands
      zero_origin  subtract each line's first value
      seed_phase   subtract the first value and add the exact theta at the
                   seed point; available only because this is analytic

    The seed-to-core branch is used, not the stitched line. Along the full
    stitched line the true phase is V-shaped but the Hilbert phase of an
    oscillatory record is monotone, so the reflected half accumulates with
    the wrong sign and griddata blends the two.
    """
    if gauge == "raw" and half == "stitched":
        return res["phase_grid_unwrapped"][:, :, 0]

    x, y = fld["x"], fld["y"]
    ith = _theta_interp(fld) if gauge == "seed_phase" else None
    lines = res["coordinate_lines"][0]
    phases = res["phase_lines_unwrapped"][0]

    pts, vals = [], []
    for ln, ph in zip(lines, phases):
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
            # the Hilbert phase accumulates as -theta, since upper-half
            # seeds step along -khat, so the seed phase enters with a plus
            # here and the result is aligned to -theta.
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


def run_one(fld, source, seed_bdry, seed_mode, n_seeds, sigma, ds, ramp,
            use_sym):
    k1d, k2d = SOURCES[source](fld, sigma=sigma, ramp=ramp, sigma_dir=sigma)

    uhu_data = {
        "x": fld["x"], "y": fld["y"],
        "u": fld["u"][:, :, None],
        "k1_sym": np.asarray(k1d)[:, :, None],
        "k2_sym": np.asarray(k2d)[:, :, None],
        "k1_orig": np.asarray(k1d)[:, :, None],
        "k2_orig": np.asarray(k2d)[:, :, None],
        "mu": fld["mu"], "tt": None, "ramp": None, "k": None, "A": None,
        "knee_bdry": seed_bdry,
        "source_path": (f"analytic PN mu={fld['mu']} nd={fld['nd']} "
                        f"director={source}"),
    }

    res = compute_pgb_phase_from_uhu(
        uhu_data, mu=trapezoid_mu(fld), n_phase_seeds=n_seeds, ds=ds,
        seed_half="upper", phase_ramp_mode="none", prefer_sym=True)

    lines = res["coordinate_lines"][0]
    reach, landing = [], []
    for ln in lines:
        ln = np.asarray(ln, float)
        if not len(ln):
            continue
        ic = int(np.argmin(np.abs(ln[:, 1])))
        reach.append(abs(ln[ic, 1]))
        landing.append(ln[ic, 0])
    res["frac_reaching_core"] = (float(np.mean(np.array(reach) < 0.5))
                                 if reach else 0.0)
    res["core_landings"] = np.array(landing)
    res["director"] = (np.asarray(k1d), np.asarray(k2d))

    key = ("phase_grid_symmetric_unwrapped" if use_sym
           else "phase_grid_unwrapped")
    return res[key][:, :, 0], res


# =====================================================================
# plateau detector, shared with the knee
# =====================================================================

def detrend_median(x, s):
    """Remove a x + c with a the MEDIAN of the local differences, which is
    what score_core does on the knee. On the knee that median is k1 and the
    staircase survives in the residual; on PN it is near zero between
    defects and the staircase survives untouched. One code path either
    way."""
    ok = np.isfinite(s)
    if ok.sum() < 8:
        return np.full_like(s, np.nan), np.nan, np.nan
    xs, ps = x[ok], s[ok]
    dx, dp = np.diff(xs), np.diff(ps)
    g = dx > 0
    a = float(np.median(dp[g] / dx[g])) if g.any() else 0.0
    c = float(np.median(ps - a * xs))
    return s - (a * x + c), a, c


def detect_plateaus(x, r, min_sep, thresh_frac=0.15, level_frac=0.6):
    """
    Locate the edges of a staircase and measure it.

    Edges are peaks of |dr/dx| above a threshold, which is the larger of
    thresh_frac of the maximum and a robust floor at the median plus five
    MADs, with peaks closer than min_sep merged. Plateau LEVELS are then
    medians over the central level_frac of each interval between edges,
    which is far more stable than reading a rise across the edge itself.
    Step heights are differences of consecutive levels.

    Returns positions, spacings, heights, widths, plus the finite-run
    bookkeeping, since griddata leaves gaps wherever the coordinate lines
    left none and a missing plateau has to be distinguishable from a
    missing sample.
    """
    ok = np.isfinite(r)
    out = {"n_gaps": 0, "gap_frac": float(1.0 - ok.mean()) if r.size else 1.0,
           "x_edges": np.array([]), "spacings": np.array([]),
           "heights": np.array([]), "widths": np.array([]),
           "levels": np.array([]), "x_used": np.array([]),
           "r_used": np.array([])}
    if ok.sum() < 16:
        return out

    # largest contiguous finite run, and how many gaps were passed over
    idx = np.flatnonzero(ok)
    brk = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[brk + 1]]
    ends = np.r_[idx[brk], idx[-1]]
    lens = ends - starts
    b = int(np.argmax(lens))
    out["n_gaps"] = int(len(starts) - 1)
    sl = slice(starts[b], ends[b] + 1)
    xs, rs = x[sl], r[sl]
    keep = np.isfinite(rs)
    xs, rs = xs[keep], rs[keep]
    if xs.size < 16:
        return out
    out["x_used"], out["r_used"] = xs, rs

    d = np.abs(np.gradient(rs, xs))
    med = np.median(d)
    mad = np.median(np.abs(d - med)) + 1e-12
    thr = max(thresh_frac * float(np.max(d)), med + 5.0 * mad)

    cand = np.flatnonzero((d > thr)
                          & (d >= np.r_[d[0], d[:-1]])
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

    # width at half the peak of |dr/dx|, a proxy for how sharp each step is
    widths = []
    for i in peaks:
        half = 0.5 * d[i]
        lo = i
        while lo > 0 and d[lo] > half:
            lo -= 1
        hi = i
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
        lo_, hi_ = xi[0], xi[-1]
        pad = 0.5 * (1.0 - level_frac) * (hi_ - lo_)
        mm = m & (x_between(xs, lo_ + pad, hi_ - pad))
        levels.append(float(np.median(rs[mm])) if mm.sum() >= 3
                      else float(np.median(rs[m])))
    levels = np.array(levels)

    out["x_edges"] = xs[peaks]
    out["spacings"] = np.diff(xs[peaks]) if peaks.size > 1 else np.array([])
    out["heights"] = np.diff(levels)
    out["widths"] = np.array(widths)
    out["levels"] = levels
    return out


def x_between(a, lo, hi):
    return (a >= lo) & (a <= hi)


# =====================================================================
# scoring
# =====================================================================

def align_to_truth_upper(phi, fld):
    """
    Put the reconstruction in the gauge of the exact theta by least squares
    on the two free conventions, phi ~ a theta + c, fitted on the UPPER HALF
    only. a should come out near -1 because upper-half seeds step along
    -khat. Restricting to the upper half is not cosmetic: theta is odd in y,
    so a fit over the whole box would be fitting two different branches at
    once.
    """
    y = fld["y"]
    up = np.tile((y > 0)[:, None], (1, len(fld["x"])))
    ok = np.isfinite(phi) & np.isfinite(fld["theta"]) & up
    if ok.sum() < 32:
        return np.full_like(phi, np.nan), np.nan, np.nan
    M = np.vstack([fld["theta"][ok], np.ones(ok.sum())]).T
    (a, c), *_ = np.linalg.lstsq(M, phi[ok], rcond=None)
    if abs(a) < 1e-6:
        return np.full_like(phi, np.nan), float(a), float(c)
    return (phi - c) / a, float(a), float(c)


def _stair_metrics(x, vals, ok, fld, thresh_frac=0.15, delta_frac=0.15):
    """
    Staircase metrics for one row, applied IDENTICALLY to the recovered
    phase and to the exact theta at the same row.

    Applying it to the exact row as well is not a formality. The true step
    at a defect has width sqrt(2 y), so a row four units above the core is
    genuinely close to a ramp and its own localization is near the ramp
    value. Scoring a reconstruction there against a target of 1 would call
    a perfect answer a failure. The target is whatever the exact field
    gives at that row, so it gets measured rather than assumed.
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
        out["step_rise"] = steps
        out["step_rise_mean"] = float(np.mean(steps))
        out["step_rise_spread"] = float(np.std(steps))
        out["between_rise_mean"] = float(np.mean(betw))
        if abs(tot) > 1e-9:
            out["localization"] = float(np.sum(steps[:-1]) / tot)

    pl = detect_plateaus(x, np.where(ok, vals, np.nan),
                         min_sep=0.35 * fld["Lsp"], thresh_frac=thresh_frac)
    out["plateaus"] = pl
    out["n_edges"] = int(pl["x_edges"].size)
    if pl["spacings"].size:
        out["spacing_mean"] = float(np.mean(pl["spacings"]))
        out["spacing_spread"] = float(np.std(pl["spacings"]))
    if pl["x_edges"].size and xd_in.size:
        d = np.abs(pl["x_edges"][:, None] - xd_in[None, :]).min(axis=1)
        out["edge_pos_err_med"] = float(np.median(d))
        out["edge_pos_err_max"] = float(d.max())
    return out


def jaggedness(x, phi, th, ok):
    """
    How much grid-scale wiggle the reconstruction carries that the truth
    does not. Second difference is the cheapest scale-free probe: it is
    dominated by the shortest wavelength present, so the ratio against the
    same quantity on the exact row says how much of the roughness is the
    method rather than the field. A principled smoother has to remove this
    without touching the steps, so it is worth having a number for it
    before designing one.
    """
    if ok.sum() < 8:
        return {}
    d2p = np.diff(np.where(ok, phi, np.nan)[ok], 2)
    d2t = np.diff(np.where(ok, th, np.nan)[ok], 2)
    rp = float(np.sqrt(np.nanmean(d2p**2)))
    rt = float(np.sqrt(np.nanmean(d2t**2)))
    return {"d2_rms": rp, "d2_rms_exact": rt,
            "d2_ratio": float(rp / rt) if rt > 1e-12 else np.nan}


def score_core_pn(phi_grid, fld, x_trim=0.15, thresh_frac=0.15,
                  core_y=None, phi_aligned=None):
    """
    Score the reconstructed core phase against the exact staircase on one
    row above y = 0.

    core_y = None selects the first row above the core, which is where the
    coordinate lines actually terminate. Any positive value selects the
    nearest row above it, which is how the resolution ladder is built: the
    true step is sqrt(2 y) wide, so on the first row it is about one cell
    and no grid can carry it, while a few units up it is several cells and
    the comparison is meaningful.
    """
    x, y = fld["x"], fld["y"]
    if core_y is None:
        iy = fld["iy_core"]
    else:
        cand = np.flatnonzero(y > 0)
        iy = int(cand[np.argmin(np.abs(y[cand] - core_y))])

    if phi_aligned is None:
        phi_al, a, c = align_to_truth_upper(phi_grid, fld)
    else:
        phi_al, a, c = phi_aligned

    phi = np.asarray(phi_al[iy, :], dtype=float)
    th = np.asarray(fld["theta"][iy, :], dtype=float)

    res = {"a_align": a, "c_align": c, "iy": iy, "y_core": float(y[iy]),
           "coverage": float(np.isfinite(phi).mean()),
           "n_ok": int(np.isfinite(phi).sum()),
           "step_width": float(np.sqrt(2.0 * y[iy])),
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

    res.update(_stair_metrics(x, phi, ok, fld, thresh_frac))
    ref = _stair_metrics(x, th, ok, fld, thresh_frac)
    res["ref"] = ref
    res.update(jaggedness(x, phi, th, ok))
    return res


# =====================================================================
# figures
# =====================================================================

def _show_u(axis, fld, ext):
    axis.imshow(fld["u"], extent=ext, origin="lower", cmap="gray",
                aspect="auto", alpha=0.85)


def sizing_figure(fld, path):
    """theta_x and theta_y along the row at y_top against (k1_ff, k2_ff),
    so the choice of nd is inspectable rather than trusted."""
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
    ax[0].legend(fontsize=8)
    ax[1].plot(x, ty, "C1-", lw=1.2)
    ax[1].axhline(fld["k2_ff"], color="k", ls="--", lw=1.2,
                  label=rf"$k_2=1-k_1^2/2={fld['k2_ff']:.4f}$")
    ax[1].set_ylabel(r"$\theta_y$ at $y_{top}$")
    ax[1].set_xlabel("x")
    ax[1].legend(fontsize=8)
    for a_ in ax:
        for c in fld["xd"]:
            a_.axvline(c, color="0.9", lw=0.6, zorder=0)
        a_.axvspan(-2 * fld["Lsp"], 2 * fld["Lsp"], color="C2", alpha=0.08,
                   zorder=0)
    fig.suptitle(rf"$\mu={fld['mu']}$, $n_d={fld['nd']}$, "
                 rf"$y_{{top}}={yj:.1f}$: plane-wave error "
                 rf"{100 * ex:.2f}% in $\theta_x$, {100 * ey:.2f}% in "
                 rf"$\theta_y$ (shaded window)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def diagnostic_figure_pn(fld, source, res, phi, path, quiver_step=24,
                         line_step=6, sc=None):
    """Nine panels. Row 1 is what went into the reconstruction, row 2 is the
    reconstruction against the truth on the UPPER HALF, row 3 is the core."""
    x, y = fld["x"], fld["y"]
    X, Y = fld["X"], fld["Y"]
    theta = fld["theta"]
    k1, Lsp = fld["k1_ff"], fld["Lsp"]
    ext = [x[0], x[-1], y[0], y[-1]]
    up = y > 0
    ext_up = [x[0], x[-1], y[up][0], y[up][-1]]

    if sc is None:
        sc = score_core_pn(phi, fld)
    phi_al, a, c = align_to_truth_upper(phi, fld)

    fig, ax = plt.subplots(3, 3, figsize=(17.0, 13.5))

    # --- row 1 ---------------------------------------------------------
    _show_u(ax[0, 0], fld, ext)
    sb = res.get("knee_bdry_phase")
    if sb is not None:
        sb = np.asarray(sb)
        ax[0, 0].plot(sb[0], sb[1], ".", color="deepskyblue", ms=3.0)
    for cxd in fld["xd"]:
        ax[0, 0].axvline(cxd, color="lime", lw=0.7, alpha=0.65)
    ax[0, 0].set_title(f"pattern, seeds, {fld['nd']} defect columns")

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

    # --- row 2, upper half only ----------------------------------------
    lo_, hi_ = np.nanpercentile(theta[up], [1, 99])
    for axis, f_, ttl in ((ax[1, 0], theta, r"exact $\theta$ (upper branch)"),
                          (ax[1, 1], phi_al,
                           f"recovered (aligned, a={a:.3f})")):
        im = axis.imshow(f_[up], extent=ext_up, origin="lower",
                         cmap="twilight", vmin=lo_, vmax=hi_, aspect="auto")
        fig.colorbar(im, ax=axis)
        axis.set_title(ttl)

    err = phi_al - theta
    e_up = np.where(up[:, None], err, np.nan)[up]
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
        for cxd in fld["xd"]:
            axis.axvline(cxd, color="0.4", lw=0.4, alpha=0.5)

    # --- row 3, the core ------------------------------------------------
    iy = sc["iy"]
    th_c = theta[iy]
    ph_c = sc.get("phi", np.full_like(x, np.nan))

    ax[2, 0].step(x, th_c / np.pi, "k-", lw=1.6, where="mid",
                  label=r"exact $\theta/\pi$")
    ax[2, 0].plot(x, ph_c / np.pi, "C3-", lw=1.2, label=r"recovered$/\pi$")
    ax[2, 0].plot(x, 0.5 * fld["u"][iy] - 1.0, color="0.7", lw=0.7,
                  label="u on core (offset)")
    for cxd in fld["xd"]:
        ax[2, 0].axvline(cxd, color="0.88", lw=0.7, zorder=0)
    ax[2, 0].set_title(rf"core row $y={sc['y_core']:+.3f}$: steps of $\pi$ "
                       f"at {fld['nd']} defects")
    ax[2, 0].set_ylabel(r"phase / $\pi$")
    ax[2, 0].legend(fontsize=8)
    ax[2, 0].set_xlabel("x")

    r = sc.get("resid_vs_exact")
    if r is not None:
        ax[2, 1].plot(x, r, "C0-", lw=1.2, label="vs exact staircase")
    saw = sc.get("sawtooth")
    if saw is not None:
        ax[2, 1].plot(x, saw, "C5-", lw=1.0, alpha=0.8,
                      label=rf"about best-fit line (ptp/$\pi$="
                            rf"{sc.get('sawtooth_ptp', np.nan) / np.pi:.2f})")
    for lv in (-np.pi / 2, 0.0, np.pi / 2):
        ax[2, 1].axhline(lv, color="0.6", ls=":", lw=0.9)
    for cxd in fld["xd"]:
        ax[2, 1].axvline(cxd, color="0.88", lw=0.7, zorder=0)
    pl = sc.get("plateaus", {})
    for xe in pl.get("x_edges", []):
        ax[2, 1].axvline(xe, color="C3", lw=0.9, ls="--", alpha=0.8)
    ax[2, 1].set_title(rf"core residuals; dashed = detected edges; "
                       rf"localization {sc.get('localization', np.nan):.2f} "
                       rf"(1 staircase, {sc.get('ramp_localization', 0.3):.2f} ramp)")
    ax[2, 1].legend(fontsize=7)
    ax[2, 1].set_xlabel("x")

    okc = np.isfinite(ph_c)
    if okc.sum() > 4:
        dphi = np.abs(np.gradient(ph_c[okc], x[okc]))
        ax[2, 2].plot(x[okc], dphi, "C2-", lw=1.0,
                      label=r"local $|d\phi/dx|$")
        w = max(int(Lsp / (x[1] - x[0])), 3)
        run = np.convolve(dphi, np.ones(w) / w, mode="same")
        ax[2, 2].plot(x[okc], run, "C4-", lw=1.4,
                      label=rf"running mean over $\pi/k_1$")
    ax[2, 2].axhline(0.0, color="k", ls="-", lw=0.8)
    ax[2, 2].axhline(k1, color="k", ls="--", lw=1.2, label=f"$k_1$={k1:.3f}")
    ax[2, 2].set_ylim(-0.05, 3.0 * max(k1, 0.1))
    ax[2, 2].set_title(r"core slope: 0 between defects, mean $k_1$")
    ax[2, 2].legend(fontsize=8)
    ax[2, 2].set_xlabel("x")

    ttl = (rf"PN  $\mu={fld['mu']}$  $n_d={fld['nd']}$  director: {source}"
           f"   edges {sc.get('n_edges', 0)} of "
           f"{sc.get('n_defects_in_window', 0)} in window")
    fig.suptitle(ttl, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def landing_figure(fld, landings, path):
    """Where the coordinate lines terminate on the core, per source. A
    plateau with no lines landing on it is a sampling gap, not a director
    failure, and the two have to be told apart."""
    fig, ax = plt.subplots(figsize=(10.0, 1.1 + 0.9 * len(landings)))
    for i, (src, ld) in enumerate(landings.items()):
        ld = np.asarray(ld)
        ld = ld[np.isfinite(ld)]
        ax.plot(ld, np.full_like(ld, i), "|", ms=12, label=src)
    for c in fld["xd"]:
        ax.axvline(c, color="0.85", lw=0.8, zorder=0)
    ax.set_yticks(range(len(landings)))
    ax.set_yticklabels(list(landings.keys()), fontsize=8)
    ax.set_xlabel("x on the core")
    ax.set_title(rf"$\mu={fld['mu']}$: coordinate-line landings, "
                 "grey lines at defect columns", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def row_ladder_figure(fld, ladder, path):
    """
    How each director does as a function of how close to the core it is
    scored. ladder is {source: [score dicts, one per row]}.

    The point of the panel is that the exact arm is not a fixed standard
    here. Its error is set by whether the grid can carry a step of width
    sqrt(2 y), so it moves with the row, and the ranking of the three
    sources can and does change along the ladder.
    """
    fig, ax = plt.subplots(1, 3, figsize=(16.0, 4.6))
    for src, scs in ladder.items():
        ys = [s["y_core"] for s in scs if "rms" in s]
        if not ys:
            continue
        ax[0].plot(ys, [s["rms"] for s in scs if "rms" in s], "o-", lw=1.4,
                   label=f"{src} director")
        ax[1].plot(ys, [s.get("sawtooth_ptp", np.nan) / np.pi
                        for s in scs if "rms" in s], "o-", lw=1.4,
                   label=f"{src} director")
        ax[2].plot(ys, [s.get("localization", np.nan)
                        for s in scs if "rms" in s], "o-", lw=1.4,
                   label=f"{src} director")
    any_scs = next(iter(ladder.values()))
    ys = [s["y_core"] for s in any_scs if "rms" in s]
    ax[1].plot(ys, [s["ref"].get("sawtooth_ptp", np.nan) / np.pi
                    for s in any_scs if "rms" in s], "k--", lw=1.6,
               label="truth at this row")
    ax[2].plot(ys, [s["ref"].get("localization", np.nan)
                    for s in any_scs if "rms" in s], "k--", lw=1.6,
               label="truth at this row")

    ax[0].set_yscale("log")
    ax[0].set_ylabel("rms vs exact staircase (rad)")
    ax[1].set_ylabel(r"sawtooth ptp / $\pi$")
    ax[2].set_ylabel("localization")
    for a_ in ax:
        a_.set_xscale("log")
        a_.set_xlabel("y of the scored row")
        a_.legend(fontsize=8)
        a_.grid(alpha=0.25)
    sw = [s["step_width"] / (fld["x"][1] - fld["x"][0])
          for s in any_scs if "rms" in s]
    ax[0].set_title("step width at these rows: "
                    + ", ".join(f"{v:.1f}" for v in sw) + " cells",
                    fontsize=9)
    fig.suptitle(rf"PN $\mu={fld['mu']}$: director performance against "
                 "proximity to the core", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def profiles_figure(fld, ladder, path, zoom=None, n_spacings=2.5):
    """
    Core profiles at every row on the ladder, all sources against the exact
    row. zoom = None draws the full covered window; a float centres a
    window of n_spacings defect spacings on that x, which is what shows
    whether the recovered curve is jagged at the grid scale.
    """
    any_scs = next(iter(ladder.values()))
    rows = [i for i, s in enumerate(any_scs) if "rms" in s]
    if not rows:
        return
    nr = len(rows)
    fig, ax = plt.subplots(nr, 2, figsize=(15.0, 3.1 * nr), squeeze=False)
    cols = {"exact": "C0", "exact_oriented": "C1", "uhu": "C2",
            "riesz": "C3"}
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
            a0.plot(x, s_["phi"] / np.pi, cols.get(src, "C4"), lw=1.1,
                    label=f"{src} director")
            a1.plot(x, s_["resid_vs_exact"], cols.get(src, "C4"), lw=1.1,
                    label=f"{src} (rms {s_['rms']:.3f})")
        for c in fld["xd"]:
            a0.axvline(c, color="0.88", lw=0.6, zorder=0)
            a1.axvline(c, color="0.88", lw=0.6, zorder=0)
        a1.axhline(0.0, color="0.4", lw=0.8)
        if zoom is not None:
            w = n_spacings * fld["Lsp"] / 2.0
            a0.set_xlim(zoom - w, zoom + w)
            a1.set_xlim(zoom - w, zoom + w)
            m = (x > zoom - w) & (x < zoom + w) & np.isfinite(sc0["phi"])
            if m.any():
                lo = np.nanmin(sc0["theta_core"][m]) / np.pi
                hi = np.nanmax(sc0["theta_core"][m]) / np.pi
                a0.set_ylim(lo - 0.6, hi + 0.6)
        a0.set_ylabel(rf"$y={sc0['y_core']:.2f}$" "\n" r"phase / $\pi$")
        a0.set_title(f"step width {sc0['step_width']:.2f} = "
                     f"{sc0['step_width'] / (x[1] - x[0]):.1f} cells",
                     fontsize=9)
        a1.set_title("recovered minus exact", fontsize=9)
        if r == 0:
            a0.legend(fontsize=7, ncol=2)
            a1.legend(fontsize=7)
    ax[-1, 0].set_xlabel("x")
    ax[-1, 1].set_xlabel("x")
    ttl = (rf"PN $\mu={fld['mu']}$: core profiles"
           + ("" if zoom is None
              else rf", zoom on $x={zoom:.1f}$ over {n_spacings} spacings"))
    fig.suptitle(ttl, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=140)
    plt.close(fig)


# =====================================================================
# knee staircase measurement
# =====================================================================

def knee_staircase(args, report):
    """Run the same plateau detector on the KNEE uHu output, which replaces
    reading step positions off a plot with a measurement. On the knee the
    exact core phase is linear, so every edge found here is spurious and its
    spacing is the thing to compare against pi/k1."""
    knee_exact = _import_first(_KNEE, "knee_exact")
    knee_run_one = _import_first(_KNEE, "run_one")
    knee_rebuild = _import_first(_KNEE, "rebuild_grid")
    knee_score = _import_first(_KNEE, "score_core")

    report("")
    report("=" * 78)
    report("plateau detector on the KNEE uHu staircase (all edges spurious)")
    report("=" * 78)
    report(f"  {'mu':>6} {'k1':>7} {'pi/k1':>7} {'edges':>6} "
           f"{'spacing mean':>13} {'spread':>7} {'height mean':>12} "
           f"{'spread':>7}")
    for mu in args.knee_mu:
        fld = knee_exact(mu, n_spacings=args.n_spacings, N=args.knee_N)
        ramp = edge_taper(fld["x"], fld["y"], frac=args.taper)
        _, res = knee_run_one(fld, "uhu", "exact", args.n_seeds, args.sigma,
                              args.ds, ramp, False)
        phi = knee_rebuild(res, fld, "seed_phase", "orig")
        sc = knee_score(phi, fld, x_trim=args.x_trim)
        if "resid" not in sc:
            report(f"  {mu:>6.3f}   too few valid points")
            continue
        pl = detect_plateaus(sc["x"], sc["resid"],
                             min_sep=0.35 * np.pi / fld["k1_ff"],
                             thresh_frac=args.thresh_frac)
        sp, h = pl["spacings"], pl["heights"][np.isfinite(pl["heights"])]
        report(f"  {mu:>6.3f} {fld['k1_ff']:>7.4f} "
               f"{np.pi / fld['k1_ff']:>7.3f} {pl['x_edges'].size:>6} "
               f"{(np.mean(sp) if sp.size else np.nan):>13.3f} "
               f"{(np.std(sp) if sp.size else np.nan):>7.3f} "
               f"{(np.mean(h) if h.size else np.nan):>12.3f} "
               f"{(np.std(h) if h.size else np.nan):>7.3f}")


# =====================================================================
# self test of the ground truth
# =====================================================================

def _selftest():
    """Check the stable log psi against the literal erf sum where the
    literal sum is still trustworthy, and check the staircase limit."""
    print("selftest: stable log psi vs literal erf sum, and the staircase")
    for mu, nd in ((0.90, 8), (0.85, 10)):
        k1 = np.sqrt(1 - mu**2)
        Lsp = np.pi / k1
        x = np.linspace(-0.6 * nd * Lsp, 0.6 * nd * Lsp, 501)
        y = np.array([0.02, 0.5, 5.0, 40.0])
        X, Y = np.meshgrid(x, y)
        xd = Lsp * (np.arange(nd) - (nd - 1) / 2.0)

        a = 0.5 * (1 + np.exp(-np.pi * nd))
        sq = np.sqrt(2 * np.maximum(np.abs(Y), 1e-6))
        th = np.arange(nd + 1) * np.pi
        psi = np.full_like(X, a)
        for j, c in enumerate(xd):
            psi += 0.5 * (np.exp(-th[j + 1]) - np.exp(-th[j])) \
                * nerf((X - c) / sq)
        lit = np.log(np.maximum(psi, 1e-300))
        sta = log_psi_pn(X, Y, xd)
        print(f"  mu={mu} nd={nd}: max |log psi difference| = "
              f"{np.max(np.abs(lit - sta)):.3e}")

        yc = np.array([1e-4])
        Xc, Yc = np.meshgrid(x, yc)
        theta_c = Yc - np.sign(Yc) * log_psi_pn(Xc, Yc, xd)
        m = np.sum(x[:, None] > xd[None, :], axis=1)
        far = np.min(np.abs(x[:, None] - xd[None, :]), axis=1) > 0.3 * Lsp
        err = np.max(np.abs(theta_c[0][far] - (yc[0] + m[far] * np.pi)))
        print(f"           max |theta - (y + m pi)| off-defect = {err:.3e}")

    for nd in (8, 20, 40):
        k1 = np.sqrt(1 - 0.90**2)
        Lsp = np.pi / k1
        xd = Lsp * (np.arange(nd) - (nd - 1) / 2.0)
        x = np.linspace(-0.6 * nd * Lsp, 0.6 * nd * Lsp, 401)
        X, Y = np.meshgrid(x, np.array([1e-4]))
        theta_c = Y - np.sign(Y) * log_psi_pn(X, Y, xd)
        m = np.sum(x[:, None] > xd[None, :], axis=1)
        far = np.min(np.abs(x[:, None] - xd[None, :]), axis=1) > 0.3 * Lsp
        err = np.max(np.abs(theta_c[0][far] - (1e-4 + m[far] * np.pi)))
        print(f"  nd={nd:>3}: staircase error at y=1e-4 = {err:.3e} "
              f"(total rise {nd * np.pi:.1f})")


# =====================================================================

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    lines = []

    def report(s=""):
        print(s, flush=True)
        lines.append(s)

    if args.selftest:
        _selftest()
        return

    report("=" * 96)
    report("T1 for PN: reconstructed core phase vs the exact staircase, by "
           "director source")
    report("=" * 96)
    report("  exact core phase: theta(x, 0+) = m(x) pi, step pi at every "
           "defect, mean slope k1")
    report("  rms and max are against the exact staircase after alignment; "
           "edges are detected, not assumed")
    report(f"  seed contour: {args.seed}   sigma: {args.sigma:.4f}   "
           f"seeds: {args.n_seeds}   ds: {args.ds}   N: {args.N}")
    report("")

    if args.knee_staircase:
        knee_staircase(args, report)
        with open(os.path.join(args.outdir, "t1_pn_report.txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        return

    store = {}
    for mu in args.mu:
        fld = pn_exact(mu, n_periods=args.n_periods, nd=args.nd, N=args.N,
                       ly_factor=args.ly_factor)
        ex, ey, yj = planewave_error(fld)
        report(f"  mu={mu:.3f}  k1={fld['k1_ff']:.4f}  "
               f"k2_ff={fld['k2_ff']:.4f}  Lsp={fld['Lsp']:.3f}  "
               f"nd={fld['nd']}  y_top={fld['y_top']:.1f}")
        report(f"     box Lx={fld['Lx']:.1f} Ly={fld['Ly']:.1f}  "
               f"dx={fld['x'][1] - fld['x'][0]:.3f} "
               f"dy={fld['y'][1] - fld['y'][0]:.3f}  "
               f"core row y={fld['y'][fld['iy_core']]:+.4f}")
        report(f"     plane-wave error at y={yj:.1f}: "
               f"{100 * ex:.2f}% in theta_x, {100 * ey:.2f}% in theta_y"
               + ("   [warn] nd may be too small" if max(ex, ey) > 0.02
                  else ""))
        w_step = np.sqrt(2 * abs(fld["y"][fld["iy_core"]]))
        report(f"     exact-k arm is resolution limited: the step at a "
               f"defect is {w_step:.3f} wide on the core row, "
               f"{w_step / (fld['x'][1] - fld['x'][0]):.1f} cells")
        sizing_figure(fld, os.path.join(
            args.outdir, f"pn_sizing_mu{mu:.3f}.png".replace(".", "p", 1)))

        ramp = edge_taper(fld["x"], fld["y"], frac=args.taper)
        seed_bdry = build_seed_contour(fld, args.seed, args.ds, args.cover_frac)

        report("")
        report(f"  {'source':>16} {'y row':>7} {'cells':>6} {'cover':>6} "
               f"{'a':>7} {'rms':>8} {'max|r|':>8} {'local':>6} "
               f"{'(exact)':>8} {'saw/pi':>7} {'(exact)':>8} {'edges':>6} "
               f"{'ndef':>5} {'spacing':>8} {'pi/k1':>7} {'d2 ratio':>9}")
        landings, ladder = {}, {}
        for src in args.sources:
            try:
                phi_pipe, res = run_one(fld, src, seed_bdry, args.seed,
                                        args.n_seeds, args.sigma, args.ds,
                                        ramp, args.symmetric)
            except Exception as exc:
                report(f"  {src:>16}   FAILED: {exc}")
                continue
            landings[src] = res["core_landings"]
            frac = res.get("frac_reaching_core", np.nan)
            if frac < 0.9:
                report(f"  {src:>16}   [warn] only {100 * frac:.0f} percent "
                       f"of lines reach the core")
            phi = rebuild_grid_pn(res, fld, args.gauge, args.half)
            # align once, on the upper half, and reuse for every row so the
            # gauge cannot drift between rows of the same run
            al = align_to_truth_upper(phi, fld)

            scs = []
            for cy in args.core_rows:
                sc = score_core_pn(phi, fld, x_trim=args.x_trim,
                                   thresh_frac=args.thresh_frac,
                                   core_y=(None if cy <= 0 else cy),
                                   phi_aligned=al)
                scs.append(sc)
                if "rms" not in sc:
                    report(f"  {src:>16} {sc['y_core']:>7.2f}   too few "
                           f"valid points on this row ({sc['n_ok']})")
                    continue
                ref = sc["ref"]
                report(f"  {src:>16} {sc['y_core']:>7.2f} "
                       f"{sc['step_width'] / (fld['x'][1] - fld['x'][0]):>6.1f} "
                       f"{sc['coverage']:>6.2f} {sc['a_align']:>7.3f} "
                       f"{sc['rms']:>8.4f} {sc['max_abs']:>8.4f} "
                       f"{sc.get('localization', np.nan):>6.2f} "
                       f"{ref.get('localization', np.nan):>8.2f} "
                       f"{sc.get('sawtooth_ptp', np.nan) / np.pi:>7.2f} "
                       f"{ref.get('sawtooth_ptp', np.nan) / np.pi:>8.2f} "
                       f"{sc['n_edges']:>6} "
                       f"{sc.get('n_defects_in_window', 0):>5} "
                       f"{sc.get('spacing_mean', np.nan):>8.3f} "
                       f"{fld['Lsp']:>7.3f} "
                       f"{sc.get('d2_ratio', np.nan):>9.2f}")
                pl = sc["plateaus"]
                if pl["n_gaps"]:
                    report(f"  {' ':>16}   [note] {pl['n_gaps']} sampling "
                           f"gaps on this row, "
                           f"{100 * pl['gap_frac']:.0f}% not covered")
            ladder[src] = scs
            if scs and "rms" in scs[0]:
                store[(mu, src)] = (fld, scs[0])
            if args.diagnose:
                dp = os.path.join(
                    args.outdir,
                    f"pn_diag_mu{mu:.3f}_{src}.png".replace(".", "p", 1))
                diagnostic_figure_pn(fld, src, res, phi, dp, sc=scs[0])
                report(f"  {' ':>16}   diagnostic: {dp}")
            report("")

        if ladder:
            tag = f"mu{mu:.3f}".replace(".", "p", 1)
            p_l = os.path.join(args.outdir, f"pn_row_ladder_{tag}.png")
            row_ladder_figure(fld, ladder, p_l)
            p_p = os.path.join(args.outdir, f"pn_profiles_{tag}.png")
            profiles_figure(fld, ladder, p_p)
            zc = fld["xd"][len(fld["xd"]) // 2]
            p_z = os.path.join(args.outdir, f"pn_profiles_zoom_{tag}.png")
            profiles_figure(fld, ladder, p_z, zoom=float(zc),
                            n_spacings=args.zoom_spacings)
            report(f"  figures: {p_l}")
            report(f"           {p_p}")
            report(f"           {p_z}")

        if landings:
            landing_figure(fld, landings, os.path.join(
                args.outdir,
                f"pn_landings_mu{mu:.3f}.png".replace(".", "p", 1)))
        report("")

    # ---- summary figure: core row, one panel per mu --------------------
    if store:
        nmu = len(args.mu)
        fig, axes = plt.subplots(nmu, 1, figsize=(10.0, 3.0 * nmu),
                                 squeeze=False)
        for i, mu in enumerate(args.mu):
            ax = axes[i, 0]
            drawn = False
            for src, cl in zip(args.sources, ("C0", "C1", "C2", "C3")):
                if (mu, src) not in store:
                    continue
                fld, sc = store[(mu, src)]
                if "phi" not in sc:
                    continue
                ax.plot(sc["x"], sc["phi"] / np.pi, cl, lw=1.2, label=src)
                drawn = True
            if drawn:
                ax.step(fld["x"], fld["theta"][fld["iy_core"]] / np.pi, "k-",
                        lw=1.6, where="mid", label="exact")
                for c in fld["xd"]:
                    ax.axvline(c, color="0.88", lw=0.6, zorder=0)
            ax.set_ylabel(r"core phase / $\pi$")
            ax.set_title(rf"$\mu={mu}$   grey lines at defect columns",
                         fontsize=9)
            if i == 0 and ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7, ncol=len(args.sources) + 1)
        axes[-1, 0].set_xlabel("x")
        fig.tight_layout()
        p = os.path.join(args.outdir, "pn_core_phase_by_source.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        report(f"figure: {p}")

    # ---- riesz width sweep --------------------------------------------
    if args.riesz_sweep is not None:
        report("")
        report("=" * 78)
        report(f"Riesz width sweep on PN at mu = {args.riesz_sweep}")
        report("=" * 78)
        fld = pn_exact(args.riesz_sweep, n_periods=args.n_periods,
                       nd=args.nd, N=args.N, ly_factor=args.ly_factor)
        ramp = edge_taper(fld["x"], fld["y"], frac=args.taper)
        seed_bdry = build_seed_contour(fld, args.seed, args.ds, args.cover_frac)
        report(f"  {'sigma_dir':>10} {'a':>7} {'rms':>8} {'max|r|':>8} "
               f"{'local':>6} {'saw/pi':>7} {'edges':>6} {'ndef':>5} "
               f"{'height':>8} {'slope':>7} {'k1':>7}")
        curves = []
        for sg in args.sweep_sigmas:
            try:
                phi_pipe, res = run_one(fld, "riesz", seed_bdry, args.seed,
                                        args.n_seeds, sg, args.ds, ramp,
                                        args.symmetric)
            except Exception as exc:
                report(f"  {sg:>10.3f}   FAILED: {exc}")
                continue
            phi = rebuild_grid_pn(res, fld, args.gauge, args.half)
            sc = score_core_pn(phi, fld, x_trim=args.x_trim,
                               thresh_frac=args.thresh_frac)
            if "rms" not in sc:
                report(f"  {sg:>10.3f}   too few valid points")
                continue
            curves.append((sg, sc))
            report(f"  {sg:>10.3f} {sc['a_align']:>7.3f} {sc['rms']:>8.4f} "
                   f"{sc['max_abs']:>8.4f} "
                   f"{sc.get('localization', np.nan):>6.2f} "
                   f"{sc.get('sawtooth_ptp', np.nan) / np.pi:>7.2f} "
                   f"{sc['n_edges']:>6} {sc['n_defects_in_window']:>5} "
                   f"{sc.get('step_rise_mean', np.nan):>8.3f} "
                   f"{sc.get('mean_slope', np.nan):>7.4f} "
                   f"{fld['k1_ff']:>7.4f}")
        if curves:
            fig, ax = plt.subplots(2, 1, figsize=(10.0, 7.0), sharex=False)
            for sg, sc in curves:
                ax[0].plot(sc["x"], sc["phi"] / np.pi, lw=1.1,
                           label=rf"$\sigma_{{dir}}={sg:.2f}$")
            ax[0].step(fld["x"], fld["theta"][fld["iy_core"]] / np.pi, "k-",
                       lw=1.6, where="mid", label="exact")
            for c in fld["xd"]:
                ax[0].axvline(c, color="0.88", lw=0.6, zorder=0)
            ax[0].set_ylabel(r"core phase / $\pi$")
            ax[0].legend(fontsize=7, ncol=3)
            ax[0].set_xlabel("x")
            sgs = [c[0] for c in curves]
            ax[1].plot(sgs, [c[1]["rms"] for c in curves], "o-", label="rms")
            ax[1].plot(sgs, [c[1].get("height_mean", np.nan) for c in curves],
                       "s-", label="mean step height")
            ax[1].axhline(np.pi, color="0.6", ls=":", lw=1.0,
                          label=r"$\pi$")
            ax[1].set_xlabel(r"$\sigma_{dir}$")
            ax[1].legend(fontsize=8)
            fig.suptitle(rf"Riesz width sweep, PN $\mu={args.riesz_sweep}$")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            p = os.path.join(args.outdir, "pn_riesz_width_sweep.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            report(f"figure: {p}")

    with open(os.path.join(args.outdir, "t1_pn_report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


class _Args:
    outdir = "out_t1_pn"
    mu = [0.85, 0.90, 0.95]
    sources = ["exact", "exact_oriented", "uhu", "riesz"]
    seed = "exact"                 # or "trapezoid"
    sigma = np.pi / 2
    ds = 0.15
    n_seeds = 256
    N = 768
    cover_frac = 0.9
    core_rows = [0.0, 2.0, 4.0]
    zoom_spacings = 2.5
    n_periods = 8.0
    nd = None                      # None means the sizing rule
    ly_factor = 2.8
    taper = 0.04
    x_trim = 0.15
    thresh_frac = 0.15
    symmetric = False
    diagnose = False
    gauge = "seed_phase"
    half = "orig"
    riesz_sweep = None
    sweep_sigmas = [0.2, 0.4, 0.8, np.pi / 2, 2.5, np.pi]
    knee_staircase = False
    knee_mu = [0.85, 0.90, .95]
    knee_N = 512
    n_spacings = 8.0
    selftest = False


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=_Args.outdir)
    p.add_argument("--mu", type=float, nargs="+", default=_Args.mu)
    p.add_argument("--sources", nargs="+", default=_Args.sources,
                   choices=list(SOURCES))
    p.add_argument("--seed", default=_Args.seed,
                   choices=["trapezoid", "exact"])
    p.add_argument("--sigma", type=float, default=_Args.sigma)
    p.add_argument("--ds", type=float, default=_Args.ds)
    p.add_argument("--n-seeds", type=int, default=_Args.n_seeds)
    p.add_argument("--N", type=int, default=_Args.N)
    p.add_argument("--cover-frac", type=float, default=_Args.cover_frac)
    p.add_argument("--core-rows", type=float, nargs="+",
                   default=_Args.core_rows,
                   help="y of each scored row; 0 means the first row above "
                        "the core, where the lines terminate")
    p.add_argument("--zoom-spacings", type=float,
                   default=_Args.zoom_spacings)
    p.add_argument("--n-periods", type=float, default=_Args.n_periods)
    p.add_argument("--nd", type=int, default=_Args.nd)
    p.add_argument("--ly-factor", type=float, default=_Args.ly_factor)
    p.add_argument("--taper", type=float, default=_Args.taper)
    p.add_argument("--x-trim", type=float, default=_Args.x_trim)
    p.add_argument("--thresh-frac", type=float, default=_Args.thresh_frac)
    p.add_argument("--gauge", default=_Args.gauge,
                   choices=["raw", "zero_origin", "seed_phase"])
    p.add_argument("--half", default=_Args.half,
                   choices=["orig", "stitched"])
    p.add_argument("--symmetric", action="store_true",
                   default=_Args.symmetric)
    p.add_argument("--diagnose", action="store_true", default=_Args.diagnose)
    p.add_argument("--riesz-sweep", type=float, default=_Args.riesz_sweep)
    p.add_argument("--sweep-sigmas", type=float, nargs="+",
                   default=_Args.sweep_sigmas)
    p.add_argument("--knee-staircase", action="store_true",
                   default=_Args.knee_staircase)
    p.add_argument("--knee-mu", type=float, nargs="+", default=_Args.knee_mu)
    p.add_argument("--knee-N", type=int, default=_Args.knee_N)
    p.add_argument("--n-spacings", type=float, default=_Args.n_spacings)
    p.add_argument("--selftest", action="store_true", default=_Args.selftest)
    main(p.parse_args())
