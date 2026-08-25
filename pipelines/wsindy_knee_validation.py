# pipelines/wsindy_knee_validation.py
"""
Weak-form (WSINDy) validation on the analytic evolving knee.

v1 (this file): first weak-form script of THREAD A. Everything here is a
METHOD test on synthetic data with a known law; nothing is a claim about SH.
v1.3: pair-augmented Pareto selection. fit_target's STLSQ threshold sweep
guarantees all single-term candidates but not all pairs; at production
resolution this left the admissible exact pair {Lap w, (div k)^2} out of the
pool, so Pareto chose 6- and 3-term models even though the forced 2-term fit
(R2 0.9983) was within the 1 percent band of the best (0.9997). fit_and_read
now adds every 2-term OLS candidate to the returned sweep and re-applies the
identical Pareto rule (sparsest within rel_tol of best test R2) over the
union. fit_target itself is untouched.
v1.2: adds --sm_sigma, the measurement-consistent library option. Measured
justification: the riesz-extracted w equals G_sigma_avg * w_true to 2.3
percent rel rms (best-fit width = sigma_avg = pi exactly), so the law obeyed
by the MEASURED variable is d_t w_s = nu Lap w_s - 2 nu G*((div k)^2); the
smoothing commutes with d_t and Lap but not with squaring. With the
(div k)^2 column smoothed by the same kernel (sm_sigma = pi), the forced
two-term fit on the riesz route went from R2 0.902 / Lap-w error 60 percent
to R2 0.997 / both coefficients within 5-6 percent (on-core, full band,
eta = 0). Applied to the nonlinear-in-derivative columns (div k)^2 and
|grad w|; w and w^2 are already functions of the measured variable and are
left alone. sm_sigma = 0 disables (prior behavior).
v1.1: adds the TUNE stage (--stage tune) and a global --rel_tol. Motivation:
the production A2 runs recovered the correct term identities but never the
exact two-term support; the persistent extras have coefficients 3-10x below
the leading terms and each buys more than the 1 percent Pareto tolerance.
TUNE refits the best cells (fixed radius, eta list) over a rel_tol_pareto
sweep to test whether the extras prune at a looser tolerance (selection
tuning) or survive (real structure in the extracted fields).

Why weak form (vs the v3 post-hoc library smoother): the regression rows are
no longer pointwise statements of the PDE. Each row is the PDE integrated
against a compactly supported test function phi(x, y, t); integration by
parts moves derivatives off the data and onto phi, which we differentiate
analytically. The data enters only through undifferentiated fields
integrated against smooth kernels. In particular the time derivative of w is
NEVER computed from the data: the target column is -Int[ w phi_t ].

Known law (from effective_vs_bare_dwdt): theta = k1 x + phi(y, t) with
phi_t = nu phi_yy implies the closed exact w-law

    d_t w = nu Lap w - 2 nu (div k)^2 ,   w = |k|^2 - 1,

with NO w^2, NO w, NO constant. Falsifiable target for every stage.

Stages
------
A1  Assembly check, exact route. The weak fit must recover support
    {Lap w, (div k)^2} with coefficients within a few percent of
    (nu, -2 nu) and R^2 ~ 1. Validates kernels, IBP signs, integration.
    Rides along: the doubled-pair operator check. On the knee the director
    angle psi = atan2(k2, k1) is known from the exact k, so the exact
    d/dt of (cos 2 psi, sin 2 psi) is available. The weak time-derivative
    column built from the RIESZ director must match it. This validates the
    weak machinery on the one target type (orientation) whose failure mode
    (branch/flip artifacts) the w-fit cannot see. Operator-level only; no
    sparse law is fitted for orientation here.

A2  Recovery diagram, riesz route. Sweep outer test-function radius
    (units of lambda_x) against inner div-mollifier width eta (units of
    lambda_x; eta = 0 means raw np.gradient). Success per cell = selected
    support is exactly {Lap w, (div k)^2}; failure modes recorded (w^2
    selected = the known chevron substitution). The pointwise v3 result
    lives at the (radius -> 0, eta = 0) corner and fails there.

A3  Jitter robustness. Add spatially smooth, temporally white noise to the
    extracted k components (the measured grad-route disease is frame
    incoherence), rebuild all columns from the jittered k, and fit the same
    law pointwise (finite-difference dw/dt target) vs weak (integral
    target) at a fixed good (radius, eta). Expect the pointwise fit to
    degrade with jitter amplitude while the weak fit holds.

Weak-form details
-----------------
Test functions are separable, phi = f(x) g(y) h(t), each factor a
Messenger-Bortz piecewise-polynomial bump

    f(s) = (1 - (s/a)^2)^p   on |s| <= a,   0 outside,

(p >= 3 so f'' is continuous at the edge), or optionally a truncated
Gaussian (--kernel gauss, sigma = a/3) for continuity with the v3 Gaussian
smoother. Each 1d factor is mass-normalized (Int f = 1) so weak columns are
comparable in scale to pointwise values. IBP identities used:

    Int[ w_t   phi ] = -Int[ w  phi_t ]              (target)
    Int[ Lap w phi ] =  Int[ w  Lap phi ]
    Int[ div k phi ] = -Int[ k . grad phi ]

Terms nonlinear in a derivative cannot be fully weakened. (div k)^2 and
|grad w| use an INNER weak derivative: k (or w) is correlated against the
analytic derivative of a small 2d mollifier of half-width eta, the
nonlinearity is applied pointwise, and the result is integrated against the
outer phi. eta is a declared, swept parameter (the principled cousin of the
v3 hand-tuned 0.25 lambda_x smoothing); eta = 0 falls back to np.gradient.

All integrals are separable correlations evaluated at query centers on a
decimated grid, restricted to a valid mask with margins >= the kernel half
widths (so boundary handling never matters).

Frozen layer untouched; imports only. evolve_knee and k_of_route are reused
by import from effective_vs_bare_dwdt.

Output (relative): results/wsindy_knee_validation/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from op_extract.t1_common import riesz_director, macro_iso         # noqa: E402
from effective_vs_bare_dwdt import evolve_knee, k_of_route         # noqa: E402
from self_dual_sindy import fit_target, r_squared                  # noqa: E402


LIB = ["1", "w", "w^2", "Lap w", "|grad w|", "div k", "(div k)^2"]
TRUE_SUPPORT = {"Lap w", "(div k)^2"}


class _FitArgs:
    thresholds = tuple(np.geomspace(1e-4, 1.0, 25))
    rel_tol_pareto = 0.01
    n_boot = 100
    boot_frac = 0.5

    def __init__(self, rel_tol=None):
        if rel_tol is not None:
            self.rel_tol_pareto = float(rel_tol)


# --------------------------------------------------------------------------
# 1d kernel factory: sampled factor + analytic first/second derivatives
# --------------------------------------------------------------------------

def kernel_1d(half_width, spacing, kind="poly", p=6):
    """Return (f, f1, f2, m): samples of the mass-normalized factor and its
    first two derivatives on offsets j = -m..m (s_j = j * spacing).

    poly:  f(s) = (1 - (s/a)^2)^p on |s| <= a (a = m * spacing), 0 outside.
    gauss: f(s) = exp(-s^2 / 2 sig^2), sig = a / 3, truncated at |s| = a.
    """
    m = max(2, int(round(half_width / spacing)))
    a = m * spacing
    s = np.arange(-m, m + 1) * spacing
    if kind == "poly":
        if p < 3:
            raise ValueError("poly kernel needs p >= 3 for continuous f''")
        q = 1.0 - (s / a) ** 2
        q = np.clip(q, 0.0, None)
        f = q ** p
        f1 = p * q ** (p - 1) * (-2.0 * s / a ** 2)
        f2 = (p * (p - 1) * q ** (p - 2) * (4.0 * s ** 2 / a ** 4)
              + p * q ** (p - 1) * (-2.0 / a ** 2))
    elif kind == "gauss":
        sig = a / 3.0
        f = np.exp(-0.5 * (s / sig) ** 2)
        f1 = f * (-s / sig ** 2)
        f2 = f * ((s ** 2 - sig ** 2) / sig ** 4)
    else:
        raise ValueError(f"unknown kernel kind {kind!r}")
    Z = f.sum() * spacing                       # mass normalization
    return f / Z, f1 / Z, f2 / Z, m


def corr1d(F, kern, axis, spacing):
    """Discrete Int[ F(s) kern(s - c) ds ] along one axis, evaluated at every
    c: correlation (kernel NOT flipped), times the spacing. Data layout is
    [Ny, Nx, Nt] with axis 0 = y, 1 = x, 2 = t."""
    k = kern[::-1]
    shape = [1, 1, 1][: F.ndim]
    shape[axis] = len(k)
    return fftconvolve(F, k.reshape(shape), mode="same", axes=axis) * spacing


def corr_sep(F, kx, ky, kt, dx, dy, dt):
    """Separable 3d correlation: Int F(s) kx(sx-cx) ky(sy-cy) kt(st-ct) ds."""
    G = corr1d(F, ky, 0, dy)
    G = corr1d(G, kx, 1, dx)
    return corr1d(G, kt, 2, dt)


# --------------------------------------------------------------------------
# inner weak derivative (for terms nonlinear in a derivative)
# --------------------------------------------------------------------------

def inner_deriv(F, eta, dx, dy, axis, kind="poly", p=6):
    """d/d(axis) of F, per frame, by correlation against the analytic
    derivative of a 2d separable mollifier of half-width eta (physical
    units). eta = 0 falls back to np.gradient. F is [Ny, Nx] or
    [Ny, Nx, Nt] (2d ops act on the first two axes)."""
    if eta <= 0:
        d = dx if axis == "x" else dy
        ax = 1 if axis == "x" else 0
        return np.gradient(F, d, axis=ax)
    fx, fx1, _, _ = kernel_1d(eta, dx, kind, p)
    fy, fy1, _, _ = kernel_1d(eta, dy, kind, p)
    if axis == "x":
        # Int F(s) d/dsx[m(s - c)] ds = -(dF/dx weakly at c) with a sign:
        # Int F m_x(s-c) ds = -Int F_x m(s-c) ds  => weak F_x = -corr(F, m_x)
        G = corr1d(F, -fx1, 1, dx)
        return corr1d(G, fy, 0, dy)
    G = corr1d(F, -fy1, 0, dy)
    return corr1d(G, fx, 1, dx)


def meas_smooth(F, sm_sigma, Lx, Ly):
    """Per-frame macro_iso at the measurement width (identity if 0)."""
    if sm_sigma <= 0:
        return F
    out = np.empty_like(F)
    for t in range(F.shape[-1]):
        out[:, :, t] = macro_iso(F[:, :, t], sm_sigma, Lx, Ly)
    return out


# --------------------------------------------------------------------------
# stack the per-frame fields for one route
# --------------------------------------------------------------------------

def route_stacks(route, theta, x, y, sigma_dir):
    """k1, k2, w as [Ny, Nx, Nt] stacks for a route."""
    Lx, Ly = x[-1] - x[0], y[-1] - y[0]
    K1, K2 = [], []
    for t in range(theta.shape[-1]):
        k1, k2 = k_of_route(route, theta[:, :, t], x, y, Lx, Ly, sigma_dir)
        K1.append(np.asarray(k1))
        K2.append(np.asarray(k2))
    k1 = np.stack(K1, axis=-1)
    k2 = np.stack(K2, axis=-1)
    return k1, k2, k1 ** 2 + k2 ** 2 - 1.0


# --------------------------------------------------------------------------
# weak system assembly
# --------------------------------------------------------------------------

def weak_system(k1, k2, w, x, y, tt, args, radius_lam, eta_lam,
                band_y=None):
    """Assemble (Theta, b, xs) at query centers.

    radius_lam: outer spatial half-width in units of lambda_x
    eta_lam:    inner mollifier half-width in units of lambda_x (0 = raw)
    Columns follow LIB. Target b = Int[w_t phi] = -Int[w phi_t].
    """
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dts = np.diff(tt)
    if not np.allclose(dts, dts[0], rtol=1e-6):
        raise ValueError("weak assembly assumes uniform frame spacing")
    dt = float(dts[0])

    lam_x = 2 * np.pi / np.sqrt(1.0 - args.mu ** 2)
    a = radius_lam * lam_x
    ay = args.radius_y_lam * lam_x
    eta = eta_lam * lam_x

    fx, fx1, fx2, mx = kernel_1d(a, dx, args.kernel, args.poly_p)
    fy, fy1, fy2, my = kernel_1d(ay, dy, args.kernel, args.poly_p)
    ht, ht1, _, mt = kernel_1d(args.t_half * dt, dt, args.kernel, args.poly_pt)

    # pointwise nonlinear-in-derivative stacks (inner weak derivative)
    divk = inner_deriv(k1, eta, dx, dy, "x", args.kernel, args.poly_p) \
        + inner_deriv(k2, eta, dx, dy, "y", args.kernel, args.poly_p)
    wx = inner_deriv(w, eta, dx, dy, "x", args.kernel, args.poly_p)
    wy = inner_deriv(w, eta, dx, dy, "y", args.kernel, args.poly_p)

    Lx, Ly = x[-1] - x[0], y[-1] - y[0]
    gradw = meas_smooth(np.hypot(wx, wy), args.sm_sigma, Lx, Ly)
    divk2 = meas_smooth(divk ** 2, args.sm_sigma, Lx, Ly)

    base = dict(kx=fx, ky=fy, kt=ht)
    C = {}
    C["1"] = np.ones_like(w)                       # Int 1*phi = 1 (unit mass)
    C["w"] = corr_sep(w, fx, fy, ht, dx, dy, dt)
    C["w^2"] = corr_sep(w ** 2, fx, fy, ht, dx, dy, dt)
    C["Lap w"] = (corr_sep(w, fx2, fy, ht, dx, dy, dt)
                  + corr_sep(w, fx, fy2, ht, dx, dy, dt))
    C["|grad w|"] = corr_sep(gradw, fx, fy, ht, dx, dy, dt)
    C["div k"] = -(corr_sep(k1, fx1, fy, ht, dx, dy, dt)
                   + corr_sep(k2, fx, fy1, ht, dx, dy, dt))
    C["(div k)^2"] = corr_sep(divk2, fx, fy, ht, dx, dy, dt)
    b = -corr_sep(w, fx, fy, ht1, dx, dy, dt)

    # query centers: strides, margins >= kernel half widths (+ route rim)
    ny, nx, nt = w.shape
    rim = int(np.ceil(args.rim_lam * lam_x / dx))
    ix = np.arange(mx + rim, nx - mx - rim, args.stride_xy)
    iy = np.arange(my + rim, ny - my - rim, args.stride_xy)
    it = np.arange(mt, nt - mt, args.stride_t)
    if band_y is not None:
        iy = iy[np.abs(y[iy]) <= band_y]
    if min(len(ix), len(iy), len(it)) == 0:
        return None
    IY, IX, IT = np.meshgrid(iy, ix, it, indexing="ij")

    Theta = np.column_stack([C[nm][IY, IX, IT].ravel() for nm in LIB])
    bv = b[IY, IX, IT].ravel()
    xs = x[IX.ravel()]
    fin = np.all(np.isfinite(Theta), axis=1) & np.isfinite(bv)
    return Theta[fin], bv[fin], xs[fin], base


def pointwise_system(k1, k2, w, x, y, tt, args, eta_lam, band_y=None):
    """The pointwise baseline on the SAME stacks: finite-difference dw/dt
    target, pointwise columns (inner derivative at the same eta so the only
    difference from the weak system is the row functional)."""
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    lam_x = 2 * np.pi / np.sqrt(1.0 - args.mu ** 2)
    eta = eta_lam * lam_x

    divk = inner_deriv(k1, eta, dx, dy, "x", args.kernel, args.poly_p) \
        + inner_deriv(k2, eta, dx, dy, "y", args.kernel, args.poly_p)
    wx = inner_deriv(w, eta, dx, dy, "x", args.kernel, args.poly_p)
    wy = inner_deriv(w, eta, dx, dy, "y", args.kernel, args.poly_p)
    lapw = np.gradient(wx, dx, axis=1) + np.gradient(wy, dy, axis=0)

    Lx, Ly = x[-1] - x[0], y[-1] - y[0]
    cols = {"1": np.ones_like(w), "w": w, "w^2": w ** 2, "Lap w": lapw,
            "|grad w|": meas_smooth(np.hypot(wx, wy), args.sm_sigma, Lx, Ly),
            "div k": divk,
            "(div k)^2": meas_smooth(divk ** 2, args.sm_sigma, Lx, Ly)}
    dwdt = np.gradient(w, tt, axis=2)

    ny, nx, nt = w.shape
    rim = int(np.ceil(args.rim_lam * lam_x / dx)) + 4
    ix = np.arange(rim, nx - rim, args.stride_xy)
    iy = np.arange(rim, ny - rim, args.stride_xy)
    it = np.arange(1, nt - 1, args.stride_t)
    if band_y is not None:
        iy = iy[np.abs(y[iy]) <= band_y]
    if min(len(ix), len(iy), len(it)) == 0:
        return None
    IY, IX, IT = np.meshgrid(iy, ix, it, indexing="ij")
    Theta = np.column_stack([cols[nm][IY, IX, IT].ravel() for nm in LIB])
    bv = dwdt[IY, IX, IT].ravel()
    xs = x[IX.ravel()]
    fin = np.all(np.isfinite(Theta), axis=1) & np.isfinite(bv)
    return Theta[fin], bv[fin], xs[fin]


# --------------------------------------------------------------------------
# fit + report helpers
# --------------------------------------------------------------------------

SKIPPED = dict(n=0, coefs={}, support=[], r2_full=float("nan"),
               r2_holdout=float("nan"), exact_support=False, has_w2=False,
               skipped=True)


def fit_and_read(system, rng, max_samples=40000, min_samples=50,
                 rel_tol=None):
    if system is None:
        return dict(SKIPPED)
    Theta, bv, xs = system[0], system[1], system[2]
    if len(bv) < min_samples:
        return dict(SKIPPED)
    if len(bv) > max_samples:
        idx = rng.choice(len(bv), max_samples, replace=False)
        Theta, bv, xs = Theta[idx], bv[idx], xs[idx]
    tr = xs < np.median(xs)
    res = fit_target(LIB, Theta, bv, _FitArgs(rel_tol), rng, tr)

    # pair augmentation: add every 2-term OLS candidate, re-apply Pareto
    sweep = list(res["sweep"])
    p_ = Theta.shape[1]
    for i in range(p_):
        for j in range(i + 1, p_):
            supp = np.zeros(p_, dtype=bool)
            supp[[i, j]] = True
            coef = np.zeros(p_)
            sol, *_ = np.linalg.lstsq(Theta[tr][:, [i, j]], bv[tr],
                                      rcond=None)
            coef[[i, j]] = sol
            sweep.append(dict(support=supp, coef=coef, n_terms=2,
                              r2_train=r_squared(bv[tr], Theta[tr] @ coef),
                              r2_test=r_squared(bv[~tr], Theta[~tr] @ coef)))
    rt = _FitArgs(rel_tol).rel_tol_pareto
    best = max(sw["r2_test"] for sw in sweep)
    adm = [sw for sw in sweep if sw["n_terms"] > 0
           and sw["r2_test"] >= best - rt * abs(best)]
    adm.sort(key=lambda sw: (sw["n_terms"], -sw["r2_test"]))
    chosen = adm[0]
    supp_mask = chosen["support"]
    coef_full = np.zeros(p_)
    coef_full[supp_mask] = np.linalg.lstsq(Theta[:, supp_mask], bv,
                                           rcond=None)[0]
    res = dict(res, chosen=chosen, support=supp_mask, coef_full=coef_full,
               r2_full=r_squared(bv, Theta @ coef_full))

    supp = np.where(res["support"])[0]
    coefs = {LIB[i]: float(res["coef_full"][i]) for i in supp}
    return dict(n=int(len(bv)), coefs=coefs,
                coef_vec=[float(c) for c in res["coef_full"]],
                support=sorted(coefs.keys()),
                r2_full=float(res["r2_full"]),
                r2_holdout=float(res["chosen"]["r2_test"]),
                exact_support=set(coefs) == TRUE_SUPPORT,
                has_w2="w^2" in coefs)


def coef_errors(coefs, nu):
    """Relative errors of the two true coefficients (None if absent)."""
    e_lap = (abs(coefs["Lap w"] - nu) / nu) if "Lap w" in coefs else None
    e_div = (abs(coefs["(div k)^2"] + 2 * nu) / (2 * nu)) \
        if "(div k)^2" in coefs else None
    return e_lap, e_div


def fmt_fit(r, nu):
    if r.get("skipped"):
        return "SKIPPED (support exceeds domain or too few rows)"
    el, ed = coef_errors(r["coefs"], nu)
    terms = " ".join(f"{c:+.4g}*{nm}"
                     for nm, c in sorted(r["coefs"].items(),
                                         key=lambda kv: -abs(kv[1])))
    tail = []
    if el is not None:
        tail.append(f"lap err {100 * el:.1f}%")
    if ed is not None:
        tail.append(f"div err {100 * ed:.1f}%")
    return (f"support={r['support']} exact={r['exact_support']} "
            f"R2(hold)={r['r2_holdout']:.4f} n={r['n']}\n"
            f"    dw/dt = {terms}   [{', '.join(tail) if tail else 'n/a'}]")


# --------------------------------------------------------------------------
# stage A1: exact-route assembly check + doubled-pair operator check
# --------------------------------------------------------------------------

def stage_a1(theta, x, y, tt, nu, args, rng, out, log):
    log("=" * 72)
    log("A1  exact route, weak assembly check "
        f"(radius {args.a1_radius} lam_x, eta 0)")
    k1, k2, w = route_stacks("exact", theta, x, y, args.sigma_dir)
    r = fit_and_read(weak_system(k1, k2, w, x, y, tt, args,
                                 args.a1_radius, 0.0, band_y=args.band_y),
                     rng, rel_tol=args.rel_tol)
    log("  " + fmt_fit(r, nu))
    ok = r["exact_support"]
    el, ed = coef_errors(r["coefs"], nu)
    if ok and el is not None and ed is not None:
        ok = el < 0.05 and ed < 0.05 and r["r2_holdout"] > 0.99
    log(f"  A1 {'PASS' if ok else 'FAIL'} "
        f"(need exact support, coef errs < 5%, holdout R2 > 0.99)")

    # doubled-pair operator check -----------------------------------------
    log("-" * 72)
    log("A1b doubled-pair weak time derivative vs exact (operator level)")
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dt = float(tt[1] - tt[0])
    Lx, Ly = x[-1] - x[0], y[-1] - y[0]
    lam_x = 2 * np.pi / np.sqrt(1.0 - args.mu ** 2)

    # exact doubled pair from exact k (unit-normalized; k1 > 0 on the knee)
    n2 = np.maximum(k1 ** 2 + k2 ** 2, 1e-12)
    c2e = (k1 ** 2 - k2 ** 2) / n2
    s2e = 2 * k1 * k2 / n2
    dc2e = np.gradient(c2e, tt, axis=2)
    ds2e = np.gradient(s2e, tt, axis=2)

    # riesz doubled pair from the director of u = cos(theta)
    c2r = np.empty_like(c2e)
    s2r = np.empty_like(s2e)
    for t in range(theta.shape[-1]):
        rd = riesz_director(np.cos(theta[:, :, t]), Lx, Ly, args.sigma_dir)
        c2r[:, :, t] = np.cos(2 * np.asarray(rd["angle"]))
        s2r[:, :, t] = np.sin(2 * np.asarray(rd["angle"]))

    fxk, _, _, mx = kernel_1d(args.a1_radius * lam_x, dx, args.kernel,
                              args.poly_p)
    fyk, _, _, my = kernel_1d(args.radius_y_lam * lam_x, dy, args.kernel,
                              args.poly_p)
    htk, ht1k, _, mt = kernel_1d(args.t_half * dt, dt, args.kernel,
                                 args.poly_pt)
    rim = int(np.ceil(args.rim_lam * lam_x / dx))
    ny, nx, nt = c2e.shape
    ix = np.arange(mx + rim, nx - mx - rim, args.stride_xy)
    iy = np.arange(my + rim, ny - my - rim, args.stride_xy)
    iy = iy[np.abs(y[iy]) <= args.band_y]
    it = np.arange(mt, nt - mt, args.stride_t)
    IY, IX, IT = np.meshgrid(iy, ix, it, indexing="ij")

    rep = {}
    for nm, riesz_f, exact_dt in (("cos2psi", c2r, dc2e),
                                  ("sin2psi", s2r, ds2e)):
        weak = -corr_sep(riesz_f, fxk, fyk, ht1k, dx, dy, dt)[IY, IX, IT]
        truth = corr_sep(exact_dt, fxk, fyk, htk, dx, dy, dt)[IY, IX, IT]
        point = np.gradient(riesz_f, tt, axis=2)[IY, IX, IT]
        scale = max(float(np.sqrt(np.mean(truth ** 2))), 1e-15)
        rw = float(np.sqrt(np.mean((weak - truth) ** 2))) / scale
        rp = float(np.sqrt(np.mean((point - truth) ** 2))) / scale
        rep[nm] = dict(rel_rms_weak=rw, rel_rms_pointwise=rp)
        log(f"  {nm}: rel rms weak {rw:.3f} | pointwise {rp:.3f} "
            f"(truth rms {scale:.3e})")
    log("  (weak should track exact; on this clean field pointwise may be "
        "comparable, the separation appears under jitter in A3)")
    return dict(fit=r, a1_pass=bool(ok), doubled_pair=rep)


# --------------------------------------------------------------------------
# stage A2: riesz-route recovery diagram over (radius, eta)
# --------------------------------------------------------------------------

def stage_a2(theta, x, y, tt, nu, args, rng, out, log):
    log("=" * 72)
    log("A2  riesz route, recovery over outer radius x inner eta "
        "(units of lambda_x)")
    k1, k2, w = route_stacks("riesz", theta, x, y, args.sigma_dir)

    radii = list(args.a2_radii)
    etas = list(args.a2_etas)
    grid = {}
    for a in radii:
        for e in etas:
            r = fit_and_read(weak_system(k1, k2, w, x, y, tt, args, a, e,
                                         band_y=args.band_y), rng,
                             rel_tol=args.rel_tol)
            grid[(a, e)] = r
            log(f"  radius={a:<4} eta={e:<5} " + fmt_fit(r, nu))

    # pointwise baseline (the v3 corner) at eta = 0
    pw = fit_and_read(pointwise_system(k1, k2, w, x, y, tt, args, 0.0,
                                       band_y=args.band_y), rng,
                      rel_tol=args.rel_tol)
    log("  pointwise eta=0 baseline: " + fmt_fit(pw, nu))

    # figure: recovery diagram
    fig, ax = plt.subplots(figsize=(1.6 * len(etas) + 2.5,
                                    1.1 * len(radii) + 2))
    M = np.zeros((len(radii), len(etas)))
    for i, a in enumerate(radii):
        for j, e in enumerate(etas):
            r = grid[(a, e)]
            M[i, j] = (np.nan if r.get("skipped") else
                       2 if r["exact_support"] else
                       (0 if r["has_w2"] else 1))
    im = ax.imshow(M, origin="lower", cmap="RdYlGn", vmin=0, vmax=2,
                   aspect="auto")
    for i, a in enumerate(radii):
        for j, e in enumerate(etas):
            r = grid[(a, e)]
            if r.get("skipped"):
                ax.text(j, i, "skip", ha="center", va="center", fontsize=8)
                continue
            el, ed = coef_errors(r["coefs"], nu)
            lines = [f"R2 {r['r2_holdout']:.2f}"]
            if r["exact_support"] and el is not None and ed is not None:
                lines.append(f"{100 * max(el, ed):.0f}% coef")
            elif r["has_w2"]:
                lines.append("w^2!")
            else:
                lines.append(",".join(r["support"]) or "0")
            ax.text(j, i, "\n".join(lines), ha="center", va="center",
                    fontsize=8)
    ax.set_xticks(range(len(etas)), [str(e) for e in etas])
    ax.set_yticks(range(len(radii)), [str(a) for a in radii])
    ax.set_xlabel("inner eta (lambda_x)")
    ax.set_ylabel("outer radius (lambda_x)")
    ax.set_title("A2 riesz-route recovery: green = exact support "
                 "{Lap w, (div k)^2}, red = w^2 substituted")
    fig.tight_layout()
    fig.savefig(out / "a2_recovery_diagram.png", dpi=160)
    plt.close(fig)
    log(f"  figure -> {out / 'a2_recovery_diagram.png'}")
    return dict(grid={f"{a}|{e}": r for (a, e), r in grid.items()},
                pointwise_eta0=pw)


# --------------------------------------------------------------------------
# stage A3: jitter robustness, pointwise vs weak
# --------------------------------------------------------------------------

def jitter_fields(k1, k2, amp, x, args, rng):
    """Spatially smooth (scale ~ jit_scale * lambda_x), temporally white
    perturbation of the k components; amp is absolute in k units
    (|k| ~ 1, so amp = 0.05 is 5 percent wavenumber jitter)."""
    if amp <= 0:
        return k1, k2
    dx = float(x[1] - x[0])
    lam_x = 2 * np.pi / np.sqrt(1.0 - args.mu ** 2)
    spx = args.jit_scale * lam_x / dx
    ny, nx, nt = k1.shape

    def noise():
        n = rng.standard_normal((ny, nx, nt))
        for t in range(nt):
            n[:, :, t] = gaussian_filter(n[:, :, t], spx, mode="reflect")
            n[:, :, t] /= max(n[:, :, t].std(), 1e-12)
        return n

    return k1 + amp * noise(), k2 + amp * noise()


def stage_a3(theta, x, y, tt, nu, args, rng, out, log):
    log("=" * 72)
    log(f"A3  jitter robustness, riesz route, weak at radius="
        f"{args.a3_radius} eta={args.a3_eta} vs pointwise eta="
        f"{args.a3_eta}")
    k1o, k2o, wo = route_stacks("riesz", theta, x, y, args.sigma_dir)

    # clean systems, assembled once: the fitted coefficients from each
    # jittered fit are scored against these (r2_clean = does the LEARNED
    # LAW match the clean dynamics, independent of unexplainable noise
    # variance in the jittered target)
    Wc = weak_system(k1o, k2o, wo, x, y, tt, args, args.a3_radius,
                     args.a3_eta, band_y=args.band_y)
    Pc = pointwise_system(k1o, k2o, wo, x, y, tt, args, args.a3_eta,
                          band_y=args.band_y)

    def clean_r2(sys_clean, r):
        if sys_clean is None or r.get("skipped"):
            return float("nan")
        Th, bc = sys_clean[0], sys_clean[1]
        return float(r_squared(bc, Th @ np.asarray(r["coef_vec"])))

    rows = []
    for amp in args.a3_amps:
        k1, k2 = jitter_fields(k1o, k2o, amp, x, args, rng)
        w = k1 ** 2 + k2 ** 2 - 1.0
        rw = fit_and_read(weak_system(k1, k2, w, x, y, tt, args,
                                      args.a3_radius, args.a3_eta,
                                      band_y=args.band_y), rng)
        rp = fit_and_read(pointwise_system(k1, k2, w, x, y, tt, args,
                                           args.a3_eta,
                                           band_y=args.band_y), rng)
        rw["r2_clean"] = clean_r2(Wc, rw)
        rp["r2_clean"] = clean_r2(Pc, rp)
        log(f"  amp={amp:<5} weak:      r2_clean={rw['r2_clean']:.3f}  "
            + fmt_fit(rw, nu))
        log(f"  amp={amp:<5} pointwise: r2_clean={rp['r2_clean']:.3f}  "
            + fmt_fit(rp, nu))
        rows.append(dict(amp=amp, weak=rw, pointwise=rp))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    amps = [r["amp"] for r in rows]
    ax.plot(amps, [r["weak"]["r2_clean"] for r in rows], "o-",
            label="weak (r2 of learned law on clean system)")
    ax.plot(amps, [r["pointwise"]["r2_clean"] for r in rows], "s--",
            label="pointwise (same metric)")
    for r in rows:
        for key, dy_ in (("weak", 0.03), ("pointwise", -0.06)):
            ax.annotate("ok" if r[key]["exact_support"]
                        else ("w^2" if r[key]["has_w2"] else "off"),
                        (r["amp"], r[key]["r2_clean"]),
                        textcoords="offset points", xytext=(0, 25 * dy_ * 10),
                        fontsize=8, ha="center")
    ax.set_xlabel("jitter amplitude (k units, temporally white)")
    ax.set_ylabel("R^2 of learned law on CLEAN system")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.set_title("A3: weak vs pointwise under frame jitter (riesz route)")
    fig.tight_layout()
    fig.savefig(out / "a3_jitter_comparison.png", dpi=160)
    plt.close(fig)
    log(f"  figure -> {out / 'a3_jitter_comparison.png'}")
    return rows


# --------------------------------------------------------------------------
# stage TUNE: rel_tol_pareto sweep on the best cells
# --------------------------------------------------------------------------

def forced_fit(system, names, nu):
    """Restricted OLS on a fixed support: fit on the x < median half, score
    holdout on the other half (same split convention as fit_target)."""
    if system is None:
        return None
    Theta, bv, xs = system[0], system[1], system[2]
    idx = [LIB.index(nm) for nm in names]
    tr = xs < np.median(xs)
    coef, *_ = np.linalg.lstsq(Theta[tr][:, idx], bv[tr], rcond=None)
    r2 = float(r_squared(bv[~tr], Theta[~tr][:, idx] @ coef))
    coefs = {nm: float(c) for nm, c in zip(names, coef)}
    el, ed = coef_errors(coefs, nu)
    return dict(coefs=coefs, r2_holdout=r2, lap_err=el, div_err=ed)


def stage_tune(theta, x, y, tt, nu, args, rng, out, log):
    log("=" * 72)
    log(f"TUNE  riesz route, radius={args.tune_radius}, "
        f"etas={list(args.tune_etas)}, rel_tol sweep "
        f"{list(args.tune_rel_tols)} (geometry: band_y={args.band_y} "
        f"radius_y={args.radius_y_lam} hole={args.band_hole})")
    k1, k2, w = route_stacks("riesz", theta, x, y, args.sigma_dir)
    res = {}
    for e in args.tune_etas:
        system = weak_system(k1, k2, w, x, y, tt, args, args.tune_radius, e,
                             band_y=args.band_y)
        ff = forced_fit(system, sorted(TRUE_SUPPORT), nu)
        if ff is not None:
            res[f"eta{e}|forced"] = ff
            log(f"  eta={e:<5} FORCED true support: "
                f"R2(hold)={ff['r2_holdout']:.4f} "
                f"Lap w={ff['coefs']['Lap w']:+.4g} "
                f"(err {100 * ff['lap_err']:.1f}%) "
                f"(div k)^2={ff['coefs']['(div k)^2']:+.4g} "
                f"(err {100 * ff['div_err']:.1f}%)")
        for rt in args.tune_rel_tols:
            r = fit_and_read(system, rng, rel_tol=rt)
            res[f"eta{e}|rt{rt}"] = r
            log(f"  eta={e:<5} rel_tol={rt:<5} " + fmt_fit(r, nu))
    return res


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s):
        print(s)
        lines.append(str(s))

    rng = np.random.default_rng(args.seed)
    log(f"evolve_knee: mu={args.mu} nu={args.nu} nframes={args.nframes} "
        f"nx={args.nx} kernel={args.kernel}")
    theta, x, y, tt, nu, core = evolve_knee(
        mu=args.mu, nu=args.nu, nframes=args.nframes, nx=args.nx,
        n_lam_x=args.n_lam_x)

    report = {"args": {k: getattr(args, k) for k in dir(args)
                       if not k.startswith("_")
                       and not callable(getattr(args, k))}}
    stages = args.stage.split(",") if isinstance(args.stage, str) \
        else list(args.stage)
    if "all" in stages:
        stages = ["a1", "a2", "a3"]
    if "a1" in stages:
        report["a1"] = stage_a1(theta, x, y, tt, nu, args, rng, out, log)
    if "a2" in stages:
        report["a2"] = stage_a2(theta, x, y, tt, nu, args, rng, out, log)
    if "a3" in stages:
        report["a3"] = stage_a3(theta, x, y, tt, nu, args, rng, out, log)
    if "tune" in stages:
        report["tune"] = stage_tune(theta, x, y, tt, nu, args, rng, out, log)

    (out / "report.json").write_text(json.dumps(report, indent=2,
                                                default=str))
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    log(f"report -> {out / 'report.json'}")


# --------------------------------------------------------------------------
# args
# --------------------------------------------------------------------------

class _Args:
    # synthetic field
    mu = 0.85
    nu = 0.05
    nframes = 60
    nx = 384
    n_lam_x = 8.0             # box width in wavelengths (radius + rim must fit)
    # extraction
    sigma_dir = 0.4
    # weak form
    kernel = "poly"           # 'poly' | 'gauss'
    poly_p = 6                # spatial bump exponent (>= 3)
    poly_pt = 4               # temporal bump exponent
    t_half = 4                # temporal half-width in FRAMES
    stride_xy = 4             # query-center stride (pixels)
    stride_t = 2              # query-center stride (frames)
    rim_lam = 0.5             # extra edge margin (lambda_x) for riesz rim
    radius_y_lam = 1.0        # outer y half-width (lambda_x), FIXED; the
                              # chevron oscillates along x, so only the x
                              # half-width is swept in A2
    band_y = 9.0              # |y| band for query centers (physical units)
    band_hole = 0.0           # keep core DATA out of every row: weak centers
                              # need |y| >= hole + radius_y (support-aware), so
                              # raise band_y / shrink radius_y_lam when using it
    seed = 0
    # stage A1
    a1_radius = 1.0           # outer radius (lambda_x)
    # stage A2
    a2_radii = (0.5, 1.0, 1.5, 2.0, 3.0)
    a2_etas = (0.0, 0.125, 0.25, 0.5)
    # stage A3
    a3_radius = 1.5
    a3_eta = 0.25
    a3_amps = (0.0, 0.02, 0.05, 0.1)
    jit_scale = 0.5           # jitter spatial scale (lambda_x)
    # measurement-consistent library
    sm_sigma = 0.0            # smooth nonlinear-in-derivative columns at this
                              # width AFTER the nonlinearity (physical units;
                              # set to the extraction's sigma_avg, i.e. pi, to
                              # match what riesz_k_field does to w; 0 = off)
    # selection
    rel_tol = 0.01            # Pareto tolerance for a1/a2/a3 fits
    # stage TUNE
    tune_radius = 1.0
    tune_etas = (0.0, 0.25)
    tune_rel_tols = (0.01, 0.02, 0.03, 0.05)
    # run control
    stage = "all"             # comma list of a1,a2,a3,tune or 'all'
    out = "results/wsindy_knee_validation"


def _cli():
    d = _Args
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--mu", type=float, default=d.mu)
    p.add_argument("--nu", type=float, default=d.nu)
    p.add_argument("--nframes", type=int, default=d.nframes)
    p.add_argument("--nx", type=int, default=d.nx)
    p.add_argument("--n_lam_x", type=float, default=d.n_lam_x)
    p.add_argument("--sigma_dir", type=float, default=d.sigma_dir)
    p.add_argument("--kernel", choices=("poly", "gauss"), default=d.kernel)
    p.add_argument("--poly_p", type=int, default=d.poly_p)
    p.add_argument("--poly_pt", type=int, default=d.poly_pt)
    p.add_argument("--t_half", type=int, default=d.t_half)
    p.add_argument("--stride_xy", type=int, default=d.stride_xy)
    p.add_argument("--stride_t", type=int, default=d.stride_t)
    p.add_argument("--rim_lam", type=float, default=d.rim_lam)
    p.add_argument("--radius_y_lam", type=float, default=d.radius_y_lam)
    p.add_argument("--band_y", type=float, default=d.band_y)
    p.add_argument("--band_hole", type=float, default=d.band_hole)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--a1_radius", type=float, default=d.a1_radius)
    p.add_argument("--a2_radii", type=float, nargs="+", default=d.a2_radii)
    p.add_argument("--a2_etas", type=float, nargs="+", default=d.a2_etas)
    p.add_argument("--a3_radius", type=float, default=d.a3_radius)
    p.add_argument("--a3_eta", type=float, default=d.a3_eta)
    p.add_argument("--a3_amps", type=float, nargs="+", default=d.a3_amps)
    p.add_argument("--jit_scale", type=float, default=d.jit_scale)
    p.add_argument("--sm_sigma", type=float, default=d.sm_sigma)
    p.add_argument("--rel_tol", type=float, default=d.rel_tol)
    p.add_argument("--tune_radius", type=float, default=d.tune_radius)
    p.add_argument("--tune_etas", type=float, nargs="+", default=d.tune_etas)
    p.add_argument("--tune_rel_tols", type=float, nargs="+",
                   default=d.tune_rel_tols)
    p.add_argument("--stage", default=d.stage)
    p.add_argument("--out", default=d.out)
    return p.parse_args()


if __name__ == "__main__":
    main(_cli() if len(sys.argv) > 1 else _Args)