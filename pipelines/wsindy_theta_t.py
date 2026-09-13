# pipelines/wsindy_theta_t.py
"""
Weak-form (WSINDy) identification of the phase rate theta_t, in two
forms, with the target and the fields taken from the seam-free
monogenic-gradient estimator (mg_estimator_validation, validated to a few
percent beyond half a spacing from a core; mg_pair_dynamics_validation
for theta_t).

Target: b = Int[ theta_t phi ], with theta_t the validated pointwise rate
(centred differences of z, Im(conj z z_t)/|z|^2) integrated against a
separable compactly supported test function phi(x, y, t). No time
integration by parts is needed and no measurement kernel enters: the
estimator's k is pointwise, so the library columns are built from the
same fields the target is built from.

RHS library (theta_t = sum c_j L_j):
    1, div k, w div k, k.grad w, Lap div k, w, Lap w, (div k)^2, w^2 div k
  Linear terms fully weak: Int[div k phi] = -Int[k . grad phi],
  Int[Lap div k phi] = -Int[k . grad(Lap phi)], Int[Lap w phi] =
  Int[w Lap phi]. Terms nonlinear in a derivative use an inner mollified
  derivative (eta) and a pointwise product before the outer integral.
  RCN truth (rcn_w_evolution_check, f(w) = w):
    tau theta_t = 2 alpha (w div k + k . grad w) - eta_b Lap div k,
  so c(w div k) = c(k.grad w) = 2 alpha/tau, c(Lap div k) = -eta_b/tau.

Variational library (theta_t = -(1/tau) delta E/delta theta, E = sum
c_i Int e_i):
    e in { w^2, w^3, (div k)^2, w (div k)^2, |grad w|^2 }
  Column i = weak form of -delta e_i/delta theta against phi:
    e = g(w):        -Int[ g'(w) 2 k . grad phi ]            (no data derivative)
    e = (div k)^2:   +2 Int[ k . grad(Lap phi) ]             (fully weak)
    e = w (div k)^2: -2 Int[ (div k)^2 k . grad phi ] - 2 Int[ w div k Lap phi ]
    e = |grad w|^2:  -4 Int[ (Lap w) k . grad phi ] + 2 Int[ ... ] (inner)
  RCN truth: E = (eta_b/2)(Lap theta)^2 + (alpha/2) w^2, so c(w^2) =
  alpha/(2 tau), c((div k)^2) = eta_b/(2 tau), all others zero.

Control (--control rcn): the 1d RCN knee relaxation of check (6),
    tau p_t = 2 alpha d_yy[ w p ] - eta_b p_yyyy,  w = k1^2 + p^2 - 1,
  p = k2 = Phi_y, from a planted sharp knee p0 = mu tanh(beta0 mu y),
  Dirichlet p = +-mu and p_yy = 0 at the ends, method of lines (BDF).
  theta = k1 x + Phi(y, t) on a 2d grid, u = cos theta, then the SAME
  estimator pipeline as on data; the exact route (exact k, exact theta_t
  from the PDE) is fitted alongside to separate method from estimator.

Data (--npz): frames from the file, window --t0/--t1, domain --domain
{all, far, band} by distance to the detected cores (mg_zigzag_theta_t
rules), query centres whose test-function support stays out of the
excluded region.

Selection: fit_target (STLSQ sweep) + every 2-term OLS candidate +
sparsest-within-rel_tol Pareto + contribution floor, spatial holdout on
x. Coefficients read by support identity and holdout R^2.

Outputs (relative): results/wsindy_theta_t/<stem>/<stamp>/report.txt,
  fits.json, fit_<form>_<route>.png

Additional targets (v1.1), same frames, same kernel, same domains:
  w_t  (w = |k|^2 - 1 pointwise from the estimator; target Int[w_t phi]
       with w_t by centred differences). Library = the RCN w terms of
       rcn_w_evolution_check check (2) plus the isotropic/anisotropic
       diffusion columns and an amplitude column:
         1, w, w^2, kk:ddw, w_nn, |grad w|^2, (k.grad w) div k, w Lap w,
         w |grad k|^2, (div k)^2, k.grad Lap div k, A_dev
       (Lap w is built but kept out of the free library, since
       Lap w = w_ss + w_nn; it enters the forced comparison
       {Lap w, w}, last month's full-domain law). RCN truth at
       alpha = eta = tau: kk:ddw 4, |grad w|^2 2, (k.grad w) div k 4,
       w Lap w 2, w |grad k|^2 -4, k.grad Lap div k -2, no w, no w_nn.
  A_t  (A = |z|; target Int[A_t phi]). Library 1, A, A^3, Lap A, w, A w,
       A^2, A w^2; forced comparisons {A, A^3} (far cubic) and
       {A, A^3, w, A w} (band law).
Modes: --mode control (the RCN knee movie; A is identically 1 there so
A_t is skipped) or --mode data --npz <file>.

Changelog
---------
v1.1  Writeup 1 constants alpha = eta = tau = 2R/3 (was the chapter's
      eta = 2 tau); radius default 1.5 spacings (the control fails at
      0.5: the crease ripple of the estimator's amplitude multiplies
      theta_t); --mode flag; w_t and A_t targets; empty-domain guard.
v1.0  first version.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import fftconvolve
from scipy.ndimage import binary_dilation

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT / "src", _ROOT / "src" / "op_extract", _ROOT / "src" / "utils",
          _HERE):
    sys.path.insert(0, str(p))

from mg_estimator_validation import mg_estimator, fdv                  # noqa: E402
from mg_zigzag_theta_t import load_npz, estimate, find_cores, dist_to_cores  # noqa: E402
from op_extract import t1_common as t1c                                # noqa: E402
from self_dual_sindy import fit_target, r_squared                      # noqa: E402
from wsindy_knee_validation import corr1d, inner_deriv, _FitArgs       # noqa: E402

import matplotlib                                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402

RHS_LIB = ["1", "div k", "w div k", "k.grad w", "Lap div k", "w", "Lap w",
           "(div k)^2", "w^2 div k"]
VAR_LIB = ["w^2", "w^3", "(div k)^2", "w (div k)^2", "|grad w|^2"]
W_LIB = ["1", "w", "w^2", "kk:ddw", "w_nn", "|grad w|^2", "(k.grad w) div k",
         "w Lap w", "w |grad k|^2", "(div k)^2", "k.grad Lap div k", "A_dev"]
W_TRUTH = {"kk:ddw": 4.0, "|grad w|^2": 2.0, "(k.grad w) div k": 4.0,
           "w Lap w": 2.0, "w |grad k|^2": -4.0, "k.grad Lap div k": -2.0}
A_LIB = ["1", "A", "A^3", "Lap A", "w", "A w", "A^2", "A w^2"]


# ---------------------------------------------------------------------
# kernel with derivatives up to third order (poly bump)
# ---------------------------------------------------------------------

def kernel_d3(half_width, spacing, p=6):
    """samples of the mass-normalised bump f(s) = (1 - (s/a)^2)^p and its
    derivatives f', f'', f''' on offsets -m..m"""
    m = max(3, int(round(half_width / spacing)))
    a = m * spacing
    s = np.arange(-m, m + 1) * spacing
    q = np.clip(1.0 - (s / a) ** 2, 0.0, None)
    qp = -2.0 * s / a ** 2
    qpp = -2.0 / a ** 2
    f = q ** p
    f1 = p * q ** (p - 1) * qp
    f2 = p * (p - 1) * q ** (p - 2) * qp ** 2 + p * q ** (p - 1) * qpp
    f3 = (p * (p - 1) * (p - 2) * q ** (p - 3) * qp ** 3
          + 3 * p * (p - 1) * q ** (p - 2) * qp * qpp)
    Z = f.sum() * spacing
    return f / Z, f1 / Z, f2 / Z, f3 / Z, m


def corr3(F, kx, ky, kt, dx, dy, dt):
    G = corr1d(F, ky, 0, dy)
    G = corr1d(G, kx, 1, dx)
    return corr1d(G, kt, 2, dt)


# ---------------------------------------------------------------------
# weak columns for both libraries
# ---------------------------------------------------------------------

def build_columns(k1, k2, th_t, x, y, tt, args, Lsp, A=None):
    """k1, k2, th_t: [ny, nx, nt]; A optional. Returns the RHS, VAR, W and
    A column dicts, the targets dict, kernel half sizes, and pointwise
    fields (for figures)."""
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    dt = float(np.median(np.diff(tt)))
    a = args.radius * Lsp
    fx, fx1, fx2, fx3, mx = kernel_d3(a, dx, args.poly_p)
    fy, fy1, fy2, fy3, my = kernel_d3(a, dy, args.poly_p)
    ht, ht1, ht2, ht3, mt = kernel_d3(args.t_half * dt, dt, args.poly_p)
    eta = args.eta * Lsp
    w = k1 ** 2 + k2 ** 2 - 1.0
    # inner derivatives for the nonlinear terms
    D = lambda F, ax: inner_deriv(F, eta, dx, dy, ax, "poly", args.poly_p)
    divk = D(k1, "x") + D(k2, "y")
    wx, wy = D(w, "x"), D(w, "y")
    lapw = D(wx, "x") + D(wy, "y")
    kgw = k1 * wx + k2 * wy

    K = lambda F, kxx, kyy, ktt: corr3(F, kxx, kyy, ktt, dx, dy, dt)
    C = {}
    # RHS library
    C["1"] = np.ones_like(w)
    C["div k"] = -(K(k1, fx1, fy, ht) + K(k2, fx, fy1, ht))
    C["w div k"] = K(w * divk, fx, fy, ht)
    C["k.grad w"] = K(kgw, fx, fy, ht)
    # Int Lap div k phi = -Int k . grad(Lap phi); grad(Lap phi) components:
    # d/dx(phi_xx + phi_yy) = f3 g h + f1 g2 h ; d/dy(...) = f2 g1 h + f g3 h
    C["Lap div k"] = -(K(k1, fx3, fy, ht) + K(k1, fx1, fy2, ht)
                       + K(k2, fx2, fy1, ht) + K(k2, fx, fy3, ht))
    C["w"] = K(w, fx, fy, ht)
    C["Lap w"] = K(w, fx2, fy, ht) + K(w, fx, fy2, ht)
    C["(div k)^2"] = K(divk ** 2, fx, fy, ht)
    C["w^2 div k"] = K(w ** 2 * divk, fx, fy, ht)
    # variational library: columns = weak -delta e/delta theta
    # e = g(w): -delta e/delta theta = div(2 g'(w) k) -> -Int 2 g' k . grad phi
    def var_g(gp):
        return -2.0 * (K(gp * k1, fx1, fy, ht) + K(gp * k2, fx, fy1, ht))
    V = {}
    V["w^2"] = var_g(2.0 * w)
    V["w^3"] = var_g(3.0 * w ** 2)
    # e = (div k)^2: -delta/delta theta = -2 Lap div k -> +2 Int k . grad(Lap phi)
    V["(div k)^2"] = -2.0 * C["Lap div k"]
    # e = w (div k)^2: -delta/delta theta = div(2 k (div k)^2) - 2 Lap(w div k)
    V["w (div k)^2"] = (-2.0 * (K(divk ** 2 * k1, fx1, fy, ht)
                                + K(divk ** 2 * k2, fx, fy1, ht))
                        - 2.0 * (K(w * divk, fx2, fy, ht) + K(w * divk, fx, fy2, ht)))
    # e = |grad w|^2: -delta/delta theta = div( 4 k * (-Lap w) )... derived:
    # e depends on grad w = 2 grad k . k; delta e/delta theta = -div(d e/d k)
    # + dd:(d e/d grad k) with d e/d grad k = 4 (grad w) (x) k, d e/d k =
    # 4 (grad k)^T grad w = 2 grad(|grad w|^2)/... use the weak identity
    # directly: -Int (delta e/delta theta) phi = Int d e/d k . grad phi
    #   - Int d e/d grad k : grad grad phi   (sign from two IBPs)
    # with pointwise inner fields; implemented as such
    gwx, gwy = wx, wy
    kx1, kx2 = D(k1, "x"), D(k1, "y")   # grad k rows (k1_x, k1_y)
    ky1, ky2 = D(k2, "x"), D(k2, "y")
    dedk1 = 4.0 * (kx1 * gwx + ky1 * gwy)      # (grad k)^T grad w, comp 1
    dedk2 = 4.0 * (kx2 * gwx + ky2 * gwy)
    # d e/d(grad k)_{ij} = 4 (grad w)_i k_j
    G11, G12 = 4.0 * gwx * k1, 4.0 * gwx * k2
    G21, G22 = 4.0 * gwy * k1, 4.0 * gwy * k2
    V["|grad w|^2"] = (K(dedk1, fx1, fy, ht) + K(dedk2, fx, fy1, ht)
                       - (K(G11, fx2, fy, ht) + K(G12, fx1, fy1, ht)
                          + K(G21, fx1, fy1, ht) + K(G22, fx, fy2, ht)))
    b = K(th_t, fx, fy, ht)
    targets = {"theta_t": b}

    # ---- w_t library (all nonlinear columns pointwise, then outer integral)
    kmag2 = np.maximum(k1 ** 2 + k2 ** 2, 1e-12)
    wxx, wxy, wyy = D(wx, "x"), D(wx, "y"), D(wy, "y")
    kkddw = k1 * k1 * wxx + 2 * k1 * k2 * wxy + k2 * k2 * wyy      # unnormalised
    w_ss_unit = kkddw / kmag2
    w_nn = lapw - w_ss_unit
    gradk2 = kx1 ** 2 + kx2 ** 2 + ky1 ** 2 + ky2 ** 2
    lapdivk = D(D(divk, "x"), "x") + D(D(divk, "y"), "y")
    kglapdiv = k1 * D(lapdivk, "x") + k2 * D(lapdivk, "y")
    Wc = {}
    Wc["1"] = np.ones_like(w)
    Wc["w"] = K(w, fx, fy, ht)
    Wc["w^2"] = K(w ** 2, fx, fy, ht)
    Wc["kk:ddw"] = K(kkddw, fx, fy, ht)
    Wc["w_nn"] = K(w_nn, fx, fy, ht)
    Wc["|grad w|^2"] = K(wx ** 2 + wy ** 2, fx, fy, ht)
    Wc["(k.grad w) div k"] = K(kgw * divk, fx, fy, ht)
    Wc["w Lap w"] = K(w * lapw, fx, fy, ht)
    Wc["w |grad k|^2"] = K(w * gradk2, fx, fy, ht)
    Wc["(div k)^2"] = C["(div k)^2"]
    Wc["k.grad Lap div k"] = K(kglapdiv, fx, fy, ht)
    Wc["Lap w"] = C["Lap w"]                                       # forced use only
    if A is not None:
        Amed = np.nanmedian(A)
        Wc["A_dev"] = K(A - Amed, fx, fy, ht)
    else:
        Wc["A_dev"] = np.zeros_like(w)
    w_t = np.gradient(w, dt, axis=2)
    targets["w_t"] = K(w_t, fx, fy, ht)

    # ---- A_t library
    if A is not None:
        Ac = {}
        Ac["1"] = np.ones_like(w)
        Ac["A"] = K(A, fx, fy, ht)
        Ac["A^3"] = K(A ** 3, fx, fy, ht)
        Ac["Lap A"] = K(A, fx2, fy, ht) + K(A, fx, fy2, ht)
        Ac["w"] = C["w"]
        Ac["A w"] = K(A * w, fx, fy, ht)
        Ac["A^2"] = K(A ** 2, fx, fy, ht)
        Ac["A w^2"] = K(A * w ** 2, fx, fy, ht)
        A_t = np.gradient(A, dt, axis=2)
        targets["A_t"] = K(A_t, fx, fy, ht)
    else:
        Ac = None
    del wxx, wxy, wyy, kkddw, w_ss_unit, w_nn, gradk2, lapdivk, kglapdiv, kgw, lapw, wx, wy
    return dict(rhs=C, var=V, w=Wc, A=Ac), targets, (mx, my, mt), dict(w=w, divk=divk)


def theta_t_from_z(zs, dt):
    """Im(conj z z_t)/|z|^2 with centred differences inside and one-sided
    differences at the two ends (an FFT correlation cannot tolerate NaN)"""
    z_t = np.empty_like(zs)
    z_t[..., 1:-1] = (zs[..., 2:] - zs[..., :-2]) / (2 * dt)
    z_t[..., 0] = (zs[..., 1] - zs[..., 0]) / dt
    z_t[..., -1] = (zs[..., -1] - zs[..., -2]) / dt
    return np.imag(np.conj(zs) * z_t) / np.maximum(np.abs(zs) ** 2, 1e-14)


def query_centres(shape, m, args, excl=None, x=None):
    """decimated grid of query centres with margins >= kernel half widths;
    excl: boolean [ny, nx, nt] of points whose test-function support may
    not touch (dilated by the half widths)"""
    ny, nx, nt = shape
    mx, my, mt = m
    ix = np.arange(mx + args.rim, nx - mx - args.rim, args.stride_xy)
    iy = np.arange(my + args.rim, ny - my - args.rim, args.stride_xy)
    it = np.arange(mt, nt - mt, args.stride_t)
    IY, IX, IT = np.meshgrid(iy, ix, it, indexing="ij")
    keep = np.ones(IY.shape, bool)
    if excl is not None:
        if args.excl_dilate:
            big = binary_dilation(excl, structure=np.ones((2 * my + 1, 2 * mx + 1, 1), bool))
            keep &= ~big[IY, IX, IT]
        else:
            keep &= ~excl[IY, IX, IT]
    if keep.sum() == 0:
        return None
    return IY[keep], IX[keep], IT[keep]


def assemble(cols, names, b, idx, x):
    IY, IX, IT = idx
    Theta = np.column_stack([cols[nm][IY, IX, IT] for nm in names])
    bv = b[IY, IX, IT]
    xs = x[IX]
    fin = np.all(np.isfinite(Theta), axis=1) & np.isfinite(bv)
    return Theta[fin], bv[fin], xs[fin]


# ---------------------------------------------------------------------
# selection (fit_target -> pairs -> Pareto -> contribution floor)
# ---------------------------------------------------------------------

def _refit(Theta, bv, supp):
    coef = np.zeros(Theta.shape[1])
    idx = np.where(supp)[0]
    if len(idx):
        coef[idx], *_ = np.linalg.lstsq(Theta[:, idx], bv, rcond=None)
    return coef


def fit_pareto(names, Theta, bv, xs, rng, rel_tol, min_contrib, max_samples=60000):
    if len(bv) < 200:
        return dict(skipped=True, reason="too few rows (%d)" % len(bv))
    if len(bv) > max_samples:
        sel = rng.choice(len(bv), max_samples, replace=False)
        Theta, bv, xs = Theta[sel], bv[sel], xs[sel]
    tr = xs < np.median(xs)
    res = fit_target(names, Theta, bv, _FitArgs(rel_tol), rng, tr)
    sweep = [s for s in res["sweep"] if np.isfinite(s["r2_test"])]
    p = Theta.shape[1]
    for i in range(p):
        for j in range(i + 1, p):
            supp = np.zeros(p, bool); supp[[i, j]] = True
            coef = np.zeros(p)
            coef[[i, j]], *_ = np.linalg.lstsq(Theta[tr][:, [i, j]], bv[tr], rcond=None)
            sweep.append(dict(support=supp, coef=coef, n_terms=2,
                              r2_train=r_squared(bv[tr], Theta[tr] @ coef),
                              r2_test=r_squared(bv[~tr], Theta[~tr] @ coef)))
    best = max(s["r2_test"] for s in sweep)
    adm = [s for s in sweep if s["n_terms"] > 0 and s["r2_test"] >= best - rel_tol * abs(best)]
    adm.sort(key=lambda s: (s["n_terms"], -s["r2_test"]))
    chosen = adm[0]
    supp = chosen["support"].copy()
    coef = _refit(Theta, bv, supp)
    if min_contrib > 0:
        sb = max(float(np.std(bv)), 1e-30)
        for _ in range(p):
            contrib = np.abs(coef) * np.std(Theta, axis=0) / sb
            weak = supp & (contrib < min_contrib)
            if not weak.any() or supp.sum() <= 1:
                break
            j = np.where(weak)[0][np.argmin(contrib[weak])]
            supp[j] = False
            coef = _refit(Theta, bv, supp)
    coefs = {names[i]: float(coef[i]) for i in np.where(supp)[0]}
    # holdout of the final support
    cf = np.zeros(p); cf[supp], *_ = np.linalg.lstsq(Theta[tr][:, supp], bv[tr], rcond=None)
    return dict(n=int(len(bv)), coefs=coefs, support=sorted(coefs),
                r2_full=float(r_squared(bv, Theta @ coef)),
                r2_holdout=float(r_squared(bv[~tr], Theta[~tr] @ cf)),
                best_r2=float(best),
                dense_r2=float(r_squared(bv[~tr], Theta[~tr] @ _refit(Theta[tr], bv[tr], np.ones(p, bool)))))


def forced_fit(names, Theta, bv, xs, want):
    tr = xs < np.median(xs)
    idx = [names.index(n) for n in want]
    c, *_ = np.linalg.lstsq(Theta[tr][:, idx], bv[tr], rcond=None)
    cf = np.zeros(Theta.shape[1]); cf[idx] = c
    return dict(coefs={n: float(v) for n, v in zip(want, c)},
                r2_holdout=float(r_squared(bv[~tr], Theta[~tr] @ cf)))


def fmt(r, truth=None):
    if r.get("skipped"):
        return "SKIPPED (%s)" % r.get("reason")
    s = " + ".join("%+.4g %s" % (v, k) for k, v in r["coefs"].items())
    s += "   holdout R2 %.4f (best any %.4f, dense %.4f, n %d)" % (
        r["r2_holdout"], r["best_r2"], r["dense_r2"], r["n"])
    if truth:
        errs = []
        for k, v in truth.items():
            if k in r["coefs"]:
                errs.append("%s %.1f%%" % (k, 100 * abs(r["coefs"][k] - v) / abs(v)))
            else:
                errs.append("%s MISSING" % k)
        extra = [k for k in r["coefs"] if k not in truth]
        s += "   vs truth: " + ", ".join(errs) + (("; extras " + str(extra)) if extra else "")
    return s


# ---------------------------------------------------------------------
# the RCN control
# ---------------------------------------------------------------------

def rcn_knee_relaxation(mu, alpha, eta_b, tau, R, beta0, Ly, ny, t_end, nt,
                        keep_quartic=False):
    """integrate tau p_t = 2 alpha d_yy[f(w) p] - eta_b p_yyyy, p = Phi_y,
    w = k1^2 + p^2 - 1, f = w (- w^3/R if keep_quartic), p(+-L) = +-mu,
    p_yy(+-L) = 0; returns y, tt, p[ny, nt], Phi[ny, nt], p_t[ny, nt]"""
    k1 = np.sqrt(1 - mu ** 2)
    y = np.linspace(-Ly / 2, Ly / 2, ny)
    dy = y[1] - y[0]
    p0 = mu * np.tanh(beta0 * mu * y)

    def d2(v):
        out = np.zeros_like(v)
        out[1:-1] = (v[2:] - 2 * v[1:-1] + v[:-2]) / dy ** 2
        return out

    def rhs(t, p):
        pp = p.copy(); pp[0], pp[-1] = -mu, mu
        w = k1 ** 2 + pp ** 2 - 1.0
        f = w - (w ** 3 / R if keep_quartic else 0.0)
        term = 2 * alpha * d2(f * pp)
        p2 = d2(pp); p2[0] = p2[-1] = 0.0            # p_yy = 0 at the ends
        p4 = d2(p2)
        dp = (term - eta_b * p4) / tau
        dp[0] = dp[-1] = 0.0
        return dp

    tt = np.linspace(0, t_end, nt)
    sol = solve_ivp(rhs, (0, t_end), p0, t_eval=tt, method="BDF", rtol=1e-7, atol=1e-9)
    p = sol.y
    pt = np.column_stack([rhs(t, p[:, i]) for i, t in enumerate(tt)])
    # gauge: theta_t vanishes in the far field (plane wave), so Phi is
    # pinned at the far end (constant in time), not at the core
    Phi = np.cumsum(0.5 * (p[1:] + p[:-1]) * dy, axis=0)
    Phi = np.vstack([np.zeros((1, nt)), Phi])
    return y, tt, p, Phi, pt


def run_control(args, out, rng, lines):
    mu = args.mu
    k1 = float(np.sqrt(1 - mu ** 2))
    Lsp = np.pi / k1
    alpha, tau, eta_b, R = args.alpha, args.tau, args.eta_b, args.R
    ny_1d = args.ny_control
    y, tt, p, Phi, pt = rcn_knee_relaxation(mu, alpha, eta_b, tau, R, args.beta0,
                                            args.Ly, ny_1d, args.t_end, args.nt)
    # 2d grid: x periodic with an integer number of stripe periods
    nper = args.n_xper
    Lx = nper * 2 * np.pi / k1
    dx = y[1] - y[0]
    nx = int(round(Lx / dx)); Lx = nx * dx
    x = (np.arange(nx) - nx // 2) * dx
    nt = len(tt)
    th = k1 * x[None, :, None] + Phi[:, None, :]                  # [ny, nx, nt]
    # exact theta_t is Phi_t: the cumulative integral of p_t in the same gauge
    Phit = np.cumsum(0.5 * (pt[1:] + pt[:-1]) * dx, axis=0)
    Phit = np.vstack([np.zeros((1, nt)), Phit])          # same gauge as Phi
    th_t_ex = np.broadcast_to(Phit[:, None, :], th.shape).copy()
    k1_ex = np.full(th.shape, k1); k2_ex = np.broadcast_to(p[:, None, :], th.shape).copy()
    lines.append("RCN control: mu %.2f, alpha %.3f, tau %.3f, eta_b %.3f, beta0 %.2f,"
                 " Ly %.1f, t in [0, %.2f] (%d frames), grid %dx%d" %
                 (mu, alpha, tau, eta_b, args.beta0, args.Ly, args.t_end, nt, ny_1d, nx))
    beta_inf = float(np.sqrt(alpha / eta_b))
    lines.append("  expected relaxed self-dual coefficient beta = sqrt(alpha/eta_b) = %.3f;"
                 " initial knee width ratio p_y(0)/mu^2: %.3f -> %.3f at t_end" %
                 (beta_inf, args.beta0, float(np.gradient(p[:, -1], y)[ny_1d // 2] / mu ** 2)))
    truth_rhs = {"w div k": 2 * alpha / tau, "k.grad w": 2 * alpha / tau, "Lap div k": -eta_b / tau}
    truth_var = {"w^2": alpha / (2 * tau), "(div k)^2": eta_b / (2 * tau)}
    truth_w = {"kk:ddw": 4 * alpha / tau, "|grad w|^2": 2 * alpha / tau,
               "(k.grad w) div k": 4 * alpha / tau, "w Lap w": 2 * alpha / tau,
               "w |grad k|^2": -4 * alpha / tau, "k.grad Lap div k": -2 * eta_b / tau}

    routes = {"exact": (k1_ex, k2_ex, th_t_ex)}
    # estimator route: u = cos theta, per-frame estimator, centred theta_t
    u = np.cos(th)
    taper = t1c.edge_taper(x, y, frac=args.taper_frac)
    Lyy = ny_1d * dx
    g = dict(Lx=Lx, Ly=Lyy)
    ests, n_prev = [], None
    for i in range(nt):
        e = estimate(u[..., i], Lx, Lyy, args.sigma_dir, n_prev, taper)
        n_prev = e["n"]; ests.append(e)
    dt = tt[1] - tt[0]
    k1m = np.stack([e["k1"] for e in ests], -1); k2m = np.stack([e["k2"] for e in ests], -1)
    zs = np.stack([e["z"] for e in ests], -1)
    th_t_m = theta_t_from_z(zs, dt)
    s = np.sign(k1m); s[s == 0] = 1                                   # align with exact (k1 > 0)
    routes["estimator"] = (k1m * s, k2m * s, th_t_m * s)
    mask_tt = np.abs(th_t_m - th_t_ex)[..., 2:-2]
    lines.append("  estimator theta_t vs exact, rel RMS over the box (frames 2..nt-3): %.4f" %
                 (np.sqrt(np.nanmean(mask_tt ** 2)) / np.sqrt(np.mean(th_t_ex[..., 2:-2] ** 2))))
    fits = {}
    for route, (K1, K2, TT) in routes.items():
        cols, targets, m, pw = build_columns(K1, K2, TT, x, y, tt, args, Lsp)
        C, V, b = cols["rhs"], cols["var"], targets["theta_t"]
        idx = query_centres(K1.shape, m, args)
        Th, bv, xs = assemble(C, RHS_LIB, b, idx, x)
        r = fit_pareto(RHS_LIB, Th, bv, xs, rng, args.rel_tol, args.min_contrib)
        f = forced_fit(RHS_LIB, Th, bv, xs, list(truth_rhs))
        lines.append("  [%s] RHS  free:   %s" % (route, fmt(r, truth_rhs)))
        lines.append("  [%s] RHS  forced: %s   holdout R2 %.4f" % (route, " + ".join("%+.4g %s" % (v, k) for k, v in f["coefs"].items()), f["r2_holdout"]))
        Tv, bv2, xs2 = assemble(V, VAR_LIB, b, idx, x)
        rv = fit_pareto(VAR_LIB, Tv, bv2, xs2, rng, args.rel_tol, args.min_contrib)
        fv = forced_fit(VAR_LIB, Tv, bv2, xs2, list(truth_var))
        lines.append("  [%s] VAR  free:   %s" % (route, fmt(rv, truth_var)))
        lines.append("  [%s] VAR  forced: %s   holdout R2 %.4f" % (route, " + ".join("%+.4g %s" % (v, k) for k, v in fv["coefs"].items()), fv["r2_holdout"]))
        fits[route] = dict(rhs_free=r, rhs_forced=f, var_free=rv, var_forced=fv)
        fig_fit(Th, bv, xs, RHS_LIB, list(truth_rhs), f, out / ("fit_rhs_%s.png" % route), route)
        # w_t on the same frames
        Tw, bw, xw = assemble(cols["w"], W_LIB, targets["w_t"], idx, x)
        rw = fit_pareto(W_LIB, Tw, bw, xw, rng, args.rel_tol, args.min_contrib)
        fw = forced_fit(W_LIB, Tw, bw, xw, list(truth_w))
        fl = forced_fit(W_LIB + ["Lap w"], np.column_stack([Tw, cols["w"]["Lap w"][idx]]), bw, xw, ["Lap w", "w"])
        lines.append("  [%s] W_T  free:   %s" % (route, fmt(rw, truth_w)))
        lines.append("  [%s] W_T  forced RCN w terms: %s   holdout R2 %.4f" % (route, " + ".join("%+.4g %s" % (v, k) for k, v in fw["coefs"].items()), fw["r2_holdout"]))
        lines.append("  [%s] W_T  forced {Lap w, w}:  %s   holdout R2 %.4f" % (route, " + ".join("%+.4g %s" % (v, k) for k, v in fl["coefs"].items()), fl["r2_holdout"]))
        fits[route].update(wt_free=rw, wt_forced_rcn=fw, wt_forced_lapw=fl)
    lines.append("  (A is identically 1 on the control; A_t not fitted)")
    lines.append("")
    return fits


def fig_fit(Theta, bv, xs, names, want, f, path, title):
    idx = [names.index(n) for n in want]
    pred = Theta[:, idx] @ np.array([f["coefs"][n] for n in want])
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot(bv, pred, ".", ms=2, alpha=0.5)
    lim = np.percentile(np.abs(bv), 99.5)
    ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("weak target Int[theta_t phi]"); ax.set_ylabel("forced model")
    ax.set_title("%s, holdout R2 %.3f" % (title, f["r2_holdout"]), fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------
# data
# ---------------------------------------------------------------------

def run_data(args, out, rng, lines):
    D = load_npz(args.npz)
    u, x, y, t, md = D["u"], D["x"], D["y"], D["t"], D["md"]
    mu = float(md.get("mu", args.mu)); k1 = np.sqrt(1 - mu ** 2); Lsp = np.pi / k1
    sel = (t >= args.t0) & (t <= args.t1)
    u, t = u[..., sel], t[sel]
    ny, nx, nt = u.shape
    dx = x[1] - x[0]; dt = float(np.median(np.diff(t)))
    Lx, Ly = nx * dx, ny * dx
    taper = t1c.edge_taper(x, y, frac=args.taper_frac)
    X, Y = np.meshgrid(x, y)
    win = (np.abs(X - x.mean()) < 0.4 * Lx) & (np.abs(Y - y.mean()) < 0.4 * Ly)
    ests, n_prev = [], None
    for i in range(nt):
        e = estimate(u[..., i], Lx, Ly, args.sigma_dir, n_prev, taper)
        n_prev = e["n"]; ests.append(e)
    k1m = np.stack([e["k1"] for e in ests], -1); k2m = np.stack([e["k2"] for e in ests], -1)
    zs = np.stack([e["z"] for e in ests], -1)
    th_t = theta_t_from_z(zs, dt)
    # orient k with k1 > 0 (zigzag convention) and theta_t consistently
    s = np.sign(k1m); s[s == 0] = 1
    k1m, k2m, th_t = k1m * s, k2m * s, th_t * s
    # exclusion by distance to detected cores
    dist = np.empty(u.shape)
    for i in range(nt):
        xc, yc = find_cores(ests[i]["A"], x, y, win, args.core_frac)
        dist[..., i] = dist_to_cores(X, Y, xc, yc) / Lsp
    if args.domain == "far":
        excl = dist < args.far_from
    elif args.domain == "band":
        excl = (dist < args.core_excl) | (dist > args.band_to)
    else:
        excl = dist < args.core_excl
    lines.append("data: %s  mu %.2f  t in [%.3f, %.3f] (%d frames, dt %.3f)  domain %s"
                 % (Path(args.npz).stem, mu, t[0], t[-1], nt, dt, args.domain))
    Am = np.stack([e["A"] for e in ests], -1)
    cols, targets, m, pw = build_columns(k1m, k2m, th_t, x, y, t, args, Lsp, A=Am)
    C, V, b = cols["rhs"], cols["var"], targets["theta_t"]
    idx = query_centres(u.shape, m, args, excl=excl, x=x)
    if idx is None:
        lines.append("  no query centres survive the domain exclusion; nothing fitted")
        lines.append("")
        return {}
    Th, bv, xs = assemble(C, RHS_LIB, b, idx, x)
    r = fit_pareto(RHS_LIB, Th, bv, xs, rng, args.rel_tol, args.min_contrib)
    f = forced_fit(RHS_LIB, Th, bv, xs, ["w div k", "k.grad w", "Lap div k"])
    lines.append("  RHS  free:   %s" % fmt(r))
    lines.append("  RHS  forced RCN support: %s   holdout R2 %.4f" % (" + ".join("%+.4g %s" % (v, k) for k, v in f["coefs"].items()), f["r2_holdout"]))
    Tv, bv2, xs2 = assemble(V, VAR_LIB, b, idx, x)
    rv = fit_pareto(VAR_LIB, Tv, bv2, xs2, rng, args.rel_tol, args.min_contrib)
    fv = forced_fit(VAR_LIB, Tv, bv2, xs2, ["w^2", "(div k)^2"])
    lines.append("  VAR  free:   %s" % fmt(rv))
    lines.append("  VAR  forced RCN energy: %s   holdout R2 %.4f" % (" + ".join("%+.4g %s" % (v, k) for k, v in fv["coefs"].items()), fv["r2_holdout"]))
    lines.append("  target scale: RMS Int[theta_t phi] = %.3e over %d rows" % (float(np.sqrt(np.mean(bv ** 2))), len(bv)))
    fig_fit(Th, bv, xs, RHS_LIB, ["w div k", "k.grad w", "Lap div k"], f, out / "fit_rhs_data.png", "data, RCN support")
    fits = dict(rhs_free=r, rhs_forced=f, var_free=rv, var_forced=fv)
    # ---- w_t
    Tw, bw, xw = assemble(cols["w"], W_LIB, targets["w_t"], idx, x)
    rw = fit_pareto(W_LIB, Tw, bw, xw, rng, args.rel_tol, args.min_contrib)
    fw = forced_fit(W_LIB, Tw, bw, xw, list(W_TRUTH))
    ThL = np.column_stack([Tw, cols["w"]["Lap w"][idx]])
    fl = forced_fit(W_LIB + ["Lap w"], ThL, bw, xw, ["Lap w", "w"])
    fa = forced_fit(W_LIB, Tw, bw, xw, ["kk:ddw", "w_nn", "w"])
    lines.append("  W_T  free:   %s" % fmt(rw))
    lines.append("  W_T  forced RCN w terms: %s   holdout R2 %.4f" % (" + ".join("%+.4g %s" % (v, k) for k, v in fw["coefs"].items()), fw["r2_holdout"]))
    lines.append("  W_T  forced {Lap w, w} (archival law 0.62 Lap w - 0.17 w): %s   holdout R2 %.4f" % (" + ".join("%+.4g %s" % (v, k) for k, v in fl["coefs"].items()), fl["r2_holdout"]))
    lines.append("  W_T  forced anisotropic {kk:ddw, w_nn, w}: %s   holdout R2 %.4f" % (" + ".join("%+.4g %s" % (v, k) for k, v in fa["coefs"].items()), fa["r2_holdout"]))
    lines.append("  target scale: RMS Int[w_t phi] = %.3e" % float(np.sqrt(np.mean(bw ** 2))))
    fits.update(wt_free=rw, wt_forced_rcn=fw, wt_forced_lapw=fl, wt_forced_aniso=fa)
    # ---- A_t
    Ta, ba, xa = assemble(cols["A"], A_LIB, targets["A_t"], idx, x)
    ra = fit_pareto(A_LIB, Ta, ba, xa, rng, args.rel_tol, args.min_contrib)
    fc = forced_fit(A_LIB, Ta, ba, xa, ["A", "A^3"])
    fb = forced_fit(A_LIB, Ta, ba, xa, ["A", "A^3", "w", "A w"])
    lines.append("  A_T  free:   %s" % fmt(ra))
    lines.append("  A_T  forced far cubic {A, A^3} (archival 0.494 A - 0.742 A^3): %s   holdout R2 %.4f" % (" + ".join("%+.4g %s" % (v, k) for k, v in fc["coefs"].items()), fc["r2_holdout"]))
    lines.append("  A_T  forced band law {A, A^3, w, A w}: %s   holdout R2 %.4f" % (" + ".join("%+.4g %s" % (v, k) for k, v in fb["coefs"].items()), fb["r2_holdout"]))
    lines.append("  target scale: RMS Int[A_t phi] = %.3e ; A median %.3f" % (float(np.sqrt(np.mean(ba ** 2))), float(np.nanmedian(Am))))
    fits.update(at_free=ra, at_forced_cubic=fc, at_forced_band=fb)
    lines.append("")
    return fits


# ---------------------------------------------------------------------

def main(args):
    rng = np.random.default_rng(args.seed)
    stem = "control_rcn" if args.mode == "control" else Path(args.npz).stem
    stamp = "rad%.2g_eta%.2g_th%d_mc%.2g_%s_t%.2f-%.2f" % (
        args.radius, args.eta, args.t_half, args.min_contrib, args.domain, args.t0, args.t1)
    out = Path(args.outdir) / stem / stamp
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as fh:
        json.dump({k: getattr(args, k) for k in vars(args)}, fh, indent=2, default=str)
    lines = ["wsindy_theta_t v1.1", "libraries: RHS %s ; VAR %s ; W %s ; A %s" % (RHS_LIB, VAR_LIB, W_LIB, A_LIB),
             "kernel radius %.2g spacings (x and y), t_half %d frames, inner eta %.2g spacings,"
             " rel_tol %.2g, min_contrib %.2g" % (args.radius, args.t_half, args.eta,
                                                  args.rel_tol, args.min_contrib), ""]
    t0 = time.time()
    if args.mode == "control":
        fits = run_control(args, out, rng, lines)
    else:
        fits = run_data(args, out, rng, lines)
    lines.append("elapsed %.1f s" % (time.time() - t0))
    rep = "\n".join(lines)
    (out / "report.txt").write_text(rep)
    with open(out / "fits.json", "w") as fh:
        json.dump(fits, fh, indent=1, default=str)
    print(rep); print("outputs ->", out)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    mode = "control"             # control: the RCN knee movie; data: --npz
    # npz = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweeps_full_Nx512_hp025_T3p125_NyF5_np18_Nsave125/raw/sh_pgb_zigzag_mu0.850_T3_N125_nx512_Ny256_lower.npz"
    npz = ""
    outdir = str(Path("results") / "wsindy_theta_t")
    # control parameters
    mu = 0.85
    alpha = 1.0 / 3.0            # Writeup 1: alpha = eta = tau = 2R/3 at R = 1/2
    tau = 1.0 / 3.0
    eta_b = 1.0 / 3.0            # (the chapter's eta = 2 tau is the open factor-two todo)
    R = 0.5
    beta0 = 1.8                  # planted sharp knee, relaxes toward sqrt(alpha/eta_b)
    Ly = 60.0
    t_end = 2.0
    nt = 41
    n_xper = 4
    ny_control = 300             # 1d grid points of the control (memory: ~30 column stacks)
    # estimator
    sigma_dir = 0.4
    taper_frac = 0.06
    core_frac = 0.70
    # weak form
    radius = 1.5                 # test-function half width, spacings (control: 0.5 fails, 1.5-2 identifies)
    t_half = 3                   # frames
    eta = 0.0                    # inner mollifier width, spacings (0 = np.gradient)
    poly_p = 6
    stride_xy = 4
    stride_t = 1
    rim = 4
    rel_tol = 0.01
    min_contrib = 0.05
    seed = 0
    # data windows / domains
    t0 = -1.0
    t1 = 1e9
    domain = "all"               # all | far | band
    core_excl = 0.5              # spacings; 0 disables (pre-winding: phase is smooth everywhere)
    far_from = 1.5
    band_to = 2.0
    excl_dilate = 0              # 1: exclude centres whose kernel touches the region; 0: centres only


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in dir(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v) if not isinstance(v, bool) else str, default=v)
    cli = p.parse_args()
    if cli.npz:
        cli.mode = "data"
    sys.exit(main(cli))
