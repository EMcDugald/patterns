# pipelines/uhu_sstar_iterate.py
"""
Fixed-point iteration of the optimal uHu window S* = c H^{-2}.

The optimal window of Chapter 4 (Section 4.6 and Appendix E of the draft)
needs the local wavevector k and the phase Hessian H, which are what the
window is meant to measure. This script asks whether the measurement can
supply them to itself:

    step 0   k^0 from the isotropic window S^0 = sigma0^2 I
    step n   H^n   = sym grad k^{n-1}   (sign-safe differences, k is a director)
             S*^n  = the two rules at (k^{n-1}, H^n), clipped to [s_min, s_cap]
             S^n   = (1 - omega) S^{n-1} + omega S*^n          (damped)
             k^n   = dominant eigenvector of <u Hess u>_{S^n}, |k|^2 = -lambda_min / <u^2>_{S^n}
    stop     when the median change of the k angle between iterations falls
             below --conv_deg (after at least two steps) or at --n_iter.

The windows are spatially varying anisotropic Gaussians, applied by
gathering over offsets: <F>(x) = sum_d w(x, d) F(x + d) / sum_d w(x, d),
w = exp(-d^T S(x)^{-1} d / 2), so each point is averaged with its own
window and the normalisation absorbs the truncation at the box edges.

S* in a form that is regular where H is singular. In the eigenframe of H
(eigenvalues hA, hB, eigenvectors eA, eB) with kA = k.eA, kB = k.eB,

    s_A = L0 hB^2 / (kA^2 hB^2 + kB^2 hA^2),
    s_B = L0 hA^2 / (kA^2 hB^2 + kB^2 hA^2),        L0 = ln(1/tol) / 2,

which is c H^{-2} with c = ln(1/tol) / (2 |H^{-1} k|^2) multiplied through
by hA^2 hB^2. It satisfies the scale rule k^T S k = L0 exactly and the
direction rule, and on the knee (hA = 0) it returns the needle
s_xx = L0 / k1^2, s_yy = 0 without a special case. It is homogeneous of
degree zero in H, so only the ratio of the curvatures sets the shape and
the scale rule sets the size; where both curvatures vanish (den = 0) the
window is set to the cap, the degenerate plane-wave case of the draft.

Masks, all written to fields.npz and drawn:
  mask_W      W = ||H||_F / |k|^2 >= W_max (the quadratic model fails;
              Chapter 4: W ~ 1 at a core; the S* demo excluded W >= 0.5)
  mask_osc    the k angle still changes by more than osc_deg at the last
              step (no fixed point there)
  mask_floor  both principal widths at the floor (nothing can be averaged)
  core mask   mask_W | mask_osc

Modes
  knee   the exact self-dual knee u = cos(k1 x + log(2 cosh(beta mu y))/beta)
         (truth k and H in closed form)
  pn     the single paraxial Pismen-Nepomnyashchy dislocation of the S*
         figures, theta = y - log(a + b erf(x / sqrt(2y))), a = (1 + e^-pi)/2,
         b = (e^-pi - 1)/2, extended oddly in y (u even), |y| clipped at y_clip;
         truth k and H from the p_n formulas (eq:pn_pcoords_app)
  data   one frame of an SH npz (u[ny, nx, nt], x, y, tt or t,
         metadata_json with mu and gb_row_in_crop), cropped to +-crop_L
         spacings about the row; no truth, so the report gives the
         convergence and the masks only
In the two synthetic modes the report also gives the oracle: one
average with S* evaluated at the TRUE k and H, which is what the
iteration would reach if it could see the truth (the Gauss-Hermite
version of this oracle is sstar_extraction_demo.py).

Outputs: <outdir>/<stem>/<stamp>/report.txt, fields.npz,
  convergence.png, maps.png, config.json

Changelog
---------
v1.0  first version (5 Oct 2026). Self-contained (numpy, scipy,
      matplotlib) except --init mg, which imports mg_estimator from
      mg_estimator_validation; data mode uses mg_zigzag_theta_t.load_npz
      when it is importable and an equivalent local loader otherwise.
      Measured on the synthetic modes (grid dx 0.25, tol 0.05, sigma_min
      0.15; median angle error to the exact k, degrees; the mg start was
      computed with stand-ins for the frozen Riesz layer, so its step-0
      row will differ on the real estimator):
        oracle (S* at the true k, H; one average) reproduces Chapter 4's
          Gauss-Hermite numbers on the knee at mu 0.9: core 57.9 -> 1.29
          (Ch. 4: 59.4 -> 1.8), flank 1.22 -> 0.034 (Ch. 4: 1.20 ->
          0.034); single PN, W < 0.5: 1.80 -> 0.35 (Ch. 4: 1.27 -> 0.20).
        two fixed points: the truth is one (--init truth, omega 1:
          converged in 2 steps at the oracle, core mask = true W >= 0.5
          exactly); the flipped knee is another (the isotropic start at
          0.9 without H smoothing stays at 58 deg in the core).
        damping toward the isotropic window flips even the truth in one
          step (core 0 -> 49.8 with omega 0.5), hence the first step is
          never damped toward iso and omega defaults to 1.
        H smoothing is what makes the correct fixed point reachable:
          knee 0.9 from mg, h_smooth 1.0: core 3.28 -> 1.30, flank 1.73
          -> 0.0345 in 4 steps (oracle 1.29, 0.0336); h_smooth 0 or 0.5
          drifts back into the flip within 5 to 8 steps. From iso,
          h_smooth 1.0: flank 0.035 but core p95 88 (part of the core
          line stays flipped). Knee 0.7 (iso) and 0.5 (mg): oracle in 3
          steps (flank 2.59 -> 0.057, 0.66 -> 0.049). PN from iso: 1.80
          -> 0.34 in 7 steps (oracle 0.35).
        mu 0.95 needs --sigma_cap 5 (the needle wants sigma_x = 3.9) and
          --h_smooth 2.0: flank 1.71 -> 0.0235 (oracle 0.022), core 7.7
          -> 4.1 (oracle 2.8), core p95 still 90; with h_smooth 1.0 it
          reaches flank 0.029 at step 4 and then drifts into the flip.
      No SH frame has been run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import erf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_ROOT / "src", _ROOT / "src" / "op_extract", _ROOT / "src" / "utils", _HERE):
    sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------

def knee_field(mu, beta, n_per, half_y, dx):
    k1 = float(np.sqrt(1.0 - mu ** 2))
    Lx = n_per * 2 * np.pi / k1
    nx = int(round(Lx / dx))
    x = (np.arange(nx) - nx // 2) * dx
    y = np.arange(-half_y, half_y + 0.5 * dx, dx)
    X, Y = np.meshgrid(x, y)
    z = beta * mu * Y
    theta = k1 * X + np.logaddexp(z, -z) / beta
    k1t = np.full_like(X, k1)
    k2t = mu * np.tanh(z)
    H11 = np.zeros_like(X); H12 = np.zeros_like(X)
    H22 = beta * mu ** 2 / np.cosh(z) ** 2
    return dict(u=np.cos(theta), x=x, y=y, X=X, Y=Y, k1=k1t, k2=k2t,
                H11=H11, H12=H12, H22=H22, mu=mu, Lsp=np.pi / k1, row=0.0)


def pn_single_field(half_x, half_y, dx, y_clip):
    a = 0.5 * (1.0 + np.exp(-np.pi)); b = 0.5 * (np.exp(-np.pi) - 1.0)
    x = np.arange(-half_x, half_x + 0.5 * dx, dx)
    y = np.arange(-half_y, half_y + 0.5 * dx, dx)
    X, Y = np.meshgrid(x, y)
    Ya = np.maximum(np.abs(Y), y_clip)
    r = np.sqrt(2.0 * Ya); s = X / r
    E1 = 2.0 / np.sqrt(np.pi) * np.exp(-s ** 2)                         # erf'
    d1, d2 = E1, -2 * s * E1
    d3, d4 = (4 * s ** 2 - 2) * E1, (-8 * s ** 3 + 12 * s) * E1
    psi = a + b * erf(s)
    p1 = b * d1 / r / psi; p2 = b * d2 / r ** 2 / psi
    p3 = b * d3 / r ** 3 / psi; p4 = b * d4 / r ** 4 / psi
    thp = Ya - np.log(psi)
    kx_p, ky_p = -p1, 1.0 - 0.5 * p2
    H11p = -(p2 - p1 ** 2); H12p = -0.5 * (p3 - p1 * p2); H22p = -0.25 * (p4 - p2 ** 2)
    up = Y >= 0
    sg = np.where(up, 1.0, -1.0)
    theta = sg * thp
    k1 = sg * kx_p                       # theta(x, y) = -theta+(x, -y) below the line
    k2 = ky_p
    H11 = sg * H11p; H12 = H12p; H22 = sg * H22p
    return dict(u=np.cos(theta), x=x, y=y, X=X, Y=Y, k1=k1, k2=k2,
                H11=H11, H12=H12, H22=H22, mu=float("nan"), Lsp=np.pi, row=0.0)


def _load_npz(path):
    try:
        from mg_zigzag_theta_t import load_npz                          # noqa: E402
        return load_npz(path)
    except Exception:                                                   # pragma: no cover
        d = np.load(path, allow_pickle=True)
        md = json.loads(str(d["metadata_json"])) if "metadata_json" in d.files else {}
        u = d["u"]
        if u.ndim == 3 and u.shape[-1] >= u.shape[0]:
            u = np.moveaxis(u, 0, -1)
        t = d["tt"] if "tt" in d.files else d["t"]
        return dict(u=u, x=d["x"], y=d["y"], t=np.asarray(t, float), md=md)


def data_field(path, frame, crop_L, mu_default):
    D = _load_npz(path)
    u, x, y, t, md = D["u"], D["x"], D["y"], D["t"], D["md"]
    mu = float(md.get("mu", mu_default)); k1 = np.sqrt(1 - mu ** 2); Lsp = np.pi / k1
    jc = int(md.get("gb_row_in_crop", u.shape[0] // 2))
    f = frame if frame >= 0 else u.shape[2] + frame
    sel = np.abs(y - y[jc]) <= crop_L * Lsp if crop_L > 0 else np.ones_like(y, bool)
    X, Y = np.meshgrid(x, y[sel])
    return dict(u=np.asarray(u[sel, :, f], float), x=x, y=y[sel], X=X, Y=Y, mu=mu, Lsp=Lsp,
                row=float(y[jc]), t=float(t[f]), frame=f, k1=None)


# ---------------------------------------------------------------------
# the tensor, the averages, the window
# ---------------------------------------------------------------------

def _d1(F, h, axis):
    G = np.pad(F, [(2, 2) if a == axis else (0, 0) for a in range(F.ndim)], mode="reflect")
    sl = lambda o: tuple(slice(2 + o, G.shape[a] - 2 + o) if a == axis else slice(None)
                         for a in range(F.ndim))
    return (-G[sl(2)] + 8 * G[sl(1)] - 8 * G[sl(-1)] + G[sl(-2)]) / (12 * h)


def _d2(F, h, axis):
    G = np.pad(F, [(2, 2) if a == axis else (0, 0) for a in range(F.ndim)], mode="reflect")
    sl = lambda o: tuple(slice(2 + o, G.shape[a] - 2 + o) if a == axis else slice(None)
                         for a in range(F.ndim))
    return (-G[sl(2)] + 16 * G[sl(1)] - 30 * G[sl(0)] + 16 * G[sl(-1)] - G[sl(-2)]) / (12 * h * h)


def uhu_pointwise(u, dx):
    """u Hess u, fourth-order differences (reflecting edges)"""
    return (u * _d2(u, dx, 1), u * _d1(_d1(u, dx, 1), dx, 0), u * _d2(u, dx, 0), u * u)


def average_iso(fields, sigma, dx, box_sigmas):
    """constant isotropic Gaussian, normalised against the box edges"""
    sg = sigma / dx
    one = np.ones_like(fields[0])
    nrm = gaussian_filter(one, sg, mode="constant", cval=0.0, truncate=box_sigmas)
    return [gaussian_filter(F, sg, mode="constant", cval=0.0, truncate=box_sigmas) / nrm
            for F in fields]


def average_var(fields, S11, S12, S22, dx, box_sigmas, sigma_cap):
    """per-point anisotropic Gaussian by gathering over offsets inside the
    disc of radius box_sigmas x the largest principal width in the field"""
    det = np.maximum(S11 * S22 - S12 ** 2, 1e-30)
    A, B, C = S22 / det, -S12 / det, S11 / det                          # S^{-1}
    lam_max = 0.5 * (S11 + S22) + np.sqrt(0.25 * (S11 - S22) ** 2 + S12 ** 2)
    smax = min(float(np.sqrt(np.nanmax(lam_max))), sigma_cap)
    R = int(np.ceil(box_sigmas * smax / dx))
    ny, nx = fields[0].shape
    pad = [np.pad(F, R, mode="constant") for F in fields]
    valid = np.pad(np.ones((ny, nx)), R, mode="constant")
    acc = [np.zeros((ny, nx)) for _ in fields]
    wsum = np.zeros((ny, nx))
    for oy in range(-R, R + 1):
        for ox in range(-R, R + 1):
            if ox * ox + oy * oy > R * R:
                continue
            ddx, ddy = ox * dx, oy * dx
            q = A * ddx * ddx + 2 * B * ddx * ddy + C * ddy * ddy
            w = np.exp(-0.5 * q) * valid[R + oy:R + oy + ny, R + ox:R + ox + nx]
            wsum += w
            for a, P in zip(acc, pad):
                a += w * P[R + oy:R + oy + ny, R + ox:R + ox + nx]
    return [a / np.maximum(wsum, 1e-300) for a in acc], R


def k_from_tensor(Txx, Txy, Tyy, U2, kmag):
    """dominant (most negative) eigenvector of <T> as the k direction, mod pi,
    oriented k1 >= 0; |k|^2 = -lambda_min / <u^2> (or -trace / <u^2>)"""
    ang = 0.5 * np.arctan2(2 * Txy, Txx - Tyy) + 0.5 * np.pi
    lam_min = 0.5 * (Txx + Tyy) - np.sqrt(0.25 * (Txx - Tyy) ** 2 + Txy ** 2)
    num = -(Txx + Tyy) if kmag == "trace" else -lam_min
    km = np.sqrt(np.maximum(num / np.maximum(U2, 1e-300), 0.0))
    k1, k2 = km * np.cos(ang), km * np.sin(ang)
    s = np.where(k1 < 0, -1.0, 1.0)
    return k1 * s, k2 * s


def hessian_signsafe(k1, k2, dx, smooth):
    """sym grad k with each neighbour aligned to the centre before the
    difference (k is a director); optional Gaussian smoothing of H"""
    def dd(F1, F2, axis):
        P1 = np.pad(F1, [(1, 1) if a == axis else (0, 0) for a in (0, 1)], mode="edge")
        P2 = np.pad(F2, [(1, 1) if a == axis else (0, 0) for a in (0, 1)], mode="edge")
        sl = lambda o: tuple(slice(1 + o, P1.shape[a] - 1 + o) if a == axis else slice(None)
                             for a in (0, 1))
        sp = np.sign(P1[sl(1)] * F1 + P2[sl(1)] * F2); sp[sp == 0] = 1
        sm = np.sign(P1[sl(-1)] * F1 + P2[sl(-1)] * F2); sm[sm == 0] = 1
        return ((sp * P1[sl(1)] - sm * P1[sl(-1)]) / (2 * dx),
                (sp * P2[sl(1)] - sm * P2[sl(-1)]) / (2 * dx))
    k1x, k2x = dd(k1, k2, 1)
    k1y, k2y = dd(k1, k2, 0)
    H11, H22, H12 = k1x, k2y, 0.5 * (k1y + k2x)
    if smooth > 0:
        H11, H12, H22 = (gaussian_filter(F, smooth / dx, mode="nearest") for F in (H11, H12, H22))
    return H11, H12, H22


def sstar(k1, k2, H11, H12, H22, tol, s_min, s_cap):
    """the two rules in the regular form; returns S11, S12, S22, the two
    principal variances and the at-cap / at-floor flags"""
    L0 = 0.5 * np.log(1.0 / tol)
    th = 0.5 * np.arctan2(2 * H12, H11 - H22)
    c, s = np.cos(th), np.sin(th)
    rad = np.sqrt(0.25 * (H11 - H22) ** 2 + H12 ** 2)
    hA, hB = 0.5 * (H11 + H22) + rad, 0.5 * (H11 + H22) - rad
    kA, kB = k1 * c + k2 * s, -k1 * s + k2 * c
    den = kA ** 2 * hB ** 2 + kB ** 2 * hA ** 2
    deg = den < 1e-24
    den = np.where(deg, 1.0, den)
    sA = np.where(deg, s_cap, L0 * hB ** 2 / den)
    sB = np.where(deg, s_cap, L0 * hA ** 2 / den)
    capA, capB = sA >= s_cap, sB >= s_cap
    sA, sB = np.clip(sA, s_min, s_cap), np.clip(sB, s_min, s_cap)
    S11 = sA * c * c + sB * s * s
    S22 = sA * s * s + sB * c * c
    S12 = (sA - sB) * c * s
    floor = (sA <= s_min * (1 + 1e-9)) & (sB <= s_min * (1 + 1e-9))
    return S11, S12, S22, sA, sB, capA | capB, floor


def mg_start(F, dx, sigma_dir):
    """the monogenic-gradient k of mg_estimator_validation (frozen layer
    through that script); its window is in the director only, so it does
    not flip on sharp knees (Chapter 5)"""
    from mg_estimator_validation import mg_estimator                    # noqa: E402
    ny, nx = F["u"].shape
    e = mg_estimator(F["u"], nx * dx, ny * dx, sigma_dir)
    s = np.where(e["k1"] < 0, -1.0, 1.0)
    return e["k1"] * s, e["k2"] * s


def angle_diff_deg(a1, a2, b1, b2):
    """angle between two directors, degrees in [0, 90]"""
    d = np.arctan2(a2, a1) - np.arctan2(b2, b1)
    d = np.abs((d + 0.5 * np.pi) % np.pi - 0.5 * np.pi)
    return np.degrees(d)


def W_measure(k1, k2, H11, H12, H22):
    return np.sqrt(H11 ** 2 + 2 * H12 ** 2 + H22 ** 2) / np.maximum(k1 ** 2 + k2 ** 2, 1e-12)


# ---------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------

def regions(F, args, R):
    """scoring regions (interior: R grid points from every edge)"""
    X, Y = F["X"], F["Y"]
    ny, nx = X.shape
    inner = np.zeros((ny, nx), bool)
    inner[R:ny - R, R:nx - R] = True
    reg = {}
    if args.mode == "knee":
        reg["core |y|<0.5"] = inner & (np.abs(Y) < 0.5)
        reg["flank 0.5<=|y|<=4"] = inner & (np.abs(Y) >= 0.5) & (np.abs(Y) <= 4.0)
    elif args.mode == "pn":
        Wt = W_measure(F["k1"], F["k2"], F["H11"], F["H12"], F["H22"])
        reg["valid W<0.5, |y|>=0.6"] = inner & (Wt < 0.5) & (np.abs(Y) >= 0.6)
        reg["near core W>=0.5"] = inner & (Wt >= 0.5)
    else:
        reg["interior"] = inner
    return reg


def run(args):
    t0 = time.time()
    if args.mode == "knee":
        F = knee_field(args.mu, args.beta, args.n_per, args.half_y, args.dx)
        stem = "knee_mu%.2f_beta%.2g" % (args.mu, args.beta)
    elif args.mode == "pn":
        F = pn_single_field(args.half_x, args.half_y, args.dx, args.y_clip)
        stem = "pn_single"
    else:
        F = data_field(args.npz, args.frame, args.crop_L, args.mu)
        stem = Path(args.npz).stem + "_f%d" % F["frame"]
    dx = float(F["x"][1] - F["x"][0])
    stamp = "tol%.2g_s0%.3g_smin%.3g_cap%.3g_om%.2g_hs%.2g_%s_init%s" % (
        args.tol, args.sigma0, args.sigma_min, args.sigma_cap, args.omega, args.h_smooth, args.kmag, args.init)
    if args.tag:
        stamp += "_" + args.tag
    out = Path(args.outdir) / stem / stamp
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as fh:
        json.dump({k: getattr(args, k) for k in vars(args)}, fh, indent=2, default=str)
    s_min, s_cap = args.sigma_min ** 2, args.sigma_cap ** 2
    truth = F.get("k1") is not None
    lines = ["uhu_sstar_iterate v1.0   %s   grid %dx%d dx %.4g" % (stem, F["u"].shape[1], F["u"].shape[0], dx),
             "tol %.3g (L0 %.4f), sigma0 %.4g, sigma_min %.3g, sigma_cap %.3g, omega %.2g,"
             " h_smooth %.3g, |k| from %s, box %.2g sigmas, W_max %.2g, osc %.2g deg" %
             (args.tol, 0.5 * np.log(1 / args.tol), args.sigma0, args.sigma_min, args.sigma_cap,
              args.omega, args.h_smooth, args.kmag, args.box_sigmas, args.W_max, args.osc_deg)]
    if args.mode == "data":
        lines.append("data frame %d (t = %.3f), mu %.3f, L %.3f, rows within %.2g L of the row" %
                     (F["frame"], F["t"], F["mu"], F["Lsp"], args.crop_L))
    T = uhu_pointwise(F["u"], dx)
    R_iso = int(np.ceil(args.box_sigmas * args.sigma0 / dx))
    R_cap = int(np.ceil(args.box_sigmas * args.sigma_cap / dx))
    reg = regions(F, args, R_cap)
    Tavg = average_iso(T, args.sigma0, dx, args.box_sigmas)
    k1, k2 = k_from_tensor(*Tavg, args.kmag)
    k1_iso, k2_iso = k1.copy(), k2.copy()
    S11 = np.full_like(k1, args.sigma0 ** 2); S12 = np.zeros_like(k1); S22 = S11.copy()
    if args.init == "truth":
        if not truth:
            raise SystemExit("--init truth needs a synthetic mode")
        k1, k2 = F["k1"].copy(), F["k2"].copy()
        lines.append("start: the exact k (tests whether the truth is a fixed point)")
    elif args.init == "mg":
        k1, k2 = mg_start(F, dx, args.sigma_dir)
        lines.append("start: the monogenic-gradient k (sigma_dir %.3g), oriented k1 >= 0" % args.sigma_dir)
    else:
        lines.append("start: the isotropic window sigma0")

    hist = []

    def score(k1_, k2_, label, extra=None):
        row = {"step": label}
        if truth:
            e = angle_diff_deg(k1_, k2_, F["k1"], F["k2"])
            for nm, m in reg.items():
                row["med " + nm] = float(np.median(e[m])) if m.any() else np.nan
                row["p95 " + nm] = float(np.percentile(e[m], 95)) if m.any() else np.nan
        if extra:
            row.update(extra)
        return row

    hist.append(score(k1, k2, 0, {"dmed": np.nan, "dp95": np.nan}))
    lines.append("step 0 (%s start) done, %.1f s" % (args.init, time.time() - t0))
    dchg = None
    for n in range(1, args.n_iter + 1):
        H11, H12, H22 = hessian_signsafe(k1, k2, dx, args.h_smooth)
        Sn11, Sn12, Sn22, sA, sB, cap, floor = sstar(k1, k2, H11, H12, H22, args.tol, s_min, s_cap)
        om = 1.0 if (n == 1 and args.init != "iso") else args.omega   # damp from S*(k^0), not from the iso window
        S11 = (1 - om) * S11 + om * Sn11
        S12 = (1 - om) * S12 + om * Sn12
        S22 = (1 - om) * S22 + om * Sn22
        Tavg, R = average_var(T, S11, S12, S22, dx, args.box_sigmas, args.sigma_cap)
        k1n, k2n = k_from_tensor(*Tavg, args.kmag)
        dchg = angle_diff_deg(k1n, k2n, k1, k2)
        allm = np.zeros_like(dchg, bool)
        for m in reg.values():
            allm |= m
        dmed, dp95 = float(np.median(dchg[allm])), float(np.percentile(dchg[allm], 95))
        k1, k2 = k1n, k2n
        hist.append(score(k1, k2, n, {"dmed": dmed, "dp95": dp95, "frac_cap": float(cap[allm].mean()),
                                      "frac_floor": float(floor[allm].mean()), "R": R}))
        lines.append("step %d: angle change median %.4g deg, p95 %.4g deg; at cap %.3f, at floor %.3f;"
                     " gather radius %d cells; %.1f s" % (n, dmed, dp95, cap[allm].mean(),
                                                          floor[allm].mean(), R, time.time() - t0))
        if n >= 2 and dmed < args.conv_deg:
            lines.append("converged: median change below %.3g deg" % args.conv_deg)
            break
    # final masks
    H11, H12, H22 = hessian_signsafe(k1, k2, dx, args.h_smooth)
    Wf = W_measure(k1, k2, H11, H12, H22)
    _, _, _, sA, sB, cap, floor = sstar(k1, k2, H11, H12, H22, args.tol, s_min, s_cap)
    mask_W = Wf >= args.W_max
    mask_osc = dchg > args.osc_deg if dchg is not None else np.zeros_like(Wf, bool)
    core = mask_W | mask_osc
    oracle = None
    if truth:
        So = sstar(F["k1"], F["k2"], F["H11"], F["H12"], F["H22"], args.tol, s_min, s_cap)
        To, _ = average_var(T, So[0], So[1], So[2], dx, args.box_sigmas, args.sigma_cap)
        oracle = k_from_tensor(*To, args.kmag)
        hist.append(score(oracle[0], oracle[1], "oracle"))
    # report table
    lines.append("")
    keys = [k for k in hist[0] if k.startswith("med ") or k.startswith("p95 ")]
    if keys:
        lines.append("angle error against the exact k (degrees), by step: " + "; ".join(keys))
        for h in hist:
            lines.append("  %-7s " % h["step"] + "  ".join("%9.4g" % h[k] for k in keys))
        if args.mode == "knee":
            lines.append("  (Chapter 4, Gauss-Hermite on the exact tensor, knee mu 0.9: core 59.4 -> 1.8,"
                         " flank 1.20 -> 0.034 deg; the oracle row here is the grid version)")
        if args.mode == "pn":
            lines.append("  (Chapter 4, single PN, W < 0.5, y >= 0.6: 1.27 -> 0.20 deg)")
    allm = np.zeros_like(Wf, bool)
    for m in reg.values():
        allm |= m
    lines.append("")
    lines.append("final masks over the scored interior: W >= %.2g %.4f, not converged (> %.2g deg) %.4f,"
                 " both widths at the floor %.4f, core mask (W or not converged) %.4f" %
                 (args.W_max, mask_W[allm].mean(), args.osc_deg, mask_osc[allm].mean(),
                  floor[allm].mean(), core[allm].mean()))
    if truth:
        Wt = W_measure(F["k1"], F["k2"], F["H11"], F["H12"], F["H22"])
        tc = Wt >= args.W_max
        if tc[allm].any():
            hit = (core & tc)[allm].sum() / tc[allm].sum()
            fa = (core & ~tc)[allm].sum() / max((~tc)[allm].sum(), 1)
            lines.append("core mask against the true W >= %.2g: recall %.3f, false-positive rate %.4f" %
                         (args.W_max, hit, fa))
    lines.append("elapsed %.1f s" % (time.time() - t0))
    rep = "\n".join(lines)
    (out / "report.txt").write_text(rep)
    np.savez_compressed(out / "fields.npz", x=F["x"], y=F["y"], u=F["u"], k1=k1, k2=k2,
                        k1_iso=k1_iso, k2_iso=k2_iso, S11=S11, S12=S12, S22=S22,
                        sigma_wide=np.sqrt(np.maximum(sA, sB)), sigma_thin=np.sqrt(np.minimum(sA, sB)),
                        W=Wf, mask_W=mask_W, mask_osc=mask_osc, mask_floor=floor, core=core)
    fig_convergence(hist, keys, out / "convergence.png", stem)
    fig_maps(F, k1, k2, k1_iso, k2_iso, oracle, sA, sB, S11, S12, S22, Wf, core, out / "maps.png", stem, truth)
    print(rep); print("outputs ->", out)
    return 0


def fig_convergence(hist, keys, path, stem):
    steps = [h["step"] for h in hist if h["step"] != "oracle"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axs[0]
    ax.semilogy(steps[1:], [h["dmed"] for h in hist[1:len(steps)]], "o-", label="median change")
    ax.semilogy(steps[1:], [h["dp95"] for h in hist[1:len(steps)]], "s--", label="p95 change")
    ax.set_xlabel("iteration"); ax.set_ylabel("k angle change (deg)"); ax.legend(fontsize=8)
    ax.set_title("convergence")
    ax = axs[1]
    if keys:
        orc = [h for h in hist if h["step"] == "oracle"]
        for i, k in enumerate(keys):
            ax.semilogy(steps, [h[k] for h in hist[:len(steps)]], "o-", color="C%d" % i, label=k)
            if orc:
                ax.axhline(orc[0][k], color="C%d" % i, ls=":", lw=1)
        ax.set_xlabel("iteration"); ax.set_ylabel("angle error vs exact (deg)")
        ax.set_title("error (dotted: oracle S* at the true k, H)")
        ax.legend(fontsize=7)
    else:
        ax.axis("off")
    fig.suptitle(stem); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_maps(F, k1, k2, k1i, k2i, oracle, sA, sB, S11, S12, S22, Wf, core, path, stem, truth):
    ext = [F["x"][0], F["x"][-1], F["y"][0], F["y"][-1]]
    pan = []
    if truth:
        pan.append(("angle error, isotropic (deg)", angle_diff_deg(k1i, k2i, F["k1"], F["k2"]), "magma", (0, 10)))
        pan.append(("angle error, fixed point (deg)", angle_diff_deg(k1, k2, F["k1"], F["k2"]), "magma", (0, 10)))
        if oracle is not None:
            pan.append(("angle error, oracle S* (deg)", angle_diff_deg(oracle[0], oracle[1], F["k1"], F["k2"]),
                        "magma", (0, 10)))
    else:
        pan.append(("u", F["u"], "gray", (None, None)))
        pan.append(("angle change iso -> fixed point (deg)", angle_diff_deg(k1, k2, k1i, k2i), "magma", (0, 10)))
    wide = np.sqrt(np.maximum(sA, sB)); thin = np.sqrt(np.minimum(sA, sB))
    pan.append(("wide width", wide, "viridis", (0, None)))
    pan.append(("thin width", thin, "viridis", (0, None)))
    th = 0.5 * np.degrees(np.arctan2(2 * S12, S11 - S22))
    pan.append(("wide-axis angle (deg)", th, "twilight", (-90, 90)))
    pan.append(("W = |H| / |k|^2", np.minimum(Wf, 2.0), "viridis", (0, 2)))
    pan.append(("core mask over u", None, None, None))
    n = len(pan); nc = 4; nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.3 * nr), squeeze=False)
    for ax, (t, G, cm, lim) in zip(axs.ravel(), pan):
        if G is None:
            ax.imshow(F["u"], origin="lower", extent=ext, cmap="gray", aspect="auto")
            ax.imshow(np.ma.masked_where(~core, np.ones_like(F["u"])), origin="lower", extent=ext,
                      cmap=matplotlib.colors.ListedColormap(["#d62728"]), alpha=0.45, aspect="auto")
        else:
            im = ax.imshow(G, origin="lower", extent=ext, cmap=cm, vmin=lim[0], vmax=lim[1], aspect="auto")
            fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(t, fontsize=9)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(stem); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    mode = "knee"                 # knee | pn | data
    npz = ""                      # data mode: SH npz
    frame = -1                    # data mode: frame index (negative from the end)
    crop_L = 3.0                  # data mode: rows within crop_L spacings of the row (0: all)
    mu = 0.9                      # knee mode (and data fallback when metadata lacks mu)
    beta = 1.0                    # knee mode self-dual coefficient
    n_per = 4                     # knee mode: x periods (2 pi / k1)
    half_x = 12.0                 # pn mode half width in x
    half_y = 16.0                 # knee and pn half height
    y_clip = 0.02                 # pn mode: |y| floor inside erf(x / sqrt(2|y|))
    dx = 0.25                     # synthetic grid step
    tol = 0.05                    # surviving oscillation (Chapter 4)
    sigma0 = float(np.pi / 2)     # isotropic first step (Chapter 4's isotropic window)
    sigma_min = 0.15              # floor of each principal width (S* demo: sigma_y 0.15)
    sigma_cap = 3.0               # cap of each principal width (glyph figure: 3)
    omega = 1.0                   # damping of the window update (1 = none; the first step is never damped toward iso)
    h_smooth = 1.0                # Gaussian smoothing of H (length units); 2.0 at mu >= 0.93
    kmag = "eig"                  # eig | trace for |k|
    init = "mg"                   # mg | iso | truth (synthetic only): the k that the first window is built from
    sigma_dir = 0.4               # init mg: director width of the monogenic-gradient estimator
    n_iter = 8
    conv_deg = 0.01               # stop when the median angle change falls below
    osc_deg = 1.0                 # last-step change above this marks a point unconverged
    W_max = 0.5                   # quadratic-model validity (S* demo standoff)
    box_sigmas = 3.0              # gather radius in principal widths
    outdir = str(_HERE / "results" / "uhu_sstar_iterate")
    tag = ""


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in dir(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    cli = p.parse_args()
    if cli.npz:
        cli.mode = "data"
    sys.exit(run(cli))
