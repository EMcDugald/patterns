# pipelines/self_dual_discovery.py
"""
Could the self-dual balance be FOUND from stationary data by someone who
did not know it existed? A companion to rcn_balance_ledger.py, which
TESTS the balance (forced one-column ratios); this script first SEARCHES
for it with no relation assumed, then runs the forced checks that a
newcomer would run next.

Stage 1, free implicit search (stationary data have no time derivative to
put on the left, so every candidate column takes a turn as the target).
Library of low-order scalars built from k alone, all weak (test function
of half width --radius spacings, the ledger's polynomial bump):
    1, w, w^2, w^3                       (no derivative; w = |k|^2 - 1)
    div k, curl k, w div k, k.grad w     (one derivative)
    (div k)^2, J, Lap w                  (two)
    Lap div k                            (three)
Targets are the columns with at least one derivative (the algebraic
columns are related to each other by powers of w and would top any
ranking for no physical reason). For each target, every support of size
1 and 2 from the remaining columns is fitted by least squares on the half
of the query centres with x below the median and scored by the holdout
R^2 on the other half. The HEADLINE relation is the size-1 relation with
the fewest total derivatives among those within --tol of the best holdout
R^2: the lowest-order statement the data make, the others in the family
being its consequences (for example (div k)^2 ~ w^2 follows from
div k ~ w).

Stage 2, forced follow-ups (the relation a newcomer would test after
seeing div k ~ w):
  (a) div k = -beta w: beta, holdout R^2; beta by distance to the line and
      to the nearest core (is it a constant or one region's fit?).
  (b) equipartition: (div k)^2 = gamma^2 w^2 on the weak columns. If (a)
      holds pointwise then gamma = |beta|: bending and stretching energy
      densities stand in a fixed ratio everywhere, which is the
      definition of self-duality (equal parts when beta^2 = zeta/eta).
      Reported as gamma / |beta| (1 for a self-dual field).
  (c) the fourth-order consequence Lap div k = lambda S, S = div(w k),
      against lambda = 2 beta^2 (exact on a self-dual field with J = 0),
      and the two-column {S, J} fit.
  (d) where it fails: rms of the self-dual residual rho = div k + beta w
      (weak) by core distance, beside the share of the J signal within
      one spacing of a core.

Modes
  knee   analytic self-dual knee theta = k1 x + log cosh(b mu y)/b
         (planted beta = --beta); expect headline div k ~ w, beta = b,
         gamma/|beta| = 1, lambda = 2 b^2.
  nonsd  NEGATIVE CONTROL: a knee with the same core slope but an erf
         profile k2 = mu erf(c mu y), c = b sqrt(pi)/2, which is NOT
         self-dual; the search must not report self-duality on it.
  pn     the Pismen-Nepomnyashchy row (pn_field.pismen_on_sh_grid) at the
         zigzag spacing pi/k1, upper half-plane (its relation is odd in
         y). Distances in dislocation spacings L = pi/k1 here, the
         paper's unit (rcn_balance_ledger's pn mode uses 2 pi instead).
  data   SH frames (--npz), estimator route, last frame by default.
Routes for the synthetic modes: exact (analytic k) and/or estimator
(u rendered and passed through mg_zigzag_theta_t.estimate); data is
estimator only.

Conventions copied from rcn_balance_ledger v1.5 so the numbers line up
with Appendix S: one scalar sign of k per frame from the component with
the larger |mean|; band = |y - y_row| < 2 L; core discs of 0.5 L removed
(centres only); rim of 10 percent of the box excluded; radius 1.5 L;
stride 4; inner derivative np.gradient (eta = 0).

Outputs: <script dir>/results/self_dual_discovery/<stem>/<stamp>/
  report.txt, stage1.csv, stage2.csv, stage1_<label>.png, config.json

Changelog
---------
v1.0  first version (6 Oct 2026). Validated on the EXACT route only (the
      sandbox has no estimator chain; run --route both locally for the
      estimator route). Grid Ly 60, 300 rows, mu 0.85, band < 2 L, core
      discs 0.5 L, radius 1.5 L unless stated.
      knee beta 1.0 / 1.3 (planted): headline div k ~ w at every radius
        (coefficient -1.000 / -1.300, holdout R2 1.000); beta 1.000 /
        1.300 at radii 0.25, 0.5, 1, 1.5 L (spread 0.00); gamma/|beta|
        0.994-0.996 / 0.991-0.993; lambda/2beta^2 0.99; pointwise
        -div k/w spread 0.03 / 0.05 across row bins (1.28-1.34 at 1.3,
        the np.gradient error of a crease 0.15 L wide).
      nonsd (erf knee, same core slope, NOT self-dual): the weak form
        cannot reject it. Headline w div k ~ w^2 (div k ~ w also at R2
        1.000), beta 1.389-1.439 across radii (spread 0.04, "persistent"),
        gamma/|beta| 0.95-0.97. Only the pointwise reading separates it:
        -div k/w rises 1.29 -> 2.12 across the profile (spread 0.53, FLAG).
        Reason: the crease (about 0.15 L) is narrower than every test
        function tried, so each weak column is the profile's integral
        times a smooth kernel, and any two co-localised fields look
        proportional. Hence (e) and the SUMMARY line.
      pn (exact, upper half-plane, 10 dislocations, unit L = pi/k1):
        at 1.5 L the kernels reach the singular cores and everything
        collapses onto core blobs (headline curl k ~ w^2, beta -0.25 at
        R2 -1.9). At 0.25 L the relation appears: beta -1.45 (R2 0.89),
        the record's -1.48 (R2 0.91; Ch. 6, tab:pn_vs_sh); pointwise
        -div k/w = -1.26 / -1.27 at 0.5-2 L from the cores, drifting to
        -2.4 within 0.2 L of the line (paraxial PN is self-dual only
        approximately there).

Run
  python self_dual_discovery.py --mode knee --beta 1.3 --route both
  python self_dual_discovery.py --mode nonsd --route both
  python self_dual_discovery.py --mode pn --route both
  python self_dual_discovery.py --npz $F85
  python self_dual_discovery.py --npz $F85 --frames window --t0 20
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.ndimage import correlate1d
from scipy import special

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_ROOT / "src", _ROOT / "src" / "op_extract", _ROOT / "src" / "utils", _HERE):
    sys.path.insert(0, str(_p))

import matplotlib                                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

# --- weak-form helpers: the frozen ones when importable, else copies of
# --- the documented fallbacks inside wsindy_theta_t (same operations)
try:
    from wsindy_theta_t import kernel_d3, corr1d, inner_deriv       # noqa: E402
    _HELPERS = "wsindy_theta_t"
except ImportError:
    _HELPERS = "local copies of the wsindy_theta_t fallbacks"

    def kernel_d3(half_width, spacing, p=6):
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

    def corr1d(F, kern, axis, spacing):
        return correlate1d(F, kern * spacing, axis=axis, mode="constant", cval=0.0)

    def inner_deriv(F, eta, dx, dy, ax, kind="poly", p=6):
        h = dx if ax == "x" else dy
        axis = 1 if ax == "x" else 0
        if eta <= 0:
            return np.gradient(F, h, axis=axis)
        f, f1, *_ = kernel_d3(eta, h, p)
        return -correlate1d(F, f1 * h, axis=axis, mode="nearest")

try:
    from op_extract.pn_field import pismen_on_sh_grid                # noqa: E402
    _PN = "op_extract.pn_field"
except ImportError:
    try:
        from pn_field import pismen_on_sh_grid                      # noqa: E402
        _PN = "pn_field"
    except ImportError:
        _PN = "local copy of pn_field.pismen_on_sh_grid"

        def pismen_on_sh_grid(X, Y, x_defects, amp=1.0):
            x_defects = np.asarray(x_defects, dtype=float)
            npts = len(x_defects)
            if npts == 0:
                return Y.copy(), amp * np.cos(Y)
            thetas = np.arange(npts + 1) * np.pi
            psi = 0.5 * (1 + np.exp(-thetas[-1]))
            for j in range(npts):
                delta = 0.5 * (np.exp(-thetas[j + 1]) - np.exp(-thetas[j]))
                psi += delta * special.erf((X - x_defects[j]) / np.sqrt(2 * np.abs(Y) + 1e-8))
            theta = Y - np.sign(Y) * np.log(psi)
            return theta, amp * np.cos(theta)


def _estimator():
    """the frozen estimator chain, imported only when a route needs it"""
    from mg_zigzag_theta_t import load_npz, estimate, find_cores, dist_to_cores   # noqa
    from op_extract import t1_common as t1c                                        # noqa
    return load_npz, estimate, find_cores, dist_to_cores, t1c


# library: name -> number of derivatives of k it carries (w counts as 0)
ORDER = {"1": 0, "w": 0, "w^2": 0, "w^3": 0,
         "div k": 1, "curl k": 1, "w div k": 1, "k.grad w": 1,
         "(div k)^2": 2, "J": 2, "Lap w": 2, "Lap div k": 3}
LIB = list(ORDER)
TARGETS = [n for n in LIB if ORDER[n] >= 1]
CORE_BINS = [(0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, np.inf)]
ROW_BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0)]


def r2(y, yh):
    return float(1.0 - np.sum((y - yh) ** 2) / (np.sum((y - np.mean(y)) ** 2) + 1e-300))


# ---------------------------------------------------------------------
# weak columns on one frame
# ---------------------------------------------------------------------

def weak_columns(k1, k2, x, y, Lsp, args, radius=None):
    """full 2D weak fields of every library column (and S for stage 2)"""
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    a = (args.radius if radius is None else radius) * Lsp
    fx, fx1, fx2, fx3, mx = kernel_d3(a, dx, args.poly_p)
    fy, fy1, fy2, fy3, my = kernel_d3(a, dy, args.poly_p)
    D = lambda F, ax: inner_deriv(F, args.eta * Lsp, dx, dy, ax, "poly", args.poly_p)
    K = lambda F, kx, ky: corr1d(corr1d(F, ky, 0, dy), kx, 1, dx)
    w = k1 ** 2 + k2 ** 2 - 1.0
    k1x, k1y, k2x, k2y = D(k1, "x"), D(k1, "y"), D(k2, "x"), D(k2, "y")
    divk = k1x + k2y
    J = k1x * k2y - k1y * k2x
    wx, wy = D(w, "x"), D(w, "y")
    S = w * divk + k1 * wx + k2 * wy
    C = {"1": np.ones_like(w),
         "w": K(w, fx, fy), "w^2": K(w ** 2, fx, fy), "w^3": K(w ** 3, fx, fy),
         "div k": -(K(k1, fx1, fy) + K(k2, fx, fy1)),                       # fully weak
         "curl k": -K(k2, fx1, fy) + K(k1, fx, fy1),                        # fully weak
         "w div k": K(w * divk, fx, fy), "k.grad w": K(k1 * wx + k2 * wy, fx, fy),
         "(div k)^2": K(divk ** 2, fx, fy), "J": K(J, fx, fy),
         "Lap w": K(w, fx2, fy) + K(w, fx, fy2),                            # fully weak
         "Lap div k": -(K(k1, fx3, fy) + K(k1, fx1, fy2) + K(k2, fx2, fy1) + K(k2, fx, fy3)),
         "S": K(S, fx, fy)}
    return C, (mx, my)


def centres(shape, m, keep, args):
    ny, nx = shape
    ix = np.arange(m[0] + args.rim, nx - m[0] - args.rim, args.stride)
    iy = np.arange(m[1] + args.rim, ny - m[1] - args.rim, args.stride)
    IY, IX = np.meshgrid(iy, ix, indexing="ij")
    ok = keep[IY, IX]
    return IY[ok], IX[ok]


def domain_keep(x, y, Lsp, y_row, d_core, args, side):
    """the kept region (domain, side, rim, core discs) and the row distance"""
    X, Y = np.meshgrid(x, y)
    Lx, Ly = x[-1] - x[0], y[-1] - y[0]
    d_row = np.abs(Y - y_row) / Lsp
    keep = ((np.abs(X - x.mean()) < (0.5 - args.rim_frac) * Lx) &
            (np.abs(Y - y.mean()) < (0.5 - args.rim_frac) * Ly))
    if args.domain == "band":
        keep &= d_row < args.band_to
    elif args.domain == "shell":
        keep &= (d_row >= args.band_to) & (d_row <= args.far_from)
    if side == "upper":
        keep &= Y > y_row
    elif side == "lower":
        keep &= Y < y_row
    if d_core is not None and args.core_excl > 0:
        keep &= d_core >= args.core_excl
    return keep, d_row


def pointwise_check(fr, args):
    """(e) the relation read pointwise, no test function: median of
    -div k / w and of |div k| / |w| by row and core distance (|w| above
    --w_min). A pointwise relation div k = -beta w gives one constant in
    every bin; a weak-only proportionality (two co-localised fields
    averaged by a kernel wider than both) does not."""
    k1, k2, x, y, L = fr["k1"], fr["k2"], fr["x"], fr["y"], fr["L"]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    D = lambda F, ax: inner_deriv(F, args.eta * L, dx, dy, ax, "poly", args.poly_p)
    w = k1 ** 2 + k2 ** 2 - 1.0
    d = D(k1, "x") + D(k2, "y")
    keep, d_row = domain_keep(x, y, L, fr["y_row"], fr["d_core"], args, fr["side"])
    keep &= np.abs(w) > args.w_min
    rows = []
    edges = [float(v) for v in str(args.pw_row_edges).split(",") if v.strip()]
    pw_row_bins = list(zip(edges[:-1], edges[1:]))
    for key, dist, bins in (("row", d_row, pw_row_bins), ("core", fr["d_core"], CORE_BINS)):
        if dist is None:
            continue
        for lo, hi in bins:
            m = keep & (dist >= lo) & (dist < hi)
            if m.sum() < 30:
                continue
            rows.append((key, lo, hi, int(m.sum()), float(np.median(-d[m] / w[m])), float(np.median(np.abs(d[m] / w[m])))))
    return rows


def sample_frame(k1, k2, x, y, Lsp, y_row, d_core, args, side, radius=None):
    """weak columns at the kept query centres of one frame, with the centre
    coordinates and distances (row, core) in spacings"""
    keep, d_row = domain_keep(x, y, Lsp, y_row, d_core, args, side)
    C, m = weak_columns(k1, k2, x, y, Lsp, args, radius)
    IY, IX = centres(k1.shape, m, keep, args)
    out = {nm: F[IY, IX] for nm, F in C.items()}
    out["_x"] = x[IX]
    out["_drow"] = d_row[IY, IX]
    out["_dcore"] = d_core[IY, IX] if d_core is not None else np.full(len(IX), np.inf)
    return out


def pool(samples):
    return {k: np.concatenate([s[k] for s in samples]) for k in samples[0]}


# ---------------------------------------------------------------------
# stage 1: free implicit search
# ---------------------------------------------------------------------

def ols(Th, b, tr):
    c, *_ = np.linalg.lstsq(Th[tr], b[tr], rcond=None)
    return c, r2(b[~tr], Th[~tr] @ c)


def vanishing(P):
    """targets that are identically zero on this domain (curl k on any
    gradient field, J on a developable phase surface): a 0 = 0 fit has
    R2 1 and says nothing, so they are reported as vanishing instead"""
    ref = max(float(np.std(P[t])) for t in TARGETS)
    return [t for t in TARGETS if float(np.std(P[t])) < 1e-6 * ref]


def free_search(P, args):
    tr = P["_x"] < np.median(P["_x"])
    res = {}
    zero = vanishing(P)
    for t in TARGETS:
        b = P[t]
        preds = [n for n in LIB if n != t]
        best = {}
        for size in range(1, args.max_size + 1):
            cand = []
            for comb in combinations(preds, size):
                Th = np.column_stack([P[n] for n in comb])
                if np.any(np.std(Th, axis=0) == 0) and "1" not in comb:
                    continue
                c, rh = ols(Th, b, tr)
                cand.append((rh, comb, c))
            cand.sort(key=lambda z: -z[0])
            best[size] = cand[:3]
        res[t] = best
    # headline: size-1 relations within tol of the best size-1 holdout R2,
    # fewest total derivatives first, then R2
    one = [(res[t][1][0][0], t, res[t][1][0][1][0], float(res[t][1][0][2][0])) for t in TARGETS
           if t not in zero]
    top = max(o[0] for o in one)
    adm = [o for o in one if o[0] >= top - args.tol and o[2] != "1"]
    adm.sort(key=lambda o: (ORDER[o[1]] + ORDER[o[2]], -o[0]))
    head = adm[0] if adm else max(one)
    return res, one, head, zero


# ---------------------------------------------------------------------
# stage 2: forced follow-ups
# ---------------------------------------------------------------------

def ratio_fit(yv, xv, tr):
    """y = c x, no intercept: c on the train half, holdout R2, full R2"""
    c = float(np.dot(xv[tr], yv[tr]) / max(np.dot(xv[tr], xv[tr]), 1e-300))
    return c, r2(yv[~tr], c * xv[~tr]), r2(yv, c * xv)


def forced_checks(P, args):
    tr = P["_x"] < np.median(P["_x"])
    d, w = P["div k"], P["w"]
    out = {}
    c, rh, rf = ratio_fit(d, w, tr)
    beta = -c
    out["beta"], out["beta_r2_holdout"], out["beta_r2_full"] = beta, rh, rf
    g2, rh2, rf2 = ratio_fit(P["(div k)^2"], P["w^2"], tr)
    out["gamma"] = float(np.sqrt(max(g2, 0.0)))
    out["gamma_r2_holdout"], out["gamma_r2_full"] = rh2, rf2
    out["gamma_over_absbeta"] = out["gamma"] / max(abs(beta), 1e-12)
    lam, rhl, rfl = ratio_fit(P["Lap div k"], P["S"], tr)
    out["lambda"], out["lambda_r2_holdout"], out["lambda_r2_full"] = lam, rhl, rfl
    out["lambda_over_2beta2"] = lam / max(2 * beta ** 2, 1e-12)
    M = np.column_stack([P["S"], P["J"]])
    cj, rj = ols(M, P["Lap div k"], tr)
    out["lambdaJ"], out["mJ"], out["SJ_r2_holdout"] = float(cj[0]), float(cj[1]), rj
    # binned beta (full fit in the bin) by row and by core distance
    for key, dist, bins in (("row", P["_drow"], ROW_BINS), ("core", P["_dcore"], CORE_BINS)):
        rows = []
        for lo, hi in bins:
            m = (dist >= lo) & (dist < hi)
            if m.sum() < 30:
                rows.append((lo, hi, int(m.sum()), np.nan, np.nan))
                continue
            cb = float(np.dot(w[m], d[m]) / max(np.dot(w[m], w[m]), 1e-300))
            rows.append((lo, hi, int(m.sum()), -cb, r2(d[m], cb * w[m])))
        out["beta_by_" + key] = rows
    # where it fails: weak residual of the self-dual relation by core distance
    rho = d + beta * w
    rms_d = float(np.sqrt(np.mean(d ** 2))) + 1e-300
    J2 = P["J"] ** 2
    rows = []
    for lo, hi in [(0.5, 1.0)] + CORE_BINS[1:]:
        m = (P["_dcore"] >= lo) & (P["_dcore"] < hi)
        if m.sum() < 30:
            rows.append((lo, hi, int(m.sum()), np.nan, np.nan))
            continue
        rows.append((lo, hi, int(m.sum()), float(np.sqrt(np.mean(rho[m] ** 2))) / rms_d,
                     float(J2[m].sum() / max(J2.sum(), 1e-300))))
    out["residual_by_core"] = rows
    return out


# ---------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------

def fmt_rel(t, comb, c):
    return "%s = %s" % (t, " + ".join("%+.3g %s" % (ci, n) for ci, n in zip(c, comb)))


def report_block(label, P, res, one, head, F, lines, rows1, rows2, out, args, zero=()):
    lines.append("")
    lines.append("=== %s   (%d query centres; holdout = centres with x above the median)" % (label, len(P["_x"])))
    lines.append("  STAGE 1, free search: best support per target (holdout R2)")
    lines.append("    %-11s | %-44s | %s" % ("target", "best single partner", "best pair"))
    for t in TARGETS:
        if t in zero:
            lines.append("    %-11s | vanishes identically on this domain (rms below 1e-6 of the largest target)%s" %
                         (t, ": k is a gradient" if t == "curl k" else ": the phase surface is developable" if t == "J" else ""))
            continue
        b1, b2 = res[t][1][0], res[t].get(2, [None])[0]
        s1 = "%-30s R2 %.3f" % (fmt_rel(t, b1[1], b1[2]).split(" = ")[1], b1[0])
        s2 = ("%s  R2 %.3f" % (fmt_rel(t, b2[1], b2[2]).split(" = ")[1], b2[0])) if b2 else ""
        lines.append("    %-11s | %-44s | %s" % (t, s1, s2))
        rows1.append(dict(field=label, target=t, partner=b1[1][0], coef=float(b1[2][0]), r2_holdout=b1[0],
                          pair=" ; ".join(b2[1]) if b2 else "", pair_coefs=" ; ".join("%.4g" % v for v in b2[2]) if b2 else "",
                          pair_r2_holdout=b2[0] if b2 else np.nan))
    lines.append("  HEADLINE (lowest-order size-1 relation within %.2g of the best): %s ~ %s,"
                 " coefficient %.4g, holdout R2 %.3f" % (args.tol, head[1], head[2], head[3], head[0]))
    sd = (set((head[1], head[2])) == {"div k", "w"})
    lines.append("  -> %s" % ("the search proposes div k proportional to w: the self-dual form, unprompted"
                             if sd else "the search does NOT put div k ~ w first (see the table for where it ranks)"))
    rk = sorted(one, key=lambda o: -o[0])
    pos = [i for i, o in enumerate(rk) if set((o[1], o[2])) == {"div k", "w"}]
    lines.append("     rank of div k ~ w among the targets' best single relations: %s of %d (R2 %.3f)" %
                 ((pos[0] + 1) if pos else "absent", len(rk),
                  rk[pos[0]][0] if pos else np.nan))
    lines.append("  STAGE 2, forced follow-ups")
    lines.append("    (a) div k = -beta w:      beta %+.3f   holdout R2 %.3f (full %.3f)" %
                 (F["beta"], F["beta_r2_holdout"], F["beta_r2_full"]))
    lines.append("        by row distance (L):  " + "; ".join(
        "%.1f-%.1f: %s (R2 %s, n %d)" % (lo, hi, ("%+.3f" % b) if np.isfinite(b) else "n/a",
                                         ("%.2f" % q) if np.isfinite(q) else "n/a", n)
        for lo, hi, n, b, q in F["beta_by_row"]))
    lines.append("        by core distance (L): " + "; ".join(
        "%.1f-%s: %s (R2 %s, n %d)" % (lo, ("%.0f" % hi) if np.isfinite(hi) else "inf",
                                       ("%+.3f" % b) if np.isfinite(b) else "n/a",
                                       ("%.2f" % q) if np.isfinite(q) else "n/a", n)
        for lo, hi, n, b, q in F["beta_by_core"]))
    lines.append("    (b) (div k)^2 = gamma^2 w^2: gamma %.3f   holdout R2 %.3f;  gamma / |beta| = %.3f"
                 "  (1 = bending and stretching densities in a fixed ratio: self-dual)" %
                 (F["gamma"], F["gamma_r2_holdout"], F["gamma_over_absbeta"]))
    lines.append("    (c) Lap div k = lambda S: lambda %+.3f   holdout R2 %.3f;  lambda / 2 beta^2 = %+.3f"
                 "  (1 on a self-dual field with J = 0);  with J: lambda_J %+.3f, m %+.3f, holdout R2 %.3f" %
                 (F["lambda"], F["lambda_r2_holdout"], F["lambda_over_2beta2"], F["lambdaJ"], F["mJ"], F["SJ_r2_holdout"]))
    if any(np.isfinite(r[3]) for r in F["residual_by_core"]):
        lines.append("    (d) rms(div k + beta w) / rms(div k), and share of the J signal, by core distance (L): " + "; ".join(
            "%.1f-%s: %s, J %s" % (lo, ("%.0f" % hi) if np.isfinite(hi) else "inf",
                                   ("%.3f" % q) if np.isfinite(q) else "n/a", ("%.2f" % s) if np.isfinite(s) else "n/a")
            for lo, hi, n, q, s in F["residual_by_core"]))
    ok_a = F["beta_r2_holdout"] >= args.r2_ok
    ok_b = abs(F["gamma_over_absbeta"] - 1.0) <= args.equi_tol
    lines.append("  READING: %s; (a) %s; (b) %s; (c) lambda %s 2 beta^2 in sign%s." % (
        "headline is the self-dual form" if sd else "headline is NOT the self-dual form",
        "holds" if ok_a else "FLAG: weak (holdout R2 below %.2f)" % args.r2_ok,
        "equipartition holds" if ok_b else "FLAG: equipartition off by more than %.0f%%" % (100 * args.equi_tol),
        "matches" if np.sign(F["lambda"]) == np.sign(2 * F["beta"] ** 2) else "OPPOSES",
        "" if F["lambda_r2_holdout"] > 0.5 else " (FLAG: lambda fit holdout R2 %.2f)" % F["lambda_r2_holdout"]))
    rows2.append(dict(field=label, headline="%s ~ %s" % (head[1], head[2]), headline_r2=head[0],
                      **{k: v for k, v in F.items() if not isinstance(v, list)}))
    figure(label, P, one, F, out / ("stage1_%s.png" % "".join(ch if ch.isalnum() else "_" for ch in label)))


def figure(label, P, one, F, path):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    rk = sorted(one, key=lambda o: -o[0])
    ax[0].barh(range(len(rk)), [max(o[0], -0.2) for o in rk],
               color=["C1" if set((o[1], o[2])) == {"div k", "w"} else "0.6" for o in rk])
    ax[0].set_yticks(range(len(rk)))
    ax[0].set_yticklabels(["%s ~ %s" % (o[1], o[2]) for o in rk], fontsize=8)
    ax[0].invert_yaxis(); ax[0].set_xlim(-0.2, 1.0); ax[0].set_xlabel("holdout R2, best single partner")
    ax[0].set_title("stage 1: free search (orange: div k ~ w)", fontsize=9)
    dc = np.clip(P["_dcore"], 0, 4) if np.isfinite(P["_dcore"]).any() else P["_drow"]
    for a_, (yk, xk, c, lab) in zip(ax[1:], (("div k", "w", -F["beta"], "div k = %+.2f w" % (-F["beta"])),
                                             ("(div k)^2", "w^2", F["gamma"] ** 2, "(div k)^2 = %.2f w^2" % F["gamma"] ** 2))):
        sc = a_.scatter(P[xk], P[yk], c=dc, s=2, cmap="viridis")
        xs = np.linspace(P[xk].min(), P[xk].max(), 2)
        a_.plot(xs, c * xs, "r-", lw=1, label=lab)
        a_.set_xlabel("weak " + xk); a_.set_ylabel("weak " + yk); a_.legend(fontsize=7)
        fig.colorbar(sc, ax=a_, label=("distance to core (L)" if np.isfinite(P["_dcore"]).any() else "distance to line (L)"))
    ax[1].set_title("(a) holdout R2 %.3f" % F["beta_r2_holdout"], fontsize=9)
    ax[2].set_title("(b) gamma/|beta| = %.3f" % F["gamma_over_absbeta"], fontsize=9)
    fig.suptitle(label, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ---------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------

def synthetic(args):
    """(label stem, list of (route, samples)) for knee | nonsd | pn"""
    mu = args.mu
    k1 = float(np.sqrt(1 - mu ** 2))
    L = np.pi / k1                                       # dislocation spacing, the distance unit
    dy = args.Ly / (args.ny - 1)
    y = np.linspace(-args.Ly / 2, args.Ly / 2, args.ny)
    xd = None
    if args.mode == "pn":
        nd = int(args.n_disl)
        Lx = (nd + 2 * args.pn_margin) * L
        nx = int(round(Lx / dy))
        x = (np.arange(nx) - nx // 2) * dy
        X, Y = np.meshgrid(x, y)
        xd = (np.arange(nd) - (nd - 1) / 2.0) * L
        th, _ = pismen_on_sh_grid(X, Y, xd)
        c, s = np.cos(th), np.sin(th)                   # cut-free gradient (as the ledger)
        k1e = c * np.gradient(s, dy, axis=1) - s * np.gradient(c, dy, axis=1)
        k2e = c * np.gradient(s, dy, axis=0) - s * np.gradient(c, dy, axis=0)
        rr = np.min(np.hypot(X[..., None] - xd[None, None, :], Y[..., None]), axis=-1)
        A = np.sqrt(4 * args.R / 3) * np.tanh(rr / args.pn_core)
        u = A * np.cos(th)
        d_core = rr / L
        side = "upper"
        stem = "pn_mu%.2f_nd%d" % (mu, nd)
    else:
        nx = int(round(args.n_xper * 2 * np.pi / k1 / dy))
        x = (np.arange(nx) - nx // 2) * dy
        X, Y = np.meshgrid(x, y)
        b = args.beta
        if args.mode == "knee":
            th = k1 * X + np.log(np.cosh(b * mu * Y)) / b
            k2e = mu * np.tanh(b * mu * Y)
            stem = "knee_beta%.2f_mu%.2f" % (b, mu)
        else:
            a_ = b * np.sqrt(np.pi) / 2 * mu             # same core slope b mu^2 as the tanh knee
            th = k1 * X + mu * (Y * special.erf(a_ * Y) + np.exp(-(a_ * Y) ** 2) / (a_ * np.sqrt(np.pi)))
            k2e = mu * special.erf(a_ * Y)
            stem = "nonsd_erf_slope%.2f_mu%.2f" % (b, mu)
        k1e = np.full_like(th, k1)
        u = np.cos(th)
        d_core = None
        side = args.side
    out = []
    if args.route in ("exact", "both"):
        out.append(("exact", [dict(k1=k1e, k2=k2e, x=x, y=y, L=L, y_row=0.0, d_core=d_core, side=side)]))
    if args.route in ("estimator", "both"):
        try:
            _, estimate, _, _, t1c = _estimator()
        except ImportError as exc:
            print("ESTIMATOR ROUTE SKIPPED: the frozen estimator chain is not importable here (%s)" % exc)
            return stem, out
        taper = t1c.edge_taper(x, y, frac=args.taper_frac)
        e = estimate(u, len(x) * dy, len(y) * dy, args.sigma_dir, None, taper)
        K1, K2 = e["k1"], e["k2"]
        use_k1 = abs(np.mean(K1)) >= abs(np.mean(K2))
        sg = np.sign(np.mean(K1 if use_k1 else K2)) or 1.0
        out.append(("estimator", [dict(k1=K1 * sg, k2=K2 * sg, x=x, y=y, L=L, y_row=0.0, d_core=d_core, side=side)]))
    return stem, out


def data_samples(path, args, lines):
    load_npz, estimate, find_cores, dist_to_cores, t1c = _estimator()
    D = load_npz(path)
    u, x, y, tt, md = D["u"], D["x"], D["y"], D["t"], D["md"]
    mu = float(md.get("mu", args.mu)); k1 = np.sqrt(1 - mu ** 2); L = np.pi / k1
    if args.frames == "last":
        sel = np.zeros(len(tt), bool); sel[-1] = True
    else:
        sel = (tt >= args.t0) & (tt <= args.t1)
    idx = np.where(sel)[0]
    ny, nx, _ = u.shape
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    Lx, Ly = nx * dx, ny * dy
    X, Y = np.meshgrid(x, y)
    win = (np.abs(X - x.mean()) < 0.4 * Lx) & (np.abs(Y - y.mean()) < 0.4 * Ly)
    taper = t1c.edge_taper(x, y, frac=args.taper_frac)
    y_row = float(y[int(md.get("gb_row_in_crop", ny // 2))])
    samples, ncores, n_prev = [], [], None
    for i in idx:
        e = estimate(u[..., i], Lx, Ly, args.sigma_dir, n_prev, taper)
        n_prev = e["n"]
        K1, K2 = e["k1"], e["k2"]
        use_k1 = abs(np.mean(K1)) >= abs(np.mean(K2))        # ledger v1.5: one scalar sign per frame
        sg = np.sign(np.mean(K1 if use_k1 else K2)) or 1.0
        xc, yc = find_cores(e["A"], x, y, win, args.core_frac, merge_dist=0.5 * L)
        ncores.append(len(xc))
        d_core = dist_to_cores(X, Y, xc, yc) / L if len(xc) else None
        samples.append(dict(k1=K1 * sg, k2=K2 * sg, x=x, y=y, L=L, y_row=y_row, d_core=d_core, side=args.side))
    lines.append("data %s: mu %.3f (L = %.2f), frames %s (t %.2f..%.2f), cores per frame %s" %
                 (Path(path).stem, mu, L, list(idx), tt[idx[0]], tt[idx[-1]], ncores))
    return Path(path).stem, [("estimator", samples)]


# ---------------------------------------------------------------------

def main(args):
    t0 = time.time()
    lines = ["self_dual_discovery v1.0   (weak helpers: %s; PN field: %s)" % (_HELPERS, _PN),
             "library %s ; targets %s ; supports up to size %d ; radius %.2g L, stride %d, rim %d,"
             " domain %s (band < %.2g L), core discs %.2g L, side %s" %
             (LIB, TARGETS, args.max_size, args.radius, args.stride, args.rim, args.domain, args.band_to,
              args.core_excl, args.side)]
    if args.npz:
        stem, routes = data_samples(args.npz, args, lines)
    else:
        stem, routes = synthetic(args)
        lines.append("synthetic %s, mu %.2f, grid %d rows, Ly %.0f, routes %s%s" %
                     (args.mode, args.mu, args.ny, args.Ly, [r for r, _ in routes],
                      "; side upper forced (the PN relation is odd in y)" if args.mode == "pn" else ""))
        if args.route in ("estimator", "both") and "estimator" not in [r for r, _ in routes]:
            lines.append("FLAG: estimator route requested but SKIPPED (mg_zigzag_theta_t / t1_common not importable"
                         " from this location); exact route only")
    stamp = "rad%.2g_%s_ce%.2g_%s" % (args.radius, args.domain, args.core_excl,
                                      args.frames if args.npz else args.route)
    out = Path(args.outdir) / stem / stamp
    out.mkdir(parents=True, exist_ok=True)
    rows1, rows2, scan_rows = [], [], []
    radii = [float(v) for v in str(args.radii).split(",") if v.strip()]
    for route, frames in routes:
        S = lambda r: pool([sample_frame(f["k1"], f["k2"], f["x"], f["y"], f["L"], f["y_row"], f["d_core"],
                                         args, f["side"], r) for f in frames])
        P = S(args.radius)
        if len(P["_x"]) < 100:
            lines.append("%s %s: only %d centres; nothing fitted" % (stem, route, len(P["_x"])))
            continue
        res, one, head, zero = free_search(P, args)
        F = forced_checks(P, args)
        report_block("%s | %s route" % (stem, route), P, res, one, head, F, lines, rows1, rows2, out, args, zero)
        # ---- scale test over test-function radii
        lines.append("  SCALE TEST: stage 1 and (a)-(c) at each test-function radius. A pointwise relation reads the"
                     " same coefficient at every radius; two co-localised fields averaged by a kernel wider than"
                     " both look proportional at large radius and drift as it shrinks.")
        lines.append("    %-6s %-6s %-36s %-18s %-20s %s" % ("radius", "n", "headline (coef, holdout R2)",
                                                           "(a) beta (R2)", "(b) gamma/|beta| (R2)", "(c) lambda/2beta^2"))
        betas, persistent, pw_const = [], None, None
        for r in radii:
            Pr = S(r)
            if len(Pr["_x"]) < 100:
                lines.append("    %-6.2f only %d centres" % (r, len(Pr["_x"])))
                continue
            _, _, hr, _ = free_search(Pr, args)
            Fr = forced_checks(Pr, args)
            betas.append(Fr["beta"])
            lines.append("    %-6.2f %-6d %-36s %+.3f (%.3f)    %.3f (%.3f)        %+.2f" % (
                r, len(Pr["_x"]), "%s ~ %s (%+.3g, %.3f)" % (hr[1], hr[2], hr[3], hr[0]),
                Fr["beta"], Fr["beta_r2_holdout"], Fr["gamma_over_absbeta"], Fr["gamma_r2_holdout"],
                Fr["lambda_over_2beta2"]))
            scan_rows.append(dict(field="%s | %s" % (stem, route), radius=r, n=len(Pr["_x"]),
                                  headline="%s ~ %s" % (hr[1], hr[2]), headline_coef=hr[3], headline_r2=hr[0],
                                  beta=Fr["beta"], beta_r2=Fr["beta_r2_holdout"], gamma_over_absbeta=Fr["gamma_over_absbeta"],
                                  gamma_r2=Fr["gamma_r2_holdout"], lambda_over_2beta2=Fr["lambda_over_2beta2"]))
        if len(betas) > 1:
            spread = (max(betas) - min(betas)) / max(np.mean(np.abs(betas)), 1e-12)
            persistent = spread <= args.scale_tol
            lines.append("    beta across radii: %s ; spread (max - min)/mean|beta| = %.2f -> %s" % (
                ", ".join("%+.3f" % v for v in betas), spread,
                "persistent" if spread <= args.scale_tol else "FLAG: drifts with the radius (not a pointwise relation at these scales)"))
        # ---- pointwise reading, last frame
        pw = pointwise_check(frames[-1], args)
        if pw:
            lines.append("  (e) POINTWISE, no test function (last frame, |w| > %.2g): median -div k / w and |div k / w| by distance (L): " % args.w_min
                         + "; ".join("%s %.2f-%s: %+.3f, %.3f (n %d)" % (k, lo, ("%.2f" % hi) if np.isfinite(hi) else "inf", m1, m2, n)
                                     for k, lo, hi, n, m1, m2 in pw))
            rv = [m1 for k, lo, hi, n, m1, m2 in pw if k == "row"]
            if len(rv) > 1:
                sp = (max(rv) - min(rv)) / max(np.mean(np.abs(rv)), 1e-12)
                pw_const = sp <= args.scale_tol
                lines.append("      pointwise -div k / w across row-distance bins: spread (max - min)/mean = %.2f -> %s" % (
                    sp, "constant: a pointwise relation" if sp <= args.scale_tol else
                    "FLAG: varies across the profile (the weak proportionality is not pointwise)"))
            if route == "estimator":
                lines.append("      (estimator route: pointwise derivatives carry the estimator's near-core and crease"
                             " oscillation; read the trend across bins, not one bin)")
        sd_head = set((head[1], head[2])) == {"div k", "w"}
        weak_ok = F["beta_r2_holdout"] >= args.r2_ok and abs(F["gamma_over_absbeta"] - 1) <= args.equi_tol
        yn = lambda v: "n/a" if v is None else ("yes" if v else "NO")
        lines.append("  SUMMARY: free search puts div k ~ w first: %s | weak (a) and (b) hold at radius %.2g: %s |"
                     " persistent across radii: %s | constant pointwise: %s" %
                     (yn(sd_head), args.radius, yn(weak_ok), yn(persistent), yn(pw_const)))
        lines.append("           (all four yes: self-duality found and confirmed; weak yes but pointwise NO: a weak"
                     " proportionality of co-localised fields, not evidence of self-duality; see the explanation file)")
    for name, rows in (("stage1.csv", rows1), ("stage2.csv", rows2), ("scale_test.csv", scan_rows)):
        if rows:
            keys = list(rows[0])
            with open(out / name, "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
    with open(out / "config.json", "w") as fh:
        json.dump({k: getattr(args, k) for k in vars(args)}, fh, indent=2, default=str)
    lines.append("")
    lines.append("elapsed %.1f s" % (time.time() - t0))
    rep = "\n".join(lines)
    (out / "report.txt").write_text(rep)
    print(rep); print("outputs ->", out)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    mode = "knee"                 # knee | nonsd | pn  (ignored when --npz is given)
    npz = ""
    outdir = str(_HERE / "results" / "1006_self_dual_discovery")
    route = "both"                # synthetic modes: exact | estimator | both
    frames = "last"               # data: last | window (t0..t1)
    t0 = -1.0
    t1 = 1e9
    mu = 0.85
    beta = 1.3                    # knee: planted self-dual coefficient; nonsd: core slope matched to it
    R = 0.5
    Ly = 60.0
    ny = 300
    n_xper = 6                    # knee / nonsd: periods 2 pi/k1 in x
    n_disl = 10                   # pn: dislocations in the row
    pn_margin = 1.5               # pn: spacings of margin at each end of the row
    pn_core = 1.5                 # pn estimator route: smooth core dip radius (x units), as ledger v1.5
    sigma_dir = 0.4
    taper_frac = 0.06
    core_frac = 0.70
    core_excl = 0.5               # spacings
    radius = 1.5                  # test-function half width, spacings
    poly_p = 6
    eta = 0.0                     # inner derivative: np.gradient
    stride = 4
    rim = 4
    rim_frac = 0.10
    domain = "band"               # band | shell | all
    band_to = 2.0
    far_from = 4.0
    side = "both"                 # both | upper | lower (pn forces upper)
    max_size = 2
    tol = 0.02                    # headline: within tol of the best size-1 holdout R2
    r2_ok = 0.8                   # reading: (a) needs holdout R2 at least this
    equi_tol = 0.15               # reading: |gamma/|beta| - 1| at most this
    radii = "0.25,0.5,1.0,1.5"    # scale test: test-function radii in spacings
    scale_tol = 0.10              # scale test: beta spread across radii at most this (relative)
    w_min = 0.02                  # pointwise reading: only where |w| exceeds this
    pw_row_edges = "0,0.05,0.1,0.15,0.2,0.3,0.5,1,2"   # pointwise reading: row-distance bin edges (L)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in dir(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    sys.exit(main(p.parse_args()))
