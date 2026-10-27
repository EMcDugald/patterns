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
v1.3.3 (12 Oct 2026) --core_def charge: discs about the clusters (within
      0.5 L) of the director disclinations of the frozen t1c.detect_defects
      at sigma_det 0.3 (the early index of the two-onset record), within the
      band and the inner 80 percent window; stamp "_cq". For comparing the
      two topological definitions empirically; the charge is present also
      on the 0.65 to 0.70 knees (sum|q| 48 at 0.70), so those get discs too.
v1.3.2 (12 Oct 2026) --core_def circulation | dips (data files). With
      circulation the core discs (and every core-distance bin) use only
      the amplitude dips whose 0.5 L loop circulation of the extracted k
      reads >= --wind_hi turns (0.9). Measured on the 28 last frames of the
      fine sweep: the 20 dips per crop at 0.71 to 0.81 (pairs 0.30 to 0.42 L
      off the line) read 0.00 turns, as do the 28 dips of the 0.79 and 0.81
      files of record at t = 25; they are a split amplitude trough of an
      unwound knee. With discs on them the band lost a quarter of its
      centres, the |w|-weighted mask fell from 74 to 79 percent to 31 to 34
      percent of the weight and the placebo ratio from 6.3 to 6.9 to 1.4 to
      2.0. Default stays "dips" so that every earlier stamp reproduces; the
      stamp gains "_cc" under circulation. Data mode only; synthetic modes
      are unchanged.
v1.3.1 (11 Oct 2026) PLACEBO for the plateau: the same plateau search on
      the pair (div k, w shifted ACROSS the row by --placebo_shift spacings,
      default 0.5 L), which keeps w's statistics and breaks its
      correspondence with div k. A shift along the row was tried first and
      captured the same share as the real mask on every file (the weak
      fields at 1.5 L are nearly functions of row distance alone, so that
      shift is close to a symmetry); the shift across the row is the null
      that tests the claim. The report prints the placebo's best captured
      |w| weight and area beside the real ones; the real mask is meaningful
      only where it captures several times more. Answers the objection
      that a tolerance window about the mode of any ratio selects some
      region.
v1.3  (11 Oct 2026) ANISOTROPIC test function (--radius_x, --radius_y in
      spacings; --ry_periods p sets the half width across the row so that
      the support spans p periods of the far-field 2theta oscillation
      across it, ry = p/2 x (pi/mu)/L = p/2 x k1/mu spacings, which is the
      minimal averaging the weak form needs there). The bump stays the
      polynomial of kernel_d3 on each axis with its own half width; the
      fully weak columns pick up the right axis automatically (as in
      wsindy_theta_t v1.9). Report prints the 2theta periods spanned along
      and across the row with a WARNING below one. The scale test keeps
      the isotropic radii of --radii. Motivation (Ed, 11 Oct): at mu 0.95
      the 1.5 L isotropic bump spans 3 periods along the row and 4.6
      across it, far wider across than needed where the structure (crease,
      collar) is narrow; a rectangular bump averages the carrier and not
      the structure.
v1.2.2 (11 Oct 2026) (0) second figure row in ratio_map_*.png: the weak
      target div k, the model -beta w with the whole-band beta, the model
      with the collar beta, and the residual of the collar model, as maps
      on the band. (1) fit_maps_<label>.png: the weak target div k, the
      prediction -beta w with the whole-band beta and with the collar beta,
      and the two residual maps, side by side on the band (Ed: a fit should
      look like its target). (2) The stage-1 figure's left title names the
      target's best partner and the top-ranked relation separately (the
      0.83 figure read as a contradiction when the two differed). (3) The
      ratio-map panels follow --band_to, so --band_to 3 (a 6 L band) draws
      the wider domain.
v1.2.1 (11 Oct 2026) on the mask: the free search (stage 1 headline and
      family) and A2 restricted to the centres inside the mask; and a MASK
      LADDER: the mask widened in steps (tolerance 10, 20, 40, 80 percent,
      then the whole band), relation A refitted at each step with the
      support fixed, so the coefficient and R2 are read against the domain
      size (Ed's seeding idea: start where the ratio is constant and grow
      the domain, watching where the fit degrades).
v1.2  (11 Oct 2026) RATIO MAP and constant-ratio mask (--ratio_map 1;
      Ed's litmus test, A4 of the handoff). On the last frame the local
      ratio r = -div k / w is mapped twice: weak (the full 2D weak
      columns of the same bump, consistent with every fit) and pointwise
      (np.gradient). The plateau is the mode of the weak ratio over the
      kept domain (|w| above --ratio_wmin x median |w|): the ratio value whose
      +-ratio_tol window captures the most |w| weight (a scan); the mask is
      |r - r_plateau| <= --ratio_tol x |r_plateau|. Reported: the plateau,
      the mask's share of the kept domain, where it sits (row and core
      distance quartiles), relation A fitted on the centres inside and
      outside the mask, and a figure ratio_map_<label>.png (pattern with
      band and discs; weak ratio with the mask outline; pointwise ratio;
      histogram with the plateau and the mask range). The stage-1 figure
      gains the rank of div k ~ w in the left panel whether or not it tops
      a ranking.
v1.1  (10 Oct 2026) workstreams A1/A2 of the research thread; every v1.0
      block and number is unchanged, the additions print after them.
      (1) FAMILY: the size-1 relations within --tol of the best, by
          derivative order, with the tie margin of the headline. Why: on
          the planted knee through the estimator the headline of v1.0 is
          w div k ~ w^3 while div k ~ w sits at the same R2 1.000 (sandbox,
          10 Oct); on a knee w is a bump of nearly fixed depth -mu^2, so
          w, w^2, w^3 are proportional in the weak form and the headline
          among that family is a rounding tie, not a measurement.
      (2) A2, fixed-target STLSQ (--a2 1): div k as the target over the
          library (constant kept, single terms and {1, w} forced into the
          pool, sparsest within --rel_tol_a2 of the best holdout R2), with
          and without the columns that contain the target (w div k,
          (div k)^2); the same with Lap div k as the target and J offered
          (relation B). Uses fit_target from self_dual_sindy.py.
      (3) SIGN diagnostic (--signs 1): fraction of centres where the weak
          div k and w have the self-dual (opposite) sign, by row and core
          bins, and relation A fitted signed, absolute (|div k| on |w|) and
          squared ((div k)^2 on w^2): a squared fit that beats the signed
          one only where the sign fraction is low is a magnitude relation
          with sign reversals, not the self-dual form.
      (4) DECOMPOSITION (synthetic modes with both routes, --decomp 1): the
          estimator's error on the stationary columns split into a gain
          (explained by the exact columns) and a residual, as
          c0_dilution_control does for the rate law, and beta refitted
          with each piece alone.
      (5) BETA CURVE (--beta_curve "0.8,1.0,...", data and knee modes):
          planted tanh knees at the file's mu through the estimator at the
          same radius; beta_read against beta_planted, and the file's beta
          placed on the curve by interpolation as a model-corrected value
          in its own line (never beside the raw one in a column).
      (6) summary_row.json per run for self_dual_summary_table.py (the
          across-mu table). Output root results/1010_self_dual_discovery_v11.
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

def radii_xy(args, Lsp, radius=None):
    """half widths (spacings) along the row (x) and across it (y): an
    explicit radius (the scale test) is isotropic; otherwise radius_x /
    radius_y when positive, ry_periods when positive, else --radius"""
    if radius is not None:
        return float(radius), float(radius)
    rx = float(getattr(args, "radius_x", -1.0)); ry = float(getattr(args, "radius_y", -1.0))
    pp = float(getattr(args, "ry_periods", 0.0))
    mu = float(args.mu); k1 = float(np.sqrt(max(1.0 - mu ** 2, 1e-12)))
    if pp > 0:
        ry = 0.5 * pp * k1 / mu                      # p periods of pi/mu across, in spacings L = pi/k1
    return (rx if rx > 0 else float(args.radius)), (ry if ry > 0 else float(args.radius))


def periods_xy(args, Lsp, rx, ry):
    mu = float(args.mu); k1 = float(np.sqrt(max(1.0 - mu ** 2, 1e-12)))
    return 2.0 * rx, 2.0 * ry * Lsp / (np.pi / mu)      # periods pi/k1 = L along, pi/mu across


def weak_columns(k1, k2, x, y, Lsp, args, radius=None):
    """full 2D weak fields of every library column (and S for stage 2)"""
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    rx, ry = radii_xy(args, Lsp, radius)
    fx, fx1, fx2, fx3, mx = kernel_d3(rx * Lsp, dx, args.poly_p)
    fy, fy1, fy2, fy3, my = kernel_d3(ry * Lsp, dy, args.poly_p)
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
    pos = [i for i, o in enumerate(rk) if set((o[1], o[2])) == {"div k", "w"}]
    sdr = [o for o in one if o[1] == "div k"]
    ax[0].set_title("stage 1: each target's best single partner (orange: div k ~ w)\n"
                    "top-ranked: %s ~ %s (R2 %.3f); target div k's own best partner: %s (R2 %.3f)" %
                    (rk[0][1], rk[0][2], rk[0][0], sdr[0][2] if sdr else "?", sdr[0][0] if sdr else np.nan), fontsize=7)
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


def director_charge_points(u2d, x, y, y_row, taper, Lsp, sigma_det=0.3, band_L=2.0, merge_L=0.5, win=None):
    """v1.2 / v1.3.3: director disclinations of the frozen _t1c().detect_defects
    (Riesz director at sigma_det, plaquette winding of its angle), the
    index of the two-onset record; plaquettes with q != 0 within band_L
    spacings of the line, clustered within merge_L spacings. Returns the
    cluster positions (x, absolute y), the |q| summed per cluster, and
    the band's sum |q|."""
    ny, nx = u2d.shape
    dx = float(x[1] - x[0])
    det = _t1c().detect_defects(u2d * taper, x, y - y_row, nx * dx, ny * dx,
                             sigma_det=sigma_det, band_y=band_L * Lsp)
    ch = det["charges"]
    if win is not None and len(ch):
        Lx = nx * dx; Ly = ny * dx
        ch = [c for c in ch if abs(c[0] - x.mean()) < 0.4 * Lx and abs(c[1] + y_row - y.mean()) < 0.4 * Ly]
    if not ch:
        return np.zeros(0), np.zeros(0), np.zeros(0), int(det["sum_abs"])
    px = np.array([c[0] for c in ch]); py = np.array([c[1] for c in ch]) + y_row
    pq = np.abs(np.array([c[2] for c in ch], float))
    order = np.argsort(px); px, py, pq = px[order], py[order], pq[order]
    used = np.zeros(len(px), bool); cx, cy, cq = [], [], []
    for i in range(len(px)):
        if used[i]:
            continue
        grp = (~used) & (np.hypot(px - px[i], py - py[i]) < merge_L * Lsp)
        used |= grp
        cx.append(px[grp].mean()); cy.append(py[grp].mean()); cq.append(pq[grp].sum())
    return np.array(cx), np.array(cy), np.array(cq), int(det["sum_abs"])


def _t1c():
    return _estimator()[4]


def _loop_circ(k1, k2, x, y, xc, yc, r):
    """v1.3.2: circulation of the extracted k on a circle of radius r about
    (xc, yc), in turns, with the director sign carried by continuity
    (mg_zigzag_theta_t v1.5's validated loop)"""
    dx = x[1] - x[0]; dy = y[1] - y[0]
    ang = np.linspace(0, 2 * np.pi, 181)
    ix = np.clip(np.round((xc + r * np.cos(ang) - x[0]) / dx).astype(int), 0, len(x) - 1)
    iy = np.clip(np.round((yc + r * np.sin(ang) - y[0]) / dy).astype(int), 0, len(y) - 1)
    kx, ky = k1[iy, ix], k2[iy, ix]
    sg = np.ones_like(ang)
    for q in range(1, len(ang)):
        sg[q] = sg[q - 1] if kx[q] * kx[q - 1] + ky[q] * ky[q - 1] >= 0 else -sg[q - 1]
    f = (kx * -np.sin(ang) + ky * np.cos(ang)) * sg
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(ang)) * r / (2 * np.pi))


def data_samples(path, args, lines):
    load_npz, estimate, find_cores, dist_to_cores, t1c = _estimator()
    D = load_npz(path)
    u, x, y, tt, md = D["u"], D["x"], D["y"], D["t"], D["md"]
    mu = float(md.get("mu", args.mu)); k1 = np.sqrt(1 - mu ** 2); L = np.pi / k1
    args.mu = mu                                                    # v1.1: the file's mu drives the beta curve
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
        ndip = len(xc)
        if str(getattr(args, "core_def", "dips")) == "circulation" and len(xc):     # v1.3.2
            circ = np.array([abs(_loop_circ(K1, K2, x, y, a, b, 0.5 * L)) for a, b in zip(xc, yc)])
            xc, yc = xc[circ >= args.wind_hi], yc[circ >= args.wind_hi]
            lines.append("  core_def circulation (v1.3.2): %d amplitude dips, %d wound (>= %.2g turns at 0.5 L); discs about the wound ones only" % (ndip, len(xc), args.wind_hi))
        elif str(getattr(args, "core_def", "dips")) == "charge":                     # v1.3.3
            xc, yc, qq, qsum = director_charge_points(u[..., i], x, y, y_row, taper, L, args.sigma_det_charge,
                                                      max(args.band_to, 2.0), 0.5, win=win)
            lines.append("  core_def charge (v1.3.3): %d amplitude dips; director charge sum|q| %d in %d clusters (sigma_det %.2g); discs about the clusters" % (ndip, qsum, len(xc), args.sigma_det_charge))
        ncores.append(len(xc))
        d_core = dist_to_cores(X, Y, xc, yc) / L if len(xc) else None
        samples.append(dict(k1=K1 * sg, k2=K2 * sg, x=x, y=y, L=L, y_row=y_row, d_core=d_core, side=args.side))
    lines.append("data %s: mu %.3f (L = %.2f), frames %s (t %.2f..%.2f), cores per frame %s" %
                 (Path(path).stem, mu, L, list(idx), tt[idx[0]], tt[idx[-1]], ncores))
    return Path(path).stem, [("estimator", samples)]


# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# v1.1 additions (A1 / A2 of the research thread)
# ---------------------------------------------------------------------

def _fit_args(rel_tol):
    class A:
        thresholds = tuple(np.geomspace(1e-4, 1.0, 25))
        rel_tol_pareto = rel_tol
        n_boot = 20
        boot_frac = 0.5
    return A()


def v11_family(one, head, args, lines):
    """the size-1 relations the data cannot tell apart (within --tol of the
    best holdout R2), by derivative order; the headline's tie margin"""
    top = max(o[0] for o in one)
    adm = [o for o in one if o[0] >= top - args.tol and o[2] != "1"]
    adm.sort(key=lambda o: (ORDER[o[1]] + ORDER[o[2]], -o[0]))
    lines.append("  FAMILY (v1.1): size-1 relations within %.2g of the best holdout R2 %.4f, by derivative order:" % (args.tol, top))
    for o in adm:
        lines.append("    %-11s ~ %-10s coef %+.4g  R2 %.4f  (order %d%s)" %
                     (o[1], o[2], o[3], o[0], ORDER[o[1]] + ORDER[o[2]],
                      ", the self-dual form" if set((o[1], o[2])) == {"div k", "w"} else ""))
    sd = [o for o in adm if set((o[1], o[2])) == {"div k", "w"}]
    same_order = [o for o in adm if ORDER[o[1]] + ORDER[o[2]] == ORDER[head[1]] + ORDER[head[2]]]
    margin = (head[0] - min(o[0] for o in same_order)) if len(same_order) > 1 else np.nan
    if sd:
        lines.append("    div k ~ w IS in the family (R2 %.4f, %.4f below the best); the headline %s ~ %s is %s" %
                     (sd[0][0], top - sd[0][0], head[1], head[2],
                      "it" if set((head[1], head[2])) == {"div k", "w"} else
                      ("a tie-break among %d relations of the same order (margin %.4f in R2): read the family, not the headline"
                       % (len(same_order), margin))))
    else:
        lines.append("    div k ~ w is NOT in the family: the self-dual form is rejected at this tolerance")
    return dict(family=[dict(target=o[1], partner=o[2], coef=o[3], r2=o[0], order=ORDER[o[1]] + ORDER[o[2]]) for o in adm],
                sd_in_family=bool(sd), sd_r2=(sd[0][0] if sd else np.nan), best_r2=top, tie_margin=margin,
                n_same_order=len(same_order))


def v11_a2(P, args, lines):
    """A2: fixed-target STLSQ (threshold sweep, single terms, Pareto) for
    div k and for Lap div k, with and without the columns that contain
    the target; {1, w} forced into the pool for div k"""
    from self_dual_sindy import fit_target, r_squared as _r2                   # noqa
    tr = P["_x"] < np.median(P["_x"])
    rng = np.random.default_rng(0)
    out = {}
    specs = [("div k", ["w div k", "(div k)^2"], "relation A"),
             ("Lap div k", [], "relation B (J offered)")]
    for target, contains, label in specs:
        b = P[target]
        for variant, drop in (("full library", []), ("without columns that contain the target", contains)):
            names = [n for n in LIB if n != target and n not in drop and not (target == "Lap div k" and n == "S")]
            Th = np.column_stack([P[n] for n in names])
            keep = np.std(Th, axis=0) > 0; keep[names.index("1")] = True
            names_k = [n for n, k in zip(names, keep) if k]; Th = Th[:, keep]
            iw = [names_k.index("1"), names_k.index("w")]
            c1w = np.zeros(len(names_k)); c1w[iw] = np.linalg.lstsq(Th[tr][:, iw], b[tr], rcond=None)[0]
            r1w = _r2(b[~tr], Th[~tr] @ c1w)
            key = "%s | %s" % (target, variant)
            out[key] = dict(target=target, variant=variant, r2_1w=float(r1w), by_tol={})
            sel_txt = []
            for tol in (float(args.rel_tol_a2), 2.0 * float(args.rel_tol_a2)):
                res = fit_target(names_k, Th, b, _fit_args(tol), rng, tr)
                best = max(sw["r2_test"] for sw in res["sweep"])
                chosen = res["chosen"]
                if r1w >= best - tol * abs(best) and chosen["n_terms"] > 2:
                    supp_names, coefs, r2c = ["1", "w"], c1w[iw], r1w
                else:
                    supp_names = [n for n, k in zip(names_k, chosen["support"]) if k]
                    coefs = chosen["coef"][chosen["support"]]; r2c = chosen["r2_test"]
                ladder = {}
                for sw in res["sweep"]:
                    ladder[sw["n_terms"]] = max(ladder.get(sw["n_terms"], -np.inf), sw["r2_test"])
                out[key]["by_tol"]["%.3g" % tol] = dict(support=supp_names, coefs=[float(v) for v in coefs],
                                                        r2_holdout=float(r2c), best_r2=float(best),
                                                        ladder={int(k): float(v) for k, v in ladder.items()})
                sel_txt.append("tol %.3g: {%s} R2 %.4f" % (tol, ", ".join("%+.3g %s" % (c, n) for c, n in zip(coefs, supp_names)), r2c))
            lines.append("  A2 (v1.1) %s, target %s, %s: %s  (best any %.4f)" % (label, target, variant, " | ".join(sel_txt), best))
            lines.append("      ladder (best holdout R2 per size): " + "; ".join(
                "%d: %.4f" % (k, ladder[k]) for k in sorted(ladder) if k > 0) +
                "   | {1, w} alone: %.4f" % r1w)
    return out


def v11_signs(P, args, lines):
    """sign diagnostic for relation A on the weak columns"""
    tr = P["_x"] < np.median(P["_x"])
    d, w = P["div k"], P["w"]
    frac = float(np.mean(d * w < 0))
    fits = {}
    for nm, yv, xv in (("signed: div k on w", d, w), ("absolute: |div k| on |w|", np.abs(d), np.abs(w)),
                       ("squared: (div k)^2 on w^2", d ** 2, w ** 2)):
        c, rh, rf = ratio_fit(yv, xv, tr)
        fits[nm] = dict(coef=c, r2_holdout=rh, r2_full=rf)
    lines.append("  SIGNS (v1.1): fraction of centres with div k and w of opposite sign: %.3f (the lower-side knee sign;"
                 " the PN upper side carries the other sign, so read the consistency max(f, 1 - f) = %.3f)" % (frac, max(frac, 1 - frac)))
    lines.append("      " + " ; ".join("%s: coef %+.3f, R2 %.3f" % (nm, f["coef"], f["r2_holdout"]) for nm, f in fits.items()))
    rows = []
    for key, dist, bins in (("row", P["_drow"], ROW_BINS), ("core", P["_dcore"], CORE_BINS)):
        for lo, hi in bins:
            m = (dist >= lo) & (dist < hi)
            if m.sum() >= 30:
                rows.append((key, lo, hi, int(m.sum()), float(np.mean((d * w < 0)[m]))))
    if rows:
        lines.append("      by distance (L): " + "; ".join("%s %.1f-%s: %.2f (n %d)" % (k, lo, ("%.0f" % hi) if np.isfinite(hi) else "inf", f, n)
                                                        for k, lo, hi, n, f in rows))
    sq, sg = fits["squared: (div k)^2 on w^2"]["r2_holdout"], fits["signed: div k on w"]["r2_holdout"]
    cons = max(frac, 1.0 - frac)                      # sign consistency, whichever sign the side carries
    if abs(sq - sg) < 0.03:
        verdict = "signed and squared agree"
    elif sq > sg and cons < 0.9:
        verdict = ("the squared relation beats the signed one by %.2f in R2 with sign consistency %.2f: a magnitude"
                   " relation with sign reversals, not the self-dual form" % (sq - sg, cons))
    elif sq > sg:
        verdict = ("the squared relation beats the signed one by %.2f in R2 with NO sign reversals (consistency %.2f):"
                   " a weighting effect, the squared fit is carried by the largest |w| centres (the cores), so the local"
                   " ratio -div k / w grows with |w| (read the by-core-distance betas)" % (sq - sg, cons))
    else:
        verdict = "the signed relation beats the squared one"
    lines.append("      reading: %s" % verdict)
    return dict(sign_fraction=frac, fits=fits, by_bins=rows, verdict=verdict)


def v11_decomposition(Pe, Pm, args, lines):
    """synthetic modes, both routes at the same centres: the estimator's
    error on each stationary column split into a gain (explained by the
    exact columns of the same relation) and a residual; beta refitted
    with each piece alone. Plain terms: does the attenuated beta come
    from a distortion of the columns (gain) or from noise in them
    (dilution)?"""
    if len(Pe["_x"]) != len(Pm["_x"]) or not np.allclose(Pe["_x"], Pm["_x"]):
        lines.append("  DECOMPOSITION (v1.1): skipped, the two routes do not share query centres")
        return
    tr = Pe["_x"] < np.median(Pe["_x"])
    lines.append("  DECOMPOSITION (v1.1, exact vs estimator columns at the same centres):")
    for yk, xk, sign in (("div k", "w", -1.0), ("(div k)^2", "w^2", 1.0), ("Lap div k", "S", 1.0)):
        ye, xe, ym, xm = Pe[yk], Pe[xk], Pm[yk], Pm[xk]
        # one scalar gain per column (on a knee the exact pair is proportional by
        # construction, so a joint regression of the error on both is singular)
        gx = float(np.dot(xe[tr], xm[tr]) / max(np.dot(xe[tr], xe[tr]), 1e-300))
        gy = float(np.dot(ye[tr], ym[tr]) / max(np.dot(ye[tr], ye[tr]), 1e-300))
        eps = np.column_stack([xm - gx * xe, ym - gy * ye])
        G = np.array([[gx - 1.0, 0.0], [0.0, gy - 1.0]])
        cases = {"exact": (xe, ye), "residual only": (xe + eps[:, 0], ye + eps[:, 1]),
                 "gain only": (gx * xe, gy * ye), "measured": (xm, ym)}
        vals = {}
        for nm, (xv, yv) in cases.items():
            c, rh, _ = ratio_fit(yv, xv, tr)
            vals[nm] = (sign * c, rh)
        rms = lambda v: float(np.sqrt(np.mean(v[tr] ** 2)))
        lines.append("    %s = c %s:  coef exact %+.3f | residual only %+.3f | gain only %+.3f | measured %+.3f  (holdout R2 %.3f, %.3f, %.3f, %.3f)" %
                     (yk, xk, vals["exact"][0], vals["residual only"][0], vals["gain only"][0], vals["measured"][0],
                      vals["exact"][1], vals["residual only"][1], vals["gain only"][1], vals["measured"][1]))
        lines.append("      column rms exact (%s, %s) = (%.3g, %.3g); error rms total (%.3g, %.3g), residual after the gain (%.3g, %.3g);"
                     " scalar gains: measured %s = %.3f x exact, measured %s = %.3f x exact (their ratio %.3f is the gain-only coefficient factor)" %
                     (xk, yk, rms(xe), rms(ye), rms(xm - xe), rms(ym - ye), rms(eps[:, 0]), rms(eps[:, 1]),
                      xk, gx, yk, gy, gy / gx))


def v11_beta_curve(args, lines, v11_rows, radii_used=1.5):
    """forward curve: planted tanh knees at the file's mu through the
    estimator at the same radius and domain; beta_read(beta_planted)"""
    betas = [float(v) for v in str(args.beta_curve).split(",") if v.strip()]
    try:
        _, estimate, _, _, t1c = _estimator()
    except ImportError as exc:
        lines.append("  BETA CURVE (v1.1): skipped (%s)" % exc); return
    mu = float(args.mu); k1 = float(np.sqrt(1 - mu ** 2)); L = np.pi / k1
    dy = args.Ly / (args.ny - 1)
    y = np.linspace(-args.Ly / 2, args.Ly / 2, args.ny)
    nx = int(round(args.n_xper * 2 * np.pi / k1 / dy)); x = (np.arange(nx) - nx // 2) * dy
    X, Y = np.meshgrid(x, y)
    taper = t1c.edge_taper(x, y, frac=args.taper_frac)
    reads, reads_ex = [], []
    side = args.side if args.side in ("both", "upper", "lower") else "both"
    for b in betas:
        th = k1 * X + np.log(np.cosh(b * mu * Y)) / b
        e = estimate(np.cos(th), nx * dy, args.ny * dy, args.sigma_dir, None, taper)
        K1, K2 = e["k1"], e["k2"]
        use_k1 = abs(np.mean(K1)) >= abs(np.mean(K2)); sg = np.sign(np.mean(K1 if use_k1 else K2)) or 1.0
        Pm = sample_frame(K1 * sg, K2 * sg, x, y, L, 0.0, None, args, side, radii_used)
        Pe = sample_frame(np.full_like(th, k1), mu * np.tanh(b * mu * Y), x, y, L, 0.0, None, args, side, radii_used)
        trm = Pm["_x"] < np.median(Pm["_x"]); tre = Pe["_x"] < np.median(Pe["_x"])
        cm, rm, _ = ratio_fit(Pm["div k"], Pm["w"], trm); ce, re_, _ = ratio_fit(Pe["div k"], Pe["w"], tre)
        reads.append((-cm, rm)); reads_ex.append((-ce, re_))
    lines.append("  BETA CURVE (v1.1, tanh knee at mu %.3f through the estimator, radius %.2g L, domain %s):" % (mu, radii_used, args.domain))
    lines.append("    planted : " + "  ".join("%.2f" % b for b in betas))
    lines.append("    exact   : " + "  ".join("%.2f" % v[0] for v in reads_ex))
    lines.append("    read    : " + "  ".join("%.2f" % v[0] for v in reads) + "   (holdout R2 " + ", ".join("%.3f" % v[1] for v in reads) + ")")
    br = np.array([v[0] for v in reads]); bp = np.array(betas)
    mono = np.all(np.diff(br) > 0)
    curve = dict(planted=betas, read=br.tolist(), exact=[v[0] for v in reads_ex], monotone=bool(mono))
    for route, row in v11_rows.items():
        beta_file = row.get("beta_signed")
        if route != "estimator" or beta_file is None or not np.isfinite(beta_file):
            continue
        if mono and br.min() <= beta_file <= br.max():
            corr = float(np.interp(beta_file, br, bp))
            lines.append("    model-corrected beta for the %s route (tanh-knee forward curve, a consistency reading): raw %.3f -> %.3f" %
                         (route, beta_file, corr))
            row["beta_model_corrected"] = corr
        else:
            lines.append("    model-corrected beta for the %s route: raw %.3f is outside the curve's range [%.2f, %.2f]%s; not corrected" %
                         (route, beta_file, br.min(), br.max(), "" if mono else " (curve not monotone)"))
            row["beta_model_corrected"] = np.nan
        row["beta_curve"] = curve


def v11_blocks(P, one, head, F, args, lines, stem, route):
    row = dict(stem=stem, route=route, n=int(len(P["_x"])), headline="%s ~ %s" % (head[1], head[2]), headline_r2=float(head[0]),
               beta_signed=float(F["beta"]), beta_r2=float(F["beta_r2_holdout"]), gamma_over_absbeta=float(F["gamma_over_absbeta"]),
               lam=float(F["lambda"]), lam_r2=float(F["lambda_r2_holdout"]), lambdaJ=float(F["lambdaJ"]), mJ=float(F["mJ"]),
               SJ_r2=float(F["SJ_r2_holdout"]))
    row["family"] = v11_family(one, head, args, lines)
    if int(args.a2):
        row["a2"] = v11_a2(P, args, lines)
    if int(args.signs):
        row["signs"] = v11_signs(P, args, lines)
    return row


# ---------------------------------------------------------------------
# v1.2: the ratio map and the constant-ratio mask
# ---------------------------------------------------------------------

def v12_ratio_map(fr, P, F, args, lines, out, label):
    """local ratio r = -div k / w on the last frame, weak and pointwise;
    the plateau (mode of the weak ratio on the kept domain) and the mask
    where the ratio stays within --ratio_tol of it; relation A inside and
    outside the mask. Plain terms: where on the domain does one constant
    beta have a chance, and what beta is it there?"""
    k1, k2, x, y, L = fr["k1"], fr["k2"], fr["x"], fr["y"], fr["L"]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    keep, d_row = domain_keep(x, y, L, fr["y_row"], fr["d_core"], args, fr["side"])
    C, m = weak_columns(k1, k2, x, y, L, args)
    Wk, Dk = C["w"], C["div k"]
    # valid interior (the weak columns are zero-padded at the box edge)
    ny, nx = k1.shape
    inner = np.zeros_like(keep); inner[m[1] + args.rim:ny - m[1] - args.rim, m[0] + args.rim:nx - m[0] - args.rim] = True
    kept = keep & inner
    wmed = float(np.median(np.abs(Wk[kept]))) if kept.any() else 1.0
    ok = kept & (np.abs(Wk) > args.ratio_wmin * wmed)
    r_weak = np.where(ok, -Dk / np.where(np.abs(Wk) > 1e-300, Wk, 1e-300), np.nan)
    # pointwise
    D = lambda Fd, ax: inner_deriv(Fd, args.eta * L, dx, dy, ax, "poly", args.poly_p)
    w_pw = k1 ** 2 + k2 ** 2 - 1.0
    d_pw = D(k1, "x") + D(k2, "y")
    ok_pw = kept & (np.abs(w_pw) > args.w_min)
    r_pw = np.where(ok_pw, -d_pw / np.where(np.abs(w_pw) > 1e-300, w_pw, 1e-300), np.nan)
    vals = r_weak[ok]
    if vals.size < 50:
        lines.append("  RATIO MAP (v1.2): fewer than 50 valid points; skipped")
        return {}
    lo, hi = np.percentile(vals, [2, 98])
    wts = np.abs(Wk[ok])                                   # signal weight: where the balance has content
    hist_w, edges = np.histogram(vals, bins=60, range=(lo, hi), weights=wts)
    hist_u, _ = np.histogram(vals, bins=60, range=(lo, hi))
    # plateau: the ratio value whose +-ratio_tol window captures the most |w| weight
    # (a scan, robust to a narrow spike beating a broad hump in a histogram)
    cand = np.linspace(lo, hi, 400)
    sv = np.sort(vals); cw = np.cumsum(wts[np.argsort(vals)]); cu = np.arange(1, len(sv) + 1)
    def captured(c, cum):
        a, b = c - args.ratio_tol * abs(c), c + args.ratio_tol * abs(c)
        ia, ib = np.searchsorted(sv, a), np.searchsorted(sv, b)
        return (cum[ib - 1] if ib > 0 else 0.0) - (cum[ia - 1] if ia > 0 else 0.0)
    cap_w = np.array([captured(c, cw) for c in cand]); cap_u = np.array([captured(c, cu) for c in cand])
    plateau = float(cand[int(np.argmax(cap_w))]); plateau_u = float(cand[int(np.argmax(cap_u))])
    jw = int(np.argmax(hist_w))
    med = float(np.median(vals))
    med_w = float(np.interp(0.5, np.cumsum(wts[np.argsort(vals)]) / wts.sum(), np.sort(vals)))
    mask = ok & (np.abs(r_weak - plateau) <= args.ratio_tol * abs(plateau))
    share = float(mask.sum() / max(ok.sum(), 1))
    # where the mask sits
    q = lambda a: np.percentile(a, [25, 50, 75]) if a.size else np.full(3, np.nan)
    rowq = q(d_row[mask]); coreq = q(fr["d_core"][mask]) if fr["d_core"] is not None else np.full(3, np.nan)
    # relation A on the centres inside / outside the mask (the centres of P, by position)
    IY, IX = centres(k1.shape, m, keep, args)
    inside = mask[IY, IX]
    res = {}
    for nm, sel in (("inside", inside), ("outside", ~inside & ok[IY, IX])):
        if sel.sum() >= 30:
            xs = x[IX][sel]; tr = xs < np.median(xs)
            c, rh, rf = ratio_fit(P["div k"][sel], P["w"][sel], tr) if len(P["div k"]) == len(IX) else (np.nan, np.nan, np.nan)
            res[nm] = dict(n=int(sel.sum()), beta=-c, r2_holdout=rh, r2_full=rf)
        else:
            res[nm] = dict(n=int(sel.sum()), beta=np.nan, r2_holdout=np.nan, r2_full=np.nan)
    share_w = float(wts[mask[ok]].sum() / wts.sum())
    # v1.3.1 placebo: w shifted along the row by half the domain
    Wp = np.roll(Wk, int(round(float(getattr(args, "placebo_shift", 0.5)) * L / dy)), axis=0)
    okp = kept & (np.abs(Wp) > args.ratio_wmin * wmed)
    r_pl = -Dk[okp] / np.where(np.abs(Wp[okp]) > 1e-300, Wp[okp], 1e-300)
    wtp = np.abs(Wp[okp])
    lo_p, hi_p = np.percentile(r_pl, [2, 98])
    svp = np.sort(r_pl); cwp = np.cumsum(wtp[np.argsort(r_pl)]); cup = np.arange(1, len(svp) + 1)
    def captured_p(c, cum):
        a_, b_ = c - args.ratio_tol * abs(c), c + args.ratio_tol * abs(c)
        ia, ib = np.searchsorted(svp, a_), np.searchsorted(svp, b_)
        return (cum[ib - 1] if ib > 0 else 0.0) - (cum[ia - 1] if ia > 0 else 0.0)
    candp = np.linspace(lo_p, hi_p, 400)
    cap_wp = np.array([captured_p(c, cwp) for c in candp]); cap_up = np.array([captured_p(c, cup) for c in candp])
    plateau_p = float(candp[int(np.argmax(cap_wp))])
    share_wp = float(cap_wp.max() / wtp.sum()); share_p = float(cap_up.max() / len(svp))
    lines.append("  RATIO MAP (v1.2): weak ratio -div k / w on the kept domain (|w| > %.2g x median): |w|-weighted plateau (max captured weight) %+.3f"
                 " (weighted median %+.3f); unweighted plateau (max captured area) %+.3f, median %+.3f; mask |r - plateau| <= %.0f%% covers %.1f%% of the"
                 " valid area and %.1f%% of the |w| weight" %
                 (args.ratio_wmin, plateau, med_w, plateau_u, med, 100 * args.ratio_tol, 100 * share, 100 * share_w))
    lines.append("      PLACEBO (v1.3.1, w shifted %.2g L across the row): best plateau %+.3f captures %.1f%% of the |w| weight"
                 " and %.1f%% of the area (real mask: %.1f%% / %.1f%%; ratio of weight shares %.1f)" %
                 (float(getattr(args, "placebo_shift", 0.5)), plateau_p, 100 * share_wp, 100 * share_p, 100 * share_w, 100 * share, share_w / max(share_wp, 1e-9)))
    lines.append("      mask location: row distance quartiles %s L; core distance quartiles %s L" %
                 ("/".join("%.2f" % v for v in rowq), "/".join(("%.2f" % v) if np.isfinite(v) else "n/a" for v in coreq)))
    lines.append("      relation A on the centres inside the mask: beta %+.3f (holdout R2 %.3f, n %d); outside: beta %+.3f (R2 %.3f, n %d); whole band (a): %+.3f (%.3f)" %
                 (res["inside"]["beta"], res["inside"]["r2_holdout"], res["inside"]["n"],
                  res["outside"]["beta"], res["outside"]["r2_holdout"], res["outside"]["n"], F["beta"], F["beta_r2_holdout"]))
    # v1.2.1: free search and A2 on the collar alone
    collar = {}
    if inside.sum() >= 100 and len(P["div k"]) == len(IX):
        Pc = {k: v[inside] for k, v in P.items()}
        resc, onec, headc, zeroc = free_search(Pc, args)
        famc = v11_family(onec, headc, args, lines) if False else None
        topc = max(o[0] for o in onec)
        admc = [o for o in onec if o[0] >= topc - args.tol and o[2] != "1"]
        admc.sort(key=lambda o: (ORDER[o[1]] + ORDER[o[2]], -o[0]))
        sdc = [o for o in admc if set((o[1], o[2])) == {"div k", "w"}]
        lines.append("      FREE FITS ON THE MASK (v1.2.1, n %d): headline %s ~ %s (coef %+.3g, R2 %.3f); family of %d: %s; div k ~ w %s" %
                     (inside.sum(), headc[1], headc[2], headc[3], headc[0], len(admc),
                      ", ".join("%s~%s (%.3f)" % (o[1], o[2], o[0]) for o in admc[:6]),
                      ("in the family at R2 %.3f" % sdc[0][0]) if sdc else "NOT in the family"))
        a2c = v11_a2(Pc, args, [])
        for key in ("div k | without columns that contain the target", "Lap div k | full library"):
            blk = a2c.get(key, {}).get("by_tol", {}).get("0.02")
            if blk:
                lines.append("      A2 on the mask, %s (tol 0.02): {%s} R2 %.3f" %
                             (key, ", ".join("%+.3g %s" % (c, n) for c, n in zip(blk["coefs"], blk["support"])), blk["r2_holdout"]))
        collar = dict(headline="%s ~ %s" % (headc[1], headc[2]), headline_r2=float(headc[0]), sd_in_family=bool(sdc),
                      family=[dict(target=o[1], partner=o[2], r2=o[0]) for o in admc], a2=a2c)
    # v1.2.1: the mask ladder, relation A with the support fixed as the domain grows
    ladder = []
    for tol in (0.1, 0.2, 0.4, 0.8, np.inf):
        msk = ok & (np.abs(r_weak - plateau) <= tol * abs(plateau)) if np.isfinite(tol) else ok
        sel = msk[IY, IX]
        if sel.sum() < 30 or len(P["div k"]) != len(IX):
            continue
        xs = x[IX][sel]; tr = xs < np.median(xs)
        c, rh, rf = ratio_fit(P["div k"][sel], P["w"][sel], tr)
        ladder.append(dict(tol=(float(tol) if np.isfinite(tol) else None), n=int(sel.sum()), area=float(msk.sum() / max(ok.sum(), 1)),
                           weight=float(wts[msk[ok]].sum() / wts.sum()), beta=-c, r2_holdout=rh))
    lines.append("      MASK LADDER (v1.2.1), relation A with the domain widened: " + "; ".join(
        "tol %s: n %d, area %.0f%%, weight %.0f%%, beta %+.3f (R2 %.3f)" %
        (("%.0f%%" % (100 * r["tol"])) if r["tol"] is not None else "all", r["n"], 100 * r["area"], 100 * r["weight"], r["beta"], r["r2_holdout"])
        for r in ladder))
    pw_vals = r_pw[ok_pw]
    if pw_vals.size:
        lines.append("      pointwise ratio on the same domain: median %+.3f, quartiles %s" %
                     (float(np.median(pw_vals)), "/".join("%.2f" % v for v in np.percentile(pw_vals, [25, 75]))))
    # figure
    fig, axs2 = plt.subplots(2, 4, figsize=(17, 7.2))
    axs = axs2[0]
    X, Y = np.meshgrid((x - x.mean()) / L, (y - fr["y_row"]) / L)
    ext = [X.min(), X.max(), Y.min(), Y.max()]
    yl = (max(Y.min(), -args.band_to - 1.0), min(Y.max(), args.band_to + 1.0))
    ax = axs[0]
    ax.imshow(np.cos(np.arctan2(k2, k1)) * 0 + kept, extent=ext, origin="lower", cmap="Greys", vmin=0, vmax=1.5, aspect="auto")
    ax.contour(X, Y, mask.astype(float), levels=[0.5], colors="C1", linewidths=0.8)
    ax.set_ylim(yl); ax.set_title("kept domain (grey) and mask (orange)", fontsize=8); ax.set_xlabel("x / L"); ax.set_ylabel("(y - y_row) / L")
    vmax = float(np.nanpercentile(np.abs(vals), 95))
    for ax, fld, ttl in ((axs[1], r_weak, "weak ratio -div k / w"), (axs[2], r_pw, "pointwise ratio -div k / w")):
        im = ax.imshow(fld, extent=ext, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.contour(X, Y, mask.astype(float), levels=[0.5], colors="k", linewidths=0.6)
        ax.set_ylim(yl); ax.set_title("%s (plateau %+.2f)" % (ttl, plateau), fontsize=8); ax.set_xlabel("x / L")
        fig.colorbar(im, ax=ax, fraction=0.04)
    ax = axs[3]
    ax.hist(vals, bins=60, range=(lo, hi), color="0.6", label="count")
    ax2 = ax.twinx(); ax2.hist(vals, bins=60, range=(lo, hi), weights=wts, histtype="step", color="C0", label="|w|-weighted"); ax2.set_yticks([])
    ax.axvline(plateau, color="C1", lw=1.5, label="plateau %+.2f" % plateau)
    ax.axvspan(plateau - args.ratio_tol * abs(plateau), plateau + args.ratio_tol * abs(plateau), color="C1", alpha=0.2, label="mask range")
    ax.axvline(F["beta"], color="k", ls="--", lw=1, label="band beta %+.2f" % F["beta"])
    ax.set_xlabel("weak -div k / w"); ax.legend(fontsize=7); ax.set_title("ratio histogram, mask share %.0f%%" % (100 * share), fontsize=8)
    # second row: target against the model the coefficients imply (v1.2.2)
    b_band = F["beta"]; b_col = res["inside"]["beta"] if np.isfinite(res["inside"]["beta"]) else b_band
    tgt = np.where(kept, Dk, np.nan)
    vm = float(np.nanpercentile(np.abs(tgt), 99))
    panels = (("weak target div k", tgt), ("model -beta w, whole-band beta %+.2f" % b_band, np.where(kept, -b_band * Wk, np.nan)),
              ("model -beta w, collar beta %+.2f" % b_col, np.where(kept, -b_col * Wk, np.nan)),
              ("residual div k + beta_collar w", np.where(kept, Dk + b_col * Wk, np.nan)))
    for ax, (ttl, fld) in zip(axs2[1], panels):
        im = ax.imshow(fld, extent=ext, origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto")
        ax.contour(X, Y, mask.astype(float), levels=[0.5], colors="k", linewidths=0.5)
        ax.set_ylim(yl); ax.set_title(ttl, fontsize=8); ax.set_xlabel("x / L")
        fig.colorbar(im, ax=ax, fraction=0.04)
    axs2[1][0].set_ylabel("(y - y_row) / L")
    fig.suptitle(label, fontsize=8); fig.tight_layout()
    fig.savefig(out / ("ratio_map_%s.png" % "".join(ch if ch.isalnum() else "_" for ch in label)), dpi=130); plt.close(fig)
    # v1.2.2: target against prediction, on the band
    b_band = F["beta"]; b_col = res["inside"]["beta"] if np.isfinite(res["inside"]["beta"]) else b_band
    T = np.where(kept, Dk, np.nan)
    vm = float(np.nanpercentile(np.abs(T), 99))
    fig, axs = plt.subplots(1, 5, figsize=(21, 3.6))
    panels = [(T, "weak target div k"), (np.where(kept, -b_band * Wk, np.nan), "prediction -beta w, band beta %+.2f" % b_band),
              (np.where(kept, Dk + b_band * Wk, np.nan), "residual, band beta"),
              (np.where(kept, -b_col * Wk, np.nan), "prediction -beta w, collar beta %+.2f" % b_col),
              (np.where(kept, Dk + b_col * Wk, np.nan), "residual, collar beta")]
    for ax, (fld, ttl) in zip(axs, panels):
        im = ax.imshow(fld, extent=ext, origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto")
        ax.contour(X, Y, mask.astype(float), levels=[0.5], colors="k", linewidths=0.5)
        ax.set_ylim(yl); ax.set_title(ttl, fontsize=8); ax.set_xlabel("x / L")
        fig.colorbar(im, ax=ax, fraction=0.04)
    rr = lambda fld: float(np.sqrt(np.nanmean(fld[kept] ** 2)) / np.sqrt(np.nanmean(T[kept] ** 2)))
    fig.suptitle("%s: relative residual rms on the band %.2f (band beta), %.2f (collar beta); on the collar %.2f, %.2f" %
                 (label, rr(panels[2][0]), rr(panels[4][0]),
                  float(np.sqrt(np.nanmean(panels[2][0][mask] ** 2)) / np.sqrt(np.nanmean(T[mask] ** 2))),
                  float(np.sqrt(np.nanmean(panels[4][0][mask] ** 2)) / np.sqrt(np.nanmean(T[mask] ** 2)))), fontsize=8)
    fig.tight_layout(); fig.savefig(out / ("fit_maps_%s.png" % "".join(ch if ch.isalnum() else "_" for ch in label)), dpi=130); plt.close(fig)
    lines.append("      FIT MAPS (v1.2.2): relative residual rms on the band %.3f (band beta) / %.3f (collar beta); on the collar %.3f / %.3f" %
                 (rr(panels[2][0]), rr(panels[4][0]),
                  float(np.sqrt(np.nanmean(panels[2][0][mask] ** 2)) / np.sqrt(np.nanmean(T[mask] ** 2))),
                  float(np.sqrt(np.nanmean(panels[4][0][mask] ** 2)) / np.sqrt(np.nanmean(T[mask] ** 2)))))
    return dict(plateau=plateau, plateau_unweighted=plateau_u, median=med, median_weighted=med_w, mask_share=share, mask_share_weighted=share_w,
                placebo_plateau=plateau_p, placebo_share_weighted=share_wp, placebo_share=share_p,
                row_quartiles=rowq.tolist(), core_quartiles=[float(v) for v in coreq],
                inside=res["inside"], outside=res["outside"], collar=collar, ladder=ladder,
                pointwise_median=(float(np.median(pw_vals)) if pw_vals.size else np.nan))


def main(args):
    t0 = time.time()
    lines = ["self_dual_discovery v1.3.3   (weak helpers: %s; PN field: %s)" % (_HELPERS, _PN),
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
    rx0, ry0 = radii_xy(args, 1.0)
    aniso = (abs(rx0 - args.radius) > 1e-9 or abs(ry0 - args.radius) > 1e-9)
    if aniso:
        Lsp0 = np.pi / np.sqrt(1 - float(args.mu) ** 2)
        px, py = periods_xy(args, Lsp0, rx0, ry0)
        lines.append("  ANISOTROPIC bump (v1.3): half widths %.3g L along the row, %.3g L across; the support spans %.2f"
                     " far-field 2theta periods along and %.2f across%s" %
                     (rx0, ry0, px, py, "  WARNING: under one period" if min(px, py) < 1.0 else ""))
    stamp = "rad%.2g_%s_ce%.2g%s_%s_v11%s%s" % (args.radius, args.domain, args.core_excl,
                                              {"circulation": "_cc", "charge": "_cq"}.get(str(getattr(args, "core_def", "dips")), "") if args.npz else "",
                                              args.frames if args.npz else args.route,
                                              ("_rx%g_ry%.2g" % (rx0, ry0)) if aniso else "",
                                              ("_" + args.tag) if getattr(args, "tag", "") else "")
    out = Path(args.outdir) / stem / stamp
    out.mkdir(parents=True, exist_ok=True)
    rows1, rows2, scan_rows = [], [], []
    v11_rows, route_P = {}, {}
    radii = [float(v) for v in str(args.radii).split(",") if v.strip()]
    for route, frames in routes:
        S = lambda r: pool([sample_frame(f["k1"], f["k2"], f["x"], f["y"], f["L"], f["y_row"], f["d_core"],
                                         args, f["side"], r) for f in frames])
        P = S(None)                                   # v1.3: the main fit takes radius_x / radius_y when set
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
        # ---- v1.1 additions
        v11 = v11_blocks(P, one, head, F, args, lines, stem, route)
        if int(getattr(args, "ratio_map", 1)):
            v11["ratio_map"] = v12_ratio_map(frames[-1], P, F, args, lines, out, "%s_%s" % (stem, route))
        v11_rows[route] = v11
        route_P[route] = P
    if int(getattr(args, "decomp", 1)) and "exact" in route_P and "estimator" in route_P:
        v11_decomposition(route_P["exact"], route_P["estimator"], args, lines)
    if str(getattr(args, "beta_curve", "")).strip() and (args.npz or args.mode == "knee"):
        v11_beta_curve(args, lines, v11_rows, radii_used=args.radius)
    with open(out / "summary_row.json", "w") as fh:
        json.dump(dict(stem=stem, mu=float(args.mu), routes=v11_rows), fh, indent=1, default=float)
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
    outdir = str(_HERE / "results" / "1010_self_dual_discovery_v11")
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
    core_def = "dips"             # dips (every amplitude dip, as before) | circulation (wound dips only, v1.3.2) | charge (director-charge clusters, v1.3.3)
    wind_hi = 0.9                 # v1.3.2: turns at the 0.5 L loop for a dip to count as a core
    sigma_det_charge = 0.3        # v1.3.3: director index width for --core_def charge
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
    # v1.1
    a2 = 1                        # fixed-target STLSQ for div k and Lap div k
    rel_tol_a2 = 0.01             # A2: sparsest support within this relative tolerance of the best holdout R2
    signs = 1                     # sign diagnostic for relation A
    decomp = 1                    # synthetic modes with both routes: gain / residual decomposition
    beta_curve = "0.8,1.0,1.2,1.3,1.5,1.8,2.2,3.0,4.0,6.0"   # planted betas for the forward curve ("" disables)
    # v1.2
    ratio_map = 1                 # the local ratio -div k / w mapped, the plateau and the constant-ratio mask
    ratio_tol = 0.2               # mask: |r - r_plateau| <= ratio_tol x |r_plateau|
    ratio_wmin = 0.1              # ratio defined where |weak w| > ratio_wmin x median |weak w| on the kept domain
    # v1.3
    radius_x = -1.0               # half width along the row, spacings (<= 0: radius)
    radius_y = -1.0               # half width across the row, spacings (<= 0: radius, unless ry_periods > 0)
    ry_periods = 0.0              # > 0: half width across set so the support spans this many 2theta periods across
    placebo_shift = 0.5           # v1.3.1: w shifted across the row by this many spacings for the plateau placebo
    tag = ""


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in dir(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    sys.exit(main(p.parse_args()))
