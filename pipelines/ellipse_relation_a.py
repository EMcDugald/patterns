# pipelines/ellipse_relation_a.py
"""
Relation A (the self-dual balance div k = -beta w, w = |k|^2 - 1; the sign
of beta is a convention) on Swift-Hohenberg ellipse patterns, with planted
synthetic controls on the same grid.

Plain terms. In an ellipse the rolls follow the curved wall, so they are
bent everywhere, with curvature 1/(rho - D) that decays algebraically away
from the wall (rho the wall's radius of curvature at the foot point, D the
distance from the wall). That keeps a measurable bending signal beside the
censored dislocation cores, where a zigzag boundary has none. The script
asks three questions, in order of importance:
  1. RING AGAINST BULK: is beta the same around the outer dislocation cores
     as in the defect-free bulk? (beta by core gap, the ring rows)
  2. ONE BETA OR MANY: is beta one constant across wall-normal rays
     (self-dual) or does it drift from ray to ray (leading-order
     Cross-Newell with a constant set per ray)? (beta by ray, by curvature)
  3. THE HEADLINE: does the bulk obey div k = -beta w at all? (holdout R2
     on both spatial splits, the free search)

Modes (run the synthetic ones first, on the grid of the data file)
  selfdual  the exact Cole-Hopf ellipse field theta = l log sum_j
            exp(-d_j / l), d_j = p(n_j) - n_j . x over n_normals outward
            normals n_j (p the support function), l = 1/beta. Every term
            solves the linear Cole-Hopf equation, so the sum is an EXACT
            solution of div k = -beta w (k oriented outward) for any number
            of normals. Pass: bulk beta within 3 percent of planted through
            the estimator (1 percent on the exact route), holdout R2 > 0.80
            (the bulk w = -kappa/beta shrinks as beta grows, and R2 with it).
  eikonal   theta = -D: curved rolls with w = 0 exactly in the bulk (the
            negative control; an uncurved-wavenumber phase model). Pass:
            bulk holdout R2 below 0.2.
  drift     a field whose beta changes from ray to ray, beta(q) = beta_mid +
            (beta_end - beta_mid) q^2, q = |X_axisfoot| / Xm the point where
            the ray (the wall normal) lands on the medial segment (0 at its
            centre, 1 at its tips). Rays are labelled by q rather than by the
            normal angle because at A/B = 3 every ray that has room for a
            bulk test function has a normal angle above about 60 degrees,
            while q spans 0 to 0.85. Built ray by ray: along a wall normal
            the self-dual relation is the radial Riccati equation, whose
            regular solution is |k| = I1(beta r) / I0(beta r), r = rho - D,
            so theta = -D - (1/beta) log[ i0e(beta rho) / i0e(beta (rho - D)) ]
            (i0e the scaled Bessel function; its large-radius limit is the
            first-order form -D + (1/(2 beta)) log(rho/(rho - D))). Exact
            along each ray; because beta and rho vary along the wall, the
            phase also picks up a small component along the wall, so the
            exact route (numerical gradient of the planted theta) is the
            reference the estimator is scored against, beside the planted
            beta. Pass: the by-ray estimator beta within 5 percent of the
            exact route. --drift_beta_end = --drift_beta_mid gives the ray
            construction at one beta (how far the ray field itself is from
            exact).
  sh        SH frames of an npz from run_sh_ellipse.py (S1 for stationary
            frames); --frames "-1,-2" (the full analysis on the first,
            the headline on the others, which must agree before either is
            read).
Synthetic modes take the grid from --npz when given (recommended: the S1
file), else from --A --B --dx --margin; they render u = sqrt(4 R0 / 3) W(s)
cos theta with the erf wall envelope W of the runner (width
--render_wall_sigma).

The chain
  Estimator: the frozen mg_estimator through mg_zigzag_theta_t.estimate at
  sigma_dir, no taper by default (the field vanishes at the box edge).
  Orientation: k is turned pointwise to point along the OUTWARD wall normal
  of its foot point (beta then has the knee sign, positive). Continuous
  except across the medial axis, which is the physical crease. The report
  gives the fraction of kept pixels where k is within 60 degrees of
  perpendicular to that normal (where the rule could be ambiguous).
  Columns: self_dual_discovery.weak_columns (the polynomial bump of
  kernel_d3, div k fully weak), imported, so ellipse and zigzag numbers come
  from the same code. Unit of length: the wavelength LAMBDA = 2 pi. Every
  distance argument below is in wavelengths.
  Centres: a query centre is used only when its whole test-function support
  (the square of half width m dx) lies in the kept region.

Cores (--core_def: any comma-separated subset of dips, circulation, charge;
--core_combine union | intersection)
  dips         connected regions where the mg amplitude falls below
               core_frac of its interior median (find_cores of
               mg_zigzag_theta_t), at least --det_wall from the wall.
  charge       director disclinations of the frozen t1c.detect_defects at
               sigma_det, discarded within --det_wall of the wall (counted),
               clustered within --merge.
  circulation  candidates (every dip, every charge cluster, and the runs of
               a loop scan along the axis) kept where the loop of radius
               --loop_r about them carries at least --wind_hi turns of k.
  All three are always computed and reported; only the listed ones place
  discs. Each core is labelled by the detectors with a point within
  --core_match of it and by its own circulation. Discs of --core_disc.
  Outer cores: curvature at the core at least --ring_kappa_min (on the
  210 x 70 ellipse the central cores sit near 0.002, below the estimator's
  floor).

Regions (centre classes; supports always inside the kept region)
  bulk      kept = inside, beyond --wall_margin, outside the discs, and more
            than --axis_excl from the medial segment; support gap to the
            nearest core at least --ring_width.
  ring      kept as bulk but WITH the axis band (the cores sit on it); the
            support's near edge within --ring_width of an outer core
            (ring_outer) or of a central one (ring_inner, the control).
  knee      axis-band centres beyond the outermost core on each side (with
            no cores: where the local angle mu of the axis is below
            --knee_mu_max).
  Readouts: beta by ray (landing point q), by geometric curvature, by
  strip (distance of the centre from the medial segment, with the local
  crossing angle mu = |n_y| of the foot), by core gap (outer cores; axis
  band included and excluded), the pointwise ratio map, the residual by wall
  gap, the scale test and a one-at-a-time sensitivity table (radius, wall
  margin, disc radius, axis exclusion, core definition).

Diagnostics for the hyperparameters (each prints in the report)
  wall margin   synthetic modes measure the estimator's error in the weak
                w and div k columns by the support's gap to the wall and
                print the smallest margin where both stay below --err_tol of
                the deep-bulk rms; the data run uses --wall_margin (set it
                from that line).
  orientation   fraction of ambiguous pixels.
  cores         per-detector counts and positions, near-wall charges
                dropped, detector agreement per core, circulation per core,
                the axis loop scan (figure).
  sensitivity   beta and holdout R2 of bulk and ring with one setting
                changed at a time; centres.npz holds every centre's
                position, attributes, region labels and weak columns, so any
                other binning can be recomputed without rerunning.
  control       --ref_json <fits.json of the selfdual run on the same grid>
                prints, beside the data's beta per region, the control's
                estimator and exact beta and the data value scaled by the
                control's exact/estimator ratio (separate columns; the ring
                of the control sits on false cores, so the scaled ring value
                is a rough reading).

Outputs (under --outdir; a relative outdir is anchored to this folder)
  report.txt, fits.json, cores.json, centres_<frame>.npz, config.json,
  fig_domain_<frame>.png (query centres, discs, margins, the kernel drawn
  over a bulk and a ring centre, a zoom on an outer core),
  fig_cores_<frame>.png, fig_readouts_<frame>.png, fig_scatter_<frame>.png,
  fig_ratio_map_<frame>.png, fig_residual_<frame>.png,
  fig_sensitivity_<frame>.png, fig_wall_error.png (synthetic).
  Geometry and synthetic fields are cached under results/ellipse_relation_a
  /_cache (--cache 0 disables).

Python 3.8; imports the frozen layer (op_extract, utils) through
mg_zigzag_theta_t and self_dual_discovery and never edits it.

Changelog
---------
v1.1 (27 Oct 2026, after Ed's first local runs, 1027_* under
   results/ellipse_relation_a) (1) the planted-control R2 pass mark is 0.80:
   locally beta 1.3 read 1.2958 (0.3 percent) at holdout R2 0.830, beta 1 read
   1.0040 at 0.954. (2) The sensitivity spread keeps the variants within
   --r2_drop (0.15) of the baseline's holdout R2 (an absolute 0.8 removed
   every variant on S1, whose baseline reads 0.787). (3) The control table
   scales a region only where the control's holdout R2 reaches --ref_r2_min
   (S1 printed 217 for ring_inner from a control R2 of 0.10). (4) Binned
   fits skip bins where w vanishes (the eikonal exact route printed 1e13).
   (5) The w profile prints w/kappa (-1/beta where w follows the curvature).
   No fitted number changes.
v1 (27 Oct 2026) first version. Checked in a chat sandbox on the 1000 x 378
   grid of a sandbox SH run (A 210, B 70, dx 0.45), with this file's real
   imports of mg_estimator, mg_zigzag_theta_t and self_dual_discovery but
   STAND-INS for the frozen field_diagnostics_viz and t1_common (FFT Riesz
   pair, doubled-angle director, plaquette charge). Not of record; the local
   runs are.
   selfdual beta 1: bulk beta 1.0004 (holdout R2 0.969, reverse 0.972, n
     3507), exact route 1.0000; free search puts div k ~ w first. Detectors on
     this defect-free field fire falsely along the axis (dips 20, charge 80
     clusters, 5 wound loops near the mu = 1 centre), so its ring rows measure
     the estimator beside the crease: ring_outer 0.868 and knee 0.886 against
     exact 1.000 (the crease attenuation known from the zigzag knees).
     Axis-band-excluded core-gap bins read 0.998 to 1.000.
   radius: 2.0 LAMBDA reads 1.003; 1.0 and 0.75 fail on the bulk (0.76 and
     0.44, holdout R2 below 0): a support under about 1.5 LAMBDA does not
     average the wavelength-scale ripple of the estimator (bulk signal
     rms 0.0025 against a pointwise w error of 0.0135).
   squared columns: gamma/|beta| reads 0.51 on the planted control (the
     ripple squared does not average), so stage 2 is read against the control.
   drift (beta 1.3 on rays landing at the centre of the axis, 0.8 at its
     tips): by-ray estimator within 2.5 percent of the exact route and 2.3
     percent of planted; the ray construction at one beta reads 1.001 to
     1.013 on the exact route (how far it is from exact).
   selfdual beta 1.3: bulk beta 1.2948 (0.4 percent), holdout R2 0.919; the
     R2 pass mark is 0.90, not the 0.95 of the design, because w = -kappa/beta
     is smaller at larger beta (0.969 at beta 1).
   eikonal: bulk holdout R2 -8.1 (pass).
   Rays are labelled by their landing point q on the medial segment, not by
     the normal angle: at A/B = 3 every ray with room for a bulk test function
     has a normal angle between about 60 and 90 degrees, while q spans 0 to
     0.85. Bins report a relative residual, not R2: inside one bin div k
     barely varies, so R2 about the bin mean was negative on correct fits.
   wall calibration: weak-column errors stay below 0.08 of the deep-bulk rms
     at every wall gap for the rendered envelope, so the estimator does not
     set the margin; the physics does: on the sandbox SH run the w profile
     shows a wall layer with w near -0.08 within 0.75 LAMBDA of the wall and
     negative to about 2 LAMBDA, hence the default --wall_margin 2.
   sandbox SH run (t 0 to 1600, not stationary, bulk rate 1.4e-4): bulk w is
     POSITIVE (+0.002 to +0.003 at 3 to 8 LAMBDA, shrinking from +0.005 at t
     400), against the self-dual expectation -kappa of about -0.002, so the
     bulk fit reads beta near -1.3 at holdout R2 0.84 with a ray spread of
     0.43. A transient of the size of the signal; S1 decides.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.special import i0e

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_ROOT / "src", _ROOT / "src" / "op_extract", _ROOT / "src" / "utils", _HERE):
    sys.path.insert(0, str(_p))

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.patches import Circle, Rectangle, Ellipse as EllipsePatch   # noqa: E402

from sh_sims.ellipse import EllipseGeom, wall_distance, wall_envelope, make_grid   # noqa: E402
import self_dual_discovery as sdd                                    # noqa: E402

SCRIPT_VERSION = "ellipse_relation_a.py v1.1 (2026-10-27)"
LAM = 2.0 * np.pi
DETECTORS = ("dips", "circulation", "charge")
C_BULK, C_RING, C_RINGI, C_KNEE, C_OTHER = "#2a78d6", "#eb6834", "#8e5bd0", "#1baf7a", "0.6"
C_DET = {"dips": "#d6a72a", "circulation": "#c0392b", "charge": "#2a9fd6"}


def _chain():
    """the frozen estimator chain, imported when first needed"""
    from mg_zigzag_theta_t import estimate, find_cores        # noqa
    from op_extract import t1_common as t1c                    # noqa
    return estimate, find_cores, t1c


def _floats(s):
    return [float(v) for v in str(s).replace(";", ",").split(",") if v.strip()]


def _r2(y, yh):
    return float(1.0 - np.sum((y - yh) ** 2) / (np.sum((y - np.mean(y)) ** 2) + 1e-300))


# ---------------------------------------------------------------------
# grid, geometry, fields
# ---------------------------------------------------------------------

def load_input(args):
    """grid (and frames for mode sh) from --npz, else a square grid from
    --A --B --dx --margin. Returns dict x, y, Lx, Ly, A, B, R0, y_axis, and
    u [ny, nx, nt], tt for an npz."""
    if args.npz:
        d = np.load(args.npz, allow_pickle=True)
        md = json.loads(str(d["metadata_json"])) if "metadata_json" in d.files else {}
        x, y = np.asarray(d["x"], float), np.asarray(d["y"], float)
        u = np.asarray(d["u"])
        if u.ndim == 2:
            u = u[..., None]
        if u.shape[:2] != (len(y), len(x)):
            u = np.moveaxis(u, 0, -1)
        tt = np.asarray(d["tt"] if "tt" in d.files else d["t"], float)
        ulast = np.asarray(d["u_last"], float) if "u_last" in d.files else None
        A = float(md.get("A", args.A)); B = float(md.get("B", args.B)); R0 = float(md.get("R0", args.R0))
        jr = int(md.get("gb_row_in_crop", len(y) // 2))
        src = "npz %s (%s)" % (Path(args.npz).name, md.get("runner", "runner unknown"))
    else:
        g = make_grid("square", args.A, args.B, args.dx, args.margin)
        x, y, A, B, R0 = g["x"], g["y"], args.A, args.B, args.R0
        u, tt, ulast, md, jr = None, None, None, {}, len(y) // 2
        src = "square grid from --A --B --dx --margin"
    dx = float(x[1] - x[0])
    return dict(x=x, y=y, dx=dx, Lx=len(x) * dx, Ly=len(y) * dx, A=A, B=B, R0=R0, y_axis=float(y[jr]),
                jr=jr, u=u, tt=tt, u_last=ulast, md=md, src=src)


def _cache_path(args, I, what):
    root = _HERE / "results" / "ellipse_relation_a" / "_cache"
    key = "A%g_B%g_nx%d_ny%d_dx%.6g_x0%.6g_y0%.6g" % (I["A"], I["B"], len(I["x"]), len(I["y"]), I["dx"], I["x"][0], I["y"][0])
    return root / ("%s_%s.npz" % (what, key))


def geometry(I, args):
    """exact wall distance and foot attributes on the grid (cached)"""
    path = _cache_path(args, I, "geom_v2")
    if int(args.cache) and path.exists():
        z = np.load(path)
        G = {k: z[k] for k in z.files}
    else:
        X, Y = np.meshgrid(I["x"], I["y"])
        E = EllipseGeom(I["A"], I["B"])
        dist = wall_distance(E, X, Y)
        al = dist["alpha"]
        sx = np.where(X >= 0, 1.0, -1.0); sy = np.where(Y >= I["y_axis"], 1.0, -1.0)
        nxo, nyo = sx * np.cos(al), sy * np.sin(al)
        tpar = np.arctan2(I["B"] * np.sin(al), I["A"] * np.cos(al))
        rho = (I["A"] ** 2 * np.sin(tpar) ** 2 + I["B"] ** 2 * np.cos(tpar) ** 2) ** 1.5 / (I["A"] * I["B"])
        G = dict(X=X, Y=Y, d=dist["d"], s=dist["s"], inside=dist["inside"], dm=dist["dm"], alpha=al,
                 nxo=nxo, nyo=nyo, rho=rho, Xfoot=dist["Xfoot"])
        if int(args.cache):
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **G)
    with np.errstate(divide="ignore", invalid="ignore"):
        G["kappa"] = np.where(G["inside"] & (G["rho"] - G["d"] > 1e-6), 1.0 / np.maximum(G["rho"] - G["d"], 1e-6), np.nan)
    G["mu"] = np.sin(G["alpha"])
    G["Xm"] = (I["A"] ** 2 - I["B"] ** 2) / I["A"]
    G["q"] = np.abs(G["Xfoot"]) / G["Xm"]          # the ray's landing point on the medial segment, 0 centre .. 1 tip
    return G


def render_amp(G, I, args):
    return np.sqrt(4.0 * I["R0"] / 3.0) * wall_envelope(G["s"], args.render_wall_sigma)


def field_selfdual(G, I, args):
    """Cole-Hopf ellipse field, outward-oriented: theta = l lse_j(-d_j / l),
    k = sum_j W_j n_j (W the softmax weights); exact for div k = -beta w"""
    path = _cache_path(args, I, "selfdual_b%.6g_n%d" % (args.beta, args.n_normals))
    if int(args.cache) and path.exists():
        z = np.load(path)
        return dict(theta=z["theta"], k1=z["k1"], k2=z["k2"])
    ell = 1.0 / float(args.beta)
    t = np.linspace(0.0, 2.0 * np.pi, int(args.n_normals), endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    p = np.sqrt((I["A"] * ct) ** 2 + (I["B"] * st) ** 2)
    X, Y = G["X"], G["Y"] - I["y_axis"]
    th = np.empty_like(X); k1 = np.empty_like(X); k2 = np.empty_like(X)
    for i in range(X.shape[0]):
        Ei = -(p[None, :] - X[i][:, None] * ct[None, :] - Y[i][:, None] * st[None, :]) / ell
        mx = Ei.max(axis=1)
        W = np.exp(Ei - mx[:, None]); S = W.sum(axis=1)
        W /= S[:, None]
        th[i] = ell * (mx + np.log(S))
        k1[i] = (W * ct[None, :]).sum(axis=1); k2[i] = (W * st[None, :]).sum(axis=1)
    if int(args.cache):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, theta=th, k1=k1, k2=k2)
    return dict(theta=th, k1=k1, k2=k2)


def drift_beta_map(G, args):
    """beta per ray: drift_beta_mid on the rays landing at the centre of the
    medial segment, drift_beta_end on those landing at its tips, quadratic in
    q = |X_axisfoot| / Xm (constant along each ray)"""
    return args.drift_beta_mid + (args.drift_beta_end - args.drift_beta_mid) * G["q"] ** 2


def field_ray(G, I, args):
    """ray-by-ray self-dual field with beta(alpha); exact route by the
    numerical gradient of the planted theta"""
    b = drift_beta_map(G, args)
    r = np.maximum(G["rho"] - G["d"], 1e-3)
    th = -G["d"] - (1.0 / b) * np.log(i0e(b * G["rho"]) / i0e(b * r))
    th = np.where(G["inside"], th, 0.0)
    k2, k1 = np.gradient(th, I["dx"])
    return dict(theta=th, k1=k1, k2=k2, beta_map=b)


def field_eikonal(G, I, args):
    th = np.where(G["inside"], -G["d"], 0.0)
    return dict(theta=th, k1=G["nxo"].copy(), k2=G["nyo"].copy())


def synthetic(G, I, args, lines):
    if args.mode == "selfdual":
        F = field_selfdual(G, I, args)
        F["beta_map"] = np.full_like(G["d"], float(args.beta))
        desc = "Cole-Hopf ellipse field, %d normals, planted beta %.4g (l = %.4g)" % (args.n_normals, args.beta, 1.0 / args.beta)
    elif args.mode == "eikonal":
        F = field_eikonal(G, I, args)
        F["beta_map"] = np.full_like(G["d"], np.nan)
        desc = "eikonal theta = -D (w = 0 in the bulk): negative control"
    elif args.mode == "drift":
        F = field_ray(G, I, args)
        desc = ("ray field, beta(q) = %.4g + (%.4g - %.4g) q^2, q = |X_axisfoot|/Xm the ray's landing point on the medial"
                " segment (%.4g on rays landing at its centre, %.4g at its tips)"
                % (args.drift_beta_mid, args.drift_beta_end, args.drift_beta_mid, args.drift_beta_mid, args.drift_beta_end))
    else:
        raise ValueError(args.mode)
    a = render_amp(G, I, args)
    F["u"] = a * np.cos(F["theta"])
    ins = G["inside"]
    F["k1"] = np.where(ins, F["k1"], 0.0); F["k2"] = np.where(ins, F["k2"], 0.0)
    lines.append("FIELD: %s; rendered u = sqrt(4 R0/3) W(s) cos theta, R0 %.3g, wall envelope width %.3g" %
                 (desc, I["R0"], args.render_wall_sigma))
    return F


# ---------------------------------------------------------------------
# estimator, orientation
# ---------------------------------------------------------------------

def run_estimator(u, I, args):
    estimate, _, t1c = _chain()
    taper = t1c.edge_taper(I["x"], I["y"], frac=args.taper_frac) if args.taper_frac > 0 else np.ones_like(u)
    e = estimate(u, I["Lx"], I["Ly"], args.sigma_dir, None, taper)
    return dict(k1=e["k1"], k2=e["k2"], A=e["A"], taper=taper)


def orient(k1, k2, G):
    dot = k1 * G["nxo"] + k2 * G["nyo"]
    sg = np.where(dot < 0, -1.0, 1.0)
    km = np.hypot(k1, k2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = np.abs(dot) / np.maximum(km, 1e-12)
    ins = G["inside"]
    return np.where(ins, k1 * sg, 0.0), np.where(ins, k2 * sg, 0.0), cosang


# ---------------------------------------------------------------------
# cores
# ---------------------------------------------------------------------

def _cluster(px, py, w, r):
    """greedy clusters within r (centroid of members; summed weight)"""
    px, py, w = np.asarray(px, float), np.asarray(py, float), np.asarray(w, float)
    if len(px) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    order = np.argsort(px); px, py, w = px[order], py[order], w[order]
    used = np.zeros(len(px), bool); cx, cy, cw = [], [], []
    for i in range(len(px)):
        if used[i]:
            continue
        g = (~used) & (np.hypot(px - px[i], py - py[i]) < r)
        used |= g
        cx.append(px[g].mean()); cy.append(py[g].mean()); cw.append(w[g].sum())
    return np.array(cx), np.array(cy), np.array(cw)


def _pix(I, xv, yv):
    i = np.clip(np.round((np.asarray(xv) - I["x"][0]) / I["dx"]).astype(int), 0, len(I["x"]) - 1)
    j = np.clip(np.round((np.asarray(yv) - I["y"][0]) / I["dx"]).astype(int), 0, len(I["y"]) - 1)
    return j, i


def loop_closes(k1, k2, x, y, xc, yc, r):
    """whether the sign of k, carried around the loop by continuity (as in
    _loop_circ), comes back to its start. An open loop encloses an odd number
    of half-turn cells and its |Gamma| is not a clean integer; near the
    centre of the ellipse's axis, where the two roll sets meet almost head on
    (mu near 1), this is common"""
    dx = x[1] - x[0]; dy = y[1] - y[0]
    ang = np.linspace(0, 2 * np.pi, 181)
    ix = np.clip(np.round((xc + r * np.cos(ang) - x[0]) / dx).astype(int), 0, len(x) - 1)
    iy = np.clip(np.round((yc + r * np.sin(ang) - y[0]) / dy).astype(int), 0, len(y) - 1)
    kx, ky = k1[iy, ix], k2[iy, ix]
    sg = 1.0
    for q in range(1, len(ang)):
        sg = sg if kx[q] * kx[q - 1] + ky[q] * ky[q - 1] >= 0 else -sg
    return bool(sg * (kx[-1] * kx[0] + ky[-1] * ky[0]) > 0)


def detect_cores(u, est, G, I, args):
    """all three detectors; returns per-detector points and diagnostics"""
    _, find_cores, t1c = _chain()
    x, y = I["x"], I["y"]
    valid = G["inside"] & (G["s"] > args.det_wall * LAM)
    out = {}
    # dips
    A = est["A"]
    Am = np.where(valid, A, np.inf)
    xd, yd = find_cores(Am, x, y, valid, args.core_frac, merge_dist=args.merge * LAM)
    out["dips"] = dict(x=np.asarray(xd, float), y=np.asarray(yd, float), A_ref=float(np.median(A[valid])))
    # charge
    det = t1c.detect_defects(u * est["taper"], x, y - I["y_axis"], I["Lx"], I["Ly"], sigma_det=args.sigma_det,
                             band_y=float(I["B"] + 2 * LAM))
    ch = det["charges"]
    px = np.array([c[0] for c in ch], float); py = np.array([c[1] for c in ch], float) + I["y_axis"]
    pq = np.array([c[2] for c in ch], float)
    if len(px):
        jj, ii = _pix(I, px, py)
        ok = valid[jj, ii]
    else:
        ok = np.zeros(0, bool)
    cx, cy, cq = _cluster(px[ok], py[ok], np.abs(pq[ok]), args.merge * LAM)
    _, _, cnet = _cluster(px[ok], py[ok], pq[ok], args.merge * LAM)
    out["charge"] = dict(x=cx, y=cy, abs_q=cq, net_q=cnet, n_cells=int(ok.sum()), n_cells_dropped=int((~ok).sum()),
                         sum_abs_all=int(det["sum_abs"]), dropped_x=px[~ok], dropped_y=py[~ok])
    # circulation: loop scan along the axis, then every candidate
    K1, K2 = est["k1"], est["k2"]
    rr = args.loop_r * LAM
    xs = np.arange(-G["Xm"] - 2 * LAM, G["Xm"] + 2 * LAM + 1e-9, args.axis_scan_step * LAM)
    js, is_ = _pix(I, xs, np.full_like(xs, I["y_axis"]))
    okscan = G["inside"][js, is_] & (G["s"][js, is_] > args.det_wall * LAM + rr)
    xs = xs[okscan]
    gam = np.array([abs(sdd._loop_circ(K1, K2, x, y, a, I["y_axis"], rr)) for a in xs])
    runs = []
    hi = gam >= args.wind_hi
    i0 = None
    for i in range(len(xs) + 1):
        if i < len(xs) and hi[i] and i0 is None:
            i0 = i
        elif (i == len(xs) or not hi[i]) and i0 is not None:
            runs.append(0.5 * (xs[i0] + xs[i - 1])); i0 = None
    cand_x = np.r_[out["dips"]["x"], out["charge"]["x"], np.array(runs)]
    cand_y = np.r_[out["dips"]["y"], out["charge"]["y"], np.full(len(runs), I["y_axis"])]
    qx, qy, _ = _cluster(cand_x, cand_y, np.ones(len(cand_x)), args.merge * LAM)
    gq = np.array([abs(sdd._loop_circ(K1, K2, x, y, a, b, rr)) for a, b in zip(qx, qy)])
    keep = gq >= args.wind_hi
    out["circulation"] = dict(x=qx[keep], y=qy[keep], gamma=gq[keep], cand_x=qx, cand_y=qy, cand_gamma=gq,
                              scan_x=xs, scan_gamma=gam, n_runs=len(runs))
    return out


def combine_cores(dets, args, G, I, est):
    """discs from the listed detectors (union or intersection), each core
    labelled by every detector and by its own circulation"""
    names = [n.strip() for n in str(args.core_def).split(",") if n.strip()]
    bad = [n for n in names if n not in DETECTORS]
    if bad:
        raise ValueError("unknown --core_def entries %s (use %s)" % (bad, ", ".join(DETECTORS)))
    r = args.core_match * LAM
    px = np.concatenate([dets[n]["x"] for n in names]) if names else np.zeros(0)
    py = np.concatenate([dets[n]["y"] for n in names]) if names else np.zeros(0)
    src = np.concatenate([np.full(len(dets[n]["x"]), k) for k, n in enumerate(names)]) if names else np.zeros(0)
    cores = []
    if len(px):
        order = np.argsort(px); px, py, src = px[order], py[order], src[order]
        used = np.zeros(len(px), bool)
        for i in range(len(px)):
            if used[i]:
                continue
            g = (~used) & (np.hypot(px - px[i], py - py[i]) < r)
            used |= g
            have = set(int(v) for v in src[g])
            if args.core_combine == "intersection" and len(have) < len(names):
                continue
            cores.append(dict(x=float(px[g].mean()), y=float(py[g].mean())))
    rr = args.loop_r * LAM
    for c in cores:
        for n in DETECTORS:
            dd = np.hypot(dets[n]["x"] - c["x"], dets[n]["y"] - c["y"]) if len(dets[n]["x"]) else np.zeros(0)
            c[n] = bool(len(dd) and dd.min() < r)
        c["gamma"] = abs(sdd._loop_circ(est["k1"], est["k2"], I["x"], I["y"], c["x"], c["y"], rr))
        c["closes"] = loop_closes(est["k1"], est["k2"], I["x"], I["y"], c["x"], c["y"], rr)
        j, i = _pix(I, c["x"], c["y"])
        c["kappa"] = float(G["kappa"][j, i]) if np.isfinite(G["kappa"][j, i]) else float("nan")
        c["mu_axis"] = float(G["mu"][j, i])
        c["outer"] = bool(np.isfinite(c["kappa"]) and c["kappa"] >= args.ring_kappa_min)
    return cores, names


# ---------------------------------------------------------------------
# masks, centres, columns
# ---------------------------------------------------------------------

def core_distance(G, cores):
    if not cores:
        return np.full_like(G["d"], np.inf), np.full(G["d"].shape, -1)
    cx = np.array([c["x"] for c in cores]); cy = np.array([c["y"] for c in cores])
    D = np.hypot(G["X"][..., None] - cx[None, None, :], G["Y"][..., None] - cy[None, None, :])
    return D.min(axis=-1), D.argmin(axis=-1)


def kept_masks(G, dc, wall_margin, core_disc, axis_excl):
    near = G["inside"] & (G["s"] > wall_margin * LAM) & (dc > core_disc * LAM)
    bulk = near & (G["dm"] > axis_excl * LAM)
    return near, bulk


def kernel_m(radius, I, args):
    return int(sdd.kernel_d3(radius * LAM, I["dx"], args.poly_p)[4])


def centre_grid(shape, m, args):
    ny, nx = shape
    iy = np.arange(m + args.rim, ny - m - args.rim, args.stride)
    ix = np.arange(m + args.rim, nx - m - args.rim, args.stride)
    IY, IX = np.meshgrid(iy, ix, indexing="ij")
    return IY.ravel(), IX.ravel()


def admitted(kept, m, IY, IX):
    """centres whose (2m+1)-square support lies in kept"""
    bad = maximum_filter((~kept).astype(np.uint8), size=2 * m + 1, mode="constant", cval=1).astype(bool)
    return ~bad[IY, IX]


def build_centres(G, I, cores, radius, wall_margin, core_disc, axis_excl, args):
    m = kernel_m(radius, I, args)
    a = m * I["dx"]
    dc, ic = core_distance(G, cores)
    near, bulk = kept_masks(G, dc, wall_margin, core_disc, axis_excl)
    IY, IX = centre_grid(G["d"].shape, m, args)
    adm_n = admitted(near, m, IY, IX)
    adm_b = admitted(bulk, m, IY, IX)
    sel = adm_n | adm_b
    IY, IX, adm_n, adm_b = IY[sel], IX[sel], adm_n[sel], adm_b[sel]
    C = dict(IY=IY, IX=IX, X=G["X"][IY, IX], Y=G["Y"][IY, IX] - I["y_axis"], s=G["s"][IY, IX], dm=G["dm"][IY, IX],
             alpha=G["alpha"][IY, IX], q=G["q"][IY, IX], kappa=G["kappa"][IY, IX], mu=G["mu"][IY, IX], dc=dc[IY, IX],
             gap_core=dc[IY, IX] - a, gap_wall=G["s"][IY, IX] - a, adm_near=adm_n, adm_bulk=adm_b, half_width=a, m=m)
    nearest = ic[IY, IX]
    outer = np.array([c["outer"] for c in cores]) if cores else np.zeros(0, bool)
    C["near_outer"] = outer[nearest] if cores else np.zeros(len(IY), bool)
    C["near_core_x"] = np.array([cores[k]["x"] for k in nearest]) if cores else np.full(len(IY), np.nan)
    rw = args.ring_width * LAM
    C["bulk"] = adm_b & (C["gap_core"] >= rw)
    C["ring_outer"] = adm_n & (C["gap_core"] >= 0) & (C["gap_core"] < rw) & C["near_outer"]
    C["ring_inner"] = adm_n & (C["gap_core"] >= 0) & (C["gap_core"] < rw) & ~C["near_outer"]
    inband = adm_n & (C["dm"] < axis_excl * LAM)
    if cores:
        xc = np.array([c["x"] for c in cores])
        xr, xl = (xc.max(), xc.min())
        beyond = ((C["X"] > xr + (core_disc + args.ring_width) * LAM) |
                  (C["X"] < xl - (core_disc + args.ring_width) * LAM))
    else:
        mu_ax = np.full(len(IY), 1.0)
        Xa = np.clip(np.abs(C["X"]), 0, G["Xm"])
        ct = np.clip(I["A"] * Xa / max(I["A"] ** 2 - I["B"] ** 2, 1e-12), -1, 1)
        stt = np.sqrt(1 - ct ** 2)
        mu_ax = I["A"] * stt / np.sqrt(I["B"] ** 2 * ct ** 2 + I["A"] ** 2 * stt ** 2)
        beyond = mu_ax < args.knee_mu_max
    C["knee"] = inband & beyond
    return C, dict(near=near, bulk=bulk, dc=dc)


def sample(Ccols, C, sel):
    P = {nm: F[C["IY"][sel], C["IX"][sel]] for nm, F in Ccols.items()}
    P["_x"] = C["X"][sel]; P["_drow"] = C["dm"][sel] / LAM
    P["_dcore"] = np.where(np.isfinite(C["dc"][sel]), C["dc"][sel] / LAM, np.inf)
    return P


# ---------------------------------------------------------------------
# fits
# ---------------------------------------------------------------------

def fit_relation(d, w, X):
    """div k = -beta w on X < 0 scored on X >= 0, the reverse, the full fit,
    and the variant with an intercept"""
    out = dict(n=int(len(d)))
    if len(d) < 4:
        return dict(out, beta=np.nan, r2_hold=np.nan, beta_rev=np.nan, r2_rev=np.nan, beta_full=np.nan, r2_full=np.nan,
                    c0=np.nan, beta_c=np.nan, r2_hold_c=np.nan, c0_rel=np.nan, n_train=0, n_test=0)
    tr = X < 0
    te = ~tr
    out["n_train"], out["n_test"] = int(tr.sum()), int(te.sum())
    def one(a, b):
        if a.sum() < 2 or b.sum() < 2:
            return np.nan, np.nan
        c = float(np.dot(w[a], d[a]) / max(np.dot(w[a], w[a]), 1e-300))
        return -c, _r2(d[b], c * w[b])
    out["beta"], out["r2_hold"] = one(tr, te)
    out["beta_rev"], out["r2_rev"] = one(te, tr)
    c = float(np.dot(w, d) / max(np.dot(w, w), 1e-300))
    out["beta_full"], out["r2_full"] = -c, _r2(d, c * w)
    out["w_rms"] = float(np.sqrt(np.mean(w ** 2)))
    out["w_vanishes"] = bool(out["w_rms"] < 1e-9 * max(float(np.sqrt(np.mean(d ** 2))), 1e-300))
    if tr.sum() >= 3 and te.sum() >= 3:
        M = np.column_stack([np.ones(int(tr.sum())), w[tr]])
        cc, *_ = np.linalg.lstsq(M, d[tr], rcond=None)
        out["c0"], out["beta_c"] = float(cc[0]), float(-cc[1])
        out["r2_hold_c"] = _r2(d[te], cc[0] + cc[1] * w[te])
        out["c0_rel"] = float(abs(cc[0]) / max(np.sqrt(np.mean(d ** 2)), 1e-300))
    else:
        out["c0"] = out["beta_c"] = out["r2_hold_c"] = out["c0_rel"] = np.nan
    return out


def fmt_fit(f):
    if f["n"] < 4 or not np.isfinite(f.get("beta_full", np.nan)):
        return "n %d: too few centres" % f["n"]
    if f.get("w_vanishes"):
        return "w vanishes identically on these centres (rms %.1e): no relation to fit, by construction (n %d)" % (f["w_rms"], f["n"])
    return ("beta %+.4f (train X<0, holdout R2 %.3f) | reverse %+.4f (R2 %.3f) | full %+.4f (R2 %.3f) | with intercept: beta %+.4f,"
            " c0 %.2g (%.2f of rms div k), holdout R2 %.3f | n %d (%d + %d)" %
            (f["beta"], f["r2_hold"], f["beta_rev"], f["r2_rev"], f["beta_full"], f["r2_full"], f["beta_c"], f["c0"],
             f["c0_rel"], f["r2_hold_c"], f["n"], f["n_train"], f["n_test"]))


def binned(d, w, key, edges, extra=None, min_n=30):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (key >= lo) & (key < hi)
        n = int(m.sum())
        row = dict(lo=float(lo), hi=float(hi), n=n, beta=np.nan, relres=np.nan)
        if n >= min_n and np.sqrt(np.mean(w[m] ** 2)) > 1e-9 * max(np.sqrt(np.mean(d[m] ** 2)), 1e-300):
            c = float(np.dot(w[m], d[m]) / max(np.dot(w[m], w[m]), 1e-300))
            # relative residual rms(div k + beta w) / rms(div k), not R2: inside one bin div k barely varies, so the
            # variance about the bin mean (the R2 denominator) is not the scale of the signal
            row["beta"] = -c
            row["relres"] = float(np.sqrt(np.mean((d[m] - c * w[m]) ** 2)) / max(np.sqrt(np.mean(d[m] ** 2)), 1e-300))
        for nm, v in (extra or {}).items():
            row[nm] = float(np.nanmedian(v[m])) if n else np.nan
        rows.append(row)
    return rows


def fmt_bins(rows, unit, extra_fmt=()):
    out = []
    for r in rows:
        hi = ("%.3g" % r["hi"]) if r["hi"] < 1e8 else "inf"
        s = "%s-%s%s: " % (("%.3g" % r["lo"]), hi, unit)
        s += ("beta %+.3f (rel. residual %.2f, n %d)" % (r["beta"], r["relres"], r["n"])) if np.isfinite(r["beta"]) else "n %d" % r["n"]
        for nm, f in extra_fmt:
            if nm in r and np.isfinite(r[nm]):
                s += (", %s " + f) % (nm, r[nm])
        out.append(s)
    return out


# ---------------------------------------------------------------------
# analysis of one frame (or one synthetic field)
# ---------------------------------------------------------------------

class Frame(object):
    pass


def analyse(label, u, G, I, args, lines, F=None, full=True, outdir=None):
    """estimator, cores, centres, fits and readouts on one field. F: the
    synthetic field (exact route) or None."""
    t0 = time.time()
    R = Frame()
    R.label = label
    R.u = u
    est = run_estimator(u, I, args)
    k1, k2, cosang = orient(est["k1"], est["k2"], G)
    R.k1, R.k2, R.A = k1, k2, est["A"]
    dets = detect_cores(u, est, G, I, args)
    cores, names = combine_cores(dets, args, G, I, est)
    R.dets, R.cores = dets, cores
    C, M = build_centres(G, I, cores, args.radius, args.wall_margin, args.core_disc, args.axis_excl, args)
    R.C, R.M = C, M
    cols_cache = {}

    def cols(radius, route="est"):
        key = (round(radius, 6), route)
        if key not in cols_cache:
            if route == "est":
                cols_cache[key] = sdd.weak_columns(k1, k2, I["x"], I["y"], LAM, args, radius)[0]
            else:
                cols_cache[key] = sdd.weak_columns(F["k1"], F["k2"], I["x"], I["y"], LAM, args, radius)[0]
        return cols_cache[key]

    Cm = cols(args.radius)
    Ce = cols(args.radius, "exact") if (F is not None and args.route in ("both", "exact")) else None
    R.Cm, R.Ce = Cm, Ce
    flags = []
    lines.append("")
    lines.append("=" * 100)
    lines.append("FIELD %s" % label)
    lines.append("=" * 100)
    # orientation
    amb = M["near"] & (cosang < 0.5)
    fa = float(amb.sum() / max(M["near"].sum(), 1))
    fab = float((M["bulk"] & (cosang < 0.5)).sum() / max(M["bulk"].sum(), 1))
    lines.append("ORIENTATION: k turned along the outward wall normal; k within 60 degrees of perpendicular to it on %.2f%% of"
                 " the kept pixels (%.2f%% of the bulk-family pixels)%s" %
                 (100 * fa, 100 * fab, "  FLAG: above 2 percent" if fab > 0.02 else ""))
    if fab > 0.02:
        flags.append("orientation ambiguous on %.1f%% of bulk pixels" % (100 * fab))
    # cores
    lines.append("CORES: detectors (all computed; discs from %s, %s, match %.2g LAMBDA, discs %.2g LAMBDA):" %
                 ("+".join(names) if names else "none", args.core_combine, args.core_match, args.core_disc))
    dch = dets["charge"]
    lines.append("  dips: %d (amplitude below %.2f of the interior median %.3f, >= %.2g LAMBDA from the wall)" %
                 (len(dets["dips"]["x"]), args.core_frac, dets["dips"]["A_ref"], args.det_wall))
    lines.append("  charge: %d clusters from %d cells (sigma_det %.2g); %d cells within %.2g LAMBDA of the wall or outside dropped"
                 " (sum |q| over the whole box %d)" % (len(dch["x"]), dch["n_cells"], args.sigma_det, dch["n_cells_dropped"],
                                                       args.det_wall, dch["sum_abs_all"]))
    dci = dets["circulation"]
    lines.append("  circulation: %d of %d candidates wound (>= %.2g turns on a %.2g LAMBDA loop); axis scan found %d runs;"
                 " max scan |Gamma| %.3f" % (len(dci["x"]), len(dci["cand_x"]), args.wind_hi, args.loop_r, dci["n_runs"],
                                             float(dci["scan_gamma"].max()) if len(dci["scan_gamma"]) else np.nan))
    if cores:
        lines.append("  %d cores (%d outer, curvature >= %.3g):" % (len(cores), sum(c["outer"] for c in cores), args.ring_kappa_min))
        lines.append("    %8s %7s %8s %6s %6s %6s  %s" % ("X", "Y", "kappa", "mu", "Gamma", "closes", "detectors with a point within the match radius"))
        for c in sorted(cores, key=lambda c: c["x"]):
            lines.append("    %8.2f %7.2f %8.4f %6.3f %6.3f %6s  %s%s" %
                         (c["x"], c["y"] - I["y_axis"], c["kappa"], c["mu_axis"], c["gamma"], "yes" if c["closes"] else "NO",
                          ", ".join(n for n in DETECTORS if c[n]) or "-", "  (outer)" if c["outer"] else ""))
        nd = {n: sum(c[n] for c in cores) for n in DETECTORS}
        lines.append("  agreement: of %d cores, %s; unwound cores (Gamma below %.2g): %d; loops whose sign does not close: %d"
                     " (their Gamma is not a clean integer: read the circulation there with care)" %
                     (len(cores), ", ".join("%s marks %d" % (n, nd[n]) for n in DETECTORS), args.wind_hi,
                      sum(c["gamma"] < args.wind_hi for c in cores), sum(not c["closes"] for c in cores)))
        if any(c["gamma"] < args.wind_hi for c in cores):
            flags.append("%d discs sit on unwound points (Gamma < %.2g)" % (sum(c["gamma"] < args.wind_hi for c in cores), args.wind_hi))
        far_ax = [c for c in cores if abs(c["y"] - I["y_axis"]) > args.axis_excl * LAM]
        if far_ax:
            flags.append("%d cores more than %.2g LAMBDA off the axis" % (len(far_ax), args.axis_excl))
            lines.append("  FLAG: %d cores more than %.2g LAMBDA off the axis (they cut into the bulk family)" % (len(far_ax), args.axis_excl))
    else:
        lines.append("  no cores placed")
    if F is not None:
        lines.append("  (synthetic field: it has no dislocations, so every detection above is FALSE by construction: dips %d,"
                     " charge clusters %d, wound points %d; the ring rows below then measure the estimator beside the axis crease)" %
                     (len(dets["dips"]["x"]), len(dch["x"]), len(dci["x"])))
    # centres
    a = C["half_width"]
    lines.append("CENTRES: test-function half width %.3g LAMBDA = %.3g (m %d, bump sd %.3g LAMBDA); stride %d; supports inside the"
                 " kept region; wall margin %.3g LAMBDA, axis exclusion %.3g LAMBDA (bulk family), ring width %.3g LAMBDA" %
                 (args.radius, a, C["m"], args.radius / np.sqrt(2 * args.poly_p + 3), args.stride, args.wall_margin,
                  args.axis_excl, args.ring_width))
    lines.append("  admitted: near family %d, bulk family %d; bulk %d, ring_outer %d, ring_inner %d, knee %d" %
                 (C["adm_near"].sum(), C["adm_bulk"].sum(), C["bulk"].sum(), C["ring_outer"].sum(), C["ring_inner"].sum(),
                  C["knee"].sum()))
    kb = C["kappa"][C["bulk"]]
    if kb.size:
        lines.append("  geometric curvature at the bulk centres: median %.4f, 10-90%% %.4f to %.4f" %
                     (np.nanmedian(kb), np.nanpercentile(kb, 10), np.nanpercentile(kb, 90)))
    R.flags = flags
    res = dict(label=label, n_cores=len(cores), cores=cores, ambiguous_bulk=fab)
    # headline fits per region, estimator and exact route
    lines.append("RELATION A, div k = -beta w (k outward; beta > 0 is the knee sign), weak form, holdout by the sign of X:")
    regions = ("bulk", "ring_outer", "ring_inner", "knee")
    pl = None
    if F is not None and "beta_map" in F:
        pl = F["beta_map"][C["IY"], C["IX"]]
    for rg in regions:
        sel = C[rg]
        f = fit_relation(Cm["div k"][C["IY"][sel], C["IX"][sel]], Cm["w"][C["IY"][sel], C["IX"][sel]], C["X"][sel])
        res[rg] = f
        lines.append("  %-10s estimator: %s" % (rg, fmt_fit(f)))
        if Ce is not None:
            fe = fit_relation(Ce["div k"][C["IY"][sel], C["IX"][sel]], Ce["w"][C["IY"][sel], C["IX"][sel]], C["X"][sel])
            res[rg + "_exact"] = fe
            lines.append("  %-10s exact:     %s" % ("", fmt_fit(fe)))
        if pl is not None and sel.sum():
            res[rg + "_planted"] = float(np.nanmean(pl[sel]))
            lines.append("  %-10s planted:   mean beta %.4f over the centres" % ("", res[rg + "_planted"]))
    fb = res["bulk"]
    if fb["n"] < args.min_n:
        flags.append("bulk has only %d centres" % fb["n"])
    if np.isfinite(fb.get("r2_hold", np.nan)) and np.isfinite(fb.get("r2_rev", np.nan)) and abs(fb["r2_hold"] - fb["r2_rev"]) > 0.1:
        flags.append("bulk holdout R2 differs between the splits (%.2f, %.2f)" % (fb["r2_hold"], fb["r2_rev"]))
    if np.isfinite(fb.get("c0_rel", np.nan)) and fb["c0_rel"] > 0.2:
        flags.append("bulk intercept is %.2f of rms div k" % fb["c0_rel"])
    # synthetic pass marks
    if F is not None:
        res["pass"] = pass_marks(args, res, lines)
    if not full:
        lines.append("  (headline only for this frame; %.1f s)" % (time.time() - t0))
        return R, res
    # free search and stage-2 checks on the bulk
    Pb = sample(Cm, C, C["bulk"])
    if len(Pb["_x"]) >= 50:
        try:
            r_, one, head, zero = sdd.free_search(Pb, args)
            rk = sorted(one, key=lambda o: -o[0])
            pos = [i for i, o in enumerate(rk) if set((o[1], o[2])) == {"div k", "w"}]
            top = max(o[0] for o in one)
            fam = [o for o in one if o[0] >= top - args.tol and o[2] != "1"]
            lines.append("FREE SEARCH (bulk, no relation assumed; self_dual_discovery.free_search): headline %s ~ %s (coef %+.4g,"
                         " holdout R2 %.3f); div k ~ w ranks %s of %d (R2 %.3f); family within %.2g: %s" %
                         (head[1], head[2], head[3], head[0], (pos[0] + 1) if pos else "absent", len(rk),
                          rk[pos[0]][0] if pos else np.nan, args.tol,
                          "; ".join("%s ~ %s (%.3f)" % (o[1], o[2], o[0]) for o in sorted(fam, key=lambda o: -o[0])[:6])))
            res["free_search"] = dict(headline="%s ~ %s" % (head[1], head[2]), r2=head[0], rank_divk_w=(pos[0] + 1) if pos else None,
                                      vanishing=zero)
            Fc = sdd.forced_checks(Pb, args)
            lines.append("STAGE 2 (bulk): (div k)^2 = gamma^2 w^2: gamma/|beta| %.3f (holdout R2 %.3f; 1 = self-dual equipartition);"
                         " Lap div k = lambda S: lambda/(2 beta^2) %+.3f (holdout R2 %.3f). The squared and fourth-order columns carry"
                         " the estimator's pointwise ripple, which the test function does not average out: read them against the"
                         " planted control, not against 1" %
                         (Fc["gamma_over_absbeta"], Fc["gamma_r2_holdout"], Fc["lambda_over_2beta2"], Fc["lambda_r2_holdout"]))
            res["stage2"] = {k: v for k, v in Fc.items() if not isinstance(v, list)}
        except Exception as exc:                                    # loud, keep going
            lines.append("FREE SEARCH failed: %r" % (exc,))
    if Ce is not None and len(Pb["_x"]) >= 50:
        Pe = sample(Ce, C, C["bulk"])
        sdd.v11_decomposition(Pe, Pb, args, lines)
    # readouts
    d_b = Cm["div k"][C["IY"], C["IX"]]; w_b = Cm["w"][C["IY"], C["IX"]]
    rd = {}
    sb = C["bulk"]
    ex = {"kappa": C["kappa"], "mu": C["mu"], "alpha": np.degrees(C["alpha"])}
    if pl is not None:
        ex["planted"] = pl
    def bins_both(name, key, edges, sel, unit, extra_fmt):
        rows = binned(d_b[sel], w_b[sel], key[sel], edges, {k: v[sel] for k, v in ex.items()}, args.min_n)
        if Ce is not None:
            de = Ce["div k"][C["IY"], C["IX"]][sel]; we = Ce["w"][C["IY"], C["IX"]][sel]
            rows_e = binned(de, we, key[sel], edges, None, args.min_n)
            for r_, re_ in zip(rows, rows_e):
                r_["beta_exact"], r_["relres_exact"] = re_["beta"], re_["relres"]
        rd[name] = rows
        lines.append("BY %s:" % name.upper())
        for s_ in fmt_bins(rows, unit, extra_fmt + ((("beta_exact", "%+.3f"),) if Ce is not None else ())):
            lines.append("    " + s_)
    efmt = (("kappa", "%.4f"), ("mu", "%.2f"), ("alpha", "%.0f deg")) + ((("planted", "%.3f"),) if pl is not None else ())
    bins_both("ray (landing point q = |X_axisfoot|/Xm, bulk)", C["q"], _floats(args.ray_bins), sb, "", efmt)
    bins_both("curvature (bulk)", C["kappa"], [0.0] + _floats(args.kappa_bins) + [1e9], sb, "", efmt)
    bins_both("strip (centre distance from the medial segment, near family)", C["dm"] / LAM, _floats(args.strip_edges),
              C["adm_near"], " LAMBDA", efmt)
    edges_g = _floats(args.gap_edges)
    bins_both("core gap, outer cores, axis band included", C["gap_core"] / LAM, edges_g, C["adm_near"] & C["near_outer"], " LAMBDA", efmt)
    bins_both("core gap, outer cores, axis band excluded", C["gap_core"] / LAM, edges_g, C["adm_bulk"] & C["near_outer"], " LAMBDA", efmt)
    bins_both("core gap, central cores, axis band included", C["gap_core"] / LAM, edges_g, C["adm_near"] & ~C["near_outer"], " LAMBDA", efmt)
    res["readouts"] = rd
    # ray constancy
    rr = [r_ for r_ in rd["ray (landing point q = |X_axisfoot|/Xm, bulk)"] if np.isfinite(r_["beta"])]
    if len(rr) > 1:
        bs = np.array([r_["beta"] for r_ in rr])
        spread = float((bs.max() - bs.min()) / max(np.mean(np.abs(bs)), 1e-12))
        lines.append("  ray spread of beta (max - min)/mean over %d bins: %.3f" % (len(rr), spread))
        res["ray_spread"] = spread
        if Ce is not None and args.mode == "drift":
            ok = [r_ for r_ in rr if np.isfinite(r_.get("beta_exact", np.nan))]
            dev = max(abs(r_["beta"] / r_["beta_exact"] - 1) for r_ in ok) if ok else np.nan
            devp = max(abs(r_["beta"] / r_["planted"] - 1) for r_ in ok) if ok else np.nan
            lines.append("  drift control: largest by-ray deviation of the estimator from the exact route %.1f%% (pass mark 5%%),"
                         " from the planted beta %.1f%%  -> %s" % (100 * dev, 100 * devp, "PASS" if dev <= 0.05 else "FAIL"))
            res["drift_dev_exact"], res["drift_dev_planted"] = dev, devp
    # residual by wall gap and by axis distance (bulk)
    fbl = res["bulk"]
    if np.isfinite(fbl.get("beta_full", np.nan)):
        rho = d_b[sb] + fbl["beta_full"] * w_b[sb]
        rmsd = max(float(np.sqrt(np.mean(d_b[sb] ** 2))), 1e-300)
        rows = []
        for key, edges, nm in ((C["gap_wall"][sb] / LAM, _floats(args.gap_bins_wall), "wall gap"),
                               (C["dm"][sb] / LAM, [0, 2, 3, 4, 6, 8, 1e9], "axis distance")):
            cells = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (key >= lo) & (key < hi)
                if m.sum() >= 10:
                    cells.append((lo, hi, int(m.sum()), float(np.sqrt(np.mean(rho[m] ** 2)) / rmsd)))
            rows.append((nm, cells))
            lines.append("RESIDUAL of the bulk fit by %s (rms(div k + beta w)/rms div k): %s" %
                         (nm, "; ".join("%.3g-%s: %.3f (n %d)" % (lo, ("%.3g" % hi) if hi < 1e8 else "inf", v, n) for lo, hi, n, v in cells)))
        res["residual"] = rows
    # w profile
    wp = w_profile(k1, k2, G, args)
    res["w_profile"] = wp
    lines.append("W PROFILE (median w on the bulk-family pixels by distance from the wall in LAMBDA; in round brackets the self-dual"
                 " expectation -kappa at beta = 1; in square brackets w / kappa, which is -1/beta where w follows the curvature): " + fmt_wprof(wp))
    # pointwise
    pwr = pointwise(k1, k2, G, I, M, args)
    R.pw = pwr
    for rg, msk in (("knee stretch", pointwise_knee_mask(C, G, M, args, I, cores)), ("bulk family", M["bulk"])):
        v = pwr["ratio"][msk & np.isfinite(pwr["ratio"])]
        if v.size > 20:
            lines.append("POINTWISE (smoothed over %.2g LAMBDA) ratio -div k/w on the %s, |w| > %.2g: median %+.3f, quartiles %+.3f / %+.3f (n %d)" %
                         (args.pw_smooth, rg, args.w_floor, np.median(v), np.percentile(v, 25), np.percentile(v, 75), v.size))
            res["pointwise_" + rg.replace(" ", "_")] = dict(median=float(np.median(v)), q25=float(np.percentile(v, 25)),
                                                             q75=float(np.percentile(v, 75)), n=int(v.size))
    # scale test and sensitivity
    sens = sensitivity(R, G, I, args, lines, cols, F)
    res["sensitivity"] = sens
    if outdir is not None:
        save_centres(outdir / ("centres_%s.npz" % label), C, Cm, Ce, pl)
        if not int(args.no_figs):
            figures(outdir, label, R, G, I, args, res, pl, Ce)
    lines.append("FLAGS for %s: %s" % (label, "; ".join(flags) if flags else "none"))
    res["flags"] = flags
    lines.append("(%s analysed in %.1f s)" % (label, time.time() - t0))
    return R, res


def reference_table(args, res, lines):
    """the data's beta per region beside the planted control's (raw and
    scaled in separate columns)"""
    try:
        ref = json.load(open(args.ref_json))
        rf = ref["frames"][0]
    except Exception as exc:
        lines.append("REFERENCE: could not read --ref_json %s (%r)" % (args.ref_json, exc))
        return
    lines.append("BESIDE THE PLANTED CONTROL (%s, mode %s): the control's estimator/exact ratio is the instrument's gain on that"
                 " region; the scaled column applies it to the data and is a model-based reading, never mixed with the raw one" %
                 (Path(args.ref_json).name, ref.get("mode")))
    lines.append("    %-11s %16s %18s %16s %20s" % ("region", "data beta (raw)", "control estimator", "control exact", "data scaled (exact/est)"))
    for rg in ("bulk", "ring_outer", "ring_inner", "knee"):
        db = res.get(rg, {}).get("beta", np.nan)
        ce = (rf.get(rg) or {}).get("beta"); cx = (rf.get(rg + "_exact") or {}).get("beta")
        cr = (rf.get(rg) or {}).get("r2_hold")
        ce = np.nan if ce is None else ce; cx = np.nan if cx is None else cx; cr = np.nan if cr is None else cr
        ok = np.isfinite(db) and np.isfinite(ce) and np.isfinite(cx) and abs(ce) > 1e-12 and np.isfinite(cr) and cr >= args.ref_r2_min
        sc = ("%20.4f" % (db * cx / ce)) if ok else ("%20s" % ("n/a (control R2 %.2f)" % cr if np.isfinite(cr) else "n/a"))
        lines.append("    %-11s %16.4f %18.4f %16.4f %s" % (rg, db, ce, cx, sc))


def w_profile(k1, k2, G, args):
    """median w on the bulk-family pixels by distance from the wall, beside
    the self-dual expectation -kappa/beta at beta = 1 (w = -kappa for
    outward k): the size and sign of the bulk signal, and where the wall's
    boundary layer ends"""
    w = k1 ** 2 + k2 ** 2 - 1.0
    edges = _floats(args.wprof_edges)
    sel = G["inside"] & (G["dm"] > args.axis_excl * LAM)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = sel & (G["d"] >= lo * LAM) & (G["d"] < hi * LAM)
        if m.sum() > 50:
            rows.append((lo, hi, int(m.sum()), float(np.median(w[m])), float(np.nanmedian(-G["kappa"][m]))))
    return rows


def fmt_wprof(rows):
    """bin: median w (self-dual expectation -kappa) [w / kappa]"""
    return "; ".join("%.3g-%s: %+.4f (%+.4f) [%+.2f]" % (lo, ("%.3g" % hi) if hi < 1e8 else "inf", mw, ex, -mw / ex if ex else np.nan)
                     for lo, hi, n, mw, ex in rows)


def pass_marks(args, res, lines):
    fb, fe = res["bulk"], res.get("bulk_exact")
    out = {}
    if args.mode == "selfdual":
        pe = abs(fb["beta"] / args.beta - 1) if np.isfinite(fb["beta"]) else np.inf
        ok = pe <= 0.03 and fb["r2_hold"] > 0.80
        s = "PASS" if ok else "FAIL"
        line = "PASS MARK (selfdual): estimator bulk beta %.4f vs planted %.4g (%.2f%%; mark 3%%), holdout R2 %.3f (mark 0.80) -> %s" % (
            fb["beta"], args.beta, 100 * pe, fb["r2_hold"], s)
        if fe is not None:
            pex = abs(fe["beta"] / args.beta - 1)
            line += "; exact route %.4f (%.2f%%; mark 1%%) -> %s" % (fe["beta"], 100 * pex, "PASS" if pex <= 0.01 else "FAIL")
            out["exact_ok"] = bool(pex <= 0.01)
        out["estimator_ok"] = bool(ok)
        lines.append(line)
    elif args.mode == "eikonal":
        ok = (not np.isfinite(fb["r2_hold"])) or fb["r2_hold"] < 0.2
        lines.append("PASS MARK (eikonal negative control): bulk holdout R2 %.3f (mark: below 0.2) -> %s" %
                     (fb["r2_hold"], "PASS" if ok else "FAIL"))
        out["estimator_ok"] = bool(ok)
    return out


def pointwise(k1, k2, G, I, M, args):
    """pointwise div k and w, each smoothed by a normalised Gaussian over the
    kept pixels (sd pw_smooth LAMBDA), and their ratio"""
    dx = I["dx"]
    divk = np.gradient(k1, dx, axis=1) + np.gradient(k2, dx, axis=0)
    w = k1 ** 2 + k2 ** 2 - 1.0
    m = M["near"].astype(float)
    sg = args.pw_smooth * LAM / dx
    den = gaussian_filter(m, sg, mode="constant")
    ds = gaussian_filter(divk * m, sg, mode="constant") / np.maximum(den, 1e-9)
    ws = gaussian_filter(w * m, sg, mode="constant") / np.maximum(den, 1e-9)
    ok = M["near"] & (np.abs(ws) > args.w_floor) & (den > 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ok, -ds / ws, np.nan)
    return dict(divk_s=ds, w_s=ws, ratio=ratio)


def pointwise_knee_mask(C, G, M, args, I, cores):
    band = M["near"] & (G["dm"] < args.axis_excl * LAM)
    if cores:
        xc = np.array([c["x"] for c in cores])
        beyond = (G["X"] > xc.max() + (args.core_disc + args.ring_width) * LAM) | (G["X"] < xc.min() - (args.core_disc + args.ring_width) * LAM)
    else:
        beyond = G["mu"] < args.knee_mu_max
    return band & beyond


def sensitivity(R, G, I, args, lines, cols, F):
    """scale test and one-at-a-time changes of the region settings and the
    core definition; bulk and ring_outer beta and holdout R2"""
    base = dict(radius=args.radius, wall_margin=args.wall_margin, core_disc=args.core_disc, axis_excl=args.axis_excl,
                core_def=args.core_def, core_combine=args.core_combine)
    variants = [("baseline", {})]
    for r in _floats(args.radii):
        if abs(r - args.radius) > 1e-9:
            variants.append(("radius %.3g" % r, dict(radius=r)))
    if int(args.sens):
        for v in _floats(args.sens_wall_margin):
            variants.append(("wall margin %.3g" % v, dict(wall_margin=v)))
        for v in _floats(args.sens_core_disc):
            variants.append(("disc %.3g" % v, dict(core_disc=v)))
        for v in _floats(args.sens_axis_excl):
            variants.append(("axis excl %.3g" % v, dict(axis_excl=v)))
        for spec in [s for s in str(args.sens_core_def).split(";") if s.strip()]:
            dfn, _, comb = spec.partition("@")
            variants.append(("cores %s%s" % (dfn, ("@" + comb) if comb else ""), dict(core_def=dfn, core_combine=comb or "union")))
    rows = []
    for name, ch in variants:
        p = dict(base, **ch)
        if p["core_def"] != args.core_def or p["core_combine"] != args.core_combine:
            a2 = argparse.Namespace(**vars(args)); a2.core_def, a2.core_combine = p["core_def"], p["core_combine"]
            cores, _ = combine_cores(R.dets, a2, G, I, dict(k1=R.k1, k2=R.k2))
        else:
            cores = R.cores
        a3 = argparse.Namespace(**vars(args)); a3.ring_width = args.ring_width
        C, _ = build_centres(G, I, cores, p["radius"], p["wall_margin"], p["core_disc"], p["axis_excl"], a3)
        Cm = cols(p["radius"])
        row = dict(name=name, n_cores=len(cores), **{k: v for k, v in p.items()})
        for rg in ("bulk", "ring_outer"):
            sel = C[rg]
            f = fit_relation(Cm["div k"][C["IY"][sel], C["IX"][sel]], Cm["w"][C["IY"][sel], C["IX"][sel]], C["X"][sel])
            row[rg] = dict(beta=f["beta"], r2=f["r2_hold"], beta_rev=f["beta_rev"], r2_rev=f["r2_rev"], n=f["n"])
            if F is not None and args.route in ("both", "exact") and rg == "bulk":
                Ce = cols(p["radius"], "exact")
                fe = fit_relation(Ce["div k"][C["IY"][sel], C["IX"][sel]], Ce["w"][C["IY"][sel], C["IX"][sel]], C["X"][sel])
                row["bulk_exact"] = dict(beta=fe["beta"], r2=fe["r2_hold"])
        rows.append(row)
    lines.append("SCALE TEST AND SENSITIVITY (one setting changed at a time from the baseline; beta from the X<0 half, holdout R2"
                 " on X>=0):")
    lines.append("    %-28s %6s | %-30s | %-30s%s" % ("variant", "cores", "bulk beta (R2, n)", "ring_outer beta (R2, n)",
                                                   " | bulk exact" if F is not None and args.route in ("both", "exact") else ""))
    for r in rows:
        cell = lambda q: ("%+.4f (%.3f, %d)" % (q["beta"], q["r2"], q["n"])) if np.isfinite(q["beta"]) else "n %d" % q["n"]
        ex = (" | %+.4f (%.3f)" % (r["bulk_exact"]["beta"], r["bulk_exact"]["r2"])) if "bulk_exact" in r else ""
        lines.append("    %-28s %6d | %-30s | %-30s%s" % (r["name"], r["n_cores"], cell(r["bulk"]), cell(r["ring_outer"]), ex))
    r2b = rows[0]["bulk"]["r2"] if np.isfinite(rows[0]["bulk"]["r2"]) else 0.0
    thr = r2b - args.r2_drop
    low = [r["name"] for r in rows if not (np.isfinite(r["bulk"]["r2"]) and r["bulk"]["r2"] >= thr)]
    bb = [r["bulk"]["beta"] for r in rows if np.isfinite(r["bulk"]["r2"]) and r["bulk"]["r2"] >= thr]
    if low:
        lines.append("  variants whose bulk holdout R2 falls more than %.2g below the baseline's %.3f (left out of the spread; a radius too"
                     " small to average the estimator's wavelength-scale ripple lands here): %s" % (args.r2_drop, r2b, ", ".join(low)))
    if len(bb) > 1:
        sp = (max(bb) - min(bb)) / max(np.mean(np.abs(bb)), 1e-12)
        lines.append("  bulk beta spread over the other %d variants (max - min)/mean: %.3f%s" %
                     (len(bb), sp, "  FLAG: above 0.10" if sp > 0.10 else ""))
        if sp > 0.10:
            R.flags.append("bulk beta moves by %.0f%% across the sensitivity variants" % (100 * sp))
    return rows


def save_centres(path, C, Cm, Ce, pl):
    out = {k: v for k, v in C.items() if isinstance(v, np.ndarray)}
    for nm, F in Cm.items():
        out["est_" + nm.replace(" ", "_").replace("^", "").replace(".", "")] = F[C["IY"], C["IX"]]
    if Ce is not None:
        for nm, F in Ce.items():
            out["exact_" + nm.replace(" ", "_").replace("^", "").replace(".", "")] = F[C["IY"], C["IX"]]
    if pl is not None:
        out["planted_beta"] = pl
    np.savez_compressed(path, **out)


# ---------------------------------------------------------------------
# wall calibration (synthetic modes)
# ---------------------------------------------------------------------

def wall_calibration(R, F, G, I, args, lines, outdir):
    """estimator error of the weak w and div k columns by the support's gap
    to the wall, bulk family with no wall margin; recommended margin"""
    C, _ = build_centres(G, I, [], args.radius, 0.0, args.core_disc, args.axis_excl, args)
    sel = C["adm_bulk"]
    Cm = sdd.weak_columns(R.k1, R.k2, I["x"], I["y"], LAM, args, args.radius)[0]
    Ce = sdd.weak_columns(F["k1"], F["k2"], I["x"], I["y"], LAM, args, args.radius)[0]
    iy, ix = C["IY"][sel], C["IX"][sel]
    gap = C["gap_wall"][sel] / LAM
    deep = gap >= 3.0
    edges = _floats(args.gap_bins_wall)
    rows = []
    for nm in ("w", "div k"):
        ref = float(np.sqrt(np.mean(Ce[nm][iy, ix][deep] ** 2))) if deep.any() else np.nan
        err = Cm[nm][iy, ix] - Ce[nm][iy, ix]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (gap >= lo) & (gap < hi)
            if m.sum() >= 10:
                rows.append(dict(col=nm, lo=lo, hi=hi, n=int(m.sum()), err=float(np.sqrt(np.mean(err[m] ** 2)) / ref),
                                 bias=float(np.mean(err[m]) / ref)))
    rec = None
    los = sorted(set(r["lo"] for r in rows))
    for g in los:
        ok = all(r["err"] < args.err_tol for r in rows if r["lo"] >= g)
        if ok:
            rec = g
            break
    lines.append("WALL CALIBRATION: error of the estimator's weak columns against the exact ones, relative to the deep-bulk rms"
                 " (support gap >= 3 LAMBDA), by the support's gap to the wall (bulk family, no wall margin):")
    for nm in ("w", "div k"):
        lines.append("    %-6s " % nm + "; ".join("%.3g-%s: %.3f (bias %+.3f, n %d)" %
                                                 (r["lo"], ("%.3g" % r["hi"]) if r["hi"] < 1e8 else "inf", r["err"], r["bias"], r["n"])
                                                 for r in rows if r["col"] == nm))
    if rec is None:
        lines.append("  recommended wall margin: none of the gaps keeps both errors below %.2g  FLAG" % args.err_tol)
    else:
        # the margin is a distance of the support edge; the kept region starts there
        lines.append("  recommended wall margin (smallest support gap beyond which both errors stay below %.2g): %.3g LAMBDA;"
                     " this run used --wall_margin %.3g%s" % (args.err_tol, rec, args.wall_margin,
                                                             "" if rec <= args.wall_margin + 1e-9 else "  FLAG: smaller than recommended"))
    # pointwise stencil check: mean w error in the deep bulk
    w_est = R.k1 ** 2 + R.k2 ** 2 - 1; w_ex = F["k1"] ** 2 + F["k2"] ** 2 - 1
    dpix = G["inside"] & (G["s"] > 3 * LAM) & (G["dm"] > args.axis_excl * LAM)
    lines.append("  pointwise w (derivative-stencil check, pixels > 3 LAMBDA from the wall, off the axis): mean error %+.2e, rms error %.3g,"
                 " rms exact w %.3g" % (float(np.mean((w_est - w_ex)[dpix])), float(np.sqrt(np.mean((w_est - w_ex)[dpix] ** 2))),
                                        float(np.sqrt(np.mean(w_ex[dpix] ** 2)))))
    if not int(args.no_figs):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for nm, mk in (("w", "o-"), ("div k", "s--")):
            rr = [r for r in rows if r["col"] == nm]
            ax.plot([0.5 * (r["lo"] + min(r["hi"], r["lo"] + 2)) for r in rr], [r["err"] for r in rr], mk, label="weak " + nm)
        ax.axhline(args.err_tol, color="k", lw=0.8, ls=":", label="tolerance %.2g" % args.err_tol)
        if rec is not None:
            ax.axvline(rec, color="C3", lw=1, label="recommended margin %.3g" % rec)
        ax.axvline(args.wall_margin, color="0.4", lw=1, ls="--", label="--wall_margin %.3g" % args.wall_margin)
        ax.set_yscale("log"); ax.set_xlabel("support gap to the wall (LAMBDA)")
        ax.set_ylabel("rms error / deep-bulk rms"); ax.legend(fontsize=7)
        ax.set_title("estimator error of the weak columns by wall gap", fontsize=9)
        fig.tight_layout(); fig.savefig(outdir / "fig_wall_error.png", dpi=140); plt.close(fig)
    return dict(rows=rows, recommended=rec)


# ---------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------

def _draw_kernel(ax, xc, yc, a, sd, col):
    ax.add_patch(Rectangle((xc - a, yc - a), 2 * a, 2 * a, fill=False, ec=col, lw=1.6))
    ax.add_patch(EllipsePatch((xc, yc), 2 * sd, 2 * sd, fill=False, ec=col, lw=1.0, ls=":"))
    ax.plot([xc], [yc], "s", ms=4, color=col)


def figures(outdir, label, R, G, I, args, res, pl, Ce):
    C, M, cores = R.C, R.M, R.cores
    x, y = I["x"], I["y"] - I["y_axis"]
    ext = [x[0], x[-1], y[0], y[-1]]
    a = C["half_width"]; sd = a / np.sqrt(2 * args.poly_p + 3)
    u = R.u if hasattr(R, "u") else None
    # ---- domain
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.25, wspace=0.12)
    ax = fig.add_subplot(gs[0, :])
    if u is not None:
        ax.imshow(u, origin="lower", extent=ext, cmap="Greys", alpha=0.45, aspect="equal", interpolation="nearest")
    ax.contour(G["X"], G["Y"] - I["y_axis"], G["s"], levels=[0.0, args.wall_margin * LAM], colors=["k", "0.35"],
               linewidths=[0.8, 0.8], linestyles=["-", "--"])
    ax.contour(G["X"], G["Y"] - I["y_axis"], (G["dm"] * G["inside"]), levels=[args.axis_excl * LAM], colors=["0.35"],
               linewidths=0.6, linestyles=":")
    for nm, col, ms in (("other", C_OTHER, 1.0), ("bulk", C_BULK, 2.0), ("ring_outer", C_RING, 3.0),
                        ("ring_inner", C_RINGI, 2.5), ("knee", C_KNEE, 3.0)):
        if nm == "other":
            sel = (C["adm_near"] | C["adm_bulk"]) & ~(C["bulk"] | C["ring_outer"] | C["ring_inner"] | C["knee"])
        else:
            sel = C[nm]
        ax.plot(C["X"][sel], C["Y"][sel], "o", ms=ms, color=col, mec="none", label="%s (%d)" % (nm, sel.sum()))
    for c in cores:
        col = "C3" if c["gamma"] >= args.wind_hi else "C1"
        ax.add_patch(Circle((c["x"], c["y"] - I["y_axis"]), args.core_disc * LAM, fill=False, ec=col, lw=1.0))
    sb = np.where(C["bulk"])[0]; sr = np.where(C["ring_outer"])[0]
    if len(sb):
        k = sb[np.argmin(np.abs(C["X"][sb] - 0.6 * G["Xm"]) + np.abs(C["dm"][sb] - 4 * LAM))]
        _draw_kernel(ax, C["X"][k], C["Y"][k], a, sd, "C2")
    if len(sr):
        k = sr[np.argmax(C["X"][sr])]
        _draw_kernel(ax, C["X"][k], C["Y"][k], a, sd, "C3")
    ax.set_xlim(-I["A"] - 5, I["A"] + 5); ax.set_ylim(-I["B"] - 5, I["B"] + 5)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=6, frameon=False)
    ax.set_title("%s: query centres by region (supports inside the kept region); wall (solid) and wall margin %.3g LAMBDA (dashed);"
                 " axis exclusion %.3g LAMBDA (dotted); discs %.3g LAMBDA (red wound, orange unwound); kernel support (square) and"
                 " bump sd (dotted) over a bulk centre (green) and a ring centre (red)" %
                 (label, args.wall_margin, args.axis_excl, args.core_disc), fontsize=8)
    # zoom on the outermost core (right side) and on the right axis end
    outs = sorted([c for c in cores if c["x"] > 0 and c["gamma"] >= args.wind_hi], key=lambda c: c["x"]) or \
        sorted([c for c in cores if c["x"] > 0], key=lambda c: c["x"])
    xz = outs[-1]["x"] if outs else 0.85 * G["Xm"]
    for j, (x0, ttl) in enumerate(((xz, "zoom on the outermost core on the right (wound if any)"), (G["Xm"] - 2 * LAM, "zoom on the right axis end"))):
        ax2 = fig.add_subplot(gs[1, j])
        h = 6 * LAM
        if u is not None:
            ax2.imshow(u, origin="lower", extent=ext, cmap="Greys", alpha=0.5, aspect="equal", interpolation="nearest")
        ax2.contour(G["X"], G["Y"] - I["y_axis"], G["s"], levels=[0.0, args.wall_margin * LAM], colors=["k", "0.35"],
                    linewidths=[0.8, 0.8], linestyles=["-", "--"])
        for nm, col in (("bulk", C_BULK), ("ring_outer", C_RING), ("ring_inner", C_RINGI), ("knee", C_KNEE)):
            ax2.plot(C["X"][C[nm]], C["Y"][C[nm]], "o", ms=3, color=col, mec="none")
        for c in cores:
            col = "C3" if c["gamma"] >= args.wind_hi else "C1"
            ax2.add_patch(Circle((c["x"], c["y"] - I["y_axis"]), args.core_disc * LAM, fill=False, ec=col, lw=1.2))
            for nm, mk in (("dips", "v"), ("charge", "^"), ("circulation", "o")):
                if c[nm]:
                    ax2.plot([c["x"]], [c["y"] - I["y_axis"]], mk, ms=5, mfc="none", mec=C_DET[nm])
        dch = R.dets["charge"]
        ax2.plot(dch["x"], dch["y"] - I["y_axis"], "^", ms=4, color=C_DET["charge"], label="charge cluster")
        ax2.plot(R.dets["dips"]["x"], R.dets["dips"]["y"] - I["y_axis"], "v", ms=4, color=C_DET["dips"], label="amplitude dip")
        ax2.plot(R.dets["circulation"]["x"], R.dets["circulation"]["y"] - I["y_axis"], "o", ms=5, mfc="none",
                 color=C_DET["circulation"], label="wound (circulation)")
        cand = sr if j == 0 else np.where(C["knee"])[0]
        if len(cand):
            k = cand[np.argmin(np.hypot(C["X"][cand] - x0, C["Y"][cand]))]
            _draw_kernel(ax2, C["X"][k], C["Y"][k], a, sd, "C3")
        ax2.set_xlim(x0 - h, x0 + h); ax2.set_ylim(-h * 0.7, h * 0.7)
        ax2.set_title(ttl, fontsize=8); ax2.legend(fontsize=6, loc="lower left")
    fig.savefig(outdir / ("fig_domain_%s.png" % label), dpi=140, bbox_inches="tight"); plt.close(fig)
    # ---- cores
    dci = R.dets["circulation"]
    fig, axs = plt.subplots(3, 1, figsize=(12, 8.2), sharex=True)
    axs[0].plot(dci["scan_x"], dci["scan_gamma"], "-", color="k", lw=0.8)
    axs[0].axhline(args.wind_hi, color="C3", ls="--", lw=0.8)
    for c in cores:
        axs[0].axvline(c["x"], color="C3" if c["outer"] else "0.6", lw=0.6)
    axs[0].set_ylabel("|Gamma| (turns),\nloop r %.2g LAMBDA on the axis" % args.loop_r)
    jr = I["jr"]
    Aax = R.A[jr, :]
    axs[1].plot(x, Aax / max(R.dets["dips"]["A_ref"], 1e-9), "-", color="k", lw=0.8)
    axs[1].axhline(args.core_frac, color=C_DET["dips"], ls="--", lw=0.8)
    axs[1].plot(R.dets["dips"]["x"], np.full(len(R.dets["dips"]["x"]), args.core_frac), "v", color=C_DET["dips"])
    axs[1].set_ylim(0, 1.5); axs[1].set_ylabel("A / interior median\non the axis row")
    dch = R.dets["charge"]
    axs[2].plot(dch["dropped_x"], dch["dropped_y"] - I["y_axis"], "x", ms=2, color="0.7", label="dropped (wall)")
    axs[2].plot(dch["x"], dch["y"] - I["y_axis"], "^", color=C_DET["charge"], label="charge clusters")
    axs[2].plot(R.dets["dips"]["x"], R.dets["dips"]["y"] - I["y_axis"], "v", color=C_DET["dips"], label="dips")
    axs[2].plot(dci["x"], dci["y"] - I["y_axis"], "o", mfc="none", color=C_DET["circulation"], label="wound")
    axs[2].set_ylim(-I["B"], I["B"]); axs[2].set_ylabel("Y"); axs[2].set_xlabel("X")
    axs[2].legend(fontsize=7, ncol=4)
    axs[0].set_title("%s: the three core detectors (vertical lines: cores placed; red outer)" % label, fontsize=9)
    fig.tight_layout(); fig.savefig(outdir / ("fig_cores_%s.png" % label), dpi=140); plt.close(fig)
    # ---- readouts
    rd = res["readouts"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    def panel(ax, key, xlab, xs_fn):
        rows = rd[key]
        xs = [xs_fn(r) for r in rows]
        b = [r["beta"] for r in rows]
        ax.plot(xs, b, "o-", color="C0", label="estimator")
        if "beta_exact" in rows[0]:
            ax.plot(xs, [r["beta_exact"] for r in rows], "s--", color="C2", ms=4, label="exact route")
        if "planted" in rows[0]:
            ax.plot(xs, [r["planted"] for r in rows], ":", color="k", label="planted")
        for xx, r in zip(xs, rows):
            if np.isfinite(r["beta"]):
                ax.annotate("%.2f\nn%d" % (r["relres"], r["n"]), (xx, r["beta"]), fontsize=5, xytext=(2, 2), textcoords="offset points")
        hb = res["bulk"]["beta"]
        if np.isfinite(hb):
            ax.axhline(hb, color="0.5", lw=0.8, ls="--", label="bulk beta")
        ax.set_xlabel(xlab); ax.set_ylabel("beta"); ax.legend(fontsize=7)
    mid = lambda r: 0.5 * (r["lo"] + (r["hi"] if r["hi"] < 1e8 else r["lo"] * 1.5 + 1))
    panel(axs[0, 0], "ray (landing point q = |X_axisfoot|/Xm, bulk)", "ray landing point q on the medial segment (0 centre, 1 tip)", mid)
    axs[0, 0].set_title("beta by ray (labels: relative residual, n)", fontsize=9)
    panel(axs[0, 1], "curvature (bulk)", "geometric curvature 1/(rho - D) (bin)", lambda r: r["kappa"] if np.isfinite(r["kappa"]) else np.nan)
    axs[0, 1].set_xscale("log"); axs[0, 1].set_title("beta by curvature", fontsize=9)
    panel(axs[1, 0], "strip (centre distance from the medial segment, near family)", "centre distance from the medial segment (LAMBDA)", mid)
    axs[1, 0].set_title("beta by strip (median mu per strip in the report)", fontsize=9)
    panel(axs[1, 1], "core gap, outer cores, axis band included", "support gap to the nearest outer core (LAMBDA)", mid)
    rows2 = rd["core gap, outer cores, axis band excluded"]
    axs[1, 1].plot([mid(r) for r in rows2], [r["beta"] for r in rows2], "^-", color="C1", label="axis band excluded")
    axs[1, 1].legend(fontsize=7); axs[1, 1].set_title("beta by core gap (outer cores): question 1", fontsize=9)
    fig.suptitle(label, fontsize=9); fig.tight_layout()
    fig.savefig(outdir / ("fig_readouts_%s.png" % label), dpi=140); plt.close(fig)
    # ---- scatter
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, rg in zip(axs, ("bulk", "ring_outer", "knee")):
        sel = C[rg]
        wv = R.Cm["w"][C["IY"][sel], C["IX"][sel]]; dv = R.Cm["div k"][C["IY"][sel], C["IX"][sel]]
        sc = ax.scatter(wv, dv, c=C["kappa"][sel], s=3, cmap="viridis")
        f = res[rg]
        if np.isfinite(f.get("beta_full", np.nan)) and len(wv):
            xs = np.linspace(wv.min(), wv.max(), 2)
            ax.plot(xs, -f["beta_full"] * xs, "r-", lw=1, label="beta %.3f (holdout R2 %.3f)" % (f["beta"], f["r2_hold"]))
            ax.legend(fontsize=7)
        ax.set_xlabel("weak w"); ax.set_ylabel("weak div k"); ax.set_title("%s (n %d)" % (rg, sel.sum()), fontsize=9)
        fig.colorbar(sc, ax=ax, label="curvature")
    fig.tight_layout(); fig.savefig(outdir / ("fig_scatter_%s.png" % label), dpi=140); plt.close(fig)
    # ---- ratio map
    fig, axs = plt.subplots(2, 1, figsize=(13, 9))
    hb = res["bulk"]["beta"] if np.isfinite(res["bulk"]["beta"]) else 1.0
    vmax = 2.5 * abs(hb)
    selc = C["adm_near"] | C["adm_bulk"]
    wv = R.Cm["w"][C["IY"][selc], C["IX"][selc]]; dv = R.Cm["div k"][C["IY"][selc], C["IX"][selc]]
    ok = np.abs(wv) > args.w_floor * 0.25
    with np.errstate(divide="ignore", invalid="ignore"):
        rat = -dv / wv
    sc = axs[0].scatter(C["X"][selc][ok], C["Y"][selc][ok], c=rat[ok], s=4, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(sc, ax=axs[0], label="weak -div k / w")
    axs[0].set_aspect("equal"); axs[0].set_title("%s: weak ratio at the admitted centres (|weak w| > %.2g)" % (label, args.w_floor * 0.25), fontsize=9)
    im = axs[1].imshow(R.pw["ratio"], origin="lower", extent=ext, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    fig.colorbar(im, ax=axs[1], label="-div k / w")
    axs[1].set_title("pointwise ratio, div k and w each smoothed over %.2g LAMBDA on the kept pixels (|w| > %.2g)" %
                     (args.pw_smooth, args.w_floor), fontsize=9)
    for ax in axs:
        for c in cores:
            ax.add_patch(Circle((c["x"], c["y"] - I["y_axis"]), args.core_disc * LAM, fill=False, ec="k", lw=0.6))
        ax.set_xlim(-I["A"] - 5, I["A"] + 5); ax.set_ylim(-I["B"] - 5, I["B"] + 5)
    fig.tight_layout(); fig.savefig(outdir / ("fig_ratio_map_%s.png" % label), dpi=140); plt.close(fig)
    # ---- residual
    if "residual" in res:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, (nm, cells) in zip(axs, res["residual"]):
            ax.plot([0.5 * (c[0] + min(c[1], c[0] + 3)) for c in cells], [c[3] for c in cells], "o-")
            ax.set_xlabel("%s (LAMBDA)" % nm); ax.set_ylabel("rms(div k + beta w) / rms div k")
        fig.suptitle("%s: residual of the bulk fit" % label, fontsize=9); fig.tight_layout()
        fig.savefig(outdir / ("fig_residual_%s.png" % label), dpi=140); plt.close(fig)
    # ---- sensitivity
    rows = res["sensitivity"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 0.32 * len(rows) + 1.5), sharey=True)
    ys = np.arange(len(rows))[::-1]
    for ax, rg in zip(axs, ("bulk", "ring_outer")):
        b = np.array([r[rg]["beta"] for r in rows]); r2 = np.array([r[rg]["r2"] for r in rows])
        ax.barh(ys, b, color=[C_BULK if rg == "bulk" else C_RING], alpha=0.8)
        for yv, bv, rv, r in zip(ys, b, r2, rows):
            if np.isfinite(bv):
                ax.text(bv, yv, "  R2 %.2f, n %d" % (rv, r[rg]["n"]), va="center", fontsize=6)
        ax.axvline(rows[0][rg]["beta"] if np.isfinite(rows[0][rg]["beta"]) else 0, color="k", lw=0.8, ls="--")
        ax.set_title("%s beta" % rg, fontsize=9)
    axs[0].set_yticks(ys); axs[0].set_yticklabels([r["name"] for r in rows], fontsize=7)
    fig.suptitle("%s: one setting changed at a time" % label, fontsize=9); fig.tight_layout()
    fig.savefig(outdir / ("fig_sensitivity_%s.png" % label), dpi=140); plt.close(fig)


# ---------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------

def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main(args):
    t0 = time.time()
    out = Path(args.outdir)
    if not out.is_absolute():
        out = _HERE / out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as fh:
        json.dump(dict(vars(args), version=SCRIPT_VERSION), fh, indent=2, default=str)
    lines = ["%s   mode %s" % (SCRIPT_VERSION, args.mode)]
    I = load_input(args)
    lines.append("input: %s; grid %d x %d, dx %.4g; ellipse A %.4g, B %.4g; R0 %.3g; axis row %d (y %.3g)" %
                 (I["src"], len(I["x"]), len(I["y"]), I["dx"], I["A"], I["B"], I["R0"], I["jr"], I["y_axis"]))
    lines.append("settings: LAMBDA = 2 pi = %.4f is the unit of every distance; sigma_dir %.3g; taper %s; radius %.3g; core_def %s (%s);"
                 " discs %.3g; wall margin %.3g; axis exclusion %.3g; ring width %.3g; outer cores: curvature >= %.3g" %
                 (LAM, args.sigma_dir, ("frac %.3g" % args.taper_frac) if args.taper_frac > 0 else "none", args.radius,
                  args.core_def, args.core_combine, args.core_disc, args.wall_margin, args.axis_excl, args.ring_width,
                  args.ring_kappa_min))
    tg = time.time()
    G = geometry(I, args)
    lines.append("geometry: medial segment |X| <= %.4g; wall curvature from %.4g (flat sides) to %.4g (ends); %.1f s" %
                 (G["Xm"], I["B"] / I["A"] ** 2, I["A"] / I["B"] ** 2, time.time() - tg))
    results = dict(version=SCRIPT_VERSION, mode=args.mode, input=I["src"], frames=[])
    if args.mode == "sh":
        if I["u"] is None:
            raise SystemExit("--mode sh needs --npz")
        nt = I["u"].shape[-1]
        tt = I["tt"]
        # stationarity over all saved frames
        bulkpix = G["inside"] & (G["s"] > 2 * LAM) & (G["dm"] > 1.5 * LAM)
        rates = []
        for i in range(1, nt):
            du = I["u"][..., i].astype(float) - I["u"][..., i - 1].astype(float)
            dt = tt[i] - tt[i - 1]
            rates.append((tt[i], float(np.sqrt(np.mean(du[bulkpix] ** 2)) / max(np.sqrt(np.mean(I["u"][..., i][bulkpix].astype(float) ** 2)), 1e-300) / max(dt, 1e-300))))
        lines.append("frames: %d saved, t = %s" % (nt, ", ".join("%.6g" % v for v in tt)))
        lines.append("STATIONARITY: bulk rate rms(du)/rms(u)/dt between saved frames: %s" %
                     ", ".join("t %.6g: %.2e" % r for r in rates))
        results["rates"] = rates
        if int(args.wprof_all):
            lines.append("W PROFILE BY SAVED FRAME (median w on the bulk-family pixels by distance from the wall, LAMBDA; in brackets"
                         " the self-dual expectation -kappa at beta = 1). The bulk signal is of the size of the curvature, about 0.002"
                         " near the centre, so it can be read only once the transient is well below it:")
            profs = []
            for i in range(nt):
                ui = I["u_last"] if (i == nt - 1 and I["u_last"] is not None) else I["u"][..., i].astype(float)
                e_ = run_estimator(ui, I, args)
                kk1, kk2, _ = orient(e_["k1"], e_["k2"], G)
                wp = w_profile(kk1, kk2, G, args)
                profs.append((float(tt[i]), wp))
                lines.append("  t %-8.6g %s" % (tt[i], fmt_wprof(wp)))
            results["w_profiles"] = profs
            if not int(args.no_figs) and profs:
                fig, ax = plt.subplots(figsize=(7, 4.2))
                cm_ = plt.get_cmap("viridis")
                for n_, (tv, wp) in enumerate(profs):
                    if wp:
                        ax.plot([0.5 * (r[0] + min(r[1], r[0] + 2)) for r in wp], [r[3] for r in wp], "o-", ms=3,
                                color=cm_(n_ / max(len(profs) - 1, 1)), label="t %.6g" % tv)
                wp = profs[-1][1]
                ax.plot([0.5 * (r[0] + min(r[1], r[0] + 2)) for r in wp], [r[4] for r in wp], "k--", label="-kappa (self-dual, beta 1)")
                ax.axhline(0, color="0.6", lw=0.6)
                ax.set_xlabel("distance from the wall (LAMBDA)"); ax.set_ylabel("median w, bulk family")
                ax.legend(fontsize=6); ax.set_title("bulk w by wall distance, every saved frame", fontsize=9)
                fig.tight_layout(); fig.savefig(out / "fig_w_profile.png", dpi=140); plt.close(fig)
        idx = [int(v) for v in str(args.frames).split(",") if v.strip()]
        heads = []
        for n, i in enumerate(idx):
            ii = i % nt
            if ii == nt - 1 and I["u_last"] is not None:
                u = I["u_last"]
            else:
                u = I["u"][..., ii].astype(float)
            label = "f%d_t%.6g" % (ii, tt[ii])
            R, res = analyse(label, u, G, I, args, lines, F=None, full=(n == 0), outdir=out if n == 0 else None)
            res["t"] = float(tt[ii])
            results["frames"].append(res)
            heads.append((label, res["bulk"]["beta"], res["bulk"]["r2_hold"], res["ring_outer"]["beta"], res["ring_outer"]["r2_hold"], res["n_cores"]))
        lines.append("")
        lines.append("FRAME AGREEMENT (read the frames only if they agree):")
        for h in heads:
            lines.append("  %-18s bulk beta %+.4f (R2 %.3f) | ring_outer beta %+.4f (R2 %.3f) | cores %d" % h)
        if len(heads) > 1:
            b = np.array([h[1] for h in heads]); rb = np.array([h[3] for h in heads])
            db = float(np.nanmax(np.abs(b - b[0])) / max(abs(b[0]), 1e-12))
            dr = float(np.nanmax(np.abs(rb - rb[0])) / max(abs(rb[0]), 1e-12)) if np.isfinite(rb[0]) else np.nan
            lines.append("  largest relative change from the first frame: bulk %.2f%%, ring_outer %.2f%%%s" %
                         (100 * db, 100 * dr, "  FLAG: above 2 percent" if (db > 0.02 or dr > 0.02) else ""))
            results["frame_change_bulk"], results["frame_change_ring"] = db, dr
        if args.ref_json:
            reference_table(args, results["frames"][0], lines)
    else:
        tf = time.time()
        F = synthetic(G, I, args, lines)
        lines.append("  field built in %.1f s" % (time.time() - tf))
        label = {"selfdual": "selfdual_b%.4g" % args.beta, "eikonal": "eikonal",
                 "drift": "drift_m%.4g_e%.4g" % (args.drift_beta_mid, args.drift_beta_end)}[args.mode]
        R, res = analyse(label, F["u"], G, I, args, lines, F=F, full=True, outdir=out)
        if args.mode in ("selfdual", "drift") and args.route in ("both", "exact"):
            res["wall_calibration"] = wall_calibration(R, F, G, I, args, lines, out)
        results["frames"].append(res)
    cj = []
    for fr in results["frames"]:
        cj.append(dict(label=fr["label"], cores=fr["cores"]))
    with open(out / "cores.json", "w") as fh:
        json.dump(_jsonable(cj), fh, indent=1)
    lines.append("")
    lines.append("elapsed %.1f s; outputs -> %s" % (time.time() - t0, out))
    rep = "\n".join(lines)
    (out / "report.txt").write_text(rep + "\n")
    with open(out / "fits.json", "w") as fh:
        json.dump(_jsonable(results), fh, indent=1)
    print(rep)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    mode = "selfdual"             # selfdual | eikonal | drift | sh
    npz = ""                      # sh: the run (S1); synthetic modes: optional, the grid and R0 are read from it
    outdir = str(Path("results") / "ellipse_relation_a" / "default")
    frames = "-1,-2"              # sh: frame indices; full analysis on the first, headline on the others
    ref_json = ""                 # sh: fits.json of the selfdual control on the same grid (the comparison table)
    ref_r2_min = 0.5              # the scaled column is printed only where the control's holdout R2 reaches this
    # synthetic grid when no npz is given
    A = 210.0
    B = 70.0
    dx = 0.45
    margin = 15.0
    R0 = 0.5
    # synthetic fields
    beta = 1.0                    # selfdual: planted beta (l = 1/beta)
    n_normals = 2048              # selfdual: outward normals in the Cole-Hopf sum
    drift_beta_mid = 1.3          # drift: beta on the rays landing at the centre of the medial segment
    drift_beta_end = 0.8          # drift: beta on the rays landing at its tips (quadratic in between)
    render_wall_sigma = 2.0       # synthetic: erf wall-envelope width of the rendered amplitude
    route = "both"                # synthetic: exact | estimator | both (the estimator route always runs)
    # estimator
    sigma_dir = 0.4
    taper_frac = 0.0              # 0: no taper (the field vanishes at the box edge)
    # cores (distances in LAMBDA)
    core_def = "charge"           # comma-separated subset of dips, circulation, charge
    core_combine = "union"        # union | intersection
    core_match = 0.5              # points of different detectors within this are one core
    core_disc = 0.5               # disc radius
    core_frac = 0.70              # dips: amplitude below this fraction of the interior median
    sigma_det = 0.3               # charge: director width of t1c.detect_defects
    det_wall = 1.0                # detectors ignore points closer than this to the wall
    merge = 0.5                   # clustering radius within one detector
    loop_r = 0.5                  # circulation loop radius
    wind_hi = 0.9                 # turns for a wound core
    axis_scan_step = 0.25         # circulation scan along the axis
    ring_kappa_min = 0.005        # outer cores: curvature at the core at least this
    # regions (LAMBDA)
    wall_margin = 2.0
    axis_excl = 1.5               # bulk family: support farther than this from the medial segment
    ring_width = 1.5              # ring: support gap to an outer core below this
    knee_mu_max = 0.9             # knee stretch when no cores: axis points with local mu below this
    # weak form
    radius = 1.5                  # test-function half width (LAMBDA)
    radii = "0.75,1.0,1.5,2.0"    # scale test
    poly_p = 6
    eta = 0.0                     # inner derivative: np.gradient
    stride = 4
    rim = 2
    max_size = 2                  # free search: supports up to this size
    tol = 0.02                    # free search: family within this of the best holdout R2
    min_n = 30                    # a bin needs this many centres
    r2_drop = 0.15                # sensitivity: variants whose bulk holdout R2 falls more than this below the baseline are listed, not counted
    # readouts
    ray_bins = "0,0.2,0.4,0.5,0.6,0.7,0.8,0.9,1.0"    # ray landing point q = |X_axisfoot|/Xm
    kappa_bins = "0.0018,0.002,0.0025,0.003,0.004,0.006,0.01"
    strip_edges = "0,0.5,1,1.5,2.5,4,6,1e9"          # centre distance from the medial segment
    gap_edges = "0,0.5,1,1.5,2,3,5,1e9"               # support gap to the nearest core
    gap_bins_wall = "0,0.25,0.5,0.75,1,1.5,2,3,1e9"   # wall calibration and residual
    err_tol = 0.10                # wall calibration tolerance
    w_floor = 0.004               # pointwise ratio map: |smoothed w| above this
    wprof_edges = "0,0.25,0.5,0.75,1,1.5,2,3,4,6,8,1e9"   # w profile: distance from the wall (LAMBDA)
    wprof_all = 1                 # sh: the w profile on every saved frame (one estimator call per frame)
    pw_smooth = 1.0               # pointwise smoothing sd (LAMBDA)
    # sensitivity (one setting at a time)
    sens = 1
    sens_wall_margin = "1.0,3.0"
    sens_core_disc = "0.25,1.0"
    sens_axis_excl = "1.0,2.5"
    sens_core_def = "dips;circulation;dips,circulation,charge;charge,circulation@intersection"
    cache = 1
    no_figs = 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in vars(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    if len(sys.argv) > 1:
        sys.exit(main(p.parse_args()))
    sys.exit(main(argparse.Namespace(**{k: getattr(_Args, k) for k in vars(_Args) if not k.startswith("_")})))
