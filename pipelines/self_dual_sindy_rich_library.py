"""
self_dual_sindy_rich_library.py  (v1.0, 27 Oct 2026)
----------------------------------------------------
Does sparse selection still choose the self-dual law div k = b - c w
(w = |k|^2 - |k|_ff^2) when it is offered a richer library that contains
none of the target, and does that hold from shallow knees out to
mu = 0.80, on the profile, on single stripe-period strips, and pointwise?
Routes: uHu (sigma 1.571) and Riesz pi/2 (sigma_dir 0.4, sigma_avg pi/2).

    python self_dual_sindy_rich_library.py --out-root results/1027_selfdual_rich_<what>
        [--roots DIR ... --root-labels L ... --mu-min --mu-max 0.80 --mu-step
         --R ... --R-default 0.5 --routes uhu riesz_pi2
         --domains profile strips pointwise --leaky --coef-cap C
         --survey-table survey_table.csv --no-frame-figures --method-dirs DIR ...]

Plain terms. Every frame is read the way self_dual_sindy_sweep_cropped.py
(v1.4) reads it: same loader rules, same x window (trim to whole stripe
periods), same extraction (knee_sweep_common_cropped.extract_k), same band,
seam and far-field offset (build_profile, build_pointwise), and the same
selection rules (stlsq_floor, cap_prune, fit_target and form_ok are
imported from that script, not copied). What is new:

  1. The library (16 columns, none containing the target). The target is
     div k, which on the profile is d k2bar / dy. Any column built from
     derivatives of w contains the target (w_y = 2 k1 k1_y + 2 k2 k2_y, and
     w_yy contains (k2_y)^2), and so does J = det grad k pointwise; those
     columns are left out of the main library and offered only with
     --leaky, labelled "[L]" everywhere. The columns, by family:
       mismatch   1, w, w^2, w^3, w^4, w^5       (up to --w-max-power)
       odd        k2, k2^3, k2 w                 (odd about the core while
                  div k is even: a working selection must reject them)
       abs        |k2|, |k2| w                   (even, not smooth at the core)
       k1         dk1, dk1^2, dk1 w              (dk1 = k1 - k1_ff)
       curl       curl k, (curl k)^2             (profile: -d k1bar/dy, the
                  x-average of curl k on a periodic crop; pointwise: k2_x - k1_y)
     k2^2 is left out on purpose: with k1 nearly constant it equals
     w + 1 - k1^2, which is w in disguise. The Gram condition number of the
     normalized library is printed per fit, and any column whose norm is
     zero to round-off is dropped and named.
  2. A Pareto curve per fit: forward selection on holdout R^2 (start from
     the constant, add the column that raises the two-fold holdout R^2
     most, repeat to --max-forward terms). A flat curve from one term on
     says the first term carries the law; the first term added and the
     gain of the second are reported.
  3. The strip test: the x-average is repeated on strips one stripe period
     wide (--strip-periods), each fit separately. Spread of c across
     strips and the fraction of strips selecting w alone are reported.
     This answers "is the x-average making the law?": if every strip
     picks w alone with the same c, the average only removes the
     stripe-scale ripple.
  4. A named comparison with RCN's quartic correction. The self-dual
     reduction with the full G^2 = w^2 - w^4 gives
     div k = -beta w sqrt(1 - w^2), about -beta (w - w^3/2). Both
     one-coefficient models, b - c w and b - c w sqrt(1 - w^2), are fit
     and their holdout R^2 compared (d_rcn = R2_rcn - R2_lin). This is a
     test of a stated prediction, not discovery, and is reported apart
     from the selection.
  5. Planted checks, run first on each frame's own grid and band
     (log and CSV only, not results figures):
       (A) the analytic self-dual knee (beta_knee_fit.synth_knee, slaved
           A1 and A3) at planted beta over --planted-ladder (0.9 to 1.8,
           wide enough to reach the beta the data reads at mu 0.8); these
           must read as w alone, and give each route's response curve for
           the calibrated column;
       (B) the same knee with RCN's quartic factor
           (dk2/dy = beta (mu^2 - k2^2) sqrt(1 - (k2^2 - mu^2)^2),
           integrated on the grid, same slaved amplitude) and its plain
           tanh partner, both at the record law's beta for that mu and R.
           Whether the route can tell the two shapes apart is reported
           (rcn_resolvable), and the data's d_rcn is placed between them:
           quartic_index = (d_data - d_tanh) / (d_quartic - d_tanh), 0 for
           "looks like a pure self-dual knee through this route", 1 for
           "looks like the RCN quartic knee". It is only read when the
           route resolves the difference.

Outputs, under --out-root / <stamp>/:
  config.json, report.txt (the console), selfdual_rich.csv (one row per
  frame, route and domain), strips.csv (one row per strip),
  planted_checks.csv; per frame <set>__<stem>/: fig_domain.png (the query
  centres of each domain and the kernel widths drawn over one centre),
  <route>_pareto.png, <route>_corr.png, <route>_profile_fit.png; set
  figures summary_terms.png, summary_r2.png, summary_c_raw.png,
  summary_c_calibrated.png, summary_strips.png, summary_planted.png.

What the profile can and cannot test. On the x-averaged knee w(y) is
monotone in |y| on each side of the core, so EVERY even column (|k2|, dk1,
curl^2, w^p) is itself a function of w there; |k2| is sqrt(w + mu^2) on an
ideal knee. The profile therefore tests the SHAPE of div k as a function
of w (is it linear?), not alternative physics. The odd columns are the
selection's sanity check. Columns independent of w only appear where the
field varies in x (pointwise and strips at intermediate angles, where the
amplitude dips sit off the line).

Form test, as read here. Near-ties between one-term shapes (w against
|k2|) are decided by noise, so the strict STLSQ pick (form_ok, "w alone")
is reported but the test reads ADMISSIBILITY: the one-term law b - c w is
within rel_tol of the best holdout R^2 any candidate reaches (STLSQ pool,
singletons, forward path). Through a smoothed route the planted self-dual
knee itself gains from extra shape columns at larger angles (the kernel
bends the measured div k against w; the measurement-consistent library
finding), so the data are also read against the planted self-dual knee at
the record law's beta through the same route: excess_gain = (library gain
beyond w on the data) - (the same on the planted knee). Near zero: nothing
beyond what the route does to a self-dual knee.

Reading rules (house rules): read by holdout R^2 and selected terms, never
by the values of collinear coefficients (w^2 .. w^5 are near-collinear on
the one-signed w range of a knee, so WHICH power is chosen is not
identifiable; whether one is chosen is). Raw and calibrated c never share
a column or a panel. Flags are loud and never block.

Changelog
v1.0  (27 Oct 2026; built and checked on sandbox stand-ins, not SH data)
  - Library, Pareto, strip test, RCN comparison and planted checks as
    above. Measured on stand-ins on the shallow and large-sweep grid
    layouts (see the results note for the numbers of Ed's run):
    the quartic planter with the square-root factor switched off
    reproduces beta_knee_fit.synth_knee to round-off; odd columns are
    rejected by the two-fold holdout; on a planted self-dual knee uHu
    selects w alone in every domain.
  - Loader: u stored as [ny, nx, nt] or [nt, ny, nx] (the
    mg_zigzag_theta_t.load_npz rule); R from metadata (Rscale, R), then
    the stem, then --R-default with a loud note (the 24 Oct large sweep
    has no R in its stems); the core row from gb_row_in_crop, otherwise
    located from the demodulated k2 profile with a loud note.
  - coef_cap is off by default (None). An absolute cap of 10 removes
    w^4 and w^5 for scale alone (a w^5 term that matters on |w| ~ 0.3
    needs a coefficient of order 0.3^-4 ~ 100), which would bias the test
    toward w; --coef-cap restores it.
  - The energy form (div k)^2 is not fit here (the sweep script has it).
  - Admissibility and the planted-referenced excess (above). Measured on
    stand-ins (analytic knees at the law's beta on the two grid layouts,
    the 0.60 and 0.78 large-grid ones with the quartic factor and 1
    percent noise): at mu 0.375 and 0.475 uHu picks w alone in every
    domain (excess 0.0000), strips spread 0.06 to 0.17 percent; at 0.60
    uHu's strict pick is |k2| on the planted self-dual knee as well as the
    stand-in (w admissible, excess +0.0014 profile); at 0.78 uHu's planted
    response falls with beta (0.43 to 0.17, RESPONSE_NOT_MONOTONE, linear
    R^2 0.90), so nothing there is readable. Riesz pi/2 picks |k2| on
    planted self-dual knees from 0.375 (gain 0.011 to 0.05 over w) and
    reads R^2 0.06 on the planted 0.78 knee; pointwise it reads R^2 0.69
    to 0.80 on planted knees at every angle. Neither route resolves RCN's
    quartic shape from the tanh shape at any angle tested (difference
    below 0.0002); the unsmoothed demod route does (0.60: -0.0005 against
    +0.0005; 0.78: -0.008 against +0.003).
  - Calibration status: CAL_OK only when the planted response rises
    with beta, the planted knees read as linear in w (admissible), and
    the data reading lies inside the planted range.
Python 3.8 compatible; frozen method layer imported, never edited.
"""
import os
import sys
import csv
import json
import glob
import time
import argparse
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.patches import Circle, Patch                 # noqa: E402
from scipy.signal import hilbert                             # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import knee_sweep_common_cropped as ksc                      # noqa: E402
import self_dual_sindy_sweep_cropped as sds                  # noqa: E402

warnings.filterwarnings("ignore")
VERSION = "self_dual_sindy_rich_library v1.0"
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

FAMILY_OF = OrderedDict()          # column name -> family (filled by library())
FAMILY_COLORS = {"mismatch": "#2a6fd6", "odd": "#d1352b", "abs": "#eb8a34",
                 "k1": "#1baf7a", "curl": "#5b3fa8", "leaky": "#77756f"}
PRETTY = {"w": "$w$", "k2": "$k_2$", "k2^3": "$k_2^3$", "k2 w": "$k_2 w$",
          "|k2|": "$|k_2|$", "|k2| w": "$|k_2|\\,w$", "dk1": "$\\delta k_1$",
          "dk1^2": "$\\delta k_1^2$", "dk1 w": "$\\delta k_1\\,w$",
          "curl": "curl $\\mathbf{k}$", "curl^2": "(curl $\\mathbf{k})^2$"}
ROUTE_COLORS = {"uhu": "C2", "riesz_pi2": "C1"}
SET_MARKERS = ["o", "s", "^", "D", "v", "P"]
REGIME_COL = {"knee": "#e9e7e2", "intermediate": "#f6e3b4", "dislocation": "#cfe0f5"}


def pretty(nm):
    if nm.startswith("w^"):
        return "$w^{%s}$" % nm[2:]
    if nm.startswith("[L] "):
        return "[L] " + nm[4:]
    return PRETTY.get(nm, nm)


# ---------------------------------------------------------------------
# loading (both stored layouts; R and core row with loud fallbacks)
# ---------------------------------------------------------------------

def _meta_from_npz(d):
    meta = {}
    for key in ("metadata_json", "sh_meta_json"):
        if key in d.files:
            try:
                meta.update(json.loads(str(d[key])))
            except Exception:
                pass
    return meta


def _u_yx_t(u):
    """[ny, nx, nt] whatever the stored layout (mg_zigzag_theta_t.load_npz rule:
    a last axis shorter than the first is time)."""
    u = np.asarray(u)
    if u.ndim == 2:
        return u[..., None]
    if u.shape[-1] < u.shape[0]:
        return u
    return np.moveaxis(u, 0, -1)


def read_run(path, R_default):
    d = np.load(path, allow_pickle=True)
    meta = _meta_from_npz(d)
    stem = Path(path).stem
    mu_s, R_s = ksc._parse_stem(stem)
    mu = float(meta.get("mu", mu_s if mu_s is not None else np.nan))
    R, R_src = None, None
    for key in ("Rscale", "R"):
        if key in meta:
            try:
                R = float(meta[key]); R_src = "metadata:" + key; break
            except Exception:
                pass
    if R is None and R_s is not None:
        R, R_src = float(R_s), "stem"
    if R is None:
        R, R_src = float(R_default), "DEFAULT"
    shp = d["u"].shape
    if len(shp) == 2:
        ny, nt = shp[0], 1
    elif shp[-1] < shp[0]:
        ny, nt = shp[0], shp[-1]
    else:
        ny, nt = shp[1], shp[0]
    jc = meta.get("gb_row_in_crop", None)
    t = None
    for key in ("tt", "t"):
        if key in d.files:
            t = np.atleast_1d(np.asarray(d[key], float)); break
    return dict(path=str(path), stem=stem, mu=mu, R=R, R_src=R_src,
                jc=(int(jc) if jc is not None else None), ny=int(ny), nt=int(nt), t=t)


def load_frame(run, frame):
    d = np.load(run["path"], allow_pickle=True)
    u = _u_yx_t(d["u"])
    idx = frame if frame >= 0 else u.shape[2] + frame
    t = run["t"][min(idx, run["t"].size - 1)] if run["t"] is not None else np.nan
    return (np.asarray(d["x"], float), np.asarray(d["y"], float),
            np.asarray(u[..., idx], float), int(idx), float(t))


def locate_core_row(u, x, y, margin=24):
    """Row of the knee core from the demodulated x-averaged k2 (lower-knee
    convention: k2 < 0 below the core, > 0 above); the crossing nearest the
    middle of the crop. Used only when gb_row_in_crop is missing."""
    dy = y[1] - y[0]
    ua = hilbert(u, axis=1)
    k2 = np.imag(np.conj(ua) * np.gradient(ua, dy, axis=0)) / (np.abs(ua) ** 2 + 1e-12)
    k2b = k2[:, margin:len(x) - margin].mean(1)
    k2b = np.convolve(k2b, np.ones(5) / 5.0, mode="same")
    up = np.where((k2b[:-1] < 0) & (k2b[1:] >= 0))[0]
    if up.size == 0:
        return len(y) // 2, "no sign change of demod k2bar; middle row used"
    j = int(up[np.argmin(np.abs(up - len(y) // 2))])
    return j, "located from demod k2bar (row %d of %d)" % (j, len(y))


def discover(cfg):
    runs = []
    for root, label in zip(cfg["roots"], cfg["root_labels"]):
        paths = sorted(glob.glob(os.path.join(root, "**", cfg["pattern"]), recursive=True))
        if not paths:
            print("   [discover] nothing under %s" % root)
        for p in paths:
            if any(tag in Path(p).stem for tag in ("_uhu_", "_phase_", "_riesz_")):
                continue
            try:
                info = read_run(p, cfg["R_default"])
            except Exception as exc:
                print("   [discover] skipping %s (%s)" % (p, exc)); continue
            mu = info["mu"]
            if not np.isfinite(mu):
                continue
            if cfg["mu_min"] is not None and mu < cfg["mu_min"] - 1e-9:
                continue
            if cfg["mu_max"] is not None and mu > cfg["mu_max"] + 1e-9:
                continue
            if cfg["mu_step"]:
                q = mu / cfg["mu_step"]
                if abs(q - round(q)) > 1e-6:
                    continue
            if cfg["R_values"] and not any(abs(info["R"] - r) < 1e-6 for r in cfg["R_values"]):
                continue
            info["set"] = label
            runs.append(info)
    runs.sort(key=lambda i: (i["set"], i["R"], i["mu"]))
    return runs


# ---------------------------------------------------------------------
# the RCN quartic knee (planted check B)
# ---------------------------------------------------------------------

def synth_knee_rcn_quartic(x, y, mu, R, beta, yc, quartic=True, n_fine=40001):
    """Knee whose k2 profile solves dk2/ds = beta (mu^2 - k2^2) F, s = |y - yc|,
    k2(0) = 0, with F = sqrt(1 - w^2), w = k2^2 - mu^2 (RCN's self-dual
    reduction with the full G^2 = w^2 - w^4) or F = 1 (quartic=False, the
    tanh knee). The phase is k1 x + int_0^s k2 + log(2)/beta (the constant
    synth_knee carries), the amplitude is slaved exactly as in
    beta_knee_fit.synth_knee. Integrated by RK4 on a fine grid in s."""
    k1 = np.sqrt(1.0 - mu * mu)
    smax = float(np.max(np.abs(y - yc))) + 1.0
    s = np.linspace(0.0, smax, n_fine); h = s[1] - s[0]

    def f(k):
        w = k * k - mu * mu
        F = np.sqrt(np.clip(1.0 - w * w, 0.0, None)) if quartic else 1.0
        return beta * (mu * mu - k * k) * F

    k2 = np.zeros_like(s)
    for i in range(n_fine - 1):
        k = k2[i]
        a = f(k); b = f(k + 0.5 * h * a); c = f(k + 0.5 * h * b); dd = f(k + h * c)
        k2[i + 1] = k + h * (a + 2 * b + 2 * c + dd) / 6.0
    Th = np.concatenate([[0.0], np.cumsum(0.5 * (k2[1:] + k2[:-1]) * h)])
    X, Y = np.meshgrid(x, y)
    S = np.abs(Y - yc)
    th = k1 * X + np.interp(S, s, Th) + np.log(2.0) / beta
    ksq = k1 * k1 + np.interp(S, s, k2) ** 2
    A1 = np.sqrt(np.maximum(4.0 / 3.0 * (R - (ksq - 1.0) ** 2), 0.0))
    A3 = -A1 ** 3 / (4.0 * (R + 79.0 * ksq ** 2 - 14.0 * ksq - 1.0))
    return A1 * np.cos(th) + A3 * np.cos(3 * th)


def beta_law(mu, R):
    c = ksc.RECORD_LAW_C.get(round(R, 6))
    if c is None:
        for r, v in ksc.RECORD_LAW_C.items():
            if abs(r - R) < 1e-6:
                c = v
    return (1.0 + c * mu * mu) if c is not None else float("nan")


# ---------------------------------------------------------------------
# sample sets for the three domains
# ---------------------------------------------------------------------

def profile_samples(k1, k2, x, y, jc, mu, route, cfg):
    prof = ksc.build_profile(k1, k2, x, y, jc, mu, route, cfg["alpha_x"], cfg["y_frac"],
                             cfg["band_w"], cfg["ff_renorm"])
    b = prof["band"]; dy = y[1] - y[0]
    curl = -np.gradient(prof["k1bar"], dy)
    S = dict(w=prof["wbar"][b], k1=prof["k1bar"][b], k2=prof["k2bar"][b], curl=curl[b],
             k1_ff=prof["k1_ff"], target=prof["dk2"][b], yy=prof["y"][b])
    leaky = OrderedDict([("|w_y|", np.abs(prof["w_y"][b])), ("w_y^2", prof["w_y"][b] ** 2),
                         ("w_yy", prof["w_yy"][b])])
    return S, leaky, prof


def strip_columns(prof, x, cfg, mu):
    """Strips of --strip-periods stripe periods inside the flat columns."""
    cols = prof["cols"]
    dx = x[1] - x[0]
    per = 2.0 * np.pi / (ksc.carrier_k1(mu) * dx) * cfg["strip_periods"]
    n_flat = cols.stop - cols.start
    n_str = int(np.floor(n_flat / per))
    if n_str < 2:
        return []
    width = int(round(per))
    start0 = cols.start + (n_flat - n_str * width) // 2
    return [slice(start0 + i * width, start0 + (i + 1) * width) for i in range(n_str)]


def strip_samples(k1, k2, y, prof, cols_s, cfg):
    dy = y[1] - y[0]; b = prof["band"]
    k1b = k1[:, cols_s].mean(1); k2b = k2[:, cols_s].mean(1)
    wb = (k1[:, cols_s] ** 2 + k2[:, cols_s] ** 2).mean(1) - 1.0
    if cfg["ff_renorm"]:
        wb = wb - prof["w_ff"]
    S = dict(w=wb[b], k1=k1b[b], k2=k2b[b], curl=-np.gradient(k1b, dy)[b],
             k1_ff=prof["k1_ff"], target=np.gradient(k2b, dy)[b], yy=prof["y"][b])
    wy = np.gradient(wb, dy)
    leaky = OrderedDict([("|w_y|", np.abs(wy[b])), ("w_y^2", wy[b] ** 2),
                         ("w_yy", np.gradient(wy, dy)[b])])
    return S, leaky


def pointwise_samples(k1, k2, x, y, jc, mu, route, cfg):
    pw = ksc.build_pointwise(k1, k2, x, y, jc, mu, route, cfg["alpha_x"], cfg["y_frac"],
                             cfg["band_w"], cfg["ff_renorm"])
    M = pw["mask"]
    S = dict(w=pw["w"], k1=k1[M], k2=k2[M], curl=pw["extras"]["curl k"],
             k1_ff=pw["prof"]["k1_ff"], target=pw["divk"], yy=pw["y"])
    leaky = OrderedDict([("J", pw["extras"]["J"]), ("Lap w", pw["extras"]["Lap w"]),
                         ("|grad w|^2", pw["extras"]["|grad w|^2"])])
    n = S["w"].size
    if cfg["pointwise_max_samples"] and n > cfg["pointwise_max_samples"]:
        idx = np.linspace(0, n - 1, int(cfg["pointwise_max_samples"])).astype(int)
        S = {k: (v[idx] if isinstance(v, np.ndarray) and v.size == n else v) for k, v in S.items()}
        leaky = OrderedDict((k, v[idx]) for k, v in leaky.items())
    return S, leaky, pw


# ---------------------------------------------------------------------
# library, fits, Pareto
# ---------------------------------------------------------------------

def library(S, cfg, leaky=None):
    """Columns in a fixed order, constant first and w second (form_ok of the
    sweep script reads support[1] as w). Degenerate columns are dropped."""
    w = S["w"]; k2 = S["k2"]; dk1 = S["k1"] - S["k1_ff"]; curl = S["curl"]
    fam = set(cfg["families"])
    C = OrderedDict(); F = OrderedDict()
    C["1"] = np.ones_like(w); F["1"] = "const"
    C["w"] = w; F["w"] = "mismatch"
    if "mismatch" in fam:
        for p in range(2, int(cfg["w_max_power"]) + 1):
            C["w^%d" % p] = w ** p; F["w^%d" % p] = "mismatch"
    if "odd" in fam:
        for nm, col in (("k2", k2), ("k2^3", k2 ** 3), ("k2 w", k2 * w)):
            C[nm] = col; F[nm] = "odd"
    if "abs" in fam:
        for nm, col in (("|k2|", np.abs(k2)), ("|k2| w", np.abs(k2) * w)):
            C[nm] = col; F[nm] = "abs"
    if "k1" in fam:
        for nm, col in (("dk1", dk1), ("dk1^2", dk1 ** 2), ("dk1 w", dk1 * w)):
            C[nm] = col; F[nm] = "k1"
    if "curl" in fam:
        for nm, col in (("curl", curl), ("curl^2", curl ** 2)):
            C[nm] = col; F[nm] = "curl"
    if leaky:
        for nm, col in leaky.items():
            C["[L] " + nm] = col; F["[L] " + nm] = "leaky"
    ref = np.linalg.norm(w) + 1e-300
    dropped = [nm for nm, col in C.items()
               if nm not in ("1", "w") and (not np.all(np.isfinite(col))
                                            or np.linalg.norm(col) < 1e-10 * ref)]
    for nm in dropped:
        C.pop(nm); F.pop(nm)
    FAMILY_OF.update(F)
    return np.column_stack(list(C.values())), list(C.keys()), dropped


def forward_path(Theta, y, folds, names, max_terms):
    p = Theta.shape[1]
    supp = np.zeros(p, dtype=bool); supp[0] = True
    path = [dict(n=0, added="1", r2=sds.holdout_r2(Theta, y, supp, folds))]
    for k in range(1, min(max_terms, p - 1) + 1):
        best = None
        for j in range(1, p):
            if supp[j]:
                continue
            s = supp.copy(); s[j] = True
            r2 = sds.holdout_r2(Theta, y, s, folds)
            if np.isfinite(r2) and (best is None or r2 > best[1]):
                best = (j, r2)
        if best is None:
            break
        supp[best[0]] = True
        path.append(dict(n=k, added=names[best[0]], r2=float(best[1])))
    return path


def fit_samples(S, cfg, leaky=None, n_boot=None, keep_theta=False):
    Theta, names, dropped = library(S, cfg, leaky if cfg["leaky"] else None)
    y = S["target"]; yy = S["yy"]; w = S["w"]
    folds = [(yy < 0, yy > 0), (yy > 0, yy < 0)]
    nb = cfg["n_boot"] if n_boot is None else n_boot
    res = sds.fit_target(Theta, y, folds, cfg["floors"], cfg["rel_tol"], max(nb, 2),
                         0.5, cfg["seed"], cfg["coef_cap"])
    res["names"] = names
    lin = np.zeros(Theta.shape[1], dtype=bool); lin[0] = lin[1] = True
    coef_lin = sds.ols_support(Theta, y, lin)
    out = dict(res=res, names=names, dropped=dropped, n=int(y.size),
               c_raw=float(-coef_lin[1]), b=float(coef_lin[0]),
               b_over_peak=float(coef_lin[0] / (np.abs(y).max() + 1e-300)),
               r2_lin=sds.holdout_r2(Theta, y, lin, folds),
               form_ok=bool(sds.form_ok(res, "divk")), terms=sds.terms_string(res),
               n_terms=int(res["support"][1:].sum()), r2_sel=float(res["r2_holdout"]),
               w_min=float(np.min(w)), w_max=float(np.max(w)),
               cond=float(np.linalg.cond(Theta / (np.linalg.norm(Theta, axis=0) + 1e-300))))
    out["gain_sel"] = out["r2_sel"] - out["r2_lin"]
    rng = np.random.RandomState(cfg["seed"]); n = y.size; m = max(2, n // 2)
    bs = [sds.ols_support(Theta[i], y[i], lin)[1] for i in
          (rng.randint(0, n, size=m) for _ in range(max(nb, 2)))]
    out["c_raw_std"] = float(np.std(bs))
    g = w * np.sqrt(np.clip(1.0 - w * w, 0.0, None))
    Th2 = np.column_stack([np.ones_like(w), g])
    two = np.ones(2, dtype=bool)
    out["c_rcn"] = float(-sds.ols_support(Th2, y, two)[1])
    out["r2_rcn"] = sds.holdout_r2(Th2, y, two, folds)
    out["d_rcn"] = out["r2_rcn"] - out["r2_lin"]
    if cfg["max_forward"] > 0:
        out["path"] = forward_path(Theta, y, folds, names, cfg["max_forward"])
        P = out["path"]
        out["fwd_1"] = P[1]["added"] if len(P) > 1 else ""
        out["fwd_2"] = P[2]["added"] if len(P) > 2 else ""
        out["fwd_gain_2"] = (P[2]["r2"] - P[1]["r2"]) if len(P) > 2 else float("nan")
    else:
        out["path"] = []; out["fwd_1"] = out["fwd_2"] = ""; out["fwd_gain_2"] = float("nan")
    # Is the one-term law in w admissible, i.e. within rel_tol of the best
    # holdout R^2 any candidate reaches (STLSQ pool, singletons, forward
    # path)? On the x-averaged knee every even column is a function of y and
    # hence of w, so near-ties between one-term shapes (w against |k2|, which
    # is sqrt(w + mu^2) on an ideal knee) are decided by noise; the strict
    # STLSQ pick is reported, and admissibility is what the form test reads.
    pool = [s_["r2_holdout"] for s_ in res["sweep"] if np.isfinite(s_["r2_holdout"])]
    pool += [p["r2"] for p in out["path"] if np.isfinite(p["r2"])]
    best = max(pool + [out["r2_lin"]])
    out["r2_best"] = float(best)
    out["w_admissible"] = bool(out["r2_lin"] >= best - cfg["rel_tol"] * abs(best))
    out["gain_best"] = float(best - out["r2_lin"])
    out["resid_ratio"] = float((1.0 - out["r2_lin"]) / max(1.0 - best, 1e-12))
    if keep_theta:
        out["Theta"] = Theta; out["y"] = y; out["yy"] = yy; out["w"] = w
        out["yhat_sel"] = Theta @ res["coef"]
        out["yhat_lin"] = Theta @ coef_lin
        out["yhat_rcn"] = Th2 @ sds.ols_support(Th2, y, two)
    return out


def fit_field(k1, k2, x, y, jc, mu, route, cfg, domains, n_boot=None, keep_theta=False):
    """All requested domains for one extracted field."""
    out = {}
    S, leaky, prof = profile_samples(k1, k2, x, y, jc, mu, route, cfg)
    out["prof"] = prof
    if "profile" in domains:
        out["profile"] = fit_samples(S, cfg, leaky, n_boot, keep_theta)
    if "strips" in domains:
        strips = []
        for cs in strip_columns(prof, x, cfg, mu):
            Ss, lk = strip_samples(k1, k2, y, prof, cs, cfg)
            cfg_s = dict(cfg, max_forward=0)
            f = fit_samples(Ss, cfg_s, lk, n_boot=2)
            f["cols"] = cs
            strips.append(f)
        out["strips"] = strips
    if "pointwise" in domains:
        Sp, lkp, _ = pointwise_samples(k1, k2, x, y, jc, mu, route, cfg)
        out["pointwise"] = fit_samples(Sp, cfg, lkp, n_boot, keep_theta)
    return out


def strip_summary(strips):
    if not strips:
        return dict(n_strips=0, strip_c_mean=np.nan, strip_c_sd=np.nan, strip_spread=np.nan,
                    strip_frac_w_alone=np.nan, strip_frac_w_admissible=np.nan,
                    strip_r2_lin_min=np.nan, strip_terms="")
    cs = np.array([s["c_raw"] for s in strips])
    extra = OrderedDict()
    for s in strips:
        for j in np.where(s["res"]["support"])[0]:
            nm = s["names"][j]
            if nm != "1":
                extra[nm] = extra.get(nm, 0) + 1
    return dict(n_strips=len(strips), strip_c_mean=float(cs.mean()), strip_c_sd=float(cs.std()),
                strip_spread=float(cs.std() / (abs(cs.mean()) + 1e-300)),
                strip_frac_w_alone=float(np.mean([s["form_ok"] for s in strips])),
                strip_frac_w_admissible=float(np.mean([s["w_admissible"] for s in strips])),
                strip_r2_lin_min=float(min(s["r2_lin"] for s in strips)),
                strip_terms=";".join("%s:%d/%d" % (k, v, len(strips)) for k, v in extra.items()))


# ---------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------

def shade_regimes(ax, regimes):
    for lo, hi, lab in regimes:
        ax.axvspan(lo, hi, color=REGIME_COL.get(lab, "w"), lw=0, zorder=0)


def fig_domain(path, u, x, y, jc, prof, strips, cfg, mu, R, stem):
    yc = y - y[jc]; cols = prof["cols"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    ax.imshow(u, extent=[x[0], x[-1], yc[0], yc[-1]], cmap="gray", vmin=-1.2, vmax=1.2,
              aspect="equal", origin="lower", rasterized=True)
    bh, sm = prof["band_half"], prof["seam"]
    for v in (bh, -bh):
        ax.axhline(v, color="#2a6fd6", lw=1.0)
    for v in (sm, -sm):
        ax.axhline(v, color="#2a6fd6", lw=0.7, ls=":")
    ax.axvline(x[cols.start], color="#eb8a34", lw=1.0)
    ax.axvline(x[cols.stop - 1], color="#eb8a34", lw=1.0)
    for i, cs in enumerate(strips):
        ax.axvline(x[cs.start], color="#d1352b", lw=0.8, ls="--")
        if i == len(strips) - 1:
            ax.axvline(x[cs.stop - 1], color="#d1352b", lw=0.8, ls="--")
    x0 = 0.5 * (x[cols.start] + x[cols.stop - 1]); y0 = 0.5 * (bh + sm)
    handles = []
    for route in cfg["routes"]:
        r = ksc.ROUTES[route]
        if r["kind"] == "uhu":
            radii = [(r["sigma"], "-", "uHu $\\sigma=%.3f$" % r["sigma"])]
        else:
            radii = [(r["sigma_avg"], "-", "Riesz $\\sigma_{avg}=%.3f$" % r["sigma_avg"]),
                     (r["sigma_dir"], ":", "Riesz $\\sigma_{dir}=%.1f$" % r["sigma_dir"])]
        for rad, ls, lab in radii:
            ax.add_patch(Circle((x0, y0), rad, fill=False, ec=ROUTE_COLORS.get(route, "k"),
                                lw=1.6, ls=ls))
            handles.append(plt.Line2D([], [], color=ROUTE_COLORS.get(route, "k"), ls=ls, label=lab))
    ax.plot([x0], [y0], "+", color="w", ms=8, mew=1.5)
    handles += [plt.Line2D([], [], color="#2a6fd6", label="band edge (profile rows; pointwise: every point inside)"),
                plt.Line2D([], [], color="#2a6fd6", ls=":", label="seam (rows excluded at the core)"),
                plt.Line2D([], [], color="#eb8a34", label="flat columns (x-average)"),
                plt.Line2D([], [], color="#d1352b", ls="--", label="strips (%g period each)" % cfg["strip_periods"])]
    ax.legend(handles=handles, fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_ylim(-1.6 * bh, 1.6 * bh)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y - y_c$")
    ax.set_title("query domains and kernel widths  $\\mu=%g$, $R=%g$\n%s" % (mu, R, stem), fontsize=8)
    fig.savefig(path, dpi=160); plt.close(fig)


def fig_pareto(path, fits, route, mu, R):
    doms = [d for d in ("profile", "pointwise") if d in fits]
    fig, axs = plt.subplots(1, len(doms), figsize=(5.0 * len(doms), 3.6),
                            constrained_layout=True, squeeze=False)
    for ax, dom in zip(axs[0], doms):
        f = fits[dom]; P = f["path"]
        if not P:
            continue
        n = [p["n"] for p in P[1:]]; e = [max(1.0 - p["r2"], 1e-6) for p in P[1:]]
        ax.semilogy(n, e, "o-", color=ROUTE_COLORS.get(route, "k"), lw=1.2)
        for p, ee in zip(P[1:], e):
            ax.annotate(pretty(p["added"]), (p["n"], ee), textcoords="offset points",
                        xytext=(3, 5), fontsize=7, color=FAMILY_COLORS.get(FAMILY_OF.get(p["added"]), "k"))
        ax.plot([f["n_terms"]], [max(1 - f["r2_sel"], 1e-6)], "o", ms=12, mfc="none", mec="r",
                mew=1.5, label="STLSQ choice (%d term%s)" % (f["n_terms"], "" if f["n_terms"] == 1 else "s"))
        ax.axhline(max(1 - f["r2_rcn"], 1e-6), color="0.4", ls="--", lw=0.9,
                   label="one-term RCN model $w\\sqrt{1-w^2}$")
        RP = f.get("ref_path", [])
        if RP:
            ax.semilogy([p["n"] for p in RP[1:]], [max(1.0 - p["r2"], 1e-6) for p in RP[1:]], "s:", color="0.55",
                        ms=3, lw=0.9, label="planted self-dual knee, same route (law $\\beta$)")
        ax.set_xlabel("terms beyond the constant (forward selection)")
        ax.set_ylabel("$1 - R^2_{\\rm holdout}$")
        ax.set_title("%s, %s  ($\\mu=%g$, $R=%g$)" % (ksc.ROUTES[route]["label"], dom, mu, R), fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
    fig.savefig(path, dpi=160); plt.close(fig)


def fig_corr(path, fits, route, mu, R):
    doms = [d for d in ("profile", "pointwise") if d in fits and "Theta" in fits[d]]
    fig, axs = plt.subplots(1, len(doms), figsize=(5.4 * len(doms), 4.8),
                            constrained_layout=True, squeeze=False)
    for ax, dom in zip(axs[0], doms):
        f = fits[dom]
        T = np.column_stack([f["y"], f["Theta"][:, 1:]])
        names = ["target"] + f["names"][1:]
        C = np.corrcoef(T.T)
        im = ax.imshow(C, cmap="seismic", vmin=-1, vmax=1)
        ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
        ax.set_xticklabels([pretty(n) if n != "target" else "div $\\mathbf{k}$" for n in names],
                           rotation=90, fontsize=6)
        ax.set_yticklabels([pretty(n) if n != "target" else "div $\\mathbf{k}$" for n in names], fontsize=6)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, "%.2f" % C[i, j], ha="center", va="center", fontsize=4,
                        color="w" if abs(C[i, j]) > 0.6 else "k")
        ax.set_title("%s, %s: Pearson correlation of the columns\n(cond %.1e)"
                     % (ksc.ROUTES[route]["label"], dom, f["cond"]), fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(path, dpi=160); plt.close(fig)


def fig_profile_fit(path, fits, route, mu, R):
    f = fits["profile"]; yy, d, w = f["yy"], f["y"], f["w"]
    o = np.argsort(yy)
    gap = np.where(np.diff(yy[o]) > 1.5 * np.median(np.diff(yy[o])))[0] + 1
    _g = lambda a: np.insert(np.asarray(a, float)[o], gap, np.nan)   # no line across the seam
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    ax = axs[0]
    ax.plot(yy, d, ".", ms=3, color="0.3", label="div $\\mathbf{k}$ (band rows)")
    ax.plot(_g(yy), _g(f["yhat_lin"]), "-", color=ROUTE_COLORS.get(route, "k"), lw=1.5,
            label="$b - c\\,w$: $c=%.3f$, $R^2_h=%.4f$" % (f["c_raw"], f["r2_lin"]))
    if not f["form_ok"]:
        ax.plot(_g(yy), _g(f["yhat_sel"]), "--", color="k", lw=1.0,
                label="STLSQ: " + ", ".join(pretty(f["names"][j]) for j in np.where(f["res"]["support"])[0]))
    ax.set_xlabel("$y - y_c$"); ax.set_ylabel("div $\\mathbf{k}$"); ax.legend(fontsize=6)
    ksc.panel_label(ax, "(a)")
    ax = axs[1]
    ow = np.argsort(w)
    ax.plot(w, d, ".", ms=3, color="0.5", label="data")
    ax.plot(w[ow], f["yhat_lin"][ow], "-", color=ROUTE_COLORS.get(route, "k"), lw=1.5, label="$b - c\\,w$")
    ax.plot(w[ow], f["yhat_rcn"][ow], ":", color="k", lw=1.2,
            label="$b - c\\,w\\sqrt{1-w^2}$ ($\\Delta R^2_h=%+.4f$)" % f["d_rcn"])
    ax.axvline(-mu * mu, color="0.6", lw=0.8, ls="--", label="$-\\mu^2$ (core value of $w$)")
    ax.set_xlabel("$w$"); ax.set_ylabel("div $\\mathbf{k}$"); ax.legend(fontsize=6)
    ksc.panel_label(ax, "(b)")
    ax = axs[2]
    st = fits.get("strips", [])
    if st:
        cs = [s["c_raw"] for s in st]
        cols = ["#1baf7a" if s["form_ok"] else ("#9fd8bf" if s["w_admissible"] else "#d1352b") for s in st]
        ax.bar(range(len(cs)), cs, color=cols)
        ax.axhline(f["c_raw"], color="k", lw=1.0, label="full x-average $c=%.3f$" % f["c_raw"])
        ax.set_ylim(0.8 * min(cs + [f["c_raw"]]), 1.1 * max(cs + [f["c_raw"]]))
        ax.set_xlabel("strip (one stripe period each)"); ax.set_ylabel("$c$ (raw)")
        ax.legend(handles=[Patch(color="#1baf7a", label="strip selects $w$ alone"),
                           Patch(color="#9fd8bf", label="another shape picked, $w$ alone admissible"),
                           Patch(color="#d1352b", label="$w$ alone not admissible"),
                           plt.Line2D([], [], color="k", label="full x-average")], fontsize=6)
    ksc.panel_label(ax, "(c)")
    fig.suptitle("%s   $\\mu=%g$  $R=%g$   profile domain" % (ksc.ROUTES[route]["label"], mu, R), fontsize=9)
    fig.savefig(path, dpi=160); plt.close(fig)


def _frame_key(r):
    return (r["R"], r["mu"])


def fig_terms(out_dir, rows, cfg, all_names):
    doms = [d for d in ("profile", "strips", "pointwise") if d in cfg["domains"]]
    routes = cfg["routes"]
    names = [n for n in all_names if n not in ("1",)]
    fig, axs = plt.subplots(len(routes), len(doms), figsize=(5.2 * len(doms), 0.28 * len(names) * len(routes) + 1.6),
                            constrained_layout=True, squeeze=False)
    for i, route in enumerate(routes):
        for j, dom in enumerate(doms):
            ax = axs[i, j]
            sel = sorted([r for r in rows if r["route"] == route and r["domain"] == dom], key=_frame_key)
            if not sel:
                ax.axis("off"); continue
            Z = np.full((len(names), len(sel)), np.nan)
            for jj, r in enumerate(sel):
                if dom == "strips":
                    counts = dict(t.split(":") for t in r["strip_terms"].split(";") if t)
                    for ii, nm in enumerate(names):
                        if nm in counts:
                            a, b = counts[nm].split("/"); Z[ii, jj] = float(a) / float(b)
                        else:
                            Z[ii, jj] = 0.0
                else:
                    chosen = set(r["terms_names"].split(";")) if r["terms_names"] else set()
                    for ii, nm in enumerate(names):
                        Z[ii, jj] = 1.0 if nm in chosen else 0.0
            ax.imshow(Z, cmap="Greys", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([pretty(n) for n in names], fontsize=6)
            for lab, nm in zip(ax.get_yticklabels(), names):
                lab.set_color(FAMILY_COLORS.get(FAMILY_OF.get(nm), "k"))
            step = max(1, len(sel) // 14)
            ax.set_xticks(range(0, len(sel), step))
            ax.set_xticklabels(["%.3g\nR%g" % (sel[k]["mu"], sel[k]["R"]) for k in range(0, len(sel), step)],
                               fontsize=5)
            for jj, r in enumerate(sel):
                if not r["w_admissible"]:
                    ax.plot([jj], [len(names) - 0.3], "v", color="r", ms=4, clip_on=False)
                if not r["planted_w_admissible"]:
                    ax.plot([jj], [len(names) + 0.4], "v", color="#eb8a34", ms=4, clip_on=False)
                rc = REGIME_COL.get(r["regime"])
                if rc:
                    ax.add_patch(plt.Rectangle((jj - 0.5, -1.4), 1.0, 0.8, color=rc, clip_on=False))
            ax.set_title("%s, %s%s" % (ksc.ROUTES[route]["label"], dom,
                                       " (fraction of strips)" if dom == "strips" else " (selected)"), fontsize=8)
    fig.suptitle("terms selected (black) by frame, ordered by R then $\\mu$; label color = family; top strip = regime; "
                 "red mark: $w$ alone not admissible on the data; orange mark: not admissible on the planted self-dual knee",
                 fontsize=8)
    fig.savefig(Path(out_dir) / "summary_terms.png", dpi=170); plt.close(fig)


def _series(rows, route, dom, key):
    out = {}
    for r in rows:
        if r["route"] == route and r["domain"] == dom:
            out.setdefault((r["set"], r["R"]), []).append((r["mu"], r[key]))
    return {k: sorted(v) for k, v in out.items()}


def fig_r2(out_dir, rows, cfg, regimes):
    routes = cfg["routes"]
    fig, axs = plt.subplots(1, len(routes), figsize=(5.4 * len(routes), 3.8), constrained_layout=True, squeeze=False)
    for ax, route in zip(axs[0], routes):
        shade_regimes(ax, regimes)
        for dom, ls in (("profile", "-"), ("pointwise", "--")):
            if dom not in cfg["domains"]:
                continue
            for si, ((st, R), pts) in enumerate(sorted(_series(rows, route, dom, "r2_lin").items())):
                mk = SET_MARKERS[si % len(SET_MARKERS)]
                mus = [p[0] for p in pts]
                for key, col, lab in (("r2_lin", ROUTE_COLORS.get(route, "k"), "$b-cw$"),
                                      ("r2_sel", "k", "STLSQ choice"),
                                      ("r2_rcn", "0.5", "$b-cw\\sqrt{1-w^2}$")):
                    v = [max(1 - q[1], 1e-6) for q in _series(rows, route, dom, key)[(st, R)]]
                    ax.semilogy(mus, v, mk + ls, color=col, ms=3, lw=0.9,
                                label="%s, %s, %s R%g" % (lab, dom, st, R))
        ax.set_xlabel("$\\mu$"); ax.set_ylabel("$1 - R^2_{\\rm holdout}$")
        ax.set_title(ksc.ROUTES[route]["label"], fontsize=9); ax.legend(fontsize=4.5, ncol=2)
    fig.savefig(Path(out_dir) / "summary_r2.png", dpi=170); plt.close(fig)


def fig_c(out_dir, rows, cfg, regimes, calibrated):
    routes = cfg["routes"]
    key = "c_cal" if calibrated else "c_raw"
    fig, axs = plt.subplots(1, len(routes), figsize=(5.4 * len(routes), 3.8), constrained_layout=True, squeeze=False)
    for ax, route in zip(axs[0], routes):
        shade_regimes(ax, regimes)
        for dom, ls in (("profile", "-"), ("strips", ""), ("pointwise", "--")):
            if dom not in cfg["domains"]:
                continue
            ser = _series(rows, route, dom, key)
            for si, ((st, R), pts) in enumerate(sorted(ser.items())):
                mk = SET_MARKERS[si % len(SET_MARKERS)]
                mus = np.array([p[0] for p in pts]); v = np.array([p[1] for p in pts], float)
                if calibrated:
                    ok = np.array([q[1] for q in _series(rows, route, dom, "cal_status")[(st, R)]]) == "CAL_OK"
                    ax.plot(mus[ok], v[ok], mk + ls, color=ROUTE_COLORS.get(route, "k"), mfc="none", ms=4, lw=0.8,
                            label="%s, %s R%g (CAL_OK)" % (dom, st, R))
                    if (~ok).any():
                        ax.plot(mus[~ok], v[~ok], "x", color="r", ms=4,
                                label="%s, %s R%g (flagged)" % (dom, st, R))
                elif dom == "strips":
                    sd = np.array([q[1] for q in _series(rows, route, dom, "strip_c_sd")[(st, R)]], float)
                    ax.errorbar(mus, v, yerr=sd, fmt=mk, color="0.4", ms=3, capsize=1.5, lw=0.7,
                                label="strips mean $\\pm$ sd, %s R%g" % (st, R))
                else:
                    ax.plot(mus, v, mk + ls, color=ROUTE_COLORS.get(route, "k"), ms=3, lw=0.9,
                            label="%s, %s R%g" % (dom, st, R))
        if calibrated:
            mg = np.linspace(0.3, max(0.82, cfg["mu_max"] or 0.8), 100)
            for R, c in sorted(ksc.RECORD_LAW_C.items()):
                ax.plot(mg, 1 + c * mg ** 2, ":", color="0.3", lw=1,
                        label="record law $1+%.2f\\mu^2$ (fit on 0.325-0.475), $R=%g$" % (c, R))
            rec = sorted(ksc.RECORD_BETA.items())
            ax.plot([k[0] for k, v in rec], [v for k, v in rec], "+", color="k", ms=6,
                    label="tanh record (shallow sweeps)")
            ax.set_ylabel("$\\beta$ calibrated through the route's planted response")
        else:
            ax.set_ylabel("$c$ raw (slope of div $\\mathbf{k} = b - c\\,w$)")
        ax.set_xlabel("$\\mu$"); ax.set_title(ksc.ROUTES[route]["label"], fontsize=9)
        ax.legend(fontsize=4.5, ncol=1)
    fig.savefig(Path(out_dir) / ("summary_c_calibrated.png" if calibrated else "summary_c_raw.png"), dpi=170)
    plt.close(fig)


def fig_excess(out_dir, rows, cfg, regimes):
    """Holdout R^2 the library gains beyond the one-term law in w, on the
    data and on the planted self-dual knee at the law's beta through the
    same route; their difference is the structure beyond what the route
    itself does to a self-dual knee."""
    routes = cfg["routes"]
    doms = [d for d in ("profile", "pointwise") if d in cfg["domains"]]
    fig, axs = plt.subplots(len(doms), len(routes), figsize=(5.4 * len(routes), 3.4 * len(doms)),
                            constrained_layout=True, squeeze=False)
    for i, dom in enumerate(doms):
        for j, route in enumerate(routes):
            ax = axs[i, j]; shade_regimes(ax, regimes)
            for si, ((st, R), pts) in enumerate(sorted(_series(rows, route, dom, "gain_best").items())):
                mk = SET_MARKERS[si % len(SET_MARKERS)]
                ref = _series(rows, route, dom, "ref_gain_best")[(st, R)]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], mk + "-", color=ROUTE_COLORS.get(route, "k"),
                        ms=3, lw=0.9, label="SH data, %s R%g" % (st, R))
                ax.plot([p[0] for p in ref], [p[1] for p in ref], mk + ":", color="0.55", mfc="none", ms=3, lw=0.9,
                        label="planted self-dual knee, %s R%g" % (st, R))
            ax.axhline(0, color="k", lw=0.5)
            ax.axhline(cfg["rel_tol"], color="r", lw=0.6, ls="--", label="rel_tol (admissibility)")
            ax.set_xlabel("$\\mu$"); ax.set_ylabel("holdout $R^2$ gain beyond $b - c\\,w$")
            ax.set_title("%s, %s" % (ksc.ROUTES[route]["label"], dom), fontsize=9); ax.legend(fontsize=5)
    fig.savefig(Path(out_dir) / "summary_excess.png", dpi=170); plt.close(fig)


def fig_strips(out_dir, rows, cfg, regimes):
    routes = cfg["routes"]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True)
    for route in routes:
        ser = _series(rows, route, "strips", "strip_spread")
        fr = _series(rows, route, "strips", "strip_frac_w_alone")
        for si, (k, pts) in enumerate(sorted(ser.items())):
            mk = SET_MARKERS[si % len(SET_MARKERS)]
            axs[0].plot([p[0] for p in pts], [100 * p[1] for p in pts], mk + "-", color=ROUTE_COLORS.get(route, "k"),
                        ms=3, lw=0.8, label="%s, %s R%g" % (route, k[0], k[1]))
            axs[1].plot([p[0] for p in fr[k]], [p[1] for p in fr[k]], mk + "-", color=ROUTE_COLORS.get(route, "k"),
                        ms=3, lw=0.8, label="%s, %s R%g" % (route, k[0], k[1]))
    for ax in axs:
        shade_regimes(ax, regimes); ax.set_xlabel("$\\mu$"); ax.legend(fontsize=5)
    axs[0].set_ylabel("spread of $c$ across strips, sd/mean (%)")
    axs[1].set_ylabel("fraction of strips selecting $w$ alone"); axs[1].set_ylim(-0.05, 1.05)
    fig.savefig(Path(out_dir) / "summary_strips.png", dpi=170); plt.close(fig)


def fig_planted(out_dir, prow, rows, cfg, regimes):
    routes = cfg["routes"]
    fig, axs = plt.subplots(len(routes), 3, figsize=(15, 3.6 * len(routes)), constrained_layout=True, squeeze=False)
    for i, route in enumerate(routes):
        P = [p for p in prow if p["route"] == route and p["domain"] == "profile"]
        lad = [p for p in P if p["kind"] == "tanh_ladder"]
        frames = sorted({(p["set"], p["R"], p["mu"]) for p in lad})
        cmap = plt.get_cmap("viridis")
        ax = axs[i, 0]
        for k, fr in enumerate(frames):
            pts = sorted((p["beta"], p["c_raw"]) for p in lad if (p["set"], p["R"], p["mu"]) == fr)
            ax.plot([a for a, b in pts], [b for a, b in pts], "-", color=cmap(k / max(len(frames) - 1, 1)), lw=0.8)
        ax.set_xlabel("planted $\\beta$"); ax.set_ylabel("$c$ read (profile)")
        ax.set_title("%s: response to the planted self-dual knee (colour: frame, low to high $\\mu$)"
                     % ksc.ROUTES[route]["label"], fontsize=8)
        ax = axs[i, 1]
        betas = sorted({p["beta"] for p in lad})
        Z = np.full((len(betas), len(frames)), np.nan)
        for p in lad:
            Z[betas.index(p["beta"]), frames.index((p["set"], p["R"], p["mu"]))] = 1.0 if p["form_ok"] else (0.6 if p["w_admissible"] else 0.0)
        ax.imshow(Z, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto", origin="lower")
        ax.set_yticks(range(len(betas))); ax.set_yticklabels(["%.2f" % b for b in betas], fontsize=6)
        step = max(1, len(frames) // 12)
        ax.set_xticks(range(0, len(frames), step))
        ax.set_xticklabels(["%.3g\nR%g" % (frames[k][2], frames[k][1]) for k in range(0, len(frames), step)], fontsize=5)
        ax.set_ylabel("planted $\\beta$"); ax.set_title("planted self-dual knee: $w$ alone picked (green), admissible (yellow), not (red); profile", fontsize=8)
        ax = axs[i, 2]
        shade_regimes(ax, regimes)
        for kind, col, lab in (("tanh_law", "#1baf7a", "planted self-dual knee"),
                               ("quartic_law", "#d1352b", "planted RCN quartic knee")):
            pts = sorted((p["mu"], p["d_rcn"]) for p in P if p["kind"] == kind)
            ax.plot([a for a, b in pts], [b for a, b in pts], "o", color=col, ms=3, label=lab)
        dat = sorted((r["mu"], r["d_rcn"]) for r in rows if r["route"] == route and r["domain"] == "profile")
        ax.plot([a for a, b in dat], [b for a, b in dat], "k.", ms=4, label="SH data")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("$\\mu$"); ax.set_ylabel("$R^2_h(w\\sqrt{1-w^2}) - R^2_h(w)$")
        ax.set_title("which shape fits better, profile (above 0: the RCN quartic shape)", fontsize=8)
        ax.legend(fontsize=6)
    fig.savefig(Path(out_dir) / "summary_planted.png", dpi=170); plt.close(fig)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

class Tee(object):
    def __init__(self):
        self.lines = []

    def __call__(self, s=""):
        print(s); self.lines.append(s)


def read_regimes(cfg, runs):
    """(lo, hi, label) spans for shading, and a per-mu label."""
    table = {}
    p = cfg.get("survey_table")
    if p and os.path.isfile(p):
        with open(p) as fh:
            for r in csv.DictReader(fh):
                try:
                    table[round(float(r["mu"]), 3)] = r["regime"]
                except Exception:
                    pass
    def label(mu):
        if table:
            k = min(table, key=lambda m: abs(m - mu))
            if abs(k - mu) < 0.006:
                return table[k]
        lo, hi = cfg["regime_edges"]
        return "knee" if mu < lo else ("intermediate" if mu < hi else "dislocation")
    mus = sorted({r["mu"] for r in runs})
    spans = []
    for k, m in enumerate(mus):
        lo = m - 0.5 * ((m - mus[k - 1]) if k > 0 else 0.01)
        hi = m + 0.5 * ((mus[k + 1] - m) if k + 1 < len(mus) else 0.01)
        spans.append((lo, hi, label(m)))
    src = ("survey table %s" % p) if table else ("fallback edges %s (charge/circulation handoff, width 0.3)"
                                                  % (tuple(cfg["regime_edges"]),))
    return spans, label, src


def stamp(cfg):
    Rs = "R" + "-".join(("%g" % r).replace(".", "p") for r in cfg["R_values"]) if cfg["R_values"] else "Rall"
    parts = ["sdrich", Rs, "mu%s-%s" % (cfg["mu_min"] if cfg["mu_min"] is not None else "min",
                                        cfg["mu_max"] if cfg["mu_max"] is not None else "max"),
             "routes-" + "-".join(cfg["routes"]), "dom-" + "-".join(cfg["domains"]),
             "xwin-" + cfg["x_window"]]
    if cfg["mu_step"]:
        parts.append("step%g" % cfg["mu_step"])
    if cfg["coef_cap"] is not None:
        parts.append("cap%g" % cfg["coef_cap"])
    if cfg["leaky"]:
        parts.append("leaky")
    return "_".join(parts)


def main(cfg):
    T0 = time.time()
    ksc.setup_method_layer(cfg.get("method_dirs", ()))
    ksc.use_house_style()
    log = Tee()
    runs = discover(cfg)
    if not runs:
        print("No runs found under", cfg["roots"]); return []
    out_dir = Path(cfg["out_root"]) / stamp(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    ksc.save_config(out_dir, dict(cfg, version=VERSION))
    regimes, regime_of, reg_src = read_regimes(cfg, runs)
    log("%s\noutput -> %s" % (VERSION, out_dir))
    log("routes %s; domains %s; library families %s (w up to w^%d)%s; coef_cap %s"
        % (cfg["routes"], cfg["domains"], cfg["families"], cfg["w_max_power"],
           "; LEAKY columns ON" if cfg["leaky"] else "", cfg["coef_cap"]))
    log("regimes from %s" % reg_src)
    log("%d frames:" % len(runs))
    for r in runs:
        note = "" if r["R_src"] != "DEFAULT" else "   !!! R not in metadata or stem: R = %g (--R-default)" % r["R"]
        log("   %-14s mu %.3f  R %g (%s)  %s%s" % (r["set"], r["mu"], r["R"], r["R_src"], r["stem"], note))

    rows, srows, prow = [], [], []
    all_names = []
    for irun, run in enumerate(runs):
        t1 = time.time()
        x, y, u, fidx, tf = load_frame(run, cfg["frame"])
        mu, R = run["mu"], run["R"]
        if run["jc"] is None:
            jc, note = locate_core_row(u, x, y)
            log("   !!! %s: gb_row_in_crop missing; core row %s" % (run["stem"], note))
        else:
            jc = run["jc"]
        u, x, alpha_eff, trim = ksc.window_frame(u, x, mu, cfg["x_window"], cfg["alpha_x"], cfg["max_seam_jump"])
        cfg_run = dict(cfg, alpha_x=alpha_eff)
        bl = beta_law(mu, R)
        reg = regime_of(mu)
        log("\n== [%d/%d] %s  %s  mu %.3f  R %g  frame %d (t = %s)  regime %s ==" % (
            irun + 1, len(runs), run["set"], run["stem"], mu, R, fidx,
            ("%g" % tf) if np.isfinite(tf) else "?", reg))
        if trim is not None:
            log("   x window: %d of %d columns, %d periods, seam jump %.3f rad"
                % (trim["n_cols"], trim["n_cols_in"], trim["n_periods"], trim["jump_rad"]))
        fdir = out_dir / ("%s__%s" % (run["set"], run["stem"])); fdir.mkdir(exist_ok=True)
        planted_doms = [d for d in cfg["domains"] if d != "strips"]
        if cfg["planted_pointwise"] == "off":
            planted_doms = [d for d in planted_doms if d != "pointwise"]

        # ---------- planted checks (A: ladder, B: law beta tanh and quartic) ----------
        synth = []
        if cfg["validate"]:
            for b in cfg["planted_ladder"]:
                synth.append(("tanh_ladder", b, ksc.synth_knee_on_grid(x, y, mu, R, b, jc)))
            if np.isfinite(bl):
                synth.append(("tanh_law", bl, synth_knee_rcn_quartic(x, y, mu, R, bl, y[jc], quartic=False)))
                synth.append(("quartic_law", bl, synth_knee_rcn_quartic(x, y, mu, R, bl, y[jc], quartic=True)))
            else:
                log("   !!! no record law for R = %g: planted check B skipped" % R)
        resp = {}; pstat = {}
        for route in cfg["routes"]:
            for kind, b, us_ in synth:
                k1s, k2s = ksc.extract_k(route, us_, x, y, jc, alpha_eff, cfg["y_frac"])
                doms = list(planted_doms)
                if kind == "tanh_ladder" and cfg["planted_pointwise"] == "law_only":
                    doms = [d for d in doms if d != "pointwise"]
                if kind == "tanh_ladder" and abs(b - 1.0) < 1e-9 and "strips" in cfg["domains"]:
                    doms = doms + ["strips"]
                ff = fit_field(k1s, k2s, x, y, jc, mu, route, cfg_run, doms, n_boot=cfg["n_boot_planted"])
                for dom in doms:
                    if dom == "strips":
                        ss = strip_summary(ff["strips"])
                        pstat[(route, "strips_floor")] = ss
                        continue
                    f = ff[dom]
                    if kind == "tanh_ladder":
                        resp.setdefault((route, dom), []).append((b, f["c_raw"]))
                    pstat.setdefault((route, dom, kind), []).append((b, f))
                    prow.append(dict(set=run["set"], stem=run["stem"], mu=mu, R=R, route=route, domain=dom,
                                     kind=kind, beta=b, c_raw=f["c_raw"], r2_lin=f["r2_lin"], r2_best=f["r2_best"],
                                     w_admissible=f["w_admissible"], gain_best=f["gain_best"],
                                     resid_ratio=f["resid_ratio"], r2_rcn=f["r2_rcn"],
                                     d_rcn=f["d_rcn"], terms=f["terms"], form_ok=f["form_ok"],
                                     fwd_1=f["fwd_1"], fwd_2=f["fwd_2"], fwd_gain_2=f["fwd_gain_2"]))
            for dom in planted_doms:
                lad = pstat.get((route, dom, "tanh_ladder"), [])
                if lad:
                    cells = "  ".join("%.2f:%.3f%s" % (b, f["c_raw"], "" if f["form_ok"] else ("~" if f["w_admissible"] else "!"))
                                      for b, f in lad)
                    bad = [b for b, f in lad if not f["w_admissible"]]
                    mono = bool(np.all(np.diff([f["c_raw"] for b, f in sorted(lad, key=lambda t: t[0])]) > 0))
                    log("   planted A %-9s %-9s c read at planted beta (~ = STLSQ picks another shape but w alone is admissible, "
                        "! = w alone not admissible): %s   -> %s%s"
                        % (route, dom, cells, ("W_NOT_ADMISSIBLE at beta %s" % ",".join("%.2f" % b for b in bad)) if bad else "W_ADMISSIBLE",
                           "" if mono else ", RESPONSE_NOT_MONOTONE"))
                tl = pstat.get((route, dom, "tanh_law")); ql = pstat.get((route, dom, "quartic_law"))
                if tl and ql:
                    ft, fq = tl[0][1], ql[0][1]
                    disc = fq["d_rcn"] - ft["d_rcn"]
                    log("   planted B %-9s %-9s beta_law %.3f: self-dual knee selects [%s] (library gain %+.4f); "
                        "d_rcn self-dual %+.4f, quartic %+.4f (difference %+.4f, %s)"
                        % (route, dom, bl, ft["terms"], ft["gain_best"], ft["d_rcn"], fq["d_rcn"], disc,
                           "resolved" if disc > cfg["rcn_resolve_tol"] else "NOT resolved"))
            fl = pstat.get((route, "strips_floor"))
            if fl:
                log("   planted A %-9s strips    beta 1: spread %.2f%%, w alone on %d/%d strips, admissible on %d/%d (the instrument's floor)"
                    % (route, 100 * fl["strip_spread"], int(round(fl["strip_frac_w_alone"] * fl["n_strips"])), fl["n_strips"],
                       int(round(fl["strip_frac_w_admissible"] * fl["n_strips"])), fl["n_strips"]))

        # ---------- data ----------
        keep = cfg["frame_figures"]
        fits_by_route = {}
        for route in cfg["routes"]:
            k1, k2 = ksc.extract_k(route, u, x, y, jc, alpha_eff, cfg["y_frac"])
            ff = fit_field(k1, k2, x, y, jc, mu, route, cfg_run, cfg["domains"], keep_theta=keep)
            fits_by_route[route] = ff
            ss = strip_summary(ff.get("strips", []))
            for i_s, s in enumerate(ff.get("strips", [])):
                srows.append(dict(set=run["set"], stem=run["stem"], mu=mu, R=R, route=route, strip=i_s,
                                  x_lo=float(x[s["cols"].start]), x_hi=float(x[s["cols"].stop - 1]),
                                  c_raw=s["c_raw"], r2_lin=s["r2_lin"], r2_best=s["r2_best"], terms=s["terms"],
                                  form_ok=s["form_ok"], w_admissible=s["w_admissible"]))
            for dom in cfg["domains"]:
                if dom == "strips":
                    if not ff.get("strips"):
                        continue
                    st_ = ff["strips"]
                    f = dict(c_raw=ss["strip_c_mean"], c_raw_std=ss["strip_c_sd"], b=np.nan, b_over_peak=np.nan,
                             r2_lin=float(np.mean([s["r2_lin"] for s in st_])),
                             r2_sel=float(np.mean([s["r2_sel"] for s in st_])),
                             r2_best=float(np.mean([s["r2_best"] for s in st_])),
                             r2_rcn=float(np.mean([s["r2_rcn"] for s in st_])),
                             terms="", n_terms=np.nan, form_ok=bool(ss["strip_frac_w_alone"] == 1.0),
                             w_admissible=bool(ss["strip_frac_w_admissible"] == 1.0),
                             n=int(sum(s["n"] for s in st_)), w_min=np.nan, w_max=np.nan,
                             cond=np.nan, dropped=[], fwd_1="", fwd_2="", fwd_gain_2=np.nan)
                    f["d_rcn"] = f["r2_rcn"] - f["r2_lin"]; f["gain_sel"] = f["r2_sel"] - f["r2_lin"]
                    f["gain_best"] = f["r2_best"] - f["r2_lin"]
                    f["resid_ratio"] = (1 - f["r2_lin"]) / max(1 - f["r2_best"], 1e-12)
                    rdom = "profile"
                else:
                    f = ff[dom]; rdom = dom
                pts = resp.get((route, rdom), [])
                c_cal = ksc.calibrate(f["c_raw"], pts)
                lad = pstat.get((route, rdom, "tanh_ladder"), [])
                lad_ok = bool(lad) and all(g["w_admissible"] for b, g in lad)
                mono = bool(lad) and bool(np.all(np.diff([g["c_raw"] for b, g in sorted(lad, key=lambda t: t[0])]) > 0))
                rr = [c for b, c in pts]
                if not pts:
                    cst = "CAL_NONE"
                elif not mono:
                    cst = "CAL_RESPONSE_NOT_MONOTONE"
                elif not lad_ok:
                    cst = "CAL_PLANTED_NOT_LINEAR"
                elif not (min(rr) <= f["c_raw"] <= max(rr)):
                    cst = "CAL_EXTRAPOLATED"
                else:
                    cst = "CAL_OK"
                tl = pstat.get((route, rdom, "tanh_law")); ql = pstat.get((route, rdom, "quartic_law"))
                if tl and ql:
                    fr = tl[0][1]
                    dt, dq = fr["d_rcn"], ql[0][1]["d_rcn"]
                    res_ok = (dq - dt) > cfg["rcn_resolve_tol"]
                    qidx = (f["d_rcn"] - dt) / (dq - dt) if abs(dq - dt) > 1e-12 else np.nan
                    ref_terms, ref_gain, ref_adm = fr["terms"], fr["gain_best"], fr["w_admissible"]
                    excess = f["gain_best"] - ref_gain
                    if dom != "strips":
                        ff[dom]["ref_path"] = fr.get("path", [])
                else:
                    dt = dq = qidx = ref_gain = excess = np.nan; res_ok = False; ref_terms = ""; ref_adm = False
                rec = ksc.record_lookup(mu, R)
                row = OrderedDict(
                    set=run["set"], stem=run["stem"], mu=mu, R=R, R_source=run["R_src"], frame=fidx, t=tf,
                    regime=reg, route=route, domain=dom, n_samples=f["n"],
                    w_min=f["w_min"], minus_mu2=-mu * mu,
                    w_cover=(f["w_min"] / (-mu * mu)) if np.isfinite(f["w_min"]) else np.nan,
                    c_raw=f["c_raw"], c_raw_std=f["c_raw_std"], b=f["b"], b_over_peak=f["b_over_peak"],
                    r2_lin=f["r2_lin"], w_admissible=f["w_admissible"], form_ok=f["form_ok"],
                    terms=f["terms"],
                    terms_names=(";".join(f["names"][j] for j in np.where(f["res"]["support"])[0])
                                 if dom != "strips" else ""),
                    n_terms=f["n_terms"], r2_sel=f["r2_sel"], gain_sel=f["gain_sel"],
                    r2_best=f["r2_best"], gain_best=f["gain_best"], resid_ratio=f["resid_ratio"],
                    ref_terms=ref_terms, ref_gain_best=ref_gain, ref_w_admissible=ref_adm, excess_gain=excess,
                    fwd_1=f["fwd_1"], fwd_2=f["fwd_2"], fwd_gain_2=f["fwd_gain_2"],
                    fwd_path=";".join("%s:%.5f" % (p["added"], p["r2"]) for p in f.get("path", [])),
                    r2_rcn=f["r2_rcn"], d_rcn=f["d_rcn"],
                    d_rcn_planted_tanh=dt, d_rcn_planted_quartic=dq, rcn_resolvable=res_ok,
                    quartic_index=qidx,
                    planted_w_admissible=lad_ok, planted_response_monotone=mono,
                    planted_not_admissible_betas=",".join("%.2f" % b for b, g in lad if not g["w_admissible"]),
                    response=",".join("%.2f:%.4f" % (b, c) for b, c in pts),
                    c_cal=c_cal, cal_status=cst, beta_law=bl,
                    cal_vs_law_pct=100 * (c_cal / bl - 1) if np.isfinite(bl) else np.nan,
                    record_beta=rec, cal_vs_record_pct=100 * (c_cal / rec - 1) if np.isfinite(rec) else np.nan,
                    cond=f["cond"], dropped=";".join(f["dropped"]),
                    **(dict(ss) if dom == "strips" else dict(
                        n_strips=np.nan, strip_c_mean=np.nan, strip_c_sd=np.nan, strip_spread=np.nan,
                        strip_frac_w_alone=np.nan, strip_frac_w_admissible=np.nan, strip_r2_lin_min=np.nan,
                        strip_terms="")))
                row["status"] = ";".join([
                    "PLANTED_W_ADMISSIBLE" if lad_ok else ("PLANTED_W_NOT_ADMISSIBLE[b=%s]" % row["planted_not_admissible_betas"]),
                    "DATA_W_ALONE" if f["form_ok"] else ("DATA_W_ADMISSIBLE" if f["w_admissible"] else "DATA_BEYOND_W"),
                    "RCN_RESOLVED" if res_ok else "RCN_UNRESOLVED", cst])
                rows.append(row)
                for nm in (ff[dom]["names"] if dom != "strips" else []):
                    if nm not in all_names:
                        all_names.append(nm)
                if dom == "strips":
                    log("   data %-9s strips    c mean %.4f sd %.4f (spread %.2f%%), w alone on %d/%d strips, admissible on %d/%d, "
                        "min strip R2 %.4f%s" % (route, ss["strip_c_mean"], ss["strip_c_sd"], 100 * ss["strip_spread"],
                                                int(round(ss["strip_frac_w_alone"] * ss["n_strips"])), ss["n_strips"],
                                                int(round(ss["strip_frac_w_admissible"] * ss["n_strips"])), ss["n_strips"],
                                                ss["strip_r2_lin_min"],
                                                ("; terms picked on the strips: " + ss["strip_terms"]) if ss["strip_terms"] else ""))
                else:
                    log("   data %-9s %-9s c %.4f (cal %.4f %s, law %.3f)  R2 lin %.4f, best %.4f -> %s  STLSQ [%s]  "
                        "gain beyond w %+.4f (self-dual plant through the route: %+.4f, excess %+.4f)  fwd %s then %s (%+.4f)  "
                        "d_rcn %+.4f (quartic index %s)  w reaches %.0f%% of -mu^2"
                        % (route, dom, f["c_raw"], c_cal, cst, bl, f["r2_lin"], f["r2_best"],
                           "w alone" if f["form_ok"] else ("w admissible" if f["w_admissible"] else "BEYOND w"),
                           f["terms"], f["gain_best"], ref_gain, excess,
                           f["fwd_1"], f["fwd_2"] or "-", f["fwd_gain_2"], f["d_rcn"],
                           ("%.2f" % qidx) if (res_ok and np.isfinite(qidx)) else "n/a",
                           100 * f["w_min"] / (-mu * mu)))
            if keep:
                fig_pareto(fdir / ("%s_pareto.png" % route), ff, route, mu, R)
                fig_corr(fdir / ("%s_corr.png" % route), ff, route, mu, R)
                if "profile" in ff:
                    fig_profile_fit(fdir / ("%s_profile_fit.png" % route), ff, route, mu, R)
        if keep:
            r0 = cfg["routes"][0]
            fig_domain(fdir / "fig_domain.png", u, x, y, jc, fits_by_route[r0]["prof"],
                       strip_columns(fits_by_route[r0]["prof"], x, cfg_run, mu), cfg, mu, R, run["stem"])
        log("   (%.1f s)" % (time.time() - t1))

    # ---------- outputs ----------
    def _write(path, rr):
        if not rr:
            return
        keys = []
        for r in rr:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(rr)
    _write(out_dir / "selfdual_rich.csv", rows)
    _write(out_dir / "strips.csv", srows)
    _write(out_dir / "planted_checks.csv", prow)
    fig_terms(out_dir, rows, cfg, all_names)
    fig_r2(out_dir, rows, cfg, regimes)
    fig_c(out_dir, rows, cfg, regimes, calibrated=False)
    fig_c(out_dir, rows, cfg, regimes, calibrated=True)
    fig_excess(out_dir, rows, cfg, regimes)
    if "strips" in cfg["domains"]:
        fig_strips(out_dir, rows, cfg, regimes)
    if prow:
        fig_planted(out_dir, prow, rows, cfg, regimes)

    log("\n== summary (read by admissibility of the one-term law and holdout R^2; 'excess' = library gain beyond w"
        " on the data minus that on the planted self-dual knee through the same route) ==")
    for route in cfg["routes"]:
        for dom in cfg["domains"]:
            sel = [r for r in rows if r["route"] == route and r["domain"] == dom]
            if not sel:
                continue
            for reg in ("knee", "intermediate", "dislocation"):
                rs = [r for r in sel if r["regime"] == reg]
                if not rs:
                    continue
                strict = sum(1 for r in rs if r["form_ok"])
                adm = sum(1 for r in rs if r["w_admissible"])
                pl = sum(1 for r in rs if r["planted_w_admissible"])
                cal = sum(1 for r in rs if r["cal_status"] == "CAL_OK")
                line = ("   %-9s %-9s %-12s %2d frames: w alone %d, w admissible %d (planted knees: admissible on %d, CAL_OK %d); "
                        "holdout R2 of b - c w %.4f .. %.4f"
                        % (route, dom, reg, len(rs), strict, adm, pl, cal,
                           min(r["r2_lin"] for r in rs), max(r["r2_lin"] for r in rs)))
                ex = [r["excess_gain"] for r in rs if np.isfinite(r["excess_gain"])]
                if ex:
                    line += "; excess %+.4f .. %+.4f" % (min(ex), max(ex))
                if dom != "strips":
                    fw = sum(1 for r in rs if r["fwd_1"] == "w")
                    line += "; forward picks w first on %d" % fw
                    qi = [r["quartic_index"] for r in rs if r["rcn_resolvable"] and np.isfinite(r["quartic_index"])]
                    line += ("; quartic index %.2f .. %.2f on %d resolved frames" % (min(qi), max(qi), len(qi))) if qi \
                        else "; RCN quartic shape not resolved by the route on any frame"
                else:
                    line += "; c spread across strips %.2f%% .. %.2f%%" % (
                        100 * np.nanmin([r["strip_spread"] for r in rs]), 100 * np.nanmax([r["strip_spread"] for r in rs]))
                log(line)
            for flag, key, lab in (("w_admissible", "w_admissible", "first frame where w alone is not admissible"),
                                   ("planted_w_admissible", "planted_w_admissible",
                                    "first frame where the planted self-dual knee is not read as linear in w")):
                first = {}
                for r in sorted(sel, key=lambda r: (r["set"], r["R"], r["mu"])):
                    k = (r["set"], r["R"])
                    if not r[key] and k not in first:
                        first[k] = r["mu"]
                if first:
                    log("   %-9s %-9s %s: %s" % (route, dom, lab, ", ".join(
                        "%s R%g at mu %.3g" % (k[0], k[1], v) for k, v in sorted(first.items()))))
    log("\nwrote %s (%.0f s)" % (out_dir, time.time() - T0))
    with open(out_dir / "report.txt", "w") as fh:
        fh.write("\n".join(log.lines) + "\n")
    return rows


# ---------------------------------------------------------------------
# config
# ---------------------------------------------------------------------

class _Args:
    roots = [
        str(_HERE / "data" / "sh_pgb_zigzag" / "mu_sweeps_shallow_v2_Rp5_Nx512_tmax500_hp5_np18_Ny5" / "raw" / "selected"),
        str(_HERE / "data" / "sh_pgb_zigzag" / "mu_sweeps_shallow_v2_R1p0_Nx512_tmax500_hp5_np18_Ny5" / "raw" / "selected"),
        str(_HERE / "data" / "sh_pgb_zigzag" / "1024_large_mu_sweep" / "raw")]
    root_labels = ["shallow", "shallow", "large"]
    pattern = "*sh_pgb_zigzag*mu*.npz"
    mu_min = None
    mu_max = 0.80                 # uHu's eigenvector flip sits at mu* ~ 0.82
    mu_step = None                # e.g. 0.02 keeps only mu on multiples of 0.02
    R_values = []                 # [] = all found
    R_default = 0.5               # used (loudly) when neither metadata nor stem carries R
    routes = ["uhu", "riesz_pi2"]
    domains = ["profile", "strips", "pointwise"]
    families = ["mismatch", "odd", "abs", "k1", "curl"]
    w_max_power = 5
    leaky = False                 # add the target-bearing columns, labelled [L]
    coef_cap = None               # absolute cap; off (see changelog)
    frame = -1
    validate = True
    planted_ladder = [0.9, 1.0, 1.2, 1.4, 1.6, 1.8]
    planted_pointwise = "full"    # 'full' | 'law_only' (pointwise planted only for check B) | 'off'
    n_boot = 200
    n_boot_planted = 10
    rcn_resolve_tol = 0.002       # planted quartic minus planted tanh d_rcn needed to call the shapes resolved
    max_forward = 6
    strip_periods = 1.0
    pointwise_max_samples = 40000
    x_window = "trim"
    max_seam_jump = 0.1
    alpha_x = 0.5
    y_frac = 0.06
    band_w = 3.0
    ff_renorm = True
    floors = list(np.geomspace(1e-3, 1.0, 25))
    rel_tol = 0.01
    seed = 0
    survey_table = None           # survey_table.csv of the large sweep (mu, regime) for shading
    regime_edges = [0.685, 0.815] # fallback: intermediates 0.69 to 0.81 at width 0.3
    frame_figures = True
    out_root = str(_HERE / "results" / "self_dual_sindy_rich_library")
    method_dirs = []


def cfg_from_args():
    return {k: getattr(_Args, k) for k in vars(_Args) if not k.startswith("_")}


def parse_cli():
    d = _Args
    p = argparse.ArgumentParser(description="self-dual SINDy with a richer, target-free library")
    p.add_argument("--roots", nargs="+", default=d.roots)
    p.add_argument("--root-labels", nargs="+", default=None)
    p.add_argument("--pattern", default=d.pattern)
    p.add_argument("--mu-min", type=float, default=d.mu_min)
    p.add_argument("--mu-max", type=float, default=d.mu_max)
    p.add_argument("--mu-step", type=float, default=d.mu_step)
    p.add_argument("--R", nargs="*", type=float, default=d.R_values)
    p.add_argument("--R-default", type=float, default=d.R_default)
    p.add_argument("--routes", nargs="+", default=d.routes, choices=list(ksc.ROUTES))
    p.add_argument("--domains", nargs="+", default=d.domains, choices=["profile", "strips", "pointwise"])
    p.add_argument("--families", nargs="+", default=d.families, choices=["mismatch", "odd", "abs", "k1", "curl"])
    p.add_argument("--w-max-power", type=int, default=d.w_max_power)
    p.add_argument("--leaky", action="store_true")
    p.add_argument("--coef-cap", type=float, default=d.coef_cap)
    p.add_argument("--frame", type=int, default=d.frame)
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--planted-ladder", nargs="+", type=float, default=d.planted_ladder)
    p.add_argument("--planted-pointwise", default=d.planted_pointwise, choices=["full", "law_only", "off"])
    p.add_argument("--n-boot", type=int, default=d.n_boot)
    p.add_argument("--max-forward", type=int, default=d.max_forward)
    p.add_argument("--strip-periods", type=float, default=d.strip_periods)
    p.add_argument("--pointwise-max-samples", type=int, default=d.pointwise_max_samples)
    p.add_argument("--x-window", default=d.x_window, choices=["tukey", "trim"])
    p.add_argument("--band-w", type=float, default=d.band_w)
    p.add_argument("--survey-table", default=d.survey_table)
    p.add_argument("--regime-edges", nargs=2, type=float, default=d.regime_edges)
    p.add_argument("--no-frame-figures", action="store_true")
    p.add_argument("--out-root", default=d.out_root)
    p.add_argument("--method-dirs", nargs="*", default=d.method_dirs)
    a = p.parse_args()
    cfg = cfg_from_args()
    labels = a.root_labels
    if labels is None:
        labels = d.root_labels if a.roots == d.roots else ["set%d" % i for i in range(len(a.roots))]
    if len(labels) != len(a.roots):
        p.error("--root-labels needs one label per root")
    cfg.update(roots=a.roots, root_labels=labels, pattern=a.pattern, mu_min=a.mu_min, mu_max=a.mu_max,
               mu_step=a.mu_step, R_values=a.R, R_default=a.R_default, routes=a.routes, domains=a.domains,
               families=a.families, w_max_power=a.w_max_power, leaky=a.leaky, coef_cap=a.coef_cap,
               frame=a.frame, validate=not a.no_validate, planted_ladder=a.planted_ladder,
               planted_pointwise=a.planted_pointwise, n_boot=a.n_boot, max_forward=a.max_forward,
               strip_periods=a.strip_periods, pointwise_max_samples=a.pointwise_max_samples,
               x_window=a.x_window, band_w=a.band_w, survey_table=a.survey_table,
               regime_edges=a.regime_edges, frame_figures=not a.no_frame_figures,
               out_root=a.out_root, method_dirs=a.method_dirs)
    return cfg


def _anchor(cfg):
    for k in ("out_root",):
        if not os.path.isabs(cfg[k]):
            cfg[k] = str(_HERE / cfg[k])
    cfg["roots"] = [r if os.path.isabs(r) else str(_HERE / r) for r in cfg["roots"]]
    if cfg.get("survey_table") and not os.path.isabs(cfg["survey_table"]):
        cfg["survey_table"] = str(_HERE / cfg["survey_table"])
    return cfg


if __name__ == "__main__":
    main(_anchor(parse_cli() if len(sys.argv) > 1 else cfg_from_args()))
