"""
self_dual_sindy_sweep.py  (v1.3)
--------------------------------
Stationary self-dual SINDy on relaxed SH knee frames, per (run, route),
across the shallow mu sweeps at R = 0.5 and R = 1.0. Routes: uhu
(sigma = 1.571), riesz (sigma_dir 0.4, sigma_avg pi, production), demod
(analytic signal along x). Session-6 handoff item 4 / pending item (a).

    python self_dual_sindy_sweep.py --roots <dir> [<dir> ...]
        [--mu-min 0.3 --mu-max 0.5 --R 0.5 1.0 --routes uhu riesz demod
         --domain profile|pointwise --no-validate --out-root DIR
         --method-dirs DIR ...]

Question answered: what stationary relation between div k and
w = |k|^2 - 1 does each extraction route yield on a relaxed knee, and
does the model FORM div k = c w survive holdout on both a planted
synthetic knee and the simulation frames. The coefficient c is reported
raw and, for the smoothed routes, calibrated through the route's own
synthetic response; the two are never mixed in one column.

Fit conventions (item 4): library [1, w, w^2, w^3]; two targets, div k
(self-dual form) and (div k)^2 (energy form, the divk_sq block); STLSQ
on column-normalized columns with the floor applied to the unit-free
contribution |xi_j| ||Theta_j|| / ||y||; floors swept over
geomspace(1e-3, 1, 25) with every single-term model and the {1, w}
pair added to the pool; two-fold spatial holdout (fit below the core,
score above, and the reverse; reported holdout R^2 is the mean);
selection = sparsest model within rel_tol of the best holdout R^2;
coefficients refit on all rows with bootstrap standard deviations.
Fits are read by holdout R^2 and selected terms; the w^2 and w^3
columns are near-collinear bumps (Gram condition printed per run).

Model-form criterion: STLSQ selects w and neither w^2 nor w^3. The
constant column is exempt from thresholding (a nuisance intercept
absorbing each route's far-field |k| offset) and does not count against
the form. The coefficient of record for every route is the slope of the
two-parameter model div k = b + c w (c_raw, beta-signed), which is
always defined and never collinear; the STLSQ selection and its holdout
gain over that model (r2_gain_stlsq) say whether the residual of the
linear form carries structure the library can express.

Validation (--validate, on by default, knees only): for every data
(mu, R) the exact analytic knee with planted beta in {0.9, 1.0, 1.1} is
built on THAT file's grid (beta_knee_fit.synth_knee, slaved A1 and A3)
and pushed through the identical pipeline. (i) FORM on every route at
every planted beta; (ii) VALUE for demod, |c/beta - 1| < 2 percent;
(iii) uhu and riesz readings recorded as the route's regression
response and used for the calibrated column, with the console tanh
response printed beside them for reference. Nothing blocks; every data
row carries validation_status.

Consistency with the record (loud, not blocking): the demod tanh
estimator of record is rerun unchanged on the untapered frame and must
reproduce RECORD_BETA within 0.5 percent; the demod SINDy slope is
compared to the same table with a 5 percent tolerance and the gap
printed as the regression-versus-tanh reading.

Changelog (measured justifications):
v1.3
  - 2D maps: the smoothing halo trimmed from the drawn columns is
    clamped so at least a quarter of the flat columns survive, and the
    title statistic falls back to the fit region when the trusted
    region is empty. Cause of the v1.2 crash: for the Riesz production
    route sigma_route = pi gives a 3 sigma halo of 9.4 units, 81
    columns per side on the 128-column flat region, so the trusted
    mask was empty and np.nanmax raised. Colour limits are now taken
    from the unblanked fit region for the same reason. Fits unchanged
    (the trusted mask is drawing only); a clamped halo is reported in
    the map title.
v1.2
  - --coef-cap C: after each STLSQ candidate is formed, terms whose
    physical coefficient exceeds C in magnitude are dropped and the
    support refit, to a fixed point (constant exempt). An absolute,
    unit-dependent cutoff, off by default; the run folder stamp carries
    it when set. Measured with C = 10 on the basic library: uHu and
    Riesz pi/2 unchanged (no large coefficients to drop); demod loses
    its w^3 terms (|coef| 17 to 48) and the w^2 terms of magnitude 11 to
    19, falling back to {1, w} or {1, w, w^2} with the small w^2.
  - 2D maps blank the tapered columns and the taper-halo rows so only
    the region the extraction can be trusted on is drawn.
v1.1
  - Coefficient of record is the slope of the two-parameter law on
    every route (always defined, never collinear); STLSQ reports its
    support and holdout gain beside it; summary line reworded so the
    count of frames with extra terms cannot be read as failures.
  - Constant column exempt from thresholding (0.87 vs 0.999 on a
    planted knee when it is not); |b|/peak reported.
  - --library extended: profile columns w_y^2, |w_y|, w_yy; pointwise
    columns |grad w|^2, Lap w, curl k, J. Measured: uHu stays minimal
    on 8/8; demod takes |w_y| and reaches R^2 0.998 at the angles where
    the basic library left 2 to 15 percent.
  - Per-stem 2D maps (u, w, div k, residual) with the fit region
    outlined; max |div k| in the region at most 1.4 x the profile peak
    on every route and frame (no singular behaviour).
  - Calibration flagged invalid unless the route's form holds on all
    planted knees and on the data; calibrated points drawn only then.
v1.0
  - Domain default 'profile' (x-averaged). Pointwise 2D on demod carries
    the 2-theta ripple: planted-knee R^2 0.15 pointwise against 0.9999
    on the profile. The knee is x-independent, so the profile holds the
    information; 'pointwise' remains available for the smoothed routes
    (uHu planted knee: R^2 0.95 pointwise, 1.000 profile).
  - Intercept in the library is load-bearing on every route: through
    the origin the demod slope on a planted beta = 1 knee reads 0.87
    (finite-difference k1 sits 0.3 percent low, shifting w by 0.006);
    with the constant column 0.997; with the x taper as well 0.999.
  - Regression and tanh readings agree on synthetics (0.999 vs 0.985 to
    0.999) but differ on data by 2 to 4 percent (e.g. 1.095 regression
    vs 1.123 tanh at mu 0.375, R 0.5): the shape-free reading sees the
    shoulder overshoot the tanh fit averages over. Both are reported.
  - Preprocessing (x taper, band, far-field renormalization) lives in
    knee_sweep_common.py with its own measured justifications.
Python 3.8 compatible; frozen method layer imported, never edited.
"""
import os
import sys
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import knee_sweep_common_v2 as ksc  # noqa: E402

LIB_NAMES = ["1", "w", "w^2", "w^3"]
TARGETS = ["divk", "divk_sq"]


# ---------------------------------------------------------------------
# library and STLSQ
# ---------------------------------------------------------------------

def build_library(w, extras=None):
    """[1, w, w^2, w^3] plus, for the extended library, the extra columns
    in `extras` (dict name -> samples). Names are appended to the
    returned list so the STLSQ report can label them."""
    cols = [np.ones_like(w), w, w ** 2, w ** 3]
    names = list(LIB_NAMES)
    if extras:
        for nm, col in extras.items():
            cols.append(col); names.append(nm)
    return np.column_stack(cols), names


def stlsq_floor(Theta, y, floor, max_iter=25, keep=(0,)):
    """Sequentially thresholded least squares where the threshold is a
    unit-free contribution floor: a term survives while
    |xi_j| ||Theta_j|| / ||y|| >= floor. Columns listed in `keep` (the
    constant, index 0) are exempt from thresholding: the intercept is a
    nuisance parameter that absorbs each route's far-field offset, and
    the sparsity question is asked of w, w^2, w^3 only. Measured: with
    the constant thresholded away, the demod slope on a planted
    beta = 1 knee read 1.024 to 1.030; retained, 0.999. Returns physical
    coefficients and the support."""
    norms = np.linalg.norm(Theta, axis=0)
    norms[norms == 0] = 1.0
    Th = Theta / norms
    ynorm = np.linalg.norm(y) + 1e-300
    p = Theta.shape[1]
    xi = np.linalg.lstsq(Th, y, rcond=None)[0]
    big = np.ones(p, dtype=bool)
    for _ in range(max_iter):
        contrib = np.abs(xi) / ynorm
        new_big = big & (contrib >= floor)
        for j in keep:
            new_big[j] = True
        if new_big.sum() == 0:
            xi = np.zeros(p)
            big = new_big
            break
        xi_new = np.zeros(p)
        xi_new[new_big] = np.linalg.lstsq(Th[:, new_big], y, rcond=None)[0]
        if np.array_equal(new_big, big):
            xi = xi_new
            break
        big, xi = new_big, xi_new
    return xi / norms, big


def cap_prune(Theta, y, supp, coef_cap, keep=(0,), max_iter=10):
    """Post-fit pruning of terms whose PHYSICAL coefficient magnitude
    exceeds coef_cap, refit, repeat to a fixed point. Off when coef_cap
    is None. This is an absolute cutoff and therefore unit dependent
    (the handoff's warning about absolute coefficient cutoffs applies):
    it is offered as an explicit, reported choice for rejecting the
    collinear large-coefficient pairs, not as a default. The constant
    is exempt."""
    if coef_cap is None:
        return supp
    supp = supp.copy()
    for _ in range(max_iter):
        if not supp.any():
            break
        coef = ols_support(Theta, y, supp)
        big = (np.abs(coef) > coef_cap) & supp
        for j in keep:
            big[j] = False
        if not big.any():
            break
        supp[big] = False
    return supp


def r_squared(y, yhat):
    ss = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - np.sum((y - yhat) ** 2) / (ss + 1e-300))


def ols_support(Theta, y, supp):
    coef = np.zeros(Theta.shape[1])
    if supp.any():
        coef[supp] = np.linalg.lstsq(Theta[:, supp], y, rcond=None)[0]
    return coef


def holdout_r2(Theta, y, supp, folds):
    """Mean test R^2 of the support over the spatial folds
    (coefficients refit on each fold's training rows)."""
    vals = []
    for tr, te in folds:
        if tr.sum() < supp.sum() + 2 or te.sum() < 4:
            continue
        coef = ols_support(Theta[tr], y[tr], supp)
        vals.append(r_squared(y[te], Theta[te] @ coef))
    return float(np.mean(vals)) if vals else float("nan")


def fit_target(Theta, y, folds, floors, rel_tol=0.01, n_boot=200,
               boot_frac=0.5, seed=0, coef_cap=None):
    """Floor sweep -> candidate supports (each cap-pruned when coef_cap
    is set) -> Pareto selection by mean holdout R^2 -> full refit +
    bootstrap on the chosen support."""
    p = Theta.shape[1]
    cands = {}
    for fl in floors:
        _, supp = stlsq_floor(Theta, y, fl)
        supp = cap_prune(Theta, y, supp, coef_cap)
        if supp.any():
            cands.setdefault(tuple(supp), []).append(fl)
    for j in range(1, p):                    # every {1, term} model
        s = np.zeros(p, dtype=bool); s[0] = s[j] = True
        s = cap_prune(Theta, y, s, coef_cap)
        if s[1:].any():
            cands.setdefault(tuple(s), []).append(np.nan)

    sweep = []
    for supp_t, fls in cands.items():
        supp = np.array(supp_t)
        sweep.append(dict(support=supp, n_terms=int(supp.sum()),
                          r2_holdout=holdout_r2(Theta, y, supp, folds),
                          floors=[f for f in fls if np.isfinite(f)]))
    best = max(s_["r2_holdout"] for s_ in sweep if np.isfinite(s_["r2_holdout"]))
    admissible = [s_ for s_ in sweep if np.isfinite(s_["r2_holdout"])
                  and s_["r2_holdout"] >= best - rel_tol * abs(best)]
    admissible.sort(key=lambda s_: (s_["n_terms"], -s_["r2_holdout"]))
    chosen = admissible[0]
    supp = chosen["support"]
    coef = ols_support(Theta, y, supp)
    norms = np.linalg.norm(Theta, axis=0)
    contrib = np.abs(coef) * norms / (np.linalg.norm(y) + 1e-300)
    rng = np.random.RandomState(seed)
    n = Theta.shape[0]; m = max(2, int(boot_frac * n))
    boots = np.zeros((n_boot, p))
    for b in range(n_boot):
        idx = rng.randint(0, n, size=m)
        boots[b] = ols_support(Theta[idx], y[idx], supp)
    return dict(support=supp, coef=coef, contrib=contrib,
                boot_std=boots.std(axis=0), r2_holdout=chosen["r2_holdout"],
                r2_full=r_squared(y, Theta @ coef), sweep=sweep,
                floors=chosen["floors"])


def terms_string(res):
    names = res.get("names", LIB_NAMES)
    return "+".join("%s:%+.4f" % (names[j], res["coef"][j])
                    for j in np.where(res["support"])[0])


def form_ok(res, target):
    """Minimal self-dual form: w selected and nothing beyond the constant
    (no w^2, w^3, nor any extended column)."""
    s = res["support"]
    if target == "divk":
        return bool(s[1] and not s[2:].any())
    return bool(s[2] and not s[3:].any() and not s[1])


# ---------------------------------------------------------------------
# one (field, route) fit
# ---------------------------------------------------------------------

def fit_one(u, x, y, jc, mu, route, cfg):
    k1, k2 = ksc.extract_k(route, u, x, y, jc, cfg["alpha_x"], cfg["y_frac"])
    extras = None
    if cfg["domain"] == "pointwise":
        pw = ksc.build_pointwise(k1, k2, x, y, jc, mu, route, cfg["alpha_x"],
                                 cfg["y_frac"], cfg["band_w"], cfg["ff_renorm"])
        prof = pw["prof"]; w, d, yy = pw["w"], pw["divk"], pw["y"]
        if cfg["library"] == "extended":
            extras = pw["extras"]
    else:
        prof = ksc.build_profile(k1, k2, x, y, jc, mu, route, cfg["alpha_x"],
                                 cfg["y_frac"], cfg["band_w"], cfg["ff_renorm"])
        b = prof["band"]; w, d, yy = prof["wbar"][b], prof["dk2"][b], prof["y"][b]
        if cfg["library"] == "extended":
            extras = {"w_y^2": prof["w_y"][b] ** 2, "|w_y|": np.abs(prof["w_y"][b]),
                      "w_yy": prof["w_yy"][b]}
    Theta, names = build_library(w, extras)
    folds = [(yy < 0, yy > 0), (yy > 0, yy < 0)]
    out = dict(prof=prof, k1=k1, k2=k2, w=w, d_samples=d, yy=yy, names=names,
               cond=float(np.linalg.cond(Theta / np.linalg.norm(Theta, axis=0))))
    for tgt in TARGETS:
        yv = d if tgt == "divk" else d ** 2
        out[tgt] = fit_target(Theta, yv, folds, cfg["floors"], cfg["rel_tol"],
                              cfg["n_boot"], cfg["boot_frac"], cfg["seed"],
                              cfg.get("coef_cap"))
        out[tgt]["names"] = names
    r = out["divk"]
    # The shape-free slope of the self-dual form div k = b + c w is the
    # coefficient of record for every route (always defined, never
    # collinear); the STLSQ selection answers the FORM question
    # separately. c_raw is beta-signed (div k = -c_raw w on the lower
    # boundary).
    supp_lin = np.zeros(Theta.shape[1], dtype=bool); supp_lin[0] = supp_lin[1] = True
    coef_lin = ols_support(Theta, d, supp_lin)
    out["c_raw"] = float(-coef_lin[1])
    out["const"] = float(coef_lin[0])
    out["const_over_peak"] = float(coef_lin[0] / (np.abs(d).max() + 1e-300))
    out["divk_peak"] = float(np.abs(d).max())
    out["r2_lin_holdout"] = holdout_r2(Theta, d, supp_lin, folds)
    out["r2_lin_full"] = r_squared(d, Theta @ coef_lin)
    rng = np.random.RandomState(cfg["seed"]); n = Theta.shape[0]; m = max(2, n // 2)
    bs = [ols_support(Theta[idx], d[idx], supp_lin)[1]
          for idx in (rng.randint(0, n, size=m) for _ in range(cfg["n_boot"]))]
    out["c_raw_std"] = float(np.std(bs))
    out["c_stlsq_w"] = float(-r["coef"][1]) if r["support"][1] else float("nan")
    out["r2_gain"] = float(r["r2_holdout"] - out["r2_lin_holdout"])
    return out


# ---------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------

def fig_profile_fit(out_path, res, route, stem, mu, R, cfg, title_extra=""):
    prof = res["prof"]; r = res["divk"]
    yy, w, d = res["yy"], res["w"], res["d_samples"]
    Theta = build_library(w, None)[0] if len(r["coef"]) == 4 else None
    yhat = (Theta @ r["coef"]) if Theta is not None else np.full_like(w, np.nan)
    ylin = res["const"] - res["c_raw"] * w
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    ax = axs[0]
    if cfg["domain"] == "profile":
        ax.plot(prof["y"], prof["dk2"], "-", color="0.6", lw=0.8,
                label=r"$d\bar k_2/dy$ (all rows)")
    ax.plot(yy, d, ".", ms=3, color="0.3", label=r"$\nabla\!\cdot\!\mathbf{k}$ (band)")
    order = np.argsort(yy)
    gaps = np.where(np.diff(yy[order]) > 1.5 * np.median(np.diff(yy[order])))[0] + 1
    _ins = lambda a: np.insert(a[order].astype(float), gaps, np.nan)
    ax.plot(_ins(yy), _ins(ylin), "-", color=ksc.ROUTE_COLORS[route], lw=1.6,
            label=r"$b + c\,w$: $c=%.3f$, $R^2_{\rm hold}=%.4f$" % (res["c_raw"], res["r2_lin_holdout"]))
    if not form_ok(r, "divk"):
        ax.plot(_ins(yy), _ins(yhat), "--", color="k", lw=1.0,
                label="STLSQ: " + terms_string(r).replace("+", ", "))
    ax.axvspan(-prof["band_half"], -prof["seam"], color="C0", alpha=0.05)
    ax.axvspan(prof["seam"], prof["band_half"], color="C0", alpha=0.05)
    ax.set_xlim(-1.3 * prof["band_half"], 1.3 * prof["band_half"])
    ax.set_xlabel("$y$"); ax.set_ylabel(r"$\nabla\!\cdot\!\mathbf{k}$")
    ax.legend(fontsize=7, loc="upper right"); ksc.panel_label(ax, "(a)")
    ax = axs[1]
    step = max(1, w.size // 4000)
    ax.plot(w[::step], d[::step], ".", ms=3, color="0.5", alpha=0.6, label="data")
    wg = np.linspace(min(w.min(), 0), max(w.max(), 0), 200)
    ax.plot(wg, res["const"] - res["c_raw"] * wg, "-", color=ksc.ROUTE_COLORS[route],
            lw=1.6, label=r"$b + c\,w$")
    if not form_ok(r, "divk"):
        if Theta is not None:
            ax.plot(wg, build_library(wg, None)[0] @ r["coef"], "--", color="k", lw=1.0,
                    label=r"STLSQ %d terms ($R^2_{\rm hold}=%.4f$)" % (int(r["support"].sum()), r["r2_holdout"]))
    ax.set_xlabel(r"$w = |\mathbf{k}|^2 - |\mathbf{k}|_{ff}^2$")
    ax.set_ylabel(r"$\nabla\!\cdot\!\mathbf{k}$"); ax.legend(fontsize=7)
    ksc.panel_label(ax, "(b)")
    ax = axs[2]
    ax.axhline(0, color="k", lw=0.6)
    ax.plot(yy, d - ylin, ".", ms=3, color=ksc.ROUTE_COLORS[route], label="linear model")
    if not form_ok(r, "divk"):
        ax.plot(yy, d - yhat, ".", ms=2, color="k", alpha=0.5, label="STLSQ model")
    ax.set_xlabel("$y$"); ax.set_ylabel("residual"); ax.legend(fontsize=7)
    ksc.panel_label(ax, "(c)")
    fig.suptitle(r"%s   $\mu=%g$  $R=%g$   %s   $c_{\rm raw}=%.3f$  %s"
                 % (ksc.ROUTES[route]["label"], mu, R, cfg["domain"],
                    res["c_raw"], title_extra), fontsize=9)
    fig.savefig(out_path); plt.close(fig)


def fig_fields_2d(out_path, res, u, x, y, jc, route, mu, R, cfg):
    """u, w, div k, and the residual div k - (b - c w) of the linear law
    on the 2D crop, with the band rows and flat columns outlined. The
    sanity check behind quoting b as a fraction of the div k peak: no
    singular behaviour anywhere in the fitted region."""
    k1, k2 = res["k1"], res["k2"]; prof = res["prof"]
    dx, dy = x[1] - x[0], y[1] - y[0]
    w2 = k1 ** 2 + k2 ** 2 - 1.0 - (prof["w_ff"] if cfg["ff_renorm"] else 0.0)
    d2 = np.gradient(k1, dx, axis=1) + np.gradient(k2, dy, axis=0)
    resid = d2 - (res["const"] - res["c_raw"] * w2)
    yc = y - y[jc]; cols = prof["cols"]
    ext = [x[0], x[-1], yc[0], yc[-1]]
    band_rows = np.where(prof["band"])[0]
    M = np.zeros_like(w2, dtype=bool); M[prof["band"], cols] = True
    # blank everything the tapers touch: columns outside the flat part of
    # the x window and rows beyond the clear-of-taper distance
    trusted = np.zeros_like(w2, dtype=bool)
    # trim the flat columns by the route's smoothing halo (3 sigma) plus
    # one column, since the kernel of a smoothed route reaches into the
    # flat region from the taper and the central difference at the first
    # flat column uses a tapered neighbour. Measured: trimming the
    # x-average itself by the same amount moves the uHu slope by at most
    # 0.002 and R^2 by 0.0003 on every frame, so the fit keeps the full
    # flat region and only the maps are trimmed.
    # The halo is clamped so that at least a quarter of the flat columns
    # survive: for a wide route (Riesz production, sigma_route = pi, halo
    # 3 sigma = 9.4 units = 75 columns) the halo would otherwise swallow
    # the whole 128-column flat region and leave nothing to draw. A
    # clamped halo is reported in the title; the fit itself never uses
    # the trusted mask.
    n_flat = cols.stop - cols.start
    halo_full = int(np.ceil(3.0 * ksc.route_sigma(route) / dx)) + 1
    halo = min(halo_full, max(n_flat // 2 - max(n_flat // 8, 4), 0))
    halo_note = "" if halo == halo_full else " (halo clamped from %d to %d columns)" % (halo_full, halo)
    trusted[np.abs(yc) < prof["y_clear"], slice(cols.start + halo, cols.stop - halo)] = True
    # statistics on the unblanked fields: the title quotes the maximum of
    # |div k| over the fit region inside the trusted columns (falling back
    # to the full fit region if the two do not intersect), and the colour
    # limits come from the fit region, so neither can be empty
    sel = M & trusted
    d_stat = np.abs(d2[sel]) if sel.any() else np.abs(d2[M])
    d_max = float(np.nanmax(d_stat)) if d_stat.size else float("nan")
    def _lim(f):
        vals = np.abs(f[M]); vals = vals[np.isfinite(vals)]
        v = float(np.percentile(vals, 99.5)) if vals.size else 1.0
        return (-v, v)
    lims = [_lim(w2), _lim(d2), _lim(resid)]
    if cfg.get("mask_taper_in_maps", True):
        w2 = np.where(trusted, w2, np.nan)
        d2 = np.where(trusted, d2, np.nan)
        resid = np.where(trusted, resid, np.nan)
    panels = [("$u$", u, "gray", (-1, 1)),
              (r"$w=|\mathbf{k}|^2-|\mathbf{k}|_{ff}^2$", w2, "seismic", lims[0]),
              (r"$\nabla\!\cdot\!\mathbf{k}$", d2, "seismic", lims[1]),
              (r"residual $\nabla\!\cdot\!\mathbf{k}-(b-c\,w)$", resid, "seismic", lims[2])]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), constrained_layout=True)
    for ax, (ttl, f, cm, lim), pl in zip(axs, panels, "abcd"):
        im = ax.imshow(f, extent=ext, cmap=cm, vmin=lim[0], vmax=lim[1],
                       aspect="auto", origin="lower", rasterized=True)
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.axhline(yc[band_rows[0]], color="k", lw=0.6, ls=":")
        ax.axhline(yc[band_rows[-1]], color="k", lw=0.6, ls=":")
        ax.axvline(x[cols.start], color="k", lw=0.6, ls=":")
        ax.axvline(x[cols.stop - 1], color="k", lw=0.6, ls=":")
        ax.set_title(ttl, fontsize=9); ax.set_xlabel("$x$")
        ksc.panel_label(ax, "(%s)" % pl, color="k" if cm != "gray" else "w")
    axs[0].set_ylabel("$y$")
    fig.suptitle(r"%s  $\mu=%g$ $R=%g$: max$|\nabla\!\cdot\!\mathbf{k}|$ in the fit region %.3f, "
                 r"profile peak %.3f, $b/$peak $=%+.3f$; dotted: band rows and flat columns; taper regions blanked%s"
                 % (ksc.ROUTES[route]["label"], mu, R, d_max,
                    res["divk_peak"], res["const_over_peak"], halo_note), fontsize=9)
    fig.savefig(out_path); plt.close(fig)


def fig_summary(out_dir, rows, cfg):
    """c(mu) per route (raw filled, calibrated open), demod with the
    1 + c(R) mu^2 record law; holdout R^2(mu) per route."""
    routes = cfg["routes"]
    Rvals = sorted({r["R"] for r in rows})
    markers = {0.5: "o", 1.0: "s"}
    fig, axs = plt.subplots(1, len(routes), figsize=(4.2 * len(routes), 3.6),
                            constrained_layout=True, squeeze=False)
    for ax, route in zip(axs[0], routes):
        for R in Rvals:
            sel = [r for r in rows if r["route"] == route and r["R"] == R]
            if not sel:
                continue
            mus = np.array([r["mu"] for r in sel])
            craw = np.array([r["c_raw"] for r in sel])
            mk = markers.get(R, "^")
            ax.errorbar(mus, craw, yerr=[r["c_raw_std"] for r in sel], fmt=mk,
                        color=ksc.ROUTE_COLORS[route], ms=6, capsize=2,
                        label="raw, $R=%g$" % R)
            bad = np.array([not r["form_ok_divk"] for r in sel])
            if bad.any():
                ax.plot(mus[bad], craw[bad], "x", color="r", ms=10, mew=1.5,
                        label="STLSQ selects extra terms" if R == Rvals[0] else None)
            ax.plot(mus, [r["r2_lin_holdout"] for r in sel], visible=False)
            if route != "demod":
                ccal = np.array([r["c_calibrated"] if r["calibration_valid"] else np.nan
                                 for r in sel])
                if np.isfinite(ccal).any():
                    ax.plot(mus, ccal, mk, mfc="none", mec=ksc.ROUTE_COLORS[route],
                            ms=7, label="calibrated, $R=%g$ (form-valid only)" % R)
            else:
                mg = np.linspace(0.28, 0.52, 50)
                c = ksc.RECORD_LAW_C.get(R)
                if c is not None:
                    ax.plot(mg, 1 + c * mg ** 2, "--", color="0.4", lw=1,
                            label=r"$1+%.2f\mu^2$ ($R=%g$, record)" % (c, R))
                rec = np.array([ksc.record_lookup(m, R) for m in mus])
                ax.plot(mus, rec, "x", color="k", ms=6, label="tanh record, $R=%g$" % R)
        ax.axhline(1.0, color="k", lw=0.5, ls=":")
        ax.set_xlabel(r"$\mu$"); ax.set_ylabel("$c$")
        ax.set_title(ksc.ROUTES[route]["label"], fontsize=10)
        ax.legend(fontsize=6, loc="best")
    fig.suptitle("self-dual coefficient by route (raw and calibrated never share a marker)", fontsize=9)
    fig.savefig(Path(out_dir) / "summary_c_vs_mu.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    for route in routes:
        for R in Rvals:
            sel = [r for r in rows if r["route"] == route and r["R"] == R]
            if sel:
                ax.plot([r["mu"] for r in sel], [r["r2_lin_holdout"] for r in sel],
                        markers.get(R, "^") + "-", color=ksc.ROUTE_COLORS[route],
                        label="%s, $R=%g$" % (route, R))
    ax.set_xlabel(r"$\mu$"); ax.set_ylabel(r"holdout $R^2$ of $\nabla\!\cdot\!\mathbf{k} = b + c\,w$")
    ax.legend(fontsize=7); fig.savefig(Path(out_dir) / "summary_holdout_r2.png")
    plt.close(fig)


def fig_response(out_dir, resp, cfg):
    """Regression response per route: c_synth vs planted beta at every
    (mu, R), with the console tanh response for reference."""
    routes = cfg["routes"]
    fig, axs = plt.subplots(1, len(routes), figsize=(4.0 * len(routes), 3.4),
                            constrained_layout=True, squeeze=False)
    for ax, route in zip(axs[0], routes):
        for (mu, R), pts in sorted(resp.get(route, {}).items()):
            bs = [b for b, c in pts]; cs = [c for b, c in pts]
            ax.plot(bs, cs, "o-", ms=4, label=r"$\mu=%g$, $R=%g$" % (mu, R))
        cons = [(b, v[route]) for (m, b), v in ksc.CONSOLE_TANH_RESPONSE.items()
                if route in v]
        if cons:
            cons.sort()
            ax.plot([b for b, v in cons], [v for b, v in cons], "x", color="k",
                    ms=5, label="console tanh response")
        ax.plot([0.88, 1.12], [0.88, 1.12], ":", color="0.5", lw=0.8)
        ax.set_xlabel(r"planted $\beta$"); ax.set_ylabel("$c$ recovered")
        ax.set_title(ksc.ROUTES[route]["label"], fontsize=10)
        ax.legend(fontsize=6)
    fig.savefig(Path(out_dir) / "summary_instrument_response.png"); plt.close(fig)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main(cfg):
    ksc.setup_method_layer(cfg.get("method_dirs", ()))
    ksc.use_house_style()
    runs = ksc.discover_runs(cfg["roots"], cfg["pattern"], cfg["mu_min"],
                             cfg["mu_max"], cfg["R_values"])
    if not runs:
        print("No runs found under", cfg["roots"]); return []
    cfg["R_values"] = sorted({r["R"] for r in runs}) if not cfg["R_values"] else cfg["R_values"]
    cfg["mu_min"] = min(r["mu"] for r in runs) if cfg["mu_min"] is None else cfg["mu_min"]
    cfg["mu_max"] = max(r["mu"] for r in runs) if cfg["mu_max"] is None else cfg["mu_max"]
    out_dir = Path(cfg["out_root"]) / ksc.config_stamp(cfg, "sdsindy")
    out_dir.mkdir(parents=True, exist_ok=True)
    ksc.save_config(out_dir, cfg)
    print("output ->", out_dir)
    print("runs:", ", ".join("%s(mu=%g,R=%g)" % (r["stem"][:12], r["mu"], r["R"]) for r in runs))

    # ---------------- validation / calibration ----------------
    resp = {route: {} for route in cfg["routes"]}       # route -> (mu,R) -> [(b, c)]
    vstat = {}                                          # (mu,R,route) -> status
    if cfg["validate"]:
        print("\n== synthetic-knee check (planted beta, data grid, every route) ==")
        print("%6s %4s %-9s | %s" % ("mu", "R", "route",
              "  ".join("b=%.1f: c (form, R2h)" % b for b in cfg["planted"])))
        for run in runs:
            x, y, _, _ = ksc.load_frame(run["path"], cfg["frame"])
            mu, R, jc = run["mu"], run["R"], run["jc"]
            for route in cfg["routes"]:
                cells = []; forms = []; devs = []
                for b in cfg["planted"]:
                    u_s = ksc.synth_knee_on_grid(x, y, mu, R, b, jc)
                    res = fit_one(u_s, x, y, jc, mu, route, cfg)
                    fo = form_ok(res["divk"], "divk")
                    forms.append(fo); devs.append(res["c_raw"] / b - 1.0)
                    resp[route].setdefault((mu, R), []).append((b, res["c_raw"]))
                    cells.append("%.4f (%s, %.4f)" % (res["c_raw"], "ok" if fo else "FAIL",
                                                      res["divk"]["r2_holdout"]))
                    if cfg["synth_figures"]:
                        sd = out_dir / ("synthetic_mu%g_R%g" % (mu, R)); sd.mkdir(exist_ok=True)
                        fig_profile_fit(sd / ("%s_beta%.2f.png" % (route, b)), res, route,
                                        "synthetic", mu, R, cfg, "planted %.2f" % b)
                st = "FORM_OK" if all(forms) else "FORM_FAIL[b=%s]" % ",".join(
                    "%.1f" % b for b, f in zip(cfg["planted"], forms) if not f)
                if route == "demod":
                    mx = 100 * max(abs(v) for v in devs)
                    st += ";VALUE_%s(%.1f%%)" % ("OK" if mx < 100 * cfg["value_tol"] else "MISS", mx)
                else:
                    c1 = [c for b, c in resp[route][(mu, R)] if abs(b - 1.0) < 1e-9]
                    st += ";RESPONSE_RECORDED(c@1=%.3f)" % (c1[0] if c1 else np.nan)
                vstat[(mu, R, route)] = st
                print("%6.3f %4.1f %-9s | %s   -> %s" % (mu, R, route, "  ".join(cells), st))
        print("   (uhu/riesz: no pass/fail on the value; console tanh response at b=1:",
              ", ".join("%s %.3f/%.3f" % (rt, ksc.CONSOLE_TANH_RESPONSE[(0.375, 1.0)].get(rt, np.nan),
                                          ksc.CONSOLE_TANH_RESPONSE[(0.475, 1.0)].get(rt, np.nan))
                        for rt in cfg["routes"] if rt != "demod"), "at mu 0.375/0.475)")
        fig_response(out_dir, resp, cfg)

    # ---------------- data ----------------
    print("\n== simulation frames (%s domain) ==" % cfg["domain"])
    print("%6s %4s %-9s | %8s %8s %7s %6s | %-32s %7s %+6s | %-22s %7s | %s" % (
        "mu", "R", "route", "c_raw", "c_cal", "R2lin", "const", "STLSQ terms (div k)", "R2hold",
        "gain", "STLSQ terms (div k)^2", "R2hold", "status"))
    rows = []
    misses = []
    for run in runs:
        x, y, u, fidx = ksc.load_frame(run["path"], cfg["frame"])
        mu, R, jc = run["mu"], run["R"], run["jc"]
        stem_dir = out_dir / run["stem"]; stem_dir.mkdir(exist_ok=True)
        tb, mff_rec, breg_rec, r2reg_rec = ksc.tanh_beta_of_record(u, x, y, jc, mu)
        rec = ksc.record_lookup(mu, R)
        dev_tanh = 100 * (tb / rec - 1) if np.isfinite(rec) else np.nan
        if np.isfinite(rec) and abs(dev_tanh) > 100 * cfg["record_tol_tanh"]:
            misses.append("tanh estimator of record at mu=%g R=%g reads %.4f vs record %.3f (%+.2f%%)"
                          % (mu, R, tb, rec, dev_tanh))
        for route in cfg["routes"]:
            res = fit_one(u, x, y, jc, mu, route, cfg)
            r = res["divk"]; r2 = res["divk_sq"]
            pts = resp.get(route, {}).get((mu, R), [])
            c_cal = ksc.calibrate(res["c_raw"], pts) if route != "demod" else float("nan")
            st = vstat.get((mu, R, route), "NOT_RUN")
            st += ";DATA_FORM_%s" % ("OK" if form_ok(r, "divk") else "FAIL")
            cal_ok = bool(route != "demod" and st.startswith("FORM_OK") and form_ok(r, "divk"))
            if route != "demod" and not cal_ok:
                st += ";CAL_UNRELIABLE"
            dev_sindy = 100 * (res["c_raw"] / rec - 1) if (route == "demod" and np.isfinite(rec)) else np.nan
            if route == "demod" and np.isfinite(rec) and abs(dev_sindy) > 100 * cfg["record_tol_sindy"]:
                misses.append("demod SINDy slope at mu=%g R=%g reads %.4f vs tanh record %.3f (%+.1f%%)"
                              % (mu, R, res["c_raw"], rec, dev_sindy))
            row = dict(
                mu=mu, R=R, route=route, domain=cfg["domain"],
                c_raw=res["c_raw"], c_raw_std=res["c_raw_std"],
                c_calibrated=c_cal, calibration_valid=cal_ok,
                r2_lin_holdout=res["r2_lin_holdout"], r2_lin_full=res["r2_lin_full"],
                const=res["const"], const_over_peak=res["const_over_peak"],
                divk_peak=res["divk_peak"], library=cfg["library"],
                c_stlsq_w=res["c_stlsq_w"], r2_gain_stlsq=res["r2_gain"],
                w_ff=res["prof"]["w_ff"], k1_ff=res["prof"]["k1_ff"], m_ff=res["prof"]["m_ff"],
                terms_divk=terms_string(r), form_ok_divk=form_ok(r, "divk"),
                holdout_r2_divk=r["r2_holdout"], r2_full_divk=r["r2_full"],
                contrib_divk=";".join("%s:%.3f" % (n, c) for n, c in zip(LIB_NAMES, r["contrib"])),
                terms_divk_sq=terms_string(r2), form_ok_divk_sq=form_ok(r2, "divk_sq"),
                coef_w2=float(r2["coef"][2]),
                sqrt_coef_w2=float(np.sqrt(max(r2["coef"][2], 0.0))),
                holdout_r2_divk_sq=r2["r2_holdout"],
                lib_cond=res["cond"],
                tanh_beta_record_rerun=tb, m_ff_record_rerun=mff_rec,
                record_value=rec, record_dev_tanh_pct=dev_tanh,
                record_dev_sindy_pct=dev_sindy,
                synth_response=",".join("%.1f:%.4f" % (b, c) for b, c in pts),
                validation_status=st, band_half=res["prof"]["band_half"],
                n_samples=int(res["w"].size), frame_index=fidx, stem=run["stem"],
                path=run["path"])
            rows.append(row)
            print("%6.3f %4.1f %-9s | %8.4f %8.4f %7.4f %+6.3f | %-32s %7.4f %+6.4f | %-22s %7.4f | %s" % (
                mu, R, route, res["c_raw"], c_cal, res["r2_lin_holdout"], res["const"],
                terms_string(r)[:32], r["r2_holdout"], res["r2_gain"],
                terms_string(r2)[:22], r2["r2_holdout"], st))
            fig_profile_fit(stem_dir / ("%s_profile_fit_residual.png" % route), res, route,
                            run["stem"], mu, R, cfg,
                            "$R^2_{hold}$=%.4f" % r["r2_holdout"])
            if cfg["field_maps"]:
                fig_fields_2d(stem_dir / ("%s_fields_2d.png" % route), res, u, x, y, jc,
                              route, mu, R, cfg)
        print("       tanh estimator of record (untapered): %.4f  record %.3f  (%+.2f%%)  m_ff %.3f"
              % (tb, rec, dev_tanh, mff_rec))

    # ---------------- outputs ----------------
    csv_path = out_dir / "self_dual_sindy_sweep.csv"
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    fig_summary(out_dir, rows, cfg)
    print("\n== consistency with the record ==")
    if misses:
        for m in misses:
            print("   !!! MISS: " + m)
    else:
        print("   tanh estimator reproduces RECORD_BETA within %.1f%% on every frame;"
              " demod SINDy slopes within %.0f%%." % (100 * cfg["record_tol_tanh"],
                                                     100 * cfg["record_tol_sindy"]))
    print("\n== form summary ==")
    for route in cfg["routes"]:
        sel = [r for r in rows if r["route"] == route]
        ok = sum(1 for r in sel if r["form_ok_divk"])
        r2s = [r["r2_lin_holdout"] for r in sel]
        bp = [abs(r["const_over_peak"]) for r in sel]
        print("   %-9s linear self-dual law div k = b - c w on all %d frames: holdout R^2 "
              "%.4f .. %.4f, c %.3f .. %.3f, |b|/peak %.1f%% .. %.1f%%; STLSQ adds higher "
              "terms on %d/%d frames" % (route, len(sel), min(r2s), max(r2s),
                 min(r["c_raw"] for r in sel), max(r["c_raw"] for r in sel),
                 100 * min(bp), 100 * max(bp), len(sel) - ok, len(sel)))
    print("\nwrote", csv_path)
    return rows


# ---------------------------------------------------------------------
# config
# ---------------------------------------------------------------------

class _Args:
    roots = [
        str(_HERE / "data" / "sh_pgb_zigzag" / "mu_sweeps_shallow_v2_Rp5_Nx512_tmax500_hp5_np18_Ny5" / "raw" / "selected"),
        str(_HERE / "data" / "sh_pgb_zigzag" / "mu_sweeps_shallow_v2_R1p0_Nx512_tmax500_hp5_np18_Ny5" / "raw" / "selected")]
    pattern = "*sh_pgb_zigzag*mu*.npz"
    mu_min = None
    mu_max = None
    R_values = []                 # [] = all found
    routes = ["uhu", "riesz", "riesz_pi2", "demod", "demod_sm"]
    domain = "profile"            # 'profile' | 'pointwise'
    library = "extended"             # 'basic' [1,w,w^2,w^3] | 'extended' (+ derivative columns)
    field_maps = True             # per-stem 2D u / w / div k / residual maps
    mask_taper_in_maps = True     # blank tapered columns and taper-halo rows in the maps
    coef_cap = 10.0               # e.g. 10.0: drop terms with |coef| > cap after the fit (unit dependent; off by default)
    frame = -1
    validate = True
    planted = [0.9, 1.0, 1.1]
    value_tol = 0.02              # demod synthetic value criterion
    record_tol_tanh = 0.005       # same estimator, same frames
    record_tol_sindy = 0.05       # regression vs tanh reading
    alpha_x = 0.5                 # Tukey x taper
    y_frac = 0.06                 # y edge taper (spectral routes)
    band_w = 3.0                  # band |y - yc| < band_w / mu
    ff_renorm = True
    floors = list(np.geomspace(1e-3, 1.0, 25))
    rel_tol = 0.01
    n_boot = 200
    boot_frac = 0.5
    seed = 0
    synth_figures = False
    out_root = str(_HERE / "results" / "self_dual_sindy_sweep_v3")
    method_dirs = []


def parse_cli():
    d = _Args
    p = argparse.ArgumentParser(description="stationary self-dual SINDy sweep on knee frames")
    p.add_argument("--roots", nargs="+", default=d.roots)
    p.add_argument("--pattern", default=d.pattern)
    p.add_argument("--mu-min", type=float, default=d.mu_min)
    p.add_argument("--mu-max", type=float, default=d.mu_max)
    p.add_argument("--R", nargs="*", type=float, default=d.R_values)
    p.add_argument("--routes", nargs="+", default=d.routes, choices=list(ksc.ROUTES))
    p.add_argument("--domain", default=d.domain, choices=["profile", "pointwise"])
    p.add_argument("--library", default=d.library, choices=["basic", "extended"])
    p.add_argument("--no-field-maps", action="store_true")
    p.add_argument("--coef-cap", type=float, default=d.coef_cap)
    p.add_argument("--frame", type=int, default=d.frame)
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--synth-figures", action="store_true")
    p.add_argument("--no-ff-renorm", action="store_true")
    p.add_argument("--alpha-x", type=float, default=d.alpha_x)
    p.add_argument("--band-w", type=float, default=d.band_w)
    p.add_argument("--n-boot", type=int, default=d.n_boot)
    p.add_argument("--out-root", default=d.out_root)
    p.add_argument("--method-dirs", nargs="*", default=d.method_dirs)
    a = p.parse_args()
    cfg = dict(roots=a.roots, pattern=a.pattern, mu_min=a.mu_min, mu_max=a.mu_max,
               R_values=a.R, routes=a.routes, domain=a.domain, library=a.library,
               field_maps=not a.no_field_maps, mask_taper_in_maps=d.mask_taper_in_maps,
               coef_cap=a.coef_cap, frame=a.frame,
               validate=not a.no_validate, planted=d.planted, value_tol=d.value_tol,
               record_tol_tanh=d.record_tol_tanh, record_tol_sindy=d.record_tol_sindy,
               alpha_x=a.alpha_x, y_frac=d.y_frac, band_w=a.band_w,
               ff_renorm=not a.no_ff_renorm, floors=d.floors, rel_tol=d.rel_tol,
               n_boot=a.n_boot, boot_frac=d.boot_frac, seed=d.seed,
               synth_figures=a.synth_figures, out_root=a.out_root,
               method_dirs=a.method_dirs)
    return cfg


def cfg_from_args():
    return {k: getattr(_Args, k) for k in vars(_Args) if not k.startswith("_")}


if __name__ == "__main__":
    main(parse_cli() if len(sys.argv) > 1 else cfg_from_args())