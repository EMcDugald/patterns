# pipelines/time_sindy_riesz_v3.py
"""
Time-SINDy v3. Same bowtie-hull domain and core-hole sweep as v2, plus three
patches and one comparison, none of which touch the frozen method layer:

  1. --w_source {direct, grad, both}
     Build w and the whole w-family (w, w^2, Lap w, |grad w|, div k, (div k)^2,
     |grad k|^2) from either kmag_direct/(k1,k2)_direct (the production director
     route) or kmag_grad/(k1,k2)_grad (the gradient-of-phase route). 'both' fits
     each and reports the difference, so you can see whether the dw/dt w^2 term
     is real physics or a direct-route artifact.

  2. condition-number guard (--cond_max)
     After the Pareto support is chosen, if the design on that support is worse-
     conditioned than cond_max, drop the highest-VIF term and refit, repeating
     until conditioning is acceptable. Kills the rho_d/J_or 1e5 cancellations in
     dtheta/dt without trusting the giant coefficients. Reports what it dropped.

  3. min-sample / degeneracy floor (--min_samples, --min_target_std)
     Skip a (hole, window) whose sample count is below the floor OR whose target
     has near-zero variance (the exhausted-domain 'R2=0, holdout=1.000' case),
     instead of emitting a constant fit.

  4. configurable libraries
     TERMS[target] is an editable list of column names; a user supplies the
     combos they want to test. The w-family names resolve per route
     automatically. Defaults reproduce v2 plus (div k)^2 in the dw_dt library.

Both-domain comparison: for w_source='both', each target/window/hole is fit four
ways -- direct and grad, each on its OWN finite domain and on the SHARED finite
domain (rows finite for both routes). direct@shared vs grad@shared is the clean
route comparison (identical samples); own-vs-shared shows the domain effect, so a
term change cannot be confused with the domain moving.

Run (smoke): python time_sindy_riesz_v3.py <op_path> --w_source both
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from run_riesz_metrics_zigzag import (          # noqa: E402
    load_op, ensure_theta_smooth, frame_fields, nan_grad,
)
from time_sindy_riesz import (                            # noqa: E402
    detect_freeze_frame, build_time_samples as base_build_samples,
    fit_window as base_fit_window, _Args as BaseArgs,
)
from time_sindy_riesz_v2 import compute_hull_domain, format_pde  # noqa: E402
from time_sindy_diagnostics import (                       # noqa: E402
    fig_reconstruction, fig_feature_overlap, fig_query_points,
    _resolve_u, _pick_frame,
)
from self_dual_sindy import fit_target, fig_library_correlation  # noqa: E402


# ---------------------------------------------------------------------------
# library configuration
# ---------------------------------------------------------------------------

W_FAMILY = {"w", "w^2", "w^3", "Lap w", "|grad w|",
            "div k", "(div k)^2", "|grad k|^2"}

# editable per-target term lists. w-family names resolve per route.
DEFAULT_TERMS = {
    "dA_dt": ["1", "w", "w^2", "dA", "dA^2", "J_or", "J_th", "rho_d",
              "curl k", "A", "A^3", "Lap A"],
    "dw_dt": ["1", "w", "w^2", "dA", "dA^2", "J_or", "J_th", "rho_d",
              "curl k", "Lap w", "div k", "|grad w|", "(div k)^2"],
    "dth_dt": ["1", "w", "w^2", "dA", "dA^2", "J_or", "J_th", "rho_d",
               "curl k", "th_xx", "th_yy"],
}

ROUTE_KMAG = {"direct": "kmag_direct", "grad": "kmag_grad"}
ROUTE_K = {"direct": ("k1_direct", "k2_direct"),
           "grad": ("k1_grad", "k2_grad")}


def _resolve(term, route):
    return f"{term}@{route}" if term in W_FAMILY else term


# ---------------------------------------------------------------------------
# per-frame column construction (both routes at once)
# ---------------------------------------------------------------------------

from scipy.ndimage import gaussian_filter, gaussian_filter1d   # noqa: E402

# derivative order (in derivatives of the extracted primitive k or theta); the
# post-hoc smoother can scale its width by this, since higher-derivative terms
# amplify the extraction artifact more (div k chevrons, Lap w bumps).
DERIV_ORDER = {
    "1": 0, "w": 0, "w^2": 0, "w^3": 0, "dA": 0, "dA^2": 0, "A": 0, "A^3": 0,
    "J_or": 1, "J_th": 1, "rho_d": 2, "curl k": 1, "th_xx": 2, "th_yy": 2,
    "Lap A": 2, "Lap w": 2, "|grad w|": 1, "div k": 1, "(div k)^2": 1,
    "|grad k|^2": 1,
}
# which base field a w-family TERM is smoothed through (smooth-then-combine)
_TERM_BASE = {"w": "w", "w^2": "w", "w^3": "w", "|grad w|": "grad w",
              "Lap w": "Lap w", "div k": "div k", "(div k)^2": "div k",
              "|grad k|^2": "gradk"}


def _term_smooth_px(term, args, lam_x, dx):
    """Smoothing width in pixels for a term, per the smoother config.
    mode 'off' -> 0; 'list' -> base width if term listed; 'deriv' -> base width
    times derivative order (0-derivative terms unsmoothed)."""
    if args.smooth_mode == "off" or args.smooth_width <= 0:
        return 0.0
    base_px = args.smooth_width * lam_x / dx
    if args.smooth_mode == "list":
        return base_px if term in args.smooth_terms else 0.0
    if args.smooth_mode == "deriv":
        return base_px * DERIV_ORDER.get(term, 0)
    return 0.0


def _nan_gauss(f, sp):
    """NaN-aware Gaussian (normalized convolution): smooths over finite values
    only and keeps the original NaN footprint, so it does not grow the eroded
    region the way plain gaussian_filter would."""
    if not sp or sp <= 0:
        return f
    m = np.isfinite(f).astype(float)
    f0 = np.where(m > 0, f, 0.0)
    num = gaussian_filter(f0, sp, mode="nearest")
    den = gaussian_filter(m, sp, mode="nearest")
    out = num / np.maximum(den, 1e-6)
    out[m == 0] = np.nan
    return out


def _wfamily(km, k1, k2, dx, dy, base_sw=None):
    """w-family columns. base_sw maps base-field name ('w','grad w','div k',
    'Lap w','gradk') -> smoothing pixels; smoothing is applied (NaN-aware) to
    the base derivative field BEFORE squares/magnitudes are formed
    (smooth-then-combine), the ordering that recovers the true (div k)^2."""
    base_sw = base_sw or {}

    def S(f, key):
        return _nan_gauss(f, base_sw.get(key, 0.0))

    w = km ** 2 - 1.0
    w_s = S(w, "w")
    wx, wy = nan_grad(w, dx, dy)
    gradw = np.sqrt(S(wx, "grad w") ** 2 + S(wy, "grad w") ** 2)
    wxx, _ = nan_grad(wx, dx, dy)
    _, wyy = nan_grad(wy, dx, dy)
    lapw = S(wxx + wyy, "Lap w")
    k1x, k1y = nan_grad(k1, dx, dy)
    k2x, k2y = nan_grad(k2, dx, dy)
    divk = S(k1x + k2y, "div k")
    gradk2 = S(k1x ** 2 + k1y ** 2 + k2x ** 2 + k2y ** 2, "gradk")
    return {
        "w": w_s, "w^2": w_s ** 2, "w^3": w_s ** 3,
        "Lap w": lapw, "|grad w|": gradw,
        "div k": divk, "(div k)^2": divk ** 2, "|grad k|^2": gradk2,
    }


def _tsmoothed_grad(op, args):
    """Temporally smooth the grad-route wavenumber. Returns a kmag_grad-shaped
    array to use in place of op['kmag_grad'] for the grad route, or None.
      tsmooth_on 'kmag'  : Gaussian-in-time smooth of kmag_grad
      tsmooth_on 'theta' : Gaussian-in-time smooth of theta_smooth, then
                           kmag = |grad(theta_t-smoothed)| per frame"""
    if args.tsmooth <= 0:
        return None
    s = float(args.tsmooth)
    if args.tsmooth_on == "kmag":
        return gaussian_filter1d(op["kmag_grad"].astype(float), s, axis=2,
                                 mode="nearest")
    if args.tsmooth_on == "theta":
        th = gaussian_filter1d(op["theta_smooth"].astype(float), s, axis=2,
                               mode="nearest")
        x, y = op["x"], op["y"]
        dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
        out = np.empty_like(th)
        for t in range(th.shape[-1]):
            gx, gy = nan_grad(th[:, :, t], dx, dy)
            out[:, :, t] = np.sqrt(gx ** 2 + gy ** 2)
        return out
    return None


def build_samples(op, args, drop, y_cap, x_trim, hole, routes):
    """Assemble band samples over interior frames with BOTH routes' columns and
    all targets. Applies the post-hoc library smoother (per-term, config in
    args) and, on the grad route, the temporal smoother (precomputed)."""
    x, y = op["x"], op["y"]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    core = int(op["core_row"]); y_core = float(y[core])
    tt = op["tt"]; Nt = op["theta"].shape[-1]
    Xc, Yc = np.meshgrid(x, y)
    A_far = op["A_far_median"] if "A_far_median" in op else None
    mu = float(op["mu"]); lam_x = 2 * np.pi / np.sqrt(max(1 - mu ** 2, 1e-9))

    # smoothing widths (pixels): w-family base fields per route, and shared terms
    base_sw = {bf: 0.0 for bf in ("w", "grad w", "div k", "Lap w", "gradk")}
    for term, bf in _TERM_BASE.items():
        base_sw[bf] = max(base_sw[bf], _term_smooth_px(term, args, lam_x, dx))
    shared_sw = {t: _term_smooth_px(t, args, lam_x, dx)
                 for t in ("curl k", "th_xx", "th_yy", "Lap A",
                           "J_or", "J_th", "rho_d")}

    kmag_grad_ts = _tsmoothed_grad(op, args)   # None unless tsmooth>0

    band = ((np.abs(Yc - y_core) <= y_cap) & (np.abs(Yc - y_core) >= hole)
            & (Xc > x[0] + x_trim) & (Xc < x[-1] - x_trim) & (~drop))

    def _S(f, key):
        return _nan_gauss(f, shared_sw.get(key, 0.0))

    colrows, tgtrows, xrows, trows, uprows = [], [], [], [], []
    for t in range(1, Nt - 1):
        dt2 = float(tt[t + 1] - tt[t - 1])
        if dt2 <= 0:
            continue
        F = frame_fields(op, t, {})
        A = op["A_mono"][:, :, t].astype(float)
        af = float(A_far[t]) if A_far is not None else float(np.nanmedian(A[band]))
        dA = A / max(af, 1e-12) - 1.0
        Ax, Ay = nan_grad(A, dx, dy)
        Axx, _ = nan_grad(Ax, dx, dy); _, Ayy = nan_grad(Ay, dx, dy)
        th = op["theta_smooth"][:, :, t].astype(float)
        tx, ty = nan_grad(th, dx, dy)
        txx, _ = nan_grad(tx, dx, dy); _, tyy = nan_grad(ty, dx, dy)

        cols = {
            "1": np.ones_like(A), "A": A, "A^3": A ** 3, "dA": dA,
            "dA^2": dA ** 2, "Lap A": _S(Axx + Ayy, "Lap A"),
            "J_or": _S(F["J_or"], "J_or"), "J_th": _S(F["J_th"], "J_th"),
            "rho_d": _S(F["rho_d"], "rho_d"), "curl k": _S(F["curl_or"], "curl k"),
            "th_xx": _S(txx, "th_xx"), "th_yy": _S(tyy, "th_yy"),
        }
        tgts = {
            "dA_dt": (op["A_mono"][:, :, t + 1].astype(float)
                      - op["A_mono"][:, :, t - 1].astype(float)) / dt2,
            "dth_dt": (op["theta_smooth"][:, :, t + 1].astype(float)
                       - op["theta_smooth"][:, :, t - 1].astype(float)) / dt2,
        }
        for r in routes:
            if r == "grad" and kmag_grad_ts is not None:
                km = kmag_grad_ts[:, :, t]
                kmp = kmag_grad_ts[:, :, t + 1]; kmm = kmag_grad_ts[:, :, t - 1]
            else:
                km = op[ROUTE_KMAG[r]][:, :, t].astype(float)
                kmp = op[ROUTE_KMAG[r]][:, :, t + 1].astype(float)
                kmm = op[ROUTE_KMAG[r]][:, :, t - 1].astype(float)
            k1 = op[ROUTE_K[r][0]][:, :, t].astype(float)
            k2 = op[ROUTE_K[r][1]][:, :, t].astype(float)
            wf = _wfamily(km, k1, k2, dx, dy, base_sw=base_sw)
            for nm, arr in wf.items():
                cols[f"{nm}@{r}"] = arr
            tgts[f"dw_dt@{r}"] = (kmp ** 2 - kmm ** 2) / dt2

        m = band
        colrows.append({k: v[m] for k, v in cols.items()})
        tgtrows.append({k: v[m] for k, v in tgts.items()})
        xrows.append(Xc[m]); trows.append(np.full(int(m.sum()), t, np.int32))
        uprows.append((Yc[m] >= y_core))

    if not colrows:
        return None
    cols = {k: np.concatenate([c[k] for c in colrows]) for k in colrows[0]}
    tgts = {k: np.concatenate([c[k] for c in tgtrows]) for k in tgtrows[0]}
    return dict(cols=cols, targets=tgts,
                x_pos=np.concatenate(xrows),
                t_index=np.concatenate(trows),
                is_up=np.concatenate(uprows))


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def _colnorm_cond(X):
    n = np.linalg.norm(X, axis=0)
    n[n < 1e-12] = 1.0
    return np.linalg.cond(X / n)


def _vifs(X):
    out = []
    for j in range(X.shape[1]):
        others = [k for k in range(X.shape[1]) if k != j]
        if not others:
            out.append(1.0); continue
        b, *_ = np.linalg.lstsq(X[:, others], X[:, j], rcond=None)
        ss = np.sum((X[:, j] - X[:, others] @ b) ** 2)
        tot = max(np.sum((X[:, j] - X[:, j].mean()) ** 2), 1e-12)
        out.append(1.0 / max(ss / tot, 1e-6))
    return np.array(out)


def condition_guard(names, Theta, yv, support, vif_min, sc_max):
    """Prune only genuine collinear cancellations: a term is dropped iff it is
    BOTH collinear (VIF > vif_min) AND carries a large standardized contribution
    (|coef|*std(col) > sc_max, in target units). This spares real collinear
    blocks like the amplitude cubic (high VIF but O(1) standardized coef) and
    catches the rho_d/J_or 1e5 cancellations (high VIF and huge standardized
    coef). Returns (support, coef, r2, dropped, max_scoef)."""
    supp = [i for i in support if names[i] != "1"]
    const = [i for i in support if names[i] == "1"]
    dropped = []
    while len(supp) >= 1:
        idx = const + supp
        X = Theta[:, idx]
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        Xs = Theta[:, supp]
        scoef = np.abs(beta[len(const):]) * Xs.std(axis=0)
        vif = _vifs(Xs)
        cand = [j for j in range(len(supp))
                if vif[j] > vif_min and scoef[j] > sc_max]
        if not cand or len(supp) == 1:
            break
        j = max(cand, key=lambda k: vif[k])
        dropped.append(names[supp[j]]); supp.pop(j)
    idx = const + supp
    X = Theta[:, idx]
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    pred = X @ beta
    r2 = 1.0 - np.sum((yv - pred) ** 2) / max(np.sum((yv - yv.mean()) ** 2), 1e-30)
    coef = {names[idx[k]]: float(beta[k]) for k in range(len(idx))}
    scoef_all = np.abs(beta) * X.std(axis=0)
    return idx, coef, float(r2), dropped, float(scoef_all.max())


# ---------------------------------------------------------------------------
# one fit (route + domain), with floor + guard
# ---------------------------------------------------------------------------

def fit_case(data, target_key, term_names, mask, args, rng, label):
    yv = data["targets"][target_key][mask]
    Theta = np.column_stack([data["cols"][nm][mask] for nm in term_names])
    xs = data["x_pos"][mask]
    fin = np.all(np.isfinite(Theta), axis=1) & np.isfinite(yv)
    Theta, yv, xs = Theta[fin], yv[fin], xs[fin]
    n = len(yv)
    # ---- floor guard ----
    if n < args.min_samples:
        return dict(skip=f"floor: N={n} < {args.min_samples}")
    tstd = float(np.std(yv))
    if tstd < args.min_target_std:
        return dict(skip=f"floor: target std {tstd:.2e} < {args.min_target_std:.0e} (exhausted)")
    if n > args.max_samples:
        i = rng.choice(n, args.max_samples, replace=False)
        Theta, yv, xs = Theta[i], yv[i], xs[i]
    res = fit_target(term_names, Theta, yv, args, rng, xs < np.median(xs))
    support = list(np.where(res["support"])[0])
    r2_full = float(res["r2_full"]); r2_hold = float(res["chosen"]["r2_test"])
    coefs = {term_names[i]: float(res["coef_full"][i]) for i in support}
    raw_cond = _colnorm_cond(Theta[:, support]) if support else 1.0
    dropped = []; max_scoef = 0.0
    # ---- condition-number guard (VIF + standardized-coefficient gated) ----
    if args.guard and support:
        idx, coefs, r2_full, dropped, max_scoef = condition_guard(
            term_names, Theta, yv, support, args.guard_vif, args.guard_scoef)
        support = idx
        tr = xs < np.median(xs)
        Xg = Theta[:, support]
        b, *_ = np.linalg.lstsq(Xg[tr], yv[tr], rcond=None)
        pr = Xg[~tr] @ b
        r2_hold = 1.0 - np.sum((yv[~tr] - pr) ** 2) / max(
            np.sum((yv[~tr] - yv[~tr].mean()) ** 2), 1e-30)
    return dict(n=n, coefs=coefs, r2_full=r2_full, r2_holdout=float(r2_hold),
                dropped=dropped, cond=float(raw_cond), max_scoef=float(max_scoef),
                has_w2=any(k.startswith("w^2") for k in coefs), label=label)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

@dataclass
class _Args:
    op_path: str = ""
    w_source: str = "both"            # direct | grad | both
    hole_units: tuple = (0, 1, 2, 3)
    margin_units: float = 1.0
    terms: dict = field(default_factory=lambda: {k: list(v)
                                                 for k, v in DEFAULT_TERMS.items()})
    cond_max: float = 1e3             # reported diagnostic only (see guard_*)
    guard: bool = True                # prune genuine collinear cancellations
    guard_vif: float = 30.0           # a term must be this collinear to be prunable
    guard_scoef: float = 20.0         # ...AND contribute this much (|coef|*std) to be pruned
    min_samples: int = 2000
    min_target_std: float = 1e-9      # exhausted-domain floor
    # ---- post-hoc library smoother ----
    smooth_mode: str = "off"          # off | list | deriv
    smooth_width: float = 0.25        # base width in units of lambda_x
    smooth_terms: tuple = ("div k", "(div k)^2")   # for smooth_mode='list'
    # ---- temporal smoother (grad route) ----
    tsmooth: float = 0.0              # Gaussian sigma in frames (0 = off)
    tsmooth_on: str = "kmag"          # kmag | theta
    # ---- v2-style figures (direct route) ----
    figs: bool = False
    figs_all: bool = False            # render figures for all three targets
    fig_hole: float = 0.0             # which core hole (units of pi/mu) to render
    fig_target: str = "dw_dt"         # dA_dt | dw_dt | dth_dt
    fig_window: str = "post-freeze"   # full | pre-freeze | post-freeze
    u_from: str = ""                  # raw SH .npz/dir for query-points underlay
    max_samples: int = 400000
    theta_smooth_w: float = 0.75
    thresholds: tuple = tuple(np.geomspace(1e-4, 1.0, 25))
    rel_tol_pareto: float = 0.01
    n_boot: int = 60
    boot_frac: float = 0.5
    seed: int = 0
    out_root: str = ""


def _target_key(base, route):
    return f"{base}@{route}" if base == "dw_dt" else base


def render_v2_figs(op, args, drop, y_cap, x_trim, hpx, fz, out, rng, target=None):
    """Reproduce the v2 diagnostic figures (reconstruction, feature-overlap,
    query-points, library-correlation) for the DIRECT route at the chosen hole,
    by delegating to the base build_time_samples + fit_window + diagnostics, the
    same pipeline v2 uses. grad-route / smoothed-term figure variants are a
    follow-up; query-points needs raw u via --u_from (else no underlay)."""
    from copy import copy
    hole = args.fig_hole * hpx
    a = BaseArgs(op_path=args.op_path)
    a.band_y = float(y_cap); a.x_trim = float(x_trim); a.band_in = float(hole)
    a.theta_smooth_w = args.theta_smooth_w
    op_work = dict(op)
    for fld in ("theta_smooth", "theta"):
        if fld in op:
            arr = np.array(op[fld], dtype=float); arr[drop] = np.nan
            op_work[fld] = arr
    data = base_build_samples(op_work, a)
    key = target or args.fig_target
    if key not in data:
        print(f"  [figs] target {key} has no samples; skipped"); return
    d = data[key]
    sel = np.ones(len(d["y"]), bool)
    if fz is not None and fz >= 3:
        sel = (d["t_index"] < fz) if args.fig_window == "pre-freeze" else \
              (d["t_index"] >= fz) if args.fig_window == "post-freeze" else sel
    r = base_fit_window(d, sel, a, rng, f"figs {key} {args.fig_window}")
    if r is None:
        print("  [figs] fit produced no model; skipped"); return
    frame = _pick_frame(data, key, sel)
    coef = r["res"]["coef_full"]; names = d["names"]
    tag = f"{key}_{args.fig_window}_hole{args.fig_hole:g}"
    u3d = _resolve_u(op, args.op_path, args.u_from or None)
    fig_reconstruction(op_work, data, key, sel, coef, names, frame,
                       out / f"reconstruction_{tag}.png", a)
    fig_feature_overlap(op_work, data, key, frame,
                        out / f"feature_overlap_{tag}.png", a)
    fig_query_points(op_work, key, frame,
                     out / f"query_points_{tag}.png", a, u3d=u3d)
    sub = rng.choice(d["Theta"].shape[0], min(50000, d["Theta"].shape[0]),
                     replace=False)
    fig_library_correlation(names, d["Theta"][sub],
                            out / f"library_correlation_{key}.png")
    print(f"  [figs] wrote v2-style figures for direct@{tag} -> {out}")


def run(args: _Args):
    rng = np.random.default_rng(args.seed)
    op = load_op(Path(args.op_path))
    op, _, _ = ensure_theta_smooth(op, args.theta_smooth_w)
    fz = detect_freeze_frame(op.get("n_dislocations"))
    mu = float(op["mu"]); k2 = max(mu, 1e-9); hpx = np.pi / k2
    routes = ["direct", "grad"] if args.w_source == "both" else [args.w_source]
    drop, y_cap, x_trim, info = compute_hull_domain(op, margin=args.margin_units * hpx)

    out = Path(args.out_root) if args.out_root else (
        _HERE / "results" / "time_sindy_riesz_v3" / Path(args.op_path).stem)
    out.mkdir(parents=True, exist_ok=True)
    print(f"loaded {Path(args.op_path).name}  mu={mu}  freeze={fz}  "
          f"pi/mu={hpx:.2f}  routes={routes}  y_cap={y_cap:.1f}")

    lines = [f"# {Path(args.op_path).name}  mu={mu}  freeze={fz}  routes={routes}",
             f"# guard cond_max={args.cond_max}  floor N>={args.min_samples} "
             f"std>={args.min_target_std:g}", ""]
    summary = {"op_path": str(args.op_path), "mu": mu, "freeze": fz,
               "routes": routes, "w_source": args.w_source, "domains": {}}

    for hu in args.hole_units:
        hole = hu * hpx
        if hole >= y_cap:
            lines.append(f"== hole {hu:g} pi/mu: SKIPPED (empties domain) ==\n")
            continue
        data = build_samples(op, args, drop, y_cap, x_trim, hole, routes)
        if data is None:
            continue
        htag = f"hole{hu:g}"
        summary["domains"][htag] = {}
        lines.append(f"== core hole {hu:g} pi/mu (|y-core|>={hole:.2f}) ==")
        print(f"  {lines[-1]}")

        windows = {"full": np.ones(len(data["x_pos"]), bool)}
        if fz is not None and fz >= 3:
            windows["pre-freeze"] = data["t_index"] < fz
            windows["post-freeze"] = data["t_index"] >= fz

        for base in ("dA_dt", "dw_dt", "dth_dt"):
            term_list = args.terms[base]
            for wl, wsel in windows.items():
                # shared finite mask across BOTH routes (+ both targets)
                shared = wsel.copy()
                for r in routes:
                    tkey = _target_key(base, r)
                    names_r = [_resolve(t, r) for t in term_list]
                    ok = np.isfinite(data["targets"][tkey])
                    for nm in names_r:
                        ok &= np.isfinite(data["cols"][nm])
                    shared &= ok
                for r in routes:
                    tkey = _target_key(base, r)
                    names_r = [_resolve(t, r) for t in term_list]
                    upmask = data["is_up"] if base == "dth_dt" else np.ones_like(wsel)
                    own = wsel & upmask
                    shr = shared & upmask
                    for dom, m in ([("own", own)] +
                                   ([("shared", shr)] if len(routes) > 1 else [])):
                        lab = f"{base}|{wl}|{r}@{dom}|{htag}"
                        res = fit_case(data, tkey, names_r, m, args, rng, lab)
                        if "skip" in res:
                            lines.append(f"  [{base}|{wl}] {r}@{dom}: {res['skip']}")
                            continue
                        pde = format_pde(base, res["coefs"],
                                         res["r2_full"], res["r2_holdout"])
                        w2 = "w^2 PRESENT" if res["has_w2"] else "no w^2"
                        drp = (f"  guard dropped {res['dropped']}"
                               if res["dropped"] else "")
                        lines.append(f"  [{base}|{wl}] {r}@{dom}  N={res['n']} "
                                     f"cond={res['cond']:.0f} "
                                     f"max|coef*std|={res['max_scoef']:.2g}  "
                                     f"[{w2}]{drp}")
                        lines.append(f"    {pde}")
                        summary["domains"][htag].setdefault(base, {}) \
                            .setdefault(wl, {})[f"{r}@{dom}"] = dict(
                            pde=pde, coefs=res["coefs"], r2_full=res["r2_full"],
                            r2_holdout=res["r2_holdout"], has_w2=res["has_w2"],
                            dropped=res["dropped"], n=res["n"])
                # route-difference line on the shared domain
                if len(routes) > 1 and base == "dw_dt":
                    d = summary["domains"][htag].get(base, {}).get(wl, {})
                    dd = d.get("direct@shared"); gg = d.get("grad@shared")
                    if dd and gg:
                        lines.append(f"    >>> route diff (shared): "
                                     f"direct w^2={dd['has_w2']} "
                                     f"grad w^2={gg['has_w2']}")
            lines.append("")
        lines.append("")

    (out / "pde_table.txt").write_text("\n".join(lines))
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    if args.figs or args.figs_all:
        targets = (["dA_dt", "dw_dt", "dth_dt"] if args.figs_all
                   else [args.fig_target])
        for tgt in targets:
            try:
                render_v2_figs(op, args, drop, y_cap, x_trim, hpx, fz, out, rng,
                               target=tgt)
            except Exception as exc:
                print(f"  [figs] {tgt} skipped: {exc}")
    print(f"  wrote {out/'pde_table.txt'}")
    print(f"done -> {out}")
    return summary


def parse_cli():
    d = _Args()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("op_path")
    p.add_argument("--w_source", choices=["direct", "grad", "both"],
                   default=d.w_source)
    p.add_argument("--hole_units", type=float, nargs="+", default=list(d.hole_units))
    p.add_argument("--no_guard", action="store_true",
                   help="disable the collinear-cancellation guard")
    p.add_argument("--guard_vif", type=float, default=d.guard_vif)
    p.add_argument("--guard_scoef", type=float, default=d.guard_scoef)
    p.add_argument("--min_samples", type=int, default=d.min_samples)
    p.add_argument("--min_target_std", type=float, default=d.min_target_std)
    p.add_argument("--terms_dw", nargs="+", default=None,
                   help="override the dw_dt term list, e.g. 1 w Lap w '(div k)^2'")
    p.add_argument("--smooth_mode", choices=["off", "list", "deriv"],
                   default=d.smooth_mode)
    p.add_argument("--smooth_width", type=float, default=d.smooth_width,
                   help="post-hoc smoother base width in units of lambda_x")
    p.add_argument("--smooth_terms", nargs="+", default=list(d.smooth_terms))
    p.add_argument("--tsmooth", type=float, default=d.tsmooth,
                   help="temporal Gaussian sigma (frames) on the grad route")
    p.add_argument("--tsmooth_on", choices=["kmag", "theta"], default=d.tsmooth_on)
    p.add_argument("--figs", action="store_true")
    p.add_argument("--figs_all", action="store_true",
                   help="render the figure set for all three targets at fig_hole")
    p.add_argument("--fig_hole", type=float, default=d.fig_hole)
    p.add_argument("--fig_target", default=d.fig_target)
    p.add_argument("--fig_window", default=d.fig_window)
    p.add_argument("--u_from", default=d.u_from)
    p.add_argument("--out_root", default=d.out_root)
    a = p.parse_args()
    terms = {k: list(v) for k, v in DEFAULT_TERMS.items()}
    if a.terms_dw:
        terms["dw_dt"] = a.terms_dw
    return _Args(op_path=a.op_path, w_source=a.w_source,
                 hole_units=tuple(a.hole_units), guard=not a.no_guard,
                 guard_vif=a.guard_vif, guard_scoef=a.guard_scoef,
                 min_samples=a.min_samples, min_target_std=a.min_target_std,
                 smooth_mode=a.smooth_mode, smooth_width=a.smooth_width,
                 smooth_terms=tuple(a.smooth_terms), tsmooth=a.tsmooth,
                 tsmooth_on=a.tsmooth_on, figs=a.figs, figs_all=a.figs_all,
                 fig_hole=a.fig_hole, fig_target=a.fig_target,
                 fig_window=a.fig_window, u_from=a.u_from, terms=terms,
                 out_root=a.out_root)


if __name__ == "__main__":
    run(parse_cli())



# python time_sindy_riesz_v4.py /Users/edwardmcdugald/patterns/pipelines/results/riesz_modes/persistence_fixed_sd0p4-pi_sapi/raw/sh_pgb_zigzag_mu0.850_T3_N125_nx512_Ny256_lower_riesz_sig0.40_ds0.300_ns128.npz --w_source both --out_root results/time_sindy_riesz_v4/baseline/85 --figs_all
# python time_sindy_riesz_v4.py /Users/edwardmcdugald/patterns/pipelines/results/riesz_modes/persistence_fixed_sd0p4-pi_sapi/raw/sh_pgb_zigzag_mu0.900_T3_N125_nx512_Ny256_lower_riesz_sig0.40_ds0.300_ns128.npz --w_source both --out_root results/time_sindy_riesz_v4/baseline/90 --figs_all

# python time_sindy_riesz_v4.py /Users/edwardmcdugald/patterns/pipelines/results/riesz_modes/persistence_fixed_sd0p4-pi_sapi/raw/sh_pgb_zigzag_mu0.850_T3_N125_nx512_Ny256_lower_riesz_sig0.40_ds0.300_ns128.npz --w_source grad --tsmooth 1.0 --tsmooth_on kmag  --out_root results/time_sindy_riesz_v4/ts_kmag_1/85 --figs_all
# python time_sindy_riesz_v4.py /Users/edwardmcdugald/patterns/pipelines/results/riesz_modes/persistence_fixed_sd0p4-pi_sapi/raw/sh_pgb_zigzag_mu0.850_T3_N125_nx512_Ny256_lower_riesz_sig0.40_ds0.300_ns128.npz --w_source grad --tsmooth 2.0 --tsmooth_on kmag  --out_root results/time_sindy_riesz_v4/ts_kmag_2/85 --figs_all
# python time_sindy_riesz_v4.py /Users/edwardmcdugald/patterns/pipelines/results/riesz_modes/persistence_fixed_sd0p4-pi_sapi/raw/sh_pgb_zigzag_mu0.850_T3_N125_nx512_Ny256_lower_riesz_sig0.40_ds0.300_ns128.npz --w_source grad --tsmooth 1.0 --tsmooth_on theta --out_root results/time_sindy_riesz_v4/ts_theta_1/85 --figs_all
# python time_sindy_riesz_v4.py /Users/edwardmcdugald/patterns/pipelines/results/riesz_modes/persistence_fixed_sd0p4-pi_sapi/raw/sh_pgb_zigzag_mu0.850_T3_N125_nx512_Ny256_lower_riesz_sig0.40_ds0.300_ns128.npz --w_source grad --tsmooth 2.0 --tsmooth_on theta --out_root results/time_sindy_riesz_v4/ts_theta_2/85 --figs_all
# python time_sindy_riesz_v4.py /Users/edwardmcdugald/patterns/pipelines/results/riesz_modes/persistence_fixed_sd0p4-pi_sapi/raw/sh_pgb_zigzag_mu0.900_T3_N125_nx512_Ny256_lower_riesz_sig0.40_ds0.300_ns128.npz --w_source grad --tsmooth 1.0 --tsmooth_on kmag  --out_root results/time_sindy_riesz_v4/ts_kmag_1/90 --figs_all
# python time_sindy_riesz_v4.py /Users/edwardmcdugald/patterns/pipelines/results/riesz_modes/persistence_fixed_sd0p4-pi_sapi/raw/sh_pgb_zigzag_mu0.900_T3_N125_nx512_Ny256_lower_riesz_sig0.40_ds0.300_ns128.npz --w_source grad --tsmooth 2.0 --tsmooth_on kmag  --out_root results/time_sindy_riesz_v4/ts_kmag_2/90 --figs_all
# python time_sindy_riesz_v4.py /Users/edwardmcdugald/patterns/pipelines/results/riesz_modes/persistence_fixed_sd0p4-pi_sapi/raw/sh_pgb_zigzag_mu0.900_T3_N125_nx512_Ny256_lower_riesz_sig0.40_ds0.300_ns128.npz --w_source grad --tsmooth 1.0 --tsmooth_on theta --out_root results/time_sindy_riesz_v4/ts_theta_1/90 --figs_all
# python time_sindy_riesz_v4.py /Users/edwardmcdugald/patterns/pipelines/results/riesz_modes/persistence_fixed_sd0p4-pi_sapi/raw/sh_pgb_zigzag_mu0.900_T3_N125_nx512_Ny256_lower_riesz_sig0.40_ds0.300_ns128.npz --w_source grad --tsmooth 2.0 --tsmooth_on theta --out_root results/time_sindy_riesz_v4/ts_theta_2/90 --figs_all

# python time_sindy_riesz_v4.py /Users/edwardmcdugald/patterns/pipelines/results/riesz_modes/persistence_fixed_sd0p4-pi_sapi/raw/sh_pgb_zigzag_mu0.850_T3_N125_nx512_Ny256_lower_riesz_sig0.40_ds0.300_ns128.npz --w_source direct --smooth_mode list --smooth_terms "div k" "(div k)^2" --smooth_width 0.25 --out_root results/time_sindy_riesz_v4/list_025/85 --figs_all
# python time_sindy_riesz_v4.py /Users/edwardmcdugald/patterns/pipelines/results/riesz_modes/persistence_fixed_sd0p4-pi_sapi/raw/sh_pgb_zigzag_mu0.850_T3_N125_nx512_Ny256_lower_riesz_sig0.40_ds0.300_ns128.npz --w_source direct --smooth_mode list --smooth_terms "div k" "(div k)^2" --smooth_width 0.5  --out_root results/time_sindy_riesz_v4/list_050/85 --figs_all
# python time_sindy_riesz_v4.py /Users/edwardmcdugald/patterns/pipelines/results/riesz_modes/persistence_fixed_sd0p4-pi_sapi/raw/sh_pgb_zigzag_mu0.900_T3_N125_nx512_Ny256_lower_riesz_sig0.40_ds0.300_ns128.npz --w_source direct --smooth_mode list --smooth_terms "div k" "(div k)^2" --smooth_width 0.25 --out_root results/time_sindy_riesz_v4/list_025/90 --figs_all
# python time_sindy_riesz_v4.py /Users/edwardmcdugald/patterns/pipelines/results/riesz_modes/persistence_fixed_sd0p4-pi_sapi/raw/sh_pgb_zigzag_mu0.900_T3_N125_nx512_Ny256_lower_riesz_sig0.40_ds0.300_ns128.npz --w_source direct --smooth_mode list --smooth_terms "div k" "(div k)^2" --smooth_width 0.5  --out_root results/time_sindy_riesz_v4/list_050/90 --figs_all

# python time_sindy_riesz_v4.py /Users/edwardmcdugald/patterns/pipelines/results/riesz_modes/persistence_fixed_sd0p4-pi_sapi/raw/sh_pgb_zigzag_mu0.850_T3_N125_nx512_Ny256_lower_riesz_sig0.40_ds0.300_ns128.npz --w_source both --smooth_mode deriv --smooth_width 0.15 --out_root results/time_sindy_riesz_v4/deriv_015/85 --figs_all
# python time_sindy_riesz_v4.py /Users/edwardmcdugald/patterns/pipelines/results/riesz_modes/persistence_fixed_sd0p4-pi_sapi/raw/sh_pgb_zigzag_mu0.850_T3_N125_nx512_Ny256_lower_riesz_sig0.40_ds0.300_ns128.npz --w_source both --smooth_mode deriv --smooth_width 0.25 --out_root results/time_sindy_riesz_v4/deriv_025/85 --figs_all
# python time_sindy_riesz_v4.py /Users/edwardmcdugald/patterns/pipelines/results/riesz_modes/persistence_fixed_sd0p4-pi_sapi/raw/sh_pgb_zigzag_mu0.850_T3_N125_nx512_Ny256_lower_riesz_sig0.40_ds0.300_ns128.npz --w_source both --smooth_mode deriv --smooth_width 0.40 --out_root results/time_sindy_riesz_v4/deriv_040/85 --figs_all
# python time_sindy_riesz_v4.py /Users/edwardmcdugald/patterns/pipelines/results/riesz_modes/persistence_fixed_sd0p4-pi_sapi/raw/sh_pgb_zigzag_mu0.900_T3_N125_nx512_Ny256_lower_riesz_sig0.40_ds0.300_ns128.npz --w_source both --smooth_mode deriv --smooth_width 0.15 --out_root results/time_sindy_riesz_v4/deriv_015/90 --figs_all
# python time_sindy_riesz_v4.py /Users/edwardmcdugald/patterns/pipelines/results/riesz_modes/persistence_fixed_sd0p4-pi_sapi/raw/sh_pgb_zigzag_mu0.900_T3_N125_nx512_Ny256_lower_riesz_sig0.40_ds0.300_ns128.npz --w_source both --smooth_mode deriv --smooth_width 0.25 --out_root results/time_sindy_riesz_v4/deriv_025/90 --figs_all
# python time_sindy_riesz_v4.py /Users/edwardmcdugald/patterns/pipelines/results/riesz_modes/persistence_fixed_sd0p4-pi_sapi/raw/sh_pgb_zigzag_mu0.900_T3_N125_nx512_Ny256_lower_riesz_sig0.40_ds0.300_ns128.npz --w_source both --smooth_mode deriv --smooth_width 0.40 --out_root results/time_sindy_riesz_v4/deriv_040/90 --figs_all