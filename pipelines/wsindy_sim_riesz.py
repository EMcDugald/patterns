# pipelines/wsindy_sim_riesz.py
"""
Weak-form SINDy on EXTRACTED Swift-Hohenberg order-parameter data (the npz
output of run_riesz_zigzag_modes). Companion to wsindy_knee_validation, which
settled the method on the analytic knee:

  - outer test-function radius 1.0 lambda_x (radius is not a sensitive knob),
  - inner eta = 0 (superseded by the measurement-consistent library),
  - sm_sigma = sigma_avg = pi: nonlinear-in-derivative library columns are
    smoothed by the extraction's own measurement kernel AFTER the
    nonlinearity, because the extracted w is G_sigma_avg * w_true (verified
    to 2.3 percent on the knee) and smoothing commutes with d_t and Lap but
    not with squares.

Targets (each fit weak AND pointwise, for the comparison):
  w      = kmag^2 - 1            library: 1, w, w^2, Lap w, |grad w|,
                                          div k, (div k)^2, |grad k|^2
  A      = amplitude             library: 1, A, A^2, A^3, Lap A, w, A*w
                                 KNOWN-ANSWER GATE: the pointwise amplitude
                                 cubic dA/dt = c1 A^3 + c2 A + c0 at
                                 R^2 >= ~0.9 is the check that the weak
                                 assembly is healthy on real data.
  theta  = reconstructed phase   library: 1, div k, w, A   (theta is NaN
                                 outside the reconstructed region; rows are
                                 restricted to kernel supports that are
                                 fully finite)
  c2,s2  = doubled director pair library: 1, c2, s2, w, div k, A
           c2 = (k1^2-k2^2)/|k|^2, s2 = 2 k1 k2/|k|^2 (flip-safe, no wrap)

No claim of a known law for w / theta / director on SH data: results are
read by holdout R^2, support stability, and interpretability, exactly as in
the pointwise work. Watch for an inflating (div k)^2 coefficient: on the
knee that was the diagnostic tell of noise rectifying through the square.

Figures (--figs):
  fields_overview.png       A, w, div k at first / middle / last frame
  library_terms_w.png       every w-library column at the three frames
  fit_vs_target_<t>.png     weak prediction vs weak target at the middle
                            frame on the center grid, plus residual map and
                            a scatter with R^2
  domain_figure.png         query centers over the A field, one kernel
                            footprint to scale, rim margins and band lines

Frozen layer untouched. Weak-form primitives are imported from
wsindy_knee_validation (kernel_1d, corr1d, corr_sep, meas_smooth);
selection reuses fit_target from self_dual_sindy with the same
pair-augmented Pareto rule validated on the knee.

v1.3 changes:
  - EXACT IDENTITY FIX: (div k)^2 + (curl k)^2 = |grad k|^2 + 2 J holds
    pointwise for any 2d vector field, so the four quadratic gradient
    columns are exactly linearly dependent and the regression gains a null
    direction (the canceling +/-0.4 coefficient patterns seen in v1.2 runs).
    Default w library now drops |grad k|^2; a warning is logged if any
    target's library contains all four.
  - --drop_terms: comma list removed from every target library (e.g.
    --drop_terms "curl k,(curl k)^2,J" to suppress the new fields per run).
  - Every fit now logs its library explicitly in the report.
  - Nucleation summary from charges_json: per-frame defect count, first
    charged frame t_nuc, and a suggested --t0/--t1 split, logged at start.
v1.2 changes:
  - New library fields: "curl k" (= k2_x - k1_y, assembled FULLY WEAK by IBP;
    a true phase gradient is curl-free, so this column lives at defects),
    "(curl k)^2" and "J" (= k1_x k2_y - k1_y k2_x, the oriented Jacobian /
    topological current from run_riesz_metrics_zigzag), both
    measurement-smoothed. Default w library gains all three; override with
    --terms_w as usual.
  - libcorr_<target>.png: correlation heatmap of the weak library columns
    with the weak target prepended, the collinearity diagnostic.
  - --t0/--t1: fit a frame window only (pre- vs post-nucleation splits);
    stamped into the run folder.
  - --sm_amp: extend measurement smoothing to the amplitude nonlinearities
    (A^3, A*w), the consistency check for the Lap A sign.
v1.1 changes (motivated by the first mu=0.85 fine-cadence campaign):
  - A library default drops A^2 (the {A, A^2, A^3} collinearity driver behind
    +/-20..77 canceling coefficients); the cubic parameterization is restored.
  - --min_contrib: post-selection prune of terms whose unit-free contribution
    |c_j| * std(col_j) / std(target) falls below the floor, iterated to a
    fixed point. This is the scale-safe version of "drop tiny coefficients":
    an absolute |c| cutoff is unsafe because coefficients carry units
    (the knee's true (div k)^2 coefficient is -0.10 on columns reaching
    hundreds near cores).
  - --ridge: optional Tikhonov penalty on the FINAL refit (normalized
    columns), a stability diagnostic against collinear blowup; selection is
    untouched and quoted coefficients should come from ridge=0 runs.
  - --terms_w/_A/_theta/_c2/_s2: per-target library overrides (comma lists).
  - --domain_strict: band/far keep only centers whose ENTIRE y-support lies
    inside/outside band_y, so the domain split is hard, not blended by the
    kernel footprint (needs band_y > radius_y * lambda_x for band rows).

Output (relative): results/wsindy_sim_riesz/<npz stem>/
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

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from wsindy_knee_validation import (kernel_1d, corr1d, corr_sep,      # noqa: E402
                                    meas_smooth)
from self_dual_sindy import (fit_target, r_squared,                   # noqa: E402
                             fig_library_correlation)


# --------------------------------------------------------------------------
# selection: fit_target + pair-augmented Pareto (as validated on the knee)
# --------------------------------------------------------------------------

class _FitArgs:
    thresholds = tuple(np.geomspace(1e-4, 1.0, 25))
    rel_tol_pareto = 0.01
    n_boot = 100
    boot_frac = 0.5

    def __init__(self, rel_tol=None):
        if rel_tol is not None:
            self.rel_tol_pareto = float(rel_tol)


SKIPPED = dict(n=0, coefs={}, support=[], r2_full=float("nan"),
               r2_holdout=float("nan"), skipped=True)


def _refit(Theta, bv, supp, ridge=0.0):
    idx = np.where(supp)[0]
    coef = np.zeros(Theta.shape[1])
    if len(idx) == 0:
        return coef
    Th = Theta[:, idx]
    if ridge > 0:
        norms = np.linalg.norm(Th, axis=0)
        norms[norms == 0] = 1.0
        Tn = Th / norms
        A_ = Tn.T @ Tn + ridge * np.eye(len(idx))
        coef[idx] = np.linalg.solve(A_, Tn.T @ bv) / norms
    else:
        coef[idx], *_ = np.linalg.lstsq(Th, bv, rcond=None)
    return coef


def fit_pareto(names, Theta, bv, xs, rng, rel_tol=0.01,
               max_samples=60000, min_samples=200,
               min_contrib=0.0, ridge=0.0):
    if Theta is None or len(bv) < min_samples:
        return dict(SKIPPED)
    if len(bv) > max_samples:
        idx = rng.choice(len(bv), max_samples, replace=False)
        Theta, bv, xs = Theta[idx], bv[idx], xs[idx]
    tr = xs < np.median(xs)
    if tr.sum() < min_samples // 2 or (~tr).sum() < min_samples // 2:
        return dict(SKIPPED, reason="degenerate train/test split")
    try:
        res = fit_target(names, Theta, bv, _FitArgs(rel_tol), rng, tr)
    except (IndexError, np.linalg.LinAlgError) as e:
        return dict(SKIPPED,
                    reason=f"selection pool empty / degenerate target ({e})")
    sweep = [sw for sw in res["sweep"] if np.isfinite(sw["r2_test"])]
    if not sweep:
        return dict(SKIPPED, reason="no finite holdout scores")
    p = Theta.shape[1]
    for i in range(p):
        for j in range(i + 1, p):
            supp = np.zeros(p, dtype=bool)
            supp[[i, j]] = True
            coef = np.zeros(p)
            sol, *_ = np.linalg.lstsq(Theta[tr][:, [i, j]], bv[tr],
                                      rcond=None)
            coef[[i, j]] = sol
            sweep.append(dict(support=supp, coef=coef, n_terms=2,
                              r2_train=r_squared(bv[tr], Theta[tr] @ coef),
                              r2_test=r_squared(bv[~tr], Theta[~tr] @ coef)))
    best = max(s["r2_test"] for s in sweep)
    adm = [s for s in sweep if s["n_terms"] > 0
           and s["r2_test"] >= best - rel_tol * abs(best)]
    adm.sort(key=lambda s: (s["n_terms"], -s["r2_test"]))
    chosen = adm[0]
    supp = chosen["support"].copy()
    coef_full = _refit(Theta, bv, supp, ridge)
    # contribution floor: prune terms explaining < min_contrib of the target
    if min_contrib > 0:
        sb = max(float(np.std(bv)), 1e-30)
        for _ in range(p):
            contrib = np.abs(coef_full) * np.std(Theta, axis=0) / sb
            weak_terms = supp & (contrib < min_contrib)
            if not weak_terms.any() or supp.sum() <= 1:
                break
            j = np.where(weak_terms)[0][np.argmin(contrib[weak_terms])]
            supp[j] = False
            coef_full = _refit(Theta, bv, supp, ridge)
    coefs = {names[i]: float(coef_full[i]) for i in np.where(supp)[0]}
    return dict(n=int(len(bv)), coefs=coefs, support=sorted(coefs),
                coef_vec=[float(c) for c in coef_full],
                r2_full=float(r_squared(bv, Theta @ coef_full)),
                r2_holdout=float(chosen["r2_test"]))


def fmt(name, r):
    if r.get("skipped"):
        why = r.get("reason", "too few valid rows")
        return f"  {name:10s} SKIPPED ({why})"
    terms = " ".join(f"{c:+.4g}*{nm}" for nm, c in
                     sorted(r["coefs"].items(), key=lambda kv: -abs(kv[1])))
    return (f"  {name:10s} R2(hold)={r['r2_holdout']:+.4f} n={r['n']}\n"
            f"             d/dt = {terms}")


# --------------------------------------------------------------------------
# data loading and derived fields
# --------------------------------------------------------------------------

def load_npz(path, A_source):
    z = np.load(path, allow_pickle=True)
    d = dict(x=np.asarray(z["x"], float), y=np.asarray(z["y"], float),
             tt=np.asarray(z["tt"], float), mu=float(z["mu"]))
    k1 = np.asarray(z["k1_direct"], float)
    k2 = np.asarray(z["k2_direct"], float)
    d["k1"], d["k2"] = k1, k2
    d["w"] = k1 ** 2 + k2 ** 2 - 1.0
    d["A"] = np.asarray(z[A_source], float)
    d["theta"] = np.asarray(z["theta_smooth"], float) \
        if "theta_smooth" in z.files else np.asarray(z["theta"], float)
    n2 = np.maximum(k1 ** 2 + k2 ** 2, 1e-12)
    d["c2"] = (k1 ** 2 - k2 ** 2) / n2
    d["s2"] = 2.0 * k1 * k2 / n2
    meta = json.loads(str(z["sh_meta_json"])) if "sh_meta_json" in z.files \
        else {}
    d["_meta"] = meta
    if "charges_json" in z.files:
        ch = json.loads(str(z["charges_json"]))
        d["defect_counts"] = np.array([len(f) for f in ch])
    d["lambda_x"] = float(meta.get("lambda_x",
                                   2 * np.pi / np.sqrt(1 - d["mu"] ** 2)))
    return d


def grad2(F, dx, dy):
    return np.gradient(F, dx, axis=1), np.gradient(F, dy, axis=0)


def pointwise_fields(d, args):
    """All pointwise library fields, [Ny, Nx, Nt]. Nonlinear-in-derivative
    terms get the measurement smoothing (sm_sigma) after the nonlinearity."""
    x, y = d["x"], d["y"]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    Lx, Ly = x[-1] - x[0], y[-1] - y[0]
    k1, k2, w, A = d["k1"], d["k2"], d["w"], d["A"]
    sm = lambda F: meas_smooth(F, args.sm_sigma, Lx, Ly)

    k1x, k1y = grad2(k1, dx, dy)
    k2x, k2y = grad2(k2, dx, dy)
    wx, wy = grad2(w, dx, dy)
    Ax, Ay = grad2(A, dx, dy)
    divk = k1x + k2y
    F = {}
    F["1"] = np.ones_like(w)
    F["w"] = w
    F["w^2"] = w ** 2
    F["Lap w"] = np.gradient(wx, dx, axis=1) + np.gradient(wy, dy, axis=0)
    F["|grad w|"] = sm(np.hypot(wx, wy))
    F["div k"] = divk
    F["(div k)^2"] = sm(divk ** 2)
    F["|grad k|^2"] = sm(k1x ** 2 + k1y ** 2 + k2x ** 2 + k2y ** 2)
    F["A"] = A
    F["A^2"] = A ** 2
    F["A^3"] = A ** 3
    F["Lap A"] = np.gradient(Ax, dx, axis=1) + np.gradient(Ay, dy, axis=0)
    F["A*w"] = A * w
    F["c2"] = d["c2"]
    F["s2"] = d["s2"]
    curl = k2x - k1y
    F["curl k"] = curl
    F["(curl k)^2"] = sm(curl ** 2)
    F["J"] = sm(k1x * k2y - k1y * k2x)
    if getattr(args, "sm_amp", False):
        F["A^3"] = sm(A ** 3)
        F["A*w"] = sm(A * w)
    return F


TARGETS = {
    "w": ["1", "w", "w^2", "Lap w", "|grad w|", "div k", "(div k)^2",
          "curl k", "(curl k)^2", "J"],
    "A": ["1", "A", "A^3", "Lap A", "w", "A*w"],
    "theta": ["1", "div k", "w", "A"],
    "c2": ["1", "c2", "s2", "w", "div k", "A"],
    "s2": ["1", "c2", "s2", "w", "div k", "A"],
}

# weak-integrable structure of the w library: these columns are assembled by
# IBP on the raw fields instead of integrating the pointwise stack
WEAK_IBP = {"Lap w", "div k", "curl k"}


# --------------------------------------------------------------------------
# weak assembly on the sim grid (NaN-aware)
# --------------------------------------------------------------------------

def kernels(d, args):
    x, y, tt = d["x"], d["y"], d["tt"]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    dts = np.diff(tt)
    if not np.allclose(dts, dts[0], rtol=1e-5):
        raise ValueError("nonuniform frame spacing")
    dt = float(dts[0])
    a = args.radius * d["lambda_x"]
    ay = args.radius_y * d["lambda_x"]
    fx = kernel_1d(a, dx, args.kernel, args.poly_p)
    fy = kernel_1d(ay, dy, args.kernel, args.poly_p)
    ht = kernel_1d(args.t_half * dt, dt, args.kernel, args.poly_pt)
    return fx, fy, ht, dx, dy, dt


def centers(d, args, valid_mask, mx, my, mt):
    """Query-center indices: strides, kernel margins, rim, band selection,
    and full-support validity (every point under the kernel finite)."""
    x, y = d["x"], d["y"]
    ny, nx, nt = valid_mask.shape
    dx = float(x[1] - x[0])
    rim = int(np.ceil(args.rim_lam * d["lambda_x"] / dx))
    ix = np.arange(mx + rim, nx - mx - rim, args.stride_xy)
    iy = np.arange(my + rim, ny - my - rim, args.stride_xy)
    it = np.arange(mt, nt - mt, args.stride_t)
    ay = (args.radius_y * d["lambda_x"]) if getattr(args, "domain_strict",
                                                    False) else 0.0
    if args.domain == "band":
        iy = iy[np.abs(y[iy]) <= args.band_y - ay]
    elif args.domain == "far":
        iy = iy[np.abs(y[iy]) > args.band_y + ay]
    if min(len(ix), len(iy), len(it)) == 0:
        return None
    IY, IX, IT = np.meshgrid(iy, ix, it, indexing="ij")
    ok = valid_mask[IY, IX, IT]
    return IY, IX, IT, ok, (iy, ix, it)


def weak_target_system(tname, d, F, args):
    """Assemble weak (Theta, b, xs, centers) for one target."""
    fx, fy, ht, dx, dy, dt = kernels(d, args)
    (fx0, fx1, fx2, mx) = fx
    (fy0, fy1, fy2, my) = fy
    (ht0, ht1, _, mt) = ht

    tgt = d[tname] if tname in d else F[tname]
    fin = np.isfinite(tgt)
    tgt0 = np.where(fin, tgt, 0.0)

    # full-support validity: kernel-weighted finite fraction must be ~1
    frac = corr_sep(fin.astype(float), fx0, fy0, ht0, dx, dy, dt)
    valid = frac > 0.999

    b = -corr_sep(tgt0, fx0, fy0, ht1, dx, dy, dt)
    cols = {}
    for nm in TARGETS[tname]:
        if nm == "Lap w" and "Lap w" in WEAK_IBP and tname == "w":
            cols[nm] = (corr_sep(d["w"], fx2, fy0, ht0, dx, dy, dt)
                        + corr_sep(d["w"], fx0, fy2, ht0, dx, dy, dt))
        elif nm == "div k" and "div k" in WEAK_IBP:
            cols[nm] = -(corr_sep(d["k1"], fx1, fy0, ht0, dx, dy, dt)
                         + corr_sep(d["k2"], fx0, fy1, ht0, dx, dy, dt))
        elif nm == "curl k" and "curl k" in WEAK_IBP:
            # Int (curl k) phi = -Int k2 phi_x + Int k1 phi_y
            cols[nm] = (-corr_sep(d["k2"], fx1, fy0, ht0, dx, dy, dt)
                        + corr_sep(d["k1"], fx0, fy1, ht0, dx, dy, dt))
        elif nm == "1":
            cols[nm] = np.ones_like(b)
        else:
            f = F[nm]
            ffin = np.isfinite(f)
            valid &= corr_sep(ffin.astype(float), fx0, fy0, ht0,
                              dx, dy, dt) > 0.999
            cols[nm] = corr_sep(np.where(ffin, f, 0.0), fx0, fy0, ht0,
                                dx, dy, dt)

    C = centers(d, args, np.broadcast_to(valid, b.shape), mx, my, mt)
    if C is None:
        return None
    IY, IX, IT, ok, axes = C
    Theta = np.column_stack([cols[nm][IY, IX, IT].ravel()
                             for nm in TARGETS[tname]])
    bv = b[IY, IX, IT].ravel()
    xs = d["x"][IX.ravel()]
    keep = ok.ravel() & np.all(np.isfinite(Theta), axis=1) & np.isfinite(bv)
    return dict(Theta=Theta[keep], b=bv[keep], xs=xs[keep], keep=keep,
                IY=IY, IX=IX, IT=IT, axes=axes, cols=cols, btab=b)


def pointwise_target_system(tname, d, F, args):
    tgt = d[tname] if tname in d else F[tname]
    tt = d["tt"]
    dtgt = np.gradient(tgt, tt, axis=2)
    x, y = d["x"], d["y"]
    dx = float(x[1] - x[0])
    ny, nx, nt = tgt.shape
    rim = int(np.ceil(args.rim_lam * d["lambda_x"] / dx)) + 4
    ix = np.arange(rim, nx - rim, args.stride_xy)
    iy = np.arange(rim, ny - rim, args.stride_xy)
    it = np.arange(1, nt - 1, args.stride_t)
    if args.domain == "band":
        iy = iy[np.abs(y[iy]) <= args.band_y]
    elif args.domain == "far":
        iy = iy[np.abs(y[iy]) > args.band_y]
    IY, IX, IT = np.meshgrid(iy, ix, it, indexing="ij")
    Theta = np.column_stack([F[nm][IY, IX, IT].ravel() if nm != "1"
                             else np.ones(IY.size)
                             for nm in TARGETS[tname]])
    bv = dtgt[IY, IX, IT].ravel()
    xs = x[IX.ravel()]
    keep = np.all(np.isfinite(Theta), axis=1) & np.isfinite(bv)
    return dict(Theta=Theta[keep], b=bv[keep], xs=xs[keep])


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def _tri_indices(nt):
    return [0, nt // 2, nt - 1]


def _imshow(ax, F, x, y, title, cmap="RdBu_r", sym=True):
    v = np.nanpercentile(np.abs(F), 99) if sym else None
    im = ax.imshow(F, origin="lower", extent=[x[0], x[-1], y[0], y[-1]],
                   cmap=cmap, aspect="auto",
                   vmin=-v if sym else None, vmax=v if sym else None)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.04)


def fig_fields(d, F, out):
    x, y, tt = d["x"], d["y"], d["tt"]
    idx = _tri_indices(len(tt))
    rows = [("A", d["A"], "viridis", False), ("w", d["w"], "RdBu_r", True),
            ("div k", F["div k"], "RdBu_r", True)]
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 3.2 * len(rows)))
    for i, (nm, Fld, cmap, sym) in enumerate(rows):
        for j, t in enumerate(idx):
            _imshow(axes[i, j], Fld[:, :, t], x, y,
                    f"{nm}, t={tt[t]:.3f}", cmap, sym)
    fig.tight_layout()
    fig.savefig(out / "fields_overview.png", dpi=150)
    plt.close(fig)


def fig_library(d, F, out, tname="w"):
    x, y, tt = d["x"], d["y"], d["tt"]
    idx = _tri_indices(len(tt))
    names = [nm for nm in TARGETS[tname] if nm != "1"]
    fig, axes = plt.subplots(len(names), 3,
                             figsize=(13, 2.6 * len(names)))
    for i, nm in enumerate(names):
        for j, t in enumerate(idx):
            _imshow(axes[i, j], F[nm][:, :, t], x, y,
                    f"{nm}, t={tt[t]:.3f}")
    fig.suptitle(f"library columns for d{tname}/dt (pointwise fields; "
                 "sm_sigma applied where marked in the runner)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"library_terms_{tname}.png", dpi=150)
    plt.close(fig)


def fig_fit(tname, d, sysw, r, out):
    """Weak target vs fitted combination at the middle center-time, on the
    center grid, plus residual map and scatter."""
    if r.get("skipped") or sysw is None:
        return
    iy, ix, it = sysw["axes"]
    tmid = it[len(it) // 2]
    IYg, IXg = np.meshgrid(iy, ix, indexing="ij")
    coef = np.asarray(r["coef_vec"])
    pred_tab = sum(c * sysw["cols"][nm]
                   for nm, c in zip(TARGETS[tname], coef) if c != 0.0)
    tgt = sysw["btab"][IYg, IXg, tmid]
    prd = pred_tab[IYg, IXg, tmid]
    res = tgt - prd
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    xg, yg = d["x"][ix], d["y"][iy]
    for ax, Fld, ttl in ((axes[0], tgt, f"weak d{tname}/dt (target)"),
                         (axes[1], prd, "fitted combination"),
                         (axes[2], res, "residual")):
        _imshow(ax, Fld, xg, yg, ttl)
    a3 = axes[3]
    a3.plot(sysw["b"], sysw["Theta"] @ coef, ".", ms=1, alpha=0.3)
    lo, hi = np.percentile(sysw["b"], [1, 99])
    a3.plot([lo, hi], [lo, hi], "k--", lw=1)
    a3.set_xlabel("weak target")
    a3.set_ylabel("prediction")
    a3.set_title(f"all rows, R2(hold)={r['r2_holdout']:.3f}", fontsize=9)
    fig.suptitle(f"d{tname}/dt: " + " ".join(
        f"{c:+.3g}*{nm}" for nm, c in sorted(r["coefs"].items(),
                                             key=lambda kv: -abs(kv[1]))),
        fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"fit_vs_target_{tname}.png", dpi=150)
    plt.close(fig)


def fig_domain(d, sysw, args, out):
    if sysw is None:
        return
    x, y, tt = d["x"], d["y"], d["tt"]
    iy, ix, it = sysw["axes"]
    tmid = it[len(it) // 2]
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(d["A"][:, :, tmid], origin="lower",
                   extent=[x[0], x[-1], y[0], y[-1]], cmap="viridis",
                   aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.04, label="A")
    IYg, IXg = np.meshgrid(iy, ix, indexing="ij")
    ok = sysw["keep"].reshape(len(iy), len(ix), len(it))[:, :, len(it) // 2]
    ax.plot(x[IXg][ok], y[IYg][ok], "w.", ms=2, alpha=0.7,
            label="query centers (valid)")
    ax.plot(x[IXg][~ok], y[IYg][~ok], "r.", ms=2, alpha=0.5,
            label="dropped (support hits NaN)")
    a = args.radius * d["lambda_x"]
    ay = args.radius_y * d["lambda_x"]
    cx, cy = x[ix[len(ix) // 3]], y[iy[len(iy) // 2]]
    ax.add_patch(plt.Rectangle((cx - a, cy - ay), 2 * a, 2 * ay,
                               fill=False, ec="cyan", lw=2,
                               label="one kernel footprint"))
    if args.domain in ("band", "far"):
        for s in (-1, 1):
            ax.axhline(s * args.band_y, color="orange", ls="--", lw=1)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"fitting domain: {args.domain}, radius={args.radius} "
                 f"lam_x, rim={args.rim_lam} lam_x, frame t={tt[tmid]:.3f}")
    fig.tight_layout()
    fig.savefig(out / "domain_figure.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def run_stamp(args):
    """Config-stamped subfolder so distinct runs on the same npz never
    overwrite: <domain>_rad<r>x<ry>_th<t_half>_sm<sigma|off>[_<tag>]."""
    sm = "off" if args.sm_sigma <= 0 else f"{args.sm_sigma:.2f}"
    dom = args.domain + ("S" if getattr(args, "domain_strict", False)
                         else "")
    s = (f"{dom}_rad{args.radius:g}x{args.radius_y:g}"
         f"_th{args.t_half}_sm{sm}")
    if args.min_contrib > 0:
        s += f"_mc{args.min_contrib:g}"
    if args.ridge > 0:
        s += f"_rg{args.ridge:g}"
    t0 = getattr(args, "t0", 0) or 0
    t1 = getattr(args, "t1", 0)
    if t0 or t1:
        s += f"_t{t0}-{t1 if t1 else 'end'}"
    if getattr(args, "sm_amp", False):
        s += "_smA"
    if getattr(args, "tag", ""):
        s += f"_{args.tag}"
    return s


def main(args):
    npz = Path(args.npz)
    out = Path(args.out) / npz.stem / run_stamp(args)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s):
        print(s)
        lines.append(str(s))

    rng = np.random.default_rng(args.seed)
    d = load_npz(npz, args.A_source)
    t0 = getattr(args, "t0", 0) or 0
    t1 = getattr(args, "t1", 0) or d["tt"].shape[0]
    if (t0, t1) != (0, d["tt"].shape[0]):
        for key in ("k1", "k2", "w", "A", "theta", "c2", "s2"):
            d[key] = d[key][:, :, t0:t1]
        d["tt"] = d["tt"][t0:t1]
    if "defect_counts" in d:
        dc = d["defect_counts"]
        nz = np.nonzero(dc)[0]
        if len(nz):
            t_nuc = int(nz[0])
            log(f"nucleation summary: first charged frame t_nuc={t_nuc} "
                f"(of {len(dc)}), peak count {int(dc.max())}; suggested "
                f"pre/post split: --t1 {t_nuc} / --t0 {t_nuc}")
        else:
            log("nucleation summary: no charged frames in charges_json")
    log(f"{npz.name}: grid {d['w'].shape}, mu={d['mu']}, "
        f"lambda_x={d['lambda_x']:.3f}, dt={d['tt'][1]-d['tt'][0]:.4g}, "
        f"A_source={args.A_source}, sm_sigma={args.sm_sigma:.4f}, "
        f"domain={args.domain}")
    F = pointwise_fields(d, args)
    drop = [t.strip() for t in getattr(args, "drop_terms", "").split(",")
            if t.strip()] if getattr(args, "drop_terms", "") else []
    QUAD4 = {"(div k)^2", "(curl k)^2", "|grad k|^2", "J"}
    for tname in list(TARGETS):
        ov = getattr(args, f"terms_{tname}", None)
        if ov:
            TARGETS[tname] = [t.strip() for t in ov.split(",")]
        if drop:
            TARGETS[tname] = [t for t in TARGETS[tname] if t not in drop]
        if QUAD4 <= set(TARGETS[tname]):
            log(f"WARNING {tname}: library contains all four of {QUAD4}; "
                "these satisfy (div k)^2+(curl k)^2 = |grad k|^2+2J exactly, "
                "so coefficients on them are not identifiable. Drop one.")

    report = {"npz": str(npz), "args": {k: getattr(args, k) for k in dir(args)
              if not k.startswith("_")
              and not callable(getattr(args, k))}}
    targets = args.targets.split(",") if isinstance(args.targets, str) \
        else list(args.targets)

    sysw_w = None
    for tname in targets:
        log("=" * 72)
        sysw = weak_target_system(tname, d, F, args)
        rw = fit_pareto(TARGETS[tname],
                        sysw["Theta"] if sysw else None,
                        sysw["b"] if sysw else np.zeros(0),
                        sysw["xs"] if sysw else np.zeros(0),
                        rng, args.rel_tol, min_contrib=args.min_contrib,
                        ridge=args.ridge) if sysw else dict(SKIPPED)
        sysp = pointwise_target_system(tname, d, F, args)
        rp = fit_pareto(TARGETS[tname], sysp["Theta"], sysp["b"],
                        sysp["xs"], rng, args.rel_tol,
                        min_contrib=args.min_contrib, ridge=args.ridge)
        log(f"  {tname} library: {TARGETS[tname]}")
        log(fmt(f"{tname} weak", rw))
        log(fmt(f"{tname} ptws", rp))
        report[tname] = dict(weak=rw, pointwise=rp)
        if args.figs:
            fig_fit(tname, d, sysw, rw, out)
            if sysw is not None:
                fig_library_correlation(
                    [f"d{tname}/dt"] + TARGETS[tname],
                    np.column_stack([sysw["b"], sysw["Theta"]]),
                    out / f"libcorr_{tname}.png")
        if tname == "w":
            sysw_w = sysw

    if args.figs:
        fig_fields(d, F, out)
        fig_library(d, F, out, "w")
        fig_domain(d, sysw_w if sysw_w is not None else
                   weak_target_system("A", d, F, args), args, out)
        log(f"figures -> {out}")

    (out / "report.json").write_text(json.dumps(report, indent=2,
                                                default=str))
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    log(f"report -> {out / 'report.json'}")


class _Args:
    npz = "data/sh_pgb_zigzag/riesz_v8_aware/example.npz"
    A_source = "A_mono"       # 'A' (ratio amplitude) | 'A_mono' (monogenic)
    # weak form (settled on the knee)
    kernel = "poly"
    poly_p = 6
    poly_pt = 4
    radius = 1.0              # outer x half-width (lambda_x)
    radius_y = 1.0            # outer y half-width (lambda_x)
    t_half = 3                # temporal half-width in frames
    sm_sigma = float(np.pi)   # measurement kernel = extraction sigma_avg
    stride_xy = 3
    stride_t = 1
    rim_lam = 0.5
    # domain
    domain = "full"           # full | band | far
    band_y = 3.0              # |y| threshold for band/far (physical units)
    domain_strict = False     # hard domain split (see docstring)
    # selection
    rel_tol = 0.01
    min_contrib = 0.0         # unit-free contribution floor (try 0.05)
    ridge = 0.0               # final-refit ridge (stability diagnostic only)
    terms_w = terms_A = terms_theta = terms_c2 = terms_s2 = None
    t0 = 0                    # first frame of fit window (0 = start)
    t1 = 0                    # one past last frame (0 = end)
    sm_amp = False            # measurement-smooth A^3 and A*w columns
    drop_terms = ""           # comma list removed from EVERY target library
    seed = 0
    # run control
    targets = "w,A,theta,c2,s2"
    figs = True
    tag = ""                  # optional extra suffix for the run folder
    out = "results/wsindy_sim_riesz_v2"


def _cli():
    d = _Args
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--npz", default=d.npz)
    p.add_argument("--A_source", choices=("A", "A_mono"), default=d.A_source)
    p.add_argument("--kernel", choices=("poly", "gauss"), default=d.kernel)
    p.add_argument("--poly_p", type=int, default=d.poly_p)
    p.add_argument("--poly_pt", type=int, default=d.poly_pt)
    p.add_argument("--radius", type=float, default=d.radius)
    p.add_argument("--radius_y", type=float, default=d.radius_y)
    p.add_argument("--t_half", type=int, default=d.t_half)
    p.add_argument("--sm_sigma", type=float, default=d.sm_sigma)
    p.add_argument("--stride_xy", type=int, default=d.stride_xy)
    p.add_argument("--stride_t", type=int, default=d.stride_t)
    p.add_argument("--rim_lam", type=float, default=d.rim_lam)
    p.add_argument("--domain", choices=("full", "band", "far"),
                   default=d.domain)
    p.add_argument("--band_y", type=float, default=d.band_y)
    p.add_argument("--domain_strict", action="store_true",
                   default=d.domain_strict)
    p.add_argument("--rel_tol", type=float, default=d.rel_tol)
    p.add_argument("--min_contrib", type=float, default=d.min_contrib)
    p.add_argument("--ridge", type=float, default=d.ridge)
    for t in ("w", "A", "theta", "c2", "s2"):
        p.add_argument(f"--terms_{t}", default=None)
    p.add_argument("--t0", type=int, default=d.t0)
    p.add_argument("--t1", type=int, default=d.t1)
    p.add_argument("--sm_amp", action="store_true", default=d.sm_amp)
    p.add_argument("--drop_terms", default=d.drop_terms)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--targets", default=d.targets)
    p.add_argument("--figs", dest="figs", action="store_true",
                   default=d.figs)
    p.add_argument("--no_figs", dest="figs", action="store_false")
    p.add_argument("--tag", default=d.tag)
    p.add_argument("--out", default=d.out)
    return p.parse_args()


if __name__ == "__main__":
    main(_cli() if len(sys.argv) > 1 else _Args)