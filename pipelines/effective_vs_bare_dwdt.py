# pipelines/effective_vs_bare_dwdt.py
"""
"Effective vs bare" control for the time-SINDy w-law.

Question (HANDOFF open item 3): the real dw/dt law is fit on EXTRACTED w. Is its
term structure a property of the dynamics, or an artifact of extraction? Evolve a
synthetic field under a KNOWN phase dynamics (no simulation, no noise), measure w
three ways, run the same STLSQ->Pareto->bootstrap selection, and compare the
SELECTED TERMS across routes. If the terms match and only coefficients move,
extraction renormalizes but preserves the model (valid effective dynamics). If a
route invents or drops terms, that is extraction breaking the law.

Bare dynamics
-------------
Linear phase diffusion on an exact isolated knee:

    theta(x, y, t) = k1 x + phi(y, t),   phi(y, 0) = log cosh(k2 y)
    d_t theta = nu * Lap theta   (=> d_t phi = nu phi_yy, a 1-D heat equation)

For this y-only field |grad k|^2 = (div k)^2, so the EXACT w-law is closed:

    d_t w = nu * Lap w  -  2 nu * (div k)^2 ,   w = |k|^2 - 1

with NO w^2 and NO w term. That is the falsifiable target: the bare (exact-grad)
route must recover {Lap w, (div k)^2} with coefficients (nu, -2 nu); whether the
extraction routes keep that support or promote w^2 / w / div k is the measurement.

Three k routes (per the session choice: run all three)
------------------------------------------------------
  exact        k = grad(theta)                          (bare, no smoothing)
  k_from_theta t1_common.k_from_theta(theta_true, ...)  (coded gradient estimator,
                                                          FFT nan_smooth + erosion)
  riesz        riesz_k_field(cos theta, ...)            (production director route)

Library (the real dw_dt library minus the identically-zero amplitude/current
columns for a clean A==1 knee, PLUS (div k)^2 so the true law is representable):

    {1, w, w^2, Lap w, |grad w|, div k, (div k)^2}

Output (relative): results/effective_vs_bare_dwdt/
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

from op_extract.t1_common import riesz_k_field, k_from_theta          # noqa: E402
from self_dual_sindy import fit_target, stlsq, r_squared             # noqa: E402


# --------------------------------------------------------------------------
# minimal args for fit_target (mirrors time_sindy_riesz._Args selection knobs)
# --------------------------------------------------------------------------

class _FitArgs:
    thresholds = tuple(np.geomspace(1e-4, 1.0, 25))
    rel_tol_pareto = 0.01
    n_boot = 100
    boot_frac = 0.5


LIB = ["1", "w", "w^2", "Lap w", "|grad w|", "div k", "(div k)^2"]


# --------------------------------------------------------------------------
# bare dynamics: 1-D heat relaxation of the knee bend
# --------------------------------------------------------------------------

def evolve_knee(mu=0.85, nu=0.05, nframes=60, save_every=8,
                nx=192, n_lam_x=4.0, n_lam_y=7.0, cfl=0.4):
    """Evolve theta = k1 x + phi(y, t), phi_t = nu phi_yy (explicit, ghost-slope
    Neumann so the far field keeps its linear +/- k2 slope). Returns
    theta[Ny,Nx,Nt], x, y, tt, and nu (the analytic Lap-w coefficient)."""
    k1, k2 = np.sqrt(1.0 - mu ** 2), mu
    lam_x, lam_y = 2 * np.pi / k1, 2 * np.pi / k2
    Lx, Ly = n_lam_x * lam_x, n_lam_y * lam_y
    dx = Lx / nx
    x = np.arange(-Lx / 2, Lx / 2, dx)
    dy = dx
    y = np.arange(-Ly / 2, Ly / 2, dy)
    ny = len(y)
    X, Y = np.meshgrid(x, y)

    phi = np.log(np.cosh(k2 * y))                 # 1-D bend profile
    r = cfl                                        # nu*dt_int/dy^2 = cfl
    dt_int = cfl * dy ** 2 / nu
    slope = k2 * np.tanh(k2 * y[-1])               # far-field d_y phi (~ +/- k2)

    thetas, tt = [], []
    step, t = 0, 0.0
    total_steps = nframes * save_every
    while step <= total_steps:
        if step % save_every == 0:
            th = k1 * X[None, 0][0] * 0.0          # placeholder, rebuilt below
            th = k1 * X + phi[:, None]
            thetas.append(th.copy())
            tt.append(t)
        lap = np.empty_like(phi)
        lap[1:-1] = phi[2:] - 2 * phi[1:-1] + phi[:-2]
        # ghost points enforcing fixed far-field slope (Neumann on d_y phi)
        gm = phi[1] - 2 * slope * dy
        gp = phi[-2] + 2 * slope * dy
        lap[0] = phi[1] - 2 * phi[0] + gm
        lap[-1] = gp - 2 * phi[-1] + phi[-2]
        phi = phi + r * lap
        t += dt_int
        step += 1

    theta = np.stack(thetas, axis=-1)
    return theta, x, y, np.asarray(tt), float(nu), int(np.argmin(np.abs(y)))


# --------------------------------------------------------------------------
# field helpers
# --------------------------------------------------------------------------

def _grad(f, dx, dy, axis):
    return np.gradient(f, dx, axis=1) if axis == "x" else np.gradient(f, dy, axis=0)


def k_of_route(route, theta_frame, x, y, Lx, Ly, sigma_dir):
    """Return (k1, k2) for one route on one frame."""
    if route == "exact":
        dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
        return np.gradient(theta_frame, dx, axis=1), np.gradient(theta_frame, dy, axis=0)
    if route == "k_from_theta":
        r = k_from_theta(theta_frame, x, y, Lx, Ly, smooth=1.0)
        return r["k1"], r["k2"]
    if route == "riesz":
        u = np.cos(theta_frame)
        r = riesz_k_field(u, x, y, Lx, Ly, sigma_dir)
        return np.asarray(r["k1"]), np.asarray(r["k2"])
    raise ValueError(route)


def library_columns(k1, k2, dx, dy):
    """Build the shared library from a route's (k1, k2)."""
    w = k1 ** 2 + k2 ** 2 - 1.0
    wx, wy = _grad(w, dx, dy, "x"), _grad(w, dx, dy, "y")
    wxx = _grad(wx, dx, dy, "x")
    wyy = _grad(wy, dx, dy, "y")
    lapw = wxx + wyy
    gradw = np.hypot(wx, wy)
    divk = _grad(k1, dx, dy, "x") + _grad(k2, dx, dy, "y")
    cols = {
        "1": np.ones_like(w), "w": w, "w^2": w ** 2,
        "Lap w": lapw, "|grad w|": gradw, "div k": divk, "(div k)^2": divk ** 2,
    }
    return w, cols


# --------------------------------------------------------------------------
# assemble samples and fit one route
# --------------------------------------------------------------------------

def samples_for_route(route, theta, x, y, tt, core, band_y, x_trim, sigma_dir,
                      hole=0.0):
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    Lx, Ly = x[-1] - x[0], y[-1] - y[0]
    X, Y = np.meshgrid(x, y)
    band = ((np.abs(Y - y[core]) <= band_y) & (np.abs(Y - y[core]) >= hole)
            & (X > x[0] + x_trim) & (X < x[-1] - x_trim))

    # precompute w per frame for the time derivative
    ws = []
    for t in range(theta.shape[-1]):
        k1, k2 = k_of_route(route, theta[:, :, t], x, y, Lx, Ly, sigma_dir)
        ws.append(k1 ** 2 + k2 ** 2 - 1.0)

    rows_T, rows_y, rows_x = [], [], []
    for t in range(1, theta.shape[-1] - 1):
        dt2 = float(tt[t + 1] - tt[t - 1])
        if dt2 <= 0:
            continue
        k1, k2 = k_of_route(route, theta[:, :, t], x, y, Lx, Ly, sigma_dir)
        w, cols = library_columns(k1, k2, dx, dy)
        dwdt = (ws[t + 1] - ws[t - 1]) / dt2
        m = band
        C = np.column_stack([cols[nm][m] for nm in LIB] + [dwdt[m]])
        fin = np.all(np.isfinite(C), axis=1)
        if not fin.any():
            continue
        C = C[fin]
        rows_T.append(C[:, :-1])
        rows_y.append(C[:, -1])
        rows_x.append(X[m][fin])
    Theta = np.vstack(rows_T)
    yv = np.concatenate(rows_y)
    xs = np.concatenate(rows_x)
    return Theta, yv, xs


def vif(names, Theta):
    out = {}
    for j, nm in enumerate(names):
        if nm == "1":
            continue
        others = [k for k in range(Theta.shape[1]) if k != j]
        beta, *_ = np.linalg.lstsq(Theta[:, others], Theta[:, j], rcond=None)
        ss = np.sum((Theta[:, j] - Theta[:, others] @ beta) ** 2)
        tot = max(np.sum((Theta[:, j] - Theta[:, j].mean()) ** 2), 1e-12)
        out[nm] = float(1.0 / max(ss / tot, 1e-6))
    return out


def fit_route(route, theta, x, y, tt, core, args, rng, hole=0.0):
    Theta, yv, xs = samples_for_route(
        route, theta, x, y, tt, core, args.band_y, args.x_trim, args.sigma_dir,
        hole=hole)
    if len(yv) > args.max_samples:
        idx = rng.choice(len(yv), args.max_samples, replace=False)
        Theta, yv, xs = Theta[idx], yv[idx], xs[idx]
    res = fit_target(LIB, Theta, yv, _FitArgs(), rng, xs < np.median(xs))
    supp = np.where(res["support"])[0]
    coefs = {LIB[i]: float(res["coef_full"][i]) for i in supp}
    errs = {LIB[i]: float(res["boots"][:, j].std()) for j, i in enumerate(supp)}
    return dict(n=len(yv), coefs=coefs, errs=errs,
                r2_full=float(res["r2_full"]),
                r2_holdout=float(res["chosen"]["r2_test"]),
                has_w2=any(k == "w^2" for k in coefs),
                vif=vif(LIB, Theta))


def fmt(coefs, errs):
    if not coefs:
        return "0"
    parts = []
    for nm, c in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
        e = errs.get(nm, 0.0)
        parts.append(f"{c:+.3g}({e:.2g})*{nm}")
    return " ".join(parts)


def run_hole_sweep(mu=0.85, nu=0.05, hole_units=(0, 1, 2),
                   out="results/effective_vs_bare_dwdt"):
    """Sweep the core hole and report, per route, the selected support, w^2
    presence, and (div k)^2 coefficient. Tests whether the direct route's
    spurious w^2 PERSISTS as the core is removed (systemic artifact) or WEAKENS
    (tied to the near-core magnitude overshoot); and confirms the exact and
    k_from_theta routes stay w^2-free at every hole."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    theta, x, y, tt, nu_val, core = evolve_knee(mu=mu, nu=nu)
    hpx = np.pi / mu

    class A:
        band_y = 9.0          # wide enough that holes leave interior samples
        x_trim = 4.0
        sigma_dir = 0.4
        max_samples = 300000

    print(f"core-hole sweep  mu={mu}  nu={nu_val}  pi/mu={hpx:.2f}")
    print(f"  TRUE law: dw/dt = {nu_val:+.3g}*Lap w {(-2*nu_val):+.3g}*(div k)^2 "
          f"(NO w^2)\n")
    print(f"  {'route':>13} {'hole':>5} {'N':>8} {'R2':>7} {'w^2?':>6} "
          f"{'w^2 coef':>9} {'(div k)^2':>10}")
    rows = {}
    for route in ["exact", "k_from_theta", "riesz"]:
        for hu in hole_units:
            r = fit_route(route, theta, x, y, tt, core, A, rng, hole=hu * hpx)
            rows[(route, hu)] = r
            w2c = r["coefs"].get("w^2", 0.0)
            dk2 = r["coefs"].get("(div k)^2", 0.0)
            print(f"  {route:>13} {hu:5g} {r['n']:8d} {r['r2_full']:7.3f} "
                  f"{('YES' if r['has_w2'] else 'no'):>6} {w2c:9.3f} {dk2:10.3f}")
        print()

    # figure: w^2 coefficient vs hole per route (the artifact-vs-hole trace)
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for route in ["exact", "k_from_theta", "riesz"]:
        hs = list(hole_units)
        w2 = [rows[(route, h)]["coefs"].get("w^2", 0.0) for h in hs]
        dk = [rows[(route, h)]["coefs"].get("(div k)^2", 0.0) for h in hs]
        ax.plot(hs, w2, "o-", label=route)
        ax2.plot(hs, dk, "o-", label=route)
    ax.axhline(0, color="0.6", lw=0.8); ax.set_xlabel("core hole (pi/mu)")
    ax.set_ylabel("selected w^2 coefficient")
    ax.set_title(f"spurious w^2 vs core removal (mu={mu})"); ax.legend(fontsize=8)
    ax2.axhline(-2 * nu_val, color="C3", ls=":", lw=1)
    ax2.axhline(0, color="0.6", lw=0.8); ax2.set_xlabel("core hole (pi/mu)")
    ax2.set_ylabel("selected (div k)^2 coefficient")
    ax2.set_title("true (div k)^2 term (dotted = -2 nu)"); ax2.legend(fontsize=8)
    fig.tight_layout()
    p = out / f"hole_sweep_mu{mu:.2f}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  figure -> {p}")
    return rows


def _wdivk(route, theta_t, x, y, sigma_dir=0.4):
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    Lx, Ly = x[-1] - x[0], y[-1] - y[0]
    k1, k2 = k_of_route(route, theta_t, x, y, Lx, Ly, sigma_dir)
    w = k1 ** 2 + k2 ** 2 - 1.0
    divk = (np.gradient(k1, dx, axis=1) + np.gradient(k2, dy, axis=0))
    return w, divk


def surface_diagnostic(mu=0.85, nu=0.05, ywin=14.0,
                       routes=("exact", "k_from_theta", "riesz"),
                       out="results/effective_vs_bare_dwdt"):
    """Surfaces of w=|k|^2-1 and div k at start/middle/final frames for each
    route, on the evolving knee whose true law has NO w^2. Comparing exact vs
    extracted shows why the direct route manufactures w^2: its w overshoots and
    its div k is corrupted, so the true (div k)^2 term cannot land and the fit
    substitutes the w-polynomial block."""
    import matplotlib.pyplot as plt
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    theta, x, y, tt, nu_val, core = evolve_knee(mu=mu, nu=nu)
    Nt = theta.shape[-1]
    frames = [0, Nt // 2, Nt - 1]
    flabels = [f"start t={tt[frames[0]]:.2f}", f"mid t={tt[frames[1]]:.2f}",
               f"final t={tt[frames[2]]:.2f}"]
    X, Y = np.meshgrid(x, y)
    ymask = np.abs(Y - y[core]) <= ywin
    xin = (X > x[0] + 0.12 * (x[-1] - x[0])) & (X < x[-1] - 0.12 * (x[-1] - x[0]))
    disp = ymask & xin
    rr = np.where(disp.any(axis=1))[0]; cc = np.where(disp.any(axis=0))[0]
    r0, r1, c0, c1 = rr[0], rr[-1] + 1, cc[0], cc[-1] + 1
    ext = [x[c0], x[c1 - 1], y[r0] - y[core], y[r1 - 1] - y[core]]

    for qname, qi in (("w", 0), ("div_k", 1)):
        fields = {}
        for route in routes:
            for fi in frames:
                q = _wdivk(route, theta[:, :, fi], x, y)[qi]
                fields[(route, fi)] = np.where(disp, q, np.nan)[r0:r1, c0:c1]
        allv = np.concatenate([np.abs(f[np.isfinite(f)]).ravel()
                               for f in fields.values()])
        v = np.nanpercentile(allv, 98)
        nR, nC = len(routes), len(frames)
        fig, ax = plt.subplots(nR, nC, figsize=(4.0 * nC, 3.1 * nR),
                               squeeze=False)
        for i, route in enumerate(routes):
            for j, fi in enumerate(frames):
                im = ax[i, j].imshow(fields[(route, fi)], extent=ext,
                                     origin="lower", cmap="RdBu_r",
                                     vmin=-v, vmax=v, aspect="auto")
                if i == 0:
                    ax[i, j].set_title(flabels[j], fontsize=9)
                if j == 0:
                    ax[i, j].set_ylabel(f"{route}\n(y - y_core)", fontsize=9)
                ax[i, j].set_xticks([])
        fig.colorbar(im, ax=ax, shrink=0.7,
                     label=("w = |k|^2 - 1" if qi == 0 else "div k"))
        fig.suptitle(f"{'w' if qi==0 else 'div k'} surfaces (mu={mu}, shared "
                     f"scale +/-{v:.2g}); true law has no w^2", fontsize=11)
        p = out / f"surfaces_{qname}_mu{mu:.2f}.png"
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f"  {qname} surfaces -> {p}")


def library_surfaces(mu=0.85, nu=0.05, frame="mid", ywin=14.0,
                     routes=("exact", "riesz"), out="results/effective_vs_bare_dwdt"):
    """Surfaces of each dw_dt library term at one frame, exact vs extracted, so
    you can see which columns are corrupted by the extraction. Shared symmetric
    scale per term (row) so the two routes are directly comparable."""
    import matplotlib.pyplot as plt
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    theta, x, y, tt, nu_val, core = evolve_knee(mu=mu, nu=nu)
    Nt = theta.shape[-1]
    fi = {"start": 1, "mid": Nt // 2, "final": Nt - 2}.get(frame, Nt // 2)
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    X, Y = np.meshgrid(x, y)
    disp = (np.abs(Y - y[core]) <= ywin) & (X > x[0] + 0.12 * (x[-1] - x[0])) \
        & (X < x[-1] - 0.12 * (x[-1] - x[0]))
    rr = np.where(disp.any(axis=1))[0]; cc = np.where(disp.any(axis=0))[0]
    r0, r1, c0, c1 = rr[0], rr[-1] + 1, cc[0], cc[-1] + 1
    ext = [x[c0], x[c1 - 1], y[r0] - y[core], y[r1 - 1] - y[core]]
    terms = ["w", "w^2", "Lap w", "|grad w|", "div k", "(div k)^2"]

    per = {}
    for route in routes:
        k1, k2 = k_of_route(route, theta[:, :, fi], x, y,
                            x[-1] - x[0], y[-1] - y[0], 0.4)
        _, cols = library_columns(k1, k2, dx, dy)
        per[route] = {t: np.where(disp, cols[t], np.nan)[r0:r1, c0:c1]
                      for t in terms}

    nT, nR = len(terms), len(routes)
    fig, ax = plt.subplots(nT, nR, figsize=(4.0 * nR, 2.5 * nT), squeeze=False)
    for i, t in enumerate(terms):
        allv = np.concatenate([np.abs(per[r][t][np.isfinite(per[r][t])]).ravel()
                               for r in routes])
        v = np.nanpercentile(allv, 98) if allv.size else 1.0
        for j, route in enumerate(routes):
            im = ax[i, j].imshow(per[route][t], extent=ext, origin="lower",
                                 cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
            if i == 0:
                ax[i, j].set_title(route, fontsize=10)
            if j == 0:
                ax[i, j].set_ylabel(t, fontsize=10)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        fig.colorbar(im, ax=ax[i, :].tolist(), shrink=0.85)
    fig.suptitle(f"dw_dt library terms, exact vs extracted "
                 f"(mu={mu}, {frame} frame t={tt[fi]:.1f})", fontsize=11)
    p = out / f"library_surfaces_mu{mu:.2f}_{frame}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  library-term surfaces -> {p}")
    return p


def run(mu=0.85, nu=0.05, out="results/effective_vs_bare_dwdt"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    theta, x, y, tt, nu_val, core = evolve_knee(mu=mu, nu=nu)
    print(f"evolved knee mu={mu} nu={nu_val}  grid {theta.shape}  "
          f"frames saved at t in [{tt[0]:.2f},{tt[-1]:.2f}]")
    print(f"  EXACT target law:  d_t w = {nu_val:+.4g}*Lap w  "
          f"{-2*nu_val:+.4g}*(div k)^2   (no w^2, no w)\n")

    class A:
        band_y = 3.0
        x_trim = 4.0
        sigma_dir = 0.4
        max_samples = 300000

    results = {}
    for route in ["exact", "k_from_theta", "riesz"]:
        r = fit_route(route, theta, x, y, tt, core, A, rng)
        results[route] = r
        print(f"[{route:>13}] N={r['n']:>7}  R2={r['r2_full']:.4f} "
              f"holdout={r['r2_holdout']:.4f}")
        print(f"                dw/dt = {fmt(r['coefs'], r['errs'])}")
        hi = {k: v for k, v in r["vif"].items() if v > 10}
        if hi:
            print("                VIF>10: " + ", ".join(
                f"{k}={v:.0f}" for k, v in sorted(hi.items(), key=lambda kv: -kv[1])))
        print()

    # verdict on the falsifiable target
    print("  ---- term-preservation verdict ----")
    true_supp = {"Lap w", "(div k)^2"}
    for route, r in results.items():
        sel = set(r["coefs"])
        extra = sel - true_supp
        missing = true_supp - sel
        w2 = "w^2" in sel
        tag = ("keeps true support" if sel == true_supp else
               ("promotes w^2" if w2 else "support differs"))
        print(f"    {route:>13}: selected={sorted(sel)}  -> {tag}"
              + (f"  [extra={sorted(extra)}]" if extra else "")
              + (f"  [missing={sorted(missing)}]" if missing else ""))

    with open(out / "summary.json", "w") as fh:
        json.dump({"mu": mu, "nu": nu_val,
                   "exact_law": {"Lap w": nu_val, "(div k)^2": -2 * nu_val},
                   "routes": results}, fh, indent=2)

    # figure: coefficient of each library term per route (bar), plus R2
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    terms = [t for t in LIB if t != "1"]
    width = 0.25
    for i, route in enumerate(["exact", "k_from_theta", "riesz"]):
        vals = [results[route]["coefs"].get(t, 0.0) for t in terms]
        ax.bar(np.arange(len(terms)) + (i - 1) * width, vals, width, label=route)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axhline(nu_val, color="C0", ls=":", lw=1)
    ax.axhline(-2 * nu_val, color="C3", ls=":", lw=1)
    ax.set_xticks(np.arange(len(terms)))
    ax.set_xticklabels(terms, rotation=30, ha="right")
    ax.set_ylabel("selected coefficient")
    ax.set_title(f"dw/dt terms per route (mu={mu}); dotted = exact nu, -2nu")
    ax.legend(fontsize=8)
    r2f = [results[r]["r2_full"] for r in ["exact", "k_from_theta", "riesz"]]
    r2h = [results[r]["r2_holdout"] for r in ["exact", "k_from_theta", "riesz"]]
    xx = np.arange(3)
    ax2.bar(xx - 0.2, r2f, 0.4, label="R2 full")
    ax2.bar(xx + 0.2, r2h, 0.4, label="R2 holdout")
    ax2.set_xticks(xx)
    ax2.set_xticklabels(["exact", "k_from_theta", "riesz"], rotation=20)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("R2")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"effective_vs_bare_mu{mu:.2f}.png", dpi=150)
    plt.close(fig)
    print(f"\n  figure -> {out / f'effective_vs_bare_mu{mu:.2f}.png'}")
    print(f"  summary -> {out / 'summary.json'}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mu", type=float, default=0.85)
    p.add_argument("--nu", type=float, default=0.05)
    # p.add_argument("--sweep", action="store_true",
    #                help="run the core-hole sweep instead of the single-band fit")
    # p.add_argument("--surfaces", action="store_true",
    #                help="plot w and div k surfaces (exact vs extracted) at "
    #                     "start/mid/final")
    # p.add_argument("--libsurf", action="store_true",
    #                help="plot all dw_dt library-term surfaces, exact vs extracted")
    p.add_argument("--surfaces", type=bool, default=False)
    p.add_argument("--sweep", type=bool, default=True)
    p.add_argument("--libsurf", type=bool, default=False)
    a = p.parse_args()
    if a.libsurf:
        library_surfaces(mu=a.mu, nu=a.nu)
    elif a.surfaces:
        surface_diagnostic(mu=a.mu, nu=a.nu)
    elif a.sweep:
        run_hole_sweep(mu=a.mu, nu=a.nu)
    else:
        run(mu=a.mu, nu=a.nu)
