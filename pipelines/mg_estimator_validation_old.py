# pipelines/mg_estimator_validation.py
"""
Stage 1 validation of the monogenic-gradient wavevector and phase
estimator on the exact knee and the exact PN multi-dislocation field.

The estimator
-------------
With the Riesz pair r = (R1 u, R2 u) and the director n = (d1, d2) from
the frozen riesz_director (doubled-angle average at sigma_dir; the
director only fixes the sign structure of the quadrature), form

    z = u + i (n . r)             (~ A exp(i theta) for u = A cos theta)
    k = Im(conj(z) grad z) / |z|^2,   theta = arg z,   A = |z|,

with the gradient taken by the product rule and the director gradient
written through the doubled angle, so the k1 >= 0 seam of n is never
differentiated (see mg_estimator). The argument is never differentiated
and never unwrapped, so the |k| zeros of the folded monogenic phase
atan2(|r|, u) in [0, pi] do not occur; the only zeros of |z| are defect
cores.

Reference instruments on the same fields
----------------------------------------
  demod  : row-by-row analytic signal z = u + i H_x[u] (beta_knee_fit.py,
           Writeup 1), same gradient formula, Tukey x taper (session 7).
  riesz  : riesz_k_field, the production route, kmag from the ratio of
           averages at sigma_avg (pi and pi/2), direction from the same
           director.
  folded : grad of the folded monogenic phase, the route abandoned a
           year ago; kept only for the |k|-zeros panel.

Metrics
-------
  k error by distance to the core (knee: |y| bins; PN: distance to the
  nearest defect in units of the spacing, upper half only), relative to
  the RMS of the exact k in the same bin and sign-agnostic (the director
  carries the k1 >= 0 convention, PN theta is odd in y);
  wrapped phase error of arg z against the exact theta;
  amplitude error of |z| against A = 1;
  fraction of the window with |k| < 0.5 (the old artifact count);
  knee self-dual slope beta from the x-averaged profile (tanh fit and
  regression with intercept, holdout-free R^2 of the linear law), the
  quantity the calibration work showed the production kmag deflates.

Outputs (relative): results/mg_estimator_validation/<stamp>/
  config.json, k_error_bins.csv, summary.csv, knee_beta.csv, report.txt,
  figures err_vs_distance_<field>.png, knee_profiles.png,
  pn_error_maps_mu<mu>.png, kzeros_panel_mu<mu>.png, phase_panel_mu<mu>.png

Changelog
---------
v1.1  seam-free product-rule form is the estimator of record. The naive
      spectral gradient of z rang across the director seam wherever the
      stripes are vertical (the far field of a finite PN row relaxes to
      theta ~ y): measured 17 to 19 percent far-field error at mu 0.4
      and 0.6, 12 percent in the [1,2) spacing bin at mu 0.85; the
      seam-free form reads 0.3 to 1.1 percent in every far bin at every
      mu, knee numbers unchanged to 4 digits. Old form kept as
      --with-mg-z.
v1.0  first version. Sandbox numbers at mu 0.85: knee k rel RMS 3.1
      percent in the window, 16 percent for |y| < 1, below 1 percent
      beyond |y| = 8; row demod on PN 11 to 20 percent (rows are not
      narrowband above a defect row).

Run:
  python mg_estimator_validation.py
  python mg_estimator_validation.py --mu 0.85 0.95 --sigma-dir 0.4
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT / "src", _ROOT / "src" / "op_extract", _ROOT / "src" / "utils",
          _HERE):
    sys.path.insert(0, str(p))

from op_extract import t1_common as t1c                      # noqa: E402
fdv = sys.modules.get("utils.field_diagnostics_viz")
if fdv is None:
    import field_diagnostics_viz as fdv                      # noqa: E402

from scipy.optimize import curve_fit                          # noqa: E402
from scipy.signal import hilbert                              # noqa: E402
from scipy.signal.windows import tukey                        # noqa: E402

import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402
try:
    from paper_style import use_paper_style
    use_paper_style()
except Exception:
    pass


# ---------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------

def _cgrad(z, Lx, Ly, which):
    """spectral derivative of a complex field"""
    return (fdv.spectral_deriv(z.real, Lx, Ly, which)
            + 1j * fdv.spectral_deriv(z.imag, Lx, Ly, which))


def k_from_z(z, Lx, Ly):
    az2 = np.abs(z) ** 2
    az2 = np.maximum(az2, 1e-14)
    k1 = np.imag(np.conj(z) * _cgrad(z, Lx, Ly, "x")) / az2
    k2 = np.imag(np.conj(z) * _cgrad(z, Lx, Ly, "y")) / az2
    return k1, k2


def mg_estimator(u, Lx, Ly, sigma_dir):
    """monogenic-gradient k, phase and amplitude, seam-free form.

    k = Im(conj(z) grad z)/|z|^2 with z = u + i v, v = n . r, expanded by
    the product rule so that the director gradient enters through the
    doubled angle 2 psi (which has no seam): grad n = n_perp grad psi,
    grad psi = (c2 grad s2 - s2 grad c2)/(2 (c2^2 + s2^2)). Under n -> -n
    the result flips sign pointwise and nothing discontinuous is
    differentiated. Differentiating z itself spectrally (mg_estimator_z)
    rings across the k1 >= 0 seam of the director wherever stripes are
    vertical, measured as 10 to 19 percent far-field error on shallow-mu
    PN fields against 0.3 to 1 percent here."""
    mono = fdv.monogenic(u, Lx, Ly)
    rd = fdv.riesz_director(u, Lx, Ly, sigma_dir)
    R1, R2, d1, d2 = mono["R1"], mono["R2"], rd["d1"], rd["d2"]
    c2, s2 = np.cos(2.0 * rd["angle"]), np.sin(2.0 * rd["angle"])
    D = lambda a, w: fdv.spectral_deriv(a, Lx, Ly, w)
    v = d1 * R1 + d2 * R2                     # n . r
    vp = -d2 * R1 + d1 * R2                   # n_perp . r
    den = np.maximum(u ** 2 + v ** 2, 1e-14)
    k = []
    for w in ("x", "y"):
        dpsi = 0.5 * (c2 * D(s2, w) - s2 * D(c2, w)) / np.maximum(
            c2 ** 2 + s2 ** 2, 1e-12)
        dv = d1 * D(R1, w) + d2 * D(R2, w) + vp * dpsi
        k.append((u * dv - v * D(u, w)) / den)
    return {"k1": k[0], "k2": k[1], "z": u + 1j * v,
            "theta": np.arctan2(v, u), "A": np.hypot(u, v),
            "d1": d1, "d2": d2}


def mg_estimator_z(u, Lx, Ly, sigma_dir):
    """the naive form: spectral gradient of z itself (seam-sensitive);
    kept as an opt-in reference (--with-mg-z)"""
    mono = fdv.monogenic(u, Lx, Ly)
    rd = fdv.riesz_director(u, Lx, Ly, sigma_dir)
    z = u + 1j * (rd["d1"] * mono["R1"] + rd["d2"] * mono["R2"])
    k1, k2 = k_from_z(z, Lx, Ly)
    return {"k1": k1, "k2": k2, "z": z, "theta": np.angle(z),
            "A": np.abs(z), "d1": rd["d1"], "d2": rd["d2"]}


def demod_estimator(u, x, y, Lx, Ly, tukey_alpha):
    """row-by-row analytic signal along x, Tukey taper on the rows"""
    T = tukey(len(x), tukey_alpha)[None, :]
    za = hilbert(u * T, axis=1)
    k1, k2 = k_from_z(za, Lx, Ly)
    return {"k1": k1, "k2": k2, "z": za, "theta": np.angle(za),
            "A": np.abs(za) / np.maximum(T, 1e-3)}


def riesz_reference(u, x, y, Lx, Ly, sigma_dir, sigma_avg):
    rk = t1c.riesz_k_field(u, x, y, Lx, Ly, sigma_dir=sigma_dir,
                           sigma_avg=sigma_avg)
    return {"k1": rk["k1"], "k2": rk["k2"], "A": rk["A"], "theta": None}


def folded_phase_k(u, x, y, Lx, Ly):
    """the abandoned route: gradient of the folded monogenic phase"""
    mono = fdv.monogenic(u, Lx, Ly)
    ph = mono["phase"]
    k1 = np.gradient(ph, x[1] - x[0], axis=1)
    k2 = np.gradient(ph, y[1] - y[0], axis=0)
    return {"k1": k1, "k2": k2, "A": mono["A"], "theta": ph}


# ---------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------

def window_and_distance(fld, win_frac):
    """evaluation window and the distance-to-core field"""
    X, Y = fld["X"], fld["Y"]
    win = (np.abs(X) < win_frac * fld["Lx"]) & (np.abs(Y) < win_frac * fld["Ly"])
    if fld["field"] == "pn":
        win &= Y > 0                                  # theta odd in y
        xd = fld["xd"]
        dist = np.min(np.hypot(X[..., None] - xd[None, None, :],
                               Y[..., None]), axis=-1) / fld["Lsp"]
    else:
        dist = np.abs(Y)
    return win, dist


def k_error_bins(fld, est, win, dist, edges):
    """sign-agnostic relative RMS error of k per distance bin"""
    k1e, k2e = fld["k1"], fld["k2"]
    ep = np.hypot(est["k1"] - k1e, est["k2"] - k2e)
    em = np.hypot(est["k1"] + k1e, est["k2"] + k2e)
    err = np.minimum(ep, em)
    kex = np.hypot(k1e, k2e)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = win & (dist >= lo) & (dist < hi)
        if m.sum() < 10:
            continue
        rel = np.sqrt(np.mean(err[m] ** 2)) / np.sqrt(np.mean(kex[m] ** 2))
        rows.append({"lo": lo, "hi": hi, "n": int(m.sum()), "rel_rms": rel,
                     "max": float(err[m].max())})
    return rows, err


def scalar_metrics(fld, est, win, dist, far_from):
    """window-wide numbers; phase and amplitude only where defined"""
    out = {}
    kex = np.hypot(fld["k1"], fld["k2"])
    ep = np.hypot(est["k1"] - fld["k1"], est["k2"] - fld["k2"])
    em = np.hypot(est["k1"] + fld["k1"], est["k2"] + fld["k2"])
    err = np.minimum(ep, em)
    out["k_rel_rms_window"] = float(np.sqrt(np.mean(err[win] ** 2))
                                    / np.sqrt(np.mean(kex[win] ** 2)))
    far = win & (dist >= far_from)
    out["k_rel_rms_far"] = float(np.sqrt(np.mean(err[far] ** 2))
                                 / np.sqrt(np.mean(kex[far] ** 2)))
    kmag = np.hypot(est["k1"], est["k2"])
    out["frac_k_below_half"] = float(np.mean(kmag[win] < 0.5))
    if est.get("theta") is not None:
        d = np.angle(np.exp(1j * (est["theta"] - fld["theta"])))
        dm = np.angle(np.exp(1j * (-est["theta"] - fld["theta"])))
        d = np.where(np.abs(d) <= np.abs(dm), d, dm)   # sign-agnostic
        out["phase_rms_rad_far"] = float(np.sqrt(np.mean(d[far] ** 2)))
        out["phase_max_rad_far"] = float(np.abs(d[far]).max())
    else:
        out["phase_rms_rad_far"] = float("nan")
        out["phase_max_rad_far"] = float("nan")
    out["amp_rel_rms_far"] = float(np.sqrt(np.mean((est["A"][far] - 1.0) ** 2)))
    return out, err


def knee_beta(fld, est, band_over_mu=3.0):
    """x-averaged profile on the knee: tanh beta, regression beta with
    intercept, R^2 of the linear law div k = b - c w"""
    x, y, mu = fld["x"], fld["y"], fld["mu"]
    sl = slice(len(x) // 4, 3 * len(x) // 4)
    band = np.abs(y) < band_over_mu / mu
    k1, k2 = est["k1"], est["k2"]
    if fld["field"] == "knee":                        # enforce odd k2
        k2 = np.abs(k2) * np.sign(fld["Y"])
    k2b = k2[:, sl].mean(1)
    wb = (k1[:, sl] ** 2 + k2[:, sl] ** 2).mean(1) - (fld["k1_ff"] ** 2
                                                       + fld["k2_ff"] ** 2)
    f = lambda yv, m, q, y0: m * np.tanh(q * (yv - y0))
    try:
        p, _ = curve_fit(f, y[band], k2b[band], p0=[mu, 0.9 * mu, 0.0],
                         maxfev=9000)
        b_tanh = abs(p[1] / p[0])
    except RuntimeError:
        b_tanh = float("nan")
    dk = np.gradient(k2b, y[1] - y[0])
    Xm = np.c_[np.ones(band.sum()), wb[band]]
    c = np.linalg.lstsq(Xm, dk[band], rcond=None)[0]
    r2 = 1.0 - np.sum((dk[band] - Xm @ c) ** 2) / np.sum(
        (dk[band] - dk[band].mean()) ** 2)
    return {"beta_tanh": float(b_tanh), "beta_regr": float(-c[1]),
            "r2_linear": float(r2), "k2b": k2b, "wb": wb, "band": band}


# ---------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------

def fig_err_vs_distance(field, table, edges, mus, sources, path):
    fig, axs = plt.subplots(1, len(mus), figsize=(3.6 * len(mus), 3.4),
                            sharey=True, squeeze=False)
    mids = 0.5 * (np.array(edges[:-1]) + np.array(edges[1:]))
    for ax, mu in zip(axs[0], mus):
        for src in sources:
            rows = [r for r in table if r["field"] == field and r["mu"] == mu
                    and r["source"] == src]
            if not rows:
                continue
            xs = [0.5 * (r["lo"] + r["hi"]) for r in rows]
            ax.plot(xs, [r["rel_rms"] for r in rows], "o-", ms=3, label=src)
        ax.set_yscale("log")
        ax.set_title("%s, mu = %.2f" % (field, mu))
        ax.set_xlabel("|y|" if field == "knee" else "distance to defect / spacing")
        ax.grid(True, which="both", alpha=0.3)
    axs[0][0].set_ylabel("relative RMS error of k")
    axs[0][-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_knee_profiles(profiles, path):
    """profiles[(mu)] = dict source -> (y, k2b, band), plus exact"""
    mus = sorted(profiles)
    fig, axs = plt.subplots(1, len(mus), figsize=(3.6 * len(mus), 3.4),
                            squeeze=False)
    for ax, mu in zip(axs[0], mus):
        d = profiles[mu]
        y = d["y"]
        band = np.abs(y) < 3.0 / mu
        ax.plot(y[band], (mu * np.tanh(mu * y))[band], "k-", lw=1.4,
                label="exact")
        for src, k2b in d["sources"].items():
            ax.plot(y[band], k2b[band], "--", lw=1, label=src)
        ax.set_title("knee, mu = %.2f" % mu)
        ax.set_xlabel("y")
        ax.set_ylabel("x-averaged k2")
    axs[0][-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_pn_error_maps(fld, errs, path, half_spacings=1.5):
    """|k error| around the central defect, one panel per source"""
    xd = fld["xd"]
    xc = xd[len(xd) // 2]
    L = half_spacings * fld["Lsp"]
    x, y = fld["x"], fld["y"]
    ix = (x > xc - L) & (x < xc + L)
    iy = (y > -L) & (y < L)
    fig, axs = plt.subplots(1, len(errs), figsize=(3.6 * len(errs), 3.4),
                            squeeze=False)
    for ax, (src, e) in zip(axs[0], errs.items()):
        im = ax.imshow(e[np.ix_(iy, ix)], origin="lower",
                       extent=[x[ix][0], x[ix][-1], y[iy][0], y[iy][-1]],
                       vmin=0, vmax=0.5, cmap="magma")
        ax.plot(xd, np.zeros_like(xd), "w+", ms=8)
        ax.set_xlim(xc - L, xc + L)
        ax.set_title("%s: |k - k_exact|" % src)
        ax.set_xlabel("x")
    axs[0][0].set_ylabel("y")
    fig.colorbar(im, ax=axs[0].tolist(), shrink=0.8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_kzeros_panel(fld, folded, mg, path, n_periods=2.0):
    """|k| from the folded phase and from the new route on a knee zoom"""
    x, y = fld["x"], fld["y"]
    L = n_periods * 2 * np.pi / fld["k1_ff"]
    ix = np.abs(x) < L
    iy = np.abs(y) < 0.8 * L
    ext = [x[ix][0], x[ix][-1], y[iy][0], y[iy][-1]]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    axs[0].imshow(fld["u"][np.ix_(iy, ix)], origin="lower", extent=ext,
                  cmap="gray")
    axs[0].set_title("u")
    for ax, est, t in ((axs[1], folded, "|grad(folded phase)|"),
                       (axs[2], mg, "|k|, monogenic-gradient")):
        im = ax.imshow(np.hypot(est["k1"], est["k2"])[np.ix_(iy, ix)],
                       origin="lower", extent=ext, vmin=0, vmax=1.5,
                       cmap="viridis")
        ax.set_title(t)
    fig.colorbar(im, ax=axs[2], shrink=0.8)
    fig.suptitle("knee, mu = %.2f" % fld["mu"])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_phase_panel(fld, mg, path, n_periods=2.0):
    x, y = fld["x"], fld["y"]
    L = n_periods * 2 * np.pi / fld["k1_ff"]
    ix = np.abs(x) < L
    iy = (y > 0.2 * L) & (y < 2.5 * L) if fld["field"] == "pn" else np.abs(y) < 2 * L
    ext = [x[ix][0], x[ix][-1], y[iy][0], y[iy][-1]]
    th_ex = np.angle(np.exp(1j * fld["theta"]))
    d = np.angle(np.exp(1j * (mg["theta"] - fld["theta"])))
    dm = np.angle(np.exp(1j * (-mg["theta"] - fld["theta"])))
    d = np.where(np.abs(d) <= np.abs(dm), d, dm)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    axs[0].imshow(th_ex[np.ix_(iy, ix)], origin="lower", extent=ext,
                  cmap="twilight")
    axs[0].set_title("exact theta (wrapped)")
    axs[1].imshow(mg["theta"][np.ix_(iy, ix)], origin="lower", extent=ext,
                  cmap="twilight")
    axs[1].set_title("arg z, monogenic-gradient")
    im = axs[2].imshow(d[np.ix_(iy, ix)], origin="lower", extent=ext,
                       cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    axs[2].set_title("wrapped difference (rad)")
    fig.colorbar(im, ax=axs[2], shrink=0.8)
    fig.suptitle("%s, mu = %.2f" % (fld["field"], fld["mu"]))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------

def build_sources(args):
    src = {}
    for sd in args.sigma_dir:
        src["mg_%.2g" % sd] = ("mg", sd)
    if args.with_mg_z:
        src["mgz_%.2g" % args.sigma_dir[0]] = ("mgz", args.sigma_dir[0])
    src["demod"] = ("demod", None)
    for sa in args.riesz_sigma_avg:
        src["riesz_%.2g/%s" % (args.riesz_sigma_dir,
                               "pi" if abs(sa - np.pi) < 1e-9 else
                               ("pi2" if abs(sa - np.pi / 2) < 1e-9 else "%.2g" % sa))] = \
            ("riesz", sa)
    return src


def run_source(kind, par, fld, args):
    u, x, y, Lx, Ly = fld["u"], fld["x"], fld["y"], fld["Lx"], fld["Ly"]
    if kind == "mg":
        return mg_estimator(u, Lx, Ly, par)
    if kind == "mgz":
        return mg_estimator_z(u, Lx, Ly, par)
    if kind == "demod":
        return demod_estimator(u, x, y, Lx, Ly, args.tukey_alpha)
    if kind == "riesz":
        return riesz_reference(u, x, y, Lx, Ly, args.riesz_sigma_dir, par)
    raise ValueError(kind)


def main(args):
    stamp = "sd%s_ta%.2g_win%.2g" % ("-".join("%.2g" % s for s in args.sigma_dir),
                                     args.tukey_alpha, args.win_frac)
    out = Path(args.outdir) / stamp
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as fh:
        json.dump({k: getattr(args, k) for k in vars(args)}, fh, indent=2,
                  default=str)
    sources = build_sources(args)
    edges = {"knee": args.knee_edges, "pn": args.pn_edges}
    far_from = {"knee": args.knee_far, "pn": args.pn_far}

    bins_table, summary, beta_rows = [], [], []
    knee_profiles = {}
    lines = ["mg_estimator_validation v1.0   stamp %s" % stamp, ""]
    t0 = time.time()
    for field in args.fields:
        build = t1c.knee_exact if field == "knee" else t1c.pn_exact
        for mu in args.mu:
            fld = build(mu)
            win, dist = window_and_distance(fld, args.win_frac)
            lines.append("== %s mu %.2f  grid %s  dx %.3f  Lsp %.2f" %
                         (field, mu, fld["u"].shape, fld["x"][1] - fld["x"][0],
                          fld["Lsp"]))
            errs_for_map, ests = {}, {}
            for name, (kind, par) in sources.items():
                est = run_source(kind, par, fld, args)
                ests[name] = est
                rows, err = k_error_bins(fld, est, win, dist, edges[field])
                for r in rows:
                    r.update({"field": field, "mu": mu, "source": name})
                bins_table.extend(rows)
                sc, _ = scalar_metrics(fld, est, win, dist, far_from[field])
                sc.update({"field": field, "mu": mu, "source": name})
                summary.append(sc)
                errs_for_map[name] = err
                lines.append("  %-14s k rel RMS window %.4f  far %.4f  "
                             "|k|<0.5 frac %.4f  phase RMS far %s  amp RMS far %.4f"
                             % (name, sc["k_rel_rms_window"], sc["k_rel_rms_far"],
                                sc["frac_k_below_half"],
                                ("%.4f" % sc["phase_rms_rad_far"]
                                 if np.isfinite(sc["phase_rms_rad_far"]) else "  n/a "),
                                sc["amp_rel_rms_far"]))
                lines.append("     bins: " + "  ".join(
                    "[%.2g,%.2g) %.4f" % (r["lo"], r["hi"], r["rel_rms"]) for r in rows))
                if field == "knee":
                    kb = knee_beta(fld, est, args.band_over_mu)
                    beta_rows.append({"mu": mu, "source": name,
                                      "beta_tanh": kb["beta_tanh"],
                                      "beta_regr": kb["beta_regr"],
                                      "r2_linear": kb["r2_linear"]})
                    knee_profiles.setdefault(mu, {"y": fld["y"], "sources": {}})
                    knee_profiles[mu]["sources"][name] = kb["k2b"]
                    lines.append("     knee beta: tanh %.4f  regr %.4f  R2 %.4f"
                                 % (kb["beta_tanh"], kb["beta_regr"], kb["r2_linear"]))
            # figures per (field, mu)
            mg_name = "mg_%.2g" % args.sigma_dir[0]
            if field == "pn":
                fig_pn_error_maps(fld, {k: errs_for_map[k] for k in
                                        (mg_name, "demod")},
                                  out / ("pn_error_maps_mu%.2f.png" % mu))
                fig_phase_panel(fld, ests[mg_name],
                                out / ("phase_panel_pn_mu%.2f.png" % mu))
            else:
                folded = folded_phase_k(fld["u"], fld["x"], fld["y"],
                                        fld["Lx"], fld["Ly"])
                fig_kzeros_panel(fld, folded, ests[mg_name],
                                 out / ("kzeros_panel_mu%.2f.png" % mu))
                fig_phase_panel(fld, ests[mg_name],
                                out / ("phase_panel_knee_mu%.2f.png" % mu))
                sc_f, _ = scalar_metrics(fld, folded, win, dist, far_from[field])
                lines.append("  %-14s |k|<0.5 frac %.4f  (folded phase, reference"
                             " for the zeros panel)" % ("folded", sc_f["frac_k_below_half"]))
            lines.append("")
        fig_err_vs_distance(field, bins_table, edges[field], args.mu,
                            list(sources), out / ("err_vs_distance_%s.png" % field))
    if knee_profiles:
        fig_knee_profiles(knee_profiles, out / "knee_profiles.png")

    # tables
    def write_csv(rows, path, drop=()):
        if not rows:
            return
        keys = [k for k in rows[0] if k not in drop]
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
    write_csv(bins_table, out / "k_error_bins.csv")
    write_csv(summary, out / "summary.csv")
    write_csv(beta_rows, out / "knee_beta.csv")
    if beta_rows:
        lines.append("== knee beta table (exact beta = 1)")
        lines.append("%6s %-14s %9s %9s %8s" % ("mu", "source", "tanh", "regr", "R2"))
        for r in beta_rows:
            lines.append("%6.2f %-14s %9.4f %9.4f %8.4f" %
                         (r["mu"], r["source"], r["beta_tanh"], r["beta_regr"],
                          r["r2_linear"]))
    lines.append("")
    lines.append("elapsed %.1f s" % (time.time() - t0))
    report = "\n".join(lines)
    (out / "report.txt").write_text(report)
    print(report)
    print("outputs ->", out)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    outdir = str(Path("results") / "mg_estimator_validation")
    fields = ["knee", "pn"]
    mu = [0.4, 0.6, 0.85, 0.90, 0.95]
    sigma_dir = [0.3, 0.4]           # monogenic-gradient director widths
    tukey_alpha = 0.5                # demod row taper (session 7 value)
    riesz_sigma_dir = 0.4
    riesz_sigma_avg = [float(np.pi), float(np.pi / 2)]
    win_frac = 0.3                   # evaluation window, fraction of Lx, Ly
    knee_edges = [0, 1, 2, 4, 8, 20, 50]
    pn_edges = [0, 0.25, 0.5, 1, 2, 4, 8]
    knee_far = 8.0                   # |y| beyond which "far" is scored
    pn_far = 2.0                     # spacings beyond which "far" is scored
    band_over_mu = 3.0               # knee beta band |y| < band_over_mu / mu
    with_mg_z = False                # also score the seam-sensitive form


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default=_Args.outdir)
    p.add_argument("--fields", nargs="+", default=_Args.fields,
                   choices=["knee", "pn"])
    p.add_argument("--mu", type=float, nargs="+", default=_Args.mu)
    p.add_argument("--sigma-dir", type=float, nargs="+", dest="sigma_dir",
                   default=_Args.sigma_dir)
    p.add_argument("--tukey-alpha", type=float, dest="tukey_alpha",
                   default=_Args.tukey_alpha)
    p.add_argument("--riesz-sigma-dir", type=float, dest="riesz_sigma_dir",
                   default=_Args.riesz_sigma_dir)
    p.add_argument("--riesz-sigma-avg", type=float, nargs="+",
                   dest="riesz_sigma_avg", default=_Args.riesz_sigma_avg)
    p.add_argument("--win-frac", type=float, dest="win_frac",
                   default=_Args.win_frac)
    p.add_argument("--band-over-mu", type=float, dest="band_over_mu",
                   default=_Args.band_over_mu)
    p.add_argument("--with-mg-z", action="store_true", dest="with_mg_z",
                   default=_Args.with_mg_z,
                   help="also score the naive spectral-z form (seam-sensitive)")
    cli = p.parse_args()
    cli.knee_edges = _Args.knee_edges
    cli.pn_edges = _Args.pn_edges
    cli.knee_far = _Args.knee_far
    cli.pn_far = _Args.pn_far
    sys.exit(main(cli))
