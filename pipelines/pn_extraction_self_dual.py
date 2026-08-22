# pipelines/pn_extraction_self_dual.py
"""
Does the self-dual balance  div k = -c*w  (w=|k|^2-1) survive EXTRACTION?

Build u = cos(theta_PN) from the exact Pismen phase, then get k three ways:
  exact : analytic gradient of theta_PN            (ground truth)
  riesz : riesz_k_field on u                        (the direct Riesz route)
  uhu   : compute_uhu_ops_smoother on u             (the uHu route)
Form div k and w for each with the SAME finite differencing, so the div stencil
is common-mode and the only thing that differs is the k estimate, and fit the
one-parameter balance over the pi/mu core-hole domains.

Two pitfalls this makes explicit, both found the hard way:
  1. div k is sign-sensitive; the Riesz/uHu director is sign-ambiguous, so the
     extracted k MUST be oriented before div k means anything. We orient against
     the exact gradient (a synthetic gauge). The unoriented Riesz column is shown
     too, so the failure mode is visible.
  2. the balance is a fine-scale pointwise identity; the magnitude averaging the
     extraction needs (sigma_avg ~ pi for |k| accuracy) does not commute with the
     nonlinear |k|^2, so extracted div k vs extracted w can hold far worse than
     the exact fields even when |k| itself is accurate.

Everything on the upper half (Y>0) to avoid the sign(Y) seam.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy import special
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from op_extract.t1_common import (          # noqa: E402
    riesz_k_field, compute_uhu_ops_smoother, macro_iso, spectral_deriv,
)


def pn_phase(X, Y, xd):
    xd = np.asarray(xd, float); n = len(xd); thetas = np.arange(n + 1) * np.pi
    psi = 0.5 * (1 + np.exp(-thetas[-1])) * np.ones_like(X)
    for j in range(n):
        d = 0.5 * (np.exp(-thetas[j + 1]) - np.exp(-thetas[j]))
        psi += d * special.erf((X - xd[j]) / np.sqrt(2 * np.abs(Y) + 1e-9))
    return Y - np.sign(Y) * np.log(psi)


def exact_k(X, Y, xd):
    """Analytic gradient of the upper-half PN phase (Y>0)."""
    xd = np.asarray(xd, float); n = len(xd); thetas = np.arange(n + 1) * np.pi
    psi = 0.5 * (1 + np.exp(-thetas[-1])) * np.ones_like(X)
    px = np.zeros_like(X); py = np.zeros_like(X); s = np.sqrt(2 * Y)
    for j in range(n):
        d = 0.5 * (np.exp(-thetas[j + 1]) - np.exp(-thetas[j]))
        a = (X - xd[j]) / s; g = (2 / np.sqrt(np.pi)) * np.exp(-a ** 2)
        psi += d * special.erf(a)
        px += d * g * (1 / s)
        py += d * g * (-(X - xd[j]) / s ** 3)
    return -px / psi, 1.0 - py / psi


def _ramp(x, frac=0.1, sharp=6.0):
    Lx = x[-1] - x[0]
    return (0.5 * (1 + np.tanh((x - x[0] - frac * Lx) * sharp / Lx))
            * 0.5 * (1 + np.tanh((x[-1] - x - frac * Lx) * sharp / Lx)))


def _orient(k1, k2, k1ref, k2ref):
    s = np.sign(k1 * k1ref + k2 * k2ref)
    s = np.where(s == 0, 1.0, s)
    return k1 * s, k2 * s


def _divw(k1, k2, dx):
    return (np.gradient(k1, dx, axis=1) + np.gradient(k2, dx, axis=0),
            k1 ** 2 + k2 ** 2 - 1.0)


def _fit(divk, w, m):
    d = divk[m]; ww = w[m]; g = np.isfinite(d) & np.isfinite(ww)
    d, ww = d[g], ww[g]
    if d.size < 100:
        return None
    A = np.column_stack([ww, np.ones_like(ww)])
    c, *_ = np.linalg.lstsq(A, d, rcond=None)
    p = A @ c
    r2 = 1.0 - np.sum((d - p) ** 2) / max(np.sum((d - d.mean()) ** 2), 1e-30)
    return dict(c=-float(c[0]), b=float(c[1]), r2=float(r2), n=int(d.size))


def _avg_variants(k1, k2, x, y, sigma):
    """Four averaging treatments of a k field, crossing the smoother (FFT-periodic
    macro_iso vs local gaussian_filter) with the w choice (<k>^2-1 vs <|k|^2>-1).
    div k is always finite-differenced from the smoothed vector, so the only
    differences are the smoother and which quantity forms w."""
    dx = float(x[1] - x[0]); Lxx, Lyy = x[-1] - x[0], y[-1] - y[0]
    out = {}

    kf1, kf2 = macro_iso(k1, sigma, Lxx, Lyy), macro_iso(k2, sigma, Lxx, Lyy)
    ksqf = macro_iso(k1 ** 2 + k2 ** 2, sigma, Lxx, Lyy)
    divf = np.gradient(kf1, dx, axis=1) + np.gradient(kf2, dx, axis=0)
    out["FFT, w=<k>^2-1"] = (divf, kf1 ** 2 + kf2 ** 2 - 1.0)
    out["FFT, w=<|k|^2>-1"] = (divf, ksqf - 1.0)

    sp = sigma / dx
    kl1 = gaussian_filter(k1, sp, mode="nearest")
    kl2 = gaussian_filter(k2, sp, mode="nearest")
    ksql = gaussian_filter(k1 ** 2 + k2 ** 2, sp, mode="nearest")
    divl = np.gradient(kl1, dx, axis=1) + np.gradient(kl2, dx, axis=0)
    out["local, w=<k>^2-1"] = (divl, kl1 ** 2 + kl2 ** 2 - 1.0)
    out["local, w=<|k|^2>-1"] = (divl, ksql - 1.0)
    return out


def averaging_recovery(k1e, k2e, x, y, edge, Y, hpx, y_top, mu,
                       hole_units, sigmas, out=None):
    """On the EXACT k, show how the smoother and the w choice each affect the
    balance, isolating the mechanism from any extraction noise. The recovery is
    'local, w=<|k|^2>-1' matching the no-smoothing control; 'FFT' anything and
    'w=<k>^2-1' are the broken corners. Also writes a figure at the first sigma:
    div k field and the div k vs w scatter for the control and the four corners,
    so the recovery is visible as the scatter collapsing onto the control line."""
    dx = float(x[1] - x[0])
    ymax = y_top - 0.05 * (2 * np.pi / mu)
    dkx = np.gradient(k1e, dx, axis=1) + np.gradient(k2e, dx, axis=0)
    wx = k1e ** 2 + k2e ** 2 - 1.0

    print("\n  ==== averaging mechanism on the EXACT k "
          "(smoother x w-choice) ====")
    print("  control, no smoothing  (div k vs |k|^2 - 1):")
    for hu in hole_units:
        ym = max(0.15 * hpx, hu * hpx); m = edge & (Y >= ym) & (Y <= ymax)
        r = _fit(dkx, wx, m)
        if r:
            print(f"    hole {hu}:  c={r['c']:6.3f}  R2={r['r2']:6.3f}")

    for sigma in sigmas:
        variants = _avg_variants(k1e, k2e, x, y, sigma)
        labels = list(variants)
        print(f"\n  sigma={sigma:.3f}:")
        print("    " + "hole".rjust(5)
              + "".join(lab.rjust(22) for lab in labels))
        for hu in hole_units:
            ym = max(0.15 * hpx, hu * hpx); m = edge & (Y >= ym) & (Y <= ymax)
            cells = []
            for lab in labels:
                dk, w = variants[lab]
                r = _fit(dk, w, m)
                cells.append(f"c={r['c']:5.2f} R2={r['r2']:4.2f}" if r else "-")
            print("    " + f"{hu:g}".rjust(5)
                  + "".join(c.rjust(22) for c in cells))

    if out is None:
        return
    # ---- figure at the first sigma: control + four corners ----
    sigma = sigmas[0]
    variants = _avg_variants(k1e, k2e, x, y, sigma)
    panels = [("control (no smooth)", dkx, wx)]
    panels += [(lab, dk, w) for lab, (dk, w) in variants.items()]
    hu_fig = hole_units[1] if len(hole_units) > 1 else hole_units[0]
    ym = max(0.15 * hpx, hu_fig * hpx); m = edge & (Y >= ym) & (Y <= ymax)
    ext = [x[0], x[-1], y[0], y[-1]]
    rng = np.random.default_rng(0)

    ncol = len(panels)
    fig, ax = plt.subplots(2, ncol, figsize=(3.2 * ncol, 6.2))
    for j, (lab, dk, w) in enumerate(panels):
        disp = np.where(m, dk, np.nan)
        vals = disp[np.isfinite(disp)]
        v = np.nanpercentile(np.abs(vals), 98) if vals.size else 1.0
        ax[0, j].imshow(disp, extent=ext, origin="lower", cmap="RdBu_r",
                        vmin=-v, vmax=v, aspect="auto")
        ax[0, j].set_title(lab, fontsize=8)
        ax[0, j].set_xticks([]); ax[0, j].set_yticks([])
        ww = w[m]; dd = dk[m]; g = np.isfinite(ww) & np.isfinite(dd)
        idx = rng.choice(np.where(g)[0], min(4000, int(g.sum())), replace=False)
        ax[1, j].scatter(ww[idx], dd[idx], s=2, alpha=0.3)
        r = _fit(dk, w, m)
        if r:
            wl = np.linspace(np.nanpercentile(ww, 1), np.nanpercentile(ww, 99), 40)
            ax[1, j].plot(wl, -r["c"] * wl + r["b"], "r-",
                          label=f"c={r['c']:.2f}\nR²={r['r2']:.2f}")
            ax[1, j].legend(fontsize=7, loc="upper left")
        ax[1, j].set_xlabel("w", fontsize=8)
        if j == 0:
            ax[1, j].set_ylabel("div k", fontsize=8)
    fig.suptitle(f"Self-dual balance under averaging (exact k, sigma={sigma:.2f}, "
                 f"hole {hu_fig}): recovery is 'local, w=<|k|^2>-1'", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\n  mechanism figure -> {out}")


def run(mu=0.85, n_def=4, ppw=48, sigma_riesz=0.4, sigma_uhu=np.pi / 2,
        hole_units=(0, 1, 2), out="results/pn_extraction_self_dual"):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    k1c = np.sqrt(1 - mu ** 2); hpx = np.pi / mu
    dx = 2 * np.pi / ppw
    Lx = 4 * (2 * np.pi / k1c); y_top = 4 * (2 * np.pi / mu)
    x = np.arange(-Lx / 2, Lx / 2, dx); y = np.arange(dx, y_top, dx)
    X, Y = np.meshgrid(x, y)
    xd = np.linspace(-0.3 * Lx, 0.3 * Lx, n_def)
    Lxx, Lyy = x[-1] - x[0], y[-1] - y[0]

    th = pn_phase(X, Y, xd); u = np.cos(th)
    ur = u * _ramp(x)[None, :]
    k1e, k2e = exact_k(X, Y, xd)

    # riesz
    rz = riesz_k_field(ur, x, y, Lxx, Lyy, sigma_riesz)
    k1r_raw, k2r_raw = np.asarray(rz["k1"]), np.asarray(rz["k2"])
    k1r, k2r = _orient(k1r_raw, k2r_raw, k1e, k2e)
    # uhu
    ramp2d = np.ones_like(u) * _ramp(x)[None, :]
    sm = lambda arr: macro_iso(arr, sigma_uhu, Lxx, Lyy)
    ops = compute_uhu_ops_smoother(u, x, y, sm, spectral_deriv, ramp2d)
    k1u, k2u = _orient(np.asarray(ops["k1_orig"]), np.asarray(ops["k2_orig"]),
                       k1e, k2e)

    methods = {
        "exact": _divw(k1e, k2e, dx),
        "riesz(orient)": _divw(k1r, k2r, dx),
        "riesz(RAW)": _divw(k1r_raw, k2r_raw, dx),
        "uhu(orient)": _divw(k1u, k2u, dx),
    }

    edge = np.ones_like(u, bool)
    edge[:3] = edge[-3:] = edge[:, :3] = edge[:, -3:] = False
    edge &= (X > x[0] + 0.15 * Lx) & (X < x[-1] - 0.15 * Lx)

    print(f"PN extraction self-dual  mu={mu}  pi/mu={hpx:.3f}  "
          f"sigma_riesz={sigma_riesz} sigma_uhu={sigma_uhu:.3f}")
    print(f"  does div k = -c*w survive extraction? exact should hold, "
          f"extracted may not\n")
    print(f"  {'hole':>5} {'method':>14} {'c':>8} {'R^2':>8} {'N':>9}")
    summary = {}
    for hu in hole_units:
        ymin = max(0.15 * hpx, hu * hpx)
        m = edge & (Y >= ymin) & (Y <= y_top - 0.05 * (2 * np.pi / mu))
        summary[hu] = {}
        for name, (dk, w) in methods.items():
            r = _fit(dk, w, m)
            if r is None:
                continue
            print(f"  {hu:5g} {name:>14} {r['c']:8.3f} {r['r2']:8.3f} "
                  f"{r['n']:9d}")
            summary[hu][name] = r
        print()

    averaging_recovery(k1e, k2e, x, y, edge, Y, hpx, y_top, mu,
                       hole_units, sigmas=(np.pi / 2, np.pi),
                       out=out / f"averaging_mechanism_mu{mu:.2f}.png")

    # figure: div k field per method at hole 0
    m0 = edge & (Y >= 0.15 * hpx) & (Y <= y_top - 0.05 * (2 * np.pi / mu))
    disp = lambda f: np.where(m0, f, np.nan)
    show = ["exact", "riesz(orient)", "uhu(orient)"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 6))
    ext = [x[0], x[-1], y[0], y[-1]]
    for j, nm in enumerate(show):
        dk, w = methods[nm]
        v = np.nanpercentile(np.abs(disp(dk)[np.isfinite(disp(dk))]), 98)
        ax[0, j].imshow(disp(dk), extent=ext, origin="lower", cmap="RdBu_r",
                        vmin=-v, vmax=v, aspect="auto")
        ax[0, j].set_title(f"div k [{nm}]", fontsize=9)
        ww = w[m0]; dd = dk[m0]; g = np.isfinite(ww) & np.isfinite(dd)
        idx = np.random.default_rng(0).choice(np.where(g)[0],
                                              min(4000, g.sum()), replace=False)
        ax[1, j].scatter(ww[idx], dd[idx], s=2, alpha=0.3)
        r = summary[0].get(nm)
        if r:
            wl = np.linspace(np.nanpercentile(ww, 1), np.nanpercentile(ww, 99), 40)
            ax[1, j].plot(wl, -r["c"] * wl + r["b"], "r-",
                          label=f"c={r['c']:.2f} R²={r['r2']:.2f}")
            ax[1, j].legend(fontsize=8)
        ax[1, j].set_xlabel("w"); ax[1, j].set_ylabel("div k")
    fig.suptitle(f"self-dual balance survives extraction?  (mu={mu})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = out / f"pn_extraction_self_dual_mu{mu:.2f}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  figure -> {p}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mu", type=float, default=0.85)
    p.add_argument("--n_def", type=int, default=4)
    p.add_argument("--ppw", type=int, default=48)
    p.add_argument("--sigma_riesz", type=float, default=0.4)
    p.add_argument("--sigma_uhu", type=float, default=float(np.pi / 2))
    a = p.parse_args()
    run(mu=a.mu, n_def=a.n_def, ppw=a.ppw,
        sigma_riesz=a.sigma_riesz, sigma_uhu=a.sigma_uhu)