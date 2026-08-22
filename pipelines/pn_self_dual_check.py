# pipelines/pn_self_dual_check.py
"""
Lean, self-contained check of whether the exact Pismen dislocation field obeys
the self-dual balance the knee obeys:

    div k = -c * w ,   w = |k|^2 - 1        (equivalently div k = c (1 - |k|^2))

No extraction, no reconstruction, no time derivatives. Build the analytic PN
phase, take k exactly as its gradient, and fit the one-parameter balance over the
same core-hole domains used in the time runner (holes in units of pi/mu). If c and
R^2 are stable as the core is removed, the balance is a field property; if they
move, it is carried by the core; if it never holds, the PN field does not obey
this self-dual form and that is the finding.

Everything is on the upper half (Y>0) to avoid the sign(Y) seam of the PN phase;
the dislocation cores sit on y=0 and are excluded by the minimal core mask.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy import special
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pismen_phase(X, Y, x_defects):
    """Pismen multi-dislocation phase, upper half safe (Y>0)."""
    x_defects = np.asarray(x_defects, float)
    npts = len(x_defects)
    thetas = np.arange(npts + 1) * np.pi
    psi = 0.5 * (1 + np.exp(-thetas[-1])) * np.ones_like(X)
    for j in range(npts):
        delta = 0.5 * (np.exp(-thetas[j + 1]) - np.exp(-thetas[j]))
        arg = (X - x_defects[j]) / np.sqrt(2 * np.abs(Y) + 1e-9)
        psi += delta * special.erf(arg)
    return Y - np.sign(Y) * np.log(psi)


def fit_balance(divk, w, mask, with_intercept=True):
    """div k = a*w (+ b). Returns c = -a (self-dual coefficient), R^2, rms."""
    d = divk[mask]; ww = w[mask]
    good = np.isfinite(d) & np.isfinite(ww)
    d, ww = d[good], ww[good]
    if d.size < 100:
        return None
    if with_intercept:
        A = np.column_stack([ww, np.ones_like(ww)])
    else:
        A = ww[:, None]
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    pred = A @ coef
    ss = np.sum((d - pred) ** 2)
    tot = max(np.sum((d - d.mean()) ** 2), 1e-30)
    r2 = 1.0 - ss / tot
    a = float(coef[0]); b = float(coef[1]) if with_intercept else 0.0
    return dict(c=-a, a=a, b=b, r2=float(r2),
                rms=float(np.sqrt(ss / d.size)), n=int(d.size))


def run(mu=0.85, n_def=4, ppw=48, hole_units=(0, 1, 2, 3),
        core_min_frac=0.15, out="results/pn_self_dual_check"):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    k1c, k2c = np.sqrt(1 - mu ** 2), mu
    lam_x, lam_y = 2 * np.pi / k1c, 2 * np.pi / k2c
    hpx = np.pi / mu                      # pi / sin(alpha)

    dx = 2 * np.pi / ppw
    Lx = 4 * lam_x
    y_top = 4.0 * lam_y
    x = np.arange(-Lx / 2, Lx / 2, dx)
    y = np.arange(dx, y_top, dx)          # upper half only, start just above 0
    X, Y = np.meshgrid(x, y)
    xd = np.linspace(-0.3 * Lx, 0.3 * Lx, n_def)

    th = pismen_phase(X, Y, xd)
    k1 = np.gradient(th, dx, axis=1)
    k2 = np.gradient(th, dx, axis=0)
    divk = np.gradient(k1, dx, axis=1) + np.gradient(k2, dx, axis=0)
    w = k1 ** 2 + k2 ** 2 - 1.0

    # trim gradient-boundary rows/cols
    edge = np.ones_like(th, bool)
    edge[:2, :] = edge[-2:, :] = edge[:, :2] = edge[:, -2:] = False
    core_min = core_min_frac * hpx        # always exclude the immediate seam/cores

    print(f"PN self-dual check   mu={mu}  n_def={n_def}  pi/mu={hpx:.3f}  "
          f"grid {X.shape}")
    print(f"  testing  div k = -c*w  (w=|k|^2-1)   [knee reference c ~ O(1)]")
    print(f"  {'hole (pi/mu)':>12} {'y_min':>7} {'c=-a':>9} {'intercept':>10} "
          f"{'R^2':>8} {'rms':>9} {'N':>9}")

    rows = []
    for hu in hole_units:
        ymin = max(core_min, hu * hpx)
        mask = edge & (Y >= ymin) & (Y <= y_top - 0.05 * lam_y)
        r = fit_balance(divk, w, mask)
        if r is None:
            print(f"  {hu:12g} {ymin:7.2f}   (too few points)")
            continue
        print(f"  {hu:12g} {ymin:7.2f} {r['c']:9.4f} {r['b']:10.4f} "
              f"{r['r2']:8.4f} {r['rms']:9.4f} {r['n']:9d}")
        rows.append((hu, ymin, r))

    # figure: field triptych + balance scatter at hole 0
    m0 = edge & (Y >= core_min) & (Y <= y_top - 0.05 * lam_y)
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    ext = [x[0], x[-1], y[0], y[-1]]
    disp = lambda f: np.where(m0, f, np.nan)
    for a, f, t, cm in [(ax[0], disp(divk), "div k", "RdBu_r"),
                        (ax[1], disp(w), "w = |k|^2-1", "RdBu_r"),
                        (ax[2], disp(divk - (rows[0][2]['a'] * w + rows[0][2]['b'])),
                         "residual (hole 0 fit)", "RdBu_r")]:
        v = np.nanpercentile(np.abs(f[np.isfinite(f)]), 98) if np.isfinite(f).any() else 1
        im = a.imshow(f, extent=ext, origin="lower", cmap=cm, vmin=-v, vmax=v,
                      aspect="auto"); a.set_title(t); fig.colorbar(im, ax=a, shrink=0.8)
        a.set_xlabel("x")
    ww = w[m0]; dd = divk[m0]; g = np.isfinite(ww) & np.isfinite(dd)
    sub = np.random.default_rng(0).choice(np.where(g)[0], min(5000, g.sum()),
                                          replace=False)
    ax[3].scatter(ww[sub], dd[sub], s=2, alpha=0.3)
    wl = np.linspace(np.nanpercentile(ww, 1), np.nanpercentile(ww, 99), 50)
    ax[3].plot(wl, rows[0][2]['a'] * wl + rows[0][2]['b'], "r-",
               label=f"c={rows[0][2]['c']:.3f}, R²={rows[0][2]['r2']:.3f}")
    ax[3].set_xlabel("w"); ax[3].set_ylabel("div k")
    ax[3].set_title("balance: div k vs w"); ax[3].legend(fontsize=8)
    fig.suptitle(f"PN self-dual balance  div k = -c w  (mu={mu})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = out / f"pn_self_dual_mu{mu:.2f}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  figure -> {p}")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mu", type=float, default=0.95)
    p.add_argument("--n_def", type=int, default=4)
    p.add_argument("--ppw", type=int, default=48)
    a = p.parse_args()
    run(mu=a.mu, n_def=a.n_def, ppw=a.ppw)
