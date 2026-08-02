# pipelines/knee_pgb_extraction_demo.py
"""
Per-configuration diagnostic: k / A1 / A3 extraction on an EXACT
log-sum-exp phase-grain-boundary knee, u = A1(x,y)cos(theta) +
A3(x,y)cos(3 theta), amplitudes evaluated at the local knee wavenumber.

Geometry
--------
theta = log(exp(k1 x + k2 y) + exp(k1 x - k2 y))
      = k1 x + log(2 cosh(k2 y))          (algebraically identical,
                                           evaluated via np.logaddexp for
                                           numerical stability)
k1 = sqrt(1 - mu^2),  k2 = mu             mu = sin(alpha), alpha = the
far-field knee HALF-angle -- this is Ed's "mu" from the SH sims, not the
bifurcation parameter R (those are independent knobs).

Both far-field legs sit EXACTLY on the critical circle |k| = 1 for any
mu (k1^2+k2^2 = 1 identically) -- only their relative angle 2*alpha
changes. Since theta is not periodic in x, the ground-truth wavevector
is taken from the closed-form gradient rather than a numerical/spectral
derivative:
    k1_true(x,y) = k1                          (constant)
    k2_true(x,y) = k2 tanh(k2 y)
    K_true(x,y)  = k1^2 + k2_true^2             (-> k1^2 at the core,
                                                    -> 1 far-field)

Amplitudes (LEADING-ORDER hierarchical closure -- this is the simplified
form Ed asked for, dropping the 6*A1^2 feedback term that the full
harmonic_moments_sympy.hierarchical_recurrence keeps; pass --full_A3 to
compare against that exact closed form instead):
    A1 = sqrt(4/3 (R - (1-K)^2))
    A3 = const * A1^3 / (4 (R - (9K-1)^2))
R and const are free knobs, independent of mu. A1_sq is clipped at 0
(roll-existence bound R > (1-K)^2) and the violated-area fraction is
reported per config.

For each (mu, R, const) triple in cfg.configs, builds the field, runs
BOTH the ordinary uHu extraction and the {1,3} moment-ladder de-mixing
(compute_uhu_ops_smoother(..., with_moments=True)), and saves one
dashboard figure per config with: input pattern, actual/extracted |k|
(+ error), actual/extracted A1 (+ error), actual/extracted A3 (+ error),
plus the ordinary-uHu |k| and its error for side-by-side comparison.

Usage
-----
python knee_pgb_extraction_demo.py --out results/knee_pgb_demo
(or run in PyCharm; edit _Args at the bottom)
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "src", _HERE / "src"):
    if _p.exists():
        sys.path.insert(0, str(_p))

# --- pipeline imports with standalone fallbacks -------------------------
try:
    from utils.spectral import macro, SpectralDerivs
except ImportError:
    from numpy.fft import fft2, ifft2, fftfreq

    def SpectralDerivs(u, Lx, Ly, type):
        ny, nx = u.shape
        kx = 2 * np.pi * fftfreq(nx, d=Lx / nx)
        ky = 2 * np.pi * fftfreq(ny, d=Ly / ny)
        KX, KY = np.meshgrid(kx, ky, indexing="xy")
        Uhat = fft2(u)
        dU = {"x": 1j * KX, "y": 1j * KY, "xx": -KX**2,
              "yy": -KY**2, "xy": -KX * KY}[type] * Uhat
        return np.real(ifft2(dU))

    def macro(arr, sigma, Lx, Ly):
        M, N = np.shape(arr)
        kx = (np.pi / (0.5 * Lx)) * fftfreq(N, d=1 / N)
        ky = (np.pi / (0.5 * Ly)) * fftfreq(M, d=1 / M)
        xi, eta = np.meshgrid(kx, ky)
        kernel = np.exp(-0.5 * sigma**2 * (xi**2 + eta**2))
        return np.real(ifft2(kernel * fft2(arr)))

try:
    from utils.geometry import build_rectangular_ramp_smooth
except ImportError:
    def build_rectangular_ramp_smooth(x, y, xmargin=0.10, ymargin=0.10,
                                      tanhscale=7.5):
        X, Y = np.meshgrid(x, y)
        Lx, Ly = x[-1] - x[0], y[-1] - y[0]
        xm, ym = xmargin * Lx, ymargin * Ly
        rx = 0.5 * (1 + np.tanh((np.minimum(X - x[0], x[-1] - X) - xm)
                                * tanhscale / Lx))
        ry = 0.5 * (1 + np.tanh((np.minimum(Y - y[0], y[-1] - Y) - ym)
                                * tanhscale / Ly))
        return rx * ry

try:
    from op_extract.uhu_smoother import compute_uhu_ops_smoother
except ImportError:
    from uhu_smoother import compute_uhu_ops_smoother

try:
    from utils.harmonic_moments_sympy import hierarchical_recurrence, Ksym, Rsym
except ImportError:
    from harmonic_moments_sympy import hierarchical_recurrence, Ksym, Rsym

try:
    from paper_style import use_paper_style, add_cbar
    use_paper_style()
except ImportError:
    add_cbar = None


# -----------------------------------------------------------------------
# Full-feedback A3 lambda (from the sympy module), lazily built once
# -----------------------------------------------------------------------

_FULL_A3_FN = None


def _full_A3_fn():
    global _FULL_A3_FN
    if _FULL_A3_FN is None:
        import sympy as sp
        rec = hierarchical_recurrence((1, 3))
        A1sym = sp.Symbol("A1", real=True)
        _FULL_A3_FN = sp.lambdify([A1sym, Ksym, Rsym], rec["A3"], "numpy")
    return _FULL_A3_FN


# -----------------------------------------------------------------------
# Grid / field construction
# -----------------------------------------------------------------------

def make_grid(Nx, Ny, Lx, Ly):
    x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y


def field_pgb_knee(X, Y, mu, R, const, full_A3=False):
    """
    theta = log(exp(k1 x+k2 y)+exp(k1 x-k2 y)), k1=sqrt(1-mu^2), k2=mu.
    Evaluated via np.logaddexp for stability; algebraically identical to
    the literal log-sum-exp form (= k1 x + log(2 cosh(k2 y))).
    """
    k1 = np.sqrt(1.0 - mu**2)
    k2 = mu
    theta = np.logaddexp(k1 * X + k2 * Y, k1 * X - k2 * Y)

    k2_true = k2 * np.tanh(k2 * Y)
    K_true = k1**2 + k2_true**2

    A1_sq = (4.0 / 3.0) * (R - (1.0 - K_true) ** 2)
    frac_invalid = float(np.mean(A1_sq < 0))
    A1_true = np.sqrt(np.clip(A1_sq, 0.0, None))

    if full_A3:
        A3_true = _full_A3_fn()(A1_true, K_true, R)
    else:
        denom = 4.0 * (R - (9.0 * K_true - 1.0) ** 2)
        A3_true = const * A1_true**3 / denom

    u = A1_true * np.cos(theta) + A3_true * np.cos(3 * theta)
    return dict(u=u, theta=theta, k1=k1, k2_true=k2_true, K_true=K_true,
               A1_true=A1_true, A3_true=A3_true, frac_invalid=frac_invalid)


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def summarize(err, mask):
    v = err[mask]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan
    return float(np.median(v)), float(np.percentile(v, 95))


# -----------------------------------------------------------------------
# Dashboard figure
# -----------------------------------------------------------------------

def make_dashboard(out_path, extent, field, res, k_true, k_ord, k_dem,
                   A1_dem, A3_dem, mask, label):
    def masked(arr):
        return np.ma.masked_where(~mask, arr)

    A1_true, A3_true = field["A1_true"], field["A3_true"]
    A3_true_abs = np.abs(A3_true)

    panels = [
        (field["u"], "input pattern u", None),
        (masked(k_true), "actual |k|", None),
        (masked(k_dem), "extracted |k| (de-mixed {1,3})", None),
        (masked(np.abs(k_dem - k_true)), "|k| error (de-mixed)", "magma"),
        (masked(A1_true), "actual A1", None),
        (masked(A1_dem), "extracted A1 (de-mixed)", None),
        (masked(np.abs(A1_dem - A1_true)), "A1 error", "magma"),
        (masked(A3_true_abs), "actual |A3|", None),
        (masked(A3_dem), "extracted A3 (de-mixed)", None),
        (masked(np.abs(A3_dem - A3_true_abs)), "A3 error", "magma"),
        (masked(k_ord), "extracted |k| (ordinary uHu)", None),
        (masked(np.abs(k_ord - k_true)), "|k| error (ordinary uHu)", "magma"),
    ]

    fig, axs = plt.subplots(4, 3, figsize=(11.5, 13.5),
                            constrained_layout=True)
    for ax, (arr, title, cmap) in zip(axs.flat, panels):
        cmap = cmap or "viridis"
        im = ax.imshow(arr, extent=extent, origin="lower", aspect="auto",
                       cmap=cmap, rasterized=True)
        ax.set_title(title, fontsize=8.5)
        if add_cbar is not None:
            add_cbar(fig, ax, im)
        else:
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(label, fontsize=10)
    fig.savefig(out_path)
    plt.close(fig)


# -----------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------

def run(cfg):
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y, X, Y = make_grid(cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly)
    Lx, Ly = cfg.Lx, cfg.Ly
    ramp = build_rectangular_ramp_smooth(
        x, y, xmargin=cfg.xmargin, ymargin=cfg.ymargin,
        tanhscale=cfg.tanhscale)
    #interior = ramp >= 0.99
    interior = ramp >= 0.999999
    extent = [x[0], x[-1], y[0], y[-1]]

    smoother = lambda arr: macro(arr, cfg.sigma, Lx, Ly)

    print("=== amplitude formulas used in this script ===")
    print("  A1 = sqrt(4/3 (R - (1-K)^2))")
    if cfg.full_A3:
        print("  A3 = exact hierarchical closed form (6 A1^2 feedback kept)")
        print("       -- see harmonic_moments_sympy.hierarchical_recurrence")
    else:
        print("  A3 = const * A1^3 / (4 (R - (9K-1)^2))"
             "   [leading-order, no A1^2 feedback]")
    print()

    rows = []
    for i, c in enumerate(cfg.configs):
        mu, R, const = c["mu"], c["R"], c["const"]
        label = c.get("label", f"mu={mu:.2f}  R={R:.2f}  const={const:.2f}")

        field = field_pgb_knee(X, Y, mu, R, const, full_A3=cfg.full_A3)
        u = field["u"]

        res = compute_uhu_ops_smoother(
            u, x, y, smoother, SpectralDerivs, ramp, with_moments=True)

        k_true = np.sqrt(field["K_true"])
        k_ord = res["k"]
        k_dem = np.sqrt(np.clip(res["K13"], 0.0, None))
        A1_dem = res["A1_13"]
        A3_dem = res["A3_13"]

        ko_m, ko_p = summarize(np.abs(k_ord - k_true), interior)
        kd_m, kd_p = summarize(np.abs(k_dem - k_true), interior)
        a1_m, a1_p = summarize(np.abs(A1_dem - field["A1_true"]), interior)
        a3_m, a3_p = summarize(np.abs(A3_dem - np.abs(field["A3_true"])),
                               interior)
        frac_nan = float(np.mean(~np.isfinite(res["K13"])[interior]))

        row = dict(label=label, mu=mu, R=R, const=const,
                  k_ord_med=ko_m, k_ord_p95=ko_p,
                  k_dem_med=kd_m, k_dem_p95=kd_p,
                  A1_err_med=a1_m, A1_err_p95=a1_p,
                  A3_err_med=a3_m, A3_err_p95=a3_p,
                  frac_invalid_existence=field["frac_invalid"],
                  frac_demix_nan=frac_nan)
        rows.append(row)

        print(f"[{i}] {label}")
        print(f"     k_ord_med={ko_m:.4e}  k_dem_med={kd_m:.4e}  "
             f"A1_err_med={a1_m:.4e}  A3_err_med={a3_m:.4e}  "
             f"frac_invalid={field['frac_invalid']:.3f}  "
             f"frac_nan={frac_nan:.3f}")

        # fig_path = out_dir / f"dashboard_{i:02d}.png"

        fname = (f"dashboard_{i:02d}_mu{mu:.2f}_R{R:.2f}_"
                 f"const{const:.2f}.png").replace(".", "p", 3)
        fig_path = out_dir / fname
        make_dashboard(fig_path, extent, field, res, k_true, k_ord, k_dem,
                      A1_dem, A3_dem, interior, label)
        print(f"     wrote {fig_path}")

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")
    return rows


def main():
    p = argparse.ArgumentParser(description="PGB knee extraction demo")
    p.add_argument("--out", type=str, default="results/knee_pgb_demo")
    p.add_argument("--Nx", type=int, default=384)
    p.add_argument("--Ny", type=int, default=192)
    p.add_argument("--Lx", type=float, default=40 * np.pi)
    p.add_argument("--Ly", type=float, default=20 * np.pi)
    p.add_argument("--sigma", type=float, default=np.pi / 2)
    p.add_argument("--xmargin", type=float, default=0.05)
    p.add_argument("--ymargin", type=float, default=0.05)
    p.add_argument("--tanhscale", type=float, default=60.0)
    p.add_argument("--full_A3", action="store_true")
    a = p.parse_args()
    a.configs = [
        dict(mu=0.3, R=1.0, const=1.0),
        dict(mu=0.5, R=1.0, const=1.0),
        dict(mu=0.7, R=1.0, const=1.0),
        dict(mu=0.5, R=0.6, const=1.0),
        dict(mu=0.5, R=1.6, const=1.0),
        dict(mu=0.5, R=1.0, const=3.0),
    ]
    run(a)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        class _Args:
            out       = "results/knee_pgb_demo"
            Nx, Ny    = 384, 192
            Lx, Ly    = 40 * np.pi, 20 * np.pi
            sigma     = np.pi / 2
            xmargin   = 0.05
            ymargin   = 0.05
            tanhscale = 60.0
            full_A3   = False
            configs   = [
                dict(mu=0.3, R=1.0, const=1.0, label="small angle, R=1"),
                dict(mu=0.5, R=1.0, const=1.0, label="mid angle, R=1"),
                dict(mu=0.7, R=1.0, const=1.0, label="large angle, R=1"),
                dict(mu=0.5, R=0.6, const=1.0, label="mid angle, low R"),
                dict(mu=0.5, R=1.6, const=1.0, label="mid angle, high R"),
                dict(mu=0.5, R=.5, const=3.0, label="mid angle, strong A3"),
                dict(mu=0.5, R=.5, const=10.0, label="mid angle, stronger A3"),
                dict(mu=0.8, R=0.6, const=1.0, label="mid angle, low R"),
                dict(mu=0.8, R=1.6, const=1.0, label="mid angle, high R"),
                dict(mu=0.8, R=.5, const=3.0, label="mid angle, strong A3"),
                dict(mu=0.8, R=.5, const=10.0, label="mid angle, stronger A3"),
                dict(mu=0.7, R=0.6, const=1.0, label="mid angle, low R"),
                dict(mu=0.7, R=1.6, const=1.0, label="mid angle, high R"),
                dict(mu=0.7, R=.5, const=3.0, label="mid angle, strong A3"),
                dict(mu=0.7, R=.5, const=10.0, label="mid angle, stronger A3"),
                dict(mu=0.9, R=0.6, const=1.0, label="mid angle, low R"),
                dict(mu=0.9, R=1.6, const=1.0, label="mid angle, high R"),
                dict(mu=0.9, R=.5, const=3.0, label="mid angle, strong A3"),
                dict(mu=0.9, R=.5, const=10.0, label="mid angle, stronger A3"),
            ]
        run(_Args)
    else:
        main()
