# pipelines/harmonic_extraction_test.py
"""
Validation battery for the moment-ladder harmonic extraction
(K, A1, A3 from M0, M1, M2) on synthetic stripe fields with KNOWN
harmonic content. Theory-agnostic: the script measures round-trip
recovery error; interpretation stays in the prose.

Tests
-----
uniform   : uniform roll, amplitudes from the exact truncated stationary
            SH Galerkin solve at (R, K). Realistic harmonic content
            (small A3 near K = 1 -- L_3 = R - (1-9K)^2 ~ -64 slaves the
            third harmonic hard).
boosted   : same roll but A3 forced to a prescribed fraction of A1.
            Unphysical for SH; a pure stress test of the de-mixing and
            of the predicted naive-uHu bias k^2_uHu/K = (s1+9 s3)/(s1+s3).
modulated : slowly varying wavenumber k(x) = k0 + dk sin(2 pi x / Lx),
            theta = integral k dx (periodic by construction), amplitudes
            slaved pointwise via the hierarchical recurrence. Tests
            recovery of spatially varying K(x), A1(x), A3(x).

Each test computes the correlators with the SAME global Gaussian macro()
smoothing used in the production uHu pipeline, so any bias seen here is
what the pipeline would see.

Outputs (per-run folder):
  <out>/summary.txt              numbers for all tests
  <out>/uniform_table.png        bar chart: truth vs naive vs de-mixed
  <out>/boosted_profiles.png     recovered vs true K, A1, A3
  <out>/modulated_profiles.png   x-profiles of recovered vs true fields

Usage
-----
python harmonic_extraction_test.py --out results/harmonic_extraction
(or run in PyCharm; edit _Args at the bottom)
"""

import argparse
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
    from utils.harmonic_moments_sympy import (
        solve_moments_13, sh_roll_amplitudes, uhu_bias_factor)
except ImportError:
    from harmonic_moments_sympy import (
        solve_moments_13, sh_roll_amplitudes, uhu_bias_factor)

try:
    from paper_style import use_paper_style
    use_paper_style()
except ImportError:
    pass


# -----------------------------------------------------------------------
# Correlator computation (mirrors the production pipeline's smoothing)
# -----------------------------------------------------------------------

def moment_fields(u, Lx, Ly, sigma):
    """M0 = <u^2>, M1 = <u Lap u>, M2 = <(Lap u)^2> with macro() smoothing."""
    lap = (SpectralDerivs(u, Lx, Ly, "xx") + SpectralDerivs(u, Lx, Ly, "yy"))
    M0 = macro(u * u, sigma, Lx, Ly)
    M1 = macro(u * lap, sigma, Lx, Ly)
    M2 = macro(lap * lap, sigma, Lx, Ly)
    return M0, M1, M2


def naive_uhu_ksq(M0, M1):
    """Single-harmonic uHu wavenumber estimate: k^2 = -M1 / M0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.maximum(-M1 / M0, 0.0)


# -----------------------------------------------------------------------
# Synthetic fields
# -----------------------------------------------------------------------

def make_grid(Nx, Ny, Lx, Ly):
    x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y


def commensurate_k(k_target, Lx):
    """Nearest wavenumber with an integer number of periods in Lx."""
    n = max(1, int(round(k_target * Lx / (2 * np.pi))))
    return 2 * np.pi * n / Lx


def uniform_roll(X, amps, k, harmonics=(1, 3, 5)):
    theta = k * X
    u = np.zeros_like(X)
    for a, n in zip(amps, harmonics):
        u += a * np.cos(n * theta)
    return u


def modulated_roll(x, X, R, k0, dk, Lx, harmonics=(1, 3, 5)):
    """
    theta(x) = k0 x + (dk Lx / 2 pi)(1 - cos(2 pi x / Lx)) so that
    k(x) = d theta/dx = k0 + dk sin(2 pi x / Lx), periodic in x.
    Amplitudes slaved pointwise: A_n(x) from the truncated Galerkin
    solve at local K(x).
    """
    q = 2 * np.pi / Lx
    kx_local = k0 + dk * np.sin(q * x)
    theta_1d = k0 * (x - x[0]) + (dk / q) * (np.cos(q * x[0]) - np.cos(q * x))

    A_local = np.array([sh_roll_amplitudes(R, kk**2, harmonics)
                        for kk in kx_local])          # (Nx, P)
    u = np.zeros((X.shape[0], X.shape[1]))
    for j, n in enumerate(harmonics):
        u += A_local[None, :, j] * np.cos(n * theta_1d)[None, :]
    K_true = kx_local**2
    return u, K_true, A_local, theta_1d


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

def interior_stats(f, frac=0.25):
    """Median over the central region (avoids any residual edge effects)."""
    ny, nx = f.shape
    sl = (slice(int(frac * ny), int((1 - frac) * ny)),
          slice(int(frac * nx), int((1 - frac) * nx)))
    return float(np.nanmedian(f[sl]))


def test_uniform(cfg, log):
    x, y, X, Y = make_grid(cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly)
    k = commensurate_k(cfg.k_target, cfg.Lx)
    K_true = k**2
    amps = sh_roll_amplitudes(cfg.R, K_true, cfg.harmonics)
    u = uniform_roll(X, amps, k, cfg.harmonics)

    M0, M1, M2 = moment_fields(u, cfg.Lx, cfg.Ly, cfg.sigma)
    K_naive = interior_stats(naive_uhu_ksq(M0, M1))
    K13, A1, A3 = solve_moments_13(M0, M1, M2)

    bias_pred = uhu_bias_factor(amps, cfg.harmonics)
    log(f"[uniform]  R={cfg.R}  k={k:.6f}  K_true={K_true:.6f}")
    log(f"  truth      A = {np.array2string(amps, precision=6)}")
    log(f"  predicted naive bias K_uhu/K = {bias_pred:.8f}")
    log(f"  measured  naive K/K_true     = {K_naive / K_true:.8f}")
    log(f"  de-mixed  K  = {interior_stats(K13):.8f}  "
        f"(rel err {interior_stats(K13)/K_true - 1:+.2e})")
    log(f"  de-mixed  A1 = {interior_stats(A1):.6f}  (truth {amps[0]:.6f})")
    log(f"  de-mixed  A3 = {interior_stats(A3):.6f}  (truth {abs(amps[1]):.6f})")
    return dict(K_true=K_true, amps=amps, K_naive=K_naive,
                K13=interior_stats(K13), A1=interior_stats(A1),
                A3=interior_stats(A3), bias_pred=bias_pred)


def test_boosted(cfg, log, out_dir):
    x, y, X, Y = make_grid(cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly)
    k = commensurate_k(cfg.k_target, cfg.Lx)
    K_true = k**2
    base = sh_roll_amplitudes(cfg.R, K_true, cfg.harmonics)
    amps = base.copy()
    amps[1] = -cfg.boost_frac * abs(base[0])   # keep SH sign convention
    u = uniform_roll(X, amps, k, cfg.harmonics)

    M0, M1, M2 = moment_fields(u, cfg.Lx, cfg.Ly, cfg.sigma)
    Kn = naive_uhu_ksq(M0, M1)
    K13, A1, A3 = solve_moments_13(M0, M1, M2)

    bias_pred = uhu_bias_factor(amps, cfg.harmonics)
    log(f"[boosted]  A3/A1 = {cfg.boost_frac}  K_true={K_true:.6f}")
    log(f"  predicted naive bias         = {bias_pred:.6f} "
        f"(k error {np.sqrt(bias_pred)-1:+.3%})")
    log(f"  measured  naive K/K_true     = {interior_stats(Kn)/K_true:.6f}")
    log(f"  de-mixed  K/K_true           = {interior_stats(K13)/K_true:.6f}")
    log(f"  de-mixed  A1 = {interior_stats(A1):.6f} (truth {abs(amps[0]):.6f})")
    log(f"  de-mixed  A3 = {interior_stats(A3):.6f} (truth {abs(amps[1]):.6f})")

    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    labels = ["$K$", "$A_1$", "$A_3$"]
    truth = [K_true, abs(amps[0]), abs(amps[1])]
    naive = [interior_stats(Kn), np.sqrt(2 * interior_stats(M0)), 0.0]
    demix = [interior_stats(K13), interior_stats(A1), interior_stats(A3)]
    xpos = np.arange(3)
    w = 0.26
    ax.bar(xpos - w, truth, w, label="truth", color="0.25")
    ax.bar(xpos, naive, w, label="naive uHu", color="tab:red", alpha=0.8)
    ax.bar(xpos + w, demix, w, label="moment ladder", color="tab:blue", alpha=0.8)
    ax.set_xticks(xpos, labels)
    ax.legend()
    ax.set_title(
        f"Boosted-harmonic roll ($A_3/A_1={cfg.boost_frac}$): the moment ladder\n"
        f"removes the naive uHu wavenumber bias predicted by "
        f"$(s_1+9s_3)/(s_1+s_3)$.", fontsize=9)
    fig.savefig(out_dir / "boosted_profiles.png")
    plt.close(fig)


def test_modulated(cfg, log, out_dir):
    x, y, X, Y = make_grid(cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly)
    k0 = commensurate_k(cfg.k_target, cfg.Lx)
    u, K_true_x, A_true, _ = modulated_roll(
        x, X, cfg.R, k0, cfg.dk, cfg.Lx, cfg.harmonics)

    M0, M1, M2 = moment_fields(u, cfg.Lx, cfg.Ly, cfg.sigma)
    Kn = naive_uhu_ksq(M0, M1)
    K13, A1, A3 = solve_moments_13(M0, M1, M2)

    row = cfg.Ny // 2
    err_K = np.nanmedian(np.abs(K13[row] - K_true_x) / K_true_x)
    log(f"[modulated]  k0={k0:.4f}  dk={cfg.dk}  "
        f"median |K13 - K_true|/K_true along midrow = {err_K:.3e}")

    fig, axs = plt.subplots(3, 1, figsize=(6.4, 6.6), sharex=True,
                            constrained_layout=True)
    axs[0].plot(x, K_true_x, "k-", lw=1.2, label="truth")
    axs[0].plot(x, Kn[row], "r--", lw=1.0, label="naive uHu")
    axs[0].plot(x, K13[row], "b-", lw=1.0, alpha=0.8, label="moment ladder")
    axs[0].set_ylabel("$K = |k|^2$")
    axs[0].legend(ncol=3)
    axs[1].plot(x, np.abs(A_true[:, 0]), "k-", lw=1.2)
    axs[1].plot(x, A1[row], "b-", lw=1.0, alpha=0.8)
    axs[1].set_ylabel("$A_1$")
    axs[2].plot(x, np.abs(A_true[:, 1]), "k-", lw=1.2)
    axs[2].plot(x, A3[row], "b-", lw=1.0, alpha=0.8)
    axs[2].set_ylabel("$A_3$")
    axs[2].set_xlabel("$x$")
    fig.suptitle(
        "Slowly modulated roll: pointwise moment-ladder recovery of "
        "$K(x)$, $A_1(x)$, $A_3(x)$\nagainst the slaved-harmonic truth "
        "(black); smoothing width $\\sigma=\\pi/2$.", fontsize=9)
    fig.savefig(out_dir / "modulated_profiles.png")
    plt.close(fig)


# -----------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------

def run(cfg):
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s):
        print(s)
        lines.append(s)

    log(f"grid {cfg.Nx}x{cfg.Ny}, Lx={cfg.Lx:.2f}, Ly={cfg.Ly:.2f}, "
        f"sigma={cfg.sigma:.4f}, harmonics={cfg.harmonics}")
    test_uniform(cfg, log)
    test_boosted(cfg, log, out_dir)
    test_modulated(cfg, log, out_dir)

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_dir}/summary.txt and figures")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--out", type=str, default="results/harmonic_extraction_0830")
    p.add_argument("--Nx", type=int, default=512)
    p.add_argument("--Ny", type=int, default=128)
    p.add_argument("--Lx", type=float, default=40 * np.pi)
    p.add_argument("--Ly", type=float, default=10 * np.pi)
    p.add_argument("--R", type=float, default=0.9)
    p.add_argument("--k_target", type=float, default=1.0)
    p.add_argument("--dk", type=float, default=0.08)
    p.add_argument("--sigma", type=float, default=np.pi / 2)
    p.add_argument("--boost_frac", type=float, default=0.3)
    a = p.parse_args()
    a.harmonics = (1, 3, 5)
    run(a)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        class _Args:
            out        = "results/harmonic_extraction_0830"
            Nx, Ny     = 512, 128
            Lx, Ly     = 40 * np.pi, 10 * np.pi
            R          = 0.9
            k_target   = 1.0
            dk         = 0.08
            sigma      = np.pi / 2
            boost_frac = 0.3
            harmonics  = (1, 3, 5)
        run(_Args)
    else:
        main()
