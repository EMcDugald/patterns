# w2_knee_error_budget.py
"""
Error budget of the uHu extraction on the exact single-harmonic knee.

Changelog / purpose
-------------------
Written for Writeup 2, Section 5 (validation). Runs the frozen
extraction (compute_uhu_ops_ramp) on the analytic field
u = cos(theta_KB), theta_KB = k1 x + log(2 cosh(mu y)), and measures:

  (1) plane-wave exactness: far-field kappa error with the lambda1
      guard shift disabled (measured ~1e-7; on a pure uniform roll,
      ~1e-11), certifying the common-mode cancellation of the carrier
      between <u H_u> and <u^2>;
  (2) the guard shift: max lambda1 before the shift, where it is
      attained, and the uniform wavenumber offset
      dkappa = shift / (2 kappa U2) it produces when it fires
      (measured +4.8e-4 shift -> +4.4e-4 k1 offset at the settings
      below);
  (3) the k2 core-band error against the kernel-broadening prediction
      G_sigma * k2 - k2 (1D convolution in y, since k2 depends on y
      only), including the scaling across sigma.

All numbers printed; figure w2_knee_error_budget.png written next to
the script. No simulation data required.

Conventions: frozen method layer imported, not edited; paths anchored
to this script's directory; _Args block at bottom mirrors the CLI.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

from op_extract.uhu_zigzag import compute_uhu_ops_ramp  # noqa: E402
from utils.geometry import build_rectangular_ramp_smooth  # noqa: E402
from utils.spectral import SpectralDerivs, macro  # noqa: E402


def run(args):
    N, L = args.N, args.L
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    dy = y[1] - y[0]
    mu = args.mu
    k1 = np.sqrt(1.0 - mu**2)
    u = np.cos(k1 * X + np.logaddexp(mu * Y, -mu * Y))
    th2_1d = mu * np.tanh(mu * y)
    th_y = mu * np.tanh(mu * Y)
    kap_true = np.sqrt(k1**2 + th_y**2)
    ramp = build_rectangular_ramp_smooth(
        x, y, xmargin=args.xmargin, ymargin=args.ymargin,
        tanhscale=args.tanhscale,
    )

    def tensor_fields(sig):
        Lx = Ly = L
        uxx = SpectralDerivs(u * ramp, Lx, Ly, "xx")
        uxy = SpectralDerivs(u * ramp, Lx, Ly, "xy")
        uyy = SpectralDerivs(u * ramp, Lx, Ly, "yy")
        T = np.moveaxis(
            np.stack(
                [
                    [macro(u * uxx, sig, Lx, Ly), macro(u * uxy, sig, Lx, Ly)],
                    [macro(u * uxy, sig, Lx, Ly), macro(u * uyy, sig, Lx, Ly)],
                ],
                axis=0,
            ),
            (0, 1),
            (-2, -1),
        )
        U2 = macro(u * u * ramp, sig, Lx, Ly)
        ev, evec = np.linalg.eigh(T)
        return ev[..., 0], evec[..., :, 0], U2

    sig = args.sigma
    lam1, e1, U2 = tensor_fields(sig)

    # ---- (2) the guard ----
    iy, ix = np.unravel_index(np.argmax(lam1), lam1.shape)
    shift = max(lam1.max(), 0.0)
    print(f"guard: max lambda1 pre-shift = {lam1.max():+.3e} "
          f"at (x={x[ix]:.1f}, y={y[iy]:.1f}, ramp={ramp[iy, ix]:.3f}); "
          f"fires = {lam1.max() > 0}")
    print(f"guard: predicted uniform kappa offset when applied = "
          f"{shift / (2 * 1.0 * np.median(U2[ramp > 0.9])):+.3e}")

    far = (np.abs(X) < 30) & (np.abs(Y) > 10) & (np.abs(Y) < 30)
    for sh, lab in [(0.0, "guard off"), (shift, "guard on ")]:
        kap = np.sqrt(np.maximum(0.0, -(lam1 - sh) / U2))
        print(f"far-field kappa error ({lab}): "
              f"median {np.median((kap - kap_true)[far]):+.3e}")

    # ---- (3) k2 band vs kernel broadening ----
    def k2_profile(s2):
        l1, ev1, u2 = tensor_fields(s2)
        kap = np.sqrt(np.maximum(0.0, -l1 / u2))
        sgn = np.sign(ev1[..., 0])
        sgn[sgn == 0] = 1
        k2m = kap * ev1[..., 1] * sgn
        cols = np.abs(x) < 28
        return np.mean(k2m[:, cols], axis=1)

    prof = k2_profile(sig)
    pred = gaussian_filter1d(th2_1d, sigma=sig / dy, mode="nearest")
    m = np.abs(y) < 12
    mis = (np.linalg.norm(prof[m] - pred[m])
           / np.linalg.norm((pred - th2_1d)[m]))
    print(f"k2 band: measured max |error| = "
          f"{np.abs(prof[m] - th2_1d[m]).max():.3e}; broadening "
          f"prediction max = {np.abs((pred - th2_1d)[m]).max():.3e}; "
          f"relative L2 mismatch of error profiles = {mis:.3f}")

    sigs = np.array([np.pi / 8, np.pi / 4, 1.0, 1.2, np.pi / 2, 1.8, 2.1])
    meas, prd = [], []
    for s2 in sigs:
        p2 = k2_profile(s2)
        meas.append(np.abs(p2[m] - th2_1d[m]).max())
        prd.append(np.abs(gaussian_filter1d(
            th2_1d, sigma=s2 / dy, mode="nearest")[m] - th2_1d[m]).max())
        print(f"  sigma = {s2:.3f}: measured {meas[-1]:.3e}, "
              f"predicted {prd[-1]:.3e}")

    # ---- figure ----
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.4),
                            constrained_layout=True)
    ax = axs[0]
    ax.plot(y[m], th2_1d[m], "k-", lw=1.4,
            label=r"exact $k_2=\mu\tanh(\mu y)$")
    ax.plot(y[m], pred[m], "C1--", lw=1.4, label=r"$G_\sigma * k_2$")
    ax.plot(y[m], prof[m], "C0-", lw=1.0, alpha=.85,
            label=r"extracted $k_2$")
    ax.set_xlabel("$y$")
    ax.set_ylabel("$k_2$")
    ax.legend(fontsize=8)
    ax.set_title(rf"core profile, $\mu={args.mu}$, $\sigma=\pi/2$")
    ax = axs[1]
    ax.plot(y[m], (prof - th2_1d)[m], "C0-", lw=1.2, label="measured error")
    ax.plot(y[m], (pred - th2_1d)[m], "C1--", lw=1.2,
            label=r"$G_\sigma * k_2 - k_2$")
    ax.axhline(0, color="k", lw=.5)
    ax.set_xlabel("$y$")
    ax.set_ylabel(r"$k_2$ error")
    ax.legend(fontsize=8)
    ax.set_title("error vs kernel-broadening prediction")
    ax = axs[2]
    ax.plot(sigs, meas, "C0o-", label="measured max error")
    ax.plot(sigs, prd, "C1s--", label="broadening prediction")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"max $|k_2$ error$|$")
    ax.legend(fontsize=8)
    ax.set_title("scaling with kernel width")
    out = _HERE / "w2_knee_error_budget.png"
    fig.savefig(out, dpi=220)
    print(f"figure -> {out}")


if __name__ == "__main__":
    import argparse

    if len(sys.argv) == 1:
        class _Args:
            mu = 0.4
            sigma = np.pi / 2
            N = 512
            L = 80.0
            xmargin = 0.03
            ymargin = 0.03
            tanhscale = 120.0
        run(_Args)
    else:
        p = argparse.ArgumentParser()
        p.add_argument("--mu", type=float, default=0.4)
        p.add_argument("--sigma", type=float, default=np.pi / 2)
        p.add_argument("--N", type=int, default=512)
        p.add_argument("--L", type=float, default=80.0)
        p.add_argument("--xmargin", type=float, default=0.03)
        p.add_argument("--ymargin", type=float, default=0.03)
        p.add_argument("--tanhscale", type=float, default=120.0)
        run(p.parse_args())
