# pipelines/knee_width_safety.py
"""
Permissible Riesz averaging widths on the knee, as a function of mu.

For each mu, a knee pattern is built on a fixed box and the defect
index (plaquette winding of the doubled smoothed Riesz director angle,
see t1_common_v5.director_winding) is evaluated over a sweep of
averaging widths. A width is PERMISSIBLE when the index is exactly zero
in the core band: the knee is defect free, so any nonzero charge is the
fold being misread as a disclination dipole. The output figure shades
the permissible region in the (mu, sigma) plane and overlays the
detection width used by riesz_auto.

The knee here is the bare pattern u = cos(k1 x + log 2 cosh(k2 y)) on a
fixed box, independent of the harness sizing budget (which grows the
box as mu changes and would confound the sweep with resolution
changes). The band is kept away from the box edges because the pattern
is not x-periodic and the spectral Riesz transform wraps there.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

from op_extract.t1_common import director_winding
from utils.field_diagnostics_viz import riesz_director


def knee_pattern(mu, L=24 * np.pi, N=768):
    k1, k2 = np.sqrt(1.0 - mu**2), mu
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    y = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y = np.meshgrid(x, y)
    u = np.cos(k1 * X + np.logaddexp(k2 * Y, -k2 * Y))
    return u, x, y


def charge_count(u, x, y, L, sigma, band_y, band_frac_x):
    rd = riesz_director(u, L, L, sigma)
    q = director_winding(rd["angle"])
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    Xc, Yc = np.meshgrid(xc, yc)
    band = (np.abs(Yc) < band_y) & (np.abs(Xc) < band_frac_x * L / 2)
    return int(np.abs(q[band]).sum())


def run(args):
    mus = np.asarray(args.mus, dtype=float)
    sigmas = np.asarray(args.sigmas, dtype=float)
    counts = np.zeros((len(mus), len(sigmas)), dtype=int)
    for i, mu in enumerate(mus):
        u, x, y = knee_pattern(mu, L=args.L, N=args.N)
        for j, sg in enumerate(sigmas):
            counts[i, j] = charge_count(u, x, y, args.L, sg,
                                        args.band_y, args.band_frac_x)
        clean = sigmas[counts[i] == 0]
        top = f"{clean.max():.2f}" if clean.size else "none"
        print(f"mu={mu:.2f}: clean widths up to {top}"
              f"   counts: {counts[i].tolist()}")

    # bounds of the CONTAMINATION window per mu (min and max dirty
    # width). This is the continuous, interpretable object: everything
    # below the window and everything above it preserves zero index.
    # Note the window's small-sigma edge is resolution dependent: it
    # holds tight sub-grid dipoles (spacing pi/k1, separation well
    # under a coarse grid cell) that dx of 0.3 to 0.5, as in the SH
    # data, cannot express.
    dirty_lo = np.full(len(mus), np.nan)
    dirty_hi = np.full(len(mus), np.nan)
    for i in range(len(mus)):
        bad = np.flatnonzero(counts[i] > 0)
        if bad.size:
            dirty_lo[i] = sigmas[bad[0]]
            dirty_hi[i] = sigmas[bad[-1]]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    P = (counts == 0).astype(float)
    ax.pcolormesh(mus, sigmas, P.T, cmap="RdYlGn", vmin=0, vmax=1,
                  alpha=0.55, shading="nearest")
    ax.plot(mus, dirty_hi, "k.-", lw=1.6, ms=7,
            label="contamination window, upper edge")
    ax.plot(mus, dirty_lo, "k.--", lw=1.2, ms=5,
            label="contamination window, lower edge (dx dependent)")
    ax.axhline(0.4, color="C0", ls="--", lw=1.4,
               label=r"riesz_auto detection width $\sigma_{det}=0.4$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"Riesz averaging width $\sigma$")
    dx = args.L / args.N
    ax.set_title("knee: widths preserving zero topological index "
                 f"(dx = {dx:.2f})\n"
                 "green: index 0 in core band; red: fake dipoles "
                 "at spacing $\\pi/k_1$; lower edge dx dependent")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"figure -> {args.out}")
    np.savez(args.out.replace(".png", ".npz"), mus=mus, sigmas=sigmas,
             counts=counts, dirty_lo=dirty_lo, dirty_hi=dirty_hi)
    return counts, dirty_lo, dirty_hi


class _Args:
    mus = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.84, 0.88,
           0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    sigmas = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00,
              1.25, 1.5708, 2.00, 2.50, 3.1416, 4.00]
    L = 24 * np.pi
    N = 768
    band_y = 2.0
    band_frac_x = 0.70
    out = "knee_width_safety.png"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mus", nargs="+", type=float, default=_Args.mus)
    p.add_argument("--sigmas", nargs="+", type=float, default=_Args.sigmas)
    p.add_argument("--L", type=float, default=_Args.L)
    p.add_argument("--N", type=int, default=_Args.N)
    p.add_argument("--band-y", type=float, default=_Args.band_y)
    p.add_argument("--band-frac-x", type=float, default=_Args.band_frac_x)
    p.add_argument("--out", default=_Args.out)
    run(p.parse_args())
