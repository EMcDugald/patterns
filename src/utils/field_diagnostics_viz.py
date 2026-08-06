# src/utils/field_diagnostics_viz.py
"""
Data-visualization helpers for the adaptive-averaging diagnostics:
monogenic signal components, Riesz director, coherence at several
averaging scales, and the window field W computed from an extracted
k-field. All functions are theory-agnostic (interpretation stays in
prose); each panel helper returns the computed arrays so they can be
reused quantitatively.

Conventions match the existing codebase:
  Riesz:      R_j = F^-1[-i (xi_j/|xi|) F u]      (riesz.riesz_analyze_pattern)
  coherence:  chi = |lam2 / lam1| from the uHu eigenvalues
              (rank-one validity: small chi = trustworthy)
  W:          ||grad grad theta|| / |grad theta|^2 evaluated from the
              EXTRACTED (k1, k2) via spectral derivatives -- needs only
              first derivatives of fields the pipeline already produces,
              no axis convention, geometry-agnostic (zigzag/diamond OK).

Demo (python field_diagnostics_viz.py): knee mu = 0.9 and an 8-defect
PN chain, sigma list (pi/8, pi/2, pi).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from scipy.special import erf as nerf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent      # .../patterns/src/utils
_SRC = _HERE.parent                          # .../patterns/src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from op_extract.uhu_smoother import compute_uhu_ops_smoother



# ---------------------------------------------------------------------
# basics
# ---------------------------------------------------------------------

def spectral_deriv(arr, Lx, Ly, which):
    ny, nx = arr.shape
    kx = 2 * np.pi * fftfreq(nx, d=Lx / nx)
    ky = 2 * np.pi * fftfreq(ny, d=Ly / ny)
    XI, ETA = np.meshgrid(kx, ky)
    op = {"x": 1j * XI, "y": 1j * ETA,
          "xx": -XI**2, "yy": -ETA**2, "xy": -XI * ETA}[which]
    return np.real(ifft2(op * fft2(arr)))


def macro_iso(arr, sigma, Lx, Ly):
    ny, nx = arr.shape
    kx = 2 * np.pi * fftfreq(nx, d=Lx / nx)
    ky = 2 * np.pi * fftfreq(ny, d=Ly / ny)
    XI, ETA = np.meshgrid(kx, ky)
    return np.real(ifft2(np.exp(-0.5 * sigma**2 * (XI**2 + ETA**2)) * fft2(arr)))


# ---------------------------------------------------------------------
# monogenic signal + Riesz director
# ---------------------------------------------------------------------

def monogenic(u, Lx, Ly):
    """R1, R2, monogenic amplitude, monogenic phase, raw orientation.
    Same transfer function as riesz.riesz_analyze_pattern."""
    ny, nx = u.shape
    kx = fftfreq(nx, d=Lx / nx)
    ky = fftfreq(ny, d=Ly / ny)
    KX, KY = np.meshgrid(kx, ky)
    Kn = np.sqrt(KX**2 + KY**2) + 1e-12
    U = fft2(u)
    R1 = np.real(ifft2(-1j * (KX / Kn) * U))
    R2 = np.real(ifft2(-1j * (KY / Kn) * U))
    A = np.sqrt(u**2 + R1**2 + R2**2)
    phase = np.arctan2(np.sqrt(R1**2 + R2**2), u)
    orientation = np.arctan2(R2, R1)
    return {"R1": R1, "R2": R2, "A": A, "phase": phase,
            "orientation": orientation}


def riesz_director(u, Lx, Ly, sigma_dir):
    """Flip-free director from the doubled-angle smoothed Riesz
    orientation (no eigenvalue comparison anywhere). Returns unit
    (d1, d2) and the alignment magnitude in [0, 1] (how coherent the
    doubled-angle average is within sigma_dir)."""
    mono = monogenic(u, Lx, Ly)
    w = mono["R1"]**2 + mono["R2"]**2          # energy weighting
    c2 = macro_iso(w * np.cos(2 * mono["orientation"]), sigma_dir, Lx, Ly)
    s2 = macro_iso(w * np.sin(2 * mono["orientation"]), sigma_dir, Lx, Ly)
    wbar = macro_iso(w, sigma_dir, Lx, Ly)
    align = np.hypot(c2, s2) / np.maximum(wbar, 1e-12)
    ang = 0.5 * np.arctan2(s2, c2)
    return {"d1": np.cos(ang), "d2": np.sin(ang), "angle": ang,
            "alignment": np.clip(align, 0, 1)}


# ---------------------------------------------------------------------
# multiscale coherence + W from extracted k
# ---------------------------------------------------------------------

def coherence_multiscale(u, x, y, sigmas, ramp=None):
    """chi = |lam2/lam1| maps from full uHu extraction at each sigma.
    Returns dict sigma -> dict(chi=..., ops=...)."""
    Lx, Ly = x[-1] - x[0], y[-1] - y[0]
    if ramp is None:
        ramp = np.ones((len(y), len(x)))
    out = {}
    for sg in sigmas:
        sm = lambda arr, s=sg: macro_iso(arr, s, Lx, Ly)
        ops = compute_uhu_ops_smoother(u, x, y, sm, spectral_deriv, ramp)
        with np.errstate(invalid="ignore", divide="ignore"):
            chi = np.abs(ops["lam2"] / np.where(
                np.abs(ops["lam1"]) < 1e-12, np.nan, ops["lam1"]))
        out[sg] = {"chi": chi, "ops": ops}
    return out


def W_from_k(k1, k2, Lx, Ly, floor=1e-8):
    """Window field from an extracted wavevector field: spectral first
    derivatives of (k1, k2) build ||grad k||_F; symmetrized so it reduces
    to ||Hessian(theta)|| when (k1,k2) is an exact gradient."""
    k1x = spectral_deriv(k1, Lx, Ly, "x")
    k1y = spectral_deriv(k1, Lx, Ly, "y")
    k2x = spectral_deriv(k2, Lx, Ly, "x")
    k2y = spectral_deriv(k2, Lx, Ly, "y")
    off = 0.5 * (k1y + k2x)                 # symmetric part; curl handled sep.
    Hn = np.sqrt(k1x**2 + 2 * off**2 + k2y**2)
    return Hn / np.maximum(k1**2 + k2**2, floor)


# ---------------------------------------------------------------------
# panel figure
# ---------------------------------------------------------------------

def diagnostics_figure(u, x, y, sigmas=(np.pi / 8, np.pi / 2, np.pi),
                       sigma_dir=np.pi / 2, title="", quiver_step=24,
                       sigma_extract=np.pi / 2, ramp=None, path=None,
                       window=None):
    """
    Grid of panels: pattern + Riesz-director quiver; monogenic amplitude;
    monogenic orientation; chi at each sigma; W from the sigma_extract
    k-field. `window` = (x0, x1, y0, y1) crops the display (not the
    computation). Returns dict of all computed fields.
    """
    Lx, Ly = x[-1] - x[0], y[-1] - y[0]
    mono = monogenic(u, Lx, Ly)
    rd = riesz_director(u, Lx, Ly, sigma_dir)
    cm = coherence_multiscale(u, x, y, sigmas, ramp=ramp)
    ops = cm[sigma_extract]["ops"] if sigma_extract in cm else \
        coherence_multiscale(u, x, y, [sigma_extract], ramp=ramp)[sigma_extract]["ops"]
    W = W_from_k(ops["k1_orig"], ops["k2_orig"], Lx, Ly)

    X, Y = np.meshgrid(x, y)
    if window is None:
        window = (x[0], x[-1], y[0], y[-1])
    x0, x1, y0, y1 = window
    mx = (x >= x0) & (x <= x1)
    my = (y >= y0) & (y <= y1)
    ext = [x0, x1, y0, y1]

    def crop(a):
        return a[np.ix_(my, mx)]

    n = 4 + len(sigmas)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.9 * nrow))
    axes = np.atleast_2d(axes).ravel()

    ax = axes[0]
    ax.imshow(crop(u), extent=ext, origin="lower", cmap="gray")
    s = quiver_step
    ax.quiver(X[np.ix_(my, mx)][::s, ::s], Y[np.ix_(my, mx)][::s, ::s],
              crop(rd["d1"])[::s, ::s], crop(rd["d2"])[::s, ::s],
              color="crimson", scale=30, headwidth=2, pivot="mid")
    ax.set_title(f"{title} u + Riesz director")

    ax = axes[1]
    im = ax.imshow(crop(mono["A"]), extent=ext, origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax); ax.set_title("monogenic amplitude")

    ax = axes[2]
    im = ax.imshow(np.degrees(crop(mono["orientation"])), extent=ext,
                   origin="lower", cmap="twilight", vmin=-180, vmax=180)
    fig.colorbar(im, ax=ax); ax.set_title("Riesz orientation (raw, deg)")

    for i, sg in enumerate(sigmas):
        ax = axes[3 + i]
        im = ax.imshow(np.clip(crop(cm[sg]["chi"]), 0, 0.5), extent=ext,
                       origin="lower", cmap="viridis", vmin=0, vmax=0.5)
        fig.colorbar(im, ax=ax)
        ax.set_title(rf"$\chi$ at $\sigma={sg:.2f}$")

    ax = axes[3 + len(sigmas)]
    im = ax.imshow(np.clip(crop(W), 0, 2), extent=ext, origin="lower",
                   cmap="inferno", vmin=0, vmax=2)
    fig.colorbar(im, ax=ax)
    ax.set_title(rf"$W$ from extracted $k$ ($\sigma={sigma_extract:.2f}$)")

    for ax in axes[4 + len(sigmas):]:
        ax.axis("off")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return {"mono": mono, "riesz_director": rd, "coherence": cm, "W": W,
            "ops": ops}


# ---------------------------------------------------------------------
# demo fields
# ---------------------------------------------------------------------

def knee_field(mu, Lx=24 * np.pi, Ly=24 * np.pi, N=768):
    k1, k2 = np.sqrt(1 - mu**2), mu
    x = np.linspace(-Lx / 2, Lx / 2, N, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, N, endpoint=False)
    X, Y = np.meshgrid(x, y)
    u = np.cos(k1 * X + np.logaddexp(k2 * Y, -k2 * Y))
    return u, x, y


def pn_chain_field(mu, nd=8, Ly=80.0, N=768, eps=1e-6):
    k1 = np.sqrt(1 - mu**2)
    Lsp = np.pi / k1
    Lx = (nd + 3) * Lsp
    x = np.linspace(-Lx / 2, Lx / 2, N, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, N, endpoint=False) + 0.5 * Ly / N
    X, Y = np.meshgrid(x, y)
    xd = Lsp * (np.arange(nd) - (nd - 1) / 2.0)
    a = 0.5 * (1 + np.exp(-np.pi * nd))
    sq = np.sqrt(2 * np.maximum(np.abs(Y), eps))
    thetas = np.arange(nd + 1) * np.pi
    psi = np.full_like(X, a)
    for j, c in enumerate(xd):
        delta = 0.5 * (np.exp(-thetas[j + 1]) - np.exp(-thetas[j]))
        psi += delta * nerf((X - c) / sq)
    theta = Y - np.sign(Y) * np.log(psi)
    return np.cos(theta), x, y


def _demo(outdir):
    os.makedirs(outdir, exist_ok=True)
    figs = []
    u, x, y = knee_field(0.9)
    diagnostics_figure(u, x, y, title=r"knee $\mu=0.9$:",
                       window=(-20, 20, -12, 12),
                       path=os.path.join(outdir, "diag_knee_mu0.90.png"))
    figs.append("diag_knee_mu0.90.png")

    u, x, y = pn_chain_field(0.9)
    diagnostics_figure(u, x, y, title=r"PN chain $\mu=0.9$:",
                       window=(-22, 22, -16, 16),
                       path=os.path.join(outdir, "diag_pn_chain_mu0.90.png"))
    figs.append("diag_pn_chain_mu0.90.png")
    return figs


class _Args:
    outdir = "out_diag_viz"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=_Args.outdir)
    _demo(p.parse_args().outdir)
