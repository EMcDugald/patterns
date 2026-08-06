# pipelines/wall_band_and_aniso_window.py
"""
Follow-through on the three next steps from the ansatz-symbolic session:

A. Wall-band prediction check on actual isotropic uHu extraction.
   Theory: above a PN chain, tilt is concentrated in erf walls of width
   sqrt(2|y|) that only merge into the smooth tilted far field at
   |y| ~ L^2/2 = pi^2 / (2 (1 - mu^2)). Prediction: a band of elevated
   chi and eigen-angle error extending to ~26 at mu = 0.9 but only ~6.6
   at mu = 0.5, under the SAME isotropic kernel. Checked here by running
   compute_uhu_ops_smoother (the real extraction path) on eps-regularized
   analytic PN chains and binning error metrics by |y|.

B. Anisotropic (sigma_x, sigma_y) window on the knee core.
   On the core the two failure channels DECOUPLE by axis exactly:
   corrugation lives in x (survival e^{-2 sigma_x^2 k1^2}, closed form),
   the flip competitor <theta_y^2>_{sigma_y} depends on sigma_y only.
   So the admissible region is a product set: sigma_x above a line,
   sigma_y below a line -- and since <theta_y^2>_{sigma_y} -> 0 as
   sigma_y -> 0, a quasi-1D (long-x, thin-y) kernel opens a window at
   EVERY mu < 1, removing the finite-mu* obstruction that closes the
   isotropic diagonal sigma_x = sigma_y.

C. Exploratory RCN line-energy crossing, knee vs PN chain, matched
   far field. Uses tilt-matched units: theta_PN = k2 y - sign(y) log psi
   with wall spacing L = pi/k1, so both structures carry far-field
   |k| = 1 at angle mu = sin(alpha). Knee line energy is closed form
   (8 mu^3 / 3, sympy-verified in uhu_symbolic). PN needs a core cutoff
   r_c (phase-only energy diverges logarithmically at the core -- itself
   part of the "phase alone cannot do nucleation" argument); crossing
   mu_c reported for several r_c and smearing choices so sensitivity is
   visible, NOT hidden. Clearly exploratory: the physically clean
   comparison is via the microscopic SH energy (existing line-energy
   scripts); treat this as the phase-level counterpart.
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

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "src", _HERE / "src"):
    if _p.exists():
        sys.path.insert(0, str(_p))

from op_extract.uhu_smoother import compute_uhu_ops_smoother


# ---------------------------------------------------------------------
# shared numerics
# ---------------------------------------------------------------------

def spectral_deriv(arr, Lx, Ly, which):
    ny, nx = arr.shape
    kx = 2 * np.pi * fftfreq(nx, d=Lx / nx)
    ky = 2 * np.pi * fftfreq(ny, d=Ly / ny)
    XI, ETA = np.meshgrid(kx, ky)
    F = fft2(arr)
    op = {"x": 1j * XI, "y": 1j * ETA,
          "xx": -XI**2, "yy": -ETA**2, "xy": -XI * ETA}[which]
    return np.real(ifft2(op * F))


def macro_iso(arr, sigma, Lx, Ly):
    ny, nx = arr.shape
    kx = 2 * np.pi * fftfreq(nx, d=Lx / nx)
    ky = 2 * np.pi * fftfreq(ny, d=Ly / ny)
    XI, ETA = np.meshgrid(kx, ky)
    ker = np.exp(-0.5 * sigma**2 * (XI**2 + ETA**2))
    return np.real(ifft2(ker * fft2(arr)))


def cos_ramp(x, y, frac=0.08):
    """Separable cosine-taper ramp, margins frac*L per side."""
    def taper(v):
        L = v[-1] - v[0]
        t = (v - v[0]) / L
        r = np.ones_like(t)
        m = t < frac
        r[m] = 0.5 * (1 - np.cos(np.pi * t[m] / frac))
        m = t > 1 - frac
        r[m] = 0.5 * (1 - np.cos(np.pi * (1 - t[m]) / frac))
        return r
    return np.outer(taper(y), taper(x))


def pn_chain_theta(X, Y, x_defects, k2=1.0, eps=1e-6):
    """theta and exact gradient for the PN chain.
    k2 = 1 reproduces the codebase form; k2 = sqrt(1-mu^2-free) rescales
    the linear phase for the tilt-matched units of Part C (erf smearing
    width scaled by 1/sqrt(k2) so k2 = 1 is unchanged)."""
    nd = len(x_defects)
    a = 0.5 * (1 + np.exp(-np.pi * nd))
    Ya = np.maximum(np.abs(Y), eps)
    sq = np.sqrt(2 * Ya / k2)
    thetas = np.arange(nd + 1) * np.pi
    pref = 2.0 / np.sqrt(np.pi)
    psi = np.full_like(X, a)
    psi_x = np.zeros_like(X)
    psi_ay = np.zeros_like(X)   # d/d|y|
    for j, xd in enumerate(x_defects):
        delta = 0.5 * (np.exp(-thetas[j + 1]) - np.exp(-thetas[j]))
        arg = (X - xd) / sq
        g = np.exp(-arg**2)
        psi += delta * nerf(arg)
        psi_x += delta * pref * g / sq
        psi_ay += delta * pref * g * (-(X - xd)) / (2 * Ya * sq) * 1.0
    sgn = np.sign(Y)
    sgn[sgn == 0] = 1.0
    theta = k2 * Y - sgn * np.log(psi)
    thx = -sgn * psi_x / psi                # odd in y
    thy = k2 - psi_ay / psi                 # even in y
    return theta, thx, thy, psi


# ---------------------------------------------------------------------
# Part A: wall-band check on real extraction
# ---------------------------------------------------------------------

def part_A(outdir, report, sigma=np.pi / 2, N=768):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
    results = {}
    for mu, colr in ((0.5, "tab:blue"), (0.9, "tab:red")):
        k1 = np.sqrt(1 - mu**2)
        Lsp = np.pi / k1
        nd = 8
        Lx = (nd + 3) * Lsp
        Ly = 90.0
        x = np.linspace(-Lx / 2, Lx / 2, N, endpoint=False)
        y = np.linspace(-Ly / 2, Ly / 2, N, endpoint=False) + 0.5 * Ly / N
        X, Y = np.meshgrid(x, y)
        xd = Lsp * (np.arange(nd) - (nd - 1) / 2.0)
        theta, thx, thy, _ = pn_chain_theta(X, Y, xd, k2=1.0)
        u = np.cos(theta)
        ramp = cos_ramp(x, y, frac=0.06)

        smoother = lambda arr: macro_iso(arr, sigma, Lx, Ly)
        ops = compute_uhu_ops_smoother(u, x, y, smoother, spectral_deriv, ramp)

        lam1, lam2 = ops["lam1"], ops["lam2"]
        with np.errstate(invalid="ignore", divide="ignore"):
            chi = np.abs(lam2 / np.where(np.abs(lam1) < 1e-12, np.nan, lam1))
        ang = np.degrees(np.arctan2(ops["e12"], ops["e11"]))
        ang_true = np.degrees(np.arctan2(thy, thx))
        aerr = np.abs(((ang - ang_true) + 90) % 180 - 90)

        # interior mask: away from ramp margins and defect cores
        Rmin = np.min(np.stack([np.hypot(X - c, Y) for c in xd]), axis=0)
        inner = (ramp > 0.999) & (Rmin > 2.0)

        yb = np.linspace(0, 40, 41)
        ym = 0.5 * (yb[1:] + yb[:-1])
        med_chi, med_ae = [], []
        for lo, hi in zip(yb[:-1], yb[1:]):
            m = inner & (np.abs(Y) >= lo) & (np.abs(Y) < hi)
            med_chi.append(np.nanmedian(chi[m]) if m.any() else np.nan)
            med_ae.append(np.nanmedian(aerr[m]) if m.any() else np.nan)
        results[mu] = (ym, np.array(med_chi), np.array(med_ae))

        merge = Lsp**2 / 2
        axes[0].plot(ym, med_chi, color=colr, label=rf"$\mu={mu}$")
        axes[0].axvline(merge, color=colr, ls=":", lw=1)
        axes[1].plot(ym, med_ae, color=colr, label=rf"$\mu={mu}$")
        axes[1].axvline(merge, color=colr, ls=":", lw=1)
        report(f"mu={mu}: L={Lsp:.2f}, predicted wall-merge height "
               f"L^2/2 = {merge:.1f}")

    axes[0].set_xlabel(r"$|y|$"); axes[0].set_ylabel(r"median $\chi$")
    axes[0].set_title(rf"Isotropic extraction ($\sigma=\pi/2$): rank-two band")
    axes[0].legend()
    axes[1].set_xlabel(r"$|y|$"); axes[1].set_ylabel("median angle error (deg)")
    axes[1].set_title("Eigen-angle error band (dotted: predicted $L^2/2$)")
    axes[1].legend()
    fig.tight_layout()
    f1 = os.path.join(outdir, "wall_band_extraction_check.png")
    fig.savefig(f1, dpi=160)
    plt.close(fig)
    return [f1]


# ---------------------------------------------------------------------
# Part B: anisotropic (sigma_x, sigma_y) window on the knee core
# ---------------------------------------------------------------------

def part_B(outdir, report, tol_corr=0.05):
    gh_s, gh_w = np.polynomial.hermite_e.hermegauss(120)
    gh_w = gh_w / gh_w.sum()

    def avg_thy2_core(mu, sy):
        return float(np.sum(gh_w * (mu * np.tanh(mu * sy * gh_s)) ** 2))

    sx = np.linspace(0.05, 2.2 * np.pi, 220)
    sy = np.linspace(0.02, 2.2 * np.pi, 220)
    SX, SY = np.meshgrid(sx, sy)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, mu in zip(axes, (0.80, 0.88, 0.95)):
        k1sq = 1 - mu**2
        corr_ok = np.exp(-2 * SX**2 * k1sq) < tol_corr
        thy2 = np.array([avg_thy2_core(mu, s) for s in sy])
        flip_ok = (thy2 < k1sq)[:, None] * np.ones_like(SX, bool)
        ok = corr_ok & flip_ok
        ax.pcolormesh(SX, SY, ok, cmap="Greens", shading="auto",
                      vmin=0, vmax=1.4)
        ax.plot(sx, sx, "k--", lw=1.2, label=r"isotropic $\sigma_x=\sigma_y$")
        sx_min = np.sqrt(np.log(1 / tol_corr) / (2 * k1sq))
        idx = np.where(thy2 < k1sq)[0]
        sy_max = sy[idx[-1]] if idx.size else 0.0
        ax.axvline(sx_min, color="tab:red", lw=1,
                   label=rf"$\sigma_x^{{\min}}={sx_min:.2f}$")
        ax.axhline(sy_max, color="tab:blue", lw=1,
                   label=rf"$\sigma_y^{{\max}}={sy_max:.2f}$")
        iso_ok = np.any(ok & np.isclose(SX, SY, atol=sx[1] - sx[0]))
        ax.set_title(rf"$\mu={mu}$: window {'open' if sx_min < sy_max else 'ANISO ONLY'}"
                     + ("" if iso_ok else " (isotropic closed)"))
        ax.set_xlabel(r"$\sigma_x$ (along core)")
        ax.legend(fontsize=7, loc="upper right")
        report(f"mu={mu}: corrugation needs sigma_x > {sx_min:.2f}; "
               f"flip needs sigma_y < {sy_max:.2f}; "
               f"isotropic feasible: {iso_ok}")
    axes[0].set_ylabel(r"$\sigma_y$ (across core)")
    fig.suptitle("Knee core: admissible kernel region (green) -- "
                 "channels decouple by axis", y=1.02)
    fig.tight_layout()
    f1 = os.path.join(outdir, "knee_aniso_window.png")
    fig.savefig(f1, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [f1]


# ---------------------------------------------------------------------
# Part C: exploratory RCN line-energy crossing (matched far field)
# ---------------------------------------------------------------------

def part_C(outdir, report, cutoffs=(np.pi / 4, np.pi / 2, np.pi), N=1024):
    mus = np.linspace(0.55, 0.98, 24)
    E_knee = 8 * mus**3 / 3

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mus, E_knee, "k-", lw=2, label=r"knee: $8\mu^3/3$ (exact)")

    for rc in cutoffs:
        E_pn = []
        for mu in mus:
            k1 = np.sqrt(1 - mu**2)
            k2 = mu
            Lsp = np.pi / k1
            nd = 8
            Lx = nd * Lsp                      # periodic cell in x
            Ly = 80.0
            x = np.linspace(-Lx / 2, Lx / 2, N, endpoint=False)
            y = np.linspace(-Ly / 2, Ly / 2, N, endpoint=False) + 0.5 * Ly / N
            X, Y = np.meshgrid(x, y)
            xd = Lsp * (np.arange(nd) - (nd - 1) / 2.0)
            theta, thx, thy, _ = pn_chain_theta(X, Y, xd, k2=k2)
            # RCN density; far field |k|^2 = k1^2 + k2^2 = 1 by matching
            div_k = spectral_deriv(thx, Lx, Ly, "x") + spectral_deriv(thy, Lx, Ly, "y")
            dens = div_k**2 + (1 - (thx**2 + thy**2))**2
            Rmin = np.min(np.stack([np.hypot(X - c, Y) for c in xd]), axis=0)
            mask = Rmin > rc
            dA = (x[1] - x[0]) * (y[1] - y[0])
            E_pn.append(np.sum(dens[mask]) * dA / Lx)   # per defect-spacing?? per unit x
        E_pn = np.array(E_pn)
        ax.plot(mus, E_pn, label=rf"PN chain, cutoff $r_c={rc:.2f}$")
        cross = np.where(np.diff(np.sign(E_pn - E_knee)))[0]
        mu_c = mus[cross[0]] if cross.size else np.nan
        report(f"cutoff r_c={rc:.2f}: crossing mu_c ~ {mu_c:.3f}"
               if np.isfinite(mu_c) else
               f"cutoff r_c={rc:.2f}: no crossing in range")

    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel("RCN line energy per unit x")
    ax.set_title("Exploratory: knee vs PN chain (matched far field, phase-only)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    f1 = os.path.join(outdir, "knee_vs_pn_energy_crossing.png")
    fig.savefig(f1, dpi=160)
    plt.close(fig)
    return [f1]


# ---------------------------------------------------------------------

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    lines = []

    def report(msg):
        print(msg)
        lines.append(str(msg))

    figs = []
    if "A" in args.parts:
        report("=== A: wall-band check on isotropic extraction ===")
        figs += part_A(args.outdir, report)
    if "B" in args.parts:
        report("\n=== B: anisotropic window on the knee core ===")
        figs += part_B(args.outdir, report)
    if "C" in args.parts:
        report("\n=== C: exploratory energy crossing ===")
        figs += part_C(args.outdir, report)

    with open(os.path.join(args.outdir, "wall_band_report.txt"), "w") as f:
        f.write("\n".join(lines))
    return figs


class _Args:
    parts = "ABC"
    outdir = "out_wall_band"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parts", default=_Args.parts)
    p.add_argument("--outdir", default=_Args.outdir)
    main(p.parse_args())
