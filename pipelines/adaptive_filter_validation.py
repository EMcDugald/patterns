# pipelines/adaptive_filter_validation.py
"""
Ground-truth validation for the adaptive anisotropic uHu averaging kernel
(anisotropic_macro.py / anisotropic_diffusion.py), before touching SH data.

Test fields (known k everywhere)
--------------------------------
knee : u = cos(theta), theta = kx x + (m / beta) log cosh(beta (y - yc)).
       k_true = (kx, m tanh(beta (y - yc))): a smoothed knee bend with
       far-field angles +-atan(m / kx) and core sharpness set by beta.
       Large beta stresses the isotropic filter exactly the way the SH
       knee cores do. Not y-periodic, so extraction uses the standard
       rectangular ramp, as in the production pipeline.
arc  : u = cos(r), concentric rolls about a center outside the frame.
       |k_true| = 1 everywhere, orientation rotates continuously --
       isolates the patch-frozen-orientation assumption from wavenumber
       gradients.

Smoothers compared (same extraction code, only the averaging differs)
---------------------------------------------------------------------
iso_global : exact global Gaussian macro() -- the production baseline.
iso_patch  : macro_aniso_field with aspect_max = 1 -- patchwise Hann
             overlap-add of the SAME isotropic kernel. Any difference
             from iso_global is pure patching artifact; this separates
             "patching cost" from "anisotropy benefit".
aniso_it1..N : the iterative anisotropic filter, seeded by iso_global.
pde (optional) : one anisotropic pass through the tensor-diffusion
             smoother, cross-checking the patch-freezing assumption.

The iteration here is a sign-aligned variant of iterate_anisotropic_uhu:
before damping k_new against k_prev, k_new is flipped wherever
dot(k_new, k_prev) < 0. The director ambiguity (k ~ -k) otherwise lets
the eigen-solver hand back either sign patch by patch, and naive vector
damping can cancel the field across a sign flip.

Metrics (interior, and separately in a band around the core)
-----------------------------------------------------------
  |k| error : median and p95 of | |k| - |k_true| |
  angle err : median and p95 of director angle error (mod pi)
plus per-iteration convergence of both, core-normal profiles of k2
against m tanh(beta y), and error maps.

Usage
-----
python adaptive_filter_validation.py --field knee --out results/adaptive_val
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
    from utils.anisotropic_macro import (macro_aniso_field, coherence_from_eigs,
                                         _hann2d, director_orientation,
                                         macro_aniso_uniform)
except ImportError:
    from anisotropic_macro import (macro_aniso_field, coherence_from_eigs,
                                   _hann2d, director_orientation,
                                   macro_aniso_uniform)

try:
    from utils.anisotropic_diffusion import macro_aniso_diffusion
    _HAVE_PDE = True
except ImportError:
    try:
        from anisotropic_diffusion import macro_aniso_diffusion
        _HAVE_PDE = True
    except ImportError:
        _HAVE_PDE = False

try:
    from op_extract.uhu_smoother import compute_uhu_ops_smoother
except ImportError:
    from uhu_smoother import compute_uhu_ops_smoother

try:
    from paper_style import use_paper_style, add_cbar
    use_paper_style()
except ImportError:
    add_cbar = None


# -----------------------------------------------------------------------
# Ground-truth test fields
# -----------------------------------------------------------------------

def make_grid(Nx, Ny, Lx, Ly):
    x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y


def field_knee(X, Y, kx=1.0, m=0.5, beta=2.0, yc=0.0):
    theta = kx * X + (m / beta) * np.log(np.cosh(beta * (Y - yc)))
    k1 = np.full_like(X, kx)
    k2 = m * np.tanh(beta * (Y - yc))
    return np.cos(theta), k1, k2


def field_arc(X, Y, r0_offset=1.5):
    """Concentric rolls, center placed below the frame by r0_offset * Ly."""
    yc = Y.min() - r0_offset * (Y.max() - Y.min())
    r = np.sqrt(X**2 + (Y - yc) ** 2)
    k1 = X / r
    k2 = (Y - yc) / r
    return np.cos(r), k1, k2


# -----------------------------------------------------------------------
# Error metrics (director-aware)
# -----------------------------------------------------------------------

def kmag_error(k1, k2, k1t, k2t):
    return np.abs(np.hypot(k1, k2) - np.hypot(k1t, k2t))


def angle_error(k1, k2, k1t, k2t):
    """Director angle error in radians, mod pi (k ~ -k)."""
    dth = np.arctan2(k2, k1) - np.arctan2(k2t, k1t)
    dth = np.mod(dth + np.pi / 2, np.pi) - np.pi / 2
    return np.abs(dth)


def summarize(err, mask):
    v = err[mask]
    v = v[np.isfinite(v)]
    return float(np.median(v)), float(np.percentile(v, 95))


# -----------------------------------------------------------------------
# Smoother constructors
# -----------------------------------------------------------------------

def smoother_iso_global(sigma, Lx, Ly):
    return lambda arr: macro(arr, sigma, Lx, Ly)


def smoother_iso_patch(sigma, Lx, Ly, patch):
    dummy = None

    def f(arr):
        z = np.zeros_like(arr)
        return macro_aniso_field(arr, k1=z, k2=z + 1.0, coherence=z,
                                 Lx=Lx, Ly=Ly, sigma_perp_floor=sigma,
                                 aspect_max=1.0, patch=patch)
    return f


def smoother_aniso(k1, k2, coh, sigma, Lx, Ly, aspect_max, patch,
                   sharpen=1.0):
    if sharpen <= 1.0:
        return lambda arr: macro_aniso_field(
            arr, k1=k1, k2=k2, coherence=coh, Lx=Lx, Ly=Ly,
            sigma_perp_floor=sigma, aspect_max=aspect_max, patch=patch)
    return lambda arr: macro_aniso_field_v2(
        arr, k1=k1, k2=k2, coherence=coh, Lx=Lx, Ly=Ly,
        sigma_base=sigma, aspect_max=aspect_max, sharpen=sharpen,
        patch=patch)


def macro_aniso_field_v2(arr, k1, k2, coherence, Lx, Ly, sigma_base,
                         aspect_max=2.5, sharpen=2.0, patch=64,
                         stride=None):
    """
    Variant of anisotropic_macro.macro_aniso_field that REDISTRIBUTES the
    averaging area instead of only growing it: where the local coherence
    c ~ 1,

        s_par  = sigma_base * (1 + (aspect_max - 1) c)
        s_perp = sigma_base / (1 + (sharpen    - 1) c)

    so the kernel narrows ACROSS the slow-variation direction only where
    the single-wavevector description is trustworthy, and relaxes back to
    the isotropic sigma_base near defects (c -> 0). Rationale: for a PGB
    the fast phase runs largely ALONG the core, so the elongated axis
    still averages away the oscillation while the narrow axis stops
    over-blurring the tanh transition. The original prescription keeps
    s_perp pinned at the floor and can only ever add blur.
    """
    ny, nx = arr.shape
    stride = stride or patch // 2
    win = _hann2d(patch, patch)
    out = np.zeros_like(arr, dtype=float)
    wsum = np.zeros_like(arr, dtype=float)

    ys = list(range(0, ny - patch + 1, stride)) or [0]
    xs = list(range(0, nx - patch + 1, stride)) or [0]
    if ys[-1] != ny - patch:
        ys.append(max(ny - patch, 0))
    if xs[-1] != nx - patch:
        xs.append(max(nx - patch, 0))

    for y0 in ys:
        for x0 in xs:
            sl = (slice(y0, y0 + patch), slice(x0, x0 + patch))
            a_patch = arr[sl]
            if a_patch.shape != (patch, patch):
                continue
            k1p, k2p, cp = k1[sl], k2[sl], coherence[sl]
            weight = np.hypot(k1p, k2p)
            theta_k = director_orientation(k1p, k2p, weight=weight)
            theta_long = theta_k + np.pi / 2.0
            c_local = float(np.median(cp))
            s_par = sigma_base * (1.0 + (aspect_max - 1.0) * c_local)
            s_perp = sigma_base / (1.0 + (sharpen - 1.0) * c_local)
            smoothed = macro_aniso_uniform(
                a_patch, theta_long, s_par, s_perp,
                Lx * patch / nx, Ly * patch / ny)
            out[sl] += smoothed * win
            wsum[sl] += win

    wsum = np.where(wsum < 1e-12, 1.0, wsum)
    return out / wsum


def smoother_pde(k1, k2, coh, sigma, Lx, Ly, aspect_max):
    theta_stripe = np.arctan2(k2, k1) + np.pi / 2.0
    return lambda arr: macro_aniso_diffusion(
        arr, theta_stripe, coh, Lx, Ly,
        sigma_perp_floor=sigma, aspect_max=aspect_max)


# -----------------------------------------------------------------------
# Sign-aligned anisotropic iteration
# -----------------------------------------------------------------------

def slowness_director(k1, k2, sigma, Lx, Ly):
    """
    Direction of fastest macroscopic variation: grad of the (pre-smoothed)
    slow fields |k| and k2, combined. Feeding this to macro_aniso_field
    as if it were the wavevector makes the kernel's LONG axis lie along
    the level sets of the slow fields (perpendicular to their gradient),
    i.e. along a PGB core -- instead of along the stripe tangent, which
    for a PGB points ACROSS the boundary and smears the transition.
    """
    kmag = macro(np.hypot(k1, k2), sigma, Lx, Ly)
    k2s = macro(k2, sigma, Lx, Ly)
    g = np.zeros((2,) + kmag.shape)
    for f, w in ((kmag, 1.0), (k2s, 1.0)):
        gx = SpectralDerivs(f, Lx, Ly, "x")
        gy = SpectralDerivs(f, Lx, Ly, "y")
        # director doubling: accumulate orientation of (gx, gy) mod pi
        g[0] += w * (gx * gx - gy * gy)
        g[1] += w * (2.0 * gx * gy)
    ang = 0.5 * np.arctan2(g[1], g[0])          # gradient orientation
    mag = np.sqrt(np.hypot(g[0], g[1]))
    return mag * np.cos(ang), mag * np.sin(ang)


def iterate_aniso_signed(u, x, y, ramp, sigma, n_iter=3, aspect_max=2.5,
                         damping=0.4, patch=64, orient="stripe",
                         sharpen=1.0):
    """
    Iterative anisotropic uHu, seeded by the exact global isotropic pass.

    orient = "stripe"   : original prescription -- long axis along the
                          stripe tangent (perpendicular to k).
    orient = "slowness" : long axis along level sets of the slow fields
                          (|k|, k2), i.e. perpendicular to their gradient.

    Damping is applied after sign-aligning k_new to k_prev (flip where
    dot < 0), which the original iterate_anisotropic_uhu omits: with the
    director ambiguity, naive vector damping can cancel the k field
    wherever the eigenvector branch flips between iterations.
    Returns [result_iso, result_it1, ...].
    """
    Lx, Ly = x[-1] - x[0], y[-1] - y[0]
    res = compute_uhu_ops_smoother(
        u, x, y, smoother_iso_global(sigma, Lx, Ly), SpectralDerivs, ramp)
    history = [res]
    k1p, k2p = res["k1_orig"], res["k2_orig"]

    for _ in range(n_iter):
        coh = coherence_from_eigs(res["lam1"], res["lam2"])
        if orient == "slowness":
            d1, d2 = slowness_director(k1p, k2p, sigma, Lx, Ly)
        else:
            d1, d2 = k1p, k2p
        sm = smoother_aniso(d1, d2, coh, sigma, Lx, Ly, aspect_max, patch,
                            sharpen=sharpen)
        new = compute_uhu_ops_smoother(u, x, y, sm, SpectralDerivs, ramp)

        flip = np.sign(new["k1_orig"] * k1p + new["k2_orig"] * k2p)
        flip = np.where(flip == 0, 1.0, flip)
        k1n = flip * new["k1_orig"]
        k2n = flip * new["k2_orig"]

        new["k1_orig"] = (1 - damping) * k1p + damping * k1n
        new["k2_orig"] = (1 - damping) * k2p + damping * k2n
        k1p, k2p = new["k1_orig"], new["k2_orig"]
        res = new
        history.append(res)
    return history


# -----------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------

def run(cfg):
    out_dir = Path(cfg.out) / cfg.field
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s):
        print(s)
        lines.append(s)

    x, y, X, Y = make_grid(cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly)
    Lx, Ly = cfg.Lx, cfg.Ly

    if cfg.field == "knee":
        u, k1t, k2t = field_knee(X, Y, kx=cfg.kx, m=cfg.m, beta=cfg.beta)
        core_band = np.abs(Y) < 2.0 / cfg.beta
    elif cfg.field == "arc":
        u, k1t, k2t = field_arc(X, Y)
        core_band = np.zeros_like(X, bool)
    else:
        raise ValueError(cfg.field)

    ramp = build_rectangular_ramp_smooth(
        x, y, xmargin=cfg.xmargin, ymargin=cfg.ymargin,
        tanhscale=cfg.tanhscale)
    interior = ramp >= 0.99

    log(f"field={cfg.field}  grid {cfg.Nx}x{cfg.Ny}  "
        f"Lx={Lx:.1f} Ly={Ly:.1f}  sigma={cfg.sigma:.4f}  "
        f"aspect_max={cfg.aspect_max}  patch={cfg.patch}  "
        f"n_iter={cfg.n_iter}  damping={cfg.damping}")
    if cfg.field == "knee":
        log(f"knee: kx={cfg.kx}  m={cfg.m}  beta={cfg.beta}  "
            f"(far-field angle {np.degrees(np.arctan2(cfg.m, cfg.kx)):.1f} deg)")

    results = {}

    # --- baselines ---
    sm0 = smoother_iso_global(cfg.sigma, Lx, Ly)
    results["iso_global"] = compute_uhu_ops_smoother(
        u, x, y, sm0, SpectralDerivs, ramp)

    smp = smoother_iso_patch(cfg.sigma, Lx, Ly, cfg.patch)
    results["iso_patch"] = compute_uhu_ops_smoother(
        u, x, y, smp, SpectralDerivs, ramp)

    # --- sign-aligned anisotropic iteration ---
    orient = getattr(cfg, "orient", "stripe")
    log(f"anisotropy orientation mode: {orient}")
    sharpen = getattr(cfg, "sharpen", 1.0)
    log(f"cross-axis sharpen factor: {sharpen}")
    hist = iterate_aniso_signed(
        u, x, y, ramp, cfg.sigma, n_iter=cfg.n_iter,
        aspect_max=cfg.aspect_max, damping=cfg.damping, patch=cfg.patch,
        orient=orient, sharpen=sharpen)
    for i, r in enumerate(hist[1:], start=1):
        results[f"aniso_it{i}"] = r

    # --- optional PDE cross-check (single pass, seeded by iso_global) ---
    if cfg.use_pde and _HAVE_PDE:
        r0 = results["iso_global"]
        coh = coherence_from_eigs(r0["lam1"], r0["lam2"])
        smd = smoother_pde(r0["k1_orig"], r0["k2_orig"], coh,
                           cfg.sigma, Lx, Ly, cfg.aspect_max)
        results["pde_pass1"] = compute_uhu_ops_smoother(
            u, x, y, smd, SpectralDerivs, ramp)

    # --- metrics table ---
    log("\nname          |k| err med/p95 (interior)   angle err med/p95 [deg]"
        "   |k| err med/p95 (core band)")
    table = {}
    for name, r in results.items():
        ek = kmag_error(r["k1_orig"], r["k2_orig"], k1t * ramp, k2t * ramp)
        ea = np.degrees(angle_error(r["k1_orig"], r["k2_orig"], k1t, k2t))
        km, kp = summarize(ek, interior)
        am, ap = summarize(ea, interior)
        if core_band.any():
            cm, cp = summarize(ek, interior & core_band)
        else:
            cm = cp = np.nan
        table[name] = (km, kp, am, ap, cm, cp)
        log(f"{name:12s}  {km:.4e} / {kp:.4e}      "
            f"{am:7.3f} / {ap:7.3f}         {cm:.4e} / {cp:.4e}")

    # --- convergence figure ---
    it_names = ["iso_global"] + [f"aniso_it{i}" for i in range(1, cfg.n_iter + 1)]
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2), constrained_layout=True)
    for j, (idx, lab) in enumerate([(0, "|k| err (median)"),
                                    (2, "angle err (median, deg)")]):
        vals = [table[n][idx] for n in it_names]
        axs[j].plot(range(len(vals)), vals, "o-", label="interior")
        if core_band.any():
            cvals = [table[n][4] if idx == 0 else np.nan for n in it_names]
            if idx == 0:
                axs[j].plot(range(len(cvals)), cvals, "s--", label="core band")
        axs[j].set_xlabel("iteration (0 = isotropic seed)")
        axs[j].set_title(lab, fontsize=10)
        axs[j].legend()
    fig.suptitle(
        f"{cfg.field}: error of the extracted wavevector vs. ground truth "
        f"across anisotropic-filter iterations.", fontsize=9)
    fig.savefig(out_dir / "convergence.png")
    plt.close(fig)

    # --- error maps: iso_global vs final aniso ---
    final = it_names[-1]
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 5.6), constrained_layout=True)
    extent = [x[0], x[-1], y[0], y[-1]]
    for row, name in enumerate(["iso_global", final]):
        r = results[name]
        ek = np.ma.masked_where(~interior, kmag_error(
            r["k1_orig"], r["k2_orig"], k1t * ramp, k2t * ramp))
        ea = np.ma.masked_where(~interior, np.degrees(angle_error(
            r["k1_orig"], r["k2_orig"], k1t, k2t)))
        im0 = axs[row, 0].imshow(ek, extent=extent, cmap="magma",
                                 vmin=0, vmax=np.nanpercentile(ek.compressed(), 99),
                                 aspect="auto", origin="lower", rasterized=True)
        im1 = axs[row, 1].imshow(ea, extent=extent, cmap="magma",
                                 vmin=0, vmax=np.nanpercentile(ea.compressed(), 99),
                                 aspect="auto", origin="lower", rasterized=True)
        axs[row, 0].set_ylabel(f"{name}\n$y$")
        for ax, im in ((axs[row, 0], im0), (axs[row, 1], im1)):
            if add_cbar is not None:
                add_cbar(fig, ax, im)
            else:
                fig.colorbar(im, ax=ax, shrink=0.85)
    axs[0, 0].set_title("$|\\,|k| - |k_{true}|\\,|$")
    axs[0, 1].set_title("director angle error [deg]")
    for ax in axs[1]:
        ax.set_xlabel("$x$")
    fig.suptitle(
        f"{cfg.field}: spatial error maps, isotropic baseline (top) vs "
        f"final anisotropic iterate (bottom).", fontsize=9)
    fig.savefig(out_dir / "error_maps.png")
    plt.close(fig)

    # --- knee core profile: k2(y) against m tanh(beta y) ---
    if cfg.field == "knee":
        col = cfg.Nx // 2
        fig, ax = plt.subplots(figsize=(5.6, 3.4), constrained_layout=True)
        ax.plot(y, k2t[:, col], "k-", lw=1.4, label="truth $m\\tanh(\\beta y)$")
        for name, style in (("iso_global", "r--"), (final, "b-")):
            r = results[name]
            ax.plot(y, r["k2_orig"][:, col], style, lw=1.0, alpha=0.85,
                    label=name)
        ax.set_xlabel("$y$")
        ax.set_ylabel("$k_2$")
        ax.legend()
        ax.set_title(
            "Knee core: transverse wavenumber profile through the bend; the\n"
            "anisotropic filter should track the tanh transition more tightly.",
            fontsize=9)
        fig.savefig(out_dir / "core_profile.png")
        plt.close(fig)

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    log(f"\nwrote {out_dir}/")
    return results, table


def main():
    p = argparse.ArgumentParser(description="Adaptive-filter ground-truth validation")
    p.add_argument("--field", choices=["knee", "arc"], default="knee")
    p.add_argument("--out", type=str, default="results/adaptive_val")
    p.add_argument("--Nx", type=int, default=512)
    p.add_argument("--Ny", type=int, default=256)
    p.add_argument("--Lx", type=float, default=40 * np.pi)
    p.add_argument("--Ly", type=float, default=20 * np.pi)
    p.add_argument("--kx", type=float, default=1.0)
    p.add_argument("--m", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--sigma", type=float, default=np.pi / 2)
    p.add_argument("--aspect_max", type=float, default=2.5)
    p.add_argument("--patch", type=int, default=64)
    p.add_argument("--n_iter", type=int, default=3)
    p.add_argument("--damping", type=float, default=0.4)
    p.add_argument("--xmargin", type=float, default=0.05)
    p.add_argument("--ymargin", type=float, default=0.10)
    p.add_argument("--tanhscale", type=float, default=60.0)
    p.add_argument("--orient", choices=["stripe", "slowness"],
                   default="stripe")
    p.add_argument("--sharpen", type=float, default=1.0)
    p.add_argument("--use_pde", action="store_true")
    a = p.parse_args()
    run(a)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        class _Args:
            field      = "knee"     # "knee" or "arc"
            out        = "results/adaptive_val"
            Nx, Ny     = 512, 256
            Lx, Ly     = 40 * np.pi, 20 * np.pi
            kx, m      = 1.0, 0.5
            beta       = 2.0
            sigma      = np.pi/2
            aspect_max = 2.5
            patch      = 64
            n_iter     = 20
            damping    = 0.4
            xmargin    = 0.05
            ymargin    = 0.10
            tanhscale  = 60.0
            orient     = "slowness" # "stripe" (original) or "slowness"
            sharpen    = 2.0        # >1: narrow cross-axis where coherent
            use_pde    = False      # True: add tensor-diffusion cross-check
        run(_Args)
    else:
        main()
