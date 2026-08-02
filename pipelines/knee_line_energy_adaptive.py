# pipelines/knee_line_energy_adaptive.py
"""
Line-energy test for the adaptive averaging on exact analytic knee bends.

For each far-field angle mu in cfg.mus, build the defect-free Hopf-Cole
knee pattern u = cos(theta), theta = k1 x + log(2 cosh(k2 y)) with
k1 = sqrt(1 - mu^2), k2 = mu, amplitude 1 (k1 is nudged to the nearest
x-commensurate value so the field is periodic in x). Then compute:

  e_micro : the microscopic SH energy density (sh_sims.core.edensity,
            0.5 * ([(1+Lap)u]^2 - R u^2 + 0.5 u^4)), evaluated on u*ramp,
            with the far-field baseline subtracted so the excess energy of
            the bend is what remains.
  e_iso   : e_micro averaged with the production isotropic Gaussian.
  e_ada   : e_micro averaged with the adaptive anisotropic kernel, built
            from the converged adaptive k field of the same frame
            (slowness orientation, coherence gating, auto aspect/sharpen
            from mu via the leakage formula).

Because the analytic knee is invariant along x, the exact macroscopic
answer is a LINE energy: a smooth ridge in y, constant in x. The
x-average of e_micro is therefore the ground-truth ridge profile, and
any x-oscillation in the averaged fields along the core is estimator
artifact. The known signature to look for is a corrugation at wavenumber
2 k1, wavelength pi/k1, which is exactly the PN dislocation spacing L.

Outputs per mu (out/<field>/mu<mu>/): a 2x3 dashboard, plus a shared
summary figure and summary.csv with, per mu and per smoother, the
relative core-row oscillation amplitude at 2 k1, the measured blob
wavelength, the integrated excess energy per unit core length (an
averaging-conservation check: all three should agree), and the extracted
core director angle (flags the large-mu eigen-flip regime).

The field construction is factored so a Pismen multi-dislocation pattern
can be added later as cfg.field = "pismen".

Usage
-----
python knee_line_energy_adaptive.py --out results/knee_line_energy
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
    from spectral import macro, SpectralDerivs

try:
    from utils.geometry import build_rectangular_ramp_smooth
except ImportError:
    from geometry import build_rectangular_ramp_smooth

try:
    from op_extract.uhu_adaptive import (
        compute_uhu_ops_adaptive, derived_kernel_params_mu,
        macro_aniso_field_v2, macro_aniso_field_v3_pixelblend,
        slowness_director,
    )
except ImportError:
    from uhu_adaptive import (
        compute_uhu_ops_adaptive, derived_kernel_params_mu,
        macro_aniso_field_v2, macro_aniso_field_v3_pixelblend,
        slowness_director,
    )

try:
    from utils.anisotropic_macro import coherence_from_eigs
except ImportError:
    from anisotropic_macro import coherence_from_eigs

try:
    from sh_sims.core import edensity as _edensity_core
    _HAVE_CORE = True
except ImportError:
    try:
        from core import edensity as _edensity_core
        _HAVE_CORE = True
    except ImportError:
        _HAVE_CORE = False

try:
    from paper_style import use_paper_style
    use_paper_style()
except ImportError:
    pass


# -----------------------------------------------------------------------
# Microscopic energy (thin wrapper so the script runs with or without
# sh_sims.core on the path; the fallback reimplements edensity verbatim)
# -----------------------------------------------------------------------

def _dealias_mask(Nx, Ny):
    Fx = np.zeros((Nx, 1), dtype=bool)
    Fy = np.zeros((Ny, 1), dtype=bool)
    Fx[int(Nx / 2 - np.round(Nx / 4)):int(1 + Nx / 2 + np.round(Nx / 4))] = True
    Fy[int(Ny / 2 - np.round(Ny / 4)):int(1 + Ny / 2 + np.round(Ny / 4))] = True
    alxi, aleta = np.meshgrid(Fx, Fy)
    return alxi | aleta


def micro_energy(u, Lx, Ly, R):
    Ny, Nx = u.shape
    from numpy.fft import fftfreq
    kx = (2.0 * np.pi / Lx) * fftfreq(Nx, 1.0 / Nx)
    ky = (2.0 * np.pi / Ly) * fftfreq(Ny, 1.0 / Ny)
    xi, eta = np.meshgrid(kx, ky)
    ind = _dealias_mask(Nx, Ny)
    if _HAVE_CORE:
        return _edensity_core(xi, eta, u, ind, R)
    from numpy.fft import fft2, ifft2
    eloc = (1 - xi ** 2 - eta ** 2) * fft2(u)
    eloc[ind] = 0
    eloc = np.real(ifft2(eloc) ** 2)
    u0sq = fft2(u ** 2); u0sq[ind] = 0; u0sq = np.real(ifft2(u0sq))
    u04th = fft2(u0sq ** 2); u04th[ind] = 0; u04th = np.real(ifft2(u04th))
    return 0.5 * (eloc - R * u0sq + 0.5 * u04th)


# -----------------------------------------------------------------------
# Analytic test fields (extension point: add "pismen" here later)
# -----------------------------------------------------------------------

def field_knee(mu, x, y, Lx):
    """Exact Hopf-Cole knee, k1 nudged x-commensurate; returns u, k1, k2."""
    k1 = np.sqrt(1.0 - mu * mu)
    k1 = 2.0 * np.pi * max(1, round(k1 * Lx / (2.0 * np.pi))) / Lx
    k2 = np.sqrt(max(1.0 - k1 * k1, 1e-12))
    X, Y = np.meshgrid(x, y)
    theta = k1 * X + np.logaddexp(k2 * Y, -k2 * Y)   # log(2 cosh(k2 y))
    return np.cos(theta), k1, k2


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def core_row_oscillation(prof, x_sel, q_target, dx):
    """Relative oscillation amplitude near q_target and its wavelength."""
    p = prof - prof.mean()
    F = np.abs(np.fft.rfft(p))
    q = 2.0 * np.pi * np.fft.rfftfreq(p.size, d=dx)
    i0 = int(np.argmin(np.abs(q - q_target)))
    lo, hi = max(i0 - 1, 1), min(i0 + 2, len(F))
    i_pk = lo + int(np.argmax(F[lo:hi]))
    amp = 2.0 * F[i_pk] / p.size
    rel = amp / (np.abs(prof).mean() + 1e-15)
    wav = 2.0 * np.pi / q[i_pk] if q[i_pk] > 0 else np.inf
    return float(rel), float(wav)


# -----------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------

def run(cfg):
    out_root = Path(cfg.out) / cfg.field
    out_root.mkdir(parents=True, exist_ok=True)
    sigma = cfg.sigma
    rows = []
    ridge_panels = []

    x = np.linspace(-cfg.Lx / 2, cfg.Lx / 2, cfg.Nx, endpoint=False)
    y = np.linspace(-cfg.Ly / 2, cfg.Ly / 2, cfg.Ny, endpoint=False)
    dx = cfg.Lx / cfg.Nx
    ramp = build_rectangular_ramp_smooth(
        x, y, xmargin=cfg.xmargin, ymargin=cfg.ymargin,
        tanhscale=cfg.tanhscale)
    interior = ramp >= 0.999999
    row0 = int(np.argmin(np.abs(y)))
    x_sel = np.abs(x) < 0.35 * cfg.Lx

    for mu in cfg.mus:
        if cfg.field == "knee":
            u, k1, k2 = field_knee(mu, x, y, cfg.Lx)
        else:
            raise ValueError(f"unknown field {cfg.field}")
        mu_eff = k2
        kscale_tag = getattr(cfg, "kernel_scale", 1.0)
        print(f"\n=== {cfg.field} mu={mu} (effective {mu_eff:.4f}, "
              f"k1={k1:.4f}) ===")

        # --- microscopic energy, far-field baseline removed ---
        e_micro = micro_energy(u * ramp, cfg.Lx, cfg.Ly, cfg.R)
        far = interior & (np.abs(np.meshgrid(x, y)[1]) > 0.25 * cfg.Ly)
        e_micro = e_micro - np.median(e_micro[far])

        # --- adaptive extraction (auto kernel params from mu, or override) ---
        aspect, sharpen = derived_kernel_params_mu(mu_eff, sigma,
                                                   delta=cfg.leak_delta)
        if getattr(cfg, "aspect_override", None) is not None:
            aspect = cfg.aspect_override
        if getattr(cfg, "sharpen_override", None) is not None:
            sharpen = cfg.sharpen_override
        print(f"  kernel: aspect_max={aspect:.2f} sharpen={sharpen:.2f} "
             f"({'override' if getattr(cfg,'sharpen_override',None) or getattr(cfg,'aspect_override',None) else 'auto from mu'})")

        tag = (f"mu{mu:.2f}_sigma{sigma:.3f}_delta{cfg.leak_delta:.3f}"
               f"_kscale{kscale_tag:.2f}_gate-{getattr(cfg,'gate','legacy')}"
               f"_kind-{getattr(cfg,'kernel_kind','gaussian')}"
               f"_blend-{getattr(cfg,'blend','patch')}"
               f"_asp{aspect:.2f}_shp{sharpen:.2f}"
               f"_patch{cfg.patch}_niter{cfg.n_iter}")
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        kscale = getattr(cfg, "kernel_scale", 1.0)
        gate = getattr(cfg, "gate", "legacy")
        kkind = getattr(cfg, "kernel_kind", "gaussian")
        blend = getattr(cfg, "blend", "patch")
        ops = compute_uhu_ops_adaptive(
            u, x, y, sigma, SpectralDerivs, ramp,
            n_iter=cfg.n_iter, aspect_max=aspect, sharpen=sharpen,
            damping=cfg.damping, patch=cfg.patch, orient="slowness",
            scale=kscale, gate=gate, kernel_kind=kkind, blend=blend)

        # --- the two macroscopic averages of the SAME micro energy ---
        # both use sigma*kernel_scale, so a kernel_scale sweep moves the
        # iso baseline and the adaptive result on the same footing.
        e_iso = macro(e_micro, sigma * kscale, cfg.Lx, cfg.Ly)
        d1, d2 = slowness_director(ops["k1_orig"], ops["k2_orig"], sigma,
                                   cfg.Lx, cfg.Ly, SpectralDerivs, macro)
        coh = coherence_from_eigs(ops["lam1"], ops["lam2"])
        if blend == "pixel":
            e_ada = macro_aniso_field_v3_pixelblend(
                e_micro, k1=d1, k2=d2, coherence=coh, Lx=cfg.Lx, Ly=cfg.Ly,
                sigma_base=sigma, aspect_max=aspect, sharpen=sharpen,
                patch=cfg.patch, scale=kscale, gate=gate, kernel_kind=kkind)
        else:
            e_ada = macro_aniso_field_v2(
                e_micro, k1=d1, k2=d2, coherence=coh, Lx=cfg.Lx, Ly=cfg.Ly,
                sigma_base=sigma, aspect_max=aspect, sharpen=sharpen,
                patch=cfg.patch, scale=kscale, gate=gate, kernel_kind=kkind)

        # ground-truth ridge: x-average of the micro energy
        ridge_truth = e_micro[:, x_sel].mean(axis=1)

        # --- defect diagnostic on the RECOVERED field: J = det(grad k) ---
        # This is the same J that concentrates on point defects in the
        # theory (main text, self-duality discussion). The synthetic knee
        # is defect-free by construction, so J should be ~0 everywhere;
        # any structure here at the core is itself a residual extraction
        # artifact, not a physical dislocation.
        k1f, k2f = ops["k1_orig"], ops["k2_orig"]
        fx = SpectralDerivs(k1f, cfg.Lx, cfg.Ly, "x")
        fy = SpectralDerivs(k1f, cfg.Lx, cfg.Ly, "y")
        gx = SpectralDerivs(k2f, cfg.Lx, cfg.Ly, "x")
        gy = SpectralDerivs(k2f, cfg.Lx, cfg.Ly, "y")
        J_field = fx * gy - fy * gx

        # --- metrics ---
        rec = {"mu": mu, "mu_eff": mu_eff, "k1": k1,
               "L_PN": np.pi / k1, "sigma": sigma,
               "leak_delta": cfg.leak_delta, "kernel_scale": kscale,
               "gate": gate, "kernel_kind": kkind, "blend": blend,
               "aspect": aspect, "sharpen": sharpen}
        for name, f in (("micro", e_micro), ("iso", e_iso), ("ada", e_ada)):
            rel, wav = core_row_oscillation(f[row0, x_sel], x_sel,
                                            2.0 * k1, dx)
            # excess energy per unit core length (conservation check)
            _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
            per_len = float(_trapz(f[:, x_sel].mean(axis=1), y))
            rec[f"{name}_osc"] = rel
            rec[f"{name}_wav"] = wav
            rec[f"{name}_E_per_len"] = per_len
        ang = np.degrees(np.arctan2(
            np.abs(ops["k2_orig"][row0, x_sel]),
            np.abs(ops["k1_orig"][row0, x_sel]))).mean()
        rec["core_dir_deg"] = float(ang)
        rows.append(rec)
        print(f"  core-row osc (rel):  iso {rec['iso_osc']:.3f}  "
              f"ada {rec['ada_osc']:.3f}   blob wavelength iso "
              f"{rec['iso_wav']:.2f} vs L={rec['L_PN']:.2f}")
        print(f"  E per unit length:   micro {rec['micro_E_per_len']:.4f}  "
              f"iso {rec['iso_E_per_len']:.4f}  ada {rec['ada_E_per_len']:.4f}")
        print(f"  extracted core director angle: {ang:.1f} deg "
              f"(0 = correct, 90 = flipped)")

        # --- dashboard ---
        extent = [x[0], x[-1], y[0], y[-1]]
        fig, axs = plt.subplots(2, 4, figsize=(17.4, 6.2),
                                constrained_layout=True)
        vmax = np.percentile(e_iso[interior], 99.5)
        panels = [
            (axs[0, 0], np.ma.masked_where(~interior, u), "pattern",
             dict(cmap="gray")),
            (axs[0, 1], np.ma.masked_where(~interior, e_micro),
             "micro energy $e$", dict(cmap="magma")),
            (axs[0, 2], np.ma.masked_where(~interior, e_iso),
             "$\\langle e\\rangle$ isotropic", dict(cmap="magma", vmin=0,
                                                    vmax=vmax)),
            (axs[1, 0], np.ma.masked_where(~interior, e_ada),
             "$\\langle e\\rangle$ adaptive", dict(cmap="magma", vmin=0,
                                                   vmax=vmax)),
        ]
        for ax, f, title, kw in panels:
            im = ax.imshow(f, extent=extent, origin="lower", aspect="auto",
                           rasterized=True, **kw)
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.85)

        # --- quiver of the recovered wavevector, on top of the pattern ---
        ax = axs[0, 0]
        # fixed, sparse arrow grid regardless of Nx/Ny (~18 x 10 arrows)
        step_x = max(int(cfg.Nx / 18), 1)
        step_y = max(int(cfg.Ny / 10), 1)
        Xq, Yq = np.meshgrid(x, y)
        sl = (slice(0, cfg.Ny, step_y), slice(0, cfg.Nx, step_x))
        qmask = interior[sl]
        ax.quiver(Xq[sl][qmask], Yq[sl][qmask],
                  k1f[sl][qmask], k2f[sl][qmask],
                  color="tab:red", angles="xy", scale_units="xy",
                  scale=0.2, width=0.004, alpha=0.9)
        ax.set_title("pattern + recovered $\\vec{k}$ (adaptive)", fontsize=9)

        # --- J = det(grad k), the point-defect diagnostic on the ---
        # --- recovered field (should be ~0 everywhere: no true defects) ---
        ax = axs[0, 3]
        Jmax = np.percentile(np.abs(J_field[interior]), 99.5) + 1e-15
        im = ax.imshow(np.ma.masked_where(~interior, J_field), extent=extent,
                       origin="lower", aspect="auto", cmap="coolwarm",
                       vmin=-Jmax, vmax=Jmax, rasterized=True)
        ax.set_title("$J=\\det(\\nabla\\vec k)$, recovered field "
                     "(truth: 0)", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85)

        ax = axs[1, 1]
        ax.plot(x[x_sel], e_iso[row0, x_sel], "r-", lw=1.0, label="iso")
        ax.plot(x[x_sel], e_ada[row0, x_sel], "b-", lw=1.0, label="adaptive")
        for n in range(1, int(0.35 * cfg.Lx / (np.pi / k1)) + 1):
            for s in (-1, 1):
                ax.axvline(s * n * np.pi / k1, color="k", lw=0.4, alpha=0.3)
        ax.set_xlabel("$x$"); ax.set_ylabel("$\\langle e\\rangle$")
        ax.legend(fontsize=8)
        ax.set_title("along the core; gridlines at multiples of "
                     "$L=\\pi/k_1$", fontsize=9)

        ax = axs[1, 2]
        sm_truth = macro(e_micro, sigma, cfg.Lx, cfg.Ly)  # for scale only
        ax.plot(y, ridge_truth, "k-", lw=1.4, label="micro, x-averaged")
        ax.plot(y, e_iso[:, x_sel].mean(axis=1), "r--", lw=1.0, label="iso")
        ax.plot(y, e_ada[:, x_sel].mean(axis=1), "b-", lw=1.0,
                label="adaptive")
        ax.set_xlabel("$y$"); ax.set_ylabel("ridge profile")
        ax.legend(fontsize=8)
        ax.set_title("cross-core line-energy ridge", fontsize=9)

        # --- local coherence: the field that drives the per-patch ---
        # --- kernel shape (s_par, s_perp) inside macro_aniso_field_v2 ---
        ax = axs[1, 3]
        im = ax.imshow(np.ma.masked_where(~interior, coh), extent=extent,
                       origin="lower", aspect="auto", cmap="viridis",
                       vmin=0, vmax=1, rasterized=True)
        ax.set_title("local coherence $c$ (sets kernel shape "
                     "per patch)", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85)

        fig.suptitle(
            f"{cfg.field} mu={mu_eff:.2f}: the analytic knee is x-invariant, "
            "so any along-core oscillation in the averaged energy is "
            "estimator artifact; the adaptive kernel should flatten the "
            "core row while preserving the ridge.", fontsize=9)
        fig.savefig(out_dir / "dashboard.png", bbox_inches="tight", dpi=160)
        plt.close(fig)

        ridge_panels.append((mu_eff, ridge_truth,
                             e_iso[:, x_sel].mean(axis=1),
                             e_ada[:, x_sel].mean(axis=1)))

    # --- shared summary figure: ridges across mu ---
    fig, axs = plt.subplots(1, len(ridge_panels),
                            figsize=(3.3 * len(ridge_panels), 3.0),
                            constrained_layout=True, sharey=False)
    if len(ridge_panels) == 1:
        axs = [axs]
    for ax, (mu_eff, rt, ri, ra) in zip(axs, ridge_panels):
        ax.plot(y, rt, "k-", lw=1.3)
        ax.plot(y, ri, "r--", lw=0.9)
        ax.plot(y, ra, "b-", lw=0.9)
        ax.set_title(f"$\\mu={mu_eff:.2f}$", fontsize=9)
        ax.set_xlabel("$y$")
        ax.set_xlim(-4 * np.pi, 4 * np.pi)
    axs[0].set_ylabel("line-energy ridge")
    axs[0].legend(["micro (x-avg)", "iso", "adaptive"], fontsize=7)
    fig.suptitle("Line-energy ridge of the analytic knee across far-field "
                 "angle: black is the exact x-averaged micro energy, which "
                 "the averaged fields should reproduce.", fontsize=9)
    fig.savefig(out_root / "ridges_summary.png", bbox_inches="tight", dpi=160)
    plt.close(fig)

    # --- summary table ---
    keys = list(rows[0].keys())
    with open(out_root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_root}/")
    return rows


def main():
    p = argparse.ArgumentParser(
        description="Line-energy test of adaptive averaging on analytic knees")
    p.add_argument("--field", choices=["knee"], default="knee")
    p.add_argument("--out", type=str, default="results/knee_line_energy")
    p.add_argument("--mus", type=float, nargs="+",
                   default=[0.3, 0.5, 0.7, 0.9])
    p.add_argument("--R", type=float, default=0.5)
    p.add_argument("--Nx", type=int, default=512)
    p.add_argument("--Ny", type=int, default=256)
    p.add_argument("--Lx", type=float, default=40 * np.pi)
    p.add_argument("--Ly", type=float, default=20 * np.pi)
    p.add_argument("--sigma", type=float, default=np.pi / 2)
    p.add_argument("--n_iter", type=int, default=5)
    p.add_argument("--damping", type=float, default=0.5)
    p.add_argument("--patch", type=int, default=96)
    p.add_argument("--leak_delta", type=float, default=0.05)
    p.add_argument("--kernel_scale", type=float, default=1.0,
                   help="overall multiplier on the kernel footprint "
                        "(applied before aspect/sharpen shape); 1.0 "
                        "reproduces the previous fixed-scale behavior")
    p.add_argument("--gate", choices=["legacy", "core_aware"],
                   default="legacy",
                   help="legacy: anisotropy strongest at HIGH coherence, "
                        "relaxes to isotropic near a core. core_aware: "
                        "anisotropy strongest at LOW coherence (on the "
                        "core itself), near-isotropic in the far field.")
    p.add_argument("--kernel_kind", choices=["gaussian", "rect"],
                   default="gaussian",
                   help="gaussian: smooth rotated Gaussian support. rect: "
                        "hard-edged rotated box average.")
    p.add_argument("--aspect_override", type=float, default=None,
                   help="force aspect_max instead of deriving it from mu")
    p.add_argument("--sharpen_override", type=float, default=None,
                   help="force sharpen instead of deriving it from mu")
    p.add_argument("--blend", choices=["patch", "pixel"], default="patch",
                   help="patch: coherence gates shape via a per-patch "
                        "median (previous behavior). pixel: shape is "
                        "FIXED (no coherence dependence) and a separate "
                        "isotropic field is blended in per pixel using "
                        "full-resolution coherence -- decouples patch "
                        "size (needed for a stable spectral estimate) "
                        "from the resolution of the gating decision.")
    p.add_argument("--xmargin", type=float, default=0.05)
    p.add_argument("--ymargin", type=float, default=0.10)
    p.add_argument("--tanhscale", type=float, default=60.0)
    a = p.parse_args()
    run(a)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        class _Args:
            field      = "knee"
            out        = "results/knee_line_energy"
            mus        = [0.3, 0.5, 0.7, 0.85, 0.9, 0.95]
            R          = 0.5
            Nx, Ny     = 512, 256
            Lx, Ly     = 40 * np.pi, 20 * np.pi
            sigma      = np.pi / 2
            n_iter     = 5
            damping    = 0.5
            patch      = 96
            leak_delta = 0.02
            kernel_scale = 1.0
            gate       = "legacy"
            kernel_kind = "gaussian"
            blend      = "patch"
            aspect_override = None
            sharpen_override = None
            xmargin    = 0.05
            ymargin    = 0.05
            tanhscale  = 60.0
        run(_Args)
    else:
        main()
