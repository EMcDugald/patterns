# pipelines/pismen_line_energy_adaptive.py
"""
Companion to knee_line_energy_adaptive.py: the same microscopic-energy
/ isotropic-average / adaptive-average pipeline, applied to a finite
Pismen-Nepomnyashchy (PN) multi-dislocation chain instead of a
defect-free knee. Field construction (`pismen_on_sh_grid`) is ported
from pismen_fit_and_residuals_v2.py -- same formula, copied here rather
than imported so this script doesn't pull in that file's unrelated
skimage/paper_style plotting dependencies.

WHY THIS IS A DIFFERENT KIND OF TEST THAN THE KNEE SCRIPT

The knee is x-invariant, so the exact macroscopic answer is a smooth
ridge and ANY along-core oscillation in the averaged fields is estimator
artifact -- that's what knee_line_energy_adaptive.py checks for.
A dislocation chain is the opposite case: it is NOT x-invariant, and its
true microscopic energy SHOULD show real periodic peaks at each defect,
spaced at L = pi/k1 = pi/cos(alpha) (cos(alpha) = sqrt(1-mu^2), the same
topological-quantization spacing used throughout this project). That is
exactly the same wavenumber (2*k1) that shows up as a SPURIOUS artifact
on a defect-free knee at the same far-field angle -- which is the whole
reason this comparison matters: it's the direct test of whether the
extraction can tell a real dislocation chain apart from the leakage
artifact that mimics one. So here, `micro_osc` (oscillation of the TRUE
energy at 2*k1) is expected to be SUBSTANTIAL, not near zero, and the
question is whether `ada_osc`/`iso_osc` PRESERVE that real signal
(recovery ratio near 1) rather than smoothing it away (recovery ratio
near 0) -- the opposite failure mode from the knee script, where the
worry was the averaged fields showing oscillation that shouldn't be
there at all.

Physical motivation (why mu >= ~0.8 matters here specifically): per the
"zippers" / enlarged variational problem in the paper draft (Ercolani &
Venkataramani), the smooth self-dual knee is the energetically preferred
representation of a PGB only up to some far-field angle; beyond that,
a dislocation chain becomes the cheaper way to accommodate the same
far-field mismatch. So confirming that Pismen chains at mu = 0.8, 0.85,
0.9+ genuinely do NOT look like a smooth line-energy (unlike the knee at
the same mu) is a check that this construction is actually representing
the physically relevant defect-chain regime, not an accidental
near-duplicate of the knee.

FINITE-CHAIN CAVEAT (from Ed directly, keep in mind reading results)

The far-field tilt implied by spacing L = pi/k1 is only exactly correct
in the limit of infinitely many defects; pismen_on_sh_grid builds a
FINITE chain of `n_defects`, so the two or so periods nearest each end
of the chain will not show the idealized far-field behavior. Ed's own
empirical finding: n_defects=8 is enough that the middle few periods
look right. This script therefore restricts all metrics and the
along-core plot to a central window of `central_periods` periods around
x=0, NOT the whole chain -- the outer defects exist to give the central
ones a realistic environment, not to be analyzed themselves.

Usage
-----
python pismen_line_energy_adaptive_v3.py --out results/pismen_line_energy
(or run in PyCharm; edit _Args at the bottom)
"""

import argparse
import csv
import sys
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "src", _HERE / "src"):
    if _p.exists():
        sys.path.insert(0, str(_p))
sys.path.insert(0, str(_HERE))

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

# reuse the exact same microscopic energy and oscillation-metric code the
# knee script uses, rather than duplicating it (avoids the two scripts'
# energy formulas silently drifting apart)
try:
    from pipelines.knee_line_energy_adaptive import (
        micro_energy, core_row_oscillation,
    )
except ImportError:
    from knee_line_energy_adaptive import (
        micro_energy, core_row_oscillation,
    )

try:
    from sh_sims.core import integrate_sh
except ImportError:
    from core import integrate_sh

try:
    from paper_style import use_paper_style
    use_paper_style()
except ImportError:
    pass


# -----------------------------------------------------------------------
# Field construction (ported from pismen_fit_and_residuals_v2.py)
# -----------------------------------------------------------------------

def pismen_on_sh_grid(X, Y, x_defects, amp=1.0):
    """Exact PN multi-dislocation phase/pattern. Defects sit along Y=0 at
    the given X positions; far from the chain the phase approaches
    Y +- log(const), i.e. a tilted-plane-wave far field whose angle is
    set by the defect spacing (see module docstring)."""
    linear_phase = Y
    x_defects = np.asarray(x_defects, dtype=float)
    npts = len(x_defects)
    if npts == 0:
        theta = linear_phase.copy()
        return theta, amp * np.cos(theta)
    thetas = np.arange(npts + 1) * np.pi
    psi = 0.5 * (1 + np.exp(-thetas[-1]))
    for j in range(npts):
        delta = 0.5 * (np.exp(-thetas[j + 1]) - np.exp(-thetas[j]))
        arg = (X - x_defects[j]) / np.sqrt(2 * np.abs(Y) + 1e-8)
        psi += delta * special.erf(arg)
    theta = linear_phase - np.sign(Y) * np.log(psi)
    return theta, amp * np.cos(theta)


def field_pismen(mu, n_defects, dx_target, Ly, Ny, edge_margin_periods=2.0, stagger=False):
    """Build a centered, finite PN chain sized to fit n_defects at the
    spacing implied by mu, plus edge_margin_periods of buffer on each
    side for the ramp. Returns u, x, y, k1, L, x_defects, Lx, Nx.

    Domain is sized dynamically per mu (unlike the knee script's fixed
    Lx) because L = pi/k1 grows with mu -- a fixed domain that fits 8
    defects at mu=0.5 would not fit 8 at mu=0.9. Nx is chosen to hold
    the grid spacing dx_target constant across mu, so spectral accuracy
    doesn't quietly change across the sweep.
    """
    k1 = np.sqrt(max(1.0 - mu * mu, 1e-12))
    L = np.pi / k1
    span_periods = (n_defects - 1) + 2.0 * edge_margin_periods
    Lx = span_periods * L
    Nx = int(np.ceil(Lx / dx_target))
    Nx += Nx % 2  # keep it even

    x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
    if stagger:
        # offset by dy/2 so the grid never samples the singular y=0 row;
        # does not fix the resolution divergence (same power law) but
        # removes the global spectral ringing the y=0 row injects
        y = y + 0.5 * (Ly / Ny)
    x_defects = L * (np.arange(n_defects) - (n_defects - 1) / 2.0)
    X, Y = np.meshgrid(x, y)
    _, u = pismen_on_sh_grid(X, Y, x_defects, amp=1.0)
    return u, x, y, k1, L, x_defects, Lx, Nx


def relax_pismen(u0, ramp, R_scalar, Lx, Ly, h=0.1, tmax=2.0):
    """Short SH relaxation of the analytic PN outer solution. The
    amplitude heals the singular cores within O(1) time, turning the
    asymptotic construction into a true finite-energy SH dislocation
    chain at the same geometry. R is ramped to zero outside the window
    so the boundary neither grows a pattern nor pollutes the interior
    on this timescale. Returns the final frame only."""
    tt, uu, *_ = integrate_sh(u0 * ramp, R_scalar * ramp, Lx, Ly,
                              h=h, tmax=tmax, nsave=1, energy=False)
    return uu[..., -1]


# -----------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------

def run(cfg):
    out_root = Path(cfg.out) / "pismen"
    out_root.mkdir(parents=True, exist_ok=True)
    sigma = cfg.sigma
    dx_target = cfg.Lx_ref / cfg.Nx_ref
    rows = []

    for mu in cfg.mus:
        # u, x, y, k1, L, x_defects, Lx, Nx = field_pismen(
        #     mu, cfg.n_defects, dx_target, cfg.Ly, cfg.Ny,
        #     edge_margin_periods=cfg.edge_margin_periods)
        u, x, y, k1, L, x_defects, Lx, Nx = field_pismen(
            mu, cfg.n_defects, dx_target, cfg.Ly, cfg.Ny,
            edge_margin_periods=cfg.edge_margin_periods,
            stagger=getattr(cfg, "stagger", False))
        dx = Lx / Nx
        ramp = build_rectangular_ramp_smooth(
            x, y, xmargin=cfg.xmargin, ymargin=cfg.ymargin,
            tanhscale=cfg.tanhscale)
        if getattr(cfg, "relax_T", 0.0) > 0.0:
            print(f"  relaxing under SH for T={cfg.relax_T} "
                  f"(dt={cfg.relax_dt}) to heal the cores")
            u = relax_pismen(u, ramp, cfg.R, Lx, cfg.Ly,
                             h=cfg.relax_dt, tmax=cfg.relax_T)
        interior = ramp >= 0.999999
        row0 = int(np.argmin(np.abs(y)))
        half_window = 0.5 * cfg.central_periods * L
        x_sel = np.abs(x) < half_window
        central_defects = x_defects[np.abs(x_defects) < half_window]

        print(f"\n=== pismen mu={mu} (k1={k1:.4f}, L={L:.3f}, "
              f"Lx={Lx:.2f}, Nx={Nx}, {cfg.n_defects} defects, "
              f"{len(central_defects)} in the analysis window) ===")

        # --- microscopic energy, far-field baseline removed ---
        e_micro = micro_energy(u * ramp, Lx, cfg.Ly, cfg.R)
        far = interior & (np.abs(np.meshgrid(x, y)[1]) > 0.25 * cfg.Ly)
        e_micro = e_micro - np.median(e_micro[far])

        # --- adaptive extraction (auto kernel params from mu, or override) ---
        aspect, sharpen = derived_kernel_params_mu(mu, sigma,
                                                   delta=cfg.leak_delta)
        if getattr(cfg, "aspect_override", None) is not None:
            aspect = cfg.aspect_override
        if getattr(cfg, "sharpen_override", None) is not None:
            sharpen = cfg.sharpen_override
        print(f"  kernel: aspect_max={aspect:.2f} sharpen={sharpen:.2f}")

        kscale = getattr(cfg, "kernel_scale", 1.0)
        gate = getattr(cfg, "gate", "legacy")
        kkind = getattr(cfg, "kernel_kind", "gaussian")
        blend = getattr(cfg, "blend", "patch")

        # tag = (f"mu{mu:.2f}_sigma{sigma:.3f}_delta{cfg.leak_delta:.3f}"
        #        f"_kscale{kscale:.2f}_gate-{gate}_kind-{kkind}_blend-{blend}"
        #        f"_asp{aspect:.2f}_shp{sharpen:.2f}_nd{cfg.n_defects}"
        #        f"_patch{cfg.patch}_niter{cfg.n_iter}")
        tag = (f"mu{mu:.2f}_sigma{sigma:.3f}_delta{cfg.leak_delta:.3f}"
               f"_kscale{kscale:.2f}_gate-{gate}_kind-{kkind}_blend-{blend}"
               f"_asp{aspect:.2f}_shp{sharpen:.2f}_nd{cfg.n_defects}"
               f"_patch{cfg.patch}_niter{cfg.n_iter}"
               f"{'_stag' if getattr(cfg, 'stagger', False) else ''}"
               f"{f'_relax{cfg.relax_T:g}' if getattr(cfg, 'relax_T', 0.0) > 0 else ''}")
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        ops = compute_uhu_ops_adaptive(
            u, x, y, sigma, SpectralDerivs, ramp,
            n_iter=cfg.n_iter, aspect_max=aspect, sharpen=sharpen,
            damping=cfg.damping, patch=cfg.patch, orient="slowness",
            scale=kscale, gate=gate, kernel_kind=kkind, blend=blend)

        # --- the two macroscopic averages of the SAME micro energy ---
        e_iso = macro(e_micro, sigma * kscale, Lx, cfg.Ly)
        d1, d2 = slowness_director(ops["k1_orig"], ops["k2_orig"], sigma,
                                   Lx, cfg.Ly, SpectralDerivs, macro)
        coh = coherence_from_eigs(ops["lam1"], ops["lam2"])
        if blend == "pixel":
            e_ada = macro_aniso_field_v3_pixelblend(
                e_micro, k1=d1, k2=d2, coherence=coh, Lx=Lx, Ly=cfg.Ly,
                sigma_base=sigma, aspect_max=aspect, sharpen=sharpen,
                patch=cfg.patch, scale=kscale, gate=gate, kernel_kind=kkind)
        else:
            e_ada = macro_aniso_field_v2(
                e_micro, k1=d1, k2=d2, coherence=coh, Lx=Lx, Ly=cfg.Ly,
                sigma_base=sigma, aspect_max=aspect, sharpen=sharpen,
                patch=cfg.patch, scale=kscale, gate=gate, kernel_kind=kkind)

        # --- J = det(grad k) on the recovered field ---
        # UNLIKE the knee case, real defects are present here, so J is
        # EXPECTED to show real, localized spikes at each x_defect --
        # that's the signal, not an artifact. Whether those spikes land
        # where they should (at x_defects) and stay near zero elsewhere
        # is the actual test of whether the extraction can detect these
        # defects at all.
        k1f, k2f = ops["k1_orig"], ops["k2_orig"]
        fx = SpectralDerivs(k1f, Lx, cfg.Ly, "x")
        fy = SpectralDerivs(k1f, Lx, cfg.Ly, "y")
        gx = SpectralDerivs(k2f, Lx, cfg.Ly, "x")
        gy = SpectralDerivs(k2f, Lx, cfg.Ly, "y")
        J_field = fx * gy - fy * gx

        if getattr(cfg, "save_op", False):
            # isotropic control: aspect_max=1.0 pins the kernel circular,
            # so the iterate reduces to the isotropic estimator and one
            # iteration suffices (the kernel cannot change shape).
            ops_iso = compute_uhu_ops_adaptive(
                u, x, y, sigma, SpectralDerivs, ramp,
                n_iter=1, aspect_max=1.0, sharpen=1.0,
                damping=cfg.damping, patch=cfg.patch, orient="slowness",
                scale=kscale, gate=gate, kernel_kind=kkind, blend=blend)
            sh_meta = json.dumps({
                "kind": "pismen_analytic", "mu": mu, "k1": k1, "L": L,
                "n_defects": cfg.n_defects,
                "x_defects": [float(v) for v in x_defects],
                "relax_T": getattr(cfg, "relax_T", 0.0),
                "relax_dt": getattr(cfg, "relax_dt", None),
                "stagger": getattr(cfg, "stagger", False), "R": cfg.R})
            for label, o in (("adaptive", ops), ("iso", ops_iso)):
                np.savez_compressed(
                    out_dir / f"pismen_op_{label}.npz",
                    x=x, y=y, u=u[..., None], tt=np.array([0.0]),
                    ramp=ramp,
                    k1_orig=o["k1_orig"][..., None],
                    k2_orig=o["k2_orig"][..., None],
                    k1=o["k1_orig"][..., None],
                    k2=o["k2_orig"][..., None],
                    k=np.hypot(o["k1_orig"], o["k2_orig"])[..., None],
                    A=o["A"][..., None],
                    lam1=o["lam1"][..., None],
                    lam2=o["lam2"][..., None],
                    coh=coherence_from_eigs(o["lam1"], o["lam2"])[..., None],
                    uhu_meta_json=json.dumps({
                        "method": f"uhu_{label}", "sigma": sigma,
                        "aspect": aspect if label == "adaptive" else 1.0,
                        "sharpen": sharpen if label == "adaptive" else 1.0,
                        "gate": gate, "kernel_kind": kkind, "blend": blend,
                        "n_iter": cfg.n_iter if label == "adaptive" else 1}),
                    sh_meta_json=sh_meta)
                print(f"  OP file -> {out_dir / f'pismen_op_{label}.npz'}")

        # --- metrics ---
        # NOTE the interpretation flip from the knee script: micro_osc
        # here is the REAL oscillation amplitude of the true energy (the
        # signal this whole test is trying to confirm is present, not
        # suppress). ada_osc/iso_osc close to micro_osc (recovery ratio
        # near 1) means the averaging preserved real defect structure;
        # close to 0 means it smoothed the defects away.
        rec = {"mu": mu, "k1": k1, "L": L, "n_defects": cfg.n_defects,
               "central_periods": cfg.central_periods, "sigma": sigma,
               "leak_delta": cfg.leak_delta, "kernel_scale": kscale,
               "gate": gate, "kernel_kind": kkind, "blend": blend,
               "aspect": aspect, "sharpen": sharpen}
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        for name, f in (("micro", e_micro), ("iso", e_iso), ("ada", e_ada)):
            rel, wav = core_row_oscillation(f[row0, x_sel], x_sel,
                                            2.0 * k1, dx)
            per_len = float(_trapz(f[:, x_sel].mean(axis=1), y))
            rec[f"{name}_osc"] = rel
            rec[f"{name}_wav"] = wav
            rec[f"{name}_E_per_len"] = per_len
        rec["ada_recovery"] = rec["ada_osc"] / (rec["micro_osc"] + 1e-15)
        rec["iso_recovery"] = rec["iso_osc"] / (rec["micro_osc"] + 1e-15)
        rows.append(rec)
        print(f"  TRUE energy oscillation at 2*k1 (micro_osc): "
             f"{rec['micro_osc']:.3f}  <- should be substantial, "
             f"this IS the defect signal, not artifact")
        print(f"  recovery ratio (ada/micro): {rec['ada_recovery']:.3f}  "
             f"(iso/micro): {rec['iso_recovery']:.3f}  "
             f"(1.0 = fully preserved, 0.0 = averaged away)")
        print(f"  E per unit length:   micro {rec['micro_E_per_len']:.4f}  "
             f"iso {rec['iso_E_per_len']:.4f}  ada {rec['ada_E_per_len']:.4f}")

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
            for xd in x_defects:
                ax.axvline(xd, color="cyan", lw=0.5, alpha=0.5, ls="--")

        # --- quiver of the recovered wavevector, on top of the pattern ---
        ax = axs[0, 0]
        step_x = max(int(Nx / 30), 1)
        step_y = max(int(cfg.Ny / 10), 1)
        Xq, Yq = np.meshgrid(x, y)
        sl = (slice(0, cfg.Ny, step_y), slice(0, Nx, step_x))
        qmask = interior[sl]
        ax.quiver(Xq[sl][qmask], Yq[sl][qmask],
                  k1f[sl][qmask], k2f[sl][qmask],
                  color="tab:red", angles="xy", scale_units="xy",
                  scale=0.2, width=0.004, alpha=0.9)
        ax.set_title("pattern + recovered $\\vec{k}$; dashed lines = "
                     "true defect positions", fontsize=9)

        # --- J field: real spikes expected at x_defects ---
        ax = axs[0, 3]
        Jmax = np.percentile(np.abs(J_field[interior]), 99.5) + 1e-15
        im = ax.imshow(np.ma.masked_where(~interior, J_field), extent=extent,
                       origin="lower", aspect="auto", cmap="coolwarm",
                       vmin=-Jmax, vmax=Jmax, rasterized=True)
        ax.set_title("$J=\\det(\\nabla\\vec k)$: real spikes expected "
                     "AT dashed lines", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85)
        for xd in x_defects:
            ax.axvline(xd, color="k", lw=0.5, alpha=0.4, ls="--")

        # --- along-core RAW profile (not x-averaged): the real bump train ---
        ax = axs[1, 1]
        ax.plot(x[x_sel], e_micro[row0, x_sel], "k-", lw=1.2, label="micro")
        ax.plot(x[x_sel], e_iso[row0, x_sel], "r-", lw=1.0, label="iso")
        ax.plot(x[x_sel], e_ada[row0, x_sel], "b-", lw=1.0, label="adaptive")
        for xd in central_defects:
            ax.axvline(xd, color="k", lw=0.4, alpha=0.3)
        ax.set_xlabel("$x$"); ax.set_ylabel("$e$ along core")
        ax.legend(fontsize=8)
        ax.set_title("along the core (NOT x-averaged): peaks at "
                     "gridlines are REAL signal", fontsize=9)

        # --- cross-core profile, x-averaged over the central window ---
        # (a conservation check across an integer-ish number of periods,
        # not a "ground truth ridge" the way it is for the knee)
        ax = axs[1, 2]
        ax.plot(y, e_micro[:, x_sel].mean(axis=1), "k-", lw=1.4,
               label="micro, x-avg over central window")
        ax.plot(y, e_iso[:, x_sel].mean(axis=1), "r--", lw=1.0, label="iso")
        ax.plot(y, e_ada[:, x_sel].mean(axis=1), "b-", lw=1.0,
               label="adaptive")
        ax.set_xlabel("$y$"); ax.set_ylabel("mean cross-section")
        ax.legend(fontsize=8)
        ax.set_title("cross-core profile (conservation check, central "
                     "window only)", fontsize=9)

        # --- local coherence ---
        ax = axs[1, 3]
        im = ax.imshow(np.ma.masked_where(~interior, coh), extent=extent,
                       origin="lower", aspect="auto", cmap="viridis",
                       vmin=0, vmax=1, rasterized=True)
        ax.set_title("local coherence $c$", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85)

        fig.suptitle(
            f"pismen mu={mu:.2f}, {cfg.n_defects} defects (L={L:.2f}, "
            f"only the central {cfg.central_periods:.0f} periods are "
            "analyzed -- outer defects exist only to give the center a "
            "realistic far-field environment): unlike the knee, real "
            "along-core structure at 2*k1 IS expected here.", fontsize=9)
        fig.savefig(out_dir / "dashboard.png", bbox_inches="tight", dpi=160)
        plt.close(fig)

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
        description="Line-energy test of adaptive averaging on Pismen "
                    "multi-dislocation chains (companion to "
                    "knee_line_energy_adaptive_v3.py)")
    p.add_argument("--out", type=str, default="results/pismen_line_energy")
    p.add_argument("--mus", type=float, nargs="+",
                   default=[0.5, 0.7, 0.8, 0.85, 0.9])
    p.add_argument("--n_defects", type=int, default=8)
    p.add_argument("--central_periods", type=float, default=4.0,
                   help="width of the analysis window in units of L, "
                        "centered at x=0; must be well inside n_defects "
                        "periods to avoid finite-chain edge effects")
    p.add_argument("--edge_margin_periods", type=float, default=2.0,
                   help="extra buffer (in periods) beyond the outermost "
                        "defects, before the ramp starts to taper")
    p.add_argument("--R", type=float, default=0.5)
    p.add_argument("--Lx_ref", type=float, default=40 * np.pi,
                   help="reference domain width used only to fix the "
                        "grid spacing dx = Lx_ref/Nx_ref; actual Lx is "
                        "computed per mu from n_defects and the spacing")
    p.add_argument("--Nx_ref", type=int, default=512)
    p.add_argument("--Ly", type=float, default=20 * np.pi)
    p.add_argument("--Ny", type=int, default=256)
    p.add_argument("--sigma", type=float, default=np.pi / 2)
    p.add_argument("--n_iter", type=int, default=5)
    p.add_argument("--damping", type=float, default=0.5)
    p.add_argument("--patch", type=int, default=96)
    p.add_argument("--leak_delta", type=float, default=0.05)
    p.add_argument("--kernel_scale", type=float, default=1.0)
    p.add_argument("--gate", choices=["legacy", "core_aware"],
                   default="legacy")
    p.add_argument("--kernel_kind", choices=["gaussian", "rect"],
                   default="gaussian")
    p.add_argument("--aspect_override", type=float, default=None)
    p.add_argument("--sharpen_override", type=float, default=None)
    p.add_argument("--blend", choices=["patch", "pixel"], default="patch")
    p.add_argument("--xmargin", type=float, default=0.05)
    p.add_argument("--ymargin", type=float, default=0.05)
    p.add_argument("--tanhscale", type=float, default=60.0)
    p.add_argument("--relax_T", type=float, default=0.0,
                   help="if > 0, SH-relax the PN field this long first")
    p.add_argument("--relax_dt", type=float, default=0.1)
    p.add_argument("--stagger", action="store_true",
                   help="offset y grid by dy/2 to avoid the y=0 row")
    p.add_argument("--save_op", action="store_true",
                   help="save adaptive+iso OP npz files for SINDy")
    a = p.parse_args()
    run(a)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        class _Args:
            out         = "results/pismen_line_energy"
            mus         = [0.8, 0.85, 0.9]
            n_defects   = 8
            central_periods = 4.0
            edge_margin_periods = 2.0
            R           = 0.5
            Lx_ref, Nx_ref = 40 * np.pi, 512
            Ly, Ny      = 20 * np.pi, 256
            sigma       = np.pi/2
            n_iter      = 5
            damping     = 0.5
            patch       = 96
            leak_delta  = 0.02
            kernel_scale = 1.0
            gate        = "core_aware"
            kernel_kind = "gaussian"
            blend       = "patch"
            aspect_override = None
            sharpen_override = None
            xmargin     = 0.05
            ymargin     = 0.05
            tanhscale   = 60.0
            relax_T = 1.0
            relax_dt = 0.1
            stagger = True
            save_op = True
        run(_Args)
    else:
        main()
