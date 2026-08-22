# pipelines/run_kuhu_auto_gate.py
"""
Auto-gated adaptive averaging applied to the uHu TENSOR itself: the
downstream step. adaptive_auto_gate.py validated the kernel prescription
on the scalar energy density; this script closes the loop by using the
same gate to average the four fields the extraction actually needs
(u*u_xx, u*u_xy, u*u_yy, u*u) before the eigen-decomposition, producing
k fields intended to (i) suppress the 2 k1 corrugation on defect-free
knees, (ii) preserve genuine dislocation signatures on chains, and
(iii) yield a core phase theta(x) = integral |k1| dx that is LINEAR on a
knee, removing the staircase / end-bulge artifact that currently makes
sharp knees read as defect-bearing in the phase pipeline.

Design decisions, stated so they can be argued with:

1. The gate is built ONCE, from the raw pattern plus the isotropic seed
   pass, exactly as in adaptive_auto_gate.run_case (core path from seed
   coherence, quantized circulation |C|/2pi with automatic tensor /
   Riesz-director selection, line score ls, slowness director, core
   mask). No iteration: the gated quantity is a quantized topological
   charge and does not improve under refinement, and single-pass keeps
   the construction non-circular. If the auto-gated k field is ever fed
   back to re-seed the slowness director, that is a second, separate
   pass, not a loop hidden inside this one.

2. The same smoother instance is applied to all four tensor fields.
   The moment algebra is filter-shape-agnostic (orthogonality of the
   phase average, not the kernel, is what de-mixes harmonics), so the
   eigenstructure algebra downstream is unchanged; only estimator
   quality differs. Using one smoother for all components keeps the
   tensor exactly symmetric and keeps -lam1/<u^2> a ratio of
   consistently averaged quantities.

3. Output schema matches compute_uhu_ops_smoother / run_kuhu_zigzag
   (k, A, k1_orig/k2_orig via eigenvectors, lam1, lam2) plus coh, the
   gate profile C(x), line score, and per-patch kernel records, so the
   existing phase pipeline, J diagnostics, and midline batteries can
   consume the npz unchanged.

Self-test (default when run without a real input): exact Hopf-Cole knee
and relaxed PN chain at the requested mus, comparing isotropic vs
auto-gated extraction on four acceptance metrics per case:

  knee:  (a) relative core-row oscillation of |k1| at 2 k1 (want: auto
             much smaller than iso),
         (b) median core |k| vs the exact value k1 (want: closer),
         (c) RMS residual of theta_core(x) = cumulative integral of
             |k1| dx from its linear fit (the phase-linearity check;
             this is the quantity whose failure produces the knee
             staircase),
  chain: (d) preservation of the defect signature: the contrast of
             chi = lam2/|lam1| between defect sites and inter-defect
             midpoints (want: auto comparable to iso, NOT smeared away).

Per case a dashboard is written: |k| iso, |k| auto, chi auto, core-row
|k1| profiles with the exact value, and the theta residual panel.

patch guidance: the Hann patch must contain the elongated kernel;
patch side length in pixels should exceed ~4 * aspect * sigma / dx.
At sigma = pi/2, dx = 40pi/512 and aspect ~ 4 this means patch >= ~103,
so 128 is the safe default here (the patch=32 runs truncate the kernel
and replace the 2 k1 corrugation with a longer-wavelength overlap-add
beat).

Usage
-----
python run_kuhu_auto_gate.py --out results/kuhu_auto
(or run in PyCharm; edit _Args at the bottom)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "src", _HERE / "src", _HERE.parent):
    if _p.exists():
        sys.path.insert(0, str(_p))

try:
    from adaptive_auto_gate_patched import (
        field_knee, field_pismen, relax_sh, core_path,
        circulation_profile, line_score_from_circulation,
        riesz_gate_field, macro_aniso_field_auto, core_row_oscillation)
except ImportError:
    from adaptive_auto_gate import (
        field_knee, field_pismen, relax_sh, core_path,
        circulation_profile, line_score_from_circulation,
        riesz_gate_field, macro_aniso_field_auto, core_row_oscillation)
import inspect as _inspect
_MAFA_KW = set(_inspect.signature(macro_aniso_field_auto).parameters)

def _mafa(*args, **kw):
    """Call macro_aniso_field_auto passing only kwargs it supports, so
    this runner works against both the original and the patched
    adaptive_auto_gate (the original lacks sharpen_weight)."""
    return macro_aniso_field_auto(
        *args, **{k: v for k, v in kw.items() if k in _MAFA_KW})
try:
    from utils.spectral import macro, SpectralDerivs
except ImportError:
    from spectral import macro, SpectralDerivs
try:
    from utils.geometry import build_rectangular_ramp_smooth
except ImportError:
    from geometry import build_rectangular_ramp_smooth
try:
    from utils.anisotropic_macro import coherence_from_eigs
except ImportError:
    from anisotropic_macro import coherence_from_eigs
try:
    from op_extract.uhu_smoother import compute_uhu_ops_smoother
except ImportError:
    from uhu_smoother import compute_uhu_ops_smoother
try:
    from op_extract.uhu_adaptive import (slowness_director,
                                         derived_kernel_params)
except ImportError:
    from uhu_adaptive import slowness_director, derived_kernel_params
try:
    from paper_style import use_paper_style
    use_paper_style()
except ImportError:
    pass


# -----------------------------------------------------------------------
# Gate construction (factored from adaptive_auto_gate.run_case)
# -----------------------------------------------------------------------

def build_auto_gate(u, x, y, ramp, sigma, cfg, L, k1_far, k2_far,
                    core_row=None):
    """One seed pass plus the topological gate; returns everything the
    auto-gated smoother needs, with no reference to any downstream
    average (non-circular by construction).

    core_row: optional row index around which to search for the core
    (priority: explicit cfg value > gb_row_in_crop metadata > global
    coherence search). Matters for files with more than one grain
    boundary in frame: the coherence argmin over the central band
    would otherwise pick whichever core it likes."""
    Lx, Ly = x[-1] - x[0] + (x[1] - x[0]), y[-1] - y[0] + (y[1] - y[0])
    Nx = x.size
    dx = Lx / Nx
    interior = ramp >= 0.99

    iso_sm = lambda a: macro(a, sigma, Lx, Ly)
    seed = compute_uhu_ops_smoother(u, x, y, iso_sm, SpectralDerivs, ramp)
    coh = coherence_from_eigs(seed["lam1"], seed["lam2"])
    d1, d2 = slowness_director(seed["k1_orig"], seed["k2_orig"], sigma,
                               Lx, Ly, SpectralDerivs, macro)

    if core_row is not None:
        dyg = Ly / y.size
        hw = max(6, int(round(3.0 * sigma / dyg)))
        j0 = max(0, int(core_row) - hw)
        j1 = min(y.size, int(core_row) + hw + 1)
        rows_c = j0 + np.argmin(coh[j0:j1, :], axis=0)
    else:
        rows_c, _ = core_path(coh, y, band_frac=0.35)
    half_w = max(4, int(round(cfg.gate_window_L * L / dx / 2)))
    rows_s = np.round(np.convolve(
        rows_c, np.ones(2 * half_w + 1) / (2 * half_w + 1),
        mode="same")).astype(int)

    use_riesz = L > cfg.L_over_sigdir_min * cfg.sigma_dir
    if use_riesz:
        # v2 gate: flip-free Riesz director times the flip-free
        # magnitude q^2 = g.T (-T) g / <u^2> (the seed |k| is corrupted
        # near the core above mu*, injecting spurious winding).
        g1, g2 = riesz_gate_field(u, Lx, Ly, cfg.sigma_dir, macro)
        ur = u * ramp
        Txx = iso_sm(u * SpectralDerivs(ur, Lx, Ly, "xx"))
        Txy = iso_sm(u * SpectralDerivs(ur, Lx, Ly, "xy"))
        Tyy = iso_sm(u * SpectralDerivs(ur, Lx, Ly, "yy"))
        uu = iso_sm(u * u * ramp)
        uu = np.where(np.abs(uu) < 1e-12, 1.0, uu)
        q = np.sqrt(np.maximum(
            -(g1 * g1 * Txx + 2 * g1 * g2 * Txy + g2 * g2 * Tyy) / uu,
            0.0))
        C = circulation_profile(q * g1, q * g2, x, y, rows_s, L)
        gate_used = "riesz_director_proj"
    else:
        gate_sm = lambda a: macro(a, cfg.sigma_gate, Lx, Ly)
        gseed = compute_uhu_ops_smoother(u, x, y, gate_sm,
                                         SpectralDerivs, ramp)
        C = circulation_profile(gseed["k1_orig"], gseed["k2_orig"],
                                x, y, rows_s, L)
        gate_used = "tensor"
    ls = line_score_from_circulation(C, cfg.circ_lo, cfg.circ_hi)

    cmin = float(coh[interior].min())
    cmed = float(np.median(coh[interior]))
    cth = cmin + cfg.core_coh_frac * (cmed - cmin)
    Y = np.meshgrid(x, y)[1]
    y_core = y[rows_s][None, :]
    band = np.abs(Y - y_core) < cfg.core_band_sigma * sigma
    core_mask = (coh < cth) & band
    ls2d = np.where(band, ls[None, :], 0.0)

    aspect, sharpen = derived_kernel_params(k1_far, k2_far, sigma,
                                            delta=cfg.leak_delta)
    return dict(seed=seed, coh=coh, d1=d1, d2=d2, rows_s=rows_s, C=C,
                ls=ls, ls2d=ls2d, core_mask=core_mask, interior=interior,
                gate_used=gate_used, aspect=aspect, sharpen=sharpen,
                Lx=Lx, Ly=Ly)


def compute_uhu_ops_auto(u, x, y, ramp, sigma, cfg, gate,
                         record_patches=False):
    """uHu extraction with the auto-gated smoother: identical algebra to
    compute_uhu_ops_smoother, smoother swapped. Returns the standard
    schema plus the gate fields."""
    g = gate
    patches = [] if record_patches else None

    def auto_sm(arr):
        return _mafa(
            arr, g["d1"], g["d2"], g["seed"]["k1_orig"],
            g["seed"]["k2_orig"], g["coh"], g["core_mask"], g["ls2d"],
            g["Lx"], g["Ly"], sigma, g["aspect"], g["sharpen"],
            patch=cfg.patch, record=patches, ramp_mask=g["interior"],
            sharpen_weight=getattr(cfg, "sharpen_weight", "coh"))

    ops = compute_uhu_ops_smoother(u, x, y, auto_sm, SpectralDerivs, ramp)
    ops["coh_seed"] = g["coh"]
    ops["C_core"] = g["C"]
    ops["ls_core"] = g["ls"]
    ops["gate_used"] = g["gate_used"]
    if record_patches:
        ops["patches"] = patches
    return ops


# -----------------------------------------------------------------------
# Acceptance metrics
# -----------------------------------------------------------------------

def theta_residual(k1_field, rows_s, x, x_sel):
    """theta_core(x) = cumulative integral of |k1| along the smoothed
    core path; RMS residual from its linear fit over the selected
    window. Linear on an exact knee; staircase artifact shows up here."""
    prof = np.abs(k1_field[rows_s, np.arange(x.size)])
    dxs = np.gradient(x)
    theta = np.cumsum(prof * dxs)
    xs, ts = x[x_sel], theta[x_sel]
    A = np.vstack([xs, np.ones_like(xs)]).T
    coef, *_ = np.linalg.lstsq(A, ts, rcond=None)
    resid = ts - A @ coef
    return float(np.sqrt(np.mean(resid ** 2))), xs, resid, float(coef[0])


def chi_defect_contrast(lam1, lam2, rows_s, x, x_defects, L):
    """chi = lam2/|lam1| sampled on the core at defect sites vs
    inter-defect midpoints; contrast = median(on) - median(off)."""
    chi = np.abs(lam2) / (np.abs(lam1) + 1e-12)
    prof = chi[rows_s, np.arange(x.size)]
    def sample(xs):
        idx = [int(np.argmin(np.abs(x - xv))) for xv in xs]
        return np.array([prof[i] for i in idx])
    on = sample(x_defects)
    off = sample(x_defects[:-1] + 0.5 * L)
    return float(np.median(on) - np.median(off)), prof


# -----------------------------------------------------------------------
# Self-test driver
# -----------------------------------------------------------------------

def run_case(cfg, field_name, mu, out_root, rows):
    sigma = cfg.sigma
    if field_name == "knee":
        x = np.linspace(-cfg.Lx / 2, cfg.Lx / 2, cfg.Nx, endpoint=False)
        y = np.linspace(-cfg.Ly / 2, cfg.Ly / 2, cfg.Ny, endpoint=False)
        u, k1, k2 = field_knee(mu, x, y, cfg.Lx)
        Lx, Nx = cfg.Lx, cfg.Nx
        L = np.pi / k1
        x_defects = np.array([])
        relax_T = cfg.relax_T_knee
    else:
        dx_target = cfg.Lx / cfg.Nx
        u, x, y, k1, L, x_defects, Lx, Nx = field_pismen(
            mu, cfg.n_defects, dx_target, cfg.Ly, cfg.Ny,
            edge_margin_periods=cfg.edge_margin_periods)
        k2 = mu
        relax_T = cfg.relax_T_pismen
    dx = Lx / Nx
    ramp = build_rectangular_ramp_smooth(
        x, y, xmargin=cfg.xmargin, ymargin=cfg.ymargin,
        tanhscale=cfg.tanhscale)
    if relax_T > 0:
        R_field = -cfg.R_out + (cfg.R + cfg.R_out) * ramp
        u = relax_sh(u * ramp, R_field, Lx, cfg.Ly, relax_T,
                     h=cfg.relax_h)

    out_dir = out_root / field_name / f"mu{mu:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {field_name} mu={mu} (k1={k1:.4f}, L={L:.2f}) ===")

    gate = build_auto_gate(u, x, y, ramp, sigma, cfg, L, k1, k2)
    iso_sm = lambda a: macro(a, sigma, Lx, cfg.Ly)
    ops_iso = compute_uhu_ops_smoother(u, x, y, iso_sm, SpectralDerivs,
                                       ramp)
    ops_auto = compute_uhu_ops_auto(u, x, y, ramp, sigma, cfg, gate)

    # Tangent-projected along-core wavenumber: on a line-like core the
    # tensor is genuinely rank-two (two-mode band) and its dominant
    # eigenvector legitimately points ACROSS the core for mu > 1/sqrt2;
    # the Cross-Newell core slope k1 lives in the projection
    # t.T T t = <u u_tt>. With the core horizontal, t = x-hat:
    # k_along = sqrt(-<u u_xx>/<u^2>). No eigen decision, no flip.
    u_xx = SpectralDerivs(u * ramp, Lx, cfg.Ly, "xx")

    def auto_sm_plain(arr):
        g = gate
        return _mafa(
            arr, g["d1"], g["d2"], g["seed"]["k1_orig"],
            g["seed"]["k2_orig"], g["coh"], g["core_mask"], g["ls2d"],
            g["Lx"], g["Ly"], sigma, g["aspect"], g["sharpen"],
            patch=cfg.patch, ramp_mask=g["interior"],
            sharpen_weight=getattr(cfg, "sharpen_weight", "coh"))

    k_along = {}
    for name, sm in (("iso", iso_sm), ("auto", auto_sm_plain)):
        Txx = sm(u * u_xx)
        uu = sm(u * u * ramp)
        uu = np.where(np.abs(uu) < 1e-12, 1.0, uu)
        k_along[name] = np.sqrt(np.maximum(-Txx / uu, 0.0))

    rows_s = gate["rows_s"]
    x_sel = np.abs(x) < (0.5 * cfg.central_periods * L
                         if field_name == "pismen" else 0.35 * Lx)
    rec = dict(field=field_name, mu=mu, k1=k1, L=L,
               gate_used=gate["gate_used"], aspect=gate["aspect"],
               sharpen=gate["sharpen"],
               C_med=float(np.median(gate["C"][x_sel])),
               ls_med=float(np.median(gate["ls"][x_sel])))

    fig, axs = plt.subplots(2, 3, figsize=(13.6, 6.4),
                            constrained_layout=True)
    extent = [x[0], x[-1], y[0], y[-1]]
    interior = gate["interior"]
    for ax, ops, name in ((axs[0, 1], ops_iso, "iso"),
                          (axs[0, 2], ops_auto, "auto")):
        kmap = np.hypot(ops["k1_orig"], ops["k2_orig"])
        im = ax.imshow(np.ma.masked_where(~interior, kmap), extent=extent,
                       origin="lower", aspect="auto", cmap="viridis",
                       vmin=0, vmax=1.2, rasterized=True)
        ax.set_title(f"$|k|$ {name}", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85)
    ax = axs[0, 0]
    ax.imshow(np.ma.masked_where(~interior, u), extent=extent,
              origin="lower", aspect="auto", cmap="gray", rasterized=True)
    ax.set_title("pattern" + (" (relaxed)" if relax_T > 0 else ""),
                 fontsize=9)

    for name, ops in (("iso", ops_iso), ("auto", ops_auto)):
        prof = np.abs(ops["k1_orig"][rows_s, np.arange(x.size)])
        rel, wav = core_row_oscillation(prof[x_sel], 2.0 * k1, dx)
        kmap = np.hypot(ops["k1_orig"], ops["k2_orig"])
        kcore = float(np.median(kmap[rows_s, np.arange(x.size)][x_sel]))
        rms, xs, resid, slope = theta_residual(ops["k1_orig"], rows_s,
                                               x, x_sel)
        ang = np.degrees(np.arctan2(
            np.abs(ops["k2_orig"][rows_s, np.arange(x.size)][x_sel]),
            np.abs(ops["k1_orig"][rows_s, np.arange(x.size)][x_sel])))
        rec[f"{name}_k1_osc"] = rel
        rec[f"{name}_kcore_med"] = kcore
        rec[f"{name}_core_angle"] = float(np.median(ang))
        rec[f"{name}_theta_rms"] = rms
        rec[f"{name}_theta_slope"] = slope
        ka_prof = k_along[name][rows_s, np.arange(x.size)]
        ka_rel, _ = core_row_oscillation(ka_prof[x_sel], 2.0 * k1, dx)
        ka_rms, ka_xs, ka_res, ka_slope = theta_residual(
            k_along[name], rows_s, x, x_sel)
        rec[f"{name}_kalong_med"] = float(np.median(ka_prof[x_sel]))
        rec[f"{name}_kalong_osc"] = ka_rel
        rec[f"{name}_kalong_theta_rms"] = ka_rms
        rec[f"{name}_kalong_slope"] = ka_slope
        axs[1, 0].plot(x[x_sel], prof[x_sel], lw=1.0,
                       color=("r" if name == "iso" else "b"), label=name)
        axs[1, 0].plot(x[x_sel], ka_prof[x_sel], lw=0.9, ls="--",
                       color=("r" if name == "iso" else "b"),
                       label=f"{name} $k_\\parallel$")
        axs[1, 1].plot(xs, resid, lw=1.0,
                       color=("r" if name == "iso" else "b"), label=name)
        axs[1, 1].plot(ka_xs, ka_res, lw=0.9, ls="--",
                       color=("r" if name == "iso" else "b"),
                       label=f"{name} $k_\\parallel$")
        if field_name == "pismen":
            contrast, chi_prof = chi_defect_contrast(
                ops["lam1"], ops["lam2"], rows_s, x, x_defects, L)
            rec[f"{name}_chi_contrast"] = contrast
            axs[1, 2].plot(x[x_sel], chi_prof[x_sel], lw=1.0,
                           color=("r" if name == "iso" else "b"),
                           label=name)
    axs[1, 0].axhline(k1, color="k", lw=0.6, ls="--", label="exact $k_1$")
    axs[1, 0].set_title("core-row $|k_1|$", fontsize=9)
    axs[1, 0].set_xlabel("$x$"); axs[1, 0].legend(fontsize=7)
    axs[1, 1].set_title("core phase $\\int |k_1|\\,dx$: residual from "
                        "linear fit", fontsize=9)
    axs[1, 1].set_xlabel("$x$"); axs[1, 1].legend(fontsize=7)
    if field_name == "pismen":
        for xm in x_defects:
            if x[x_sel].min() < xm < x[x_sel].max():
                axs[1, 2].axvline(xm, color="k", lw=0.4, alpha=0.3)
        axs[1, 2].set_title("core-row $\\chi=\\lambda_2/|\\lambda_1|$ "
                            "(gridlines = true defects)", fontsize=9)
        axs[1, 2].set_xlabel("$x$"); axs[1, 2].legend(fontsize=7)
    else:
        chi = np.abs(ops_auto["lam2"]) / (np.abs(ops_auto["lam1"]) + 1e-12)
        im = axs[1, 2].imshow(np.ma.masked_where(~interior, chi),
                              extent=extent, origin="lower",
                              aspect="auto", cmap="magma", vmin=0,
                              vmax=0.5, rasterized=True)
        axs[1, 2].set_title("$\\chi$ auto (two-mode diagnostic)",
                            fontsize=9)
        fig.colorbar(im, ax=axs[1, 2], shrink=0.85)
    fig.suptitle(
        f"{field_name} mu={mu:.2f}: auto-gated uHu tensor "
        f"(gate={gate['gate_used']}, |C| med {rec['C_med']:.2f}, "
        f"ls med {rec['ls_med']:.2f}, patch={cfg.patch})", fontsize=9)
    fig.savefig(out_dir / "kuhu_dashboard.png", bbox_inches="tight",
                dpi=150)
    plt.close(fig)

    rows.append(rec)
    if field_name == "knee":
        print(f"  |k1| corrugation @2k1: iso {rec['iso_k1_osc']:.3f} -> "
              f"auto {rec['auto_k1_osc']:.3f}")
        print(f"  core |k| median (exact {k1:.3f}): iso "
              f"{rec['iso_kcore_med']:.3f} -> auto "
              f"{rec['auto_kcore_med']:.3f}")
        print(f"  theta linearity RMS: iso {rec['iso_theta_rms']:.4f} -> "
              f"auto {rec['auto_theta_rms']:.4f}   (slopes "
              f"{rec['iso_theta_slope']:.3f} / "
              f"{rec['auto_theta_slope']:.3f}, exact {k1:.3f})")
        print(f"  core direction angle (0 = correct, 90 = flipped): iso "
              f"{rec['iso_core_angle']:.1f} -> auto "
              f"{rec['auto_core_angle']:.1f} deg")
        print(f"  PROJECTION k_along (exact {k1:.3f}): iso "
              f"{rec['iso_kalong_med']:.3f} (osc {rec['iso_kalong_osc']:.3f},"
              f" slope {rec['iso_kalong_slope']:.3f}) -> auto "
              f"{rec['auto_kalong_med']:.3f} (osc "
              f"{rec['auto_kalong_osc']:.3f}, slope "
              f"{rec['auto_kalong_slope']:.3f}, theta RMS "
              f"{rec['auto_kalong_theta_rms']:.4f})")
    else:
        print(f"  chi defect contrast: iso "
              f"{rec.get('iso_chi_contrast', np.nan):.3f} -> auto "
              f"{rec.get('auto_chi_contrast', np.nan):.3f} "
              f"(want auto comparable, not smeared)")
        print(f"  theta RMS (chain, staircase expected): iso "
              f"{rec['iso_theta_rms']:.3f} auto "
              f"{rec['auto_theta_rms']:.3f}")

    np.savez_compressed(
        out_dir / "op_auto.npz",
        x=x, y=y, u=u, ramp=ramp,
        k=ops_auto["k"], A=ops_auto["A"],
        k1=ops_auto["k1_orig"], k2=ops_auto["k2_orig"],
        lam1=ops_auto["lam1"], lam2=ops_auto["lam2"],
        coh=gate["coh"], C_core=gate["C"], ls_core=gate["ls"],
        k_along_iso=k_along["iso"], k_along_auto=k_along["auto"],
        uhu_meta_json=json.dumps(dict(
            method="uhu_auto_gate", sigma=sigma, patch=cfg.patch,
            aspect=gate["aspect"], sharpen=gate["sharpen"],
            gate_used=gate["gate_used"], mu=mu)))
    return rec



# -----------------------------------------------------------------------
# Data mode: real SH .npz input (same contract as run_kuhu_zigzag.py)
# -----------------------------------------------------------------------

def load_sh_npz(sh_path):
    """Standardised SH .npz: x, y, u (Ny, Nx, Nt) with a transpose
    guard for (Nt, Ny, Nx), tt, metadata_json carrying mu. Cloned from
    run_kuhu_zigzag.load_sh_npz so both runners accept the same files."""
    sh_path = Path(sh_path)
    data = np.load(sh_path, allow_pickle=True)
    x, y = data["x"], data["y"]
    u = data["u"]
    if u.ndim != 3:
        raise ValueError(f"{sh_path.name}: expected u ndim=3, got "
                         f"{u.shape}")
    if u.shape[0] == len(y) and u.shape[1] == len(x):
        pass
    elif u.shape[1] == len(y) and u.shape[2] == len(x):
        u = np.transpose(u, (1, 2, 0))
    else:
        raise ValueError(f"{sh_path.name}: cannot reconcile u shape "
                         f"{u.shape} with x ({len(x)},) y ({len(y)},)")
    tt = data["tt"] if "tt" in data else None
    metadata_json = None
    if "metadata_json" in data:
        raw = data["metadata_json"]
        metadata_json = raw.item() if hasattr(raw, "item") else str(raw)
    mu = None
    if metadata_json:
        try:
            mu = json.loads(metadata_json).get("mu")
        except Exception:
            pass
    if mu is None and "mu" in data:
        mu = float(data["mu"])
    return dict(x=x, y=y, u=u, tt=tt, mu=mu,
                metadata_json=metadata_json, source_path=str(sh_path))


def process_sh_file(sh_path, out_dir, cfg):
    """Auto-gated uHu extraction on one SH time series.

    Per frame: isotropic seed pass, topological gate (automatic
    tensor / riesz_director_proj selection), auto-gated smoothing of
    the four tensor fields, eigen extraction, and the tangent-projected
    k_along = sqrt(-<u u_xx>/<u^2>) with the same auto smoother.

    Output npz matches run_kuhu_zigzag exactly (x, y, u, tt, ramp, k,
    A, k1, k2, lam1, lam2, uhu_meta_json, sh_meta_json) so the phase
    pipeline, J diagnostics, and midline batteries consume it
    unchanged, plus the nucleation-facing extras: coh (Ny, Nx, Nt),
    k_along (Ny, Nx, Nt), C_core (Nt, Nx), ls_core (Nt, Nx),
    ycore_idx (Nt, Nx) and gate_used (Nt) as strings.
    """
    sh_path = Path(sh_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sh = load_sh_npz(sh_path)
    x, y, u_all, tt = sh["x"], sh["y"], sh["u"], sh["tt"]
    mu = cfg.mu_override if getattr(cfg, "mu_override", None) else sh["mu"]
    if mu is None:
        raise ValueError(
            f"{sh_path.name}: no mu in metadata; pass --mu_override")
    mu = float(mu)
    k1_far = np.sqrt(max(1.0 - mu * mu, 1e-12))
    k2_far = mu
    L = np.pi / k1_far
    sigma = cfg.sigma
    Ny, Nx, Nt_all = u_all.shape
    frames = list(range(0, Nt_all, max(1, cfg.frame_stride)))
    if getattr(cfg, "max_frames", 0):
        frames = frames[: cfg.max_frames]
    Nt = len(frames)
    Lx = x[-1] - x[0] + (x[1] - x[0])
    Ly = y[-1] - y[0] + (y[1] - y[0])
    ramp = build_rectangular_ramp_smooth(
        x, y, xmargin=cfg.xmargin, ymargin=cfg.ymargin,
        tanhscale=cfg.tanhscale)

    F = {k: np.empty((Ny, Nx, Nt))
         for k in ("k", "A", "k1", "k2", "lam1", "lam2")}
    coh = np.empty((Ny, Nx, Nt), dtype=np.float32)
    k_along = np.empty((Ny, Nx, Nt), dtype=np.float32)
    C_core = np.empty((Nt, Nx))
    ls_core = np.empty((Nt, Nx))
    ycore_idx = np.empty((Nt, Nx), dtype=np.int32)
    gates = []

    core_row = getattr(cfg, "core_row", None)
    if core_row is None and sh["metadata_json"]:
        try:
            gb = json.loads(sh["metadata_json"]).get("gb_row_in_crop")
            core_row = int(gb) if gb is not None else None
        except Exception:
            core_row = None
    if core_row is not None:
        print(f"  core row from metadata/cfg: {core_row}")

    for it, t in enumerate(frames):
        u = u_all[:, :, t]
        print(f"  auto-gate frame {it + 1}/{Nt} (t index {t}, "
              f"{sh_path.name})")
        gate = build_auto_gate(u, x, y, ramp, sigma, cfg, L,
                               k1_far, k2_far, core_row=core_row)
        ops = compute_uhu_ops_auto(u, x, y, ramp, sigma, cfg, gate)
        for k_out, k_in in (("k", "k"), ("A", "A"),
                            ("k1", "k1_orig"), ("k2", "k2_orig"),
                            ("lam1", "lam1"), ("lam2", "lam2")):
            F[k_out][:, :, it] = ops[k_in]
        coh[:, :, it] = gate["coh"]
        C_core[it] = gate["C"]
        ls_core[it] = gate["ls"]
        ycore_idx[it] = gate["rows_s"]
        gates.append(gate["gate_used"])
        # tangent-projected along-core wavenumber, same auto smoother
        g = gate
        u_xx = SpectralDerivs(u * ramp, Lx, Ly, "xx")
        Txx = _mafa(u * u_xx, g["d1"], g["d2"], g["seed"]["k1_orig"],
                    g["seed"]["k2_orig"], g["coh"], g["core_mask"],
                    g["ls2d"], g["Lx"], g["Ly"], sigma, g["aspect"],
                    g["sharpen"], patch=cfg.patch,
                    ramp_mask=g["interior"],
                    sharpen_weight=getattr(cfg, "sharpen_weight", "coh"))
        uu = _mafa(u * u * ramp, g["d1"], g["d2"],
                   g["seed"]["k1_orig"], g["seed"]["k2_orig"], g["coh"],
                   g["core_mask"], g["ls2d"], g["Lx"], g["Ly"], sigma,
                   g["aspect"], g["sharpen"], patch=cfg.patch,
                   ramp_mask=g["interior"],
                   sharpen_weight=getattr(cfg, "sharpen_weight", "coh"))
        uu = np.where(np.abs(uu) < 1e-12, 1.0, uu)
        k_along[:, :, it] = np.sqrt(np.maximum(-Txx / uu, 0.0))

    uhu_meta = dict(method="uhu_auto_gate_v2", sigma=sigma,
                    patch=cfg.patch, mu=mu, L=L,
                    aspect=gate["aspect"], sharpen=gate["sharpen"],
                    sharpen_weight=getattr(cfg, "sharpen_weight", "coh"),
                    frame_stride=cfg.frame_stride,
                    frames=frames, xmargin=cfg.xmargin,
                    ymargin=cfg.ymargin, tanhscale=cfg.tanhscale,
                    source_file=str(sh_path))
    out_path = out_dir / (sh_path.stem
                          + f"_kuhu_auto_sigma{sigma:.3f}"
                            f"_patch{cfg.patch}.npz")
    save = dict(x=x, y=y, u=u_all[:, :, frames],
                tt=(tt[frames] if tt is not None else np.array([])),
                ramp=ramp, uhu_meta_json=json.dumps(uhu_meta),
                coh=coh, k_along=k_along, C_core=C_core,
                ls_core=ls_core, ycore_idx=ycore_idx,
                gate_used=np.array(gates))
    if sh["metadata_json"] is not None:
        save["sh_meta_json"] = sh["metadata_json"]
    save.update(F)
    np.savez_compressed(out_path, **save)
    print(f"  saved -> {out_path}")

    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = out_path.stem
    make_data_summary_plot(F, ramp, x, y, mu, cfg,
                           fig_dir / f"{stem}.png")
    make_data_gate_plot(F, k_along, C_core, ls_core, ycore_idx, ramp,
                        x, y, save["tt"], mu, cfg,
                        fig_dir / f"{stem}_gate.png")
    return out_path



def make_data_summary_plot(F, ramp, x, y, mu, cfg, fig_path):
    """Clone of run_kuhu_zigzag.make_summary_plot so the auto-gate OP
    output is visually comparable frame for frame: top row k initial /
    k final / arg(k1 + i k2) final / ramp; bottom row k1 and k2
    initial and final. White outside the interior, equal aspect."""
    Nt = F["k"].shape[-1]
    fi0, fi1 = 0, max(Nt - 1, 0)
    extent = [x[0], x[-1], y[0], y[-1]]
    msk = lambda f: np.ma.masked_where(ramp < (1.0 - 1e-12), f)
    cmap_k = plt.cm.viridis.copy(); cmap_k.set_bad(color="white")
    cmap_v = plt.cm.coolwarm.copy(); cmap_v.set_bad(color="white")
    cmap_p = plt.cm.twilight.copy(); cmap_p.set_bad(color="white")
    phase_f = np.angle(F["k1"][:, :, fi1] + 1j * F["k2"][:, :, fi1])
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    panels = (
        (axs[0, 0], msk(F["k"][:, :, fi0]), "k  (initial)", cmap_k),
        (axs[0, 1], msk(F["k"][:, :, fi1]), "k  (final)", cmap_k),
        (axs[0, 2], msk(phase_f), "arg(k1 + i k2)  (final)", cmap_p),
        (axs[0, 3], ramp, "rectangular ramp", "gray"),
        (axs[1, 0], msk(F["k1"][:, :, fi0]), "k1  (initial)", cmap_v),
        (axs[1, 1], msk(F["k1"][:, :, fi1]), "k1  (final)", cmap_v),
        (axs[1, 2], msk(F["k2"][:, :, fi0]), "k2  (initial)", cmap_v),
        (axs[1, 3], msk(F["k2"][:, :, fi1]), "k2  (final)", cmap_v),
    )
    for ax, f, title, cm in panels:
        im = ax.imshow(f, cmap=cm, origin="lower", extent=extent)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"mu={mu:.3f}  sigma={cfg.sigma:.3f}  auto-gated uHu  "
                 f"patch={cfg.patch}  xmargin={cfg.xmargin:.2f}  "
                 f"ymargin={cfg.ymargin:.2f}", y=0.99)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  summary plot -> {fig_path}")


def make_data_gate_plot(F, k_along, C_core, ls_core, ycore_idx, ramp,
                        x, y, tt, mu, cfg, fig_path):
    """Auto-gate diagnostics per file, the nucleation-facing views:
    spacetime maps of the line score and defect index on the core, the
    final k_along map, on-core k_along profiles first vs final with
    the exact far-field k1 for reference, their residuals from a
    linear-fit core phase, and the two nucleation time series."""
    Nt = ls_core.shape[0]
    k1_exact = float(np.sqrt(max(1.0 - mu * mu, 1e-12)))
    tvals = (np.asarray(tt, dtype=float)
             if tt is not None and len(np.atleast_1d(tt)) == Nt
             else np.arange(Nt, dtype=float))
    extent_xt = [x[0], x[-1], tvals[0], tvals[-1] if Nt > 1 else 1.0]
    Nx = x.size
    xin = np.abs(x - 0.5 * (x[0] + x[-1])) < 0.35 * (x[-1] - x[0])

    def core_prof(field3, it):
        rows = ycore_idx[it]
        return field3[:, :, it][rows, np.arange(Nx)]

    fig, axs = plt.subplots(2, 3, figsize=(15.6, 7.2),
                            constrained_layout=True)
    im = axs[0, 0].imshow(ls_core, extent=extent_xt, origin="lower",
                          aspect="auto", cmap="coolwarm", vmin=0, vmax=1)
    axs[0, 0].set_title("line score ls(x, t) on the core "
                        "(1 knee-like, 0 defect-like)", fontsize=9)
    axs[0, 0].set_xlabel("$x$"); axs[0, 0].set_ylabel("$t$")
    fig.colorbar(im, ax=axs[0, 0], shrink=0.85)

    im = axs[0, 1].imshow(C_core, extent=extent_xt, origin="lower",
                          aspect="auto", cmap="inferno", vmin=0,
                          vmax=1.1)
    axs[0, 1].set_title("defect index $|C|/2\\pi$ (x, t) on the core",
                        fontsize=9)
    axs[0, 1].set_xlabel("$x$"); axs[0, 1].set_ylabel("$t$")
    fig.colorbar(im, ax=axs[0, 1], shrink=0.85)

    kaf = np.ma.masked_where(ramp < (1.0 - 1e-12),
                             k_along[:, :, Nt - 1])
    im = axs[0, 2].imshow(kaf, extent=[x[0], x[-1], y[0], y[-1]],
                          origin="lower", aspect="auto", cmap="viridis",
                          vmin=0, vmax=1.2)
    axs[0, 2].set_title("$k_\\parallel$ (final): "
                        "$\\sqrt{-\\langle u u_{xx}\\rangle/"
                        "\\langle u^2\\rangle}$", fontsize=9)
    fig.colorbar(im, ax=axs[0, 2], shrink=0.85)

    ax = axs[1, 0]
    for it, lab, col in ((0, "initial", "0.4"), (Nt - 1, "final", "b")):
        ax.plot(x[xin], core_prof(k_along, it)[xin], lw=1.0, color=col,
                label=f"$k_\\parallel$ {lab}")
    ax.axhline(k1_exact, color="k", lw=0.6, ls="--",
               label="far-field $k_1$")
    ax.set_title("on-core $k_\\parallel$ profiles", fontsize=9)
    ax.set_xlabel("$x$"); ax.legend(fontsize=7)

    ax = axs[1, 1]
    dxs = np.gradient(x)
    for it, lab, col in ((0, "initial", "0.4"), (Nt - 1, "final", "b")):
        prof = core_prof(k_along, it)
        th = np.cumsum(prof * dxs)
        xs, ts = x[xin], th[xin]
        A = np.vstack([xs, np.ones_like(xs)]).T
        coef, *_ = np.linalg.lstsq(A, ts, rcond=None)
        ax.plot(xs, ts - A @ coef, lw=1.0, color=col, label=lab)
    ax.set_title("core phase $\\int k_\\parallel dx$: residual from "
                 "linear fit", fontsize=9)
    ax.set_xlabel("$x$"); ax.legend(fontsize=7)

    ax = axs[1, 2]
    ka_min = [float(np.min(core_prof(k_along, it)[xin]))
              for it in range(Nt)]
    ls_med = [float(np.median(ls_core[it][xin])) for it in range(Nt)]
    ax.plot(tvals, ka_min, "b-", lw=1.2,
            label="$\\min_x k_\\parallel$ on core")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\\min_x k_\\parallel$", color="b")
    ax2 = ax.twinx()
    ax2.plot(tvals, ls_med, "r-", lw=1.0, label="median ls")
    ax2.set_ylabel("median line score", color="r")
    ax2.set_ylim(-0.05, 1.05)
    ax.set_title("nucleation time series", fontsize=9)
    fig.suptitle(f"mu={mu:.3f}: auto-gate diagnostics "
                 f"(sigma={cfg.sigma:.3f}, patch={cfg.patch})",
                 fontsize=10)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  gate plot -> {fig_path}")



def run(cfg):
    out_root = Path(cfg.out)
    out_root.mkdir(parents=True, exist_ok=True)
    if getattr(cfg, "input", None):
        paths = [Path(cfg.input)]
        if paths[0].is_dir():
            paths = sorted(paths[0].glob("*.npz"))
        for pth in paths:
            process_sh_file(pth, out_root, cfg)
        return []
    rows = []
    for field_name in cfg.fields:
        for mu in cfg.mus:
            run_case(cfg, field_name, mu, out_root, rows)
    with open(out_root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(
            {k for r in rows for k in r}))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_root}/summary.csv")
    return rows


def main():
    p = argparse.ArgumentParser(
        description="Auto-gated uHu tensor extraction (self-test)")
    p.add_argument("--out", type=str, default="results/kuhu_auto")
    p.add_argument("--fields", nargs="+", default=["knee", "pismen"])
    p.add_argument("--mus", type=float, nargs="+", default=[0.85, 0.9])
    p.add_argument("--R", type=float, default=0.5)
    p.add_argument("--R_out", type=float, default=0.2)
    p.add_argument("--Nx", type=int, default=512)
    p.add_argument("--Ny", type=int, default=256)
    p.add_argument("--Lx", type=float, default=40 * np.pi)
    p.add_argument("--Ly", type=float, default=20 * np.pi)
    p.add_argument("--sigma", type=float, default=np.pi / 2)
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--leak_delta", type=float, default=0.05)
    p.add_argument("--n_defects", type=int, default=8)
    p.add_argument("--central_periods", type=float, default=4.0)
    p.add_argument("--edge_margin_periods", type=float, default=2.0)
    p.add_argument("--relax_T_pismen", type=float, default=3.0)
    p.add_argument("--relax_T_knee", type=float, default=0.0)
    p.add_argument("--relax_h", type=float, default=0.05)
    p.add_argument("--gate_window_L", type=float, default=2.0)
    p.add_argument("--sigma_gate", type=float, default=np.pi / 8)
    p.add_argument("--sigma_dir", type=float, default=np.pi / 4)
    p.add_argument("--L_over_sigdir_min", type=float, default=8.5)
    p.add_argument("--circ_lo", type=float, default=0.25)
    p.add_argument("--circ_hi", type=float, default=0.5)
    p.add_argument("--core_coh_frac", type=float, default=0.5)
    p.add_argument("--core_band_sigma", type=float, default=2.0)
    p.add_argument("--xmargin", type=float, default=0.05)
    p.add_argument("--ymargin", type=float, default=0.10)
    p.add_argument("--tanhscale", type=float, default=60.0)
    p.add_argument("--sharpen_weight", type=str, default="ls",
                   choices=["coh", "ls"])
    p.add_argument("--input", type=str, default=None,
                   help="SH .npz file or directory of them (data mode; "
                        "omit for the synthetic self-test)")
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--mu_override", type=float, default=None)
    p.add_argument("--core_row", type=int, default=None,
                   help="explicit core row index (else gb_row_in_crop "
                        "metadata, else coherence search)")
    run(p.parse_args())


if __name__ == "__main__":
    if len(sys.argv) == 1:
        class _Args:
            out = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/uhu_auto_gate/mu_sweeps_full_Nx512_hp025_T3p125_NyF5_np18_Nsave125/sig_pio2/"
            fields = ["knee", "pismen"]
            mus = [0.85, 0.9]
            R, R_out = 0.5, 0.2
            Nx, Ny = 512, 256
            Lx, Ly = 40 * np.pi, 20 * np.pi
            sigma = np.pi / 2
            patch = 128
            leak_delta = 0.05
            n_defects = 8
            central_periods = 4.0
            edge_margin_periods = 2.0
            relax_T_pismen = 3.0
            relax_T_knee = 0.0
            relax_h = 0.05
            gate_window_L = 2.0
            sigma_gate = np.pi / 8
            sigma_dir = np.pi / 4
            L_over_sigdir_min = 8.5
            circ_lo, circ_hi = 0.25, 0.5
            cf_lo, cf_hi = 1.10, 1.35
            core_coh_frac = 0.5
            core_band_sigma = 2.0
            xmargin, ymargin = 0.05, 0.10
            tanhscale = 60.0
            sharpen_weight = "ls"
            input = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweeps_full_Nx512_hp025_T3p125_NyF5_np18_Nsave125/raw"
            frame_stride = 1
            max_frames = 0
            mu_override = None
            core_row = None
        run(_Args)
    else:
        main()
