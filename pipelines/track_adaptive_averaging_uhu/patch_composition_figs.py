"""
patch_composition_figs.py  how the auto-gated average is assembled.

Two methods figures that make the patchwise Hann overlap-add construction
of macro_aniso_field_auto visible, on the same synthetic knee / PN cases
as adaptive_auto_gate_patched (identical config, no manual switch):

  patch_tiling_<field>_mu<mu>.png
    (1) the pattern with the patch grid and each patch's branch
        (edge / core / far) and kernel ellipse, so the branch logic and
        the ls-gated elongation are one picture;
    (2) the Hann weight sum wsum of the overlap-add, with the stride
        grid, showing that coverage is uniform in the interior and why
        the final division by wsum is benign;
    (3) one core patch in detail: its Hann window, the elongated kernel
        footprint at 1 and 2 sigma, and the containment rule
        patch >= ~4 aspect sigma that sets the minimum patch size.

  overlap_add_<field>_mu<mu>.png
    the along-core assembly of the averaged energy: every patch whose
    footprint covers the core row contributes a Hann-weighted smoothed
    segment (thin curves); their sum divided by wsum is the pipeline's
    e_auto on the core (thick curve), compared against the isotropic
    average. A second row repeats the assembly at a deliberately
    undersized patch to show the kernel-truncation beat that motivates
    the containment rule (those runs are diagnostic only; no numbers
    from them are quoted).

Run with no arguments for the default knee + pismen at mu = 0.90, or
via the CLI. Imports the pipeline functions directly so every panel
depicts what the pipeline computes.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adaptive_auto_gate_patched import (
    micro_energy, macro_aniso_field_auto, derived_kernel_params,
    sample_along_path, _hann2d)
from defect_index_figures import prepare_case
try:
    from utils.spectral import macro
except ImportError:
    from spectral import macro
try:
    from paper_style import use_paper_style
    use_paper_style()
except ImportError:
    pass

BRANCH_COLORS = {"edge": "0.55", "far": "goldenrod", "core": "crimson"}


# -----------------------------------------------------------------------
# Guidance fields: the exact core_mask / ls2d construction of run_case
# -----------------------------------------------------------------------

def guidance_fields(case, cfg):
    coh, ls, sigma = case["coh"], case["ls"], case["sigma"]
    x, y, rows_s = case["x"], case["y"], case["rows_s"]
    interior = case["interior"]
    cmin = float(coh[interior].min())
    cmed = float(np.median(coh[interior]))
    cth = cmin + cfg.core_coh_frac * (cmed - cmin)
    Y = np.meshgrid(x, y)[1]
    band = np.abs(Y - y[rows_s][None, :]) < cfg.core_band_sigma * sigma
    core_mask = (coh < cth) & band
    ls2d = np.where(band, ls[None, :], 0.0)
    return core_mask, ls2d


def averaged_energy(case, cfg, patch, record):
    """e_micro, e_iso, e_auto with per-patch record, exactly as in
    run_case (far-field baseline removed before averaging)."""
    u, x, y, ramp = case["u"], case["x"], case["y"], case["ramp"]
    Lx, sigma = case["Lx"], case["sigma"]
    interior = case["interior"]
    e_micro = micro_energy(u * ramp, Lx, cfg.Ly, cfg.R)
    far = interior & (np.abs(np.meshgrid(x, y)[1]) > 0.25 * cfg.Ly)
    e_micro = e_micro - np.median(e_micro[far])
    core_mask, ls2d = guidance_fields(case, cfg)
    mu_eff = case["k2"]
    aspect, sharpen = derived_kernel_params(case["k1"], mu_eff, sigma,
                                            delta=cfg.leak_delta)
    e_iso = macro(e_micro, sigma, Lx, cfg.Ly)
    seed = case["seed"]
    e_auto = macro_aniso_field_auto(
        e_micro, case["d1"], case["d2"], seed["k1_orig"], seed["k2_orig"],
        case["coh"], core_mask, ls2d, Lx, cfg.Ly, sigma, aspect, sharpen,
        patch=patch, record=record, ramp_mask=interior)
    return e_micro, e_iso, e_auto, aspect, sharpen


# -----------------------------------------------------------------------
# Figure 1: tiling, branches, weight sum, one patch in detail
# -----------------------------------------------------------------------

def fig_patch_tiling(case, cfg, patch, out_dir):
    record = []
    e_micro, e_iso, e_auto, aspect, sharpen = averaged_energy(
        case, cfg, patch, record)
    x, y = case["x"], case["y"]
    dx = case["Lx"] / case["Nx"]
    dy = cfg.Ly / cfg.Ny
    extent = [x[0], x[-1], y[0], y[-1]]
    sigma = case["sigma"]

    fig = plt.figure(figsize=(13.6, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.5, 1.0])

    # (1) branch + ellipse map
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(np.ma.masked_where(~case["interior"], case["u"]),
              extent=extent, origin="lower", aspect="auto", cmap="gray",
              alpha=0.85, rasterized=True)
    tt = np.linspace(0, 2 * np.pi, 60)
    for p in record:
        cx, cy = x[0] + p["cx"] * dx, y[0] + p["cy"] * dy
        col = BRANCH_COLORS[p["branch"]]
        if p["branch"] == "edge":
            ax.plot(cx, cy, ".", color=col, ms=3)
            continue
        c, s = np.cos(p["theta"]), np.sin(p["theta"])
        ex = p["s_par"] * np.cos(tt) * c - p["s_perp"] * np.sin(tt) * s
        ey = p["s_par"] * np.cos(tt) * s + p["s_perp"] * np.sin(tt) * c
        lw = 1.4 if p["branch"] == "core" else 0.9
        ax.plot(cx + ex, cy + ey, color=col, lw=lw)
    for lab, col in (("core (ls-gated)", "crimson"),
                     ("far field (along $k$)", "goldenrod"),
                     ("edge (isotropic)", "0.55")):
        ax.plot([], [], color=col, lw=1.4, label=lab)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.set_title("patch branches and kernel ellipses "
                 f"(patch = {patch} px, stride = {patch // 2})",
                 fontsize=9)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")

    # (2) Hann weight sum
    ax = fig.add_subplot(gs[0, 1])
    ny, nx = case["u"].shape
    stride = patch // 2
    win = _hann2d(patch, patch)
    wsum = np.zeros((ny, nx))
    ys = list(range(0, ny - patch + 1, stride)) or [0]
    xs = list(range(0, nx - patch + 1, stride)) or [0]
    if ys[-1] != ny - patch:
        ys.append(max(ny - patch, 0))
    if xs[-1] != nx - patch:
        xs.append(max(nx - patch, 0))
    for y0 in ys:
        for x0 in xs:
            wsum[y0:y0 + patch, x0:x0 + patch] += win
    im = ax.imshow(wsum, extent=extent, origin="lower", aspect="auto",
                   cmap="viridis")
    for x0 in xs:
        ax.axvline(x[0] + x0 * dx, color="w", lw=0.3, alpha=0.5)
    for y0 in ys:
        ax.axhline(y[0] + y0 * dy, color="w", lw=0.3, alpha=0.5)
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title("Hann weight sum of the overlap-add\n"
                 "(uniform in the interior; the average divides by it)",
                 fontsize=9)
    ax.set_xlabel("$x$")

    # (3) one core patch in detail: window + kernel footprint
    ax = fig.add_subplot(gs[0, 2])
    core_ps = [p for p in record if p["branch"] == "core"]
    p = core_ps[len(core_ps) // 2] if core_ps else record[len(record) // 2]
    side_x, side_y = patch * dx, patch * dy
    xx = np.linspace(-side_x / 2, side_x / 2, patch)
    yy = np.linspace(-side_y / 2, side_y / 2, patch)
    ax.imshow(win, extent=[xx[0], xx[-1], yy[0], yy[-1]],
              origin="lower", aspect="equal", cmap="Blues", alpha=0.8)
    c, s = np.cos(p["theta"]), np.sin(p["theta"])
    for nsig, lw in ((1.0, 1.6), (2.0, 1.0)):
        ex = nsig * (p["s_par"] * np.cos(tt) * c
                     - p["s_perp"] * np.sin(tt) * s)
        ey = nsig * (p["s_par"] * np.cos(tt) * s
                     + p["s_perp"] * np.sin(tt) * c)
        ax.plot(ex, ey, color="crimson", lw=lw)
    ax.set_title("one core patch: Hann window,\nkernel at 1 and 2 "
                 f"$\\sigma$ ($s_\\parallel$ = {p['s_par']:.2f}, "
                 f"$s_\\perp$ = {p['s_perp']:.2f})", fontsize=9)
    ax.set_xlabel("$x$ (patch frame)")

    fig.suptitle(f"{case['field']} $\\mu$ = {case['mu']:.2f}: patchwise "
                 f"composition (aspect = {aspect:.2f}, sharpen = "
                 f"{sharpen:.2f}, gate = {case['gate_used']})", fontsize=9)
    fig.savefig(out_dir / f"patch_tiling_{case['field']}"
                          f"_mu{case['mu']:.2f}.png",
                bbox_inches="tight", dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 2: overlap-add assembly along the core, adequate vs truncated
# -----------------------------------------------------------------------

def fig_overlap_add(case, cfg, patches_px, out_dir):
    """Rows: one per patch size. Left: per-patch Hann-weighted
    contributions along the core (colored by branch) and their
    normalized sum vs the pipeline e_auto (identity check). Right:
    e_auto on the core for this patch size vs the isotropic average."""
    try:
        from anisotropic_macro import macro_aniso_uniform, _hann2d as _h2
    except ImportError:
        from utils.anisotropic_macro import macro_aniso_uniform, \
            _hann2d as _h2

    x, y = case["x"], case["y"]
    rows_s = case["rows_s"]
    dx = case["Lx"] / case["Nx"]
    dy = cfg.Ly / cfg.Ny
    n_rows = len(patches_px)
    fig, axs = plt.subplots(n_rows, 2, figsize=(12.8, 3.4 * n_rows),
                            constrained_layout=True, squeeze=False)

    for r, patch in enumerate(patches_px):
        record = []
        e_micro, e_iso, e_auto, aspect, sharpen = averaged_energy(
            case, cfg, patch, record)
        win = _h2(patch, patch)
        wsum = np.zeros_like(e_micro)
        acc = np.zeros_like(e_micro)
        segs = []
        for p in record:
            y0, x0 = p["y0"], p["x0"]
            sl = (slice(y0, y0 + patch), slice(x0, x0 + patch))
            sm = macro_aniso_uniform(
                e_micro[sl], p["theta"], p["s_par"], p["s_perp"],
                case["Lx"] * patch / e_micro.shape[1],
                cfg.Ly * patch / e_micro.shape[0])
            contrib = sm * win
            acc[sl] += contrib
            wsum[sl] += win
            # does this patch cover the core path anywhere?
            cols = np.arange(x0, x0 + patch)
            rows_here = rows_s[cols]
            inside = (rows_here >= y0) & (rows_here < y0 + patch)
            if inside.any():
                vals = contrib[rows_here[inside] - y0,
                               cols[inside] - x0]
                segs.append((cols[inside], vals, p["branch"]))
        wsafe = np.where(wsum < 1e-12, 1.0, wsum)
        e_re = acc / wsafe

        ax = axs[r, 0]
        for cols, vals, branch in segs:
            ax.plot(x[cols], vals, color=BRANCH_COLORS[branch], lw=0.7,
                    alpha=0.65)
        prof_re = sample_along_path(e_re, rows_s)
        prof_auto = sample_along_path(e_auto, rows_s)
        ax.plot(x, prof_re, "k-", lw=1.6,
                label="sum / weight (reassembled)")
        ax.plot(x, prof_auto, "c--", lw=1.0, label="pipeline e_auto")
        ax.set_title(f"patch = {patch} px: per-patch Hann-weighted "
                     "contributions on the core", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_ylabel("$\\langle e\\rangle$ contributions")

        ax = axs[r, 1]
        xin = np.abs(x) < 0.35 * case["Lx"]
        ax.plot(x[xin], sample_along_path(e_iso, rows_s)[xin], "r-",
                lw=1.0, label="iso")
        ax.plot(x[xin], prof_auto[xin], "b-", lw=1.2, label="auto")
        ok = np.allclose(prof_re[xin], prof_auto[xin], atol=1e-10)
        ax.set_title(f"on-core averages (reassembly identity: "
                     f"{'exact' if ok else 'MISMATCH'})", fontsize=9)
        ax.legend(fontsize=7)
        if r == n_rows - 1:
            for a in axs[r]:
                a.set_xlabel("$x$")

    fig.suptitle(f"{case['field']} $\\mu$ = {case['mu']:.2f}: overlap-add "
                 "assembly; an undersized patch truncates the elongated "
                 "kernel and beats at the stride scale", fontsize=9)
    fig.savefig(out_dir / f"overlap_add_{case['field']}"
                          f"_mu{case['mu']:.2f}.png",
                bbox_inches="tight", dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------

def run(cfg):
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for field in cfg.fields:
        for mu in cfg.mus:
            case = prepare_case(cfg, field, mu)
            fig_patch_tiling(case, cfg, cfg.patch, out_dir)
            fig_overlap_add(case, cfg, (cfg.patch, cfg.patch_small),
                            out_dir)
            print(f"{field} mu={mu:.2f}: figures written")


def main():
    p = argparse.ArgumentParser(
        description="Visualize the patchwise overlap-add composition")
    p.add_argument("--out", type=str, default="results/patch_figs")
    p.add_argument("--fields", nargs="+", default=["knee", "pismen"])
    p.add_argument("--mus", type=float, nargs="+", default=[0.90])
    p.add_argument("--patch", type=int, default=96)
    p.add_argument("--patch_small", type=int, default=32)
    p.add_argument("--R", type=float, default=0.5)
    p.add_argument("--R_out", type=float, default=0.2)
    p.add_argument("--Nx", type=int, default=512)
    p.add_argument("--Ny", type=int, default=256)
    p.add_argument("--Lx", type=float, default=40 * np.pi)
    p.add_argument("--Ly", type=float, default=20 * np.pi)
    p.add_argument("--sigma", type=float, default=np.pi / 2)
    p.add_argument("--leak_delta", type=float, default=0.02)
    p.add_argument("--n_defects", type=int, default=8)
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
    p.add_argument("--cf_lo", type=float, default=1.10)
    p.add_argument("--cf_hi", type=float, default=1.35)
    p.add_argument("--core_coh_frac", type=float, default=0.5)
    p.add_argument("--core_band_sigma", type=float, default=2.0)
    p.add_argument("--xmargin", type=float, default=0.05)
    p.add_argument("--ymargin", type=float, default=0.10)
    p.add_argument("--tanhscale", type=float, default=60.0)
    run(p.parse_args())


if __name__ == "__main__":
    if len(sys.argv) == 1:
        class _Args:
            out = "results/patch_figs"
            fields = ["knee", "pismen"]
            mus = [0.90]
            patch = 96
            patch_small = 32
            R, R_out = 0.5, 0.2
            Nx, Ny = 512, 256
            Lx, Ly = 40 * np.pi, 20 * np.pi
            sigma = np.pi / 2
            leak_delta = 0.02
            n_defects = 8
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
        run(_Args)
    else:
        main()
