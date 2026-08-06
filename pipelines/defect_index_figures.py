# pipelines/defect_index_figures.py
"""
Methods-section figures for the quantized defect index |C|/2pi, the seed
coherence, and the auto-gated kernel field, on the two reference
geometries: an exact defect-free Hopf-Cole knee and a relaxed
Pismen-Nepomnyashchy dislocation chain.

This script computes nothing new. Every quantity is produced by the same
functions `adaptive_auto_gate.py` uses at run time (seed pass, core path,
gate selection, circulation windows, kernel branches), imported directly,
so the figures depict the pipeline as it actually operates rather than a
re-implementation of it.

Figures (per mu, under out/):

  index_maps_mu<..>.png     Two rows (knee / chain). Left: pattern with
                            the coarse-grid defect-index overlay, cropped
                            to an interior eroded well inside the ramp
                            collar. Right: the on-core profile |C(x)|/2pi
                            with the quantization levels 0 and 1 marked.
                            The intended reading: the index is negligible
                            everywhere on a defect-free knee and locked to
                            one quantum on the chain, with no threshold
                            tuning.

  loop_sweep_mu<..>.png     Left: median on-core |C|/2pi versus the loop
                            half-width in units of L = pi/k1, knee and
                            chain on one axis, with the IQR band. The
                            knee's cancellation is exact when the x extent
                            is an integer number of artifact periods and
                            remains small in between; the chain reads the
                            enclosed charge, stepping to 2 once the window
                            spans two defects. Right: chain index maps at
                            three window sizes, exhibiting the steps.

  coherence_mu<..>.png      Seed coherence for both geometries with the
                            detected core path. Both cores are
                            low-coherence lines: this is the figure that
                            justifies why coherence alone cannot gate the
                            kernel and a topological measurement is
                            needed.

  kernel_field_mu<..>.png   Kernel ellipses on a sparse patch grid over a
                            light coherence underlay, colored by branch:
                            far-field ellipses along k, core ellipses
                            colored by line score (red = elongated along
                            the core on the knee, blue = isotropic on the
                            chain). One config, no manual switch.

All maps are restricted to an interior eroded by the extraction scale
plus the circulation half-window, so no ramp-collar artifact enters any
panel.

Usage
-----
python defect_index_figures.py --out results/index_figs --mus 0.9
(or run in PyCharm; edit _Args at the bottom)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, median_filter

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "src", _HERE / "src", _HERE.parent):
    if _p.exists():
        sys.path.insert(0, str(_p))

from adaptive_auto_gate_patched import (
    field_knee, field_pismen, relax_sh, core_path, loop_circulation,
    circulation_profile, line_score_from_circulation, riesz_gate_field,
    macro_aniso_field_auto)
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
# Case preparation: identical sequence to adaptive_auto_gate.run_case up
# to (and including) the gate, factored for figure use.
# -----------------------------------------------------------------------

def prepare_case(cfg, field_name, mu):
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
    ramp = build_rectangular_ramp_smooth(
        x, y, xmargin=cfg.xmargin, ymargin=cfg.ymargin,
        tanhscale=cfg.tanhscale)
    interior = ramp >= 0.99
    if relax_T > 0:
        R_field = -cfg.R_out + (cfg.R + cfg.R_out) * ramp
        u = relax_sh(u * ramp, R_field, Lx, cfg.Ly, relax_T, h=cfg.relax_h)

    iso_sm = lambda a: macro(a, sigma, Lx, cfg.Ly)
    seed = compute_uhu_ops_smoother(u, x, y, iso_sm, SpectralDerivs, ramp)
    coh = coherence_from_eigs(seed["lam1"], seed["lam2"])
    d1, d2 = slowness_director(seed["k1_orig"], seed["k2_orig"], sigma,
                               Lx, cfg.Ly, SpectralDerivs, macro)

    rows_c, _ = core_path(coh, y, band_frac=0.35)
    dx = Lx / Nx
    half_w = max(4, int(round(cfg.gate_window_L * L / dx / 2)))
    rows_s = np.round(np.convolve(
        rows_c, np.ones(2 * half_w + 1) / (2 * half_w + 1),
        mode="same")).astype(int)

    use_riesz = L > cfg.L_over_sigdir_min * cfg.sigma_dir
    if use_riesz:
        # Flip-free direction (Riesz director) times the flip-free
        # magnitude: the tensor projected onto the director,
        # q^2 = g.T (-T) g / <u^2>. The seed |k| is corrupted near the
        # core above mu*, and that corruption puts spurious winding
        # into the gate field; the projection has no eigen decision and
        # is exact on the ideal knee, so the constructed field carries
        # the physical topology (zero circulation on a knee) instead of
        # the estimator's.
        g1, g2 = riesz_gate_field(u, Lx, cfg.Ly, cfg.sigma_dir, macro)
        ur = u * ramp
        Txx = iso_sm(u * SpectralDerivs(ur, Lx, cfg.Ly, "xx"))
        Txy = iso_sm(u * SpectralDerivs(ur, Lx, cfg.Ly, "xy"))
        Tyy = iso_sm(u * SpectralDerivs(ur, Lx, cfg.Ly, "yy"))
        uu = iso_sm(u * u * ramp)
        uu = np.where(np.abs(uu) < 1e-12, 1.0, uu)
        q = np.sqrt(np.maximum(
            -(g1 * g1 * Txx + 2 * g1 * g2 * Txy + g2 * g2 * Tyy) / uu,
            0.0))
        kgx, kgy = q * g1, q * g2
        gate_used = "riesz_director_proj"
    else:
        gate_sm = lambda a: macro(a, cfg.sigma_gate, Lx, cfg.Ly)
        gseed = compute_uhu_ops_smoother(u, x, y, gate_sm,
                                         SpectralDerivs, ramp)
        kgx, kgy = gseed["k1_orig"], gseed["k2_orig"]
        gate_used = "tensor"
    C = circulation_profile(kgx, kgy, x, y, rows_s, L)
    ls = line_score_from_circulation(C, cfg.circ_lo, cfg.circ_hi)

    return dict(field=field_name, mu=mu, u=u, x=x, y=y, k1=k1, k2=k2,
                rows_c=rows_c,
                L=L, Lx=Lx, Nx=Nx, x_defects=x_defects, ramp=ramp,
                interior=interior, seed=seed, coh=coh, d1=d1, d2=d2,
                rows_s=rows_s, kgx=kgx, kgy=kgy, C=C, ls=ls,
                gate_used=gate_used, sigma=sigma)


def eroded_interior(case, cfg, extra_len=0.0):
    """Interior mask eroded by 2 sigma plus the circulation half-window
    plus extra_len (all in length units), so no collar artifact enters."""
    dxg = case["Lx"] / case["Nx"]
    dyg = cfg.Ly / cfg.Ny
    pad = 2.0 * case["sigma"] + 0.5 * case["L"] + extra_len
    n_er = max(1, int(round(pad / min(dxg, dyg))))
    return binary_erosion(case["interior"], iterations=n_er)


def crop_extent(mask, x, y):
    """Bounding box of a mask as an imshow extent, for tight display."""
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return ([x[cols[0]], x[cols[-1]], y[rows[0]], y[rows[-1]]],
            (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)))


# -----------------------------------------------------------------------
# Coarse-grid defect index map at a given loop size
# -----------------------------------------------------------------------

def index_map(case, cfg, wfrac=0.5, x_stride_px=2, ny_pts=48,
              clear_px=3):
    """|C|/2pi on a grid of window centers, half-widths wfrac * L.
    Centers are dense in x (every x_stride_px pixels) so the one-window
    median can outvote clustered bad windows (adjacent columns share
    most of their loop, so snapped-pixel errors on a vertical edge
    contaminate several neighbors at once). Windows whose horizontal
    edge would ride along the core path are masked (NaN) rather than
    resized: an edge-on pass is poisoned end to end, and masking keeps
    the displayed enclosure rectangles honest."""
    x, y, L = case["x"], case["y"], case["L"]
    kgx, kgy = case["kgx"], case["kgy"]
    rows_c = case["rows_c"]
    dxg = case["Lx"] / case["Nx"]
    dyg = cfg.Ly / cfg.Ny
    rxh = int(round(wfrac * L / dxg))
    ryh = int(round(wfrac * L / dyg))
    Nx, Ny = case["Nx"], cfg.Ny
    sx = x_stride_px
    sy = max(1, Ny // ny_pts)
    yc = np.arange(ryh + 1, Ny - ryh - 1, sy)
    xc = np.arange(rxh + 1, Nx - rxh - 1, sx)
    Cmap = np.full((yc.size, xc.size), np.nan)
    for a, jc in enumerate(yc):
        for b, icx in enumerate(xc):
            seg = rows_c[max(icx - rxh, 0):icx + rxh + 1]
            if min(np.min(np.abs((jc + ryh) - seg)),
                   np.min(np.abs((jc - ryh) - seg))) <= clear_px:
                continue
            Cmap[a, b] = abs(loop_circulation(
                kgx, kgy, x, y, icx, jc, rxh, ryh)) / (2.0 * np.pi)
    # NaN-aware median over one full window width in center space
    half = max(1, int(round(rxh / sx)))
    pad = np.pad(Cmap, ((0, 0), (half, half)), mode="edge")
    sw = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1,
                                                  axis=1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Cmap = np.nanmedian(sw, axis=-1)
    ext = [x[xc[0]], x[xc[-1]], y[yc[0]], y[yc[-1]]]
    return Cmap, ext


def core_profile_geom(case, cfg, rx_len, ry_len):
    """On-core |C|/2pi with independent loop half-extents rx_len,
    ry_len (length units), one-window median along x as in the
    pipeline. Separating the two extents matters: the knee cancellation
    is a property of the x extent being an integer number of
    corrugation periods; the y extent is free."""
    x, y, rows_s = case["x"], case["y"], case["rows_s"]
    kgx, kgy = case["kgx"], case["kgy"]
    dxg, dyg = x[1] - x[0], y[1] - y[0]
    rxh = int(round(rx_len / dxg))
    ry = int(round(ry_len / dyg))
    n = x.size
    C = np.zeros(n)
    for i in range(rxh + 1, n - rxh - 1):
        r0 = int(np.clip(rows_s[i], ry + 1, kgx.shape[0] - ry - 2))
        C[i] = abs(loop_circulation(kgx, kgy, x, y, i, r0, rxh, ry)
                   ) / (2.0 * np.pi)
    C = median_filter(C, size=2 * rxh + 1, mode="nearest")
    return C


# -----------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------

def _index_cmap(vmax_col="inferno"):
    """Inferno with NaN (masked windows) shown as flat light gray so
    exclusions read as exclusions, not as bright pattern bands."""
    cm = plt.get_cmap(vmax_col).copy()
    cm.set_bad((0.75, 0.75, 0.75, 0.75))
    return cm


def fig_index_maps(cases, cfg, out_dir, mu):
    fig, axs = plt.subplots(2, 2, figsize=(11.6, 5.6),
                            constrained_layout=True,
                            gridspec_kw=dict(width_ratios=[1.35, 1.0]))
    for row, case in enumerate(cases):
        disp = eroded_interior(case, cfg)
        ext, sl = crop_extent(disp, case["x"], case["y"])
        ax = axs[row, 0]
        ax.imshow(np.ma.masked_where(~disp, case["u"])[sl], extent=ext,
                  origin="lower", aspect="auto", cmap="gray",
                  rasterized=True)
        Cmap, extc = index_map(case, cfg, wfrac=0.5)
        im = ax.imshow(Cmap, extent=extc, origin="lower", aspect="auto",
                       cmap=_index_cmap(), vmin=0, vmax=1.1, alpha=0.45,
                       rasterized=True)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        label = ("defect-free knee" if case["field"] == "knee"
                 else "PN multi-dislocation")
        ax.set_title(f"{label}: defect index $|C|/2\\pi$", fontsize=9)
        ax.set_ylabel("$y$")
        if row == 1:
            ax.set_xlabel("$x$")
        fig.colorbar(im, ax=ax, shrink=0.85)

        ax = axs[row, 1]
        xin = np.abs(case["x"]) < 0.35 * case["Lx"]
        ax.plot(case["x"][xin], case["C"][xin], "k-", lw=1.2)
        ax.axhline(0.0, color="0.6", lw=0.6, ls="--")
        ax.axhline(1.0, color="0.6", lw=0.6, ls="--")
        for xm in case["x_defects"]:
            if case["x"][xin].min() < xm < case["x"][xin].max():
                ax.axvline(xm, color="k", lw=0.4, alpha=0.25)
        ax.set_ylim(-0.08, 1.35)
        ax.set_ylabel("$|C|/2\\pi$ on the core")
        med = float(np.median(case["C"][xin]))
        ax.set_title(f"gate = {case['gate_used']},  median = {med:.2f}",
                     fontsize=9)
        if row == 1:
            ax.set_xlabel("$x$")
    fig.savefig(out_dir / f"index_maps_mu{mu:.2f}.png",
                bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_loop_sweep(cases, cfg, out_dir, mu,
                   hfracs=(0.35, 0.5, 0.75, 1.0, 1.25, 1.5),
                   mwidths=(1, 2, 3)):
    fig = plt.figure(figsize=(12.4, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1.0])
    axc = fig.add_subplot(gs[:, 0])
    styles = {"knee": dict(color="tab:blue", marker="o"),
              "pismen": dict(color="tab:red", marker="s")}
    for case in cases:
        L = case["L"]
        xin = np.abs(case["x"]) < 0.30 * case["Lx"]
        st = styles[case["field"]]
        lab = ("knee (defect-free)" if case["field"] == "knee"
               else "PN (one quantum per defect)")
        med, q25, q75 = [], [], []
        for hf in hfracs:
            prof = core_profile_geom(case, cfg, 0.5 * L, hf * L)[xin]
            med.append(np.median(prof))
            q25.append(np.percentile(prof, 25))
            q75.append(np.percentile(prof, 75))
        axc.plot(hfracs, med, lw=1.3, ms=4, label=lab, **st)
        axc.fill_between(hfracs, q25, q75, color=st["color"], alpha=0.18,
                         lw=0)
        wmed = [np.median(core_profile_geom(
            case, cfg, m * 0.5 * L, 0.5 * L)[xin]) for m in mwidths]
        axc.plot([0.5 * m for m in mwidths], wmed, ls="none", ms=7,
                 marker="D", mfc="none", color=st["color"])
    for lev in (0, 1, 2, 3):
        axc.axhline(lev, color="0.6", lw=0.6, ls="--")
    axc.set_xlabel("loop half-extent / $L$   (circles: height sweep at "
                   "period-matched width; diamonds: width in integer "
                   "multiples of $L$)", fontsize=8)
    axc.set_ylabel("on-core $|C|/2\\pi$ (median, IQR)")
    axc.set_title("quantization vs loop geometry: zero on the knee for "
                  "any height,\ncharge counts enclosed defects as the "
                  "width grows", fontsize=9)
    axc.legend(fontsize=8)

    chain = next(c for c in cases if c["field"] == "pismen")
    for i, m in enumerate(mwidths):
        ax = fig.add_subplot(gs[i, 1])
        disp = eroded_interior(chain, cfg,
                               extra_len=(m - 1) * 0.5 * chain["L"])
        ext, sl = crop_extent(disp, chain["x"], chain["y"])
        ax.imshow(np.ma.masked_where(~disp, chain["u"])[sl], extent=ext,
                  origin="lower", aspect="auto", cmap="gray",
                  rasterized=True)
        Cmap, extc = index_map(chain, cfg, wfrac=0.5 * m, x_stride_px=4,
                               ny_pts=32)
        im = ax.imshow(Cmap, extent=extc, origin="lower", aspect="auto",
                       cmap=_index_cmap(), vmin=0, vmax=3.2, alpha=0.5,
                       rasterized=True)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_yticks([])
        if i < len(mwidths) - 1:
            ax.set_xticks([])
        ax.set_title(f"PN, width $= {m}\\,L$", fontsize=8, pad=2)
        fig.colorbar(im, ax=ax, shrink=0.9)
    fig.savefig(out_dir / f"loop_sweep_mu{mu:.2f}.png",
                bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_coherence(cases, cfg, out_dir, mu):
    fig, axs = plt.subplots(2, 1, figsize=(7.6, 5.4),
                            constrained_layout=True)
    for ax, case in zip(axs, cases):
        disp = eroded_interior(case, cfg)
        ext, sl = crop_extent(disp, case["x"], case["y"])
        im = ax.imshow(np.ma.masked_where(~disp, case["coh"])[sl],
                       extent=ext, origin="lower", aspect="auto",
                       cmap="viridis", vmin=0, vmax=1, rasterized=True)
        xin = (case["x"] >= ext[0]) & (case["x"] <= ext[1])
        ax.plot(case["x"][xin], case["y"][case["rows_s"]][xin], "r-",
                lw=0.7)
        label = ("defect-free knee" if case["field"] == "knee"
                 else "PN multi-dislocation")
        ax.set_title(f"{label}: seed coherence + core path; both cores "
                     "are low-coherence lines", fontsize=9)
        ax.set_ylabel("$y$")
        fig.colorbar(im, ax=ax, shrink=0.85)
    axs[1].set_xlabel("$x$")
    fig.savefig(out_dir / f"coherence_mu{mu:.2f}.png",
                bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_kernel_field(cases, cfg, out_dir, mu):
    fig, axs = plt.subplots(2, 1, figsize=(8.4, 6.0),
                            constrained_layout=True)
    tt = np.linspace(0, 2 * np.pi, 60)
    for ax, case in zip(axs, cases):
        x, y, L = case["x"], case["y"], case["L"]
        seed, coh = case["seed"], case["coh"]
        interior = case["interior"]
        # core mask + ls2d as in run_case
        cmin = float(coh[interior].min())
        cmed = float(np.median(coh[interior]))
        cth = cmin + cfg.core_coh_frac * (cmed - cmin)
        Y = np.meshgrid(x, y)[1]
        y_core = y[case["rows_s"]][None, :]
        band = np.abs(Y - y_core) < cfg.core_band_sigma * case["sigma"]
        core_mask = (coh < cth) & band
        ls2d = np.where(band, case["ls"][None, :], 0.0)
        aspect, sharpen = derived_kernel_params(
            case["k1"], case["k2"], case["sigma"], delta=cfg.leak_delta)
        patches = []
        _ = macro_aniso_field_auto(
            np.zeros_like(case["u"]), case["d1"], case["d2"],
            seed["k1_orig"], seed["k2_orig"], coh, core_mask, ls2d,
            case["Lx"], cfg.Ly, case["sigma"], aspect, sharpen,
            patch=cfg.patch, record=patches, ramp_mask=interior)
        disp = eroded_interior(case, cfg)
        ext, sl = crop_extent(disp, x, y)
        ax.imshow(np.ma.masked_where(~disp, coh)[sl], extent=ext,
                  origin="lower", aspect="auto", cmap="Greys_r",
                  vmin=-0.6, vmax=1.6, rasterized=True, alpha=0.9)
        dxg = case["Lx"] / case["Nx"]
        dyg = cfg.Ly / cfg.Ny
        for p in patches[::2]:
            cx = x[0] + p["cx"] * dxg
            cy = y[0] + p["cy"] * dyg
            if not (ext[0] < cx < ext[1] and ext[2] < cy < ext[3]):
                continue
            c, s = np.cos(p["theta"]), np.sin(p["theta"])
            ex = p["s_par"] * np.cos(tt) * c - p["s_perp"] * np.sin(tt) * s
            ey = p["s_par"] * np.cos(tt) * s + p["s_perp"] * np.sin(tt) * c
            if p["branch"] == "core":
                col = plt.cm.coolwarm(p["ls"])
                lw = 1.1
            elif p["branch"] == "far":
                col, lw = "y", 0.7
            else:
                continue
            ax.plot(cx + ex, cy + ey, color=col, lw=lw)
        label = ("defect-free knee" if case["field"] == "knee"
                 else "PN multi-dislocation")
        ax.set_title(f"{label}: kernels from one config (far field "
                     "yellow, along $k$; core colored by line score: "
                     "red = elongated, blue = isotropic)", fontsize=8.5)
        ax.set_ylabel("$y$")
    axs[1].set_xlabel("$x$")
    sm = plt.cm.ScalarMappable(cmap="coolwarm",
                               norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axs, shrink=0.7, label="line score")
    fig.savefig(out_dir / f"kernel_field_mu{mu:.2f}.png",
                bbox_inches="tight", dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------

def run(cfg):
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for mu in cfg.mus:
        print(f"\n=== figures at mu = {mu} ===")
        cases = [prepare_case(cfg, f, mu) for f in ("knee", "pismen")]
        for c in cases:
            xin = np.abs(c["x"]) < 0.30 * c["Lx"]
            print(f"  {c['field']:6s}: gate={c['gate_used']:14s} "
                  f"|C| med={np.median(c['C'][xin]):.3f} "
                  f"ls med={np.median(c['ls'][xin]):.3f}")
        fig_index_maps(cases, cfg, out_dir, mu)
        fig_loop_sweep(cases, cfg, out_dir, mu)
        fig_coherence(cases, cfg, out_dir, mu)
        fig_kernel_field(cases, cfg, out_dir, mu)
        print(f"  wrote 4 figures to {out_dir}")


def main():
    p = argparse.ArgumentParser(description="Defect-index method figures")
    p.add_argument("--out", type=str, default="results/index_figs")
    p.add_argument("--mus", type=float, nargs="+", default=[0.9])
    p.add_argument("--R", type=float, default=0.5)
    p.add_argument("--R_out", type=float, default=0.2)
    p.add_argument("--Nx", type=int, default=512)
    p.add_argument("--Ny", type=int, default=256)
    p.add_argument("--Lx", type=float, default=40 * np.pi)
    p.add_argument("--Ly", type=float, default=20 * np.pi)
    p.add_argument("--sigma", type=float, default=np.pi / 2)
    p.add_argument("--patch", type=int, default=96)
    p.add_argument("--leak_delta", type=float, default=0.05)
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
    p.add_argument("--core_coh_frac", type=float, default=0.5)
    p.add_argument("--core_band_sigma", type=float, default=2.0)
    p.add_argument("--xmargin", type=float, default=0.05)
    p.add_argument("--ymargin", type=float, default=0.10)
    p.add_argument("--tanhscale", type=float, default=60.0)
    run(p.parse_args())


if __name__ == "__main__":
    if len(sys.argv) == 1:
        class _Args:
            out = "results/index_figs"
            mus = [0.85,0.9,0.95]
            R, R_out = 0.5, 0.2
            Nx, Ny = 512, 256
            Lx, Ly = 40 * np.pi, 20 * np.pi
            sigma = np.pi / 2
            patch = 96
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
