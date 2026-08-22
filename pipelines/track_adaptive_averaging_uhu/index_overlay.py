# """
# index_overlay.py  cleaner rendering of the defect index over the pattern.
#
# Design rule: the overlay encodes |C|/2pi in OPACITY as well as color, so
# a zero index is fully transparent and the pattern underneath stays crisp.
# A defect-free knee then displays as a clean pattern (the claim "no fake
# topology anywhere" is the picture itself), while a PN boundary displays
# a saturated band on the core. Masked windows (edge-riding loops) render
# as a light dotted hatch, not as filled gray blocks, so exclusions read
# as "not measured" rather than as data. The coarse window grid is shown
# with nearest-neighbor cells on purpose: each cell IS one loop median,
# and smoothing it for display would misrepresent the estimator.
#
# Importable: overlay_index(ax, ...) for reuse in the pipeline dashboards,
# fig_index_maps_v2(...) as a drop-in replacement for
# defect_index_figures.fig_index_maps.
# """
#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import colors as mcolors
#
# from defect_index_figures import (index_map, eroded_interior, crop_extent)
#
#
# def index_to_rgba(Cmap, vmax=1.1, alpha_lo=0.06, alpha_hi=0.45,
#                   alpha_max=0.90, cmap="inferno"):
#     """RGBA image for an index map: color from `cmap`, opacity ramping
#     from 0 below alpha_lo to alpha_max at alpha_hi (a smoothstep), NaN
#     fully transparent. alpha_lo sits above the estimator noise floor;
#     alpha_hi at the gate ramp end, so anything visually solid is at
#     least half a quantum."""
#     cm = plt.get_cmap(cmap)
#     v = np.clip(np.nan_to_num(Cmap, nan=0.0) / vmax, 0, 1)
#     rgba = cm(v)
#     t = np.clip((np.nan_to_num(Cmap, nan=0.0) - alpha_lo)
#                 / (alpha_hi - alpha_lo), 0, 1)
#     t = t * t * (3 - 2 * t)                     # smoothstep
#     rgba[..., 3] = alpha_max * t
#     return rgba
#
#
# def overlay_index(ax, case, cfg, wfrac=0.5, show_mask_hatch=True,
#                   show_core_path=True, cbar_mappable=None):
#     """Pattern + alpha-ramped index overlay + hatched exclusions on ax.
#     Returns a ScalarMappable for a shared colorbar."""
#     disp = eroded_interior(case, cfg)
#     ext, sl = crop_extent(disp, case["x"], case["y"])
#     ax.imshow(np.ma.masked_where(~disp, case["u"])[sl], extent=ext,
#               origin="lower", aspect="auto", cmap="gray",
#               rasterized=True)
#     Cmap, extc = index_map(case, cfg, wfrac=wfrac)
#     rgba = index_to_rgba(Cmap)
#     ax.imshow(rgba, extent=extc, origin="lower", aspect="auto",
#               interpolation="nearest", rasterized=True)
#     if show_mask_hatch and np.isnan(Cmap).any():
#         mask = np.isnan(Cmap).astype(float)
#         xe = np.linspace(extc[0], extc[1], Cmap.shape[1])
#         ye = np.linspace(extc[2], extc[3], Cmap.shape[0])
#         cs = ax.contourf(xe, ye, mask, levels=[0.5, 1.5],
#                          colors="none", hatches=["...."])
#         for coll in getattr(cs, "collections", [cs]):
#             try:
#                 coll.set_edgecolor((0.55, 0.55, 0.55, 0.55))
#                 coll.set_linewidth(0.0)
#             except AttributeError:
#                 pass
#     if show_core_path:
#         ax.plot(case["x"], case["y"][case["rows_s"]], color="tab:cyan",
#                 lw=0.6, ls=(0, (4, 3)), alpha=0.8)
#     ax.set_xlim(ext[0], ext[1])
#     ax.set_ylim(ext[2], ext[3])
#     sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 1.1),
#                                cmap="inferno")
#     return sm
#
#
# def fig_index_maps_v2(cases, cfg, out_dir, mu):
#     """Drop-in replacement for fig_index_maps: same 2x2 content, with
#     the alpha-ramped overlay, hatched exclusions, a single shared
#     colorbar, and matched profile axes."""
#     fig, axs = plt.subplots(2, 2, figsize=(11.6, 5.6),
#                             constrained_layout=True,
#                             gridspec_kw=dict(width_ratios=[1.35, 1.0]))
#     sm = None
#     for row, case in enumerate(cases):
#         ax = axs[row, 0]
#         sm = overlay_index(ax, case, cfg)
#         label = ("defect-free knee" if case["field"] == "knee"
#                  else "PN multi-dislocation")
#         ax.set_title(f"{label} (gate = {case['gate_used']})", fontsize=9)
#         ax.set_ylabel("$y$")
#         if row == 1:
#             ax.set_xlabel("$x$")
#
#         ax = axs[row, 1]
#         xin = np.abs(case["x"]) < 0.35 * case["Lx"]
#         ax.plot(case["x"][xin], case["C"][xin], "k-", lw=1.2)
#         for yv, lab in ((0.0, "0"), (1.0, "1")):
#             ax.axhline(yv, color="0.6", lw=0.6, ls="--")
#         ax.axhspan(0.25, 0.5, color="tab:orange", alpha=0.08, lw=0)
#         for xm in case["x_defects"]:
#             if case["x"][xin].min() < xm < case["x"][xin].max():
#                 ax.axvline(xm, color="k", lw=0.4, alpha=0.25)
#         ax.set_ylim(-0.08, 1.35)
#         ax.set_ylabel("$|C|/2\\pi$ on the core")
#         med = float(np.median(case["C"][xin]))
#         ax.set_title(f"on-core profile,  median = {med:.2f}", fontsize=9)
#         if row == 1:
#             ax.set_xlabel("$x$")
#     cb = fig.colorbar(sm, ax=axs[:, 0], shrink=0.8, pad=0.01)
#     cb.set_label("$|C|/2\\pi$ (transparent where $\\approx 0$;\n"
#                  "dotted hatch = masked edge-riding windows)", fontsize=8)
#     fig.suptitle(f"defect index $|C|/2\\pi$, $\\mu={mu:.2f}$; shaded band "
#                  "in the profiles = the line-score ramp $[0.25,0.5]$",
#                  fontsize=9)
#     fig.savefig(out_dir / f"index_maps_v2_mu{mu:.2f}.png",
#                 bbox_inches="tight", dpi=200)
#     plt.close(fig)



"""
index_overlay.py  cleaner rendering of the defect index over the pattern.

Design rule: the overlay encodes |C|/2pi in OPACITY as well as color, so
a zero index is fully transparent and the pattern underneath stays crisp.
A defect-free knee then displays as a clean pattern (the claim "no fake
topology anywhere" is the picture itself), while a PN boundary displays
a saturated band on the core. Masked windows (edge-riding loops) render
as a light dotted hatch, not as filled gray blocks, so exclusions read
as "not measured" rather than as data. The coarse window grid is shown
with nearest-neighbor cells on purpose: each cell IS one loop median,
and smoothing it for display would misrepresent the estimator.

Importable: overlay_index(ax, ...) for reuse in the pipeline dashboards,
fig_index_maps_v2(...) as a drop-in replacement for
defect_index_figures.fig_index_maps.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from defect_index_figures import (index_map, eroded_interior, crop_extent)


def index_to_rgba(Cmap, vmax=1.1, alpha_lo=0.06, alpha_hi=0.45,
                  alpha_max=0.90, cmap="inferno"):
    """RGBA image for an index map: color from `cmap`, opacity ramping
    from 0 below alpha_lo to alpha_max at alpha_hi (a smoothstep), NaN
    fully transparent. alpha_lo sits above the estimator noise floor;
    alpha_hi at the gate ramp end, so anything visually solid is at
    least half a quantum."""
    cm = plt.get_cmap(cmap)
    v = np.clip(np.nan_to_num(Cmap, nan=0.0) / vmax, 0, 1)
    rgba = cm(v)
    t = np.clip((np.nan_to_num(Cmap, nan=0.0) - alpha_lo)
                / (alpha_hi - alpha_lo), 0, 1)
    t = t * t * (3 - 2 * t)                     # smoothstep
    rgba[..., 3] = alpha_max * t
    return rgba


def overlay_index(ax, case, cfg, wfrac=0.5, show_mask_hatch=True,
                  show_core_path=True, cbar_mappable=None):
    """Pattern + alpha-ramped index overlay + hatched exclusions on ax.
    Returns a ScalarMappable for a shared colorbar."""
    disp = eroded_interior(case, cfg)
    ext, sl = crop_extent(disp, case["x"], case["y"])
    ax.imshow(np.ma.masked_where(~disp, case["u"])[sl], extent=ext,
              origin="lower", aspect="auto", cmap="gray",
              rasterized=True)
    Cmap, extc = index_map(case, cfg, wfrac=wfrac)
    rgba = index_to_rgba(Cmap)
    ax.imshow(rgba, extent=extc, origin="lower", aspect="auto",
              interpolation="nearest", rasterized=True)
    if show_mask_hatch and np.isnan(Cmap).any():
        mask = np.isnan(Cmap).astype(float)
        xe = np.linspace(extc[0], extc[1], Cmap.shape[1])
        ye = np.linspace(extc[2], extc[3], Cmap.shape[0])
        cs = ax.contourf(xe, ye, mask, levels=[0.5, 1.5],
                         colors="none", hatches=["...."])
        for coll in getattr(cs, "collections", [cs]):
            try:
                coll.set_edgecolor((0.55, 0.55, 0.55, 0.55))
                coll.set_linewidth(0.0)
            except AttributeError:
                pass
    if show_core_path:
        ax.plot(case["x"], case["y"][case["rows_s"]], color="tab:cyan",
                lw=0.6, ls=(0, (4, 3)), alpha=0.8)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 1.1),
                               cmap="inferno")
    return sm


def fig_index_maps_v2(cases, cfg, out_dir, mu):
    """Drop-in replacement for fig_index_maps: same 2x2 content, with
    the alpha-ramped overlay, hatched exclusions, a single shared
    colorbar, and matched profile axes."""
    fig, axs = plt.subplots(2, 2, figsize=(11.6, 5.6),
                            constrained_layout=True,
                            gridspec_kw=dict(width_ratios=[1.35, 1.0]))
    sm = None
    for row, case in enumerate(cases):
        ax = axs[row, 0]
        sm = overlay_index(ax, case, cfg)
        label = ("defect-free knee" if case["field"] == "knee"
                 else "PN multi-dislocation")
        ax.set_title(f"{label} (gate = {case['gate_used']})", fontsize=9)
        ax.set_ylabel("$y$")
        if row == 1:
            ax.set_xlabel("$x$")

        ax = axs[row, 1]
        xin = np.abs(case["x"]) < 0.35 * case["Lx"]
        ax.plot(case["x"][xin], case["C"][xin], "k-", lw=1.2)
        for yv, lab in ((0.0, "0"), (1.0, "1")):
            ax.axhline(yv, color="0.6", lw=0.6, ls="--")
        ax.axhspan(0.25, 0.5, color="tab:orange", alpha=0.08, lw=0)
        for xm in case["x_defects"]:
            if case["x"][xin].min() < xm < case["x"][xin].max():
                ax.axvline(xm, color="k", lw=0.4, alpha=0.25)
        ax.set_ylim(-0.08, 1.35)
        ax.set_ylabel("$|C|/2\\pi$ on the core")
        med = float(np.median(case["C"][xin]))
        ax.set_title(f"on-core profile,  median = {med:.2f}", fontsize=9)
        if row == 1:
            ax.set_xlabel("$x$")
    cb = fig.colorbar(sm, ax=axs[:, 0], shrink=0.8, pad=0.01)
    cb.set_label("$|C|/2\\pi$ (transparent where $\\approx 0$;\n"
                 "dotted hatch = masked edge-riding windows)", fontsize=8)
    fig.suptitle(f"defect index $|C|/2\\pi$, $\\mu={mu:.2f}$; shaded band "
                 "in the profiles = the line-score ramp $[0.25,0.5]$",
                 fontsize=9)
    fig.savefig(out_dir / f"index_maps_mu{mu:.2f}.png",
                bbox_inches="tight", dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------
# Driver: produces index_maps_v2_mu<mu>.png for each mu
# -----------------------------------------------------------------------

def run(cfg):
    from pathlib import Path
    from defect_index_figures import prepare_case
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for mu in cfg.mus:
        cases = [prepare_case(cfg, "knee", mu),
                 prepare_case(cfg, "pismen", mu)]
        fig_index_maps_v2(cases, cfg, out_dir, mu)
        print(f"mu={mu:.2f}: index_maps_v2 written")


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Alpha-ramped defect index maps (v2 rendering)")
    p.add_argument("--out", type=str, default="results/index_maps_v2")
    p.add_argument("--mus", type=float, nargs="+", default=[0.90])
    p.add_argument("--R", type=float, default=0.5)
    p.add_argument("--R_out", type=float, default=0.2)
    p.add_argument("--Nx", type=int, default=512)
    p.add_argument("--Ny", type=int, default=256)
    p.add_argument("--Lx", type=float, default=40 * np.pi)
    p.add_argument("--Ly", type=float, default=20 * np.pi)
    p.add_argument("--sigma", type=float, default=np.pi / 2)
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
    import sys
    if len(sys.argv) == 1:
        class _Args:
            out = "results/index_maps"
            mus = [0.90]
            R, R_out = 0.5, 0.2
            Nx, Ny = 512, 256
            Lx, Ly = 40 * np.pi, 20 * np.pi
            sigma = np.pi / 2
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
