# pipelines/adaptive_introspection.py
"""
Introspection dashboard for the adaptive anisotropic uHu extraction.

Purpose: NOT a validation script (knee_line_energy_adaptive and
pismen_line_energy_adaptive already score the method); this one opens the
hood. For an analytic knee and/or a Pismen dislocation chain it shows the
intermediate objects the fixed-point loop actually computes:

  <case>_guidance.png   (a) seed (isotropic-pass) coherence, (b) final
                        coherence, (c) slowness director quiver over the
                        smoothed k2 field (with the stripe direction in
                        grey for contrast -- at the core they disagree,
                        which is the whole point of orient="slowness"),
                        (d) the patch-level gate value g actually used by
                        the smoother (tile resolution, not pixel).
  <case>_kernels.png    pattern with the 1-sigma kernel ellipse drawn at
                        every other patch center, colored by gate value,
                        plus a gallery of the real-space kernels at three
                        probe points (far field / on core / off-core).
  <case>_iterations.png iter_delta convergence curve; per-iteration
                        midline profiles of k1 and of the director angle
                        (does the loop converge, and toward what).
  <case>_midline.png    core-row profiles, isotropic control vs adaptive:
                        k1, k2, A, and chi = lam2/lam1.

The per-patch kernel table is reconstructed by replicating, line for
line, the decisions macro_aniso_field_v2 makes (director_orientation on
the slowness director, median coherence with optional coh_window, legacy
or core_aware gate). One honest caveat: the table is built from the FINAL
iterate's director/coherence, whereas the last smoothing pass inside the
loop used the previous iterate's; after convergence (small iter_delta)
these coincide to the plotted accuracy.

Conventions: _Args block at the bottom for PyCharm, CLI mirrors it.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import cm, colors as mcolors

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "src", _HERE / "src"):
    if _p.exists():
        sys.path.insert(0, str(_p))

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
        slowness_director,
    )
except ImportError:
    from uhu_adaptive import (
        compute_uhu_ops_adaptive, derived_kernel_params_mu,
        slowness_director,
    )

try:
    from utils.anisotropic_macro import (
        coherence_from_eigs, director_orientation, _kernel_transfer,
    )
except ImportError:
    from anisotropic_macro import (
        coherence_from_eigs, director_orientation, _kernel_transfer,
    )

try:
    from paper_style import use_paper_style
    use_paper_style()
except ImportError:
    pass

from scipy import special


# -----------------------------------------------------------------------
# Analytic fields (same constructions as the two line-energy scripts)
# -----------------------------------------------------------------------

def field_knee(mu, x, y, Lx):
    """Exact Hopf-Cole knee, k1 nudged x-commensurate; returns u, k1, k2."""
    k1 = np.sqrt(1.0 - mu * mu)
    k1 = 2.0 * np.pi * max(1, round(k1 * Lx / (2.0 * np.pi))) / Lx
    k2 = np.sqrt(max(1.0 - k1 * k1, 1e-12))
    X, Y = np.meshgrid(x, y)
    theta = k1 * X + np.logaddexp(k2 * Y, -k2 * Y)   # log(2 cosh(k2 y))
    return np.cos(theta), k1, k2


def pismen_on_grid(X, Y, x_defects, amp=1.0):
    """Pismen outer solution (verbatim construction from the pipelines)."""
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


def field_pismen(mu, n_defects, dx_target, Ly, Ny,
                 edge_margin_periods=2.0, stagger=True):
    """Pismen chain on a (staggered, by default) periodic grid."""
    k1 = np.sqrt(1.0 - mu * mu)     # NOTE: k1 = cos(alpha), L = pi/k1
    L = np.pi / k1
    Lx = (n_defects - 1 + 2 * edge_margin_periods) * L
    Nx = int(np.ceil(Lx / dx_target))
    Nx += Nx % 2
    x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
    if stagger:
        y = y + 0.5 * (Ly / Ny)     # never sample the singular y=0 row
    x_defects = L * (np.arange(n_defects) - (n_defects - 1) / 2.0)
    X, Y = np.meshgrid(x, y)
    _, u = pismen_on_grid(X, Y, x_defects, amp=1.0)
    return u, x, y, k1, L, x_defects, Lx, Nx


# -----------------------------------------------------------------------
# Per-patch kernel table: replicate macro_aniso_field_v2's decisions
# -----------------------------------------------------------------------

def patch_kernel_table(d1, d2, coh, Lx, Ly, sigma, aspect_max, sharpen,
                       patch, stride=None, gate="legacy", coh_window=None,
                       scale=1.0):
    """
    One row per tile: (y_center, x_center [physical], theta_long,
    s_par, s_perp, g). Mirrors the loop in macro_aniso_field_v2 exactly:
    doubled-angle circular mean of the (slowness) director over the tile,
    +pi/2 for the long axis, median coherence (optionally over a centered
    coh_window sub-window), legacy/core_aware gate, area-redistribution
    axis formulas.
    """
    ny, nx = coh.shape
    stride = stride or patch // 2
    sigma_eff = sigma * scale
    dx = Lx / nx
    dy = Ly / ny

    ys = list(range(0, ny - patch + 1, stride)) or [0]
    xs = list(range(0, nx - patch + 1, stride)) or [0]
    if ys[-1] != ny - patch:
        ys.append(max(ny - patch, 0))
    if xs[-1] != nx - patch:
        xs.append(max(nx - patch, 0))

    rows = []
    for y0 in ys:
        for x0 in xs:
            sl = (slice(y0, y0 + patch), slice(x0, x0 + patch))
            if coh[sl].shape != (patch, patch):
                continue
            k1p, k2p, cp = d1[sl], d2[sl], coh[sl]
            weight = np.hypot(k1p, k2p)
            theta_k = director_orientation(k1p, k2p, weight=weight)
            theta_long = theta_k + np.pi / 2.0
            if coh_window is not None and coh_window < patch:
                off = (patch - coh_window) // 2
                c_sub = cp[off:off + coh_window, off:off + coh_window]
            else:
                c_sub = cp
            c_local = float(np.median(c_sub))
            g = c_local if gate == "legacy" else (1.0 - c_local)
            s_par = sigma_eff * (1.0 + (aspect_max - 1.0) * g)
            s_perp = sigma_eff / (1.0 + (sharpen - 1.0) * g)
            rows.append(((y0 + patch / 2) * dy, (x0 + patch / 2) * dx,
                         theta_long, s_par, s_perp, g))
    return np.array(rows)   # columns: yc, xc, theta_long, s_par, s_perp, g


def realspace_kernel(theta, s_par, s_perp, half=8.0, n=161, kind="gaussian"):
    """Small real-space image of the frozen kernel, for the gallery row."""
    L = 2.0 * half
    tf = _kernel_transfer((n, n), theta, s_par, s_perp, L, L, kind=kind)
    ker = np.real(np.fft.ifft2(tf))
    ker = np.fft.fftshift(ker)
    return ker / ker.max()


# -----------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------

def _imshow(ax, f, extent, title, mask=None, **kw):
    if mask is not None:
        f = np.ma.masked_where(~mask, f)
    im = ax.imshow(f, extent=extent, origin="lower", aspect="auto",
                   rasterized=True, **kw)
    ax.set_title(title, fontsize=9)
    return im


def fig_guidance(out, u, ops, seed_coh, d1, d2, x, y, Lx, Ly, sigma,
                 interior, tab, patch, deriv_fun):
    extent = [x[0], x[-1], y[0], y[-1]]
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 6.4),
                            constrained_layout=True)

    im = _imshow(axs[0, 0], seed_coh, extent, "coherence c, isotropic seed",
                 mask=interior, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=axs[0, 0], shrink=0.85)

    im = _imshow(axs[0, 1], ops["coh"], extent, "coherence c, final iterate",
                 mask=interior, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=axs[0, 1], shrink=0.85)

    # slowness director (black) vs stripe direction (grey) over smoothed k2
    ax = axs[1, 0]
    k2s = macro(ops["k2_orig"], sigma, Lx, Ly)
    im = _imshow(ax, k2s, extent, "slowness director (black) vs stripe "
                 "dir (grey), over smoothed $k_2$", mask=interior,
                 cmap="magma")
    fig.colorbar(im, ax=ax, shrink=0.85)
    ny, nx = u.shape
    sy, sx = max(ny // 14, 1), max(nx // 22, 1)
    sl = (slice(sy // 2, ny, sy), slice(sx // 2, nx, sx))
    Xq, Yq = np.meshgrid(x, y)
    qm = interior[sl]
    n1 = np.hypot(d1, d2) + 1e-15
    ax.quiver(Xq[sl][qm], Yq[sl][qm], (ops["k1_orig"] / (ops["k_vec"] + 1e-15))[sl][qm],
              (ops["k2_orig"] / (ops["k_vec"] + 1e-15))[sl][qm],
              color="0.7", angles="xy", pivot="mid",
              headwidth=2.5, width=0.003, alpha=0.8)
    ax.quiver(Xq[sl][qm], Yq[sl][qm], (d1 / n1)[sl][qm], (d2 / n1)[sl][qm],
              color="k", angles="xy", pivot="mid",
              headwidth=2.5, width=0.003)

    # patch-level gate values as a tile image
    ax = axs[1, 1]
    yc = np.unique(tab[:, 0]) + y[0]
    xc = np.unique(tab[:, 1]) + x[0]
    G = np.full((len(yc), len(xc)), np.nan)
    for (yy, xx, _, _, _, g) in tab:
        iy = int(np.argmin(np.abs(yc - (yy + y[0]))))
        ix = int(np.argmin(np.abs(xc - (xx + x[0]))))
        G[iy, ix] = g
    im = ax.imshow(G, extent=[xc[0], xc[-1], yc[0], yc[-1]],
                   origin="lower", aspect="auto", cmap="cividis",
                   vmin=0, vmax=1)
    ax.set_title(f"gate value g per tile (patch={patch}, "
                 "stride=patch/2)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85)

    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_kernels(out, u, tab, x, y, interior, probes, kind="gaussian"):
    extent = [x[0], x[-1], y[0], y[-1]]
    fig = plt.figure(figsize=(11.5, 6.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.1, 1.0])

    ax = fig.add_subplot(gs[0, :])
    ax.imshow(np.ma.masked_where(~interior, u), extent=extent,
              origin="lower", aspect="auto", cmap="gray",
              vmin=-1, vmax=1, rasterized=True)
    cmap = cm.plasma
    norm = mcolors.Normalize(vmin=0, vmax=1)
    for i, (yy, xx, th, sp, sq, g) in enumerate(tab):
        if i % 2:               # every other tile, else it is unreadable
            continue
        e = Ellipse((xx + x[0], yy + y[0]), width=2 * sp, height=2 * sq,
                    angle=np.degrees(th), fill=False,
                    color=cmap(norm(g)), lw=1.1, alpha=0.9)
        ax.add_patch(e)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, shrink=0.8, label="gate value g")
    ax.set_title("1-sigma kernel ellipse at every other patch center "
                 "(color = gate value)", fontsize=9)

    # gallery: real-space kernels at three probe tiles
    for j, (label, row) in enumerate(probes):
        yy, xx, th, sp, sq, g = row
        axk = fig.add_subplot(gs[1, j])
        half = max(2.5 * sp, 4.0)
        ker = realspace_kernel(th, sp, sq, half=half, kind=kind)
        axk.imshow(ker, extent=[-half, half, -half, half], origin="lower",
                   cmap="magma")
        axk.set_title(f"{label}\n(x={xx + x[0]:.1f}, y={yy + y[0]:.1f}), "
                      f"g={g:.2f}\n$s_\\parallel$={sp:.2f}, "
                      f"$s_\\perp$={sq:.2f}, $\\theta$={np.degrees(th):.0f}"
                      "$^\\circ$", fontsize=8)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_iterations(out, history, x, row0, interior):
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.4),
                            constrained_layout=True)
    n_hist = len(history)
    cmap = cm.viridis

    ax = axs[0]
    deltas = history[-1].get("iter_delta", None)
    # history entries are per-iteration dicts; iter_delta lives on the
    # final output of compute_uhu_ops_adaptive, so recompute if missing
    if deltas is None or np.ndim(deltas) == 0 or len(deltas) == 0:
        deltas = []
        for a, b in zip(history[:-1], history[1:]):
            m = interior
            deltas.append(np.sqrt(np.mean(
                (b["k1_orig"] - a["k1_orig"])[m] ** 2
                + (b["k2_orig"] - a["k2_orig"])[m] ** 2)))
    ax.semilogy(np.arange(1, len(deltas) + 1), deltas, "o-")
    ax.set_xlabel("iteration")
    ax.set_ylabel("interior RMS $\\Delta k$")
    ax.set_title("fixed-point convergence", fontsize=9)

    ax = axs[1]
    for i, h in enumerate(history):
        ax.plot(x, h["k1_orig"][row0, :], lw=1.0,
                color=cmap(i / max(n_hist - 1, 1)),
                label=("seed (iso)" if i == 0 else f"iter {i}")
                if i in (0, n_hist - 1) else None)
    ax.set_xlabel("$x$"); ax.set_ylabel("$k_1$ on core row")
    ax.legend(fontsize=8)
    ax.set_title("midline $k_1$ per iteration", fontsize=9)

    ax = axs[2]
    for i, h in enumerate(history):
        ang = np.degrees(np.arctan2(h["k2_orig"], h["k1_orig"]))[row0, :]
        ax.plot(x, ang, lw=1.0, color=cmap(i / max(n_hist - 1, 1)))
    ax.set_xlabel("$x$"); ax.set_ylabel("director angle (deg)")
    ax.set_title("midline director angle per iteration", fontsize=9)

    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_midline(out, ops_ada, ops_iso, x, row0, k1_true=None):
    fig, axs = plt.subplots(1, 4, figsize=(14.4, 3.2),
                            constrained_layout=True)
    def chi_of(o):
        with np.errstate(invalid="ignore", divide="ignore"):
            return o["lam2"] / np.where(np.abs(o["lam1"]) < 1e-14,
                                        np.nan, o["lam1"])
    panels = [
        ("$k_1$", lambda o: o["k1_orig"][row0, :], k1_true),
        ("$k_2$", lambda o: o["k2_orig"][row0, :], None),
        ("$A$", lambda o: o["A"][row0, :], None),
        ("$\\chi=\\lambda_2/\\lambda_1$", lambda o: chi_of(o)[row0, :],
         None),
    ]
    for ax, (lab, f, ref) in zip(axs, panels):
        ax.plot(x, f(ops_iso), "r-", lw=1.0, label="isotropic")
        ax.plot(x, f(ops_ada), "b-", lw=1.0, label="adaptive")
        if ref is not None:
            ax.axhline(ref, color="k", ls=":", lw=0.8, label="far-field")
        ax.set_xlabel("$x$"); ax.set_ylabel(lab)
        ax.legend(fontsize=7)
    fig.suptitle("core-row profiles, isotropic control vs adaptive",
                 fontsize=10)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------

def run_case(case, mu, cfg, out_root):
    dx_target = cfg.Lx_ref / cfg.Nx_ref
    if case == "knee":
        Nx = cfg.Nx_ref
        Lx = cfg.Lx_ref
        x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
        y = np.linspace(-cfg.Ly / 2, cfg.Ly / 2, cfg.Ny, endpoint=False)
        u, k1, k2 = field_knee(mu, x, y, Lx)
    else:
        u, x, y, k1, L, x_defects, Lx, Nx = field_pismen(
            mu, cfg.n_defects, dx_target, cfg.Ly, cfg.Ny,
            stagger=cfg.stagger)

    ramp = build_rectangular_ramp_smooth(
        x, y, xmargin=cfg.xmargin, ymargin=cfg.ymargin,
        tanhscale=cfg.tanhscale)
    interior = ramp >= 0.999
    row0 = int(np.argmin(np.abs(y)))
    Ly = cfg.Ly

    aspect, sharpen = derived_kernel_params_mu(mu, cfg.sigma,
                                               delta=cfg.leak_delta)
    print(f"[{case} mu={mu}] auto kernel: aspect_max={aspect:.2f} "
          f"sharpen={sharpen:.2f} (delta={cfg.leak_delta})")

    ops = compute_uhu_ops_adaptive(
        u, x, y, cfg.sigma, SpectralDerivs, ramp,
        n_iter=cfg.n_iter, aspect_max=aspect, sharpen=sharpen,
        damping=cfg.damping, patch=cfg.patch, orient="slowness",
        gate=cfg.gate, coh_window=cfg.coh_window, with_history=True)
    history = ops["history"]
    seed_coh = coherence_from_eigs(history[0]["lam1"], history[0]["lam2"])

    ops_iso = compute_uhu_ops_adaptive(
        u, x, y, cfg.sigma, SpectralDerivs, ramp,
        n_iter=1, aspect_max=1.0, sharpen=1.0,
        damping=cfg.damping, patch=cfg.patch, orient="slowness")

    # final-iterate guidance fields, as the smoother would see them
    d1, d2 = slowness_director(ops["k1_orig"], ops["k2_orig"], cfg.sigma,
                               Lx, Ly, SpectralDerivs, macro)
    tab = patch_kernel_table(d1, d2, ops["coh"], Lx, Ly, cfg.sigma,
                             aspect, sharpen, cfg.patch, gate=cfg.gate,
                             coh_window=cfg.coh_window)

    # probe tiles: far field (max |y_c|), on core (min |y_c|, center x),
    # and off-core at the same x but ~Ly/4 up
    ycs = tab[:, 0] + y[0]
    xcs = tab[:, 1] + x[0]
    i_far = int(np.argmax(np.abs(ycs)))
    mid = np.abs(xcs) < 0.15 * Lx
    i_core = int(np.argmin(np.abs(ycs) + 1e6 * (~mid)))
    off = np.abs(np.abs(ycs) - 0.25 * Ly)
    i_off = int(np.argmin(off + 1e6 * (~mid)))
    probes = [("far field", tab[i_far]), ("on core", tab[i_core]),
              ("off core", tab[i_off])]

    tag = (f"{case}_mu{mu:.2f}_sig{cfg.sigma:.3f}_d{cfg.leak_delta:.3f}"
           f"_p{cfg.patch}_it{cfg.n_iter}_g-{cfg.gate}"
           + (f"_cw{cfg.coh_window}" if cfg.coh_window else ""))
    out = out_root / tag
    out.mkdir(parents=True, exist_ok=True)

    fig_guidance(out / "guidance.png", u, ops, seed_coh, d1, d2, x, y,
                 Lx, Ly, cfg.sigma, interior, tab, cfg.patch,
                 SpectralDerivs)
    fig_kernels(out / "kernels.png", u, tab, x, y, interior, probes)
    fig_iterations(out / "iterations.png", history, x, row0, interior)
    fig_midline(out / "midline.png", ops, ops_iso, x, row0, k1_true=k1)
    print(f"  wrote {out}/")


def run(cfg):
    out_root = Path(cfg.out)
    for case in cfg.cases:
        for mu in cfg.mus:
            run_case(case, mu, cfg, out_root)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=["knee", "pismen"],
                   choices=["knee", "pismen"])
    p.add_argument("--mus", nargs="+", type=float,
                   default=[0.85, 0.90, 0.95])
    p.add_argument("--sigma", type=float, default=np.pi / 2)
    p.add_argument("--leak_delta", type=float, default=0.05)
    p.add_argument("--patch", type=int, default=96)
    p.add_argument("--n_iter", type=int, default=5)
    p.add_argument("--damping", type=float, default=0.5)
    p.add_argument("--gate", default="legacy",
                   choices=["legacy", "core_aware"])
    p.add_argument("--coh_window", type=int, default=None)
    p.add_argument("--Lx_ref", type=float, default=125.0)
    p.add_argument("--Nx_ref", type=int, default=512)
    p.add_argument("--Ly", type=float, default=64.0)
    p.add_argument("--Ny", type=int, default=256)
    p.add_argument("--n_defects", type=int, default=9)
    p.add_argument("--stagger", action="store_true", default=True)
    p.add_argument("--xmargin", type=float, default=0.05)
    p.add_argument("--ymargin", type=float, default=0.05)
    p.add_argument("--tanhscale", type=float, default=60.0)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent
                                        / "results"
                                        / "adaptive_introspection"))
    run(p.parse_args())


class _Args:
    cases = ["knee", "pismen"]
    mus = [0.90]
    sigma = np.pi / 2
    leak_delta = 0.05
    patch = 96
    n_iter = 5
    damping = 0.5
    gate = "legacy"
    coh_window = 12
    Lx_ref = 125.0
    Nx_ref = 512
    Ly = 64.0
    Ny = 256
    n_defects = 9
    stagger = True
    xmargin = 0.05
    ymargin = 0.05
    tanhscale = 60.0
    out = str(Path(__file__).resolve().parent / "results"
              / "adaptive_introspection")


if __name__ == "__main__":
    RUN_IN_IDE = True
    if RUN_IN_IDE and len(sys.argv) == 1:
        run(_Args())
    else:
        main()