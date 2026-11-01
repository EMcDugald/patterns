#!/usr/bin/env python
"""
director_walkthrough.py  v1.0 (29 Oct 2026)

The Riesz director built step by step on one real SH boundary, for the talk: each step of the construction as
its own panel, and the same steps along a one-dimensional cut straight across the stripes, where each step's
reason is visible. The steps, exactly as utils.field_diagnostics_viz.riesz_director computes them (imported
through the frozen method layer, edited nowhere):

  1. the Riesz pair  R u = (R_1 u, R_2 u):  on a plane wave R u = A sin(theta) k-hat, the quarter-shifted partner
     pointing across the stripes, with a sign that flips every half period (sin(theta) changes sign)
  2. its raw angle  psi_raw = atan2(R_2 u, R_1 u):  jumps by 180 degrees every half period
  3. the doubled angle  2 psi_raw:  the jumps of 180 become jumps of 360, that is, no jumps: smooth, except
     where the stripe direction itself is singular
  4. the weight  w = (R_1 u)^2 + (R_2 u)^2 = A^2 sin^2(theta):  zero on the crests (where the raw angle is
     noise), largest between them, small where the amplitude is small
  5. the Gaussian average at width sigma_dir:  C = G * (w cos 2 psi_raw), S = G * (w sin 2 psi_raw); the
     director angle psi = atan2(S, C) / 2 and the alignment |C + iS| / (G * w), which drops where the stripe
     direction is not single valued within the window (the defect cores)
  6. the half-turn cells: the plaquette winding of 2 psi (op_extract.t1_common.director_winding)

Outputs in <outdir>/ (names begin with dw_):
  dw_steps.png / .pdf      the eight panels on a zoom about two cores
  dw_cut.png / .pdf        the steps along a cut across the stripes in the far field (above the line)
  dw_director.png / .pdf   the director as headless line segments over the pattern, with the cells
  dw_notes.txt             far-field angle of the director against alpha, alignment at the cores, cell count
Input: one relaxed frame (the large sweep, e.g. data/sh_pgb_zigzag/1024_large_mu_sweep/raw/<mu 0.90 file>.npz,
last frame), or any movie file with --frame. Run from ~/patterns/pipelines (DIRECTOR_WALK_CLI_1029.txt).
Python 3.8 compatible.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT / "src", ROOT / "src" / "op_extract", ROOT / "src" / "utils", HERE):
    sys.path.insert(0, str(_p))
from mg_zigzag_theta_t import load_npz                                       # noqa: E402
from mg_estimator_validation import fdv                                      # noqa: E402
from op_extract import t1_common as t1c                                      # noqa: E402
import matplotlib                                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                              # noqa: E402
from matplotlib.collections import LineCollection                           # noqa: E402
from matplotlib.lines import Line2D                                          # noqa: E402
from scipy.ndimage import map_coordinates                                    # noqa: E402

VERSION = "director_walkthrough v1.0 (29 Oct 2026)"
CX, CV = "#d62728", "#1f77b4"


def steps(u, Lx, Ly, sigma):
    """every intermediate of riesz_director, in its own words"""
    mono = fdv.monogenic(u, Lx, Ly)
    R1, R2 = mono["R1"], mono["R2"]
    psi_raw = np.arctan2(R2, R1)
    w = R1 ** 2 + R2 ** 2
    C = fdv.macro_iso(w * np.cos(2 * psi_raw), sigma, Lx, Ly)
    S = fdv.macro_iso(w * np.sin(2 * psi_raw), sigma, Lx, Ly)
    wbar = fdv.macro_iso(w, sigma, Lx, Ly)
    align = np.clip(np.hypot(C, S) / np.maximum(wbar, 1e-12), 0, 1)
    psi = 0.5 * np.arctan2(S, C)
    rd = fdv.riesz_director(u, Lx, Ly, sigma)                 # the frozen function, for the check below
    assert np.allclose(rd["angle"], psi), "the steps do not reproduce riesz_director"
    q = t1c.director_winding(psi)
    return dict(R1=R1, R2=R2, psi_raw=psi_raw, w=w, C=C, S=S, wbar=wbar, align=align, psi=psi, q=q,
                A=np.hypot(u, np.hypot(R1, R2)))


def wrap_deg(a):
    return (np.degrees(a) + 180) % 360 - 180


def main(args):
    t0 = time.time()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    D = load_npz(args.npz)
    u = D["u"]
    if u.ndim == 2:
        u = u[..., None]
    i = args.frame if args.frame >= 0 else u.shape[-1] + args.frame
    u2 = np.asarray(u[..., i], float)
    x, y = np.asarray(D["x"], float), np.asarray(D["y"], float)
    md = dict(D["md"])
    mu = float(md.get("mu", args.mu))
    jr = int(md.get("gb_row_in_crop", u2.shape[0] // 2)); y_row = float(y[jr])
    nx = u2.shape[1]; ny = u2.shape[0]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    Lx, Ly = nx * dx, ny * dy
    alpha = float(np.arcsin(mu)); k1 = float(np.cos(alpha)); L = np.pi / k1
    lam = 2 * np.pi                                                   # stripe wavelength (|k| = 1)
    taper = t1c.edge_taper(x, y, frac=args.taper_frac)
    st = steps(u2 * taper, Lx, Ly, args.sigma_dir)
    X, Y = np.meshgrid(x, y)
    lines = [VERSION, "args: " + json.dumps(vars(args)), "file %s frame %d mu %.3f L %.3f" % (args.npz, i, mu, L)]

    # ---- the zoom: zoom_L spacings along the line about the centre, +- zoom_y_L across
    xm = float(x.mean())
    ix = np.abs(x - xm) <= 0.5 * args.zoom_L * L
    jy = np.abs(y - y_row) <= args.zoom_y_L * L
    sub = lambda a: a[np.ix_(jy, ix)]                                 # noqa: E731
    ext = [(x[ix][0] - xm) / L, (x[ix][-1] - xm) / L, (y[jy][0] - y_row) / L, (y[jy][-1] - y_row) / L]
    cells = np.nonzero(st["q"])
    qc = st["q"][cells]
    xc = 0.5 * (x[:-1] + x[1:])[cells[1]]; yc = 0.5 * (y[:-1] + y[1:])[cells[0]]
    keep = (np.abs(xc - xm) <= 0.5 * args.zoom_L * L) & (np.abs(yc - y_row) <= args.zoom_y_L * L)
    cxL, cyL, cq = (xc[keep] - xm) / L, (yc[keep] - y_row) / L, qc[keep]

    def cells_on(ax, size=7):
        for sel, mk, col in ((cq > 0, "^", CX), (cq < 0, "v", CV)):
            ax.plot(cxL[sel], cyL[sel], mk, ms=size, mfc=col, mec="white", mew=0.8, ls="none", zorder=6)

    plt.rcParams.update({"font.size": 8})
    fig, axs = plt.subplots(2, 4, figsize=(15.5, 7.8), constrained_layout=True)
    ax = axs.ravel()
    im = ax[0].imshow(sub(u2), origin="lower", extent=ext, cmap="gray", aspect="equal"); ax[0].set_title("1. the pattern $u = A\\cos\\theta$\n(a relaxed boundary)")
    # Riesz pair as arrows over |Ru|
    mag = np.hypot(st["R1"], st["R2"])
    ax[1].imshow(sub(mag), origin="lower", extent=ext, cmap="magma", aspect="equal", vmin=0)
    step = max(1, int(round(args.arrow_frac_lam * lam / dx)))
    sl = (slice(step // 2, None, step), slice(step // 2, None, step))
    Xs, Ys = (sub(X) - xm) / L, (sub(Y) - y_row) / L
    r1, r2 = sub(st["R1"]), sub(st["R2"])
    nr = np.hypot(r1, r2) + 1e-12
    ax[1].quiver(Xs[sl], Ys[sl], (r1 / nr)[sl], (r2 / nr)[sl], color="white", scale=28, width=0.004, pivot="mid", alpha=0.9)
    ax[1].set_title("2. the Riesz pair $\\mathcal{R}u = A\\sin\\theta\\,\\hat{\\mathbf{k}}$\narrows: direction; colour: $|\\mathcal{R}u|$")
    ax[2].imshow(wrap_deg(sub(st["psi_raw"])), origin="lower", extent=ext, cmap="twilight", vmin=-180, vmax=180, aspect="equal")
    ax[2].set_title("3. its raw angle $\\psi_{\\rm raw}$ (deg)\njumps by 180 every half period")
    ax[3].imshow(wrap_deg(2 * sub(st["psi_raw"])), origin="lower", extent=ext, cmap="twilight", vmin=-180, vmax=180, aspect="equal")
    ax[3].set_title("4. the doubled angle $2\\psi_{\\rm raw}$ (deg)\nno jumps, except at the cores")
    ax[4].imshow(sub(st["w"]) / np.percentile(st["w"], 99), origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax[4].set_title("5. the weight $w = |\\mathcal{R}u|^2 = A^2\\sin^2\\theta$\nzero on the crests, small where $A$ is small")
    ax[5].imshow(wrap_deg(2 * sub(st["psi"])), origin="lower", extent=ext, cmap="twilight", vmin=-180, vmax=180, aspect="equal")
    ax[5].set_title("6. weighted average at width %.2g\n$2\\psi = \\arg(C + iS)$ (deg)" % args.sigma_dir)
    ax[6].imshow(sub(st["align"]), origin="lower", extent=ext, cmap="cividis", vmin=0, vmax=1, aspect="equal")
    ax[6].set_title("7. alignment $|C+iS|\\,/\\,\\overline{w}$\ndrops where the direction is not single valued")
    cells_on(ax[6])
    # director segments over the pattern
    ax[7].imshow(sub(u2), origin="lower", extent=ext, cmap="gray", aspect="equal", alpha=0.45)
    stp = max(1, int(round(args.seg_frac_lam * lam / dx)))
    sl2 = (slice(stp // 2, None, stp), slice(stp // 2, None, stp))
    d1, d2 = np.cos(sub(st["psi"]))[sl2], np.sin(sub(st["psi"]))[sl2]
    hl = 0.45 * stp * dx / L
    segs = np.stack([np.stack([Xs[sl2] - hl * d1, Ys[sl2] - hl * d2], -1), np.stack([Xs[sl2] + hl * d1, Ys[sl2] + hl * d2], -1)], -2)
    ax[7].add_collection(LineCollection(segs.reshape(-1, 2, 2), colors="#1a6f9e", linewidths=1.3))
    cells_on(ax[7], size=8)
    ax[7].set_title("8. the director (a line, no arrowhead)\nand its half-turn cells")
    for a_ in ax:
        a_.set_xlim(ext[0], ext[1]); a_.set_ylim(ext[2], ext[3]); a_.tick_params(labelsize=7)
        a_.axhline(0, color="0.6", lw=0.6, ls="--")
    for a_ in axs[1]:
        a_.set_xlabel("along the line (L)")
    for a_ in axs[:, 0]:
        a_.set_ylabel("across the line (L)")
    fig.suptitle("%s   mu %.2f (%.0f deg): the Riesz director, step by step" % (Path(args.npz).stem, mu, np.degrees(alpha)), fontsize=10)
    fig.savefig(out / "dw_steps.png", dpi=160); fig.savefig(out / "dw_steps.pdf"); plt.close(fig)

    # ---- the cut: straight across the stripes in the upper far field, through x = xm
    y0 = y_row + args.cut_y_L * L
    s = np.linspace(-args.cut_half_lam * lam, args.cut_half_lam * lam, 600)
    kx, ky = np.cos(alpha), np.sin(alpha)                          # far-field normal above the line
    px, py = xm + s * kx, y0 + s * ky
    coords = [(py - y[0]) / dy, (px - x[0]) / dx]
    samp = lambda a: map_coordinates(a, coords, order=1, mode="nearest")   # noqa: E731
    cu = samp(u2); cr1, cr2 = samp(st["R1"]), samp(st["R2"])
    rn = cr1 * kx + cr2 * ky                                       # component of Ru along the far-field normal
    cpsi_raw = np.arctan2(cr2, cr1); cw = samp(st["w"]); cpsi = samp(st["psi"]); cal = samp(st["align"])
    fig, axs = plt.subplots(5, 1, figsize=(8.5, 8.2), sharex=True, constrained_layout=True)
    sl_ = s / lam
    axs[0].plot(sl_, cu, color="k", lw=1.4, label="$u$"); axs[0].plot(sl_, rn, color="#e67e22", lw=1.4, label="$\\mathcal{R}u\\cdot\\hat{\\mathbf{k}}$")
    axs[0].set_ylabel("fields"); axs[0].legend(fontsize=8, ncol=2, frameon=False)
    axs[0].set_title("along a cut across the stripes (far field above the line): the quarter-shifted partner changes sign every half period", fontsize=8.5)
    axs[1].plot(sl_, wrap_deg(cpsi_raw), ".", ms=3, color="#8e44ad"); axs[1].set_ylabel("raw angle\n$\\psi_{\\rm raw}$ (deg)"); axs[1].set_ylim(-190, 190)
    axs[1].axhline(np.degrees(alpha), color="0.6", lw=0.8, ls="--"); axs[1].axhline(np.degrees(alpha) - 180, color="0.6", lw=0.8, ls="--")
    axs[2].plot(sl_, wrap_deg(2 * cpsi_raw), ".", ms=3, color="#8e44ad"); axs[2].set_ylabel("doubled\n$2\\psi_{\\rm raw}$ (deg)"); axs[2].set_ylim(-190, 190)
    axs[2].axhline(wrap_deg(2 * alpha), color="0.6", lw=0.8, ls="--")
    axs[3].plot(sl_, cw / cw.max(), color="#27ae60", lw=1.4); axs[3].set_ylabel("weight $w$\n(normalised)")
    axs[4].plot(sl_, np.degrees(cpsi), color="#1a6f9e", lw=1.8, label="director $\\psi$"); axs[4].axhline(np.degrees(alpha), color="0.6", lw=0.8, ls="--", label="far-field angle $\\alpha$")
    ax2 = axs[4].twinx(); ax2.plot(sl_, cal, color="#7f8c8d", lw=1.0, ls=":"); ax2.set_ylim(0, 1.05); ax2.set_ylabel("alignment", color="#7f8c8d", fontsize=8)
    axs[4].set_ylabel("averaged\n$\\psi$ (deg)"); axs[4].legend(fontsize=8, frameon=False, loc="lower left")
    axs[4].set_xlabel("distance along the cut (stripe wavelengths)")
    fig.savefig(out / "dw_cut.png", dpi=160); fig.savefig(out / "dw_cut.pdf"); plt.close(fig)

    # ---- the director alone, larger
    fig, a_ = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    a_.imshow(sub(u2), origin="lower", extent=ext, cmap="gray", aspect="equal", alpha=0.45)
    a_.add_collection(LineCollection(segs.reshape(-1, 2, 2), colors="#1a6f9e", linewidths=1.4))
    cells_on(a_, size=9)
    a_.set_xlim(ext[0], ext[1]); a_.set_ylim(ext[2], ext[3]); a_.axhline(0, color="0.6", lw=0.6, ls="--")
    a_.set_xlabel("along the line (L)"); a_.set_ylabel("across the line (L)")
    a_.set_title("the director at width %.2g over the pattern, and its half-turn cells (X red, V blue)" % args.sigma_dir, fontsize=9)
    a_.legend(handles=[Line2D([], [], marker="^", ls="none", mfc=CX, mec="white", ms=8, label="X cell (+1/2 turn)"),
                       Line2D([], [], marker="v", ls="none", mfc=CV, mec="white", ms=8, label="V cell (-1/2 turn)")], fontsize=8, loc="upper right")
    fig.savefig(out / "dw_director.png", dpi=160); fig.savefig(out / "dw_director.pdf"); plt.close(fig)

    # ---- numbers
    far = (np.abs(Y - y_row) > args.far_from_L * L) & (np.abs(X - xm) < 0.4 * Lx) & (np.abs(Y - y.mean()) < 0.4 * Ly)
    if far.sum() < 50:                                                 # a short box: take the outer third instead
        far = (np.abs(Y - y_row) > 0.33 * (y.max() - y.min()) / 2) & (np.abs(X - xm) < 0.4 * Lx)
        lines.append("LOUD: far field taken as |y - y_row| > a third of the half-height (the box is shorter than %.2g L)" % args.far_from_L)
    err_far = np.degrees(np.abs(((st["psi"] - np.sign(Y - y_row) * alpha) + np.pi / 2) % np.pi - np.pi / 2))
    lines.append("far-field director angle against alpha = %.2f deg: median error %.3f deg, 99th percentile %.3f deg"
                 % (np.degrees(alpha), np.median(err_far[far]), np.percentile(err_far[far], 99)))
    lines.append("alignment: median far field %.4f; minimum within 0.25 L of a cell %.3f" % (
        np.median(st["align"][far]),
        float(min([st["align"][int(round((yy - y[0]) / dy)), int(round((xx - x[0]) / dx))] for xx, yy in zip(xc, yc)]) if len(xc) else float("nan"))))
    lines.append("cells in the zoom: %d X, %d V; whole frame within 2 L of the line: %d X, %d V" % (
        int((cq > 0).sum()), int((cq < 0).sum()),
        int(((qc > 0) & (np.abs(yc - y_row) < 2 * L)).sum()), int(((qc < 0) & (np.abs(yc - y_row) < 2 * L)).sum())))
    lines += ["", "done in %.0f s" % (time.time() - t0)]
    (out / "dw_notes.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print("outputs in %s" % out)
    return 0


class _Args:
    npz = ""                       # one frame file (movie layout); the last frame is used unless --frame
    frame = -1
    mu = 0.90                      # used only if the file's metadata has no mu
    outdir = str(Path("results") / "director_walkthrough")
    sigma_dir = 0.4                # the director width (mg's)
    taper_frac = 0.06
    zoom_L = 2.2                   # width of the zoom along the line (L)
    zoom_y_L = 0.9                 # half-height of the zoom (L)
    arrow_frac_lam = 0.14          # arrow spacing in the Riesz panel, as a fraction of the stripe wavelength
    seg_frac_lam = 0.33            # director segment spacing, as a fraction of the stripe wavelength
    cut_y_L = 1.6                  # the cut runs through (x_mid, y_row + cut_y_L L), across the stripes
    cut_half_lam = 2.0             # half-length of the cut (stripe wavelengths)
    far_from_L = 3.0               # the far field for the check: beyond this distance from the line (L)


def default_args():
    return argparse.Namespace(**{k: getattr(_Args, k) for k in vars(_Args) if not k.startswith("_")})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in vars(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    sys.exit(main(p.parse_args()))
