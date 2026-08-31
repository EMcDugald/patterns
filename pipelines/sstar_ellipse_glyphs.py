# pipelines/sstar_ellipse_glyphs.py
"""S* as ellipse glyphs over the pattern u = cos(theta): the intuitive
companion to the three heat-map parameter fields of sigma_window_maps.
Knee row: needle glyphs (constant, along the wall). PN row (single
Pismen-Nepomnyashchy dislocation): S* = c H^{-2} glyphs, capped on the
plateaus, collapsing and rotating around the defect.
Run: python sstar_ellipse_glyphs.py [--outdir DIR] [--tol 0.05] [--mu 0.9]
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
from utils.uhu_symbolic import (KneeAnsatz, PNAnsatz, lambdify_np,
                                sstar_closed_form, sstar_knee_needle)
try:
    from paper_style import use_paper_style; use_paper_style()
except Exception:
    pass

def main(outdir, tol, mu):
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    scale = 0.55   # display scale for glyph radii

    # ---- knee ----
    a = KneeAnsatz()
    k1v = float(np.sqrt(1 - mu**2))
    th = lambdify_np((a.x, a.y, a.k1, a.k2), a.theta)
    xs = np.linspace(-12, 12, 600); ys = np.linspace(-12, 12, 600)
    X, Y = np.meshgrid(xs, ys)
    axes[0].imshow(np.cos(th(X, Y, k1v, mu)), extent=[-12, 12, -12, 12],
                   origin="lower", cmap="gray", vmin=-1, vmax=1)
    Sn = sstar_knee_needle(mu, tol=tol, s_yy=0.15**2)
    wx, wy = np.sqrt(Sn[0, 0]), np.sqrt(Sn[1, 1])
    for gx in np.linspace(-9, 9, 5):
        for gy in np.linspace(-9, 9, 5):
            axes[0].add_patch(Ellipse((gx, gy), 2*scale*wx, 2*scale*wy,
                                      angle=0, fill=False, color="C3", lw=1.6))
    axes[0].set_title(rf"knee $\mu={mu}$: needle glyphs"
                      rf" ($\sigma_x={wx:.2f}$, $\sigma_y\to0$)")

    # ---- PN ----
    b = PNAnsatz(numeric_weights=True)
    thp = lambdify_np((b.x, b.y), b.theta)
    fk = {nm: lambdify_np((b.x, b.y), e) for nm, e in
          (("kx", b.thx), ("ky", b.thy), ("H11", b.hess[0, 0]),
           ("H12", b.hess[0, 1]), ("H22", b.hess[1, 1]))}
    xs = np.linspace(-10, 10, 600); ys = np.linspace(0.25, 10, 600)
    X, Y = np.meshgrid(xs, ys)
    axes[1].imshow(np.cos(thp(X, Y)), extent=[-10, 10, 0.25, 10],
                   origin="lower", cmap="gray", vmin=-1, vmax=1)
    for gx in np.linspace(-8.5, 8.5, 7):
        for gy in np.linspace(1.0, 9.2, 6):
            v = {nm: float(fn(gx, gy)) for nm, fn in fk.items()}
            st = sstar_closed_form(v["kx"], v["ky"], v["H11"], v["H12"],
                                   v["H22"], tol=tol, cap=9.0)
            w1, w2 = st["widths"][0], st["widths"][1]
            ang = float(np.degrees(st["angle_wide"]))
            capped = w1 >= np.sqrt(9.0) - 1e-9
            axes[1].add_patch(Ellipse((gx, gy), 2*scale*w1, 2*scale*w2,
                                      angle=ang, fill=False,
                                      color=("C0" if capped else "C3"),
                                      lw=1.6))
    axes[1].set_title("PN dislocation: $S^*=cH^{-2}$ glyphs"
                      "  (blue = cap active)")
    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    path = os.path.join(outdir, "sstar_ellipse_glyphs.png")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    print("wrote", path)

class _Args:
    outdir = "out_sstar_glyphs"; tol = 0.05; mu = 0.9

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=_Args.outdir)
    p.add_argument("--tol", type=float, default=_Args.tol)
    p.add_argument("--mu", type=float, default=_Args.mu)
    a = p.parse_args()
    main(a.outdir, a.tol, a.mu)
