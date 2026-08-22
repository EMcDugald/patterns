# pipelines/validate_adaptive_director.py
"""
Reproduces the adaptive-width director validation figure on the two analytic
fields, where the answer is known:

  knee : smooth zigzag fold, defect free -> weight must be 0, director unchanged
  pn   : Pismen dislocation row          -> weight rises only at cores

This proves the modulator earns its place before it touches simulation data:
on the knee it collapses to the small width (does nothing), on PN it applies
the large width spatially, only at the cores.

Run:  python validate_adaptive_director.py
Writes results/adaptive_director_val.png (relative path).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from utils.field_diagnostics_viz import riesz_director
from op_extract.loop_persistence import adaptive_director
from op_extract.pn_field import pismen_on_sh_grid


def make_grid(mu, n_xper=4, n_yper=6, ppw=28):
    k1 = np.sqrt(1.0 - mu**2); k2 = mu
    dx = 2.0 * np.pi / ppw
    Lx = n_xper * (2 * np.pi / max(k1, 0.2)); Ly = n_yper * (2 * np.pi / k2)
    Nx = int(round(Lx / dx)); Ny = int(round(Ly / dx))
    x = (np.arange(Nx) - Nx // 2) * dx
    y = (np.arange(Ny) - Ny // 2) * dx
    X, Y = np.meshgrid(x, y)
    return dict(x=x, y=y, X=X, Y=Y, Lx=x[-1] - x[0], Ly=y[-1] - y[0],
                k1=k1, k2=k2)


def field_knee(g):
    pp = g["k1"] * g["X"] + g["k2"] * g["Y"]
    pm = g["k1"] * g["X"] - g["k2"] * g["Y"]
    m = np.maximum(pp, pm)
    theta = m + np.log(np.exp(pp - m) + np.exp(pm - m))
    return np.cos(theta)


def field_pn(g, n_def=4):
    Lx = g["x"][-1] - g["x"][0]
    xd = np.linspace(-0.3 * Lx, 0.3 * Lx, n_def)
    return pismen_on_sh_grid(g["X"], g["Y"], xd, 1.0)[1]


def _dir_change_deg(ang, ang_ref):
    """Director angle change (mod pi), in degrees."""
    d = (ang - ang_ref + np.pi / 2) % np.pi - np.pi / 2
    return np.degrees(np.abs(d))


def run(mu=0.90, band_y=3.0, out="results/adaptive_director_val.png",
        sigma_small=0.4, sigma_large=np.pi):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    g = make_grid(mu)
    fields = {"knee": field_knee(g), "pn": field_pn(g)}

    fig, axs = plt.subplots(2, 5, figsize=(19, 7))
    ext = [g["x"][0], g["x"][-1], g["y"][0], g["y"][-1]]
    band = np.abs(g["y"]) < band_y
    for r, (nm, u) in enumerate(fields.items()):
        out_d = adaptive_director(u, g["Lx"], g["Ly"], riesz_director,
                                  sigma_small=sigma_small,
                                  sigma_large=sigma_large)
        dchg = _dir_change_deg(out_d["angle"], out_d["angle_small"])

        axs[r, 0].imshow(u, extent=ext, origin="lower", cmap="gray",
                         vmin=-1, vmax=1, aspect="auto")
        axs[r, 0].set_ylabel(nm, fontsize=11)
        axs[r, 1].imshow(np.degrees(out_d["angle_small"]), extent=ext,
                         origin="lower", cmap="twilight", vmin=-90, vmax=90,
                         aspect="auto")
        axs[r, 2].imshow(out_d["weight"], extent=ext, origin="lower",
                         cmap="magma", vmin=0, vmax=1, aspect="auto")
        axs[r, 3].imshow(np.degrees(out_d["angle"]), extent=ext,
                         origin="lower", cmap="twilight", vmin=-90, vmax=90,
                         aspect="auto")
        axs[r, 4].imshow(dchg, extent=ext, origin="lower", cmap="inferno",
                         vmin=0, vmax=20, aspect="auto")

        print(f"{nm:5s}: weight max={out_d['weight'].max():.3f}  "
              f"band-mean weight={out_d['weight'][band].mean():.3f}  "
              f"director change (deg, band max)={dchg[band].max():.2f}")
        if r == 0:
            titles = ["u", f"director @ small ({sigma_small:.2g})",
                      "blend weight w", "director blended",
                      "|change| vs small (deg)"]
            for a, t in zip(axs[r], titles):
                a.set_title(t, fontsize=9)
        for a in axs[r]:
            a.set_xticks([]); a.set_yticks([])

    fig.suptitle(f"Adaptive-width director on analytic fields (mu={mu}): "
                 "knee untouched (w=0), PN widened only at cores", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"figure -> {out}")


if __name__ == "__main__":
    run()
