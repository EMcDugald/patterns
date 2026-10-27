# pipelines/c0_loss_surface_fig.py
"""
The least-squares valley of the forced pair {S, B} on the RCN knee
control, drawn from the columns saved by c0_dilution_control.py
(columns_beta<b>.npz: both routes at the same query centres).

For a grid of coefficient pairs (c_S, c_B) the misfit is
    M(c) = sum over centres of (b - c_S S - c_B B)^2 / sum b^2
(so M = 1 - R^2 of that coefficient pair on the training half). The
contours are drawn for the exact route and for the estimator route,
with the truth (2, -1), each route's own minimum, and the Gram
eigen-directions through the truth. Panels: one beta0 per column.

Outputs: <run folder>/fig_loss_surface.png

Changelog
---------
v1.0  10 Oct 2026, companion figure for workstream C0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent


def misfit_grid(X, b, cS, cB):
    G = X.T @ X; h = X.T @ b; bb = b @ b
    CS, CB = np.meshgrid(cS, cB)
    C = np.stack([CS.ravel(), CB.ravel()], 1)
    M = (bb - 2 * C @ h + np.einsum("ni,ij,nj->n", C, G, C)) / bb
    return M.reshape(CS.shape)


def main(args):
    run = Path(args.run) if Path(args.run).is_absolute() else _HERE / args.run
    betas = [float(v) for v in args.beta0s.split(",")]
    fig, axs = plt.subplots(1, len(betas), figsize=(4.6 * len(betas), 4.2))
    axs = np.atleast_1d(axs)
    cS = np.linspace(-0.5, 3.0, 281); cB = np.linspace(-2.5, 0.5, 241)
    for ax, b0 in zip(axs, betas):
        Z = np.load(run / ("columns_beta%g.npz" % b0))
        xs = Z["xs"]; tr = xs < np.median(xs)
        Xe = np.column_stack([Z["exact_w_div_k"] + Z["exact_kgrad_w"], Z["exact_Lap_div_k"]])[tr]
        Xm = np.column_stack([Z["est_w_div_k"] + Z["est_kgrad_w"], Z["est_Lap_div_k"]])[tr]
        be, bm = Z["exact_b"][tr], Z["est_b"][tr]
        Me = misfit_grid(Xe, be, cS, cB); Mm = misfit_grid(Xm, bm, cS, cB)
        ce = np.linalg.lstsq(Xe, be, rcond=None)[0]; cm = np.linalg.lstsq(Xm, bm, rcond=None)[0]
        levels = [0.002, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
        ax.contour(cS, cB, Me, levels=levels, colors="0.35", linewidths=0.9)
        ax.contour(cS, cB, Mm, levels=levels, colors="C3", linewidths=0.9, linestyles="--")
        lam, U = np.linalg.eigh(Xe.T @ Xe)
        for j, (col, lab) in enumerate((("C0", "weak axis"), ("C2", "strong axis"))):
            d = U[:, j] * 1.2
            ax.plot([2 - d[0], 2 + d[0]], [-1 - d[1], -1 + d[1]], color=col, lw=1.2, label=lab)
        ax.plot(2, -1, "k*", ms=11, label="truth (2, -1)")
        ax.plot(ce[0], ce[1], "o", color="0.35", ms=6, label="exact-route minimum (%.2f, %.2f)" % tuple(ce))
        ax.plot(cm[0], cm[1], "s", color="C3", ms=6, label="estimator-route minimum (%.2f, %.2f)" % tuple(cm))
        ax.set_xlabel("c_S (stretching, tied pair)"); ax.set_ylabel("c_B (bending)")
        ax.set_title("beta0 %.2f: misfit 1 - R^2 (solid exact, dashed mg)\ncurvatures %.3g (weak) / %.3g (strong)" %
                     (b0, lam[0], lam[1]), fontsize=9)
        ax.legend(fontsize=6, loc="lower left"); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(run / "fig_loss_surface.png", dpi=150); plt.close(fig)
    print("wrote", run / "fig_loss_surface.png")
    return 0


class _Args:
    run = "results/c0_dilution_control/1010_c0_dilution_rad1.5_th3_b01.8-3-2.5-1.4-1.2-1.05_b0sweep"
    beta0s = "3,1.8,1.2,1.05"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in dir(_Args) if not a.startswith("_")]:
        p.add_argument("--" + k, type=type(getattr(_Args, k)), default=getattr(_Args, k))
    sys.exit(main(p.parse_args()))
