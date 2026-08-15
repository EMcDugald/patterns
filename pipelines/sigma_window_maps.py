# pipelines/sigma_window_maps.py
"""
Plotting layer for the optimal kernel covariance S*. The former content
(allowed-sigma window maps) is superseded by the S* field: the direction
rule (H khat)^T S (H that) = 0 with isotropizing solution S = c H^{-2},
and the scale rule k^T S k = ln(1/tol)/2, both derived and gated in
task2_averaging.py. The H S H symbolic verification below is retained;
it is the identity behind the O(eps^2) DC-mixing term that S* controls.

Derivation (verified symbolically below). For a kernel with covariance matrix
S (any centered kernel, not just Gaussian), the second-order expansion is

    <f>_S = f + (1/2) S_ab d_a d_b f + O(S^2)

Applying that to Q_ij = theta_i theta_j and using S_ab = S_ba,

    (1/2) S_ab d_a d_b (theta_i theta_j)
        = (H S H)_ij + (1/2) S_ab [ theta_iab theta_j + theta_i theta_jab ]

so the product-of-second-derivatives term generalizes from sigma^2 H^2 to

    H S H          (isotropic S = sigma^2 I recovers sigma^2 H^2)

which is the whole answer to "how does this work for an ellipse".

Design consequence. Diagonalize H = R diag(h1, h2) R^T and write the kernel
covariance in H's eigenframe as S' = R^T S R with widths s1, s2. Then

    (H S H)  in that frame has entries  h_a s_ab h_b,

i.e. each direction is penalized by the SQUARE of the Hessian eigenvalue
along it. The kernel should therefore be thin along the large-|h| axis of
Hess(theta) and may be long along the small-|h| axis. That is a different
rule from "align with k" and from "align with the core", and it is derived
rather than guessed.

The two channels then constrain DIFFERENT directions:
    channel 1 (corrugation) constrains the extent along khat
    channel 2 (DC mixing)   constrains the extent along the large-|h| axis
When those directions coincide, no kernel works, which is the W ~ 1
obstruction. When they differ, an anisotropic kernel satisfies both. On the
knee core khat = xhat while H = diag(0, k2^2), so they are orthogonal and the
window is open for every mu, reproducing handoff Section 6 from the tensor
algebra instead of from the on-core competition.

Run: python sigma_window_maps.py [--outdir DIR] [--tol1 0.05] [--tol2 0.05]
"""

import argparse
import os

import numpy as np
import sympy as sp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.uhu_symbolic import (KneeAnsatz, PNAnsatz, lambdify_np,
                                eig2x2_numeric, sstar_closed_form,
                                sstar_knee_needle)


# =====================================================================
# symbolic check of the H S H claim
# =====================================================================

def verify_HSH():
    x, y = sp.symbols("x y", real=True)
    th = sp.Function("theta")(x, y)
    c = {1: x, 2: y}
    S = sp.Matrix(2, 2, sp.symbols("S11 S12 S12 S22"))
    S[1, 0] = S[0, 1]
    H = sp.Matrix([[sp.diff(th, c[a], c[b]) for b in (1, 2)] for a in (1, 2)])
    HSH = H * S * H
    ok = True
    for i in (1, 2):
        for j in (1, 2):
            lhs = sp.Rational(1, 2) * sum(
                S[a - 1, b - 1] * sp.diff(sp.diff(th, c[i]) * sp.diff(th, c[j]),
                                          c[a], c[b])
                for a in (1, 2) for b in (1, 2))
            third = sp.Rational(1, 2) * sum(
                S[a - 1, b - 1] * (sp.diff(th, c[i], c[a], c[b]) * sp.diff(th, c[j])
                                   + sp.diff(th, c[i]) * sp.diff(th, c[j], c[a], c[b]))
                for a in (1, 2) for b in (1, 2))
            if sp.simplify(lhs - HSH[i - 1, j - 1] - third) != 0:
                ok = False
    return ok


# =====================================================================
# pointwise window maps
# =====================================================================

# =====================================================================

def build_knee(mu, n=400, half=14.0):
    a = KneeAnsatz()
    f = {nm: lambdify_np((a.x, a.y, a.k1, a.k2), e) for nm, e in
         (("kx", a.thx), ("ky", a.thy), ("H11", a.hess[0, 0]),
          ("H12", a.hess[0, 1]), ("H22", a.hess[1, 1]))}
    xs = np.linspace(-half, half, n)
    ys = np.linspace(-half, half, n)
    X, Y = np.meshgrid(xs, ys)
    k1v = np.sqrt(1 - mu**2)
    out = {nm: fn(X, Y, k1v, mu) for nm, fn in f.items()}
    out.update({"X": X, "Y": Y, "extent": [-half, half, -half, half]})
    return out


def build_pn(n=400, half=10.0, ymin=0.25):
    a = PNAnsatz(numeric_weights=True)
    f = {nm: lambdify_np((a.x, a.y), e) for nm, e in
         (("kx", a.thx), ("ky", a.thy), ("H11", a.hess[0, 0]),
          ("H12", a.hess[0, 1]), ("H22", a.hess[1, 1]))}
    xs = np.linspace(-half, half, n)
    ys = np.linspace(ymin, half, n)
    X, Y = np.meshgrid(xs, ys)
    out = {nm: fn(X, Y) for nm, fn in f.items()}
    out.update({"X": X, "Y": Y, "extent": [-half, half, ymin, half]})
    return out


def panel_row_sstar(fig, axes, d, tol, label, needle_mu=None):
    """Four panels: W for context, then the S* field (wide width, thin
    width, wide-axis angle). If needle_mu is given the row is the knee,
    whose H is singular everywhere: the closed form does not apply and
    the exact optimum is the x-needle; the panels then show the needle
    prescription as constant fields with the values printed."""
    kmag = np.hypot(d["kx"], d["ky"])
    W = np.sqrt(d["H11"]**2 + 2 * d["H12"]**2 + d["H22"]**2) \
        / np.maximum(kmag**2, 1e-30)
    ext = d["extent"]

    ax = axes[0]
    im = ax.imshow(np.clip(W, 0, 2), extent=ext, origin="lower",
                   cmap="inferno", vmin=0, vmax=2, aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{label}: $W=\\|H\\|_F/|k|^2$")

    if needle_mu is not None:
        Sn = sstar_knee_needle(needle_mu, tol=tol)
        wide = np.full_like(W, np.sqrt(Sn[0, 0]))
        thin = np.full_like(W, np.sqrt(Sn[1, 1]))
        ang = np.zeros_like(W)
        note = (f"needle: $\\sigma_x={np.sqrt(Sn[0,0]):.2f}$, "
                f"$\\sigma_y\\to 0$ (imposed)")
    else:
        st = sstar_closed_form(d["kx"], d["ky"], d["H11"], d["H12"],
                               d["H22"], tol=tol, cap=25.0)
        wide, thin = st["widths"][..., 0], st["widths"][..., 1]
        ang = np.degrees(st["angle_wide"]) % 180
        note = None

    for ax, F, ttl, vmax, cm in (
            (axes[1], wide, r"$\sigma_1^*$ (wide)", 8, "viridis"),
            (axes[2], thin, r"$\sigma_2^*$ (thin)", 2, "viridis"),
            (axes[3], ang, "wide-axis angle (deg)", 180, "twilight")):
        im = ax.imshow(F, extent=ext, origin="lower", cmap=cm,
                       vmin=0, vmax=vmax, aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_title(ttl + ("" if note is None or ax is not axes[1]
                            else f"  [{note}]"))


def main(outdir, tol1, tol2, mu):
    os.makedirs(outdir, exist_ok=True)
    print(f"H S H generalization verified symbolically: {verify_HSH()}",
          flush=True)

    knee = build_knee(mu)
    pn = build_pn()
    tol = tol2   # single tolerance now: the corrugation/scale rule

    fig, axes = plt.subplots(2, 4, figsize=(19.5, 8.4))
    panel_row_sstar(fig, axes[0], knee, tol, rf"knee $\mu={mu}$",
                    needle_mu=mu)
    panel_row_sstar(fig, axes[1], pn, tol, "PN defect")
    fig.tight_layout()
    path = os.path.join(outdir, "sstar_field_maps.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"figure: {path}")


class _Args:
    outdir = "out_sigma_window"
    tol1 = 0.05
    tol2 = 0.05
    mu = 0.9


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=_Args.outdir)
    p.add_argument("--tol1", type=float, default=_Args.tol1)
    p.add_argument("--tol2", type=float, default=_Args.tol2)
    p.add_argument("--mu", type=float, default=_Args.mu)
    a = p.parse_args()
    main(a.outdir, a.tol1, a.tol2, a.mu)
