# pipelines/sigma_window_maps.py
"""
Anisotropic generalization of the channel-2 error, and pointwise maps of the
sigma window it implies.

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

from utils.uhu_symbolic import KneeAnsatz, PNAnsatz, lambdify_np, eig2x2_numeric


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

def window_fields(kmag, H11, H12, H22, tol1=0.05, tol2=0.05):
    """
    Pointwise sigma window from the two channels.

    channel 1 (isotropic Gaussian): survival exp(-2 sigma^2 |k|^2) < tol1
        =>  sigma > sigma_min = sqrt(ln(1/tol1) / 2) / |k|
    channel 2 (isotropic): relative corruption (sigma |k| W)^2 < tol2
        =>  sigma < sigma_max = sqrt(tol2) / (|k| W)
    feasible isotropically iff sigma_min < sigma_max, i.e.
        W < sqrt(tol2 * 2 / ln(1/tol1))
    which is a quantitative version of "W << 1".

    Anisotropic version. Let (h1, h2) be the eigenvalues of H and psi the
    angle of khat in H's eigenframe. Channel 2 caps the width along each
    Hessian axis at s_a < sqrt(tol2) |k| / |h_a|, so the largest extent
    achievable along khat is
        ext_max = sqrt(tol2) |k| sqrt(cos^2 psi / h1^2 + sin^2 psi / h2^2),
    and channel 1 needs ext_max > sigma_min. The ratio of the two is the
    anisotropic feasibility margin, > 1 where an ellipse exists.
    """
    W = np.sqrt(H11**2 + 2 * H12**2 + H22**2) / np.maximum(kmag**2, 1e-30)
    s_min = np.sqrt(np.log(1 / tol1) / 2) / np.maximum(kmag, 1e-30)
    s_max = np.sqrt(tol2) / np.maximum(kmag * W, 1e-30)

    eg = eig2x2_numeric(H11, H12, H22, dominant="max")
    h1, h2 = eg["lam_plus"], eg["lam_minus"]
    return {"W": W, "sigma_min": s_min, "sigma_max": s_max,
            "iso_margin": s_max / s_min, "h1": h1, "h2": h2}


def aniso_margin(kx, ky, H11, H12, H22, tol1=0.05, tol2=0.05, floor=1e-12):
    kmag = np.hypot(kx, ky)
    eg = eig2x2_numeric(H11, H12, H22, dominant="max")
    h1, h2 = eg["lam_plus"], eg["lam_minus"]
    ang = eg["angle_plus"]                       # axis of h1
    # angle of khat measured in H's eigenframe
    psi = np.arctan2(ky, kx) - ang
    ext_max = np.sqrt(tol2) * kmag * np.sqrt(
        np.cos(psi)**2 / np.maximum(h1**2, floor)
        + np.sin(psi)**2 / np.maximum(h2**2, floor))
    s_min = np.sqrt(np.log(1 / tol1) / 2) / np.maximum(kmag, 1e-30)
    return ext_max / s_min


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


def panel_row(fig, axes, d, tol1, tol2, label):
    kmag = np.hypot(d["kx"], d["ky"])
    w = window_fields(kmag, d["H11"], d["H12"], d["H22"], tol1, tol2)
    am = aniso_margin(d["kx"], d["ky"], d["H11"], d["H12"], d["H22"],
                      tol1, tol2)
    ext = d["extent"]

    ax = axes[0]
    im = ax.imshow(np.clip(w["W"], 0, 2), extent=ext, origin="lower",
                   cmap="inferno", vmin=0, vmax=2, aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{label}: $W=\\|H\\|_F/|k|^2$")

    ax = axes[1]
    im = ax.imshow(np.clip(w["sigma_max"], 0, 6), extent=ext, origin="lower",
                   cmap="viridis", vmin=0, vmax=6, aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(rf"largest allowed $\sigma$ (tol$_2$={tol2})")

    ax = axes[2]
    im = ax.imshow(np.clip(w["sigma_min"], 0, 6), extent=ext, origin="lower",
                   cmap="viridis", vmin=0, vmax=6, aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(rf"smallest allowed $\sigma$ (tol$_1$={tol1})")

    ax = axes[3]
    m = np.log10(np.clip(w["iso_margin"], 1e-3, 1e3))
    im = ax.imshow(m, extent=ext, origin="lower", cmap="RdBu", vmin=-2, vmax=2,
                   aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.contour(d["X"], d["Y"], w["iso_margin"], levels=[1.0], colors="k",
               linewidths=1.2)
    ax.set_title(r"isotropic margin $\sigma_{max}/\sigma_{min}$ (log$_{10}$)")

    ax = axes[4]
    m = np.log10(np.clip(am, 1e-3, 1e3))
    im = ax.imshow(m, extent=ext, origin="lower", cmap="RdBu", vmin=-2, vmax=2,
                   aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.contour(d["X"], d["Y"], am, levels=[1.0], colors="k", linewidths=1.2)
    ax.set_title("anisotropic margin (log$_{10}$)")
    return w, am


def main(outdir, tol1, tol2, mu):
    os.makedirs(outdir, exist_ok=True)
    print(f"H S H generalization verified symbolically: {verify_HSH()}",
          flush=True)

    knee = build_knee(mu)
    pn = build_pn()

    fig, axes = plt.subplots(2, 5, figsize=(24, 8.4))
    wk, amk = panel_row(fig, axes[0], knee, tol1, tol2, rf"knee $\mu={mu}$")
    wp, amp = panel_row(fig, axes[1], pn, tol1, tol2, "PN defect")
    fig.tight_layout()
    path = os.path.join(outdir, "sigma_window_maps.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)

    def frac(a):
        return float(np.mean(np.asarray(a) > 1.0))

    print(f"\nknee mu={mu}: isotropically feasible area fraction "
          f"{frac(wk['iso_margin']):.3f}, anisotropically {frac(amk):.3f}")
    print(f"PN defect  : isotropically feasible area fraction "
          f"{frac(wp['iso_margin']):.3f}, anisotropically {frac(amp):.3f}")

    # on-core transect for the knee, the case Section 6 studied
    n = knee["W"].shape[0] if "W" in knee else wk["W"].shape[0]
    mid = n // 2
    print(f"\nknee on-core row (y = 0):")
    print(f"  W          = {wk['W'][mid, mid]:.3f}")
    print(f"  sigma_min  = {wk['sigma_min'][mid, mid]:.3f}")
    print(f"  sigma_max  = {wk['sigma_max'][mid, mid]:.3f}")
    print(f"  iso margin = {wk['iso_margin'][mid, mid]:.3f} "
          f"({'open' if wk['iso_margin'][mid, mid] > 1 else 'CLOSED'})")
    print(f"  ani margin = {amk[mid, mid]:.3e} "
          f"({'open' if amk[mid, mid] > 1 else 'CLOSED'})")
    print(f"\nfigure: {path}")


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
