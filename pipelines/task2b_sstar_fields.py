# pipelines/task2b_sstar_fields.py
"""
Task 2, second half. Evaluate the optimal kernel covariance S* as explicit
fields on the two ansatze, exhibit the gap-collapse locus, and validate
S* against isotropic averaging WITHIN the quadratic-phase model.

From task2_averaging.py (all derived and checked there):
  - O(eps)   error ~ e^{-2 k^T S k} x poly(S, H) sin(2 th0)   [oscillatory]
  - O(eps^2) error ~ -(1/2)(H khat)^T S (H that) / gap        [DC mixing]
  - gap at O(1): -(kap^2/2)(1 + e^{-2 k^T S k} cos 2 th0)
  - direction rule: (H khat)^T S (H that) = 0; isotropizing solution
    S = c H^{-2}  (H invertible)
  - scale rule: k^T S k = ln(1/tol)/2 pins the corrugation at tol

Closed form when H is invertible: S* = c H^{-2},
    c = ln(1/tol) / (2 |H^{-1} k|^2),
    widths along H's eigenaxes: sigma_a = sqrt(ln(1/tol)/2) / (|h_a| |H^{-1} k|),
    orientation = H's eigenframe.

KNEE: H = diag(0, h), h = k2^2 sech^2(k2 y), singular EVERYWHERE, so the
closed form does not apply; the singular limit does, and it is better:
theta is EXACTLY linear in x, so the x-extent is unbounded with zero model
error, and the direction rule reduces to s_yy -> 0 off core (on core it
holds for ANY S). S*_knee = needle along x. This part is exact, not
second-order: every third and higher derivative in x vanishes identically.

PN: H in p-coordinates (Task 0), everything rational in p1..p4.

Run: python task2b_sstar_fields.py [--outdir DIR]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.special import erf as nerf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.uhu_symbolic import KneeAnsatz, PNAnsatz, is_zero, pn_chain_defects

FAILED = []


def report(msg=""):
    print(msg, flush=True)


def check(name, ok):
    tag = "PASS" if ok else "FAIL"
    report(f"  [{tag}] {name}")
    if not ok:
        FAILED.append(name)


# =====================================================================
# 1. knee: the exact needle, symbolically
# =====================================================================

def part_knee():
    report("=" * 70)
    report("1. KNEE S*: the exact x-needle")
    report("=" * 70)
    a = KneeAnsatz()
    x, y = a.x, a.y

    # exact linearity in x: all higher x-derivatives vanish identically
    ok = all(is_zero(sp.diff(a.theta, x, n)) for n in (2, 3, 4))
    check("theta is EXACTLY linear in x (d^n theta/dx^n = 0, n >= 2)", ok)

    # direction rule on the knee: (H khat)^T S (H that) = h^2 kh2 th2 s_yy
    sxx, syy = sp.symbols("s_xx s_yy", nonnegative=True)
    sxy = sp.symbols("s_xy", real=True)
    S = sp.Matrix([[sxx, sxy], [sxy, syy]])
    kvec = sp.Matrix([a.thx, a.thy])
    kn = sp.sqrt(a.thx**2 + a.thy**2)
    khat = kvec / kn
    that = sp.Matrix([-a.thy, a.thx]) / kn
    mix = sp.simplify((khat.T * a.hess * S * a.hess * that)[0, 0])
    h = a.hess[1, 1]
    expect = sp.simplify(h**2 * khat[1] * that[1] * syy)
    check("(H khat)^T S (H that) = h^2 khat_y that_y s_yy "
          "(only s_yy enters; s_xx, s_xy free)",
          is_zero(mix - expect))
    check("on core (y = 0) the DC mixing vanishes for ANY S",
          is_zero(mix.subs(y, 0)))

    report("    => S*_knee: s_yy -> 0, s_xy = 0, and s_xx set by the")
    report("       corrugation rule k^T S k = ln(1/tol)/2, i.e.")
    report("       s_xx = ln(1/tol) / (2 k1^2).   Exact at every (x, y),")
    report("       every mu; the x-extent carries NO model error because")
    report("       linearity in x is exact. The bound on s_yy is imposed")
    report("       (grid/noise), not derived: the model leaves it free,")
    report("       which is the degeneracy theorem showing up in the field.")
    tol = 0.05
    for mu in (0.5, 0.9):
        k1v = np.sqrt(1 - mu**2)
        report(f"       mu = {mu}: s_xx = {np.log(1/tol)/(2*k1v**2):.3f} "
               f"(sigma_x = {np.sqrt(np.log(1/tol)/2)/k1v:.3f}), "
               f"vs isotropic pi/2 = {np.pi/2:.3f}")
    report("")


# =====================================================================
# 2. PN: S* in p-coordinates, symbolic + numeric field
# =====================================================================

def pn_p_fields(X, Y, nd, k1):
    """p1..p4 for the nd-defect chain by the closed-form Gaussian
    derivatives of the erf steps. Naive assembly, VALID ONLY for nd <= ~11
    (Task 0 L2b regime); floors at e^{-nd pi}."""
    xd, L = pn_chain_defects(nd, k1)
    aw = 0.5 * (1.0 + np.exp(-np.pi * nd))
    dl = 0.5 * (np.exp(-np.pi * np.arange(1, nd + 1))
                - np.exp(-np.pi * np.arange(nd)))
    w = np.sqrt(2.0 * Y)
    c0 = 2.0 / np.sqrt(np.pi)
    psi = np.full_like(X, aw)
    d = [np.zeros_like(X) for _ in range(4)]
    for j, cj in enumerate(xd):
        z = (X - cj) / w
        e = np.exp(-z**2)
        psi += dl[j] * nerf(z)
        d[0] += dl[j] * c0 * e / w
        d[1] += dl[j] * c0 * e * (-2 * z) / w**2
        d[2] += dl[j] * c0 * e * (4 * z**2 - 2) / w**3
        d[3] += dl[j] * c0 * e * (-8 * z**3 + 12 * z) / w**4
    return [di / psi for di in d], L, xd


def sstar_from_kH(kx, ky, H11, H12, H22, tol=0.05, cap=None):
    """S* = c H^{-2} with c = ln(1/tol)/(2 |H^{-1}k|^2). Returns widths
    (sqrt eigenvalues of S*) and orientation of the wide axis. Where H is
    near singular the widths blow up; cap them if asked."""
    detH = H11 * H22 - H12**2
    # H^{-1} k
    a1 = (H22 * kx - H12 * ky) / detH
    a2 = (-H12 * kx + H11 * ky) / detH
    c = np.log(1.0 / tol) / (2.0 * (a1**2 + a2**2))
    tr = H11 + H22
    disc = np.sqrt(np.maximum((H11 - H22)**2 + 4 * H12**2, 0.0))
    habs = np.sort(np.abs(np.stack([(tr + disc) / 2, (tr - disc) / 2])),
                   axis=0)                       # |h_small|, |h_big|
    sig_wide = np.sqrt(c) / np.maximum(habs[0], 1e-300)
    sig_thin = np.sqrt(c) / np.maximum(habs[1], 1e-300)
    ang_big = 0.5 * np.arctan2(2 * H12, H11 - H22)  # axis of h with max
    # wide axis of S* is the SMALL-|h| axis of H
    if cap is not None:
        sig_wide = np.minimum(sig_wide, cap)
        sig_thin = np.minimum(sig_thin, cap)
    return sig_wide, sig_thin, ang_big, c


def part_pn(outdir):
    report("=" * 70)
    report("2. PN S* IN p-COORDINATES")
    report("=" * 70)
    pn = PNAnsatz(numeric_weights=False)
    x, y = pn.x, pn.y
    p1 = sp.diff(pn.psi, x) / pn.psi
    p2 = sp.diff(pn.psi, x, 2) / pn.psi
    p3 = sp.diff(pn.psi, x, 3) / pn.psi
    p4 = sp.diff(pn.psi, x, 4) / pn.psi

    Hp = sp.Matrix([[-(p2 - p1**2), -(p3 - p1 * p2) / 2],
                    [-(p3 - p1 * p2) / 2, -(p4 - p2**2) / 4]])
    Hdirect = sp.Matrix([[sp.diff(pn.theta, x, 2),
                          sp.diff(pn.theta, x, y)],
                         [sp.diff(pn.theta, x, y),
                          sp.diff(pn.theta, y, 2)]])
    check("H in p-coordinates matches direct differentiation",
          all(is_zero(sp.simplify(Hp[i, j] - Hdirect[i, j]))
              for i in range(2) for j in range(2)))

    detH = sp.simplify(Hp.det())
    report("    det H, rational in p1..p4 (erf/log free, as promised):")
    report(f"      {sp.factor(detH)}")
    report("    S* = c H^{-2}, c = ln(1/tol)/(2 |H^{-1} k|^2), with")
    report("    k = (-p1, 1 - p2/2). Every entry rational in p1..p4.")
    report("")

    # numeric field near the core, nd = 6 chain (Task 0 validated regime)
    mu = 0.9
    k1v = float(np.sqrt(1 - mu**2))
    nx, ny = 261, 121
    xd_probe, L = pn_chain_defects(6, k1v)
    xs = np.linspace(-1.6 * L, 1.6 * L, nx)
    ys = np.linspace(0.15, 0.75 * L, ny)
    X, Y = np.meshgrid(xs, ys)
    (P1, P2, P3, P4), L, xd = pn_p_fields(X, Y, 6, k1v)
    kx, ky = -P1, 1 - P2 / 2
    H11 = -(P2 - P1**2)
    H12 = -(P3 - P1 * P2) / 2
    H22 = -(P4 - P2**2) / 4
    sw, st, ang, c = sstar_from_kH(kx, ky, H11, H12, H22, cap=4 * L)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    ext = [xs[0] / L, xs[-1] / L, ys[0], ys[-1]]
    for ax, F, ttl, vmax in ((axes[0], sw, r"wide width $\sigma_1^*$", 4 * L),
                             (axes[1], st, r"thin width $\sigma_2^*$", L),
                             (axes[2], np.degrees(ang) % 180,
                              "large-|h| axis angle (deg)", 180)):
        im = ax.imshow(F, extent=ext, origin="lower", aspect="auto",
                       cmap="viridis", vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax)
        for cj in xd:
            ax.axvline(cj / L, color="w", ls=":", lw=0.6)
        ax.set_title(ttl)
        ax.set_xlabel(r"$x / L$")
    axes[0].set_ylabel(r"$y$")
    fig.suptitle(rf"PN S* field, $\mu = {mu}$, cap $4L$; "
                 "dotted lines = defect columns")
    fig.tight_layout()
    p = os.path.join(outdir, "pn_sstar_field.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    report(f"    figure: {p}")

    iy = np.argmin(np.abs(ys - 0.3 * L))
    ip = np.argmin(np.abs(xs - 0.5 * (xd[2] + xd[3])))   # plateau center
    ic = np.argmin(np.abs(xs - xd[3]))                    # defect column
    report(f"    transect at y = {ys[iy]:.2f} (0.3 L):")
    report(f"      plateau center: widths ({sw[iy, ip]:.2f}, "
           f"{st[iy, ip]:.2f}), capped at {4 * L:.1f} = 4L: "
           f"{'yes' if sw[iy, ip] >= 4 * L - 1e-9 else 'no'}")
    report(f"      defect column : widths ({sw[iy, ic]:.2f}, "
           f"{st[iy, ic]:.2f})")
    report("    Reading: on plateaus H -> 0 (locally a perfect plane wave),")
    report("    S* is unbounded and the cap is doing the work: the bound is")
    report("    imposed, not derived, same degeneracy as the knee. Near")
    report("    defect columns S* collapses and rotates: small kernels")
    report("    where curvature concentrates.")
    report("")
    return dict(mu=mu, L=L, xd=xd)


# =====================================================================
# 3. gap collapse and within-model validation
# =====================================================================

def part_gap_and_validation(outdir):
    report("=" * 70)
    report("3. GAP COLLAPSE, AND S* vs ISOTROPIC WITHIN THE MODEL")
    report("=" * 70)
    report("  gap(O(1)) = -(kap^2/2)(1 + e^{-2 k^T S k} cos 2 th0).")
    report("  Perturbation theory dies where e^{-2 k^T S k} -> 1 at")
    report("  cos 2 th0 = -1. Under S* the survival is PINNED at tol, so")
    report("  the worst-case gap shrink is exactly tol (5%):")
    for mu, sig in ((0.9, np.pi / 2),):
        k1v = np.sqrt(1 - mu**2)
        surv_iso = np.exp(-2 * sig**2 * k1v**2)
        report(f"    knee core, mu = {mu}: isotropic sigma = pi/2 keeps "
               f"{surv_iso:.2f} of the")
        report(f"    corrugation (39% gap shrink at worst phase, and the")
        report(f"    surviving osc feeds the flip); S* keeps 0.05 by "
               f"construction.")

    # within-model validation: exact GH average of the model tensor at
    # PN-like (k, H) samples, angle error under S* vs isotropic pi/2.
    from task2_averaging import avg_model
    report("")
    report("  angle error of the dominant eigenvector, EXACT average of the")
    report("  quadratic-phase model (GH quadrature), worst case over th0:")
    report(f"  {'case':<26} {'iso pi/2':>10} {'S*':>10}")

    mu = 0.9
    k1v = float(np.sqrt(1 - mu**2))
    xs_p, L = pn_chain_defects(6, k1v)
    samples = []
    for dx, yv, tag in ((0.5 * L, 0.30 * L, "PN plateau, y=0.3L"),
                        (0.25 * L, 0.30 * L, "PN quarter,  y=0.3L"),
                        (0.10 * L, 0.20 * L, "PN near col, y=0.2L"),
                        (0.5 * L, 0.60 * L, "PN plateau, y=0.6L")):
        X = np.array([[xs_p[2] + dx]])
        Y = np.array([[yv]])
        (P1, P2, P3, P4), _, _ = pn_p_fields(X, Y, 6, k1v)
        k = np.array([-P1[0, 0], 1 - P2[0, 0] / 2])
        H = np.array([[-(P2 - P1**2)[0, 0], (-(P3 - P1 * P2) / 2)[0, 0]],
                      [(-(P3 - P1 * P2) / 2)[0, 0],
                       (-(P4 - P2**2) / 4)[0, 0]]])
        samples.append((tag, k, H))

    def angle_err(k, H, S, th0):
        kap = np.hypot(*k)
        khat = k / kap
        R = np.array([[khat[0], khat[1]], [-khat[1], khat[0]]])
        Sr, Hr = R @ S @ R.T, R @ H @ R.T

        def integ(x1v, x2v):
            th = th0 + kap * x1v + 0.5 * (Hr[0, 0] * x1v**2
                                          + 2 * Hr[0, 1] * x1v * x2v
                                          + Hr[1, 1] * x2v**2)
            g1 = kap + Hr[0, 0] * x1v + Hr[0, 1] * x2v
            g2 = Hr[0, 1] * x1v + Hr[1, 1] * x2v
            out = np.empty((3,) + np.shape(x1v))
            out[0] = -0.5 * g1 * g1 * (1 + np.cos(2 * th)) \
                - 0.5 * np.sin(2 * th) * Hr[0, 0]
            out[1] = -0.5 * g1 * g2 * (1 + np.cos(2 * th)) \
                - 0.5 * np.sin(2 * th) * Hr[0, 1]
            out[2] = -0.5 * g2 * g2 * (1 + np.cos(2 * th)) \
                - 0.5 * np.sin(2 * th) * Hr[1, 1]
            return out

        L2 = np.linalg.cholesky(Sr)
        z1, z2, w = (lambda n: (lambda s, wt: (
            *np.meshgrid(s, s, indexing="ij"), np.outer(wt, wt)))(
            *np.polynomial.hermite_e.hermegauss(n)))(60)
        w = (w / w.sum()).ravel()
        xi = L2 @ np.vstack([z1.ravel(), z2.ravel()])
        vals = integ(xi[0], xi[1])
        Txx, Txy, Tyy = (np.sum(w * vals[i], axis=-1) for i in range(3))
        ang = 0.5 * np.arctan2(2 * Txy, Txx - Tyy)
        # dominant (most negative) eigenvector angle vs khat = 0 in frame
        lam1 = 0.5 * (Txx + Tyy) + 0.5 * np.hypot(Txx - Tyy, 2 * Txy)
        lam2 = 0.5 * (Txx + Tyy) - 0.5 * np.hypot(Txx - Tyy, 2 * Txy)
        a = ang if lam2 <= lam1 else ang + np.pi / 2   # angle of lam2 axis
        # angle of the most negative eigenvalue's eigenvector:
        a_dom = ang + (np.pi / 2 if abs(lam1) > abs(lam2) and lam1 < lam2
                       else 0.0)
        # simpler: eigenvector of min eigenvalue
        Tm = np.array([[Txx, Txy], [Txy, Tyy]])
        ev, V = np.linalg.eigh(Tm)
        v = V[:, 0]                                    # min eigenvalue
        return abs(np.degrees(np.arctan2(v[1], v[0])) % 180 - 0
                   ) if abs(np.degrees(np.arctan2(v[1], v[0])) % 180
                            ) < 90 else abs(
            np.degrees(np.arctan2(v[1], v[0])) % 180 - 180)

    tol = 0.05
    for tag, k, H in samples:
        e_iso = max(angle_err(k, H, (np.pi / 2)**2 * np.eye(2), t)
                    for t in np.linspace(0, np.pi, 9))
        Hi = np.linalg.inv(H)
        c = np.log(1 / tol) / (2 * float((Hi @ k) @ (Hi @ k)))
        Ss = c * Hi @ Hi
        # cap the model-validity radius at 4L equivalent: scale down if
        # the wide width exceeds it (keeps k^T S k >= target impossible
        # to violate only through the cap; report if capped)
        wmax = np.sqrt(np.max(np.linalg.eigvalsh(Ss)))
        capw = 4 * L
        if wmax > capw:
            Ss = Ss * (capw / wmax)**2
        e_st = max(angle_err(k, H, Ss, t)
                   for t in np.linspace(0, np.pi, 9))
        report(f"  {tag:<26} {e_iso:>9.3f}d {e_st:>9.3f}d"
               f"{'   (capped)' if wmax > capw else ''}")
    report("")
    report("  Model-level statement only; the exact-field version (true PN")
    report("  u, matrix-S Gauss-Hermite) is the Task 3 promotion of")
    report("  exact_convolution_check.")
    report("")


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    part_knee()
    part_pn(outdir)
    part_gap_and_validation(outdir)
    report("=" * 70)
    if FAILED:
        report(f"TASK 2b CHECKS FAILED: {len(FAILED)}")
        for f in FAILED:
            report(f"  - {f}")
        sys.exit(1)
    report("TASK 2b DONE: S* fields explicit for both ansatze.")


class _Args:
    outdir = "out_sstar"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=_Args.outdir)
    main(p.parse_args().outdir)
