# pipelines/uhu_ansatz_symbolic.py
"""
Exact uHu tensor structure for (A) a general single-phase ansatz,
(B) the knee-bend ansatz, and (C) the PN (single/multi) dislocation
ansatz, before and after averaging.

Organizing identity (Part A, sympy-verified):

    u = A cos(theta)
    u d_i d_j u = T^DC_ij + [.] cos(2 theta) + [.] sin(2 theta)
    T^DC_ij = (1/2) (A d_i d_j A - A^2 theta_i theta_j)

Ideal phase averaging kills the 2-theta pieces exactly (filter-shape
agnostic -- same orthogonality as the moment ladder). So for phase-only
fields (A = 1) the ideal-averaged tensor is EXACTLY -1/2 grad(theta)
grad(theta)^T: rank one, dominant eigenvector = true k, everywhere theta
is smooth. Every extraction failure is therefore a finite-kernel effect,
and there are exactly two channels:

  (1) corrugation residual: the 2-theta piece survives with factor
      ~ exp(-2 sigma^2 |grad theta|^2)  (exact for the knee's x-average),
      demanding sigma >~ 1/|k|;
  (2) DC mixing: a width-sigma kernel averages theta_i theta_j over a
      region where grad(theta) itself varies. Second-order (Gaussian)
      kernel expansion:
          <f>_sigma = f + (sigma^2/2) Lap f + O(sigma^4),
      so the relative DC error is ~ sigma^2 ||grad grad theta||^2 /
      |grad theta|^2, demanding sigma <~ |grad theta|/||grad grad theta||.

  A valid sigma window exists iff

      W := ||grad grad theta|| / |grad theta|^2  << 1,

  i.e. iff the phase contours turn slowly on the scale of one wavelength.
  W is computable pointwise from any ansatz, with no reference to a
  kernel: where W ~ 1, NO kernel size is simultaneously large enough to
  kill corrugation and small enough not to corrupt the DC tensor --
  "averaging is not defined there."

Knee core:  W = k2^2/k1^2 = mu^2/(1-mu^2)   (W = 1 at mu = 1/sqrt(2)).
PN core:    |grad theta| ~ 1/r and ||grad grad theta|| ~ 1/r^2, so
            W -> O(1) constant as r -> 0: the window never opens near
            the core, at ANY sigma -- a different failure mechanism from
            the knee's finite-mu* threshold. Part C checks this claim
            numerically rather than trusting the scaling argument.

Parts:
  A (symbolic): derive/verify the DC + 2theta split and the second-order
      kernel-expansion error term for general A, theta.
  B (knee): specialize symbolically (reproduces handoff Claim 1), then
      numerics: mu*(sigma) crossing curve from the DC competition
      k1^2 vs <theta_y^2>_sigma, exact corrugation-survival factor
      exp(-2 sigma^2 (1-mu^2)), eigen-angle error transects.
  C (PN): symbolic derivatives of theta = y - sign(y) log(psi) on y > 0
      (heat-equation check psi_y = psi_xx / 2), symmetry-extended to the
      full plane; numeric Gaussian-averaged tensor; chi = lam2/lam1 and
      eigen-angle error vs r for several sigma, tested for collapse in
      r/sigma; W maps for knee vs PN side by side.

Interpretation stays in the paper prose; this script only computes.
"""

import argparse
import os

import numpy as np
import sympy as sp
from numpy.fft import fft2, ifft2, fftfreq
from scipy.optimize import brentq
from scipy.special import erf as nerf
from sympy.simplify.fu import TR8

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
# Part A: general single-phase ansatz, exact DC/2theta split (symbolic)
# =====================================================================

def part_A(report):
    x, y = sp.symbols("x y", real=True)
    sig = sp.symbols("sigma", positive=True)
    A = sp.Function("A")(x, y)
    th = sp.Function("theta")(x, y)
    u = A * sp.cos(th)

    coords = {"x": x, "y": y}

    def split_component(i, j):
        """u d_i d_j u -> (DC, coeff of cos 2th, coeff of sin 2th), exact."""
        expr = sp.expand(TR8(sp.expand(u * sp.diff(u, coords[i], coords[j]))))
        c2, s2 = sp.cos(2 * th), sp.sin(2 * th)
        dc = expr.subs({c2: 0, s2: 0})
        cc = sp.simplify(expr.coeff(c2))
        cs = sp.simplify(expr.coeff(s2))
        # exactness check: nothing left over
        resid = sp.simplify(expr - dc - cc * c2 - cs * s2)
        assert resid == 0, f"split not exact for ({i}{j}): {resid}"
        return sp.simplify(dc), cc, cs

    report("\n=== Part A: general u = A cos(theta) ===")
    dcs = {}
    for (i, j) in [("x", "x"), ("x", "y"), ("y", "y")]:
        dc, cc, cs = split_component(i, j)
        dcs[i + j] = dc
        report(f"\nT_{i}{j} = DC + [C]cos2th + [S]sin2th, with")
        report(f"  DC = {dc}")
        report(f"  C  = {cc}")
        report(f"  S  = {cs}")
        # verify DC == (1/2)(A A_ij - A^2 th_i th_j)
        target = sp.Rational(1, 2) * (
            A * sp.diff(A, coords[i], coords[j])
            - A**2 * sp.diff(th, coords[i]) * sp.diff(th, coords[j])
        )
        assert sp.simplify(dc - target) == 0
    report("\nVerified exactly: DC_ij = (1/2)(A d_i d_j A - A^2 th_i th_j).")
    report("Phase-only (A=1): ideal-averaged tensor = -(1/2) grad(th) grad(th)^T,")
    report("rank one with eigenvector = k, for ANY smooth theta.")

    # ---- second-order Gaussian kernel expansion of the DC tensor -------
    # <f>_sigma = f + (sigma^2/2) Lap f + O(sigma^4)   (isotropic Gaussian)
    report("\n--- Second-order kernel expansion, phase-only DC tensor ---")
    thx, thy = sp.diff(th, x), sp.diff(th, y)
    err = {}
    for (nm, Q) in [("xx", thx * thx), ("xy", thx * thy), ("yy", thy * thy)]:
        lapQ = sp.diff(Q, x, 2) + sp.diff(Q, y, 2)
        err[nm] = sp.simplify(sp.Rational(1, 2) * lapQ)
        report(f"  <Q_{nm}>_sigma - Q_{nm} = sigma^2 * [{err[nm]}] + O(sigma^4)")
    report(
        "\nEach bracket is built from second derivatives of theta (products of\n"
        "grad grad theta, plus grad theta . grad of Hessian entries). Relative\n"
        "to the DC scale |grad th|^2, the leading corruption is\n"
        "    ~ sigma^2 ||grad grad th||^2 / |grad th|^2  (+ third-deriv terms),\n"
        "giving the upper edge of the window sigma <~ |grad th|/||grad grad th||.\n"
        "Corrugation kill needs sigma >~ 1/|grad th| (lower edge). Window exists\n"
        "iff W := ||grad grad th|| / |grad th|^2 << 1."
    )
    return dcs


# =====================================================================
# Part B: knee ansatz
# =====================================================================

def part_B(outdir, report, sigmas_show=(np.pi / 8, np.pi / 4, np.pi / 2, np.pi)):
    x, y, s = sp.symbols("x y s", real=True)
    k1, k2, sig = sp.symbols("k_1 k_2 sigma", positive=True)

    th = k1 * x + sp.log(2 * sp.cosh(k2 * y))
    u = sp.cos(th)

    report("\n=== Part B: knee theta = k1 x + log(2 cosh(k2 y)) ===")

    # -- symbolic checks of the handoff Claim-1 structure ---------------
    Txx = sp.simplify(u * sp.diff(u, x, 2))
    assert sp.simplify(Txx + k1**2 * u**2) == 0
    report("T_xx = -k1^2 u^2 exactly (no averaging needed).")

    Txy = sp.simplify(u * sp.diff(u, x, y))
    # odd in y => zero on the core for any y-symmetric kernel
    assert sp.simplify(Txy + Txy.subs(y, -y)) == 0
    report("T_xy odd in y: vanishes on the core row under any symmetric kernel.")

    Tyy = sp.expand(TR8(sp.expand(u * sp.diff(u, y, 2))))
    # DC = drop every trig atom carrying the fast x-phase (TR8 may rewrite
    # the argument of cos/sin(2 theta), so match on x-dependence, not form)
    fast = {f: 0 for f in Tyy.atoms(sp.cos, sp.sin) if f.args[0].has(x)}
    dc_yy = sp.simplify(Tyy.subs(fast))
    assert sp.simplify(dc_yy + sp.Rational(1, 2) * k2**2 * sp.tanh(k2 * y) ** 2) == 0
    report("T_yy DC = -(1/2) k2^2 tanh^2(k2 y).")

    # -- exact corrugation survival under the x-average -----------------
    # theta = k1 x + g(y): <cos(2 k1 x + phi)>_{sigma, x} has closed form.
    phi = sp.symbols("phi", real=True)
    gauss = sp.exp(-(s**2) / (2 * sig**2)) / (sig * sp.sqrt(2 * sp.pi))
    cf = sp.integrate(sp.exp(sp.I * 2 * k1 * s) * gauss, (s, -sp.oo, sp.oo))
    surv = sp.simplify(sp.re(sp.expand(cf * sp.exp(sp.I * (2 * k1 * x + phi)))))
    assert sp.simplify(surv - sp.exp(-2 * sig**2 * k1**2) * sp.cos(2 * k1 * x + phi)) == 0
    report(f"\nExact x-average of the 2theta piece: {surv}")
    report(
        "=> survival factor exp(-2 sigma^2 k1^2) = exp(-2 sigma^2 (1 - mu^2)):\n"
        "corrugation suppression WEAKENS as mu -> 1 (k1 -> 0), exactly the\n"
        "regime where the DC competition also worsens. Both channels point at\n"
        "sharp knees."
    )

    # -- numeric: DC eigen-flip competition k1^2 vs <theta_y^2>_sigma ---
    # tanh^2 depends on y only, so the 2D isotropic average reduces to 1D.
    gh_s, gh_w = np.polynomial.hermite_e.hermegauss(120)  # weight e^{-s^2/2}

    def avg_thy2(y0, mu, sigma):
        """< k2^2 tanh^2(k2 (y0 + s)) >, s ~ N(0, sigma^2), k2 = mu."""
        yy = y0 + sigma * gh_s
        vals = (mu * np.tanh(mu * yy)) ** 2
        return float(np.sum(gh_w * vals) / np.sum(gh_w))

    def mu_star(sigma):
        f = lambda mu: (1.0 - mu * mu) - avg_thy2(0.0, mu, sigma)
        return brentq(f, 1e-6, 1 - 1e-9)

    sig_grid = np.linspace(0.05, 1.25 * np.pi, 160)
    mu_curve = np.array([mu_star(sg) for sg in sig_grid])

    spot = {np.pi / 8: 0.950, np.pi / 4: 0.890, np.pi / 2: 0.810, np.pi: 0.770}
    report("\nmu*(sigma), DC-only competition vs full-convolution spot checks:")
    for sg, ref in spot.items():
        report(f"  sigma = {sg:.4f}: mu*_DC = {mu_star(sg):.3f}   (full-conv ref {ref:.3f})")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.plot(sig_grid, mu_curve, lw=2)
    ax.scatter(list(spot.keys()), list(spot.values()), color="crimson", zorder=5,
               label="full-convolution spot checks")
    ax.axhline(1 / np.sqrt(2), color="gray", ls=":", lw=1)
    ax.text(0.06, 1 / np.sqrt(2) + 0.006, r"$W=1$: $\mu = 1/\sqrt{2}$",
            fontsize=9, color="gray")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$\mu^*$")
    ax.set_title(r"Eigen-flip threshold: $k_1^2 = \langle\theta_y^2\rangle_\sigma$ at $y=0$")
    ax.legend(fontsize=8)

    ax = axes[1]
    mus = np.linspace(0.5, 0.999, 200)
    for sg in sigmas_show:
        comp = np.array([avg_thy2(0.0, m, sg) for m in mus])
        ax.plot(mus, comp, label=rf"$\langle\theta_y^2\rangle_\sigma$, $\sigma={sg:.2f}$")
    ax.plot(mus, 1 - mus**2, "k--", lw=2, label=r"$k_1^2 = 1-\mu^2$")
    ax.set_xlabel(r"$\mu$")
    ax.set_title("On-core DC competition")
    ax.legend(fontsize=8)

    ax = axes[2]
    MU, SG = np.meshgrid(np.linspace(0.5, 0.999, 200), np.linspace(0.05, np.pi, 200))
    surv_num = np.exp(-2 * SG**2 * (1 - MU**2))
    pc = ax.pcolormesh(MU, SG, surv_num, shading="auto", cmap="magma")
    fig.colorbar(pc, ax=ax, label=r"$e^{-2\sigma^2(1-\mu^2)}$")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\sigma$")
    ax.set_title("Exact corrugation survival (x-average)")
    fig.tight_layout()
    f1 = os.path.join(outdir, "knee_flip_and_corrugation.png")
    fig.savefig(f1, dpi=160)
    plt.close(fig)

    # -- eigen-angle error transects (DC model) -------------------------
    # DC tensor along a y-transect: diag part from averaged quantities.
    def dc_tensor_transect(mu, sigma, ys):
        k1v = np.sqrt(1 - mu * mu)
        Txx = np.full_like(ys, -0.5 * k1v**2)
        Txy = np.empty_like(ys)
        Tyy = np.empty_like(ys)
        for n, y0 in enumerate(ys):
            yy = y0 + sigma * gh_s
            w = gh_w / np.sum(gh_w)
            t = np.tanh(mu * yy)
            Txy[n] = -0.5 * k1v * np.sum(w * mu * t)
            Tyy[n] = -0.5 * np.sum(w * (mu * t) ** 2)
        return Txx, Txy, Tyy

    ys = np.linspace(0, 6, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, mu in zip(axes, (0.80, 0.90)):
        k1v = np.sqrt(1 - mu * mu)
        th_true = np.degrees(np.arctan2(mu * np.tanh(mu * ys), k1v))
        for sg in (np.pi / 8, np.pi / 2, np.pi):
            Txx, Txy, Tyy = dc_tensor_transect(mu, sg, ys)
            # dominant (most negative) eigenvector angle of the 2x2 tensor
            ang = 0.5 * np.arctan2(2 * Txy, Txx - Tyy)
            # eigen pair angles: ang gives principal axis of (T - tr/2 I);
            # pick the axis of the more negative eigenvalue.
            tr, dd = Txx + Tyy, np.hypot(Txx - Tyy, 2 * Txy)
            lam_a = 0.5 * (tr - dd)  # more negative
            # axis angle for lam_a:
            ang_a = np.degrees(0.5 * np.arctan2(2 * Txy, Txx - Tyy))
            # arctan2 principal axis corresponds to the larger of (Txx-Tyy)
            # projection; resolve by direct check at each point
            c, s_ = np.cos(np.radians(ang_a)), np.sin(np.radians(ang_a))
            q = Txx * c * c + 2 * Txy * c * s_ + Tyy * s_ * s_
            other = np.where(np.isclose(q, lam_a, rtol=1e-6), ang_a, ang_a + 90.0)
            err = np.abs(((other - th_true) + 90) % 180 - 90)
            ax.plot(ys, err, label=rf"$\sigma={sg:.2f}$")
        ax.set_title(rf"$\mu={mu}$: eigenvector angle error vs $y$")
        ax.set_xlabel(r"$y$")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("angle error (deg)")
    fig.tight_layout()
    f2 = os.path.join(outdir, "knee_eigenangle_error.png")
    fig.savefig(f2, dpi=160)
    plt.close(fig)

    return [f1, f2]


# =====================================================================
# Part C: PN ansatz
# =====================================================================

def pn_symbolic(report):
    """Exact symbolic derivatives on the y > 0 half plane."""
    x, y = sp.symbols("x y", positive=False, real=True)
    yp = sp.symbols("y", positive=True)  # y > 0 branch
    a, b = sp.symbols("a b", real=True)

    psi = a + b * sp.erf(x / sp.sqrt(2 * yp))
    # Hopf-Cole / heat-equation check: psi_y = (1/2) psi_xx
    heat = sp.simplify(sp.diff(psi, yp) - sp.Rational(1, 2) * sp.diff(psi, x, 2))
    assert heat == 0
    report("\n=== Part C: PN theta = y - sign(y) log(psi), y > 0 branch ===")
    report("Verified symbolically: psi_y = (1/2) psi_xx (heat equation, |y| = time).")

    th = yp - sp.log(psi)
    thx = sp.simplify(sp.diff(th, x))
    thy = sp.simplify(sp.diff(th, yp))
    report(f"theta_x = {thx}")
    report(f"theta_y = {thy}")

    grads = {
        "thx": thx,
        "thy": thy,
        "thxx": sp.diff(th, x, 2),
        "thxy": sp.diff(th, x, yp),
        "thyy": sp.diff(th, yp, 2),
    }
    lam = {
        k: sp.lambdify((x, yp, a, b), sp.simplify(v), modules="numpy")
        for k, v in grads.items()
    }
    return lam


def pn_fields(lam, X, Y, x_defects, eps=1e-6):
    """
    Full-plane first-derivative fields (thx, thy) from exact closed forms
    on the y > 0 branch, extended by the exact symmetry
    theta(x,-y) = -theta(x,y):  thx odd in y, thy even in y.
    Multi-defect: psi and its derivatives are sums of the same closed
    forms (heat-equation linearity). Second derivatives are taken
    spectrally downstream, off the eps-regularized core -- mirroring the
    role SH relaxation plays on the real data.
    """
    aval = 0.5 * (1 + np.exp(-np.pi * len(x_defects)))
    Ya = np.maximum(np.abs(Y), eps)
    sq = np.sqrt(2 * Ya)

    psi = np.full_like(X, aval)
    psi_x = np.zeros_like(X)
    psi_y = np.zeros_like(X)  # derivative w.r.t. |y|
    thetas = np.arange(len(x_defects) + 1) * np.pi
    pref = 2.0 / np.sqrt(np.pi)
    for j, xd in enumerate(x_defects):
        delta = 0.5 * (np.exp(-thetas[j + 1]) - np.exp(-thetas[j]))
        arg = (X - xd) / sq
        g = np.exp(-arg**2)
        psi += delta * nerf(arg)
        psi_x += delta * pref * g / sq
        psi_y += delta * pref * g * (-(X - xd)) / (2 * Ya * sq)

    sgn = np.sign(Y)
    sgn[sgn == 0] = 1.0
    thx_h = -psi_x / psi          # y > 0 branch
    thy_h = 1.0 - psi_y / psi

    out = {
        "thx": sgn * thx_h,  # odd in y
        "thy": thy_h,        # even in y
    }
    return out, psi


def gaussian_avg_fft(arr, sigma, Lx, Ly):
    ny, nx = arr.shape
    kx = 2 * np.pi * fftfreq(nx, d=Lx / nx)
    ky = 2 * np.pi * fftfreq(ny, d=Ly / ny)
    XI, ETA = np.meshgrid(kx, ky)
    ker = np.exp(-0.5 * sigma**2 * (XI**2 + ETA**2))
    return np.real(ifft2(ker * fft2(arr)))


def part_C(outdir, report, L=100.0, N=1024, sigmas=(np.pi / 8, np.pi / 4, np.pi / 2, np.pi),
           x_defects=(0.0,), tag=""):
    lam = pn_symbolic(report)

    xs = np.linspace(-L / 2, L / 2, N, endpoint=False)
    ys = np.linspace(-L / 2, L / 2, N, endpoint=False) + 0.5 * (L / N)  # stagger
    X, Y = np.meshgrid(xs, ys)
    fields, psi = pn_fields(lam, X, Y, x_defects=list(x_defects))

    thx, thy = fields["thx"], fields["thy"]
    Rall = np.stack([np.hypot(X - xd, Y) for xd in x_defects])
    R = Rall.min(axis=0)

    report("\nPN gradient magnitudes near core (single defect):")
    for r in (0.5, 1.0, 2.0, 4.0):
        m = (R > 0.9 * r) & (R < 1.1 * r)
        report(f"  r ~ {r}: |grad th| mean = {np.hypot(thx, thy)[m].mean():.3f}")

    # ---- W field: ||grad grad th|| / |grad th|^2 -----------------------
    # second derivatives spectrally from thx, thy (smooth off-core; the
    # eps-regularized core row keeps FFTs sane, mirroring the SH-relaxed
    # field's smoothing role)
    def ddx(f):
        k = 2 * np.pi * fftfreq(N, d=L / N)
        return np.real(ifft2(1j * k[None, :] * fft2(f)))

    def ddy(f):
        k = 2 * np.pi * fftfreq(N, d=L / N)
        return np.real(ifft2(1j * k[:, None] * fft2(f)))

    thxx, thxy = ddx(thx), ddy(thx)
    thyy = ddy(thy)
    Hnorm = np.sqrt(thxx**2 + 2 * thxy**2 + thyy**2)
    gradsq = thx**2 + thy**2
    W = Hnorm / np.maximum(gradsq, 1e-12)

    # knee W field for side-by-side (mu = 0.9)
    mu = 0.9
    k1v, k2v = np.sqrt(1 - mu**2), mu
    Yk = Y
    W_knee = (k2v**2 / np.cosh(k2v * Yk) ** 2) / (k1v**2 + (k2v * np.tanh(k2v * Yk)) ** 2)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ax = axes[0]
    ext = [-12, 12, -12, 12]
    m = (np.abs(X) <= 12) & (np.abs(Y) <= 12)
    pc = ax.imshow(np.clip(W, 0, 2)[np.ix_((np.abs(ys) <= 12), (np.abs(xs) <= 12))],
                   extent=ext, origin="lower", cmap="viridis", vmin=0, vmax=2)
    fig.colorbar(pc, ax=ax, label=r"$W$")
    ax.set_title(r"PN: $W=\|\nabla\nabla\theta\|/|\nabla\theta|^2$")

    ax = axes[1]
    rline = np.linspace(0.3, 12, 200)
    # W along +x axis and +y axis through the core
    ix0 = np.argmin(np.abs(ys))
    iy0 = np.argmin(np.abs(xs))
    ax.plot(xs[xs > 0.2], W[ix0, xs > 0.2], label="PN, along x (core line)")
    ax.plot(ys[ys > 0.2], W[ys > 0.2, iy0], label="PN, along y")
    Wk_line = (k2v**2 / np.cosh(k2v * rline) ** 2) / (
        k1v**2 + (k2v * np.tanh(k2v * rline)) ** 2)
    ax.plot(rline, Wk_line, "--", label=rf"knee $\mu={mu}$, transect $y$")
    ax.axhline(1, color="gray", ls=":", lw=1)
    ax.set_xlabel("distance from core")
    ax.set_ylabel(r"$W$")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2.5)
    ax.set_title("Window criterion vs distance")
    ax.legend(fontsize=8)

    ax = axes[2]
    musw = np.linspace(0.3, 0.999, 300)
    ax.plot(musw, musw**2 / (1 - musw**2), lw=2)
    ax.axhline(1, color="gray", ls=":")
    ax.axvline(1 / np.sqrt(2), color="gray", ls=":")
    ax.set_ylim(0, 6)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$W$ on knee core")
    ax.set_title(r"Knee core: $W=\mu^2/(1-\mu^2)$")
    fig.tight_layout()
    f1 = os.path.join(outdir, f"pn_vs_knee_window_criterion{tag}.png")
    fig.savefig(f1, dpi=160)
    plt.close(fig)

    # ---- averaged DC tensor: chi and eigen-angle error vs r/sigma ------
    Qxx, Qxy, Qyy = thx * thx, thx * thy, thy * thy
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    cols = plt.cm.plasma(np.linspace(0.15, 0.85, len(sigmas)))
    rbins = np.linspace(0.3, 14, 45)
    rmid = 0.5 * (rbins[1:] + rbins[:-1])

    for c, sg in zip(cols, sigmas):
        Axx = gaussian_avg_fft(Qxx, sg, L, L)
        Axy = gaussian_avg_fft(Qxy, sg, L, L)
        Ayy = gaussian_avg_fft(Qyy, sg, L, L)
        tr = Axx + Ayy
        dd = np.hypot(Axx - Ayy, 2 * Axy)
        lam1 = 0.5 * (tr + dd)   # dominant (Q is +semidef; -T convention)
        lam2 = 0.5 * (tr - dd)
        chi = lam2 / np.maximum(lam1, 1e-12)
        ang = 0.5 * np.arctan2(2 * Axy, Axx - Ayy)  # dominant axis of Q
        ang_true = np.arctan2(thy, thx)
        aerr = np.abs(((np.degrees(ang - ang_true)) + 90) % 180 - 90)

        chib, aeb = [], []
        for lo, hi in zip(rbins[:-1], rbins[1:]):
            m = (R >= lo) & (R < hi)
            chib.append(np.median(chi[m]))
            aeb.append(np.median(aerr[m]))
        axes[0].plot(rmid, chib, color=c, label=rf"$\sigma={sg:.2f}$")
        axes[1].plot(rmid, aeb, color=c)
        axes[2].plot(rmid / sg, chib, color=c)

    axes[0].set_xlabel("r"); axes[0].set_ylabel(r"median $\chi=\lambda_2/\lambda_1$")
    axes[0].set_title("Rank-two contamination vs distance")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("r"); axes[1].set_ylabel("median eigen-angle error (deg)")
    axes[1].set_title("Direction error vs distance")
    axes[2].set_xlabel(r"$r/\sigma$"); axes[2].set_ylabel(r"median $\chi$")
    axes[2].set_title(r"Collapse test: $\chi$ vs $r/\sigma$")
    fig.tight_layout()
    f2 = os.path.join(outdir, f"pn_chi_angle_vs_sigma{tag}.png")
    fig.savefig(f2, dpi=160)
    plt.close(fig)

    return [f1, f2]


# =====================================================================

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    lines = []

    def report(msg):
        print(msg)
        lines.append(str(msg))

    figs = []
    if "A" in args.parts:
        part_A(report)
    if "B" in args.parts:
        figs += part_B(args.outdir, report)
    if "C" in args.parts:
        figs += part_C(args.outdir, report, L=args.L, N=args.N)  # single defect
        if args.pn_ndef > 1:
            k1 = np.sqrt(1 - args.pn_mu**2)
            Lsp = np.pi / k1
            xd = Lsp * (np.arange(args.pn_ndef) - (args.pn_ndef - 1) / 2.0)
            report(f"\nPN chain: mu={args.pn_mu}, spacing L={Lsp:.3f}, "
                   f"{args.pn_ndef} defects; wall-merge height ~ L^2/2 = {Lsp**2/2:.1f}")
            figs += part_C(args.outdir, report, L=args.L, N=args.N,
                           x_defects=tuple(xd), tag=f"_chain_mu{args.pn_mu:.2f}")

    with open(os.path.join(args.outdir, "uhu_ansatz_symbolic_report.txt"), "w") as f:
        f.write("\n".join(lines))
    return figs


class _Args:
    parts = "ABC"
    outdir = "out_uhu_ansatz"
    L = 100.0
    N = 1024
    pn_mu = 0.9      # chain tilt: spacing L = pi/sqrt(1-mu^2)
    pn_ndef = 9      # 1 => skip the chain run


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parts", default=_Args.parts)
    p.add_argument("--outdir", default=_Args.outdir)
    p.add_argument("--L", type=float, default=_Args.L)
    p.add_argument("--N", type=int, default=_Args.N)
    p.add_argument("--pn_mu", type=float, default=_Args.pn_mu)
    p.add_argument("--pn_ndef", type=int, default=_Args.pn_ndef)
    main(p.parse_args())
