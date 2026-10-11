# pipelines/rcn2d_control.py
"""
Two-dimensional RCN control with planted x-structure (tracker T1.2).

Purpose. The 1d control (wsindy_theta_t --mode control) is a knee with no
x-dependence, so J = det grad k = 0 and the two w-diffusion columns
kk:ddw and w_nn are one column. This control integrates the same RCN
phase equation on a doubly periodic box holding a two-knee zigzag whose
LOWER crease carries a planted x-modulation, runs the same estimator and
the same weak-form fits on both routes (exact k and theta_t from the PDE;
the estimator's k and theta_t from u = cos theta), and writes the field
as a data-format npz so the scripts of record (wsindy_theta_t data
mode, rcn_balance_ledger, rcn_free_stationary) run on it unchanged. The
questions it answers: can the estimator manufacture a J-like column, a
negative stationary J, or a stripe-parallel w diffusion from crease
ripple on a field where RCN is the truth; what {S, J} reads on RCN.

Equation (Writeup 1 constants alpha = eta = tau = 2R/3, f(w) = w):
    tau theta_t = 2 alpha div(w k) - eta Lap^2 theta,  k = grad theta,
    w = |k|^2 - 1,  theta = k1 x + Phi(x, y, t),  Phi doubly periodic.
Pseudo-spectral, 2/3 dealiasing of the cubic term, ETDRK4 in time with
the biharmonic exact. theta_t on every saved frame is the right side of
the PDE evaluated on the saved Phi (so the exact route is consistent
with the equation to spectral accuracy, whatever the time step).

Planted field. A periodic two-knee zigzag: a V crease at y_l = -Ly/4
(k2 = -mu below, +mu above: div k > 0, the SH lower boundary's
orientation) and an inverted crease at y_u = +Ly/4,
    Phi = G_{b(x)}(y - y_l) + P(x, y) - G_{b0}(y - y_u) - mu y,
    G_b(eta) = (1/b) [log cosh(b mu eta) + log 2]   (-> mu |eta| far away,
    with no b dependence far away, so a modulation of b is confined to
    the crease).
Two modulations of the lower crease, both periodic in x with mode m or a
train of bumps:
  sharpness  b(x) = beta0 (1 + eps_b s(x)): J = Phi_xx F'' - Phi_xy^2 is
             EVEN across the crease, confined to it; at a sharpness
             maximum (s'' < 0) J > 0 on the crease; with s(x) a train of
             positive bumps the planted J has one sign at every bump,
             the footprint of a row of cores (the SH row's weak J is even
             across the row, R18-19).
  position   P = -mu a_pos s(x) tanh(b0 mu eta) exp(-eta^2 / 2 sig_env^2),
             eta = y - y_l: the first-order displacement of the crease by
             a_pos s(x), confined by the envelope (a displaced crease
             between two perfect far fields cannot exist: the mismatch
             2 mu y_c would have to be carried as a far-field k1
             modulation mu y_c'(x) that never decays; the envelope makes
             it a local phase bump). J is ODD across the crease.
  phase bumps  Phi += a_bump exp(-(d^2 + eta^2) / 2 sig_bump^2) at n_bumps
             positions along the crease (d the periodic x distance): J of
             a bump c G is c^2 (G_xx G_yy - G_xy^2), positive at the
             centre and negative on a ring, EVEN across the crease with
             the same sign at every bump: the SH row's weak-J footprint
             (R18-19: + on the row between the cores, - on both flanks),
             at a level set by a_bump^2 / sig_bump^4.
  s(x) = sin(2 pi m x / Lx)  (--profile sin)  or  sum of Gaussian bumps of
  width sig_b at n_bumps positions (--profile bumps, unit peak).
  Measured (19 Sep): the sharpness route plants J at 4 percent of the SH
  level (the crease is too thin to carry Gaussian curvature); the
  position route at 130 percent at t 0 decaying to 40 percent by t 1;
  the phase bumps are the configuration of record for T1.2.
At zero modulation the field is the 1d control on a periodic box and
the fits must reproduce its numbers (validation).

Fits: wsindy_theta_t v1.6 machinery (kernel_sizes, query_centres,
build_columns, fit_blocks) on the domain --domain band (|y - y_l| <
band_to L) or all_lower (|y - y_l| < Ly/4, the 1d control's box), both
routes, with the extended library (the exact route carries A = 1, so its
amplitude columns are zero by construction). Before any fit the report
gives the planted J's footprint on the band centres (weak column with
the same bump, exact and estimator: rms, median, fraction negative, by
row distance) against the SH 0.85 tail's (R18-19), and the rms of the
S and Lap div k columns for scale.

Outputs: results/rcn2d_control/<stamp>/: report.txt, fits.json,
config.json, control field npz (u, x, y, tt, metadata_json, k1_exact,
k2_exact, theta_t_exact, A_exact), planted_field.png, control_movie.png,
library_terms_*.png, fit_*.png (from fit_blocks).

Changelog
---------
v1.1  (20 Sep) T2.10: --a_dip plants a static amplitude with core-like
      dips A = A_eq tanh(r / dip_rc) at n_bumps positions on the crease
      (the ledger's --pn_amp 2 construction on the knee), u = A cos theta;
      the exact route carries the planted A in its amplitude columns and
      core discs of core_excl L about the dips are removed as on data.
      The question: what do the amplitude columns A_dev div k and
      k.grad A read through the estimator when A has real dips and the
      phase is RCN (the control the SH amplitude channel lacks, B3/O9).
v1.0  (19 Sep) first version. Validation at zero modulation against the
      1d control of record recorded in RESULTS_addendum_0919 (R18-20).

Run:
  python rcn2d_control.py --eps_b 0 --a_pos 0 --domain all_lower      # validation
  python rcn2d_control.py --eps_b 0.5 --profile bumps --n_bumps 8       # sharpness bumps
  python rcn2d_control.py --a_pos 1.0 --profile sin --m 2               # position
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT / "src", _ROOT / "src" / "op_extract", _ROOT / "src" / "utils", _HERE):
    sys.path.insert(0, str(p))

import wsindy_theta_t as W                                              # noqa: E402
from mg_zigzag_theta_t import estimate, find_cores                      # noqa: E402
from op_extract import t1_common as t1c                                 # noqa: E402

import matplotlib                                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402

VERSION = "rcn2d_control v1.1"
# SH 0.85 tail, t 99.8, band centres, weak J with the 1.5 L bump (R18-19)
SH_REF = dict(rms=1.80e-3, frac_neg=0.93,
              by_row={"0-0.5": +1.98e-3, "0.5-1": -2.71e-3, "1-1.5": -1.37e-3, "1.5-2": -1.22e-4},
              rmsS=5.87e-3, rmsB=1.09e-2)


# ---------------------------------------------------------------------
# planted field
# ---------------------------------------------------------------------

def G_b(eta, b, mu):
    z = np.abs(b * mu * eta)
    return (z + np.log1p(np.exp(-2.0 * z))) / b


def profile_s(x, Lx, args):
    if args.profile == "sin":
        return np.sin(2 * np.pi * args.m * x / Lx)
    xb = (np.arange(args.n_bumps) + 0.5) * Lx / args.n_bumps + x[0]
    s = np.zeros_like(x)
    for xj in xb:
        d = (x - xj + Lx / 2) % Lx - Lx / 2
        s += np.exp(-d ** 2 / (2 * args.sig_b ** 2))
    return s


def planted_phi(X, Y, x, Lx, mu, y_l, y_u, args):
    s = profile_s(x, Lx, args)[None, :]
    b = args.beta0 * (1.0 + args.eps_b * s)
    eta = Y - y_l
    Phi = G_b(eta, b, mu) - G_b(Y - y_u, args.beta0, mu) - mu * Y
    if args.a_pos != 0.0:
        Phi = Phi - mu * args.a_pos * s * np.tanh(args.beta0 * mu * eta) * np.exp(-eta ** 2 / (2 * args.sig_env ** 2))
    if args.a_bump != 0.0:
        # Gaussian phase bumps centred on the crease at n_bumps positions:
        # J of a bump c G is c^2 (G_xx G_yy - G_xy^2): positive at the
        # centre, negative on a ring (even, the SH row's footprint)
        xb = (np.arange(args.n_bumps) + 0.5) * Lx / args.n_bumps + x[0]
        for xj in xb:
            d = (X - xj + Lx / 2) % Lx - Lx / 2
            Phi = Phi + args.a_bump * np.exp(-(d ** 2 + eta ** 2) / (2 * args.sig_bump ** 2))
    return Phi, b[0], s[0]


# ---------------------------------------------------------------------
# spectral RCN integrator (ETDRK4)
# ---------------------------------------------------------------------

class RCN2D:
    def __init__(self, nx, ny, Lx, Ly, k1, alpha, eta_b, tau, dt):
        self.k1, self.dt = k1, dt
        self.c_stretch = 2 * alpha / tau
        qx = 2 * np.pi * np.fft.fftfreq(nx, Lx / nx)
        qy = 2 * np.pi * np.fft.fftfreq(ny, Ly / ny)
        self.iqx, self.iqy = 1j * qx[None, :], 1j * qy[:, None]
        q2 = qx[None, :] ** 2 + qy[:, None] ** 2
        self.L = -(eta_b / tau) * q2 ** 2
        # 2/3 dealiasing mask
        self.mask = ((np.abs(qx) < (2.0 / 3.0) * np.abs(qx).max())[None, :]
                     & (np.abs(qy) < (2.0 / 3.0) * np.abs(qy).max())[:, None])
        # ETDRK4 coefficients (Kassam-Trefethen contour mean)
        L = self.L
        E = np.exp(dt * L); E2 = np.exp(dt * L / 2)
        M = 32
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
        LR = dt * L[..., None] + r[None, None, :]
        self.E, self.E2 = E, E2
        self.Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=-1))
        self.f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=-1))
        self.f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=-1))
        self.f3 = dt * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=-1))

    def grad(self, Ph):
        return (self.k1 + np.real(np.fft.ifft2(self.iqx * Ph)),
                np.real(np.fft.ifft2(self.iqy * Ph)))

    def N(self, Ph):
        """spectral N = (2 alpha / tau) div(w k), dealiased"""
        k1, k2 = self.grad(Ph)
        w = k1 ** 2 + k2 ** 2 - 1.0
        F1 = np.fft.fft2(w * k1) * self.mask
        F2 = np.fft.fft2(w * k2) * self.mask
        return self.c_stretch * (self.iqx * F1 + self.iqy * F2)

    def step(self, Ph):
        Nv = self.N(Ph)
        a = self.E2 * Ph + self.Q * Nv
        Na = self.N(a)
        b = self.E2 * Ph + self.Q * Na
        Nb = self.N(b)
        c = self.E2 * a + self.Q * (2 * Nb - Nv)
        Nc = self.N(c)
        return self.E * Ph + self.f1 * Nv + 2 * self.f2 * (Na + Nb) + self.f3 * Nc

    def rhs(self, Ph):
        """theta_t = L Phi + N(Phi) in physical space"""
        return np.real(np.fft.ifft2(self.L * Ph + self.N(Ph)))


def integrate(Phi0, nx, ny, Lx, Ly, k1, alpha, eta_b, tau, t_end, nt, dt):
    tt = np.linspace(0, t_end, nt)
    nsub = max(1, int(np.ceil((tt[1] - tt[0]) / dt)))
    dt_eff = (tt[1] - tt[0]) / nsub
    solver = RCN2D(nx, ny, Lx, Ly, k1, alpha, eta_b, tau, dt_eff)
    Ph = np.fft.fft2(Phi0)
    Phi = np.empty((ny, nx, nt)); K1 = np.empty_like(Phi); K2 = np.empty_like(Phi); TT = np.empty_like(Phi)
    for j in range(nt):
        if j > 0:
            for _ in range(nsub):
                Ph = solver.step(Ph)
        Phi[..., j] = np.real(np.fft.ifft2(Ph))
        K1[..., j], K2[..., j] = solver.grad(Ph)
        TT[..., j] = solver.rhs(Ph)
    return tt, Phi, K1, K2, TT, dt_eff


# ---------------------------------------------------------------------
# footprint of the planted J on the band centres (spatial bump only)
# ---------------------------------------------------------------------

def j_footprint(k1, k2, x, y, Lsp, y_l, args, lines, label):
    dx = float(x[1] - x[0])
    a = args.radius * Lsp
    fx, fx1, fx2, fx3, mx = W.kernel_d3(a, dx, args.poly_p)
    D = lambda F, ax: W.inner_deriv(F, 0.0, dx, dx, ax, "poly", args.poly_p)
    K = lambda F: W.corr1d(W.corr1d(F, fx, 0, dx), fx, 1, dx)
    w = k1 ** 2 + k2 ** 2 - 1
    k1x, k1y, k2x, k2y = D(k1, "x"), D(k1, "y"), D(k2, "x"), D(k2, "y")
    J = k1x * k2y - k1y * k2x
    divk = k1x + k2y
    Spt = w * divk + k1 * D(w, "x") + k2 * D(w, "y")
    Jw, Sw = K(J), K(Spt)
    Bw = -(W.corr1d(W.corr1d(k1, fx, 0, dx), fx3, 1, dx) + W.corr1d(W.corr1d(k1, fx2, 0, dx), fx1, 1, dx)
           + W.corr1d(W.corr1d(k2, fx1, 0, dx), fx2, 1, dx) + W.corr1d(W.corr1d(k2, fx3, 0, dx), fx, 1, dx))
    ny, nx = w.shape
    ix = np.arange(mx + 4, nx - mx - 4, args.stride_xy); iy = np.arange(mx + 4, ny - mx - 4, args.stride_xy)
    IY, IX = np.meshgrid(iy, ix, indexing="ij")
    dr = np.abs(y[IY] - y_l) / Lsp
    sel = dr < args.band_to
    jw, sw, bw = Jw[IY, IX][sel], Sw[IY, IX][sel], Bw[IY, IX][sel]
    lines.append("  %s: pointwise |J| max %.3e; weak J on band centres n %d: rms %.3e, median %+.3e, frac<0 %.2f;"
                 " rms S %.3e, rms Lap div k %.3e" %
                 (label, np.abs(J).max(), len(jw), np.sqrt(np.mean(jw ** 2)), np.median(jw), np.mean(jw < 0),
                  np.sqrt(np.mean(sw ** 2)), np.sqrt(np.mean(bw ** 2))))
    parts = []
    for lo, hi in ((0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2)):
        m_ = (dr[sel] >= lo) & (dr[sel] < hi)
        parts.append("%g-%g L %+.2e (frac<0 %.2f)" % (lo, hi, np.median(jw[m_]) if m_.any() else np.nan,
                                                       np.mean(jw[m_] < 0) if m_.any() else np.nan))
    lines.append("      weak J median by row distance: " + "; ".join(parts))
    return dict(rms=float(np.sqrt(np.mean(jw ** 2))), median=float(np.median(jw)), frac_neg=float(np.mean(jw < 0)),
                rmsS=float(np.sqrt(np.mean(sw ** 2))), rmsB=float(np.sqrt(np.mean(bw ** 2))), Jmax=float(np.abs(J).max()))


# ---------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------

def fig_planted(x, y, u, K1, K2, TT, tt, y_l, Lsp, path):
    dx = float(x[1] - x[0])
    D = lambda F, ax: W.inner_deriv(F, 0.0, dx, dx, ax, "poly", 6)
    sel = (np.abs(y - y_l) < 3 * Lsp)
    ext = [x[0], x[-1], y[sel][0], y[sel][-1]]
    fig, axs = plt.subplots(3, 3, figsize=(13, 9))
    for r, j in enumerate((0, len(tt) // 2, len(tt) - 1)):
        k1, k2 = K1[..., j], K2[..., j]
        w = k1 ** 2 + k2 ** 2 - 1
        J = D(k1, "x") * D(k2, "y") - D(k1, "y") * D(k2, "x")
        for c, (F, lab, cm, sym) in enumerate(((u[..., j], "u", "gray", False), (w, "w", "RdBu_r", True), (J, "J", "RdBu_r", True))):
            v = np.percentile(np.abs(F[sel]), 99.5) if sym else None
            im = axs[r, c].imshow(F[sel], origin="lower", extent=ext, cmap=cm, aspect="auto",
                                  vmin=-v if sym else None, vmax=v if sym else None)
            axs[r, c].set_title("%s, t = %.2f" % (lab, tt[j]), fontsize=9)
            if sym:
                fig.colorbar(im, ax=axs[r, c], shrink=0.8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ---------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------

def main(args):
    t_start = time.time()
    rng = np.random.default_rng(args.seed)
    mu = args.mu
    k1c = float(np.sqrt(1 - mu ** 2))
    Lsp = np.pi / k1c
    alpha, tau, eta_b = args.alpha, args.tau, args.eta_b
    Lx = args.n_xper * 2 * np.pi / k1c
    nx = int(round(Lx / args.dx_target))
    dx = Lx / nx
    ny = int(round(args.Ly / dx)); ny += ny % 2
    Ly = ny * dx
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dx
    X, Y = np.meshgrid(x, y)
    y_l, y_u = -Ly / 4, Ly / 4
    jc = int(np.argmin(np.abs(y - y_l)))
    mod = "b%.2g_%s%d_ep%.2g_ap%.2g_ab%.2g" % (args.beta0, args.profile, args.m if args.profile == "sin" else args.n_bumps,
                                               args.eps_b, args.a_pos, args.a_bump) + ("_dip%.2g" % args.dip_rc if args.a_dip else "")
    stamp = "%s_rad%.2g_th%d_%s_t%.2g" % (mod, args.radius, args.t_half, args.domain, args.t_end)
    out = Path(args.outdir) / stamp
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as fh:
        json.dump({k: getattr(args, k) for k in vars(args)}, fh, indent=2, default=str)
    lines = [VERSION, "RCN 2d, alpha %.3f tau %.3f eta_b %.3f (beta_inf %.2f), mu %.2f (L %.2f), box %.1f x %.1f (%dx%d, dx %.4f),"
             " knees at y_l %.1f (V, the lower boundary's orientation) and y_u %.1f (inverted)" %
             (alpha, tau, eta_b, np.sqrt(alpha / eta_b), mu, Lsp, Lx, Ly, nx, ny, dx, y_l, y_u),
             "planted: beta0 %.2f, profile %s (m %d, n_bumps %d, sig_b %.2f), sharpness eps_b %.3g, position a_pos %.3g"
             " (envelope %.2f), phase bumps a_bump %.3g (sig_bump %.2f)" % (args.beta0, args.profile, args.m, args.n_bumps, args.sig_b,
                                                                            args.eps_b, args.a_pos, args.sig_env, args.a_bump, args.sig_bump)]
    Phi0, b_x, s_x = planted_phi(X, Y, x, Lx, mu, y_l, y_u, args)
    lines.append("  b(x) range %.3f .. %.3f; s(x) range %.2f .. %.2f" % (b_x.min(), b_x.max(), s_x.min(), s_x.max()))
    tt, Phi, K1, K2, TT, dt_eff = integrate(Phi0, nx, ny, Lx, Ly, k1c, alpha, eta_b, tau, args.t_end, args.nt, args.dt)
    dt_fr = tt[1] - tt[0]
    lines.append("  integrated to t %.2f, %d frames (dt_frame %.4f), ETDRK4 step %.5f; far-field theta_t rms %.2e;"
                 " crease width ratio p_y(0)/mu^2 at the lower knee: %.3f -> %.3f" %
                 (args.t_end, args.nt, dt_fr, dt_eff, np.sqrt(np.mean(TT[np.abs(Y - y_l) > 4 * Lsp][..., 0] ** 2)) if True else 0,
                  float(np.gradient(K2[:, nx // 2, 0], dx)[jc] / mu ** 2), float(np.gradient(K2[:, nx // 2, -1], dx)[jc] / mu ** 2)))
    lines.append("  time kernel half width t_half x dt = %.3f time units" % (args.t_half * dt_fr))
    # T2.10: planted amplitude with core-like dips at n_bumps positions on the
    # crease (the ledger's --pn_amp 2 construction on the knee): A = A_eq
    # tanh(r / dip_rc), static in time; the phase is RCN regardless of A.
    if args.a_dip:
        xb = (np.arange(args.n_bumps) + 0.5) * Lx / args.n_bumps + x[0]
        dxp = (X[..., None] - xb[None, None, :] + Lx / 2) % Lx - Lx / 2
        rr = np.min(np.hypot(dxp, (Y - y_l)[..., None]), axis=-1)
        A_pl = np.sqrt(4.0 * args.R / 3.0) * np.tanh(rr / max(args.dip_rc, 1e-6))
        core_xy = (xb, np.full(len(xb), y_l))
        lines.append("  planted amplitude: %d dips at spacing %.2f on the crease, A = %.3f tanh(r / %.2f), static"
                     % (len(xb), Lx / args.n_bumps, np.sqrt(4.0 * args.R / 3.0), args.dip_rc))
    else:
        A_pl = np.ones_like(X); core_xy = None
    u = A_pl[..., None] * np.cos(k1c * X[..., None] + Phi)
    fig_planted(x, y, u, K1, K2, TT, tt, y_l, Lsp, out / "planted_field.png")
    # data-format npz for the scripts of record
    md = dict(mu=mu, alpha=float(np.arcsin(mu)), k1=k1c, k2=mu, Rscale=args.R, dx=dx, dy=dx, which_gb="lower",
              gb_row_in_crop=jc, control="rcn2d", beta0=args.beta0, profile=args.profile, m=args.m, n_bumps=args.n_bumps,
              eps_b=args.eps_b, a_pos=args.a_pos, sig_env=args.sig_env, a_bump=args.a_bump, sig_bump=args.sig_bump,
              alpha_rcn=alpha, tau=tau, eta_b=eta_b,
              t_end=args.t_end, nt=args.nt, y_l=y_l, y_u=y_u, Lx=Lx, Ly=Ly)
    npz_path = out / ("rcn2d_%s.npz" % mod)
    np.savez(npz_path, u=u.astype(np.float32), x=x, y=y, tt=tt, metadata_json=np.array(json.dumps(md)),
             k1_exact=K1.astype(np.float32), k2_exact=K2.astype(np.float32), theta_t_exact=TT.astype(np.float32),
             A_exact=np.broadcast_to(A_pl[..., None], u.shape).astype(np.float32))
    lines.append("  field -> %s" % npz_path.name)
    # estimator route
    taper = t1c.edge_taper(x, y, frac=args.taper_frac)
    ests, n_prev = [], None
    for i in range(args.nt):
        e = estimate(u[..., i], Lx, Ly, args.sigma_dir, n_prev, taper)
        n_prev = e["n"]; ests.append(e)
    k1m = np.stack([e["k1"] for e in ests], -1); k2m = np.stack([e["k2"] for e in ests], -1)
    zs = np.stack([e["z"] for e in ests], -1); Am = np.stack([e["A"] for e in ests], -1)
    win = (np.abs(X - x.mean()) < 0.4 * Lx) & (np.abs(Y - y_l) < 0.2 * Ly)
    ncores = [len(find_cores(e["A"], x, y, win, args.core_frac, merge_dist=0.5 * Lsp)[0]) for e in ests]
    del ests
    th_t_m = W.theta_t_from_z(zs, dt_fr)
    del zs
    s = np.sign(np.mean(k1m, axis=(0, 1)))[None, None, :]; s[s == 0] = 1
    k1m, k2m, th_t_m = k1m * s, k2m * s, th_t_m * s
    band3 = np.abs(Y - y_l) < 2 * Lsp
    rel = np.sqrt(np.mean((th_t_m - TT)[band3][:, 2:-2] ** 2)) / np.sqrt(np.mean(TT[band3][:, 2:-2] ** 2))
    lines.append("  estimator theta_t vs exact on the band (frames 2..nt-3): rel RMS %.4f; amplitude dips detected per frame %d..%d"
                 " (core_excl 0: none removed)" % (rel, min(ncores), max(ncores)))
    # footprint of the planted J before any fit
    lines.append("")
    lines.append("planted J footprint (weak column, bump radius %.2g L, band centres |y - y_l| < %g L), SH 0.85 tail reference"
                 " (R18-19): rms %.2e, frac<0 %.2f, by row distance 0-0.5 %+.2e, 0.5-1 %+.2e, 1-1.5 %+.2e, 1.5-2 %+.2e;"
                 " rms S %.2e, rms Lap div k %.2e" %
                 (args.radius, args.band_to, SH_REF["rms"], SH_REF["frac_neg"], SH_REF["by_row"]["0-0.5"],
                  SH_REF["by_row"]["0.5-1"], SH_REF["by_row"]["1-1.5"], SH_REF["by_row"]["1.5-2"], SH_REF["rmsS"], SH_REF["rmsB"]))
    fp = {}
    for j in (0, args.nt // 2):
        fp["exact_f%d" % j] = j_footprint(K1[..., j], K2[..., j], x, y, Lsp, y_l, args, lines, "exact, frame %d (t %.2f)" % (j, tt[j]))
        fp["estimator_f%d" % j] = j_footprint(k1m[..., j], k2m[..., j], x, y, Lsp, y_l, args, lines, "estimator, frame %d" % j)
    # domain and query centres
    if args.domain == "all_lower":
        dom = np.abs(Y - y_l) < Ly / 4 - args.rim * dx
    else:
        dom = np.abs(Y - y_l) < args.band_to * Lsp
    excl = ~dom
    if args.core_excl > 0 and core_xy is not None:
        dcore = np.min(np.hypot(X[..., None] - core_xy[0][None, None, :], Y[..., None] - core_xy[1][None, None, :]), axis=-1) / Lsp
        excl = excl | (dcore < args.core_excl)
        lines.append("  core discs of %g L removed about the %d planted dips" % (args.core_excl, len(core_xy[0])))
    excl = np.broadcast_to(excl[..., None], u.shape)
    m = W.kernel_sizes(args, dx, dx, dt_fr, Lsp)
    idx = W.query_centres(u.shape, m, args, excl=excl)
    lines.append("")
    lines.append("domain %s: %d query centres (stride %d, %d); rim %d px" % (args.domain, len(idx[0]), args.stride_xy, args.stride_t, args.rim))
    truth = dict(rhs={"w div k": 2 * alpha / tau, "k.grad w": 2 * alpha / tau, "Lap div k": -eta_b / tau},
                 energy={"w^2": alpha / (2 * tau), "(div k)^2": eta_b / (2 * tau)},
                 w={"kk:ddw": 4 * alpha / tau, "|grad w|^2": 2 * alpha / tau, "(k.grad w) div k": 4 * alpha / tau,
                    "w Lap w": 2 * alpha / tau, "w |grad k|^2": -4 * alpha / tau, "k.grad Lap div k": -2 * eta_b / tau})
    lines.append("truth: RHS %s ; ENERGY %s ; {S, J} on an exactly self-dual RCN field with coefficient b reads"
                 " (2 alpha/tau)(1 - b^2) S - (4 b eta/tau) J plus chi terms (identity R18-1)" % (truth["rhs"], truth["energy"]))
    routes = {"exact": (K1, K2, TT, np.broadcast_to(A_pl[..., None], u.shape).copy()), "estimator": (k1m, k2m, th_t_m, Am)}
    fits = {"footprint": fp, "rel_rms_theta_t": float(rel), "ncores": ncores}
    for route, (A1, A2, T_, AA) in routes.items():
        far = np.abs(Y - y_l) > 4 * Lsp
        Abar = float(np.nanmedian(AA[far]))
        cols, targets, fields = W.build_columns(A1, A2, T_, x, y, tt, args, Lsp, idx, A=AA, Abar=Abar,
                                                snap=args.nt // 2 if route == "estimator" else None)
        R = dict(rhs=cols["rhs"], var=cols["var"], w=cols["w"], A=cols["A"], targets=targets, xs=x[idx[1]],
                 score=cols["score"], Abar=Abar, A_median=float(np.nanmedian(AA[win])), mu=mu, n_rows=len(idx[0]))
        lines.append("")
        lines.append("  [route %s]" % route)
        fits[route] = W.fit_blocks(R, "rcn2d %s | %s route" % (mod, route), args, rng, lines, out, truth=truth, route=route)
        if fields is not None:
            W.fig_library_terms(fields, x, y, out / "library_terms_estimator.png", "rcn2d %s, estimator route, frame %d" % (mod, args.nt // 2))
    lines.append("")
    lines.append("elapsed %.1f s" % (time.time() - t_start))
    rep = "\n".join(lines)
    (out / "report.txt").write_text(rep)
    with open(out / "fits.json", "w") as fh:
        json.dump(fits, fh, indent=1, default=str)
    print(rep); print("outputs ->", out)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below (wsindy_theta_t
    v1.6 fit parameters are shared by name)."""
    outdir = str(_HERE / "results" / "0925_rcn2d_control")
    mu = 0.85
    R = 0.5
    alpha = 1.0 / 3.0
    tau = 1.0 / 3.0
    eta_b = 1.0 / 3.0
    beta0 = 1.8
    # box and integration
    n_xper = 4                   # x periods (Lx = n_xper 2 pi / k1 = 8 L)
    dx_target = 0.2
    Ly = 120.0                   # two knees, 10 L apart
    t_end = 2.0
    nt = 41
    dt = 0.01                    # ETDRK4 step (sub-stepped to the frame cadence); 0.01, 0.004, 0.002, 0.001 agree to 2e-7 at t 0.5
    # planted modulation of the lower crease
    profile = "sin"              # sin | bumps
    m = 2                        # sin mode over Lx
    n_bumps = 8                  # bumps per box (one per spacing at n_xper 4)
    sig_b = 1.0                  # bump width, x units
    eps_b = 0.0                  # sharpness modulation b = beta0 (1 + eps_b s)
    a_pos = 1.0                  # position modulation amplitude, x units
    sig_env = 5.96               # envelope of the position modulation (one spacing at 0.85)
    a_bump = 0.0                 # phase bumps on the crease at n_bumps positions, amplitude in radians
    sig_bump = 2.0               # bump width, x units
    a_dip = 0                    # T2.10: 1 = plant amplitude dips A = A_eq tanh(r / dip_rc) at n_bumps positions on the crease
    dip_rc = 1.5                 # dip radius, x units (0.25 L at 0.85; the SH core: A 0.77 at 0.6 L)
    # estimator
    sigma_dir = 0.4
    taper_frac = 0.06
    core_frac = 0.70
    # weak form and fits (wsindy v1.6 names)
    radius = 1.5
    t_half = 3
    eta = 0.0
    poly_p = 6
    stride_xy = 4
    stride_t = 1
    rim = 4
    rel_tol = 0.01
    abs_tol = 0.02
    min_contrib = 0.05
    seed = 0
    lib_ext = 1
    domain = "band"              # band (|y - y_l| < band_to L) | all_lower (the 1d control's box)
    band_to = 2.0
    core_excl = 0.0
    excl_dilate = 0
    score_h = 6
    chi_m = 1.0
    bias_json = ""


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in dir(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    cli = p.parse_args()
    sys.exit(main(cli))
