#!/usr/bin/env python
"""
phase_vs_amplitude_charge_v11.py  v1.1 (27 Oct 2026)

v1.1 changes (v1.0 is kept unchanged as phase_vs_amplitude_charge.py; no v1.0 column changes value):
  (1) Fold diagnostics. v1.0's run of record found cos(phi_LS) reading |Gamma| about 1 at 0.82 to 0.94 although
      phi_LS is single valued; on the stand-in at 0.88 the plain circulation of grad phi_LS on the same loops
      was 0.000 to 0.002 turn while the sign-carried circulation (loop_grid) of grad phi_LS read 0.96 to 0.98,
      with two sign flips per loop where the loop crosses the line (|grad phi_LS| 0.41 to 0.54 there): the
      phase folds across the line, like |y|, and the carried sign counts each reversal. New columns:
      G_plain_max_gradphi and G_carried_max_gradphi (the two circulations of grad phi_LS itself, largest on
      u's dips), flips_med_gradphi and flips_med_u (median number of carried-sign flips per loop, for grad
      phi_LS and for mg's k of u), fold_frac_line_gradphi and fold_frac_line_k_u (fraction of the line, across
      the window, where the vector fold_off_L above the line points against the one below), and
      fold_len_per_L_gradphi and fold_len_per_L_k_u (total length of pixel edges across which the vector
      reverses, in the core band, per unit length of line).
  (2) A check on the cos(theta) circulation zeros (0.82 to 0.85 in both sweeps while u reads 1). New field
      cos_theta_reg = u / sqrt(A^2 + (reg_eps A_far)^2): the amplitude divided out except where A is below
      about reg_eps of far field. If the zeros come from dividing by small A at the cores, cos_theta_reg
      reads 1 where cos(theta) reads 0.
  (3) Summary figure: (a) adds cos_theta_reg; (b) adds the plain and carried circulation of grad phi_LS;
      (c) shows the curl fraction and the fold measures (the residual rms(u - A cos phi_LS) is dropped
      from the figure, still in the table: on the large sweep a 1 percent error in k accumulates to a few
      tenths of a radian over the 9 L box, about 0.3 even in the knees).

phase_vs_amplitude_charge.py  v1.0 (27 Oct 2026)

Is the director charge of a zigzag boundary carried by the phase or by the
amplitude? Three experiments, one script.

  Experiment 1, the planted control (--control 1). Synthetic fields built on
  the exact knee phase theta_K = k1 x + log(2 cosh(k2 y)), which has no charge,
  with three kinds of planted amplitude:
    knee   u = cos(theta_K), nothing planted.
    mult   u = a(x, y) cos(theta_K) with a real, non-negative amplitude a
           carrying Gaussian dips of depth d (two per dislocation spacing at
           +- 0.3 L off the line, like the intermediates, or one per spacing on
           the line, like the dislocations). d = 1 puts a zero of a at each
           dip centre.
    pair   u = Re(W exp(i theta_K)) with a complex envelope W that has a
           vortex and an antivortex on the line at separation s (alternating,
           period 2 L, so s = L gives evenly spaced cores of alternating sign).
           A negative s means the pair has not formed yet: |W| has a dip and
           no zero. The parameter passes through s = 0 continuously, so this
           is a nucleation with an arbitrarily small separation early on.
  Each control field goes through the same measurements as the data below.

  Experiment 2, the identity check (always run on the data). mg's z = u + i v
  gives A = |z| and theta = arg z with A cos(theta) = u exactly, and 2D
  unwrapping of theta adds multiples of 2 pi, so cos(unwrapped theta) equals
  cos(theta). The script verifies both numerically and runs the director-
  charge detector on u/A = cos(theta) itself: the field with the amplitude
  divided out. If cos(theta) carries the same charge as u, the charge does not
  need the amplitude.

  Experiment 3, the phase-model test (always run on the data). phi_LS is the
  single-valued smooth phase whose gradient is closest to mg's k (least-squares
  integration, Neumann Poisson, the validation script's integrate_k), i.e. the
  best a phase model can do with the measured wavevector. Detector charge and
  circulation are measured on cos(phi_LS) and on A cos(phi_LS), alongside the
  part of k that phi_LS cannot represent (the curl fraction) and the leftover
  envelope phase psi = arg(z exp(-i phi_LS)).

Measurements, per field F (u, cos theta, cos unwrapped theta, cos phi_LS,
A cos phi_LS): director charge per L (t1c.detect_defects at sigma_det 0.3 and
the survey's band and window, exactly as pgb_charge_circ_survey_v12), counts
of V and X cells, and the largest closed-loop circulation |Gamma| of the mg k
re-estimated from F, on 0.5 L loops about the amplitude dips found in u (for
the control: about the planted dips or zeros).

phi_LS is the Helmholtz projection of k (its curl-free part). It is smooth and has no cuts, so cos(phi_LS)
can show director cells only where phi_LS has critical points (extrema and saddles, full-turn defects of
grad phi_LS) or where its level sets bend more sharply than the detection width resolves. The option
--ls_mode seam instead gives zero weight to the pixels where mg's k reverses between neighbours; that lets
phi jump across those pixels, i.e. it puts in cuts, and is kept only for comparison.

Stand-in preview (27 Oct, not of record): knee 0.60 nothing anywhere; intermediates 0.72 and 0.78:
u 3.9 to 4.0 per L, cos(theta) 0, cos(phi_LS) 0, A cos(phi_LS) 3.9 to 4.2, no circulation; dislocations
0.88 and 0.95: u and cos(theta) charged with |Gamma| 1, cos(phi_LS) charged through critical points
(0.95) or sharp bends (0.88). Control: real dips of depth 0.5 or more off the line charge u while
cos(theta) stays at 0; a complex-envelope pair charges u before its zeros exist, and the circulation
about one zero reads 0 until the separation reaches about 0.75 to 1 L, while the curl fraction rises
continuously from below s = 0.

Outputs (in --outdir):
  table.csv / table.json   one row per file (data) or per variant (control)
  fig_summary.png/.pdf     data: charge per L and circulation of each field
                           against mu, and the curl fraction and residuals
  <stem>/fig_fields.png    data, at the --fig_mus angles: the four fields with
                           their V and X cells, then theta (wrapped), phi_LS,
                           psi and the curl residual |k - grad phi_LS|
  control/fig_control_mu<mu>.png   the control fields (u, mg amplitude, cos phi_LS) with their cells
  control/fig_control_summary.png  (a) charge against the depth of real dips; (b) charge against the pair
                           separation; (c) circulation about one zero, curl fraction, amplitude minimum
  control/control_table.csv/.json
  notes.txt                version, inputs, flags, one line per row

Imports pgb_charge_circ_survey_v12 (loader, estimator, dip patches, which in
turn import pgb_marker_survey), pgb_marker_survey.loop_grid and
mg_estimator_validation.integrate_k; edits none of them. Python 3.8
compatible.

Run from ~/patterns/pipelines (see PHASE_AMP_CHARGE_CLI_1027.txt).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgb_charge_circ_survey_v12 as PCS                                   # noqa: E402
from pgb_charge_circ_survey_v12 import load_frame, estimate, t1c, find_dip_patches  # noqa: E402
from pgb_marker_survey import loop_grid, expand_inputs                     # noqa: E402
from mg_estimator_validation import integrate_k                            # noqa: E402
import matplotlib                                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402

VERSION = "phase_vs_amplitude_charge v1.1 (27 Oct 2026)"

FIELDS = ["u", "cos_theta", "cos_theta_reg", "cos_unwrap", "cos_phiLS", "A_cos_phiLS"]
FIELD_LABEL = {"u": "u (the pattern)",
               "cos_theta": "cos(theta) = u/A",
               "cos_theta_reg": "u / sqrt(A^2 + eps^2)",
               "cos_unwrap": "cos(unwrapped theta)",
               "cos_phiLS": "cos(phi_LS)",
               "A_cos_phiLS": "A cos(phi_LS)"}
FIELD_COLOR = {"u": "k", "cos_theta": "C0", "cos_theta_reg": "C2", "cos_unwrap": "C9",
               "cos_phiLS": "C3", "A_cos_phiLS": "C1"}
CIRC_FIELDS = ["u", "cos_theta", "cos_theta_reg", "cos_phiLS", "A_cos_phiLS"]   # re-estimated for circulation


# ---------------------------------------------------------------------
# the measurement shared by data and control
# ---------------------------------------------------------------------

def geometry(u2, x, y, y_row, mu, args):
    k1t = float(np.sqrt(1 - mu ** 2)); L = np.pi / k1t
    ny, nx = u2.shape
    dx = float(x[1] - x[0]); dy = float(y[1] - y[0])
    Lx, Ly = nx * dx, ny * dy
    X, Y = np.meshgrid(x, y)
    xm, ym = float(x.mean()), float(y.mean())
    win = (np.abs(X - xm) < 0.4 * Lx) & (np.abs(Y - ym) < 0.4 * Ly)
    core = win & (np.abs(Y - y_row) < args.core_band_L * L)
    taper = t1c.edge_taper(x, y, frac=args.taper_frac)
    return dict(mu=mu, L=L, Lx=Lx, Ly=Ly, X=X, Y=Y, xm=xm, ym=ym, win=win, core=core,
                taper=taper, y_row=y_row, line_len_L=0.8 * Lx / L, dx=dx, dy=dy)


def cells(F, x, y, G, args):
    """director cells of F in the survey's band and window (pgb_charge_circ_survey_v12, step for step)"""
    L, y_row = G["L"], G["y_row"]
    det = t1c.detect_defects(F * G["taper"], x, y - y_row, G["Lx"], G["Ly"],
                             sigma_det=args.sigma_det, band_y=args.band_L * L)
    q = det["q"]
    J, I = np.nonzero(q)
    px, py, pq = det["xc"][I], det["yc"][J] + y_row, q[J, I].astype(int)
    keep = ((np.abs(px - G["xm"]) < 0.4 * G["Lx"]) & (np.abs(py - G["ym"]) < 0.4 * G["Ly"])
            & (np.abs(py - y_row) < args.band_L * L))
    return px[keep], py[keep], pq[keep]


def seam_mask(k1, k2, dilate=1):
    """pixels where mg's k reverses between neighbours: the sign seams of the director
    convention, which sit beside the half-turn cells (no sign choice removes them there)"""
    sx = (k1[:, 1:] * k1[:, :-1] + k2[:, 1:] * k2[:, :-1]) < 0
    sy = (k1[1:] * k1[:-1] + k2[1:] * k2[:-1]) < 0
    m = np.zeros(k1.shape, bool)
    m[:, 1:] |= sx; m[:, :-1] |= sx; m[1:] |= sy; m[:-1] |= sy
    if dilate > 0:
        from scipy.ndimage import binary_dilation
        m = binary_dilation(m, iterations=dilate)
    return m


def weighted_ls_phase(k1, k2, dx, dy, w, tol=1e-8, maxiter=2000):
    """phi minimising sum over grid edges of w_edge (D phi - k_edge)^2, with forward
    differences and edge weights w_edge = w at both ends (natural, Neumann, boundaries).
    Conjugate gradients preconditioned by the unweighted Neumann Laplacian (a DCT solve),
    which is exact when every weight is 1. Returns (phi, iterations, relative residual)."""
    from scipy.fft import dctn, idctn
    from scipy.sparse.linalg import LinearOperator, cg
    ny, nx = k1.shape
    wx = w[:, 1:] * w[:, :-1]
    wy = w[1:] * w[:-1]
    kx = 0.5 * (k1[:, 1:] + k1[:, :-1]); ky = 0.5 * (k2[1:] + k2[:-1])

    def Dt(gx, gy):                                    # minus the divergence: D^T applied to edge fields
        o = np.zeros((ny, nx))
        o[:, :-1] -= gx / dx; o[:, 1:] += gx / dx
        o[:-1] -= gy / dy; o[1:] += gy / dy
        return o

    def Mv(v):
        f = v.reshape(ny, nx)
        gx = wx * (f[:, 1:] - f[:, :-1]) / dx
        gy = wy * (f[1:] - f[:-1]) / dy
        return Dt(gx, gy).ravel()
    lam = ((2 - 2 * np.cos(np.pi * np.arange(ny) / ny))[:, None] / dy ** 2
           + (2 - 2 * np.cos(np.pi * np.arange(nx) / nx))[None, :] / dx ** 2)
    lam[0, 0] = 1.0

    def Pv(v):
        F = dctn(v.reshape(ny, nx), type=2, norm="ortho") / lam
        F[0, 0] = 0.0
        return idctn(F, type=2, norm="ortho").ravel()
    b = Dt(wx * kx, wy * ky).ravel()
    b -= b.mean()
    it = [0]

    def cb(_):
        it[0] += 1
    M = LinearOperator((nx * ny, nx * ny), matvec=Mv, dtype=float)
    Pc = LinearOperator((nx * ny, nx * ny), matvec=Pv, dtype=float)
    try:
        sol, info = cg(M, b, rtol=tol, maxiter=maxiter, M=Pc, callback=cb)
    except TypeError:                                  # scipy < 1.12
        sol, info = cg(M, b, tol=tol, maxiter=maxiter, M=Pc, callback=cb)
    res = float(np.linalg.norm(Mv(sol) - b) / max(np.linalg.norm(b), 1e-30))
    phi = sol.reshape(ny, nx)
    return phi - phi.mean(), it[0], res


def smooth_phase(e, x, y, G, args):
    """phi_LS: the single-valued phase whose gradient is closest to mg's k, shifted by the
    constant that best aligns exp(i phi) with z on the window. ls_mode "seam" (default)
    gives the seam pixels zero weight; "plain" is the validation script's integrate_k."""
    info = {}
    if args.ls_mode == "plain":
        phi = integrate_k(e["k1"], e["k2"], x, y)
        info.update(ls_iter=0, ls_res=0.0, seam_frac_core=float("nan"))
    else:
        seam = seam_mask(e["k1"], e["k2"], args.seam_dilate)
        w = np.where(seam, 0.0, 1.0)
        phi, nit, res = weighted_ls_phase(e["k1"], e["k2"], G["dx"], G["dy"], w)
        info.update(ls_iter=nit, ls_res=res, seam_frac_core=float(np.mean(seam[G["core"]])))
        G["_seam"] = seam
    c = float(np.angle(np.sum((e["z"] * np.exp(-1j * phi))[G["win"]])))
    return phi + c, info


def _trap(f, a):
    """trapezoid rule along the last axis (np.trapz is renamed in numpy 2; this works in both)"""
    return np.sum(0.5 * (f[..., 1:] + f[..., :-1]) * np.diff(a), axis=-1)


def loop_plain(k1, k2, x, y, xc, yc, r, nang):
    """circulation (turns) on the same circles and samples as loop_grid, without carrying a sign: the
    ordinary line integral, which is zero for the gradient of any single-valued phase; also returns the
    number of places on each loop where consecutive samples point against each other (the flips the
    carried sign would make)"""
    dx = x[1] - x[0]; dy = y[1] - y[0]
    ang = np.linspace(0, 2 * np.pi, nang)
    px = xc[:, None] + r * np.cos(ang)[None, :]
    py = yc[:, None] + r * np.sin(ang)[None, :]
    ix = np.clip(np.round((px - x[0]) / dx).astype(int), 0, len(x) - 1)
    iy = np.clip(np.round((py - y[0]) / dy).astype(int), 0, len(y) - 1)
    kx, ky = k1[iy, ix], k2[iy, ix]
    circ = _trap(kx * (-np.sin(ang))[None, :] + ky * np.cos(ang)[None, :], ang) * r / (2 * np.pi)
    flips = np.sum(kx[:, 1:] * kx[:, :-1] + ky[:, 1:] * ky[:, :-1] < 0, axis=1)
    return circ, flips


def fold_measures(k1, k2, y, G, args):
    """where a vector field reverses: (a) the fraction of the line, across the window, where the vector
    fold_off_L above the line points against the one below; (b) the length of pixel edges in the core band
    across which neighbouring vectors point against each other, per unit length of line"""
    L = G["L"]
    jr = int(np.argmin(np.abs(y - G["y_row"])))
    off = max(1, int(round(args.fold_off_L * L / G["dy"])))
    cols = np.abs(G["X"][0] - G["xm"]) < 0.4 * G["Lx"]
    a, b = min(jr + off, len(y) - 1), max(jr - off, 0)
    dot = k1[a, cols] * k1[b, cols] + k2[a, cols] * k2[b, cols]
    frac_line = float(np.mean(dot < 0))
    core = G["core"]
    vx = (k1[:, 1:] * k1[:, :-1] + k2[:, 1:] * k2[:, :-1] < 0) & core[:, 1:] & core[:, :-1]
    vy = (k1[1:] * k1[:-1] + k2[1:] * k2[:-1] < 0) & core[1:] & core[:-1]
    length = float(vx.sum() * G["dy"] + vy.sum() * G["dx"])
    return frac_line, length / (0.8 * G["Lx"])


def measure(u2, x, y, G, args, dips=None, keep_maps=False):
    """every field, its cells and circulation, and the phase diagnostics"""
    out, maps = {}, {}
    e = estimate(u2, G["Lx"], G["Ly"], args.sigma_dir, None, G["taper"])
    A, th = e["A"], e["theta"]
    win, core = G["win"], G["core"]
    A_far = float(np.median(A[win & (np.abs(G["Y"] - G["y_row"]) > args.far_from * G["L"])])) \
        if np.sum(win & (np.abs(G["Y"] - G["y_row"]) > args.far_from * G["L"])) > 50 else float(np.median(A[win]))
    Asafe = np.maximum(A, 1e-12)

    # experiment 2: the identities
    cos_th = np.cos(th)
    out["id_err_Acos_theta"] = float(np.max(np.abs(A * cos_th - u2)[win]))
    out["id_err_u_over_A"] = float(np.max(np.abs(u2 / Asafe - cos_th)[win]))
    rows = np.where(win.any(axis=1))[0]; cols = np.where(win.any(axis=0))[0]
    rs, cs = slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)
    cos_un = cos_th.copy()
    try:
        from skimage.restoration import unwrap_phase
        th_un = np.asarray(unwrap_phase(th[rs, cs]))
        cos_un[rs, cs] = np.cos(th_un)
        out["id_err_unwrap"] = float(np.max(np.abs(cos_un - cos_th)[win]))
        steps = np.round((th_un - th[rs, cs]) / (2 * np.pi))
        out["unwrap_n_levels"] = int(len(np.unique(steps)))
    except Exception:
        th_un = None
        out["id_err_unwrap"] = float("nan"); out["unwrap_n_levels"] = -1

    # experiment 3: the smooth phase
    phi, linfo = smooth_phase(e, x, y, G, args)
    out.update(linfo)
    gy, gx = np.gradient(phi, G["dy"], G["dx"])
    rk = np.hypot(e["k1"] - gx, e["k2"] - gy)
    out["curl_frac_core"] = float(np.sum(rk[core] ** 2) / np.sum((e["k1"] ** 2 + e["k2"] ** 2)[core]))
    out["curl_frac_far"] = float(np.sum(rk[win & ~core] ** 2) / np.sum((e["k1"] ** 2 + e["k2"] ** 2)[win & ~core]))
    psi = np.angle(e["z"] * np.exp(-1j * phi))
    out["psi_rms_core"] = float(np.sqrt(np.mean(psi[core] ** 2)))
    out["psi_frac_gt_halfpi_core"] = float(np.mean(np.abs(psi[core]) > 0.5 * np.pi))
    rmsu = float(np.sqrt(np.mean(u2[core] ** 2)))
    out["res_Acos_core"] = float(np.sqrt(np.mean((u2 - A * np.cos(phi))[core] ** 2)) / rmsu)
    out["res_cos_core"] = float(np.sqrt(np.mean((u2 - A_far * np.cos(phi))[core] ** 2)) / rmsu)
    out["A_min_core_over_far"] = float(A[core].min() / A_far)

    out["fold_frac_line_gradphi"], out["fold_len_per_L_gradphi"] = fold_measures(gx, gy, y, G, args)
    out["fold_frac_line_k_u"], out["fold_len_per_L_k_u"] = fold_measures(e["k1"], e["k2"], y, G, args)
    cos_reg = u2 / np.sqrt(A ** 2 + (args.reg_eps * A_far) ** 2)
    fields = {"u": u2, "cos_theta": cos_th, "cos_theta_reg": cos_reg, "cos_unwrap": cos_un,
              "cos_phiLS": np.cos(phi), "A_cos_phiLS": A * np.cos(phi)}
    if dips is None:
        xd, yd, _, _, _ = find_dip_patches(A, x, y, win, args.core_frac, 0.5 * G["L"])
    else:
        xd, yd = np.asarray(dips[0], float), np.asarray(dips[1], float)
    out["n_dips"] = int(len(xd))
    r = args.loop_L * G["L"]
    if len(xd):
        cp, fl = loop_plain(gx, gy, x, y, xd, yd, r, args.nang)
        cc, ccl = loop_grid(gx, gy, x, y, xd, yd, r, args.nang)
        out["G_plain_max_gradphi"] = float(np.max(np.abs(cp)))
        cc = np.abs(cc); ccl = np.asarray(ccl, bool)
        out["G_carried_max_gradphi"] = float(cc[ccl].max()) if ccl.any() else float("nan")
        out["flips_med_gradphi"] = float(np.median(fl))
        _, flu = loop_plain(e["k1"], e["k2"], x, y, xd, yd, r, args.nang)
        out["flips_med_u"] = float(np.median(flu))
    else:
        out["G_plain_max_gradphi"] = 0.0; out["G_carried_max_gradphi"] = 0.0
        out["flips_med_gradphi"] = 0.0; out["flips_med_u"] = 0.0
    for name in FIELDS:
        px, py, pq = cells(fields[name], x, y, G, args)
        out["T_per_L_" + name] = float(np.abs(pq).sum() / G["line_len_L"])
        out["nV_" + name] = int(np.sum(pq < 0)); out["nX_" + name] = int(np.sum(pq > 0))
        if keep_maps:
            maps["cells_" + name] = (px, py, pq)
        if name in CIRC_FIELDS:
            if len(xd):
                eF = e if name == "u" else estimate(fields[name], G["Lx"], G["Ly"], args.sigma_dir, None, G["taper"])
                circ, closed = loop_grid(eF["k1"], eF["k2"], x, y, xd, yd, r, args.nang)
                circ = np.abs(circ); closed = np.asarray(closed, bool)
                on = np.abs(yd - G["y_row"]) <= args.on_line_L * G["L"]
                out["G_max_closed_" + name] = float(circ[closed].max()) if closed.any() else float("nan")
                out["G_on_med_" + name] = float(np.median(circ[on & closed])) if (on & closed).any() else float("nan")
                out["n_closed_" + name] = int(closed.sum())
            else:
                out["G_max_closed_" + name] = 0.0; out["G_on_med_" + name] = float("nan"); out["n_closed_" + name] = 0
    if out["T_per_L_u"] == 0:
        out["regime"] = "knee"
    elif np.isfinite(out.get("G_on_med_u", np.nan)) and out["G_on_med_u"] >= args.wind_hi:
        out["regime"] = "dislocation"
    else:
        out["regime"] = "intermediate"
    if keep_maps:
        maps.update(fields=fields, theta=th, theta_un=th_un, rs=rs, cs=cs, phi=phi, psi=psi, rk=rk,
                    A=A, A_far=A_far, xd=xd, yd=yd)
    return out, maps


# ---------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------

def _crop(x, y, G, args):
    L = G["L"]
    i = np.abs(x - G["xm"]) <= args.crop_x_L * L / 2
    j = np.abs(y - G["y_row"]) <= args.crop_y_L * L
    return i, j


def _cells_on(ax, cl, G):
    px, py, pq = cl
    L, x0, y0 = G["L"], G["xm"], G["y_row"]
    ax.plot((px[pq > 0] - x0) / L, (py[pq > 0] - y0) / L, "^", ms=5, mfc="#d62728", mec="w", mew=0.6, ls="none")
    ax.plot((px[pq < 0] - x0) / L, (py[pq < 0] - y0) / L, "v", ms=5, mfc="#1f77b4", mec="w", mew=0.6, ls="none")


def fig_fields(stem, x, y, G, M, row, path, args):
    i, j = _crop(x, y, G, args)
    ext = [(x[i][0] - G["xm"]) / G["L"], (x[i][-1] - G["xm"]) / G["L"],
           (y[j][0] - G["y_row"]) / G["L"], (y[j][-1] - G["y_row"]) / G["L"]]
    sub = lambda a: a[np.ix_(j, i)]                                        # noqa: E731
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.2), constrained_layout=True)
    names = ["u", "cos_theta", "cos_phiLS", "A_cos_phiLS"]
    for ax, nm in zip(axs[0], names):
        F = M["fields"][nm]
        ax.imshow(sub(F), origin="lower", extent=ext, cmap="Greys", aspect="equal",
                  vmin=-np.max(np.abs(sub(F))), vmax=np.max(np.abs(sub(F))))
        _cells_on(ax, M["cells_" + nm], G)
        g = row.get("G_max_closed_" + nm, float("nan"))
        ax.set_title("%s\ncharge %.2f per L,  max |Gamma| %s" % (
            FIELD_LABEL[nm], row["T_per_L_" + nm], ("%.2f" % g) if np.isfinite(g) else "n/a"), fontsize=9)
    ax = axs[1, 0]
    ax.imshow(sub(M["theta"]), origin="lower", extent=ext, cmap="twilight", vmin=-np.pi, vmax=np.pi, aspect="equal")
    ax.set_title("theta = arg z (wrapped)", fontsize=9)
    ax = axs[1, 1]
    ph = sub(M["phi"]); im = ax.imshow(ph - np.median(ph), origin="lower", extent=ext, cmap="viridis", aspect="equal")
    ax.contour(np.linspace(ext[0], ext[1], ph.shape[1]), np.linspace(ext[2], ext[3], ph.shape[0]),
               np.cos(ph), levels=[0], colors="w", linewidths=0.4)
    plt.colorbar(im, ax=ax, shrink=0.8, label="rad")
    ax.set_title("phi_LS (smooth, single valued),\nwhite: zeros of cos(phi_LS)", fontsize=9)
    ax = axs[1, 2]
    im = ax.imshow(sub(M["psi"]), origin="lower", extent=ext, cmap="twilight", vmin=-np.pi, vmax=np.pi, aspect="equal")
    _cells_on(ax, M["cells_u"], G)
    plt.colorbar(im, ax=ax, shrink=0.8, label="rad")
    ax.set_title("psi = arg(z exp(-i phi_LS)): what phi_LS misses\n(cells of u)", fontsize=9)
    ax = axs[1, 3]
    im = ax.imshow(sub(M["rk"]), origin="lower", extent=ext, cmap="magma", vmin=0, vmax=1.0, aspect="equal")
    _cells_on(ax, M["cells_u"], G)
    plt.colorbar(im, ax=ax, shrink=0.8, label="|k - grad phi_LS|")
    ax.set_title("curl residual (core band: %.1f%% of |k|^2)" % (100 * row["curl_frac_core"]), fontsize=9)
    for ax in axs.ravel():
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel("x / L", fontsize=8); ax.set_ylabel("y / L", fontsize=8); ax.tick_params(labelsize=7)
    fig.suptitle("%s   mu %.3f (%.0f deg), %s.  red up-triangles X cells, blue down-triangles V cells" % (
        stem, G["mu"], np.degrees(np.arcsin(G["mu"])), row["regime"]), fontsize=10)
    fig.savefig(path, dpi=140); plt.close(fig)


def _regime_spans(ax, rows):
    mus = np.array([r["mu"] for r in rows]); reg = [r["regime"] for r in rows]
    col = {"knee": "#e8f1fa", "intermediate": "#fdf1dc", "dislocation": "#f9e3e3"}
    o = np.argsort(mus); mus = mus[o]; reg = [reg[k] for k in o]
    if len(mus) < 2:
        return
    edges = np.concatenate([[mus[0] - 0.005], 0.5 * (mus[1:] + mus[:-1]), [mus[-1] + 0.005]])
    for k in range(len(mus)):
        ax.axvspan(edges[k], edges[k + 1], color=col[reg[k]], lw=0, zorder=0)


def fig_summary(rows, path):
    rows = sorted(rows, key=lambda r: r["mu"])
    mu = np.array([r["mu"] for r in rows])
    col = lambda k: np.array([r.get(k, np.nan) for r in rows], float)     # noqa: E731
    fig, axs = plt.subplots(3, 1, figsize=(8, 11), sharex=True, constrained_layout=True)
    mk = {"u": "o", "cos_theta": "s", "cos_theta_reg": "D", "cos_unwrap": "x", "cos_phiLS": "v", "A_cos_phiLS": "^"}
    ax = axs[0]; _regime_spans(ax, rows)
    for nm in FIELDS:
        ax.plot(mu, col("T_per_L_" + nm), "-" if nm == "u" else "--", marker=mk[nm], ms=4,
                color=FIELD_COLOR[nm], label=FIELD_LABEL[nm], lw=1.2 if nm == "u" else 0.9)
    ax.set_ylabel("director charge per L\n(half turns, sigma_det 0.3)")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("(a) which fields carry the director charge", fontsize=10, loc="left")
    ax = axs[1]; _regime_spans(ax, rows)
    for nm in CIRC_FIELDS:
        ax.plot(mu, col("G_max_closed_" + nm), "-" if nm == "u" else "--", marker=mk[nm], ms=4,
                color=FIELD_COLOR[nm], label=FIELD_LABEL[nm])
    ax.plot(mu, col("G_carried_max_gradphi"), ":", marker="P", ms=5, color="C4", label="grad phi_LS itself, sign carried")
    ax.plot(mu, col("G_plain_max_gradphi"), ":", marker="*", ms=6, color="0.4", label="grad phi_LS itself, plain line integral")
    ax.set_ylabel("largest closed-loop |Gamma|\nabout u's dips (turns)")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("(b) circulation: re-estimated from each field, and of grad phi_LS", fontsize=10, loc="left")
    ax = axs[2]; _regime_spans(ax, rows)
    ax.plot(mu, col("curl_frac_core"), "-o", ms=3, color="C4", label="curl fraction of k, core band")
    ax.plot(mu, col("fold_frac_line_gradphi"), "-v", ms=3, color="C3", label="fraction of the line where grad phi_LS folds")
    ax.plot(mu, col("fold_frac_line_k_u"), "--s", ms=3, color="k", mfc="none", label="fraction of the line where mg's k of u folds")
    ax.plot(mu, col("psi_frac_gt_halfpi_core"), "-^", ms=3, color="C2", label="fraction of core band with |psi| > pi/2")
    ax.set_ylabel("fraction"); ax.set_xlabel("mu = sin(alpha)")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("(c) what the smooth phase cannot represent, and where vectors reverse", fontsize=10, loc="left")
    fig.savefig(path, dpi=150); fig.savefig(str(path).replace(".png", ".pdf")); plt.close(fig)


# ---------------------------------------------------------------------
# experiment 1: the planted control
# ---------------------------------------------------------------------

def control_grid(mu, args):
    k1 = np.sqrt(1 - mu ** 2); L = np.pi / k1
    Lx = args.ctl_box_x_L * L
    nx = args.ctl_nx; dx = Lx / nx
    ny = int(2 * round(0.5 * args.ctl_box_y_L * L / dx))
    x = (np.arange(nx) - nx / 2) * dx
    y = (np.arange(ny) - ny / 2) * dx
    return x, y, L, k1


def control_fields(mu, args):
    """yields (variant, parameter, u, (xd, yd)) for the planted control at one mu"""
    x, y, L, k1 = control_grid(mu, args)
    X, Y = np.meshgrid(x, y)
    th = k1 * X + np.logaddexp(mu * Y, -mu * Y)                           # log(2 cosh(mu y))
    yield "knee", 0.0, np.cos(th), (np.zeros(0), np.zeros(0)), x, y
    w = args.ctl_dip_w_L * L
    xs = (np.arange(-int(args.ctl_box_x_L), int(args.ctl_box_x_L) + 1) + 0.5) * L
    xs = xs[np.abs(xs) < 0.5 * args.ctl_box_x_L * L]
    for layout in ["off", "on"]:
        if layout == "off":
            cx = np.concatenate([xs, xs]); cy = np.concatenate([np.full(len(xs), args.ctl_off_L * L), np.full(len(xs), -args.ctl_off_L * L)])
        else:
            cx, cy = xs, np.zeros(len(xs))
        g = np.zeros_like(X)
        for a, b in zip(cx, cy):
            g = np.maximum(g, np.exp(-((X - a) ** 2 + (Y - b) ** 2) / (2 * w ** 2)))
        for d in [float(s) for s in args.ctl_depths.split(",")]:
            yield "mult_" + layout, d, (1 - d * g) * np.cos(th), (cx, cy), x, y
    P = 2 * L
    ww = args.ctl_pair_w_L * L
    for s in [float(v) for v in args.ctl_seps.split(",")]:
        c0 = np.cos(np.pi * s * L / P) if s >= 0 else np.cosh(np.pi * abs(s) * L / P)
        q = (1 + np.cos(2 * np.pi * X / P)) / (1 + c0)
        env = np.exp(-Y ** 2 / ww ** 2)
        W = (1 - env * q) + 1j * (Y / ww) * env * args.ctl_pair_beta
        if s > 0:
            base = np.arange(-int(args.ctl_box_x_L), int(args.ctl_box_x_L) + 1) * P
            zx = np.concatenate([base - 0.5 * s * L, base + 0.5 * s * L])
            zx = np.sort(zx[np.abs(zx) < 0.5 * args.ctl_box_x_L * L])
        else:
            base = np.arange(-int(args.ctl_box_x_L), int(args.ctl_box_x_L) + 1) * P
            zx = base[np.abs(base) < 0.5 * args.ctl_box_x_L * L]
        yield "pair", s, np.real(W * np.exp(1j * th)), (zx, np.zeros(len(zx))), x, y


def run_control(args, out):
    cdir = out / "control"; cdir.mkdir(parents=True, exist_ok=True)
    rows, keep = [], []
    show = [("knee", 0.0), ("mult_off", 0.5), ("mult_off", 0.7), ("mult_on", 0.7), ("pair", -0.2),
            ("pair", 0.2), ("pair", 0.5), ("pair", 1.0)]
    for mu in [float(v) for v in args.ctl_mus.split(",")]:
        for var, par, u2, dips, x, y in control_fields(mu, args):
            G = geometry(u2, x, y, 0.0, mu, args)
            row, M = measure(u2, x, y, G, args, dips=dips, keep_maps=True)
            row.update(variant=var, param=par, mu=mu)
            # the pair's own circulation on a loop small enough to hold one zero
            if var == "pair" and par > 0 and len(dips[0]):
                e = estimate(u2, G["Lx"], G["Ly"], args.sigma_dir, None, G["taper"])
                rr = min(args.loop_L * G["L"], 0.4 * par * G["L"])
                c, cl = loop_grid(e["k1"], e["k2"], x, y, dips[0], dips[1], rr, args.nang)
                row["G_small_loop_med_u"] = float(np.median(np.abs(c)[cl])) if np.any(cl) else float("nan")
                row["small_loop_r_L"] = rr / G["L"]
            rows.append(row)
            if (var, par) in show:
                keep.append((mu, var, par, x, y, G, M, row))
            print("control mu %.2f %-9s %5.2f  charge per L: u %.2f, cos theta %.2f, cos phiLS %.2f, A cos phiLS %.2f;"
                  "  max|Gamma| u %s, cos phiLS %s;  A_min %.2f" % (
                      mu, var, par, row["T_per_L_u"], row["T_per_L_cos_theta"], row["T_per_L_cos_phiLS"],
                      row["T_per_L_A_cos_phiLS"], _f(row.get("G_max_closed_u")), _f(row.get("G_max_closed_cos_phiLS")),
                      row["A_min_core_over_far"]))
    write_rows(rows, cdir / "control_table")
    for mu in sorted(set(k[0] for k in keep)):
        sel = [k for k in keep if k[0] == mu]
        fig, axs = plt.subplots(len(sel), 3, figsize=(12, 2.1 * len(sel)), constrained_layout=True)
        for r_, (mu_, var, par, x, y, G, M, row) in enumerate(sel):
            i, j = _crop(x, y, G, args)
            ext = [(x[i][0] - G["xm"]) / G["L"], (x[i][-1] - G["xm"]) / G["L"],
                   (y[j][0]) / G["L"], (y[j][-1]) / G["L"]]
            sub = lambda a: a[np.ix_(j, i)]                                # noqa: E731
            for c_, (nm, img, cmap) in enumerate([("u", M["fields"]["u"], "Greys"), ("A", M["A"] / M["A_far"], "viridis"),
                                                  ("cos_phiLS", M["fields"]["cos_phiLS"], "Greys")]):
                ax = axs[r_, c_]
                ax.imshow(sub(img), origin="lower", extent=ext, cmap=cmap, aspect="equal",
                          vmin=(0 if nm == "A" else -1.0), vmax=(1.1 if nm == "A" else 1.0))
                if nm != "A":
                    _cells_on(ax, M["cells_" + nm], G)
                    ax.set_title("%s %g: %s, charge %.2f per L, max |Gamma| %s" % (
                        var, par, "u" if nm == "u" else "cos(phi_LS)", row["T_per_L_" + nm],
                        _f(row.get("G_max_closed_" + nm))), fontsize=8)
                else:
                    ax.set_title("A / A_far (mg), min %.2f" % row["A_min_core_over_far"], fontsize=8)
                ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
                ax.tick_params(labelsize=6)
        fig.suptitle("planted control on the exact knee phase, mu %.2f (red up X cells, blue down V cells)" % mu, fontsize=10)
        fig.savefig(cdir / ("fig_control_mu%.2f.png" % mu), dpi=130); plt.close(fig)
    # summary against the planted parameters
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
    mus = sorted(set(r["mu"] for r in rows))
    for im_, mu in enumerate(mus):
        ls = "-" if im_ == 0 else ":"
        for lay, m_ in [("mult_off", "s"), ("mult_on", "D")]:
            Q = sorted([r for r in rows if r["variant"] == lay and r["mu"] == mu], key=lambda r: r["param"])
            d = [r["param"] for r in Q]
            axs[0].plot(d, [r["T_per_L_u"] for r in Q], ls, marker=m_, color="k", label="u, %s, mu %.2f" % (lay, mu))
            axs[0].plot(d, [r["T_per_L_cos_theta"] for r in Q], ls, marker=m_, color="C0", mfc="none",
                        label="cos(theta), %s, mu %.2f" % (lay, mu))
        P = sorted([r for r in rows if r["variant"] == "pair" and r["mu"] == mu], key=lambda r: r["param"])
        sp = [r["param"] for r in P]
        for nm in ["u", "cos_theta", "cos_phiLS"]:
            axs[1].plot(sp, [r["T_per_L_" + nm] for r in P], ls, marker="o", ms=4, color=FIELD_COLOR[nm],
                        label="%s, mu %.2f" % (FIELD_LABEL[nm], mu))
        axs[2].plot(sp, [r.get("G_small_loop_med_u", np.nan) for r in P], ls, marker="o", ms=4, color="k",
                    label="|Gamma| of u about one zero, mu %.2f" % mu)
        axs[2].plot(sp, [r["curl_frac_core"] for r in P], ls, marker="s", ms=4, color="C4",
                    label="curl fraction of k, mu %.2f" % mu)
        axs[2].plot(sp, [r["A_min_core_over_far"] for r in P], ls, marker="^", ms=4, color="C2",
                    label="amplitude minimum / far field, mu %.2f" % mu)
    axs[0].set_xlabel("depth d of the real dips (A_min about 1 - d)"); axs[0].set_ylabel("director charge per L")
    axs[0].set_title("(a) real, non-negative amplitude on the knee phase", fontsize=9, loc="left")
    for ax in axs[1:]:
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_xlabel("pair separation s / L  (s < 0: no zero yet)")
    axs[1].set_ylabel("director charge per L")
    axs[1].set_title("(b) complex envelope: a vortex pair forms at s = 0", fontsize=9, loc="left")
    axs[2].set_ylabel("turns / fraction")
    axs[2].set_title("(c) the pair: circulation, curl and amplitude", fontsize=9, loc="left")
    for ax in axs:
        ax.legend(fontsize=6.5)
    fig.savefig(cdir / "fig_control_summary.png", dpi=140); plt.close(fig)
    return rows


# ---------------------------------------------------------------------
# tables and main
# ---------------------------------------------------------------------

def _f(v):
    return ("%.2f" % v) if (v is not None and np.isfinite(v)) else "n/a"


def write_rows(rows, base):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(str(base) + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(str(base) + ".json", "w") as fh:
        json.dump(rows, fh, indent=1, default=float)


def main(args):
    t0 = time.time()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    lines = [VERSION, "args: " + json.dumps(vars(args)), ""]
    if args.control:
        crow = run_control(args, out)
        lines.append("control: %d variants written to control/" % len(crow))
    files = expand_inputs(args.npz) if args.npz else []
    rows = []
    fig_mus = [float(v) for v in args.fig_mus.split(",")] if args.fig_mus else []
    sargs = PCS.default_args(); sargs.frame = args.frame
    for p in files:
        R = load_frame(p, sargs)
        u2, x, y, md = R["u2"], R["x"], R["y"], R["md"]
        mu = float(md.get("mu", args.mu))
        jr = int(md.get("gb_row_in_crop", u2.shape[0] // 2)); y_row = float(y[jr])
        G = geometry(u2, x, y, y_row, mu, args)
        want = any(abs(mu - m) < 0.0051 for m in fig_mus)
        row, M = measure(u2, x, y, G, args, keep_maps=want)
        row = dict(stem=R["stem"], mu=mu, alpha_deg=float(np.degrees(np.arcsin(mu))), t=R["t"], frame=R["frame"], **row)
        rows.append(row)
        if want:
            d = out / R["stem"]; d.mkdir(exist_ok=True)
            fig_fields(R["stem"], x, y, G, M, row, d / "fig_fields.png", args)
        msg = ("%s mu %.3f %-12s charge per L: u %.2f, cos theta %.2f, cos unwrap %.2f, cos phiLS %.2f, A cos phiLS %.2f;"
               "  cos theta reg %.2f;  fold line %.2f (k of u %.2f), Gamma of grad phiLS plain %.3f carried %s"
               "  max|Gamma| u %s, cos theta %s, cos theta reg %s, cos phiLS %s, A cos phiLS %s;  curl %.3f;  id errs %.1e %.1e %.1e" % (
                   R["stem"], mu, row["regime"], row["T_per_L_u"], row["T_per_L_cos_theta"], row["T_per_L_cos_unwrap"],
                   row["T_per_L_cos_phiLS"], row["T_per_L_A_cos_phiLS"], row["T_per_L_cos_theta_reg"],
                   row["fold_frac_line_gradphi"], row["fold_frac_line_k_u"], row["G_plain_max_gradphi"],
                   _f(row["G_carried_max_gradphi"]), _f(row["G_max_closed_u"]),
                   _f(row["G_max_closed_cos_theta"]), _f(row["G_max_closed_cos_theta_reg"]), _f(row["G_max_closed_cos_phiLS"]),
                   _f(row["G_max_closed_A_cos_phiLS"]), row["curl_frac_core"], row["id_err_Acos_theta"],
                   row["id_err_u_over_A"], row["id_err_unwrap"]))
        print(msg); lines.append(msg)
    if rows:
        write_rows(rows, out / "table")
        fig_summary(rows, out / "fig_summary.png")
    lines.append("")
    lines.append("done in %.0f s" % (time.time() - t0))
    (out / "notes.txt").write_text("\n".join(lines) + "\n")
    print("done in %.0f s; outputs in %s" % (time.time() - t0, out))
    return 0


class _Args:
    """defaults; mirrored by the CLI below"""
    npz = ""                       # comma-separated files and/or folders (as the survey); "" for control only
    outdir = str(Path("results") / "phase_vs_amplitude_charge_v11")
    frame = -1                     # frame index; -1 = last
    mu = 0.85                      # fallback when a file carries no mu
    control = 0                    # 1: run experiment 1 (the planted control)
    fig_mus = "0.60,0.74,0.80,0.83,0.88,0.95"   # draw fig_fields for files within 0.005 of these
    sigma_dir = 0.4                # mg director width (production value)
    taper_frac = 0.06
    sigma_det = 0.3                # director-charge width (the survey's)
    band_L = 2.0                   # band half-height for the charge, L (the survey's)
    core_band_L = 1.0              # band half-height for curl, residuals and psi, L
    far_from = 4.0                 # far field for A_far, L
    core_frac = 0.70               # dips: A below this fraction of the window median (the survey's)
    loop_L = 0.5                   # loop radius, L (the survey's)
    fold_off_L = 0.05              # v1.1: the fold test compares vectors this far above and below the line, L
    reg_eps = 0.1                  # v1.1: cos_theta_reg = u / sqrt(A^2 + (reg_eps A_far)^2)
    ls_mode = "plain"              # smooth phase: "plain" (integrate_k, the Helmholtz projection: smooth, no cuts) or
                                   # "seam" (seam pixels get zero weight; lets phi jump across the seams, i.e. cuts)
    seam_dilate = 1                # seam mask dilation, pixels
    nang = 181
    on_line_L = 0.1
    wind_hi = 0.9
    crop_x_L = 4.0
    crop_y_L = 1.5
    # control
    ctl_mus = "0.74,0.88"
    ctl_box_x_L = 8.0              # control box along the line, L (an even number keeps the pair period)
    ctl_box_y_L = 5.0              # control box height, L
    ctl_nx = 512
    ctl_dip_w_L = 0.3              # Gaussian width of the planted real dips, L (0.15 is narrower than a stripe and unphysical)
    ctl_off_L = 0.3                # off-line dips at +- this, L
    ctl_depths = "0.3,0.5,0.7,0.9,1.0"
    ctl_seps = "-0.4,-0.2,0,0.1,0.2,0.3,0.5,0.75,1.0"   # pair separations, L
    ctl_pair_w_L = 0.3             # height of the pair's envelope, L
    ctl_pair_beta = 1.0


def default_args():
    return argparse.Namespace(**{k: getattr(_Args, k) for k in vars(_Args) if not k.startswith("_")})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in vars(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    sys.exit(main(p.parse_args()))
