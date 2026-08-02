# pipelines/pgb_signature_explorer.py
"""
Exploratory search for knee-vs-dislocation signatures along a PGB core.

Motivation: every "shape" measure tried so far is spectrally degenerate
between a defect-free sharp knee and a dislocation chain -- the knee's
core row oscillates at k1, the chain's gap alternation is a square wave
with fundamental k1, the corrugation artifact and the real bead train
both sit at 2*k1. This script therefore concentrates on measures the
degeneracy cannot reach:

  A. ENVELOPE (raw pattern, no averaging): Hilbert envelope of u along
     the core row. Knee: ~constant (phase oscillates, envelope doesn't).
     Chain: vanishes at each core (a dislocation IS an amplitude zero).
     Metrics: env_min / env_median, number of deep minima.

  B. TWIST (quantized charge): disk integrals of J = det(grad k) from a
     plain ISOTROPIC extraction, at points along the midline. Signed
     integration cancels the corrugation artifact (alternating sign) but
     accumulates a real dislocation's quantized circulation.
     Metrics: max |twist| and its contrast against the off-core level.

  C. GABOR ANGLE (raw pattern): windowed 2D spectrum at points along the
     midline, energy folded into director angle mod pi. Knee: two fixed
     angular peaks (the +-k1 branches), stationary in x. Chain: dominant
     angle swings toward the pi/2-rotated (horizontal-k) content at each
     core and back. Metrics: range of the dominant angle along the core;
     mean secondary-peak fraction.

  D. BEADING (control, KNOWN degenerate): along-core power of lam2 at
     2*k1 from the isotropic pass. Included deliberately so the summary
     table shows it failing to separate the cases, as a caution.

Runs on the analytic knee and Pismen chain (staggered grid), and
optionally on an SH OP/state npz (--sh_npz, expects keys u, x, y, ramp;
last frame) so the same battery can be pointed at real data.

Outputs per case: <case>_signatures.png (4-panel dashboard) and a
combined summary.csv with one row per (case, mu) and one column per
metric, so "which metric separates" is a single table read.

_Args at the bottom for PyCharm; CLI mirrors it.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
#from scipy.signal import hilbert
from collections import deque

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "src", _HERE / "src"):
    if _p.exists():
        sys.path.insert(0, str(_p))

try:
    from utils.spectral import SpectralDerivs
except ImportError:
    from spectral import SpectralDerivs

try:
    from utils.geometry import build_rectangular_ramp_smooth
except ImportError:
    from geometry import build_rectangular_ramp_smooth

try:
    from op_extract.uhu_adaptive import compute_uhu_ops_adaptive
except ImportError:
    from uhu_adaptive import compute_uhu_ops_adaptive

try:
    from paper_style import use_paper_style
    use_paper_style()
except ImportError:
    pass


try:
    from utils.kfield_calcs import (orient_vector_field,
                                    safe_central_derivs,
                                    phi_jump_mask_inner)
except ImportError:
    from kfield_calcs import (orient_vector_field, safe_central_derivs,
                              phi_jump_mask_inner)


# -----------------------------------------------------------------------
# Analytic fields (same constructions as the line-energy scripts)
# -----------------------------------------------------------------------

def field_knee(mu, x, y, Lx):
    k1 = np.sqrt(1.0 - mu * mu)
    k1 = 2.0 * np.pi * max(1, round(k1 * Lx / (2.0 * np.pi))) / Lx
    k2 = np.sqrt(max(1.0 - k1 * k1, 1e-12))
    X, Y = np.meshgrid(x, y)
    theta = k1 * X + np.logaddexp(k2 * Y, -k2 * Y)
    return np.cos(theta), k1, k2


def pismen_on_grid(X, Y, x_defects, amp=1.0):
    linear_phase = Y
    x_defects = np.asarray(x_defects, dtype=float)
    npts = len(x_defects)
    if npts == 0:
        theta = linear_phase.copy()
        return theta, amp * np.cos(theta)
    thetas = np.arange(npts + 1) * np.pi
    psi = 0.5 * (1 + np.exp(-thetas[-1]))
    for j in range(npts):
        delta = 0.5 * (np.exp(-thetas[j + 1]) - np.exp(-thetas[j]))
        arg = (X - x_defects[j]) / np.sqrt(2 * np.abs(Y) + 1e-8)
        psi += delta * special.erf(arg)
    theta = linear_phase - np.sign(Y) * np.log(psi)
    return theta, amp * np.cos(theta)


def field_pismen(mu, n_defects, dx_target, Ly, Ny,
                 edge_margin_periods=2.0, stagger=True):
    k1 = np.sqrt(1.0 - mu * mu)
    L = np.pi / k1
    Lx = (n_defects - 1 + 2 * edge_margin_periods) * L
    Nx = int(np.ceil(Lx / dx_target))
    Nx += Nx % 2
    x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
    if stagger:
        y = y + 0.5 * (Ly / Ny)
    x_defects = L * (np.arange(n_defects) - (n_defects - 1) / 2.0)
    X, Y = np.meshgrid(x, y)
    _, u = pismen_on_grid(X, Y, x_defects, amp=1.0)
    return u, x, y, k1, L, x_defects, Lx


# -----------------------------------------------------------------------
# A. Envelope along the core row (raw, no averaging)
# -----------------------------------------------------------------------

# def envelope_signature(u, x, row0, x_sel, n_rows=3):
#     """Hilbert envelope along x, median over a few rows straddling the
#     core, restricted to the analysis window. Returns profile + metrics."""
#     envs = []
#     for r in range(row0 - n_rows // 2, row0 + n_rows // 2 + 1):
#         envs.append(np.abs(hilbert(u[r, :])))
#     env = np.median(np.array(envs), axis=0)[x_sel]
#     med = float(np.median(env))
#     metrics = {
#         "env_min_over_med": float(env.min() / (med + 1e-15)),
#         "env_n_deep_minima": int(np.sum(
#             (env[1:-1] < env[:-2]) & (env[1:-1] < env[2:])
#             & (env[1:-1] < 0.5 * med))),
#     }
#     return env, metrics

def riesz_envelope(u, Lx, Ly):
    """2D monogenic (Riesz) amplitude sqrt(u^2 + |Ru|^2). Unlike the 1D
    Hilbert envelope along x, this sees a dislocation's point zero even
    when the core row is locally a plateau (near-DC) in x."""
    ny, nx = u.shape
    qx = 2 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    qy = 2 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)
    QX, QY = np.meshgrid(qx, qy)
    Q = np.hypot(QX, QY)
    Q[0, 0] = 1.0
    U = np.fft.fft2(u)
    r1 = np.real(np.fft.ifft2(-1j * QX / Q * U))
    r2 = np.real(np.fft.ifft2(-1j * QY / Q * U))
    return np.sqrt(u ** 2 + r1 ** 2 + r2 ** 2)


def envelope_signature(u, ramp, x, row0, x_sel, Lx, Ly, n_rows=5):
    """Riesz amplitude, MIN over rows straddling the core (hunting point
    zeros, so min, not median), plus the crest factor of the raw core
    row (sqrt(2) for a sinusoid, -> 1 for +-plateaus)."""
    A = riesz_envelope(u * ramp, Lx, Ly)
    env = A[row0 - n_rows // 2: row0 + n_rows // 2 + 1, :].min(axis=0)[x_sel]
    med = float(np.median(env))
    row = u[row0, :][x_sel]
    crest = float(np.max(np.abs(row))
                  / (np.sqrt(np.mean(row ** 2)) + 1e-15))
    metrics = {
        "env_min_over_med": float(env.min() / (med + 1e-15)),
        "env_n_deep_minima": int(np.sum(
            (env[1:-1] < env[:-2]) & (env[1:-1] < env[2:])
            & (env[1:-1] < 0.5 * med))),
        "crest_factor": crest,
    }
    return env, metrics


# -----------------------------------------------------------------------
# B. Twist: disk-integrated J from an isotropic extraction
# -----------------------------------------------------------------------

# def twist_signature(ops, x, y, row0, x_sel, Lx, Ly, radius, stride_px=4):
#     k1f, k2f = ops["k1_orig"], ops["k2_orig"]
#     fx = SpectralDerivs(k1f, Lx, Ly, "x")
#     fy = SpectralDerivs(k1f, Lx, Ly, "y")
#     gx = SpectralDerivs(k2f, Lx, Ly, "x")
#     gy = SpectralDerivs(k2f, Lx, Ly, "y")
#     J = fx * gy - fy * gx
#
#     dx = x[1] - x[0]
#     dy = y[1] - y[0]
#     ry = int(round(radius / dy))
#     rx = int(round(radius / dx))
#     yy, xx = np.ogrid[-ry:ry + 1, -rx:rx + 1]
#     disk = (yy * dy) ** 2 + (xx * dx) ** 2 <= radius ** 2
#     dA = dx * dy
#
#     idx = np.where(x_sel)[0][::stride_px]
#     idx = idx[(idx > rx) & (idx < len(x) - rx - 1)]
#     tw = np.full(len(idx), np.nan)
#     for n, ix in enumerate(idx):
#         sl = (slice(row0 - ry, row0 + ry + 1), slice(ix - rx, ix + rx + 1))
#         patchJ = J[sl]
#         if patchJ.shape != disk.shape:
#             continue
#         tw[n] = np.nansum(patchJ[disk]) * dA
#     metrics = {
#         "twist_maxabs": float(np.nanmax(np.abs(tw))),
#         "twist_rms": float(np.sqrt(np.nanmean(tw ** 2))),
#     }
#     return x[idx], tw, metrics

def twist_signature(ops, ramp, x, y, row0, x_sel, radius, stride_px=4):
    """Disk-integrated J on the ORIENTED k-field: BFS sign propagation,
    pi-jump masking, masked central differences (the pre-treatment
    validated in pismen_fit_and_residuals_v2). Without it, the arbitrary
    eigh sign sheet across the core manufactures a spurious J ridge on
    the knee, which was the ~0.13 floor in the first run."""
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    mask = ramp >= 0.999
    f_m, g_m = orient_vector_field(ops["k1_orig"], ops["k2_orig"], mask)
    f_or, g_or = np.asarray(f_m), np.asarray(g_m)
    phi = np.arctan2(g_or, f_or)
    jump = phi_jump_mask_inner(phi, tol=np.pi / 10, inner_mask=mask)
    mask_ok = mask & (~jump)
    fx, fy = safe_central_derivs(f_or, dx, dy, mask_ok)
    gx, gy = safe_central_derivs(g_or, dx, dy, mask_ok)
    J = fx * gy - fy * gx

    ry = int(round(radius / dy))
    rx = int(round(radius / dx))
    yy, xx = np.ogrid[-ry:ry + 1, -rx:rx + 1]
    disk = (yy * dy) ** 2 + (xx * dx) ** 2 <= radius ** 2
    dA = dx * dy

    idx = np.where(x_sel)[0][::stride_px]
    idx = idx[(idx > rx) & (idx < len(x) - rx - 1)]
    tw = np.full(len(idx), np.nan)
    for n, ix in enumerate(idx):
        sl = (slice(row0 - ry, row0 + ry + 1), slice(ix - rx, ix + rx + 1))
        patchJ = J[sl]
        if patchJ.shape != disk.shape:
            continue
        tw[n] = np.nansum(patchJ[disk]) * dA
    metrics = {
        "twist_maxabs": float(np.nanmax(np.abs(tw))),
        "twist_rms": float(np.sqrt(np.nanmean(tw ** 2))),
    }
    return x[idx], tw, metrics


# -----------------------------------------------------------------------
# C. Gabor angular spectrum along the core
# -----------------------------------------------------------------------

# def gabor_angle_signature(u, x, y, row0, x_sel, win=64, stride_px=8,
#                           q_lo=0.4, q_hi=1.6, n_bins=36):
#     """Windowed 2D FFT at points along the midline; energy in the annulus
#     q_lo < |q| < q_hi is folded into director angle mod pi. Returns the
#     dominant-angle profile (deg), secondary-peak fraction, and metrics."""
#     ny, nx = u.shape
#     dx = x[1] - x[0]
#     dy = y[1] - y[0]
#     h = win // 2
#     taper = np.outer(np.hanning(win), np.hanning(win))
#     qx = 2 * np.pi * np.fft.fftfreq(win, d=dx)
#     qy = 2 * np.pi * np.fft.fftfreq(win, d=dy)
#     QX, QY = np.meshgrid(qx, qy)
#     Q = np.hypot(QX, QY)
#     ann = (Q > q_lo) & (Q < q_hi)
#     ang = np.mod(np.arctan2(QY, QX), np.pi)          # director fold
#     bins = np.linspace(0, np.pi, n_bins + 1)
#     ibin = np.clip(np.digitize(ang, bins) - 1, 0, n_bins - 1)
#
#     idx = np.where(x_sel)[0][::stride_px]
#     idx = idx[(idx > h) & (idx < nx - h - 1)]
#     if row0 - h < 0 or row0 + h > ny:
#         raise ValueError("gabor window exceeds domain in y; reduce win")
#
#     dom = np.full(len(idx), np.nan)
#     sec = np.full(len(idx), np.nan)
#     for n, ix in enumerate(idx):
#         w = u[row0 - h:row0 + h, ix - h:ix + h] * taper
#         P = np.abs(np.fft.fft2(w)) ** 2
#         E = np.bincount(ibin[ann].ravel(), weights=P[ann].ravel(),
#                         minlength=n_bins)
#         i1 = int(np.argmax(E))
#         e1 = E[i1]
#         Ez = E.copy()
#         for k in range(-3, 4):                        # zero out +-3 bins
#             Ez[(i1 + k) % n_bins] = 0.0
#         e2 = Ez.max()
#         dom[n] = np.degrees(0.5 * (bins[i1] + bins[i1 + 1]))
#         sec[n] = float(e2 / (e1 + 1e-15))
#     # angle range on the pi-circle (doubled-angle circular spread)
#     good = np.isfinite(dom)
#     z = np.exp(2j * np.radians(dom[good]))
#     circ_spread = float(np.sqrt(-2 * np.log(np.abs(np.mean(z)) + 1e-15)))
#     metrics = {
#         "gabor_angle_range_deg": float(np.nanmax(dom) - np.nanmin(dom)),
#         "gabor_circ_spread": circ_spread,
#         "gabor_secondary_mean": float(np.nanmean(sec)),
#     }
#     return x[idx], dom, sec, metrics

def gabor_angle_signature(u, x, y, row0, x_sel, win_x=96, win_y=32,
                          stride_px=8, q_lo=0.4, q_hi=1.6,
                          sector_deg=30.0):
    """Windowed 2D spectrum in a STRIP straddling the core (win_y ~ a
    few core widths, so the far field cannot dominate the window).
    Per point: dom = power-weighted doubled-angle centroid (continuous,
    no argmax ties); r_sector = energy with director angle near 90 deg
    (k across the line) over energy near 0 deg (k along the line). On
    the core line the two cases are orthogonal: knee core k = (k1, 0),
    chain gap k ~ (0, 1), so knee << 1 and chain >> 1."""
    ny, nx = u.shape
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    hx, hy = win_x // 2, win_y // 2
    taper = np.outer(np.hanning(win_y), np.hanning(win_x))
    qx = 2 * np.pi * np.fft.fftfreq(win_x, d=dx)
    qy = 2 * np.pi * np.fft.fftfreq(win_y, d=dy)
    QX, QY = np.meshgrid(qx, qy)
    Q = np.hypot(QX, QY)
    ann = (Q > q_lo) & (Q < q_hi)
    ang = np.mod(np.arctan2(QY, QX), np.pi)
    s = np.radians(sector_deg)
    sec_along = ann & (np.minimum(ang, np.pi - ang) < s)
    sec_across = ann & (np.abs(ang - np.pi / 2) < s)

    idx = np.where(x_sel)[0][::stride_px]
    idx = idx[(idx > hx) & (idx < nx - hx - 1)]
    if row0 - hy < 0 or row0 + hy > ny:
        raise ValueError("gabor strip exceeds domain in y; reduce win_y")

    dom = np.full(len(idx), np.nan)
    r_sector = np.full(len(idx), np.nan)
    for n, ix in enumerate(idx):
        w = u[row0 - hy:row0 + hy, ix - hx:ix + hx] * taper
        P = np.abs(np.fft.fft2(w)) ** 2
        zc = np.sum(P[ann] * np.exp(2j * ang[ann]))
        dom[n] = np.degrees(np.mod(0.5 * np.angle(zc), np.pi))
        r_sector[n] = float(P[sec_across].sum()
                            / (P[sec_along].sum() + 1e-15))
    good = np.isfinite(dom)
    z = np.exp(2j * np.radians(dom[good]))
    R = min(float(np.abs(np.mean(z))), 1.0)
    circ_spread = float(np.sqrt(max(-2.0 * np.log(R + 1e-15), 0.0)))
    metrics = {
        "gabor_circ_spread": circ_spread,
        "gabor_rsector_median": float(np.nanmedian(r_sector)),
        "gabor_rsector_p10": float(np.nanpercentile(r_sector, 10)),
        "gabor_rsector_swing": float(np.nanpercentile(r_sector, 90)
                                     - np.nanpercentile(r_sector, 10)),
    }
    return x[idx], dom, r_sector, metrics


# -----------------------------------------------------------------------
# D. Beading control (known degenerate; kept as a caution)
# -----------------------------------------------------------------------

def beading_signature(ops, x, row0, x_sel, k1):
    lam2 = ops["lam2"][row0, :][x_sel]
    prof = lam2 - lam2.mean()
    n = len(prof)
    dx = x[1] - x[0]
    q = 2 * np.pi * np.fft.rfftfreq(n, d=dx)
    P = np.abs(np.fft.rfft(prof * np.hanning(n))) ** 2
    band = (q > 1.6 * k1) & (q < 2.4 * k1)
    metrics = {
        "beading_frac_at_2k1": float(P[band].sum() / (P[1:].sum() + 1e-15)),
    }
    return q, P, metrics


# -----------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------

def run_case(case, mu, cfg):
    dx_target = cfg.Lx_ref / cfg.Nx_ref
    x_defects = None
    if case == "knee":
        Lx = cfg.Lx_ref
        x = np.linspace(-Lx / 2, Lx / 2, cfg.Nx_ref, endpoint=False)
        y = np.linspace(-cfg.Ly / 2, cfg.Ly / 2, cfg.Ny, endpoint=False)
        u, k1, k2 = field_knee(mu, x, y, Lx)
        L = np.pi / k1
    elif case == "pismen":
        u, x, y, k1, L, x_defects, Lx = field_pismen(
            mu, cfg.n_defects, dx_target, cfg.Ly, cfg.Ny,
            stagger=cfg.stagger)
    else:                                             # case == "sh"
        d = np.load(cfg.sh_npz)
        u = d["u"][..., -1] if d["u"].ndim == 3 else d["u"]
        x, y = d["x"], d["y"]
        Lx = x[-1] - x[0]
        k1 = np.sqrt(1.0 - mu * mu)
        L = np.pi / k1
    Ly = y[-1] - y[0] + (y[1] - y[0])

    ramp = build_rectangular_ramp_smooth(
        x, y, xmargin=cfg.xmargin, ymargin=cfg.ymargin,
        tanhscale=cfg.tanhscale)
    row0 = int(np.argmin(np.abs(y - 0.5 * (y[0] + y[-1]))))
    x_c = x - 0.5 * (x[0] + x[-1])
    x_sel = np.abs(x_c) < 0.5 * cfg.window_frac * Lx

    # isotropic extraction (B and D need it; A and C do not)
    ops = compute_uhu_ops_adaptive(
        u, x, y, cfg.sigma, SpectralDerivs, ramp,
        n_iter=1, aspect_max=1.0, sharpen=1.0, damping=0.5,
        patch=cfg.patch, orient="slowness")

    # env, m_a = envelope_signature(u, x, row0, x_sel)
    # xt, tw, m_b = twist_signature(ops, x, y, row0, x_sel, Lx, Ly,
    #                               radius=cfg.twist_radius)
    # xg, dom, sec, m_c = gabor_angle_signature(u, x, y, row0, x_sel,
    #                                           win=cfg.gabor_win)

    env, m_a = envelope_signature(u, ramp, x, row0, x_sel, Lx, Ly)
    xt, tw, m_b = twist_signature(ops, ramp, x, y, row0, x_sel,
                                  radius=cfg.twist_radius)
    xg, dom, r_sector, m_c = gabor_angle_signature(
        u, x, y, row0, x_sel, win_x=cfg.gabor_win_x,
        win_y=cfg.gabor_win_y)
    q, P, m_d = beading_signature(ops, x, row0, x_sel, k1)


    rec = {"case": case, "mu": mu, "k1": k1, "L": L}
    for m in (m_a, m_b, m_c, m_d):
        rec.update(m)

    # ---------------- dashboard ----------------
    fig, axs = plt.subplots(2, 2, figsize=(11.8, 6.2),
                            constrained_layout=True)

    ax = axs[0, 0]
    ax.plot(x[x_sel], env, "k-", lw=1.1)
    ax.set_title("A: raw Hilbert envelope on core row "
                 "(chain: zeros at cores; knee: flat)", fontsize=9)
    ax.set_xlabel("$x$"); ax.set_ylabel("envelope")

    ax = axs[0, 1]
    ax.plot(xt, tw, "o-", ms=3, lw=1.0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"B: disk-integrated twist (r={cfg.twist_radius:.2f}) "
                 "(chain: quantized bumps; knee: ~0)", fontsize=9)
    ax.set_xlabel("$x$"); ax.set_ylabel("$\\oint J\\,dA$")

    ax = axs[1, 0]
    ax.plot(xg, dom, "o-", ms=3, lw=1.0, label="dominant angle")
    ax2 = ax.twinx()
    ax2.semilogy(xg, r_sector, "s--", ms=2.5, lw=0.8, color="tab:orange")
    ax2.axhline(1.0, color="tab:orange", lw=0.5, ls=":")
    ax2.set_ylabel("across/along sector ratio", color="tab:orange")
    ax.set_title("C: Gabor dominant director angle "
                 "(chain: swings toward rotated content)", fontsize=9)
    ax.set_xlabel("$x$"); ax.set_ylabel("angle (deg)")

    ax = axs[1, 1]
    ax.semilogy(q[1:], P[1:], "k-", lw=1.0)
    ax.axvline(2 * k1, color="r", ls="--", lw=0.9, label="$2k_1$")
    ax.set_title("D: along-core $\\lambda_2$ spectrum "
                 "(CONTROL: degenerate between cases)", fontsize=9)
    ax.set_xlabel("$q$"); ax.set_ylabel("power")
    ax.legend(fontsize=8)

    if x_defects is not None:
        for panel in (axs[0, 0], axs[0, 1], axs[1, 0]):
            for xd in x_defects:
                panel.axvline(xd, color="c", lw=0.4, alpha=0.5, ls=":")

    fig.suptitle(f"{case}, mu={mu:.2f}: candidate knee-vs-dislocation "
                 "signatures (A, C kernel-free; B artifact-cancelling; "
                 "D degenerate control)", fontsize=10)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{case}_mu{mu:.2f}_signatures.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)
    return rec


def run(cfg):
    rows = []
    for case in cfg.cases:
        for mu in cfg.mus:
            print(f"[{case} mu={mu}] ...")
            rows.append(run_case(case, mu, cfg))
    keys = list(rows[0].keys())
    out = Path(cfg.out)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # console contrast table: knee vs pismen per metric, per mu
    print("\nmetric".ljust(28), "knee".rjust(12), "pismen".rjust(12))
    for mu in cfg.mus:
        kn = next((r for r in rows if r["case"] == "knee"
                   and r["mu"] == mu), None)
        pi = next((r for r in rows if r["case"] == "pismen"
                   and r["mu"] == mu), None)
        if kn is None or pi is None:
            continue
        print(f"--- mu = {mu}")
        for k in keys[4:]:
            print(f"  {k}".ljust(28),
                  f"{kn[k]:12.4f}", f"{pi[k]:12.4f}")
    print(f"\nwrote {out}/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=["knee", "pismen"],
                   choices=["knee", "pismen", "sh"])
    p.add_argument("--mus", nargs="+", type=float,
                   default=[0.85, 0.90, 0.95])
    p.add_argument("--sh_npz", default=None,
                   help="OP/state npz with u, x, y (used for case 'sh')")
    p.add_argument("--sigma", type=float, default=np.pi / 2)
    p.add_argument("--patch", type=int, default=96)
    p.add_argument("--twist_radius", type=float, default=np.pi / 2)
    p.add_argument("--gabor_win_x", type=int, default=96)
    p.add_argument("--window_frac", type=float, default=0.6)
    p.add_argument("--Lx_ref", type=float, default=125.0)
    p.add_argument("--Nx_ref", type=int, default=512)
    p.add_argument("--Ly", type=float, default=64.0)
    p.add_argument("--Ny", type=int, default=256)
    p.add_argument("--n_defects", type=int, default=9)
    p.add_argument("--stagger", action="store_true", default=True)
    p.add_argument("--xmargin", type=float, default=0.05)
    p.add_argument("--ymargin", type=float, default=0.05)
    p.add_argument("--tanhscale", type=float, default=60.0)
    p.add_argument("--gabor_win_y", type=int, default=32)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent
                                        / "results"
                                        / "pgb_signature_explorer"))
    run(p.parse_args())


class _Args:
    cases = ["knee", "pismen"]
    mus = [0.85, 0.90, 0.95]
    sh_npz = None
    sigma = np.pi / 2
    patch = 96
    twist_radius = np.pi / 2
    gabor_win_x = 96
    gabor_win_y = 32
    window_frac = 0.6
    Lx_ref = 125.0
    Nx_ref = 512
    Ly = 64.0
    Ny = 256
    n_defects = 9
    stagger = True
    xmargin = 0.05
    ymargin = 0.05
    tanhscale = 60.0
    out = str(Path(__file__).resolve().parent / "results"
              / "pgb_signature_explorer")


if __name__ == "__main__":
    RUN_IN_IDE = True
    if RUN_IN_IDE and len(sys.argv) == 1:
        run(_Args())
    else:
        main()