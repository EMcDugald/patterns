"""knee_rowavg_laws.py  v1.1  (13 Oct 2026)   workstream G21 (the kernel question) of the research thread

Row-averaged weak laws on the zigzag knee, valid before, across and after the
topological event.

Why it works. Every zigzag movie of record satisfies the shift-flip symmetry
u(x + L, y, t) = -u(x, y, t), L = pi/k1 (measured 13 Oct on the six last
frames of record and the 0.65 to 0.92 sweep: residual 4e-8, the float32
floor; a half shift misses by 1.41). Swift-Hohenberg without a quadratic term
preserves it, so it holds at every time. Every field built from u that is
unchanged under u -> -u (A, k up to orientation, w, div k, J, every library
column) is then exactly L-periodic in x, which is also why the cores sit at
exactly one per L (spacing 1.00 L at 0.83 to 0.95). For an L-periodic field
the weak integral against a test function whose x-profile is a box of length
L (smoothed) does not depend on the x-centre: it is the x-average over one
period. So the spacing-matched test function of G21 is exactly a fit in
(y, t) of x-averaged columns, and a law that holds pointwise holds for the
averages with the same coefficients (averaging is a choice of test
function, not a coarse graining). The 2k1 ripple of the estimator and every
harmonic of the core row drop out of any column in which they enter
linearly; the steady part of a product of two ripples does not (measured on
the controls below).

What it does, per movie: (1) the shift-flip residual per frame (FLAG when it
fails; the averaging is then only approximate); (2) the frozen mg estimator
per frame (wsindy_theta_t v1.9's estimate and theta_t from z, orientation as
in the driver), pointwise first derivatives of its k and A, and the
x-average under a Hann window spanning an integer number of periods inside
the untapered region (Hann of length N L has no response at any multiple of
2 pi/L for N >= 2; the leakage on the actual grid is printed); the averaged
profiles are cached (profiles_<stem>.npz); (3) one-dimensional weak columns
in (y, t) with the polynomial bump of knee_demod_laws (half-width r_y across
the row, t_half frames in time); derivatives across the row are moved onto
the bump wherever the averaged column is an exact y-derivative:
  <div k> = d_y <k_y>, <Lap div k> = d_yyy <k_y>, <Lap w> = d_yy <w>,
  <S> = d_y <w k_y>, <S_full> = d_y <k_y (w - w^3/R)>,
  <J> = -d_y <k_x d_x k_y>   (exact for any smooth L-periodic field:
                              J = d_x(k_x d_y k_y) - d_y(k_x d_x k_y)),
the rest are averages of pointwise products; (4) fits per window (whole,
pre = before the first partial circulation, event = t_circ -+ event_half,
post = from the first winding; from --events): free selection and forced
laws for theta_t (RCN, tied {S, Lap div k}, RCN + J, tied {S, J},
{S pair, J}, tied {S, Lap div k, J}, tied {S_full, Lap div k}, RCN +
w k.grad lnA), free and slaved laws for A_t; (5) identity checks per window
(<S> flux form against <w div k> + <k.grad w>; <J> flux form against the
average of pointwise J; the curl-free part -1/2 d_yy <k_x^2> against <J>);
(6) the one-sided profile: centres binned by distance d from the line, each
bin fitted with a bump of half-width min(r_max, lower bin edge) so that no
support crosses the line; the forced RCN and tied {S, B} coefficients
against d are the direct test of whether the bending changes sign away from
the line.

Controls:
  control_rcn  the RCN knee relaxation of v1.9 (mu 0.85, alpha = eta_b =
               tau = 1/3, beta0 1.8) on an exactly periodic x-grid of 8 L;
               exact route must return (2, 2, -1); the mg route measures
               what averaging does and does not remove: <J> through mg
               against 0 (the true J is 0) is the steady part of the
               estimator's ripple products.
  control_row  a translating dislocation row at spacing L with the zigzag
               far field: theta = mu y + arg(a+ e^{i k1 x} + a- e^{-i k1 x}),
               a+- = (1 +- tanh(y'/ell))/2, y' = y - v t, amplitude
               tanh(|F|/eps) (dips only at the cores, stripes end there),
               u = A cos(theta). Shift-flip symmetric, one winding per
               period, k_x flips across the row (k_y = mu on both sides).
               Planted law theta_t = -v k_y (orientation-free, so the
               director lift of mg cannot change it); the exact route and
               mg must return -v on the transport column with cores moving
               through the band. (The PN row of self_dual_discovery is not
               periodic and not on the zigzag geometry, so it is not used.)

Outputs under --outdir: per movie <stem>/report.txt, fits.json,
profiles_<stem>.npz, fig_kymo_<stem>.png, fig_onesided_<stem>.png; for the set
rowavg_table.csv/.md and fig_rowavg_vs_mu.png.

Reading the one-sided profile: use the tied {S, Lap div k} pair. Inside a
narrow bin the two stretching columns are collinear (max |r| 0.997 on the
control) and the forced RCN triple splits them arbitrarily (w div k 1.7 to
5.1 on the exact control while k.grad w and the bending stay at 2 and -1).

Changelog
  v1.1  (Ed, 13 Oct evening: why is mg's J not zero on the control knee?) For a
        curl-free field that repeats every L, <J> = -1/2 d_yy <k_x^2>, and the
        mean of k_x is pinned at k1 on every core-free row, so <J> is
        -1/2 d_yy V with V = var_x(k_x), the spread of k_x along the row.
        On the RCN control through mg <k_x> on the line is 0.5268 (k1 to four
        digits) and 99.9 percent of <J> is V of mg's 2k1 ripple (std 0.17 on
        the line, 1e-4 far away): the steady part of the ripple products.
        Filtering along x cannot remove it, because on this geometry the
        ripple and the cores' own structure both repeat every L: a local
        demodulation of k at 0.5 L gives J = 0 on the knee and 6e-6 on the
        dislocation row (exact: order 1). Fix: the matched null. Each frame's
        single-mode x-invariant part 2 Re(cbar(y) exp(i k1 x)), cbar the row
        mean of u exp(-i k1 x), goes through mg as well; V_excess = V(frame) -
        V(null) and J_excess = <J>(frame) - <J>(null) are exactly zero on an
        x-invariant single-mode frame and keep what the cores add.
        Controls: RCN knee through mg, sqrt(V_excess)/k1 on the line 2e-4
        (raw 0.40), J_excess rms 9e-11 against <J> 2.7e-4; translating row,
        mg excess against the exact along-row spread at 0.15, 0.25, 0.35, 0.5,
        0.7 L from the row: 0.44/0.55, 0.25/0.27, 0.13/0.13, 0.046/0.055,
        0.004/0.014 (80 to 95 percent from 0.15 to 0.5 L). Keeping the third
        and fifth row harmonics in the null (--null_harm 1,3,5) removes the
        cores' structure with the ripple (ratios 0.12, 0.06, 0.03): the
        cores live in those harmonics, so the default is the fundamental
        only. The exact winding field has V growing like 1/distance as a row
        approaches a core; mg's smoothing caps it, so the line maximum is an
        estimator-regularised quantity and the control comparison is made on
        rows 0.15 L and more from the cores.
        Estimator-free harmonic content: |c_3|^2/|c_1|^2 and |c_5|^2/|c_1|^2 on
        the line (row means of u exp(-i n k1 x)); 1e-14 on the RCN control,
        8.4e-2 on the dislocation-row control, and on the stand-in SH movie
        7e-6 in the far field (the stripe's own third harmonic), 4e-4 on the
        line already at t = 0.2 and 1.9e-3 at t = 25: the cubic combination
        tones of the two crossing roll sets put 3k1 content on the line long
        before any core, so on SH the matched-null excess is not zero before
        the cores (stand-in: 0.19 at t = 0.4, plateau 0.22 to 0.28), and is
        read against the events; J_excess rms on the stand-in rises fivefold
        at its winding (0.13 to 0.20 before t 2.2, 0.69 to 0.72 at t 2.4 to
        2.8). New outputs: spread_series_<stem>.csv, fig_spread_<stem>.png,
        spread_table.csv, fig_spread_vs_mu.png; J_excess columns and fits
        (forced RCN + J_excess, tied {S, J_excess}, tied {S, B, J_excess}),
        and J_excess replaces J in free selection. Symmetry check by a cubic
        spline shift on the interior (works on x-crops; the spectral shift of
        v1.0 wraps a crop: the fine-cadence files span 9 L, an odd number, so
        they cannot be a periodic shift-flip domain, and v1.0 flagged them at
        0.667). --null 0 restores v1.0's single pass.
  v1.0  first version. Sandbox validation (13 Oct):
        symmetry: residual 3.4e-8 on the six last frames of record, 2.9e-8
        on the RCN control (grid built on 8 L exactly), 3.4e-8 on the
        stand-in movie; Hann over N periods leaks 1e-7 at 2 pi n/L.
        control_rcn, exact route: forced RCN (2.008, 2.008, -1.010), tied
        {S, B} (2.002, -0.999), R2 1.0000; one-sided tied pair (2.00 to
        2.06, -0.98 to -1.00) in every bin down to d 0.25 L with r_y
        0.25 L; identities: S flux against pointwise 0.018, div k 0.001.
        control_rcn, mg route: forced RCN (1.805, 1.810, -1.612) R2 0.987,
        tied {S, B} (1.861, -1.645) R2 0.986 (the 2-D mg fit of record on
        the same control: (1.41, 1.43, -1.37) at 0.955). <J> through mg is
        not zero: rms 2.7e-4 against 1.3e-2 for <Lap div k>, and it is the
        curl-free part -1/2 d_yy <k_x^2> to 1 percent, i.e. mg's k_x deficit
        across the crease, a y-profile shaped like the bending: tied {S, J}
        fits the control at R2 0.985 with J +90, RCN + J gives J +48 and the
        bending -0.74. So on data a J coefficient is read against +48 to +90
        on this control, not against 0. One-sided through mg: bins below
        0.75 L fail (R2 < 0.6), 0.75 to 1.25 L read the tied pair (0.78,
        -2.02) and (0.61, -1.59) at R2 0.6 to 0.77, 1.25 to 2 L (0.87, -1.45)
        and (1.26, -1.09) at R2 0.90 to 0.94, and 2 to 3 L (1.27, +0.34) at
        R2 0.88: mg alone gives a positive bending in the outer bin, so a
        sign change on data is read bin by bin against these values.
        control_row (v 0.5, ell 0.3 L, eps 0.2): free selection returns
        {k_y} alone with -0.500 on both routes (R2 1.0000), with the cores
        moving through the band. The exact oriented k has <k_x> = +k1 above
        and -k1 below (one winding per period); mg reads +k1 on both sides
        and flips k_y instead: through mg a dislocated row has the knee's
        averaged orientation. On mg the J identities hold (flux against
        pointwise 0.020, curl-free part 0.043): mg's k is close to curl-free
        across a dislocation row, so its <J> carries no topological source.
        mg k on the SH last frames (single frame, no frame-to-frame lift) is
        L-periodic to 0.2 to 0.6 percent on the 0.65 knee and 3 to 6 percent
        at 0.75 to 0.85, with 0.1 to 0.3 percent of band pixels whose k
        reverses under the shift; the amplitude is periodic to 1e-3.
        Stand-in movie (not of record, whole window): the amplitude block
        returns A +0.55 and A^3 -0.83 with transport, as G17; the curl-free
        part of <J> departs from <J> (rel diff 0.92), unlike the controls.
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
from scipy.linalg import solve_banded

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT / "src", ROOT / "src" / "op_extract", ROOT / "src" / "utils", HERE):
    sys.path.insert(0, str(_p))
import wsindy_theta_t_v19 as V                                           # noqa: E402
from wsindy_theta_t_v19 import (load_npz, estimate, t1c, kernel_d3, assemble, fit_pareto,   # noqa: E402
                                forced_fit, collinearity, cstr, r_squared)
from knee_demod_laws import corr_yt                                      # noqa: E402
import matplotlib                                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SP = ["w div k", "k.grad w"]
RCN = ["w div k", "k.grad w", "Lap div k"]
PH_LIB = ["div k", "w div k", "k.grad w", "Lap div k", "w", "Lap w", "(div k)^2", "w^2 div k", "w k.grad w",
          "A_dev", "A_dev div k", "k.grad A", "w k.grad lnA", "w theta_t", "J", "chi_J"]
A_LIB = ["A", "w^2 A", "A^3", "Lap A", "k.grad A", "w A", "A div k"]
EXTRA = ["S", "S_full", "S_A", "k_y"]
TRUTH_RCN = {"w div k": 2.0, "k.grad w": 2.0, "Lap div k": -1.0}
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ---------------------------------------------------------------------
# the symmetry and the averaging window
# ---------------------------------------------------------------------

def shift_flip_residual(u2, x, Lsp):
    """|u(x + L) + u(x)| / |u| by a spectral shift along x (needs a periodic x-domain)"""
    nx = len(x); dx = float(x[1] - x[0])
    q = 2 * np.pi * np.fft.fftfreq(nx, dx)
    us = np.real(np.fft.ifft(np.fft.fft(u2, axis=1) * np.exp(1j * q * Lsp)[None, :], axis=1))
    return float(np.linalg.norm(us + u2) / max(np.linalg.norm(u2), 1e-300))


def shift_flip_residual_interior(u2, x, Lsp):
    """v1.1: |u(x + L) + u(x)| / |u| with u(x + L) from a cubic spline along x, on the columns with
    x + L inside the domain; works on x-crops (the spectral shift above wraps a crop and fails)"""
    from scipy.interpolate import CubicSpline
    keep = x + Lsp <= x[-1]
    if keep.sum() < 4:
        return np.nan
    us = CubicSpline(x, u2, axis=1)(x[keep] + Lsp)
    uu = u2[:, keep]
    return float(np.linalg.norm(us + uu) / max(np.linalg.norm(uu), 1e-300))


def xinv_part(u2, x, k1, wx, harm=(1,)):
    """v1.1: the single-mode x-invariant part of a frame, 2 Re(cbar(y) exp(i k1 x)) with
    cbar(y) = <u exp(-i k1 x)> (the row projection under the averaging window): the knee with the
    frame's own amplitude A(y) and phase phi(y); its k is (k1, phi_y), independent of x"""
    out = np.zeros_like(u2)
    for n in harm:
        cbar = (u2 * np.exp(-1j * n * k1 * x)[None, :] * wx[None, :]).sum(axis=1)
        out += np.real(2.0 * cbar[:, None] * np.exp(1j * n * k1 * x)[None, :])
    return out


def hann_window(x, Lsp, frac):
    """Hann weights over an integer number N >= 2 of periods L, centred, inside frac of the domain"""
    dx = float(x[1] - x[0]); Lx = len(x) * dx
    N = int(np.floor(frac * Lx / Lsp))
    if N < 2:
        N = 2
    half = 0.5 * N * Lsp
    xc = 0.5 * (x[0] + x[-1])
    s = (x - xc) / half
    w = np.where(np.abs(s) < 1, 0.5 * (1 + np.cos(np.pi * s)), 0.0)
    # leakage on the actual grid: response to cos(2 n pi x/L + phase), n = 1, 2, 3 (one should read ~0)
    leak = [float(abs(np.sum(w * np.exp(2j * np.pi * n * x / Lsp)) / np.sum(w))) for n in (1, 2, 3)]
    return w / w.sum(), N, leak


# ---------------------------------------------------------------------
# pointwise fields -> x-averaged profiles
# ---------------------------------------------------------------------

PROF = ["ky", "kx", "kx2", "w", "wky", "wD", "kgw", "D", "D2", "w2D", "wkgw", "w3D", "w2kgw", "sfull",
        "J", "FJ", "curl", "A", "A3", "w2A", "wA", "AD", "kgA", "wkglnA", "tht", "wtht", "A2"]


def avg_frame(kx, ky, A, tht, dx, dy, R, wx, A_floor):
    """x-averages (weights wx) of the pointwise quantities of one frame; returns {name: (ny,)}"""
    g = lambda F, ax: np.gradient(F, dx if ax == 1 else dy, axis=ax)
    w = kx ** 2 + ky ** 2 - 1.0
    kxx, kxy, kyx, kyy = g(kx, 1), g(kx, 0), g(ky, 1), g(ky, 0)
    D = kxx + kyy
    wx_, wy_ = g(w, 1), g(w, 0)
    kgw = kx * wx_ + ky * wy_
    Ax, Ay = g(A, 1), g(A, 0)
    kgA = kx * Ax + ky * Ay
    Q = dict(ky=ky, kx=kx, kx2=kx ** 2, w=w, wky=w * ky, wD=w * D, kgw=kgw, D=D, D2=D ** 2, w2D=w ** 2 * D,
             wkgw=w * kgw, w3D=w ** 3 * D, w2kgw=w ** 2 * kgw, sfull=ky * (w - w ** 3 / R),
             J=kxx * kyy - kxy * kyx, FJ=kx * kyx, curl=kyx - kxy,
             A=A, A3=A ** 3, w2A=w ** 2 * A, wA=w * A, AD=A * D, kgA=kgA,
             wkglnA=w * kgA / np.maximum(A, A_floor), tht=tht, wtht=w * tht, A2=A ** 2)
    return {k: (v * wx[None, :]).sum(axis=1) for k, v in Q.items()}


def profiles_from_fields(frames, x, y, t, R, wx, A_floor):
    """frames: iterable of (kx, ky, A, tht) per frame (2-D arrays); returns {name: (ny, nt)}"""
    dx = float(x[1] - x[0]); dy = float(y[1] - y[0])
    P = {k: np.empty((len(y), len(t))) for k in PROF}
    for i, (kx, ky, A, tht) in enumerate(frames):
        q = avg_frame(kx, ky, A, tht, dx, dy, R, wx, A_floor)
        for k in PROF:
            P[k][:, i] = q[k]
    return P


def mg_frames(u, x, y, t, args, lines, y_row, Lsp, transform=None):
    """generator of (kx, ky, A, theta_t) through the frozen mg estimator, as the v1.3 driver orients them:
    frame-to-frame director continuity inside estimate, theta_t = Im(conj z z_t)/|z|^2 (centred,
    one-sided at the ends), then the per-frame sign s = sign(mean k_x) applied to k and theta_t"""
    ny, nx, nt = u.shape
    dx = float(x[1] - x[0]); dy = float(y[1] - y[0])
    Lx, Ly = nx * dx, ny * dy
    taper = t1c.edge_taper(x, y, frac=args.taper_frac)
    dt = float(np.median(np.diff(t))) if nt > 1 else 1.0
    buf = {}
    n_prev = None

    def est(i):
        nonlocal n_prev
        ui = u[..., i].astype(float)
        if transform is not None:
            ui = transform(ui)
        e = estimate(ui, Lx, Ly, args.sigma_dir, n_prev, taper)
        n_prev = e["n"]
        return dict(k1=e["k1"], k2=e["k2"], A=e["A"], z=e["z"])

    buf[0] = est(0)
    if nt > 1:
        buf[1] = est(1)
    for i in range(nt):
        if i + 1 < nt and (i + 1) not in buf:
            buf[i + 1] = est(i + 1)
        zi = buf[i]["z"]
        if nt == 1:
            zt = np.zeros_like(zi)
        elif i == 0:
            zt = (buf[1]["z"] - zi) / dt
        elif i == nt - 1:
            zt = (zi - buf[i - 1]["z"]) / dt
        else:
            zt = (buf[i + 1]["z"] - buf[i - 1]["z"]) / (2 * dt)
        tht = np.imag(np.conj(zi) * zt) / np.maximum(np.abs(zi) ** 2, 1e-14)
        s = np.sign(np.mean(buf[i]["k1"])) or 1.0
        out = (buf[i]["k1"] * s, buf[i]["k2"] * s, buf[i]["A"], tht * s)
        buf.pop(i - 1, None)
        yield out


# ---------------------------------------------------------------------
# one-dimensional weak columns of the averaged profiles
# ---------------------------------------------------------------------

def screened_1d(Jp, dy, m):
    """(d_yy - m^2)^{-1} Jp per column, zero Dirichlet ends (Jp: (ny, nt))"""
    ny = Jp.shape[0]
    ab = np.zeros((3, ny))
    ab[0, 1:] = 1.0 / dy ** 2; ab[1, :] = -2.0 / dy ** 2 - m * m; ab[2, :-1] = 1.0 / dy ** 2
    return solve_banded((1, 1), ab, Jp)


def cols_1d(P, y, t, Lsp, R, ry_L, args, Afar):
    dy = float(y[1] - y[0]); dt = float(np.median(np.diff(t)))
    fy, fy1, fy2, fy3, my = kernel_d3(ry_L * Lsp, dy, args.poly_p)
    ht, ht1, _, _, mt = kernel_d3(args.t_half * dt, dt, args.poly_p)
    fy, fy1, fy2, fy3 = fy * dy, fy1 * dy, fy2 * dy, fy3 * dy
    ht, ht1 = ht * dt, ht1 * dt
    K = lambda F, a, b=None: corr_yt(F, a, ht if b is None else b)
    C = {}
    C["div k"] = -K(P["ky"], fy1)
    C["w div k"] = K(P["wD"], fy)
    C["k.grad w"] = K(P["kgw"], fy)
    C["Lap div k"] = -K(P["ky"], fy3)
    C["w"] = K(P["w"], fy)
    C["Lap w"] = K(P["w"], fy2)
    C["(div k)^2"] = K(P["D2"], fy)
    C["w^2 div k"] = K(P["w2D"], fy)
    C["w k.grad w"] = K(P["wkgw"], fy)
    C["A_dev"] = K(P["A"] - Afar, fy)
    C["A_dev div k"] = K(P["AD"], fy) - Afar * C["div k"]
    C["k.grad A"] = K(P["kgA"], fy)
    C["w k.grad lnA"] = K(P["wkglnA"], fy)
    C["w theta_t"] = K(P["wtht"], fy)
    C["J"] = K(P["FJ"], fy1)                                              # <J> = -d_y <k_x d_x k_y>, by parts
    C["J_pointwise"] = K(P["J"], fy)
    C["J_curlfree"] = -0.5 * K(P["kx2"], fy2)                              # -1/2 d_yy <k_x^2>
    C["chi_J"] = K(screened_1d(-(np.gradient(P["FJ"], dy, axis=0)), dy, args.chi_m), fy)
    # v1.1: the matched null. FJ0 is <k_x d_x k_y> of mg applied to the frame's own x-invariant part, so
    # J_excess is the averaged J with the estimator's ripple products removed (exactly 0 on any x-invariant frame)
    FJx = P["FJ"] - P["FJ0"] if "FJ0" in P else P["FJ"]
    C["J_excess"] = K(FJx, fy1)
    C["chi_J_excess"] = K(screened_1d(-(np.gradient(FJx, dy, axis=0)), dy, args.chi_m), fy)
    C["S"] = -K(P["wky"], fy1)                                           # <S> = d_y <w k_y>
    C["S_pointwise"] = C["w div k"] + C["k.grad w"]
    C["S_full"] = -K(P["sfull"], fy1)
    C["S_A"] = C["S"] + 2.0 * C["w k.grad lnA"]
    C["k_y"] = K(P["ky"], fy)                                            # transport column (control_row)
    C["div k_pointwise"] = K(P["D"], fy)
    T = {"theta_t": K(P["tht"], fy), "A_t": -K(P["A"], fy, ht1)}
    Ac = {"A": K(P["A"], fy), "w^2 A": K(P["w2A"], fy), "A^3": K(P["A3"], fy), "Lap A": K(P["A"], fy2),
          "k.grad A": C["k.grad A"], "w A": K(P["wA"], fy), "A div k": K(P["AD"], fy)}
    return C, Ac, T, (my, mt)


def centres(y, t, Lsp, args, m, d_lo=0.0, d_hi=None, ry_L=None):
    """query centres (iy, it): full support in y and t, |y| in [d_lo, d_hi) L; with ry_L the
    support must not cross the line (|y| >= ry_L L)"""
    my, mt = m
    d = np.abs(y) / Lsp
    d_hi = args.band_to if d_hi is None else d_hi
    iy_ok = (d >= d_lo) & (d < d_hi) & (np.arange(len(y)) >= my) & (np.arange(len(y)) < len(y) - my)
    if ry_L is not None:
        iy_ok &= np.abs(y) >= ry_L * Lsp - 1e-9
    it_ok = (np.arange(len(t)) >= mt) & (np.arange(len(t)) < len(t) - mt)
    IY, IT = np.meshgrid(np.where(iy_ok)[0], np.where(it_ok)[0], indexing="ij")
    return IY.ravel(), IT.ravel()


# ---------------------------------------------------------------------
# fits
# ---------------------------------------------------------------------

def fit_window(C, Ac, T, idx, t, rng, args, lines, truth=None, control_row=False, null_on=False):
    iy, it = idx
    cols = {k: v[iy, it] for k, v in C.items()}
    acols = {k: v[iy, it] for k, v in Ac.items()}
    b = T["theta_t"][iy, it]; bA = T["A_t"][iy, it]
    ts = t[it]
    xs = rng.random(len(b))
    fits = dict(n=int(len(b)), target_rms=float(np.sqrt(np.mean(b ** 2))) if len(b) else np.nan)
    if len(b) < 30:
        lines.append("  too few centres (%d); skipped" % len(b))
        fits["skipped"] = True
        return fits
    lib = PH_LIB + (["k_y"] if control_row else [])
    if null_on:                                                          # v1.1: free selection sees the matched-null J
        lib = [("J_excess" if k == "J" else "chi_J_excess" if k == "chi_J" else k) for k in lib]
    # columns that vanish to round-off (A_dev on a uniform amplitude, J on an x-invariant exact field) are
    # dropped from free selection: their least-squares coefficients are noise times a large number
    rms = {k: float(np.sqrt(np.mean(cols[k] ** 2))) for k in lib}
    top = max(rms.values()) if rms else 1.0
    dead = [k for k in lib if rms[k] < 1e-9 * top]
    if dead:
        lines.append("  columns at round-off, left out of free selection: %s" % ", ".join(dead))
    lib = [k for k in lib if k not in dead]
    full = lib + EXTRA + [k for k in dead if k in ("J", "J_excess", "w k.grad lnA")]
    for k in ("J", "J_excess", "chi_J", "chi_J_excess"):
        if k not in full:
            full.append(k)

    def free(names, key, tag, CC, bb):
        Th, bv, xv = assemble(CC, names, bb, xs)
        r = fit_pareto(names, Th, bv, xv, rng, args.rel_tol, args.min_contrib, abs_tol=args.abs_tol)
        if not r.get("skipped"):
            r["collinearity"] = collinearity(names, Th)
        fits[key] = r
        lines.append("  %-44s %s" % (tag + ":", V.fmt(r, truth if key == "free_phase" else None)))

    def forced(names, want, key, tag, CC, bb):
        Th, bv, xv = assemble(CC, names, bb, xs)
        r = forced_fit(names, Th, bv, xv, want)
        r["collinearity"] = collinearity(want, Th[:, [names.index(n) for n in want]])
        Th2, bv2, tv2 = assemble(CC, names, bb, ts)
        r2t = forced_fit(names, Th2, bv2, tv2, want)
        r["r2_time_split"] = r2t["r2_holdout"]
        fits[key] = r
        lines.append("  %-44s %s   holdout R2 %.4f (n %d; earlier half -> later half R2 %.3f)  [cond %.3g, max |r| %.3f]" % (
            tag + ":", cstr(r["coefs"]), r["r2_holdout"], r["n"], r["r2_time_split"], r["collinearity"]["cond"],
            abs(r["collinearity"]["max_corr"])))

    lines.append("  phase block (<theta_t>); rms target %.3e:" % fits["target_rms"])
    free(lib, "free_phase", "free selection", cols, b)
    forced(full, RCN, "forced_rcn", "forced RCN {w div k, k.grad w, Lap div k}", cols, b)
    forced(full, ["S", "Lap div k"], "tied_SB", "tied {S, Lap div k}", cols, b)
    forced(full, RCN + ["J"], "forced_rcn_J", "forced RCN + J", cols, b)
    forced(full, ["S", "J"], "tied_SJ", "tied {S, J}", cols, b)
    forced(full, SP + ["J"], "forced_SPJ", "forced {S pair, J}", cols, b)
    forced(full, ["S", "Lap div k", "J"], "tied_SBJ", "tied {S, Lap div k, J}", cols, b)
    forced(full, ["S_full", "Lap div k"], "tied_SfullB", "tied {S_full, Lap div k}", cols, b)
    forced(full, RCN + ["w k.grad lnA"], "forced_rcn_lnA", "forced RCN + w k.grad lnA", cols, b)
    if null_on:
        jx = float(np.sqrt(np.mean(cols["J_excess"] ** 2)))
        if jx > 1e-5 * max(float(np.sqrt(np.mean(cols["S"] ** 2))), 1e-300):
            forced(full, RCN + ["J_excess"], "forced_rcn_Jx", "forced RCN + J_excess (matched null)", cols, b)
            forced(full, ["S", "J_excess"], "tied_SJx", "tied {S, J_excess}", cols, b)
            forced(full, ["S", "Lap div k", "J_excess"], "tied_SBJx", "tied {S, Lap div k, J_excess}", cols, b)
        else:
            lines.append("  J_excess is zero to round-off on these centres (an x-invariant window): no J_excess fits")
    if control_row:
        forced(full, ["k_y"], "forced_transport", "forced {k_y} (planted theta_t = -v k_y)", cols, b)
    lines.append("  amplitude block (<A>_t); rms target %.3e:" % float(np.sqrt(np.mean(bA ** 2))))
    free(A_LIB, "free_amp", "free selection", acols, bA)
    forced(A_LIB, ["A", "w^2 A", "A^3"], "forced_slaved", "forced slaved {A, w^2 A, A^3}", acols, bA)
    # identities
    def rel(a, c):
        den = float(np.sqrt(np.mean(c ** 2)))
        return float(np.sqrt(np.mean((a - c) ** 2)) / den) if den > 1e-12 else np.nan
    corr = lambda a, c: float(np.corrcoef(a, c)[0, 1]) if np.std(a) > 0 and np.std(c) > 0 else np.nan
    idn = dict(S_rel=rel(cols["S_pointwise"], cols["S"]), S_corr=corr(cols["S_pointwise"], cols["S"]),
               divk_rel=rel(cols["div k_pointwise"], cols["div k"]),
               J_rel=rel(cols["J_pointwise"], cols["J"]), J_corr=corr(cols["J_pointwise"], cols["J"]),
               Jcf_rel=rel(cols["J_curlfree"], cols["J"]), Jcf_corr=corr(cols["J_curlfree"], cols["J"]),
               J_rms=float(np.sqrt(np.mean(cols["J"] ** 2))), Jx_rms=float(np.sqrt(np.mean(cols["J_excess"] ** 2))), Lapdivk_rms=float(np.sqrt(np.mean(cols["Lap div k"] ** 2))),
               S_rms=float(np.sqrt(np.mean(cols["S"] ** 2))))
    fits["identities"] = idn
    lines.append("  identities on the centres: <S> flux form against <w div k> + <k.grad w>: rel diff %.3f (corr %.4f);"
                 " <div k> by parts against the average of pointwise div k: rel diff %.3f" % (idn["S_rel"], idn["S_corr"], idn["divk_rel"]))
    lines.append("     <J> flux form against the average of pointwise J: rel diff %.3f (corr %.3f); curl-free part"
                 " -1/2 d_yy<k_x^2> against <J>: rel diff %.3f (corr %.3f); rms <J> %.3e, rms J_excess %.3e, against rms <Lap div k> %.3e and <S> %.3e" % (
                     idn["J_rel"], idn["J_corr"], idn["Jcf_rel"], idn["Jcf_corr"], idn["J_rms"], idn["Jx_rms"], idn["Lapdivk_rms"], idn["S_rms"]))
    return fits


def onesided_profile(P, y, t, Lsp, R, args, rng, lines, Afar):
    edges = [float(v) for v in args.os_edges.split(",")]
    rows = []
    lines.append("  one-sided profile (support never crosses the line; r_y = min(%.2g L, lower bin edge)); forced RCN | tied {S, Lap div k}:" % args.os_rmax)
    for d0, d1 in zip(edges[:-1], edges[1:]):
        ry = min(args.os_rmax, d0)
        if ry <= 0:
            continue
        C, Ac, T, m = cols_1d(P, y, t, Lsp, R, ry, args, Afar)
        iy, it = centres(y, t, Lsp, args, m, d_lo=d0, d_hi=d1, ry_L=ry)
        b = T["theta_t"][iy, it]
        if len(b) < 30:
            continue
        cols = {k: v[iy, it] for k, v in C.items()}
        xs = rng.random(len(b))
        out = dict(d0=d0, d1=d1, ry=ry, n=int(len(b)))
        for want, key in ((RCN, "rcn"), (["S", "Lap div k"], "SB"), (["S", "Lap div k", "J"], "SBJ")):
            Th, bv, xv = assemble(cols, PH_LIB + EXTRA, b, xs)
            r = forced_fit(PH_LIB + EXTRA, Th, bv, xv, want)
            out[key] = dict(coefs=r["coefs"], r2=r["r2_holdout"])
        rows.append(out)
        lines.append("     d %.2f..%.2f L (r_y %.2f, n %d): (%s) R2 %.3f | (%s) R2 %.3f | S,B,J (%s) R2 %.3f" % (
            d0, d1, ry, out["n"], ", ".join("%.3f" % v for v in out["rcn"]["coefs"].values()), out["rcn"]["r2"],
            ", ".join("%.3f" % v for v in out["SB"]["coefs"].values()), out["SB"]["r2"],
            ", ".join("%.3f" % v for v in out["SBJ"]["coefs"].values()), out["SBJ"]["r2"]))
    return rows


# ---------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------

def fig_kymo(P, y, t, Lsp, stem, mu, ev, path):
    band = np.abs(y) < 3 * Lsp
    yb = y[band] / Lsp
    J = -np.gradient(P["FJ"], y, axis=0)[band] * Lsp ** 2
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    for a_, F, lab, cm in ((ax[0], P["ky"][band], "<k_y>", "RdBu_r"), (ax[1], P["kx"][band], "<k_x>", "viridis"),
                           (ax[2], J, "<J> L^2 (flux form)", "RdBu_r"), (ax[3], P["A"][band], "<A>", "magma")):
        lim = np.nanpercentile(np.abs(F), 99) if cm == "RdBu_r" else None
        im = a_.pcolormesh(t, yb, F, shading="auto", cmap=cm,
                           vmin=(-lim if lim else None), vmax=(lim if lim else None))
        fig.colorbar(im, ax=a_); a_.set_title(lab, fontsize=9); a_.set_xlabel("t"); a_.set_ylabel("y / L")
        for k_, c_ in (("t_charge", C3), ("t_circ", C4), ("t_wound", "k")):
            v = ev.get(k_, np.nan)
            if np.isfinite(v):
                a_.axvline(v, color=c_, lw=0.8, ls="--")
    fig.suptitle("%s (mu %.3f): row averages over integer periods; dashed: first charge (green), partial circulation (amber), winding (black)" % (stem, mu), fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def harmonic_ratios(u, x, y, k1, Lsp, wx, args):
    """v1.1: estimator-free x-structure. Under the shift-flip symmetry u = sum over odd n of c_n(y) exp(i n k1 x);
    the row means c_n come straight from u. A single-mode x-invariant knee has c_3, c_5 at the level of the
    stripe's own harmonics (the far-field value); anything above that on the line is x-structure of u:
    the cubic combination tones of the two crossing roll sets, then the cores"""
    nt = u.shape[2]
    line = np.abs(y) < args.line_halfwidth * Lsp
    far = (np.abs(y) > 3 * Lsp) & (np.abs(y) < 5 * Lsp)
    if not far.any():
        far = np.abs(y) > 0.6 * np.abs(y).max()
    e = {n: np.exp(-1j * n * k1 * x)[None, :] * wx[None, :] for n in (1, 3, 5)}
    out = dict(P3_line=np.empty(nt), P5_line=np.empty(nt), P3_far=np.empty(nt))
    for i in range(nt):
        u2 = u[..., i].astype(float)
        c = {n: np.abs((u2 * e[n]).sum(axis=1)) ** 2 for n in (1, 3, 5)}
        r3 = c[3] / np.maximum(c[1], 1e-30); r5 = c[5] / np.maximum(c[1], 1e-30)
        out["P3_line"][i] = r3[line].max(); out["P5_line"][i] = r5[line].max(); out["P3_far"][i] = float(np.median(r3[far]))
    return out


def spread_series(P, y, t, Lsp, k1, ev, args, rep, null_on):
    """v1.1: the along-row spread of k_x. For a curl-free field that repeats every L, <J> = -1/2 d_yy V
    with V = var_x(k_x) (the mean of k_x is pinned at k1 on every core-free row); V is zero on any
    x-invariant knee and finite at cores, where the pointwise J of a winding phase is not integrable.
    s = sqrt(max(V_excess, 0))/k1, V_excess = V(frame) - V(mg on the frame's x-invariant part)"""
    line = np.abs(y) < args.line_halfwidth * Lsp
    band = np.abs(y) < args.band_to * Lsp
    dy = float(y[1] - y[0])
    sx = np.sqrt(np.maximum(P["Vx"], 0.0)) / k1
    sraw = np.sqrt(np.maximum(P["V"], 0.0)) / k1
    snull = np.sqrt(np.maximum(P["V0"], 0.0)) / k1 if "V0" in P else np.zeros_like(sraw)
    FJx = P["FJ"] - P["FJ0"] if "FJ0" in P else P["FJ"]
    Jx = -np.gradient(FJx, dy, axis=0) * Lsp ** 2
    ts = dict(s_line=sx[line].max(axis=0), s_raw_line=sraw[line].max(axis=0), s_null_line=snull[line].max(axis=0),
              Vx_band=P["Vx"][band].sum(axis=0) * dy / (Lsp * k1 ** 2), Jx_rms=np.sqrt(np.mean(Jx[band] ** 2, axis=0)),
              Vx_neg_share=np.mean(P["Vx"][band] < -1e-6 * max(np.abs(P["V"][band]).max(), 1e-300), axis=0))
    on = np.where(ts["s_line"] > args.spread_onset)[0]
    t_on = float(t[on[0]]) if len(on) else np.nan
    pick = lambda key: (float(np.interp(ev[key], t, ts["s_line"])) if np.isfinite(ev.get(key, np.nan)) else np.nan)
    rep.append("  along-row spread of k_x on the line (|y| < %.2g L), s = sqrt(V_excess)/k1%s: %.4f at t = 0, %.4f at the end; raw %.4f and %.4f;"
               " null %.4f and %.4f; first exceeds %.3g at t = %s; at first charge %.4f, partial circulation %.4f, winding %.4f;"
               " share of band rows with V_excess < 0 at the end %.2f" % (
                   args.line_halfwidth, "" if null_on else " (no null: exact fields)", ts["s_line"][0], ts["s_line"][-1],
                   ts["s_raw_line"][0], ts["s_raw_line"][-1], ts["s_null_line"][0], ts["s_null_line"][-1], args.spread_onset,
                   ("%.3f" % t_on) if np.isfinite(t_on) else "never", pick("t_charge"), pick("t_circ"), pick("t_wound"), ts["Vx_neg_share"][-1]))
    ts["t_onset"] = np.full(len(t), t_on)
    return ts


def fig_spread(P, y, t, Lsp, k1, ts, stem, mu, ev, path, null_on):
    band = np.abs(y) < 3 * Lsp
    yb = y[band] / Lsp
    dy = float(y[1] - y[0])
    sx = np.sqrt(np.maximum(P["Vx"], 0.0))[band] / k1
    FJx = P["FJ"] - P["FJ0"] if "FJ0" in P else P["FJ"]
    Jx = (-np.gradient(FJx, dy, axis=0) * Lsp ** 2)[band]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4))
    im = ax[0].pcolormesh(t, yb, sx, shading="auto", cmap="magma", vmin=0, vmax=max(np.nanpercentile(sx, 99.5), 1e-6))
    fig.colorbar(im, ax=ax[0]); ax[0].set_title("along-row spread of k_x, sqrt(V_excess)/k1" + ("" if null_on else " (exact)"), fontsize=9)
    lim = max(np.nanpercentile(np.abs(Jx), 99.5), 1e-9)
    im = ax[1].pcolormesh(t, yb, Jx, shading="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=ax[1]); ax[1].set_title("<J> excess L^2 = -1/2 d_yy V_excess L^2 (+ curl part)", fontsize=9)
    for a_ in ax[:2]:
        a_.set_xlabel("t"); a_.set_ylabel("y / L")
    ax[2].plot(t, ts["s_line"], color=C1, label="excess (matched null)")
    ax[2].plot(t, ts["s_raw_line"], color=C2, lw=0.8, label="mg on the frame")
    ax[2].plot(t, ts["s_null_line"], color="0.5", lw=0.8, ls="--", label="mg on its x-invariant part")
    ax[2].set_xlabel("t"); ax[2].set_title("max on the line of sqrt(V)/k1; dotted (right axis): |c_3/c_1|^2 on the line, no estimator", fontsize=8)
    ax[2].legend(fontsize=7, loc="upper left")
    if "P3_line" in ts:
        a2 = ax[2].twinx(); a2.semilogy(t, np.maximum(ts["P3_line"], 1e-16), ":", color=C3); a2.semilogy(t, np.maximum(ts["P3_far"], 1e-16), ":", color="0.6")
    for a_ in ax:
        for k_, c_ in (("t_charge", C3), ("t_circ", C4), ("t_wound", "k")):
            v = ev.get(k_, np.nan)
            if np.isfinite(v):
                a_.axvline(v, color=c_, lw=0.8, ls="--")
    fig.suptitle("%s (mu %.3f): dashed verticals: first charge (green), partial circulation (amber), winding (black)" % (stem, mu), fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def fig_set_spread(store, path):
    if not store:
        return
    mus = np.array([r["mu"] for r in store])
    cmap = plt.get_cmap("viridis")
    nrm = lambda m: (m - mus.min()) / max(mus.max() - mus.min(), 1e-9)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    for r in sorted(store, key=lambda r: r["mu"]):
        c_ = cmap(nrm(r["mu"]))
        ax[0].semilogy(r["t"], np.maximum(r["ts"]["s_line"], 1e-5), color=c_, lw=0.9)
        for k_, mk in (("t_charge", "o"), ("t_circ", "s"), ("t_wound", "^")):
            v = r["ev"].get(k_, np.nan)
            if np.isfinite(v):
                ax[0].semilogy([v], [max(float(np.interp(v, r["t"], r["ts"]["s_line"])), 1e-5)], mk, color=c_, ms=4)
    ax[0].set_xlabel("t"); ax[0].set_title("along-row spread of k_x on the line (excess), colour = mu;\no first charge, s partial circulation, ^ winding", fontsize=8)
    for k_, mk, lab in (("t_charge", "o", "first charge"), ("t_circ", "s", "partial circulation"), ("t_wound", "^", "winding")):
        xs_, ys_ = [], []
        for r in store:
            v = r["ev"].get(k_, np.nan)
            if np.isfinite(v):
                xs_.append(r["mu"]); ys_.append(float(np.interp(v, r["t"], r["ts"]["s_line"])))
        if xs_:
            ax[1].semilogy(xs_, np.maximum(ys_, 1e-5), mk, label=lab)
    ax[1].set_xlabel("mu"); ax[1].set_title("spread on the line at each event", fontsize=9); ax[1].legend(fontsize=7)
    xs_, ys_, ys2 = [], [], []
    for r in store:
        xs_.append(r["mu"]); ys_.append(r["ts"]["t_onset"][0]); ys2.append(r["ev"].get("t_charge", np.nan))
    ax[2].semilogy(xs_, ys_, "o", color=C1, label="spread first exceeds the onset level")
    ax[2].semilogy(xs_, ys2, "x", color=C3, label="first director charge (12 Oct record)")
    ax[2].set_xlabel("mu"); ax[2].set_ylabel("t"); ax[2].set_title("onset times", fontsize=9); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def fig_onesided(prof_by_win, stem, mu, path, truth=None):
    if not prof_by_win:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for wname, rows in prof_by_win.items():
        if not rows:
            continue
        d = [0.5 * (r["d0"] + r["d1"]) for r in rows]
        ax[0].plot(d, [r["SB"]["coefs"]["Lap div k"] for r in rows], "o-", label=wname)
        ax[1].plot(d, [r["SB"]["coefs"]["S"] for r in rows], "o-", label=wname)
        ax[2].plot(d, [r["SB"]["r2"] for r in rows], "o-", label=wname)
    ax[0].axhline(0, color="0.6", lw=0.6); ax[1].axhline(0, color="0.6", lw=0.6)
    if truth:
        ax[0].axhline(truth["Lap div k"], color="k", ls=":", lw=0.8)
    ax[0].set_title("bending (tied {S, Lap div k}) against distance from the line", fontsize=9)
    ax[1].set_title("stretching S (tied {S, Lap div k})", fontsize=9)
    ax[2].set_title("holdout R2 (tied {S, Lap div k})", fontsize=9)
    for a_ in ax:
        a_.set_xlabel("d / L (bin centre)"); a_.legend(fontsize=7)
    fig.suptitle("%s (mu %.3f): one-sided profile; supports never cross the line" % (stem, mu), fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def fig_set(table, path):
    rows = [r for r in table if np.isfinite(r.get("mu", np.nan))]
    if not rows:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for wname, c_ in (("whole", C1), ("pre", C2), ("event", C3), ("post", C4)):
        rr = sorted([r for r in rows if r["window"] == wname], key=lambda r: r["mu"])
        if not rr:
            continue
        mu = [r["mu"] for r in rr]
        ax[0].plot(mu, [r["rcn_B"] for r in rr], "o-", color=c_, label=wname)
        ax[1].plot(mu, [r["SJ_J"] for r in rr], "o-", color=c_, label=wname)
        ax[2].plot(mu, [r["rcn_r2"] for r in rr], "o-", color=c_, label=wname + " RCN")
        ax[2].plot(mu, [r["SJ_r2"] for r in rr], "s--", color=c_, label=wname + " {S, J}")
    ax[0].axhline(0, color="0.6", lw=0.6); ax[1].axhline(0, color="0.6", lw=0.6)
    ax[0].set_title("bending (forced RCN, row averages)", fontsize=9)
    ax[1].set_title("J coefficient (tied {S, J})", fontsize=9)
    ax[2].set_title("holdout R2", fontsize=9)
    for a_ in ax:
        a_.set_xlabel("mu"); a_.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ---------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------

def control_rcn(args):
    mu = args.mu; k1 = float(np.sqrt(1 - mu ** 2)); Lsp = np.pi / k1
    y, tt, p, Phi, pt = V.rcn_knee_relaxation(mu, 1 / 3, 1 / 3, 1 / 3, 0.5, 1.8, 60.0, 300, 2.0, 41)
    dy = y[1] - y[0]
    nx = int(round(8 * Lsp / dy)); dx = 8 * Lsp / nx
    x = (np.arange(nx) - nx // 2) * dx
    u = np.cos(k1 * x[None, :, None] + Phi[:, None, :]).astype(np.float32)
    Phit = np.vstack([np.zeros((1, len(tt))), np.cumsum(0.5 * (pt[1:] + pt[:-1]) * dy, axis=0)])
    ex = []
    for i in range(len(tt)):
        ex.append((np.full((len(y), nx), k1), np.repeat(p[:, i][:, None], nx, 1), np.ones((len(y), nx)),
                   np.repeat(Phit[:, i][:, None], nx, 1)))
    md = dict(mu=mu, Rscale=0.5, control="rcn", gb_row_in_crop=int(np.argmin(np.abs(y))))
    return u, x, y, tt, md, "control_rcn_mu%.2f" % mu, TRUTH_RCN, ex


def control_row(args):
    mu = args.mu; k1 = float(np.sqrt(1 - mu ** 2)); Lsp = np.pi / k1
    v, ell, eps = args.row_v, args.row_ell * Lsp, args.row_eps
    y = np.linspace(-30.0, 30.0, 300); dy = y[1] - y[0]
    nx = int(round(8 * Lsp / dy)); dx = 8 * Lsp / nx
    x = (np.arange(nx) - nx // 2) * dx
    tt = np.linspace(0.0, 2.0, 41)
    X, Y = np.meshgrid(x, y)
    u = np.empty((len(y), nx, len(tt)), np.float32)
    ex = []
    for i, ti in enumerate(tt):
        Yp = Y - v * ti
        ap, am = 0.5 * (1 + np.tanh(Yp / ell)), 0.5 * (1 - np.tanh(Yp / ell))
        dap = 0.5 / ell / np.cosh(Yp / ell) ** 2
        e1, e2 = np.exp(1j * k1 * X), np.exp(-1j * k1 * X)
        F = ap * e1 + am * e2
        Fx = 1j * k1 * (ap * e1 - am * e2)
        Fy = dap * e1 - dap * e2
        th = mu * Yp + np.angle(F)
        A = np.tanh(np.abs(F) / eps)
        u[..., i] = (A * np.cos(th)).astype(np.float32)
        kx = np.imag(Fx / F); ky = mu + np.imag(Fy / F)
        ex.append((kx, ky, A, -v * ky))
    md = dict(mu=mu, Rscale=0.5, control="row", gb_row_in_crop=int(np.argmin(np.abs(y))))
    truth = {"k_y": -v}
    return u, x, y, tt, md, "control_row_mu%.2f_v%.2f" % (mu, v), truth, ex


# ---------------------------------------------------------------------
# one movie
# ---------------------------------------------------------------------

def read_events(stem, args):
    ev = {}
    if args.events:
        try:
            for r in csv.DictReader(open(args.events)):
                if r["stem"] == stem:
                    for k in ("t_charge", "t_dips", "t_circ", "t_wound"):
                        try:
                            ev[k] = float(r[k])
                        except (KeyError, ValueError):
                            ev[k] = np.nan
        except Exception as exc:
            ev["error"] = repr(exc)
    return ev


def windows_for(t, ev, args):
    dt = float(np.median(np.diff(t)))
    W = {}
    want = [s.strip() for s in args.windows.split(",") if s.strip()]
    tc, tw = ev.get("t_circ", np.nan), ev.get("t_wound", np.nan)
    if "whole" in want:
        W["whole"] = (t[0], t[-1])
    if "pre" in want and np.isfinite(tc):
        W["pre"] = (t[0], tc - dt)
    if "event" in want and np.isfinite(tc):
        W["event"] = (max(t[0], tc - args.event_half), min(t[-1], tc + args.event_half))
    if "post" in want and np.isfinite(tw):
        W["post"] = (tw, t[-1])
    return W


def run_one(u, x, y, t, md, stem, args, rng, out, lines, truth=None, exact=None, route="mg", table=None, spread_store=None):
    mu = float(md.get("mu", args.mu)); k1 = float(np.sqrt(1 - mu ** 2)); Lsp = np.pi / k1
    R = float(md.get("Rscale", args.R))
    jr = int(md.get("gb_row_in_crop", int(np.argmin(np.abs(y)))))
    yy = y - y[jr]
    tag = stem + ("" if route == "mg" else "_" + route)
    fo = out / tag; fo.mkdir(parents=True, exist_ok=True)
    rep = ["[%s] mu %.3f, R %.3g, L %.3f, %d frames (t %.3f..%.3f, dt %.4g), route %s" % (
        stem, mu, R, Lsp, len(t), t[0], t[-1], float(np.median(np.diff(t))) if len(t) > 1 else 0.0, route)]
    # (1) the symmetry
    picks = sorted(set([0, len(t) // 2, len(t) - 1] + list(range(0, len(t), max(1, len(t) // 12)))))
    res = [shift_flip_residual_interior(u[..., i].astype(float), x, Lsp) for i in picks]
    res_sp = [shift_flip_residual(u[..., i].astype(float), x, Lsp) for i in picks[:3]]
    rep.append("  shift-flip residual |u(x+L) + u|/|u| (spline shift on the interior, v1.1) on %d frames: max %.2e (t %.3f), median %.2e%s;"
               " spectral shift (valid only on a periodic x-domain) %.2e%s" % (
        len(picks), max(res), t[picks[int(np.argmax(res))]], float(np.median(res)),
        "" if max(res) < args.sym_tol else "   FLAG: above %.0e; the row average is not exact on this movie" % args.sym_tol,
        max(res_sp), "" if max(res_sp) < args.sym_tol else " (the x-domain is not periodic: a crop, or an odd number of L)"))
    # (2) averaged profiles
    wx, N, leak = hann_window(x, Lsp, args.avg_frac)
    rep.append("  x-average: Hann over N = %d periods (%.1f of %.1f), leakage at 2 pi n/L, n = 1, 2, 3: %.1e, %.1e, %.1e" % (
        N, N * Lsp, len(x) * float(x[1] - x[0]), leak[0], leak[1], leak[2]))
    cache = fo / ("profiles_%s.npz" % tag)
    if cache.exists() and int(args.use_cache):
        d = np.load(cache)
        P = {k: d[k] for k in PROF}
        rep.append("  profiles read from the cache %s" % cache.name)
    else:
        t0_ = time.time()
        if exact is not None:
            frames = exact
        else:
            frames = mg_frames(u, x, y, t, args, rep, y[jr], Lsp)
        far = np.abs(yy) > args.far_from * Lsp
        A_floor = 0.25 * (float(np.median(np.abs(u[far, :, 0]))) * np.pi / 2 if far.any() else 0.5)
        P = profiles_from_fields(frames, x, yy, t, R, wx, A_floor)
        np.savez_compressed(cache, **P)
        rep.append("  profiles computed (%s) in %.1f s and cached" % ("exact fields" if exact is not None else "mg", time.time() - t0_))
    # v1.1: the matched null, mg applied to each frame's single-mode x-invariant part
    null_on = (exact is None) and int(args.null)
    if null_on:
        harm = tuple(int(v) for v in str(args.null_harm).split(",") if v.strip())
        cache0 = fo / ("profiles_null%s_%s.npz" % ("".join(str(h) for h in harm), tag))
        if cache0.exists() and int(args.use_cache):
            d0 = np.load(cache0); P0 = {k: d0[k] for k in PROF}
            rep.append("  null profiles read from the cache %s" % cache0.name)
        else:
            t0_ = time.time()
            frames0 = mg_frames(u, x, y, t, args, rep, y[jr], Lsp, transform=lambda F: xinv_part(F, x, k1, wx, harm))
            P0 = profiles_from_fields(frames0, x, yy, t, R, wx, 0.1)
            np.savez_compressed(cache0, **P0)
            rep.append("  null profiles (mg on each frame's x-invariant part) computed in %.1f s and cached" % (time.time() - t0_))
        P["FJ0"] = P0["FJ"]
        P["V0"] = P0["kx2"] - P0["kx"] ** 2
    P["V"] = P["kx2"] - P["kx"] ** 2
    P["Vx"] = P["V"] - P["V0"] if "V0" in P else P["V"].copy()
    # far field: beyond far_from L (or 0.3 of the height if that is shorter) and inside the untapered 0.4 of the height
    Ly = len(y) * float(y[1] - y[0]); ymid = 0.5 * (y[0] + y[-1])
    fl = min(args.far_from * Lsp, 0.3 * Ly)
    inside = np.abs(y - ymid) < 0.4 * Ly
    far = (np.abs(yy) > fl) & inside
    Afar = float(np.median(P["A"][far])) if far.any() else float(np.median(P["A"]))
    up, dn = (yy > fl) & inside, (yy < -fl) & inside
    rep.append("  far fields (%.2f L < |y| inside the untapered height), first and last frame: <k_x> above %.3f, %.3f, below %.3f, %.3f; <k_y> above %.3f, %.3f, below %.3f, %.3f (k1 %.3f, mu %.3f)" % (
        fl / Lsp, P["kx"][up, 0].mean(), P["kx"][up, -1].mean(), P["kx"][dn, 0].mean(), P["kx"][dn, -1].mean(),
        P["ky"][up, 0].mean(), P["ky"][up, -1].mean(), P["ky"][dn, 0].mean(), P["ky"][dn, -1].mean(), k1, mu))
    ev = read_events(stem, args)
    if ev:
        rep.append("  events: %s" % ", ".join("%s %.3f" % (k, v) for k, v in ev.items() if isinstance(v, float)))
    try:
        fig_kymo(P, yy, t, Lsp, tag, mu, ev, fo / ("fig_kymo_%s.png" % tag))
    except Exception as exc:
        rep.append("  kymograph failed: %r" % exc)
    hr = harmonic_ratios(u, x, yy, k1, Lsp, wx, args)
    ts = spread_series(P, yy, t, Lsp, k1, ev, args, rep, null_on)
    ts.update(hr)
    rep.append("  estimator-free harmonic content on the line, max over |y| < %.2g L of |c_3|^2/|c_1|^2 (|c_5|^2/|c_1|^2), c_n = row mean of u exp(-i n k1 x):"
               " %.2e (%.2e) at t = 0, %.2e (%.2e) at the end; far field %.2e (the stripe's own third harmonic)" % (
                   args.line_halfwidth, hr["P3_line"][0], hr["P5_line"][0], hr["P3_line"][-1], hr["P5_line"][-1], hr["P3_far"][-1]))
    ts_rows = [dict(t=float(t[i]), **{k: float(v[i]) for k, v in ts.items()}) for i in range(len(t))]
    with open(fo / ("spread_series_%s.csv" % tag), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(ts_rows[0].keys())); wr.writeheader(); wr.writerows(ts_rows)
    try:
        fig_spread(P, yy, t, Lsp, k1, ts, tag, mu, ev, fo / ("fig_spread_%s.png" % tag), null_on)
    except Exception as exc:
        rep.append("  spread figure failed: %r" % exc)
    if spread_store is not None:
        spread_store.append(dict(tag=tag, mu=mu, t=t, ts=ts, ev=ev))
    W = windows_for(t, ev, args)
    allfits, profs = {}, {}
    for wname, (ta, tb) in W.items():
        sel = (t >= ta - 1e-9) & (t <= tb + 1e-9)
        n = int(sel.sum())
        rep.append(" window %s: t %.3f..%.3f, %d frames" % (wname, ta, tb, n))
        if n < 2 * args.t_half + 3:
            rep.append("  fewer than %d frames; skipped" % (2 * args.t_half + 3)); continue
        Pw = {k: v[:, sel] for k, v in P.items() if np.ndim(v) == 2}
        tw = t[sel]
        C, Ac, T, m = cols_1d(Pw, yy, tw, Lsp, R, args.ry, args, Afar)
        idx = centres(yy, tw, Lsp, args, m)
        rep.append("  weak form: r_y %.2g L, t_half %d frames, %d centres on |y| < %.2g L" % (args.ry, args.t_half, len(idx[0]), args.band_to))
        fits = fit_window(C, Ac, T, idx, tw, rng, args, rep, truth=truth if truth and "Lap div k" in truth else None,
                          control_row=(md.get("control") == "row"), null_on=null_on)
        if int(args.onesided):
            fits["onesided"] = onesided_profile(Pw, yy, tw, Lsp, R, args, rng, rep, Afar)
            profs[wname] = fits["onesided"]
        allfits[wname] = fits
        if table is not None and not fits.get("skipped"):
            g = lambda key, nm: (fits.get(key) or {}).get("coefs", {}).get(nm, np.nan)
            r2 = lambda key: (fits.get(key) or {}).get("r2_holdout", np.nan)
            fp = fits.get("free_phase", {})
            table.append(dict(file=tag, mu=mu, window=wname, t0=float(ta), t1=float(tb), n_frames=n, n=fits["n"],
                              rcn_S1=g("forced_rcn", "w div k"), rcn_S2=g("forced_rcn", "k.grad w"), rcn_B=g("forced_rcn", "Lap div k"),
                              rcn_r2=r2("forced_rcn"), rcn_r2_time=(fits.get("forced_rcn") or {}).get("r2_time_split", np.nan),
                              SB_S=g("tied_SB", "S"), SB_B=g("tied_SB", "Lap div k"), SB_r2=r2("tied_SB"),
                              rcnJ_B=g("forced_rcn_J", "Lap div k"), rcnJ_J=g("forced_rcn_J", "J"), rcnJ_r2=r2("forced_rcn_J"),
                              SJ_S=g("tied_SJ", "S"), SJ_J=g("tied_SJ", "J"), SJ_r2=r2("tied_SJ"),
                              SBJ_B=g("tied_SBJ", "Lap div k"), SBJ_J=g("tied_SBJ", "J"), SBJ_r2=r2("tied_SBJ"),
                              free_support="{%s}" % ", ".join(fp.get("coefs", {}).keys()), free_r2=fp.get("r2_holdout", np.nan),
                              amp_A=(fits.get("free_amp") or {}).get("coefs", {}).get("A", np.nan),
                              amp_A3=(fits.get("free_amp") or {}).get("coefs", {}).get("A^3", np.nan),
                              amp_r2=(fits.get("free_amp") or {}).get("r2_holdout", np.nan),
                              J_rel=fits["identities"]["J_rel"], Jcf_rel=fits["identities"]["Jcf_rel"], S_rel=fits["identities"]["S_rel"],
                              J_rms=fits["identities"]["J_rms"], Jx_rms=fits["identities"]["Jx_rms"],
                              rcnJx_B=g("forced_rcn_Jx", "Lap div k"), rcnJx_J=g("forced_rcn_Jx", "J_excess"), rcnJx_r2=r2("forced_rcn_Jx"),
                              SJx_S=g("tied_SJx", "S"), SJx_J=g("tied_SJx", "J_excess"), SJx_r2=r2("tied_SJx"),
                              SBJx_B=g("tied_SBJx", "Lap div k"), SBJx_J=g("tied_SBJx", "J_excess"), SBJx_r2=r2("tied_SBJx")))
    fig_onesided(profs, tag, mu, fo / ("fig_onesided_%s.png" % tag), truth=truth if truth and "Lap div k" in truth else None)
    with open(fo / "fits.json", "w") as f:
        json.dump(allfits, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    (fo / "report.txt").write_text("\n".join(rep) + "\n")
    lines.extend(rep)


def main(args):
    rng = np.random.default_rng(args.seed)
    out = Path(args.outdir)
    if not out.is_absolute():
        out = HERE / out
    out.mkdir(parents=True, exist_ok=True)
    lines = ["knee_rowavg_laws v1.0; r_y %.2g L, t_half %d, poly p %d, band_to %.2g L, Hann over integer periods in %.2g of the domain;"
             " sigma_dir %.2g, taper %.2g; windows %s (event_half %.2g); one-sided %s (edges %s, r_max %.2g L)" % (
                 args.ry, args.t_half, args.poly_p, args.band_to, args.avg_frac, args.sigma_dir, args.taper_frac,
                 args.windows, args.event_half, "on" if int(args.onesided) else "off", args.os_edges, args.os_rmax)]
    table = []
    store = []
    if args.mode in ("control_rcn", "control_row"):
        u, x, y, t, md, stem, truth, ex = (control_rcn if args.mode == "control_rcn" else control_row)(args)
        args.windows = "whole"
        for route in [r.strip() for r in args.routes.split(",")]:
            run_one(u, x, y, t, md, stem, args, rng, out, lines, truth=truth, exact=(ex if route == "exact" else None),
                    route=route, table=table, spread_store=store)
        if md.get("control") == "row" and len(store) >= 2:
            # rows at fixed distance from the moving row (the exact oriented k winds, so V grows like 1/distance as a
            # row approaches a core; the comparison is made on rows that miss the cores by 0.15 L and more)
            P_e = np.load(out / (stem + "_exact") / ("profiles_%s_exact.npz" % stem))
            P_m = np.load(out / stem / ("profiles_%s.npz" % stem)); P_0 = np.load(out / stem / ("profiles_null%s_%s.npz" % ("".join(v.strip() for v in str(args.null_harm).split(",") if v.strip()), stem)))
            k1_ = float(np.sqrt(1 - args.mu ** 2)); L_ = np.pi / k1_
            for i in (0, len(t) - 1):
                yc = y - args.row_v * t[i]
                se_, sm_ = [], []
                for dd in (0.15, 0.25, 0.35, 0.5, 0.7):
                    j = int(np.argmin(np.abs(yc - dd * L_)))
                    Ve = P_e["kx2"][j, i] - P_e["kx"][j, i] ** 2
                    Vm = (P_m["kx2"][j, i] - P_m["kx"][j, i] ** 2) - (P_0["kx2"][j, i] - P_0["kx"][j, i] ** 2)
                    se_.append(np.sqrt(max(Ve, 0)) / k1_); sm_.append(np.sqrt(max(Vm, 0)) / k1_)
                lines.append("control_row, t %.2f: along-row spread sqrt(V)/k1 at 0.15, 0.25, 0.35, 0.5, 0.7 L from the row: exact %s, mg excess %s, ratio %s" % (
                    t[i], np.round(se_, 4), np.round(sm_, 4), np.round(np.array(sm_) / np.maximum(se_, 1e-12), 3)))
            # v1.1: the matched-null spread through mg against the exact along-row spread of the oriented field
            se = [r for r in store if r["tag"].endswith("_exact")]
            sm = [r for r in store if not r["tag"].endswith("_exact")]
            if se and sm:
                lines.append("control_row: spread on the line, exact against mg excess (frames 0, mid, last): %s against %s; ratio %s" % (
                    np.round(se[0]["ts"]["s_line"][[0, len(t) // 2, -1]], 4), np.round(sm[0]["ts"]["s_line"][[0, len(t) // 2, -1]], 4),
                    np.round(sm[0]["ts"]["s_line"][[0, len(t) // 2, -1]] / np.maximum(se[0]["ts"]["s_line"][[0, len(t) // 2, -1]], 1e-12), 3)))
    elif args.mode == "symmetry":
        for p in [s for s in args.npz.split(",") if s.strip()]:
            D = load_npz(p); u, x, md = D["u"], D["x"], D["md"]
            mu = float(md.get("mu", args.mu)); Lsp = np.pi / np.sqrt(1 - mu ** 2)
            res = [shift_flip_residual_interior(u[..., i].astype(float), x, Lsp) for i in range(u.shape[2])]
            lines.append("%s: mu %.3f, %d frames, shift-flip residual max %.2e median %.2e%s" % (
                Path(p).stem, mu, len(res), max(res), float(np.median(res)), "" if max(res) < args.sym_tol else "  FLAG"))
    else:
        for p in [s for s in args.npz.split(",") if s.strip()]:
            D = load_npz(p)
            run_one(D["u"], D["x"], D["y"], D["t"], D["md"], Path(p).stem, args, rng, out, lines, table=table, spread_store=store)
    if table:
        keys = list(table[0].keys())
        with open(out / "rowavg_table.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(table)
        md_ = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
        for r in table:
            md_.append("| " + " | ".join(("%.4g" % v) if isinstance(v, float) else str(v) for v in r.values()) + " |")
        (out / "rowavg_table.md").write_text("\n".join(md_) + "\n")
        fig_set(table, out / "fig_rowavg_vs_mu.png")
    if store and args.mode == "data":
        fig_set_spread(store, out / "fig_spread_vs_mu.png")
        rows = []
        for r in store:
            pk = lambda key: (float(np.interp(r["ev"][key], r["t"], r["ts"]["s_line"])) if np.isfinite(r["ev"].get(key, np.nan)) else np.nan)
            rows.append(dict(file=r["tag"], mu=r["mu"], s_line_t0=float(r["ts"]["s_line"][0]), s_line_end=float(r["ts"]["s_line"][-1]),
                             t_onset=float(r["ts"]["t_onset"][0]), t_charge=r["ev"].get("t_charge", np.nan), t_circ=r["ev"].get("t_circ", np.nan),
                             t_wound=r["ev"].get("t_wound", np.nan), s_at_charge=pk("t_charge"), s_at_circ=pk("t_circ"), s_at_wound=pk("t_wound"),
                             s_null_line_max=float(np.max(r["ts"]["s_null_line"])), Jx_rms_end=float(r["ts"]["Jx_rms"][-1]),
                             Jx_rms_at_circ=(float(np.interp(r["ev"]["t_circ"], r["t"], r["ts"]["Jx_rms"])) if np.isfinite(r["ev"].get("t_circ", np.nan)) else np.nan),
                             Jx_rms_max=float(np.max(r["ts"]["Jx_rms"])), t_Jx_max=float(r["t"][int(np.argmax(r["ts"]["Jx_rms"]))]),
                             P3_line_t0=float(r["ts"]["P3_line"][0]), P3_line_end=float(r["ts"]["P3_line"][-1]), P3_far_end=float(r["ts"]["P3_far"][-1])))
        with open(out / "spread_table.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    (out / "log.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


class _Args:
    mode = "data"             # data | control_rcn | control_row | symmetry
    npz = ""                  # one movie or a comma-separated list
    outdir = "results/knee_rowavg_laws"
    routes = "exact,mg"       # controls only
    events = ""               # events.csv of knee_amplitude_sweep (t_charge, t_circ, t_wound per stem)
    windows = "whole,pre,event,post"
    event_half = 1.0          # the event window is t_circ -+ this (time units)
    mu = 0.85                 # controls, and fallback when a file has no mu
    R = 0.5                   # fallback when a file has no Rscale
    sigma_dir = 0.4           # mg (v1.9 default)
    taper_frac = 0.06         # mg (v1.9 default)
    avg_frac = 0.8            # the Hann window spans the largest integer number of periods in this fraction of the x-domain
    sym_tol = 1e-4            # shift-flip residual above this is flagged
    use_cache = 1
    band_to = 2.0             # fitted rows: |y| < band_to L
    far_from = 4.0            # far field, L
    ry = 1.5                  # bump half-width across the row, L (the 2-D fits' width)
    t_half = 3                # bump half-width in time, frames
    poly_p = 6
    chi_m = 1.0               # screening of chi_J = (d_yy - m^2)^{-1} <J> (v1.9 units)
    onesided = 1
    os_edges = "0.25,0.5,0.75,1.0,1.25,1.5,2.0,3.0"
    os_rmax = 1.5
    row_v = 0.5               # control_row: planted speed
    row_ell = 0.3             # control_row: envelope width, L
    row_eps = 0.2             # control_row: core amplitude width (|F| units)
    null_harm = "1"           # v1.1: odd harmonics of k1 kept in the null field (1: the single-mode knee; "1,3" or "1,3,5"
                              #       also keep the row means of u exp(-i n k1 x), which carry the cores' own x-structure)
    null = 1                  # v1.1: the matched null (mg on each frame's x-invariant part); doubles the mg time
    line_halfwidth = 0.5      # v1.1: the line for the spread time series, |y| < this L
    spread_onset = 0.05       # v1.1: onset level of the along-row spread sqrt(V_excess)/k1 on the line
    rel_tol = 0.01
    abs_tol = 0.0
    min_contrib = 0.05
    seed = 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in vars(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    main(p.parse_args())
