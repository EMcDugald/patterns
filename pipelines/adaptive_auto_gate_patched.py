# pipelines/adaptive_auto_gate.py
"""
Auto-gated adaptive averaging: one prescription that yields LINE-supported
averaged energy on defect-free knees and POINT-supported averaged energy on
Pismen-Nepomnyashchy dislocation chains, with no manual legacy/core_aware
switch. The switch is replaced by a measurement.

Why a measurement is required at all: the legacy gate imposes elongation
along the core wherever coherence is high nearby, so a chain gets smeared
into a line; core_aware imposes relaxation at every low-coherence core, so
a knee keeps its beaded corrugation. Each fixed mode bakes its own answer
into both geometries. The information that distinguishes the two cases is
not in the coherence field (both cores are low-coherence lines); it is in
the ALONG-CORE waveform of the raw pattern, and it must be read off before
any anisotropic kernel is chosen, i.e. from the raw frame plus the
isotropic seed pass only. Otherwise the construction is circular.

THE GATE: local crest factor of the raw pattern on the core path.

    CF_w(x) = P99( |u| ) / rms( u )   over a sliding window of width w ~ 2 L,
              u sampled along the core path y_core(x)

  - Defect-free knee: on the core row theta = k1 x + log 2, so
    u = cos(k1 x + const) is a pure sinusoid and CF = sqrt(2) = 1.414.
  - PN chain: between defects the core-row signal saturates to +-1
    plateaus with narrow sign flips at the defects, so rms -> peak and
    CF -> 1 (measured ~1.05). SH relaxation carves amplitude dips at the
    cores but they are narrow: P99 stays ~1, rms barely moves, CF stays
    low. The discrimination is stable under relaxation.

  line score  ls(x) = clip( (CF(x) - cf_lo) / (cf_hi - cf_lo), 0, 1 )
  with (cf_lo, cf_hi) = (1.10, 1.35): knee -> 1, chain -> 0, and a knee
  destabilizing into a chain passes continuously through the gate. The
  same ls(x, t) on time-resolved data is therefore a nucleation order
  parameter: nucleation = the moment a stretch of core starts scoring
  chain-like.

KERNEL CONSTRUCTION (per Hann patch, all guidance from the seed pass):

  core patches (contain low-coherence core pixels):
      orientation = slowness director (long axis along the level sets of
                    the slow fields, i.e. along the core)
      s_par  = sigma * (1 + (aspect - 1) * ls)      elongate ONLY if line-like
      s_perp = sigma / (1 + (sharpen - 1) * ls * c)  sharpen ONLY if line-like
      chain-like core (ls -> 0) => isotropic sigma => blobs preserved.

  far-field patches (no core pixels):
      orientation = stripe director, long axis ALONG k (normal to the
      stripe crests). This is the fix for the far-field orientation wart:
      the slowness gradient is exponentially small out there, so its
      orientation is decided by noise; but the only contaminant of a
      far-field roll is the single-mode harmonic at q = 2k, which points
      along k, so the long axis along k faces it with s_par and the
      residual exp(-2 s_par^2 |k|^2) is negligible. The ellipse overlay
      should now show far-field ellipses normal to the stripes.
      s_perp = sigma: no far-field sharpening (nothing to resolve there,
      and shrinking s_perp risks re-admitting the 2k harmonic).

DIAGNOSTIC (reported, not gated on): coherent envelope depth.

    env(x) = |analytic signal| of the core-path row (Hilbert transform
    along x); depth_w(x) = min(env) / median(env) over the same window.

  This is the analytic amplitude. On the PHASE-ONLY analytic PN field
  |A| = 1 identically, so there are no zeros to see -- the discriminating
  degree of freedom at a dislocation is exactly the one a phase-only
  description deletes. After a short SH relaxation (or on real SH data)
  the amplitude physics is present and env must vanish at each core
  (the defect is a zero of the complex order parameter), while the knee
  floor stays bounded below by the two-mode beat. The script computes
  depth on the chain BEFORE and AFTER relaxation to exhibit exactly this.

RELAXATION: cfg.relax_T_pismen > 0 runs the analytic PN pattern through
ETDRK4 SH dynamics (R ramped negative outside the interior so the
non-x-periodic chain far field is absorbed in a decaying collar) before
any extraction, per-field. The knee defaults to no relaxation so it stays
the exact defect-free reference. A local relaxer is included because
core.integrate_sh's progress print references `t` before assignment.

OUTPUTS per (field, mu), under out/<field>/mu<..>/:
  dashboard.png    pattern, micro e, iso <e>, auto <e>, along-core rows,
                   cross-core ridge
  gate.png         seed coherence + core path, line-score field, CF(x) and
                   envelope depth(x), kernel ellipse overlay colored by
                   branch/ls
plus a single out/summary.csv with, per run: CF and env-depth statistics,
core-row oscillation at 2 k1 for micro/iso/auto, recovery ratio (chain) or
residual corrugation (knee), energy-per-unit-length conservation, and the
per-branch kernel bookkeeping. The acceptance test is one line:
  knee:  auto_osc small (flat line, corrugation suppressed)
  chain: auto_osc / micro_osc order 1 (blobs preserved), with identical
         config for both.

Usage
-----
python adaptive_auto_gate.py --out results/auto_gate
(or run in PyCharm; edit _Args at the bottom)
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import special
from scipy.signal import hilbert
from numpy.fft import fft2, ifft2

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "src", _HERE / "src"):
    if _p.exists():
        sys.path.insert(0, str(_p))

try:
    from utils.spectral import macro, SpectralDerivs
except ImportError:
    from spectral import macro, SpectralDerivs
try:
    from utils.geometry import build_rectangular_ramp_smooth
except ImportError:
    from geometry import build_rectangular_ramp_smooth
try:
    from utils.anisotropic_macro import (
        _hann2d, director_orientation, macro_aniso_uniform,
        coherence_from_eigs)
except ImportError:
    from anisotropic_macro import (
        _hann2d, director_orientation, macro_aniso_uniform,
        coherence_from_eigs)
try:
    from op_extract.uhu_smoother import compute_uhu_ops_smoother
except ImportError:
    from uhu_smoother import compute_uhu_ops_smoother
try:
    from op_extract.uhu_adaptive import (slowness_director,
                                         derived_kernel_params)
except ImportError:
    from uhu_adaptive import slowness_director, derived_kernel_params
try:
    from sh_sims.core import etdrk4_coeffs, dealias_mask, edensity
except ImportError:
    from core import etdrk4_coeffs, dealias_mask, edensity
try:
    from paper_style import use_paper_style
    use_paper_style()
except ImportError:
    pass


# -----------------------------------------------------------------------
# Test fields
# -----------------------------------------------------------------------

def field_knee(mu, x, y, Lx):
    """Exact Hopf-Cole knee, k1 nudged x-commensurate; u, k1, k2."""
    k1 = np.sqrt(1.0 - mu * mu)
    k1 = 2.0 * np.pi * max(1, round(k1 * Lx / (2.0 * np.pi))) / Lx
    k2 = np.sqrt(max(1.0 - k1 * k1, 1e-12))
    X, Y = np.meshgrid(x, y)
    theta = k1 * X + np.logaddexp(k2 * Y, -k2 * Y)
    return np.cos(theta), k1, k2


def pismen_on_sh_grid(X, Y, x_defects, amp=1.0):
    """Exact PN multi-dislocation pattern (as in
    pismen_line_energy_adaptive_v3.pismen_on_sh_grid)."""
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


def field_pismen(mu, n_defects, dx_target, Ly, Ny, edge_margin_periods=2.0,
                 stagger=True):
    """Finite centered PN chain sized per mu; staggered grid keeps the
    singular y=0 row off the mesh. Returns u, x, y, k1, L, x_defects,
    Lx, Nx."""
    k1 = np.sqrt(max(1.0 - mu * mu, 1e-12))
    L = np.pi / k1
    span_periods = (n_defects - 1) + 2.0 * edge_margin_periods
    Lx = span_periods * L
    Nx = int(np.ceil(Lx / dx_target))
    Nx += Nx % 2
    x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
    if stagger:
        y = y + 0.5 * (Ly / Ny)
    x_defects = L * (np.arange(n_defects) - (n_defects - 1) / 2.0)
    X, Y = np.meshgrid(x, y)
    _, u = pismen_on_sh_grid(X, Y, x_defects, amp=1.0)
    return u, x, y, k1, L, x_defects, Lx, Nx


# -----------------------------------------------------------------------
# Microscopic energy and SH relaxation
# -----------------------------------------------------------------------

def micro_energy(u, Lx, Ly, R):
    Ny, Nx = u.shape
    from numpy.fft import fftfreq
    kx = (2.0 * np.pi / Lx) * fftfreq(Nx, 1.0 / Nx)
    ky = (2.0 * np.pi / Ly) * fftfreq(Ny, 1.0 / Ny)
    xi, eta = np.meshgrid(kx, ky)
    ind = dealias_mask(Nx, Ny)
    return edensity(xi, eta, u, ind, np.full_like(u, R))


def relax_sh(u0, R_field, Lx, Ly, T, h=0.05):
    """Minimal ETDRK4 SH relaxation (local loop; core.integrate_sh's
    progress print references t before assignment)."""
    if T <= 0:
        return u0
    Ny, Nx = u0.shape
    xi, eta, E, E2, Q, f1, f2, f3 = etdrk4_coeffs(Lx, Ly, Nx, Ny, h)
    ind = dealias_mask(Nx, Ny)
    Rf = fft2(R_field); Rf[ind] = 0
    R = np.real(ifft2(Rf))
    vv = fft2(u0); vv[ind] = 0
    u = np.real(ifft2(vv))
    Q = Q.copy(); Q[ind] = 0
    n = int(round(T / h))
    for _ in range(n):
        Nv = fft2(R * u - u ** 3)
        a = E2 * vv + Q * Nv
        ua = np.real(ifft2(a))
        Na = fft2(R * ua - ua ** 3)
        b = E2 * vv + Q * Na
        ub = np.real(ifft2(b))
        Nb = fft2(R * ub - ub ** 3)
        c = E2 * a + Q * (2 * Nb - Nv)
        uc = np.real(ifft2(c))
        Nc = fft2(R * uc - uc ** 3)
        vv = E * vv + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        vv[ind] = 0
        u = np.real(ifft2(vv))
    return u


# -----------------------------------------------------------------------
# The gate: core path, local crest factor, envelope depth
# -----------------------------------------------------------------------

def core_path(coh, y, band_frac=0.35):
    """Per-column argmin of the seed coherence within |y| < band_frac*Ly:
    the core meander y_core(x) (row indices) plus that minimum value."""
    ny = coh.shape[0]
    band = np.abs(y) < band_frac * (y[-1] - y[0] + (y[1] - y[0]))
    sub = np.where(band[:, None], coh, np.inf)
    rows = np.argmin(sub, axis=0)
    vals = sub[rows, np.arange(coh.shape[1])]
    return rows, vals


def sample_along_path(field, rows):
    return field[rows, np.arange(field.shape[1])]


def windowed_crest(sig, half_w):
    """CF(x) = P99(|s|)/rms(s) over sliding window of half-width half_w
    samples. Sinusoid -> sqrt(2); saturated plateau train -> 1."""
    n = sig.size
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half_w), min(n, i + half_w + 1)
        s = sig[lo:hi]
        pk = np.percentile(np.abs(s), 99.0)
        r = np.sqrt(np.mean(s * s)) + 1e-15
        out[i] = pk / r
    return out


def windowed_env_depth(sig, half_w):
    """depth(x) = min/median of the Hilbert envelope over the window.
    Knee: floor set by the two-mode beat (~0.6). Amplitude-bearing chain:
    -> ~0 at cores. Phase-only chain: ~0.9 (no amplitude dof)."""
    env = np.abs(hilbert(sig - sig.mean()) + sig.mean())
    n = sig.size
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half_w), min(n, i + half_w + 1)
        e = env[lo:hi]
        out[i] = e.min() / (np.median(e) + 1e-15)
    return out, env


def line_score_from_crest(cf, cf_lo=1.10, cf_hi=1.35):
    return np.clip((cf - cf_lo) / (cf_hi - cf_lo), 0.0, 1.0)


# -----------------------------------------------------------------------
# Primary gate: quantized circulation of the gate-seed k around a
# period-matched window
# -----------------------------------------------------------------------

def loop_circulation(k1f, k2f, x, y, ic, r0, rxh, ry):
    """Circulation of the (sign-lift-propagated) seed wavevector around
    the rectangle [ic-rxh, ic+rxh] x [r0-ry, r0+ry] (index units).
    Chain window enclosing one dislocation -> +-2 pi (quantized);
    knee window -> 0 with the corrugation cancelling identically because
    the x-extent is exactly one artifact period L and the top/bottom
    edges are periodic images traversed oppositely (Stokes replacement
    for the area integral of J, which is poisoned by eigen-branch
    snapping exactly on the core row)."""
    xs_b = np.arange(ic - rxh, ic + rxh + 1)
    ys_b = np.full_like(xs_b, r0 - ry)
    xs_r = np.full(2 * ry, ic + rxh)
    ys_r = np.arange(r0 - ry + 1, r0 + ry + 1)
    xs_t = np.arange(ic + rxh - 1, ic - rxh - 1, -1)
    ys_t = np.full_like(xs_t, r0 + ry)
    xs_l = np.full(2 * ry - 1, ic - rxh)
    ys_l = np.arange(r0 + ry - 1, r0 - ry, -1)
    ii = np.concatenate([xs_b, xs_r, xs_t, xs_l])
    jj = np.concatenate([ys_b, ys_r, ys_t, ys_l])
    kx = k1f[jj, ii].copy()
    ky = k2f[jj, ii].copy()
    for n in range(1, kx.size):        # 1D hemisphere-lift propagation
        if kx[n] * kx[n - 1] + ky[n] * ky[n - 1] < 0:
            kx[n] = -kx[n]; ky[n] = -ky[n]
    px, py = x[ii], y[jj]
    dxs = np.diff(np.append(px, px[0]))
    dys = np.diff(np.append(py, py[0]))
    kxm = 0.5 * (kx + np.roll(kx, -1))
    kym = 0.5 * (ky + np.roll(ky, -1))
    return float(np.sum(kxm * dxs + kym * dys))


def circulation_profile(k1f, k2f, x, y, rows_s, L, med_filt=True):
    """|C(x)|/2pi for the L-matched window sliding along the core path.
    An isolated bad lift (a window whose side edge crosses the core at a
    snapped pixel) contaminates ~one window, so a median filter of
    half-width L/2 along x removes those spikes on a knee (true value 0)
    without touching a chain (true value 1 everywhere)."""
    dx, dy = x[1] - x[0], y[1] - y[0]
    rxh = int(round(0.5 * L / dx))
    ry = int(round(0.5 * L / dy))
    n = x.size
    C = np.zeros(n)
    for i in range(rxh + 1, n - rxh - 1):
        r0 = int(np.clip(rows_s[i], ry + 1, k1f.shape[0] - ry - 2))
        C[i] = abs(loop_circulation(k1f, k2f, x, y, i, r0, rxh, ry)
                   ) / (2.0 * np.pi)
    if med_filt:
        from scipy.ndimage import median_filter
        C = median_filter(C, size=2 * rxh + 1, mode="nearest")
    return C


def line_score_from_circulation(C, c_lo=0.25, c_hi=0.5):
    """|C| below a quarter quantum -> line (knee); above half -> point
    (chain). The threshold is set by the quantization, not tuned."""
    return 1.0 - np.clip((C - c_lo) / (c_hi - c_lo), 0.0, 1.0)


def riesz_gate_field(u, Lx, Ly, sigma_dir, macro_fn):
    """Flip-free gate vector DIRECTION from the raw pattern: Riesz
    components -> doubled-angle components smoothed at sigma_dir ->
    director. Unlike the tensor eigenvector, the doubled-angle average
    across a crease interpolates to the bisector instead of snapping
    between branches, so there is no eigen-flip. Valid while the defect
    spacing satisfies L >~ 8.5 sigma_dir; in tighter geometries the
    smoothing washes out the near-line tilt structure (measured on the
    mu=0.85 chain), which is why the tensor gate covers that regime."""
    from numpy.fft import fftfreq
    ny, nx = u.shape
    kx = fftfreq(nx, Lx / nx); ky = fftfreq(ny, Ly / ny)
    KX, KY = np.meshgrid(kx, ky)
    Kn = np.sqrt(KX ** 2 + KY ** 2) + 1e-12
    U = fft2(u)
    R1 = np.real(ifft2(-1j * (KX / Kn) * U))
    R2 = np.real(ifft2(-1j * (KY / Kn) * U))
    s1 = macro_fn(R1 * R1 - R2 * R2, sigma_dir, Lx, Ly)
    s2 = macro_fn(2.0 * R1 * R2, sigma_dir, Lx, Ly)
    phi = 0.5 * np.arctan2(s2, s1)
    return np.cos(phi), np.sin(phi)


# -----------------------------------------------------------------------
# Auto-gated patchwise anisotropic smoother
# -----------------------------------------------------------------------

def macro_aniso_field_auto(arr, d1, d2, k1s, k2s, coh, core_mask, ls2d,
                           Lx, Ly, sigma, aspect_max, sharpen,
                           patch=96, stride=None, record=None,
                           ramp_mask=None, sharpen_weight="coh"):
    """
    Patchwise Hann overlap-add smoothing of `arr` with the branch logic in
    the module docstring. All guidance fields come from the seed pass:

      d1, d2      slowness director (orientation for core patches)
      k1s, k2s    seed wavevector field (orientation for far patches)
      coh         seed coherence
      core_mask   bool, pixels on/near a core
      ls2d        line score in [0,1] on the core band (0 elsewhere)

    record, if a list, receives per-patch dicts (center, branch, theta,
    s_par, s_perp, ls) for the ellipse overlay and the summary.
    """
    ny, nx = arr.shape
    stride = stride or patch // 2
    win = _hann2d(patch, patch)
    out = np.zeros_like(arr, dtype=float)
    wsum = np.zeros_like(arr, dtype=float)

    ys = list(range(0, ny - patch + 1, stride)) or [0]
    xs = list(range(0, nx - patch + 1, stride)) or [0]
    if ys[-1] != ny - patch:
        ys.append(max(ny - patch, 0))
    if xs[-1] != nx - patch:
        xs.append(max(nx - patch, 0))

    for y0 in ys:
        for x0 in xs:
            sl = (slice(y0, y0 + patch), slice(x0, x0 + patch))
            if arr[sl].shape != (patch, patch):
                continue
            cm = core_mask[sl]
            if ramp_mask is not None and (~ramp_mask[sl]).mean() > 0.25:
                branch = "edge"      # touches the ramp collar: isotropic,
                theta_long = 0.0     # so the collar's energy step is not
                ls = 0.0             # smeared into the interior
                s_par = sigma
                s_perp = sigma
            elif cm.any():
                branch = "core"
                # orientation: slowness director, weighted by its magnitude
                w = np.hypot(d1[sl], d2[sl])
                th_slow = director_orientation(d1[sl], d2[sl], weight=w)
                theta_long = th_slow + np.pi / 2.0
                ls = float(np.median(ls2d[sl][cm]))
                c_loc = float(np.median(coh[sl]))
                s_par = sigma * (1.0 + (aspect_max - 1.0) * ls)
                # sharpen_weight="coh": legacy, coherence must co-sign
                #   the sharpening (safe, but coherence is low exactly on
                #   the core, so the kernel relaxes toward isotropic
                #   where the quasi-1D window is most needed).
                # sharpen_weight="ls": the topological gate alone
                #   authorizes sharpening; on a certified line-like core
                #   the orientation comes from the slowness director,
                #   not the (unreliable) tensor eigenvector, so the
                #   coherence veto is redundant there.
                w_sh = c_loc if sharpen_weight == "coh" else 1.0
                s_perp = sigma / (1.0 + (sharpen - 1.0) * ls * w_sh)
            else:
                branch = "far"
                # orientation: long axis ALONG k (normal to stripes)
                w = np.hypot(k1s[sl], k2s[sl])
                theta_long = director_orientation(k1s[sl], k2s[sl], weight=w)
                ls = 0.0
                s_par = sigma * aspect_max
                s_perp = sigma
            smoothed = macro_aniso_uniform(
                arr[sl], theta_long, s_par, s_perp,
                Lx * patch / nx, Ly * patch / ny)
            out[sl] += smoothed * win
            wsum[sl] += win
            if record is not None:
                record.append(dict(
                    x0=x0, y0=y0, cx=x0 + patch / 2, cy=y0 + patch / 2,
                    branch=branch, theta=float(theta_long),
                    s_par=float(s_par), s_perp=float(s_perp), ls=ls))

    wsum = np.where(wsum < 1e-12, 1.0, wsum)
    return out / wsum


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def core_row_oscillation(prof, q_target, dx):
    p = prof - prof.mean()
    F = np.abs(np.fft.rfft(p))
    q = 2.0 * np.pi * np.fft.rfftfreq(p.size, d=dx)
    i0 = int(np.argmin(np.abs(q - q_target)))
    lo, hi = max(i0 - 1, 1), min(i0 + 2, len(F))
    i_pk = lo + int(np.argmax(F[lo:hi]))
    amp = 2.0 * F[i_pk] / p.size
    rel = amp / (np.abs(prof).mean() + 1e-15)
    wav = 2.0 * np.pi / q[i_pk] if q[i_pk] > 0 else np.inf
    return float(rel), float(wav)


# -----------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------

def run_case(cfg, field_name, mu, out_root, rows):
    sigma = cfg.sigma
    if field_name == "knee":
        x = np.linspace(-cfg.Lx / 2, cfg.Lx / 2, cfg.Nx, endpoint=False)
        y = np.linspace(-cfg.Ly / 2, cfg.Ly / 2, cfg.Ny, endpoint=False)
        u, k1, k2 = field_knee(mu, x, y, cfg.Lx)
        Lx, Nx = cfg.Lx, cfg.Nx
        L = np.pi / k1
        x_defects = np.array([])
        relax_T = cfg.relax_T_knee
    else:
        dx_target = cfg.Lx / cfg.Nx
        u, x, y, k1, L, x_defects, Lx, Nx = field_pismen(
            mu, cfg.n_defects, dx_target, cfg.Ly, cfg.Ny,
            edge_margin_periods=cfg.edge_margin_periods)
        k2 = mu
        relax_T = cfg.relax_T_pismen
    dx = Lx / Nx
    mu_eff = k2

    out_dir = out_root / field_name / f"mu{mu:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ramp = build_rectangular_ramp_smooth(
        x, y, xmargin=cfg.xmargin, ymargin=cfg.ymargin,
        tanhscale=cfg.tanhscale)
    interior = ramp >= 0.99
    half_w = max(4, int(round(cfg.gate_window_L * L / dx / 2)))

    print(f"\n=== {field_name} mu={mu} (mu_eff={mu_eff:.3f}, k1={k1:.4f}, "
          f"L={L:.2f}, Nx={Nx}) relax_T={relax_T} ===")

    # envelope-depth BEFORE relaxation (phase-only reference on the chain)
    row0 = int(np.argmin(np.abs(y)))
    dep_pre, _ = windowed_env_depth(u[row0], half_w)
    dep_pre_med = float(np.median(dep_pre[np.abs(x) < 0.3 * Lx]))

    # optional SH relaxation with a decaying collar outside the interior
    if relax_T > 0:
        R_field = -cfg.R_out + (cfg.R + cfg.R_out) * ramp
        u = relax_sh(u * ramp, R_field, Lx, cfg.Ly, relax_T, h=cfg.relax_h)

    # --- microscopic energy, far-field baseline removed ---
    e_micro = micro_energy(u * ramp, Lx, cfg.Ly, cfg.R)
    far = interior & (np.abs(np.meshgrid(x, y)[1]) > 0.25 * cfg.Ly)
    e_micro = e_micro - np.median(e_micro[far])

    # --- isotropic seed pass: ALL kernel guidance comes from here ---
    iso_sm = lambda a: macro(a, sigma, Lx, cfg.Ly)
    seed = compute_uhu_ops_smoother(u, x, y, iso_sm, SpectralDerivs, ramp)
    coh = coherence_from_eigs(seed["lam1"], seed["lam2"])
    d1, d2 = slowness_director(seed["k1_orig"], seed["k2_orig"], sigma,
                               Lx, cfg.Ly, SpectralDerivs, macro)

    # --- core path + gate signals on the RAW pattern ---
    rows_c, coh_on_core = core_path(coh, y, band_frac=0.35)
    # smooth the path (it should be slowly meandering, not pixel noise)
    rows_s = np.round(np.convolve(
        rows_c, np.ones(2 * half_w + 1) / (2 * half_w + 1),
        mode="same")).astype(int)
    sig = sample_along_path(u, rows_s)
    cf = windowed_crest(sig, half_w)
    dep, env = windowed_env_depth(sig, half_w)
    ls_cf = line_score_from_crest(cf, cfg.cf_lo, cfg.cf_hi)

    # PRIMARY GATE: quantized circulation from a dedicated narrow-sigma
    # gate seed. The gate sigma is deliberately smaller than the
    # extraction sigma: the eigen-flip threshold mu*(sigma) rises as
    # sigma shrinks, keeping the on-core eigenvector unflipped so the
    # hemisphere lift propagates correctly around the loop; the larger
    # corrugation this narrow seed carries is cancelled identically by
    # the period-matched window. Validated at mu <= 0.90; at mu = 0.95
    # the flip survives any usable sigma on phase-only synthetic fields
    # (documented limitation; on amplitude-bearing data the core zeros
    # anchor the lift and this should be re-tested).
    # Two estimators of the same quantized charge with complementary,
    # a priori validity regions (selection is automatic in k1, not a
    # manual switch; at mu = 0.9 both are valid and agree):
    #   tensor gate (narrow-sigma seed): valid until its seed eigen-flips
    #     at sharp knees; fails only at small k1.
    #   Riesz-director gate (|k_seed| times flip-free Riesz director):
    #     valid while L >~ 8.5 sigma_dir; fails only at large k1
    #     (tight chains), where the tensor gate is safely unflipped.
    use_riesz = L > cfg.L_over_sigdir_min * cfg.sigma_dir
    if use_riesz:
        # Direction: flip-free Riesz director. Magnitude: the seed |k|
        # is corrupted near the core above mu*, and that corruption
        # injects spurious winding into the constructed gate field; use
        # instead the tensor projected onto the director,
        # q^2 = g.T (-T) g / <u^2>, which involves no eigen decision
        # and is exact on the ideal knee, so the gate field carries the
        # physical topology (zero circulation on a knee, one quantum
        # per dislocation).
        g1, g2 = riesz_gate_field(u, Lx, cfg.Ly, cfg.sigma_dir, macro)
        ur = u * ramp
        Txx_g = iso_sm(u * SpectralDerivs(ur, Lx, cfg.Ly, "xx"))
        Txy_g = iso_sm(u * SpectralDerivs(ur, Lx, cfg.Ly, "xy"))
        Tyy_g = iso_sm(u * SpectralDerivs(ur, Lx, cfg.Ly, "yy"))
        uu_g = iso_sm(u * u * ramp)
        uu_g = np.where(np.abs(uu_g) < 1e-12, 1.0, uu_g)
        q_g = np.sqrt(np.maximum(
            -(g1 * g1 * Txx_g + 2 * g1 * g2 * Txy_g
              + g2 * g2 * Tyy_g) / uu_g, 0.0))
        C = circulation_profile(q_g * g1, q_g * g2, x, y, rows_s, L)
        gate_used = "riesz_director_proj"
    else:
        gate_sm = lambda a: macro(a, cfg.sigma_gate, Lx, cfg.Ly)
        gseed = compute_uhu_ops_smoother(u, x, y, gate_sm,
                                         SpectralDerivs, ramp)
        C = circulation_profile(gseed["k1_orig"], gseed["k2_orig"], x, y,
                                rows_s, L)
        gate_used = "tensor"
    ls = line_score_from_circulation(C, cfg.circ_lo, cfg.circ_hi)

    # core mask: coherence below an adaptive threshold near the path
    cmin = float(coh[interior].min())
    cmed = float(np.median(coh[interior]))
    cth = cmin + cfg.core_coh_frac * (cmed - cmin)
    Y = np.meshgrid(x, y)[1]
    y_core = y[rows_s][None, :]
    band = np.abs(Y - y_core) < cfg.core_band_sigma * sigma
    core_mask = (coh < cth) & band
    ls2d = np.where(band, ls[None, :], 0.0)

    # --- kernel params from the leakage formula, then the two averages ---
    aspect, sharpen = derived_kernel_params(k1, mu_eff, sigma,
                                            delta=cfg.leak_delta)
    print(f"  kernel: aspect_max={aspect:.2f} sharpen={sharpen:.2f}  "
          f"|C| med={np.median(C[np.abs(x) < 0.3 * Lx]):.3f}  "
          f"ls med={np.median(ls[np.abs(x) < 0.3 * Lx]):.3f}  "
          f"CF med={np.median(cf[np.abs(x) < 0.3 * Lx]):.3f}  "
          f"env depth med pre/post = {dep_pre_med:.2f}/"
          f"{np.median(dep[np.abs(x) < 0.3 * Lx]):.2f}")
    e_iso = macro(e_micro, sigma, Lx, cfg.Ly)
    patches = []
    e_auto = macro_aniso_field_auto(
        e_micro, d1, d2, seed["k1_orig"], seed["k2_orig"], coh,
        core_mask, ls2d, Lx, cfg.Ly, sigma, aspect, sharpen,
        patch=cfg.patch, record=patches, ramp_mask=interior)

    # --- metrics on the central window ---
    x_sel = np.abs(x) < (0.5 * cfg.central_periods * L
                         if field_name == "pismen" else 0.35 * Lx)
    rec = dict(field=field_name, mu=mu, mu_eff=mu_eff, k1=k1, L=L,
               aspect=aspect, sharpen=sharpen, relax_T=relax_T,
               gate_used=gate_used,
               cf_med=float(np.median(cf[x_sel])),
               circ_med=float(np.median(C[x_sel])),
               circ_p95=float(np.percentile(C[x_sel], 95)),
               ls_med=float(np.median(ls[x_sel])),
               ls_cf_med=float(np.median(ls_cf[x_sel])),
               env_depth_pre=dep_pre_med,
               env_depth_post=float(np.median(dep[x_sel])),
               n_core_patches=sum(p["branch"] == "core" for p in patches),
               n_far_patches=sum(p["branch"] == "far" for p in patches))
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    for name, f in (("micro", e_micro), ("iso", e_iso), ("auto", e_auto)):
        prof = sample_along_path(f, rows_s)[x_sel]
        rel, wav = core_row_oscillation(prof, 2.0 * k1, dx)
        rec[f"{name}_osc"] = rel
        rec[f"{name}_wav"] = wav
        rec[f"{name}_E_per_len"] = float(_trapz(f[:, x_sel].mean(axis=1), y))
    rec["auto_recovery"] = rec["auto_osc"] / (rec["micro_osc"] + 1e-15)
    rec["iso_recovery"] = rec["iso_osc"] / (rec["micro_osc"] + 1e-15)
    rec["cons_err_auto"] = (rec["auto_E_per_len"] / rec["micro_E_per_len"]
                            - 1.0)
    rows.append(rec)
    verdict = ("corrugation residual (want SMALL)" if field_name == "knee"
               else "defect-signal recovery (want ORDER 1)")
    key = (rec["auto_osc"] if field_name == "knee"
           else rec["auto_recovery"])
    print(f"  osc@2k1: micro {rec['micro_osc']:.3f}  iso "
          f"{rec['iso_osc']:.3f}  auto {rec['auto_osc']:.3f}   "
          f"{verdict}: {key:.3f}")
    print(f"  E/len: micro {rec['micro_E_per_len']:.4f} iso "
          f"{rec['iso_E_per_len']:.4f} auto {rec['auto_E_per_len']:.4f} "
          f"(auto cons err {100 * rec['cons_err_auto']:+.1f}%)")

    # --- dashboard ---
    from scipy.ndimage import binary_erosion
    n_er = max(1, int(round(2.0 * sigma / (cfg.Ly / cfg.Ny))))
    disp = binary_erosion(interior, iterations=n_er)   # display-only mask
    extent = [x[0], x[-1], y[0], y[-1]]
    fig, axs = plt.subplots(2, 3, figsize=(13.6, 6.4),
                            constrained_layout=True)
    vmax = np.percentile(e_iso[interior], 99.5)
    for ax, f, title, kw in (
        (axs[0, 0], np.ma.masked_where(~interior, u), "pattern"
         + (" (relaxed)" if relax_T > 0 else ""), dict(cmap="gray")),
        (axs[0, 1], np.ma.masked_where(~interior, e_micro),
         "micro energy $e$", dict(cmap="magma")),
        (axs[0, 2], np.ma.masked_where(~disp, e_iso),
         "$\\langle e\\rangle$ isotropic", dict(cmap="magma", vmin=0,
                                                vmax=vmax)),
        (axs[1, 0], np.ma.masked_where(~disp, e_auto),
         "$\\langle e\\rangle$ auto-gated", dict(cmap="magma", vmin=0,
                                                 vmax=vmax)),
    ):
        im = ax.imshow(f, extent=extent, origin="lower", aspect="auto",
                       rasterized=True, **kw)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axs[1, 1]
    ax.plot(x[x_sel], sample_along_path(e_iso, rows_s)[x_sel], "r-",
            lw=1.0, label="iso")
    ax.plot(x[x_sel], sample_along_path(e_auto, rows_s)[x_sel], "b-",
            lw=1.0, label="auto")
    marks = (x_defects if field_name == "pismen"
             else np.arange(-10, 11) * L)
    for xm in marks:
        if x[x_sel].min() < xm < x[x_sel].max():
            ax.axvline(xm, color="k", lw=0.4, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlabel("$x$")
    ax.set_title("along the core path"
                 + ("; gridlines = true defects" if field_name == "pismen"
                    else "; gridlines at multiples of $L$"), fontsize=9)

    ax = axs[1, 2]
    ax.plot(y, e_micro[:, x_sel].mean(axis=1), "k-", lw=1.3,
            label="micro x-avg")
    ax.plot(y, e_iso[:, x_sel].mean(axis=1), "r--", lw=0.9, label="iso")
    ax.plot(y, e_auto[:, x_sel].mean(axis=1), "b-", lw=0.9, label="auto")
    ax.set_xlabel("$y$"); ax.legend(fontsize=8)
    ax.set_title("cross-core ridge (conservation)", fontsize=9)
    fig.suptitle(
        f"{field_name} mu={mu:.2f}: auto gate (defect index |C| med "
        f"{rec['circ_med']:.2f} -> line score {rec['ls_med']:.2f}, "
        f"gate={gate_used}); same config for knee and chain, no manual "
        "switch.", fontsize=9)
    fig.savefig(out_dir / "dashboard.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # --- coarse 2D defect-index map (same loop integral on a grid) ---
    if gate_used == "riesz_director_proj":
        kgx, kgy = q_g * g1, q_g * g2
    else:
        kgx, kgy = gseed["k1_orig"], gseed["k2_orig"]
    dxg, dyg = Lx / Nx, cfg.Ly / cfg.Ny
    rxh = int(round(0.5 * L / dxg))
    ryh = int(round(0.5 * L / dyg))
    sx = 3                       # dense in x so the median can outvote
    sy = max(1, cfg.Ny // 48)    # clustered bad windows
    clear_px = 3
    yc_idx = np.arange(ryh + 1, cfg.Ny - ryh - 1, sy)
    xc_idx = np.arange(rxh + 1, Nx - rxh - 1, sx)
    Cmap = np.full((yc_idx.size, xc_idx.size), np.nan)
    for a, jc in enumerate(yc_idx):
        for b, icx in enumerate(xc_idx):
            seg = rows_s[max(icx - rxh, 0):icx + rxh + 1]
            if min(np.min(np.abs((jc + ryh) - seg)),
                   np.min(np.abs((jc - ryh) - seg))) <= clear_px:
                continue         # edge would ride the core: masked
            Cmap[a, b] = abs(loop_circulation(
                kgx, kgy, x, y, icx, jc, rxh, ryh)) / (2.0 * np.pi)
    half = max(1, int(round(rxh / sx)))
    _pad = np.pad(Cmap, ((0, 0), (half, half)), mode="edge")
    _sw = np.lib.stride_tricks.sliding_window_view(_pad, 2 * half + 1,
                                                   axis=1)
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        Cmap = np.nanmedian(_sw, axis=-1)

    # --- gate diagnostics + ellipse overlay ---
    fig, axs = plt.subplots(2, 3, figsize=(18.6, 7.2),
                            constrained_layout=True)
    ax = axs[0, 0]
    im = ax.imshow(coh, extent=extent, origin="lower", aspect="auto",
                   cmap="viridis", vmin=0, vmax=1)
    ax.plot(x, y[rows_s], "r-", lw=0.7)
    ax.set_title("seed coherence + detected core path", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axs[0, 2]
    ax.imshow(np.ma.masked_where(~interior, u), extent=extent,
              origin="lower", aspect="auto", cmap="gray")
    ext_c = [x[xc_idx[0]], x[xc_idx[-1]], y[yc_idx[0]], y[yc_idx[-1]]]
    imc = ax.imshow(Cmap, extent=ext_c, origin="lower", aspect="auto",
                    cmap="inferno", vmin=0, vmax=1.1, alpha=0.45)
    ax.set_title("defect index $|C|/2\\pi$ over the pattern "
                 "(window-centered, coarse grid)", fontsize=9)
    fig.colorbar(imc, ax=ax, shrink=0.85)

    ax = axs[0, 1]
    ax.plot(x, C, "c-", lw=1.4, label="|C|/2$\\pi$ (gate)")
    ax.plot(x, cf, "k-", lw=1.0, label="crest factor")
    ax.axhline(np.sqrt(2), color="b", lw=0.6, ls="--",
               label="$\\sqrt{2}$ (sinusoid/knee)")
    ax.axhline(1.0, color="g", lw=0.6, ls="--", label="1 (plateaus/chain)")
    ax.axhline(cfg.cf_lo, color="0.6", lw=0.5)
    ax.axhline(cfg.cf_hi, color="0.6", lw=0.5)
    ax2 = ax.twinx()
    ax2.plot(x, dep, "m-", lw=0.8, alpha=0.7)
    ax2.set_ylabel("envelope depth", color="m", fontsize=8)
    ax2.set_ylim(0, 1.2)
    ax.set_title("gate signals along the core path (CF black, envelope "
                 "depth magenta)", fontsize=9)
    ax.legend(fontsize=7); ax.set_xlabel("$x$")

    ax = axs[1, 0]
    im = ax.imshow(np.where(band, ls2d, np.nan), extent=extent,
                   origin="lower", aspect="auto", cmap="coolwarm",
                   vmin=0, vmax=1)
    ax.set_title("line score field (1 = elongate along core, "
                 "0 = stay isotropic)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axs[1, 1]
    ax.imshow(np.ma.masked_where(~interior, u), extent=extent,
              origin="lower", aspect="auto", cmap="gray")
    tt = np.linspace(0, 2 * np.pi, 60)
    for p in patches[:: 1]:
        cx = x[0] + p["cx"] * dx
        cy = y[0] + p["cy"] * (cfg.Ly / cfg.Ny)
        c, s = np.cos(p["theta"]), np.sin(p["theta"])
        ex = p["s_par"] * np.cos(tt) * c - p["s_perp"] * np.sin(tt) * s
        ey = p["s_par"] * np.cos(tt) * s + p["s_perp"] * np.sin(tt) * c
        col = plt.cm.coolwarm(p["ls"]) if p["branch"] == "core" else "y"
        ax.plot(cx + ex, cy + ey, color=col, lw=1.0)
    ax.set_title("kernel ellipses: yellow = far field (long axis along "
                 "$k$, normal to stripes); core colored by line score",
                 fontsize=9)
    axs[1, 2].axis("off")
    for p in patches:
        if p["branch"] == "edge":
            pass  # edge patches drawn implicitly as absent core/far rings
    fig.suptitle(f"{field_name} mu={mu:.2f}: gate diagnostics "
                 f"(gate={gate_used})", fontsize=9)
    fig.savefig(out_dir / "gate.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return rec


def run(cfg):
    out_root = Path(cfg.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for field_name in cfg.fields:
        for mu in cfg.mus:
            run_case(cfg, field_name, mu, out_root, rows)
    keys = list(rows[0].keys())
    with open(out_root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_root}/summary.csv")
    print("\nAcceptance view (same config, no switch):")
    for r in rows:
        if r["field"] == "knee":
            print(f"  knee   mu={r['mu']:.2f}: corrugation iso "
                  f"{r['iso_osc']:.3f} -> auto {r['auto_osc']:.3f}   "
                  f"(|C|={r['circ_med']:.2f}, ls={r['ls_med']:.2f}, gate={r['gate_used']})")
        else:
            print(f"  pismen mu={r['mu']:.2f}: recovery iso "
                  f"{r['iso_recovery']:.2f} -> auto "
                  f"{r['auto_recovery']:.2f}   (|C|={r['circ_med']:.2f}, ls={r['ls_med']:.2f}, "
                  f"env depth {r['env_depth_pre']:.2f}->"
                  f"{r['env_depth_post']:.2f})")
    return rows


def main():
    p = argparse.ArgumentParser(description="Auto-gated adaptive averaging")
    p.add_argument("--out", type=str, default="results/auto_gate")
    p.add_argument("--fields", nargs="+", default=["knee", "pismen"])
    p.add_argument("--mus", type=float, nargs="+", default=[0.85, 0.9, 0.95])
    p.add_argument("--R", type=float, default=0.5)
    p.add_argument("--R_out", type=float, default=0.2)
    p.add_argument("--Nx", type=int, default=512)
    p.add_argument("--Ny", type=int, default=256)
    p.add_argument("--Lx", type=float, default=40 * np.pi)
    p.add_argument("--Ly", type=float, default=20 * np.pi)
    p.add_argument("--sigma", type=float, default=np.pi / 2)
    p.add_argument("--patch", type=int, default=96)
    p.add_argument("--leak_delta", type=float, default=0.05)
    p.add_argument("--n_defects", type=int, default=8)
    p.add_argument("--central_periods", type=float, default=4.0)
    p.add_argument("--edge_margin_periods", type=float, default=2.0)
    p.add_argument("--relax_T_pismen", type=float, default=3.0)
    p.add_argument("--relax_T_knee", type=float, default=0.0)
    p.add_argument("--relax_h", type=float, default=0.05)
    p.add_argument("--gate_window_L", type=float, default=2.0)
    p.add_argument("--sigma_gate", type=float, default=np.pi / 8)
    p.add_argument("--sigma_dir", type=float, default=np.pi / 4)
    p.add_argument("--L_over_sigdir_min", type=float, default=8.5)
    p.add_argument("--circ_lo", type=float, default=0.25)
    p.add_argument("--circ_hi", type=float, default=0.5)
    p.add_argument("--cf_lo", type=float, default=1.10)
    p.add_argument("--cf_hi", type=float, default=1.35)
    p.add_argument("--core_coh_frac", type=float, default=0.5)
    p.add_argument("--core_band_sigma", type=float, default=2.0)
    p.add_argument("--xmargin", type=float, default=0.05)
    p.add_argument("--ymargin", type=float, default=0.10)
    p.add_argument("--tanhscale", type=float, default=60.0)
    run(p.parse_args())


if __name__ == "__main__":
    if len(sys.argv) == 1:
        class _Args:
            out = "results/auto_gate"
            fields = ["knee", "pismen"]
            mus = [0.85, 0.9, 0.95]
            R, R_out = 0.5, 0.2
            Nx, Ny = 512, 256
            Lx, Ly = 40 * np.pi, 20 * np.pi
            sigma = np.pi / 2
            patch = 32
            leak_delta = 0.02
            n_defects = 8
            central_periods = 4.0
            edge_margin_periods = 2.0
            relax_T_pismen = 3.0
            relax_T_knee = 0.0
            relax_h = 0.05
            gate_window_L = 2.0
            sigma_gate = np.pi / 8
            sigma_dir = np.pi / 4
            L_over_sigdir_min = 8.5
            circ_lo, circ_hi = 0.25, 0.5
            cf_lo, cf_hi = 1.10, 1.35
            core_coh_frac = 0.5
            core_band_sigma = 2.0
            xmargin, ymargin = 0.05, 0.10
            tanhscale = 60.0
        run(_Args)
    else:
        main()