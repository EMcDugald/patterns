# src/op_extract/uhu_adaptive.py
"""
Adaptive anisotropic uHu wavevector extraction.

Production version of the iteration validated in
pipelines/adaptive_filter_validation.py (knee and arc ground-truth fields):
the fixed isotropic Gaussian average in the uHu pipeline is replaced by a
spatially varying elliptical kernel whose orientation and axis lengths are
set from the CURRENT k-field estimate, iterated to self-consistency.

Design choices carried over from the validation study (do not silently
change these, they were the difference between improving and degrading on
the knee field):

  orient = "slowness"  The kernel long axis lies along level sets of the
      slow fields |k| and k2, i.e. along the PGB core, not along the
      stripe tangent. The original stripe-tangent prescription DEGRADES
      monotonically at a knee, because there the stripe tangent points
      across the boundary.
  area redistribution  s_par = sigma (1 + (aspect_max - 1) c),
      s_perp = sigma / (1 + (sharpen - 1) c), coherence-gated, so the
      kernel narrows across the slow direction only where the rank-one
      description is trustworthy and relaxes to the isotropic sigma near
      defects (c -> 0).
  sign-aligned damping  k_new is flipped wherever dot(k_new, k_prev) < 0
      before blending; with the director ambiguity (k ~ -k) a naive blend
      can cancel the field across an eigenbranch flip.

Output dict matches compute_uhu_ops_ramp / compute_uhu_ops_smoother keys
(k, A, k1_orig, k2_orig, lam1, lam2, ramp, ...) so existing runners,
J diagnostics, and the phase pipeline consume it unchanged. Extra keys:
coh (final coherence field), iter_delta (per-iteration RMS director-aligned
k change on the interior), n_iter_run.

A warm start (seed=...) accepts the converged result of the previous frame
of a time series, skipping the isotropic seed pass; besides the speedup
this keeps the kernel field temporally coherent across frames.
"""

import numpy as np

try:
    from utils.anisotropic_macro import (
        _hann2d, director_orientation, macro_aniso_uniform,
        coherence_from_eigs,
    )
except ImportError:
    from anisotropic_macro import (
        _hann2d, director_orientation, macro_aniso_uniform,
        coherence_from_eigs,
    )

try:
    from op_extract.uhu_smoother import compute_uhu_ops_smoother
except ImportError:
    from uhu_smoother import compute_uhu_ops_smoother

try:
    from utils.spectral import macro as _macro_default
except ImportError:
    from spectral import macro as _macro_default


# -----------------------------------------------------------------------
# Spatially varying kernel: area-redistributing patchwise smoother
# (promoted from adaptive_filter_validation.macro_aniso_field_v2)
# -----------------------------------------------------------------------

def macro_aniso_field_v3_pixelblend(arr, k1, k2, coherence, Lx, Ly,
                                    sigma_base, aspect_max=2.5, sharpen=2.0,
                                    patch=64, stride=None, scale=1.0,
                                    warn_support=True, gate="legacy",
                                    kernel_kind="gaussian"):
    """
    Alternative to the patch-median gate in macro_aniso_field_v2: decouples
    "how big an area to average" (still needs to be many stripe
    wavelengths for the local spectral estimate to mean anything -- that's
    what `patch` sets) from "at what resolution to decide how anisotropic
    to be" (the actual crease is narrower than the SPACING between patch
    centers, so no patch-level summary, however it's computed inside the
    patch, can resolve it).

    Two full fields are built, each with a FIXED shape (no coherence
    dependence at all):
      iso   : exact single-FFT isotropic average at sigma_eff.
      aniso : patchwise average at the MAXIMAL anisotropic shape
              (s_par = sigma_eff*aspect_max, s_perp = sigma_eff/sharpen
              everywhere). Only the ORIENTATION still varies by patch
              (from the director field) -- that part is fine, it varies
              slowly compared to the patch spacing.

    They're then blended PER PIXEL using the coherence field at its
    native, full pixel resolution (not a patch median):
      gate="legacy":     out = c*aniso + (1-c)*iso
      gate="core_aware": out = (1-c)*aniso + c*iso
    Because the blend happens after both full fields already exist, the
    coherence value used is whatever it is at that exact pixel -- there's
    no patch grid left for it to fall between.
    """
    sigma_eff = sigma_base * scale
    ny, nx = arr.shape

    iso = macro_aniso_uniform(arr, 0.0, sigma_eff, sigma_eff, Lx, Ly,
                              kind=kernel_kind)

    stride = stride or patch // 2
    win = _hann2d(patch, patch)
    out = np.zeros_like(arr, dtype=float)
    wsum = np.zeros_like(arr, dtype=float)
    s_par = sigma_eff * aspect_max
    s_perp = sigma_eff / max(sharpen, 1.0)

    if warn_support:
        patch_Lx = Lx * patch / nx
        patch_Ly = Ly * patch / ny
        if s_par > 0.25 * patch_Lx or s_perp > 0.25 * patch_Ly:
            print(f"  [macro_aniso_field_v3_pixelblend] warning: kernel "
                 f"size (s_par={s_par:.2f}, s_perp={s_perp:.2f}) "
                 f"approaches the patch footprint ({patch_Lx:.2f} x "
                 f"{patch_Ly:.2f}); increase `patch` or reduce `scale`.")

    ys = list(range(0, ny - patch + 1, stride)) or [0]
    xs = list(range(0, nx - patch + 1, stride)) or [0]
    if ys[-1] != ny - patch:
        ys.append(max(ny - patch, 0))
    if xs[-1] != nx - patch:
        xs.append(max(nx - patch, 0))

    for y0 in ys:
        for x0 in xs:
            sl = (slice(y0, y0 + patch), slice(x0, x0 + patch))
            a_patch = arr[sl]
            if a_patch.shape != (patch, patch):
                continue
            k1p, k2p = k1[sl], k2[sl]
            weight = np.hypot(k1p, k2p)
            theta_k = director_orientation(k1p, k2p, weight=weight)
            theta_long = theta_k + np.pi / 2.0
            smoothed = macro_aniso_uniform(
                a_patch, theta_long, s_par, s_perp,
                Lx * patch / nx, Ly * patch / ny, kind=kernel_kind)
            out[sl] += smoothed * win
            wsum[sl] += win
    wsum = np.where(wsum < 1e-12, 1.0, wsum)
    aniso = out / wsum

    c = coherence
    if gate == "legacy":
        return c * aniso + (1.0 - c) * iso
    else:
        return (1.0 - c) * aniso + c * iso


def macro_aniso_field_v2(arr, k1, k2, coherence, Lx, Ly, sigma_base,
                         aspect_max=2.5, sharpen=2.0, patch=64,
                         stride=None, scale=1.0, warn_support=True,
                         gate="legacy", kernel_kind="gaussian",
                         coh_window=None):
    """
    Patchwise Hann overlap-add anisotropic smoothing of `arr`, oriented by
    the director field (k1, k2), with axis lengths redistributed by the
    local coherence c in [0, 1].

    gate="legacy" (original behavior, kept as default for backward
    compatibility with v2):
        s_par  = sigma_eff * (1 + (aspect_max - 1) c)   (long axis)
        s_perp = sigma_eff / (1 + (sharpen    - 1) c)   (cross axis)
    i.e. anisotropy is strongest where coherence is HIGH, and relaxes
    toward isotropic as c -> 0 (near a core/defect). Conservative: near a
    core, where the orientation estimate itself is least trustworthy,
    this backs off toward isotropic rather than committing harder to a
    (possibly wrong) direction.

    gate="core_aware" (new): the same formulas with c replaced by (1-c),
        s_par  = sigma_eff * (1 + (aspect_max - 1) (1-c))
        s_perp = sigma_eff / (1 + (sharpen    - 1) (1-c))
    i.e. anisotropy is WEAKEST (near-isotropic, small footprint) in the
    high-coherence far field, and STRONGEST exactly where c is lowest --
    on the coherence "line" (a PGB core). This is the "forbid crossing
    near the line, go maximally along-core exactly on the line" behavior:
    approaching a core, the kernel first shrinks (c dropping shrinks
    BOTH axes toward sigma_eff from whichever side they started), then at
    the core itself becomes maximally elongated along it and maximally
    narrow across it. The orientation itself (which way is "along the
    line") is unaffected by this switch -- it was already being computed
    correctly upstream via the slowness director, see the `orient`
    argument of compute_uhu_ops_adaptive: (k1, k2) passed in here should
    be that slowness director, and theta_long = theta_k + pi/2 below
    rotates it to point along the level sets of the slow fields, i.e.
    along the core, not across it. What "legacy" got wrong was only the
    SIZE gating (which end of the coherence range gets the elongation),
    not the orientation, which was already right.

    `scale` is an overall size multiplier applied BEFORE the aspect/
    sharpen redistribution, so it grows or shrinks the whole kernel
    footprint while leaving the shape (the s_par/s_perp ratio at a given
    coherence) unchanged. `aspect_max`/`sharpen` alone only control that
    ratio, not the absolute size -- this is the separate "how big" knob.

    kernel_kind="gaussian" (default) or "rect": the per-patch average is
    either a smooth rotated Gaussian or a hard-edged rotated box (uniform
    weight over a rotated rectangle of half-widths s_par x s_perp, zero
    outside). "rect" is the literal "rectangular support that slices
    the pattern perpendicular to the stripe" option -- useful to check
    whether the Gaussian's soft tails (which always leak a bit of
    cross-core content even when s_perp is small) are contributing to
    the leakage independent of the aspect/sharpen shape choice.

    coh_window: size (pixels) of a sub-window CENTERED on the patch,
    used instead of the full patch when computing c_local. Defaults to
    the full patch (previous behavior). A patch spans a region much
    larger than the crease (~1/k2), so its whole-patch median coherence
    is dominated by far-field pixels even when the patch is centered
    exactly on a core -- the low-coherence core signal gets diluted
    before it ever reaches the gate. Setting coh_window to something
    closer to the crease scale (a handful of pixels, not patch//1)
    makes c_local responsive to whether THIS patch is actually centered
    on a core, rather than to how much far field happens to surround it.

    The long axis is taken PERPENDICULAR to (k1, k2); pass the slowness
    director (gradient of the slow fields) as (k1, k2) to align the long
    axis with the level sets of the slow fields.

    Each patch is smoothed via a single-FFT kernel computed ON THAT
    PATCH ALONE (periodic within the patch), then stitched back with
    Hann overlap-add. That means the patch is the hard ceiling on kernel
    support: if s_par approaches the patch's physical half-width, the
    kernel wraps around within the patch instead of decaying (or, for
    "rect", is truncated by the periodic tiling), which is a truncation
    artifact, not a feature. If warn_support is True, this prints once
    when the worst-case s_par or s_perp exceeds patch_width/4.
    """
    ny, nx = arr.shape
    stride = stride or patch // 2
    win = _hann2d(patch, patch)
    out = np.zeros_like(arr, dtype=float)
    wsum = np.zeros_like(arr, dtype=float)
    sigma_eff = sigma_base * scale

    if warn_support:
        patch_Lx = Lx * patch / nx
        patch_Ly = Ly * patch / ny
        s_par_worst = sigma_eff * aspect_max
        s_perp_worst = sigma_eff / max(sharpen, 1.0)
        if s_par_worst > 0.25 * patch_Lx or s_perp_worst > 0.25 * patch_Ly:
            print(f"  [macro_aniso_field_v2] warning: worst-case kernel "
                 f"size (s_par={s_par_worst:.2f}, s_perp={s_perp_worst:.2f}) "
                 f"approaches the patch footprint ({patch_Lx:.2f} x "
                 f"{patch_Ly:.2f}); increase `patch` or reduce `scale` to "
                 f"avoid periodic wraparound/tiling truncation within a "
                 f"patch.")

    ys = list(range(0, ny - patch + 1, stride)) or [0]
    xs = list(range(0, nx - patch + 1, stride)) or [0]
    if ys[-1] != ny - patch:
        ys.append(max(ny - patch, 0))
    if xs[-1] != nx - patch:
        xs.append(max(nx - patch, 0))

    for y0 in ys:
        for x0 in xs:
            sl = (slice(y0, y0 + patch), slice(x0, x0 + patch))
            a_patch = arr[sl]
            if a_patch.shape != (patch, patch):
                continue
            k1p, k2p, cp = k1[sl], k2[sl], coherence[sl]
            weight = np.hypot(k1p, k2p)
            theta_k = director_orientation(k1p, k2p, weight=weight)
            theta_long = theta_k + np.pi / 2.0
            if coh_window is not None and coh_window < patch:
                off = (patch - coh_window) // 2
                c_sub = cp[off:off + coh_window, off:off + coh_window]
            else:
                c_sub = cp
            c_local = float(np.median(c_sub))
            g = c_local if gate == "legacy" else (1.0 - c_local)
            s_par = sigma_eff * (1.0 + (aspect_max - 1.0) * g)
            s_perp = sigma_eff / (1.0 + (sharpen - 1.0) * g)
            smoothed = macro_aniso_uniform(
                a_patch, theta_long, s_par, s_perp,
                Lx * patch / nx, Ly * patch / ny, kind=kernel_kind)
            out[sl] += smoothed * win
            wsum[sl] += win

    wsum = np.where(wsum < 1e-12, 1.0, wsum)
    return out / wsum


# -----------------------------------------------------------------------
# Slowness director (promoted from adaptive_filter_validation)
# -----------------------------------------------------------------------

def slowness_director(k1, k2, sigma, Lx, Ly, deriv_fun, macro_fn=None):
    """
    Direction of fastest macroscopic variation: doubled-angle-averaged
    orientation of grad|k| and grad(k2), pre-smoothed at scale sigma.
    Feeding this to the kernel constructor puts the kernel LONG axis along
    the level sets of the slow fields, i.e. along a PGB core.
    """
    macro_fn = macro_fn or _macro_default
    kmag = macro_fn(np.hypot(k1, k2), sigma, Lx, Ly)
    k2s = macro_fn(k2, sigma, Lx, Ly)
    g = np.zeros((2,) + kmag.shape)
    for f in (kmag, k2s):
        gx = deriv_fun(f, Lx, Ly, "x")
        gy = deriv_fun(f, Lx, Ly, "y")
        g[0] += gx * gx - gy * gy
        g[1] += 2.0 * gx * gy
    ang = 0.5 * np.arctan2(g[1], g[0])
    mag = np.sqrt(np.hypot(g[0], g[1]))
    return mag * np.cos(ang), mag * np.sin(ang)


# -----------------------------------------------------------------------
# Derived kernel parameters from the leakage calculation
# -----------------------------------------------------------------------

def derived_kernel_params(k1_far, k2_far, sigma, delta=0.05,
                          aspect_cap=4.0, core_width_frac=0.7):
    """
    Kernel axis lengths derived from the near-core leakage calculation
    rather than tuned by hand.

    Near a PGB core the field is locally u ~ 2A cos(k1 x) cos(k2 y), so
    u^2 (and every uHu tensor component) carries modulations at 2*k1
    along the core and 2*k2 across it. A Gaussian of width s passes a
    modulation at wavenumber q with residual exp(-s^2 q^2 / 2); the
    isotropic sigma kills the 2*k2 term but at large far-field angle
    passes the 2*k1 term almost untouched, imprinting a spurious
    periodic signature at wavelength pi/k1, which is exactly the PN
    dislocation spacing L.

    Requiring the along-core residual exp(-2 s_par^2 k1^2) <= delta
    gives s_par = sqrt(ln(1/delta)/2) / k1, hence

        aspect_max = s_par / sigma            (capped at aspect_cap)

    The cross axis is set to resolve the crease: s_perp =
    core_width_frac / k2 (the crease width is ~ 1/k2), i.e.

        sharpen = sigma * k2 / core_width_frac  (never below 1)

    with the caveat that shrinking s_perp re-admits the 2*k2 term at
    rate exp(-2 s_perp^2 k2^2); core_width_frac ~ 0.7 keeps that
    residual at the few-percent level while roughly halving core blur.

    Returns (aspect_max, sharpen).
    """
    k1_far = abs(float(k1_far))
    k2_far = abs(float(k2_far))
    if k1_far < 1e-8 or k2_far < 1e-8:
        return 1.0, 1.0
    s_par = np.sqrt(np.log(1.0 / delta) / 2.0) / k1_far
    aspect = min(max(s_par / sigma, 1.0), aspect_cap)
    s_perp = core_width_frac / k2_far
    sharpen = max(sigma / s_perp, 1.0)
    return float(aspect), float(sharpen)


def derived_kernel_params_mu(mu, sigma, **kw):
    """Convenience wrapper: far-field k1 = sqrt(1 - mu^2), k2 = mu."""
    mu = float(mu)
    return derived_kernel_params(np.sqrt(max(1.0 - mu * mu, 0.0)), mu,
                                 sigma, **kw)


# -----------------------------------------------------------------------
# The main entry point: adaptive extraction on one frame
# -----------------------------------------------------------------------

def compute_uhu_ops_adaptive(u, x, y, sigma, deriv_fun, ramp,
                             n_iter=3,
                             aspect_max=2.5,
                             sharpen=2.0,
                             damping=0.4,
                             patch=64,
                             orient="slowness",
                             seed=None,
                             tol=None,
                             macro_fn=None,
                             with_history=False,
                             scale=1.0,
                             gate="legacy",
                             kernel_kind="gaussian",
                             coh_window=None,
                             blend="patch"):
    """
    Adaptive anisotropic uHu extraction for a single frame.

    Parameters
    ----------
    u, x, y, sigma, deriv_fun, ramp : as in compute_uhu_ops_ramp; sigma is
        the isotropic base width, reused as the kernel area scale.
    n_iter : max anisotropic iterations after the seed pass.
    aspect_max, sharpen : kernel area redistribution, see
        macro_aniso_field_v2. sharpen = 1.0 recovers the grow-only kernel.
    damping : blend fraction for the sign-aligned k update.
    patch : patch size in pixels for the overlap-add smoother.
    orient : "slowness" (recommended) or "stripe".
    seed : optional dict with keys k1_orig, k2_orig, lam1, lam2 (e.g. the
        previous frame's converged output). If given, the isotropic seed
        pass is skipped.
    tol : optional early-stop threshold on the interior RMS director-
        aligned change of (k1, k2) per iteration.
    macro_fn : isotropic Gaussian used inside slowness_director and for
        the seed pass; defaults to utils.spectral.macro.
    with_history : if True, also return the full list of per-iteration
        result dicts under key "history".

    Returns
    -------
    dict with the compute_uhu_ops_ramp schema (k, A, k1_orig, k2_orig,
    lam1, lam2, ramp, eigenvector components), where k1_orig / k2_orig are
    the damped, sign-aligned iterates, plus:
        k_vec   : hypot(k1_orig, k2_orig), magnitude of the blended vector
                  (equals k * ramp in the interior),
        coh     : final coherence field,
        iter_delta : array of per-iteration interior RMS k changes,
        n_iter_run : number of anisotropic iterations actually run.
    """
    macro_fn = macro_fn or _macro_default
    Lx, Ly = x[-1] - x[0], y[-1] - y[0]
    interior = ramp >= 0.99999

    if seed is None:
        iso_sm = lambda arr: macro_fn(arr, sigma * scale, Lx, Ly)
        res = compute_uhu_ops_smoother(u, x, y, iso_sm, deriv_fun, ramp)
        k1p, k2p = res["k1_orig"], res["k2_orig"]
        lam1, lam2 = res["lam1"], res["lam2"]
        history = [res]
    else:
        res = None
        k1p, k2p = seed["k1_orig"], seed["k2_orig"]
        lam1, lam2 = seed["lam1"], seed["lam2"]
        history = []

    deltas = []
    n_run = 0
    for _ in range(n_iter):
        coh = coherence_from_eigs(lam1, lam2)
        if orient == "slowness":
            d1, d2 = slowness_director(k1p, k2p, sigma, Lx, Ly,
                                       deriv_fun, macro_fn)
        else:
            d1, d2 = k1p, k2p

        if blend == "pixel":
            sm = lambda arr: macro_aniso_field_v3_pixelblend(
                arr, k1=d1, k2=d2, coherence=coh, Lx=Lx, Ly=Ly,
                sigma_base=sigma, aspect_max=aspect_max, sharpen=sharpen,
                patch=patch, scale=scale, gate=gate, kernel_kind=kernel_kind)
        else:
            sm = lambda arr: macro_aniso_field_v2(
                arr, k1=d1, k2=d2, coherence=coh, Lx=Lx, Ly=Ly,
                sigma_base=sigma, aspect_max=aspect_max, sharpen=sharpen,
                patch=patch, scale=scale, gate=gate, kernel_kind=kernel_kind,
                coh_window=coh_window)
        new = compute_uhu_ops_smoother(u, x, y, sm, deriv_fun, ramp)

        # sign-align, then damp
        flip = np.sign(new["k1_orig"] * k1p + new["k2_orig"] * k2p)
        flip = np.where(flip == 0, 1.0, flip)
        k1n = flip * new["k1_orig"]
        k2n = flip * new["k2_orig"]
        k1b = (1 - damping) * k1p + damping * k1n
        k2b = (1 - damping) * k2p + damping * k2n

        d_rms = float(np.sqrt(np.mean(
            (k1b - k1p)[interior] ** 2 + (k2b - k2p)[interior] ** 2)))
        deltas.append(d_rms)

        new["k1_orig"], new["k2_orig"] = k1b, k2b
        k1p, k2p = k1b, k2b
        lam1, lam2 = new["lam1"], new["lam2"]
        res = new
        n_run += 1
        history.append(res)
        if tol is not None and d_rms < tol:
            break

    if res is None:
        raise ValueError("n_iter = 0 with a warm-start seed leaves nothing "
                         "to compute; use n_iter >= 1.")

    out = dict(res)
    out["k_vec"] = np.hypot(out["k1_orig"], out["k2_orig"])
    out["coh"] = coherence_from_eigs(out["lam1"], out["lam2"])
    out["iter_delta"] = np.asarray(deltas)
    out["n_iter_run"] = n_run
    if with_history:
        out["history"] = history
    return out
