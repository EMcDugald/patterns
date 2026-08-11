# pipelines/t1_director_source_test.py
"""
T1: where are the pi jumps born?

The phase pipeline is a chain:

    u -> uHu + averaging -> khat -> coordinate lines -> sample u along each
    line -> 1D Hilbert, unwrap -> griddata back to the grid

Running it end to end cannot attribute a failure to a stage. This script
holds every stage fixed except the director, and swaps in four sources:

    exact           k = grad(theta) in closed form, no extraction at all
    exact_oriented  the same, pushed through orient_vector_field_v2
    uhu             the current extraction, isotropic Gaussian
    riesz           the doubled-angle smoothed Riesz director

Note that compute_phase_from_k_knee normalizes k immediately (line ~290,
"dx = dir_sign * kx / mag * ds") and never uses |k| again, so the phase
reconstruction depends on the extraction ONLY through the director. That is
what makes the four sources interchangeable here, and it is also why any
positive scalar field multiplying k (a ramp, for instance) is invisible to
this test wherever it is nonzero.

Ground truth. On the knee core y = 0,

    theta(x, 0) = k1 x + log 2

exactly linear with slope k1, for every mu. So the reconstructed core phase
is compared against that, up to one fitted additive constant, and the
residual is the measurement. A pi jump shows up as a step of size ~pi in the
residual. The x positions of any steps are compared against the spacing
pi / k1, which is the Pismen-Nepomnyashchy defect spacing at the same mu:
if the spurious structure sits at that spacing, a defect-free knee is
impersonating a dislocation chain.

Reading the result:

    exact clean, uhu not       the director is the problem, and the
                               optimal-kernel work is aimed correctly
    exact also fails           no averaging kernel can fix it, since no
                               averaging happened; the coordinate line,
                               stitch, Hilbert or griddata stage is the
                               source
    exact clean, exact_oriented not
                               the orientation pass alone injects the jumps
    riesz clean where uhu is not
                               consistent with the corrugation channel being
                               the culprit, since a linear filter of u never
                               forms the 2 theta term at all

Scope: knee only, all fields from closed forms, no simulation data.

Run:
    python t1_director_source_test.py
    python t1_director_source_test.py --mu 0.83 0.846 0.86 --seed exact
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _import_first(names, attr):
    """Import `attr` from the first module path in `names` that provides it.
    The repo layout has moved between sessions; this avoids hard-coding one."""
    tried = []
    for nm in names:
        try:
            mod = importlib.import_module(nm)
        except Exception as exc:
            tried.append(f"{nm}: {exc}")
            continue
        if hasattr(mod, attr):
            return getattr(mod, attr)
        tried.append(f"{nm}: no attribute {attr}")
    raise ImportError(f"could not import {attr}; tried:\n  " + "\n  ".join(tried))


compute_pgb_phase_from_uhu = _import_first(
    ["op_extract.phase_zigzag", "utils.phase_zigzag", "phase_zigzag"],
    "compute_pgb_phase_from_uhu")
compute_uhu_ops_smoother = _import_first(
    ["op_extract.uhu_smoother", "utils.uhu_smoother", "uhu_smoother"],
    "compute_uhu_ops_smoother")
orient_vector_field_v2 = _import_first(
    ["utils.kfield_calcs", "op_extract.kfield_calcs", "kfield_calcs"],
    "orient_vector_field_v2")
riesz_director = _import_first(
    ["utils.field_diagnostics_viz", "field_diagnostics_viz"],
    "riesz_director")
spectral_deriv = _import_first(
    ["utils.field_diagnostics_viz", "field_diagnostics_viz"],
    "spectral_deriv")
macro_iso = _import_first(
    ["utils.field_diagnostics_viz", "field_diagnostics_viz"],
    "macro_iso")


# =====================================================================
# the exact knee
# =====================================================================

def knee_exact(mu, n_spacings=8.0, N=512, y_far_min=4.0, **kw):
    """
    theta = k1 x + log(2 cosh(k2 y)),  u = cos(theta),  k = grad theta.

    The box is sized from the coverage wanted on the core, not fixed in
    advance, because the covered region is set by the geometry of the
    coordinate lines and gets small fast if you guess.

    Seeds sit on one constant-phase contour at height y_top. That contour
    spans dx_contour = log(2 cosh(k2 y_top)) / k1 in x. Each line then
    drifts LEFT by about y_top k1 / k2 on its way down to y = 0, since
    upper-half seeds step along -khat. So the core interval that actually
    receives samples has width

        cover = drift + dx_contour ~ y_top (1.25 k1/k2 + k2/k1)

    Inverting that for a requested cover of n_spacings * pi/k1 fixes y_top,
    and Lx is set with margin around it. y_top is floored so that k2 y_top
    stays comfortably above 1, otherwise the seeds are not in the far field.
    """
    k1, k2 = float(np.sqrt(1 - mu**2)), float(mu)
    cover = n_spacings * np.pi / k1
    y_top = cover / (1.25 * k1 / k2 + k2 / k1)
    y_top = max(y_top, y_far_min / k2)

    drift = 1.25 * y_top * k1 / k2
    dx_contour = float(np.logaddexp(k2 * y_top, -k2 * y_top)) / k1
    Lx = 1.45 * (drift + dx_contour)
    Ly = 2.8 * y_top

    x = np.linspace(-Lx / 2, Lx / 2, N, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, N, endpoint=False)
    X, Y = np.meshgrid(x, y)

    theta = k1 * X + np.logaddexp(k2 * Y, -k2 * Y)
    u = np.cos(theta)
    return {"x": x, "y": y, "X": X, "Y": Y, "u": u, "theta": theta,
            "k1": np.full_like(X, k1), "k2": k2 * np.tanh(k2 * Y),
            "k1_ff": k1, "k2_ff": k2, "Lx": Lx, "Ly": Ly, "mu": mu,
            "y_top": y_top, "drift": drift, "dx_contour": dx_contour}


def edge_taper(x, y, frac=0.04):
    """Cosine taper in y only, over the outer `frac` of the box, so the
    spectral derivatives inside the extraction do not ring off the y seam.
    Identically 1 over the interior, so it does not touch the core row."""
    Ly = y[-1] - y[0]
    d = np.minimum(y - y[0], y[-1] - y) / Ly
    w = np.ones_like(y)
    m = d < frac
    w[m] = 0.5 * (1 - np.cos(np.pi * d[m] / frac))
    return np.tile(w[:, None], (1, len(x)))


# =====================================================================
# seed contour
# =====================================================================

def exact_level_set_contour(fld, n_per_edge=400):
    """
    A true constant-phase contour of the exact knee, from y_top down through
    the core to -y_top. Solving k1 x + log(2 cosh(k2 y)) = C for x is closed
    form. Placed so that the core interval it feeds is centred on x = 0.

    The trapezoid from build_knee_boundary_from_mu is the far-field straight
    line version of this same curve; --seed trapezoid uses that instead, and
    comparing the two is a free check on the seeding step.
    """
    k1, k2 = fld["k1_ff"], fld["k2_ff"]
    y_top, drift, dxc = fld["y_top"], fld["drift"], fld["dx_contour"]
    x_top = 0.5 * (drift - dxc)
    C = k1 * x_top + np.logaddexp(k2 * y_top, -k2 * y_top)

    ys = np.linspace(y_top, -y_top, 2 * n_per_edge + 1)
    xs = (C - np.logaddexp(k2 * ys, -k2 * ys)) / k1
    inside = (xs > fld["x"][0]) & (xs < fld["x"][-1])
    if not np.all(inside):
        print(f"  [warn] level set leaves the box at "
              f"{np.sum(~inside)} / {xs.size} points; dropped")
    return np.vstack([xs[inside], ys[inside]])


# =====================================================================
# the four director sources
# =====================================================================

def director_exact(fld, **kw):
    return fld["k1"].copy(), fld["k2"].copy()


def director_exact_oriented(fld, pi_tol=np.pi / 10, **kw):
    a, b = orient_vector_field_v2(fld["k1"], fld["k2"], pi_tol=pi_tol)
    return (np.ma.filled(a, 0.0).astype(float),
            np.ma.filled(b, 0.0).astype(float))


def director_uhu(fld, sigma=np.pi / 2, ramp=None, **kw):
    Lx, Ly = fld["Lx"], fld["Ly"]
    if ramp is None:
        ramp = np.ones_like(fld["u"])
    sm = lambda arr: macro_iso(arr, sigma, Lx, Ly)
    ops = compute_uhu_ops_smoother(fld["u"], fld["x"], fld["y"], sm,
                                   spectral_deriv, ramp)
    return ops["k1_orig"], ops["k2_orig"]


def director_riesz(fld, sigma_dir=np.pi / 2, **kw):
    rd = riesz_director(fld["u"], fld["Lx"], fld["Ly"], sigma_dir)
    return rd["d1"], rd["d2"]


SOURCES = {
    "exact": director_exact,
    "exact_oriented": director_exact_oriented,
    "uhu": director_uhu,
    "riesz": director_riesz,
}


# =====================================================================
# run one (mu, source) and score the core row
# =====================================================================

def rebuild_grid(res, fld, gauge, half="orig"):
    """
    Re-grid the phase from the pipeline's OWN coordinate lines and Hilbert
    phases, changing only the per-line unwrapping origin.

    The block enforcing phi = 0 at the seed is present but commented out in
    phase_zigzag.py (lines 360 to 372), so each line currently starts at
    arg z at its first sample, which is a Hilbert startup transient rather
    than the seed phase. Neighbouring lines therefore land on the core with
    different additive constants, and the gridded core row is a patchwork.
    That is a candidate explanation for a staircase which has nothing to do
    with the director, so it has to be separated out before anything else
    can be read.

      raw          the pipeline as it stands
      zero_origin  subtract each line's first value (the commented block)
      seed_phase   subtract the first value and add the EXACT theta at the
                   seed point, which is available only because this is an
                   analytic test; the ideal gauge

    Everything else, the lines, the sampling, the Hilbert transform, the
    interpolation method, is untouched.
    """
    from scipy.interpolate import griddata

    if gauge == "raw" and half == "stitched":
        return res["phase_grid_unwrapped"][:, :, 0]

    x, y = fld["x"], fld["y"]
    k1, k2 = fld["k1_ff"], fld["k2_ff"]
    lines = res["coordinate_lines"][0]
    phases = res["phase_lines_unwrapped"][0]

    pts, vals = [], []
    for ln, ph in zip(lines, phases):
        ln = np.asarray(ln, dtype=float)
        ph = np.asarray(ph, dtype=float)
        if ln.shape[0] != ph.shape[0] or ph.size < 2:
            continue
        # Use only the seed-to-core branch. Along the full stitched line the
        # true phase is V-shaped, decreasing from the seed to the core and
        # increasing again out to the mirrored seed, but the Hilbert phase of
        # an oscillatory record is monotone, so the reflected half accumulates
        # with the wrong sign and griddata blends the two. That shows up as an
        # error field antisymmetric about y = 0.
        if half == "orig":
            ic = int(np.argmin(np.abs(ln[:, 1])))
            if ic < 8:
                continue
            ln, ph = ln[:ic + 1], ph[:ic + 1]
        off = ph[0]
        if gauge == "seed_phase":
            # the Hilbert phase accumulates as -theta along the line, since
            # upper-half seeds step along -khat, so the seed phase enters
            # with a minus sign. Getting this backwards doubles the
            # seed-contour mismatch instead of removing it.
            xs0, ys0 = ln[0]
            off = off + (k1 * xs0 + np.logaddexp(k2 * ys0, -k2 * ys0))
        pts.append(ln)
        vals.append(ph - off)
    if not pts:
        return res["phase_grid_unwrapped"][:, :, 0]

    pts = np.vstack(pts)
    vals = np.concatenate(vals)
    Xg, Yg = np.meshgrid(x, y)
    return griddata(pts, vals, (Xg, Yg), method="linear")


def run_one(fld, source, seed_mode, n_seeds, sigma, ds, ramp, use_sym):
    k1d, k2d = SOURCES[source](fld, sigma=sigma, ramp=ramp, sigma_dir=sigma)

    uhu_data = {
        "x": fld["x"], "y": fld["y"],
        "u": fld["u"][:, :, None],
        "k1_sym": np.asarray(k1d)[:, :, None],
        "k2_sym": np.asarray(k2d)[:, :, None],
        "k1_orig": np.asarray(k1d)[:, :, None],
        "k2_orig": np.asarray(k2d)[:, :, None],
        "mu": fld["mu"], "tt": None, "ramp": None,
        "k": None, "A": None,
        "knee_bdry": (exact_level_set_contour(fld)
                      if seed_mode == "exact" else None),
        "source_path": f"analytic knee mu={fld['mu']} director={source}",
    }

    res = compute_pgb_phase_from_uhu(
        uhu_data, mu=fld["mu"], n_phase_seeds=n_seeds, ds=ds,
        seed_half="upper", phase_ramp_mode="none", prefer_sym=True)

    # how many coordinate lines actually reach y = 0? A line that stops
    # early has been truncated by the box, and everything downstream of it
    # is interpolation over a line that never sampled the core.
    lines = res["coordinate_lines"][0]
    reach = [float(np.min(np.abs(np.asarray(ln, float)[:, 1]))) for ln in lines
             if len(ln)]
    res["frac_reaching_core"] = (float(np.mean(np.array(reach) < 0.5))
                                 if reach else 0.0)

    res["director"] = (np.asarray(k1d), np.asarray(k2d))

    key = ("phase_grid_symmetric_unwrapped" if use_sym
           else "phase_grid_unwrapped")
    return res[key][:, :, 0], res


def score_core(phi_grid, fld, x_trim=0.15):
    """
    Score the reconstructed core phase against the exact one, in a gauge
    that the pipeline's own conventions cannot fake.

    Two conventions have to be quotiented out. The coordinate lines for
    upper-half seeds run along -khat (dir_sign = -1 at phase_zigzag.py:294),
    so the Hilbert phase accumulates as -theta rather than +theta. And each
    line carries its own unwrapping origin, since the block enforcing phi = 0
    at the seed is commented out, so the additive constant is not fixed.

    So fit phi ~ a x + c on the core row and report:

      slope |a| against the exact k1, which is the wavenumber check
      the residual about that line, which is the linearity check

    On the knee both are known exactly: |a| = k1 and residual identically
    zero, for every mu. The slope is fitted from the MEDIAN of the local
    differences rather than by least squares, so that a pi step does not
    tilt the fitted line and hide itself.
    """
    x, y = fld["x"], fld["y"]
    iy = int(np.argmin(np.abs(y)))
    phi = np.asarray(phi_grid[iy, :], dtype=float)

    # trim relative to the interval that actually received samples, not to
    # the box: griddata leaves everything outside the hull of the lines NaN,
    # and scoring a window fixed in box coordinates silently measures a
    # different region for every configuration.
    fin = np.isfinite(phi)
    if fin.sum() < 16:
        return {"n_ok": int(fin.sum()), "coverage": float(fin.mean())}
    idx = np.flatnonzero(fin)
    pad = int(x_trim * (idx[-1] - idx[0] + 1))
    nx = len(x)
    ok = np.zeros(nx, dtype=bool)
    ok[idx[0] + pad: idx[-1] - pad + 1] = True
    ok &= fin
    if ok.sum() < 16:
        return {"n_ok": int(ok.sum()), "coverage": float(fin.mean())}

    xs, ps = x[ok], phi[ok]
    dp, dx = np.diff(ps), np.diff(xs)
    good = dx > 0
    a = float(np.median(dp[good] / dx[good])) if good.any() else np.nan
    c = float(np.median(ps - a * xs))

    resid = np.full(len(x), np.nan)
    resid[ok] = ps - (a * xs + c)
    r = resid[ok]
    dr = np.diff(r)

    k1 = fld["k1_ff"]
    return {
        "n_ok": int(ok.sum()),
        "coverage": float(fin.mean()),
        "slope": a,
        "slope_err": float(abs(abs(a) - k1) / k1),
        "rms": float(np.sqrt(np.mean(r**2))),
        "max_abs": float(np.max(np.abs(r))),
        "max_step": float(np.max(np.abs(dr))) if dr.size else np.nan,
        "peak_to_peak": float(np.ptp(r)),
        "x": x, "resid": resid, "phi": phi, "iy": iy,
        "x_ok": xs, "r_ok": r,
    }


# =====================================================================
# 2D diagnostics for a single (mu, source)
# =====================================================================

def align_to_truth(phi, theta):
    """
    Put the reconstructed phase in the same gauge as the exact theta, by
    least squares on the two free conventions: phi ~ a theta + c, where a
    should come out near -1 because upper-half seeds step along -khat, and
    c absorbs the arbitrary origin. Returns the aligned field, a and c.
    Anything other than a near +-1 means the reconstruction is not a
    reparametrized copy of the truth at all, which is itself the diagnosis.
    """
    ok = np.isfinite(phi) & np.isfinite(theta)
    if ok.sum() < 32:
        return np.full_like(phi, np.nan), np.nan, np.nan
    M = np.vstack([theta[ok], np.ones(ok.sum())]).T
    (a, c), *_ = np.linalg.lstsq(M, phi[ok], rcond=None)
    if abs(a) < 1e-6:
        return np.full_like(phi, np.nan), a, c
    return (phi - c) / a, float(a), float(c)


def diagnostic_figure(fld, source, res, phi, path, gauge="seed_phase",
                      quiver_step=18, line_step=6):
    """
    Nine panels for one configuration. Top row is what went IN to the phase
    reconstruction, middle row is the reconstruction against the truth on
    the full domain, bottom row is the core profile.
    """
    x, y = fld["x"], fld["y"]
    X, Y = fld["X"], fld["Y"]
    u, theta = fld["u"], fld["theta"]
    k1, k2 = fld["k1_ff"], fld["k2_ff"]
    ext = [x[0], x[-1], y[0], y[-1]]

    phi_al, a, c = align_to_truth(phi, theta)
    err = phi_al - theta
    ok = np.isfinite(err)
    if ok.any():
        err = err - np.median(err[ok])

    fig, ax = plt.subplots(3, 3, figsize=(16.5, 13.0))

    def show_u(axis):
        axis.imshow(u, extent=ext, origin="lower", cmap="gray",
                    aspect="auto", alpha=0.85)

    # --- row 1: inputs -------------------------------------------------
    show_u(ax[0, 0])
    sb = res.get("knee_bdry_phase")
    if sb is not None:
        sb = np.asarray(sb)
        ax[0, 0].plot(sb[0], sb[1], ".", color="deepskyblue", ms=3.5)
    ax[0, 0].set_title("pattern + phase seed points")

    show_u(ax[0, 1])
    d1, d2 = res.get("director", (None, None))
    if d1 is not None:
        m = np.hypot(d1, d2)
        s_ = quiver_step
        ax[0, 1].quiver(X[::s_, ::s_], Y[::s_, ::s_],
                        (d1 / np.maximum(m, 1e-12))[::s_, ::s_],
                        (d2 / np.maximum(m, 1e-12))[::s_, ::s_],
                        color="crimson", scale=28, headwidth=2, pivot="mid")
    ax[0, 1].set_title(f"pattern + {source} director (unit)")

    show_u(ax[0, 2])
    lines = res["coordinate_lines"][0]
    for j, ln in enumerate(lines):
        if j % line_step:
            continue
        ln = np.asarray(ln, float)
        ax[0, 2].plot(ln[:, 0], ln[:, 1], "-", color="gold", lw=0.9)
    reach = res.get("frac_reaching_core", np.nan)
    ax[0, 2].set_title(f"pattern + coordinate lines "
                       f"({100 * reach:.0f}% reach core)")

    # --- row 2: full domain --------------------------------------------
    lo = np.nanpercentile(theta, 1)
    hi = np.nanpercentile(theta, 99)
    for axis, fldd, ttl in ((ax[1, 0], theta, r"expected $\theta$"),
                            (ax[1, 1], phi_al,
                             f"recovered (aligned, a={a:.3f})")):
        im = axis.imshow(fldd, extent=ext, origin="lower", cmap="twilight",
                         vmin=lo, vmax=hi, aspect="auto")
        fig.colorbar(im, ax=axis)
        axis.set_title(ttl)
        axis.axhline(0.0, color="w", lw=0.7, ls="--")

    v = np.nanpercentile(np.abs(err), 98) if ok.any() else 1.0
    im = ax[1, 2].imshow(err, extent=ext, origin="lower", cmap="RdBu_r",
                         vmin=-v, vmax=v, aspect="auto")
    fig.colorbar(im, ax=ax[1, 2])
    ax[1, 2].set_title(f"error (98th pct = {v:.2f} rad)")
    ax[1, 2].axhline(0.0, color="k", lw=0.7, ls="--")

    # --- row 3: core profile -------------------------------------------
    iy = int(np.argmin(np.abs(y)))
    th_c, ph_c, er_c = theta[iy], phi_al[iy], err[iy]

    ax[2, 0].plot(x, th_c, "k-", lw=1.6, label=r"expected $\theta$")
    ax[2, 0].plot(x, ph_c, "C3-", lw=1.2, label="recovered")
    ax[2, 0].set_title(f"core row: slope should be k1 = {k1:.4f}")
    ax[2, 0].legend(fontsize=8)
    ax[2, 0].set_xlabel("x")

    ax[2, 1].plot(x, er_c, "C0-", lw=1.2)
    for lv in (-np.pi, 0.0, np.pi):
        ax[2, 1].axhline(lv, color="0.6", ls=":", lw=0.9)
    sp = np.pi / k1
    for m_ in range(-60, 61):
        if x[0] < m_ * sp < x[-1]:
            ax[2, 1].axvline(m_ * sp, color="0.88", lw=0.6, zorder=0)
    ax[2, 1].set_title(r"core error, gridlines at $\pi/k_1$, "
                       "dotted at $\pm\pi$")
    ax[2, 1].set_xlabel("x")

    okc = np.isfinite(ph_c)
    if okc.sum() > 4:
        dphi = np.gradient(ph_c[okc], x[okc])
        ax[2, 2].plot(x[okc], np.abs(dphi), "C2-", lw=1.0)
    ax[2, 2].axhline(k1, color="k", ls="--", lw=1.2, label=f"$k_1$={k1:.3f}")
    ax[2, 2].set_ylim(0, 2.0 * max(k1, 0.1))
    ax[2, 2].set_title(r"local $|d\phi/dx|$ on the core")
    ax[2, 2].legend(fontsize=8)
    ax[2, 2].set_xlabel("x")

    fig.suptitle(rf"$\mu={fld['mu']}$   director source: {source}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"a": a, "c": c, "err": err, "phi_aligned": phi_al}


# =====================================================================

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    lines = []

    def report(s=""):
        print(s, flush=True)
        lines.append(s)

    sources = args.sources
    report("=" * 74)
    report("T1: reconstructed knee core phase vs exact, by director source")
    report("=" * 74)
    report("  exact core phase: theta(x, 0) = k1 x + log 2, slope k1")
    report("  residual is phi_reconstructed - exact, minus its median")
    report("  all values in radians; a pi jump reads ~3.14 in max_step")
    report(f"  seed contour: {args.seed}   sigma: {args.sigma:.4f}   "
           f"seeds: {args.n_seeds}   ds: {args.ds}")
    report("")
    report(f"  {'mu':>6} {'source':>16} {'gauge':>12} {'cover':>6} "
           f"{'|slope|':>8} {'k1':>7} {'rms':>8} {'max|r|':>8} "
           f"{'max step':>9} {'pi/k1':>7}")

    store = {}
    for mu in args.mu:
        fld = knee_exact(mu, n_spacings=args.n_spacings, N=args.N)
        ramp = edge_taper(fld["x"], fld["y"], frac=args.taper)
        for src in sources:
            try:
                _, res = run_one(fld, src, args.seed, args.n_seeds,
                                 args.sigma, args.ds, ramp, args.symmetric)
            except Exception as exc:
                report(f"  {mu:>6.3f} {src:>16}   FAILED: {exc}")
                continue
            if args.diagnose:
                dp = os.path.join(
                    args.outdir,
                    f"t1_diag_mu{mu:.3f}_{src}.png".replace(".", "p", 1))
                diagnostic_figure(fld, src, res,
                                  rebuild_grid(res, fld, args.diag_gauge, args.half),
                                  dp, gauge=args.diag_gauge)
                report(f"  {mu:>6.3f} {src:>16}   diagnostic: {dp}")
            frac = res.get("frac_reaching_core", np.nan)
            if frac < 0.9:
                report(f"  {mu:>6.3f} {src:>16}   [warn] only "
                       f"{100 * frac:.0f} percent of lines reach the core")
            for gauge in args.gauges:
                phi = rebuild_grid(res, fld, gauge, args.half)
                sc = score_core(phi, fld, x_trim=args.x_trim)
                store[(mu, src, gauge)] = (fld, sc)
                if "rms" not in sc:
                    report(f"  {mu:>6.3f} {src:>16} {gauge:>12}   too few "
                           f"valid points ({sc['n_ok']})")
                    continue
                report(f"  {mu:>6.3f} {src:>16} {gauge:>12} "
                       f"{sc['coverage']:>6.2f} "
                       f"{abs(sc['slope']):>8.4f} "
                       f"{fld['k1_ff']:>7.4f} {sc['rms']:>8.4f} "
                       f"{sc['max_abs']:>8.4f} {sc['max_step']:>9.4f} "
                       f"{np.pi / fld['k1_ff']:>7.3f}")
        report("")

    # ---- figure: residual on the core row, one panel per mu -----------
    nmu = len(args.mu)
    fig, axes = plt.subplots(nmu, 1, figsize=(9.0, 2.5 * nmu), squeeze=False)
    for i, mu in enumerate(args.mu):
        ax = axes[i, 0]
        spacing = None
        gp = args.gauges[-1]
        for src, cl in zip(sources, ("C0", "C1", "C2", "C3")):
            if (mu, src, gp) not in store:
                continue
            fld, sc = store[(mu, src, gp)]
            if "resid" not in sc:
                continue
            spacing = np.pi / fld["k1_ff"]
            ax.plot(sc["x"], sc["resid"], cl, lw=1.3, label=src)
        if spacing is not None:
            x0 = axes[i, 0].get_xlim()[0]
            for m in range(-40, 41):
                xv = m * spacing
                if x0 < xv < axes[i, 0].get_xlim()[1]:
                    ax.axvline(xv, color="0.85", lw=0.6, zorder=0)
        for lv in (-np.pi, 0.0, np.pi):
            ax.axhline(lv, color="0.6", ls=":", lw=0.8)
        ax.set_ylabel("residual (rad)")
        ax.set_title(rf"$\mu={mu}$   core row, gridlines at $\pi/k_1$",
                     fontsize=9)
        if i == 0 and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, ncol=len(sources))
    axes[-1, 0].set_xlabel("x")
    fig.tight_layout()
    p = os.path.join(args.outdir, "t1_core_phase_residual.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    report(f"figure: {p}")

    with open(os.path.join(args.outdir, "t1_director_source_test.txt"),
              "w") as fh:
        fh.write("\n".join(lines) + "\n")


class _Args:
    outdir = "out_t1_kb"
    mu = [0.85, 0.9, 0.95]
    sources = ["exact", "exact_oriented", "uhu", "riesz"]
    seed = "exact"              # or "trapezoid"
    sigma = np.pi / 2
    ds = 0.15
    n_seeds = 96
    N = 512
    n_spacings = 8.0
    taper = 0.04
    x_trim = 0.15
    symmetric = False
    diagnose = False
    diag_gauge = "seed_phase"
    half = "orig"
    gauges = ["raw", "zero_origin", "seed_phase"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=_Args.outdir)
    p.add_argument("--mu", type=float, nargs="+", default=_Args.mu)
    p.add_argument("--sources", nargs="+", default=_Args.sources,
                   choices=list(SOURCES))
    p.add_argument("--seed", default=_Args.seed,
                   choices=["trapezoid", "exact"])
    p.add_argument("--sigma", type=float, default=_Args.sigma)
    p.add_argument("--ds", type=float, default=_Args.ds)
    p.add_argument("--n-seeds", type=int, default=_Args.n_seeds)
    p.add_argument("--N", type=int, default=_Args.N)
    p.add_argument("--n-spacings", type=float, default=_Args.n_spacings)
    p.add_argument("--taper", type=float, default=_Args.taper)
    p.add_argument("--x-trim", type=float, default=_Args.x_trim)
    p.add_argument("--half", default=_Args.half,
                   choices=["orig", "stitched"])
    p.add_argument("--diag-gauge", default=_Args.diag_gauge,
                   choices=["raw", "zero_origin", "seed_phase"])
    p.add_argument("--diagnose", action="store_true",
                   default=_Args.diagnose)
    p.add_argument("--gauges", nargs="+", default=_Args.gauges,
                   choices=["raw", "zero_origin", "seed_phase"])
    p.add_argument("--symmetric", action="store_true",
                   default=_Args.symmetric)
    main(p.parse_args())