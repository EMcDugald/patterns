# pipelines/run_riesz_zigzag.py
"""
Riesz-based order-parameter extraction for PGB zigzag SH time series.

Reads a standardised SH .npz (produced by run_sh_pgb_zigzag.py), extracts
order parameters frame by frame with the validated Riesz method layer, and
writes a standardised OP .npz plus truth-free figures and two phase GIFs.

This mirrors run_kuhu_zigzag.py and run_phase_zigzag.py, but it reads SH data
directly (no uHu file in between) and injects a Riesz director in place of the
uHu director, exactly the way run_one does in t1_common.

What comes out, per input file
------------------------------
  arrays (Ny, Nx, Nt), float32:
    theta            symmetric unwrapped phase from the pipeline
    k1_direct/k2_direct/kmag_direct/A   direct Riesz k field and amplitude
    k1_grad/k2_grad/kmag_grad           k from grad(theta)
    mutual_dang      direct-vs-grad angle difference (deg), the truth-free
                     per-point trust map
  per-frame scalars (Nt,):
    sigma_dir, sum_abs, n_dislocations, A_min_band, mutual_median
  figures at a frame stride:
    defect index (reused), truth-free k comparison, phase summary
  GIFs:
    <stem>_phase2d.gif        the extracted 2d phase over time
    <stem>_core_profile.gif   the phase along the core row over time

Three adaptations to real cropped SH data (see inline notes)
------------------------------------------------------------
  1. FFT extents are the CROP extents, Lx = Nx_crop*dx and Ly = Ny_crop*dy,
     read from metadata dx/dy. metadata Lx is the full 512-wide domain and
     would put the wrong wavenumbers in.
  2. The core is recentered to y = 0 (y_ext = y - midline) so detect_defects
     and the rest of the t1 stack, which assume the core at y = 0, line up
     with the pipeline's y_shift.
  3. The crop is non-periodic in x and y, so a rectangular ramp tapers u
     before every FFT-based Riesz op. The ramp is 1 in the interior and only
     touches the trimmed rim, so the core and the detection band are
     untouched. The RAW (unramped) u is handed to the phase pipeline, because
     its Hilbert transforms sample the true pattern amplitude along the lines.

Stop and check on first runs
----------------------------
  * Frame 0 of each file: detection should read zero and the core phase
    should be linear with no staircase. If mu = 0.95 (finer dx, sharper knee)
    throws spurious charges at sigma_det = 0.4, raise --sigma_det.
  * Look at the frame-0 phase summary before trusting anything downstream.
  * A staircase in the core profile at t = 0 is the failure the method exists
    to prevent. If it appears, stop.

Seed contour and gauge (defaults)
---------------------------------
The seeds come from a data-driven crest contour by default: a real u crest
traced per frame with the smoothed Riesz director (trace_crest_contour),
anchored frame to frame so the tracer stays on the same material crest.
Theta is constant along a true crest, so the gauge resolves to zero_origin
(exact, no analytic reference). If a trace fails its gate, that frame falls
back to the analytic knee template with the seed_phase gauge, and says so.
Widths default to defect-aware: 0.4 on clean frames, pi/2 on charged ones.

Usage examples
--------------
# Single file (validate first); crest seeds + defect-aware are the defaults
python run_riesz_zigzag.py <path>/sh_pgb_zigzag_mu0.900_T3_N125_nx512_Ny256_lower.npz

# Old behavior: analytic template seeds, one fixed width
python run_riesz_zigzag.py <path>/...lower.npz --seed_contour template \
    --no-defect_aware

# Batch a directory
python run_riesz_zigzag.py --all --input_dir <dir>/raw
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
from scipy.interpolate import RegularGridInterpolator

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

# The method layer is truth. detect_defects, riesz_k_field, k_from_theta,
# k_consistency and defect_index_figure all come from t1_common (the renamed
# t1_common_v5). compute_pgb_phase_from_uhu is the phase pipeline it resolves.
from op_extract.t1_common import (
    detect_defects,
    riesz_k_field,
    k_from_theta,
    k_consistency,
    defect_index_figure,
    macro_iso,
    spectral_deriv,
    compute_uhu_ops_smoother,
    rebuild_grid,
    monogenic,
    riesz_director,
    nan_smooth,
)
from op_extract.phase_zigzag_v3 import compute_pgb_phase_from_uhu
from utils.geometry import build_rectangular_ramp_smooth


# argparse.BooleanOptionalAction only exists from Python 3.9; provide a
# minimal backport so --defect_aware / --no-defect_aware work on 3.8
# (the debug Args block never hits this, which is why it went unnoticed).
if not hasattr(argparse, "BooleanOptionalAction"):
    class _BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None,
                     required=False, help=None, **kwargs):
            opts = []
            for opt in option_strings:
                opts.append(opt)
                if opt.startswith("--"):
                    opts.append("--no-" + opt[2:])
            super().__init__(opts, dest, nargs=0, default=default,
                             required=required, help=help)

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest,
                    not option_string.startswith("--no-"))

    argparse.BooleanOptionalAction = _BooleanOptionalAction


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def reflect_phase_y(theta, core_row, mode="even"):
    """Define the lower half of the phase from the upper half by reflection.

    Only the upper arm is seeded and traced, so rebuild_grid leaves y < 0 as
    NaN. The knee phase is even in y, so the correct lower half is the mirror
    of the upper half about the core (mode="even"). mode="odd" mirrors with a
    sign flip about the core value, for a dislocation-like state; mode="none"
    leaves the one-sided surface as is. The reflection is taken about
    core_row, the true grain-boundary grid row, not the array centre; on an
    even crop those differ by half a cell and mirroring about the wrong one
    leaves a small asymmetry in |k|.

    "even" and "odd" fill only NaN cells, keeping any genuinely
    reconstructed lower-half value. That mix is what produced the visible
    asymmetry in the 2d phase panels: after nucleation the lines pierce the
    core and carry the odd PN branch below it, and at t=0 the thin griddata
    strip below the core is linear interpolation, not an exact mirror.
    mode="mirror" (the default) instead overwrites everything below the
    core row with the mirror of the upper half. Since the pattern is the
    cosine of the phase, the even branch is a legitimate choice, and the
    upper half, which the analysis uses, is untouched. The lower half is
    visualization only in every mode.
    """
    if mode == "none":
        return theta
    Ny = theta.shape[0]
    k = np.arange(Ny)
    midx = 2 * core_row - k
    valid = (midx >= 0) & (midx < Ny)
    mirror = np.full_like(theta, np.nan)
    mirror[valid] = theta[midx[valid]]
    out = theta.copy()
    if mode == "mirror":
        below = k < core_row
        out[below] = mirror[below]
    elif mode == "even":
        fill = ~np.isfinite(out) & np.isfinite(mirror)
        out[fill] = mirror[fill]
    elif mode == "odd":
        core = theta[core_row][None, :]
        odd = 2.0 * core - mirror
        fill = ~np.isfinite(out) & np.isfinite(odd)
        out[fill] = odd[fill]
    else:
        raise ValueError(
            f"reflect_y must be mirror, even, odd, or none, got {mode!r}")
    return out


def build_knee_theta(x, y, mu):
    """Analytic lower-GB knee phase on a grid whose core is at y = 0.

    Same softmax form as the IC in pgb_zigzag. Used only as the seed-phase
    gauge reference when a file has no saved theta_initial.
    """
    k1 = np.sqrt(1.0 - mu ** 2)
    k2 = mu
    X, Y = np.meshgrid(x, y)
    pp = k1 * X + k2 * Y
    pm = k1 * X - k2 * Y
    m = np.maximum(pp, pm)
    return m + np.log(np.exp(pp - m) + np.exp(pm - m))


_LAM = 2.0 * np.pi


def trace_crest_contour(u, x, y, d1, d2, y_start=3.0, y_stop=1.0,
                        x_margin=2.0, y_margin=2.0, ds=0.30, anchor=None,
                        max_steps=4000):
    """Trace one stripe crest of u in the upper half: the data-driven seed
    contour, the SH analog of t1_common.build_seed_contour. On a true crest
    theta is constant, so gauge="zero_origin" is exact and the analytic knee
    template drops out of the seeding entirely.

    Returns (con, p0): con is a (2, N) contour ordered from its low-right
    end (descending branch, nearest the core) to where the ascending branch
    leaves the trimmed box; p0 is the start point for frame-to-frame
    anchoring. Returns (None, p0_or_None) when the trace fails its gate.

    Start: one wavelength in from the right edge at y_start, snapped to the
    nearest crest (window one wavelength, so a crossing is always inside the
    unmasked segment). Two branches from the same start: ascending (tangent
    oriented ty > 0) into the far field, descending (ty < 0) toward the
    apex, stopped at y_stop or when u on the crest degrades (a pinch).
    anchor is the previous frame's p0; it keeps the tracer on the same
    material crest across frames (quarter-wavelength window), so the
    zero_origin gauge does not jump by 2 pi between frames.
    """
    iu = RegularGridInterpolator((y, x), u, bounds_error=False,
                                 fill_value=np.nan)
    id1 = RegularGridInterpolator((y, x), d1, bounds_error=False,
                                  fill_value=np.nan)
    id2 = RegularGridInterpolator((y, x), d2, bounds_error=False,
                                  fill_value=np.nan)
    x_lo, x_hi = x[0] + x_margin, x[-1] - x_margin
    y_hi = y[-1] - y_margin
    y_floor = y_stop

    def director_at(p):
        v = np.array([float(id1((p[1], p[0]))), float(id2((p[1], p[0])))])
        n = np.hypot(*v)
        return v / n if np.isfinite(n) and n > 0 else None

    def snap(p, window_lam, floor):
        """Nearest interior local max of u along the director line through
        p. A global argmax would jump to the brightest crest crossing in
        the window (or pin at the mask edge, including across the midline);
        the crest wanted is the one the point is already on, so take the
        local max closest to s = 0, parabolic-refined. floor is the lowest
        admissible y: the START snap must use y_start, because a start
        crest inside the fold zone (width 1/k2) bends onto the fold line
        and both branches then creep along the midline; branch correctors
        use y_stop."""
        d = director_at(p)
        if d is None:
            return None
        s = np.linspace(-window_lam * _LAM, window_lam * _LAM,
                        int(80 * window_lam) + 1)
        pts = p[None, :] + s[:, None] * d[None, :]
        v = iu(np.column_stack([pts[:, 1], pts[:, 0]]))
        inbox = ((pts[:, 1] >= floor) & (pts[:, 1] <= y_hi)
                 & (pts[:, 0] >= x_lo) & (pts[:, 0] <= x_hi))
        v = np.where(inbox, v, np.nan)
        loc = np.zeros(len(s), dtype=bool)
        loc[1:-1] = (np.isfinite(v[1:-1]) & np.isfinite(v[:-2])
                     & np.isfinite(v[2:])
                     & (v[1:-1] >= v[:-2]) & (v[1:-1] >= v[2:]))
        if not loc.any():
            return None
        cand = np.flatnonzero(loc)
        i = int(cand[np.argmin(np.abs(s[cand]))])
        denom = v[i - 1] - 2 * v[i] + v[i + 1]
        off = 0.5 * (v[i - 1] - v[i + 1]) / denom if denom < 0 else 0.0
        off = float(np.clip(off, -1.0, 1.0)) * (s[1] - s[0])
        return pts[i] + off * d

    if anchor is not None:
        p0 = snap(np.asarray(anchor, dtype=float), 0.25, y_floor)
    else:
        p0 = snap(np.array([x_hi - _LAM, y_start]), 1.0, y_start)
    if p0 is None:
        return None, None
    u0 = float(iu((p0[1], p0[0])))

    def branch(sign):
        pts = []
        p = p0.copy()
        for _ in range(max_steps):
            d = director_at(p)
            if d is None:
                break
            t = np.array([-d[1], d[0]])
            if sign * t[1] < 0 or (t[1] == 0 and t[0] > 0):
                t = -t
            q = snap(p + ds * t, 0.25, y_floor)
            if q is None:
                break
            if not (x_lo <= q[0] <= x_hi and y_floor <= q[1] <= y_hi):
                break
            uq = float(iu((q[1], q[0])))
            if np.isfinite(uq) and uq < 0.5 * u0:
                break                      # crest degrading (pinch), stop
            pts.append(q)
            p = q
        return pts

    up = branch(+1.0)
    down = branch(-1.0)
    con = np.asarray(list(reversed(up)) + [p0.tolist()] + down).T
    con = con[:, ::-1]                     # low-right end first

    vals = iu(np.column_stack([con[1], con[0]]))
    med = np.nanmedian(vals)
    y_span = float(con[1].max() - con[1].min())
    if (con.shape[1] < 20 or med <= 0 or np.nanmin(vals) < 0.5 * med
            or y_span < 1.5 * _LAM):
        # the span test catches a contour that never left the fold zone
        # (a midline creep); such a contour passes the length and
        # amplitude tests but seeds a degenerate reconstruction
        return None, p0
    return con, p0


def load_sh_npz(sh_path):
    """
    Load a standardised SH .npz and return arrays with a recentered grid.

    The grid comes back centered on the core: x is already centered on the
    knee apex, and y is shifted so the midline core sits at y = 0. dx and dy
    are read from metadata, never from diff(x), because a corrupted first
    coordinate would otherwise poison the spacing.
    """
    sh_path = Path(sh_path)
    data = np.load(sh_path, allow_pickle=True)

    u = data["u"]
    if u.ndim != 3:
        raise ValueError(f"{sh_path.name}: expected u (Ny, Nx, Nt), got {u.shape}")

    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)

    # guard: reconcile (Ny, Nx, Nt) vs (Nt, Ny, Nx)
    if u.shape[0] == len(y) and u.shape[1] == len(x):
        pass
    elif u.shape[1] == len(y) and u.shape[2] == len(x):
        u = np.transpose(u, (1, 2, 0))
    else:
        raise ValueError(
            f"{sh_path.name}: cannot reconcile u {u.shape} with "
            f"x ({len(x)},) y ({len(y)},)"
        )

    tt = data["tt"] if "tt" in data else None

    meta = None
    if "metadata_json" in data:
        raw = data["metadata_json"]
        meta = json.loads(raw.item() if hasattr(raw, "item") else str(raw))

    Ny, Nx, Nt = u.shape

    # dx, dy from metadata; fall back to a robust median of diff if absent
    if meta is not None and "dx" in meta and "dy" in meta:
        dx = float(meta["dx"])
        dy = float(meta["dy"])
    else:
        dx = float(np.median(np.diff(x)))
        dy = float(np.median(np.diff(y)))

    mu = float(meta["mu"]) if (meta is not None and "mu" in meta) else None
    lambda_x = float(meta["lambda_x"]) if (meta is not None and "lambda_x" in meta) \
        else (np.pi / np.sqrt(1.0 - mu ** 2) if mu is not None else np.nan)

    # core row: the recorded GB grid row puts the true core on a grid line.
    # Recenter y on it so the core sits at y = 0 exactly and the reflection
    # axis is that row, not the half-cell-off array centre.
    if meta is not None and "gb_row_in_crop" in meta:
        core_row = int(meta["gb_row_in_crop"])
    else:
        core_row = int(np.argmin(np.abs(y - 0.5 * (y[0] + y[-1]))))

    x_ext = x - 0.5 * (x[0] + x[-1])
    y_ext = y - y[core_row]

    # crop extents for the FFT ops (NOT the full-domain Lx in metadata)
    Lx = Nx * dx
    Ly = Ny * dy

    # gauge reference for the seed-phase reconstruction: the analytic knee
    # phase. Prefer the saved theta_initial (already on this crop grid); fall
    # back to building it from mu. The seeds sit in the far field, so this
    # fixes the per-line integration constant without using evolved truth.
    if "theta_initial" in data and np.asarray(data["theta_initial"]).ndim == 2:
        theta_ref = np.asarray(data["theta_initial"], dtype=float)
    elif mu is not None:
        theta_ref = build_knee_theta(x_ext, y_ext, mu)
    else:
        theta_ref = None

    return {
        "u": u, "x": x_ext, "y": y_ext, "tt": tt, "mu": mu,
        "dx": dx, "dy": dy, "Lx": Lx, "Ly": Ly,
        "lambda_x": lambda_x, "core_row": core_row, "theta_ref": theta_ref,
        "meta": meta, "source_path": str(sh_path),
    }


def build_out_stem(sh_path, cfg):
    sh_path = Path(sh_path)
    tag = "defectaware" if cfg["defect_aware"] else f"sig{cfg['sigma_fixed']:.2f}"
    return f"{sh_path.stem}_riesz_{tag}_ds{cfg['ds']:.3f}_ns{int(cfg['n_phase_seeds'])}"


# ---------------------------------------------------------------------------
# Per-frame extraction
# ---------------------------------------------------------------------------

def extract_one_frame(u_frame, sh, ramp, cfg, crest_state=None):
    """
    One frame: detect, pick width, direct k, phase, grad(theta) k, mutual map.

    crest_state is a per-series mutable dict carrying the crest anchor
    across frames. Seed contour resolution when seed_contour="crest":
    anchored trace, then a fresh trace, then the analytic template, in that
    order, each fallback reported. The gauge follows the contour when
    gauge="auto": zero_origin on a crest (exact, theta constant along a
    crest), seed_phase on the template (needs the analytic reference).

    Returns a dict of 2d fields and scalars, plus the detection record for the
    figure. Never raises on a single-frame failure; on error it returns NaN
    fields and marks ok=False so the series keeps going.
    """
    x, y = sh["x"], sh["y"]
    Lx, Ly = sh["Lx"], sh["Ly"]
    mu = sh["mu"]

    band_y = cfg["band_y"]
    band_x = (1.0 - cfg["band_x_frac"]) * 0.5 * Lx  # keep the inner (1-frac)

    u_ramped = u_frame * ramp
    nan2d = np.full_like(u_frame, np.nan, dtype=float)

    try:
        # 1. detection and width selection on the ramped field
        det = detect_defects(u_ramped, x, y, Lx, Ly,
                             sigma_det=cfg["sigma_det"],
                             band_y=band_y, band_x=band_x)
        if cfg["defect_aware"] and det["sum_abs"] > 0:
            sigma_dir = cfg["sigma_large"]
        else:
            sigma_dir = cfg["sigma_fixed"]

        # 2. direct Riesz k field and amplitude at the selected width
        kd = riesz_k_field(u_ramped, x, y, Lx, Ly, sigma_dir)
        use_uhu = (cfg["director"] == "uhu"
                   or (cfg["director"] == "aware" and det["sum_abs"] > 0))
        if use_uhu:
            # A/B test: uHu eigenvector director through the identical
            # seeding and reconstruction. uHu needs sigma |k| of order
            # one, so its width is floored at sigma_uhu_min even on
            # clean frames where Riesz runs at 0.4.
            sg_u = max(float(sigma_dir), float(cfg["sigma_uhu_min"]))
            sm = lambda a, s_=sg_u: macro_iso(a, s_, Lx, Ly)
            ops = compute_uhu_ops_smoother(u_frame, x, y, sm,
                                           spectral_deriv, ramp)
            ku1 = np.asarray(ops["k1_orig"])
            ku2 = np.asarray(ops["k2_orig"])
            nrm = np.hypot(ku1, ku2)
            nrm = np.where(nrm > 0, nrm, 1.0)
            d1a, d2a = ku1 / nrm, ku2 / nrm
        else:
            d1a = np.asarray(kd["d1"])
            d2a = np.asarray(kd["d2"])

        # 2b. seed contour: data-driven crest by default, template fallback.
        # The trace uses RAW u (the crest is a feature of the true pattern)
        # and the director at the selected width.
        contour_source = "template"
        knee_bdry = None
        if crest_state is None:
            crest_state = {}
        if cfg["seed_contour"] == "crest":
            kw = dict(y_start=cfg["crest_y_start"], y_stop=cfg["crest_y_stop"],
                      x_margin=cfg["xmargin"] * Lx + 1.0,
                      y_margin=cfg["ymargin"] * Ly + 1.0, ds=cfg["ds"])
            con, p0 = trace_crest_contour(u_frame, x, y, d1a, d2a,
                                          anchor=crest_state.get("anchor"),
                                          **kw)
            if con is None and crest_state.get("anchor") is not None:
                print("    [warn] anchored crest trace failed; "
                      "retrying from a fresh start")
                con, p0 = trace_crest_contour(u_frame, x, y, d1a, d2a,
                                              anchor=None, **kw)
            if con is not None:
                knee_bdry = con
                contour_source = "crest"
                crest_state["anchor"] = p0
            else:
                print("    [warn] crest trace failed its gate; "
                      "falling back to the analytic template this frame")

        gauge = cfg["gauge"]
        if gauge == "auto":
            gauge = "zero_origin" if contour_source == "crest" \
                else "seed_phase"

        # 3. phase pipeline with the Riesz director injected, RAW u sampled
        d1 = d1a[:, :, None]
        d2 = d2a[:, :, None]
        uhu_data = {
            "x": x, "y": y, "u": u_frame[:, :, None],
            "k1_sym": d1, "k2_sym": d2, "k1_orig": d1, "k2_orig": d2,
            "mu": mu, "tt": None, "ramp": None, "k": None, "A": None,
            "knee_bdry": knee_bdry,  # None -> analytic knee from mu
            "source_path": sh["source_path"],
        }
        res = compute_pgb_phase_from_uhu(
            uhu_data, mu=mu,
            n_phase_seeds=cfg["n_phase_seeds"],
            seed_half=cfg["seed_half"],
            ds=cfg["ds"], max_steps=cfg["max_steps"],
            prefer_sym=True, phase_ramp_mode="none",
            grid_output=False,
        )
        # Rebuild from the seed-to-core branch of each line, the way the t1
        # harness does. The pipeline's own stitched grid blends the reflected
        # half with the wrong sign and flattens the core; that is what made the
        # core profile flat. seed_phase fixes each line's constant from the
        # analytic far-field knee phase at its seed. odd_in_y=False triggers the
        # mirror that keeps the y=0 core row inside the interpolation hull.
        fld = {"x": x, "y": y, "theta": sh["theta_ref"], "odd_in_y": False}
        theta = rebuild_grid(res, fld, gauge=gauge, half="orig",
                             band=cfg["ds"])
        # the upper arm is the only one traced, so define the lower half by
        # reflection across the core (even for the knee)
        theta = reflect_phase_y(theta, sh["core_row"], mode=cfg["reflect_y"])

        # 4. k from grad(theta), then the truth-free mutual comparison
        kg = k_from_theta(theta, x, y, Lx, Ly, smooth=cfg["grad_smooth"])
        cons = k_consistency(kd, kg)

        # 5. amplitude observables. kd["A"] is sqrt(2 <u^2>) at sigma_avg
        # (of order pi) and barely registers cores: measured 2 percent core
        # dips at the final frame against 23 percent for the pointwise
        # monogenic amplitude. A_mono = sqrt(u^2 + R1^2 + R2^2) is the
        # nucleation field: it dips at cores but stays smooth and nonzero.
        # A_min_band tracks its band minimum; A_far_median (same field, a
        # far-field band) is saved so the normalized dip
        # A_min_band / A_far_median is available at analysis time, immune
        # to the overall amplitude relaxation of the run.
        mono = monogenic(u_ramped, Lx, Ly)
        A_mono = np.asarray(mono["A"])
        X, Y = np.meshgrid(x, y)
        band = (np.abs(Y) < band_y) & (np.abs(X) < band_x)
        far = ((np.abs(Y) > 3.0 * band_y) & (np.abs(Y) < 6.0 * band_y)
               & (np.abs(X) < band_x))
        A_min_band = float(np.nanmin(A_mono[band])) if band.any() else np.nan
        A_far_median = float(np.nanmedian(A_mono[far])) if far.any() \
            else np.nan

        return {
            "ok": True, "det": det, "sigma_dir": float(sigma_dir),
            "contour_source": contour_source, "gauge": gauge,
            "theta": theta.astype(np.float32),
            "k1_direct": np.asarray(kd["k1"], dtype=np.float32),
            "k2_direct": np.asarray(kd["k2"], dtype=np.float32),
            "kmag_direct": np.asarray(kd["kmag"], dtype=np.float32),
            "A": np.asarray(kd["A"], dtype=np.float32),
            "A_mono": A_mono.astype(np.float32),
            "k1_grad": np.asarray(kg["k1"], dtype=np.float32),
            "k2_grad": np.asarray(kg["k2"], dtype=np.float32),
            "kmag_grad": np.asarray(kg["kmag"], dtype=np.float32),
            "mutual_dang": np.asarray(cons["dang"], dtype=np.float32),
            "sum_abs": int(det["sum_abs"]),
            "n_dislocations": int(det["n_dislocations"]),
            "A_min_band": A_min_band,
            "A_far_median": A_far_median,
            "mutual_median": float(cons.get("dang_med_abs", np.nan)),
            "knee_bdry_phase": np.asarray(res.get("knee_bdry_phase")),
            "coordinate_lines": res["coordinate_lines"][0],
            "charges": [[float(c[0]), float(c[1]), int(c[2])]
                        for c in det["charges"]],
        }
    except Exception as exc:  # keep the series alive; report the frame
        print(f"    [warn] frame extraction failed: {exc}")
        return {
            "ok": False, "det": None, "sigma_dir": np.nan,
            "contour_source": "failed", "gauge": "none",
            "theta": nan2d.astype(np.float32),
            "k1_direct": nan2d.astype(np.float32),
            "k2_direct": nan2d.astype(np.float32),
            "kmag_direct": nan2d.astype(np.float32),
            "A": nan2d.astype(np.float32),
            "A_mono": nan2d.astype(np.float32),
            "k1_grad": nan2d.astype(np.float32),
            "k2_grad": nan2d.astype(np.float32),
            "kmag_grad": nan2d.astype(np.float32),
            "mutual_dang": nan2d.astype(np.float32),
            "sum_abs": -1, "n_dislocations": -1,
            "A_min_band": np.nan, "A_far_median": np.nan,
            "mutual_median": np.nan,
            "knee_bdry_phase": None, "coordinate_lines": None, "charges": [],
        }


def run_riesz_series(sh_path, out_path, cfg):
    """Extract the whole time series from one SH file and save the OP .npz."""
    sh = load_sh_npz(sh_path)
    u = sh["u"]
    Ny, Nx, Nt = u.shape
    x, y = sh["x"], sh["y"]

    print(f"  {Path(sh_path).name}: Ny={Ny} Nx={Nx} Nt={Nt} "
          f"mu={sh['mu']} dx={sh['dx']:.4f} Lx={sh['Lx']:.2f} Ly={sh['Ly']:.2f}")

    ramp = build_rectangular_ramp_smooth(
        x, y, xmargin=cfg["xmargin"], ymargin=cfg["ymargin"],
        tanhscale=cfg["tanhscale"],
    )

    stacks = {k: np.empty((Ny, Nx, Nt), dtype=np.float32) for k in
              ("theta", "k1_direct", "k2_direct", "kmag_direct", "A",
               "A_mono", "k1_grad", "k2_grad", "kmag_grad", "mutual_dang")}
    sigma_dir = np.empty(Nt)
    sum_abs = np.empty(Nt, dtype=int)
    ndis = np.empty(Nt, dtype=int)
    A_min = np.empty(Nt)
    A_far = np.empty(Nt)
    mutual_med = np.empty(Nt)
    charges_per_frame = []
    contour_sources = []
    dets = {}  # keep detection records only for the frames we will plot
    snaps = {}  # seed contour and coordinate lines for the same frames
    crest_state = {}  # carries the crest anchor from frame to frame

    fig_frames = _fig_frame_set(Nt)

    t_start = time.time()
    for t in range(Nt):
        f = extract_one_frame(u[:, :, t], sh, ramp, cfg,
                              crest_state=crest_state)
        for k in stacks:
            stacks[k][:, :, t] = f[k]
        sigma_dir[t] = f["sigma_dir"]
        sum_abs[t] = f["sum_abs"]
        ndis[t] = f["n_dislocations"]
        A_min[t] = f["A_min_band"]
        A_far[t] = f["A_far_median"]
        mutual_med[t] = f["mutual_median"]
        charges_per_frame.append(f["charges"])
        contour_sources.append(f["contour_source"])
        if t in fig_frames and f["ok"]:
            dets[t] = f["det"]
            snaps[t] = {"knee_bdry_phase": f["knee_bdry_phase"],
                        "coordinate_lines": f["coordinate_lines"],
                        "sigma_dir": f["sigma_dir"]}
        if (t + 1) % 10 == 0 or t == Nt - 1:
            dt = time.time() - t_start
            print(f"    frame {t+1}/{Nt}  n_dis={f['n_dislocations']}  "
                  f"sigma={f['sigma_dir']:.2f}  seeds={f['contour_source']}  "
                  f"mutual_med={f['mutual_median']:.2f}  "
                  f"[{dt / (t + 1):.2f} s/frame]")

    riesz_meta = {
        "sigma_fixed": cfg["sigma_fixed"], "defect_aware": cfg["defect_aware"],
        "sigma_large": cfg["sigma_large"], "sigma_det": cfg["sigma_det"],
        "band_y": cfg["band_y"], "band_x_frac": cfg["band_x_frac"],
        "ds": cfg["ds"], "n_phase_seeds": cfg["n_phase_seeds"],
        "seed_half": cfg["seed_half"], "grad_smooth": cfg["grad_smooth"],
        "seed_contour": cfg["seed_contour"], "gauge": cfg["gauge"],
        "crest_y_start": cfg["crest_y_start"],
        "crest_y_stop": cfg["crest_y_stop"],
        "reflect_y": cfg["reflect_y"],
        "xmargin": cfg["xmargin"], "ymargin": cfg["ymargin"],
        "tanhscale": cfg["tanhscale"], "core_row": sh["core_row"],
        "source_file": str(sh_path),
    }


    # theta_smooth: lightly smoothed display copy of theta, width
    # theta_smooth_w (default 0.75), below the pi/k1 plateau scale (~7)
    # and the riser width (1 to 2), so the staircase steps survive.
    # Stored as its OWN stack; raw theta is never replaced. Used only by
    # the core-profile GIF, the phase-summary profile panels, and the
    # downstream metrics runner.
    w_sm = float(cfg.get("theta_smooth_w", 0.75))
    stacks["theta_smooth"] = np.empty((Ny, Nx, Nt), dtype=np.float32)
    for t_ in range(Nt):
        ts_ = nan_smooth(stacks["theta"][:, :, t_].astype(float),
                         w_sm, sh["Lx"], sh["Ly"])
        # the normalized convolution extends past the traced hull; mask
        # back to the raw support so the display stack never contains
        # values where the reconstruction has none
        ts_[~np.isfinite(stacks["theta"][:, :, t_])] = np.nan
        stacks["theta_smooth"][:, :, t_] = ts_.astype(np.float32)

    # truth-free shift-periodicity residual: theta(x) - theta(x + pi/k1)
    # should equal pi for the knee, the PN chain, and any intermediate,
    # so median |D - pi| along the profile row is an error meter that
    # needs no analytic reference and works at every frame
    k1_ = float(np.sqrt(1.0 - float(sh["mu"]) ** 2))
    dy_ = float(sh["y"][1] - sh["y"][0])
    row_ = sh["core_row"] + max(1, int(round(cfg["profile_y"] / dy_)))
    n_sh = int(round((np.pi / k1_) / (sh["x"][1] - sh["x"][0])))
    shift_residual = np.full(Nt, np.nan)
    # interior variant: same statistic on the same raw theta, but with
    # band_x_frac trimmed off each x edge BEFORE the shift, so the ramp
    # rim and the coverage-limited right end never enter. This is the
    # number the paper claims; the full-row scalar is kept for
    # continuity with earlier runs.
    shift_residual_int = np.full(Nt, np.nan)
    i_trim = int(round(cfg["band_x_frac"] * Nx))
    if 0 < n_sh < Nx:
        for t_ in range(Nt):
            p_ = stacks["theta"][row_, :, t_]
            D = p_[:-n_sh] - p_[n_sh:]
            fin = np.isfinite(D)
            if fin.any():
                shift_residual[t_] = float(
                    np.nanmedian(np.abs(D[fin] - np.pi)))
            pi_ = p_[i_trim:Nx - i_trim] if i_trim > 0 else p_
            if pi_.size > n_sh:
                Di = pi_[:-n_sh] - pi_[n_sh:]
                fi_ = np.isfinite(Di)
                if fi_.any():
                    shift_residual_int[t_] = float(
                        np.nanmedian(np.abs(Di[fi_] - np.pi)))
    print(f"    shift-periodicity residual (median |D - pi|, rad): "
          f"f0 {shift_residual[0]:.3f}  fLast {shift_residual[-1]:.3f}  "
          f"(interior: f0 {shift_residual_int[0]:.3f}  "
          f"fLast {shift_residual_int[-1]:.3f})")

    save_dict = {
        "x": x, "y": y,
        "tt": sh["tt"] if sh["tt"] is not None else np.array([]),
        "mu": np.nan if sh["mu"] is None else sh["mu"],
        "core_row": sh["core_row"],
        "sigma_dir": sigma_dir, "sum_abs": sum_abs,
        "n_dislocations": ndis, "A_min_band": A_min,
        "A_far_median": A_far, "shift_residual": shift_residual,
        "shift_residual_int": shift_residual_int,
        "theta_smooth_w": np.float64(w_sm),
        "mutual_median": mutual_med,
        "riesz_meta_json": json.dumps(riesz_meta),
        "charges_json": json.dumps(charges_per_frame),
        "contour_source": np.array(contour_sources),
    }
    if sh["meta"] is not None:
        save_dict["sh_meta_json"] = json.dumps(sh["meta"])
    save_dict.update(stacks)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **save_dict)
    print(f"  saved -> {out_path}")
    return sh, stacks, dets, snaps, ramp


# ---------------------------------------------------------------------------
# Figures (truth free)
# ---------------------------------------------------------------------------

def _fig_frame_set(Nt):
    """Exactly three figure frames: first, one tenth through, last.
    Stacks and GIFs still cover every frame."""
    return {t for t in (0, Nt // 10, Nt - 1) if 0 <= t < Nt}


def _robust_range(stack, lo=2.0, hi=98.0):
    finite = stack[np.isfinite(stack)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.nanpercentile(finite, lo)), float(np.nanpercentile(finite, hi))


def make_defect_index_figure(sh, u_frame, det, path, t, mu):
    """Reuse the method-layer defect index figure via a minimal field dict.

    On real data there is no ground truth, so nd is None and xd is empty; the
    figure's ", defect free" tag refers to that absence of truth, while the
    DETECTED dislocation count in the title is the real signal.
    """
    fld = {
        "u": u_frame, "x": sh["x"], "y": sh["y"],
        "field": f"zigzag SH (frame {t})",
        "mu": mu, "nd": None, "xd": [], "Lsp": sh["lambda_x"],
    }
    defect_index_figure(fld, det, path, zoom_pairs=2.0)


def make_kfield_figure_truthfree(sh, stacks, path, t, kbar_range,
                                 a_range=None):
    """Direct |k|, grad(theta) |k|, their ratio, the mutual angle map, the
    monogenic amplitude A, and A along the core band.

    The mutual map is the per-point trust check that survives without
    truth. The amplitude panels are there because A dips near cores but
    stays smooth and nonzero; the dip, not the phase, is the continuous
    signal before topology changes, and A_min_band tracks its minimum.
    """
    x, y = sh["x"], sh["y"]
    ext = [x[0], x[-1], y[0], y[-1]]
    kd = stacks["kmag_direct"][:, :, t]
    kg = stacks["kmag_grad"][:, :, t]
    dang = stacks["mutual_dang"][:, :, t]
    A = stacks["A_mono"][:, :, t]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(np.isfinite(kg), kg / np.maximum(kd, 1e-9), np.nan)

    lo, hi = kbar_range
    a_lo, a_hi = a_range if a_range is not None else _robust_range(A)
    band_y = sh_cfg_band(sh)
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    panels = (
        (ax[0, 0], kd, "|k| direct Riesz", "viridis", lo, hi),
        (ax[0, 1], kg, "|k| from grad(theta)", "viridis", lo, hi),
        (ax[0, 2], A, "monogenic amplitude A (pointwise)", "magma",
         a_lo, a_hi),
        (ax[1, 0], ratio, "|k| grad / direct", "RdBu_r", 0.9, 1.1),
        (ax[1, 1], dang, "direct vs grad angle (deg), truth free",
         "coolwarm", -20, 20),
    )
    for axis, f_, ttl, cm, vlo, vhi in panels:
        im = axis.imshow(f_, extent=ext, origin="lower", cmap=cm,
                         vmin=vlo, vmax=vhi, aspect="auto")
        fig.colorbar(im, ax=axis, shrink=0.85)
        axis.axhline(0.0, color="k", lw=0.6, ls="--", alpha=0.6)
        for s in (band_y, -band_y):
            axis.axhline(s, color="orange", lw=0.7, ls=":", alpha=0.7)
        axis.set_title(ttl, fontsize=10)

    axp = ax[1, 2]
    core = sh["core_row"]
    drow = max(1, int(round(band_y / (y[1] - y[0]))))
    # band-edge rows first, the core row last so it draws on top
    for row, lbl, styl, lw in (
            (core + drow, f"y={y[core + drow]:+.2f}", "C1--", 1.0),
            (core - drow, f"y={y[core - drow]:+.2f}", "C2--", 1.0),
            (core, f"y={y[core]:+.2f} (core)", "C0-", 1.5)):
        if 0 <= row < len(y):
            axp.plot(x, A[row, :], styl, lw=lw, label=lbl)
    axp.set_ylim(0.0, a_hi * 1.05)
    axp.set_xlim(x[0], x[-1])
    axp.grid(alpha=0.3)
    axp.legend(fontsize=8)
    axp.set_xlabel("x")
    axp.set_title("A along the core and band edges", fontsize=10)

    med = float(np.nanmedian(np.abs(dang)))
    fig.suptitle(f"mu={sh['mu']}  frame {t}  mutual median {med:.2f} deg",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def sh_cfg_band(sh):
    # a light helper so figures can draw the same band the detector used
    return 2.0


def make_riesz_panels_figure(sh, u_ramped, sigma_dir, path, t):
    """The Riesz construction, truth-free: R1, R2, raw orientation, the
    smoothed director angle at the selected width, its alignment magnitude,
    and the pattern with the averaged director quiver. This mirrors the t1
    riesz panels but replaces the exact-error tile (no truth here) with the
    director over the pattern.
    """
    x, y = sh["x"], sh["y"]
    Lx, Ly = sh["Lx"], sh["Ly"]
    ext = [x[0], x[-1], y[0], y[-1]]
    mono = monogenic(u_ramped, Lx, Ly)
    rd = riesz_director(u_ramped, Lx, Ly, sigma_dir)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(2, 3, figsize=(17.0, 9.0))
    v = float(np.nanpercentile(np.abs(mono["R1"]), 99))
    panels = (
        (ax[0, 0], mono["R1"], r"$R_1$", "RdBu_r", -v, v),
        (ax[0, 1], mono["R2"], r"$R_2$", "RdBu_r", -v, v),
        (ax[0, 2], np.degrees(mono["orientation"]),
         "raw Riesz orientation (deg)", "twilight", -180, 180),
        (ax[1, 0], np.degrees(rd["angle"]),
         f"smoothed director angle, sigma_dir={sigma_dir:.2f} (deg)",
         "twilight", -90, 90),
        (ax[1, 1], rd["alignment"], "alignment magnitude", "viridis", 0, 1),
    )
    for axis, f_, ttl, cm, lo, hi in panels:
        im = axis.imshow(f_, extent=ext, origin="lower", cmap=cm,
                         vmin=lo, vmax=hi, aspect="auto")
        fig.colorbar(im, ax=axis)
        axis.axhline(0.0, color="k", lw=0.6, ls="--", alpha=0.6)
        axis.set_title(ttl, fontsize=10)

    umax = np.nanmax(np.abs(u_ramped))
    umax = 1.0 if not np.isfinite(umax) or umax == 0 else umax
    ax[1, 2].imshow(u_ramped, extent=ext, origin="lower", cmap="gray",
                    vmin=-umax, vmax=umax, aspect="auto", alpha=0.85)
    s = 26
    for sign in (1.0, -1.0):
        ax[1, 2].quiver(X[::s, ::s], Y[::s, ::s],
                        sign * rd["d1"][::s, ::s], sign * rd["d2"][::s, ::s],
                        color="cyan", scale=32, headwidth=2, pivot="mid",
                        alpha=0.8)
    ax[1, 2].set_title("pattern + averaged director", fontsize=10)
    fig.suptitle(f"mu={sh['mu']}  frame {t}: Riesz construction, "
                 f"sigma_dir={sigma_dir:.3f}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def make_lines_figure(sh, u_frame, snap, path, t):
    """Two panels on the pattern: the seed contour (knee boundary) used for
    phase seeding, and the coordinate lines traced from the averaged director.
    """
    x, y = sh["x"], sh["y"]
    ext = [x[0], x[-1], y[0], y[-1]]
    umax = np.nanmax(np.abs(u_frame))
    umax = 1.0 if not np.isfinite(umax) or umax == 0 else umax

    fig, ax = plt.subplots(1, 2, figsize=(15.0, 6.0))
    for axis in ax:
        axis.imshow(u_frame, extent=ext, origin="lower", cmap="gray",
                    vmin=-umax, vmax=umax, aspect="auto", alpha=0.85)
        axis.axhline(0.0, color="orange", lw=0.7, ls="--", alpha=0.7)
        axis.set_xlim(x[0], x[-1]); axis.set_ylim(y[0], y[-1])
        axis.set_xticks([]); axis.set_yticks([])

    kb = np.asarray(snap["knee_bdry_phase"]) if snap["knee_bdry_phase"] \
        is not None else None
    if kb is not None and kb.ndim == 2 and kb.shape[0] == 2:
        ax[0].plot(kb[0], kb[1], ".", color="lime", ms=3, alpha=0.9)
    ax[0].set_title(f"seed contour (knee boundary), frame {t}", fontsize=10)

    lines = snap["coordinate_lines"]
    lines = [] if lines is None else list(lines)
    stride = max(1, len(lines) // 40)
    for ln in lines[::stride]:
        ln = np.asarray(ln, dtype=float)
        if ln.size:
            ax[1].plot(ln[:, 0], ln[:, 1], "-", color="cyan", lw=0.6,
                       alpha=0.85)
    ax[1].set_title(f"coordinate lines from averaged director, frame {t}",
                    fontsize=10)
    fig.suptitle(f"mu={sh['mu']}  frame {t}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def make_theta_figure(sh, theta, path, t):
    """theta itself, plus an explicit symmetry check: the field, the residual
    theta minus its y-mirror about the core, and vertical profiles at a few x.
    For an even knee phase the residual should be near zero and the profiles
    should fold onto themselves about y = 0.
    """
    x, y = sh["x"], sh["y"]
    core = sh["core_row"]
    ext = [x[0], x[-1], y[0], y[-1]]

    Ny = theta.shape[0]
    midx = 2 * core - np.arange(Ny)
    valid = (midx >= 0) & (midx < Ny)
    mirror = np.full_like(theta, np.nan)
    mirror[valid] = theta[midx[valid]]
    asym = theta - mirror

    fig, ax = plt.subplots(1, 3, figsize=(18.0, 5.5))

    fin = np.isfinite(theta)
    lo, hi = (np.nanpercentile(theta[fin], [2, 98]) if fin.any() else (0.0, 1.0))
    im0 = ax[0].imshow(np.ma.masked_invalid(theta), extent=ext, origin="lower",
                       cmap="twilight", vmin=lo, vmax=hi, aspect="auto")
    ax[0].axhline(0.0, color="k", lw=0.6, ls="--", alpha=0.6)
    ax[0].set_title(f"theta (frame {t})")
    fig.colorbar(im0, ax=ax[0], shrink=0.85)

    fa = np.isfinite(asym)
    va = float(np.nanpercentile(np.abs(asym[fa]), 98)) if fa.any() else 1.0
    va = va if va > 0 else 1.0
    im1 = ax[1].imshow(np.ma.masked_invalid(asym), extent=ext, origin="lower",
                       cmap="RdBu_r", vmin=-va, vmax=va, aspect="auto")
    ax[1].axhline(0.0, color="k", lw=0.6, ls="--", alpha=0.6)
    ax[1].set_title("theta minus its y-mirror (should be ~0 if symmetric)")
    fig.colorbar(im1, ax=ax[1], shrink=0.85)

    ncol = theta.shape[1]
    for j in (int(0.3 * ncol), int(0.5 * ncol), int(0.7 * ncol)):
        ax[2].plot(y, theta[:, j], lw=1.0, label=f"x={x[j]:.1f}")
    ax[2].axvline(0.0, color="k", lw=0.6, ls="--", alpha=0.6)
    ax[2].set_xlabel("y"); ax[2].set_title("theta vs y at fixed x")
    ax[2].grid(alpha=0.3); ax[2].legend(fontsize=8)

    fig.suptitle(f"mu={sh['mu']}  frame {t}: phase and symmetry check",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def make_phase_summary_figure(sh, u, stacks, path, theta_range, core_offset=1):
    """Initial vs final: pattern, 2d phase, and the core-row phase profile.

    The profile is read core_offset rows above the core. Exactly at the core
    the PN phase is odd and degenerate, so one row up gives a clean profile in
    both the knee and the dislocation state, matching the t1 convention.
    """
    x, y = sh["x"], sh["y"]
    ext = [x[0], x[-1], y[0], y[-1]]
    core = int(np.clip(sh["core_row"] + core_offset, 0, len(y) - 1))
    Nt = u.shape[-1]
    fi = [0, Nt - 1]
    lo, hi = theta_range

    fig, axs = plt.subplots(2, 3, figsize=(16, 8.5))
    for r, t in enumerate(fi):
        umax = np.nanmax(np.abs(u[:, :, t]))
        umax = 1.0 if not np.isfinite(umax) or umax == 0 else umax
        axs[r, 0].imshow(u[:, :, t], cmap="bwr", origin="lower", extent=ext,
                         vmin=-umax, vmax=umax, aspect="auto")
        axs[r, 0].set_title(f"pattern (frame {t})")

        th = stacks["theta"][:, :, t]
        thm = np.ma.masked_invalid(th)
        im = axs[r, 1].imshow(thm, cmap="twilight", origin="lower", extent=ext,
                              vmin=lo, vmax=hi, aspect="auto")
        axs[r, 1].axhline(0.0, color="k", lw=0.6, ls="--", alpha=0.6)
        axs[r, 1].set_title("symmetric phase (masked to traced region)")
        fig.colorbar(im, ax=axs[r, 1], shrink=0.85)

        # display smoothing only: theta_smooth if present, raw otherwise
        th_prof = stacks.get("theta_smooth", stacks["theta"])
        p = th_prof[core, :, t]
        axs[r, 2].plot(x, p, "C0-", lw=1.2, label="theta(x)")
        # shift-periodicity check: for the knee and the PN chain alike,
        # theta(x + pi/k1) = theta(x) - pi, so the dashed curve should
        # coincide with the solid one everywhere the extraction is right
        k1 = float(np.sqrt(1.0 - sh["mu"] ** 2))
        n_sh = int(round((np.pi / k1) / (x[1] - x[0])))
        if 0 < n_sh < len(x):
            ps = np.full_like(p, np.nan)
            ps[:-n_sh] = p[n_sh:] + np.pi
            axs[r, 2].plot(x, ps, "C3--", lw=0.9, alpha=0.8,
                           label="theta(x+pi/k1)+pi")
            axs[r, 2].legend(fontsize=7, loc="upper right")
        axs[r, 2].set_ylim(lo, hi)
        axs[r, 2].set_xlim(x[0], x[-1])
        axs[r, 2].set_title(f"phase along y={y[core]:+.2f}, frame {t}")
        axs[r, 2].grid(alpha=0.3)

    for axis in (axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]):
        axis.set_xticks([]); axis.set_yticks([])
    fig.suptitle(f"mu={sh['mu']}  initial vs final phase", y=0.99, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# GIFs (buffer_rgba, not the removed tostring_rgb)
# ---------------------------------------------------------------------------

def _fig_to_frame(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


def make_phase2d_gif(sh, theta, out_path, fps=10, vrange=None):
    """The extracted 2d phase over time, masked to the traced region."""
    x, y = sh["x"], sh["y"]
    ext = [x[0], x[-1], y[0], y[-1]]
    tt = sh["tt"]
    Nt = theta.shape[-1]
    lo, hi = vrange if vrange is not None else _robust_range(theta)

    frames = []
    for t in range(Nt):
        fig, ax = plt.subplots(figsize=(6, 6))
        thm = np.ma.masked_invalid(theta[:, :, t])
        ax.imshow(thm, cmap="twilight", origin="lower", extent=ext,
                  vmin=lo, vmax=hi, aspect="auto")
        ax.axhline(0.0, color="k", lw=0.6, ls="--", alpha=0.5)
        lbl = f"frame {t}" + (f"  t={tt[t]:.2f}" if tt is not None
                              and len(tt) == Nt else "")
        ax.set_title(lbl, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        frames.append(_fig_to_frame(fig))
        plt.close(fig)

    imageio.mimsave(out_path, frames, fps=fps)
    print(f"  phase 2d gif -> {out_path}")


def make_core_profile_gif(sh, theta_disp, out_path, charges_per_frame=None,
                          fps=10, yrange=None, core_offset=1,
                          band_x_frac=0.10, shift_residual_int=None,
                          smooth_w=None):
    """Paper-deliverable core-profile GIF.

    Per frame: theta_smooth along the profile row (solid), the
    shift-periodicity overlay theta(x + pi/k1) + pi (dashed, should
    coincide with the solid curve wherever a single-phase description
    holds), detected charge columns, and the rim shaded gray. The
    shift-periodicity claim is made in the interior only: the shaded
    rim is compromised by the ramp taper and by coordinate-line
    coverage, and is excluded from the annotated residual. Axes are
    fixed across frames; frame time and the interior residual are
    annotated. One row above the core avoids the odd-parity degeneracy
    at the exact core once a dislocation forms.

    A one-sentence caption is written next to the GIF as <stem>.caption.txt.
    """
    from utils.paper_style import PAPER_RC

    x, yv = sh["x"], sh["y"]
    core = int(np.clip(sh["core_row"] + core_offset, 0, len(yv) - 1))
    tt = sh["tt"]
    Nt = theta_disp.shape[-1]
    lo, hi = yrange if yrange is not None else _robust_range(theta_disp)

    k1 = float(np.sqrt(1.0 - float(sh["mu"]) ** 2))
    n_sh = int(round((np.pi / k1) / (x[1] - x[0])))
    Nx = len(x)
    i_trim = int(round(band_x_frac * Nx))
    x_lo, x_hi = x[i_trim], x[Nx - 1 - i_trim]

    frames = []
    with matplotlib.rc_context(PAPER_RC):
        for t in range(Nt):
            fig, ax = plt.subplots(figsize=(7.5, 3.6))
            p = theta_disp[core, :, t]
            ax.plot(x, p, "C0-", lw=1.3, label=r"$\theta(x)$")
            if 0 < n_sh < Nx:
                ps = np.full_like(p, np.nan)
                ps[:-n_sh] = p[n_sh:] + np.pi
                ax.plot(x, ps, "C3--", lw=0.9, alpha=0.85,
                        label=r"$\theta(x+\pi/k_1)+\pi$")
            if charges_per_frame is not None:
                for c in charges_per_frame[t]:
                    ax.axvline(c[0], color="0.6", lw=0.7, alpha=0.6)
            # rim shading: outside the interior the ramp taper and the
            # line coverage compromise the profile; stated, not hidden
            ax.axvspan(x[0], x_lo, color="0.85", alpha=0.6, zorder=0)
            ax.axvspan(x_hi, x[-1], color="0.85", alpha=0.6, zorder=0)
            ax.set_ylim(lo, hi)
            ax.set_xlim(x[0], x[-1])
            ann = f"frame {t}"
            if tt is not None and len(tt) == Nt:
                ann += f"   $t={tt[t]:.2f}$"
            if shift_residual_int is not None and t < len(shift_residual_int) \
                    and np.isfinite(shift_residual_int[t]):
                ann += (f"   interior shift residual "
                        f"$={shift_residual_int[t]:.3f}$ rad")
            ax.text(0.02, 0.04, ann, transform=ax.transAxes, fontsize=9,
                    va="bottom", ha="left")
            ax.text(0.98, 0.04, "shaded rim excluded (ramp, line coverage)",
                    transform=ax.transAxes, fontsize=7, va="bottom",
                    ha="right", color="0.35")
            ax.set_title(rf"phase along $y={yv[core]:+.2f}$"
                         + ("" if smooth_w is None
                            else rf"  (display smoothing $w={smooth_w:g}$)"))
            ax.set_xlabel("$x$")
            ax.legend(loc="upper right")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            frames.append(_fig_to_frame(fig))
            plt.close(fig)

    imageio.mimsave(out_path, frames, fps=fps)
    print(f"  core profile gif -> {out_path}")

    caption = (
        "Extracted phase along the row one unit above the grain-boundary "
        "core (lightly smoothed for display), with the half-period shift "
        "overlay theta(x + pi/k1) + pi coinciding with theta(x) wherever a "
        "single-phase description holds; detected charge columns are "
        "marked, and the shaded rim, compromised by the ramp taper and "
        "coordinate-line coverage, is excluded from the annotated interior "
        "residual.")
    cap_path = Path(out_path).with_suffix(".caption.txt")
    cap_path.write_text(caption + "\n")
    print(f"  caption -> {cap_path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_one(sh_path, out_dir, fig_dir, gif_dir, cfg):
    sh_path = Path(sh_path)
    stem = build_out_stem(sh_path, cfg)
    out_path = out_dir / f"{stem}.npz"

    if out_path.exists() and not cfg.get("overwrite", False):
        print(f"  skip (exists): {out_path.name}")
        return

    sh, stacks, dets, snaps, ramp = run_riesz_series(sh_path, out_path, cfg)
    u = sh["u"]
    mu = sh["mu"]

    dy = float(sh["y"][1] - sh["y"][0])
    core_offset = max(1, int(round(cfg["profile_y"] / dy)))
    print(f"  profile row: core + {core_offset} (y = "
          f"{sh['y'][sh['core_row'] + core_offset]:+.2f})")

    theta_range = _robust_range(stacks["theta"])
    kbar_range = _robust_range(stacks["kmag_direct"])
    a_range = _robust_range(stacks["A_mono"])

    if not cfg.get("no_plot", False):
        for t in sorted(snaps.keys()):
            make_defect_index_figure(
                sh, u[:, :, t], dets[t],
                fig_dir / f"{stem}_defidx_f{t:03d}.png", t, mu)
            make_kfield_figure_truthfree(
                sh, stacks, fig_dir / f"{stem}_kfield_f{t:03d}.png", t,
                kbar_range, a_range)
            make_riesz_panels_figure(
                sh, u[:, :, t] * ramp, snaps[t]["sigma_dir"],
                fig_dir / f"{stem}_riesz_f{t:03d}.png", t)
            make_lines_figure(
                sh, u[:, :, t], snaps[t],
                fig_dir / f"{stem}_lines_f{t:03d}.png", t)
            make_theta_figure(
                sh, stacks["theta"][:, :, t],
                fig_dir / f"{stem}_theta_f{t:03d}.png", t)
        make_phase_summary_figure(
            sh, u, stacks, fig_dir / f"{stem}_phase_summary.png", theta_range,
            core_offset=core_offset)
        print(f"  figures -> {fig_dir}")

    if not cfg.get("no_gif", False):
        charges = json.loads(_read_charges(out_path))
        d_op = np.load(out_path, allow_pickle=True)
        sr_int = d_op["shift_residual_int"] \
            if "shift_residual_int" in d_op.files else None
        make_phase2d_gif(sh, stacks["theta"], gif_dir / f"{stem}_phase2d.gif",
                         fps=cfg["gif_fps"], vrange=theta_range)
        make_core_profile_gif(sh, stacks["theta_smooth"],
                              gif_dir / f"{stem}_core_profile.gif",
                              charges_per_frame=charges,
                              fps=cfg["gif_fps"], yrange=theta_range,
                              core_offset=core_offset,
                              band_x_frac=cfg["band_x_frac"],
                              shift_residual_int=sr_int,
                              smooth_w=cfg.get("theta_smooth_w", 0.75))


def _read_charges(out_path):
    d = np.load(out_path, allow_pickle=True)
    return d["charges_json"].item() if hasattr(d["charges_json"], "item") \
        else str(d["charges_json"])


def run_with_cfg(cfg, args):
    out_root = Path(cfg["output_dir"])
    raw_dir = ensure_dir(out_root / "raw")
    fig_dir = ensure_dir(out_root / "figures")
    gif_dir = ensure_dir(out_root / "gifs")

    if args.sh_path is not None:
        process_one(args.sh_path, raw_dir, fig_dir, gif_dir, cfg)
    elif args.all or cfg.get("input_dir"):
        input_dir = Path(cfg.get("input_dir", args.input_dir))
        files = sorted(input_dir.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files in {input_dir}")
        print(f"Processing {len(files)} file(s) from {input_dir}")
        for p in files:
            process_one(p, raw_dir, fig_dir, gif_dir, cfg)
    else:
        raise SystemExit("Provide an sh_path or use --all with --input_dir.")
    print("Done.")


def _cfg_from_args(args):
    return {
        "sigma_fixed": args.sigma, "defect_aware": args.defect_aware,
        "sigma_large": args.sigma_large, "sigma_det": args.sigma_det,
        "band_y": args.band_y, "band_x_frac": args.band_x_frac,
        "ds": args.ds, "n_phase_seeds": args.n_phase_seeds,
        "seed_half": args.seed_half, "gauge": args.gauge,
        "director": args.director, "sigma_uhu_min": args.sigma_uhu_min,
        "seed_contour": args.seed_contour,
        "crest_y_start": args.crest_y_start,
        "crest_y_stop": args.crest_y_stop,
        "reflect_y": args.reflect_y, "profile_y": args.profile_y,
        "theta_smooth_w": args.theta_smooth_w,
        "max_steps": args.max_steps, "grad_smooth": args.grad_smooth,
        "xmargin": args.xmargin, "ymargin": args.ymargin,
        "tanhscale": args.tanhscale,
        "gif_fps": args.gif_fps,
        "output_dir": args.output_dir, "input_dir": args.input_dir,
        "no_plot": args.no_plot, "no_gif": args.no_gif,
        "overwrite": args.overwrite,
    }


def main():
    p = argparse.ArgumentParser(
        description="Riesz OP extraction for PGB zigzag SH time series.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("sh_path", nargs="?", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--input_dir", type=str,
                   default="results/sh_pgb_zigzag/raw")
    p.add_argument("--output_dir", type=str,
                   default="results/sh_pgb_zigzag/riesz")

    # width scheme
    p.add_argument("--sigma", type=float, default=0.4,
                   help="Riesz direction width on defect-free frames "
                        "(the fixed width when defect_aware is off).")
    p.add_argument("--defect_aware", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Switch width by the defect index (small/large). "
                        "On by default; --no-defect_aware for fixed width.")
    p.add_argument("--sigma_large", type=float, default=float(np.pi / 2),
                   help="Width used where charge is nonzero (defect_aware). "
                        "pi/2 matches the old uHu runs and the clean t1 PN "
                        "figures; pi over-smooths the cores.")
    p.add_argument("--sigma_det", type=float, default=0.3,
                   help="Detection width. 0.4 throws spurious charges on "
                        "the ANALYTIC knee at mu=0.95 at the SH grid "
                        "spacings (dx 0.21 to 0.35), which flips "
                        "defect_aware to the wide width at t=0 and "
                        "corrugates the knee. 0.2 to 0.3 read zero on the "
                        "knee at all three mu at these dx and still fire "
                        "on defected frames. Re-check against the "
                        "width-safety map if dx changes.")
    p.add_argument("--band_y", type=float, default=2.0)
    p.add_argument("--band_x_frac", type=float, default=0.10,
                   help="Trim this fraction of x off each edge before counting.")

    # phase pipeline
    p.add_argument("--ds", type=float, default=0.30)
    p.add_argument("--n_phase_seeds", type=int, default=128)
    p.add_argument("--seed_half", type=str, default="upper",
                   choices=["upper", "lower"],
                   help="rebuild_grid conventions assume upper.")
    p.add_argument("--director", type=str, default="riesz",
                   choices=["riesz", "uhu", "aware"],
                   help="Which director drives the crest tracer and the "
                        "coordinate lines. uhu injects the uHu "
                        "eigenvector through the identical seeding and "
                        "reconstruction, so the director is the only "
                        "variable in the A/B. aware switches with the "
                        "defect gate: riesz on clean frames, uhu on "
                        "charged ones (measured on the relaxed T100 "
                        "file: riesz wins the knee frame, uhu wins the "
                        "defected frames by 6x in the shift residual).")
    p.add_argument("--sigma_uhu_min", type=float, default=float(np.pi / 2),
                   help="Width floor for the uHu director (needs "
                        "sigma |k| of order one).")
    p.add_argument("--seed_contour", type=str, default="crest",
                   choices=["crest", "template"],
                   help="crest traces a real u crest per frame (data "
                        "driven, no analytic reference); template is the "
                        "analytic knee arm. Crest falls back to template "
                        "per frame if its gate fails.")
    p.add_argument("--crest_y_start", type=float, default=3.0,
                   help="Height of the crest trace start point.")
    p.add_argument("--crest_y_stop", type=float, default=1.0,
                   help="The descending crest branch stops at this height.")
    p.add_argument("--gauge", type=str, default="auto",
                   choices=["auto", "seed_phase", "zero_origin"],
                   help="auto: zero_origin on a crest contour (exact, "
                        "theta constant along a crest), seed_phase on the "
                        "template (fixes each line from the analytic knee).")
    p.add_argument("--reflect_y", type=str, default="mirror",
                   choices=["mirror", "even", "odd", "none"],
                   help="Lower half from the upper half. mirror overwrites "
                        "the whole lower half (visualization only); even "
                        "and odd fill only untraced cells.")
    p.add_argument("--profile_y", type=float, default=1.0,
                   help="Height above the core, in physical units, where "
                        "the summary profile and core GIF read the phase. "
                        "Measured on relaxed SH data (T100 file, frame 5): "
                        "profile wobble 0.138 rad at y=+0.25, 0.048 at "
                        "+1.0, 0.019 at +2.0; the error is a core boundary "
                        "layer of width about 1/k2, so read just outside "
                        "it. The exact PN ansatz has no such layer, which "
                        "is why t1 could read at +0.15.")
    p.add_argument("--theta_smooth_w", type=float, default=0.75,
                   help="Width of the NaN-aware Gaussian for theta_smooth, "
                        "the display copy of the phase (own stack; raw "
                        "theta is untouched). Keep below the pi/k1 plateau "
                        "scale (~7) and the riser width (1 to 2) so the "
                        "staircase steps survive; tighten toward 0.5 if "
                        "risers look rounded at mu 0.85.")
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--grad_smooth", type=float, default=1.0)

    # ramp (mirrors the uHu runner defaults for this data)
    p.add_argument("--xmargin", type=float, default=0.025)
    p.add_argument("--ymargin", type=float, default=0.025)
    p.add_argument("--tanhscale", type=float, default=120.0)

    # figures / gifs
    p.add_argument("--gif_fps", type=int, default=10)
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--no_gif", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--config", type=str, default=None)

    args = p.parse_args()
    if args.config is not None:
        with open(args.config) as f:
            cfg = json.load(f)
    else:
        cfg = _cfg_from_args(args)
    run_with_cfg(cfg, args)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Debug config: validate on one mu=0.90 dt=0.025 file first, then
        # flip all=True (and/or point input_dir at the other regime).
        _BASE = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag"
        _FINE = f"{_BASE}/mu_sweeps_full_Nx512_hp025_T3p125_NyF5_np18_Nsave125/raw"
        #_FINE = f"{_BASE}/mu_sweeps_full_Nx512_hp2_T25_NyF5_np18_Nsave125/raw"

        class Args:
            sh_path = f"{_FINE}/sh_pgb_zigzag_mu0.900_T3_N125_nx512_Ny256_lower.npz"
            #sh_path = f"{_FINE}/sh_pgb_zigzag_mu0.900_T25_N125_nx512_Ny256_lower.npz"
            all = False
            input_dir = _FINE
            output_dir = (f"{_BASE}/riesz_normal/"
                          "mu_sweeps_full_Nx512_hp025_T3p125_NyF5_np18_Nsave125")
            # output_dir = (f"{_BASE}/riesz_normal/"
            #               "mu_sweeps_full_Nx512_hp2_T25_NyF5_np18_Nsave125")
            sigma = 0.4
            defect_aware = False
            sigma_large = float(np.pi / 2)
            sigma_det = 0.3
            band_y = 2.0
            band_x_frac = 0.10
            ds = 0.30
            n_phase_seeds = 128
            seed_half = "upper"
            director = "riesz"
            sigma_uhu_min = float(np.pi / 2)
            seed_contour = "crest"
            crest_y_start = 3.0
            crest_y_stop = 1.0
            gauge = "auto"
            reflect_y = "mirror"
            profile_y = 1.0
            theta_smooth_w = 0.75
            max_steps = 10000
            grad_smooth = 1.0
            xmargin = 0.025
            ymargin = 0.025
            tanhscale = 120.0
            gif_fps = 10
            no_plot = False
            no_gif = False
            overwrite = True

        run_with_cfg(_cfg_from_args(Args), Args)
    else:
        main()