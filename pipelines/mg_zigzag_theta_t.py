# pipelines/mg_zigzag_theta_t.py
"""
The seam-free monogenic-gradient estimator on simulation frames: theta,
k, A per frame; theta_t, k_t, w_t, A_t by centred differences in time;
distance-to-core binning from the amplitude minima; two consistency
checks that need no ground truth; an amplitude-weighted continuous
phase; and a movie.

Consistency checks (a phase model predicts both to vanish away from cores)
  C1  k_t against grad(theta_t): if k is the gradient of the phase whose
      rate theta_t is, the two agree; reported as a relative RMS per bin.
  C2  curl k and curl k_t per bin: zero for a gradient field; curl k_t is
      the rate of change of defect density and is the quantity that
      nucleation must make nonzero.

Continuous phase (figure for one frame): least-squares integration of k
(Neumann Poisson, mg_estimator_validation.integrate_k) on each side of
the defect row separately, with the two traces along the row drawn
together: the staircase. A single-valued surface on the whole domain
cannot carry the 2 pi windings (they are a pure curl field and any
least-squares projection removes them, with or without an amplitude
weight; weighted_integrate is kept for the record of that test), so the
full-domain solve is drawn beside it as the curl-free part.

Input: npz with u[ny, nx, nframes], x, y, tt (or t), metadata_json with
mu and gb_row_in_crop (Ed's zigzag layout). The crop is not periodic in
x (4.5 periods) nor in y; an edge taper is applied before the estimator
(an amplitude envelope factors through the gradient formula).

Outputs (relative): results/mg_zigzag_theta_t/<stem>/<stamp>/
  report.txt, bins.csv, series.csv, phase_surfaces_f<i>.png,
  snapshot_f<i>.png, movie_theta_t.gif

Changelog
---------
v1.5  find_cores merges blobs closer than half a spacing into one core
      (at R = 1 the amplitude dip of each core splits above and below
      the row; every core was listed twice at the same x and the two
      off-centre r = 0.5 loops read 0.28 turns at mu 0.85, 1.0 at 0.90).
v1.4  find_cores labels the amplitude dips on the whole array and keeps
      blobs by centroid inside the window (was: mask intersected with the
      window before labelling). A dip straddling the window edge was
      truncated, its centroid shifted inward, and its r = 0.5 loop read
      about half a turn; on the tall (512x768) T25 crops this added two
      such blobs to the 13 real cores (mean 0.936, min 0.5 at 0.85;
      a 0.92 plateau with min 0.4 at 0.90 for t = 3 to 8). Reproduced on
      the 0.85 tail excerpt with the window edge moved into a blob (old
      0.44 to 0.52 per truncated core, new 0.999 on every core). Report
      header now carries the version. series.csv gains cores_x,
      circ_r0.5_per_core (semicolon lists ordered by x) and n_wound_r0.5
      (cores at or above 0.9 turns), so a staggered winding along the
      row can be read per core.
v1.3  the raw max|u| envelope test of v1.2 removed (uninformative: a
      wavelength box always holds a full crest); |z| minimum over median
      reported instead. Report states which loop radius is the validated
      reading. A raw-u nodal-crossing test at plateau midpoints was also
      tried and does not separate the exact knee from the exact PN row
      (both give two crossings), so the circulation of the estimator's k
      remains the instrument for the phase winding.
v1.2  per frame: raw envelope minimum (max|u| over a wavelength box) and
      the frozen director charge count, beside the k circulation, so the
      three readings that separate a bound V-X pair from a dislocation
      with a phase winding are reported together. Continuity rule of the
      loop circulation fixed (validated: exact PN through the estimator
      reads 0.96 and 1.00 turns at r = 0.3 and 0.5 spacings).
v1.1  core-row phase profiles per frame (two half-surface traces at the
      row edges and the on-row integral of k1) as a second movie, and the
      circulation of k around each detected core at several loop radii
      (the k-field version of loop persistence; 2 pi per dislocation).
v1.0  first version, validated pipeline from mg_estimator_validation
      (Stage 1) and mg_pair_dynamics_validation (Stage 2).

Run:
  python mg_zigzag_theta_t.py --npz <file>
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")   # numpy 1.x / 2.x
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import label, center_of_mass, gaussian_filter, binary_erosion

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT / "src", _ROOT / "src" / "op_extract", _ROOT / "src" / "utils",
          _HERE):
    sys.path.insert(0, str(p))

from mg_estimator_validation import mg_estimator, fdv, integrate_k  # noqa: E402
from op_extract import t1_common as t1c                              # noqa: E402

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
try:
    from paper_style import use_paper_style
    use_paper_style()
except Exception:
    pass


# ---------------------------------------------------------------------
# pieces
# ---------------------------------------------------------------------

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    md = json.loads(str(d["metadata_json"])) if "metadata_json" in d.files else {}
    u = d["u"]
    if u.ndim == 3 and u.shape[-1] < u.shape[0]:
        pass                                  # [ny, nx, nt]
    elif u.ndim == 3:
        u = np.moveaxis(u, 0, -1)             # [nt, ny, nx] -> [ny, nx, nt]
    t = d["tt"] if "tt" in d.files else d["t"]
    return dict(u=u, x=d["x"], y=d["y"], t=np.asarray(t, float), md=md)


def estimate(u, Lx, Ly, sigma_dir, n_prev, taper):
    est = mg_estimator(u * taper, Lx, Ly, sigma_dir)
    d1, d2 = est["d1"], est["d2"]
    if n_prev is not None:
        s = np.where(d1 * n_prev[0] + d2 * n_prev[1] < 0, -1.0, 1.0)
        d1, d2 = d1 * s, d2 * s
        est["z"] = est["z"].real + 1j * s * est["z"].imag
        est["theta"] = np.arctan2(est["z"].imag, est["z"].real)
        est["k1"], est["k2"] = est["k1"] * s, est["k2"] * s
    est["n"] = (d1, d2)
    est["A"] = np.abs(est["z"]) / np.maximum(taper, 1e-3)   # undo the taper
    return est


def find_cores(A, x, y, win, frac, merge_dist=0.0):
    """cores = connected regions where the amplitude falls below frac of
    its window median, labelled on the whole array and kept when their
    centroid lies inside the window; returns (xc, yc) arrays. v1.4: the
    dip mask is no longer intersected with the window before labelling,
    which truncated any dip straddling the window edge, shifted its
    centroid inward, and made the r = 0.5 loop read about half a turn
    (13 real cores + 2 edge blobs gave mean 0.936, min 0.5 on the tall
    0.85 T25 run; reproduced on the 0.85 tail excerpt by moving the
    window edge into a blob: old 0.44-0.52, new 0.999 on every core)."""
    ref = np.median(A[win])
    lab, n = label(A < frac * ref)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    cms = center_of_mass(np.ones_like(A), lab, range(1, n + 1))
    iy = np.array([c[0] for c in cms]); ix = np.array([c[1] for c in cms])
    keep = win[np.clip(np.round(iy).astype(int), 0, A.shape[0] - 1),
               np.clip(np.round(ix).astype(int), 0, A.shape[1] - 1)]
    ix, iy = ix[keep], iy[keep]
    xc, yc = np.interp(ix, np.arange(len(x)), x), np.interp(iy, np.arange(len(y)), y)
    # v1.5: merge blobs closer than merge_dist (one core whose dip splits
    # above and below the row, seen at R = 1: every core listed twice at
    # the same x, both loops off-centre, 0.28 turns at mu 0.85)
    if len(xc) > 1 and merge_dist > 0:
        order = np.argsort(xc); xc, yc = xc[order], yc[order]
        groups, cur = [], [0]
        for j in range(1, len(xc)):
            if np.hypot(xc[j] - xc[cur[0]], yc[j] - yc[cur[0]]) < merge_dist:
                cur.append(j)
            else:
                groups.append(cur); cur = [j]
        groups.append(cur)
        xc = np.array([xc[g].mean() for g in groups]); yc = np.array([yc[g].mean() for g in groups])
    return xc, yc


def dist_to_cores(X, Y, xc, yc):
    if len(xc) == 0:
        return np.full_like(X, np.inf)
    return np.min(np.hypot(X[..., None] - xc[None, None, :],
                           Y[..., None] - yc[None, None, :]), axis=-1)


def weighted_integrate(k1, k2, w, dx, dy):
    """div(w grad phi) = div(w k) on the box, natural boundary conditions,
    five-point stencil with face weights = harmonic means; mean pinned."""
    ny, nx = k1.shape
    N = ny * nx
    idx = np.arange(N).reshape(ny, nx)
    hm = lambda a, b: 2 * a * b / np.maximum(a + b, 1e-30)
    wx = hm(w[:, :-1], w[:, 1:])          # faces between (i,j) and (i,j+1)
    wy = hm(w[:-1, :], w[1:, :])          # faces between (i,j) and (i+1,j)
    kx = 0.5 * (k1[:, :-1] + k1[:, 1:])
    ky = 0.5 * (k2[:-1, :] + k2[1:, :])
    rows, cols, vals = [], [], []
    rhs = np.zeros(N)
    # x faces
    a, b = idx[:, :-1].ravel(), idx[:, 1:].ravel()
    cx = (wx / dx ** 2).ravel()
    fx = (wx * kx / dx).ravel()
    rows += [a, b, a, b]; cols += [a, b, b, a]; vals += [cx, cx, -cx, -cx]
    np.add.at(rhs, a, -fx); np.add.at(rhs, b, fx)
    # y faces
    a, b = idx[:-1, :].ravel(), idx[1:, :].ravel()
    cy = (wy / dy ** 2).ravel()
    fy = (wy * ky / dy).ravel()
    rows += [a, b, a, b]; cols += [a, b, b, a]; vals += [cy, cy, -cy, -cy]
    np.add.at(rhs, a, -fy); np.add.at(rhs, b, fy)
    L = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                  np.concatenate(cols))),
                          shape=(N, N)).tocsr()
    # sign: the assembled L is -div(w grad); rhs assembled as -div(w k)
    # pin the mean with a rank-one term
    e = np.ones(N) / np.sqrt(N)
    Lp = L + sparse.csr_matrix(np.outer(e, e) * L.diagonal().mean()) \
        if N <= 4096 else L + sparse.eye(N) * 1e-8 * L.diagonal().mean()
    phi = spsolve(Lp.tocsc(), rhs)
    phi = phi.reshape(ny, nx)
    return phi - phi.mean()


def rel_rms(err, ref, m):
    if m.sum() < 10:
        return float("nan")
    return float(np.sqrt(np.mean(err[m] ** 2)) / max(np.sqrt(np.mean(ref[m] ** 2)), 1e-14))


# ---------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------

def run(args):
    D = load_npz(args.npz)
    u, x, y, t, md = D["u"], D["x"], D["y"], D["t"], D["md"]
    ny, nx, nt = u.shape
    dx, dy = x[1] - x[0], y[1] - y[0]
    Lx, Ly = nx * dx, ny * dy
    dt = float(np.median(np.diff(t)))
    mu = float(md.get("mu", args.mu))
    mu_k1 = float(np.sqrt(1 - mu ** 2))
    Lsp = np.pi / mu_k1
    X, Y = np.meshgrid(x, y)
    taper = t1c.edge_taper(x, y, frac=args.taper_frac)
    win = (np.abs(X - x.mean()) < args.win_frac * Lx) & \
          (np.abs(Y - y.mean()) < args.win_frac * Ly)
    edges = args.edges
    Dsp = lambda a, wch: fdv.spectral_deriv(a, Lx, Ly, wch)

    stem = Path(args.npz).stem
    stamp = "sd%.2g_tp%.2g_cf%.2g" % (args.sigma_dir, args.taper_frac, args.core_frac)
    out = Path(args.outdir) / stem / stamp
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as fh:
        json.dump({k: getattr(args, k) for k in vars(args)}, fh, indent=2, default=str)

    t0 = time.time()
    ests, n_prev = [], None
    for i in range(nt):
        e = estimate(u[..., i], Lx, Ly, args.sigma_dir, n_prev, taper)
        n_prev = e["n"]
        ests.append(e)
    lines = ["mg_zigzag_theta_t v1.5   %s" % stem, "",
             "mu %.3f  Lsp %.3f  grid %dx%d  dx %.4f  dt %.4f  t = %.3f .. %.3f"
             " (%d frames)  window %.2f  taper %.2g" %
             (mu, Lsp, nx, ny, dx, dt, t[0], t[-1], nt, args.win_frac, args.taper_frac),
             "estimator %.1f s for %d frames" % (time.time() - t0, nt), ""]

    series, bins, movie = [], [], []
    jc0 = int(md.get("gb_row_in_crop", ny // 2))
    for i in range(1, nt - 1):
        e0, e1, e2 = ests[i - 1], ests[i], ests[i + 1]
        z_t = (e2["z"] - e0["z"]) / (2 * dt)
        th_t = np.imag(np.conj(e1["z"]) * z_t) / np.maximum(np.abs(e1["z"]) ** 2, 1e-14)
        k1_t = (e2["k1"] - e0["k1"]) / (2 * dt)
        k2_t = (e2["k2"] - e0["k2"]) / (2 * dt)
        w = e1["k1"] ** 2 + e1["k2"] ** 2 - 1.0
        w_t = 2 * (e1["k1"] * k1_t + e1["k2"] * k2_t)
        A_t = (e2["A"] - e0["A"]) / (2 * dt)
        xc, yc = find_cores(e1["A"], x, y, win, args.core_frac, merge_dist=0.5 * Lsp)
        dist = dist_to_cores(X, Y, xc, yc) / Lsp
        # C1: k_t vs grad theta_t (theta_t is smooth; spectral gradient on
        # the tapered-away field)
        g1 = Dsp(th_t * taper, "x"); g2 = Dsp(th_t * taper, "y")
        c1_err = np.hypot(k1_t - g1, k2_t - g2)
        c1_ref = np.hypot(g1, g2)
        # C2: curls
        curl_k = Dsp(e1["k2"] * taper, "x") - Dsp(e1["k1"] * taper, "y")
        curl_kt = Dsp(k2_t * taper, "x") - Dsp(k1_t * taper, "y")
        gradk = np.hypot(np.hypot(Dsp(e1["k1"] * taper, "x"), Dsp(e1["k1"] * taper, "y")),
                         np.hypot(Dsp(e1["k2"] * taper, "x"), Dsp(e1["k2"] * taper, "y")))
        gradkt = np.hypot(np.hypot(g1, g2), 1e-14)
        row = {"i": i, "t": t[i], "n_cores": len(xc)}
        # amplitude minimum of |z| relative to the window median (a raw
        # max|u| envelope over a wavelength box was tried and is
        # uninformative: it always contains a full crest, even at a core)
        row["A_min_over_median"] = float(e1["A"][win].min() / np.median(e1["A"][win]))
        try:
            det = t1c.detect_defects(u[..., i] * taper, x, y - y[jc0], Lx, Ly,
                                     sigma_det=args.sigma_det, band_y=3.0)
            row["director_sum_abs_q"] = int(det["sum_abs"])
        except Exception:
            row["director_sum_abs_q"] = -1
        # weak-form versions: mask to the window and outside the cores,
        # integrate against a Gaussian test function of test_sigma
        # spacings, score on the window eroded by 2.5 test widths, and
        # normalise by the window-wide RMS of the reference (both the
        # target and its reference are small in the far field, so a
        # per-bin ratio of two small quantities is not informative)
        sg = args.test_sigma * Lsp / dx
        keep = (win & (dist >= args.core_excl)).astype(float)
        inner = binary_erosion(win & (dist >= args.core_excl), iterations=int(np.ceil(2.5 * sg)))
        W = lambda F: gaussian_filter(F * keep, sg, mode="nearest")
        c1w = np.hypot(W(k1_t - g1), W(k2_t - g2))
        c1r = np.hypot(W(g1), W(g2))
        curlk_w, curlkt_w = W(curl_k), W(curl_kt)
        divk_w = W(Dsp(e1["k1"] * taper, "x") + Dsp(e1["k2"] * taper, "y"))
        divkt_w = W(Dsp(k1_t * taper, "x") + Dsp(k2_t * taper, "y"))
        row["C1_weak"] = rel_rms(c1w, c1r, inner)
        row["curlk_over_divk_weak"] = rel_rms(curlk_w, divk_w, inner)
        row["curlkt_over_divkt_weak"] = rel_rms(curlkt_w, divkt_w, inner)
        row["n_inner"] = int(inner.sum())
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = win & (dist >= lo) & (dist < hi)
            key = "[%g,%g)" % (lo, hi)
            b = {"i": i, "t": t[i], "lo": lo, "hi": hi, "n": int(m.sum()),
                 "rms_theta_t": float(np.sqrt(np.mean(th_t[m] ** 2))) if m.sum() else np.nan,
                 "rms_w_t": float(np.sqrt(np.mean(w_t[m] ** 2))) if m.sum() else np.nan,
                 "rms_A_t": float(np.sqrt(np.mean(A_t[m] ** 2))) if m.sum() else np.nan,
                 "rms_w": float(np.sqrt(np.mean(w[m] ** 2))) if m.sum() else np.nan,
                 "C1_pointwise": rel_rms(c1_err, c1_ref, m),
                 "C1_weak_bin": (float(np.sqrt(np.mean(c1w[m & inner] ** 2)) / max(np.sqrt(np.mean(c1r[inner] ** 2)), 1e-14)) if (m & inner).sum() > 10 else np.nan),
                 "curlk_weak_bin": (float(np.sqrt(np.mean(curlk_w[m & inner] ** 2)) / max(np.sqrt(np.mean(divk_w[inner] ** 2)), 1e-14)) if (m & inner).sum() > 10 else np.nan),
                 "curlkt_weak_bin": (float(np.sqrt(np.mean(curlkt_w[m & inner] ** 2)) / max(np.sqrt(np.mean(divkt_w[inner] ** 2)), 1e-14)) if (m & inner).sum() > 10 else np.nan)}
            bins.append(b)
            for q in ("rms_theta_t", "rms_w_t", "rms_A_t", "C1_pointwise",
                      "C1_weak_bin", "curlk_weak_bin", "curlkt_weak_bin"):
                row[q + key] = b[q]
        # core-row phase profiles: the two half-surface traces at the row
        # edges and the on-row integral of k1 (uncensored; riser heights
        # carry the estimator's core error), gauge-matched far from the row
        jc = int(md.get("gb_row_in_crop", ny // 2))
        gap = int(np.ceil(args.row_gap * Lsp / dy))
        sl_x = slice(int(0.08 * nx), int(0.92 * nx))
        prof = {}
        for lab, sl_ in (("above", slice(jc + gap, int(0.92 * ny))),
                         ("below", slice(int(0.08 * ny), jc - gap + 1))):
            ph = integrate_k(e1["k1"][sl_, sl_x], e1["k2"][sl_, sl_x], x[sl_x], y[sl_])
            tr = ph[0, :] if lab == "above" else ph[-1, :]
            prof[lab] = tr - tr[len(tr) // 2]
        from scipy.integrate import cumulative_trapezoid as _ctz
        onrow = _ctz(e1["k1"][jc, sl_x], dx=dx, initial=0.0)
        prof["on row"] = onrow - onrow[len(onrow) // 2]
        row["prof_rms_above_minus_below"] = float(np.sqrt(np.mean((prof["above"] - prof["below"]) ** 2)))
        # circulation of k around each core at several loop radii (a
        # dislocation gives 2 pi; the loop stays outside the quarter-spacing
        # disk where the estimator's k is unreliable)
        circ = {}
        for rr in args.loop_radii:
            vals = []
            for xc_, yc_ in zip(xc, yc):
                ang = np.linspace(0, 2 * np.pi, 181)
                px = xc_ + rr * Lsp * np.cos(ang); py = yc_ + rr * Lsp * np.sin(ang)
                ix = np.clip(np.round((px - x[0]) / dx).astype(int), 0, nx - 1)
                iy = np.clip(np.round((py - y[0]) / dy).astype(int), 0, ny - 1)
                kx_, ky_ = e1["k1"][iy, ix], e1["k2"][iy, ix]
                # tangent direction, with the director sign resolved along the
                # loop by continuity (k is defined up to sign)
                sgn = np.ones_like(ang)
                for q in range(1, len(ang)):
                    dot = kx_[q] * kx_[q - 1] + ky_[q] * ky_[q - 1]
                    sgn[q] = sgn[q - 1] if dot >= 0 else -sgn[q - 1]
                tx, ty = -np.sin(ang), np.cos(ang)
                vals.append(_trapz((kx_ * tx + ky_ * ty) * sgn, ang) * rr * Lsp / (2 * np.pi))
            circ[rr] = np.array(vals)
            row["circ_mean_r%.2g" % rr] = float(np.mean(np.abs(circ[rr]))) if len(vals) else np.nan
            row["circ_min_r%.2g" % rr] = float(np.min(np.abs(circ[rr]))) if len(vals) else np.nan
            if rr == 0.5:
                # per-core values at the validated radius, ordered by x, so
                # the staggering of the winding along the row can be read
                order = np.argsort(xc)
                row["cores_x"] = ";".join("%.2f" % v for v in np.asarray(xc)[order])
                row["circ_r0.5_per_core"] = ";".join("%.3f" % v for v in np.abs(circ[rr])[order])
                row["n_wound_r0.5"] = int(np.sum(np.abs(circ[rr]) >= 0.9)) if len(vals) else 0
        series.append(row)
        movie.append((t[i], th_t, w_t, e1["A"], xc, yc, prof, x[sl_x]))
        if i == args.snap_frame:
            snap = (i, e1, th_t, w_t, A_t, xc, yc, dist)

    # report tables
    lines.append("cores found per frame (amplitude below %.2f of the window"
                 " median): %s" % (args.core_frac, [r["n_cores"] for r in series]))
    lines.append("")
    for row in (series[0], series[len(series) // 2], series[-1]):
        lines.append("frame %d  t = %.3f   weak checks on the interior (n = %d, test"
                     " width %.2g spacing): C1 |k_t - grad theta_t| / |grad theta_t|"
                     " = %.3f;  |curl k| / |div k| = %.3f;  |curl k_t| / |div k_t| = %.3f" %
                     (row["i"], row["t"], row["n_inner"], args.test_sigma,
                      row["C1_weak"], row["curlk_over_divk_weak"],
                      row["curlkt_over_divkt_weak"]))
        lines.append("  %-10s %9s %9s %9s | %8s %8s %8s %8s" %
                     ("dist/Lsp", "rms th_t", "rms w_t", "rms A_t", "C1 ptws", "C1 weak",
                      "curl k", "curl k_t"))
        for lo, hi in zip(edges[:-1], edges[1:]):
            key = "[%g,%g)" % (lo, hi)
            lines.append("  %-10s %9.4f %9.4f %9.4f | %8.3f %8.3f %8.3f %8.3f" %
                         (key, row["rms_theta_t" + key], row["rms_w_t" + key],
                          row["rms_A_t" + key], row["C1_pointwise" + key],
                          row["C1_weak_bin" + key], row["curlk_weak_bin" + key],
                          row["curlkt_weak_bin" + key]))
        lines.append("")
    lines.append("circulation of k around detected cores, in units of 2 pi, |mean| (min) per"
                 " frame at loop radii %s spacings. READ r = 0.5 (validated on the exact PN"
                 " through this code: 1.00 turns); r = 0.3 lies inside the quarter-spacing"
                 " zone where the estimator is unreliable; r >= 0.75 overlaps neighbouring"
                 " cores on a one-spacing row. Partial values mark transition frames."
                 % list(args.loop_radii))
    for row in series:
        lines.append("  t = %.3f  " % row["t"] + "  ".join(
            "r%.2g: %.3f (%.3f)" % (rr, row["circ_mean_r%.2g" % rr], row["circ_min_r%.2g" % rr])
            for rr in args.loop_radii)
            + "   row traces above vs below RMS %.3f rad" % row["prof_rms_above_minus_below"]
            + "   |z| min/median %.3f   director sum|q| %d" %
            (row["A_min_over_median"], row["director_sum_abs_q"]))
    lines.append("")
    lines.append("bin columns: C1 ptws = per-bin pointwise ratio (noise-dominated where"
                 " the rates are small, kept for reference); C1 weak, curl k, curl k_t ="
                 " weak residual in the bin over the window-wide weak RMS of grad theta_t,"
                 " div k, div k_t respectively")

    def write_csv(rows, path):
        keys = list(rows[0].keys())
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
    write_csv(series, out / "series.csv")
    write_csv(bins, out / "bins.csv")

    # ---- continuous phase for the snapshot frame
    # The 2 pi circulation of k around each dislocation is a pure curl
    # field, and any least-squares surface on a simply connected domain
    # projects it out, whatever the weight (measured: an amplitude weight
    # (A/median)^p with p = 2, 8, 20 changes the across-row jump by 0.01
    # rad). So the continuous phase is built on each side of the defect
    # row separately, and the two traces along the row are the staircase.
    i, e1, th_t, w_t, A_t, xc, yc, dist = snap
    jc = int(md.get("gb_row_in_crop", ny // 2))
    gap = int(np.ceil(args.row_gap * Lsp / dy))
    sl_x = slice(int(0.08 * nx), int(0.92 * nx))
    sl_up = slice(jc + gap, int(0.92 * ny))
    sl_lo = slice(int(0.08 * ny), jc - gap + 1)
    t1 = time.time()
    phi_up = integrate_k(e1["k1"][sl_up, sl_x], e1["k2"][sl_up, sl_x], x[sl_x], y[sl_up])
    phi_lo = integrate_k(e1["k1"][sl_lo, sl_x], e1["k2"][sl_lo, sl_x], x[sl_x], y[sl_lo])
    phi_full = integrate_k(e1["k1"][int(0.08 * ny):int(0.92 * ny), sl_x],
                           e1["k2"][int(0.08 * ny):int(0.92 * ny), sl_x], x[sl_x],
                           y[int(0.08 * ny):int(0.92 * ny)])
    # gauge: match each half to the full-domain surface far from the row
    phi_up += np.mean((phi_full[sl_up.start - int(0.08 * ny):, :] - phi_up)[-20:, :])
    phi_lo += np.mean((phi_full[:sl_lo.stop - int(0.08 * ny), :] - phi_lo)[:20, :])
    lines.append("")
    lines.append("continuous phase, frame %d: two half-domain solves %.1f s;"
                 " row gap %.2g spacing" % (i, time.time() - t1, args.row_gap))
    for lab, phi_, sl_ in (("upper", phi_up, sl_up), ("lower", phi_lo, sl_lo)):
        gy, gx = np.gradient(phi_, dy, dx)
        kk = np.hypot(e1["k1"], e1["k2"])[sl_, sl_x]
        mm = dist[sl_, sl_x] > 0.5
        lines.append("  %s half: |grad phi| vs |k| rel RMS beyond half a spacing: %.4f"
                     % (lab, rel_rms(np.hypot(gx, gy) - kk, kk, mm)))
    trace_up, trace_lo = phi_up[0, :], phi_lo[-1, :]
    ext = [x[sl_x][0], x[sl_x][-1], y[int(0.08 * ny)], y[int(0.92 * ny) - 1]]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))
    axs[0, 0].imshow(u[..., i][int(0.08 * ny):int(0.92 * ny), sl_x], origin="lower", extent=ext, cmap="gray")
    axs[0, 0].set_title("u, t = %.3f" % t[i])
    im = axs[0, 1].imshow(e1["A"][int(0.08 * ny):int(0.92 * ny), sl_x], origin="lower", extent=ext, cmap="viridis")
    axs[0, 1].set_title("A = |z|"); fig.colorbar(im, ax=axs[0, 1], shrink=0.8)
    im = axs[0, 2].imshow(e1["theta"][int(0.08 * ny):int(0.92 * ny), sl_x], origin="lower", extent=ext, cmap="twilight")
    axs[0, 2].set_title("arg z (wrapped)"); fig.colorbar(im, ax=axs[0, 2], shrink=0.8)
    both = np.full_like(phi_full, np.nan)
    both[sl_up.start - int(0.08 * ny):, :] = phi_up
    both[:sl_lo.stop - int(0.08 * ny), :] = phi_lo
    v = np.nanpercentile(np.abs(both), 99)
    im = axs[1, 0].imshow(both, origin="lower", extent=ext, cmap="viridis", vmin=-v, vmax=v)
    axs[1, 0].set_title("continuous phase, each side of the row"); fig.colorbar(im, ax=axs[1, 0], shrink=0.8)
    im = axs[1, 1].imshow(phi_full, origin="lower", extent=ext, cmap="viridis", vmin=-v, vmax=v)
    axs[1, 1].set_title("full-domain least squares (curl-free part)"); fig.colorbar(im, ax=axs[1, 1], shrink=0.8)
    axs[1, 2].plot(x[sl_x], trace_up, "r-", lw=1.2, label="row edge above, y = %.1f" % y[sl_up.start])
    axs[1, 2].plot(x[sl_x], trace_lo, "b-", lw=1.2, label="row edge below, y = %.1f" % y[sl_lo.stop - 1])
    axs[1, 2].plot(x[sl_x], trace_up - trace_lo, "k--", lw=1, label="difference")
    for xx_ in xc:
        axs[1, 2].axvline(xx_, color="0.7", lw=0.6)
    axs[1, 2].set_xlabel("x"); axs[1, 2].set_ylabel("phi"); axs[1, 2].legend(fontsize=7)
    axs[1, 2].set_title("traces along the defect row: the staircase")
    for ax in axs.ravel()[:5]:
        ax.plot(xc, yc, "w+", ms=6, mew=1)
    fig.tight_layout()
    fig.savefig(out / ("phase_surfaces_f%d.png" % i), dpi=160)
    plt.close(fig)

    # ---- snapshot of the rates
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (lab, F, cm) in zip(axs, (("theta_t", th_t, "RdBu_r"), ("w_t", w_t, "RdBu_r"),
                                      ("A_t", A_t, "RdBu_r"), ("w", e1["k1"] ** 2 + e1["k2"] ** 2 - 1, "RdBu_r"))):
        v = np.percentile(np.abs(F[win]), 98)
        im = ax.imshow(F, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap=cm, vmin=-v, vmax=v)
        ax.plot(xc, yc, "k+", ms=6)
        ax.set_title("%s, t = %.3f" % (lab, t[i])); fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / ("snapshot_f%d.png" % i), dpi=150)
    plt.close(fig)

    # ---- movie (theta_t, w_t, A), fixed colour scales across frames
    try:
        from PIL import Image
        vth = np.percentile(np.abs(np.stack([m_[1] for m_ in movie])[:, win]), 98)
        vw = np.percentile(np.abs(np.stack([m_[2] for m_ in movie])[:, win]), 98)
        frames_png = []
        for (tt, th_, wt_, A_, xc_, yc_, prof_, xp_) in movie:
            fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
            for ax, (lab, F, v, cm) in zip(axs, (("theta_t", th_, vth, "RdBu_r"),
                                                 ("w_t", wt_, vw, "RdBu_r"),
                                                 ("A", A_, None, "viridis"))):
                if v is None:
                    im = ax.imshow(F, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap=cm, vmin=0, vmax=1.1 * np.median(A_[win]))
                else:
                    im = ax.imshow(F, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap=cm, vmin=-v, vmax=v)
                ax.plot(xc_, yc_, "k+", ms=5)
                ax.set_title("%s, t = %.3f" % (lab, tt))
            fig.tight_layout()
            pth = out / ("_frame_%.3f.png" % tt)
            fig.savefig(pth, dpi=90); plt.close(fig)
            frames_png.append(pth)
        ims = [Image.open(p_) for p_ in frames_png]
        ims[0].save(out / "movie_theta_t.gif", save_all=True, append_images=ims[1:],
                    duration=350, loop=0)
        for p_ in frames_png:
            p_.unlink()
        lines.append("movie -> movie_theta_t.gif (%d frames)" % len(ims))
        # core-row profile movie, fixed axes
        allv = np.concatenate([np.concatenate(list(m_[6].values())) for m_ in movie])
        lo_, hi_ = np.percentile(allv, [0.5, 99.5])
        frames_png = []
        for (tt, th_, wt_, A_, xc_, yc_, prof_, xp_) in movie:
            fig, ax = plt.subplots(figsize=(7, 3.8))
            ax.plot(xp_, prof_["above"], "r-", lw=1.3, label="row edge above (+%.2g spacing)" % args.row_gap)
            ax.plot(xp_, prof_["below"], "b-", lw=1.0, label="row edge below")
            ax.plot(xp_, prof_["on row"], "k:", lw=1.0, label="on-row integral of k1")
            ax.plot(xp_, mu_k1 * (xp_ - xp_[len(xp_) // 2]), color="0.6", lw=0.8, ls="--", label="k1 x (knee)")
            for xx_ in xc_:
                ax.axvline(xx_, color="0.85", lw=0.6)
            ax.set_ylim(lo_, hi_)
            ax.set_xlabel("x along the core"); ax.set_ylabel("phase")
            ax.set_title("phase profile along the defect row, t = %.3f" % tt)
            ax.legend(fontsize=7, loc="upper left")
            fig.tight_layout()
            pth = out / ("_prof_%.3f.png" % tt)
            fig.savefig(pth, dpi=90); plt.close(fig)
            frames_png.append(pth)
        ims = [Image.open(p_) for p_ in frames_png]
        ims[0].save(out / "movie_core_profile.gif", save_all=True, append_images=ims[1:],
                    duration=350, loop=0)
        for p_ in frames_png:
            p_.unlink()
        lines.append("movie -> movie_core_profile.gif (%d frames)" % len(ims))
    except Exception as exc:
        lines.append("movie skipped: %s" % exc)

    report = "\n".join(lines)
    (out / "report.txt").write_text(report)
    print(report)
    print("outputs ->", out)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    npz = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweeps_full_Nx512_hp025_T3p125_NyF5_np18_Nsave125/raw/sh_pgb_zigzag_mu0.900_T3_N125_nx512_Ny256_lower.npz"
    outdir = str(Path("results") / "mg_zigzag_theta_t")
    mu = 0.85                    # used only if metadata lacks mu
    sigma_dir = 0.4
    taper_frac = 0.08
    win_frac = 0.36
    test_sigma = 0.5             # weak-form test width, spacings
    core_excl = 0.5              # spacings excluded around cores in weak checks
    row_gap = 0.25               # spacings excluded on each side of the defect row
    loop_radii = (0.3, 0.5, 0.75, 1.0)   # circulation loops, spacings
    sigma_det = 0.3              # director index width (production value)
    core_frac = 0.70             # amplitude threshold for core detection (SH cores: A dips to ~0.6 of the median)
    edges = [0, 0.25, 0.5, 1, 2, 4]
    snap_frame = 5


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", default=_Args.npz, required=not _Args.npz)
    p.add_argument("--outdir", default=_Args.outdir)
    p.add_argument("--mu", type=float, default=_Args.mu)
    p.add_argument("--sigma-dir", type=float, dest="sigma_dir", default=_Args.sigma_dir)
    p.add_argument("--taper-frac", type=float, dest="taper_frac", default=_Args.taper_frac)
    p.add_argument("--win-frac", type=float, dest="win_frac", default=_Args.win_frac)
    p.add_argument("--core-frac", type=float, dest="core_frac", default=_Args.core_frac)
    p.add_argument("--snap-frame", type=int, dest="snap_frame", default=_Args.snap_frame)
    p.add_argument("--test-sigma", type=float, dest="test_sigma", default=_Args.test_sigma)
    p.add_argument("--row-gap", type=float, dest="row_gap", default=_Args.row_gap)
    p.add_argument("--sigma-det", type=float, dest="sigma_det", default=_Args.sigma_det)
    p.add_argument("--core-excl", type=float, dest="core_excl", default=_Args.core_excl)
    cli = p.parse_args()
    cli.edges = _Args.edges
    cli.loop_radii = _Args.loop_radii
    sys.exit(run(cli))
