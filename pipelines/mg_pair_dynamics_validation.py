# pipelines/mg_pair_dynamics_validation.py
"""
Stage 2 validation of the monogenic-gradient estimator in time: theta_t,
k_t and w_t on a separating dislocation pair whose time dependence is
exact by construction (parametric movie, no PDE solved).

Field (nucleation_measures_validation.field_pair, carrier rotated to the
production angle so the far field sits off the director seam):

    theta(x, y; d) = k1 x + k2 y + atan2(y, x + d/2) - atan2(y, x - d/2)
    A(x, y; d)     = tanh(r_L / xi) tanh(r_R / xi),   u = A cos theta
    k1 = sqrt(1 - mu^2), k2 = mu,  r_L,R = distance to the two cores.

With d(t) prescribed, theta_t = theta_d d'(t) exactly, where
    theta_d = -y / (2 r_L^2) - y / (2 r_R^2),
k_t = grad theta_t and w_t = 2 k . k_t (k = grad theta in closed form).

Estimator (mg_estimator_validation.mg_estimator, seam-free form) per
frame, with the director sign made continuous in time so that z is a
single-valued function of t at every pixel; then

    theta_t = Im(conj(z) z_t) / |z|^2,       z_t by centred differences,
    k_t     by centred differences of the per-frame k,
    w_t     = 2 k . k_t   (estimator k).

The time step is a pure discretisation knob: halving it must leave the
errors unchanged; the report prints both dt and dt/2 for one frame.

Metrics per frame and per distance-to-core bin (spacing units): relative
RMS error of theta_t, |k_t| and w_t against the exact fields, restricted
to |x| < win_frac Lx, |y| < win_frac Ly. Time series of the far-bin and
near-bin errors against d(t), and a snapshot figure at a mid separation.

Outputs (relative): results/mg_pair_dynamics_validation/<stamp>/
  config.json, timeseries.csv, bins.csv, report.txt,
  errors_vs_separation.png, snapshot_d<d>.png

Changelog
---------
v1.1  edge taper before the estimator (the dipole phase term does not
      close on the box; the FFT leakage moved with d and read as a far
      field k_t); k_t and w_t scored against the window RMS as well as per
      bin (both decay like 1/r^2); weak-form proxy (Gaussian test function
      of half a spacing, core cells masked) added.
v1.0  first version.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT / "src", _ROOT / "src" / "op_extract", _ROOT / "src" / "utils",
          _HERE):
    sys.path.insert(0, str(p))

from mg_estimator_validation import mg_estimator, fdv           # noqa: E402
from op_extract import t1_common as t1c                        # noqa: E402

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
try:
    from paper_style import use_paper_style
    use_paper_style()
except Exception:
    pass


# ---------------------------------------------------------------------
# the parametric field
# ---------------------------------------------------------------------

def make_grid(mu, n_xper, n_yper, ppw):
    k1, k2 = float(np.sqrt(1 - mu ** 2)), float(mu)
    dx = 2 * np.pi / ppw
    Lsp = np.pi / k1
    Lx = n_xper * Lsp
    Ly = n_yper * (2 * np.pi / k2)
    Nx, Ny = int(round(Lx / dx)), int(round(Ly / dx))
    Lx, Ly = Nx * dx, Ny * dx
    x = (np.arange(Nx) - Nx // 2) * dx
    y = (np.arange(Ny) - Ny // 2) * dx + 0.5 * dx        # no row on y = 0
    X, Y = np.meshgrid(x, y)
    return dict(x=x, y=y, X=X, Y=Y, dx=dx, Lx=Lx, Ly=Ly, k1=k1, k2=k2,
                mu=mu, Lsp=Lsp)


def pair_field(g, d, xi):
    X, Y = g["X"], g["Y"]
    xL, xR = -0.5 * d, 0.5 * d
    rL2 = (X - xL) ** 2 + Y ** 2
    rR2 = (X - xR) ** 2 + Y ** 2
    theta = (g["k1"] * X + g["k2"] * Y
             + np.arctan2(Y, X - xL) - np.arctan2(Y, X - xR))
    A = np.tanh(np.sqrt(rL2) / xi) * np.tanh(np.sqrt(rR2) / xi)
    # closed-form k = grad theta (grad of atan2(y, x - x0) = (-y, x - x0)/r^2)
    k1 = g["k1"] - Y / rL2 + Y / rR2
    k2 = g["k2"] + (X - xL) / rL2 - (X - xR) / rR2
    theta_d = -Y / (2 * rL2) - Y / (2 * rR2)
    # grad theta_d, closed form
    thd_x = (Y * (X - xL) / rL2 ** 2 + Y * (X - xR) / rR2 ** 2)
    thd_y = (-0.5 / rL2 + Y ** 2 / rL2 ** 2 - 0.5 / rR2 + Y ** 2 / rR2 ** 2)
    return dict(theta=theta, A=A, u=A * np.cos(theta), k1=k1, k2=k2,
                theta_d=theta_d, thd_x=thd_x, thd_y=thd_y,
                dist=np.minimum(np.sqrt(rL2), np.sqrt(rR2)) / g["Lsp"])


# ---------------------------------------------------------------------
# estimator in time
# ---------------------------------------------------------------------

def estimate_frame(u, g, sigma_dir, n_prev, taper=None):
    if taper is not None:
        u = u * taper
    est = mg_estimator(u, g["Lx"], g["Ly"], sigma_dir)
    d1, d2 = est["d1"], est["d2"]
    if n_prev is not None:                       # temporal sign continuity
        flip = (d1 * n_prev[0] + d2 * n_prev[1]) < 0
        s = np.where(flip, -1.0, 1.0)
        d1, d2 = d1 * s, d2 * s
        est["z"] = u + 1j * s * est["z"].imag
        est["theta"] = np.arctan2(est["z"].imag, u)
        est["k1"], est["k2"] = est["k1"] * s, est["k2"] * s
    est["n"] = (d1, d2)
    return est


def rel_rms(err, ref, m):
    return float(np.sqrt(np.mean(err[m] ** 2)) / max(np.sqrt(np.mean(ref[m] ** 2)), 1e-14))


def run(args):
    g = make_grid(args.mu, args.n_xper, args.n_yper, args.ppw)
    win = (np.abs(g["X"]) < args.win_frac * g["Lx"]) & \
          (np.abs(g["Y"]) < args.win_frac * g["Ly"])
    edges = args.edges
    ts = np.arange(0.0, args.t_end + 1e-12, args.dt)
    dfun = lambda t: args.d0 + args.v * t
    dprime = args.v

    taper = t1c.edge_taper(g["x"], g["y"], frac=args.taper_frac) \
        if args.taper_frac > 0 else None
    # rolling three-frame window: fields and estimates are not all kept
    n_prev = None
    t0 = time.time()
    def make(i, n_prev):
        fr = pair_field(g, dfun(ts[i]), args.xi)
        e = estimate_frame(fr["u"], g, args.sigma_dir, n_prev, taper)
        return fr, e
    fr0, e0 = make(0, None)
    fr1, e1 = make(1, e0["n"])
    t_est = time.time() - t0

    lines_hdr = ["mg_pair_dynamics_validation v1.1", "",
             "mu %.2f  Lsp %.3f  grid %dx%d  dx %.3f  xi %.2f  d(t) = %.2f + %.2f t,"
             " t in [0, %.2f], dt %.3f (%d frames)  window %.2f" %
             (args.mu, g["Lsp"], len(g["x"]), len(g["y"]), g["dx"], args.xi,
              args.d0, args.v, args.t_end, args.dt, len(ts), args.win_frac),
             ""]
    series, bins = [], []
    snap = None
    for i in range(1, len(ts) - 1):
        fr2, e2 = make(i + 1, e1["n"])
        fr = fr1
        z_t = (e2["z"] - e0["z"]) / (2 * args.dt)
        th_t = np.imag(np.conj(e1["z"]) * z_t) / np.maximum(np.abs(e1["z"]) ** 2, 1e-14)
        k1_t = (e2["k1"] - e0["k1"]) / (2 * args.dt)
        k2_t = (e2["k2"] - e0["k2"]) / (2 * args.dt)
        w_t = 2 * (e1["k1"] * k1_t + e1["k2"] * k2_t)
        th_t_ex = fr["theta_d"] * dprime
        k1t_ex, k2t_ex = fr["thd_x"] * dprime, fr["thd_y"] * dprime
        w_t_ex = 2 * (fr["k1"] * k1t_ex + fr["k2"] * k2t_ex)
        # the estimator's k is a director-signed field; compare k_t and w_t
        # after aligning the sign of k with the exact k pointwise
        sgn = np.sign(e1["k1"] * fr["k1"] + e1["k2"] * fr["k2"])
        sgn[sgn == 0] = 1
        k1_t, k2_t, th_t_s = k1_t * sgn, k2_t * sgn, th_t * sgn
        ekt = np.hypot(k1_t - k1t_ex, k2_t - k2t_ex)
        rkt = np.hypot(k1t_ex, k2t_ex)
        eth = th_t_s - th_t_ex
        ewt = w_t - w_t_ex
        d = dfun(ts[i])
        row = {"t": ts[i], "d": d, "d_over_Lsp": d / g["Lsp"]}
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = win & (fr["dist"] >= lo) & (fr["dist"] < hi)
            if m.sum() < 10:
                continue
            b = {"t": ts[i], "d": d, "lo": lo, "hi": hi, "n": int(m.sum()),
                 "theta_t": rel_rms(eth, th_t_ex, m),
                 "k_t": rel_rms(ekt, rkt, m),
                 "w_t": rel_rms(ewt, w_t_ex, m),
                 # bin error against the window-wide RMS of the exact field
                 "k_t_glob": float(np.sqrt(np.mean(ekt[m] ** 2))
                                   / np.sqrt(np.mean(rkt[win] ** 2))),
                 "w_t_glob": float(np.sqrt(np.mean(ewt[m] ** 2))
                                   / np.sqrt(np.mean(w_t_ex[win] ** 2)))}
            bins.append(b)
            for q in ("theta_t", "k_t", "w_t", "k_t_glob", "w_t_glob"):
                row["%s[%g,%g)" % (q, lo, hi)] = b[q]
        # whole-window numbers (the honest single figure of merit)
        mw = win & (fr["dist"] >= args.core_excl)
        row["theta_t_win"] = rel_rms(eth, th_t_ex, mw)
        row["k_t_win"] = rel_rms(ekt, rkt, mw)
        row["w_t_win"] = rel_rms(ewt, w_t_ex, mw)
        # weak-form proxy: the pointwise error, masked to the window and to
        # the outside of the cores, integrated against a Gaussian test
        # function of width test_sigma spacings, scored on the window eroded
        # by three test widths (so neither the cores nor the taper leak in).
        # Normalised by the RMS of the UNSMOOTHED exact field on the same
        # interior, so the weak number is comparable to the pointwise one.
        sg = args.test_sigma * g["Lsp"] / g["dx"]
        keep = mw.astype(float)
        inner = binary_erosion(mw, iterations=int(np.ceil(2.5 * sg)))
        row["n_inner"] = int(inner.sum())
        def weak_res(err_f):
            return gaussian_filter(err_f * keep, sg, mode="nearest")
        row["theta_t_weak"] = rel_rms(weak_res(eth), th_t_ex, inner)
        row["w_t_weak"] = rel_rms(weak_res(ewt), w_t_ex, inner)
        a1 = weak_res(k1_t - k1t_ex)
        a2 = weak_res(k2_t - k2t_ex)
        row["k_t_weak"] = float(np.sqrt(np.mean((a1 ** 2 + a2 ** 2)[inner]))
                                / np.sqrt(np.mean(rkt[inner] ** 2)))
        # the same interior, pointwise, for a like-for-like comparison
        row["theta_t_inner"] = rel_rms(eth, th_t_ex, inner)
        row["k_t_inner"] = rel_rms(ekt, rkt, inner)
        row["w_t_inner"] = rel_rms(ewt, w_t_ex, inner)
        series.append(row)
        if snap is None and d >= args.snap_d:
            snap = (i, th_t_s, th_t_ex, k1_t, k1t_ex, w_t, w_t_ex)
        fr0, e0, fr1, e1 = fr1, e1, fr2, e2
    t_est = time.time() - t0

    lines = lines_hdr + ["estimator time %.1f s" % t_est, ""]
    # time-step check on one frame: redo with dt/2 using fresh frames
    i = len(ts) // 2
    tt = ts[i]
    out_dt = []
    for h in (args.dt, args.dt / 2):
        fr0 = pair_field(g, dfun(tt - h), args.xi)
        fr2 = pair_field(g, dfun(tt + h), args.xi)
        frc = pair_field(g, dfun(tt), args.xi)
        e0 = estimate_frame(fr0["u"], g, args.sigma_dir, None, taper)
        e1 = estimate_frame(frc["u"], g, args.sigma_dir, e0["n"], taper)
        e2 = estimate_frame(fr2["u"], g, args.sigma_dir, e1["n"], taper)
        z_t = (e2["z"] - e0["z"]) / (2 * h)
        th_t = np.imag(np.conj(e1["z"]) * z_t) / np.maximum(np.abs(e1["z"]) ** 2, 1e-14)
        sgn = np.sign(e1["k1"] * frc["k1"] + e1["k2"] * frc["k2"]); sgn[sgn == 0] = 1
        th_t_ex = frc["theta_d"] * dprime
        m = win & (frc["dist"] >= 1.0)
        out_dt.append((h, rel_rms(th_t * sgn - th_t_ex, th_t_ex, m)))
    lines.append("time-step check at t = %.2f (d = %.2f), theta_t rel RMS beyond one"
                 " spacing: dt %.3f -> %.4f, dt/2 %.3f -> %.4f" %
                 (tt, dfun(tt), out_dt[0][0], out_dt[0][1], out_dt[1][0], out_dt[1][1]))
    lines.append("")

    # table: errors by bin at three separations
    lines.append("relative RMS errors by distance-to-core bin (spacing units);"
                 " k_t and w_t also against the window-wide RMS of the exact"
                 " field (glob), since both decay like 1/r^2 away from the pair")
    picks = [series[0], series[len(series) // 2], series[-1]]
    for row in picks:
        lines.append("  d = %.2f (%.2f spacings)   window beyond %.2g"
                     " spacings: theta_t %.4f  k_t %.4f  w_t %.4f" %
                     (row["d"], row["d_over_Lsp"], args.core_excl,
                      row["theta_t_win"], row["k_t_win"], row["w_t_win"]))
        lines.append("     interior (eroded by 2.5 test widths, n = %d):"
                     " pointwise theta_t %.4f  k_t %.4f  w_t %.4f ;"
                     " weak residual (test width %.2g spacing): theta_t %.4f"
                     "  k_t %.4f  w_t %.4f" %
                     (row["n_inner"], row["theta_t_inner"], row["k_t_inner"],
                      row["w_t_inner"], args.test_sigma, row["theta_t_weak"],
                      row["k_t_weak"], row["w_t_weak"]))
        for lo, hi in zip(edges[:-1], edges[1:]):
            key = "[%g,%g)" % (lo, hi)
            if ("theta_t" + key) in row:
                lines.append("    %-10s theta_t %.4f   k_t %.4f (glob %.4f)"
                             "   w_t %.4f (glob %.4f)" %
                             (key, row["theta_t" + key], row["k_t" + key],
                              row["k_t_glob" + key], row["w_t" + key],
                              row["w_t_glob" + key]))
    lines.append("")

    # outputs
    stamp = "mu%.2f_xi%.2g_sd%.2g_ppw%d" % (args.mu, args.xi, args.sigma_dir, args.ppw)
    out = Path(args.outdir) / stamp
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as fh:
        json.dump({k: getattr(args, k) for k in vars(args)}, fh, indent=2, default=str)

    def write_csv(rows, path):
        keys = sorted({k for r in rows for k in r}, key=lambda s: (s not in ("t", "d", "d_over_Lsp", "lo", "hi", "n"), s))
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
    write_csv(series, out / "timeseries.csv")
    write_csv(bins, out / "bins.csv")

    # figure: errors vs separation, one curve per bin, three panels
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    dd = [r["d_over_Lsp"] for r in series]
    for ax, q in zip(axs, ("theta_t", "k_t_glob", "w_t_glob")):
        for lo, hi in zip(edges[:-1], edges[1:]):
            key = q + "[%g,%g)" % (lo, hi)
            vals = [r.get(key, np.nan) for r in series]
            ax.plot(dd, vals, "-", lw=1.2, label="[%g,%g)" % (lo, hi))
        qw = q.replace("_glob", "") + "_weak"
        ax.plot(dd, [r[qw] for r in series], "k--", lw=1.4, label="weak, window")
        ax.set_yscale("log")
        ax.set_xlabel("pair separation / spacing")
        ax.set_title(q)
        ax.grid(True, which="both", alpha=0.3)
    axs[0].set_ylabel("relative RMS error")
    axs[-1].legend(fontsize=7, title="dist/spacing")
    fig.tight_layout()
    fig.savefig(out / "errors_vs_separation.png", dpi=180)
    plt.close(fig)

    # snapshot figure
    if snap is not None:
        i, th_t_s, th_t_ex, k1_t, k1t_ex, w_t, w_t_ex = snap
        d = dfun(ts[i])
        L = 2.0 * g["Lsp"] + d / 2
        ix = np.abs(g["x"]) < L
        iy = np.abs(g["y"]) < 1.5 * g["Lsp"]
        ext = [g["x"][ix][0], g["x"][ix][-1], g["y"][iy][0], g["y"][iy][-1]]
        fig, axs = plt.subplots(3, 3, figsize=(11, 8))
        for r_, (lab, est_, ex_) in enumerate((("theta_t", th_t_s, th_t_ex),
                                               ("k1_t", k1_t, k1t_ex),
                                               ("w_t", w_t, w_t_ex))):
            v = np.percentile(np.abs(ex_[np.ix_(iy, ix)]), 98)
            for c_, (t_, F) in enumerate((("exact", ex_), ("estimator", est_),
                                          ("difference", est_ - ex_))):
                im = axs[r_, c_].imshow(F[np.ix_(iy, ix)], origin="lower", extent=ext,
                                        cmap="RdBu_r", vmin=-v, vmax=v)
                axs[r_, c_].set_title("%s, %s" % (lab, t_), fontsize=9)
                axs[r_, c_].plot([-d / 2, d / 2], [0, 0], "k+", ms=7)
            fig.colorbar(im, ax=axs[r_, 2], shrink=0.85)
        fig.suptitle("separating pair, mu = %.2f, d = %.2f (%.2f spacings)" %
                     (args.mu, d, d / g["Lsp"]))
        fig.tight_layout()
        fig.savefig(out / ("snapshot_d%.1f.png" % d), dpi=170)
        plt.close(fig)

    report = "\n".join(lines)
    (out / "report.txt").write_text(report)
    print(report)
    print("outputs ->", out)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    outdir = str(Path("results") / "mg_pair_dynamics_validation")
    mu = 0.85
    xi = 1.0                    # core amplitude width (nucleation bench value)
    sigma_dir = 0.4
    ppw = 20                    # points per wavelength
    n_xper = 16                 # box width in PN spacings
    n_yper = 10                 # box height in vertical periods
    d0 = 0.5                    # initial separation
    v = 1.0                     # separation speed
    t_end = 16.0
    dt = 0.05
    win_frac = 0.35
    taper_frac = 0.08           # edge taper before the estimator (0 = off)
    core_excl = 0.5             # spacings excluded for the whole-window numbers
    test_sigma = 0.5            # weak-form proxy: Gaussian test width, spacings
    edges = [0, 0.25, 0.5, 1, 2, 4]
    snap_d = 8.0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default=_Args.outdir)
    p.add_argument("--mu", type=float, default=_Args.mu)
    p.add_argument("--xi", type=float, default=_Args.xi)
    p.add_argument("--sigma-dir", type=float, dest="sigma_dir", default=_Args.sigma_dir)
    p.add_argument("--ppw", type=int, default=_Args.ppw)
    p.add_argument("--n-xper", type=float, dest="n_xper", default=_Args.n_xper)
    p.add_argument("--n-yper", type=float, dest="n_yper", default=_Args.n_yper)
    p.add_argument("--d0", type=float, default=_Args.d0)
    p.add_argument("--v", type=float, default=_Args.v)
    p.add_argument("--t-end", type=float, dest="t_end", default=_Args.t_end)
    p.add_argument("--dt", type=float, default=_Args.dt)
    p.add_argument("--win-frac", type=float, dest="win_frac", default=_Args.win_frac)
    p.add_argument("--snap-d", type=float, dest="snap_d", default=_Args.snap_d)
    p.add_argument("--taper-frac", type=float, dest="taper_frac", default=_Args.taper_frac)
    p.add_argument("--core-excl", type=float, dest="core_excl", default=_Args.core_excl)
    p.add_argument("--test-sigma", type=float, dest="test_sigma", default=_Args.test_sigma)
    cli = p.parse_args()
    cli.edges = _Args.edges
    sys.exit(run(cli))
