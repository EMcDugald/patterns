# pipelines/rcn_rollout.py
"""
Free rollouts of the identified amplitude and wavenumber laws (tracker
T2.3): integrate a law forward from one measured frame on its OWN output
and compare with the measured field at later frames. This is the
closed-model test that the teacher-forced prediction scores of
wsindy_theta_t v1.6 (R18-9, R18-10) do not make: there the measured
fields are fed in at every step, here nothing is fed in after the
initial frame.

Laws rolled out (coefficients fitted here by OLS on the named support
over the fit window, or given with --coef_*):
  A far / A band   A_t = c1 A + c3 A^3 + c4 A w^2        (SH: (R, -3/4, -1))
                   an ODE per pixel; w is frozen at the initial frame's
                   value when --w_frozen 1 (default), or taken from the
                   measured frames when --w_frozen 0 (semi-free rollout).
  w band           w_t = D Lap w - g w                    (archival law, R-4)
                   a linear diffusion PDE on the crop, spectral in x
                   (periodic) and in y with the taper zone acting as the
                   boundary; Crank-Nicolson in Fourier space (exact for a
                   linear constant-coefficient law), sub-stepped.
Rolled fields are compared with the measured field at horizons h frames
after the start, on the domain (far > far_from L; band < band_to L about
the row with core discs of core_excl L removed; kernel-averaged when
--smooth > 0, in spacings), by
    score(h) = rms(rolled - measured) / rms(measured - initial)
so the null "the field does not move" scores 1, and by the relative
error rms(rolled - measured) / rms(measured). Both are printed per
horizon with the persistence null beside them.

Reading: a score well below 1 that stays below 1 over several
relaxation times is a closed model of that field's evolution; a score
that starts below 1 and climbs past it is a law that describes the
rate but not the trajectory (error feedback), which is what the
teacher-forced scores cannot see.

Changelog
---------
v1.0  (21 Sep) first version. Smoke-tested on the 2d RCN control (A
      static, w relaxing under RCN) and on the 0.85 tail (stationary,
      both nulls score ~1 by construction).

Run (from pipelines/):
  python rcn_rollout.py --npz <T25 or fine npz> --law A_far  --t0 0 --t_fit 1.0 --horizons 0.2,0.4,0.8,1.6
  python rcn_rollout.py --npz <T25 npz> --law A_band --t0 <post start> --t_fit 2.0 --horizons 0.4,0.8,1.6,3.2 --core_excl 0.5
  python rcn_rollout.py --npz <T25 npz> --law w_band --t0 <post start> --t_fit 2.0 --horizons 0.4,0.8,1.6,3.2 --core_excl 0.5
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT / "src", _ROOT / "src" / "op_extract", _ROOT / "src" / "utils", _HERE):
    sys.path.insert(0, str(p))
from mg_zigzag_theta_t import load_npz, estimate, find_cores, dist_to_cores   # noqa: E402
from op_extract import t1_common as t1c                                        # noqa: E402

VERSION = "rcn_rollout v1.0"


def r2(y, yh):
    return float(1 - np.sum((y - yh) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30))


def laplacian_spectral(F, Lx, Ly):
    ny, nx = F.shape
    qx = 2 * np.pi * np.fft.fftfreq(nx, Lx / nx); qy = 2 * np.pi * np.fft.fftfreq(ny, Ly / ny)
    q2 = qx[None, :] ** 2 + qy[:, None] ** 2
    return np.real(np.fft.ifft2(-q2 * np.fft.fft2(F)))


def main(args):
    t_start = time.time()
    D = load_npz(args.npz)
    u, x, y, tt, md = D["u"], D["x"], D["y"], D["t"], D["md"]
    mu = float(md.get("mu", args.mu)); k1c = np.sqrt(1 - mu ** 2); Lsp = np.pi / k1c
    dx = float(x[1] - x[0]); Lx, Ly = len(x) * dx, len(y) * dx
    X, Y = np.meshgrid(x, y)
    y_row = float(y[int(md.get("gb_row_in_crop", len(y) // 2))])
    dt = float(np.median(np.diff(tt)))
    horizons = [float(h) for h in str(args.horizons).split(",") if h.strip()]
    i0 = int(np.argmin(np.abs(tt - args.t0)))
    i_fit1 = int(np.argmin(np.abs(tt - (args.t0 + args.t_fit))))
    i_end = min(len(tt) - 1, int(np.argmin(np.abs(tt - (args.t0 + max(horizons))))))
    frames = list(range(i0, i_end + 1))
    stem = Path(args.npz).stem
    out = Path(args.outdir) / stem / ("%s_t%.2f_fit%.2f_ce%g_sm%g" % (args.law, args.t0, args.t_fit, args.core_excl, args.smooth))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as fh:
        json.dump({k: getattr(args, k) for k in vars(args)}, fh, indent=2, default=str)
    lines = [VERSION, "%s  mu %.3f (L %.2f)  law %s  start t %.3f (frame %d)  fit window %.3f-%.3f  horizons %s  dt %.4f"
             % (stem, mu, Lsp, args.law, tt[i0], i0, tt[i0], tt[i_fit1], horizons, dt)]

    # estimator on the needed frames
    taper = t1c.edge_taper(x, y, frac=args.taper_frac)
    A, W, n_prev = {}, {}, None
    for i in frames:
        e = estimate(u[..., i], Lx, Ly, args.sigma_dir, n_prev, taper); n_prev = e["n"]
        s = np.sign(e["k1"].mean()) or 1.0
        A[i] = e["A"]; W[i] = (e["k1"] * s) ** 2 + (e["k2"] * s) ** 2 - 1.0
        if i == i0:
            win = (np.abs(X - x.mean()) < 0.4 * Lx) & (np.abs(Y - y.mean()) < 0.4 * Ly)
            xc, yc = find_cores(e["A"], x, y, win, args.core_frac, merge_dist=0.5 * Lsp)
    lines.append("  estimator on %d frames, %.1f s; %d cores at the start frame" % (len(frames), time.time() - t_start, len(xc)))
    d_row = np.abs(Y - y_row) / Lsp
    dcore = dist_to_cores(X, Y, xc, yc) / Lsp
    if args.law == "A_far":
        dom = (d_row > args.far_from) & win
    else:
        dom = (d_row < args.band_to) & win & (dcore > args.core_excl)
    lines.append("  domain %s: %d pixels" % ("far" if args.law == "A_far" else "band", int(dom.sum())))
    sm = args.smooth * Lsp / dx
    S = (lambda F: gaussian_filter(F, sm)) if args.smooth > 0 else (lambda F: F)

    # ---- fit the law on the fit window (centred differences in time, pointwise on the domain)
    fit_frames = [i for i in frames if i0 < i < i_fit1]
    if args.law in ("A_far", "A_band"):
        rows, rhs = [], []
        for i in fit_frames:
            At = (A[i + 1] - A[i - 1]) / (2 * dt)
            a, w = A[i][dom], W[i][dom]
            rows.append(np.column_stack([a, a ** 3, a * w ** 2])); rhs.append(At[dom])
        Th = np.vstack(rows); b = np.concatenate(rhs)
        if args.coef_A:
            c = np.array([float(v) for v in args.coef_A.split(",")])
            lines.append("  coefficients given: (c1, c3, c4) = (%.3f, %.3f, %.3f)" % tuple(c))
        else:
            c, *_ = np.linalg.lstsq(Th, b, rcond=None)
            lines.append("  fitted A_t = %.3f A + %.3f A^3 + %.3f A w^2   (pointwise OLS on %d rows, R2 %.3f; SH (R, -3/4, -1))"
                         % (c[0], c[1], c[2], len(b), r2(b, Th @ c)))
        # rollout: RK4 per pixel, w frozen or measured
        nsub = max(1, int(np.ceil(dt / args.dt_sub)))
        h = dt / nsub
        Ar = A[i0].copy(); Wf = W[i0].copy()
        rolled = {i0: Ar.copy()}
        for i in frames[1:]:
            for _ in range(nsub):
                wloc = Wf if args.w_frozen else W[i - 1]
                f = lambda a: c[0] * a + c[1] * a ** 3 + c[2] * a * wloc ** 2
                k1_ = f(Ar); k2_ = f(Ar + 0.5 * h * k1_); k3_ = f(Ar + 0.5 * h * k2_); k4_ = f(Ar + h * k3_)
                Ar = Ar + h / 6 * (k1_ + 2 * k2_ + 2 * k3_ + k4_)
            rolled[i] = Ar.copy()
        field = A
    else:
        rows, rhs = [], []
        for i in fit_frames:
            wt = (W[i + 1] - W[i - 1]) / (2 * dt)
            lap = laplacian_spectral(W[i] * taper, Lx, Ly)
            rows.append(np.column_stack([S(lap)[dom], S(W[i])[dom]])); rhs.append(S(wt)[dom])
        Th = np.vstack(rows); b = np.concatenate(rhs)
        if args.coef_w:
            c = np.array([float(v) for v in args.coef_w.split(",")])
            lines.append("  coefficients given: (D, g) = (%.3f, %.3f)" % tuple(c))
        else:
            c, *_ = np.linalg.lstsq(Th, b, rcond=None)
            lines.append("  fitted w_t = %.3f Lap w + %.3f w   (pointwise OLS on the smoothed fields, %d rows, R2 %.3f; archival 0.62 Lap w - 0.17 w)"
                         % (c[0], c[1], len(b), r2(b, Th @ c)))
        Dc, gc = c[0], c[1]
        # Crank-Nicolson in Fourier space on the tapered field
        ny, nx = W[i0].shape
        qx = 2 * np.pi * np.fft.fftfreq(nx, Lx / nx); qy = 2 * np.pi * np.fft.fftfreq(ny, Ly / ny)
        q2 = qx[None, :] ** 2 + qy[:, None] ** 2
        lam = -Dc * q2 + gc
        nsub = max(1, int(np.ceil(dt / args.dt_sub))); h = dt / nsub
        prop = ((1 + 0.5 * h * lam) / (1 - 0.5 * h * lam)) ** nsub
        Wh = np.fft.fft2(W[i0] * taper)
        rolled = {i0: W[i0].copy()}
        for i in frames[1:]:
            Wh = Wh * prop
            rolled[i] = np.real(np.fft.ifft2(Wh))
        field = W

    # ---- scores
    lines.append("")
    lines.append("  %-8s %-6s %10s %10s %10s %10s   (score = rms(rolled - measured) / rms(measured - initial); null 1)"
                 % ("horizon", "frame", "score", "rel_err", "rel_null", "R2_incr"))
    res = []
    F0 = S(field[i0])
    for hz in horizons:
        i = min(len(tt) - 1, int(round(i0 + hz / dt)))
        if i not in rolled:
            continue
        Fm, Fr = S(field[i]), S(rolled[i])
        incr = (Fm - F0)[dom]; err = (Fr - Fm)[dom]
        if np.sqrt(np.mean(incr ** 2)) < 1e-12:
            lines.append("  %-8.3f %-6d   field stationary on the domain (no measured change); nothing to score" % (tt[i] - tt[i0], i))
            continue
        score = float(np.sqrt(np.mean(err ** 2)) / np.sqrt(np.mean(incr ** 2)))
        rel = float(np.sqrt(np.mean(err ** 2)) / np.sqrt(np.mean(Fm[dom] ** 2)))
        rel_null = float(np.sqrt(np.mean(incr ** 2)) / np.sqrt(np.mean(Fm[dom] ** 2)))
        rr = r2(incr, (Fr - F0)[dom])
        lines.append("  %-8.3f %-6d %10.3f %10.4f %10.4f %10.3f" % (tt[i] - tt[i0], i, score, rel, rel_null, rr))
        res.append(dict(horizon=float(tt[i] - tt[i0]), frame=i, score=score, rel_err=rel, rel_null=rel_null, R2_incr=rr))
    lines.append("")
    lines.append("elapsed %.1f s" % (time.time() - t_start))
    rep = "\n".join(lines)
    (out / "report.txt").write_text(rep)
    with open(out / "scores.json", "w") as fh:
        json.dump(dict(coef=[float(v) for v in c], scores=res), fh, indent=1)
    print(rep); print("outputs ->", out)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    npz = ""
    outdir = str(_HERE / "results" / "rcn_rollout")
    law = "A_far"                # A_far | A_band | w_band
    t0 = 0.0                     # start frame (time)
    t_fit = 1.0                  # fit window length after t0 (time units)
    horizons = "0.2,0.4,0.8,1.6" # comparison horizons after t0 (time units)
    coef_A = ""                  # "c1,c3,c4" to skip the fit
    coef_w = ""                  # "D,g" to skip the fit
    w_frozen = 1                 # A laws: 1 = w frozen at t0 (free rollout); 0 = measured w per frame (semi-free)
    dt_sub = 0.01                # integrator sub-step
    smooth = 0.0                 # kernel-average the compared fields, spacings (0 = pointwise); 0.5 recommended for w
    mu = 0.85
    far_from = 4.0
    band_to = 2.0
    core_excl = 0.5
    sigma_dir = 0.4
    taper_frac = 0.06
    core_frac = 0.70


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in dir(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    cli = p.parse_args()
    if not cli.npz:
        p.error("--npz is required")
    sys.exit(main(cli))