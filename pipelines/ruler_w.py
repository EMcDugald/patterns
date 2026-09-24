# pipelines/ruler_w.py
"""
Estimator-free wavenumber ruler for a stripe pattern about a row of cores.

Idea. Wherever u = A cos theta, the phase advances by exactly pi between
consecutive zero crossings of u along any line, for any positive
amplitude. Along a vertical line x = const the crossing spacing s gives
the y-component of the wavevector, k2 = pi / s, at the interval's
midpoint. The x-component k1 = cos alpha is pinned by the periodicity of
the crop (an x-ruler reads it in every bin of every field), so
    w = k1^2 + k2^2 - 1
at each crossing midpoint, with no phase estimator and no smoothing.
Each value is tagged by its distance to the nearest core (in spacings
L = pi/k1) and binned. Where the stripes curve strongly (within about
one spacing of a core) the pinned-k1 assumption fails and the ruler is
not read; between the cores it is validated on the exact Pismen
pattern to three decimals (RESULTS_addendum_0918 R18-12).

Modes: --npz <SH tail> (last frame; cores from the mg amplitude dip),
--mode pn (the Pismen row of pn_field.pismen_on_sh_grid at the zigzag
spacing on a matching grid; analytic w from grad theta for reference).
Outputs (results/ruler_w/<stem>/): table.txt, ruler.csv, ruler_w.png
(pattern with the ruler lines and crossings; one line's profile with
the intervals and their k2; w against core distance).

Changelog
---------
v1.1  (19 Sep, Ed's catch) pn mode keeps only the tilted part of the row:
      the construction's weights decay as exp(-j pi), so beyond the first
      ~9 cores the pattern is untilted rolls where the pinned k1 is false
      (every crossing there reads w = +0.28). Columns with |k1| < 0.7 cos
      alpha at 2 L above the row are masked and the cores there dropped.
v1.0  (19 Sep) from ruler_y_test.py of 18 Sep; same numbers.
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT / "src", _ROOT / "src" / "op_extract", _ROOT / "src" / "utils", _HERE):
    sys.path.insert(0, str(p))
from mg_zigzag_theta_t import load_npz, estimate, find_cores        # noqa: E402
from op_extract import t1_common as t1c                              # noqa: E402
try:
    from op_extract.pn_field import pismen_on_sh_grid                # noqa: E402
except ImportError:                                                  # pragma: no cover
    from pn_field import pismen_on_sh_grid                           # noqa: E402
import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

BINS = [(0.5, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0)]


def zero_crossings(f, y):
    if not np.all(np.isfinite(f)):
        return np.zeros(0)
    s = np.sign(f)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    return y[idx] - f[idx] * (y[idx + 1] - y[idx]) / (f[idx + 1] - f[idx])


def vertical_ruler(u, x, y, y_row, xc, yc, L, k1c, ymax_L, stride):
    """rows: (x, y_mid, spacing, k2, w, dist_to_core_L, side)"""
    rows = []
    for i in range(0, len(x), stride):
        z = zero_crossings(u[:, i], y)
        z = z[np.abs(z - y_row) < ymax_L * L]
        for a, b in zip(z[:-1], z[1:]):
            ym = 0.5 * (a + b); k2 = np.pi / (b - a)
            dist = float(np.min(np.hypot(x[i] - xc, ym - yc)) / L)
            rows.append((float(x[i]), float(ym), float(b - a), float(k2), float(k1c ** 2 + k2 ** 2 - 1), dist, float(np.sign(ym - y_row))))
    return np.array(rows)


def table(arr, label):
    lines = ["%s  (n %d crossing intervals)" % (label, len(arr)),
             " bin (L)      n    median w   IQR               frac(w<0)  upper   lower   median k2"]
    for lo, hi in BINS:
        m = (arr[:, 5] >= lo) & (arr[:, 5] < hi)
        if m.sum() < 10:
            continue
        w = arr[m, 4]; q = np.percentile(w, [25, 75])
        up = arr[m & (arr[:, 6] > 0), 4]; lo_ = arr[m & (arr[:, 6] < 0), 4]
        lines.append(" %4.2f-%-4.2f  %5d   %+.3f   [%+.3f,%+.3f]   %.2f      %+.3f  %+.3f   %.3f" %
                     (lo, hi, m.sum(), np.median(w), q[0], q[1], np.mean(w < 0),
                      np.median(up) if len(up) else np.nan, np.median(lo_) if len(lo_) else np.nan, np.median(arr[m, 3])))
    return "\n".join(lines)


def figure(u, x, y, y_row, xc, yc, L, k1c, arr, exact, out, title, args):
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])
    # (a) pattern, ruler lines, crossings
    ax = fig.add_subplot(gs[0, :2])
    ysel = np.abs(y - y_row) < 3.2 * L
    ax.imshow(np.nan_to_num(u[ysel]), origin="lower", extent=[x[0], x[-1], y[ysel][0], y[ysel][-1]], cmap="gray", aspect="equal")
    xs_lines = x[::args.stride * 8]
    for xl in xs_lines:
        ax.axvline(xl, color="tab:orange", lw=0.5, alpha=0.6)
    sel = np.isin(arr[:, 0], xs_lines)
    ax.scatter(arr[sel, 0], arr[sel, 1] - arr[sel, 2] / 2, s=4, color="tab:cyan", label="zero crossings of u along x = const")
    ax.plot(xc, yc, "o", mfc="none", mec="red", ms=8, label="cores")
    ax.axhline(y_row, color="red", lw=0.6, ls="--")
    ax.set_title("(a) %s: vertical ruler lines (every 8th shown), zero crossings, cores" % title, fontsize=9)
    ax.legend(fontsize=7, loc="upper right"); ax.set_xlabel("x"); ax.set_ylabel("y")
    # (b) one line's profile between two cores, crossings and k2 per interval
    ax = fig.add_subplot(gs[0, 2])
    xm = 0.5 * (xc[len(xc) // 2] + xc[len(xc) // 2 + 1]) if len(xc) > 1 else x[len(x) // 2]
    i = int(np.argmin(np.abs(x - xm)))
    prof = u[:, i]; z = zero_crossings(prof, y); z = z[np.abs(z - y_row) < 3 * L]
    ax.plot(y[ysel] - y_row, prof[ysel], "k-", lw=0.8)
    ax.plot(z - y_row, np.zeros_like(z), "o", color="tab:cyan", ms=4)
    for a, b in zip(z[:-1], z[1:]):
        k2 = np.pi / (b - a); w = k1c ** 2 + k2 ** 2 - 1
        ax.annotate("%.2f" % k2, (0.5 * (a + b) - y_row, 0.85), fontsize=6, ha="center", color="tab:blue")
        ax.annotate("%+.3f" % w, (0.5 * (a + b) - y_row, -0.9), fontsize=6, ha="center", color="tab:red")
    ax.set_ylim(-1.05, 1.05); ax.set_xlabel("y - y_row"); ax.set_title("(b) u along x = %.1f (midway between two cores)\nk2 = pi / spacing above; w = k1^2 + k2^2 - 1 below" % x[i], fontsize=8)
    # (c) w against core distance
    ax = fig.add_subplot(gs[1, :])
    ax.scatter(arr[:, 5], arr[:, 4], s=3, alpha=0.25, color="gray", label="every crossing interval")
    cen, med, q1, q3 = [], [], [], []
    for lo, hi in BINS:
        m = (arr[:, 5] >= lo) & (arr[:, 5] < hi)
        if m.sum() < 10:
            continue
        cen.append(0.5 * (lo + hi)); med.append(np.median(arr[m, 4])); q = np.percentile(arr[m, 4], [25, 75]); q1.append(q[0]); q3.append(q[1])
    ax.errorbar(cen, med, yerr=[np.array(med) - np.array(q1), np.array(q3) - np.array(med)], fmt="o-", color="tab:blue", capsize=3, label="ruler: median and IQR per bin")
    if exact is not None:
        ax.plot(exact[0], exact[1], "s--", color="tab:orange", label="analytic w of the PN phase (median per bin)")
    ax.axhline(0, color="k", lw=0.6); ax.axvspan(0, 1.0, color="red", alpha=0.06, label="ruler not read (< 1 L: curved stripes)")
    ax.set_xlim(0.4, 4.0); ax.set_ylim(-0.25, 0.25)
    ax.set_xlabel("distance to nearest core (spacings L)"); ax.set_ylabel("w = |k|^2 - 1")
    ax.set_title("(c) wavenumber deviation from the ruler; w < 0 stretched, w > 0 compressed", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(out / "ruler_w.png", dpi=130); plt.close(fig)


def main(args):
    if args.mode == "pn":
        mu = args.mu; k1c = float(np.sqrt(1 - mu ** 2)); L = np.pi / k1c
        dx = args.dx; y = np.arange(-args.Ly / 2, args.Ly / 2, dx); nx = int(round(args.n_xper * 2 * np.pi / k1c / dx))
        x = (np.arange(nx) - nx // 2) * dx; X, Y = np.meshgrid(x, y); y_row = 0.0
        nd = max(1, int(round(nx * dx / L))); xd = (np.arange(nd) - nd / 2 + 0.5) * L
        th, _ = pismen_on_sh_grid(X, Y, xd); u = np.cos(th)
        k1e = np.gradient(th, dx, axis=1); k2e = np.gradient(th, dx, axis=0); we = k1e ** 2 + k2e ** 2 - 1
        # v1.1: the construction's weights decay as exp(-j pi) along the row, so beyond the
        # first several cores the pattern is untilted rolls (k1 -> 0); the ruler's pinned k1
        # is false there. Keep only the columns whose far field (2 L above the row) is tilted,
        # |k1| > pn_tilt_frac * cos alpha, and only the cores inside that range.
        jrow = int(np.argmin(np.abs(y - 2 * L)))
        tilted = np.abs(k1e[jrow]) > args.pn_tilt_frac * k1c
        xgood = x[tilted]
        keep_cols = tilted
        xd_good = xd[(xd > xgood.min()) & (xd < xgood.max())]
        u = np.where(keep_cols[None, :], u, np.nan)                  # masked columns yield no crossings
        xc, yc = xd_good, np.zeros(len(xd_good))
        print("pn mode: %d of %d cores lie in the tilted part x in [%.1f, %.1f]; the rest is rolls and is excluded" %
              (len(xd_good), nd, xgood.min(), xgood.max()))
        dc = np.min(np.hypot(X[..., None] - xd_good[None, None, :], Y[..., None]), axis=-1) / L
        exact = ([0.5 * (lo + hi) for lo, hi in BINS], [np.median(we[(dc >= lo) & (dc < hi) & (np.abs(Y) < 3 * L) & keep_cols[None, :]]) for lo, hi in BINS])
        stem = "pn_row_mu%.2f" % mu; title = "Pismen row, %d cores, mu %.2f" % (nd, mu)
    else:
        D = load_npz(args.npz); u3, x, y, tt, md = D["u"], D["x"], D["y"], D["t"], D["md"]
        mu = float(md.get("mu", args.mu)); k1c = float(np.sqrt(1 - mu ** 2)); L = np.pi / k1c
        dx = float(x[1] - x[0]); Lx, Ly = len(x) * dx, len(y) * dx
        X, Y = np.meshgrid(x, y); y_row = float(y[int(md.get("gb_row_in_crop", len(y) // 2))])
        u = u3[..., args.frame]
        e = estimate(u, Lx, Ly, args.sigma_dir, None, t1c.edge_taper(x, y, frac=args.taper_frac))
        win = (np.abs(X - x.mean()) < 0.4 * Lx) & (np.abs(Y - y.mean()) < 0.4 * Ly)
        xc, yc = find_cores(e["A"], x, y, win, args.core_frac, merge_dist=0.5 * L)
        exact = None; stem = Path(args.npz).stem; title = "%s, frame %d (t %.1f), %d cores" % (stem[:40], args.frame, tt[args.frame], len(xc))
    arr = vertical_ruler(u, x, y, y_row, np.asarray(xc), np.asarray(yc), L, k1c, args.ymax_L, args.stride)
    out = Path(args.outdir) / stem; out.mkdir(parents=True, exist_ok=True)
    txt = "ruler_w v1.0   k1 = cos alpha = %.4f pinned; L = %.2f; lines every %d columns; |y - y_row| < %.1f L\n" % (k1c, L, args.stride, args.ymax_L)
    txt += table(arr, title)
    if exact is not None:
        txt += "\n analytic PN w per bin: " + "  ".join("%.2f-%.2f: %+.3f" % (lo, hi, v) for (lo, hi), v in zip(BINS, exact[1]))
    (out / "table.txt").write_text(txt)
    with open(out / "ruler.csv", "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["x", "y_mid", "spacing", "k2", "w", "dist_core_L", "side"]); wr.writerows(arr.tolist())
    figure(u, x, y, y_row, np.asarray(xc), np.asarray(yc), L, k1c, arr, exact, out, title, args)
    print(txt); print("outputs ->", out)
    return 0


class _Args:
    mode = "data"            # data | pn
    npz = ""
    frame = -1               # data: frame index (last = -1)
    outdir = str(Path("results") / "ruler_w")
    mu = 0.85                # pn mode (data: from the file)
    Ly = 80.0                # pn mode box height
    dx = 0.21                # pn mode grid step (the SH grid's)
    n_xper = 8               # pn mode periods in x
    stride = 2               # columns between ruler lines
    ymax_L = 3.0             # crossings used within this many spacings of the row
    sigma_dir = 0.4
    taper_frac = 0.06
    core_frac = 0.70
    pn_tilt_frac = 0.7       # pn mode: a column is genuine if |k1| at 2 L above the row exceeds this fraction of cos alpha


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in dir(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k); p.add_argument("--" + k, type=type(v), default=v)
    cli = p.parse_args()
    if cli.npz:
        cli.mode = "data"
    sys.exit(main(cli))
