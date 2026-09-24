# pipelines/rcn_free_stationary.py
"""
Free stationary fit of the fourth-order balance, weak form: target
Int[Lap div k phi] on a library of stationary columns, best subset per
support size with a holdout half in x, beside the two named laws
(relation B alone, Lap div k = lambda S, and B with J, Lap div k =
lambda_J S + m J). No rates are fitted. This is the R18-14 instrument
(RESULTS_addendum_0918), promoted from free_stationary.py with the
numerics unchanged: kernel radius 1.5 L, poly bump p = 6, inner
derivatives by the poly mollifier at eta 0, query centres every 4
pixels with the kernel inside the box, holdout x < median.

Library (weak columns at the query centres):
    S = w div k + k.grad w,  J = det grad k,  w,  Lap w,  div k,
    (div k)^2,  w^2 div k,  A_dev div k,  w div k
Target B = Int[Lap div k phi] = -Int[k . grad(Lap phi)] (fully weak).
Linear columns are fully weak (div k, w, Lap w); the products use the
inner mollified derivative before the outer integral, as in
wsindy_theta_t.

Modes
  data  --npz F[,F2,...]: one frame per file (--frame, default the last),
        the mg estimator (sigma_dir, taper), cores from the amplitude dip,
        domain |y - y_row| < band_to L two-sided (or --side upper/lower)
        with core discs of core_excl L removed and the window
        |x - xm| < win_frac Lx, |y - ym| < win_frac Ly. The estimator's
        k gets ONE scalar sign per frame (mean of k1 > 0). If the file
        carries k1_exact, k2_exact (2d or [ny, nx, nt]; the 2d RCN control
        writes them) the exact route is fitted beside the estimator route.
  pn    the Pismen row of pn_field.pismen_on_sh_grid at the zigzag spacing
        pi/k1 with the smooth core dip A = 0.816 tanh(r / pn_core)
        (ledger v1.5 --pn_amp 2, the fair estimator control), exact k
        (product rule on cos/sin of the analytic phase) and estimator k,
        upper band |Y| in (0, band_to L), discs core_excl L.

Numbers of record reproduced by this script (R18-14, 18 Sep):
  SH 0.85 tail estimator band: size 1 -1.85 S (0.967); size 2 -1.47 S
    - 1.32 J (0.974)
  SH 0.90 tail: size 1 -2.07 S (0.939); size 2 -1.35 S - 2.25 J (0.959)
  PN exact k: size 1 -0.20 S (0.514); {S, J} -0.28 S - 0.38 J (0.707)
  PN estimator: size 1 -8.9 J (0.883); {S, J} +0.47 S - 9.4 J (0.882)
Gauge: B and S are odd under theta -> -theta (the x-mirror of a zigzag
file, mirror_npz.py), J, w, A_dev are even, so on a mirrored file the J,
w and A_dev coefficients must flip sign and the rest must not (T1.6).

Outputs: <outdir>/<stem>/<stamp>/report.txt, table.txt (one line per
field and support with the run stamp as provenance), fits.csv,
config.json.  Default outdir: results/rcn_free_stationary/ beside this
script.

Changelog
---------
v1.0  (19 Sep) from free_stationary.py (18 Sep, R18-14): _Args + CLI,
      script-anchored output, stamped table, csv, --frame, --side,
      several files, the exact route from k1_exact/k2_exact when present
      (for the 2d RCN control, T1.2). Numerics identical: the R18-14
      numbers reproduce to the printed digits (checked on the 0.85 tail
      and the PN row, 19 Sep).

Run:
  python rcn_free_stationary.py --npz <tail npz>[,<tail npz>]
  python rcn_free_stationary.py --mode pn [--mu 0.85]
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT / "src", _ROOT / "src" / "op_extract", _ROOT / "src" / "utils", _HERE):
    sys.path.insert(0, str(p))

from mg_zigzag_theta_t import load_npz, estimate, find_cores, dist_to_cores   # noqa: E402
from wsindy_theta_t import kernel_d3, corr1d, inner_deriv                     # noqa: E402
from op_extract import t1_common as t1c                                        # noqa: E402
try:
    from op_extract.pn_field import pismen_on_sh_grid                          # noqa: E402
except ImportError:                                                            # pragma: no cover
    from pn_field import pismen_on_sh_grid                                     # noqa: E402

VERSION = "rcn_free_stationary v1.0"
LIB = ["S", "J", "w", "Lap w", "div k", "(div k)^2", "w^2 div k", "A_dev div k", "w div k"]


# ---------------------------------------------------------------------
# weak columns
# ---------------------------------------------------------------------

def columns(k1, k2, A, x, y, L, radius, poly_p, stride):
    """weak columns and the target at the query centres (IY, IX)"""
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    a = radius * L
    fx, fx1, fx2, fx3, mx = kernel_d3(a, dx, poly_p)
    fy, fy1, fy2, fy3, my = kernel_d3(a, dy, poly_p)
    K = lambda F, kx, ky: corr1d(corr1d(F, ky, 0, dy), kx, 1, dx)
    D = lambda F, ax: inner_deriv(F, 0.0, dx, dy, ax, "poly", poly_p)
    w = k1 ** 2 + k2 ** 2 - 1
    k1x, k1y, k2x, k2y = D(k1, "x"), D(k1, "y"), D(k2, "x"), D(k2, "y")
    divk = k1x + k2y
    J = k1x * k2y - k1y * k2x
    Ad = A - np.nanmedian(A)
    Spt = w * divk + k1 * D(w, "x") + k2 * D(w, "y")
    ny, nx = w.shape
    ix = np.arange(mx + 4, nx - mx - 4, stride)
    iy = np.arange(my + 4, ny - my - 4, stride)
    IY, IX = np.meshgrid(iy, ix, indexing="ij")
    at = lambda F: F[IY, IX]
    C = {"S": at(K(Spt, fx, fy)),
         "J": at(K(J, fx, fy)),
         "w": at(K(w, fx, fy)),
         "Lap w": at(K(w, fx2, fy) + K(w, fx, fy2)),
         "div k": at(-(K(k1, fx1, fy) + K(k2, fx, fy1))),
         "(div k)^2": at(K(divk ** 2, fx, fy)),
         "w^2 div k": at(K(w ** 2 * divk, fx, fy)),
         "A_dev div k": at(K(Ad * divk, fx, fy)),
         "w div k": at(K(w * divk, fx, fy))}
    B = at(-(K(k1, fx3, fy) + K(k1, fx1, fy2) + K(k2, fx2, fy1) + K(k2, fx, fy3)))
    return C, B, IY, IX, dict(J=J, w=w, divk=divk)


def _r2(y, yh):
    return float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))


def fit_support(C, B, sel, xs, names):
    tr = xs < np.median(xs)
    Th = np.column_stack([C[n][sel] for n in names])
    b = B[sel]
    trs = tr[sel]
    c, *_ = np.linalg.lstsq(Th[trs][:, :], b[trs], rcond=None)
    return c, _r2(b[~trs], Th[~trs] @ c)


def best_subsets(C, B, sel, xs, names, maxk):
    out = {}
    for k in range(1, maxk + 1):
        best = None
        for comb in itertools.combinations(names, k):
            c, s = fit_support(C, B, sel, xs, list(comb))
            if best is None or s > best[0]:
                best = (s, list(comb), c)
        out[k] = best
    return out


def one_field(label, k1, k2, A, x, y, L, sel_fn, args, lines, rows, stamp):
    C, B, IY, IX, pt = columns(k1, k2, A, x, y, L, args.radius, args.poly_p, args.stride)
    sel = sel_fn(IY, IX)
    X, Y = np.meshgrid(x, y)
    xs = X[IY, IX]
    n = int(sel.sum())
    lines.append("")
    lines.append("%s: n %d   rms B %.3e  rms S %.3e  rms J %.3e" %
                 (label, n, np.sqrt(np.mean(B[sel] ** 2)), np.sqrt(np.mean(C["S"][sel] ** 2)),
                  np.sqrt(np.mean(C["J"][sel] ** 2))))
    if n < 20:
        lines.append("   too few centres; skipped")
        return
    for k, (s, nm, c) in best_subsets(C, B, sel, xs, LIB, args.max_size).items():
        lines.append("   size %d: R2 %6.3f  " % (k, s) + " + ".join("%+.2f %s" % (ci, ni) for ci, ni in zip(c, nm)))
        rows.append(dict(stamp=stamp, field=label, kind="free size %d" % k, R2=s, n=n,
                         support=" | ".join(nm), coefs=json.dumps([float(v) for v in c])))
    for want in (["S"], ["S", "J"]):
        c, s = fit_support(C, B, sel, xs, want)
        lines.append("   forced %s: R2 %6.3f  " % (want, s) + " + ".join("%+.2f %s" % (ci, ni) for ci, ni in zip(c, want)))
        rows.append(dict(stamp=stamp, field=label, kind="forced {%s}" % ", ".join(want), R2=s, n=n,
                         support=" | ".join(want), coefs=json.dumps([float(v) for v in c])))


# ---------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------

def run_data(args, files, lines, rows, stamp):
    for f in files:
        D = load_npz(f)
        u, x, y, tt, md = D["u"], D["x"], D["y"], D["t"], D["md"]
        mu = float(md.get("mu", args.mu))
        k1c = np.sqrt(1 - mu ** 2)
        L = np.pi / k1c
        dx = float(x[1] - x[0])
        Lx, Ly = len(x) * dx, len(y) * dx
        X, Y = np.meshgrid(x, y)
        y_row = float(y[int(md.get("gb_row_in_crop", len(y) // 2))])
        fr = args.frame if args.frame >= 0 else u.shape[-1] + args.frame
        e = estimate(u[..., fr], Lx, Ly, args.sigma_dir, None, t1c.edge_taper(x, y, frac=args.taper_frac))
        k1, k2, A = e["k1"], e["k2"], e["A"]
        sg = np.sign(k1.mean())
        k1, k2 = k1 * sg, k2 * sg
        win = (np.abs(X - x.mean()) < args.win_frac * Lx) & (np.abs(Y - y.mean()) < args.win_frac * Ly)
        xc, yc = find_cores(A, x, y, win, args.core_frac, merge_dist=0.5 * L)
        dc = dist_to_cores(X, Y, xc, yc) / L
        side = {"both": lambda YY: np.abs(YY - y_row) < args.band_to * L,
                "upper": lambda YY: (YY - y_row > 0) & (YY - y_row < args.band_to * L),
                "lower": lambda YY: (y_row - YY > 0) & (y_row - YY < args.band_to * L)}[args.side]

        def sel_fn(IY, IX):
            return (side(Y[IY, IX]) & (dc[IY, IX] > args.core_excl)
                    & (np.abs(X[IY, IX] - x.mean()) < args.win_frac * Lx)
                    & (np.abs(Y[IY, IX] - y.mean()) < args.win_frac * Ly))
        stem = Path(f).stem
        lines.append("")
        lines.append("file %s: mu %.3f (L %.2f), frame %d (t %.3f), %d cores, y_row %.2f, side %s, band < %g L, discs %g L"
                     % (stem, mu, L, fr, float(tt[fr]), len(xc), y_row, args.side, args.band_to, args.core_excl))
        one_field("%s frame %d, estimator k" % (stem, fr), k1, k2, A, x, y, L, sel_fn, args, lines, rows, stamp)
        d = np.load(f, allow_pickle=True)
        if "k1_exact" in d.files and "k2_exact" in d.files:
            k1e, k2e = d["k1_exact"], d["k2_exact"]
            if k1e.ndim == 3:
                k1e, k2e = k1e[..., fr], k2e[..., fr]
            Ae = d["A_exact"][..., fr] if ("A_exact" in d.files and d["A_exact"].ndim == 3) else \
                (d["A_exact"] if "A_exact" in d.files else np.ones_like(k1e))
            one_field("%s frame %d, exact k" % (stem, fr), k1e, k2e, Ae, x, y, L, sel_fn, args, lines, rows, stamp)


def run_pn(args, lines, rows, stamp):
    mu = args.mu
    k1c = np.sqrt(1 - mu ** 2)
    L = np.pi / k1c
    Ly_, ny = args.Ly, args.ny_control
    dx = Ly_ / (ny - 1)
    y = np.linspace(-Ly_ / 2, Ly_ / 2, ny)
    nx = int(round(args.n_xper * 2 * np.pi / k1c / dx))
    x = (np.arange(nx) - nx // 2) * dx
    X, Y = np.meshgrid(x, y)
    sp = L if args.pn_spacing <= 0 else args.pn_spacing
    nd = int(round(nx * dx / sp))
    xd = (np.arange(nd) - nd / 2 + 0.5) * sp
    th, _ = pismen_on_sh_grid(X, Y, xd)
    c_, sn = np.cos(th), np.sin(th)
    k1e = c_ * np.gradient(sn, dx, axis=1) - sn * np.gradient(c_, dx, axis=1)
    k2e = c_ * np.gradient(sn, dx, axis=0) - sn * np.gradient(c_, dx, axis=0)
    rr = np.min(np.hypot(X[..., None] - xd[None, None, :], Y[..., None]), axis=-1)
    Apn = 0.816 * np.tanh(rr / args.pn_core)
    u = Apn * np.cos(th)
    e = estimate(u, nx * dx, ny * dx, args.sigma_dir, None, t1c.edge_taper(x, y, frac=args.taper_frac))
    K1, K2 = e["k1"], e["k2"]
    sg = np.sign(K1.mean())
    K1, K2 = K1 * sg, K2 * sg
    dc = rr / L
    lines.append("")
    lines.append("Pismen row: mu %.3f (L %.2f), %d cores at spacing %.2f, box %.1f x %.1f, dx %.3f, smooth dip r_c %.2f, upper band < %g L, discs %g L"
                 % (mu, L, nd, sp, nx * dx, ny * dx, dx, args.pn_core, args.band_to, args.core_excl))

    def sel_fn(IY, IX):
        return ((Y[IY, IX] > 0) & (Y[IY, IX] < args.band_to * L) & (dc[IY, IX] > args.core_excl)
                & (np.abs(X[IY, IX]) < args.win_frac * nx * dx))
    one_field("PN row mu %.2f, exact k" % mu, k1e, k2e, Apn, x, y, L, sel_fn, args, lines, rows, stamp)
    one_field("PN row mu %.2f, estimator k" % mu, K1, K2, Apn, x, y, L, sel_fn, args, lines, rows, stamp)


def main(args):
    t0 = time.time()
    files = [s for s in str(args.npz).split(",") if s.strip()]
    if files:
        args.mode = "data"
    stem = Path(files[0]).stem if len(files) == 1 else ("multi_%d" % len(files) if files else "pn_mu%.2f" % args.mu)
    stamp = "rad%g_ce%g_band%g_%s_fr%d" % (args.radius, args.core_excl, args.band_to, args.side, args.frame)
    out = Path(args.outdir) / stem / stamp
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as fh:
        json.dump({k: getattr(args, k) for k in vars(args)}, fh, indent=2, default=str)
    lines = [VERSION, "mode %s   kernel radius %g L, poly p %d, stride %d, holdout x < median, max support %d"
             % (args.mode, args.radius, args.poly_p, args.stride, args.max_size),
             "target Int[Lap div k phi]; library " + ", ".join(LIB)]
    rows = []
    if args.mode == "pn":
        run_pn(args, lines, rows, stamp)
    else:
        run_data(args, files, lines, rows, stamp)
    lines.append("")
    lines.append("elapsed %.1f s" % (time.time() - t0))
    report = "\n".join(lines)
    (out / "report.txt").write_text(report)
    with open(out / "fits.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["stamp", "field", "kind", "R2", "n", "support", "coefs"])
        wr.writeheader()
        wr.writerows(rows)
    tab = ["%s   provenance: %s/%s" % (VERSION, stem, stamp),
           "%-44s %-16s %6s %7s  %s" % ("field", "kind", "R2", "n", "law")]
    for r in rows:
        cs = json.loads(r["coefs"])
        law = " + ".join("%+.2f %s" % (ci, ni) for ci, ni in zip(cs, r["support"].split(" | ")))
        tab.append("%-44s %-16s %6.3f %7d  %s" % (r["field"][:44], r["kind"], r["R2"], r["n"], law))
    (out / "table.txt").write_text("\n".join(tab))
    print(report)
    print("outputs ->", out)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    mode = "data"                # data (--npz) | pn (the Pismen row, both routes)
    npz = ""                     # one file or a comma-separated list; one frame each
    outdir = str(_HERE / "results" / "rcn_free_stationary")
    frame = -1                   # frame index (negative from the end)
    mu = 0.85                    # pn mode, or data files without metadata
    side = "both"                # both | upper | lower (data mode; pn mode is always upper)
    band_to = 2.0                # band half width in spacings
    core_excl = 0.5              # core discs removed, spacings
    win_frac = 0.4               # window |x - xm| < win_frac Lx, |y - ym| < win_frac Ly
    radius = 1.5                 # kernel half width, spacings
    poly_p = 6
    stride = 4
    max_size = 3                 # largest free support
    sigma_dir = 0.4
    taper_frac = 0.06
    core_frac = 0.70
    # pn mode
    Ly = 60.0
    ny_control = 300
    n_xper = 4
    pn_spacing = 0.0             # 0 = pi/k1 (the zigzag core spacing)
    pn_core = 1.5                # smooth dip radius r_c, x units


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in dir(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    cli = p.parse_args()
    sys.exit(main(cli))
