"""pn_signal_size_1012.py  v1.0  (12 Oct 2026)

Answers one question of the research thread: how large are w and div k on the
SH grain boundary (through mg) compared with the exact PN row at the same
angle, binned by distance to the nearest core, and how large is the mg error
on the PN row in the same bins.  Nothing is fitted.

Output: results/<MMDD>_pn_signal_size/<stem>/signal_size.txt and .csv

Changelog
  v1.0  rms of w, div k by core-distance bin (0.25-0.5, 0.5-1, 1-2, 2-3 L) on
        (a) the SH last frame through mg, (b) the exact PN row, (c) the PN row
        through mg, with the mg error on PN in the same bins; SH / PN ratios.
"""
import argparse, sys, csv, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import self_dual_discovery as SD
import stationary_checks_1012 as SC


def rms(a, m):
    return float(np.sqrt(np.mean(a[m] ** 2))) if m.any() else float("nan")


def sh_fields(path, args):
    load_npz, estimate, find_cores, dist_to_cores, t1c = SD._estimator()
    D = load_npz(path)
    u, x, y, md = D["u"], D["x"], D["y"], D["md"]
    mu = float(md.get("mu", args.mu)); L = np.pi / np.sqrt(1 - mu ** 2)
    ny, nx, _ = u.shape
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    X, Y = np.meshgrid(x, y)
    win = (np.abs(X - x.mean()) < 0.4 * nx * dx) & (np.abs(Y - y.mean()) < 0.4 * ny * dy)
    e = estimate(u[..., -1], nx * dx, ny * dy, args.sigma_dir, None, t1c.edge_taper(x, y, frac=0.06))
    K1, K2 = e["k1"], e["k2"]
    sg = np.sign(np.mean(K1 if abs(np.mean(K1)) >= abs(np.mean(K2)) else K2)) or 1.0
    xc, yc = find_cores(e["A"], x, y, win, args.core_frac, merge_dist=0.5 * L)
    rr = dist_to_cores(X, Y, xc, yc)
    y_row = float(y[int(md.get("gb_row_in_crop", ny // 2))])
    return dict(k1=K1 * sg, k2=K2 * sg, dy=dy, L=L, rr=rr, X=X, Y=Y, mu=mu, ncores=len(xc), y_row=y_row, win=win)


def wdiv(k1, k2, dy):
    return k1 ** 2 + k2 ** 2 - 1, np.gradient(k1, dy, axis=1) + np.gradient(k2, dy, axis=0)


def main(args):
    stamp = time.strftime("%m%d")
    stem = Path(args.npz).stem
    out = Path(args.outdir) if args.outdir else HERE / "results" / ("%s_pn_signal_size" % stamp) / stem
    out.mkdir(parents=True, exist_ok=True)
    S = sh_fields(args.npz, args)
    mu = S["mu"]
    P = SC.pn_fields(mu=mu)
    K1p, K2p, _ = SC.mg(P)
    w_s, d_s = wdiv(S["k1"], S["k2"], S["dy"])
    w_pe, d_pe = wdiv(P["k1"], P["k2"], P["dy"])
    w_pm, d_pm = wdiv(K1p, K2p, P["dy"])
    bins = ((0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0))
    lines = ["pn_signal_size_1012 v1.0: %s, mu %.3f (L %.2f), SH cores %d" % (stem, mu, S["L"], S["ncores"]),
             "  rms by distance to the nearest core; SH through mg (last frame, |y - y_row| < 3 L, inner 80%% window); PN exact and PN through mg (upper half, |x| < 4 L)",
             "  %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s" % ("bin (L)", "SH w", "PN w ex", "PN w mg", "PNw err", "SH divk", "PN divk ex", "PN divk err")]
    rows = []
    band_s = (np.abs(S["Y"] - S["y_row"]) < 3 * S["L"]) & S["win"]
    up = (P["Y"] > 0) & (np.abs(P["X"]) < 4 * P["L"])
    for lo, hi in bins:
        ms = band_s & (S["rr"] / S["L"] >= lo) & (S["rr"] / S["L"] < hi)
        mp = up & (P["rr"] / P["L"] >= lo) & (P["rr"] / P["L"] < hi)
        r = dict(lo=lo, hi=hi, sh_w=rms(w_s, ms), pn_w_exact=rms(w_pe, mp), pn_w_mg=rms(w_pm, mp), pn_w_err=rms(w_pm - w_pe, mp),
                 sh_divk=rms(d_s, ms), pn_divk_exact=rms(d_pe, mp), pn_divk_mg=rms(d_pm, mp), pn_divk_err=rms(d_pm - d_pe, mp))
        rows.append(r)
        lines.append("  %4.2f-%4.2f  %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f" % (lo, hi, r["sh_w"], r["pn_w_exact"], r["pn_w_mg"], r["pn_w_err"], r["sh_divk"], r["pn_divk_exact"], r["pn_divk_err"]))
    lines.append("  ratios SH / PN exact:")
    for r in rows:
        lines.append("  %4.2f-%4.2f  w %6.1f x   div k %6.1f x   (PN mg error / SH signal: w %.2f, div k %.2f)" %
                     (r["lo"], r["hi"], r["sh_w"] / r["pn_w_exact"], r["sh_divk"] / r["pn_divk_exact"], r["pn_w_err"] / r["sh_w"], r["pn_divk_err"] / r["sh_divk"]))
    (out / "signal_size.txt").write_text("\n".join(lines) + "\n")
    with open(out / "signal_size.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print("\n".join(lines)); print("wrote", out)


class _Args:
    npz = ""
    mu = 0.85
    sigma_dir = 0.4
    core_frac = 0.70
    outdir = ""


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in vars(_Args) if not a.startswith("_")]:
        p.add_argument("--" + k, type=type(getattr(_Args, k)), default=getattr(_Args, k))
    a = p.parse_args()
    main(a)
