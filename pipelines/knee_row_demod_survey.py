# pipelines/knee_row_demod_survey.py
"""
Exact single-mode demodulation of a straight phase grain boundary, row
by row, as a ground truth for knee frames, beside the frozen mg
estimator.

The idea. On the zigzag geometry the boundary row runs along x and the
x-wavenumber k1 is the same on every row (it is conserved along the
boundary). While the boundary has no cores, the pattern is invariant
along x up to the carrier, so every row is
    u(x, y) = Re[ c(y) exp(i k1 x) ] + harmonics (3 k1, ...),
and the complex coefficient c(y) is recovered EXACTLY by projecting each
row onto exp(i k1 x) (Hann weight along x, so a box that does not hold a
whole number of periods does not leak). No window acts across the row,
so nothing is averaged across the crease. From c(y):
    A(y) = |c(y)|              the amplitude,
    phi(y) = arg c(y)          the phase (unwrapped in y),
    k2(y) = phi'(y)            the across-row wavenumber, k = (k1, k2(y)).
For a Cross-Newell self-dual knee k2 = mu tanh(beta mu y) with A
constant; anything else (an amplitude structure, a non-monotone k2) is
read directly.

Two indices are printed per frame.
  r_A = -(A''/A) / k1^2 on the row. The uHu tensor of u = A cos(theta)
      is <u Hess u> = (1/2) A Hess A - (1/2) A^2 k k^T, so on the row
      (k2 = 0, A' = 0) its yy entry carries A A''/2 and its xx entry
      -A^2 k1^2/2: for r_A > 1 the amplitude curvature outweighs the
      phase and any uHu window reads k ACROSS the row (the flip seen in
      E0 on the mu 0.95 fine run, frame 30). r_A < 0 (an amplitude dip
      on the row) cannot flip it.
  beta_row = k2'(0) / mu^2, the knee slope on the row as the exact
      demodulation reads it, beside the same slope from mg's x-median k2.
Also: the extremes of k2 and A off the row (overshoot of k2 above mu and
the matching amplitude trough are the signature of the two far-field
modes beating, period pi/mu across the row), the x-invariance of the
row (fraction of the row variance in the k1 mode; near 1 on a knee
frame, lower once cores have formed, where the demodulation is an
x-average and no longer exact), and the number of amplitude cores.

Synthetic mode (--mode synthetic): planted fields with known c(y): the
self-dual knee (A = 1) at several beta, the knee with an amplitude dip,
and two beating modes a+ cos(k1 x + mu y) + a- cos(k1 x - mu y) with
tanh-switched amplitudes, on whole and on half-integer numbers of
periods, so the demodulation is checked against the analytic A and k2
before any SH frame is read.

Outputs: <outdir>/report.txt, survey.csv, profiles_<stem>.png (A(y) and
k2(y) from the demodulation and from mg, per surveyed frame), config.json

Changelog
---------
v1.0  first version (7 Oct 2026). Motivated by E0: on the mu 0.95 fine
      run, frame 30, the row demodulation (x box = 6.0 periods of k1)
      reads A 0.958 on the row with a trough of 0.689 at 1.65 = pi/(2 mu)
      and k2 rising from 0 to an overshoot of 1.144 there before settling
      to mu; r_A = 4.8 on the row, which explains why every uHu window
      read k across the row there; mg read k2 0.36 and 0.57 on the first
      two rows against 0.16 and 0.33 (near-row slope 1.70 against 0.74
      in beta units) and no overshoot. Synthetic checks (sandbox, frozen
      layer copies): A recovered to 1e-4 and k2 to 2e-4 on the beating
      modes, on 4 and on 4.5 periods in x; on self-dual knees k2 to
      6e-3 (beta 1) to 9e-2 (beta 4, a 1.2-row crease, where the
      3-point slope beta_row reads 2.8 for 4); planted r_A of an
      amplitude dip recovered within 5 percent. The mg amplitude-dip
      count ("mg dips") is not a core count on knee frames: it reports 6
      on frame 30, which is x-invariant to 0.995 on the row; read the
      x-invariance column for cores.
"""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_ROOT / "src", _ROOT / "src" / "op_extract", _ROOT / "src" / "utils", _HERE):
    sys.path.insert(0, str(_p))

from mg_estimator_validation import mg_estimator                      # noqa: E402
from mg_zigzag_theta_t import load_npz, find_cores                    # noqa: E402
from op_extract import t1_common as t1c                               # noqa: E402


# ---------------------------------------------------------------------
# the demodulation
# ---------------------------------------------------------------------

def fit_k1(u, x, k1_guess, rel=0.15, n=601):
    """the x-wavenumber that maximises the Hann-weighted projection power
    summed over rows (k1 is the same on every row)"""
    w = np.hanning(len(x))
    ks = np.linspace(k1_guess * (1 - rel), k1_guess * (1 + rel), n)
    P = [np.sum(np.abs((u * w[None, :]) @ np.exp(-1j * k * x)) ** 2) for k in ks]
    j = int(np.argmax(P))
    if 0 < j < n - 1:                                                   # parabolic refinement
        a, b, c = P[j - 1], P[j], P[j + 1]
        den = a - 2 * b + c
        if den != 0:
            return float(ks[j] + 0.5 * (a - c) / den * (ks[1] - ks[0]))
    return float(ks[j])


def row_demod(u, x, k1):
    """c(y) = 2 sum_x u w exp(-i k1 x) / sum_x w, w = Hann; and the
    fraction of each row's variance carried by Re[c exp(i k1 x)]"""
    w = np.hanning(len(x))
    E = np.exp(-1j * k1 * x)
    c = 2.0 * (u * w[None, :]) @ E / w.sum()
    rec = np.real(c[:, None] * np.conj(E)[None, :])
    var = np.sum(w[None, :] * u ** 2, axis=1)
    res = np.sum(w[None, :] * (u - rec) ** 2, axis=1)
    return c, 1.0 - res / np.maximum(var, 1e-300)


def profile_numbers(y, jrow, c, k1, mu, Afar_band):
    """A, k2 and the indices from c(y); y is the row coordinate (row at jrow)"""
    dy = float(y[1] - y[0])
    A = np.abs(c)
    ph = np.unwrap(np.angle(c))
    k2 = np.gradient(ph, dy)
    # orient so that k2 > 0 above the row
    up = (y - y[jrow] > 0) & (y - y[jrow] < 3.0)
    if np.median(k2[up]) < 0:
        k2 = -k2
    App = (A[jrow + 1] - 2 * A[jrow] + A[jrow - 1]) / dy ** 2
    r_A = -(App / A[jrow]) / k1 ** 2
    slope = (k2[jrow + 1] - k2[jrow - 1]) / (2 * dy)
    yrel = y - y[jrow]
    far = (np.abs(yrel) > Afar_band[0]) & (np.abs(yrel) < Afar_band[1])
    Afar = float(np.median(A[far])) if far.any() else float("nan")
    side = (yrel > 0) & (yrel < 6.0)
    j_k2max = np.where(side)[0][np.argmax(k2[side])]
    j_Amin = np.where(side)[0][np.argmin(A[side])]
    return dict(A=A, k2=k2, yrel=yrel, A_row=float(A[jrow]), A_far=Afar, r_A=float(r_A),
                beta_row=float(slope / mu ** 2), k2max=float(k2[j_k2max]), y_k2max=float(yrel[j_k2max]),
                Amin=float(A[j_Amin]), y_Amin=float(yrel[j_Amin]))


def mg_numbers(u, x, y, jrow, mu, args):
    """the frozen mg estimator on the same frame: x-median k2 and |k| per
    row, its slope on the row, and the amplitude cores"""
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    taper = t1c.edge_taper(x, y, frac=args.taper_frac)
    e = mg_estimator(u * taper, len(x) * dx, len(y) * dy, args.sigma_dir)
    s = np.sign(np.mean(e["k1"]))
    s = 1.0 if s == 0 else s
    k1m, k2m = s * e["k1"], s * e["k2"]
    nx = len(x)
    xs = slice(nx // 4, 3 * nx // 4)
    k2b = np.median(k2m[:, xs], axis=1)
    up = (y - y[jrow] > 0) & (y - y[jrow] < 3.0)
    if np.median(k2b[up]) < 0:
        k2b = -k2b
    kb = np.median(np.hypot(k1m, k2m)[:, xs], axis=1)
    slope = (k2b[jrow + 1] - k2b[jrow - 1]) / (2 * dy)
    A = np.abs(e["z"]) / np.maximum(taper, 1e-3)
    Lsp = np.pi / np.sqrt(1 - mu ** 2)
    win = np.zeros_like(u, bool)
    win[:, xs] = True
    win &= (np.abs(y - y[jrow]) < Lsp)[:, None]
    xc, _ = find_cores(A, x, y, win, args.core_frac, merge_dist=0.5 * Lsp)
    return dict(k2=k2b, kmag=kb, beta_row=float(slope / mu ** 2), n_cores=int(len(xc)))


# ---------------------------------------------------------------------
# synthetic checks
# ---------------------------------------------------------------------

def synth_fields(mu, dx, n_per, half_y):
    k1 = np.sqrt(1 - mu ** 2)
    nx = int(round(n_per * 2 * np.pi / k1 / dx))
    x = np.arange(nx) * dx
    y = np.arange(-half_y, half_y + 0.5 * dx, dx)
    X, Y = np.meshgrid(x, y)
    out = []
    for beta in (1.0, 2.0, 4.0):
        z = beta * mu * Y
        phi = np.logaddexp(z, -z) / beta
        out.append(("self-dual knee, beta %.0f, A = 1" % beta, np.cos(k1 * X + phi),
                    np.ones_like(y), mu * np.tanh(beta * mu * y), beta, 0.0))
    z = 2.0 * mu * Y
    phi = np.logaddexp(z, -z) / 2.0
    Ad = 1 - 0.3 / np.cosh(y) ** 2
    out.append(("knee beta 2 with an amplitude dip 0.3 sech^2(y)", Ad[:, None] * np.cos(k1 * X + phi),
                Ad, mu * np.tanh(2.0 * mu * y), 2.0, -0.6 / 0.7 / k1 ** 2))
    ap, am = 0.45 * (1 + np.tanh(y)), 0.45 * (1 - np.tanh(y))
    cy = ap * np.exp(1j * mu * y) + am * np.exp(-1j * mu * y)
    k2t = np.gradient(np.unwrap(np.angle(cy)), dx)
    out.append(("two beating modes, a+- = 0.45 (1 +- tanh y)",
                ap[:, None] * np.cos(k1 * X + mu * Y) + am[:, None] * np.cos(k1 * X - mu * Y),
                np.abs(cy), k2t, float("nan"), float("nan")))
    return x, y, k1, out


def run_synthetic(args, out, lines):
    rows = []
    lines.append("SYNTHETIC (planted c(y)): demodulated A and k2 against the analytic ones, max abs error over"
                 " |y| < 6; r_A and beta_row against their planted values; mg slope beside")
    for mu in args.syn_mus:
        for n_per in (4.0, 4.5):
            x, y, k1, fields = synth_fields(mu, args.dx, n_per, args.syn_half_y)
            jrow = int(np.argmin(np.abs(y)))
            lines.append("  mu %.2f, %.1f periods in x (k1 %.4f), dx %.4f" % (mu, n_per, k1, args.dx))
            for nm, u, At, k2t, beta, rA_t in fields:
                k1f = fit_k1(u, x, k1)
                c, xinv = row_demod(u, x, k1f)
                P = profile_numbers(y, jrow, c, k1f, mu, (8.0, 14.0))
                sel = np.abs(y) < 6.0
                eA = float(np.max(np.abs(P["A"][sel] - At[sel])))
                ek = float(np.max(np.abs(P["k2"][sel] - k2t[sel])))
                M = mg_numbers(u, x, y, jrow, mu, args)
                lines.append("    %-48s k1 fit %.5f | max|dA| %.1e max|dk2| %.1e | r_A %7.2f (planted %s) |"
                             " beta_row %6.3f (planted %s) | mg beta_row %6.3f | x-inv on row %.4f"
                             % (nm, k1f, eA, ek, P["r_A"], ("%.2f" % rA_t) if np.isfinite(rA_t) else "n/a",
                                P["beta_row"], ("%.2f" % beta) if np.isfinite(beta) else "n/a",
                                M["beta_row"], xinv[jrow]))
                rows.append(dict(kind="synthetic", file=nm, mu=mu, frame=-1, t=float("nan"), k1=k1f,
                                 n_cores=M["n_cores"], xinv_row=float(xinv[jrow]), A_row=P["A_row"],
                                 A_far=P["A_far"], r_A=P["r_A"], beta_row=P["beta_row"],
                                 beta_row_mg=M["beta_row"], k2max=P["k2max"], y_k2max=P["y_k2max"],
                                 Amin=P["Amin"], y_Amin=P["y_Amin"], maxdA=eA, maxdk2=ek))
    lines.append("  (planted r_A for the dip: -A''/A/k1^2 on the row = -(0.6)/(0.7)/k1^2; a dip has r_A < 0)")
    return rows


# ---------------------------------------------------------------------
# data
# ---------------------------------------------------------------------

def frame_list(nt, args):
    if args.frames:
        fr = [int(f) for f in args.frames.split(",")]
        return [f if f >= 0 else nt + f for f in fr]
    return sorted(set(int(round(fr * (nt - 1))) for fr in args.frame_fracs))


def run_data(args, out, lines):
    files = []
    if args.npz:
        files += [f.strip() for f in args.npz.split(",") if f.strip()]
    if args.data_dir:
        files += sorted(glob.glob(str(Path(args.data_dir) / args.pattern)))
    if not files:
        lines.append("no input files (pass --npz and/or --data_dir)")
        return []
    rows = []
    for path in files:
        D = load_npz(path)
        u, x, y, t, md = D["u"], D["x"], D["y"], D["t"], D["md"]
        mu = float(md.get("mu", args.mu))
        Lsp = np.pi / np.sqrt(1.0 - mu ** 2)
        jc = int(md.get("gb_row_in_crop", u.shape[0] // 2))
        sel = np.where(np.abs(y - y[jc]) <= args.crop_L * Lsp)[0]
        yc = y[sel]
        jrow = int(np.argmin(np.abs(yc - y[jc])))
        stem = Path(path).stem
        dx = float(x[1] - x[0])
        k1n = np.sqrt(1 - mu ** 2)
        lines.append("")
        lines.append("== %s   mu %.3f  L %.3f  grid dx %.4f dy %.4f  x box %.2f periods of the nominal k1  frames %d"
                     " (t %.3f to %.3f)" % (stem, mu, Lsp, dx, y[1] - y[0], len(x) * dx * k1n / (2 * np.pi),
                                           u.shape[2], t[0], t[-1]))
        lines.append("  %5s %7s | %5s %6s | %7s | %6s %6s %7s | %8s %8s | %13s %13s"
                     % ("frame", "t", "mgdip", "x-inv", "k1 fit", "A row", "A far", "r_A",
                        "beta_row", "mg", "k2 max at y", "A min at y"))
        prof = []
        for f in frame_list(u.shape[2], args):
            uf = np.asarray(u[sel, :, f], float)
            k1f = fit_k1(uf, x, k1n)
            c, xinv = row_demod(uf, x, k1f)
            P = profile_numbers(yc, jrow, c, k1f, mu, (0.6 * args.crop_L * Lsp, 0.95 * args.crop_L * Lsp))
            M = mg_numbers(uf, x, yc, jrow, mu, args)
            flag = "" if xinv[jrow] > args.xinv_min else "  (not x-invariant on the row: demodulation is an x-average)"
            lines.append("  %5d %7.3f | %5d %6.3f | %7.4f | %6.3f %6.3f %7.2f | %8.3f %8.3f | %6.3f %6.2f %6.3f %6.2f%s"
                         % (f, t[f], M["n_cores"], xinv[jrow], k1f, P["A_row"], P["A_far"], P["r_A"],
                            P["beta_row"], M["beta_row"], P["k2max"], P["y_k2max"], P["Amin"], P["y_Amin"], flag))
            rows.append(dict(kind="data", file=stem, mu=mu, frame=f, t=float(t[f]), k1=k1f, n_cores=M["n_cores"],
                             xinv_row=float(xinv[jrow]), A_row=P["A_row"], A_far=P["A_far"], r_A=P["r_A"],
                             beta_row=P["beta_row"], beta_row_mg=M["beta_row"], k2max=P["k2max"],
                             y_k2max=P["y_k2max"], Amin=P["Amin"], y_Amin=P["y_Amin"],
                             maxdA=float("nan"), maxdk2=float("nan")))
            prof.append((f, float(t[f]), P, M))
        fig_profiles(prof, mu, out / ("profiles_%s.png" % stem), stem)
        del D, u
    return rows


def fig_profiles(prof, mu, path, stem):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
    cm = plt.get_cmap("viridis")
    for i, (f, t, P, M) in enumerate(prof):
        col = cm(i / max(len(prof) - 1, 1))
        sel = (P["yrel"] > -0.5) & (P["yrel"] < 8.0)
        axs[0].plot(P["yrel"][sel], P["k2"][sel], "-", color=col, lw=1.2,
                    label="t %.2f (%d cores)" % (t, M["n_cores"]))
        axs[0].plot(P["yrel"][sel], M["k2"][sel], ":", color=col, lw=1.0)
        axs[1].plot(P["yrel"][sel], P["A"][sel], "-", color=col, lw=1.2)
    axs[0].axhline(mu, color="k", lw=0.6, ls="--")
    axs[0].set_xlabel("y - row"); axs[0].set_ylabel("k2  (solid: row demodulation; dotted: mg x-median)")
    axs[1].set_xlabel("y - row"); axs[1].set_ylabel("A = |c|  (row demodulation)")
    axs[1].axvline(np.pi / (2 * mu), color="k", lw=0.6, ls="--")
    axs[0].legend(fontsize=6)
    fig.suptitle("%s: across-row profiles (dashed: mu; pi/(2 mu))" % stem, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main(args):
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as fh:
        json.dump({k: getattr(args, k) for k in vars(args)}, fh, indent=2, default=str)
    lines = ["knee_row_demod_survey v1.0   mode %s   (mg: sigma_dir %.2g, taper %.2g; cores at %.2f of the median"
             " amplitude)" % (args.mode, args.sigma_dir, args.taper_frac, args.core_frac),
             "r_A = -(A''/A)/k1^2 on the row (above 1: amplitude curvature outweighs the phase in any uHu tensor);"
             " beta_row = k2'(0)/mu^2 from the demodulation, mg = the same slope from mg's x-median k2;"
             " x-inv = fraction of the row's variance in the k1 mode", ""]
    rows = run_synthetic(args, out, lines) if args.mode == "synthetic" else run_data(args, out, lines)
    if rows:
        with open(out / "survey.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader(); wr.writerows(rows)
    rep = "\n".join(lines)
    (out / "report.txt").write_text(rep)
    print(rep); print("outputs ->", out)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    mode = "data"                 # data | synthetic
    npz = ""                      # comma-separated SH npz files
    data_dir = ""                 # and/or a directory scanned with --pattern
    pattern = "sh_pgb_zigzag_mu*.npz"
    frames = ""                   # comma-separated frame indices (negative from the end); empty: frame_fracs
    frame_fracs = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
    mu = 0.85                     # fallback when the metadata lacks mu
    crop_L = 2.0                  # rows within crop_L spacings of the defect row
    xinv_min = 0.95               # flag rows whose variance is less than this fraction in the k1 mode
    sigma_dir = 0.4
    taper_frac = 0.08             # mg_zigzag_theta_t default
    core_frac = 0.70              # production amplitude threshold for cores
    dx = 0.2358                   # synthetic grid step (the SH grid)
    syn_mus = (0.85, 0.95)
    syn_half_y = 20.0
    outdir = str(_HERE / "results" / "knee_row_demod_survey" / "default")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in dir(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        if isinstance(v, tuple):
            p.add_argument("--" + k, type=float, nargs="+", default=list(v))
        else:
            p.add_argument("--" + k, type=type(v), default=v)
    sys.exit(main(p.parse_args()))
