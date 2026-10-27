# pipelines/stationary_checks_1012.py
"""
Follow-up checks on workstream A (12 Oct 2026), answering Ed's questions
of 11 Oct, night. Imports self_dual_discovery.py v1.3.1 and the frozen
estimator chain; edits nothing.

  (1) errors   PN row through mg against the exact PN field: k2, w,
               div k along vertical slices (through a core, and midway
               between two cores), averaged along x over one dislocation
               spacing about the slice, and the error maps (the PN
               counterpart of fig_mg_ripple_knee.png).
  (2) profile  why a narrow bump across the row misreads the knee: the
               local weak ratio -div k / w against y at several half
               widths across the row (1.5 L along), exact and mg, and mg
               after symmetrising k about the row (k2 odd, k1 even).
  (3) pair     the "knee identity": on a tanh knee w^2 + k.grad w/(2 b)
               = -mu^2 w exactly, so div k on {w^2, k.grad w} is relation
               A rewritten. Forced fits of div k on {w} and on {w^2,
               k.grad w} with holdout R2, and the relative rms of the
               identity residual, on the knee and PN controls (exact and
               mg) and on SH last frames, on the band and on the plateau
               mask: is the pair a knee-only rewriting or does it hold on
               a defected control?
  (4) discs    relation A on the band and the collar with the core discs
               at 0.5, 0.35, 0.25 L (does the collar extend inward?).
  (5) cores    core positions on SH last frames: how many per dislocation
               spacing, and whether they sit in pairs (two dips per
               dislocation) or singly.

Outputs: results/stationary_checks_1012/<check>_<stamp>/ report.txt and
figures. Run: python stationary_checks_1012.py --check errors (or
profile, pair, discs, cores, all) [--npz file[,file...]]

Changelog
---------
v1.0  12 Oct 2026, first version.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import self_dual_discovery as SD                                       # noqa: E402

import matplotlib                                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402


def sd_args(**kw):
    a = SD._Args()
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def pn_fields(mu=0.85, nd=10, margin=1.5, Ly=60.0, ny=300, R=0.5, pn_core=1.5):
    k1 = float(np.sqrt(1 - mu ** 2)); L = np.pi / k1
    dy = Ly / (ny - 1); y = np.linspace(-Ly / 2, Ly / 2, ny)
    Lx = (nd + 2 * margin) * L; nx = int(round(Lx / dy)); x = (np.arange(nx) - nx // 2) * dy
    X, Y = np.meshgrid(x, y)
    xd = (np.arange(nd) - (nd - 1) / 2.0) * L
    th, _ = SD.pismen_on_sh_grid(X, Y, xd)
    c, s = np.cos(th), np.sin(th)
    k1e = c * np.gradient(s, dy, axis=1) - s * np.gradient(c, dy, axis=1)
    k2e = c * np.gradient(s, dy, axis=0) - s * np.gradient(c, dy, axis=0)
    rr = np.min(np.hypot(X[..., None] - xd[None, None, :], Y[..., None]), axis=-1)
    A = np.sqrt(4 * R / 3) * np.tanh(rr / pn_core)
    return dict(x=x, y=y, X=X, Y=Y, L=L, k1=k1e, k2=k2e, u=A * c, xd=xd, rr=rr, dy=dy, mu=mu)


def knee_fields(mu=0.85, b=1.3, Ly=60.0, ny=300, nper=6):
    k1 = float(np.sqrt(1 - mu ** 2)); L = np.pi / k1
    dy = Ly / (ny - 1); y = np.linspace(-Ly / 2, Ly / 2, ny)
    nx = int(round(nper * 2 * np.pi / k1 / dy)); x = (np.arange(nx) - nx // 2) * dy
    X, Y = np.meshgrid(x, y)
    th = k1 * X + np.log(np.cosh(b * mu * Y)) / b
    return dict(x=x, y=y, X=X, Y=Y, L=L, k1=np.full_like(th, k1), k2=mu * np.tanh(b * mu * Y), u=np.cos(th), dy=dy, mu=mu, b=b)


def mg(F):
    _, estimate, _, _, t1c = SD._estimator()
    x, y, dy = F["x"], F["y"], F["dy"]
    taper = t1c.edge_taper(x, y, frac=0.06)
    e = estimate(F["u"], len(x) * dy, len(y) * dy, 0.4, None, taper)
    K1, K2 = e["k1"], e["k2"]
    use_k1 = abs(np.mean(K1)) >= abs(np.mean(K2)); sg = np.sign(np.mean(K1 if use_k1 else K2)) or 1.0
    return K1 * sg, K2 * sg, e["A"]


# ---------------------------------------------------------------------
# (1) PN errors
# ---------------------------------------------------------------------

def check_errors(args, out, lines):
    F = pn_fields(mu=args.mu)
    K1, K2, Am = mg(F)
    x, y, L, dy = F["x"], F["y"], F["L"], F["dy"]
    w_e = F["k1"] ** 2 + F["k2"] ** 2 - 1; w_m = K1 ** 2 + K2 ** 2 - 1
    d_e = np.gradient(F["k1"], dy, axis=1) + np.gradient(F["k2"], dy, axis=0)
    d_m = np.gradient(K1, dy, axis=1) + np.gradient(K2, dy, axis=0)
    xd = F["xd"]; jcore = int(np.argmin(np.abs(x - xd[len(xd) // 2]))); jmid = int(np.argmin(np.abs(x - (xd[len(xd) // 2] + 0.5 * L))))
    half = int(round(0.5 * L / dy))
    up = F["Y"] > 0
    sl = (y > 0) & (y < 3 * L)
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for col, (fe, fm, nm, lim) in enumerate(((F["k2"], K2, "k2", None), (w_e, w_m, "w = |k|^2 - 1", (-0.3, 0.3)), (d_e, d_m, "div k", (-0.6, 0.6)))):
        for row, (j, lab) in enumerate(((jcore, "slice through a core"), (jmid, "slice midway between cores"))):
            ax = axs[row, col]
            ax.plot(y[sl] / L, fe[sl, j], "k-", lw=2, label="exact")
            ax.plot(y[sl] / L, fm[sl, j], "-", color="C3", lw=1, label="mg, this column")
            ax.plot(y[sl] / L, fm[sl, j - half:j + half + 1].mean(axis=1), "--", color="C0", lw=1.2, label="mg, averaged over one spacing in x")
            ax.plot(y[sl] / L, fe[sl, j - half:j + half + 1].mean(axis=1), ":", color="0.4", lw=1.2, label="exact, same average")
            if lim: ax.set_ylim(lim)
            ax.set_title("%s, %s" % (nm, lab), fontsize=9); ax.set_xlabel("y / L (distance above the row)"); ax.legend(fontsize=6)
        ax = axs[2, col]
        err = (fm - fe)
        v = float(np.nanpercentile(np.abs(err[sl]), 98))
        im = ax.imshow(err[sl], extent=[x[0] / L, x[-1] / L, y[sl][0] / L, y[sl][-1] / L], origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
        ax.plot(xd / L, np.zeros_like(xd), "k^", ms=4)
        ax.set_title("mg %s minus exact (error map; triangles: cores)" % nm, fontsize=9); ax.set_xlabel("x / L"); ax.set_ylabel("y / L")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("Pismen-Nepomnyashchy row, mu %.2f, 10 dislocations (upper half plane), through mg against exact" % args.mu, fontsize=10)
    fig.tight_layout(); fig.savefig(out / "fig_mg_errors_pn.png", dpi=140); plt.close(fig)
    for nm, fe, fm in (("k2", F["k2"], K2), ("w", w_e, w_m), ("div k", d_e, d_m)):
        for lo, hi in ((0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0)):
            m = up & (F["rr"] / L >= lo) & (F["rr"] / L < hi) & (np.abs(F["X"]) < 4 * L)
            lines.append("  %-6s core distance %.2f-%.2f L: rms error %.4f, rms exact %.4f, ratio %.2f" %
                         (nm, lo, hi, np.sqrt(np.mean((fm - fe)[m] ** 2)), np.sqrt(np.mean(fe[m] ** 2)),
                          np.sqrt(np.mean((fm - fe)[m] ** 2)) / max(np.sqrt(np.mean(fe[m] ** 2)), 1e-12)))
    return dict(K1=K1, K2=K2, F=F)


# ---------------------------------------------------------------------
# (2) ratio profile against the bump height
# ---------------------------------------------------------------------

def weak_pair(k1, k2, x, y, L, rx, ry):
    a = sd_args(radius=1.5, radius_x=rx, radius_y=ry)
    C, m = SD.weak_columns(k1, k2, x, y, L, a)
    return C["div k"], C["w"], m


def check_profile(args, out, lines):
    F = knee_fields(mu=args.mu, b=1.3)
    K1, K2, _ = mg(F)
    K1s = 0.5 * (K1 + K1[::-1]); K2s = 0.5 * (K2 - K2[::-1])            # symmetrised about the row (y -> -y)
    x, y, L = F["x"], F["y"], F["L"]
    jc = len(x) // 2
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (k1v, k2v, ttl) in zip(axs, ((F["k1"], F["k2"], "exact"), (K1, K2, "mg"), (K1s, K2s, "mg symmetrised about the row"))):
        for ry, col in zip((0.31, 0.62, 1.0, 1.5), ("C3", "C1", "C2", "C0")):
            Dk, Wk, m = weak_pair(k1v, k2v, x, y, L, 1.5, ry)
            prof = -Dk[:, jc - 3:jc + 4].mean(axis=1) / Wk[:, jc - 3:jc + 4].mean(axis=1)
            ok = (np.abs(y) < 1.5 * L) & (np.abs(Wk[:, jc]) > 0.05 * np.abs(Wk[:, jc]).max())
            ax.plot(y[ok] / L, prof[ok], "-", color=col, lw=1.2, label="half width across %.2f L" % ry)
            band = ok & (np.abs(y) < L)
            lines.append("  %-30s ry %.2f L: weak ratio over |y| < 1 L, mean %.3f, min %.3f, max %.3f (planted 1.3)" %
                         (ttl, ry, np.mean(prof[band]), np.min(prof[band]), np.max(prof[band])))
        ax.axhline(1.3, color="k", lw=0.8, ls=":"); ax.set_ylim(0, 2.5)
        ax.set_xlabel("(y - y_row) / L"); ax.set_ylabel("weak -div k / w at the bump centre"); ax.set_title(ttl, fontsize=9); ax.legend(fontsize=7)
    fig.suptitle("planted knee, beta 1.3, mu %.2f: the local weak ratio against the bump centre's height, 1.5 L along the row" % args.mu, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "fig_ratio_profile_vs_bump_height.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------------
# (3) the knee-identity pair on controls and data
# ---------------------------------------------------------------------

def forced(Th, b, xs):
    tr = xs < np.median(xs)
    c, *_ = np.linalg.lstsq(Th[tr], b[tr], rcond=None)
    pred = Th[~tr] @ c
    r2 = 1 - np.sum((b[~tr] - pred) ** 2) / np.sum((b[~tr] - b[~tr].mean()) ** 2)
    return c, float(r2)


def pair_block(label, k1, k2, x, y, L, y_row, d_core, side, mu, lines, radius=1.5, band_to=2.0, beta_ref=None):
    a = sd_args(radius=radius, band_to=band_to, mu=mu)
    P = SD.sample_frame(k1, k2, x, y, L, y_row, d_core, a, side, None)
    C, m = SD.weak_columns(k1, k2, x, y, L, a)
    # pointwise identity residual on the kept domain, w^2 + k.grad w/(2b) + mu^2 w, with b the fitted beta
    xs = P["_x"]
    cw, r2w = forced(P["w"][:, None], P["div k"], xs)
    beta = -float(cw[0]) if beta_ref is None else beta_ref
    cp, r2p = forced(np.column_stack([P["w^2"], P["k.grad w"]]), P["div k"], xs)
    ident = P["w^2"] + P["k.grad w"] / (2 * beta) + mu ** 2 * P["w"]
    rel = float(np.sqrt(np.mean(ident ** 2)) / np.sqrt(np.mean((mu ** 2 * P["w"]) ** 2)))
    # what the pair implies on a tanh knee of this beta: a = -1/... ; the identity gives div k = -b w = (b/mu^2)(w^2 + k.grad w/(2b))
    pred_a, pred_b = beta / mu ** 2, 0.5 / mu ** 2
    lines.append("  %-44s n %4d | {w}: beta %+.3f R2 %.3f | {w^2, k.grad w}: (%+.3f, %+.3f) R2 %.3f [knee identity predicts (%+.3f, %+.3f)] |"
                 " identity residual rms / rms(mu^2 w) = %.3f" % (label, len(xs), -cw[0], r2w, cp[0], cp[1], r2p, pred_a, pred_b, rel))
    return dict(beta=-float(cw[0]), r2_w=r2w, pair=cp.tolist(), r2_pair=r2p, identity_rel=rel)


def check_pair(args, out, lines):
    mu = args.mu
    lines.append("knee identity: on k2 = mu tanh(b mu y), w^2 + k.grad w/(2b) = -mu^2 w exactly, so div k = -b w = (b/mu^2) w^2 + (1/(2 mu^2)) k.grad w")
    F = knee_fields(mu=mu, b=1.3); K1, K2, _ = mg(F)
    pair_block("knee b 1.3, exact, band", F["k1"], F["k2"], F["x"], F["y"], F["L"], 0.0, None, "both", mu, lines)
    pair_block("knee b 1.3, mg, band", K1, K2, F["x"], F["y"], F["L"], 0.0, None, "both", mu, lines)
    P = pn_fields(mu=mu); K1, K2, _ = mg(P)
    for rad in (1.5, 0.25):
        pair_block("PN, exact, band, %.2f L" % rad, P["k1"], P["k2"], P["x"], P["y"], P["L"], 0.0, P["rr"] / P["L"], "upper", mu, lines, radius=rad)
        pair_block("PN, mg, band, %.2f L" % rad, K1, K2, P["x"], P["y"], P["L"], 0.0, P["rr"] / P["L"], "upper", mu, lines, radius=rad)
    load_npz, estimate, find_cores, dist_to_cores, t1c = SD._estimator()
    for f in args.npz_list:
        D = load_npz(f); md = D["md"]; m_ = float(md.get("mu", mu))
        u, x, y = D["u"][..., -1], D["x"], D["y"]
        k1f = np.sqrt(1 - m_ ** 2); L = np.pi / k1f; dx = float(x[1] - x[0]); dy = float(y[1] - y[0])
        taper = t1c.edge_taper(x, y, frac=0.06)
        e = estimate(u, len(x) * dx, len(y) * dy, 0.4, None, taper)
        K1, K2 = e["k1"], e["k2"]; use_k1 = abs(np.mean(K1)) >= abs(np.mean(K2)); sg = np.sign(np.mean(K1 if use_k1 else K2)) or 1.0
        K1, K2 = K1 * sg, K2 * sg
        X, Y = np.meshgrid(x, y)
        win = (np.abs(X - x.mean()) < 0.4 * len(x) * dx) & (np.abs(Y - y.mean()) < 0.4 * len(y) * dy)
        xc, yc = find_cores(e["A"], x, y, win, 0.70, merge_dist=0.5 * L)
        dco = dist_to_cores(X, Y, xc, yc) / L if len(xc) else None
        y_row = float(y[int(md.get("gb_row_in_crop", len(y) // 2))])
        pair_block("SH mu %.2f, band 6 L" % m_, K1, K2, x, y, L, y_row, dco, "both", m_, lines, band_to=3.0)
    return {}


# ---------------------------------------------------------------------
# (4) smaller core discs, (5) core positions
# ---------------------------------------------------------------------

def check_cores(args, out, lines):
    load_npz, estimate, find_cores, dist_to_cores, t1c = SD._estimator()
    for f in args.npz_list:
        D = load_npz(f); md = D["md"]; m_ = float(md.get("mu", 0.85))
        u, x, y = D["u"][..., -1], D["x"], D["y"]
        k1f = np.sqrt(1 - m_ ** 2); L = np.pi / k1f; dx = float(x[1] - x[0]); dy = float(y[1] - y[0])
        taper = t1c.edge_taper(x, y, frac=0.06)
        e = estimate(u, len(x) * dx, len(y) * dy, 0.4, None, taper)
        X, Y = np.meshgrid(x, y)
        win = (np.abs(X - x.mean()) < 0.4 * len(x) * dx) & (np.abs(Y - y.mean()) < 0.4 * len(y) * dy)
        xc, yc = find_cores(e["A"], x, y, win, 0.70, merge_dist=0.5 * L)
        y_row = float(y[int(md.get("gb_row_in_crop", len(y) // 2))])
        o = np.argsort(xc); xc, yc = xc[o], yc[o]
        gaps = np.diff(xc) / L
        width = 0.8 * len(x) * dx / L
        lines.append("  %s: mu %.2f, L %.2f, window %.1f L wide, %d cores (%.2f per spacing); x gaps (L): %s; y offsets from the row (L): %s" %
                     (Path(f).stem[:40], m_, L, width, len(xc), len(xc) / width,
                      " ".join("%.2f" % g for g in gaps), " ".join("%+.2f" % v for v in (yc - y_row) / L)))
        A = e["A"]
        fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
        sl = np.abs(y - y_row) < 2 * L
        ax[0].imshow(u[sl], extent=[x[0] / L, x[-1] / L, (y[sl][0] - y_row) / L, (y[sl][-1] - y_row) / L], origin="lower", cmap="gray", aspect="auto")
        ax[0].plot(xc / L, (yc - y_row) / L, "r+", ms=8); ax[0].set_title("pattern, mu %.2f, detected cores" % m_, fontsize=9)
        ax[1].imshow(A[sl], extent=[x[0] / L, x[-1] / L, (y[sl][0] - y_row) / L, (y[sl][-1] - y_row) / L], origin="lower", cmap="magma", aspect="auto")
        ax[1].plot(xc / L, (yc - y_row) / L, "c+", ms=8); ax[1].set_title("mg amplitude |z|", fontsize=9)
        for a_ in ax: a_.set_xlabel("x / L"); a_.set_ylabel("(y - y_row) / L")
        fig.tight_layout(); fig.savefig(out / ("cores_mu%.2f.png" % m_), dpi=130); plt.close(fig)
    return {}


def check_discs(args, out, lines):
    for f in args.npz_list:
        for ce in (0.5, 0.35, 0.25):
            a = sd_args(npz=f, band_to=3.0, core_excl=ce, outdir=str(out / ("discs_%.2f" % ce)), a2=0, decomp=0, beta_curve="")
            a.radii = "1.5"
            SD.main(a)
            rep = sorted((out / ("discs_%.2f" % ce)).glob("%s/*/report.txt" % Path(f).stem))[-1].read_text()
            lines.append("  file %s" % Path(f).stem[:44])
            for ln in rep.splitlines():
                if "(a) div k = -beta w" in ln or "RATIO MAP" in ln or "relation A on the centres inside" in ln or "mask location" in ln:
                    lines.append("  discs %.2f L | %s" % (ce, ln.strip()[:230]))
    return {}


def main(args):
    stamp = time.strftime("%m%d_%H%M")
    checks = ["errors", "profile", "pair", "cores", "discs"] if args.check == "all" else args.check.split(",")
    args.npz_list = [v for v in str(args.npz).split(",") if v.strip()]
    for ch in checks:
        out = (_HERE / args.outdir / ("%s_%s" % (ch, stamp))) if not Path(args.outdir).is_absolute() else Path(args.outdir) / ("%s_%s" % (ch, stamp))
        out.mkdir(parents=True, exist_ok=True)
        lines = ["stationary_checks_1012 v1.0, check %s (%s)" % (ch, time.strftime("%Y-%m-%d %H:%M"))]
        globals()["check_" + ch](args, out, lines)
        (out / "report.txt").write_text("\n".join(lines))
        print("\n".join(lines)); print("outputs ->", out)
    return 0


class _Args:
    check = "all"                  # errors | profile | pair | cores | discs | all (comma list allowed)
    npz = ""                       # SH last-frame files (comma-separated) for pair, cores, discs
    mu = 0.85                      # controls
    outdir = str(Path("results") / "stationary_checks_1012")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in dir(_Args) if not a.startswith("_")]:
        p.add_argument("--" + k, type=type(getattr(_Args, k)), default=getattr(_Args, k))
    sys.exit(main(p.parse_args()))
