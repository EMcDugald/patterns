"""seam_audit.py  (pipelines/)

How much do the sign seams of the mg wavevector move the phase-law fit?

The mg estimator returns the director, so its wavevector is known only up
to sign.  The frame-to-frame alignment (mg_zigzag_theta_t.estimate) makes
the sign continuous in time at each pixel, and wsindy_theta_t applies one
scalar sign per frame (v1.6), but nothing makes the sign continuous in
SPACE: wherever the true k leaves the half-plane of the representative
the first frame chose, the extracted k is reversed, and the boundary of
that region is a seam across which the extracted k jumps to -k.  Near a
dislocation the true k turns through every direction, so seams run out of
the core region.  The weak fits remove query CENTRES inside the core
discs, not data: a test function centred just outside a disc integrates
the fields inside it, seam included (Chapter 6, the fitting paragraph).
This script measures that contribution on a fit window of record.

What it does, on the window and settings of a wsindy_theta_t data run
(read from its config.json, as rerun_from_config.py does):
  1. the estimator and the sign rules exactly as wsindy_theta_t_v19
     load_file_rows: estimate() with frame-to-frame alignment, the scalar
     sign per frame, find_cores, the core discs, the domain, the
     no-crossing option and query_centres;
  2. seam pixels: neighbouring pixels (in x or in y) whose unit extracted
     wavevectors are antiparallel, k.k' < seam_dot |k||k'| with
     seam_dot = 0, both |k| above kmin; a mask M is the seam pixels
     dilated by seam_dilate grid steps in x and y (frame by frame), so it
     covers every finite-difference stencil that straddles a seam;
  3. statistics: fraction of domain pixels in M; distance of the seam
     pixels to the nearest detected core in spacings (percentiles, and
     the fraction inside the core discs); fraction of the kept query
     centres whose test-function support contains a seam pixel;
  4. a placebo mask with the same area and the same distances from the
     cores: every pixel of M turned half a turn about its nearest core,
     which keeps its distance from the row and so its domain.
     Masking the seam neighbourhood also removes near-core signal, where
     J lives; the seam's own effect is what the seam mask changes beyond
     what the placebo changes;
  5. the columns of the forced phase law, whose integrands are pointwise
     in wsindy_theta_t (w div k, k.grad w, J, and the target theta_t),
     built twice: as wsindy_theta_t builds them, and with every pointwise
     integrand multiplied by (1 - M), and by (1 - placebo); the relative change of each column
     at the kept centres; and the forced fits {w div k, k.grad w, J} and
     the RCN triple {w div k, k.grad w, Lap div k} with holdout R^2 by
     the split wsindy_theta_t uses (forced_fit: x below the median of the
     centres trains).  For the triple, Lap div k is taken POINTWISE (three
     derivatives of k) in both the plain and the masked run so that the
     two are comparable; wsindy_theta_t's own Lap div k column is weak
     (derivatives on the test function) and cannot be masked.  The
     report prints, as a check, the plain columns against those of
     wsindy_theta_t_v19.build_columns: they must agree to rounding.
  6. seam_map.png: one frame (the middle one among those with the most cores), the
     seam pixels of the aligned estimate (what the fits see) and of the
     same frame estimated without alignment (the atan2 representative),
     over the amplitude, with the core discs and the domain edges.

Reading.  If masking moves the J coefficient d by a few percent and
leaves the comparison between J and Lap div k unchanged, the seams do
not carry the J law, and Chapter 5's sentence can say so with the number.
If it moves d by tens of percent, the attribution of J to SH needs the
masked fit beside it.

Usage
  python seam_audit.py --config results/<run of record>/config.json --tag pre85
  python seam_audit.py --config ... --seam_dilate 3 --tag dil3

Changelog
v1.0 (9 Oct 2026) first version.  Python 3.8.  Imports
      wsindy_theta_t_v19 and mg_zigzag_theta_t; edits nothing.  Validated
      in a sandbox with stand-ins for the frozen Riesz layer on a
      synthetic boundary with moving dislocations: the plain columns
      reproduce build_columns to machine precision, a seam planted by
      flipping the director in a half-disc is found to the pixel, and
      masking a region with no seam changes no column.  The placebo mask
      (the seam mask turned half a turn about each nearest core) was added
      after the first synthetic run: masking the seam neighbourhood also
      removes near-core signal, where J lives, so the seam's effect is the
      change beyond the placebo's, not the change itself.
v1.1 (9 Oct 2026) two fixes found on the first real runs; no number
      changes.  (1) The output folder names the window and the core
      discs, t<t0>-<t1>_ce<core_excl>_dil<d>_kmin<k>_<tag>: in v1.0 the
      0.85 pre-winding (t 0.2-3.6) and straddle (t 3-7) audits share a
      stem and the tag "rec", and the second overwrote the first.
      (2) seam_map.png draws the core discs of the run (core_excl); when
      the run removes none (core_excl 0, the 0.85 pre-winding window) it
      draws dashed 0.5 L circles labelled as a reference only.  v1.0
      drew solid 0.5 L discs in every case, which on the pre-winding map
      showed discs the fit never removed.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))

import wsindy_theta_t_v19 as ws                                         # noqa: E402
from mg_zigzag_theta_t import load_npz, estimate, find_cores, dist_to_cores  # noqa: E402
from op_extract import t1_common as t1c                                # noqa: E402

import matplotlib                                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.patches import Circle                                  # noqa: E402


def settings_from_config(path, overrides):
    """wsindy_theta_t_v19._Args, then the config of record, then overrides"""
    cfg = json.loads(Path(path).read_text())
    a = argparse.Namespace()
    known = [k for k in dir(ws._Args) if not k.startswith("_")]
    for k in known:
        setattr(a, k, getattr(ws._Args, k))
    for k, v in cfg.items():
        if k in known:
            setattr(a, k, v)
    for k, v in overrides.items():
        setattr(a, k, v)
    return a, cfg


def seam_pixels(k1, k2, kmin, seam_dot=0.0):
    """pixels with an x or y neighbour whose wavevector is antiparallel"""
    mag = np.sqrt(k1 ** 2 + k2 ** 2)
    ok = mag > kmin
    S = np.zeros(k1.shape, bool)
    for ax in (0, 1):
        sl0 = [slice(None)] * k1.ndim; sl1 = [slice(None)] * k1.ndim
        sl0[ax] = slice(0, -1); sl1[ax] = slice(1, None)
        sl0, sl1 = tuple(sl0), tuple(sl1)
        dot = k1[sl0] * k1[sl1] + k2[sl0] * k2[sl1]
        hit = (dot < seam_dot * mag[sl0] * mag[sl1]) & ok[sl0] & ok[sl1]
        S[sl0] |= hit; S[sl1] |= hit
    return S


def placebo_mask(S, F):
    """each pixel of the (dilated) seam mask moved to the pixel at the same distance from its
    nearest detected core, turned half a turn about that core: a mask
    with the seam's distance-to-core distribution but not on the seam,
    so that masking it removes as much near-core signal as masking the
    seam does"""
    x, y = F["x"], F["y"]; dx = float(x[1] - x[0]); dy = float(y[1] - y[0])
    Pm = np.zeros_like(S)
    ny, nx, nt = S.shape
    for i in range(nt):
        yy, xx = np.nonzero(S[..., i])
        xc, yc = F["cores_xy"][i]
        if len(yy) == 0 or len(xc) == 0:
            continue
        px, py = x[xx], y[yy]
        j = np.argmin(np.hypot(px[:, None] - xc[None, :], py[:, None] - yc[None, :]), axis=1)
        rx, ry = px - xc[j], py - yc[j]
        qx, qy = xc[j] - rx, yc[j] - ry           # half turn: keeps |y - y_core|, so the domain
        ix = np.round((qx - x[0]) / dx).astype(int); iy = np.round((qy - y[0]) / dy).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        Pm[iy[ok], ix[ok], i] = True
    return Pm


def fields_window(a, lines):
    """the estimator, sign rules, cores, domain and centres of
    wsindy_theta_t_v19.load_file_rows, keeping what this audit needs"""
    D = load_npz(a.npz)
    u, x, y, t, md = D["u"], D["x"], D["y"], D["t"], D["md"]
    ss = max(1, int(getattr(a, "space_stride", 1)))
    if ss > 1:
        md = dict(md); md["gb_row_in_crop"] = int(md.get("gb_row_in_crop", u.shape[0] // 2)) // ss
        u, x, y = u[::ss, ::ss], x[::ss], y[::ss]
    mu = float(md.get("mu", a.mu)); k1f = np.sqrt(1 - mu ** 2); Lsp = np.pi / k1f
    sel = (t >= a.t0) & (t <= a.t1)
    u, t = u[..., sel], t[sel]
    fs = max(1, int(getattr(a, "frame_stride", 1)))
    if fs > 1:
        u, t = u[..., ::fs], t[::fs]
    ny, nx, nt = u.shape
    dx = float(x[1] - x[0]); dt = float(np.median(np.diff(t)))
    Lx, Ly = nx * dx, ny * dx
    taper = t1c.edge_taper(x, y, frac=a.taper_frac)
    X, Y = np.meshgrid(x, y)
    win = (np.abs(X - x.mean()) < 0.4 * Lx) & (np.abs(Y - y.mean()) < 0.4 * Ly)
    jc0 = int(md.get("gb_row_in_crop", ny // 2))
    d_row = np.abs(Y - y[jc0]) / Lsp
    ests, n_prev = [], None
    for i in range(nt):
        e = estimate(u[..., i], Lx, Ly, a.sigma_dir, n_prev, taper)
        n_prev = e["n"]; ests.append(e)
    k1m = np.stack([e["k1"] for e in ests], -1); k2m = np.stack([e["k2"] for e in ests], -1)
    zs = np.stack([e["z"] for e in ests], -1)
    Am = np.stack([e["A"] for e in ests], -1)
    th_t = ws.theta_t_from_z(zs, dt)
    del zs
    s = np.sign(np.mean(k1m, axis=(0, 1)))[None, None, :]; s[s == 0] = 1
    k1m, k2m, th_t = k1m * s, k2m * s, th_t * s
    excl = np.zeros(u.shape, bool); ncores, cores_xy = [], []
    for i in range(nt):
        xc, yc = find_cores(Am[..., i], x, y, win, a.core_frac, merge_dist=0.5 * Lsp)
        ncores.append(len(xc)); cores_xy.append((xc, yc))
        if a.core_excl > 0 and len(xc):
            excl[..., i] = dist_to_cores(X, Y, xc, yc) / Lsp < a.core_excl
    cores = excl.copy()
    if a.domain == "far":
        dom = d_row > a.far_from
    elif a.domain == "band":
        dom = d_row < a.band_to
    elif a.domain == "row":
        dom = d_row < a.row_to
    elif a.domain == "shell":
        dom = (d_row >= a.band_to) & (d_row <= a.far_from)
    elif a.domain == "all":
        dom = np.ones_like(d_row, bool)
    else:
        raise SystemExit("domain %s is not supported by seam_audit (band, shell, row, far, all)" % a.domain)
    if a.side == "upper":
        dom &= Y > y[jc0]
    elif a.side == "lower":
        dom &= Y < y[jc0]
    excl |= ~dom[..., None]
    if int(getattr(a, "no_cross", 0)):
        my_nc = ws.kernel_sizes(a, dx, dx, dt, Lsp)[1]
        excl |= (np.abs(Y - y[jc0]) < my_nc * dx - 1e-9)[..., None]
    m = ws.kernel_sizes(a, dx, dx, dt, Lsp)
    idx = ws.query_centres(u.shape, m, a, excl=excl, cores=cores)
    lines.append("data: %s  mu %.2f  t in [%.3f, %.3f] (%d frames, dt %.4g)  crop %dx%d  domain %s"
                 "  cores per frame %d..%d  core discs %.2g L  centres %s"
                 % (Path(a.npz).stem, mu, t[0], t[-1], nt, dt, nx, ny, a.domain,
                    min(ncores), max(ncores), a.core_excl, "none" if idx is None else len(idx[0])))
    return dict(u=u, x=x, y=y, t=t, X=X, Y=Y, k1=k1m, k2=k2m, th_t=th_t, A=Am, mu=mu, Lsp=Lsp,
                dx=dx, dt=dt, jc0=jc0, dom=dom, excl=excl, cores=cores, cores_xy=cores_xy,
                ncores=ncores, idx=idx, m=m, taper=taper, Lx=Lx, Ly=Ly)


def pointwise(k1, k2, a, dx, dy, Lsp):
    """the inner fields of build_columns (same derivative operator)"""
    eta = a.eta * Lsp
    D = lambda F, ax: ws.inner_deriv(F, eta, dx, dy, ax, "poly", a.poly_p)
    w = k1 ** 2 + k2 ** 2 - 1.0
    divk = D(k1, "x") + D(k2, "y")
    wx, wy = D(w, "x"), D(w, "y")
    kgw = k1 * wx + k2 * wy
    Jk = D(k1, "x") * D(k2, "y") - D(k1, "y") * D(k2, "x")
    lapdivk = D(D(divk, "x"), "x") + D(D(divk, "y"), "y")
    return dict(wdivk=w * divk, kgw=kgw, J=Jk, lapdivk=lapdivk)


def weak_columns(P, th_t, keep, F, a):
    """K(f) at the centres for each pointwise integrand f, times keep"""
    x, y, t, Lsp, idx = F["x"], F["y"], F["t"], F["Lsp"], F["idx"]
    dx, dy, dt = float(x[1] - x[0]), float(y[1] - y[0]), F["dt"]
    rx, ry = ws.radii(a)
    fx = ws.kernel_d3(rx * Lsp, dx, a.poly_p)[0]
    fy = ws.kernel_d3(ry * Lsp, dy, a.poly_p)[0]
    ht = ws.kernel_d3(a.t_half * dt, dt, a.poly_p)[0]
    IY, IX, IT = idx
    K = lambda G: ws.corr3(G * keep, fx, fy, ht, dx, dy, dt)[IY, IX, IT]
    return {"w div k": K(P["wdivk"]), "k.grad w": K(P["kgw"]), "J": K(P["J"]),
            "Lap div k (pointwise)": K(P["lapdivk"]), "theta_t": K(th_t)}


def rel(a_, b_):
    return float(np.linalg.norm(a_ - b_) / max(np.linalg.norm(b_), 1e-300))


def fits(C, xs):
    out = {}
    for lab, want in (("phase law {w div k, k.grad w, J}", ["w div k", "k.grad w", "J"]),
                      ("RCN triple, pointwise Lap div k", ["w div k", "k.grad w", "Lap div k (pointwise)"])):
        names = ["w div k", "k.grad w", "J", "Lap div k (pointwise)"]
        Theta, b, x_ = ws.assemble(C, names, C["theta_t"], xs)
        out[lab] = ws.forced_fit(names, Theta, b, x_, want)
    return out


def map_frame(F):
    nc = np.asarray(F["ncores"]); cand = np.where(nc == nc.max())[0]
    return int(cand[len(cand) // 2])


def seam_map(F, S_al, S_raw, path, stem, ce=0.5):
    i = map_frame(F)
    x, y, Lsp, jc0 = F["x"], F["y"], F["Lsp"], F["jc0"]
    xm = 0.5 * (x[0] + x[-1])
    ext = [(x[0] - xm) / Lsp, (x[-1] - xm) / Lsp, (y[0] - y[jc0]) / Lsp, (y[-1] - y[jc0]) / Lsp]
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.4), sharey=True)
    for ax, S, ttl in ((axs[0], S_al[..., i], "aligned estimate (what the fits see)"),
                       (axs[1], S_raw, "same frame, no alignment (atan2 representative)")):
        ax.imshow(F["A"][..., i], origin="lower", extent=ext, cmap="Greys", aspect="equal")
        yy, xx = np.nonzero(S)
        ax.plot((x[xx] - xm) / Lsp, (y[yy] - y[jc0]) / Lsp, ".", color="#c0392b", ms=1.2,
                label="seam pixels")
        xc, yc = F["cores_xy"][i]
        for cx, cy in zip(xc, yc):
            ax.add_patch(Circle(((cx - xm) / Lsp, (cy - y[jc0]) / Lsp), ce if ce > 0 else 0.5,
                                fill=False, color="#2471a3", lw=1.0, ls="-" if ce > 0 else "--"))
        ax.axhline(0, color="0.3", lw=0.6, ls="--")
        ax.set_ylim(-3, 3)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("x (spacings)")
    axs[0].set_ylabel("y from the row (spacings)")
    axs[0].legend(loc="upper right", fontsize=8, markerscale=6)
    disc = ("core discs %.2g L (removed by the fit)" % ce if ce > 0 else
            "dashed: 0.5 L about each core, reference only (this window removes no core discs)")
    fig.suptitle("%s, t = %.3f: seams of the extracted wavevector over the amplitude; %s"
                 % (stem, F["t"][i], disc), fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return i


def main(c):
    t0 = time.time()
    a, cfg = settings_from_config(c.config, {})
    if not a.npz:
        print("the config has no npz: not a data run"); return 2
    stem = Path(a.npz).stem
    out = Path(c.outdir) / stem / ("t%.2f-%.2f_ce%.2g_dil%d_kmin%.2g_%s"
                                    % (float(a.t0), float(a.t1), float(a.core_excl),
                                       c.seam_dilate, c.kmin, c.tag or "run"))   # v1.1
    out.mkdir(parents=True, exist_ok=True)
    lines = ["seam_audit v1.1   config %s" % c.config,
             "  settings of record: radius %s (x %s, y %s), t_half %s, eta %s, domain %s, core_excl %s, no_cross %s"
             % (a.radius, a.radius_x, a.radius_y, a.t_half, a.eta, a.domain, a.core_excl,
                getattr(a, "no_cross", 0)),
             "  seam test: neighbours in x or y with k.k' < 0, both |k| > %.2g; mask dilated by %d grid steps"
             % (c.kmin, c.seam_dilate), ""]
    F = fields_window(a, lines)
    if F["idx"] is None:
        lines.append("no query centres; nothing to audit")
        (out / "report.txt").write_text("\n".join(lines)); print("\n".join(lines)); return 1
    k1, k2, Lsp, dx = F["k1"], F["k2"], F["Lsp"], F["dx"]
    S = seam_pixels(k1, k2, c.kmin)
    M = ws.box_dilate(S, c.seam_dilate, c.seam_dilate, 0) if c.seam_dilate > 0 else S.copy()
    if a.eta > 0 and c.seam_dilate * dx < a.eta * Lsp:
        lines.append("  FLAG: eta %.3g L mollifies derivatives over %.3g length units, wider than the"
                     " mask dilation %.3g; masked columns still see the seam" % (a.eta, a.eta * Lsp,
                                                                              c.seam_dilate * dx))
    dom3 = np.broadcast_to(F["dom"][..., None], S.shape)
    nt = S.shape[2]
    # 3. statistics
    frac_dom = float(M[dom3].mean())
    dist = []
    for i in range(nt):
        yy, xx = np.nonzero(S[..., i] & F["dom"])
        xc, yc = F["cores_xy"][i]
        if len(yy) and len(xc):
            d = dist_to_cores(F["x"][xx], F["y"][yy], xc, yc) / Lsp
            dist.append(np.asarray(d).ravel())
    dist = np.concatenate(dist) if dist else np.zeros(0)
    IY, IX, IT = F["idx"]
    mx, my, mt = F["m"]
    touched = ws.box_dilate(S, my, mx, mt)[IY, IX, IT]
    lines.append("seam pixels: %.4f of the domain pixels (%.4f after dilation), %d pixel-frames"
                 % (float(S[dom3].mean()), frac_dom, int(S[dom3].sum())))
    if len(dist):
        p = np.percentile(dist, [50, 90, 99, 100])
        lines.append("  distance of seam pixels to the nearest detected core (spacings): median %.2f,"
                     " 90%% %.2f, 99%% %.2f, max %.2f; inside the %.2g L core discs: %.3f"
                     % (p[0], p[1], p[2], p[3], a.core_excl, float((dist < a.core_excl).mean())))
    else:
        lines.append("  no seam pixel in the domain on a frame with detected cores")
    lines.append("  kept query centres whose test-function support contains a seam pixel: %.3f of %d"
                 % (float(touched.mean()), len(IY)))
    # 4. columns plain and masked
    P = pointwise(k1, k2, a, F["dx"], float(F["y"][1] - F["y"][0]), Lsp)
    plain = weak_columns(P, F["th_t"], 1.0, F, a)
    masked = weak_columns(P, F["th_t"], (~M).astype(float), F, a)
    Mp = placebo_mask(M, F)          # the dilated mask itself is turned, so the two masks have equal area
    placebo = weak_columns(P, F["th_t"], (~Mp).astype(float), F, a)
    lines.append("  placebo mask (the seam mask turned half a turn about each nearest core): %.4f of the"
                 " domain pixels after dilation, overlap with the seam mask %.3f"
                 % (float(Mp[dom3].mean()), float((Mp & M)[dom3].sum() / max(M[dom3].sum(), 1))))
    xs = F["x"][IX]
    # check against build_columns (same fields, same centres)
    cols, targets, _ = ws.build_columns(k1, k2, F["th_t"], F["x"], F["y"], F["t"], a, Lsp, F["idx"],
                                        A=None, Abar=None, snap=None)
    chk = [(nm, rel(plain[nm], cols["rhs"][nm])) for nm in ("w div k", "k.grad w", "J")]
    chk.append(("theta_t", rel(plain["theta_t"], targets["theta_t"])))
    lines.append("")
    lines.append("check, plain columns against wsindy_theta_t_v19.build_columns (relative difference): "
                 + ", ".join("%s %.1e" % c_ for c_ in chk))
    lines.append("relative change of each column at the kept centres when a mask is taken out of the"
                 " integrands (seam mask / placebo mask at the same distances from the cores):")
    for nm in ("w div k", "k.grad w", "J", "Lap div k (pointwise)", "theta_t"):
        lines.append("  %-24s %.4f / %.4f" % (nm, rel(masked[nm], plain[nm]), rel(placebo[nm], plain[nm])))
    fp, fm, fq = fits(plain, xs), fits(masked, xs), fits(placebo, xs)
    lines.append("")
    lines.append("forced fits, holdout R^2 by the split of wsindy_theta_t (x below the median trains):")
    for lab in fp:
        cp, cm = fp[lab]["coefs"], fm[lab]["coefs"]
        lines.append("  %s" % lab)
        lines.append("    plain  : " + ", ".join("%s %+.4g" % (k, v) for k, v in cp.items())
                     + "   R2 %.4f" % fp[lab]["r2_holdout"])
        lines.append("    masked : " + ", ".join("%s %+.4g" % (k, v) for k, v in cm.items())
                     + "   R2 %.4f" % fm[lab]["r2_holdout"])
        cq = fq[lab]["coefs"]
        lines.append("    placebo: " + ", ".join("%s %+.4g" % (k, v) for k, v in cq.items())
                     + "   R2 %.4f" % fq[lab]["r2_holdout"])
        lines.append("    change, seam / placebo: " + ", ".join(
            "%s %+.1f%% / %+.1f%%" % (k, 100 * (cm[k] - cp[k]) / max(abs(cp[k]), 1e-300),
                                       100 * (cq[k] - cp[k]) / max(abs(cp[k]), 1e-300)) for k in cp))
    adv_p = (fp["phase law {w div k, k.grad w, J}"]["r2_holdout"]
             - fp["RCN triple, pointwise Lap div k"]["r2_holdout"])
    adv_m = (fm["phase law {w div k, k.grad w, J}"]["r2_holdout"]
             - fm["RCN triple, pointwise Lap div k"]["r2_holdout"])
    adv_q = (fq["phase law {w div k, k.grad w, J}"]["r2_holdout"]
             - fq["RCN triple, pointwise Lap div k"]["r2_holdout"])
    lines.append("  holdout advantage of J over Lap div k (pointwise): plain %+.4f, seam masked %+.4f,"
                 " placebo masked %+.4f" % (adv_p, adv_m, adv_q))
    lines.append("  reading: the seam carries the J law only if the seam mask moves d and the advantage"
                 " well beyond what the placebo moves them")
    # 5. figure: aligned seams and the raw representative on the frame with most cores
    i = map_frame(F)
    e_raw = estimate(F["u"][..., i], F["Lx"], F["Ly"], a.sigma_dir, None, F["taper"])
    sraw = np.sign(np.mean(e_raw["k1"])) or 1.0
    S_raw = seam_pixels(e_raw["k1"] * sraw, e_raw["k2"] * sraw, c.kmin)
    seam_map(F, S, S_raw, out / "seam_map.png", stem, ce=float(a.core_excl))
    lines.append("")
    lines.append("seam_map.png: frame %d (t = %.3f, %d cores); seam pixels in the band there: aligned %d, unaligned %d"
                 % (i, F["t"][i], F["ncores"][i], int((S[..., i] & F["dom"]).sum()),
                    int((S_raw & F["dom"]).sum())))
    lines.append("elapsed %.1f s" % (time.time() - t0))
    rep = "\n".join(lines)
    (out / "report.txt").write_text(rep)
    with open(out / "config.json", "w") as fh:
        json.dump(dict(config=c.config, seam_dilate=c.seam_dilate, kmin=c.kmin, tag=c.tag,
                       record=cfg), fh, indent=2, default=str)
    print(rep); print("outputs ->", out)
    return 0


class _Args:
    config = ""                  # config.json of a wsindy_theta_t data run of record
    outdir = str(_HERE / "results" / "seam_audit")
    seam_dilate = 2              # grid steps added around each seam pixel (covers the np.gradient stencil)
    kmin = 0.3                   # both |k| above this for a pair to count (keeps the amplitude zeros out)
    tag = ""


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [k for k in dir(_Args) if not k.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    sys.exit(main(p.parse_args()))
