"""seam_control.py  (pipelines/)

Do sign seams on the knee core line make the J term of the phase law?

The question (critical todos C2 and C4 of TODO_TRIAGE_1008).  On the SH
runs the J law theta_t = a1 w div k + a2 k.grad w - d J (d about 1.5) is
carried by test functions whose support contains the knee core line, and
the same line carries the sign seams of the mg wavevector (seam_audit,
9 Oct: seam pixels on the row about once per spacing, outside the core
discs, touching 76 to 82 percent of the supports).  The RCN controls,
whose free selection never takes J, have a knee core line but no seam
(k1 > 0 everywhere).  This script gives the zero-modulation RCN control
seams like the data's and asks whether the J law appears.

Mechanism, for the reading.  A reversal of the extracted k over a patch
leaves w = |k|^2 - 1 and det grad k unchanged inside the patch (for a
2 x 2 matrix det(-M) = det M) and flips theta_t, div k, k.grad w and
Lap div k there.  At the patch edge grad k carries a jump term a (x) n
with a = -2k and n the edge normal; det(G + a (x) n) = det G +
cof(G) : (a (x) n), so J gains an edge term linear in the jump and
weighted by the smooth gradient G, which on the knee core line is the
large k2_y.  On a disc the term changes sign around the rim (n1 odd) and
largely cancels under a test function; on a half-disc the straight edge
has one normal and does not cancel.  The identity w_t = 2 k.grad theta_t
is unchanged inside a patch (both factors flip) and broken on its edge.

What it does.
  1. Loads a control field written by rcn2d_control.py (data-format npz
     with the exact fields; default the zero-modulation control of
     record, b1.8_sin2_ep0_ap0_ab0, t 0 to 2, 41 frames) and reproduces
     the estimator route of rcn2d_control v1.1 line for line: estimate()
     with the frame-to-frame alignment, theta_t from z, one scalar sign
     per frame, the band |y - y_l| < band_to L, query_centres, Abar from
     |y - y_l| > 4 L.  The fit settings are those of the control of
     record (abs_tol 0.02 etc.; _Args below).
  2. Plants reversals: for each case, a static patch mask P on the knee
     core line y = y_l + patch_y L, one patch per spacing (n_patch,
     default Lx / L = 8), and k1, k2, theta_t -> -k1, -k2, -theta_t inside
     P on every frame (after the scalar sign rule; the patches are a few
     tenths of a percent of the box and cannot change that rule).
     Shapes: disc (radius r L); half (the half of that disc on one side
     of a line through its centre at angle tilt from the x axis; with
     --alternate 1 the side alternates from patch to patch); dash (a
     segment along the line, half length r L, half thickness dash_h grid
     steps).  The case "none" is the control itself.
  3. Seam statistics with the detector of seam_audit (k.k' < 0 between
     x or y neighbours, both |k| > kmin): seam pixels as a fraction of
     the band pixels (the data: 0.0025 to 0.0062; 0.0139 to 0.0269
     after dilating by 2), the fraction of kept centres whose support
     contains a seam pixel (the data: 0.76 to 0.82), the distance of
     the seam pixels from the line.  A case whose seam fraction lies in
     the data's range is marked DATA-SIZED.
  4. The fits of record on each case: wsindy_theta_t_v19.build_columns
     and fit_blocks, the same blocks as a data run (free RHS, free RHS +
     ext with J, forced RCN triple, forced {w div k, k.grad w, J}, RCN +
     J, exhaustive supports of sizes 3 and 4, the identity).  The full
     block of each case goes to its own folder; report.txt carries the
     summary table.

Reading (fixed before the first run).  The SH J law has the form
(a1, a2, d) = (+, +, -) with d about -1.5 to -1.9 and is selected
freely.  The zero-modulation control through the estimator returns
(-0.57, -0.49, +3.21) on {w div k, k.grad w, J} and never selects J.
On a DATA-SIZED case:
  - "seams make the J law" if the forced fit takes the SH form (a1 > 0,
    a2 > 0, d < 0) with d at -0.5 or beyond and free selection on RHS +
    ext takes J with a negative coefficient;
  - "seams do not make the J law" if the forced fit keeps the control's
    form (d > 0 or the stretching pair negative) and free selection does
    not take J;
  - anything else is reported as it is.
The larger and smaller cases give the trend.

Check.  The case "none" must reproduce the control of record:
forced triple (1.392, 1.411, -1.355) at holdout R^2 0.9504, {w div k,
k.grad w, J} (-0.5677, -0.493, +3.208) at 0.7615, identity R^2 0.9999
(rcn2d_control report of 20 Sep, b1.8_sin2_ep0_ap0_ab0_rad1.5_th3_band_t2).
The report prints the comparison.

Usage
  python rcn2d_control.py --eps_b 0 --a_pos 0          # only if the npz is missing (about 3 min)
  python seam_control.py --npz results/rcn2d_control/b1.8_sin2_ep0_ap0_ab0_rad1.5_th3_band_t2/rcn2d_b1.8_sin2_ep0_ap0_ab0.npz

Outputs (relative): results/seam_control/<npz stem>/<stamp>/report.txt,
  summary.json, config.json, seam_cases.png, case_<name>/ (the full
  fit_blocks report of the case and its figures).

Changelog
v1.0 (9 Oct 2026) first version.  Python 3.8.  Imports wsindy_theta_t_v19,
      seam_audit (the seam detector) and mg_zigzag_theta_t; edits
      nothing.  Validated in a sandbox carrying verbatim copies of the
      functions of the frozen layer it calls (edge_taper, the monogenic
      and Riesz-director functions of field_diagnostics_viz): the case
      "none" reproduces the control of record to every printed digit
      (the check above), and a planted half-disc is found by the seam
      detector on its edge only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parent / "src", _HERE.parent / "src" / "op_extract", _HERE.parent / "src" / "utils", _HERE):
    sys.path.insert(0, str(_p))

import wsindy_theta_t_v19 as ws                                        # noqa: E402
import seam_audit as sa                                                # noqa: E402
from mg_zigzag_theta_t import load_npz, estimate                       # noqa: E402
from op_extract import t1_common as t1c                               # noqa: E402

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

VERSION = "seam_control v1.0"
# the control of record (rcn2d_control, 20 Sep, zero modulation, band, estimator route)
RECORD = {"triple": ((1.392, 1.411, -1.355), 0.9504), "SJ": ((-0.5677, -0.493, 3.208), 0.7615),
          "identity": 0.9999}
DATA_SEAM_FRAC = (0.0025, 0.0062)          # seam_audit 9 Oct, band pixels, undilated
FIT_KEYS = ("sigma_dir", "taper_frac", "core_frac", "radius", "t_half", "eta", "poly_p", "stride_xy",
            "stride_t", "rim", "rel_tol", "abs_tol", "min_contrib", "seed", "lib_ext", "band_to",
            "core_excl", "excl_dilate", "score_h", "chi_m", "exhaustive", "exh_sizes", "exh_top",
            "free_no_amp", "lib_cn")


def settings(c):
    """wsindy_theta_t_v19._Args, then this script's fit settings (those of the control of record)"""
    a = argparse.Namespace()
    for k in [k for k in dir(ws._Args) if not k.startswith("_")]:
        setattr(a, k, getattr(ws._Args, k))
    for k in FIT_KEYS:
        setattr(a, k, getattr(c, k))
    a.domain = "band"; a.no_cross = 0; a.radius_x = -1.0; a.radius_y = -1.0; a.mode = "data"
    return a


def control_fields(c, a, lines):
    """the estimator route of rcn2d_control v1.1 on its own npz"""
    D = load_npz(c.npz)
    u, x, y, t, md = D["u"], D["x"], D["y"], D["t"], D["md"]
    raw = np.load(c.npz, allow_pickle=True)
    sel = (t >= c.t0) & (t <= c.t1)
    u, t = u[..., sel], t[sel]
    mu = float(md.get("mu", 0.85)); k1c = np.sqrt(1 - mu ** 2); Lsp = np.pi / k1c
    ny, nx, nt = u.shape
    dx = float(x[1] - x[0]); dt = float(np.median(np.diff(t)))
    Lx, Ly = nx * dx, ny * dx
    y_l = float(md.get("y_l", y[int(md.get("gb_row_in_crop", ny // 4))]))
    X, Y = np.meshgrid(x, y)
    taper = t1c.edge_taper(x, y, frac=a.taper_frac)
    ests, n_prev = [], None
    for i in range(nt):
        e = estimate(u[..., i], Lx, Ly, a.sigma_dir, n_prev, taper)
        n_prev = e["n"]; ests.append(e)
    k1m = np.stack([e["k1"] for e in ests], -1); k2m = np.stack([e["k2"] for e in ests], -1)
    zs = np.stack([e["z"] for e in ests], -1); Am = np.stack([e["A"] for e in ests], -1)
    del ests
    th_t = ws.theta_t_from_z(zs, dt)
    del zs
    s = np.sign(np.mean(k1m, axis=(0, 1)))[None, None, :]; s[s == 0] = 1
    k1m, k2m, th_t = k1m * s, k2m * s, th_t * s
    dom = np.abs(Y - y_l) < a.band_to * Lsp
    excl = np.broadcast_to((~dom)[..., None], u.shape)
    m = ws.kernel_sizes(a, dx, dx, dt, Lsp)
    idx = ws.query_centres(u.shape, m, a, excl=excl)
    win = (np.abs(X - x.mean()) < 0.4 * Lx) & (np.abs(Y - y_l) < 0.2 * Ly)
    far = np.abs(Y - y_l) > 4 * Lsp
    Abar = float(np.nanmedian(Am[far]))
    exact = None
    if "k1_exact" in raw.files:
        exact = (raw["k1_exact"][..., sel].astype(float), raw["k2_exact"][..., sel].astype(float),
                 raw["theta_t_exact"][..., sel].astype(float), raw["A_exact"][..., sel].astype(float))
    lines.append("control: %s  mu %.2f (L %.3f)  t in [%.3f, %.3f] (%d frames, dt %.4g)  grid %dx%d (dx %.4f)"
                 "  knee core line y_l %.2f  band |y - y_l| < %g L  query centres %d"
                 % (Path(c.npz).name, mu, Lsp, t[0], t[-1], nt, dt, nx, ny, dx, y_l, a.band_to, len(idx[0])))
    return dict(x=x, y=y, t=t, X=X, Y=Y, u=u, k1=k1m, k2=k2m, th_t=th_t, A=Am, Abar=Abar, win=win, dom=dom,
                idx=idx, m=m, Lsp=Lsp, Lx=Lx, dx=dx, dt=dt, y_l=y_l, mu=mu, exact=exact)


def patch_mask(F, shape, r, c):
    """static patches on the knee core line, one per spacing"""
    x, X, Y, Lsp, Lx = F["x"], F["X"], F["Y"], F["Lsp"], F["Lx"]
    n = int(c.n_patch) if c.n_patch > 0 else int(round(Lx / Lsp))
    xc = x[0] + (np.arange(n) + 0.5) * Lx / n
    yc = F["y_l"] + c.patch_y * Lsp
    P = np.zeros(X.shape, bool)
    ang = np.radians(c.tilt)
    for j, x0 in enumerate(xc):
        dxp = (X - x0 + Lx / 2) % Lx - Lx / 2
        dy_ = Y - yc
        if shape == "dash":
            P |= (np.abs(dxp) <= r * Lsp) & (np.abs(dy_) <= c.dash_h * F["dx"] + 1e-9)
            continue
        disc = np.hypot(dxp, dy_) <= r * Lsp
        if shape == "disc":
            P |= disc
        else:                                                           # half
            side = -1.0 if (c.alternate and j % 2) else 1.0
            P |= disc & (side * (-np.sin(ang) * dxp + np.cos(ang) * dy_) > 0)
    return P, xc, yc


def seam_stats(k1, k2, F, c):
    S = sa.seam_pixels(k1, k2, c.kmin)
    dom3 = np.broadcast_to(F["dom"][..., None], S.shape)
    Md = ws.box_dilate(S, c.seam_dilate, c.seam_dilate, 0)
    IY, IX, IT = F["idx"]
    mx, my, mt = F["m"]
    touched = ws.box_dilate(S, my, mx, mt)[IY, IX, IT]
    yy = np.nonzero(S & dom3)[0]
    dl = np.abs(F["y"][yy] - F["y_l"]) / F["Lsp"]
    return dict(S=S, frac=float(S[dom3].mean()), frac_dil=float(Md[dom3].mean()),
                touched=float(touched.mean()),
                dist_med=float(np.median(dl)) if len(dl) else float("nan"),
                dist_90=float(np.percentile(dl, 90)) if len(dl) else float("nan"))


def run_case(name, k1, k2, th_t, A, Abar, F, a, c, out, rng):
    cdir = out / ("case_" + name)
    cdir.mkdir(parents=True, exist_ok=True)
    cl = ["%s | case %s" % (VERSION, name)]
    cols, targets, _ = ws.build_columns(k1, k2, th_t, F["x"], F["y"], F["t"], a, F["Lsp"], F["idx"],
                                        A=A, Abar=Abar, snap=None)
    R = dict(rhs=cols["rhs"], var=cols["var"], w=cols["w"], A=cols["A"], score=cols["score"], targets=targets,
             xs=F["x"][F["idx"][1]], Abar=Abar, A_median=float(np.nanmedian(A[F["win"]])), mu=F["mu"],
             n_rows=len(F["idx"][0]))
    fits = ws.fit_blocks(R, "seam control | %s" % name, a, rng, cl, cdir, route="")
    (cdir / "report.txt").write_text("\n".join(cl))
    return fits


def summarize(fits):
    g = lambda k: fits.get(k) or {}
    tri, sj, rj, fe = g("theta_rhs_forced"), g("theta_rhs_forced_SJ"), g("theta_rhs_forced_rcn_J"), g("theta_rhs_ext_free")
    ex3 = (g("theta_exhaustive").get(3) or {}) if isinstance(g("theta_exhaustive"), dict) else {}
    bj = ex3.get("best with J") or {}
    cf = sj.get("coefs", {})
    return dict(triple=[tri.get("coefs", {}).get(k) for k in ("w div k", "k.grad w", "Lap div k")],
                triple_r2=tri.get("r2_holdout"),
                SJ=[cf.get(k) for k in ("w div k", "k.grad w", "J")], SJ_r2=sj.get("r2_holdout"),
                rcnJ_J=rj.get("coefs", {}).get("J"), rcnJ_bend=rj.get("coefs", {}).get("Lap div k"),
                rcnJ_r2=rj.get("r2_holdout"),
                free_ext=fe.get("coefs", {}), free_ext_r2=fe.get("r2_holdout"),
                free_J=fe.get("coefs", {}).get("J"),
                ex3_best=ex3.get("best_r2"), ex3_J_rank=bj.get("rank"), ex3_J_r2=bj.get("r2"),
                identity=(g("wt_identity").get("r2"), g("wt_identity").get("rms_ratio")))


def verdict(s, data_sized):
    a1, a2, d = s["SJ"]
    sh_form = a1 is not None and a1 > 0 and a2 > 0 and d < 0
    free_neg = s["free_J"] is not None and s["free_J"] < 0
    if sh_form and d <= -0.5 and free_neg:
        v = "J law of the SH form, selected freely: seams of this size make it"
    elif (not sh_form) and s["free_J"] is None:
        v = "control form kept, J not selected: seams of this size do not make the J law"
    else:
        v = "mixed: SH form %s (d %+.3g), free J %s" % ("yes" if sh_form else "no", d,
                                                        "none" if s["free_J"] is None else "%+.3g" % s["free_J"])
    return ("DATA-SIZED: " if data_sized else "") + v


def fig_cases(F, cases, path):
    i = len(F["t"]) // 2
    x, y, Lsp, y_l = F["x"], F["y"], F["Lsp"], F["y_l"]
    sel = np.abs(y - y_l) < 1.0 * Lsp
    n = len(cases)
    fig, axs = plt.subplots(n, 1, figsize=(11, 1.25 * n + 0.6), sharex=True, squeeze=False)
    for ax, (name, P, S) in zip(axs[:, 0], cases):
        ext = [(x[0]) / Lsp, (x[-1]) / Lsp, (y[sel][0] - y_l) / Lsp, (y[sel][-1] - y_l) / Lsp]
        ax.imshow(np.where(P[sel], 1.0, np.nan), origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=1.5,
                  aspect="auto")
        yy, xx = np.nonzero(S[..., i][sel])
        ax.plot(x[xx] / Lsp, (y[sel][yy] - y_l) / Lsp, ".", color="#c0392b", ms=1.2)
        ax.axhline(0, color="0.4", lw=0.5, ls="--")
        ax.set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center")
        ax.tick_params(labelsize=7)
    axs[-1, 0].set_xlabel("x (spacings)")
    fig.suptitle("planted reversals (blue) and the seam pixels the detector finds (red), frame %d; y from the line"
                 " in spacings" % i, fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def main(c):
    t_start = time.time()
    rng = np.random.default_rng(c.seed)
    a = settings(c)
    stem = Path(c.npz).stem
    shapes = [s.strip() for s in c.shapes.split(",") if s.strip()]
    radii = [float(v) for v in c.radii.split(",") if v.strip()]
    stamp = "tilt%g_py%g_alt%d_kmin%.2g%s" % (c.tilt, c.patch_y, c.alternate, c.kmin, ("_" + c.tag) if c.tag else "")
    out = Path(c.outdir) / stem / stamp
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as fh:
        json.dump({k: getattr(c, k) for k in vars(c)}, fh, indent=2, default=str)
    lines = [VERSION + "   (wsindy_theta_t v1.9 fit blocks; seam detector of seam_audit v1.1)",
             "fit settings (the control of record): " + ", ".join("%s %s" % (k, getattr(a, k)) for k in FIT_KEYS[:14]),
             "patches: shapes %s, radii %s L, one per spacing on the line (offset %g L), tilt %g deg, alternate %d,"
             " dash half thickness %d grid steps; reversal of k1, k2, theta_t inside, every frame"
             % (",".join(shapes), ",".join("%g" % r for r in radii), c.patch_y, c.tilt, c.alternate, c.dash_h), ""]
    F = control_fields(c, a, lines)
    routes = [r.strip() for r in c.routes.split(",") if r.strip()]
    summary, figcases = {}, []
    case_list = [("none", None, None)] + [("%s_r%g" % (sh, r), sh, r) for sh in shapes for r in radii]
    for route in routes:
        if route == "exact" and F["exact"] is None:
            lines.append("route exact: the npz carries no exact fields; skipped"); continue
        if route == "exact":
            K1, K2, TT, AA = F["exact"]
            far = np.abs(F["Y"] - F["y_l"]) > 4 * F["Lsp"]
            Abar = float(np.nanmedian(AA[far]))
        else:
            K1, K2, TT, AA, Abar = F["k1"], F["k2"], F["th_t"], F["A"], F["Abar"]
        lines.append("")
        lines.append("[route %s]" % route)
        hdr = ("%-12s %8s %8s %7s %6s | %-24s %6s | %-26s %6s | %-8s %6s | %-15s %s"
               % ("case", "seams", "dil 2", "touch", "d_med", "forced triple", "R2", "forced {wdivk, kgradw, J}",
                  "R2", "free J", "R2", "identity", "verdict"))
        rows = []
        for name, sh, r in case_list:
            if sh is None:
                P = np.zeros(F["X"].shape, bool)
            else:
                P, xc, yc = patch_mask(F, sh, r, c)
            sgn = np.where(P, -1.0, 1.0)[..., None]
            k1, k2, th = K1 * sgn, K2 * sgn, TT * sgn
            st = seam_stats(k1, k2, F, c)
            fits = run_case("%s_%s" % (route, name), k1, k2, th, AA, Abar, F, a, c, out, rng)
            s = summarize(fits)
            s.update(seam_frac=st["frac"], seam_frac_dil=st["frac_dil"], touched=st["touched"],
                     dist_med=st["dist_med"], dist_90=st["dist_90"], patch_frac=float(P[F["dom"]].mean()))
            data_sized = DATA_SEAM_FRAC[0] <= st["frac"] <= DATA_SEAM_FRAC[1]
            s["verdict"] = verdict(s, data_sized) if sh is not None else "the control itself"
            summary["%s/%s" % (route, name)] = s
            if route == routes[0]:
                figcases.append((name, P, st["S"]))
            fmt3 = lambda v: "(%s)" % ", ".join("%+.3g" % q if q is not None else "-" for q in v)
            rows.append("%-12s %8.4f %8.4f %7.3f %6.2f | %-24s %6.3f | %-26s %6.3f | %-8s %6.3f | %-15s %s"
                        % (name, st["frac"], st["frac_dil"], st["touched"], st["dist_med"], fmt3(s["triple"]),
                           s["triple_r2"], fmt3(s["SJ"]), s["SJ_r2"],
                           ("%+.3g" % s["free_J"]) if s["free_J"] is not None else "not sel.",
                           s["free_ext_r2"] if s["free_ext_r2"] is not None else float("nan"),
                           "%.4f (%.3f)" % s["identity"], s["verdict"]))
            print(rows[-1]); sys.stdout.flush()
        lines.append(hdr)
        lines.extend(rows)
        if route == "estimator":
            s0 = summary["estimator/none"]
            lines.append("check, case none against the control of record (rcn2d_control 20 Sep): forced triple %s R2 %.4f"
                         " (record (1.392, 1.411, -1.355) 0.9504); {w div k, k.grad w, J} %s R2 %.4f (record"
                         " (-0.5677, -0.493, +3.208) 0.7615); identity %.4f (record 0.9999)"
                         % (", ".join("%+.4g" % v for v in s0["triple"]), s0["triple_r2"],
                            ", ".join("%+.4g" % v for v in s0["SJ"]), s0["SJ_r2"], s0["identity"][0]))
        lines.append("columns: seams = seam pixels / band pixels (data 0.0025 to 0.0062); dil 2 = after dilation by 2"
                     " (data 0.0139 to 0.0269); touch = fraction of kept centres whose support holds a seam pixel"
                     " (data 0.76 to 0.82); d_med = median distance of the seam pixels from the line, spacings; free J"
                     " = coefficient of J in the free RHS+ext selection; identity = R2 (RMS ratio) of"
                     " w_t = 2 k.grad theta_t")
        for name, sh, r in case_list:
            s = summary["%s/%s" % (route, name)]
            lines.append("  %-12s free RHS+ext: %s   exhaustive size 3 best %.4f, best with J #%s %.4f;"
                         " RCN + J: J %+.3g, bending %+.3g (R2 %.4f)"
                         % (name, ws.cstr(s["free_ext"]) if s["free_ext"] else "-",
                            s["ex3_best"] if s["ex3_best"] is not None else float("nan"),
                            s["ex3_J_rank"], s["ex3_J_r2"] if s["ex3_J_r2"] is not None else float("nan"),
                            s["rcnJ_J"] if s["rcnJ_J"] is not None else float("nan"),
                            s["rcnJ_bend"] if s["rcnJ_bend"] is not None else float("nan"),
                            s["rcnJ_r2"] if s["rcnJ_r2"] is not None else float("nan")))
    fig_cases(F, figcases, out / "seam_cases.png")
    lines.append("")
    lines.append("reading (fixed before the first run): on a DATA-SIZED case, the SH form (a1 > 0, a2 > 0, d <= -0.5)"
                 " with J selected freely at a negative coefficient means seams of that size make the J law; the"
                 " control's form kept with J not selected means they do not")
    lines.append("seam_cases.png: the patches and the detected seam pixels on the middle frame, route %s" % routes[0])
    lines.append("elapsed %.1f s" % (time.time() - t_start))
    rep = "\n".join(lines)
    (out / "report.txt").write_text(rep)
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print(rep); print("outputs ->", out)
    return 0


class _Args:
    npz = ""                         # control npz written by rcn2d_control.py (zero modulation of record)
    outdir = str(_HERE / "results" / "seam_control")
    routes = "estimator"             # estimator | estimator,exact (exact: seams planted on the exact fields)
    shapes = "disc,half,dash"
    radii = "0.03,0.06,0.1,0.2"      # spacings (dash: half length)
    tilt = 45.0                      # degrees, the cut of the half-disc
    alternate = 0                    # 1: half-disc side alternates from patch to patch
    patch_y = 0.0                    # patch centres off the line, spacings
    n_patch = 0                      # patches along the line (0: one per spacing)
    dash_h = 1                       # dash half thickness, grid steps
    kmin = 0.3                       # seam detector (seam_audit)
    seam_dilate = 2
    t0 = -1.0
    t1 = 1e9
    tag = ""
    # fit settings of the control of record (rcn2d_control 20 Sep); v1.9 additions at their defaults
    sigma_dir = 0.4
    taper_frac = 0.06
    core_frac = 0.70
    radius = 1.5
    t_half = 3
    eta = 0.0
    poly_p = 6
    stride_xy = 4
    stride_t = 1
    rim = 4
    rel_tol = 0.01
    abs_tol = 0.02
    min_contrib = 0.05
    seed = 0
    lib_ext = 1
    band_to = 2.0
    core_excl = 0.0
    excl_dilate = 0
    score_h = 6
    chi_m = 1.0
    exhaustive = 1
    exh_sizes = "3,4"
    exh_top = 8
    free_no_amp = 1
    lib_cn = 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [k for k in dir(_Args) if not k.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    sys.exit(main(p.parse_args()))
