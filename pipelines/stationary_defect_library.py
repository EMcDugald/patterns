"""stationary_defect_library.py  v1.0  (13 Oct 2026)   workstream G25 of the research thread

The stationary library for defected fields, from Newell, Passot, Bowman,
Ercolani and Indik (defects_are_weak, eqs. 1.16 to 1.22). Self-duality
grad^2 theta = s G holds where J = 0; near point defects grad^2 theta =
s (G + chi) (1.18), chi sourced by J through a linear equation (1.19) and
decaying like 1/r. The sign s is local, s = sgn(div k . w) in the paper's
convention (here sigma = sgn(-div k . w), +1 on every SH knee of record,
so that div k = -sigma beta w on the self-dual part).

In the thread's variables (RCN with G = w at leading order, eq. 1.20 of the
paper) the ledger identity of v1.5 is (1.19):
    chi_b = div k + sigma b w,      Lap chi_b - 2 b div(sigma chi_b k) = -4 b sigma J,
exact for every smooth gradient field with the stationary RCN equation
(relation B, Lap div k = 2 b^2 S) as its residual. So a chi solved from
(1.19) with the measured k, J, sigma equals the measured deviation from
self-duality exactly when the field satisfies the stationary RCN equation
with that b; their difference is relation B's residual, propagated through
the (1.19) operator.

Stages (per field, both routes on the synthetic modes):
  2  sign-blind equipartition (div k)^2 = beta^2 G^2 with G^2 = w^2 (the
     leading order) and G^2 = w^2 - w^4 (Swift-Hohenberg, the draft's eq. 10;
     theory beta = sqrt(alpha/eta) = 0.866), weak at --radius L, by bins of
     core distance (wound cores) or row distance (knees); the collar value
     is the self-dual beta where J = 0; the pointwise median of |div k|/|G|
     by the same bins beside it. Row-averaged version on data (x-averages
     under a Hann window over integer periods, 1-D weak in y).
  3  chi measured: chi_m = sgn(w)(|div k|/beta_c - |G|) with beta_c the
     collar value (sign-blind in the sector sign), ring-averaged about the
     wound cores from 0.25 to 2.5 L with the kept-domain mask; power-law
     exponent of the sign-blind ring mean over the rings where it is
     positive (the paper: 1/r).
  4  chi solved from (1.19) with beta_c on the untapered interior (zero
     Dirichlet at the box edges; chi decays like 1/r so the far boundary is
     flagged), sparse 5-point Laplacian plus central advection
     (grid Peclet 2 b |k| dx below 1); fits of weak div k on {-sigma w} and
     on {-sigma w, chi_solved}: the second coefficient must read 1 and the
     fit must hold on the near-core rings where the first fails; ring
     comparison of chi_solved against the measured chi_b = div k + sigma
     beta_c w (ratio, correlation); the relation-B residual
     Lap div k - 2 beta_c^2 S on the same rings as the floor of the test.
  1  (no computation) the reread of the 11 Oct free fits is in the
     explanation file.

Controls: --mode knee (planted self-dual knee: chi = 0, J = 0, every stage
must read zero correction and beta planted), --mode pn (the exact
Pismen-Nepomnyashchy row: J at the cores, chi and its decay computable,
relation B's residual is the CN-weak approximation's floor), both routes.

Outputs under --outdir: per field <stem>_<route>/report.txt, equi_bins.csv,
chi_rings.csv, fig_equi_*.png, fig_chi_*.png, chi maps; set table
library_table.csv and fig_library_vs_mu.png.

Changelog
  v1.0  first version. Sandbox validation (13 Oct): planted self-dual knee,
        exact route: equipartition beta 1.291 (planted 1.3, the weak
        squared form at 1.5 L), stage 4 beta 1.300 with c_chi 0.000 and
        relation B residual 0.00 (lambda 3.33 = 2 b^2); through mg:
        sign-blind collar beta 1.15 (the signed weak reading of record is
        1.00), chi_b rms 0.51 on the line from the ripple alone, pointwise
        |div k|/|w| medians 0.3 to 0.9 and the PN-sign share about 0.5 in
        every row bin (the pointwise sign and ratio are the ripple's; only
        weak readings survive through mg). Target (--mode target, the
        paper's section 2.6; BVP k ~ 0.824 r at the core, the paper's 0.8 r),
        exact route with --beta_fixed 1: chi_solved equals the measured
        chi_b ring by ring to three digits (corr 1.00), c_chi 1.000 and
        beta 1.002 in every ring at R2 1.000, relation B residual 0.00
        (lambda 1.99 = 2 b^2), decay exponent of chi -0.93 (the paper: 1/r);
        the target is not self-dual anywhere, so its collar beta is
        meaningless and b must be given. Exact PN row at 0.25 L: pointwise
        collar beta 1.27 (the record's 1.27), chi concentrated inside 0.75 L
        of the cores (sign-blind ring means +0.10, +0.016, 0.000 at 0.25 to
        1 L), relation B residual 0.6 to 1.2 with lambda of either sign:
        PN is not a stationary RCN solution and (1.19) does not reproduce
        its chi (c_chi -0.5 to +0.1), the expected negative control.
        Data (six last frames of record and the 28 of the sweep, mg, 1.5 L,
        discs 0.25 L about wound cores): sign-blind collar beta on w^2 1.44,
        1.49, 1.47, 1.55, 1.72, 2.64 (0.79 to 0.95; R2 0.96 to 0.99),
        0.44 to 0.67 of the signed weak beta of the same centres (w changes
        sign near the cores and the signed column partly cancels); the
        row-averaged version agrees to 2 percent; the sweep reads 1.53 to
        1.74 on the knees 0.65 to 0.71, a dip to 1.30 at 0.76 to 0.78, then
        1.33 to 1.64. With the matched-null J as the source, chi_solved is
        thirty times smaller than the measured chi_b near the cores (rms
        0.05 against 1.4 at 0.25 to 0.5 L), uncorrelated with it (0.1 to
        0.3), with c_chi scattered about zero and no R2 gain; the measured
        chi_b near the cores is at the level the mg ripple gives on a knee
        with no core (0.5 to 1.4 against 0.5). Relation B residual 2.5 to
        3.6 with lambda -2 on every defected file. So through mg, (1.19)
        with RCN's operator does not describe the near-core deviation from
        self-duality, as relation B's inverted sign already said.
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT / "src", ROOT / "src" / "op_extract", ROOT / "src" / "utils", HERE):
    sys.path.insert(0, str(_p))
import self_dual_discovery as SD                                         # noqa: E402
from self_dual_discovery import kernel_d3, corr1d, inner_deriv, domain_keep, centres, r2   # noqa: E402
from knee_rowavg_laws import hann_window                                 # noqa: E402
import matplotlib                                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BETA_THEORY = np.sqrt(3.0) / 2.0                                         # sqrt(alpha/eta) for SH, the draft's eq. 10
CORE_BINS = [(0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]
ROW_BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 2.0)]
RINGS = np.arange(0.25, 2.51, 0.25)


# ---------------------------------------------------------------------
# the target: the paper's axisymmetric stationary solution (section 2.6), the positive control for stage 4
# ---------------------------------------------------------------------

def target_frames(args):
    """concentric rolls solving the stationary RCN equation Lap div k = 2 b^2 div(w k) exactly:
    k_rr + k_r / r - k / r^2 = 2 b^2 (k^2 - 1) k, k(0) = 0, k -> 1 - 1/(4 b^2 r^2); J = k k_r / r confined to the
    core (the paper: k ~ 0.8 r near the core, support of J about a third of a wavelength). Here L = 2 pi, one
    wavelength, is the distance unit; the core is the single 'wound core' at the origin"""
    from scipy.integrate import solve_bvp
    b = float(args.beta)
    Rm = 1.2 * args.target_half * 2 * np.pi
    rr = np.linspace(1e-3, Rm, 4000)

    def f(r, Y):
        k, kp = Y
        return np.vstack([kp, 2 * b * b * (k * k - 1) * k - kp / r + k / r ** 2])

    def bc(Ya, Yb):
        return np.array([Ya[0], Yb[0] - (1 - 1 / (4 * b * b * Rm ** 2))])
    sol = solve_bvp(f, bc, rr, np.vstack([np.tanh(0.8 * rr), 0.8 / np.cosh(0.8 * rr) ** 2]), tol=1e-8, max_nodes=400000)
    k_of_r = lambda r: sol.sol(np.clip(r, rr[0], rr[-1]))[0]
    th_of_r = np.concatenate([[0.0], np.cumsum(0.5 * (sol.sol(rr)[0][1:] + sol.sol(rr)[0][:-1]) * np.diff(rr))])
    dx = args.target_dx
    half = args.target_half * 2 * np.pi
    n = int(round(2 * half / dx)); x = (np.arange(n) - n // 2) * dx; y = x.copy()
    X, Y = np.meshgrid(x, y); r = np.hypot(X, Y)
    kr = k_of_r(r)
    k1e, k2e = kr * X / np.maximum(r, 1e-9), kr * Y / np.maximum(r, 1e-9)
    th = np.interp(r, rr, th_of_r)
    u = np.cos(th)
    L = 2 * np.pi
    d_core = r / L
    stem = "target_b%.2f" % b
    out = [("exact", [dict(k1=k1e, k2=k2e, x=x, y=y, L=L, y_row=0.0, d_core=d_core, side="both")])]
    if args.route in ("estimator", "both"):
        _, estimate, _, _, t1c = SD._estimator()
        taper = t1c.edge_taper(x, y, frac=args.taper_frac)
        e = estimate(u, n * dx, n * dx, args.sigma_dir, None, taper)
        # orient the estimator's director radially (the target's k points outward)
        sg = np.sign(e["k1"] * X + e["k2"] * Y); sg[sg == 0] = 1.0
        out.append(("estimator", [dict(k1=e["k1"] * sg, k2=e["k2"] * sg, x=x, y=y, L=L, y_row=0.0, d_core=d_core, side="both")]))
    return stem, out, dict(k_slope=float(sol.sol(np.array([0.05]))[0][0] / 0.05), status=int(sol.status))


# ---------------------------------------------------------------------
# pointwise fields of a frame
# ---------------------------------------------------------------------

def fields(fr, args):
    k1, k2, x, y, L = fr["k1"], fr["k2"], fr["x"], fr["y"], fr["L"]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    D_ = lambda F, ax: inner_deriv(F, args.eta * L, dx, dy, ax, "poly", args.poly_p)
    w = k1 ** 2 + k2 ** 2 - 1.0
    k1x, k1y, k2x, k2y = D_(k1, "x"), D_(k1, "y"), D_(k2, "x"), D_(k2, "y")
    divk = k1x + k2y
    J = k1x * k2y - k1y * k2x
    wx, wy = D_(w, "x"), D_(w, "y")
    S = w * divk + k1 * wx + k2 * wy
    G2sh = np.maximum(w ** 2 - w ** 4, 0.0)
    Gsh = np.sign(w) * np.sqrt(G2sh)
    # the sector sign from smoothed fields (the pointwise sign flips with the estimator's ripple)
    sg = 0.5 * L / dx
    Ds, ws = gaussian_filter(divk, sg, mode="nearest"), gaussian_filter(w, sg, mode="nearest")
    sigma = -np.sign(Ds * ws)
    sigma[sigma == 0] = 1.0
    return dict(w=w, divk=divk, J=J, S=S, Gsh=Gsh, G2sh=G2sh, sigma=sigma, k1=k1, k2=k2,
                LapD=D_(D_(divk, "x"), "x") + D_(D_(divk, "y"), "y"), dx=dx, dy=dy)


def weak(F, x, y, L, args, radius=None):
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    r = float(args.radius if radius is None else radius)
    fx, _, _, _, mx = kernel_d3(r * L, dx, args.poly_p)
    fy, _, _, _, my = kernel_d3(r * L, dy, args.poly_p)
    return corr1d(corr1d(F, fy, 0, dy), fx, 1, dx), (mx, my)


def weak_divk(k1, k2, x, y, L, args, radius=None):
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    r = float(args.radius if radius is None else radius)
    fx, fx1, _, _, mx = kernel_d3(r * L, dx, args.poly_p)
    fy, fy1, _, _, my = kernel_d3(r * L, dy, args.poly_p)
    K = lambda F, kx, ky: corr1d(corr1d(F, ky, 0, dy), kx, 1, dx)
    return -(K(k1, fx1, fy) + K(k2, fx, fy1)), (mx, my)


# ---------------------------------------------------------------------
# stage 2: sign-blind equipartition
# ---------------------------------------------------------------------

def equipartition(fr, P, args, lines, out):
    x, y, L = fr["x"], fr["y"], fr["L"]
    keep, d_row = domain_keep(x, y, L, fr["y_row"], fr["d_core"], args, fr["side"])
    T, m = weak(P["divk"] ** 2, x, y, L, args)
    Cw, _ = weak(P["w"] ** 2, x, y, L, args)
    Cg, _ = weak(P["G2sh"], x, y, L, args)
    IY, IX = centres(P["w"].shape, m, keep, args)
    dist = (fr["d_core"] if fr["d_core"] is not None else d_row)
    bins = CORE_BINS if fr["d_core"] is not None else ROW_BINS
    kind = "core" if fr["d_core"] is not None else "row"
    t, cw, cg, dd = T[IY, IX], Cw[IY, IX], Cg[IY, IX], dist[IY, IX]
    xs = x[IX]
    rows = []
    ratio = lambda a, b_, sel: (float(np.dot(a[sel], b_[sel]) / max(np.dot(a[sel], a[sel]), 1e-300)) if sel.sum() else np.nan)
    hold = xs < np.median(xs)
    for lo, hi in [(0.0, np.inf)] + bins:
        sel = (dd >= lo) & (dd < hi)
        if sel.sum() < 20:
            continue
        b2w = ratio(cw, t, sel & hold); b2g = ratio(cg, t, sel & hold)
        r2w = r2(t[sel & ~hold], b2w * cw[sel & ~hold]); r2g = r2(t[sel & ~hold], b2g * cg[sel & ~hold])
        pw_sel = keep & (dist >= lo) & (dist < hi) & (np.abs(P["w"]) > args.w_min)
        pwG = float(np.median(np.abs(P["divk"][pw_sel]) / np.maximum(np.abs(P["Gsh"][pw_sel]), 1e-9))) if pw_sel.sum() > 30 else np.nan
        pww = float(np.median(np.abs(P["divk"][pw_sel]) / np.abs(P["w"][pw_sel]))) if pw_sel.sum() > 30 else np.nan
        pn_share = float(np.mean(P["divk"][pw_sel] * P["w"][pw_sel] > 0)) if pw_sel.sum() > 30 else np.nan
        rows.append(dict(kind=kind, lo=lo, hi=(hi if np.isfinite(hi) else -1), n=int(sel.sum()),
                         beta_w=float(np.sqrt(max(b2w, 0))), r2_w=r2w, beta_G=float(np.sqrt(max(b2g, 0))), r2_G=r2g,
                         pw_beta_w=pww, pw_beta_G=pwG, pw_pn_share=pn_share))
    lines.append("  stage 2, sign-blind equipartition (div k)^2 = beta^2 G^2, weak at %.2g L, by %s distance (L); beta on G^2 = w^2 | on G^2 = w^2 - w^4 (holdout R2) | pointwise median |div k|/|w| and |div k|/|G|, share of pixels with the PN sign (div k w > 0):" % (args.radius, kind))
    for r_ in rows:
        lines.append("     %s..%s: n %5d  beta %.3f (%.3f) | %.3f (%.3f) | pw %.3f, %.3f, PN share %.2f" % (
            ("%.2f" % r_["lo"]), ("%.2f" % r_["hi"]) if r_["hi"] >= 0 else "inf", r_["n"], r_["beta_w"], r_["r2_w"], r_["beta_G"], r_["r2_G"], r_["pw_beta_w"], r_["pw_beta_G"], r_["pw_pn_share"]))
    # the collar value: 0.5 to 1 L from the cores (or the whole band on a knee)
    if kind == "core":
        cands = [r_ for r_ in rows if (r_["lo"], r_["hi"]) in ((0.5, 0.75), (0.75, 1.0))]
    else:
        cands = [r_ for r_ in rows if r_["lo"] == 0.0 and r_["hi"] < 0]
    beta_c = float(np.mean([r_["beta_G"] for r_ in cands])) if cands else rows[0]["beta_G"]
    beta_cw = float(np.mean([r_["beta_w"] for r_ in cands])) if cands else rows[0]["beta_w"]
    lines.append("     collar beta (G^2 = w^2 - w^4): %.3f; (w^2): %.3f; theory sqrt(alpha/eta) = %.3f" % (beta_c, beta_cw, BETA_THEORY))
    with open(out / "equi_bins.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    return rows, beta_c, beta_cw


def equipartition_rowavg(fr, P, args, lines):
    """data only: x-averaged (div k)^2 and G^2 profiles, 1-D weak in y at --radius L, beta by row-distance bins"""
    x, y, L = fr["x"], fr["y"], fr["L"]
    wx, N, _ = hann_window(x, L, 0.8)
    dy = float(y[1] - y[0])
    fy, _, _, _, my = kernel_d3(args.radius * L, dy, args.poly_p)
    av = lambda F: (F * wx[None, :]).sum(axis=1)
    K = lambda v: corr1d(v, fy, 0, dy)
    t, cg, cw = K(av(P["divk"] ** 2)), K(av(P["G2sh"])), K(av(P["w"] ** 2))
    d = np.abs(y - fr["y_row"]) / L
    i = np.arange(len(y)); ok = (i >= my) & (i < len(y) - my)
    rows = []
    for lo, hi in [(0.0, args.band_to)] + ROW_BINS:
        sel = ok & (d >= lo) & (d < hi)
        if sel.sum() < 5:
            continue
        bg = float(np.dot(cg[sel], t[sel]) / max(np.dot(cg[sel], cg[sel]), 1e-300))
        bw = float(np.dot(cw[sel], t[sel]) / max(np.dot(cw[sel], cw[sel]), 1e-300))
        rows.append(dict(lo=lo, hi=hi, n=int(sel.sum()), beta_G=float(np.sqrt(max(bg, 0))), beta_w=float(np.sqrt(max(bw, 0))),
                         r2_G=r2(t[sel], bg * cg[sel])))
    lines.append("  stage 2b, row-averaged equipartition (Hann over %d periods), beta on w^2 - w^4 | w^2 (R2) by row distance:" % N)
    lines.append("     " + "; ".join("%.2f..%.2f: %.3f | %.3f (%.3f)" % (r_["lo"], r_["hi"], r_["beta_G"], r_["beta_w"], r_["r2_G"]) for r_ in rows))
    return rows


# ---------------------------------------------------------------------
# stage 3: chi measured, ring averages
# ---------------------------------------------------------------------

def ring_profiles(fr, P, beta_c, args, lines, out, extra=None):
    x, y, L = fr["x"], fr["y"], fr["L"]
    keep, d_row = domain_keep(x, y, L, fr["y_row"], None, args, fr["side"])
    if fr["d_core"] is None:
        lines.append("  stage 3: no cores (a knee); chi measured on row bins instead of rings")
        dist = d_row
        edges = [0.0] + [b[1] for b in ROW_BINS]
    else:
        dist = fr["d_core"]
        edges = list(RINGS)
    absD = np.abs(P["divk"]); absG = np.abs(P["Gsh"] if args.Gform == "sh" else P["w"])
    chi_blind = absD / beta_c - absG                                     # chi sgn(w)
    chi_signed = np.sign(P["w"]) * chi_blind                             # the paper's chi
    chi_b = P["divk"] + P["sigma"] * beta_c * P["w"]                     # the ledger's chi_b (leading order G = w)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = keep & (dist >= lo) & (dist < hi)
        if sel.sum() < 20:
            continue
        r_ = dict(lo=lo, hi=hi, n=int(sel.sum()), absD_over_beta=float(np.mean(absD[sel])) / beta_c, absG=float(np.mean(absG[sel])),
                  chi_blind=float(np.mean(chi_blind[sel])), chi_signed=float(np.mean(chi_signed[sel])), chi_b=float(np.mean(chi_b[sel])),
                  chi_b_rms=float(np.sqrt(np.mean(chi_b[sel] ** 2))), J=float(np.mean(P["J"][sel])) * L ** 2,
                  J_abs=float(np.mean(np.abs(P["J"][sel]))) * L ** 2, sigma_neg=float(np.mean(P["sigma"][sel] < 0)),
                  Jx_abs=(float(np.mean(np.abs((P["J"] - P["J0"])[sel]))) * L ** 2) if "J0" in P else np.nan)
        if extra is not None:
            for k_, F in extra.items():
                r_[k_] = float(np.mean(F[sel])); r_[k_ + "_rms"] = float(np.sqrt(np.mean(F[sel] ** 2)))
                r_[k_ + "_corr"] = float(np.corrcoef(F[sel], chi_b[sel])[0, 1]) if np.std(F[sel]) > 0 and np.std(chi_b[sel]) > 0 else np.nan
        rows.append(r_)
    # decay exponent of the sign-blind chi over the rings where it is positive, from 0.5 L
    rr = np.array([0.5 * (r_["lo"] + r_["hi"]) for r_ in rows]); cc = np.array([r_["chi_blind"] for r_ in rows])
    ok = (cc > 0) & (rr >= 0.5)
    expo = np.nan
    if ok.sum() >= 3:
        p = np.polyfit(np.log(rr[ok]), np.log(cc[ok]), 1); expo = float(p[0])
    lines.append("  stage 3, chi measured with beta_c %.3f (G = %s); by %s (L): |div k|/beta, |G|, chi sign-blind, chi signed, chi_b (ledger), rms chi_b, <J> L^2, <|J|> L^2, share sigma = -1%s:" % (
        beta_c, "w sqrt(1 - w^2)" if args.Gform == "sh" else "w", "core rings" if fr["d_core"] is not None else "row bins", "; chi_solved (stage 4), rms, corr with chi_b" if extra else ""))
    for r_ in rows:
        s_ = "     %.2f..%.2f n %5d: %.3f %.3f | chi %+.4f %+.4f | chi_b %+.4f (rms %.4f) | J %+.3f |J| %.3f |J - J0| %s | neg %.2f" % (
            r_["lo"], r_["hi"], r_["n"], r_["absD_over_beta"], r_["absG"], r_["chi_blind"], r_["chi_signed"], r_["chi_b"], r_["chi_b_rms"], r_["J"], r_["J_abs"],
            ("%.3f" % r_["Jx_abs"]) if np.isfinite(r_["Jx_abs"]) else "n/a", r_["sigma_neg"])
        if extra:
            s_ += " | " + " ".join("%s %+.4f (rms %.4f, corr %.2f)" % (k_, r_[k_], r_[k_ + "_rms"], r_[k_ + "_corr"]) for k_ in extra)
        lines.append(s_)
    lines.append("     decay of the sign-blind chi over the rings where it is positive (from 0.5 L): exponent %s (the paper: -1)" % (("%.2f" % expo) if np.isfinite(expo) else "n/a"))
    with open(out / "chi_rings.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    return rows, expo, dict(chi_blind=chi_blind, chi_signed=chi_signed, chi_b=chi_b)


# ---------------------------------------------------------------------
# stage 4: chi solved from (1.19)
# ---------------------------------------------------------------------

def null_J(path, fr, args, lines):
    """data: J of mg applied to the last frame's single-mode x-invariant part (the matched null of
    knee_rowavg_laws v1.1), on the same grid and taper; J - J0 is the source free of the ripple products"""
    load_npz, estimate, _, _, t1c = SD._estimator()
    D = load_npz(path)
    u = D["u"][..., -1].astype(float); x, y = fr["x"], fr["y"]
    k1 = np.sqrt(1 - float(args.mu) ** 2)
    wx, N, _ = hann_window(x, fr["L"], 0.8)
    cbar = (u * np.exp(-1j * k1 * x)[None, :] * wx[None, :]).sum(axis=1)
    u0 = np.real(2.0 * cbar[:, None] * np.exp(1j * k1 * x)[None, :])
    ny, nx = u.shape; dx = float(x[1] - x[0])
    taper = t1c.edge_taper(x, y, frac=args.taper_frac)
    e = estimate(u0, nx * dx, ny * dx, args.sigma_dir, None, taper)
    sg = np.sign(np.mean(e["k1"])) or 1.0
    P0 = fields(dict(k1=e["k1"] * sg, k2=e["k2"] * sg, x=x, y=y, L=fr["L"]), args)
    lines.append("  matched null: mg on the frame's x-invariant part (Hann over %d periods); rms J0 L^2 on the band %.3f" % (
        N, float(np.sqrt(np.mean(P0["J"][np.abs(y - fr["y_row"]) < 2 * fr["L"]] ** 2))) * fr["L"] ** 2))
    return P0["J"]


def solve_chi(P, x, y, L, beta, args, lines):
    """Lap chi - 2 b d_i(sigma k_i chi) = -4 b sigma J on the untapered interior box, chi = 0 on its edges"""
    ny, nx = P["w"].shape
    dx, dy = P["dx"], P["dy"]
    fr_ = args.rim_frac
    i0, i1 = int(fr_ * ny), int((1 - fr_) * ny); j0, j1 = int(fr_ * nx), int((1 - fr_) * nx)
    ax = (P["sigma"] * P["k1"])[i0:i1, j0:j1]; ay = (P["sigma"] * P["k2"])[i0:i1, j0:j1]
    Jsrc = P["J"] - P["J0"] if ("J0" in P and int(args.null)) else P["J"]
    src = (-4.0 * beta * P["sigma"] * Jsrc)[i0:i1, j0:j1]
    m, n = ax.shape
    N = m * n
    idx = np.arange(N).reshape(m, n)
    # Laplacian
    main = np.full(N, -2.0 / dx ** 2 - 2.0 / dy ** 2)
    rows_, cols_, vals_ = [idx.ravel()], [idx.ravel()], [main]

    def add(ii, jj, v):
        rows_.append(ii.ravel()); cols_.append(jj.ravel()); vals_.append(v.ravel())
    # x neighbours: Laplacian 1/dx^2, advection -2b (a_x chi)_x central: -2b (a_{j+1} chi_{j+1} - a_{j-1} chi_{j-1})/(2 dx)
    add(idx[:, :-1], idx[:, 1:], 1.0 / dx ** 2 - 2 * beta * ax[:, 1:] / (2 * dx))
    add(idx[:, 1:], idx[:, :-1], 1.0 / dx ** 2 + 2 * beta * ax[:, :-1] / (2 * dx))
    add(idx[:-1, :], idx[1:, :], 1.0 / dy ** 2 - 2 * beta * ay[1:, :] / (2 * dy))
    add(idx[1:, :], idx[:-1, :], 1.0 / dy ** 2 + 2 * beta * ay[:-1, :] / (2 * dy))
    A = sparse.csc_matrix((np.concatenate(vals_), (np.concatenate(rows_), np.concatenate(cols_))), shape=(N, N))
    t0 = time.time()
    chi = spsolve(A, src.ravel()).reshape(m, n)
    pe = float(2 * beta * np.max(np.hypot(ax, ay)) * max(dx, dy))
    lines.append("  stage 4: (1.19) solved on the interior box %d x %d (zero Dirichlet at its edges, %.1f L from the row at the nearest), grid Peclet %.2f, %.1f s; source %s" % (
        m, n, min(abs(y[i0] - y[len(y) // 2]), abs(y[i1 - 1] - y[len(y) // 2])) / L, pe, time.time() - t0,
        "J - J0 (matched null)" if ("J0" in P and int(args.null)) else "J"))
    full = np.zeros_like(P["w"]); full[i0:i1, j0:j1] = chi
    return full


def chi_fits(fr, P, chi_s, beta_c, args, lines):
    x, y, L = fr["x"], fr["y"], fr["L"]
    keep, d_row = domain_keep(x, y, L, fr["y_row"], fr["d_core"], args, fr["side"])
    Dw, m = weak_divk(fr["k1"], fr["k2"], x, y, L, args)
    Cw, _ = weak(-P["sigma"] * P["w"], x, y, L, args)
    Cchi, _ = weak(chi_s, x, y, L, args)
    IY, IX = centres(P["w"].shape, m, keep, args)
    dist = (fr["d_core"] if fr["d_core"] is not None else d_row)
    bins = CORE_BINS if fr["d_core"] is not None else ROW_BINS
    b, cw, cc, dd, xs = Dw[IY, IX], Cw[IY, IX], Cchi[IY, IX], dist[IY, IX], x[IX]
    hold = xs < np.median(xs)
    rows = []
    dead = float(np.sqrt(np.mean(cc ** 2))) < 1e-4 * float(np.sqrt(np.mean(cw ** 2)))
    if dead:
        lines.append("  stage 4: chi_solved is at round-off on the centres (no J source after the null): its coefficient is not read")
    for lo, hi in [(0.0, np.inf)] + bins:
        sel = (dd >= lo) & (dd < hi)
        if sel.sum() < 20:
            continue
        tr, te = sel & hold, sel & ~hold
        c1, *_ = np.linalg.lstsq(cw[tr, None], b[tr], rcond=None)
        Th = np.column_stack([cw, cc])
        c2, *_ = np.linalg.lstsq(Th[tr], b[tr], rcond=None)
        if dead:
            c2 = np.array([c1[0], np.nan])
        rows.append(dict(lo=lo, hi=(hi if np.isfinite(hi) else -1), n=int(sel.sum()), beta_alone=float(c1[0]), r2_alone=r2(b[te], c1[0] * cw[te]),
                         beta_with=float(c2[0]), c_chi=float(c2[1]), r2_with=(r2(b[te], Th[te] @ c2) if not dead else np.nan),
                         r2_fixed=r2(b[te], beta_c * cw[te] + cc[te])))
    lines.append("  stage 4 fits of weak div k: {-sigma w} alone -> beta (R2) | {-sigma w, chi_solved} -> beta, c_chi (R2; c_chi must read 1) | R2 of the fixed law -sigma beta_c w + chi_solved:")
    for r_ in rows:
        lines.append("     %s..%s n %5d: %.3f (%.3f) | %.3f, %.3f (%.3f) | fixed %.3f" % (
            "%.2f" % r_["lo"], ("%.2f" % r_["hi"]) if r_["hi"] >= 0 else "inf", r_["n"], r_["beta_alone"], r_["r2_alone"], r_["beta_with"], r_["c_chi"], r_["r2_with"], r_["r2_fixed"]))
    return rows


def relation_B_residual(fr, P, beta_c, args, lines):
    """the relation-B residual Lap div k - 2 beta^2 S, weak, by ring; the floor of the stage-4 test (it is the (1.19) residual of the measured chi_b)"""
    x, y, L = fr["x"], fr["y"], fr["L"]
    keep, d_row = domain_keep(x, y, L, fr["y_row"], fr["d_core"], args, fr["side"])
    dx, dy = P["dx"], P["dy"]
    fx, fx1, fx2, fx3, mx = kernel_d3(args.radius * L, dx, args.poly_p)
    fy, fy1, fy2, fy3, my = kernel_d3(args.radius * L, dy, args.poly_p)
    K = lambda F, kx, ky: corr1d(corr1d(F, ky, 0, dy), kx, 1, dx)
    k1, k2 = fr["k1"], fr["k2"]
    LapD = -(K(k1, fx3, fy) + K(k1, fx1, fy2) + K(k2, fx2, fy1) + K(k2, fx, fy3))
    Sw = K(P["S"], fx, fy)
    IY, IX = centres(P["w"].shape, (mx, my), keep, args)
    dist = (fr["d_core"] if fr["d_core"] is not None else d_row)
    bins = CORE_BINS if fr["d_core"] is not None else ROW_BINS
    a, s_, dd = LapD[IY, IX], Sw[IY, IX], dist[IY, IX]
    rows = []
    for lo, hi in [(0.0, np.inf)] + bins:
        sel = (dd >= lo) & (dd < hi)
        if sel.sum() < 20:
            continue
        res = a[sel] - 2 * beta_c ** 2 * s_[sel]
        lam = float(np.dot(s_[sel], a[sel]) / max(np.dot(s_[sel], s_[sel]), 1e-300))
        rows.append(dict(lo=lo, hi=(hi if np.isfinite(hi) else -1), n=int(sel.sum()), rel_resid=float(np.sqrt(np.mean(res ** 2)) / max(np.sqrt(np.mean(a[sel] ** 2)), 1e-300)),
                         lambda_fit=lam, lambda_pred=2 * beta_c ** 2))
    lines.append("  relation B at beta_c (Lap div k = 2 beta^2 S, predicted lambda %.3f): relative residual and the free lambda by bin: %s" % (
        2 * beta_c ** 2, "; ".join("%s..%s %.2f (lam %+.2f)" % ("%.2f" % r_["lo"], ("%.2f" % r_["hi"]) if r_["hi"] >= 0 else "inf", r_["rel_resid"], r_["lambda_fit"]) for r_ in rows)))
    return rows


# ---------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------

def fig_chi(fr, P, chis, chi_s, rings, label, path, beta_c):
    x, y, L = fr["x"], fr["y"], fr["L"]
    yr = fr["y_row"]
    band = np.abs(y - yr) < 2.5 * L
    fig, ax = plt.subplots(2, 3, figsize=(17, 8))
    ext = [x[0] / L, x[-1] / L, (y[band][0] - yr) / L, (y[band][-1] - yr) / L]
    for a_, F, lab in ((ax[0, 0], chis["chi_b"], "chi_b = div k + sigma beta_c w (measured)"), (ax[0, 1], chi_s, "chi solved from (1.19)"),
                       (ax[0, 2], P["J"] * L ** 2, "J L^2"), (ax[1, 0], P["sigma"], "sigma = sgn(-div k w), smoothed 0.5 L")):
        Fb = F[band]
        lim = np.nanpercentile(np.abs(Fb), 99) if lab[:5] != "sigma" else 1
        im = a_.imshow(Fb, origin="lower", extent=ext, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        fig.colorbar(im, ax=a_); a_.set_title(lab, fontsize=9); a_.set_xlabel("x / L"); a_.set_ylabel("(y - y_row) / L")
    rr = [0.5 * (r_["lo"] + r_["hi"]) for r_ in rings]
    ax[1, 1].plot(rr, [r_["absD_over_beta"] for r_ in rings], "o-", color=C1, label="|div k| / beta_c")
    ax[1, 1].plot(rr, [r_["absG"] for r_ in rings], "s-", color=C2, label="|G|")
    ax[1, 1].plot(rr, [r_["chi_blind"] for r_ in rings], "^-", color=C3, label="chi sign-blind")
    ax[1, 1].plot(rr, [r_["chi_b"] for r_ in rings], "v-", color=C4, label="chi_b (ledger)")
    if "chi_solved" in rings[0]:
        ax[1, 1].plot(rr, [r_["chi_solved"] for r_ in rings], "d--", color="k", label="chi solved")
    ax[1, 1].axhline(0, color="0.6", lw=0.6); ax[1, 1].set_xlabel("distance (L)"); ax[1, 1].legend(fontsize=7)
    ax[1, 1].set_title("ring means (beta_c %.3f)" % beta_c, fontsize=9)
    ax[1, 2].plot(rr, [r_["J"] for r_ in rings], "o-", color=C1, label="<J> L^2"); ax[1, 2].plot(rr, [r_["J_abs"] for r_ in rings], "s--", color=C2, label="<|J|> L^2")
    ax[1, 2].axhline(0, color="0.6", lw=0.6); ax[1, 2].set_xlabel("distance (L)"); ax[1, 2].legend(fontsize=7); ax[1, 2].set_title("J by ring", fontsize=9)
    fig.suptitle(label, fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fig_set(table, path):
    rows = [r_ for r_ in table if np.isfinite(r_.get("mu", np.nan))]
    if not rows:
        return
    rows = sorted(rows, key=lambda r_: r_["mu"])
    mu = [r_["mu"] for r_ in rows]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(mu, [r_["beta_collar_G"] for r_ in rows], "o-", color=C1, label="collar, G^2 = w^2 - w^4")
    ax[0].plot(mu, [r_["beta_collar_w"] for r_ in rows], "s--", color=C2, label="collar, w^2")
    ax[0].plot(mu, [r_["beta_near_G"] for r_ in rows], "^-", color=C3, label="0.25 to 0.5 L from the cores")
    ax[0].axhline(BETA_THEORY, color="k", ls=":", lw=0.8, label="sqrt(alpha/eta) = 0.866")
    ax[0].set_xlabel("mu"); ax[0].set_title("equipartition beta", fontsize=9); ax[0].legend(fontsize=7)
    ax[1].plot(mu, [r_["c_chi_band"] for r_ in rows], "o-", color=C1, label="band"); ax[1].plot(mu, [r_["c_chi_near"] for r_ in rows], "^-", color=C3, label="0.25 to 0.5 L")
    ax[1].axhline(1, color="k", ls=":", lw=0.8); ax[1].set_xlabel("mu"); ax[1].set_title("coefficient of chi solved (1 if (1.19) holds)", fontsize=9); ax[1].legend(fontsize=7)
    ax[2].plot(mu, [r_["r2_alone_near"] for r_ in rows], "o-", color=C2, label="{-sigma w} alone, near cores")
    ax[2].plot(mu, [r_["r2_with_near"] for r_ in rows], "s-", color=C1, label="with chi solved, near cores")
    ax[2].plot(mu, [r_["r2_alone_band"] for r_ in rows], "o--", color=C2, alpha=0.5, label="alone, band"); ax[2].plot(mu, [r_["r2_with_band"] for r_ in rows], "s--", color=C1, alpha=0.5, label="with chi, band")
    ax[2].set_xlabel("mu"); ax[2].set_title("holdout R2 of weak div k", fontsize=9); ax[2].legend(fontsize=7); ax[2].set_ylim(-0.2, 1.02)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ---------------------------------------------------------------------
# one field
# ---------------------------------------------------------------------

def run_field(stem, route, fr, args, lines, out, table, mu, path=None):
    tag = "%s_%s" % (stem, route)
    fo = out / tag; fo.mkdir(parents=True, exist_ok=True)
    rep = ["[%s] route %s, L %.3f, grid %d x %d, cores %s, side %s" % (
        stem, route, fr["L"], fr["k1"].shape[0], fr["k1"].shape[1], "none" if fr["d_core"] is None else "yes", fr["side"])]
    P = fields(fr, args)
    if path is not None and int(args.null):
        try:
            P["J0"] = null_J(path, fr, args, rep)
        except Exception as exc:
            rep.append("  matched null failed: %r" % exc)
    keep, _ = domain_keep(fr["x"], fr["y"], fr["L"], fr["y_row"], fr["d_core"], args, fr["side"])
    rep.append("  sigma = -1 on %.3f of the kept domain; max |w| on it %.3f (G^2 = w^2 - w^4 needs |w| < 1)" % (float(np.mean(P["sigma"][keep] < 0)), float(np.abs(P["w"][keep]).max())))
    erows, beta_c, beta_cw = equipartition(fr, P, args, rep, fo)
    rrows = None
    if route == "estimator" and args.npz:
        rrows = equipartition_rowavg(fr, P, args, rep)
    if args.beta_fixed > 0:
        beta_use, src = float(args.beta_fixed), "--beta_fixed"
    elif route == "exact":
        key = "pw_beta_G" if args.Gform == "sh" else "pw_beta_w"
        cands = [r_[key] for r_ in erows if (r_["lo"], r_["hi"]) in ((0.5, 0.75), (0.75, 1.0)) and np.isfinite(r_[key])]
        beta_use = float(np.mean(cands)) if cands else (beta_c if args.Gform == "sh" else beta_cw)
        src = "pointwise collar median (exact route)" if cands else "weak collar (no pointwise collar)"
    else:
        beta_use, src = (beta_c if args.Gform == "sh" else beta_cw), "weak collar value on G^2 = %s" % ("w^2 - w^4" if args.Gform == "sh" else "w^2")
    rep.append("  beta for stages 3 and 4: %.3f (%s)" % (beta_use, src))
    chi_s = solve_chi(P, fr["x"], fr["y"], fr["L"], beta_use, args, rep)
    rings, expo, chis = ring_profiles(fr, P, beta_use, args, rep, fo, extra={"chi_solved": chi_s})
    frows = chi_fits(fr, P, chi_s, beta_use, args, rep)
    brows = relation_B_residual(fr, P, beta_use, args, rep)
    try:
        fig_chi(fr, P, chis, chi_s, rings, tag, fo / ("fig_chi_%s.png" % tag), beta_use)
    except Exception as exc:
        rep.append("  figure failed: %r" % exc)
    g = lambda rows, lo, hi, key: next((r_[key] for r_ in rows if abs(r_["lo"] - lo) < 1e-9 and (r_["hi"] == hi or (hi < 0 and r_["hi"] < 0))), np.nan)
    table.append(dict(field=tag, mu=mu, beta_collar_G=beta_c, beta_collar_w=beta_cw, beta_band_G=g(erows, 0.0, -1, "beta_G"), beta_band_w=g(erows, 0.0, -1, "beta_w"),
                      beta_near_G=g(erows, 0.25, 0.5, "beta_G"), r2_band_G=g(erows, 0.0, -1, "r2_G"),
                      chi_expo=expo, chi_blind_near=g(rings, 0.25, 0.5, "chi_blind") if fr["d_core"] is not None else np.nan,
                      chi_solved_near=g(rings, 0.25, 0.5, "chi_solved") if fr["d_core"] is not None else np.nan,
                      chi_corr_near=g(rings, 0.25, 0.5, "chi_solved_corr") if fr["d_core"] is not None else np.nan,
                      chi_corr_collar=g(rings, 0.5, 0.75, "chi_solved_corr") if fr["d_core"] is not None else np.nan,
                      c_chi_band=g(frows, 0.0, -1, "c_chi"), r2_alone_band=g(frows, 0.0, -1, "r2_alone"), r2_with_band=g(frows, 0.0, -1, "r2_with"),
                      c_chi_near=g(frows, 0.25, 0.5, "c_chi"), r2_alone_near=g(frows, 0.25, 0.5, "r2_alone"), r2_with_near=g(frows, 0.25, 0.5, "r2_with"),
                      relB_resid_band=g(brows, 0.0, -1, "rel_resid"), relB_lambda_band=g(brows, 0.0, -1, "lambda_fit"),
                      rowavg_beta_band=(rrows[0]["beta_G"] if rrows else np.nan)))
    with open(fo / "fits.json", "w") as f:
        json.dump(dict(equipartition=erows, rowavg=rrows, rings=rings, chi_fits=frows, relation_B=brows, beta_c=beta_c, beta_cw=beta_cw, chi_expo=expo), f, indent=1, default=float)
    (fo / "report.txt").write_text("\n".join(rep) + "\n")
    lines.extend(rep)


def main(args):
    out = Path(args.outdir)
    if not out.is_absolute():
        out = HERE / out
    out.mkdir(parents=True, exist_ok=True)
    lines = ["stationary_defect_library v1.0; radius %.2g L, poly p %d, eta %.2g, stride %d, domain %s (band_to %.2g), core discs %.2g L, core_def %s, w_min %.2g; theory beta %.3f" % (
        args.radius, args.poly_p, args.eta, args.stride, args.domain, args.band_to, args.core_excl, args.core_def, args.w_min, BETA_THEORY)]
    table = []
    if args.npz:
        for p in [s for s in args.npz.split(",") if s.strip()]:
            stem, routes = SD.data_samples(p, args, lines)
            for route, samples in routes:
                run_field(stem, route, samples[-1], args, lines, out, table, float(args.mu), path=p)
    elif args.mode == "target":
        stem, routes, info = target_frames(args)
        args.domain = "all"
        lines.append("target (the paper's section 2.6), b %.2f: k ~ %.3f r at the core (the paper: 0.8 r), BVP status %d; L = 2 pi (one wavelength); domain all; routes %s" % (
            args.beta, info["k_slope"], info["status"], [r_ for r_, _ in routes]))
        for route, samples in routes:
            run_field(stem, route, samples[-1], args, lines, out, table, float("nan"))
    else:
        stem, routes = SD.synthetic(args)
        lines.append("synthetic %s, mu %.2f, routes %s" % (args.mode, args.mu, [r_ for r_, _ in routes]))
        for route, samples in routes:
            run_field(stem, route, samples[-1], args, lines, out, table, float(args.mu))
    if table:
        keys = list(table[0].keys())
        with open(out / "library_table.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(table)
        md_ = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
        for r_ in table:
            md_.append("| " + " | ".join(("%.4g" % v) if isinstance(v, float) else str(v) for v in r_.values()) + " |")
        (out / "library_table.md").write_text("\n".join(md_) + "\n")
        if len(table) > 2:
            fig_set(table, out / "fig_library_vs_mu.png")
    (out / "log.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


class _Args:
    mode = "knee"                 # knee | pn | target (synthetic, both routes) ; ignored when --npz is given
    target_half = 8.0             # target: half side of the box, wavelengths
    target_dx = 0.2               # target: grid spacing
    npz = ""                      # last-frame files, comma-separated
    outdir = "results/stationary_defect_library"
    route = "both"                # synthetic: exact | estimator | both
    frames = "last"
    t0 = 0.0
    t1 = 1e9
    mu = 0.85
    beta = 1.3                    # knee: planted self-dual coefficient
    R = 0.5
    Ly = 60.0
    ny = 300
    n_xper = 6
    n_disl = 10
    pn_margin = 1.5
    pn_core = 1.5
    sigma_dir = 0.4
    taper_frac = 0.06
    core_frac = 0.70
    core_excl = 0.25              # core discs, L (the rings start at 0.25 L)
    core_def = "circulation"
    wind_hi = 0.9
    sigma_det_charge = 0.3
    radius = 1.5                  # test-function half width, L
    poly_p = 6
    eta = 0.0
    stride = 4
    rim = 4
    rim_frac = 0.10
    domain = "band"
    band_to = 2.0
    far_from = 4.0
    side = "both"
    w_min = 0.02
    beta_fixed = -1.0             # > 0: use this beta in stages 3 and 4 instead of the collar value
    null = 1                      # data: the matched-null J (J minus mg's J on the frame's x-invariant part) as the source of (1.19)
    Gform = "w"                   # G used for beta_c and chi measured: w (the paper's 1.20, the identity's form) | sh (w sqrt(1 - w^2), the draft's eq. 10)
    radius_x = -1.0
    radius_y = -1.0
    ry_periods = 0.0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in vars(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    main(p.parse_args())
