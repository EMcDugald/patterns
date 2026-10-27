"""knee_jt_laws.py  v1.0  (13 Oct 2026)   workstreams G13 and G26 of the research thread

The J_t and chi_J_t laws on row averages, and the growth of the x-structure.

On the zigzag every even field repeats along the line every L = pi/k1
(knee_rowavg_laws v1.1), and for a curl-free k the row average of J is
    <J> = -1/2 d_yy V,     V(y, t) = var_x(k_x)
(the mean of k_x is pinned at k1 on every core-free row). So the J_t law is
a law for V_t, and <J_t> = -1/2 d_yy V_t and <chi_J,t> = (d_yy - m^2)^{-1} <J_t>
follow from it. V is read through mg with the matched null of v1.1
(V_excess = V(frame) - V(mg on the frame's single-mode x-invariant part)).
A second target carries no estimator at all: r3(y, t) = |c3|/|c1|, the
third row harmonic of u relative to the fundamental (c_n = row mean of
u exp(-i n k1 x)); a single-mode knee has r3 at the stripes' own level,
the overlap's combination tones and then the cores raise it on the line.

Tests, per movie, one-dimensional weak form in (y, t) (polynomial bump of
half-width r_y across the row, t_half frames in time; the time derivative
of every target is moved onto the bump):
  (1) the conservation identity: for a gradient field
      <J_t> = d_y <k_xx theta_ty - k_xy theta_tx> exactly; the measured weak
      <J_t> (from V, by parts) against that flux built from mg's own theta_t
      (both with the null removed): relative difference, correlation, and
      the regression coefficient (1 if k is a gradient on the window);
  (2) the growth law: free and forced fits of V_t on
      {V, V w, V^2, Lap V, d_y V, d_y(V k_y), V A_dev, V theta_t, V div k}
      and of r3_t on {r3, r3 w, r3^3, Lap r3, d_y r3, r3 A_dev, r3 theta_t,
      r3 div k}; forced {V, Lap V}, {V, V^2, Lap V}, {V, V w, Lap V} and
      the r3 analogues; the growth rate gamma of r3 on the line from the
      log-slope of its line maximum over the early window (from the first
      frame where r3 exceeds growth_start times the far-field level to the
      first partial circulation, or to where it reaches half its maximum),
      and the same for sqrt(V_excess); the instantaneous d ln r3/dt series;
  (3) the rate law seen through J: the flux is linear in theta_t, so with
      FS = <k_xx S_y - k_xy S_x> and FB = <k_xx B_y - k_xy B_x> (S = div(w k),
      B = Lap div k pointwise per frame) the measured <J_t> fitted on
      {d_y FS, d_y FB} returns the coefficients (a, b) of theta_t = a S + b B
      as J sees them (RCN: (2, -1); G21's tied pair about (0.5, +0.2)); the
      same with the amplitude flux FA for theta_t = ... + c w k.grad ln A.
  Windows: whole, growth, pre, event, post (from --events, as v1.1).

Controls:
  control_rcn     V = 0 and r3 = 0 to round-off: both targets flagged, no fit.
  control_row     the translating dislocation row of v1.1: V_t = -v d_y V
                  exactly, so forced {d_y V} must read -v on the exact
                  route; the identity (1) holds exactly there.
  control_growth  a knee plus a planted third harmonic eps0 exp(gamma t)
                  g(y) cos(3 k1 x + 3 phi(y)), g a Gaussian of width g_L L
                  about the line: r3 on the line grows at exactly gamma on
                  both routes; the mg route shows V's response to a known
                  growth (the stand-in's V floor).

Outputs under --outdir: per movie <stem>[_exact]/report.txt, fits.json,
series_<stem>.csv, fig_jt_<stem>.png; for the set jt_table.csv/.md,
fig_growth_vs_mu.png.

Changelog
  v1.0  first version. Sandbox validation (13 Oct): control_rcn flags both
        targets at the far-field level and fits nothing. control_growth
        (planted gamma 2.0, eps0 1e-4): r3's log-slope reads 1.999 (R2 of
        the exponential 1.000) and the free fit of r3_t returns +1.999 r3
        alone at R2 1.0000; through mg the identity <J_t> = d_y<flux of
        theta_t> reads 1.023 at R2 1.000, and sqrt(V_excess) grows at 1.004,
        half the planted rate: the matched-null excess is linear in the
        planted amplitude through its cross term with the estimator's
        ripple (V_excess reads the x-structure's amplitude, not its
        variance), so V_t = 1.96 V and r3_t = 2.0 r3 are the same growth.
        control_row: r3_t returns -0.4997 d_y r3 alone at R2 1.0000 on both
        routes (planted -0.5); V_t through mg returns d_y V -0.575 at R2
        0.76 (mg regularises the 1/distance divergence of the exact V at
        the cores, which makes the exact-route V_t unfittable: R2 0.00);
        the gradient identity does not hold across the row's branch cuts
        on the exact route (coefficient 21 at R2 0.02) and reads 0.86 at R2
        0.30 through mg. Stand-in SH movie (dt 0.2, not of record): r3 rises
        from the stripes' level (2.6e-3) to half its maximum within the
        first frame, so the growth rate is a lower bound at this cadence
        (the fine runs resolve it); the identity through mg fails on SH
        (coefficient -0.23, R2 0.007), so mg's (k, theta_t) are not a
        consistent gradient pair on the defected SH field and the rate law
        through J (test 3) is not readable through mg; the whole-run laws of
        V_t and r3_t are the slow saturation phase (R2 0.5 to 0.8 free,
        forced below 0.42), to be read on the growth window of the fine runs.
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT / "src", ROOT / "src" / "op_extract", ROOT / "src" / "utils", HERE):
    sys.path.insert(0, str(_p))
import wsindy_theta_t_v19 as V19                                         # noqa: E402
from wsindy_theta_t_v19 import (load_npz, kernel_d3, assemble, fit_pareto, forced_fit, collinearity, cstr, r_squared)  # noqa: E402
from knee_demod_laws import corr_yt                                      # noqa: E402
from knee_rowavg_laws import (hann_window, xinv_part, mg_frames, shift_flip_residual_interior, read_events,   # noqa: E402
                              screened_1d, control_rcn as _control_rcn, control_row as _control_row)
import matplotlib                                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
PROF = ["kx", "kx2", "ky", "w", "D", "A", "tht", "FJ", "Fjt", "FS", "FB", "FA", "wky"]
V_LIB = ["V", "V w", "V^2", "Lap V", "d_y V", "d_y(V k_y)", "V A_dev", "V theta_t", "V div k"]
R_LIB = ["r3", "r3 w", "r3^3", "Lap r3", "d_y r3", "r3 A_dev", "r3 theta_t", "r3 div k"]


# ---------------------------------------------------------------------
# averaged profiles
# ---------------------------------------------------------------------

def avg_frame(kx, ky, A, tht, dx, dy, wx, A_floor):
    g = lambda F, ax: np.gradient(F, dx if ax == 1 else dy, axis=ax)
    w = kx ** 2 + ky ** 2 - 1.0
    kxx, kxy, kyx, kyy = g(kx, 1), g(kx, 0), g(ky, 1), g(ky, 0)
    D = kxx + kyy
    S = g(w * kx, 1) + g(w * ky, 0)
    B = g(g(D, 1), 1) + g(g(D, 0), 0)
    lnA = np.log(np.maximum(A, A_floor))
    Aterm = w * (kx * g(lnA, 1) + ky * g(lnA, 0))
    flux = lambda F: kxx * g(F, 0) - kxy * g(F, 1)
    Q = dict(kx=kx, kx2=kx ** 2, ky=ky, w=w, D=D, A=A, tht=tht, FJ=kx * kyx, Fjt=flux(tht), FS=flux(S), FB=flux(B), FA=flux(Aterm), wky=w * ky)
    return {k: (v * wx[None, :]).sum(axis=1) for k, v in Q.items()}


def profiles(frames, x, y, t, wx, A_floor):
    dx = float(x[1] - x[0]); dy = float(y[1] - y[0])
    P = {k: np.empty((len(y), len(t))) for k in PROF}
    for i, (kx, ky, A, tht) in enumerate(frames):
        q = avg_frame(kx, ky, A, tht, dx, dy, wx, A_floor)
        for k in PROF:
            P[k][:, i] = q[k]
    return P


def harmonics(u, x, k1, wx):
    """r3 = |c3|/|c1| and r5 per row per frame, c_n the Hann-weighted row mean of u exp(-i n k1 x)"""
    nt = u.shape[2]
    e = {n: np.exp(-1j * n * k1 * x)[None, :] * wx[None, :] for n in (1, 3, 5)}
    r3 = np.empty((u.shape[0], nt)); r5 = np.empty_like(r3); c1 = np.empty_like(r3)
    for i in range(nt):
        u2 = u[..., i].astype(float)
        c = {n: np.abs((u2 * e[n]).sum(axis=1)) for n in (1, 3, 5)}
        c1[:, i] = c[1]; r3[:, i] = c[3] / np.maximum(c[1], 1e-30); r5[:, i] = c[5] / np.maximum(c[1], 1e-30)
    return r3, r5, c1


# ---------------------------------------------------------------------
# weak columns
# ---------------------------------------------------------------------

def columns(P, r3, y, t, Lsp, args, Afar):
    dy = float(y[1] - y[0]); dt = float(np.median(np.diff(t)))
    fy, fy1, fy2, fy3, my = kernel_d3(args.ry * Lsp, dy, args.poly_p)
    ht, ht1, _, _, mt = kernel_d3(args.t_half * dt, dt, args.poly_p)
    fy, fy1, fy2, fy3 = fy * dy, fy1 * dy, fy2 * dy, fy3 * dy
    ht, ht1 = ht * dt, ht1 * dt
    K = lambda F, a, b=None: corr_yt(F, a, ht if b is None else b)
    Vx = P["Vx"]; w = P["w"]; D = P["D"]; Ad = P["A"] - Afar; tht = P["tht"]; ky = P["ky"]
    dVy = np.gradient(Vx, dy, axis=0)
    C = {"V": K(Vx, fy), "V w": K(Vx * w, fy), "V^2": K(Vx ** 2, fy), "Lap V": K(Vx, fy2), "d_y V": -K(Vx, fy1),
         "d_y(V k_y)": -K(Vx * ky, fy1), "V A_dev": K(Vx * Ad, fy), "V theta_t": K(Vx * tht, fy), "V div k": K(Vx * D, fy),
         "k_y d_y V": K(ky * dVy, fy)}
    C.update({"r3": K(r3, fy), "r3 w": K(r3 * w, fy), "r3^3": K(r3 ** 3, fy), "Lap r3": K(r3, fy2), "d_y r3": -K(r3, fy1),
              "r3 A_dev": K(r3 * Ad, fy), "r3 theta_t": K(r3 * tht, fy), "r3 div k": K(r3 * D, fy)})
    # the J_t targets and the fluxes (all with the null removed where available)
    FJx = P["FJ"] - P["FJ0"] if "FJ0" in P else P["FJ"]
    Fjtx = P["Fjt"] - P["Fjt0"] if "Fjt0" in P else P["Fjt"]
    T = {"V_t": -K(Vx, fy, ht1), "r3_t": -K(r3, fy, ht1),
         "J_t": -K(FJx, fy1, ht1),                                        # <J> = -d_y FJ, weak K(FJ, fy1); d_t by parts
         "Jcf_t": 0.5 * K(Vx, fy2, ht1)}                                  # -1/2 d_yy V, d_t by parts: -(-1/2) K(V, fy2, ht1)
    C["flux theta_t"] = -K(Fjtx, fy1)                                     # d_y <k_xx theta_ty - k_xy theta_tx>, weak
    C["flux S"] = -K(P["FS"] - P.get("FS0", 0.0), fy1)
    C["flux B"] = -K(P["FB"] - P.get("FB0", 0.0), fy1)
    C["flux A"] = -K(P["FA"] - P.get("FA0", 0.0), fy1)
    # chi_J,t = (d_yy - m^2)^{-1} J_t: apply the inverse to the pointwise J_t profile then the bump
    Jt_point = np.gradient(-np.gradient(FJx, dy, axis=0), dt, axis=1)
    T["chiJ_t"] = K(screened_1d(Jt_point, dy, args.chi_m), fy)
    C["chi flux theta_t"] = K(screened_1d(np.gradient(Fjtx, dy, axis=0), dy, args.chi_m), fy)
    return C, T, (my, mt)


def centres(y, t, Lsp, args, m):
    my, mt = m
    iy_ok = (np.abs(y) < args.band_to * Lsp) & (np.arange(len(y)) >= my) & (np.arange(len(y)) < len(y) - my)
    it_ok = (np.arange(len(t)) >= mt) & (np.arange(len(t)) < len(t) - mt)
    IY, IT = np.meshgrid(np.where(iy_ok)[0], np.where(it_ok)[0], indexing="ij")
    return IY.ravel(), IT.ravel()


# ---------------------------------------------------------------------
# fits of one window
# ---------------------------------------------------------------------

def fit_window(C, T, idx, t, rng, args, lines, truth=None, control=None, live=(True, True)):
    iy, it = idx
    cols = {k: v[iy, it] for k, v in C.items()}
    tg = {k: v[iy, it] for k, v in T.items()}
    ts = t[it]
    xs = rng.random(len(ts))
    fits = dict(n=int(len(ts)))
    if len(ts) < 30:
        lines.append("  too few centres (%d); skipped" % len(ts)); fits["skipped"] = True; return fits
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
    fits["rms"] = {k: rms(v) for k, v in tg.items()}
    fits["rms"].update({"V": rms(cols["V"]), "r3": rms(cols["r3"])})

    def free(names, key, tag, bb, tr=None):
        Th, bv, xv = assemble(cols, names, bb, xs)
        r = fit_pareto(names, Th, bv, xv, rng, args.rel_tol, args.min_contrib, abs_tol=args.abs_tol)
        if not r.get("skipped"):
            r["collinearity"] = collinearity(names, Th)
        fits[key] = r
        lines.append("  %-40s %s" % (tag + ":", V19.fmt(r, tr)))

    def forced(names, want, key, tag, bb):
        Th, bv, xv = assemble(cols, names, bb, xs)
        r = forced_fit(names, Th, bv, xv, want)
        r["collinearity"] = collinearity(want, Th[:, [names.index(n) for n in want]])
        Th2, bv2, tv2 = assemble(cols, names, bb, ts)
        r["r2_time_split"] = forced_fit(names, Th2, bv2, tv2, want)["r2_holdout"]
        fits[key] = r
        lines.append("  %-40s %s   holdout R2 %.4f (n %d; earlier -> later R2 %.3f)  [cond %.3g]" % (
            tag + ":", cstr(r["coefs"]), r["r2_holdout"], r["n"], r["r2_time_split"], r["collinearity"]["cond"]))

    # (1) the conservation identity
    a, c = tg["J_t"], cols["flux theta_t"]
    ok = rms(c) > 1e-12
    if ok:
        coef = float(np.dot(c, a) / max(np.dot(c, c), 1e-300))
        idn = dict(rel=float(np.sqrt(np.mean((a - c) ** 2)) / max(rms(a), 1e-300)), corr=float(np.corrcoef(a, c)[0, 1]) if np.std(a) > 0 and np.std(c) > 0 else np.nan,
                   coef=coef, r2=float(r_squared(a, coef * c)), cf_rel=float(np.sqrt(np.mean((tg["Jcf_t"] - a) ** 2)) / max(rms(a), 1e-300)))
        fits["identity"] = idn
        lines.append("  (1) identity <J_t> = d_y<k_xx theta_ty - k_xy theta_tx>: rel diff %.3f, corr %.3f, coefficient %.3f (R2 %.3f); curl-free part -1/2 d_yy V_t against <J_t>: rel diff %.3f; rms <J_t> %.3e" % (
            idn["rel"], idn["corr"], idn["coef"], idn["r2"], idn["cf_rel"], rms(a)))
    # (2) the growth laws
    if not live[0]:
        lines.append("  (2) V_excess never left its far-field level on this movie (an x-invariant single-mode field): no V_t fit")
    else:
        lines.append("  (2) V_t laws; rms V_t %.3e, rms V %.3e:" % (fits["rms"]["V_t"], fits["rms"]["V"]))
        free(V_LIB, "free_V", "free selection", tg["V_t"], truth)
        forced(V_LIB, ["V", "Lap V"], "forced_V_lin", "forced {V, Lap V}", tg["V_t"])
        forced(V_LIB, ["V", "V^2", "Lap V"], "forced_V_sat", "forced {V, V^2, Lap V}", tg["V_t"])
        forced(V_LIB, ["V", "V w", "Lap V"], "forced_V_w", "forced {V, V w, Lap V}", tg["V_t"])
        forced(V_LIB, ["d_y V"], "forced_V_transport", "forced {d_y V} (control_row: -v)", tg["V_t"])
        forced(V_LIB, ["V", "Lap V", "d_y V"], "forced_V_lin_tr", "forced {V, Lap V, d_y V}", tg["V_t"])
    if not live[1]:
        lines.append("  (2) r3 never left its far-field level on this movie: no r3_t fit")
    else:
        lines.append("  (2) r3_t laws; rms r3_t %.3e, rms r3 %.3e:" % (fits["rms"]["r3_t"], fits["rms"]["r3"]))
        free(R_LIB, "free_r3", "free selection", tg["r3_t"])
        forced(R_LIB, ["r3", "Lap r3"], "forced_r3_lin", "forced {r3, Lap r3}", tg["r3_t"])
        forced(R_LIB, ["r3", "r3^3", "Lap r3"], "forced_r3_sat", "forced {r3, r3^3, Lap r3}", tg["r3_t"])
        forced(R_LIB, ["r3", "r3 w", "Lap r3"], "forced_r3_w", "forced {r3, r3 w, Lap r3}", tg["r3_t"])
    # (3) the rate law through J
    if ok and rms(cols["flux S"]) > 1e-12:
        lines.append("  (3) the rate law seen through J: <J_t> on the fluxes of S and B (RCN: 2, -1), and with the amplitude flux:")
        forced(["flux S", "flux B", "flux A", "flux theta_t"], ["flux S", "flux B"], "throughJ_SB", "forced {flux S, flux B}", tg["J_t"])
        forced(["flux S", "flux B", "flux A", "flux theta_t"], ["flux S", "flux B", "flux A"], "throughJ_SBA", "forced {flux S, flux B, flux A}", tg["J_t"])
        forced(["flux S", "flux B", "flux A", "flux theta_t"], ["flux theta_t"], "throughJ_identity", "forced {flux theta_t} (the identity, 1)", tg["J_t"])
        # chi_J,t
        a2, c2 = tg["chiJ_t"], cols["chi flux theta_t"]
        if rms(c2) > 1e-14:
            coef2 = float(np.dot(c2, a2) / max(np.dot(c2, c2), 1e-300))
            fits["chi_identity"] = dict(coef=coef2, r2=float(r_squared(a2, coef2 * c2)))
            lines.append("  chi_J,t = (d_yy - m^2)^{-1} <J_t> against the same inverse of the flux: coefficient %.3f (R2 %.3f)" % (coef2, fits["chi_identity"]["r2"]))
    return fits


# ---------------------------------------------------------------------
# growth rates
# ---------------------------------------------------------------------

def growth(series, t, ev, args, lines, label, far_level):
    """log-slope of a line series over the early window: from the first frame above growth_start x far level
    to min(t_circ, the time it first reaches half its maximum)"""
    s = np.maximum(series, 1e-30)
    above = np.where(s > args.growth_start * max(far_level, 1e-30))[0]
    out = dict(gamma=np.nan, t_a=np.nan, t_b=np.nan, n=0, r2=np.nan)
    if len(above) == 0:
        lines.append("  growth of %s: never above %.0f x the far-field level (%.2e)" % (label, args.growth_start, far_level)); return out
    ia = above[0]
    tb = ev.get("t_circ", np.nan)
    ihalf = np.where(s >= 0.5 * s.max())[0]
    t_half = t[ihalf[0]] if len(ihalf) else t[-1]
    tb = min(tb if np.isfinite(tb) else t[-1], t_half)
    sel = (t >= t[ia]) & (t <= tb)
    flag = ""
    if sel.sum() >= 3:
        p = np.polyfit(t[sel], np.log(s[sel]), 1)
        pred = np.polyval(p, t[sel])
        out = dict(gamma=float(p[0]), t_a=float(t[ia]), t_b=float(tb), n=int(sel.sum()), r2=float(r_squared(np.log(s[sel]), pred)))
    elif ia + 1 < len(t):
        # the rise is faster than the cadence resolves: the slope between the first frame above the floor and the next, flagged
        j = max(ia, 1)
        out = dict(gamma=float((np.log(s[j]) - np.log(max(s[j - 1], far_level))) / (t[j] - t[j - 1])), t_a=float(t[j - 1]), t_b=float(t[j]), n=2, r2=np.nan)
        flag = "   FLAG: the rise reaches half its maximum within one or two frames; this cadence does not resolve it (a lower bound from the first step above the floor)"
    out["flag"] = flag != ""
    lines.append("  growth of %s on the line: log-slope %.3f per time unit over t %.3f..%.3f (%d frames, R2 of the exponential %.3f); far-field level %.2e, end level %.2e%s" % (
        label, out["gamma"], out["t_a"], out["t_b"], out["n"], out["r2"], far_level, s[-1], flag))
    return out


# ---------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------

def control_growth(args):
    mu = args.mu; k1 = float(np.sqrt(1 - mu ** 2)); Lsp = np.pi / k1
    b = 1.3
    y = np.linspace(-30.0, 30.0, 300); dy = y[1] - y[0]
    nx = int(round(8 * Lsp / dy)); dx = 8 * Lsp / nx
    x = (np.arange(nx) - nx // 2) * dx
    tt = np.linspace(0.0, 2.0, 41)
    phi = np.log(np.cosh(b * mu * y)) / b
    g = np.exp(-0.5 * (y / (args.g_L * Lsp)) ** 2)
    u = np.empty((len(y), nx, len(tt)), np.float32)
    for i, ti in enumerate(tt):
        eps = args.eps0 * np.exp(args.gamma0 * ti)
        u[..., i] = (np.cos(k1 * x[None, :] + phi[:, None]) + eps * g[:, None] * np.cos(3 * k1 * x[None, :] + 3 * phi[:, None])).astype(np.float32)
    md = dict(mu=mu, Rscale=0.5, control="growth", gb_row_in_crop=int(np.argmin(np.abs(y))))
    return u, x, y, tt, md, "control_growth_mu%.2f_g%.2f" % (mu, args.gamma0), {"gamma": args.gamma0}, None


# ---------------------------------------------------------------------
# one movie
# ---------------------------------------------------------------------

def windows_for(t, ev, args):
    dt = float(np.median(np.diff(t)))
    W = {"whole": (t[0], t[-1])}
    tc, tw = ev.get("t_circ", np.nan), ev.get("t_wound", np.nan)
    if np.isfinite(tc):
        W["pre"] = (t[0], tc - dt)
        W["event"] = (max(t[0], tc - args.event_half), min(t[-1], tc + args.event_half))
    if np.isfinite(tw):
        W["post"] = (tw, t[-1])
    return W


def run_one(u, x, y, t, md, stem, args, rng, out, lines, truth=None, exact=None, route="mg", table=None):
    mu = float(md.get("mu", args.mu)); k1 = float(np.sqrt(1 - mu ** 2)); Lsp = np.pi / k1
    jr = int(md.get("gb_row_in_crop", int(np.argmin(np.abs(y))))); yy = y - y[jr]
    tag = stem + ("" if route == "mg" else "_" + route)
    fo = out / tag; fo.mkdir(parents=True, exist_ok=True)
    rep = ["[%s] mu %.3f, L %.3f, %d frames (t %.3f..%.3f, dt %.4g), route %s" % (stem, mu, Lsp, len(t), t[0], t[-1], float(np.median(np.diff(t))), route)]
    picks = sorted(set([0, len(t) // 2, len(t) - 1]))
    res = [shift_flip_residual_interior(u[..., i].astype(float), x, Lsp) for i in picks]
    rep.append("  shift-flip residual (interior spline shift) on 3 frames: max %.2e%s" % (max(res), "" if max(res) < args.sym_tol else "   FLAG: the row average is not exact"))
    wx, N, leak = hann_window(x, Lsp, args.avg_frac)
    rep.append("  x-average: Hann over N = %d periods; leakage at 2 pi/L %.1e" % (N, leak[0]))
    dx = float(x[1] - x[0]); Ly = len(y) * float(y[1] - y[0]); ymid = 0.5 * (y[0] + y[-1])
    fl = min(args.far_from * Lsp, 0.3 * Ly); inside = np.abs(y - ymid) < 0.4 * Ly
    far = (np.abs(yy) > fl) & inside
    cache = fo / ("profiles_jt_%s.npz" % tag)
    if cache.exists() and int(args.use_cache):
        d = np.load(cache); P = {k: d[k] for k in d.files}
        rep.append("  profiles read from the cache")
    else:
        t0_ = time.time()
        A_floor = 0.25 * (float(np.median(np.abs(u[far, :, 0]))) * np.pi / 2 if far.any() else 0.5)
        frames = exact if exact is not None else mg_frames(u, x, y, t, args, rep, y[jr], Lsp)
        P = profiles(frames, x, yy, t, wx, A_floor)
        if exact is None and int(args.null):
            frames0 = mg_frames(u, x, y, t, args, rep, y[jr], Lsp, transform=lambda F: xinv_part(F, x, k1, wx))
            P0 = profiles(frames0, x, yy, t, wx, A_floor)
            for k in ("kx", "kx2", "FJ", "Fjt", "FS", "FB", "FA"):
                P[k + "0"] = P0[k]
        np.savez_compressed(cache, **P)
        rep.append("  profiles computed (%s%s) in %.1f s and cached" % ("exact fields" if exact is not None else "mg", "" if exact is not None or not int(args.null) else " + matched null", time.time() - t0_))
    P["V"] = P["kx2"] - P["kx"] ** 2
    P["Vx"] = P["V"] - (P["kx20"] - P["kx0"] ** 2) if "kx20" in P else P["V"].copy()
    r3, r5, c1 = harmonics(u, x, k1, wx)
    Afar = float(np.median(P["A"][far])) if far.any() else float(np.median(P["A"]))
    ev = read_events(stem, args)
    if ev:
        rep.append("  events: %s" % ", ".join("%s %.3f" % (k, v) for k, v in ev.items() if isinstance(v, float)))
    # series on the line
    line = np.abs(yy) < args.line_halfwidth * Lsp
    r3_line = r3[line].max(axis=0); r3_far = float(np.median(r3[far])) if far.any() else float(np.median(r3[:, 0]))
    sV_line = np.sqrt(np.maximum(P["Vx"][line], 0)).max(axis=0) / k1
    sV_far = float(np.median(np.sqrt(np.maximum(P["Vx"][far], 0)) / k1)) if far.any() else 1e-6
    gr = growth(r3_line, t, ev, args, rep, "r3 = |c3|/|c1| (no estimator)", r3_far)
    gV = growth(sV_line, t, ev, args, rep, "sqrt(V_excess)/k1 (mg, matched null)", max(sV_far, 1e-4))
    dln = np.gradient(np.log(np.maximum(r3_line, 1e-30)), t)
    rows = [dict(t=float(t[i]), r3_line=float(r3_line[i]), dln_r3_dt=float(dln[i]), sV_line=float(sV_line[i]), r5_line=float(r5[line].max(axis=0)[i])) for i in range(len(t))]
    with open(fo / ("series_%s.csv" % tag), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    if truth and "gamma" in truth:
        rep.append("  planted growth rate %.3f: r3 reads %.3f, sqrt(V) reads %.3f" % (truth["gamma"], gr["gamma"], gV["gamma"]))
    # windows
    W = windows_for(t, ev, args)
    if np.isfinite(gr["t_b"]) and gr["n"] >= 2 * args.t_half + 3:
        W["growth"] = (gr["t_a"], gr["t_b"])
    allfits = {}
    for wname, (ta, tb) in W.items():
        sel = (t >= ta - 1e-9) & (t <= tb + 1e-9); n = int(sel.sum())
        rep.append(" window %s: t %.3f..%.3f, %d frames" % (wname, ta, tb, n))
        if n < 2 * args.t_half + 3:
            rep.append("  fewer than %d frames; skipped" % (2 * args.t_half + 3)); continue
        Pw = {k: v[:, sel] for k, v in P.items() if np.ndim(v) == 2}
        C, T, m = columns(Pw, r3[:, sel], yy, t[sel], Lsp, args, Afar)
        idx = centres(yy, t[sel], Lsp, args, m)
        rep.append("  weak form: r_y %.2g L, t_half %d, %d centres on |y| < %.2g L" % (args.ry, args.t_half, len(idx[0]), args.band_to))
        fits = fit_window(C, T, idx, t[sel], rng, args, rep, truth=None, live=(sV_line.max() > args.growth_start * max(sV_far, 1e-4), r3_line.max() > args.growth_start * r3_far))
        allfits[wname] = fits
        if table is not None and not fits.get("skipped"):
            g = lambda key, nm: (fits.get(key) or {}).get("coefs", {}).get(nm, np.nan)
            r2 = lambda key: (fits.get(key) or {}).get("r2_holdout", np.nan)
            idn = fits.get("identity", {})
            table.append(dict(file=tag, mu=mu, window=wname, t0=float(ta), t1=float(tb), n_frames=n,
                              gamma_r3=gr["gamma"], gamma_r3_r2=gr["r2"], gamma_r3_unresolved=int(gr.get("flag", False)), gamma_sV=gV["gamma"], t_a=gr["t_a"], t_b=gr["t_b"],
                              r3_end=float(r3_line[-1]), r3_far=r3_far,
                              id_coef=idn.get("coef", np.nan), id_corr=idn.get("corr", np.nan), id_rel=idn.get("rel", np.nan),
                              V_a=g("forced_V_lin", "V"), V_c=g("forced_V_lin", "Lap V"), V_r2=r2("forced_V_lin"),
                              Vsat_a=g("forced_V_sat", "V"), Vsat_b=g("forced_V_sat", "V^2"), Vsat_r2=r2("forced_V_sat"),
                              r3_a=g("forced_r3_lin", "r3"), r3_c=g("forced_r3_lin", "Lap r3"), r3_r2=r2("forced_r3_lin"),
                              r3sat_a=g("forced_r3_sat", "r3"), r3sat_b=g("forced_r3_sat", "r3^3"), r3sat_r2=r2("forced_r3_sat"),
                              freeV="{%s}" % ", ".join((fits.get("free_V") or {}).get("coefs", {}).keys()),
                              freer3="{%s}" % ", ".join((fits.get("free_r3") or {}).get("coefs", {}).keys()),
                              J_a=g("throughJ_SB", "flux S"), J_b=g("throughJ_SB", "flux B"), J_r2=r2("throughJ_SB"),
                              J_id_r2=r2("throughJ_identity"), J_id_coef=g("throughJ_identity", "flux theta_t"),
                              chi_coef=(fits.get("chi_identity") or {}).get("coef", np.nan)))
    try:
        fig_movie(P, r3, yy, t, Lsp, k1, r3_line, sV_line, dln, ev, gr, tag, mu, fo / ("fig_jt_%s.png" % tag))
    except Exception as exc:
        rep.append("  figure failed: %r" % exc)
    with open(fo / "fits.json", "w") as f:
        json.dump(allfits, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    (fo / "report.txt").write_text("\n".join(rep) + "\n")
    lines.extend(rep)


def fig_movie(P, r3, y, t, Lsp, k1, r3_line, sV_line, dln, ev, gr, stem, mu, path):
    band = np.abs(y) < 3 * Lsp; yb = y[band] / Lsp
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    im = ax[0].pcolormesh(t, yb, np.log10(np.maximum(r3[band], 1e-12)), shading="auto", cmap="magma"); fig.colorbar(im, ax=ax[0]); ax[0].set_title("log10 r3 = |c3|/|c1| (no estimator)", fontsize=9)
    Jx = -np.gradient(P["FJ"] - P.get("FJ0", 0.0), y, axis=0) * Lsp ** 2
    lim = max(np.nanpercentile(np.abs(Jx[band]), 99.5), 1e-9)
    im = ax[1].pcolormesh(t, yb, Jx[band], shading="auto", cmap="RdBu_r", vmin=-lim, vmax=lim); fig.colorbar(im, ax=ax[1]); ax[1].set_title("<J> L^2 (matched null)", fontsize=9)
    for a_ in ax[:2]:
        a_.set_xlabel("t"); a_.set_ylabel("y / L")
    ax[2].semilogy(t, np.maximum(r3_line, 1e-14), color=C1, label="r3 on the line")
    ax[2].semilogy(t, np.maximum(sV_line, 1e-14), color=C2, label="sqrt(V_excess)/k1 on the line")
    if np.isfinite(gr["t_a"]):
        ax[2].axvspan(gr["t_a"], gr["t_b"], color="0.9")
    ax[2].set_xlabel("t"); ax[2].legend(fontsize=7); ax[2].set_title("growth window shaded; log-slope of r3 %.2f" % gr["gamma"], fontsize=9)
    ax[3].plot(t, dln, color=C1); ax[3].axhline(0, color="0.6", lw=0.6); ax[3].set_xlabel("t"); ax[3].set_title("d ln r3 / dt on the line", fontsize=9)
    ax[3].set_ylim(-1, max(5, np.nanpercentile(dln, 98)))
    for a_ in ax:
        for k_, c_ in (("t_charge", C3), ("t_circ", C4), ("t_wound", "k")):
            v = ev.get(k_, np.nan)
            if np.isfinite(v):
                a_.axvline(v, color=c_, lw=0.8, ls="--")
    fig.suptitle("%s (mu %.3f)" % (stem, mu), fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fig_set(table, path):
    rows = [r for r in table if r["window"] == "whole" and np.isfinite(r.get("mu", np.nan))]
    if len(rows) < 2:
        return
    rows = sorted(rows, key=lambda r: r["mu"]); mu = [r["mu"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(mu, [r["gamma_r3"] for r in rows], "o-", color=C1, label="log-slope of r3 (no estimator)")
    ax[0].plot(mu, [r["gamma_sV"] for r in rows], "s--", color=C2, label="log-slope of sqrt(V_excess)")
    ax[0].axhline(0, color="0.6", lw=0.6); ax[0].set_xlabel("mu"); ax[0].set_title("early growth rate of the x-structure on the line", fontsize=9); ax[0].legend(fontsize=7)
    ax[1].plot(mu, [r["J_a"] for r in rows], "o-", color=C1, label="a (S) through J"); ax[1].plot(mu, [r["J_b"] for r in rows], "s-", color=C2, label="b (B) through J")
    ax[1].axhline(0, color="0.6", lw=0.6); ax[1].set_xlabel("mu"); ax[1].set_title("theta_t = a S + b B as J sees it (RCN 2, -1)", fontsize=9); ax[1].legend(fontsize=7)
    ax[2].plot(mu, [r["id_coef"] for r in rows], "o-", color=C1, label="identity coefficient (1)"); ax[2].plot(mu, [r["id_corr"] for r in rows], "s--", color=C2, label="identity correlation")
    ax[2].axhline(1, color="0.6", lw=0.6); ax[2].set_xlabel("mu"); ax[2].set_title("<J_t> against the theta_t flux (whole run)", fontsize=9); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main(args):
    rng = np.random.default_rng(args.seed)
    out = Path(args.outdir)
    if not out.is_absolute():
        out = HERE / out
    out.mkdir(parents=True, exist_ok=True)
    lines = ["knee_jt_laws v1.0; r_y %.2g L, t_half %d, band_to %.2g L, Hann over integer periods in %.2g; growth window from %.0f x far level to min(t_circ, half max); chi_m %.2g" % (
        args.ry, args.t_half, args.band_to, args.avg_frac, args.growth_start, args.chi_m)]
    table = []
    if args.mode in ("control_rcn", "control_row", "control_growth"):
        fn = {"control_rcn": _control_rcn, "control_row": _control_row, "control_growth": control_growth}[args.mode]
        u, x, y, t, md, stem, truth, ex = fn(args)
        for route in [r.strip() for r in args.routes.split(",")]:
            if route == "exact" and ex is None:
                continue
            run_one(u, x, y, t, md, stem, args, rng, out, lines, truth=truth, exact=(ex if route == "exact" else None), route=route, table=table)
    else:
        for p in [s for s in args.npz.split(",") if s.strip()]:
            D = load_npz(p)
            run_one(D["u"], D["x"], D["y"], D["t"], D["md"], Path(p).stem, args, rng, out, lines, table=table)
    if table:
        keys = list(table[0].keys())
        with open(out / "jt_table.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(table)
        md_ = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
        for r in table:
            md_.append("| " + " | ".join(("%.4g" % v) if isinstance(v, float) else str(v) for v in r.values()) + " |")
        (out / "jt_table.md").write_text("\n".join(md_) + "\n")
        fig_set(table, out / "fig_growth_vs_mu.png")
    (out / "log.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


class _Args:
    mode = "data"             # data | control_rcn | control_row | control_growth
    npz = ""
    outdir = "results/knee_jt_laws"
    routes = "exact,mg"       # controls
    events = ""
    event_half = 1.0
    mu = 0.85
    R = 0.5
    sigma_dir = 0.4
    taper_frac = 0.06
    avg_frac = 0.8
    sym_tol = 1e-4
    use_cache = 1
    null = 1
    null_harm = "1"
    band_to = 2.0
    far_from = 4.0
    line_halfwidth = 0.5
    ry = 1.5
    t_half = 3
    poly_p = 6
    chi_m = 1.0
    growth_start = 3.0        # growth window starts where the line series exceeds this times its far-field level
    gamma0 = 2.0              # control_growth: planted rate
    eps0 = 1e-4               # control_growth: planted amplitude of the third harmonic at t = 0
    g_L = 0.5                 # control_growth: Gaussian width of the planted harmonic about the line, L
    row_v = 0.5
    row_ell = 0.3
    row_eps = 0.2
    rel_tol = 0.01
    abs_tol = 0.0
    min_contrib = 0.05
    seed = 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in vars(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    main(p.parse_args())
