"""knee_demod_laws.py  v1.1  (13 Oct 2026)   workstreams G17, G22, G23 of the research thread

Estimator-free rate laws on the x-invariant stage of the knee. Before the
first circulation the zigzag frames are x-invariant, so the projection of
each row onto exp(i k1 x) (knee_row_demod_survey.row_demod, Hann weight,
k1 fitted per file) gives the amplitude A(y, t) and the phase phi(y, t)
exactly, with no estimator. From phi alone: k2 = phi_y, w = k1^2 + k2^2 - 1,
div k = k2_y, theta_t = phi_t; for an x-invariant field J = 0 identically.

Fits (one-dimensional weak form in (y, t): polynomial bump (1 - s^2)^p of
half-width r_y across the row and t_half frames in time; the derivative
columns and the targets are integrated by parts, so only phi, k2, w and A
are differentiated once pointwise):
  phase:  free selection (wsindy_theta_t_v19.fit_pareto, holdout = a
          random half of the query centres; the forced fits also report
          the R2 of a fit on the earlier half of the window tested on the
          later half, which asks the law to extrapolate the decay) on the library
          {div k, w div k, k.grad w, Lap div k, w, Lap w, (div k)^2,
          w^2 div k, w k.grad w, A_dev, A_dev div k, k.grad A, w k.grad lnA,
          w theta_t}; forced RCN {w div k, k.grad w, Lap div k}; forced
          {w div k, k.grad w, w} (the three-term law of the 12 Oct run);
          forced {w div k, k.grad w, w, w theta_t}; tied {S, Lap div k};
          each forced fit also at r_y in --ry_sweep.
  amplitude: free selection and the forced slaved law on {A, w^2 A, A^3,
          Lap A, k.grad A, w A, A div k} for A_t; the single-mode slaving
          A^2 = (4/3)(R - w^2) predicts (R, -1, -3/4) on the first three.
  slaving: per frame, A^2 against (4/3)(R - w^2) on the band (R from the
          metadata, Rscale): holdout-free R2 and the median ratio.

Frames used: from t = 0 to --t_end, or to the frame before the first
partial circulation of the file in --events (events.csv of
knee_amplitude_sweep), or to the first frame whose median row
x-invariance on the band falls below --xinv_gate. The gate alone is not a
defect detector: a row of dislocations at the zigzag spacing keeps the
k1 mode (x-invariance 0.999 on the stand-in after winding), so the
circulation record is what ends the window.

Controls (--mode control_rcn, control_amp):
  control_rcn  the RCN knee relaxation of wsindy_theta_t v1.9 (alpha =
               eta_b = tau = 1/3, beta0 1.8, mu 0.85) as u = cos(k1 x +
               Phi(y, t)); the demodulated phase must return (2, 2, -1) and
               must not return the three-term law.
  control_amp  a static knee (k2 = mu tanh(b mu y), theta_t = 0) whose
               amplitude obeys A_t = (R - w^2) A - (3/4) A^3 from A = 0.3
               (the slaved single-mode law); the A_t block must return
               (R, -1, -0.75) and the phase block must return nothing.

Outputs under --outdir: per file <stem>/report.txt, fits.json,
fig_profiles_<stem>.png (A, k2, w, phi_t at four times), fig_slaving_<stem>.png,
fig_ry_<stem>.png (forced coefficients against r_y); for the set
demod_laws_table.csv/.md and fig_laws_vs_mu.png.

Changelog
  v1.1  (G23, from Alan's note of 13 Oct) relation A with the higher-order
        terms of G^2: per frame (y-only weak form, r_y = ry_A), div k on
        {w}, {w, w^3}, {w, w|w|} and the full form w sqrt(1 - w^2/(2R))
        with R fixed (one coefficient; the truncated w^3 fit reads the
        planted 1/(4R) as 0.58 on the control because the expansion is
        poor at |w| = 0.7, so the full form is the test of record). R2 does
        not separate the forms on smooth profiles (1 - R2 below 1e-4 for
        both on the controls); the test is that the right form returns the
        same beta within split_A L of the line and beyond it.
        For the SH single-mode stretching
        B = w - w^3/R, G^2 is proportional to w^2 (1 - w^2/(2R)), so
        div k = -beta w + gamma w^3 with gamma/beta = 1/(4R) (B'' = 0 at
        kb, so the first correction is cubic). The rate law with the full
        stretching: tied {S_full, Lap div k}, S_full = div(k (w - w^3/R))
        with R from the metadata, and the RCN triple plus {w^3 div k,
        w^2 k.grad w}. (G22) the amplitude coupling in the phase law:
        RCN + w k.grad lnA, tied {S_A, Lap div k} with S_A = div(A^2 w k)/A^2,
        tied {S, Lap div k, w k.grad lnA}. Files with too few x-invariant
        frames (the shallow knees: two frames) get relation A only.
        Controls --mode control_selfdual (a static knee solving the full
        sqrt(G^2) relation, and the plain one) and --mode control_rcn4 (the
        RCN relaxation with the quartic stretching; truth 2 S_full - Lap div k).
  v1.0  first version; validation in the sandbox on both controls and on
        the stand-in SH movie (see the explanation file of 13 Oct).
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT / "src", ROOT / "src" / "op_extract", ROOT / "src" / "utils", HERE):
    sys.path.insert(0, str(_p))
import wsindy_theta_t_v19 as V                                           # noqa: E402
from wsindy_theta_t_v19 import (load_npz, kernel_d3, assemble, fit_pareto, forced_fit,   # noqa: E402
                                collinearity, cstr, r_squared)
from knee_row_demod_survey import fit_k1, row_demod                      # noqa: E402
import matplotlib                                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MUTED = "#77756f"
SP = ["w div k", "k.grad w"]
RCN = ["w div k", "k.grad w", "Lap div k"]
PH_LIB = ["div k", "w div k", "k.grad w", "Lap div k", "w", "Lap w", "(div k)^2", "w^2 div k", "w k.grad w",
          "A_dev", "A_dev div k", "k.grad A", "w k.grad lnA", "w theta_t"]
A_LIB = ["A", "w^2 A", "A^3", "Lap A", "k.grad A", "w A", "A div k"]
TRUTH_RCN = {"w div k": 2.0, "k.grad w": 2.0, "Lap div k": -1.0}


# ---------------------------------------------------------------------
# demodulation of a movie
# ---------------------------------------------------------------------

def demodulate(u, x, y, t, mu, args, lines):
    """A(y, t), phi(y, t), xinv(y, t), k1; phi oriented so k2 > 0 above the row"""
    k1 = fit_k1(u[..., 0].astype(float), x, np.sqrt(1 - mu ** 2))
    nt = u.shape[2]
    A = np.empty((len(y), nt)); ph = np.empty_like(A); xi = np.empty_like(A)
    for i in range(nt):
        c, xinv = row_demod(u[..., i].astype(float), x, k1)
        A[:, i] = np.abs(c); ph[:, i] = np.unwrap(np.angle(c)); xi[:, i] = xinv
    jr = int(np.argmin(np.abs(y)))
    up = (y > 0) & (y < 3.0)
    k2 = np.gradient(ph, y, axis=0)
    if np.median(k2[up, 0]) < 0:
        ph = -ph
    # gauge: phi pinned at the far end (theta_t = 0 in the far field), per frame
    ph = ph - ph[-1:, :]
    lines.append("  k1 fitted %.4f (sqrt(1 - mu^2) %.4f); phase pinned at y = %.1f" % (k1, np.sqrt(1 - mu ** 2), y[-1]))
    return A, ph, xi, k1, jr


def gate_frames(xi, y, t, Lsp, args, lines, t_end):
    band = np.abs(y) < args.band_to * Lsp
    med = np.median(xi[band], axis=0)
    ok = med >= args.xinv_gate
    ng = int(np.argmax(~ok)) if (~ok).any() else len(t)
    n = ng
    if t_end > 0:
        n = min(n, int(np.sum(t <= t_end + 1e-9)))
    lines.append("  x-invariance on the band: %.4f at t = 0, %.4f at the last used frame; %d of %d frames used (t <= %.3f); gate %.3f %s; t_end %s" % (
        med[0], med[max(n - 1, 0)], n, len(t), t[max(n - 1, 0)], args.xinv_gate,
        ("first fails at t = %.3f" % t[ng]) if ng < len(t) else "never fails (it does not see dislocations whose row keeps the k1 mode; rely on t_end)",
        ("%.3f (from %s)" % (t_end, args.t_end_source)) if t_end > 0 else "not given"))
    return n


# ---------------------------------------------------------------------
# one-dimensional weak columns
# ---------------------------------------------------------------------

def corr_yt(F, fy, ht):
    """Int F(y', t') fy(y' - y) ht(t' - t) dy' dt' on the grid (zero outside)"""
    from scipy.ndimage import correlate1d
    dy_ = 1.0
    G = correlate1d(F, fy, axis=0, mode="constant", cval=0.0)
    return correlate1d(G, ht, axis=1, mode="constant", cval=0.0)


def build_1d(A, ph, y, t, k1, Lsp, R, args, ry_L, lines=None):
    dy = float(y[1] - y[0]); dt = float(np.median(np.diff(t)))
    fy, fy1, fy2, fy3, my = kernel_d3(ry_L * Lsp, dy, args.poly_p)
    ht, ht1, _, _, mt = kernel_d3(args.t_half * dt, dt, args.poly_p)
    fy, fy1, fy2, fy3 = fy * dy, fy1 * dy, fy2 * dy, fy3 * dy
    ht, ht1 = ht * dt, ht1 * dt
    k2 = np.gradient(ph, dy, axis=0)
    w = k1 ** 2 + k2 ** 2 - 1.0
    D = np.gradient(k2, dy, axis=0)
    wy = np.gradient(w, dy, axis=0)
    Ay = np.gradient(A, dy, axis=0)
    far = np.abs(y) > args.far_from * Lsp
    Afar = float(np.median(A[far])) if far.any() else float(np.median(A))
    Ad = A - Afar
    pht = np.gradient(ph, dt, axis=1)
    K = lambda F, a, b: corr_yt(F, a, b)
    C = {}
    C["div k"] = -K(k2, fy1, ht)
    C["w div k"] = K(w * D, fy, ht)
    C["k.grad w"] = K(k2 * wy, fy, ht)
    C["Lap div k"] = -K(k2, fy3, ht)
    C["w"] = K(w, fy, ht)
    C["Lap w"] = K(w, fy2, ht)
    C["(div k)^2"] = K(D ** 2, fy, ht)
    C["w^2 div k"] = K(w ** 2 * D, fy, ht)
    C["w k.grad w"] = K(w * k2 * wy, fy, ht)
    C["A_dev"] = K(Ad, fy, ht)
    C["A_dev div k"] = K(Ad * D, fy, ht)
    C["k.grad A"] = K(k2 * Ay, fy, ht)
    C["w k.grad lnA"] = K(w * k2 * Ay / np.maximum(A, 0.25 * Afar), fy, ht)
    C["w theta_t"] = K(w * pht, fy, ht)
    C["S"] = C["w div k"] + C["k.grad w"]
    # v1.1 (G23, Alan): the full single-mode stretching B(k^2) = w - w^3/R (SH, amplitude slaved), tied, R known
    C["S_full"] = -K(k2 * (w - w ** 3 / R), fy1, ht)
    C["w^3 div k"] = K(w ** 3 * D, fy, ht)
    C["w^2 k.grad w"] = K(w ** 2 * k2 * wy, fy, ht)
    # v1.1 (G22): the amplitude-weighted stretching div(A^2 w k)/A^2 = S + 2 w k.grad lnA
    C["S_A"] = C["S"] + 2.0 * C["w k.grad lnA"]
    T = {"theta_t": -K(ph, fy, ht1), "A_t": -K(A, fy, ht)}
    T["A_t"] = -K(A, fy, ht1)
    Ac = {"A": K(A, fy, ht), "w^2 A": K(w ** 2 * A, fy, ht), "A^3": K(A ** 3, fy, ht), "Lap A": K(A, fy2, ht),
          "k.grad A": C["k.grad A"], "w A": K(w * A, fy, ht), "A div k": K(A * D, fy, ht)}
    # query centres: full support, inside the band
    Y, Tt = np.meshgrid(y, t, indexing="ij")
    ok = (np.abs(Y) < args.band_to * Lsp) & (np.arange(len(y))[:, None] >= my) & (np.arange(len(y))[:, None] < len(y) - my) \
        & (np.arange(len(t))[None, :] >= mt) & (np.arange(len(t))[None, :] < len(t) - mt)
    ok &= (np.arange(len(y))[:, None] % max(1, args.stride_y) == 0)
    idx = np.where(ok)
    cols = {k: v[idx] for k, v in C.items()}
    acols = {k: v[idx] for k, v in Ac.items()}
    tg = {k: v[idx] for k, v in T.items()}
    fields = dict(k2=k2, w=w, D=D, pht=pht, Ad=Ad, Afar=Afar)
    return cols, acols, tg, Tt[idx], Y[idx], fields, (my, mt)


# ---------------------------------------------------------------------
# fits of one file
# ---------------------------------------------------------------------

def relation_A_frame(ph_col, y, k1, Lsp, R, args, ry_L):
    """one frame: weak div k = -Int k2 phi_y against Int w phi, Int w^3 phi,
    Int w|w| phi, on centres |y| < band_to L with full support; forced fits
    {w}, {w, w^3}, {w, w|w|} (least squares, random half holdout)"""
    from scipy.ndimage import correlate1d
    dy = float(y[1] - y[0])
    fy, fy1, _, _, my = kernel_d3(ry_L * Lsp, dy, args.poly_p)
    fy, fy1 = fy * dy, fy1 * dy
    k2 = np.gradient(ph_col, dy)
    w = k1 ** 2 + k2 ** 2 - 1.0
    K = lambda F, f: correlate1d(F, f, mode="constant", cval=0.0)
    tgt = -K(k2, fy1)
    cols = {"w": K(w, fy), "w^3": K(w ** 3, fy), "w|w|": K(w * np.abs(w), fy),
            "w_G": K(w * np.sqrt(np.maximum(1.0 - w ** 2 / (2.0 * R), 0.0)), fy)}
    i = np.arange(len(y))
    ok = (np.abs(y) < args.band_to * Lsp) & (i >= my) & (i < len(y) - my)
    out = {}
    hold = np.random.default_rng(1).random(int(ok.sum())) < 0.5
    b = tgt[ok]
    for key, names in (("A1", ["w"]), ("A3", ["w", "w^3"]), ("A2", ["w", "w|w|"]), ("AG", ["w_G"])):
        Th = np.column_stack([cols[n][ok] for n in names])
        c, *_ = np.linalg.lstsq(Th[~hold], b[~hold], rcond=None)
        out[key] = dict(coefs={n: float(v) for n, v in zip(names, c)}, r2=float(r_squared(b[hold], Th[hold] @ c)))
    # the discriminating test: the right form gives the same beta near the line and away from it
    yo = y[ok]
    inner = np.abs(yo) < args.split_A * Lsp
    for key, n_ in (("A1", "w"), ("AG", "w_G")):
        th = cols[n_][ok]
        bi = float(np.dot(th[inner], b[inner]) / max(np.dot(th[inner], th[inner]), 1e-300))
        bo = float(np.dot(th[~inner], b[~inner]) / max(np.dot(th[~inner], th[~inner]), 1e-300))
        out[key]["beta_inner"], out[key]["beta_outer"] = -bi, -bo
        out[key]["resid"] = float(1.0 - out[key]["r2"])
    cw, c3 = out["A3"]["coefs"]["w"], out["A3"]["coefs"]["w^3"]
    out["gamma_over_beta"] = float(c3 / (-cw)) if cw != 0 else float("nan")
    out["sh_prediction"] = 1.0 / (4.0 * R)
    out["w_line"] = float(w[int(np.argmin(np.abs(y)))])
    return out


def relation_A_block(ph, y, t, k1, Lsp, R, args, lines, which):
    res = {}
    for tag, j in which:
        r = relation_A_frame(ph[:, j], y, k1, Lsp, R, args, args.ry_A)
        res[tag] = dict(t=float(t[j]), **r)
        lines.append("  relation A at %s (t %.2f, w on the line %.3f; r_y %.2g L): div k = %.3f w (R2 %.4f) | %.3f w + %.3f w^3 (R2 %.4f; gamma/beta %.3f, SH single mode 1/(4R) = %.3f) | %.3f w + %.3f w|w| (R2 %.4f)" % (
            tag, t[j], r["w_line"], args.ry_A, r["A1"]["coefs"]["w"], r["A1"]["r2"], r["A3"]["coefs"]["w"], r["A3"]["coefs"]["w^3"], r["A3"]["r2"],
            r["gamma_over_beta"], r["sh_prediction"], r["A2"]["coefs"]["w"], r["A2"]["coefs"]["w|w|"], r["A2"]["r2"]))
        lines.append("     the full form div k = -beta w sqrt(1 - w^2/(2R)) with R %.3g fixed: beta %.3f, 1 - R2 %.2e (against %.2e for -beta w);"
                     " beta within %.2g L of the line and beyond: %.3f and %.3f (full form) against %.3f and %.3f (-beta w)" % (
            R, -r["AG"]["coefs"]["w_G"], r["AG"]["resid"], r["A1"]["resid"], args.split_A,
            r["AG"]["beta_inner"], r["AG"]["beta_outer"], r["A1"]["beta_inner"], r["A1"]["beta_outer"]))
    return res


def fit_file(A, ph, y, t, k1, Lsp, R, mu, args, rng, lines, out, truth=None):
    fits = {}
    cols, acols, tg, ts, ys, fields, (my, mt) = build_1d(A, ph, y, t, k1, Lsp, R, args, args.ry, lines)
    b = tg["theta_t"]
    xs = rng.random(len(b))                                               # random half as holdout (the time split is reported beside it)
    lines.append("  weak form: r_y %.2g L (%d rows), t_half %d frames; %d query centres on |y| < %.2g L, holdout = a random half (time-split R2 beside the forced fits); rms target %.3e" % (
        args.ry, 2 * my + 1, args.t_half, len(b), args.band_to, float(np.sqrt(np.mean(b ** 2)))))
    fits["n_rows"] = int(len(b)); fits["target_rms"] = float(np.sqrt(np.mean(b ** 2)))
    scale = float(np.sqrt(np.mean(cols["w div k"] ** 2))) + 1e-300
    fits["target_over_column"] = fits["target_rms"] / scale
    if fits["target_over_column"] < 1e-3:
        lines.append("  FLAG: the phase target is below 1e-3 of the w div k column (%.2e): the phase is static to round-off here and the phase fits below are noise" % fits["target_over_column"])

    def free(names, key, tag, C, bb):
        Th, bv, xv = assemble(C, names, bb, xs)
        r = fit_pareto(names, Th, bv, xv, rng, args.rel_tol, args.min_contrib, abs_tol=args.abs_tol)
        if not r.get("skipped"):
            r["collinearity"] = collinearity(names, Th)
        fits[key] = r
        lines.append("  %-38s %s" % (tag + ":", V.fmt(r, truth if key == "free_phase" else None)))
        return r

    def forced(names, want, key, tag, C, bb):
        Th, bv, xv = assemble(C, names, bb, xs)
        if len(bv) < 20:
            fits[key] = dict(skipped=True, reason="too few rows"); return None
        r = forced_fit(names, Th, bv, xv, want)
        r["collinearity"] = collinearity(want, Th[:, [names.index(n) for n in want]])
        Th2, bv2, tv2 = assemble(C, names, bb, ts)
        r2t = forced_fit(names, Th2, bv2, tv2, want)                       # train on the earlier half in time, test on the later
        r["r2_time_split"] = r2t["r2_holdout"]; r["coefs_early"] = r2t["coefs"]
        fits[key] = r
        lines.append("  %-38s %s   holdout R2 %.4f (n %d; fitted on the earlier half, later half R2 %.3f)  [cond %.3g, max |r| %.3f]" % (
            tag + ":", cstr(r["coefs"]), r["r2_holdout"], r["n"], r["r2_time_split"], r["collinearity"]["cond"], abs(r["collinearity"]["max_corr"])))
        return r

    lines.append("  phase block (theta_t):")
    free(PH_LIB, "free_phase", "free selection", cols, b)
    forced(PH_LIB, RCN, "forced_rcn", "forced RCN {w div k, k.grad w, Lap div k}", cols, b)
    forced(PH_LIB, SP + ["w"], "forced_SPw", "forced {w div k, k.grad w, w}", cols, b)
    forced(PH_LIB, SP + ["w", "w theta_t"], "forced_SPw_tau", "forced {w div k, k.grad w, w, w theta_t}", cols, b)
    forced(PH_LIB, RCN + ["w"], "forced_rcn_w", "forced RCN + w", cols, b)
    forced(PH_LIB + ["S"], ["S", "Lap div k"], "tied_SB", "tied {S, Lap div k}", cols, b)
    # v1.1 (G23): Alan's full stretching. SH single mode: B = w - w^3/R, so the tied law is S_full with R fixed;
    # the free version admits w^3 div k and w^2 k.grad w beside the RCN pair (SH ratio to the pair: -1/R and -3/R)
    PX = PH_LIB + ["S", "S_full", "w^3 div k", "w^2 k.grad w", "S_A"]
    forced(PX, ["S_full", "Lap div k"], "tied_SfullB", "tied {S_full = div(k(w - w^3/R)), Lap div k}", cols, b)
    forced(PX, RCN + ["w^3 div k", "w^2 k.grad w"], "forced_rcn_cubic", "forced RCN + {w^3 div k, w^2 k.grad w}", cols, b)
    # v1.1 (G22): does the amplitude coupling carry the anti-RCN bending?
    forced(PX, RCN + ["w k.grad lnA"], "forced_rcn_lnA", "forced RCN + w k.grad lnA", cols, b)
    forced(PX, ["S_A", "Lap div k"], "tied_SAB", "tied {S_A = div(A^2 w k)/A^2, Lap div k}", cols, b)
    forced(PX, ["S", "Lap div k", "w k.grad lnA"], "tied_SB_lnA", "tied {S, Lap div k, w k.grad lnA}", cols, b)
    # r_y sweep of the forced laws
    sweep = []
    for ry in [float(v) for v in str(args.ry_sweep).split(",") if v.strip()]:
        c2, _, t2, ts2, _, _, _ = build_1d(A, ph, y, t, k1, Lsp, R, args, ry)
        row = dict(ry=ry, n=int(len(t2["theta_t"])))
        xs2 = rng.random(len(t2["theta_t"]))
        for want, key in ((RCN, "rcn"), (SP + ["w"], "SPw")):
            Th, bv, xv = assemble(c2, PH_LIB, t2["theta_t"], xs2)
            if len(bv) < 20:
                continue
            r = forced_fit(PH_LIB, Th, bv, xv, want)
            row[key] = dict(coefs=r["coefs"], r2=r["r2_holdout"])
        sweep.append(row)
    fits["ry_sweep"] = sweep
    lines.append("  r_y sweep (forced RCN | forced {S pair, w}; coefficients and holdout R2):")
    for row in sweep:
        g = lambda k: ("(%s) %.3f" % (", ".join("%.2f" % v for v in row[k]["coefs"].values()), row[k]["r2"])) if k in row else "n/a"
        lines.append("     r_y %.2f L: %s | %s   (n %d)" % (row["ry"], g("rcn"), g("SPw"), row["n"]))
    # amplitude block
    bA = tg["A_t"]
    lines.append("  amplitude block (A_t); rms target %.3e:" % float(np.sqrt(np.mean(bA ** 2))))
    if float(np.sqrt(np.mean(bA ** 2))) < 1e-3 * float(np.sqrt(np.mean(acols["A"] ** 2))) * float(np.median(np.diff(t))):
        lines.append("  FLAG: the amplitude is static to round-off here; the amplitude fits below are noise")
    free(A_LIB, "free_amp", "free selection", acols, bA)
    forced(A_LIB, ["A", "w^2 A", "A^3"], "forced_slaved", "forced slaved {A, w^2 A, A^3} (single mode: R, -1, -3/4)", acols, bA)
    forced(A_LIB, ["A", "w^2 A", "A^3", "Lap A"], "forced_slaved_lap", "forced {A, w^2 A, A^3, Lap A}", acols, bA)
    # slaving per frame
    w = fields["w"]; band = np.abs(y) < args.band_to * Lsp
    sl = []
    for i in range(len(t)):
        pred = (4.0 / 3.0) * (R - w[band, i] ** 2)
        meas = A[band, i] ** 2
        sl.append(dict(t=float(t[i]), r2=float(r_squared(meas, pred)), ratio=float(np.median(meas / np.maximum(pred, 1e-6))),
                       min_pred=float(pred.min()), min_meas=float(meas.min())))
    fits["slaving"] = sl
    lines.append("  slaving A^2 = (4/3)(R - w^2) on the band (R %.3g): R2 %.3f at t = 0, %.3f at the middle, %.3f at the last used frame; median ratio measured/slaved %.3f, %.3f, %.3f; min slaved %.3f against min measured %.3f at the end" % (
        R, sl[0]["r2"], sl[len(sl) // 2]["r2"], sl[-1]["r2"], sl[0]["ratio"], sl[len(sl) // 2]["ratio"], sl[-1]["ratio"], sl[-1]["min_pred"], sl[-1]["min_meas"]))
    return fits, fields


# ---------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------

def fig_profiles(A, ph, y, t, fields, Lsp, stem, mu, path):
    picks = [0, len(t) // 3, 2 * len(t) // 3, len(t) - 1]
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6))
    for j, i in enumerate(picks):
        c = plt.get_cmap("viridis")(j / 3.0)
        axs[0].plot(y / Lsp, A[:, i], color=c, label="t %.2f" % t[i])
        axs[1].plot(y / Lsp, fields["k2"][:, i], color=c)
        axs[2].plot(y / Lsp, fields["w"][:, i], color=c)
        axs[3].plot(y / Lsp, fields["pht"][:, i], color=c)
    for ax, lab in zip(axs, ("A(y) (row demodulation)", "k2 = phi_y", "w = k1^2 + k2^2 - 1", "theta_t = phi_t")):
        ax.set_title(lab, fontsize=9); ax.set_xlabel("y / L"); ax.set_xlim(-3, 3)
    axs[0].legend(fontsize=7)
    fig.suptitle("%s, mu %.3f: demodulated profiles on the x-invariant stage" % (stem, mu), fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def fig_slaving(A, y, t, fields, Lsp, R, stem, mu, sl, path):
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    picks = [0, len(t) // 3, 2 * len(t) // 3, len(t) - 1]
    for j, i in enumerate(picks):
        c = plt.get_cmap("viridis")(j / 3.0)
        axs[0].plot(y / Lsp, A[:, i] ** 2, color=c, label="measured A^2, t %.2f" % t[i])
        axs[0].plot(y / Lsp, (4.0 / 3.0) * (R - fields["w"][:, i] ** 2), color=c, ls="--", lw=1.0)
    axs[0].set_xlim(-3, 3); axs[0].set_title("A^2 (solid) against the slaved (4/3)(R - w^2) (dashed)", fontsize=9); axs[0].set_xlabel("y / L"); axs[0].legend(fontsize=7)
    axs[1].plot(t, [s["r2"] for s in sl], color=C1); axs[1].set_title("R2 of the slaving on the band against t", fontsize=9); axs[1].set_xlabel("t"); axs[1].set_ylim(-1, 1.05)
    axs[2].plot(t, [s["ratio"] for s in sl], color=C2); axs[2].axhline(1, color=MUTED, lw=0.8); axs[2].set_title("median A^2 / slaved against t", fontsize=9); axs[2].set_xlabel("t")
    fig.suptitle("%s, mu %.3f: is the amplitude slaved to the wavenumber?" % (stem, mu), fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def fig_ry(sweep, stem, mu, path, truth=None):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    ry = [r["ry"] for r in sweep]
    for key, ax, names in (("rcn", axs[0], RCN), ("SPw", axs[1], SP + ["w"])):
        for j, nm in enumerate(names):
            v = [r[key]["coefs"][nm] if key in r else np.nan for r in sweep]
            ax.plot(ry, v, "-o", color=(C1, C2, C3)[j], label=nm, ms=4)
            if truth and nm in truth and key == "rcn":
                ax.axhline(truth[nm], color=(C1, C2, C3)[j], lw=0.8, ls="--")
        r2 = [r[key]["r2"] if key in r else np.nan for r in sweep]
        ax2 = ax.twinx(); ax2.plot(ry, r2, "x:", color=MUTED, label="holdout R2"); ax2.set_ylim(0, 1.05); ax2.set_ylabel("holdout R2", color=MUTED)
        ax.set_xlabel("r_y / L"); ax.legend(fontsize=7, loc="upper left")
    axs[0].set_title("forced RCN triple against the bump half-width", fontsize=9)
    axs[1].set_title("forced {w div k, k.grad w, w} against the bump half-width", fontsize=9)
    fig.suptitle("%s, mu %.3f" % (stem, mu), fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def fig_relA(table, path):
    rows = sorted([r for r in table if np.isfinite(r.get("mu", np.nan))], key=lambda r: r["mu"])
    if len(rows) < 2:
        return
    mu = np.array([r["mu"] for r in rows])
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(mu, [r["A_beta"] for r in rows], "-o", color=C1, ms=4, label="beta, div k = -beta w")
    axs[0].plot(mu, [r["A3_beta"] for r in rows], "-s", color=C2, ms=4, label="beta, with the w^3 correction")
    axs[0].plot(mu, [r["A1_beta_in"] for r in rows], ":", color=C1, lw=1, label="-beta w: near the line")
    axs[0].plot(mu, [r["A1_beta_out"] for r in rows], "--", color=C1, lw=1, label="-beta w: away from it")
    axs[0].plot(mu, [r["AG_beta_in"] for r in rows], ":", color=C3, lw=1.5, label="full form: near")
    axs[0].plot(mu, [r["AG_beta_out"] for r in rows], "--", color=C3, lw=1.5, label="full form: away")
    axs[0].set_title("relation A on the demodulated profile (last used frame)", fontsize=9)
    axs[1].plot(mu, [r["A3_gamma_over_beta"] for r in rows], "-o", color=C3, ms=4, label="measured gamma / beta")
    axs[1].plot(mu, [r["sh_gamma_over_beta"] for r in rows], "--", color=MUTED, label="SH single mode 1/(4R)")
    axs[1].axhline(0, color=MUTED, lw=0.6)
    axs[1].set_title("correction: div k = -beta w + gamma w^3", fontsize=9)
    axs[2].plot(mu, [r["A_r2"] for r in rows], "-o", color=C1, ms=4, label="{w}")
    axs[2].plot(mu, [r["A3_r2"] for r in rows], "-s", color=C2, ms=4, label="{w, w^3}")
    axs[2].plot(mu, [r["A2_r2"] for r in rows], "-^", color=C4, ms=4, label="{w, w|w|}")
    axs[2].plot(mu, [r["AG_r2"] for r in rows], "-x", color=C3, ms=5, label="full form, R fixed")
    axs[2].set_title("holdout R2", fontsize=9)
    for ax in axs:
        ax.set_xlabel("mu"); ax.legend(fontsize=7)
    fig.suptitle("Relation A with the higher-order terms of G^2", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_set(table, path):
    rows = sorted([r for r in table if np.isfinite(r["mu"])], key=lambda r: r["mu"])
    if len(rows) < 2:
        return
    mu = [r["mu"] for r in rows]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for nm, c in zip(RCN, (C1, C2, C3)):
        axs[0].plot(mu, [r["rcn_" + nm] for r in rows], "-o", color=c, label=nm, ms=4)
    axs[0].set_title("forced RCN on the demodulated profile", fontsize=9)
    for nm, c in zip(SP + ["w"], (C1, C2, C3)):
        axs[1].plot(mu, [r["SPw_" + nm] for r in rows], "-o", color=c, label=nm, ms=4)
    axs[1].set_title("forced {w div k, k.grad w, w} on the demodulated profile", fontsize=9)
    axs[2].plot(mu, [r["rcn_r2"] for r in rows], "-o", color=C1, label="RCN", ms=4)
    axs[2].plot(mu, [r["SPw_r2"] for r in rows], "-s", color=C2, label="{S pair, w}", ms=4)
    axs[2].plot(mu, [r["free_r2"] for r in rows], "-x", color=MUTED, label="free", ms=4)
    axs[2].plot(mu, [r["slaved_r2"] for r in rows], "-^", color=C4, label="A_t slaved law", ms=4)
    axs[2].set_title("holdout R2 (later half in time)", fontsize=9)
    for ax in axs:
        ax.set_xlabel("mu"); ax.legend(fontsize=7)
    fig.suptitle("Estimator-free laws on the x-invariant stage against mu", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ---------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------

def control_rcn(args):
    mu = args.mu; k1 = float(np.sqrt(1 - mu ** 2))
    y, tt, p, Phi, pt = V.rcn_knee_relaxation(mu, 1 / 3, 1 / 3, 1 / 3, 0.5, 1.8, 60.0, 300, 2.0, 41)
    dx = y[1] - y[0]
    nx = int(round(4 * 2 * np.pi / k1 / dx)); x = (np.arange(nx) - nx // 2) * dx
    u = np.cos(k1 * x[None, :, None] + Phi[:, None, :]).astype(np.float32)
    md = dict(mu=mu, Rscale=0.5, control="rcn")
    return u, x, y, tt, md, "control_rcn_mu%.2f" % mu, TRUTH_RCN


def control_amp(args):
    mu = args.mu; k1 = float(np.sqrt(1 - mu ** 2)); R = 0.5; b = 1.0
    dx = 0.2
    y = np.arange(-150, 151) * dx
    nx = int(round(4 * 2 * np.pi / k1 / dx)); x = (np.arange(nx) - nx // 2) * dx
    z = b * mu * y
    phi = np.log(np.cosh(z)) / b
    k2 = mu * np.tanh(z); w = k1 ** 2 + k2 ** 2 - 1.0
    tt = np.linspace(0, 6, 61)
    sol = solve_ivp(lambda t, a: (R - w ** 2) * a - 0.75 * a ** 3, (0, 6), 0.3 * np.ones_like(y), t_eval=tt, rtol=1e-8, atol=1e-10)
    Aa = sol.y
    u = (Aa[:, None, :] * np.cos(k1 * x[None, :, None] + phi[:, None, None])).astype(np.float32)
    md = dict(mu=mu, Rscale=R, control="amp")
    return u, x, y, tt, md, "control_amp_mu%.2f" % mu, {"A": R, "w^2 A": -1.0, "A^3": -0.75}


# ---------------------------------------------------------------------

def control_selfdual(args, gamma_on=True):
    """static x-invariant knee solving div k = -beta w sqrt(1 - w^2/(2R)) (Alan's
    sqrt(G^2) with the SH single-mode B = w - w^3/R), or the plain -beta w"""
    mu = args.mu; k1 = float(np.sqrt(1 - mu ** 2)); R = args.R; beta = 1.5
    dx = 0.1
    y = np.arange(-600, 601) * dx
    def rhs(yy, k2):
        w = k2 ** 2 - mu ** 2
        return -beta * w * (np.sqrt(np.maximum(1 - w ** 2 / (2 * R), 0.0)) if gamma_on else 1.0)
    sol = solve_ivp(rhs, (0, y[-1]), [0.0], t_eval=y[y >= 0], rtol=1e-10, atol=1e-12, max_step=dx)
    k2p = sol.y[0]
    k2 = np.concatenate([-k2p[:0:-1], k2p])
    phi = np.concatenate([[0.0], np.cumsum(0.5 * (k2[1:] + k2[:-1]) * dx)])
    phi -= phi[len(y) // 2]
    nx = int(round(4 * 2 * np.pi / k1 / dx)); x = (np.arange(nx) - nx // 2) * dx
    tt = np.linspace(0, 1, 3)
    u = np.repeat(np.cos(k1 * x[None, :] + phi[:, None])[..., None], len(tt), axis=2).astype(np.float32)
    md = dict(mu=mu, Rscale=R, control="selfdual")
    truth = dict(gamma_over_beta=(1.0 / (4 * R)) if gamma_on else 0.0, beta=beta)
    return u, x, y, tt, md, "control_selfdual_%s_mu%.2f" % ("full" if gamma_on else "plain", mu), truth


def control_rcn4(args):
    """the RCN knee relaxation with the quartic stretching f = w - w^3/R
    (v1.9 rcn_knee_relaxation keep_quartic); truth theta_t = 2 S_full - Lap div k"""
    mu = args.mu; k1 = float(np.sqrt(1 - mu ** 2))
    y, tt, p, Phi, pt = V.rcn_knee_relaxation(mu, 1 / 3, 1 / 3, 1 / 3, args.R, 1.8, 60.0, 300, 2.0, 41, keep_quartic=True)
    dx = y[1] - y[0]
    nx = int(round(4 * 2 * np.pi / k1 / dx)); x = (np.arange(nx) - nx // 2) * dx
    u = np.cos(k1 * x[None, :, None] + Phi[:, None, :]).astype(np.float32)
    md = dict(mu=mu, Rscale=args.R, control="rcn4")
    return u, x, y, tt, md, "control_rcn4_mu%.2f" % mu, {"S_full": 2.0, "Lap div k": -1.0}


def t_end_for(stem, t, args):
    """--t_end if given; else from --events (events.csv of knee_amplitude_sweep):
    the frame before the first partial circulation of that stem, or the end"""
    args.t_end_source = "--t_end"
    if args.t_end > 0:
        return float(args.t_end)
    if args.events:
        try:
            for r in csv.DictReader(open(args.events)):
                if r["stem"] == stem:
                    tc = float(r["t_circ"])
                    args.t_end_source = "events.csv t_circ %.3f minus one frame" % tc if np.isfinite(tc) else "events.csv (no circulation)"
                    if np.isfinite(tc):
                        dt = float(np.median(np.diff(t)))
                        return float(tc - dt)
                    return -1.0
            args.t_end_source = "events.csv (stem not found)"
        except Exception as exc:
            args.t_end_source = "events.csv unreadable: %r" % exc
    return -1.0


def run_one(u, x, y, t, md, stem, args, rng, out, lines, truth=None):
    mu = float(md.get("mu", args.mu)); Lsp = np.pi / np.sqrt(1 - mu ** 2)
    R = float(md.get("Rscale", args.R))
    jr = int(md.get("gb_row_in_crop", int(np.argmin(np.abs(y)))))
    yy = y - y[jr]
    fo = out / stem; fo.mkdir(parents=True, exist_ok=True)
    lines.append("[%s] mu %.3f, R %.3g, L %.3f, %d frames dt %.4g" % (stem, mu, R, Lsp, len(t), float(np.median(np.diff(t)))))
    A, ph, xi, k1, _ = demodulate(u, x, yy, t, mu, args, lines)
    n = gate_frames(xi, yy, t, Lsp, args, lines, t_end_for(stem, t, args))
    if n < 1:
        lines.append("  no x-invariant frame; skipped"); return None
    relA = relation_A_block(ph, yy, t, k1, Lsp, R, args, lines, [("the first frame", 0), ("the last used frame", n - 1)])
    L_ = relA["the last used frame"]
    rowA = dict(file=stem, mu=mu, R=R, n_frames=int(n), t_end=float(t[n - 1]),
                A_beta=-L_["A1"]["coefs"]["w"], A_r2=L_["A1"]["r2"], A3_beta=-L_["A3"]["coefs"]["w"], A3_gamma=L_["A3"]["coefs"]["w^3"],
                A3_r2=L_["A3"]["r2"], A3_gamma_over_beta=L_["gamma_over_beta"], sh_gamma_over_beta=1.0 / (4.0 * R),
                A2_c=L_["A2"]["coefs"]["w|w|"], A2_r2=L_["A2"]["r2"], w_line=L_["w_line"],
                AG_beta=-L_["AG"]["coefs"]["w_G"], AG_r2=L_["AG"]["r2"],
                AG_beta_in=L_["AG"]["beta_inner"], AG_beta_out=L_["AG"]["beta_outer"], A1_beta_in=L_["A1"]["beta_inner"], A1_beta_out=L_["A1"]["beta_outer"])
    if truth is not None and "gamma_over_beta" in truth:
        lines.append("  CONTROL: gamma/beta %.3f against the planted %.3f; beta %.3f against %.3f" % (
            rowA["A3_gamma_over_beta"], truth["gamma_over_beta"], rowA["A3_beta"], truth["beta"]))
    if n < 2 * args.t_half + 3:
        lines.append("  %d x-invariant frames: relation A only (the rate blocks need %d)" % (n, 2 * args.t_half + 3))
        with open(fo / "fits.json", "w") as fh:
            json.dump(dict(relation_A=relA), fh, indent=1, default=str)
        (fo / "report.txt").write_text("\n".join(lines) + "\n")
        return rowA
    A, ph, t2 = A[:, :n], ph[:, :n], t[:n]
    t0 = time.time()
    fits, fields = fit_file(A, ph, yy, t2, k1, Lsp, R, mu, args, rng, lines, fo, truth)
    fits["relation_A"] = relA
    if truth is not None and "gamma_over_beta" not in truth:
        key = "tied_SfullB" if "S_full" in truth else ("forced_rcn" if "w div k" in truth else "forced_slaved")
        cf = fits[key]["coefs"]
        lines.append("  CONTROL: forced coefficients against the truth: " + ", ".join("%s %.3f (truth %.3f, %.1f%%)" % (k, cf[k], v, 100 * abs(cf[k] - v) / abs(v)) for k, v in truth.items()))
    fig_profiles(A, ph, yy, t2, fields, Lsp, stem, mu, fo / ("fig_profiles_%s.png" % stem))
    fig_slaving(A, yy, t2, fields, Lsp, R, stem, mu, fits["slaving"], fo / ("fig_slaving_%s.png" % stem))
    fig_ry(fits["ry_sweep"], stem, mu, fo / ("fig_ry_%s.png" % stem), TRUTH_RCN if (truth and "w div k" in truth) else None)
    with open(fo / "fits.json", "w") as fh:
        json.dump(fits, fh, indent=1, default=str)
    (fo / "report.txt").write_text("\n".join(lines) + "\n")
    lines.append("  %.1f s" % (time.time() - t0))
    g = lambda key, nm: (fits.get(key) or {}).get("coefs", {}).get(nm, np.nan)
    r2 = lambda key: (fits.get(key) or {}).get("r2_holdout", np.nan)
    row = dict(file=stem, mu=mu, R=R, n_frames=int(n), t_end=float(t2[-1]), n_rows=fits["n_rows"],
               free_support=cstr((fits.get("free_phase") or {}).get("coefs", {})), free_r2=r2("free_phase"),
               SPw_r2=r2("forced_SPw"), rcn_r2=r2("forced_rcn"), SPw_r2_time=(fits.get("forced_SPw") or {}).get("r2_time_split", np.nan), rcn_r2_time=(fits.get("forced_rcn") or {}).get("r2_time_split", np.nan), SPw_tau_r2=r2("forced_SPw_tau"), rcn_w_r2=r2("forced_rcn_w"),
               free_amp_support=cstr((fits.get("free_amp") or {}).get("coefs", {})), free_amp_r2=r2("free_amp"),
               slaved_r2=r2("forced_slaved"), slaved_lap_r2=r2("forced_slaved_lap"),
               slaving_r2_end=fits["slaving"][-1]["r2"], slaving_ratio_end=fits["slaving"][-1]["ratio"])
    for nm in RCN:
        row["rcn_" + nm] = g("forced_rcn", nm)
    for nm in SP + ["w"]:
        row["SPw_" + nm] = g("forced_SPw", nm)
    for nm in SP + ["w", "w theta_t"]:
        row["SPwtau_" + nm] = g("forced_SPw_tau", nm)
    for nm in ("A", "w^2 A", "A^3"):
        row["slaved_" + nm] = g("forced_slaved", nm)
    for nm in ("S", "Lap div k"):
        row["tiedSB_" + nm] = g("tied_SB", nm)
    for nm in ("S_full", "Lap div k"):
        row["tiedSfull_" + nm] = g("tied_SfullB", nm)
    row["tied_SB_r2"] = r2("tied_SB"); row["tied_Sfull_r2"] = r2("tied_SfullB")
    for nm in RCN + ["w^3 div k", "w^2 k.grad w"]:
        row["cubic_" + nm] = g("forced_rcn_cubic", nm)
    row["cubic_r2"] = r2("forced_rcn_cubic")
    for nm in RCN + ["w k.grad lnA"]:
        row["lnA_" + nm] = g("forced_rcn_lnA", nm)
    row["lnA_r2"] = r2("forced_rcn_lnA")
    for nm in ("S_A", "Lap div k"):
        row["tiedSA_" + nm] = g("tied_SAB", nm)
    row["tied_SA_r2"] = r2("tied_SAB")
    row.update({k: v for k, v in rowA.items() if k not in row})
    return row


def main(args):
    rng = np.random.default_rng(args.seed)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    lines = ["knee_demod_laws v1.1; r_y %.2g L (sweep %s), t_half %d, poly p %d, band_to %.2g L, x-invariance gate %.3f, rel_tol %.2g, min_contrib %.2g" % (
        args.ry, args.ry_sweep, args.t_half, args.poly_p, args.band_to, args.xinv_gate, args.rel_tol, args.min_contrib), ""]
    table = []
    jobs = []
    if args.mode == "control_rcn":
        jobs.append(control_rcn(args))
    elif args.mode == "control_amp":
        jobs.append(control_amp(args))
    elif args.mode == "control_selfdual":
        jobs.append(control_selfdual(args, True)); jobs.append(control_selfdual(args, False))
    elif args.mode == "control_rcn4":
        jobs.append(control_rcn4(args))
    else:
        for f in [s.strip() for s in args.npz.split(",") if s.strip()]:
            try:
                D = load_npz(f)
                jobs.append((D["u"], D["x"], D["y"], D["t"], D["md"], Path(f).stem, None))
            except Exception as exc:
                lines.append("FAILED to load %s: %r" % (f, exc))
    for (u, x, y, t, md, stem, truth) in jobs:
        try:
            row = run_one(u, x, y, t, md, stem, args, rng, out, lines, truth)
            if row:
                table.append(row)
        except Exception as exc:                                         # loud flag, keep going
            lines.append("FAILED %s: %r" % (stem, exc))
        lines.append("")
        (out / "report.txt").write_text("\n".join(lines) + "\n")
    if table:
        allk = []
        for r in table:
            allk += [k for k in r if k not in allk]
        with open(out / "demod_laws_table.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=allk, restval=""); wr.writeheader(); wr.writerows(table)
        keys = ["file", "mu", "n_frames", "t_end", "rcn_w div k", "rcn_k.grad w", "rcn_Lap div k", "rcn_r2", "rcn_r2_time", "SPw_w div k", "SPw_k.grad w", "SPw_w", "SPw_r2", "SPw_r2_time",
                "SPwtau_w", "SPwtau_w theta_t", "SPw_tau_r2", "free_support", "free_r2", "slaved_A", "slaved_w^2 A", "slaved_A^3", "slaved_r2", "slaving_r2_end", "slaving_ratio_end", "free_amp_support", "free_amp_r2"]
        fmt = lambda v: ("%.3f" % v if isinstance(v, float) and np.isfinite(v) else str(v))
        keys += ["AG_beta", "AG_r2", "AG_beta_in", "AG_beta_out", "A1_beta_in", "A1_beta_out", "A_beta", "A_r2", "A3_beta", "A3_gamma", "A3_gamma_over_beta", "sh_gamma_over_beta", "A3_r2", "w_line",
                 "tiedSB_S", "tiedSB_Lap div k", "tied_SB_r2", "tiedSfull_S_full", "tiedSfull_Lap div k", "tied_Sfull_r2",
                 "cubic_w^3 div k", "cubic_w^2 k.grad w", "cubic_Lap div k", "cubic_r2", "lnA_w k.grad lnA", "lnA_Lap div k", "lnA_r2", "tiedSA_S_A", "tiedSA_Lap div k", "tied_SA_r2"]
        md_lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)] + ["| " + " | ".join(fmt(r.get(k, "")) for k in keys) + " |" for r in table]
        (out / "demod_laws_table.md").write_text("\n".join(md_lines) + "\n")
        fig_set([r for r in table if "rcn_r2" in r], out / "fig_laws_vs_mu.png")
        fig_relA(table, out / "fig_relationA_correction_vs_mu.png")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    with open(out / "config.json", "w") as fh:
        json.dump({k: getattr(args, k) for k in vars(args) if not k.startswith("_")}, fh, indent=2, default=str)
    print("\n".join(lines)); print("outputs ->", out)


class _Args:
    mode = "data"             # data | control_rcn | control_amp | control_rcn4 | control_selfdual (v1.1)
    npz = ""                  # one movie or a comma-separated list
    outdir = "results/knee_demod_laws"
    mu = 0.85                 # controls, and fallback when a file has no mu
    R = 0.5                   # fallback when a file has no Rscale
    t_end = -1.0              # use frames up to this time (<= 0: from --events, else the x-invariance gate alone)
    events = ""               # events.csv of knee_amplitude_sweep: t_end = the frame before each stem's first partial circulation
    t_end_source = ""
    xinv_gate = 0.98          # frames are used while the band's median row x-invariance is at least this
    band_to = 2.0             # fitted rows: |y| < band_to L
    far_from = 4.0            # far field for A_far, L
    ry = 0.5                  # bump half-width across the row, L
    ry_A = 1.5                # bump half-width of the per-frame relation A fit (v1.1), L
    split_A = 0.75            # relation A: beta read separately within and beyond this distance from the line, L
    ry_sweep = "0.25,0.5,0.75,1.0,1.5"
    t_half = 3                # bump half-width in time, frames
    poly_p = 6
    stride_y = 1
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
