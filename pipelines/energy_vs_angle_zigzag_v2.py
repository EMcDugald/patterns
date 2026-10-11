# pipelines/energy_vs_angle_zigzag.py
"""
Line energy of a single zigzag phase grain boundary as a function of the
far-field angle alpha, with topological (nucleation) tagging.

What this does
--------------
For each mu in a grid (alpha = arcsin(mu), k1 = cos(alpha) = sqrt(1 - mu^2),
k2 = sin(alpha) = mu):

  1. Relax an SH zigzag configuration to t = tmax (gradient flow), saving the
     last few frames so the energy plateau can be checked.
  2. From the final frame's energy density (already computed spectrally by the
     solver), form the row-averaged profile e_bar(y) = mean_x e(x, y), read the
     straight-roll baseline e_roll off the outermost rows on each side of the
     boundary, and integrate the excess:

         E_GB(alpha) = integral over y of ( e_bar(y) - e_roll ) dy

     This is the excess free energy per unit length along the boundary. The
     baseline is measured from the relaxed far field rather than an analytic
     constant, so it subtracts whatever wavenumber the pattern actually selects.
  3. Tag whether a dislocation has nucleated using a lightweight Riesz
     rho_d -> local-winding proxy computed straight from u (a stand-in for the
     paper-grade J from zigzag_defect_J_analysis.py, which needs the full OP
     pipeline). A rectangular ramp is applied before the FFT because the crop is
     non-periodic.

Then E_GB(alpha) is overlaid with three candidate ansaetze,

     C  * sin(alpha)^3          (small-angle branch),
     C1 * (1 - sin(alpha))      (large-angle ansatz, slope 1 in 1-mu),
     K  * cos(alpha)            (large-angle ansatz, slope 1/2 in 1-mu),

and the decisive local log-log slope of E_GB versus (1 - mu) near mu -> 1 is
reported: slope ~ 1/2 supports K*cos(alpha), slope ~ 1 supports (1 - sin alpha),
because cos(alpha) = sqrt(2) (1 - mu)^{1/2} while 1 - sin(alpha) = 1 - mu.

Normalization note
------------------
E_GB is a per-unit-length quantity: mean over x then integrate over y. The
straight-roll energy is subtracted, so the plotted value is the boundary's
excess over defect-free rolls occupying the same strip. As a sanity overlay the
leading-order roll density -R^2/6 is drawn, but at Rscale = 1 the pattern is well
past onset, so the measured baseline (not the analytic one) is the reference.

Outputs
-------
  <output_dir>/cache/<stem>.npz     per-mu compact record (re-run friendly)
  <output_dir>/energy_vs_angle.csv  summary table
  <output_dir>/energy_vs_angle.npz  summary arrays
  <output_dir>/figures/*.png        main curve, log-log slope, e_bar profiles

v1.1 changelog (measured justification):
  - Acceptance mask applied before every displayed fit (the Writeup 1
    red todo on the K(phi) figures). A point is accepted when its
    resolution is above the floor (ppw_y > accept_ppw_floor, default
    8.0, the eight-points-per-wavelength floor the sweep hits at
    mu = 0.97), its tails are flat (flatness <= accept_flat_tol,
    default 0.05, the existing warning threshold), and its measured
    baseline sits within accept_base_tol (default 0.15 relative) of
    -R^2/6 (the mu >= 0.996 points read -0.0556 and -0.0604 against
    -0.0417 and are excluded by this test). Dislocation-branch fits
    (C1 (1-sin), K cos, the log-log slope, the K(alpha) single fit and
    extrapolation) additionally require n_dislocations >= 1, which
    removes the metastable un-nucleated mu = 0.999 point from those
    fits; the sin^3 branch fit requires n_dislocations == 0. Excluded
    points are still plotted, as open grey symbols, and listed in the
    console and CSV (accepted column). The E_GB numbers themselves are
    unchanged; only the fits and overlay lines move.
  - No change to the sin^3 exponent handling: the small-angle fit was
    already the one-parameter amplitude fit with the exponent pinned
    at 3 (C_sin3, window mu <= sin3_fit_mu_max).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

from sh_sims.pgb_zigzag import solve_sh_pgb_zigzag
from op_extract.riesz import riesz_analyze_pattern, local_winding_number

try:
    from utils.geometry import build_rectangular_ramp_smooth
except Exception:  # standalone fallback
    def build_rectangular_ramp_smooth(x, y, xmargin=0.10, ymargin=0.10,
                                      tanhscale=7.5):
        X, Y = np.meshgrid(x, y)
        Lx = x[-1] - x[0]
        Ly = y[-1] - y[0]
        xm = xmargin * Lx
        ym = ymargin * Ly
        dxl = X - x[0]
        dxr = x[-1] - X
        dyb = Y - y[0]
        dyt = y[-1] - Y
        rx = 0.5 * (1.0 + np.tanh((np.minimum(dxl, dxr) - xm) * tanhscale / Lx))
        ry = 0.5 * (1.0 + np.tanh((np.minimum(dyb, dyt) - ym) * tanhscale / Ly))
        return rx * ry


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_stem(mu, cfg):
    return (
        f"Egb_zigzag_mu{mu:.3f}"
        f"_T{int(cfg['tmax'])}"
        f"_nx{cfg['Nx']}"
        f"_np{cfg['n_periods']}"
        f"_NyF{cfg['Ny_factor']:g}"
        f"_R{cfg['Rscale']:g}"
    )


# ---------------------------------------------------------------------------
# Energy normalization
# ---------------------------------------------------------------------------

def line_energy_from_frame(e_last, y, gb_row, edge_frac=0.12):
    """
    Excess line energy per unit length from one energy-density frame.

    e_last : (Ny, Nx) energy density of the final frame (cropped around one GB).
    y      : (Ny,) cropped y coordinate.
    gb_row : int, grid row of the boundary core within the crop.
    edge_frac : fraction of rows at the very top and very bottom used to read
                the straight-roll baseline.

    Returns dict with E_GB, e_bar(y), the two edge baselines, their relative
    mismatch (a flatness / adequacy check), and the excess profile.
    """
    Ny = e_last.shape[0]
    dy = float(y[1] - y[0])

    e_bar = e_last.mean(axis=1)          # (Ny,) row average over x

    n_edge = max(2, int(round(edge_frac * Ny)))
    e_roll_bot = float(np.mean(e_bar[:n_edge]))
    e_roll_top = float(np.mean(e_bar[-n_edge:]))
    e_roll = 0.5 * (e_roll_bot + e_roll_top)

    scale = max(abs(e_roll), 1e-12)
    flatness = abs(e_roll_top - e_roll_bot) / scale   # 0 = perfectly flat tails

    excess = e_bar - e_roll
    E_GB = float(np.sum(excess) * dy)

    return {
        "E_GB": E_GB,
        "e_bar": e_bar,
        "excess": excess,
        "e_roll": e_roll,
        "e_roll_bot": e_roll_bot,
        "e_roll_top": e_roll_top,
        "flatness": flatness,
        "gb_row": int(gb_row),
    }


# ---------------------------------------------------------------------------
# Nucleation tag (lightweight Riesz proxy for J)
# ---------------------------------------------------------------------------

# The production detector, if importable in this environment. Same function
# the defect-index GIFs use. If it is not on the path we fall back to an
# equivalent plaquette-winding count below (validated: knee 0, PN nonzero).
try:
    from op_extract.t1_common import detect_defects as _detect_defects
except Exception:
    _detect_defects = None


def _wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def _plaquette_winding_count(u, dx, dy, gb_row, band_rows, sigma_det=0.4):
    """
    Fallback tag: winding of the doubled Riesz director at the trusted small
    width, counted in a band around the core. Mirrors detect_defects' core
    computation (sum|q| / 2 dislocations) and was validated on the analytic
    knee (0) and PN (nonzero). Used only when detect_defects cannot be imported.
    """
    from scipy.ndimage import gaussian_filter
    rz = riesz_analyze_pattern(u, dx, dy)
    b = rz["orientation"]
    c2 = gaussian_filter(np.cos(2 * b), sigma_det / dx, mode="nearest")
    s2 = gaussian_filter(np.sin(2 * b), sigma_det / dy, mode="nearest")
    a = np.arctan2(s2, c2)
    d1 = _wrap(a[:-1, 1:] - a[:-1, :-1]); d2 = _wrap(a[1:, 1:] - a[:-1, 1:])
    d3 = _wrap(a[1:, :-1] - a[1:, 1:]);   d4 = _wrap(a[:-1, :-1] - a[1:, :-1])
    q = np.round((d1 + d2 + d3 + d4) / (2 * np.pi))
    r0 = max(0, gb_row - band_rows); r1 = min(q.shape[0], gb_row + band_rows + 1)
    sum_abs = float(np.abs(q[r0:r1, :]).sum())
    return sum_abs, sum_abs / 2.0


def defect_indicators(u_last, x, y, dx, dy, gb_row, meta,
                      band_periods=2.0, sigma_det=0.4, **_ignored):
    """
    Nucleation tag from the production detector detect_defects at the trusted
    small width (0.4). The core is recentred to y=0 (detect_defects reads a band
    about y=0), so the band picks up the boundary. Returns the dislocation count
    and sum|q|. On relaxed knee endpoints this reads 0, which is the trusted way
    of saying the state sits on the knee branch.

    Falls back to an equivalent plaquette-winding count if detect_defects is not
    importable, so the sweep never depends on the full director stack being present.
    """
    lam_y = float(meta.get("lambda_y", np.pi))
    band_y = band_periods * lam_y
    band_rows = max(3, int(round(band_y / dy)))

    if _detect_defects is not None:
        Lx = float(x[-1] - x[0]); Ly = float(y[-1] - y[0])
        try:
            det = _detect_defects(u_last, x, y - y[gb_row], Lx, Ly,
                                  sigma_det=sigma_det, band_y=band_y, band_x=None)
            n_disl = float(det.get("n_dislocations", np.nan))
            sum_abs = float(det.get("sum_abs", np.nan))
            return {"n_dislocations": n_disl, "sum_abs": sum_abs,
                    "detector": "detect_defects"}
        except Exception as exc:
            print(f"    [detect_defects failed: {exc}; using fallback]")

    sum_abs, n_disl = _plaquette_winding_count(u_last, dx, dy, gb_row,
                                               band_rows, sigma_det)
    return {"n_dislocations": n_disl, "sum_abs": sum_abs,
            "detector": "plaquette_fallback"}


# ---------------------------------------------------------------------------
# One mu
# ---------------------------------------------------------------------------

def process_mu(mu, cfg, cache_dir):
    stem = build_stem(mu, cfg)
    cache_path = cache_dir / f"{stem}.npz"

    if cache_path.exists() and not cfg.get("overwrite", False):
        d = np.load(cache_path, allow_pickle=True)
        rec = {k: d[k] for k in d.files}
        # backfill the new tag for caches written before the detect_defects swap
        if "n_dislocations" not in rec and "u_last" in rec:
            dx = float(rec["dx"]); dy = float(rec["dy"])
            u_last = np.asarray(rec["u_last"]); yv = np.asarray(rec["y"])
            Nxc = u_last.shape[1]
            xv = (np.arange(Nxc) - Nxc // 2) * dx
            meta = {"lambda_y": 2.0 * np.pi / max(float(rec["mu"]), 1e-9)}
            tag = defect_indicators(u_last, xv, yv, dx, dy,
                                    int(rec["gb_row"]), meta,
                                    band_periods=cfg.get("band_periods", 2.0),
                                    sigma_det=cfg.get("sigma_det", 0.4))
            rec["n_dislocations"] = tag["n_dislocations"]
            rec["sum_abs"] = tag["sum_abs"]
        print(f"  cached  mu={mu:.3f}  E_GB={float(d['E_GB']):.5g}")
        return rec

    print(f"  solve   mu={mu:.3f} ...")
    res = solve_sh_pgb_zigzag(
        Nx=cfg["Nx"], mu=float(mu), h=cfg["h"], tmax=cfg["tmax"],
        nsave=cfg["nsave"], n_periods=cfg["n_periods"],
        Ny_factor=cfg["Ny_factor"], Rscale=cfg["Rscale"], amp=cfg["amp"],
        y_centering=cfg["y_centering"], energy=True,
        t_save_window=cfg.get("t_save_window"), which_gb=cfg["which_gb"],
        crop_Nx=cfg.get("crop_Nx"), crop_Ny=cfg.get("crop_Ny"),
        save_initial_phase=False,
    )

    meta = json.loads(res["metadata_json"])
    x = res["x"]; y = res["y"]
    dx = float(meta["dx"]); dy = float(meta["dy"])
    gb_row = int(meta["gb_row_in_crop"])
    k2 = float(meta["k2"])
    lam_y_full = (2.0 * np.pi / k2) if k2 > 0 else np.inf
    ppw_y = lam_y_full / dy   # points per full y-wavelength

    e = res["e"]                     # (Ny, Nx, nsave+1)
    u = res["u"]
    tt = res["tt"]

    en = line_energy_from_frame(e[..., -1], y, gb_row,
                                edge_frac=cfg.get("edge_frac", 0.12))

    # plateau check: line energy of the previous saved frame
    E_prev = np.nan
    if e.shape[-1] >= 2:
        en_prev = line_energy_from_frame(e[..., -2], y, gb_row,
                                         edge_frac=cfg.get("edge_frac", 0.12))
        E_prev = en_prev["E_GB"]

    tag = defect_indicators(u[..., -1], x, y, dx, dy, gb_row, meta,
                            band_periods=cfg.get("band_periods", 2.0),
                            sigma_det=cfg.get("sigma_det", 0.4))

    record = {
        "mu": float(mu),
        "alpha": float(np.arcsin(mu)),
        "E_GB": en["E_GB"],
        "E_GB_prev": float(E_prev),
        "e_roll": en["e_roll"],
        "e_roll_bot": en["e_roll_bot"],
        "e_roll_top": en["e_roll_top"],
        "flatness": en["flatness"],
        "n_dislocations": tag["n_dislocations"],
        "sum_abs": tag["sum_abs"],
        "ppw_y": float(ppw_y),
        "dx": dx, "dy": dy,
        "t_final": float(tt[-1]),
        "y": y,
        "e_bar": en["e_bar"],
        "excess": en["excess"],
        "u_last": u[..., -1].astype(np.float32),
        "e_last": e[..., -1].astype(np.float32),
        "gb_row": gb_row,
    }
    np.savez_compressed(cache_path, **record)

    dE = abs(record["E_GB"] - E_prev) / max(abs(record["E_GB"]), 1e-12)
    warn = ""
    if np.isfinite(dE) and dE > 0.02:
        warn += f"  [not plateaued: dE/E={dE:.2%}]"
    if ppw_y < 8:
        warn += f"  [y under-resolved: {ppw_y:.1f} pts/wave]"
    if en["flatness"] > 0.05:
        warn += f"  [tails not flat: {en['flatness']:.2%}]"
    print(f"    E_GB={record['E_GB']:.5g}  n_disl={record['n_dislocations']:.1f}"
          f"  ppw_y={ppw_y:.1f}{warn}")
    return record


# ---------------------------------------------------------------------------
# Ansatz fits
# ---------------------------------------------------------------------------

def _ls_amp(basis, E):
    """Least-squares single amplitude a minimizing || a*basis - E ||."""
    basis = np.asarray(basis); E = np.asarray(E)
    denom = float(np.sum(basis * basis))
    if denom <= 0:
        return np.nan
    return float(np.sum(basis * E) / denom)


def acceptance_mask(records, cfg):
    """Per-point quality acceptance, applied before every displayed fit.

    accept = (ppw_y > ppw_floor) & (flatness <= flat_tol)
             & (|e_roll - (-R^2/6)| <= base_tol * R^2/6)

    Rationale (Writeup 1, results section): at mu >= 0.996 the measured
    baseline shifts from -0.0417 to -0.0556 and -0.0604 while the
    resolution falls to the eight-points-per-wavelength floor, so the
    line-energy reading is no longer trustworthy there. The branch
    masks below add the topology requirement (nucleated points for the
    dislocation branch, un-nucleated for the sin^3 branch), which also
    removes the metastable un-nucleated mu = 0.999 point from the
    dislocation-branch fits.
    """
    ppw = np.array([float(r.get("ppw_y", np.inf)) for r in records])
    flat = np.array([float(r.get("flatness", 0.0)) for r in records])
    e_roll = np.array([float(r.get("e_roll", np.nan)) for r in records])
    base = -cfg["Rscale"]**2 / 6.0
    ok_ppw = ppw > cfg.get("accept_ppw_floor", 8.0)
    ok_flat = flat <= cfg.get("accept_flat_tol", 0.05)
    ok_base = np.abs(e_roll - base) <= cfg.get("accept_base_tol", 0.15) * abs(base)
    return ok_ppw & ok_flat & ok_base


def fit_ansatze(mu, alpha, E, cfg, accept=None, ndis=None):
    mu = np.asarray(mu); alpha = np.asarray(alpha); E = np.asarray(E)
    s = np.sin(alpha); c = np.cos(alpha)
    if accept is None:
        accept = np.ones_like(mu, dtype=bool)
    if ndis is None:
        ndis = np.zeros_like(mu)
    nuc = np.asarray(ndis) >= 1.0

    # branch masks: window AND quality AND topology
    lo = (mu <= cfg.get("sin3_fit_mu_max", 0.7)) & accept & (~nuc)
    hi = (mu >= cfg.get("large_fit_mu_min", 0.85)) & accept & nuc

    C_sin3 = _ls_amp(s[lo]**3, E[lo]) if lo.sum() >= 1 else np.nan
    C1_oms = _ls_amp((1.0 - s)[hi], E[hi]) if hi.sum() >= 1 else np.nan
    K_cos = _ls_amp(c[hi], E[hi]) if hi.sum() >= 1 else np.nan

    # decisive local slope near mu -> 1, accepted nucleated points only
    ll = (mu >= cfg.get("loglog_mu_min", 0.9)) & accept & nuc
    slope = intercept = np.nan
    if ll.sum() >= 2:
        one_minus_mu = 1.0 - mu[ll]
        good = (one_minus_mu > 0) & (E[ll] > 0)
        if good.sum() >= 2:
            X = np.log(one_minus_mu[good])
            Y = np.log(E[ll][good])
            slope, intercept = np.polyfit(X, Y, 1)

    peak_idx = int(np.argmax(E)) if E.size else -1

    return {
        "C_sin3": C_sin3, "C1_oms": C1_oms, "K_cos": K_cos,
        "loglog_slope": float(slope), "loglog_intercept": float(intercept),
        "peak_mu": float(mu[peak_idx]) if peak_idx >= 0 else np.nan,
        "peak_alpha_deg": float(np.degrees(alpha[peak_idx])) if peak_idx >= 0 else np.nan,
        "lo_mask": lo, "hi_mask": hi, "ll_mask": ll,
        "accept_mask": accept,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(records, fits, cfg, fig_dir):
    mu = np.array([r["mu"] for r in records])
    order = np.argsort(mu)
    mu = mu[order]
    records = [records[i] for i in order]

    alpha = np.array([float(r["alpha"]) for r in records])
    E = np.array([float(r["E_GB"]) for r in records])
    ndis = np.array([float(r["n_dislocations"]) for r in records])
    e_roll = np.array([float(r["e_roll"]) for r in records])
    ppw = np.array([float(r["ppw_y"]) for r in records])
    adeg = np.degrees(alpha)

    accept = fits.get("accept_mask", np.ones_like(mu, dtype=bool))
    rej = ~accept

    # ---- main curve --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    sc = ax.scatter(adeg[accept], E[accept], c=ndis[accept], cmap="viridis",
                    s=32, zorder=3, edgecolor="k", linewidth=0.3)
    if rej.any():
        ax.scatter(adeg[rej], E[rej], s=32, facecolor="none",
                   edgecolor="0.6", linewidth=1.0, zorder=2,
                   label="excluded (resolution/baseline/flatness)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("dislocation count (detect_defects, sig=0.4)")

    nuc = ndis >= 1.0
    if nuc.any():
        ax.scatter(adeg[nuc], E[nuc], s=90, facecolor="none",
                   edgecolor="crimson", linewidth=1.4, zorder=4,
                   label="tagged nucleated (n_disl >= 1)")

    a_dense = np.linspace(alpha.min(), alpha.max(), 400)
    s_d = np.sin(a_dense); c_d = np.cos(a_dense); ad_deg = np.degrees(a_dense)
    if np.isfinite(fits["C_sin3"]):
        ax.plot(ad_deg, fits["C_sin3"] * s_d**3, "b-", lw=1.6,
                label=r"$C\,\sin^3\alpha$")
    if np.isfinite(fits["C1_oms"]):
        ax.plot(ad_deg, fits["C1_oms"] * (1.0 - s_d), "g--", lw=1.6,
                label=r"$C_1(1-\sin\alpha)$")
    if np.isfinite(fits["K_cos"]):
        ax.plot(ad_deg, fits["K_cos"] * c_d, "r-.", lw=1.6,
                label=r"$K\cos\alpha$")

    if np.isfinite(fits["peak_alpha_deg"]):
        ax.axvline(fits["peak_alpha_deg"], color="0.5", ls=":", lw=1.0,
                   label=f"peak ~ {fits['peak_alpha_deg']:.1f} deg")

    ax.set_xlabel(r"far-field angle $\alpha$ (deg)")
    ax.set_ylabel(r"line energy per unit length  $E_{GB}$")
    ax.set_title(f"Zigzag PGB line energy vs angle  (R={cfg['Rscale']:g}, "
                 f"t={cfg['tmax']:g})")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p1 = fig_dir / "energy_vs_angle.png"
    fig.savefig(p1, dpi=160); plt.close(fig)

    # ---- decisive log-log slope near mu -> 1 -------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    omm = 1.0 - mu
    good = (omm > 0) & (E > 0) & accept
    shown_rej = (omm > 0) & (E > 0) & rej
    ax.loglog(omm[good], E[good], "ko", ms=5, label="measured (accepted)")
    if shown_rej.any():
        ax.loglog(omm[shown_rej], E[shown_rej], "o", ms=6, mfc="none",
                  mec="0.6", label="excluded")
    if np.isfinite(fits["loglog_slope"]):
        xs = np.logspace(np.log10(max(omm[good].min(), 1e-4)),
                         np.log10(omm[good].max()), 100)
        ys = np.exp(fits["loglog_intercept"]) * xs**fits["loglog_slope"]
        ax.loglog(xs, ys, "m-", lw=1.6,
                  label=f"fit slope = {fits['loglog_slope']:.2f}")
        # guide lines anchored at the smallest (1-mu)
        i0 = np.argmin(omm[good])
        x0 = omm[good][i0]; y0 = E[good][i0]
        ax.loglog(xs, y0 * (xs / x0)**0.5, "r--", lw=1.0,
                  label=r"slope 1/2  ($\cos\alpha$)")
        ax.loglog(xs, y0 * (xs / x0)**1.0, "g:", lw=1.2,
                  label=r"slope 1  ($1-\sin\alpha$)")
    ax.set_xlabel(r"$1-\mu$")
    ax.set_ylabel(r"$E_{GB}$")
    ax.set_title(r"Large-angle exponent: slope 1/2 ($\cos\alpha$) "
                 r"vs 1 ($1-\sin\alpha$)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    p2 = fig_dir / "energy_loglog_slope.png"
    fig.savefig(p2, dpi=160); plt.close(fig)

    # ---- K(alpha) = E_GB / cos(alpha) --------------------------------------
    # The cos ansatz predicts this collapses to a constant as mu -> 1. Plotting
    # the ratio directly shows whether K(alpha) is flattening and to what value,
    # rather than assuming a single fitted K.
    ccos = np.cos(alpha)
    good_k = (ccos > 1e-6) & accept
    rej_k = (ccos > 1e-6) & rej
    K_alpha = E[good_k] / ccos[good_k]
    figK, axK = plt.subplots(figsize=(7.0, 5.0))
    axK.plot(adeg[good_k], K_alpha, "o-", color="crimson", ms=5, lw=1.2,
             label="accepted")
    if rej_k.any():
        axK.plot(adeg[rej_k], E[rej_k] / ccos[rej_k], "o", ms=6, mfc="none",
                 mec="0.6", label="excluded")
    if np.isfinite(fits["K_cos"]):
        axK.axhline(fits["K_cos"], color="0.4", ls="--", lw=1.0,
                    label=f"single-fit K = {fits['K_cos']:.3f}")
    # local trend toward mu -> 1: fit K vs (1-mu) over the resolved large-angle end
    hi = fits.get("hi_mask",
                  (mu >= cfg.get("large_fit_mu_min", 0.85)) & accept)
    if hi.sum() >= 2:
        om = 1.0 - mu[hi]
        Kh = E[hi] / np.cos(alpha[hi])
        sl, ic = np.polyfit(om, Kh, 1)
        axK.axhline(ic, color="green", ls=":", lw=1.2,
                    label=f"K extrapolated to mu=1: {ic:.3f}")
    axK.set_xlabel(r"far-field angle $\alpha$ (deg)")
    axK.set_ylabel(r"$K(\alpha) = E_{GB}/\cos\alpha$")
    axK.set_title(r"$K(\alpha)$: flattening as $\mu\to1$ tests the $\cos\alpha$ branch")
    axK.legend(fontsize=8, loc="best")
    axK.grid(alpha=0.25)
    figK.tight_layout()
    pK = fig_dir / "K_of_alpha.png"
    figK.savefig(pK, dpi=160); plt.close(figK)

    # ---- normalization sanity: e_bar(y) profiles ---------------------------
    n_show = min(6, len(records))
    pick = np.linspace(0, len(records) - 1, n_show).round().astype(int)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for i in pick:
        r = records[i]
        yy = r["y"]; gb = int(r["gb_row"])
        ax.plot(yy - yy[gb], r["excess"], lw=1.3,
                label=f"mu={r['mu']:.2f}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"$y - y_{core}$")
    ax.set_ylabel(r"$\bar e(y) - e_{roll}$  (excess density)")
    ax.set_title("Baseline check: excess localizes at the core, tails -> 0")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p3 = fig_dir / "ebar_profiles.png"
    fig.savefig(p3, dpi=160); plt.close(fig)

    # ---- baseline vs -R^2/6 and resolution diagnostic ----------------------
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].plot(adeg, e_roll, "ko-", ms=4, label="measured e_roll")
    axs[0].axhline(-cfg["Rscale"]**2 / 6.0, color="r", ls="--",
                   label=r"$-R^2/6$ (leading order)")
    axs[0].set_xlabel(r"$\alpha$ (deg)"); axs[0].set_ylabel("roll baseline")
    axs[0].legend(fontsize=8); axs[0].grid(alpha=0.25)
    axs[0].set_title("Far-field baseline")

    axs[1].plot(adeg, ppw, "bo-", ms=4)
    axs[1].axhline(8.0, color="r", ls="--", label="8 pts/wave floor")
    axs[1].set_xlabel(r"$\alpha$ (deg)")
    axs[1].set_ylabel("points per y-wavelength")
    axs[1].legend(fontsize=8); axs[1].grid(alpha=0.25)
    axs[1].set_title("y-resolution across the sweep")
    fig.tight_layout()
    p4 = fig_dir / "diagnostics.png"
    fig.savefig(p4, dpi=160); plt.close(fig)

    return [p1, p2, pK, p3, p4]


# ---------------------------------------------------------------------------
# Field figures: integration region + fundamental period
# ---------------------------------------------------------------------------

def _draw_overlays(ax, x, y, gb_row, lam_x, n_edge):
    """GB core line, baseline read-off lines, and one fundamental x-period box."""
    ax.axhline(y[gb_row], color="k", lw=1.1)                       # boundary core
    ax.axhline(y[n_edge - 1], color="w", lw=0.9, ls="--")          # baseline inner edge
    ax.axhline(y[-n_edge], color="w", lw=0.9, ls="--")             # (rolls read outside)
    rect = plt.Rectangle((-lam_x / 2.0, y[0]), lam_x, y[-1] - y[0],
                         edgecolor="magenta", facecolor="none", lw=2.0)
    ax.add_patch(rect)


def make_field_figures(records, cfg, fig_dir):
    """
    One 3-panel figure per mu: the u field, the energy-density field, and the
    x-row-averaged profile that actually gets integrated. Overlaid on the fields
    are the boundary core (solid black), the far-field rows used for the baseline
    (dashed white, rolls read outside them), and a magenta box marking one
    fundamental x-period lambda_x = 2*pi/k1. The full x-extent is an integer
    number of these boxes, and the energy is averaged over all of them.
    """
    fields_dir = ensure_dir(fig_dir / "fields")
    recs = sorted(records, key=lambda r: float(r["mu"]))
    made = []
    for r in recs:
        mu = float(r["mu"]); alpha = float(r["alpha"])
        dx = float(r["dx"]); y = np.asarray(r["y"]); gb = int(r["gb_row"])
        u = np.asarray(r["u_last"]); e = np.asarray(r["e_last"])
        Ny, Nx = e.shape
        x = (np.arange(Nx) - Nx // 2) * dx
        k1 = np.sqrt(max(1e-12, 1.0 - mu**2)); k2 = max(mu, 1e-12)
        lam_x = 2.0 * np.pi / k1; lam_y = 2.0 * np.pi / k2
        n_xper = (x[-1] - x[0]) / lam_x
        e_bar = e.mean(axis=1); e_roll = float(r["e_roll"])
        n_edge = max(2, int(round(cfg.get("edge_frac", 0.12) * Ny)))
        extent = [x[0], x[-1], y[0], y[-1]]

        fig, axs = plt.subplots(1, 3, figsize=(13.5, 5.2), sharey=True,
                                gridspec_kw={"width_ratios": [1, 1, 0.55]})

        vmax = float(np.nanmax(np.abs(u))) + 1e-12
        im0 = axs[0].imshow(u, extent=extent, origin="lower", cmap="bwr",
                            aspect="auto", vmin=-vmax, vmax=vmax)
        fig.colorbar(im0, ax=axs[0], shrink=0.8)
        _draw_overlays(axs[0], x, y, gb, lam_x, n_edge)
        axs[0].set_title("u (final frame)"); axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")

        im1 = axs[1].imshow(e, extent=extent, origin="lower", cmap="viridis",
                            aspect="auto")
        fig.colorbar(im1, ax=axs[1], shrink=0.8)
        _draw_overlays(axs[1], x, y, gb, lam_x, n_edge)
        axs[1].set_title("energy density (final frame)"); axs[1].set_xlabel("x")

        axp = axs[2]
        axp.plot(e_bar, y, "k-", lw=1.3)
        axp.axvline(e_roll, color="r", ls="--", lw=1.0, label=r"$e_{roll}$")
        axp.fill_betweenx(y, e_roll, e_bar, where=(e_bar > e_roll),
                          color="orange", alpha=0.45, label="excess (integrated)")
        axp.axhline(y[gb], color="k", lw=1.0)
        axp.set_xlabel(r"$\bar e(y)$  (x-averaged)")
        axp.set_title("row average"); axp.legend(fontsize=7, loc="best")
        axp.grid(alpha=0.2)

        fig.suptitle(
            f"mu={mu:.3f}   alpha={np.degrees(alpha):.1f} deg   "
            f"lambda_x=2pi/k1={lam_x:.2f}   lambda_y=2pi/k2={lam_y:.2f}   "
            f"x-extent = {n_xper:.0f} periods   E_GB={float(r['E_GB']):.4g}",
            fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        p = fields_dir / f"field_mu{mu:.3f}.png"
        fig.savefig(p, dpi=140); plt.close(fig)
        made.append(p)
    return made


def make_integration_overview(records, cfg, fig_dir):
    """
    One montage of energy-density fields across the sweep, each with the boundary
    core and a magenta box marking one fundamental x-period. Quick way to see how
    the period and the boundary structure change with angle.
    """
    recs = sorted(records, key=lambda r: float(r["mu"]))
    n = len(recs)
    k = min(9, n)
    pick = np.linspace(0, n - 1, k).round().astype(int)
    ncol = min(3, k); nrow = int(np.ceil(k / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.5 * nrow),
                            squeeze=False)
    for slot in range(nrow * ncol):
        ax = axs[slot // ncol][slot % ncol]
        if slot >= k:
            ax.axis("off"); continue
        r = recs[pick[slot]]
        mu = float(r["mu"]); dx = float(r["dx"])
        y = np.asarray(r["y"]); gb = int(r["gb_row"])
        e = np.asarray(r["e_last"]); Ny, Nx = e.shape
        x = (np.arange(Nx) - Nx // 2) * dx
        k1 = np.sqrt(max(1e-12, 1.0 - mu**2)); lam_x = 2.0 * np.pi / k1
        n_edge = max(2, int(round(cfg.get("edge_frac", 0.12) * Ny)))
        ax.imshow(e, extent=[x[0], x[-1], y[0], y[-1]], origin="lower",
                  cmap="viridis", aspect="auto")
        _draw_overlays(ax, x, y, gb, lam_x, n_edge)
        ax.set_title(f"mu={mu:.2f}, a={np.degrees(np.arcsin(mu)):.0f} deg",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Energy field with fundamental x-period box (magenta) "
                 "and boundary core (black)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = fig_dir / "integration_regions_overview.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(cfg):
    out_root = ensure_dir(cfg["output_dir"])
    cache_dir = ensure_dir(out_root / "cache")
    fig_dir = ensure_dir(out_root / "figures")

    # The GB sits Ly/4 from both the y=0 stitch and the domain edge, so a crop
    # taller than Ly/2 would swallow the neighbouring boundary. Default to ~0.4*Ny
    # (odd -> core on centre row), leaving flat far-field rolls on each side.
    if cfg.get("crop_Ny") is None:
        Ny = int(round(cfg["Ny_factor"] * cfg["Nx"]))
        cn = int(0.4 * Ny)
        if cn % 2 == 0:
            cn += 1
        cfg["crop_Ny"] = cn
        print(f"Derived crop_Ny = {cn} (of Ny={Ny}) to isolate one GB.")

    mus = cfg["mus"]
    print(f"Sweeping {len(mus)} mu value(s) from {mus[0]:.3f} to {mus[-1]:.3f}")

    records = []
    for mu in mus:
        records.append(process_mu(mu, cfg, cache_dir))

    mu = np.array([float(r["mu"]) for r in records])
    alpha = np.array([float(r["alpha"]) for r in records])
    E = np.array([float(r["E_GB"]) for r in records])
    ndis = np.array([float(r["n_dislocations"]) for r in records])

    accept = acceptance_mask(records, cfg)
    for r, a in zip(records, accept):
        r["accepted"] = float(a)
    if (~accept).any():
        print("\n--- excluded from fits (resolution/baseline/flatness) ---")
        for r, a in zip(records, accept):
            if not a:
                print(f"  mu={float(r['mu']):.3f}  ppw_y={float(r['ppw_y']):.1f}"
                      f"  flatness={float(r['flatness']):.3f}"
                      f"  e_roll={float(r['e_roll']):.4f}"
                      f"  n_disl={float(r['n_dislocations']):g}")

    fits = fit_ansatze(mu, alpha, E, cfg, accept=accept, ndis=ndis)

    print("\n--- fit summary (accepted points only; dislocation-branch "
          "fits use accepted nucleated points) ---")
    print(f"  C  (sin^3)        = {fits['C_sin3']:.5g}")
    print(f"  C1 (1 - sin)      = {fits['C1_oms']:.5g}")
    print(f"  K  (cos)          = {fits['K_cos']:.5g}")
    print(f"  log-log slope     = {fits['loglog_slope']:.3f}   "
          f"(0.5 -> cos, 1.0 -> 1-sin)")
    print(f"  peak E_GB at      = {fits['peak_alpha_deg']:.2f} deg "
          f"(mu={fits['peak_mu']:.3f})")

    figs = make_plots(records, fits, cfg, fig_dir)

    figs.append(make_integration_overview(records, cfg, fig_dir))
    if cfg.get("field_figs", True):
        field_paths = make_field_figures(records, cfg, fig_dir)
        print(f"  wrote {len(field_paths)} per-mu field figures to "
              f"{fig_dir / 'fields'}")

    # summary table
    cols = ["mu", "alpha", "E_GB", "E_GB_prev", "e_roll", "flatness",
            "n_dislocations", "sum_abs", "ppw_y", "t_final", "accepted"]
    rows = np.column_stack([[float(r[c]) for r in records] for c in cols])
    np.savetxt(out_root / "energy_vs_angle.csv", rows, delimiter=",",
               header=",".join(cols), comments="")
    np.savez_compressed(out_root / "energy_vs_angle.npz",
                        **{c: rows[:, j] for j, c in enumerate(cols)},
                        **{f"fit_{k}": v for k, v in fits.items()
                           if np.isscalar(v)})
    print(f"\nWrote table + npz to {out_root}")
    for p in figs:
        print(f"  figure -> {p}")
    return records, fits


# ---------------------------------------------------------------------------
# Config / CLI
# ---------------------------------------------------------------------------

def cfg_from_args(a):
    mus = list(np.round(np.arange(a.mu_min, a.mu_max + 1e-9, a.mu_step), 6))
    return {
        "mus": mus,
        "Nx": a.Nx, "n_periods": a.n_periods, "Ny_factor": a.Ny_factor,
        "h": a.h, "tmax": a.tmax, "nsave": a.nsave,
        "t_save_window": a.t_save_window,
        "Rscale": a.Rscale, "amp": a.amp, "y_centering": a.y_centering,
        "which_gb": a.which_gb, "crop_Nx": a.crop_Nx, "crop_Ny": a.crop_Ny,
        "edge_frac": a.edge_frac, "band_periods": a.band_periods,
        "x_keep_frac": a.x_keep_frac, "defect_threshold": a.defect_threshold,
        "sin3_fit_mu_max": a.sin3_fit_mu_max,
        "accept_ppw_floor": a.accept_ppw_floor,
        "accept_flat_tol": a.accept_flat_tol,
        "accept_base_tol": a.accept_base_tol,
        "large_fit_mu_min": a.large_fit_mu_min,
        "loglog_mu_min": a.loglog_mu_min,
        "output_dir": a.output_dir, "overwrite": a.overwrite,
        "field_figs": not a.no_field_figs,
    }


def build_parser():
    p = argparse.ArgumentParser(
        description="Zigzag PGB line energy vs far-field angle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mu_min", type=float, default=0.40)
    p.add_argument("--mu_max", type=float, default=0.99)
    p.add_argument("--mu_step", type=float, default=0.01)
    p.add_argument("--Nx", type=int, default=256)
    p.add_argument("--n_periods", type=int, default=12)
    p.add_argument("--Ny_factor", type=float, default=4.0)
    p.add_argument("--h", type=float, default=0.25)
    p.add_argument("--tmax", type=float, default=1000.0)
    p.add_argument("--nsave", type=int, default=5)
    p.add_argument("--t_save_window", type=float, default=None)
    p.add_argument("--Rscale", type=float, default=1.0)
    p.add_argument("--amp", type=float, default=1.0)
    p.add_argument("--y_centering", type=str, default="node",
                   choices=["node", "cell"])
    p.add_argument("--which_gb", type=str, default="lower",
                   choices=["lower", "upper"])
    p.add_argument("--crop_Nx", type=int, default=None)
    p.add_argument("--crop_Ny", type=int, default=None)
    p.add_argument("--edge_frac", type=float, default=0.12)
    p.add_argument("--band_periods", type=float, default=2.0)
    p.add_argument("--x_keep_frac", type=float, default=0.6)
    p.add_argument("--defect_threshold", type=float, default=0.25)
    p.add_argument("--sin3_fit_mu_max", type=float, default=0.7)
    p.add_argument("--accept_ppw_floor", type=float, default=8.0)
    p.add_argument("--accept_flat_tol", type=float, default=0.05)
    p.add_argument("--accept_base_tol", type=float, default=0.15)
    p.add_argument("--large_fit_mu_min", type=float, default=0.85)
    p.add_argument("--loglog_mu_min", type=float, default=0.9)
    p.add_argument("--output_dir", type=str,
                   default="results/energy_vs_angle_zigzag_0830")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_field_figs", action="store_true",
                   help="Skip the per-mu field figures (still makes the overview).")
    return p


def main():
    run(cfg_from_args(build_parser().parse_args()))


# if __name__ == "__main__":
#     if len(sys.argv) == 1:
#         # in-code debug config (edit in PyCharm)
#         class _Args:
#             mu_min = 0.97
#             mu_max = 0.999
#             mu_step = 0.001
#             Nx = 512
#             n_periods = 12
#             Ny_factor = 4.0
#             h = 0.5
#             tmax = 250.0
#             nsave = 5
#             t_save_window = None
#             Rscale = 0.5
#             amp = 1.0
#             y_centering = "node"
#             which_gb = "lower"
#             crop_Nx = None
#             crop_Ny = None
#             edge_frac = 0.12
#             band_periods = 2.0
#             x_keep_frac = 0.6
#             defect_threshold = 0.25
#             sin3_fit_mu_max = 0.7
#             accept_ppw_floor = 8.0
#             accept_flat_tol = 0.05
#             accept_base_tol = 0.15
#             large_fit_mu_min = 0.85
#             loglog_mu_min = 0.9
#             output_dir = "results/energy_vs_angle_zigzag_finer_res_Rp5_tail_0929"
#             overwrite = False
#             no_field_figs = False
#         run(cfg_from_args(_Args()))
#     else:
#         main()



if __name__ == "__main__":
    if len(sys.argv) == 1:
        # in-code debug config (edit in PyCharm)
        class _Args:
            mu_min = 0.40
            mu_max = 0.99
            mu_step = 0.01
            #Nx = 512
            Nx = 768
            #n_periods = 12
            #n_periods = 6
            n_periods = 4
            Ny_factor = 4.0
            h = 0.5
            tmax = 250.0
            nsave = 5
            t_save_window = None
            Rscale = 0.5
            amp = 1.0
            y_centering = "node"
            which_gb = "lower"
            crop_Nx = None
            crop_Ny = None
            edge_frac = 0.12
            band_periods = 2.0
            x_keep_frac = 0.6
            defect_threshold = 0.25
            sin3_fit_mu_max = 0.7
            accept_ppw_floor = 8.0
            accept_flat_tol = 0.05
            accept_base_tol = 0.15
            large_fit_mu_min = 0.85
            loglog_mu_min = 0.9
            output_dir = "results/energy_vs_angle_zigzag_0929_np4_full"
            overwrite = False
            no_field_figs = False
        run(cfg_from_args(_Args()))
    else:
        main()