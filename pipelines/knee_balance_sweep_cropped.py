"""
knee_balance_sweep.py  (v1.2)
-----------------------------
Physical-balance diagnostics of the self-dual knee, per (run, route),
decoupled from any sparse fitting. Companion to self_dual_sindy_sweep.py
(same discovery, routes, profiles; knee_sweep_common.py). In the
pattern of knee_bend_fits_and_residuals.py, reduced to the x-averaged
profile where the knee lives.

    python knee_balance_sweep.py --roots <dir> [<dir> ...]
        [--mu-min --mu-max --R --routes --no-validate --out-root
         --method-dirs ...]

Per (file, route) along the profile y:
  bending density     E_bend(y) = eta (d k2bar/dy)^2
  stretching density  E_str(y)  = alpha G^2(wbar),
                      G^2 = w^2 - w^4/(2R)  (exact SH constitutive form,
                      shifted variable; w^2 - w^4 at R = 1/2)
  with the corrected coefficients alpha = eta = 2R/3 (half convention,
  session-6 handoff item 1), their difference, the pointwise relative
  error |E_bend - E_str| / (E_bend + E_str), and the RMS of the
  global-scaled relative error |diff| / (E_bend + E_str + C), C the
  larger of the two means on the band (knee_bend_fits convention).

Balance-map lesson (handoff item 2): a residual map at fixed (alpha,
eta) tests only whether sqrt(alpha/eta) equals the field's own
effective coefficient. So every panel is drawn twice: at the nominal
pair (beta = 1) and at the effective pair eta_eff = alpha / beta_fit^2,
where beta_fit is the slope of div k = b + c w fitted first on the
scatter (panel d). The nominal residual is the physics statement; the
effective residual is the instrument's self-consistency.

Validation (--validate, on by default, knees only): a planted beta = 1
knee on each file's grid is run through every route so the nominal
balance has a reference. On demod the nominal RMS relative error
should be at the percent level; on the smoothed routes it is the
route's instrument residual, recorded, not judged.

Outputs: one tidy CSV (mu, R, route, beta_fit, holdout-free R^2 of the
linear fit, RMS rel. error nominal and effective, peak ratio
E_bend/E_str at the core, band, w_ff, synthetic reference, stem); a
four-panel figure per (stem, route); a summary of RMS relative error
against mu per route.

Changelog:
v1.2  --x-window trim (integer-period trim, no taper; knee_sweep_common
      v1.2); the estimator of record still sees the archived frame.
v1.1  demod_sm route in the default list; line plots broken at the seam;
      np.trapz / np.trapezoid fallback.
v1.0  first version; preprocessing and its measured justifications in
      knee_sweep_common.py v1.0.
Python 3.8 compatible; frozen method layer imported, never edited.
"""
import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import knee_sweep_common_cropped as ksc  # noqa: E402

# np.trapz (old numpy) / np.trapezoid (numpy >= 2): lazy getattr fallback
_trapz = getattr(np, "trapz", None) or getattr(np, "trapezoid")


# ---------------------------------------------------------------------
# balance on one profile
# ---------------------------------------------------------------------

def balance_one(u, x, y, jc, mu, R, route, cfg):
    k1, k2 = ksc.extract_k(route, u, x, y, jc, cfg["alpha_x"], cfg["y_frac"])
    prof = ksc.build_profile(k1, k2, x, y, jc, mu, route, cfg["alpha_x"],
                             cfg["y_frac"], cfg["band_w"], cfg["ff_renorm"])
    b = prof["band"]
    w, d, yy = prof["wbar"][b], prof["dk2"][b], prof["y"][b]
    c, b0, r2 = ksc.linear_fit(w, d)          # d = b0 + c w
    beta_fit = -c
    alpha = eta = 2.0 * R / 3.0
    G2 = w ** 2 - w ** 4 / (2.0 * R)
    out = dict(prof=prof, w=w, divk=d, y=yy, beta_fit=beta_fit, b0=b0,
               r2_lin=r2, alpha=alpha, eta=eta, G2=G2)
    for tag, eta_use in (("nominal", eta), ("effective", alpha / beta_fit ** 2)):
        E_b = eta_use * d ** 2
        E_s = alpha * G2
        diff = E_b - E_s
        rel = np.abs(diff) / (E_b + E_s + 1e-14)
        C = max(E_b.mean(), E_s.mean())
        rel_g = np.abs(diff) / (E_b + E_s + C + 1e-14)
        jc_ = int(np.argmin(np.abs(yy)))
        out[tag] = dict(eta=eta_use, E_bend=E_b, E_str=E_s, diff=diff, rel=rel,
                        rel_global=rel_g, rms_rel_global=float(np.sqrt(np.mean(rel_g ** 2))),
                        rms_diff=float(np.sqrt(np.mean(diff ** 2))),
                        peak_ratio=float(E_b[jc_] / max(E_s[jc_], 1e-14)),
                        int_bend=float(_trapz(E_b, yy)), int_str=float(_trapz(E_s, yy)))
    return out


# ---------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------

def _break_at_gaps(yy, *arrays):
    """Insert NaN where the band skips rows (the seam around the core)
    so line plots do not bridge the excluded rows."""
    order = np.argsort(yy)
    ys = yy[order]
    dy = np.median(np.diff(ys))
    gaps = np.where(np.diff(ys) > 1.5 * dy)[0]
    out = []
    for a in (ys,) + tuple(np.asarray(a)[order] for a in arrays):
        out.append(np.insert(a.astype(float), gaps + 1, np.nan))
    return out


def fig_balance(out_path, res, route, mu, R, cfg, title_extra=""):
    yy, w, d = res["y"], res["w"], res["divk"]
    nom = dict(res["nominal"]); eff = dict(res["effective"])
    keys = ["E_bend", "E_str", "diff", "rel", "rel_global"]
    yy_b, nom["E_bend"], nom["E_str"], nom["diff"], nom["rel"], nom["rel_global"] = \
        _break_at_gaps(yy, *[res["nominal"][k] for k in keys])
    _, eff["E_bend"], eff["E_str"], eff["diff"], eff["rel"], eff["rel_global"] = \
        _break_at_gaps(yy, *[res["effective"][k] for k in keys])
    yy_scatter = yy
    yy = yy_b
    col = ksc.ROUTE_COLORS[route]
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    ax = axs[0, 0]
    ax.plot(yy, nom["E_str"], "-", color="C1", lw=1.6, label=r"$\alpha\,G^2$")
    ax.plot(yy, nom["E_bend"], "-", color="C0", lw=1.6,
            label=r"$\eta\,(\nabla\!\cdot\!\mathbf{k})^2$, $\eta=\alpha=2R/3$")
    ax.plot(yy, eff["E_bend"], "--", color="C0", lw=1.2,
            label=r"$\eta_{\rm eff}\,(\nabla\!\cdot\!\mathbf{k})^2$, $\eta_{\rm eff}=\alpha/\beta_{\rm fit}^2$")
    ax.set_xlabel("$y$"); ax.set_ylabel("energy density")
    ax.set_title("bending vs stretching", fontsize=10); ax.legend(fontsize=7)
    ksc.panel_label(ax, "(a)")
    ax = axs[0, 1]
    ax.axhline(0, color="k", lw=0.6)
    ax.plot(yy, nom["diff"], "-", color="C0", lw=1.4,
            label=r"nominal (RMS %.2e)" % nom["rms_diff"])
    ax.plot(yy, eff["diff"], "--", color="C0", lw=1.2,
            label=r"effective (RMS %.2e)" % eff["rms_diff"])
    ax.set_xlabel("$y$"); ax.set_ylabel(r"$\eta(\nabla\!\cdot\!\mathbf{k})^2 - \alpha G^2$")
    ax.set_title("difference", fontsize=10); ax.legend(fontsize=7)
    ksc.panel_label(ax, "(b)")
    ax = axs[1, 0]
    ax.plot(yy, nom["rel_global"], "-", color="C0", lw=1.4,
            label="nominal, global-scaled (RMS %.3f)" % nom["rms_rel_global"])
    ax.plot(yy, eff["rel_global"], "--", color="C0", lw=1.2,
            label="effective, global-scaled (RMS %.3f)" % eff["rms_rel_global"])
    ax.plot(yy, np.clip(nom["rel"], 0, 1), ":", color="0.5", lw=1.0,
            label="nominal, pointwise (clipped at 1)")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("$y$"); ax.set_ylabel("relative error")
    ax.set_title("relative error", fontsize=10); ax.legend(fontsize=7)
    ksc.panel_label(ax, "(c)")
    ax = axs[1, 1]
    ax.plot(w, d, ".", ms=3, color="0.5", alpha=0.7, label="data")
    wg = np.linspace(min(w.min(), 0), max(w.max(), 0), 100)
    ax.plot(wg, res["b0"] - res["beta_fit"] * wg, "-", color=col, lw=1.6,
            label=r"fit: $\nabla\!\cdot\!\mathbf{k} = %+.4f - %.3f\,w$ ($R^2=%.4f$)"
                  % (res["b0"], res["beta_fit"], res["r2_lin"]))
    ax.plot(wg, -1.0 * wg, ":", color="k", lw=1.0, label=r"$\beta=1$ ($\alpha=\eta$)")
    ax.set_xlabel(r"$w = |\mathbf{k}|^2 - |\mathbf{k}|_{ff}^2$")
    ax.set_ylabel(r"$\nabla\!\cdot\!\mathbf{k}$")
    ax.set_title("self-dual balance scatter", fontsize=10); ax.legend(fontsize=7)
    ksc.panel_label(ax, "(d)")
    fig.suptitle(r"%s   $\mu=%g$  $R=%g$   $\beta_{\rm fit}=%.3f$   %s"
                 % (ksc.ROUTES[route]["label"], mu, R, res["beta_fit"], title_extra),
                 fontsize=10)
    fig.savefig(out_path); plt.close(fig)


def fig_summary(out_dir, rows, cfg):
    routes = cfg["routes"]
    Rvals = sorted({r["R"] for r in rows})
    markers = {0.5: "o", 1.0: "s"}
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for route in routes:
        for R in Rvals:
            sel = [r for r in rows if r["route"] == route and r["R"] == R and not r["synthetic"]]
            syn = [r for r in rows if r["route"] == route and r["R"] == R and r["synthetic"]]
            if not sel:
                continue
            mus = [r["mu"] for r in sel]; mk = markers.get(R, "^")
            axs[0].plot(mus, [r["rms_rel_global_nominal"] for r in sel], mk + "-",
                        color=ksc.ROUTE_COLORS[route], label="%s, $R=%g$" % (route, R))
            if syn:
                axs[0].plot([r["mu"] for r in syn], [r["rms_rel_global_nominal"] for r in syn],
                            mk, mfc="none", mec=ksc.ROUTE_COLORS[route], ms=7,
                            label="%s synthetic $\\beta=1$, $R=%g$" % (route, R))
            axs[1].plot(mus, [r["rms_rel_global_effective"] for r in sel], mk + "-",
                        color=ksc.ROUTE_COLORS[route], label="%s, $R=%g$" % (route, R))
            axs[2].plot(mus, [r["beta_fit"] for r in sel], mk + "-",
                        color=ksc.ROUTE_COLORS[route], label="%s, $R=%g$" % (route, R))
    axs[0].set_title(r"RMS rel. error, nominal $\alpha=\eta=2R/3$", fontsize=10)
    axs[1].set_title(r"RMS rel. error, effective $\eta=\alpha/\beta_{\rm fit}^2$", fontsize=10)
    axs[2].set_title(r"$\beta_{\rm fit}$ (slope of the scatter)", fontsize=10)
    axs[2].axhline(1.0, color="k", lw=0.5, ls=":")
    for ax in axs:
        ax.set_xlabel(r"$\mu$"); ax.legend(fontsize=6)
    fig.savefig(Path(out_dir) / "summary_balance_vs_mu.png"); plt.close(fig)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main(cfg):
    ksc.setup_method_layer(cfg.get("method_dirs", ()))
    ksc.use_house_style()
    runs = ksc.discover_runs(cfg["roots"], cfg["pattern"], cfg["mu_min"],
                             cfg["mu_max"], cfg["R_values"])
    if not runs:
        print("No runs found under", cfg["roots"]); return []
    cfg["R_values"] = sorted({r["R"] for r in runs}) if not cfg["R_values"] else cfg["R_values"]
    cfg["mu_min"] = min(r["mu"] for r in runs) if cfg["mu_min"] is None else cfg["mu_min"]
    cfg["mu_max"] = max(r["mu"] for r in runs) if cfg["mu_max"] is None else cfg["mu_max"]
    out_dir = Path(cfg["out_root"]) / ksc.config_stamp(cfg, "kbal")
    out_dir.mkdir(parents=True, exist_ok=True)
    ksc.save_config(out_dir, cfg)
    print("output ->", out_dir)

    rows = []
    print("%6s %4s %-9s %5s | %8s %7s | %9s %9s | %8s %8s | %s" % (
        "mu", "R", "route", "synth", "beta_fit", "R2lin", "rms_nom", "rms_eff",
        "peak_nom", "int_b/s", "band"))
    for run in runs:
        x, y, u, fidx = ksc.load_frame(run["path"], cfg["frame"])
        mu, R, jc = run["mu"], run["R"], run["jc"]
        stem_dir = out_dir / run["stem"]; stem_dir.mkdir(exist_ok=True)
        u, x, alpha_eff, trim = ksc.window_frame(u, x, mu, cfg["x_window"],
                                                 cfg["alpha_x"], cfg["max_seam_jump"])
        cfg_run = dict(cfg, alpha_x=alpha_eff)
        if trim is not None:
            print("       x window: trimmed to %d of %d columns (%d periods, seam jump %.3f rad)"
                  % (trim["n_cols"], trim["n_cols_in"], trim["n_periods"], trim["jump_rad"]))
        cases = []
        if cfg["validate"]:
            cases.append((True, ksc.synth_knee_on_grid(x, y, mu, R, 1.0, jc)))
        cases.append((False, u))
        for is_syn, uu in cases:
            for route in cfg["routes"]:
                res = balance_one(uu, x, y, jc, mu, R, route, cfg_run)
                nom, eff = res["nominal"], res["effective"]
                row = dict(mu=mu, R=R, route=route, synthetic=is_syn,
                           beta_fit=res["beta_fit"], intercept=res["b0"], r2_lin=res["r2_lin"],
                           alpha=res["alpha"], eta_nominal=res["eta"], eta_effective=eff["eta"],
                           rms_rel_global_nominal=nom["rms_rel_global"],
                           rms_rel_global_effective=eff["rms_rel_global"],
                           rms_diff_nominal=nom["rms_diff"], rms_diff_effective=eff["rms_diff"],
                           peak_ratio_nominal=nom["peak_ratio"],
                           int_ratio_nominal=nom["int_bend"] / max(nom["int_str"], 1e-14),
                           w_ff=res["prof"]["w_ff"], band_half=res["prof"]["band_half"],
                           n_rows=res["prof"]["n_rows"], frame_index=fidx,
                           stem=run["stem"], path=run["path"])
                rows.append(row)
                print("%6.3f %4.1f %-9s %5s | %8.4f %7.4f | %9.4f %9.4f | %8.3f %8.3f | %.1f"
                      % (mu, R, route, "yes" if is_syn else "no", res["beta_fit"], res["r2_lin"],
                         nom["rms_rel_global"], eff["rms_rel_global"], nom["peak_ratio"],
                         row["int_ratio_nominal"], res["prof"]["band_half"]))
                if not is_syn or cfg["synth_figures"]:
                    fig_balance(stem_dir / ("%s_%sbalance_four_panel.png"
                                            % (route, "synthetic_" if is_syn else "")),
                                res, route, mu, R, cfg, "synthetic $\\beta=1$" if is_syn else "")
    csv_path = out_dir / "knee_balance_sweep.csv"
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    fig_summary(out_dir, rows, cfg)
    print("\n== balance summary (data frames) ==")
    for route in cfg["routes"]:
        sel = [r for r in rows if r["route"] == route and not r["synthetic"]]
        syn = [r for r in rows if r["route"] == route and r["synthetic"]]
        print("   %-9s beta_fit %.3f .. %.3f; RMS rel. error nominal %.3f .. %.3f, effective "
              "%.3f .. %.3f%s" % (
                  route, min(r["beta_fit"] for r in sel), max(r["beta_fit"] for r in sel),
                  min(r["rms_rel_global_nominal"] for r in sel), max(r["rms_rel_global_nominal"] for r in sel),
                  min(r["rms_rel_global_effective"] for r in sel), max(r["rms_rel_global_effective"] for r in sel),
                  ("; synthetic beta=1 nominal %.3f .. %.3f" % (
                      min(r["rms_rel_global_nominal"] for r in syn),
                      max(r["rms_rel_global_nominal"] for r in syn))) if syn else ""))
    print("\nwrote", csv_path)
    return rows


class _Args:
    roots = [
        str(_HERE / "data" / "sh_pgb_zigzag" / "mu_sweeps_shallow_v2_Rp5_Nx512_tmax500_hp5_np18_Ny5" / "raw" / "selected"),
        str(_HERE / "data" / "sh_pgb_zigzag" / "mu_sweeps_shallow_v2_R1p0_Nx512_tmax500_hp5_np18_Ny5" / "raw" / "selected")]
    pattern = "*sh_pgb_zigzag*mu*.npz"
    mu_min = None
    mu_max = None
    R_values = []
    routes = ["uhu", "riesz", "riesz_pi2", "demod", "demod_sm"]
    domain = "profile"
    frame = -1
    validate = True
    synth_figures = False
    x_window = "tukey"      # 'tukey' (run of record) or 'trim' (integer periods, no taper)
    max_seam_jump = 0.1
    alpha_x = 0.5
    y_frac = 0.06
    band_w = 3.0
    ff_renorm = True
    out_root = str(_HERE / "results" / "knee_balance_sweep_v2")
    method_dirs = []


def parse_cli():
    d = _Args
    p = argparse.ArgumentParser(description="knee self-dual balance diagnostics per route")
    p.add_argument("--roots", nargs="+", default=d.roots)
    p.add_argument("--pattern", default=d.pattern)
    p.add_argument("--mu-min", type=float, default=d.mu_min)
    p.add_argument("--mu-max", type=float, default=d.mu_max)
    p.add_argument("--R", nargs="*", type=float, default=d.R_values)
    p.add_argument("--routes", nargs="+", default=d.routes, choices=list(ksc.ROUTES))
    p.add_argument("--frame", type=int, default=d.frame)
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--synth-figures", action="store_true")
    p.add_argument("--no-ff-renorm", action="store_true")
    p.add_argument("--x-window", default=d.x_window, choices=["tukey", "trim"])
    p.add_argument("--max-seam-jump", type=float, default=d.max_seam_jump)
    p.add_argument("--alpha-x", type=float, default=d.alpha_x)
    p.add_argument("--band-w", type=float, default=d.band_w)
    p.add_argument("--out-root", default=d.out_root)
    p.add_argument("--method-dirs", nargs="*", default=d.method_dirs)
    a = p.parse_args()
    return dict(roots=a.roots, pattern=a.pattern, mu_min=a.mu_min, mu_max=a.mu_max,
                R_values=a.R, routes=a.routes, domain=d.domain, frame=a.frame,
                validate=not a.no_validate, synth_figures=a.synth_figures,
                x_window=a.x_window, max_seam_jump=a.max_seam_jump,
                alpha_x=a.alpha_x, y_frac=d.y_frac, band_w=a.band_w,
                ff_renorm=not a.no_ff_renorm, out_root=a.out_root,
                method_dirs=a.method_dirs)


def cfg_from_args():
    return {k: getattr(_Args, k) for k in vars(_Args) if not k.startswith("_")}


if __name__ == "__main__":
    main(parse_cli() if len(sys.argv) > 1 else cfg_from_args())
