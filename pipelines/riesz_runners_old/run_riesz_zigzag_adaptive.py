# pipelines/run_riesz_zigzag_adaptive_v2.py
"""
Adaptive-width Riesz extraction (v2), a thin wrapper over run_riesz_zigzag.

It adds one new director mode, "adaptive", that replaces the global small/large
width gate (select_sigma_riesz / the defect_aware branch) with a per-point width
driven by the loop-persistence field. Everything else, loading, seeding, phase
reconstruction, figures, GIFs, npz layout, is REUSED from run_riesz_zigzag with
no changes to that file: this runner imports the base module and monkeypatches
only extract_one_frame, whose body below is verbatim from the base except the
director-selection block (clearly marked NEW).

Direction (matches select_sigma_riesz): where the local loop-persistence is
high the width goes to sigma_large; where it is zero it stays at sigma_fixed;
the doubled-angle directors of the two widths are blended per point. On a clean
knee the weight is zero everywhere, so this reduces exactly to the fixed small
width and touches nothing.

Two ways to run:
  1. Validate first (no simulation data needed):
       python run_riesz_zigzag_adaptive_v2.py --validate
     writes results/adaptive_director_val.png and prints the knee/PN check.
  2. Full extraction: set director="adaptive" in the Args block (or pass
     --director adaptive via the base CLI is NOT wired; use the in-code Args
     path below, matching the base runner's convention).

New cfg knobs (defaults match the analytic bench):
  loop_half_sizes    loop scales for persistence      (1,2,3,4,6,8)
  persist_smooth_px  spatial smoothing of the weight  1.5
  persist_gamma      ramp sharpness toward cores      1.0
"""

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

# the base runner; everything not touched here comes from it unchanged
import run_riesz_zigzag as base
from run_riesz_zigzag import (
    detect_defects, riesz_k_field, riesz_director, macro_iso, spectral_deriv,
    compute_uhu_ops_smoother, compute_pgb_phase_from_uhu, rebuild_grid,
    k_from_theta, k_consistency, monogenic, reflect_phase_y, trace_crest_contour,
)
from op_extract.loop_persistence import adaptive_director


def extract_one_frame_adaptive(u_frame, sh, ramp, cfg, crest_state=None):
    """
    One frame: detect, pick width, direct k, phase, grad(theta) k, mutual map.

    crest_state is a per-series mutable dict carrying the crest anchor
    across frames. Seed contour resolution when seed_contour="crest":
    anchored trace, then a fresh trace, then the analytic template, in that
    order, each fallback reported. The gauge follows the contour when
    gauge="auto": zero_origin on a crest (exact, theta constant along a
    crest), seed_phase on the template (needs the analytic reference).

    Returns a dict of 2d fields and scalars, plus the detection record for the
    figure. Never raises on a single-frame failure; on error it returns NaN
    fields and marks ok=False so the series keeps going.
    """
    x, y = sh["x"], sh["y"]
    Lx, Ly = sh["Lx"], sh["Ly"]
    mu = sh["mu"]

    band_y = cfg["band_y"]
    band_x = (1.0 - cfg["band_x_frac"]) * 0.5 * Lx  # keep the inner (1-frac)

    u_ramped = u_frame * ramp
    nan2d = np.full_like(u_frame, np.nan, dtype=float)

    try:
        # 1. detection and width selection on the ramped field
        det = detect_defects(u_ramped, x, y, Lx, Ly,
                             sigma_det=cfg["sigma_det"],
                             band_y=band_y, band_x=band_x)
        if cfg["director"] == "adaptive":
            # ---- NEW: spatial width modulator (loop-persistence blend) ----
            # Blend the small (clean-knee) and large (defect) director widths
            # per point, driven by the local loop-persistence of the small-width
            # director. Direction only; the magnitude / comparison path (kd) is
            # read at the large far-field width, per the agreed first version.
            ad = adaptive_director(
                u_ramped, Lx, Ly, riesz_director,
                sigma_small=cfg["sigma_fixed"], sigma_large=cfg["sigma_large"],
                half_sizes=tuple(cfg.get("loop_half_sizes", (1, 2, 3, 4, 6, 8))),
                smooth_px=cfg.get("persist_smooth_px", 1.5),
                gamma=cfg.get("persist_gamma", 1.0),
            )
            d1a = np.asarray(ad["d1"])
            d2a = np.asarray(ad["d2"])
            kd = riesz_k_field(u_ramped, x, y, Lx, Ly, cfg["sigma_large"])
            wgt = np.asarray(ad["weight"])
            sigma_dir = float(cfg["sigma_fixed"] + float(np.mean(wgt))
                              * (cfg["sigma_large"] - cfg["sigma_fixed"]))
        else:
            if cfg["defect_aware"] and det["sum_abs"] > 0:
                sigma_dir = cfg["sigma_large"]
            else:
                sigma_dir = cfg["sigma_fixed"]

            # 2. direct Riesz k field and amplitude at the selected width
            kd = riesz_k_field(u_ramped, x, y, Lx, Ly, sigma_dir)
            use_uhu = (cfg["director"] == "uhu"
                       or (cfg["director"] == "aware" and det["sum_abs"] > 0))
            if use_uhu:
                # A/B test: uHu eigenvector director through the identical
                # seeding and reconstruction. uHu needs sigma |k| of order
                # one, so its width is floored at sigma_uhu_min even on
                # clean frames where Riesz runs at 0.4.
                sg_u = max(float(sigma_dir), float(cfg["sigma_uhu_min"]))
                sm = lambda a, s_=sg_u: macro_iso(a, s_, Lx, Ly)
                ops = compute_uhu_ops_smoother(u_frame, x, y, sm,
                                               spectral_deriv, ramp)
                ku1 = np.asarray(ops["k1_orig"])
                ku2 = np.asarray(ops["k2_orig"])
                nrm = np.hypot(ku1, ku2)
                nrm = np.where(nrm > 0, nrm, 1.0)
                d1a, d2a = ku1 / nrm, ku2 / nrm
            else:
                d1a = np.asarray(kd["d1"])
                d2a = np.asarray(kd["d2"])

        # 2b. seed contour: data-driven crest by default, template fallback.
        # The trace uses RAW u (the crest is a feature of the true pattern)
        # and the director at the selected width.
        contour_source = "template"
        knee_bdry = None
        if crest_state is None:
            crest_state = {}
        if cfg["seed_contour"] == "crest":
            kw = dict(y_start=cfg["crest_y_start"], y_stop=cfg["crest_y_stop"],
                      x_margin=cfg["xmargin"] * Lx + 1.0,
                      y_margin=cfg["ymargin"] * Ly + 1.0, ds=cfg["ds"])
            con, p0 = trace_crest_contour(u_frame, x, y, d1a, d2a,
                                          anchor=crest_state.get("anchor"),
                                          **kw)
            if con is None and crest_state.get("anchor") is not None:
                print("    [warn] anchored crest trace failed; "
                      "retrying from a fresh start")
                con, p0 = trace_crest_contour(u_frame, x, y, d1a, d2a,
                                              anchor=None, **kw)
            if con is not None:
                knee_bdry = con
                contour_source = "crest"
                crest_state["anchor"] = p0
            else:
                print("    [warn] crest trace failed its gate; "
                      "falling back to the analytic template this frame")

        gauge = cfg["gauge"]
        if gauge == "auto":
            gauge = "zero_origin" if contour_source == "crest" \
                else "seed_phase"

        # 3. phase pipeline with the Riesz director injected, RAW u sampled
        d1 = d1a[:, :, None]
        d2 = d2a[:, :, None]
        uhu_data = {
            "x": x, "y": y, "u": u_frame[:, :, None],
            "k1_sym": d1, "k2_sym": d2, "k1_orig": d1, "k2_orig": d2,
            "mu": mu, "tt": None, "ramp": None, "k": None, "A": None,
            "knee_bdry": knee_bdry,  # None -> analytic knee from mu
            "source_path": sh["source_path"],
        }
        res = compute_pgb_phase_from_uhu(
            uhu_data, mu=mu,
            n_phase_seeds=cfg["n_phase_seeds"],
            seed_half=cfg["seed_half"],
            ds=cfg["ds"], max_steps=cfg["max_steps"],
            prefer_sym=True, phase_ramp_mode="none",
            grid_output=False,
        )
        # Rebuild from the seed-to-core branch of each line, the way the t1
        # harness does. The pipeline's own stitched grid blends the reflected
        # half with the wrong sign and flattens the core; that is what made the
        # core profile flat. seed_phase fixes each line's constant from the
        # analytic far-field knee phase at its seed. odd_in_y=False triggers the
        # mirror that keeps the y=0 core row inside the interpolation hull.
        fld = {"x": x, "y": y, "theta": sh["theta_ref"], "odd_in_y": False}
        theta = rebuild_grid(res, fld, gauge=gauge, half="orig",
                             band=cfg["ds"])
        # the upper arm is the only one traced, so define the lower half by
        # reflection across the core (even for the knee)
        theta = reflect_phase_y(theta, sh["core_row"], mode=cfg["reflect_y"])

        # 4. k from grad(theta), then the truth-free mutual comparison
        kg = k_from_theta(theta, x, y, Lx, Ly, smooth=cfg["grad_smooth"])
        cons = k_consistency(kd, kg)

        # 5. amplitude observables. kd["A"] is sqrt(2 <u^2>) at sigma_avg
        # (of order pi) and barely registers cores: measured 2 percent core
        # dips at the final frame against 23 percent for the pointwise
        # monogenic amplitude. A_mono = sqrt(u^2 + R1^2 + R2^2) is the
        # nucleation field: it dips at cores but stays smooth and nonzero.
        # A_min_band tracks its band minimum; A_far_median (same field, a
        # far-field band) is saved so the normalized dip
        # A_min_band / A_far_median is available at analysis time, immune
        # to the overall amplitude relaxation of the run.
        mono = monogenic(u_ramped, Lx, Ly)
        A_mono = np.asarray(mono["A"])
        X, Y = np.meshgrid(x, y)
        band = (np.abs(Y) < band_y) & (np.abs(X) < band_x)
        far = ((np.abs(Y) > 3.0 * band_y) & (np.abs(Y) < 6.0 * band_y)
               & (np.abs(X) < band_x))
        A_min_band = float(np.nanmin(A_mono[band])) if band.any() else np.nan
        A_far_median = float(np.nanmedian(A_mono[far])) if far.any() \
            else np.nan

        return {
            "ok": True, "det": det, "sigma_dir": float(sigma_dir),
            "contour_source": contour_source, "gauge": gauge,
            "theta": theta.astype(np.float32),
            "k1_direct": np.asarray(kd["k1"], dtype=np.float32),
            "k2_direct": np.asarray(kd["k2"], dtype=np.float32),
            "kmag_direct": np.asarray(kd["kmag"], dtype=np.float32),
            "A": np.asarray(kd["A"], dtype=np.float32),
            "A_mono": A_mono.astype(np.float32),
            "k1_grad": np.asarray(kg["k1"], dtype=np.float32),
            "k2_grad": np.asarray(kg["k2"], dtype=np.float32),
            "kmag_grad": np.asarray(kg["kmag"], dtype=np.float32),
            "mutual_dang": np.asarray(cons["dang"], dtype=np.float32),
            "sum_abs": int(det["sum_abs"]),
            "n_dislocations": int(det["n_dislocations"]),
            "A_min_band": A_min_band,
            "A_far_median": A_far_median,
            "mutual_median": float(cons.get("dang_med_abs", np.nan)),
            "knee_bdry_phase": np.asarray(res.get("knee_bdry_phase")),
            "coordinate_lines": res["coordinate_lines"][0],
            "charges": [[float(c[0]), float(c[1]), int(c[2])]
                        for c in det["charges"]],
        }
    except Exception as exc:  # keep the series alive; report the frame
        print(f"    [warn] frame extraction failed: {exc}")
        return {
            "ok": False, "det": None, "sigma_dir": np.nan,
            "contour_source": "failed", "gauge": "none",
            "theta": nan2d.astype(np.float32),
            "k1_direct": nan2d.astype(np.float32),
            "k2_direct": nan2d.astype(np.float32),
            "kmag_direct": nan2d.astype(np.float32),
            "A": nan2d.astype(np.float32),
            "A_mono": nan2d.astype(np.float32),
            "k1_grad": nan2d.astype(np.float32),
            "k2_grad": nan2d.astype(np.float32),
            "kmag_grad": nan2d.astype(np.float32),
            "mutual_dang": nan2d.astype(np.float32),
            "sum_abs": -1, "n_dislocations": -1,
            "A_min_band": np.nan, "A_far_median": np.nan,
            "mutual_median": np.nan,
            "knee_bdry_phase": None, "coordinate_lines": None, "charges": [],
        }


# ---------------------------------------------------------------------------
# install the adaptive extractor in place of the base one. run_riesz_series
# calls extract_one_frame by bare name, so this override propagates without
# editing run_riesz_zigzag.py.
# ---------------------------------------------------------------------------
base.extract_one_frame = extract_one_frame_adaptive


# ---------------------------------------------------------------------------
# analytic validation (self-contained; no simulation data)
# ---------------------------------------------------------------------------

def _make_grid(mu, n_xper=4, n_yper=6, ppw=28):
    k1 = np.sqrt(1.0 - mu**2); k2 = mu
    dx = 2.0 * np.pi / ppw
    Lx = n_xper * (2 * np.pi / max(k1, 0.2)); Ly = n_yper * (2 * np.pi / k2)
    Nx = int(round(Lx / dx)); Ny = int(round(Ly / dx))
    x = (np.arange(Nx) - Nx // 2) * dx; y = (np.arange(Ny) - Ny // 2) * dx
    X, Y = np.meshgrid(x, y)
    return dict(x=x, y=y, X=X, Y=Y, Lx=x[-1] - x[0], Ly=y[-1] - y[0], k1=k1, k2=k2)


def _field_knee(g):
    pp = g["k1"] * g["X"] + g["k2"] * g["Y"]; pm = g["k1"] * g["X"] - g["k2"] * g["Y"]
    m = np.maximum(pp, pm)
    return np.cos(m + np.log(np.exp(pp - m) + np.exp(pm - m)))


def _field_pn(g, n_def=4):
    from op_extract.pn_field import pismen_on_sh_grid
    Lx = g["x"][-1] - g["x"][0]
    xd = np.linspace(-0.3 * Lx, 0.3 * Lx, n_def)
    return pismen_on_sh_grid(g["X"], g["Y"], xd, 1.0)[1]


def validate(mu=0.90, band_y=3.0, out="results/adaptive_director_val.png",
             sigma_small=0.4, sigma_large=float(np.pi)):
    """Prove the modulator on the analytic knee (weight 0, untouched) and PN
    (weight only at cores) before it touches simulation data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    g = _make_grid(mu)
    fields = {"knee": _field_knee(g), "pn": _field_pn(g)}
    fig, axs = plt.subplots(2, 5, figsize=(19, 7))
    ext = [g["x"][0], g["x"][-1], g["y"][0], g["y"][-1]]
    band = np.abs(g["y"]) < band_y
    for r, (nm, u) in enumerate(fields.items()):
        o = adaptive_director(u, g["Lx"], g["Ly"], riesz_director,
                              sigma_small=sigma_small, sigma_large=sigma_large)
        d = (o["angle"] - o["angle_small"] + np.pi / 2) % np.pi - np.pi / 2
        dchg = np.degrees(np.abs(d))
        axs[r, 0].imshow(u, extent=ext, origin="lower", cmap="gray", vmin=-1,
                         vmax=1, aspect="auto"); axs[r, 0].set_ylabel(nm)
        axs[r, 1].imshow(np.degrees(o["angle_small"]), extent=ext, origin="lower",
                         cmap="twilight", vmin=-90, vmax=90, aspect="auto")
        axs[r, 2].imshow(o["weight"], extent=ext, origin="lower", cmap="magma",
                         vmin=0, vmax=1, aspect="auto")
        axs[r, 3].imshow(np.degrees(o["angle"]), extent=ext, origin="lower",
                         cmap="twilight", vmin=-90, vmax=90, aspect="auto")
        axs[r, 4].imshow(dchg, extent=ext, origin="lower", cmap="inferno",
                         vmin=0, vmax=20, aspect="auto")
        print(f"{nm:5s}: weight max={o['weight'].max():.3f}  "
              f"band-mean={o['weight'][band].mean():.3f}  "
              f"dir change deg (band max)={dchg[band].max():.2f}")
        if r == 0:
            for a, t in zip(axs[r], ["u", f"director small ({sigma_small:.2g})",
                                     "blend weight", "director blended",
                                     "|change| deg"]):
                a.set_title(t, fontsize=9)
        for a in axs[r]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Adaptive-width director (mu={mu}): knee untouched, "
                 "PN widened at cores", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"figure -> {out}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--validate" in sys.argv:
        validate()
        sys.exit(0)

    # In-code config, mirroring the base runner's Args block, with the adaptive
    # director selected and the three new knobs added. Edit paths for your tree.
    class Args:
        sh_path = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweeps_full_Nx512_hp025_T3p125_NyF5_np18_Nsave125/raw/sh_pgb_zigzag_mu0.900_T3_N125_nx512_Ny256_lower.npz"           # set to one .npz, or use all=True + input_dir
        all = False
        input_dir = None
        output_dir = "results/riesz_adaptive_sig_pio2_defect_aware_false"
        sigma = 0.4               # small (clean-knee) width  -> sigma_fixed
        defect_aware = False       # ignored by the adaptive branch
        sigma_large = float(np.pi/2)  # large (defect) width; base default was pi/2
        sigma_det = 0.3
        band_y = 2.0
        band_x_frac = 0.10
        ds = 0.30
        n_phase_seeds = 128
        seed_half = "upper"
        director = "adaptive"     # <-- the new mode
        sigma_uhu_min = float(np.pi / 2)
        seed_contour = "crest"
        crest_y_start = 3.0
        crest_y_stop = 1.0
        gauge = "auto"
        reflect_y = "mirror"
        profile_y = 1.0
        theta_smooth_w = 0.75
        max_steps = 10000
        grad_smooth = 1.0
        xmargin = 0.025
        ymargin = 0.025
        tanhscale = 120.0
        gif_fps = 10
        no_plot = False
        no_gif = False
        overwrite = True

    cfg = base._cfg_from_args(Args)
    # inject the new adaptive knobs (base._cfg_from_args does not know them)
    cfg["loop_half_sizes"] = (1, 2, 3, 4, 6, 8)
    cfg["persist_smooth_px"] = 1.5
    cfg["persist_gamma"] = 1.0

    if Args.sh_path is None and not Args.all:
        print("Set Args.sh_path to one .npz (or all=True + input_dir), or run "
              "with --validate first.")
        sys.exit(0)
    base.run_with_cfg(cfg, Args)
