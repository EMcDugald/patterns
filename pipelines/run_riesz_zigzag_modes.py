# pipelines/run_riesz_zigzag_modes.py
"""
Riesz-only PGB extraction with ONE director-width policy, chosen from three, and
NO uHu path anywhere. This replaces the tangled `director` string of
run_riesz_zigzag_adaptive (which silently swapped charged frames to a uHu
eigenvector director under director="aware") with a single, explicit choice:

  --mode constant     one fixed director width on every frame, charged or not
  --mode gated        small width where the core band has zero charge,
                      large width where it has charge (binary, per frame)
  --mode persistence  continuous per-point width from the loop-persistence of
                      the small-width director (small on clean stretches, large
                      only where locally defected)

The magnitude |k| = sqrt(<|grad u|^2>/<u^2>) is a separate averaging with its
own width sigma_avg. Two policies:

  --mag_mode fixed    sigma_avg is one fixed width (default pi), decoupled from
                      the director width. This is the measured-good setting:
                      the |k| average must span the 2*theta oscillation, so it
                      needs sigma_avg*|k| ~ 1 REGARDLESS of the director width.
  --mag_mode inherit  sigma_avg follows the director width that frame (scalar
                      for constant/gated; a per-point small/large blend for
                      persistence, by the same loop-persistence weight). This is
                      the "curiosity" mode: with a small director width it makes
                      the magnitude average small too, which the decoupling was
                      designed to avoid. Kept so the failure is measurable, not
                      recommended for production.

The director is ALWAYS the doubled-angle Riesz director. There is no uHu branch.

Two entry points, mirroring run_riesz_zigzag_adaptive:
  1. python run_riesz_zigzag_modes.py --validate
       Self-contained analytic check (no simulation data, no base runner): runs
       the extraction CORE on the exact knee and PN for every (mode, mag_mode),
       reporting director-width behaviour and |k| accuracy against the exact
       gradient. Writes results/riesz_modes_validation.png.
  2. python run_riesz_zigzag_modes.py   (edit the Args block)
       Full extraction: imports run_riesz_zigzag as `base`, monkeypatches
       extract_one_frame with the Riesz-only extractor below, and runs the base
       series/reconstruction/npz path unchanged. Requires your tree (base +
       raw .npz); this path is NOT exercised by --validate.
"""

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

# --- CORE deps: available standalone (no base runner needed) ---------------
from op_extract.t1_common import (
    riesz_k_field, riesz_director, macro_iso, spectral_deriv,
)
from op_extract.loop_persistence import adaptive_director


# ===========================================================================
# The extraction CORE: director-width policy + magnitude-width policy.
# Riesz-only. This is the piece validated on analytic fields below; the frame
# extractor and the runner both go through it, so what is proven here is what
# runs on data.
# ===========================================================================

def _rebuild_kd(kmag, d1, d2, A):
    """Assemble a kd dict from a magnitude, a (blended) director, and A."""
    kmag = np.asarray(kmag)
    return {"kmag": kmag, "A": np.asarray(A),
            "k1": kmag * d1, "k2": kmag * d2, "d1": d1, "d2": d2}


def extract_k_modes(u_ramped, x, y, Lx, Ly, det, cfg):
    """
    Return the Riesz k field and director under the selected mode / mag_mode.

    det : the detect_defects record for this frame (needs det["sum_abs"] for the
          gated mode); pass det=None only when charge is irrelevant (constant).
    cfg keys used:
      mode        'constant' | 'gated' | 'persistence'
      mag_mode    'fixed' | 'inherit'
      sigma_const constant-mode director width           (default 0.4)
      sigma_small clean/small director width             (default 0.4)
      sigma_large defect/large director width            (default pi)
      sigma_avg   fixed magnitude width                  (default pi)
      loop_half_sizes, persist_smooth_px, persist_gamma  (persistence only)

    Returns dict(kd, d1, d2, weight, sigma_dir_report, sigma_avg_report).
    kd has k1, k2, kmag, A, d1, d2 with k1/k2 built from the SELECTED director.
    """
    mode = cfg["mode"]
    mag_mode = cfg.get("mag_mode", "fixed")
    s_small = float(cfg.get("sigma_small", 0.4))
    s_large = float(cfg.get("sigma_large", np.pi))
    s_const = float(cfg.get("sigma_const", s_small))
    s_avg_fixed = float(cfg.get("sigma_avg", np.pi))

    if mode == "constant":
        sigma_dir = s_const
        s_avg = s_avg_fixed if mag_mode == "fixed" else sigma_dir
        kd = riesz_k_field(u_ramped, x, y, Lx, Ly, sigma_dir, sigma_avg=s_avg)
        d1, d2 = np.asarray(kd["d1"]), np.asarray(kd["d2"])
        kd = _rebuild_kd(kd["kmag"], d1, d2, kd["A"])
        return dict(kd=kd, d1=d1, d2=d2, weight=None,
                    sigma_dir_report=sigma_dir, sigma_avg_report=float(s_avg))

    if mode == "gated":
        if det is None:
            raise ValueError("gated mode needs the detect_defects record")
        sigma_dir = s_large if det["sum_abs"] > 0 else s_small
        s_avg = s_avg_fixed if mag_mode == "fixed" else sigma_dir
        kd = riesz_k_field(u_ramped, x, y, Lx, Ly, sigma_dir, sigma_avg=s_avg)
        d1, d2 = np.asarray(kd["d1"]), np.asarray(kd["d2"])
        kd = _rebuild_kd(kd["kmag"], d1, d2, kd["A"])
        return dict(kd=kd, d1=d1, d2=d2, weight=None,
                    sigma_dir_report=sigma_dir, sigma_avg_report=float(s_avg))

    if mode == "persistence":
        ad = adaptive_director(
            u_ramped, Lx, Ly, riesz_director,
            sigma_small=s_small, sigma_large=s_large,
            half_sizes=tuple(cfg.get("loop_half_sizes", (1, 2, 3, 4, 6, 8))),
            smooth_px=float(cfg.get("persist_smooth_px", 1.5)),
            gamma=float(cfg.get("persist_gamma", 1.0)),
        )
        d1, d2 = np.asarray(ad["d1"]), np.asarray(ad["d2"])
        w = np.asarray(ad["weight"])
        if mag_mode == "fixed":
            km = riesz_k_field(u_ramped, x, y, Lx, Ly, s_small,
                               sigma_avg=s_avg_fixed)
            kmag, A = np.asarray(km["kmag"]), np.asarray(km["A"])
            s_avg_rep = s_avg_fixed
        else:
            # per-point magnitude: blend the small- and large-width averages by
            # the same loop-persistence weight that blends the director
            km_s = riesz_k_field(u_ramped, x, y, Lx, Ly, s_small, sigma_avg=s_small)
            km_l = riesz_k_field(u_ramped, x, y, Lx, Ly, s_large, sigma_avg=s_large)
            kmag = (1.0 - w) * np.asarray(km_s["kmag"]) + w * np.asarray(km_l["kmag"])
            A = (1.0 - w) * np.asarray(km_s["A"]) + w * np.asarray(km_l["A"])
            s_avg_rep = float(s_small + w.mean() * (s_large - s_small))
        kd = _rebuild_kd(kmag, d1, d2, A)
        sigma_dir = float(s_small + w.mean() * (s_large - s_small))
        return dict(kd=kd, d1=d1, d2=d2, weight=w,
                    sigma_dir_report=sigma_dir, sigma_avg_report=s_avg_rep)

    raise ValueError(f"unknown mode {mode!r} (use constant|gated|persistence)")


# ===========================================================================
# Frame extractor: the adaptive-runner body with the director block replaced by
# extract_k_modes and the uHu branch removed. Everything after the director
# block (seed, reconstruct, k_from_theta, monogenic, assemble) is unchanged, so
# the npz layout matches the base runner. Base functions are imported lazily so
# --validate works without the base runner present.
# ===========================================================================

def extract_one_frame_modes(u_frame, sh, ramp, cfg, crest_state=None):
    from run_riesz_zigzag import (  # lazy: only the data path needs base
        detect_defects, compute_pgb_phase_from_uhu, rebuild_grid,
        k_from_theta, k_consistency, monogenic, reflect_phase_y,
        trace_crest_contour,
    )
    x, y = sh["x"], sh["y"]
    Lx, Ly = sh["Lx"], sh["Ly"]
    mu = sh["mu"]
    band_y = cfg["band_y"]
    band_x = (1.0 - cfg["band_x_frac"]) * 0.5 * Lx
    u_ramped = u_frame * ramp
    nan2d = np.full_like(u_frame, np.nan, dtype=float)

    try:
        det = detect_defects(u_ramped, x, y, Lx, Ly, sigma_det=cfg["sigma_det"],
                             band_y=band_y, band_x=band_x)

        # ---- director + magnitude policy (Riesz-only, single mode) ----------
        ek = extract_k_modes(u_ramped, x, y, Lx, Ly, det, cfg)
        kd = ek["kd"]
        d1a, d2a = ek["d1"], ek["d2"]
        sigma_dir = ek["sigma_dir_report"]
        # ---------------------------------------------------------------------

        contour_source = "template"
        knee_bdry = None
        if crest_state is None:
            crest_state = {}
        if cfg["seed_contour"] == "crest":
            kw = dict(y_start=cfg["crest_y_start"], y_stop=cfg["crest_y_stop"],
                      x_margin=cfg["xmargin"] * Lx + 1.0,
                      y_margin=cfg["ymargin"] * Ly + 1.0, ds=cfg["ds"])
            con, p0 = trace_crest_contour(u_frame, x, y, d1a, d2a,
                                          anchor=crest_state.get("anchor"), **kw)
            if con is None and crest_state.get("anchor") is not None:
                print("    [warn] anchored crest trace failed; retrying fresh")
                con, p0 = trace_crest_contour(u_frame, x, y, d1a, d2a,
                                              anchor=None, **kw)
            if con is not None:
                knee_bdry = con
                contour_source = "crest"
                crest_state["anchor"] = p0
            else:
                print("    [warn] crest trace failed its gate; template fallback")

        gauge = cfg["gauge"]
        if gauge == "auto":
            gauge = "zero_origin" if contour_source == "crest" else "seed_phase"

        d1 = d1a[:, :, None]
        d2 = d2a[:, :, None]
        uhu_data = {
            "x": x, "y": y, "u": u_frame[:, :, None],
            "k1_sym": d1, "k2_sym": d2, "k1_orig": d1, "k2_orig": d2,
            "mu": mu, "tt": None, "ramp": None, "k": None, "A": None,
            "knee_bdry": knee_bdry, "source_path": sh["source_path"],
        }
        res = compute_pgb_phase_from_uhu(
            uhu_data, mu=mu, n_phase_seeds=cfg["n_phase_seeds"],
            seed_half=cfg["seed_half"], ds=cfg["ds"], max_steps=cfg["max_steps"],
            prefer_sym=True, phase_ramp_mode="none", grid_output=False,
        )
        fld = {"x": x, "y": y, "theta": sh["theta_ref"], "odd_in_y": False}
        theta = rebuild_grid(res, fld, gauge=gauge, half="orig", band=cfg["ds"])
        theta = reflect_phase_y(theta, sh["core_row"], mode=cfg["reflect_y"])

        kg = k_from_theta(theta, x, y, Lx, Ly, smooth=cfg["grad_smooth"])
        cons = k_consistency(kd, kg)

        mono = monogenic(u_ramped, Lx, Ly)
        A_mono = np.asarray(mono["A"])
        X, Y = np.meshgrid(x, y)
        band = (np.abs(Y) < band_y) & (np.abs(X) < band_x)
        far = ((np.abs(Y) > 3.0 * band_y) & (np.abs(Y) < 6.0 * band_y)
               & (np.abs(X) < band_x))
        A_min_band = float(np.nanmin(A_mono[band])) if band.any() else np.nan
        A_far_median = float(np.nanmedian(A_mono[far])) if far.any() else np.nan

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
            "A_min_band": A_min_band, "A_far_median": A_far_median,
            "mutual_median": float(cons.get("dang_med_abs", np.nan)),
            "knee_bdry_phase": np.asarray(res.get("knee_bdry_phase")),
            "coordinate_lines": res["coordinate_lines"][0],
            "charges": [[float(c[0]), float(c[1]), int(c[2])]
                        for c in det["charges"]],
        }
    except Exception as exc:
        print(f"    [warn] frame extraction failed: {exc}")
        return {
            "ok": False, "det": None, "sigma_dir": np.nan,
            "contour_source": "failed", "gauge": "none",
            "theta": nan2d.astype(np.float32),
            "k1_direct": nan2d.astype(np.float32),
            "k2_direct": nan2d.astype(np.float32),
            "kmag_direct": nan2d.astype(np.float32),
            "A": nan2d.astype(np.float32), "A_mono": nan2d.astype(np.float32),
            "k1_grad": nan2d.astype(np.float32),
            "k2_grad": nan2d.astype(np.float32),
            "kmag_grad": nan2d.astype(np.float32),
            "mutual_dang": nan2d.astype(np.float32),
            "sum_abs": -1, "n_dislocations": -1,
            "A_min_band": np.nan, "A_far_median": np.nan,
            "mutual_median": np.nan, "knee_bdry_phase": None,
            "coordinate_lines": None, "charges": [],
        }


# ===========================================================================
# analytic validation of the CORE (self-contained; no base, no sim data)
# ===========================================================================

def _ramp(x, frac=0.12, sharp=6.0):
    Lx = x[-1] - x[0]
    return (0.5 * (1 + np.tanh((x - x[0] - frac * Lx) * sharp / Lx))
            * 0.5 * (1 + np.tanh((x[-1] - x - frac * Lx) * sharp / Lx)))


def _knee(mu, n_xper=4, n_yper=6, ppw=28):
    k1, k2 = np.sqrt(1.0 - mu ** 2), mu
    dx = 2.0 * np.pi / ppw
    Lx = n_xper * (2 * np.pi / max(k1, 0.2)); Ly = n_yper * (2 * np.pi / k2)
    Nx, Ny = int(round(Lx / dx)), int(round(Ly / dx))
    x = (np.arange(Nx) - Nx // 2) * dx; y = (np.arange(Ny) - Ny // 2) * dx
    X, Y = np.meshgrid(x, y)
    theta = k1 * X + np.log(2.0 * np.cosh(k2 * Y))
    u = np.cos(theta)
    k1e = np.full_like(X, k1)
    k2e = k2 * np.tanh(k2 * Y)
    return dict(u=u, x=x, y=y, X=X, Y=Y, Lx=x[-1] - x[0], Ly=y[-1] - y[0],
                k1e=k1e, k2e=k2e, core_row=int(np.argmin(np.abs(y))))


def _pn(mu, n_def=4, n_xper=4, n_yper=6, ppw=28):
    """Full-plane PN (both halves) so the dislocation winding is present and the
    persistence director actually fires. Exact k is the accurate finite-diff
    gradient of the analytic phase; the y=0 seam row is excluded downstream."""
    from scipy import special
    k1, k2 = np.sqrt(1.0 - mu ** 2), mu
    dx = 2.0 * np.pi / ppw
    Lx = n_xper * (2 * np.pi / max(k1, 0.2)); Ly = n_yper * (2 * np.pi / k2)
    Nx, Ny = int(round(Lx / dx)), int(round(Ly / dx))
    x = (np.arange(Nx) - Nx // 2) * dx; y = (np.arange(Ny) - Ny // 2) * dx
    X, Y = np.meshgrid(x, y)
    xd = np.linspace(-0.3 * Lx, 0.3 * Lx, n_def)
    thetas = np.arange(n_def + 1) * np.pi
    psi = 0.5 * (1 + np.exp(-thetas[-1])) * np.ones_like(X)
    sq = np.sqrt(2 * np.maximum(np.abs(Y), 1e-6))
    for j in range(n_def):
        d = 0.5 * (np.exp(-thetas[j + 1]) - np.exp(-thetas[j]))
        psi += d * special.erf((X - xd[j]) / sq)
    theta = Y - np.sign(Y) * np.log(psi)
    u = np.cos(theta)
    k1e = np.gradient(theta, dx, axis=1)
    k2e = np.gradient(theta, dx, axis=0)
    return dict(u=u, x=x, y=y, X=X, Y=Y, Lx=x[-1] - x[0], Ly=y[-1] - y[0],
                k1e=k1e, k2e=k2e, core_row=int(np.argmin(np.abs(y))))


def _kmag_err(kd, fld, core_skip, region):
    """rel rms |k| error vs exact. region='interior' excludes |y-core|<core_skip
    (and the seam); region='near' keeps only that core strip. Both trim x-edges
    and the far y-tail."""
    X, Y = fld["X"], fld["Y"]
    x, y = fld["x"], fld["y"]
    dcore = np.abs(Y - y[fld["core_row"]])
    m = (dcore > core_skip) if region == "interior" else \
        (dcore < core_skip) & (dcore > 1.5 * (y[1] - y[0]))   # skip seam row
    m &= dcore < 8.0
    m &= (X > x[0] + 0.2 * (x[-1] - x[0])) & (X < x[-1] - 0.2 * (x[-1] - x[0]))
    ke = np.hypot(fld["k1e"], fld["k2e"]); km = np.asarray(kd["kmag"])
    g = np.isfinite(km) & np.isfinite(ke) & m
    if g.sum() < 50:
        return np.nan
    return float(np.sqrt(np.mean((km[g] - ke[g]) ** 2)) / np.mean(ke[g]))


def _ang_err(ek, fld, core_skip, region):
    """median |director - exact| angle mod pi, in degrees. This is what the
    DIRECTOR width governs (magnitude policy does not enter)."""
    X, Y = fld["X"], fld["Y"]
    x, y = fld["x"], fld["y"]
    dcore = np.abs(Y - y[fld["core_row"]])
    m = (dcore > core_skip) if region == "interior" else \
        (dcore < core_skip) & (dcore > 1.5 * (y[1] - y[0]))
    m &= dcore < 8.0
    m &= (X > x[0] + 0.2 * (x[-1] - x[0])) & (X < x[-1] - 0.2 * (x[-1] - x[0]))
    ae = np.arctan2(ek["d2"], ek["d1"])
    ke = np.arctan2(fld["k2e"], fld["k1e"])
    d = (ae - ke + np.pi / 2) % np.pi - np.pi / 2      # mod-pi director diff
    g = np.isfinite(d) & m
    if g.sum() < 50:
        return np.nan
    return float(np.degrees(np.median(np.abs(d[g]))))


def validate(mu=0.90, out="results/riesz_modes_validation.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)

    fields = {"knee": _knee(mu), "pn": _pn(mu)}
    det_for = {"knee": {"sum_abs": 0}, "pn": {"sum_abs": 4}}
    modes = ["constant", "gated", "persistence"]
    mag_modes = ["fixed", "inherit"]
    core_skip = np.pi / mu                     # one core-spacing unit

    print(f"validate mu={mu}  core_skip=pi/mu={core_skip:.2f}\n"
          f"  |k| = rel-rms magnitude error (governed by sigma_avg / mag_mode)\n"
          f"  ang = median director error deg mod pi (governed by the mode's "
          f"director width)\n")
    header = (f"  {'field':6}{'mode':13}{'mag':8}"
              f"{'|k| int':>8}{'|k| near':>9}{'ang int':>8}{'ang near':>9}"
              f"{'sig_dir':>8}{'sig_avg':>8}{'w_max':>6}")
    print(header); print("  " + "-" * (len(header) - 2))
    rows = {}
    for nm, fld in fields.items():
        R = _ramp(fld["x"])[None, :]
        for mode in modes:
            for mm in mag_modes:
                cfg = dict(mode=mode, mag_mode=mm, sigma_small=0.4,
                           sigma_large=float(np.pi), sigma_const=0.4,
                           sigma_avg=float(np.pi))
                ek = extract_k_modes(fld["u"] * R, fld["x"], fld["y"],
                                     fld["Lx"], fld["Ly"], det_for[nm], cfg)
                e_in = _kmag_err(ek["kd"], fld, core_skip, "interior")
                e_nr = _kmag_err(ek["kd"], fld, core_skip, "near")
                a_in = _ang_err(ek, fld, core_skip, "interior")
                a_nr = _ang_err(ek, fld, core_skip, "near")
                wmax = 0.0 if ek["weight"] is None else float(ek["weight"].max())
                rows[(nm, mode, mm)] = dict(kmag_int=e_in, kmag_near=e_nr,
                                            ang_int=a_in, ang_near=a_nr, ek=ek)
                print(f"  {nm:6}{mode:13}{mm:8}{e_in:8.3f}{e_nr:9.3f}"
                      f"{a_in:8.2f}{a_nr:9.2f}"
                      f"{ek['sigma_dir_report']:8.2f}"
                      f"{ek['sigma_avg_report']:8.2f}{wmax:6.2f}")
        print()

    # figure: near-core director error (the metric the modes govern), fixed mag
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, nm in zip(axes, fields):
        xs = np.arange(len(modes))
        vals = [rows[(nm, mode, "fixed")]["ang_near"] for mode in modes]
        ax.bar(xs, vals, 0.6, color=["C0", "C1", "C2"])
        ax.set_xticks(xs); ax.set_xticklabels(modes, rotation=15)
        ax.set_ylabel("near-core director error (deg)")
        ax.set_title(f"{nm}  (mu={mu}, mag=fixed)")
        for xi, v in zip(xs, vals):
            ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Riesz-only extraction: near-core director accuracy by mode "
                 "(constant vs gated vs persistence)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"figure -> {out}")
    return rows


# ===========================================================================
# CONFIG: the whole editable interface lives here. Set an input (one file OR a
# directory), an output location, the director policy, and any reconstruction
# knob, then run `python run_riesz_zigzag_modes.py`. You should not need to open
# the base runner: file discovery, per-mode output foldering, and the loop over
# files are all in this script (run_all below). The base runner is used only for
# the per-file load / phase reconstruction / npz save, which you do not edit.
# ===========================================================================

# ---------------------------------------------------------------------------
# output-folder width tag
# ---------------------------------------------------------------------------

def _wtag(v):
    v = float(v)
    if abs(v - np.pi) < 1e-3:
        return "pi"
    if abs(v - np.pi / 2) < 1e-3:
        return "pihalf"
    return f"{v:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def run_tag(cfg):
    """Folder name reflecting the mode, the magnitude policy, AND the widths in
    play, e.g. constant_fixed_sd0p4_sapi, gated_fixed_sd0p4-pi_sapi,
    persistence_inherit_sd0p4-pi_sainherit."""
    if cfg.mode == "constant":
        d = f"sd{_wtag(cfg.sigma_const)}"
    else:
        d = f"sd{_wtag(cfg.sigma)}-{_wtag(cfg.sigma_large)}"
    m = f"sa{_wtag(cfg.sigma_avg)}" if cfg.mag_mode == "fixed" else "sainherit"
    return f"{cfg.mode}_{cfg.mag_mode}_{d}_{m}"


# ---------------------------------------------------------------------------
# loop-persistence density field (frame 0 vs final frame)
# ---------------------------------------------------------------------------

def _persistence_count(angle, half_sizes=(1, 2, 3, 4, 6, 8)):
    """Per-plaquette count of loop sizes still enclosing net winding (0 on a
    smooth field, rising at a resolved core). Grid (Ny-1, Nx-1)."""
    from op_extract.loop_persistence import loop_persistence_field
    count, q = loop_persistence_field(angle, half_sizes=tuple(half_sizes))
    return count, q


def _angle_from_u(u, x, y, sigma_small=0.4):
    Lx, Ly = x[-1] - x[0], y[-1] - y[0]
    return np.asarray(riesz_director(u, Lx, Ly, float(sigma_small))["angle"])


def persistence_figure(items, x, y, out, half_sizes=(1, 2, 3, 4, 6, 8),
                       sigma_small=0.4):
    """items: list of (title, kind, data).
       kind='u'     -> data is a 2d pattern; the small-width director is taken here
       kind='dir'   -> data is (k1, k2); angle = atan2(k2, k1)
       kind='angle' -> data is a director angle already
    Top row shows the pattern/director, bottom row the loop-persistence count."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = [x[0], x[-1], y[0], y[-1]]
    n = len(items)
    cmax = max(1, len(half_sizes))
    fig, ax = plt.subplots(2, n, figsize=(5.2 * n, 8), squeeze=False)
    for j, (title, kind, data) in enumerate(items):
        if kind == "u":
            ang = _angle_from_u(data, x, y, sigma_small)
            top, tlab, cmap = data, "u", "gray"
        elif kind == "dir":
            k1, k2 = data
            ang = np.arctan2(np.asarray(k2), np.asarray(k1))
            top, tlab, cmap = np.degrees(ang), "director angle (deg)", "twilight"
        else:
            ang = np.asarray(data)
            top, tlab, cmap = np.degrees(ang), "director angle (deg)", "twilight"
        count, _ = _persistence_count(ang, half_sizes)
        ax[0, j].imshow(top, extent=ext, origin="lower", cmap=cmap, aspect="auto")
        ax[0, j].set_title(f"{title}\n{tlab}", fontsize=9)
        im = ax[1, j].imshow(count, extent=ext, origin="lower", cmap="magma",
                             vmin=0, vmax=cmax, aspect="auto")
        ax[1, j].set_title("loop-persistence count", fontsize=9)
        fig.colorbar(im, ax=ax[1, j], shrink=0.8)
        for a in (ax[0, j], ax[1, j]):
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("loop-persistence density (0 for a clean knee, bumps at cores)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def persistence_from_raw(sh_path, cfg, out):
    """Load a raw SH .npz, take frame 0 and the final frame, and plot the
    loop-persistence field of the small-width director for each. Raw layout is
    assumed to match the op layout (u as (Ny, Nx, Nt), x, y); adjust here if
    your raw files differ."""
    d = np.load(sh_path, allow_pickle=True)
    if "u" not in d:
        raise KeyError(f"{Path(sh_path).name} has no 'u' (keys: {list(d.keys())})")
    u = np.asarray(d["u"]); x = np.asarray(d["x"]); y = np.asarray(d["y"])
    if u.ndim != 3:
        raise ValueError(f"expected 3d u, got {u.shape}")
    R = _ramp(x)[None, :]
    items = [("frame 0", "u", u[..., 0] * R),
             ("frame T (final)", "u", u[..., -1] * R)]
    persistence_figure(items, x, y, out, half_sizes=tuple(cfg.loop_half_sizes),
                       sigma_small=float(cfg.sigma))
    print(f"    persistence figure -> {out}")


def persistence_demo(op_path=None, mu=0.85, out_dir="results/persistence_demo"):
    """Analytic contrast (knee -> ~0, PN -> density at cores). If op_path is a
    saved op npz with k1_direct/k2_direct, also show its first vs last stored
    frame from the extracted director (real dislocations)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    knee, pn = _knee(mu), _pn(mu)
    persistence_figure([("knee (defect-free)", "u",
                         knee["u"] * _ramp(knee["x"])[None, :])],
                       knee["x"], knee["y"],
                       out_dir / f"persistence_knee_mu{mu:.2f}.png")
    persistence_figure([("PN (dislocations)", "u",
                         pn["u"] * _ramp(pn["x"])[None, :])],
                       pn["x"], pn["y"],
                       out_dir / f"persistence_pn_mu{mu:.2f}.png")
    print(f"analytic persistence figures -> {out_dir}")
    if op_path and Path(op_path).exists():
        d = np.load(op_path, allow_pickle=True)
        if "k1_direct" in d:
            x, y = np.asarray(d["x"]), np.asarray(d["y"])
            k1, k2 = np.asarray(d["k1_direct"]), np.asarray(d["k2_direct"])
            items = [("stored frame 0", "dir", (k1[..., 0], k2[..., 0])),
                     ("stored frame -1", "dir", (k1[..., -1], k2[..., -1]))]
            persistence_figure(items, x, y,
                               out_dir / "persistence_op_frames.png")
            print(f"op-file persistence figure -> {out_dir}")


class CONFIG:
    # ---- INPUT: set exactly one of input_path or input_dir -----------------
    input_path = None          # a single raw SH .npz
    #input_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweeps_full_Nx512_hp2_T25_NyF5_np18_Nsave125/raw"           # a folder; every file matching input_glob is run
    input_dir = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweeps_shallow_R1p0_Nx512_tmax500_hp5_np18_Ny5/raw"
    input_glob = "*.npz"       # pattern used when input_dir is set

    # ---- OUTPUT ------------------------------------------------------------
    output_root = "results/riesz_modes_shallow_mu_R1p0"   # op npz + figures land under here
    per_mode_subdir = True     # append "<mode>_<mag_mode>" so runs never collide
    overwrite = True

    # ---- THE ONE DIRECTOR POLICY (choose via `mode`) -----------------------
    mode = "persistence"          # 'constant' | 'gated' | 'persistence'
    mag_mode = "fixed"         # 'fixed' | 'inherit'
    sigma_const = 0.4          # constant-mode director width
    sigma = 0.4                # small width (sigma_small; gated/persistence)
    sigma_large = float(np.pi) # large width (gated/persistence)
    sigma_avg = float(np.pi)   # fixed magnitude width (used when mag_mode=fixed)
    loop_half_sizes = (1, 2, 3, 4, 6, 8)   # persistence loop scales
    persist_smooth_px = 1.5                 # persistence weight smoothing
    persist_gamma = 1.0                     # persistence ramp sharpness

    # ---- detection / reconstruction (base pipeline knobs) ------------------
    sigma_det = 0.3
    band_y = 2.0
    band_x_frac = 0.10
    ds = 0.30
    n_phase_seeds = 128
    seed_half = "upper"
    sigma_uhu_min = float(np.pi / 2)   # unused (no uHu); kept for cfg parity
    defect_aware = False               # unused; gating is via `mode`
    director = "modes"                 # marker; real choice is `mode`
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

    # ---- figures / gifs ----------------------------------------------------
    no_plot = False
    no_gif = False
    gif_fps = 10
    persistence_frames_plot = True   # also save loop-persistence(frame 0, final)

    # ---- base-runner compatibility fields (set per file by run_all) --------
    sh_path = None
    all = False
    output_dir = None


def discover_inputs(cfg):
    """Return the list of raw .npz files to process, from input_path or a glob
    over input_dir. Fully controlled here, not by the base runner."""
    import glob
    if cfg.input_dir:
        files = sorted(glob.glob(str(Path(cfg.input_dir) / cfg.input_glob)))
        if not files:
            raise SystemExit(f"[modes] no files matching {cfg.input_glob!r} "
                             f"in {cfg.input_dir}")
        return files
    if cfg.input_path:
        p = Path(cfg.input_path)
        if not p.exists():
            raise SystemExit(f"[modes] input_path does not exist: {p}")
        return [str(p)]
    raise SystemExit("[modes] set CONFIG.input_path (one file) OR "
                     "CONFIG.input_dir (a folder).")


def run_all(cfg):
    """Own the whole run: discover inputs, set the Riesz-only extractor, choose
    the output folder, and loop base.run_with_cfg over each file."""
    import run_riesz_zigzag as base
    base.extract_one_frame = extract_one_frame_modes      # Riesz-only override

    out_root = Path(cfg.output_root)
    if cfg.per_mode_subdir:
        out_root = out_root / run_tag(cfg)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_root)                        # base reads this

    files = discover_inputs(cfg)
    print(f"[modes] mode={cfg.mode}  mag_mode={cfg.mag_mode}  "
          f"sigma_avg={cfg.sigma_avg:.3f}\n"
          f"[modes] {len(files)} file(s) -> {out_root}")

    base_cfg = base._cfg_from_args(cfg)                   # internal cfg dict
    base_cfg["mode"] = cfg.mode
    base_cfg["mag_mode"] = cfg.mag_mode
    base_cfg["sigma_const"] = cfg.sigma_const
    base_cfg["sigma_avg"] = cfg.sigma_avg
    base_cfg["loop_half_sizes"] = tuple(cfg.loop_half_sizes)
    base_cfg["persist_smooth_px"] = cfg.persist_smooth_px
    base_cfg["persist_gamma"] = cfg.persist_gamma

    for i, f in enumerate(files, 1):
        print(f"\n[modes {i}/{len(files)}] {Path(f).name}")
        cfg.sh_path = f
        cfg.all = False
        base.run_with_cfg(base_cfg, cfg)
        if getattr(cfg, "persistence_frames_plot", False):
            try:
                persistence_from_raw(
                    f, cfg, out_root / f"{Path(f).stem}_persistence.png")
            except Exception as exc:
                print(f"    [warn] persistence plot skipped: {exc}")
    print(f"\n[modes] done -> {out_root}")


# ===========================================================================
# entry point.  --validate runs the analytic check (no base, no data). A light
# CLI mirror overrides the most-used CONFIG fields so you can launch variants
# without editing the file; everything else is edited in CONFIG above.
# ===========================================================================

if __name__ == "__main__":
    if "--validate" in sys.argv:
        validate()
        sys.exit(0)
    if "--persistence-demo" in sys.argv:
        # optional: pass an op npz path after the flag to include real frames
        i = sys.argv.index("--persistence-demo")
        op = sys.argv[i + 1] if i + 1 < len(sys.argv) else None
        persistence_demo(op_path=op)
        sys.exit(0)

    import argparse
    p = argparse.ArgumentParser(description="Riesz-only PGB extraction (modes).")
    p.add_argument("--mode", choices=["constant", "gated", "persistence"])
    p.add_argument("--mag_mode", choices=["fixed", "inherit"])
    p.add_argument("--input_path")
    p.add_argument("--input_dir")
    p.add_argument("--output_root")
    a = p.parse_args()
    for k in ("mode", "mag_mode", "input_path", "input_dir", "output_root"):
        v = getattr(a, k)
        if v is not None:
            setattr(CONFIG, k, v)

    run_all(CONFIG)
