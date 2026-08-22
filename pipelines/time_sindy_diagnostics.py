# pipelines/time_sindy_diagnostics.py
"""
Diagnostics companion for time_sindy_riesz.py. No file is patched: it imports
that script's own build_time_samples / _Args / detect_freeze_frame and the same
underlying helpers (load_op, frame_fields, nan_grad, fit_target), so every
diagnostic reads the same samples and the same fitted coefficients the tables do.

Four views, per (target, window):
  1. reconstructed vs actual   the fitted law's prediction of the rate field on
                               a representative frame, beside the actual rate and
                               the residual, plus the A field for reference.
  2. library feature overlap   every column z-scored along the core row on one
                               plot (visual collinearity) and a VIF table printed
                               (numeric collinearity).
  3. query points on pattern   u for one frame with the sampled band points
                               scattered on top, so coverage vs core / ramp / edges
                               is visible.
  4. interior-trim scan        rebuilds samples at a few (x_trim, band_in) and
                               prints the holdout, to see if pulling the interior
                               in lifts the numbers.

The 2D reconstruction recomputes one frame's library on the grid, mirroring
build_time_samples' per-frame block. That recompute is self-checked against the
stored sample rows for the same frame; a mismatch prints a loud warning so the
diagnostic can never silently diverge from the fit it is explaining.

Run:
  python time_sindy_diagnostics.py <op_path> --target dA_dt --window pre-freeze
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

# bind to the real pipeline
from time_sindy_riesz import _Args, build_time_samples, detect_freeze_frame
from run_riesz_metrics_zigzag import (
    load_op, ensure_theta_smooth, frame_fields, nan_grad,
)
from self_dual_sindy import fit_target, PAPER_RC

# for the query-points underlay: the extraction's own loader recenters the raw
# SH exactly onto the OP grid, so reusing it guarantees alignment. Both imports
# are optional; without them the query panel simply omits the pattern.
try:
    from run_riesz_zigzag import load_sh_npz
except Exception:
    load_sh_npz = None
try:
    from run_riesz_metrics_zigzag import locate_sh_file
except Exception:
    locate_sh_file = None


# ---------------------------------------------------------------------------
# one-frame 2D library, mirroring build_time_samples' per-frame block
# ---------------------------------------------------------------------------

def _band_masks(op, args):
    x, y = op["x"], op["y"]
    core = int(op["core_row"]); y_core = float(y[core])
    Xc, Yc = np.meshgrid(x, y)
    band = (np.abs(Yc - y_core) <= args.band_y) \
        & (np.abs(Yc - y_core) >= args.band_in) \
        & (Xc > x[0] + args.x_trim) & (Xc < x[-1] - args.x_trim)
    band_up = band & (Yc >= y_core)
    return band, band_up, Xc, Yc


def frame_library_2d(op, t, args):
    """Recompute the 2D library and target rate fields for frame t, exactly as
    build_time_samples does per frame (kept 2D, not masked/flattened)."""
    x, y = op["x"], op["y"]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    tt = op["tt"]
    band, band_up, Xc, Yc = _band_masks(op, args)
    A_far = op["A_far_median"] if "A_far_median" in op else None

    F = frame_fields(op, t, {})
    A = op["A_mono"][:, :, t].astype(float)
    af = float(A_far[t]) if A_far is not None else float(np.nanmedian(A[band]))
    dA = A / max(af, 1e-12) - 1.0
    km = op["kmag_direct"][:, :, t].astype(float)
    w = km ** 2 - 1.0
    th = op["theta_smooth"][:, :, t].astype(float)

    Ax, Ay = nan_grad(A, dx, dy)
    Axx, _ = nan_grad(Ax, dx, dy); _, Ayy = nan_grad(Ay, dx, dy)
    lapA = Axx + Ayy
    wx, wy = nan_grad(w, dx, dy)
    wxx, _ = nan_grad(wx, dx, dy); _, wyy = nan_grad(wy, dx, dy)
    lapw = wxx + wyy; gradw = np.sqrt(wx ** 2 + wy ** 2)
    tx, ty = nan_grad(th, dx, dy)
    txx, _ = nan_grad(tx, dx, dy); _, tyy = nan_grad(ty, dx, dy)

    shared = {
        "1": np.ones_like(w), "w": w, "w^2": w ** 2, "dA": dA, "dA^2": dA ** 2,
        "J_or": F["J_or"], "J_th": F["J_th"], "rho_d": F["rho_d"],
        "curl k": F["curl_or"],
    }
    lib2d = {
        "dA_dt": dict(shared, **{"A": A, "A^3": A ** 3, "Lap A": lapA}),
        "dw_dt": dict(shared, **{"Lap w": lapw, "div k": F["div_or"],
                                 "|grad w|": gradw}),
        "dth_dt": dict(shared, **{"th_xx": txx, "th_yy": tyy}),
    }
    dt2 = float(tt[t + 1] - tt[t - 1])
    tgt2d = {
        "dA_dt": (op["A_mono"][:, :, t + 1].astype(float)
                  - op["A_mono"][:, :, t - 1].astype(float)) / dt2,
        "dw_dt": (op["kmag_direct"][:, :, t + 1].astype(float) ** 2
                  - op["kmag_direct"][:, :, t - 1].astype(float) ** 2) / dt2,
        "dth_dt": (op["theta_smooth"][:, :, t + 1].astype(float)
                   - op["theta_smooth"][:, :, t - 1].astype(float)) / dt2,
    }
    def _fin(k, base_mask):
        f = base_mask.copy()
        for arr in lib2d[k].values():
            f &= np.isfinite(arr)
        f &= np.isfinite(tgt2d[k])
        return f
    masks = {"dA_dt": _fin("dA_dt", band), "dw_dt": _fin("dw_dt", band),
             "dth_dt": _fin("dth_dt", band_up)}
    base = {"dA_dt": A, "dw_dt": w, "dth_dt": th}
    return dict(lib2d=lib2d, tgt2d=tgt2d, masks=masks, base=base, A=A, u=None,
                band=band, band_up=band_up, Xc=Xc, Yc=Yc, dt2=dt2)


def _find_sh(op, op_path, u_from):
    """Locate the source SH .npz. u_from may be a file or a directory of raw
    SH sims; matching is by the OP's recorded source_file basename, then by any
    related .npz stem, then by the pipeline's own locate_sh_file search."""
    if u_from:
        p = Path(u_from)
        if p.is_file():
            return p
        if p.is_dir():
            src = (op.get("riesz_meta") or {}).get("source_file")
            cands = []
            if src:
                cands.append(p / Path(src).name)
            for c in cands:
                if c.exists():
                    return c
            stem = Path(op_path).stem
            for c in sorted(p.glob("*.npz")):
                if c.stem in stem or stem in c.stem:
                    return c
    if locate_sh_file is not None:
        override = u_from if (u_from and Path(u_from).is_file()) else None
        return locate_sh_file(op, op_path, override)
    return None


def _resolve_u(op, op_path, u_from):
    """A 3D u aligned to the OP grid, or None. Prefers u already in the OP file;
    otherwise loads the raw SH through the extraction's load_sh_npz so the crop
    and recentering match the OP exactly, then verifies grid agreement before
    using it (a mismatch is skipped, never drawn misaligned)."""
    if "u" in op:
        return np.asarray(op["u"])
    if load_sh_npz is None:
        print("  [u] run_riesz_zigzag.load_sh_npz not importable; no underlay")
        return None
    sh_path = _find_sh(op, op_path, u_from)
    if sh_path is None:
        print("  [u] no source SH found (pass --u_from DIR); "
              "query panel drawn without the pattern")
        return None
    try:
        sh = load_sh_npz(sh_path)
    except Exception as e:
        print(f"  [u] failed to load {Path(sh_path).name}: {e}")
        return None
    u = np.asarray(sh["u"])
    ok = (u.ndim == 3 and u.shape[0] == len(op["y"])
          and u.shape[1] == len(op["x"])
          and np.allclose(sh["x"], op["x"], atol=1e-3)
          and np.allclose(sh["y"], op["y"], atol=1e-3))
    if not ok:
        print(f"  [u] {Path(sh_path).name} grid does not match the OP grid; "
              "skipping underlay rather than drawing it misaligned")
        return None
    print(f"  [u] underlay from {Path(sh_path).name}")
    return u


def _pick_frame(data, key, sel):
    frames = np.unique(data[key]["t_index"][sel])
    return int(np.median(frames))


def _coef_vector(op, data, key, sel, args, rng):
    d = data[key]
    Theta, yv, xs = d["Theta"][sel], d["y"][sel], d["x_pos"][sel]
    res = fit_target(d["names"], Theta, yv, args, rng, xs < 0.0)
    return res["coef_full"], res, d["names"]


# ---------------------------------------------------------------------------
# 1. reconstructed vs actual
# ---------------------------------------------------------------------------

def fig_reconstruction(op, data, key, sel, coef, names, frame, out, args):
    fl = frame_library_2d(op, frame, args)
    m = fl["masks"][key]
    lib = fl["lib2d"][key]
    pred = np.zeros_like(fl["A"])
    for c, nm in zip(coef, names):
        if c != 0.0:
            pred = pred + c * lib[nm]
    actual = fl["tgt2d"][key]

    # self-check: masked recompute must match the stored samples for this frame
    d = data[key]
    fr = d["t_index"] == frame
    if fr.any():
        stored_pred = d["Theta"][fr] @ coef
        here_pred = pred[m]
        finite = np.isfinite(here_pred)
        n = min(here_pred[finite].size, stored_pred.size)
        if n > 0:
            diff = np.nanmax(np.abs(np.sort(here_pred[finite])[:n]
                                    - np.sort(stored_pred)[:n]))
            if diff > 1e-6 * (1 + np.nanstd(stored_pred)):
                print(f"  [WARN] recompute mismatch at frame {frame}: "
                      f"max|diff|={diff:.2e}; diagnostic may not match the fit")

    disp = lambda f: np.where(m, f, np.nan)
    vals = actual[m]; vals = vals[np.isfinite(vals)]
    lim = np.nanpercentile(np.abs(vals), 98) if vals.size else 1.0
    ext = [op["x"][0], op["x"][-1], op["y"][0], op["y"][-1]]

    base_titles = {"dA_dt": "A", "dw_dt": r"w = $|k|^2-1$",
                   "dth_dt": "theta_smooth"}
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))
    ax[0].imshow(fl["base"][key], extent=ext, origin="lower", cmap="viridis",
                 aspect="auto")
    ax[0].set_title(f"{base_titles[key]} (frame {frame})")
    for a, f, ttl in [(ax[1], disp(actual), f"actual {key}"),
                      (ax[2], disp(pred), f"predicted {key}")]:
        im = a.imshow(f, extent=ext, origin="lower", cmap="RdBu_r",
                      vmin=-lim, vmax=lim, aspect="auto"); a.set_title(ttl)
        fig.colorbar(im, ax=a, shrink=0.8)
    res = disp(actual - pred)
    rl = np.nanpercentile(np.abs(res[np.isfinite(res)]), 98) if np.isfinite(res).any() else lim
    im = ax[3].imshow(res, extent=ext, origin="lower", cmap="RdBu_r",
                      vmin=-rl, vmax=rl, aspect="auto")
    ax[3].set_title("residual (actual - predicted)")
    fig.colorbar(im, ax=ax[3], shrink=0.8)
    for a in ax:
        a.set_xlabel("x")
    fig.suptitle(f"{key} reconstruction from fitted library, frame {frame}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  figure -> {out}")


# ---------------------------------------------------------------------------
# 2. feature overlap: z-scored core-row traces + VIF table
# ---------------------------------------------------------------------------

def fig_feature_overlap(op, data, key, frame, out, args):
    fl = frame_library_2d(op, frame, args)
    lib = fl["lib2d"][key]
    core = int(op["core_row"]); x = op["x"]
    names = data[key]["names"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for nm in names:
        if nm == "1":
            continue
        row = lib[nm][core, :]
        s = np.nanstd(row)
        z = (row - np.nanmean(row)) / (s if s > 0 else 1.0)
        ax.plot(x, z, lw=1.0, label=nm)
    ax.set_xlabel("x (core row)"); ax.set_ylabel("z-scored feature")
    ax.set_title(f"{key} library along core row (overlap = collinearity)")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  figure -> {out}")

    # VIF from the masked sample matrix (drop the intercept column)
    Theta = data[key]["Theta"]; nm_list = data[key]["names"]
    sub = Theta[np.random.default_rng(0).choice(
        Theta.shape[0], min(50000, Theta.shape[0]), replace=False)]
    print(f"  VIF for {key} (variance inflation; >10 is strong collinearity):")
    for j, nm in enumerate(nm_list):
        if nm == "1":
            continue
        others = [k for k in range(Theta.shape[1]) if k != j]
        Xo = sub[:, others]; yj = sub[:, j]
        good = np.all(np.isfinite(Xo), axis=1) & np.isfinite(yj)
        if good.sum() < 100:
            continue
        beta, *_ = np.linalg.lstsq(Xo[good], yj[good], rcond=None)
        r2 = 1.0 - np.sum((yj[good] - Xo[good] @ beta) ** 2) \
            / max(np.sum((yj[good] - yj[good].mean()) ** 2), 1e-12)
        vif = 1.0 / max(1.0 - r2, 1e-6)
        flag = "  <-- collinear" if vif > 10 else ""
        print(f"    {nm:>8}: VIF={vif:8.1f}{flag}")


# ---------------------------------------------------------------------------
# 3. query points over the pattern
# ---------------------------------------------------------------------------

def fig_query_points(op, key, frame, out, args, u3d=None, max_pts=6000):
    band, band_up, Xc, Yc = _band_masks(op, args)
    m = band_up if key == "dth_dt" else band
    if "theta_smooth" in op and frame < op["theta_smooth"].shape[-1]:
        m = m & np.isfinite(op["theta_smooth"][:, :, frame])
    u = (u3d[:, :, frame] if (u3d is not None and frame < u3d.shape[2])
         else None)
    ext = [op["x"][0], op["x"][-1], op["y"][0], op["y"][-1]]
    xs, ys = Xc[m], Yc[m]
    if xs.size > max_pts:
        idx = np.random.default_rng(0).choice(xs.size, max_pts, replace=False)
        xs, ys = xs[idx], ys[idx]
    fig, ax = plt.subplots(figsize=(11, 5))
    if u is not None:
        ax.imshow(u, extent=ext, origin="lower", cmap="gray", vmin=-1.3,
                  vmax=1.3, aspect="auto")
    ax.scatter(xs, ys, s=1.0, c="magenta", alpha=0.35, linewidths=0)
    ax.axhline(op["y"][int(op["core_row"])], color="cyan", lw=0.8)
    ax.set_title(f"{key} sampled band points on the pattern, frame {frame} "
                 f"(x_trim={args.x_trim}, band_in={args.band_in})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  figure -> {out}")
    if u is None:
        print("  [note] op has no 'u'; scattered points shown without the "
              "pattern. Re-extract with u saved to see the pattern underlay.")


# ---------------------------------------------------------------------------
# 4. interior-trim scan
# ---------------------------------------------------------------------------

def interior_trim_scan(op, key, window, args, rng, trims, band_ins):
    from copy import copy
    fz = detect_freeze_frame(op.get("n_dislocations"))
    print(f"  interior-trim scan for {key} [{window}] "
          "(holdout R2 vs interior censoring):")
    print(f"    {'x_trim':>7} {'band_in':>8} {'N':>9} {'holdout':>8}")
    for xt in trims:
        for bi in band_ins:
            a = copy(args); a.x_trim = xt; a.band_in = bi
            data = build_time_samples(op, a)
            if key not in data:
                continue
            d = data[key]
            if window == "full" or fz is None or fz < 3:
                sel = np.ones(len(d["y"]), dtype=bool)
            elif window == "pre-freeze":
                sel = d["t_index"] < fz
            else:
                sel = d["t_index"] >= fz
            yv = d["y"][sel]
            if len(yv) < 2000:
                print(f"    {xt:7.1f} {bi:8.1f} {len(yv):9d}   (too few)")
                continue
            xs = d["x_pos"][sel]
            res = fit_target(d["names"], d["Theta"][sel], yv, a, rng, xs < 0.0)
            print(f"    {xt:7.1f} {bi:8.1f} {len(yv):9d} "
                  f"{res['chosen']['r2_test']:8.4f}")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _make_windows(data, key, fz):
    windows = {"full": np.ones(len(data[key]["y"]), dtype=bool)}
    if fz is not None and fz >= 3:
        windows["pre-freeze"] = data[key]["t_index"] < fz
        windows["post-freeze"] = data[key]["t_index"] >= fz
    return windows


def _process_combo(op, data, key, wl, sel, args, rng, out, u3d=None):
    n = int(sel.sum())
    if n < 2000:
        print(f"  {key} [{wl}] too few samples ({n}), skipped")
        return
    frame = args.frame if args.frame is not None else _pick_frame(data, key, sel)
    coef, res, names = _coef_vector(op, data, key, sel, args, rng)
    sel_names = [names[i] for i in np.where(res["support"])[0]]
    print(f"  {key} [{wl}] selected={sel_names}  "
          f"holdout={res['chosen']['r2_test']:.4f}  frame={frame}")
    tag = f"{key}_{wl}"
    fig_reconstruction(op, data, key, sel, coef, names, frame,
                       out / f"reconstruction_{tag}.png", args)
    fig_feature_overlap(op, data, key, frame,
                        out / f"feature_overlap_{tag}.png", args)
    fig_query_points(op, key, frame, out / f"query_points_{tag}.png", args,
                     u3d=u3d)
    if args.trim_scan:
        interior_trim_scan(op, key, wl, args, rng,
                           trims=[args.x_trim, args.x_trim * 2, args.x_trim * 4],
                           band_ins=[0.0, 0.5, 1.0])


def run(args):
    matplotlib.rcParams.update(PAPER_RC)
    rng = np.random.default_rng(args.seed)
    op = load_op(Path(args.op_path))
    op, _, computed = ensure_theta_smooth(op, args.theta_smooth_w)
    fz = detect_freeze_frame(op.get("n_dislocations"))
    print(f"loaded {Path(args.op_path).name}  mu={float(op['mu'])}  freeze={fz}")

    out = Path(args.out_root) if args.out_root else (
        _HERE / "results" / "time_sindy_diagnostics" / Path(args.op_path).stem)
    out.mkdir(parents=True, exist_ok=True)

    data = build_time_samples(op, args)
    u3d = _resolve_u(op, args.op_path, args.u_from)

    targets = (["dA_dt", "dw_dt", "dth_dt"] if args.target == "all"
               else [args.target])
    targets = [k for k in targets if k in data]
    if not targets:
        raise SystemExit("no samples for the requested target(s)")

    for key in targets:
        windows = _make_windows(data, key, fz)
        if args.window != "all":
            if args.window not in windows:
                print(f"  {key}: window '{args.window}' unavailable "
                      f"(freeze={fz}); skipping")
                continue
            windows = {args.window: windows[args.window]}
        for wl, sel in windows.items():
            _process_combo(op, data, key, wl, sel, args, rng, out, u3d=u3d)
    print(f"done -> {out}")


def parse_cli():
    d = _Args(op_path="")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("op_path")
    p.add_argument("--target", default="all",
                   choices=["all", "dA_dt", "dw_dt", "dth_dt"])
    p.add_argument("--window", default="all",
                   choices=["all", "full", "pre-freeze", "post-freeze"])
    p.add_argument("--frame", type=int, default=None)
    p.add_argument("--band_y", type=float, default=d.band_y)
    p.add_argument("--band_in", type=float, default=d.band_in)
    p.add_argument("--x_trim", type=float, default=d.x_trim)
    p.add_argument("--theta_smooth_w", type=float, default=d.theta_smooth_w)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--out_root", default=None)
    p.add_argument("--u_from", default=None,
                   help="file or directory of raw SH .npz for the query-points "
                        "pattern underlay; matched to the OP by source_file")
    p.add_argument("--trim_scan", action="store_true")
    # carry through the fit knobs fit_target needs, from _Args defaults
    a = p.parse_args()
    # carry through every _Args field the fit machinery might read (alpha, eta,
    # thresholds, rel_tol_pareto, n_boot, boot_frac, max_samples, ...) unless the
    # CLI already set it, so fit_target never hits a missing attribute.
    import dataclasses
    for f in dataclasses.fields(_Args):
        if not hasattr(a, f.name):
            setattr(a, f.name, getattr(d, f.name))
    return a


if __name__ == "__main__":
    run(parse_cli())