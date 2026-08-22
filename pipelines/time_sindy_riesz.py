# pipelines/time_sindy_riesz.py
"""
Time-dependent SINDy on the Riesz OP time series (zigzag PGB).

Identifies models for the time derivatives of the three order-parameter
fields, sampled on a band around the grain-boundary core:

    dA/dt      A = A_mono (pointwise monogenic amplitude)
    dw/dt      w = |k|^2 - 1 from kmag_direct (stationary-consistent state)
    dtheta/dt  theta = theta_smooth (display-smoothed reconstructed phase;
               upper half only, the lower half is a mirror, not data)

Time derivatives are centered differences over the saved frames, so this
is meant for the dt = 0.025 production files; the dt = 0.2 files are too
coarse in time for pointwise derivatives (a weak/integral-form variant is
the future fix if even 0.025 proves noisy).

Each library contains the THREE current-density conventions as competing
columns (J_or from the deterministically oriented direct k, J_th from
grad(theta_smooth), rho_d from the doubled-angle director), so the fit
itself reports which J convention is most explanatory for each target.

Every target is fit over three time windows: the full record, the
pre-freeze window (before the detected defect count reaches its final
value: nucleation), and the post-freeze window (relaxation). A model that
holds in one window and not the other is regime information, not failure.

Machinery: STLSQ -> Pareto -> bootstrap imported verbatim from
self_dual_sindy.py; field derivations imported from
run_riesz_metrics_zigzag.py (both must sit in this directory).

Usage:
    python time_sindy_riesz.py <op_path>
    python time_sindy_riesz.py <op_path> --band_y 2.0 --band_in 0.5
Defaults in _Args at the bottom, mirrored by the CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from self_dual_sindy import (  # noqa: E402
    PAPER_RC, panel_label, fit_target,
    fig_library_correlation, fig_pareto,
)
from run_riesz_metrics_zigzag import (  # noqa: E402
    load_op, ensure_theta_smooth, frame_fields, nan_grad,
)
from self_dual_sindy_riesz import fig_contributions  # noqa: E402


@dataclass
class _Args:
    op_path: str = ""
    band_y: float = 2.0          # |y - y_core| <= band_y sample band
    band_in: float = 0.0         # optional core censoring inside the band
    x_trim: float = 5.0
    theta_smooth_w: float = 0.75
    max_samples: int = 400000    # random subsample cap per fit
    alpha: float = 2.0 / 3.0     # kept for fit_target compatibility
    eta: float = 8.0 / 9.0
    thresholds: tuple = tuple(np.geomspace(1e-4, 1.0, 25))
    n_boot: int = 100
    boot_frac: float = 0.5
    rel_tol_pareto: float = 0.01
    seed: int = 0
    out_root: str = ""


# ---------------------------------------------------------------------------
# Sample assembly
# ---------------------------------------------------------------------------

def detect_freeze_frame(n_dis):
    """First frame from which the defect count equals its final nonzero
    value for the rest of the record."""
    n_dis = np.asarray(n_dis)
    if n_dis[-1] <= 0:
        return None
    for i in range(len(n_dis)):
        if np.all(n_dis[i:] == n_dis[-1]):
            return i
    return None


def build_time_samples(op, args):
    """Loop over interior frames, build per-target (library, target) sample
    tables restricted to the band. Returns dict target -> dict with names,
    Theta, y, t_index, x_pos (for holdout and window splits)."""
    x, y = op["x"], op["y"]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    core = int(op["core_row"])
    y_core = float(y[core])
    tt = op["tt"]
    Nt = op["theta"].shape[-1]
    Xc, Yc = np.meshgrid(x, y)

    band = (np.abs(Yc - y_core) <= args.band_y) \
        & (np.abs(Yc - y_core) >= args.band_in) \
        & (Xc > x[0] + args.x_trim) & (Xc < x[-1] - args.x_trim)
    band_up = band & (Yc >= y_core)   # theta: traced half only

    A_far = op["A_far_median"] if "A_far_median" in op else None

    rows = {"dA_dt": [], "dw_dt": [], "dth_dt": []}
    names = {}

    for t in range(1, Nt - 1):
        dt2 = float(tt[t + 1] - tt[t - 1])
        if dt2 <= 0:
            continue
        F = frame_fields(op, t, {})   # rho_d, J_or, curl_or, div_or, J_th ...

        A = op["A_mono"][:, :, t].astype(float)
        af = float(A_far[t]) if A_far is not None else \
            float(np.nanmedian(A[band]))
        dA = A / max(af, 1e-12) - 1.0
        km = op["kmag_direct"][:, :, t].astype(float)
        w = km ** 2 - 1.0
        th = op["theta_smooth"][:, :, t].astype(float)

        Ax, Ay = nan_grad(A, dx, dy)
        Axx, _ = nan_grad(Ax, dx, dy)
        _, Ayy = nan_grad(Ay, dx, dy)
        lapA = Axx + Ayy
        wx, wy = nan_grad(w, dx, dy)
        wxx, _ = nan_grad(wx, dx, dy)
        _, wyy = nan_grad(wy, dx, dy)
        lapw = wxx + wyy
        gradw = np.sqrt(wx ** 2 + wy ** 2)
        tx, ty = nan_grad(th, dx, dy)
        txx, _ = nan_grad(tx, dx, dy)
        _, tyy = nan_grad(ty, dx, dy)

        shared = {
            "1": np.ones_like(w),
            "w": w, "w^2": w ** 2, "dA": dA, "dA^2": dA ** 2,
            "J_or": F["J_or"], "J_th": F["J_th"], "rho_d": F["rho_d"],
            "curl k": F["curl_or"],
        }
        libs = {
            "dA_dt": dict(shared, **{"A": A, "A^3": A ** 3, "Lap A": lapA}),
            "dw_dt": dict(shared, **{"Lap w": lapw, "div k": F["div_or"],
                                     "|grad w|": gradw}),
            "dth_dt": dict(shared, **{"th_xx": txx, "th_yy": tyy}),
        }
        targets = {
            "dA_dt": (op["A_mono"][:, :, t + 1].astype(float)
                      - op["A_mono"][:, :, t - 1].astype(float)) / dt2,
            "dw_dt": (op["kmag_direct"][:, :, t + 1].astype(float) ** 2
                      - op["kmag_direct"][:, :, t - 1].astype(float) ** 2
                      ) / dt2,
            "dth_dt": (op["theta_smooth"][:, :, t + 1].astype(float)
                       - op["theta_smooth"][:, :, t - 1].astype(float)
                       ) / dt2,
        }

        for key in rows:
            m = band_up if key == "dth_dt" else band
            lib = libs[key]
            if key not in names:
                names[key] = list(lib.keys())
            cols = [lib[nm][m] for nm in names[key]]
            yv = targets[key][m]
            T = np.column_stack(cols + [yv])
            fin = np.all(np.isfinite(T), axis=1)
            if not fin.any():
                continue
            T = T[fin]
            xs = Xc[m][fin]
            rows[key].append((T, np.full(T.shape[0], t, dtype=np.int32),
                              xs))

    out = {}
    for key, chunks in rows.items():
        if not chunks:
            continue
        T = np.vstack([c[0] for c in chunks])
        t_idx = np.concatenate([c[1] for c in chunks])
        xs = np.concatenate([c[2] for c in chunks])
        out[key] = dict(names=names[key], Theta=T[:, :-1], y=T[:, -1],
                        t_index=t_idx, x_pos=xs)
    return out


# ---------------------------------------------------------------------------
# Fitting per window
# ---------------------------------------------------------------------------

def fit_window(data, sel, args, rng, label):
    Theta = data["Theta"][sel]
    yv = data["y"][sel]
    xs = data["x_pos"][sel]
    n = len(yv)
    if n < 2000:
        print(f"    [{label}] too few samples ({n}), skipped")
        return None
    if n > args.max_samples:
        idx = rng.choice(n, args.max_samples, replace=False)
        Theta, yv, xs = Theta[idx], yv[idx], xs[idx]
    res = fit_target(data["names"], Theta, yv, args, rng, xs < 0.0)
    sel_names = [data["names"][i] for i in np.where(res["support"])[0]]
    coefs = {data["names"][i]: float(res["coef_full"][i])
             for i in np.where(res["support"])[0]}
    errs = {data["names"][i]: float(res["boots"][:, j].std())
            for j, i in enumerate(np.where(res["support"])[0])}
    print(f"    [{label}] N={len(yv):7d}  selected={sel_names}  "
          f"R2={res['r2_full']:.4f} holdout={res['chosen']['r2_test']:.4f}")
    return dict(res=res, Theta_std=Theta.std(axis=0), n=len(yv),
                coefs=coefs, errs=errs,
                r2_full=float(res["r2_full"]),
                r2_holdout=float(res["chosen"]["r2_test"]))


def fig_window_comparison(out, per_window, target):
    """Holdout R^2 and selected-term count across the three windows."""
    labs = [k for k in ("pre-freeze", "post-freeze", "full") if k in per_window]
    r2 = [per_window[k]["r2_holdout"] for k in labs]
    nt = [len(per_window[k]["coefs"]) for k in labs]
    fig, ax = plt.subplots(figsize=(5.4, 3.2), constrained_layout=True)
    ax.bar(labs, r2, color="C0", alpha=0.85)
    ax.set_ylabel(r"holdout $R^2$", color="C0")
    ax.set_ylim(0, 1)
    ax2 = ax.twinx()
    ax2.plot(labs, nt, "s--", color="0.4")
    ax2.set_ylabel("selected terms", color="0.4")
    ax.set_title(f"target: {target}")
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: _Args):
    matplotlib.rcParams.update(PAPER_RC)
    rng = np.random.default_rng(args.seed)
    op_path = Path(args.op_path)
    print(f"loading {op_path.name}")
    op = load_op(op_path)
    op, _, computed = ensure_theta_smooth(op, args.theta_smooth_w)
    if computed:
        print(f"  theta_smooth computed on the fly (w={args.theta_smooth_w})")
    tt = op["tt"]
    Nt = op["theta"].shape[-1]
    mu = float(op["mu"])

    fz = detect_freeze_frame(op.get("n_dislocations"))
    print(f"  mu={mu}  Nt={Nt}  freeze frame: {fz}"
          + (f" (t={tt[fz]:.3f})" if fz is not None else ""))

    out_root = Path(args.out_root) if args.out_root else (
        _HERE / "results" / "time_sindy_riesz" / op_path.stem)
    out_root.mkdir(parents=True, exist_ok=True)

    print("  assembling samples (all interior frames)...")
    data = build_time_samples(op, args)

    summary = {"op_path": str(op_path), "mu": mu,
               "freeze_frame": fz,
               "freeze_time": float(tt[fz]) if fz is not None else None,
               "band_y": args.band_y, "band_in": args.band_in,
               "targets": {}}

    for key in ("dA_dt", "dw_dt", "dth_dt"):
        if key not in data:
            print(f"  target {key}: no samples, skipped")
            continue
        d = data[key]
        print(f"  target {key}: {d['Theta'].shape[0]} samples, "
              f"library {d['names']}")
        windows = {"full": np.ones(len(d["y"]), dtype=bool)}
        if fz is not None and fz >= 3:
            windows["pre-freeze"] = d["t_index"] < fz
            windows["post-freeze"] = d["t_index"] >= fz
        per_window = {}
        for wl, sel in windows.items():
            r = fit_window(d, sel, args, rng, f"{key} {wl}")
            if r is not None:
                per_window[wl] = r
        if not per_window:
            continue

        # figures from the full-window fit
        fw = per_window.get("full")
        if fw is not None:
            fig_contributions(out_root / f"contributions_{key}.png",
                              d["names"], fw["res"], fw["Theta_std"], key)
            fig_pareto(fw["res"], key, out_root / f"pareto_{key}.png")
        fig_window_comparison(out_root / f"windows_{key}.png",
                              per_window, key)
        sub = rng.choice(d["Theta"].shape[0],
                         min(50000, d["Theta"].shape[0]), replace=False)
        fig_library_correlation(d["names"], d["Theta"][sub],
                                out_root / f"library_correlation_{key}.png")

        summary["targets"][key] = {
            wl: dict(selected=r["coefs"], boot_std=r["errs"],
                     r2_full=r["r2_full"], r2_holdout=r["r2_holdout"],
                     n=r["n"])
            for wl, r in per_window.items()}

    with open(out_root / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {out_root / 'summary.json'}")


def parse_cli() -> _Args:
    d = _Args()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("op_path", nargs="?", default=d.op_path)
    p.add_argument("--band_y", type=float, default=d.band_y)
    p.add_argument("--band_in", type=float, default=d.band_in)
    p.add_argument("--x_trim", type=float, default=d.x_trim)
    p.add_argument("--theta_smooth_w", type=float, default=d.theta_smooth_w)
    p.add_argument("--max_samples", type=int, default=d.max_samples)
    p.add_argument("--out_root", default=d.out_root)
    p.add_argument("--seed", type=int, default=d.seed)
    a = p.parse_args()
    return _Args(**{**d.__dict__, **vars(a)})


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(parse_cli())
    else:
        main(_Args())