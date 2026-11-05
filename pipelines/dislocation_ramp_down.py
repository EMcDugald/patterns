"""
dislocation_ramp_down.py  v1.0  (2 Nov 2026)   do dislocations survive when the far-field angle is lowered?

Plain terms. Every run of the large sweep starts from a knee, so it only says where the boundary changes as the
angle goes UP (knee to intermediate at mu 0.74, intermediate to dislocation at 0.82, detector width 0.4). This
script runs the same Swift-Hohenberg zigzag set-up starting from DISLOCATIONS and lowers the angle, so the survey
can say where the boundary changes on the way DOWN. Same transitions both ways: no hysteresis. Dislocations
surviving below 0.82: a hysteresis window (Alan's question of 2 Nov; plan in HYSTERESIS_PLAN_1102.md).

The angle cannot change inside a run (the periodic box is a whole number of half periods pi/k1, k1 = cos alpha), so
"lowering the angle" means: relax a dislocated boundary at mu_seed (from a knee, as in the large sweep), then remap
its full field onto the box of the next lower mu and relax again there, and so on (mode ramp); or remap the one seed
onto every target directly (mode jump).

The remap. Around each boundary line (y = yl for the lower, y = yr for the upper), the new field at (x, y) is the
old field at
    x_old = x * k1 / k1_old,       y_old = y_line_old + (y - y_line) * k2 / k2_old,
which carries far-field stripes of wavevector (k1_old, +-k2_old) onto (k1, +-k2), both of unit length, so the far
field arrives at the preferred wavenumber. In x the box maps onto the old box exactly (both are n_periods half
periods), so x stays periodic and the number of dislocations per boundary (n_periods) is kept; their spacing
becomes L = pi/k1, the spacing the new angle requires. In y the map only compresses (k2 / k2_old < 1 when mu falls),
so every sampled point lies inside the old half-domain of the same boundary. The two halves are joined by tanh
blends of width Ly/40 at y = 0 (as in build_pgb_zigzag_ic) and across the periodic edge y = +-Ly/2. Only lowering
mu is allowed: raising it would sample outside the old half-domain (loud flag, the target is skipped).

Outputs (under --outdir, anchored to this script's folder when relative; default results/dislocation_ramp_down/):
  raw/<stem>.npz        one file per run in exactly the runner's format (u, e, x, y, x_full, y_full, tt,
                        metadata_json; cropped about the lower boundary as in the large sweep), so the stationary
                        survey reads them unchanged:  large_sweep_charge_circ_v13.py --raw <outdir>/raw.
                        The seed run (from a knee) is saved here too, tagged knee_seed. metadata_json carries the
                        runner's keys plus ic, seed_mu, seed_from, remap_order.
  seeds/<tag>_full.npz  the full-domain final field of every run (float32), the seed of the next ramp step; a
                        rerun resumes from these.
  figures/<stem>_ic.png the crop of the remapped initial condition beside the final frame (one per target).
  report.txt, config.json.

Checks printed and written to the report (loud flags, nothing blocks):
  identity remap: the seed remapped onto its own angle must reproduce itself (max |difference| / max |u|; flag above
    1e-3, which would mean the interpolation or the grids are wrong);
  every remapped initial condition: its min and max, and the fraction of the domain where the blend weights are
    between 0.01 and 0.99 (the seams).

Python 3.8 compatible. Imports the solver (sh_sims.core.integrate_sh) and the grid and crop helpers
(sh_sims.pgb_zigzag) unchanged, the way run_sh_pgb_zigzag.py does; nothing in src/ is edited. Measures nothing:
every measure is the survey's.

Changelog
  v1.0  (2 Nov 2026) first version. Defaults reproduce the large sweep of 24 Oct exactly (Nx 512, n_periods 18,
        Ny_factor 5, R 0.5, h 0.25, t 200, crop 256 x 256 about the lower boundary), with 40 saved frames (every 5
        time units) so the loss of circulation can be timed; the survey's t = 100 frame is then --frame_early 20.
        Sandbox check: identity remap and a short ramp on a reduced grid with the same solver (not of record).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from sh_sims.core import integrate_sh                                            # noqa: E402
from sh_sims.pgb_zigzag import build_pgb_zigzag_ic, crop_around_gb, _check_gb_commensurability  # noqa: E402

VERSION = "dislocation_ramp_down v1.0"


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def geometry(mu, a):
    """the box of one angle, exactly as the runner builds it (the knee IC comes with it)"""
    return build_pgb_zigzag_ic(Nx=a.Nx, mu=mu, n_periods=a.n_periods, Ny_factor=a.Ny_factor,
                               Rscale=a.Rscale, amp=a.amp, y_centering=a.y_centering)


def upper_weight(yy, Ly):
    """weight of the upper boundary's half: 1 inside it, 0 inside the lower half, tanh blends of width Ly/40 at
    y = 0 and across the periodic edge y = +-Ly/2"""
    w = Ly / 40.0
    W = 0.5 * (1.0 + np.tanh(yy / w))
    hi = yy > Ly / 4.0
    lo = yy < -Ly / 4.0
    W[hi] = 0.5 * (1.0 + np.tanh((Ly / 2.0 - yy[hi]) / w))
    W[lo] = 0.5 * (1.0 - np.tanh((Ly / 2.0 + yy[lo]) / w))
    return W


def _sample(u_old, g_old, Xs, Ys, order, flags):
    """u_old at physical points (Xs, Ys) of the old box: x periodic, y inside the old domain"""
    dx = float(g_old["dx"])
    ix = (Xs - g_old["xx"][0]) / dx
    iy = (Ys - g_old["yy"][0]) / dx
    try:
        return map_coordinates(u_old, [iy, ix], order=order, mode="grid-wrap")
    except (ValueError, RuntimeError):
        if "wrap" not in " ".join(flags):
            flags.append("scipy has no mode 'grid-wrap' (older than 1.6): used mode 'wrap', which is off by one grid"
                         " cell at the periodic edge in x; check the identity-remap error")
        return map_coordinates(u_old, [iy, ix], order=order, mode="wrap")


def remap(u_old, mu_old, mu_new, a, flags):
    """the old full-domain field (relaxed at mu_old) on the box of mu_new, by the map in the module docstring"""
    g_old, g_new = geometry(mu_old, a), geometry(mu_new, a)
    k1o, k2o, k1n, k2n = g_old["k1"], g_old["k2"], g_new["k1"], g_new["k2"]
    f = k2n / k2o
    if mu_new > mu_old + 1e-12:
        raise ValueError("mu %.3f above the seed's %.3f: the map would sample outside the old half-domain" % (mu_new, mu_old))
    X, Y = g_new["X"], g_new["Y"]
    Xs = X * k1n / k1o
    Ys_lo = g_old["yl"] + (Y - g_new["yl"]) * f
    Ys_up = g_old["yr"] + (Y - g_new["yr"]) * f
    u_lo = _sample(u_old, g_old, Xs, Ys_lo, a.remap_order, flags)
    u_up = _sample(u_old, g_old, Xs, Ys_up, a.remap_order, flags)
    W = upper_weight(Y, g_new["Ly"])
    seam = float(np.mean((W > 0.01) & (W < 0.99)))
    return (1.0 - W) * u_lo + W * u_up, g_new, seam


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------

def stem_for(mu, a, tag):
    return ("sh_pgb_zigzag_mu%.3f_T%d_N%d_nx%d_Ny%s_%s_%s"
            % (mu, int(a.tmax), a.nsave, a.Nx, a.crop_Ny if a.crop_Ny else "full", a.which_gb, tag))


def run_one(u0, g, mu, a, tmax, nsave, ic_label, extra_meta):
    """integrate SH from u0 on the box g, crop as the runner does; returns (result dict, full final field)"""
    Ny = int(round(a.Ny_factor * a.Nx))
    _check_gb_commensurability(Ny, a.y_centering)
    tt, uu, ee, _, _, _, _ = integrate_sh(u0, g["R"], g["Lx"], g["Ly"], h=a.h, tmax=tmax, nsave=nsave, energy=True)
    cu = crop_around_gb(g["xx"], g["yy"], uu, g["yl"], g["yr"], a.which_gb, a.crop_Nx, a.crop_Ny)
    ce = crop_around_gb(g["xx"], g["yy"], ee, g["yl"], g["yr"], a.which_gb, a.crop_Nx, a.crop_Ny)
    lam_x = np.pi / g["k1"]
    lam_y = np.pi / g["k2"] if g["k2"] > 0 else np.inf
    gb_y = g["yl"] if a.which_gb == "lower" else g["yr"]
    meta = dict(mu=float(mu), alpha=float(np.arcsin(mu)), k1=float(g["k1"]), k2=float(g["k2"]), lambda_x=float(lam_x),
                lambda_y=float(lam_y), Lx_periods=float(g["Lx"] / lam_x), Ly_periods=float(g["Ly"] / lam_y),
                n_periods=int(a.n_periods), Nx=int(a.Nx), Ny=int(Ny), Lx=float(g["Lx"]), Ly=float(g["Ly"]),
                dx=float(g["dx"]), dy=float(g["dy"]), y_centering=a.y_centering, h=float(a.h), tmax=float(tmax),
                nsave=int(nsave), t_save_window=None, Rscale=float(a.Rscale), amp=float(a.amp), which_gb=a.which_gb,
                crop_Nx=a.crop_Nx, crop_Ny=a.crop_Ny, crop_x_slice=cu["x_slice"], crop_y_slice=cu["y_slice"],
                gb_row_in_crop=int(np.argmin(np.abs(cu["yy"] - gb_y))),
                crop_Ny_parity=("odd" if a.crop_Ny and a.crop_Ny % 2 else "even" if a.crop_Ny else "no_crop"),
                ic=ic_label, script=VERSION)
    meta.update(extra_meta)
    res = dict(tt=tt, x=cu["xx"], y=cu["yy"], x_full=g["xx"], y_full=g["yy"], u=cu["data"], e=ce["data"],
               metadata_json=json.dumps(meta))
    return res, uu[..., -1].copy()


def save_run(res, path):
    np.savez_compressed(path, **res)


def fig_ic(res, path, mu, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u = res["u"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.4))
    for ax, j, lab in ((axs[0], 0, "initial (remapped)"), (axs[1], -1, "final")):
        ax.imshow(u[..., j], origin="lower", cmap="gray")
        ax.set_title("%s, t = %.1f" % (lab, res["tt"][j]), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("mu %.3f (%.1f deg): %s" % (mu, np.degrees(np.arcsin(mu)), title), fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def target_mus(a):
    if a.mus.strip():
        mus = [float(v) for v in a.mus.split(",") if v.strip()]
    else:
        n = int(round((a.mu_start - a.mu_stop) / a.dmu)) + 1
        mus = [round(a.mu_start - i * a.dmu, 6) for i in range(n)]
    return sorted(mus, reverse=True)


def main(a):
    out = Path(a.outdir)
    if not out.is_absolute():
        out = HERE / out
    raw, seeds, figs = out / "raw", out / "seeds", out / "figures"
    for d in (raw, seeds, figs):
        d.mkdir(parents=True, exist_ok=True)
    mus = target_mus(a)
    with open(out / "config.json", "w") as fh:
        json.dump(dict(vars(a), version=VERSION, targets=mus), fh, indent=2, default=str)
    lines = [VERSION, "mode %s, seed mu %.3f (from a knee, t %.0f), targets %s" % (
        a.mode, a.seed_mu, a.seed_t, ", ".join("%.3f" % m for m in mus)), ""]
    flags = []
    T0 = time.time()

    # 1. the seed: a knee relaxed at seed_mu, as in the large sweep (resumed if its full field exists)
    seed_tag = "knee_seed"
    seed_full = seeds / ("mu%.3f_%s_full.npz" % (a.seed_mu, seed_tag))
    if seed_full.exists() and not a.overwrite:
        u_seed = np.load(seed_full)["u"].astype(float)
        lines.append("seed: loaded %s" % seed_full.name)
    else:
        t0 = time.time()
        g = geometry(a.seed_mu, a)
        res, u_seed = run_one(g["u0"], g, a.seed_mu, a, a.seed_t, a.nsave, "knee (seed of the ramp)", dict(seed_mu=None, seed_from=None))
        save_run(res, raw / (stem_for(a.seed_mu, a, seed_tag) + ".npz"))
        np.savez_compressed(seed_full, u=u_seed.astype(np.float32), mu=a.seed_mu)
        lines.append("seed: mu %.3f relaxed from a knee to t %.0f in %.0f s" % (a.seed_mu, a.seed_t, time.time() - t0))
    print(lines[-1])

    # 2. identity check of the remap
    u_id, _, _ = remap(u_seed, a.seed_mu, a.seed_mu, a, flags)
    err = float(np.max(np.abs(u_id - u_seed)) / max(np.max(np.abs(u_seed)), 1e-12))
    W = upper_weight(geometry(a.seed_mu, a)["Y"], geometry(a.seed_mu, a)["Ly"])
    core = (W < 0.01) | (W > 0.99)
    err_core = float(np.max(np.abs(u_id - u_seed)[core]) / max(np.max(np.abs(u_seed)), 1e-12))
    lines.append("identity remap at mu %.3f: max |difference| / max |u| = %.2e away from the seams, %.2e with them"
                 " (the seams blend the two halves, so only the first is a test)" % (a.seed_mu, err_core, err))
    if err_core > 1e-3:
        flags.append("LOUD: identity remap error %.2e above 1e-3 away from the seams" % err_core)
    print(lines[-1])

    # 3. the targets
    prev_mu, prev_u = a.seed_mu, u_seed
    for mu in mus:
        if (a.mode == "ramp" and mu >= prev_mu - 1e-9) or (a.mode == "jump" and mu >= a.seed_mu - 1e-9):
            lines.append("FLAG: mu %.3f is not below its seed; skipped (only lowering the angle is supported)" % mu)
            print(lines[-1]); continue
        src_mu, src_u = (prev_mu, prev_u) if a.mode == "ramp" else (a.seed_mu, u_seed)
        tag = "%sdown_from%.3f" % ("ramp" if a.mode == "ramp" else "jump", src_mu)
        stem = stem_for(mu, a, tag)
        full = seeds / ("mu%.3f_%s_full.npz" % (mu, tag))
        if full.exists() and (raw / (stem + ".npz")).exists() and not a.overwrite:
            prev_mu, prev_u = mu, np.load(full)["u"].astype(float)
            lines.append("mu %.3f: exists, resumed (%s)" % (mu, stem)); print(lines[-1]); continue
        t0 = time.time()
        try:
            u0, g, seam = remap(src_u, src_mu, mu, a, flags)
        except ValueError as exc:
            lines.append("FLAG: mu %.3f: %s" % (mu, exc)); print(lines[-1]); continue
        res, u_end = run_one(u0, g, mu, a, a.tmax, a.nsave, "dislocations remapped from mu %.3f (%s)" % (src_mu, a.mode),
                             dict(seed_mu=float(a.seed_mu), seed_from=float(src_mu), remap_order=int(a.remap_order)))
        save_run(res, raw / (stem + ".npz"))
        np.savez_compressed(full, u=u_end.astype(np.float32), mu=mu)
        fig_ic(res, figs / (stem + "_ic.png"), mu, "from mu %.3f (%s)" % (src_mu, a.mode))
        lines.append("mu %.3f (%.1f deg): from %.3f; initial u in [%.3f, %.3f], seams %.1f%% of the domain; final u in"
                     " [%.3f, %.3f]; %.0f s -> %s" % (mu, np.degrees(np.arcsin(mu)), src_mu, u0.min(), u0.max(), 100 * seam,
                                                   u_end.min(), u_end.max(), time.time() - t0, stem))
        print(lines[-1])
        prev_mu, prev_u = mu, u_end

    lines += ["", "flags:"] + (flags if flags else ["none"]) + ["", "elapsed %.0f s" % (time.time() - T0),
              "next: python large_sweep_charge_circ_v13.py --raw %s --outdir %s --frame_early %d"
              % (raw, out / "survey", a.nsave // 2)]
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))
    return 0


class _Args:
    """PyCharm-friendly defaults (the large sweep of 24 Oct, from dislocations); mirrored by the CLI below."""
    outdir = str(Path("results") / "dislocation_ramp_down")
    mode = "ramp"                  # ramp: each angle seeded by the one above it; jump: every angle from the one seed
    seed_mu = 0.87                 # the seed: relaxed from a knee here (a dislocation boundary in the large sweep)
    seed_t = 200.0                 # the seed's relaxation time
    mu_start = 0.86                # targets mu_start, mu_start - dmu, ..., mu_stop (used when mus is empty)
    mu_stop = 0.65
    dmu = 0.01
    mus = ""                       # or an explicit comma-separated list, e.g. "0.86,0.84,0.80"
    tmax = 200.0                   # relaxation time at every target (the large sweep's)
    nsave = 40                     # frames after t = 0 (every 5 time units at tmax 200; the survey's t = 100 is frame 20)
    h = 0.25
    Nx = 512
    n_periods = 18
    Ny_factor = 5.0
    Rscale = 0.5
    amp = 1.0
    y_centering = "node"
    which_gb = "lower"
    crop_Nx = 256
    crop_Ny = 256
    remap_order = 3                # spline order of the remap (3: cubic)
    overwrite = 0                  # 0: resume from existing seeds and raw files; 1: recompute everything


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [k for k in vars(_Args) if not k.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    sys.exit(main(p.parse_args()))
