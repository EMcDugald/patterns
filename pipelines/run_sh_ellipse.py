# pipelines/run_sh_ellipse.py
"""
Runner for Swift-Hohenberg ellipse simulations (src/sh_sims/ellipse.py), the
modernized version of the 2025 run_sh_eikonal_ellipse.py and
run_sh_eikonal_ellipse_hollow.py. Two modes.

--mode run
    One simulation: square-cell grid (default) or the 2025 legacy grid,
    distance ramp (default) or the 2025 ramp, eikonal / hollow / legacy
    initial condition, or a restart from a frame of an earlier output of this
    runner. Saves, under --outdir:
      <stem>.npz          u [ny, nx, nframes] (float32 by default), x, y, tt,
                          R (unfiltered ramp), E_total (Lyapunov energy per
                          frame), u_last (last frame in float64, for exact
                          restarts), e (energy density, only with --save_e 1),
                          metadata_json (A, B, ramp, ic, h, gb_row_in_crop =
                          the major-axis row, ...). Same key layout as the
                          zigzag runner, so mg_zigzag_theta_t.load_npz reads it.
      <stem>_report.txt   filter effects on R and u0, per-frame table of the
                          Lyapunov energy, bulk amplitude and the relative
                          rate of change of u in the bulk, axis band and wall
                          band; loud flags.
      <stem>_frames.png   u at several frames with the wall drawn.
      <stem>_diagnostics.png  energy, bulk amplitude and rates against time.
      <stem>_setup.png    ramp along both axes (raw and filtered), u0, regions.
      <stem>_config.json  the full configuration.
--mode validate
    Checks that run before any production run.
      V0  ramp filter: max change of R under the integrator's dealiasing
          filter, distance ramp against legacy ramp, at the production
          geometry.
      V1  planted amplitude: uniform R, u0 = a0 cos x; the cos x amplitude
          must reach the stationary amplitude of the same Galerkin truncation
          the solver keeps (odd modes below the dealiasing cutoff), computed
          independently by Newton's method. Single-mode value sqrt(4R/3)
          printed beside it.
      V2  order in h: a short window from a developed state at h, h/2, h/4,
          h/8; observed order log2(|u_h - u_h/2| / |u_h/2 - u_h/4|).
      V3  chunked against single-call integration (the restart path).
      V4  reproduction of the 2025 solver: for each --legacy_npz written by
          the 2025 code, rebuild its grid, ramp and initial condition with the
          legacy options and step with the current core.integrate_sh; every
          frame must agree to round-off.

Usage (single lines, run from pipelines/)
  python run_sh_ellipse.py --mode validate --legacy_npz results/1027_ellipse_legacy_reference/legacy_eik_nx192_ny128_h0.5_T50_n10.npz,results/1027_ellipse_legacy_reference/legacy_hollow_nx192_ny128_h0.5_T50_n10.npz --outdir results/1027_sh_ellipse_validate
  python run_sh_ellipse.py --mode run --ic eikonal --A 210 --B 70 --h 0.5 --tmax 1000 --nsave 125 --outdir results/1027_sh_ellipse_eik_ab3

Changelog
---------
v1 (2026-10-27). First version. The validation numbers measured in the
    porting session are in the session handoff; they are sandbox numbers, and
    only local runs are of record.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

from sh_sims import ellipse as el                      # noqa: E402

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SCRIPT_VERSION = "run_sh_ellipse.py v1 (2026-10-27)"
LAMBDA = 2.0 * np.pi
C_BULK, C_AXIS, C_WALL, C_INK, C_MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8984"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _argdict(args):
    """plain dict of the run arguments (works for the argparse namespace and the _Args class)"""
    return {k: getattr(args, k) for k in dir(args) if not k.startswith("_")}


def build_stem(md):
    if md["grid"] == "legacy":
        g = "legacyLx%.4gLy%.4g_nx%dny%d" % (md["Lx"], md["Ly"], md["Nx"], md["Ny"])
    else:
        g = "A%gB%g_dx%g" % (md["A"], md["B"], md["dx"])
    r = "rdist%g" % md["ramp_w"] if md["ramp"] == "distance" else "rlegacy"
    t0 = md.get("t_offset", 0.0)
    t1 = t0 + md["tmax"]
    return "sh_ellipse_%s_%s_%s_h%g_t%g-%g_N%d" % (md["ic"], g, r, md["h"], t0, t1, md["nsave"])


def frame_diagnostics(tt, uu, regions):
    """per frame: bulk amplitude (sqrt(2) rms u over the bulk) and, between
    consecutive frames, rms(u_{j+1} - u_j) / rms(u_j) / dt in each region"""
    nt = uu.shape[-1]
    amp = np.full(nt, np.nan)
    rates = {k: np.full(nt, np.nan) for k in ("bulk", "axis", "wall")}
    for j in range(nt):
        m = regions["bulk"]
        if m.sum() > 10:
            amp[j] = np.sqrt(2.0) * np.sqrt(np.mean(uu[..., j][m] ** 2))
        if j > 0:
            dt = tt[j] - tt[j - 1]
            for k, mk in regions.items():
                if mk.sum() > 10 and dt > 0:
                    a = np.sqrt(np.mean((uu[..., j][mk] - uu[..., j - 1][mk]) ** 2))
                    b = np.sqrt(np.mean(uu[..., j - 1][mk] ** 2))
                    rates[k][j] = a / max(b, 1e-300) / dt
    return amp, rates


def save_result(res, args, outdir, lines):
    md = res["metadata"]
    md["runner"] = SCRIPT_VERSION
    stem = args.stem if args.stem else build_stem(md)
    nt = res["u"].shape[-1]
    ny = res["u"].shape[0]
    if nt >= ny:
        lines.append("LOUD FLAG: %d frames >= ny = %d; load_npz in mg_zigzag_theta_t decides the axis order "
                     "by u.shape[-1] < u.shape[0] and would transpose this file" % (nt, ny))
    dtype = np.float64 if args.save_dtype == "float64" else np.float32
    out = dict(u=res["u"].astype(dtype), x=res["x"], y=res["y"], tt=res["tt"], R=res["R"].astype(np.float32),
               u_last=res["u"][..., -1].astype(np.float64), metadata_json=json.dumps(md))
    if res["E_total"] is not None:
        out["E_total"] = res["E_total"]
    if args.save_e and res["e"] is not None:
        out["e"] = res["e"].astype(np.float32)
    path = outdir / ("%s.npz" % stem)
    np.savez_compressed(path, **out)
    return stem, path


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _wall(ax, A, B):
    th = np.linspace(0, 2 * np.pi, 600)
    ax.plot(A * np.cos(th), B * np.sin(th), color=C_MUTED, lw=0.6)


def fig_frames(res, path, n_show=6):
    x, y, tt, uu = res["x"], res["y"], res["tt"], res["u"]
    A, B = res["grid"]["A"], res["grid"]["B"]
    nt = uu.shape[-1]
    sel = np.unique(np.round(np.linspace(0, nt - 1, min(n_show, nt))).astype(int))
    asp = (y[-1] - y[0]) / (x[-1] - x[0])
    fig, axs = plt.subplots(len(sel), 1, figsize=(9, max(2.0, 9 * asp) * len(sel) + 0.4), squeeze=False)
    for ax, j in zip(axs[:, 0], sel):
        v = max(np.percentile(np.abs(uu[..., j]), 99.5), 1e-12)
        ax.imshow(uu[..., j], extent=[x[0], x[-1], y[0], y[-1]], origin="lower", cmap="gray", vmin=-v, vmax=v,
                  interpolation="bilinear")
        _wall(ax, A, B)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("t = %.4g   (grey scale +-%.2g)" % (tt[j], v), fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_diagnostics(res, amp, rates, path):
    tt = res["tt"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    if res["E_total"] is not None:
        axs[0].plot(tt, res["E_total"], "-o", ms=2.5, color=C_INK)
    axs[0].set_xlabel("t"); axs[0].set_ylabel("Lyapunov energy (box)"); axs[0].set_title("energy (must not rise)")
    axs[1].plot(tt, amp, "-o", ms=2.5, color=C_BULK, label="bulk sqrt(2) rms u")
    axs[1].axhline(np.sqrt(4 * res["metadata"]["R0"] / 3.0), color=C_MUTED, ls="--", lw=1,
                   label="sqrt(4 R0 / 3)")
    axs[1].set_xlabel("t"); axs[1].set_title("bulk amplitude"); axs[1].legend(fontsize=7, frameon=False)
    for k, c in (("bulk", C_BULK), ("axis", C_AXIS), ("wall", C_WALL)):
        ok = np.isfinite(rates[k]) & (rates[k] > 0) & (tt > 0)
        if ok.any():
            axs[2].loglog(tt[ok], rates[k][ok], "-o", ms=2.5, color=c, label=k)
    axs[2].set_xlabel("t"); axs[2].set_ylabel("rms(du) / rms(u) / dt")
    axs[2].set_title("relative rate of change of u"); axs[2].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_setup(res, path):
    x, y = res["x"], res["y"]
    A, B = res["grid"]["A"], res["grid"]["B"]
    R, Rf, u0 = res["R"], res["R_filtered"], res["u0"]
    jc = int(np.argmin(np.abs(y))); ic = int(np.argmin(np.abs(x)))
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0])
    sel = np.abs(x - A) < 6
    ax.plot(x[sel] - A, R[jc, sel], "-", color=C_INK, lw=1.2, label="R")
    ax.plot(x[sel] - A, Rf[jc, sel], "--", color=C_AXIS, lw=1.0, label="R after the dealiasing filter")
    ax.set_xlabel("x - A along the major axis"); ax.set_title("ramp at the end of the major axis", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax = fig.add_subplot(gs[0, 1])
    sel = np.abs(y - B) < 6
    ax.plot(y[sel] - B, R[sel, ic], "-", color=C_INK, lw=1.2, label="R")
    ax.plot(y[sel] - B, Rf[sel, ic], "--", color=C_AXIS, lw=1.0, label="filtered")
    ax.set_xlabel("y - B along the minor axis"); ax.set_title("ramp at the end of the minor axis", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax = fig.add_subplot(gs[0, 2])
    reg = np.zeros(R.shape)
    reg[res["regions"]["wall"]] = 1; reg[res["regions"]["axis"]] = 2; reg[res["regions"]["bulk"]] = 3
    cmap = matplotlib.colors.ListedColormap(["#ffffff", C_WALL, C_AXIS, C_BULK])
    ax.imshow(reg, extent=[x[0], x[-1], y[0], y[-1]], origin="lower", cmap=cmap, vmin=-0.5, vmax=3.5,
              interpolation="nearest")
    _wall(ax, A, B); ax.set_aspect("equal")
    ax.set_title("regions: wall band (green), axis band (orange), bulk (blue)", fontsize=8)
    ax = fig.add_subplot(gs[1, :2])
    v = max(np.abs(u0).max(), 1e-12)
    ax.imshow(u0, extent=[x[0], x[-1], y[0], y[-1]], origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    _wall(ax, A, B); ax.set_aspect("equal"); ax.set_title("u0 (before the integrator's filter)", fontsize=9)
    ax = fig.add_subplot(gs[1, 2])
    lu = np.log10(np.abs(res["u0_filtered"]) + 1e-16)
    im = ax.imshow(lu, extent=[x[0], x[-1], y[0], y[-1]], origin="lower", cmap="magma", vmin=-12, vmax=0)
    _wall(ax, A, B); ax.set_aspect("equal"); fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("log10 |u0| after the filter", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# run mode
# ---------------------------------------------------------------------------

def cfg_from_args(args):
    keys = ["grid", "A", "B", "dx", "margin", "Lx", "Ly", "Nx", "Ny", "beta", "R0", "ramp", "ramp_w", "ic", "amp",
            "wall_sigma", "d_shell", "edge_sigma", "amp_seed", "h", "tmax", "nsave", "stop_tol", "chunk_T", "wall_margin",
            "axis_band"]
    c = {k: getattr(args, k) for k in keys}
    for k in ("Lx", "Ly"):
        c[k] = None if c[k] is None or c[k] <= 0 else float(c[k])
    for k in ("Nx", "Ny"):
        c[k] = None if c[k] is None or c[k] <= 0 else int(c[k])
    c["t_save_window"] = None if args.t_save_window <= 0 else float(args.t_save_window)
    c["energy"] = True
    return c


def run_mode(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    lines = [SCRIPT_VERSION + "   " + el.MODULE_VERSION, ""]
    cfg = cfg_from_args(args)
    u_init, t_offset = None, 0.0
    if args.ic == "restart":
        if not args.restart_npz:
            raise ValueError("--ic restart needs --restart_npz")
        D = np.load(args.restart_npz, allow_pickle=True)
        md0 = json.loads(str(D["metadata_json"]))
        for k in ("grid", "A", "B", "dx", "margin", "Lx", "Ly", "Nx", "Ny", "beta", "R0", "ramp", "ramp_w",
                  "wall_margin", "axis_band"):
            if k in md0:
                cfg[k] = md0[k]
        if cfg["grid"] == "square":
            cfg["Lx"] = cfg["Ly"] = cfg["Nx"] = cfg["Ny"] = None
        tt0 = np.asarray(D["tt"], float)
        fr = args.restart_frame if args.restart_frame >= 0 else len(tt0) + args.restart_frame
        if fr == len(tt0) - 1 and "u_last" in D.files:
            u_init = np.asarray(D["u_last"], float)
            src = "u_last (float64)"
        else:
            u_init = np.asarray(D["u"][..., fr], float)
            src = "u[..., %d] (%s)" % (fr, D["u"].dtype)
            if D["u"].dtype != np.float64:
                lines.append("LOUD FLAG: restarting from a %s frame; the restart carries about 1e-7 relative "
                             "rounding (only the last frame is stored in float64, as u_last)" % D["u"].dtype)
        t_offset = float(tt0[fr])
        lines.append("restart from %s frame %d (t = %.6g) of %s; grid and ramp taken from its metadata"
                     % (src, fr, t_offset, args.restart_npz))
        cfg["ic"] = "restart"
    with open(outdir / "_last_config.json", "w") as fh:
        json.dump(_argdict(args), fh, indent=2, default=str)
    print("building and integrating ...", flush=True)
    res = el.solve_sh_ellipse(cfg, u_init=u_init, t_offset=t_offset, quiet=not args.verbose)
    md = res["metadata"]
    lines += res["flags"]
    stem, path = save_result(res, args, outdir, lines)
    with open(outdir / ("%s_config.json" % stem), "w") as fh:
        json.dump(dict(args=_argdict(args), metadata=md), fh, indent=2, default=str)
    G = res["grid"]
    lines.append("grid %s: %d x %d, dx %.6g, dy %.6g, box %.6g x %.6g; ellipse A %.6g, B %.6g, medial half-length "
                 "%.6g; major axis on row %d" % (G["grid"], G["nx"], G["ny"], G["dx"], G["dy"], G["Lx"], G["Ly"],
                                                 G["A"], G["B"], md["Xm"], md["gb_row_in_crop"]))
    lines.append("ramp %s (w %.3g), R0 %.3g; ic %s (amp %.3g%s); h %.4g, tmax %.6g, nsave %d%s; stop: %s"
                 % (md["ramp"], md["ramp_w"], md["R0"], md["ic"], md["amp"],
                    ((", wall envelope %.3g" % md["wall_sigma"]) if md["ic"] in ("eikonal", "hollow") else "")
                    + ((", shell to d = %.3g, edge sigma %.3g, interior seed %.3g" %
                        (md["d_shell"], md["edge_sigma"], md["amp_seed"])) if md["ic"] == "hollow" else ""), md["h"], md["tmax"], md["nsave"],
                    (", chunks of %.4g" % md["chunk_T"]) if md["chunk_T"] else "", md["stop"]))
    lines.append("timing: build %.1f s, integration %.1f s" % (md["seconds_build"], md["seconds_run"]))
    lines.append("")
    # filter effects
    reg = res["regions"]
    dR = np.abs(res["R_filtered"] - res["R"])
    lines.append("dealiasing filter applied by core.integrate_sh: max |R_filtered - R| %.3e overall, %.3e inside "
                 "the ellipse beyond 2 from the wall, %.3e in the bulk" %
                 (dR.max(), dR[res["dist"]["inside"] & (res["dist"]["d"] > 2)].max(),
                  dR[reg["bulk"]].max() if reg["bulk"].any() else np.nan))
    du0 = np.abs(res["u0_filtered"] - res["u0"])
    deep2 = res["dist"]["inside"] & (res["dist"]["d"] > 2)
    lines.append("dealiasing filter on u0: max |u0_filtered - u0| %.3e overall, %.3e inside beyond 2 from the "
                 "wall (the eikonal start's crease along the medial axis, and the wall kink when wall_sigma = 0) (max |u0| %.3e)" %
                 (du0.max(), du0[deep2].max() if deep2.any() else np.nan, np.abs(res["u0"]).max()))
    if "hole" in res["ic_info"]:
        hole = res["ic_info"]["hole"]
        deep = hole & (res["dist"]["d"] > md["d_shell"] + 6.0 * md["edge_sigma"] + LAMBDA)
        lines.append("hollow interior (more than 6 edge widths inside the shell edge): max |u0| before the filter "
                     "%.3e (seed %.3g); after the filter %.3e in the hole, %.3e a further wavelength in. This is the "
                     "run's effective interior seed: at growth rate R0 a seed e reaches order one in about "
                     "ln(1/e)/R0" %
                     (np.abs(res["u0"][hole]).max() if hole.any() else np.nan, md["amp_seed"],
                      np.abs(res["u0_filtered"][hole]).max() if hole.any() else np.nan,
                      np.abs(res["u0_filtered"][deep]).max() if deep.any() else np.nan))
    lines.append("regions (points): bulk %d, axis band %d, wall band %d (wall margin %.3g, axis band %.3g "
                 "wavelengths)" % (reg["bulk"].sum(), reg["axis"].sum(), reg["wall"].sum(),
                                   md["wall_margin"], md["axis_band"]))
    lines.append("")
    amp, rates = frame_diagnostics(res["tt"], res["u"], reg)
    Et = res["E_total"]
    if Et is not None and len(Et) > 1:
        rises = np.where(np.diff(Et) > 1e-9 * max(1.0, np.abs(Et).max()))[0]
        if rises.size:
            lines.append("LOUD FLAG: the Lyapunov energy rose between %d pairs of frames, first at t = %.6g"
                         % (rises.size, res["tt"][rises[0] + 1]))
        else:
            lines.append("Lyapunov energy non-increasing over all %d frames (%.6g -> %.6g)" % (len(Et), Et[0], Et[-1]))
    lines.append("%10s %14s %10s %11s %11s %11s" % ("t", "E_total", "bulk amp", "rate bulk", "rate axis",
                                                   "rate wall"))
    for j in range(len(res["tt"])):
        lines.append("%10.4f %14.6g %10.5f %11.3e %11.3e %11.3e" %
                     (res["tt"][j], Et[j] if Et is not None else np.nan, amp[j], rates["bulk"][j],
                      rates["axis"][j], rates["wall"][j]))
    lines.append("(rate = rms(u_j - u_{j-1}) / rms(u_{j-1}) / dt in the region; bulk amp = sqrt(2) rms u in the "
                 "bulk, single-mode saturation sqrt(4 R0 / 3) = %.4f)" % np.sqrt(4 * md["R0"] / 3.0))
    lines.append("")
    if not args.no_plot:
        fig_frames(res, outdir / ("%s_frames.png" % stem))
        fig_diagnostics(res, amp, rates, outdir / ("%s_diagnostics.png" % stem))
        fig_setup(res, outdir / ("%s_setup.png" % stem))
    lines.append("saved %s  (u %s, %d frames, %.1f MB)" % (path.name, args.save_dtype, res["u"].shape[-1],
                                                          path.stat().st_size / 1e6))
    rep = "\n".join(lines)
    (outdir / ("%s_report.txt" % stem)).write_text(rep)
    print(rep)
    print("outputs ->", outdir)
    return 0


# ---------------------------------------------------------------------------
# validate mode
# ---------------------------------------------------------------------------

def galerkin_amplitude(R0, kc, n_modes=None):
    """Stationary 1d SH solution u = sum_m a_m cos(m x) over odd m < kc (the
    modes the dealiased solver keeps), by Newton on the projected equations
    0 = (R0 - (1 - m^2)^2) a_m - P_m[u^3], with the cubic projected exactly on
    a fine grid. Returns the coefficients (a_1 first)."""
    from scipy.optimize import fsolve
    ms = np.array([m for m in range(1, int(np.ceil(kc)) + 1, 2) if m < kc - 1e-9])
    if n_modes:
        ms = ms[:n_modes]
    xg = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    C = np.cos(np.outer(ms, xg))

    def F(a):
        u = a @ C
        P = (C @ (u ** 3)) * (2.0 / len(xg))
        return (R0 - (1 - ms ** 2) ** 2) * a - P
    a0 = np.zeros(len(ms)); a0[0] = np.sqrt(4 * R0 / 3.0)
    a, info, ier, msg = fsolve(F, a0, full_output=True, xtol=1e-14)
    return ms, a, float(np.max(np.abs(F(a))))


def v1_planted_amplitude(lines, R0=0.5):
    lines.append("V1  planted amplitude: uniform R = %.3g, u0 = 0.1 cos x on a box of 8 x 2 wavelengths" % R0)
    nx, ny = 128, 32
    Lx, Ly = 8 * LAMBDA, 2 * LAMBDA
    x = (Lx / nx) * (np.arange(nx) - nx // 2)
    y = (Ly / ny) * (np.arange(ny) - ny // 2)
    X, Y = np.meshgrid(x, y)
    u0 = 0.1 * np.cos(X)
    R = R0 * np.ones_like(X)
    kc = (nx / 4) * 2 * np.pi / Lx                  # largest kept |k_x| under the 1/2-rule mask
    ms, a_true, res = galerkin_amplitude(R0, kc)
    out = []
    for h in (0.5, 0.25):
        tt, uu, ee, *_ = el._integrate(u0, R, Lx, Ly, h, 80.0, 8, False, None, True)
        a1 = 2.0 * np.mean(uu[..., -1] * np.cos(X))
        a1_prev = 2.0 * np.mean(uu[..., -2] * np.cos(X))
        out.append((h, a1, a1_prev))
    lines.append("    kept odd modes m < %.3g: %s; Galerkin truth a1 = %.10f (Newton residual %.1e); single mode "
                 "sqrt(4R/3) = %.10f" % (kc, [int(m) for m in ms], a_true[0], res, np.sqrt(4 * R0 / 3.0)))
    for h, a1, a1p in out:
        lines.append("    h = %.3g: a1(t = 80) = %.10f, relative error against the Galerkin truth %.2e "
                     "(change over the last 10 time units %.1e)" % (h, a1, abs(a1 / a_true[0] - 1), abs(a1 - a1p)))
    worst = max(abs(a1 / a_true[0] - 1) for _, a1, _ in out)
    lines.append("    %s (pass mark: relative error below 1e-6)" % ("PASS" if worst < 1e-6 else "LOUD FLAG: FAIL"))
    return worst


def _small_cfg(**kw):
    c = dict(grid="square", A=40.0, B=20.0, dx=0.45, margin=10.0, ramp="distance", ramp_w=2.0, ic="eikonal",
             amp=0.1, R0=0.5, energy=False)
    c.update(kw)
    return c


def v2_time_error(lines):
    """(a) planted linear problem through core.integrate_sh: amplitude 1e-6,
    uniform R, so u_t = R u - (1 + Lap)^2 u to within 1e-12 relative and the
    exact answer is exp(T (R - (1 - |k|^2)^2)) on every kept Fourier mode;
    (b) self-convergence on the nonlinear ellipse problem from a developed
    state, with the step error set against the signal rms(u(T) - u(0))."""
    lines.append("V2a time order on a planted linear problem: periodic box 64 x 64 at dx 0.45, uniform R = 0.5, "
                 "smooth random start of amplitude 1e-6, window T = 4")
    from scipy.fft import fft2, ifft2, fftfreq
    from sh_sims.core import dealias_mask
    n, dx = 64, 0.45
    L = n * dx
    rng = np.random.default_rng(3)
    k = 2 * np.pi * fftfreq(n, dx)
    KX, KY = np.meshgrid(k, k)
    spec = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * np.exp(-((np.hypot(KX, KY) - 1.0) ** 2) / 0.1)
    u0 = np.real(ifft2(spec))
    u0 *= 1e-6 / np.abs(u0).max()
    R = 0.5 * np.ones_like(u0)
    ind = dealias_mask(n, n)
    U0 = fft2(u0); U0[ind] = 0
    exact = np.real(ifft2(np.exp(4.0 * (0.5 - (1 - KX ** 2 - KY ** 2) ** 2)) * U0))
    errs = []
    for h in (0.5, 0.25, 0.125, 0.0625):
        tt, uu, ee, *_ = el._integrate(u0, R, L, L, h, 4.0, 1, False, None, True)
        errs.append(np.abs(uu[..., -1] - exact).max() / np.abs(exact).max())
    ords = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    lines.append("    relative max error at h = 0.5, 0.25, 0.125, 0.0625: %s; orders %s   (ETDRK4: 4)" %
                 (", ".join("%.2e" % e for e in errs), ", ".join("%.2f" % o for o in ords)))
    lines.append("    %s (pass mark: last order above 3.5)" % ("PASS" if ords[-1] > 3.5 else "LOUD FLAG: below 3.5"))
    lines.append("V2b step error on the nonlinear ellipse problem: A = 40, B = 20, dx 0.45; developed state at t = 20 "
                 "(from h 0.05), then a window of T = 4")
    base = el.solve_sh_ellipse(_small_cfg(h=0.05, tmax=20.0, nsave=1), quiet=True)
    u_start = base["u"][..., -1]
    hs = (0.5, 0.25, 0.125, 0.0625, 0.03125)
    sols = {h: el.solve_sh_ellipse(_small_cfg(h=h, tmax=4.0, nsave=1), u_init=u_start, quiet=True)["u"][..., -1]
            for h in hs}
    signal = np.sqrt(np.mean((sols[hs[-1]] - u_start) ** 2))
    e2 = [np.sqrt(np.mean((sols[hs[i]] - sols[hs[i + 1]]) ** 2)) for i in range(len(hs) - 1)]
    o2 = [np.log2(e2[i] / e2[i + 1]) for i in range(len(e2) - 1)]
    lines.append("    signal rms(u(T) - u(0)) = %.3e" % signal)
    for i in range(len(e2)):
        lines.append("    h = %-7.4g rms |u_h - u_h/2| %.3e  = %.2e of the signal" % (hs[i], e2[i], e2[i] / signal))
    lines.append("    observed orders %s: the Cox-Matthews ETDRK4 shows stiff order reduction on the nonlinear "
                 "problem (order rises toward 4 only as h shrinks); read the error against the signal, not the "
                 "order" % ", ".join("%.2f" % o for o in o2))
    lines.append("    %s (pass mark: step error at h = 0.5 below 1e-3 of the signal)" %
                 ("PASS" if e2[0] / signal < 1e-3 else "LOUD FLAG: above 1e-3 of the signal"))
    return errs, e2, signal


def v3_chunks(lines):
    lines.append("V3  chunked (restart) against single-call integration: A = 40, B = 20, t = 0 to 20, h 0.2, "
                 "frames every 2")
    single = el.solve_sh_ellipse(_small_cfg(h=0.2, tmax=20.0, nsave=10), quiet=True)
    chunked = el.solve_sh_ellipse(_small_cfg(h=0.2, tmax=20.0, nsave=1, chunk_T=2.0), quiet=True, log=lambda s: None)
    n = min(single["u"].shape[-1], chunked["u"].shape[-1])
    dt_ok = np.allclose(single["tt"][:n], chunked["tt"][:n])
    diff = np.sqrt(np.mean((single["u"][..., :n] - chunked["u"][..., :n]) ** 2, axis=(0, 1)))
    step = np.sqrt(np.mean(np.diff(single["u"][..., :n], axis=-1) ** 2, axis=(0, 1)))
    ratio = diff[1:].max() / step.min()
    lines.append("    frame times agree: %s; rms |u_single - u_chunked| per frame up to %.3e; smallest rms change "
                 "between frames %.3e; worst ratio %.2e" % (dt_ok, diff.max(), step.min(), ratio))
    lines.append("    (the restart zeroes the integrator's masked band, which a continuous run carries at one step's "
                 "worth)")
    lines.append("    %s (pass mark: difference below 1e-3 of the smallest frame-to-frame change)" %
                 ("PASS" if ratio < 1e-3 else "LOUD FLAG: above 1e-3"))
    return ratio


def v4_legacy(lines, paths):
    out = []
    for p in paths:
        p = p.strip()
        if not p:
            continue
        lines.append("V4  legacy reproduction: %s" % p)
        if not Path(p).exists():
            lines.append("    LOUD FLAG: file not found; skipped")
            continue
        D = np.load(p, allow_pickle=True)
        Lx, Ly = float(D["Lx"]), float(D["Ly"])
        xx, yy, tt_old, uu_old = D["xx"], D["yy"], np.asarray(D["tt"], float), np.asarray(D["uu"], float)
        h = float(D["h"]); R0 = float(D["Rscale"]); beta = float(D["beta"]); amp = float(D["amplitude"])
        ic = "hollow_legacy" if "hollow" in Path(p).name else "eikonal_legacy"
        cfg = dict(grid="legacy", Lx=Lx, Ly=Ly, Nx=len(xx), Ny=len(yy), beta=beta, R0=R0, ramp="legacy", ic=ic,
                   amp=amp, h=h, tmax=float(tt_old[-1]), nsave=len(tt_old) - 1, energy=False)
        t0 = time.time()
        res = el.solve_sh_ellipse(cfg, quiet=True)
        gx = np.abs(res["x"] - xx).max(); gy = np.abs(res["y"] - yy).max()
        dR = np.abs(res["R"] - np.asarray(D["R"], float)).max()
        dt_ = np.abs(res["tt"] - tt_old).max()
        diff = np.abs(res["u"] - uu_old)
        per = diff.max(axis=(0, 1))
        rel = diff.max() / max(np.abs(uu_old).max(), 1e-300)
        lines.append("    ic %s on the legacy grid %d x %d (%.1f s): grid max |dx| %.1e, |dy| %.1e; ramp max |dR| "
                     "%.1e; times max |dt| %.1e" % (ic, len(xx), len(yy), time.time() - t0, gx, gy, dR, dt_))
        lines.append("    max |u_new - u_2025| per frame: %s" % ", ".join("%.1e" % v for v in per))
        lines.append("    overall max |u_new - u_2025| %.3e (max |u| %.3e, relative %.2e)  %s" %
                     (diff.max(), np.abs(uu_old).max(), rel, "PASS" if rel < 1e-8 else "LOUD FLAG: above 1e-8"))
        out.append(rel)
    return out


def v0_ramp(lines, A=210.0, B=70.0, dx=0.45, margin=15.0, w=2.0):
    lines.append("V0  ramp under the integrator's dealiasing filter at the production geometry (A %g, B %g, dx %g)"
                 % (A, B, dx))
    G = el.make_grid("square", A=A, B=B, dx=dx, margin=margin)
    X, Y = np.meshgrid(G["x"], G["y"])
    dist = el.wall_distance(el.EllipseGeom(A, B), X, Y)
    lines.append("    grid %d x %d; the 1/2-rule mask keeps |k| below about %.3f" % (G["nx"], G["ny"],
                                                                                   (G["nx"] / 4) * 2 * np.pi / G["Lx"]))
    for kind in ("distance", "legacy"):
        R = el.build_ramp(kind, X, Y, dist, A, B, 0.5, w)
        dR = np.abs(el.dealias_filter(R) - R)
        ins = dist["inside"]
        lines.append("    %-8s ramp: max |R_filtered - R| %.3e overall; %.3e inside beyond 2 from the wall; %.3e "
                     "inside beyond 4" % (kind, dR.max(), dR[ins & (dist["d"] > 2)].max(),
                                          dR[ins & (dist["d"] > 4)].max()))


def validate_mode(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    lines = [SCRIPT_VERSION + "   " + el.MODULE_VERSION, "validation", ""]
    t0 = time.time()
    v0_ramp(lines, A=args.A, B=args.B, dx=args.dx, margin=args.margin, w=args.ramp_w)
    lines.append("")
    v1_planted_amplitude(lines, R0=args.R0)
    lines.append("")
    v2_time_error(lines)
    lines.append("")
    v3_chunks(lines)
    lines.append("")
    if args.legacy_npz:
        v4_legacy(lines, args.legacy_npz.split(","))
    else:
        lines.append("V4  skipped (no --legacy_npz)")
    lines.append("")
    lines.append("elapsed %.1f s" % (time.time() - t0))
    rep = "\n".join(lines)
    (outdir / "validate_report.txt").write_text(rep)
    print(rep)
    print("outputs ->", outdir)
    return 0


# ---------------------------------------------------------------------------

def main(args):
    if args.mode == "validate":
        return validate_mode(args)
    return run_mode(args)


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    mode = "run"                 # run | validate
    outdir = str(_HERE / "results" / "run_sh_ellipse")
    stem = ""                    # output file stem (empty: built from the parameters)
    # grid
    grid = "square"              # square (dx = dy, FFT-friendly, axis on a row) | legacy (the 2025 box)
    A = 210.0                    # semi-major axis (square grid)
    B = 70.0                     # semi-minor axis (square grid)
    dx = 0.45                    # grid spacing (square grid)
    margin = 15.0                # box margin beyond the wall on every side (square grid)
    Lx = -1.0                    # legacy grid only (-1: unused)
    Ly = -1.0
    Nx = -1
    Ny = -1
    beta = 0.45                  # legacy grid: semi-axes beta Lx, beta Ly
    # ramp
    R0 = 0.5
    ramp = "distance"            # distance (R0 tanh(s / w), uniform width) | legacy (the 2025 formula)
    ramp_w = 2.0                 # distance ramp width
    # initial condition
    ic = "eikonal"               # eikonal | hollow | eikonal_legacy | hollow_legacy (legacy grid) | restart
    amp = 0.1
    wall_sigma = 2.0             # eikonal and hollow: erf wall envelope width (0: hard cut at the wall, as in the stand-in)
    d_shell = 20.0               # hollow: shell thickness from the wall (about 0.1 A at A = 210, as in 2025)
    edge_sigma = 3.0             # hollow: Gaussian width of the shell's inner edge (erfc profile)
    amp_seed = 0.0               # hollow: deliberate interior seed amplitude (the filter leaves a floor near 1e-7)
    restart_npz = ""             # restart: an output of this runner
    restart_frame = -1           # restart: frame index (-1: last, read from u_last in float64)
    # time stepping
    h = 0.5
    tmax = 100.0                 # duration of this run (a restart adds it to the restart time)
    nsave = 10                   # frames after t = 0 (per chunk in chunked mode)
    t_save_window = -1.0         # > 0: the nsave frames span the last t_save_window of the run
    chunk_T = 0.0                # > 0: integrate in chunks of this length (needed for stop_tol)
    stop_tol = 0.0               # > 0 with chunk_T: stop when the bulk rate falls below this per unit time
    # regions (diagnostics only)
    wall_margin = 2.0            # wavelengths
    axis_band = 1.5              # wavelengths
    # output
    save_dtype = "float32"       # float32 | float64
    save_e = 0                   # 1: also save the energy density field
    no_plot = 0
    verbose = 0                  # 1: let core.integrate_sh print its step counter
    # validate
    legacy_npz = ""              # comma-separated 2025 reference files for V4


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in vars(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    if len(sys.argv) > 1:
        sys.exit(main(p.parse_args()))
    sys.exit(main(_Args()))
