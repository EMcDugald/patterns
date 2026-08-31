"""
crop_windowing_forensics.py  (v1.1)
-----------------------------------
Why FFT-based extraction fails on the shallow-sweep crops, and what the
x taper does about it: a reproducible record of the windowing finding
of the session-7 handoff, with a taper sweep.

    python crop_windowing_forensics.py [--crop-file F] [--periodic-file F]
                                       [--mu 0.4 --R 0.5 --beta 1.0]
                                       [--routes uhu demod] [--out-root DIR]

The fact: the sweep files store 256 of 512 columns of a box that is
n_periods * pi/k1 wide (pgb_zigzag.make_x_grid; pi/k1 is the HALF
stripe period, so n_periods = 18 gives 9 full periods and the crop holds
4.5), with no ramp. The 0312 knee files are 512 columns, 12 full periods,
periodic, with a sharp tanh ramp. Every spectral operator (Hilbert along
x, spectral derivatives, Gaussian averages) assumes the row is periodic;
on the crop the periodic extension has a jump and the spectrum leaks.

What is measured, on a planted RCN knee (beta known) and optionally on
one real crop and one real periodic file:
  1. the same knee on a periodic 512^2 box with the old ramp (reference)
  2. the crop under a sweep of x windows: box (no window), the old
     runner's tanh ramp (2.5 percent margins, tanhscale 120), and Tukey
     windows of alpha 0.1 .. 1.0 (alpha 1 = Hann)
  for each window and route: the profile regression c and R^2 of
  div k = b - c w on the band, the far-field w offset, the k1 ripple
  along a far-field row, and the columns that survive the window.
Outputs: forensics.csv, fig_taper_sweep.png (c, R^2, far-field w, k1
ripple against window), fig_w_maps.png (w on the crop for box, tanh
ramp, Tukey 0.5), fig_profiles.png (wbar and dk2bar for the same three).

Measured in the sandbox run (planted beta 1 on the mu 0.375, R 0.5 crop
grid; uHu 1.571 / demod): periodic reference c 0.810 / 1.000, R^2
0.9997 / 1.0000. Crop, box window: 0.369 / 1.006, R^2 0.30 / 0.999
(demod survives the box because the intercept and the far-field
renormalization absorb the leakage; its k1 ripple is still 0.020).
Old-runner tanh ramp: 0.869 / 0.936, R^2 0.992 / 0.936, far-field w
+0.048 on uHu (7 percent high on c: the ramp is nearly a box). Tukey
0.25: 0.805 / 0.990, R^2 0.9999 / 0.9997; Tukey 0.5: 0.809 / 0.996,
R^2 0.9996 / 0.9997; Hann (Tukey 1.0, 52 usable columns): 0.814 / 1.000,
R^2 0.9999 / 0.9999. Any Tukey alpha of 0.25 or more puts the crop
within 1 percent of the periodic reference; the price is columns.
Changelog:
v1.1  'trim' entry in the window sweep: the crop cut to an integer
      number of periods with no window (knee_sweep_common v1.2), the
      alternative to any taper for the archived crops.
v1.0  first version.
Python 3.8 compatible; frozen layer imported, never edited.
"""
import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.signal.windows import tukey

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import knee_sweep_common as ksc  # noqa: E402


# ---------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------

def window_1d(nx, kind, param):
    """Return the window and the slice of columns where it is flat."""
    if kind == "box":
        return np.ones(nx), slice(24, nx - 24)
    if kind == "tanh":
        # old-runner ramp: 0.5(1+tanh(s(x-a))) * 0.5(1-tanh(s(x-b))) with
        # margins param of the box and tanhscale 120 in box units
        xs = np.linspace(0.0, 1.0, nx)
        s = 120.0
        w = 0.25 * (1 + np.tanh(s * (xs - param))) * (1 - np.tanh(s * (xs - (1 - param))))
        w /= w.max()
        flat = int(np.ceil(2 * param * nx))
        return w, slice(flat, nx - flat)
    if kind == "tukey":
        w = tukey(nx, param)
        flat = np.where(w >= 0.999)[0]
        if flat.size < 32:                 # Hann-like: no flat top, take w >= 0.9
            flat = np.where(w >= 0.9)[0]
        return w, slice(int(flat[0]), int(flat[-1]) + 1)
    raise ValueError(kind)


WINDOWS = [("box", 0.0), ("tanh", 0.025), ("tukey", 0.1), ("tukey", 0.25),
           ("tukey", 0.5), ("tukey", 0.75), ("tukey", 1.0), ("trim", 0.1)]
# ("trim", max_jump): no window at all; the crop is cut to an integer
# number of stripe periods (knee_sweep_common.trim_to_periods) so the
# periodic seam closes within max_jump radians, and the record's
# 24-column margin is the flat region.


def wlabel(kind, param):
    return {"box": "box", "tanh": "tanh ramp", "trim": "trim (integer periods)"}.get(
        kind, "Tukey %.2f" % param)


# ---------------------------------------------------------------------
# extraction under a window (mirrors knee_sweep_common.extract_k with an
# arbitrary window instead of the Tukey taper)
# ---------------------------------------------------------------------

def extract_windowed(route, u, x, y, jc, win, y_frac=0.06, ramp=None):
    M = ksc.setup_method_layer()
    tc, us, fdv = M["tc"], M["us"], M["fdv"]
    r = ksc.ROUTES[route]
    Lx, Ly = ksc.box_lengths(x, y)
    ux = u * win[None, :]
    s = np.sign(y[:, None] - y[jc]); s[s == 0] = 1.0
    if r["kind"] == "demod":
        dx, dy = x[1] - x[0], y[1] - y[0]
        ua = hilbert(ux, axis=1); den = np.abs(ua) ** 2 + 1e-12
        return (np.imag(np.conj(ua) * np.gradient(ua, dx, axis=1)) / den,
                np.imag(np.conj(ua) * np.gradient(ua, dy, axis=0)) / den)
    uy = ux * tc.edge_taper(x, y, frac=y_frac)
    if r["kind"] == "uhu":
        sm = us.make_gaussian_smoother(r["sigma"], Lx, Ly, fdv.macro_iso)
        ops = us.compute_uhu_ops_smoother(uy, x, y, sm, fdv.spectral_deriv,
                                          np.ones_like(u) if ramp is None else ramp)
        return np.abs(ops["k1_orig"]), np.abs(ops["k2_orig"]) * s
    if r["kind"] == "riesz":
        rk = tc.riesz_k_field(uy, x, y, Lx, Ly, sigma_dir=r["sigma_dir"], sigma_avg=r["sigma_avg"])
        return np.abs(rk["k1"]), np.abs(rk["k2"]) * s
    raise ValueError(route)


def measure(route, k1, k2, x, y, jc, mu, cols, band_w=3.0):
    """Profile regression on the given columns; far-field w offset; k1
    ripple on a far-field row."""
    dy = y[1] - y[0]
    yc = y - y[jc]
    k2b = k2[:, cols].mean(1)
    wb = (k1[:, cols] ** 2 + k2[:, cols] ** 2).mean(1) - 1.0
    dk2 = np.gradient(k2b, dy)
    seam = ksc.route_seam(route)
    y_clear = ksc.y_clear_distance(y, 0.06, route)
    bh = min(band_w / mu, y_clear)
    band = (np.abs(yc) < bh) & (np.abs(yc) > seam); band[:2] = band[-2:] = False
    ff = (np.abs(yc) >= bh) & (np.abs(yc) < y_clear); ff[:2] = ff[-2:] = False
    if ff.sum() < 8:
        ff = band & (np.abs(yc) > 0.75 * bh)
    w_ff = float(wb[ff].mean())
    c, b, r2 = ksc.linear_fit(wb[band] - w_ff, dk2[band])
    jff = int(np.argmin(np.abs(yc - 0.9 * y_clear)))
    ripple = float(k1[jff, cols].std())
    return dict(c=-c, b=b, r2=r2, w_ff=w_ff, k1_ripple=ripple,
                n_cols=int(cols.stop - cols.start), wb=wb, dk2=dk2, band=band, yc=yc)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main(cfg):
    ksc.setup_method_layer(cfg.get("method_dirs", ()))
    ksc.use_house_style()
    M = ksc.setup_method_layer(); tc, us, fdv = M["tc"], M["us"], M["fdv"]
    out = Path(cfg["out_root"]) / ("forensics_mu%g_R%g_beta%g" % (cfg["mu"], cfg["R"], cfg["beta"]))
    out.mkdir(parents=True, exist_ok=True)
    ksc.save_config(out, cfg)
    mu, R, beta = cfg["mu"], cfg["R"], cfg["beta"]
    rows = []

    # ---- grids: the crop grid (from the crop file when given) ----
    if cfg["crop_file"]:
        xc, ycrop, u_data, _ = ksc.load_frame(cfg["crop_file"], -1)
        info = ksc.read_meta(cfg["crop_file"]); jc = info["jc"]
        mu_d, R_d = info["mu"], info["R"]
    else:
        n = 256; dxc = 0.1255
        xc = np.arange(n) * dxc; xc -= xc.mean(); ycrop = (np.arange(n) - n // 2) * dxc
        u_data = None; jc = n // 2; mu_d, R_d = None, None
    k1ff = np.sqrt(1 - mu ** 2); lam = 2 * np.pi / k1ff
    periods_crop = (xc[-1] - xc[0] + (xc[1] - xc[0])) / lam
    print("crop grid: %d columns, %.2f stripe periods in x" % (len(xc), periods_crop))

    # ---- 1. periodic reference: 512^2 box, integer periods, old ramp ----
    N = 512; nper = 9; Lx = nper * lam; dx = Lx / N
    xp = np.arange(N) * dx - Lx / 2; yp = np.arange(N) * dx - N * dx / 2
    Xp, Yp = np.meshgrid(xp, yp)
    up = ksc.synth_knee_on_grid(xp, yp, mu, R, beta, N // 2)
    ramp = 0.25 * ((1 + np.tanh((Xp + 0.85 * Lx / 2) / 3)) * (1 - np.tanh((Xp - 0.85 * Lx / 2) / 3)))
    ramp *= 0.25 * ((1 + np.tanh((Yp + 0.85 * N * dx / 2) / 3)) * (1 - np.tanh((Yp - 0.85 * N * dx / 2) / 3)))
    ref = {}
    for route in cfg["routes"]:
        if route == "uhu":
            sm = us.make_gaussian_smoother(ksc.ROUTES["uhu"]["sigma"], Lx, N * dx, fdv.macro_iso)
            ops = us.compute_uhu_ops_smoother(up * ramp, xp, yp, sm, fdv.spectral_deriv, ramp)
            s = np.sign(yp[:, None] - yp[N // 2]); s[s == 0] = 1
            k1, k2 = np.abs(ops["k1_orig"]), np.abs(ops["k2_orig"]) * s
        else:
            k1, k2 = extract_windowed(route, up * ramp, xp, yp, N // 2, np.ones(N))
        rn = (ramp - ramp.min()) / (ramp.max() - ramp.min())
        inner = np.where(rn[N // 2] >= 0.999)[0]; cols = slice(inner[0] + 48, inner[-1] - 48)
        m = measure(route, k1, k2, xp, yp, N // 2, mu, cols)
        ref[route] = m
        rows.append(dict(field="planted", layout="periodic_512_ramp", window="ramp", param=0.85,
                         route=route, c=m["c"], r2=m["r2"], b=m["b"], w_ff=m["w_ff"],
                         k1_ripple=m["k1_ripple"], n_cols=m["n_cols"], periods=nper))
        print("periodic reference %-6s: c=%.4f R2=%.4f w_ff=%+.4f ripple=%.4f" % (route, m["c"], m["r2"], m["w_ff"], m["k1_ripple"]))

    # ---- 2. crop under the window sweep: planted knee, then data ----
    fields = [("planted", ksc.synth_knee_on_grid(xc, ycrop, mu, R, beta, jc), mu)]
    if u_data is not None:
        fields.append(("data", u_data, mu_d))
    keep = {}
    for fname, u, mu_use in fields:
        for kind, param in WINDOWS:
            uu, xx, nper_win = u, xc, periods_crop
            if kind == "trim":
                uu, xx, info = ksc.trim_to_periods(u, xc, ksc.carrier_k1(mu_use), param)
                win, cols = np.ones(len(xx)), slice(24, len(xx) - 24)
                nper_win = info["n_periods"] + info["jump_rad"] / (2 * np.pi)
                print("trim: %d of %d columns, %d periods, seam jump %.3f rad"
                      % (info["n_cols"], info["n_cols_in"], info["n_periods"], info["jump_rad"]))
            else:
                win, cols = window_1d(len(xc), kind, param)
            for route in cfg["routes"]:
                k1, k2 = extract_windowed(route, uu, xx, ycrop, jc, win)
                m = measure(route, k1, k2, xx, ycrop, jc, mu_use, cols)
                rows.append(dict(field=fname, layout="crop_%d" % len(xx), window=kind, param=param,
                                 route=route, c=m["c"], r2=m["r2"], b=m["b"], w_ff=m["w_ff"],
                                 k1_ripple=m["k1_ripple"], n_cols=m["n_cols"], periods=nper_win))
                print("%-7s %-10s %-6s: c=%.4f R2=%.4f w_ff=%+.4f ripple=%.4f cols=%d"
                      % (fname, wlabel(kind, param), route, m["c"], m["r2"], m["w_ff"], m["k1_ripple"], m["n_cols"]))
                if fname == "planted" and (kind, param) in (("box", 0.0), ("tanh", 0.025), ("tukey", 0.5)):
                    keep[(kind, param, route)] = (k1, k2, m)

    with open(out / "forensics.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

    # ---- figures ----
    labels = [wlabel(k, p) for k, p in WINDOWS]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.4), constrained_layout=True)
    for route in cfg["routes"]:
        for fname, ls in (("planted", "-"), ("data", "--")):
            sel = [r for r in rows if r["route"] == route and r["field"] == fname and r["layout"].startswith("crop")]
            if not sel:
                continue
            xs = range(len(sel))
            axs[0].plot(xs, [r["c"] for r in sel], ls, marker="o", color=ksc.ROUTE_COLORS[route], label="%s %s" % (route, fname))
            axs[1].plot(xs, [r["r2"] for r in sel], ls, marker="o", color=ksc.ROUTE_COLORS[route], label="%s %s" % (route, fname))
            axs[2].plot(xs, [r["w_ff"] for r in sel], ls, marker="o", color=ksc.ROUTE_COLORS[route], label="%s %s" % (route, fname))
            axs[3].plot(xs, [r["k1_ripple"] for r in sel], ls, marker="o", color=ksc.ROUTE_COLORS[route], label="%s %s" % (route, fname))
        if route in ref:
            axs[0].axhline(ref[route]["c"], color=ksc.ROUTE_COLORS[route], lw=0.8, ls=":", label="%s periodic reference" % route)
            axs[1].axhline(ref[route]["r2"], color=ksc.ROUTE_COLORS[route], lw=0.8, ls=":")
    for ax, ttl in zip(axs, ("$c$ (planted $\\beta=%g$)" % beta, "$R^2$ of $\\nabla\\!\\cdot\\!\\mathbf{k}=b-c\\,w$",
                             "far-field $w$ offset", "$k_1$ ripple on a far-field row")):
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax.set_title(ttl, fontsize=9)
    axs[1].set_ylim(-0.05, 1.02); axs[0].legend(fontsize=6)
    fig.suptitle("crop of %.2f stripe periods: extraction against the x window (dotted: same knee on a periodic box)" % periods_crop, fontsize=9)
    fig.savefig(out / "fig_taper_sweep.png"); plt.close(fig)

    if cfg["routes"]:
        route = cfg["routes"][0]
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
        ext = [xc[0], xc[-1], ycrop[0] - ycrop[jc], ycrop[-1] - ycrop[jc]]
        for ax, (kind, param) in zip(axs, (("box", 0.0), ("tanh", 0.025), ("tukey", 0.5))):
            k1, k2, m = keep[(kind, param, route)]
            w2 = k1 ** 2 + k2 ** 2 - 1.0
            im = ax.imshow(w2, extent=ext, cmap="seismic", vmin=-0.3, vmax=0.3, aspect="auto", origin="lower")
            fig.colorbar(im, ax=ax, shrink=0.85)
            ax.set_title("%s, %s: far-field $w$ %+.3f, $R^2$ %.3f" % (route, wlabel(kind, param), m["w_ff"], m["r2"]), fontsize=9)
            ax.set_xlabel("$x$")
        axs[0].set_ylabel("$y$")
        fig.suptitle(r"$w=|\mathbf{k}|^2-1$ on the crop, planted knee: the window is the difference", fontsize=9)
        fig.savefig(out / "fig_w_maps.png"); plt.close(fig)

        fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
        for (kind, param) in (("box", 0.0), ("tanh", 0.025), ("tukey", 0.5)):
            k1, k2, m = keep[(kind, param, route)]
            axs[0].plot(m["yc"], m["wb"], label=wlabel(kind, param))
            axs[1].plot(m["yc"], m["dk2"], label=wlabel(kind, param))
        axs[0].plot(m["yc"], -mu ** 2 / np.cosh(beta * mu * m["yc"]) ** 2, "k:", lw=1, label="planted $-\\mu^2{\\rm sech}^2$")
        axs[1].plot(m["yc"], beta * mu ** 2 / np.cosh(beta * mu * m["yc"]) ** 2, "k:", lw=1, label="planted")
        axs[0].set_title(r"$\bar w(y)$, %s" % route, fontsize=9); axs[1].set_title(r"$d\bar k_2/dy$, %s" % route, fontsize=9)
        for ax in axs:
            ax.set_xlabel("$y$"); ax.legend(fontsize=7)
        fig.savefig(out / "fig_profiles.png"); plt.close(fig)
    print("wrote", out)
    return rows


class _Args:
    crop_file = str(
        _HERE / "data/sh_pgb_zigzag/mu_sweeps_shallow_v2_Rp5_Nx512_tmax500_hp5_np18_Ny5/raw/Rp5_sh_pgb_zigzag_mu0.375_T500_N2_nx512_Ny256_lower.npz")
    periodic_file = ""      # informational; the periodic reference is planted on its own box
    mu = 0.375
    R = 0.5
    beta = 1.0
    routes = ["uhu", "demod"]
    out_root = str(_HERE / "results" / "crop_windowing_forensics")
    method_dirs = []


if __name__ == "__main__":
    d = _Args
    p = argparse.ArgumentParser(description="windowing forensics for FFT-based extraction on crops")
    p.add_argument("--crop-file", default=d.crop_file)
    p.add_argument("--periodic-file", default=d.periodic_file)
    p.add_argument("--mu", type=float, default=d.mu)
    p.add_argument("--R", type=float, default=d.R)
    p.add_argument("--beta", type=float, default=d.beta)
    p.add_argument("--routes", nargs="+", default=d.routes, choices=list(ksc.ROUTES))
    p.add_argument("--out-root", default=d.out_root)
    p.add_argument("--method-dirs", nargs="*", default=d.method_dirs)
    a = p.parse_args()
    main(dict(crop_file=a.crop_file if Path(a.crop_file).exists() else "", periodic_file=a.periodic_file,
              mu=a.mu, R=a.R, beta=a.beta, routes=a.routes, out_root=a.out_root, method_dirs=a.method_dirs))
