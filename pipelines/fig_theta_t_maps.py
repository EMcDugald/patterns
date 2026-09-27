# pipelines/fig_theta_t_maps.py
"""
Figure for Section 3C: maps on the band of the measured phase rate
theta_t, the fitted law theta_t = a1 w div k + a2 k.grad w + c_J J, the
residual, and the Gaussian curvature J, at one frame of one file, with
the detected cores marked. The coefficients are TAKEN FROM THE TABLE of
record (sysid_scan runs.csv / Table tab:phase_law_windows) and passed on
the command line; nothing is fitted here.

The fields are the mg estimator's (mg_zigzag_theta_t.estimate: theta,
k = Im(conj z grad z)/|z|^2, A = |z|), theta_t = Im(conj z z_t)/|z|^2
by centred differences over frames j-1, j+1, and the derivatives of k
are spectral on the tapered field (the same route as mg_zigzag_theta_t
and rcn_balance_ledger). With --smooth > 0 (in spacings L) every map is
Gaussian-smoothed at that width before display, which imitates the
weak-form kernel of the regression (radius 1.5 L bump, whose rms width
is about 0.5 L); with --smooth 0 the maps are pointwise, where the
fourth-order and the near-core pixels are noisy, as the text says.

Imports Ed's tree (pipelines/mg_zigzag_theta_t.py, src/op_extract,
src/utils) the way ruler_w.py does; place this file in pipelines/.

Panels: theta_t, law, residual (shared colour scale, symmetric at the
--clip quantile of |theta_t|), J (own scale). Cores from find_cores at
the frame (amplitude below --core_frac of the window median) as circles
of radius --core_excl L, the discs the regression excludes.

Run:
  python fig_theta_t_maps.py --npz <T25 mu0.90> --t 4.0 --coef 0.80,0.85,-1.63 --smooth 0.5
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT / "src", _ROOT / "src" / "op_extract", _ROOT / "src" / "utils", _HERE):
    sys.path.insert(0, str(p))
from figcommon import fnum, outdir_for, write_config, write_table, save, cli, plt  # noqa: E402
try:
    from mg_zigzag_theta_t import load_npz, estimate, find_cores            # noqa: E402
    from mg_estimator_validation import fdv                                 # noqa: E402
    from op_extract import t1_common as t1c                                 # noqa: E402
except ImportError as e:                                                    # pragma: no cover
    raise SystemExit("fig_theta_t_maps needs Ed's tree on the path (pipelines/mg_zigzag_theta_t.py, "
                     "src/op_extract, src/utils): %s" % e)


def run(args):
    D = load_npz(args.npz)
    u, x, y, t, md = D["u"], D["x"], D["y"], D["t"], D["md"]
    ny, nx, nt = u.shape
    dx, dy = x[1] - x[0], y[1] - y[0]
    Lx, Ly = nx * dx, ny * dy
    dt = float(np.median(np.diff(t)))
    mu = float(md.get("mu", args.mu))
    Lsp = np.pi / np.sqrt(1 - mu ** 2)
    j = int(np.argmin(np.abs(t - args.t))) if args.frame < 0 else int(args.frame)
    j = min(max(j, 1), nt - 2)
    X, Y = np.meshgrid(x, y)
    taper = t1c.edge_taper(x, y, frac=args.taper_frac)
    win = (np.abs(X - x.mean()) < args.win_frac * Lx) & (np.abs(Y - y.mean()) < args.win_frac * Ly)
    y_row = y[int(md.get("gb_row_in_crop", ny // 2))]
    Dsp = lambda a, wch: fdv.spectral_deriv(a * taper, Lx, Ly, wch)

    ests, n_prev = [], None
    for i in (j - 1, j, j + 1):
        e = estimate(u[..., i], Lx, Ly, args.sigma_dir, n_prev, taper)
        n_prev = e["n"]
        ests.append(e)
    e0, e1, e2 = ests
    s = np.sign(e1["k1"].mean()) or 1.0                    # one scalar sign per frame
    z_t = (e2["z"] - e0["z"]) / (2 * dt)
    th_t = np.imag(np.conj(e1["z"]) * z_t) / np.maximum(np.abs(e1["z"]) ** 2, 1e-14)
    k1, k2 = e1["k1"] * s, e1["k2"] * s
    w = k1 ** 2 + k2 ** 2 - 1.0
    k1x, k1y, k2x, k2y = Dsp(k1, "x"), Dsp(k1, "y"), Dsp(k2, "x"), Dsp(k2, "y")
    divk = k1x + k2y
    kgw = k1 * Dsp(w, "x") + k2 * Dsp(w, "y")
    J = k1x * k2y - k1y * k2x
    a1, a2, cJ = [float(v) for v in args.coef.split(",")]
    law = a1 * w * divk + a2 * kgw + cJ * J
    res = th_t - law
    xc, yc = find_cores(e1["A"], x, y, win, args.core_frac, merge_dist=0.5 * Lsp)

    sm = args.smooth * Lsp / dx
    S = (lambda F: gaussian_filter(F * taper, sm)) if sm > 0 else (lambda F: F)
    band = (np.abs(Y - y_row) < args.band_to * Lsp) & win
    iy = np.where(band.any(axis=1))[0]
    ix = np.where(band.any(axis=0))[0]
    crop = lambda F: F[iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1]
    ext = [x[ix[0]], x[ix[-1]], y[iy[0]], y[iy[-1]]]
    maps = [("$\\theta_t$ (measured)", S(th_t)), ("law $a_1 w\\nabla\\cdot k + a_2 k\\cdot\\nabla w + c_J J$", S(law)),
            ("residual", S(res)), ("$J$", S(J))]
    v = float(np.nanquantile(np.abs(crop(maps[0][1])), args.clip))
    vJ = float(np.nanquantile(np.abs(crop(maps[3][1])), args.clip))

    out = outdir_for(__file__, args.outdir)
    write_config(out, args)
    fig, ax = plt.subplots(len(maps), 1, figsize=(7.0, 1.0 + 1.6 * len(maps)), sharex=True)
    for a, (name, F) in zip(ax, maps):
        lim = vJ if name == "$J$" else v
        im = a.imshow(crop(F), origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for xx, yy in zip(xc, yc):
            a.add_patch(plt.Circle((xx, yy), args.core_excl * Lsp, fill=False, ec="k", lw=0.6))
        a.set_title(name, fontsize=8)
        a.set_ylabel("$y$")
        fig.colorbar(im, ax=a, fraction=0.03, pad=0.01)
    ax[-1].set_xlabel("$x$")
    fig.suptitle("%s, t = %.3f (frame %d), mu %.2f; coefficients (%.2f, %.2f, %.2f); smooth %.2g L; %d cores"
                 % (Path(args.npz).stem, t[j], j, mu, a1, a2, cJ, args.smooth, len(xc)), fontsize=8)
    fig.tight_layout()
    save(fig, out)
    rms = lambda F: float(np.sqrt(np.nanmean(crop(F)[crop(band)] ** 2)))
    tab = [dict(npz=args.npz, t=float(t[j]), frame=j, mu=mu, a1=a1, a2=a2, cJ=cJ, smooth_L=args.smooth,
                n_cores=len(xc), rms_theta_t=rms(maps[0][1]), rms_law=rms(maps[1][1]), rms_residual=rms(maps[2][1]),
                rms_J=rms(maps[3][1]), cores_json=json.dumps([[float(a), float(b)] for a, b in zip(xc, yc)]))]
    write_table(out, tab)
    np.savez_compressed(out / "maps.npz", x=x[ix[0]:ix[-1] + 1], y=y[iy[0]:iy[-1] + 1],
                        theta_t=crop(maps[0][1]), law=crop(maps[1][1]), residual=crop(maps[2][1]), J=crop(maps[3][1]))
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    npz = ""                 # a T25 (dt 0.2) zigzag file
    t = 4.0                  # time of the frame (nearest frame is used) unless frame >= 0
    frame = -1
    mu = 0.9                 # fallback if the npz metadata carries no mu
    coef = "0.80,0.85,-1.63"  # a1, a2, c_J of record for the window (Table tab:phase_law_windows)
    sigma_dir = 0.4          # mg_zigzag_theta_t defaults follow
    taper_frac = 0.08
    win_frac = 0.36
    core_frac = 0.70
    core_excl = 0.5          # spacings; the disc the regression excludes, drawn only
    band_to = 3.0            # spacings; the band shown
    smooth = 0.5             # spacings; 0 = pointwise maps
    clip = 0.98              # quantile of |theta_t| for the shared colour limit
    outdir = "results"


if __name__ == "__main__":
    sys.exit(cli(_Args, __doc__, run))
