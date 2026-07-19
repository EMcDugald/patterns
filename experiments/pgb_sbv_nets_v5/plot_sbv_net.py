# experiments/pgb_sbv_nets/plot_sbv_net.py
"""
Make diagnostic plots for a completed run directory produced by
train_sbv_net.py (results/net_runs/<run_name>/).

Usage
-----
python plot_sbv_net.py --run_dir results/net_runs/<run_name>
python plot_sbv_net.py --run_dir ... --frame 1     # if multiple fields_*.npz

Figures written to <run_dir>/figures/:
    recon_frame###.png   : phase, cos(theta), correction, energies, split
    jumpset_frame###.png : mu_s with sin(theta)=0 contours; concentration
                           check (the main validation: mu_s ~ 0 in smooth
                           regions, concentrated in defected ones)
    history.png          : loss terms and calibration coefficients vs iter
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _crop_to_mask(fields, pad=12):
    """Crop all fields to the bounding box of the valid mask (plus pad
    cells), so figures focus on the supervised region — matters once
    --x_window shrinks the mask to a sub-rectangle. Profiles inherit the
    crop automatically."""
    mask = np.asarray(fields["mask"]).astype(bool)
    if not mask.any():
        return dict(fields)
    ys, xs = np.where(mask)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, mask.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, mask.shape[1])
    out = {}
    for k in fields:
        v = np.asarray(fields[k])
        if v.ndim == 2 and v.shape == mask.shape:
            out[k] = v[y0:y1, x0:x1]
        elif k == "x":
            out[k] = v[x0:x1]
        elif k == "y":
            out[k] = v[y0:y1]
        else:
            out[k] = v
    return out


def _imshow(ax, arr, extent, cmap="viridis", title=None, mask=None,
            sym=False, vmin=None, vmax=None):
    arr = np.asarray(arr, dtype=float)
    if mask is not None:
        arr = np.ma.masked_where(~mask.astype(bool), arr)
    if sym and vmin is None and vmax is None:
        vmax = np.nanpercentile(np.abs(np.asarray(arr).ravel()), 99.5)
        vmin = -vmax
    im = ax.imshow(arr, extent=extent, origin="lower", cmap=cmap,
                   vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return im


def plot_recon(fields, out_path, title=None):
    x, y = fields["x"], fields["y"]
    extent = [x[0], x[-1], y[0], y[-1]]
    mask = fields["mask"].astype(bool)
    theta = fields["theta"]
    resid = fields["e_model"] - fields["e_macro"]

    fig, axs = plt.subplots(4, 3, figsize=(15, 14))
    axs = axs.ravel()

    im = _imshow(axs[0], fields["u"], extent, cmap="copper", title="u")
    plt.colorbar(im, ax=axs[0], shrink=0.8)
    im = _imshow(axs[1], theta, extent, cmap="twilight",
                 title="theta (reconstructed)", mask=mask)
    plt.colorbar(im, ax=axs[1], shrink=0.8)
    im = _imshow(axs[2], np.cos(theta) * mask, extent, cmap="gray",
                 title="cos(theta) * mask")
    plt.colorbar(im, ax=axs[2], shrink=0.8)

    dt = fields["dtheta"]
    im = _imshow(axs[3], dt, extent, cmap="RdBu_r",
                 title="correction dtheta", mask=mask, sym=True)
    plt.colorbar(im, ax=axs[3], shrink=0.8)
    # shared color limits so target and reconstruction are comparable
    em_vals = fields["e_macro"][mask]
    vlo, vhi = np.percentile(em_vals, [0.5, 99.5])
    im = _imshow(axs[4], fields["e_macro"], extent, cmap="inferno",
                 title="e_macro (target)", mask=mask, vmin=vlo, vmax=vhi)
    plt.colorbar(im, ax=axs[4], shrink=0.8)
    im = _imshow(axs[5], fields["e_model"], extent, cmap="inferno",
                 title="e_model (reconstruction)", mask=mask,
                 vmin=vlo, vmax=vhi)
    plt.colorbar(im, ax=axs[5], shrink=0.8)

    im = _imshow(axs[6], resid, extent, cmap="RdBu_r",
                 title="e_model - e_macro", mask=mask, sym=True)
    plt.colorbar(im, ax=axs[6], shrink=0.8)
    im = _imshow(axs[7], fields["q"], extent, cmap="RdBu_r",
                 title="q = G_s * Lap(theta)", mask=mask, sym=True)
    plt.colorbar(im, ax=axs[7], shrink=0.8)
    im = _imshow(axs[8], fields["rho_s"], extent, cmap="RdBu_r",
                 title="rho_s (a.c. part)", mask=mask, sym=True)
    plt.colorbar(im, ax=axs[8], shrink=0.8)

    im = _imshow(axs[9], fields["mu_s"], extent, cmap="RdBu_r",
                 title="mu_s (singular part)", mask=mask, sym=True)
    plt.colorbar(im, ax=axs[9], shrink=0.8)
    im = _imshow(axs[10], fields["well"], extent, cmap="magma",
                 title="(|grad theta|^2 - 1)^2", mask=mask)
    plt.colorbar(im, ax=axs[10], shrink=0.8)
    im = _imshow(axs[11], fields["abs_sin"], extent, cmap="gray",
                 title="|sin(theta)|", mask=mask)
    plt.colorbar(im, ax=axs[11], shrink=0.8)

    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_jumpset(fields, out_path, title=None):
    """|mu_s| with sin(theta)=0 contours overlaid: the emergence of
    Delta_k inside Gamma_D is the central check, and mu_s should vanish in
    the smooth regions of the same frame under the same calibration.
    Panels: (0) |mu_s| map; (1) y-marginal (spike at the GB row, raised
    baseline = any vertical seam); (2) x-profile: |mu_s| integrated in y
    over a narrow band around the GB row, i.e. the recovered jump
    amplitude [[grad theta]].nu as a function of x (diamond eikonal
    theory: 2*sqrt(1-mu^2), constant along the GB up to beading)."""
    x, y = fields["x"], fields["y"]
    extent = [x[0], x[-1], y[0], y[-1]]
    X, Y = np.meshgrid(x, y)
    mask = fields["mask"].astype(bool)
    mu = np.abs(fields["mu_s"])
    mu_m = np.where(mask, mu, 0.0)
    sin_t = np.sin(fields["theta"])
    dy = float(y[1] - y[0])

    fig, axs = plt.subplots(1, 3, figsize=(19, 5))

    im = _imshow(axs[0], mu, extent, cmap="magma",
                 title="|mu_s| with sin(theta)=0 contours", mask=mask)
    plt.colorbar(im, ax=axs[0], shrink=0.8)
    axs[0].contour(X, Y, np.where(mask, sin_t, np.nan), levels=[0.0],
                   colors="cyan", linewidths=0.4)

    # transverse concentration profile: integrate |mu_s| over x vs y
    prof_y = mu_m.sum(axis=1)
    axs[1].plot(prof_y, y, lw=1.0)
    axs[1].set_xlabel("sum_x |mu_s|")
    axs[1].set_ylabel("y")
    axs[1].set_title("y-marginal (spike = GB row;\n"
                     "raised baseline = vertical seam)", fontsize=9)

    # x-profile: band-integrated jump amplitude along the GB
    j0 = int(np.argmax(prof_y))            # GB row
    hw = 8                                  # band half-width (rows)
    lo, hi = max(j0 - hw, 0), min(j0 + hw + 1, mu.shape[0])
    amp_x = mu_m[lo:hi, :].sum(axis=0) * dy      # ~ [[grad theta]].nu (x)
    marg_x = mu_m.sum(axis=0) * dy               # includes vertical seam
    axs[2].plot(x, amp_x, lw=1.0,
                label=f"band int., rows {lo}:{hi} (y~{y[j0]:.1f})")
    axs[2].plot(x, marg_x, lw=0.7, ls="--", alpha=0.6,
                label="full y-marginal")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("int |mu_s| dy")
    axs[2].set_title("jump amplitude along GB\n"
                     "(eikonal theory: 2*sqrt(1-mu^2))", fontsize=9)
    axs[2].legend(fontsize=7)

    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_jumpsurf(fields, out_path, title=None, max_dim=384):
    """Surface plot of the signed singular density mu_s (masked to 0
    outside). Downsampled so the mesh stays renderable."""
    x, y = fields["x"], fields["y"]
    mask = fields["mask"].astype(bool)
    Z = np.where(mask, fields["mu_s"], 0.0)

    sy = max(1, Z.shape[0] // max_dim)
    sx = max(1, Z.shape[1] // max_dim)
    Xs, Ys = np.meshgrid(x[::sx], y[::sy])
    Zs = Z[::sy, ::sx]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection="3d")
    vmax = np.percentile(np.abs(Zs), 99.9) or 1.0
    surf = ax.plot_surface(Xs, Ys, Zs, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax,
                           rstride=1, cstride=1,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, ax=ax, shrink=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("mu_s")
    ax.set_title(title or "mu_s (signed singular part)", fontsize=9)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_history(history, out_path):
    it = np.arange(1, len(history) + 1)
    loss_keys = [k for k in ("total", "energy", "gauge", "small", "smooth")
                 if k in history[0]]
    cal_keys = [k for k in ("c0", "c_bend", "c_well", "c_sing", "kappa")
                if k in history[0]]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for k in loss_keys:
        axs[0].semilogy(it, [max(h[k], 1e-16) for h in history], label=k)
    axs[0].set_xlabel("iteration")
    axs[0].set_title("loss terms")
    axs[0].legend(fontsize=8)

    for k in cal_keys:
        axs[1].plot(it, [h[k] for h in history], label=k)
    axs[1].set_xlabel("iteration")
    axs[1].set_title("calibration coefficients")
    axs[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_midline(fields, out_path, title=None):
    """theta0 and reconstructed theta along the y ~ 0 midline (the GB row
    for the diamond), restricted to valid-mask columns."""
    x, y = fields["x"], fields["y"]
    j = int(np.argmin(np.abs(y)))          # row closest to y = 0
    mask_row = fields["mask"].astype(bool)[j]
    xm = np.where(mask_row, x, np.nan)

    theta = fields["theta"][j]
    theta0 = fields["theta0"][j]
    have0 = np.any(np.isfinite(theta0))

    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                            gridspec_kw={"height_ratios": [2, 1]})
    axs[0].plot(xm, theta, lw=1.2, label="theta (reconstructed)")
    if have0:
        axs[0].plot(xm, theta0, lw=1.0, ls="--", label="theta0 (base)")
    axs[0].set_ylabel("phase")
    axs[0].set_title(f"midline y = {y[j]:.2f}", fontsize=9)
    axs[0].legend(fontsize=8)

    if have0:
        axs[1].plot(xm, theta - theta0, lw=1.0, color="tab:red")
        axs[1].set_ylabel("theta - theta0")
    axs[1].set_xlabel("x")

    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--frame", type=int, default=None,
                   help="Frame number of fields_frame###.npz; default: all.")
    ns = p.parse_args(argv)

    run_dir = Path(ns.run_dir)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    tag = (f"{cfg['repr_mode']}/{cfg['energy_mode']}/init-"
           f"{cfg['init_mode']} | s={cfg['s_test']:.3g} "
           f"sigma_f={cfg['macro_sigma']:.3g} kappa0={cfg['kappa_init']} "
           f"delta={cfg['delta']}")

    field_files = sorted(run_dir.glob("fields_frame*.npz"))
    if ns.frame is not None:
        field_files = [run_dir / f"fields_frame{ns.frame:03d}.npz"]
    if not field_files:
        raise SystemExit(f"No fields_frame*.npz in {run_dir}")

    for ff in field_files:
        raw = np.load(ff)
        fields = _crop_to_mask({k: raw[k] for k in raw.files})
        frame_tag = ff.stem.replace("fields_", "")
        plot_recon(fields, fig_dir / f"recon_{frame_tag}.png",
                   title=f"{run_dir.name} | {tag}")
        plot_jumpset(fields, fig_dir / f"jumpset_{frame_tag}.png",
                     title=f"{run_dir.name} | {tag}")
        plot_jumpsurf(fields, fig_dir / f"jumpsurf_{frame_tag}.png",
                      title=f"{run_dir.name}")
        plot_midline(fields, fig_dir / f"midline_{frame_tag}.png",
                     title=f"{run_dir.name}")
        print(f"  plotted {ff.name}")

    hist_path = run_dir / "history.json"
    if hist_path.exists():
        with open(hist_path) as f:
            history = json.load(f)
        plot_history(history, fig_dir / "history.png")
        print("  plotted history.png")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # No CLI args: use in-script settings (edit freely).
        class _Args:
            run_dir = str(Path(__file__).resolve().parent / "results" /
                          "net_runs" / "sh_pgb_diamond_mu0.450_T50_N2_Ny1536_Ly188.496_margin0.45_knee_uhu_sigma1.571_dm0.400_ts0.40_gsnone_phase_diamond_ns192_ds0.125_bdinner_ksym_prmsaved_sif0.200_srm0.990_rst0.000__field_sbv_init-data_f002_it3000_xw0.56-0.94_fd")
            frame = None

        a = _Args()
        argv = ["--run_dir", a.run_dir]
        if a.frame is not None:
            argv += ["--frame", str(a.frame)]
        main(argv)
    else:
        main()
