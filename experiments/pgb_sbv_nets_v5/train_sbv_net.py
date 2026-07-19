# experiments/pgb_sbv_nets/train_sbv_net.py
"""
Train phase-estimation models against the filtered macro energy.

Usage examples
--------------
# Full SBV energy, neural field, corrector on theta0 (theta_init_from_data),
# last frame:
python train_sbv_net.py \
    --probe_dir results/sbv_phase_probe/<stem> \
    --repr field --energy sbv --init data --frame -1 --coarsen 2

# Bulk-only baseline (no singular measure):
python train_sbv_net.py --probe_dir ... --energy bulk

# Grid-based representation for comparison:
python train_sbv_net.py --probe_dir ... --repr grid --lam_smooth 1e-2

# All frames, fitted last-to-first with warm start:
python train_sbv_net.py --probe_dir ... --all_frames

Outputs go to results/net_runs/<run_name>/ :
    config.json, history.json, checkpoint.pt, fields_frame###.npz
Plots are made separately by plot_sbv_net.py.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from sbv_net_core import (SBVNetConfig, SBVModel, SpectralOps,
                          compute_losses, load_probe, spectral_grad_np,
                          fd_grad_np, make_x_grid, save_config)


def build_run_dir(base, name):
    run_dir = Path(base) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def train_one_frame(cfg: SBVNetConfig, probe_dir, frame, out_base,
                    warm_model=None, log_every=10):
    d = load_probe(probe_dir, frame, coarsen=cfg.coarsen,
                   mask_erode=cfg.mask_erode,
                   macro_sigma_override=cfg.macro_sigma
                   if cfg.macro_sigma > 0 else None,
                   x_window=cfg.x_window or None)
    cfg.macro_sigma = d["sigma_f"]
    Ny, Nx = d["u"].shape
    Lx, Ly = d["Lx"], d["Ly"]
    dev = cfg.device

    s_test = cfg.s_test
    if s_test <= 0:
        s_test = 2.5 * min(Lx / Nx, Ly / Ny)
        cfg.s_test = s_test

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # base phase
    theta0, grad0 = None, None
    if cfg.init_mode == "data":
        if not np.all(np.isfinite(d["theta0"])):
            raise ValueError("theta0 contains non-finite values; "
                             "cannot use init_mode='data'.")
        theta0 = d["theta0"]
        grad0 = (fd_grad_np(theta0, d["x"], d["y"])
                 if cfg.deriv == "fd"
                 else spectral_grad_np(theta0, Lx, Ly))
    elif cfg.lam_small > 0 or cfg.lam_smooth > 0:
        print("    WARNING: init_mode='none' with lam_small/lam_smooth > 0.\n"
              "    In scratch mode the 'correction' IS theta, so lam_small\n"
              "    pulls theta toward 0 and lam_smooth penalizes |grad\n"
              "    theta|^2, fighting the well term (which wants |grad\n"
              "    theta| = 1). Recommend --lam_small 0 --lam_smooth 0\n"
              "    for scratch fits.")

    # ramp for tapering the correction (never applied to the base phase)
    ramp, grad_ramp = None, None
    if cfg.taper_correction:
        ramp = d["ramp_n"]
        grad_ramp = (fd_grad_np(ramp, d["x"], d["y"])
                     if cfg.deriv == "fd"
                     else spectral_grad_np(ramp, Lx, Ly))

    model = SBVModel(cfg, Ny, Nx, Lx, Ly, theta0=theta0,
                     grad_theta0=grad0, ramp=ramp,
                     grad_ramp=grad_ramp).to(dev)
    if warm_model is not None:
        try:
            model.load_state_dict(warm_model.state_dict())
            print("    warm-started from previous frame.")
        except Exception as e:
            print(f"    warm start failed ({e}); starting fresh.")

    ops = SpectralOps(Ny, Nx, Lx, Ly, cfg.macro_sigma, s_test,
                      device=dev, deriv=cfg.deriv).to(dev)
    x_grid = make_x_grid(d["x"], d["y"], dev)

    e_macro = torch.as_tensor(d["e_macro"], dtype=torch.float32, device=dev)
    mask = torch.as_tensor(d["mask"].astype(np.float32), device=dev)

    opt = torch.optim.Adam([
        {"params": model.corr.parameters(), "lr": cfg.lr},
        {"params": model.calib.parameters(), "lr": cfg.lr_calib},
    ])

    run_name = cfg.run_name(d["stem"], d["frame"])
    run_dir = build_run_dir(out_base, run_name)
    save_config(cfg, run_dir / "config.json")

    history = []
    t0 = time.time()
    for it in range(cfg.iters):
        loss, parts, _ = compute_losses(model, ops, x_grid, e_macro, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        parts.update(model.calib.summary())
        history.append(parts)
        if (it + 1) % log_every == 0 or it == 0:
            msg = "  ".join(f"{k}={v:.4e}" for k, v in parts.items()
                            if k in ("total", "energy", "gauge",
                                     "small", "smooth"))
            cal = model.calib.summary()
            print(f"    [{it+1:5d}/{cfg.iters}] {msg}"
                  f"  | kappa={cal['kappa']:.4f}"
                  f" c_well={cal['c_well']:.4f}"
                  f" c_sing={cal['c_sing']:.4f}"
                  f" ({time.time()-t0:.1f}s)")

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f)

    torch.save({"model": model.state_dict(),
                "config": cfg.__dict__}, run_dir / "checkpoint.pt")

    # final reconstruction, saved for the plotting script
    with torch.no_grad() if cfg.repr_mode == "grid" else torch.enable_grad():
        _, _, diag = compute_losses(model, ops, x_grid, e_macro, mask)
    np.savez_compressed(
        run_dir / f"fields_frame{d['frame']:03d}.npz",
        x=d["x"], y=d["y"],
        u=d["u"], e_macro=d["e_macro"], e_micro=d["e_micro"],
        mask=d["mask"].astype(np.uint8),
        theta0=d["theta0"] if theta0 is not None
        else np.full_like(d["u"], np.nan),
        theta=diag["theta"].detach().cpu().numpy(),
        dtheta=diag["dtheta"].detach().cpu().numpy(),
        q=diag["q"].detach().cpu().numpy(),
        rho_s=diag["rho_s"].detach().cpu().numpy(),
        mu_s=diag["mu_s"].detach().cpu().numpy(),
        well=diag["well"].detach().cpu().numpy(),
        e_model=diag["e_model"].detach().cpu().numpy(),
        abs_sin=diag["abs_sin"].detach().cpu().numpy(),
    )
    print(f"    saved -> {run_dir}")
    return model, run_dir


def main(argv=None):
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--probe_dir", type=str, required=True,
                   help="results/sbv_phase_probe/<stem>/ directory.")
    p.add_argument("--out_base", type=str, default=None,
                   help="Default: experiments/pgb_sbv_nets_v5/results/net_runs")
    p.add_argument("--repr", dest="repr_mode", choices=["field", "grid"],
                   default="field")
    p.add_argument("--energy", dest="energy_mode", choices=["sbv", "bulk"],
                   default="sbv")
    p.add_argument("--init", dest="init_mode", choices=["data", "none"],
                   default="data")
    p.add_argument("--frame", type=int, default=-1)
    p.add_argument("--all_frames", action="store_true",
                   help="Fit every frame, last to first, warm-starting.")
    p.add_argument("--macro_sigma", type=float, default=-1.0,
                   help="Override sigma_f (default: read probe_summary).")
    p.add_argument("--s_test", type=float, default=-1.0)
    p.add_argument("--deriv", choices=["spectral", "fd"],
                   default="spectral",
                   help="Derivative discretization for grad(theta0), "
                        "grad(ramp), grid-mode gradients, and the "
                        "divergence in the tested measure. The sigma_f "
                        "macro filter is always spectral (probe "
                        "convention).")
    p.add_argument("--coarsen", type=int, default=1)
    p.add_argument("--mask_erode", type=int, default=3)
    p.add_argument("--x_window", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Restrict the LOSS MASK to fractions [LO, HI] of "
                        "the x-range (grid/FFTs stay global). Middle 80%% "
                        "of the right half: --x_window 0.55 0.95")
    p.add_argument("--no_taper", action="store_true",
                   help="Disable ramp-tapering of the learned correction.")
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_calib", type=float, default=1e-2)
    p.add_argument("--lam_small", type=float, default=0.0)
    p.add_argument("--lam_smooth", type=float, default=1e-3)
    p.add_argument("--c0", dest="c0_init", type=float, default=0.0,
                   help="Init (or frozen value, see --fix_calib) of the "
                        "affine offset. Roll theory: -R^2/6.")
    p.add_argument("--c_bend", dest="c_bend_init", type=float, default=1.0)
    p.add_argument("--c_well", dest="c_well_init", type=float, default=1.0,
                   help="Roll theory: R/3.")
    p.add_argument("--c_sing", dest="c_sing_init", type=float, default=1.0)
    p.add_argument("--fix_calib", type=str, default="",
                   help="Comma list from {c0,c_bend,c_well,c_sing} or 'all' "
                        "to freeze at the values above. E.g. for the theory "
                        "test at R=0.45: --c0 -0.03375 --c_well 0.15 "
                        "--fix_calib c0,c_well")
    p.add_argument("--kappa_init", type=float, default=0.1)
    p.add_argument("--fix_kappa", action="store_true",
                   help="Freeze the shrinkage threshold scale kappa "
                        "(recommended with delta_gauge > 0).")
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--delta_gauge", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--omega", type=float, default=30.0)
    p.add_argument("--chunk", type=int, default=65536)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    ns = p.parse_args(argv)

    cfg = SBVNetConfig(
        repr_mode=ns.repr_mode, energy_mode=ns.energy_mode,
        init_mode=ns.init_mode, macro_sigma=ns.macro_sigma,
        s_test=ns.s_test, deriv=ns.deriv,
        kappa_init=ns.kappa_init, delta=ns.delta,
        fix_kappa=ns.fix_kappa,
        c0_init=ns.c0_init, c_bend_init=ns.c_bend_init,
        c_well_init=ns.c_well_init, c_sing_init=ns.c_sing_init,
        fix_calib=ns.fix_calib,
        delta_gauge=ns.delta_gauge, lam_small=ns.lam_small,
        lam_smooth=ns.lam_smooth, width=ns.width, depth=ns.depth,
        omega=ns.omega, iters=ns.iters, lr=ns.lr, lr_calib=ns.lr_calib,
        chunk=ns.chunk, seed=ns.seed, coarsen=ns.coarsen,
        mask_erode=ns.mask_erode,
        x_window=tuple(ns.x_window) if ns.x_window else (),
        taper_correction=not ns.no_taper,
    )
    if ns.device is not None:
        cfg.device = ns.device

    out_base = Path(ns.out_base) if ns.out_base is not None \
        else _HERE / "results" / "net_runs"
    out_base.mkdir(parents=True, exist_ok=True)

    if ns.all_frames:
        d0 = load_probe(ns.probe_dir, -1)
        T = d0["T"]
        warm = None
        for frame in range(T - 1, -1, -1):
            print(f"=== frame {frame} ===")
            warm, _ = train_one_frame(cfg, ns.probe_dir, frame, out_base,
                                      warm_model=warm)
    else:
        train_one_frame(cfg, ns.probe_dir, ns.frame, out_base)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # -------------------------------------------------------------
        # No CLI args given: use in-script settings (edit these freely).
        # Any CLI usage overrides this block entirely.
        # -------------------------------------------------------------
        class _Args:
            # probe_dir = str(_HERE / "results" / "sbv_phase_probe" /
            #                 "/Users/edwardmcdugald/patterns/experiments/pgb_sbv_nets_v5/results/sbv_phase_probe/sh_pgb_diamond_mu0.450_T50_N2_Ny1536_Ly188.496_margin0.45_knee_uhu_sigma1.571_dm0.400_ts0.40_gsnone_phase_diamond_ns192_ds0.125_bdinner_ksym_prmsaved_sif0.200_srm0.990_rst0.000")
            probe_dir = str(_HERE / "results" / "sbv_phase_probe_2" /
                            "/Users/edwardmcdugald/patterns/experiments/pgb_sbv_nets_v5/results/sbv_phase_probe_2/sh_pgb_diamond_mu0.600_T100_N2_Ny1536_Ly188.496_margin0.45_knee_uhu_sigma1.250_dm0.400_ts0.40_gsnone_phase_diamond_ns192_ds0.125_bdinner_ksym_prmsaved_sif0.200_srm0.990_rst0.000")
            out_base = None            # default results/net_runs
            repr_mode = "field"        # "field" | "grid"
            energy_mode = "sbv"        # "sbv" | "bulk"
            init_mode = "data"         # "data" | "none"
            frame = -1
            all_frames = False
            #macro_sigma = -1.0         # <=0: read from probe_summary.json
            macro_sigma = 1.25
            s_test = -1.0              # <=0: 2.5 * min(dx, dy)
            deriv = "fd"         # "spectral" | "fd"
            coarsen = 1
            mask_erode = 3
            x_window = (.56,.94)            # e.g. (0.55, 0.95): right half, mid 80%
            taper = True               # ramp-taper the learned correction
            iters = 3000
            lr = 1e-4
            lr_calib = 1e-2
            lam_small = 0.0
            lam_smooth = 1e-3
            c0 = -0.5**2/6.                  # theory: -R^2/6 (R=0.45 -> -0.03375)
            c_bend = 4./9.
            #c_well = 0.1666667               # theory: R/3 (R=0.45 -> 0.15)
            c_well = 1./6.
            c_sing = 1.0
            fix_calib = "c_well,c_bend"             # e.g. "c0,c_well" or "all"
            kappa_init = 0.4           # with fix_kappa: the classification
                                       # scale; GB q ~ 0.4-0.5, smooth q small
            fix_kappa = False           # recommended (see sbv_net_core note)
            delta = 0.05
            delta_gauge = 1e-3
            width = 32
            depth = 4
            omega = 30.0
            chunk = 65536
            seed = 0
            device = None              # None: auto (cuda if available)

        a = _Args()
        argv = [
            "--probe_dir", a.probe_dir,
            "--repr", a.repr_mode,
            "--energy", a.energy_mode,
            "--init", a.init_mode,
            "--frame", str(a.frame),
            "--macro_sigma", str(a.macro_sigma),
            "--s_test", str(a.s_test),
            "--deriv", a.deriv,
            "--coarsen", str(a.coarsen),
            "--mask_erode", str(a.mask_erode),
            "--iters", str(a.iters),
            "--lr", str(a.lr),
            "--lr_calib", str(a.lr_calib),
            "--lam_small", str(a.lam_small),
            "--lam_smooth", str(a.lam_smooth),
            "--c0", str(a.c0),
            "--c_bend", str(a.c_bend),
            "--c_well", str(a.c_well),
            "--c_sing", str(a.c_sing),
            "--fix_calib", a.fix_calib,
            "--kappa_init", str(a.kappa_init),
            "--delta", str(a.delta),
            "--delta_gauge", str(a.delta_gauge),
            "--width", str(a.width),
            "--depth", str(a.depth),
            "--omega", str(a.omega),
            "--chunk", str(a.chunk),
            "--seed", str(a.seed),
        ]
        if a.x_window is not None:
            argv += ["--x_window", str(a.x_window[0]), str(a.x_window[1])]
        if a.out_base is not None:
            argv += ["--out_base", a.out_base]
        if a.device is not None:
            argv += ["--device", a.device]
        if a.all_frames:
            argv.append("--all_frames")
        if not a.taper:
            argv.append("--no_taper")
        if a.fix_kappa:
            argv.append("--fix_kappa")
        main(argv)
    else:
        main()
