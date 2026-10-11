"""
sh_ellipse_standin.py

Stand-in Swift-Hohenberg ellipse runner for comparison with the RCN zipper
construction. It reproduces the setup visible in the 2025 scripts (the
`eikonal` helper in run_stitch_zippers_on_ellipse.py and the parameters of
run_sh_eikonal_ellipse.py); the src/sh_sims solver itself was not available
here, so this is a minimal re-implementation, not that code.

Model:   u_t = R(x) u - (1 + lap)^2 u - u^3   on a periodic box Lx x Ly
Ramp:    R = R0 tanh( k_ramp * sqrt(Lx^2+Ly^2) * (beta - sqrt((X/Lx)^2+(Y/Ly)^2)) / 2 )
         (R0 inside the ellipse with semi-axes beta*Lx, beta*Ly, -R0 outside)
Initial: u0 = amp * sin(d), d = exact distance to the ellipse wall (eikonal,
         phase 0 on the wall), zero outside.
Stepping: pseudo-spectral, semi-implicit (linear operator implicit, R u - u^3
         explicit), dt = 0.5 by default.

Outputs: pipelines/results/sh_ellipse_standin/<tag>.npz with keys xx, yy,
tt, uu (ny, nx, nt; same key names as the 2025 runner), R, A, B.

Changelog
---------
v1 (2026-10-08). First version, for the aspect-ratio sweep comparison.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import fft as sfft

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_NAME = Path(__file__).stem
sys.path.insert(0, str(SCRIPT_DIR))
from ellipse_zipper_assemble import Ellipse  # noqa: E402  (exact wall distance)


def run(args):
    out = Path(args.out_dir) if args.out_dir else SCRIPT_DIR / "results" / SCRIPT_NAME
    out.mkdir(parents=True, exist_ok=True)
    Lx, Ly, nx, ny = args.Lx, args.Ly, args.nx, args.ny
    x = (Lx / nx) * (np.arange(nx) - nx // 2)
    y = (Ly / ny) * (np.arange(ny) - ny // 2)
    X, Y = np.meshgrid(x, y)
    A, B = args.beta * Lx, args.beta * Ly
    R = args.R0 * np.tanh(args.k_ramp * np.sqrt(Lx ** 2 + Ly ** 2)
                          * (args.beta - np.sqrt((X / Lx) ** 2 + (Y / Ly) ** 2)) / 2.0)
    E = Ellipse(A, B)
    _, d, _, _ = E.foot(np.abs(X), np.abs(Y))
    ins = E.inside(X, Y)
    u = np.where(ins, args.amp * np.sin(d), 0.0)
    if args.noise > 0:
        u = u + args.noise * np.random.default_rng(args.seed).standard_normal(u.shape)
    kx = 2 * np.pi * sfft.fftfreq(nx, Lx / nx)
    ky = 2 * np.pi * sfft.rfftfreq(ny, Ly / ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")          # rfft along y (axis 0 of u.T)
    Lop = (1.0 - (KX ** 2 + KY ** 2)) ** 2
    denom = 1.0 + args.dt * Lop
    save_t = sorted(set([0.0] + list(args.save_times)))
    frames, tt = [], []
    uT = u.T.copy()                                       # (nx, ny)
    uh = sfft.rfft2(uT, workers=args.workers)
    t, step = 0.0, 0
    nsteps = int(round(args.tmax / args.dt))
    t0 = time.time()
    for step in range(nsteps + 1):
        t = step * args.dt
        if any(abs(t - s) < 0.5 * args.dt for s in save_t):
            frames.append(uT.T.astype(np.float32).copy()); tt.append(t)
            print("  t=%7.1f  max|u|=%.3f  [%.0fs]" % (t, np.abs(uT).max(), time.time() - t0), flush=True)
        if step == nsteps:
            break
        N = R.T * uT - uT ** 3
        uh = (uh + args.dt * sfft.rfft2(N, workers=args.workers)) / denom
        uT = sfft.irfft2(uh, s=uT.shape, workers=args.workers)
    uu = np.stack(frames, axis=-1)
    np.savez_compressed(out / ("%s.npz" % args.tag), xx=x, yy=y, tt=np.array(tt), uu=uu, R=R.astype(np.float32),
                        A=A, B=B, Lx=Lx, Ly=Ly, R0=args.R0, dt=args.dt)
    (out / ("%s_args.json" % args.tag)).write_text(json.dumps({k: getattr(args, k) for k in dir(_Args)
                                                               if not k.startswith("_")}, indent=2, default=str))
    print("wrote", out / ("%s.npz" % args.tag), "in %.0fs" % (time.time() - t0))


class _Args(object):
    tag = "ellipse_ab2"
    Lx = 600.0
    Ly = 300.0
    nx = 864
    ny = 432
    beta = 0.45
    R0 = 0.5
    k_ramp = 3.0
    amp = 0.1
    noise = 0.0
    seed = 0
    dt = 0.5
    tmax = 3000.0
    save_times = [50.0, 200.0, 500.0, 1000.0, 2000.0, 3000.0]
    workers = 1
    out_dir = None


def _parse_cli():
    d = _Args()
    p = argparse.ArgumentParser(description="Stand-in SH ellipse runner")
    p.add_argument("--tag", default=d.tag)
    for k in ["Lx", "Ly", "beta", "R0", "k_ramp", "amp", "noise", "dt", "tmax"]:
        p.add_argument("--" + k, type=float, default=getattr(d, k))
    for k in ["nx", "ny", "seed", "workers"]:
        p.add_argument("--" + k, type=int, default=getattr(d, k))
    p.add_argument("--save_times", type=float, nargs="+", default=d.save_times)
    p.add_argument("--out_dir", default=d.out_dir)
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_cli() if len(sys.argv) > 1 else _Args())
