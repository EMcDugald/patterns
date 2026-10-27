"""sh_zigzag_standin.py  v1.0  (12 Oct 2026)

Stand-in Swift-Hohenberg zigzag movie for validating sweep drivers in the
sandbox.  NOT the solver of the data runs (src/sh_sims was not available
here): a minimal periodic pseudo-spectral integration of

    u_t = R u - (1 + Lap)^2 u - u^3

from the sharp knee u0 = amp cos(k1 x + mu |y|) on a box that holds two
knees (y = 0 and the periodic edge), saved in the layout of the data files
(u [ny, nx, nt] float32, x, y, tt, metadata_json with mu and
gb_row_in_crop) so that theta_t_mu_sweep.py reads it unchanged.

Output: <outdir>/sh_zigzag_standin_mu<mu>_T<T>_N<nsave>_nx<nx>_Ny<ny>.npz

Changelog
  v1.0  semi-implicit Euler (linear part implicit, R u - u^3 explicit),
        dt 0.05; frames every T/(nsave-1); k1 = sqrt(1 - mu^2) rounded to an
        integer number of periods across the box.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy import fft as sfft


def run(args):
    mu = args.mu
    k1 = np.sqrt(1 - mu ** 2)
    nper = max(1, int(round(args.nx * args.dx * k1 / (2 * np.pi))))
    Lx = nper * 2 * np.pi / k1
    dx = Lx / args.nx
    Ly = args.ny * dx
    x = (np.arange(args.nx) - args.nx // 2) * dx
    y = (np.arange(args.ny) - args.ny // 2) * dx
    X, Y = np.meshgrid(x, y)
    u = args.amp * np.cos(k1 * X + mu * np.abs(Y))
    if args.noise > 0:
        u = u + args.noise * np.random.default_rng(args.seed).standard_normal(u.shape)
    qx = 2 * np.pi * sfft.fftfreq(args.nx, dx)
    qy = 2 * np.pi * sfft.fftfreq(args.ny, dx)
    QX, QY = np.meshgrid(qx, qy)
    Lin = (1.0 - (QX ** 2 + QY ** 2)) ** 2
    dt = args.dt
    nsave = args.nsave
    tsave = np.linspace(0.0, args.T, nsave)
    frames = np.empty((args.ny, args.nx, nsave), np.float32)
    frames[..., 0] = u
    t, isave = 0.0, 1
    t0 = time.time()
    uh = sfft.fft2(u)
    while isave < nsave:
        nl = sfft.fft2(args.R * u - u ** 3)
        uh = (uh + dt * nl) / (1.0 + dt * Lin)
        u = np.real(sfft.ifft2(uh)); t += dt
        if t >= tsave[isave] - 1e-9:
            frames[..., isave] = u; isave += 1
    md = dict(mu=mu, k1=float(k1), k2=mu, Rscale=args.R, amp=args.amp, h=dt, tmax=args.T, nsave=nsave,
              dx=float(dx), dy=float(dx), Nx=args.nx, Ny=args.ny, gb_row_in_crop=int(args.ny // 2),
              which_gb="standin", standin=True, note="sh_zigzag_standin v1.0, not the data solver")
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    name = "sh_zigzag_standin_mu%.3f_T%g_N%d_nx%d_Ny%d.npz" % (mu, args.T, nsave, args.nx, args.ny)
    np.savez_compressed(out / name, u=frames, x=x, y=y, tt=tsave, metadata_json=json.dumps(md))
    print("wrote %s  (%.1f s; %d periods, Lx %.1f, Ly %.1f, dx %.3f)" % (out / name, time.time() - t0, nper, Lx, Ly, dx))


class _Args:
    mu = 0.85
    R = 0.5
    amp = 1.0
    nx = 256
    ny = 768
    dx = 0.21
    dt = 0.05
    T = 25.0
    nsave = 126
    noise = 0.0
    seed = 0
    outdir = "results/standin_movies"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in vars(_Args) if not a.startswith("_")]:
        p.add_argument("--" + k, type=type(getattr(_Args, k)), default=getattr(_Args, k))
    run(p.parse_args())
