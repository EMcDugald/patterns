import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.spectral import SpectralDerivs, macro
from utils.geometry import build_rectangular_ramp_smooth
from op_extract.uhu_smoother import compute_uhu_ops_smoother
from op_extract.uhu_adaptive import compute_uhu_ops_adaptive
from op_extract.kfield_emb import compute_kfield_emb

Nx, Ny = 256, 128
Lx, Ly = 40 * np.pi, 20 * np.pi
x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
X, Y = np.meshgrid(x, y)

kx, m, beta = 1.0, 0.5, 4.0   # sharp knee
theta = kx*X + (m/beta)*np.log(np.cosh(beta*Y))
u = np.cos(theta)
k1t = np.full_like(X, kx)
k2t = m*np.tanh(beta*Y)

ramp = build_rectangular_ramp_smooth(x, y, xmargin=0.05, ymargin=0.10, tanhscale=60.0)
interior = ramp >= 0.99
core = np.abs(Y) < 2.0/beta
sigma = np.pi/2

def report(name, k1, k2, t):
    ek = np.abs(np.hypot(k1, k2) - np.hypot(k1t*ramp, k2t*ramp))
    dth = np.arctan2(k2, k1) - np.arctan2(k2t, k1t)
    dth = np.abs(np.mod(dth + np.pi/2, np.pi) - np.pi/2)
    msk = interior
    mc = interior & core
    print(f"{name:12s} |k|err med {np.median(ek[msk]):.4f} p95 {np.percentile(ek[msk],95):.4f}"
          f" | core med {np.median(ek[mc]):.4f} p95 {np.percentile(ek[mc],95):.4f}"
          f" | ang med {np.degrees(np.median(dth[msk])):.2f} deg | {t:.1f}s")

t0 = time.time()
iso = compute_uhu_ops_smoother(u, x, y, lambda a: macro(a, sigma, Lx, Ly), SpectralDerivs, ramp)
report("iso_global", iso["k1_orig"], iso["k2_orig"], time.time()-t0)

t0 = time.time()
ada = compute_uhu_ops_adaptive(u, x, y, sigma, SpectralDerivs, ramp,
                               n_iter=3, aspect_max=2.5, sharpen=2.0,
                               damping=0.4, patch=64, orient="slowness")
report("adaptive", ada["k1_orig"], ada["k2_orig"], time.time()-t0)
print("  iter_delta:", np.round(ada["iter_delta"], 5), " n_iter_run:", ada["n_iter_run"])

# warm start second call, as the runner does frame-to-frame
t0 = time.time()
seed = {k: ada[k] for k in ("k1_orig", "k2_orig", "lam1", "lam2")}
ada2 = compute_uhu_ops_adaptive(u, x, y, sigma, SpectralDerivs, ramp,
                                n_iter=2, seed=seed, orient="slowness")
report("adaptive_ws", ada2["k1_orig"], ada2["k2_orig"], time.time()-t0)

t0 = time.time()
emb = compute_kfield_emb(u, x, y, SpectralDerivs, ramp, box_wavelengths=1.0, n_smooth=2)
report("emb", emb["k1_orig"], emb["k2_orig"], time.time()-t0)

# core k2 profile through the bend, mid column
col = Nx//2
print("\nk2 profile through the core (y in [-2,2]):")
sel = np.abs(y) < 2.0
for name, f in (("truth", k2t), ("iso", iso["k2_orig"]), ("adaptive", ada["k2_orig"]), ("emb", emb["k2_orig"])):
    prof = f[sel, col]
    print(f"{name:9s}", np.round(prof[::2], 3))

# core |k| at the very center: truth = kx = 1
row = np.argmin(np.abs(y))
print("\n|k| at core row, mid column: truth", kx)
for name, r in (("iso", iso), ("adaptive", ada), ("emb", emb)):
    print(f"{name:9s}", round(float(np.hypot(r['k1_orig'][row, col], r['k2_orig'][row, col])), 4))
