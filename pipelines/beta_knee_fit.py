"""
beta_knee_fit.py

Measures the self-dual balance coefficient beta (div k = -beta w,
w = |k|^2 - 1) on relaxed SH knee frames, using estimators validated on
synthetic knees with planted beta.

Pipeline (validated to ~2% on synthetics):
  1. Demodulate u by the analytic signal along x (no smoothing).
  2. x-average the wavevector fields: the knee layer is x-independent,
     so x-averaging removes the 2-theta harmonic ripple exactly,
     with no broadening of the layer in y.
  3. Estimator A: fit k2bar(y) = m tanh(q (y - y0)) with free amplitude;
     beta = q/m.  (Primary; insensitive to differentiation noise.)
  4. Estimator B: 1D regression d k2bar/dy = -beta * wbar on the layer.
     (Corroborating; attenuation-biased low, use for consistency only.)

Modes:
  --validate   run the synthetic recovery table (plant beta, recover it)
  --smoothing  forward-model the uhu kernel bias: smooth a true-beta
               tanh at sigma and report the apparent beta
  default      analyze npz frames (expects x, y, u[...,frame],
               sh_meta_json with mu, Rscale, gb_row_in_crop)

Provenance note: uhu k fields extracted at sigma = 1.571 are broadened;
tanh fits on them read beta ~ 0.78-0.92 when the true beta is ~1.1.
Do not fit beta from wide-kernel fields without forward-modeling the
kernel.
"""
import json
import numpy as np
from pathlib import Path
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


def synth_knee(x, y, mu, R, beta, yc=0.0):
    k1 = np.sqrt(1.0 - mu*mu)
    X, Y = np.meshgrid(x, y)
    arg = beta*mu*(Y - yc)
    th = k1*X + (1.0/beta)*(np.abs(arg) + np.log1p(np.exp(-2*np.abs(arg))))
    ksq = k1*k1 + (mu*np.tanh(arg))**2
    A1 = np.sqrt(np.maximum(4.0/3.0*(R - (ksq-1.0)**2), 0.0))
    A3 = -A1**3/(4.0*(R + 79.0*ksq**2 - 14.0*ksq - 1.0))
    return A1*np.cos(th) + A3*np.cos(3*th)


def profile_estimators(u, x, y, jc, mu, edge_cols=20):
    dx, dy = x[1]-x[0], y[1]-y[0]
    ua = hilbert(u, axis=1); eps = 1e-12
    k1 = np.imag(np.conj(ua)*np.gradient(ua, dx, axis=1))/(np.abs(ua)**2+eps)
    k2 = np.imag(np.conj(ua)*np.gradient(ua, dy, axis=0))/(np.abs(ua)**2+eps)
    sl = slice(edge_cols, len(x)-edge_cols)
    k2b = k2[:, sl].mean(1)
    wb = (k1[:, sl]**2 + k2[:, sl]**2).mean(1) - 1.0
    band = np.abs(y - y[jc]) < 4.5/(0.9*mu)
    band[:10] = band[-10:] = False
    f = lambda yv, m, q, y0: m*np.tanh(q*(yv-y0))
    p, _ = curve_fit(f, y[band], k2b[band], p0=[mu, 0.9*mu, y[jc]],
                     maxfev=9000)
    beta_tanh, m_ff = abs(p[1]/p[0]), abs(p[0])
    dk2 = np.gradient(k2b, dy)
    m2 = band & (np.abs(wb) > 0.2*mu*mu)
    X, Y = wb[m2], dk2[m2]
    c = (X @ Y)/(X @ X)
    r2 = 1.0 - np.sum((Y - c*X)**2)/np.sum((Y - Y.mean())**2)
    return beta_tanh, m_ff, -c, r2


def run_validation():
    print("synthetic recovery (tanh estimator is primary):")
    print(f"{'mu':>4} {'R':>4} {'b_true':>7} | {'tanh':>7} {'m_ff':>6} | {'reg':>7} {'R2':>6}")
    for mu, R in [(0.3, 0.5), (0.4, 1.0), (0.5, 1.0)]:
        x = np.arange(256)*0.12; y = (np.arange(256)-128)*0.12
        for bt in (0.87, 1.00, 1.15):
            u = synth_knee(x, y, mu, R, bt)
            b1, mf, b2, r2 = profile_estimators(u, x, y, 128, mu)
            print(f"{mu:4.1f} {R:4.1f} {bt:7.2f} | {b1:7.4f} {mf:6.3f} | {b2:7.4f} {r2:6.3f}")


def run_smoothing_model(sigma=1.571):
    f = lambda yv, m, q, y0: m*np.tanh(q*(yv-y0))
    dy = 0.12; y = (np.arange(2000)-1000)*dy
    print(f"apparent beta after Gaussian smoothing at sigma={sigma}:")
    for mu, bt in [(0.3, 1.08), (0.4, 1.08), (0.5, 1.16)]:
        k2s = gaussian_filter1d(mu*np.tanh(bt*mu*y), sigma/dy)
        p, _ = curve_fit(f, y, k2s, p0=[mu, 0.9*mu, 0.0], maxfev=9000)
        print(f"  mu={mu}: beta_true={bt} -> apparent {abs(p[1]/p[0]):.4f}")


def run_data(paths, frame=-1):
    print(f"{'mu':>4} {'R':>4} | {'tanh beta':>9} {'m_ff':>6} | {'reg beta':>8} {'R2':>6}")
    for p in paths:
        d = np.load(p, allow_pickle=True)
        sh = json.loads(str(d['sh_meta_json']))
        b1, mf, b2, r2 = profile_estimators(d['u'][..., frame], d['x'],
                                            d['y'], sh['gb_row_in_crop'],
                                            sh['mu'])
        print(f"{sh['mu']:4.1f} {sh['Rscale']:4.1f} | {b1:9.4f} {mf:6.3f} | "
              f"{b2:8.4f} {r2:6.3f}")


class _Args:
    mode = 'data'          # 'data' | 'validate' | 'smoothing'
    # EDIT THIS to your data directory (the /mnt path only exists in the
    # Claude sandbox where uploads land):
    data_dir = Path('/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweeps_shallow_v2_Rp5_Nx512_tmax500_hp5_np18_Ny5/raw')
    pattern = 'sh_pgb_zigzag_mu0_*.npz'
    frame = -1

    @classmethod
    def get_paths(cls):
        paths = sorted(cls.data_dir.glob(cls.pattern))
        if not paths:
            print(f'No files matched {cls.data_dir}/{cls.pattern}; '
                  'edit _Args.data_dir/pattern or pass paths on the CLI.')
        return paths


if __name__ == '__main__':
    import sys
    a = sys.argv[1:]
    if '--validate' in a:
        run_validation()
    elif '--smoothing' in a:
        run_smoothing_model()
    else:
        paths = [Path(p) for p in a if p.endswith('.npz')] or _Args.get_paths()
        run_data(paths, _Args.frame)