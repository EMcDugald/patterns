"""
extraction_comparison.py

Three-way beta comparison on raw SH knee frames (run from a directory
where src/utils, src/op_extract or flat copies of the frozen method
layer are importable; see _Args):

  1. calibration: synthetic knees (planted beta) through the REAL
     riesz_k_field (t1_common) at sigma_dir = 0.4 and 1.571, and
     through the demodulation estimator of beta_knee_fit.py
  2. data: the same three extractors on raw-u npz files
     (keys: x, y, u[...,frame], metadata_json with mu, Rscale,
      gb_row_in_crop)

Read the data through the calibration: e.g. if riesz(0.4) reads 0.71
on a synthetic knee of true beta 1.0 at this mu, then a data reading of
0.79 implies true beta ~ 1.1. The demodulation estimator is unbiased at
the 2 percent level on tanh knees and needs no correction.

Result of record (final frames, T=500):
  beta(mu, R) ~ 1 + c(R) mu^2  with  c(0.5) ~ 0.9,  c(1.0) ~ 0.55
  (both families extrapolate to beta -> 1 as mu -> 0)
"""
import sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit


def tanh_beta(k2sig, y, jc, mu):
    band = np.abs(y - y[jc]) < 4.5/(0.9*mu); band[:10] = band[-10:] = False
    f = lambda yv, m, q, y0: m*np.tanh(q*(yv-y0))
    p, _ = curve_fit(f, y[band], k2sig[band], p0=[mu, 0.9*mu, y[jc]],
                     maxfev=9000)
    return abs(p[1]/p[0]), abs(p[0])


def riesz_beta(tc, u, x, y, jc, mu, sigma_dir, sigma_avg=None):
    Lx = x[-1]-x[0]+(x[1]-x[0]); Ly = y[-1]-y[0]+(y[1]-y[0])
    taper = tc.edge_taper(x, y, frac=0.06)
    rk = tc.riesz_k_field(u*taper, x, y, Lx, Ly, sigma_dir=sigma_dir,
                          sigma_avg=sigma_avg)
    sl = slice(24, len(x)-24)
    return tanh_beta(rk['k2'][:, sl].mean(1), y, jc, mu)


def uhu_beta(tc, us, fdv, u, x, y, jc, mu, sigma):
    """Genuine uHu tensor route: compute_uhu_ops_smoother -> eigen k ->
    sign-oriented k2 (odd about the core) -> x-averaged profile ->
    tanh fit with the seam rows near the core excluded (the sign flip
    of |k2| leaves a seam there because the smoothed |k2| does not
    reach zero)."""
    import numpy as np
    from scipy.optimize import curve_fit
    Lx = x[-1]-x[0]+(x[1]-x[0]); Ly = y[-1]-y[0]+(y[1]-y[0])
    taper = tc.edge_taper(x, y, frac=0.06)
    sm = us.make_gaussian_smoother(sigma, Lx, Ly, fdv.macro_iso)
    ops = us.compute_uhu_ops_smoother(u*taper, x, y, sm, fdv.spectral_deriv,
                                      np.ones_like(u))
    s = np.sign(y[:, None]-y[jc]); s[s == 0] = 1
    k2 = np.abs(ops['k2_orig'])*s
    sl = slice(24, len(x)-24)
    k2b = k2[:, sl].mean(1)
    keep = np.abs(y - y[jc]) > max(0.8, 0.6*sigma)
    band = (np.abs(y-y[jc]) < 4.5/(0.9*mu)) & keep
    band[:10] = band[-10:] = False
    f = lambda yv, m, q, y0: m*np.tanh(q*(yv-y0))
    p, _ = curve_fit(f, y[band], k2b[band], p0=[mu, 0.9*mu, y[jc]],
                     maxfev=9000)
    return abs(p[1]/p[0]), abs(p[0])


def make_figures(tc, u, x, y, jc, mu, R, beta_fit, stem, out_root):
    """Diagnostic figures in the style of knee_bend_fits_and_residuals /
    self_dual_sindy_patch: fields, bending-vs-stretching balance with
    errors, x-averaged k2 profiles with tanh fits, and the div k vs w
    scatter with the fitted slope. Uses the demodulation k fields
    (validated unbiased at the 2 percent level on tanh knees)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import hilbert
    out = Path(out_root) / stem
    out.mkdir(parents=True, exist_ok=True)
    dx, dy = x[1]-x[0], y[1]-y[0]
    ua = hilbert(u, axis=1); eps = 1e-12
    k1 = np.imag(np.conj(ua)*np.gradient(ua, dx, axis=1))/(np.abs(ua)**2+eps)
    k2 = np.imag(np.conj(ua)*np.gradient(ua, dy, axis=0))/(np.abs(ua)**2+eps)
    sl = slice(24, len(x)-24)
    k1b, k2b = k1[:, sl].mean(1), k2[:, sl].mean(1)
    wb = (k1[:, sl]**2 + k2[:, sl]**2).mean(1) - 1.0
    dk2b = np.gradient(k2b, dy)
    band = np.abs(y - y[jc]) < 4.5/(0.9*mu); band[:10] = band[-10:] = False

    # RCN coefficients, corrected values (half convention): alpha = eta = 2R/3
    ALPHA = ETA = 2.0*R/3.0
    G2 = wb**2 - wb**4/(2.0*R)          # exact SH stretching function
    E_bend = ETA*dk2b**2                 # 1D: div k = d k2bar / dy
    E_str = ALPHA*G2
    denom = E_bend + E_str + 1e-14

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    axs[0, 0].plot(y[band], E_bend[band], label=r"$\eta\,(\nabla\!\cdot k)^2$")
    axs[0, 0].plot(y[band], E_str[band], label=r"$\alpha\,G^2$")
    axs[0, 0].set_title(f"bending vs stretching  (alpha=eta=2R/3, R={R})")
    axs[0, 0].legend(); axs[0, 0].set_xlabel("y")
    axs[0, 1].plot(y[band], (E_bend-E_str)[band])
    axs[0, 1].set_title("difference  eta(div k)^2 - alpha G^2")
    axs[0, 1].set_xlabel("y")
    axs[1, 0].plot(y[band], (np.abs(E_bend-E_str)/denom)[band])
    axs[1, 0].set_title("relative error"); axs[1, 0].set_xlabel("y")
    X, Y = wb[band], dk2b[band]
    axs[1, 1].plot(X, Y, ".", ms=3, alpha=0.6)
    xx = np.linspace(X.min(), X.max(), 50)
    axs[1, 1].plot(xx, -beta_fit*xx, "r-",
                   label=f"div k = -{beta_fit:.3f} w")
    axs[1, 1].set_xlabel("w = |k|^2 - 1"); axs[1, 1].set_ylabel("div k")
    axs[1, 1].legend(); axs[1, 1].set_title("self-dual balance scatter")
    fig.suptitle(f"{stem}   mu={mu}  R={R}  beta_fit={beta_fit:.3f}")
    fig.tight_layout(); fig.savefig(out/"balance_and_scatter.png", dpi=200)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    ext = [x[0], x[-1], y[0], y[-1]]
    for ax, F, t in [(axs[0], u, "u (final)"),
                     (axs[1], k1**2+k2**2-1.0, "w = |k|^2-1 (demod)"),
                     (axs[2], np.gradient(k1, dx, axis=1)
                              + np.gradient(k2, dy, axis=0), "div k (demod)")]:
        im = ax.imshow(F, origin="lower", extent=ext, cmap="RdBu_r")
        ax.set_title(t); fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(out/"fields.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(y[band], k2b[band], "k.", ms=3, label="x-averaged k2 (demod)")
    yy = np.linspace(y[band].min(), y[band].max(), 400)
    ax.plot(yy, mu*np.tanh(beta_fit*mu*(yy-y[jc])), "r-",
            label=f"mu tanh(beta mu y), beta={beta_fit:.3f}")
    ax.plot(yy, mu*np.tanh(0.866*mu*(yy-y[jc])), "g--", lw=1,
            label="beta=0.866 (historical)")
    ax.plot(yy, mu*np.tanh(1.0*mu*(yy-y[jc])), "b:", lw=1, label="beta=1")
    ax.set_xlabel("y"); ax.set_ylabel("k2"); ax.legend()
    ax.set_title(f"knee profile  mu={mu} R={R}")
    fig.tight_layout(); fig.savefig(out/"k2_profile_fits.png", dpi=200)
    plt.close(fig)
    print(f"   figures -> {out}")


def main(paths, do_calibration=True):
    import t1_common as tc
    import uhu_smoother as us
    import field_diagnostics_viz as fdv
    from beta_knee_fit import synth_knee, profile_estimators
    cal = {}   # cal[(mu, col)] = list of (b_true, reading)
    if do_calibration:
        print("== calibration: synthetic u through the real extractors ==")
        print(f"{'mu':>5} {'b_true':>7} | {'riesz 0.4/pi':>12} {'r 0.4/pi2':>9} | {'uhu 1.571':>9} {'uhu 0.9':>8} | {'demod':>7}")
        x = np.arange(256)*0.122; y = (np.arange(256)-128)*0.122
        for mu in (0.375, 0.475):
            for bt in (0.9, 1.0, 1.1):
                u = synth_knee(x, y, mu, 0.5, bt)
                bpi, _ = riesz_beta(tc, u, x, y, 128, mu, 0.4)
                bp2, _ = riesz_beta(tc, u, x, y, 128, mu, 0.4, np.pi/2)
                u157, _ = uhu_beta(tc, us, fdv, u, x, y, 128, mu, 1.571)
                u09, _ = uhu_beta(tc, us, fdv, u, x, y, 128, mu, 0.9)
                bd, _, _, _ = profile_estimators(u, x, y, 128, mu)
                for col, v in [("rpi", bpi), ("rp2", bp2), ("u157", u157),
                               ("u09", u09), ("demod", bd)]:
                    cal.setdefault((mu, col), []).append((bt, v))
                print(f"{mu:5.3f} {bt:7.2f} | {bpi:12.4f} {bp2:9.4f} | {u157:9.4f} {u09:8.4f} | {bd:7.4f}")
    def cal_invert(mu, col, reading):
        """linear interpolation reading -> implied true beta, using the
        nearest calibrated mu; returns nan if no calibration."""
        if not cal:
            return float('nan')
        mus = sorted({m for (m, c) in cal if c == col})
        if not mus:
            return float('nan')
        mc = min(mus, key=lambda m: abs(m-mu))
        pts = sorted(cal[(mc, col)], key=lambda t: t[1])
        bs = [b for b, r in pts]; rs = [r for b, r in pts]
        return float(np.interp(reading, rs, bs,
                     left=bs[0]+(reading-rs[0])*(bs[1]-bs[0])/(rs[1]-rs[0]),
                     right=bs[-1]+(reading-rs[-1])*(bs[-1]-bs[-2])/(rs[-1]-rs[-2])))

    rows = []
    print("\n== data (final frames) ==")
    print(f"{'mu':>5} {'R':>4} | {'riesz 0.4/pi':>12} {'r 0.4/pi2':>9} | {'uhu 1.571':>9} {'uhu 0.9':>8} | {'demod':>7} {'m_ff':>6}")
    def key(q):
        md = json.loads(str(np.load(q, allow_pickle=True)['metadata_json']))
        return (md['mu'], md['Rscale'])
    for p in sorted(paths, key=key):
        d = np.load(p, allow_pickle=True)
        md = json.loads(str(d['metadata_json']))
        mu, R, jc = md['mu'], md['Rscale'], md['gb_row_in_crop']
        x, yv = d['x'], d['y']; u = d['u'][..., -1]
        bpi, _ = riesz_beta(tc, u, x, yv, jc, mu, 0.4)
        bp2, _ = riesz_beta(tc, u, x, yv, jc, mu, 0.4, np.pi/2)
        u157, _ = uhu_beta(tc, us, fdv, u, x, yv, jc, mu, 1.571)
        u09, _ = uhu_beta(tc, us, fdv, u, x, yv, jc, mu, 0.9)
        bd, mfd, _, _ = profile_estimators(u, x, yv, jc, mu)
        rows.append((mu, R, bpi, bp2, u157, u09, bd))
        print(f"{mu:5.3f} {R:4.1f} | {bpi:12.4f} {bp2:9.4f} | {u157:9.4f} {u09:8.4f} | {bd:7.4f} {mfd:6.3f}")
        if _Args.make_figures:
            make_figures(tc, u, x, yv, jc, mu, R, bd, p.stem,
                         _Args.results_root)
    if cal and rows:
        print("\n== implied true beta (each column read through its own")
        print("   synthetic calibration; demod shown raw, unbiased +-2%) ==")
        print(f"{'mu':>5} {'R':>4} | {'riesz pi':>8} {'riesz pi2':>9} | {'uhu 1.571':>9} {'uhu 0.9':>8} | {'demod':>7}")
        for (mu, R, bpi, bp2, u157, u09, bd) in rows:
            print(f"{mu:5.3f} {R:4.1f} | "
                  f"{cal_invert(mu,'rpi',bpi):8.3f} {cal_invert(mu,'rp2',bp2):9.3f} | "
                  f"{cal_invert(mu,'u157',u157):9.3f} {cal_invert(mu,'u09',u09):8.3f} | {bd:7.3f}")


class _Args:
    # Directories containing the method layer, inserted onto sys.path
    # in order. src/ itself must be present so t1_common's internal
    # package imports (op_extract.uhu_smoother, utils.field_diagnostics_viz)
    # resolve.
    method_dirs = ['/Users/edwardmcdugald/patterns/src/op_extract',
                   '/Users/edwardmcdugald/patterns/src/utils',
                   '/Users/edwardmcdugald/patterns/src']
    code_dirs = ['.', './src', '/Users/edwardmcdugald/patterns/pipelines']
    make_figures = True
    results_root = Path(__file__).resolve().parent / 'results' / 'extraction_comparison_R1p0'
    data_dir = Path('/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/mu_sweeps_shallow_v2_R1p0_Nx512_tmax500_hp5_np18_Ny5/raw')
    pattern = 'R*_sh_pgb_zigzag_mu0*.npz'


def _discover_method_layer(root, targets=('t1_common.py', 'uhu_smoother.py',
                                          'beta_knee_fit.py'),
                           skip=('data', 'results', 'figs', 'figures',
                                 '.git', '__pycache__', 'venv', '.idea'),
                           max_depth=6):
    """Walk the repo for the method-layer modules, pruning heavy dirs.
    For each hit, add both its own directory (flat imports) and its
    parent (so package imports like utils.x / op_extract.x resolve)."""
    import os
    found = []
    root = str(root)
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        if dirpath.count(os.sep) - base_depth >= max_depth:
            dirnames[:] = []
        for t in targets:
            if t in filenames:
                found.append(dirpath)
                found.append(str(Path(dirpath).parent))
    return found


if __name__ == '__main__':
    _HERE = Path(__file__).resolve().parent
    _ROOT = _HERE.parent
    search_dirs = (list(_Args.method_dirs)
                   + [str(_HERE), str(_ROOT / "src")]
                   + _Args.code_dirs)
    for cd in dict.fromkeys(search_dirs):          # dedupe, keep order
        sys.path.insert(0, cd)
    try:
        import t1_common  # noqa: F401  (fail fast with a useful message)
    except ModuleNotFoundError:
        print("Could not import t1_common. Searched (in order):")
        for cd in dict.fromkeys(search_dirs):
            print("  ", cd)
        print("Fix: edit _Args.method_dirs so it lists the directories "
              "containing t1_common.py, field_diagnostics_viz.py, and "
              "uhu_smoother.py, plus src/ itself; and keep "
              "beta_knee_fit.py next to this script.")
        sys.exit(1)
    args = [a for a in sys.argv[1:] if a.endswith('.npz')]
    paths = [Path(a) for a in args] or sorted(_Args.data_dir.glob(_Args.pattern))
    if not paths:
        print(f"No files matched {_Args.data_dir}/{_Args.pattern}")
    else:
        main(paths)