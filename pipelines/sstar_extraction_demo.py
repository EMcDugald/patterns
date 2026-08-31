# pipelines/sstar_extraction_demo.py
"""
Head-to-head k-extraction on EXACT fields: isotropic uHu averaging
(sigma = pi/2) vs the S* prescription, both against the exact k.
No Taylor model anywhere: the true pointwise tensor of u = cos(theta)
is averaged by Gauss-Hermite quadrature with the actual kernel, the
dominant eigenvector is compared to the exact wavevector direction.

Knee (mu = 0.9): full (x, y) map over one artifact period x one flank
span; S* = the needle.  PN (single Pismen-Nepomnyashchy dislocation):
map over y >= 0.6 with per-point S* from the closed form; points with
W > 0.5 are masked (model invalid near the core; declared standoff).

Outputs: error maps (angle to exact khat, deg) and a summary table of
median / p95 by band.
Run: python sstar_extraction_demo.py [--outdir DIR]
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
from utils.uhu_symbolic import (KneeAnsatz, PNAnsatz, lambdify_np,
                                gh_average_matrix, sstar_closed_form,
                                sstar_knee_needle)

def wrapf(g, *extra):
    return lambda X, Y: np.broadcast_to(
        np.asarray(g(X, Y, *extra), float),
        np.broadcast(np.asarray(X), np.asarray(Y)).shape).copy()

def avg_tensor_grid(fns, X, Y, Smat, n_gh):
    """average the 3 tensor components at every point of (X, Y) with a
    constant kernel covariance Smat."""
    shp = X.shape
    comps = [gh_average_matrix(f, X.ravel(), Y.ravel(), Smat, n_gh)
             .reshape(shp) for f in fns]
    return comps  # Txx, Txy, Tyy

def dom_angle(Txx, Txy, Tyy):
    """angle of the most-negative-eigenvalue eigenvector, mod pi."""
    # eigen decomposition of [[a, b], [b, c]]
    a, b, c = Txx, Txy, Tyy
    lam = 0.5 * (a + c) - np.sqrt(0.25 * (a - c) ** 2 + b ** 2)  # min eig
    vx, vy = b, lam - a
    bad = np.hypot(vx, vy) < 1e-14
    vx = np.where(bad, 1.0, vx); vy = np.where(bad, 0.0, vy)
    return np.arctan2(vy, vx) % np.pi

def ang_err_deg(est, truth):
    d = np.abs((est - truth + np.pi / 2) % np.pi - np.pi / 2)
    return np.degrees(d)

def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    n_gh = 28
    report = []

    # ================= KNEE =================
    mu = 0.9; k1 = float(np.sqrt(1 - mu**2))
    A = KneeAnsatz()
    Tf = A.uhu_tensor("full")
    args = (A.x, A.y, A.k1, A.k2)
    fns = [wrapf(lambdify_np(args, Tf[i, j]), k1, mu)
           for (i, j) in ((0, 0), (0, 1), (1, 1))]
    thx = lambdify_np(args, A.thx); thy = lambdify_np(args, A.thy)
    xs = np.linspace(0, np.pi / k1, 36, endpoint=False)
    ys = np.linspace(-4, 4, 56)
    X, Y = np.meshgrid(xs, ys)
    truth = np.arctan2(thy(X, Y, k1, mu), thx(X, Y, k1, mu)) % np.pi

    S_iso = (np.pi / 2) ** 2 * np.eye(2)
    S_ndl = sstar_knee_needle(mu, tol=0.05, s_yy=0.15**2)
    maps = {}
    for name, S in (("iso", S_iso), ("sstar", S_ndl)):
        Ts = avg_tensor_grid(fns, X, Y, S, n_gh)
        maps[name] = ang_err_deg(dom_angle(*Ts), truth)
    core = np.abs(Y) < 0.5; flank = ~core
    for name in ("iso", "sstar"):
        m = maps[name]
        report.append(f"KNEE mu={mu} {name:5s}: core median "
                      f"{np.median(m[core]):7.3f} p95 {np.percentile(m[core],95):7.3f} | "
                      f"flank median {np.median(m[flank]):6.3f} p95 "
                      f"{np.percentile(m[flank],95):6.3f} deg")

    # ================= PN (single dislocation) =================
    B = PNAnsatz(numeric_weights=True)
    Tp = B.uhu_tensor("full")

    def parity_wrap(g, odd):
        """the single-PN formula lives on y > 0; kernels cross the wall,
        so extend by the measured parities: Txx, Tyy even, Txy odd."""
        def f(X, Y):
            Ya = np.clip(np.abs(Y), 2e-2, None)
            with np.errstate(all="ignore"):
                v = np.asarray(g(X, Ya), float)
            v = np.broadcast_to(v, np.broadcast(np.asarray(X),
                                                np.asarray(Y)).shape).copy()
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            return v * np.sign(Y) if odd else v
        return f

    fp = [parity_wrap(lambdify_np((B.x, B.y), Tp[i, j]), odd)
          for (i, j), odd in ((( 0, 0), False), ((0, 1), True),
                              ((1, 1), False))]
    kx = lambdify_np((B.x, B.y), B.thx); ky = lambdify_np((B.x, B.y), B.thy)
    hf = {nm: lambdify_np((B.x, B.y), e) for nm, e in
          (("H11", B.hess[0, 0]), ("H12", B.hess[0, 1]),
           ("H22", B.hess[1, 1]))}
    xs = np.linspace(-8, 8, 40); ys = np.linspace(0.6, 8, 32)
    X, Y = np.meshgrid(xs, ys)
    KX, KY = kx(X, Y), ky(X, Y)
    truth = np.arctan2(KY, KX) % np.pi
    Hh = {nm: f(X, Y) for nm, f in hf.items()}
    W = np.sqrt(Hh["H11"]**2 + 2*Hh["H12"]**2 + Hh["H22"]**2) / \
        np.maximum(KX**2 + KY**2, 1e-30)
    valid = W < 0.5

    # iso
    Ts = avg_tensor_grid(fp, X, Y, S_iso, n_gh)
    err_iso = ang_err_deg(dom_angle(*Ts), truth)
    # S*: per-point kernels -> loop (constant-S calls per point)
    st = sstar_closed_form(KX, KY, Hh["H11"], Hh["H12"], Hh["H22"],
                           tol=0.05, cap=(np.pi/2*2)**2)
    Sfield = st["S"]
    est = np.full_like(X, np.nan)
    for idx in np.ndindex(X.shape):
        if not valid[idx]:
            continue
        Sv = Sfield[idx]
        comps = [gh_average_matrix(f, np.array([X[idx]]), np.array([Y[idx]]),
                                   Sv, n_gh)[0] for f in fp]
        est[idx] = dom_angle(*[np.array([c]) for c in comps])[0]
    err_st = ang_err_deg(est, truth)
    for name, m in (("iso", err_iso), ("sstar", err_st)):
        v = m[valid & np.isfinite(m)]
        if v.size == 0:
            report.append(f"PN single      {name:5s}: no valid points")
            continue
        report.append(f"PN single      {name:5s}: valid-region median "
                      f"{np.median(v):7.3f} p95 {np.percentile(v,95):7.3f} deg"
                      f"   (mask: W < 0.5, y >= 0.6; parity-extended"
                      f" across the wall)")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2))
    for j, (name, ttl) in enumerate((("iso", r"isotropic $\sigma=\pi/2$"),
                                     ("sstar", r"$S^*$"))):
        im = axes[0, j].imshow(maps[name], origin="lower",
                               extent=[0, np.pi/k1, -4, 4], aspect="auto",
                               vmin=0, vmax=90, cmap="magma")
        axes[0, j].set_title(f"knee $\\mu=0.9$, {ttl}")
        fig.colorbar(im, ax=axes[0, j], label="angle err (deg)")
    for j, (m, ttl) in enumerate(((err_iso, r"isotropic $\sigma=\pi/2$"),
                                  (err_st, r"$S^*$ (per-point)"))):
        mm = np.where(valid, m, np.nan)
        im = axes[1, j].imshow(mm, origin="lower", extent=[-8, 8, 0.6, 8],
                               aspect="auto", vmin=0, vmax=20, cmap="magma")
        axes[1, j].set_title(f"PN dislocation, {ttl}")
        fig.colorbar(im, ax=axes[1, j], label="angle err (deg)")
    fig.tight_layout()
    path = os.path.join(outdir, "sstar_extraction_demo.png")
    fig.savefig(path, dpi=170); plt.close(fig)

    print("\n".join(report))
    with open(os.path.join(outdir, "sstar_extraction_demo.txt"), "w") as fh:
        fh.write("\n".join(report) + "\n")
    print("figure:", path)

class _Args:
    outdir = "out_extraction_demo"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=_Args.outdir)
    main(p.parse_args().outdir)
