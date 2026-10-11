# pipelines/run_sh_defect_gallery.py
"""
Canonical defect gallery, stage B: the stage A fields used as small
amplitude initial conditions for Swift-Hohenberg, relaxed under a compact
elliptical ramp of R (R = Rscale inside the ellipse, negative outside, the
boundary treatment of the diamond and ellipse runs), and read with the
same diagnostics as stage A (defect_gallery.diagnose). Two log-sum-exp
constructions with |k| = 1 by design are added, ported from the old
sh_sims modules: the reflected knee (a saddle with PGBs on the diagonals,
the Passot-Newell regularized saddle) and the four-direction log-sum-exp
(a square target with four PGBs), both at the production angle mu = 0.85,
and the Passot-Newell Roman arch (semicircles joined to straight rolls,
|k| = k0 everywhere) as the convex disclination with a pattern-like
initial condition. The harmonic X drifts toward the ramp under
relaxation because its far-field wavenumber is below k0.

Circuits on the relaxed fields are placed automatically: the charged
plaquettes at the production width are clustered (link distance 2), one
circuit of radius r_cluster around each cluster, one enclosing circuit
about the origin. There is no analytic k, so G true is not available;
the expected twist is that of the initial condition's topology.

Integrator: sh_sims.core.integrate_sh (ETDRK4, 2/3 dealiasing), the same
as the zigzag and diamond runs.

Changelog
  v1  first version.
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE))

from sh_sims.core import integrate_sh                                   # noqa: E402
import defect_gallery as dg                                             # noqa: E402


class _Args:
    N = 384
    L = 30.0 * np.pi
    dt = 0.5
    tmax = 1000.0
    nsave = 4
    Rscale = 0.5
    beta = 0.45             # ellipse ramp: R > 0 inside x^2/(beta L)^2 + y^2/(beta L)^2 < 1
    amplitude = 0.1
    mu_lse = 0.85           # log-sum-exp constructions at the production angle (k1 = mu)
    r_cluster = 3.0
    link = 2.0
    count_frac = 0.7        # charges counted inside a disk of radius count_frac * beta * L (rim ripple excluded)
    reuse = True            # reuse a saved run if its npz exists
    out_dir = _HERE / "out_sh_defect_gallery_1006"
    fields = ("X", "roman_arch", "V", "saddle", "saddle_lse", "target_lse", "target", "spiral", "dislocation", "handle")
    # stage A scalings reused where they apply; |k| = 1 at r_unit for X and V
    fa = dict(r_unit=20.0, s_saddle=6.0, c_saddle=0.05, rc=1.0, k0=1.0, d_handle=5.0,
              sigma_dets=(0.3, 0.2), sigma_dir=0.4, half_sizes=(1, 2, 3, 4, 6, 8),
              ramp_margin=0.06, ramp_scale=60.0, count_box=0.35, n_circuit=720, zoom=20.0)


def _ellipse_ramp(X, Y, L, beta, Rscale):
    """Ported from sh_sims.saddle_logsum2_compact._ellipse_ramp."""
    return Rscale * np.tanh(3.0 * np.sqrt(2.0) * L * (beta - np.sqrt((X / L)**2 + (Y / L)**2)) / 2.0)


def _theta_knee(X, Y, mu):
    k1, k2 = mu, np.sqrt(1.0 - mu**2)
    return np.log(np.exp(k1 * X + k2 * Y) + np.exp(k1 * X - k2 * Y))


def theta_saddle_lse(X, Y, a):
    """Reflected knee (sh_sims.saddle_logsum2): a saddle with PGBs."""
    return np.where(X >= 0.0, _theta_knee(-X, Y, a.mu_lse), _theta_knee(X, Y, a.mu_lse))


def theta_roman_arch(X, Y, a):
    """Regularized convex disclination of Passot and Newell: semicircular
    rolls for x < 0 joined C^1 to straight rolls for x > 0, |k| = k0
    everywhere. u = cos(theta) is smooth because cos is even in y."""
    r = np.hypot(X, Y)
    return a.fa["k0"] * np.where(X < 0.0, r, np.abs(Y))


def theta_target_lse(X, Y, a):
    """Four-direction log-sum-exp (sh_sims.target): square target."""
    k1, k2 = a.mu_lse, np.sqrt(1.0 - a.mu_lse**2)
    return np.log(np.exp(k1 * X + k2 * Y) + np.exp(k1 * X - k2 * Y)
                  + np.exp(-k1 * X + k2 * Y) + np.exp(-k1 * X - k2 * Y))


# enclosing-circuit radius, where the automatic choice would cross a
# dislocation chain at a core; the value crosses the vertical fold
# between chain sites (site spacing about 5.5 at mu = 0.85)
ALL_RADIUS = {"target_lse": 12.0, "saddle_lse": 12.0}

EXPECTED_T = {"X": +1, "roman_arch": +1, "V": -1, "saddle": -2, "saddle_lse": -2, "target": +2, "target_lse": +2,
              "spiral": +2, "dislocation": 0, "handle": 0}


def build_ic(name, X, Y, fa):
    class A(dg._Args):
        pass
    for k, v in fa.items():
        setattr(A, k, v)
    if name == "roman_arch":
        return theta_roman_arch(X, Y, _Args), A
    if name == "saddle_lse":
        return theta_saddle_lse(X, Y, _Args), A
    if name == "target_lse":
        return theta_target_lse(X, Y, _Args), A
    theta = dg.FIELDS[name](X, Y, A)[0]
    return theta, A


def run_field(name, a):
    x = (a.L / a.N) * np.arange(-a.N / 2, a.N / 2)
    X, Y = np.meshgrid(x, x)
    theta0, fa = build_ic(name, X, Y, a.fa)
    R = _ellipse_ramp(X, Y, a.L, a.beta, a.Rscale)
    u0 = a.amplitude * np.cos(theta0)
    tag = f"_mu{a.mu_lse:.2f}" if name.endswith("_lse") else ""
    npz = a.out_dir / f"sh_gallery_{name}{tag}_N{a.N}_L{a.L:.0f}_T{a.tmax:g}.npz"
    if a.reuse and npz.exists():
        d = np.load(npz); tt = d["tt"]; uu = np.transpose(d["uu"], (1, 2, 0))
        print(f"{name}: reusing {npz.name}", flush=True)
    else:
        print(f"{name}: integrating SH, N={a.N}, L={a.L:.1f}, tmax={a.tmax}, dt={a.dt} ...", flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            tt, uu, _, *_ = integrate_sh(u0, R, a.L, a.L, a.dt, a.tmax, a.nsave, energy=False)
    uT = uu[..., -1]
    disk = a.count_frac * a.beta * a.L
    # circuits from the relaxed field's own charges
    probe = dg.diagnose(uT, x, a.L, fa, [], ramp=False, count_disk=disk)
    clusters = dg.cluster_charges(probe["charges"][fa.sigma_dets[0]], a.link)
    circuits = [(f"c{i + 1}", (cx, cy), a.r_cluster) for i, (cx, cy, _, _) in enumerate(clusters)]
    rmax = max([np.hypot(cx, cy) for cx, cy, _, _ in clusters] + [3.0])
    circuits.append(("all", (0.0, 0.0), ALL_RADIUS.get(name, rmax + 4.0)))
    expected = [(None, None)] * len(clusters) + [(EXPECTED_T[name], None)]
    r = dg.diagnose(uT, x, a.L, fa, circuits, expected=expected, ramp=False, count_disk=disk)
    r.update(name=name, cfg=fa, u_ic=u0 / a.amplitude, cores=[(cx, cy) for cx, cy, _, _ in clusters]
             or [(0.0, 0.0)], clusters=clusters, tt=tt)
    if not npz.exists():
        np.savez_compressed(npz, tt=tt, xx=x, yy=x, uu=np.transpose(uu, (2, 0, 1)), R=R, theta0=theta0)
    return r


def main(a=_Args):
    a.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = (f"sdet{'-'.join(dg.width_label(sd).replace('/', 'o') for sd in a.fa['sigma_dets'])}"
             f"_N{a.N}_L{a.L:.0f}_T{a.tmax:g}_R{a.Rscale:g}")
    results = [run_field(name, a) for name in a.fields]
    class _T:
        sigma_dets = a.fa["sigma_dets"]; half_sizes = a.fa["half_sizes"]
    print()
    dg.print_table(results, _T)
    for r in results:
        print(f"{r['name']:12s} clusters (x, y, net q, n plaquettes) at sigma_det="
              f"{a.fa['sigma_dets'][0]}: {r['clusters']}")
    dg.gallery_figure(results, _T, a.out_dir / f"sh_defect_gallery_{stamp}.png",
                      extra_cols=[("u_ic", "initial condition"), ("A_mono", "monogenic amplitude, relaxed")])
    rec = {r["name"]: dict(clusters=r["clusters"], charges={str(k): v for k, v in r["charges"].items()},
                          reads=[{k: (list(v) if isinstance(v, tuple) else
                                      {str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v)
                                  for k, v in rd.items()} for rd in r["reads"]])
           for r in results}
    with open(a.out_dir / f"sh_defect_gallery_{stamp}.json", "w") as fh:
        json.dump(rec, fh, indent=1)
    print("wrote", a.out_dir / f"sh_defect_gallery_{stamp}.png")


if __name__ == "__main__":
    args = sys.argv[1:]
    for i, tok in enumerate(args):
        if tok == "--N": _Args.N = int(args[i + 1])
        if tok == "--tmax": _Args.tmax = float(args[i + 1])
        if tok == "--fields": _Args.fields = tuple(args[i + 1].split(","))
        if tok == "--sigma_dets":   # e.g. --sigma_dets 0.3,pi/4
            _Args.fa["sigma_dets"] = tuple(eval(v, {"pi": np.pi}) for v in args[i + 1].split(","))
    main()
