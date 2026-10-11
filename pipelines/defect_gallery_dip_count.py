# pipelines/defect_gallery_dip_count.py
"""
Calibration of the plaquette count against amplitude dips on the saved
stage B runs (run_sh_defect_gallery.py .npz files).

A dislocation core is where the amplitude dips. The monogenic amplitude
sqrt(u^2 + |R|^2), smoothed at sigma_amp to remove the harmonic ripple,
is searched for local minima inside the ellipse that fall below
dip_ratio times the reference amplitude (the median inside the ellipse).
These dips are the width-free count of dislocation cores. Against them,
for each detection width, the script reports sum|q| / 2 (the pair count,
the harness convention: one +1/-1 pair per dislocation), the number of
dips that have at least one charged plaquette within match_radius, and
the number of charged plaquettes with no dip within match_radius
(orphans, either fake pairs or charges at disclination cores rather than
dislocations). The enclosing-circuit twist is repeated per width so the
invariant and the count can be read together.

Two dip thresholds are reported so the sensitivity of the dip count
itself is visible; the count is only as good as the dip detector, and
concave cores and PGB corrugation also modulate the amplitude.

Changelog
  v1  first version.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE))

import defect_gallery as dg                                             # noqa: E402
from utils.field_diagnostics_viz import monogenic, macro_iso            # noqa: E402


class _Args:
    npz_dir = _HERE / "out_sh_defect_gallery_1006"
    pattern = "sh_gallery_*_N384_L94_T1000.npz"
    sigma_dets = (0.3, 0.2, np.pi / 4, np.pi / 2)
    sigma_amp = 0.5            # smoothing of the monogenic amplitude before the dip search; dislocation cores read 0.55 to 0.62 of the median at this width, ripple minima above 0.85
    dip_window = 1.5           # local-minimum window, units of the pattern (about a quarter wavelength)
    dip_ratios = (0.7, 0.8)    # dip if A_min < ratio * median A inside the ellipse
    match_radius = 2.0
    beta = 0.45                # ellipse of the run
    count_frac = 0.7           # charges and dips counted inside count_frac * beta * L
    out_dir = _HERE / "out_sh_defect_gallery_1006"


def _label(sd):
    for num, name in ((np.pi / 2, "pi/2"), (np.pi / 4, "pi/4"), (np.pi / 8, "pi/8")):
        if abs(sd - num) < 1e-9:
            return name
    return f"{sd:g}"


def find_dips(u, x, L, a, ratio):
    mono = monogenic(u, L, L)
    A = macro_iso(mono["A"], a.sigma_amp, L, L)
    X, Y = np.meshgrid(x, x)
    disk = np.hypot(X, Y) < a.count_frac * a.beta * L
    ref = np.median(A[disk])
    dx = x[1] - x[0]
    win = max(3, int(round(a.dip_window / dx)) | 1)
    ismin = (A == minimum_filter(A, size=win)) & disk & (A < ratio * ref)
    j, i = np.nonzero(ismin)
    return [(float(X[jj, ii]), float(Y[jj, ii]), float(A[jj, ii] / ref)) for jj, ii in zip(j, i)], A, ref


def run(a=_Args):
    paths = sorted(a.npz_dir.glob(a.pattern))
    if not paths:
        print(f"no runs matched {a.npz_dir}/{a.pattern}"); return
    class Fa(dg._Args):
        sigma_dets = a.sigma_dets
    rows = []
    for p in paths:
        name = p.name.split("sh_gallery_")[1].split("_N")[0]
        d = np.load(p)
        x = d["xx"]; L = float(x[-1] - x[0] + (x[1] - x[0])); u = d["uu"][-1]
        disk_r = a.count_frac * a.beta * L
        dips = {r: find_dips(u, x, L, a, r)[0] for r in a.dip_ratios}
        r = dg.diagnose(u, x, L, Fa, [("all", (0.0, 0.0), disk_r)], ramp=False, count_disk=disk_r)
        print(f"\n{name}: dips (ratio {a.dip_ratios[0]}) = {len(dips[a.dip_ratios[0]])}, "
              f"dips (ratio {a.dip_ratios[1]}) = {len(dips[a.dip_ratios[1]])}")
        print(f"  {'sigma_det':>9s} {'sum|q|/2':>9s} {'net q':>6s} {'T all':>6s} "
              f"{'dips matched':>13s} {'orphan plaq':>12s}")
        dref = dips[a.dip_ratios[0]]
        for sd in a.sigma_dets:
            ch = r["charges"][sd]
            matched = sum(1 for (cx, cy, _) in dref if any(np.hypot(cx - px, cy - py) <= a.match_radius for px, py, _ in ch))
            orphan = sum(1 for (px, py, _) in ch if not any(np.hypot(cx - px, cy - py) <= a.match_radius for cx, cy, _ in dref))
            T = r["reads"][0]["T_circuit"][sd]
            print(f"  {_label(sd):>9s} {r['sum_abs_q'][sd] / 2:9.1f} {sum(c[2] for c in ch):+6d} {T:+6.0f} "
                  f"{matched:6d}/{len(dref):<6d} {orphan:12d}")
            rows.append((name, _label(sd), r["sum_abs_q"][sd] / 2, len(dref), matched, orphan, T))
        _figure(name, u, x, L, dips, r, a)
    return rows


def _figure(name, u, x, L, dips, r, a):
    A, _, ref = find_dips(u, x, L, a, a.dip_ratios[0])[1], None, None
    _, A, ref = find_dips(u, x, L, a, a.dip_ratios[0])
    ext = (x[0], x[-1], x[0], x[-1])
    n = len(a.sigma_dets)
    fig, axs = plt.subplots(1, n + 1, figsize=(3.4 * (n + 1), 3.4))
    ax = axs[0]
    ax.imshow(A / ref, cmap="viridis", origin="lower", extent=ext, vmin=0, vmax=1.5)
    for (cx, cy, _) in dips[a.dip_ratios[0]]:
        ax.plot(cx, cy, "x", color="white", ms=6, mew=1.2)
    for (cx, cy, _) in dips[a.dip_ratios[1]]:
        ax.plot(cx, cy, "+", color="white", ms=5, mew=0.8)
    ax.set_title(f"{name}: smoothed amplitude / median; x dips < {a.dip_ratios[0]}, + dips < {a.dip_ratios[1]}", fontsize=7)
    for k, sd in enumerate(a.sigma_dets):
        ax = axs[k + 1]
        ax.imshow(u, cmap="gray", origin="lower", extent=ext)
        for (cx, cy, _) in dips[a.dip_ratios[0]]:
            ax.plot(cx, cy, "x", color="yellow", ms=6, mew=1.2)
        for (px, py, qq) in r["charges"][sd]:
            ax.plot(px, py, "o" if qq > 0 else "s", mfc="none", mec="red" if qq > 0 else "deepskyblue", mew=1.2, ms=6)
        ax.set_title(f"sigma_det = {_label(sd)}: sum|q|/2 = {r['sum_abs_q'][sd] / 2:g}, dips = {len(dips[a.dip_ratios[0]])}", fontsize=7)
    zoom = 0.8 * a.count_frac * a.beta * L
    for ax in axs:
        ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom); ax.set_aspect("equal"); ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(a.out_dir / f"dip_count_{name}.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    args = sys.argv[1:]
    for i, tok in enumerate(args):
        if tok == "--pattern": _Args.pattern = args[i + 1]
        if tok == "--sigma_dets": _Args.sigma_dets = tuple(eval(v, {"pi": np.pi, "np": np}) for v in args[i + 1].split(","))
    run()
