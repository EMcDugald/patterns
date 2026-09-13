# pipelines/defect_gallery.py
"""
Canonical defect gallery, stage A: the harmonic (B = 1) point defects of
Passot and Newell 1994, Table 2, built analytically and read by the Riesz
diagnostics of Writeup 3. Nothing about the diagnostics is reinvented:
the director, the plaquette charge, and the loop persistence are imported
from the frozen method layer.

Fields (theta = Re of the antiderivative of w = f - i g, k = (f, g)):
  X  convex disclination     w = z^(-1/2)   theta = 2 c r^(1/2) cos(a/2)
  V  concave disclination    w = z^(1/2)    theta = (2/3) c r^(3/2) cos(3a/2)
  VV saddle (separated)      w = c (z^2 - s^2)^(1/2), theta by line integral
     (the coincident saddle w = z has a zero-wavenumber core that the
     director cannot read as stripes; see the changelog)
  XX target (regularized)                   theta = k0 sqrt(r^2 + rc^2)
  XX spiral (vortex with |k| near k0)       theta = k0 r - a
  VVXX dislocation           w = k0 + i/z   theta = k0 x - a
  VX handle                  w = ((z-a)/(z-b))^(1/2), theta by line integral
The constant c scales the harmonic fields so that |k| = 1 at r = r_unit.
The target is the Passot-Newell regularized target (circles), not log r,
because log r has an unresolvable core on any grid.

Diagnostics per field:
  q      plaquette charge of the doubled director angle at sigma_det
  P      loop persistence of the same angle
  T(C)   twist on a circuit C: (1/2) sum of wrapped increments of the
         doubled angle along C, in units of pi; also the enclosed sum of q
  Gamma  phase circulation on C from the monogenic phase with the sign
         fixed by the flip-corrected director along C, in units of 2 pi;
         reported only when the director returns to itself on C (even
         twist), undefined otherwise, which is the Passot-Newell statement;
         not attempted on circuits smaller than a wavelength across
  truth  the analytic circulation of k on C

Expected (Passot and Newell 1994, Table 2; twist in units of pi,
circulation in units of 2 pi): X +1; V -1; target +2, 0; spiral +2, -1;
saddle -2, 0; dislocation 0, -1 with +2 at the pole and -2 at the zero;
handle 0, 0 with +1 at the pole and -1 at the zero.

Table columns: r is the circuit radius; q(s) the sum of plaquette charges
enclosed by the circuit at detection width s; T(s) the twist read directly
along the circuit at width s, in units of pi; T exp the Passot-Newell
value; G est the circulation estimate from the signed monogenic phase, in
units of 2 pi; G true the analytic circulation of k; G exp the Table 2
value. Two detection widths are read: the production sigma_det = 0.3 and 0.2.
The harmonic dislocation's four-disclination core (vortex at the pole,
saddle at the zero, 1/k0 apart) is resolved at 0.2 and below and
annihilates completely at 0.3; sweeps at 0.05 and 0.1 grow spurious
pairs, so the window is narrow. Concave-type cores (V, saddle, the V of
the handle) have k -> 0, and the charge is placed where the pattern
has stripes, not at the zero of w; circuits enclosing the whole
low-wavenumber region read the exact twist.

Changelog
  v1  first version. Sign convention of q established empirically against
      the Passot-Newell twist: X reads +1, V reads -1 with the plaquette
      loop counterclockwise in (x, y). Coincident harmonic saddle rejected
      (core blob reads +2 with four compensating -1 on the diagonals at
      sigma |k| of order 0.3); separated saddle used instead. A first
      attempt with w = c (z - s)(z + s) read -2 at each root and -4 in
      all, correctly: a simple zero of w is a double V, so that field is
      two saddles. Per-field box and scaling so that |k| is near 1 where
      the circuits are read; regions with |k| below about 0.5 read as a
      bubble (+2) with a compensating ring of -1 at production width.
  v2  record fields added, not in the default gallery: saddle_coincident
      (w = c z) and X_small (X with r_unit = 2), run with
      --fields saddle_coincident,X_small. The dislocation width sweep is
      --fields dislocation --sigma_dets 0.05,0.1,0.15,0.2,0.3.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from utils.field_diagnostics_viz import riesz_director, monogenic    # noqa: E402
from op_extract.loop_persistence import (director_winding,           # noqa: E402
                                         loop_persistence_field)
try:                                                                   # noqa: E402
    from utils.geometry import build_rectangular_ramp_smooth
except ImportError:                                                    # older layout
    from op_extract.geometry import build_rectangular_ramp_smooth


# ---------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------

_DEFAULT_FIELDS = ("X", "V", "saddle", "target", "spiral", "dislocation", "handle")


class _Args:
    N = 512                 # grid points per side
    L = 40.0                # box side, centered at the origin
    r_unit = 8.0            # |k| = 1 at this radius for X, V, saddle
    k0 = 1.0                # carrier wavenumber for target, spiral, dislocation, handle
    rc = 1.0                # target core radius
    d_handle = 2.0          # handle: V at (-d, 0), X at (+d, 0)
    sigma_det = 0.3         # production detection width
    sigma_dir = 0.4         # director width for the circuit reads (sigma_small)
    half_sizes = (1, 2, 3, 4, 6, 8)
    ramp_margin = 0.06      # convention; ramp keeps the FFT clean on a non-periodic field
    ramp_scale = 60.0
    count_box = 0.35        # charges counted only for |x|, |y| < count_box * L
    n_circuit = 720         # samples per circuit
    out_dir = _HERE / "out_defect_gallery"
    fields = ("X", "V", "saddle", "target", "spiral", "dislocation", "handle")
    sigma_dets = (0.3, 0.2)   # production width first; 0.2 resolves the dislocation core
    s_saddle = 1.5            # separated saddle: roots of w at (+-s, 0)
    c_saddle = 0.9            # |k| = c |z^2 - s^2|^(1/2); 1 within 0.4 of each root


# per-field box, grid, and scaling; None inherits _Args
FIELD_CFG = {
    "X":           dict(L=40.0, N=1024, r_unit=8.0, zoom=9.0),
    "X_small":     dict(L=20.0, N=1024, r_unit=2.0, zoom=9.0),      # record: spurious pairs where |k| < 0.6
    "saddle_coincident": dict(L=40.0, N=1024, r_unit=8.0, zoom=9.0),  # record: +2 blob and -1 ring
    "V":           dict(L=40.0, N=1024, r_unit=1.5, zoom=9.0),
    "saddle":      dict(L=8.0, N=1024, zoom=3.5),
    "target":      dict(L=20.0, N=1024, zoom=6.0),
    "spiral":      dict(L=20.0, N=1024, zoom=6.0),
    "dislocation": dict(L=20.0, N=1024, zoom=4.0),
    "handle":      dict(L=20.0, N=1024, zoom=6.0),
}


# ---------------------------------------------------------------------
# analytic fields
# ---------------------------------------------------------------------

def _grid(N, L):
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y = np.meshgrid(x, x)
    return x, X, Y


def _polar(X, Y):
    return np.hypot(X, Y), np.arctan2(Y, X)


def _grad_alpha(X, Y):
    """Gradient of the polar angle; undefined (nan) at the origin node,
    which is never sampled by a circuit."""
    r2 = X**2 + Y**2
    with np.errstate(divide="ignore", invalid="ignore"):
        return -Y / r2, X / r2


def field_X(X, Y, a):
    r, al = _polar(X, Y)
    c = np.sqrt(a.r_unit)
    theta = 2.0 * c * np.sqrt(r) * np.cos(al / 2)
    # w = c z^(-1/2): k = (Re w, -Im w)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = c * (X + 1j * Y) ** (-0.5)           # infinite at the origin node
    return theta, w.real, -w.imag, [(0.0, 0.0)]


def field_V(X, Y, a):
    r, al = _polar(X, Y)
    c = 1.0 / np.sqrt(a.r_unit)
    theta = (2.0 / 3.0) * c * r**1.5 * np.cos(1.5 * al)
    w = c * (X + 1j * Y) ** 0.5
    return theta, w.real, -w.imag, [(0.0, 0.0)]


def field_saddle(X, Y, a):
    """Separated saddle of Passot and Newell, w = c (z^2 - s^2)^(1/2): one
    concave disclination at each root (+-s, 0), the cut on the segment
    between them. Written as sqrt(z - s) sqrt(z + s) so that the only cut
    is that segment. A simple zero w = c (z - s) would be a coincident
    double V (twist -2 pi at one point), which is Alan's VV = (z - a)."""
    s = a.s_saddle
    c = a.c_saddle
    z = X + 1j * Y
    w = c * np.sqrt(z - s) * np.sqrt(z + s)
    theta, f, g = _theta_from_w(w, X, Y)
    return theta, f, g, [(s, 0.0), (-s, 0.0)]


def field_saddle_coincident(X, Y, a):
    """Coincident harmonic saddle w = c z (Alan's VV = (z - a)), theta =
    c (x^2 - y^2) / 2, |k| = c r = 1 at r = r_unit. Kept as the record
    behind the vanishing-wavenumber rule: its core reads as a +2 blob
    with a compensating ring of -1 cells."""
    c = 1.0 / a.r_unit
    theta = 0.5 * c * (X**2 - Y**2)
    return theta, c * X, -c * Y, [(0.0, 0.0)]


def field_target(X, Y, a):
    r = np.hypot(X, Y)
    rr = np.sqrt(r**2 + a.rc**2)
    theta = a.k0 * rr
    return theta, a.k0 * X / rr, a.k0 * Y / rr, [(0.0, 0.0)]


def field_spiral(X, Y, a):
    r, al = _polar(X, Y)
    theta = a.k0 * r - al
    gx, gy = _grad_alpha(X, Y)
    with np.errstate(divide="ignore", invalid="ignore"):
        return theta, a.k0 * X / r - gx, a.k0 * Y / r - gy, [(0.0, 0.0)]


def field_dislocation(X, Y, a):
    r, al = _polar(X, Y)
    theta = a.k0 * X - al
    gx, gy = _grad_alpha(X, Y)
    # w = k0 + i/z: pole (vortex) at 0, zero (saddle) at z = -i/k0
    return theta, a.k0 - gx, -gy, [(0.0, 0.0), (0.0, -1.0 / a.k0)]


def _theta_from_w(w, X, Y):
    """theta = Re of the antiderivative of w, by line integration, for a w
    whose only cut is a segment of the real axis. Paths never cross the
    cut: columns integrated upward in the lower half and downward in the
    upper half, the two halves tied together along the right edge."""
    f, g = w.real, -w.imag
    x = X[0]
    y = Y[:, 0]
    N = len(x)
    jl = np.searchsorted(y, 0.0)          # first row with y >= 0
    theta = np.zeros_like(X)
    # lower half: along the bottom row, then up each column
    bottom = cumulative_trapezoid(f[0], x, initial=0.0)
    for i in range(N):
        theta[:jl, i] = bottom[i] + cumulative_trapezoid(g[:jl, i], y[:jl], initial=0.0)
    # tie: up the right edge from the bottom to the top row
    right = cumulative_trapezoid(g[:, -1], y, initial=0.0) + bottom[-1]
    # upper half: along the top row leftward, then down each column
    top = right[-1] + cumulative_trapezoid(f[-1], x, initial=0.0) - cumulative_trapezoid(f[-1], x, initial=0.0)[-1]
    for i in range(N):
        seg = cumulative_trapezoid(g[::-1][: N - jl, i], y[::-1][: N - jl], initial=0.0)
        theta[jl:, i] = (top[i] + seg)[::-1]
    return theta, f, g


def field_handle(X, Y, a):
    """w = ((z-a)/(z-b))^(1/2), V at a = -d, X at b = +d, principal branch
    with the cut on the segment [a, b], which is a phase contour."""
    d = a.d_handle
    z = X + 1j * Y
    w = np.sqrt((z + d) / (z - d))
    theta, f, g = _theta_from_w(w, X, Y)
    return theta, f, g, [(-d, 0.0), (d, 0.0)]


FIELDS = {"X": field_X, "X_small": field_X, "V": field_V, "saddle": field_saddle,
          "saddle_coincident": field_saddle_coincident,
          "target": field_target, "spiral": field_spiral,
          "dislocation": field_dislocation, "handle": field_handle}

EXPECTED = {   # twist / pi, circulation / 2 pi (None = undefined), per circuit
    "X": [("core", 0, +1, None)],
    "X_small": [("core", 0, +1, None), ("far", 0, +1, None)],
    "V": [("core", 0, -1, None)],
    "saddle_coincident": [("core", 0, -2, 0), ("ring", 0, -2, 0)],
    "saddle": [("V1", 0, -1, None), ("V2", 1, -1, None), ("both", -1, -2, 0)],
    "target": [("core", 0, +2, 0)],
    "spiral": [("core", 0, +2, -1)],
    "dislocation": [("pole", 0, +2, None), ("zero", 1, -2, None), ("both", -1, 0, -1)],
    "handle": [("V", 0, -1, None), ("X", 1, +1, None), ("both", -1, 0, 0)],
}

CIRCUITS = {   # (name, center index or -1 for the midpoint of all, radius)
    "X": [("core", 0, 3.0)],
    "X_small": [("core", 0, 1.5), ("far", 0, 8.0)],
    "V": [("core", 0, 3.0)],
    "saddle_coincident": [("core", 0, 3.0), ("ring", 0, 9.0)],
    "saddle": [("V1", 0, 0.8), ("V2", 1, 0.8), ("both", -1, 3.0)],
    "target": [("core", 0, 3.0)],
    "spiral": [("core", 0, 3.0)],
    "dislocation": [("pole", (0.0, 0.05), 0.45), ("zero", (0.0, -0.65), 0.45), ("both", -1, 3.0)],
    "handle": [("V", (-0.8, 0.0), 1.6), ("X", (2.0, 0.0), 1.5), ("both", -1, 4.5)],
}


# ---------------------------------------------------------------------
# circuit reads
# ---------------------------------------------------------------------

def width_label(sd):
    """Print a detection width as a fraction of pi when it is one."""
    for num, name in ((np.pi / 2, "pi/2"), (np.pi / 4, "pi/4"), (np.pi / 8, "pi/8"), (np.pi, "pi")):
        if abs(sd - num) < 1e-9:
            return name
    return f"{sd:g}"


def _wrap(d, period):
    return (d + period / 2) % period - period / 2


def _sample(field, x, cx, cy, radius, n):
    """Nearest-node samples of a node field along a circle."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    px = cx + radius * np.cos(t)
    py = cy + radius * np.sin(t)
    dx = x[1] - x[0]
    i = np.clip(np.round((px - x[0]) / dx).astype(int), 0, len(x) - 1)
    j = np.clip(np.round((py - x[0]) / dx).astype(int), 0, len(x) - 1)
    return field[j, i], px, py, t


def circuit_twist(ang, x, cx, cy, radius, n):
    """Twist on the circuit in units of pi: winding of the doubled angle
    over 2 pi, read directly from the samples."""
    a2 = 2.0 * _sample(ang, x, cx, cy, radius, n)[0]
    return float(np.round(np.sum(_wrap(np.diff(np.r_[a2, a2[0]]), 2 * np.pi)) / (2 * np.pi)))


def enclosed_charge(q, xc, cx, cy, radius):
    Xc, Yc = np.meshgrid(xc, xc)
    inside = (Xc - cx)**2 + (Yc - cy)**2 < radius**2
    return int(q[inside].sum())


def circuit_circulation(mono, ang, x, cx, cy, radius, n):
    """Phase circulation in units of 2 pi. The director angle is
    flip-corrected along the circuit (unwrap the doubled angle, halve),
    which makes d a continuous unit vector field on C if and only if the
    twist is even. The local phase is psi = atan2(R . d, u), the signed
    monogenic phase, and Gamma = sum of wrapped increments of psi. Returns
    None when d does not return to itself (odd twist)."""
    a2 = 2.0 * _sample(ang, x, cx, cy, radius, n)[0]
    a2u = np.concatenate([[a2[0]], a2[0] + np.cumsum(_wrap(np.diff(a2), 2 * np.pi))])
    closing = _wrap(a2[0] - a2u[-1], 2 * np.pi)      # last -> first step
    total = a2u[-1] + closing - a2u[0]
    if int(np.round(total / (2 * np.pi))) % 2 != 0:
        return None
    al = 0.5 * a2u
    d1, d2 = np.cos(al), np.sin(al)
    u = _sample(mono["u"], x, cx, cy, radius, n)[0]
    R1 = _sample(mono["R1"], x, cx, cy, radius, n)[0]
    R2 = _sample(mono["R2"], x, cx, cy, radius, n)[0]
    psi = np.arctan2(R1 * d1 + R2 * d2, u)
    return float(np.sum(_wrap(np.diff(np.r_[psi, psi[0]]), 2 * np.pi)) / (2 * np.pi))


def true_circulation(f, g, x, cx, cy, radius, n):
    fs, px, py, t = _sample(f, x, cx, cy, radius, n)
    gs = _sample(g, x, cx, cy, radius, n)[0]
    tx, ty = -np.sin(t), np.cos(t)
    ds = 2 * np.pi * radius / n
    return float(np.sum(fs * tx + gs * ty) * ds / (2 * np.pi))


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def _cfg(name, base):
    class A(base):
        pass
    for k, v in FIELD_CFG.get(name, {}).items():
        setattr(A, k, v)
    return A


def resolve_circuits(circuits, cores):
    """Circuit spec -> list of (name, (cx, cy), radius). A center may be an
    explicit (x, y), a core index, or -1 for the mean of all cores."""
    out = []
    for cname, where, radius in circuits:
        if isinstance(where, tuple):
            cx, cy = where
        elif where >= 0:
            cx, cy = cores[where]
        else:
            cx = float(np.mean([c[0] for c in cores])); cy = float(np.mean([c[1] for c in cores]))
        out.append((cname, (float(cx), float(cy)), float(radius)))
    return out


def cluster_charges(charges, link=2.0):
    """Greedy clustering of (x, y, q) plaquette charges within `link` of
    each other; returns [(cx, cy, sum q, n)] for automatic circuits."""
    pts = [(cx, cy, qq) for cx, cy, qq in charges]
    clusters = []
    for cx, cy, qq in pts:
        for cl in clusters:
            if any(np.hypot(cx - px, cy - py) <= link for px, py, _ in cl):
                cl.append((cx, cy, qq)); break
        else:
            clusters.append([(cx, cy, qq)])
    return [(float(np.mean([p[0] for p in cl])), float(np.mean([p[1] for p in cl])),
             int(sum(p[2] for p in cl)), len(cl)) for cl in clusters]


def diagnose(u, x, L, a, circuits, f=None, g=None, expected=None, ramp=True, count_disk=None):
    """Run the diagnostics on one pattern u on a square grid x (side L).
    circuits: list of (name, (cx, cy), radius). f, g: analytic k for the
    true circulation, or None. expected: list of (T_exp, G_exp) or None."""
    ur = u * build_rectangular_ramp_smooth(x, x, a.ramp_margin, a.ramp_margin, a.ramp_scale) if ramp else u
    angs = {sd: riesz_director(ur, L, L, sd)["angle"] for sd in a.sigma_dets}
    qs = {sd: director_winding(ang) for sd, ang in angs.items()}
    P, _ = loop_persistence_field(angs[a.sigma_dets[0]], a.half_sizes)
    dirr = riesz_director(ur, L, L, a.sigma_dir)
    mono = monogenic(ur, L, L)
    mono["u"] = ur
    xc = 0.5 * (x[:-1] + x[1:])
    Xc, Yc = np.meshgrid(xc, xc)
    if count_disk is None:
        box = (np.abs(Xc) < a.count_box * L) & (np.abs(Yc) < a.count_box * L)
    else:
        box = np.hypot(Xc, Yc) < count_disk
    charges = {sd: [(round(float(Xc[j, i]), 2), round(float(Yc[j, i]), 2), int(qq[j, i]))
                    for j, i in zip(*np.nonzero((qq != 0) & box))] for sd, qq in qs.items()}
    reads = []
    for n, (cname, (cx, cy), radius) in enumerate(circuits):
        T_exp, G_exp = expected[n] if expected else (None, None)
        reads.append(dict(
            circuit=cname, center=(cx, cy), radius=radius,
            T_expected=T_exp, G_expected=G_exp,
            q_enclosed={sd: enclosed_charge(qq, xc, cx, cy, radius) for sd, qq in qs.items()},
            T_circuit={sd: circuit_twist(ang, x, cx, cy, radius, a.n_circuit) for sd, ang in angs.items()},
            G_estimate=(circuit_circulation(mono, dirr["angle"], x, cx, cy, radius, a.n_circuit)
                        if radius >= 2.0 else "n/a"),
            G_true=(true_circulation(f, g, x, cx, cy, radius, a.n_circuit) if f is not None else None),
        ))
    return dict(x=x, u=ur, A_mono=mono["A"], q=qs[a.sigma_dets[0]], qs=qs, P=P, xc=xc, charges=charges,
                reads=reads, sum_abs_q={sd: int(np.abs(qq[box]).sum()) for sd, qq in qs.items()},
                align=dirr["alignment"])


def analyze(name, base):
    a = _cfg(name, base)
    x, X, Y = _grid(a.N, a.L)
    theta, f, g, cores = FIELDS[name](X, Y, a)
    circuits = resolve_circuits(CIRCUITS[name], cores)
    expected = [(T, G) for _, _, T, G in EXPECTED[name]]
    r = diagnose(np.cos(theta), x, a.L, a, circuits, f, g, expected)
    r.update(name=name, theta=theta, cores=cores, cfg=a)
    return r


def print_table(results, a):
    sds = a.sigma_dets
    hdr = " ".join(f"{'q(' + width_label(sd) + ')':>7s} {'T(' + width_label(sd) + ')':>7s}" for sd in sds)
    print(f"{'field':12s} {'circuit':8s} {'r':>5s} {hdr} {'T exp':>6s} "
          f"{'G est':>7s} {'G true':>7s} {'G exp':>6s}   (T in pi, G in 2 pi)")
    for r in results:
        for rd in r["reads"]:
            ge = ("undef" if rd["G_estimate"] is None else rd["G_estimate"]
                  if isinstance(rd["G_estimate"], str) else f"{rd['G_estimate']:+.3f}")
            gx = ("   n/a" if rd["T_expected"] is None else
                  "undef" if rd["G_expected"] is None else f"{rd['G_expected']:+d}")
            cols = " ".join(f"{rd['q_enclosed'][sd]:+7d} {rd['T_circuit'][sd]:+7.0f}" for sd in sds)
            tx = "   n/a" if rd["T_expected"] is None else f"{rd['T_expected']:+6d}"
            gt = "    n/a" if rd["G_true"] is None else f"{rd['G_true']:+7.3f}"
            print(f"{r['name']:12s} {rd['circuit']:8s} {rd['radius']:5.2f} {cols} "
                  f"{tx} {ge:>7s} {gt} {gx:>6s}")
        for sd in sds:
            print(f"{'':12s} charges at sigma_det = {width_label(sd)}: {r['charges'][sd]}")


def _charge_panel(ax, charges, zq, title):
    """Charges as large markers with the count in the corner, so an empty
    panel is unambiguous."""
    for (cx, cy, qq) in charges:
        ax.plot(cx, cy, "o" if qq > 0 else "s", mfc="none",
                mec="red" if qq > 0 else "blue", mew=1.6, ms=9)
    npos = sum(1 for c in charges if c[2] > 0); nneg = sum(1 for c in charges if c[2] < 0)
    ax.text(0.02, 0.97, f"{npos} plaquettes +1, {nneg} plaquettes -1\nnet {npos - nneg:+d}"
            if charges else "no charged plaquettes", transform=ax.transAxes, va="top",
            fontsize=7, bbox=dict(fc="white", ec="0.7", alpha=0.9))
    ax.set_xlim(-zq, zq); ax.set_ylim(-zq, zq)
    ax.set_title(title, fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.5)


def gallery_figure(results, a, path, pattern_key="u", extra_cols=None):
    n = len(results)
    sds = a.sigma_dets
    ncol = 4 + (len(extra_cols) if extra_cols else 0)
    fig, axs = plt.subplots(n, ncol, figsize=(3.4 * ncol, 3.3 * n), squeeze=False)
    for row, r in enumerate(results):
        zoom = r["cfg"].zoom
        cores = r.get("cores") or [rd["center"] for rd in r["reads"]]
        zq = max(zoom / 3, max(abs(c[0]) for c in cores) + 1.0, max(abs(c[1]) for c in cores) + 1.0)
        x = r["x"]; ext = (x[0], x[-1], x[0], x[-1])
        col = 0
        if extra_cols:
            for key, ttl in extra_cols:
                ax = axs[row, col]; col += 1
                cmap = "gray" if key != "A_mono" else "viridis"
                ax.imshow(r[key], cmap=cmap, origin="lower", extent=ext,
                          vmin=0 if key == "A_mono" else None)
                ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom)
                ax.set_title(f"{r['name']}: {ttl}", fontsize=8)
        ax = axs[row, col]; col += 1
        ax.imshow(r[pattern_key], cmap="gray", origin="lower", extent=ext)
        for (cx, cy, qq) in r["charges"][sds[0]]:
            ax.plot(cx, cy, "o" if qq > 0 else "s", mfc="none",
                    mec="red" if qq > 0 else "deepskyblue", mew=1.3, ms=7)
        for rd in r["reads"]:
            t = np.linspace(0, 2 * np.pi, 200)
            ax.plot(rd["center"][0] + rd["radius"] * np.cos(t),
                    rd["center"][1] + rd["radius"] * np.sin(t), "-", color="orange", lw=0.9)
            T = rd["T_circuit"][sds[0]]
            ax.annotate(f"{rd['circuit']}: T={T:+.0f}pi", (rd["center"][0], rd["center"][1] + rd["radius"]),
                        fontsize=6, color="orange", ha="center", va="bottom")
        ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom)
        ax.set_title(f"{r['name']}: pattern, charges at sigma_det={width_label(sds[0])} (o +1, square -1), circuits",
                     fontsize=7)
        for sd in sds[:2]:
            _charge_panel(axs[row, col], r["charges"][sd], zq, f"plaquette charges, sigma_det = {width_label(sd)}")
            col += 1
        ax = axs[row, col]
        ax.imshow(r["P"], cmap="magma", vmin=0, vmax=len(a.half_sizes), origin="lower",
                  extent=(r["xc"][0], r["xc"][-1], r["xc"][0], r["xc"][-1]))
        ax.set_xlim(-zq, zq); ax.set_ylim(-zq, zq)
        ax.set_title(f"loop persistence P, sigma_det = {width_label(sds[0])}", fontsize=8)
        for ax in axs[row]:
            ax.set_aspect("equal"); ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(a=_Args):
    a.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"sdet{'-'.join(width_label(sd).replace('/', 'o') for sd in a.sigma_dets)}_sdir{a.sigma_dir:g}"
    if tuple(a.fields) != _DEFAULT_FIELDS:
        stamp += "_" + "-".join(a.fields)      # record runs never overwrite the default gallery
    results = [analyze(name, a) for name in a.fields]
    print_table(results, a)
    gallery_figure(results, a, a.out_dir / f"defect_gallery_{stamp}.png")
    def _j(v):
        if isinstance(v, tuple): return list(v)
        if isinstance(v, dict): return {str(k): _j(w) for k, w in v.items()}
        return v
    rec = {r["name"]: dict(N=r["cfg"].N, L=r["cfg"].L, charges=_j(r["charges"]),
                          sum_abs_q=_j(r["sum_abs_q"]),
                          reads=[{k: _j(v) for k, v in rd.items()} for rd in r["reads"]])
           for r in results}
    with open(a.out_dir / f"defect_gallery_{stamp}.json", "w") as fh:
        json.dump(rec, fh, indent=1)
    print("wrote", a.out_dir / f"defect_gallery_{stamp}.png")
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    for i, tok in enumerate(args):
        if tok == "--N": _Args.N = int(args[i + 1])
        if tok == "--L": _Args.L = float(args[i + 1])
        if tok == "--sigma_dets":   # e.g. --sigma_dets 0.3,0.2,pi/4
            _Args.sigma_dets = tuple(eval(v, {"pi": np.pi}) for v in args[i + 1].split(","))
        if tok == "--fields": _Args.fields = tuple(args[i + 1].split(","))
    main()
