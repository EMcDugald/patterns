# pipelines/nucleation_measures_validation.py
"""
v2 bench. Adds the unrounded director circulation as a continuous measure and
sweeps mu so the sharp-knee behaviour is visible (the knee corrugation is worst
as mu -> 1, which is exactly where a continuous, unrounded measure is at risk).

Fields (core at y = 0):
  straight : single-orientation rolls   (trivial null)
  knee     : smooth zigzag fold         (hard null; sharper as mu grows)
  pn       : Pismen dislocation row     (positive control)

Measures, all read in a band |y| <= band_y about the core:
  trusted count      round(circulation), sum|q|/2      (quantized baseline)
  loop-persistence   fixed sig=0.4, sweep loop size    (winner from v1)
  unrounded max      max |circulation| before rounding (new; near 1 at a core)
  unrounded sum      sum |circulation| before rounding (new; total frac charge)
  unrounded smooth   Gaussian(unrounded), band sum     (new; coarse density)

The question v2 answers: does each measure stay knee-clean (knee << pn) as the
knee sharpens across mu = 0.85, 0.90, 0.95?
"""

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from op_extract.riesz import riesz_analyze_pattern
from op_extract.pn_field import pismen_on_sh_grid


# ---------------------------------------------------------------------------
# grid + fields
# ---------------------------------------------------------------------------

def make_grid(mu, n_xper=4, n_yper=6, ppw=28):
    k1 = np.sqrt(1.0 - mu**2); k2 = mu
    dx = 2.0 * np.pi / ppw
    Lx = n_xper * (2 * np.pi / max(k1, 0.2))
    Ly = n_yper * (2 * np.pi / k2)
    Nx = int(round(Lx / dx)); Ny = int(round(Ly / dx))
    x = (np.arange(Nx) - Nx // 2) * dx
    y = (np.arange(Ny) - Ny // 2) * dx
    X, Y = np.meshgrid(x, y)
    return dict(x=x, y=y, X=X, Y=Y, dx=dx, dy=dx, k1=k1, k2=k2, mu=mu)


def field_straight(g):
    return np.cos(g["k1"] * g["X"] + g["k2"] * g["Y"])


def field_knee(g):
    pp = g["k1"] * g["X"] + g["k2"] * g["Y"]
    pm = g["k1"] * g["X"] - g["k2"] * g["Y"]
    m = np.maximum(pp, pm)
    theta = m + np.log(np.exp(pp - m) + np.exp(pm - m))
    return np.cos(theta)


def field_pn(g, n_def=4):
    Lx = g["x"][-1] - g["x"][0]
    xd = np.linspace(-0.3 * Lx, 0.3 * Lx, n_def)
    _, u = pismen_on_sh_grid(g["X"], g["Y"], xd, amp=1.0)
    return u


# ---------------------------------------------------------------------------
# director + circulation
# ---------------------------------------------------------------------------

def _wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def doubled_angle(u, dx, sigma):
    rz = riesz_analyze_pattern(u, dx, dx)
    b = rz["orientation"]
    c2 = gaussian_filter(np.cos(2 * b), sigma / dx, mode="nearest")
    s2 = gaussian_filter(np.sin(2 * b), sigma / dx, mode="nearest")
    return np.arctan2(s2, c2)


def circulation(ang2):
    """Unrounded plaquette circulation of angle field ang2, in winding units."""
    a = ang2
    w = (_wrap(a[:-1, 1:] - a[:-1, :-1]) + _wrap(a[1:, 1:] - a[:-1, 1:])
         + _wrap(a[1:, :-1] - a[1:, 1:]) + _wrap(a[:-1, :-1] - a[1:, :-1]))
    return w / (2.0 * np.pi)


def band_rows(y, band_y):
    yc = 0.5 * (y[:-1] + y[1:])
    return np.abs(yc) <= band_y


# ---------------------------------------------------------------------------
# measures -> band scalar
# ---------------------------------------------------------------------------

def measures(u, g, band_y=3.0, sigma_det=0.4):
    bm = band_rows(g["y"], band_y)
    ang2 = doubled_angle(u, g["dx"], sigma_det)
    circ = circulation(ang2)
    q = np.round(circ)

    out = {}
    out["trusted count"] = float(np.abs(q[bm]).sum()) / 2.0
    out["unrounded max"] = float(np.abs(circ[bm]).max())
    out["unrounded sum"] = float(np.abs(circ[bm]).sum())
    circ_s = gaussian_filter(np.abs(circ), 1.0 / g["dx"], mode="nearest")
    out["unrounded smooth"] = float(circ_s[bm].sum())

    # loop-persistence (v1 winner): net enclosed winding over growing loops
    count = np.zeros_like(q)
    for mloop in (1, 2, 3, 4, 6, 8):
        L = 2 * mloop + 1
        boxsum = uniform_filter(q, size=L, mode="constant") * (L * L)
        count += (np.abs(boxsum) >= 0.5).astype(float)
    out["loop-persistence"] = float(count[bm].sum())

    return out, dict(circ=circ, q=q, count=count, ang2=ang2)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(mus=(0.85, 0.90, 0.95), band_y=3.0, out="results/measures_nucleation"):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    names = ["trusted count", "loop-persistence", "unrounded max",
             "unrounded sum", "unrounded smooth"]

    print(f"band_y={band_y}. For each measure: straight / knee / pn per mu.")
    print("Good measure: straight~0 and knee<<pn at every mu.\n")
    header = f"{'measure':<18}" + "".join(f"  mu={m:<4}" for m in mus)
    fields_order = ["straight", "knee", "pn"]

    store = {}
    for mu in mus:
        g = make_grid(mu)
        us = {"straight": field_straight(g), "knee": field_knee(g),
              "pn": field_pn(g)}
        store[mu] = {"g": g, "u": us,
                     "m": {f: measures(u, g, band_y)[0] for f, u in us.items()}}

    for nm in names:
        print(nm)
        for f in fields_order:
            row = f"  {f:<8}"
            for mu in mus:
                row += f"  {store[mu]['m'][f][nm]:8.3f}"
            print(row)
        print()

    # figure at the sharpest mu: unrounded circ field, knee vs pn
    mu = mus[-1]; g = store[mu]["g"]
    _, k_aux = measures(store[mu]["u"]["knee"], g, band_y)
    _, p_aux = measures(store[mu]["u"]["pn"], g, band_y)
    fig, axs = plt.subplots(2, 3, figsize=(13, 6))
    for row, (lab, aux, u) in enumerate(
            [("knee", k_aux, store[mu]["u"]["knee"]),
             ("pn", p_aux, store[mu]["u"]["pn"])]):
        ext = [g["x"][0], g["x"][-1], g["y"][0], g["y"][-1]]
        axs[row, 0].imshow(u, extent=ext, origin="lower", cmap="gray",
                           vmin=-1, vmax=1, aspect="auto")
        axs[row, 0].set_ylabel(lab, fontsize=11)
        v = np.abs(aux["circ"]).max()
        im = axs[row, 1].imshow(aux["circ"], cmap="RdBu_r", vmin=-1, vmax=1,
                                aspect="auto")
        axs[row, 2].imshow(aux["count"], cmap="magma", aspect="auto")
        if row == 0:
            axs[row, 0].set_title("u", fontsize=9)
            axs[row, 1].set_title("unrounded circulation (max "
                                  f"{v:.2f})", fontsize=9)
            axs[row, 2].set_title("loop-persistence", fontsize=9)
        else:
            axs[row, 1].set_title(f"max {v:.2f}", fontsize=9)
        for a in axs[row]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Continuous measures at the sharpest knee (mu={mu}): "
                 "does the unrounded field stay quiet on the knee?",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = out / f"unrounded_vs_loop_mu{mu:.2f}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"figure -> {p}")


def field_pair(g, d, xi=1.0):
    """A +1/-1 dislocation pair separated by d along x at y=0, single valued."""
    X, Y = g["X"], g["Y"]
    xL, xR = -0.5 * d, 0.5 * d
    phase = Y + np.arctan2(Y, X - xL) - np.arctan2(Y, X - xR)
    rL = np.hypot(X - xL, Y); rR = np.hypot(X - xR, Y)
    A = np.tanh(rL / xi) * np.tanh(rR / xi)
    return A * np.cos(phase)


def run_separation(mu=0.90, ds=(0.3, 0.6, 1.0, 1.6, 2.6, 4.0, 6.0),
                   band_y=3.0, out="results/measures_nucleation"):
    """As a pair separates from sub-resolution to well split, does the unrounded
    measure climb continuously while the rounded count jumps?"""
    out = Path(out)
    g = make_grid(mu)
    rows = []
    for d in ds:
        u = field_pair(g, d)
        m, _ = measures(u, g, band_y)
        rows.append((d, m["trusted count"], m["unrounded sum"],
                     m["unrounded smooth"], m["loop-persistence"]))
    rows = np.array(rows)

    print(f"\nseparation sweep (mu={mu}): does unrounded ramp while count steps?")
    print(f"{'d':>6} {'count':>8} {'unrd_sum':>9} {'unrd_smth':>10} {'loop':>8}")
    for r in rows:
        print(f"{r[0]:6.2f} {r[1]:8.1f} {r[2]:9.3f} {r[3]:10.3f} {r[4]:8.0f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].plot(rows[:, 0], rows[:, 1], "ks-", label="rounded count (/2)")
    ax[0].plot(rows[:, 0], rows[:, 2], "ro-", label="unrounded sum")
    ax[0].plot(rows[:, 0], rows[:, 3], "b^-", label="unrounded smooth")
    ax[0].set_xlabel("pair separation d"); ax[0].set_ylabel("band scalar")
    ax[0].set_title("continuity: unrounded ramps, count steps")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(rows[:, 0], rows[:, 4], "mD-")
    ax[1].set_xlabel("pair separation d"); ax[1].set_ylabel("loop-persistence")
    ax[1].set_title("loop-persistence vs separation")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = out / f"separation_sweep_mu{mu:.2f}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"figure -> {p}")


if __name__ == "__main__":
    run()
    run_separation()
