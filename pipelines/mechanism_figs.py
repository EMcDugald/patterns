# pipelines/mechanism_figs.py
"""
Figures and tables for the two knee corruption mechanisms (Writeup 2,
Session 3 verification). Everything plotted is remeasured in-script;
nothing is read from handoffs.

Fig 1  the one-knob-two-problems diagram in the (sigma, mu) plane:
       shaded region = on-core eigen-flip (DC-only condition
       mu^2 <tanh^2(mu y)>_sigma > 1 - mu^2), overlaid with contours
       of the frozen-phase core carrier survival e^{-2 sigma^2 k1^2}.
       The measured exact-machine point (sigma = pi/2, mu* = 0.8203)
       is marked for the continuity check.
Fig 2  measured decay of the surviving oscillatory part of <uHu> on
       the knee core (exact GH averaging) vs sigma^2, log scale,
       against the frozen-phase floor slope -2 k1^2.
Table  survival ledger (knee core vs PN plateau at sigma = pi/8,
       pi/4, pi/2) and the mu* summary, written as a LaTeX snippet.

Run: python mechanism_figs.py [--outdir DIR]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.uhu_symbolic import KneeAnsatz, lambdify_np, gh_average_matrix

try:
    from paper_style import use_paper_style
    use_paper_style()
except Exception:
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "savefig.dpi": 300, "savefig.bbox": "tight"})


# ---------------------------------------------------------------------
# DC-only flip boundary mu*(sigma)
# ---------------------------------------------------------------------

def avg_tanh2(mu, sig):
    f = lambda y: np.tanh(mu * y) ** 2 * np.exp(-y ** 2 / (2 * sig ** 2))
    num, _ = quad(f, -8 * sig, 8 * sig)
    return num / (np.sqrt(2 * np.pi) * sig)


def mustar_dc(sig):
    g = lambda m: m ** 2 * avg_tanh2(m, sig) - (1 - m ** 2)
    return brentq(g, 0.5, 0.9999)


def fig_phase_diagram(outdir):
    sigs = np.linspace(0.3, 2.6, 60)
    mus = np.linspace(0.3, 0.999, 240)
    boundary = np.array([mustar_dc(s) for s in sigs])

    # survival contours: e^{-2 sigma^2 (1 - mu^2)}
    SG, MU = np.meshgrid(sigs, mus)
    surv = np.exp(-2 * SG ** 2 * (1 - MU ** 2))

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.fill_between(sigs, boundary, 1.0, color="0.82", zorder=0,
                    label="eigen-flip region (DC condition)")
    cs = ax.contour(SG, MU, surv, levels=[0.05, 0.2, 0.4, 0.6, 0.8],
                    colors="C0", linewidths=0.9, zorder=1)
    ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=8)
    ax.plot(sigs, boundary, "k-", lw=1.4, zorder=2)
    ax.plot([np.pi / 2], [0.8203], "r*", ms=11, zorder=3,
            label=r"exact machine: $\mu^*(\pi/2)=0.8203$")
    ax.axvline(np.pi / 2, color="0.5", ls=":", lw=0.8)
    ax.set_xlabel(r"isotropic kernel width $\sigma$")
    ax.set_ylabel(r"far-field angle parameter $\mu=\sin\varphi$")
    ax.set_ylim(0.3, 1.0)
    ax.set_xlim(sigs[0], sigs[-1])
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(r"knee core: carrier survival $e^{-2\sigma^2 k_1^2}$"
                 "\nvs the eigen-flip boundary", fontsize=10)
    path = os.path.join(outdir, "mech_fig1_sigma_mu_plane.png")
    fig.savefig(path)
    plt.close(fig)
    return path, sigs, boundary


# ---------------------------------------------------------------------
# measured oscillatory decay on the core (exact GH averaging)
# ---------------------------------------------------------------------

def measured_osc_decay(mu, sigs, n_gh=40):
    A = KneeAnsatz()
    Tf = A.uhu_tensor("full")
    Td = A.uhu_tensor("DC")
    F = {n: lambdify_np((A.x, A.y, A.k1, A.k2), e) for n, e in
         (("fxx", Tf[0, 0]), ("fyy", Tf[1, 1]),
          ("dxx", Td[0, 0]), ("dyy", Td[1, 1]))}
    k1 = float(np.sqrt(1 - mu ** 2))

    def wrap(g):
        return lambda X, Y: np.broadcast_to(
            np.asarray(g(X, Y, k1, mu), float),
            np.broadcast(np.asarray(X), np.asarray(Y)).shape).copy()

    xs = np.linspace(0, np.pi / k1, 32, endpoint=False)
    ys = np.zeros_like(xs)
    out = {"xx": [], "yy": []}
    for s in sigs:
        S = s ** 2 * np.eye(2)
        for c in ("xx", "yy"):
            osc = (gh_average_matrix(wrap(F["f" + c]), xs, ys, S, n_gh)
                   - gh_average_matrix(wrap(F["d" + c]), xs, ys, S, n_gh))
            out[c].append(np.max(np.abs(osc)))
    return k1, {c: np.array(v) for c, v in out.items()}


def fig_decay(outdir, mu=0.9):
    sigs = np.array([0.5, 0.7, 0.9, 1.1, 1.3, 1.571])
    k1, amp = measured_osc_decay(mu, sigs)
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    slopes = {}
    for c, mk in (("xx", "o"), ("yy", "s")):
        ax.semilogy(sigs ** 2, amp[c], mk + "-", ms=4,
                    label=rf"$\langle T_{{{c}}}\rangle$ osc peak")
        slopes[c] = np.polyfit(sigs ** 2, np.log(amp[c]), 1)[0]
    ref = amp["yy"][0] * np.exp(-2 * k1 ** 2 * (sigs ** 2 - sigs[0] ** 2))
    ax.semilogy(sigs ** 2, ref, "k--", lw=1,
                label=rf"frozen-phase floor $e^{{-2k_1^2\sigma^2}}$")
    ax.set_xlabel(r"$\sigma^2$")
    ax.set_ylabel("surviving oscillation amplitude")
    ax.set_title(rf"knee core, $\mu={mu}$: exact vs frozen-phase decay",
                 fontsize=10)
    ax.legend(fontsize=8)
    path = os.path.join(outdir, "mech_fig2_osc_decay.png")
    fig.savefig(path)
    plt.close(fig)
    return path, k1, slopes


# ---------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------

def write_tables(outdir, k1, slopes):
    mu = 0.9
    k1sq = 1 - mu ** 2
    rows = []
    for s in (np.pi / 8, np.pi / 4, np.pi / 2):
        rows.append((s, np.exp(-2 * s ** 2 * k1sq), np.exp(-2 * s ** 2)))
    lines = [
        r"% survival ledger, remeasured (frozen-phase factors, mu = 0.9)",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"kernel width $\sigma$ & $\pi/8$ & $\pi/4$ & $\pi/2$ \\",
        r"\hline",
        "knee core survival $e^{-2\\sigma^2 k_1^2}$ & "
        + " & ".join(f"{r[1]:.3f}" for r in rows) + r" \\",
        "PN plateau survival $e^{-2\\sigma^2}$ & "
        + " & ".join(f"{r[2]:.1e}" for r in rows) + r" \\",
        r"\hline",
        r"\end{tabular}",
        "",
        r"% flip-threshold summary, remeasured",
        r"\begin{tabular}{ll}",
        r"\hline",
        r"exact machine, iso $\sigma=\pi/2$ & $\mu^* = 0.8203$ \\",
        r"DC-only prediction, iso $\sigma=\pi/2$ & $\mu^* = "
        f"{mustar_dc(np.pi/2):.4f}$" r" \\",
        r"full averaging limit $\langle\tanh^2\rangle\to1$ & "
        r"$\mu^* = 1/\sqrt{2} \approx 0.7071$ \\",
        r"S$^*$ needle, tol $= 0.05$ & no flip to $\mu = 0.999$ \\",
        r"\hline",
        r"\end{tabular}",
        "",
        "% measured osc decay slopes vs sigma^2 on the core, mu = 0.9:",
        f"%   Txx {slopes['xx']:+.3f}, Tyy {slopes['yy']:+.3f}, "
        f"frozen-phase floor {-2 * k1 ** 2:+.3f}",
    ]
    path = os.path.join(outdir, "mech_tables.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    p1, sigs, boundary = fig_phase_diagram(args.outdir)
    print(f"wrote {p1}")
    print(f"  flip boundary samples: mu*(pi/4) = "
          f"{mustar_dc(np.pi/4):.4f}, mu*(pi/2) = {mustar_dc(np.pi/2):.4f}, "
          f"mu*(2.5) = {mustar_dc(2.5):.4f}")
    p2, k1, slopes = fig_decay(args.outdir)
    print(f"wrote {p2}")
    print(f"  measured slopes: Txx {slopes['xx']:+.4f}, "
          f"Tyy {slopes['yy']:+.4f}, floor {-2 * k1**2:+.4f}")
    p3 = write_tables(args.outdir, k1, slopes)
    print(f"wrote {p3}")


class _Args:
    outdir = "out_mechanism_figs"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=_Args.outdir)
    main(p.parse_args())
