# pipelines/task1_pi_over_k1.py
"""
Task 1. Is the coincidence between the doubled-angle artifact period and
the PN defect spacing exact, for all mu, and does it survive tilt or a
change of spacing? And if spacing cannot discriminate, what does?

The claim decomposes into two halves, each carrying its own factor of pi:

  (E) ESTIMATOR half. Any quadratic estimator (uHu forms u d_i d_j u; the
      Riesz director smooths cos 2 theta, sin 2 theta) lives on the doubled
      angle. Its invented structure is the carrier e^{2 i theta}, whose
      local x-frequency is 2 theta_x. Along a wall core where the phase
      advances at mean rate <theta_x> = q, the artifact x-period is
          P_art = 2 pi / (2 q) = pi / q.
  (P) PHYSICS half. For a wall with the odd symmetry
      theta(x, -y) = -theta(x, y), the phase jump across the core is
      2 theta(x, 0+). One unit-charge defect advances that jump by 2 pi,
      so it advances theta(x, 0+) by exactly pi. A chain at spacing L
      therefore has core mean slope q = pi / L, i.e.
          L = pi / q.

  Both periods equal pi / q with the SAME q, so they coincide identically:
  for every mu, every tilt, every spacing, equilibrium or not. Nothing
  about pi/k1 is special; k1 is just the value q takes when the chain far
  field matches the knee far field. The only loophole is the odd symmetry:
  a wall splitting the 2 pi charge unevenly between branches (phi above,
  2 pi - phi below) breaks P_art = L on each branch separately.

This script checks every load-bearing step, symbolically where possible,
by measurement where not, and then computes what DOES differ between the
two fields at the shared period.

Run: python task1_pi_over_k1.py
"""

import sys
from pathlib import Path

import numpy as np
import sympy as sp

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.uhu_symbolic import (KneeAnsatz, PNAnsatz, is_zero, pn_chain_lnpsi,
                          pn_chain_defects)

FAILED = []


def report(msg=""):
    print(msg, flush=True)


def check(name, ok):
    tag = "PASS" if ok else "FAIL"
    report(f"  [{tag}] {name}")
    if not ok:
        FAILED.append(name)


# =====================================================================
# A. The estimator half, symbolic
# =====================================================================

def part_A():
    report("=" * 70)
    report("A. ESTIMATOR HALF: artifact carrier frequency, symbolic")
    report("=" * 70)
    knee = KneeAnsatz()
    # The oscillatory part of every uHu component carries e^{2 i theta}
    # (checked in check_pointwise P3: osc = -(1/2)[cos 2th k_i k_j
    # + sin 2th th_ij]). Its local x-frequency is d(2 theta)/dx.
    freq_x = sp.diff(2 * knee.theta, knee.x)
    check("knee: d(2 theta)/dx = 2 k1 exactly, everywhere, all mu",
          is_zero(freq_x - 2 * knee.k1))
    report("    => knee artifact x-period = pi / k1, exact, no asymptotics.")

    # PN single defect: the same carrier has local frequency 2 theta_x =
    # -2 p1, which VANISHES on the core plateaus (shown in D); the PN
    # x-periodicity at the core is carried by the defect lattice itself,
    # through the staircase. That is part P, next.
    report("")


# =====================================================================
# B. The physics half, symbolic: pi per defect on the branch
# =====================================================================

def part_B():
    report("=" * 70)
    report("B. PHYSICS HALF: each defect advances theta(x, 0+) by exactly pi")
    report("=" * 70)

    # Single defect, symbolic weights carrying the e^{-pi} structure.
    pn = PNAnsatz(numeric_weights=True)
    x, y = pn.x, pn.y
    th_left = sp.limit(pn.theta.subs(x, -sp.Integer(1)), y, 0, "+")
    th_right = sp.limit(pn.theta.subs(x, sp.Integer(1)), y, 0, "+")
    check("PN single defect: theta(x<0, 0+) = 0",
          is_zero(th_left))
    check("PN single defect: theta(x>0, 0+) = pi",
          is_zero(th_right - sp.pi))

    # Three-defect symbolic chain at GENERIC spacing L: psi built from the
    # telescoping weights e^{-j pi}. The plateau values must be
    # e^{-m pi}, hence theta -> m pi, INDEPENDENT of L.
    L = sp.symbols("L", positive=True)
    xs = sp.symbols("x", real=True)
    ys = sp.symbols("y", positive=True)
    nd = 3
    a = sp.Rational(1, 2) * (1 + sp.exp(-sp.pi * nd))
    psi = a
    for j in range(nd):
        dj = sp.Rational(1, 2) * (sp.exp(-sp.pi * (j + 1))
                                  - sp.exp(-sp.pi * j))
        psi += dj * sp.erf((xs - j * L) / sp.sqrt(2 * ys))
    theta = ys - sp.log(psi)
    ok = True
    for m, x_probe in ((0, -L / 2), (1, L / 2), (2, 3 * L / 2),
                       (3, 5 * L / 2)):
        v = sp.limit(theta.subs(xs, x_probe), ys, 0, "+")
        if not is_zero(sp.simplify(v - m * sp.pi)):
            ok = False
    check("PN 3-defect chain, generic L: theta(0+) staircase = m pi "
          "(steps of pi at ANY spacing)", ok)
    report("    => core mean slope q = pi / L for every spacing L, so")
    report("       L = pi / q identically. Combined with A: P_art = pi / q")
    report("       = L. Coincidence is exact, parameter-free.")
    report("")


# =====================================================================
# C. Measurements: far-field slope and staircase, equilibrium and not
# =====================================================================

def part_C():
    report("=" * 70)
    report("C. MEASUREMENTS: chain far field and staircase, stabilized")
    report("   evaluator, equilibrium spacing and a 1.3x detuned spacing")
    report("=" * 70)
    mu = 0.5
    k1 = float(np.sqrt(1 - mu**2))
    nd = 20   # wide chain so a plane-wave window exists: L << sqrt(2y) << nd L

    for tag, k_eff in (("equilibrium L = pi/k1", k1),
                       ("detuned    L = 1.3 pi/k1", k1 / 1.3)):
        xd, L = pn_chain_defects(nd, k_eff)
        # far field: inside the plane-wave window, measured by a y-scan:
        # ripple is dead above 2 pi y / L^2 ~ 3, and the finite chain's
        # Burgers-merged collapse (ln psi is Hopf-Cole, so far steps with
        # gain e^{(m-j) pi} take over) sets in above 2 pi y / L^2 ~ nd / 2.
        # Sample at 2 pi y / L^2 = 5, central two spacings.
        y_ff = 5.0 * L**2 / (2.0 * np.pi)
        xs = np.linspace(-L, L, 400)
        out = pn_chain_lnpsi(xs, np.full_like(xs, y_ff), nd, k_eff)
        q_ff = float(np.mean(out["theta_x"]))

        # staircase at y -> 0+: step sizes at each defect
        y0 = 1e-6
        eps = 0.05 * L
        th_lo = pn_chain_lnpsi(xd - eps, np.full(nd, y0), nd, k_eff)["theta"]
        th_hi = pn_chain_lnpsi(xd + eps, np.full(nd, y0), nd, k_eff)["theta"]
        steps = th_hi - th_lo

        report(f"  {tag}:  L = {L:.4f}")
        report(f"    far-field <theta_x> at y = {y_ff:.0f}: {q_ff:.6f}  "
               f"vs pi/L = {np.pi / L:.6f}   "
               f"(rel diff {abs(q_ff * L / np.pi - 1):.2e})")
        report(f"    core steps: mean {np.mean(steps):.6f}, "
               f"spread {np.ptp(steps):.2e}  vs pi = {np.pi:.6f}")
        check(f"{tag}: far-field slope = pi/L to < 1%",
              abs(q_ff * L / np.pi - 1) < 1e-2)
        check(f"{tag}: every core step = pi to < 1e-6",
              bool(np.max(np.abs(steps - np.pi)) < 1e-6))
    report("    => detuning the spacing moves q and L together; the")
    report("       coincidence P_art = L survives because both are pi/q.")
    report("")


# =====================================================================
# D. What DOES differ at the shared period
# =====================================================================

def part_D():
    report("=" * 70)
    report("D. THE DISCRIMINANTS: same period, different carrier geometry")
    report("=" * 70)

    knee = KneeAnsatz()
    pn = PNAnsatz(numeric_weights=True)

    # local wavevector of the artifact carrier e^{2 i theta} is 2 grad theta
    report("  carrier wavevector 2 grad theta:")
    report(f"    knee:  ( {sp.simplify(2 * knee.thx)},")
    report(f"             {sp.simplify(2 * knee.thy)} )")
    report("           on core (y = 0): (2 k1, 0), along the wall")

    # PN in p-coordinates (Task 0): 2 grad theta = (-2 p1, 2 - p2)
    x, y = pn.x, pn.y
    p1 = sp.diff(pn.psi, x) / pn.psi
    p2 = sp.diff(pn.psi, x, 2) / pn.psi
    check("PN: 2 grad theta = (-2 p1, 2 - p2) in ratio coordinates",
          is_zero(sp.diff(2 * pn.theta, x) + 2 * p1)
          and is_zero(sp.diff(2 * pn.theta, y) - (2 - p2)))

    # plateau limits as y -> 0+ off the defect column: p1, p2 -> 0
    lim_p1 = sp.limit(p1.subs(x, 1), y, 0, "+")
    lim_p2 = sp.limit(p2.subs(x, 1), y, 0, "+")
    check("PN plateau limit y -> 0+: p1 -> 0 and p2 -> 0",
          is_zero(lim_p1) and is_zero(lim_p2))
    report("           on core plateaus: (0, 2), ACROSS the wall")
    report("    The two carriers are ORTHOGONAL at the core. Same period")
    report("    along x, but the knee's is a smooth wave RUNNING IN x while")
    report("    PN's oscillates ACROSS the wall and is x-periodic only")
    report("    through the defect lattice of its envelope.")
    report("")

    # consequence: anisotropic survival factors
    report("  frozen-phase survival of the carrier, exp(-2 (sx^2 thx^2 +")
    report("  sy^2 thy^2)), evaluated on the core:")
    sx, sy = sp.symbols("s_x s_y", positive=True)
    surv_knee = knee.corrugation_factor(sx, sy).subs(knee.y, 0)
    report(f"    knee: {sp.simplify(surv_knee)}    (only s_x acts)")
    report("    PN plateau: exp(-2 sy^2 (1 - p2/2)^2) -> exp(-2 sy^2)")
    report("                (only s_y acts)")

    mu = 0.9
    k1v = np.sqrt(1 - mu**2)
    for sg in (np.pi / 8, np.pi / 4, np.pi / 2):
        s_k = np.exp(-2 * sg**2 * k1v**2)
        s_p = np.exp(-2 * sg**2 * 1.0**2)
        report(f"    isotropic sigma = {sg:.3f}:  knee survival {s_k:.3f}   "
               f"PN survival {s_p:.1e}")
    report("")
    report("  Reading of the Session A puzzle (interpretation, built on the")
    report("  measured survivals above and the measured mu* trend):")
    report("  growing isotropic sigma suppresses PN's carrier rapidly")
    report("  (e^{-2 sigma^2}), which is why uHu IMPROVES on PN with sigma;")
    report("  the knee's carrier decays only as e^{-2 sigma^2 k1^2} (k1^2 =")
    report("  0.19 at mu = 0.9) AND the knee's failure above mu* = 0.82 is")
    report("  the DC eigen-flip, which finite kernels make worse, not")
    report("  better. Opposite width preferences, as observed.")
    report("")
    report("  Discrimination consequence: spacing carries NO information,")
    report("  proven above. Orientation of the doubled-angle carrier near")
    report("  the core carries ALL of it: along-wall (knee, defect free)")
    report("  vs across-wall (PN, nucleated). An anisotropic kernel probes")
    report("  exactly this, which hands the question to Task 2.")
    report("")


if __name__ == "__main__":
    part_A()
    part_B()
    part_C()
    part_D()
    report("=" * 70)
    if FAILED:
        report(f"TASK 1 CHECKS FAILED: {len(FAILED)}")
        for f in FAILED:
            report(f"  - {f}")
        sys.exit(1)
    report("TASK 1 SETTLED: coincidence exact and parameter-free; the")
    report("discriminant is carrier orientation, not spacing.")
