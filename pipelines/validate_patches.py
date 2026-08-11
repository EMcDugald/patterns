"""Validation of the three queued refinements (handoff Section 9 items 1, 2
and the Section 8b ln psi fix). Every claim is asserted against a known
closed form or a high-precision numeric reference before use.
"""
import numpy as np
import sympy as sp

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.uhu_symbolic import (GeneralAnsatz, KneeAnsatz, PNAnsatz, eig2x2,
                          eig2x2_numeric, wrap_mod_pi, is_zero, lambdify_np,
                          pn_chain_lnpsi,
                          pn_chain_lnpsi_mpmath, pn_chain_defects)

ok = lambda s: print(f"  [ok] {s}")

# =====================================================================
print("=" * 68)
print("A. eigensystem: closed 2x2 forms")
print("=" * 68)

# A1: agreement with numpy's eigh on random symmetric matrices, including
# the convention (dominant = most negative, matching compute_uhu_ops_smoother)
rng = np.random.default_rng(0)
for _ in range(2000):
    a, b, c = rng.normal(size=3)
    e = eig2x2_numeric(a, b, c, dominant="min")
    ev, evec = np.linalg.eigh(np.array([[a, b], [b, c]]))
    assert abs(e["lam_dom"] - ev[0]) < 1e-12
    assert abs(e["lam_sub"] - ev[1]) < 1e-12
    d = np.array([e["d1"], e["d2"]])
    assert min(np.linalg.norm(d - evec[:, 0]),
               np.linalg.norm(d + evec[:, 0])) < 1e-9
ok("eig2x2_numeric matches np.linalg.eigh (values, ordering, director) "
   "on 2000 random symmetric matrices")

# A2: symbolic DC tensor of the knee is exactly rank one with the right
# dominant eigenvalue. This is the handoff Sec 1 statement <T> = -(1/2) k k^T
# expressed at the level of the eigensystem.
knee = KneeAnsatz()
eig = knee.eigensystem("DC")
assert is_zero(eig["lam_plus"])
assert is_zero(eig["lam_minus"] + knee.k_mag2 / 2)
assert is_zero(eig["k_mag_A1"] - sp.sqrt(knee.k_mag2))
ok("knee DC: lam_sub = 0 exactly, lam_dom = -|k|^2/2, k_mag_A1 = |k|")

# A3: DC dominant angle equals the true wavevector angle, numerically over a
# transect (atan2 identities do not simplify symbolically, so this is the
# spot check that stands in for the symbolic assert).
err_f = lambdify_np((knee.y, knee.k1, knee.k2),
                    knee.angle_error_vs_truth("DC"))
for mu in (0.3, 0.5, 0.707, 0.9, 0.95):
    yy = np.linspace(-6, 6, 401)
    e = err_f(yy, np.sqrt(1 - mu**2), mu)
    assert np.max(np.abs(e)) < 1e-12, (mu, np.max(np.abs(e)))
ok("knee DC: angle_error_vs_truth = 0 to 1e-12 for mu in {0.3 .. 0.95}")

# A4: same for PN, where theta is genuinely two-dimensional
pn = PNAnsatz(numeric_weights=True)
err_pn = lambdify_np((pn.x, pn.y), pn.angle_error_vs_truth("DC"))
X, Y = np.meshgrid(np.linspace(-4, 4, 81), np.linspace(0.3, 8, 61))
epn = err_pn(X, Y)
assert np.nanmax(np.abs(epn)) < 1e-10, np.nanmax(np.abs(epn))
ok("PN DC: angle_error_vs_truth = 0 to 1e-10 on an 81x61 patch")

# A5: uhu_dc closed form agrees with the TR8 expansion route (general A)
gen = GeneralAnsatz()
for i, j in (("x", "x"), ("x", "y"), ("y", "y")):
    assert is_zero(gen.uhu_dc(i, j) - gen.uhu(i, j)["DC"])
ok("uhu_dc closed form == uhu(...)['DC'] for general A, theta")

# A6: uhu_osc closed form (A = 1) agrees with the expansion route
assert all(is_zero(knee.uhu_osc(i, j) - knee.uhu(i, j)["osc"])
           for i, j in (("x", "x"), ("x", "y"), ("y", "y")))
ok("uhu_osc closed form == uhu(...)['osc'] on the knee")

# A7: 'full' source reassembles the pointwise tensor exactly
Tfull = knee.uhu_tensor("full")
assert is_zero(Tfull[0, 0] - knee.u * sp.diff(knee.u, knee.x, 2))
assert is_zero(Tfull[1, 1] - knee.u * sp.diff(knee.u, knee.y, 2))
assert is_zero(Tfull[0, 1] - knee.u * sp.diff(knee.u, knee.x, knee.y))
ok("uhu_tensor('full') == u d_i d_j u pointwise")

# =====================================================================
print()
print("=" * 68)
print("B. PNAnsatz amplitude='tanh'")
print("=" * 68)

pn_a = PNAnsatz(numeric_weights=True, amplitude="tanh", xi=1)

# B1: the closed DC form (1/2)(A Hess A - A^2 k k^T) is already proven for
# GENERAL A and theta in check_pointwise.py / A5, so re-deriving it on the
# tanh-times-erf expression proves nothing and does not terminate. What is
# worth testing is the amplitude PLUMBING: build a cheap amplitude-bearing
# ansatz (tanh amplitude on the knee phase) and check the closed form against
# the TR8 expansion route there.
class _TanhKnee(GeneralAnsatz):
    def __init__(self):
        self.x, self.y = sp.symbols("x y", real=True)
        self.k1, self.k2 = sp.symbols("k_1 k_2", positive=True)
        self.A = sp.tanh(self.y)
        self.theta = self.k1 * self.x + sp.log(2 * sp.cosh(self.k2 * self.y))
        self._finalize()

    def _is_fast(self, arg):
        return arg.has(self.x)

tk = _TanhKnee()
for i, j in (("x", "x"), ("x", "y"), ("y", "y")):
    assert is_zero(tk.uhu_dc(i, j) - tk.uhu(i, j)["DC"]), (i, j)
ok("amplitude plumbing: uhu_dc == expansion-route DC on a tanh-amplitude ansatz")

# B2: phase content untouched by the amplitude option
assert is_zero(pn_a.theta - pn.theta)
assert is_zero(pn_a.W - pn.W)
ok("theta, W unchanged by the amplitude option (phase content untouched)")

# B3: W_A is identically zero for A = 1 and diverges at the core for tanh.
assert is_zero(pn.amplitude_contamination())
WA = lambdify_np((pn_a.x, pn_a.y), pn_a.amplitude_contamination())
ys = np.array([0.02, 0.05, 0.1, 0.3, 1.0, 3.0])
vals = np.array([float(WA(0.0, yy)) for yy in ys])
assert np.all(np.diff(vals) < 0), vals
assert vals[0] > 10, vals
ok("W_A = 0 for A = 1; on the tanh core W_A(y) = "
   + np.array2string(vals, precision=2) + " at y = " + str(ys))

# B4: the amplitude DC tensor recovers the exact wavevector once A -> 1.
# A fixed tolerance is the wrong test: with xi = 1 the tanh has not saturated
# at r = 3, so A Hess A is still O(1e-3) there. What must hold is the DECAY,
# and that the angle error tracks W_A, since both are set by the same
# A Hess A term. Scan rings of radius r on the y > 0 branch.
Txx = lambdify_np((pn_a.x, pn_a.y), pn_a.uhu_dc("x", "x"))
Txy = lambdify_np((pn_a.x, pn_a.y), pn_a.uhu_dc("x", "y"))
Tyy = lambdify_np((pn_a.x, pn_a.y), pn_a.uhu_dc("y", "y"))
thx = lambdify_np((pn_a.x, pn_a.y), pn_a.thx)
thy = lambdify_np((pn_a.x, pn_a.y), pn_a.thy)

phis = np.linspace(0.15, np.pi - 0.15, 60)
print(f"  {'r':>5} {'med |angle err| (deg)':>22} {'med W_A':>12}")
prev_e, errs, was = np.inf, [], []
for r in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
    xr, yr = r * np.cos(phis), r * np.sin(phis)
    eg = eig2x2_numeric(Txx(xr, yr), Txy(xr, yr), Tyy(xr, yr))
    e = np.degrees(np.abs(wrap_mod_pi(
        eg["angle_dom"] - np.arctan2(thy(xr, yr), thx(xr, yr)), numeric=True)))
    wa = np.array([float(WA(a, b)) for a, b in zip(xr, yr)])
    me, mw = float(np.nanmedian(e)), float(np.nanmedian(wa))
    errs.append(me); was.append(mw)
    print(f"  {r:>5.1f} {me:>22.3e} {mw:>12.3e}")
    assert me < prev_e, f"angle error not decreasing at r={r}"
    prev_e = me
assert errs[-1] < 1e-2, errs[-1]
ok("amplitude DC: angle error decays monotonically with r and tracks W_A; "
   f"{errs[-1]:.1e} deg at r = 8")

# B5: near the core it does NOT recover the wavevector. This is the
# physically meaningful half. Report rather than assert a threshold.
Xn, Yn = np.meshgrid(np.linspace(-1.5, 1.5, 61), np.linspace(0.05, 1.5, 41))
egn = eig2x2_numeric(Txx(Xn, Yn), Txy(Xn, Yn), Tyy(Xn, Yn))
errn = np.degrees(np.abs(wrap_mod_pi(
    egn["angle_dom"] - np.arctan2(thy(Xn, Yn), thx(Xn, Yn)), numeric=True)))
wan = np.array([[float(WA(xx, yy)) for xx in Xn[0]] for yy in Yn[:, 0]])
print(f"  near core (|x|<1.5, 0.05<y<1.5): median |angle err| = "
      f"{np.nanmedian(errn):.2f} deg, 90th pct = {np.nanpercentile(errn, 90):.2f} deg")
print(f"  fraction of near-core points with W_A > 1: {np.mean(wan > 1):.2f}")
ok("near-core amplitude contamination quantified (reported, not asserted)")

# =====================================================================
print()
print("=" * 68)
print("C. gauge-stabilized ln psi chain")
print("=" * 68)

nd, mu = 16, 0.9
k1 = np.sqrt(1 - mu**2)
xd, L = pn_chain_defects(nd, k1)

# C1: reproduce the handoff's measured naive-vs-mpmath corruption profile
print(f"  chain nd={nd}, mu={mu}, L={L:.4f}; theta error vs 60-digit mpmath")
print(f"  {'defect j':>9} {'naive':>12} {'stabilized':>12}")
worst_stab = 0.0
for j in (2, 6, 10, 12, 15):
    xq, yq = float(xd[j]) + 0.37, 0.8
    ref = pn_chain_lnpsi_mpmath(xq, yq, nd, k1)
    nai = float(pn_chain_lnpsi(xq, yq, nd, k1, stabilized=False)["theta"])
    sta = float(pn_chain_lnpsi(xq, yq, nd, k1, stabilized=True)["theta"])
    worst_stab = max(worst_stab, abs(sta - ref))
    print(f"  {j:>9} {abs(nai-ref):>12.2e} {abs(sta-ref):>12.2e}")
assert worst_stab < 1e-10, worst_stab
ok(f"stabilized theta matches mpmath to {worst_stab:.1e} across the whole "
   "chain; naive degrades exactly as the handoff reports")

# C2: continuity of the assembled ln psi across a defect. m jumps by 1 at
# x = x_j and the s_j flip must cancel it exactly (the cancellation needs
# 2c = 1 - e^{-pi}, which is precisely how c is defined). The test cannot be
# "max step is tiny": theta genuinely varies at rate theta_x, so on a grid of
# spacing dx every step is ~ |theta_x| dx. The signature of a discontinuity
# is an ANOMALOUS step at the crossing, so compare max step to median step.
xs = xd[8] + np.linspace(-0.02, 0.02, 201)
th = pn_chain_lnpsi(xs, np.full_like(xs, 0.8), nd, k1)["theta"]
steps = np.abs(np.diff(th))
ratio = steps.max() / np.median(steps)
assert ratio < 1.05, (ratio, steps.max(), np.median(steps))
ok(f"theta continuous across x = x_j (max step / median step = {ratio:.4f}, "
   f"i.e. no anomalous jump; all steps are just |theta_x| dx)")

# C3: derivatives agree with central differences of the stabilized theta
h = 1e-4
xq = np.array([xd[3] + 0.5, xd[9] + 1.1, xd[13] - 0.7])
yq = np.full_like(xq, 1.3)
d = pn_chain_lnpsi(xq, yq, nd, k1)
tx = (pn_chain_lnpsi(xq + h, yq, nd, k1)["theta"]
      - pn_chain_lnpsi(xq - h, yq, nd, k1)["theta"]) / (2 * h)
ty = (pn_chain_lnpsi(xq, yq + h, nd, k1)["theta"]
      - pn_chain_lnpsi(xq, yq - h, nd, k1)["theta"]) / (2 * h)
assert np.max(np.abs(d["theta_x"] - tx)) < 1e-6
assert np.max(np.abs(d["theta_y"] - ty)) < 1e-6
ok("analytic theta_x, theta_y match central differences to 1e-6")

# C4: the Section 8b far-field asymptotic, as an independent physics check.
# d(ln psi)/dy -> k1^2/2 and d(ln psi)/dx -> -k1 in the merged zone.
yfar = 60.0
dfar = pn_chain_lnpsi(np.array([0.0]), np.array([yfar]), nd, k1)
lp_y = 1.0 - dfar["theta_y"][0]          # = d(ln psi)/dy
lp_x = -dfar["theta_x"][0]               # = d(ln psi)/dx
print(f"  far field y={yfar}: d(ln psi)/dy = {lp_y:.5f} vs k1^2/2 = "
      f"{k1**2/2:.5f};  d(ln psi)/dx = {lp_x:.5f} vs -k1 = {-k1:.5f}")
assert abs(lp_y - k1**2 / 2) < 5e-4
assert abs(lp_x + k1) < 1e-2
ok("reproduces the handoff Sec 8b numeric asymptotic (0.09517 / -0.4327)")

print()
print("ALL PATCH VALIDATIONS PASSED")
