"""Corrected check. The previous projection failed only because sympy
expands theta inside the trig args. The right test: the oscillatory
part's x-dependence enters solely through the carrier at frequency
2 k1 (i.e. 2 theta_x), so it must satisfy the harmonic ODE
    d^2(osc)/dx^2 + (2 k1)^2 osc = 0
identically, which holds iff osc is a pure 2theta harmonic (no DC in
x, no higher multiples). Envelope coefficients may depend on y."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import sympy as sp
from utils.uhu_symbolic import KneeAnsatz, is_zero

A = KneeAnsatz()
for name, ij in (("Txx", (0, 0)), ("Txy", (0, 1)), ("Tyy", (1, 1))):
    full = A.uhu_tensor("full")[ij]
    dc   = A.uhu_tensor("DC")[ij]
    osc  = sp.simplify(full - dc)
    ode = sp.simplify(sp.diff(osc, A.x, 2) + (2 * A.k1)**2 * osc)
    ok1 = is_zero(ode)
    # and the DC part is x-independent (all x-dependence is in osc)
    ok2 = is_zero(sp.diff(dc, A.x))
    print(f"{name}: osc''_xx + (2 k1)^2 osc == 0: {'PASS' if ok1 else 'FAIL'}"
          f"   |  DC x-independent: {'PASS' if ok2 else 'FAIL'}")
print()
print("=> oscillatory corruption of the knee uHu tensor lives entirely")
print("   on the single carrier at x-frequency 2 k1 (the 2theta mode);")
print("   artifact x-period = pi/k1, no higher harmonics present.")
