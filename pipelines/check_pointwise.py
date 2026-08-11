"""Scratch: verify the exact pointwise uHu closed forms before building on them.

Claims to test (A = 1, phase-only):
  (P1) T_ij = -cos^2(theta) * (theta_i theta_j + tan(theta) theta_ij)
  (P2) DC part = -(1/2) theta_i theta_j            [handoff Sec 1]
  (P3) osc part = -(1/2)[cos(2th) theta_i theta_j + sin(2th) theta_ij]

General A:
  (G1) DC_ij = (1/2)(A A_ij - A^2 theta_i theta_j) [handoff Sec 1]
"""
import sympy as sp

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.uhu_symbolic import GeneralAnsatz

gen = GeneralAnsatz()
x, y, A, th = gen.x, gen.y, gen.A, gen.theta

# ---- G1: general-A DC formula -------------------------------------
for i, j in (("x", "x"), ("x", "y"), ("y", "y")):
    c = {"x": x, "y": y}
    dc = gen.uhu(i, j)["DC"]
    expected = sp.Rational(1, 2) * (A * sp.diff(A, c[i], c[j])
                                    - A**2 * sp.diff(th, c[i]) * sp.diff(th, c[j]))
    assert sp.simplify(dc - expected) == 0, f"G1 failed at {i}{j}: {sp.simplify(dc-expected)}"
print("G1 ok: DC_ij = (1/2)(A A_ij - A^2 th_i th_j) for general A, theta")


# ---- phase-only specialization ------------------------------------
class PhaseOnly(GeneralAnsatz):
    def __init__(self):
        self.x, self.y = sp.symbols("x y", real=True)
        self.A = sp.Integer(1)
        self.theta = sp.Function("theta")(self.x, self.y)
        self._finalize()


po = PhaseOnly()
x, y, th = po.x, po.y, po.theta
c = {"x": x, "y": y}

for i, j in (("x", "x"), ("x", "y"), ("y", "y")):
    parts = po.uhu(i, j)
    ti, tj = sp.diff(th, c[i]), sp.diff(th, c[j])
    tij = sp.diff(th, c[i], c[j])

    # P1 pointwise
    p1 = -sp.cos(th)**2 * (ti * tj + sp.tan(th) * tij)
    assert sp.simplify(parts["full"] - p1) == 0, f"P1 failed at {i}{j}"

    # P2 DC
    assert sp.simplify(parts["DC"] + sp.Rational(1, 2) * ti * tj) == 0, f"P2 failed at {i}{j}"

    # P3 osc
    p3 = -sp.Rational(1, 2) * (sp.cos(2 * th) * ti * tj + sp.sin(2 * th) * tij)
    assert sp.simplify(parts["osc"] - p3) == 0, f"P3 failed at {i}{j}"

print("P1 ok: T_ij = -cos^2(th) (th_i th_j + tan(th) th_ij)  [pointwise, exact]")
print("P2 ok: DC_ij = -(1/2) th_i th_j")
print("P3 ok: osc_ij = -(1/2)[cos(2th) th_i th_j + sin(2th) th_ij]")

# ---- consequence: contamination ratio is exactly W |tan theta| -----
# ||tan(th) H||_F / ||k k^T||_2  = |tan th| * ||H||_F / |grad th|^2 = W |tan th|
print("\nconsequence: relative Hessian contamination of the pointwise tensor")
print("  = |tan(theta)| * ||H||_F / |grad theta|^2 = W * |tan(theta)|")
print("  W as defined in handoff Sec 2:", sp.simplify(po.W))
