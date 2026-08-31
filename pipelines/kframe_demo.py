# pipelines/kframe_demo.py
"""
The change-of-basis machinery behind the 'k-frame', fully written out
and machine-verified. Companion to sstar_derivation_walkthrough.py.

  A  R built from k alone; R k = (kappa, 0)
  B  conjugation A' = R A R^T: symmetry preserved, eigenvalues
     invariant, eigenvectors rotate by R
  C  scalars are invariant: k^T S k = k'^T S' k' = kappa^2 s'_11,
     the full sandwich written out
  D  the component dictionary: primed entries of S' and H' in terms
     of lab entries and the rotation angle alpha
  E  averaging commutes with the frame change:
     < R T R^T >_{S'} = R < T >_S R^T   (linearity + covariance)
"""
import sys
from pathlib import Path
import numpy as np
import sympy as sp

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
from utils.uhu_symbolic import is_zero

kx, ky = sp.symbols("k_x k_y", real=True)
al = sp.atan2(ky, kx)
kap = sp.sqrt(kx**2 + ky**2)
c, s = sp.cos(al), sp.sin(al)
R = sp.Matrix([[c, s], [-s, c]])
k = sp.Matrix([kx, ky])
s11, s12, s22 = sp.symbols("s_11 s_12 s_22", real=True)
h11, h12, h22 = sp.symbols("h_11 h_12 h_22", real=True)
S = sp.Matrix([[s11, s12], [s12, s22]])
H = sp.Matrix([[h11, h12], [h12, h22]])

print("A) R from k alone: rows are khat^T and that^T")
sp.pprint(sp.simplify(R))
Rk = sp.simplify(R * k)
print("   R k =", Rk.T, " -> (kappa, 0) with kappa = |k|:",
      is_zero(sp.simplify(Rk[0] - kap)) and is_zero(Rk[1]))

print("\nB) conjugation A' = R A R^T")
Sp_ = sp.simplify(R * S * R.T)
print("   symmetry: (R S R^T)^T - R S R^T = 0:",
      sp.simplify(Sp_ - Sp_.T) == sp.zeros(2, 2))
lam = sp.symbols("lambda")
charpoly_lab = sp.expand(sp.det(S - lam * sp.eye(2)))
charpoly_rot = sp.expand(sp.simplify(sp.det(Sp_ - lam * sp.eye(2))))
print("   char polys equal (eigenvalues invariant):",
      is_zero(sp.simplify(charpoly_lab - charpoly_rot)))
# eigenvector rotation, numeric spot check
An = np.array([[0.4, -0.9], [-0.9, 2.1]])
a0 = 1.1
Rn = np.array([[np.cos(a0), np.sin(a0)], [-np.sin(a0), np.cos(a0)]])
w0, V0 = np.linalg.eigh(An)
w1, V1 = np.linalg.eigh(Rn @ An @ Rn.T)
v = V0[:, 0]; vr = V1[:, 0]
print("   eigenvectors rotate by R (numeric):",
      min(np.linalg.norm(Rn @ v - vr), np.linalg.norm(Rn @ v + vr)) < 1e-12)

print("\nC) the scalar k^T S k, both routes")
lab = sp.expand((k.T * S * k)[0, 0])
print("   lab frame      : k^T S k =", lab)
kp = R * k
sandwich = sp.simplify(sp.expand((kp.T * Sp_ * kp)[0, 0]))
print("   k'^T S' k' = (Rk)^T (R S R^T) (Rk) = k^T (R^T R) S (R^T R) k")
print("   equality (invariance):", is_zero(sp.simplify(sandwich - lab)))
sp11 = sp.simplify(Sp_[0, 0])
print("   and since k' = (kappa, 0):  k'^T S' k' = kappa^2 s'_11, i.e.")
print("   s'_11 =", sp.trigsimp(sp11))
print("   check kappa^2 s'_11 = k^T S k:",
      is_zero(sp.simplify(sp.expand(kap**2 * sp11) - lab)))

print("\nD) component dictionary (alpha = atan2(k_y, k_x)); with")
print("   ca = cos alpha, sa = sin alpha:")
a_ = sp.symbols("alpha", real=True)
ca_, sa_ = sp.cos(a_), sp.sin(a_)
Rg = sp.Matrix([[ca_, sa_], [-sa_, ca_]])
Sg = sp.simplify(Rg * S * Rg.T)
for i, j, name in ((0, 0, "s'_11"), (0, 1, "s'_12"), (1, 1, "s'_22")):
    print(f"   {name} =", sp.expand_trig(sp.simplify(Sg[i, j])))
print("   (H' identical with h_ij; every primed entry is a quadratic")
print("    form in the lab entries with rotation-angle coefficients.)")

print("\nE) averaging commutes with the frame change (numeric gate):")
rng = np.random.default_rng(3)
L = rng.normal(size=(2, 2)); Sn = L @ L.T + 0.4 * np.eye(2)
kn = rng.normal(size=2); Hn = rng.normal(size=(2, 2)); Hn = .5*(Hn+Hn.T)
from utils.uhu_symbolic import model_averaged_tensor
a1 = 0.7
Rn = np.array([[np.cos(a1), np.sin(a1)], [-np.sin(a1), np.cos(a1)]])
lhs = Rn @ model_averaged_tensor(kn, Hn, Sn, 0.37) @ Rn.T
rhs = model_averaged_tensor(Rn @ kn, Rn @ Hn @ Rn.T, Rn @ Sn @ Rn.T, 0.37)
print("   || R <T>_S R^T - <T'>_{S'} || =", f"{np.abs(lhs - rhs).max():.2e}")
