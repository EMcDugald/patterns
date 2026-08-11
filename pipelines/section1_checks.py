"""Checks for the Section 1 walkthrough questions."""
import numpy as np
import sympy as sp

x, y = sp.symbols("x y", real=True)
th = sp.Function("theta")(x, y)
c = {1: x, 2: y}

# ---- Q4a: Lap(theta_i theta_j) identity ---------------------------
# claim: Lap(th_i th_j) = 2 H_ik H_kj + th_j d_i(Lap th) + th_i d_j(Lap th)
lap = lambda f: sp.diff(f, x, 2) + sp.diff(f, y, 2)
H = lambda a, b: sp.diff(th, c[a], c[b])
ok = True
for i in (1, 2):
    for j in (1, 2):
        lhs = lap(sp.diff(th, c[i]) * sp.diff(th, c[j]))
        rhs = (2 * sum(H(i, k) * H(k, j) for k in (1, 2))
               + sp.diff(th, c[j]) * sp.diff(lap(th), c[i])
               + sp.diff(th, c[i]) * sp.diff(lap(th), c[j]))
        if sp.simplify(lhs - rhs) != 0:
            ok = False
            print(f"  FAIL at ({i},{j}): {sp.simplify(lhs-rhs)}")
print(f"Q4a  Lap(th_i th_j) = 2 (H^2)_ij + sym(k (x) grad Lap th):  {ok}")

# ---- Q4b: Gaussian moments giving the expansion --------------------
# <f>_sigma = f + (sigma^2/2) Lap f + O(sigma^4) needs
#   int G = 1, int G s_i = 0, int G s_i s_j = sigma^2 delta_ij
s1, s2, sig = sp.symbols("s_1 s_2 sigma", real=True, positive=True)
G = sp.exp(-(s1**2 + s2**2) / (2 * sig**2)) / (2 * sp.pi * sig**2)
m0 = sp.integrate(sp.integrate(G, (s1, -sp.oo, sp.oo)), (s2, -sp.oo, sp.oo))
m11 = sp.integrate(sp.integrate(G * s1**2, (s1, -sp.oo, sp.oo)),
                   (s2, -sp.oo, sp.oo))
m12 = sp.integrate(sp.integrate(G * s1 * s2, (s1, -sp.oo, sp.oo)),
                   (s2, -sp.oo, sp.oo))
print(f"Q4b  int G = {sp.simplify(m0)}, int G s1^2 = {sp.simplify(m11)}, "
      f"int G s1 s2 = {sp.simplify(m12)}")

# ---- Q3: does a boxcar null the 2theta term EXACTLY? ---------------
# On the knee core theta = k1 x + g(y), so the oscillation in x is
# cos(2 k1 x + phi). A Gaussian leaves exp(-2 sigma^2 k1^2), never zero.
# A boxcar of length L leaves sinc(k1 L) = sin(k1 L)/(k1 L), which is
# EXACTLY zero at L = n pi / k1.
k1s, L, phi = sp.symbols("k_1 L phi", positive=True)
box = sp.integrate(sp.cos(2 * k1s * (x + s1) + phi), (s1, -L / 2, L / 2)) / L
box = sp.simplify(sp.expand_trig(sp.simplify(box)))
print(f"Q3   boxcar average of cos(2 k1 x + phi) over length L:\n     {box}")
print(f"     at L = pi/k1: {sp.simplify(box.subs(L, sp.pi / k1s))}")
print(f"     at L = 2 pi/k1: {sp.simplify(box.subs(L, 2 * sp.pi / k1s))}")

gauss = sp.exp(-2 * sig**2 * k1s**2)
print(f"     Gaussian, for comparison: {gauss}  (never zero)")

# numeric: half-wavelength boxcar vs Gaussian of equal RMS width
for mu in (0.5, 0.8, 0.9, 0.95):
    k1v = np.sqrt(1 - mu**2)
    Lv = np.pi / k1v                 # one period of the 2theta oscillation
    sig_eq = Lv / np.sqrt(12)        # Gaussian with the same RMS width
    print(f"     mu={mu:.2f}  k1={k1v:.3f}  L_null={Lv:.2f}  "
          f"sigma_equiv={sig_eq:.2f}  gaussian survival="
          f"{np.exp(-2*sig_eq**2*k1v**2):.3e}  boxcar survival=0")
