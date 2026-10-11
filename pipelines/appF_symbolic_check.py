# appF_symbolic_check.py
"""
SymPy checks for Appendix F (the Riesz director error to second order),
written 28 Sep 2026. Writeup 3 sec:ansatz-status and
narrowband_verification.py are the sources; this script checks the
closed forms the appendix prints.

Checks
  (1) the derivative of the Riesz symbol m_j = -i xi_j/|xi| is
      -(i/|xi|) (delta_jl - xi_j xi_l/|xi|^2), i.e. -(i/|xi|) P_perp.
  (2) the third derivative of |xi| is the tensor T of eq:Ttensor,
      T_jln = (3 k_j k_l k_n - d_jn k_l - d_ln k_j - d_jl k_n)/|k|^2
      (unit k hats).
  (3) the contraction T_j : M with a symmetric M, in the frame of k
      (k = (1, 0) |k|): the component along k_perp is -2 M_12/|k|^2 and
      the component along k is -M_22/|k|^2.
  (4) the director consequence: with R_par = A sin theta, R_perp =
      -(g/|k|) cos theta, w cos 2phi = A^2 sin^2 - (g/k)^2 cos^2 and
      w sin 2phi = -(A g/k) sin 2theta; the non-oscillating part of the
      former is (A^2/2)(1 - eps^2), of the latter zero.
  (5) the second-order terms on the test fields: on A = 1 + a sin(q y),
      theta = x, the ratio rms(T-term) / rms(first-order term) is q/2
      (the "q/2|k|" of Writeup 3); on theta = x + (eps_k/q) sin(q y) the
      mixed component of Hess theta vanishes.

Run:  python appF_symbolic_check.py
"""
import sympy as sp

ok = True


def report(name, cond):
    global ok
    print("%-70s %s" % (name, "PASS" if cond else "FAIL"))
    ok = ok and cond


x1, x2 = sp.symbols("xi_1 xi_2", real=True)
xi = sp.Matrix([x1, x2])
r = sp.sqrt(x1**2 + x2**2)
m = -sp.I * xi / r
# (1)
Pperp = sp.eye(2) - xi * xi.T / r**2
dm = sp.Matrix(2, 2, lambda j, l: sp.diff(m[j], xi[l]))
report("(1) d m_j / d xi_l = -(i/|xi|) P_perp", sp.simplify(dm + sp.I / r * Pperp) == sp.zeros(2, 2))
# (2)
T = sp.MutableDenseNDimArray.zeros(2, 2, 2)
Tform = sp.MutableDenseNDimArray.zeros(2, 2, 2)
kh = xi / r
for j in range(2):
    for l in range(2):
        for n in range(2):
            T[j, l, n] = sp.diff(r, xi[j], xi[l], xi[n])
            Tform[j, l, n] = (3 * kh[j] * kh[l] * kh[n] - sp.KroneckerDelta(j, n) * kh[l]
                              - sp.KroneckerDelta(l, n) * kh[j] - sp.KroneckerDelta(j, l) * kh[n]) / r**2
report("(2) third derivative of |xi| = T of eq:Ttensor",
       all(sp.simplify(T[j, l, n] - Tform[j, l, n]) == 0 for j in range(2) for l in range(2) for n in range(2)))
# (3) contraction in the frame of k: xi = (kappa, 0)
kap = sp.symbols("kappa", positive=True)
M11, M12, M22 = sp.symbols("M11 M12 M22", real=True)
M = sp.Matrix([[M11, M12], [M12, M22]])
contr = [sum(Tform[j, l, n].subs({x1: kap, x2: 0}) * M[l, n] for l in range(2) for n in range(2)) for j in range(2)]
report("(3) (T:M)_perp = -2 M_12 / kappa^2", sp.simplify(contr[1] + 2 * M12 / kap**2) == 0)
report("(3) (T:M)_par  = -M_22 / kappa^2", sp.simplify(contr[0] + M22 / kap**2) == 0)
# (4)
A, g, th = sp.symbols("A g theta", positive=True)
Rpar, Rperp = A * sp.sin(th), -(g / kap) * sp.cos(th)
wc2 = sp.expand_trig(Rpar**2 - Rperp**2)
ws2 = 2 * Rpar * Rperp
report("(4) w cos 2phi = A^2 sin^2 - (g/k)^2 cos^2", sp.simplify(wc2 - (A**2 * sp.sin(th)**2 - g**2 / kap**2 * sp.cos(th)**2)) == 0)
report("(4) w sin 2phi = -(A g/k) sin 2theta", sp.simplify(ws2 + A * g / kap * sp.sin(2 * th)) == 0)
avg = lambda f: sp.integrate(f, (th, 0, 2 * sp.pi)) / (2 * sp.pi)
report("(4) <w cos 2phi> = (A^2/2)(1 - eps_perp^2), eps = g/(A k)", sp.simplify(avg(wc2) - A**2 / 2 * (1 - g**2 / (A**2 * kap**2))) == 0)
report("(4) <w sin 2phi> = 0", sp.simplify(avg(ws2)) == 0)
# (5)
X, Y, a, ek = sp.symbols("x y a epsilon_k", real=True)
q = sp.symbols("q", positive=True)
Af = 1 + a * sp.sin(q * Y)
thf = X                                         # k = (1, 0), |k| = 1
first = sp.cos(thf) * sp.diff(Af, Y)            # |(cos theta / |k|) P_perp grad A|, P_perp grad A = (0, A_y)
HA = sp.Matrix([[sp.diff(Af, X, 2), sp.diff(Af, X, Y)], [sp.diff(Af, X, Y), sp.diff(Af, Y, 2)]])
Tterm = sp.Matrix([sp.Rational(1, 2) * sp.sin(thf) * sum(Tform[j, l, n].subs({x1: 1, x2: 0}) * HA[l, n] for l in range(2) for n in range(2)) for j in range(2)])
ms = lambda f: sp.integrate(sp.integrate(f, (X, 0, 2 * sp.pi)), (Y, 0, 2 * sp.pi / q)) * q / (4 * sp.pi**2)
ratio = sp.simplify(sp.sqrt(ms(Tterm[0]**2 + Tterm[1]**2) / ms(first**2)))
report("(5) rms(second-order amplitude term) / rms(first-order term) = q/2", sp.simplify(ratio - q / 2) == 0)
thb = X + (ek / q) * sp.sin(q * Y)
Hth = sp.Matrix([[sp.diff(thb, X, 2), sp.diff(thb, X, Y)], [sp.diff(thb, X, Y), sp.diff(thb, Y, 2)]])
report("(5) bent test phase: mixed Hessian component theta_xy = 0", sp.simplify(Hth[0, 1]) == 0)

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
