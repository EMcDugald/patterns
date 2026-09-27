# appA_symbolic_check.py
"""
SymPy checks of the derivations of Appendix A (RCN from the SH energy),
written 27 Sep 2026 for the paper. The writeup scripts
rcn_real_field_check.py, rcn_complex_check.py,
stationary_sh_harmonic_balance.py and sh_insertion_ledger.py check the
same chain independently; this script is the paper's own record.

Checks
  (1) linear growth rate sigma(k) = R - (1 - k^2)^2.
  (2) cos^3 theta = (3/4) cos theta + (1/4) cos 3 theta (odd harmonics
      closed under the cubic).
  (3) the period average of (A1 cos + A3 cos 3)^4 and the averaged energy
      E0(A1, A3; k^2); stationarity in A1 at A3 = 0 gives A1^2 = (4/3)(R -
      (k^2 - 1)^2); E0 at the slaved A1 is -(1/6)(R - (k^2 - 1)^2)^2.
  (4) the harmonic coefficients C1, C3 of the residual and the identity
      C_n = -2 dE0/dA_n; the slaved A3 = -A1^3 / (4 (R + 79 k^4 - 14 k^2 -
      1)); the ratio A3/A1 at k = 1 and R = 1/2.
  (5) B = -2 dE0/dk^2 = -(4/3) w (R - w^2); zeta = -(1/2) dB/dk^2 at w = 0
      = 2R/3; G^2 = -(1/zeta) Int B = w^2 - w^4/(2R); the stretching
      identity E0(w) - E0(0) = (zeta/2) G^2.
  (6) the Laplacian of a modulated stripe A cos theta and the bending
      coefficient eta = (1/2) A1^2(0) = 2R/3.
  (7) the Eckhaus wavenumber: B + 2 k^2 B' = 0 at R = 1/2 gives k_E.
  (8) the projection of SH on cos theta at constant k: A_t = (R - w^2) A
      - (3/4) A^3 (w = 0: the far-field law).
  (9) the stationary RCN residual with the linear B on the self-dual knee
      is zero (J = 0), and on a general field equals
      eta (Lap rho - 2 beta div(rho k) + 4 beta J), rho = div k + beta w.
 (10) the sin^3 law: the line energy of the knee at beta.

Run:  python appA_symbolic_check.py
"""
import sympy as sp

ok = True


def report(name, cond):
    global ok
    print("%-64s %s" % (name, "PASS" if cond else "FAIL"))
    ok = ok and cond


R, k2, th = sp.symbols("R k2 theta", positive=True)
A1, A3 = sp.symbols("A1 A3", real=True)
w = sp.symbols("w", real=True)

# (1)
kx, ky, x, y, t = sp.symbols("k_x k_y x y t", real=True)
v = sp.exp(sp.I * (kx * x + ky * y))
lin = R * v - (v + sp.diff(v, x, 2) + sp.diff(v, y, 2)
               + sp.diff(v + sp.diff(v, x, 2) + sp.diff(v, y, 2), x, 2)
               + sp.diff(v + sp.diff(v, x, 2) + sp.diff(v, y, 2), y, 2))
# simpler: (1 + Lap)^2 v = (1 - k^2)^2 v
lap = lambda f: sp.diff(f, x, 2) + sp.diff(f, y, 2)
op = lambda f: f + lap(f)
sigma = sp.simplify((R * v - op(op(v))) / v)
report("(1) sigma = R - (1 - k^2)^2", sp.simplify(sigma - (R - (1 - kx**2 - ky**2)**2)) == 0)

# (2)
report("(2) cos^3 = 3/4 cos + 1/4 cos 3", sp.simplify(sp.cos(th)**3 - (sp.Rational(3, 4) * sp.cos(th)
                                                                       + sp.Rational(1, 4) * sp.cos(3 * th))) == 0)

# (3)
u2 = A1 * sp.cos(th) + A3 * sp.cos(3 * th)
avg = lambda f: sp.integrate(sp.expand(f), (th, 0, 2 * sp.pi)) / (2 * sp.pi)
q4 = sp.simplify(avg(u2**4))
report("(3) <u^4> = 3/8 A1^4 + 3/8 A3^4 + 3/2 A1^2 A3^2 + 1/2 A1^3 A3",
       sp.simplify(q4 - (sp.Rational(3, 8) * A1**4 + sp.Rational(3, 8) * A3**4
                         + sp.Rational(3, 2) * A1**2 * A3**2 + sp.Rational(1, 2) * A1**3 * A3)) == 0)
E0 = (sp.Rational(1, 4) * (1 - k2)**2 * A1**2 + sp.Rational(1, 4) * (1 - 9 * k2)**2 * A3**2
      - R / 4 * (A1**2 + A3**2) + q4 / 4)
# check E0 is the period average of the SH density on the ansatz at constant k
dens = (sp.Rational(1, 2) * ((1 - k2) * A1 * sp.cos(th) + (1 - 9 * k2) * A3 * sp.cos(3 * th))**2
        - R / 2 * u2**2 + u2**4 / 4)
report("(3) E0 = period average of the SH density", sp.simplify(avg(dens) - E0) == 0)
stat1 = sp.solve(sp.diff(E0.subs(A3, 0), A1), A1)
A1sq = [sp.simplify(s**2) for s in stat1 if s != 0][0]
report("(3) A1^2 = 4/3 (R - (k^2 - 1)^2)", sp.simplify(A1sq - sp.Rational(4, 3) * (R - (k2 - 1)**2)) == 0)
E0s = sp.simplify(E0.subs(A3, 0).subs(A1**2, A1sq))
report("(3) E0 at the slaved A1 = -(1/6)(R - (k^2-1)^2)^2",
       sp.simplify(E0s + sp.Rational(1, 6) * (R - (k2 - 1)**2)**2) == 0)

# (4) harmonic coefficients of the residual N[u] = R u - (1 + Lap)^2 u - u^3 at constant k
Nres = sp.expand(R * u2 - ((1 - k2)**2 * A1 * sp.cos(th) + (1 - 9 * k2)**2 * A3 * sp.cos(3 * th)) - u2**3)
C = {n: sp.simplify(avg(Nres * sp.cos(n * th)) * 2) for n in (1, 3, 5, 7, 9)}
C1 = (R - (1 - k2)**2) * A1 - sp.Rational(3, 4) * A1**3 - sp.Rational(3, 4) * A1**2 * A3 - sp.Rational(3, 2) * A1 * A3**2
C3 = (R - (1 - 9 * k2)**2) * A3 - sp.Rational(1, 4) * A1**3 - sp.Rational(3, 2) * A1**2 * A3 - sp.Rational(3, 4) * A3**3
report("(4) C1 as written", sp.simplify(C[1] - C1) == 0)
report("(4) C3 as written", sp.simplify(C[3] - C3) == 0)
report("(4) C5 = -3/4 A1 A3 (A1 + A3)", sp.simplify(C[5] + sp.Rational(3, 4) * A1 * A3 * (A1 + A3)) == 0)
report("(4) C_n = -2 dE0/dA_n", sp.simplify(C[1] + 2 * sp.diff(E0, A1)) == 0 and sp.simplify(C[3] + 2 * sp.diff(E0, A3)) == 0)
A3lin = sp.solve(sp.Eq((R - (1 - 9 * k2)**2 - sp.Rational(3, 2) * A1**2) * A3, sp.Rational(1, 4) * A1**3), A3)[0]
A3s = sp.simplify(A3lin.subs(A1**2, A1sq))
report("(4) A3 = -A1^3 / (4 (R + 79 k^4 - 14 k^2 - 1))",
       sp.simplify(A3s + A1**3 / (4 * (R + 79 * k2**2 - 14 * k2 - 1))) == 0)
ratio = sp.simplify((A3s / A1).subs(k2, 1).subs(A1**2, A1sq.subs(k2, 1)))
report("(4) A3/A1 at k = 1 is -R/(3(64 + R)); -2.6e-3 at R = 1/2",
       sp.simplify(ratio + R / (3 * (64 + R))) == 0 and abs(float(ratio.subs(R, sp.Rational(1, 2))) + 0.002584) < 1e-5)

# (5) B, zeta, G^2, the stretching identity
E0w = -sp.Rational(1, 6) * (R - w**2)**2
B = -2 * sp.diff(E0w, w)            # d/dk^2 = d/dw
report("(5) B = -(4/3) w (R - w^2)", sp.simplify(B + sp.Rational(4, 3) * w * (R - w**2)) == 0)
zeta = sp.simplify(-sp.Rational(1, 2) * sp.diff(B, w).subs(w, 0))
report("(5) zeta = 2R/3", sp.simplify(zeta - 2 * R / 3) == 0)
s = sp.symbols("s", real=True)
G2 = sp.simplify(-sp.integrate(B.subs(w, s), (s, 0, w)) / zeta)
report("(5) G^2 = w^2 - w^4/(2R)", sp.simplify(G2 - (w**2 - w**4 / (2 * R))) == 0)
report("(5) E0(w) - E0(0) = (zeta/2) G^2", sp.simplify(E0w - E0w.subs(w, 0) - zeta / 2 * G2) == 0)
report("(5) at R = 1/2: zeta = 1/3, G^2 = w^2 - w^4",
       zeta.subs(R, sp.Rational(1, 2)) == sp.Rational(1, 3) and sp.simplify(G2.subs(R, sp.Rational(1, 2)) - (w**2 - w**4)) == 0)

# (6) Laplacian of a modulated stripe and the bending coefficient
Af = sp.Function("A", real=True)(x, y)
Tf = sp.Function("theta", real=True)(x, y)
uf = Af * sp.cos(Tf)
k1f, k2f = sp.diff(Tf, x), sp.diff(Tf, y)
divk = sp.diff(k1f, x) + sp.diff(k2f, y)
ksq = k1f**2 + k2f**2
lhs = lap(uf)
rhs = (lap(Af) - ksq * Af) * sp.cos(Tf) - (Af * divk + 2 * (k1f * sp.diff(Af, x) + k2f * sp.diff(Af, y))) * sp.sin(Tf)
report("(6) Lap(A cos theta) = (Lap A - k^2 A) cos - (A div k + 2 k.grad A) sin", sp.simplify(sp.expand(lhs - rhs)) == 0)
# bending term (1/4) A1^2(0) (div k)^2 = (eta/2)(div k)^2 with A1^2(0) = 4R/3
eta = sp.Rational(1, 2) * A1sq.subs(k2, 1)
report("(6) eta = (1/2) A1^2(k = 1) = 2R/3", sp.simplify(eta - 2 * R / 3) == 0)

# (7) Eckhaus wavenumber at R = 1/2: B + 2 k^2 B' = 0 with k^2 = 1 + w
Bw = B.subs(R, sp.Rational(1, 2))
eck = sp.expand(Bw + 2 * (1 + w) * sp.diff(Bw, w))
roots = [r_ for r_ in sp.Poly(eck, w).nroots() if r_.is_real and 0 < r_ < sp.sqrt(sp.Rational(1, 2))]
wE = float(roots[0])
kE = (1 + wE) ** 0.5
print("    Eckhaus: w_E = %.4f, k_E^2 = %.4f, k_E = %.4f at R = 1/2 (band edge k^2 = %.4f)" % (wE, 1 + wE, kE, 1 + 0.5**0.5))
report("(7) a single root 0 < w_E < sqrt(R) exists", len(roots) == 1)

# (8) projection of SH on cos theta at constant k with |k|^2 = 1 + w, A = A(t)
At = sp.Function("A", positive=True)(t)
kk = sp.symbols("kappa", positive=True)
thl = kk * x
ul = At * sp.cos(thl)
sh = sp.diff(ul, t) - (R * ul - op(op(ul)) - ul**3)
proj = sp.simplify(sp.integrate(sp.expand(sh * sp.cos(thl)), (x, 0, 2 * sp.pi / kk)) * kk / sp.pi)
target = sp.diff(At, t) - ((R - (kk**2 - 1)**2) * At - sp.Rational(3, 4) * At**3)
report("(8) A_t = (R - w^2) A - 3/4 A^3 at constant k, w = k^2 - 1", sp.simplify(proj - target) == 0)

# (9) the stationary RCN residual with the linear B
beta_, eta_ = sp.symbols("beta eta", positive=True)
mu = sp.symbols("mu", positive=True)
thK = sp.sqrt(1 - mu**2) * x + sp.log(2 * sp.cosh(beta_ * mu * y)) / beta_
kx_, ky_ = sp.diff(thK, x), sp.diff(thK, y)
wK = kx_**2 + ky_**2 - 1
divK = sp.diff(kx_, x) + sp.diff(ky_, y)
report("(9) knee: div k + beta w = 0", sp.simplify(divK + beta_ * wK) == 0)
JK = sp.diff(kx_, x) * sp.diff(ky_, y) - sp.diff(kx_, y) * sp.diff(ky_, x)
report("(9) knee: J = 0", sp.simplify(JK) == 0)
zeta_ = eta_ * beta_**2
resK = eta_ * lap(lap(thK)) - 2 * zeta_ * (sp.diff(wK * kx_, x) + sp.diff(wK * ky_, y))
report("(9) knee: stationary RCN residual eta Lap^2 theta - 2 zeta div(w k) = 0", sp.simplify(resK) == 0)
# general field
Tg = sp.Function("phi", real=True)(x, y)
kxg, kyg = sp.diff(Tg, x), sp.diff(Tg, y)
wg = kxg**2 + kyg**2 - 1
divg = sp.diff(kxg, x) + sp.diff(kyg, y)
Jg = sp.diff(kxg, x) * sp.diff(kyg, y) - sp.diff(kxg, y) * sp.diff(kyg, x)
rho = divg + beta_ * wg
resg = eta_ * lap(divg) - 2 * zeta_ * (sp.diff(wg * kxg, x) + sp.diff(wg * kyg, y))
ident = eta_ * (lap(rho) - 2 * beta_ * (sp.diff(rho * kxg, x) + sp.diff(rho * kyg, y)) + 4 * beta_ * Jg)
report("(9) general: residual = eta (Lap rho - 2 beta div(rho k) + 4 beta J)", sp.simplify(sp.expand(resg - ident)) == 0)

# (10) the sin^3 law: 2 x (zeta/2) Int w^2 dy on the knee with G^2 ~ w^2
gam = sp.simplify(2 * (zeta_ / 2) * sp.integrate((mu**2 / sp.cosh(beta_ * mu * y)**2)**2, (y, -sp.oo, sp.oo)))
report("(10) line energy = (4 zeta / 3 beta) mu^3", sp.simplify(gam - 4 * zeta_ * mu**3 / (3 * beta_)) == 0)

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
