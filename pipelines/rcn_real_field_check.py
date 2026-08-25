# pipelines/rcn_real_field_check.py
"""
SymPy companion to Writeup 1: validates the closed-form claims of the
theory sections (Sections 2-3) from first principles.

Changelog
---------
v1: validates
  (1) A1^2 = (4/3)(R - (k^2-1)^2)                       [eq. A1_slaving]
  (2) A3   = -A1^3 / (4(R + 79 k^4 - 14 k^2 - 1))       [eq. A3_slaving]
  (3) E0bar = -(R - (k^2-1)^2)^2 / 6, roll baseline -R^2/6
                                                        [eqs. real_E0,
                                                         roll_baseline]
  (4) B = -(4/3)(k^2-1)(R-(k^2-1)^2), zeros at k^2 = 1, 1 +/- sqrt(R)
                                                        [eq. B_SH]
  (5) alpha = 2R/3 via alpha = -(1/2) dB/dk^2 at k^2=1;
      G^2 = w^2 - w^4/(2R)  -> w^2 - w^4 at R = 1/2     [eq. alpha_G_SH]
  (6) <u_theta^2> at k_B = 2R/3 (the quantity entering the literal
      bending-coefficient identification; see the eta TODO in the text)
  (7) sech^4 integral and the sin^3 law prefactor 4/(3 lambda) * alpha
  (8) the nipple identity 2 sin^3(pi/4 - a/2)/cos(pi/4 + a/2)
      = 1 - sin(a), and the critical angle root of s^3 + s - 1 = 0

Open (see red TODO in the writeup): the full O(eps^2) real-field
averaging with all cross terms, which should settle the working eta.
"""
import sympy as sp

def main():
    th, R, k2, y, a = sp.symbols('theta R k2 y a', positive=True)
    A1, A3, w, lam = sp.symbols('A1 A3 w lambda', real=True)

    avg = lambda e: sp.simplify(sp.integrate(e, (th, 0, 2*sp.pi))/(2*sp.pi))
    u = A1*sp.cos(th) + A3*sp.cos(3*th)
    lin = A1*(1-k2)*sp.cos(th) + A3*(1-9*k2)*sp.cos(3*th)
    E0 = sp.expand(avg(sp.Rational(1,2)*lin**2 - R/2*u**2
                       + sp.Rational(1,4)*u**4))

    # (1) fundamental slaving
    brack = sp.expand(sp.diff(E0, A1).subs(A3, 0)/A1)*8
    A1sq = sp.solve(sp.Eq(brack, 0), A1**2)[0]
    assert sp.simplify(A1sq - sp.Rational(4,3)*(R-(k2-1)**2)) == 0
    print("(1) A1^2 = (4/3)(R-(k^2-1)^2)                       OK")

    # (2) slaved third harmonic
    eq3 = sp.expand(sp.diff(E0, A3)).subs(A3**3, 0).subs(A3**2, 0)
    A3sol = sp.simplify(sp.solve(eq3, A3)[0].subs(A1**2, A1sq))
    target = -A1**3/(4*(R + 79*k2**2 - 14*k2 - 1))
    assert sp.simplify(A3sol - target.subs(A1**2, A1sq)) == 0 or \
           sp.simplify(sp.factor(A3sol) - sp.factor(target)) == 0
    print("(2) A3 = -A1^3/(4(R+79k^4-14k^2-1)); -A1^3/(4(R+64)) at k=1  OK")

    # (3) slaved averaged energy and roll baseline
    E0s = sp.factor(sp.simplify(
        E0.subs(A3, 0).subs(A1**4, A1sq**2).subs(A1**2, A1sq)))
    assert sp.simplify(E0s + (R-(k2-1)**2)**2/6) == 0
    assert sp.simplify(E0s.subs(k2, 1) + R**2/6) == 0
    print("(3) E0bar = -(R-(k^2-1)^2)^2/6, baseline -R^2/6      OK")

    # (4) constitutive function
    B = sp.factor(sp.simplify(-2*sp.diff(E0s, k2)))
    assert sp.simplify(B + sp.Rational(4,3)*(k2-1)*(R-(k2-1)**2)) == 0
    assert set(sp.solve(B, k2)) == {1, 1-sp.sqrt(R), 1+sp.sqrt(R)}
    print("(4) B = -(4/3)(k^2-1)(R-(k^2-1)^2), zeros 1, 1+/-sqrt(R)  OK")

    # (5) alpha and G^2
    alpha = sp.simplify(-sp.Rational(1,2)*sp.diff(B, k2).subs(k2, 1))
    assert sp.simplify(alpha - 2*R/3) == 0
    s = sp.symbols('s')
    G2 = sp.expand(sp.simplify(
        -sp.integrate(B.subs(k2, s), (s, 1, k2))/alpha).subs(k2, w+1))
    assert sp.simplify(G2 - (w**2 - w**4/(2*R))) == 0
    assert sp.simplify(G2.subs(R, sp.Rational(1,2)) - (w**2 - w**4)) == 0
    print("(5) alpha = 2R/3; G^2 = w^2 - w^4/(2R) -> w^2-w^4 at R=1/2  OK")

    # (6) <u_theta^2> at k_B (enters the literal eta identification)
    wth2 = sp.simplify(avg(sp.diff(u, th)**2)
                       .subs(A3, 0).subs(A1**2, A1sq.subs(k2, 1)))
    assert sp.simplify(wth2 - 2*R/3) == 0
    print("(6) <u_theta^2>|_kB = 2R/3 (literal eta = 4R/3; working eta")
    print("    ratio alpha/eta = 3/4 fixed empirically; see TODO)      OK")

    # (7) sin^3 law integral: verify the antiderivative
    # F(y) = (tanh(ay) - tanh^3(ay)/3)/a  satisfies F' = sech^4(ay),
    # and F(+oo) - F(-oo) = (2/a)(1 - 1/3) = 4/(3a).
    F = (sp.tanh(a*y) - sp.tanh(a*y)**3/3)/a
    assert sp.simplify(sp.diff(F, y) - sp.sech(a*y)**4) == 0
    val = sp.limit(F, y, sp.oo) - sp.limit(F, y, -sp.oo)
    assert sp.simplify(val - sp.Rational(4,3)/a) == 0
    print("(7) int sech^4(a y) dy = 4/(3a) -> sin^3 law prefactor      OK")

    # (8) nipple identity and critical angle
    al = sp.symbols('alpha_ff', positive=True)
    lhs = 2*sp.sin(sp.pi/4 - al/2)**3/sp.cos(sp.pi/4 + al/2)
    assert sp.simplify(lhs - (1 - sp.sin(al))) == 0
    root = sp.nsolve(s**3 + s - 1, s, 0.7)
    ang = sp.deg(sp.asin(root))
    assert abs(float(ang) - 43.0) < 0.5
    print(f"(8) nipple identity OK; critical angle = {float(ang):.2f} deg")

    print("\nAll checks passed.")

if __name__ == "__main__":
    main()
