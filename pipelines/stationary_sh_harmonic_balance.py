"""
Symbolic harmonic-balance companion for the stationary Swift--Hohenberg
equation

    R u - (1 + Laplacian)^2 u - u^3 = 0.

v2 changelog (measured justification):
  - Projection onto cos(n theta) now uses the exponential rewrite
    (collect the coefficient of e^{i n theta}) instead of
    sp.integrate over [0, 2 pi]; identical results, runs in ~1 s
    instead of minutes (sympy's trig integrals are the bottleneck).
  - The exact algebraic solve of the coupled two-mode system
    (several cubic branches, very slow) is gated behind
    SOLVE_EXACT_BRANCHES = False.
  - Added the variational cross-check C_n = -2 dE0/dA_n, proving the
    projection route and the averaged-energy (Galerkin) route are the
    same equations (factor -2 from <cos^2> = 1/2).

For straight rolls u(theta), theta = k x, the Laplacian acts as
Laplacian cos(n theta) = -n^2 k^2 cos(n theta). This script:
  1. builds u = A1 cos(theta) + A3 cos(3 theta),
  2. projects the stationary-SH residual onto cosine harmonics,
  3. prints the mode coefficients C_1, C_3 (retained) and
     C_5, C_7, C_9 (residual, quantifying the truncation),
  4. verifies C_1 = -2 dE0/dA1 and C_3 = -2 dE0/dA3,
  5. reports the leading and linear-in-A3 third-harmonic relations,
     with the slaved substitution reducing the latter to
     A3 = -A1^3 / (4 (R + 79 k^4 - 14 k^2 - 1)).
"""
import sympy as sp

SOLVE_EXACT_BRANCHES = False   # coupled cubics; slow; enable if wanted


def cosine_coefficient(expr, theta, n):
    """Coefficient of cos(n theta) via the exponential rewrite:
    write everything in e^{i m theta}; for n >= 1 the cos(n theta)
    coefficient is twice the e^{i n theta} coefficient (the
    e^{-i n theta} half carries the other, equal share)."""
    ex = sp.expand(sp.expand_trig(expr).rewrite(sp.exp))
    if n == 0:
        return sp.simplify(ex.coeff(sp.exp(sp.I*theta), 0)
                           .coeff(sp.exp(-sp.I*theta), 0))
    return sp.simplify(2*ex.coeff(sp.exp(sp.I*n*theta)))


def main():
    theta = sp.symbols("theta", real=True)
    R, k2 = sp.symbols("R k2", real=True)
    A1, A3 = sp.symbols("A1 A3", real=True)

    modes = (1, 3)
    generated_modes = (1, 3, 5, 7, 9)

    u = A1*sp.cos(theta) + A3*sp.cos(3*theta)
    linear_operator_u = sum(A*(1 - n**2*k2)**2*sp.cos(n*theta)
                            for n, A in zip(modes, (A1, A3)))
    residual_raw = sp.expand(R*u - linear_operator_u - u**3)

    coeffs = {n: cosine_coefficient(residual_raw, theta, n)
              for n in generated_modes}

    print("Mode coefficients of the stationary residual:")
    for n in generated_modes:
        print(f"  C_{n} = {sp.factor(coeffs[n])}")

    # variational cross-check: C_n = -2 dE0/dA_n
    avg = lambda e: sp.integrate(e, (theta, 0, 2*sp.pi))/(2*sp.pi)
    lin = A1*(1-k2)*sp.cos(theta) + A3*(1-9*k2)*sp.cos(3*theta)
    E0 = sp.expand(avg(sp.Rational(1, 2)*lin**2 - R/2*u**2
                       + sp.Rational(1, 4)*u**4))
    ok1 = sp.simplify(coeffs[1] + 2*sp.diff(E0, A1)) == 0
    ok3 = sp.simplify(coeffs[3] + 2*sp.diff(E0, A3)) == 0
    print(f"\nVariational identity C_1 = -2 dE0/dA1: {ok1}")
    print(f"Variational identity C_3 = -2 dE0/dA3: {ok3}")

    # leading-order third harmonic (drop both cross terms)
    L3 = R - (1 - 9*k2)**2
    print("\nLeading-order:  A3 = A1^3 / (4 (R - (1-9k^2)^2))"
          "  [negative for R < 64 near k = 1]")
    print(f"  = {sp.factor(A1**3/(4*L3))}")

    # linear-in-A3, keeping the (3/2) A1^2 A3 cross term
    A3_lin = sp.factor(sp.solve(sp.Eq(coeffs[3].subs(A3**3, 0), 0), A3)[0])
    print("\nLinear-in-A3 with the cross term:")
    print(f"  A3 = {A3_lin}")

    # slaved substitution A1^2 = (4/3)(R - (k^2-1)^2)
    A1sq = sp.Rational(4, 3)*(R - (k2 - 1)**2)
    A3_slaved = sp.factor(sp.simplify(A3_lin.subs(A1**2, A1sq)))
    print("\nAfter slaving A1^2 = (4/3)(R - (k^2-1)^2):")
    print(f"  A3 = {A3_slaved}")
    target = -A1**3/(4*(R + 79*k2**2 - 14*k2 - 1))
    print("  matches -A1^3/(4(R + 79k^4 - 14k^2 - 1)):",
          sp.simplify(A3_slaved - target) == 0)

    if SOLVE_EXACT_BRANCHES:
        print("\nExact two-mode branches (slow):")
        sols = sp.solve((sp.Eq(coeffs[1], 0), sp.Eq(coeffs[3], 0)),
                        (A1, A3), dict=True)
        for i, sol in enumerate(sols, 1):
            print(f"  Branch {i}: {sol}")


if __name__ == "__main__":
    main()