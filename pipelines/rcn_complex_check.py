"""
rcn_complex_check.py

Symbolic verification of the complex-field route of Writeup 1
(Section 2.2) and of the phase-diffusion derivation (Section 2.5).

v1.0 changelog (measured justification):
  - New standalone script; the frozen method layer is untouched.
  - Part A, the complex route. Checks, on the modulated ansatz
    w = A(X) exp(i theta(x)) with slow real amplitude and k = grad theta:
      (A1) the factorization (1 + lap) w = exp(i theta) [C + i S] with
           C = (1 - k^2) A + lap A and S = 2 k.grad A + A div k
           (writeup eqs (5)-(7)), obtained by direct differentiation;
      (A2) the pointwise cancellation exp(i theta) exp(-i theta) = 1, so
           that |(1+lap) w|^2 = C^2 + S^2, |w|^2 = A^2, |w|^4 = A^4, with
           no averaging;
      (A3) the eps expansion (writeup eq (8)): the O(1) density
           (k^2-1)^2 A^2 - R A^2 + A^4/2 and the O(eps^2) density
           (2 k.grad A + A div k)^2 + 2 (1-k^2) A lap A, no O(eps) term,
           and the dropped remainder is exactly eps^4 (lap_X A)^2;
      (A4) the slaving relation A^2 = R - (k^2-1)^2 from
           d Ebar0 / d(A^2) = 0 and the compact form
           Ebar0 = -(1/2)(R - (k^2-1)^2)^2 (writeup eqs (9), (11));
      (A5) the doubled-convention relation: the complex energy density
           (writeup eq (5)) equals the reference's (2.1) with beta = 0,
           k0 = 1, written as a conjugate pair, and each of its
           coefficients is twice the real half-convention coefficient.
      (A6) the single mode closes: |w|^2 w for w = A exp(i theta) is
           A^3 exp(i theta), one mode; no harmonic generation.
  - Part B, the phase diffusion equations. Checks:
      (B1) the variational derivative of int Ebar0(k^2) with respect to
           theta equals -div(2 Ebar0'(k^2) k), by computing the
           Euler-Lagrange expression of the Lagrangian Ebar0(theta_x^2
           + theta_y^2) and comparing with the divergence form;
      (B2) with B = -2 dEbar0/dk^2 the gradient flow
           tau theta_t = -delta Ebar/delta theta is
           tau theta_t + div(k B(k^2)) = 0 (writeup eq (25));
      (B3) for SH, B = -(4/3)(k^2-1)(R - (k^2-1)^2) (writeup eq (26)),
           and B vanishes at k^2 = 1 and k^2 = 1 +- sqrt(R);
      (B4) the bending term (eta/2)(lap theta)^2 contributes
           eta lap^2 theta to delta Ebar/delta theta, so the regularized
           flow is tau theta_t = -div(k B) - eta lap^2 theta and its
           stationary equation is eta lap^2 theta + div(k B) = 0;
      (B5) alpha = -(1/2) dB/dk^2 at k^2 = 1 equals 2R/3, and
           G^2 = -(1/alpha) int_{1}^{k^2} B ds = w^2 - w^4/(2R)
           (writeup eqs (28)-(29)).
  - All claims are assertion-backed; the script exits nonzero on any
    failed check.

Equivalent hand route: the author's oral_comp_2.tex derives the same
expressions from the limit definition of the variational derivative
with separate flows for w and w*; this script checks the shorter route
used in the writeup, which gives the same results.

Run:  python rcn_complex_check.py            (defaults)
      python rcn_complex_check.py --quiet    (checks only, no displays)
"""

import argparse
import sys

import sympy as sp


CHECKS = []


def check(name, condition):
    """Record and print an assertion-backed check."""
    ok = bool(condition)
    CHECKS.append((name, ok))
    print(("  [ok]  " if ok else "  [FAIL]") + name)
    return ok


def zero(expr):
    return sp.simplify(sp.expand(expr)) == 0


# ----------------------------------------------------------------------
# Part A: the complex route
# ----------------------------------------------------------------------
def part_a(verbose=True):
    print("\nPart A: complex field, modulated single mode")
    x, y = sp.symbols("x y", real=True)
    eps = sp.symbols("epsilon", positive=True)
    R = sp.symbols("R", positive=True)

    # theta is a general smooth real function of x, y; A depends on the
    # slow variables X = eps x, Y = eps y (so each derivative of A brings
    # a factor eps). We realize this by A = a(eps x, eps y).
    # theta = Theta(X, Y)/eps with X = eps x: the wavevector k = grad_X Theta
    # is O(1) and every slow derivative of k or A brings an explicit eps.
    Th = sp.Function("Theta", real=True)
    theta = Th(eps * x, eps * y) / eps
    a = sp.Function("a", real=True)
    A = a(eps * x, eps * y)

    w = A * sp.exp(sp.I * theta)
    lap = lambda f: sp.diff(f, x, 2) + sp.diff(f, y, 2)
    grad = lambda f: sp.Matrix([sp.diff(f, x), sp.diff(f, y)])

    k = grad(theta)
    k2 = k.dot(k)
    divk = lap(theta)
    gradA = grad(A)
    lapA = lap(A)

    # (A1) factorization
    C = (1 - k2) * A + lapA
    S = 2 * k.dot(gradA) + A * divk
    lhs = (lap(w) + w)
    rhs = sp.exp(sp.I * theta) * (C + sp.I * S)
    check("(A1) (1+lap)(A e^{i theta}) = e^{i theta}[C + i S]",
          zero(lhs - rhs))

    # (A2) pointwise cancellation of the phase: differentiate the
    # conjugate field w* = A e^{-i theta} directly and multiply
    wstar = A * sp.exp(-sp.I * theta)
    lhs_star = lap(wstar) + wstar
    mod2 = sp.simplify(sp.expand(lhs * lhs_star))
    check("(A2) (1+lap)w (1+lap)w* = C^2 + S^2 (phase cancels pointwise)",
          zero(mod2 - (C**2 + S**2)))
    check("(A2) w w* = A^2 and (w w*)^2 = A^4",
          zero(sp.simplify(w * wstar) - A**2)
          and zero(sp.simplify((w * wstar)**2) - A**4))
    check("(A2) e^{i theta} e^{-i theta} = 1",
          zero(sp.exp(sp.I * theta) * sp.exp(-sp.I * theta) - 1))

    # (A3) eps expansion of the energy density
    dens = C**2 + S**2 - R * A**2 + sp.Rational(1, 2) * A**4
    dens = sp.expand(dens)

    # collect by powers of eps: derivatives of a(eps x, eps y) bring eps
    # explicitly; replace the function and its (Subs/Derivative) atoms by
    # plain symbols so that eps is the only generator left.
    def freeze(expr):
        mapping = {}
        for cls in (sp.Subs, sp.Derivative):
            for at in sorted(expr.atoms(cls), key=str):
                mapping[at] = sp.Symbol("f%d" % len(mapping))
            expr = expr.xreplace(mapping)
        for at in sorted(expr.atoms(sp.core.function.AppliedUndef),
                         key=str):
            mapping[at] = sp.Symbol("f%d" % len(mapping))
        return sp.expand(expr.xreplace(mapping)), mapping

    gradA_slow = gradA / eps          # = grad_X A
    lapA_slow = lapA / eps**2         # = lap_X A
    divk_slow = divk / eps            # = div_X k
    target0 = (k2 - 1)**2 * A**2 - R * A**2 + sp.Rational(1, 2) * A**4
    target2 = (2 * k.dot(gradA_slow) + A * divk_slow)**2 \
        + 2 * (1 - k2) * A * lapA_slow
    resid = sp.expand(dens - target0 - eps**2 * target2)
    resid_f, _ = freeze(resid)
    poly = sp.Poly(resid_f, eps)
    coeffs = {m[0]: sp.expand(c) for m, c in zip(poly.monoms(),
                                                 poly.coeffs())}
    check("(A3) energy density = O(1) local part + eps^2 [(2 k.grad_X A "
          "+ A div_X k)^2 + 2(1-k^2) A lap_X A] + O(eps^4), writeup eq (8)",
          all(coeffs.get(n, 0) == 0 for n in (0, 1, 2, 3))
          and set(coeffs) <= {4})
    check("(A3) no O(eps) term", coeffs.get(1, 0) == 0)
    check("(A3) the dropped remainder is exactly eps^4 (lap_X A)^2",
          zero(resid - eps**4 * lapA_slow**2))

    # (A4) slaving and compact form, with k^2 a symbol
    K2, A2s = sp.symbols("k2 A2", positive=True)
    E0 = (K2 - 1)**2 * A2s - R * A2s + sp.Rational(1, 2) * A2s**2
    slav = sp.solve(sp.diff(E0, A2s), A2s)[0]
    check("(A4) d Ebar0 / d(A^2) = 0 gives A^2 = R - (k^2-1)^2",
          zero(slav - (R - (K2 - 1)**2)))
    check("(A4) Ebar0 with A slaved is -(1/2)(R - (k^2-1)^2)^2",
          zero(E0.subs(A2s, slav) + sp.Rational(1, 2) * (R - (K2 - 1)**2)**2))

    # (A5) doubled convention: reference (2.1) with beta = 0, k0 = 1
    wc = sp.symbols("w")
    ws = sp.symbols("w_star")
    Lw, Lws = sp.symbols("Lw Lws")  # stand-ins for (lap+1)w, (lap+1)w*
    ref = Lw * Lws - R * wc * ws + sp.Rational(1, 2) * wc**2 * ws**2
    ours = sp.Abs(Lw)**2 - R * sp.Abs(wc)**2 + sp.Rational(1, 2) * sp.Abs(wc)**4
    # identify |z|^2 = z z*
    ours_pair = ours.subs({sp.Abs(Lw)**2: Lw * Lws,
                           sp.Abs(wc)**2: wc * ws,
                           sp.Abs(wc)**4: (wc * ws)**2})
    check("(A5) writeup eq (5) equals reference (2.1) at beta = 0, k0 = 1 "
          "(conjugate-pair form)", zero(ours_pair - ref))
    # each coefficient is twice the real half-convention coefficient
    u = sp.symbols("u", real=True)
    Lu = sp.symbols("Lu", real=True)
    half = sp.Rational(1, 2) * Lu**2 - R / 2 * u**2 + sp.Rational(1, 4) * u**4
    doubled = ref.subs({Lw: Lu, Lws: Lu, wc: u, ws: u})
    check("(A5) restricted to a real field, eq (5) is exactly twice the "
          "half-convention SH density", zero(doubled - 2 * half))

    # (A6) single mode closes under the cubic
    th = sp.symbols("theta", real=True)
    Ar = sp.symbols("A", positive=True)
    wm = Ar * sp.exp(sp.I * th)
    cubic = sp.simplify(wm * sp.conjugate(wm) * wm)
    check("(A6) |w|^2 w = A^3 e^{i theta}: the single mode closes, no "
          "harmonic generation", zero(cubic - Ar**3 * sp.exp(sp.I * th)))

    if verbose:
        print("    C =", C)
        print("    S =", S)


# ----------------------------------------------------------------------
# Part B: phase diffusion, unregularized and regularized
# ----------------------------------------------------------------------
def part_b(verbose=True):
    print("\nPart B: phase diffusion from the gradient flow of Ebar")
    x, y = sp.symbols("x y", real=True)
    R = sp.symbols("R", positive=True)
    theta = sp.Function("theta", real=True)(x, y)
    E0f = sp.Function("E0")          # generic averaged energy of k^2
    s = sp.symbols("s", positive=True)

    tx, ty, txx, txy, tyy = sp.symbols("tx ty txx txy tyy", real=True)
    k2 = tx**2 + ty**2
    # Lagrangian L(theta_x, theta_y) = E0(k^2); Euler-Lagrange expression
    # delta E / delta theta = -d/dx (dL/dtx) - d/dy (dL/dty)
    L = E0f(k2)
    dLdtx = sp.diff(L, tx)
    dLdty = sp.diff(L, ty)
    # substitute derivatives of theta and take total derivatives
    sub = {tx: sp.diff(theta, x), ty: sp.diff(theta, y)}
    EL = -(sp.diff(dLdtx.subs(sub), x) + sp.diff(dLdty.subs(sub), y))
    # divergence form: -div( 2 E0'(k^2) k )
    kx, ky = sp.diff(theta, x), sp.diff(theta, y)
    K2 = kx**2 + ky**2
    E0p = sp.Derivative(E0f(s), s).subs(s, K2)
    # build E0'(k^2) as the derivative function evaluated at K2
    E0prime = sp.diff(E0f(s), s)
    flux = 2 * E0prime.subs(s, K2) * sp.Matrix([kx, ky])
    divform = -(sp.diff(flux[0], x) + sp.diff(flux[1], y))
    check("(B1) delta/delta theta of int E0(|grad theta|^2) = "
          "-div(2 E0'(k^2) k)", zero((EL - divform).doit()))

    # (B2) with B = -2 E0', the flow tau theta_t = -delta E/delta theta
    # reads tau theta_t + div(k B) = 0
    Bf = -2 * E0prime.subs(s, K2)
    divkB = sp.diff(Bf * kx, x) + sp.diff(Bf * ky, y)
    check("(B2) -delta E/delta theta = -div(k B) with B = -2 dE0/dk^2, "
          "i.e. tau theta_t + div(k B) = 0", zero((-divform + divkB).doit()))

    # (B3) SH constitutive function
    K2s = sp.symbols("k2", positive=True)
    w = K2s - 1
    E0_sh = -sp.Rational(1, 6) * (R - w**2)**2
    B_sh = -2 * sp.diff(E0_sh, K2s)
    B_target = -sp.Rational(4, 3) * (K2s - 1) * (R - (K2s - 1)**2)
    check("(B3) B = -2 dE0/dk^2 = -(4/3)(k^2-1)(R-(k^2-1)^2) for SH",
          zero(B_sh - B_target))
    roots = sp.solve(B_sh, K2s)
    check("(B3) B vanishes at k^2 = 1 and k^2 = 1 +- sqrt(R)",
          set(sp.simplify(r) for r in roots)
          == {sp.Integer(1), 1 + sp.sqrt(R), 1 - sp.sqrt(R)})

    # (B4) bending term: variational derivative of (eta/2)(lap theta)^2
    eta = sp.symbols("eta", positive=True)
    Lb = sp.Rational(1, 2) * eta * (txx + tyy)**2
    subs2 = {txx: sp.diff(theta, x, 2), tyy: sp.diff(theta, y, 2)}
    # Euler-Lagrange for second-order Lagrangian:
    # dL/dtheta_xx -> + d^2/dx^2 (dL/dtxx), etc.
    ELb = (sp.diff(sp.diff(Lb, txx).subs(subs2), x, 2)
           + sp.diff(sp.diff(Lb, tyy).subs(subs2), y, 2))
    lap = lambda f: sp.diff(f, x, 2) + sp.diff(f, y, 2)
    check("(B4) delta/delta theta of int (eta/2)(lap theta)^2 = "
          "eta lap^2 theta", zero(ELb - eta * lap(lap(theta))))
    # regularized flow and its stationary form
    flow_rhs = -divkB - eta * lap(lap(theta))
    stationary = eta * lap(lap(theta)) + divkB
    check("(B4) regularized flow tau theta_t = -div(k B) - eta lap^2 theta; "
          "stationary form eta lap^2 theta + div(k B) = 0",
          zero((-flow_rhs - stationary).doit()))

    # (B5) alpha and G^2
    alpha = -sp.Rational(1, 2) * sp.diff(B_sh, K2s).subs(K2s, 1)
    check("(B5) alpha = -(1/2) dB/dk^2 at k^2 = 1 equals 2R/3",
          zero(alpha - 2 * R / 3))
    G2 = -sp.integrate(B_sh.subs(K2s, s), (s, 1, K2s)) / alpha
    check("(B5) G^2 = -(1/alpha) int_1^{k^2} B ds = w^2 - w^4/(2R)",
          zero(sp.expand(G2 - (w**2 - w**4 / (2 * R)))))
    ws = sp.symbols("w", real=True)
    G2w = sp.expand(G2.subs(K2s, ws + 1))
    check("(B5) G^2 = (k^2-1)^2 at leading order (its w^2 coefficient is 1)",
          sp.Poly(G2w, ws).coeff_monomial(ws**2) == 1)

    if verbose:
        print("    B_SH =", sp.factor(B_sh))
        print("    alpha =", alpha, "; G^2 =", sp.expand(G2))


# ----------------------------------------------------------------------
# args (per project conventions: _Args at the bottom + CLI mirror)
# ----------------------------------------------------------------------
class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    quiet = False


def main(args):
    part_a(verbose=not args.quiet)
    part_b(verbose=not args.quiet)
    n_fail = sum(1 for _, ok in CHECKS if not ok)
    print("\n%d checks, %d failed" % (len(CHECKS), n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--quiet", dest="quiet", action="store_true",
                   default=_Args.quiet)
    sys.exit(main(p.parse_args()))
