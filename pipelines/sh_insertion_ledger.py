"""
sh_insertion_ledger.py

Symbolic verification of the single-insertion derivation of the RCN
energy from Swift--Hohenberg (Writeup 1, Sections 2.4-2.6), plus the
harmonic-projection ledger and its numerical comparison table
(Section 2.3 recurrence subsection).

v1.0 changelog (measured justification):
  - New standalone script; the frozen method layer is untouched.
  - Part A mechanizes the insertion calculation end to end: modulated
    ansatz into the SH energy density, epsilon bookkeeping for slow
    derivatives under the distinguished balance w ~ eps, fast-phase
    averaging by the exponential-rewrite trick (constant Fourier
    coefficient; sympy's trig integrals are minutes-slow, the rewrite
    is ~1 s), truncation at O(eps^2), and classification of every
    surviving term as baseline / stretching / bending, with alpha,
    eta, and beta read off and asserted.
  - Part B reproduces the harmonic ledger C1..C9, the variational
    identity C_n = -2 dE0/dA_n, both closed forms for A3 (full
    linearized and linear response), and the exact-solve comparison
    table quoted in the writeup (relative errors at k = 1).
  - All claims are assertion-backed; the script exits nonzero on any
    failed check.

Conventions: half convention throughout,
    F = int [ (1/2)((1+lap)u)^2 - (R/2) u^2 + (1/4) u^4 ].
Slaving: A1^2 = (4/3)(R - w^2), w = k^2 - 1.
Targets: E0bar = -(R - w^2)^2/6, baseline -R^2/6, alpha = 2R/3,
G^2 = w^2 - w^4/(2R), eta = A1^2(k_B)/2 = 2R/3, beta = 1,
two-mode bending addition (9/4) A3^2 (div k)^2.

Run:  python sh_insertion_ledger.py            (defaults)
      python sh_insertion_ledger.py --with-a3  (include A3 in Part A)
      python sh_insertion_ledger.py --exact    (exact two-mode solve)
"""

import argparse
import sys

import sympy as sp


# ----------------------------------------------------------------------
# shared symbols
# ----------------------------------------------------------------------
TH = sp.symbols("theta", real=True)          # fast phase
R = sp.symbols("R", positive=True)
EPS = sp.symbols("epsilon", positive=True)   # modulation parameter

CHECKS = []


def check(name, condition):
    """Record and print an assertion-backed check."""
    ok = bool(condition)
    CHECKS.append((name, ok))
    print(("  [ok]  " if ok else "  [FAIL]") + name)
    return ok


def phase_average(expr):
    """Average over the fast phase: constant Fourier coefficient via
    the exponential rewrite (fast; equals (1/2pi) int_0^{2pi})."""
    ex = sp.expand(sp.expand_trig(sp.expand(expr)).rewrite(sp.exp))
    ex = sp.expand(ex)
    # collect the theta-free part: successively strip exp(I n theta)
    const = sp.S(0)
    for term in sp.Add.make_args(ex):
        if term.has(TH):
            continue
        const += term
    return sp.simplify(const)


def cosine_coefficient(expr, n):
    """Coefficient of cos(n theta), n >= 1, via the exponential
    rewrite (twice the e^{i n theta} coefficient)."""
    ex = sp.expand(sp.expand_trig(sp.expand(expr)).rewrite(sp.exp))
    return sp.simplify(2 * ex.coeff(sp.exp(sp.I * n * TH)))


# ----------------------------------------------------------------------
# Part A: the insertion ledger
# ----------------------------------------------------------------------
def part_a(with_a3):
    print("\n================ Part A: insertion ledger ================")
    # Slow-derivative placeholders, each carrying its epsilon power.
    # W  ~ w = k^2 - 1              (order eps, distinguished balance)
    # D  ~ div k                     (order eps)
    # Gn ~ k . grad A_n              (order eps)
    # Ln ~ lap A_n                   (order eps^2)
    W, D = sp.symbols("W D", real=True)
    G1, L1 = sp.symbols("G1 L1", real=True)
    A1 = sp.symbols("A1", positive=True)
    k2 = 1 + EPS * W                       # k^2 at the point

    def op_on_mode(n, An, Gn, Ln):
        """(1 + lap) acting on An cos(n theta) with slow modulation:
        [(1 - n^2 k^2) An + eps^2 Ln] cos - eps n (An D + 2 Gn) sin."""
        cos_part = (1 - n**2 * k2) * An + EPS**2 * Ln
        sin_part = -EPS * n * (An * D + 2 * Gn)
        return cos_part * sp.cos(n * TH) + sin_part * sp.sin(n * TH)

    modes = [(1, A1, G1, L1)]
    if with_a3:
        A3, G3, L3 = sp.symbols("A3 G3 L3", real=True)
        modes.append((3, A3, G3, L3))

    u = sum(An * sp.cos(n * TH) for n, An, _, _ in modes)
    opu = sum(op_on_mode(n, An, Gn, Ln) for n, An, Gn, Ln in modes)

    density = sp.Rational(1, 2) * opu**2 - R / 2 * u**2 \
        + sp.Rational(1, 4) * u**4
    avg = sp.expand(phase_average(density))

    # ---- truncate at O(eps^2): the writeup's ordering rule
    trunc = sp.expand(avg.series(EPS, 0, 3).removeO())

    # every dropped term must be O(eps^3) or higher
    dropped = sp.expand(avg - trunc)
    lead = sp.degree(sp.Poly(sp.expand(dropped), EPS), EPS) if dropped != 0 \
        else None
    low = None
    if dropped != 0:
        low = min(m.as_poly(EPS).monoms()[0][0]
                  for m in [sp.Poly(dropped, EPS)])
        low = min(mon[0] for mon in sp.Poly(dropped, EPS).monoms())
    check("dropped terms are O(eps^3) or higher",
          dropped == 0 or (low is not None and low >= 3))

    # ---- classify the O(eps^2)-truncated average
    # local piece: set all slow-derivative placeholders to zero
    local = trunc.subs({D: 0, G1: 0, L1: 0})
    bend_bracket = sp.simplify(trunc - local)

    if with_a3:
        A3, G3, L3 = [s for s in modes[1][1:]]
        local = trunc.subs({D: 0, G1: 0, L1: 0, G3: 0, L3: 0})
        bend_bracket = sp.simplify(trunc - local)

    # split the modulation terms: the sin-bracket squares are the
    # bending energy; with A3 present there is additionally a
    # legitimate O(eps^2) amplitude-diffusion cross term
    # (1/2)(1 - 9 k^2) A3 lap A3, because the third harmonic sits at
    # an O(1) mismatch (1 - 9k^2) ~ -8 rather than at band center.
    expect_bend = sp.Rational(1, 4) * EPS**2 * (A1 * D + 2 * G1)**2
    lap_cross = sp.S(0)
    if with_a3:
        expect_bend = sp.Rational(1, 4) * EPS**2 * (
            (A1 * D + 2 * G1)**2 + 9 * (A3 * D + 2 * G3)**2)
        # sin(theta) sin(3theta) cross terms average to zero: no
        # A1 A3 cross term may appear in the bending part
        lap_cross = -4 * EPS**2 * A3 * L3
    check("modulation terms = bending brackets"
          + (" + lap A3 cross term (-4 A3 lap A3)" if with_a3 else ""),
          sp.simplify(bend_bracket - expect_bend - lap_cross) == 0)
    if with_a3:
        check("no A1 A3 cross term in the bending part "
              "(<sin sin3> = 0)",
              sp.expand(expect_bend).coeff(A1 * A3) == 0
              and sp.expand(bend_bracket - lap_cross).coeff(A1).coeff(A3)
              == 0)

    # (lap A) survives only at O(eps^3): the cross term with the
    # mismatch carries w * lap A
    lap_terms = sp.expand(avg).coeff(L1)
    lap_lead = min(mon[0] for mon in sp.Poly(
        sp.expand(lap_terms * L1), EPS).monoms())
    check("lap A1 enters the average first at O(eps^3)", lap_lead >= 3)

    # ---- exact-in-w identities with slaving (fundamental only)
    w = sp.symbols("w", real=True)
    A1s = sp.sqrt(sp.Rational(4, 3) * (R - w**2))
    E0 = sp.Rational(1, 4) * w**2 * A1**2 - R / 4 * A1**2 \
        + sp.Rational(3, 32) * A1**4
    E0s = sp.simplify(E0.subs(A1, A1s))
    check("local ledger collapses to -(R - w^2)^2/6",
          sp.simplify(E0s + (R - w**2)**2 / 6) == 0)
    check("roll baseline -R^2/6",
          sp.simplify(E0s.subs(w, 0) + R**2 / 6) == 0)

    alpha = sp.simplify(sp.diff(E0s, w, 2).subs(w, 0))
    check("alpha = E0''(0) = 2R/3",
          sp.simplify(alpha - sp.Rational(2, 3) * R) == 0)
    G2 = sp.simplify(2 * (E0s - E0s.subs(w, 0)) / alpha)
    check("G^2 = w^2 - w^4/(2R)",
          sp.simplify(G2 - (w**2 - w**4 / (2 * R))) == 0)

    # grad A1 carries an explicit factor of w
    dA1dw = sp.simplify(sp.diff(A1s, w))
    check("dA1/dw vanishes at w = 0 (grad A1 dies at band center)",
          sp.simplify(dA1dw.subs(w, 0)) == 0)

    eta = sp.simplify(A1s.subs(w, 0)**2 / 2)
    check("eta = A1^2(k_B)/2 = 2R/3",
          sp.simplify(eta - sp.Rational(2, 3) * R) == 0)
    check("beta = sqrt(alpha/eta) = 1",
          sp.simplify(sp.sqrt(alpha / eta)) == 1)

    if with_a3:
        # two-mode bending addition at band center: (9/4) A3^2 D^2
        A3sym = modes[1][1]
        add = sp.expand(expect_bend).coeff(D**2).coeff(A3sym**2)
        check("two-mode bending addition coefficient 9/4",
              sp.simplify(add - sp.Rational(9, 4) * EPS**2) == 0)

    print("  full averaged density at O(eps^2), band center, slaved:")
    print("    -R^2/6 + (1/2)(2R/3) G^2 + (1/2)(2R/3) (div k)^2")


# ----------------------------------------------------------------------
# Part B: harmonic projection ledger and comparison table
# ----------------------------------------------------------------------
def part_b(r_values, exact):
    print("\n============ Part B: harmonic projection ledger ============")
    A1, A3, k = sp.symbols("A1 A3 k", real=True)

    u = A1 * sp.cos(TH) + A3 * sp.cos(3 * TH)
    lin = (R - (1 - k**2)**2) * A1 * sp.cos(TH) \
        + (R - (1 - 9 * k**2)**2) * A3 * sp.cos(3 * TH)
    resid = sp.expand(lin - u**3)

    C = {n: sp.expand(cosine_coefficient(resid, n)) for n in (1, 3, 5, 7, 9)}
    print("  ledger:")
    for n in (1, 3, 5, 7, 9):
        print(f"    C{n} = {sp.factor(C[n])}")

    check("C5 = -(3/4) A1 A3 (A1 + A3)",
          sp.simplify(C[5] + sp.Rational(3, 4) * A1 * A3 * (A1 + A3)) == 0)
    check("C7 = -(3/4) A1 A3^2",
          sp.simplify(C[7] + sp.Rational(3, 4) * A1 * A3**2) == 0)
    check("C9 = -(1/4) A3^3",
          sp.simplify(C[9] + sp.Rational(1, 4) * A3**3) == 0)

    # variational identity against the averaged energy
    E0 = sp.Rational(1, 4) * (1 - k**2)**2 * A1**2 \
        + sp.Rational(1, 4) * (1 - 9 * k**2)**2 * A3**2 \
        - R / 4 * (A1**2 + A3**2) \
        + sp.Rational(1, 4) * (sp.Rational(3, 8) * A1**4
                               + sp.Rational(3, 8) * A3**4
                               + sp.Rational(3, 2) * A1**2 * A3**2
                               + sp.Rational(1, 2) * A1**3 * A3)
    check("C1 = -2 dE0/dA1",
          sp.simplify(C[1] + 2 * sp.diff(E0, A1)) == 0)
    check("C3 = -2 dE0/dA3",
          sp.simplify(C[3] + 2 * sp.diff(E0, A3)) == 0)

    # closed forms
    A3_full = A1**3 / (4 * (R - (1 - 9 * k**2)**2
                            - sp.Rational(3, 2) * A1**2))
    A3_lr = A1**3 / (4 * (R - (1 - 9 * k**2)**2))
    A1s = sp.sqrt(sp.Rational(4, 3) * (R - (k**2 - 1)**2))
    A3_full_slv = sp.simplify(A3_full.subs(A1, A1s))
    check("full linearized + slaving = -A1^3/(4(R + 79 k^4 - 14 k^2 - 1))",
          sp.simplify(A3_full_slv
                      + A1s**3 / (4 * (R + 79 * k**4 - 14 * k**2 - 1))) == 0)
    reldiff = sp.simplify((A3_lr - A3_full) / A3_full)
    reldiff_k1 = sp.simplify(reldiff.subs({A1: A1s, k: 1}))
    check("closed forms differ by 2R/(64 - R) at k = 1 "
          "(linear response above full linearized)",
          sp.simplify(reldiff_k1 - 2 * R / (64 - R)) == 0)

    if not exact:
        print("  (exact two-mode solve skipped; pass --exact to run)")
        return

    print("  exact two-mode solve at k = 1 (comparison table):")
    header = ("    R      A3/A1 exact   full-lin err   lin-resp err   "
              "mutual diff")
    print(header)
    for Rv in r_values:
        subs0 = {R: sp.Float(Rv), k: 1}
        guess1 = float((4 * Rv / 3) ** 0.5)
        sols = sp.nsolve([C[1].subs(subs0), C[3].subs(subs0)],
                         [A1, A3], [guess1, -0.003], prec=30)
        a1e, a3e = float(sols[0]), float(sols[1])
        a1l = guess1
        fullv = float(sp.N(A3_full.subs({A1: a1l, **subs0})))
        lrv = float(sp.N(A3_lr.subs({A1: a1l, **subs0})))
        print("    %.2f   %+.4e    %+7.2f %%      %+7.2f %%      %6.2f %%"
              % (Rv, a3e / a1e,
                 100 * (fullv - a3e) / a3e,
                 100 * (lrv - a3e) / a3e,
                 100 * abs((fullv - lrv) / fullv)))
        # negligibility ledger
        a3ratio2 = (a3e / a1e) ** 2
        a5 = -0.75 * a1l**2 * lrv / 576.0
        fb = abs(0.75 * a1l**2 * a5 / ((Rv - 64.0) * lrv))
        print("           (A3/A1)^2 = %.1e ; |A5| ~ %.1e ; "
              "A5 feedback shifts A3 at rel ~ %.1e"
              % (a3ratio2, abs(a5), fb))


# ----------------------------------------------------------------------
# args (per project conventions: _Args at the bottom + CLI mirror)
# ----------------------------------------------------------------------
class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    with_a3 = True     # include the third harmonic in Part A
    exact = True       # run the exact two-mode solve in Part B
    r_values = (0.5, 1.0)


def main(args):
    part_a(with_a3=args.with_a3)
    part_b(r_values=args.r_values, exact=args.exact)
    n_fail = sum(1 for _, ok in CHECKS if not ok)
    print("\n%d checks, %d failed" % (len(CHECKS), n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--with-a3", dest="with_a3", action="store_true",
                   default=_Args.with_a3)
    p.add_argument("--no-a3", dest="with_a3", action="store_false")
    p.add_argument("--exact", dest="exact", action="store_true",
                   default=_Args.exact)
    p.add_argument("--no-exact", dest="exact", action="store_false")
    p.add_argument("--R", dest="r_values", type=float, nargs="+",
                   default=list(_Args.r_values))
    sys.exit(main(p.parse_args()))
