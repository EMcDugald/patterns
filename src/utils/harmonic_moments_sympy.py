# src/utils/harmonic_moments_sympy.py
"""
Python (SymPy) counterpart of harmonic_moments.wl.

Three roles:

  1. Symbolic verification of the moment ladder for the odd-harmonic
     stripe ansatz  u = sum_n A_n cos(n theta), theta = kx x + ky y:

         M_m := <u Laplacian^m u> = (-1)^m (K^m / 2) sum_n n^(2m) A_n^2,

     the uHu wavenumber bias factor, and the {1,3} de-mixing quadratic

         9 M0 K^2 + 10 M1 K + M2 = 0        (M1 < 0),

     whose physical root is the LARGER K (the smaller root K/9 is the
     alias reading all energy as third harmonic of a lower fundamental).

  2. Symbolic Galerkin projection of the stationary Swift-Hohenberg
     equation  (R - (1 + Lap)^2) u - u^3 = 0  onto cos(n theta),
     giving the coupled algebraic system for the A_n and the
     hierarchical recurrence A1^2 = (4/3)(R - (1-K)^2),
     A3 = A3(A1; K, R), A5 = A5(A1, A3; K, R) as rational functions.

  3. Importable NUMERIC helpers used by harmonic_extraction_test.py:
         solve_moments_13(M0, M1, M2)   -> K, A1, A3 (vectorized)
         uhu_bias_factor(amps, harmonics)
         sh_roll_amplitudes(R, K, harmonics) -> exact truncated-roll A_n
         synthesize_roll(theta, amps, harmonics) -> u field

Run as a script to print the symbolic derivations; import for the helpers.
All symbolic work is done in the single fast-phase variable t (valid
because theta is linear; curvature corrections enter only through
sin(n theta) terms that phase-average to zero against u -- verified
separately in the Mathematica script, Part 1b).
"""

import numpy as np
import sympy as sp


# ---------------------------------------------------------------------
# Symbolic core: products of cosines via the complex-exponential route
# ---------------------------------------------------------------------

t = sp.Symbol("t", real=True)          # fast phase theta
Ksym = sp.Symbol("K", positive=True)   # K = |k|^2
Rsym = sp.Symbol("R", positive=True)
k1sym = sp.Symbol("k1", real=True)     # x-component of k (theta = k.x)
k2sym = sp.Symbol("k2", real=True)     # y-component of k


def _amp_symbols(harmonics):
    return [sp.Symbol(f"A{n}", real=True) for n in harmonics]


def ansatz(harmonics, amps=None):
    amps = amps if amps is not None else _amp_symbols(harmonics)
    return sum(a * sp.cos(n * t) for a, n in zip(amps, harmonics)), amps


def lap_power(expr, m):
    """
    Laplacian^m acting on a trig polynomial in the fast phase t, for a
    LINEAR phase with |grad theta|^2 = K:
        Lap^m cos(n t) = (-K n^2)^m cos(n t)   (same for sin).
    Implemented by expanding into exp(i n t) modes.
    """
    e = sp.expand(expr.rewrite(sp.exp))
    out = 0
    for term in sp.Add.make_args(e):
        c, rest = term.as_independent(t)
        if rest == 1:
            out += c  # n = 0 mode: Laplacian kills it for m >= 1
            if m >= 1:
                out -= c
            continue
        # rest = exp(i n t)
        n = sp.simplify(sp.log(rest) / (sp.I * t))
        out += c * (-Ksym * n**2) ** m * rest
    return sp.simplify(out.rewrite(sp.cos).expand(trig=True))


def phase_avg(expr):
    """Ideal phase average (1/2pi) Int_0^{2pi} expr dt."""
    return sp.simplify(sp.integrate(sp.expand_trig(expr), (t, 0, 2 * sp.pi)) / (2 * sp.pi))


def moments_symbolic(harmonics, mmax=None):
    """M_m = <u Lap^m u>, m = 0..mmax (default: len(harmonics))."""
    mmax = mmax if mmax is not None else len(harmonics)
    u, amps = ansatz(harmonics)
    return [phase_avg(u * lap_power(u, m)) for m in range(mmax + 1)], amps


def uHu_tensor_symbolic(harmonics):
    """
    The uHu tensor u*Hessian(u), BEFORE and AFTER phase averaging, for
    u = sum_n A_n cos(n theta), theta = k1 x + k2 y (so |grad theta|^2 =
    k1^2+k2^2 = K). Since theta is linear, every second partial reduces
    to the SAME scalar u_tt(t) := d^2u/dtheta^2, just scaled by a
    component of the outer product k (x) k:

        u_xx = k1^2 u_tt,   u_xy = k1 k2 u_tt,   u_yy = k2^2 u_tt

    so H_u = u_tt(t) * [[k1^2, k1 k2], [k1 k2, k2^2]] -- a scalar
    OSCILLATING prefactor times a FIXED rank-one matrix. Multiplying by u
    and phase-averaging kills every cross-harmonic term (same
    orthogonality as moments_symbolic) and leaves a tensor that is
    EXACTLY proportional to k (x) k -- i.e. exactly rank-one -- for ANY
    amplitude content sharing this one theta. (A second wavevector
    direction, not more harmonics of this one, is what actually breaks
    rank-one-ness; see the T_lam2 note below.)

    Returns dict with:
      u, u_tt                        -- theta-domain building blocks
      Hxx, Hxy, Hyy                  -- Hessian components (before avg)
      uHxx, uHxy, uHyy               -- u*H components (before avg, the
                                         raw oscillating field your code
                                         actually smooths)
      Txx, Txy, Tyy                  -- averaged tensor T = <u*H>
      T_lam1, T_lam2                 -- eigenvalues of T (lam2 = 0 exact)
    """
    u, amps = ansatz(harmonics)
    u_tt = sp.diff(u, t, 2)

    Hxx = sp.expand(k1sym**2 * u_tt)
    Hxy = sp.expand(k1sym * k2sym * u_tt)
    Hyy = sp.expand(k2sym**2 * u_tt)

    uHxx = sp.expand(u * Hxx)
    uHxy = sp.expand(u * Hxy)
    uHyy = sp.expand(u * Hyy)

    Txx = sp.simplify(phase_avg(uHxx))
    Txy = sp.simplify(phase_avg(uHxy))
    Tyy = sp.simplify(phase_avg(uHyy))

    # T is exactly rank-one (proportional to k1,k2 outer product): trace
    # is the nonzero eigenvalue, the perpendicular eigenvalue is exactly 0.
    T_lam1 = sp.simplify(Txx + Tyy)
    T_lam2 = sp.Integer(0)

    return {
        "u": u, "u_tt": u_tt,
        "Hxx": Hxx, "Hxy": Hxy, "Hyy": Hyy,
        "uHxx": uHxx, "uHxy": uHxy, "uHyy": uHyy,
        "Txx": Txx, "Txy": Txy, "Tyy": Tyy,
        "T_lam1": T_lam1, "T_lam2": T_lam2,
        "amps": amps,
    }


def harmonic_projection(expr, n):
    """Coefficient of cos(n t) in a trig polynomial (n >= 1)."""
    e = sp.expand(expr.rewrite(sp.exp))
    coeff = 0
    target = sp.exp(sp.I * n * t)
    for term in sp.Add.make_args(e):
        c, rest = term.as_independent(t)
        if sp.simplify(rest - target) == 0:
            coeff += c
    return sp.simplify(2 * coeff)  # real field: exp(+int) and exp(-int) split evenly


def galerkin_system(harmonics):
    """
    Project (R - (1+Lap)^2) u - u^3 onto cos(n t), n in harmonics.
    Returns list of residual expressions (== 0) and the amplitude symbols.
    Linear part: L_n A_n with L_n = R - (1 - n^2 K)^2.
    """
    u, amps = ansatz(harmonics)
    cubic = sp.expand(u**3)
    residuals = []
    for a, n in zip(amps, harmonics):
        Ln = Rsym - (1 - n**2 * Ksym) ** 2
        residuals.append(sp.simplify(Ln * a - harmonic_projection(cubic, n)))
    return residuals, amps


def hierarchical_recurrence(harmonics=(1, 3, 5)):
    """
    Solve the Galerkin system hierarchically:
        A1^2 from the cos(t) equation with higher amps = 0,
        A3 from the cos(3t) equation linearized in A3 (A5 = 0),
        A5 from the cos(5t) equation linearized in A5.
    Returns dict of sympy expressions.
    """
    residuals, amps = galerkin_system(list(harmonics))
    A1 = amps[0]
    out = {}

    r1 = residuals[0].subs({a: 0 for a in amps[1:]})
    r1 = sp.cancel(r1 / A1)              # discard the trivial A1 = 0 root
    sol = sp.solve(sp.Eq(r1, 0), A1**2)
    out["A1_sq"] = sp.simplify(sol[0]) if isinstance(sol, list) else sol

    if len(harmonics) >= 2:
        A3 = amps[1]
        r3 = residuals[1].subs({a: 0 for a in amps[2:]})
        r3_lin = r3.series(A3, 0, 2).removeO()
        out["A3"] = sp.simplify(sp.solve(sp.Eq(r3_lin, 0), A3)[0])

    if len(harmonics) >= 3:
        A5 = amps[2]
        r5 = residuals[2]
        r5_lin = r5.series(A5, 0, 2).removeO()
        out["A5"] = sp.simplify(sp.solve(sp.Eq(r5_lin, 0), A5)[0])

    return out


# ---------------------------------------------------------------------
# Numeric helpers (importable; no sympy at call time)
# ---------------------------------------------------------------------

def uhu_bias_factor(amps, harmonics):
    """k^2_uHu / K_true = sum n^2 A_n^2 / sum A_n^2 (>= 1)."""
    amps = np.asarray(amps, float)
    ns = np.asarray(harmonics, float)
    return float(np.sum(ns**2 * amps**2) / np.sum(amps**2))


def solve_moments_13(M0, M1, M2, prefer="larger"):
    """
    Pointwise de-mixing for the {1,3} ansatz. Vectorized over arrays.

        M0 = (s1 + s3)/2
        M1 = -(K/2)(s1 + 9 s3)      (= lambda1 when T is rank-one)
        M2 = (K^2/2)(s1 + 81 s3)    (= <(Lap u)^2>)

    K solves 9 M0 K^2 + 10 M1 K + M2 = 0. Both roots are positive when
    the moments are consistent; the physical root is the larger one
    (smaller root is the K/9 third-harmonic alias). Returns (K, A1, A3)
    with NaN where no admissible root exists (s1 or s3 < 0 within a
    small tolerance is clipped; grossly negative -> NaN).
    """
    M0 = np.asarray(M0, float)
    M1 = np.asarray(M1, float)
    M2 = np.asarray(M2, float)

    a, b, c = 9.0 * M0, 10.0 * M1, M2
    disc = b * b - 4.0 * a * c
    bad = (disc < 0) | (np.abs(a) < 1e-300)
    disc = np.where(bad, 0.0, disc)
    sq = np.sqrt(disc)
    roots = np.stack([(-b + sq) / (2 * a), (-b - sq) / (2 * a)], axis=0)
    roots = np.where(roots > 0, roots, np.nan)

    def amps_from_K(Kr):
        with np.errstate(invalid="ignore", divide="ignore"):
            s3 = -(M1 + Kr * M0) / (4.0 * Kr)
            s1 = 2.0 * M0 - s3
        return s1, s3

    def admissible(Kr, tol_frac=0.05):
        s1, s3 = amps_from_K(Kr)
        floor = -tol_frac * 2.0 * np.abs(M0)
        return np.isfinite(Kr) & (s1 >= floor) & (s3 >= floor)

    ok = np.stack([admissible(roots[0]), admissible(roots[1])], axis=0)
    if prefer == "larger":
        first, second = 0, 1  # roots[0] is the larger root (a > 0)
    else:
        first, second = 1, 0
    K = np.where(ok[first], roots[first],
                 np.where(ok[second], roots[second], np.nan))
    K = np.where(bad, np.nan, K)

    s1, s3 = amps_from_K(K)
    A1 = np.sqrt(np.clip(s1, 0.0, None))
    A3 = np.sqrt(np.clip(s3, 0.0, None))
    return K, A1, A3


_GALERKIN_CACHE = {}


def _galerkin_fns(harmonics):
    """Cache lambdified residuals F_n(R, K, A...) per harmonics tuple."""
    key = tuple(harmonics)
    if key not in _GALERKIN_CACHE:
        residuals, amps = galerkin_system(list(key))
        fns = [sp.lambdify([Rsym, Ksym] + amps, r, "numpy")
               for r in residuals]
        _GALERKIN_CACHE[key] = fns
    return _GALERKIN_CACHE[key]


def sh_roll_amplitudes(R, K, harmonics=(1, 3, 5)):
    """
    Exact amplitudes of the truncated stationary SH roll at parameters
    (R, K): solve the Galerkin polynomial system numerically, seeded by
    the hierarchical recurrence. Returns array of A_n (A1 > 0 branch).
    """
    from scipy.optimize import fsolve

    harmonics = list(harmonics)
    fns = _galerkin_fns(harmonics)

    def F(a):
        return np.array([f(float(R), float(K), *a) for f in fns], float)

    # seed from the hierarchy
    L = lambda n: R - (1 - n**2 * K) ** 2
    A1_0 = np.sqrt(max((4.0 / 3.0) * L(1), 1e-12))
    seed = [A1_0]
    if len(harmonics) >= 2:
        seed.append(A1_0**3 / (4.0 * L(3)))
    if len(harmonics) >= 3:
        seed.append(0.0)
    seed = np.array(seed[: len(harmonics)])

    sol, info, ier, msg = fsolve(F, seed, full_output=True)
    if ier != 1 or sol[0] <= 0:
        raise RuntimeError(f"SH roll solve failed at R={R}, K={K}: {msg}")
    return sol


def synthesize_roll(theta, amps, harmonics=(1, 3, 5)):
    """u(x,y) = sum_n A_n cos(n theta) for a given phase field theta."""
    u = np.zeros_like(np.asarray(theta, float))
    for a, n in zip(amps, harmonics):
        u += a * np.cos(n * theta)
    return u


# ---------------------------------------------------------------------
# Script mode: print the derivations
# ---------------------------------------------------------------------

def _main():
    sp.init_printing(use_unicode=False)

    harmonics = [1, 3]
    print("=== moment ladder, harmonics", harmonics, "===")
    Ms, amps = moments_symbolic(harmonics)
    for m, M in enumerate(Ms):
        print(f"  M_{m} =", M)
    closed = [sp.Rational((-1) ** m, 2) * Ksym**m
              * sum(n ** (2 * m) * a**2 for a, n in zip(amps, harmonics))
              for m in range(len(Ms))]
    ok = all(sp.simplify(Ms[m] - closed[m]) == 0 for m in range(len(Ms)))
    print("  closed-form  M_m = (-1)^m (K^m/2) sum n^(2m) A_n^2 :", ok)

    bias = sp.simplify(-Ms[1] / (Ksym * Ms[0]))
    print("  uHu bias factor k^2_uHu/K =", bias)

    print("\n=== uHu tensor (harmonics", harmonics, "), before and after averaging ===")
    tens = uHu_tensor_symbolic(harmonics)
    print("  u        =", tens["u"])
    print("  u_tt     =", tens["u_tt"], "   (u_tt := d^2u/dtheta^2)")
    print("  H_xx = k1^2 u_tt =", tens["Hxx"])
    print("  H_xy = k1 k2 u_tt =", tens["Hxy"])
    print("  H_yy = k2^2 u_tt =", tens["Hyy"])
    print("  (u H)_xx  before averaging =", tens["uHxx"])
    print("  (u H)_xy  before averaging =", tens["uHxy"])
    print("  (u H)_yy  before averaging =", tens["uHyy"])
    print("  T_xx = <(u H)_xx> =", tens["Txx"])
    print("  T_xy = <(u H)_xy> =", tens["Txy"])
    print("  T_yy = <(u H)_yy> =", tens["Tyy"])
    print("  T eigenvalues: lam1 =", tens["T_lam1"], "  lam2 =", tens["T_lam2"],
          " (lam2 = 0 EXACTLY -- T stays rank-one for any harmonic content")
    print("  sharing this one theta; a genuinely different wavevector")
    print("  direction is what actually makes lam2 != 0.)")

    print("\n=== {1,3} de-mixing quadratic ===")
    M0s, M1s, M2s = sp.symbols("M0 M1 M2", real=True)
    s1, s3, Kq = sp.symbols("s1 s3 Kq", positive=True)
    eqs = [sp.Eq(M0s, (s1 + s3) / 2),
           sp.Eq(M1s, -(Kq / 2) * (s1 + 9 * s3)),
           sp.Eq(M2s, (Kq**2 / 2) * (s1 + 81 * s3))]
    elim = sp.groebner([e.lhs - e.rhs for e in eqs], s1, s3, Kq,
                       order="lex").exprs[-1]
    print("  eliminant in K:", sp.factor(elim), " == 0")
    print("  (expect proportional to 9 M0 K^2 + 10 M1 K + M2)")

    print("\n=== stationary SH Galerkin system, harmonics {1,3,5} ===")
    residuals, amps135 = galerkin_system([1, 3, 5])
    for n, r in zip([1, 3, 5], residuals):
        print(f"  cos({n}t):  ", sp.expand(r), " == 0")

    print("\n=== hierarchical recurrence ===")
    rec = hierarchical_recurrence((1, 3, 5))
    for k, v in rec.items():
        print(f"  {k} =", v)

    print("\n=== numeric example: exact truncated roll at R=0.5, K=1 ===")
    a = sh_roll_amplitudes(0.5, 1.0, (1, 3, 5))
    print("  A =", a)
    print("  uHu bias factor on this roll:", uhu_bias_factor(a, (1, 3, 5)))
    print("  fractional k error if single harmonic assumed:",
          np.sqrt(uhu_bias_factor(a, (1, 3, 5))) - 1.0)

    print("\n=== round trip: moments of that roll -> solve_moments_13 ===")
    ns = np.array([1, 3, 5], float)
    Ktrue = 1.0
    M0 = 0.5 * np.sum(a**2)
    M1 = -0.5 * Ktrue * np.sum(ns**2 * a**2)
    M2 = 0.5 * Ktrue**2 * np.sum(ns**4 * a**2)
    Kr, A1r, A3r = solve_moments_13(M0, M1, M2)
    print(f"  recovered K={Kr:.6f}  A1={A1r:.6f}  A3={A3r:.6f}")
    print(f"  truth      K={Ktrue:.6f}  A1={a[0]:.6f}  A3={abs(a[1]):.6f}")
    print("  (small residual error is the neglected A5 leaking into the "
          "{1,3} de-mixing -- the model-truncation floor, not a bug)")


if __name__ == "__main__":
    _main()
