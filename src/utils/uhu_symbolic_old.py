# src/utils/uhu_symbolic.py
"""
Symbolic helper for pushing stripe-pattern ansatze through the uHu
machinery: exact phases, wavevectors, first/second derivatives and their
squares, all uHu tensor components (DC + oscillatory parts separately),
the window field W, |k|, div k, curl k, J = det(grad k), the RCN phase
energy density, and local-plane-wave monogenic/Riesz expressions --
plus composable averaging operators (ideal phase average, second-order
kernel expansion with anisotropic sigma, exact/numeric 1D Gaussian).

Conventions match the numerical codebase (kfield_calcs.kfield_diagnostics
and riesz.riesz_analyze_pattern):

    k = grad(theta)
    div_k  = k1_x + k2_y          curl_k = k2_x - k1_y  (0 for exact phases)
    J      = k1_x k2_y - k1_y k2_x   (Jacobian determinant / twist)
    E_RCN  = (div k)^2 + (1 - |k|^2)^2
    Riesz  R_j = F^-1[-i (xi_j/|xi|) F u]  =>  (R1, R2) ~ A sin(theta) khat
           (local plane-wave approximation; exact Riesz is nonlocal)

Ansatze:
    GeneralAnsatz : u = A(x,y) cos(theta(x,y)), both unspecified
    KneeAnsatz    : A = 1, theta = k1 x + log(2 cosh(k2 y))
    PNAnsatz      : A = 1, theta = y - log(psi), psi = a + b erf(x/sqrt(2 y)),
                    valid on the y > 0 branch; full plane by the exact
                    symmetry theta(x,-y) = -theta(x,y)  (thx odd, thy even)

Averaging operators (all take/return sympy expressions):
    ideal_avg(expr)                : drop every trig atom carrying the fast
                                     phase (exact orthogonality step; the
                                     same algebra as the moment ladder,
                                     filter-shape agnostic)
    kernel_expand(expr, sx, sy)    : expr + (sx^2/2) expr_xx + (sy^2/2) expr_yy
                                     (second-order anisotropic-Gaussian
                                     kernel expansion; isotropic: sx = sy)
    gaussian_avg_y(expr, sigma)    : exact 1D Gaussian y-average as a sympy
                                     Integral; .doit() attempted, and a
                                     Gauss-Hermite lambdified evaluator is
                                     always returned alongside
    corrugation_factor(sx, sy)     : exp(-2 (sx^2 thx^2 + sy^2 thy^2)),
                                     the frozen-phase survival factor of the
                                     2 theta piece under an axis-aligned
                                     anisotropic Gaussian (exact for a
                                     locally plane wave; exact globally for
                                     the knee's x-average on the core)

Eigen-analysis (closed 2x2 forms only; sympy's generic eigenvects is NOT
used, it does not simplify on these entries):
    eig2x2(Txx, Txy, Tyy)          : lam_pm = (tr +- sqrt((Txx-Tyy)^2
                                     + 4 Txy^2))/2, axis angle
                                     = (1/2) atan2(2 Txy, Txx - Tyy)
    eig2x2_numeric(...)            : same formulas on numpy arrays, so the
                                     symbolic and numeric conventions cannot
                                     drift apart
    GeneralAnsatz.eigensystem(source=...) : eigenpairs of the uHu tensor for
                                     source in {full, corrugated, kernel, DC}
                                     or a caller-supplied tensor
    GeneralAnsatz.angle_error_vs_truth(...) : dominant-axis angle minus
                                     atan2(theta_y, theta_x), wrapped mod pi

Sign convention (matches compute_uhu_ops_smoother, whose eigh returns
ascending eigenvalues): uHu is negative semidefinite for phase-dominated
fields, so the DOMINANT eigenpair is the MOST NEGATIVE one, and chi =
lam_subdominant / lam_dominant is the rank-one validity diagnostic.

Numeric PN chains: pn_chain_lnpsi() evaluates ln psi and its derivatives
in the gauge-stabilized form (saturated erf parts telescope exactly to
e^{-m(x) pi}), which removes the catastrophic cancellation that corrupts
naive left-to-right chains beyond ~11 defects.

Everything returns plain sympy expressions so results can be piped into
sp.latex(...) for the paper or sp.lambdify(...) for numerics. The demo
(python uhu_symbolic.py) prints the full catalog for the knee and PN and
verifies the closed-form knee line energy E = 8 k2^3 / 3.
"""

import argparse

import numpy as np
import sympy as sp
from scipy.special import erfcx
from sympy.simplify.fu import TR8


# =====================================================================
# Closed-form 2x2 symmetric eigensystem (symbolic and numeric)
# =====================================================================

def eig2x2(Txx, Txy, Tyy, dominant="min"):
    """
    Eigenpairs of the symmetric 2x2 tensor [[Txx, Txy], [Txy, Tyy]] in
    closed form. Sympy expressions in, sympy expressions out.

    Writing the traceless part as (d/2) [[cos 2p, sin 2p], [sin 2p,
    -cos 2p]] with d = sqrt((Txx - Tyy)^2 + 4 Txy^2) >= 0 gives
    immediately that (cos p, sin p) is the eigenvector of lam_plus and
    (-sin p, cos p) that of lam_minus, with
    p = (1/2) atan2(2 Txy, Txx - Tyy).

    `dominant` selects which branch is reported as dominant: "min" (the
    pipeline convention, since uHu is negative semidefinite) or "max".
    chi = lam_sub / lam_dom is the rank-one validity diagnostic, matching
    compute_uhu_ops_smoother's chi = lam2 / lam1.
    """
    tr = Txx + Tyy
    disc = sp.sqrt((Txx - Tyy)**2 + 4 * Txy**2)
    lam_plus = (tr + disc) / 2
    lam_minus = (tr - disc) / 2
    ang_plus = sp.atan2(2 * Txy, Txx - Tyy) / 2
    ang_minus = ang_plus + sp.pi / 2

    if dominant == "min":
        lam_dom, lam_sub, ang_dom = lam_minus, lam_plus, ang_minus
    elif dominant == "max":
        lam_dom, lam_sub, ang_dom = lam_plus, lam_minus, ang_plus
    else:
        raise ValueError("dominant must be 'min' or 'max'")

    return {
        "lam_plus": lam_plus, "lam_minus": lam_minus,
        "angle_plus": ang_plus, "angle_minus": ang_minus,
        "lam_dom": lam_dom, "lam_sub": lam_sub, "angle_dom": ang_dom,
        "vec_dom": sp.Matrix([sp.cos(ang_dom), sp.sin(ang_dom)]),
        "chi": lam_sub / lam_dom,
        "trace": tr, "disc": disc,
    }


def eig2x2_numeric(Txx, Txy, Tyy, dominant="min"):
    """Identical formulas on numpy arrays. Kept in the same module as the
    symbolic version on purpose: the two conventions cannot drift apart if
    they are written once, side by side."""
    Txx, Txy, Tyy = np.asarray(Txx), np.asarray(Txy), np.asarray(Tyy)
    tr = Txx + Tyy
    disc = np.sqrt((Txx - Tyy)**2 + 4 * Txy**2)
    lam_plus = (tr + disc) / 2
    lam_minus = (tr - disc) / 2
    ang_plus = 0.5 * np.arctan2(2 * Txy, Txx - Tyy)
    ang_minus = ang_plus + np.pi / 2
    if dominant == "min":
        lam_dom, lam_sub, ang_dom = lam_minus, lam_plus, ang_minus
    elif dominant == "max":
        lam_dom, lam_sub, ang_dom = lam_plus, lam_minus, ang_plus
    else:
        raise ValueError("dominant must be 'min' or 'max'")
    with np.errstate(invalid="ignore", divide="ignore"):
        chi = lam_sub / np.where(np.abs(lam_dom) < 1e-300, np.nan, lam_dom)
    return {"lam_plus": lam_plus, "lam_minus": lam_minus,
            "angle_plus": ang_plus, "angle_minus": ang_minus,
            "lam_dom": lam_dom, "lam_sub": lam_sub, "angle_dom": ang_dom,
            "d1": np.cos(ang_dom), "d2": np.sin(ang_dom), "chi": chi}


def lambdify_np(args, expr):
    """
    Array-safe lambdify. modules="numpy" alone is wrong for anything built
    on the PN phase: numpy has no erf, so lambdify falls back to a module
    set whose atan2 is math.atan2, and the generated function then raises
    "only 0-dimensional arrays can be converted to Python scalars" the first
    time it is handed a grid. Putting scipy first maps erf to
    scipy.special.erf and leaves everything else on numpy.
    """
    return sp.lambdify(args, expr, modules=["scipy", "numpy"])


def is_zero(expr, tries=None):
    """
    Robust "is this expression identically zero" check.

    sp.simplify alone is not enough here: it routinely stalls on the
    hyperbolic Pythagorean identity that the knee's discriminant produces
    (lam_plus lands as k1^2 (sinh^2 - cosh^2 + 1) / (4 cosh^2), which IS
    zero but is not recognized). The fallbacks rewrite in exponentials and
    run trigsimp, which close every stall met so far. Returns a bool; use
    it for asserts, not inside expression pipelines.
    """
    if tries is None:
        tries = (sp.simplify,
                 lambda e: sp.simplify(sp.simplify(e)),      # closes the knee stall
                 lambda e: sp.trigsimp(sp.simplify(e)),
                 lambda e: sp.simplify(sp.simplify(e).rewrite(sp.tanh)),
                 lambda e: sp.simplify(e.rewrite(sp.exp)),
                 lambda e: sp.simplify(sp.expand(sp.trigsimp(e))))
    for f in tries:
        try:
            if f(expr) == 0:
                return True
        except Exception:
            continue
    return False


def wrap_mod_pi(expr, numeric=False):
    """Wrap an angle difference into (-pi/2, pi/2] via the doubled-angle
    trick, which is the only branch-safe way to do a mod-pi comparison of
    director-valued quantities."""
    if numeric:
        return 0.5 * np.arctan2(np.sin(2 * expr), np.cos(2 * expr))
    return sp.atan2(sp.sin(2 * expr), sp.cos(2 * expr)) / 2


# =====================================================================
# Base class
# =====================================================================

class GeneralAnsatz:
    """u = A(x, y) cos(theta(x, y)); A, theta symbolic functions."""

    def __init__(self):
        self.x, self.y = sp.symbols("x y", real=True)
        self.A = sp.Function("A")(self.x, self.y)
        self.theta = sp.Function("theta")(self.x, self.y)
        self._finalize()

    def _finalize(self):
        x, y = self.x, self.y
        self.u = self.A * sp.cos(self.theta)
        self.thx = sp.diff(self.theta, x)
        self.thy = sp.diff(self.theta, y)
        self.k = sp.Matrix([self.thx, self.thy])
        self.hess = sp.Matrix([
            [sp.diff(self.theta, x, 2), sp.diff(self.theta, x, y)],
            [sp.diff(self.theta, x, y), sp.diff(self.theta, y, 2)],
        ])
        self.k_mag2 = self.thx**2 + self.thy**2
        self.div_k = self.hess[0, 0] + self.hess[1, 1]      # = Lap(theta)
        # curl_k (0 for any exact phase) and J = det(grad k) are stored RAW
        # and simplified lazily on first attribute access. Simplifying them
        # eagerly here hangs indefinitely once a, b carry concrete
        # exp(-pi) weights, and hangs worse with an amplitude factor; the
        # values are unchanged, only the moment of simplification moves.
        self._curl_k_raw = sp.diff(self.thy, x) - sp.diff(self.thx, y)
        # written out rather than self.hess.det(): sympy's Bareiss path runs
        # cancellation on the entries and does not terminate once they carry
        # concrete exp(-pi) weights. For 2x2 the explicit form is exact.
        self._J_raw = (self.hess[0, 0] * self.hess[1, 1]
                       - self.hess[0, 1] * self.hess[1, 0])
        self._curl_k_cache = None
        self._J_cache = None
        self.W = sp.sqrt(
            self.hess[0, 0]**2 + 2 * self.hess[0, 1]**2 + self.hess[1, 1]**2
        ) / self.k_mag2
        self.E_rcn = self.div_k**2 + (1 - self.k_mag2)**2

    @property
    def curl_k(self):
        """k2_x - k1_y; identically 0 for any exact phase. Simplified once,
        on first access."""
        if self._curl_k_cache is None:
            self._curl_k_cache = sp.simplify(self._curl_k_raw)
        return self._curl_k_cache

    @property
    def J(self):
        """det(grad k) = det(Hess theta). Simplified once, on first access."""
        if self._J_cache is None:
            self._J_cache = sp.simplify(self._J_raw)
        return self._J_cache

    @property
    def J_raw(self):
        """Unsimplified det(Hess theta), for lambdify without the
        simplification cost (identical function, different expression tree)."""
        return self._J_raw

    # ---- uHu tensor ------------------------------------------------
    def uhu(self, i, j, simplify=True):
        """
        u d_i d_j u split exactly into DC + oscillatory parts.
        Returns dict(full=..., DC=..., osc=...). The DC part equals
        (1/2)(A d_i d_j A - A^2 theta_i theta_j) identically.
        """
        c = {"x": self.x, "y": self.y}
        full = sp.expand(TR8(sp.expand(self.u * sp.diff(self.u, c[i], c[j]))))
        dc = self.ideal_avg(full)
        osc = sp.expand(full - dc)
        if simplify:
            dc, osc = sp.simplify(dc), sp.simplify(osc)
        return {"full": full, "DC": dc, "osc": osc}

    def uhu_dc(self, i, j):
        """
        DC component via the closed form (1/2)(A A_ij - A^2 theta_i theta_j)
        instead of the TR8 expansion. Verified identical to uhu(i,j)["DC"]
        for general A and theta (assert in _demo). Needed because the
        expansion route is prohibitively slow once A and theta are concrete
        messy expressions, e.g. tanh(r/xi) times the PN erf phase.
        """
        c = {"x": self.x, "y": self.y}
        return sp.Rational(1, 2) * (
            self.A * sp.diff(self.A, c[i], c[j])
            - self.A**2 * sp.diff(self.theta, c[i]) * sp.diff(self.theta, c[j])
        )

    def uhu_osc(self, i, j):
        """
        Oscillatory component in closed form, for A = 1:
            -(1/2)[cos(2 theta) theta_i theta_j + sin(2 theta) theta_ij].
        For general A the expansion route is used (the A-derivative terms
        do not collapse to a two-term form).
        """
        c = {"x": self.x, "y": self.y}
        if self.A == 1:
            ti, tj = sp.diff(self.theta, c[i]), sp.diff(self.theta, c[j])
            tij = sp.diff(self.theta, c[i], c[j])
            return -sp.Rational(1, 2) * (sp.cos(2 * self.theta) * ti * tj
                                         + sp.sin(2 * self.theta) * tij)
        return self.uhu(i, j, simplify=False)["osc"]

    def uhu_tensor_dc(self, closed=True):
        """2x2 sympy Matrix of the DC (ideal-averaged) uHu tensor."""
        f = self.uhu_dc if closed else (lambda i, j: self.uhu(i, j)["DC"])
        return sp.Matrix([[f("x", "x"), f("x", "y")],
                          [f("x", "y"), f("y", "y")]])

    # ---- tensors under each averaging model --------------------------
    def uhu_tensor(self, source="DC", sx=None, sy=None):
        """
        The uHu tensor under one of the averaging models of the handoff.
        The four sources form the ladder from no averaging to ideal:

          "full"        pointwise u d_i d_j u, nothing removed. For A = 1
                        this factors exactly as
                            -cos^2(theta) (k k^T + tan(theta) Hess(theta)),
                        so the Hessian contamination relative to the k k^T
                        scale is exactly W |tan theta|.
          "corrugated"  DC + corrugation_factor * osc: channel (1) only,
                        the frozen-phase model, exact for a locally plane
                        wave and exact globally for the knee's on-core
                        x-average.
          "kernel"      second-order kernel expansion applied to DC:
                        channel (2) only, ideal 2-theta removal followed by
                        finite-kernel DC mixing.
          "DC"          both channels removed. For A = 1 this is exactly
                        -(1/2) k k^T, rank one, dominant eigenvector = k.

        sx, sy default to symbols s_x, s_y when needed.
        """
        if source == "DC":
            return self.uhu_tensor_dc()
        if source == "full":
            return sp.Matrix([
                [self.uhu_dc("x", "x") + self.uhu_osc("x", "x"),
                 self.uhu_dc("x", "y") + self.uhu_osc("x", "y")],
                [self.uhu_dc("x", "y") + self.uhu_osc("x", "y"),
                 self.uhu_dc("y", "y") + self.uhu_osc("y", "y")]])
        if sx is None:
            sx = sp.symbols("s_x", positive=True)
        if sy is None:
            sy = sp.symbols("s_y", positive=True)
        if source == "corrugated":
            g = self.corrugation_factor(sx, sy)
            return sp.Matrix([
                [self.uhu_dc("x", "x") + g * self.uhu_osc("x", "x"),
                 self.uhu_dc("x", "y") + g * self.uhu_osc("x", "y")],
                [self.uhu_dc("x", "y") + g * self.uhu_osc("x", "y"),
                 self.uhu_dc("y", "y") + g * self.uhu_osc("y", "y")]])
        if source == "kernel":
            return sp.Matrix([
                [self.kernel_expand(self.uhu_dc("x", "x"), sx, sy),
                 self.kernel_expand(self.uhu_dc("x", "y"), sx, sy)],
                [self.kernel_expand(self.uhu_dc("x", "y"), sx, sy),
                 self.kernel_expand(self.uhu_dc("y", "y"), sx, sy)]])
        raise ValueError(f"unknown source {source!r}")

    # ---- eigensystem -------------------------------------------------
    def eigensystem(self, source="DC", sx=None, sy=None, tensor=None,
                    dominant="min", simplify=False):
        """
        Closed-form eigenpairs of the uHu tensor. `source` selects an
        averaging model (see uhu_tensor); alternatively pass `tensor` as a
        2x2 sympy Matrix, which is how numeric-averaged tensors enter (build
        the averaged components however you like, then hand them over here
        so the eigen convention is shared).

        Adds to the eig2x2 output:
          k_angle  : the dominant-axis angle, i.e. the wavevector estimate
                     direction
          k_mag_A1 : the magnitude estimate sqrt(-2 lam_dom), valid where
                     <u^2> = A^2/2 = 1/2, matching k = sqrt(-lam1 / <u^2>)
                     in compute_uhu_ops_smoother
        """
        T = self.uhu_tensor(source, sx, sy) if tensor is None else tensor
        eig = eig2x2(T[0, 0], T[0, 1], T[1, 1], dominant=dominant)
        if simplify:
            eig = {k: sp.simplify(v) for k, v in eig.items()}
        eig["tensor"] = T
        eig["source"] = source if tensor is None else "supplied"
        eig["k_angle"] = eig["angle_dom"]
        eig["k_mag_A1"] = sp.sqrt(-2 * eig["lam_dom"])
        return eig

    def truth_angle(self):
        """Ground-truth wavevector direction atan2(theta_y, theta_x)."""
        return sp.atan2(self.thy, self.thx)

    def angle_error_vs_truth(self, source="DC", sx=None, sy=None,
                             tensor=None, dominant="min", simplify=False):
        """
        Dominant-eigenvector angle minus the true wavevector angle, wrapped
        into (-pi/2, pi/2]. Wrapping is mod pi because the eigenvector is a
        director: +k and -k are the same estimate.
        """
        eig = self.eigensystem(source, sx, sy, tensor, dominant)
        err = wrap_mod_pi(eig["k_angle"] - self.truth_angle())
        return sp.simplify(err) if simplify else err

    def amplitude_contamination(self):
        """
        W_A = ||A Hess A||_F / (A^2 |grad theta|^2): the size of the
        amplitude term in the DC tensor relative to the phase term. This is
        the amplitude-side analogue of W and is identically zero for A = 1.
        Where W_A >~ 1 the dominant eigenvector of the DC tensor is set by
        the amplitude, not the phase, so no amount of averaging recovers a
        wavevector there.
        """
        Axx = sp.diff(self.A, self.x, 2)
        Axy = sp.diff(self.A, self.x, self.y)
        Ayy = sp.diff(self.A, self.y, 2)
        num = self.A * sp.sqrt(Axx**2 + 2 * Axy**2 + Ayy**2)
        return num / (self.A**2 * self.k_mag2)

    # ---- fast-phase detection --------------------------------------
    def _is_fast(self, trig_arg):
        """Does this trig argument carry the fast phase? Overridable.
        Default: contains theta (general) or depends on x (knee-like)."""
        return trig_arg.has(self.theta)

    def ideal_avg(self, expr):
        """Exact orthogonality step: zero every sin/cos atom whose
        argument carries the fast phase (2 theta terms and harmonics)."""
        expr = sp.expand(TR8(sp.expand(expr)))
        sub = {f: 0 for f in expr.atoms(sp.cos, sp.sin)
               if self._is_fast(f.args[0])}
        return expr.subs(sub)

    # ---- averaging operators ----------------------------------------
    def kernel_expand(self, expr, sx, sy=None):
        """Second-order anisotropic-Gaussian kernel expansion:
        <f>_(sx,sy) = f + (sx^2/2) f_xx + (sy^2/2) f_yy + O(sigma^4)."""
        sy = sx if sy is None else sy
        return (expr
                + sp.Rational(1, 2) * sx**2 * sp.diff(expr, self.x, 2)
                + sp.Rational(1, 2) * sy**2 * sp.diff(expr, self.y, 2))

    def gaussian_avg_y(self, expr, sigma):
        """Exact 1D Gaussian y-average. Returns (symbolic, numeric_fn):
        symbolic is the Integral (doit() attempted); numeric_fn(y0, **params)
        evaluates it by 120-point Gauss-Hermite quadrature."""
        s = sp.symbols("s_avg", real=True)
        g = sp.exp(-(s**2) / (2 * sigma**2)) / (sigma * sp.sqrt(2 * sp.pi))
        integ = sp.Integral(expr.subs(self.y, self.y + s) * g,
                            (s, -sp.oo, sp.oo))
        try:
            closed = integ.doit()
            if closed.has(sp.Integral):
                closed = integ
        except Exception:
            closed = integ

        free = sorted(
            (expr.free_symbols | {sigma}) - {self.x, self.y},
            key=lambda q: q.name)
        f = sp.lambdify([self.x, self.y] + free, expr, modules="numpy")
        gh_s, gh_w = np.polynomial.hermite_e.hermegauss(120)
        gh_w = gh_w / gh_w.sum()

        def numeric_fn(y0, x0=0.0, **params):
            sig_v = params.pop(str(sigma), params.pop("sigma", None))
            vals = f(x0, y0 + sig_v * gh_s,
                     *[params[q.name] if q.name in params else sig_v
                       for q in free])
            return float(np.sum(gh_w * vals))

        return closed, numeric_fn

    def corrugation_factor(self, sx, sy=None):
        """Frozen-phase survival of the 2 theta piece under an
        axis-aligned anisotropic Gaussian."""
        sy = sx if sy is None else sy
        return sp.exp(-2 * (sx**2 * self.thx**2 + sy**2 * self.thy**2))

    # ---- monogenic signal (local plane-wave approximation) -----------
    def monogenic_lpw(self):
        """(R1, R2, A_mono, orientation) in the local plane-wave
        approximation R ~ A sin(theta) khat. Valid to
        O(|grad A|/|k|, |grad khat|/|k|); exact Riesz is nonlocal --
        compare numerically via riesz.riesz_analyze_pattern."""
        kmag = sp.sqrt(self.k_mag2)
        R1 = self.A * sp.sin(self.theta) * self.thx / kmag
        R2 = self.A * sp.sin(self.theta) * self.thy / kmag
        A_mono = self.A
        orientation = sp.atan2(R2, R1)
        return {"R1": R1, "R2": R2, "A_mono": A_mono,
                "orientation": orientation}

    # ---- reporting ---------------------------------------------------
    def catalog(self, latex=False):
        """Ordered dict of everything, optionally latexed."""
        items = {
            "theta": self.theta, "u": self.u,
            "k1 = theta_x": self.thx, "k2 = theta_y": self.thy,
            "theta_xx": self.hess[0, 0], "theta_xy": self.hess[0, 1],
            "theta_yy": self.hess[1, 1],
            "k1^2": sp.simplify(self.thx**2),
            "k2^2": sp.simplify(self.thy**2),
            "|k|^2": sp.simplify(self.k_mag2),
            "div k": sp.simplify(self.div_k),
            "curl k": self.curl_k,
            "J = det(grad k)": self.J,
            "W": sp.simplify(self.W),
            "E_RCN": sp.simplify(self.E_rcn),
        }
        for ij in ("xx", "xy", "yy"):
            comp = self.uhu(ij[0], ij[1])
            items[f"uHu T_{ij} (DC)"] = comp["DC"]
            items[f"uHu T_{ij} (osc)"] = comp["osc"]
        mono = self.monogenic_lpw()
        items["Riesz R1 (LPW)"] = sp.simplify(mono["R1"])
        items["Riesz R2 (LPW)"] = sp.simplify(mono["R2"])
        if latex:
            items = {k: sp.latex(v) for k, v in items.items()}
        return items


# =====================================================================
# Knee
# =====================================================================

class KneeAnsatz(GeneralAnsatz):
    """theta = k1 x + log(2 cosh(k2 y)), A = 1.  mu = k2, k1 = sqrt(1-mu^2)."""

    def __init__(self):
        self.x, self.y = sp.symbols("x y", real=True)
        self.k1, self.k2 = sp.symbols("k_1 k_2", positive=True)
        self.A = sp.Integer(1)
        self.theta = self.k1 * self.x + sp.log(2 * sp.cosh(self.k2 * self.y))
        self._finalize()

    def _is_fast(self, trig_arg):
        # the fast phase carries x (theta = k1 x + slow(y))
        return trig_arg.has(self.x)

    def line_energy(self):
        """Closed-form RCN line energy per unit x under the self-dual
        constraint k1^2 + k2^2 = 1 (far-field |k| = 1, so the density
        decays and the transect integral converges).
        Returns (density, integral) = (2 k2^4 sech^4(k2 y), 8 k2^3 / 3)."""
        dens = sp.simplify(self.E_rcn.subs(self.k1**2, 1 - self.k2**2))
        I = sp.integrate(dens, (self.y, -sp.oo, sp.oo))
        return dens, sp.simplify(I)


# =====================================================================
# PN (single defect, y > 0 branch)
# =====================================================================

class PNAnsatz(GeneralAnsatz):
    """
    theta = y - log(psi), psi = a + b erf(x / sqrt(2 y)), y > 0 branch.
    Single defect at the origin: a = (1 + e^-pi)/2, b = (e^-pi - 1)/2.
    Multi-defect: psi is a sum of the same erf steps (heat-equation
    linearity) -- assemble numerically via lambdified single-step pieces.
    Full plane: theta(x,-y) = -theta(x,y), so thx odd / thy even in y.
    """

    def __init__(self, numeric_weights=False, amplitude="one", xi=None):
        self.x = sp.symbols("x", real=True)
        self.y = sp.symbols("y", positive=True)   # y > 0 branch
        if numeric_weights:
            self.a = sp.Rational(1, 2) * (1 + sp.exp(-sp.pi))
            self.b = sp.Rational(1, 2) * (sp.exp(-sp.pi) - 1)
        else:
            self.a, self.b = sp.symbols("a b", real=True)

        # amplitude: "one" is the phase-only field of the earlier sessions;
        # "tanh" inserts A(r) = tanh(r/xi), the model core profile of
        # handoff Section 8. The point is that the DC tensor becomes
        # (1/2)(A Hess A - A^2 k k^T): the A^2 prefactor kills the 1/r^2
        # blow-up of the phase term at the core, at the cost of a new
        # rank-two term A Hess A that has no wavevector interpretation.
        self.amplitude = amplitude
        if amplitude == "one":
            self.A = sp.Integer(1)
            self.xi = None
        elif amplitude == "tanh":
            self.xi = sp.symbols("xi", positive=True) if xi is None \
                else sp.nsimplify(xi)
            self.r = sp.sqrt(self.x**2 + self.y**2)
            self.A = sp.tanh(self.r / self.xi)
        else:
            raise ValueError("amplitude must be 'one' or 'tanh'")

        self.psi = self.a + self.b * sp.erf(self.x / sp.sqrt(2 * self.y))
        self.theta = self.y - sp.log(self.psi)
        self._finalize()

    def _is_fast(self, trig_arg):
        # fast phase carries theta's y-linear part; any trig atom whose
        # argument contains theta (or y linearly through it) is fast
        return trig_arg.has(self.y) or trig_arg.has(self.x)

    def heat_check(self):
        """psi_y = (1/2) psi_xx (Hopf-Cole)."""
        return sp.simplify(sp.diff(self.psi, self.y)
                           - sp.Rational(1, 2) * sp.diff(self.psi, self.x, 2))


# =====================================================================
# Numeric PN chains: gauge-stabilized ln psi  (handoff Section 8b)
# =====================================================================

def pn_chain_defects(nd, k1):
    """Defect positions and spacing for an nd-defect chain, matching
    field_diagnostics_viz.pn_chain_field: L = pi / k1, centered."""
    L = np.pi / k1
    return L * (np.arange(nd) - (nd - 1) / 2.0), L


def pn_chain_lnpsi(x, y, nd, k1, stabilized=True, exp_clip=700.0):
    """
    ln psi and its derivatives for an nd-defect Pismen-Nepomnyashchy chain
    on the y > 0 branch, with theta = y - ln psi.

    Naive form (stabilized=False), the one every earlier script used:

        psi = a + sum_j delta_j erf((x - x_j) / sqrt(2 y))
        a = (1 + e^{-nd pi}) / 2,  delta_j = (e^{-(j+1) pi} - e^{-j pi}) / 2

    Near defect m the partial sums telescope from O(1) down to e^{-m pi},
    so the float64 relative error in psi is ~ 1e-16 e^{+m pi} and theta is
    O(1) wrong beyond m ~ 11.7 (measured in the handoff against 60-digit
    mpmath: 1e-3 at defect 10, 0.46 at defect 12, 8.9 at defect 15).

    Stabilized form (default). Split each erf about its saturated value,
        erf_j = s_j - s_j erfc(|x - x_j| / w),   s_j = sign(x - x_j),
    and the saturated parts telescope EXACTLY:
        a + sum_j delta_j s_j = e^{-m(x) pi},   m(x) = #{j : x_j < x}.
    Hence
        psi = e^{-m pi} [ 1 + S ],
        S   = sum_j delta_j e^{m pi} (erf_j - s_j)
            = sum_j c e^{(m - j) pi} s_j erfc(|x - x_j| / w),
        c   = (1 - e^{-pi}) / 2,
    and ln psi = -m pi + log1p(S) with no cancellation: the exponentially
    large weight e^{(m-j) pi} always multiplies an erfc evaluated at
    |x - x_j| / w ~ (m - j) L / sqrt(2 y), which is exponentially small.
    Each term is computed in log space (erfcx for the tail) so the two
    exponentials never form separately.

    Derivatives are taken in the same gauge. m(x) is piecewise constant, so
    within a cell d(ln psi)/dx = S_x / (1 + S) exactly, and likewise in y;
    the jumps of m across x = x_j are cancelled identically by the jumps of
    s_j, so the assembled ln psi is continuous.

    Returns dict with lnpsi, theta, theta_x, theta_y, m, S, and a
    `clipped` flag reporting whether any log-space exponent hit exp_clip.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = np.broadcast_arrays(x, y)
    if np.any(y <= 0):
        raise ValueError("pn_chain_lnpsi is the y > 0 branch; mirror with "
                         "theta(x, -y) = -theta(x, y)")

    xd, _ = pn_chain_defects(nd, k1)
    w = np.sqrt(2.0 * y)

    if not stabilized:
        from scipy.special import erf as _erf
        a = 0.5 * (1.0 + np.exp(-np.pi * nd))
        thetas = np.arange(nd + 1) * np.pi
        psi = np.full_like(x, a)
        psi_x = np.zeros_like(x)
        psi_y = np.zeros_like(x)
        for j, xj in enumerate(xd):
            d = 0.5 * (np.exp(-thetas[j + 1]) - np.exp(-thetas[j]))
            z = (x - xj) / w
            psi += d * _erf(z)
            g = (2.0 / np.sqrt(np.pi)) * np.exp(-z**2)
            psi_x += d * g / w
            psi_y += d * g * (-z / (2.0 * y))
        lnpsi = np.log(psi)
        lnpsi_x = psi_x / psi
        lnpsi_y = psi_y / psi
        m = np.searchsorted(xd, x.ravel(), side="right").reshape(x.shape)
        S = np.expm1(lnpsi + m * np.pi)
        clipped = False
    else:
        m = np.searchsorted(xd, x.ravel(), side="right").reshape(x.shape)
        log_c = np.log(0.5 * (1.0 - np.exp(-np.pi)))
        S = np.zeros_like(x)
        S_x = np.zeros_like(x)
        S_y = np.zeros_like(x)
        clipped = False

        for j, xj in enumerate(xd):
            zs = (x - xj) / w                 # signed
            az = np.abs(zs)
            # s must use the SAME tie-break as m (searchsorted side="right",
            # i.e. x == x_j counts as being to the right). np.sign would
            # return 0 there and silently drop the term, which shows up as a
            # single-point jump of order 1 whenever a defect sits exactly on
            # a grid node. That is the common case, not a rare one.
            s = np.where(x >= xj, 1.0, -1.0)
            gain = (m - j) * np.pi            # the telescoped weight, in log

            # value: c e^{gain} s erfc(az),  log erfc = log erfcx - az^2
            e_val = log_c + gain + np.log(erfcx(az)) - az**2
            # d/dx of delta_j e^{m pi} erf_j  (sign is that of delta_j < 0)
            e_dx = log_c + gain + np.log(2.0 / (w * np.sqrt(np.pi))) - az**2
            # d/dy likewise, extra factor |zs| / (2 y)
            with np.errstate(divide="ignore"):
                e_dy = (log_c + gain + np.log(2.0 / np.sqrt(np.pi)) - az**2
                        + np.log(np.maximum(az, 1e-300)) - np.log(2.0 * y))

            for e in (e_val, e_dx, e_dy):
                if np.any(e > exp_clip):
                    clipped = True
            S += s * np.exp(np.minimum(e_val, exp_clip))
            S_x -= np.exp(np.minimum(e_dx, exp_clip))
            S_y += s * np.exp(np.minimum(e_dy, exp_clip))

        lnpsi = -m * np.pi + np.log1p(S)
        lnpsi_x = S_x / (1.0 + S)
        lnpsi_y = S_y / (1.0 + S)

    return {"lnpsi": lnpsi, "theta": y - lnpsi,
            "theta_x": -lnpsi_x, "theta_y": 1.0 - lnpsi_y,
            "m": m, "S": S, "clipped": clipped, "x_defects": xd}


def pn_chain_lnpsi_mpmath(x, y, nd, k1, dps=60):
    """60-digit reference implementation of the naive form, for validation
    only. Scalar in, scalar out."""
    import mpmath as mp
    with mp.workdps(dps):
        xd, _ = pn_chain_defects(nd, k1)
        a = (1 + mp.e**(-mp.pi * nd)) / 2
        psi = a
        w = mp.sqrt(2 * mp.mpf(y))
        for j, xj in enumerate(xd):
            d = (mp.e**(-mp.pi * (j + 1)) - mp.e**(-mp.pi * j)) / 2
            psi += d * mp.erf((mp.mpf(x) - mp.mpf(xj)) / w)
        return float(mp.mpf(y) - mp.log(psi))


# =====================================================================
# Demo / report
# =====================================================================

def _demo():
    sp.init_printing(use_unicode=False)

    print("=" * 70)
    print("KNEE ANSATZ")
    print("=" * 70)
    knee = KneeAnsatz()
    for name, expr in knee.catalog().items():
        print(f"\n{name}:\n  {expr}")

    dens, I = knee.line_energy()
    print(f"\nRCN line energy density: {dens}")
    print(f"Line energy integral over y: {I}   (expect 8 k2^3 / 3)")
    assert sp.simplify(I - sp.Rational(8, 3) * knee.k2**3) == 0

    # averaging demo: <theta_y^2>_sigma exact integral + numeric evaluator
    sig = sp.symbols("sigma", positive=True)
    closed, fn = knee.gaussian_avg_y(knee.thy**2, sig)
    print(f"\n<theta_y^2>_sigma symbolic: {closed}")
    print(f"<theta_y^2>_sigma at y=0, mu=0.9, sigma=pi/2: "
          f"{fn(0.0, k_1=np.sqrt(1-0.81), k_2=0.9, sigma=np.pi/2):.4f} "
          f"(vs k2^2 = 0.81, k1^2 = 0.19)")

    print("\n" + "=" * 70)
    print("PN ANSATZ (single defect, y > 0 branch)")
    print("=" * 70)
    pn = PNAnsatz()
    print(f"\nheat_check (should be 0): {pn.heat_check()}")
    for name, expr in pn.catalog().items():
        print(f"\n{name}:\n  {expr}")

    print("\n" + "=" * 70)
    print("GENERAL ANSATZ u = A cos(theta)")
    print("=" * 70)
    gen = GeneralAnsatz()
    for ij in ("xx", "xy", "yy"):
        comp = gen.uhu(ij[0], ij[1])
        print(f"\nT_{ij} DC:  {comp['DC']}")
    print(f"\nW: {gen.W}")
    print(f"corrugation factor (sx, sy): "
          f"{gen.corrugation_factor(*sp.symbols('s_x s_y', positive=True))}")


if __name__ == "__main__":
    _demo()
