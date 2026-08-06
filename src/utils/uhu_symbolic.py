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

Everything returns plain sympy expressions so results can be piped into
sp.latex(...) for the paper or sp.lambdify(...) for numerics. The demo
(python uhu_symbolic.py) prints the full catalog for the knee and PN and
verifies the closed-form knee line energy E = 8 k2^3 / 3.
"""

import numpy as np
import sympy as sp
from sympy.simplify.fu import TR8


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
        self.curl_k = sp.simplify(
            sp.diff(self.thy, x) - sp.diff(self.thx, y))    # 0 for exact phase
        # J = det(grad k) with grad k = Hessian(theta) for exact phases
        self.J = sp.simplify(self.hess.det())
        self.W = sp.sqrt(
            self.hess[0, 0]**2 + 2 * self.hess[0, 1]**2 + self.hess[1, 1]**2
        ) / self.k_mag2
        self.E_rcn = self.div_k**2 + (1 - self.k_mag2)**2

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

    def uhu_tensor_dc(self):
        """2x2 sympy Matrix of the DC (ideal-averaged) uHu tensor."""
        return sp.Matrix([
            [self.uhu("x", "x")["DC"], self.uhu("x", "y")["DC"]],
            [self.uhu("x", "y")["DC"], self.uhu("y", "y")["DC"]],
        ])

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

    def __init__(self, numeric_weights=False):
        self.x = sp.symbols("x", real=True)
        self.y = sp.symbols("y", positive=True)   # y > 0 branch
        if numeric_weights:
            self.a = sp.Rational(1, 2) * (1 + sp.exp(-sp.pi))
            self.b = sp.Rational(1, 2) * (sp.exp(-sp.pi) - 1)
        else:
            self.a, self.b = sp.symbols("a b", real=True)
        self.A = sp.Integer(1)
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
