# pipelines/rcn_w_evolution_check.py
"""
SymPy companion to Writeup 4, part (b): the evolution equation for
w = |k|^2 - 1 implied by the RCN gradient flow, derived from first
principles and checked by assertion, so the weak-SINDy w law can be read
against a theoretical prediction rather than against intuition.

Conventions (chapter ch3_pgbs, "Recall the evolution equations"):
    tau theta_t = -delta Ebar / delta theta,
    Ebar = int (eta/2)(Lap theta)^2 + (alpha/2) G^2(k^2),
    k = grad theta,  w = |k|^2 - 1,
    G^2 = w^2 - w^4/(2R),  B = -alpha dG^2/dk^2 = -2 alpha f(w),
    f(w) = w - w^3/R.
tau, eta, alpha, R stay symbolic throughout; in the SH reduction of
Writeup 1, alpha = 2R/3 and tau = <u_theta^2> = 2R/3, and the chapter's
regularized form uses eta = 2 tau. Nothing below depends on those values
except the numerical evaluations, which state them.

Changelog
---------
v1: checks
  (1) Euler-Lagrange: delta Ebar/delta theta = eta Lap^2 theta
      + div(k B(k^2)) on a generic theta(x,y).
  (2) Closed form for tau w_t on a generic field, with the gradient
      structure k = grad theta used (grad k symmetric):
        tau w_t = 4 alpha f'(w) [ kk:grad grad w + |grad w|^2/2
                                  + (k.grad w) div k ]
                + 4 alpha f''(w) (k.grad w)^2
                + 2 alpha f(w) Lap w - 4 alpha f(w) |grad k|_F^2
                - 2 eta k.grad(Lap div k).
  (3) Linearization about rolls theta = k0 x + eps phi: the dispersion
      relation, the parallel and perpendicular diffusion constants,
        D_par = (2 alpha/tau)(2 k0^2 f'(w0) + f(w0)),
        D_perp = (2 alpha/tau) f(w0),
      and the fact that w1 = 2 k0 phi_x obeys the SAME linear operator,
      so the far-field w law of RCN is
        w_t = D_par w_ss + D_perp w_nn - (eta/tau) Lap^2 w,
      with NO linear w term and NO isotropic Lap w term.
  (4) The self-dual knee theta = k1 x + (1/beta) log cosh(beta mu y),
      beta = sqrt(alpha/eta), is stationary under the flow with the
      quartic of G^2 dropped, for arbitrary (alpha, eta).
  (5) With the quartic kept, the knee is not stationary; the residual
      flux is -(2 alpha/R) d/dy (w^3 g'), relative size w^2/R of the
      retained flux (numbers at mu = 0.475, R = 0.5 and mu = 0.85).
  (6) One-dimensional reduction: for theta = k1 x + phi(y,t) the w
      equation closes in w(y,t) alone (printed), which is the RCN
      replacement for the linear-diffusion knee control of
      wsindy_knee_validation.py.
"""
import sympy as sp


def main():
    x, y = sp.symbols('x y', real=True)
    tau, eta, alpha, R = sp.symbols('tau eta alpha R', positive=True)
    th = sp.Function('theta')(x, y)

    k1, k2 = sp.diff(th, x), sp.diff(th, y)
    k = sp.Matrix([k1, k2])
    w = k1**2 + k2**2 - 1
    lap = lambda e: sp.diff(e, x, 2) + sp.diff(e, y, 2)
    div = lambda v: sp.diff(v[0], x) + sp.diff(v[1], y)
    grad = lambda e: sp.Matrix([sp.diff(e, x), sp.diff(e, y)])

    wsym = sp.symbols('w', real=True)
    G2 = wsym**2 - wsym**4/(2*R)
    f = sp.Rational(1, 2)*sp.diff(G2, wsym)          # f(w) = w - w^3/R
    B = -alpha*sp.diff(G2, wsym)                       # B = -2 alpha f
    assert sp.simplify(B + 2*alpha*f) == 0

    # ---------------------------------------------------------------
    # (1) Euler-Lagrange of Ebar
    # ---------------------------------------------------------------
    L = eta/2*lap(th)**2 + alpha/2*G2.subs(wsym, w)
    from sympy.calculus.euler import euler_equations
    dE = euler_equations(L, [th], [x, y])[0].lhs      # delta Ebar / delta theta
    target1 = eta*lap(lap(th)) + div(k*B.subs(wsym, w))
    assert sp.simplify(sp.expand(dE - target1)) == 0
    print("(1) dEbar/dtheta = eta Lap^2 theta + div(k B(k^2))          OK")

    # ---------------------------------------------------------------
    # (2) closed form for tau w_t = 2 k . grad(tau theta_t)
    # ---------------------------------------------------------------
    tau_theta_t = -dE
    tau_w_t = 2*(k.T*grad(tau_theta_t))[0]

    fw = f.subs(wsym, w)
    fp = sp.diff(f, wsym).subs(wsym, w)
    fpp = sp.diff(f, wsym, 2).subs(wsym, w)
    gw = grad(w)
    H = sp.hessian(w, (x, y))
    kk_ddw = (k.T*H*k)[0]
    k_gw = (k.T*gw)[0]
    gradk = sp.Matrix([[sp.diff(k1, x), sp.diff(k1, y)],
                       [sp.diff(k2, x), sp.diff(k2, y)]])
    gradk_F2 = sum(gradk[i, j]**2 for i in range(2) for j in range(2))
    lap_div = lap(div(k))
    closed = (4*alpha*fp*(kk_ddw + gw.dot(gw)/2 + k_gw*div(k))
              + 4*alpha*fpp*k_gw**2
              + 2*alpha*fw*lap(w) - 4*alpha*fw*gradk_F2
              - 2*eta*(k.T*grad(lap_div))[0])
    assert sp.simplify(sp.expand(tau_w_t - closed)) == 0
    print("(2) tau w_t closed form in (k, w) and derivatives            OK")

    # the two identities used to get there, stated for the writeup
    assert sp.simplify(sp.expand(lap(w) - 2*(k.T*grad(div(k)))[0]
                                 - 2*gradk_F2)) == 0
    assert sp.simplify(sp.expand((gradk*k).dot(gw) - gw.dot(gw)/2)) == 0
    print("    Lap w = 2 k.grad div k + 2|grad k|_F^2;  (k.grad k).grad w"
          " = |grad w|^2/2  (grad k symmetric)              OK")

    # ---------------------------------------------------------------
    # (3) linearization about rolls
    # ---------------------------------------------------------------
    k0, eps, qx, qy, sig, t = sp.symbols('k0 epsilon q_x q_y sigma t',
                                         real=True)
    phi = sp.exp(sp.I*(qx*x + qy*y) + sig*t)
    th_lin = k0*x + eps*phi
    rhs = tau_theta_t.subs(th, th_lin).doit()
    rhs1 = sp.diff(rhs, eps).subs(eps, 0)              # O(eps) part
    disp = sp.simplify(rhs1/phi)                        # tau sigma
    w0 = k0**2 - 1
    f0 = f.subs(wsym, w0)
    fp0 = sp.diff(f, wsym).subs(wsym, w0)
    target3 = -(2*alpha*(2*k0**2*fp0 + f0)*qx**2 + 2*alpha*f0*qy**2
                + eta*(qx**2 + qy**2)**2)
    assert sp.simplify(disp - target3) == 0
    D_par = sp.simplify(2*alpha/tau*(2*k0**2*fp0 + f0))
    D_perp = sp.simplify(2*alpha/tau*f0)
    print("(3) tau sigma = -[2a(2k0^2 f' + f) qx^2 + 2a f qy^2 + eta q^4]"
          "  OK")
    print("    D_par  = (2 alpha/tau)(2 k0^2 f'(w0) + f(w0))")
    print("    D_perp = (2 alpha/tau) f(w0)   -> D_perp ~ (2 alpha/tau) w0"
          " (zigzag for w0<0)")
    print("    at alpha = tau, w0 -> 0:  D_par -> 4,  D_perp -> 2 w0")
    # w1 = 2 k0 phi_x is a linear function of phi with a constant-
    # coefficient operator, so it obeys the identical dispersion: the
    # linear w law is w_t = D_par w_ss + D_perp w_nn - (eta/tau) Lap^2 w.
    w_full = w.subs(th, th_lin).doit()
    w1 = sp.diff(w_full, eps).subs(eps, 0)
    assert sp.simplify(w1 - 2*k0*sp.diff(eps*phi, x).subs(eps, 1)) == 0
    print("    w1 = 2 k0 phi_x, same operator: no w term, no isotropic"
          " Lap w in the far field              OK")

    # Eckhaus (D_par = 0) and zigzag (D_perp = 0) boundaries in w0
    w0s = sp.symbols('w0', real=True)
    Dpar_w0 = 2*(1 + w0s)*sp.diff(f, wsym).subs(wsym, w0s) + f.subs(wsym, w0s)
    print("    Eckhaus boundary: root of 2(1+w0)f'(w0) + f(w0) = 0;"
          " zigzag boundary: f(w0) = 0, i.e. w0 = 0")
    for Rv in (sp.Rational(1, 2), 1):
        poly = sp.Poly(sp.expand(Dpar_w0.subs(R, Rv)*Rv), w0s)
        roots = [complex(r) for r in poly.nroots()]
        neg = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real < 0]
        print("      R = %s: Eckhaus w0 = %.4f (sqrt(R/3) = %.4f for"
              " reference)" % (Rv, max(neg), float(sp.sqrt(Rv/3))))

    # ---------------------------------------------------------------
    # (4) stationarity of the self-dual knee, quartic dropped
    # ---------------------------------------------------------------
    mu = sp.symbols('mu', positive=True)
    beta = sp.sqrt(alpha/eta)
    th_knee = sp.sqrt(1 - mu**2)*x + sp.log(sp.cosh(beta*mu*y))/beta
    L_noq = eta/2*lap(th)**2 + alpha/2*wsym**2
    L_noq = L_noq.subs(wsym, w)
    dE_noq = euler_equations(L_noq, [th], [x, y])[0].lhs
    res4 = sp.simplify(dE_noq.subs(th, th_knee).doit())
    assert res4 == 0
    print("(4) knee theta = k1 x + (1/beta) log cosh(beta mu y),"
          " beta^2 = alpha/eta: stationary with G^2 = w^2       OK")
    w_knee = w.subs(th, th_knee).doit()
    assert sp.simplify(lap(th_knee) + beta*w_knee) == 0
    print("    Lap theta = -beta w on the knee                             OK")

    # ---------------------------------------------------------------
    # (5) the quartic makes it non-stationary; size of the residual
    # ---------------------------------------------------------------
    res5 = sp.simplify(dE.subs(th, th_knee).doit())
    g1 = sp.diff(th_knee, y)
    target5 = -(2*alpha/R)*sp.diff(w_knee**3*g1, y)      # extra flux
    assert sp.simplify(res5 + target5) == 0               # dE = -(-...)
    print("(5) with the quartic: tau theta_t = -(2 alpha/R) d/dy(w^3 g')"
          " on the knee, retained flux 2 alpha d/dy(w g')  OK")
    q_flux = sp.diff(w_knee**3*g1, y)/R
    r_flux = sp.diff(w_knee*g1, y)
    for muv, Rv in ((0.475, 0.5), (0.85, 0.5), (0.85, 1.0)):
        sub = {mu: muv, R: Rv, alpha: 1, eta: 1}
        ys = [i*0.02 for i in range(0, 600)]
        qv = max(abs(float(q_flux.subs(sub).subs(y, yy))) for yy in ys)
        rv = max(abs(float(r_flux.subs(sub).subs(y, yy))) for yy in ys)
        print("    mu = %.3f, R = %.1f: peak|quartic flux|/peak|retained"
              " flux| = %.3f  (w^2/R at the core = %.3f)"
              % (muv, Rv, qv/rv, muv**4/Rv))

    # ---------------------------------------------------------------
    # (6) one-dimensional reduction, theta = k1 x + Phi(y,t)
    # ---------------------------------------------------------------
    k1c = sp.symbols('k_1', positive=True)
    th1 = sp.Function('theta1')(x, y)
    expr = closed.subs(th, th1).doit()
    Phi = sp.Function('Phi')(y)
    e1 = expr.subs({th1: k1c*x + Phi}).doit()
    p = sp.diff(Phi, y)
    w1d = k1c**2 + p**2 - 1
    Wf = sp.Function('W')(y)
    cand = (4*alpha*sp.diff(f, wsym).subs(wsym, Wf)
            * (p**2*sp.diff(Wf, y, 2) + sp.diff(Wf, y)**2/2
               + p*sp.diff(Wf, y)*sp.diff(p, y))
            + 4*alpha*sp.diff(f, wsym, 2).subs(wsym, Wf)*(p*sp.diff(Wf, y))**2
            + 2*alpha*f.subs(wsym, Wf)*sp.diff(Wf, y, 2)
            - 4*alpha*f.subs(wsym, Wf)*sp.diff(p, y)**2
            - 2*eta*p*sp.diff(p, y, 4))
    diff6 = sp.simplify(sp.expand(e1 - cand.subs(Wf, w1d).doit()))
    assert diff6 == 0
    print("(6) 1d reduction (theta = k1 x + Phi(y)), with p = Phi_y, W ="
          " k1^2 + p^2 - 1:")
    print("    tau W_t = 4a f'(W)[p^2 W_yy + W_y^2/2 + p W_y p_y]"
          " + 4a f''(W)(p W_y)^2 + 2a f(W) W_yy - 4a f(W) p_y^2"
          " - 2 eta p p_yyyy")
    print("    (p_y = div k, p^2 = W + 1 - k1^2, so p closes in W up to a"
          " sign)                                        OK")

    # ---------------------------------------------------------------
    # leading order in w, for the library translation
    # ---------------------------------------------------------------
    print()
    print("Leading order in w (f -> w, f' -> 1, f'' -> 0):")
    print("  tau w_t = 4 alpha [ kk:grad grad w + |grad w|^2/2"
          " + (k.grad w) div k ]")
    print("          + 2 alpha w Lap w - 4 alpha w |grad k|_F^2"
          " - 2 eta k.grad(Lap div k)")
    print("  Terms in the current wsindy w library: none of these"
          " (Lap w, w, w^2, |grad w|, (div k)^2, (curl k)^2, J, |grad k|^2"
          " are present; kk:grad grad w, |grad w|^2, (k.grad w) div k,"
          " w Lap w, w |grad k|^2, k.grad(Lap div k) are not).")
    print("All checks passed.")


if __name__ == "__main__":
    main()
