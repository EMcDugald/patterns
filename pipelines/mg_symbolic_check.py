# mg_symbolic_check.py
"""
SymPy verification of the symbolic identities of Appendix J (the
monogenic-gradient estimator), written 27 Sep 2026 for the paper.

Checks
  (1) k = Im(conj(z) grad z)/|z|^2 = (u grad v - v grad u)/(u^2 + v^2)
      and, for z = A e^{i theta}, equals grad theta.
  (2) theta_t = Im(conj(z) z_t)/|z|^2 and A_t = Re(conj(z) z_t)/|z| for
      z = A e^{i theta}, A > 0.
  (3) grad n = n_perp grad psi for n = (cos psi, sin psi), and
      (grad n) . r = (n_perp . r) grad psi  (componentwise).
  (4) grad psi = (c2 grad s2 - s2 grad c2)/(2 (c2^2 + s2^2)) with
      c2 = cos 2 psi, s2 = sin 2 psi, and the identity
      grad atan2(s, c) = (c grad s - s grad c)/(c^2 + s^2).
  (5) the envelope argument: with z -> E z, E real,
      Im(conj(E z) grad(E z))/|E z|^2 = Im(conj(z) grad z)/|z|^2.
  (6) the sign flip: under n -> -n, v -> -v, n_perp . r -> -(n_perp . r),
      grad psi unchanged, so k -> -k.
  (7) Bedrosian's theorem on a product of two cosines with q < k:
      H[cos(q x) cos(k x)] = cos(q x) sin(k x).

Run:  python mg_symbolic_check.py
"""
import sympy as sp

x, y, t = sp.symbols("x y t", real=True)
A = sp.Function("A", positive=True)(x, y, t)
th = sp.Function("theta", real=True)(x, y, t)
u = sp.Function("u", real=True)(x, y)
v = sp.Function("v", real=True)(x, y)
E = sp.Function("E", positive=True)(x, y)
psi = sp.Function("psi", real=True)(x, y)
R1 = sp.Function("R1", real=True)(x, y)
R2 = sp.Function("R2", real=True)(x, y)

ok = True


def report(name, cond):
    global ok
    print("%-60s %s" % (name, "PASS" if cond else "FAIL"))
    ok = ok and cond


# (1) the gradient identity, explicit form and grad theta.  Real symbols
# stand for the fields and their derivatives so that re/im are exact.
U, V, Ux, Vx = sp.symbols("U V U_x V_x", real=True)
zs = U + sp.I * V
dzs = Ux + sp.I * Vx
k_expl = sp.im(sp.expand(sp.conjugate(zs) * dzs)) / (U**2 + V**2)
k_form = (U * Vx - V * Ux) / (U**2 + V**2)
report("(1) explicit form of k", sp.simplify(k_expl - k_form) == 0)
Aa, Ax, Th, Thx = sp.symbols("A A_x Theta Theta_x", real=True, positive=False)
Aa = sp.symbols("A", positive=True)
zA = Aa * sp.exp(sp.I * Th)
dzA = (Ax + sp.I * Aa * Thx) * sp.exp(sp.I * Th)        # d/dx of A e^{i theta}
prod = sp.expand(sp.simplify(sp.conjugate(zA) * dzA))
report("(1) conj z grad z = A A_x + i A^2 theta_x",
       sp.simplify(prod - (Aa * Ax + sp.I * Aa**2 * Thx)) == 0)
report("(1) k = grad theta for z = A e^{i theta}",
       sp.simplify(sp.im(prod) / Aa**2 - Thx) == 0)

# (2) the rate identity (same algebra with t in place of x)
At, Tht = sp.symbols("A_t Theta_t", real=True)
zt = (At + sp.I * Aa * Tht) * sp.exp(sp.I * Th)
prod_t = sp.expand(sp.simplify(sp.conjugate(zA) * zt))
report("(2) conj z z_t = A A_t + i A^2 theta_t",
       sp.simplify(prod_t - (Aa * At + sp.I * Aa**2 * Tht)) == 0)
report("(2) theta_t = Im(conj z z_t)/|z|^2", sp.simplify(sp.im(prod_t) / Aa**2 - Tht) == 0)
report("(2) A_t = Re(conj z z_t)/|z|", sp.simplify(sp.re(prod_t) / Aa - At) == 0)

# (3) grad n = n_perp grad psi and (grad n) . r = (n_perp . r) grad psi
n = sp.Matrix([sp.cos(psi), sp.sin(psi)])
n_perp = sp.Matrix([-sp.sin(psi), sp.cos(psi)])
r = sp.Matrix([R1, R2])
for var in (x, y):
    dn = n.diff(var)
    report("(3) d n/d%s = n_perp d psi/d%s" % (var, var),
           sp.simplify(dn - n_perp * sp.diff(psi, var)) == sp.zeros(2, 1))
    lhs = (dn.T * r)[0]
    rhs = (n_perp.T * r)[0] * sp.diff(psi, var)
    report("(3) (d n/d%s) . r = (n_perp . r) d psi/d%s" % (var, var),
           sp.simplify(lhs - rhs) == 0)
# the full product rule for v = n . r
vv = (n.T * r)[0]
for var in (x, y):
    lhs = sp.diff(vv, var)
    rhs = (n.T * r.diff(var))[0] + (n_perp.T * r)[0] * sp.diff(psi, var)
    report("(3) d v/d%s = n . d r/d%s + (n_perp . r) d psi/d%s" % (var, var, var),
           sp.simplify(lhs - rhs) == 0)

# (4) grad psi through the doubled angle, and the atan2 derivative
c2, s2 = sp.cos(2 * psi), sp.sin(2 * psi)
for var in (x, y):
    rhs = (c2 * sp.diff(s2, var) - s2 * sp.diff(c2, var)) / (2 * (c2**2 + s2**2))
    report("(4) d psi/d%s from the doubled angle" % var,
           sp.simplify(rhs - sp.diff(psi, var)) == 0)
c, s = sp.Function("c", real=True)(x), sp.Function("s", real=True)(x)
# atan2(s, c) = atan(s/c) on the branch c > 0; the derivative formula is
# branch-independent, so the check on that branch suffices
lhs = sp.diff(sp.atan(s / c), x)
rhs = (c * sp.diff(s, x) - s * sp.diff(c, x)) / (c**2 + s**2)
report("(4) d/dx atan2(s, c) = (c s' - s c')/(c^2 + s^2)", sp.simplify(lhs - rhs) == 0)

# (5) a real envelope drops out of k
Ee, Ex = sp.symbols("E E_x", positive=True)
zE = Ee * zs
dzE = Ex * zs + Ee * dzs
prodE = sp.expand(sp.conjugate(zE) * dzE)
report("(5) Im(conj(Ez) grad(Ez)) = E^2 Im(conj z grad z)",
       sp.simplify(sp.im(prodE) - Ee**2 * sp.im(sp.expand(sp.conjugate(zs) * dzs))) == 0)
report("(5) Re(conj(Ez) grad(Ez)) = E^2 Re(conj z grad z) + E grad E |z|^2",
       sp.simplify(sp.re(prodE) - Ee**2 * sp.re(sp.expand(sp.conjugate(zs) * dzs))
                   - Ee * Ex * (U**2 + V**2)) == 0)
report("(5) envelope E drops out of k",
       sp.simplify(sp.im(prodE) / (Ee**2 * (U**2 + V**2)) - k_form) == 0)

# (6) the sign flip under n -> -n
vv_m = ((-n).T * r)[0]
for var in (x, y):
    dv_p = (n.T * r.diff(var))[0] + (n_perp.T * r)[0] * sp.diff(psi, var)
    dv_m = ((-n).T * r.diff(var))[0] + ((-n_perp).T * r)[0] * sp.diff(psi, var)
    kp = (u * dv_p - vv * sp.diff(u, var)) / (u**2 + vv**2)
    km = (u * dv_m - vv_m * sp.diff(u, var)) / (u**2 + vv_m**2)
    report("(6) k -> -k under n -> -n, d/d%s" % var, sp.simplify(kp + km) == 0)

# (7) Bedrosian on cos(qx) cos(kx), 0 < q < k: H[cos] = sin on positive
# frequencies; the product is (cos((k-q)x) + cos((k+q)x))/2, both at
# positive frequency, so H = (sin((k-q)x) + sin((k+q)x))/2 = cos(qx) sin(kx)
q, kk = sp.symbols("q k", positive=True)
lhs = (sp.sin((kk - q) * x) + sp.sin((kk + q) * x)) / 2
rhs = sp.cos(q * x) * sp.sin(kk * x)
report("(7) Bedrosian: H[cos qx cos kx] = cos qx sin kx (q < k)",
       sp.simplify(sp.expand_trig(lhs - rhs)) == 0)

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
