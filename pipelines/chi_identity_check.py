"""
Identity behind relations A and B (SymPy, then numerics on the exact PN row).

Relation A (second order in theta):  div k = -beta w + chi   defines chi.
Relation B (fourth order in theta):  Lap div k = lambda S,  S = w div k + k.grad w.
Claim: for ANY gradient field k = grad theta with w = |k|^2 - 1 and J = det Hess theta,
    Lap div k - 2 beta^2 S  =  Lap chi - 2 beta div(chi k) + 4 beta J .
So on a field with chi = 0 (exact A) and J = 0, B holds with lambda = 2 beta^2 > 0,
whatever the sign of beta: A and anti-B cannot coexist on an exact J-free gradient field.
"""
import sympy as sp
import numpy as np

x, y, beta = sp.symbols('x y beta', real=True)
th = sp.Function('theta')(x, y)
k = sp.Matrix([sp.diff(th, x), sp.diff(th, y)])
w = k[0]**2 + k[1]**2 - 1
divk = sp.diff(k[0], x) + sp.diff(k[1], y)
lap = lambda e: sp.diff(e, x, 2) + sp.diff(e, y, 2)
grad = lambda e: sp.Matrix([sp.diff(e, x), sp.diff(e, y)])
J = sp.diff(th, x, 2)*sp.diff(th, y, 2) - sp.diff(th, x, y)**2
S = w*divk + (k.T*grad(w))[0]
chi = divk + beta*w
lhs = lap(divk) - 2*beta**2*S
rhs = lap(chi) - 2*beta*(sp.diff(chi*k[0], x) + sp.diff(chi*k[1], y)) + 4*beta*J
assert sp.simplify(sp.expand(lhs - rhs)) == 0
print("SymPy: Lap div k - 2 beta^2 S == Lap chi - 2 beta div(chi k) + 4 beta J   OK (any gradient field)")

# also the ingredient identities, for the writeup
assert sp.simplify(sp.expand(lap(w) - 2*(sp.diff(k[0],x)**2 + sp.diff(k[0],y)**2 + sp.diff(k[1],x)**2 + sp.diff(k[1],y)**2) - 2*(k.T*grad(divk))[0])) == 0
assert sp.simplify(sp.expand((sp.diff(k[0],x)**2 + sp.diff(k[0],y)**2 + sp.diff(k[1],x)**2 + sp.diff(k[1],y)**2) - (divk**2 - 2*J))) == 0
print("   ingredients: Lap w = 2|grad k|^2 + 2 k.grad div k ;  |grad k|^2 = (div k)^2 - 2J   OK")

# ---------------- numerics on the exact PN row (pn_field.pismen_on_sh_grid) ----------------
import sys
sys.path.insert(0, '/Users/edwardmcdugald/patterns/src/op_extract')
from pn_field import pismen_on_sh_grid

L = np.pi/0.526782687642637          # zigzag spacing at mu 0.85, 5.96
dx = 0.05
xs = np.arange(-4*L, 4*L, dx); ys = np.arange(0.3, 40, dx)     # upper half-plane, cores at y = 0
X, Y = np.meshgrid(xs, ys)
xd = (np.arange(-3.5, 4.0, 1.0))*L                              # 8 cores
theta, _ = pismen_on_sh_grid(X, Y, xd)

def d(f, ax): return np.gradient(f, dx, axis=ax)
k2, k1 = d(theta, 0), d(theta, 1)          # axis 0 = y, axis 1 = x
w = k1**2 + k2**2 - 1
divk = d(k1, 1) + d(k2, 0)
J = d(k1, 1)*d(k2, 0) - d(k1, 0)*d(k2, 1)
lapdivk = d(d(divk, 1), 1) + d(d(divk, 0), 0)
S = w*divk + k1*d(w, 1) + k2*d(w, 0)
# nearest-core distance in spacings
dist = np.min(np.stack([np.hypot(X - xj, Y) for xj in xd]), axis=0)/L
inner = (np.abs(X) < 3*L)

def fit_slope(a, b, m):
    a, b = a[m], b[m]
    A = np.vstack([a, np.ones_like(a)]).T
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    r2 = 1 - np.sum((b - A@c)**2)/np.sum((b - b.mean())**2)
    return c[0], r2

print("\nExact PN row of 8 at the zigzag spacing, upper half-plane, bins by distance to nearest core")
print(" bin(L)   betaA(div k=-beta w)  R2     lambdaB(Lapdivk=lam S)  R2    2*betaA^2   |4J|/|LapS| rms")
for lo, hi in [(0.5,1),(1,2),(2,4),(4,8)]:
    m = inner & (dist>=lo) & (dist<hi)
    bA, r2A = fit_slope(-w, divk, m)
    lam, r2B = fit_slope(S, lapdivk, m)
    ratio = np.sqrt(np.mean((4*bA*J[m])**2))/np.sqrt(np.mean((lapdivk[m])**2))
    print(f" {lo:>3}-{hi:<3}  {bA:8.3f}          {r2A:5.2f}   {lam:8.3f}             {r2B:5.2f}   {2*bA**2:7.3f}   {ratio:8.3f}")

# decomposition of the B-residual with beta = betaA of the 1-2 L bin
m = inner & (dist>=1) & (dist<4)
bA, _ = fit_slope(-w, divk, m)
chi = divk + bA*w
lapchi = d(d(chi,1),1) + d(d(chi,0),0)
divchik = d(chi*k1, 1) + d(chi*k2, 0)
res = lapdivk - 2*bA**2*S
recon = lapchi - 2*bA*divchik + 4*bA*J
print(f"\nbeta_A (1-4 L) = {bA:.3f};  identity residual max |LHS-RHS| / rms LHS = {np.max(np.abs(res-recon)[m])/np.sqrt(np.mean(res[m]**2)):.2e}")
for name, f in [("Lap chi", lapchi), ("-2 beta div(chi k)", -2*bA*divchik), ("4 beta J", 4*bA*J), ("B residual", res), ("Lap div k", lapdivk)]:
    print(f"   rms on 1-4 L: {name:20s} {np.sqrt(np.mean(f[m]**2)):.3e}")
