"""Verification of the workhorse lemma (narrowband derivation, step 1):
   m0(D)[f cos t] = (f/k) cos t - (khat.grad f / k^2) sin t + O(q^2)
   m0(D)[f sin t] = (f/k) sin t + (khat.grad f / k^2) cos t + O(q^2)

Part 1 (symbolic, exact): for a single-mode envelope f = exp(i q.x),
f e^{i theta} is one plane wave at k+q, so m0(D) multiplies by the exact
number 1/|k+q|. The lemma is then the sympy series of 1/|k+q| in q.
By linearity over Fourier modes this proves the lemma for all envelopes.

Part 2 (numeric, spectral): the real sin/cos forms on a periodic box,
with the residual after the lemma measured to scale as q^2."""
import numpy as np
import sympy as sp

print("Part 1: symbolic, single-mode envelope")
k1, k2, q1, q2, e = sp.symbols('k1 k2 q1 q2 e', real=True, positive=False)
kap = sp.sqrt(k1**2 + k2**2)
# exact multiplier at k + e*q, expanded in e
exact = 1/sp.sqrt((k1+e*q1)**2 + (k2+e*q2)**2)
series = sp.series(exact, e, 0, 2).removeO()
lemma  = 1/kap - e*(k1*q1 + k2*q2)/kap**3     # 1/k - khat.q / k^2
print("  series(1/|k+q|) - lemma prediction =",
      sp.simplify(sp.expand(series - lemma)))
# and the Riesz symbol's gradient while we're here: grad(1/|xi|) = -xihat/|xi|^2
g1 = sp.simplify(sp.diff(1/kap, k1) + k1/kap**3)
g2 = sp.simplify(sp.diff(1/kap, k2) + k2/kap**3)
print("  grad(1/|xi|) + xihat/|xi|^2 =", (g1, g2))

print()
print("Part 2: numeric, real carriers, residual ~ q^2")
N, L = 1024, 40*np.pi
x = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, x, indexing="xy")
kx = 2*np.pi*np.fft.fftfreq(N, d=L/N)
KX, KY = np.meshgrid(kx, kx, indexing="xy")
Kd = np.hypot(KX, KY); Kd[0,0] = 1.0
def m0(u):
    U = np.fft.fft2(u); U = U/Kd; U[0,0] = 0.0
    return np.real(np.fft.ifft2(U))
m = slice(N//8, 7*N//8)
a, kap0 = 0.16, 1.0            # carrier theta = x, |k| = 1, khat = xhat
print(f"  {'q':>7} {'longitudinal: |resid|':>22} {'resid/q^2':>10}"
      f" {'transverse: |resid|':>20} {'resid/q^2':>10}")
for nper in [2, 4, 8, 16]:
    q = 2*np.pi*nper/L
    # longitudinal envelope f = 1 + a sin(qx): khat.grad f = a q cos(qx)
    f = 1 + a*np.sin(q*X)
    lhs = m0(f*np.sin(X))
    rhs = f*np.sin(X) + a*q*np.cos(q*X)*np.cos(X)     # kappa = 1
    rL = np.sqrt(np.mean((lhs-rhs)[m,m]**2))
    # transverse envelope f = 1 + a sin(qy): khat.grad f = 0, no correction
    f = 1 + a*np.sin(q*Y)
    lhs = m0(f*np.sin(X))
    rhs = f*np.sin(X)
    rT = np.sqrt(np.mean((lhs-rhs)[m,m]**2))
    print(f"  {q:7.4f} {rL:22.3e} {rL/q**2:10.3f} {rT:20.3e} {rT/q**2:10.3f}")
print("  (both residual columns divided by q^2 should be roughly constant)")
