# appB_variance_check.py
"""
Appendix B.6 check (28 Sep 2026): the kernel-broadening prediction
k2_meas = G_sigma * k2 (eq:broadened_tanh) against the exact eigenvector
of the window-averaged non-oscillating tensor on the knee.

On the knee k = (k1, k2(y)), k1 = sqrt(1 - mu^2), k2 = mu tanh(mu y), the
non-oscillating part of T averaged over the Gaussian window is

    <T> = -(A^2/2) [[k1^2, k1 <k2>], [k1 <k2>, <k2^2>]],  <.> = G_sigma * .

If <k2^2> were <k2>^2 the matrix would be rank one with eigenvector
(k1, <k2>), and the extraction would return exactly G_sigma * k2. The
difference <k2^2> - <k2>^2 is the variance of k2 within the window
(~ sigma^2 k2'^2), and the dominant eigenvector of the rank-two matrix
differs from (k1, <k2>). This script computes both on the mu = 0.4 knee
at sigma = pi/2 and reports the maximum errors and their L2 mismatch,
to be compared with w2_knee_error_budget.py (measured maximum 3.9e-2
against the predicted 4.5e-2, relative L2 mismatch 11 percent).

Run:  python appB_variance_check.py
"""
import numpy as np

mu, sig = 0.4, np.pi / 2
k1 = np.sqrt(1 - mu**2)
y = np.linspace(-40, 40, 32001)
k2 = mu * np.tanh(mu * y)
g = np.exp(-y**2 / (2 * sig**2))
g /= g.sum()
conv = lambda f: np.convolve(f, g, mode="same")
m, q = conv(k2), conv(k2**2)
band = np.abs(y) < 8

k2_eig = []
for i in np.where(band)[0]:
    T = -np.array([[k1**2, k1 * m[i]], [k1 * m[i], q[i]]])
    w, v = np.linalg.eigh(T)
    e = v[:, 0] * np.sign(v[0, 0])
    k2_eig.append(np.sqrt(-w[0]) * e[1])
k2_eig = np.array(k2_eig)
kb, mb = k2[band], m[band]

print("knee mu = %.2f, sigma = pi/2" % mu)
print("max |k2_eig  - k2| (eigenvector of the averaged tensor): %.4f" % np.abs(k2_eig - kb).max())
print("max |G*k2    - k2| (eq:broadened_tanh, full convolution): %.4f" % np.abs(mb - kb).max())
print("max |k2_eig - G*k2|                                    : %.4f" % np.abs(k2_eig - mb).max())
print("relative L2 of (k2_eig - G*k2) over (G*k2 - k2)        : %.3f" % (np.linalg.norm(k2_eig - mb) / np.linalg.norm(mb - kb)))
print("variance of k2 in the window, max                       : %.2e" % (q - m**2)[band].max())
print("sigma^2 max k2'^2                                       : %.2e" % (sig**2 * (mu**2 / np.cosh(mu * y)**2)**2)[band].max())
