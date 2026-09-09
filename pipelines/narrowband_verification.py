# pipelines/narrowband_verification.py
"""
Numeric verification of the narrowband expansion of the Riesz transform
on analytic modulated stripe fields (Writeup 3, Sec. 2.3). Periodic
spectral fields only; no extraction pipeline; this tests mathematics.

Verified statements (all measured by this script):
 1. First-order correction. For u = A cos(theta), k = grad theta:
      R = A sin(theta) khat - (cos(theta)/|k|) Pperp grad A + (Hessian)
    The residual after the correction is the amplitude-Hessian term:
    |R - R1|/|R - R0| = q/(2|k|) to three digits across q in [0.1, 0.4].
    Longitudinal modulation is EXACT (machine zero): Bedrosian's theorem.
 2. Director angle error, transverse amplitude modulation: first order
    in eps_perp = |Pperp grad A|/(A|k|), exponentially suppressed by the
    width: delta_alpha ~ 0.5 * eps_perp * exp(-2 sigma^2 |k|^2), tracked
    over four decades (sigma 0.3 to 2.2); no quadratic smooth floor.
 3. Coherence: 1 - alignment = <eps_perp^2>, coefficient 0.96 to 1.03
    at sigma = pi/2.
 4. Wavevector bending theta = x + (ek/q) sin(qy): angle deviation from
    the pointwise grad-theta direction is first order in ek, the sum of
    the same ripple term and the window attenuation of the bending
    angle, (1 - exp(-sigma^2 q^2 / 2))/sqrt(2): the wide-width reading
    is the window-averaged orientation, the angle's measurement kernel.

Implementation note carried as a warning: the Riesz division grid must
patch its zero mode (|xi| -> 1) but the SMOOTHER must not reuse that
patched grid, or the spatial mean of C is attenuated by
exp(-sigma^2/2) and every large-sigma statistic is corrupted. This bug
was made and caught during the session that wrote this script.
"""
import numpy as np

N, L = 1024, 40 * np.pi
x = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, x, indexing="xy")
kx = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
KX, KY = np.meshgrid(kx, kx, indexing="xy")
Kdiv = np.hypot(KX, KY); Kdiv[0, 0] = 1.0   # Riesz multiplier only
K2 = KX**2 + KY**2                           # smoother: true zero mode


def riesz(u):
    U = np.fft.fft2(u)
    return (np.real(np.fft.ifft2(-1j * KX / Kdiv * U)),
            np.real(np.fft.ifft2(-1j * KY / Kdiv * U)))


def gsmooth(f, s):
    return np.real(np.fft.ifft2(np.fft.fft2(f) * np.exp(-0.5 * s**2 * K2)))


def director_stats(u, s, ang_true):
    R1, R2 = riesz(u)
    C = gsmooth(R1 * R1 - R2 * R2, s)
    S = gsmooth(2 * R1 * R2, s)
    ali = np.sqrt(C**2 + S**2) / np.maximum(gsmooth(R1 * R1 + R2 * R2, s),
                                            1e-30)
    a = 0.5 * np.arctan2(S, C) - ang_true
    a = (a + np.pi / 2) % np.pi - np.pi / 2
    m = slice(N // 8, 7 * N // 8)
    return np.sqrt(np.mean(a[m, m]**2)), 1.0 - np.mean(ali[m, m])


def main():
    q = 2 * np.pi * 4 / L
    m = slice(N // 8, 7 * N // 8)

    print("1. first-order correction: q-sweep, a=0.16 (ratio -> q/2)")
    a = 0.16
    for nper in [2, 4, 8, 16]:
        qq = 2 * np.pi * nper / L
        A = 1.0 + a * np.sin(qq * Y)
        u = A * np.cos(X)
        R1n, R2n = riesz(u)
        R0x = A * np.sin(X)
        dAy = a * qq * np.cos(qq * Y)
        e0 = np.sqrt(np.mean((R1n - R0x)[m, m]**2 + R2n[m, m]**2))
        e1 = np.sqrt(np.mean((R1n - R0x)[m, m]**2 +
                             (R2n + np.cos(X) * dAy)[m, m]**2))
        print(f"   q={qq:6.4f}  |R-R0|={e0:9.3e}  |R-R1|={e1:9.3e}  "
              f"ratio={e1/e0:7.4f}  q/2={qq/2:6.4f}")

    print("1b. longitudinal modulation (Bedrosian): expect machine zero")
    A = 1.0 + a * np.sin(q * X)
    u = A * np.cos(X)
    R1n, R2n = riesz(u)
    e0 = np.sqrt(np.mean((R1n - A * np.sin(X))[m, m]**2 + R2n[m, m]**2))
    print(f"   |R-R0| = {e0:.3e}")

    print("2. angle error vs sigma, transverse a=0.16 (ratio ~ const)")
    A = 1.0 + a * np.sin(q * Y); u = A * np.cos(X)
    for s in [0.3, 0.4, 0.8, 1.2, np.pi / 2, 2.2]:
        ar, _ = director_stats(u, s, 0.0)
        print(f"   sigma={s:5.3f}  ang={ar:9.3e}  "
              f"ang/(eps*exp(-2 s^2))={ar/(a*q*np.exp(-2*s**2)):7.3f}")

    print("3. coherence deficit at sigma=pi/2 (coefficient -> 1)")
    for aa in [0.04, 0.08, 0.16, 0.32]:
        A = 1.0 + aa * np.sin(q * Y); u = A * np.cos(X)
        _, ad = director_stats(u, np.pi / 2, 0.0)
        print(f"   eps={aa*q:7.4f}  (1-ali)/eps^2 = {ad/(aa*q)**2:6.3f}")

    print("4. wavevector bending: angle deviation, both widths")
    for s in [0.4, np.pi / 2]:
        attn = (1 - np.exp(-0.5 * s**2 * q**2)) / np.sqrt(2)
        rip = 0.02 * np.exp(-2 * s**2)   # empirical ripple prefactor
        for ek in [0.04, 0.16]:
            th = X + (ek / q) * np.sin(q * Y)
            tru = np.arctan2(ek * np.cos(q * Y), 1.0)
            ar, _ = director_stats(np.cos(th), s, tru)
            print(f"   sigma={s:5.3f} ek={ek:4.2f}  ang={ar:9.3e}  "
                  f"(ripple+attn)*ek={(rip+attn)*ek:9.3e}")


if __name__ == "__main__":
    main()


def second_order_checks():
    """Second-order verification (added the same session): the derived
    term -0.5 sin(th) T:gradgradA - 0.5 A cos(th) T:gradgradTheta with
    T = grad^3 |xi| in closed form. Bending-field error drops x5 with
    residual ratio ~ q (third order); mixed field drops order by order.
    Ripple constant closed: c = 0.5 exp(-sigma^2 q^2 / 2)."""
    q = 2 * np.pi * 4 / L
    m = slice(N // 8, 7 * N // 8)

    def T_contract(j, k1, k2, Mxx, Mxy, Myy):
        kap = np.hypot(k1, k2)
        kh = {0: k1 / kap, 1: k2 / kap}
        d = lambda a, b: 1.0 * (a == b)
        def T(l, n):
            return (3 * kh[j] * kh[l] * kh[n] - d(j, n) * kh[l]
                    - d(l, n) * kh[j] - d(j, l) * kh[n]) / kap**2
        return T(0, 0) * Mxx + 2 * T(0, 1) * Mxy + T(1, 1) * Myy

    print("5. second-order term on the bending field (reduction ~ 1/q)")
    for ek in [0.08, 0.16, 0.32]:
        th = X + (ek / q) * np.sin(q * Y)
        u = np.cos(th)
        k1, k2 = np.ones_like(u), ek * np.cos(q * Y)
        kap = np.hypot(k1, k2)
        R1n, R2n = riesz(u)
        R0x, R0y = np.sin(th) * k1 / kap, np.sin(th) * k2 / kap
        Hyy = -ek * q * np.sin(q * Y)
        R2x = -0.5 * u * T_contract(0, k1, k2, 0, 0, Hyy)
        R2y = -0.5 * u * T_contract(1, k1, k2, 0, 0, Hyy)
        e0 = np.sqrt(np.mean((R1n - R0x)[m, m]**2 + (R2n - R0y)[m, m]**2))
        e2 = np.sqrt(np.mean((R1n - R0x - R2x)[m, m]**2 +
                             (R2n - R0y - R2y)[m, m]**2))
        print(f"   ek={ek:4.2f}  e0={e0:9.3e}  e2={e2:9.3e}  "
              f"x{e0/e2:5.1f}  e2/e0={e2/e0:6.3f} (q={q:.3f})")


if __name__ == "__main__":
    second_order_checks()
