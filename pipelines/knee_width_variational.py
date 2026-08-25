"""
Decisive numerical test of the self-dual knee width, no epsilon expansion.

Ansatz: u(x,y) = A1(k^2(y)) cos(theta) + A3(k^2(y)) cos(3 theta),
        theta = k1 x + (1/lt) log(2 cosh(lt * mu * y)),  k1 = sqrt(1-mu^2),
so theta_y = mu tanh(lt mu y), far field |k| = 1, and lt is the knee
width parameter. Amplitudes are slaved: A1^2 = (4/3)(R - (k^2-1)^2),
A3 = -A1^3 / (4 (R + 79 k^4 - 14 k^2 - 1)).

Compute the exact SH energy per unit x (x handled spectrally, y by fine
finite differences), subtract the far-field density, and minimize the
excess over lt. RCN predicts lt* -> sqrt(alpha/eta) as mu -> 0.
Candidates: 1.0 (naive direct insertion), 0.866 (ratio 3/4, the working
value, measured 0.871 on relaxed SH), 0.707 (ratio 1/2, literal (2.45)).
"""
import numpy as np

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

R = 0.5

def energy_excess(lt, mu, Ly=60.0, ny=6001, nx=64):
    k1 = np.sqrt(1.0 - mu*mu)
    y = np.linspace(-Ly, Ly, ny); dy = y[1]-y[0]
    x = np.linspace(0, 2*np.pi/k1, nx, endpoint=False)
    X, Y = np.meshgrid(x, y)
    f = (1.0/lt)*(np.abs(lt*mu*Y) + np.log1p(np.exp(-2*np.abs(lt*mu*Y))) )  # log(2cosh) stable, minus log2 const
    th = k1*X + f
    k2y = mu*np.tanh(lt*mu*Y)
    ksq = k1*k1 + k2y*k2y
    A1 = np.sqrt(np.maximum(4.0/3.0*(R - (ksq-1.0)**2), 0.0))
    A3 = -A1**3/(4.0*(R + 79.0*ksq**2 - 14.0*ksq - 1.0))
    u = A1*np.cos(th) + A3*np.cos(3*th)
    # Laplacian: spectral in x, 4th-order FD in y
    kxf = 2*np.pi*np.fft.rfftfreq(nx, d=x[1]-x[0])
    uxx = np.fft.irfft(-(kxf**2)*np.fft.rfft(u, axis=1), n=nx, axis=1)
    uyy = np.empty_like(u)
    uyy[2:-2,:] = (-u[4:,:] + 16*u[3:-1,:] - 30*u[2:-2,:]
                   + 16*u[1:-3,:] - u[:-4,:])/(12*dy*dy)
    uyy[:2,:] = uyy[2:3,:]; uyy[-2:,:] = uyy[-3:-2,:]
    Lu = u + uxx + uyy
    e = 0.5*Lu**2 - 0.5*R*u**2 + 0.25*u**4
    ebar = e.mean(axis=1)                      # x-average -> e(y)
    eroll = ebar[[5, -6]].mean()               # far-field reference
    core = slice(2, ny-2)
    return _trapz(ebar[core]-eroll, y[core])

def minimize_width(mu):
    lts = np.linspace(0.60, 1.25, 27)
    Es = np.array([energy_excess(l, mu) for l in lts])
    i = np.argmin(Es)
    # parabolic refine
    l0, l1, l2 = lts[i-1], lts[i], lts[i+1]
    e0, e1, e2 = Es[i-1], Es[i], Es[i+1]
    lstar = l1 - 0.5*((l1-l0)**2*(e1-e2)-(l1-l2)**2*(e1-e0)) / \
                 ((l1-l0)*(e1-e2)-(l1-l2)*(e1-e0))
    return lstar, Es[i]

print(f"{'mu':>6} {'lt* (min width)':>16} {'E_excess':>12}")
for mu in [0.5, 0.4, 0.3, 0.2, 0.1]:
    lstar, E = minimize_width(mu)
    print(f"{mu:6.2f} {lstar:16.4f} {E:12.6f}")
print("\ncandidates: 1.0 (ratio 1) | 0.8660 (ratio 3/4) | 0.7071 (ratio 1/2)")
print("measured on relaxed SH at mu=0.4-0.5: 0.871 (inherited)")