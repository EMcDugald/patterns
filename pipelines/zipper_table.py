"""
zipper_table.py

Stage 0 of the ellipse-from-zippers construction: a table of regularized
Cross-Newell (RCN) zipper minimizers over the far-field angle alpha.

Problem (variational paper, Sec. 5, eqs. 52 and 54)
---------------------------------------------------
Far-field wavevectors k = (cos alpha, +/- sin alpha); eps = cos alpha,
q = sin alpha, period ell = pi / eps. On the fundamental domain
[0, ell) x [0, L] we minimize

    F[theta; a, delta] = int ( (lap theta)^2 + (1 - |grad theta|^2)^2 ) dx dy

subject to
    theta(x, 0)   = 0                 for 0 <= x <= a ell    (Dirichlet part)
    theta_y(x, 0) = 0                 for a ell <= x <= ell  (Neumann part)
    theta(x + ell, y) = theta(x, y) + pi                     (shift periodicity)
    theta(x, L)   = eps x + q L + delta                      (delta minimized over)

and scan a over [0, 1). a = 0 is the knee (exact minimizer on the half line:
theta = eps x + log cosh(q y), delta = -log 2). a > 0 is the zipper branch
(a concave/convex disclination pair per period).

For every alpha the table keeps BOTH branches: the knee and the best a > 0
zipper. Later stages place dislocations by several rules (RCN switch, SH
energy, the 43 degree rule, positions read from an SH run), and some of those
put zippers at angles where RCN itself prefers the knee.

Two discretizations
-------------------
disc = "bfs" (default): H^2-conforming Bogner-Fox-Schmit elements (bicubic
    Hermite; nodal dofs theta, theta_x, theta_y, theta_xy). Every discrete
    function lies in H^2 and satisfies the boundary conditions exactly along
    whole edges, so each discrete energy is an upper bound (up to quadrature)
    that decreases to the continuum value. Switch nodes (where Dirichlet and
    Neumann meet) are clamped: theta = grad theta = 0 there, which every C^1
    admissible function satisfies anyway.
disc = "fd": the paper's finite differences (5-point Laplacian, |grad|^2 as
    the mean of four one-sided squares, odd/even reflection ghosts). Kept for
    the documented comparison only; see the v2 changelog entry.

Both energies are sums of squared residuals, minimized by Levenberg-Marquardt
with a sparse Jacobian and a sparse LU solve per step.

Phase-offset convention
-----------------------
delta is the far-field phase offset measured from the START of a Dirichlet
segment (x = 0). Extrapolate the far field eps x + q y + delta to y = 0:
Dirichlet segments start where (far-field phase - delta) is a multiple of pi.
For the knee (a = 0) the gauge is fixed by theta(0, 0) = 0, so delta is
(far-field phase) - (core phase) there too.

Modes
-----
controls : planted knee control (energy and delta against the exact knee,
           at several h) and convergence of one zipper in h and L.
table    : the alpha sweep; writes zipper_table.npz / .csv / summary .json,
           per-alpha fields (theta at the nodes, energy-vs-a curve), figures.

Outputs go to pipelines/results/zipper_table/ (anchored to this file).

Changelog
---------
v1 (2026-10-08). First version, written fresh (no code carried over from the
    2025 scripts), with the paper's finite differences.
v2 (2026-10-08). Added the conforming "bfs" discretization and made it the
    default. Reason, measured with v1 at alpha = 64 deg, a = 0.45, L = 16:
    the fd zipper energy per length rose with every refinement,
    0.722, 0.774, 0.835, 0.890, 0.947 for h = 0.25, 0.177, 0.125, 0.088,
    0.0625, while the knee converged at second order (0.962 to 0.968). The
    excess sat at the Dirichlet-to-Neumann switch (the convex disclination):
    energy within r < 0.5 grew 1.74, 1.99, 2.19, 2.36, 2.50, and the phase on
    the Neumann side rose like r^(1/2) (|k| about 1.4 to 1.5 at the first
    node). That mode costs a logarithmically divergent stretching energy in
    the continuum, but only about log(1/h) on a grid, so fd energies are
    biased low by an amount that decays only logarithmically. Not an
    optimizer effect: gradient norms about 1e-9, and restarts from
    interpolated coarse minima return the same energies. The fd knee/zipper
    crossing therefore drifts with h (about 52 deg at h = 0.25). Also in
    v1: the single-node Dirichlet segment (k = 1) is an fd lattice artifact
    (energy below the knee by 0.0138, 0.0033, 0.0008 per length at
    h = 0.25, 0.125, 0.0625), so fd scans start at k = 2.
v2.1 (2026-10-09). Table mode saves each alpha as soon as it finishes and
    reuses saved alphas on a rerun (--overwrite to recompute); the field files
    carry the row under 'rec_<key>' and also plain copies of eps, q, ell, L,
    delta_zip, delta_knee, k_zip for ellipse_zipper_assemble.py. Figure-label
    thinning in fig1.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_NAME = Path(__file__).stem

LOG2 = np.log(2.0)


def _logcosh(v):
    return np.logaddexp(v, -v) - LOG2


def default_L(alpha, L_min, c_q, c_eps):
    """Domain height. Two decay lengths matter: the knee profile decays like
    exp(-2 q y), and the x-periodic part of a zipper decays like
    exp(-2 eps^2 y) (slow root of the RCN operator linearized about the plane
    wave; the same rate appears in the paper's large-y asymptotics)."""
    eps, q = np.cos(alpha), np.sin(alpha)
    return float(max(L_min, c_q / q, c_eps / (2.0 * eps ** 2)))


# ---------------------------------------------------------------------------
# Shared Levenberg-Marquardt machinery
# ---------------------------------------------------------------------------

class _LMProblem(object):
    """Subclasses provide residual(z) -> (r, aux), jacobian(aux), and
    self.n_res1 = number of bending residuals (the rest are stretching)."""

    def energy_parts(self, z):
        r, _ = self.residual(z)
        Eb = float(r[:self.n_res1] @ r[:self.n_res1])
        Es = float(r[self.n_res1:] @ r[self.n_res1:])
        return Eb + Es, Eb, Es

    def solve(self, z0, max_iter=300, gtol=1e-8, rtol=1e-13, verbose=False):
        z = z0.copy()
        r, aux = self.residual(z)
        E = float(r @ r)
        lam = 1e-3
        small = 0
        it = 0
        status = "max_iter"
        gnorm = np.inf
        for it in range(1, max_iter + 1):
            J = self.jacobian(aux)
            g = J.T @ r
            gnorm = float(np.max(np.abs(g)))
            if gnorm < gtol:
                status = "gtol"
                break
            JTJ = (J.T @ J).tocsc()
            D = JTJ.diagonal() + 1e-14
            accepted = False
            while lam < 1e12:
                A = (JTJ + sp.diags(lam * D)).tocsc()
                step = splu(A).solve(-g)
                zn = z + step
                rn, auxn = self.residual(zn)
                En = float(rn @ rn)
                if En < E:
                    accepted = True
                    break
                lam *= 4.0
            if not accepted:
                status = "stalled"
                break
            dE = E - En
            z, r, aux, E = zn, rn, auxn, En
            lam = max(lam / 3.0, 1e-12)
            small = small + 1 if dE < rtol * max(E, 1e-30) else 0
            if verbose and it % 10 == 0:
                print("      LM it %4d  E=%.10f  |g|=%.2e  lam=%.1e" % (it, E, gnorm, lam))
            if small >= 3:
                status = "rtol"
                break
        return z, dict(E=E, iters=it, gnorm=gnorm, status=status)


class _Geom(object):
    def _geom(self, alpha, h, L):
        self.alpha = float(alpha)
        self.eps = float(np.cos(alpha))
        self.q = float(np.sin(alpha))
        self.ell = np.pi / self.eps
        self.L = float(L)
        self.h = float(h)
        self.m = max(4, int(round(self.ell / h)))
        self.n = max(4, int(round(self.L / h)))
        self.eta = self.ell / self.m
        self.zeta = self.L / self.n
        self.x = self.eta * np.arange(self.m)
        self.y = self.zeta * np.arange(self.n + 1)
        self._k = None


# ---------------------------------------------------------------------------
# Discretization 1: conforming bicubic Hermite (Bogner-Fox-Schmit)
# ---------------------------------------------------------------------------

def _hermite_1d(t, h):
    """Cubic Hermite basis on one interval of length h, in physical units.
    Order: value-left, slope-left, value-right, slope-right.
    Returns (f, df/dx, d2f/dx2), each of shape (4, len(t))."""
    t = np.asarray(t, dtype=float)
    f = np.array([1 - 3 * t ** 2 + 2 * t ** 3, h * (t - 2 * t ** 2 + t ** 3),
                  3 * t ** 2 - 2 * t ** 3, h * (-t ** 2 + t ** 3)])
    d1 = np.array([-6 * t + 6 * t ** 2, h * (1 - 4 * t + 3 * t ** 2),
                   6 * t - 6 * t ** 2, h * (-2 * t + 3 * t ** 2)]) / h
    d2 = np.array([-6 + 12 * t, h * (-4 + 6 * t), 6 - 12 * t, h * (-2 + 6 * t)]) / h ** 2
    return f, d1, d2


class ZipperBFS(_Geom, _LMProblem):
    """Nodes (i, j), i = 0..m (node m = node 0 shifted by pi), j = 0..n.
    Dofs per node: t = 0 theta, 1 theta_x, 2 theta_y, 3 theta_xy."""

    disc = "bfs"

    def __init__(self, alpha, h, L, nq=5, y_fine=8.0, grade=1.15, h_far=1.0):
        self._geom(alpha, h, L)
        self.nq = nq
        # graded mesh in y: uniform spacing h up to y_fine, then sizes growing
        # by the factor `grade` up to h_far (the far field is nearly a plane
        # wave, and bicubic elements represent it well at coarse size).
        ys = [0.0]
        dy = self.h
        while ys[-1] < self.L - 1e-12:
            if ys[-1] >= y_fine - 1e-12:
                dy = min(h_far, dy * grade)
            ys.append(ys[-1] + dy)
        ys = np.array(ys)
        ys *= self.L / ys[-1]
        self.y = ys
        self.n = ys.size - 1
        self.zeta = float(np.min(np.diff(ys)))
        self._build_operators()

    def _ext(self, i, j, t):
        return ((i * (self.n + 1)) + j) * 4 + t

    def _build_operators(self):
        m, n, hx, nq = self.m, self.n, self.eta, self.nq
        hys = np.diff(self.y)                      # element heights, row by row
        g, w = np.polynomial.legendre.leggauss(nq)
        tq, wq = 0.5 * (g + 1.0), 0.5 * w
        fx, d1x, d2x = _hermite_1d(tq, hx)
        nq2 = nq * nq
        loc = [(cx, cy, t) for cx in (0, 1) for cy in (0, 1) for t in range(4)]
        # local basis b = (cx, cy, t) at quad points (qx, qy), one set per element row
        Bx = np.zeros((n, nq2, 16)); By = np.zeros_like(Bx); Bl = np.zeros_like(Bx)
        W = np.zeros((n, nq2))
        for je in range(n):
            fy, d1y, d2y = _hermite_1d(tq, hys[je])
            for b, (cx, cy, t) in enumerate(loc):
                ix = 2 * cx + (1 if t in (1, 3) else 0)
                iy = 2 * cy + (1 if t in (2, 3) else 0)
                Bx[je, :, b] = np.outer(d1x[ix], fy[iy]).ravel()
                By[je, :, b] = np.outer(fx[ix], d1y[iy]).ravel()
                Bl[je, :, b] = (np.outer(d2x[ix], fy[iy]) + np.outer(fx[ix], d2y[iy])).ravel()
            W[je] = (np.outer(wq, wq) * hx * hys[je]).ravel()

        IE, JE = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
        IE, JE = IE.ravel(), JE.ravel()
        nE = IE.size
        rows = (np.arange(nE)[:, None, None] * nq2 + np.arange(nq2)[None, :, None])
        rows = np.broadcast_to(rows, (nE, nq2, 16)).ravel()
        cols_b = np.stack([self._ext(IE + cx, JE + cy, t) for (cx, cy, t) in loc], axis=1)  # (nE, 16)
        cols = np.broadcast_to(cols_b[:, None, :], (nE, nq2, 16)).ravel()
        Next = (m + 1) * (n + 1) * 4
        NQ = nE * nq2

        def op(B):
            vals = B[JE].ravel()                   # (nE, nq2, 16), per-row basis
            return sp.csr_matrix((vals, (rows, cols)), shape=(NQ, Next))

        self.Ox, self.Oy, self.Ol = op(Bx), op(By), op(Bl)
        self.sw = np.sqrt(W[JE].ravel())
        self.n_res1 = NQ

    def set_k(self, k):
        if self._k == k:
            return
        m, n, eps, q, L = self.m, self.n, self.eps, self.q, self.L
        F = np.ones((m, n + 1, 4), dtype=bool)   # free dofs, nodes i < m
        F[:, n, :] = False                        # top: plane wave
        const = np.zeros((m, n + 1, 4))
        const[:, n, 0] = eps * self.x + q * L
        const[:, n, 1] = eps
        const[:, n, 2] = q
        if k == 0:
            F[:, 0, 2:] = False                   # Neumann everywhere
            F[0, 0, 0] = False                    # gauge theta(0,0) = 0
        else:
            F[0, 0, :] = False                    # switch node N->D (clamped)
            F[k, 0, :] = False                    # switch node D->N (clamped)
            F[1:k, 0, 0:2] = False                # Dirichlet: theta = theta_x = 0
            F[k + 1:, 0, 2:] = False              # Neumann: theta_y = theta_xy = 0
        idx = -np.ones(F.shape, dtype=np.int64)
        idx[F] = np.arange(F.sum())
        N = int(F.sum())
        Nz = N + 1
        nrow = m * (n + 1) * 4
        r_free = np.nonzero(F.ravel())[0]
        rows = [r_free]
        cols = [idx.ravel()[r_free]]
        vals = [np.ones(r_free.size)]
        top = np.array([self._ext(i, n, 0) for i in range(m)])
        rows.append(top); cols.append(np.full(m, N)); vals.append(np.ones(m))
        Pc = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(nrow, Nz))
        c = const.ravel().copy()
        # node i = m: copy of node 0, value shifted by pi
        blk = slice(0, (n + 1) * 4)
        cm = c[blk].copy()
        cm[0::4] += np.pi
        P = sp.vstack([Pc, Pc[blk]]).tocsr()
        cext = np.concatenate([c, cm])
        sw = sp.diags(self.sw)
        self.Al = (sw @ (self.Ol @ P)).tocsr()
        self.bl = self.sw * (self.Ol @ cext)
        self.Ax = (self.Ox @ P).tocsr(); self.bx = self.Ox @ cext
        self.Ay = (self.Oy @ P).tocsr(); self.by = self.Oy @ cext
        self.Pc, self.cc, self.F, self.N, self.Nz = Pc, c, F, N, Nz
        self._k = k

    def residual(self, z):
        lap = self.Al @ z + self.bl
        gx = self.Ax @ z + self.bx
        gy = self.Ay @ z + self.by
        return np.concatenate([lap, self.sw * (1.0 - gx * gx - gy * gy)]), (gx, gy)

    def jacobian(self, aux):
        gx, gy = aux
        Jg = sp.diags(-2.0 * self.sw * gx) @ self.Ax + sp.diags(-2.0 * self.sw * gy) @ self.Ay
        return sp.vstack([self.Al, Jg]).tocsr()

    # state = dof array (m, n+1, 4)
    def z_from_state(self, D, delta):
        z = np.empty(self.Nz)
        z[:self.N] = D[self.F]
        z[self.N] = delta
        return z

    def state_from_z(self, z):
        return (self.Pc @ z + self.cc).reshape(self.m, self.n + 1, 4)

    @staticmethod
    def nodal_theta(D):
        return D[:, :, 0]

    @staticmethod
    def gauge_shift(D, c):
        D = D.copy()
        D[:, :, 0] -= c
        return D

    def _state(self, th, thx, thy):
        D = np.zeros((self.m, self.n + 1, 4))
        D[:, :, 0], D[:, :, 1], D[:, :, 2] = th, thx, thy
        return D

    def knee_state_exact(self):
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        D = self._state(self.eps * X + _logcosh(self.q * Y), self.eps + 0 * X, self.q * np.tanh(self.q * Y))
        return D, knee_delta_L(self)

    def knee_state_offset(self):
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        s = np.sqrt((self.q * Y) ** 2 + 4.0)
        return self._state(self.eps * X + s - 2.0, self.eps + 0 * X, self.q ** 2 * Y / s), 0.0


# ---------------------------------------------------------------------------
# Discretization 2: the paper's finite differences (comparison only)
# ---------------------------------------------------------------------------

class ZipperFD(_Geom, _LMProblem):
    disc = "fd"

    def __init__(self, alpha, h, L):
        self._geom(alpha, h, L)
        self._build_stencils()

    def _e(self, i, j):
        return (i + 1) * (self.n + 3) + (j + 1)

    def _build_stencils(self):
        m, n, eta, zeta = self.m, self.n, self.eta, self.zeta
        I, J = np.meshgrid(np.arange(m), np.arange(n + 1), indexing="ij")
        I, J = I.ravel(), J.ravel()
        Nn = I.size
        Ne = (m + 2) * (n + 3)
        p = np.arange(Nn)
        e = self._e

        def mat(entries):
            rows = np.concatenate([p for _ in entries])
            cols = np.concatenate([c for c, _ in entries])
            vals = np.concatenate([np.full(Nn, v) for _, v in entries])
            return sp.csr_matrix((vals, (rows, cols)), shape=(Nn, Ne))

        self.S_lap = mat([(e(I + 1, J), 1 / eta ** 2), (e(I - 1, J), 1 / eta ** 2),
                          (e(I, J), -2 / eta ** 2 - 2 / zeta ** 2),
                          (e(I, J + 1), 1 / zeta ** 2), (e(I, J - 1), 1 / zeta ** 2)])
        self.S_d = [mat([(e(I + 1, J), 1 / eta), (e(I, J), -1 / eta)]),
                    mat([(e(I, J), 1 / eta), (e(I - 1, J), -1 / eta)]),
                    mat([(e(I, J + 1), 1 / zeta), (e(I, J), -1 / zeta)]),
                    mat([(e(I, J), 1 / zeta), (e(I, J - 1), -1 / zeta)])]
        w = np.ones(Nn)
        w[(J == 0) | (J == n)] = 0.5
        self.s = np.sqrt(eta * zeta * w)
        self.n_res1 = Nn

    def set_k(self, k):
        if self._k == k:
            return
        m, n, eps, q, L, zeta = self.m, self.n, self.eps, self.q, self.L, self.zeta
        free = np.ones((m, n), dtype=bool)
        free[:k, 0] = False
        idx = -np.ones((m, n), dtype=np.int64)
        idx.T[free.T] = np.arange(free.sum())
        N = int(free.sum())
        Nz = N + 1
        nb = n + 3
        rows, cols, vals = [], [], []
        const = np.zeros(m * nb)

        def core(i, j):
            return i * nb + (j + 1)

        ii, jj = np.nonzero(free)
        rows.append(core(ii, jj)); cols.append(idx[ii, jj]); vals.append(np.ones(ii.size))
        i_all = np.arange(m)
        rows.append(core(i_all, n)); cols.append(np.full(m, N)); vals.append(np.ones(m))
        const[core(i_all, n)] = eps * self.x + q * L
        rows.append(core(i_all, n + 1)); cols.append(np.full(m, N)); vals.append(np.ones(m))
        const[core(i_all, n + 1)] = eps * self.x + q * (L + zeta)
        sgn = np.where(i_all < k, -1.0, 1.0)
        rows.append(core(i_all, -1)); cols.append(idx[i_all, 1]); vals.append(sgn)
        Pc = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(m * nb, Nz))
        last = slice((m - 1) * nb, m * nb)
        first = slice(0, nb)
        P = sp.vstack([Pc[last], Pc, Pc[first]]).tocsr()
        c = np.concatenate([const[last] - np.pi, const, const[first] + np.pi])
        s = sp.diags(self.s)
        self.A_lap = (s @ (self.S_lap @ P)).tocsr()
        self.b_lap = self.s * (self.S_lap @ c)
        self.A_d = [(Sd @ P).tocsr() for Sd in self.S_d]
        self.b_d = [Sd @ c for Sd in self.S_d]
        self.Pc, self.cc = Pc, const
        self.free, self.N, self.Nz = free, N, Nz
        self._k = k

    def residual(self, z):
        lap = self.A_lap @ z + self.b_lap
        d = [A @ z + b for A, b in zip(self.A_d, self.b_d)]
        g = 0.5 * sum(di * di for di in d) - 1.0
        return np.concatenate([lap, self.s * g]), d

    def jacobian(self, d):
        Jg = sum(sp.diags(di) @ A for di, A in zip(d, self.A_d))
        return sp.vstack([self.A_lap, sp.diags(self.s) @ Jg]).tocsr()

    # state = theta at nodes (m, n+1)
    def z_from_state(self, theta, delta):
        z = np.empty(self.Nz)
        z[:self.N] = theta[:, :self.n].T[self.free.T]
        z[self.N] = delta
        return z

    def state_from_z(self, z):
        F = (self.Pc @ z + self.cc).reshape(self.m, self.n + 3)
        return F[:, 1:self.n + 2].copy()

    @staticmethod
    def nodal_theta(theta):
        return theta

    @staticmethod
    def gauge_shift(theta, c):
        return theta - c

    def knee_state_exact(self):
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        return self.eps * X + _logcosh(self.q * Y), knee_delta_L(self)

    def knee_state_offset(self):
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        return self.eps * X + np.sqrt((self.q * Y) ** 2 + 4.0) - 2.0, 0.0


def make_problem(disc, alpha, h, L, nq=5, y_fine=8.0, grade=1.15, h_far=1.0):
    if disc == "bfs":
        return ZipperBFS(alpha, h, L, nq=nq, y_fine=y_fine, grade=grade, h_far=h_far)
    if disc == "fd":
        return ZipperFD(alpha, h, L)
    raise ValueError("disc must be 'bfs' or 'fd'")


# ---------------------------------------------------------------------------
# Exact knee and solve helpers
# ---------------------------------------------------------------------------

def knee_delta_L(P):
    """delta that the exact knee has at the top row y = L."""
    return float(_logcosh(P.q * P.L) - P.q * P.L)


def knee_energy_analytic(P):
    """Exact knee energy per period on [0, ell) x [0, L] (self-dual, so the
    integrand is 2 q^4 sech^4(q y)); tends to 4 pi (1 - eps^2)^(3/2) / (3 eps),
    i.e. (4/3) sin^3(alpha) per unit length."""
    T = np.tanh(P.q * P.L)
    return 2.0 * P.ell * P.q ** 3 * (T - T ** 3 / 3.0)


def _pack(P, z, info, k):
    E, Eb, Es = P.energy_parts(z)
    return dict(k=k, a=k / P.m, state=P.state_from_z(z), delta=float(z[-1]), E=E, Eb=Eb, Es=Es,
                iters=info["iters"], gnorm=info["gnorm"], status=info["status"])


def solve_knee(P, **kw):
    P.set_k(0)
    S0, d0 = P.knee_state_offset()   # deliberately not the knee
    z, info = P.solve(P.z_from_state(S0, d0), **kw)
    sol = _pack(P, z, info, 0)
    # a = 0 has no Dirichlet node, so theta -> theta + c is a symmetry. Gauge:
    # theta(0,0) = 0 (bfs pins it; fd is shifted after the solve).
    c = float(P.nodal_theta(sol["state"])[0, 0])
    sol["state"] = P.gauge_shift(sol["state"], c)
    sol["delta"] -= c
    return sol


def solve_k(P, k, state0, delta0, **kw):
    P.set_k(k)
    z, info = P.solve(P.z_from_state(state0, delta0), **kw)
    return _pack(P, z, info, k)


def zipper_at_a(P, a_target, n_steps=6, **kw):
    """Continuation from the knee to the k nearest a_target."""
    knee = solve_knee(P, **kw)
    k = max(1, int(round(a_target * P.m)))
    prev = knee
    for kk in np.unique(np.round(np.linspace(1, k, n_steps)).astype(int)):
        prev = solve_k(P, int(kk), prev["state"], prev["delta"], **kw)
    return knee, prev


# ---------------------------------------------------------------------------
# One alpha: knee + scan over a
# ---------------------------------------------------------------------------

def scan_alpha(alpha_deg, args, verbose=True):
    t0 = time.time()
    alpha = np.radians(alpha_deg)
    L = default_L(alpha, args.L_min, args.c_q, args.c_eps) if args.L is None else args.L
    P = make_problem(args.disc, alpha, args.h, L, nq=args.nq, y_fine=args.y_fine, grade=args.grade, h_far=args.h_far)
    kw = dict(max_iter=args.max_iter, gtol=args.gtol)
    k_min = args.k_min if args.k_min is not None else (1 if args.disc == "bfs" else 2)
    lines = ["alpha=%.2f deg  disc=%s  eps=%.4f  ell=%.3f  L=%.1f  m=%d n=%d"
             % (alpha_deg, args.disc, P.eps, P.ell, P.L, P.m, P.n)]

    knee = solve_knee(P, **kw)
    sols = {0: knee}
    coarse = np.unique(np.round(np.linspace(k_min, P.m - 1, args.n_a_coarse)).astype(int))
    prev = knee
    for k in coarse:
        sol = solve_k(P, int(k), prev["state"], prev["delta"], **kw)
        sols[int(k)] = sol
        prev = sol
        lines.append("   k=%3d a=%.3f  E/ell=%.6f  delta=%+.4f  %s(%d)"
                     % (k, k / P.m, sol["E"] / P.ell, sol["delta"], sol["status"], sol["iters"]))

    ks = sorted(k for k in sols if k > 0)
    kb = min(ks, key=lambda k: sols[k]["E"])
    pos = ks.index(kb)
    kL = ks[pos - 1] if pos > 0 else ks[0]
    kR = ks[pos + 1] if pos < len(ks) - 1 else ks[-1]
    prev = sols[kL]
    for k in range(kL + 1, kR):
        if k in sols:
            prev = sols[k]
            continue
        sol = solve_k(P, k, prev["state"], prev["delta"], **kw)
        sols[k] = sol
        prev = sol
    ks = sorted(k for k in sols if k > 0)
    kb = min(ks, key=lambda k: sols[k]["E"])
    edge = kb in (ks[0], ks[-1])
    zb = sols[kb]
    n_bad = sum(sols[k]["status"] in ("stalled", "max_iter") for k in sols)

    rec = dict(alpha_deg=float(alpha_deg), eps=P.eps, q=P.q, ell=P.ell, L=P.L, m=P.m, n=P.n,
               h=P.h, eta=P.eta, zeta=P.zeta,
               E_knee=knee["E"], Eb_knee=knee["Eb"], Es_knee=knee["Es"], delta_knee=knee["delta"],
               E_knee_exact=knee_energy_analytic(P), delta_knee_exact=knee_delta_L(P),
               k_zip=int(kb), a_zip=kb / P.m, E_zip=zb["E"], Eb_zip=zb["Eb"], Es_zip=zb["Es"],
               delta_zip=zb["delta"], neumann_len=(P.m - kb) * P.eta,
               edge_flag=bool(edge), n_unconverged=int(n_bad),
               branch="zipper" if zb["E"] < knee["E"] else "knee",
               seconds=time.time() - t0)
    ksort = sorted(sols)
    curve = dict(k=np.array(ksort), a=np.array(ksort) / P.m,
                 E=np.array([sols[k]["E"] for k in ksort]),
                 delta=np.array([sols[k]["delta"] for k in ksort]))
    fields = dict(x=P.x, y=P.y, theta_knee=P.nodal_theta(knee["state"]),
                  theta_zip=P.nodal_theta(zb["state"]))
    if P.disc == "bfs":
        fields["dofs_zip"] = zb["state"]
        fields["dofs_knee"] = knee["state"]
    flag = "  !! EDGE: best a>0 at the scan boundary" if edge else ""
    flag += ("  !! %d solves not converged" % n_bad) if n_bad else ""
    lines.append("   -> knee E/ell=%.6f (exact %.6f) | zipper k=%d a=%.3f E/ell=%.6f delta=%+.4f | %s  [%.0fs]%s"
                 % (knee["E"] / P.ell, rec["E_knee_exact"] / P.ell, kb, kb / P.m, zb["E"] / P.ell,
                    zb["delta"], rec["branch"], rec["seconds"], flag))
    if verbose:
        print("\n".join(lines), flush=True)
    return rec, curve, fields


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def run_controls(args, out):
    lines = []

    def say(s):
        print(s, flush=True)
        lines.append(s)

    kw = dict(max_iter=args.max_iter, gtol=args.gtol)
    say("== Planted control 1: exact knee (a = 0), disc = %s ==" % args.disc)
    say("alpha  h      E(exact field)/E_exact-1  E_min/E_exact-1  delta_min-delta_exact  "
        "max|theta_min-theta_exact|  Eb/Es   LM")
    for adeg in args.control_alphas:
        alpha = np.radians(adeg)
        L = default_L(alpha, args.L_min, args.c_q, args.c_eps)
        for h in args.control_hs:
            P = make_problem(args.disc, alpha, h, L, nq=args.nq, y_fine=args.y_fine, grade=args.grade, h_far=args.h_far)
            P.set_k(0)
            S_ex, d_ex = P.knee_state_exact()
            E_disc = P.energy_parts(P.z_from_state(S_ex, d_ex))[0]
            E_ex = knee_energy_analytic(P)
            kn = solve_knee(P, **kw)
            err = np.max(np.abs(P.nodal_theta(kn["state"]) - P.nodal_theta(S_ex)))
            say("%5.1f  %.3f  %+.3e  %+.3e  %+.3e  %.3e  %.5f  %s(%d)"
                % (adeg, h, E_disc / E_ex - 1, kn["E"] / E_ex - 1, kn["delta"] - d_ex, err,
                   kn["Eb"] / kn["Es"], kn["status"], kn["iters"]))

    say("")
    say("== Control 2: one zipper (alpha = %.1f deg, a near %.2f), convergence in h then L =="
        % (args.control_zip_alpha, args.control_zip_a))
    alpha = np.radians(args.control_zip_alpha)
    L0 = default_L(alpha, args.L_min, args.c_q, args.c_eps)
    combos = [(h, L0) for h in args.control_zip_hs] + [(args.control_zip_hs[0], f * L0) for f in (1.5, 2.0)]
    for h, L in combos:
        t0 = time.time()
        P = make_problem(args.disc, alpha, h, L, nq=args.nq, y_fine=args.y_fine, grade=args.grade, h_far=args.h_far)
        knee, z = zipper_at_a(P, args.control_zip_a, **kw)
        say("  h=%.4f L=%5.1f  knee E/ell=%.6f | zipper a=%.4f E/ell=%.6f delta=%+.5f | "
            "dE/ell=%+.6f  %s  [%.0fs]"
            % (h, L, knee["E"] / P.ell, z["a"], z["E"] / P.ell, z["delta"],
               (z["E"] - knee["E"]) / P.ell, z["status"], time.time() - t0))
    (out / ("controls_%s.txt" % args.disc)).write_text("\n".join(lines) + "\n")
    print("wrote", out / ("controls_%s.txt" % args.disc))


# ---------------------------------------------------------------------------
# Table + figures
# ---------------------------------------------------------------------------

def find_switch(alpha_deg, dE):
    """Sign changes of dE = (E_zip - E_knee)/ell from + to - (linear interp)."""
    out = []
    for i in range(len(dE) - 1):
        if (dE[i] > 0 >= dE[i + 1]) or (dE[i] >= 0 > dE[i + 1]):
            t = dE[i] / (dE[i] - dE[i + 1])
            out.append(float(alpha_deg[i] + t * (alpha_deg[i + 1] - alpha_deg[i])))
    return out


def _save_alpha(out, rec, curve, fields):
    fdir = out / "fields"
    fdir.mkdir(exist_ok=True)
    tag = "alpha%05.2f" % rec["alpha_deg"]
    payload = dict(fields)
    payload.update({"curve_" + k: v for k, v in curve.items()})
    payload.update({"rec_" + k: v for k, v in rec.items()})
    # plain copies of the scalars that ellipse_zipper_assemble.py reads (format
    # of the first v2 table run), so old and new readers both work
    for k in ("eps", "q", "ell", "L", "delta_zip", "delta_knee", "k_zip"):
        payload[k] = rec[k]
    np.savez(fdir / ("%s.npz" % tag), **payload)


def _load_alpha(out, adeg):
    p = out / "fields" / ("alpha%05.2f.npz" % adeg)
    if not p.exists():
        return None
    F = np.load(p, allow_pickle=False)
    if "rec_alpha_deg" not in F.files:
        return None
    rec = {k[4:]: F[k].item() for k in F.files if k.startswith("rec_")}
    curve = {k[6:]: F[k] for k in F.files if k.startswith("curve_")}
    fields = {k: F[k] for k in F.files if not (k.startswith("rec_") or k.startswith("curve_"))}
    return rec, curve, fields


def _scan_star(a_args):
    return scan_alpha(*a_args)


def run_table(args, out):
    """Each alpha is saved to fields/ as soon as it finishes; alphas already on
    disk (same out_dir) are reused unless --overwrite is given."""
    alphas = list(args.alphas_deg)
    results = []
    todo = []
    for a in alphas:
        got = None if args.overwrite else _load_alpha(out, float(a))
        if got is not None:
            print("reusing saved alpha = %.2f" % a)
            results.append(got)
        else:
            todo.append(a)
    if args.n_jobs > 1 and len(todo) > 1:
        from multiprocessing import Pool
        with Pool(args.n_jobs) as pool:
            for res in pool.imap_unordered(_scan_star, [(a, args, True) for a in todo]):
                _save_alpha(out, *res)
                results.append(res)
    else:
        for a in todo:
            res = scan_alpha(a, args, True)
            _save_alpha(out, *res)
            results.append(res)
    results.sort(key=lambda r: r[0]["alpha_deg"])

    recs = [r[0] for r in results]
    keys = list(recs[0].keys())
    table = {k: np.array([r[k] for r in recs]) for k in keys}
    dE = (table["E_zip"] - table["E_knee"]) / table["ell"]
    switches = find_switch(table["alpha_deg"], dE)

    np.savez(out / "zipper_table.npz", **table)
    with open(out / "zipper_table.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in recs:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    summary = dict(script=SCRIPT_NAME, disc=args.disc, h=args.h, nq=args.nq,
                   L_rule=dict(L_min=args.L_min, c_q=args.c_q, c_eps=args.c_eps),
                   rcn_switch_deg=switches,
                   note=("switch = alpha where the best a>0 zipper and the knee have equal energy; "
                         "range reported in the variational paper: 0.35 pi to 0.37 pi (63 to 66.6 deg), "
                         "computed there with the fd discretization"),
                   edge_flag_alphas=[float(a) for a, e in zip(table["alpha_deg"], table["edge_flag"]) if e],
                   unconverged_alphas=[float(a) for a, n in zip(table["alpha_deg"], table["n_unconverged"]) if n])
    (out / "zipper_table_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nRCN switch angle(s): %s deg" % ", ".join("%.2f" % s for s in switches))
    if summary["edge_flag_alphas"]:
        print("!! LOUD FLAG: best a>0 at the scan edge (no interior zipper minimum) for alpha =",
              summary["edge_flag_alphas"])
    if summary["unconverged_alphas"]:
        print("!! LOUD FLAG: unconverged LM solves for alpha =", summary["unconverged_alphas"])
    make_figures(table, results, switches, out, args)


def make_figures(table, results, switches, out, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "axes.grid": True, "grid.color": "#e6e5e0",
                         "grid.linewidth": 0.6})
    A = table["alpha_deg"]
    ell = table["ell"]
    edge = table["edge_flag"].astype(bool)

    # 1. line energy vs a
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cmap = plt.get_cmap("Blues")
    for j, (rec, curve, _) in enumerate(results):
        col = cmap(0.35 + 0.65 * j / max(1, len(results) - 1))
        ax.plot(curve["a"], curve["E"] / rec["ell"], color=col, lw=1.5)
        ax.plot([0], [rec["E_knee"] / rec["ell"]], "o", color=col, ms=4)
        if not rec["edge_flag"]:
            ax.plot([rec["a_zip"]], [rec["E_zip"] / rec["ell"]], "o", mfc="none", mec=col, ms=8, mew=1.5)
        if j in (0, len(results) // 2, len(results) - 2, len(results) - 1):
            ax.annotate("%.0f deg" % rec["alpha_deg"], (curve["a"][-1], curve["E"][-1] / rec["ell"]),
                        fontsize=7, color=INK, xytext=(3, 0), textcoords="offset points")
    ax.set_xlabel("Dirichlet fraction a  (a = 0 is the knee)")
    ax.set_ylabel("energy per unit length  E / ell")
    ax.set_title("RCN energy against a, one curve per far-field angle (disc = %s, h = %.3g)\n"
                 "dot: knee; ring: interior a > 0 minimum" % (args.disc, args.h), fontsize=9)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out / "fig1_energy_vs_a.png", dpi=180)
    plt.close(fig)

    # 2. branch line energies vs alpha
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.axvspan(63.0, 66.6, color="#d9d8d3", alpha=0.6, lw=0,
               label="switch range reported in the variational paper")
    ax.axvline(43.0, color=MUTED, ls=":", lw=1, label="43 deg sharp-interface rule")
    ax.axvline(47.3, color=MUTED, ls="--", lw=1, label="SH first dislocation (about 47 deg)")
    for s in switches:
        ax.axvline(s, color=INK, lw=1.2, label="RCN switch, this table (%.1f deg)" % s)
    ag = np.linspace(A.min(), A.max(), 400)
    ax.plot(ag, (4.0 / 3.0) * np.sin(np.radians(ag)) ** 3, color=MUTED, lw=1,
            label="knee, exact: (4/3) sin^3 alpha")
    ax.plot(A, table["E_knee"] / ell, "o", color=BLUE, ms=5, label="knee (a = 0), computed")
    ax.plot(A[~edge], (table["E_zip"] / ell)[~edge], "-o", color=ORANGE, ms=5, lw=2,
            label="zipper, interior a > 0 minimum")
    if edge.any():
        ax.plot(A[edge], (table["E_zip"] / ell)[edge], "x", color=ORANGE, ms=6,
                label="zipper, no interior minimum (scan edge)")
    ax.set_xlabel("far-field angle alpha (deg)")
    ax.set_ylabel("energy per unit length  E / ell")
    ax.set_title("Knee and zipper branches (disc = %s, h = %.3g)" % (args.disc, args.h), fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig2_branches.png", dpi=180)
    plt.close(fig)

    # 3. parameters (interior minima only)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    axs[0].plot(A[~edge], table["a_zip"][~edge], "-o", color=ORANGE, ms=4, lw=1.5)
    axs[0].set_ylabel("best a  (zipper branch)")
    axs[1].plot(A[~edge], table["delta_zip"][~edge], "-o", color=ORANGE, ms=4, lw=1.5, label="zipper delta*")
    axs[1].plot(A, table["delta_knee"], "o", color=BLUE, ms=4, label="knee delta")
    axs[1].axhline(-LOG2, color=MUTED, lw=1, ls="--", label="-log 2")
    axs[1].set_ylabel("far-field phase offset delta")
    axs[1].legend(fontsize=7, frameon=False)
    axs[2].plot(A[~edge], table["neumann_len"][~edge], "-o", color=ORANGE, ms=4, lw=1.5)
    axs[2].set_ylabel("Neumann length (1 - a) ell")
    for ax in axs:
        ax.set_xlabel("alpha (deg)")
        for s in switches:
            ax.axvline(s, color=INK, lw=1)
    fig.suptitle("Zipper-branch parameters at interior minima (vertical line: RCN switch, this table)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig3_parameters.png", dpi=180)
    plt.close(fig)

    # 4. gallery of cos(theta), two periods, mirrored about the core
    pick = [r for r in results if r[0]["alpha_deg"] in [float(g) for g in args.gallery_alphas]] or results[::3]
    fig, axs = plt.subplots(2, len(pick), figsize=(2.3 * len(pick), 5.8), squeeze=False)
    for c, (rec, _, fl) in enumerate(pick):
        for r_, (key, lab) in enumerate([("theta_knee", "knee"),
                                         ("theta_zip", "zipper a=%.2f" % rec["a_zip"])]):
            th = fl[key]
            x, y = fl["x"], fl["y"]
            th2 = np.concatenate([th, th + np.pi], axis=0)
            x2 = np.concatenate([x, x + rec["ell"]])
            ymax = min(y[-1], max(4 * np.pi, 0.6 * rec["ell"]))
            sel = y <= ymax
            u = np.cos(th2[:, sel])
            U = np.concatenate([u[:, :0:-1], u], axis=1)
            Yf = np.concatenate([-y[sel][:0:-1], y[sel]])
            ax = axs[r_, c]
            # pcolormesh with the true coordinates (the bfs mesh is graded in y)
            ax.pcolormesh(x2, Yf, U.T, cmap="RdBu_r", vmin=-1, vmax=1, shading="gouraud")
            ax.set_aspect("equal")
            ttl = "%.0f deg, %s" % (rec["alpha_deg"], lab)
            if r_ == 1 and rec["edge_flag"]:
                ttl += "\n(no interior min)"
            ax.set_title(ttl, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("cos(theta), two periods, mirrored about the core line", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig4_gallery.png", dpi=180)
    plt.close(fig)
    print("figures written to", out)


# ---------------------------------------------------------------------------

def main(args):
    out = Path(args.out_dir) if args.out_dir else SCRIPT_DIR / "results" / SCRIPT_NAME
    out.mkdir(parents=True, exist_ok=True)
    print("output:", out)
    (out / ("args_%s.json" % args.mode)).write_text(
        json.dumps({k: getattr(args, k) for k in dir(_Args) if not k.startswith("_")}, indent=2, default=str))
    if args.mode == "controls":
        run_controls(args, out)
    elif args.mode == "table":
        run_table(args, out)
    elif args.mode == "figures":
        redraw_figures(args, out)
    else:
        raise ValueError("mode must be 'controls', 'table' or 'figures'")


def redraw_figures(args, out):
    """Rebuild the figures from a finished table in `out` (no solving)."""
    T = dict(np.load(out / "zipper_table.npz"))
    results = []
    for i, adeg in enumerate(T["alpha_deg"]):
        rec = {k: (v[i].item() if hasattr(v[i], "item") else v[i]) for k, v in T.items()}
        F = np.load(out / "fields" / ("alpha%05.2f.npz" % adeg))
        curve = dict(k=F["curve_k"], a=F["curve_a"], E=F["curve_E"], delta=F["curve_delta"])
        fields = dict(x=F["x"], y=F["y"], theta_knee=F["theta_knee"], theta_zip=F["theta_zip"])
        results.append((rec, curve, fields))
    dE = (T["E_zip"] - T["E_knee"]) / T["ell"]
    make_figures(T, results, find_switch(T["alpha_deg"], dE), out, args)


class _Args(object):
    mode = "table"              # "controls", "table", or "figures" (redraw from a finished table)
    disc = "bfs"                # "bfs" (conforming, default) or "fd" (paper's finite differences)
    h = 0.25                    # target grid / element size (wavelength is 2 pi)
    nq = 5                      # Gauss points per direction per element (bfs)
    y_fine = 8.0                # bfs: uniform elements of size h for y < y_fine
    grade = 1.15                # bfs: growth factor of element height beyond y_fine
    h_far = 1.0                 # bfs: largest element height
    L = None                    # domain height; None -> default_L rule
    L_min = 30.0
    c_q = 10.0                  # L >= c_q / q           (knee decay exp(-2 q y))
    c_eps = 4.0                 # L >= c_eps / (2 eps^2) (zipper decay exp(-2 eps^2 y))
    alphas_deg = [30, 35, 40, 43, 45, 47, 50, 53, 56, 59, 62, 64, 66, 68, 71, 75, 80]
    n_a_coarse = 16             # coarse a-scan points before local refinement
    k_min = None                # smallest Dirichlet segment in elements; None -> 1 (bfs), 2 (fd)
    max_iter = 300
    gtol = 1e-8
    n_jobs = 1
    overwrite = False           # table: recompute alphas already saved in out_dir/fields
    control_alphas = [40.0, 60.0, 75.0]
    control_hs = [0.5, 0.25, 0.125]
    control_zip_alpha = 64.0
    control_zip_a = 0.45
    control_zip_hs = [0.5, 0.354, 0.25, 0.177, 0.125]
    gallery_alphas = [40, 53, 62, 68, 75, 80]
    out_dir = None


def _parse_cli():
    d = _Args()
    p = argparse.ArgumentParser(description="RCN zipper table (Stage 0 of the ellipse construction)")
    p.add_argument("--mode", default=d.mode, choices=["controls", "table", "figures"])
    p.add_argument("--disc", default=d.disc, choices=["bfs", "fd"])
    p.add_argument("--h", type=float, default=d.h)
    p.add_argument("--nq", type=int, default=d.nq)
    p.add_argument("--y_fine", type=float, default=d.y_fine)
    p.add_argument("--grade", type=float, default=d.grade)
    p.add_argument("--h_far", type=float, default=d.h_far)
    p.add_argument("--L", type=float, default=d.L)
    p.add_argument("--L_min", type=float, default=d.L_min)
    p.add_argument("--c_q", type=float, default=d.c_q)
    p.add_argument("--c_eps", type=float, default=d.c_eps)
    p.add_argument("--alphas_deg", type=float, nargs="+", default=d.alphas_deg)
    p.add_argument("--n_a_coarse", type=int, default=d.n_a_coarse)
    p.add_argument("--k_min", type=int, default=d.k_min)
    p.add_argument("--max_iter", type=int, default=d.max_iter)
    p.add_argument("--gtol", type=float, default=d.gtol)
    p.add_argument("--n_jobs", type=int, default=d.n_jobs)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--control_alphas", type=float, nargs="+", default=d.control_alphas)
    p.add_argument("--control_hs", type=float, nargs="+", default=d.control_hs)
    p.add_argument("--control_zip_alpha", type=float, default=d.control_zip_alpha)
    p.add_argument("--control_zip_a", type=float, default=d.control_zip_a)
    p.add_argument("--control_zip_hs", type=float, nargs="+", default=d.control_zip_hs)
    p.add_argument("--gallery_alphas", type=float, nargs="+", default=d.gallery_alphas)
    p.add_argument("--out_dir", default=d.out_dir)
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_cli() if len(sys.argv) > 1 else _Args())
