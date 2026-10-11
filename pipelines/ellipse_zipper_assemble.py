"""
ellipse_zipper_assemble.py

Stages 1 and 2 of the ellipse-from-zippers construction: lay out the
dislocations along the major axis of an ellipse (Stage 1) and assemble a
phase field theta on the whole ellipse from the RCN zipper table (Stage 2).
Pure Cross-Newell: no Swift-Hohenberg data is used except for an optional
side-by-side comparison.

Geometry (verified numerically in the checks)
---------------------------------------------
Ellipse X^2/A^2 + Y^2/B^2 = 1, c^2 = A^2 - B^2. The wall normals meet on the
medial segment |X| <= c^2/A. On that segment the distance to the wall is
    d(X) = B sqrt(1 - X^2/c^2)          (exact),
and the angle alpha of the wall normal (= local zipper angle) obeys
    cos alpha = |d'(X)|.
So one zipper period pi/cos(alpha) is exactly a drop of pi in d.

Phase convention
----------------
Rolls parallel to the wall: the far field is the eikonal
    theta_far(P) = Theta0 - d(P),
increasing toward the wall (the zipper convention: phase grows away from the
core). Theta0, the wall phase, is free; it is set through s_center below.

Stage 1: layout along the axis (right half; the left half is the mirror)
-------------------------------------------------------------------------
Phase variable  s(X) = Theta0 - d(X) - delta*(alpha(X)).
In the zipper table, delta is measured from the start of a Dirichlet segment,
so along the axis:
    Dirichlet segment starts (concave junctions):  s = n pi
    convex tips (D -> N switch):                    s = n pi + a*(alpha) pi
in the zipper zone alpha(X) >= alpha_switch. Each period between successive
starts is one "strip" and gets the table zipper nearest its tip angle. The
center is the middle of a Dirichlet segment (a roll lies along the axis);
s_center = s(0) - n0 pi in [0, a* pi) sets Theta0 and hence how long that
central roll is. Leading-order RCN does not fix s_center; it is swept.

Stage 2: assembly
-----------------
Every point P (upper half; the pattern cos theta is mirrored) lies on one wall
normal with foot X_foot on the axis and height |Y| above it. Then
    theta(P) = theta_far(P) + R(P),
with R the zipper correction (zipper zone) or the exact self-dual knee
correction (elsewhere). Zipper zone: the zipper frame point is
x = (s(X_foot) - n pi)/eps_n + |Y| cot(alpha_n), y = |Y|, and R is the stored
zipper minus its own far field, evaluated with the bicubic Hermite dofs.
Knee zone: theta_SD = Theta0 + log v, v(P) = sum_j w_j exp(n_j . P - h_j) over
wall normals n_j with support values h_j. Each term solves lap v = v (|n_j| = 1),
so theta_SD solves the self-dual equation lap theta = 1 - |grad theta|^2
exactly; the weights are iterated so that v = 1 (theta = Theta0) on the wall.
This is the self-dual knee-everywhere solution of the global-description paper,
written as a superposition, and it treats the two terminal foci correctly.
The zipper zone ends at the last convex tip with alpha >= alpha_switch; R is
blended to the knee over the following Neumann stretch.

Thresholds: this version implements the pure RCN switch (from the zipper table
summary). Thresholds below the zipper spinodal (about 65 deg) need constrained
zippers (Stage 0b) and are refused with a loud flag.

Outputs: pipelines/results/ellipse_zipper_assemble/

Changelog
---------
v1.1 (2026-10-09). ZipperLookup reads both field-file formats written by
    zipper_table.py (plain keys, or 'rec_' prefixed from incremental saving);
    a locally regenerated table failed with KeyError 'eps' before this.
v1 (2026-10-08). First version. Measured on the runner ellipse (A = 63.6,
    B = 42.4, dx = 0.25): axis formula exact to 1e-12; self-dual wall
    condition 2e-10 and finite-difference residual median 4e-6; knee-core
    offset against the single-branch outer field = log 2 to 5 digits; outer
    focusing term within 0.08 of 0.5 log(rho/(rho-d)). Two design choices
    came from these runs. (i) The far field is the single-branch self-dual
    field, not the bare eikonal: with the eikonal the knee-core offset read
    1.21 to 1.44 instead of log 2. (ii) Zippers are blended between the two
    table angles bracketing the local alpha, after a tip-aligned phase remap:
    with one table zipper per period the Dirichlet-core mismatch reached
    |sin theta| = 0.15 and the period seams showed jumps of 0.37 rad; with
    the blend these are 0.002 and 0.10 (below the within-period 99th
    percentile 0.14).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_NAME = Path(__file__).stem
LOG2 = np.log(2.0)


# ---------------------------------------------------------------------------
# Ellipse geometry
# ---------------------------------------------------------------------------

class Ellipse(object):
    def __init__(self, A, B):
        self.A, self.B = float(A), float(B)
        self.c2 = self.A ** 2 - self.B ** 2
        self.Xm = self.c2 / self.A           # half-length of the medial segment

    # ---- on the axis -------------------------------------------------------
    def d_axis(self, X):
        X = np.minimum(np.abs(X), self.Xm)
        return self.B * np.sqrt(1.0 - X ** 2 / self.c2)

    def alpha_axis(self, X):
        """Angle of the wall normal through (X, 0), measured from the axis."""
        X = np.minimum(np.abs(X), self.Xm)
        ct = np.clip(self.A * X / self.c2, -1.0, 1.0)     # cos of parametric angle
        st = np.sqrt(1.0 - ct ** 2)
        return np.arctan2(st / self.B, ct / self.A)

    # ---- nearest wall point for points in the first quadrant -------------------
    def foot(self, X, Y, n_seed=2048, n_newton=8):
        """Parametric angle t of the nearest wall point for (X, Y) with X, Y >= 0.
        Returns t, distance d, axis foot X_foot = c^2 cos t / A, normal angle."""
        A, B, c2 = self.A, self.B, self.c2
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        ts = np.linspace(0.0, 0.5 * np.pi, n_seed)
        cs, sn = np.cos(ts), np.sin(ts)
        t = np.empty(X.shape)
        flatX, flatY, flatt = X.ravel(), Y.ravel(), t.ravel()
        for i0 in range(0, flatX.size, 4096):
            sl = slice(i0, i0 + 4096)
            D = (flatX[sl, None] - A * cs[None, :]) ** 2 + (flatY[sl, None] - B * sn[None, :]) ** 2
            flatt[sl] = ts[np.argmin(D, axis=1)]
        t = flatt.reshape(X.shape)
        for _ in range(n_newton):
            g = A * X * np.sin(t) - B * Y * np.cos(t) - c2 * np.sin(t) * np.cos(t)
            gp = A * X * np.cos(t) + B * Y * np.sin(t) - c2 * np.cos(2 * t)
            step = np.where(np.abs(gp) > 1e-12, g / np.where(np.abs(gp) > 1e-12, gp, 1.0), 0.0)
            t = np.clip(t - step, 0.0, 0.5 * np.pi)
        d = np.sqrt((X - A * np.cos(t)) ** 2 + (Y - B * np.sin(t)) ** 2)
        Xf = c2 * np.cos(t) / A
        al = np.arctan2(np.sin(t) / B, np.cos(t) / A)
        return t, d, Xf, al

    def inside(self, X, Y):
        return (X / self.A) ** 2 + (Y / self.B) ** 2 <= 1.0

    # ---- self-dual knee everywhere ---------------------------------------------
    def selfdual_weights(self, M=4096, n_iter=60):
        """Weights w_j on M wall normals so that v = sum_j w_j exp(n_j.P - h_j)
        equals 1 on the wall (theta_SD = Theta0 there)."""
        t = 2 * np.pi * (np.arange(M) + 0.5) / M
        N = np.sqrt(np.cos(t) ** 2 / self.A ** 2 + np.sin(t) ** 2 / self.B ** 2)
        nx, ny = np.cos(t) / self.A / N, np.sin(t) / self.B / N
        h = 1.0 / N                           # support function
        Bx, By = self.A * np.cos(t), self.B * np.sin(t)
        logK = (Bx[:, None] * nx[None, :] + By[:, None] * ny[None, :]) - h[None, :]
        # Laplace start: w ~ sqrt(rho / 2 pi) times the parametric spacing factor
        logw = np.zeros(M)
        hist = []
        for _ in range(n_iter):
            lv = _lse(logK + logw[None, :], axis=1)
            hist.append(float(np.max(np.abs(lv))))
            logw = logw - lv
        lv = _lse(logK + logw[None, :], axis=1)
        hist.append(float(np.max(np.abs(lv))))
        self._sd = dict(nx=nx, ny=ny, h=h, logw=logw)
        return hist

    def theta_selfdual(self, X, Y, Theta0, branch="both"):
        """branch = "both": the exact self-dual knee-everywhere field.
        branch = "upper": only the upper-wall normals (n_y > 0). In the upper half,
        away from the medial ends, this is the single-sheet outer field: eikonal
        plus the geometric-optics focusing term 0.5 log(rho/(rho - d)). It is the
        far field the zipper sees."""
        sd = self._sd
        keep = np.ones(sd["ny"].size, bool) if branch == "both" else (sd["ny"] > 0)
        nx, ny, h, lw = sd["nx"][keep], sd["ny"][keep], sd["h"][keep], sd["logw"][keep]
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        out = np.empty(X.size)
        fx, fy = X.ravel(), Y.ravel()
        for i0 in range(0, fx.size, 2048):
            sl = slice(i0, i0 + 2048)
            E = fx[sl, None] * nx[None, :] + fy[sl, None] * ny[None, :] - h[None, :] + lw[None, :]
            out[sl] = _lse(E, axis=1)
        return Theta0 + out.reshape(X.shape)


def _lse(E, axis):
    mx = np.max(E, axis=axis, keepdims=True)
    return (mx + np.log(np.sum(np.exp(E - mx), axis=axis, keepdims=True))).squeeze(axis)


# ---------------------------------------------------------------------------
# Zipper table access (bicubic Hermite evaluation of stored zippers)
# ---------------------------------------------------------------------------

def _herm(t, h):
    f = np.array([1 - 3 * t ** 2 + 2 * t ** 3, h * (t - 2 * t ** 2 + t ** 3),
                  3 * t ** 2 - 2 * t ** 3, h * (-t ** 2 + t ** 3)])
    return f


def _scalar(F, key):
    """Read a per-alpha scalar from a zipper-table field file. Files written by
    zipper_table.py v2 store the row under 'rec_<key>' (incremental saving);
    the first v2 table run stored plain '<key>'. Accept both."""
    if key in F.files:
        return float(F[key])
    if ("rec_" + key) in F.files:
        return float(F["rec_" + key])
    raise KeyError("neither '%s' nor 'rec_%s' in zipper field file" % (key, key))


class ZipperLookup(object):
    def __init__(self, table_dir):
        table_dir = Path(table_dir)
        T = dict(np.load(table_dir / "zipper_table.npz", allow_pickle=True))
        S = json.loads((table_dir / "zipper_table_summary.json").read_text())
        self.switch_deg = S["rcn_switch_deg"]
        good = ~T["edge_flag"].astype(bool)
        self.alpha_deg = T["alpha_deg"][good].astype(float)
        self.a = T["a_zip"][good].astype(float)
        self.delta = T["delta_zip"][good].astype(float)
        self.table_dir = table_dir
        self._a_int = PchipInterpolator(self.alpha_deg, self.a, extrapolate=True)
        self._d_int = PchipInterpolator(self.alpha_deg, self.delta, extrapolate=True)
        self.zips = {}
        for adeg in self.alpha_deg:
            F = np.load(table_dir / "fields" / ("alpha%05.2f.npz" % adeg))
            D = F["dofs_zip"]                       # (m, n+1, 4)
            eps, q = _scalar(F, "eps"), _scalar(F, "q")
            ell, L = _scalar(F, "ell"), _scalar(F, "L")
            x, y = F["x"], F["y"]
            m = D.shape[0]
            # append node m (= node 0, value + pi) for periodic evaluation
            Dm = D[:1].copy(); Dm[..., 0] += np.pi
            Dext = np.concatenate([D, Dm], axis=0)
            self.zips[float(adeg)] = dict(D=Dext, eps=eps, q=q, ell=ell, L=L, x=np.append(x, ell), y=y,
                                          delta=_scalar(F, "delta_zip"), m=m,
                                          a=float(self.a[list(self.alpha_deg).index(adeg)]))
        self.amin, self.amax = float(self.alpha_deg.min()), float(self.alpha_deg.max())

    def rcn_switch(self):
        """The knee/zipper switch from the table summary. A table with no sign
        change (e.g. a partial table whose angles are all zipper or all knee)
        has none: loud flag, fall back to the smallest tabulated zipper angle."""
        if self.switch_deg:
            return float(self.switch_deg[0])
        print("!! LOUD FLAG: the zipper table summary has no knee/zipper switch angle "
              "(no sign change among its angles). Using the smallest tabulated zipper angle "
              "%.2f deg. Regenerate the full table to get the real switch." % self.amin)
        return self.amin

    def a_star(self, adeg):
        return self._a_int(np.clip(adeg, self.amin, self.amax))

    def delta_star(self, adeg):
        return self._d_int(np.clip(adeg, self.amin, self.amax))

    def nearest(self, adeg):
        k = float(self.alpha_deg[np.argmin(np.abs(self.alpha_deg - np.clip(adeg, self.amin, self.amax)))])
        return k, self.zips[k]

    @staticmethod
    def R(z, x, y):
        """Zipper correction R = theta - (eps x + q y + delta) at (x, y), via the
        Hermite dofs; x is taken modulo the period; R = 0 above the domain top."""
        ell, L = z["ell"], z["L"]
        x = np.mod(x, ell)
        above = y >= L
        yc = np.clip(y, 0.0, L * (1 - 1e-12))
        xs, ys = z["x"], z["y"]
        i = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, z["m"] - 1)
        j = np.clip(np.searchsorted(ys, yc, side="right") - 1, 0, ys.size - 2)
        hx = xs[i + 1] - xs[i]; hy = ys[j + 1] - ys[j]
        tx = (x - xs[i]) / hx; ty = (yc - ys[j]) / hy
        fx = _herm(tx, hx); fy = _herm(ty, hy)
        D = z["D"]
        th = np.zeros_like(x)
        for cx in (0, 1):
            for cy in (0, 1):
                dof = D[i + cx, j + cy]                     # (..., 4)
                vx, sx = fx[2 * cx], fx[2 * cx + 1]
                vy, sy = fy[2 * cy], fy[2 * cy + 1]
                th += dof[..., 0] * vx * vy + dof[..., 1] * sx * vy + dof[..., 2] * vx * sy + dof[..., 3] * sx * sy
        R = th - (z["eps"] * x + z["q"] * yc + z["delta"])
        return np.where(above, 0.0, R)


# ---------------------------------------------------------------------------
# Stage 1: layout
# ---------------------------------------------------------------------------

STRIPS = dict(n=None, switch=None)


def strip_angle(adeg):
    """Coarse mode: replace the local angle by the midpoint angle of its strip.
    Strips split [switch, 90] deg into STRIPS['n'] equal angle ranges (the
    original thick-strip picture). Fine mode (n = None): identity. The phase
    variable s keeps the continuous delta*, so coarse strips change only the
    zipper shape and a*, never the phase bookkeeping."""
    n = STRIPS["n"]
    if not n:
        return adeg
    lo = STRIPS["switch"]
    edges = np.linspace(lo, 90.0, n + 1)
    k = np.clip(np.digitize(adeg, edges) - 1, 0, n - 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    return np.where(adeg >= lo, mid[k], adeg)


def axis_far_field(E, nX=4001):
    """Single-branch outer phase on the axis, minus Theta0 (needs selfdual_weights)."""
    Xc = np.linspace(0.0, E.Xm, nX)
    return Xc, E.theta_selfdual(Xc, np.zeros_like(Xc), 0.0, branch="upper")


def layout(E, Z, switch_deg, s_center, far=None, nX=200001):
    """Defect positions on the right half of the axis. Returns a dict.
    far = (Xc, phi_c): outer phase on the axis without Theta0; None -> bare eikonal -d."""
    X = np.linspace(0.0, E.Xm, nX)
    d = E.d_axis(X)
    adeg = np.degrees(E.alpha_axis(X))
    dstar = Z.delta_star(adeg)
    astar = Z.a_star(strip_angle(adeg))
    phi = -d if far is None else np.interp(X, far[0], far[1])
    s_raw = phi - dstar                        # s without Theta0
    Theta0 = s_center - s_raw[0]               # s(0) = s_center
    s = Theta0 + s_raw
    zone = adeg >= switch_deg
    Xsw = float(X[zone][-1]) if zone.any() else 0.0
    # crossings: starts at s = n pi, tips at s = n pi + a* pi
    n_of = np.floor(s / np.pi).astype(int)
    frac = s / np.pi - n_of
    starts, tips = [], []
    for i in range(1, nX):
        if X[i] > Xsw:
            break
        if n_of[i] > n_of[i - 1]:
            starts.append(float(X[i]))
        f0, f1 = frac[i - 1] - astar[i - 1], frac[i] - astar[i]
        if n_of[i] == n_of[i - 1] and f0 < 0 <= f1:
            tips.append(float(X[i]))
    # Charge bookkeeping: tips are +1/2, junctions -1/2. The central roll's two
    # tips already carry the +1 the wall requires, so each side must end on a
    # tip (followed by the Neumann/knee stretch): keep only junctions that are
    # followed by a tip inside the zone.
    if tips:
        starts = [x for x in starts if x < tips[-1]]
    else:
        starts = []
    smono = bool(np.all(np.diff(s) > 0))
    return dict(X=X, d=d, adeg=adeg, s=s, astar=astar, dstar=dstar, Theta0=float(Theta0),
                Xsw=Xsw, starts=starts, tips=tips, s_monotone=smono, n0=int(n_of[0]))


# ---------------------------------------------------------------------------
# Stage 2: assembly on the first quadrant, then mirrored
# ---------------------------------------------------------------------------

args_blend_frac = [0.6]


def assemble(E, Z, lay, xq, yq):
    """theta on the first-quadrant grid (xq, yq); NaN outside the ellipse."""
    XX, YY = np.meshgrid(xq, yq, indexing="xy")
    ins = E.inside(XX, YY)
    t, d, Xf, al = E.foot(XX, YY)
    Theta0 = lay["Theta0"]
    th_eik = Theta0 - d
    th_far = E.theta_selfdual(XX, YY, Theta0, branch="upper")   # outer field (eikonal + focusing)
    th_sd = E.theta_selfdual(XX, YY, Theta0)                    # knee everywhere
    R_knee = th_sd - th_far

    # s at the foot, by interpolation on the layout axis grid
    s_f = np.interp(Xf, lay["X"], lay["s"])
    n_f = np.floor(s_f / np.pi).astype(int)
    adeg_f = np.degrees(al)

    tips = lay["tips"]
    R_zip = np.zeros_like(XX)
    w_zip = np.zeros_like(XX)
    info = []
    if tips:
        X_last = tips[-1]
        # each period n uses the zipper nearest the angle at its tip
        n_tip = [int(np.floor(np.interp(xt, lay["X"], lay["s"]) / np.pi)) for xt in tips]
        tip_by_n = dict(zip(n_tip, tips))
        n_last = n_tip[-1]
        # blend width: the Neumann stretch after the last tip
        a_last = float(Z.a_star(np.degrees(E.alpha_axis(X_last))))
        k_last, z_last = Z.nearest(np.degrees(E.alpha_axis(X_last)))
        L_blend = (1.0 - z_last["a"]) * z_last["ell"]
        # Continuum strips: at each point blend the two table zippers that bracket
        # the local angle alpha(X_foot). Each zipper's phase is remapped piecewise
        # linearly so that its own tip (a_j pi) lands at the layout tip a*(alpha) pi
        # and its junctions stay at multiples of pi. The remap fixes 0 and pi, so R
        # is continuous across period boundaries by construction.
        sel = ins & (Xf <= X_last + L_blend)
        al_loc = np.clip(strip_angle(adeg_f[sel]), Z.amin, Z.amax)
        a_loc = Z.a_star(al_loc)
        sig = s_f[sel] - np.pi * n_f[sel]                    # in [0, pi)
        tabs = Z.alpha_deg
        jhi = np.clip(np.searchsorted(tabs, al_loc, side="right"), 1, tabs.size - 1)
        jlo = jhi - 1
        lam = np.clip((al_loc - tabs[jlo]) / (tabs[jhi] - tabs[jlo]), 0.0, 1.0)
        Rsel = np.zeros(sel.sum())
        yabs = np.abs(YY[sel])
        for j, adeg_j in enumerate(tabs):
            wj = np.where(jlo == j, 1.0 - lam, 0.0) + np.where(jhi == j, lam, 0.0)
            use = wj > 0
            if not use.any():
                continue
            z = Z.zips[float(adeg_j)]
            aj = z["a"]
            sj = np.where(sig[use] < a_loc[use] * np.pi,
                          sig[use] * aj / a_loc[use],
                          aj * np.pi + (sig[use] - a_loc[use] * np.pi) * (1 - aj) / (1 - a_loc[use]))
            xz = sj / z["eps"] + yabs[use] / np.tan(np.radians(adeg_j))
            Rsel[use] += wj[use] * Z.R(z, xz, yabs[use])
        R_zip[sel] = Rsel
        info = [("blend of table zippers", list(map(float, tabs)))]
        # weights: 1 up to the last tip, smooth ramp to 0 over the first part of
        # the Neumann stretch (keeps the tip wrap-around, avoids a half-formed
        # concave junction at the end of the stretch)
        L_blend = args_blend_frac[0] * L_blend
        u = np.clip((Xf - X_last) / L_blend, 0.0, 1.0)
        w_zip = np.where(Xf <= X_last, 1.0, 0.5 * (1 + np.cos(np.pi * u)))
        w_zip = np.where(Xf <= X_last + L_blend, w_zip, 0.0)
    R = w_zip * R_zip + (1.0 - w_zip) * R_knee
    th = th_far + R
    for arr in (th, th_far, th_sd, R):
        arr[~ins] = np.nan
    th_eik[~ins] = np.nan
    return dict(theta=th, theta_far=th_far, theta_eik=th_eik, theta_sd=th_sd, R=R, w_zip=w_zip, Xf=Xf, d=d,
                s_f=s_f, inside=ins, periods=info, XX=XX, YY=YY)


def mirror_full(Q):
    """First-quadrant array (ny, nx) with x,y >= 0 -> full ellipse (mirror both axes)."""
    top = np.concatenate([Q[:, :0:-1], Q], axis=1)
    return np.concatenate([top[:0:-1, :], top], axis=0)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def run_checks(E, Z, lay, A_, xq, yq, sd_hist, say):
    say("== Checks ==")
    # 1. axis formula d(X) = B sqrt(1 - X^2/c^2) against brute-force nearest distance
    Xt = np.linspace(0, E.Xm, 400)
    _, dbf, Xfb, alb = E.foot(Xt, np.zeros_like(Xt) + 1e-12)
    say("1. axis distance formula: max |d_formula - d_bruteforce| = %.2e ; "
        "max |alpha_formula - alpha_bruteforce| = %.2e rad"
        % (np.max(np.abs(E.d_axis(Xt) - dbf)), np.max(np.abs(E.alpha_axis(Xt) - alb))))
    dd = np.gradient(E.d_axis(Xt), Xt)
    say("   cos(alpha) = |d'(X)|: max deviation %.2e (interior points)"
        % np.max(np.abs(np.abs(dd[2:-2]) - np.cos(E.alpha_axis(Xt[2:-2])))))
    # 2. self-dual field
    say("2. self-dual weights: max |log v| on the wall after iteration = %.2e (start %.2e)"
        % (sd_hist[-1], sd_hist[0]))
    th = A_["theta_sd"]
    dx = xq[1] - xq[0]
    gy, gx = np.gradient(th, dx, dx)
    lap = np.gradient(gx, dx, axis=1) + np.gradient(gy, dx, axis=0)
    res = lap - (1 - gx ** 2 - gy ** 2)
    inner = A_["inside"] & (A_["d"] > 3 * dx)
    inner[:, :3] = False; inner[:3, :] = False
    say("   self-dual residual |lap theta - (1 - |grad theta|^2)| (finite differences, interior): "
        "median %.2e, 99th pct %.2e, against median |lap theta| %.2e"
        % (np.median(np.abs(res[inner])), np.percentile(np.abs(res[inner]), 99),
           np.median(np.abs(lap[inner]))))
    Xa = xq[(xq > lay["Xsw"] + 5) & (xq < 0.9 * E.Xm)]
    if Xa.size:
        core = np.interp(Xa, xq, A_["theta_sd"][0]) - np.interp(Xa, xq, A_["theta_far"][0])
        say("   knee core offset theta_SD - theta_far(single branch) on the axis (knee zone): "
            "%.5f to %.5f (log 2 = %.5f)" % (core.min(), core.max(), LOG2))
        # focusing term of the outer field against geometric optics 0.5 log(rho/(rho - d))
        Xb = xq[(xq > 2) & (xq < 0.9 * E.Xm)]
        ct = np.clip(E.A * Xb / E.c2, -1, 1); st = np.sqrt(1 - ct ** 2)
        rho = (E.A ** 2 * st ** 2 + E.B ** 2 * ct ** 2) ** 1.5 / (E.A * E.B)
        dX = E.d_axis(Xb)
        pred = 0.5 * np.log(rho / (rho - dX))
        meas = np.interp(Xb, xq, A_["theta_far"][0]) - (lay["Theta0"] - dX)
        say("   outer-field focusing term on the axis: measured %.3f to %.3f, geometric optics "
            "0.5 log(rho/(rho-d)) %.3f to %.3f, max difference %.3f"
            % (meas.min(), meas.max(), pred.min(), pred.max(), np.max(np.abs(meas - pred))))
    # 3. Dirichlet-core mismatch in the zipper zone: theta on the axis should be n pi
    th = A_["theta"][0]
    s0 = np.interp(xq, lay["X"], lay["s"])
    a0 = Z.a_star(np.degrees(E.alpha_axis(xq)))
    frac = s0 / np.pi - np.floor(s0 / np.pi)
    dirich = (xq <= (lay["tips"][-1] if lay["tips"] else 0)) & (frac < a0 - 0.05) & (frac > 0.05)
    if dirich.any():
        mis = np.abs(np.sin(th[dirich]))
        say("3. Dirichlet core: |sin theta| on the axis over Dirichlet segments: median %.3f, max %.3f "
            "(0 means theta = n pi exactly)" % (np.median(mis), mis.max()))
    # 4. seams: jump of theta across period boundaries s = n pi, off the axis
    th2 = A_["theta"]
    s_f = A_["s_f"]
    nf = np.floor(s_f / np.pi)
    jump = np.abs(np.diff(th2, axis=1))
    across = (np.diff(nf, axis=1) != 0) & np.isfinite(jump) & (A_["w_zip"][:, 1:] > 0.99)
    within = (np.diff(nf, axis=1) == 0) & np.isfinite(jump) & (A_["w_zip"][:, 1:] > 0.99)
    if across.any():
        say("4. seams: |theta jump| between neighbouring grid points across period boundaries: "
            "median %.3f, max %.3f ; within periods: median %.3f, 99th pct %.3f"
            % (np.median(jump[across]), jump[across].max(), np.median(jump[within]),
               np.percentile(jump[within], 99)))


# ---------------------------------------------------------------------------
# Energy density (upper half, finite differences)
# ---------------------------------------------------------------------------

def energy_density(th, dx):
    gy, gx = np.gradient(th, dx, dx)
    lap = np.gradient(gx, dx, axis=1) + np.gradient(gy, dx, axis=0)
    return lap ** 2 + (1 - gx ** 2 - gy ** 2) ** 2


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figures(E, Z, lay, A_, xq, yq, out, args, sweep=None, sh=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, BLUE, ORANGE = "#0b0b0b", "#8a8984", "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    xf = np.concatenate([-xq[:0:-1], xq]); yf = np.concatenate([-yq[:0:-1], yq])
    tips, starts = lay["tips"], lay["starts"]

    def marks(ax):
        tx = np.array(tips); sx = np.array(starts)
        if tx.size:
            ax.plot(np.r_[tx, -tx], np.zeros(2 * tx.size), "o", ms=4, mfc="none", mec="k", mew=1.0,
                    label="convex tip (predicted)")
        if sx.size:
            ax.plot(np.r_[sx, -sx], np.zeros(2 * sx.size), "^", ms=4, mfc="none", mec="k", mew=1.0,
                    label="concave junction (predicted)")

    # 1. three patterns
    panels = [("eikonal, cos(Theta0 - d)", A_["theta_eik"]),
              ("self-dual knee everywhere", A_["theta_sd"]),
              ("zipper construction (RCN switch %.2f deg)" % args.switch_used, A_["theta"])]
    if sh is not None:
        panels.append(("SH (last frame, for comparison)", None))
    fig, axs = plt.subplots(len(panels), 1, figsize=(7.2, 4.4 * len(panels)))
    for ax, (ttl, th) in zip(np.atleast_1d(axs), panels):
        if th is None:
            ax.imshow(sh["u"], extent=sh["extent"], origin="lower", cmap="RdBu_r")
            ax.set_xlim(xf[0], xf[-1]); ax.set_ylim(yf[0], yf[-1])
        else:
            U = mirror_full(np.cos(th))
            ax.imshow(U, extent=[xf[0], xf[-1], yf[0], yf[-1]], origin="lower", cmap="RdBu_r",
                      vmin=-1, vmax=1, interpolation="bilinear")
        tt = np.linspace(0, 2 * np.pi, 400)
        ax.plot(E.A * np.cos(tt), E.B * np.sin(tt), color=MUTED, lw=0.6)
        ax.plot([-E.Xm, E.Xm], [0, 0], color=MUTED, lw=0.4, ls=":")
        if "zipper" in ttl:
            marks(ax)
            ax.axvline(lay["Xsw"], color=MUTED, lw=0.6, ls="--"); ax.axvline(-lay["Xsw"], color=MUTED, lw=0.6, ls="--")
            ax.legend(fontsize=7, loc="upper right", frameon=False)
        ax.set_title(ttl, fontsize=9)
        ax.set_aspect("equal")
    fig.suptitle("Ellipse A=%.1f, B=%.1f (A/B=%.2f), s_center=%.2f, Theta0=%.3f"
                 % (E.A, E.B, E.A / E.B, args.s_center_used, lay["Theta0"]), fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig1_patterns.png", dpi=170)
    plt.close(fig)

    # 2. zoom on the zipper zone
    fig, ax = plt.subplots(figsize=(8, 4))
    th = A_["theta"]
    U = mirror_full(np.cos(th))
    ax.imshow(U, extent=[xf[0], xf[-1], yf[0], yf[-1]], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
              interpolation="bilinear")
    marks(ax)
    lim = min(E.A, lay["Xsw"] + 15)
    ax.set_xlim(-lim, lim); ax.set_ylim(-0.45 * lim, 0.45 * lim)
    ax.set_aspect("equal")
    ax.set_title("Zipper zone, zoom (dashed: RCN switch position)", fontsize=9)
    ax.axvline(lay["Xsw"], color=MUTED, lw=0.6, ls="--"); ax.axvline(-lay["Xsw"], color=MUTED, lw=0.6, ls="--")
    fig.tight_layout()
    fig.savefig(out / "fig2_zipper_zone.png", dpi=200)
    plt.close(fig)

    # 3. energy density, upper half, log scale
    dx = xq[1] - xq[0]
    e = energy_density(A_["theta"], dx)
    e_sd = energy_density(A_["theta_sd"], dx)
    fig, axs = plt.subplots(2, 1, figsize=(8, 6.4))
    for ax, ee, ttl in [(axs[0], e_sd, "self-dual knee everywhere"), (axs[1], e, "zipper construction")]:
        ee = ee.copy(); ee[~A_["inside"]] = np.nan
        Ef = np.concatenate([ee[:, :0:-1], ee], axis=1)
        im = ax.imshow(np.log10(Ef + 1e-6), extent=[xf[0], xf[-1], yq[0], yq[-1]], origin="lower",
                       cmap="magma", vmin=-4, vmax=0.5)
        ax.set_aspect("equal")
        ax.set_title("log10 RCN energy density, upper half: %s" % ttl, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "fig3_energy_density.png", dpi=170)
    plt.close(fig)

    # 4. axis layout
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    X = lay["X"]
    axs[0].plot(X, lay["adeg"], color=INK, lw=1.2)
    axs[0].axhline(args.switch_used, color=MUTED, ls="--", lw=1)
    axs[0].set_ylabel("alpha(X) (deg)")
    axs[0].set_title("Right half of the medial axis", fontsize=9)
    frac = lay["s"] / np.pi - np.floor(lay["s"] / np.pi)
    axs[1].plot(X, frac, color=BLUE, lw=1.0, label="s / pi mod 1")
    axs[1].plot(X, lay["astar"], color=ORANGE, lw=1.2, label="a*(alpha) (tip where they meet)")
    for xt in tips:
        axs[1].axvline(xt, color=INK, lw=0.6)
    for xs_ in starts:
        axs[1].axvline(xs_, color=INK, lw=0.6, ls=":")
    axs[1].axvline(lay["Xsw"], color=MUTED, lw=1, ls="--")
    axs[1].set_xlabel("X along the axis")
    axs[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig4_axis_layout.png", dpi=170)
    plt.close(fig)

    # 5. sweep over s_center
    if sweep is not None:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for sc, tp, st in zip(sweep["s_center"], sweep["tips"], sweep["starts"]):
            ax.plot([sc] * len(tp), tp, "o", ms=3, mfc="none", mec=INK)
            ax.plot([sc] * len(st), st, "^", ms=3, mfc="none", mec=MUTED)
        ax.axhline(lay["Xsw"], color=MUTED, ls="--", lw=1)
        ax.set_xlabel("s_center (free wall phase, central-roll setting)")
        ax.set_ylabel("X of predicted defects (right half)")
        ax.set_title("Predicted positions against the free wall phase\n"
                     "o convex tips, ^ concave junctions; dashed: RCN switch", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "fig5_wall_phase_sweep.png", dpi=170)
        plt.close(fig)


# ---------------------------------------------------------------------------

def main(args):
    out = Path(args.out_dir) if args.out_dir else SCRIPT_DIR / "results" / SCRIPT_NAME
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(s):
        print(s, flush=True)
        lines.append(s)

    table_dir = Path(args.table_dir) if args.table_dir else SCRIPT_DIR / "results" / "zipper_table"
    Z = ZipperLookup(table_dir)
    switch = args.switch_deg if args.switch_deg is not None else Z.rcn_switch()
    if switch < Z.amin - 1e-9:
        say("!! LOUD FLAG: switch angle %.2f deg is below the smallest tabulated zipper (%.1f deg). "
            "RCN has no free zipper there; this needs constrained zippers (Stage 0b). Using %.2f instead."
            % (switch, Z.amin, Z.amin))
        switch = Z.amin
    args.switch_used = switch
    STRIPS["n"] = args.n_strips if args.n_strips else None
    STRIPS["switch"] = switch
    A, B = args.beta * args.Lx, args.beta * args.Ly
    E = Ellipse(A, B)
    a_c = float(Z.a_star(90.0))
    s_center = args.s_center_frac * a_c * np.pi
    args.s_center_used = s_center
    say("ellipse A=%.3f B=%.3f c^2/A=%.3f ; switch %.2f deg ; table angles %s ; a*(center, clamped)=%.3f"
        % (A, B, E.Xm, switch, list(Z.alpha_deg), a_c))

    sd_hist = E.selfdual_weights(M=args.M, n_iter=args.n_iter_sd)
    far = axis_far_field(E)
    lay = layout(E, Z, switch, s_center, far=far)
    say("layout: Theta0=%.4f  X_switch=%.3f  tips=%s  junctions=%s  s monotone=%s"
        % (lay["Theta0"], lay["Xsw"], ["%.2f" % v for v in lay["tips"]],
           ["%.2f" % v for v in lay["starts"]], lay["s_monotone"]))
    if not lay["s_monotone"]:
        say("!! LOUD FLAG: s(X) not monotone; period mapping ambiguous")

    sweep = dict(s_center=[], tips=[], starts=[])
    for f in np.linspace(0.02, 0.98, args.n_sweep):
        L2 = layout(E, Z, switch, f * a_c * np.pi, far=far, nX=40001)
        sweep["s_center"].append(f * a_c * np.pi); sweep["tips"].append(L2["tips"]); sweep["starts"].append(L2["starts"])
    ntips = [len(t) for t in sweep["tips"]]
    say("wall-phase sweep: convex tips per side %d to %d, i.e. dislocations (junction-tip pairs) per side "
        "%d to %d, over s_center in (0, a* pi)" % (min(ntips), max(ntips), min(ntips) - 1, max(ntips) - 1))

    t0 = time.time()
    xq = np.arange(0.0, A * 1.01 + args.dx, args.dx)
    yq = np.arange(0.0, B * 1.01 + args.dx, args.dx)
    A_ = assemble(E, Z, lay, xq, yq)
    say("assembly on %d x %d first-quadrant grid (dx=%.3f) in %.0fs ; periods used (n, tip X, table alpha): %s"
        % (yq.size, xq.size, args.dx, time.time() - t0, A_["periods"]))
    run_checks(E, Z, lay, A_, xq, yq, sd_hist, say)

    # total RCN energy (first quadrant x 4, half weights on the axes), against the
    # free wall phase; the knee-everywhere field does not depend on it
    def total_energy(th):
        e = energy_density(th, args.dx)
        wgt = np.ones_like(e); wgt[0, :] *= 0.5; wgt[:, 0] *= 0.5
        ok = A_["inside"] & np.isfinite(e)
        return 4.0 * args.dx ** 2 * float(np.sum((e * wgt)[ok]))
    E_sd = total_energy(A_["theta_sd"])
    ens = dict(frac=[], E=[], ntips=[])
    if args.n_energy_sweep > 0:
        for f in np.linspace(0.05, 0.95, args.n_energy_sweep):
            L3 = layout(E, Z, switch, f * a_c * np.pi, far=far)
            A3 = assemble(E, Z, L3, xq, yq)
            ens["frac"].append(float(f)); ens["E"].append(total_energy(A3["theta"])); ens["ntips"].append(len(L3["tips"]))
        say("energy: knee everywhere %.2f ; zipper construction %.2f (this s_center) ; over the wall-phase sweep "
            "%.2f to %.2f, minimum at s_center/(a* pi) = %.2f with %d tip(s) per side"
            % (E_sd, total_energy(A_["theta"]), min(ens["E"]), max(ens["E"]),
               ens["frac"][int(np.argmin(ens["E"]))], ens["ntips"][int(np.argmin(ens["E"]))]))
        for f, e_, nt in zip(ens["frac"], ens["E"], ens["ntips"]):
            say("   s_center/(a* pi)=%.3f  E=%.3f  tips/side=%d" % (f, e_, nt))

    sh = None
    if args.sh_npz:
        try:
            S = np.load(args.sh_npz)
            u = S["uu"]
            u = u[..., -1] if u.ndim == 3 else u
            xx, yy = S["xx"], S["yy"]
            sh = dict(u=u, extent=[xx.min(), xx.max(), yy.min(), yy.max()])
        except Exception as ex:   # loud flag, not a hard block
            say("!! LOUD FLAG: could not read SH file %s (%s); skipping comparison" % (args.sh_npz, ex))

    figures(E, Z, lay, A_, xq, yq, out, args, sweep=sweep, sh=sh)
    if ens["E"]:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.axhline(E_sd, color="#8a8984", lw=1, ls="--", label="knee everywhere (self-dual)")
        ax.plot(ens["frac"], ens["E"], "-o", color="#eb6834", ms=4, lw=1.5, label="zipper construction")
        for f, e_, nt in zip(ens["frac"], ens["E"], ens["ntips"]):
            ax.annotate(str(nt - 1), (f, e_), fontsize=7, xytext=(0, 5), textcoords="offset points", ha="center")
        ax.set_xlabel("s_center / (a* pi)   (free wall phase)")
        ax.set_ylabel("total RCN energy (finite differences)")
        ax.set_title("Total energy against the wall phase; labels: dislocations per side", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(out / "fig6_energy_vs_wall_phase.png", dpi=170); plt.close(fig)
    np.savez(out / "ellipse_construction.npz", x=xq, y=yq, theta=A_["theta"], theta_far=A_["theta_far"],
             theta_sd=A_["theta_sd"], R=A_["R"], w_zip=A_["w_zip"], inside=A_["inside"],
             Theta0=lay["Theta0"], tips=np.array(lay["tips"]), starts=np.array(lay["starts"]),
             X_switch=lay["Xsw"], A=A, B=B, note="first quadrant; mirror both axes for the full ellipse")
    (out / "assembly_log.txt").write_text("\n".join(lines) + "\n")
    (out / "args.json").write_text(json.dumps({k: getattr(args, k) for k in dir(_Args) if not k.startswith("_")},
                                              indent=2, default=str))
    print("wrote", out)


class _Args(object):
    Lx = 45 * np.pi             # runner box (ellipse semi-axes = beta * Lx, beta * Ly)
    Ly = 30 * np.pi
    beta = 0.45
    table_dir = None            # default: results/zipper_table next to this script
    switch_deg = None           # None -> RCN switch from the table summary
    s_center_frac = 0.5         # s(0) as a fraction of a*(center) pi: sets the free wall phase
    n_strips = 0                # 0 = fine (continuum blend); N > 0 = coarse, N strips by normal angle
    n_sweep = 41                # wall-phase sweep resolution (layout only)
    n_energy_sweep = 19         # wall-phase sweep with full assembly and total energy (0 = off)
    dx = 0.25                   # assembly grid spacing
    M = 4096                    # wall normals in the self-dual superposition
    n_iter_sd = 60
    sh_npz = None               # optional SH npz (keys xx, yy, uu) for a side-by-side panel
    out_dir = None


def _parse_cli():
    d = _Args()
    p = argparse.ArgumentParser(description="Ellipse from RCN zippers: layout and assembly")
    for k in ["Lx", "Ly", "beta", "s_center_frac", "dx"]:
        p.add_argument("--" + k, type=float, default=getattr(d, k))
    p.add_argument("--switch_deg", type=float, default=d.switch_deg)
    for k in ["n_sweep", "n_energy_sweep", "M", "n_iter_sd", "n_strips"]:
        p.add_argument("--" + k, type=int, default=getattr(d, k))
    p.add_argument("--table_dir", default=d.table_dir)
    p.add_argument("--sh_npz", default=d.sh_npz)
    p.add_argument("--out_dir", default=d.out_dir)
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_cli() if len(sys.argv) > 1 else _Args())
