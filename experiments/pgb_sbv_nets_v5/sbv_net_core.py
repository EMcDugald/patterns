# experiments/pgb_sbv_nets/sbv_net_core.py
"""
Core components for phase estimation by macro-energy matching with the
(relaxed, SBV) Cross-Newell energy. Conventions match sbv_phase_probe.py:
fields are (Ny, Nx) with x the last axis, Lx = x[-1]-x[0], and the macro
filter is macro(., sigma, Lx, Ly) i.e. kernel exp(-0.5 sigma^2 |k|^2) with
k = (2 pi / L) * fftfreq-integers.

Configuration axes
------------------
repr_mode   : "field" (SIREN neural field, autodiff gradients)
              "grid"  (theta values on the grid, spectral gradients)
energy_mode : "bulk"  (e = c_bend q^2 + c_well (|grad theta|^2-1)^2 + c0,
                       q = G_s * Lap theta; no singular measure)
              "sbv"   (split q into rho_s + mu_s by weighted shrinkage;
                       singular term |sin theta||mu_s| filtered at sigma_f)
init_mode   : "data"  (theta = theta0 + correction; theta0 from probe)
              "none"  (theta = correction alone; expected-hard baseline)

Never differentiates twice: the tested measure
    q(x) = (G_s * Div D theta)(x) = -int grad G_s(x-y).grad theta(y) dy
is computed spectrally from the sampled gradient of theta, so the singular
part of the distributional Laplacian is captured without differencing
grad theta. The a.c./singular split is the SBV codes' shrinkage step:
    mu_s = shrink(q, kappa (delta + |sin theta|)),   rho_s = q - mu_s,
so the network learns ONLY theta; rho and mu are closed-form functionals.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field as dc_field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SBVNetConfig:
    repr_mode: str = "field"      # "field" | "grid"
    energy_mode: str = "sbv"      # "sbv" | "bulk"
    init_mode: str = "data"       # "data" | "none"

    # scales
    macro_sigma: float = np.pi / 2   # sigma_f of the TARGET filter (probe)
    s_test: float = -1.0             # test width; <=0 -> 2.5 * min(dx, dy)

    # derivative discretization for: base grad(theta0), grad(ramp),
    # grid-mode correction gradients, and the divergence inside the tested
    # measure. "spectral" = FFT ik symbols (global, periodic; Gibbs-rings
    # at the theta0 kink). "fd" = 2nd-order central differences, one-sided
    # at box edges (local, non-periodic; kink errors stay local -- closer
    # to the paper's per-element FEM gradients). The sigma_f macro filter
    # on the singular term is ALWAYS spectral, to match the probe's target
    # convention exactly. Field-mode correction gradients are autodiff in
    # both modes.
    deriv: str = "spectral"          # "spectral" | "fd"

    # SBV split
    kappa_init: float = 0.1
    fix_kappa: bool = False   # freeze kappa (recommended: learnable kappa +
                              # delta_gauge is degenerate — the optimizer can
                              # empty mu_s by raising the threshold)
    delta: float = 0.05
    delta_gauge: float = 1e-3

    # calibration inits
    c0_init: float = 0.0
    c_bend_init: float = 1.0
    c_well_init: float = 1.0
    c_sing_init: float = 1.0
    # comma-separated subset of {c0, c_bend, c_well, c_sing} (or "all")
    # to FREEZE at their *_init values instead of learning them. Rationale:
    # the coefficients are the affine map from the unit-weight CN/SBV model
    # densities to SH energy units; roll-averaging predicts c0 ~ -R^2/6 and
    # c_well ~ R/3, so freezing those tests the theory and breaks the
    # well/bend/sing degeneracy at the grain boundary (all three channels
    # produce a GB-localized ridge, so single-frame energy matching cannot
    # apportion the ridge between them on its own).
    fix_calib: str = ""

    # regularizers on the correction
    lam_small: float = 0.0
    lam_smooth: float = 1e-3

    # SIREN
    width: int = 128
    depth: int = 4
    omega: float = 30.0

    # optimization
    iters: int = 3000
    lr: float = 1e-4
    lr_calib: float = 1e-2
    chunk: int = 65536
    seed: int = 0

    # data handling
    coarsen: int = 1
    mask_erode: int = 3
    # optional x-window for the LOSS MASK, as fractions of the full x-range
    # (0 = left edge, 1 = right edge). E.g. (0.55, 0.95) = middle 80% of the
    # right half. The grid/FFT domain is NOT cropped (spectral ops need the
    # full periodic box); theta is global, supervision is windowed.
    x_window: tuple = ()
    taper_correction: bool = True   # multiply correction by ramp_n so all
                                    # learned fields decay smoothly to zero
                                    # before the periodic box edges (FFT-safe)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def run_name(self, stem: str, frame: int) -> str:
        xw = (f"_xw{self.x_window[0]:.2f}-{self.x_window[1]:.2f}"
              if self.x_window else "")
        fd = "_fd" if self.deriv == "fd" else ""
        return (f"{stem}__{self.repr_mode}_{self.energy_mode}_"
                f"init-{self.init_mode}_f{frame:03d}_it{self.iters}"
                f"{xw}{fd}")


# ---------------------------------------------------------------------------
# Spectral operators (probe conventions)
# ---------------------------------------------------------------------------

def wavenumbers(Ny, Nx, Lx, Ly):
    kx = (2.0 * np.pi / Lx) * np.fft.fftfreq(Nx, d=1.0 / Nx)
    ky = (2.0 * np.pi / Ly) * np.fft.fftfreq(Ny, d=1.0 / Ny)
    return np.meshgrid(kx, ky)  # (Ny, Nx) each, 'xy' indexing


class SpectralOps(nn.Module):
    """Holds all Fourier symbols: macro filter (matches probe macro()),
    tested-measure kernels i k G_s_hat, and plain derivative symbols.
    deriv="fd" swaps the derivative parts (grad, and the divergence inside
    the tested measure) for 2nd-order central differences with one-sided
    stencils at the box edges; the Gaussian mollifications (G_f, G_s)
    remain spectral convolutions in both modes."""

    def __init__(self, Ny, Nx, Lx, Ly, sigma_f, s_test, device,
                 dtype=torch.float32, deriv="spectral"):
        super().__init__()
        if deriv not in ("spectral", "fd"):
            raise ValueError(f"deriv must be 'spectral' or 'fd', got "
                             f"{deriv!r}")
        self.deriv = deriv
        # true grid spacing under the probe convention Lx = x[-1]-x[0]
        self.dx = Lx / (Nx - 1)
        self.dy = Ly / (Ny - 1)

        KX, KY = wavenumbers(Ny, Nx, Lx, Ly)
        K2 = KX**2 + KY**2
        cplx = torch.complex64 if dtype == torch.float32 else torch.complex128

        Gf = np.exp(-0.5 * sigma_f**2 * K2)
        Gs = np.exp(-0.5 * s_test**2 * K2)

        self.register_buffer("Gf_hat", torch.as_tensor(
            Gf, device=device, dtype=dtype))
        self.register_buffer("Gs_hat", torch.as_tensor(
            Gs, device=device, dtype=dtype))
        self.register_buffer("ikGs_x", torch.as_tensor(
            1j * KX * Gs, device=device, dtype=cplx))
        self.register_buffer("ikGs_y", torch.as_tensor(
            1j * KY * Gs, device=device, dtype=cplx))
        self.register_buffer("ik_x", torch.as_tensor(
            1j * KX, device=device, dtype=cplx))
        self.register_buffer("ik_y", torch.as_tensor(
            1j * KY, device=device, dtype=cplx))

    def macro(self, f):
        return torch.real(torch.fft.ifft2(self.Gf_hat * torch.fft.fft2(f)))

    def smooth_s(self, f):
        """Gaussian mollification at the test width s (spectral)."""
        return torch.real(torch.fft.ifft2(self.Gs_hat * torch.fft.fft2(f)))

    # -- finite differences: 2nd-order central, one-sided at edges -------
    @staticmethod
    def _fd_axis(f, h, dim):
        n = f.shape[dim]
        if n < 3:
            raise ValueError("grid too small for FD stencil")
        sl = [slice(None)] * f.ndim

        def take(a, b):
            sl2 = list(sl)
            sl2[dim] = slice(a, b)
            return f[tuple(sl2)]

        interior = (take(2, n) - take(0, n - 2)) / (2.0 * h)
        left = (take(1, 2) - take(0, 1)) / h
        right = (take(n - 1, n) - take(n - 2, n - 1)) / h
        return torch.cat([left, interior, right], dim=dim)

    def fd_dx(self, f):
        return self._fd_axis(f, self.dx, dim=-1)

    def fd_dy(self, f):
        return self._fd_axis(f, self.dy, dim=-2)

    def grad_fd(self, f):
        return self.fd_dx(f), self.fd_dy(f)

    def grad_spectral(self, f):
        fh = torch.fft.fft2(f)
        gx = torch.real(torch.fft.ifft2(self.ik_x * fh))
        gy = torch.real(torch.fft.ifft2(self.ik_y * fh))
        return gx, gy

    def grad(self, f):
        return self.grad_fd(f) if self.deriv == "fd" \
            else self.grad_spectral(f)

    def tested_measure(self, gx, gy):
        """q = G_s * Div(D theta) from sampled grad theta. Spectral mode:
        FFT(grad).(ik G_s_hat), the jump never being differenced. FD mode:
        FD divergence of the sampled gradient (a ~jump/h spike at the jump
        row, mass-preserving) immediately mollified by G_s -- the regular-
        grid analog of the FEM stiffness-matrix tested divergence."""
        if self.deriv == "fd":
            div = self.fd_dx(gx) + self.fd_dy(gy)
            return self.smooth_s(div)
        return torch.real(torch.fft.ifft2(
            torch.fft.fft2(gx) * self.ikGs_x
            + torch.fft.fft2(gy) * self.ikGs_y))


def shrink(q, tau):
    return torch.sign(q) * torch.clamp(torch.abs(q) - tau, min=0.0)


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------

class SineLayer(nn.Module):
    def __init__(self, cin, cout, omega, first=False):
        super().__init__()
        self.omega = omega
        self.lin = nn.Linear(cin, cout)
        with torch.no_grad():
            if first:
                self.lin.weight.uniform_(-1.0 / cin, 1.0 / cin)
            else:
                b = math.sqrt(6.0 / cin) / omega
                self.lin.weight.uniform_(-b, b)

    def forward(self, x):
        return torch.sin(self.omega * self.lin(x))


class FieldCorrection(nn.Module):
    """SIREN correction theta_s(x); coords normalized internally."""

    def __init__(self, Lx, Ly, width, depth, omega):
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor([2.0 / Lx, 2.0 / Ly]), requires_grad=False)
        layers = [SineLayer(2, width, omega, first=True)]
        for _ in range(depth - 1):
            layers.append(SineLayer(width, width, omega))
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)
        with torch.no_grad():
            b = math.sqrt(6.0 / width) / omega
            self.head.weight.uniform_(-b, b)
            self.head.bias.zero_()

    def theta_and_grad(self, x_phys, chunk):
        """x_phys: (N,2). Returns values (N,) and grads (N,2), autodiff."""
        vals, grads = [], []
        for i0 in range(0, x_phys.shape[0], chunk):
            x = x_phys[i0:i0 + chunk].clone().requires_grad_(True)
            v = self.head(self.body(x * self.scale)).squeeze(-1)
            (g,) = torch.autograd.grad(v.sum(), x, create_graph=True)
            vals.append(v)
            grads.append(g)
        return torch.cat(vals), torch.cat(grads)


class GridCorrection(nn.Module):
    """Correction as a free (Ny, Nx) tensor; gradients taken spectrally."""

    def __init__(self, Ny, Nx):
        super().__init__()
        self.dtheta = nn.Parameter(torch.zeros(Ny, Nx))


# ---------------------------------------------------------------------------
# Calibration parameters
# ---------------------------------------------------------------------------

def _inv_softplus(v):
    return float(np.log(np.expm1(max(v, 1e-6))))


class Calibration(nn.Module):
    _FIXABLE = ("c0", "c_bend", "c_well", "c_sing")

    def __init__(self, cfg: SBVNetConfig):
        super().__init__()
        self.c0 = nn.Parameter(torch.tensor(cfg.c0_init))
        self._cb = nn.Parameter(torch.tensor(_inv_softplus(cfg.c_bend_init)))
        self._cw = nn.Parameter(torch.tensor(_inv_softplus(cfg.c_well_init)))
        self._cs = nn.Parameter(torch.tensor(_inv_softplus(cfg.c_sing_init)))
        self._kp = nn.Parameter(torch.tensor(_inv_softplus(cfg.kappa_init)))
        if getattr(cfg, "fix_kappa", False):
            self._kp.requires_grad_(False)

        fixed = getattr(cfg, "fix_calib", "") or ""
        names = [t.strip() for t in fixed.split(",") if t.strip()]
        if names == ["all"]:
            names = list(self._FIXABLE)
        bad = [n for n in names if n not in self._FIXABLE]
        if bad:
            raise ValueError(f"fix_calib: unknown coefficient(s) {bad}; "
                             f"choose from {self._FIXABLE} or 'all'.")
        param_of = {"c0": self.c0, "c_bend": self._cb,
                    "c_well": self._cw, "c_sing": self._cs}
        for n in names:
            param_of[n].requires_grad_(False)
        self.fixed = tuple(names)

    c_bend = property(lambda s: F.softplus(s._cb))
    c_well = property(lambda s: F.softplus(s._cw))
    c_sing = property(lambda s: F.softplus(s._cs))
    kappa = property(lambda s: F.softplus(s._kp))

    def summary(self):
        return {"c0": float(self.c0), "c_bend": float(self.c_bend),
                "c_well": float(self.c_well), "c_sing": float(self.c_sing),
                "kappa": float(self.kappa)}


# ---------------------------------------------------------------------------
# Model: assembles theta and the energy terms for either representation
# ---------------------------------------------------------------------------

class SBVModel(nn.Module):
    def __init__(self, cfg: SBVNetConfig, Ny, Nx, Lx, Ly,
                 theta0=None, grad_theta0=None, ramp=None, grad_ramp=None):
        """
        theta0, grad_theta0: (Ny, Nx) and ((Ny,Nx),(Ny,Nx)) numpy arrays or
        None (init_mode == "none"). grad_theta0 is precomputed spectrally so
        the base gradient is exact and shared by both representations.
        ramp, grad_ramp: normalized ramp and its precomputed gradient, used
        (if cfg.taper_correction) to smoothly confine the learned correction
        to the diamond interior: dtheta_eff = ramp * dtheta. The BASE phase
        is never tapered (theta0 ~ O(50): theta0 * grad ramp would inject a
        huge spurious wavevector at the ramp shoulder); base seam artifacts
        are handled instead by the domain margin plus the eroded mask.
        """
        super().__init__()
        self.cfg = cfg
        self.Ny, self.Nx = Ny, Nx
        dev = cfg.device

        if theta0 is not None:
            self.register_buffer("theta0", torch.as_tensor(
                theta0, dtype=torch.float32, device=dev))
            self.register_buffer("g0x", torch.as_tensor(
                grad_theta0[0], dtype=torch.float32, device=dev))
            self.register_buffer("g0y", torch.as_tensor(
                grad_theta0[1], dtype=torch.float32, device=dev))
        else:
            self.theta0 = None

        if cfg.taper_correction and ramp is not None:
            self.register_buffer("ramp", torch.as_tensor(
                ramp, dtype=torch.float32, device=dev))
            self.register_buffer("rgx", torch.as_tensor(
                grad_ramp[0], dtype=torch.float32, device=dev))
            self.register_buffer("rgy", torch.as_tensor(
                grad_ramp[1], dtype=torch.float32, device=dev))
        else:
            self.ramp = None

        if cfg.repr_mode == "field":
            self.corr = FieldCorrection(Lx, Ly, cfg.width, cfg.depth,
                                        cfg.omega)
        elif cfg.repr_mode == "grid":
            self.corr = GridCorrection(Ny, Nx)
        else:
            raise ValueError(cfg.repr_mode)

        self.calib = Calibration(cfg)

    def forward(self, x_grid, ops: SpectralOps):
        """Returns theta, (gx, gy), (dtheta, dgx, dgy) all (Ny, Nx)."""
        Ny, Nx = self.Ny, self.Nx
        if self.cfg.repr_mode == "field":
            v, g = self.corr.theta_and_grad(x_grid, self.cfg.chunk)
            raw = v.view(Ny, Nx)
            rgx = g[:, 0].view(Ny, Nx)
            rgy = g[:, 1].view(Ny, Nx)
            if self.ramp is not None:
                # product rule with the precomputed ramp gradient (exact
                # pointwise; no FFT of a non-decaying field is ever taken)
                dtheta = self.ramp * raw
                dgx = self.ramp * rgx + raw * self.rgx
                dgy = self.ramp * rgy + raw * self.rgy
            else:
                dtheta, dgx, dgy = raw, rgx, rgy
        else:
            raw = self.corr.dtheta
            dtheta = self.ramp * raw if self.ramp is not None else raw
            # dtheta now decays smoothly to zero before the box edges, so
            # its spectral gradient is free of wrap-seam ringing; in fd
            # mode the stencil is local and non-periodic anyway
            dgx, dgy = ops.grad(dtheta)

        if self.theta0 is not None:
            theta = self.theta0 + dtheta
            gx = self.g0x + dgx
            gy = self.g0y + dgy
        else:
            theta, gx, gy = dtheta, dgx, dgy
        return theta, (gx, gy), (dtheta, dgx, dgy)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_losses(model: SBVModel, ops: SpectralOps, x_grid,
                   e_macro, mask, weights=None):
    cfg = model.cfg
    cal = model.calib
    theta, (gx, gy), (dtheta, dgx, dgy) = model(x_grid, ops)

    q = ops.tested_measure(gx, gy)
    well = (gx**2 + gy**2 - 1.0) ** 2
    abs_sin = torch.abs(torch.sin(theta))

    if cfg.energy_mode == "sbv":
        tau = cal.kappa * (cfg.delta + abs_sin)
        mu_s = shrink(q, tau)
        rho_s = q - mu_s
        e_model = (cal.c_bend * rho_s**2 + cal.c_well * well
                   + cal.c_sing * ops.macro(abs_sin * torch.abs(mu_s))
                   + cal.c0)
    elif cfg.energy_mode == "bulk":
        mu_s = torch.zeros_like(q)
        rho_s = q
        e_model = cal.c_bend * q**2 + cal.c_well * well + cal.c0
    else:
        raise ValueError(cfg.energy_mode)

    w = mask if weights is None else mask * weights
    denom = torch.clamp(w.sum(), min=1.0)
    L_energy = torch.sum(w * (e_model - e_macro) ** 2) / denom

    loss = L_energy
    parts = {"energy": float(L_energy)}

    if cfg.energy_mode == "sbv" and cfg.delta_gauge > 0:
        L_gauge = cfg.delta_gauge * torch.sum(mask * torch.abs(mu_s)) \
            / denom
        loss = loss + L_gauge
        parts["gauge"] = float(L_gauge)

    if cfg.lam_small > 0:
        L_small = cfg.lam_small * torch.sum(mask * dtheta**2) / denom
        loss = loss + L_small
        parts["small"] = float(L_small)

    if cfg.lam_smooth > 0:
        L_smooth = cfg.lam_smooth * torch.sum(
            mask * (dgx**2 + dgy**2)) / denom
        loss = loss + L_smooth
        parts["smooth"] = float(L_smooth)

    parts["total"] = float(loss)
    diag = {"theta": theta, "dtheta": dtheta, "q": q, "rho_s": rho_s,
            "mu_s": mu_s, "well": well, "e_model": e_model,
            "abs_sin": abs_sin}
    return loss, parts, diag


# ---------------------------------------------------------------------------
# Data loading (probe_fields.npz)
# ---------------------------------------------------------------------------

def load_probe(probe_dir, frame, coarsen=1, mask_erode=0,
               macro_sigma_override=None, x_window=None):
    """
    probe_dir: .../results/sbv_phase_probe/<stem>/  (contains data/)
    Returns dict of numpy arrays for one frame plus geometry and sigma_f.
    x_window: optional (lo, hi) fractions of the full x-range; the valid
    mask is intersected with lo <= (x - x0)/(x1 - x0) <= hi. The grid is
    not cropped, so all spectral operators are unchanged.
    """
    probe_dir = Path(probe_dir)
    data = np.load(probe_dir / "data" / "probe_fields.npz")
    with open(probe_dir / "data" / "probe_summary.json") as f:
        summary = json.load(f)

    sigma_f = macro_sigma_override
    if sigma_f is None:
        sigma_f = summary["selected_fields"].get("macro_sigma", None)
    if sigma_f is None:
        raise ValueError("macro_sigma not found; pass --macro_sigma.")

    x = np.asarray(data["x"]).ravel()
    y = np.asarray(data["y"]).ravel()
    T = data["u"].shape[2]
    if frame < 0:
        frame = T + frame

    def fr(key):
        arr = np.asarray(data[key])
        return arr[..., frame] if arr.ndim == 3 else arr

    out = {
        "u": fr("u"),
        "e_macro": fr("macro_energy"),
        "e_micro": fr("micro_energy"),
        "mask": fr("valid_mask").astype(bool),
        "ramp_n": fr("ramp_n").astype(np.float64),
        "theta0": np.asarray(data["theta0"]),
        "frame": frame, "T": T,
    }

    if mask_erode > 0:
        from scipy.ndimage import binary_erosion
        out["mask"] = binary_erosion(out["mask"], iterations=mask_erode)

    c = max(int(coarsen), 1)
    if c > 1:
        for k in ("u", "e_macro", "e_micro", "theta0", "ramp_n"):
            out[k] = out[k][::c, ::c]
        out["mask"] = out["mask"][::c, ::c]
        x = x[::c]
        y = y[::c]

    if x_window:
        lo, hi = float(x_window[0]), float(x_window[1])
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError(f"x_window must satisfy 0 <= lo < hi <= 1, "
                             f"got {x_window}")
        # xf = (x - x[0]) / (x[-1] - x[0])
        # out["mask"] &= ((xf >= lo) & (xf <= hi))[None, :]
        cols = np.where(out["mask"].any(axis=0))[0]  # mask's x-extent
        xm0, xm1 = x[cols[0]], x[cols[-1]]
        xlo = xm0 + lo * (xm1 - xm0)
        xhi = xm0 + hi * (xm1 - xm0)
        out["mask"] &= ((x >= xlo) & (x <= xhi))[None, :]
        if not out["mask"].any():
            raise ValueError("x_window left the mask empty.")


    out["x"], out["y"] = x, y
    out["Lx"] = float(x[-1] - x[0])   # probe convention
    out["Ly"] = float(y[-1] - y[0])
    out["sigma_f"] = float(sigma_f)
    out["stem"] = summary.get("stem", probe_dir.name)
    return out


def spectral_grad_np(f, Lx, Ly):
    Ny, Nx = f.shape
    KX, KY = wavenumbers(Ny, Nx, Lx, Ly)
    fh = np.fft.fft2(f)
    gx = np.real(np.fft.ifft2(1j * KX * fh))
    gy = np.real(np.fft.ifft2(1j * KY * fh))
    return gx, gy


def fd_grad_np(f, x, y):
    """2nd-order central differences (one-sided at edges) on the actual
    coordinate arrays; errors from the theta0 kink stay local instead of
    Gibbs-ringing globally as with the spectral gradient."""
    gy, gx = np.gradient(f, y, x)
    return gx, gy


def make_x_grid(x, y, device):
    X, Y = np.meshgrid(x, y)
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
    return torch.as_tensor(pts, device=device)


def save_config(cfg: SBVNetConfig, path):
    with open(path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)
