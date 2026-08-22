# pipelines/verify_director_structure_tensor.py
"""
Verification: the smoothed Riesz director equals the leading eigendirection
of the smoothed structure tensor of the Riesz vector, and the alignment
equals its coherence, exactly.

Claim (for the paper): with T = G_sigma * [(R1,R2)(R1,R2)^T], i.e.
    T11 = <R1^2>,  T12 = <R1 R2>,  T22 = <R2^2>,
the pipeline's doubled-angle averages satisfy the identities
    C = <w cos 2 phi> = T11 - T22,     S = <w sin 2 phi> = 2 T12,
    ang = (1/2) atan2(S, C)           = leading eigendirection of T,
    alignment = sqrt(C^2+S^2)/<w>     = (lam1 - lam2)/(lam1 + lam2),
where phi = atan2(R2, R1), w = R1^2 + R2^2, lam1 >= lam2 >= 0.
The pointwise identities w cos 2 phi = R1^2 - R2^2 and
w sin 2 phi = 2 R1 R2 are algebraic; the smoothing is linear, so they
survive averaging. The eigendirection statement holds because the two
eigenvectors of a symmetric 2x2 are orthogonal, hence antipodal in the
doubled angle, so the deviatoric part of T is (lam1 - lam2) times the
doubled-angle unit vector of e1: eigenvalue weights scale the length of
(C, S), never its direction, and the direction is e1 exactly (undefined,
with alignment -> 0, when lam1 = lam2).

Three routes are compared on the analytic knee and the PN chain:
  A. pipeline route: riesz_director (C, S averages)
  B. tensor-formula route: ang = 0.5 atan2(2 T12, T11 - T22),
     coh = sqrt((T11-T22)^2 + 4 T12^2) / (T11 + T22)
  C. brute-force route: per-pixel numpy.linalg.eigh of T on a subsample
Angles are compared mod pi. Expected: agreement at floating point level.

Usage: python verify_director_structure_tensor.py [--sigma 1.5708]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

from utils.field_diagnostics_viz import (monogenic, macro_iso,  # noqa: E402
                                         riesz_director, knee_field,
                                         pn_chain_field)


def angdiff_mod_pi(a, b):
    """Absolute director-angle difference, compared mod pi."""
    d = a - b
    return np.abs((d + np.pi / 2) % np.pi - np.pi / 2)


def verify_on(u, x, y, sigma, label, n_eig=4000, seed=0):
    Lx, Ly = x[-1] - x[0], y[-1] - y[0]

    # A: pipeline route
    rd = riesz_director(u, Lx, Ly, sigma)

    # B: structure tensor formulas
    mono = monogenic(u, Lx, Ly)
    R1, R2 = mono["R1"], mono["R2"]
    T11 = macro_iso(R1 * R1, sigma, Lx, Ly)
    T12 = macro_iso(R1 * R2, sigma, Lx, Ly)
    T22 = macro_iso(R2 * R2, sigma, Lx, Ly)
    ang_T = 0.5 * np.arctan2(2 * T12, T11 - T22)
    tr = T11 + T22
    coh_T = np.sqrt((T11 - T22) ** 2 + 4 * T12 ** 2) / np.maximum(tr, 1e-12)

    err_ang = angdiff_mod_pi(rd["angle"], ang_T)
    err_coh = np.abs(rd["alignment"] - np.clip(coh_T, 0, 1))
    print(f"[{label}] A vs B  max angle err (rad): {err_ang.max():.3e}   "
          f"max alignment err: {err_coh.max():.3e}")

    # C: brute-force eigh on a random subsample
    rng = np.random.default_rng(seed)
    ny, nx = u.shape
    ii = rng.integers(0, ny, n_eig)
    jj = rng.integers(0, nx, n_eig)
    T = np.empty((n_eig, 2, 2))
    T[:, 0, 0] = T11[ii, jj]
    T[:, 0, 1] = T[:, 1, 0] = T12[ii, jj]
    T[:, 1, 1] = T22[ii, jj]
    evals, evecs = np.linalg.eigh(T)      # ascending
    lam1, lam2 = evals[:, 1], evals[:, 0]
    e1 = evecs[:, :, 1]
    ang_eig = np.arctan2(e1[:, 1], e1[:, 0])
    coh_eig = (lam1 - lam2) / np.maximum(lam1 + lam2, 1e-12)

    err_ang_c = angdiff_mod_pi(rd["angle"][ii, jj], ang_eig)
    err_coh_c = np.abs(rd["alignment"][ii, jj] - np.clip(coh_eig, 0, 1))
    print(f"[{label}] A vs C  max angle err (rad): {err_ang_c.max():.3e}   "
          f"max alignment err: {err_coh_c.max():.3e}   "
          f"(n={n_eig} sample pixels)")
    return {"err_ang_B": float(err_ang.max()),
            "err_coh_B": float(err_coh.max()),
            "err_ang_C": float(err_ang_c.max()),
            "err_coh_C": float(err_coh_c.max())}


def main(sigma):
    print(f"sigma = {sigma}")
    u, x, y = knee_field(0.9)
    verify_on(u, x, y, sigma, "knee mu=0.9")
    u, x, y = pn_chain_field(0.9)
    verify_on(u, x, y, sigma, "PN chain mu=0.9")


class _Args:
    sigma = np.pi / 2


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sigma", type=float, default=_Args.sigma)
    main(p.parse_args().sigma)
