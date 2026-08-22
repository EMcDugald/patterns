# src/op_extract/pn_field.py
"""
Standalone Pismen-Nepomnyashchy dislocation-row field, lifted verbatim from
pismen_fit_and_residuals_v2.pismen_on_sh_grid so the nucleation-measure bench
can import it without pulling in that runner's heavier dependencies.

theta(X, Y) = Y - sign(Y) * log(psi),  psi built from erf terms at each
x_defect, giving a periodic row of edge dislocations along y = 0 in an
otherwise straight (k = (0, 1)) stripe pattern.
"""

import numpy as np
from scipy import special


def pismen_on_sh_grid(X, Y, x_defects, amp=1.0):
    linear_phase = Y
    x_defects = np.asarray(x_defects, dtype=float)
    npts = len(x_defects)
    if npts == 0:
        theta = linear_phase.copy()
        return theta, amp * np.cos(theta)

    thetas = np.arange(npts + 1) * np.pi
    psi = 0.5 * (1 + np.exp(-thetas[-1]))
    for j in range(npts):
        delta = 0.5 * (np.exp(-thetas[j + 1]) - np.exp(-thetas[j]))
        arg = (X - x_defects[j]) / np.sqrt(2 * np.abs(Y) + 1e-8)
        psi += delta * special.erf(arg)
    theta = linear_phase - np.sign(Y) * np.log(psi)
    return theta, amp * np.cos(theta)
