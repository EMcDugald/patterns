"""
beta_law_figure.py

Summary figure for the measured self-dual coefficient: beta against
mu^2 at R = 0.5 and R = 1.0, with intercept-pinned linear fits
beta = 1 + c(R) mu^2, visualizing the empirical law and the
extrapolation to beta = 1 in the shallow-angle limit
(Writeup 1, Section "How closely do SH knees match RCN knees").

v1.1 changelog (measured justification):
  - Default output path anchored to the script's directory rather
    than the working directory (same fix as the figure combiners).

v1.0 changelog:
  - New standalone figure script; frozen method layer untouched.
  - Numbers are inherited, read-only, with provenance: the demod
    column of the data block of extraction_comparison.py on the
    T = 500 zigzag lower-boundary frames (console runs of record,
    R = 0.5 and R = 1.0 sweeps), plus the two older-pipeline frames
    (t = 100, R = 0.5) recorded in the session 6 handoff.
  - The slope c(R) is a least-squares fit of beta - 1 against mu^2
    through the pinned intercept beta(0) = 1 (arithmetic on the
    inherited numbers, no new extraction).

Output: pipelines/results/beta_law_figure/beta_law.png
"""

import argparse
import os
import sys

import numpy as np

try:
    from paper_style import use_paper_style
except ImportError:                       # paper_style lives in src/utils
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                    "..", "src", "utils"))
    try:
        from paper_style import use_paper_style
    except ImportError:
        def use_paper_style():
            pass

import matplotlib.pyplot as plt


# ---- inherited numbers (read-only; provenance in the docstring) ----
MU = np.array([0.325, 0.375, 0.425, 0.475])
BETA_R05 = np.array([1.0938, 1.1227, 1.1693, 1.2193])   # demod, R = 0.5
BETA_R10 = np.array([1.0560, 1.0713, 1.1034, 1.1415])   # demod, R = 1.0
MU_OLD = np.array([0.40, 0.50])                          # older pipeline
BETA_OLD = np.array([1.136, 1.239])                      # t = 100, R = 0.5


def pinned_slope(mu, beta):
    """Least squares for beta - 1 = c mu^2 through the pinned
    intercept beta(0) = 1."""
    x = mu**2
    return float(np.sum((beta - 1.0) * x) / np.sum(x * x))


def main(args):
    use_paper_style()
    c05 = pinned_slope(MU, BETA_R05)
    c10 = pinned_slope(MU, BETA_R10)
    print("fitted c(0.5) = %.3f   c(1.0) = %.3f" % (c05, c10))

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    xx = np.linspace(0.0, 0.27, 200)
    ax.plot(xx, 1.0 + c05 * xx, "-", color="tab:red", lw=1.2,
            label=r"$1 + c\,\mu^2$,  $c(0.5) = %.2f$" % c05)
    ax.plot(xx, 1.0 + c10 * xx, "-", color="tab:blue", lw=1.2,
            label=r"$1 + c\,\mu^2$,  $c(1.0) = %.2f$" % c10)
    ax.plot(MU**2, BETA_R05, "o", color="tab:red", ms=5,
            label=r"measured, $R = 0.5$")
    ax.plot(MU**2, BETA_R10, "s", color="tab:blue", ms=5,
            label=r"measured, $R = 1.0$")
    ax.plot(MU_OLD**2, BETA_OLD, "o", mfc="none", mec="tab:red", ms=6,
            label=r"older pipeline frames, $R = 0.5$")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.annotate(r"$\beta = 1$  (RCN, $\alpha = \eta$)",
                xy=(0.005, 1.010), fontsize=9)
    ax.set_xlabel(r"$\mu^2$")
    ax.set_ylabel(r"measured $\beta$")
    ax.set_xlim(0, 0.27)
    ax.set_ylim(0.98, 1.27)
    ax.legend(loc="upper left")

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "beta_law.png")
    fig.savefig(out)
    print("saved", out)
    return 0


_HERE_DIR = os.path.dirname(os.path.abspath(__file__))


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below.
    Output path is anchored to this script's directory."""
    outdir = os.path.join("results", "beta_law_figure")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--outdir", default=_Args.outdir)
    sys.exit(main(p.parse_args()))
