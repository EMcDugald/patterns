# pipelines/uhu_tensor_report.py
"""
Write out the FULL uHu tensor before and after averaging, for the knee and
PN ansatze, together with its eigensystem and the ground-truth phase
gradients, in three forms: sympy expressions, LaTeX, and numbers on a
transect.

Four averaging models (the rungs of handoff Section 1's two channels):

    full        u d_i d_j u pointwise, nothing removed
    corrugated  DC + g * osc, g = exp(-2 (sx^2 thx^2 + sy^2 thy^2))
                channel (1) only, the frozen-phase model
    kernel      second-order kernel expansion of DC at (sx, sy)
                channel (2) only
    DC          ideal averaging, both channels removed; for A = 1 this is
                exactly -(1/2) k k^T

For each model the report gives T_xx, T_xy, T_yy, then lam_dom, lam_sub,
chi = lam_sub/lam_dom, the dominant eigenvector angle, the implied |k|
estimate sqrt(-2 lam_dom), and the angle error against the true wavevector
angle atan2(theta_y, theta_x) wrapped mod pi.

Ground truth block: theta, k = (theta_x, theta_y), |k|, arg k, the Hessian,
div k, J = det(grad k), W, and (with an amplitude) W_A.

Outputs written to --outdir:
    uhu_tensor_<tag>.txt    readable report, symbolic + numeric
    uhu_tensor_<tag>.tex    LaTeX fragment of the same expressions
    uhu_tensor_<tag>.csv    the numeric transect, one row per point

Examples:
    python uhu_tensor_report.py --ansatz knee --mu 0.9
    python uhu_tensor_report.py --ansatz pn --y0 1.0
    python uhu_tensor_report.py --ansatz pn --amplitude tanh --xi 1
    python uhu_tensor_report.py --ansatz knee --symbolic none   # numbers only
"""

import argparse
import os

import numpy as np
import sympy as sp

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.uhu_symbolic import (KneeAnsatz, PNAnsatz, eig2x2, wrap_mod_pi,
                          lambdify_np, is_zero)

SOURCES = ("full", "corrugated", "kernel", "DC")


# =====================================================================
# setup
# =====================================================================

def build(ansatz, amplitude="one", xi=1.0):
    """Return (instance, free_params, tag). free_params is the tuple of
    extra symbols that must be supplied when lambdifying."""
    if ansatz == "knee":
        a = KneeAnsatz()
        return a, (a.k1, a.k2), "knee"
    if ansatz == "pn":
        a = PNAnsatz(numeric_weights=True, amplitude=amplitude,
                     xi=(None if amplitude == "one" else xi))
        if amplitude == "tanh" and a.xi.free_symbols:
            raise ValueError("pass a numeric --xi with --amplitude tanh")
        return a, (), f"pn_{amplitude}"
    raise ValueError(ansatz)


def truth_block(a):
    """Ground-truth quantities, as sympy expressions."""
    out = {
        "theta": a.theta,
        "k1 = theta_x": a.thx,
        "k2 = theta_y": a.thy,
        "|k|": sp.sqrt(a.k_mag2),
        "arg k": sp.atan2(a.thy, a.thx),
        "theta_xx": a.hess[0, 0],
        "theta_xy": a.hess[0, 1],
        "theta_yy": a.hess[1, 1],
        "div k": a.div_k,
        "J = det(grad k)": a.J_raw,
        "W": a.W,
        "A": a.A,
    }
    if a.A != 1:
        out["W_A"] = a.amplitude_contamination()
    return out


def tensor_block(a, source, sx, sy):
    """The tensor and its eigensystem for one averaging model."""
    T = a.uhu_tensor(source, sx, sy)
    eig = eig2x2(T[0, 0], T[0, 1], T[1, 1], dominant="min")
    err = wrap_mod_pi(eig["angle_dom"] - a.truth_angle())
    return {
        "T_xx": T[0, 0], "T_xy": T[0, 1], "T_yy": T[1, 1],
        "lam_dom": eig["lam_dom"], "lam_sub": eig["lam_sub"],
        "chi": eig["chi"], "angle_dom": eig["angle_dom"],
        "k_mag_est": sp.sqrt(-2 * eig["lam_dom"]),
        "angle_err": err,
    }, T, eig


# =====================================================================
# numeric transect
# =====================================================================

def transect_points(ansatz, n, x0, y0, span, along):
    t = np.linspace(-span, span, n) if along == "y" else \
        np.linspace(-span, span, n)
    if along == "y":
        return np.full(n, x0), (t if ansatz == "knee" else np.abs(t) + y0)
    return t + x0, np.full(n, y0)


def evaluate(a, params, blocks, xs, ys, sx, sy, sxv, syv, pvals):
    """Lambdify everything once and evaluate on the transect."""
    args = (a.x, a.y) + tuple(params) + (sx, sy)
    vals = {}
    for name, expr in blocks.items():
        try:
            f = lambdify_np(args, expr)
            vals[name] = np.asarray(
                f(xs, ys, *pvals, sxv, syv), dtype=float) * np.ones_like(xs)
        except Exception as exc:                # keep going, report the gap
            vals[name] = np.full_like(xs, np.nan)
            vals[name + " [ERR]"] = str(exc)[:60]
    return vals


# =====================================================================
# report
# =====================================================================

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    a, params, tag = build(args.ansatz, args.amplitude, args.xi)
    sx, sy = sp.symbols("s_x s_y", positive=True)

    pvals = (np.sqrt(1 - args.mu**2), args.mu) if args.ansatz == "knee" else ()
    lines, tex = [], []

    def w(s=""):
        print(s, flush=True)
        lines.append(s)

    w("=" * 72)
    w(f"uHu TENSOR REPORT: ansatz={args.ansatz} amplitude={args.amplitude}")
    if args.ansatz == "knee":
        w(f"  mu = {args.mu}  ->  k1 = {pvals[0]:.6f}, k2 = {pvals[1]:.6f}")
    w(f"  sigma = ({args.sx}, {args.sy})")
    w("  convention: uHu is negative semidefinite, so the DOMINANT eigenpair")
    w("  is the MOST NEGATIVE one, matching compute_uhu_ops_smoother")
    w("=" * 72)

    # ---- ground truth ------------------------------------------------
    truth = truth_block(a)
    w("")
    w("-" * 72)
    w("GROUND TRUTH (exact, from the ansatz)")
    w("-" * 72)
    if args.symbolic != "none":
        for k, v in truth.items():
            w(f"\n{k}:\n  {sp.simplify(v) if args.simplify else v}")
            tex.append(f"% {k}\n\\[ {sp.latex(v)} \\]")

    # ---- tensors -----------------------------------------------------
    blocks = {}
    for src in SOURCES:
        blk, T, eig = tensor_block(a, src, sx, sy)
        w("")
        w("-" * 72)
        w(f"AVERAGING MODEL: {src}")
        w("-" * 72)
        want_sym = (args.symbolic == "all"
                    or (args.symbolic == "dc" and src == "DC"))
        for k, v in blk.items():
            blocks[f"{src}:{k}"] = v
            if want_sym:
                w(f"\n{k}:\n  {sp.simplify(v) if args.simplify else v}")
                tex.append(f"% {src} {k}\n\\[ {sp.latex(v)} \\]")
        if not want_sym:
            w("  (symbolic printing suppressed; numbers in the transect below)")

    # ---- exactness check for A = 1 ------------------------------------
    if a.A == 1:
        chk = is_zero(a.uhu_tensor("DC")[0, 0] + a.thx**2 / 2)
        w("")
        w(f"[check] DC tensor equals -(1/2) k k^T for A = 1: {chk}")

    # ---- numeric transect ---------------------------------------------
    xs, ys = transect_points(args.ansatz, args.n, args.x0, args.y0,
                             args.span, args.along)
    allb = dict(truth)
    allb.update(blocks)
    vals = evaluate(a, params, allb, xs, ys, sx, sy, args.sx, args.sy, pvals)

    w("")
    w("-" * 72)
    w(f"NUMERIC TRANSECT ({args.n} points along {args.along})")
    w("-" * 72)
    cols = ["x", "y", "k1 = theta_x", "k2 = theta_y", "|k|", "W"] + \
           [f"{s}:angle_err" for s in SOURCES] + \
           [f"{s}:chi" for s in SOURCES]
    cols = [c for c in cols if c in vals or c in ("x", "y")]
    hdr = "  " + "".join(f"{c[:11]:>13}" for c in cols)
    w(hdr)
    step = max(1, args.n // 12)
    for i in range(0, args.n, step):
        row = []
        for c in cols:
            v = xs[i] if c == "x" else ys[i] if c == "y" else vals[c][i]
            if "angle_err" in c:
                v = np.degrees(v)
            row.append(f"{v:>13.4g}")
        w("  " + "".join(row))
    w("")
    w("  angle_err columns are in DEGREES, wrapped mod pi")

    # ---- write files ---------------------------------------------------
    base = os.path.join(args.outdir, f"uhu_tensor_{tag}")
    with open(base + ".txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(base + ".tex", "w") as fh:
        fh.write("\n\n".join(tex) + "\n")
    keys = [k for k in vals if not k.endswith("[ERR]")]
    arr = np.column_stack([xs, ys] + [vals[k] for k in keys])
    np.savetxt(base + ".csv", arr, delimiter=",",
               header=",".join(["x", "y"] + keys), comments="")
    print(f"\nwrote {base}.txt / .tex / .csv")


class _Args:
    ansatz = "pn"
    amplitude = "one"
    xi = 1.0
    mu = 0.9
    sx = np.pi / 2
    sy = np.pi / 2
    n = 121
    x0 = 0.0
    y0 = 1.0
    span = 6.0
    along = "y"
    symbolic = "all"          # all | dc | none
    simplify = False
    outdir = "out_uhu_report"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ansatz", default=_Args.ansatz, choices=["knee", "pn"])
    p.add_argument("--amplitude", default=_Args.amplitude,
                   choices=["one", "tanh"])
    p.add_argument("--xi", type=float, default=_Args.xi)
    p.add_argument("--mu", type=float, default=_Args.mu)
    p.add_argument("--sx", type=float, default=_Args.sx)
    p.add_argument("--sy", type=float, default=_Args.sy)
    p.add_argument("--n", type=int, default=_Args.n)
    p.add_argument("--x0", type=float, default=_Args.x0)
    p.add_argument("--y0", type=float, default=_Args.y0)
    p.add_argument("--span", type=float, default=_Args.span)
    p.add_argument("--along", default=_Args.along, choices=["x", "y"])
    p.add_argument("--symbolic", default=_Args.symbolic,
                   choices=["all", "dc", "none"])
    p.add_argument("--simplify", action="store_true")
    p.add_argument("--outdir", default=_Args.outdir)
    main(p.parse_args())
