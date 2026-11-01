#!/usr/bin/env python
"""
energy_fit_review.py  v1.2 (31 Oct 2026)   the three laws on the line-energy curve, fitted the right way

v1.2 (for the chapter): the old least-squares K line is off by default (--show_old 1 restores it); the straight-line
extrapolations in panel (b) are drawn only when they are the source of K_inf (--show_extrap 1 restores them); the
figure is two panels by default (--layout abcd gives the four-panel version with the measured E_n and the spacing);
the two panels are also written on their own as fig_energy_curve and fig_K_of_alpha, and with --tail_csv a
fig_tail_energy (the tail's line energy with the same laws) replaces the old tail figure. The tail's settled and
unsettled points are now distinguished in panel (b). No number changes.

v1.1 (same evening), after the advisor's note on the energy slide: (1) the protuberance law C (1 - sin alpha) is drawn
solid up to 43.0 deg and dotted from there to the dislocation onset (--onset_mu, default 0.82, the first angle at which
the survey's circulation reads one turn, 55.1 deg), since beyond 43 deg the configuration carries a disclination end
whose cost E_n the 2003 construction ignores; (2) that cost is now measured as the residual E_n(alpha) = E_SH -
C (1 - sin alpha) over the charged angles up to the onset, tabulated and drawn in a new panel (c); (3) a panel (d)
with the dislocation spacing k_B L = pi / cos alpha (k_B = 1 here) against the angle. No fitted number of v1.0 changes.

Reads the energy_vs_angle.csv written by energy_vs_angle_zigzag_v2.py (no SH runs here; nothing in that script
or its caches is touched) and redraws the line energy against the far-field angle with the three laws placed as
the theory, not the least-squares routine, dictates:

  knee law        E = C sin^3(alpha)      C fitted on the knee branch only: accepted points with no director
                                          charge and mu <= sin3_mu_max (0.70).  (Ercolani, Indik, Newell, Passot
                                          2003, eq. 4.10: (8/3) sin^3 in RCN units; in SH the constant is measured.)
  protuberance    E = C (1 - sin alpha)   the SAME C, not a second fit: the 2003 paper's competing configuration
                                          (Sec. 7, eq. 7.1) is the knee formula applied to two tilted boundaries,
                                          2 / cos(pi/4 + alpha/2) * sin^3(pi/4 - alpha/2) = 1 - sin alpha, so the
                                          prefactor is the knee's and the crossing sin^3 = 1 - sin at
                                          sin alpha = 0.6823 (43.0 deg) is parameter free.  v2 fitted C1 on the
                                          charged branch instead (C1 = 1.40 against C = 0.35 on the 29 Oct sweep),
                                          which put the crossing at 57.8 deg and made the line meaningless.
  dislocation law E = K_inf cos(alpha)    K_inf is the LIMIT of K(alpha) = E / cos(alpha) as mu -> 1 (one
                                          dislocation per spacing L = pi / cos alpha, so K is the energy per
                                          dislocation over pi, and its limit is the isolated dislocation's), read
                                          by a straight line in (1 - mu) over the accepted charged points with
                                          mu >= K_mu_min (0.85) and reported for two other windows.  v2 fitted a
                                          single K by least squares over the same points, which averages a
                                          quantity that rises by a factor of two across the window; that line is
                                          kept dotted for comparison.

Also reported: the angle of the first director charge, the crossing of the knee law with the dislocation law,
the peak of the measured curve, and the log-log slope near mu -> 1 (v2's exponent test), all from the CSV.
Optional: --tail_csv overlays a second sweep (the tail sweep 0.95 to 0.999) in the K(alpha) panel, and K_inf is then
read as the mean of E / cos alpha over its settled charged points (the plateau), which is the right reading when a
plateau exists: the straight-line extrapolation from the main sweep's window 0.85 to 0.99 lies on the rising,
concave part of K(alpha) and overshoots (0.495 against a plateau near 0.45 on the 4 Oct sweeps); --E_d <value> draws K_d cos(alpha) with K_d = E_d / pi for a separately measured isolated
dislocation energy, as an independent check of K_inf.

Outputs (in --outdir): fig_energy_fits.png/.pdf (two panels: the curve with the three laws; K(alpha) against
1 - mu with the extrapolation), fit_report.txt, fits.json.  Python 3.8 compatible.  Run from ~/patterns/pipelines.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
VERSION = "energy_fit_review v1.2 (31 Oct 2026)"
ALPHA_43 = float(np.degrees(np.arcsin(0.6823278)))     # root of s^3 + s - 1 = 0


def read_sweep(path, args):
    rows = list(csv.DictReader(open(path)))
    g = lambda k, d=np.nan: np.array([float(r.get(k, d)) if r.get(k, "") not in ("", None) else d for r in rows])  # noqa: E731
    mu = g("mu"); E = g("E_GB"); nd = g("n_dislocations", 0.0); Ep = g("E_GB_prev", np.nan)
    settled = np.where(np.isfinite(Ep), np.abs(E - Ep) <= args.settle_tol * np.maximum(np.abs(E), 1e-12), True)
    if "accepted" in rows[0]:
        acc = g("accepted") > 0.5
    else:                                                  # v1.0 CSVs: recompute the acceptance of v1.1
        ppw = g("ppw_y", np.inf); flat = g("flatness", 0.0); er = g("e_roll")
        base = -args.Rscale ** 2 / 6.0
        acc = (ppw > args.accept_ppw_floor) & (flat <= args.accept_flat_tol) & (np.abs(er - base) <= args.accept_base_tol * abs(base))
    o = np.argsort(mu)
    return dict(mu=mu[o], E=E[o], nd=nd[o], acc=acc[o], settled=settled[o], alpha=np.arcsin(mu[o]), path=str(path))


def _amp(basis, E):
    d = float(np.sum(basis * basis))
    return float(np.sum(basis * E) / d) if d > 0 else float("nan")


def fits(S, args):
    mu, E, nd, acc, a = S["mu"], S["E"], S["nd"], S["acc"], S["alpha"]
    s, c = np.sin(a), np.cos(a)
    out = {}
    knee = acc & (nd < 1) & (mu <= args.sin3_mu_max)
    out["C"] = _amp(s[knee] ** 3, E[knee]); out["n_knee"] = int(knee.sum())
    out["C_resid_rms_rel"] = float(np.sqrt(np.mean((E[knee] - out["C"] * s[knee] ** 3) ** 2)) / np.mean(E[knee])) if knee.any() else float("nan")
    ch = acc & (nd >= 1)
    out["first_charge_mu"] = float(mu[nd >= 1].min()) if (nd >= 1).any() else float("nan")
    out["first_charge_deg"] = float(np.degrees(np.arcsin(out["first_charge_mu"]))) if np.isfinite(out["first_charge_mu"]) else float("nan")
    hi = ch & (mu >= args.K_mu_min)
    out["K_fit_old"] = _amp(c[hi], E[hi]); out["C1_fit_old"] = _amp((1 - s)[hi], E[hi])
    K = E / np.maximum(c, 1e-12)
    for name, lo in (("K_inf", args.K_mu_min), ("K_inf_b", args.K_mu_min_b), ("K_inf_c", args.K_mu_min_c)):
        m = ch & (mu >= lo)
        if m.sum() >= 3:
            sl, ic = np.polyfit(1 - mu[m], K[m], 1)
            out[name] = float(ic); out[name + "_slope"] = float(sl); out[name + "_n"] = int(m.sum()); out[name + "_mu_min"] = float(lo)
        else:
            out[name] = float("nan"); out[name + "_slope"] = float("nan"); out[name + "_n"] = int(m.sum()); out[name + "_mu_min"] = float(lo)
    top = ch & (mu >= args.K_top_mu)
    out["K_top_mean"] = float(np.mean(K[top])) if top.any() else float("nan"); out["K_top_n"] = int(top.sum())
    st = ch & S["settled"]
    out["K_plateau_mean"] = float(np.mean(K[st])) if st.any() else float("nan"); out["K_plateau_n"] = int(st.sum())
    out["K_plateau_std"] = float(np.std(K[st])) if st.any() else float("nan")
    out["K_all_mean"] = float(np.mean(K[ch])) if ch.any() else float("nan"); out["K_all_n"] = int(ch.sum())
    out["K_last_points"] = [[float(m_), float(k_)] for m_, k_ in zip(mu[ch][-4:], K[ch][-4:])]
    # log-log slope of E against 1 - mu near mu -> 1 (v2's exponent test)
    ll = ch & (mu >= args.loglog_mu_min) & (E > 0) & (mu < 1)
    out["loglog_slope"] = float(np.polyfit(np.log(1 - mu[ll]), np.log(E[ll]), 1)[0]) if ll.sum() >= 2 else float("nan")
    ip = int(np.argmax(E)); out["peak_mu"] = float(mu[ip]); out["peak_deg"] = float(np.degrees(a[ip])); out["peak_E"] = float(E[ip])
    # crossings on a fine grid of sin alpha
    ss = np.linspace(0.3, 0.9999, 200001); cc = np.sqrt(1 - ss ** 2)
    def cross(f):
        v = f(ss); i = int(np.argmin(np.abs(v)))
        return float(np.degrees(np.arcsin(ss[i])))
    out["cross_43_deg"] = ALPHA_43
    out["cross_sin3_Kinf_deg"] = cross(lambda t: out["C"] * t ** 3 - out["K_inf"] * np.sqrt(1 - t ** 2)) if np.isfinite(out["K_inf"]) else float("nan")
    out["cross_sin3_Kold_deg"] = cross(lambda t: out["C"] * t ** 3 - out["K_fit_old"] * np.sqrt(1 - t ** 2))
    out["cross_sin3_C1old_deg"] = cross(lambda t: out["C"] * t ** 3 - out["C1_fit_old"] * (1 - t))
    # where the data leave the knee law: first accepted angle at which E is below C sin^3 by more than leave_tol
    dev = (out["C"] * s ** 3 - E) / np.maximum(out["C"] * s ** 3, 1e-12)
    m = acc & (dev > args.leave_tol)
    out["leave_knee_mu"] = float(mu[m].min()) if m.any() else float("nan")
    out["leave_knee_deg"] = float(np.degrees(np.arcsin(out["leave_knee_mu"]))) if np.isfinite(out["leave_knee_mu"]) else float("nan")
    # the one-constant protuberance line against the data at a few angles
    out["C_oms_vs_data"] = [[float(d), float(out["C"] * (1 - np.sin(np.radians(d)))),
                             float(np.interp(np.radians(d), a[acc], E[acc]))] for d in (50, 60, 70, 80) if np.radians(d) <= a[acc].max()]
    if args.E_d > 0:
        out["K_d"] = float(args.E_d / np.pi)
    # v1.1: the residual above the one-constant protuberance law, E_n(alpha) = E - C (1 - sin alpha), charged angles up to the onset
    En = E - out["C"] * (1 - s)
    sel = acc & (mu >= min(0.6823278, out["first_charge_mu"] if np.isfinite(out["first_charge_mu"]) else 1.0)) & (mu <= args.onset_mu + 1e-9)
    out["E_n_table"] = [[float(np.degrees(a[i])), float(mu[i]), float(E[i]), float(out["C"] * (1 - s[i])), float(En[i])] for i in np.where(sel)[0]]
    if sel.sum() >= 3:
        sl_, ic_ = np.polyfit(np.degrees(a[sel]), En[sel], 1)
        out["E_n_slope_per_deg"] = float(sl_); out["E_n_zero_deg"] = float(-ic_ / sl_) if sl_ != 0 else float("nan")
    else:
        out["E_n_slope_per_deg"] = float("nan"); out["E_n_zero_deg"] = float("nan")
    out["onset_mu"] = float(args.onset_mu); out["onset_deg"] = float(np.degrees(np.arcsin(args.onset_mu)))
    i_on = int(np.argmin(np.abs(mu - args.onset_mu)))
    out["E_at_onset"] = float(E[i_on]); out["E_n_at_onset"] = float(En[i_on]); out["K_needed_at_onset"] = float(E[i_on] / c[i_on])
    return out


def FT_plateau(T, args):
    c = np.cos(T["alpha"]); st = T["acc"] & (T["nd"] >= 1) & T["settled"]
    return float(np.mean((T["E"] / np.maximum(c, 1e-12))[st])) if st.any() else float("nan")


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(alpha=0.25)


def draw_a(ax, S, F, args, legend_loc=(0.0, 0.80)):
    """panel (a): the line energy against the angle with the three laws"""
    mu, E, nd, acc, a = S["mu"], S["E"], S["nd"], S["acc"], S["alpha"]
    ad = np.degrees(a)
    ag = np.radians(np.linspace(max(20, ad.min() - 3), min(86, ad.max() + 3), 600)); sg, cg, agd = np.sin(ag), np.cos(ag), np.degrees(ag)
    k0 = acc & (nd < 1); k1 = acc & (nd >= 1); rj = ~acc
    ax.plot(ad[k0], E[k0], "o", color="0.2", ms=4.5, label="relaxed boundary, no director charge")
    ax.plot(ad[k1], E[k1], "o", mfc="w", mec="crimson", mew=1.3, ms=5.5, label="relaxed boundary, director charge (width %g)" % args.det_width)
    if rj.any():
        ax.plot(ad[rj], E[rj], "o", mfc="none", mec="0.6", ms=5, label="excluded (resolution, baseline or flatness)")
    ax.plot(agd, F["C"] * sg ** 3, "-", color="#1f4fd6", lw=1.6,
            label=r"knee law $C\sin^3\alpha$, $C$ = %.3f from the knee branch ($\mu\leq%.2f$)" % (F["C"], args.sin3_mu_max))
    on_deg = float(np.degrees(np.arcsin(args.onset_mu)))
    seg1 = agd <= ALPHA_43; seg2 = (agd >= ALPHA_43) & (agd <= on_deg)
    ax.plot(agd[seg1], F["C"] * (1 - sg[seg1]), "-", color="#2a8f3c", lw=1.6, label=r"$C\,(1-\sin\alpha)$, the same $C$ (crossing at %.1f$^\circ$)" % ALPHA_43)
    if args.dotted_segment:
        ax.plot(agd[seg2], F["C"] * (1 - sg[seg2]), ":", color="#2a8f3c", lw=1.8, label=r"$C\,(1-\sin\alpha)+E_n$ with $E_n$ unknown, to the onset at %.1f$^\circ$" % on_deg)
    if np.isfinite(F["K_used"]):
        src = "tail plateau" if "tail" in F["K_used_from"] else ("given" if "given" in F["K_used_from"] else "extrapolated")
        ax.plot(agd, F["K_used"] * cg, "-.", color="#d62728", lw=1.7, label=r"$K_\infty\cos\alpha$, $K_\infty$ = %.3f (%s)" % (F["K_used"], src))
    if args.show_old:
        ax.plot(agd, F["K_fit_old"] * cg, ":", color="#d62728", lw=1.0, label=r"old: $K\cos\alpha$ least squares on $\mu\geq%.2f$ (%.3f)" % (args.K_mu_min, F["K_fit_old"]))
    if "K_d" in F:
        ax.plot(agd, F["K_d"] * cg, "-", color="#ff7f0e", lw=1.0, label=r"$K_d\cos\alpha$, $K_d = E_d/\pi$ = %.3f (isolated dislocation)" % F["K_d"])
    ymax = 1.08 * float(np.nanmax(E[acc])) * 2.4
    marks = [(ALPHA_43, r"%.1f$^\circ$: $\sin^3\alpha=1-\sin\alpha$" % ALPHA_43),
             (F["first_charge_deg"], r"%.1f$^\circ$: first charge" % F["first_charge_deg"]),
             (F["cross_sin3_Kused_deg"], r"%.1f$^\circ$: $C\sin^3\alpha=K_\infty\cos\alpha$" % F["cross_sin3_Kused_deg"]),
             (F["peak_deg"], r"%.1f$^\circ$: peak" % F["peak_deg"])]
    for j, (xv, lab) in enumerate(marks):
        if np.isfinite(xv):
            ax.axvline(xv, color="0.65", ls=":", lw=1)
            ax.text(xv + 0.3, ymax * (0.97 - 0.045 * j), lab, fontsize=7, color="0.3", va="top")
    ax.set_ylim(0, ymax); ax.set_xlim(agd[0], agd[-1])
    ax.set_xlabel(r"far-field angle $\alpha$ (deg)"); ax.set_ylabel(r"line energy per unit length $E_{GB}$")
    ax.legend(fontsize=6.8, loc="upper left", bbox_to_anchor=legend_loc)
    _style(ax)


def draw_b(ax, S, F, T, args):
    """panel (b): K(alpha) = E / cos alpha against 1 - mu, the main sweep and the tail, with the plateau"""
    mu, E, nd, acc, a = S["mu"], S["E"], S["nd"], S["acc"], S["alpha"]
    c = np.cos(a); k1 = acc & (nd >= 1); rj = ~acc
    K = E / np.maximum(c, 1e-12); om = 1 - mu
    ax.plot(om[k1], K[k1], "o", mfc="w", mec="crimson", mew=1.3, ms=5.5, label=r"main sweep: $E_{GB}/\cos\alpha$, charged, accepted")
    if (rj & (nd >= 1)).any():
        ax.plot(om[rj & (nd >= 1)], K[rj & (nd >= 1)], "o", mfc="none", mec="0.6", ms=5, label="excluded")
    if T is not None:
        kt = T["E"] / np.maximum(np.cos(T["alpha"]), 1e-12); mt = T["acc"] & (T["nd"] >= 1) & T["settled"]; mu_ = T["acc"] & (T["nd"] >= 1) & ~T["settled"]
        ax.plot(1 - T["mu"][mt], kt[mt], "s", mfc="w", mec="#7b3fa0", mew=1.1, ms=4.5, label="tail sweep, settled (change between the last two frames at most %g%%)" % (100 * args.settle_tol))
        if mu_.any():
            ax.plot(1 - T["mu"][mu_], kt[mu_], "s", mfc="none", mec="#c9b3dd", mew=1.0, ms=4.5, label="tail sweep, not settled")
    xmax = float(np.max(om[k1])) * 1.05 if k1.any() else 0.2
    xx = np.linspace(0, xmax, 50)
    show_extrap = bool(args.show_extrap) or ("straight line" in F["K_used_from"])
    if show_extrap and np.isfinite(F["K_inf"]):
        ax.plot(xx, F["K_inf"] + F["K_inf_slope"] * xx, "-", color="#d62728", lw=1.3,
                label=r"straight line through the main sweep, $\mu\geq%.2f$: extrapolates to %.3f" % (args.K_mu_min, F["K_inf"]))
    if T is not None and np.isfinite(FT_plateau(T, args)):
        ax.axhline(FT_plateau(T, args), color="#7b3fa0", ls="-", lw=1.3, label=r"$K_\infty$ = %.3f, the mean over the tail's settled points" % FT_plateau(T, args))
    elif "given" in F["K_used_from"]:
        ax.axhline(F["K_used"], color="#7b3fa0", ls="-", lw=1.3, label=r"$K_\infty$ = %.3f (given)" % F["K_used"])
    if args.show_old:
        ax.axhline(F["K_fit_old"], color="0.5", ls=":", lw=1, label=r"old constant $K$ = %.3f" % F["K_fit_old"])
    if "K_d" in F:
        ax.axhline(F["K_d"], color="#ff7f0e", lw=1.0, label=r"$K_d = E_d/\pi$ = %.3f" % F["K_d"])
    ax.set_xlim(xmax, -0.005)
    ax.set_xlabel(r"$1-\mu$  (0 is $\alpha$ = 90$^\circ$)"); ax.set_ylabel(r"$K(\alpha)=E_{GB}/\cos\alpha$  (energy per dislocation over $\pi$)")
    ax.legend(fontsize=7, loc="lower right")
    _style(ax)


def draw_c(ax, S, F, args):
    """panel (c): the measured residual E_n between 43 deg and the onset"""
    on_deg = float(np.degrees(np.arcsin(args.onset_mu)))
    tab = np.array(F["E_n_table"]) if F["E_n_table"] else np.zeros((0, 5))
    if len(tab):
        ax.plot(tab[:, 0], tab[:, 4], "o-", color="#2a8f3c", ms=4, lw=1.2, label=r"$E_n(\alpha) = E_{GB} - C\,(1-\sin\alpha)$, measured")
        ax.plot(tab[:, 0], tab[:, 2], "o", mfc="w", mec="crimson", mew=1.2, ms=5, label=r"$E_{GB}$ (charged)")
        ax.plot(tab[:, 0], tab[:, 3], "--", color="#2a8f3c", lw=1.0, label=r"$C\,(1-\sin\alpha)$")
    ax.axvline(ALPHA_43, color="0.65", ls=":", lw=1); ax.axvline(on_deg, color="0.65", ls=":", lw=1)
    ax.text(ALPHA_43 + 0.3, 0.02, r"43.0$^\circ$", fontsize=7, color="0.3"); ax.text(on_deg + 0.3, 0.02, r"onset %.1f$^\circ$" % on_deg, fontsize=7, color="0.3")
    ax.set_xlabel(r"far-field angle $\alpha$ (deg)"); ax.set_ylabel("energy per unit length")
    ax.legend(fontsize=7, loc="upper left"); _style(ax)


def draw_d(ax, args):
    """panel (d): the dislocation spacing against the angle"""
    on_deg = float(np.degrees(np.arcsin(args.onset_mu)))
    agL = np.radians(np.linspace(20, 86, 400))
    ax.plot(np.degrees(agL), np.pi / np.cos(agL), "-", color="0.2", lw=1.4, label=r"$k_B L = \pi/\cos\alpha$ ($k_B = 1$): one dislocation per $L$")
    ax.plot(np.degrees(agL), 2 * np.pi / np.cos(agL), "--", color="0.5", lw=1.0, label=r"$2\pi/\cos\alpha$: the pattern's period along the line")
    ax.axhline(2 * np.pi, color="0.7", ls=":", lw=1, label=r"one stripe wavelength $2\pi$")
    ax.axvline(ALPHA_43, color="0.65", ls=":", lw=1); ax.axvline(on_deg, color="0.65", ls=":", lw=1)
    ax.set_ylim(0, 40); ax.set_xlabel(r"far-field angle $\alpha$ (deg)"); ax.set_ylabel("length (stripe units, $k_B = 1$)")
    ax.legend(fontsize=7, loc="upper left"); _style(ax)


def draw_tail(ax, T, F, args):
    """the tail sweep's line energy against the angle with the same laws (replaces the old tail figure)"""
    mu, E, nd, acc, a = T["mu"], T["E"], T["nd"], T["acc"], T["alpha"]
    ad = np.degrees(a); st = T["settled"]
    ax.plot(ad[acc & st], E[acc & st], "o", mfc="w", mec="crimson", mew=1.2, ms=5, label="tail sweep, settled")
    ax.plot(ad[acc & ~st], E[acc & ~st], "o", mfc="none", mec="#e8a0a8", mew=1.0, ms=5, label="tail sweep, not settled")
    if (~acc).any():
        ax.plot(ad[~acc], E[~acc], "o", mfc="none", mec="0.6", ms=5, label="excluded")
    ag = np.radians(np.linspace(ad.min() - 1, ad.max() + 1, 300)); agd = np.degrees(ag)
    ax.plot(agd, F["K_used"] * np.cos(ag), "-.", color="#d62728", lw=1.7, label=r"$K_\infty\cos\alpha$, $K_\infty$ = %.3f" % F["K_used"])
    ax.plot(agd, F["C"] * (1 - np.sin(ag)), "--", color="#2a8f3c", lw=1.2, label=r"$C\,(1-\sin\alpha)$, the knee's $C$")
    ax.set_xlabel(r"far-field angle $\alpha$ (deg)"); ax.set_ylabel(r"$E_{GB}$"); ax.legend(fontsize=7.5, loc="upper right"); _style(ax)


def figure(S, F, T, args, out):
    # the combined figure, two or four panels
    if args.layout == "abcd":
        fig, axs2 = plt.subplots(2, 2, figsize=(12.8, 9.6), dpi=170)
        draw_a(axs2[0, 0], S, F, args); draw_b(axs2[0, 1], S, F, T, args); draw_c(axs2[1, 0], S, F, args); draw_d(axs2[1, 1], args)
        axs2[0, 0].set_title("(a) the three laws with one knee constant and the dislocation law from its limit", fontsize=9, loc="left")
        axs2[0, 1].set_title("(b) the energy per dislocation rises toward its isolated value", fontsize=9, loc="left")
        axs2[1, 0].set_title(r"(c) the cost the protuberance law leaves out, read from the data between 43$^\circ$ and the onset", fontsize=9, loc="left")
        axs2[1, 1].set_title("(d) the dislocation spacing against the angle", fontsize=9, loc="left")
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12.8, 5.0), dpi=170)
        draw_a(axs[0], S, F, args); draw_b(axs[1], S, F, T, args)
        axs[0].set_title("(a) the line energy with the knee law, the same-constant protuberance law and the dislocation law", fontsize=9, loc="left")
        axs[1].set_title(r"(b) the energy per dislocation, $E_{GB}/\cos\alpha$, and its limit", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(out / "fig_energy_fits.png"); fig.savefig(out / "fig_energy_fits.pdf"); plt.close(fig)
    # the single panels, for the chapter
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=170); draw_a(ax, S, F, args, legend_loc=(0.0, 0.84))
    fig.tight_layout(); fig.savefig(out / "fig_energy_curve.png"); fig.savefig(out / "fig_energy_curve.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=170); draw_b(ax, S, F, T, args)
    fig.tight_layout(); fig.savefig(out / "fig_K_of_alpha.png"); fig.savefig(out / "fig_K_of_alpha.pdf"); plt.close(fig)
    if T is not None:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=170); draw_tail(ax, T, F, args)
        fig.tight_layout(); fig.savefig(out / "fig_tail_energy.png"); fig.savefig(out / "fig_tail_energy.pdf"); plt.close(fig)


def report(S, F, T, FT, args, out):
    L = [VERSION, "sweep: %s  (%d angles, %d accepted, %d charged)" % (S["path"], len(S["mu"]), int(S["acc"].sum()), int((S["nd"] >= 1).sum())),
         "args: " + json.dumps(vars(args)), ""]
    L.append("knee law: C = %.4f from %d accepted uncharged points with mu <= %.2f (rms relative residual %.3f)" % (F["C"], F["n_knee"], args.sin3_mu_max, F["C_resid_rms_rel"]))
    L.append("protuberance law C (1 - sin alpha) with the same C: crosses the knee law at %.2f deg by construction (sin^3 = 1 - sin)." % ALPHA_43)
    L.append("   against the data: " + "; ".join("%g deg: line %.4f, data %.4f" % tuple(t) for t in F["C_oms_vs_data"]))
    L.append("   (v2 fitted C1 = %.4f on the charged branch; that line crosses the knee law at %.1f deg)" % (F["C1_fit_old"], F["cross_sin3_C1old_deg"]))
    L.append("first director charge at mu %.3f (%.1f deg); the data leave the knee law by more than %.0f%% from mu %.3f (%.1f deg)" % (
        F["first_charge_mu"], F["first_charge_deg"], 100 * args.leave_tol, F["leave_knee_mu"], F["leave_knee_deg"]))
    L.append("dislocation law: K(alpha) = E / cos alpha, straight line in 1 - mu:")
    for nm in ("K_inf", "K_inf_b", "K_inf_c"):
        L.append("   mu >= %.2f (%d points): K_inf = %.4f, slope %.3f per unit (1 - mu)" % (F[nm + "_mu_min"], F[nm + "_n"], F[nm], F[nm + "_slope"]))
    L.append("   mean of K over mu >= %.2f (%d points): %.4f;  last points (mu, K): %s" % (args.K_top_mu, F["K_top_n"], F["K_top_mean"], ", ".join("(%.3f, %.3f)" % tuple(p) for p in F["K_last_points"])))
    L.append("   (v2's single least-squares K over mu >= %.2f: %.4f; its line crosses the knee law at %.1f deg)" % (args.K_mu_min, F["K_fit_old"], F["cross_sin3_Kold_deg"]))
    L.append("   K_inf cos alpha lies above the data wherever K(alpha) < K_inf, i.e. everywhere below the top of the sweep, and meets it from above.")
    if "K_d" in F:
        L.append("   isolated dislocation: E_d = %.4f given, K_d = E_d / pi = %.4f" % (args.E_d, F["K_d"]))
    L.append("   plateau means of this sweep: all charged %.4f (%d), settled charged (|E - E_prev| <= %.0f%%) %.4f +- %.4f (%d)" % (
        F["K_all_mean"], F["K_all_n"], 100 * args.settle_tol, F["K_plateau_mean"], F["K_plateau_std"], F["K_plateau_n"]))
    L.append("K_inf USED for the drawn line: %.4f (%s)" % (F["K_used"], F["K_used_from"]))
    L.append("   K_inf cos alpha over the data (ratio line/data) at the charged angles: " + ", ".join("%.0f deg %.2f" % tuple(t) for t in F["Kused_line_over_data"]))
    L.append("crossing of the knee law with the dislocation law: %.1f deg (with the extrapolated K_inf: %.1f deg); measured peak at mu %.3f (%.1f deg, E = %.4f)" % (
        F["cross_sin3_Kused_deg"], F["cross_sin3_Kinf_deg"], F["peak_mu"], F["peak_deg"], F["peak_E"]))
    L.append("log-log slope of E against 1 - mu over mu >= %.2f: %.3f (0.5 is cos alpha, 1 is 1 - sin alpha)" % (args.loglog_mu_min, F["loglog_slope"]))
    L.append("")
    L.append("the cost the protuberance law leaves out, E_n(alpha) = E_GB - C (1 - sin alpha), from 43 deg to the onset (mu %.2f, %.1f deg); a straight line in the angle has slope %.4f per degree and crosses zero at %.1f deg:" % (F["onset_mu"], F["onset_deg"], F["E_n_slope_per_deg"], F["E_n_zero_deg"]))
    for t in F["E_n_table"]:
        L.append("   %.1f deg (mu %.3f): E %.4f, C(1 - sin) %.4f, E_n %.4f" % tuple(t))
    L.append("at the onset: E_GB = %.4f, E_n = %.4f; a dislocation law K cos alpha passing through the data there needs K = %.3f, against K_inf = %.3f (the isolated value): the energy per dislocation at the onset spacing is %.0f%% of the isolated one" % (
        F["E_at_onset"], F["E_n_at_onset"], F["K_needed_at_onset"], F["K_used"], 100 * F["K_needed_at_onset"] / F["K_used"]))
    if T is not None:
        L.append(""); L.append("tail sweep: %s (%d angles, %d accepted)" % (T["path"], len(T["mu"]), int(T["acc"].sum())))
        for nm in ("K_inf", "K_inf_b", "K_inf_c"):
            L.append("   mu >= %.2f (%d points): K_inf = %.4f, slope %.3f" % (FT[nm + "_mu_min"], FT[nm + "_n"], FT[nm], FT[nm + "_slope"]))
        L.append("   last points (mu, K): %s" % ", ".join("(%.3f, %.3f)" % tuple(p) for p in FT["K_last_points"]))
        L.append("   plateau means: all charged %.4f (%d), settled charged %.4f +- %.4f (%d); v2's single fit %.4f" % (
            FT["K_all_mean"], FT["K_all_n"], FT["K_plateau_mean"], FT["K_plateau_std"], FT["K_plateau_n"], FT["K_fit_old"]))
        L.append("   log-log slope over mu >= %.2f: %.3f" % (args.loglog_mu_min, FT["loglog_slope"]))
    (out / "fit_report.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(args):
    out = Path(args.outdir)
    if not out.is_absolute():
        out = HERE / out
    out.mkdir(parents=True, exist_ok=True)
    S = read_sweep(args.csv, args)
    F = fits(S, args)
    T = FT = None
    if args.tail_csv:
        T = read_sweep(args.tail_csv, args); FT = fits(T, args)
    src = args.K_source if args.K_source in ("tail", "extrap", "value") else ("tail" if T is not None else "extrap")
    if src == "tail" and FT is not None:
        F["K_used"] = FT["K_plateau_mean"]; F["K_used_from"] = "tail plateau: mean of E/cos over %d settled charged points of %s" % (FT["K_plateau_n"], args.tail_csv)
    elif src == "value" and args.K_value > 0:
        F["K_used"] = float(args.K_value); F["K_used_from"] = "given by --K_value"
    else:
        F["K_used"] = F["K_inf"]; F["K_used_from"] = "straight line in 1 - mu over mu >= %.2f of the sweep" % args.K_mu_min
    ss = np.linspace(0.3, 0.9999, 200001)
    v = F["C"] * ss ** 3 - F["K_used"] * np.sqrt(1 - ss ** 2); F["cross_sin3_Kused_deg"] = float(np.degrees(np.arcsin(ss[int(np.argmin(np.abs(v)))])))
    c_ = np.cos(S["alpha"]); m_ = S["acc"] & (S["nd"] >= 1)
    F["Kused_line_over_data"] = [[float(np.degrees(S["alpha"][i])), float(F["K_used"] * c_[i] / S["E"][i])] for i in np.where(m_)[0]]
    figure(S, F, T, args, out)
    report(S, F, T, FT, args, out)
    with open(out / "fits.json", "w") as fh:
        json.dump({"sweep": F, "tail": FT, "args": vars(args), "version": VERSION}, fh, indent=1, default=float)
    print("outputs ->", out)
    return 0


class _Args:
    """defaults; mirrored by the CLI"""
    csv = str(Path("results") / "energy_vs_angle_zigzag_0929_np4_full" / "energy_vs_angle.csv")   # the full sweep (R 0.5, t 250)
    tail_csv = ""                  # optional second sweep for panel (b), e.g. the tail sweep 0.97 to 0.999
    outdir = str(Path("results") / "energy_fit_review")
    sin3_mu_max = 0.70             # knee branch: accepted, uncharged, mu at or below this (v2's window)
    K_mu_min = 0.85                # K_inf: straight line in 1 - mu over accepted charged points from here (v2's window)
    K_mu_min_b = 0.92              # second window, reported beside it
    K_mu_min_c = 0.95              # third window
    K_top_mu = 0.96                # plain mean of K over the top accepted charged points, reported
    loglog_mu_min = 0.90           # v2's exponent test window
    leave_tol = 0.05               # the data "leave the knee law" when below C sin^3 by more than this fraction
    E_d = 0.0                      # > 0: a measured isolated dislocation energy; draws K_d = E_d / pi
    K_source = ""                  # which K_inf the drawn line uses: "tail" (plateau mean of the tail sweep's settled charged
                                   # points; the default when --tail_csv is given), "extrap" (straight line in 1 - mu over
                                   # mu >= K_mu_min of the main sweep; the default otherwise) or "value" (--K_value)
    K_value = 0.0
    settle_tol = 0.02              # a point is settled when |E - E_prev| <= this fraction of E (the chapter's two percent)
    onset_mu = 0.82                # v1.1: the dislocation onset (first angle with one turn of circulation in the survey), 55.1 deg
    det_width = 0.4                # the detector width of the charge tag, for the legend only
    show_old = 0                   # 1: draw v2's least-squares K line dotted for comparison (v1.2: off by default)
    show_extrap = 0                # 1: draw the straight-line extrapolation in panel (b) even when K_inf comes from the tail
    dotted_segment = 1             # 1: draw C (1 - sin) dotted from 43 deg to the onset (the advisor's convention)
    layout = "ab"                  # "ab": two panels (the chapter's); "abcd": adds the E_n panel and the spacing panel
    Rscale = 0.5                   # only used to recompute acceptance for CSVs without an accepted column
    accept_ppw_floor = 8.0
    accept_flat_tol = 0.05
    accept_base_tol = 0.15


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in vars(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    sys.exit(main(p.parse_args()))
