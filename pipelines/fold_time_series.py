#!/usr/bin/env python
"""
fold_time_series.py  v1.0 (27 Oct 2026)

Experiment 4 of the phase-or-amplitude study: does the fold of the phase grow smoothly in time while the
circulation jumps, and is the jump set by where the loop sits?

Background (PHASE_AMP_V11_LOCAL_RESULTS_1027.md, stationary frames against angle): the smooth phase phi_LS
fitted to mg's k folds across the line (its gradient just above the line points against the one just below)
over a fraction of the line that grows with no step through the circulation switch, and the measured one turn
of the dislocations is reproduced by that folded single-valued phase (plain line integral 0, sign-carried
integral about 1, two flips per loop). This script asks the same questions frame by frame.

Per frame, imported from phase_vs_amplitude_charge_v11.measure (not copied): director charge per L of u,
cos(theta) (the phase alone), cos(phi_LS) and A cos(phi_LS); the fold fraction of the line for grad phi_LS and
for mg's k of u; the curl fraction; the depth of the amplitude minimum; the plain and sign-carried circulation
of grad phi_LS about u's dips. New here:

  (1) Fixed loops. Centres on the line at the x positions of the last frame's dips (dips within 0.3 L in x
      merged, so an off-line pair gives one centre), radii --radii_L, the
      same loops in every frame, so the circulation no longer depends on finding dips (default radii 0.25, 0.35,
      0.5, 0.75 L; the nearest neighbouring core is about 1 L away, so 0.75 L still holds one). For each radius: the
      median |Gamma| over the centres (sign carried, as loop_grid), the fraction of centres whose loop closes
      with |Gamma| >= 0.5, and the median number of flips per loop. A fold growing along the line reaches a
      small loop's crossing points before a large one's, so a continuous fold predicts radius-ordered jump
      times; one frame for every radius means the event is abrupt at this cadence.
  (2) Circulation about each dip (sign carried, 0.5 L, as the campaign): the largest closed on-line value.
  (3) Amplitude at the fold: along the line row, the median and minimum of A / A_far on the stretch where
      grad phi_LS folds and on the stretch where it does not (the same for mg's k of u).
  (4) Dip positions: number of dips on the line (within on_line_L) and off it, median offset.

Event times per movie (first frame at which): u charged (amplitude-carried charge), cos(theta) charged
(phase-carried charge), fold fraction of grad phi_LS >= fold_on, a dip on the line, an on-line dip wound
(closed, >= wind_hi), and each fixed radius with half its centres at >= 0.5 turn.

The planted check (--control 1): a synthetic movie on the exact knee phase with the complex-envelope vortex
pair of phase_vs_amplitude_charge (same formula, re-implemented in pair_movie_frame), separation s from
ctl_s_from to ctl_s_to L; s plays the role of time. The truth is known: a loop of radius r centred at the
final core position contains one vortex once s > 1 - 2r (L units), so the true jump order is the largest loop
first (0.5 L at s = 0, 0.35 at 0.3, 0.25 at 0.5, 0.15 at 0.7). What the pipeline reports against that truth
says how far its readings in time can be trusted.

Sandbox run of the planted check (27 Oct, mu 0.88, s from -0.4 to 1.0 in steps of 0.05, loops on the true final
vortex positions). True first winding against what the pipeline reports: r 0.50 L at s 0.05, reported 0.70;
r 0.35 at 0.35, reported 1.00; r 0.75 at 0.55, reported 0.95; r 0.25 at 0.55, never (0.15 L, tried first and
dropped, also never). The pipeline reports a true, continuous separation late, by 0.4 to 0.65 L of separation,
and loops of 0.25 L or less are below what the estimator resolves (sigma_dir 0.4). The median |Gamma| of the
0.5 L loops, open loops included, rises steadily from 0.69 to 0.99 over the same range, so
the dotted lines of panel (b) carry the continuous part that the wound fraction cuts into a step.

Outputs (in --outdir; every name begins with fold_ so they cannot be confused with the charge_circ or
phase_amp outputs):
  fold_events.csv / .json           one row per movie: the event times above
  fold_event_order.png              the event times of every movie, fine and coarse cadence in separate panels
  <stem>/fold_series.csv            one row per frame
  <stem>/fold_series.png            (a) charges (b) circulation: fixed loops by radius and on-line dips
                                    (c) fold fractions and curl (d) amplitude on and off the fold, minimum depth
                                    (e) dip offsets
  <stem>/fold_kymo.png              space-time pictures, x along the line across and time down: where grad
                                    phi_LS folds, where mg's k of u folds, A / A_far on the line, with the
                                    fixed-loop jump times marked
  control/...                       the same for the planted movie
  fold_notes.txt                    version, inputs, flags, one line per movie

Imports phase_vs_amplitude_charge_v11 (which imports pgb_charge_circ_survey_v12, pgb_marker_survey,
mg_estimator_validation); edits none of them. Python 3.8 compatible. Run from ~/patterns/pipelines
(FOLD_TIME_CLI_1027.txt).
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_vs_amplitude_charge_v11 as P                                  # noqa: E402
from pgb_marker_survey import loop_grid, expand_inputs, load_npz          # noqa: E402
import matplotlib                                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402

VERSION = "fold_time_series v1.0 (27 Oct 2026, on %s)" % P.VERSION

# the per-frame field set: charge for four fields, circulation re-estimated only for u (keeps a frame near 1 s)
P.FIELDS = ["u", "cos_theta", "cos_phiLS", "A_cos_phiLS"]
P.CIRC_FIELDS = ["u"]


# ---------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------

def mu_of(path):
    try:
        d = np.load(path, allow_pickle=True)
        if "metadata_json" in d.files:
            md = json.loads(str(d["metadata_json"]))
            if "mu" in md:
                return float(md["mu"])
        if "mu" in d.files:
            return float(d["mu"])
    except Exception:
        pass
    m = re.search(r"mu([0-9]*\.[0-9]+)", Path(path).name)
    return float(m.group(1)) if m else float("nan")


def select_files(spec, mus):
    files = expand_inputs(spec)
    if not mus:
        return files
    want = [float(v) for v in mus.split(",")]
    return [f for f in files if any(abs(mu_of(f) - w) < 0.0051 for w in want)]


def movie(path):
    D = load_npz(path)
    u = D["u"]
    if u.ndim == 2:
        u = u[..., None]
    md = dict(D["md"])
    return u, np.asarray(D["x"], float), np.asarray(D["y"], float), np.atleast_1d(np.asarray(D["t"], float)), md


def pair_movie_frame(mu, s, args):
    """the planted vortex pair on the exact knee phase (phase_vs_amplitude_charge.control_fields, 'pair')"""
    x, y, L, k1 = P.control_grid(mu, args)
    X, Y = np.meshgrid(x, y)
    th = k1 * X + np.logaddexp(mu * Y, -mu * Y)
    Pd = 2 * L
    ww = args.ctl_pair_w_L * L
    c0 = np.cos(np.pi * s * L / Pd) if s >= 0 else np.cosh(np.pi * abs(s) * L / Pd)
    q = (1 + np.cos(2 * np.pi * X / Pd)) / (1 + c0)
    env = np.exp(-Y ** 2 / ww ** 2)
    W = (1 - env * q) + 1j * (Y / ww) * env * args.ctl_pair_beta
    return np.real(W * np.exp(1j * th)), x, y


# ---------------------------------------------------------------------
# per-frame measurement
# ---------------------------------------------------------------------

def line_rows(y, G, args):
    jr = int(np.argmin(np.abs(y - G["y_row"])))
    off = max(1, int(round(args.fold_off_L * G["L"] / G["dy"])))
    return jr, min(jr + off, len(y) - 1), max(jr - off, 0)


def fold_line(k1, k2, a, b, cols):
    return (k1[a, cols] * k1[b, cols] + k2[a, cols] * k2[b, cols]) < 0


def centres_from(xd, L, xm, Lx):
    """one centre per distinct x (dips within 0.3 L in x merged)"""
    xs = np.sort(np.asarray(xd, float))
    xs = xs[np.abs(xs - xm) < 0.4 * Lx]
    out = []
    for v in xs:
        if out and abs(v - out[-1][-1]) < 0.3 * L:
            out[-1].append(v)
        else:
            out.append([v])
    return np.array([np.mean(g) for g in out])


def frame_record(u2, x, y, y_row, mu, args, centres=None):
    G = P.geometry(u2, x, y, y_row, mu, args)
    row, M = P.measure(u2, x, y, G, args, keep_maps=True)
    L = G["L"]
    e = P.estimate(u2, G["Lx"], G["Ly"], args.sigma_dir, None, G["taper"])
    A, A_far = M["A"], M["A_far"]
    phi = M["phi"]
    gy, gx = np.gradient(phi, G["dy"], G["dx"])
    jr, a, b = line_rows(y, G, args)
    cols = np.abs(x - G["xm"]) < 0.4 * G["Lx"]
    fphi = fold_line(gx, gy, a, b, cols)
    fku = fold_line(e["k1"], e["k2"], a, b, cols)
    Aline = A[jr, cols] / A_far
    for tag, f in [("gradphi", fphi), ("k_u", fku)]:
        row["A_fold_med_" + tag] = float(np.median(Aline[f])) if f.any() else float("nan")
        row["A_fold_min_" + tag] = float(Aline[f].min()) if f.any() else float("nan")
        row["A_unfold_med_" + tag] = float(np.median(Aline[~f])) if (~f).any() else float("nan")
        row["A_unfold_min_" + tag] = float(Aline[~f].min()) if (~f).any() else float("nan")
    xd, yd = M["xd"], M["yd"]
    off = np.abs(yd - y_row) / L
    on = off <= args.on_line_L
    row["n_dips_on"] = int(on.sum()); row["n_dips_off"] = int((~on).sum())
    row["dip_off_med_L"] = float(np.median(off)) if len(off) else float("nan")
    row["dip_off_min_L"] = float(off.min()) if len(off) else float("nan")
    if len(xd):
        c, cl = loop_grid(e["k1"], e["k2"], x, y, xd, yd, args.loop_L * L, args.nang)
        c = np.abs(c); cl = np.asarray(cl, bool)
        row["G_dip_on_max"] = float(c[on & cl].max()) if (on & cl).any() else 0.0
        row["G_dip_max"] = float(c[cl].max()) if cl.any() else 0.0
    else:
        row["G_dip_on_max"] = 0.0; row["G_dip_max"] = 0.0
    if centres is not None and len(centres):
        cy = np.full(len(centres), y_row)
        for r_L in [float(v) for v in args.radii_L.split(",")]:
            c, cl = loop_grid(e["k1"], e["k2"], x, y, centres, cy, r_L * L, args.nang)
            c = np.abs(c); cl = np.asarray(cl, bool)
            _, fl = P.loop_plain(e["k1"], e["k2"], x, y, centres, cy, r_L * L, args.nang)
            tag = ("%.2f" % r_L).replace(".", "p")
            row["G_fix_med_r" + tag] = float(np.median(c))
            row["G_fix_frac_wound_r" + tag] = float(np.mean(cl & (c >= 0.5)))
            row["flips_fix_med_r" + tag] = float(np.median(fl))
    kymo = dict(fold_phi=fphi, fold_ku=fku, A_line=Aline, x=(x[cols] - G["xm"]) / L,
                dips=np.c_[(xd - G["xm"]) / L, off] if len(xd) else np.zeros((0, 2)))
    for k in ["fields", "theta", "theta_un", "phi", "psi", "rk"]:
        M.pop(k, None)
    return row, kymo, G


def first_time(ts, cond):
    hit = np.where(np.asarray(cond, bool))[0]
    return float(ts[hit[0]]) if len(hit) else float("nan")


def events(rows, args):
    ts = np.array([r["t"] for r in rows])
    g = lambda k: np.array([r.get(k, np.nan) for r in rows], float)      # noqa: E731
    ev = dict(t_charge_u=first_time(ts, g("T_per_L_u") > 0),
              t_charge_cos_theta=first_time(ts, g("T_per_L_cos_theta") > 0),
              t_fold_phi=first_time(ts, g("fold_frac_line_gradphi") >= args.fold_on),
              t_fold_ku=first_time(ts, g("fold_frac_line_k_u") >= args.fold_on),
              t_dip_on_line=first_time(ts, g("n_dips_on") > 0),
              t_wound_dip=first_time(ts, g("G_dip_on_max") >= args.wind_hi))
    for r_L in [float(v) for v in args.radii_L.split(",")]:
        tag = ("%.2f" % r_L).replace(".", "p")
        ev["t_fix_r" + tag] = first_time(ts, g("G_fix_frac_wound_r" + tag) >= 0.5)
    return ev


# ---------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------

def radii(args):
    return [float(v) for v in args.radii_L.split(",")]


def _tag(r_L):
    return ("%.2f" % r_L).replace(".", "p")


def fig_series(rows, ev, title, path, args, tlabel="t"):
    ts = np.array([r["t"] for r in rows])
    g = lambda k: np.array([r.get(k, np.nan) for r in rows], float)      # noqa: E731
    fig, axs = plt.subplots(5, 1, figsize=(8.5, 13), sharex=True, constrained_layout=True)
    ax = axs[0]
    for nm, c, lab in [("u", "k", "u (amplitude included)"), ("cos_theta", "C0", "cos(theta) (phase alone)"),
                       ("cos_phiLS", "C3", "cos(phi_LS)"), ("A_cos_phiLS", "C1", "A cos(phi_LS)")]:
        ax.plot(ts, g("T_per_L_" + nm), "-", color=c, lw=1.6 if nm == "u" else 1.0, label=lab)
    ax.set_ylabel("director charge\nper L"); ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("(a) charge: amplitude-carried (u) and phase-carried (cos theta)", fontsize=9, loc="left")
    ax = axs[1]
    cm = plt.get_cmap("viridis")
    rr = radii(args)
    for i, r_L in enumerate(rr):
        ax.plot(ts, g("G_fix_frac_wound_r" + _tag(r_L)), "-", color=cm(0.1 + 0.8 * i / max(len(rr) - 1, 1)),
                label="fixed loops r = %.2f L: fraction wound" % r_L)
        ax.plot(ts, g("G_fix_med_r" + _tag(r_L)), ":", color=cm(0.1 + 0.8 * i / max(len(rr) - 1, 1)), lw=0.9)
    ax.plot(ts, g("G_dip_on_max"), "k--", lw=1, label="on-line dips, 0.5 L: largest closed |Gamma|")
    ax.plot(ts, g("G_plain_max_gradphi"), ":", color="0.5", label="grad phi_LS, plain line integral (largest)")
    ax.set_ylabel("fraction / turns"); ax.legend(fontsize=7, loc="upper left")
    ax.set_title("(b) circulation. Solid: fraction of fixed loops closed with |Gamma| >= 0.5, by radius;\n"
                 "dotted: their median |Gamma| (open loops included); dashed black: on-line dips", fontsize=9, loc="left")
    ax = axs[2]
    ax.plot(ts, g("fold_frac_line_gradphi"), "-", color="C3", label="fraction of the line where grad phi_LS folds")
    ax.plot(ts, g("fold_frac_line_k_u"), "--", color="k", label="fraction of the line where mg's k of u folds")
    ax.plot(ts, g("curl_frac_core"), "-", color="C4", label="curl fraction of k, core band")
    ax.set_ylabel("fraction"); ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("(c) the fold", fontsize=9, loc="left")
    ax = axs[3]
    ax.plot(ts, g("A_fold_med_gradphi"), "-", color="C3", label="A on the folded stretch of the line (median)")
    ax.plot(ts, g("A_fold_min_gradphi"), ":", color="C3", label="A on the folded stretch (minimum)")
    ax.plot(ts, g("A_unfold_med_gradphi"), "-", color="C2", label="A on the unfolded stretch (median)")
    ax.plot(ts, g("A_min_core_over_far"), "-", color="k", lw=1, label="deepest A in the core band")
    ax.set_ylabel("A / A_far"); ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("(d) amplitude on and off the fold", fontsize=9, loc="left")
    ax = axs[4]
    ax.plot(ts, g("dip_off_min_L"), "-", color="C0", label="nearest dip to the line")
    ax.plot(ts, g("dip_off_med_L"), "--", color="C0", label="median dip offset")
    ax.axhline(args.on_line_L, color="0.6", lw=0.8)
    ax2 = ax.twinx(); ax2.step(ts, g("n_dips_on"), where="mid", color="C1", lw=1); ax2.set_ylabel("dips on the line", color="C1")
    ax.set_ylabel("offset (L)"); ax.set_xlabel(tlabel); ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("(e) where the dips are", fontsize=9, loc="left")
    marks = [("t_charge_u", "k", "u charged"), ("t_charge_cos_theta", "C0", "cos theta charged"),
             ("t_fold_phi", "C3", "fold onset"), ("t_dip_on_line", "C1", "dip on line"),
             ("t_wound_dip", "m", "dip wound")]
    for ax in axs:
        for k, c, lab in marks:
            if np.isfinite(ev.get(k, np.nan)):
                ax.axvline(ev[k], color=c, lw=0.7, alpha=0.6)
    axs[0].text(0.99, 0.02, "  ".join("%s %s" % (lab, ("%.4g" % ev[k]) if np.isfinite(ev.get(k, np.nan)) else "never")
                                      for k, c, lab in marks), transform=axs[0].transAxes, ha="right", va="bottom", fontsize=6.5)
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=140); plt.close(fig)


def fig_kymo(kys, ts, ev, title, path, args, tlabel="t"):
    xk = kys[0]["x"]
    ext = [xk[0], xk[-1], ts[-1], ts[0]]
    fig, axs = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True, sharey=True)
    Fp = np.array([k["fold_phi"] for k in kys], float)
    Fk = np.array([k["fold_ku"] for k in kys], float)
    Al = np.array([k["A_line"] for k in kys], float)
    axs[0].imshow(Fp, aspect="auto", extent=ext, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    axs[0].set_title("where grad phi_LS folds across the line (red)", fontsize=9)
    axs[1].imshow(Fk, aspect="auto", extent=ext, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    axs[1].set_title("where mg's k of u folds across the line (black)", fontsize=9)
    im = axs[2].imshow(Al, aspect="auto", extent=ext, cmap="viridis", vmin=0, vmax=1.1, interpolation="nearest")
    plt.colorbar(im, ax=axs[2], shrink=0.8, label="A / A_far on the line")
    axs[2].set_title("amplitude along the line; dots: dips (white on the line, grey off it)", fontsize=9)
    for k, t in zip(kys, ts):
        d = k["dips"]
        if len(d):
            on = d[:, 1] <= args.on_line_L
            axs[2].plot(d[on, 0], np.full(on.sum(), t), ".", color="w", ms=2)
            axs[2].plot(d[~on, 0], np.full((~on).sum(), t), ".", color="0.6", ms=1.5)
    cm = plt.get_cmap("viridis")
    rr = radii(args)
    for ax in axs:
        for i, r_L in enumerate(rr):
            tt = ev.get("t_fix_r" + _tag(r_L), np.nan)
            if np.isfinite(tt):
                ax.axhline(tt, color=cm(0.1 + 0.8 * i / max(len(rr) - 1, 1)), lw=1.0, ls="--")
        ax.set_xlabel("x / L along the line")
    axs[0].set_ylabel(tlabel + "  (dashed: fixed loops wound, colour by radius as in fold_series)")
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=140); plt.close(fig)


def fig_event_order(evs, path, args):
    fine = [e for e in evs if e.get("dt", 1) < 0.1]
    coarse = [e for e in evs if e.get("dt", 1) >= 0.1]
    groups = [g for g in [("fine cadence", fine), ("coarse cadence", coarse)] if g[1]]
    if not groups:
        return
    fig, axs = plt.subplots(1, len(groups), figsize=(7 * len(groups), 1.2 + 0.55 * max(len(g[1]) for g in groups)),
                            constrained_layout=True, squeeze=False)
    cm = plt.get_cmap("viridis")
    rr = radii(args)
    keys = [("t_charge_u", "o", "k", "u charged"), ("t_charge_cos_theta", "s", "C0", "cos theta charged"),
            ("t_fold_phi", "D", "C3", "fold onset (grad phi_LS)"), ("t_dip_on_line", "^", "C1", "dip on the line"),
            ("t_wound_dip", "*", "m", "on-line dip wound")]
    for ax, (name, E) in zip(axs[0], groups):
        E = sorted(E, key=lambda e: e["mu"])
        for j, e in enumerate(E):
            for k, m, c, lab in keys:
                ax.plot(e[k], j, m, color=c, ms=7 if m == "*" else 5, label=lab if j == 0 else None, ls="none")
            for i, r_L in enumerate(rr):
                ax.plot(e["t_fix_r" + _tag(r_L)], j + 0.25, "|", color=cm(0.1 + 0.8 * i / max(len(rr) - 1, 1)), ms=9, mew=2,
                        label=("fixed loop r = %.2f L wound" % r_L) if j == 0 else None)
        ax.set_yticks(range(len(E))); ax.set_yticklabels(["mu %.3f" % e["mu"] for e in E])
        ax.set_xlabel("t"); ax.set_title(name + " (no marker: never)", fontsize=9)
        ax.legend(fontsize=6.5, loc="lower right")
    fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------

def write_rows(rows, base):
    P.write_rows(rows, base)


def run_frames(frames, mu, y_row, title, outdir, args, tlabel="t", centres=None):
    """frames: list of (t, u2, x, y). The centres come from the last frame's dips unless given."""
    outdir.mkdir(parents=True, exist_ok=True)
    tL, uL, xL, yL = frames[-1]
    G0 = P.geometry(uL, xL, yL, y_row, mu, args)
    if centres is None:
        eL = P.estimate(uL, G0["Lx"], G0["Ly"], args.sigma_dir, None, G0["taper"])
        xd, yd, _, _, _ = P.find_dip_patches(eL["A"], xL, yL, G0["win"], args.core_frac, 0.5 * G0["L"])
        centres = centres_from(xd, G0["L"], G0["xm"], G0["Lx"])
    rows, kys = [], []
    for i, (t, u2, x, y) in enumerate(frames):
        row, ky, G = frame_record(u2, x, y, y_row, mu, args, centres=centres)
        row = dict(t=float(t), frame=i, **row)
        rows.append(row); kys.append(ky)
    ev = events(rows, args)
    ev.update(mu=mu, n_frames=len(rows), n_centres=int(len(centres)),
              dt=float(frames[1][0] - frames[0][0]) if len(frames) > 1 else float("nan"))
    write_rows(rows, outdir / "fold_series")
    fig_series(rows, ev, title, outdir / "fold_series.png", args, tlabel)
    fig_kymo(kys, np.array([r["t"] for r in rows]), ev, title, outdir / "fold_kymo.png", args, tlabel)
    return rows, ev


def ev_line(name, ev, args):
    f = lambda v: ("%.4g" % v) if np.isfinite(v) else "never"           # noqa: E731
    return ("%s mu %.3f (%d frames, %d centres): u charged %s, cos theta charged %s, fold onset %s (k of u %s), dip on line %s,"
            " on-line dip wound %s; fixed loops wound: %s" % (
                name, ev["mu"], ev["n_frames"], ev["n_centres"], f(ev["t_charge_u"]), f(ev["t_charge_cos_theta"]),
                f(ev["t_fold_phi"]), f(ev["t_fold_ku"]), f(ev["t_dip_on_line"]), f(ev["t_wound_dip"]),
                ", ".join("r %.2f: %s" % (r_L, f(ev["t_fix_r" + _tag(r_L)])) for r_L in radii(args))))


def main(args):
    t0 = time.time()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    lines = [VERSION, "args: " + json.dumps(vars(args)), ""]
    evs = []
    if args.control:
        s_vals = np.arange(args.ctl_s_from, args.ctl_s_to + 1e-9, args.ctl_ds)
        frames = []
        for s in s_vals:
            u2, x, y = pair_movie_frame(args.ctl_mu, float(s), args)
            frames.append((float(s), u2, x, y))
        # the centres are the true vortex positions at the last separation (base +- s_to / 2, base on the 2 L period)
        x0, _, L0, _ = P.control_grid(args.ctl_mu, args)
        base = np.arange(-int(args.ctl_box_x_L), int(args.ctl_box_x_L) + 1) * 2 * L0
        zx = np.sort(np.concatenate([base - 0.5 * args.ctl_s_to * L0, base + 0.5 * args.ctl_s_to * L0]))
        zx = zx[np.abs(zx - x0.mean()) < 0.35 * args.ctl_box_x_L * L0]
        def first_true(r_L):
            # a loop of radius r about the final position of one vortex holds that vortex and not its partner
            ok = [sv for sv in s_vals if sv > 0 and abs(0.5 * args.ctl_s_to - 0.5 * sv) < r_L < 0.5 * args.ctl_s_to + 0.5 * sv]
            return ("%.2f" % ok[0]) if ok else "never"
        truth = ", ".join("r %.2f at s %s" % (r_L, first_true(r_L)) for r_L in radii(args))
        rows, ev = run_frames(frames, args.ctl_mu, 0.0, "planted vortex pair on the knee phase, mu %.2f; time axis = pair "
                              "separation s / L\nfixed loops on the true final vortex positions; true first winding: %s" % (args.ctl_mu, truth),
                              out / "control", args, tlabel="pair separation s / L", centres=zx)
        ev["truth"] = truth
        ev["stem"] = "control_pair_mu%.2f" % args.ctl_mu
        msg = ev_line("CONTROL", ev, args); print(msg); lines.append(msg)
        P.write_rows([ev], out / "control" / "fold_control_events")
    files = select_files(args.npz, args.mus) if args.npz else []
    if args.npz and not files:
        msg = "LOUD: no movie matched --npz %s --mus %s" % (args.npz, args.mus); print(msg); lines.append(msg)
    for p in files:
        u, x, y, t, md = movie(p)
        mu = float(md.get("mu", mu_of(p)))
        jr = int(md.get("gb_row_in_crop", u.shape[0] // 2)); y_row = float(y[jr])
        nt = u.shape[-1]
        idx = list(range(0, nt, max(1, args.frame_step)))
        if idx[-1] != nt - 1:
            idx.append(nt - 1)
        frames = [(float(t[min(i, t.size - 1)]), np.asarray(u[..., i], float), x, y) for i in idx]
        del u
        stem = Path(p).stem
        t1 = time.time()
        rows, ev = run_frames(frames, mu, y_row, "%s   mu %.3f (%.0f deg)" % (stem, mu, np.degrees(np.arcsin(mu))),
                              out / stem, args)
        ev["stem"] = stem
        evs.append(ev)
        msg = ev_line(stem, ev, args) + "  [%.0f s]" % (time.time() - t1); print(msg); lines.append(msg)
    if evs:
        P.write_rows(evs, out / "fold_events")
        fig_event_order(evs, out / "fold_event_order.png", args)
    lines += ["", "done in %.0f s" % (time.time() - t0)]
    (out / "fold_notes.txt").write_text("\n".join(lines) + "\n")
    print("done in %.0f s; outputs in %s" % (time.time() - t0, out))
    return 0


class _Args:
    """defaults; mirrored by the CLI below. Keys shared with phase_vs_amplitude_charge_v11 keep its values."""
    npz = ""                       # comma-separated movie files and/or folders
    mus = ""                       # comma-separated mu values to keep from --npz ("" keeps every file)
    outdir = str(Path("results") / "fold_time_series")
    frame_step = 1                 # every frame_step-th frame (the last frame always)
    radii_L = "0.25,0.35,0.5,0.75" # fixed-loop radii, L (0.15 dropped: never wound in the planted check)
    fold_on = 0.05                 # fold onset: this fraction of the line folded (the planted pair reads 0.02 to 0.03
                                   # before its zeros exist, so 0.02 is inside the noise)
    control = 0                    # 1: the planted vortex-pair movie first
    ctl_mu = 0.88
    ctl_s_from = -0.4
    ctl_s_to = 1.0
    ctl_ds = 0.05
    # shared with phase_vs_amplitude_charge_v11 (same values)
    mu = 0.85
    sigma_dir = 0.4
    taper_frac = 0.06
    sigma_det = 0.3
    band_L = 2.0
    core_band_L = 1.0
    far_from = 4.0
    core_frac = 0.70
    loop_L = 0.5
    fold_off_L = 0.05
    reg_eps = 0.1
    ls_mode = "plain"
    seam_dilate = 1
    nang = 181
    on_line_L = 0.1
    wind_hi = 0.9
    crop_x_L = 4.0
    crop_y_L = 1.5
    ctl_box_x_L = 8.0
    ctl_box_y_L = 5.0
    ctl_nx = 512
    ctl_pair_w_L = 0.3
    ctl_pair_beta = 1.0


def default_args():
    return argparse.Namespace(**{k: getattr(_Args, k) for k in vars(_Args) if not k.startswith("_")})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in vars(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    sys.exit(main(p.parse_args()))
