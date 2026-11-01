#!/usr/bin/env python
"""
core_track_time_series.py  v1.0 (27 Oct 2026)

Per-core analysis of the moment the circulation about a core reaches one turn. Companion to fold_time_series.py
(FOLD_TIME_LOCAL_RESULTS_1027.md), which found that quantities measured along the whole line (deepest amplitude,
fold fraction, charge, curl) do not move in the frame where the circulation reaches one turn, and that cores
reach one turn one at a time. This script measures each core on its own, in a small window around it, and asks
which local quantity, if any, changes in that core's own event frame.

Vocabulary: "circulation" is Gamma, the line integral of mg's k about a loop with the stripe direction carried
around the loop by continuity (loop_grid, the campaign's measure), in turns. "One turn of circulation" means a
closed loop reading |Gamma| >= turn_hi (0.9). The plain line integral (no sign carried) is reported beside it;
for the gradient of any single-valued phase it is zero. The word winding is not used.

Cores. Fixed for the whole movie: one centre on the line at each x position of the last frame's amplitude dips
(dips within 0.3 L in x merged), as in fold_time_series. A core's event time is the first frame at which its
0.5 L loop closes with |Gamma| >= turn_hi; a core whose loop later falls below 0.5 again is flagged.

Per core and frame, in the window |x - x_core| < 0.5 L, |y - y_line| < 0.75 L:
  G_carried, G_plain, flips        the 0.5 L loop: circulation, plain line integral, reversals on the loop
  A_min_loc, A_centre, A_low_area  amplitude: window minimum / A_far, A on the line at the core / A_far, area
                                   below 0.7 A_far in units of L^2
  fold_loc_phi, fold_loc_ku        fraction of the line within 0.5 L of the core where the vector 0.05 L above
                                   the line points against the one below (grad phi_LS; mg's k of u)
  cross_phi, cross_ku              at how many of the two points where the loop crosses the line (x_core +- 0.5 L)
                                   the line is folded (0, 1 or 2): tests the reading that the loop reads one turn
                                   when it crosses the fold twice
  nV_u, nX_u, V_off_L, VX_dist_L   half-turn cells of u (detector at sigma_det 0.3) within 0.5 L of the core:
                                   counts, median V offset from the line, nearest V-X distance
  n_cells_cos                      cells of cos(theta) (the phase alone) within 0.5 L of the core
  curl_loc                         share of |k|^2 in the window that no smooth phase represents

The test. For each core and quantity, the jump score: the change across the core's event frame divided by the
median frame-to-frame change of the same quantity over the rest of that core's history (floor: 1 percent of the
quantity's range, so a quantity that never moves scores 0 and one that moves only at the event scores high;
capped at 100). Near 1: nothing special at the event. Large: it changes exactly when the circulation does.

Outputs (every name begins with core_; folder names in the command list begin 1027_core_track_):
  core_events.csv / .json        one row per core of every movie: event time, flag, value before and after and
                                 jump score of every quantity
  core_jump_scores.png           median jump score per quantity and movie (the one-glance answer)
  <stem>/core_series.csv         one row per frame and core
  <stem>/core_aligned.png        every quantity against time from each core's own event, all cores overlaid,
                                 amplitude first
  <stem>/core_snapshots.png      one core (--snap_core, default the first to reach one turn) two frames before,
                                 one before, at and one after its event: pattern with cells, loop and fold marks;
                                 amplitude with the same
  control/...                    --control 1 only: the planted vortex pair, for checking the pipeline (not a
                                 results figure)
  core_notes.txt

Sandbox checks (27 Oct; not of record): the planted pair (mu 0.88, separation as time) and the SH stand-in
frames run end to end; see the notes in the chat log of that date.

Imports fold_time_series (movie loading, centres, the planted pair) and phase_vs_amplitude_charge_v11
(estimator, smooth phase, cell detector, loops); edits neither. Python 3.8 compatible. Run from
~/patterns/pipelines (CORE_TRACK_CLI_1027.txt).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fold_time_series as F                                               # noqa: E402
P = F.P
from pgb_marker_survey import loop_grid                                   # noqa: E402
import matplotlib                                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.patches import Circle                                     # noqa: E402

VERSION = "core_track_time_series v1.0 (27 Oct 2026, on %s)" % F.VERSION

QUANT = [  # key, label, panel group
    ("A_min_loc", "amplitude minimum near the core / A_far", "amplitude"),
    ("A_centre", "amplitude on the line at the core / A_far", "amplitude"),
    ("A_low_area", "area with A < 0.7 A_far (L^2)", "amplitude"),
    ("fold_loc_phi", "folded share of the line near the core (grad phi_LS)", "fold"),
    ("fold_loc_ku", "folded share of the line near the core (mg's k of u)", "fold"),
    ("cross_phi", "loop crossings on the fold (grad phi_LS)", "fold"),
    ("cross_ku", "loop crossings on the fold (mg's k of u)", "fold"),
    ("nV_u", "V cells of u near the core", "cells"),
    ("nX_u", "X cells of u near the core", "cells"),
    ("V_off_L", "V cell offset from the line (L)", "cells"),
    ("VX_dist_L", "nearest V-X distance (L)", "cells"),
    ("n_cells_cos", "cells of cos(theta) near the core", "cells"),
    ("curl_loc", "curl share of |k|^2 near the core", "phase"),
    ("G_plain", "plain line integral of k (turns)", "phase"),
    ("flips", "reversals of k on the loop", "phase"),
    ("G_carried", "circulation (turns), the defining quantity", "phase"),
]
GROUP_COLOR = {"amplitude": "#2a7f62", "fold": "#c0392b", "cells": "#2c5aa0", "phase": "#6c3483"}


# ---------------------------------------------------------------------
# per-frame, per-core measurement
# ---------------------------------------------------------------------

def frame_fields(u2, x, y, y_row, mu, args):
    G = P.geometry(u2, x, y, y_row, mu, args)
    e = P.estimate(u2, G["Lx"], G["Ly"], args.sigma_dir, None, G["taper"])
    A = e["A"]
    farm = G["win"] & (np.abs(G["Y"] - y_row) > args.far_from * G["L"])
    A_far = float(np.median(A[farm])) if farm.sum() > 50 else float(np.median(A[G["win"]]))
    phi, _ = P.smooth_phase(e, x, y, G, args)
    gy, gx = np.gradient(phi, G["dy"], G["dx"])
    jr, a, b = F.line_rows(y, G, args)
    allc = np.ones(len(x), bool)
    fphi = F.fold_line(gx, gy, a, b, allc)
    fku = F.fold_line(e["k1"], e["k2"], a, b, allc)
    rk2 = (e["k1"] - gx) ** 2 + (e["k2"] - gy) ** 2
    kk2 = e["k1"] ** 2 + e["k2"] ** 2
    cu = P.cells(u2, x, y, G, args)
    cc = P.cells(np.cos(e["theta"]), x, y, G, args)
    return dict(G=G, e=e, A=A, A_far=A_far, jr=jr, fphi=fphi, fku=fku, rk2=rk2, kk2=kk2, cu=cu, cc=cc)


def core_record(D, x, y, xc, args):
    G, e, L = D["G"], D["e"], D["G"]["L"]
    y_row = G["y_row"]
    r = args.loop_L * L
    c, cl = loop_grid(e["k1"], e["k2"], x, y, np.array([xc]), np.array([y_row]), r, args.nang)
    cp, fl = P.loop_plain(e["k1"], e["k2"], x, y, np.array([xc]), np.array([y_row]), r, args.nang)
    W = (np.abs(G["X"] - xc) < args.win_x_L * L) & (np.abs(G["Y"] - y_row) < args.win_y_L * L)
    A, A_far = D["A"], D["A_far"]
    ic = int(np.argmin(np.abs(x - xc)))
    near = np.abs(x - xc) < args.win_x_L * L
    il, ir = int(np.argmin(np.abs(x - (xc - r)))), int(np.argmin(np.abs(x - (xc + r))))
    rec = dict(G_carried=float(abs(c[0])), closed=int(bool(cl[0])), G_plain=float(abs(cp[0])), flips=float(fl[0]),
               A_min_loc=float(A[W].min() / A_far), A_centre=float(A[D["jr"], ic] / A_far),
               A_low_area=float(np.sum(A[W] < args.low_frac * A_far) * G["dx"] * G["dy"] / L ** 2),
               fold_loc_phi=float(np.mean(D["fphi"][near])), fold_loc_ku=float(np.mean(D["fku"][near])),
               cross_phi=float(int(D["fphi"][il]) + int(D["fphi"][ir])),
               cross_ku=float(int(D["fku"][il]) + int(D["fku"][ir])),
               curl_loc=float(np.sum(D["rk2"][W]) / max(np.sum(D["kk2"][W]), 1e-30)))
    px, py, pq = D["cu"]
    m = np.abs(px - xc) < args.win_x_L * L
    V = m & (pq < 0); X = m & (pq > 0)
    rec["nV_u"] = float(V.sum()); rec["nX_u"] = float(X.sum())
    rec["V_off_L"] = float(np.median(np.abs(py[V] - y_row)) / L) if V.any() else float("nan")
    if V.any() and X.any():
        d = np.hypot(px[V][:, None] - px[X][None, :], py[V][:, None] - py[X][None, :])
        rec["VX_dist_L"] = float(d.min() / L)
    else:
        rec["VX_dist_L"] = float("nan")
    qx, qy, qq = D["cc"]
    rec["n_cells_cos"] = float(np.sum(np.abs(qx - xc) < args.win_x_L * L))
    return rec


def jump_scores(series, i_e, min_rest=5):
    """series: dict key -> array over frames; i_e: event frame index (> 0); no score with fewer than min_rest
    other frame-to-frame changes"""
    out = {}
    for k, _, _ in QUANT:
        q = np.asarray(series[k], float)
        before, after = q[i_e - 1], q[i_e]
        d = np.abs(np.diff(q))
        d_rest = np.delete(d, i_e - 1)
        d_rest = d_rest[np.isfinite(d_rest)]
        if len(d_rest) < min_rest:                         # too short a history to say what a typical change is
            out[k + "_before"] = float(q[i_e - 1]); out[k + "_after"] = float(q[i_e]); out[k + "_score"] = float("nan")
            continue
        rng = np.nanmax(q) - np.nanmin(q) if np.isfinite(q).any() else 0.0
        floor = 1e-9 + 0.01 * (rng if np.isfinite(rng) else 0.0)
        base = max(float(np.median(d_rest)) if len(d_rest) else 0.0, floor)
        de = abs(after - before) if np.isfinite(after) and np.isfinite(before) else float("nan")
        out[k + "_before"] = float(before); out[k + "_after"] = float(after)
        out[k + "_score"] = float(min(de / base, 100.0)) if np.isfinite(de) else float("nan")
    return out


# ---------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------

def fig_aligned(ts, S, evs, title, path, args, tlabel="t"):
    keys = ["A_min_loc", "A_centre", "fold_loc_phi", "cross_phi", "nV_u", "VX_dist_L", "n_cells_cos", "curl_loc",
            "G_plain", "G_carried"]
    lab = {k: l for k, l, _ in QUANT}
    lab["G_carried"] = "circulation (turns): solid where the loop closes, dotted where it does not"
    grp = {k: g for k, _, g in QUANT}
    fig, axs = plt.subplots(len(keys), 1, figsize=(7.2, 1.25 * len(keys) + 1.2), sharex=True, constrained_layout=True)
    cm = plt.get_cmap("viridis")
    nc = len(evs)
    for j, ev in enumerate(evs):
        if not np.isfinite(ev["t_event"]):
            continue
        tt = ts - ev["t_event"]
        col = cm(0.1 + 0.8 * j / max(nc - 1, 1))
        for ax, k in zip(axs, keys):
            if k == "G_carried":
                g = S[j][k]; cl = S[j]["closed"] > 0
                ax.plot(tt, np.where(cl, g, np.nan), "-", color=col, lw=1.1)
                ax.plot(tt, np.where(~cl, g, np.nan), ":", color=col, lw=0.9, alpha=0.7)
            else:
                ax.plot(tt, S[j][k], "-", color=col, lw=1.0, alpha=0.9)
    for ax, k in zip(axs, keys):
        ax.axvline(0, color="0.35", lw=0.8, ls="--")
        ax.set_title(lab[k], fontsize=7.5, loc="left", color=GROUP_COLOR[grp[k]], pad=2)
        ax.tick_params(labelsize=7)
    axs[-1].set_xlabel("%s from the core's own event (dashed line: its circulation first reaches one turn on a closed loop)"
                       % tlabel, fontsize=7.5)
    fig.suptitle(title + "\none curve per core", fontsize=8.5)
    fig.savefig(path, dpi=160); plt.close(fig)


def fig_scores(table, path):
    movies = sorted(set(r["stem"] for r in table), key=lambda s: [r["mu"] for r in table if r["stem"] == s][0])
    keys = [k for k, _, _ in QUANT]
    M = np.full((len(keys), len(movies)), np.nan)
    N = np.zeros(len(movies), int)
    for j, s in enumerate(movies):
        rows = [r for r in table if r["stem"] == s and np.isfinite(r["t_event"])]
        N[j] = len(rows)
        for i, k in enumerate(keys):
            v = [r[k + "_score"] for r in rows if np.isfinite(r.get(k + "_score", np.nan))]
            if v:
                M[i, j] = np.median(v)
    fig, ax = plt.subplots(figsize=(1.9 + 1.15 * len(movies), 0.42 * len(keys) + 1.8), constrained_layout=True)
    im = ax.imshow(np.log10(np.clip(M, 0.1, 100)), cmap="magma_r", vmin=-1, vmax=2, aspect="auto")
    for i in range(len(keys)):
        for j in range(len(movies)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, ("%.0f" % M[i, j]) if M[i, j] >= 10 else ("%.1f" % M[i, j]), ha="center", va="center",
                        fontsize=7, color="w" if M[i, j] > 8 else "k")
    lab = [l for _, l, _ in QUANT]
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(lab, fontsize=7)
    for t, (_, _, g) in zip(ax.get_yticklabels(), QUANT):
        t.set_color(GROUP_COLOR[g])
    mu = [[r["mu"] for r in table if r["stem"] == s][0] for s in movies]
    dts = [[r["dt"] for r in table if r["stem"] == s][0] for s in movies]
    ax.set_xticks(range(len(movies)))
    ax.set_xticklabels(["mu %.2f\ndt %.3g\n%d cores" % (m, d, n) for m, d, n in zip(mu, dts, N)], fontsize=7)
    cb = plt.colorbar(im, ax=ax, shrink=0.7)
    cb.set_ticks([-1, 0, 1, 2]); cb.set_ticklabels(["0.1", "1", "10", "100"])
    cb.set_label("median jump score at the core's event\n(change in the event frame / typical change per frame)", fontsize=7)
    ax.set_title("What changes, core by core,\nin the frame where the circulation reaches one turn", fontsize=9)
    fig.savefig(path, dpi=160); fig.savefig(str(path).replace(".png", ".pdf")); plt.close(fig)


def fig_snapshots(frames, i_e, xc, y_row, mu, args, title, path):
    idx = [i for i in [i_e - 2, i_e - 1, i_e, i_e + 1] if 0 <= i < len(frames)]
    fig, axs = plt.subplots(2, len(idx), figsize=(3.6 * len(idx), 6.2), constrained_layout=True, squeeze=False)
    for col, i in enumerate(idx):
        t, u2, x, y = frames[i]
        D = frame_fields(u2, x, y, y_row, mu, args)
        rec = core_record(D, x, y, xc, args)
        L = D["G"]["L"]
        ix = np.abs(x - xc) <= args.snap_half_L * L
        jy = np.abs(y - y_row) <= args.snap_half_L * L
        ext = [(x[ix][0] - xc) / L, (x[ix][-1] - xc) / L, (y[jy][0] - y_row) / L, (y[jy][-1] - y_row) / L]
        for row, (img, cmap, vmin, vmax) in enumerate([(u2, "Greys", -np.max(np.abs(u2)), np.max(np.abs(u2))),
                                                       (D["A"] / D["A_far"], "viridis", 0.3, 1.15)]):
            ax = axs[row, col]
            im = ax.imshow(img[np.ix_(jy, ix)], origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            px, py, pq = D["cu"]
            ax.plot((px[pq > 0] - xc) / L, (py[pq > 0] - y_row) / L, "^", ms=6, mfc="#d62728", mec="w", mew=0.7, ls="none")
            ax.plot((px[pq < 0] - xc) / L, (py[pq < 0] - y_row) / L, "v", ms=6, mfc="#1f77b4", mec="w", mew=0.7, ls="none")
            ax.add_patch(Circle((0, 0), args.loop_L, fill=False, ec="#f39c12", lw=1.4))
            fx = (x[D["fphi"]] - xc) / L
            fx = fx[np.abs(fx) <= args.snap_half_L]
            ax.plot(fx, np.zeros_like(fx), "s", ms=1.6, color="#c0392b", ls="none")
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title("t = %.4g%s\ncirculation %.2f turn (%s), plain %.2f" % (
                    t, "  (event)" if i == i_e else "", rec["G_carried"], "closed" if rec["closed"] else "open",
                    rec["G_plain"]), fontsize=8)
            else:
                ax.set_title("A / A_far: min %.3f near the core" % rec["A_min_loc"], fontsize=8)
            if col == 0:
                ax.set_ylabel("y / L", fontsize=8)
            if row == 1:
                ax.set_xlabel("(x - x_core) / L", fontsize=8)
        if col == len(idx) - 1:
            plt.colorbar(im, ax=axs[1, col], shrink=0.8)
    fig.suptitle(title + "\norange circle: the 0.5 L loop; red squares on the line: folded stretch (grad phi_LS); "
                 "red up-triangles X cells, blue down-triangles V cells", fontsize=8.5)
    fig.savefig(path, dpi=160); plt.close(fig)


# ---------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------

def run_movie(frames, mu, y_row, stem, title, outdir, args, tlabel="t", centres=None):
    outdir.mkdir(parents=True, exist_ok=True)
    tL, uL, xL, yL = frames[-1]
    if centres is None:
        G0 = P.geometry(uL, xL, yL, y_row, mu, args)
        eL = P.estimate(uL, G0["Lx"], G0["Ly"], args.sigma_dir, None, G0["taper"])
        xd, yd, _, _, _ = P.find_dip_patches(eL["A"], xL, yL, G0["win"], args.core_frac, 0.5 * G0["L"])
        centres = F.centres_from(xd, G0["L"], G0["xm"], G0["Lx"])
    ts = np.array([f[0] for f in frames])
    S = [dict((k, []) for k, _, _ in QUANT + [("closed", "", "")]) for _ in centres]
    rows = []
    for i, (t, u2, x, y) in enumerate(frames):
        D = frame_fields(u2, x, y, y_row, mu, args)
        for j, xc in enumerate(centres):
            rec = core_record(D, x, y, xc, args)
            for k in S[j]:
                S[j][k].append(rec[k])
            rows.append(dict(t=float(t), frame=i, core=j, x_core_L=float((xc - D["G"]["xm"]) / D["G"]["L"]), **rec))
    for j in range(len(centres)):
        for k in S[j]:
            S[j][k] = np.asarray(S[j][k], float)
    P.write_rows(rows, outdir / "core_series")
    dt = float(ts[1] - ts[0]) if len(ts) > 1 else float("nan")
    evs = []
    for j, xc in enumerate(centres):
        on = (S[j]["closed"] > 0) & (S[j]["G_carried"] >= args.turn_hi)
        hit = np.where(on)[0]
        ev = dict(stem=stem, mu=mu, dt=dt, core=j, x_core=float(xc), t_event=float("nan"), event_frame=-1, falls_back=0)
        if len(hit) and hit[0] > 0:
            i_e = int(hit[0])
            ev.update(t_event=float(ts[i_e]), event_frame=i_e,
                      falls_back=int(np.any(S[j]["G_carried"][i_e:] < 0.5)))
            ev.update(jump_scores(S[j], i_e))
        elif len(hit) and hit[0] == 0:
            ev["note"] = "one turn already in the first frame"
        evs.append(ev)
    fig_aligned(ts, S, evs, title, outdir / "core_aligned.png", args, tlabel)
    ok = [e for e in evs if np.isfinite(e["t_event"])]
    if ok:
        pick = [e for e in ok if e["core"] == args.snap_core] or sorted(ok, key=lambda e: e["t_event"])[:1]
        e0 = pick[0]
        fig_snapshots(frames, e0["event_frame"], e0["x_core"], y_row, mu, args,
                      "%s, core %d (x = %.2f L)" % (title, e0["core"], (e0["x_core"] - frames[0][2].mean()) /
                                                     (np.pi / np.sqrt(1 - mu ** 2))), outdir / "core_snapshots.png")
    return evs


def ev_summary(evs):
    ok = [e for e in evs if np.isfinite(e["t_event"])]
    if not ok:
        return "no core reaches one turn"
    keys = [k for k, _, _ in QUANT]
    med = {}
    for k in keys:
        v = [e.get(k + "_score", np.nan) for e in ok]
        v = [a for a in v if np.isfinite(a)]
        med[k] = float(np.median(v)) if v else float("nan")
    top = sorted([k for k in keys if k != "G_carried"], key=lambda k: -med[k])[:4]
    return ("%d of %d cores reach one turn (t %.4g to %.4g; %d fall back); median jump scores: amplitude min %.2g, "
            "A at core %.2g; largest besides the circulation: %s" % (
                len(ok), len(evs), min(e["t_event"] for e in ok), max(e["t_event"] for e in ok),
                sum(e["falls_back"] for e in ok), med["A_min_loc"], med["A_centre"],
                ", ".join("%s %.2g" % (k, med[k]) for k in top)))


def main(args):
    t0 = time.time()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    lines = [VERSION, "args: " + json.dumps(vars(args)), ""]
    table = []
    if args.control:
        s_vals = np.arange(args.ctl_s_from, args.ctl_s_to + 1e-9, args.ctl_ds)
        frames = []
        for s in s_vals:
            u2, x, y = F.pair_movie_frame(args.ctl_mu, float(s), args)
            frames.append((float(s), u2, x, y))
        x0, _, L0, _ = P.control_grid(args.ctl_mu, args)
        base = np.arange(-int(args.ctl_box_x_L), int(args.ctl_box_x_L) + 1) * 2 * L0
        zx = np.sort(base + 0.5 * args.ctl_s_to * L0)
        zx = zx[np.abs(zx - x0.mean()) < 0.35 * args.ctl_box_x_L * L0]
        evs = run_movie(frames, args.ctl_mu, 0.0, "control_pair", "planted vortex pair (pipeline check; time = separation s / L)",
                        out / "control", args, tlabel="s / L", centres=zx)
        P.write_rows(evs, out / "control" / "core_control_events")
        msg = "CONTROL: " + ev_summary(evs); print(msg); lines.append(msg)
    files = F.select_files(args.npz, args.mus) if args.npz else []
    excl = [s for s in args.exclude.split(",") if s]
    files = [f for f in files if not any(s in Path(f).name for s in excl)]
    if args.npz and not files:
        msg = "LOUD: no movie matched --npz %s --mus %s" % (args.npz, args.mus); print(msg); lines.append(msg)
    for p in files:
        t1 = time.time()
        u, x, y, t, md = F.movie(p)
        mu = float(md.get("mu", F.mu_of(p)))
        jr = int(md.get("gb_row_in_crop", u.shape[0] // 2)); y_row = float(y[jr])
        nt = u.shape[-1]
        frames = [(float(t[min(i, t.size - 1)]), np.asarray(u[..., i], float), x, y) for i in range(nt)]
        del u
        stem = Path(p).stem
        evs = run_movie(frames, mu, y_row, stem, "%s   mu %.3f (%.0f deg)" % (stem, mu, np.degrees(np.arcsin(mu))),
                        out / stem, args)
        table += evs
        msg = "%s mu %.3f: %s  [%.0f s]" % (stem, mu, ev_summary(evs), time.time() - t1); print(msg); lines.append(msg)
    if table:
        P.write_rows(table, out / "core_events")
        if any(np.isfinite(r["t_event"]) for r in table):
            fig_scores(table, out / "core_jump_scores.png")
    lines += ["", "done in %.0f s" % (time.time() - t0)]
    (out / "core_notes.txt").write_text("\n".join(lines) + "\n")
    print("done in %.0f s; outputs in %s" % (time.time() - t0, out))
    return 0


class _Args:
    """defaults; mirrored by the CLI below. Keys shared with phase_vs_amplitude_charge_v11 keep its values."""
    npz = ""                       # comma-separated movie files and/or folders
    mus = ""                       # mu values to keep ("" keeps every file)
    exclude = "last10,f4-84"       # skip files whose name contains any of these (the excerpt files)
    outdir = str(Path("results") / "core_track_time_series")
    turn_hi = 0.9                  # one turn of circulation: a closed loop at or above this
    win_x_L = 0.5                  # core window half-width along the line, L
    win_y_L = 0.75                 # core window half-height, L
    low_frac = 0.7                 # A_low_area counts A below this fraction of A_far
    snap_core = -1                 # core for the snapshots (-1: the first to reach one turn)
    snap_half_L = 0.9              # snapshot half-width, L
    control = 0
    ctl_mu = 0.88
    ctl_s_from = -0.4
    ctl_s_to = 1.0
    ctl_ds = 0.05
    # shared with phase_vs_amplitude_charge_v11 / fold_time_series (same values)
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
    ls_mode = "plain"
    seam_dilate = 1
    nang = 181
    on_line_L = 0.1
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
