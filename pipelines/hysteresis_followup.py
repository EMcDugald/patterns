"""
hysteresis_followup.py  v1.0  (4 Nov 2026)   the coexistence zone near mu 0.81 to 0.82, run longer and finer

Plain terms. The ramp of 2 Nov (dislocation_ramp_down.py) found no hysteresis except one step at the intermediate to
dislocation boundary: coming down, mu 0.81 kept one wound dip at t = 200 (after reading unwound at t = 100), while
going up the first winding is at 0.82, itself mixed (one of three dips wound). This script reruns that zone on both
branches to t = 400 and adds the half steps 0.805 and 0.815, so the survey can say whether the two branches converge
(no hysteresis, a coexistence zone that settles) or stay apart (a window narrower than 0.01).

  up branch    every target relaxed from a knee (the large sweep's initial condition), t = 0 to tmax.
  down branch  every target remapped from the FINAL full field of a run of the 2 Nov ramp (its seeds/ folder) by the
               ramp's own remap, then relaxed t = 0 to tmax. Defaults: 0.82 from 0.83 (the ramp's own step, now run
               twice as long), 0.815 and 0.81 from 0.82, 0.805 from 0.81.

Every run starts at t = 0 from its initial condition, so the first 200 time units of 0.82 and 0.81 down repeat the
2 Nov runs exactly (same IC, same solver); the t = 200 frame is the check (compare it with the ramp's final frame).

Outputs (under --outdir, anchored to this script's folder when relative; default results/hysteresis_followup/):
  up/raw/<stem>.npz, down/raw/<stem>.npz   the runner's format, cropped about the lower boundary, so
                        large_sweep_charge_circ_v13.py reads them unchanged (survey each branch separately).
  down/seeds/, up/seeds/  full final fields (float32), for any later continuation.
  figures/<stem>_ic.png   initial crop beside the final one.
  report.txt, config.json; per down target the max |difference| between this run's frame at t = 200 and the 2 Nov
                        ramp's final crop at the same angle when that run exists (expected below 1e-6: the same IC up to
                        float32 rounding, since the ramp seeded from the field in memory and this from the saved
                        float32 copy; 7e-10 in the sandbox).

Imports dislocation_ramp_down.py (same folder) for the geometry, the remap and the run, and sh_sims unchanged.
Python 3.8 compatible. Measures nothing itself: every measure is the survey's.

Changelog
  v1.0  (4 Nov 2026) first version. Sandbox check on a reduced grid (Nx 128, 6 periods, t 30; not of record):
        up and down runs written in the runner's format, repeat check 7.4e-10 against the ramp's own step, a missing
        source field flagged LOUD and skipped.
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dislocation_ramp_down as R                                                  # noqa: E402

VERSION = "hysteresis_followup v1.0"


def _abs(p):
    p = Path(p)
    return p if p.is_absolute() else HERE / p


def parse_down(spec):
    """'0.82:0.83,0.815:0.82' -> [(0.82, 0.83), (0.815, 0.82)]"""
    out = []
    for item in [s for s in spec.split(",") if s.strip()]:
        t, s = item.split(":")
        out.append((float(t), float(s)))
    return out


def find_seed(seed_dir, mu):
    """the ramp's final full field at mu (ramp tag preferred, then the knee seed)"""
    hits = sorted(glob.glob(str(seed_dir / ("mu%.3f_rampdown_from*_full.npz" % mu))))
    if not hits:
        hits = sorted(glob.glob(str(seed_dir / ("mu%.3f_*_full.npz" % mu))))
    return Path(hits[0]) if hits else None


def find_ramp_raw(ramp_raw, mu):
    hits = sorted(glob.glob(str(ramp_raw / ("sh_pgb_zigzag_mu%.3f_*rampdown*.npz" % mu))))
    return Path(hits[0]) if hits else None


def one(branch, mu, u0, g, label, extra, a, out, lines):
    raw, seeds = out / branch / "raw", out / branch / "seeds"
    tag = extra.pop("_tag")
    stem = R.stem_for(mu, a, tag)
    if (raw / (stem + ".npz")).exists() and not a.overwrite:
        lines.append("%s mu %.3f: exists, skipped (%s)" % (branch, mu, stem)); print(lines[-1]); return None
    t0 = time.time()
    res, u_end = R.run_one(u0, g, mu, a, a.tmax, a.nsave, label, extra)
    R.save_run(res, raw / (stem + ".npz"))
    np.savez_compressed(seeds / ("mu%.3f_%s_full.npz" % (mu, tag)), u=u_end.astype(np.float32), mu=mu)
    R.fig_ic(res, out / "figures" / (stem + "_ic.png"), mu, "%s branch (%s)" % (branch, label))
    lines.append("%s mu %.4f (%.2f deg): %s; final u in [%.3f, %.3f]; %.0f s -> %s" % (
        branch, mu, np.degrees(np.arcsin(mu)), label, u_end.min(), u_end.max(), time.time() - t0, stem))
    print(lines[-1])
    return res


def main(a):
    out = _abs(a.outdir)
    for d in ("up/raw", "up/seeds", "down/raw", "down/seeds", "figures"):
        (out / d).mkdir(parents=True, exist_ok=True)
    ramp = _abs(a.ramp_dir)
    up = [float(v) for v in a.mus_up.split(",") if v.strip()]
    down = parse_down(a.down)
    with open(out / "config.json", "w") as fh:
        json.dump(dict(vars(a), version=VERSION), fh, indent=2, default=str)
    lines = [VERSION, "up (from knees): %s" % ", ".join("%.3f" % m for m in up),
             "down (target from source): %s" % ", ".join("%.3f from %.3f" % p for p in down),
             "tmax %.0f, nsave %d (frame every %.1f time units)" % (a.tmax, a.nsave, a.tmax / a.nsave), ""]
    flags = []
    T0 = time.time()
    for mu in up:
        g = R.geometry(mu, a)
        one("up", mu, g["u0"], g, "knee", dict(_tag="knee_long", seed_mu=None, seed_from=None, script=VERSION), a, out, lines)
    for mu, src in down:
        sf = find_seed(ramp / "seeds", src)
        if sf is None:
            flags.append("LOUD: no full field at mu %.3f in %s; down target %.3f skipped" % (src, ramp / "seeds", mu))
            print(flags[-1]); continue
        u_src = np.load(sf)["u"].astype(float)
        try:
            u0, g, seam = R.remap(u_src, src, mu, a, flags)
        except ValueError as exc:
            flags.append("LOUD: down %.3f: %s" % (mu, exc)); print(flags[-1]); continue
        res = one("down", mu, u0, g, "dislocations remapped from the ramp's final field at %.3f (%s)" % (src, sf.name),
                  dict(_tag="down_from%.3f_long" % src, seed_from=float(src), seed_file=sf.name,
                       remap_order=int(a.remap_order), script=VERSION), a, out, lines)
        # repeat check: when this is the ramp's own step, the frame at the ramp's tmax must match the ramp's final crop
        rr = find_ramp_raw(ramp / "raw", mu)
        if res is not None and rr is not None:
            d = np.load(rr, allow_pickle=True)
            meta = json.loads(str(d["metadata_json"]))
            if abs(float(meta.get("seed_from", -1)) - src) < 1e-9:
                t_old = float(d["tt"][-1]); j = int(np.argmin(np.abs(res["tt"] - t_old)))
                if abs(res["tt"][j] - t_old) < 1e-9 and res["u"].shape[:2] == d["u"].shape[:2]:
                    err = float(np.max(np.abs(res["u"][..., j] - d["u"][..., -1])))
                    lines.append("   repeat check at t = %.0f against %s: max |difference| %.2e" % (t_old, rr.name, err))
                    if err > 1e-6:
                        flags.append("LOUD: down %.3f differs from the 2 Nov run at t = %.0f by %.2e" % (mu, t_old, err))
                else:
                    lines.append("   repeat check skipped: no frame at t = %.0f or crops differ" % t_old)
                print(lines[-1])
    lines += ["", "flags:"] + (flags if flags else ["none"]) + ["", "elapsed %.0f s" % (time.time() - T0), "next:"]
    for b in ("up", "down"):
        lines.append("  python large_sweep_charge_circ_v13.py --raw %s --outdir %s --frame_early %d"
                     % (out / b / "raw", out / b / "survey", a.nsave // 2))
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-8:]))
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below. Grid and solver settings are the large sweep's."""
    outdir = str(Path("results") / "hysteresis_followup")
    ramp_dir = str(Path("results") / "dislocation_ramp_down")   # the 2 Nov ramp (its seeds/ and raw/)
    mus_up = "0.805,0.81,0.815,0.82"
    down = "0.82:0.83,0.815:0.82,0.81:0.82,0.805:0.81"     # target:source, source = a ramp angle with a saved full field
    tmax = 400.0
    nsave = 80                     # a frame every 5 time units, as in the ramp; frame 40 is t = 200
    h = 0.25
    Nx = 512
    n_periods = 18
    Ny_factor = 5.0
    Rscale = 0.5
    amp = 1.0
    y_centering = "node"
    which_gb = "lower"
    crop_Nx = 256
    crop_Ny = 256
    remap_order = 3
    overwrite = 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [k for k in vars(_Args) if not k.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v), default=v)
    sys.exit(main(p.parse_args()))
