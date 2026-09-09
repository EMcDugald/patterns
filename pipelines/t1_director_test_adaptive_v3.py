# pipelines/t1_director_test_adaptive_v3.py
"""
Changes from t1_director_test_adaptive_v2:

  - All parameters now live in THIS file: one _Args block below holds both
    the harness overrides (mu, fields, sources, outdir, gate widths) and
    the adaptive-director knobs (its own sigma_small / sigma_large, loop
    half sizes, weight smoothing, gamma), with a CLI mirror. v2 read the
    harness knobs from t1_director_test._Args and fixed the adaptive knobs
    as function-signature defaults, so configuring a run meant editing two
    files, and the two large widths could silently disagree.
  - They DID disagree in the v2 report (t1_report.txt): riesz_auto gated
    to sigma_large = pi/2 while riesz_adaptive blended toward pi. The v3
    defaults are width-matched (both pi/2), so the PN comparison reads
    policy against policy rather than endpoint against endpoint.
  - Default sweep restricted to the production range mu = 0.85, 0.90,
    0.95.
  - sigma_det default 0.3, matching the production runner
    (run_riesz_zigzag_modes.CONFIG). The v2 report ran 0.4; both are
    validated clean on the knee in the production range, but the harness
    and the production runner should quote one width. Set --sigma-det 0.4
    to reproduce the v2 detection.
  - Optional second adaptive source "riesz_adaptive_wide" with extended
    loop half sizes (1, 2, 3, 4, 6, 8, 12, 16), off by default. At the
    mu = 0.90 PN spacing the persistence bumps of neighbouring cores can
    merge at these scales, so this variant shows the blend's limit toward
    the global gate. It is a mechanism illustration, not a competing
    policy.

Registers the adaptive source(s) into the shared t1_common.SOURCES dict,
then drives t1_director_test.main unchanged, exactly as v2 did.

Run:
  python t1_director_test_adaptive_v3.py
  python t1_director_test_adaptive_v3.py --sigma-large-adaptive 3.1416
  python t1_director_test_adaptive_v3.py --wide-variant --mu 0.90
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from op_extract import t1_common as t1c
from op_extract.loop_persistence import adaptive_director


# ---------------------------------------------------------------------
# all parameters for a run
# ---------------------------------------------------------------------

class _Args:
    # ---- harness sweep -------------------------------------------------
    outdir = "out_t1_adaptive_pio2"
    fields = ["knee", "pn"]
    mu = [0.85, 0.90, 0.95]          # production range
    sources = ["exact", "riesz", "riesz_auto", "riesz_adaptive"]
    sigma_riesz = 1.0                # fixed width of the plain riesz source
    sigmas = [np.pi / 2]             # uHu averaging scale (unused unless
                                     # "uhu" is added to sources)

    # ---- the binary gate (riesz_auto) ----------------------------------
    sigma_det = 0.3                  # production detection width; v2 ran 0.4
    sigma_small = 0.4
    sigma_large_gate = float(np.pi/2)

    # ---- the persistence blend (riesz_adaptive) ------------------------
    # width-matched to the gate by default, so the PN columns compare the
    # policies at the same endpoints
    adaptive_sigma_small = 0.4
    sigma_large_adaptive = float(np.pi/2)
    loop_half_sizes = (1, 2, 3, 4, 6, 8)
    persist_smooth_px = 1.5
    persist_gamma = 1.0

    # ---- optional wide-loop variant ------------------------------------
    wide_variant = False
    wide_half_sizes = (1, 2, 3, 4, 6, 8, 12, 16)

    # ---- figures -------------------------------------------------------
    surfaces = True
    surface_sources = ["riesz_auto", "riesz_adaptive"]
    diagnose = True
    riesz_panels = True


def _make_adaptive_source(sigma_small, sigma_large, half_sizes,
                          smooth_px, gamma):
    """Build a SOURCES-compatible director closed over the given knobs.
    Extra kwargs that run_one passes (sigma, ramp, sigma_dir) are absorbed
    and ignored: the blend fixes its own two widths rather than using the
    swept one."""
    def director(fld, **kw):
        ad = adaptive_director(
            fld["u"], fld["Lx"], fld["Ly"], t1c.riesz_director,
            sigma_small=sigma_small, sigma_large=sigma_large,
            half_sizes=tuple(half_sizes), smooth_px=smooth_px, gamma=gamma,
        )
        return ad["d1"], ad["d2"]
    return director


def _register(args):
    """Mutate (never rebind) the shared dicts so t1_director_test.run_one
    sees the new sources through its own imported reference."""
    t1c.SOURCES["riesz_adaptive"] = _make_adaptive_source(
        args.adaptive_sigma_small, args.sigma_large_adaptive,
        args.loop_half_sizes, args.persist_smooth_px, args.persist_gamma)
    t1c.SOURCE_COLORS.setdefault("riesz_adaptive", "C4")
    if args.wide_variant:
        t1c.SOURCES["riesz_adaptive_wide"] = _make_adaptive_source(
            args.adaptive_sigma_small, args.sigma_large_adaptive,
            args.wide_half_sizes, args.persist_smooth_px,
            args.persist_gamma)
        t1c.SOURCE_COLORS.setdefault("riesz_adaptive_wide", "C5")
        if "riesz_adaptive_wide" not in args.sources:
            args.sources = list(args.sources) + ["riesz_adaptive_wide"]


def _harness_namespace(args):
    """Start from the harness's own defaults, override with this file's
    settings. Everything the harness needs and this file does not expose
    keeps its harness default."""
    import t1_director_test as tdt
    ns = argparse.Namespace(**{k: getattr(tdt._Args, k)
                               for k in dir(tdt._Args)
                               if not k.startswith("_")})
    ns.outdir = args.outdir
    ns.fields = list(args.fields)
    ns.mu = list(args.mu)
    ns.sources = list(args.sources)
    ns.sigma_riesz = float(args.sigma_riesz)
    ns.sigmas = list(args.sigmas)
    ns.sigma_det = float(args.sigma_det)
    ns.sigma_small = float(args.sigma_small)
    ns.sigma_large = float(args.sigma_large_gate)
    ns.surfaces = bool(args.surfaces)
    ns.surface_sources = list(args.surface_sources)
    ns.diagnose = bool(args.diagnose)
    ns.riesz_panels = bool(args.riesz_panels)
    return tdt, ns


def main(args):
    _register(args)
    tdt, ns = _harness_namespace(args)
    print(f"[adaptive_v3] gate: sigma_det={ns.sigma_det}, "
          f"small={ns.sigma_small}, large={ns.sigma_large}")
    print(f"[adaptive_v3] blend: small={args.adaptive_sigma_small}, "
          f"large={args.sigma_large_adaptive}, "
          f"half_sizes={tuple(args.loop_half_sizes)}, "
          f"smooth_px={args.persist_smooth_px}, "
          f"gamma={args.persist_gamma}")
    if args.wide_variant:
        print(f"[adaptive_v3] wide variant on: "
              f"half_sizes={tuple(args.wide_half_sizes)}")
    tdt.main(ns)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default=_Args.outdir)
    p.add_argument("--fields", nargs="+", default=_Args.fields,
                   choices=["knee", "pn"])
    p.add_argument("--mu", type=float, nargs="+", default=_Args.mu)
    p.add_argument("--sources", nargs="+", default=_Args.sources)
    p.add_argument("--sigma-riesz", type=float, default=_Args.sigma_riesz,
                   help="fixed width of the plain riesz source")
    p.add_argument("--sigma-det", type=float, default=_Args.sigma_det,
                   help="gate detection width (production 0.3; v2 ran 0.4)")
    p.add_argument("--sigma-small", type=float, default=_Args.sigma_small,
                   help="gate small width")
    p.add_argument("--sigma-large-gate", type=float,
                   default=_Args.sigma_large_gate,
                   help="gate large width (riesz_auto on charged fields)")
    p.add_argument("--adaptive-sigma-small", type=float,
                   default=_Args.adaptive_sigma_small,
                   help="blend small width")
    p.add_argument("--sigma-large-adaptive", type=float,
                   default=_Args.sigma_large_adaptive,
                   help="blend large width; default matches the gate")
    p.add_argument("--half-sizes", type=int, nargs="+",
                   dest="loop_half_sizes", default=list(_Args.loop_half_sizes),
                   help="loop-persistence half sizes")
    p.add_argument("--smooth-px", type=float, dest="persist_smooth_px",
                   default=_Args.persist_smooth_px)
    p.add_argument("--gamma", type=float, dest="persist_gamma",
                   default=_Args.persist_gamma)
    p.add_argument("--wide-variant", action="store_true",
                   default=_Args.wide_variant,
                   help="also score riesz_adaptive_wide (extended loop "
                        "half sizes; limit-behaviour illustration)")
    p.add_argument("--no-surfaces", dest="surfaces", action="store_false",
                   default=_Args.surfaces)
    p.add_argument("--no-diagnose", dest="diagnose", action="store_false",
                   default=_Args.diagnose)
    cli = p.parse_args()
    cli.sigmas = _Args.sigmas
    cli.surface_sources = _Args.surface_sources
    cli.riesz_panels = _Args.riesz_panels
    cli.wide_half_sizes = _Args.wide_half_sizes
    cli.sigma_large_gate = getattr(cli, "sigma_large_gate",
                                   _Args.sigma_large_gate)
    main(cli)
