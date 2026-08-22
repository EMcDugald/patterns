# pipelines/t1_director_test_adaptive_v2.py
"""
Adds the adaptive-width Riesz director to your director-test harness as a
scored source, so it is compared against exact / uhu / riesz / riesz_auto on the
analytic knee and PN with the same truth-free metrics, in the same run.

No file is patched. This registers ONE new source, "riesz_adaptive", into the
shared t1_common.SOURCES dict at import time, then drives t1_director_test.main
unchanged. Because t1_director_test imports the same SOURCES object, the new
source is visible to run_one and to the --sources choices without any edit.

The adaptive source is the loop-persistence spatial width modulator:
  small width (0.4) on clean stretches, large width (pi) only where the local
  loop-persistence says a defect is present, blended per point on the
  doubled-angle vectors. On the knee the weight is 0 everywhere, so it should
  score identically to riesz at the small width; on PN it should track
  riesz_auto's large-width behaviour at the cores.

Run:
  python t1_director_test_adaptive_v2.py
Edit the Args namespace below (fields, mu, sources) the way you would the base
harness's _Args block.
"""

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from op_extract import t1_common as t1c
from op_extract.loop_persistence import adaptive_director


def director_riesz_adaptive(fld, sigma_small=0.4, sigma_large=float(np.pi),
                            loop_half_sizes=(1, 2, 3, 4, 6, 8),
                            persist_smooth_px=1.5, persist_gamma=1.0, **kw):
    """SOURCES-compatible director: (fld, ...) -> (d1, d2). Extra kwargs that
    run_one passes (sigma, ramp, sigma_dir) are absorbed and ignored, since the
    adaptive blend fixes its own two widths rather than using a swept one."""
    ad = adaptive_director(
        fld["u"], fld["Lx"], fld["Ly"], t1c.riesz_director,
        sigma_small=sigma_small, sigma_large=sigma_large,
        half_sizes=tuple(loop_half_sizes),
        smooth_px=persist_smooth_px, gamma=persist_gamma,
    )
    return ad["d1"], ad["d2"]


# register into the SHARED dicts (mutation, not rebinding, so t1_director_test
# and run_one see it through their own imported reference)
t1c.SOURCES["riesz_adaptive"] = director_riesz_adaptive
t1c.SOURCE_COLORS.setdefault("riesz_adaptive", "C4")

# import AFTER registration so the harness's --sources choices include it
import t1_director_test as tdt


if __name__ == "__main__":
    # Namespace built from the harness's own defaults, with the adaptive source
    # added to the comparison. Edit like the base _Args block.
    import argparse
    args = argparse.Namespace(**{k: getattr(tdt._Args, k)
                                 for k in dir(tdt._Args)
                                 if not k.startswith("_")})
    args.sources = ["exact", "riesz", "riesz_auto", "riesz_adaptive"]
    args.outdir = "out_t1_adaptive"  # <- output directory
    # a first comparison is cheaper without the extra figure passes; flip back on as needed
    args.surfaces = True
    tdt.main(args)
