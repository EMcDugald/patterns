# pipelines/mirror_npz.py
"""
Reflect a zigzag npz in x (or y) and write it as a new npz, so the
scripts of record (wsindy_theta_t, rcn_balance_ledger, ruler_w, ...)
run on the reflected field unchanged. Nothing else is touched.

Why (T1.6, the gauge test of the J term). In pgb_zigzag.py the upper
knee is built from phi = -k1 X +- k2 Yr and the lower from phi = k1 X
+- k2 Yl, so theta_upper(X, Y) = theta_lower(-X, Y): the upper boundary
is the x-mirror image of the lower one, and Swift-Hohenberg is
reflection symmetric, so the upper crop is the x-mirror of the lower
crop for all time (up to the roundoff-seeded symmetry breaking of a
staggered winding; the stitch and the periodic seam are 22 spacings
from either row and the far field is quiet, theta_t 1e-6). In the
gauge k1 > 0 that the estimator's scalar-sign rule imposes, the mirror
has k2 = -mu above the row: the inverted crease, div k < 0 on the row,
beta_A < 0, the opposite circulation. Under x -> -x followed by the
gauge flip k -> -k: theta_t, S = div(w k), div k, Lap div k, curl k,
k.grad A and A_dev div k change sign; w, J = det grad k, A_dev,
(curl k)^2 and chi_J do not. So for a covariant pipeline the reflected
file must read the same S pair, the same bending, the same R2, and J
with the OPPOSITE sign (and the ledger's beta_A with the opposite sign,
lambda and m_J unchanged). Any other reading is an orientation
dependence inside the instrument (Riesz / director sign conventions of
the estimator, the scalar-sign rule, find_cores), not physics.

The y-mirror is a null: the lower knee is even about its row
(theta = k1 x + F(y), F even) so the y-reflected field is the same
field and every reading must reproduce the original.

Arrays reflected: u, e (if present), theta_initial, cos_theta_initial
(if present), along axis 1 (x) or 0 (y). x and y are kept (only the
spacing and the extent enter the scripts; the origin is arbitrary).
metadata_json gains "mirror": "x"|"y" and which_gb gets the suffix
"_xmirror"|"_ymirror"; gb_row_in_crop is unchanged for the x-mirror and
recomputed (ny - 1 - jc) for the y-mirror.

Outputs: <outdir>/<stem>_<axis>mirror.npz  (default outdir results/mirror_npz/)

Changelog
---------
v1.0  (19 Sep) first version. Checked: the x-mirror of the x-mirror is
      the original array to the bit; keys and dtypes preserved.

Run:
  python mirror_npz.py --npz <file> [--axis x|y] [--outdir results/mirror_npz]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent


def mirror_npz(npz, axis="x", outdir=None):
    d = np.load(npz, allow_pickle=True)
    ax = 1 if axis == "x" else 0
    out = {}
    for k in d.files:
        v = d[k]
        if k in ("u", "e", "theta_initial", "cos_theta_initial") and v.ndim >= 2:
            out[k] = np.flip(v, axis=ax)
        else:
            out[k] = v
    md = json.loads(str(d["metadata_json"])) if "metadata_json" in d.files else {}
    md["mirror"] = axis
    md["mirror_source"] = Path(npz).name
    md["which_gb"] = str(md.get("which_gb", "")) + "_%smirror" % axis
    if axis == "y" and "gb_row_in_crop" in md:
        ny = out["u"].shape[0]
        md["gb_row_in_crop"] = int(ny - 1 - int(md["gb_row_in_crop"]))
    out["metadata_json"] = np.array(json.dumps(md))
    outdir = Path(outdir) if outdir else (_HERE / "results" / "mirror_npz")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / ("%s_%smirror.npz" % (Path(npz).stem, axis))
    np.savez(path, **out)
    # round trip check
    back = np.flip(out["u"], axis=ax)
    assert np.array_equal(back, d["u"]), "mirror round trip failed"
    print("mirror_npz v1.0  axis %s  %s -> %s" % (axis, Path(npz).name, path))
    print("  u %s  gb_row_in_crop %s  which_gb %s" % (out["u"].shape, md.get("gb_row_in_crop"), md["which_gb"]))
    return path


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    npz = ""
    axis = "x"                   # x: the upper-boundary image (gauge test); y: the null (same field)
    outdir = str(_HERE / "results" / "mirror_npz")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in dir(_Args) if not a.startswith("_")]:
        p.add_argument("--" + k, type=str, default=getattr(_Args, k))
    cli = p.parse_args()
    if not cli.npz:
        p.error("--npz is required")
    mirror_npz(cli.npz, cli.axis, cli.outdir)
    sys.exit(0)
