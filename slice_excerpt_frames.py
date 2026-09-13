#!/usr/bin/env python3
"""
Write a small excerpt of a raw or OP .npz: a window of time frames with a
stride, cast to float32, optionally cropped in x.

Any array whose last axis matches the number of frames (u, e, A_mono, ...,
and tt/n_dislocations) is sliced with frames[start:stop:stride]; float64
frame arrays are cast to float32 (the estimator works at the 1e-3 level,
so nothing is lost); 1D coordinate axes (x, y) and scalars/metadata are
copied as-is. Frame-indexed metadata such as freeze frames is NOT
re-based, so treat the excerpt as a standalone clip. With --cols the x
axis is cropped to a centred band of that many columns (u, e, x and any
other array with an x axis of the full width are cropped together); the
metadata's gb_row_in_crop is unaffected since only x is cropped.

Usage:
  python slice_excerpt_frames.py <in.npz> [--start 4] [--stop 84] [--stride 2]
                                 [--cols 256] [--out out.npz]

Default (start 4, stop 84, stride 2) on a 126-frame dt = 0.025 file gives
40 frames at dt = 0.05 covering t = 0.1 to 2.075, about 10 MB for a
256 x 256 crop in float32. For a smaller file use --stride 4 or --cols 128.
"""
import argparse
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("inp")
p.add_argument("--start", type=int, default=4)
p.add_argument("--stop", type=int, default=84)
p.add_argument("--stride", type=int, default=2)
p.add_argument("--cols", type=int, default=None,
               help="crop x to this many centred columns (default: keep all)")
p.add_argument("--out", default=None)
a = p.parse_args()

inp = Path(a.inp)
out = Path(a.out) if a.out else inp.with_name(
    inp.stem + "_f%d-%d-s%d%s.npz" % (a.start, a.stop, a.stride,
                                       ("_c%d" % a.cols) if a.cols else ""))

d = np.load(inp, allow_pickle=True)

# frame count from tt if present, else the most common trailing dim
if "tt" in d.files:
    nt = len(np.atleast_1d(d["tt"]))
else:
    from collections import Counter
    nt = Counter(d[k].shape[-1] for k in d.files
                 if getattr(d[k], "ndim", 0) >= 1).most_common(1)[0][0]
sel = slice(a.start, min(a.stop, nt), a.stride)

# x crop
nx = len(d["x"]) if "x" in d.files else None
if a.cols and nx and a.cols < nx:
    c0 = (nx - a.cols) // 2
    csl = slice(c0, c0 + a.cols)
else:
    csl = None

saved, sliced, cropped = {}, [], []
for k in d.files:
    arr = d[k]
    if getattr(arr, "ndim", 0) >= 1 and arr.shape[-1] == nt and k not in ("x", "y"):
        arr = arr[..., sel]
        sliced.append(k)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
    if csl is not None and getattr(arr, "ndim", 0) >= 1:
        if k == "x":
            arr = arr[csl]
            cropped.append(k)
        elif arr.ndim >= 2 and arr.shape[1] == nx:      # [ny, nx, ...]
            arr = arr[:, csl]
            cropped.append(k)
    saved[k] = arr

np.savez_compressed(out, **saved)
mb = out.stat().st_size / 1e6
tt = np.atleast_1d(saved["tt"]) if "tt" in saved else None
print("nt=%d -> kept %d frames (%s)%s. wrote %s  (%.1f MB)"
      % (nt, len(range(*sel.indices(nt))),
         ("t = %.3f .. %.3f, dt %.4f" % (tt[0], tt[-1], np.median(np.diff(tt)))) if tt is not None and len(tt) > 1 else "no tt",
         (", x cropped to %d columns" % a.cols) if csl is not None else "", out, mb))
print("sliced keys:", sliced)
if cropped:
    print("cropped keys:", cropped)
