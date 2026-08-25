#!/usr/bin/env python3
"""
slice_old_raw_last_frame.py

Slice the LAST frame from an old-format raw zigzag npz (keys tt, xx,
yy, uu, ee with frame-FIRST layout uu[(nt, ny, nx)]) and write a small
standalone npz in the new-ish layout the extraction comparison scripts
expect:

  x, y          coordinate axes (renamed from xx, yy)
  u             (ny, nx, 1)  last frame, trailing frame axis
  e             (ny, nx, 1)  last frame of the energy density
  tt            (1,)         time of the kept frame
  metadata_json {"mu": ..., "Rscale": ..., "gb_row_in_crop": ...}

The core row is detected as the argmax of the x-averaged energy
density of the last frame (the boundary carries the excess energy),
so no metadata beyond mu is needed from the old file. Rscale is not
stored in the old files; pass it on the CLI (defaults to 0.5, which
is what the old shallow-knee runs used).

Usage:
  python slice_old_raw_last_frame.py in.npz [Rscale] [out.npz]
"""
import sys, json
from pathlib import Path
import numpy as np

inp = Path(sys.argv[1])
Rscale = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
out = Path(sys.argv[3]) if len(sys.argv) > 3 else \
    inp.with_name(inp.stem + "_lastframe.npz")

d = np.load(inp, allow_pickle=True)
u_last = np.asarray(d["uu"][-1])          # frame-first in the old files
e_last = np.asarray(d["ee"][-1])
x, y = np.asarray(d["xx"]), np.asarray(d["yy"])
mu = float(np.asarray(d["mu"]))
tt_last = np.atleast_1d(np.asarray(d["tt"]))[-1:]

ebar = e_last.mean(axis=1)                # x-average -> e(y)
jc = int(np.argmax(ebar))                 # boundary row carries the excess

md = {"mu": mu, "Rscale": Rscale, "gb_row_in_crop": jc,
      "source": inp.name, "t_kept": float(tt_last[0])}

np.savez_compressed(
    out, x=x, y=y,
    u=u_last[..., None], e=e_last[..., None],
    tt=tt_last, metadata_json=json.dumps(md))

print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB); "
      f"mu={mu}, Rscale={Rscale}, core row jc={jc} "
      f"(y={y[jc]:.2f}), t={tt_last[0]}")
