#!/usr/bin/env python3
"""
Write a small copy of an OP .npz keeping only the last N time frames.

Any array whose last axis matches the number of frames (u, A_mono, kmag_direct,
theta, theta_smooth, ..., and tt/n_dislocations) is sliced to the last N; 1D
coordinate axes (x, y) and scalars (mu, core_row) are copied as-is. Frame-indexed
metadata like freeze frames is NOT re-based, so treat the slice as a standalone
short clip, not as frame-aligned to the original.

Usage:
  python slice_last_frames.py <in.npz> [N] [out.npz]
"""
import sys
from pathlib import Path
import numpy as np

inp = Path(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
out = Path(sys.argv[3]) if len(sys.argv) > 3 else inp.with_name(inp.stem + f"_last{N}.npz")

d = np.load(inp, allow_pickle=True)

# infer frame count from tt if present, else the most common trailing dim
if "tt" in d.files:
    nt = len(np.atleast_1d(d["tt"]))
else:
    from collections import Counter
    nt = Counter(a.shape[-1] for a in (d[k] for k in d.files)
                 if getattr(d[k], "ndim", 0) >= 1).most_common(1)[0][0]

N = min(N, nt)
saved = {}
for k in d.files:
    a = d[k]
    if getattr(a, "ndim", 0) >= 1 and a.shape[-1] == nt and k not in ("x", "y"):
        saved[k] = a[..., -N:]
    else:
        saved[k] = a

np.savez_compressed(out, **saved)
mb = out.stat().st_size / 1e6
print(f"nt={nt} -> kept last {N}. wrote {out}  ({mb:.1f} MB)")
print("sliced keys:", [k for k in d.files
                       if getattr(d[k], "ndim", 0) >= 1
                       and d[k].shape[-1] == nt and k not in ("x", "y")])
