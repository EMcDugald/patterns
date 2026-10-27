# pipelines/slice_last_frames.py
"""Slice the last N frames of the six SH last-frame files for the research
thread (self_dual_discovery data route), keeping x, y, t and metadata_json.
Run from pipelines/:  python slice_last_frames.py            (last 3 frames)
                      python slice_last_frames.py --n_last 5
Output: results/1010_lastframes/<stem>_last<N>.npz"""
import argparse, os
import numpy as np

# D1 = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/1004_tall_coarse_mu79_81_83_t_in_0_25_dt_p2_125frames_Rp5/raw"
# D2 = "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/0913_tall_coarse_mu85_90_95_t_in_0_25_dt_p2_125frames/raw"
# FILES = [D1 + "/sh_pgb_zigzag_mu0.790_T25_N125_nx512_Ny768_lower.npz",
#          D1 + "/sh_pgb_zigzag_mu0.810_T25_N125_nx512_Ny768_lower.npz",
#          D1 + "/sh_pgb_zigzag_mu0.830_T25_N125_nx512_Ny768_lower.npz",
#          D2 + "/sh_pgb_zigzag_mu0.850_T25_N125_nx512_Ny768_lower_Rp5.npz",
#          D2 + "/sh_pgb_zigzag_mu0.900_T25_N125_nx512_Ny768_lower_Rp5.npz",
#          D2 + "/sh_pgb_zigzag_mu0.950_T25_N125_nx512_Ny768_lower_Rp5.npz"]

import glob
FILES = sorted(glob.glob("/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_zigzag/1009_tall_coarse_mu65_to_92_t_in_0_25_dt_p2_125frames_Rp5/raw/*.npz"))

p = argparse.ArgumentParser()
p.add_argument("--n_last", type=int, default=3)
p.add_argument("--outdir", default="results/1010_lastframes")
a = p.parse_args()
os.makedirs(a.outdir, exist_ok=True)
for f in FILES:
    d = np.load(f, allow_pickle=True)
    u = d["u"]
    tkey = "tt" if "tt" in d.files else "t"
    time_last = u.ndim == 3 and u.shape[-1] < u.shape[0]      # [ny, nx, nt] vs [nt, ny, nx]
    u_s = u[..., -a.n_last:] if time_last else u[-a.n_last:]
    out = dict(u=u_s.astype(np.float32), x=d["x"], y=d["y"])
    out[tkey] = np.asarray(d[tkey])[-a.n_last:]
    for k in d.files:
        if k in ("u", "x", "y", tkey):
            continue
        arr = d[k]
        if k == "metadata_json" or arr.size <= 1024:  # metadata and small arrays only
            out[k] = arr
        elif arr.ndim == 3 and arr.shape == u.shape:  # a per-frame field such as e: slice it like u
            out[k] = (arr[..., -a.n_last:] if time_last else arr[-a.n_last:]).astype(np.float32)
        else:
            print("  dropping", k, arr.shape)                                      # metadata_json and any other small arrays
    stem = os.path.splitext(os.path.basename(f))[0]
    path = os.path.join(a.outdir, "%s_last%d.npz" % (stem, a.n_last))
    np.savez_compressed(path, **out)
    print(stem, "u", u.shape, "->", u_s.shape, "t", out[tkey], "->", path)