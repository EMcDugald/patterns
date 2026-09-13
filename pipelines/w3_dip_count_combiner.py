# pipelines/w3_dip_count_combiner.py
"""
Stack two dip-count figures (defect_gallery_dip_count.py outputs) into
the single Section 5.5 figure of Writeup 3. No computation; PNGs only.

Inputs (edit _Args or pass on the CLI, in order):
  top     dip_count_saddle_lse_mu0.85.png   (the clean chain)
  bottom  dip_count_dislocation.png         (the isolated SH dislocation)
Output:
  figs/w3/w3_dip_count.png
"""

import sys
from pathlib import Path

from PIL import Image

_HERE = Path(__file__).resolve().parent


class _Args:
    top = _HERE / "out_sh_defect_gallery_1" / "dip_count_saddle_lse_mu0.85.png"
    bottom = _HERE / "out_sh_defect_gallery_1" / "dip_count_dislocation.png"
    out = _HERE.parent / "pgb_paper" / "figs" / "w3" / "w3_dip_count.png"


def combine(top, bottom, out):
    a = Image.open(top).convert("RGB")
    b = Image.open(bottom).convert("RGB")
    w = max(a.width, b.width)
    if b.width != w:
        b = b.resize((w, int(b.height * w / b.width)))
    if a.width != w:
        a = a.resize((w, int(a.height * w / a.width)))
    canvas = Image.new("RGB", (w, a.height + b.height), "white")
    canvas.paste(a, (0, 0))
    canvas.paste(b, (0, a.height))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print("wrote", out)


if __name__ == "__main__":
    args = sys.argv[1:]
    top = Path(args[0]) if len(args) > 0 else _Args.top
    bottom = Path(args[1]) if len(args) > 1 else _Args.bottom
    out = Path(args[2]) if len(args) > 2 else _Args.out
    combine(top, bottom, out)
