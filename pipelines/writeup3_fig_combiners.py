# pipelines/writeup3_fig_combiners.py
"""
Figure combiners for Writeup 3 (Riesz-based order-parameter extraction).

Stacks existing harness PNGs into the composite figures the writeup can
use, so no vector-graphics editor is needed. Pure PIL: no re-rendering,
the source panels are pasted at native resolution with a thin label
strip. Combiners provided:

  defect_pair   : knee and multi-dislocation defect-index figures
                  stacked vertically (charge-free knee above the
                  charge-pair row), one contrast figure for Sec. 4.
  profiles_pair : knee and multi-dislocation profile figures stacked,
                  the two halves of the twin criteria, for Sec. 6 if a
                  single combined figure is preferred over the two
                  separate ones currently placed.
  panels_pair   : the two riesz_panels construction figures stacked,
                  for Sec. 2/3 if a knee-plus-defected construction
                  figure is wanted.

All paths are anchored to this script's directory unless absolute.
Edit _Args or use the CLI mirror.

  python writeup3_fig_combiners.py --which defect_pair \
      --indir out_t1_adaptive_v3 --mu 0p900
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent


class _Args:
    indir = "out_t1_adaptive_v3"   # folder holding the harness PNGs
    outdir = "writeup3_figs"       # composites land here
    mu = "0p900"                   # filename mu tag
    sig = "1p571"                  # filename sigma tag
    which = ["panels_pair"]        # any of: defect_pair, profiles_pair,
                                   #         panels_pair
    label_px = 46                  # height of the label strip per panel
    pad_px = 12                    # white padding between panels


_RECIPES = {
    "defect_pair": (
        ("defects_knee_mu{mu}_sig{sig}.png", "knee: charge-free"),
        ("defects_pn_mu{mu}_sig{sig}.png",
         "multi-dislocation boundary: one charge pair per core"),
    ),
    "profiles_pair": (
        ("profiles_knee_mu{mu}_sig{sig}.png", "knee core row"),
        ("profiles_pn_mu{mu}_sig{sig}.png",
         "multi-dislocation row ladder"),
    ),
    "panels_pair": (
        ("riesz_panels_knee_mu{mu}_sig{sig}.png", "knee"),
        ("riesz_panels_pn_mu{mu}_sig{sig}.png",
         "multi-dislocation boundary"),
    ),
}


def _load(path):
    img = Image.open(path).convert("RGB")
    return img


def _label_strip(width, height, text):
    strip = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", int(height * 0.55))
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, height // 5), text, fill="black", font=font)
    return strip

def _stack(images_with_labels, label_px, pad_px):
    width = max(img.width for img, _ in images_with_labels)
    pieces = []
    for img, label in images_with_labels:
        if img.width != width:  # match widths, preserve aspect
            h = round(img.height * width / img.width)
            img = img.resize((width, h), Image.LANCZOS)
        pieces.append(_label_strip(width, label_px, label))
        pieces.append(img)
        pieces.append(Image.new("RGB", (width, pad_px), "white"))
    total_h = sum(p.height for p in pieces[:-1])  # drop trailing pad
    out = Image.new("RGB", (width, total_h), "white")
    y = 0
    for p in pieces[:-1]:
        out.paste(p, (0, y))
        y += p.height
    return out


def combine(which, indir, outdir, mu, sig, label_px, pad_px):
    indir = Path(indir) if Path(indir).is_absolute() else _HERE / indir
    outdir = (Path(outdir) if Path(outdir).is_absolute()
              else _HERE / outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    recipe = _RECIPES[which]
    loaded = []
    for fname, label in recipe:
        path = indir / fname.format(mu=mu, sig=sig)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; check --indir and the mu/sig tags")
        loaded.append((_load(path), label))
    out = _stack(loaded, label_px, pad_px)
    outpath = outdir / f"w3_{which}_mu{mu}.png"
    out.save(outpath, dpi=(200, 200))
    print(f"[combiner] wrote {outpath} ({out.width} x {out.height})")
    return outpath


def main(args):
    for which in args.which:
        combine(which, args.indir, args.outdir, args.mu, args.sig,
                args.label_px, args.pad_px)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default=_Args.indir)
    p.add_argument("--outdir", default=_Args.outdir)
    p.add_argument("--mu", default=_Args.mu)
    p.add_argument("--sig", default=_Args.sig)
    p.add_argument("--which", nargs="+", default=_Args.which,
                   choices=sorted(_RECIPES))
    p.add_argument("--label-px", type=int, dest="label_px",
                   default=_Args.label_px)
    p.add_argument("--pad-px", type=int, dest="pad_px",
                   default=_Args.pad_px)
    main(p.parse_args())
