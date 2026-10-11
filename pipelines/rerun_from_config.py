"""rerun_from_config.py  (pipelines/)

Rerun a wsindy_theta_t data fit of record from its own config.json, with
a few settings overridden, so that a check changes only what it means to
change.  Built for the critical todo C4 of TODO_TRIAGE_1008 (does the
identity w_t = 2 k.grad theta_t recover, and is J still selected, once
the test-function supports no longer cover the knee core?), usable for
any later ablation of a run of record.

How it works.  The settings start from wsindy_theta_t_v19._Args (v1.9
reproduces v1.8 line for line at its defaults), every key of the config
that _Args also has is copied over (so the run of record is reproduced),
then each --set key=value is applied, with the value cast to the type of
the _Args default.  Keys of the config that v1.9 does not know are
listed and ignored.  The output goes to --outdir (not the config's
outdir), and --tag is appended to the stamp, so a rerun never writes
into the folder of record.

  --list ROOT     print every config.json under ROOT written by a data
                  fit (npz basename, t0, t1, domain, radius, core_excl,
                  version line of the report) and exit; use it to find
                  the runs of record.

Usage
  python rerun_from_config.py --list results
  python rerun_from_config.py --config results/1004d_wsindy_low/<stem>/<stamp>/config.json \
      --outdir results/1009_c4 --tag base
  python rerun_from_config.py --config ... --outdir results/1009_c4 --tag nc \
      --set no_cross=1 radius_y=0.5

Changelog
v1.0 (9 Oct 2026) first version.  Python 3.8.  Imports
      wsindy_theta_t_v19 only; edits nothing.
"""
import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def _cast(default, text):
    if isinstance(default, bool):
        return text.lower() in ("1", "true", "yes")
    if isinstance(default, int):
        return int(float(text))
    if isinstance(default, float):
        return float(text)
    return text


def list_configs(root):
    rows = []
    for cfg in sorted(Path(root).rglob("config.json")):
        try:
            c = json.loads(cfg.read_text())
        except Exception:
            continue
        if not c.get("npz"):
            continue
        rep = cfg.parent / "report.txt"
        ver = rep.read_text().splitlines()[0][:40] if rep.exists() else ""
        rows.append("%-34s t %5s-%-6s %-6s rad %-4s ce %-4s %s\n    %s" % (
            Path(str(c.get("npz"))).name[:34], c.get("t0"), c.get("t1"), c.get("domain", ""),
            c.get("radius", ""), c.get("core_excl", ""), ver, cfg))
    print("\n".join(rows) if rows else "no data configs under %s" % root)


def main(a):
    if a.list:
        list_configs(a.list)
        return 0
    import wsindy_theta_t_v19 as ws
    cfg = json.loads(Path(a.config).read_text())
    args = argparse.Namespace()
    known = [k for k in dir(ws._Args) if not k.startswith("_")]
    for k in known:
        setattr(args, k, getattr(ws._Args, k))
    copied, unknown = [], []
    for k, v in cfg.items():
        if k in known:
            setattr(args, k, v); copied.append(k)
        else:
            unknown.append(k)
    for item in a.set or []:
        k, _, v = item.partition("=")
        if k not in known:
            print("unknown setting %s (not in wsindy_theta_t_v19._Args)" % k)
            return 2
        setattr(args, k, _cast(getattr(ws._Args, k), v))
    args.outdir = a.outdir
    args.tag = (str(cfg.get("tag", "")) + "_" if cfg.get("tag") else "") + a.tag
    args.mode = "data"
    print("rerun of %s\n  copied %d settings; ignored (not in v1.9): %s\n  overrides: %s\n  outdir %s, tag %s"
          % (a.config, len(copied), ", ".join(unknown) or "none", " ".join(a.set or []) or "none",
             args.outdir, args.tag))
    return ws.main(args)


class _Args:
    config = ""                     # config.json of the run of record
    outdir = str(_HERE / "results" / "rerun_from_config")
    tag = "rerun"
    list = ""                       # root folder to list data configs under (then exit)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=_Args.config)
    p.add_argument("--outdir", default=_Args.outdir)
    p.add_argument("--tag", default=_Args.tag)
    p.add_argument("--list", default=_Args.list)
    p.add_argument("--set", nargs="*", default=None, help="key=value overrides of wsindy_theta_t_v19 settings")
    sys.exit(main(p.parse_args()))
