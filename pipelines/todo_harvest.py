#!/usr/bin/env python3
"""
todo_harvest.py, version 1.1 (1 Nov 2026)

Collects, from the local tree, everything the bin-A todos of
TODO_SCAN_step13_1031.md ask to be "opened and transcribed": the reports,
run stamps, configs and metadata of the named result directories, the
docstring, changelog header, _Args block and argparse defaults of the named
scripts, and the metadata arrays of the named npz runs. Everything goes into
one folder and one zip with a MANIFEST.md keyed by todo number, so that the
chat holding the dissertation tree can transcribe the numbers with SOURCE
lines without any pipeline being rerun.

The script reads only. It never modifies a file, never imports a pipeline
module and never runs a simulation. It copies text files only (reports,
json, csv, yaml, log, md, txt); large csv and log files are cut to a head
and a tail. Figures and arrays are listed by name and size, not copied.

Python 3.8, standard library plus numpy (numpy is used only to read npz
headers; if it is missing the npz part is skipped and the manifest says so).

Usage, from the pipelines directory (the one that holds results/):
    python todo_harvest.py --outdir results/1101_todo_harvest
Add directories or scripts the mapping does not know with
    --extra_dirs KEYWORD [KEYWORD ...]   --extra_scripts STEM [STEM ...]

Changelog
  v1.0  first version; mapping taken from Part 4 and the "Bin A grouped by
        script" list of TODO_SCAN_step13_1031.md.
  v1.1  the argparse pattern of v1.0 backtracked exponentially (about 11 s on
        a single add_argument line holding a lambda, hours on a long script);
        replaced by a parenthesis-depth scan. The docstring pattern no longer
        runs under DOTALL. Large text files are cut to head and tail by
        streaming (v1.0 read the whole file). A progress line per todo entry.
"""
import argparse
import io
import json
import os
import re
import shutil
import sys
import time
import zipfile
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# The mapping: todo number -> what to look for.
#   dirs    : lower-case substrings matched against result directory names
#             (any depth under --results)
#   scripts : script stems (without .py) to find under --script_roots
#   npz     : glob-like substrings matched against .npz file names under root
#   note    : what the todo asks, in a few words (printed in the manifest)
# ---------------------------------------------------------------------------
TODO_MAP = [
    dict(todos=[14, 23], dirs=["knee_error_budget", "error_budget"],
         scripts=["w2_knee_error_budget"],
         note="two printed offsets 4.4e-4 (14); grid spacing 0.16 or 0.25 of the fig:uhu_knee run (23)"),
    dict(todos=[15], dirs=["knee_sweep", "uhu_knee"], scripts=["knee_sweep_common"],
         note="max lambda_1 on the Tukey-tapered crops of Example 1 (Ch 3 line 384)"),
    dict(todos=[16, 27, 39, 119], dirs=[], scripts=[],
         note="captions: R, t, panel layout of the final-frame figures; served by the generic stamp sweep (section G of the manifest)"),
    dict(todos=[19], dirs=[], scripts=["phase_zigzag_v3"],
         note="inset of the trapezoid, margins and steepness of the second ramp"),
    dict(todos=[20, 63], dirs=["uhu_defect", "defect_diagnostics", "k_based_metrics"],
         scripts=["k_based_metrics_zigzag"],
         note="mu, sigma, t of fig:uhu_defect_diagnostics (20); whether it was made from _Args (63)"),
    dict(todos=[21], dirs=["k_based_metrics"], scripts=["k_based_metrics_zigzag"],
         note="rho of the integrated measures in the other runs"),
    dict(todos=[22, 24], dirs=["knee_bend_fits", "fits_and_residuals"],
         scripts=["knee_bend_fits_and_residuals"],
         note="maxima and rms of the three residual fields"),
    dict(todos=[28, 29], dirs=["mechanism"], scripts=["mechanism_figs"],
         note="tolerance, shading, axis ranges of fig:uhu_mechanisms (28); script and width behind 1e-18 (29)"),
    dict(todos=[33, 84, 88, 89, 91, 92, 93], dirs=["wsindy_knee_validation", "knee_validation"],
         scripts=["wsindy_knee_validation", "evolve_knee"],
         note="smoothing-width region (33); initial sharpness and amplitude (84); frame spacing of the 60 frames (88); exact-route coefficients, stage A1 (89); how the fits fail at five percent, stage A3 (91); constant column and eta (92); width-pi script (93)"),
    dict(todos=[34, 80], dirs=["mg_zigzag_theta_t", "mg_theta_t", "dip"],
         scripts=["mg_zigzag_theta_t"],
         note="runs and frames of the 23 to 41 percent dip (34); unit and core set of the summed charge 108 to 72 (80)"),
    dict(todos=[36, 37], dirs=["nucleation_measures_validation", "measures_validation"],
         scripts=["nucleation_measures_validation"],
         note="what mu sets in the synthetic pair sweep (36); sigma_det, grid spacing, two charge variants (37)"),
    dict(todos=[38], dirs=["persistence"], scripts=["loop_persistence"],
         note="R, frame times, sigma_small of fig:persistence_sh"),
    dict(todos=[40], dirs=["crest", "tracer"], scripts=[],
         note="start and lowest height of the crest tracer"),
    dict(todos=[42], dirs=["phase_compar", "unwrap"], scripts=[],
         note="mu of the 0.02 to 0.60 rad comparison"),
    dict(todos=[43, 46], dirs=["riesz_zigzag_modes", "riesz_modes", "coarse"],
         scripts=["run_riesz_zigzag_modes"],
         note="run and smoothing of the profile residuals (43); sigma_avg pi or pi/2 of the coarse-series fits (46)"),
    dict(todos=[47], dirs=["director", "alignment"], scripts=[],
         note="width of the director and alignment panels"),
    dict(todos=[49, 50, 52, 48], dirs=["mg_estimator_validation", "estimator_validation"],
         scripts=["mg_estimator_validation", "pn_exact"],
         note="dislocation count and grid spacing per mu (49); sigma_dir 0.3 bin values (50); 0.3 against 0.4 continuous-phase lines (52)"),
    dict(todos=[54], dirs=["mg_stage2", "stage2"], scripts=[],
         note="period of the oscillation of fig:mg_stage2 (timeseries.csv)"),
    dict(todos=[56, 66], dirs=["extraction_comparison"], scripts=["extraction_comparison"],
         note="plateau values of the fitted tanh (56); name of the eighth frame (66)"),
    dict(todos=[59, 60], dirs=["defect_gallery", "gallery"], scripts=["defect_gallery", "run_sh_defect_gallery"],
         note="radii of the circuits (59); threshold below which no circulation is computed (60); which component lies along each PGB (61)"),
    dict(todos=[65], dirs=["beta_knee_fit", "beta_fit"], scripts=["beta_knee_fit"],
         note="the 0.006 offset in w-bar; crop spacing 0.116 against 0.110"),
    dict(todos=[70, 71], dirs=["ruler", "pn_vs_sh"], scripts=["ruler_w"],
         note="rule selecting the tilted part (70); slashes and arrows of tab:pn_vs_sh (71)"),
    dict(todos=[72, 73, 125], dirs=["energy_vs_angle"], scripts=["energy_vs_angle_zigzag"],
         note="grid (Nx, Ny, box) of the version 2 rerun (72) and of the tail sweep (73); lower edge of the first sweep's exponent window (125)"),
    dict(todos=[78], dirs=["rcn_balance", "balance_ledger"], scripts=["rcn_balance_ledger"],
         note="28 against 16 cores on the 0.75 tail"),
    dict(todos=[81, 82], dirs=["winding"], scripts=["fig_winding_time"],
         note="file-to-R assignment of fig:winding_time (81); detection width of its charge-onset series (82)"),
    dict(todos=[83, 85, 87, 96, 102], dirs=["wsindy_theta_t", "theta_t"], scripts=["wsindy_theta_t"],
         note="fields behind 70 to 120 (83); taper control (85); collinear pairs and |r| (87); statistic behind 1e-6 against 1e-2 (96); sense of the pointwise-derivative statement (102)"),
    dict(todos=[86], dirs=["1004c_wsindy_band", "wsindy_band"], scripts=[],
         note="domain figure of the 0.85 run"),
    dict(todos=[94, 95, 101, 127, 128], dirs=["rcn2d", "rcn_control", "2d_control"], scripts=["rcn2d_control"],
         note="annulus mean of w about a dip (94); rms of A_t on the static control (95); one percent on the stretching coefficients (101); norms of the convergence checks (128)"),
    dict(todos=[99, 100], dirs=["taper", "periodic"], scripts=[],
         note="test that found the far-field Riesz mismatch displaced on tapered crops (99); periodic crops, if any (100)"),
    dict(todos=[107, 109, 110], dirs=["harmonic"], scripts=["harmonic_moments_sympy"],
         note="determinant of the averaged tensor (107); width behind the left panel of fig:harmonic_tests (109)"),
    dict(todos=[112, 113], dirs=["task2", "averaging"], scripts=["task2_averaging"],
         note="script holding the Monte Carlo check (112); the assembled form (113)"),
    dict(todos=[114, 115, 116], dirs=["sstar", "exact_convolution", "sigma_window"],
         scripts=["exact_convolution_matrix", "sigma_window_maps", "sstar_ellipse_glyphs",
                  "sstar_extraction_demo", "uhu_symbolic"],
         note="sigma_y per figure (114); ridge selection and the medians script (115); PN values that move at the percent level (116)"),
    dict(todos=[117, 118], dirs=[], scripts=["core", "pgb_zigzag"],
         note="symbol size per grid (117); mask of f_1, f_2, f_3 (118)"),
    dict(todos=[57, 120, 122], dirs=["crop_windowing", "forensics"],
         scripts=["crop_windowing_forensics"],
         npz=["Rp5_", "R1p0_"],
         note="parameters of the shallow-knee runs from npz metadata (120); whether any was periodic in x (57); k_1 ripple 0.035 against 0.020 (122)"),
    dict(todos=[121], dirs=["late", "theta_t_rms", "settle"], scripts=[],
         note="script and per-run values of the late-time theta_t rms"),
    dict(todos=[124], dirs=["diamond"], scripts=["phase_centerline_and_field_plots_diamond", "pgb_diamond"],
         note="estimator and width behind fig:diamond_profiles; which diagonal carries the knee"),
    dict(todos=[126], dirs=["0.999", "0p999", "mu0999"], scripts=[],
         note="what besides the grid differs between the two 0.999 runs"),
    dict(todos=[129, 130, 131], dirs=["closure", "rollout", "w_law", "riesz_w"], scripts=[],
         note="norm behind 5 to 37 times w_t (129); free or forced for eq:riesz_w_law (130); supports of the band fits (131)"),
]

TEXT_EXT = {".txt", ".md", ".json", ".csv", ".tsv", ".yaml", ".yml", ".log", ".cfg",
            ".ini", ".toml", ".out", ".rst"}
STAMP_WORDS = ("stamp", "args", "config", "meta", "param", "setting", "report",
               "summary", "ledger", "table", "fit_report", "readme", "changelog", "manifest")
SKIP_DIRS = {".git", "__pycache__", ".idea", ".venv", "venv", "node_modules", ".mypy_cache"}
EXCLUDE_PREFIXES: List[str] = []   # absolute paths never walked (the harvest's own output, earlier harvests)


class _Args:
    """Defaults; mirrored by the command line below (PyCharm and terminal)."""
    root = "."                                 # repository or pipelines directory
    results = "results"                        # results tree, relative to root
    script_roots = [".", "..", "../src", "../pipelines", "src"]  # where scripts live
    outdir = "results/1101_todo_harvest"       # output folder (zip beside it)
    max_file_kb = 400                          # text files above this are cut to head and tail
    head_lines = 80
    tail_lines = 30
    max_depth = 4                              # depth inside a matched result directory
    max_files = 6000                           # guard on the number of copied files
    max_total_mb = 250                         # guard on the copied total
    extra_dirs = []                            # more keywords for result directories
    extra_scripts = []                         # more script stems
    generic_stamps = 1                         # also sweep every stamp/config/args/report file under results
    npz_meta = 1                               # read npz metadata (needs numpy)
    small_array = 64                           # npz arrays with at most this many elements are printed


def parse_args() -> argparse.Namespace:
    d = _Args()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=d.root)
    p.add_argument("--results", default=d.results)
    p.add_argument("--script_roots", nargs="*", default=d.script_roots)
    p.add_argument("--outdir", default=d.outdir)
    p.add_argument("--max_file_kb", type=int, default=d.max_file_kb)
    p.add_argument("--head_lines", type=int, default=d.head_lines)
    p.add_argument("--tail_lines", type=int, default=d.tail_lines)
    p.add_argument("--max_depth", type=int, default=d.max_depth)
    p.add_argument("--max_files", type=int, default=d.max_files)
    p.add_argument("--max_total_mb", type=int, default=d.max_total_mb)
    p.add_argument("--extra_dirs", nargs="*", default=d.extra_dirs)
    p.add_argument("--extra_scripts", nargs="*", default=d.extra_scripts)
    p.add_argument("--generic_stamps", type=int, default=d.generic_stamps)
    p.add_argument("--npz_meta", type=int, default=d.npz_meta)
    p.add_argument("--small_array", type=int, default=d.small_array)
    return p.parse_args()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def walk(top: str, max_depth: Optional[int] = None):
    top = os.path.abspath(top)
    base_depth = top.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(top):
        dirnames[:] = [x for x in dirnames if x not in SKIP_DIRS and "todo_harvest" not in x
                       and not any(os.path.join(dirpath, x).startswith(e) for e in EXCLUDE_PREFIXES)]
        if any(os.path.abspath(dirpath).startswith(e) for e in EXCLUDE_PREFIXES):
            continue
        if max_depth is not None and dirpath.count(os.sep) - base_depth >= max_depth:
            dirnames[:] = []
        yield dirpath, dirnames, filenames


def is_text(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in TEXT_EXT


def head_tail(path: str, nh: int, nt: int) -> str:
    """First nh and last nt lines, read in one pass without holding the file."""
    import collections
    head = []
    tail = collections.deque(maxlen=nt)
    n = 0
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                if n < nh:
                    head.append(line)
                else:
                    tail.append(line)
                n += 1
    except OSError as e:
        return "could not read: %s\n" % e
    cut = n - len(head) - len(tail)
    if cut <= 0:
        return "".join(head) + "".join(tail)
    return "".join(head) + "\n... [%d lines cut by todo_harvest] ...\n\n" % cut + "".join(tail)


class Copier:
    """Copies each source file once; records destination paths."""

    def __init__(self, outdir: str, args):
        self.outdir = outdir
        self.args = args
        self.copied: Dict[str, str] = {}
        self.total = 0
        self.count = 0
        self.stopped = False

    def copy(self, src: str, rel_root: str) -> Optional[str]:
        src = os.path.abspath(src)
        if src in self.copied:
            return self.copied[src]
        if self.stopped:
            return None
        rel = os.path.relpath(src, rel_root)
        rel = rel.replace("..", "_up_")
        dst = os.path.join(self.outdir, "files", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        size = os.path.getsize(src)
        cap = self.args.max_file_kb * 1024
        if size > cap:
            dst = dst + ".head_tail.txt"
            with open(dst, "w") as f:
                f.write("# todo_harvest: %s is %d bytes; head %d and tail %d lines follow\n"
                        % (src, size, self.args.head_lines, self.args.tail_lines))
                f.write(head_tail(src, self.args.head_lines, self.args.tail_lines))
            size = os.path.getsize(dst)
        else:
            shutil.copy2(src, dst)
        self.copied[src] = dst
        self.total += size
        self.count += 1
        if self.count >= self.args.max_files or self.total >= self.args.max_total_mb * 1024 * 1024:
            self.stopped = True
        return dst


def balanced_calls(src: str, token: str, limit: int = 400) -> List[str]:
    """Every call that starts with token, up to its matching parenthesis
    (strings are not parsed; a parenthesis inside a string can end a call
    early, which only shortens the printed line)."""
    out = []
    i = 0
    while len(out) < limit:
        j = src.find(token, i)
        if j < 0:
            break
        k = j + len(token)
        depth = 1
        while k < len(src) and depth > 0 and k - j < 2000:
            c = src[k]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            k += 1
        line_start = src.rfind("\n", 0, j) + 1
        out.append(re.sub(r"\s+", " ", src[line_start:k]).strip())
        i = k
    return out


def extract_script(path: str, small: int = 0) -> str:
    """Docstring, changelog header, _Args block and argparse defaults of a script."""
    try:
        src = open(path, "r", errors="replace").read()
    except OSError as e:
        return "could not read %s: %s\n" % (path, e)
    out = io.StringIO()
    out.write("# %s  (%d lines, modified %s)\n\n" % (
        path, src.count("\n") + 1, time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))))
    # leading comment block and module docstring
    m = re.match(r'(?:[ \t]*#[^\n]*\n|[ \t]*\n)*[ \t]*[rubRUB]*(?:"""([\s\S]*?)"""|\'\'\'([\s\S]*?)\'\'\')', src)
    if m:
        out.write("## module docstring\n")
        out.write((m.group(1) or m.group(2) or "").strip() + "\n\n")
    lead = re.match(r'((?:\s*#.*\n)+)', src)
    if lead and lead.group(1).strip():
        out.write("## leading comments\n" + lead.group(1) + "\n")
    # changelog-like lines anywhere in comments
    chl = [ln for ln in src.splitlines() if re.search(r'#.*\b(v\d+\.\d+|changelog|change log|\d{1,2} (Sep|Oct|Nov)\b)', ln)]
    if chl:
        out.write("## changelog-like comment lines\n" + "\n".join(chl[:80]) + "\n\n")
    # _Args class block
    m = re.search(r'^class _Args\b.*?(?=^\S)', src, re.S | re.M)
    if m:
        out.write("## class _Args\n" + m.group(0).rstrip() + "\n\n")
    else:
        m = re.search(r'^_Args\s*=.*?(?=^\S)', src, re.S | re.M)
        if m:
            out.write("## _Args assignment\n" + m.group(0).rstrip() + "\n\n")
    # argparse add_argument lines (with continuation lines)
    adds = balanced_calls(src, "add_argument(")
    if adds:
        out.write("## argparse add_argument\n" + "\n".join(a.strip() for a in adds) + "\n\n")
    # constants at module level that look like parameters
    consts = re.findall(r'^([A-Z][A-Z0-9_]{2,})\s*=\s*(.{0,100})$', src, re.M)
    if consts:
        out.write("## module-level constants\n" + "\n".join("%s = %s" % c for c in consts[:60]) + "\n\n")
    # def signatures with defaults (a compact view of the parameters)
    sigs = re.findall(r'^def (\w+)\(([^)]*)\)', src, re.M)
    if sigs:
        out.write("## function signatures\n" + "\n".join("%s(%s)" % (n, re.sub(r'\s+', ' ', a)) for n, a in sigs[:120]) + "\n")
    return out.getvalue()


def npz_metadata(path: str, small: int) -> str:
    out = io.StringIO()
    out.write("# %s  (%d bytes)\n" % (path, os.path.getsize(path)))
    try:
        import numpy as np
        from numpy.lib import format as npf
    except Exception as e:  # noqa
        out.write("numpy not importable (%s); metadata not read\n" % e)
        return out.getvalue()
    try:
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if not name.endswith(".npy"):
                    out.write("  %-28s (non-npy member, %d bytes)\n" % (name, z.getinfo(name).file_size))
                    continue
                with z.open(name) as f:
                    try:
                        ver = npf.read_magic(f)
                        if ver == (1, 0):
                            shape, fortran, dtype = npf.read_array_header_1_0(f)
                        else:
                            shape, fortran, dtype = npf.read_array_header_2_0(f)
                    except Exception as e:  # noqa
                        out.write("  %-28s header unreadable: %s\n" % (name, e))
                        continue
                n = 1
                for s in shape:
                    n *= s
                key = name[:-4]
                if n <= small and dtype.kind != "O":
                    try:
                        with z.open(name) as f2:
                            arr = np.load(io.BytesIO(f2.read()), allow_pickle=False)
                        val = np.array2string(arr, precision=6, max_line_width=200)
                    except Exception as e:  # noqa
                        val = "(unreadable: %s)" % e
                    out.write("  %-28s shape=%s dtype=%s  value=%s\n" % (key, shape, dtype, val))
                elif dtype.kind == "O":
                    out.write("  %-28s shape=%s dtype=object (pickled; not read)\n" % (key, shape))
                else:
                    out.write("  %-28s shape=%s dtype=%s\n" % (key, shape, dtype))
    except zipfile.BadZipFile:
        out.write("  not a zip-format npz\n")
    return out.getvalue()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    results = os.path.join(root, args.results)
    outdir = os.path.abspath(args.outdir)
    if os.path.exists(outdir):
        print("outdir exists, files will be added or replaced:", outdir)
    os.makedirs(outdir, exist_ok=True)
    EXCLUDE_PREFIXES.append(outdir)
    copier = Copier(outdir, args)
    t_start = time.time()
    manifest = io.StringIO()
    manifest.write("# todo_harvest v1.0 manifest\n\n")
    manifest.write("root %s\nresults %s\nmade %s\n\n" % (root, results, time.strftime("%Y-%m-%d %H:%M")))
    manifest.write("Every path below is relative to files/ inside this harvest. A todo with "
                   "'nothing found' needs a keyword added with --extra_dirs or --extra_scripts, "
                   "or its output was never written to results/.\n\n")

    # 1. index the results tree once
    print("indexing", results)
    result_dirs: List[str] = []
    if os.path.isdir(results):
        for dirpath, dirnames, filenames in walk(results):
            result_dirs.append(dirpath)
    else:
        print("results directory not found:", results)
    print("  %d directories" % len(result_dirs))

    # 2. index candidate scripts once
    script_index: Dict[str, List[str]] = {}
    for sr in args.script_roots:
        srp = os.path.abspath(os.path.join(root, sr))
        if not os.path.isdir(srp):
            continue
        for dirpath, dirnames, filenames in walk(srp, max_depth=6):
            if os.path.abspath(dirpath).startswith(os.path.abspath(results)):
                continue
            for fn in filenames:
                if fn.endswith(".py"):
                    script_index.setdefault(fn[:-3], []).append(os.path.join(dirpath, fn))
    print("  %d script stems indexed" % len(script_index))

    # 3. per-todo harvest
    entries = list(TODO_MAP)
    if args.extra_dirs or args.extra_scripts:
        entries.append(dict(todos=["extra"], dirs=list(args.extra_dirs), scripts=list(args.extra_scripts),
                            note="added on the command line"))
    nothing: List[str] = []
    for ent in entries:
        tag = ", ".join(str(t) for t in ent["todos"])
        print("  todo %-28s %6.0f s, %d files so far" % (tag, time.time() - t_start, copier.count), flush=True)
        manifest.write("## todo %s\n%s\n\n" % (tag, ent.get("note", "")))
        found_any = False
        # result directories (all keywords of the entry, each directory once)
        hits = []
        for kw in ent.get("dirs", []):
            kwl = kw.lower()
            hits += [d for d in result_dirs if kwl in os.path.basename(d).lower() and d not in hits]
        # keep the shallowest hits (a matched directory covers its subtree)
        hits = [d for d in hits if not any(d != h and d.startswith(h + os.sep) for h in hits)]
        if hits:
            for d in sorted(hits):
                found_any = True
                manifest.write("- directory `%s`\n" % os.path.relpath(d, results))
                listing = []
                for dirpath, dirnames, filenames in walk(d, max_depth=args.max_depth):
                    for fn in sorted(filenames):
                        p = os.path.join(dirpath, fn)
                        try:
                            size = os.path.getsize(p)
                        except OSError:
                            continue
                        listing.append("%10d  %s" % (size, os.path.relpath(p, d)))
                        if is_text(p):
                            dst = copier.copy(p, results)
                            if dst:
                                manifest.write("  - `%s`\n" % os.path.relpath(dst, os.path.join(outdir, "files")))
                lst = os.path.join(outdir, "listings", os.path.relpath(d, results).replace(os.sep, "__") + ".txt")
                os.makedirs(os.path.dirname(lst), exist_ok=True)
                with open(lst, "w") as f:
                    f.write("# listing of %s\n" % d + "\n".join(listing) + "\n")
        # scripts
        for stem in ent.get("scripts", []):
            paths = script_index.get(stem, [])
            if not paths:
                manifest.write("- script `%s.py`: not found under script roots\n" % stem)
                continue
            for p in paths:
                found_any = True
                ext = os.path.join(outdir, "scripts", stem + ".extract.txt")
                os.makedirs(os.path.dirname(ext), exist_ok=True)
                mode = "a" if os.path.exists(ext) else "w"
                with open(ext, mode) as f:
                    f.write(extract_script(p))
                    f.write("\n" + "=" * 78 + "\n")
                manifest.write("- script `%s` -> scripts/%s.extract.txt\n" % (os.path.relpath(p, root), stem))
        # npz metadata
        if args.npz_meta:
            for pat in ent.get("npz", []):
                hits = []
                seen_roots: Set[str] = set()
                for sr in [root] + [os.path.join(root, s) for s in args.script_roots]:
                    srp = os.path.abspath(sr)
                    if srp in seen_roots or not os.path.isdir(srp):
                        continue
                    seen_roots.add(srp)
                    for dirpath, dirnames, filenames in walk(srp, max_depth=8):
                        for fn in filenames:
                            if fn.endswith(".npz") and pat in fn:
                                p = os.path.join(dirpath, fn)
                                if p not in hits:
                                    hits.append(p)
                if not hits:
                    manifest.write("- npz `*%s*.npz`: none found under root\n" % pat)
                for p in sorted(hits)[:64]:
                    found_any = True
                    meta = os.path.join(outdir, "npz_meta", os.path.basename(p) + ".meta.txt")
                    os.makedirs(os.path.dirname(meta), exist_ok=True)
                    with open(meta, "w") as f:
                        f.write(npz_metadata(p, args.small_array))
                    manifest.write("- npz `%s` -> npz_meta/%s.meta.txt\n" % (os.path.relpath(p, root), os.path.basename(p)))
        if not found_any:
            manifest.write("- nothing found\n")
            nothing.append(tag)
        manifest.write("\n")

    # 4. generic sweep of stamps, configs, args and reports under results
    if args.generic_stamps and os.path.isdir(results):
        print("  generic sweep of stamps and reports ...", flush=True)
        manifest.write("## G. generic sweep: every stamp, config, args, meta, report, summary, ledger and table file under results\n\n")
        n0 = copier.count
        for dirpath, dirnames, filenames in walk(results):
            for fn in filenames:
                fl = fn.lower()
                if is_text(os.path.join(dirpath, fn)) and any(w in fl for w in STAMP_WORDS):
                    copier.copy(os.path.join(dirpath, fn), results)
        manifest.write("%d files copied in this sweep (see files/).\n\n" % (copier.count - n0))

    # 5. tree of results directories with mtimes
    tree_path = os.path.join(outdir, "FOUND_TREE.txt")
    with open(tree_path, "w") as f:
        f.write("# result directories under %s (depth 2), newest first\n" % results)
        rows = []
        if os.path.isdir(results):
            for d in result_dirs:
                depth = os.path.relpath(d, results).count(os.sep)
                if depth <= 1:
                    rows.append((os.path.getmtime(d), os.path.relpath(d, results)))
        for mt, rel in sorted(rows, reverse=True):
            f.write("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M", time.localtime(mt)), rel))
    with open(os.path.join(outdir, "SCRIPT_INDEX.txt"), "w") as f:
        for stem in sorted(script_index):
            for p in script_index[stem]:
                f.write("%s\t%s\n" % (stem, os.path.relpath(p, root)))

    manifest.write("## totals\n%d files copied, %.1f MB; stopped early: %s\n" % (
        copier.count, copier.total / 1e6, copier.stopped))
    if nothing:
        manifest.write("\nTodos with nothing found: %s\n" % "; ".join(nothing))
    with open(os.path.join(outdir, "MANIFEST.md"), "w") as f:
        f.write(manifest.getvalue())

    # 6. zip
    zpath = outdir.rstrip(os.sep) + ".zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirnames, filenames in os.walk(outdir):
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                z.write(p, os.path.relpath(p, os.path.dirname(outdir)))
    print("wrote", os.path.join(outdir, "MANIFEST.md"))
    print("wrote", zpath, "(%.1f MB)" % (os.path.getsize(zpath) / 1e6))
    print("%d files copied, %.1f MB" % (copier.count, copier.total / 1e6))
    if nothing:
        print("nothing found for todos:", "; ".join(nothing))
        print("  add keywords with --extra_dirs / --extra_scripts and rerun (same outdir is fine)")


if __name__ == "__main__":
    main()
