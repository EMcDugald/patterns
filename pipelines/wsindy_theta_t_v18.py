# pipelines/wsindy_theta_t_v18.py
"""
Weak-form (WSINDy) identification of the phase rate theta_t, in two
forms, with the target and the fields taken from the seam-free
monogenic-gradient estimator (mg_estimator_validation, validated to a few
percent beyond half a spacing from a core; mg_pair_dynamics_validation
for theta_t).

Target: b = Int[ theta_t phi ], with theta_t the validated pointwise rate
(centred differences of z, Im(conj z z_t)/|z|^2) integrated against a
separable compactly supported test function phi(x, y, t). No time
integration by parts is needed and no measurement kernel enters: the
estimator's k is pointwise, so the library columns are built from the
same fields the target is built from.

RHS library (theta_t = sum c_j L_j):
    1, div k, w div k, k.grad w, Lap div k, w, Lap w, (div k)^2, w^2 div k
  Linear terms fully weak: Int[div k phi] = -Int[k . grad phi],
  Int[Lap div k phi] = -Int[k . grad(Lap phi)], Int[Lap w phi] =
  Int[w Lap phi]. Terms nonlinear in a derivative use an inner mollified
  derivative (eta) and a pointwise product before the outer integral.
  RCN truth (rcn_w_evolution_check, f(w) = w):
    tau theta_t = 2 alpha (w div k + k . grad w) - eta_b Lap div k,
  so c(w div k) = c(k.grad w) = 2 alpha/tau, c(Lap div k) = -eta_b/tau.
  Extended RHS library (data, --lib_ext 1): the base library plus the
  amplitude columns A_dev = A - Abar, A_dev div k, k.grad A (is the
  pre-winding -w term, or the bending sign, the amplitude channel?) and
  the defect columns curl k, J = det(grad k), (curl k)^2. Identity trap:
  (div k)^2 + (curl k)^2 = |grad k|^2 + 2 J, so |grad k|^2 is never in
  this library; with it absent the three are independent. Abar is the
  median of A over the far domain (fallback: window median).

ENERGY library (theta_t = -(1/tau) delta E/delta theta, E = sum
c_i Int e_i; the variational ansatz in closed form):
    e in { w^2, w^3, (div k)^2, w (div k)^2, |grad w|^2 }
  Column i = weak form of -delta e_i/delta theta against phi:
    e = g(w):        -Int[ g'(w) 2 k . grad phi ]            (no data derivative)
    e = (div k)^2:   +2 Int[ k . grad(Lap phi) ]             (fully weak)
    e = w (div k)^2: -2 Int[ (div k)^2 k . grad phi ] - 2 Int[ w div k Lap phi ]
    e = |grad w|^2:  -4 Int[ (Lap w) k . grad phi ] + 2 Int[ ... ] (inner)
  RCN truth: E = (eta_b/2)(Lap theta)^2 + (alpha/2) w^2, so c(w^2) =
  alpha/(2 tau), c((div k)^2) = eta_b/(2 tau), all others zero.

Control (--mode control): the 1d RCN knee relaxation of check (6),
    tau p_t = 2 alpha d_yy[ w p ] - eta_b p_yyyy,  w = k1^2 + p^2 - 1,
  p = k2 = Phi_y, from a planted sharp knee p0 = mu tanh(beta0 mu y),
  Dirichlet p = +-mu and p_yy = 0 at the ends, method of lines (BDF).
  theta = k1 x + Phi(y, t) on a 2d grid, u = cos theta, then the SAME
  estimator pipeline as on data; the exact route (exact k, exact theta_t
  from the PDE) is fitted alongside to separate method from estimator.
  The control writes bias.json: the factor truth/estimator for each
  forced RCN coefficient at this kernel radius (the crease bias).

Data (--npz file[,file,...]): frames from each file, window --t0/--t1,
domain --domain {all, far, band, row, shell, active}, --side {both,
upper, lower}. Domains are cut by the distance
to the defect ROW (|y - y_row|, in spacings; y_row from gb_row_in_crop):
  all   everything;  row  d_row < row_to;  band  d_row < band_to;
  far   d_row > far_from,
and in every domain the disc of core_excl spacings around each detected
amplitude core is removed (core_excl 0 disables; pre-winding the phase is
smooth on the row and 0 is the reading of record). The row distance,
not the core distance, cuts the domains so that the far and band domains
exist before any core has formed. Query centres are those whose
test-function support stays inside the box; --excl_dilate 1 also keeps
the support out of the excluded region. With several files the rows are
pooled (RCN coefficients depend on R, not mu: a pooled fit across mu is
a direct test) and each file is also fitted alone.

Selection: fit_target (STLSQ sweep) + every 2-term OLS candidate +
sparsest-within-rel_tol Pareto + contribution floor, spatial holdout on
x. Coefficients read by support identity and holdout R^2.

Additional targets, same frames, same kernel, same domains:
  w_t  (w = |k|^2 - 1 pointwise from the estimator; target Int[w_t phi]
       with w_t by centred differences). Library = the RCN w terms of
       rcn_w_evolution_check check (2) plus the isotropic/anisotropic
       diffusion columns and an amplitude column:
         1, w, w^2, kk:ddw, w_nn, |grad w|^2, (k.grad w) div k, w Lap w,
         w |grad k|^2, (div k)^2, k.grad Lap div k, A_dev
       (Lap w is built but kept out of the free library, since
       Lap w = w_ss + w_nn; it enters the forced comparison
       {Lap w, w}). RCN truth at alpha = eta = tau: kk:ddw 4,
       |grad w|^2 2, (k.grad w) div k 4, w Lap w 2, w |grad k|^2 -4,
       k.grad Lap div k -2, no w, no w_nn.
  A_t  (A = |z|; target Int[A_t phi]). Library 1, A, A^3, Lap A, w, A w,
       A^2, A w^2; forced comparisons {A, A^3} (far cubic) and
       {A, A^3, w, A w} (band law).

Report blocks are labelled "<target> | <library> | free" and
"<target> | <library> | forced {support}" and every block writes a
holdout figure fit_<target>_<library>_<kind>[_<route>].png. One
library-term surface figure per file (the pointwise fields at the
middle frame of the window) and, on the control, a movie figure (phase
surface Phi(y, t), pattern at t0 and t_end, k2 profile vs t).

Outputs (relative): results/wsindy_theta_t/<stem>/<stamp>/report.txt,
  fits.json, bias.json (control), fit_*.png, library_terms_<stem>.png,
  control_movie.png

Changelog
---------
v1.8  Questions left by the 1004b runs (k.grad A first at 0.85; tied S
      costs 0.14 at 0.90; RCN-signed bending in the shells).
      (1) Amplitude-coupled control (--control_amp none|slaved|lagged),
      the coupled-amplitude control of the draft's open questions. The
      RCN knee movie is rendered as u = A cos theta with A(y, t):
        slaved: A = sqrt(max(4/3 (R - w^2), A_floor^2)) (eq:A1_slaving);
        lagged: A_t = (R - w^2) A - 3/4 A^3 integrated pointwise along the
                movie (RK4, w linear in t between frames), from the slaved
                value at t = 0 (eq:SH_amplitude_band, the phase driving the
                amplitude, no feedback).
      The phase law is pure RCN in both, so every amplitude or defect
      term the fits return on this control is leakage: the exact route
      must return zero, the estimator route measures what the extraction
      manufactures from a moving amplitude. Both routes now carry A
      (exact: the planted A; estimator: |z|), so the amplitude columns,
      the RHS+ext blocks, the exhaustive search and the A_t blocks run on
      the control; on the lagged control the A_t truth is (R, -3/4, -1) on
      {A, A^3, A w^2}. At R = 1/2 the knee's core wavenumber k1^2 = 1 - mu^2
      lies below the band edge 1 - sqrt(R) once mu > 0.841, so the slaved
      amplitude would vanish on the crease line; A_floor (--amp_floor,
      fraction of the far amplitude) keeps u defined.
      (2) Cross-Newell amplitude columns (--lib_cn 1). With the flux of
      tau theta_t = -div(k B) carrying A^2, an amplitude that is not
      slaved gives div(A^2 w k)/A^2 = S + 2 w k.grad(ln A). New columns
      "w k.grad lnA" = w (k.grad A)/max(A, a_floor) (a_floor = 0.25 of the
      far median) and S_A = S + 2 (w k.grad lnA), the one-coefficient
      law. Forced blocks: {S_A}, {S_A, J}, {S_A, Lap div k},
      {w div k, k.grad w, k.grad A}, {w div k, k.grad w, w k.grad lnA},
      tied {S, k.grad A}; and an exhaustive size-3 search over the 16
      columns plus "w k.grad lnA" (680 triples). The v1.7 library and its
      exhaustive search are unchanged, so v1.7 numbers stay comparable.
      (3) Spatial decimation (--space_stride k): keep every k-th grid
      point in x and y before the estimator (the row index follows). If
      the fits do not move at half resolution, a finer simulation grid is
      not what limits the near-row fits.
      (5) fit_pareto takes the best holdout R2 over non-empty supports
      with a finite R2 only. When every fitted support scored below the
      empty model (all R2 negative, met on the ENERGY library of a
      synthetic file at space_stride 2), the empty STLSQ support set the
      best, no non-empty support was admissible and the run crashed.
      Pools in which some non-empty support beats the empty one (every
      data report so far) select exactly as before.
      (4) fits.json gains target_rms per block set and the config of the
      strip; wsindy_strip_profile.py reads a set of shell runs cut into
      strips by row distance and plots the bending, the unexplained
      variance of each law and the target scale against distance.
v1.7.1  Domain figure (--domain_fig 1, default on): domains_<stem>.png per
      data file, the fitting domain drawn on the pattern. Left panel: u at
      one frame of the window that carries query centres (the one with the
      most detected cores, latest on ties), in coordinates of spacings
      about the row ((x - x_mid)/L, (y - y_row)/L); the row, band and far
      edges as dashed lines with this run's domain shaded; the core discs
      of radius core_excl; the dilated exclusion when --excl_dilate or
      --dilate_cores_only is set (the cores-only dilation is in x, y and
      t, so a disc of a neighbouring frame can shade this one); the query
      centres of that frame, kept (dots) and censored (crosses: inside the
      domain and on the centre grid but removed by the core discs or the
      dilation); and one test-function support box of half width
      radius x L about a kept centre nearest the row, with the bump's
      standard deviation a/sqrt(2p + 3) in the legend. Right panel, sharing
      the y axis: kept and candidate centres over all frames, binned by
      signed row distance (0.25 L bins), so the censored fraction can be
      read by height above the row. The figure states that the core discs
      remove centres, not data (only the dilation keeps core data out of
      every test function). No fit or report number changes; the frames of
      u are held as float32 only while the figure is on.
v1.7  (1) Exhaustive supports (--exhaustive 1, --exh_sizes "3,4",
      --exh_top). The v1.3 ladder is built from the STLSQ sweep plus every
      2-term candidate, so at sizes >= 3 it lists only the supports the
      sweep happened to visit, and "best at size 3" was never a search
      over all triples. Every support of the requested sizes in the
      theta_t RHS+ext library (16 columns: C(16,3) = 560, C(16,4) = 1820;
      the base RHS library of 9 on the control) is now fitted by least
      squares on the same spatial holdout split as forced_fit (train
      x < median, holdout R2 on the rest, all rows, no subsampling), so
      an exhaustive R2 equals the forced block's R2 for the same support
      (printed as a check). Report per size: the top supports with
      coefficients, the rank and R2 of the reference supports (RCN triple,
      {w div k, k.grad w, J}, RCN + J, ...), the best support containing
      J, the best containing Lap div k and the sign of its bending
      coefficient, the best support without amplitude columns, the count
      of supports within the admissibility tolerance of the best (a
      near-tie count), and the inclusion count of each term among the top
      supports. The question this answers: is {S, J} the best triple, or
      one of several statistically tied triples, and does every good
      triple carrying Lap div k carry it with the RCN sign?
      (2) Tied-coefficient laws. S = div(w k) = w div k + k.grad w as ONE
      column (C["S"], also a pointwise field for the scores): forced
      tied {S, J}, {S, Lap div k}, {S, Lap div k, J}. The untied
      {w div k, k.grad w, J} of v1.6 has three coefficients; the tied
      law is the conservation form the phase-diffusion reading claims,
      and the R2 cost of the tie is the test of that form. S is kept
      out of every free and exhaustive library (it is exactly the sum of
      two library columns).
      (3) Free fit without the amplitude columns (--free_no_amp 1): the
      RHS+ext library minus A_dev, A_dev div k, k.grad A. Whether the
      free selection reaches J when the amplitude channel cannot absorb
      the near-core signal.
      (4) Core-only dilation (--dilate_cores_only 1). v1.6 dilated the
      whole exclusion mask, which includes the domain complement, so a
      shell or band domain with --excl_dilate 1 kept no centres (1004
      censoring runs: shell and band_dilate empty). With this flag only
      the core discs are dilated, by the full test-function support in
      x, y AND t (the bump is a box of half widths a, a, t_half dt, so
      box dilation is exact), and the domain cut applies to the centres
      only. A centre survives iff no part of its support touches a core
      disc in any frame of its time support: the core-free reference.
      Dilation is now a separable maximum filter (identical to
      binary_dilation with a box structure, border 0; checked on random
      masks), which removes the cost of the 3d box structure.
      (5) True frame-spacing ablation (--frame_stride k): frames
      t0..t1 are decimated to every k-th frame AFTER the window cut and
      BEFORE the estimator, so theta_t (centred differences of z), the
      time kernel and the time quadrature all see the coarse cadence.
      --stride_t only decimates query centres and never changed the
      frame spacing. Hold the time kernel fixed in time units with
      t_half x frame_stride = const, and the horizons with score_h =
      2 t_half; --stride_t = 8 / frame_stride keeps the query centres
      on the same time slices as the dt 0.2 record. The time kernel has a
      floor of 3 frames (kernel_d3: m = max(3, ...)), so a t_half below 3
      is silently 3; keep t_half >= 3 at the coarsest stride.
      Checks (3 Oct, sandbox, helper modules stubbed): synthetic library
      of 16 correlated columns with a planted {w div k, k.grad w, J} law
      (1.4, 1.6, -1.9): ranked #1 of 560 triples, coefficients within
      0.1 percent, exhaustive R2 equal to forced_fit to 1e-12, a zero
      column is skipped (455 triples). RCN control, exact route: the RCN
      triple ranks #1 of 84 triples at (2.008, 2.008, -1.01), with 9
      triples within rel_tol of it, every one carrying Lap div k with the
      RCN (negative) sign. box_dilate equals binary_dilation on random
      masks. On a synthetic data file every v1.6 report line is unchanged.
      (6) Output stamp gains _fs<k> (frame_stride > 1), _dco (core-only
      dilation), _dil (excl_dilate) and _<tag> (--tag) so ablation runs
      do not overwrite each other; v1.6 stamps are unchanged when these
      are at their defaults.
v1.6  (1) Prediction score (tracker T1.1). At every query centre the
      measured phase increment over a horizon h (t_half and 2 t_half
      frames; Delta theta = trapezoid of the estimator's theta_t, which
      is the centred difference of the phase, so this is the unwrapped
      phase increment) is compared with each law's prediction: Euler
      from the first frame (h dt x law(t)) and teacher-forced multi-step
      (trapezoid of the law over the measured frames), each pointwise
      at the centres and spatially weak (same bump, no time kernel).
      Score = rms error / rms Delta theta on the holdout half in x, so
      the null law theta_t = 0 scores 1. Laws scored: forced RCN, forced
      RCN + {curl k, J}, forced RCN + amplitude, the three-term
      competitors below, free RHS, free RHS+ext, and the theory law
      (2, 2, -1). Validation on the control (18 Sep, radius 1.5, beta0
      1.8, weak multi-step, h 3 and 6): exact route forced RCN 0.001,
      theory 0.014, Euler 0.05-0.10 (the knee relaxes within the
      horizon); estimator route forced RCN 0.21, free 0.09, theory 1.13
      (worse than the null: the crease attenuation as a prediction
      failure). Pointwise scores are unusable on both routes (0.8 exact:
      fourth differences on a 3-point crease; 70-120 estimator) and are
      printed as an instrument statement only. The scalar-sign change
      moves the control's forced RCN from (1.403, 1.422, -1.363) to
      (1.408, 1.427, -1.366); the record stands.
      (2) Three-term competitors at equal support size: forced
      {w div k, k.grad w, J}, {w div k, k.grad w, A_dev div k},
      {w div k, k.grad w, chi_J}, beside forced RCN {.., Lap div k}: the
      "fourth term" question asked as which THIRD term at equal size.
      (3) chi_J = (Lap - m^2)^{-1} J (m = --chi_m, default 1 = beta k_B),
      the screened potential of the Gaussian curvature, solved by FFT per
      frame (periodic; J is confined to the cores so the y seam is
      harmless). In RCN's own variables J enters the stationary balance
      and the dynamics only through this potential (defects_are_weak
      2.12, universal 3.16; ledger v1.5 changelog), never as a local
      term; a free fit that takes J and rejects chi_J is a statement that
      the SH departure is not RCN-with-defects. Added to RHS_EXT.
      (4) Sign of the estimator's k: ONE scalar per frame from the mean of
      k1 (was np.sign(k1) pointwise, which flipped the ~3 percent of
      pixels next to the cores where k1 < 0 is real structure and made
      div k, S and Lap div k wrong there; J and w are invariant). Control
      unaffected (k1 > 0 everywhere); on data windows with core_excl 0
      (pre-winding) the change can move the near-core columns; rerun one
      window of record to measure it.
      (5) Score horizons need the frames t..t+h inside the window; centres
      whose horizon leaves the window are dropped from the score only.
v1.5  collinearity readout in every block: condition number of the
      column-standardised library (free blocks) or of the forced
      support, and the most correlated pair with |r|; a pair above 0.95
      is interchangeable and is to be quoted jointly. Core merge from
      mg_zigzag_theta_t v1.5 inherited (find_cores merge_dist 0.5 L).
v1.4  consistency block: the identity w_t = 2 k.grad theta_t is applied
      to the measured theta_t (tests the estimator's gradient structure
      on the domain), and the symbolic map of rcn_w_evolution_check
      check (2) turns the identified theta_t law (free and forced RCN)
      into fixed coefficients on the W columns, compared with the
      measured weak w_t by R2 and RMS ratio (no new derivatives; a
      pointwise route through fifth derivatives of k was tried and is
      dominated by the crease ripple on the estimator route). Residual block: forced
      RCN coefficients fixed, the remainder fitted sparsely on the
      non-RCN columns. SH amplitude law forced {A, A^3, A w^2} (truth
      (R, -3/4, -1)) beside the archival band law. Domains: shell
      (band_to..far_from), active (local w^2 + (div k)^2 above
      active_frac x band median), and --side upper/lower.
v1.3  parsimony readout: every free block prints the ladder (best
      holdout R2 at each support size) and the same-size admissible
      alternatives, so an interchangeable pair such as {w^2, w^3} vs
      {w^2, (div k)^2} is visible; --abs_tol (absolute R2 tolerance,
      default 0) beside rel_tol for a sparser official answer; holdout
      R2 and n in every figure title. Control finding recorded in the
      header: the crease bias is not a constant factor (estimator-route
      forced RCN (1.40, 1.42, -1.36) at beta0 1.8, (1.06, 1.08, -1.08)
      at 1.4, (0.61, 0.62, -0.67) at 1.2, (0.09, 0.10, -0.14) at 1.05;
      exact route within 3 percent of (2, 2, -1) at every beta0), so
      the bias.json factors transfer only to windows as far from the
      self-dual balance as the control.
v1.2  (handoff Section 6) block labels "<target> | <library> | free/
      forced"; a holdout figure per block; domains far/band/row cut by
      the row distance (the core distance only removes the core discs);
      extended RHS library with A_dev, A_dev div k, k.grad A, curl k, J,
      (curl k)^2 (--lib_ext, data only; the base library is fitted as
      well so v1.1 numbers stay comparable); multi-npz pooling with
      per-file and pooled blocks; control writes bias.json and data
      reports the crease-bias-corrected forced RCN coefficients as a
      separate, labelled line (--bias_json), never applied silently;
      library-term surface figure and control movie figure; report
      states the time-kernel half width in time units (cadence);
      columns are sampled at the query centres as they are built (the
      full column stacks are never held: ~40 stacks of ny x nx x nt was
      the v1.1 memory and does not fit a 512x768 crop); the three
      wsindy_knee_validation helpers are imported with documented local
      fallbacks. Smoke-tested on the RCN control and on the 0.85 tail
      excerpt (frame 0 dropped, all four domains).
v1.1  Writeup 1 constants alpha = eta = tau = 2R/3 (was the chapter's
      eta = 2 tau); radius default 1.5 spacings (the control fails at
      0.5: the crease ripple of the estimator's amplitude multiplies
      theta_t); --mode flag; w_t and A_t targets; empty-domain guard.
v1.0  first version.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from itertools import combinations
from scipy.ndimage import binary_dilation, correlate1d, maximum_filter  # noqa: F401

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT / "src", _ROOT / "src" / "op_extract", _ROOT / "src" / "utils",
          _HERE):
    sys.path.insert(0, str(p))

from mg_estimator_validation import mg_estimator, fdv                  # noqa: E402,F401
from mg_zigzag_theta_t import load_npz, estimate, find_cores, dist_to_cores  # noqa: E402
from op_extract import t1_common as t1c                                # noqa: E402
from self_dual_sindy import fit_target, r_squared                      # noqa: E402

import matplotlib                                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402


# ---------------------------------------------------------------------
# kernel with derivatives up to third order (poly bump)
# ---------------------------------------------------------------------

def kernel_d3(half_width, spacing, p=6):
    """samples of the mass-normalised bump f(s) = (1 - (s/a)^2)^p and its
    derivatives f', f'', f''' on offsets -m..m"""
    m = max(3, int(round(half_width / spacing)))
    a = m * spacing
    s = np.arange(-m, m + 1) * spacing
    q = np.clip(1.0 - (s / a) ** 2, 0.0, None)
    qp = -2.0 * s / a ** 2
    qpp = -2.0 / a ** 2
    f = q ** p
    f1 = p * q ** (p - 1) * qp
    f2 = p * (p - 1) * q ** (p - 2) * qp ** 2 + p * q ** (p - 1) * qpp
    f3 = (p * (p - 1) * (p - 2) * q ** (p - 3) * qp ** 3
          + 3 * p * (p - 1) * q ** (p - 2) * qp * qpp)
    Z = f.sum() * spacing
    return f / Z, f1 / Z, f2 / Z, f3 / Z, m


# the three wsindy_knee_validation helpers, with local fallbacks that
# implement the same operations (used only when that script is absent)
try:
    from wsindy_knee_validation import corr1d, inner_deriv, _FitArgs   # noqa: E402
    _HELPERS = "wsindy_knee_validation"
except ImportError:                                                     # pragma: no cover
    _HELPERS = "local fallbacks"

    def corr1d(F, kern, axis, spacing):
        """Int F(s) kern(s - s0) ds along one axis, zero outside the box"""
        return correlate1d(F, kern * spacing, axis=axis, mode="constant", cval=0.0)

    def inner_deriv(F, eta, dx, dy, ax, kind="poly", p=6):
        """derivative along ax: np.gradient when eta = 0, otherwise the
        derivative of the mollified field (correlation with -f')"""
        h = dx if ax == "x" else dy
        axis = 1 if ax == "x" else 0
        if eta <= 0:
            return np.gradient(F, h, axis=axis)
        f, f1, *_ = kernel_d3(eta, h, p)
        return -correlate1d(F, f1 * h, axis=axis, mode="nearest")

    class _FitArgs:
        def __init__(self, rel_tol):
            self.thresholds = tuple(np.geomspace(1e-4, 1.0, 25))
            self.rel_tol_pareto = rel_tol
            self.n_boot = 20
            self.boot_frac = 0.5


RHS_LIB = ["1", "div k", "w div k", "k.grad w", "Lap div k", "w", "Lap w",
           "(div k)^2", "w^2 div k"]
RHS_EXT = ["A_dev", "A_dev div k", "k.grad A", "curl k", "J", "(curl k)^2", "chi_J"]
VAR_LIB = ["w^2", "w^3", "(div k)^2", "w (div k)^2", "|grad w|^2"]
W_LIB = ["1", "w", "w^2", "kk:ddw", "w_nn", "|grad w|^2", "(k.grad w) div k",
         "w Lap w", "w |grad k|^2", "(div k)^2", "k.grad Lap div k", "A_dev"]
W_TRUTH = {"kk:ddw": 4.0, "|grad w|^2": 2.0, "(k.grad w) div k": 4.0,
           "w Lap w": 2.0, "w |grad k|^2": -4.0, "k.grad Lap div k": -2.0}
A_LIB = ["1", "A", "A^3", "Lap A", "w", "A w", "A^2", "A w^2"]
RCN_RHS = ["w div k", "k.grad w", "Lap div k"]
RCN_ENERGY = ["w^2", "(div k)^2"]
AMP_COLS = ["A_dev", "A_dev div k", "k.grad A"]                      # v1.7
CN_COL = "w k.grad lnA"                                                 # v1.8
# v1.7: reference supports located in the exhaustive ranking
EXH_REFS = [["w div k", "k.grad w", "Lap div k"], ["w div k", "k.grad w", "J"],
            ["w div k", "k.grad w", "chi_J"], ["w div k", "k.grad w", "A_dev div k"],
            ["w div k", "k.grad w", "Lap div k", "J"], ["w div k", "k.grad w", "J", "chi_J"],
            ["w div k", "k.grad w", "Lap div k", "curl k"], ["w div k", "k.grad w", "J", "w"],
            ["w div k", "k.grad w", "k.grad A"], ["w div k", "k.grad w", "w k.grad lnA"],
            ["w div k", "k.grad w", "k.grad A", "J"], ["w div k", "k.grad w", "Lap div k", "k.grad A"]]


def corr3(F, kx, ky, kt, dx, dy, dt):
    G = corr1d(F, ky, 0, dy)
    G = corr1d(G, kx, 1, dx)
    return corr1d(G, kt, 2, dt)


def kernel_sizes(args, dx, dy, dt, Lsp):
    a = args.radius * Lsp
    return (kernel_d3(a, dx, args.poly_p)[4], kernel_d3(a, dy, args.poly_p)[4],
            kernel_d3(args.t_half * dt, dt, args.poly_p)[4])


def screened_potential(J, dx, dy, m=1.0):
    """chi_J = (Lap - m^2)^{-1} J per frame, periodic FFT (v1.6). J: [ny, nx, nt]."""
    ny, nx = J.shape[:2]
    qx = 2 * np.pi * np.fft.fftfreq(nx, dx); qy = 2 * np.pi * np.fft.fftfreq(ny, dy)
    q2 = qy[:, None] ** 2 + qx[None, :] ** 2
    out = np.empty_like(J)
    for j in range(J.shape[2]):
        F = np.fft.fft2(J[..., j])
        out[..., j] = np.real(np.fft.ifft2(-F / (q2 + m * m)))
    return out


# ---------------------------------------------------------------------
# weak columns for all libraries, sampled at the query centres
# ---------------------------------------------------------------------

def build_columns(k1, k2, th_t, x, y, tt, args, Lsp, idx, A=None, Abar=None,
                  snap=None):
    """k1, k2, th_t: [ny, nx, nt]; A optional. Every weak column is
    sampled at idx = (IY, IX, IT) as soon as it is built, so only the
    pointwise inner fields are held as full stacks. Returns the column
    dicts (vectors), the targets (vectors) and, if snap is a frame index,
    a dict of pointwise fields at that frame for the surface figure."""
    IY, IX, IT = idx
    S = lambda F: F[IY, IX, IT]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    dt = float(np.median(np.diff(tt)))
    a = args.radius * Lsp
    fx, fx1, fx2, fx3, mx = kernel_d3(a, dx, args.poly_p)
    fy, fy1, fy2, fy3, my = kernel_d3(a, dy, args.poly_p)
    ht, ht1, ht2, ht3, mt = kernel_d3(args.t_half * dt, dt, args.poly_p)
    eta = args.eta * Lsp
    w = k1 ** 2 + k2 ** 2 - 1.0
    D = lambda F, ax: inner_deriv(F, eta, dx, dy, ax, "poly", args.poly_p)
    divk = D(k1, "x") + D(k2, "y")
    wx, wy = D(w, "x"), D(w, "y")
    lapw = D(wx, "x") + D(wy, "y")
    kgw = k1 * wx + k2 * wy
    kx1, kx2 = D(k1, "x"), D(k1, "y")            # grad k rows (k1_x, k1_y)
    ky1, ky2 = D(k2, "x"), D(k2, "y")
    curlk = ky1 - kx2
    Jk = kx1 * ky2 - kx2 * ky1

    K = lambda F, kxx, kyy, ktt: S(corr3(F, kxx, kyy, ktt, dx, dy, dt))
    n = len(IY)
    C = {}
    # RHS library
    C["1"] = np.ones(n)
    C["div k"] = -(K(k1, fx1, fy, ht) + K(k2, fx, fy1, ht))
    C["w div k"] = K(w * divk, fx, fy, ht)
    C["k.grad w"] = K(kgw, fx, fy, ht)
    # Int Lap div k phi = -Int k . grad(Lap phi); grad(Lap phi) components:
    # d/dx(phi_xx + phi_yy) = f3 g h + f1 g2 h ; d/dy(...) = f2 g1 h + f g3 h
    C["Lap div k"] = -(K(k1, fx3, fy, ht) + K(k1, fx1, fy2, ht)
                       + K(k2, fx2, fy1, ht) + K(k2, fx, fy3, ht))
    C["w"] = K(w, fx, fy, ht)
    C["Lap w"] = K(w, fx2, fy, ht) + K(w, fx, fy2, ht)
    C["(div k)^2"] = K(divk ** 2, fx, fy, ht)
    C["w^2 div k"] = K(w ** 2 * divk, fx, fy, ht)
    C["S"] = C["w div k"] + C["k.grad w"]                       # v1.7: div(w k), tied; never in a free library
    # extended columns (amplitude and defect channels)
    if A is not None:
        Ad = A - (Abar if Abar is not None else np.nanmedian(A))
        C["A_dev"] = K(Ad, fx, fy, ht)
        C["A_dev div k"] = K(Ad * divk, fx, fy, ht)
        kgA = k1 * D(A, "x") + k2 * D(A, "y")
        C["k.grad A"] = K(kgA, fx, fy, ht)
        a_floor = 0.25 * float(Abar if Abar is not None else np.nanmedian(A))   # v1.8
        wkglnA = w * kgA / np.maximum(A, a_floor)
        C[CN_COL] = K(wkglnA, fx, fy, ht)
        C["S_A"] = C["S"] + 2.0 * C[CN_COL]                      # div(A^2 w k)/A^2, one coefficient
        del kgA
    else:
        Ad = None
        wkglnA = None
        for nm in ("A_dev", "A_dev div k", "k.grad A", CN_COL, "S_A"):
            C[nm] = np.zeros(n)
    C["curl k"] = K(k2, fx1, fy, ht) - K(k1, fx, fy1, ht)          # fully weak
    C["J"] = K(Jk, fx, fy, ht)
    C["(curl k)^2"] = K(curlk ** 2, fx, fy, ht)
    chiJ = screened_potential(Jk, dx, dy, args.chi_m)               # v1.6
    C["chi_J"] = K(chiJ, fx, fy, ht)
    # energy library: columns = weak -delta e/delta theta
    # e = g(w): -delta e/delta theta = div(2 g'(w) k) -> -Int 2 g' k . grad phi
    def var_g(gp):
        return -2.0 * (K(gp * k1, fx1, fy, ht) + K(gp * k2, fx, fy1, ht))
    V = {}
    V["w^2"] = var_g(2.0 * w)
    V["w^3"] = var_g(3.0 * w ** 2)
    # e = (div k)^2: -delta/delta theta = -2 Lap div k -> +2 Int k . grad(Lap phi)
    V["(div k)^2"] = -2.0 * C["Lap div k"]
    # e = w (div k)^2: -delta/delta theta = div(2 k (div k)^2) - 2 Lap(w div k)
    V["w (div k)^2"] = (-2.0 * (K(divk ** 2 * k1, fx1, fy, ht)
                                + K(divk ** 2 * k2, fx, fy1, ht))
                        - 2.0 * (K(w * divk, fx2, fy, ht) + K(w * divk, fx, fy2, ht)))
    # e = |grad w|^2: -Int (delta e/delta theta) phi = Int d e/d k . grad phi
    #   - Int d e/d grad k : grad grad phi   (two IBPs), pointwise inner fields
    dedk1 = 4.0 * (kx1 * wx + ky1 * wy)
    dedk2 = 4.0 * (kx2 * wx + ky2 * wy)
    G11, G12 = 4.0 * wx * k1, 4.0 * wx * k2
    G21, G22 = 4.0 * wy * k1, 4.0 * wy * k2
    V["|grad w|^2"] = (K(dedk1, fx1, fy, ht) + K(dedk2, fx, fy1, ht)
                       - (K(G11, fx2, fy, ht) + K(G12, fx1, fy1, ht)
                          + K(G21, fx1, fy1, ht) + K(G22, fx, fy2, ht)))
    targets = {"theta_t": K(th_t, fx, fy, ht)}
    # identity target: w_t = 2 k . grad theta_t applied to the MEASURED
    # theta_t (a test of the estimator's gradient structure on the domain)
    lapdivk_pw = D(D(divk, "x"), "x") + D(D(divk, "y"), "y")
    # ---- v1.6: prediction-score data at the centres over the horizon
    P = None
    if int(getattr(args, "score_h", 0)) > 0:
        nt = th_t.shape[2]
        H = int(args.score_h)
        pw = {"1": np.ones_like(w), "div k": divk, "w div k": w * divk, "k.grad w": kgw,
              "Lap div k": lapdivk_pw, "w": w, "Lap w": lapw, "(div k)^2": divk ** 2,
              "w^2 div k": w ** 2 * divk, "curl k": curlk, "J": Jk, "(curl k)^2": curlk ** 2,
              "chi_J": chiJ, "S": w * divk + kgw}
        if Ad is not None:
            pw.update({"A_dev": Ad, "A_dev div k": Ad * divk,
                       "k.grad A": k1 * D(A, "x") + k2 * D(A, "y"),
                       CN_COL: wkglnA, "S_A": w * divk + kgw + 2.0 * wkglnA})
        ok = (IT + H) <= nt - 1
        fr = np.arange(H + 1)
        ITc = np.minimum(IT[:, None] + fr[None, :], nt - 1)                       # [n, H+1]
        P = dict(ok=ok, H=H, dt=dt, names=list(pw),
                 pw={nm: F[IY[:, None], IX[:, None], ITc] for nm, F in pw.items()},
                 th_t_pw=th_t[IY[:, None], IX[:, None], ITc])
        # spatially weak versions (bump in x, y only), linear terms fully weak
        K2 = lambda F, kxx, kyy: corr1d(corr1d(F, kyy, 0, dy), kxx, 1, dx)
        def spatial(nm, f):
            if nm == "div k":
                return -(K2(k1[..., f], fx1, fy) + K2(k2[..., f], fx, fy1))
            if nm == "Lap div k":
                return -(K2(k1[..., f], fx3, fy) + K2(k1[..., f], fx1, fy2)
                         + K2(k2[..., f], fx2, fy1) + K2(k2[..., f], fx, fy3))
            if nm == "Lap w":
                return K2(w[..., f], fx2, fy) + K2(w[..., f], fx, fy2)
            if nm == "curl k":
                return K2(k2[..., f], fx1, fy) - K2(k1[..., f], fx, fy1)
            return K2(pw[nm][..., f], fx, fy)
        need = sorted(set(ITc.ravel().tolist()))
        wk = {nm: np.zeros(ITc.shape) for nm in pw}
        wk_tt = np.zeros(ITc.shape)
        for f in need:
            hit = ITc == f
            if not hit.any():
                continue
            rows, cols_ = np.where(hit)
            for nm in pw:
                G = spatial(nm, f)
                wk[nm][rows, cols_] = G[IY[rows], IX[rows]]
            G = K2(th_t[..., f], fx, fy)
            wk_tt[rows, cols_] = G[IY[rows], IX[rows]]
        P["wk"] = wk; P["th_t_wk"] = wk_tt
    kg = lambda F: 2.0 * (k1 * D(F, "x") + k2 * D(F, "y"))
    targets["w_t_identity"] = K(kg(th_t), fx, fy, ht)

    # ---- w_t library (all nonlinear columns pointwise, then outer integral)
    kmag2 = np.maximum(k1 ** 2 + k2 ** 2, 1e-12)
    wxx, wxy, wyy = D(wx, "x"), D(wx, "y"), D(wy, "y")
    kkddw = k1 * k1 * wxx + 2 * k1 * k2 * wxy + k2 * k2 * wyy      # unnormalised
    w_nn = lapw - kkddw / kmag2
    del wxx, wxy, wyy
    gradk2 = kx1 ** 2 + kx2 ** 2 + ky1 ** 2 + ky2 ** 2
    lapdivk = D(D(divk, "x"), "x") + D(D(divk, "y"), "y")
    kglapdiv = k1 * D(lapdivk, "x") + k2 * D(lapdivk, "y")
    del lapdivk
    Wc = {}
    Wc["1"] = np.ones(n)
    Wc["w"] = C["w"]
    Wc["w^2"] = K(w ** 2, fx, fy, ht)
    Wc["kk:ddw"] = K(kkddw, fx, fy, ht)
    Wc["w_nn"] = K(w_nn, fx, fy, ht)
    Wc["|grad w|^2"] = K(wx ** 2 + wy ** 2, fx, fy, ht)
    Wc["(k.grad w) div k"] = K(kgw * divk, fx, fy, ht)
    Wc["w Lap w"] = K(w * lapw, fx, fy, ht)
    Wc["w |grad k|^2"] = K(w * gradk2, fx, fy, ht)
    Wc["(div k)^2"] = C["(div k)^2"]
    Wc["k.grad Lap div k"] = K(kglapdiv, fx, fy, ht)
    Wc["Lap w"] = C["Lap w"]                                       # forced use only
    Wc["k.grad w"] = C["k.grad w"]                                 # consistency use only
    Wc["A_dev"] = C["A_dev"]
    w_t = np.gradient(w, dt, axis=2)
    targets["w_t"] = K(w_t, fx, fy, ht)
    del w_t, kkddw, w_nn, gradk2, kglapdiv

    # ---- A_t library
    if A is not None:
        Ac = {}
        Ac["1"] = np.ones(n)
        Ac["A"] = K(A, fx, fy, ht)
        Ac["A^3"] = K(A ** 3, fx, fy, ht)
        Ac["Lap A"] = K(A, fx2, fy, ht) + K(A, fx, fy2, ht)
        Ac["w"] = C["w"]
        Ac["A w"] = K(A * w, fx, fy, ht)
        Ac["A^2"] = K(A ** 2, fx, fy, ht)
        Ac["A w^2"] = K(A * w ** 2, fx, fy, ht)
        A_t = np.gradient(A, dt, axis=2)
        targets["A_t"] = K(A_t, fx, fy, ht)
        del A_t
    else:
        Ac = None
    fields = None
    if snap is not None:
        j = int(snap)
        fields = {"theta_t": th_t[..., j], "w": w[..., j], "div k": divk[..., j],
                  "w div k": (w * divk)[..., j], "k.grad w": kgw[..., j],
                  "Lap div k": lapdivk_pw[..., j],
                  "curl k": curlk[..., j], "J": Jk[..., j]}
        if Ad is not None:
            fields["A - Abar"] = Ad[..., j]
    return dict(rhs=C, var=V, w=Wc, A=Ac, score=P), targets, fields


def theta_t_from_z(zs, dt):
    """Im(conj z z_t)/|z|^2 with centred differences inside and one-sided
    differences at the two ends (an FFT correlation cannot tolerate NaN)"""
    z_t = np.empty_like(zs)
    z_t[..., 1:-1] = (zs[..., 2:] - zs[..., :-2]) / (2 * dt)
    z_t[..., 0] = (zs[..., 1] - zs[..., 0]) / dt
    z_t[..., -1] = (zs[..., -1] - zs[..., -2]) / dt
    return np.imag(np.conj(zs) * z_t) / np.maximum(np.abs(zs) ** 2, 1e-14)


def box_dilate(mask, hy, hx, ht=0):
    """v1.7: binary dilation by the box of half widths (hy, hx, ht), border
    0; a separable maximum filter, identical to binary_dilation with
    np.ones((2hy+1, 2hx+1, 2ht+1)) and far cheaper for kernel-sized boxes"""
    return maximum_filter(mask.astype(np.uint8), size=(2 * hy + 1, 2 * hx + 1, 2 * ht + 1),
                          mode="constant", cval=0).astype(bool)


def query_centres(shape, m, args, excl=None, cores=None):
    """decimated grid of query centres with margins >= kernel half widths;
    excl: boolean [ny, nx, nt] of points whose test-function support may
    not touch (dilated by the half widths when excl_dilate).
    v1.7: with --dilate_cores_only, cores (the core discs alone) is
    dilated by the full support (x, y and t) and excl (domain complement
    included) removes centres only."""
    ny, nx, nt = shape
    mx, my, mt = m
    ix = np.arange(mx + args.rim, nx - mx - args.rim, args.stride_xy)
    iy = np.arange(my + args.rim, ny - my - args.rim, args.stride_xy)
    it = np.arange(mt, nt - mt, args.stride_t)
    if len(ix) == 0 or len(iy) == 0 or len(it) == 0:
        return None
    IY, IX, IT = np.meshgrid(iy, ix, it, indexing="ij")
    keep = np.ones(IY.shape, bool)
    if excl is not None:
        if int(getattr(args, "dilate_cores_only", 0)) and cores is not None:
            keep &= ~excl[IY, IX, IT]
            if cores.any():
                big = box_dilate(cores, my, mx, mt)
                keep &= ~big[IY, IX, IT]
        elif args.excl_dilate:
            big = box_dilate(excl, my, mx, 0)          # v1.6 behaviour (time extent 1)
            keep &= ~big[IY, IX, IT]
        else:
            keep &= ~excl[IY, IX, IT]
    if keep.sum() == 0:
        return None
    return IY[keep], IX[keep], IT[keep]


def assemble(cols, names, b, xs):
    Theta = np.column_stack([cols[nm] for nm in names])
    fin = np.all(np.isfinite(Theta), axis=1) & np.isfinite(b)
    return Theta[fin], b[fin], xs[fin]


def collinearity(names, Theta):
    """condition number of the column-standardised library and its most
    correlated pair: a reading of how identifiable the coefficients are
    (a pair above 0.95 is interchangeable; quote it jointly)"""
    keep = [j for j in range(Theta.shape[1]) if np.std(Theta[:, j]) > 0]
    if len(keep) < 2:
        return dict(cond=np.nan, max_corr=np.nan, pair=("", ""))
    Z = Theta[:, keep]
    Z = (Z - Z.mean(0)) / Z.std(0)
    C = np.corrcoef(Z, rowvar=False)
    np.fill_diagonal(C, 0.0)
    i, j = np.unravel_index(np.argmax(np.abs(C)), C.shape)
    return dict(cond=float(np.linalg.cond(Z)), max_corr=float(C[i, j]),
                pair=(names[keep[i]], names[keep[j]]))


# ---------------------------------------------------------------------
# selection (fit_target -> pairs -> Pareto -> contribution floor)
# ---------------------------------------------------------------------

def _refit(Theta, bv, supp):
    coef = np.zeros(Theta.shape[1])
    idx = np.where(supp)[0]
    if len(idx):
        coef[idx], *_ = np.linalg.lstsq(Theta[:, idx], bv, rcond=None)
    return coef


def fit_pareto(names, Theta, bv, xs, rng, rel_tol, min_contrib, max_samples=60000, abs_tol=0.0):
    """Sparse selection. Admissible = holdout R2 within max(rel_tol |best|,
    abs_tol) of the best; the chosen model is the sparsest admissible
    one (ties: highest R2). Also returns the ladder (best support at
    every size present in the candidate pool) and the admissible
    alternatives of the chosen size, so an interchangeable pair (a
    collinear alternative within tolerance) is visible in the report."""
    if len(bv) < 200:
        return dict(skipped=True, reason="too few rows (%d)" % len(bv))
    if len(bv) > max_samples:
        sel = rng.choice(len(bv), max_samples, replace=False)
        Theta, bv, xs = Theta[sel], bv[sel], xs[sel]
    tr = xs < np.median(xs)
    res = fit_target(names, Theta, bv, _FitArgs(rel_tol), rng, tr)
    sweep = [s for s in res["sweep"] if np.isfinite(s["r2_test"])]
    p = Theta.shape[1]
    for i in range(p):
        for j in range(i + 1, p):
            supp = np.zeros(p, bool); supp[[i, j]] = True
            coef = np.zeros(p)
            coef[[i, j]], *_ = np.linalg.lstsq(Theta[tr][:, [i, j]], bv[tr], rcond=None)
            sweep.append(dict(support=supp, coef=coef, n_terms=2,
                              r2_train=r_squared(bv[tr], Theta[tr] @ coef),
                              r2_test=r_squared(bv[~tr], Theta[~tr] @ coef)))
    sweep = [s for s in sweep if np.isfinite(s["r2_test"]) and s["n_terms"] > 0]   # v1.8: see changelog (5)
    if not sweep:
        return dict(skipped=True, reason="no finite holdout R2 in the candidate pool")
    best = max(s["r2_test"] for s in sweep)
    tol = max(rel_tol * abs(best), abs_tol)
    adm = [s for s in sweep if s["n_terms"] > 0 and s["r2_test"] >= best - tol]
    adm.sort(key=lambda s: (s["n_terms"], -s["r2_test"]))
    chosen = adm[0]
    ladder = {}
    for s_ in sweep:
        n_ = int(s_["n_terms"])
        if n_ > 0 and (n_ not in ladder or s_["r2_test"] > ladder[n_][1]):
            ladder[n_] = ([names[i] for i in np.where(s_["support"])[0]], float(s_["r2_test"]))
    alts = []
    seen = {tuple(np.where(chosen["support"])[0])}
    for s_ in adm:
        key = tuple(np.where(s_["support"])[0])
        if s_["n_terms"] == chosen["n_terms"] and key not in seen:
            seen.add(key)
            alts.append(([names[i] for i in key], float(s_["r2_test"])))
    supp = chosen["support"].copy()
    coef = _refit(Theta, bv, supp)
    if min_contrib > 0:
        sb = max(float(np.std(bv)), 1e-30)
        for _ in range(p):
            contrib = np.abs(coef) * np.std(Theta, axis=0) / sb
            weak = supp & (contrib < min_contrib)
            if not weak.any() or supp.sum() <= 1:
                break
            j = np.where(weak)[0][np.argmin(contrib[weak])]
            supp[j] = False
            coef = _refit(Theta, bv, supp)
    coefs = {names[i]: float(coef[i]) for i in np.where(supp)[0]}
    cf = np.zeros(p); cf[supp], *_ = np.linalg.lstsq(Theta[tr][:, supp], bv[tr], rcond=None)
    return dict(n=int(len(bv)), coefs=coefs, support=sorted(coefs),
                r2_full=float(r_squared(bv, Theta @ coef)),
                r2_holdout=float(r_squared(bv[~tr], Theta[~tr] @ cf)),
                best_r2=float(best),
                dense_r2=float(r_squared(bv[~tr], Theta[~tr] @ _refit(Theta[tr], bv[tr], np.ones(p, bool)))),
                names=list(names), tol=float(tol),
                ladder={int(k): v for k, v in sorted(ladder.items())}, alternatives=alts[:4])


def forced_fit(names, Theta, bv, xs, want):
    tr = xs < np.median(xs)
    idx = [names.index(n) for n in want]
    c, *_ = np.linalg.lstsq(Theta[tr][:, idx], bv[tr], rcond=None)
    cf = np.zeros(Theta.shape[1]); cf[idx] = c
    return dict(n=int(len(bv)), coefs={n: float(v) for n, v in zip(want, c)},
                r2_holdout=float(r_squared(bv[~tr], Theta[~tr] @ cf)), names=list(names))


def exhaustive_supports(names, Theta, bv, xs, sizes=(3, 4), keep_top=40):
    """v1.7: least squares on EVERY support of the given sizes, same split
    as forced_fit (train x < median, holdout R2 on the rest, all rows).
    Columns with zero RMS (absent amplitude on the control) are skipped.
    Returns {size: [(r2_holdout, (names...), {name: coef}), ...]} sorted
    by holdout R2 (top keep_top kept) plus, per size, the full list of
    (r2, support) for ranking the references."""
    tr = xs < np.median(xs)
    Ttr, btr = np.ascontiguousarray(Theta[tr]), bv[tr]
    Tte, bte = np.ascontiguousarray(Theta[~tr]), bv[~tr]
    usable = [j for j in range(Theta.shape[1]) if np.sqrt(np.mean(Theta[:, j] ** 2)) > 0]
    out = {}
    for k in sizes:
        rows = []
        for idx in combinations(usable, k):
            idx = list(idx)
            c, *_ = np.linalg.lstsq(Ttr[:, idx], btr, rcond=None)
            r2 = float(r_squared(bte, Tte[:, idx] @ c))
            rows.append((r2, tuple(names[i] for i in idx), c))
        rows.sort(key=lambda r: -r[0] if np.isfinite(r[0]) else np.inf)
        out[k] = dict(n_supports=len(rows),
                      top=[(r2, sp, {nm: float(v) for nm, v in zip(sp, c)}) for r2, sp, c in rows[:keep_top]],
                      all=[(r2, sp) for r2, sp, _ in rows],
                      coefs={sp: {nm: float(v) for nm, v in zip(sp, c)} for r2, sp, c in rows})
    return out


def exhaustive_report(ex, names, lines, args, forced_r2=None, top=8):
    """v1.7: compact report of exhaustive_supports; returns a json-able
    summary. forced_r2: {support tuple (sorted): R2 of a forced block}
    for the equality check."""
    summ = {}
    for k, E in sorted(ex.items()):
        allr = E["all"]
        if not allr:
            continue
        best = allr[0][0]
        tol = max(args.rel_tol * abs(best), args.abs_tol)
        ntie = sum(1 for r2, _ in allr if r2 >= best - tol)
        rank = {tuple(sorted(sp)): (i + 1, r2) for i, (r2, sp) in enumerate(allr)}
        lines.append("  theta_t | RHS%s | exhaustive size %d (%d supports, all rows, forced_fit split):"
                     " best %.4f; %d within tol %.3g of the best" %
                     ("+ext" if len(names) > len(RHS_LIB) else "", k, E["n_supports"], best, ntie, tol))
        for i, (r2, sp, cf) in enumerate(E["top"][:top]):
            lines.append("      %2d  %.4f  %s" % (i + 1, r2, cstr(cf)))
        def first(pred):
            for i, (r2, sp) in enumerate(allr):
                if pred(sp):
                    return i + 1, r2, sp
            return None
        cells = []
        for lab, pred in (("best with J", lambda sp: "J" in sp),
                          ("best with Lap div k", lambda sp: "Lap div k" in sp),
                          ("best without J or Lap div k", lambda sp: "J" not in sp and "Lap div k" not in sp),
                          ("best without amplitude columns", lambda sp: not any(a in sp for a in AMP_COLS)),
                          ("best with J and Lap div k", lambda sp: "J" in sp and "Lap div k" in sp)):
            f = first(pred)
            if f is None:
                continue
            cf = E["coefs"][f[2]]
            cells.append("%s: #%d %.4f %s" % (lab, f[0], f[1], cstr(cf)))
            summ.setdefault(k, {})[lab] = dict(rank=f[0], r2=f[1], coefs=cf)
        for c_ in cells:
            lines.append("      " + c_)
        # bending sign among the supports carrying Lap div k within tol and in the top 20
        bend = [E["coefs"][sp]["Lap div k"] for r2, sp in allr if "Lap div k" in sp and r2 >= best - tol]
        bend20 = [E["coefs"][sp]["Lap div k"] for r2, sp in allr[:20] if "Lap div k" in sp]
        if bend or bend20:
            rng_ = lambda v: ("[%+.3g, %+.3g]" % (min(v), max(v))) if v else "-"
            lines.append("      bending sign (RCN: negative): within tol %d positive / %d, range %s;"
                         " top 20 %d positive / %d, range %s"
                         % (sum(v > 0 for v in bend), len(bend), rng_(bend),
                            sum(v > 0 for v in bend20), len(bend20), rng_(bend20)))
        incl = {}
        for r2, sp in allr[:max(top, 1)]:
            for nm in sp:
                incl[nm] = incl.get(nm, 0) + 1
        lines.append("      inclusion in the top %d: %s" % (top, ", ".join(
            "%s %d" % (nm, v) for nm, v in sorted(incl.items(), key=lambda kv: -kv[1]))))
        refs = []
        for ref in EXH_REFS:
            if len(ref) != k or any(nm not in names for nm in ref):
                continue
            key = tuple(sorted(ref))
            if key in rank:
                rk, r2 = rank[key]
                chk = ""
                if forced_r2 and key in forced_r2 and np.isfinite(forced_r2[key]):
                    chk = " (forced block %.4f)" % forced_r2[key]
                refs.append("{%s} #%d %.4f%s" % (", ".join(ref), rk, r2, chk))
                summ.setdefault(k, {})["ref {%s}" % ", ".join(ref)] = dict(rank=rk, r2=r2)
        if refs:
            lines.append("      references: " + "; ".join(refs))
        summ.setdefault(k, {})["best_r2"] = best
        summ[k]["n_within_tol"] = ntie
        summ[k]["top"] = [dict(r2=r2, coefs=cf) for r2, sp, cf in E["top"][:top]]
    return summ


def cstr(coefs):
    return " + ".join("%+.4g %s" % (v, k) for k, v in coefs.items()) if coefs else "(empty)"


def fmt(r, truth=None):
    if r.get("skipped"):
        return "SKIPPED (%s)" % r.get("reason")
    s = cstr(r["coefs"])
    s += "   holdout R2 %.4f (best any %.4f, dense %.4f, n %d)" % (
        r["r2_holdout"], r["best_r2"], r["dense_r2"], r["n"])
    if truth:
        errs = []
        for k, v in truth.items():
            if k in r["coefs"]:
                errs.append("%s %.1f%%" % (k, 100 * abs(r["coefs"][k] - v) / abs(v)))
            else:
                errs.append("%s MISSING" % k)
        extra = [k for k in r["coefs"] if k not in truth]
        s += "   vs truth: " + ", ".join(errs) + (("; extras " + str(extra)) if extra else "")
    if r.get("ladder"):
        s += "\n" + " " * 46 + "ladder (best holdout R2 per size): " + "; ".join(
            "%d: %.3f {%s}" % (n_, v[1], ", ".join(v[0])) for n_, v in r["ladder"].items() if n_ <= 5)
    if r.get("alternatives"):
        s += "\n" + " " * 46 + "same-size alternatives within tol %.3g: " % r["tol"] + "; ".join(
            "%.3f {%s}" % (v[1], ", ".join(v[0])) for v in r["alternatives"])
    return s


def fig_fit(Theta, bv, names, coefs, path, title, target):
    """holdout figure of one block: weak target vs the block's model"""
    cf = np.array([coefs.get(nm, 0.0) for nm in names])
    pred = Theta @ cf
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot(bv, pred, ".", ms=2, alpha=0.4)
    lim = np.percentile(np.abs(bv), 99.5) if len(bv) else 1.0
    lim = lim if lim > 0 else 1.0
    ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("weak target Int[%s phi]" % target); ax.set_ylabel("model")
    ax.set_title(title, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _slug(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")


def prediction_scores(P, laws, xs, args, lines):
    """v1.6: error ratio to the null of each law's phase-increment
    prediction over horizons h (frames), holdout half in x, centres whose
    horizon stays inside the window. Returns dict law -> h -> scores."""
    if P is None:
        return {}
    ok = P["ok"] & (xs >= np.median(xs))
    if ok.sum() < 50:
        lines.append("  prediction score: too few centres with a full horizon (%d); skipped" % int(ok.sum()))
        return {}
    dt, H = P["dt"], P["H"]
    horizons = sorted({h for h in (args.t_half, 2 * args.t_half, H) if 0 < h <= H})
    trap = lambda F, h: dt * (0.5 * F[:, 0] + F[:, 1:h].sum(1) + 0.5 * F[:, h])
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
    out = {}
    lines.append("  prediction score (v1.6): rms(Delta theta - prediction) / rms(Delta theta) on %d holdout"
                 " centres; null = 1; columns: h frames | pointwise Euler, multi-step | weak Euler, multi-step"
                 " [read the WEAK multi-step column: pointwise fourth derivatives are discretisation on the"
                 " control's crease (0.8 exact route) and noise on the estimator (>70); control exact route:"
                 " forced RCN 0.001, theory 0.014; estimator route: forced RCN 0.21, free 0.09, theory 1.13]" % int(ok.sum()))
    for lname, coefs in laws.items():
        if not coefs:
            continue
        miss = [k for k in coefs if k not in P["pw"]]
        if miss:
            lines.append("    %-34s (terms without pointwise fields: %s; not scored)" % (lname, miss))
            continue
        law_pw = sum(c * P["pw"][k][ok] for k, c in coefs.items())
        law_wk = sum(c * P["wk"][k][ok] for k, c in coefs.items())
        cells, out[lname] = [], {}
        for h in horizons:
            dth_pw = trap(P["th_t_pw"][ok], h); dth_wk = trap(P["th_t_wk"][ok], h)
            e_pw = rms(dth_pw - h * dt * law_pw[:, 0]) / max(rms(dth_pw), 1e-300)
            m_pw = rms(dth_pw - trap(law_pw, h)) / max(rms(dth_pw), 1e-300)
            e_wk = rms(dth_wk - h * dt * law_wk[:, 0]) / max(rms(dth_wk), 1e-300)
            m_wk = rms(dth_wk - trap(law_wk, h)) / max(rms(dth_wk), 1e-300)
            out[lname][h] = dict(euler_pw=e_pw, multi_pw=m_pw, euler_wk=e_wk, multi_wk=m_wk)
            cells.append("h %d | %.3f, %.3f | %.3f, %.3f" % (h, e_pw, m_pw, e_wk, m_wk))
        lines.append("    %-34s %s" % (lname + ":", "  ||  ".join(cells)))
    return out


# ---------------------------------------------------------------------
# one full block of fits (theta_t RHS/ENERGY, w_t, A_t) on assembled rows
# ---------------------------------------------------------------------

def fit_blocks(R, tag, args, rng, lines, out, truth=None, bias=None, route=""):
    """R: dict with keys rhs, var, w, A (column dicts of vectors), targets,
    xs. tag: report label. truth: dict of truth dicts on the control.
    bias: dict of factor dicts (data) for the reported correction."""
    truth = truth or {}
    fits = {}
    sfx = ("_" + route) if route else ""
    xs = R["xs"]

    def block(target, libname, names, cols, b, kind, want=None, note="", tr_truth=None, fname=None):
        Th, bv, xv = assemble(cols, names, b, xs)
        label = "%s | %s library | %s" % (target, libname, kind)
        if want is None:
            r = fit_pareto(names, Th, bv, xv, rng, args.rel_tol, args.min_contrib, abs_tol=args.abs_tol)
            cl = collinearity(names, Th)
            r["collinearity"] = cl
            lines.append("  %-42s %s" % (label + ":", fmt(r, tr_truth)))
            lines.append(" " * 46 + "library conditioning: cond %.3g, most correlated pair |r| %.3f {%s, %s}" %
                         (cl["cond"], abs(cl["max_corr"]), cl["pair"][0], cl["pair"][1]))
        else:
            if len(bv) < 20:
                r = dict(skipped=True, reason="too few rows (%d)" % len(bv))
                lines.append("  %-42s %s" % (label + ":", fmt(r)))
                return r
            r = forced_fit(names, Th, bv, xv, want)
            cl = collinearity(want, Th[:, [names.index(nm) for nm in want]])
            r["collinearity"] = cl
            lines.append("  %-42s %s   holdout R2 %.4f (n %d)%s   [support cond %.3g, max |r| %.3f {%s, %s}]" %
                         (label + ":", cstr(r["coefs"]), r["r2_holdout"], r["n"], note,
                          cl["cond"], abs(cl["max_corr"]), cl["pair"][0], cl["pair"][1]))
        if not r.get("skipped"):
            fig_fit(Th, bv, names, r["coefs"],
                    out / ("fit_%s_%s_%s%s.png" % (_slug(target), _slug(libname), fname or _slug(kind), sfx)),
                    "%s\n%s\nholdout R2 %.3f, n %d" % (tag, label, r["r2_holdout"], r["n"]), target)
        return r

    C, V, Wc, Ac, T = R["rhs"], R["var"], R["w"], R["A"], R["targets"]
    b = T["theta_t"]
    lines.append("  target scale: RMS Int[theta_t phi] = %.3e over %d rows" %
                 (float(np.sqrt(np.mean(b ** 2))), len(b)))
    fits["target_rms"] = float(np.sqrt(np.mean(b ** 2)))                # v1.8
    fits["n_rows"] = int(len(b))
    fits["theta_rhs_free"] = block("theta_t", "RHS", RHS_LIB, C, b, "free", tr_truth=truth.get("rhs"))
    if args.lib_ext and Ac is not None:
        fits["theta_rhs_ext_free"] = block("theta_t", "RHS+ext", RHS_LIB + RHS_EXT, C, b, "free")
        if int(getattr(args, "free_no_amp", 0)):                          # v1.7
            NOAMP = [nm for nm in RHS_LIB + RHS_EXT if nm not in AMP_COLS]
            fits["theta_rhs_ext_noamp_free"] = block("theta_t", "RHS+ext-noA", NOAMP, C, b, "free",
                                                     fname="free_noamp")
    f = block("theta_t", "RHS", RHS_LIB, C, b, "forced {w div k, k.grad w, Lap div k}", RCN_RHS, fname="forced_rcn")
    fits["theta_rhs_forced"] = f
    if bias and not f.get("skipped"):
        corr = {k: f["coefs"][k] * bias["rhs_factor"][k] for k in RCN_RHS if k in bias["rhs_factor"]}
        lines.append("  %-42s %s   (factors truth/estimator from the control at radius %.2g: %s;"
                     " reported, not applied)" %
                     ("theta_t | RHS | forced, crease-bias corrected:", cstr(corr), bias["radius"],
                      ", ".join("%s %.2f" % (k, v) for k, v in bias["rhs_factor"].items())))
        fits["theta_rhs_forced_bias_corrected"] = corr
    if args.lib_ext and Ac is not None:
        fits["theta_rhs_forced_plus_A"] = block(
            "theta_t", "RHS+ext", RHS_LIB + RHS_EXT, C, b,
            "forced {w div k, k.grad w, Lap div k, A_dev, A_dev div k, k.grad A}",
            RCN_RHS + ["A_dev", "A_dev div k", "k.grad A"], fname="forced_rcn_plus_A")
        fits["theta_rhs_forced_plus_defect"] = block(
            "theta_t", "RHS+ext", RHS_LIB + RHS_EXT, C, b,
            "forced {w div k, k.grad w, Lap div k, curl k, J}", RCN_RHS + ["curl k", "J"], fname="forced_rcn_plus_defect")
        # v1.6: three-term competitors at the size of the RCN law
        SP = ["w div k", "k.grad w"]
        fits["theta_rhs_forced_SJ"] = block("theta_t", "RHS+ext", RHS_LIB + RHS_EXT, C, b,
                                            "forced {w div k, k.grad w, J}", SP + ["J"], fname="forced_SJ")
        fits["theta_rhs_forced_SA"] = block("theta_t", "RHS+ext", RHS_LIB + RHS_EXT, C, b,
                                            "forced {w div k, k.grad w, A_dev div k}", SP + ["A_dev div k"], fname="forced_SA")
        fits["theta_rhs_forced_SchiJ"] = block("theta_t", "RHS+ext", RHS_LIB + RHS_EXT, C, b,
                                               "forced {w div k, k.grad w, chi_J}", SP + ["chi_J"], fname="forced_SchiJ")
        fits["theta_rhs_forced_plus_chiJ"] = block("theta_t", "RHS+ext", RHS_LIB + RHS_EXT, C, b,
                                                   "forced {w div k, k.grad w, Lap div k, J, chi_J}", RCN_RHS + ["J", "chi_J"], fname="forced_rcn_plus_J_chiJ")
        # v1.7: RCN + J alone (the size-4 reference) and the tied-S laws
        fits["theta_rhs_forced_rcn_J"] = block("theta_t", "RHS+ext", RHS_LIB + RHS_EXT, C, b,
                                               "forced {w div k, k.grad w, Lap div k, J}", RCN_RHS + ["J"],
                                               fname="forced_rcn_J")
        LT = RHS_LIB + RHS_EXT + ["S"]
        fits["theta_tied_SJ"] = block("theta_t", "RHS+ext+S", LT, C, b, "forced tied {S, J}", ["S", "J"],
                                      fname="forced_tied_SJ")
        fits["theta_tied_SB"] = block("theta_t", "RHS+ext+S", LT, C, b, "forced tied {S, Lap div k}",
                                      ["S", "Lap div k"], fname="forced_tied_SB")
        fits["theta_tied_SBJ"] = block("theta_t", "RHS+ext+S", LT, C, b, "forced tied {S, Lap div k, J}",
                                       ["S", "Lap div k", "J"], fname="forced_tied_SBJ")
        # v1.8: amplitude-gradient laws and the Cross-Newell one-coefficient law
        LC = LT + [CN_COL, "S_A"]
        fits["theta_rhs_forced_SkA"] = block("theta_t", "RHS+ext", LC, C, b, "forced {w div k, k.grad w, k.grad A}",
                                             SP + ["k.grad A"], fname="forced_SkA")
        fits["theta_tied_SkA"] = block("theta_t", "RHS+ext+S", LC, C, b, "forced tied {S, k.grad A}",
                                       ["S", "k.grad A"], fname="forced_tied_SkA")
        if int(getattr(args, "lib_cn", 1)):
            fits["theta_rhs_forced_SwlnA"] = block("theta_t", "RHS+ext+CN", LC, C, b,
                                                   "forced {w div k, k.grad w, w k.grad lnA}", SP + [CN_COL],
                                                   fname="forced_SwlnA")
            fits["theta_cn_SA"] = block("theta_t", "RHS+ext+CN", LC, C, b, "forced CN {S_A}", ["S_A"],
                                        note="   (S_A = div(A^2 w k)/A^2 = S + 2 w k.grad lnA)", fname="forced_cn_SA")
            fits["theta_cn_SAJ"] = block("theta_t", "RHS+ext+CN", LC, C, b, "forced CN {S_A, J}", ["S_A", "J"],
                                         fname="forced_cn_SAJ")
            fits["theta_cn_SAB"] = block("theta_t", "RHS+ext+CN", LC, C, b, "forced CN {S_A, Lap div k}",
                                         ["S_A", "Lap div k"], fname="forced_cn_SAB")
    # v1.7: exhaustive supports over the theta_t library of this run
    if int(getattr(args, "exhaustive", 0)):
        exnames = RHS_LIB + RHS_EXT if (args.lib_ext and Ac is not None) else list(RHS_LIB)
        Th_, bv_, xv_ = assemble(C, exnames, b, xs)
        sizes = tuple(int(v) for v in str(args.exh_sizes).split(",") if v.strip())
        if len(bv_) >= 200:
            t_ex = time.time()
            ex = exhaustive_supports(exnames, Th_, bv_, xv_, sizes=sizes)
            fr2 = {}
            for key in ("theta_rhs_forced", "theta_rhs_forced_SJ", "theta_rhs_forced_SchiJ", "theta_rhs_forced_SA",
                        "theta_rhs_forced_rcn_J", "theta_rhs_forced_SkA", "theta_rhs_forced_SwlnA"):
                ft = fits.get(key)
                if ft and not ft.get("skipped") and ft.get("coefs"):
                    fr2[tuple(sorted(ft["coefs"]))] = ft["r2_holdout"]
            fits["theta_exhaustive"] = exhaustive_report(ex, exnames, lines, args, forced_r2=fr2,
                                                         top=int(args.exh_top))
            for k_ in sizes:
                if ex.get(k_) and ex[k_]["top"]:
                    fits["theta_exh_best_%d" % k_] = dict(coefs=ex[k_]["top"][0][2], r2_holdout=ex[k_]["top"][0][0])
            lines.append("      (exhaustive search %.1f s)" % (time.time() - t_ex))
            # v1.8: size-3 search with the Cross-Newell column added
            if int(getattr(args, "lib_cn", 1)) and args.lib_ext and Ac is not None and 3 in sizes:
                cnn = exnames + [CN_COL]
                Thc, bvc, xvc = assemble(C, cnn, b, xs)
                exc = exhaustive_supports(cnn, Thc, bvc, xvc, sizes=(3,))
                lines.append("  [with the Cross-Newell column %s]" % CN_COL)
                fits["theta_exhaustive_cn"] = exhaustive_report(exc, cnn, lines, args, forced_r2=fr2,
                                                                top=int(args.exh_top))
        else:
            lines.append("  theta_t | exhaustive: too few rows (%d); skipped" % len(bv_))
    # residual block: fix the forced RCN coefficients and fit what is left
    # sparsely against the non-RCN columns (the departure from RCN, read
    # as an experimenter who takes RCN as the base model)
    if not f.get("skipped"):
        resid_names = [nm for nm in RHS_LIB if nm not in RCN_RHS] + (RHS_EXT if (args.lib_ext and Ac is not None) else [])
        pred_rcn = sum(f["coefs"][nm] * C[nm] for nm in RCN_RHS)
        fits["theta_residual_free"] = block("theta_t - RCN forced", "residual", resid_names, C,
                                            b - pred_rcn, "free", fname="residual")
    # v1.6: prediction scores of the theta laws
    laws = {"theory (2, 2, -1) [alpha=eta=tau]": (truth.get("rhs") if truth else {"w div k": 2.0, "k.grad w": 2.0, "Lap div k": -1.0})}
    for key, lab in (("theta_rhs_forced", "forced RCN"), ("theta_rhs_forced_plus_defect", "forced RCN + {curl k, J}"),
                     ("theta_rhs_forced_plus_A", "forced RCN + amplitude"), ("theta_rhs_forced_SJ", "forced {S, J}"),
                     ("theta_rhs_forced_SA", "forced {S, A_dev div k}"), ("theta_rhs_forced_SchiJ", "forced {S, chi_J}"),
                     ("theta_rhs_forced_plus_chiJ", "forced RCN + {J, chi_J}"),
                     ("theta_rhs_forced_rcn_J", "forced RCN + J"),
                     ("theta_tied_SJ", "forced tied {S, J}"), ("theta_tied_SB", "forced tied {S, Lap div k}"),
                     ("theta_tied_SBJ", "forced tied {S, Lap div k, J}"),
                     ("theta_exh_best_3", "exhaustive best size 3"), ("theta_exh_best_4", "exhaustive best size 4"),
                     ("theta_rhs_ext_noamp_free", "free RHS+ext-noA"),
                     ("theta_rhs_forced_SkA", "forced {S pair, k.grad A}"), ("theta_tied_SkA", "forced tied {S, k.grad A}"),
                     ("theta_rhs_forced_SwlnA", "forced {S pair, w k.grad lnA}"), ("theta_cn_SA", "forced CN {S_A}"),
                     ("theta_cn_SAJ", "forced CN {S_A, J}"), ("theta_cn_SAB", "forced CN {S_A, Lap div k}"),
                     ("theta_rhs_free", "free RHS"), ("theta_rhs_ext_free", "free RHS+ext")):
        ft = fits.get(key)
        if ft and not ft.get("skipped") and ft.get("coefs"):
            laws[lab] = ft["coefs"]
    fits["prediction_scores"] = prediction_scores(R.get("score"), laws, xs, args, lines)
    fits["theta_energy_free"] = block("theta_t", "ENERGY", VAR_LIB, V, b, "free", tr_truth=truth.get("energy"))
    fe = block("theta_t", "ENERGY", VAR_LIB, V, b, "forced {w^2, (div k)^2}", RCN_ENERGY, fname="forced_rcn")
    fits["theta_energy_forced"] = fe
    if bias and not fe.get("skipped"):
        corr = {k: fe["coefs"][k] * bias["energy_factor"][k] for k in RCN_ENERGY if k in bias["energy_factor"]}
        lines.append("  %-42s %s   (reported, not applied)" %
                     ("theta_t | ENERGY | forced, crease-bias corrected:", cstr(corr)))
        fits["theta_energy_forced_bias_corrected"] = corr
    # ---- w_t
    bw = T["w_t"]
    lines.append("  target scale: RMS Int[w_t phi] = %.3e" % float(np.sqrt(np.mean(bw ** 2))))
    fits["wt_free"] = block("w_t", "W", W_LIB, Wc, bw, "free", tr_truth=truth.get("w"))
    fits["wt_forced_rcn"] = block("w_t", "W", W_LIB, Wc, bw, "forced RCN w terms", list(W_TRUTH), fname="forced_rcn")
    fits["wt_forced_lapw"] = block("w_t", "W+Lap w", W_LIB + ["Lap w"], Wc, bw,
                                   "forced {Lap w, w}", ["Lap w", "w"],
                                   note="   (archival law 0.62 Lap w - 0.17 w)", fname="forced_lapw")
    fits["wt_forced_aniso"] = block("w_t", "W", W_LIB, Wc, bw, "forced {kk:ddw, w_nn, w}",
                                    ["kk:ddw", "w_nn", "w"], fname="forced_aniso")
    # ---- consistency of the theta_t and w_t laws through w_t = 2 k.grad theta_t
    # identity on the measured fields, then the symbolic map of
    # rcn_w_evolution_check check (2): a theta_t law
    #   theta_t = a (w div k + k.grad w) + b Lap div k + c w
    # implies w_t = 2a kk:ddw + a |grad w|^2 + 2a (k.grad w) div k + a w Lap w
    #              - 2a w |grad k|^2 + 2b k.grad Lap div k + 2c k.grad w,
    # evaluated on the existing W columns with the coefficients FIXED
    # (a = mean of the pair; other theta terms have no W column and are
    # listed as unmapped).
    wid = T["w_t_identity"]
    fin = np.isfinite(wid) & np.isfinite(bw)
    r2_id = float(r_squared(bw[fin], wid[fin]))
    ratio_id = float(np.sqrt(np.mean(wid[fin] ** 2)) / max(np.sqrt(np.mean(bw[fin] ** 2)), 1e-300))
    lines.append("  %-42s R2 %.4f, RMS ratio (identity / measured) %.3f" %
                 ("w_t | identity 2 k.grad theta_t (measured):", r2_id, ratio_id))
    fits["wt_identity"] = dict(r2=r2_id, rms_ratio=ratio_id)
    for key, lab in (("theta_rhs_free", "free"), ("theta_rhs_forced", "forced RCN")):
        ft = fits.get(key, {})
        if ft.get("skipped") or not ft.get("coefs"):
            continue
        cf = ft["coefs"]
        a_ = 0.5 * (cf.get("w div k", 0.0) + cf.get("k.grad w", 0.0))
        b_ = cf.get("Lap div k", 0.0); c_ = cf.get("w", 0.0)
        wmap = {"kk:ddw": 2 * a_, "|grad w|^2": a_, "(k.grad w) div k": 2 * a_, "w Lap w": a_,
                "w |grad k|^2": -2 * a_, "k.grad Lap div k": 2 * b_, "k.grad w": 2 * c_}
        unmapped = [k for k in cf if k not in ("w div k", "k.grad w", "Lap div k", "w")]
        pred = sum(v * Wc[k] for k, v in wmap.items())
        fin = np.isfinite(pred) & np.isfinite(bw)
        r2 = float(r_squared(bw[fin], pred[fin]))
        ratio = float(np.sqrt(np.mean(pred[fin] ** 2)) / max(np.sqrt(np.mean(bw[fin] ** 2)), 1e-300))
        lines.append("  %-42s R2 %.4f, RMS ratio (implied / measured) %.3f   (a %.3f, b %.3f, c %.3f fixed%s)" %
                     ("w_t | implied by theta_t %s law:" % lab, r2, ratio, a_, b_, c_,
                      ("; unmapped: " + ", ".join(unmapped)) if unmapped else ""))
        fits["wt_implied_" + key] = dict(r2=r2, rms_ratio=ratio, a=a_, b=b_, c=c_, unmapped=unmapped)
    # ---- A_t
    if Ac is not None and "A_t" in T:
        ba = T["A_t"]
        lines.append("  target scale: RMS Int[A_t phi] = %.3e ; A median %.3f ; Abar (far) %.3f" %
                     (float(np.sqrt(np.mean(ba ** 2))), R.get("A_median", np.nan), R.get("Abar", np.nan)))
        fits["at_free"] = block("A_t", "A", A_LIB, Ac, ba, "free")
        fits["at_forced_cubic"] = block("A_t", "A", A_LIB, Ac, ba, "forced {A, A^3}", ["A", "A^3"],
                                        note="   (SH law 1/2, 3/4; archival far cubic 0.494, -0.742)", fname="forced_cubic")
        fits["at_forced_band"] = block("A_t", "A", A_LIB, Ac, ba, "forced {A, A^3, w, A w}",
                                       ["A", "A^3", "w", "A w"], fname="forced_band")
        fits["at_forced_sh"] = block("A_t", "A", A_LIB, Ac, ba, "forced {A, A^3, A w^2}",
                                     ["A", "A^3", "A w^2"], fname="forced_sh",
                                     note="   (SH amplitude law A_t = (R - w^2) A - 3/4 A^3: (R, -3/4, -1))")
    else:
        lines.append("  (A identically 1 or absent; A_t not fitted)")
    return fits


# ---------------------------------------------------------------------
# the RCN control
# ---------------------------------------------------------------------

def rcn_knee_relaxation(mu, alpha, eta_b, tau, R, beta0, Ly, ny, t_end, nt,
                        keep_quartic=False):
    """integrate tau p_t = 2 alpha d_yy[f(w) p] - eta_b p_yyyy, p = Phi_y,
    w = k1^2 + p^2 - 1, f = w (- w^3/R if keep_quartic), p(+-L) = +-mu,
    p_yy(+-L) = 0; returns y, tt, p[ny, nt], Phi[ny, nt], p_t[ny, nt]"""
    k1 = np.sqrt(1 - mu ** 2)
    y = np.linspace(-Ly / 2, Ly / 2, ny)
    dy = y[1] - y[0]
    p0 = mu * np.tanh(beta0 * mu * y)

    def d2(v):
        out = np.zeros_like(v)
        out[1:-1] = (v[2:] - 2 * v[1:-1] + v[:-2]) / dy ** 2
        return out

    def rhs(t, p):
        pp = p.copy(); pp[0], pp[-1] = -mu, mu
        w = k1 ** 2 + pp ** 2 - 1.0
        f = w - (w ** 3 / R if keep_quartic else 0.0)
        term = 2 * alpha * d2(f * pp)
        p2 = d2(pp); p2[0] = p2[-1] = 0.0            # p_yy = 0 at the ends
        p4 = d2(p2)
        dp = (term - eta_b * p4) / tau
        dp[0] = dp[-1] = 0.0
        return dp

    tt = np.linspace(0, t_end, nt)
    sol = solve_ivp(rhs, (0, t_end), p0, t_eval=tt, method="BDF", rtol=1e-7, atol=1e-9)
    p = sol.y
    pt = np.column_stack([rhs(t, p[:, i]) for i, t in enumerate(tt)])
    # gauge: theta_t vanishes in the far field (plane wave), so Phi is
    # pinned at the far end (constant in time), not at the core
    Phi = np.cumsum(0.5 * (p[1:] + p[:-1]) * dy, axis=0)
    Phi = np.vstack([np.zeros((1, nt)), Phi])
    return y, tt, p, Phi, pt


def fig_control_movie(y, tt, p, Phi, u, x, path):
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))
    im = axs[0].imshow(Phi, aspect="auto", origin="lower", extent=[tt[0], tt[-1], y[0], y[-1]])
    axs[0].set_xlabel("t"); axs[0].set_ylabel("y"); axs[0].set_title("phase surface Phi(y, t)", fontsize=9)
    fig.colorbar(im, ax=axs[0], fraction=0.05)
    for j, lab in ((0, "pattern t = %.2f" % tt[0]), (-1, "pattern t = %.2f" % tt[-1])):
        ax = axs[1 if j == 0 else 2]
        ax.imshow(u[..., j], origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap="gray", aspect="equal")
        ax.set_title(lab, fontsize=9); ax.set_xlabel("x")
    sel = np.unique(np.linspace(0, len(tt) - 1, 6).astype(int))
    for j in sel:
        axs[3].plot(y, p[:, j], lw=1, label="t %.2f" % tt[j])
    axs[3].set_xlabel("y"); axs[3].set_ylabel("k2 = Phi_y"); axs[3].legend(fontsize=6)
    axs[3].set_title("k2 profile vs t (knee relaxing)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def run_control(args, out, rng, lines):
    mu = args.mu
    k1 = float(np.sqrt(1 - mu ** 2))
    Lsp = np.pi / k1
    alpha, tau, eta_b, R = args.alpha, args.tau, args.eta_b, args.R
    ny_1d = args.ny_control
    y, tt, p, Phi, pt = rcn_knee_relaxation(mu, alpha, eta_b, tau, R, args.beta0,
                                            args.Ly, ny_1d, args.t_end, args.nt)
    nper = args.n_xper
    Lx = nper * 2 * np.pi / k1
    dx = y[1] - y[0]
    nx = int(round(Lx / dx)); Lx = nx * dx
    x = (np.arange(nx) - nx // 2) * dx
    nt = len(tt)
    th = k1 * x[None, :, None] + Phi[:, None, :]                  # [ny, nx, nt]
    Phit = np.cumsum(0.5 * (pt[1:] + pt[:-1]) * dx, axis=0)
    Phit = np.vstack([np.zeros((1, nt)), Phit])          # same gauge as Phi
    th_t_ex = np.broadcast_to(Phit[:, None, :], th.shape).copy()
    k1_ex = np.full(th.shape, k1); k2_ex = np.broadcast_to(p[:, None, :], th.shape).copy()
    lines.append("RCN control: mu %.2f, alpha %.3f, tau %.3f, eta_b %.3f, beta0 %.2f,"
                 " Ly %.1f, t in [0, %.2f] (%d frames, dt %.4f), grid %dx%d" %
                 (mu, alpha, tau, eta_b, args.beta0, args.Ly, args.t_end, nt, tt[1] - tt[0], ny_1d, nx))
    beta_inf = float(np.sqrt(alpha / eta_b))
    lines.append("  expected relaxed self-dual coefficient beta = sqrt(alpha/eta_b) = %.3f;"
                 " initial knee width ratio p_y(0)/mu^2: %.3f -> %.3f at t_end" %
                 (beta_inf, args.beta0, float(np.gradient(p[:, -1], y)[ny_1d // 2] / mu ** 2)))
    truth = dict(rhs={"w div k": 2 * alpha / tau, "k.grad w": 2 * alpha / tau, "Lap div k": -eta_b / tau},
                 energy={"w^2": alpha / (2 * tau), "(div k)^2": eta_b / (2 * tau)},
                 w={"kk:ddw": 4 * alpha / tau, "|grad w|^2": 2 * alpha / tau,
                    "(k.grad w) div k": 4 * alpha / tau, "w Lap w": 2 * alpha / tau,
                    "w |grad k|^2": -4 * alpha / tau, "k.grad Lap div k": -2 * eta_b / tau})
    lines.append("  truth: RHS %s ; ENERGY %s" % (truth["rhs"], truth["energy"]))
    # v1.8: amplitude-coupled control
    camp = str(getattr(args, "control_amp", "none"))
    A_ex = None
    if camp in ("slaved", "lagged"):
        A_far = float(np.sqrt(4.0 / 3.0 * R))
        floor = float(args.amp_floor) * A_far
        w1 = k1 ** 2 + p ** 2 - 1.0                                    # [ny, nt], x-independent
        slav = np.sqrt(np.maximum(4.0 / 3.0 * (R - w1 ** 2), floor ** 2))
        if camp == "slaved":
            A1 = slav
        else:
            A1 = np.empty_like(w1); A1[:, 0] = slav[:, 0]
            nsub = 20
            for j in range(nt - 1):
                a = A1[:, j].copy(); h = (tt[j + 1] - tt[j]) / nsub
                for q in range(nsub):
                    def f(aa, s_):
                        ww = w1[:, j] + s_ * (w1[:, j + 1] - w1[:, j])
                        return (R - ww ** 2) * aa - 0.75 * aa ** 3
                    s0 = q / nsub; s1 = (q + 0.5) / nsub; s2 = (q + 1.0) / nsub
                    q1 = f(a, s0); q2 = f(a + 0.5 * h * q1, s1); q3 = f(a + 0.5 * h * q2, s1); q4 = f(a + h * q3, s2)
                    a = a + h * (q1 + 2 * q2 + 2 * q3 + q4) / 6.0
                A1[:, j + 1] = np.maximum(a, floor)
        A_ex = np.broadcast_to(A1[:, None, :], th.shape).copy()
        lines.append("  amplitude-coupled control (v1.8): %s amplitude, floor %.2g x A_far (A_far %.3f);"
                     " on the crease line A = %.3f at t = 0 and %.3f at t_end; the phase law is pure RCN, so every"
                     " amplitude or defect term returned below is leakage" %
                     (camp, args.amp_floor, A_far, A1[ny_1d // 2, 0], A1[ny_1d // 2, -1]))
        if camp == "lagged":
            truth["A"] = {"A": R, "A^3": -0.75, "A w^2": -1.0}
    u = np.cos(th) if A_ex is None else A_ex * np.cos(th)
    fig_control_movie(y, tt, p, Phi, u, x, out / "control_movie.png")
    routes = {"exact": (k1_ex, k2_ex, th_t_ex)}
    amps = {"exact": A_ex}
    taper = t1c.edge_taper(x, y, frac=args.taper_frac)
    Lyy = ny_1d * dx
    ests, n_prev = [], None
    for i in range(nt):
        e = estimate(u[..., i], Lx, Lyy, args.sigma_dir, n_prev, taper)
        n_prev = e["n"]; ests.append(e)
    dt = tt[1] - tt[0]
    k1m = np.stack([e["k1"] for e in ests], -1); k2m = np.stack([e["k2"] for e in ests], -1)
    zs = np.stack([e["z"] for e in ests], -1)
    amps["estimator"] = np.stack([e["A"] for e in ests], -1) if A_ex is not None else None
    del ests
    th_t_m = theta_t_from_z(zs, dt)
    s = np.sign(np.mean(k1m, axis=(0, 1)))[None, None, :]; s[s == 0] = 1   # align with exact (k1 > 0); v1.6 scalar per frame
    routes["estimator"] = (k1m * s, k2m * s, th_t_m * s)
    mask_tt = np.abs(th_t_m - th_t_ex)[..., 2:-2]
    lines.append("  estimator theta_t vs exact, rel RMS over the box (frames 2..nt-3): %.4f" %
                 (np.sqrt(np.nanmean(mask_tt ** 2)) / np.sqrt(np.mean(th_t_ex[..., 2:-2] ** 2))))
    lines.append("  time kernel half width t_half x dt = %.3f time units" % (args.t_half * dt))
    m = kernel_sizes(args, dx, dx, dt, Lsp)
    idx = query_centres(th.shape, m, args)
    fits = {}
    for route, (K1, K2, TT) in routes.items():
        Ar = amps.get(route)
        Abar_r = None
        if Ar is not None:
            farr = np.abs(y) > 0.3 * (y[-1] - y[0])
            Abar_r = float(np.nanmedian(Ar[farr]))
        cols, targets, fields = build_columns(K1, K2, TT, x, y, tt, args, Lsp, idx, A=Ar, Abar=Abar_r,
                                              snap=nt // 2 if route == "estimator" else None)
        R = dict(rhs=cols["rhs"], var=cols["var"], w=cols["w"], A=cols["A"],
                 targets=targets, xs=x[idx[1]], score=cols["score"],
                 Abar=(Abar_r if Abar_r is not None else np.nan),
                 A_median=(float(np.nanmedian(Ar)) if Ar is not None else np.nan))
        lines.append("  [route %s]" % route)
        fits[route] = fit_blocks(R, "control %s route" % route, args, rng, lines, out,
                                 truth=truth, route=route)
        if fields is not None:
            fig_library_terms(fields, x, y, out / "library_terms_control_estimator.png",
                              "control, estimator route, frame %d" % (nt // 2))
    # crease bias: factor truth/estimator on the forced RCN coefficients
    fe = fits["estimator"]
    bias = dict(radius=args.radius, t_half=args.t_half, mu=mu,
                rhs_factor={k: truth["rhs"][k] / fe["theta_rhs_forced"]["coefs"][k] for k in RCN_RHS},
                energy_factor={k: truth["energy"][k] / fe["theta_energy_forced"]["coefs"][k] for k in RCN_ENERGY},
                rhs_forced_estimator=fe["theta_rhs_forced"]["coefs"],
                energy_forced_estimator=fe["theta_energy_forced"]["coefs"])
    with open(out / "bias.json", "w") as fh:
        json.dump(bias, fh, indent=1)
    lines.append("  crease bias at radius %.2g (truth/estimator, forced RCN): RHS %s ; ENERGY %s -> bias.json" %
                 (args.radius, ", ".join("%s %.3f" % kv for kv in bias["rhs_factor"].items()),
                  ", ".join("%s %.3f" % kv for kv in bias["energy_factor"].items())))
    lines.append("")
    return fits


def fig_library_terms(fields, x, y, path, title):
    names = list(fields)
    ncol = 3
    nrow = int(np.ceil(len(names) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False)
    for ax, nm in zip(axs.ravel(), names):
        F = fields[nm]
        ny_, nx_ = F.shape
        inner = F[int(0.1 * ny_):int(0.9 * ny_), int(0.1 * nx_):int(0.9 * nx_)]
        v = np.nanpercentile(np.abs(inner), 99)   # scale from the interior (edge ringing excluded)
        v = v if np.isfinite(v) and v > 0 else 1.0
        im = ax.imshow(F, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap="RdBu_r",
                       vmin=-v, vmax=v, aspect="auto")
        ax.set_title(nm, fontsize=9); fig.colorbar(im, ax=ax, fraction=0.05)
    for ax in axs.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ---------------------------------------------------------------------
# data
# ---------------------------------------------------------------------

C_KEPT, C_CENS, C_BOX, C_CORE = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"   # v1.7.1


def fig_domains(u, x, y, t, jc0, Lsp, dom, excl, cores, cores_xy, idx, m, args, path, stem, mu):
    """v1.7.1: the fitting domain of this run drawn on the pattern (see the
    v1.7.1 changelog). u: float32 [ny, nx, nt]; dom: [ny, nx]; excl, cores:
    [ny, nx, nt] booleans as passed to query_centres; idx: kept centres."""
    from matplotlib.patches import Circle, Rectangle
    from matplotlib.lines import Line2D
    IY, IX, IT = idx
    mx, my, mt = m
    ny, nx, nt = u.shape
    # frame: among frames carrying centres, the one with the most cores (latest on ties)
    fr = np.unique(IT)
    nc = np.array([len(cores_xy[f][0]) for f in fr])
    f = int(fr[np.where(nc == nc.max())[0][-1]])
    xm = 0.5 * (x[0] + x[-1])
    X = (x - xm) / Lsp; Yr = (y - y[jc0]) / Lsp
    # candidate centres on the grid (same rule as query_centres, before exclusion)
    ix = np.arange(mx + args.rim, nx - mx - args.rim, args.stride_xy)
    iy = np.arange(my + args.rim, ny - my - args.rim, args.stride_xy)
    it = np.arange(mt, nt - mt, args.stride_t)
    GY, GX = np.meshgrid(iy, ix, indexing="ij")
    cand_f = dom[GY, GX]
    kept_f = np.zeros_like(cand_f)
    sel = IT == f
    kept_set = set(zip(IY[sel].tolist(), IX[sel].tolist()))
    for a_, (j, i) in enumerate(zip(GY.ravel(), GX.ravel())):
        if (j, i) in kept_set:
            kept_f.ravel()[a_] = True
    cens_f = cand_f & ~kept_f
    # dilated exclusion shown on this frame
    dil = None
    if int(getattr(args, "dilate_cores_only", 0)) and cores.any():
        dil = box_dilate(cores, my, mx, mt)[..., f]
    elif args.excl_dilate:
        dil = box_dilate(excl, my, mx, 0)[..., f]
    a = args.radius * Lsp
    sd = a / np.sqrt(2 * args.poly_p + 3)

    fig = plt.figure(figsize=(11.5, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.0], wspace=0.04)
    ax = fig.add_subplot(gs[0]); axh = fig.add_subplot(gs[1], sharey=ax)
    ext = [X[0], X[-1], Yr[0], Yr[-1]]
    ax.imshow(u[..., f], origin="lower", extent=ext, cmap="gray", vmin=-1.2, vmax=1.2,
              aspect="equal", interpolation="nearest", alpha=0.85)
    ax.imshow(np.ma.masked_where(~dom, np.ones_like(dom, float)), origin="lower", extent=ext,
              cmap=matplotlib.colors.ListedColormap([C_KEPT]), alpha=0.13, aspect="equal")
    if dil is not None:
        ax.imshow(np.ma.masked_where(~dil, np.ones_like(dil, float)), origin="lower", extent=ext,
                  cmap=matplotlib.colors.ListedColormap([C_CORE]), alpha=0.30, aspect="equal")
    edges = {"row": args.row_to, "band": args.band_to, "far": args.far_from}
    for lab, d in edges.items():
        for sgn in (1, -1):
            ax.axhline(sgn * d, color="0.15", lw=0.8, ls="--")
        ax.text(X[-1], d, " %s %.3g L" % (lab, d), va="center", ha="left", fontsize=8, color="0.15",
                clip_on=False)
    ax.axhline(0.0, color="0.15", lw=1.0)
    xc, yc = cores_xy[f]
    if args.core_excl > 0:
        for cx, cy in zip(xc, yc):
            ax.add_patch(Circle(((cx - xm) / Lsp, (cy - y[jc0]) / Lsp), args.core_excl,
                                fill=False, ec=C_CORE, lw=1.4))
    ax.plot(X[GX[kept_f]], Yr[GY[kept_f]], "o", ms=2.6, color=C_KEPT, mec="none")
    ax.plot(X[GX[cens_f]], Yr[GY[cens_f]], "x", ms=3.2, mew=0.8, color=C_CENS)
    if sel.any():
        k = np.argmin(np.abs(Yr[IY[sel]]))
        bx, by = X[IX[sel][k]], Yr[IY[sel][k]]
        ax.add_patch(Rectangle((bx - args.radius, by - args.radius), 2 * args.radius, 2 * args.radius,
                               fill=False, ec=C_BOX, lw=2.0))
        ax.add_patch(Circle((bx, by), sd / Lsp, fill=False, ec=C_BOX, lw=1.2, ls=":"))
        ax.plot([bx], [by], "s", ms=5, color=C_BOX)
    ylim = max(args.far_from + 1.5 * args.radius, args.band_to + 2.0 * args.radius)
    ax.set_ylim(max(Yr[0], -ylim), min(Yr[-1], ylim))
    ax.set_xlim(X[0], X[-1])
    ax.set_xlabel("(x - x_mid) / L"); ax.set_ylabel("(y - y_row) / L")
    excl_txt = ("core discs dilated by the full x, y, t support (core-free)"
                if int(getattr(args, "dilate_cores_only", 0)) else
                ("whole exclusion mask dilated in x, y" if args.excl_dilate else
                 "core discs remove centres only; test functions still overlap cores"))
    ax.set_title("%s, mu %.2f: domain '%s', frame t = %.3f (%d cores)\n"
                 "radius %.3g L, core discs %.3g L; %s" %
                 (stem, mu, args.domain, t[f], len(xc), args.radius, args.core_excl, excl_txt), fontsize=9)
    hnd = [Line2D([], [], marker="o", ls="none", color=C_KEPT, ms=5, label="kept centres (this frame)"),
           Line2D([], [], marker="x", ls="none", color=C_CENS, ms=5, label="censored centres"),
           Line2D([], [], color=C_CORE, lw=1.4, label="core disc"),
           Line2D([], [], color=C_BOX, lw=2.0, label="test-function support (half width %.3g L)" % args.radius),
           Line2D([], [], color=C_BOX, lw=1.2, ls=":", label="bump std a/sqrt(2p+3) = %.2g L" % (sd / Lsp))]
    if dil is not None:
        hnd.append(matplotlib.patches.Patch(color=C_CORE, alpha=0.3, label="dilated exclusion"))
    hnd.append(matplotlib.patches.Patch(color=C_KEPT, alpha=0.13, label="domain '%s'" % args.domain))
    ax.legend(handles=hnd, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3,
              frameon=False)
    # right panel: centres over all frames, per row of the centre grid
    rows_y = Yr[iy]
    n_cand_rows = np.array([dom[j, ix].sum() for j in iy], float) * len(it)
    n_kept_rows = np.array([(IY == j).sum() for j in iy], float)
    hgt = 0.8 * abs(rows_y[1] - rows_y[0]) if len(rows_y) > 1 else 0.1
    axh.barh(rows_y, n_cand_rows, height=hgt, color="none", ec="0.45", lw=0.6, label="candidates")
    axh.barh(rows_y, n_kept_rows, height=hgt, color=C_KEPT, ec="none", label="kept")
    axh.set_xlabel("centres, all %d frames" % len(it), fontsize=8)
    axh.tick_params(labelleft=False, labelsize=7)
    axh.legend(fontsize=7, loc="upper right", framealpha=0.85)
    axh.set_title("kept %d of %d\n(per row of the centre grid)" % (len(IY), int(n_cand_rows.sum())), fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def load_file_rows(path, args, out, lines):
    """estimator on the window, domain masks, columns sampled at the
    query centres; returns the row dict R (or None) and the file label"""
    D = load_npz(path)
    u, x, y, t, md = D["u"], D["x"], D["y"], D["t"], D["md"]
    ss = max(1, int(getattr(args, "space_stride", 1)))                 # v1.8: spatial decimation
    if ss > 1:
        md = dict(md)
        md["gb_row_in_crop"] = int(md.get("gb_row_in_crop", u.shape[0] // 2)) // ss
        u, x, y = u[::ss, ::ss], x[::ss], y[::ss]
    mu = float(md.get("mu", args.mu)); k1 = np.sqrt(1 - mu ** 2); Lsp = np.pi / k1
    sel = (t >= args.t0) & (t <= args.t1)
    u, t = u[..., sel], t[sel]
    fs = max(1, int(getattr(args, "frame_stride", 1)))                 # v1.7: true frame-spacing ablation
    if fs > 1:
        u, t = u[..., ::fs], t[::fs]
    ny, nx, nt = u.shape
    stem = Path(path).stem
    if nt < 2 * args.t_half + 3:
        lines.append("data: %s  only %d frames in [%.3f, %.3f]; need >= %d; skipped" %
                     (stem, nt, args.t0, args.t1, 2 * args.t_half + 3))
        return None, stem
    dx = x[1] - x[0]; dt = float(np.median(np.diff(t)))
    Lx, Ly = nx * dx, ny * dx
    taper = t1c.edge_taper(x, y, frac=args.taper_frac)
    X, Y = np.meshgrid(x, y)
    win = (np.abs(X - x.mean()) < 0.4 * Lx) & (np.abs(Y - y.mean()) < 0.4 * Ly)
    jc0 = int(md.get("gb_row_in_crop", ny // 2))
    d_row = np.abs(Y - y[jc0]) / Lsp
    ests, n_prev = [], None
    for i in range(nt):
        e = estimate(u[..., i], Lx, Ly, args.sigma_dir, n_prev, taper)
        n_prev = e["n"]; ests.append(e)
    shape = u.shape
    u_fig = u.astype(np.float32) if int(getattr(args, "domain_fig", 0)) else None   # v1.7.1
    del u, D
    k1m = np.stack([e["k1"] for e in ests], -1); k2m = np.stack([e["k2"] for e in ests], -1)
    zs = np.stack([e["z"] for e in ests], -1)
    Am = np.stack([e["A"] for e in ests], -1)
    th_t = theta_t_from_z(zs, dt)
    del zs
    for e in ests:
        for key in ("z", "theta", "k1", "k2", "d1", "d2"):
            e.pop(key, None)
    s = np.sign(np.mean(k1m, axis=(0, 1)))[None, None, :]; s[s == 0] = 1     # v1.6: scalar per frame
    k1m, k2m, th_t = k1m * s, k2m * s, th_t * s
    # exclusion: core discs (core distance) and the domain (row distance)
    excl = np.zeros(shape, bool)
    ncores, cores_xy = [], []
    for i in range(nt):
        xc, yc = find_cores(ests[i]["A"], x, y, win, args.core_frac, merge_dist=0.5 * Lsp)
        ncores.append(len(xc)); cores_xy.append((xc, yc))
        if args.core_excl > 0 and len(xc):
            excl[..., i] = dist_to_cores(X, Y, xc, yc) / Lsp < args.core_excl
    del ests
    cores = excl.copy()                                                # v1.7: core discs alone
    if args.domain == "far":
        dom = d_row > args.far_from
    elif args.domain == "band":
        dom = d_row < args.band_to
    elif args.domain == "row":
        dom = d_row < args.row_to
    elif args.domain == "shell":
        dom = (d_row >= args.band_to) & (d_row <= args.far_from)
    else:
        dom = np.ones_like(d_row, bool)
    if args.side == "upper":
        dom &= Y > y[jc0]
    elif args.side == "lower":
        dom &= Y < y[jc0]
    excl |= ~dom[..., None]
    if args.domain == "active":
        # points whose local RCN energy density (alpha = eta = 1 weights)
        # w^2 + (div k)^2 exceeds active_frac of its band median, per frame
        wloc = k1m ** 2 + k2m ** 2 - 1.0
        dloc = np.gradient(k1m, dx, axis=1) + np.gradient(k2m, dx, axis=0)
        eloc = wloc ** 2 + dloc ** 2
        del wloc, dloc
        band = d_row < args.band_to
        for i in range(nt):
            ref = np.nanmedian(eloc[..., i][band])
            excl[..., i] |= eloc[..., i] < args.active_frac * ref
        del eloc
    far = d_row > args.far_from
    Abar = float(np.nanmedian(Am[far]) if far.any() else np.nanmedian(Am[win]))
    lines.append("data: %s  mu %.2f  t in [%.3f, %.3f] (%d frames, dt %.3f)  crop %dx%d = +-%.1f"
                 " spacings about the row  domain %s  cores per frame %d..%d" %
                 (stem, mu, t[0], t[-1], nt, dt, nx, ny, 0.5 * Ly / Lsp, args.domain,
                  min(ncores), max(ncores)))
    lines.append("  domain cut by row distance: row_to %.2g, band_to %.2g, far_from %.2g spacings;"
                 " core discs of %.2g spacings removed; time kernel half width t_half x dt = %.3f"
                 % (args.row_to, args.band_to, args.far_from, args.core_excl, args.t_half * dt))
    m = kernel_sizes(args, dx, dx, dt, Lsp)
    idx = query_centres(shape, m, args, excl=excl, cores=cores)
    if idx is None:
        lines.append("  no query centres survive the domain exclusion; nothing fitted")
        return None, stem
    if u_fig is not None:                                              # v1.7.1
        try:
            fig_domains(u_fig, x, y, t, jc0, Lsp, dom, excl, cores, cores_xy, idx, m, args,
                        out / ("domains_%s.png" % stem), stem, mu)
            lines.append("  domain figure -> domains_%s.png" % stem)
        except Exception as exc:                                        # pragma: no cover
            lines.append("  domain figure skipped: %s" % exc)
        del u_fig
    cols, targets, fields = build_columns(k1m, k2m, th_t, x, y, t, args, Lsp, idx,
                                          A=Am, Abar=Abar, snap=nt // 2)
    fig_library_terms(fields, x, y, out / ("library_terms_%s.png" % stem),
                      "%s, frame %d (t = %.3f)" % (stem, nt // 2, t[nt // 2]))
    R = dict(rhs=cols["rhs"], var=cols["var"], w=cols["w"], A=cols["A"], score=cols["score"],
             targets=targets, xs=x[idx[1]], Abar=Abar, A_median=float(np.nanmedian(Am[win])),
             mu=mu, n_rows=len(idx[0]))
    lines.append("  query centres %d (stride %d, %d); frame_stride %d (dt %.4g); space_stride %d (dx %.4g); exclusion %s" % (
        len(idx[0]), args.stride_xy, args.stride_t, fs, dt, ss, dx,
        "core discs dilated by the full x, y, t support (core-free)" if int(getattr(args, "dilate_cores_only", 0))
        else ("whole mask dilated in x, y (v1.6)" if args.excl_dilate else "centres only")))
    return R, stem


def pool_rows(Rs):
    keys = ["rhs", "var", "w", "A"]
    P = {}
    for k in keys:
        if any(R[k] is None for R in Rs):
            P[k] = None
            continue
        P[k] = {nm: np.concatenate([R[k][nm] for R in Rs]) for nm in Rs[0][k]}
    P["targets"] = {nm: np.concatenate([R["targets"][nm] for R in Rs])
                    for nm in Rs[0]["targets"] if all(nm in R["targets"] for R in Rs)}
    P["xs"] = np.concatenate([R["xs"] for R in Rs])
    P["score"] = None                                   # v1.6: scores are per file
    P["Abar"] = float(np.mean([R["Abar"] for R in Rs]))
    P["A_median"] = float(np.mean([R["A_median"] for R in Rs]))
    return P


def run_data(args, out, rng, lines):
    files = [f.strip() for f in args.npz.split(",") if f.strip()]
    bias = None
    if args.bias_json:
        bias = json.load(open(args.bias_json))
        lines.append("crease-bias factors from %s (control radius %.2g, t_half %d)" %
                     (args.bias_json, bias["radius"], bias["t_half"]))
        if abs(bias["radius"] - args.radius) > 1e-9 or bias["t_half"] != args.t_half:
            lines.append("  WARNING: control kernel (radius %.2g, t_half %d) differs from this run"
                         " (radius %.2g, t_half %d); the factors do not transfer" %
                         (bias["radius"], bias["t_half"], args.radius, args.t_half))
    Rs, stems, fits = [], [], {}
    for f in files:
        R, stem = load_file_rows(f, args, out, lines)
        if R is None:
            lines.append("")
            continue
        Rs.append(R); stems.append(stem)
        tag = "%s | t %.2f-%.2f | %s" % (stem, args.t0, args.t1, args.domain)
        fits[stem] = fit_blocks(R, tag, args, rng, lines, out, bias=bias,
                                route=("" if len(files) == 1 else stem))
        lines.append("")
    if len(Rs) > 1:
        P = pool_rows(Rs)
        lines.append("pooled: %d files (%s), %d rows; a pooled fit tests that the coefficients"
                     " depend on R and not on mu" % (len(Rs), ", ".join("mu %.2f" % R["mu"] for R in Rs),
                                                    len(P["xs"])))
        fits["pooled"] = fit_blocks(P, "pooled %d files | %s" % (len(Rs), args.domain), args, rng,
                                    lines, out, bias=bias, route="pooled")
        lines.append("")
    return fits


# ---------------------------------------------------------------------

def main(args):
    rng = np.random.default_rng(args.seed)
    if args.mode == "control":
        stem = "control_rcn"
    else:
        files = [f.strip() for f in args.npz.split(",") if f.strip()]
        stem = Path(files[0]).stem if len(files) == 1 else "pooled_%d" % len(files)
    stamp = "rad%.2g_eta%.2g_th%d_mc%.2g_%s_ce%.2g_ext%d_t%.2f-%.2f" % (
        args.radius, args.eta, args.t_half, args.min_contrib, args.domain, args.core_excl,
        args.lib_ext, args.t0, args.t1)
    if int(getattr(args, "frame_stride", 1)) > 1:                     # v1.7 suffixes
        stamp += "_fs%d" % int(args.frame_stride)
    if int(getattr(args, "dilate_cores_only", 0)):
        stamp += "_dco"
    elif args.excl_dilate:
        stamp += "_dil"
    if int(getattr(args, "space_stride", 1)) > 1:                      # v1.8
        stamp += "_ss%d" % int(args.space_stride)
    if str(getattr(args, "control_amp", "none")) != "none" and args.mode == "control":
        stamp += "_amp" + str(args.control_amp)
    if args.mode == "control" and abs(float(args.mu) - 0.85) > 1e-9:
        stamp += "_mu%.2f" % float(args.mu)
    if args.domain == "shell":
        stamp += "_s%.3g-%.3g" % (args.band_to, args.far_from)
    if getattr(args, "tag", ""):
        stamp += "_" + _slug(str(args.tag))
    out = Path(args.outdir) / stem / stamp
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as fh:  # noqa
        json.dump({k: getattr(args, k) for k in vars(args)}, fh, indent=2, default=str)
    lines = ["wsindy_theta_t v1.8   (helpers: %s)" % _HELPERS,
             "libraries: RHS %s ; RHS ext %s ; ENERGY %s ; W %s ; A %s" %
             (RHS_LIB, RHS_EXT, VAR_LIB, W_LIB, A_LIB),
             "kernel radius %.2g spacings (x and y), t_half %d frames, inner eta %.2g spacings,"
             " rel_tol %.2g, abs_tol %.2g, min_contrib %.2g; crease bias of the control at this radius"
             " applies to every data coefficient (stretching read low, bending high) and is NOT a"
             " constant factor: it grows as the row approaches the self-dual balance" %
             (args.radius, args.t_half, args.eta, args.rel_tol, args.abs_tol, args.min_contrib), ""]
    t0 = time.time()
    if args.mode == "control":
        fits = run_control(args, out, rng, lines)
    else:
        fits = run_data(args, out, rng, lines)
    lines.append("elapsed %.1f s" % (time.time() - t0))
    rep = "\n".join(lines)
    (out / "report.txt").write_text(rep)
    with open(out / "fits.json", "w") as fh:
        json.dump(fits, fh, indent=1, default=str)
    print(rep); print("outputs ->", out)
    return 0


class _Args:
    """PyCharm-friendly defaults; mirrored by the CLI below."""
    mode = "control"             # control: the RCN knee movie; data: --npz
    npz = ""                     # data file, or a comma-separated list (pooled)
    outdir = str(Path("results") / "wsindy_theta_t")
    bias_json = ""               # bias.json of a control run at the same radius and t_half
    # control parameters
    mu = 0.85
    alpha = 1.0 / 3.0            # Writeup 1: alpha = eta = tau = 2R/3 at R = 1/2
    tau = 1.0 / 3.0
    eta_b = 1.0 / 3.0            # (the chapter's eta = 2 tau is the open factor-two todo)
    R = 0.5
    beta0 = 1.8                  # planted sharp knee, relaxes toward sqrt(alpha/eta_b)
    Ly = 60.0
    t_end = 2.0
    nt = 41
    n_xper = 4
    ny_control = 300
    # estimator
    sigma_dir = 0.4
    taper_frac = 0.06
    core_frac = 0.70
    # weak form
    radius = 1.5                 # test-function half width, spacings (control: 0.5 fails, 1.5-2 identifies)
    t_half = 3                   # frames
    eta = 0.0                    # inner mollifier width, spacings (0 = np.gradient)
    poly_p = 6
    stride_xy = 4
    stride_t = 1
    rim = 4
    rel_tol = 0.01
    abs_tol = 0.0                # absolute R2 tolerance for admissibility (0: rel_tol only); 0.02 favours sparser supports
    min_contrib = 0.05
    seed = 0
    lib_ext = 1                  # data: also fit the extended RHS library (amplitude, defect columns)
    # data windows / domains
    t0 = -1.0
    t1 = 1e9
    domain = "all"               # all | far | band | row | shell (band_to..far_from) | active (energy above active_frac x band median)
    side = "both"                # both | upper | lower  (y above / below the row)
    active_frac = 0.5            # for --domain active
    core_excl = 0.5              # spacings around detected cores removed; 0 disables (pre-winding)
    row_to = 0.5                 # row domain: d_row < row_to
    band_to = 2.0                # band domain: d_row < band_to
    far_from = 4.0               # far domain: d_row > far_from (tall crop: +-13 spacings available)
    excl_dilate = 0              # 1: exclude centres whose kernel touches the region; 0: centres only
    score_h = 6                  # v1.6: longest prediction horizon in frames (0 disables); horizons t_half, 2 t_half, score_h
    chi_m = 1.0                  # v1.6: screening mass of chi_J = (Lap - m^2)^{-1} J (1 = beta k_B)
    # v1.7
    exhaustive = 1               # 1: least squares on every theta_t support of the sizes below
    exh_sizes = "3,4"            # comma-separated support sizes (C(16,3) = 560, C(16,4) = 1820)
    exh_top = 8                  # supports printed per size
    free_no_amp = 1              # 1: extra free fit of RHS+ext without A_dev, A_dev div k, k.grad A
    dilate_cores_only = 0        # 1: dilate the core discs alone by the full x, y, t support (core-free reference)
    frame_stride = 1             # keep every k-th frame of the window before the estimator (frame-spacing ablation)
    tag = ""                     # appended to the output stamp
    domain_fig = 1               # v1.7.1: domains_<stem>.png, the fitting domain drawn on the pattern
    # v1.8
    control_amp = "none"         # control: none | slaved | lagged (amplitude coupled to the RCN phase, no feedback)
    amp_floor = 0.05             # control: floor of the planted amplitude, fraction of A_far = sqrt(4R/3)
    lib_cn = 1                   # Cross-Newell amplitude column w k.grad lnA, the law div(A^2 w k)/A^2, size-3 search with it
    space_stride = 1             # keep every k-th grid point in x and y before the estimator


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in [a for a in dir(_Args) if not a.startswith("_")]:
        v = getattr(_Args, k)
        p.add_argument("--" + k, type=type(v) if not isinstance(v, bool) else str, default=v)
    cli = p.parse_args()
    if cli.npz:
        cli.mode = "data"
    sys.exit(main(cli))
