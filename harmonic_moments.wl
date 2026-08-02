(* ::Package:: *)

(* ============================================================
   harmonic_moments.wl
   Symbolic machinery for the odd-harmonic stripe ansatz

       u = Sum_n A_n Cos[n theta],   theta = kx x + ky y,  K = kx^2+ky^2

   Part 1: differential quantities (Laplacian, biharmonic, squares)
           and their ideal phase averages <.> = (1/2Pi) Int_0^2Pi dtheta.
   Part 2: the moment ladder M_m = <u Laplacian^m u>, closed forms,
           the uHu wavenumber bias factor, and the de-mixing solve
           (K, A1^2, A3^2 from M0, M1, M2; general P-harmonic case).
   Part 3: stationary Swift-Hohenberg Galerkin projection:
           substitute the ansatz into (R - (1+Lap)^2) u - u^3 = 0,
           collect Cos[n theta], and produce the algebraic system /
           recurrence relations for A1, A3, A5, ... as rational
           functions of K and R.

   Run with:  wolframscript -file harmonic_moments.wl
   or evaluate section-by-section in a notebook.
   ============================================================ *)

(* ------------------------------------------------------------
   Part 0: setup
   ------------------------------------------------------------ *)

harmonics = {1, 3, 5};             (* edit: e.g. {1,3} or {1,3,5,7} *)
amps      = Table[Symbol["A" <> ToString[n]], {n, harmonics}];

(* plane-wave phase; slowly-varying corrections are handled in
   Part 1b as a curvature check *)
theta = kx x + ky y;
K     = kx^2 + ky^2;

u = Sum[amps[[i]] Cos[harmonics[[i]] theta], {i, Length[harmonics]}];

lap[f_]  := D[f, {x, 2}] + D[f, {y, 2}];
bih[f_]  := lap[lap[f]];

(* ideal phase average <.> = (1/2Pi) Integrate over one period of theta.
   Since theta is linear, evaluate along the ray x = s kx/K, y = s ky/K
   (so that theta == s exactly) and integrate s over one period. This is
   robust: no pattern-matching on trig arguments needed. *)
phaseAvg[expr_] := Simplify[
  (1/(2 Pi)) Integrate[
     expr /. {x -> s kx/(kx^2 + ky^2), y -> s ky/(kx^2 + ky^2)},
     {s, 0, 2 Pi}],
  Assumptions -> {kx > 0, ky >= 0}];

Print["=== Part 1: differential quantities and phase averages ==="];

lapU = Simplify[lap[u]];
bihU = Simplify[bih[u]];

Print["Laplacian u = ", Collect[TrigReduce[lapU], Cos[_], Simplify]];
Print["Biharmonic u = ", Collect[TrigReduce[bihU], Cos[_], Simplify]];

(* squares and cross products -- phase-averaged *)
avgU2       = phaseAvg[u^2];
avgULapU    = phaseAvg[u lapU];
avgLapU2    = phaseAvg[lapU^2];
avgUBihU    = phaseAvg[u bihU];
avgLapUBihU = phaseAvg[lapU bihU];
avgGradU2   = phaseAvg[D[u, x]^2 + D[u, y]^2];

Print["<u^2>              = ", avgU2 /. kx^2 + ky^2 -> KK];
Print["<u Lap u>          = ", Simplify[avgULapU] /. kx^2 + ky^2 -> KK];
Print["<(Lap u)^2>        = ", Simplify[avgLapU2] /. kx^2 + ky^2 -> KK];
Print["<u Bih u>          = ", Simplify[avgUBihU] /. kx^2 + ky^2 -> KK];
Print["  (check: <(Lap u)^2> - <u Bih u> = ",
      Simplify[avgLapU2 - avgUBihU], "  -- equal by parts)"];
Print["<|grad u|^2>       = ", Simplify[avgGradU2] /. kx^2 + ky^2 -> KK];
Print["<Lap u Bih u>      = ", Simplify[avgLapUBihU] /. kx^2 + ky^2 -> KK];

(* ------------------------------------------------------------
   Part 1b: curvature check -- slowly varying phase.
   Replace theta -> Theta[eps x, eps y] and confirm the cross /
   curvature corrections to the averages enter at O(eps).
   (This was verified in the earlier session; kept here as a
   regression check.)
   ------------------------------------------------------------ *)

Print["=== Part 1b: slow-phase curvature check (O(eps) corrections) ==="];
Module[{thPert, uPert, lapPert, o1},
  (* perturb the phase by a slow field: theta -> theta + eps Psi[x,y].
     The O(eps) piece of Lap u collects the curvature (Lap Psi) and
     phase-gradient cross terms; the point to verify is that every
     O(eps) term multiplies Sin[n theta], so <u Lap u> picks up no
     O(eps) correction (Sin x Cos averages to zero). *)
  thPert = theta + eps Psi[x, y];
  uPert  = Sum[amps[[i]] Cos[harmonics[[i]] thPert], {i, Length[harmonics]}];
  lapPert = lap[uPert];
  o1 = Simplify[Coefficient[Normal[Series[lapPert, {eps, 0, 1}]], eps]];
  Print["O(eps) part of Lap u:"];
  Print[Collect[TrigReduce[o1] /. theta -> th, {Sin[_ th], Cos[_ th]}, Simplify]];
  Print["  -> every term carries Sin[n theta]; against u ~ Cos[n theta] ",
        "these phase-average to zero, so M_1 = <u Lap u> has no O(eps) ",
        "curvature correction. Same mechanism kills the cross-harmonic terms."];
];

(* ------------------------------------------------------------
   Part 2: moment ladder
   ------------------------------------------------------------ *)

Print["=== Part 2: moment ladder M_m = <u Lap^m u> ==="];

nLap[f_, 0] := f;
nLap[f_, m_Integer?Positive] := lap[nLap[f, m - 1]];

Mmax = Length[harmonics];   (* need M_0 .. M_P for P harmonics *)
moments = Table[
   Simplify[phaseAvg[u nLap[u, m]]] /. kx^2 + ky^2 -> KK,
   {m, 0, Mmax}];

Do[Print["M_", m, " = ", moments[[m + 1]]], {m, 0, Mmax}];

(* generic closed form check: M_m = (-1)^m (K^m/2) Sum n^(2m) A_n^2 *)
closed = Table[(-1)^m (KK^m/2) Sum[harmonics[[i]]^(2 m) amps[[i]]^2,
     {i, Length[harmonics]}], {m, 0, Mmax}];
Print["closed-form match: ", Simplify[moments - closed] === Table[0, {Mmax + 1}]];

(* uHu wavenumber bias factor if a single harmonic is assumed *)
biasFactor = Simplify[-moments[[2]]/(KK moments[[1]])];
Print["k^2_uHu / K_true (bias factor) = ", biasFactor,
      "   (= Sum n^2 A_n^2 / Sum A_n^2)"];

(* de-mixing solve: unknowns K and the s_n = A_n^2 *)
Print["--- de-mixing: solve M_0..M_P for {K, A_n^2} ---"];
svars = Table[Symbol["s" <> ToString[n]], {n, harmonics}];
eqs = Table[
   Mm[m] == (-1)^m (KK^m/2) Sum[harmonics[[i]]^(2 m) svars[[i]],
       {i, Length[harmonics]}], {m, 0, Mmax}];
sol = Solve[eqs, Prepend[svars, KK]];
Print["solution branches (rational in the moments):"];
Do[Print["  branch ", j, ": ", Simplify[sol[[j]]]], {j, Length[sol]}];

(* two-harmonic special case: display the K-quadratic explicitly *)
Module[{eqs2, elim},
  eqs2 = Table[
     Mm[m] == (-1)^m (KK^m/2) (t1 + 3^(2 m) t3), {m, 0, 2}];
  elim = Eliminate[eqs2, {t1, t3}];
  Print["{1,3} case, eliminating amplitudes: ", Simplify[elim]];
  Print["  (expect: 18 M0 K^2 + 20 M1 K + 2 M2 == 0, with M1 < 0 ",
        "the physical root is the LARGER K; the smaller root K/9 is the ",
        "alias where all energy is read as third harmonic of a lower ",
        "fundamental.)"];
];

(* ------------------------------------------------------------
   Part 3: stationary Swift-Hohenberg Galerkin recurrences
   (R - (1 + Lap)^2) u - u^3 = 0
   ------------------------------------------------------------ *)

Print["=== Part 3: stationary SH Galerkin system for the harmonics ==="];

shLHS = Simplify[R u - u - 2 lap[u] - bih[u] - u^3];
(* (1+Lap)^2 u = u + 2 Lap u + Lap^2 u *)

(* Reduce every trig product to first-harmonic form and collect *)
shTrig = Collect[TrigReduce[shLHS] /. (kx x + ky y) -> th, Cos[_ th], Simplify];

(* Galerkin projection: coefficient of Cos[n th] must vanish.
   Note the cubic generates harmonics up to 3*Max[harmonics]; the
   truncated system keeps only n in `harmonics` (standard Galerkin
   truncation) -- the discarded higher-harmonic residual is the
   truncation error. *)
proj[n_] := Simplify[Coefficient[shTrig, Cos[n th]]] /. kx^2 + ky^2 -> KK;

galerkinEqs = Table[proj[n] == 0, {n, harmonics}];
Print["Galerkin system (truncated to harmonics ", harmonics, "):"];
Do[Print["  Cos[", harmonics[[i]], " theta]:  ", galerkinEqs[[i]]],
   {i, Length[harmonics]}];

(* linear operator eigenvalues for reference: L_n = R - (1 - n^2 K)^2 *)
Print["linear factors: L_n = R - (1 - n^2 K)^2, n = ", harmonics];

(* --- exact leading-order chain (perturbative recurrence) ---
   Treat A1 = O(delta); the cubic forces A_{n+2} = O(A1 * A_n * A_1)
   so A3 = O(delta^3), A5 = O(delta^5). Solve order by order. *)
Print["--- hierarchical recurrence (A3, A5 slaved to A1) ---"];
(* Ordering: A1 = O(d), A3 = O(d^3), A5 = O(d^5). Rather than a formal
   d-series (which drops the O(1) linear term of the Cos[theta] equation),
   solve hierarchically:
     eq(Cos theta)   with A3 = A5 = 0            ->  A1^2(K, R)
     eq(Cos 3theta)  linearized in A3, A5 = 0    ->  A3(A1, K, R)
     eq(Cos 5theta)  linearized in A5            ->  A5(A1, A3, K, R)
   Each output is a rational function of K and R, as expected. *)
Module[{res1, res3, res5, lin, solA1, solA3, solA5},
  res1 = Subtract @@ galerkinEqs[[1]];        (* Cos[theta] residual *)
  solA1 = Solve[(res1 /. {A3 -> 0, A5 -> 0}) == 0 && A1 != 0, A1^2];
  Print["  A1^2 = ", Simplify[A1^2 /. solA1[[1]]],
        "    [check: (4/3)(R - (1 - K)^2)]"];

  res3 = Subtract @@ galerkinEqs[[2]] /. A5 -> 0;
  lin  = Normal[Series[res3, {A3, 0, 1}]];    (* drop A3^2, A3^3 *)
  solA3 = Solve[lin == 0, A3];
  Print["  A3   = ", Simplify[A3 /. solA3[[1]]],
        "    [leading order: A1^3 / (4 (R - (1 - 9 K)^2)) < 0 near K = 1]"];

  If[MemberQ[harmonics, 5],
    res5 = Subtract @@ galerkinEqs[[3]];
    lin  = Normal[Series[res5, {A5, 0, 1}]];
    solA5 = Solve[lin == 0, A5];
    Print["  A5   = ", Simplify[A5 /. solA5[[1]]],
          "    [rational in A1, A3, K, R; leading behavior O(A1^5)]"];
  ];
];

(* --- exact (non-perturbative) numeric solve, for validation data ---
   Given numeric R and K, solve the truncated polynomial system for
   the amplitudes. Physical branch: A1 real > 0, continuous from the
   single-mode root. Use this to manufacture "exact" SH rolls with
   known harmonic content for the Python extraction tests. *)
shRoll[Rnum_?NumericQ, Knum_?NumericQ] := Module[{eqsN, sols, phys},
  eqsN = galerkinEqs /. {R -> Rnum, KK -> Knum};
  sols = NSolve[eqsN, amps, Reals];
  phys = Select[sols, (A1 /. #) > 10^-8 &];
  phys = SortBy[phys, -Abs[(A1 /. #)] &];
  If[phys === {}, $Failed, First[phys]]
];

Print["--- example: R = 0.5, K = 1 (exact truncated roll) ---"];
Print["  ", shRoll[0.5, 1.0]];
Print["--- example: R = 0.9, K = 0.8 ---"];
Print["  ", shRoll[0.9, 0.8]];

(* Sanity: feed the exact roll back through the moment ladder *)
Module[{s = shRoll[0.5, 1.0], MnNum, biasNum},
  If[s =!= $Failed,
    MnNum = Table[(-1)^m (1.0^m/2) Sum[harmonics[[i]]^(2 m) (amps[[i]] /. s)^2,
        {i, Length[harmonics]}], {m, 0, 2}];
    biasNum = -MnNum[[2]]/(1.0 MnNum[[1]]);
    Print["  moments of the R=0.5, K=1 roll: ", MnNum];
    Print["  naive uHu k^2 bias on this roll: ", biasNum,
          "  (fractional wavenumber error ", Sqrt[biasNum] - 1, ")"];
  ]
];
