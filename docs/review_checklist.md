# Review checklist — Assignment 03 (report-review gate, 2026-11-16)

Reviewer stance: critical. ✅ = satisfied (where), ❌ = not satisfied (fix / reason), 👤 = needs Daniel.

## A. Mandatory structure
- ✅ Title block: Daniel Rodríguez Alonso, "Big Data -- Individual Assignment 03", research title.
- ✅ Abstract: question, method (equivalent baseline, p = 1–24, n = 256–3072, 5 rounds, medians, bootstrap CI),
  main result with variability (S(6) = 4.79, 95 % CI [3.93, 5.80] at n = 3072), scope sentence. *(Fix applied:
  CI added.)*
- ✅ Introduction: one RQ (italic), factor p (+ n), outcomes, hypotheses H1–H7 written before the benchmark.
- ✅ Background: blocking [lam1991], libraries [goto2008], OpenMP [dagum1998, openmp52], Amdahl, Gustafson,
  roofline, SMT [marr2002], turbo [intel9750h], benchmarking [hoefler2015, mytkowicz2009], error bound [higham2002].
- ✅ Methodology: implementations, equivalent baseline, inputs, environment (Table I), correctness, TSan,
  protocol, interference, frequency, tile pilot, code placement / three runs, sizes and budgets. *(Fix applied:
  `-falign-loops=32` and run 1/2/3 story added; env table shows both kernel versions.)*
- ✅ Results: Fig. 1, Tables II–V, Figs. 2–5, appendix Tables VI–VII; units, k, IQR/CI; failures (0) stated;
  day effect and per-round values above p reported.
- ✅ Discussion: explicit answer to the RQ, verdict for H1–H7, explanations tied to measurements, conjectures
  labelled, limitations (i)–(vi).
- ✅ Conclusions: scoped (machine, compiler, kernel, binding, schedule); unresolved questions listed.
- ✅ References (IEEEtran, 13 entries) + Reproducibility Materials (repo URL; "last commit of main") + Use of
  Generative AI.

## B. Assignment-specific requirements (context/assignment_brief.md)
- ✅ Equivalent sequential baseline + parallel implementation (OpenMP): Methodology §III-A.
- ✅ Same algorithm, type, layout, inputs; difference documented (only the OpenMP runtime; libgomp linkage).
- ✅ Several thread counts including 1: p ∈ {1,…,6,8,10,12,16,24}.
- ✅ Physical cores, logical processors, affinity recorded (Table I, §III-C, `affinity_display.txt`, per-row
  `cpus`/`threads_actual`); oversubscription identified (p > 12, hatched region).
- ✅ Common protocol (warm-ups, reps, rounds, medians, IQR, lock).
- ✅ Correctness and safe access to shared data (tests, bit-identity, checksums, disjoint rows, TSan with caveat).
- ✅ Internal library threads controlled (OPENBLAS/MKL = 1, OMP_DYNAMIC=false).
- ✅ S(p), E(p) defined; statistic stated (median; ratio of medians, bootstrap CI).
- ✅ Runtime, variability, memory, speedup, efficiency reported (Table II, Fig. 1, Fig. 5, appendix).
- ✅ Parallel implementation with one thread (Table III; plus the run-1/2 artefact, Table IV).
- ✅ Unusual / superlinear speedup explained with evidence (per-round S > p traced to baseline noise; p = 8 dip
  explained with affinity evidence and cpu_util).
- ✅ Optional vectorisation: flags, evidence that it occurs (compiler report, disassembly), controlled comparison
  at fixed p, SIMD separated from thread count (Table V, Fig. 4, Discussion H7).

## C. Experimental design & correctness
- ✅ Same inputs for all methods; generator PCG64 seed 42, U[-1,1), float64; SHA-256 checksums.
- ✅ Environment fully documented (Table I + `results/environment*.md`).
- ✅ Timer boundary = kernel, identical for all comparisons; loading outside the timer.
- ✅ Warm-ups (1 or 0 per size) excluded; k = 150/50/15/10/5, ≥ 5 everywhere; median + IQR.
- ✅ Memory in every configuration: peak RSS via GNU time %M (named); analytical storage shown separately.
- ✅ Correctness tests incl. identity, zero, known, random vs reference; tolerance 2γ_n|A||B| justified [higham2002].
- ✅ Largest attempted size n = 3072 completed in all configurations; 0 timeouts/OOM; budgets stated. *(Fix
  applied: explicit sentence in Results.)*

## D. Integrity
- ✅ Number traceability: every number in Results/Discussion/abstract comes from `report/generated/numbers.tex`
  or generated tables. Spot-check recomputed directly from `results/raw/*.csv` (independent pandas script,
  2026-11-16), all equal: T_seq(3072) = 16.7 s, T_6(3072) = 3.49 s, S(6, 3072) = 4.79, k_seq(3072) = 5,
  S(6, 1024) = 4.99, T_par(1)/T_seq(3072) = 0.985, S_ikj(6, 2048) = 5.16, AVX2/scalar(p=1, 1024) = 2.30,
  GFLOP/s FMA(p=6, 2048) = 36.9, f(1) = 3770 MHz, f(6) = 2900 MHz, RSS_seq(3072) = 217.5 MiB,
  measured rows = 3210, S(12, 3072) = 4.63.
  Hand-derived model predictions in the Discussion (0.67, 0.83, 1, 0.89) are labelled as model values from the
  formula m·6/p with m read from `affinity_display.txt`.
- ✅ No placeholders: `grep TODO/todo{/XX/??` on main.tex and on the PDF text → none (only the `\todo` macro
  definition).
- ✅ References: 13 entries verified on 2026-10-27/2026-11-13 (Crossref DOI or official page, noted in the .bib).
  New uses this session checked for support: Mytkowicz et al. (code layout/link order biases measurements) for the
  code-placement paragraph; gcc 13.3 Optimize-Options page documents `-falign-loops` and `-ffp-contract`.
- 👤 AI disclosure present and specific (tool, model, purposes, affected parts); **Daniel must confirm the wording**.
- ✅ No universal claims; abstract and conclusions scoped to this machine, compiler and kernel.

## E. Communication
- ✅ Figures: labelled axes with units, legends name configurations, captions give statistic, error bars, affinity;
  Okabe–Ito colour-blind-safe palette; all referenced. *(Fix applied: legends moved off the data in Figs. 1, 3, 5.)*
- ✅ English: consistent terminology (S, E, T_seq, T_p, SMT, oversubscribed), abbreviations defined at first use.
- ✅ Length: references start on page 8 → main body ≈ 8 pages (+ references + 1-page appendix = 10 pages). *(Fix
  applied: redundant runtime figure dropped from the report.)* LaTeX log: no errors, undefined references or
  overfull boxes. *(Fix applied: 4 overfull boxes removed.)*

## F. Reproducibility package
- ✅ README: build/run/analysis commands, dependency versions, hardware, expected runtime, run history. *(Fix
  applied: flags with `-falign-loops=32`, two-day note, run history, bench time.)*
- ✅ `make analysis report` from scratch (deleted `results/summary`, `report/figures`, `report/generated`):
  regenerated, `report/generated` byte-identical to before.
- ✅ Raw CSVs with schema (`results/README.md`, now listing archived runs and both environment captures);
  input generator + checksums.
- ✅ Reused materials: A2 path + commit `9c21829` (no code copied; same loop nest and tile size).
- ✅ `deliverables/Assignment03_RodriguezAlonso_Report.pdf` and `..._Materials.zip` built by `make package` from the
  committed tree (`git archive`: excludes `.venv`, `bin/`, input `.bin` files).
- 👤 Git: everything committed locally; remote URL in report/README, **Daniel must create the repo and push**.

## G. Oral defence
- ✅ `docs/defensa_oral.md` (Spanish): RQ, fairness, code walkthrough, correctness, 8 key results with numbers,
  limitations, live reproduction commands, Q&A (run 1/2/3, p = 8 dip, efficiency, superlinear rounds, two days,
  vectorisation evidence, FMA, tile size).

## Items for Daniel
1. Confirm the "Use of Generative AI" statement (report + README §7).
2. Create `danirodriguezz/bigdata-assignment03-parallel-matmul` on GitHub and push `main`.
3. Submission window (9–13 Nov 2026) has passed — check with the lecturer.
