# Progress / handoff — Assignment 03

**Paused:** 2026-11-13 ~12:10 (local), at Daniel's request (laptop shutdown). No benchmark of this
assignment is running. All raw CSV rows are intact.

## Phase reached: Phase 4 (main benchmark, run 3) part-way through
| Phase | State |
|---|---|
| 0 Orientation | DONE (`docs/DECISIONS.md`, `results/environment.md`) |
| 1 Plan | DONE (`docs/experiment_plan.md`) |
| 2 Implement | DONE. `make test` passes (now also tests b = 256). Binaries depend on the Makefile (bug fixed). |
| 3 Pilot | DONE. Tile size b = 256 (`pilot_tile2/3`). Evidence collected (`results/evidence/`). |
| 4 Benchmark | **Run 3 (main, aligned) partial**: round 1 complete, round 2 81/104 jobs. |
| 5 Analysis | `analysis/make_all.py` works (tested on runs 1–2); extended with the 3-run alignment table and the frequency bound. |
| 6 Report | `report/main.tex`: abstract, Results, Discussion, Conclusions and AI statement still to write (placeholders ABSTRACT / RESULTS / DISCUSSION / CONCLUSIONS / AISTATEMENT). Intro, Background, Methodology, Materials written. **Methodology must be updated** with `-falign-loops=32` in the build flags + the run 1/2/3 story. |
| 7 Review/package | README written; `docs/defensa_oral.md` drafted (sections "RESULTADOS" to fill, Q&A about run 1 to update with the run-2 replicate). `docs/review_checklist.md` not written. |

## Key finding of this session (must go in the report)
- **Run 1** (`results/raw/run1_default_alignment/`, gcc default loop alignment): omp binary with p = 1 was
  19–26 % slower than the sequential baseline for the blocked kernel at every n, while `omp_scalar` (identical
  kernel machine code) matched seq, and omp_ikj(1) was 8–16 % *faster* than seq_ikj.
- `results/evidence/loop_placement.csv`: the slow binaries are exactly those whose hot inner loop straddles a
  **64-byte cache-line boundary** (seq-ikj, omp-blocked); fast ones do not (seq-blocked, omp-ikj, omp_scalar).
  (32-byte-window crossing is NOT the discriminator: omp-ikj crosses a 32 B window and is fast.)
- **Run 2** (`results/raw/run2_default_alignment_replicate/`): intended as the aligned run, but `make build` did
  not rebuild (stale binaries; rows record `build_flags=-O2 -fopenmp`) → it is an independent **replicate** of
  run 1 (same effect: ratio 1.21–1.27).
- Fix: `LAYOUT := -falign-loops=32` in every variant + Makefile dependency. Check under the lock
  (`results/logs/align32_check.log`): seq 0.50 s ≈ omp(1) 0.50 s (blocked), 0.53 ≈ 0.53 (ikj).
- Run-1/2 results otherwise (to compare later with run 3): S(6) ≈ 4.0 at n = 3072, p = 8 dip (SMT imbalance,
  cpu_util 6.3), p = 12 ≈ p = 6, freq 3.9 GHz (p=1) → 3.15 GHz (p=6) → min(p,6)·f(p)/f(1) = 4.85 ≈ relative speedup
  T_par(1)/T_6 ≈ 4.8; AVX2 ≈ 2.2× at p=1, ≈ 1.9–2.0× at p=6; FMA +5 %.
- Machine time used: pilots ~6 + run1 22 + run2 22 + run3 8 ≈ 58 min; remaining run 3 ≈ 14 min → ~72 min ≤ 90.

## Next steps to resume (from this folder)
```bash
make build && flock ~/.cache/bigdata_bench.lock make test     # sanity (binaries must say -falign-loops=32)
.venv/bin/python scripts/run_experiments.py --resume --dry-run | grep -c dry-run   # expect 311
.venv/bin/python scripts/run_experiments.py --resume >> results/logs/bench.log 2>&1  # ~14 min, background
make env    # optional re-capture
.venv/bin/python analysis/make_all.py      # (= make analysis)
# then: write Results/Discussion/Conclusions/abstract in report/main.tex (numbers via generated/numbers.tex),
# add \input{generated/table_alignment.tex}, update Methodology flags, compile (make report), review
# (report-review skill -> docs/review_checklist.md), finish docs/defensa_oral.md, commit, make package.
```
Close the browser (Brave) during the benchmark if possible (background noise is recorded in other_busy_cpus).

## Open questions for Daniel
1. Confirm the "Use of Generative AI" statement once written (tool: Claude Code by Anthropic, model Claude Opus 5.5).
2. Create the GitHub repo `bigdata-assignment03-parallel-matmul` (user danirodriguezz) and push (the agent cannot push).
3. Submission window ends today (13 Nov 2026).
