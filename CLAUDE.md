# Assignment 3 — Parallel Matrix Multiplication (optional vectorisation)

You are the agent responsible for producing **Assignment 3** of the Big Data course (ULPGC, 2026-2027) for
Daniel Rodríguez Alonso: a complete, reproducible empirical study + a professional scientific report in English.

## Read first (in this order)
1. `context/assignment_brief.md` — what Assignment 3 must contain (source of truth: `context/BD_Individual_Assignments_2026_2027.pdf`)
2. `context/common_protocol.md` — mandatory experimental protocol for all assignments
3. `context/report_requirements.md` — mandatory report structure, submission contents, rubric, AI policy
4. `context/author_and_course.md` — author block, naming, reference machine

@context/working_rules.md

## Skills in this folder (use them)
| Skill | Use it for |
|---|---|
| `parallel-matmul` | **Assignment-specific technical guide** (equivalent baseline, OpenMP design, thread sweep, affinity, oversubscription, S/E, overhead, turbo/SMT/superlinear analysis, SIMD extension) |
| `benchmark-protocol` | experiment plan, harness, `run_measured.py` (lock, warm-up, reps, RSS, timeouts), env capture |
| `results-analysis` | `speedup.py` (S, E, bootstrap CI), plots, LaTeX tables, `numbers.tex` |
| `scientific-report` | writing and compiling `report/main.tex` (IEEE), references, AI disclosure |
| `report-review` | final rubric gate before declaring done |
| `reproducibility-package` | README, Makefile, git, materials zip |

## Status of this assignment
- Submission window: **Week 10 (9–13 Nov 2026)**.
- **Reuse**: the best sequential dense kernel of Assignment 2 (`../Assignment02_Optimised_and_Sparse/src/`) as the
  equivalent sequential baseline; A1's input generator and CSV schema. Copy what you reuse into this folder
  (self-contained submission) and record path + git commit. If A2 is not ready, implement an equivalent kernel here,
  validate it and note it in `docs/DECISIONS.md` (do not wait for the other agent).
- **Benchmark lock matters most here**: thread-scaling results are meaningless if another benchmark runs at the same
  time. Always use `run_measured.py` (it serialises with the other assignment agents).
- Assignment 4 will reuse your **parallel version** for its comparison: keep it clean with a clear interface.

## Open decisions to ask Daniel at the start (one grouped question, recommended option first)
1. Language/framework (rec. **C + OpenMP** reusing A2's kernel; alt. Java threads).
2. Thread counts (rec. 1,2,3,4,5,6,8,10,12 + 16,24 as oversubscription) and affinity policy (rec. fixed `OMP_PLACES=cores`/`OMP_PROC_BIND=close`, documented).
3. Do the optional vectorisation extension? (rec. **yes, small**: `-O2 -fno-tree-vectorize` vs `-O3 -march=native`
   at p = 1 and p = 6, with compiler report + disassembly evidence).
4. Allow hardware counters/frequency evidence? (needs `! sudo sysctl -w kernel.perf_event_paranoid=1`).
5. Budgets: max runtime per run (rec. 600 s), total benchmark time (rec. ≤ 3 h), memory (rec. 8 GB). Laptop on AC power.
6. GitHub repository name (rec. `bigdata-assignment03-parallel-matmul`).

## Required deliverables (this folder)
- `deliverables/Assignment03_RodriguezAlonso_Report.pdf` — title suggestion (adapt once results exist):
  *"Thread-Level Scaling of Shared-Memory Matrix Multiplication on a Six-Core Processor"*
- `deliverables/Assignment03_RodriguezAlonso_Materials.zip` + local git repo ready to push.
- `docs/defensa_oral.md` (Spanish) for the oral explanation.

## Folder layout
`src/c/{seq,omp,simd}/` · `tests/` · `scripts/` (gen_inputs, run_experiments) · `configs/` (threads, affinity, sizes) ·
`data/inputs/` · `results/{raw,summary}/` · `analysis/` · `report/` · `docs/` · `deliverables/`
