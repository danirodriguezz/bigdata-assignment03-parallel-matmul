# Assignment 03 — Thread-Level Scaling of Shared-Memory Matrix Multiplication on a Six-Core Processor

**Author:** Daniel Rodríguez Alonso · Grado en Ciencia e Ingeniería de Datos · ULPGC
**Course:** Big Data, academic year 2026–2027 · **Report:** `deliverables/Assignment03_RodriguezAlonso_Report.pdf`
**Repository:** https://github.com/danirodriguezz/bigdata-assignment03-parallel-matmul

**Research question.** On a six-core / twelve-thread laptop CPU (Intel Core i7-9750H), how does the number of
OpenMP threads affect the runtime, speedup S(p) = T_seq/T_p and efficiency E(p) = S(p)/p of a cache-blocked
dense matrix multiplication relative to an equivalent sequential baseline, and for which sizes n does parallel
execution pay off? Optional extension: SIMD vectorisation (scalar vs AVX2 vs AVX2+FMA) at fixed thread count.

## 1. Environment used for the reported results
Full capture in `results/environment.md`. Intel Core i7-9750H (6 cores / 12 logical CPUs, L3 12 MiB, AVX2+FMA),
15.5 GiB RAM, Ubuntu 24.04.5 (Linux 7.0.0-42), governor `powersave` with turbo enabled, laptop on AC power.
Core k = logical CPUs {k, k+6}.

## 2. Dependencies
| Tool | Version | Notes |
|---|---|---|
| gcc | 13.3.0 | `-std=c11`; variants: seq `-O2`, omp `-O2 -fopenmp`, omp_scalar `-O2 -fno-tree-vectorize -fopenmp`, omp_avx2 `-O3 -march=native -fopenmp`, omp_fma `-O3 -march=native -ffp-contract=fast -fopenmp` |
| OpenMP runtime | libgomp (gcc 13) | |
| GNU time, taskset, flock | system | used by `run_measured.py` |
| Python | 3.12.3 | venv `.venv`; packages pinned in `requirements-lock.txt` (NumPy, pandas, Matplotlib, SciPy) |
| LaTeX | TinyTeX + IEEEtran | report only |

## 3. Reproduce
```bash
make setup      # venv + all binaries (bin/matmul_{seq,omp,omp_scalar,omp_avx2,omp_fma})
make test       # correctness tests (identity, zero, known, random vs reference, bit-identity for p = 1..24, vs NumPy)
make inputs     # shared inputs A, B for n = 256..3072 (NumPy PCG64, seed 42, U[-1,1), float64) + checksums
make env        # results/environment.md
make evidence   # vectorisation report + disassembly counts, affinity display, linkage, ThreadSanitizer
make pilot      # optional smoke test (< 1 min) -> results/raw/pilot_*.csv
make bench      # full experiments (~30 min of machine time on the machine above) -> results/raw/*.csv
make analysis   # results/raw -> results/summary, report/figures, report/generated (tables + numbers.tex)
make report     # PDF -> deliverables/Assignment03_RodriguezAlonso_Report.pdf
make package    # materials zip from the committed git tree
```
Configuration: `configs/experiment.json` (sizes, thread counts, tile size b = 256, rounds = 5, warm-ups and
measured repetitions per size, affinity, budgets). Partial re-runs:
`.venv/bin/python scripts/run_experiments.py --experiments scaling --sizes 3072 --rounds 6`.

Thread/affinity settings of every parallel run: `OMP_NUM_THREADS=p OMP_PLACES=cores OMP_PROC_BIND=close
OMP_DYNAMIC=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`; the sequential binary is pinned with
`taskset -c 0,6` (physical core 0). p ≤ 6: one thread per core; 7–12: SMT; 16, 24: oversubscribed.

**Shared laptop.** All timed runs go through `.claude/skills/benchmark-protocol/scripts/run_measured.py`,
which serialises benchmarks with the machine-wide lock `~/.cache/bigdata_bench.lock`.

## 4. Data
- Inputs: `scripts/gen_inputs.py` (raw little-endian float64, row-major); the `.bin` files are not committed,
  `data/inputs/checksums.sha256` and the JSON sidecars are.
- Raw results: `results/raw/*.csv` — schema in `results/README.md`. Main files: `scaling.csv` (blocked
  kernel, thread sweep), `scaling_ikj.csv`, `vectorisation.csv`, `freq_probe.csv`; pilots `pilot_*.csv`.
- Evidence (non-timing): `results/evidence/` (vectorisation report, disassembly, affinity, linkage, TSan).
- Logs: `results/logs/` (tests, pilots, benchmark driver).

## 5. Reused materials
No code was copied from Assignment 2 (it had no committed version when this kernel was written). The blocked
loop nest (ii-kk-jj, ikj inside the tile) and the tile size b = 256 match Assignment 2's blocked kernel
(`../Assignment02_Optimised_and_Sparse`, commit `9c21829`). The input generator and CSV schema follow the
course-wide protocol shared with Assignments 1–2.

## 6. Repository layout
```
src/c/matmul.{h,c}     kernels (blocked, ikj, naive reference); same file -> seq and OpenMP builds
src/c/bench_main.c     timing harness (JSON line per repetition)
tests/                 test_matmul.c (C unit tests), test_vs_numpy.py (end-to-end)
scripts/               gen_inputs.py, run_experiments.py, freq_probe.py, collect_evidence.sh, pilot_*.sh
configs/experiment.json
analysis/make_all.py   statistics, figures, LaTeX tables, numbers.tex
results/               raw/, summary/, evidence/, logs/, environment.md
report/                main.tex, references.bib, figures/, generated/
docs/                  DECISIONS.md, experiment_plan.md, review_checklist.md, defensa_oral.md (Spanish)
```

## 7. Use of Generative AI
Claude Code (Anthropic, model Claude Opus 5.5) was used as an assistant for planning the experiments, drafting
and debugging the C harness, the test programs, the experiment driver and the analysis scripts, and for
drafting and revising the LaTeX report. The author reviewed and ran all code; every measurement comes from runs
on the machine described above (raw data in `results/raw/`), and every reference was checked against its DOI or
official page.

## License
MIT
