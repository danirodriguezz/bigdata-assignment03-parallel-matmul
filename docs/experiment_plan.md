# Experiment plan — Assignment 3 (parallel matrix multiplication, optional vectorisation)

Written before the full benchmark (Phase 1); budget figures updated after the pilot (Phase 3).

## Research question
**RQ.** On a six-core/twelve-thread laptop CPU (Intel Core i7-9750H), how does the number of OpenMP
threads p affect the runtime, speedup S(p) = T_seq/T_p and efficiency E(p) = S(p)/p of a
cache-blocked dense matrix multiplication, relative to an equivalent sequential baseline, and for
which matrix sizes n does parallel execution pay off?

- **Factor varied:** thread count p ∈ {1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24}; matrix size n as a
  second factor. Secondary factors (separate experiments): kernel memory behaviour (blocked vs
  un-tiled ikj), SIMD code generation (scalar vs AVX2 vs AVX2+FMA) at fixed p.
- **Outcomes:** kernel runtime (median, IQR), S(p), E(p), T_par(1)/T_seq, peak RSS.

## Hypotheses (predictions, to be checked against the results)
- **H1** For large n, S(p) grows almost linearly up to the 6 physical cores but stays below p,
  partly because the all-core turbo frequency is lower than the single-core turbo.
- **H2** Using the second hardware thread of each core (p = 8–12, SMT) gives little additional
  speedup for this compute-bound scalar kernel.
- **H3** Oversubscription (p = 16, 24 > 12 logical CPUs) gives no gain and may be slower.
- **H4** For small n (256) fork/join and load-imbalance overheads limit S(p) well below the
  large-n values.
- **H5** The parallel binary with one thread is within a few percent of the sequential baseline.
- **H6** The un-tiled ikj kernel scales worse than the blocked kernel at n = 2048, where B (32 MiB)
  exceeds the 12 MiB L3 cache (memory-bandwidth limit).
- **H7** AVX2 vectorisation gives a further speedup at fixed p, and its relative gain is smaller at
  p = 6 than at p = 1 if the vectorised kernel becomes memory-bound.

## Implementations (one source file, several builds)
| Binary | Flags | Role |
|---|---|---|
| `bin/matmul_seq` | `-O2` (no `-fopenmp`) | equivalent sequential baseline T_seq |
| `bin/matmul_omp` | `-O2 -fopenmp` | parallel version (main sweep), p threads |
| `bin/matmul_omp_scalar` | `-O2 -fno-tree-vectorize -fopenmp` | SIMD extension: scalar |
| `bin/matmul_omp_avx2` | `-O3 -march=native -fopenmp` | SIMD extension: AVX2 (no FMA, ISO C) |
| `bin/matmul_omp_fma` | `-O3 -march=native -ffp-contract=fast -fopenmp` | SIMD extension: AVX2 + FMA |

All: `-std=c11`, gcc 13.3, float64, row-major contiguous, same input files.
Kernels: `blocked` (tile size bs chosen by the pilot) and `ikj` (no tiling).

## Inputs
`scripts/gen_inputs.py`: NumPy `default_rng(42)` (PCG64), A then B ~ U[-1, 1), float64, raw
little-endian files + SHA-256 (`data/inputs/checksums.sha256`). Same files for all methods.

## Correctness (before any benchmark)
`make test`: identity, zero, known 2×2/3×3, random vs naive ijk with the componentwise bound
2γ_n|A||B| (γ_n = nu/(1−nu), u = 2⁻⁵³), odd sizes, n < p, tile sizes 1/7/64/200; bit-identity of the
parallel result for p ∈ {1,…,24}; end-to-end test of every binary vs NumPy and vs the sequential
binary (`tests/test_vs_numpy.py`). Race-freedom: disjoint row ownership + ThreadSanitizer run.

## Measurement protocol
- Timer: `clock_gettime(CLOCK_MONOTONIC)` around one kernel call (**kernel** boundary: zeroing C +
  product; for OpenMP also the fork/join of the team, which was created before timing by a probe
  parallel region). Input loading/allocation outside the timer. Checksum of C after every rep.
- Process runs via `run_measured.py` (lock, peak RSS via GNU time, timeout 600 s).
- Repetitions: R = 5 interleaving rounds × m in-process measured reps (+ w warm-ups) per size:
  see `configs/experiment.json` (`inproc_reps`). Order shuffled within each round.
- Statistic: **median** of the measured reps for T_seq and T_p; IQR for variability; 95%
  bootstrap CI for S(p). E(p) = S(p)/p.
- Memory: peak RSS (GNU time `%M`, KiB) per run; analytical array storage 3·8n² bytes.
- Thread/affinity: `OMP_PLACES=cores OMP_PROC_BIND=close OMP_DYNAMIC=false`, seq pinned with
  `taskset -c 0,6`; `OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`. Real team size and CPU of each thread
  recorded in every row; `OMP_DISPLAY_AFFINITY` saved in `results/evidence/`.
- Interference: `other_busy_cpus` per rep (CPU time of other processes / wall time) and loadavg.
  Rule: all rows are kept; the main analysis uses all rows (median is robust); a sensitivity
  analysis repeats S(p) with reps where `other_busy_cpus > 1.0` excluded. Configurations with many
  contaminated reps are re-run in extra rounds (all rows kept).
- Frequency evidence (no hardware counters): separate `freq_probe` runs sampling cpufreq.

## Sizes, budgets, stopping rule
- Correctness: n ∈ {1, 2, 3, 5, 67, 100, 129, 130, 131, 203, 256, 257}.
- Main sweep (blocked): n ∈ {256, 512, 1024, 2048, 3072}. ikj: n ∈ {1024, 2048}.
  Vectorisation: n ∈ {1024, 2048}, p ∈ {1, 6}.
- Budgets (agreed with Daniel): ≤ 600 s per process run, ≤ 8 GB memory (3·8·3072² = 226 MB, far
  below), ≤ 90 min total machine time for this assignment's benchmarks.
- A configuration that fails (timeout/oom/error) is recorded with its status and not repeated; no
  time is ever assigned to it.
