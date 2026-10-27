---
name: parallel-matmul
description: Technical guide for Assignment 3 — shared-memory parallel dense matrix multiplication (OpenMP in C, or Java threads) vs an equivalent sequential baseline, thread-count sweep including 1 thread, physical vs logical cores, affinity, oversubscription, speedup/efficiency with stated statistics, 1-thread overhead, superlinear speedup analysis, turbo-frequency effects, and the optional SIMD vectorisation extension with evidence. Use when designing, coding, testing or analysing Assignment 3.
---

# Parallel matrix multiplication (+ optional vectorisation)

## 0. Language and baseline
- Recommended: **C + OpenMP** (`gcc -O2 -fopenmp`), reusing the **best sequential dense kernel of Assignment 2**
  (`../Assignment02_*/src/c/dense/`, e.g. `blocked_ikj` or `reorder_ikj`) as the **equivalent sequential baseline**:
  same algorithm, layout (row-major contiguous float64), block size and inputs. Record provenance (path + commit).
  If A2 is not ready, implement `reorder_ikj` (and/or `blocked_ikj`) here, validate it and say so.
- The sequential baseline must be **compiled without `-fopenmp`** (pure sequential binary) — this is T_seq. The parallel
  binary run with `OMP_NUM_THREADS=1` is the **"parallel with 1 thread"** configuration (overhead measurement).
- Alternative: Java (`ExecutorService`/`ForkJoinPool`/parallel streams) reusing A2's Java kernel; then also control
  JIT warm-up and the common pool size (`-Djava.util.concurrent.ForkJoinPool.common.parallelism`).
- A comparison with the naive ijk (A1) may be added, **never as a replacement** for the equivalent baseline.

## 1. Parallelisation design (explain in the report)
- Parallelise the **outer row loop of C** (or the outer block-row loop when blocked): `#pragma omp parallel for
  schedule(static)` — each thread writes disjoint rows of C ⇒ **no data races, no locks, no reductions**.
  A, B are read-only and shared. Optional: `schedule(dynamic, chunk)` or `collapse(2)` as a separate variant.
- Because each C element is computed by one thread in the same order as the sequential code, results should be
  **bit-identical** to the sequential baseline: test exact equality (strong correctness evidence) for all p.
- Safety evidence: argue disjoint writes + optionally run a small case under ThreadSanitizer
  (`gcc -fsanitize=thread -fopenmp -g`, note TSan may report false positives with libgomp unless built for it —
  state it) or under Helgrind if available.
- Allocation and first-touch: allocate/initialise C (and inputs) outside the timed region; mention first-touch is
  irrelevant on this single-NUMA-node laptop (`lscpu`: 1 NUMA node).

## 2. Thread configurations
Reference laptop: i7-9750H = **6 physical cores, 12 logical (Hyper-Threading)**, 1 socket (re-check with lscpu).
- p ∈ {1, 2, 3, 4, 5, 6} (≤ physical), {8, 10, 12} (SMT siblings), {16, 24} (**oversubscription**: more threads than
  hardware contexts — label as such in tables and plots, e.g. hatched/hollow markers).
- Affinity: choose **one** policy for the main sweep and keep it fixed, e.g. `OMP_PLACES=cores OMP_PROC_BIND=close`
  (threads fill distinct physical cores first while p ≤ 6) — check with `OMP_DISPLAY_AFFINITY=true` that this is
  what actually happens on this CPU and save that output in `results/`. An affinity comparison (close vs spread vs
  unbound) is an optional **separate** study.
- Control internal library threads: `OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` (no BLAS used, but state it);
  `OMP_DYNAMIC=false`; verify the real thread count (e.g. print `omp_get_num_threads()` in the JSON).
- Sizes: a few n that span cache regimes, e.g. n ∈ {512, 1024, 2048, 3072} (+ 4096 if within budget); speedup
  depends on n — show it.

## 3. Measurement & metrics
- Timer: `omp_get_wtime()` or `clock_gettime(CLOCK_MONOTONIC)` around the kernel; boundary `kernel` for both
  sequential and parallel (thread-team creation inside the timed region for the parallel version — state it; the
  first parallel region pays team start-up: do a warm-up run in-process).
- ≥ 5 measured reps (+ 1 warm-up) per (n, p). Use **medians** for T_seq and T_p (state it), bootstrap CI for S
  (`speedup.py --resource threads`). E = S/p.
- Report **T_par(1)/T_seq** (overhead of the parallel version with 1 thread).
- Memory: peak RSS per (n, p) (thread stacks add little; show it) + analytical 3·8n².
- Optional CPU utilisation (`/usr/bin/time %P`) to support explanations.

## 4. Explaining the curves (evidence, not assumptions)
- **Turbo / frequency scaling**: single-core turbo (up to 4.5 GHz on this CPU) is higher than all-core turbo, so
  E(p) drops even with perfect parallel code. Evidence: sample `/proc/cpuinfo` MHz or `perf stat -e cycles` +
  `task-clock` during runs (cycles/time ≈ effective frequency). `perf` needs `sudo sysctl -w kernel.perf_event_paranoid=1`
  (ask Daniel). Laptop thermal throttling: interleave configurations, record loadavg, check variability over time.
- **SMT (p > 6)**: two threads share one core's execution units and L1/L2 → small or negative gains for compute-bound
  kernels; memory-bound kernels may gain more.
- **Memory bandwidth bound**: if E(p) saturates early for large n with the naive kernel but not with the blocked one,
  that indicates bandwidth limits (Roofline idea — Williams et al. 2009, verify).
- **Superlinear speedup**: possible when the per-thread working set fits in private L1/L2 caches; demonstrate with
  counters (LLC/L2 misses per thread count) or a size where it disappears; otherwise treat it as suspicious and
  re-check the baseline and timing.
- **Amdahl's law** (Amdahl 1967) to discuss sequential fractions (thread start-up, imbalance). Gustafson (1988) only
  if you vary n with p.

## 5. Optional vectorisation extension (keep it separate from threads)
- Variants at **fixed p** (p = 1 and p = 6): `-O2 -fno-tree-vectorize` (scalar) vs `-O3 -march=native` (auto-vectorised
  AVX2/FMA) and/or explicit AVX2 intrinsics (`_mm256_fmadd_pd`) or `#pragma omp simd`.
- **Evidence that vectorisation occurs**: `-fopt-info-vec-optimized` compiler report (save it in `results/`) and
  disassembly (`objdump -d bin | grep -c vfmadd231pd`, ymm registers) of the kernel.
- FMA changes rounding ⇒ results no longer bit-identical: switch to the tolerance check (Higham bound) and say why.
- Report the vectorisation speedup at fixed p, then the combined effect separately.

## 6. Analysis checklist
Runtime vs p per n (with IQR); S(p) vs p with ideal line and markers for SMT/oversubscription region; E(p) vs p;
T_par(1)/T_seq; speedup vs n at p = 6; memory vs p; (optional) vectorisation table/figure; discussion of when
parallelism helps (small n: overhead dominates; large n: scales until cores/bandwidth saturate).

## Hypotheses (predictions to label as such)
H1 S(p) grows nearly linearly up to the 6 physical cores for large n; H2 p > 6 (SMT) yields little gain; H3
oversubscription does not help and may hurt; H4 for small n, overhead limits speedup; (H5 vectorisation gives a
further speedup independent of p, if done).

## References to verify (candidates)
Dagum & Menon 1998 (OpenMP), OpenMP 5.x specification (official), Amdahl 1967, Gustafson 1988, Williams et al. 2009,
Hennessy & Patterson 2017, Hoefler & Belli 2015, GCC auto-vectorisation docs, Intel Intrinsics Guide (official).
