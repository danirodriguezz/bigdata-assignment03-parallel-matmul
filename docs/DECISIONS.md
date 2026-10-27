# Decisions log — Assignment 03

Record every design decision with date, the options considered and Daniel's answer.
Sibling assignments read this file to stay consistent.

| Date | Decision | Choice | Rationale / who decided |
|---|---|---|---|
| 2026-10-27 | Benchmark budget | ≤ 90 min machine time for this assignment; 600 s max per process run; 8 GB memory | Daniel |
| 2026-10-27 | Hardware counters (perf) | Not available (perf_event_paranoid=4) — state as limitation | Daniel |
| 2026-10-27 | Other open choices (language, methods, threads, repo name) | Recommended defaults from CLAUDE.md, delegated to the agent | Daniel delegated |
| 2026-10-27 | Language / framework | C (gcc 13.3, `-std=c11`) + OpenMP (libgomp) | Recommended default, delegated by Daniel |
| 2026-10-27 | Equivalent sequential baseline | The SAME source `src/c/matmul.c` compiled without `-fopenmp` (`bin/matmul_seq`); the pragmas are ignored and the team-size helpers return a 1-thread team | Guarantees same algorithm, loop order, layout, type, flags and inputs; only the OpenMP runtime differs |
| 2026-10-27 | Reuse of A2 kernel | Not copied: at the time of implementation `../Assignment02_Optimised_and_Sparse` had no commit (no validated version). Kernels implemented here; the loop nests (`ii-kk-jj-i-k-j` blocked and `i-k-j`) are the same as in A2's draft `src/c/dense/matmul_dense.c`. Validated here by `make test` | Agent (rule: do not wait for A2) |
| 2026-10-27 | Source layout | One kernel source + build variants (`bin/matmul_{seq,omp,omp_scalar,omp_avx2,omp_fma}`) instead of separate `src/c/{seq,omp,simd}` copies | Avoids code duplication, makes the equivalence of the variants explicit (only flags differ) |
| 2026-10-27 | Parallel decomposition | Rows of C split into p contiguous ranges of n/p (±1) rows inside one `omp parallel` region (blocked kernel); `omp parallel for schedule(static)` over rows for the ikj kernel | Disjoint writes (no races, no locks); perfect balance up to one row (block-row distribution would cap E at 0.89 for p=6) |
| 2026-10-27 | FP contraction | `-std=c11` (ISO mode) implies `-ffp-contract=off`: no FMA even with `-march=native`; FMA studied as a separate variant `omp_fma` (`-ffp-contract=fast`) | Found while inspecting the disassembly; keeps scalar/AVX2 results bit-identical |
| 2026-10-27 | `-O2` vs `-O2 -fno-tree-vectorize` | Produce identical `.text` for `matmul.c` (gcc 13 very-cheap cost model does not vectorise the j loop); the main sweep is therefore scalar code | Evidence: `results/evidence/vectorisation_evidence.csv` |
| 2026-10-27 | Affinity policy (main sweep) | `OMP_PLACES=cores OMP_PROC_BIND=close OMP_DYNAMIC=false`; sequential baseline pinned with `taskset -c 0,6` (= OpenMP place 0, physical core 0) | One fixed policy; seq and p=1 run on the same core so T_par(1)/T_seq isolates the runtime overhead |
| 2026-10-27 | Thread counts | 1,2,3,4,5,6 (≤ physical cores), 8,10,12 (SMT), 16,24 (oversubscription, > 12 logical CPUs) | Recommended default |
| 2026-10-27 | Library threads | `OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` set for every run (no BLAS is linked; NumPy only in tests) | Brief: control internal library threads |
| 2026-10-27 | Interference monitoring | Harness records `other_busy_cpus` (CPU time of other processes during each repetition, /proc/stat) and `cpu_util`; all rows kept | Other agents work on the laptop at the same time |
| 2026-10-27 | Repository name | `bigdata-assignment03-parallel-matmul` (local repo only; Daniel creates the remote and pushes) | Recommended default |
