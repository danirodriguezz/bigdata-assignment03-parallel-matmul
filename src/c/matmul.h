/*
 * matmul.h -- dense square matrix multiplication kernels, C = A * B (Assignment 3).
 *
 * Public interface (also the interface reused by Assignment 4):
 *   - All matrices are n x n, double precision (float64), row-major and contiguous:
 *     element (i, j) is stored at M[i * n + j].
 *   - The kernels OVERWRITE C (they zero it first); A and B are read-only.
 *   - A, B and C must not overlap (restrict).
 *
 * The SAME source file (matmul.c) is compiled twice:
 *   - without -fopenmp  -> purely sequential baseline (the OpenMP pragmas are ignored),
 *   - with    -fopenmp  -> parallel version; the number of threads is taken from the OpenMP
 *                          runtime (OMP_NUM_THREADS or omp_set_num_threads()).
 * Each element C[i][j] is computed by exactly one thread with exactly the same sequence of
 * floating-point operations as in the sequential build, so the parallel result is
 * bit-identical to the sequential result for every thread count (same compiler flags).
 */
#ifndef MATMUL_H
#define MATMUL_H

#include <stddef.h>

/* Default tile size of the blocked kernel (selected by a pilot, see docs/DECISIONS.md). */
#define MATMUL_DEFAULT_BS 64

/* Cache-blocked i-k-j kernel (the kernel studied in Assignment 3).
 * Rows of C are split into p contiguous ranges of n/p (+-1) rows, one per thread. */
void matmul_blocked(const double *restrict A, const double *restrict B,
                    double *restrict C, int n, int bs);

/* Plain loop-interchanged i-k-j kernel (no tiling); parallelised over rows of C. */
void matmul_ikj(const double *restrict A, const double *restrict B,
                double *restrict C, int n);

/* Textbook i-j-k triple loop; used only as a trusted reference in the tests (never parallel). */
void matmul_naive_ijk(const double *restrict A, const double *restrict B,
                      double *restrict C, int n);

/* Number of threads the kernels will use (1 in the sequential build). */
int matmul_max_threads(void);

/* 1 if this object file was compiled with OpenMP, 0 otherwise. */
int matmul_is_openmp(void);

#endif /* MATMUL_H */
