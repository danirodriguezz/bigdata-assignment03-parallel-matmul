/*
 * matmul.c -- kernels declared in matmul.h.
 *
 * Compiled without -fopenmp it is the sequential baseline; with -fopenmp it is the parallel
 * version. Without OpenMP the "#pragma omp" lines are ignored and the two helper functions
 * below return the values of a one-thread team, so both builds execute the same code.
 */
#include "matmul.h"

#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#define THREAD_ID() omp_get_thread_num()
#define TEAM_SIZE() omp_get_num_threads()
#else
#define THREAD_ID() 0
#define TEAM_SIZE() 1
#endif

static inline int imin(int a, int b) { return a < b ? a : b; }

/*
 * One tile update: C[i0:i1, j0:j1] += A[i0:i1, k0:k1] * B[k0:k1, j0:j1].
 * Kept as a separate (not inlined) function so that the compiler has few live variables in the
 * innermost loop and keeps all pointers in registers (when it was inlined into blocked_rows,
 * gcc -O2 spilled the B-row pointer and the loop bound to the stack: two extra loads per
 * iteration, about 30% slower in the pilot).
 */
__attribute__((noinline))
static void tile_update(const double *restrict A, const double *restrict B, double *restrict C,
                        int n, int i0, int i1, int k0, int k1, int j0, int j1)
{
    for (int i = i0; i < i1; ++i) {
        double *restrict c = &C[(size_t)i * n];
        const double *restrict a = &A[(size_t)i * n];
        for (int k = k0; k < k1; ++k) {
            const double aik = a[k];
            const double *restrict b = &B[(size_t)k * n];
            for (int j = j0; j < j1; ++j)
                c[j] += aik * b[j];
        }
    }
}

/*
 * Blocked i-k-j product restricted to rows [lo, hi) of C.
 * Loop order: ii (row tile) -> kk (k tile) -> jj (column tile) -> [i -> k -> j inside the tile].
 * A bs x bs tile of B is reused for all rows of the current row tile, which keeps it in cache.
 * For a fixed element C[i][j] the products A[i][k]*B[k][j] are accumulated in increasing k
 * (kk ascending, then k ascending), i.e. in the same order as the plain i-k-j and i-j-k loops.
 */
static void blocked_rows(const double *restrict A, const double *restrict B,
                         double *restrict C, int n, int bs, int lo, int hi)
{
    for (int i = lo; i < hi; ++i)
        memset(&C[(size_t)i * n], 0, (size_t)n * sizeof(double));

    for (int ii = lo; ii < hi; ii += bs)
        for (int kk = 0; kk < n; kk += bs)
            for (int jj = 0; jj < n; jj += bs)
                tile_update(A, B, C, n, ii, imin(ii + bs, hi), kk, imin(kk + bs, n),
                            jj, imin(jj + bs, n));
}

void matmul_blocked(const double *restrict A, const double *restrict B,
                    double *restrict C, int n, int bs)
{
    if (bs <= 0)
        bs = MATMUL_DEFAULT_BS;
    /* One parallel region; thread t owns rows [n*t/p, n*(t+1)/p) of C.
     * Writes are disjoint (no two threads write the same row), A and B are only read:
     * no locks, atomics or reductions are needed. The implicit barrier at the end of the
     * region guarantees that C is complete when the function returns. */
#pragma omp parallel default(none) shared(A, B, C, n, bs)
    {
        const int t = THREAD_ID();
        const int p = TEAM_SIZE();
        const int lo = (int)(((long long)n * t) / p);
        const int hi = (int)(((long long)n * (t + 1)) / p);
        blocked_rows(A, B, C, n, bs, lo, hi);
    }
}

void matmul_ikj(const double *restrict A, const double *restrict B,
                double *restrict C, int n)
{
    /* schedule(static) gives each thread one contiguous chunk of about n/p rows. */
#pragma omp parallel for schedule(static) default(none) shared(A, B, C, n)
    for (int i = 0; i < n; ++i) {
        double *restrict c = &C[(size_t)i * n];
        const double *restrict a = &A[(size_t)i * n];
        memset(c, 0, (size_t)n * sizeof(double));
        for (int k = 0; k < n; ++k) {
            const double aik = a[k];
            const double *restrict b = &B[(size_t)k * n];
            for (int j = 0; j < n; ++j)
                c[j] += aik * b[j];
        }
    }
}

void matmul_naive_ijk(const double *restrict A, const double *restrict B,
                      double *restrict C, int n)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int k = 0; k < n; ++k)
                s += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = s;
        }
}

int matmul_max_threads(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int matmul_is_openmp(void)
{
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}
