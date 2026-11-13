/*
 * test_matmul.c -- correctness tests for matmul.c (built for every variant by `make test`).
 *
 * Checks, for the blocked and ikj kernels:
 *   1. identity:  A * I == A and I * A == A exactly;
 *   2. zero:      A * 0 == 0 exactly;
 *   3. known:     a 2x2 and a 3x3 product with integer entries (exact in float64);
 *   4. random:    |C - C_ref| <= 2 * gamma_n * (|A| |B|) element-wise against the naive i-j-k
 *                 reference (Higham's componentwise bound, gamma_n = n u / (1 - n u),
 *                 u = 2^-53; the factor 2 covers rounding in both the kernel and the reference);
 *   5. bit-identity (OpenMP build only): for p in {1,2,3,4,5,6,7,8,12,16,24} the parallel result
 *                 equals the result with one thread bit for bit (no FMA contraction changes this,
 *                 because the operation sequence per element is independent of p);
 *   6. awkward shapes: n not a multiple of the tile size, n smaller than the number of threads,
 *                 n = 1, and tile sizes 1, 7, 64, 200 (plus 256, the benchmark tile size).
 * Exit status 0 = all tests passed.
 */
#include "matmul.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static int failures = 0;
static int checks = 0;

#define CHECK(cond, ...)                                     \
    do {                                                     \
        ++checks;                                            \
        if (!(cond)) {                                       \
            ++failures;                                      \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                    \
            fprintf(stderr, "\n");                           \
        }                                                    \
    } while (0)

/* Small deterministic generator (xorshift64*) -> uniform values in [-1, 1). */
static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static double urand(void)
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    uint64_t x = rng_state * 2685821657736338717ULL;
    return ((double)(x >> 11) / 9007199254740992.0) * 2.0 - 1.0;
}

static double *mat(int n) { return calloc((size_t)n * n, sizeof(double)); }

static void kernel(int which, const double *A, const double *B, double *C, int n, int bs)
{
    if (which == 0)
        matmul_blocked(A, B, C, n, bs);
    else
        matmul_ikj(A, B, C, n);
}

static const char *kname(int which) { return which == 0 ? "blocked" : "ikj"; }

static void test_identity_zero(int which, int n, int bs)
{
    double *A = mat(n), *I = mat(n), *Z = mat(n), *C = mat(n);
    for (size_t i = 0; i < (size_t)n * n; ++i)
        A[i] = urand();
    for (int i = 0; i < n; ++i)
        I[(size_t)i * n + i] = 1.0;
    for (size_t i = 0; i < (size_t)n * n; ++i)
        C[i] = 123.0; /* garbage: the kernel must overwrite C */
    kernel(which, A, I, C, n, bs);
    CHECK(memcmp(A, C, (size_t)n * n * sizeof(double)) == 0, "%s A*I != A (n=%d bs=%d)", kname(which), n, bs);
    kernel(which, I, A, C, n, bs);
    CHECK(memcmp(A, C, (size_t)n * n * sizeof(double)) == 0, "%s I*A != A (n=%d bs=%d)", kname(which), n, bs);
    kernel(which, A, Z, C, n, bs);
    int allzero = 1;
    for (size_t i = 0; i < (size_t)n * n; ++i)
        allzero &= (C[i] == 0.0);
    CHECK(allzero, "%s A*0 != 0 (n=%d bs=%d)", kname(which), n, bs);
    free(A); free(I); free(Z); free(C);
}

static void test_known(int which)
{
    const double A2[4] = {1, 2, 3, 4}, B2[4] = {5, 6, 7, 8}, R2[4] = {19, 22, 43, 50};
    double C2[4];
    kernel(which, A2, B2, C2, 2, 64);
    CHECK(memcmp(C2, R2, sizeof R2) == 0, "%s 2x2 known product", kname(which));
    kernel(which, A2, B2, C2, 2, 1);
    CHECK(memcmp(C2, R2, sizeof R2) == 0, "%s 2x2 known product, bs=1", kname(which));

    const double A3[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9}, B3[9] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    const double R3[9] = {30, 24, 18, 84, 69, 54, 138, 114, 90};
    double C3[9];
    kernel(which, A3, B3, C3, 3, 2);
    CHECK(memcmp(C3, R3, sizeof R3) == 0, "%s 3x3 known product", kname(which));
}

/* Random inputs vs naive reference with the componentwise error bound. */
static void test_random(int which, int n, int bs)
{
    double *A = mat(n), *B = mat(n), *C = mat(n), *R = mat(n);
    for (size_t i = 0; i < (size_t)n * n; ++i) {
        A[i] = urand();
        B[i] = urand();
    }
    kernel(which, A, B, C, n, bs);
    matmul_naive_ijk(A, B, R, n);
    const double u = ldexp(1.0, -53);
    const double gamma = n * u / (1.0 - n * u);
    double worst = 0.0;
    int ok = 1;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double absprod = 0.0; /* (|A| |B|)_ij */
            for (int k = 0; k < n; ++k)
                absprod += fabs(A[(size_t)i * n + k]) * fabs(B[(size_t)k * n + j]);
            double err = fabs(C[(size_t)i * n + j] - R[(size_t)i * n + j]);
            double bound = 2.0 * gamma * absprod;
            if (err > bound)
                ok = 0;
            if (absprod > 0 && err / absprod > worst)
                worst = err / absprod;
        }
    CHECK(ok, "%s random n=%d bs=%d exceeds 2*gamma_n bound", kname(which), n, bs);
    printf("  %-7s random n=%4d bs=%3d  max |err|/(|A||B|) = %.3e  (2*gamma_n = %.3e)\n",
           kname(which), n, bs, worst, 2.0 * gamma);
    free(A); free(B); free(C); free(R);
}

#ifdef _OPENMP
/* The parallel result must be bit-identical for every thread count. */
static void test_bit_identity(int which, int n, int bs)
{
    static const int ps[] = {1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24};
    double *A = mat(n), *B = mat(n), *C1 = mat(n), *Cp = mat(n);
    for (size_t i = 0; i < (size_t)n * n; ++i) {
        A[i] = urand();
        B[i] = urand();
    }
    omp_set_num_threads(1);
    kernel(which, A, B, C1, n, bs);
    for (size_t q = 0; q < sizeof ps / sizeof ps[0]; ++q) {
        omp_set_num_threads(ps[q]);
        kernel(which, A, B, Cp, n, bs);
        CHECK(memcmp(C1, Cp, (size_t)n * n * sizeof(double)) == 0,
              "%s p=%d differs from p=1 (n=%d bs=%d)", kname(which), ps[q], n, bs);
    }
    free(A); free(B); free(C1); free(Cp);
}
#endif

int main(void)
{
    printf("test_matmul: build %s, OpenMP=%d, max threads=%d\n",
#ifdef VARIANT
           VARIANT,
#else
           "?",
#endif
           matmul_is_openmp(), matmul_max_threads());
    static const int bss[] = {1, 7, 64, 200};
    for (int which = 0; which < 2; ++which) {
        test_known(which);
        for (int s = 0; s < 4; ++s) {
            test_identity_zero(which, 1, bss[s]);
            test_identity_zero(which, 5, bss[s]);
            test_identity_zero(which, 67, bss[s]);
            test_identity_zero(which, 130, bss[s]);
        }
        test_random(which, 3, 2);
        test_random(which, 100, 64);
        test_random(which, 129, 7);
        test_random(which, 257, 64);
        test_random(which, 300, 256);  /* tile size used in the benchmarks, n not a multiple */
#ifdef _OPENMP
        test_bit_identity(which, 5, 2);    /* n < p for most p */
        test_bit_identity(which, 131, 64); /* n not a multiple of bs */
        test_bit_identity(which, 256, 64);
        test_bit_identity(which, 300, 256); /* benchmark tile size */
#endif
    }
    printf("test_matmul: %d checks, %d failures -> %s\n", checks, failures, failures ? "FAILED" : "PASSED");
    return failures ? 1 : 0;
}
