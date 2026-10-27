/*
 * bench_main.c -- timing harness for the kernels in matmul.c (Assignment 3).
 *
 * Usage:
 *   matmul_<variant> --n N --a A.bin --b B.bin [--kernel blocked|ikj] [--bs 64]
 *                    [--warmup 1] [--reps 5] [--dump C.bin] [--round R]
 *
 * A.bin / B.bin: n*n little-endian float64 values, row-major (written by scripts/gen_inputs.py).
 * Input loading and allocation happen OUTSIDE the timed region. The timed region ("kernel"
 * boundary) is exactly one call to the kernel, which zeroes C and computes C = A*B (for the
 * OpenMP build it also includes forking and joining the thread team of that call).
 *
 * One JSON line is printed per repetition (warm-up repetitions are flagged is_warmup=true);
 * run_measured.py turns every line into a CSV row. The checksum (sum of all C entries) is
 * computed after each repetition, which consumes the result and prevents dead-code elimination.
 *
 * --round R is only a label (the interleaving round of the experiment driver), echoed as "round".
 *
 * Extra diagnostic fields per repetition:
 *   cpu_util        process CPU time (user+sys, getrusage) / wall time of the repetition
 *                   (~p for a perfectly busy p-thread team)
 *   other_busy_cpus CPU time used by OTHER processes during the repetition / wall time,
 *                   from /proc/stat (10 ms resolution, meaningful only for reps >~ 0.5 s);
 *                   used to detect interference from other activity on the shared laptop.
 */
#define _GNU_SOURCE
#include "matmul.h"

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef BUILD_FLAGS
#define BUILD_FLAGS "unknown"
#endif
#ifndef VARIANT
#define VARIANT "unknown"
#endif

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static double own_cpu_s(void)
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (double)ru.ru_utime.tv_sec + 1e-6 * (double)ru.ru_utime.tv_usec +
           (double)ru.ru_stime.tv_sec + 1e-6 * (double)ru.ru_stime.tv_usec;
}

/* Machine-wide busy CPU time in seconds (all CPUs), from the first line of /proc/stat. */
static double machine_busy_s(void)
{
    FILE *f = fopen("/proc/stat", "r");
    if (!f)
        return -1.0;
    unsigned long long v[10] = {0};
    int got = fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu",
                     &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8], &v[9]);
    fclose(f);
    if (got < 8)
        return -1.0;
    /* user nice system idle iowait irq softirq steal: busy = all but idle and iowait */
    unsigned long long busy = v[0] + v[1] + v[2] + v[5] + v[6] + v[7];
    return (double)busy / (double)sysconf(_SC_CLK_TCK);
}

static double *alloc_matrix(int n)
{
    size_t bytes = (size_t)n * (size_t)n * sizeof(double);
    size_t rounded = (bytes + 63) / 64 * 64;
    double *p = aligned_alloc(64, rounded);
    if (!p) {
        fprintf(stderr, "Cannot allocate memory (%zu bytes)\n", bytes);
        exit(2);
    }
    return p;
}

static void read_matrix(const char *path, double *M, int n)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        exit(1);
    }
    size_t count = (size_t)n * (size_t)n;
    if (fread(M, sizeof(double), count, f) != count) {
        fprintf(stderr, "%s: expected %zu float64 values\n", path, count);
        exit(1);
    }
    fclose(f);
}

static const char *env_or(const char *name, const char *dflt)
{
    const char *v = getenv(name);
    return v ? v : dflt;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s --n N --a A.bin --b B.bin [--kernel blocked|ikj] [--bs 64]\n"
            "          [--warmup 1] [--reps 5] [--dump C.bin] [--round R]\n", prog);
    exit(1);
}

int main(int argc, char **argv)
{
    int n = 0, bs = MATMUL_DEFAULT_BS, warmup = 1, reps = 5, round_tag = -1;
    const char *a_path = NULL, *b_path = NULL, *dump = NULL, *kernel = "blocked";
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc)
            usage(argv[0]);
        if (!strcmp(argv[i], "--n")) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--a")) a_path = argv[++i];
        else if (!strcmp(argv[i], "--b")) b_path = argv[++i];
        else if (!strcmp(argv[i], "--bs")) bs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup")) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps")) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kernel")) kernel = argv[++i];
        else if (!strcmp(argv[i], "--dump")) dump = argv[++i];
        else if (!strcmp(argv[i], "--round")) round_tag = atoi(argv[++i]);
        else usage(argv[0]);
    }
    if (n <= 0 || !a_path || !b_path || bs <= 0 || reps < 1 || warmup < 0)
        usage(argv[0]);
    int use_blocked = !strcmp(kernel, "blocked");
    if (!use_blocked && strcmp(kernel, "ikj"))
        usage(argv[0]);

    double *A = alloc_matrix(n), *B = alloc_matrix(n), *C = alloc_matrix(n);
    read_matrix(a_path, A, n);
    read_matrix(b_path, B, n);
    memset(C, 0, (size_t)n * n * sizeof(double)); /* touch every page of C before timing */

    /* Record the real team size and the CPU each thread runs on (outside the timed region). */
    int team = 1;
    char cpus[1024];
    int off = snprintf(cpus, sizeof cpus, "[");
#ifdef _OPENMP
    int cpu_of[1024];
#pragma omp parallel
    {
#pragma omp single
        team = omp_get_num_threads();
        int t = omp_get_thread_num();
        if (t < 1024)
            cpu_of[t] = sched_getcpu();
    }
    for (int t = 0; t < team && t < 1024 && off < (int)sizeof cpus - 8; ++t)
        off += snprintf(cpus + off, sizeof cpus - off, "%s%d", t ? "," : "", cpu_of[t]);
#else
    off += snprintf(cpus + off, sizeof cpus - off, "%d", sched_getcpu());
#endif
    snprintf(cpus + off, sizeof cpus - off, "]");

    const double flops = 2.0 * (double)n * (double)n * (double)n;
    for (int r = 0; r < warmup + reps; ++r) {
        const double busy0 = machine_busy_s(), cpu0 = own_cpu_s();
        const double t0 = now_s();
        if (use_blocked)
            matmul_blocked(A, B, C, n, bs);
        else
            matmul_ikj(A, B, C, n);
        const double t1 = now_s();
        const double cpu1 = own_cpu_s(), busy1 = machine_busy_s();
        const double wall = t1 - t0;

        double checksum = 0.0;
        for (size_t idx = 0; idx < (size_t)n * n; ++idx)
            checksum += C[idx];

        const double own = cpu1 - cpu0;
        const double other = (busy0 >= 0 && busy1 >= 0) ? ((busy1 - busy0) - own) / wall : -1.0;
        printf("{\"time_s\": %.9f, \"timer_boundary\": \"kernel\", \"repetition\": %d, "
               "\"is_warmup\": %s, \"checksum\": %.17g, \"memory_metric\": \"array_storage_estimate\", "
               "\"memory_bytes\": %zu, \"kernel\": \"%s\", \"bs\": %d, \"variant\": \"%s\", "
               "\"openmp\": %d, \"threads_actual\": %d, \"cpus\": %s, \"gflops\": %.4f, "
               "\"cpu_util\": %.3f, \"other_busy_cpus\": %.3f, \"omp_num_threads\": \"%s\", "
               "\"omp_proc_bind\": \"%s\", \"omp_places\": \"%s\", \"omp_dynamic\": \"%s\", "
               "\"round\": %d, \"build_flags\": \"%s\"}\n",
               wall, r - warmup < 0 ? r : r - warmup, r < warmup ? "true" : "false", checksum,
               (size_t)3 * n * n * sizeof(double), kernel, use_blocked ? bs : 0, VARIANT,
               matmul_is_openmp(), team, cpus, flops / wall / 1e9, own / wall, other,
               env_or("OMP_NUM_THREADS", ""), env_or("OMP_PROC_BIND", ""),
               env_or("OMP_PLACES", ""), env_or("OMP_DYNAMIC", ""), round_tag, BUILD_FLAGS);
        fflush(stdout);
    }

    if (dump) {
        FILE *f = fopen(dump, "wb");
        if (!f || fwrite(C, sizeof(double), (size_t)n * n, f) != (size_t)n * n) {
            perror(dump);
            return 1;
        }
        fclose(f);
    }
    free(A);
    free(B);
    free(C);
    return 0;
}
