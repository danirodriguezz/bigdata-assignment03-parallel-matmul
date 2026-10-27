#!/usr/bin/env python3
"""End-to-end correctness test of the benchmark binaries on the shared input files.

For a small n (not a multiple of the tile size) every binary is run through its real command-line
interface with --dump, and its result is compared with
  1. NumPy (float64 matmul, trusted reference): |C - C_np| <= 2 * gamma_n * (|A| |B|) element-wise,
     gamma_n = n u / (1 - n u), u = 2^-53 (Higham's componentwise bound for both computations);
  2. the sequential baseline: bit-identical for the parallel build at p in {1, 2, 6, 7, 12, 24}
     and for the auto-vectorised AVX2 build (no FMA contraction, same operation order);
     the FMA build only needs to satisfy the tolerance of point 1.
Exit status 0 = all checks passed. Run: .venv/bin/python tests/test_vs_numpy.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gen_inputs import generate, paths  # noqa: E402

N, SEED = 203, 7  # small dedicated test input (seed differs from the benchmark seed)


def run(binary, a, b, out, threads=None, kernel="blocked", bs=64):
    env = dict(os.environ, OMP_PROC_BIND="false", OMP_DYNAMIC="false")
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)
    subprocess.run([str(ROOT / "bin" / binary), "--n", str(N), "--a", str(a), "--b", str(b),
                    "--kernel", kernel, "--bs", str(bs), "--warmup", "0", "--reps", "1",
                    "--dump", str(out)], check=True, env=env, stdout=subprocess.DEVNULL)
    return np.fromfile(out, dtype="<f8").reshape(N, N)


def main():
    generate(N, SEED)
    pa, pb, _ = paths(N, SEED)
    A = np.fromfile(pa, dtype="<f8").reshape(N, N)
    B = np.fromfile(pb, dtype="<f8").reshape(N, N)
    ref = A @ B
    u = 2.0 ** -53
    gamma = N * u / (1 - N * u)
    bound = 2 * gamma * (np.abs(A) @ np.abs(B))
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "C.bin"
        for kernel in ("blocked", "ikj"):
            seq = run("matmul_seq", pa, pb, out, kernel=kernel)
            cases = [("matmul_seq", None, "tol")]
            cases += [("matmul_omp", p, "bit") for p in (1, 2, 6, 7, 12, 24)]
            cases += [("matmul_omp_scalar", 6, "bit"), ("matmul_omp_avx2", 1, "bit"),
                      ("matmul_omp_avx2", 6, "bit"), ("matmul_omp_fma", 1, "tol"), ("matmul_omp_fma", 6, "tol")]
            for binary, p, mode in cases:
                C = run(binary, pa, pb, out, threads=p, kernel=kernel)
                err = np.abs(C - ref)
                ok_tol = bool(np.all(err <= bound))
                ok_bit = bool(np.array_equal(C.view(np.uint64), seq.view(np.uint64)))
                ok = ok_tol and (ok_bit if mode == "bit" else True)
                failures += not ok
                print(f"{'PASS' if ok else 'FAIL'} {kernel:7s} {binary:18s} p={p!s:4s} "
                      f"max|err|/bound={np.max(err / bound):.3e} bit-identical-to-seq={ok_bit}")
    print("test_vs_numpy:", "PASSED" if failures == 0 else f"FAILED ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
