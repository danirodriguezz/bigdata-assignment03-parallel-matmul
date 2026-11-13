#!/usr/bin/env bash
# Pilot 3 (Phase 3): tile size of the blocked kernel with the parallel binary (p=6) and the
# sequential baseline, bs in {64,128,256}, two interleaved process runs per configuration.
# Run after pilot_tile2 showed that bs=256 was fastest sequentially at n=2048.
set -u
cd "$(dirname "$0")/.."
RM=".venv/bin/python .claude/skills/benchmark-protocol/scripts/run_measured.py"
EXP=pilot_tile3
OUT=results/raw/${EXP}.csv
for rnd in 1 2; do
for n in 512 1024 2048; do
  IN="--a data/inputs/n${n}_s42_A.bin --b data/inputs/n${n}_s42_B.bin"
  for bs in 64 128 256; do
    $RM --out $OUT --warmup 0 --reps 1 --timeout 600 \
      --label assignment=A3 experiment=$EXP method=seq_blocked language=c n=$n dtype=float64 seed=42 threads=1 \
      --env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      -- taskset -c 0,6 bin/matmul_seq --n $n $IN --kernel blocked --bs $bs --warmup 1 --reps 3 --round $rnd
    $RM --out $OUT --warmup 0 --reps 1 --timeout 600 \
      --label assignment=A3 experiment=$EXP method=omp_blocked language=c n=$n dtype=float64 seed=42 threads=6 \
      --env OMP_NUM_THREADS=6 OMP_PLACES=cores OMP_PROC_BIND=close OMP_DYNAMIC=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      -- bin/matmul_omp --n $n $IN --kernel blocked --bs $bs --warmup 1 --reps 3 --round $rnd
  done
done
done
