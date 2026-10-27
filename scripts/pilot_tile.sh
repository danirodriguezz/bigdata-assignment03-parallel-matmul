#!/usr/bin/env bash
# Pilot (Phase 3): choose the tile size of the blocked kernel and gauge runtimes.
# Sequential baseline pinned to core 0 (logical CPUs 0,6); 1 process x (1 warm-up + 3 reps).
set -u
cd "$(dirname "$0")/.."
RM=".venv/bin/python .claude/skills/benchmark-protocol/scripts/run_measured.py"
OUT=results/raw/pilot_tile.csv
ENVS="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
for n in 1024 2048; do
  IN="--a data/inputs/n${n}_s42_A.bin --b data/inputs/n${n}_s42_B.bin"
  for bs in 32 64 128 256; do
    $RM --out $OUT --warmup 0 --reps 1 --timeout 600 \
      --label assignment=A3 experiment=pilot_tile method=seq_blocked language=c n=$n dtype=float64 seed=42 threads=1 \
      --env $ENVS -- taskset -c 0,6 bin/matmul_seq --n $n $IN --kernel blocked --bs $bs --warmup 1 --reps 3
  done
  $RM --out $OUT --warmup 0 --reps 1 --timeout 600 \
    --label assignment=A3 experiment=pilot_tile method=seq_ikj language=c n=$n dtype=float64 seed=42 threads=1 \
    --env $ENVS -- taskset -c 0,6 bin/matmul_seq --n $n $IN --kernel ikj --warmup 1 --reps 3
  for bs in 64 128; do
    $RM --out $OUT --warmup 0 --reps 1 --timeout 600 \
      --label assignment=A3 experiment=pilot_tile method=omp_blocked language=c n=$n dtype=float64 seed=42 threads=6 \
      --env OMP_NUM_THREADS=6 OMP_PLACES=cores OMP_PROC_BIND=close OMP_DYNAMIC=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      -- bin/matmul_omp --n $n $IN --kernel blocked --bs $bs --warmup 1 --reps 3
  done
done
