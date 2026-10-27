#!/usr/bin/env bash
# Pilot diagnostic (Phase 3): is the sequential blocked kernel slowed down by pinning to core 0?
# Compares pinned/unpinned runs of bin/matmul_seq (and, for reference only, a copy of the
# Assignment 2 draft binary, which has the same loop nest). Results: results/raw/pilot_diag.csv
set -u
cd "$(dirname "$0")/.."
RM=".venv/bin/python .claude/skills/benchmark-protocol/scripts/run_measured.py"
OUT=results/raw/pilot_diag.csv
A2BIN="${A2BIN:-/tmp/claude-1000/-home-daniel-Documentos-Universidad-4Year-Big-Data-Teoria-Assigments/07ba51ec-b0fb-480d-a6fe-e3f366d49270/scratchpad/a2_bench_dense}"
n=1024; IN="--a data/inputs/n${n}_s42_A.bin --b data/inputs/n${n}_s42_B.bin"
L="assignment=A3 experiment=pilot_diag language=c n=$n dtype=float64 seed=42 threads=1"
for round in 1 2; do
  $RM --out $OUT --reps 1 --label $L method=seq_blocked_pinned06 --env OMP_NUM_THREADS=1 -- taskset -c 0,6 bin/matmul_seq --n $n $IN --bs 64 --warmup 1 --reps 3
  $RM --out $OUT --reps 1 --label $L method=seq_blocked_pinned3 --env OMP_NUM_THREADS=1 -- taskset -c 3 bin/matmul_seq --n $n $IN --bs 64 --warmup 1 --reps 3
  $RM --out $OUT --reps 1 --label $L method=seq_blocked_unpinned --env OMP_NUM_THREADS=1 -- bin/matmul_seq --n $n $IN --bs 64 --warmup 1 --reps 3
  [ -x "$A2BIN" ] && $RM --out $OUT --reps 1 --label $L method=a2draft_blocked_unpinned --env OMP_NUM_THREADS=1 -- "$A2BIN" --method blocked_ikj --n $n $IN --bs 64 --warmup 1 --reps 3
  [ -x "$A2BIN" ] && $RM --out $OUT --reps 1 --label $L method=a2draft_blocked_pinned06 --env OMP_NUM_THREADS=1 -- taskset -c 0,6 "$A2BIN" --method blocked_ikj --n $n $IN --bs 64 --warmup 1 --reps 3
done
