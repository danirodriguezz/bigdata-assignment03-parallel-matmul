#!/usr/bin/env bash
# Collect the non-timing evidence used in the report (written to results/evidence/):
#   1. vectorisation: gcc -fopt-info-vec report and instruction counts in the disassembly of
#      the kernels for every build variant;
#   2. linkage: the sequential binary does not link libgomp (it is a purely sequential program);
#   3. affinity: OMP_DISPLAY_AFFINITY output of the parallel binary for every thread count of the
#      sweep, with the affinity policy of the experiment (OMP_PLACES=cores, OMP_PROC_BIND=close);
#   4. data races: a ThreadSanitizer build of the test program (see caveat in the output file).
# Usage: bash scripts/collect_evidence.sh   (from the assignment root; needs `make build`)
set -u
cd "$(dirname "$0")/.."
EV=results/evidence
mkdir -p "$EV"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

declare -A FLAGS=(
  [seq]="-O2" [omp]="-O2 -fopenmp" [omp_scalar]="-O2 -fno-tree-vectorize -fopenmp"
  [omp_avx2]="-O3 -march=native -fopenmp" [omp_fma]="-O3 -march=native -ffp-contract=fast -fopenmp")

# ---- 1. vectorisation report + disassembly counts -------------------------------------------
{
  echo "variant,flags,vectorized_loops_reported,scalar_mulsd,scalar_addsd,packed_mulpd,packed_addpd,fma_pd,fma_sd,ymm_refs,text_sha256"
  for v in seq omp omp_scalar omp_avx2 omp_fma; do
    f="${FLAGS[$v]}"
    gcc -std=c11 -Wall -Wextra -Wno-unknown-pragmas $f -fopt-info-vec-optimized=$EV/vec_report_$v.txt \
        -c src/c/matmul.c -o "$TMP/$v.o"
    objdump -d --no-show-raw-insn "$TMP/$v.o" > "$EV/disasm_matmul_$v.txt"
    d="$EV/disasm_matmul_$v.txt"
    cnt() { grep -cE "$1" "$d" || true; }
    nvec=$(grep -c "loop vectorized" "$EV/vec_report_$v.txt" || true)
    sha=$(objcopy -O binary --only-section=.text "$TMP/$v.o" "$TMP/$v.text" && sha256sum "$TMP/$v.text" | cut -c1-16)
    echo "$v,\"$f\",$nvec,$(cnt '\smulsd\s|\svmulsd\s'),$(cnt '\saddsd\s|\svaddsd\s'),$(cnt '\s(v)?mulpd\s'),$(cnt '\s(v)?addpd\s'),$(cnt 'vfmadd[0-9]+pd'),$(cnt 'vfmadd[0-9]+sd'),$(cnt '%ymm'),$sha"
  done
} > "$EV/vectorisation_evidence.csv"
cat "$EV/vectorisation_evidence.csv"

# Innermost-loop excerpt of the blocked kernel (AVX2 build) for the report/appendix
grep -nE 'vmulpd|vaddpd|vfmadd' "$EV/disasm_matmul_omp_avx2.txt" | head -20 > "$EV/disasm_excerpt_avx2.txt"
grep -nE 'vfmadd' "$EV/disasm_matmul_omp_fma.txt" | head -20 > "$EV/disasm_excerpt_fma.txt"

# ---- 2. linkage -----------------------------------------------------------------------------
{
  for v in seq omp; do echo "== bin/matmul_$v"; ldd "bin/matmul_$v"; done
} > "$EV/linkage.txt"

# ---- 3. affinity (tiny run, 1 rep; under the machine lock so it does not disturb benchmarks) --
AFF="$EV/affinity_display.txt"
: > "$AFF"
for p in 1 2 3 4 5 6 8 10 12 16 24; do
  echo "== p=$p OMP_PLACES=cores OMP_PROC_BIND=close" >> "$AFF"
  flock ~/.cache/bigdata_bench.lock env OMP_NUM_THREADS=$p OMP_PLACES=cores OMP_PROC_BIND=close OMP_DYNAMIC=false \
      OMP_DISPLAY_AFFINITY=true OMP_AFFINITY_FORMAT="thread %0.2n bound to CPUs {%A}" \
      bin/matmul_omp --n 256 --a data/inputs/n256_s42_A.bin --b data/inputs/n256_s42_B.bin \
      --warmup 0 --reps 1 2>&1 | grep -E '^thread|"cpus"' | sed -E 's/.*("threads_actual": [0-9]+, "cpus": \[[0-9,]*\]).*/\1/' >> "$AFF"
done
echo "OMP_DISPLAY_AFFINITY of p=1..24 -> $AFF"
echo "== OMP_DISPLAY_ENV (p=6)" >> "$AFF"
flock ~/.cache/bigdata_bench.lock env OMP_NUM_THREADS=6 OMP_PLACES=cores OMP_PROC_BIND=close OMP_DYNAMIC=false \
    OMP_DISPLAY_ENV=true bin/matmul_omp --n 64 --a data/inputs/n256_s42_A.bin --b data/inputs/n256_s42_B.bin \
    --warmup 0 --reps 1 2>&1 | grep -E "OMP_|GOMP" >> "$AFF"

# ---- 4. ThreadSanitizer ---------------------------------------------------------------------
# The full test program is far too slow under TSan (bit-identity loop over p = 1..24), so TSan is
# run on the benchmark harness itself: both kernels, n = 300 (not a multiple of the tile), p = 2, 6, 12.
TS="$EV/tsan.txt"
{
  echo "ThreadSanitizer build: gcc -std=c11 -fsanitize=thread -fopenmp -g -O1 src/c/matmul.c src/c/bench_main.c"
  echo "Caveat: the system libgomp is not instrumented for TSan, so its internal synchronisation"
  echo "(barriers) is invisible to TSan; reports inside libgomp would be false positives."
  if gcc -std=c11 -fsanitize=thread -fopenmp -g -O1 -DVARIANT='"tsan"' src/c/matmul.c src/c/bench_main.c -o "$TMP/tsan" -lm 2>&1; then
    head -c $((300*300*8)) data/inputs/n512_s42_A.bin > "$TMP/A300.bin"
    head -c $((300*300*8)) data/inputs/n512_s42_B.bin > "$TMP/B300.bin"
    for k in blocked ikj; do for p in 2 6 12; do
      echo "== kernel=$k p=$p n=300 bs=64"
      flock ~/.cache/bigdata_bench.lock env OMP_NUM_THREADS=$p OMP_PROC_BIND=false TSAN_OPTIONS="halt_on_error=0" \
          setarch "$(uname -m)" -R "$TMP/tsan" --n 300 --a "$TMP/A300.bin" --b "$TMP/B300.bin" \
          --kernel $k --bs 64 --warmup 0 --reps 1 2>&1 | grep -E "WARNING|SUMMARY|checksum" | sed -E 's/.*("checksum": [^,]*).*/\1/'
      echo "exit status: ${PIPESTATUS[0]}"
    done; done
    # Full report of one run, to classify the warnings: which source lines and which memory
    # (stack addresses 0x7ff... = OpenMP shared-variable block / locals; heap = matrices A, B, C).
    OMP_NUM_THREADS=2 OMP_PROC_BIND=false setarch "$(uname -m)" -R "$TMP/tsan" --n 300 --a "$TMP/A300.bin" \
        --b "$TMP/B300.bin" --kernel blocked --bs 64 --warmup 0 --reps 1 > /dev/null 2> "$EV/tsan_full_blocked_p2.txt"
    echo "== classification of the full report (blocked, p=2): accessed addresses"
    grep -oE "(Read|Write) of size [0-9]+ at 0x[0-9a-f]+" "$EV/tsan_full_blocked_p2.txt" | awk '{print $NF}' \
        | sed -E 's/^0x7ff.*/stack/; s/^0x[0-9a-f]+$/non-stack/' | sort | uniq -c
    echo "== reports whose access is inside tile_update (the only code that writes C during the product):"
    grep -cE "#0 tile_update" "$EV/tsan_full_blocked_p2.txt" || true
  else
    echo "TSan build failed"
  fi
} > "$TS" 2>&1
grep -cE "WARNING: ThreadSanitizer" "$TS" | sed 's/^/TSan warnings: /'
echo "evidence written to $EV/"
