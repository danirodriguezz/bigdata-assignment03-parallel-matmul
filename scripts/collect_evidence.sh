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

L="-falign-loops=32"   # same LAYOUT flag as the Makefile
declare -A FLAGS=(
  [seq]="-O2 $L" [omp]="-O2 -fopenmp $L" [omp_scalar]="-O2 -fno-tree-vectorize -fopenmp $L"
  [omp_avx2]="-O3 -march=native -fopenmp $L" [omp_fma]="-O3 -march=native -ffp-contract=fast -fopenmp $L")

# ---- 1. vectorisation report + disassembly counts -------------------------------------------
{
  echo "variant,flags,vectorized_loops_reported,scalar_mulsd,scalar_addsd,packed_mulpd,packed_addpd,fma_pd,fma_sd,ymm_refs,text_sha256"
  for v in seq omp omp_scalar omp_avx2 omp_fma; do
    f="${FLAGS[$v]}"
    rm -f "$EV/vec_report_$v.txt"   # -fopt-info appends: start from an empty report
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

# ---- 1b. code placement of the hot inner loop (alignment artefact of run 1) -----------------
# For the seq/omp binaries built WITHOUT (default, as in run 1) and WITH -falign-loops=32, find the
# innermost loop (the backward jne of the c[j] += aik*b[j] loop) of the blocked kernel (tile_update)
# and of the ikj kernel, and report whether it straddles a 32-byte window and a 64-byte cache line.
{
  echo "layout,binary,kernel,loop_start,loop_end,crosses_32B_window,crosses_64B_line"
  for layout in default align32; do
    lf=""; [ "$layout" = align32 ] && lf="-falign-loops=32"
    for v in seq omp omp_scalar; do
      f="-O2"; [ "$v" = omp ] && f="-O2 -fopenmp"; [ "$v" = omp_scalar ] && f="-O2 -fno-tree-vectorize -fopenmp"
      gcc -std=c11 -Wno-unknown-pragmas $f $lf -DVARIANT="\"$v\"" src/c/matmul.c src/c/bench_main.c -o "$TMP/pl_$v" -lm
      objdump -d --no-show-raw-insn "$TMP/pl_$v" > "$TMP/pl_$v.s"
      for k in blocked ikj; do
        fn="tile_update"; [ $k = ikj ] && { [ $v != seq ] && fn="matmul_ikj._omp_fn.0" || fn="matmul_ikj"; }
        # the inner loop is the first backward jne after the first mulsd of the function
        line=$(awk -v fn="<$fn>:" '$2==fn{f=1;next} f&&/^$/{exit} f' "$TMP/pl_$v.s" | awk '/mulsd/{m=1} m&&/jne/{print; exit}')
        end=$(echo "$line" | awk '{print $1}' | tr -d ':'); start=$(echo "$line" | awk '{print $3}')
        e=$((16#$end + 2)); s=$((16#$start))
        c32=$([ $((s / 32)) -ne $(((e - 1) / 32)) ] && echo yes || echo no)
        c64=$([ $((s / 64)) -ne $(((e - 1) / 64)) ] && echo yes || echo no)
        echo "$layout,matmul_$v,$k,0x$start,$(printf '0x%x' $e),$c32,$c64"
      done
    done
  done
} > "$EV/loop_placement.csv"
cat "$EV/loop_placement.csv"

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
