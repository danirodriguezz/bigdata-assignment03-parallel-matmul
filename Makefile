# Reproducibility pipeline -- Assignment 03 (Daniel Rodríguez Alonso, Big Data ULPGC 2026-2027)
NN      := 03
NAME    := Assignment$(NN)_RodriguezAlonso
VENV    := .venv
PY      := $(VENV)/bin/python
SKILLS  := .claude/skills

CC      := gcc
STD     := -std=c11 -Wall -Wextra
SRC     := src/c/matmul.c src/c/bench_main.c
HDR     := src/c/matmul.h

# Build variants: SAME source files, different flags.
#   seq        : sequential baseline (no -fopenmp: the OpenMP pragmas are ignored)
#   omp        : parallel version (main thread sweep)
#   omp_scalar : vectorisation extension, auto-vectorisation disabled
#   omp_avx2   : vectorisation extension, -O3 + AVX2 (ISO C mode => no FMA contraction)
#   omp_fma    : vectorisation extension, -O3 + AVX2 + FMA contraction allowed
FLAGS_seq        := -O2
FLAGS_omp        := -O2 -fopenmp
FLAGS_omp_scalar := -O2 -fno-tree-vectorize -fopenmp
FLAGS_omp_avx2   := -O3 -march=native -fopenmp
FLAGS_omp_fma    := -O3 -march=native -ffp-contract=fast -fopenmp
VARIANTS := seq omp omp_scalar omp_avx2 omp_fma
BINS     := $(addprefix bin/matmul_,$(VARIANTS))
TESTBINS := $(addprefix bin/test_,$(VARIANTS))

.PHONY: all setup build test inputs pilot bench env evidence analysis report package clean

all: setup test inputs bench analysis report package

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PY) -m pip install -q -r requirements.txt
	$(PY) -m pip freeze > requirements-lock.txt
	touch $(VENV)/bin/activate

setup: $(VENV)/bin/activate build    ## venv + compile all variants

build: $(BINS) $(TESTBINS)

# -Wno-unknown-pragmas: the sequential build deliberately ignores "#pragma omp".
bin/matmul_%: $(SRC) $(HDR)
	@mkdir -p bin
	$(CC) $(STD) -Wno-unknown-pragmas $(FLAGS_$*) -DVARIANT='"$*"' -DBUILD_FLAGS='"$(FLAGS_$*)"' $(SRC) -o $@ -lm

bin/test_%: tests/test_matmul.c src/c/matmul.c $(HDR)
	@mkdir -p bin
	$(CC) $(STD) -Wno-unknown-pragmas $(FLAGS_$*) -DVARIANT='"$*"' -Isrc/c tests/test_matmul.c src/c/matmul.c -o $@ -lm

test: setup                          ## correctness tests -- must pass before any benchmark
	@for v in $(VARIANTS); do OMP_PROC_BIND=false ./bin/test_$$v || exit 1; done
	$(PY) tests/test_vs_numpy.py

inputs: setup                        ## shared input files + checksums
	$(PY) scripts/gen_inputs.py --config configs/experiment.json

env:
	bash $(SKILLS)/benchmark-protocol/scripts/capture_env.sh results/environment.md

evidence: build                      ## vectorisation report, disassembly counts, affinity, linkage
	bash scripts/collect_evidence.sh

pilot: test inputs env               ## quick end-to-end smoke run (small n, 1 round)
	$(PY) scripts/run_experiments.py --config configs/experiment.json --pilot

bench: test inputs env evidence      ## full experiment (serialised by the machine-wide lock)
	$(PY) scripts/run_experiments.py --config configs/experiment.json

analysis: $(VENV)/bin/activate       ## results/raw -> results/summary, report/figures, report/generated
	$(PY) analysis/make_all.py

report:                              ## compile PDF and copy to deliverables/
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	mkdir -p deliverables && cp report/main.pdf deliverables/$(NAME)_Report.pdf

package: report                      ## materials zip from the committed git tree
	bash $(SKILLS)/reproducibility-package/templates/package.sh $(NN)

clean:
	cd report && latexmk -C || true
	rm -rf bin build
