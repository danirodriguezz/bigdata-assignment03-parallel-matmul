# Progress / handoff — Assignment 03

**Updated:** 2026-11-16. No benchmark of this assignment is running.

| Phase | State |
|---|---|
| 0 Orientation | DONE |
| 1 Plan | DONE |
| 2 Implement | DONE (`make test` passes; binaries record `-falign-loops=32`) |
| 3 Pilot | DONE |
| 4 Benchmark | DONE. Run 3 complete (5 rounds × 104 jobs, 0 failures). Rounds 1–2 on 2026-11-13, rounds 3–5 on 2026-11-16 |
| 5 Analysis | DONE. `make analysis` regenerates everything (checked from scratch: identical numbers) |
| 6 Report | DONE. `report/main.pdf` (10 pages: ≈ 8 main + references + appendix), compiles cleanly |
| 7 Review/package | `docs/review_checklist.md` written; `docs/defensa_oral.md` complete; zip built with `make package` |

Machine time used: pilots ~6 + run1 22 + run2 22 + run3 22 ≈ 72 min (budget 90 min).

## Pending — needs Daniel
1. Confirm the "Use of Generative AI" statement in `report/main.tex` (and README §7).
2. Create the GitHub repo `bigdata-assignment03-parallel-matmul` (user danirodriguezz) and push:
   `git remote add origin git@github.com:danirodriguezz/bigdata-assignment03-parallel-matmul.git && git push -u origin main`
3. The official submission window (9–13 Nov 2026) has passed: check late-submission rules with the lecturer.
