# Assignment 3 — Parallel Matrix Multiplication and Optional Vectorisation

Source: *BD_Individual_Assignments_2026_2027.pdf*, page 6 (transcribed). Submission window: **Week 10, 9–13 November 2026**.

## Research objective
Investigate **how thread count affects the performance** of a shared-memory matrix multiplication implementation,
and **explain when parallel execution helps**.

## Required implementations and comparisons
- Provide an **equivalent sequential baseline** and a **parallel implementation** using threads or a framework such
  as OpenMP. **Comparing the parallel method with its sequential counterpart is mandatory.**
- Use the **same algorithm, numeric type, storage layout and input matrices** where possible. Document differences.
  A comparison with a much slower basic algorithm can be added, but **cannot replace the equivalent baseline**.
- Test **several thread counts** supported by the available hardware, **including one thread**. Record **physical
  cores, logical processors and any affinity settings**. **Identify oversubscription** if you test more threads than
  hardware execution contexts.
- Apply the common timing and repetition protocol. **Check correctness and safe access to shared data.** **Control
  internal library threads** so that the reported thread count describes the actual configuration.

## Metrics
Let T_seq(n) be the sequential baseline time and T_p(n) the parallel time for the same input size with p threads.
Define **speedup S(p) = T_seq(n) / T_p(n)** and **efficiency E(p) = S(p) / p**. **State which summary statistic you
use for these times.**

Report **runtime, variability, memory, speedup and efficiency**. Also report the **parallel implementation with one
thread** to reveal framework or implementation overhead. **Explain any unusual or superlinear speedup using evidence**
rather than assuming a measurement is correct.

## Optional vectorisation extension
You may implement or evaluate SIMD vectorisation. Describe compiler options or explicit instructions and **the
evidence that vectorisation occurs**. Compare vectorised and non-vectorised versions under controlled conditions.
**Clearly separate vectorisation effects from changes in thread count.** If you do not undertake this extension, no
vectorisation result is required.

## Submission
Report, code, thread configurations and measurements (see `report_requirements.md`).
