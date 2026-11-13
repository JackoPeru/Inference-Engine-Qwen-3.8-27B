# Benchmark provenance and invalidated references

All performance measurements must identify the **exact commit, model file,
CUDA toolkit, concurrency, prompt tokens, and decode/prefill path**. Compare
HERMES-27 and ExLlamaV3 only under matching workloads and clearly label MTP.

## Historical data — not current regression oracles

- `golden_m2.json`: **INVALID as a correctness oracle** after the November 13
  fixes for GGUF `ssm_a` (already `-exp(A_log)`) and GDN Q/K head order (`h%16`).
  The old E1/E2 CPU paths shared an incorrect interpretation. Do not assert
  model equivalence from this fixture.
- `baseline_m0*_raw.json`: early ExLlamaV3 performance observations, not a
  same-checkpoint, same-decoding-config comparison to the current engine.
- `P3_BASELINE.md` and `gpu_pp2_baseline.json`: PP2 performance **before**
  the new GGUF semantic fixes and serving integration; retain for historical
  performance comparison only.

For current semantic validation first run `tests/gate_math_gguf.py`, then
`tests/compare_llama.py` against an independent llama.cpp server using the
**same actual GGUF** and identical token IDs. See `tests/README.md`.

For a measured CUDA bottleneck breakdown see `PROFILING_GUIDE.md`. Do not
assume that removing allocations or CUDA Graphs alone will fix throughput.
