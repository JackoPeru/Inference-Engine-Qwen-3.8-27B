# HERMES-27 — opt-in RAW seeded last-head-only experiment (NOT promoted)

**Status: CUDA BUILD PASS / CPU CONTRACT PASS / REAL GPU NUMERICAL GATE BLOCKED.**
No speedup, full-model quality improvement or production readiness is
claimed. No installed source/binary, active inference manager, TabbyAPI,
ingress, GGUF or service was modified.

## Observed inefficiency and bounded alternative

The real current RAW seeded path at `distributed/pp2_fwd.cu`,
`prefill_into_decode_state`, calls
`prefill_impl(..., &ds, true)` with the default `last_only=false`.
The output head therefore materializes and transfers
`T * 248320 * sizeof(float)` logits for a T-token prompt. Both
continuous and legacy HTTP serving use **only the last row**.
At T64 this is ~63.56 MB of head output instead of ~0.994 MB,
plus 64 head evaluations instead of 1. This is a source-level
redundancy, **not** a measured bottleneck contribution.

The original `prefill_impl` already has a verified-shape
`last_only` argument for its separate chunked entrypoint.
The disposable experiment
`experiments/prefill_last_head_scratch_patch.py` opts in to this
existing head subpath only for the one-shot RAW seeded API with
`HERMES_PP2_SEEDED_LAST_HEAD_ONLY=1`. It does not change the
64 PP2 layers, absolute positions, attention, residual, GDN/conv
states, KV updates, or prefill chunk placement.

The scratch adapter changes the return shape only under that opt-in,
from T*VOC to VOC, and updates both HTTP adapters and the CLI
accordingly. It verifies all remaining 248320 logits finite.
BF16 prototype quality mode is **rejected** under last-only,
preserving the existing mandatory all-position BF16 comparison.
The opt-in remains absent from production PP2/server code.

## Exact isolated build/provenance

- Published source checkpoint at experiment creation:
  `d6663e3bbd1a143fb07bcb3a8668538711f74867`.
- Actual unchanged GGUF:
  `/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf`,
  previously authenticated physical SHA
  `db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1`.
- Clean independent source snapshot:
  `/tmp/h27-prefill-lastonly-source-jFNcP1`.
- Patched only disposable candidate:
  `/tmp/h27-prefill-lastonly-candidate-npGPch`.
- Patch source scope:
  `distributed/pp2_fwd.cu`, `distributed/pp2_batch.cu`,
  `serving/server.cpp` and `tools/hermes_generate.cpp`
  **inside the scratch candidate only**.
- CUDA 12.8 SM120a `hermes_generate` and `hermes_server`
  compilation/link: **PASS**. Initial missing `strcmp`
  declaration in the experimental CLI/batch adapter was corrected in
  the patcher before the successful final build.
- Source-contract tests:
  `experiments/tests/test_prefill_last_head_patch.py` **3/3 PASS**
  (real-source anchors, tampered anchor rejection, already-patched
  source rejection).
- Offline integrated scratch `ctest`: **7/7 PASS** after compiling
  the previously unbuilt `test_gguf_synth` target; the initial
  `Not Run` due solely to missing target was not counted as a pass.
- CLI/server `--help` without model allocation: PASS.
- Post-test GPUs idle GPU0 486 MiB, GPU1 1289 MiB, services
  `hermes-tabby` and `hermes-gpu-manager` both active.

## Missing quality and performance gates

The attempted execution of the real-GGUF opt-in T1 CUDA quality test
was **blocked by the execution environment**. No retry via an alternate
route was made. Therefore **there is no T1/T11/T64 GPU last-head
quality result, no matched output-head timing, no HTTP throughput
measurement, no memory-peak measurement and no GPU stress acceptance**
for this experiment. It must remain **UNPROMOTED**.

The exact remaining experimental gate is: under exclusive GPU
ownership and when execution is permitted, run original RAW seeded
`T*VOC` and opt-in last-row-only seeded output sequentially on
T1/T11/T64 authentic GGUF prompts and compare **all 248320 values**
of the last row at the same position. Require full-vocab
maxabs <=0.002, identical argmax, equal resulting KV/GDN/conv
continuation and deterministic ten-repeat behavior; retain stricter
thresholds whenever source oracle requires them. Benchmark
N1/N2/N4/N8 with equal fixed prompt/output mix, SSE usage, cancellation,
VRAM and p95, including a BF16-mode rejection control. Do not
retconclude quality from only matching generated token IDs.

This optimization could remove redundant head calculations and
transfers but cannot by itself remove the remaining 64 layer
`cudaDeviceSynchronize` barriers, per-layer allocations, CPU
attention-gate transfers or token-loop GEMV. The observed
throughput plateau therefore remains a distinct performance
investigation.

**Decision: source-contract/build-only experiment, default unchanged,
strict independent llama.cpp full-model gate still FAIL, production NO-GO.**
