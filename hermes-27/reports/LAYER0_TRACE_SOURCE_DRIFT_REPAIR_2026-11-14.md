# HERMES-27: layer-0 scratch tracer compatibility and fail-closed dispatch

**Date:** 2026-11-14. **Scope:** CPU/offline source patcher and its tests only.
This does **not** change the PP2 inference source, installed runtimes,
GPU Manager, TabbyAPI, model weights, or any live service.

## Reproduced fault

After publication of opt-in sliding KV, the PP2 `decode_step` entry gained
a sliding-window preflight block between the asynchronous dispatch and
`ds.ever_advanced=true`. The previously published scratch-only layer0
patcher expected those two lines to be adjacent. Running its exact
`test_pp2_layer0_trace_patch.py` against GitHub HEAD
`67f288d10da317bd625be6705a30dcf18f571875` produced **5 ERROR,
1 PASS**. The patcher failed closed with `PP2 source layout changed`;
no tensor was captured and no CUDA kernel was executed.

## Repair and safeguards

- The patcher's exact serial anchor now uses the unique adjacent lines
  `ds.ever_advanced=true;` and `cudaSetDevice(0);` following sliding
  preflight. It also uniquely checks the real asynchronous dispatch point.
- An added guard **before** the asynchronous branch returns an explicit
  error if a scratch-trace request would enter an asynchronous or sliding
  state. This matters because sliding mode automatically creates a private
  async context even when `HERMES_PP2_ASYNC_STREAMS` is absent.
- The layer0 hook still verifies precisely token **6205** at position **0**
  followed by token **161421** at position **1**, on the same `DecodeState`.
  It cannot claim independent llama.cpp provenance from these runtime
  sidecars alone. The verifier must still check physical tensor files,
  model hash, source hash, executable and explicit token/position contexts.
- Before writing trace data, the directory must be an existing canonical
  `/tmp/h27-*` location. Any symlinked or noncanonical path fails closed.
  Files are created exclusively; partially written captures are invalid.
- Updated negative source-contract tests require both exact anchors,
  reject missing/reordered anchors, prevent repatching or overwriting an
  existing scratch output and confirm early async/sliding rejection.

## Actual validation

| Gate | Observed result |
| --- | --- |
| Previous patcher against published sliding-KV PP2 source | **FAIL: 5 ERROR, 1 PASS** (expected source drift discovery) |
| Updated six focused tests on a **fresh Git archive** of published HEAD with only patcher/test overlaid | **6/6 PASS** |
| Patched PP2 scratch source generation with unique source anchors | **PASS** |
| CMake Release CUDA 12.8 `hermes_generate` compile and device link, separate temporary snapshot | **PASS** |
| Real GPU stage tensor capture | **BLOCKED / NOT RUN** |
| Independent llama GDN/attention comparison | **BLOCKED / NOT RUN** |

The first attempted retest accidentally reused a scratch directory whose
PP2 file was **already instrumented** by the prior build. Its 5 ERROR and
1 FAIL were correctly caused by the explicit double-patching guard. No
production source was touched; the final valid test ran against a **new**
`git archive` and passed. This distinction prevents wrongly treating an
expected scratch refusal as a production regression.

Published PP2 fwd source at integration input:
`f063341cadfc57dbcdc6314f48e4620017d362b9f49c58c741f8bf88284b8aa9`.
Final bounded scratch patch:
`6a263d7b36c6425fda3cc69e002926186cd135cf432f742d3fda9fa84d19c9aa`.
New CUDA-linked **scratch-only** `hermes_generate`:
`fe846142a9efd6bbc9d19d5e2a6c836723c7aa331dba550f61a3a07915c0c06b`.
These artifacts reside solely under
`/tmp/h27-tracer-serialguard-20261114` and
`/tmp/h27-tracer-serialguard-patched-20261114`.

**Remaining numerical release gate:** authenticated physical llama/HERMES
two-token post-convolution stages and a full-vocabulary 12/12 comparison
with unchanged thresholds. Idle GPU memory is not an exclusive lease;
all-route ingress, in-flight SSE requests and two-GPU owner identity
remain unverified. Production **NO-GO**.
