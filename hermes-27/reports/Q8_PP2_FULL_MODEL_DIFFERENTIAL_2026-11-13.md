# HERMES-27: first full-model Q8_1/DP4A scratch decode differential

Date: 2026-11-13. **Result: full-model release numerics FAIL / NO-GO.**
This is an experimental *scratch-only* end-to-end validation of the
new single-projection Q8_1/DP4A implementation, not an accepted inference
mode, release commit, new production service, or a throughput claim.

## Build and configuration

An isolated source candidate was created from `git archive HEAD hermes-27`
without touching live `/home/matteo/hermes-27`. Its parent source and
candidate were placed under:

- Original snapshot: `/tmp/h27-q8-e2e-source-IEzZEO`
- Experimental only: `/tmp/h27-q8-e2e-candidate-szKYST`
- Reproducible patcher: `experiments/q8_dp4a_pp2_scratch_patch.py`

The patcher, **not production**, imports the experimental native
`nvfp4_gemv_q8_1_experimental.cu` into the scratch CUDA library and
provides the real FP32 activation (before BF16 conversion) at 14 T1
decode projection sites, with the same Q8_1 quantizer and DP4A mapping
used in the independently tested first-QKV benchmark. The switch is
`HERMES_PP2_Q8_DP4A_EXPERIMENTAL=1`; it is **absent from committed
production PP2 code** and is unavailable by default. All 14 pointer
sites are replaced only if exact source anchors match; the script
refuses a non-disposable output and prior scratch edits.

The experimental T1 GEMV scratch is allocated/freed on its CUDA stream
using `cudaMallocAsync/cudaFreeAsync` to avoid a device-wide free sync
and preserve request stream ordering. This is an isolated correctness
experiment; no claim about allocator overhead, concurrent safety or
model-wide VRAM peak is made.

CUDA 12.8 SM120a configure, compilation and device link of
`hermes_generate` against real `hermes_pp2`, `hermes_kernels` and
the new Q8 kernel **PASS**. No installed binaries, manager or systemd
units were changed.

## Authentic GGUF and independent comparator

- Actual GGUF: `Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf`,
  SHA256 `db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1`.
- NVIDIA RTX 5060 Ti 16GB, two GPU PP split (32+32 layers), CUDA 12.8.
- Candidate prompt numeric IDs:
  `6205,161421,303,220,17,1374,9911,1147,27053,6059,13`.
- Independent prior llama.cpp raw FP32 references:
  `/tmp/h27-release-worker1-X3ZZUN/llama-live` and its
  previously authenticated model/source manifests. The source is
  independent llama.cpp, not HERMES or another PP2 run.
- Candidate output:
  `/tmp/h27-q8-e2e-a11-20261113`, three
  `prompt_0_step_*.f32` files, each 248320 FP32 logits.
- Full comparator:
  `tools/compare_fp32_logits.py` with **unchanged** maximum absolute
  threshold **0.20** and maximum RMSE **0.02**.

Candidate runtime on the actual model: upload **2.76 s**, GPU model
allocation after `decode_init(capacity=13)` GPU0 9848 MiB,
GPU1 11463 MiB; 11 prompt tokens seeded in **4852.314 ms**;
two subsequent `decode_step` calls in **881.768 ms** total.
These are single-run measurements, not stabilized throughput or A/B
against the identical BF16 backend.
The generated IDs are `[271,248068,198]`, identical to the prior
llama.cpp greedy result, **but all full-vocab numerical steps FAIL**.

| Step | Max absolute error | RMSE | Greedy top-1 | Top-10 overlap |
| ---: | ---: | ---: | --- | --- |
| 0 | 0.44395447 | 0.074186224 | 271 / 271 | 9/10 |
| 1 | 0.23571253 | 0.040707038 | 248068 / 248068 | 9/10 |
| 2 | 0.33959675 | 0.073222515 | 198 / 198 | 10/10 |

Comparison process exited **1** (expected numerical FAIL); the
experimental inference executable exited **0**. No thresholds or
argmax tests were relaxed.

The original BF16 HERMES run against the same llama references
previously had worst max_abs ~0.565 and RMSE ~0.056. The new Q8 path
does not provide a uniform error reduction: step0's RMSE is worse,
and **3/3** traces still fail. Therefore changing **every** PP2 GEMV
to Q8_1 is **not sufficient** to establish full-model mathematical
equivalence. Possibilities include other differing layers/precision,
GDN recurrence, attention, FFN/head behavior, or propagation of
quantization, none of which is yet isolated by this three-step
full-vocab comparison.

## Isolated positive controls, and limits

The independent layer0 pos0 CPU Q8 input/GGUF projection reproduces
the llama QKV tensor to RMSE `1.58e-7`. The real GPU DP4A kernel
matches captured llama QKV at maxabs `3.81e-6`.
The independent CPU Q8 QKV → depthwise conv+SiLU chain additionally
reproduces captured llama conv to RMSE `2.08e-8`.
These positive controls prove only the **first projection and first
token causal conv**, not all future layers or model logits.

The run loaded HERMES and the earlier independent llama evidence
**sequentially**, never two full model copies concurrently; both
live services `hermes-tabby` and `hermes-gpu-manager` remained active.
GPU allocations returned to their 486 MiB / 1289 MiB baseline.
An attempted *new* llama exporter run was declined by the execution
environment; no alternate bypass was used. Existing authenticated
llama traces remain the comparison authority.

## Acceptance blockers

1. Capture authenticated full stage outputs from the **current**
   HERMES model across RMSNorm→QKV→conv→GDN→out→residual→FFN
   and compare with independent llama stages at same token/position
   where available. Do not assume identical first QKV implies downstream
   equivalence.
2. Verify reference's actual per-projection quantization/precision
   dispatch, including later full-attention and output-head layers.
3. Re-run 12-step independent full-vocab gate only after a specific
   proven correction and preserve strict thresholds.
4. Prove request concurrency, cancellation, KV/GDN state integrity and
   sustained TTFT/TPOT before considering any PP2/HTTP opt-in.

**Decision: NO-GO; retain the published Q8_1/DP4A as a standalone
experiment, without switching the main inference engine.**

## Bounded follow-up: only GDN QKV projections use Q8_1

To distinguish an error propagation caused by changing *all* GEMVs
from the first proven QKV quantization mismatch, the scratch patcher
adds a `HERMES_PP2_Q8_DP4A_EXPERIMENTAL_SCOPE=gdn-qkv` selector.
It applies Q8_1 only to GDN `HID=5120 -> CC=10240` QKV projections
in the token-by-token decoder; every other projection still executes
the existing BF16 Warp4 path. It is strictly **scratch-only**, not a
new production option.

Same GGUF, independent llama trace, prompt IDs, 248320 full FP32
logits/step and unchanged strict thresholds. Isolated scratch binary:
`/tmp/h27-q8-e2e-candidate-nwrBNa/build/hermes_generate`;
candidate output: `/tmp/h27-q8-gdnqkv-a11-20261113`.
CUDA12.8 compiled and linked with no errors.

| Step | All projections Q8 RMSE | GDN QKV-only Q8 RMSE | GDN QKV-only max abs |
| ---: | ---: | ---: | ---: |
| 0 | 0.074186224 | 0.062683179 | 0.38897526 |
| 1 | 0.040707038 | 0.042388489 | 0.23970795 |
| 2 | 0.073222515 | 0.057432097 | 0.2879684 |

Greedy tokens remain `[271,248068,198]`. Independent full-logit
release gate still **FAIL 3/3**. Model upload was unchanged; measured
11-token seed time 2214.540 ms, and two incremental decode steps
401.592 ms (single execution, NOT a matched benchmark).
Both GPUs returned to their previous idle allocations after test.

**Conclusion:** even a selective same-NVFP4 Q8/GDN-QKV correction
does not make the entire model numerically equivalent to llama.
Changing every projection to Q8 also does not provide monotonic
full-logit improvement. The next responsible step is authenticated
downstream GDN/full-attention intermediate comparison, not
further global quantizer substitutions or a relaxed release gate.
