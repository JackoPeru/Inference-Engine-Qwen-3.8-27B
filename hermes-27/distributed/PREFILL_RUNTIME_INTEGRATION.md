# PP2 batched prefill and decode-state seeding (experimental)

## Verification status

Implemented in distributed/pp2_fwd.cu and .hpp. CUDA 12.8 SM120a compilation
and full executable link against cuBLAS: **PASS**.
CPU-only contract test: **PASS (102 independent convolution-tail cases)**.
Real-GGUF GPU quality gate: **RAW-GGUF seeded T=2/11/64 PASS**. BF16
Tensor Core path **FAIL-CLOSED / NO-GO** on T=2/11/64. Full results below.
Multiuser prefill latency/throughput, T>64 and HTTP integration are
**not verified**, so this is still **NO-GO for automatic production routing**.

## New API and rollback

New explicit experimental API:

    pp2::prefill_into_decode_state(ids, decode_state, stats, error, h_last_out)

Prepare an EMPTY request state with pp2::decode_init first. Require
1 <= T <= 64 and capacity >= T; every token ID must be within vocabulary.
On failure, the state is invalidated and decode_free is called. On success,
all T prefill logits [T,248320] are returned and decode_step may continue
at absolute position T with the same state. The per-request async DecodeState
mutex is acquired with try_lock, so a concurrent step on the same state is
rejected. The caller must still prevent concurrent destruction of the state.
The normal prefill(), decode_step(), HTTP API and async-stream mode are not
changed or redirected by this explicit API.

Strict optional environment flag HERMES_PP2_PREFILL_BF16=1 selects worker-2's
experimental exact BF16 weight unpack + cuBLAS GEMM only inside the new API;
all other values retain raw-GGUF GEMV. This does NOT enable or disable
HERMES_PP2_ASYNC_STREAMS, HERMES_NVFP4_GEMV or HERMES_HEAD_BF16.

## Direct state seeding, no replay

The T-token prefill already batches RMSNorm, causal attention, GDN and head,
but its GGUF projections currently perform one GEMV launch per token.
The new API uses the same T-token forward pass to seed these request buffers:

- GDN (48 layers): recurrence writes directly into DecodeState::S[li] on
  its owning GPU, in token order; initial S comes from zeroed decode_init.
- Convolution: seed_conv_tail_k extracts the last THREE pre-convolution QKV
  values into CS[c*3+j], left-padded with zero for T=1/2. This is the
  exact layout that conv_decode_k updates when continuing at token T.
- Full attention (16 layers): copy K after normalization/RoPE and V from
  the same batched pass into the first T positions of Kc/Vc. Set klen=T
  after the layer completes successfully.
- The output remains all FP32 logits [T,VOC]; h_last_out is the last hidden
  before output normalization.

State is per DecodeState and never shared across users. All cache mutating
steps are performed before a successful API return.

## BF16 exact weight route, memory and quality gate

Uses worker-2's frozen APIs: bf16_weight_scratch_bytes,
dequantize_gguf_to_bf16 and gemm_bf16_fp32. BF16 input activations are
produced by the existing f32_to_bf16_k. GGUF raw UE4M3 x E2M1
dequantized weights are exactly representable in BF16; cuBLAS FP32
accumulation can still differ from the original GEMV reduction.

One private BF16 weight scratch area of 8,388,608 bytes (8 MiB)
PER GPU holds tiled portions of each FFN/hidden projection. Worker-2's
ReusableBF16WeightScratch dispatches vectorized exact GGUF unpack and cuBLAS
FP32 GEMM across output-row tiles, avoiding a full-size (170 MiB) BF16 matrix
and a second permanent copy of all model weights. Each slot has its own
cuBLAS handle and completion event; shutdown drains pending work before freeing
the scratch. There is one slot per GPU per prefill request.
VRAM preflight requires 512 MiB free per GPU after allocating the decode
state; BF16 additionally requires 8 MiB scratch + 512 MiB remaining.
All per-layer/frame CUDA allocations are checked and owned by RAII guards.
GPU0->GPU1 pinned host staging is per call, not the global g_pinned buffer.
The existing legacy prefill() still uses its shared g_pinned staging area,
but now holds g_xfer_mu across both blocking boundary copies, consistent
with legacy decode_step(). This final transfer-safety adjustment has passed
offline build/link only; no additional GPU tests were run after release.

BF16 mode deliberately computes a SECOND full prefill with baseline GGUF
GEMV, then compares EVERY logit across all T positions (all finite,
maximum absolute difference <= 1e-3, same argmax for every position).
Any discrepancy, CUDA/cuBLAS error, invalid state, bad token, capacity or
VRAM failure returns NO logits and destroys the partially seeded state.
This runtime gate DOUBLES the work: BF16 opt-in is a quality prototype,
NOT a demonstrated performance improvement. A separate state/continuation
gate is still mandatory; logit equality alone is insufficient.

### Full-model GGUF test results (13 November 2026)

Checkpoint: Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf,
two RTX 5060 Ti 16 GB, CUDA 12.8, Warp4 GGUF GEMV
and BF16 Warp4 output head. The test executes the original T-token
batched prefill, a token-by-token reference decode, then the new
seeded batched prefill followed by one decode token. Each test used
an exclusive GPU window and a 120-second timeout.

| T | RAW seeded prefill vs original logits | GDN S / conv CS / full K,V vs decode | Next-token async logits | Exit |
| --- | --- | --- | --- | --- |
| 2 | max_abs = **0** | All 64 layer states **max_abs=0** | **max_abs=0** | **PASS / 0** |
| 11 | max_abs = **0** | All 64 layer states **max_abs=0** | **max_abs=0** | **PASS / 0** |
| 64 | max_abs = **0** | All 64 layer states **max_abs=0** | **max_abs=0** | **PASS / 0** |

For BF16 projection enabled with HERMES_PP2_PREFILL_BF16=1:

| T | Maximum absolute full-logit error | Relative RMSE | Argmax changes | Gate | Exit |
| --- | ---: | ---: | ---: | --- | --- |
| 2 | 0.0404090881 | 0.0032253171 | 0 | **FAIL** | 1 |
| 11 | 0.0703074932 | 0.00228745488 | 0 | **FAIL** | 1 |
| 64 | 5.14553571 | 0.0520876424 | 1 | **FAIL** | 1 |

BF16 is NOT enabled in the default runtime, and its fail-closed guard
correctly rejects all three tested sizes. Exact BF16 representation of
GGUF weights does not guarantee numerically equivalent full-model
FP32 logits when changing accumulation order. **Do not loosen the
max_abs=1e-3 quality threshold solely to obtain higher throughput.**
At T=64 an argmax was actually different.

After every test, observed GPU memory returned to idle
486 MiB (GPU0) / 1289 MiB (GPU1). There were no CUDA OOM or
residual-VRAM failures. Logs on the Ubuntu server:
/tmp/hermes-worker4-prefill-seed-20261113/run_raw_T2.log,
run_raw_T11.log, run_raw_T64.log and run_bf16_T2_diagnostic.log,
run_bf16_T11_diagnostic.log, run_bf16_T64_diagnostic.log.
Exact scratch binary for the final BF16 diagnostic tests (before the
subsequent legacy-pinned transfer lock):
SHA256 b31aeb8fad2cae6598520815251518e999c7d96cc5737372522a78390c184402;
source pp2_fwd.cu SHA256
8c3585342b233e72512eed081e2702f8584a45b1714dc76f25039c290d6cb4e7.
The RAW seeded tests were run on the immediately preceding scratch
binary, whose only later code change was BF16 diagnostic aggregation.
No live service was modified and GPU ownership was released.

Final source after the last legacy-pinned transfer-lock change:
pp2_fwd.cu SHA256
b22a8f867db5b9c7f35df54384a79b792afcd3e21349d7ac68bdfd733b8b3818,
pp2_fwd.hpp SHA256
af5cf6f5897fbed9d4b92110d514caf3ef1128f135a08d2df0f7ee3c708bd4ce.
Offline nvcc SM120a compilation plus full test executable link succeeded,
binary SHA256
4f347600687333ff2af6a296cc2329bec1b49d68b8afd9c952f75438fc829e9c.
The final linked binary was deliberately NOT executed on GPU after release.

The existing prefill uses cudaDeviceSynchronize per layer. The new seeded
path retains this until a separately validated stream-aware batched path
exists. It can therefore delay other users sharing the GPU despite using
independent request buffers. No multiuser acceleration is claimed.

## Integration/build

Prime owns CMakeLists.txt. To link the new code, add
kernels/src/nvfp4_prefill_exact.cu to hermes_kernels, run
find_package(CUDAToolkit REQUIRED), and link CUDA::cublas transitively
through hermes_kernels or hermes_pp2. The experimental executable
distributed/test_prefill_integration_real.cu should link hermes_pp2,
hermes_kernels, hermes_gguf, Threads::Threads and cuBLAS.
Scratch offline build/link currently PASS at
/tmp/hermes-worker4-prefill-seed-20261113/test_prefill_integration_real.

Run only after exclusive GPU authorization: test T=2,11,64, first with
HERMES_PP2_PREFILL_BF16 unset, then with it equal to 1. The real test
compares whole-model prefill logits to the original prefill, all 48 GDN S
states, 48 CS tails and 16 K/V caches to sequential token-by-token decode,
and one extra token of decode continuation from the seeded async state.
Thresholds are 1e-4 for baseline prefill and 1e-3 for BF16, state,
continuation, always requiring identical argmax. Every failure exits
nonzero. Do not mark unexecuted GPU cases as passed.

## Why chunked prefill beyond T64 is not yet safe

The batched attention has a 64-score shared-memory buffer, the pinned
transfer has PIN_TMAX=64, and RoPE uses chunk-relative positions.
Simply looping 64-token calls would reset or omit history: new chunks
must use absolute RoPE positions, append to existing K/V, attend over all
past tokens causally, and carry forward GDN S and convolution CS. Required
next work: append_chunk(ds, ids, absolute_offset), streamed attention over
past+chunk, bounded scratch and VRAM, followed by golden comparisons for
T=129/1024/8192 and multiuser concurrency. T>64 is rejected now, not
silently split or represented as a measured speedup.
