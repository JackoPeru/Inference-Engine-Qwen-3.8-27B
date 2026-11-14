# HERMES-27 — bounded stream synchronization for chunked prefill

Date: 2026-11-14. **STATUS: CPU NEGATIVE/DEPENDENCY TEST PASS, CUDA 12.8 COMPILE-ONLY PASS, PHYSICAL GPU VALIDATION NOT RUN. PRODUCTION NO-GO.**

User priority: aggregate multiuser throughput > prompt prefill > single-request decode. No speedup or numerical equivalence is claimed.

## Source freeze, ownership and exact edits

| Artifact | SHA256 |
| --- | --- |
| Baseline distributed/pp2_fwd.cu | 87364d1f21062a6dd2c009ca15efe16cde7d92636a35a0c576b08f855edbf71e |
| Final distributed/pp2_fwd.cu | 6faccbda1aaf568deb7767393a53c5ea89a025a97fbc9746a0fc97a61e29469e |
| NEW distributed/tests/test_prefill_chunk_stream_barrier_cpu.py | b84041919d50fbe02ca2d7eb11d2e80bbec7a6f943e3d56a5e8bcd9dbdff4007 |
| Read-only distributed/pp2_fwd.hpp | 97649e4027ca8db47bed7f0323101edffbe3ddf68ea547f2534e9501d7696b2a |

Only the two end-of-layer synchronization sites in run_gdn_layer and run_full_layer changed. Each now selects **pool ? cudaStreamSynchronize(0) : cudaDeviceSynchronize()**, checks the original layer_result against cudaSuccess and returns false on error. The old device-wide barrier remains selected when pool is null (normal prefill, one-shot seeded T1..64, debug/tests). There is no source change to layer math, kernel launches, async decoder, stream construction, CUDA copy, state mutation, error strings, allocation, pooling or server scheduling. The embedded-token and head device synchronizations in prefill_impl are unchanged.

The opt-in pooled chunked route T65..8192 invokes 48 GDN layers and 16 full-attention layers per chunk. Hence each successful chunk now has **64 stream-0 waits** in place of 64 device-wide end-of-layer barriers. T65 (64+1) means 128 such waits; T129 (64+64+1) means 192. The **number** of host waits is unchanged. cudaMalloc/cudaFree, other barriers and synchronous host/device copies can still restrict concurrency; a measured speedup is unknown.

## Source-based dependency and stream proof

- **GPU0 entry:** prefill_impl allocates GPU0 frame/token buffers with allocations.alloc(0,...), synchronously copies the token IDs, launches embed_gather_k on stream 0, and keeps its original cudaDeviceSynchronize before the first GPU0 layer.
- **GDN layer:** the complete RMS, BF16-conversion, GEMV, convolution seed/read/update, norm, token-ordered GDN recurrence, gating and per-position FFN pipeline is enqueued on the caller's stream 0. All kernel launch configurations are implicit default stream, gemv_batch and gdn_step_cuda default to cudaStream_t=0, and the latter forwards the stream to gdn_step_k. The new wait is after the last residual add_k, before reuse of request-local pooled memory or any next layer.
- **Full-attention layer:** Q/K/V GEMV, RoPE, K/V copy to DecodeState caches, CPU-gated Q extraction, causal SDPA, FFN and final residual writes all use stream 0. KV and gating cudaMemcpy calls are synchronous API calls; no cudaMemcpyAsync or alternative stream occurs in these prefill layer bodies. Host seed->klen[li]=pos0+T is committed only after successful stream completion; later chunks read the correct absolute KV history.
- **GPU boundary:** after GPU0's layers, the code explicitly selects GPU0 and synchronously copies the hidden buffer to a private pinned host frame, selects GPU1 and synchronously copies it into GPU1 frame, then runs GPU1 layers. The device-selection calls, copies and both frame owners are unchanged.
- **Head:** GPU1 output norm and output head still execute on default stream and retain the existing output-head device-wide synchronization and logits copy.
- **Unrelated CUDA work:** async decode's stream0/stream1 are created with cudaStreamNonBlocking, per DecodeState, with ready0/ready1 events. Those events are for a distinct decode path; pooled prefill has no cross-stream producers. The per-state step_mutex try-lock disallows overlapping decode with prefill on the *same* state. Independent users own distinct GDN, conv, KV, scratch, pinned buffers and events. It is thus not necessary to wait on unrelated users' nonblocking decode streams to protect this request's buffers.
- **Lock, pool, error and cancellation:** pooled slot allocation verifies the selected GPU and poisons on wrong-device/OOM/free failures. Layers continue to wait for their stream before borrowing the same slot again. A wait failure returns false and invalidates the request. The previously frozen inner scope destroys both GPU pools while the request lock remains held; only afterwards does fail() unlock and decode_free release GDN/KV/async context. Exceptions unwind in the same order. The exact behavior of CUDA-driver failure or forced cancellation during a failed cudaFree cannot be certified from source alone.

Stream 0 means the default stream on the currently selected device in the invoking host thread. All pooled layer producers/consumers are submitted on that same stream. This remains valid under per-thread default-stream or legacy default-stream semantics for the single host thread that owns a request. Any future introduction of a nondefault stream to these layer functions must add a specific CUDA event dependency or restore the device barrier. CUDA runtime global allocator fences and legacy default-stream ordering may still serialize other work.

## Independent CPU evidence

Run with Python 3.13 from repo root: python -B distributed/tests/test_prefill_chunk_stream_barrier_cpu.py.

Observed exit 0:

    PREFILL_CHUNK_SYNC_BARRIER_CPU_PASS source=52 stream_model=21 source_mutants=11 GPU_CUDA_EXECUTION=NOT_TESTED

Source guards inspect *actual* run_gdn_layer and run_full_layer bodies and require the exact pooled conditional barrier, proper after-FFN placement, error checks, stream-0 kernel/GEMV/GDN routes, sync memcpy, state and GPU owner, KV host commit ordering, pool lifetime and clean fallback. An independent CPU queued-stream model compares per-request synthetic output, recurrent state and KV length between stream-local and device-wide synchronization for T65/T129, chunk sizes 1/3/64 and 3 independent request identities (18 positive cases). It keeps unrelated nonblocking decode work pending under stream-only waits. Three negative scenarios detect missing sync, wrong-stream sync and injected synchronization error before scratch reuse. Eleven mutated production-source variants are rejected: wrong stream, lost fallback, dropped error check, scratch detachment, async KV/gating copies, second GDN stream, loss of nonblocking decode stream, loss of mutex and cross-device pool wiring. These tests cannot prove actual CUDA numerical parity.

Additional actual CPU-only regressions:

| Test | Outcome |
| --- | --- |
| distributed/tests/test_prefill_chunk_workspace_reuse_cpu.py | PASS: T65/T129/T1024/N4, 10 fault negatives, 8 mutants, 5 cleanup checks |
| kernels/tests/test_prefill_exact_state_invariants_cpu.py | PASS: state=37 tiled=18 source=18 |
| distributed/test_prefill_chunk_cpu.py | PASS: conv=91, attention/rope=64, KV=42, static=20 |
| distributed/test_prefill_integration_contract.py | PASS: 102 CPU tail scenarios and source contracts |
| distributed/test_async_stream_contract.py | **RED**: pre-existing exact decode launch count expects 29 and sees 30, caused by the already present sliding-KV ring decode launch. This work changes no decode layer launch or async stream. That older test is outside assigned ownership and was left unchanged. |

The CPU harness runs in Python, so native C++ ASAN/UBSAN is not meaningful for this host model. No memory-safety certificate or GPU race test is implied.

## CUDA 12.8 compile-only evidence

On Ubuntu CUDA 12.8.93 / SM120a, source was copied to an isolated directory /tmp/h27-prefill-chunk-barrier-worker14-20261114, then compiled without execution via:

    /usr/local/cuda-12.8/bin/nvcc -std=c++20 -arch=sm_120a --expt-relaxed-constexpr -c /tmp/h27-prefill-chunk-barrier-worker14-20261114/pp2_fwd.cu -I/home/matteo/hermes-27-gh/hermes-27/distributed -I/home/matteo/hermes-27-gh/hermes-27/kernels/include -I/home/matteo/hermes-27-gh/hermes-27/model/include -o /tmp/h27-prefill-chunk-barrier-worker14-20261114/pp2_fwd.o

Actual nvcc exit **0**, warnings only for previously existing unused variables/constants. Input source SHA256 6faccbda1aaf568deb7767393a53c5ea89a025a97fbc9746a0fc97a61e29469e. Ubuntu header SHA256 97649e4027ca8db47bed7f0323101edffbe3ddf68ea547f2534e9501d7696b2a matches Windows. Compiled object SHA256 **195cd29719d01727b8c82027ac3dde505b30dde69e7e9215cfb2675b7ef08f0c**. No CUDA kernel, GPU inference, live service, model, Git or CI action was executed.

## Verdict and remaining gates

**Source-backed narrowed barrier is compile-ready for independent review; production remains NO-GO.** The real-GGUF full-model acceptance gate must compare all logits, per-layer GDN/conv/KV and one-step continuation between original and candidate at T64/65/129 (and long contexts), including cancellation, OOM, poison and simultaneous N1/N2/N4/N8 requests. A real CUDA trace must verify actual stream ownership, p95/TTFT/TPOT and allocation/VRAM behavior. Strict llama.cpp logit equivalence remains an independent release blocker. No physical throughput, latency or numeric improvement has been measured for this candidate.
