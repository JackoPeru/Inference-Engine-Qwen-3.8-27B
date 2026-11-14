# HERMES-27 — bounded request-local allocation reuse for chunked prefill

Date: 2026-11-14. **Status: SOURCE PATCH + CPU NEGATIVE MODEL PASS + CUDA 12.8 COMPILE PASS; existing stale source-anchor test RED; real GGUF/GPU parity and throughput NOT RUN. Production NO-GO.** Priority: aggregate multiuser throughput, then prompt prefill, then single-user decode.

## Ownership and provenance

Only `distributed/pp2_fwd.cu` was modified. New validation file: `distributed/tests/test_prefill_chunk_workspace_reuse_cpu.py`. This report is new. No edits to `pp2_fwd.hpp`, `pp2_batch`, HTTP serving, kernels, CMake, Git, models, CI or installed services.

| Artifact | SHA256 |
| --- | --- |
| **Input** `distributed/pp2_fwd.cu` | `0b8b8bcbcb6d60ee1c49fb240d7e925772fafc96def5c03151fe02b0d50b93ed` |
| **Final patched** `distributed/pp2_fwd.cu` | `87364d1f21062a6dd2c009ca15efe16cde7d92636a35a0c576b08f855edbf71e` |
| **New** `distributed/tests/test_prefill_chunk_workspace_reuse_cpu.py` | `f34878b3a232c850ed8a5eb68508b04ef9da479c50646691c52190ea1bb63ad8` |
| Read-only `distributed/pp2_fwd.hpp` | `97649e4027ca8db47bed7f0323101edffbe3ddf68ea547f2534e9501d7696b2a` |

The Windows source mirror has no `.git`. Remote read-only `/home/matteo/hermes-27-gh/hermes-27/distributed/pp2_fwd.cu` matched the input SHA. Prime reported remote Git HEAD `b5e0f254` at assignment; later read-only check **actually observed** `95fcd54384c5cb137eeea8549b45738fd9e2e670`, with the same input fwd SHA. Git state is advancing independently. The patch is present in the shared Windows source and a **disposable `/tmp` CUDA compile scratch**, with no Git commit/push inferred from this report.

## Existing bottleneck proved from source

`run_gdn_layer` has 23 allocation sites, but seeded prefill borrows `DecodeState::S[li]` and allocates only **22** temporaries. `run_full_layer` has **18** allocations. There are 48 GDN and 16 full-attention layers in 64 total. Thus **48*22 + 16*18 = 1,344 per chunk** `cudaMalloc` calls and corresponding `cudaFree` calls. Each successful layer performs `cudaDeviceSynchronize` once (64 device-wide barriers per chunk). Frame input/output and pinned staging allocations are additional and remain unmodified.

| Prompt tokens, chunk=64 | Chunks | Old per-layer malloc calls | Predicted pooled malloc calls | Explicit per-layer barriers, unchanged |
| ---: | ---: | ---: | ---: | ---: |
| 65 | 2 | 2,688 | 60 | 128 |
| 129 | 3 | 4,032 | 60 | 192 |
| 1,024 | 16 | 21,504 | 60 | 1,024 |

The **60** count is from the source-derived CPU allocation-shape model, including one-time slot growth when switching from GDN to full-attention layout. It is not an actual CUDA runtime allocation trace. The maximum simulated retained buffer bytes per GPU is **49,152,000 B** (46.875 MiB); two GPUs retain ~93.75 MiB total per active chunked request, beyond their pre-existing decode/cache allocations. Four simultaneous request pools model 240 layer allocation calls vs the old 16,128 for T129, but actual multiuser GPU execution is not established. Existing 512 MiB free-memory preflight is unchanged; CUDA allocation failure still aborts the request.

## Patch design and correctness boundaries

The new `PrefillChunkScratchPool` owns up to 23 independent temporary slots on a **single CUDA device**. Each `prefill_chunks_into_decode_state` invocation creates its own GPU0/GPU1 pool pair on its host thread; it never stores scratch in globals, `DecodeState`, or `pp2_batch`. A slot is indexed by allocation call order within each layer and grows only when a subsequent layer needs more bytes. Growing allocates a new block first, then releases the previous one; the pool poisons itself on allocator, free, size, or device mismatch failures. Request error handling already invalidates and frees the partially seeded `DecodeState`. The ordinary one-shot raw prefill and one-shot seeded T<=64 routes still pass null pools and use their original per-layer allocation behavior.

No layer logic, GGUF matmul, attention, convolution, GDN recurrence, FFN, absolute positions, KV lengths, head/logits, output shape, or stream selection changed. `PrefillLayerScratch` continues to own/free its normal standalone allocations; when explicitly borrowing a pool it returns the slot to its request scope instead. The same `run_gdn_layer` and `run_full_layer` still enforce device-wide synchronization and reject failure **before any next layer or next chunk can borrow the same slots**. No attempt was made to eliminate the 64 per-chunk device-wide barriers, because reusing scratch across asynchronous streams without new event ownership is unsafe.

Thread and cancellation checks: `prefill_chunks_into_decode_state` retains the existing `AsyncDecodeContext::step_mutex` `try_lock` for its state and rejects concurrent decode of the same request. No cache is shared between concurrent requests, and the CUDA-device check prevents accidentally borrowing GPU0 storage while executing GPU1. **The first candidate had an error-ordering defect:** calling `return fail()` directly while the two stack pools were still in scope caused `decode_free(ds)` to run before their destructors. Following independent prime review, the final source now places both pools inside an inner C++ lexical scope. Incomplete logits, nonfinite logits and KV mismatch set `complete=false`, break out of the loop, **destroy both pools while the state lock is still held**, then call `fail()` to unlock and free the `DecodeState`. Exceptions unwind both pools before entering the outer catch and calling `fail()`. A successful return likewise destroys the pools before the lock leaves scope. This preserves per-device scratch lifetime across default-stream kernels and avoids releasing state ownership while its scratch is still being cleaned up. On CUDA device/context errors, success of low-level CUDA synchronization and free calls cannot be established without physical fault-injection; the destructor skips freeing on the wrong/inaccessible device, and no runtime recovery claim is made. Other users' CUDA streams still encounter the original `cudaDeviceSynchronize` barriers, so this change alone cannot establish a throughput gain.

## Actual verification

Windows Python 3.13, repository root, `python -B distributed/tests/test_prefill_chunk_workspace_reuse_cpu.py`, exit **0**:

```
PREFILL_CHUNK_WORKSPACE_CPU_PASS cases=T65/T129/T1024/N4 malloc_old=2688,4032,21504 malloc_pool=60,60,60 N4_mall=240/16128 peak_pool_per_gpu=49152000 fault_negative=10 source_mutants=8 cleanup_order=5 GPU_GGUF_PARITY=NOT_TESTED
```

Test derives the actual source allocation shapes for both layer families, simulates interleaved distinct request owners on two devices, compares the original allocation budget to retained peak bytes, checks unique non-overlapping allocations within each layer, and verifies final pool frees. Ten deliberate negative cases cover overlapping layer access before synchronization, wrong device, failed synchronization, poison reuse, injected OOM, undrained teardown, double free and invalid chunk size. Eight mutated source variants must be rejected by source guards, including wrong pool device, same pool on both GPUs, dropping per-layer scratch ownership, bypassing device checks, loss of sync failure gate, loss of per-state lock **and reintroducing early `return fail()` before destructor cleanup**. Five host lifetime witnesses verify the required ordering for incomplete, nonfinite, KV mismatch and exception paths, with a deliberately unsafe unlock-before-destroy negative control. This is **CPU-only source and lifetime evidence**, not CUDA numerical parity or measured latency.

Existing CPU regressions:

| Check | Observed result |
| --- | --- |
| `python distributed/test_prefill_chunk_cpu.py` | **PASS / exit 0**, conv 91, attention/RoPE 64, KV 42, static 20 |
| `python distributed/test_prefill_integration_contract.py` | **PASS / exit 0**, 102 CPU tail cases plus source guards |
| `python kernels/tests/test_prefill_exact_state_invariants_cpu.py` | **RED / exit 1**: stale literal `"(int)offset,true);"` check in its `T65+ chunk offsets` source guard; call now includes `,&gpu0_pool,&gpu1_pool` |

The last RED is a **known test-source-anchor incompatibility**, not observed state/numerical divergence. That existing file is outside this task's ownership. Its owner must update the source guard to check the new call/parameter flow, then rerun it; this report explicitly does not count it as PASS.

CUDA 12.8.93 `sm_120a` **compile-only** successfully completed on Ubuntu via read-only production Git checkout headers and a disposable copy of patched fwd:

```
/usr/local/cuda-12.8/bin/nvcc -std=c++20 -arch=sm_120a \
  --expt-relaxed-constexpr -c \
  /tmp/h27-prefill-chunk-pool-worker14-20261114/pp2_fwd_cleanup.cu \
  -I/home/matteo/hermes-27-gh/hermes-27/distributed \
  -I/home/matteo/hermes-27-gh/hermes-27/kernels/include \
  -I/home/matteo/hermes-27-gh/hermes-27/model/include \
  -o /tmp/h27-prefill-chunk-pool-worker14-20261114/pp2_fwd_cleanup.o
```

Exit **0**, final object SHA256 `24d3166beece276e74dd5eabfd922460a12b14052f71eb4c76e1648b8c831084`, input scratch SHA matches final patched file `87364d1f...`, header SHA matches the Windows header. The earlier candidate without corrected failure ordering remains a separately retained scratch source/object, **not** the final one. Compiler produced only existing-style unused variable warnings (e.g. `old0`, `N`, `d_wc`, `gemv_check`); no errors. Compilation did not launch kernels, allocate GPU memory or modify any live model/service.

## Remaining gates

**Not release-ready.** First resolve the old static-anchor RED through its owner. Under exclusive physical GPU ownership, the real GGUF must prove T64/65/129 full-model outputs, all GDN/CS/KV state, continuation and deterministic failure cleanup against the unmodified code before any promotion. Then run equal-workload N1/N2/N4/N8 A/B with peak per-device VRAM, throughput, TTFT/TPOT, queue/cancel/OOM, p95 and CUDA trace of actual `cudaMalloc/cudaFree` counts. The strict independent llama.cpp numerical gate remains a separate release blocker. No CUDA speedup, physical numerical correctness, memory saving at runtime or multiuser throughput gain is claimed from CPU modeling and compile success alone.
