# HERMES-27 — coherent integration gate for request-local chunked-prefill CUDA scratch pool

**14 November 2026. SOURCE / CPU / CUDA 12.8 compile-only PASS; full-model numerical and GPU throughput NOT TESTED. Release NO-GO.**

## Immutable source and change set

The base source was a freshly archived exact Git commit
`95fcd54384c5cb137eeea8549b45738fd9e2e670`.
In a new, isolated Ubuntu scratch under
`/tmp/h27-prefill-pool-coherent-candidate-20261114`, only the
SHA-frozen reviewed changes were overlaid:

| Source | SHA-256 |
| --- | --- |
| Modified `distributed/pp2_fwd.cu` | `87364d1f21062a6dd2c009ca15efe16cde7d92636a35a0c576b08f855edbf71e` |
| New CPU tests `distributed/tests/test_prefill_chunk_workspace_reuse_cpu.py` | `f34878b3a232c850ed8a5eb68508b04ef9da479c50646691c52190ea1bb63ad8` |
| Worker report `reports/PREFILL_CHUNK_ALLOCATION_REUSE_NEXT.md` | `1dd83b7ea3e442fe983ed812a08f7fe13e129325bc4dc0d370cc2f09275919b6` |
| Updated source guard `kernels/tests/test_prefill_exact_state_invariants_cpu.py` | `b991c9dc76d96e28e894d400609f2f129a3d5559779c9f73d8c8940e4ba50540` |

The source guard change replaces the stale old `(int)offset,true);`
literal with the exact new source contract
`(int)offset,true,&gpu0_pool,&gpu1_pool);`.
It does not relax mathematical expectations.

## Functional change and safety review

The actual `prefill_chunks_into_decode_state` path now owns two independent
scratch pools, one per CUDA device, for the lifetime of a **single**
request's bounded T65+ chunk series. Each has 23 typed slots allocated
or grown as needed. Previous per-layer `cudaMalloc/cudaFree` operations
are replaced by request-local slot borrowing. There is no global
cache, new scheduler, shared request state or implicit opt-in.
Existing device-wide synchronization after each of 64 layers per chunk
remains intact.

Independent source review caught a critical **early-return lifetime**
hazard in the initial candidate: `fail()` previously unlocked the request
and freed `DecodeState` before C++ destructors drained the pooled buffers.
The final source corrects this by enclosing the pools in an inner lexical
scope. Incomplete logits, nonfinite logits or KV mismatch now break,
destroy both pools while the request's state lock is still held, and
only then unlock/free the poisoned `DecodeState`. C++ exceptions unwind
the pool before entering the outer failure handler.

Wrong-device, allocator failure and unexpected reuse poison the pool
and reject the request. CUDA device/context failure cleanup has **not**
been physically fault-injected; a host-only test cannot certify it.

## Exact source-scoped verification

The independent Ubuntu scratch tests ran against the **overlaid actual
final source**, not a later modified Git worktree. Observed exit 0:

- New request-local allocation model: T65/T129/T1024 and N4 PASS;
  ten negative fault cases, eight mutated-source guards, five
  cleanup-order checks all PASS.
- Existing stateful source/geometry witness: `state=37 tiled=18 source=18` PASS.
- Existing chunk prefill CPU witness: convolution=91, attention/RoPE=64,
  KV append=42, static=20 PASS.
- Existing prefill integration source contract: 102 CPU tail cases PASS.
- Scratch layer0 tracer source anchors: **6/6 PASS**.
- The independent HERMES/llama capture comparator's new exact-source
  transformation/physical-negative suite: **27/27 PASS** on the revised
  `pp2_fwd.cu`. No real tensor capture was executed.

The worker's allocation-shape model predicts **per-request** CUDA
allocation counts of 2,688→60 for T65, 4,032→60 for T129 and
21,504→60 for T1024, with retained scratch peaking at 49,152,000 bytes
per GPU. These are **source-derived CPU counts**, not CUDA measured
timings or actual VRAM counters. Mandatory device barriers and
host-to-device transfers remain.

## Coherent full-target CUDA 12.8 compilation

CUDA compiler version **12.8.93**; configured with
`HERMES_REQUIRE_SERVER=ON`, production CUDA architecture SM120a,
and `HERMES_REGISTER_GPU_TESTS=OFF`. The immutable base plus source
overlay compiled and CUDA-device-linked **both** actual
`hermes_server` and `hermes_generate` targets successfully (exit 0).
Scratch executable SHA-256:

| Output | SHA-256 |
| --- | --- |
| `hermes_server` | `c0aabae492944804ff00b5e831df99dca69551a8380d8bd4df7456e3e6792208` |
| `hermes_generate` | `e3fe60c9c023aae317081a68d02a85b5bf34c455aec2f0307242156ef01aba07` |

The only NVCC diagnostics were existing-style unused-variable warnings.
This was **not** a GPU inference run, benchmark, model checkpoint upload,
HTTP service launch or cutover. The build files remain private `/tmp`
artifacts and are not committed or deployed.

## Release policy

The new chunked-prefill workspace path remains **strict opt-in** and is
not promoted to a default or concurrent production route. The independent
real-GGUF CUDA numerical comparison at T64/65/129, state continuation,
determinism after allocation faults, N1/N2/N4/N8 fairness and throughput
A/B benchmarks, and physical ingress/GPU ownership attestation remain
**BLOCKED**. The full-vocabulary independent 12/12 gate remains FAIL.
