# HERMES PP2 chunked prefill: experimental integration gate

## Verified status and API

New runtime: pp2::prefill_chunks_into_decode_state(ids,ds,stats,error,
chunk_tokens=64,h_last_out=nullptr). Strict opt-in environment variable
HERMES_PP2_CHUNKED_PREFILL=1; valid prompt sizes 65..8192 tokens, chunk
sizes 1..64 and an EMPTY decode_init state with capacity >= prompt length.
Returns ONLY the final prompt token logits [248320]. Any error
invalidates/frees the partial DecodeState. The existing default prefill(),
T<=64 seeded prefill and async decode APIs are unchanged.

Example:

    pp2::DecodeState ds{};
    std::string err;
    pp2::Stats stats{};
    // Set HERMES_PP2_CHUNKED_PREFILL=1 first.
    if (!pp2::decode_init(ds,err,130)) { /* fail */ }
    auto last=pp2::prefill_chunks_into_decode_state(ids129,ds,stats,err,64);
    if (!last.empty()) {
       // Next token may be decoded at absolute position 129.
    } else {
       // Fail closed; ds is already freed.
    }
    if(ds.inited) pp2::decode_free(ds);

At prompt T8192, capacity cannot exceed MAX_CONTEXT=8192. This API can
produce last logits and a valid full cache, but decode_step at position
8192 must be rejected: no KV eviction/sliding window exists. The
T8192 continuation requirement is **BLOCKED** until a separate
eviction design is implemented. T129/T1024 can continue with
capacity130/1025 respectively.

## Causality, cache and memory

The first chunk uses the existing tested batched GDN/attention seeding.
Subsequent chunks read GDN recurrent S and convolution CS. The new
conv_silu_chunk_k uses the three previous pre-convolution QKV samples
from CS when local time is negative; append_conv_tail_k updates CS
AFTER convolution and handles T1/T2 final chunks without overwriting
needed history. All 48 GDN S states are persistent.

RoPE in each full attention block uses absolute position pos0+t.
K/V tensors are appended at offset pos0*1024; klen must equal pos0
before append and is advanced to pos0+T on success.
New sdpa_gqa_chunk_k reads all KV positions 0..pos0+t causally
and follows the FP32 reduction and stable softmax structure of the
single-token decode attention kernel. Only a scores array per block
is allocated: (8192+256)*4 = 33792 bytes MAX shared memory.
No dense T*context score matrix and no second permanent weights.

Per-chunk workspace is bounded by T<=64 and the head computes ONE
token logits, avoiding full-context logits buffers. The state contains
complete K/V caches allocated once by decode_init; VRAM headroom
is checked (minimum 512 MiB free/device). BF16 projection flag
HERMES_PP2_PREFILL_BF16=1 is REJECTED, because previous real-GGUF
T64 tests failed strict full-model quality and changed argmax.
The experimental packed4 GEMV abort-on-error path and HERMES_L26
debug are also rejected. The default code/quality gates remain as-is.

Existing layer-wide cudaDeviceSynchronize calls remain: they may
block other requests. Long-context attention remains quadratic;
there is NO measured throughput improvement or latency claim.

## CPU/offline evidence

Run: python3 distributed/test_prefill_chunk_cpu.py

PASS: 91 convolution continuation cases (lengths 1..8192 and
chunk sizes 1,2,3,7,17,32,64); 64 attention/absolute-RoPE boundary
checks; 42 K/V append cases; 20 source/VRAM/bounds checks.
CUDA12.8 sm120a C++20 source and standalone real-GGUF test compiled
and linked in isolated scratch. **No GPU kernels were executed
by this worker, because prime retains both GPUs exclusively.**
Offline results are not semantic full-checkpoint parity.

## Real GGUF numerical gate — prime must grant exclusive GPUs

Use a linked test executable and the exact model checkpoint:

    HERMES_NVFP4_GEMV=warp4 HERMES_HEAD_BF16=warp4 timeout 120s ./test_prefill_chunk_real --n=129 --chunk=64 --oracle=1
    HERMES_NVFP4_GEMV=warp4 HERMES_HEAD_BF16=warp4 timeout 120s ./test_prefill_chunk_real --n=129 --chunk=17 --oracle=1

The T129 test requires ALL last logits to agree with independent
token-by-token decode within max absolute 1e-4, identical argmax,
48 GDN S, 48 CS, 16 K/V caches to agree within 1e-4 and an
additional next-token continuation comparison. Exit0 means PASS;
exit1 means FAIL.

For T1024 or T8192, --oracle=1 explicitly authorizes the expensive
sequential reference. Without it, the test reports UNVERIFIED with
exit code 3, never a false numerical PASS. T8192 cannot test
continuation at token8192 because capacity is exhausted.
Do not run these tests without an independent GPU lease and
realistic timeouts. Check post-test VRAM returns to baseline.

## CMake handoff (prime ownership)

Build distributed/test_prefill_chunk_real.cu as a separate manual
target linked to hermes_pp2, hermes_kernels, hermes_gguf,
Threads::Threads and transitive CUDA::cublas. Do not add an
unattended GPU CTest without exclusive-resource locking and
checkpoint availability. The Python test is CPU-only.
