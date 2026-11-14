# KV-ring CUDA numerical/algorithm edge audit — 2026-11-14

**Scope:** worker-4 exclusively owned `distributed/pp2_fwd.cu/.hpp`,
new `distributed/tests/test_sliding_kv_cuda_source_edge_cpu.py` and this
report. All other product/serving/batch/CMake/kernels/worker files remain
untouched. **No GPU or real checkpoint inference**, live service changes,
or Git commands were performed.

## Published baseline and one narrowly scoped source fix

The first action recorded the pre-edit, previously published source hashes:

- `distributed/pp2_fwd.cu` SHA256
  `f063341cadfc57dbcdc6314f48e4620017d362b9f49c58c741f8bf88284b8aa9`
- `distributed/pp2_fwd.hpp` SHA256
  `97649e4027ca8db47bed7f0323101edffbe3ddf68ea547f2534e9501d7696b2a`

**Demonstrated safety-invariant hole:** `DecodeState` is a *publicly
mutable struct*, including `sliding_kv_enabled` and `async_ctx`.
`decode_init()` correctly selects request-local `cudaStreamNonBlocking`
streams when `HERMES_PP2_SLIDING_KV=1`, but the downstream
`decode_step()` and `run_full_decode_layer()` previously trusted the
boolean without verifying the prerequisite `async_ctx`. A caller that
toggles `ds.sliding_kv_enabled=true` on a state initialized without
the opt-in can enter the legacy default-stream decode branch and
execute ring slot reuse **without its required per-request mutex,
private streams or event guarantees**. It is not ordinary authorized
API usage, but was not fail-closed despite the comment explicitly
claiming the opt-in requires private streams.

**Safe fix** (only production edit): inside
`sliding_kv_step_preflight()`, after the existing poison check
and before any per-layer/GPU mutation:

```cpp
if(!ds.async_ctx){
  err="sliding KV requires initialized private request streams";
  return false;
}
```

The existing callers of this preflight poison sliding state when
validation fails. Normal genuine `decode_init(8192)` under the exact
opt-in always allocates `async_ctx`, so the valid CUDA data path is
unchanged. Default non-sliding decode bypasses this guard entirely,
and its position-capacity rejection still happens before CUDA.
No change to the ABI beyond the previously published struct layout,
head grouping, attention arithmetic, KV commit sequence, or
per-layer synchronization.

## Algorithm and CUDA source findings

### Grouped heads and numerical reduction

The dedicated `sdpa_gqa_ring_decode_k` uses 24 query heads,
4 K/V heads, head dimension 256 and mapping `g=hq/6`. For every
query head and lane, `Q[(hq*256)+tid]`, `Kc/Vc[slot*1024 + g*256+tid]`
and staged `Knew/Vnew[g*256+tid]` are dimensionally consistent.
Each of the 256 threads multiplies one Q/K dimension, reduces the
dot with the same 128→1 tree order as the old decode kernel,
scales by `1/16` (=1/sqrt(256)), then applies a stable
max-subtracted FP32 softmax and weighted V sum in chronological order.
The kernel implements a **two-pass stored-score softmax**, not an
online streaming softmax; it is nevertheless bounded:
`(8192+256)*4 = 33,792 bytes` per block, below 48 KiB.

No source-anchored mismatch in the grouping, staged token inclusion
or reduction order was found. CPU scalar/vector oracles use
`math.fsum` and `math.exp` rather than CUDA's tree reduction/expf,
so they are **algorithm witnesses only**, not GPU bitwise parity.

### First and later wrap

At absolute position 8192 the visible 8192 positions are
`1..8192`. The old cache physical slot 0 still contains key/value
for absolute token 0, but the attention kernel reads **staged**
new K/V for 8192, never old slot 0. It reads other physical slots
`p%8192` for chronological 1..8191. Only after the attention
stream synchronization completes does it copy K and V into
physical slot `8192%8192=0`. At 8193 the visible positions are
2..8193, physical slot 1 is next to be evicted, and at 16384
slot 0 wraps again. The code correctly uses `(int)pos` for both
query/key RoPE, **not** the circular slot number.

The total window remains 8192. Dropping old K/V deliberately
changes semantics relative to unlimited history. The 48 GDN
recurrent S tensors and 48 convolution CS tails are not reset or
evicted; they are separate persistent state. None of the above
proves downstream full-model quality or equivalence with llama.cpp.

### GPU0/GPU1 stream and early-failure ordering

The request-local async decoder selects device 0 before
`ctx->stream0` work on layers 0..31 and waits for
`ready0` after the cross-device hidden D2H. It selects
device 1 before `ctx->stream1` work on layers 32..63, and
waits for `ready1` after the final logits D2H. There are
8 full-attention layers per GPU; each uses only its GPU-local
K/V allocation and request stream. Ordinary GDN S and CS are
updated on that stream.

The ring sequence in `run_full_decode_layer` is:

1. Launch `sdpa_gqa_ring_decode_k` using staged K/V.
2. Check `cudaPeekAtLastError` and synchronize the same request
   stream. If it fails, return **before overwriting the oldest
   physical KV slot**.
3. Enqueue K and V D2D copies to `pos%8192` on that same stream.
   Synchronize the stream and check the result **before**
   incrementing that layer's absolute `klen`.
4. Continue FFN/GDN/head. Complete GPU1 logits event and finite
   checks. Only then clear sliding poison for the next token.

A K-copy-success/V-copy-failure can leave a partially changed
cache. No kernel rollback is attempted: the state stays poisoned,
and `decode_free()` drains streams before freeing. This is
fail-closed, not a claim of atomic restoration. The top-level
decode guard can return an out-of-range error before mutation;
no rollback is needed in that case.

**Performance blocker unchanged:** 16 full-attention layers
× 2 `cudaStreamSynchronize(stream)` calls each = **32
request-stream host synchronizations** per post-wrap token, in
addition to the two-GPU handoff and final event waits. No
device-global sync is introduced by the ring branch, but these
waits likely limit latency. Do not remove either synchronization
without a demonstrated event-ordered kernel/commit design,
request-lifetime guarantees, negative fault tests, and real N2/4/8
throughput evidence. Optimization was intentionally deferred.

**Cancellation limitation:** no cancellation flag is checked
inside an already launched ring attention kernel. A cancellation
between enqueue and stream completion may still consume GPU time,
and a K/V commit already enqueued on the stream may finish.
Only request poison + drain/free can make subsequent reuse safe.
Do not claim immediate GPU preemption or zero physical writes
after a late cancellation.

## CPU negative/math tests

Added new standalone standard-library Python test
`distributed/tests/test_sliding_kv_cuda_source_edge_cpu.py`.
It reads the **actual** PP2 source and guards:

- The exact 24:4 GQA head mapping, physical
  `p%8192` indexing, absolute RoPE, staged new K and V,
  causal 8192-score bound, original default OOB guard.
- Correct order: ring attention launch → request-stream wait →
  K/V commit → second request-stream wait → advance `klen`.
- Correct dual-device 0/1 stream and event ordering; private
  stream invariant in the new preflight guard.
- A separate chronological `deque(maxlen=8192)` reference
  through first/second wraps: **all 8192 positions** are checked
  for correct chronological tags/values, while nonzero-query
  FP64 CPU softmax is checked on the last 16 positions for
  representatives of all four KV groups. All 24 query heads
  are additionally tested against an independent short 16-slot
  wrapped deque with nonzero queries. This is intentionally
  **not** a 8192-position numerical CUDA oracle.
- Negative source mutants for erroneous `hq/5` grouping,
  old-slot V substituted for staged V, removed attention
  completion wait, RoPE replaced by modulo position,
  weakened default OOB check, and absent private-stream check.
  Each mutant must fail.

The CPU reference uses toy head dimension 2 (or 16-slot short
window) to keep execution bounded. This does not change the
actual CUDA dimension of 256 and is **not** a numerical CUDA
test or a real-checkpoint quality gate.

**Final result: 16/16 PASS, 0.145 seconds** on CPU Python 3 in the
isolated scratch. A preliminary RED run was a **test-oracle bug**,
not a PP2 CUDA mismatch: the expected 8192-element deque entries
were accidentally reduced from `(abs_position, values)` to
`values` only. This produced a false mismatch at position 8191
and triggered an expensive `unittest` difflib explanation on
the long lists. The oracle was corrected to preserve its
absolute-position tag, the diagnostic was bounded, and
the final test suite was independently rerun from exact
current source with **all 16 tests green**. No CUDA source
was modified in response to the false mismatch.

## Offline CUDA compilation and release decision

In isolated Ubuntu scratch `/tmp/h27-w4-edge-20261114`,
the modified `pp2_fwd.cu` was compiled using
`/usr/local/cuda-12.8/bin/nvcc`, `-std=c++20`,
`-arch=sm_120a`, `--expt-relaxed-constexpr`, `-O2`,
without running CUDA kernels. **Result: PASS**, only previously
present unused-local compiler warnings. Source SHA at compile
time: `0b8b8bcbcb6d60ee1c49fb240d7e925772fafc96def5c03151fe02b0d50b93ed`.
Header unchanged SHA
`97649e4027ca8db47bed7f0323101edffbe3ddf68ea547f2534e9501d7696b2a`.
Compiled object SHA
`9580d227d325c84f4f5b604924e0807f594367c00a56280e20fe6361dbec4156`.

**Decision: CPU/source-only SAFE FIX / GPU NUMERICS BLOCKED.**
Maintain default max8192 fail-closed behavior and keep
`HERMES_PP2_SLIDING_KV` experimental/off in serving.
Do not claim a ring speedup or real-GGUF numerical PASS.
Prime should clean-rebuild from this SHA, independently assess
source provenance, run exclusive-GPU first/second-wrap
comparison against a true sliding-window oracle, inject
GPU0/GPU1 commit/cancel faults and measure throughput before
any serving promotion.
