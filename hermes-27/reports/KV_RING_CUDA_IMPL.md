# PP2 bounded KV-ring CUDA continuation beyond absolute position 8191

**Date:** 2026-11-14
**Ownership:** worker-4, PP2 source and new opt-in tests only
**Status:** implementation complete at source level; **CPU 17/17 PASS**;
CUDA 12.8 SM120a compile **PASS**, isolated symbol link **PASS**.
**GPU inference / GGUF numerical correctness / throughput: NOT TESTED.**
The ring remains strictly **experimental and NO-GO for serving**.

## Scope and pre-existing proof

Modified exclusively:

- `distributed/pp2_fwd.cu`
- `distributed/pp2_fwd.hpp`

New source test: `distributed/tests/test_sliding_kv_optin_contract.py`.
This report is new. No changes to `serving/server.cpp`,
`distributed/pp2_batch.*`, CMake, `kernels/src/attn_step.cu`, previously
published 8192-boundary tests/reports, installed processes, or Git.

Before this change, `decode_step(pos=8192)` rejected an 8192-capacity
DecodeState, correctly preventing an out-of-bounds KV write.
`benchmarks/CHUNKED_PREFILL_REAL_2026-11-13.md` records real-GGUF
same-engine T129/1024 chunked-prefill parity. We did **not** duplicate
these GPU tests and did **not** execute T8192 inference.
`reports/CONTEXT8192_BOUNDARY_NEXT.md` is the frozen CPU boundary
specification; it was read only, not edited.

## Public contract and strict opt-in

There is no new public function. The existing decode API acquires its
option from the environment **once**, when the state is initialized:

```cpp
// Before calling decode_init:
// HERMES_PP2_SLIDING_KV=1
pp2::DecodeState ds{};
std::string err;
if (!pp2::decode_init(ds, err, 8192)) { /* reject */ }

// The 8192-token initial prompt can be seeded by the existing separate
// chunked API, with HERMES_PP2_CHUNKED_PREFILL=1 and RAW GGUF enabled.
auto prompt_logits=pp2::prefill_chunks_into_decode_state(
    ids8192, ds, stats, err, 64);
// First generated token is sampled FROM the last prompt logits row.
auto continuation=pp2::decode_step(first_generated_token, ds, 8192, err);
// This produces logits for the SECOND generated token.
pp2::decode_free(ds);
```

The existing API freezes two flags in `DecodeState`:
`sliding_kv_enabled` and `sliding_kv_poisoned`. The opt-in requires
exactly `HERMES_PP2_SLIDING_KV=1` and `decode_init(capacity=8192)`.
Other capacities are rejected rather than silently reinterpreted.
Values `0`, `yes`, `true`, `01` do not enable the mode. A process
which changes the environment after initialization does not retroactively
change the request's KV policy.

When enabled, `decode_init` automatically allocates the existing
`AsyncDecodeContext` per-request CUDA0/CUDA1 nonblocking streams/events,
private pinned staging, and a single-request `step_mutex` even if the
general `HERMES_PP2_ASYNC_STREAMS` flag was not set. This eliminates
concurrent same-state ring writes and device-global synchronization
in the **new ring path**. Invalid packed4, BF16 activation-prefill
and `HERMES_L26` combinations are rejected before allocation.

Absolute token positions are bounded by
`SLIDING_KV_ABSOLUTE_POSITION_LIMIT=1048576`, exclusive. This is
deliberately well below int32 overflow and float's 2^24 exact integer
representation boundary. The precision of long-range sin/cos angles
and model quality remains a separate GPU/numerical gate; no claim of
high-quality million-token context is made.

**Default stays unchanged:** with the opt-in absent,
`decode_init` retains its original stream behavior, and
`decode_step(pos>=capacity)` still returns `pos OOB` **before**
CUDA. Full-attention klen also retains the original `L<capacity`
check. The new kernel is not reached before wrap.

## CUDA ring algorithm and ordering

The existing 16 full-attention layers keep their
`capacity * 1024 * sizeof(float)` K and V arrays. With capacity
8192 these remain **512 MiB per GPU** (8 layers × K/V).
The recurrent states for the other 48 GDN layers and their
three-tap convolution CS tails are **not evicted or zeroed**.

All full-attention `ds.klen[li]` values continue to count
**absolute committed tokens** (0,1,...,8192,8193,...), rather
than pretending the resident length grows above 8192. The
new full-attention branch runs only when
`ds.sliding_kv_enabled && pos>=ds.capacity`. It rejects an
absolute-position mismatch before new kernels are launched.

For a new token at absolute position `p`, compute:

```text
count = min(p + 1, 8192)
first_abs = p - count + 1
for logical_abs = first_abs .. p:
    if logical_abs == p:
        read staged Knew/Vnew
    else:
        read Kc/Vc[(logical_abs % 8192)]
softmax in chronological order, weighted sum of V
after successful attention:
    physical_slot = p % 8192
    write both Knew and Vnew into that physical slot
    confirm commit, increment klen to p+1
```

At the **first wrap** `p=8192`, attention sees logical
absolute tokens **1..8192**, including the staged current token;
old slot 0 (absolute token 0) is read by neither the ring attention
nor the new current-token read. It remains allocated and unchanged
until attention finishes. Then slot 0 receives the new K/V.
At `p=8193`, logical tokens **2..8193** map to physical slots
2..8191,0,1 in chronological order. At `p=16384`, physical
slot 0 wraps a second time.

The new `sdpa_gqa_ring_decode_k` kernel is private to
`pp2_fwd.cu` and uses the existing 24 query heads, 4 KV heads,
256-dimensional per-head scores, FP32 tree reduction and
stable softmax logic. It launches on the request's GPU stream,
not the CUDA default stream. Its dynamic shared memory is bounded
by `(8192 + 256)*4 = 33,792 bytes` per attention block, below
48 KiB. It never materializes a dense T×context scores matrix.
The new K/V already live in the persistent decode workspace
(`d_K1`, `d_vv`), so no new permanent GPU allocation, whole
KV cache duplicate or additional BF16 weight storage is required.

Both Q and K are normalized/rotated by the **existing** absolute
RoPE kernel with `(int)pos`, not by `pos % 8192`. GDN S/CS
update normally on the same per-request stream regardless of ring
wrap. This is **hybrid recurrent + sliding full-attention**;
it is NOT unrestricted 8193-token full-history attention.

### Fail-closed transaction boundary

Ring attention MUST consume all predecessor K/V while the oldest
physical slot still contains its original data. The code waits
for `cudaStreamSynchronize(request_stream)` **after attention**
and checks the status before enqueuing any K/V overwrite.
Only afterward it enqueues `cudaMemcpyAsync` for K and V to
`p % 8192` **on the same stream**, and synchronizes that stream
again before host `klen` is advanced. If attention fails, no
old ring slot is overwritten by this step. If one of the two
commit copies fails, the request is terminally poisoned.

Preflight checks all 16 full-layer `klen==pos` and K/V pointers,
all 48 GDN S/CS pointers, capacity=8192, token vocabulary and
absolute-position limit **before the first GDN mutation**.
Once any GPU work starts `sliding_kv_poisoned=true`
until GPU1 logits transfer and full finiteness check succeed.
On CUDA/copy/event/cancellation or other step failure, the state
remains poisoned: do **not** replay, decrement klen or manually
restore cache/GDN; `decode_free` must drain the per-request streams
before releasing persistent state. A failed partial K/V commit
may already have modified one physical slot; poisoning prevents
its unsafe reuse. This is fail-closed rather than an impossible
promise of atomic rollback for all 64 layers.

The two stream-local completion checks per full-attention ring
layer introduce **16×2 host waits per token**. They block only
the corresponding request stream, not the entire device; their
latency/throughput cost is **unmeasured**. This conservative
implementation prioritizes correctness, cancellation isolation and
single-request ownership over an unproven speedup.

## CPU contract and scratch compile

New Python standard-library test:
`distributed/tests/test_sliding_kv_optin_contract.py`

- Independently compares physical 8192-slot ring attention order to
  a chronological `collections.deque(maxlen=8192)` oracle through
  positions 8191/8192/8193/8194 and 16383/16384/16385, with
  stable scalar softmax checks.
- Explicitly proves eviction changes semantics relative to
  a full 8193-position attention.
- Verifies 16 independent full-attention layers and non-eviction
  of the 48 GDN S/CS states, a poisoned/cancelled staged step
  never commits, and stale/overlapping steps fail closed. An
  injected partial K-copy/V-copy failure proves the state must
  be poisoned rather than pretending it can restore the evicted
  old entry or safely replay the token.
- Asserts opt-in exact-value/capacity/absolute-RoPE guards,
  default `pos>=capacity` rejection, private request streams,
  score-memory bounds, staged K/V reads and attention-success
  barrier before K/V overwrite.
- Includes **negative source mutants** that deliberately delete
  the pre-eviction wait, remove modulo indexing, weaken default
  bounds or incorrectly wrap RoPE; each mutated source must
  fail the source contract.

Initial test run: 15/16 with **one test-harness exception-type
mismatch** (the negative mutant triggered `ValueError` instead
of `AssertionError`). The harness was corrected without changing
the runtime; a separate partial-K/V failure test was then added.
**Final CPU test run 17/17 PASS**, 6.582 s.

Offline CUDA build from an **isolated scratch source copy**
at `/tmp/h27-worker4-sliding-offline-20261114/`:

- `/usr/local/cuda-12.8/bin/nvcc -std=c++20 -arch=sm_120a
  --expt-relaxed-constexpr -O2 -c distributed/pp2_fwd.cu`:
  **PASS** (only previous unused-variable warnings).
- Offline symbol resolution link with a minimal no-op `main`,
  the newly compiled `pp2_sliding.o` and existing isolated
  GGUF/GDN/prefill-exact scratch libraries with `-lcublas`:
  **PASS**. This is *not* a production CMake build nor a runnable
  real-GGUF quality test; the executable was **not executed**.
- No GPU models were loaded; no native CUDA kernels, real inference,
  SSH service changes, TabbyAPI changes or systemd changes occurred.

## Gates remaining before enabling a real consumer

**No GPU numerical witness exists for this change.** Prime must
clean-rebuild every source depending on `pp2_fwd.hpp` because
`DecodeState` grew by two bool fields. Do **not** use a pre-change
test binary or stale server object to check ring behavior.

Recommended future independent real-GGUF tests, only under an
exclusive GPU grant:

1. Run `decode_init(ds,8192)` with flag off and verify
   `decode_step(pos=8192)` stays fail-closed with zero cache
   mutation, while earlier T129/T1024 paths retain parity.
2. With flag on, seed T8192 through existing RAW chunked prefill;
   verify all 16 K/V caches and 48 GDN/CS states before first wrap.
   Sample first generated token from the prompt's last logits;
   decode it at absolute `pos=8192`. Verify exact expected
   ring slot overwrite, all 16 `klen=8193`, no VRAM growth,
   finiteness, and the logits against a **separate 8192-window
   reference implementation**, not the original full-history oracle.
3. Continue at 8193, 16383, 16384 and check chronological
   attention and absolute RoPE across two physical wraps
   (can use synthetic injected KV to avoid thousands of costly
   full-model autoregressive steps, with separately validated
   checkpoint parity).
4. Exercise N1/N2/N4 request isolation, concurrent attempt on the
   *same state*, cancellation, device fault or injected K/V-copy
   error, single-stage failure on GPU0/GPU1, stream drain,
   post-free VRAM return and manager lease gates. Verify
   no request succeeds after partial write.
5. Benchmark aggregated throughput and real per-token latency
   versus the non-sliding path and other users. Current
   stream-local waits may be expensive; no performance claim.

The existing HTTP serving admission still enforces
`capacity=prompt_len+max_new_tokens-1 <=8192` and
its chunked mode currently disallows general async operation.
**No serving changes were authorized.** Consequently an
HTTP request for T8192 + 2 completions remains rejected and
must not be represented as integrated sliding service support.
Prime/worker3 must separately design an explicit HTTP flag and
updated window-admission contract **after** GPU numerical proof.

## Acceptance decision

**CPU/OFFLINE IMPLEMENTATION PASS, GPU PRODUCTION NO-GO.**
Default max8192 behavior remains fail closed. The new path
has bounded memory and a source-level safe ordering, but it
has not yet been verified numerically on real CUDA hardware or
measured under multiuser load. This is not equivalent to
unlimited full-history attention.

## Frozen source/build SHA256 manifest

The following refer to the exact bytes compiled in isolated scratch:

| Item | SHA256 |
| --- | --- |
| `distributed/pp2_fwd.cu` | `f063341cadfc57dbcdc6314f48e4620017d362b9f49c58c741f8bf88284b8aa9` |
| `distributed/pp2_fwd.hpp` | `97649e4027ca8db47bed7f0323101edffbe3ddf68ea547f2534e9501d7696b2a` |
| `distributed/tests/test_sliding_kv_optin_contract.py` | `f178826bdfb0ac6a1e14204b9463814b313d1c9758b63239c8f4e284c6470da0` |
| Scratch `pp2_sliding.o` (CUDA 12.8, SM120a) | `cea72b2d2a119e61def95060a195c720b3770fdf8847bb34c3a82037891f90e2` |
| Scratch `pp2_sliding_offline_link` (symbol-only no-op executable, **NOT run**) | `2b65c52d17dde9ec132865fcc75e6572f6f858700da651abd13f62d34d547be4` |

The original frozen CPU15/15 boundary witness and all other owner files
were left unchanged. The shared Windows source folder has no `.git`
metadata visible through the workspace shell; the prime integrator must
apply these exact source hashes to the correct Git checkout, clean-rebuild
consumers of the enlarged `DecodeState`, and independently gate any
release candidate.
