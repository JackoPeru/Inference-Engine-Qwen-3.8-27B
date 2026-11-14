# HERMES-27: context-8192 continuation, source contract and bounded-KV proposal

**Date:** 2026-11-14
**Owner:** worker-4, CPU/OFFLINE contract follow-up
**Status:** current end-of-context behavior **confirmed**; proposed ring-window
transition **CPU-model tested, NOT implemented**. No CUDA, checkpoint inference,
HTTP service, kernel or production source was modified.

## Evidence boundary and previous work

This is a *new* 8192-boundary contract, **not** a rerun of the already passed
real GGUF oracle at T129/T1024. The earlier real-checkpoint report,
`benchmarks/CHUNKED_PREFILL_REAL_2026-11-13.md`, records T129 with 64/17
chunks and T1024 with 64 chunks: internal sequential-vs-batched PP2 logits,
GDN/CS/KV and continuation all max_abs=0. That is **same-engine correctness**,
not independent llama.cpp parity. It explicitly left T8192 unverified; T1024
chunked prefill alone took about 106 seconds and was not a speedup claim.

This audit inspected the actual `distributed/pp2_fwd.cu/.hpp`,
`serving/server.cpp` and HTTP readiness/test sources available in the shared
workspace. Its Windows workspace path does not expose Git metadata, so this
audit does **not** independently attest a particular Git HEAD. Prime must
reconcile the frozen source against the published `6d2fe0a` checkout before
using the source contract in release gates.

## Existing behavior: hard stop before any out-of-bounds KV write

The current configuration supports a **maximum of 8192 consumed positions**:

- `DecodeState::MAX_CONTEXT=8192`, `decode_init(capacity)` rejects capacities
  outside `1..8192`. Each of the 16 full-attention layers allocates K and V
  with `capacity*1024*sizeof(float)`, 8 layers per GPU.
- `decode_step(tok,ds,pos)` rejects `pos<0 || pos>=ds.capacity` **before**
  launching async or legacy compute. In a full-attention layer the host
  additionally requires `L=ds.klen[li]`, `0<=L<capacity` and `L==pos`,
  then writes K/V at linear offset `L*1024`, attends to `0..L`, and advances
  `klen=L+1`. There is **no KV eviction or ring mapping**.
- `prefill_chunks_into_decode_state` permits prompt T65..8192, each
  subchunk <=64, validates `T<=capacity` and per-layer K/V length. It returns
  only the final prompt-token logits, not an extra decode step.
- The HTTP legacy scheduler admits a request with
  `capacity=prompt_len+max_new_tokens-1`. It samples the **first completion**
  from the final prompt logits. Only later completion tokens require
  `decode_step` on the previous sampled token at absolute position T.
  This is why a full context can generate **one** token, not zero.

| Prompt tokens | Requested completions | Required cache capacity | Current outcome | Additional decode position |
| ---: | ---: | ---: | --- | --- |
| 8191 | 1 | 8191 | Admitted, first from prompt logits | None |
| 8191 | 2 | 8192 | Admitted | 8191 |
| 8191 | 3 | 8193 | Rejected | None |
| 8192 | 1 | 8192 | Admitted, **terminal** first token | None |
| 8192 | 2 | 8193 | **Rejected** | 8192 would be required |
| 8193 | 1 | 8193 | **Rejected** | None |

Do **not** change the current guards merely to allow position 8192:
the linear `Kc/Vc + L*1024` write would then address beyond the allocated
8192-token buffer. The `sdpa_gqa_decode_k` score loop also assumes dense
linear positions `0..L`. An arbitrary `klen` clamp would be wrong because
`klen==pos` currently encodes the absolute number of processed tokens.

## Proposed opt-in: bounded sliding window of 8192 K/V tokens

**Semantics must be explicit.** Continue at absolute position 8192 by keeping
the **8192 most recent** key/value positions: the token processed at
`pos=8192` attends to absolute positions **1..8192**, not 0..8192.
At `pos=8193`, it attends to **2..8193**. At exactly 8192 prompt tokens,
the first generated token still comes unchanged from the prefill logits.
Only computing the second generated token needs sliding logic.

This changes attention semantics relative to *full* 8193-token history.
It cannot be advertised as numerically identical to unrestricted
full-context attention: the CPU counterexample gives the oldest value a
non-zero contribution under full history but excludes it from the sliding
window. This is an opt-in **bounded-window** policy, not full-context
continuation by increasing a constant.

### Minimal request metadata and GPU layout

Keep the per-request K and V allocation size fixed to
`WINDOW * 1024 * sizeof(float)`, where `WINDOW=8192`, with 16 independent
attention layers. Each request needs, explicitly:

1. `next_absolute_position` (monotonically increasing, not equal to
   resident length once full), `window_start_absolute` and
   `resident_tokens<=8192`; for multi-device state, track/validate the
   committed absolute offset **per full-attention layer**.
2. `slot = absolute_position % WINDOW` for K/V physical storage, with
   `window_start_absolute = max(0, next_absolute_position-WINDOW)`.
   Attention reads keys in **chronological order** from logical positions
   `max(0,pos-WINDOW+1)..pos` via their slot mapping. It must not directly
   iterate contiguous `Kc[p]` beyond the first wrap.
3. Preserve RoPE's original **absolute** position, independently of the
   ring slot: the key at `pos=8192` uses RoPE angle for **8192**, not 0.
   Existing GPU kernels currently cast `long pos` to `int` and then
   to `float`; any future long-running session needs an explicit
   absolute-position precision/overflow policy (e.g. stop well before
   float-integer precision loss rather than silently wrap).
4. Retain all 48 GDN recurrent S tensors and all 48 convolution CS tails
   without resetting or evicting them. They are summaries of history,
   **not** keys/values belonging to a fixed sliding-attention window.

Do not reuse current `klen` for two incompatible meanings. In sliding mode
it can represent `resident_tokens`, while a **separate** absolute counter
provides position validation and RoPE. The normal legacy `klen==pos`
contract must remain intact when the sliding flag is absent.

### Correct ordering and fail-closed behavior

For a new absolute position `p>=WINDOW`, avoid overwriting physical
`p % WINDOW` until the current token's attention has finished reading its
required predecessor set:

1. Reject stale/duplicate/out-of-order `p`, any overlapping step of the
   same request, invalid device state or exceeded absolute-position budget
   **before** CUDA enqueue.
2. Produce the new K/V into **one-token scratch** (the existing decode
   workspace already contains the projected K and V). Compute attention
   over the previous **WINDOW-1** committed keys, followed by staged
   K/V at `p`. Exclude the oldest physical slot from the logical attention
   set; use a bounded reduction/softmax that knows the chronological
   ring mapping.
3. Once the attention kernel has finished reading the old cache and new
   scratch, enqueue K/V commit to slot `p % WINDOW` on the **same
   per-request CUDA stream**. GPU1 and GPU0 stage completion events must
   prevent any premature scratch or cache reuse. Advance host offset
   metadata only after its corresponding commit/completion is known
   successful.
4. If enqueue, event, device, cancellation or numerical checks fail, mark
   the whole `DecodeState` poisoned and drain its work before freeing.
   Earlier GDN layers may already have mutated S/CS; no rollback/replay
   is safe. No success response or reuse of partial state.

One full-attention layer needs at most one transient K and one transient V
vector of 1024 floats each (8 KiB, usually reusable from the current
workspace). If separately allocated across 16 layers, the extra bounded
budget is **128 KiB total / 64 KiB per GPU**, not an unbounded KV copy.
The score scratch at WINDOW=8192 uses
`(8192+256)*4=33,792 bytes/block`, under the normal 48 KiB CUDA
dynamic shared-memory budget. No `WINDOW*WINDOW` matrix is needed.

### Physical memory accounting (excluding model weights/CUDA overhead)

| Per GPU, 8 attention + 24 GDN layers | Bytes | MiB |
| --- | ---: | ---: |
| K/V ring: `8 * 2 * 8192 * 1024 * 4` | 536,870,912 | 512 |
| GDN S: `24 * 48 * 128 * 128 * 4` | 75,497,472 | 72 |
| Convolution CS: `24 * 10240 * 3 * 4` | 2,949,120 | 2.8125 |
| **Total persistent K/V + S + CS** | **615,317,504** | **586.8125** |

These numbers are a source-derived budget, **not measured free VRAM**.
Fixed WINDOW memory is independent of the **absolute** token position
(8192, 8193, 16,385, ...). Growing an exact-history KV allocation by
one position would add 65,536 bytes **per GPU**; growing from 8192 to
16,384 would add another 512 MiB per GPU and would still not solve
the requirement for indefinitely bounded memory.

## HTTP contract, risk and recommended acceptance gates

The current HTTP `--context` is the total prompt-plus-completion budget
(`prompt+max_new-1`). Never silently reinterpret it as the KV window.
Introduce a **separate explicit sliding-mode flag** only if the product
accepts its changed semantics and numerical risk. Admission in that mode
must separately validate (a) initial prompt fit, (b) max generated tokens,
(c) KV **window** budget and (d) bounded absolute position. Keep the
existing legacy, continuous and chunked routes unchanged by default;
enforce one owner/in-flight token per state and fail-closed cancellation.
The existing chunked HTTP path is legacy-only, `--active 1`; enabling
multiuser overlap needs its own stream/provenance tests.

The following are **required future gates, not tests claimed here**:

- Existing no-window baseline at T8191+2 and T8192+1 must stay unchanged
  with the flag off. T8192+2 and T8193+1 must still be rejected in
  default mode; no malformed requests reach the GPU.
- With explicitly enabled sliding: T8192 prompt + 2 completions should
  process generated token at absolute 8192, evict only key 0 in each
  full-attention layer, preserve all GDN S/CS, and use RoPE(8192).
  Verify T8193 output through an **independent sliding-window** golden
  oracle, not merely the old full-history decode. Check later wrap at
  positions 16,384+ and strict chronological causal order.
- Check actual CUDA stream/event sequencing on GPU0 and GPU1, fault
  injection/cancellation after GPU0 commit, pointer bounds, VRAM return
  to baseline, N1/N2/N4 fairness, and request replay rejection.
- At full context, model-level behavior differs from unlimited
  attention by design. Do not claim exact full-model parity, no quality
  regression, TPS gains or production readiness before numerical and
  performance evidence.

Increasing `MAX_CONTEXT` to 8193 would buy **only one** extra step,
violate the fixed cap, and move the failure to 8193; it is not a
bounded continuation design. Offloading all prior KV to CPU or
recomputing unbounded history are alternative research projects,
not a small or validated extension of the current PP2 kernel.

## Executed CPU-only tests

**New file:** `distributed/tests/test_context8192_continuation_contract_cpu.py`.

The script is Python standard library only and reads the **actual local
PP2/HTTP source** to assert current guards and admission equations. It also
uses a separate deterministic ring/chronological-attention model
to exercise:

- Prompt 8191/8192/8193 admissions, max_new=1/2/3 and next-token positions.
- Fail-close at position 8192 in the **existing** linear KV backend.
- First and second physical ring wrap, resident window bounds and
  16 independent full-attention layers.
- Staging without early eviction, cancellation poison and rejected replay.
- Causal chronological attention against a separate deque + stable-softmax
  oracle (toy scalar keys/values), including positions 8191/8192/8193.
- GDN/conv state invariants across attention-only eviction, fixed
  VRAM arithmetic and proof that window-limited attention changes semantics.

Run (CPU only):

```bash
python3 -B distributed/tests/test_context8192_continuation_contract_cpu.py
```

**Final observed result: 15/15 PASS (1.939 s, Python 3, no GPU).**
Execution was performed on a CPU-only temporary copy of the
source tree at `/tmp/h27-worker4-ctx8192-offline/` because the Windows
workstation shell does not have Python on PATH. No CUDA model was loaded
and no service was started, stopped or changed.

SHA256 evidence for the exact local source/test snapshot:

- New CPU test: `0a4630630885eda4c0ff1130af1d4771b687013cc1f6ae10614aae387a439464`
- Read-only PP2 source: `0d11b9a15974e56d6ef3003384127759b36a678b8c9fc79e3dcf277781fd145c`
- Read-only PP2 header: `5b583a12693f455e43a7f5cb0d11184916133629c2a2374e9a01e182ad6502bd`
- Read-only HTTP server: `1f5c2b684a1fda58af8c6dea123ef97005c197cae373dd90a89d85bd573b97ca`

**Decision:** leave existing T8192 terminal-only path unchanged and
fail-closed. The ring-window implementation above requires a distinct,
explicitly authorized production change and independent numerical/GPU
validation. This follow-up **only** adds tests and this report.
