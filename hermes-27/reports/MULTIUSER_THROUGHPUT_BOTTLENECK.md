# HERMES-27 multiuser throughput bottleneck audit — 2026-11-13

**Scope:** source and pre-existing benchmark evidence only; CPU/OFFLINE audit.
**Code changed:** only this new report. No CUDA inference, profiling, live
service changes, model upload, source edits or new throughput measurements.

**User priority:** sustained aggregate throughput for multiple users first;
prompt prefill second; single-request decode speed third.

## Executive decision

1. **The opt-in `ContinuousScheduler` is host-serial by construction.**
   It owns one `pump_mutex_`, and `tick()` synchronously completes an entire
   `backend_.step()` (including GPU0, CPU handoff and GPU1) before selecting
   the next request. Real N1/N2/N4/N8 HTTP throughput confirms **no N-scaling**
   for its nonseeded variant, despite the actual PP2 backend having per-request
   CUDA streams. This is a demonstrated scheduling constraint, **not** a
   limitation of CUDA streams themselves.
2. **Current RAW seeded prefill is not a multiuser accelerator.** The
   real HTTP A/B shows good N1 improvement but a plateau/slight decline at
   N4/N8. The seeded PP2 path contains actual per-layer
   `cudaDeviceSynchronize()`, repeated `cudaMalloc/cudaFree`, synchronous
   device-to-host/host-to-device detours for full-attention gating, and (for
   the one-shot seeded API) calculates/transfers *all* T vocabulary rows even
   though serving only uses the last one. These are **proven code paths**;
   attributing the measured plateau to any one of them remains a
   **falsifiable hypothesis, not an established causal profile**.
3. **Legacy async threaded HTTP is the better measured N4/N8 configuration
   in the available short-load experiments**. Keep production defaults
   unchanged; do **not** enable `--continuous-scheduler` or
   `--batched-prefill` as a multiuser performance default yet.
4. The actual async PP2 decoder **already has** a private two-GPU stream,
   private pinned staging and persistent decode workspace **per request**.
   Replacing it with another independent scheduler or globally shared scratch
   would risk double-state advancement and regress correctness. Improve the
   existing single state owner / stage handoff instead.

## 1. Grounded numerical evidence (not an independent rerun)

Sources are the repository's pre-existing isolated **real GGUF / real HTTP**
reports, not synthetic CPU scheduler scores. The common short benchmark uses
33 input tokens/request, exactly **2 completion tokens/request**, active=4,
queue=32, context=256, two rounds per N. Aggregate throughput is *completion
tokens divided by round wall time*, including prompt preparation, HTTP,
tokenization and inference. It is **not** GPU decode TPS, prefill tok/s or
an independently established percentile.

| Clients | Nonseeded continuous HTTP | Nonseeded legacy async HTTP | RAW seeded continuous HTTP | RAW seeded legacy async HTTP |
| ---: | ---: | ---: | ---: | ---: |
| N1 | 0.645–0.647 | 0.644–0.647 | 0.940 / 0.960 | 0.945 / 0.954 |
| N2 | ~0.648 | 0.840–0.841 | 0.963 / 0.965 | 0.965 / 0.966 |
| N4 | ~0.647 | 1.107–1.110 | 0.966 / 0.964 | 0.973 / 0.969 |
| N8 | ~0.647 | 1.110–1.111 | 0.962 / 0.962 | 0.974 / 0.979 |

Evidence: `benchmarks/HTTP_SCHEDULER_AB_2026-11-13.md:40-62` and
`benchmarks/HTTP_SEEDED_PREFILL_AB_2026-11-13.md:43-74`.
The real seeded benchmark reports a successful model-backed smoke on both
server branches, SSE/nonstream parity and seeded metrics
`seeded=5 failed=0 prompt_tokens=303`, followed by normal VRAM restoration
(`HTTP_SEEDED_PREFILL_AB:29-41`). Thus this is a **genuinely exercised
seeded path**, not just a recognized flag.

**Observations (limited to this benchmark):**

- Continuous nonseeded throughput remains ~0.647 completion tok/s even at
  N8, with reported p95 request latency growing from ~3.10 s (N1) to
  ~24.71 s (N8); legacy async grows to ~1.11 at N4/N8 and p95 ~14.41 s
  at N8 (`HTTP_SCHEDULER_AB:49-60`).
- Continuous seeded is ~0.96 at N4/N8, still flat; legacy seeded is
  ~0.97 at N4/N8, lower than legacy **nonseeded** ~1.11 there.
- With only two output tokens per request, this experiment is especially
  sensitive to **prefill + queue admission**, not steady decode saturation.
  It would be incorrect to claim `~0.96 tok/s` is the actual PP2
  token-by-token decoder speed.
- A separate **47-prompt-token/16-output-token** real legacy async HTTP
  experiment reported N1 ~2.83, N2 ~3.70 and N4 ~4.83–4.86
  aggregate completion tok/s (`benchmarks/HTTP_CONCURRENCY_REAL_2026-11-13.md:56-72`).
  Its different token mix cannot be A/B-substituted for the 33/2 rows,
  but proves workload mix matters.
- N8 uses **active=4**, so 8 clients do not mean 8 simultaneously resident
  decode states. A flat N4→N8 curve is not by itself evidence of an N8
  GPU scheduling defect.
- Samples are only two short rounds each. No p50/p95 **TTFT/TPOT under a
  sustained mixed workload**, profiler trace or statistically stable
  throughput uncertainty bound is present.

**Reproducibility / provenance limit:** the approved Windows source mirror
does not contain a `.git` directory (`git rev-parse --show-toplevel`
fails there). This audit confirms the benchmark *files and statements* in the
current shared tree, **not** an independently verified Git commit hash or
fresh GPU samples. Benchmark documents record the scratch/test environment.
Real model correctness and the unresolved independent llama.cpp FP32 logit
disagreement remain release blockers; see
`reports/INTEGRATED_RELEASE_QA_2026-11-13.md:91-109`.

## 2. Exact current execution paths and ownership

### A. Opt-in continuous host scheduler — provably serial PP2 calls

- `serving/server.cpp:394-416,636-651`: one server `Scheduler` worker
  constructs one `pp2batch::ContinuousScheduler` and calls
  `continuous_->tick()` in its worker loop. The
  `--continuous-scheduler` flag is optional; no second legacy
  `queue_/active_` decode loop runs for the same request.
- `distributed/pp2_batch.cu:568-680`: `tick()` holds
  `pump_mutex_` over its **entire** invocation, rotates one request at a
  time, calls `backend_.step(...)` **synchronously** at lines 621-630,
  waits for full logits, then picks the next request. Increasing
  `steps_per_tick` only extends the *serial* loop, not GPU overlap.
- `pp2_batch.cu:410-565`: `promote()` allocates states and optionally
  calls `backend_.prefill()` **synchronously before admitting the next
  request**. At most 16 admission attempts/tick; queue expiry and mixed
  FP32 context admission are checked. Cancelling while a CUDA operation runs
  marks state, then completion/cancellation happens after the call returns.
- `pp2_batch.cu:705-789`: `PP2StepBackend` opens **one**
  per-request `DecodeState`; `step` forwards directly to
  `pp2::decode_step`, and the seeded method calls
  `pp2::prefill_into_decode_state` on this **same** state, not another one.
  Closing drains/frees that state's resources.

**Consequence:** the continuous host pump cannot schedule GPU0 of request
B while it is synchronously waiting for GPU1 logits of request A. This is
a direct source-level serialization and matches the nonseeded real A/B
pattern. It does **not** establish how much time is spent on GPU kernels
versus CPU or transfers.

### B. Legacy async threaded HTTP — some overlap is already possible

- `serving/server.cpp:938-1070`: after bounded active/queue admission,
  legacy mode assembles an active round, launches a **host thread per
  request** to `pp2::decode_step` (lines 1027-1043), and joins the whole
  round before sampling/advancing. Thus individual PP2 calls can overlap,
  but each round has a **straggler barrier** and thread creation/join cost.
- `pp2_fwd.cu:1271-1336`: `AsyncDecodeContext` contains
  `cudaStreamNonBlocking` **stream0/stream1**, separate completion
  events, private pinned `host_token`, `host_hidden`,
  `host_logits`, a **request-local** `step_mutex`, and a poison flag.
  The request state and its buffers are released only after stream drain.
- `pp2_fwd.cu:1902-1985`: the async path uses explicit streams for
  embedding, all layers, head and copies. `cudaEventSynchronize(ready0)`
  blocks that **request's host thread** until GPU0's hidden copy completes
  (1944-1948). GPU1 then runs; `cudaEventSynchronize(ready1)` blocks
  that same thread until output logits are ready (1973-1978).
  Other host threads can enqueue GPU0 while that thread waits for GPU1.
  It returns full FP32 logits only after completion/finite validation.
- `pp2_fwd.cu:1339-1419` proves decode layer
  `run_gdn_decode_layer` uses the explicit stream in GEMV and GDN
  calls and skips device-wide layer sync for nonzero stream.
  The full-attention decode layer follows that stream argument too.
- `pp2_fwd.cu:61-64,2018-2021` has **global `g_pinned` and
  `g_xfer_mu`**, but that protected buffer is the **legacy sync
  `decode_step` path**, not `decode_step_async`. Do **not** blame
  async multiuser throughput on a mutex it does not acquire.
- `pp2_fwd.cu:1351-1377`: each decode state's `ds.ws0/ws1`
  is a private persistently allocated workspace, reused across layers
  **for one request**. This is not a global shared mutable scratch buffer.
  Copying it into a global pool without ownership/stream fencing is unsafe.

**Consequence:** stream independence exists in the decode kernel path,
but the host API is still synchronous end-to-end per token. Legacy HTTP
threads exploit some concurrency; the single continuous pump forfeits it.
Neither route has proven stage-level saturation/overlap percentages.

### C. Seeded RAW prefill — concrete cross-request serialization candidates

**Actual prefill is NOT the same as the async decode stream path.**
The `prefill_into_decode_state` call enters `prefill_impl`, whose
layer functions are launched on the **default CUDA stream** rather than
the request-private async streams:

1. `pp2_fwd.cu:485-582,584-680`: each of the 48 GDN and 16
   full-attention prefill layers finishes with
   **`cudaDeviceSynchronize()`** (lines 576 and 674).
   Prefill also synchronizes after GPU0 embedding (849) and GPU1
   output head (920). Device-wide barriers can await unrelated
   concurrent nonblocking-stream work; whether/how much they did in
   the measured N4/N8 run is **unmeasured**.
2. `pp2_fwd.cu:793-811,818-842,485-519,599-609`:
   `PrefillFrameBuffers` and per-layer
   `PrefillLayerScratch` create/free temporary `cudaMalloc`
   allocations repeatedly. These are not the private, persistent
   `ds.ws0/ws1` used by async *decode*. Allocator and implicit
   synchronization overhead are **plausible**, not independently timed.
3. `pp2_fwd.cu:633-652`: each full-attention prefill layer
   copies all `d_qq` to a host `std::vector`, loops on CPU to
   extract the attention gate `G`, then copies `G` **back to GPU**.
   This is a real host-side transfer/sync detour in the layer hot path,
   irrespective of inter-request mutex contention.
4. `pp2_fwd.cu:549-552`: GDN seed loops over each of the T
   input positions and calls `gdn_step_cuda` separately, preserving
   recurrent order but not creating one fused recurrent GPU pass.
   `gemv_batch` likewise iterates over T and launches per-token
   projections (`pp2_fwd.cu:403-424`). Seeded prefill need not have
   `O(1)` GPU launch overhead as T increases.
5. `pp2_fwd.cu:818-935`: seeded
   `prefill_impl(...,private_staging=true)` uses its **own**
   pinned transfer buffer, so it **does not acquire `g_xfer_mu`**
   (869-877). However, the handoff still uses synchronous copies,
   and the routine handles **default-stream** scratch.
6. `pp2_fwd.cu:903-930` supports an internal `last_only`
   flag for the output head, but current one-shot seeded entry calls
   `prefill_impl(...,&ds,true)` with the default
   **`last_only=false`** (`pp2_fwd.cu:1716`).
   It calculates/transfers **T×248320 FP32 logits** and the scheduler
   retains **only the final 248320-row**
   (`pp2_batch.cu:756-781`). For T=33 this is
   32,778,240 bytes of host logits instead of 993,280 bytes for the
   last row, plus 32 head evaluations. A potential last-row-only
   optimization already has a *code shape*, but must preserve the
   required independent full-logit quality gates; this audit did not
   execute or validate it.

**Do not overstate:** no one has established via per-stage timing which of
these costs dominates, nor whether eliminating a barrier would improve
N4/N8 while the GEMV and memory subsystem may already be saturated.

### D. Other CPU / serving limits (candidate, not measured culprit)

- `serving/server.cpp:258-284`: **one** `TokenizerBridge::call()`
  mutex protects synchronous request/response communication with one
  Python helper for encode, `stream_start`, `stream_push`,
  `stream_finish` and reference `decode`.
- `server.cpp:536-635`: continuous scheduler callbacks perform
  tokenizer operations *synchronously in the pump thread*; a slow
  tokenizer or SSE callback delays GPU scheduling. `complete()`
  also performs a full reference decode for output parity
  (`server.cpp:721-750`). This may become material **after** GPU work
  is made concurrent; there is not yet a measured proportion.
- SSE delivery is bounded and cancellation-aware
  (`server.cpp:689-719`); moving work to other threads must not
  bypass limits or block under scheduler state locks.
- Legacy server's thread-per-active-request/round joins
  (`server.cpp:1027-1043`) impose a barrier; raising `--active`
  above the tested 4 is not automatically beneficial and increases
  FP32 KV memory pressure.

## 3. Prioritized, isolated experiments (proposal, NOT executed here)

All proposed GPU experiments need **prime's exclusive grant**, a
scratch binary, same GGUF SHA and fixed input token IDs; neither worker-6
nor this report may change `serving/`, `PP2`, kernels or CMake.
Owners must explicitly coordinate file scopes and choose one request-state
owner. Each experiment is opt-in, reverts cleanly and has a **fail-closed
numerical gate before throughput comparison**.

### P0 — Attribution first: trace the existing unchanged code (highest priority)

**Question:** is the N4/N8 seed plateau dominated by device-wide
synchronization, allocator overhead, default-stream contention, CPU
full-attention gating, the single server pump, memory bandwidth or tokenizer?

**Bounded method:** in a future authorized isolated run, use one
process/binary and matched 33/2 plus 47/16 token fixtures.
Collect CUDA events/Nsight Systems for per-request
`state_open`, seed GPU0 layers, D2H boundary, seed GPU1/head,
`state_close`, decode GPU0, handoff wait, decode GPU1, tokenizer bridge,
`on_event`, HTTP queue and request. Distinguish host elapsed time
from device kernel durations. Capture stream IDs, launch count, allocation
count/bytes, `cudaDeviceSynchronize` call sites and GPU0/GPU1
busy/overlap. Start with N1/N4; only extend to N2/N8 if healthy.
No profiler run in this audit.

**Falsifiable outcomes:** if seed barriers/allocations account for most
wall time while other clients' GPU queues are stalled, target P2; if
GPU0/GPU1 remain busy yet the single pump serializes all calls,
target P1; if tokenizer dominates after scheduling parallelism, target P4.
If memory bandwidth is saturated at N1, do not predict linear scaling.
**Gate:** profiler must preserve identical generated tokens and baseline
VRAM/HTTP health; report undecidable if traces are incomplete.

### P1 — Real cross-request stage overlap in ONE scheduler

**Question:** can the server maintain bounded request-level GPU0/GPU1
overlap without spawning a second scheduler or ever advancing the same
`DecodeState` concurrently?

**Proposal:** opt-in staged async PP2 API owned by the PP2+serving
workers: separate request-local `launch_gpu0(token,pos)`,
`poll_gpu0`, `launch_gpu1`, `poll_logits` with persistent
per-request streams/events/staging; the **single** request owner holds
one outstanding state transition (Idle→Stage0→Stage1→Ready/Cancelled)
and fair ready queues. Scheduler pump performs short nonblocking
transitions rather than synchronous `backend_.step()`. Multiple
requests progress without a per-round global join, but at most one
token is in flight per request. Reuse PP2's existing private buffers;
share only immutable resident weights. No new model copy.

**CPU test first:** deterministic fake event completions in different
orders, N1/2/4/8, asymmetric stage delays, cancellation at every state,
EOS, timeout, admission rejection, stream backpressure, shutdown while
GPU events are incomplete and poisoned state. Assert no duplicate step,
no use-after-free, exact event order, bounded active/queued slots and
single free per state. The existing `pump_mutex_` may still guard
short scheduler transitions, **not CUDA event waits**; use no second
legacy scheduler on these jobs. GPU event completion must precede
releasing `ws0/ws1` or pinned buffers.

**Real gate:** N2/N4 full FP32 vocabulary logits vs original
async `decode_step`, exact positions/GDN/KV, EOS and generated text,
then N1/2/4/8 end-to-end aggregate tok/s plus per-stage overlap and p95
TTFT/TPOT. If no measured gain on N4/N8 with stable quality/VRAM,
disable the feature. True overlap **may** help; no promised factor.

### P2 — Remove seeded prefill's global barriers/host detours safely

**Question:** does preserving the single RAW seed's numerical/state
semantics while isolating its workspace/stream restore N4/N8 scaling?

**Separate A/B changes, one at a time:**

1. Per-request prefill stream(s), explicit CUDA stream propagation
   through **all** GEMV, GDN, head, SDPA and state-seeding kernels,
   and event-scoped completion in place of the per-layer
   `cudaDeviceSynchronize` (including embedding/head). Validate no
   default-stream fallthrough or implicit allocator sync.
2. Private or bounded reusable prefill scratch/stream-ordered allocation;
   avoid repeated synchronous `cudaMalloc/cudaFree` per layer.
   Account projected scratch against **both GPU** admission budgets
   before allocation. Never reuse a poisoned state or buffer before
   the owning completion event.
3. GPU-side extraction of attention gate `G` from `d_qq`, replacing
   the full-layer CPU D2H/extract/H2D path while preserving exact
   indexing and arithmetic. One independent microkernel + full model
   gate before moving the next bottleneck.
4. Experimental `last_only=true` seeded head: compute/read only final
   vocabulary row for serving. Keep all-row reference tests separately;
   reject if final full-vocab logits, GDN/conv/KV or continuation differ.
5. Only if P0 proves launch-bound, investigate bounded recurrent
   kernel fusion/chunked prefill without changing absolute RoPE,
   causal masks or GDN recurrence. BF16 projection previously failed
   its strict whole-model quality gate: **not** a shortcut.

**CPU/offline tests:** source-level audit for accidental stream0 launches,
fake stream/event and allocation lifecycle; exact head index and
full-layer gate extraction reference. **GPU-required gates later:** RAW
T1/11/33/64 seeded vs same-input reference full 248320 logits
(`max_abs=0` on prior established RAW parity when claiming bit-exact),
all recurrent state, next-token logits, mixed T65/128 fallback, multiuser
N2/N4 and cancel during prefill. Record extra scratch VRAM under load
before attempting N8. Failure is terminal; never replay a partly seeded
state. Rebenchmark *both* prefill and completion throughput.

### P3 — Eliminate legacy batch straggler barriers (after P1 feasibility)

**Question:** how much N4/N8 latency/throughput loss results from joining
all active host threads once per token?

**Proposal:** stage/ready-queue ownership from P1 removes the full-round
`join` without introducing another concurrent PP2 state owner. An
isolated CPU model should compare deterministic variable GPU0/GPU1
durations and verify fair completion order, bounded outstanding work,
prioritization of ready Stage1 and no starvation. A real A/B must
separate benefits of removing the join from stream changes; the legacy
threaded route is the control until the candidate passes parity.

### P4 — Tokenizer/SSE and host scheduling only if P0 proves material

Time the single Python tokenizer IPC lock, full final decode, event
callbacks, serialization and HTTP work separately. Consider a bounded
worker queue or local incremental-decoder contract that preserves
per-job token order, Unicode codepoint boundaries, stop-sequence
suppression and final exact-text parity. Do **not** call the tokenizer
simultaneously through a shared pipe without serialization, remove
the final parity check, or let slow SSE readers hold the GPU pump.
Only invest after instrumentation shows >small/nonnumeric overhead.

### P5 — If GPU memory bandwidth is the true limiting resource

Explicit stage overlap may fail to increase throughput when either
GPU is already bandwidth saturated. Next candidates are *verified*
multi-request fused weight reads / GEMV, fewer launch boundaries,
KV precision and eventually MTP if its quality/speed gates pass.
Each requires dedicated kernel/API owners, independent llama.cpp
numerical parity and VRAM admission. No report here claims their
implementation or achievable speedup.

## 4. Non-negotiable comparison and correctness contract

For any later authorized test:

1. Pin exact GGUF/source/binary hashes, CUDA flags, tokenizer, GPU clocks
   and concurrent workload. Compare all candidate modes against
   **legacy async nonseeded N4/N8**, the measured high-throughput control.
   Include both 33/2 and 47/16 loads, mixed 1/11/33/64/65/128/129
   prompts and a context budget scenario; no synthetic-only TPS claims.
2. At least **five** independent complete rounds at each N1/2/4/8
   and an additional bounded steady-load/soak test after initial
   correctness, subject to an explicit GPU grant. Report aggregate
   HTTP completion tok/s, prompt tok/s, actual GPU decode steps/s,
   total wall time, queue delay, p50/p95 TTFT and TPOT, failures and
   cancellation rates. Do not report 2-token EOS workloads as
   pure decode.
3. Snapshot and compare full 248320 FP32 logits, token IDs, state
   `S/CS/K/V`, absolute positions and EOS/stop/usage. Preserve the
   strict original RAW seeded bit-exact gate where already required;
   no tolerance relaxation to obtain throughput. The independent
   llama.cpp full-logit NO-GO remains separate and unresolved.
4. Verify one owner for each `DecodeState`: a token cannot be submitted
   again before its GPU completion event; no double prompt seeding,
   stale input buffer, cross-user logits, or host `klen` update before
   stream completion. For a partial failed seed, close and return a
   terminal failure; never replay. Cancel/timeout/error must drain
   streams before free, and no callback may block holding PP2 state
   locks. Admit based on actual mixed-context FP32 KV + prefill scratch
   and live free VRAM; fail closed on OOM.
5. Record GPU0 and GPU1 utilization, stage concurrency, per-request
   stream IDs, allocator stalls and VRAM high-water. Restore the
   original production services untouched and confirm scratch process
   exit and baseline VRAM. Running overlapping scratch GPU servers
   invalidates the A/B: the earlier benchmark reports intermittent
   upload OOM, including one as-yet unexplained isolated startup OOM
   (`HTTP_SCHEDULER_AB:64-86`).
6. **Decision rule:** promote only a genuinely better N4/N8
   *aggregate* completion throughput with stable correctness and
   acceptable p95 latency/VRAM. A lone N1 improvement, CPU mock test,
   synthetic event overlap, per-projection kernel speedup or successful
   CUDA compilation is insufficient.

## Bottom line

**Proven:** the continuous scheduler synchronously serializes PP2 requests;
the real async decoder already has per-request streams/private workspace;
seeded prefill currently includes global layer barriers, repeated
allocations and CPU round trips; real HTTP short-load throughput plateaus
near 0.96–0.98 seeded and the best measured N4/N8 mode is the legacy
nonseeded async host scheduler at ~1.11 completion tok/s.

**Not proven:** which seeded prefill sub-operation is the dominant wall
time, whether stage overlap would improve memory-bound performance,
whether reducing sync/head work would raise real N4/N8 throughput,
or production reliability. P0 is the first gate; P1/P2 are
opt-in engineering hypotheses with falsifiable correctness and
throughput tests, **not speedup claims**.
