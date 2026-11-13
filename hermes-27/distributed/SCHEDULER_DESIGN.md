# HERMES-27 PP2 continuous scheduling (worker-6, 2026-11-13)

## Delivered code

`pp2_batch.hpp/.cu` now expose an opt-in **`ContinuousScheduler`** with
`StepBackend` injection for deterministic CPU tests and a real
`PP2StepBackend` that allocates `pp2::DecodeState` with
`pp2::decode_init(capacity)`, invokes `pp2::decode_step(token, state, pos)`,
and releases it with `pp2::decode_free`. The experimental optional
`StepBackend::prefill` seeds that **same** open state using
`pp2::prefill_into_decode_state` for T=1..64 and selects the first
completion token from the last prefill logits row.

This implements **continuous host scheduling** of variable-length requests,
with shared *resident model weights* and strictly independent per-request
GDN recurrent state, convolution tails and full-attention K/V caches.
It is **not a fused GPU batch**, does not change the CUDA kernel layout or
promise better aggregate tokens/s. By default each turn executes exactly
one PP2 token step per selected request. When `enable_batched_prefill=true`,
up to 64 prompt tokens can be processed in one *stateful batched pass* on
the same per-request state; continuation remains incremental. One pump
thread serializes these calls; the underlying PP2 worker may later provide
a separately verified multi-stream API.

The existing `BatchState`, `batch_step[_parallel]` and `admit_check`
interfaces remain unchanged. `serving/server.cpp` now exposes an opt-in
`--continuous-scheduler` path implemented by its owning worker. The new
batched-prefill option requires a **separate explicit server opt-in** wired
by that same worker; the scheduler default stays unchanged. No HTTP
integration or real-GPU speedup is established by CPU mock tests.

## API and scheduling policy

Create an adapter and scheduler **after** the model upload succeeds:

```cpp
pp2batch::PP2StepBackend backend;
pp2batch::SchedulerConfig cfg;
cfg.max_active = 4;              // hard range 1..16
cfg.max_queued = 32;             // hard range 1..32
cfg.steps_per_tick = 4;          // total decode_step calls per pump invocation
cfg.max_retained = 64;           // terminal snapshots; set by consumer capacity
cfg.eos_token_id = pp2::eos_token_id();
cfg.enable_batched_prefill = false; // set true ONLY for explicit opt-in
cfg.on_event = [](const pp2batch::SchedulerEvent& event) {
  // EventKind::admitted: PP2 GPU state exists; streaming response may now
  // send HTTP 200 and SSE headers.
  // EventKind::token: append token ID to a bounded, nonblocking output queue.
  // EventKind::terminal: finish HTTP job with generated IDs/status/error;
  // if admitted never occurred, send HTTP 429/503 before SSE begins.
  // Consumer must synchronize its own Job state.
};
pp2batch::ContinuousScheduler scheduler(cfg, backend);
auto accepted = scheduler.enqueue({"request-1", prompt_ids, 64});
// The serving event loop calls tick() repeatedly while handling HTTP IO.
scheduler.tick();
pp2batch::JobSnapshot result;
if (scheduler.inspect("request-1", result)) { /* consume and retire result */ }
```

- Enqueue validates the request ID length (1..128 bytes), nonempty prompt, token range,
  `max_new_tokens>=1`, and `prompt_len+max_new_tokens-1<=max_context`.
  The last sampled output token requires no further decode step.
- With `enable_batched_prefill=false` (default), no backend prefill call
  happens; the previous token-by-token prompt path is unchanged. Opt-in
  prefill executes **inside `tick()->promote()`** after allocating one
  fresh DecodeState and **before `EventKind::admitted`**. It is not a
  separate CUDA state and must **never** be called again by serving.
  `StepBackend::prefill(state,prompt,last_logits,error)` returns:
  - `PrefillResult::success` (alias `seeded`): the entire prompt was
    consumed and stored in GDN recurrent S, conv CS and full-attention K/V.
    `last_logits` contains exactly one full vocabulary row for the final
    prompt token. Scheduler commits `prompt_consumed=T`, emits `admitted`,
    then samples/emits precisely **one first output token**. If generation
    continues, the next `decode_step` consumes that generated token at
    absolute `position=T`, not 0 and not T-1.
  - `PrefillResult::unsupported`: backend guarantees absolutely no state
    mutation. The scheduler emits `admitted` and uses the original
    token-by-token loop on the **same** allocation, positions 0..T-1.
    Prompt lengths above 64 bypass `prefill` automatically.
  - `PrefillResult::failed`: backend may have poisoned or already
    `decode_free()`'d the state. The scheduler releases the owning object,
    sends one `terminal failed` event **without `admitted`**, and
    **never retries, replays or initializes another state**. Malformed
    logits, thrown exceptions and failed PP2 quality gates also fail closed.
  `StepBackend` supplies an inline default implementation that returns
  `unsupported`, preserving existing mock/legacy subclasses.
- The real `PP2StepBackend::prefill` invokes exactly one
  `pp2::prefill_into_decode_state(prompt,ds,Stats,err)`, extracts the
  last 248320-element FP32 logits row, and uses no extra permanent weights.
  The underlying API restricts T=1..64 and itself destroys/invalidates a
  mutated DecodeState on failure. `close()` remains safe because
  `decode_free` is idempotent when `ds.inited=false`. The separate strict
  `HERMES_PP2_PREFILL_BF16` quality gate remains unchanged/fail-closed.
- Once prefill is seeded, even a cancelled request must **never** fall back
  to `decode_step` for its prompt: cancellation discards pending logits
  and frees the state. A callback may cancel on `admitted` (before first
  output event) or on `token` (SSE backpressure), stopping further decode
  work. EOS or `max_new_tokens=1` completes from the prefill output with
  **zero** incremental decode calls.
- Metrics separately expose `prefill_attempts_total`,
  `prefill_seeded_total`, `prefill_unsupported_total`,
  `prefill_failed_total` (backend failure/exception) and
  `prefill_prompt_tokens_total`. `decode_steps_total` and
  `tick()`'s return value still count only actual `decode_step` calls.
  Each tick still bounds total open/prefill attempts to 16; fairness is
  admission FIFO followed by per-request decode round-robin. Waiting
  deadlines are rechecked **between** slow stateful prefill admissions.
- Only the **waiting** deque occupies `max_queued` slots. At admission,
  oldest queued requests are considered first (FIFO). A request that does not
  fit the active VRAM envelope waits for active states to finish, or expires.
  Small requests do not overtake the waiting head, preventing indefinite
  bypass of larger requests. Queued requests have no allocated GPU state.
- The active deque rotates after **one decoder token per turn**. A new request
  can enter when an active request exits, independent of other prompts' lengths.
  `steps_per_tick` bounds the total GPU decode calls of each pump invocation
  to 1..16; up to 16 state-open attempts may occur in that same invocation,
  including failed attempts. This separate admission cap prevents an OOM
  retry callback from causing an infinite allocation loop. No padding is done.
- `cancel(id)` removes queued jobs immediately. For a request currently
  opening GPU state or decoding, it sets a cancellation flag; the pump closes
  its private state after the non-interruptible backend call returns. The
  token returned by that step is discarded if cancellation is seen before
  it is committed. A token already committed while cancellation races may
  still be delivered, always before the terminal event. Neither cancellation nor
  `inspect` holds a lock across a CUDA operation.
- `tick(now)` expires all queued jobs whose monotonic deadline has elapsed,
  including when the active set is full. The deadline is rechecked between
  backend steps in the same tick to account for slow GPU calls. Call `tick()`
  regularly; this class does not spawn a hidden scheduling thread or run
  timers in the background. The injectable `now` parameter permits
  deterministic deadline tests while still advancing by elapsed real time
  during each tick.
- `shutdown()` first sets the stopping flag under the short metadata lock,
  then waits for any in-flight `tick()` to return. New enqueue attempts are
  rejected as soon as the flag is visible (see `metrics().stopping`). Active
  and queued requests are flagged for cancellation while any GPU step is
  still running. It then drains all jobs and frees every allocated state.
  Concurrent `tick()` calls are serialized; the caller must not destroy the
  backend before the scheduler.
- Terminal `JobSnapshot` records are kept up to `max_retained` (0..1024)
  then oldest are evicted. Consumers should collect and `retire(id)` responses
  as they complete. The optional synchronous `on_event` delivers:
  - `EventKind::admitted`: **exactly once**, after successful PP2 state
    allocation and active registration, always before the first token.
    A job that fails to allocate, expires or is cancelled before admission
    emits a terminal event without any admission event. The HTTP adapter
    must wait for `admitted` before sending success/SSE headers.
  - `EventKind::token`: every generated token ID after logits validation.
    Prompt processing produces no token event until its final prompt step.
  - `EventKind::terminal`: **exactly once per accepted request**,
    including queued cancellation, timeout, OOM, backend failure and shutdown,
    without depending on retained history.
  A terminal event carries complete generated IDs, status and error. The
  callback runs outside scheduler locks. It may safely call
  `enqueue/cancel/inspect/retire` (including cancelling the same request
  on token-output backpressure and retiring it in its terminal callback),
  but must not call `tick/shutdown` (pump recursion). Its implementation
  must be thread-safe because queued cancellations can invoke it from a
  different caller thread. It must also be nonblocking and must queue only
  a bounded amount of SSE output; if that queue is full, cancel the request.
  Callbacks never own PP2 GPU state or model logits. Exceptions are captured
  and counted in `callback_errors_total`; consumers must check that metric
  and avoid throwing in critical HTTP delivery paths. The callback object
  and any captured HTTP Job registry must outlive the scheduler.
- Errors from failed allocation, invalid/nonfinite logits, wrong output size,
  backend step exceptions, and invalid contexts terminate the affected job
  and release its state. `SchedulerMetrics` reports queue/active occupancy,
  cumulative accepted/rejected/completed/cancelled/expired/failed and actual
  decode-step invocations.

## FP32 KV admission

For each state with context capacity `L`, each GPU allocates:

```
GDN state+conv:  24 * (48*128*128 + 10240*3) * 4 bytes
FP32 K/V:       8 full layers * 2 (K/V) * 1024 float * 4 * L
                = 65,536 * L bytes/GPU
workspace:      568,064 bytes/GPU
step buffers:   GPU0 = 4 + 5120*4; GPU1 = 2*5120*4 + 248320*4
```

`admit_mixed({L1,L2,...}, detail)` sums these actual per-request capacities.
It uses the same conservative baseline as the existing `admit_check`
(GPU0 9856 MiB, GPU1 11470 MiB, limit 15360 MiB, safety reserve 1024 MiB).
Static GPU1 budget is 2866 MiB. Four 8K states fit the projection; five do
not. The estimate is **not a proof of runtime memory availability**.

`PP2StepBackend::open` also checks live free/used VRAM on both GPUs before
allocating, calls `decode_init` (which performs its own exact CUDA preflight),
and checks live utilization again after initialization. A refused allocation
ends only that request. The adapter does not attempt automatic GPU eviction,
memory overcommit, or allocation retries during another process's growth.

## Verification and limitations

CPU test (isolated scratch only, no model load or GPU use):

```sh
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -O2 \
  -DHERMES_ADMISSION_CPU_ONLY -x c++ \
  pp2_batch.cu test_pp2_batch.cu -o batch_cpu
./batch_cpu
```

As of 2026-11-13 the same sources passed **11 original FP32 admission
regressions** and **174 scheduler mock assertions, 0 failures**, including
mixed-context admission, queue bounds, fairness, per-job token/position
sequence, EOS, queued/in-flight/opening cancellation, queue expiry under load
and within one slow backend tick,
state-allocation failure, invalid logits, exceptions, bounded history and
shutdown, queued cancellation with zero retained history (map eviction
under lock), a five-request 8K FP32 pressure scenario with four admitted slots
and delayed fifth-state admission, exact admission/token/terminal ordering
including failed-open terminal without admission, callback-driven cancellation
and retirement, shutdown from a concurrent thread during an in-flight decode,
and pathological OOM callbacks that immediately enqueue retries (16 bounded
admission attempts per tick), event delivery with zero retained history
and callback exception isolation. Seeded-prefill tests also cover T=1/11/64
successful one-pass state continuation; T=65 automatic untouched fallback;
the flag-off baseline; unsupported-backend fallback on the same state;
EOS and max-output termination without extra decode; partial prefill failure,
bad-size/nonfinite logits and thrown exceptions with no replay; cancellation
inside the in-flight prefill and from HTTP admission/token callbacks;
expiry of queued requests during a slow prefill; concurrent shutdown during
an in-flight prefill; exact prompt/completion usage, event order, metrics,
and one state allocation/free per request. An additional four-request
mixed-prompt regression (T=1,11,64,65; max output=2 per request) verifies
exactly three seeded passes and 69 incremental decode steps instead of
145 token steps, with all four requests sharing weights but using exactly
four separate allocate/free state pairs, eight token events and four
terminal events.
The host test also passed with `-fsanitize=address,undefined`
and leak detection (earlier 161-assertion suite repeated 12 consecutive
times with no failures). An independent `-fsanitize=thread` build succeeded, but
its runtime is **not a valid race gate** on the Ubuntu test host: a direct
run exits 66 with `ThreadSanitizer: unexpected memory mapping` before the
tests can complete; retry with a non-PIE binary reproduced the same runtime
failure. Concurrency correctness therefore remains supported by deterministic
CPU tests and ASAN, not by a verified TSAN run. The real seeded-prefill
adapter translation unit compiled with CUDA 12.8 `nvcc -c` offline,
without running model kernels.

**Real-GPU seeded prefill correctness and throughput remain UNTESTED for this
new scheduler integration.** This is not evidence of a speedup or whole-model
logit parity. The independent PP2 seeded-prefill path has its own raw/BF16
validation requirements; none are loosened by this adapter.
The prime must authorize exclusive GPU access before measuring it against the
target-only PP2 path with varied prompt lengths, EOS, 1/2/4/8K contexts,
N=1/2/4, cancellation and VRAM leak tracking. Do not infer speedup from
the mock tests or from the CUDA compilation result.

For serving integration, the owner of `serving/server.cpp` must select one
and only one **PP2 state owner per request**:

1. Enqueue the HTTP Job **only** in `ContinuousScheduler` via
   `ScheduledRequest{id,prompt,max_new_tokens}`. Do not additionally put it
   in the existing `serving::Scheduler::active_/queue_` and do not call
   `pp2::decode_step` directly for the same request.
2. One server worker calls `tick()` repeatedly. Its `on_event` callback
   uses the request ID to find the corresponding HTTP Job. On `admitted`,
   set the HTTP Job admitted flag and signal `Job::cv`; the request thread
   may now begin HTTP 200/SSE. On `token`, append a decoded text delta
   into a bounded queue, or call `cancel(id)` on overflow. On `terminal`,
   set final status/usage, signal `Job::cv`, and `retire(id)`. For jobs
   that never emitted `admitted`, the HTTP handler can return an appropriate
   HTTP 429/503 status, avoiding a false 200 stream. The callback must not
   hold scheduler locks (the library does not hold them) and must never
   write directly to a potentially blocking network sink.
3. HTTP cancellation and client disconnect call `cancel(id)`. The server
   must bound its per-request SSE chunk buffers and cancel slow consumers.
   `SubmitCode::queue_full` should become HTTP 429; model allocation
   failures should become HTTP 503. `JobStatus::expired` is a 429 timeout
   unless the HTTP client has already disconnected.
4. Once streaming parity, exact token/usage counts and cancellation are
   validated, remove the old server queue/active worker loop from that
   request path. **Do not run two scheduler loops on the same requests.**
5. To expose the seeded path, the serving owner must add a separate
   `--batched-prefill` explicit opt-in compatible with
   `--continuous-scheduler`, and assign
   `options.enable_batched_prefill = true` only for that mode.
   **Do not** call `pp2::prefill_into_decode_state` separately in serving,
   do not allocate a second `pp2::DecodeState` and do not reset absolute
   decode positions. The adapter owns the complete prefill and continuation
   lifecycle. Preserve the current default SSE and non-stream behavior and
   validate 1/11/64/65-token prompts, failed prefill before SSE headers,
   cancellation mid-prefill, and exact input/output usage. A failure after
   mutated state is terminal-only; there is no fallback on that request.

The existing HTTP server is unchanged by worker-6. This component schedules
token-by-token prompt processing by default and optional T<=64 stateful
prefill only after explicit opt-in. It performs greedy decoding only; it has
no fused decode batching, MTP rollback or SSE transport.
