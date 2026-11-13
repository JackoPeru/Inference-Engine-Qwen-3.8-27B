# HERMES-27 PP2 continuous scheduling (worker-6, 2026-11-13)

## Delivered code

`pp2_batch.hpp/.cu` now expose an opt-in **`ContinuousScheduler`** with
`StepBackend` injection for deterministic CPU tests and a real
`PP2StepBackend` that allocates `pp2::DecodeState` with
`pp2::decode_init(capacity)`, invokes `pp2::decode_step(token, state, pos)`,
and releases it with `pp2::decode_free`.

This implements **continuous host scheduling** of variable-length requests,
with shared *resident model weights* and strictly independent per-request
GDN recurrent state, convolution tails and full-attention K/V caches.
It is **not a fused GPU batch**, does not change the CUDA kernel layout or
promise better aggregate tokens/s. Each turn currently executes exactly one
real PP2 token step per selected request. One pump thread serializes steps;
the underlying PP2 worker may later provide a verified multi-stream API.

The existing `BatchState`, `batch_step[_parallel]` and `admit_check`
interfaces remain unchanged. `serving/server.cpp` currently has its **own**
working scheduler; this new component is reusable but **not yet wired into
the HTTP service**. Serving integration must be coordinated with the owning
worker and must not be counted as done based on the CPU tests here.

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
cfg.on_event = [](const pp2batch::SchedulerEvent& event) {
  // Deliver token IDs to a bounded SSE queue, or mark the HTTP Job complete
  // on EventKind::terminal. The consumer must synchronize its own Job state.
};
pp2batch::ContinuousScheduler scheduler(cfg, backend);
auto accepted = scheduler.enqueue({"request-1", prompt_ids, 64});
// The serving event loop calls tick() repeatedly while handling HTTP IO.
scheduler.tick();
pp2batch::JobSnapshot result;
if (scheduler.inspect("request-1", result)) { /* consume and retire result */ }
```

- Enqueue validates the nonempty request ID, nonempty prompt, token range,
  `max_new_tokens>=1`, and `prompt_len+max_new_tokens-1<=max_context`.
  The last sampled output token requires no further decode step.
- Only the **waiting** deque occupies `max_queued` slots. At admission,
  oldest queued requests are considered first (FIFO). A request that does not
  fit the active VRAM envelope waits for active states to finish, or expires.
  Small requests do not overtake the waiting head, preventing indefinite
  bypass of larger requests. Queued requests have no allocated GPU state.
- The active deque rotates after **one decoder token per turn**. A new request
  can enter when an active request exits, independent of other prompts' lengths.
  `steps_per_tick` bounds the work of each pump call; no padding is done.
- `cancel(id)` removes queued jobs immediately. For a request currently
  opening GPU state or decoding, it sets a cancellation flag; the pump closes
  its private state after the non-interruptible backend call returns. The
  token returned by that step is discarded. Neither cancellation nor
  `inspect` holds a lock across a CUDA operation.
- `tick(now)` expires all queued jobs whose monotonic deadline has elapsed,
  including when the active set is full. The deadline is rechecked between
  backend steps in the same tick to account for slow GPU calls. Call `tick()`
  regularly; this class does not spawn a hidden scheduling thread or run
  timers in the background. The injectable `now` parameter permits
  deterministic deadline tests while still advancing by elapsed real time
  during each tick.
- `shutdown()` blocks until any in-flight `tick()` returns, cancels all
  remaining jobs and frees every allocated state. Concurrent `tick()` calls
  are serialized; the caller must not destroy the backend before the scheduler.
- Terminal `JobSnapshot` records are kept up to `max_retained` (0..1024)
  then oldest are evicted. Consumers should collect and `retire(id)` responses
  as they complete. The optional synchronous `on_event` delivers **every
  produced token ID** and **one terminal event per accepted request**,
  including queue cancellation/expiry, without depending on retained history.
  A terminal event carries complete generated IDs, status and error. The
  callback runs outside scheduler locks. It may safely call
  `enqueue/cancel/inspect/retire`, but must not call `tick/shutdown` (pump
  recursion). Its implementation must be thread-safe because queued
  cancellations can invoke it from a different caller thread. Exceptions
  are captured and counted in `callback_errors_total`; consumers must check
  that metric and avoid throwing in critical HTTP delivery paths.
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
regressions** and **78 scheduler mock assertions, 0 failures**, including
mixed-context admission, queue bounds, fairness, per-job token/position
sequence, EOS, queued/in-flight/opening cancellation, queue expiry under load
and within one slow backend tick,
state-allocation failure, invalid logits, exceptions, bounded history and
shutdown, queued cancellation with zero retained history (map eviction
under lock), a five-request 8K FP32 pressure scenario with four admitted slots
and delayed fifth-state admission, exact token/terminal event ordering, event delivery with zero
retained history and callback exception isolation. The host test also passed with `-fsanitize=address,undefined`
and leak detection. The real adapter translation unit compiled with CUDA
12.8 `nvcc -c` offline, without running model kernels.

**GPU correctness and throughput remain UNTESTED for this new adapter.**
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
   uses the request ID to find the corresponding HTTP Job; token events
   append a decoded text delta and terminal events set final status/usage,
   signal `Job::cv`, and allow `retire(id)`. The callback must not hold
   scheduler locks (the library does not hold them).
3. HTTP cancellation and client disconnect call `cancel(id)`. The server
   must bound its per-request SSE chunk buffers and cancel slow consumers.
   `SubmitCode::queue_full` should become HTTP 429; model allocation
   failures should become HTTP 503. `JobStatus::expired` is a 429 timeout
   unless the HTTP client has already disconnected.
4. Once streaming parity, exact token/usage counts and cancellation are
   validated, remove the old server queue/active worker loop from that
   request path. **Do not run two scheduler loops on the same requests.**

The existing HTTP server is unchanged by worker-6. This component schedules token-by-token prompt
processing (no chunked prefill), performs greedy decoding only, and has no
fused GEMM batching, GPU events/streams, MTP rollback or SSE transport.
