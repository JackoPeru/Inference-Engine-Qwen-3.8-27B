# HERMES-27 — strict opt-in ready-completion ContinuousScheduler

**Status:** source implementation + CPU/OFFLINE tests. **No GPU/model/runtime/service execution and no demonstrated CUDA or HTTP throughput gain.** Prime's starting Git HEAD: \`30561c8\`; the Windows mirror is not itself a Git checkout, so this worker cannot attest the branch/commit from that copy. Only worker-owned \`distributed/pp2_batch.cu\`, \`distributed/pp2_batch.hpp\`, a **new** \`distributed/tests/test_ready_completion_optin_scheduler.cpp\`, and this report were edited. No \`serving/server.cpp\`, \`pp2_fwd.cu\`, CMake, ops, existing test witness or existing benchmark script was changed.

## Why this opt-in exists

The published N1/N2/N4/N8 SSE16 (47 prompt / 16 completion tokens, two rounds, active=4) comparison had **no meaningful advantage** for parallel continuous over legacy async:

| N | Continuous parallel completion tok/s | Legacy async completion tok/s | Parallel delta |
| ---: | ---: | ---: | ---: |
| 4 | 4.871126 | 4.839380 | +0.656% |
| 8 | 4.856234 | 4.858091 | -0.038% |

Source-level and real CPU worker-6 witnesses proved the previous \`tick_parallel()\` joins **every** host worker in a wave before emitting the first completion event. A ready request B could therefore wait for unrelated slow A despite its backend work having completed. The new option removes **that particular host-side join-before-delivery constraint**. It does **not** fix synchronous tokenizer IPC callbacks or PP2's two per-request CUDA event waits, does not introduce new GPU kernels, and must not be treated as guaranteed faster.

## Strict opt-in API and unchanged defaults

New field in \`distributed/pp2_batch.hpp\`:

~~~cpp
pp2batch::SchedulerConfig options;
options.enable_parallel_steps = true;             // existing opt-in
options.enable_ready_completion_dispatch = true;  // NEW, default false
pp2batch::ContinuousScheduler scheduler(options, backend);
~~~

The constructor **fails closed** with \`std::invalid_argument\` unless **all** are true:

1. \`enable_parallel_steps == true\`;
2. \`StepBackend::supports_parallel_steps() == true\`;
3. \`enable_batched_prefill == false\`.

The real \`PP2StepBackend::supports_parallel_steps()\` was already limited to the **exact** environment value \`HERMES_PP2_ASYNC_STREAMS=1\`. This contract still applies. The existing legacy/synchronous path, sequential \`tick()\` and **existing default wave-join \`tick_parallel()\`** remain unchanged whenever \`enable_ready_completion_dispatch=false\`. The new ready branch is selected first only when the new flag is explicitly true.

**Exact serving integration contract for prime or the separate serving owner (NOT implemented here):** introduce, for example, \`--ready-completion\`, and reject **before model upload** unless \`--continuous-scheduler --parallel-steps\` are also selected, \`HERMES_PP2_ASYNC_STREAMS\` is exactly \`1\`, and \`--batched-prefill\`/\`--chunked-prefill\` are disabled. Then set only:

~~~cpp
options.enable_parallel_steps = cfg.parallel_steps;
options.enable_ready_completion_dispatch = cfg.ready_completion;
~~~

Use the existing **single** \`ContinuousScheduler\` and its original \`run_continuous()\` pump; do not create another scheduler, per-request writer or GPU state. **No serving flag currently exists for this new feature**; it is not reachable from the current HTTP CLI until the serving owner explicitly wires it. Do not enable this experimental option by default.

## Implementation and ownership contract

### Fixed-size host worker ready queue

\`tick_ready()\`, called under the original \`pump_mutex_\`, maintains:

- At most \`steps_per_tick <= 16\` total \`std::thread\` launches per tick, with an absolute concurrent bound of \`max_active <= 16\`.
- Fixed-size \`std::array<Work,16>\` and \`std::array<size_t,16>\` completion indices (zero unbounded worker-side allocation for ready queue nodes). A local \`std::mutex\`, \`std::condition_variable\`, published count and consumed count form the ready queue.
- Each worker calls **only** \`backend_.step\` on its own \`DecodeState\`, writes a private logits vector/error, then publishes its **stable array index** under the CV mutex and signals it. Workers never choose output tokens, advance token positions, emit SSE, modify scheduler counters, launch another step or close the CUDA state.
- The sole pump can consume a ready index while other workers are still blocked. It **joins that exact finished worker first**, establishing complete host-worker/logits memory visibility, before clearing \`ready_step_in_flight\`, validating every float/logits length, greedily choosing a token, updating prompt/output counters, emitting events or invoking \`backend_.close\`. There is no detached thread and no possible same-state step overlap in the intended contract.
- A per-\`Job\` \`ready_step_in_flight\` flag, read/written exclusively under the metadata mutex, prohibits double submission/duplicate absolute position while a previous step is outstanding. The pump rotates the original active deque on **launch**, preserving bounded round-robin fairness. A request that finishes early can launch its next token while a different request's earlier step remains in flight, within the same tick quantum.
- No completion tuple is matched on a mutable job ID: a \`Work\` holds its own \`shared_ptr<Job>\` and its \`DecodeState\` pointer for the lifetime of that worker; indices are never reused within the tick. This avoids stale-ID/ABA commits without a global second dispatcher. Normal \`retire()\` may remove the map entry after terminal without invalidating in-flight worker-owned state.
- The existing \`parallel_waves_total\` increments when a new active tranche starts from zero outstanding work; \`parallel_dispatched_steps_total\` and \`decode_steps_total\` count **actually started and reaped** calls, not failed thread constructions. No extra metrics transport or server instrumentation was added.

### Queue deadlines, admission, cancellation and failure

- The local CV waits for a worker-ready signal with a **bounded 25 ms wake interval**, then rechecks active cancellation/stop and queued deadline expiration. A queued request can expire while unrelated CUDA work is still in progress; expiration sends its ordinary terminal event, without allocating a PP2 state.
- To avoid silently introducing CUDA default-stream/global allocator synchronization during an already in-flight async decode, fresh PP2 state \`promote()\`/opening occurs **only when no backend worker step is outstanding**. This is deliberately conservative: a free capacity slot may remain unused until a slow outstanding step returns. Thus this opt-in removes ready **token event** head-of-line blocking and allows already-admitted fast clients to continue; it does **not** yet remove all admission head-of-line blocking.
- A concurrent \`cancel()\` marks an active request promptly. After its own step finishes and **is joined**, the pump discards that request's logits, releases the state safely and emits exactly one cancelled terminal event. No token on a cancelled step. An already-emitted token before later cancellation remains visible as before.
- \`shutdown()\` marks all queued/active work cancelled before waiting on the existing pump mutex. The pump keeps joining every already-started worker; no thread detaches or state closes early. Late enqueues get \`SubmitCode::stopped\`.
- Wrong-sized/nonfinite logits, backend error/exception and invalid output fail **only the affected request**. All logits are validated before an emitted token. No step replay or retry on a possibly mutated recurrent/KV state.
- A \`std::thread\` construction exception fails the unstarted request immediately, with no backend step/replay. Already-started workers are joined and processed; the rest of that tick launches **no new threads**, leaving unaffected active requests eligible on the next tick. A no-throw \`JoinRemaining\` guard joins any remaining started workers on unexpected pump exceptions before releasing \`pump_mutex_\`.
- An **unexpected exception in the pump itself** (e.g. \`std::bad_alloc\` after a worker completes, before its result is committed) uses a separate fail-closed recovery: join **every** started worker first, latch \`stopping_=true\`, clear any remaining per-job \`ready_step_in_flight\` marker, account for reaped/unaccounted steps, then mark all pending active/queued jobs **failed** and close their state only after joins. This prevents stale in-flight markers and an unsafe next-tick replay. If the allocator fails again *during* emergency terminal reporting, the scheduler remains stopped rather than retrying; an owner-driven shutdown is required. Fatal process-level OOM remains outside this scoped guarantee.
- Under **\`HERMES_ADMISSION_CPU_ONLY\` only**, \`SchedulerConfig::ready_before_worker_spawn(started_count)\` injects deterministic partial worker creation failure. It is completely excluded from the CUDA/production configuration. No environment variables, anonymous background threads or testing hooks are added to production execution.
- A second CPU-only hook, \`ready_before_commit(joined_index)\`, injects \`std::bad_alloc\` after the first worker joins while the second is still inside its backend step. The test requires **zero state closes before the blocked worker is released**, all workers joined, every unfinished request failed exactly once, no invalid token, and the scheduler rejecting late enqueue. Separate tests verify that a throwing user \`on_event\` callback is contained by the preexisting \`callback_errors_total\` contract.

### Remaining synchronization limitations

The real async PP2 fwd path already has per-request GPU0/GPU1 nonblocking CUDA streams, per-request pinned buffers and a per-request step mutex. It still blocks the host worker at \`cudaEventSynchronize(ready0)\` and \`cudaEventSynchronize(ready1)\`; this implementation does not change those calls. Its \`async_release\` still drains both request streams before free.

The existing synchronous \`continuous_event\` callback still calls the single serialized Python \`TokenizerBridge\` and may delay handling other ready completions. **No unsafe concurrent tokenizer pipe access was introduced.** \`backend_.close\` may itself invoke device allocation cleanup and cause stalls; no GPU profiler evidence exists to quantify the new path's benefit.

RAW batched/chunked prefill remains excluded: its known per-layer \`cudaDeviceSynchronize\` and host D2H/H2D boundaries have **not** been made independent-state CUDA-stream-safe. The legacy chunked path remains isolated under \`--active 1\`; parallel+seeded must not be promoted.

## RED -> GREEN CPU evidence

The new test \`distributed/tests/test_ready_completion_optin_scheduler.cpp\` was **compiled before implementing the API** and failed as expected: missing \`SchedulerConfig::enable_ready_completion_dispatch\` and the CPU-only failure injection hook. After implementing the path, its original behavioral RED condition becomes GREEN:

    PASS ready=0 B_token_before_A_release=0
    PASS ready=1 B_token_before_A_release=1

The test uses the **actual unchanged backend-independent scheduling mechanisms**, with a fake stateful backend that blocks A using a condition variable. No timing-dependent artificial GPU simulation or mock scheduler replaces \`ContinuousScheduler\`. B finishes while A remains inside its step. Tests guarantee the worker is joined before B's state is freed or reused. The final new test suite covers:

1. Explicit opt-in rejection without parallel flag, concurrency-safe backend or when seeded prefill enabled.
2. Previous parallel mode keeps B's SSE token withheld until A returns.
3. New ready mode emits B's token and terminal while A is held.
4. B emits **three successive** completion tokens with exact 0/1/2 absolute positions before A's first step ends, staying within N2/4-step bounds.
5. N8 FIFO admissions: first four active states then next four, peak threads <=4, one state per request and exact terminal events.
6. Waiting-queue timeout while A remains in-flight, with zero PP2 state allocation for the expired request.
7. EOS after exactly one step/token per request; two simultaneous \`tick()\` callers still serialize on the pump.
8. Cancellation of A while it owns a worker: no token and no early free, with exactly one terminal.
9. One backend failure, exception, nonfinite logits or wrong-sized logits does not poison other requests or leak/double-free states.
10. SSE callback-driven B cancellation and reentrant \`retire()\`, without an extra token or second step.
11. Concurrent \`shutdown()\`: new arrivals rejected, outstanding A joined before its state is released.
12. CPU-only injected failure creating the **second** host thread: first worker completes, second never executes/never replays, both states closed once.
13. Unexpected \`std::bad_alloc\` after one joined worker with an unrelated worker still blocked: **no premature free**, complete join, failed terminals, no replay and scheduler permanently stopped; independent throwing SSE callback stays contained.

**Exact offline compile/run** (on isolated CPU scratch, no GPU model or HTTP server):

~~~bash
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -O1 -g -pthread \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -DHERMES_ADMISSION_CPU_ONLY -Idistributed -x c++ \
  distributed/pp2_batch.cu \
  distributed/tests/test_ready_completion_optin_scheduler.cpp \
  -o /tmp/h27-ready-completion-asan
ASAN_OPTIONS=detect_leaks=1 timeout 30 /tmp/h27-ready-completion-asan
~~~

**Verification as of the report freeze:**

| Gate | Actual outcome |
| --- | --- |
| New ready test, forced B-before-A and old B-after-A | **PASS** |
| New ready suite incl N8, queue deadline, EOS, cancellation, shutdown, injected \`bad_alloc\` | **151 CPU assertions PASS**, final local MSVC 2022 \`/W4 /WX /fsanitize=address\` |
| New MSVC AddressSanitizer repeated before/after final worker exception-message hardening | **15/15 PASS before**, then **5/5 PASS on final source** |
| Earlier Linux GCC \`-Wall -Wextra -Werror -pedantic\`, ASAN+UBSAN leak detection | **PASS for earlier 136-check version before exception recovery**; **final 151-check GCC/UBSAN NOT rerun** because SSH later required additional Tailscale authentication |
| Final optimized local MSVC \`/O2 /W4 /WX\` new suite | **151/151 PASS**; earlier Linux GCC \`-O2\` also PASS for the prior 85-check version |
| Existing \`distributed/test_pp2_batch.cu\`, CPU-only | **267 original scheduler mock assertions + 11 FP32 admission cases PASS**, retested on final source under local MSVC AddressSanitizer |
| Immutable published \`test_throughput_stream_event_scheduler.cpp\` | **4/4 PASS** on final source under local MSVC \`/O2\` (original wave-join behavior remains) |
| Immutable published \`test_throughput_stream_event_source.py\` | **11/11 PASS** |
| CUDA 12.8 \`nvcc -c\` compile only | **PASS before final exception-recovery edit**; **latest CUDA translation unit NOT recompiled** due Tailscale SSH authentication gate; no CUDA execution permitted |
| Real GGUF GPU correctness, async stream overlap or HTTP N4/N8 throughput | **NOT RUN; prime-exclusive GPU grant required** |

Do not count test compilation or CPU concurrency as a real GPU performance gain. The prior SSE16 full-logit independent llama.cpp mismatch remains a separate production NO-GO, unaffected by host scheduling.

## Next gated work by prime / serving owner

1. Wire a separate explicit \`--ready-completion\` startup flag and its strict conditions; do not alter old scheduler opt-ins or expose this implicitly.
2. CPU integration with the existing server fake: SSE Unicode/stop filtering, exact response usage, callback order, disconnect/cancel, backpressure, queued timeout, early close, and simultaneous shutdown. Add server-level measurement of backend-ready-to-SSE latency; do not weaken the tokenizer pipe lock.
3. Only after **explicit prime-exclusive GPU permission** and authenticated owner fencing: isolated real GGUF PP2 and HTTP A/B N1/2/4/8 against both legacy async and current wave-parallel, with 47/16 SSE and 33/2 controls, VRAM high-water, correctness per absolute position, response checksum and p50/p95 **model-token** TTFT/TPOT (not only client-visible TTFC).
4. Reject promotion on any output/usage/GDN/conv/KV/full-logit discrepancy, CUDA error, memory leak, cancellation defect or lack of material aggregate throughput improvement over existing legacy async. No production cutover while independent full-vocabulary llama.cpp fidelity remains FAIL.

**Conclusion:** the requested strict opt-in removes the demonstrable whole-wave join before fast token delivery without compromising the existing default host path. It is a **CPU-validated implementation milestone, not an inference throughput or production acceptance claim**.
