# PP2 ready-completion: adversarial read-only review

Status: **CPU-only static review**, 14 November 2026. This document is a source-control-free audit of a moving shared Windows workspace. **No GPU, network/SSH, service, production edit or test execution was performed.** Findings below are derived from control flow and explicit adversarial schedules; fault injections are **proposed falsification tests, not observed test failures**.

## Exact read-only source snapshot

| File | Snapshot SHA-256 | Relevant owner |
| --- | --- | --- |
| distributed/pp2_batch.cu | e2eba6d4d86eea56a09dfde6a8db69e89709d9d98ccdb6010face515d7b7b9fc | worker6, actively changing |
| distributed/pp2_batch.hpp | d84efd5f066406d90625a1365fe2653a08d3bfb803e1bb92f326bde53a23ce2e | worker6 |
| distributed/tests/test_ready_completion_optin_scheduler.cpp | a99b93f3eaadb4f111436f439f8e7948fa88eda3951cc2a0bdd88d502bc060fe | worker6 |
| serving/server.cpp | a906e9cbc0eedb55c8e713faa652c71f29188ebf43c9d63354be1d14a5ee174e | prime serving |
| serving/tests/prod_sched_ready_test.py | 036055082f148ce844d1be525cc1d5e2653e7dc43ebc78667888667c6b4f2079 | prime serving |

Earlier in the same review, pp2_batch.cu was observed at 1366 lines with SHA-256 1844a0ca04c5b2e41766a33cad73a35bc1b3b125302d8864c80d3a5f7aead967, later at 1370 lines with the hash above. The **line anchors here refer to the latter 1370-line snapshot**. The explicit ownership requirement forbids touching worker6 sources. Reconfirm anchors and hashes after worker6 freezes the code; a newer patch could invalidate these observations.

## BLOCKER CANDIDATE A — C++ data race in serving tokenizer capability (independently grounded)

**Source anchors**: serving/server.cpp lines 152, 177, 228–232, 271–295, 575–593.

- TokenizerBridge::incremental_supported() const returns the member incremental_ as a plain bool **without locking** (line 152).
- stop_child() writes incremental_=false (line 177); launch_child() writes the newly reported incremental capability (line 231). These operations can occur during call() error recovery, which holds the bridge mutex (lines 271–295).
- Simultaneously the continuous-scheduler pump's admitted callback invokes tokenizer_.incremental_supported() (line 579) on another host thread. The getter is not synchronized with the writer. This is a genuine potential C++ data race if the helper is restarted during admission.

**Executable falsification scenario (CPU-only, for prime)**: Run the existing no-CUDA HTTP mock with a controllable tokenizer helper. Enqueue request A until the scheduler is just about to handle an admitted event; concurrently make a separate HTTP request B cause a helper transport error and restart through call(). Coordinate a barrier at incremental_supported()/stop_child() or use ThreadSanitizer against a dedicated mock. Expected: no data-race report and a consistent streaming capability decision. Current plain-bool getter gives no happens-before edge. A stress test without a controlled barrier may miss this race; that is not evidence of correctness.

**Safe correction**: Make incremental_ atomic<bool> with explicit load/store (relaxed is sufficient for this single capability flag), or make the getter acquire a mutable mutex_ and use the same mutex for modifications. Separately fail closed if the tokenizer's restarted process invalidates an already-open incremental stream. The race is conditional on restart, not a claim that every normal request is corrupted.

**Classification**: **BLOCKER for thread-safety release**, supported by unsynchronized read/write code paths. Not experimentally reproduced here.

## BLOCKER CANDIDATE B — finish() can leak a DecodeState under allocation failure

**Source anchors**: distributed/pp2_batch.cu lines 392–412 and 759–803.

- finish() detaches raw void* state from job->state (lines 400–401) and commits terminal status at line 402.
- Fallible operations **after** detachment include assigning potentially allocating job->error = error (line 403), and deque finished_order_.push_back(job->request.id) (line 408).
- If either throws std::bad_alloc, local raw state has **no RAII destructor**; backend_.close(state) at line 411 is skipped. Job remains closed with state==nullptr. The ready pump's unexpected-exception recovery joins all workers but skips already-closed jobs at line 777, so neither that state nor its missing terminal event is recovered.
- The map/history can become inconsistent if push_back fails after status is terminal: no history item for the closed job, while the state and completion signal have been lost.

**Executable falsification scenario (no model/GPU)**: In a dedicated worker6 CPU-only test, use a deterministic fault injector at terminal-history append (or at job->error assignment with a long error). Prepare a backend that counts open/close and returns valid logits, call tick() with ready flag, inject exactly one std::bad_alloc in finish(). Assert that after tick/shutdown *every* opened backend state was closed and the job has exactly one terminal notification. Current code detaches before the throw and cannot meet the close/notification assertions. A generic ready_before_commit fault (test_ready_completion_optin_scheduler.cpp lines 453–505) is **earlier** than finish() and does not exercise this path.

**Safe correction**: Ensure the state pointer is always protected by an allocation-free/noexcept cleanup guard, and move all potentially throwing metadata preparation before irreversible terminal status and ownership transfer. Do not mark a terminal job closed until either a guaranteed terminal delivery has been arranged or an alternative fail/wakeup channel exists. Preserve join-before-close.

**Classification**: **BLOCKER for exception-safety under memory pressure**. Conditional allocation-failure path; not a claim of an observed leak during normal runs.

## BLOCKER CANDIDATE C — terminal notification can be lost, stranding HTTP waiters

**Source anchors**:

- distributed/pp2_batch.cu lines 365–389, 392–412, 775–800;
- serving/server.cpp lines 622–661, 748–823, 1568–1574, 1526–1564;
- distributed/tests/test_ready_completion_optin_scheduler.cpp lines 430–451.

Two independently reachable mechanisms exist:

1. **Snapshot allocation before callback catch**. emit_event() assembles event.id, terminal event.generated and event.error (pp2_batch.cu lines 368–380) **before** the try around config_.on_event(event) (lines 384–389). If any copy throws, there is no callback, and finish() has already committed the terminal job (line 402) and usually closed its state (line 411). The ready recovery skips a closed job; no second terminal notification is generated.
2. **Swallowed serving callback failure**. emit_event() intentionally catches any exception from the consumer and increments callback_errors_total, but never retries. The serving consumer's terminal branch calls complete(). Construction of JSON response and terminal SSE chunks (server.cpp lines 788–820) may allocate and throw **before** job->done.store(true) (line 821). Unlike some tokenizer operations, these response and queue operations are not wrapped by an independent guaranteed cleanup. Consequently a non-stream handler waiting for done at lines 1568–1569 or an SSE client waiting at lines 1552–1553 may wait without termination, even though the ContinuousScheduler job is closed and retired.

**Executable falsification scenario (CPU-only)**: First, force one allocation failure while emit_event() copies a non-empty terminal generated vector; assert terminal callback count==1 and waiter unblocks. Next, inject one std::bad_alloc in the serving complete() JSON/terminal-queue assembly after the scheduler has emitted terminal and observe bounded HTTP completion. Existing throwing-callback test at worker6 lines 430–451 checks that the scheduler keeps running and increments callback_errors_total, but does **not** assert that a real HTTP request is completed if its terminal callback throws. Prime's serving/tests/prod_sched_ready_test.py lines 64–103 tests normal HTTP/SSE success only, not this failure mode.

**Safe correction**: Give terminal delivery an explicit failsafe: catch snapshot-construction errors, do not silently discard a required terminal event, and make the HTTP completion callback set done/notify_all in a no-throw finalization path even when response serialization fails. Preserve one terminal event per accepted request and ensure no state is freed while backend worker remains in flight.

**Classification**: **BLOCKER for fail-closed HTTP request lifecycle under memory pressure**. Conditional but exact source paths; not experimentally executed.

## Related queued cancellation exception path

**Source anchors**: distributed/pp2_batch.cu lines 310–334. cancel() removes a queued request, changes status to cancelled and increments counters while holding metadata mutex (lines 321–325), then appends to finished_order_ (line 326). A std::bad_alloc during that append leaves status cancelled and the request removed from waiting_ but skips the terminal emit_event() at line 332. In a serving request waiting for admission/terminal, the pending client can be stranded. Unlike the ready-pump exception handler, queued cancellation is invoked on the HTTP canceller's own thread.

**Falsification**: Single queued fake backend request, cancel concurrently with worker A blocking an active slot, inject failure at finished_order_.push_back, assert queued request receives exactly one terminal and no state allocation. Current source does not have an exception-safe terminal fallback. This is the **same terminal-commit exception-safety family** as B/C, not a separate routine queue race.

## Conservative limitation L1 — new admission remains HOL behind any in-flight backend step

**Source anchor**: distributed/pp2_batch.cu lines 813–819. promote() is called only if outstanding==0, because state allocation/init may synchronize CUDA default streams. This is a deliberate conservative safety boundary, not an accidental missed completion-queue wakeup.

**Executable scenario**: In a CPU fake, max_active=2, A blocked in backend.step, B fast finishes and frees a state while A remains blocked. Enqueue C. The ready pump can commit B's token and terminal, and even relaunch an already-admitted fast B in a later step; **it cannot admit C** while A still runs, because outstanding==1. When C exceeds its queue timeout, expire_waiting() may cancel C before admission. Thus ready-completion removes *per-wave commit HOL* but does not remove *new-request admission HOL*.

**Suggested improvement**: First verify PP2 GPU allocations can safely overlap in-flight async work before changing this restriction. Until then document the limitation and do not market this mode as eliminating all queue wait HOL. It is **not a safety blocker**, and worker6's blocked-A/fast-B scheduler test does correctly cover the narrower claim.

## Conservative limitation L2 — serving stop propagates to core scheduler late

**Source anchors**: serving/server.cpp lines 547–560, 663–679; distributed/pp2_batch.cu lines 1245–1259. Scheduler::stop() flags the outer server stopping_ first, but joins the pump worker **before** calling continuous_->shutdown(). During a long tick_ready, the inner scheduler stopping_ remains false and may continue dispatching up to its current steps_per_tick quantum. After the pump returns, shutdown propagates cancellation.

**Executable scenario**: A slow mock step is blocked, send server.stop() on separate thread, inspect core scheduler metrics while slow A remains blocked; the inner stopping flag is not yet set, although the outer server has stopped accepting new requests. Release A; shutdown finishes. The implementation never frees in-flight state before join, so this is a shutdown-latency and delayed-cancellation limitation, **not a demonstrated UAF or deadlock**. If stop needs prompt queue notifications, propagate the shutdown flag to the inner scheduler before joining the outer pump, while respecting the inner pump_mutex_ join contract.

## Conservative limitation L3 — synchronous tokenization in the single pump

**Source anchors**: serving/server.cpp lines 579–620, 748–783; TokenizerBridge::call() lines 271–297, read_line poll lines 248–269. The pump runs the token callback synchronously and may wait for the shared Python tokenizer bridge lock/process. A 15-second read timeout (or a blocking pipe write) can stall delivery of other already-ready completions, notwithstanding independent CUDA step completion. This is an availability/latency bound, not a direct scheduler-state memory race.

**Executable scenario**: CPU fake with A/B decoding quickly while a controllable tokenizer stalls one stream_push for 1 second. Compare timestamp of B backend completion and B HTTP/SSE delivery; ready queue removes GPU wave join HOL but cannot bypass a blocked synchronous tokenizer callback. Consider bounded separate tokenizer/output workers only with explicit lifetime and ordering contracts; do not spawn detached callbacks from this scheduler.

## Audited safety mechanisms: no proven failure in ordinary paths

- **Join-before-close**: tick_ready uses a stack-owned std::array<Work,16>, a CV queue of 16 completion indices, and join for each selected worker before commit, close, or relaunch (pp2_batch.cu lines 701–751 and 892–968). The exception path joins all surviving workers before cleanup (759–763). No ordinary-path UAF was proven from this protocol.
- **Finite queues and workers**: SchedulerConfig validates max_active<=16, max_queued<=32, steps_per_tick<=16 (pp2_batch.cu lines 224–247). ready_indices cannot overflow as each launched worker publishes only once, with at most steps_per_tick=16 workers, barring an unrelated backend bug. Fixed 50 slots in fail_closed recovery cover <=16 active + <=32 queued + an opening request. No bounded-array overwrite was identified.
- **Lock ordering**: pump_mutex_ serializes tick/shutdown; metadata mutex_ is released before backend.step and before invoking config_.on_event (pp2_batch.cu 365–390). Server callbacks acquire server mutex only for lookup (server.cpp 565–569), then release it before calling continuous_->cancel. Server::retire does hold server mutex while calling core retire, but the core never calls server callback while holding core metadata mutex. No *proved* metadata/server mutex inversion deadlock in documented callback use. Calling tick()/shutdown() reentrantly inside event callback is expressly forbidden by SchedulerConfig contract.
- **Active cancellation and expiry**: active cancel only sets cancel_requested, and the ready worker is joined before finish (pp2_batch.cu 310–334 and 971–984); queued expiry is rechecked at 25ms intervals via condition-variable wait and before promote (813–819, 953–962). No regular cancel-vs-in-flight close race proven. This does not cover the above throwing-history exception.
- **Strict opt-in**: the ready flag requires parallel steps/backend capability and rejects seeded prefill (pp2_batch.cu 238–247); production capability checks HERMES_PP2_ASYNC_STREAMS=1 (1314–1320), serving CLI enforces the same and excludes chunked prefill (server.cpp 1354–1383). No unsafe opt-in bypass found on the reviewed paths.
- **Lifecycle**: serving Scheduler declares backend_ before continuous_ (server.cpp 1125–1127), so the core scheduler normally destructs before its referenced backend; stop joins its pump worker before destruction (547–560). No normal destructor UAF identified.
- **Test coverage**: worker6 tests do cover slow/fast in-flight completion, per-state monotonic token position, two-tick exclusivity, cancel-after-join, queue timeout, EOS, exception in backend step, throwing on_event, shutdown and partial thread creation (test_ready_completion_optin_scheduler.cpp approximately 159–581). They do **not** cover the specifically named failpoints B/C or TokenizerBridge restart race. Prime's HTTP test validates correct normal nonstream/SSE parity using uniform FAKE_STEP_MS=12; it is not a slow-A/fast-B or terminal-failure serving test.

## Priority corrections and release decision

1. **P0: Fix TokenizerBridge::incremental_ synchronization** and add a deterministic restart/admitted concurrency test. This is the clearest real C++ data-race path.
2. **P0: Make finish()/queued cancellation exception safe**: no raw DecodeState detachment prior to fallible metadata preparation without a no-throw cleanup owner; no stranded closed state; terminal status/history/event consistency required.
3. **P0: Guarantee HTTP terminal wakeup on callback/snapshot failure**. A swallowed terminal event must not leave nonstream/SSE waiting forever. Add fault-injected fake-serving negative tests.
4. **P1: Document admission HOL and shutdown cancellation delay**; only optimize after proving the PP2 async allocation/preemption contracts.

**Assessment:** These are source-derived, executable hypotheses supported by exact anchors; none has been dynamically falsified in this read-only task. The standard ready-completion per-worker join path has **no demonstrated race/deadlock/use-after-free** from the code inspected. Source owners should address the three P0 exception/race concerns or demonstrate deterministic counterexamples before advertising the opt-in as reliable under adverse conditions. Re-review after the moving worker6 source is frozen.
