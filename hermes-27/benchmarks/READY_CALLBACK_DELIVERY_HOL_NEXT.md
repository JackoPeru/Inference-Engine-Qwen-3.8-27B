# HERMES-27 — Strict opt-in ready host callback isolation, CPU witness

**Evidence:** 2026-11-14. **Disposition:** Implemented narrowly in
`distributed/pp2_batch.cu/.hpp`; **not enabled by default or in the
prime-owned serving configuration**. New deterministic test:
`distributed/tests/test_ready_callback_delivery_hol_cpu.cpp`. No changes
were made to `serving/server.cpp`, `pp2_fwd.cu/.hpp`, CUDA kernels,
CMake, tools, Git, GPU state or services.

## Published source reviewed before changes

The shared Windows source mirror matched both supplied hashes exactly:

| Input | Initial SHA-256 |
| --- | --- |
| `distributed/pp2_batch.cu` | `279C76C0C502D10F540C0F16DD754BBADC4F746692A44D3E4BEDFBB85135FB06` |
| `distributed/pp2_batch.hpp` | `C0A40C0EFB109ACE1A5D1C06ED44582BEC86CC76C3A4824C99D766F80E213E1F` |

Git main `fe95ae70` was provided by the prime but **not** independently
verified as this Windows mirror has no `.git` metadata. Prior CPU fairness
tests already demonstrated a slow synchronous SSE callback could delay a
ready sibling after that sibling's backend result had returned.

## Change and explicit opt-in

`SchedulerConfig::enable_ready_callback_delivery` defaults to **false**.
When false, the existing `emit_event()` invokes `on_event` synchronously
and preserves its prior failure and timing behavior, including wave/ready
defaults. When true, construction rejects invalid combinations unless:

- ready completion and backend parallel-step capability are enabled;
- `max_active >= 2` (up to the existing hard bound of 16);
- `on_event` is supplied **and documented thread-safe**;
- `on_terminal_delivery_failure` is supplied and obeys its borrowed-ID,
  nonblocking, no-allocation/no-throw contract.

The new path (header `pp2_batch.hpp` around lines 145–155 and 328–378,
`pp2_batch.cu` around lines 368–425, 479–640, 759–854,
1306–1337, 1370–1375, 1478–1492, 1585–1596, 1811–1845) is:

1. **Constant-size mailbox:** 64 fixed `DeliverySlot` elements; each
   holds one moved `SchedulerEvent`, a `shared_ptr<Job>` and an ordering
   number. There is no dynamically allocated task queue and no detached
   callback thread. The snapshot itself can allocate terminal token
   vectors, but its existing exception handler remains active. Moving
   snapshots into a free slot and dropping poisoned slots allocate nothing.
2. **Bounded host callbacks:** At construction, start exactly
   `max_active` managed callback threads (2..16). Roll back and join
   already-started threads if a subsequent thread spawn fails. Workers
   execute only `config_.on_event` and observation code; **never**
   `backend_.open/step/close`, free a DecodeState, or acquire the GPU pump
   lock. A fixed 64-slot scan picks the oldest eligible event.
3. **Same-request FIFO:** A `Job` may have at most one callback in flight.
   An older event for the same `shared_ptr<Job>` must return before a
   later event can begin. Distinct active jobs can be delivered
   concurrently by independent callback threads. `admitted`, `token`
   and `terminal` retain their per-request order.
4. **Backpressure acknowledgment:** Each job's atomic
   `callback_unconsumed` counts enqueued and in-flight host events.
   Ready dispatch skips a state until its prior admitted/token callback
   has finished. A successful **final** token is marked
   `ready_pending_final`; the **pump**, not a callback worker, decides its
   completed versus cancelled terminal only after that callback's
   acknowledgment. Thus a slow SSE sink can still call `cancel` on a
   full buffer without extra same-request GPU steps after the callback
   returns. Sibling requests remain eligible for backend work.
5. **Fail-closed overflow/OOM:** If posting an event exceeds the 64-slot
   bound or cannot be prepared, the request's mailbox is poisoned,
   its pending events dropped, its in-flight backend step flagged for
   cancellation, and the borrowed-ID emergency notifier invoked at most
   once. These operations do not invoke user callbacks under either the
   scheduler metadata mutex or the mailbox mutex. No event or token is
   silently omitted while reporting a successful HTTP response; consumers
   must treat emergency notification as a terminal transport failure.
6. **Shutdown and cancellation producers:** The existing single-pump
   `pump_mutex_` still joins all backend workers before closing states.
   Queued cancellation may prepare a terminal record under `mutex_` and
   then emit outside it; a bounded emitter counter fences these in-flight
   producers before shutdown stops the mailbox. Shutdown drains the queue,
   waits for active callbacks and **joins all callback threads**, with a
   serialized idempotent shutdown path. A non-cooperative callback that
   blocks forever can still delay shutdown, as with the synchronous
   path; no unsafe detach/cancel of foreign code is attempted.
7. **Low-idle CPU:** When all active states await a callback and no backend
   worker is outstanding, ready `tick()` uses a bounded 5ms
   condition-variable wait instead of busy-spinning the outer serving loop.
   This is enabled only with the new opt-in.

The original hard 16-step backend quantum, per-state in-flight marker,
joined worker completion queue, PP2/GDN/DecodeState ownership and published
observability counter schema remain unchanged. The opt-in may use more
host threads and increase CPU overhead; no performance advantage is
asserted without real comparative measurements.

## Independent P0 adversarial callback exception review

A prime-supplied independent read-only review found an actual race in the
first candidate: after a token callback threw, the worker could decrement
`callback_unconsumed` before poisoning the request. Concurrent ready
`tick()` could then see `ready_pending_final`, an acknowledged token and
`cancel_requested == false`, and wrongly mark the failed consumer's
request `completed`.

**Fixed before freeze:** On a throwing callback, retain both the per-job
`callback_running` marker and the final unconsumed ack until
`poison_callback_delivery()` has latched `cancel_requested`, dropped
queued events and invoked the no-throw emergency notifier. Only then
decrement the ack and notify the waiting pump; this never holds the mailbox
mutex while acquiring the scheduler metadata mutex. CPU-only hook
`callback_before_failure_recovery` deliberately freezes the worker at
this boundary. An independent concurrent `tick()` must return without
premature completion/close; after releasing the hook, the result must be
cancelled with **one** emergency notification and **zero** successful
completions. This targeted adversarial case is part of the final test.

**Exact source/test anchors:** `distributed/pp2_batch.cu:590–618`
keeps `callback_unconsumed` nonzero through
`poison_callback_delivery(job)` at line 605 and decrements it only at
line 614. `distributed/pp2_batch.cu:1306–1336` is the single-pump
finalization read of that acknowledgment and the cancellation flag.
`distributed/tests/test_ready_callback_delivery_hol_cpu.cpp:473–552`
injects a held callback exception at the CPU-only hook and executes a
second `tick()` against the old false-success window.

### Exact synthetic causal chronology

| Witness | Controlled order (CV latches, not measured elapsed time) | Expected observation |
| --- | --- | --- |
| Baseline, test around line 286 | A token callback enters blocking gate; B backend step returns; A remains blocked | B terminal absent |
| Async opt-in, same test | A token callback enters blocking gate; B backend step returns; B callback worker delivers token and terminal; A is then released | B terminal present *before* A release; both have exact admitted/token/terminal order |
| Throwing callback, test around line 473 | A token callback throws; worker stops at `callback_before_failure_recovery` before poisoning; another pump `tick()` executes; hook is released; cancellation/emergency occurs | At held hook: active, completed=0, backend not closed; after release: cancelled=1, completed=0, emergency=1 |
| Queue saturation, test around line 627 | A/B callback workers both held; 72 queued requests are successively cancelled; 64 mailbox slots fill; at least 8 more fail closed; both workers released; shutdown joins | Every Q receives exactly one ordinary terminal or emergency; no early backend free, no lost wake |

### Explicit fail-closed and rollback conditions

| Boundary | Mandatory behavior |
| --- | --- |
| Opt-in off | Original inline callback path, no callback worker threads |
| Invalid mode, one active slot, missing sink or missing emergency notifier | Reject constructor with `std::invalid_argument` |
| Partial callback worker construction failure | Stop and **join** all previously created threads before constructor rethrows |
| Event snapshot allocation failure | Preserve existing minimal failed-terminal fallback; use borrowed-ID emergency when the ID itself is uncopyable |
| Fixed mailbox full, failed event post, or throwing sink | Per-job poison, drop undelivered events, set cancellation if still active, notify emergency exactly once; never pretend transport success |
| Throwing last-token sink racing pump | Cancellation must be set **before** releasing the final callback acknowledgment; never commit a false completed terminal |
| Shutdown while callback blocked | Stop pump/backend states safely, fence queued cancellation emitters, drain mailbox and join callback workers; waits for cooperative callback release, no detach |

## Executed validation — Windows CPU only

Real `ContinuousScheduler` with fake positional `StepBackend`,
no CUDA, model weights, network, HTTP or GPU. Compiled with MSVC 2022 x64:

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cl.exe /nologo /std:c++17 /EHsc /utf-8 /W4 /WX /TP ^
  /DHERMES_ADMISSION_CPU_ONLY /Idistributed /fsanitize=address /Zi /Od ^
  distributed\pp2_batch.cu ^
  distributed\tests\test_ready_callback_delivery_hol_cpu.cpp ^
  /Fe:ready_callback_asan.exe
ready_callback_asan.exe
```

| Actual CPU ASAN gate built against modified source | Result |
| --- | --- |
| **NEW** callback isolation/overflow/OOM/shutdown/throwing-callback test | **265/265 PASS**, **30/30 consecutive runs PASS** |
| Existing ready opt-in scheduling | **151/151 PASS** |
| Existing ready terminal/history fault hardening | **21/21 PASS** |
| Existing observability fake clock | **28/28 PASS** |
| Existing terminal ID snapshot exception hardening | **21/21 PASS** |
| Existing N2/N4/N8 fairness and backpressure | **253/253 PASS** |
| Existing scheduler mock and admission | **267/267 scheduler assertions + 11 admission PASS** |
| Existing four wave-join/synchronous callback witnesses | **4/4 PASS** |

New test cases use CV barriers, not time-based assertions for causal
ordering. They establish:

- **Old sync (off):** with A's token sink held and B's backend returned,
  B's terminal is *not* delivered until A's sink is released.
- **New opt-in (on):** with exactly that A callback held, another worker
  delivers B's token and terminal *before* A is released.
- Both A and B retain `admitted -> token -> terminal` order, one in-flight
  consumer callback per request, exact terminal generated-token vector,
  prompt-consumed usage and error field.
- A callback cancelling on SSE-style backpressure prevents the **next**
  A backend step, without starving B.
- A throwing token callback fails closed with exactly one borrowed-ID
  terminal transport wakeup; the adversarial ack race cannot produce
  a successful terminal.
- Terminal ID-copy and generated-vector allocation failures follow
  previous emergency/minimal-failed-terminal semantics even when
  asynchronous.
- Two held callback workers and 72 queued cancellations saturate the
  **64-slot** mailbox: 64 queued notifications are delivered after
  release, at least 8 additional requests receive exactly one emergency.
  Every accepted queued cancellation is accounted for without a lost
  terminal/notification; shutdown racing a full mailbox also wakes A/B.
- `shutdown()` waits for CV-held user code and joins all managed workers
  after release; idempotent second shutdown; partial worker-construction
  failure rolls back without leaked joinable threads.

The direct host timing value of a callback, any observed Windows thread
peak, and prior synthetic fake-clock metrics are **not** GPU throughput
evidence. CPU ASAN green cannot independently establish correctness of
real CUDA stream/allocator behavior or the integration thread-safety
of the HTTP/tokenizer sink. The server's current configuration has **not**
enabled the opt-in, and this worker did not modify it.

**Linux GCC ASAN/UBSAN was not executed:** the owned Windows environment
has no installed GCC, clang++ or usable WSL Linux distribution. Running
Linux UBSAN or CUDA would require an independently authorized environment.
Suggested CPU-only Linux follow-up:

```sh
g++ -std=c++17 -Wall -Wextra -Werror -O1 -g -pthread \
  -DHERMES_ADMISSION_CPU_ONLY -Idistributed -x c++ \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  distributed/pp2_batch.cu \
  distributed/tests/test_ready_callback_delivery_hol_cpu.cpp \
  -o /tmp/hermes-ready-callback-hol
ASAN_OPTIONS=detect_leaks=1 /tmp/hermes-ready-callback-hol
```

**Deployment gate:** Before enabling in serving, separately audit the
thread-safety of its shared tokenizer and stream/event sink when different
requests' callbacks overlap, and run bounded real-GPU aggregate
tokens/sec plus p50/p95 tail latency A/B. This report makes **no GPU
speedup claim** and does **not** authorize enabling the flag in production.
