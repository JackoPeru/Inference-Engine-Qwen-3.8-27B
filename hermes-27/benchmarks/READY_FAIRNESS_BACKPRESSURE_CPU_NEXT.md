# HERMES-27 — Ready scheduler fairness, credit reuse, backpressure (CPU only)

**Evidence date:** 2026-11-14. **Status:** deterministic CPU/offline
concurrency evidence **PASS**. No product/source changes. The scheduler
source inputs were independently verified by SHA-256 both before and after
the test work:

| Published source, read only | SHA-256 |
| --- | --- |
| `distributed/pp2_batch.cu` | `279C76C0C502D10F540C0F16DD754BBADC4F746692A44D3E4BEDFBB85135FB06` |
| `distributed/pp2_batch.hpp` | `C0A40C0EFB109ACE1A5D1C06ED44582BEC86CC76C3A4824C99D766F80E213E1F` |

The Git publication/revision identifier was provided by the prime, but the
Windows mirror does not expose a local Git checkout. This worker did **not**
checkout, fetch, stage, commit, push, or edit either source file.

New exclusive file:
`distributed/tests/test_ready_fairness_credits_cpu.cpp`. It runs the
**actual** `ContinuousScheduler` against a fake stateful backend (no CUDA
or inference). A's step is held by a condition variable, no sleeps are used
to establish event ordering, and all waits are bounded deadlock guards,
**not** latency benchmarks. Every synthetic observation timestamp comes
from a test-only atomic monotonic fake clock. The fake backend rejects
replayed/overlapping positions, counts state opens/closes, and checks
join-before-close. No application event payloads, logs or real tokens
are materialized in the scheduler's opt-in metrics.

## Verified concurrency contracts

| Scenario | N=2 | N=4 | N=8 |
| --- | --- | --- | --- |
| Mid-flight new cohort and queue-full backpressure | PASS | PASS | PASS |
| Four-tick round-robin with quantum N/2 | PASS | PASS | PASS |
| Two tokens per fast peer while A remains in its first backend step | PASS | PASS | PASS |
| Queued deadline, queued cancel, failed backend B while A blocked | PASS | PASS | PASS |

### 1. New arrival with slow state and bounded queue

The initial N clients A..N each start a single step. A is held inside its
backend call; B..N deliver terminal results before A is released. A second
cohort Q0..Q(N-1) **arrives while A remains in-flight**. It is accepted into
the queue but not admitted to a new backend state. The queue is filled to
`max_queued=N`; an overflow request is rejected with `queue_full`.
Cancelling queued Q1 emits one cancelled terminal, releases one waiting
capacity credit, and admits a replacement *to the waiting queue*. A is then
released; the next `tick()` admits the surviving Q requests and replacement
in exact FIFO order and they all complete.

Observed exact finished counters: N=2 gives 4 completed / 1 cancelled /
1 rejected; N=4 gives 8/1/1; N=8 gives 16/1/1. Each scenario opens and
closes precisely 2N state objects, never frees one while a backend step is
in-flight, and samples queue/active/outstanding high-water with configured
bounds. The instantaneous fake backend `peak_steps` is not consistently N
(Windows may not schedule every CPU thread simultaneously); the test asserts
its upper bound rather than inventing achieved parallelism.

### 2. Round-robin host launch fairness

With N distinct requests, each requesting two output tokens, and
`steps_per_tick=N/2`, four successive `tick()` calls produce:

- tick 1: first N/2 states receive one token; the other half zero
- tick 2: all N states have one token
- tick 3: first N/2 states have two tokens; the other half one
- tick 4: all N states completed with two tokens

The independent backend checks each state receives **exactly one** position
0 and **exactly one** position 1, with no duplicate concurrent same-state
worker. Total sample counts are 2N backend steps and 2N host token commits.
The test validates RR **launch fairness at successive finite quanta**, not
GPU scheduling fairness under unbounded arrivals or OS thread fairness.

### 3. Throughput-credit reuse despite one blocked peer

With A blocked in step position 0 and a total step allowance of 2N,
the other N−1 requests each complete *two* backend steps and emit *two*
token events **before A returns**. Final backend steps are 2N−1: 3 for N=2,
7 for N=4, **15 for N=8**, safely within the scheduler's 16-step task array.
The fast peers' completed tokens while A is blocked are 2, 6, **14**
respectively. This is a deterministic causal/concurrency property,
**not** measured tokens/sec or GPU speedup.

### 4. Deadlines, cancellations, error classification

A remains in a backend step; the test enqueues one request with an
explicitly backdated clock argument (`SchedulerClock::now() - 1 s`) against
an 80 ms queue deadline. This yields a deterministic queued expiry in the
ready pump's bounded deadline polling even while A is held. A distinct
queued client is cancelled and its terminal arrives immediately. Backend
B independently returns an error and produces **no** token event.

For each N, result totals are **1 expired, 1 cancelled, 1 failed, N−1
completed**; the two queued terminal requests never have an admitted event
or backend state. The `observed_queue_admissions`, backend attempt count,
successful token commits, and successful ready terminal samples match
their distinct semantics. This does not depend on wall-clock waiting to
trigger expiration.

### 5. Callback stalls and bounded SSE-style backpressure

In a separate N=4 test, B's token sink invokes `inspect()` and
`metrics()`, then `cancel("B")` (a simulated full per-request SSE
buffer), and intentionally waits on a controlled CV. C and D return from
their backend calls while B's callback is still blocked; **neither C nor D
can emit a token until B's callback returns**, because callbacks run on
the single pump. Once released, C/D deliver tokens while A remains
in-flight; B emits one cancellation terminal and never another token.
The three survivors complete on a subsequent tick. The callback's
fake-clock advance of **7,000,000 ns** appears in
`SchedulerMetrics::callback_ns_max` (about 7,003,000 synthetic ns in
the observed runs). The callback can reenter the permitted scheduler APIs
without metadata-lock deadlock. This asserts bounded cancellation behavior
and exposes global callback head-of-line blocking; it does not exercise
the actual serving HTTP transport.

## Source-anchored throughput analysis

1. **Strict ready dispatch bounds:** `pp2_batch.cu` lines 965–995 define
   the fixed 16-task array and at-most-`max_active` in-flight workers.
   Lines 1099–1115 rotate `running_` round-robin at launch, skipping
   an already-in-flight same-state request. Lines 1158–1183 publish worker
   completion under the CV; lines 1233–1258 join before token commit,
   cancellation, close or state reuse. Tests above verify these invariants
   for N2/N4/N8, not the GPU implementation.
2. **Throughput opportunity (NOT classified P0):** lines 1078–1085 call
   `promote()` only when `outstanding == 0`, guarding against implicit
   synchronization during current PP2 state allocation/seed. A newly
   arriving client therefore **cannot start a new backend state** while
   an older slow backend step remains in-flight, **even if fast jobs have
   already closed and capacity exists**. Existing active peers can still
   re-launch and use the remaining step credits. This is a conservative
   correctness gate and a potential throughput bottleneck requiring a
   separately proven async-allocation CUDA contract before relaxing it;
   no source edit or P0 claim is justified by fake CPU timing alone.
3. **Callback HOL (known nonblocking contract):** `emit_event()` runs the
   consumer synchronously outside `mutex_` (lines 539–624). Ready pump
   commits and invokes token callbacks sequentially (lines 1303–1314).
   A slow sink can delay ready sibling commits despite backend completion.
   The API documents that sinks must remain nonblocking and bound their
   transport queues (header lines 171–178). The proper next measurement
   is real serving sink queue stalls and pipeline backpressure, **not**
   changing PP2 state ownership or putting callbacks on backend workers.
4. **Fairness limit:** A fast peer may use multiple remaining steps during
   the same tick while a slow peer remains blocked, as intended for
   throughput. The fixed step quantum limits per-tick work, but the pump
   still waits for all in-flight workers before releasing `pump_mutex_`;
   any next tick or new-state admission must wait if A cannot complete.
   This is not a general starvation-freedom proof for arbitrary live load.

**No demonstrated safety P0** emerged from these bounded scenarios. No
request to modify `pp2_batch.cu/.hpp` was made and no edits were performed.

## Executed build/test evidence

Windows MSVC 2022 x64, CPU/offline, compiled
`distributed/pp2_batch.cu` as C++:

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cl.exe /nologo /std:c++17 /EHsc /utf-8 /W4 /WX /TP ^
  /DHERMES_ADMISSION_CPU_ONLY /Idistributed /fsanitize=address /Zi /Od ^
  distributed\pp2_batch.cu ^
  distributed\tests\test_ready_fairness_credits_cpu.cpp ^
  /Fe:ready_fairness_asan.exe
ready_fairness_asan.exe
```

- **253/253 assertions PASS** across 13 deterministic scenario variations
  (N2/N4/N8 × four categories, plus N4 callback backpressure).
- Final test source: **12 consecutive MSVC ASAN reruns PASS**, each emitted
  `PASS ready fairness/backpressure CPU assertions=253`.
- The backend's step wall and callback timings come entirely from
  `FakeClock::now()`; do **not** treat sample nanoseconds, peak concurrency
  or Windows test runtime as a device benchmark.
- Existing 151 ready, 21 fault, 28 observability and 21 terminal-exception
  test suites were **not rerun** because the scheduler source was unchanged.
  Their prior green results remain provenance from earlier assigned work.
- **Linux GCC ASAN/UBSAN not run:** no installed GCC or WSL distribution
  was available in the worker's Windows environment, and the request forbids
  software installation, network/SSH, GPUs or services.

Suggested Linux CPU-only follow-up, from an environment with GCC already
available:

```sh
g++ -std=c++17 -Wall -Wextra -Werror -O1 -g -pthread \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -DHERMES_ADMISSION_CPU_ONLY -Idistributed -x c++ \
  distributed/pp2_batch.cu \
  distributed/tests/test_ready_fairness_credits_cpu.cpp \
  -o /tmp/hermes-ready-fairness-asan
ASAN_OPTIONS=detect_leaks=1 /tmp/hermes-ready-fairness-asan
```

**Next real acceptance gate:** controlled multi-client GPU runs comparing
aggregate output tokens/s and per-job time-to-first-token with
`ready_completion` enabled/disabled, outstanding allocation busy-gate
frequency and `callback_ns_max`, under the same PP2 stream/capacity model.
These measurements require separate GPU/service authorization and were
not attempted here.
