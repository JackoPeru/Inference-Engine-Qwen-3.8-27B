# HERMES-27 PP2 ready-completion host observability

**Status (2026-11-14):** Opt-in CPU-side instrumentation implemented in
`distributed/pp2_batch.cu/.hpp` with deterministic CPU-only tests. No CUDA,
model, GPU execution, inference throughput measurement, service, SSH, or Git
operation was performed by this worker. The Windows mirror has no `.git`.
Only the two scheduler sources, new
`distributed/tests/test_ready_observability_optin_scheduler.cpp`, and this
new report were edited. Existing ready/fault, wave and mock tests unchanged.

## Instrumentation contract

Set `SchedulerConfig::enable_observability = true` explicitly. The default
`false` retains original scheduling and makes **zero clock calls** for these
metrics; the disabled test deliberately installs a throwing fake clock and
verifies it is never called. Opt-in uses `SchedulerClock::steady_clock` in
production. The optional `SchedulerConfig::observability_now` test clock
exists **only** under `HERMES_ADMISSION_CPU_ONLY`; it must be thread-safe,
nonblocking and must not reenter the scheduler.

Each job/backend state owns precisely two relaxed atomic timestamps for
the most recent step start/end and one immutable queue-entry timestamp.
Exactly one backend worker can own one state at a time. The pump reads
timestamps only after the same worker has joined; worker threads do not
modify scheduler counters or invoke event callbacks. No dynamic metric
buffer, token array, job ID, prompt, user identifier or per-request histogram
is retained for observability. The existing scheduler mutex guards every
aggregate sample; user callbacks and the clock are called without holding
it. The instrumentation does not allocate a GPU event or synchronize GPU
streams. Individual `SchedulerEvent` values and job retention rules are
unchanged.

`ContinuousScheduler::metrics()` returns a fixed-size
`SchedulerMetrics` snapshot. Durations are integer nanoseconds, as
**saturating unsigned 64-bit sums and maxima**, with explicit count fields.
All maxima, counters and high-water fields start at zero. Samples with an
invalid or reversed injected clock pair are skipped. The clock probe catches
its own exceptions without interrupting decode.

| Counters (count, nanosecond sum, maximum) | Clock interval / inclusion |
| --- | --- |
| `observed_queue_admissions`, `queue_wait_ns_total/max` | Accepted enqueue to successful state open/admission, including state allocation time. Queued cancel, expiry, or failed open have no admission sample |
| `observed_backend_steps`, `backend_step_ns_total/max` | Per-state worker start to backend result/error handling; includes attempted failed/cancelled steps that actually ran |
| `observed_token_commits`, `backend_ready_to_token_commit_ns_total/max` | Backend step completion timestamp to host sampled-token commit, before user callback. Only committed output tokens are counted; excludes failed/cancelled steps and prompt-only steps |
| `observed_ready_terminals`, `ready_terminal_ns_total/max` | Last step completion to **successful terminal status commit** with ready dispatch enabled. Excludes errors, cancelled/expired jobs and backend-free queue terminals |
| `observed_wave_terminals`, `wave_terminal_ns_total/max` | Same interval for ordinary wave-join parallel dispatch |
| `observed_serial_terminals`, `serial_terminal_ns_total/max` | Same interval for the original serial scheduler |
| `observed_callbacks`, `callback_ns_total/max` | Synchronous `on_event` invocation duration for all event kinds, including a throwing sink |
| `observed_terminal_callbacks`, `terminal_callback_ns_total/max` | Synchronous terminal sink duration, including the minimal failure-terminal fallback |
| `queued_high_water`, `active_high_water`, `outstanding_step_high_water` | High-water counts for waiting queue, admitted active states, and simultaneous backend worker steps; bounded by configured queue/active/quantum capacity |

For any pair, arithmetic average host duration in milliseconds is
`(total_ns / count) / 1e6` for a nonzero count. These are **host wall
intervals** and can contain pump scheduling delay, callback contention and
CPU processing. They are not CUDA kernel durations, token processing time
or true GPU bandwidth. The terminal metric measures when scheduler status
is committed; callback stall is measured separately and the transport may
add more latency.

## Executed CPU test and falsification scenarios

MSVC 2022 x64 locally, `cl.exe /std:c++17 /EHsc /utf-8 /W4 /WX /TP
/DHERMES_ADMISSION_CPU_ONLY /Idistributed /fsanitize=address /Zi /Od`.
All suites build `distributed/pp2_batch.cu` as C++ and use fake memory
states, with no CUDA dependency.

| Executed gate | Actual outcome |
| --- | --- |
| NEW `test_ready_observability_optin_scheduler.cpp` | **28/28 PASS**; final version **12 consecutive PASS runs** under AddressSanitizer |
| Existing `test_ready_completion_optin_scheduler.cpp` | **151/151 PASS**, ready fast B and default wave barrier |
| Existing `test_ready_completion_fault_hardening.cpp` | **21/21 PASS**, injected `bad_alloc`, terminal wakeup and state ownership |
| Existing `test_pp2_batch.cu admission` | **267/267 original scheduler assertions + 11/11 admission PASS** |
| Existing `test_throughput_stream_event_scheduler.cpp` | **4/4 PASS**, wave-join and callback-stall witness |

The new fake clock advances a positive amount at every observation. B's
backend step reaches its recorded end timestamp while A remains blocked by
a condition variable; only then does the test release A and inject an
additional **50,000,000 simulated nanoseconds**. This check initially
exposed a test synchronization race (B had signalled an impending return
but had not yet taken its final timestamp). A separate timestamp-sampled
barrier fixed that race. After correction, 12/12 AddressSanitizer reruns
pass. The resulting original wave mode consistently reports a
`wave_terminal_ns_max` around 50,011,000 **synthetic** ns, while the
ready mode delivers B's terminal **before** A is released. There is no
measured production GPU speedup or claimed real 50ms wall latency.

Other deliberate assertions: a cancelled queued C and a backend-failed B
do not inflate successful token/terminal/admission counts; two attempted
backend steps are still counted, all states are closed after joins, and
callback stalls of 1.5ms token/2.5ms terminal are visible through the
synthetic monotonic clock. One serial-mode test verifies its terminal
samples are not mislabeled as wave/ready. With observability disabled,
no clock invocations or observation samples occur.

**Linux GCC ASAN+UBSAN:** not run from the Windows worker. GCC is unavailable
on PATH and WSL reports no Linux distribution; the task forbids installation,
SSH and GPU/service activity. Targeted integration follow-up on an
available Linux compiler:

```sh
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -O1 -g -pthread \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -DHERMES_ADMISSION_CPU_ONLY -Idistributed -x c++ \
  distributed/pp2_batch.cu \
  distributed/tests/test_ready_observability_optin_scheduler.cpp \
  -o /tmp/hermes-ready-obs-asan
ASAN_OPTIONS=detect_leaks=1 /tmp/hermes-ready-obs-asan
```

The serving owner may read and export the aggregate
`ContinuousScheduler::metrics()` fields in an HTTP/metrics or benchmark
adapter. This worker did not modify `serving/server.cpp` or any production
configuration, so no real deployed endpoint exposure is asserted.
