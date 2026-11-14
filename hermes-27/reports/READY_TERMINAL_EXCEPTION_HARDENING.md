# HERMES-27 — ready scheduler terminal exception audit and bounded fallback

**Date:** 2026-11-14
**Scope:** CPU-only exception hardening and source review. No GPU, model
inference, service, SSH/network, Git, serving/server.cpp, PP2 CUDA forward,
CMake, or unrelated workspace edits.

## Starting provenance and finding

The reviewed Windows workspace source hashes **matched** the supplied
published ready/observability snapshot:

| File | SHA-256 before edits |
| --- | --- |
| `distributed/pp2_batch.cu` | `75ACC314D85564B0F76A9C366FD568352C6B2673520A497223AD31A49FD84077` |
| `distributed/pp2_batch.hpp` | `7EA54D1CD16B142132391216CA3FA1A58ECF4FC3E1B0876E5AAD6318F2B9FBB2` |

The externally reported published Git commit is `db063f14`. This worker's
Windows mirror is **not** itself a Git checkout, so the commit identity is
supplied provenance rather than independently validated here.

Worker8's earlier conditional P0 is largely **already addressed in the
current sources**, so its old line numbers and failure sequence must not be
treated as an outstanding bug:

- `finish()`, now around lines 627–663, prepares its potentially allocating
  error string inside a catch and attempts bounded, noexcept
  `retain_finished_locked()` before detaching the raw `DecodeState`.
  `backend_.close(state)` follows the metadata lock and precedes the terminal
  event. Unretained history entries are forgotten after notification.
- Queued `cancel()`, around lines 485–506, already uses the same guarded
  retention helper, and calls `emit_event(terminal)` outside the metadata
  lock even when history allocation fails.
- `emit_event() noexcept`, around lines 539–624, already handles event
  snapshot and user callback exceptions; its minimal failed terminal
  fallback and `on_terminal_delivery_failure` borrowed-ID hook protect
  HTTP waiters. The prime-owned server already wires this hook in
  `serving/server.cpp` around lines 442–448, with a no-throw notification
  implementation around lines 575–608. The serving code was **read only**.

### Proven narrow gap

Before this edit, if allocating the **initial** terminal event ID failed,
`emit_event()` caught the exception but attempted `event.id =
job->request.id` again while constructing a minimal failed terminal.
Under sustained host allocation pressure, that is a second vulnerable
allocation in a recovery path. Since the original `shared_ptr<Job>` still
owns its request ID, a consumer with
`on_terminal_delivery_failure(id,status)` already has a direct allocation-
free way to wake its waiter.

**Correction:** If terminal snapshot construction throws while
`event.id` is empty **and** an emergency notifier exists, invoke the
notifier with the borrowed ID immediately, without a second ID copy.
When no emergency notifier is registered, preserve the old best-effort
minimal-event retry so standalone consumers can still recover from one
transient allocation failure. If the ID was successfully copied and later
terminal payload allocation fails, preserve the existing minimal failed
terminal behavior. This changes no backend state lifecycle, lock ordering,
wave scheduling, default opt-in, or metrics schema.

The production path adds only a conditional inside the already-existing
terminal exception handler (`distributed/pp2_batch.cu` approximately
lines 597–624). The additional
`SchedulerConfig::terminal_before_event_id_copy` deterministic failpoint
exists **only** under `HERMES_ADMISSION_CPU_ONLY` (header around line 163;
source around lines 580–585). It does not exist in CUDA/production builds.

## RED / GREEN evidence

The new file `distributed/tests/test_ready_terminal_exception_id_copy.cpp`
uses a fake `StepBackend` with open/step/close counters and an explicitly
thrown `std::bad_alloc` at the ID-copy boundary:

1. **RED compilation:** The independent new test initially failed to compile
   because the CPU-only `terminal_before_event_id_copy` hook did not yet
   exist (`MSVC C2039`).
2. **RED behavior:** After adding the injection hook **without** changing the
   fallback, the new test built but failed its assertion that the emergency
   borrowed-ID notifier is used instead of an allocating retry.
3. **GREEN behavior:** After the bounded fix, the test passes. It asserts no
   terminal callback receives an incomplete/unknown recipient, exactly one
   emergency wake for a failed ID copy, reentrant `metrics()` and `inspect()`
   from that notifier without lock inversion, and close only after the
   worker returned.
4. **Compatibility:** A separate test disables the emergency notifier,
   injects one ID-copy failure and confirms a subsequent minimal failed
   terminal is still delivered to the original `on_event` consumer.
   Additional tests inject simultaneous history/snapshot failure and a
   throwing terminal consumer after a backend failure; each receives exactly
   one terminal or emergency wake, with no state leak.

## Targeted local validation

All actual builds use **MSVC 2022 x64** and
`/nologo /std:c++17 /EHsc /utf-8 /W4 /WX /TP
/DHERMES_ADMISSION_CPU_ONLY /Idistributed /fsanitize=address /Zi /Od`.
`pp2_batch.cu` was compiled as C++. They use no real CUDA, HTTP, or model.

| Executed suite | Result |
| --- | --- |
| New terminal exception test, final code | **21/21 PASS**, then **8/8 consecutive reruns PASS** |
| Existing `test_ready_completion_optin_scheduler.cpp` | **151/151 PASS** after final patch; ready fast-B and original wave behavior |
| Existing `test_ready_completion_fault_hardening.cpp` | **21/21 PASS** after final patch |
| Existing `test_ready_observability_optin_scheduler.cpp` | **28/28 PASS** after final patch; metric counts and ready/wave synthetic timing |

The new terminal test covers backend failure and successful backend completion,
without changing the original `SchedulerMetrics` structure or observability
semantics. The 50ms wave latency in observability tests is **fake clock time**.
No CUDA speedup, production inference reliability or server completion claim
is inferred from these CPU checks.

**Linux GCC ASAN/UBSAN:** Not run because the local Windows environment
provides no installed WSL Linux distro or GCC, and the ownership scope
forbids installs, SSH/network or service/GPU access. The prime can run:

```sh
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -O1 -g -pthread \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -DHERMES_ADMISSION_CPU_ONLY -Idistributed -x c++ \
  distributed/pp2_batch.cu \
  distributed/tests/test_ready_terminal_exception_id_copy.cpp \
  -o /tmp/hermes-ready-terminal-exception
ASAN_OPTIONS=detect_leaks=1 /tmp/hermes-ready-terminal-exception
```

## Boundaries of the guarantee

The scheduler can guarantee emergency notification only when the consumer
has installed and correctly implemented its no-throw callback. Without such
a callback and under **persistent** allocation failure, the best-effort
fallback event may still fail and cannot force an external HTTP waiter to
finish. The CPU tests model deterministic exceptions, not global physical
memory exhaustion. No measurement of actual GPU completion latency or
host-memory allocation overhead was performed.
