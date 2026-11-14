# HERMES-27 PP2 ready-completion terminal fault hardening

**Scope:** CPU/offline code hardening, 14 November 2026. Exclusive edits to
`distributed/pp2_batch.cu`, `distributed/pp2_batch.hpp`,
`distributed/tests/test_ready_completion_fault_hardening.cpp` and this report.
No serving, CUDA kernels, CMake, Git, SSH, model, service or GPU executions.
The Windows mirror does not contain `.git`, so no commit/branch is claimed.

## Corrected failure sequences

1. **Previously:** `finish()` detached a raw DecodeState and marked terminal
   before copying error text and appending history; either `bad_alloc` skipped
   `backend.close()` and the terminal callback. **Now:** fallible error/history
   preparation is contained before detaching the state. The state is detached
   only after metadata is fully committed, then `close()` is called exactly
   once. History is best effort; on failure, the terminal is still emitted and
   its unretained map entry is removed after delivery to maintain a bound.
2. **Previously:** queued `cancel()` removed a job and marked it cancelled
   before a fallible `finished_order_.push_back()`; a thrown exception could
   strand the client. **Now:** cancellation shares the no-throw history helper,
   emits one terminal outside the lock even on failed retention, and discards
   its unretained map entry after callback.
3. **Previously:** `emit_event()` copied terminal ID/generated/error outside
   its callback `try`; an allocation failure could skip the callback entirely.
   **Now:** `emit_event() noexcept` catches snapshot and consumer failures.
   A failed snapshot is retried as a minimal `JobStatus::failed` terminal,
   with no unverified generated tokens. If this also throws, or the original
   terminal callback throws, `on_terminal_delivery_failure(id,status)` is
   invoked with a borrowed job ID. The consumer can set its terminal error and
   notify waiters without allocating. This is a new **server integration
   obligation**; the callback must not throw, retain ID, or call tick/shutdown.

Added metrics `terminal_history_failures_total` and
`terminal_delivery_failures_total` distinguish memory-pressure degradation
from completed backend decode. Existing `callback_errors_total` also counts
snapshot failures. Three deterministic fault callbacks are exposed **only**
under `HERMES_ADMISSION_CPU_ONLY`:
`terminal_before_error_copy`, `terminal_before_history_append`, and
`terminal_before_snapshot_copy`. They are absent from the CUDA build.

## Test evidence

Actual runs use MSVC 2022 x64 `cl.exe /nologo /std:c++17 /EHsc /utf-8
/W4 /WX /TP /DHERMES_ADMISSION_CPU_ONLY /Idistributed
/fsanitize=address /Zi /Od`, with `vcvars64.bat` environment and
`pp2_batch.cu` compiled as C++. Backend allocates only fake CPU states.

| Suite | Observed result | Contract |
| --- | --- | --- |
| New `test_ready_completion_fault_hardening.cpp` | **21/21 PASS** after final `emit_event() noexcept` edit | History `bad_alloc` with fast B while A blocked, queued cancel history `bad_alloc`, terminal snapshot `bad_alloc`, throwing terminal consumer/wakeup, double snapshot+consumer failure, error-copy `bad_alloc`; state closes and notification checks |
| Existing `test_ready_completion_optin_scheduler.cpp` | **151/151 PASS** | Ready B delivered before slow A; default wave-join unchanged; cancellation, shutdown, fault recovery |
| Existing `test_pp2_batch.cu` (CPU-only `admission`) | **267/267 scheduler + 11/11 admission PASS** | Original scheduler and admission contracts |
| Existing `test_throughput_stream_event_scheduler.cpp` | **4/4 PASS** | Original wave-join and synchronous callback behavior |

The three existing suites were built/run with MSVC AddressSanitizer before
the final source-only `noexcept` declaration alignment; the new suite was
rebuilt and rerun after that final edit. No CUDA or physical-GPU inference
measurement is claimed.

**GNU GCC/ASAN+UBSAN:** not run on this Windows host. No GCC is installed on
PATH and WSL reports no Linux distribution; installing tools or using SSH
was outside this worker's authorized scope. The integration owner should run
the final snapshot under Linux GCC with
`-std=c++17 -Wall -Wextra -Werror -pedantic -pthread -O1 -g
-fsanitize=address,undefined -fno-omit-frame-pointer
-DHERMES_ADMISSION_CPU_ONLY -Idistributed -x c++` for both the new fault test
and existing ready test. The same build verifies that no final-source Ubuntu
UBSan regression exists.

## Remaining server obligation and limits

- Wire `SchedulerConfig::on_terminal_delivery_failure` in
  `serving/server.cpp` so a thrown terminal callback wakes non-stream and
  SSE jobs with an explicit 500/error. The hook is needed even if
  `complete()` is already `noexcept`, because earlier callback work can
  throw before `complete()` is reached. Do not infer HTTP correctness from
  the scheduler-only fault tests.
- If host allocation remains permanently impossible, even a minimal terminal
  snapshot may fail; the borrowed-ID callback is the last-resort channel.
  No scheduler can force a throwing/absent external consumer to notify its
  own waiters. The caller must install the hook and honor its no-throw contract.
- A snapshot failure reports a failed terminal to the consumer, while the
  scheduler's completed-backend counter can still reflect an already-completed
  decode. The new delivery-failure metric records that distinction.
- Ready completion continues to allow fast already-admitted requests to
  progress while a slow sibling is running; new PP2 state admission remains
  conservatively gated until all in-flight steps finish. No full HTTP/GPU
  throughput assertion follows from these CPU tests.
