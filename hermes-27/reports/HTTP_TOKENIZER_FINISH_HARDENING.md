# HERMES-27: serving tokenizer concurrency and terminal HTTP/SSE hardening

**Scope:** narrow owner worker8, 14 Nov 2026. Production file ownership:
`serving/server.cpp` only. Additional NEW test/report files only. No edits
to PP2 batch/fwd, scheduler header, production build, CMake, Git, deployed
services or GPU. No SSH/network activity.

## Baseline / provenance

Source inspected on assignment:
`serving/server.cpp` SHA-256
`933226139d2b08bc69fda55ca8c6e53307483c0643fa7bac79a56c57adcf582f`,
78,324 bytes, as read from the shared Windows workspace.

**Crucial distinction:** that *already current* source had addressed part of
the prior adversarial report. Specifically,
`TokenizerBridge::incremental_` was already
`std::atomic<bool> incremental_{false}`; the getter used a relaxed atomic
load and restart/child lifecycle used relaxed atomic stores (source
lines 155–159, 185, 239–240, 314 in the input snapshot).
The code also already contained a `noexcept` wrapper around
`complete_impl`, the `on_terminal_delivery_failure` scheduler hook,
an emergency `done.store(true)` + `notify_all()` fallback, and
`HERMES_ADMISSION_CPU_ONLY` mock fault injectors for
`HERMES_TEST_FAIL_COMPLETE_ONCE` and
`HERMES_TEST_FAIL_TERMINAL_EVENT_ONCE`.
These preexisting corrections were **preserved**. It would be inaccurate to
attribute the atomic conversion or the initial noexcept wrapper to this task.

### Specific residual hazard fixed here

The old `complete_impl()` appended a finish-reason success SSE frame,
then an optional usage SSE frame, then `[DONE]`, through separate fallible
deque allocations *under the HTTP job mutex*. If usage or `[DONE]` insertion
failed after the success finish had already entered the queue, the noexcept
fallback would append only a best-effort `[DONE]`. An SSE client could
therefore see an apparently successful finish with missing usage, or an
incomplete terminal sequence. Memory pressure also made the best-effort
emergency delimiter unreliable.

## Source corrections, `serving/server.cpp`

1. Kept the preexisting relaxed atomic `incremental_` reads and writes.
   The getter no longer races with helper restart's capability stores.
   Atomicity does not guarantee that the Python helper's incremental stream
   state survives a process restart; existing `call()` retry rules still
   fail closed for non-idempotent `stream_*` operations.
2. Added `kTerminalTransportFailureSse`: one compile-time literal containing
   a JSON SSE `error` event with generic terminal failure message followed
   by exactly one `data: [DONE]`. Neither string serialization nor a
   dynamic deque append is needed to write it in the emergency SSE sender.
3. Added `Job::terminal_transport_failed` (plain bool guarded by
   `job->mutex`). The preexisting `fail_terminal_transport_noexcept()`
   now clears any unsent response chunks, resets their byte count, sets
   terminal transport failure, HTTP status 500, `done=true`, and
   `cv.notify_all()`. This prevents unsent partial success terminal
   frames from escaping after an exception.
4. The SSE chunked-content provider checks that flag under the **same**
   job mutex before dequeuing ordinary frames. It writes the static
   error+`[DONE]` payload, calls `sink.done()` and returns. A failed socket
   write retains the existing cancel/fail transport behavior.
5. `complete_impl()` now constructs the full success finish/optional
   usage/`[DONE]` sequence in a local `std::string terminal_sse` *before*
   modifying the response queue. It enqueues the entire terminal
   sequence with one deque insertion and then sets `done=true` while
   holding the job mutex. If construction or queue insertion throws, the
   noexcept wrapper selects the emergency error path, rather than exposing
   part of the normal successful terminal sequence. The external SSE
   content is still the same series of `data:` frames separated by blank
   lines; only the internal deque grouping changes.
6. Both non-stream HTTP and pre-admission streamed HTTP error handlers
   now return a generic nonempty `request finalization failed` message
   if an allocation failure has cleared `job->error` and set status 500.
   Existing status classifications for 429/503, bearer authentication,
   normal finish_reason `length`/`stop`/`cancelled`, actual token counts,
   optional usage, SSE `[DONE]` and retry headers are unchanged.
7. Added two strictly compile-time-gated (`HERMES_ADMISSION_CPU_ONLY`)
   deterministic single-shot injectors:
   `HERMES_TEST_FAIL_TERMINAL_USAGE_ONCE` (after local success frame
   construction, before usage) and
   `HERMES_TEST_FAIL_TERMINAL_QUEUE_ONCE` (before the final deque commit).
   These are absent from the production CUDA build; no environment flag
   can enable them in that build. Added explicit `<new>` for
   `std::bad_alloc`.

## New targeted tests

1. `serving/tests/test_tokenizer_bridge_exception_source.ps1`
   is a **source-only**, Windows PowerShell 5.1 local contract test. It
   parses actual `server.cpp`, verifies the atomic field/getter/store
   contract, the noexcept terminal wakeup, one atomic full terminal SSE
   commit, static error+`[DONE]` fallback, both old and new CPU-only
   fault injectors, and scheduler emergency callback wiring. Four
   **in-memory source mutations** deliberately replace the atomic member,
   disable the failure marker, sabotage the complete terminal deque
   insertion, or remove the static SSE write; each invalidates its
   corresponding source contract.
2. `serving/tests/test_tokenizer_bridge_exception_cpu.py`
   contains **seven dynamic Linux CPU fake-backend HTTP tests** using the
   existing `prod_sched_state_test.running_server` and
   `HERMES_ADMISSION_CPU_ONLY` build, **not the GPU/production binary**.
   It exercises original bearer auth 401/authorized 200; failing
   nonstream terminal JSON and successful retry; failing the entire
   terminal callback and recovery; and four independent SSE failures:
   before full terminal construction, after locally staged success frame,
   at final queue insertion, and at event delivery. All error cases
   require an explicit error event plus one `[DONE]`, prohibit false
   success finish/usage, and require a later normal successful completion
   with preserved `finish_reason=length` and correct usage. All network
   requests in this *future test design* are local loopback only, with
   bounded 8-second HTTP timeouts; this worker **did not run them**.

### Actual validation executed in this turn

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File \
  serving/tests/test_tokenizer_bridge_exception_source.ps1
```

**PASS 13/13 source-contract checks**, including **4/4 negative in-memory
mutants correctly rejected**. Exit code **0**. No Python, compiler, fake
server, CUDA, model, sockets, network or service were invoked.

**Not run:** The seven dynamic fake-server HTTP tests, full serving binary
compilation, sanitizers/TSan, and end-to-end GPU inference. The shared Windows
host exposed **no Python executable, no C++ compiler, and no installed WSL
distro**; the user's ban on SSH/network/service activity also precluded
running the existing Linux CPU fake-server environment. Static source tests
are not evidence of binary/runtime correctness. Prime must compile the
preexisting CPU-only mock as usual and run the new tests with explicit
`--server`, `--model`, `--gguf-dump`, `--tokenizer`, `--python`
arguments matching `prod_sched_state_test` on its authorized isolated
Linux CPU runner. **Do not use the production GPU executable**.

## Residual caveats and limits

- An SSE HTTP **200** header already sent after GPU admission cannot
  subsequently be changed to HTTP 500. The emergency branch must
  signal failure via SSE `error` plus `[DONE]`. Already-transmitted
  incremental content is not retractable; unsent chunks are discarded
  to prevent publishing a spurious successful finish.
- If a client disconnects or a `DataSink::write` fails, the existing
  socket-close/cancellation behavior applies. The static error payload
  avoids *host allocation*, not a broken socket or process termination.
- A synchronous `TokenizerBridge` operation may still block on Python
  I/O until the existing transport timeout, and prior incremental decoder
  state can be invalidated by a helper restart. Atomic capability access
  removes C++ undefined behavior, not this semantic restart limitation.
- Core `ContinuousScheduler` exception safety, event construction and
  GPU worker join logic are worker6-owned and **were not changed**.
  The serving emergency callback relies on the core's existing
  `on_terminal_delivery_failure` contract.
- An actual TSan run and injected Linux fake-server E2E tests remain
  mandatory before promoting this as runtime verified. No full-model
  numerical or throughput claim follows from this source hardening.

## Frozen source and test hashes

SHA-256, computed locally after the last source correction and the 13
source tests:

- `serving/server.cpp`:
  `ad2522dc7297bb9e3826a617778ec580e591d917e6bde3796adc3146e596a7f4`
- `serving/tests/test_tokenizer_bridge_exception_cpu.py`:
  `9232c4c9c7f8fca5b9df5a902794010927bd8b8afa838532878e72756830c94d`
- `serving/tests/test_tokenizer_bridge_exception_source.ps1`:
  `71d44b71278ee769249bc628f7858bf545d7d9b3bf5f30dc82ee5d523d1dcfbc`

Review result: **source contract PASS; runtime verification BLOCKED by
local CPU compiler/mock unavailability.** No release acceptance or production
deployment claim.
