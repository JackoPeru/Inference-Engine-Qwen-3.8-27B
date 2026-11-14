# HERMES-27: Unicode, stop and SSE terminal CPU regression + legacy crash fix

**Worker 16; 2026-11-14. Status: verified local Windows source correction; Git publication belongs to prime.**

Scope: `serving/server.cpp` (only the prime-authorized `legacy_token()` exception containment for streamed chunks and output-vector growth, plus one preinitialized error message), NEW `serving/tests/test_http_unicode_stop_terminal_cpu.py`, and this NEW report. No other tracked source, CMake, Git, service, systemd, GPU, real inference endpoint, or production ingress changed. All Linux builds were isolated under `/tmp/h27-worker16-unicode-5c74e211` on Ubuntu; the original published checkout was read-only.

## Frozen provenance

| Item | SHA-256 / Git ID |
| --- | --- |
| Published Ubuntu `serving/server.cpp` (before and after; unchanged in Git) | `ad2522dc7297bb9e3826a617778ec580e591d917e6bde3796adc3146e596a7f4` |
| Shared Windows `serving/server.cpp`, FINAL narrow fix | `12340ecd1d649799304328e7f8929ae35e43bd8010279c0bf265df86c7709b28` |
| Intermediate source (stream_text fixed, before output-vector fix) | `f13bfe9763ec0760381e057712e33a886d0c5b40b371bf5dfca7c125b86f27d6` |
| NEW test `serving/tests/test_http_unicode_stop_terminal_cpu.py`, FINAL | `aa4694a617920b72b597ad676f62879b36a53fb367402d5d664ec801e96eae82` |
| Baseline CPU-only binary (real server + fake PP2) | `a42c71fdd75a54130d54ad2a7a654b32c8fe7617058db7fd315f4368868ba956` |
| FINAL fixed CPU-only binary | `6b08083b72075cdbc07a88709dad9267e4e9c1e033063254b2c20cad7b5bbe1a` |
| Initial stream fault injected baseline source / binary (SCRATCH ONLY) | `9a4f46a6c9c4712fa31f794d5577489c06ac54b9e8ecc93d1ddfe38cde41d8d7` / `92dbe44b49caa776e64eef823fa2018c35157c7bf4fdff7b87294ca1056c6275` |
| FINAL stream fault injected source / binary (SCRATCH ONLY) | `38fb93ec1ff64c96903f81e5663a319e4e5c3866197ad52c0afba0f3a2d4ba35` / `c8295f4efa103ac610b8f253f52d3658889fe19b472f4b61a845699bb697ad83` |
| Output-vector fault RED source / binary (SCRATCH ONLY, intermediate source) | `b62069eb1956c82149a41a229c0dc3b4b1cfaf9d6e6b068b17f4fea86eee6693` / `120e685869e843f7dccb6910f53664ccc382a1c0646e0bd0aef3a28e90440337` |
| FINAL output-vector fault injected source / binary (SCRATCH ONLY) | `3f8eb09a255b010057d20c00c060a6d52269799e19d0534899ea576e40d93263` / `942b45a942007be6c5184d5eb715bfe8eab65a2e29fa201ec374614b735544c3` |
| FINAL fixed CPU-only ASAN binary | `2ce023e89ae41d5ac83b3fb9fef1a50a8b10ff879026dd1e637167116dc1baa2` |

The published Ubuntu `main` Git HEAD was `5c74e211e19a7b4a6f3d05a7a62b63430aedb718` at initial inspection; it advanced independently through `727d9cdf7a418e3888fc4d33730a8779a9108e07` to **`8a829136aeefdf68bb964ed9c39fc5a2d007d154`** at final read-only check. Its `serving/server.cpp` remained SHA `ad2522dc…a7f4`, and the Ubuntu `git status --short --branch` showed clean `main...origin/main`. **The Windows source fix is not yet published or GPU-tested.** No Git actions were performed by this worker.

## Reproducible source bug, before correction

In published `serving/server.cpp`, `legacy_token()` (roughly lines 988–1000) called `stream_text(job)` immediately **after** its `try/catch` for `tokenizer_.stream_push` / stop processing. `stream_text()` (roughly lines 794–811) performs fallible `substr`, nlohmann JSON, serialization and deque allocations. `std::bad_alloc` there escaped the legacy worker thread, invoked `std::terminate`, aborted the entire HTTP process and truncated the active SSE response. The independent continuous scheduler already enclosed its `stream_text()` call inside the token callback's catch path.

Deterministic fault witness: in a **separate copy** of the identical source, insert the following CPU-only, once-per-process exception immediately after the unique source anchor `const std::string delta = job->text.substr(job->sent_bytes);` inside `stream_text()`:

```cpp
#ifdef HERMES_ADMISSION_CPU_ONLY
    if (std::getenv("HERMES_TEST_FAIL_STREAM_TEXT_ONCE")) {
      static std::atomic<unsigned> stream_text_faults{0};
      if (stream_text_faults.fetch_add(1) == 0) throw std::bad_alloc();
    }
#endif
```

No fault injection was introduced into `server.cpp` itself. This is a **controlled simulation of an allocation failure**, not evidence of an actual physical host OOM. The same fault location and the same fake CPU PP2 backend were used before and after the production-source fix.

**RED, baseline injected binary:** real HTTP loopback streaming request (`include_usage=true`, `max_tokens=2`) in legacy mode: C++ process `exit=-6` (`SIGABRT`), stderr `terminate called after throwing an instance of 'std::bad_alloc'`, Python HTTP client `IncompleteRead(184 bytes read)` and no complete error+`[DONE]` terminal. This failure was reproduced in the new seven-case opt-in test: 6 passed, 1 failed (`mode='legacy'`, exit 1, 21.119 seconds). The same injection on unmodified ready-completion instead returned HTTP 200 with one SSE `server_error` plus `[DONE]`, and the process exited cleanly (0 after normal SIGTERM).

**Second independently reproduced RED, after the first fix:** `legacy_token()` still executed `job->output.push_back(token)` outside any `try`. That first vector growth may allocate before any SSE text is processed. A separate injected CPU-only copy inserted a once-per-process `throw std::bad_alloc()` immediately **before the unique `job->output.push_back(token)` source anchor** (the same effect as `std::vector<int>::push_back` failing its allocation). The opt-in `test_legacy_output_growth_bad_alloc_terminalizes_and_recovers` failed with `IncompleteRead(184 bytes read)`; suite result **6 passed, 1 failed, 1 skipped**, 19.620 s, exit 1. The scratch fault used `HERMES_TEST_FAIL_OUTPUT_PUSH_ONCE=1` and **did not alter** the shared server source. Like the original stream chunk bug, the failing call occurs in the legacy worker thread and can escape to `std::terminate`.

## Narrow product fix

1. Wrap `legacy_token()`'s `job->output.push_back(token)` in a narrow `std::bad_alloc` catch; call `complete(job, 500, kStreamAllocationError, "length")` then return `true`, so the caller closes the same state once and does not advance or reuse an incomplete token sequence.
2. Move `stream_text(job)` inside the existing `legacy_token()` `try`, so any stream JSON, string or queue exception follows the request's existing completion/cancellation cleanup and stops generation without throwing through the scheduler thread.
3. Catch `std::bad_alloc` *before* general `std::exception` in that `try`, using the same existing `complete(...)` path. The immutable error `std::string` is initialized before accepting requests, avoiding an additional error-message allocation in the allocation-failure catch itself; `complete()` is already `noexcept` and has the static emergency `error`+`[DONE]` fallback if its normal terminal build fails.
4. Preserve the previous generic tokenizer exception catch, decoded text/stop logic, usage calculation, successful SSE chunk order, and all other scheduler paths.

The exact scratch diff versus published `server.cpp` was limited to **one new preinitialized error string, moving one `stream_text()` call inside its adjacent `try`, and two narrow `std::bad_alloc` catches (stream chunk and output vector)**. Both demonstrated legacy allocation exceptions are handled before unwinding out of the scheduler thread. It does not claim to harden every potential allocation in the HTTP server or PP2 scheduler.

## Actual CPU-only validation

| Check on exact compiled sources | Result |
| --- | --- |
| Unmodified published source, new focused Unicode/stop/terminal tests | **PASS 6/6** (18.668 s; allocation fault option not present) |
| Original source + scratch-only injected `stream_text` `std::bad_alloc`, new optional gate | **RED as expected:** 6 passed, 1 failed; legacy `IncompleteRead`, process SIGABRT; ready handled failure |
| Intermediate source + scratch-only injected `output.push_back` `std::bad_alloc` | **RED as expected:** 6 passed, 1 failed, 1 skipped; legacy `IncompleteRead` |
| FINAL corrected source + BOTH separately injected allocation faults, new opt-in gate | **GREEN 8/8** (20.576 s); each failure yields one SSE error+[DONE], subsequent request succeeds, state ownership closes once |
| FINAL corrected source, existing `test_tokenizer_bridge_exception_cpu.py` | **PASS 7/7** (6.129 s): terminal event/queue/usage exceptions, auth, fallback and recovery |
| FINAL corrected source, selected existing `test_production_http.py` | **PASS 5/5** (4.631 s): cross-token stop, Unicode stop/prefix, stream usage, concurrent streams and post-SSE decode failure recovery |
| FINAL corrected source, `-fsanitize=address -fno-omit-frame-pointer`, new focused tests | **PASS 6/6**, optional fault fixtures **SKIP 2**, 18.488 s, no ASAN reports |

Normal new-test coverage uses the **real published ByteLevel tokenizer** and `tokenizer_bridge.py` with the real HTTP handler and linked `distributed/pp2_batch.cu`, but substitutes `serving/tests/prod_sched_mock_pp2.cpp` for GPU inference, with `-DHERMES_ADMISSION_CPU_ONLY`. It checks individual raw UTF-8 byte tokens splitting an emoji, Unicode stop crossing byte/token boundaries, withholding then flushing a partial stop prefix, replacement for incomplete UTF-8 on finalization, actual thrown Python tokenizer `stream_push` and `stream_finish` after `stream_start`, concurrent terminal-allocation failure isolation, and concurrent stream cancellation plus unrelated completion. It runs legacy, continuous, and strict ready-completion where applicable. The Python exception wrappers are created only in temporary local directories and import the untouched real bridge.

Build command used (once for the standard binary and once for each dedicated injection variant, changing only the source/output file path):

```bash
scratch=/tmp/h27-worker16-unicode-5c74e211
g++ -std=c++20 -O0 -pthread -DHERMES_ADMISSION_CPU_ONLY -x c++ \
    -I"$scratch" -I/opt/hermes/src/llama-cpp-turboquant/vendor/cpp-httplib \
    -I/opt/hermes/src/llama-cpp-turboquant/vendor \
    "$scratch/server_fixed2.cpp" \
    "$scratch/serving/tests/prod_sched_mock_pp2.cpp" \
    "$scratch/pp2_batch.cu" \
    /opt/hermes/src/llama-cpp-turboquant/vendor/cpp-httplib/httplib.cpp \
    -o "$scratch/build/hermes_server_fixed2"
```

Run the optional negative/positive allocation-failure regressions only with the dedicated **scratch-injected CPU mock binaries**. For the final GREEN gate, set `HERMES_STREAM_CHUNK_FAULT_BINARY` to `.../build/hermes_server_fixed2_stream_fault`, `HERMES_OUTPUT_PUSH_FAULT_BINARY` to `.../build/hermes_server_fixed2_output_fault`, and `--server` to the ordinary non-injected `.../build/hermes_server_fixed2`. For source-anchored RED, use baseline `.../build/hermes_server_stream_fault` for the stream fault and intermediate `.../build/hermes_server_output_fault_red` for vector growth. Other required arguments follow the existing `prod_sched_state_test.py` (`--model .../fake.gguf`, `--gguf-dump .../gguf_dump_fake.py`, `--tokenizer` real tokenizer fixture, `--python` existing CPU Python venv). Each optional test is skipped when its environment variable is unset. **Never target a production server binary with these tests.**

## Remaining bounds / release decision

Both demonstrated legacy allocation exceptions (`output.push_back` and `stream_text`) are fixed in the shared Windows file with separate CPU RED→GREEN witnesses. Physical GPU inference, full-model numerical parity, exclusive device ownership, ingress release, real host-OOM recovery, and throughput were **not** tested or claimed; HERMES-27 remains **NO-GO** for production admission until those independent gates pass. The prime must reconcile latest upstream HEAD and review/publish this local change under its Git ownership. Other potential scheduler-thread exceptions outside the two demonstrated `legacy_token()` boundaries require their own evidence and ownership; they are not implicitly repaired by this patch.
