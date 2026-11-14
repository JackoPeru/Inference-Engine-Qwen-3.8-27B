# HERMES-27 — legacy HTTP/SSE allocation-failure containment integration

**2026-11-14. Two real host-side exception escape paths repaired; CUDA compilation + isolated CPU HTTP checks PASS. Full-model/GPU and production gate NO-GO.**

## Provenance and scope

The source was reconstructed from immutable Git commit
`8a829136aeefdf68bb964ed9c39fc5a2d007d154`
under `/tmp/h27-sse-gdn-coherent-20261114/src/hermes-27`, then
overlaid with the **frozen** worker16 sources:

- Original published `serving/server.cpp` SHA256:
  `ad2522dc7297bb9e3826a617778ec580e591d917e6bde3796adc3146e596a7f4`
- Corrected `serving/server.cpp` SHA256:
  `12340ecd1d649799304328e7f8929ae35e43bd8010279c0bf265df86c7709b28`
- New `serving/tests/test_http_unicode_stop_terminal_cpu.py` SHA256:
  `aa4694a617920b72b597ad676f62879b36a53fb367402d5d664ec801e96eae82`
- Worker evidence `reports/HTTP_UNICODE_SSE_TERMINAL_NEXT.md` SHA256:
  `9d26a85eb165f77fc494026a87a651231ac683d0243c62033a97d2faaef436e2`

The reference archive includes the already-published CUDA prefill-pool
correction and the current PP2 scheduler. Neither was changed for the
HTTP fix.

## The two reproduced defects

In the production-source `legacy_token()`, a fallible JSON/stream chunk
allocation inside `stream_text(job)` occurred *after* the existing
try/catch. Separately, `job->output.push_back(token)` could allocate
*before* that try/catch. Each was independently proven on the real
CPU-only HTTP server linked with fake PP2. Separate one-shot `bad_alloc`
fault injections into disposable source copies produced process
`SIGABRT (-6)` and truncated SSE without `[DONE]`, not a clean
request error.

The correction guards `output.push_back`, moves `stream_text` inside
the existing try, handles `std::bad_alloc` using a preinitialized error
message, and invokes the existing `noexcept complete()`/emergency
SSE terminalization path. The scheduler can close the *same* PP2 state,
without duplicate seed, second scheduler or unhandled host-thread throw.

The owner's controlled RED→GREEN fixture then passed **8/8** with
both injections enabled separately: one error event, one terminal
`[DONE]`, no false success, and a successful next request. Existing
tokenizer terminal **7/7** and selected HTTP Unicode/stop/usage **5/5**
also passed. Those tests are CPU fake-inference, not CUDA execution.

## Fresh integration from latest production-source snapshot

On the immutable-Git snapshot plus this final fix:

- Configured CUDA **12.8.93**, `HERMES_REQUIRE_SERVER=ON`,
  `HERMES_REGISTER_GPU_TESTS=OFF`.
- Full production `hermes_server` compilation, **CUDA device link PASS**.
  Scratch binary SHA256
  `c39c1c9d7a350f77576e82d455d362a5674377f65bef3d14716bb1eafbe6f8a6`.
- Built the same real `server.cpp`, real PP2 host scheduler and cpp-httplib
  against the test-only mock PP2 backend with
  `HERMES_ADMISSION_CPU_ONLY`. Host fake server SHA256
  `7d9a446b9b50f1d25eed98f422b79bfc959bb487afdc2e3e71aa2071e2a10d3a`.
- Executed new real-loopback HTTP Unicode/stop/cancellation tests on
  that fresh coherent fake server: **6 PASS, 2 SKIP**, no failures.
  The two skips are deliberate, because their special fault-injected
  scratch binaries were already exercised separately by the owner.
  Both successful and failing requests are bounded to temporary
  `127.0.0.1` listeners, with mock logits only; no installed endpoint.

The first local attempt to run those host tests incorrectly placed the
fake binary one directory too high; the test expects its
`serving/tokenizer_bridge.py` relative to the binary's ancestor source
root and raised `FileNotFoundError` before running any tests.
Moving the scratch fake binary into
`src/hermes-27/build-host/` corrected this test-fixture placement.
The actual rerun passed all six runnable HTTP cases.

## Safety boundary

There was **no real GGUF upload, GPU context, service restart, Tailscale
change or production cutover**. Allocation-failure testing proves the
specific host exception boundaries, not arbitrary system OOM resilience.
The independent 248,320-logit **12/12 full-model FAIL**, real N1/N2/N4/N8
GPU performance and physically authenticated all-ingress/GPU fencing
remain blocking. **Release verdict NO-GO.**
