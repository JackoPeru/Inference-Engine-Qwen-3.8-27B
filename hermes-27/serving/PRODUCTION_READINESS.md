# HERMES-27 Serving — Production Readiness (CPU audit, 2026-11-13)

## HTTP contract

- Routes: POST /v1/chat/completions, GET /v1/models, GET /health, GET /metrics, and custom POST /v1/cancel.
- JSON requests require Content-Type application/json, optionally with charset parameters. Wrong media type returns HTTP 415 with error code unsupported_media_type.
- One configured --model-id; mismatched ID returns HTTP 404 model_not_found. With one served model the model field may be omitted.
- Messages accept plain-text content and arrays of text parts only. Roles: system, developer (mapped to system in the Qwen template), user, assistant. Optional message name is a string and is discarded when rendering. Non-text media and tool roles are rejected.
- Greedy sampling only: temperature=0, top_p=1, n=1, presence_penalty=0, frequency_penalty=0. Other values return HTTP 400 structured errors. Optional null values retain defaults.
- max_tokens or max_completion_tokens must be 1..context. Exactly one may be non-null. Default context 128; --context supports up to 8192 tokens.
- stop accepts null, a nonempty string, or 1..4 nonempty UTF-8 strings each up to 256 bytes. Matches across tokens and Unicode; SSE withholds partial stop prefixes. Stop-triggering tokens count toward completion usage. A nonmatching partial prefix at generation end is flushed normally.
- Streaming has initial assistant-role chunk, content deltas, final finish_reason, and [DONE]. stream_options.include_usage adds one trailing choices-empty usage chunk. SSE/nonstream text parity is checked against the actual Rust tokenizer.
- Invalid request parameters return OpenAI-shaped error objects with type, code and param. Incorrect model IDs, overflow integer n, invalid JSON, and unsupported content are handled explicitly.

## Limits, recovery and security

- Bind defaults to 127.0.0.1. A non-loopback bind requires an API key. Use --api-key-file with an owner-only regular file (mode 0600) to avoid exposing secrets through process arguments. Existing --api-key also works, but both flags cannot be combined.
- No TLS is provided. External exposure should use a secured TLS reverse proxy. No wildcard CORS is configured.
- HTTP worker pool: 64 concurrent threads plus at most 64 queued tasks. Scheduler defaults: 4 active / 32 queued requests, 60-second queue deadline. VRAM admission checks conservative GPU0/GPU1 capacity then backend allocation validates actual limits.
- Request bodies capped at 1 MiB; HTTP read/write operations timeout after 15s. Per-SSE-client unsent backlog capped at 256 chunks or 1 MiB. Slow-reader overflow cancels the job and increments hermes_stream_backpressure_total.
- Tokenizer I/O has a 15-second read timeout. On subprocess failure the bridge is terminated/restarted and safe encode/decode operations are retried once. Existing incremental stream states cannot be restored and fail closed; subsequent requests continue after restart.
- SIGINT/SIGTERM joins the scheduler before the HTTP listener, preventing long-lived completion handlers from stalling shutdown. Production logs include token counts and status, but no generated token IDs.

## CPU-only validation commands

The test binary in the commands below was compiled from real serving/server.cpp, cpp-httplib and an isolated CPU mock PP2 implementation. It cannot use GPU; scratch files live at /tmp/hermes27-worker3-serving-check.

The server now references worker-6's ContinuousScheduler even when the
legacy scheduler remains selected. Link the **real** pp2_batch.cu in
CPU-only mode, using the NEW prod_sched_mock_pp2.cpp (not the older
mock_pp2_cpu.cpp, which defines a duplicate admit_check symbol).
The one-command runner copies all sources to a fresh isolated scratch
directory, builds this link configuration, and runs both HTTP backends:

    bash serving/tests/prod_sched_cpu_gate.sh

The runner defaults to /tmp/hermes27-prod-sched-cpu and accepts
HERMES_CPU_SCRATCH, HERMES_CPP_HTTPLIB_ROOT, HERMES_TEST_TOKENIZER and
HERMES_TEST_PYTHON environment overrides. Its fake model file is empty
and its mock GGUF template prints a one-token prompt. It never executes
the CUDA PP2 adapter or actual model upload.

For individual test execution, substitute the runner's scratch directory
in the commands below. Production tests can run directly against the
scratch executable (legacy) or through
serving/tests/prod_sched_launcher.py (continuous opt-in; requires
HERMES_TEST_SERVER_BIN set to the scratch executable).

Tokenizer, with real Qwen tokenizer:

    HERMES_TEST_TOKENIZER=/opt/hermes/models/Qwen3.8-27B-Uncensored-EXL3-3.07bpw/tokenizer.json /opt/hermes/runtimes/exllamav3-venv/bin/python serving/tests/test_bytelevel_stream.py -v

HTTP production contract:

    /opt/hermes/runtimes/exllamav3-venv/bin/python serving/tests/test_production_http.py --server /tmp/hermes27-worker3-serving-check/build/hermes_server --model /tmp/hermes27-worker3-serving-check/fake.gguf --gguf-dump /tmp/hermes27-worker3-serving-check/gguf_dump_fake.py --tokenizer /opt/hermes/models/Qwen3.8-27B-Uncensored-EXL3-3.07bpw/tokenizer.json --python /opt/hermes/runtimes/exllamav3-venv/bin/python

Run the long slow-reader test with --slow-reader if the CPU mock honors FAKE_STEP_MS=0.

HTTP SSE Unicode parity:

    /opt/hermes/runtimes/exllamav3-venv/bin/python serving/tests/test_stream_mock_http.py --server /tmp/hermes27-worker3-serving-check/build/hermes_server --model /tmp/hermes27-worker3-serving-check/fake.gguf --gguf-dump /tmp/hermes27-worker3-serving-check/gguf_dump_fake.py --tokenizer /opt/hermes/models/Qwen3.8-27B-Uncensored-EXL3-3.07bpw/tokenizer.json --python /opt/hermes/runtimes/exllamav3-venv/bin/python

The serving/smoke.py HTTP smoke also validates mixed SSE/nonstream, cancellation, invalid inputs, and text parity.

### Final observed CPU results

- HTTP production-contract unittest: **19 passed / 19 run**, with --slow-reader explicitly enabled (29.414 seconds for the full suite).
- Tokenizer unittest: **5 passed / 5 run** with the real Qwen tokenizer, including 1,200 randomized token-ID parity sequences.
- Standalone SSE HTTP Unicode/ByteLevel parity script: **9 passed / 9 scenarios**.
- serving/smoke.py: **M3 smoke PASS**, including concurrent nonstream/SSE, full-text parity, and explicit cancellation.
- Native g++ linking of the current serving/server.cpp against the CPU mock PP2 and installed cpp-httplib: **PASS**.

Tests ran against separate localhost ephemeral ports. No production Hermes service was started, stopped, or modified. No GPU workloads were performed.

## Remaining limitations

- CPU-only tests cannot establish real GPU correctness, throughput, VRAM occupancy or multiuser CUDA safety. Do not claim production GPU readiness until prime builds and tests the integrated snapshot with a permitted isolated GPU slot.
- Host-threaded per-request decode is not fused GPU batching. Chunked prefill, broad sampling, tool calls, multimodal input and native 262K context are not implemented by the serving API.
- Python tokenizer pipe remains serialized. Unsupported decoder types use final-only SSE fallback, or full-prefix decodes for stop detection. The streaming backpressure cancellation and finish_reason cancelled are Hermes extensions.
- Once SSE headers have been sent, later generation errors appear as SSE error events. Nonstream clients disconnecting without explicit cancellation can continue occupying scheduler resources.
- The API has neither native TLS nor per-user rate limits. A correctly configured TLS/authenticated proxy is required for public network deployment.
- Before the real-backend launch, prime must rebuild gguf_dump using the integrated model/parser source. An old gguf_dump binary may have stale metadata/chat-template semantics.

## Experimental opt-in: worker-6 ContinuousScheduler (2026-11-13)

Enable with --continuous-scheduler when starting a **new, isolated** server
instance. There is no runtime switch. The preexisting host-threaded
Scheduler remains the default, with no changes to its queue/decode loop.

The opt-in path enqueues each HTTP request exactly once into
pp2batch::ContinuousScheduler and never places it in the legacy
queue_/active_ structures. The outer jobs_ map only tracks the HTTP
response lifetime; it is not a second work queue. PP2StepBackend owns the
sole DecodeState for each admitted request; the HTTP Job's unused legacy
DecodeState is never initialized, stepped or freed.

Worker-6's EventKind::admitted occurs **after** PP2 state allocation
and before the first token event. The callback initializes per-request
ByteLevel tokenizer state and only then releases the HTTP 200/SSE headers.
A failed state open therefore returns HTTP 503 (or HTTP 429 for a
specifically identified live VRAM preflight rejection) before SSE.
Expired waiting requests return HTTP 429 without any state allocation.

Token events feed the existing UTF-8-safe incremental decoder, stop
sequence filter, and bounded SSE buffer. A text stop or slow-client
backpressure invokes ContinuousScheduler::cancel, and the backend frees
its state. Terminal events compare exact generated ID lists, finalize
the reference tokenizer parity, populate nonstream/SSE usage and finish
reason, and wake HTTP waiters. Client cancellation and SIGTERM follow
the same callback path. The single scheduler pump calls tick(), which
performs at most max_active sequential PP2 steps per turn; it is **not**
fused GPU batching and no speedup is claimed.

### Verified isolated CPU gates for the opt-in mode

- Pre-integration legacy baseline: 19/19 existing HTTP tests PASS.
- Integrated source, legacy default: **19/19 existing HTTP tests PASS**.
- Integrated source, --continuous-scheduler: **19/19 HTTP tests PASS**.
- Continuous opt-in, NEW prod_sched_state_test.py: **7/7 PASS** with
  FAKE_TRACE_FILE checks proving exactly one PP2 state open and close
  per admitted request, matching decode-step/token counts, no state
  allocation on queue timeout/open failure, correct stop/cancel,
  6 concurrent states and cleanup on SIGTERM.
- Continuous opt-in, existing SSE/Unicode parity regression: **9/9 PASS**
  (ASCII, CJK, accents, emoji and split/invalid UTF-8 byte sequences).
- Existing independent real-tokenizer tests: 5/5 PASS.

Build used g++ -std=c++20 -DHERMES_ADMISSION_CPU_ONLY with the real
distributed/pp2_batch.cu, and serving/tests/prod_sched_mock_pp2.cpp.
No GPUs, services or CMake files were touched. Worker-6's
EventKind::admitted API **must** be present in both the deployed
pp2_batch.hpp and pp2_batch.cu. Prime must rebuild the integrated
server and gguf_dump from one consistent source snapshot before
authorizing GPU tests.

**Not yet proven:** real GPU multiuser correctness, VRAM under real
Qwen weights, throughput, response latency, real CUDA cancellation
or concurrent backend safety. Until GPU gate and production soak are
completed, leave --continuous-scheduler disabled by default.
