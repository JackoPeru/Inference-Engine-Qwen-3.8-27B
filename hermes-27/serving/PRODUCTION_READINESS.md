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
- Host-threaded per-request decode is not fused GPU batching. The experimental --batched-prefill seeds a fresh state for T<=64 only; general chunked prefill, broad sampling, tool calls, multimodal input and native 262K context are not implemented by the serving API.
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

## Experimental raw-GGUF stateful prefill (2026-11-13; CPU-only gate)

Enable `--batched-prefill` explicitly on a new isolated server instance.
The option is independent of `--continuous-scheduler`: both the default
legacy Scheduler and the ContinuousScheduler support it. Neither is
enabled automatically and BOTH flags can be omitted for the fully
unchanged legacy default. When the input prompt has **1..64 tokenizer
tokens**, the server asks PP2 to seed a freshly allocated DecodeState
using `pp2::prefill_into_decode_state(ids, state, stats, error)`.
For **T>=65**, each scheduler uses the original token-by-token prompt
loop; it never truncates, chunks or seeds half a prompt.

### Exactly one state owner and first-token semantics

- Legacy: after `decode_init(capacity = prompt_len + max_new_tokens - 1)`,
  `seed_legacy_prompt` consumes the whole prompt in a SINGLE stateful
  prefill. From its `T*VOC` FP32 logits, only the **last row** determines
  the first generated token by greedy argmax. The HTTP Job advances its
  `prompt_pos` to `T`, processes that generated token through the same
  existing incremental decoder/stop/SSE/usage routine as the legacy path,
  then resumes `decode_step` at absolute position **T**, with the
  generated token as input. There are zero prompt replay decode steps.
- Continuous: worker-6 extended StepBackend with opt-in
  `prefill(state, prompt, last_logits, error) -> PrefillResult` and
  `SchedulerConfig.enable_batched_prefill=false` by default.
  `PP2StepBackend` owns the real state and calls the exact same raw
  PP2 seeded API, returning only the last VOC row; the scheduler's
  `promote` performs the prefill on the SAME opened state before the
  `admitted` event, emits the first completion token once, and never
  steps prompt positions again. Token/terminal callbacks are unchanged
  and retain Unicode, usage, SSE and stop semantics.
- Cancellation is checked after the potentially non-interruptible
  seeded operation; a cancelled job is closed and does not generate
  another step. Failed/invalid logits must fail closed and must never
  retry token-by-token on an already seeded/poisoned state. The PP2
  implementation itself may have called `decode_free` on failure; the
  serving legacy path consults `DecodeState.inited`, while the worker-6
  adapter calls `decode_free` idempotently before deleting its host
  state. No second GPU state or duplicate job queue is created.
- `HERMES_PP2_PREFILL_BF16=1` is a **NO-GO**: with
  `--batched-prefill`, server startup rejects it with exit code 2,
  before model upload. Raw GGUF computation is the only permitted
  seeded path. The BF16 full-logit quality gate in the PP2 source is
  NOT relaxed or bypassed.

### Counters and reproducible CPU validation

Prometheus `/metrics` includes
`hermes_prefill_attempts_total`, `hermes_prefill_seeded_total`,
`hermes_prefill_failed_total`, and
`hermes_prefill_prompt_tokens_total`, with the same names for legacy
and ContinuousScheduler. These count actual seeded requests/tokens,
not theoretical or estimated speedups. T>=65 fallback and default
flag-off requests leave these counts at zero.

Run `bash serving/tests/prod_sched_cpu_gate.sh` from the project root.
It now compiles real `serving/server.cpp` + current worker-6
`distributed/pp2_batch.cu` with `-DHERMES_ADMISSION_CPU_ONLY`
and runs `serving/tests/prod_sched_prefill_test.py` alongside the
earlier HTTP/tokenizer/SSE suites. The seeded tests exercise T=1,
T=11, T=64, fallback T=65, default-off parity, concurrent mixed-context
N6 with four active states, EOS/text stop, Unicode SSE, usage,
cancellation, invalid logits, injected prefill failure, context
rejection, and BF16 fail-closed startup. Instrumented fake PP2 traces
require one `open` and one `close` per job, exactly one `prefill T`
when eligible, and only continuation `step` calls at positions >=T.

**Do not infer production throughput, TTFT or numerical quality from
these CPU mocks.** The raw PP2 stateful prefill's isolated numerical
parity tests are separate from the HTTP integration gate. Prime must
run authorized exclusive-GPU N1/N2/N4 throughput and correctness
tests with the same integrated source snapshot, including VRAM return
to baseline and cancellation under real kernel load. Until that
happens keep both experimental flags off in production.

## Experimental legacy-only T65..8192 RAW chunked stateful HTTP prefill

The new --chunked-prefill flag is a separate **non-default** diagnostic
HTTP route. It implies --batched-prefill: T1..T64 uses the prior
one-shot RAW stateful API, while T65..T8192 calls the existing
PP2 prefill_chunks_into_decode_state once on the **same untouched
DecodeState**, using at most 64 input tokens per chunk. It returns
one final-token 248320-logit row. The legacy scheduler samples the
first completion token from that row and continues at absolute T.
It never replays seeded prompt positions or allocates a second state.

Startup fails with exit code 2 before model upload unless all conditions
are satisfied: legacy scheduler only, --active 1, exact
HERMES_PP2_CHUNKED_PREFILL=1, async streams disabled, neither packed4
GEMV nor HERMES_L26 debug enabled, and BF16 prefill disabled. This
deliberately prevents unverified cross-request default-stream overlap.
It does not improve multiuser concurrency. Normal --batched-prefill
without this new flag retains the prior T65+ stepwise fallback;
neither flag set retains the full legacy default.

Partial chunk failure and nonfinite last logits terminate the request:
the potentially poisoned state is never replayed or reused.
At T8192 and max_new_tokens=1 the first completion can be sampled
from the last prompt row without decoding position 8192. Continuation
past 8192 is unavailable until separately proven KV eviction.

See reports/HTTP_CHUNKED_PREFILL_CPU_GATE_2026-11-14.md for CPU/mock
test evidence and native CUDA compilation. Prior standalone T129/T1024
same-engine state/logit parity is not an HTTP GPU integration test.
No throughput, independent llama fidelity or production readiness
is established by CPU mocks. A GPU-exclusive quality gate and
trusted ingress drain are still mandatory. Production remains NO-GO.

## Experimental ready-completion HTTP scheduler (default OFF)

The strictly opt-in `--ready-completion` flag is a separate mode of the
existing single `ContinuousScheduler`, not a new request scheduler or
a replacement for the legacy HTTP route. It requires both
`--continuous-scheduler` and `--parallel-steps`, together with the exact
`HERMES_PP2_ASYNC_STREAMS=1` backend capability. Batched and chunked
prefill are explicitly excluded. Unsupported combinations must fail
before uploading any model weights.

The intended improvement is narrower than full GPU continuous batching:
the scheduler can commit a completed request's result after joining that
request's worker thread, instead of waiting for every request in a
host-side wave. Each PP2 DecodeState retains a single owner, separate
CUDA streams, KV/GDN/conv state and request-private transfer buffers.
Cancellation and shutdown must not release a state until its in-flight
backend call returns and the relevant streams have drained.

`serving/tests/prod_sched_ready_test.py` is the scoped CPU/mock HTTP
integration contract for startup rejection, five independent request
states, SSE/non-stream output parity, Unicode/token usage, and no prompt
replay. Independent C++ scheduler tests must also prove fast B completion
while unrelated slow A remains blocked, old-wave behavior unchanged,
finite logits, exact step positions, callback cancellation and no early
GPU-state free. CPU checks do **not** measure actual GPU overlap or N4/N8
throughput. Synchronous tokenizer IPC remains a potential bottleneck.

No production default changes are justified without real, matching
GGUF+binary CUDA N1/N2/N4/N8 measurements, p50/p95 TTFT/TPOT, full
248320-logit parity, state isolation/rollback, VRAM, timeouts, shutdown,
SSE byte/usage parity and repeatable soak. Compare directly to existing
legacy async and wave-parallel under identical prompts/concurrency;
reject the opt-in if it does not materially improve aggregate throughput.
The previously measured +0.656% at N4 and -0.038% at N8 are **not**
evidence of successful scaling. The full independent llama numerical gate
remains FAIL and production remains NO-GO.

## Strict experimental rolling-KV HTTP continuation past T8192

`--sliding-kv` is a separately gated legacy-serving integration of
`HERMES_PP2_SLIDING_KV=1`. The caller must also specify
`--chunked-prefill --active 1 --context 8192` and
`HERMES_PP2_CHUNKED_PREFILL=1`; continuous/parallel/ready scheduling,
async stream opt-ins, packed4/BF16 prefill and layer26 debug modes are
excluded. An environment flag without the explicit CLI flag and unsafe
combinations are rejected at startup, before model upload. Defaults are
unchanged.

In this mode, a request whose prompt has at most 8192 tokens allocates a
physical **8192-token** KV cache and may request further generated tokens
up to the separate conservative absolute-position bound 2^20. The last
prompt logits yield the first generated token without replay; each further
token enters the **same** PP2 DecodeState at its absolute position.
The PP2 opt-in ring reads the last 8192 chronological K/V slots, including
the staged current K/V, before committing eviction. GDN and convolution
state remain persistent; CUDA failure poisons the state rather than replaying
or rolling back partially modified recurrence buffers.

`serving/tests/prod_sched_sliding_test.py` uses the real HTTP server and a
CPU-only fake PP2 backend to check T8192+completion SSE/non-stream,
absolute step positions, exactly one initialization/chunked seed and close,
short-prompt fixed capacity, default boundary rejection and unsafe-startup
rejection. It does **not** execute CUDA attention or compare logits against
an independent model. Ring attention is **windowed**, not mathematically
equivalent to unrestricted full-history attention after the boundary.

The GPU implementation requires genuine 8192/8193/16384-token generation
and a matching independent rolling-window oracle, full-logit/GDN/conv/KV
checks, GPU VRAM/time/fairness and cancellation/OOM tests before readiness.
It currently serializes two stream completions per full-attention layer
after wrap, so latency may be worse. The legacy active=1 restriction means
this is not a multiuser throughput optimization. It remains an explicitly
disabled experimental path and cannot be used to justify production GO.
