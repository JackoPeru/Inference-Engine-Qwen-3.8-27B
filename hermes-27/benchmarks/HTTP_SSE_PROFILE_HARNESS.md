# HERMES-27 — HTTP SSE profiling harness (staging only)

**CPU/OFFLINE implementation only. No real HTTP server or GPU was accessed.**
Prime-reported baseline HEAD `91e3841`; context8192 files were staged
uncommitted under another owner. The local Windows source copy lacks a
Git checkout, so this worker did not independently verify HEAD or touch Git.
Production/GPU performance claims: **NONE**.

## Exact owned NEW files

- `scripts/scratch_http_sse_profile.py` — bounded, opt-in streaming profiler
- `scripts/tests/test_scratch_http_sse_profile.py` — injected offline tests
- `benchmarks/HTTP_SSE_PROFILE_HARNESS.md` — this report

No existing soak, scheduler, PP2, serving, CMake, ops, GPU service or
network code was modified. No GPU allocation, model load, listening
socket, service start/stop, credentials, systemctl or sudo.

## Exact, locked-down target

The profiler uses the Python standard library `http.client.HTTPConnection`
directly to **literal `127.0.0.1:18127` only**. No user-configurable
URL/host/port, DNS lookups, HTTP_PROXY environment routing, redirect
opener or external URLs. Non-200 responses including 3xx redirects
are errors, and are **never followed**. Only these endpoints exist:

| HTTP method | Path | Trigger |
| --- | --- | --- |
| POST | `/v1/chat/completions` | Explicit `--run` only |
| GET | `/metrics` | Explicit `--run --metrics-vram` only |

**Default safe invocation (no network):**

```bash
python3 scripts/scratch_http_sse_profile.py
```

This prints `DRY_RUN_NO_NETWORK` with N1/N2/N4/N8, 2 rounds and
30 planned requests. `--run` is required for any connection. The script
does not create output files; future authorized operators may redirect
its JSON to a new, deliberately chosen artifact.

**Future run recipe only — NOT executed in this task.** Requires prime's
exclusive scratch/GPU/port authorization and an independently started
server on the exact localhost port:

```bash
python3 scripts/scratch_http_sse_profile.py \
  --run --clients 1 2 4 8 --rounds 2 \
  --max-tokens 32 --timeout 120 \
  --max-sse-bytes 4194304 --run-label context8192 \
  --metrics-vram
```

The script does not start or stop a server, connect to an installed
production endpoint, or request GPU ownership. Validate external
server/provenance separately before using `--run`.

## Time measurements and semantic correctness

An HTTP response header or SSE role-only delta does **not** count as
generated content. The first nonempty string in
`choices[0].delta.content` or `choices[0].delta.reasoning_content` is
the first client-observable **semantic generated-output event**. This
includes generated reasoning if the model streams reasoning before
visible final content. The tool measures monotonic time from immediately
before the HTTP request to the read containing that event.

| Field | Meaning | Hard limitation |
| --- | --- | --- |
| `ttfc_ms` | Time to First (semantic) Content, including client/server/network prefill delay | **Not verified model TTFT**: model may generate invisible tokens or buffer text |
| `semantic_fragment_gaps_ms` | Adjacent semantic SSE fragment arrival differences | **Not TPOT**: fragments may represent 0, 1 or many model tokens |
| `semantic_fragment_gap_ms` | Per-request p50/p95 of those gaps | Zero gaps when separate events coalesce in a single socket read |
| `ttft_ms` and `tpot_ms` | Always JSON null | True token boundary / GPU generation timestamps are unavailable |
| `completion_tokens` | Strict integer from final `usage.completion_tokens` | Server-reported; never inferred from chunk count |
| `wall_completion_ms` | Request start through EOF after finish, usage and [DONE] | Includes queuing, prefill, generation and transport |
| `completion_tokens_per_wall_second` | Server-reported completion tokens / complete HTTP wall seconds | Gross wall-rate, not decode-only rate |

**Do not call TTFC true TTFT or SSE inter-chunk timing token TPOT.**
Even when final usage says 4 tokens and there are 4 SSE events, it
does not establish a one-token-per-event relationship. The profiler
intentionally does **not invent TPOT**. The first semantic event is
an upper bound on the client's observation of generated content,
not an authenticated GPU first-token timing.

Timing is sampled when `HTTPResponse.read1` yields bytes. This is
preferable to waiting for a full `read(4096)` when stream chunks are
small. UTF-8 sequences may be split across reads, and multiple SSE
events can arrive together: every event from the same read has the
same measured timestamp. Reported `coalesced_adjacent_semantic_events`
and zero gaps signal this uncertainty, **not zero model TPOT**.

The SHA-256 output fingerprint hashes concatenated final-text and
reasoning channel contents, independent of SSE chunk boundaries.
No raw generated text or prompt is included in output JSON.

## Protocol correctness and bounded failure handling

Each OpenAI `chat.completions` request uses the same fixed prompt,
`temperature=0`, `stream=true`, `max_tokens` from the CLI, and
`stream_options.include_usage=true`. A stable request correlation
header `X-Profile-Request-ID` is constructed as
`h27-{run_label}-n{N}-r{round:02d}-q{case:02d}`.
It is stable across repeat runs with the same settings. It is **not**
a cryptographic server acknowledgement; completion `id`, if returned,
must remain consistent across SSE events.

The byte-oriented parser supports fragmented UTF-8, HTTP-dechunked
OpenAI SSE, multiline `data:`, comment/keepalive lines, CRLF/LF, role
events, content and reasoning events, and usage-only terminal events.
It requires a valid 200 `text/event-stream` response, a valid final
`finish_reason` (stop/length), positive final completion-token usage,
consistent usage totals if supplied, and a single `[DONE]` before EOF.

Fail-closed errors include missing/duplicate usage/finish/DONE,
truncation, JSON errors, unexpected SSE types, unsupported tool/audio
generation, inconsistent completion IDs, nontext deltas, invalid UTF-8,
redirects, HTTP errors, timeouts, cancellation and buffer overflow.
Errors have sanitized codes; server error bodies/prompts are never
printed. Failed requests have **no synthetic completion token rate**.

All bounds are explicit: concurrency levels subset N1/N2/N4/N8;
1–16 rounds; 1–4096 max completion tokens; 1–1200-second
per-request timeout; 1 KiB–64 MiB max SSE bytes; 1–64 KiB
socket read; max individual SSE event 128 KiB and line 64 KiB;
optional /metrics text max 256 KiB and 64 VRAM-like series.
Cancellation is cooperative at transport read boundaries, not proof
the server cancelled computation. A blocked socket read is subject to
the configured timeout and remaining monotonic budget.

## N1/N2/N4/N8 and statistical semantics

For each requested concurrency, a barrier releases N worker threads
at approximately the same time. Each opens a **new independent direct
HTTP connection**; keepalive reuse is not assumed. Default is
two rounds of each N1/N2/N4/N8 (30 attempted requests).
Identical input and decoding settings improve comparability but
cannot guarantee deterministic backend scheduling or equivalent
hardware work; separately record server build/model/engine variant.

The JSON report contains complete per-request records, round reports,
by-concurrency pooled reports, and an overall diagnostic. For each
group, success-only samples produce **p50/p95 of TTFC, HTTP wall
latency, per-request wall completion tokens/s, and semantic SSE gaps**.
Percentiles use type-7 linear interpolation. Empty distributions
yield null, never a fabricated zero.

Aggregate completion tokens/s is:

```text
sum(final usage.completion_tokens) / measured concurrency-round wall seconds
```

For multiple rounds, add round wall intervals, **not** concurrently
overlapping per-request latencies. If any request fails, the affected
group (and overall) aggregate rate is null with `INCOMPLETE`;
successful request observations remain visible but never misrepresented
as full N-client throughput. The `overall` mixed-N statistic is a
diagnostic; compare the **same N** across server variants.

## Optional /metrics VRAM

With `--metrics-vram`, the profiler separately samples GET `/metrics`
before and after each round on the same fixed localhost origin.
Recognized raw numeric gauge names contain `vram` or GPU-memory
used/allocated/reserved wording. Units are marked
`bytes_from_metric_name` only for names ending `_bytes`;
otherwise `unknown_raw_metric_unit`. The tool neither converts
unknown units nor claims actual hardware ownership/VRAM from metrics.
Absent metrics, non-200 responses or redirects appear as separate
diagnostics, **not** invented zero GPU memory or token rates.

## Offline validation

```bash
python3 -m unittest discover \
  -s scripts/tests -p 'test_scratch_http_sse_profile.py' -v
python3 -m py_compile \
  scripts/scratch_http_sse_profile.py \
  scripts/tests/test_scratch_http_sse_profile.py
python3 scripts/scratch_http_sse_profile.py
```

The tests use injected FakeHTTPConnection and deterministic monotonic
clocks, including a complete mocked 30-request workload. Tests do not
create a socket or touch any real service/GPU. Passing them validates
**only the profiler's offline measurement/format contract**, not
the speed of HERMES-27 and not an actual token-level TTFT/TPOT.

**Status: staging instrumentation ready for independent review;
real-world GPU/HTTP benchmarking not authorized or performed.**
