# HERMES-27: HTTP soak harness N4/N8 — offline deliverable

**Status:** Python CPU mock tests PASS, real HTTP/GPU soak NOT RUN. Prime controls GPU exclusivity. No model upload, CUDA allocation, service management, live HTTP request, or modification of existing PP2, serving, CMake and benchmark scripts was performed.

## Files owned

- scripts/scratch_http_soak.py — independent bounded HTTP client; reads a preexisting loopback scratch server only when invoked.
- scripts/tests/test_scratch_http_soak.py — injected urllib responses + fake monotonic time, no actual network.
- benchmarks/HTTP_SOAK_HARNESS.md — this operational contract and verification.

## Access contract and strict preflight

The **only** destination is the literal http://127.0.0.1:18127, optionally with one trailing slash. Alternate hosts (including localhost), addresses, ports, paths, URL userinfo, HTTPS, query strings and fragments fail before contacting anything. The real urllib opener explicitly uses ProxyHandler({}) (ignores HTTP_PROXY/HTTPS_PROXY/ALL_PROXY) and denies redirects; it also rejects any response whose effective URL differs from the requested URL. No subprocesses or GPU APIs are imported/called.

Before the first inference POST, GET /health must return HTTP 200 JSON status=ok ready=true, GET /v1/models must advertise hermes-27, and GET /metrics must expose ready=1, active=0, queued=0, cumulative counters and GPU0+GPU1 VRAM gauges. Missing/duplicated, negative, noninteger or nonfinite required gauges/counters fail closed.

The hostname/model/health checks **do not attest process identity**: the prime-owned isolated scratch runner must prove port 18127 belongs to the intended scratch binary and preserve live Tabby / GPU Manager / ingress leases. Do not run this against an unrelated local process using the same port.

If the scratch server requires a bearer token, use --api-key-env EXISTING_VAR_NAME. Only the specified environment variable is read and the secret is never printed or serialized. No credential file is created.

## Bounded workload

Default: N4 and N8, five rounds each, at most eight host request threads, at most eight model completion tokens per request: **60 requests** total. Limits: clients chosen uniquely from 4/8; 2–12 rounds; output max 1–64; each HTTP timeout 1–120 seconds; drain timeout 0.1–60 seconds and poll interval 0.01–2 seconds. A failure aborts subsequent rounds; current in-flight requests are bounded by their individual timeouts.

Two prompt styles: arithmetic asks for 2+2; explain asks for short technical statements. A given N/client slot receives the **same exact prompt each round**: no round number. Every call uses model hermes-27, temperature zero, reasoning effort none and stream=false. This checks structural validity and deterministic repeated output; it does not claim independent mathematical or llama.cpp numerical verification.

Each response must be HTTP 200 application/json, with no error object, object=chat.completion, the expected model, nonempty ID, exactly one choice at index zero, role assistant, nonempty text, and finish_reason exclusively stop/length. Prompt, completion and total usage must be positive integer counts, never boolean, total=prompt+completion, and completion cannot exceed requested maximum. Cancelled responses, malformed data and HTTP errors fail.

Each case reports two SHA-256 values: full UTF-8 assistant content, plus canonical JSON over content, finish_reason and all three usage counts. The **canonical checksum is compared between rounds** for the same (N,slot); different output, finish or usage fails. Dynamic response IDs and timestamps are intentionally omitted.

After every N-response round, cumulative /metrics deltas must be exactly:

- requests_total = accepted_total = completed_total = N;
- prompt_tokens_total and completion_tokens_total equal sums of the just-observed HTTP usage;
- inference_errors_total, requests_rejected_total, requests_cancelled_total, stream_backpressure_total and prefill_failed_total all unchanged.

Counter reset, concurrent foreign load, rejected request or mismatch is a hard failure, not an ignorable warning.

## Queue drain and VRAM sample contract

Every round, poll /metrics until active=0 AND queued=0. Require **two consecutive idle readings with identical VRAM values on BOTH GPUs**, separated by a polling interval. This avoids sampling after removal from the scheduler queue but before final asynchronous CUDA state cleanup. If occupancy/VRAM do not settle before drain_timeout (default 15 seconds), fail closed. Every successful round reports per-GPU memory from the server's hermes_vram_used_mib gauges. The harness itself never invokes nvidia-smi, CUDA APIs or memory allocation on either GPU.

GPU0 and GPU1 are assessed independently. Preflight establishes resident-memory baseline. **Each distinct N4/N8 cohort gets its own first-round warmup reference** (label warmup_cache_unclassified), permitting new lazy allocator/cache establishment when workload shape changes. That warmup must not itself grow by more than **256 MiB** per GPU relative to its pre-cohort baseline, or the harness fails closed on unexplained allocations. Later stable samples within that cohort are labelled stable or cache_or_allocator_plateau_not_proven_leak_free. Defaults: **FAIL** if any post-warmup sample is >64 MiB above that cohort's first settled round; **FAIL** if memory increases by at least 8 MiB on three consecutive post-warmup round-to-round increments. Flags: --vram-warmup-limit-mib (0–1024), --vram-allowance-mib (0–512), --vram-step-mib (1–128), --rising-steps (2–6).

**Interpretation:** a memory plateau is *compatible* with allocator cache or a bounded leak and does **not** prove leak freedom. Increasing memory may be a leak, lazy cache growth, or memory used by another process. These thresholds are deliberate conservative suspected-leak gates, NOT allocator attribution. Metrics report device-wide memory, not solely allocations by this process. A later authorized run must include isolated per-process VRAM/GPU evidence.

## Exact commands

OFFLINE tests actually run on Hermes Python 3.12.3, with only mocked urllib/time. No HTTP request:

    PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
      -s scripts/tests -p 'test_scratch_http_soak.py' -v

**Result: 19/19 PASS, zero errors/failures.** Negative tests cover wrong host/port/scheme/URL credentials, redirects/proxies, malformed/absent/duplicate metrics, error HTTP status, invalid JSON MIME/usage, deterministic output drift, inference errors, foreign traffic, invalid limits, active-before-load, timeout waiting to drain, VRAM not yet stable, N4-to-N8 cache establishment, excessive warmup growth, absolute postwarmup memory growth and sustained monotonic rise.

Real HTTP/GPU soak command below is **DOCUMENTED ONLY, NOT EXECUTED**. Requires a new explicit prime GPU grant, proven scratch process identity, and the already running isolated server:

    set -o pipefail
    python3 -B scripts/scratch_http_soak.py \
      --base-url http://127.0.0.1:18127 \
      --clients 4 8 --rounds 5 --max-tokens 8 \
      --prompt-style arithmetic --timeout 90 \
      --drain-timeout 15 --poll-interval 0.25 \
      --vram-warmup-limit-mib 256 --vram-allowance-mib 64 \
      --vram-step-mib 8 --rising-steps 3 \
      | tee /tmp/hermes27-http-soak-arithmetic.jsonl

Optional longer-answer workload, also NOT EXECUTED:

    set -o pipefail
    python3 -B scripts/scratch_http_soak.py \
      --base-url http://127.0.0.1:18127 \
      --clients 4 8 --rounds 5 --max-tokens 16 \
      --prompt-style explain \
      | tee /tmp/hermes27-http-soak-explain.jsonl

**Important runner constraint:** scripts/scratch_gpu_http_smoke.sh starts and stops its own isolated scratch server. Running it to completion and then starting this harness will fail because the server no longer exists. The prime must integrate the soak command **inside the approved scratch server lifetime** (or use a verified equivalent) and handle start/stop itself. Worker-6 did not alter that runner or touch any GPU service.

Each completed round prints HERMES_HTTP_SOAK_ROUND JSONL (N, round, wall, token totals, per-case SHA, latency, GPU0/1 VRAM, inference errors, drain and memory assessment); completion prints HERMES_HTTP_SOAK_PASS. On failure HERMES_HTTP_SOAK_FAIL_CLOSED is sent to stderr with exit 1. Earlier successful rounds do not constitute complete soak success. End-to-end completion tokens per wall second includes tokenization, prompt and HTTP: it is NOT GPU decode TPS or proof of a speedup.

## Release status

| Gate | Status |
| --- | --- |
| Bounded mocked N4/N8, strict response/metric/VRAM negative tests | PASS 19/19 CPU-only |
| Redirect/proxy/isolation; no CUDA/service/process action | PASS source and mock inspection |
| Real GGUF N4/N8 soak and VRAM trend | NOT RUN — PRIME GPU grant required |
| Full-vocabulary llama.cpp parity and production readiness | NOT CLAIMED |

Do not infer real stability, leak freedom, throughput or release readiness from the offline test results.
