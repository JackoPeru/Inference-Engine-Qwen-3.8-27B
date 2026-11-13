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

To build the CPU-only test server independently from the real production binary, create a scratch directory and copy the named files from the current source tree. Use headers from the current distributed/ directory without modifying them:

    mkdir -p /tmp/hermes27-serving-cpu/{build,serving/tests}
    cp serving/server.cpp /tmp/hermes27-serving-cpu/server.cpp
    cp distributed/pp2_fwd.hpp distributed/pp2_batch.hpp /tmp/hermes27-serving-cpu/
    cp serving/tokenizer_bridge.py /tmp/hermes27-serving-cpu/serving/tokenizer_bridge.py
    cp serving/tests/mock_pp2_cpu.cpp /tmp/hermes27-serving-cpu/serving/tests/
    cp serving/tests/mock_gguf_dump.py /tmp/hermes27-serving-cpu/gguf_dump_fake.py
    chmod 755 /tmp/hermes27-serving-cpu/gguf_dump_fake.py
    touch /tmp/hermes27-serving-cpu/fake.gguf
    g++ -std=c++20 -O0 -pthread -I/tmp/hermes27-serving-cpu -I/opt/hermes/src/llama-cpp-turboquant/vendor/cpp-httplib -I/opt/hermes/src/llama-cpp-turboquant/vendor /tmp/hermes27-serving-cpu/server.cpp /tmp/hermes27-serving-cpu/serving/tests/mock_pp2_cpu.cpp /opt/hermes/src/llama-cpp-turboquant/vendor/cpp-httplib/httplib.cpp -o /tmp/hermes27-serving-cpu/build/hermes_server

Use your actual scratch path in the test commands. The test executable's parent directory must contain serving/tokenizer_bridge.py, as expected by the server's normal runtime helper lookup. The mock environment variables include FAKE_IDS, FAKE_STEP_MS, FAKE_INIT_FAIL, FAKE_DECODE_FAIL_ONCE and FAKE_GPU_USED_MIB.

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
