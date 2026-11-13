# HERMES-27 streaming decoder — CPU validation (2026-11-13)

## Behavior

- Non-stream requests call the original Rust tokenizer decode once at completion.
- Stream requests call stream_start, stream_push and stream_finish on the Python bridge only for a pure ByteLevel tokenizer whose complete non-special vocabulary passes the byte-alphabet startup check.
- The bridge maps token pieces to raw bytes, then feeds an incremental UTF-8 decoder with errors=replace. It sends only complete Unicode characters. The final flush handles incomplete byte sequences exactly.
- Special tokens are skipped; missing tokenizer IDs produce no text, as in the Rust reference.
- Both paths run one final reference Rust decode. The stream path validates exact final text parity and fails on mismatch.
- Unsupported tokenizer decoders fall back to one reference decode at completion, emitting the result as a single SSE content chunk.
- Each session has independent decoder state. C++ tracks the number of bytes already sent, avoiding repeated copies and comparisons of growing text prefixes.

OpenAI chat SSE envelopes, the terminal finish chunk, [DONE], and request validation are preserved.

## CPU tests

On the Hermes host with the real Qwen tokenizer:

    HERMES_TEST_TOKENIZER=/opt/hermes/models/Qwen3.8-27B-Uncensored-EXL3-3.07bpw/tokenizer.json /opt/hermes/runtimes/exllamav3-venv/bin/python serving/tests/test_bytelevel_stream.py -v

The unit suite covers 1,200 random ID sequences, Unicode, malformed and incomplete UTF-8, specials and missing IDs, unsupported-decoder fallback, Jinja chat-template rendering with both thinking modes, and interleaved stream sessions on the actual bridge process.

HTTP mock tests run the unmodified HTTP implementation linked to an isolated CPU-only PP2 fake backend. The fake selects FAKE_IDS[pos] as the greedy token and uses EOS=248046. The command below assumes the scratch binary and helper have already been built:

    python serving/tests/test_stream_mock_http.py --server /tmp/hermes27-worker3-serving-check/build/hermes_server --model /tmp/hermes27-worker3-serving-check/fake.gguf --gguf-dump /tmp/hermes27-worker3-serving-check/gguf_dump_fake.py --tokenizer /opt/hermes/models/Qwen3.8-27B-Uncensored-EXL3-3.07bpw/tokenizer.json --python /opt/hermes/runtimes/exllamav3-venv/bin/python

Nine HTTP cases passed: ASCII, accents, CJK, emoji, skipped specials, EOS, split emoji bytes, incomplete UTF-8, invalid UTF-8. Every SSE-content concatenation exactly matched the non-stream response and Rust tokenizer decode. The isolated mock also passed serving/smoke.py, including cancellation, concurrent nonstream/SSE, and SSE/nonstream text parity.

## Decoder-only measurements

On Hermes CPU with the real Qwen tokenizer and random valid token IDs, excluding IPC, HTTP and model/GPU work:

| Generated tokens | Legacy full-prefix decode each token | Incremental | Speedup |
| ---: | ---: | ---: | ---: |
| 256 | 3.926 ms | 0.309 ms | 12.7x |
| 1024 | 58.747 ms | 1.066 ms | 55.1x |
| 2048 | 261.204 ms | 2.181 ms | 119.8x |

These are CPU decoder-only measurements, not end-to-end inference throughput. Stream IPC still passes through one mutex-protected Python process. The final reference check incurs one linear decode, intentionally retained for correctness.

Additional measurement using the actual persistent Python bridge protocol
(JSON lines over stdin/stdout, Python caller, single process, same random IDs):

| Output tokens | Legacy per-prefix bridge RPC | Incremental push RPC | Bridge speedup |
| ---: | ---: | ---: | ---: |
| 256 | 12.958 ms | 4.373 ms | 2.96x |
| 1024 | 154.297 ms | 15.015 ms | 10.28x |
| 2048 | 644.063 ms | 33.655 ms | 19.14x |

The bridge benchmark includes Python JSON serialization and pipe round trips,
but excludes native C++/HTTP costs and GPU execution. It does not include the
single final reference decode in the new streaming path.

## Remaining verification

Prime should build and smoke-test the real backend in isolated scratch after granting GPU access. Benchmark end-to-end latency and aggregate tokens/sec, Unicode SSE parity and concurrency against the previous serving build. Do not interrupt production Hermes.
