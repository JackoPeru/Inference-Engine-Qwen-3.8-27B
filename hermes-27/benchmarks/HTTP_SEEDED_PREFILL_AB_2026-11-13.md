# HERMES-27 real HTTP RAW seeded prefill A/B — 2026-11-13

## Status and recommendation

**Correctness smoke PASS, multiuser performance NO-GO for default
activation.** The new `--batched-prefill` server option is explicitly
opt-in and disabled by default. It calls the real stateful RAW-GGUF
T<=64 seeded API, never the BF16 experimental kernel that previously
failed the full-model quality gate.

The server/scheduler implement the same per-request state owner for the
entire request; prompt logits are used once for the first generated
token, then `decode_step` continues at absolute position `T`. For
prompt T65+ the server falls back to the tested stepwise seed.

## Build and verification

- Actual checkpoint:
  `/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf`
- Hardware: 2× RTX 5060 Ti 16 GiB, CUDA 12.8 SM120a.
- Isolated server on `127.0.0.1:18127`, fresh `gguf_dump`,
  real tokenizer, `--context 256 --active 4 --queue 32`.
- GPU env explicitly `HERMES_PP2_ASYNC_STREAMS=1`,
  `HERMES_NVFP4_GEMV=packed4`,
  `HERMES_HEAD_BF16=warp4`,
  `HERMES_DECODE_SKIP_LAYER_SYNC=1`.
  `HERMES_PP2_PREFILL_BF16` and
  `HERMES_PP2_CHUNKED_PREFILL` are **unset**.
- Source: `serving/server.cpp` and `distributed/pp2_batch.cu`
  linked in the same scratch build, not installed as a service.
- **70/70 CPU tests PASS**: 19 legacy HTTP, 19 continuous HTTP,
  7 state ownership/cancel, 11 seeded HTTP, 9 Unicode SSE cases and
  5 real-tokenizer tests.
- Actual GPU server smoke PASS with `--batched-prefill` in both
  legacy and continuous schedulers: concurrent nonstream/SSE,
  SSE text parity, stream cancellation, graceful shutdown.
  Both observed Prometheus prefill counters
  `seeded=5 failed=0 prompt_tokens=303` in their smoke,
  proving the flag was **actually exercised**, not silently ignored.
  Both processes exited cleanly and returned GPU VRAM to the
  same **486/1289 MiB** baseline.

## Benchmark: HTTP end-to-end completion throughput

Each row is **two short rounds**, 33 input tokens and exactly two
completion tokens **per request**. The reported aggregate tok/s is
`all_completion_tokens / total_round_wall_time`: it **includes**
prompt prefill, tokenizer and HTTP. These are not isolated GPU decode
tok/s and not a statistical throughput confidence interval.

| Concurrent clients | Legacy async, no seed, tok/s | Legacy async, RAW seeded, tok/s | Continuous, RAW seeded, tok/s |
| ---: | ---: | ---: | ---: |
| 1 | ~0.645 | 0.945 / 0.954 | 0.940 / 0.960 |
| 2 | ~0.840 | 0.965 / 0.966 | 0.963 / 0.965 |
| 4 | ~1.108 | 0.973 / 0.969 | 0.966 / 0.964 |
| 8 | ~1.110 | 0.974 / 0.979 | 0.962 / 0.962 |

The nonseeded continuous scheduler measured ~0.647 aggregate tok/s
across N1/2/4/8 in an earlier same-day run, likewise including
33-token prompts.

At **N1**, RAW batched seed improves wall completion throughput
approximately **47%** versus legacy async stepwise; at **N2** roughly
**15%**. At **N4/N8**, however, the batched legacy path is about
**12% slower** than **legacy async without seeded prefill**.
Continuous seeded also does not scale materially beyond ~0.96 tok/s.
The likely bottleneck is the current per-layer device-wide
synchronization and shared staging during batched seed, but **this
causal explanation has not been isolated by profiling**.

**Do not auto-enable `--batched-prefill` for multiuser production.**
The winning observed choice at N4/N8 is currently the original
legacy async path without seeded prompt batching. This is still
far below the throughput needed to replace the user's incumbent
engine.

## Reproduction

```bash
HERMES_TEST_ROOT=/tmp/hermes27-serving-prefill-prime-20261113 \
HERMES_TEST_MODE=async \
HERMES_TEST_SCHEDULER=legacy \
HERMES_TEST_BATCHED_PREFILL=1 \
HERMES_TEST_LOAD=1 HERMES_TEST_LOAD_STYLE=arithmetic \
bash scripts/scratch_gpu_http_smoke.sh

# Repeat using HERMES_TEST_SCHEDULER=continuous.
# To compare the legacy baseline, use HERMES_TEST_BATCHED_PREFILL=0.
```

The smoke checks `/metrics` after inference to ensure the configured
seed flag is reflected in actual seeded counters and rejects missing,
non-numeric, failed or unexpectedly nonzero counters.

## Remaining release gates

This verifies HTTP mechanics, short-context RAW seeded state and
end-to-end short-load behavior only. It does not resolve:

- independent llama.cpp full-vocabulary FP32 mismatch (12/12 traces
  fail strict gates despite matching greedy token IDs);
- longer context 1024/8192 and multi-client soak/tail-latency;
- recurring GPU startup allocation failure recovery and GPU Manager
  watchdog/ingress-drain lease;
- genuine overlapped batched prefill, fused inference throughput or
  a production-ready MTP K>1 speedup.

Production cutover remains **NO-GO**.
