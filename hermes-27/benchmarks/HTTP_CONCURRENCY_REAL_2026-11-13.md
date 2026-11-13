# Isolated real-GGUF HTTP concurrency, 2026-11-13

## Setup and verification

Dual RTX 5060 Ti 16 GiB, actual Qwen3.8-27B Uncensored NVFP4 GGUF,
independent scratch build, native `hermes_server` plus actual tokenizer.
Loopback `127.0.0.1:18127`, `--active 4 --queue 32 --context 256`.
Strict opt-in:

```text
HERMES_NVFP4_GEMV=packed4
HERMES_HEAD_BF16=warp4
HERMES_DECODE_SKIP_LAYER_SYNC=1
HERMES_PP2_ASYNC_STREAMS=1
```

Test scripts:

```sh
HERMES_TEST_MODE=async HERMES_TEST_LOAD=1 \
  bash scripts/scratch_gpu_http_smoke.sh
HERMES_TEST_MODE=async HERMES_TEST_LOAD=1 HERMES_TEST_LOAD_STYLE=explain \
  bash scripts/scratch_gpu_http_smoke.sh
```

Both runs passed the real GPU smoke: `/health`, model list/metrics,
invalid requests, concurrent SSE+nonstream, SSE/nonstream Unicode exact-text
parity, and explicit streaming cancellation.
Two repeated load rounds per client count completed without HTTP errors.
The test process was stopped gracefully, no production service was changed,
and GPU memory returned to its exact starting values:
GPU0 **486 MiB**, GPU1 **1289 MiB** (0% utilization).

## End-to-end request timings

All numbers include chat-template tokenization, prompt prefill, GPU
generation, HTTP scheduling, text decoding, and network loopback. They are
**not kernel decode tokens/sec** and have only two rounds per setting.

### Short arithmetic prompt, usually EOS after two tokens

33 prompt tokens per request. Each generated just two completion tokens
even when the request allowed up to eight.

| Clients | Requests | Total completion tokens | Wall, round 0 / round 1 (s) | Aggregate completion tok/s, rounds 0 / 1 |
|---|---:|---:|---:|---:|
| 1 | 1 | 2 | 3.1024 / 3.0901 | 0.645 / 0.647 |
| 2 | 2 | 4 | 4.7531 / 4.7526 | 0.842 / 0.842 |
| 4 | 4 | 8 | 7.2215 / 7.2028 | 1.108 / 1.111 |
| 8 | 8 | 16 | 14.3680 / 14.4078 | 1.114 / 1.111 |

For eight clients the request-level p50 latency was ~10.7 s and p95
~14.4 s. With `--active 4`, eight clients cause some requests to queue.
The short EOS makes these figures primarily a prefill/queue benchmark.

### Explanatory prompt, sixteen completion tokens each

47 prompt tokens per request; all requests produced the full 16-token
completion. This is a more useful end-to-end concurrency comparison,
but still includes prefill overhead.

| Clients | Requests | Total completion tokens | Wall, round 0 / round 1 (s) | Aggregate completion tok/s, rounds 0 / 1 |
|---|---:|---:|---:|---:|
| 1 | 1 | 16 | 5.6544 / 5.6430 | 2.830 / 2.835 |
| 2 | 2 | 32 | 8.6514 / 8.6544 | 3.699 / 3.698 |
| 4 | 4 | 64 | 13.2461 / 13.1627 | 4.832 / 4.862 |

N4 versus N1 increased aggregate end-to-end completion throughput
approximately **1.71x** on this narrow two-round workload, while serving
four times as many concurrent prompts. This must not be conflated with
the separate internal 15-step PP2 correctness test reporting ~14.56
aggregate decode steps/sec at N4, which excludes prompt preparation.

## Release decision

**HTTP path tested: PASS. Production release: NO-GO.** These tests do
not cover an 8K context, day-long soak, real crash recovery, broad
sampling, multiuser token/state equivalence at N8, or independent
llama.cpp full-vocabulary numerical equivalence. The latter currently
fails independently of this HTTP test; see `reports/RELEASE_GATE.md`.
No production TabbyAPI or GPU Manager services were stopped or reconfigured.
