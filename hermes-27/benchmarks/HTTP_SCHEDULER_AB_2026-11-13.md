# Real-GGUF native HTTP scheduler A/B — 2026-11-13

## Environment and safety

- Actual GGUF: `Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf`, PP2 32+32
  layers, **two RTX 5060 Ti 16 GiB**, Ubuntu/CUDA 12.8.
- Server and tokenizer are built from the same isolated Git source
  snapshot. Port `127.0.0.1:18127` only, `--context 256 --active 4
  --queue 32`, real tokenizer and newly built `gguf_dump`.
- `HERMES_NVFP4_GEMV=packed4`,
  `HERMES_HEAD_BF16=warp4`,
  `HERMES_DECODE_SKIP_LAYER_SYNC=1`,
  `HERMES_PP2_ASYNC_STREAMS=1`.
  **No** `HERMES_PP2_PREFILL_BF16=1` (failed numerical quality gate).
- Isolated scratch smoke `scripts/scratch_gpu_http_smoke.sh` and
  `scripts/scratch_http_load.py`; no production service changes.
  `hermes-tabby` and `hermes-gpu-manager` stayed active.

## Real HTTP correctness

**Both** default legacy host scheduler and opt-in
`--continuous-scheduler` passed an actual model-backed HTTP smoke:
health/models/metrics, strict error codes, concurrent SSE plus
nonstream, Unicode SSE/nonstream exact-text parity, stream cancellation
and graceful shutdown. Both commands returned **exit 0**.

The final integrated PP2 source independently passed the manual real
GGUF seeded RAW prefill T11 test (all FP32 logits, GDN state,
convolution tail, K/V and next-token async continuation max_abs **0**,
exit 0). Earlier isolated PP2 source had also passed T2/T11/T64 before
a final shared pinned-buffer mutex adjustment; those are not claimed
as retested for the exact final source at all T values.

## End-to-end arithmetic load (includes prompt prefill)

**Two rounds for each N=1/2/4/8 setting**, yielding 2/4/8/16
responses per setting respectively. Each request's prompt token count is **33** and
its completion token count **2**. The completion speed below is
`sum(completion_tokens) / batch_wall_time`, including all prompt
prefill, tokenizer and HTTP overhead. This is **NOT** raw decode TPS.
There were no request failures in the successful benchmark runs.

| Clients | Continuous tok/s | Legacy async tok/s | Continuous p95 latency | Legacy p95 latency |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.645–0.647 | 0.644–0.647 | ~3.10 s | ~3.10 s |
| 2 | 0.648 | 0.840–0.841 | ~6.17 s | ~4.75 s |
| 4 | 0.647 | 1.107–1.110 | ~12.35 s | ~7.21 s |
| 8 | 0.647 | 1.110–1.111 | ~24.71 s | ~14.41 s |

The serial single-pump ContinuousScheduler has **no observed aggregate
scaling** through N8 in this prefill-dominated workload.
Legacy async host concurrency improves aggregate measured throughput by
approximately **71–72%** at N4/N8 but itself plateaus at N4.
Two short rounds do not establish tail latency or steady-state
confidence intervals. Do not extrapolate these low values to
independent decode TPS.

## Fault observed and disposition

The first scratch legacy-mode launch immediately after the continuous
run **failed to start**: actual GGUF upload reported
`OOM blk.26.ffn_up.weight dev=0 bytes=50135040`. Model startup
did not become healthy. The script failed closed and restored both
GPUs to idle baseline **486/1289 MiB**. A subsequent isolated
legacy smoke and full 1/2/4/8-client legacy load **both passed without
configuration changes**. This intermittent allocation failure is
**UNRESOLVED**; it is a release blocker for recovery/fault tolerance
until reproducibly explained or mitigated.

The scratch server was terminated after each run. GPU memory returned
to the same baseline, with no observed residual allocation after the
successful tests. Production TabbyAPI and GPU Manager remained active.

## Commands and next performance gate

```bash
HERMES_TEST_ROOT=/tmp/hermes27-prime-http-20261113 \
HERMES_TEST_MODE=async HERMES_TEST_SCHEDULER=continuous \
HERMES_TEST_LOAD=1 HERMES_TEST_LOAD_STYLE=arithmetic \
bash scripts/scratch_gpu_http_smoke.sh

# Repeat with HERMES_TEST_SCHEDULER=legacy.
```

The experimental batched state-seeding API currently does **not**
automatically accelerate the HTTP prompt path. Implement and validate
an opt-in integration, plus bounded parallel scheduling, before claiming
better multiuser throughput. Mandatory long-context, independent
llama.cpp full-logit, stress/soak and GPU Manager lease gates remain
**NO-GO**. See `reports/RELEASE_GATE.md`.
