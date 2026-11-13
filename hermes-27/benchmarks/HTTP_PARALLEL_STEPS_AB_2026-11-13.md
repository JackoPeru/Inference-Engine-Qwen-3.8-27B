# HERMES-27: strict opt-in continuous parallel host-step dispatch — real 2-GPU HTTP gate

**Run:** 2026-11-13 UTC (the workstation display may show 2026-11-14
after midnight Europe/Rome). **Decision:** real-GGUF HTTP correctness
and short-load concurrency PASS; **not** a production rollout or a
statistically established speedup over legacy async.

## Candidate and isolation

- Original real checkpoint: `/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf`,
  SHA256 `db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1`
  (previously physically verified).
- Isolated source/build: `/tmp/h27-parallel-scheduler-NTlic1`; HTTP test
  root `/tmp/h27-parallel-http-20261113`; local bind
  `127.0.0.1:18127` only. No installed PP2 binary, service unit,
  TabbyAPI, GPU Manager, gateway or existing ingress was changed.
- Runtime: CUDA 12.8 on 2 x RTX 5060 Ti 16 GiB, PP2 GPU0 layers
  0..31 and GPU1 layers 32..63.
- New strict opt-in: `HERMES_PP2_ASYNC_STREAMS=1` and
  `--continuous-scheduler --parallel-steps`, with
  `--batched-prefill` **disabled** and default decode otherwise
  identical to old nonseeded async path.
- Script:
  `HERMES_TEST_ROOT=/tmp/h27-parallel-http-20261113
  HERMES_TEST_MODE=async HERMES_TEST_SCHEDULER=continuous
  HERMES_TEST_BATCHED_PREFILL=0 HERMES_TEST_PARALLEL_STEPS=1
  HERMES_TEST_LOAD=1 HERMES_TEST_LOAD_STYLE=arithmetic
  bash scripts/scratch_gpu_http_smoke.sh`.
- Captured raw stdout:
  `/tmp/h27-parallel-http-real-n1n2n4n8.log`; server diagnostics:
  `/tmp/h27-parallel-http-20261113/http-smoke-async-continuous-seed0-parallel1.log`.

## Numerical correctness / API evidence

The scratch GPU HTTP smoke returned exit **0**. Real model-backed:
concurrent SSE plus non-stream PASS; exact SSE vs non-stream text
parity PASS; stream cancellation PASS. The raw nonseeded test reports:

    hermes_prefill_seeded_total=0
    hermes_prefill_failed_total=0
    hermes_prefill_prompt_tokens_total=0
    hermes_parallel_waves_total=198
    hermes_parallel_dispatched_steps_total=257

These counters were captured **after the smoke and before the
concurrency-load rounds**. There was exactly one request state owner
and no double scheduler in this opt-in path by source contract.
The independent llama.cpp full-vocabulary logits gate remains FAIL
from previous inference evaluations; this run does NOT demonstrate
that PP2 matches llama at full precision. A GPU smoke with
correct API semantics is not equivalent to strict numerical model
correctness.

## Measured aggregate HTTP throughput (two rounds per N)

Workload uses **33 prompt tokens and exactly 2 completion tokens per
request**, maximum 4 concurrently active requests, queue length 32.
Throughput = completion tokens / whole round wall time including
prompt processing and HTTP, **not** raw decode tok/s.

| Clients | Round 0 tok/s | Round 1 tok/s | p50 request sec (range) | p95 request sec (range) |
| --- | ---: | ---: | ---: | ---: |
| N1 | 0.644 | 0.646 | 3.0952–3.1066 | 3.0952–3.1066 |
| N2 | 0.847 | 0.848 | 4.7155–4.7178 | 4.7170–4.7195 |
| N4 | 1.112 | 1.115 | 7.1729–7.1887 | 7.1769–7.1927 |
| N8 | 1.116 | 1.112 | 10.7507–10.7914 | 14.3318–14.3857 |

Previous independently run *same-workload* `continuous` **serial**
HTTP was ~0.647 aggregate completion tok/s for N1–N8.
The new opt-in parallel scheduler therefore achieves ~1.114 at N8,
about **72% higher throughput**, and N8 p95 about **14.36 seconds**
versus ~24.71 seconds. This is a comparison of two isolated real
experiments, not exact simultaneous A/B; runtime noise and model
configuration must be considered. Previous *legacy async* HTTP
achieved ~1.11 tok/s at N4/N8 under the same workload, so the
new parallel scheduler does **not** prove a net advantage over the
existing legacy async throughput.

N8 active requests remain capped at 4. Therefore eight HTTP clients
do not imply eight concurrent GPU DecodeState instances, and N4→N8
plateau cannot independently establish a GPU bottleneck.
Samples are only two rounds and just two output tokens per request.
**TTFT, TPOT p50/p95, sustained mixed prompt lengths, GPU event
profiling, memory fragmentation, 24h soak, and independent
full-vocab quality were NOT measured**.

## Safe restoration and fail-closed acceptance

The isolated test's shutdown returned GPU idle allocations to:

    GPU0: 486 MiB, 0% utilization
    GPU1: 1289 MiB, 0% utilization

TabbyAPI and GPU Manager remained `active` in checks during the
run. The test script's preflight refuses either GPU if it already
hosts a full model or the scratch port is occupied; it only
terminates its own scratch `hermes_server`. It never stops Tabby,
manager, ingress or systemd. This is **not** evidence that a
production live cutover can safely drain every external user.

The final source build `hermes_server`, `hermes_generate` and
`test_gguf_synth` linked with CUDA 12.8; `ctest` ran 6/6
CPU tests PASS. The CPU HTTP test script passed **72 checks**:
19 legacy, 19 continuous, 7 state ownership, 11 prefill, 2
parallel admission/HTTP, 9 SSE Unicode scenarios and 5 tokenizer
cases. The worker's separate CPU mock scheduler checked 267
assertions, plus 11 FP32 admission tests with ASAN/UBSAN.

Separately, `test_pp2_batch` was compiled and started manually
against the real GGUF under a **90-second timeout** with CMake's
`HERMES_REGISTER_GPU_TESTS=OFF` setting. Its first GPU N2
four-step bitwise logit parity gate and parallel sanity gate
**PASS** with max error 0 for all compared steps; the following
long N1/2/4 throughput phase did **not** finish before timeout.
The complete standalone `test_pp2_batch` target is therefore
**INCOMPLETE / timed out**, not an all-tests-PASS result. The
separate bounded HTTP N1/2/4/8 smoke/load test did finish with exit
0 and is the throughput evidence above. After the timed-out test,
both GPUs returned to 486/1289 MiB baseline.

The concrete source change is additive and opt-in:
`distributed/pp2_batch.cu/.hpp` performs bounded concurrent
host threads over **distinct** request states, one owned pump,
join-before-commit, deterministic FIFO token events and no
shared-state double advancement; `serving/server.cpp` wires the
flag and strict admission. RAW seeded prefill and MTP remain
separately experimental. The default HTTP path remains
unchanged and must not be auto-switched.

**Gate: isolated short-load multiuser performance PASS, model
fidelity NO-GO, production lease/ingress owner attestation NO-GO.**
