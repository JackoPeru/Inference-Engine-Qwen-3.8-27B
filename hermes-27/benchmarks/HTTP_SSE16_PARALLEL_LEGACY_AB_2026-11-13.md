# HERMES-27: N1/N2/N4/N8 streaming A/B, 16 output tokens, real two-GPU PP2

**Run:** 2026-11-13 UTC, checkpoint
`Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf`
(physically verified GGUF SHA256
`db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1`).
**Decision:** exact same-engine text/usage parity PASS on 30 real
streaming requests per variant; no demonstrated advantage of the
continuous parallel scheduler over legacy async on this workload.
Independent full-vocabulary llama.cpp numerical fidelity remains FAIL;
production deployment remains NO-GO.

## Same-source, sequential A/B controls

Both runs used the same isolated CUDA 12.8 binary and checkpoint,
`/tmp/h27-parallel-scheduler-NTlic1/build/hermes_server`;
PP2 GPU0/GPU1 split 32+32 layers, NVFP4 raw weights, packed4 decode,
FP64-equivalent head selection unchanged, asynchronous request-local
streams enabled, `max_active=4`, queue 32, context 256, localhost
`127.0.0.1:18127`. Exactly one HERMES server existed at a time.

The new parallel run used
`--continuous-scheduler --parallel-steps`; the baseline used the
unchanged **legacy async** HTTP scheduler. Neither used seeded prefill
or MTP. Every request contained a deterministic 47-token prompt and
requested **16** greedy completion tokens; 2 rounds per N. The
same-source benchmark `scripts/scratch_http_sse_latency.py` read
the actual SSE stream, checked role, finish, 1 usage event and 1
`[DONE]`, and hashed all decoded visible text.

Strict preflight in `scripts/scratch_gpu_http_smoke.sh` verified
the scratch port free, service health, no competing model in GPU
memory, and locally bound only. Both runs passed existing real
model-backed streaming/nonstream parity, concurrent SSE and
cancellation smoke before the streaming load.

## A/B numerical results

All values below are **means of two rounds** for each scheduler and N.
Completion tokens per second includes input prefill and HTTP round
wall time, not just GPU decode. Latencies are p95 from each
round, then averaged (only 2 rounds; no confidence interval).

| Clients | Parallel tok/s | Legacy tok/s | Relative Δ | Parallel p95 request | Legacy p95 request |
|---:|---:|---:|---:|---:|---:|
| N1 | 2.829256 | 2.828674 | +0.021% | 5.655 s | 5.656 s |
| N2 | 3.711921 | 3.695010 | +0.458% | 8.620 s | 8.655 s |
| N4 | 4.871126 | 4.839380 | +0.656% | 13.137 s | 13.223 s |
| N8 | 4.856234 | 4.858091 | −0.038% | 26.355 s | 26.342 s |

With four simultaneous DecodeState owners, N8 executes as two
bounded cohorts. The plateau N4→N8 is consistent with this
configuration and does not alone prove a GPU scheduler fault.

## TTFC and why it is not exact TTFT/TPOT

The tool measures timestamp at the first **nonempty visible content
delta** (TTFC). This is a client-side SSE metric, **not** necessarily
the first generated token: Unicode buffering, control/special tokens,
network flush and the leading SSE role event can defer text.

| Clients | Parallel TTFC p95 | Legacy TTFC p95 |
|---:|---:|---:|
| N1 | 4.281 s | 4.280 s |
| N2 | 6.525 s | 6.601 s |
| N4 | 9.918 s | 10.016 s |
| N8 | 23.108 s | 23.173 s |

The benchmark also records per-visible-delta gap p50/p95 per round,
but those gaps are not token-aligned TPOT. Precise model TTFT/TPOT
needs server/GPU-token event timestamps and correlated request IDs,
not inferred timestamps from text chunk boundaries.

## Exact output parity and robustness

**PASS: 30/30** parallel requests had the same SHA256 of the complete
decoded text, same usage.completion_tokens=16, same 47 prompt tokens
and `finish_reason=length` as their corresponding legacy async
requests (same case/round). Independent `compare_sse_ab.py` rejects
any different output hash, missing/duplicate N/round, invalid token
usage, missing SSE PASS or missing GPU post-run restoration.
Its synthetic CPU suite and the SSE parser CPU suite passed **14/14**.

Post-run GPU memory returned after **both** runs to exactly:

    GPU0: 486 MiB, 0%
    GPU1: 1289 MiB, 0%

TabbyAPI and GPU Manager both remained `active` throughout.
No live ingress, installed server, systemd unit, lease, GGUF or
service state was modified; no `kill -9` was used.

Evidence:

- Parallel raw stdout: `/tmp/h27-parallel-sse-16tok-2rounds.log`,
  SHA256 `6d9dee23823e436d82faa923a0cc424f7759d5ec5c3420543fc51bd75f2248b9`.
- Legacy raw stdout: `/tmp/h27-legacy-sse-16tok-2rounds.log`,
  SHA256 `4d56d58655bacba13bf8b691f533ed119d8c278a1fc0951fbb673ddf56525f95`.
- Fail-closed comparator summary: `benchmarks/HTTP_SSE16_PARALLEL_LEGACY_AB_2026-11-13.json`,
  copied unchanged from
  `/tmp/h27-parallel-sse-ab-16tok-result.json` with SHA256
  `019965ca3358941b45d62c35bcc00c75c566a228fa323ffc4cfb499ad5607cf8`.
- Runnable benchmark harness:
  `scripts/scratch_http_sse_latency.py`,
  `scripts/compare_sse_ab.py`,
  `scripts/tests/test_scratch_http_sse_latency.py`,
  `scripts/tests/test_compare_sse_ab.py`.

## Final bounded conclusion

This workload verifies correctness of **HTTP-visible text/usage
relative to the same PP2 engine**, plus realistic multi-client
overlap and behavior. The new continuous scheduler no longer suffers
the old nonseeded single-pump flat N4/N8 rate, but has no meaningful
measured throughput advantage over legacy async. It remains opt-in.

Important unverified production gates: independent 248320-vocab
llama logits (strict gate currently FAIL), long context and
T8192 continuation, true per-model-token TTFT/TPOT, sustained
failure-injection/soak, MTP acceptance rollback, full trusted
ingress drain, PID owner attestation/fencing and rollback
under the actual GPU Manager watchdog. **No cutover authorized.**
