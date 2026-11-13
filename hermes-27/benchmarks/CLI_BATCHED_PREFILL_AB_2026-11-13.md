# Stateful RAW-GGUF batched CLI prefill A/B — 2026-11-13

## Verdict and limits

Experimental `hermes_generate --batched-prefill` uses
`pp2::prefill_into_decode_state` to process a full prompt of
**1..64 tokens** once and seed the same per-request GDN S, convolution
tail and full-attention K/V state used by subsequent `decode_step`.
The default CLI path remains token-by-token. The option is **not**
automatically enabled for the HTTP server and does not implement
chunked prefill beyond T64.

Two **single-run** real-GGUF A/B checks on two RTX 5060 Ti 16 GiB:

| Input length | Token-by-token seed | RAW batched seed | Seed latency reduction | Seed speedup |
| ---: | ---: | ---: | ---: | ---: |
| 11 tokens | 993.944 ms | 694.445 ms | 30.1% | 1.43x |
| 64 tokens | 5797.819 ms | 3791.870 ms | 34.6% | 1.53x |

The decoding after seed was unchanged (~11 decode steps/s in these
short traces), and the timed seed numbers exclude model upload and
initial reference checks. This is **not** a statistically replicated
benchmark, a fused GEMM improvement, or evidence of production
throughput. The prefill is still far slower than the user's incumbent
runtime, and long-context performance remains a blocker.

## Full-vocabulary correctness

Both configurations generated five tokens; for **each** of the five
greedy positions the raw FP32 vector of 248320 logits was saved with
`--dump-logits-dir` and compared with strict absolute 1e-4 and RMSE
1e-5 tolerances.

| Prompt length | Greedy generated IDs (both paths) | Max abs / RMSE | Gate |
| ---: | --- | ---: | --- |
| 11 | `271,248068,198,760,1156` | **0 / 0** on 5 traces | PASS |
| 64 | `6059,13,271,248068,198` | **0 / 0** on 5 traces | PASS |

This establishes exact **same-GGUF token-by-token-vs-seeded-path**
full-logit parity for these traces, including four steps **after**
initialization. It is not an independent llama.cpp oracle and does
not resolve its existing numerical-release failure.

## Reproduction

Binary: `/tmp/hermes27-cpu-static-gates/hermes_generate`.
Checkpoint: `/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf`.
Use fixed environment:

```bash
export HERMES_PP2_ASYNC_STREAMS=1
export HERMES_NVFP4_GEMV=packed4
export HERMES_HEAD_BF16=warp4
export HERMES_DECODE_SKIP_LAYER_SYNC=1
unset HERMES_PP2_PREFILL_BF16

# Use the SAME explicit prompt IDs for each run; for T64 repeat the
# 11-token seed sequence until reaching exactly 64 IDs.
mkdir /tmp/seed-serial /tmp/seed-batched
hermes_generate --model /path/to/GGUF --prompt-ids IDs \
  --max-new-tokens 5 --dump-logits-dir /tmp/seed-serial
hermes_generate --model /path/to/GGUF --prompt-ids IDs \
  --max-new-tokens 5 --batched-prefill \
  --dump-logits-dir /tmp/seed-batched
python3 tools/compare_fp32_logits.py /tmp/seed-serial /tmp/seed-batched \
  --max-abs 0.0001 --max-rmse 0.00001
```

The live test used isolated directories
`/tmp/h27_prime_seed_compare_t11_20261113_{serial,batched}`
and `/tmp/h27_prime_seed_compare_t64_20261113_{serial,batched}`,
with JSONL and timing logs adjacent.

The optional BF16 Tensor Core experiment remains **NO-GO** for full
model quality and is never selected by `--batched-prefill` alone.
`HERMES_PP2_PREFILL_BF16=1` still requests the separate strict,
fail-closed diagnostic that has previously failed its quality gate.

CLI argument regression `tools/test_generate_args_cpu.py`
verifies that unsupported T65, invalid tokens, zero generation and
context overflow are rejected before any model upload. It never
initializes CUDA and is registered as `generate_batched_args_cpu`
in CTest.
