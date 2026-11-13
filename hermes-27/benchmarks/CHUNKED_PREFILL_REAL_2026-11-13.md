# Experimental chunked PP2 stateful prefill — real checkpoint quality gate

Date: 2026-11-13. **Correctness PASS on T129 and T1024; throughput
and full 8192 context remain NO-GO for daily use.**

## Implementation and boundaries

The independently selectable
`pp2::prefill_chunks_into_decode_state(ids, state, stats, err, chunk_tokens)`
provides a bounded 1..64-token tile loop for **input length 65..8192**,
behind **`HERMES_PP2_CHUNKED_PREFILL=1`** and a fresh private
`DecodeState` with capacity at least T. It preserves full-attention
K/V, GDN recurrent state and convolution tails between tiles; RoPE and
causal attention use absolute positions. It returns the final
248320-token logits and permits a subsequent `decode_step` at pos=T
only when T<8192. **It is not used by the default CLI or HTTP routes.**

Fail-closed constraints include nonempty IDs, capacity fit, invalid
tiles, insufficient VRAM, and disallowing
`HERMES_PP2_PREFILL_BF16=1` (the BF16 Tensor Core path failed its
full-model strict numerical quality gate).

## Hardware and independent-internal oracle

- Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf, 64-layer PP2 split 32+32
  over **two NVIDIA RTX 5060 Ti 16 GiB**, CUDA 12.8, SM120a.
- `HERMES_NVFP4_GEMV=warp4`,
  `HERMES_HEAD_BF16=warp4`, no BF16 activation prefill.
- Scratch binary `test_prefill_chunk_real`, with
  `--oracle=1` **required** for pass: builds an independent target
  `DecodeState` from **sequential token-by-token `decode_step`**,
  then compares all final logits, all GDN/conv tails and full K/V,
  and all logits on one autoregressive continuation token.
- This is a **same-engine stateful oracle**, not an independent llama.cpp
  semantics oracle. It does not resolve the release gate's original
  full-vocabulary mismatch with llama.cpp.

| Input T | Tile size | Tile prefill GPU wall | GPU-stage H2D transfer | Final logits / states / continuation |
| ---: | ---: | ---: | ---: | --- |
| 129 | 64 | 13,297.307 ms | 0.549 ms | PASS, all max_abs **0** |
| 129 | 17 | 13,697.253 ms | 1.187 ms | PASS, all max_abs **0** |
| 1024 | 64 | 106,068.786 ms | 3.965 ms | PASS, all max_abs **0** |

The 1024-token test took several minutes including the sequential
reference. Both GPUs returned to their original idle allocation after
every run (GPU0 486 MiB, GPU1 1289 MiB). The T1024 state initialization
observed **9944 MiB GPU0 / 11557 MiB GPU1**, under the configured
15360 MiB cap. No live service was stopped or changed.

The standalone CPU test `distributed/test_prefill_chunk_cpu.py`
passed **217 checks** (91 convolution tail, 64 causal attention/RoPE,
42 K/V append, 20 static contracts). The experimental source and
test binary compiled and linked using CUDA 12.8 with no error. The
new CMake option **`HERMES_BUILD_CHUNK_PREFILL_TEST=ON`** builds the
real test binary and registers **only the CPU test** with CTest;
the GPU/oracle test is deliberately manual for exclusive maintenance
windows. The default CMake option is OFF.

## Reproduction (exclusive GPUs only)

```bash
# This path is an isolated scratch build, NOT the system's live installation.
export HERMES_NVFP4_GEMV=warp4
export HERMES_HEAD_BF16=warp4
unset HERMES_PP2_PREFILL_BF16

timeout 360 /tmp/hermes27-chunk-prime-20261113/build/test_prefill_chunk_real \
  /home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf \
  --n=1024 --chunk=64 --oracle=1
```

The T1024 test output is recorded in
`/tmp/h27-prime-chunk-1024-oracle.log`;
the T129/64 and T129/17 outputs are in
`/tmp/h27-prime-chunk-129-oracle.log` and
`/tmp/h27-prime-chunk-129-17-oracle.log`.
When `--oracle=0`, the test **must** return exit 3 UNVERIFIED rather
than claim success without state/logit comparison.

## Remaining blockers

- **8192 tokens untested** end-to-end and no next-token continuation
  possible at exactly 8192 without sliding-window or KV eviction.
- T1024 raw tile prefill alone cost **106 seconds**. No speedup is
  established over the serial path; latency is not production-usable.
- Strict independent full-vocabulary comparison against llama.cpp
  still fails, even though short greedy outputs can match.
- Real concurrency N2/4/8 and cancellation during chunked prefill
  remain untested; current tile kernels contain global stream
  synchronization and must not be promoted as concurrent serving.
- GPU Manager fenced lease and ingress drain are not installed;
  production cutover remains **NO-GO**.
