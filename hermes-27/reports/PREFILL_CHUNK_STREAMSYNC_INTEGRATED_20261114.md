# Prefill stream-scoped barrier: independently integrated CUDA 12.8 gate

Date: 2026-11-14. **Physical release verdict: NO-GO**.
This report records a clean-source CPU and full CUDA build of the narrowly
scoped synchronization change in `PREFILL_CHUNK_SYNC_BARRIER_NEXT.md`.

## Exact source identity

- Git source snapshot: `a5c079bfa4c646e4236bd25aa7f5272ef86dce28`,
  archived without using or modifying the installed HERMES runtime.
- Scratch: `/tmp/h27-streamsync-integrated-a5c079b-20261114`.
- Original `distributed/pp2_fwd.cu` SHA256:
  `87364d1f21062a6dd2c009ca15efe16cde7d92636a35a0c576b08f855edbf71e`.
- Candidate `distributed/pp2_fwd.cu` SHA256:
  `6faccbda1aaf568deb7767393a53c5ea89a025a97fbc9746a0fc97a61e29469e`.
- New CPU dependency test SHA256:
  `b84041919d50fbe02ca2d7eb11d2e80bbec7a6f943e3d56a5e8bcd9dbdff4007`.
- Fixed stale `distributed/test_async_stream_contract.py` SHA256:
  `4853e62dfd9d46c1b11477d21e9a9db2bfb73342c41f99dc2b7b4356a663afe9`.

An independent `diff -u` of the original and candidate `pp2_fwd.cu`
showed only two end-of-layer changes (at original source lines 649 and 748):
`cudaDeviceSynchronize()` becomes
`pool ? cudaStreamSynchronize(0) : cudaDeviceSynchronize()`.
The CUDA wait result is still checked and normal unpooled calls retain the
device-wide barrier. The opt-in pooled chunked path has **64 stream-local
end-of-layer waits per chunk, not zero synchronization points**.

## Results on the integrated isolated source

- CPU source and synthetic stream dependencies:
  `PREFILL_CHUNK_SYNC_BARRIER_CPU_PASS source=52 stream_model=21 source_mutants=11`.
- Existing pool lifecycle/allocator:
  `PREFILL_CHUNK_WORKSPACE_CPU_PASS` for T65/T129/T1024/N4;
  10 fault negatives, 8 mutants, 5 cleanup checks.
- Exact-state BF16 source invariants:
  `state=37 tiled=18 source=18` PASS.
- Chunked convolution, attention/RoPE, KV, static contracts:
  `conv=91 attention_rope=64 kv_append=42 static=20` PASS.
- Prefill admission/integration contract: **102 CPU tail cases PASS**.
- Previously stale `distributed/test_async_stream_contract.py`
  produced RED on the immutable source: `expected 29 decode-layer launches,
  found 30`. The already-existing opt-in sliding KV branch adds one
  mutually exclusive stream-bound `sdpa_gqa_ring_decode_k` launch; the source
  was not changed to make the test pass. Updated the CPU test to require this
  ring launch under `if(sliding_wrap)` and check all 30 launch configurations.
  GREEN: `30 layer kernels, 14 GEMVs, 1 GDN call and 3 stage/head kernels
  all stream-bound`.
- CMake Release with `/usr/local/cuda-12.8/bin/nvcc`,
  `HERMES_REQUIRE_SERVER=ON`, `HERMES_REGISTER_GPU_TESTS=OFF`:
  full `hermes_server` and `hermes_generate` builds **PASS**, including
  NVCC objects and CUDA device links, without launching GPU work.

Frozen CUDA executable SHA256:

- `hermes_server`:
  `1bf79f41cb5150e7ec065a57fe649f176f9cdac0f2e0931a2f584f53a5965bf7`.
- `hermes_generate`:
  `ed068ebea90169d77eab95c9ee0b0fb194f0330f2acf2316baca398efcf95960`.

## Qualification and remaining blockers

Source audits establish that pooled layer operations use this host thread's
CUDA default stream and independent async decode uses request-private
nonblocking streams. Neither CPU stream models nor a successful CUDA build
prove that real drivers, mixed N1/N2/N4/N8 workloads or failed CUDA commands
preserve full-model parity. No GPU was accessed and no timing, VRAM, TTFT,
TPOT, fairness or speedup claim is supported. Original GGUF/llama full-model
248320-logit 12/12 and physical all-ingress lease+GPU owner evidence remain
blocked; do not enable production cutover or MTP K>1.
