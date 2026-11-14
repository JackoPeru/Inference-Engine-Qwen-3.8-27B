# Published 36b4b113 — coherent CUDA12.8 build and 14/14 CPU host CTest

Date: 2026-11-14. **Physical / numerical release verdict: NO-GO.**

## Provenance and isolation

- Exact published Git baseline:
  `36b4b113ea74fb45749215b4dba4541456153bf1`
  (`main`, clean).
- Clean Git archive, no overlaid production files:
  `/tmp/h27-coherent-git36b4-cpu-cuda-20261114/src/hermes-27`.
- Original `serving/server.cpp` SHA256:
  `90cd825f4fea3990ffe7b61dc5ace3e0eb1a36dca0a8d378cd398a4375d22621`.
- Original `CMakeLists.txt` SHA256:
  `5c03e054dabf43b904dcef5f6de4ef823a66707a385a25efcf86430c2b534e72`.
- Linux Ubuntu CMake Release, `/usr/local/cuda-12.8/bin/nvcc`,
  `HERMES_REQUIRE_SERVER=ON`,
  `HERMES_REGISTER_GPU_TESTS=OFF`.

Built genuine release `hermes_server` and `hermes_generate` with NVCC
objects and CUDA device-link **PASS**, without launching any CUDA kernels
or loading the 19.7 GB GGUF model.

Built binary SHA256:

- `hermes_server`:
  `2f7760016e4e126727010d557133779e80b178f0c8d07a443bde1984ef05f064`.
- `hermes_generate`:
  `393a9120b901676eeb860c03dc4d4257676307ed4d34ed354a705cc76cd2a8a3`.

## CPU host tests

The first `ctest -R host_` pass was incomplete: six scheduler tests and
five source-only tests passed, but three CLI host tests lacked their
executables because the build command omitted their independent CPU
mock targets. **This was a test-runner/build-target omission, not an
inference failure.** Compiled the three missing executable targets in the
same isolated snapshot, then repeated the full command with
`--output-on-failure`.

Final result: **14/14 PASS**, 4.54 seconds. These include:

- Six genuine scheduler C++ host targets, including bounded callback
  delivery, ready completion, fairness and terminal fault handling.
- Three standalone generator CLI CPU-mock tests: prefill shapes,
  sliding-KV guards, and private nonoverwrite logits dump.
- Five Python CPU source/staging contracts: pooled prefill stream
  dependency, async launch binding, external monotonic authority,
  six-route signed physical ingress proof model, post-FFN scratch tracer.

The five staging/source contracts are labelled
`cpu-only;source-contract;no-gpu`. Synthetic signer/ingress/nonce
positive fixtures cannot become privileged physical evidence.

## Read-only physical audit, not an owner grant

On 2026-11-14 at 13:49:40 UTC, GPU0 UUID
`GPU-5a3820fd-8771-64b3-7808-1404b1d71a85` reported 21 MiB and
GPU1 UUID `GPU-8777b420-7899-af2c-f518-d0b12a56c82c` reported
26 MiB, both 0% utilization; compute-app query returned no applications.
`hermes-tabby.service` and `hermes-gpu-manager.service` were inactive.
However direct, loopback, dashboard/hub and Tailscale listeners remain.
No all-ingress L7 drain, persistent lease antirollback issuer, complete
inflight inventory or both-card GPU PID/start-time ownership was attested.

One **isolated CPU-only** server left from an earlier local mock
test (`/tmp/h27-poison-health-red-4ec788a-20261114/host/build/hermes_server`,
PID 2716931, bound only to `127.0.0.1:58805`) was independently
matched to its exact fixture executable and command line, then stopped
gracefully with SIGTERM; the process exited. No Tabby, GPU Manager,
production HERMES or unrelated services were changed.

**Not certified:** genuine HERMES/llama post-convolution tensor parity,
independent full-vocabulary 248320-logit 12/12, real T8192+ CUDA
continuation, N1/N2/N4/N8 throughput, TTFT/TPOT/p95/fairness,
physical GPU failures/recovery, MTP K>1 or reversible cutover.
Physical release remains **NO-GO**.
