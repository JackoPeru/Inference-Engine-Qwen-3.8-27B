# HERMES-27 — HTTP strict ready-callback delivery integration

Date: 2026-11-14. **PRODUCTION / GPU RELEASE: NO-GO.**

## Scope

Build on published Git commit `3ca223f260bf96e939bfd2ff4a4b168eb5de7db7`.
The new scheduler callback worker implementation is already published but
remained private to `SchedulerConfig`; this change exposes a strict,
default-OFF `--ready-callback-delivery` HTTP server option and permanently
registers its existing adversarial C++ CPU mock as a host CTest.

Changed paths:

- `serving/server.cpp`: explicit flag, usage, early argument rejection,
  `enable_ready_callback_delivery` forwarded to the **same** continuous
  scheduler; no new scheduler or GPU owner.
- `CMakeLists.txt`: adds
  `host_test_ready_callback_delivery_hol_cpu` to the preexisting CPU-only
  scheduler-test registration loop (does not alter production CUDA language).
- `serving/tests/prod_sched_ready_callback_test.py`: NEW real-server HTTP
  negative gate, multiuser nonstream/SSE, real tokenizer Unicode/usage parity,
  single-state ownership, callback exception and terminal-allocation recovery.
- `serving/tests/prod_sched_cpu_gate.sh`: adds the NEW test to the existing
  offline fake-backend serving integration script.

Frozen source SHA256:

- `CMakeLists.txt`: `41b496414c42dc6535d3a5e20f5306eb2fa3b15dc638eabf6a08991277f77cc9`.
- `serving/server.cpp`: `aee8fdeb0bd8ffdc5913f737d0f34a014100d9c394ea72194902b4e04c719f00`.
- `serving/tests/prod_sched_ready_callback_test.py`:
  `a9686a04e64f9fbe958046538105da5e85bc05261e439ddba4862552b32f529e`.
- `serving/tests/prod_sched_cpu_gate.sh`:
  `700c88b7aa72d31937d0e6628a41499e498d99b59e6d1fa34a4b1b28e9b4016d`.

### Activation boundaries

Requires `--continuous-scheduler --parallel-steps --ready-completion
--ready-callback-delivery --active >=2` AND
`HERMES_PP2_ASYNC_STREAMS=1`; the new mode explicitly rejects `HERMES_L26`.
Missing or incompatible flags return code 2
**before model upload**; seeded/chunked and sliding-KV remain excluded
by their existing gates. Default mode and existing `--ready-completion`
without the new flag remain synchronous callbacks.

The real `TokenizerBridge::call` uses a mutex around each pipe transaction,
so a callback worker cannot interleave byte-level requests/responses with
another worker. Per-request callback acknowledgement/fence ensures only one
`continuous_event` callback per job at a time; the actual scheduler source
also keeps `callback_unconsumed` outstanding until emergency/cancellation is
latched after any callback exception. This integration does **not** claim
parallelization of the shared Python tokenizer itself.

## Immutable snapshot and checks

- Source base:
  `3ca223f260bf96e939bfd2ff4a4b168eb5de7db7` via `git archive`.
- Isolated Linux scratch:
  `/tmp/h27-callback-http-integrated-3ca223f-20261114`.
- CPU HTTP fake server was built from actual `serving/server.cpp`,
  actual `distributed/pp2_batch.cu` with `HERMES_ADMISSION_CPU_ONLY`,
  real cpp-httplib and existing fake PP2 backend/tokenizer bridge.
- Initial new-test **RED** came from the test expecting an incomplete set
  of valid fail-close diagnostics: sliding-KV correctly rejected a
  nonlegacy mode with `--sliding-kv requires ...`. The fixture now accepts
  that expected diagnostic without weakening the code-2/no-upload check;
  the targeted **5/5** tests pass. Subsequently the strict pre-upload
  admission was extended to reject debug `HERMES_L26=1`, and the mock HTTP
  server plus both CUDA executables were rebuilt on this final source.
- Fresh five new HTTP cases: all PASS, including concurrent N4 requests,
  Unicode composed/decomposed accents and emoji across token boundaries,
  SSE usage and single terminal `[DONE]`, terminal exception/host allocation
  failure and next-request survival.
- Existing HTTP tests exercised on the same compiled server and returned
  PASS: legacy 19, continuous 19, state ownership 7, seeded prefill 11,
  chunked 7, parallel admission 2, ready completion 6, sliding-KV 5,
  streaming Unicode nine scenarios, tokenizer ByteLevel 5. The earlier
  serving suite stopped at the original new-test expectation mismatch; the
  remaining sliding/Unicode/ByteLevel tests were then run independently.
- New CMake host CTest
  `host_test_ready_callback_delivery_hol_cpu`: PASS, 1/1 registered test
  encompassing **265 adversarial C++ CPU checks**.
- Full release CMake CUDA 12.8 compiler, GPU tests unregistered, builds both
  `hermes_server` and `hermes_generate` through full device link: PASS.

Built SHA256:

- CUDA-linked `hermes_server`:
  `5384287dc9ca00a140f92bf3c2d62c1f9504fae50bfefebdd160abd5d3148632`.
- CUDA-linked `hermes_generate`:
  `3c0ef4a565dc6b592ef7e4c7f811dfca6052ad737ef28e01b8baebb1613394f0`.
- CPU-only adversarial host CTest binary:
  `172428c72fa9f339b4c5f2f5bf63d9b2536bc14a9e80f7ae40babcd809008542`.

## Not demonstrated

No GPU context was acquired and no benchmark, user-facing production HTTP
endpoint, authenticated ingress drain, real GGUF throughput, independent
248320-logit parity or cancellation under physical CUDA failure was run.
All-inbound/UUID/PID/start-time ownership and independently monotonic lease
authority remain BLOCKED. Measured N1/N2/N4/N8 TTFT/TPOT/p95/VRAM, long
context real CUDA, 12/12 full vocabulary, MTP K>1 and reversible physical
deployment remain required for a release GO. Physical throughput speedup is
not inferred from any CPU callback scheduling test.
