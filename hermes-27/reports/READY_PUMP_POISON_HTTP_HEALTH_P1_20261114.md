# Fatal ready-pump poison must withdraw HTTP readiness and fail-stop admissions

Date: 2026-11-14. **CPU/OFFLINE P1 FIX VERIFIED; PHYSICAL RELEASE NO-GO.**

## Original source fault

Published baseline `4ec788a8f21da311cc3529590f7eb936a0eeff05`
contains a scheduler fail-stop path: a ready-pump host exception latches
`ContinuousScheduler::stopping_=true` after joining all outstanding steps
and failing affected jobs. However the HTTP layer previously:

- always returned `{"status":"ok","ready":true}` from `/health`;
- always exported `hermes_server_ready 1` in Prometheus metrics;
- reduced every unsuccessful `Scheduler::enqueue`, including
  `SubmitCode::stopped`, to HTTP 429 `queue_full`;
- kept the HTTP pump loop sleeping/ticking despite the internal terminal
  poison latch.

That combination could advertise a permanently failed model as healthy and
mislead clients or a load balancer into treating it as transient congestion.

## Narrow production change

`serving/server.cpp` now:

1. Exposes `Scheduler::ready()` backed by its own stopped flag and the
   existing batch scheduler `metrics().stopping`, without a new scheduler
   or GPU owner.
2. Returns HTTP 503 with `{"status":"unavailable","ready":false}` from
   `/health`, and exports `hermes_server_ready 0`, once stopped.
3. Classifies a refused `SubmitCode::stopped` as HTTP 503
   `server_error/scheduler_stopped` with a restart-required diagnostic;
   ordinary full-queue rejects remain HTTP 429 `queue_full`.
4. Exits the HTTP side's continuous pump loop on terminal batch poison,
   without invoking any unsafe automatic model/GPU restart.
5. Adds `HERMES_ADMISSION_CPU_ONLY`-guarded one-shot
   `HERMES_TEST_READY_PUMP_POISON_ONCE`, which throws at the REAL scheduler
   `ready_before_commit` post-join hook. This hook is excluded completely
   from the compiled production CUDA target.

The test is `serving/tests/test_scheduler_poison_readiness_cpu.py`, and
`serving/tests/prod_sched_cpu_gate.sh` registers it in the offline HTTP suite.
The same commit also adds two CPU-only ready callback HTTP stop/cancel tests,
reported separately in `READY_CALLBACK_HTTP_STOP_CANCEL_NEXT_20261114.md`.

## Authenticated RED/GREEN witness

- Immutable Git source base: `4ec788a8f21da311cc3529590f7eb936a0eeff05`.
- Scratch: `/tmp/h27-poison-health-red-4ec788a-20261114`.
- The initial source overlay contained ONLY the CPU failpoint (no readiness
  or HTTP admission correction). An initial harness setup with the mock
  `gguf_dump_fake.py` not executable failed startup; this was corrected
  with the test fixture's required chmod and is **not** counted as a bug.
- **Authentic RED:** post-join exception was injected and the first
  inference request received HTTP 500, but the follow-up assertion that
  `/health` must return HTTP 503 failed: `HTTPError not raised`.
- **GREEN:** with the actual production serving correction, the same
  independently compiled fake-PP2 server and same fault injector passed:
  HTTP 500 for the affected request; `/health` HTTP 503 JSON
  `ready=false`; `/metrics` `hermes_server_ready 0`; next admission HTTP
  503 `scheduler_stopped`, zero additional backend-state opens; one
  backend close for the failed request. Test 1/1 PASS.

## Integrated regression and CUDA12.8 build

- The real `serving/server.cpp` and `distributed/pp2_batch.cu` were
  compiled into the offline HTTP server against the fake PP2 backend and
  real cpp-httplib/tokenizer bridge.
- Full `serving/tests/prod_sched_cpu_gate.sh` PASS on coherent final
  source, including legacy 19, continuous 19, state ownership 7, seeded
  prefill 11, chunked 7, parallel 2, ready-completion 6,
  ready-callback HTTP **7**, poisoned readiness **1**, sliding 5,
  streaming Unicode 9 scenarios and tokenizer ByteLevel 5.
- Fresh CMake `nvcc` CUDA12.8.93 Release, no CUDA execution, builds genuine
  `hermes_server` and `hermes_generate` through device-link PASS.
- Registered `host_test_ready_callback_delivery_hol_cpu` PASS 1/1 on the
  same snapshot.

Source SHA256:

- `serving/server.cpp`:
  `7482738e52cb1a560eb0a2c74ae070f6dc8b013bd4d3f742382ed335f9c34fe6`.
- `serving/tests/test_scheduler_poison_readiness_cpu.py`:
  `cc1a7b6f1ae184452a56834a16552152e3d01b6da962d3c6d66a107be5da34d0`.
- `serving/tests/prod_sched_ready_callback_test.py`:
  `e9295178d770970de66606a9160fa7779f4b038767bf973edf662fa5ab649f44`.
- `serving/tests/prod_sched_cpu_gate.sh`:
  `aa924c9438d8708aeb94e5237d06cc01d5ad1e53ee6929ee3aff4b0d10c82f67`.

Compiled binary SHA256:

- Release CUDA `hermes_server`:
  `396e49346b73ac7ab3652e342d699de3ae7eaa7fe79c268cb78bcc9c374de4d6`.
- Release CUDA `hermes_generate`:
  `3947c8cc9068115437d05ef9630eaeabecfe0735b3a6861cd3e13e64044cf4e3`.
- Final CPU fake HTTP `hermes_server`:
  `03d618b415507a388ed5432a1f2bc41a8061b86bc959aec1564bf45f5597e998`.

## Qualification

No live GPU context, model inference, physical producer/consumer fault,
all-ingress zero-inflight proof, external monotonic issuer, 12/12 authentic
full-vocabulary equality, or N1/N2/N4/N8 throughput/TTFT/TPOT/VRAM was
tested. The change prevents false HTTP readiness after a **known host-side
fail-stop**; it cannot detect every possible CUDA-driver hang, stalled
tokenizer subprocess, or crashed process. Real release stays NO-GO.
