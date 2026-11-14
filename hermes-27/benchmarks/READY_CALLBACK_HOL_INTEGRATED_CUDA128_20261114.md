# Ready-callback isolation: independent CPU and CUDA12.8 integration

Date: 2026-11-14. Release gate **NO-GO**. User priority: aggregate multiuser
throughput before prefill and individual decode. No physical speedup is claimed.

## Exact frozen scope

Clean Git archive source base:
`ebf8a28e9cdaaf09da62262f2c38d63aec874b9f`.
Isolated scratch:
`/tmp/h27-readycallback-integrated-ebf8-20261114`.
Only four SHA-verified worker-9 artifacts overlaid:

- `distributed/pp2_batch.cu`:
  `859abe1a9181dbd5c3b84a10a1402812b566221428e3f53e56f3613bfefe1192`.
- `distributed/pp2_batch.hpp`:
  `913bf42656b4f35bbf67b5c12ac764a027a0b16efc1f081378b6e97660e7d706`.
- `distributed/tests/test_ready_callback_delivery_hol_cpu.cpp`:
  `5fb97a8b59893f9bf928cb39a46825d9fd708b88679a6db8f8ffcda6d7b5eddb`.
- `benchmarks/READY_CALLBACK_DELIVERY_HOL_NEXT.md`:
  `b1a666a274c131ec090fe3f51626fdb19373b9c744335229660e7eafa512d565`.

This source was built and run only in a new isolated CPU/CUDA compile-only
workspace, not on HERMES's live inference GPU or production services.

## Source review and race correction

`enable_ready_callback_delivery` defaults **false**. It can only be
constructed with ready-completion/parallel backend, active slots >=2, event
consumer and no-throw emergency notifier. Opt-in queues event snapshots in
64 bounded preallocated slots, dispatches on <=max_active managed joinable
callback threads, enforces per-request callback ordering and defers final
success until previous token callback acknowledgements are consumed. No
callback worker has DecodeState ownership.

Independent review found an actual P0 ordering error in the worker's initial
candidate: callback `on_event` throw cleared
`callback_unconsumed` **before** `poison_callback_delivery` latched
`cancel_requested`. A concurrent ready pump could mark a failed event's
request completed. The worker corrected the **actual source**, retaining
`callback_running` and callback ack until poison/cancellation/emergency are
published. Deterministic CPU test uses a CV gate before exception recovery,
forces another `tick()` during the gap and verifies
`completed_total==0`, then one failed emergency and cancellation; no premature
state close, duplicate terminal, or success.

## Independent Ubuntu Linux receipt

From the frozen scratch, compiled the **real** scheduler (CPU-only translation
unit inclusion `distributed/tests/pp2_batch_cpu_only.cpp`) and the new
adversarial test with
`g++ -std=c++20 -O1 -g -Wall -Wextra -Werror -pthread
-DHERMES_ADMISSION_CPU_ONLY=1 -fsanitize=address,undefined
-fno-omit-frame-pointer`.

With `ASAN_OPTIONS=detect_leaks=1:abort_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1`:

- **PASS** new callback contract **265 CPU checks**, including baseline
  synchronous A callback blocking B and opt-in B terminal delivered while
  A callback remains blocked; no sanitizer errors.
- GCC sanitizer witness executable SHA256:
  `fd2514c48b2040dd3bf4171d1dbd7db024c46a4d48e4ed06999109d63476121a`.
- CMake Release CUDA12.8 with `HERMES_REQUIRE_SERVER=ON`,
  `HERMES_REGISTER_GPU_TESTS=OFF`: both real `hermes_server` and
  `hermes_generate` compiled/device-linked PASS.
- Existing ready scheduler host CTests **5/5 PASS**:
  ready completion optin; fault hardening; observability; terminal ID copy
  exceptions; fairness credits.

CUDA-linked executable SHA256:

- `hermes_server`:
  `4cb849ab1001ae1e72e884982ce25490e9993a74324ad63d7dc3c3cfd4370619`.
- `hermes_generate`:
  `215f266d56726225cd4fe8172b8e0a2cc42ffc26fec110704c15f553b0fb75b2`.

## Activation boundary

The changed scheduler is production-build-compatible but the **new mode is
not enabled by the server CLI** in this commit, and the default synchronous
behavior is retained. Server's shared `TokenizerBridge::call` currently
serializes pipe traffic with a mutex; new multiworker callback use and HTTP
SSE semantics still require targeted integration/negative tests before any
opt-in flag may be exposed. Physical GPU N1/N2/N4/N8 concurrent requests,
TTFT/TPOT/p95, VRAM, cancellation/long-context parity and 12/12 independent
full vocabulary are unverified. All-inbound drain and external GPU ownership
lease are not certified, so production remains **NO-GO**.
