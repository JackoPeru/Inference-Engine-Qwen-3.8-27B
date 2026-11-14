# HERMES-27 — parallel PP2 next scaling / stream-event contract

**Assessment: 2026-11-14. CPU/OFFLINE only.** Prime baseline: \`6d2fe0a\` for this handoff; earlier analysis referred to \`80f4a66\`. The Windows source mirror has no \`.git\`, so neither revision was independently attested here. Original PP2, scheduler, serving, CMake, scripts, and existing benchmarks are unchanged. No network or real GPU run is claimed.

## Decision

**The most directly demonstrable multiuser head-of-line bottleneck is the unconditional whole-wave host join in \`ContinuousScheduler::tick_parallel()\`, followed by synchronous tokenizer/SSE callback processing in its single committing pump.** This is a source-level and actual-CPU-runtime ordering constraint. It is **not** evidence of a CUDA race, a GPU saturation cause, or an achievable throughput speedup.

The smallest next **strict opt-in experiment** is a bounded ready-completion queue **inside the same scheduler and same single request-state owner**. Independently finished backend steps could then be committed without awaiting an unrelated slow wave member. Keep existing legacy, serial continuous and wave-parallel defaults. First measure tokenizer callback wait; if material, move token delivery to a bounded per-request-ordered consumer while preserving the single Python IPC pipe mutex.

## Existing real SSE16 A/B: NOT re-run

Source: \`benchmarks/HTTP_SSE16_PARALLEL_LEGACY_AB_2026-11-13.md\` and JSON. Same verified GGUF SHA256 \`db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1\`, one isolated CUDA 12.8 server binary, 47 input/16 greedy output tokens, SSE, two rounds per N, active=4, separate nonblocking PP2 request streams, no seeded/chunked prefill or MTP.

| Clients | Continuous parallel completion tok/s | Legacy async tok/s | Relative delta | Parallel / legacy p95 latency |
| ---: | ---: | ---: | ---: | ---: |
| N1 | 2.829256 | 2.828674 | +0.021% | 5.655 / 5.656 s |
| N2 | 3.711921 | 3.695010 | +0.458% | 8.620 / 8.655 s |
| N4 | 4.871126 | 4.839380 | +0.656% | 13.137 / 13.223 s |
| N8 | 4.856234 | 4.858091 | -0.038% | 26.355 / 26.342 s |

All **30/30 SSE requests per variant** passed same-PP2 text SHA256 and token usage parity. This does **not** prove full 248320-vocabulary independent llama.cpp numerical fidelity; that gate remains FAIL. Completion tok/s includes HTTP/prompt/tokenization wall time, not raw GPU decode throughput. Client-first-visible-text TTFC is not true model TTFT or TPOT. N8 TTFC p95: ~23.108 s parallel vs ~23.173 s legacy. N8 uses only four simultaneously resident states and two short rounds cannot establish scaling or a confidence interval. No material parallel-vs-legacy advantage is shown.

The preceding 33-input/2-output results (~0.647 tok/s serial continuous vs ~1.11 parallel continuous at N4/N8) show host concurrency removed *one* serial bottleneck, not that another change will increase GPU throughput.

## Direct source evidence

1. **One wave-wide join before events.** \`distributed/pp2_batch.cu:689-880\`: choose \`min(running,max_active,steps_per_tick_remaining)\` distinct states; launch one \`backend_.step\` host thread per state; **line ~797 joins all workers before lines ~805-875 start to validate, commit and emit any tokens/terminal events**. \`pump_mutex_\` serializes whole ticks, but the metadata \`mutex_\` is short, so \`cancel()\` can mark a blocked request. Fast request results cannot be delivered before the slowest unrelated worker finishes. This is host head-of-line blocking, not an intrinsic GPU-wide barrier.
2. **Actual async PP2 is already request-local.** \`distributed/pp2_fwd.cu:1269-1336,1902-1985\`: a \`DecodeState\` owns nonblocking GPU0/GPU1 streams, \`ready0/ready1\` events, a \`step_mutex\` and private pinned token/hidden/logits staging. Its worker records \`ready0\`, calls \`cudaEventSynchronize(ready0)\` before GPU1 launch, then records \`ready1\` and calls \`cudaEventSynchronize(ready1)\` before returning full logits. **No \`cudaDeviceSynchronize()\` or legacy \`g_xfer_mu\` in the async step path**. Other request threads may enqueue on their own streams during those host waits. \`async_release\` drains both streams before any buffer/state free; preserve that fence.
3. **Token callbacks can block the one pump.** \`serving/server.cpp:559-617\`: \`continuous_event(token)\` synchronously invokes \`tokenizer_.stream_push(job_id,token_id)\` then \`stream_text(job)\`. \`serving/server.cpp:268-295\`: \`TokenizerBridge::call\` holds one mutex across request/response on the single Python helper pipe; \`read_line\` has bounded 15-second poll. A slow callback blocks later already-ready sibling events. This is demonstrably possible from source, **not** proof that actual N4/N8 time is dominated by tokenizer IPC. Do not remove the pipe mutex.
4. **Real RAW seeded/chunked prefill has CUDA-wide sync but is NOT concurrent with this SSE16 route.** \`distributed/pp2_fwd.cu:485-680\`: GDN/full layer prefill uses per-layer \`cudaDeviceSynchronize()\`; full-attention gating includes D2H/extract/H2D. The real T1024 chunked prefill prior test took ~106 seconds, not a multiuser performance result. However \`serving/server.cpp:1350-1375\` rejects parallel steps with batched prefill and permits chunked only on legacy \`--active 1\` with no async. The observed nonseeded parallel SSE16 plateau therefore **cannot** be attributed to an exercised concurrent seeded global CUDA barrier or a proven GPU state race.
5. **Soak is not per-token event profiling.** \`scripts/scratch_http_soak.py\` uses \`stream=false\`, checks repeated content/usage checksums, cumulative counters, active/queue drain and settled GPU0/GPU1 VRAM. It does not observe GPU0/GPU1 completion timestamps, token-ready times or callback queue delay. Its future PASS cannot refute this head-of-line witness.

## Newly added deterministic CPU witnesses — actual code, no original tests repeated

**\`distributed/tests/test_throughput_stream_event_scheduler.cpp\`:** links actual unmodified \`ContinuousScheduler\` under \`HERMES_ADMISSION_CPU_ONLY\`; injected state-private fake backend uses condition-variable barriers, not timing-dependent sleeps. Four cases:

- N2 and N4 force all backend calls to overlap; fast calls return while one slow call is blocked. No fast SSE token event arrives before slow release. Peak concurrent calls equals N, with one open/step/close per distinct request state.
- N4 cancel slow: cancel is responsive while backend blocks; slow logits are discarded, other jobs complete, one terminal event per job and no premature close.
- Token callback stall: even after backend outputs are complete, a deliberately blocked first token callback holds the next user's ready token; cancellation remains responsive and no double-free occurs.

These **PASS on current code precisely because they reproduce the unwanted wait edge**. A feature implementation must introduce a new **RED test with the opposite success criterion** (fast B event delivered while A still blocked), preserving token/state lifecycle.

**\`distributed/tests/test_throughput_stream_event_source.py\`:** eleven source-anchored and virtual event lifecycle Python unittest cases. Source gates check join-before-commit, actual PP2 request-local event/stream semantics and two host waits, raw prefill barriers with HTTP exclusion, synchronous tokenizer callback and IPC mutex, nonstreaming soak scope, plus the existing SSE16 parity JSON. Virtual request-local tickets reject duplicate absolute positions, premature free, cancelled token commit, repeat/invalid GPU0 or GPU1 event and double owner. Their illustrative timestamps are **not** real GPU/HTTP timing measurements.

### Offline test reproduction

From the HERMES source tree (no CMake modification, GPU registration or service invocation):

~~~bash
g++ -std=c++17 -O1 -g -Wall -Wextra -Werror -pedantic -pthread \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -DHERMES_ADMISSION_CPU_ONLY -Idistributed -x c++ \
  distributed/pp2_batch.cu \
  distributed/tests/test_throughput_stream_event_scheduler.cpp \
  -o /tmp/h27-stream-event-witness
ASAN_OPTIONS=detect_leaks=1 timeout 15 /tmp/h27-stream-event-witness
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest -v \
  distributed/tests/test_throughput_stream_event_source.py
~~~

**Previously executed in isolated Hermes CPU scratch** (no model/GPU): real scheduler witness **4/4 PASS**; GCC \`-Wall -Wextra -Werror -pedantic\` with ASAN/UBSAN and leak detection **PASS**; \`-O2\` optimized witness **PASS**; Python source plus virtual event tests **11/11 PASS**, Python 3.12.3. Preparation-only fixes in the *new tests* resolved a \`-Werror=comment\` backslash, made the N4 concurrency assertion deterministic via a bounded rendezvous, and corrected a too-short static method anchor. They do not change the original source. No new real GPU/HTTP benchmarks, full serving regression, or source modification are claimed.

## Next strict opt-in experiment — proposed, NOT implemented

**P0: bounded request-ready commit in the current scheduler.** Proposed \`enable_ready_completion_dispatch=false\`, requiring the existing explicitly async PP2 backend. Keep normal and legacy default routes. A single scheduler maintains FIFO admission, fair round-robin *launch*, request positions and generated IDs, cancellation, shutdown and state lifetime. At most \`max_active\` host worker calls and **one outstanding step/absolute position per DecodeState**. A worker only executes the existing blocking PP2 step on its private state, then publishes a bounded completion tuple \`{request_id, generation_epoch,position,logits_or_error}\`. Worker never samples tokens, emits SSE, frees state or starts a second scheduler. The sole pump commits independently ready results, per-request ordered \`admitted -> token* -> terminal\`, with no dependency on an unrelated slow worker. After a worker completes, reap/join that worker before its request state is reused/freed. Cancellation/timeout/shutdown mark jobs and suppress future commits, **but do not free in-flight streams/pinned buffers until own backend step returned and streams are drained**. Use a generation epoch to reject stale completions if an ID is retired/reused. Full 248320 logits require ~993280 bytes per ready item, so cap completion queue to \`max_active\` or lower verified budget. No scheduler metadata lock across GPU wait or tokenizer callback.

**Secondary gate:** synchronous tokenizer IPC can still block the committing pump after P0. If profiling proves material, use bounded per-request FIFO token event queues and a serialized consumer for the existing Python pipe. Preserve Unicode, stop sequence suppression, per-job SSE order, bounded backpressure/cancellation and terminal-after-tokens. Never bypass the single pipe mutex or let another scheduler own output state.

**P1 only after P0 profiling:** a distinct PP2 two-stage async ticket API could launch GPU0, wait/query \`ready0\`, transfer through the request-local pinned handoff, then launch GPU1 and query \`ready1\`; keep exact per-state streams/events and no premature free/relaunch. This requires independent PP2-fwd ownership and deeper CUDA correctness tests. Do not simply remove \`cudaEventSynchronize\` or copy device buffers before events have completed.

**P2 separately:** raw seeded/chunked multiuser prefill requires request-local stream/event isolation, owned scratch, strict GDN/conv/KV and full-vocab parity and live VRAM admission. Do not combine the current parallel and seeded/chunked flags.

## Promotion/falsification requirements

1. **CPU RED then GREEN:** fast B token and terminal before slow A returns; EOS, duplicate step prevention, position/epoch reuse, nonfinite logits, error, cancel at GPU0/GPU1, callback stall, SSE Unicode and stop, timeout, shutdown, no early/free twice. ASAN/UBSAN; functional TSAN only if environment supports it (previous host TSAN mapping failure is NOT a PASS).
2. **Only after prime-exclusive GPU grant:** compare against **both** legacy async and existing wave-parallel route, same GGUF/source/binary N1/N2/N4/N8, at least five rounds of 47/16 SSE and 33/2 controls plus bounded soak. Record GPU0/GPU1 event times, *model-token* ready and committed timestamps, tokenizer wait, real p50/p95 TTFT/TPOT, client TTFC separately, whole-HTTP-wall-time completion tok/s, VRAM and failure rates.
3. **Numerical and lifecycle gates:** same-engine full 248320 FP32 logits and GDN S/CS/KV/absolute positions; HTTP SSE bytes/usage/Unicode/stop parity, no state replay or mixed-user data. Independent llama.cpp full-vocabulary mismatch remains an unresolved production NO-GO.
4. **Reject opt-in** if no material N4/N8 throughput gain over existing legacy async with preserved p95/VRAM, or if any correctness regression occurs. These CPU tests demonstrate an ordering opportunity, **not a physical GPU speedup**.

**Bottom line:** request-ready completion within one bounded scheduler is the smallest falsifiable next throughput experiment. GPU stage polling and raw parallel prefill are separate gated follow-ups; production defaults must remain unchanged.
