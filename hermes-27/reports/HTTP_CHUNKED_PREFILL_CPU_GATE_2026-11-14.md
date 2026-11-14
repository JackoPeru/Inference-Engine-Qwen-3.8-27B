# HERMES-27: experimental HTTP RAW chunked prefill on one legacy state

**2026-11-14. CUDA build PASS / HTTP CPU mock integration PASS;
real HTTP GPU correctness and multiuser throughput UNVERIFIED.
Production NO-GO.**

## Actual integration and source scope

This follow-up starts from the immutable GitHub source commit
80f4a66636aa9b7a53cebf17a1ddbaf50181919d. Prime-owned changes
are ONLY serving/server.cpp and the new/updated serving/tests and this
documentation. No distributed PP2 implementation, scheduler, model,
GPU Manager, Tabby unit, CUDA kernel or production service was edited.
The source was compiled in a fresh Git archive scratch:

    /tmp/h27-http-chunked-BZlTbg

The existing pp2::prefill_chunks_into_decode_state API already
handles input T65..8192, 1..64-token tiles, absolute RoPE/causal KV,
recurrent GDN S and depthwise convolution CS. New HTTP route:

- Explicit --chunked-prefill implies the existing --batched-prefill.
  Prompt sizes 1..64 still use the unchanged one-shot RAW stateful
  function. Sizes 65..8192 invoke chunked prefill **exactly once**
  on the existing freshly opened per-request DecodeState.
- Chunked API returns only VOC=248320 last prompt logits, unlike the
  old T*VOC output. HTTP detects the correct shape without converting
  the last-row response into a second prompt replay.
- The legacy scheduler samples first generated token from those
  last logits and, if a continuation is required, calls decode_step
  with this token at absolute position T, never a prompt token at
  position 0..T-1 again. The same original state is closed once.
- A partial seeded state failure or non-finite row is terminal; no
  retry on the poisoned state and no second state allocation.
  Original EOS, Unicode SSE, stop and usage code remains shared.
- This route is intentionally **legacy-only and max_active=1**,
  until real GPU overlap and default-stream safety can be proven.
  It cannot be used with the continuous parallel scheduler; enabling
  both would violate the single-owner/stream audit requirements.

Start-up fail-closed conditions are checked before model upload:
exact HERMES_PP2_CHUNKED_PREFILL=1, --active 1, legacy scheduler,
no HERMES_PP2_ASYNC_STREAMS enabled, no packed4 NVFP4 GEMV, no L26
debug capture and no BF16 activation prefill. In every other case
the CLI exits 2. Absence of --chunked-prefill keeps both previously
existing defaults unchanged: plain legacy stepwise or optional
one-shot <=64 RAW seeded with T65+ stepwise fallback.

## Fresh CPU-only checks on actual source

The source-level CPU PP2 fake traces open, prefill, chunked, decode
absolute positions and close, with a realistic 248320-vector last
logit row and failure/NaN injections. It does not load the GGUF.

| Gate | Result | Scope |
| --- | --- | --- |
| Default legacy HTTP regression | 19/19 PASS | actual server source + CPU PP2 fake |
| Continuous HTTP regression | 19/19 PASS | parallel option not silently changed |
| State ownership/cancel | 7/7 PASS | no double PP2 state owner |
| Existing RAW seeded prefill | 11/11 PASS | T1/T11/T64, T65 fallback |
| **New chunked HTTP suite** | **7/7 PASS** | T64, T65/T129, SSE, T8192 first-token, failure, startup rejects |
| Parallel scheduler HTTP admission | 2/2 PASS | parallel remains opt-in |
| Unicode/SSE scenarios | 9/9 PASS | decoded visible SSE parity |
| Real tokenizer CPU fixtures | 5/5 PASS | no model CUDA inference |
| **Total HTTP CPU checks** | **79/79 PASS** | executed sequentially on same CPU mock binary |
| Native CTest host-only | **14/14 PASS** | built gguf_synth and mtp_rollback_cpu first |

The full CPU serving script initially exited 2 while the new Python
unittest runner incorrectly consumed application-specific --server
and --tokenizer arguments. The test runner was fixed to pass those
parameters to the shared harness parser instead. The subsequent
7/7 chunked suite passed, and the remaining unaffected
parallel/SSE/tokenizer tests were run explicitly. Do not count
the initial parser failure as a successful test.

Seven focused chunked tests prove:

1. T65/T129: exactly one open, one chunked seed, three continuation
   decode steps at T/T+1/T+2 for four output tokens, one close,
   exact prompt/completion usage and seeded counters.
2. T129 SSE: full content, one finish, one usage event, one DONE,
   no duplicated state and no prompt replay.
3. Default-off with environment variable present: T129 remains
   entirely stepwise; one-shot T64 remains unchanged.
4. T8192 input with max_new_tokens=1: first completion emitted
   without attempting invalid decode at absolute position 8192.
5. Partial chunk error and non-finite last logits: terminal response
   without attempting any step on the corrupted state.
6. Invalid flag combinations reject before loading any model.

## Fresh native CUDA build without GPU execution

CUDA 12.8 SM120a CMake build of hermes_server and hermes_generate
PASS in the isolated scratch source, including PP2, GDN and HTTP
linking. The normal unused-local NVCC warnings remain. CMake
registration defaults to HERMES_REGISTER_GPU_TESTS=OFF and no
full checkpoint was loaded in this follow-up.

The prior real-GGUF standalone chunked PP2 gate in
benchmarks/CHUNKED_PREFILL_REAL_2026-11-13.md had full last-logit,
S/CS/KV and continuation equality on T129 and T1024 against the
same-engine serial oracle, NOT against independent llama.cpp.
T1024 took approximately 106 seconds for chunked prefill alone:
the current implementation does **not** establish acceptable daily
latency or throughput. T8192 full-GPU inference has not been
validated and cannot continue after position 8191.

**Real HTTP N1/N2/N4/N8, cancellation during CUDA chunks, VRAM
high-water, p50/p95 TTFT/TPOT, soak and independent full-vocabulary
llama parity remain UNTESTED or FAILED** for this new HTTP route.
With --active 1 this route cannot increase aggregate N4/N8
throughput. Do not promote it as a performance optimization;
it is a bounded correctness integration milestone only.

## Reproduction (CPU-only, never use the installed runtime)

From an exact source checkout, with the installed real tokenizer
and cpp-httplib headers available:

    HERMES_CPU_SCRATCH=/tmp/h27-chunked-serving-cpu-gate \
      bash serving/tests/prod_sched_cpu_gate.sh

For a CUDA compile without GPU model execution:

    cmake -S . -B /tmp/h27-chunked-build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DHERMES_REQUIRE_SERVER=ON -DHERMES_REGISTER_GPU_TESTS=OFF
    cmake --build /tmp/h27-chunked-build -j3 \
      --target hermes_server hermes_generate

Do not attempt a real long-context HTTP CUDA run until an
independent, authoritative all-client ingress drain and both-GPU
process ownership fence protects existing TabbyAPI users. Never
modify /home/matteo/hermes-27 or the running GPU Manager.

**Release decision: NO-GO.**
