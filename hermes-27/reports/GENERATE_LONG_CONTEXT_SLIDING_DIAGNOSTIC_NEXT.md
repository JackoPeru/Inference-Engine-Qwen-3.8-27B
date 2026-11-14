# hermes_generate: explicit sliding-KV continuation at absolute T8192

Date: 2026-11-14. Scope: worker-19, CPU mock + CUDA compile-only.

**Outcome: source/CLI implementation PASS; numerical and physical CUDA
continuation UNVERIFIED.** The backend's bounded 8192-token KV ring remains
experimental. No GPU was opened, no model was uploaded, no HTTP/service state
was changed, and no Git operation was performed.

## Frozen source identity

| Source | SHA256 |
|---|---|
| `tools/hermes_generate.cpp` before edit | `62b6e852121f1dc0abe3da71df766e8f40bf253462dfd5ab54781893c7a9868b` |
| `tools/hermes_generate.cpp` after edit | `f284a6b8daf39761b8b6d2bb9dedfed35db37ea0f17f88cb40b7cc4fdcdc3cea` |
| NEW `tools/tests/mock_generate_sliding_backend_cpu.cpp` | `f130ad5c84ccac9e01a295fe5078c0e3e0d9704c802f76aa9c304631005fc443` |
| NEW `tools/tests/test_generate_sliding_kv_args_cpu.py` | `5a4cad7f3716c2571f146a50222d2767b00ae6f374558ea2605f23b295ad87f8` |
| READ-ONLY `distributed/pp2_fwd.hpp` | `97649e4027ca8db47bed7f0323101edffbe3ddf68ea547f2534e9501d7696b2a` |
| READ-ONLY `distributed/pp2_fwd.cu` | `87364d1f21062a6dd2c009ca15efe16cde7d92636a35a0c576b08f855edbf71e` |

The frozen PP2 sources have identical SHA256 on the shared Windows source and
the Ubuntu source at `/home/matteo/hermes-27-gh/hermes-27`. Another worker
owns PP2 and serving; no edits were made to either.

## Source-anchored API compatibility

- `distributed/pp2_fwd.hpp:50-95`: `MAX_CONTEXT=8192`, per-state sliding flag
  and `SLIDING_KV_ABSOLUTE_POSITION_LIMIT=(1<<20)`.
- `distributed/pp2_fwd.cu:1674-1694`: `decode_init(state,err,capacity)`
  admits sliding only with *exact* `HERMES_PP2_SLIDING_KV=1` and capacity
  **exactly 8192**; rejects packed4/BF16-prefill/L26 debug.
- `distributed/pp2_fwd.cu:1770-1779,2108-2169`: sliding initializes
  private streams and mutex; preflight checks all 16 full-attention KV
  mirrors and GDN state, poisoning on failure.
- `distributed/pp2_fwd.cu:1601-1631,2243-2259`: at absolute
  `pos>=8192`, full attention uses the bounded ring; absolute
  `pos>=1<<20` is rejected. The old non-sliding position-capacity rejection
  remains intact.
- `tools/hermes_generate.cpp:153-155,190-213`: CLI flag and explicit
  environment validation happen in argument parsing, *before* `pp2::upload`.
  An environment-only opt-in without CLI flag also fails closed.
- `tools/hermes_generate.cpp:234-249,355-358`: sliding prompt size is
  limited to 8192, generation retains existing 1..8192 bound, absolute
  position is bounded and `decode_init` receives capacity **8192**.
  Default path still uses `prompt.size()+max_new_tokens-1 <= 8192`.
- `tools/hermes_generate.cpp:378-385,435-444`: sequential seed and
  generated-token continuation use the same `pp2::decode_step`; no second
  seed/prefill path was introduced.

**Opt-in CLI contract:** `--sliding-kv` requires a process environment with
`HERMES_PP2_SLIDING_KV=1`, a nonempty prompt of 1..8192 token IDs, and
`--max-new-tokens` in 1..8192. The flag rejects `--verify-prefill`,
`--batched-prefill`, `HERMES_PP2_PREFILL_BF16=1`,
`HERMES_NVFP4_GEMV=packed4` and any `HERMES_L26` value.
Sliding window is limited to 8192 resident KV slots, with absolute positions
and recurrent GDN/CS state preserved per PP2. It is **not** full-history
8193-token attention. No implicit activation is permitted.

The existing seed consumes prompt tokens at positions `0..N-1`.
The logits for generated token 1 come from the final seed call.
Only when a second token is requested does the CLI call
`decode_step(first_generated, state, N, err)`. For `N=8192`,
this is the first ring call at absolute `pos=8192`, used to choose
generated token 2. It does **not** claim a GPU computation occurred.

## RED -> GREEN witness, no model or GPU

Before modifying the CLI, the freshly built test binary with the new CPU mock
rejected `--sliding-kv` as an unknown/valueless flag and its help omitted the
option. The original `prompt=8192, max_new=2` was rejected by the context
guard. The initial new test run was RED (8 test methods; multiple expected
failures). One test initially used an *in-range* default pair
`prompt=1,max_new=8192` as an invalid example; corrected to
`prompt=2,max_new=8192` before the final successful run.

After the CLI change:

- **11/11 new tests PASS**, on Windows MSVC and Ubuntu GCC. Positive mock
  `prompt=8192,max_new=2`: `MOCK_INIT_CAPACITY=8192 SLIDING=1`,
  `MOCK_SEED_COMPLETE=8192`, `MOCK_POS=8192`, two generated tokens,
  and exactly one `decode_free` and `free_all`. Absolute position calls
  must be monotonic; the mock does not contain or simulate a physical KV ring.
- The mock verifies that invalid/missing exact env, incompatible flags, bad
  max token counts, prompt over 8192 and implicit env-only activation are
  rejected before the first `MOCK_UPLOAD` marker.
- Two successive prompts get fresh state. Both injected failures
  (`decode_step(pos=8192)` returning empty and non-finite logits) produce
  terminal errors with exactly one decode-state release and no JSON result.
  Injected failure at `pos=0` during seed also closes once.
- Existing `tools/test_generate_args_cpu.py`: **7/7 PASS on Ubuntu**;
  Windows **5 PASS + 2 POSIX-specific SKIP**, with no failures.

Windows isolated compilation used Visual Studio 2022 MSVC 14.44 in
`%TEMP%/h27_worker19_20261114`. The Ubuntu isolated build took place only
under `/tmp/h27-worker19-cuda128-GsnULA`, with newly staged CLI/mock/test
files. The source repository was read-only.

CUDA12.8 compile-only Ubuntu command and targets:

```sh
cmake -S /tmp/h27-worker19-cuda128-GsnULA \
  -B /tmp/h27-worker19-cuda128-GsnULA/build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/h27-worker19-cuda128-GsnULA/build \
  --target mock_generate pp2_cuda_syntax -j2
python3 -B /tmp/h27-worker19-cuda128-GsnULA/test_generate_sliding_kv_args_cpu.py \
  /tmp/h27-worker19-cuda128-GsnULA/build/mock_generate
python3 -B /tmp/h27-worker19-cuda128-GsnULA/test_generate_args_cpu.py \
  /tmp/h27-worker19-cuda128-GsnULA/build/mock_generate
```

`CMake` detected NVIDIA CUDA **12.8.93**, `sm_120a` targeting,
GCC **13.3.0** and configured successfully. `mock_generate` compiled and
linked. Real, unchanged `distributed/pp2_fwd.cu` compiled as an **OBJECT
library only**, with **10 unused-variable warnings**, zero errors.
CUDA compiler invocation here does not run a CUDA kernel. No full PP2/model
link or device runtime test was performed in this witness.

## Remaining gates

The gate establishes only CLI argument behavior and that the actual PP2
translation unit still compiles with the pinned CUDA compiler. It does not
establish numerical parity, correct physical eviction, real 8192-token
prefill/decode, GGUF quality, throughput, VRAM residency or service readiness.
Those require an exclusive hardware window, owned by the prime after
GPU/ingress authorization and independent workload evidence. Keep this mode
explicitly experimental until that gate passes.
