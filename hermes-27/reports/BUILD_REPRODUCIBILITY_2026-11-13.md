# HERMES-27 clean Git-checkout build gate — 2026-11-13

## What was checked

The GitHub checkout on the actual Ubuntu 24.04 server
`/home/matteo/hermes-27-gh/hermes-27` initially **could not build** without
an untracked `thirdparty/cutlass` directory. The CMake project
unconditionally included `kernels/src/nvfp4_gemm_w4a4.cu` and expected
unavailable CUTLASS headers. The production GEMV does **not** use W4A4;
its activation quantization previously failed the strict prefill quality gate.

The build now selects the prototype **only if**
`-DHERMES_BUILD_W4A4_PROTOTYPE=ON`, with a caller-provided
`-DHERMES_CUTLASS_ROOT=/path/to/cutlass`. An opt-out compatibility
implementation preserves the diagnostic `Nvfp4GemmW4A4` ABI and always
returns unsupported/zero sizes; it never launches or silently substitutes
the low-quality W4A4 kernel. PP2's production GEMV and BF16-exact path
remain independent of CUTLASS.

The optional-ON negative configuration was actually checked against a
checkout without CUTLASS: CMake exited **1**, stating precisely that the
prototype requires an include path containing `cutlass/cutlass.h`.

## Clean checkout build

```bash
cmake -S /home/matteo/hermes-27-gh/hermes-27 \
  -B /tmp/hermes27-cpu-static-gates \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DHERMES_REQUIRE_SERVER=ON \
  -DHERMES_BUILD_W4A4_PROTOTYPE=OFF \
  -DHERMES_BUILD_REAL_ASYNC_TEST=ON \
  -DHERMES_BUILD_EXACT_PREFILL_BENCH=ON \
  -DHERMES_BUILD_MTP_GENERATOR=ON
cmake --build /tmp/hermes27-cpu-static-gates -j3
ctest --test-dir /tmp/hermes27-cpu-static-gates -j3 --output-on-failure
```

CUDA compiler: **12.8.93**, GPU architecture **SM120a**. Complete
100% build/link **PASS**, including `hermes_generate`,
`hermes_mtp_generate`, `hermes_server`, `test_async_multiuser`,
`test_pp2_batch` and `prefill_exact_bench`. No GPU inference was
started by this build.

## Test policy

Default CTest is deliberately **CPU-safe**: real GGUF GPU tests must
not start while the user's ExLlama/Tabby services own the GPUs. The
following CTest subset passed **5/5 in 3.97 seconds**:

| CTest | Result |
| --- | --- |
| logit_comparator_cpu | PASS |
| release_harness_cpu | PASS |
| async_stream_contract_cpu | PASS |
| gguf_synth | PASS |
| prefill_exact_cpu | PASS |

Heavy CPU real-checkpoint tests `model_fwd` and `mtp_accept` are
now opt-in via `HERMES_REGISTER_HEAVY_CPU_TESTS=ON`, with
1800-second timeouts. The former **timed out after 120 seconds** when
run manually before the option was introduced; the latter was stopped
after ~92 seconds because its source explicitly warns of two ~10-minute
full CPU prefill computations. Neither test is counted as a pass.

GPU tests remain available as individually compiled binaries, but
automatic CTest registration requires `HERMES_REGISTER_GPU_TESTS=ON`
and the model's full GPU-exclusive maintenance window. External GDN
fixture tests require `HERMES_REGISTER_EXTERNAL_FIXTURE_TESTS=ON`.

This is a **build reproducibility gate, not a production release gate**.
Independent same-GGUF llama.cpp FP32 logits still fail strict numerical
thresholds, long context and live multi-client soak remain unresolved,
and GPU Manager watchdog coordination blocks safe cutover. See
`reports/RELEASE_GATE.md`.
