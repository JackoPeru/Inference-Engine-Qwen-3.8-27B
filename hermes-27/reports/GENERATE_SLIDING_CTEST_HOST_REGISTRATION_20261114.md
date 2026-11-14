# CPU-only CTest registration for sliding-KV generator

Date: 2026-11-14. Scope: `CMakeLists.txt` only plus this new report.
Release verdict remains **NO-GO**.

## Why

The experimental `hermes_generate --sliding-kv` already had direct CPU mock
tests and genuine CUDA12.8 compile/link receipts (see
`GENERATE_SLIDING_CUDA128_INTEGRATED_20261114.md`), but its new black-box
argument tests were not yet registered with CTest. Registration makes it
possible to run the CPU-only gate by name without loading the real GGUF or
creating a CUDA context.

## Change

- New `test_generate_sliding_kv_host` executable combines the **real**
  `tools/hermes_generate.cpp` with
  `tools/tests/mock_generate_sliding_backend_cpu.cpp` and the public
  `distributed` includes. It does **not** link `hermes_pp2` or CUDA.
- New CTest `generate_sliding_kv_args_host_cpu` invokes Python `-B`
  `tools/tests/test_generate_sliding_kv_args_cpu.py` with the exact host
  target path; timeout 120 seconds. All GPU tests retain their prior
  opt-in registration policy. Existing generator mock/host targets unchanged.

## Independently executed integration receipt

- Git base: `5e9813691c0ba1abbd734fa737d4552d649ea7c8`.
- Only the modified `CMakeLists.txt` (SHA256
  `44f29e7f059ebf7c0df34cf145570c497e915c396ec0d0266126dffedfb2b158`)
  was overlaid onto a clean `git archive`.
- Ubuntu immutable scratch:
  `/tmp/h27-sliding-ctest-5e98136-20261114`.
- CMake Release configured with `/usr/local/cuda-12.8/bin/nvcc`,
  `HERMES_REQUIRE_SERVER=ON` and `HERMES_REGISTER_GPU_TESTS=OFF`.
- `cmake --build ... --target test_generate_sliding_kv_host -j2`: PASS.
  Host binary SHA256:
  `dcc9b0cab113672d381bbed25f239f46f17e7734cd37a03c6d9e9c9403cd9280`.
- Linux `ldd` lists only libc/libstdc++/libgcc/libm/loader: **no CUDA**
  or GPU runtime dynamic library.
- `ctest -R generate_sliding_kv_args_host_cpu`: 1/1 registered target
  PASS (underlying 11/11 Python tests).
- Existing `generate_prefill_shape_host_cpu`: PASS; existing
  `generate_logits_dump_atomic_host_cpu`: PASS. Thus three relevant
  generator host CTest targets PASS on this coherent snapshot.

**Boundary:** this is CPU-mock behavior and CMake registration, not an
inference benchmark, physical ring test or proof of full-vocabulary numerical
equivalence. The external GPU/ingress/lease gate was not opened.
