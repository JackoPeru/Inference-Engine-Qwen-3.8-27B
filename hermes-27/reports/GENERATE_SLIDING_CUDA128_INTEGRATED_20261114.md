# Standalone sliding-KV CLI: independent integrated CPU/CUDA build

Date: 2026-11-14. Release gate: **NO-GO**.

## Scope and source provenance

- Immutable Git base: `fe95ae70ca6f22cc8e960761a1a57d30d36e304d`, matched to GitHub `main` before testing; Ubuntu Git worktree clean.
- Snapshot: `/tmp/h27-cli-integrated-fe95ae70-20261114/src/hermes-27` generated with `git archive`. Four frozen worker-19 sources were overlaid by SCP and their SHA256 reverified. Neither the live server repository nor services were modified during validation.
- `tools/hermes_generate.cpp`: `f284a6b8daf39761b8b6d2bb9dedfed35db37ea0f17f88cb40b7cc4fdcdc3cea`.
- `tools/tests/mock_generate_sliding_backend_cpu.cpp`: `f130ad5c84ccac9e01a295fe5078c0e3e0d9704c802f76aa9c304631005fc443`.
- `tools/tests/test_generate_sliding_kv_args_cpu.py`: `5a4cad7f3716c2571f146a50222d2767b00ae6f374558ea2605f23b295ad87f8`.
- `reports/GENERATE_LONG_CONTEXT_SLIDING_DIAGNOSTIC_NEXT.md`: `9a6144b8e787be9a9eb753300d1329fca272893a6dcd1eb864094a5d55713d88`.
- Published `distributed/pp2_fwd.cu` unchanged at SHA256 `87364d1f21062a6dd2c009ca15efe16cde7d92636a35a0c576b08f855edbf71e` for this snapshot.

## Executed validation

1. On the isolated snapshot, `g++ -std=c++20 -O2 -Wall -Wextra -I distributed tools/hermes_generate.cpp tools/tests/mock_generate_sliding_backend_cpu.cpp` linked the real CLI with a CPU-only mock.
2. `python3 -B tools/tests/test_generate_sliding_kv_args_cpu.py <mock_binary>`: **11/11 PASS**. The T8192 prompt + two generated tokens checks `decode_step(pos=8192)` in the mock, bounded capacity 8192, one state release, explicit flag plus exact environment and negative pre-upload admission.
3. `python3 -B tools/test_generate_args_cpu.py <mock_binary>`: **7/7 PASS** (unchanged legacy CLI cases).
4. Independent GCC `-fsanitize=address,undefined -fno-omit-frame-pointer -O1` rebuilt the same real CLI + mock; **11/11 PASS** with `ASAN_OPTIONS=detect_leaks=1:abort_on_error=1` and `UBSAN_OPTIONS=halt_on_error=1`, with no sanitizer finding.
5. CMake Release configured in isolated `/tmp/h27-cli-integrated-fe95ae70-20261114/build`, `CMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc`, `HERMES_REQUIRE_SERVER=ON`, `HERMES_REGISTER_GPU_TESTS=OFF`. `cmake --build ... --target hermes_server hermes_generate -j2` completed both genuine CUDA target links, including generated device-link objects. NVIDIA CUDA 12.8 compiler, **no CUDA kernels launched**.

Frozen binary SHA256:

- CPU optimized CLI mock: `c818bcbdcd8cc6e4b3ad1275a023d4c0a2e21804d8c51228f5433ed86ce6338f`.
- CPU ASAN/UBSAN CLI mock: `4cefa600875a458dece41b4efbb0be7cd8ae8e100a9a7d47af82ebe63d76b073`.
- CUDA-linked `hermes_server`: `35fbe772b0e078f0bb2fec5246ebe5499115dab6220f75a6e8478e2f67235fcd`.
- CUDA-linked `hermes_generate`: `729d2a974ea95027aeb2c131d0f205d3775e4238fd76315e93e5fd7553f64f78`.

## What this does not establish

The mock intentionally has **no CUDA KV ring**, authentic GGUF numerical operations or native device state. CPU/ASAN and complete CUDA build only demonstrate CLI contract, lifetime errors visible in that mock, and compile/link compatibility with the real project. Physical T8192+ continuation, independent full-vocabulary logits 12/12, multiuser throughput, p95/VRAM, stream ownership and reversible deployment remain unproved. The CPU independent window-equivalence oracle is not a full-model oracle. MTP K>1 remains disabled. No user-serving ingress has been drained or an exclusive GPU grant attested. Never interpret GPU idle as authorization.
