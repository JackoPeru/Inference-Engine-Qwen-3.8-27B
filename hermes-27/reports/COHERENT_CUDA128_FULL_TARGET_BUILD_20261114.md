# HERMES-27: coherent immutable-Git CUDA 12.8 server + generator build

**14 November 2026 — compile/link only; not a numerical or deployment PASS.**

## Snapshot, isolation and source identity

- Complete source tree was extracted from the **exact immutable Git commit**
  `a8cef7e8aeb3b61a19875ad4d095931cd7119266` using
  `git archive <sha> hermes-27` into
  `/tmp/h27-head-a8-coherent-cuda128-20261114/src` on Ubuntu.
- Compilation used only this extracted snapshot and a separate clean
  `/tmp/h27-head-a8-coherent-cuda128-20261114/build`, never the live
  deployment directory, installed production binaries, Python environments,
  Git worktree source files or actual checkpoint.
- CMake was configured with CUDA compiler
  `/usr/local/cuda-12.8/bin/nvcc`, **CUDA 12.8.93**, C++20,
  `HERMES_REQUIRE_SERVER=ON`,
  `HERMES_REGISTER_GPU_TESTS=OFF` and
  `HERMES_REGISTER_HEAVY_CPU_TESTS=OFF`. Optional W4A4 CUTLASS path
  stayed disabled. No model execution or GPU context acquisition occurred.

## Observed actual compilation/link

From the immutable extracted snapshot:

```bash
cmake -S src/hermes-27 -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DHERMES_REQUIRE_SERVER=ON \
  -DHERMES_REGISTER_GPU_TESTS=OFF \
  -DHERMES_REGISTER_HEAVY_CPU_TESTS=OFF
cmake --build build --target hermes_server hermes_generate -j2
```

**Result: both targets PASS, compiler exit 0, including CUDA device link.**
The build included actual published `distributed/pp2_fwd.cu`,
`distributed/pp2_batch.cu`, the shared CUDA kernel library, GGUF parser,
cpp-httplib compilation and serving runtime. The generator target also
compiled `gguf_dump` as a dependency.

Exact isolated executable SHA-256:

| Output binary | SHA-256 |
| --- | --- |
| `build/hermes_server` | `46222658553197423ab09e901258ee1bbd4a7cecf78e42befc25aaa105cefee6` |
| `build/hermes_generate` | `a8101298a8f9ccda3f8e0023e6c6c733a546825b864ed6f03d8be32d0d8e58bc` |

NVCC emitted existing nonfatal unused-variable warnings in
`nvfp4_gemv.cu` and `pp2_fwd.cu`; no compilation/link errors were
observed. These binaries remain scratch artifacts and were **not**
installed, served, signed or promoted.

## Limitations and next gate

Compilation does **not** prove that the CUDA scheduler and real kernel
produce numerically correct full-model logits, nor that N1/N2/N4/N8 GPU
throughput improved, nor that a continuous run works beyond T8192.
No production systemd service, GPU Manager, TabbyAPI, HTTP ingress, tailscale
listener, GPU context or checkpoint was touched. An independently verified
physical owner grant is required before real model execution.

After a future `pp2_fwd.cu` or `pp2_batch.cu` modification, redo the
relevant CUDA build from a **new immutable** source SHA. Keep the release
at **NO-GO** until the independent 12/12 full-vocabulary numerical gate and
the physical deployment/throughput/long-context gates pass.
