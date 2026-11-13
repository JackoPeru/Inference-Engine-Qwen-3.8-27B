# HERMES-27 NVFP4 Warp variants, isolated benchmark (2026-11-13)

**Result:** `packed-u32x4` beats the current production Warp4 by **1.47×, 2.46×, and 2.44×** for the tested decode GEMV shapes, passing full-output comparisons against **both** the original scalar and current production Warp4 and 286-row independent double CPU samples for each full-size shape. This is an isolated synthetic measurement; no production kernel was changed. **Recommendation:** candidate for guarded opt-in integration and real-model output/logit parity validation by prime. Do not extrapolate these values to end-to-end tokens/sec.

## Build/run

Run in a separate `/tmp` directory on an RTX 5060 Ti SM120 with CUDA exclusive clearance. Copy the **existing production** `nvfp4_gemv.cu`, `rmsnorm.cu` and their two headers, and the **new experimental** `nvfp4_gemv_warp_variants.cu/.hpp`, `bench_warp_variants.cu` into the same relative paths:

```bash
nvcc -O3 -std=c++20 -arch=sm_120 -I kernels/include \
  kernels/src/nvfp4_gemv.cu kernels/src/rmsnorm.cu \
  kernels/src/nvfp4_gemv_warp_variants.cu \
  kernels/tests/bench_warp_variants.cu -o bench_warp_variants
timeout 120 ./bench_warp_variants --device=0 --loops=20 --rounds=5
```

Raw GGUF NVFP4: 36 bytes/64 weights (four UE4M3 scales, then 32 packed E2M1 nibbles). Activation input is BF16 and output FP32. The original scalar and current Warp4 kernels are linked from the actual production source, not reimplementations. All 8 alternatives reuse exactly the same input and GGUF matrix per shape. Benchmark reports the median of five CUDA-event batches of 20 launches, with warmup, rotating variant order and no transfer/allocation in timed regions. GPU=0, NVIDIA GeForce RTX 5060 Ti (SM120).

## Benchmark measured values (first full run)

| Raw GGUF matrix `(H,F)` | Scalar ms | Production Warp4 ms | PackedU16x2 ms | PackedU32x4 ms | Warp4 / PackedU32x4 | PackedU32 effective GGUF GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `(5120,10240)` | 0.951378 | 0.133403 | 0.109078 | **0.090603** | **1.472×** | 325.5 |
| `(17408,5120)` | 1.061603 | 0.399381 | 0.226269 | **0.162211** | **2.462×** | 309.1 |
| `(5120,17408)` | 1.635387 | 0.375467 | 0.207683 | **0.154118** | **2.436×** | 325.3 |

After recompilation, a second identical 5×20 run reproduced the result:

| `(H,F)` | Production Warp4 ms | PackedU32x4 ms | Speedup |
| --- | ---: | ---: | ---: |
| `(5120,10240)` | 0.132045 | 0.090173 | 1.464× |
| `(17408,5120)` | 0.399131 | 0.162728 | 2.453× |
| `(5120,17408)` | 0.375350 | 0.154027 | 2.437× |

`unroll-2`, `split-2warps`, `split-4warps`, and `warp8-rows` also passed numerical checks but did not clear the 1.2× threshold on all three shapes. PackedU32x4 packs **four adjacent 36-byte GGUF blocks into one warp iteration**, each lane loading a 4-byte `uint32_t` of four E2M1-packed bytes, then accumulating eight BF16-input products. The 4B scale word and 4B packed E2M1 loads are aligned; there is no weight repack/preprocessing or additional VRAM buffer.

## Numerical verification

- All 7 benchmark shapes PASS: `(64,1), (64,5), (128,9), (512,64)`, plus three large shapes. Every output row is compared with original scalar and Warp4 (`abs error <= 1e-4 + 2e-4 * max(abs(scalar), abs(Warp4))`). All outputs finite, trailing 8 guard words unchanged, greedy `argmax` matches scalar. Error counts are 0 for every kernel and shape.
- For each large shape, an independent double-precision CPU dequant/dot oracle checked 286 rows including both ends and evenly spaced samples, tolerance `1e-4 + 2e-4 * abs(reference)`.
- PackedU32x4: maximum absolute difference from original scalar `1.9073486e-6`, `3.8146973e-6`, `1.9073486e-6` respectively. Max absolute deviation from independent CPU oracle `9.536743e-7`, `2.861023e-6`, `9.536743e-7` respectively.
- Caveat: synthetic GGUF bytes and BF16 activations are representative of the tensor layout, not proof of real-checkpoint logit parity. Kernel reductions are FP32 and may differ from reference summation order at extreme values or near-tie greedy candidates.

## VRAM and integration gate

Weight tensor bytes: 29,491,200 (`28.125 MiB`) for 5120×10240 and 50,135,040 (`47.8125 MiB`) for the other two. Measured aggregate CUDA allocations including input and outputs (CUDA allocator granularity) were **32 MiB**, **50 MiB**, **50 MiB**. All variants operate directly on the same resident weights/activation, with **zero extra runtime scratch buffer or persistent GPU allocations** per GEMV launch. The measured memory is test harness resident allocation, not model VRAM or a device-wide sampling peak. Final peak can differ after production integration.

For integration, retain the existing production Warp4 as a selectable fallback. Recheck full model logits/greedy on representative prompts, including near ties, before changing defaults; then measure decoded tokens/sec and concurrent request throughput separately. No production files or CMake were touched by this experiment.

**API constraints / safe fallback:** `nvfp4_gemv_warp_experiment()` requires `H > 0 && H%64 == 0`, `F>0`, non-null device pointers and GGUF weight address aligned to at least four bytes. `PackedU32x4` uses aligned 32-bit loads for both scale word and four packed E2M1 bytes. `H/64` need **not** be a multiple of four (tail warps skip missing blocks); arbitrary positive `F` is supported, including incomplete last CTA. All UE4M3 scale bytes, including zero and `0x7F`, are handled exactly as production Warp4. Invalid arguments return `cudaErrorInvalidValue`, with **no automatic fallback**: integration must explicitly select the proven production Warp4 when any precondition fails. No dequantization layout conversion is performed.
