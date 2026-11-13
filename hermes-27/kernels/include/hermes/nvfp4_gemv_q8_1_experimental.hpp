#pragma once

// ISOLATED EXPERIMENT; never silently route serving or PP2 through this API.
// GGUF W remains raw NVFP4: 36 bytes per 64 weights, row-major [N,K].
// X is *float32* [T,K], unlike the existing BF16 activation GEMV.
//
// Mirrors the llama.cpp CUDA quantize_q8_1 contract (quantize.cu):
//   per 32 elements: d_f32 = max(abs(x_i)) / 127.0f
//   q_i = (amax == 0 ? 0 : int8_t(roundf(x_i / d_f32)))
//   d_f16 = round_to_nearest_even_fp16(d_f32)
//   s_f16 = round_to_nearest_even_fp16(sum(x_i))
// IMPORTANT: q uses d_f32, *not* the stored d_f16.
// Negative 0 and half overflow follow CUDA/llama semantics.
//
// No CUDA malloc/free, host synchronizations or global state. Caller owns
// stream and device buffers; do not reuse scratch until its stream finishes.
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace hermes::q8_1_experimental {

struct alignas(4) Q8_1Block {
  uint16_t d;       // IEEE 754 FP16 activation quant scale
  uint16_t s;       // IEEE 754 FP16 sum of original float32 inputs
  int8_t qs[32];    // CUDA roundf(x/(amax/127)), NOT round-even
};
static_assert(sizeof(Q8_1Block) == 36);
static_assert(offsetof(Q8_1Block, qs) == 4);

// Returns 0 on invalid dimensions or size overflow. K multiple of 64.
size_t q8_1_scratch_bytes(int T, int K);

// One independent CUDA warp quantizes one 32-element block. Float input
// values must be finite (the caller enforces the upstream model invariant).
cudaError_t quantize_f32_q8_1(
    const float* x_f32, Q8_1Block* q8_scratch, int T, int K,
    cudaStream_t stream = nullptr);

// Direct raw-GGUF NVFP4 x Q8_1 projection, FP32 Y[T,N]; no repacking,
// no BF16 conversion and no persistent weights. Integer products of the
// E2M1*2 weights and Q8 int8 activations are summed before applying
// UE4M3 weight scale, stored FP16 Q8_1 d, and factor 0.5.
// Integer partials are exact. Floating accumulation order is not a
// guaranteed replica of llama.cpp's dp4a/MMQ output kernel.
cudaError_t gemv_q8_1_nvfp4(
    const Q8_1Block* x_q8, const uint8_t* gguf_nvfp4_w,
    float* y_f32, int T, int N, int K, cudaStream_t stream = nullptr);

// Convenience two-kernel enqueue; same scratch belongs to the same stream
// until both kernels have completed. Uses no CUDA allocations or sync.
cudaError_t project_f32_q8_1_nvfp4(
    const float* x_f32, const uint8_t* gguf_nvfp4_w,
    Q8_1Block* q8_scratch, float* y_f32, int T, int N, int K,
    cudaStream_t stream = nullptr);

}  // namespace hermes::q8_1_experimental
