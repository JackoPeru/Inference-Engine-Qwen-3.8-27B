#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

namespace hermes {

// Optional BF16 output projection, intended for token decoding.
//   logits[t, v] = sum_{k=0}^{H-1} xn[t, k] * BF16(W[v, k])
// xn/logits are FP32; raw W is row-major BF16, *not* a GGUF NVFP4 tensor.
// Each CTA serves four vocabulary rows using one 32-lane warp per row;
// accumulation and final warp reduction use FP32 instead of the existing
// pp2_fwd.cu reference's FP64 accumulation. Expect small rounding differences.
// H must be positive and even; T and vocab must be positive. Device pointers
// must be valid; W aligned to 4 bytes (cudaMalloc suffices).
// Returns launch/argument status; does not synchronize or allocate memory.
// This is opt-in only: pp2_fwd.cu's scalar kernel remains untouched.
cudaError_t head_bf16_warp_cuda(float* logits, const float* xn,
                               const uint16_t* W, int T, int H, int vocab,
                               cudaStream_t stream = 0);

} // namespace hermes
