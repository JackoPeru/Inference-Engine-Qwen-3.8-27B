#pragma once

// Experimental quality-preserving BF16-activation NVFP4 prefill.
// All pointers are device pointers; the caller owns VRAM, stream, and cuBLAS
// handle. No process-global state, cudaMalloc, cudaFree, or synchronization.
// NEVER included in the production build without an explicit integration.
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

namespace hermes::prefill_exact {

// Raw GGUF rows [N,K] using 36 bytes/64 weights. Device scratch for BF16
// [N,K] is 2*N*K bytes, reused across successive weights/layers.
// GGUF UE4M3 * E2M1 products are BF16-exact for every finite code pair.
size_t bf16_weight_scratch_bytes(int N, int K);

// Decode raw GGUF row-wise into row-major BF16 W[N,K] on device.
cudaError_t dequantize_gguf_to_bf16(const uint8_t* gguf_w,
                                   uint16_t* weight_bf16,
                                   int N, int K, cudaStream_t stream = nullptr);

// Computes Y[T,N] FP32 = X[T,K] BF16 * W[N,K] BF16 transposed.
// cuBLAS observes column-major shapes Wcol[K,N], Xcol[K,T], Ycol[N,T].
// Reusing an existing cuBLAS handle avoids handle creation per call.
// Call from one owning thread, with stream configured by this wrapper.
cublasStatus_t gemm_bf16_fp32(cublasHandle_t handle,
                              const uint16_t* x_bf16,
                              const uint16_t* weight_bf16,
                              float* y_fp32, int T, int N, int K,
                              cudaStream_t stream = nullptr);

// Batched direct GGUF kernel, no BF16 weight scratch; computes 4 tokens in
// one pass over each GGUF output row. Each warp owns one row, four warps/CTA.
// Outputs FP32 and preserves input BF16 / GGUF weight values (accum FP32).
// Alternative when transient BF16 scratch / weight-decode cannot be afforded.
cudaError_t batch4_direct_gguf(const uint16_t* x_bf16,
                               const uint8_t* gguf_w,
                               float* y_fp32, int T, int N, int K,
                               cudaStream_t stream = nullptr);

// Comparison baseline: one output-row warp per token, matching the
// arithmetic and launch geometry of existing nvfp4_gemv_warp4_k.
// This is a stand-alone copy for A/B measurement, not production routing.
cudaError_t repeated_warp4_gguf(const uint16_t* x_bf16,
                                const uint8_t* gguf_w, float* y_fp32,
                                int T, int N, int K,
                                cudaStream_t stream = nullptr);

}  // namespace hermes::prefill_exact
