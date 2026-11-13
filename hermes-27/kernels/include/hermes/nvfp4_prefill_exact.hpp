#pragma once

// Experimental quality-preserving BF16-activation NVFP4 prefill.
// All pointers are device pointers; the caller owns VRAM, stream, and cuBLAS
// handle. No process-global state, cudaMalloc, cudaFree, or synchronization.
// NEVER included in the production build without an explicit integration.
#include <cstddef>
#include <cstdint>
#include <mutex>
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

// One warp per GGUF 64-weight block; every lane loads one E2M1 byte,
// broadcasts scales via shfl, and writes a contiguous packed BF16 pair.
// Results must be byte-exact vs dequantize_gguf_to_bf16.
cudaError_t dequantize_gguf_to_bf16_vector(
    const uint8_t* gguf_w, uint16_t* weight_bf16,
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

// A single caller-allocated BF16 scratch buffer can serve many projections
// without keeping their BF16 weights resident. It can even be smaller than
// W[N,K]: enqueue tiles over the output rows, writing directly into Y[T,N].
//
// init() and shutdown() may allocate/destroy only CUDA events/cuBLAS handle;
// they NEVER allocate/free a weight buffer. Caller owns buffer for the whole
// lifetime, must not use it outside this slot while in flight, and must keep
// GGUF/X/Y alive until the relevant CUDA stream reaches the final GEMM.
//
// enqueue() is thread-safe and cross-stream scratch-safe: host lock serializes
// submissions and the next stream waits on a CUDA event recorded *after* the
// preceding cuBLAS GEMM. There is no stream/device synchronization in enqueue.
// shutdown() waits for the final event before releasing handles. Do not call
// shutdown() while another thread may still invoke enqueue().
//
// Per-slot cuBLAS handle is PRIVATE (cublasSetStream on a shared handle would
// be a race). One slot per GPU device; other devices require separate slots.
// CUDA graph capture, externally modifying/freeing scratch and cross-device
// pointers are unsupported; failures poison the slot (fail closed).
enum class ScratchCode : uint8_t {
  Ok, Uninitialized, InvalidArguments, InsufficientCapacity,
  WrongDevice, CudaFailure, CublasFailure, Poisoned
};
struct ScratchResult {
  ScratchCode code = ScratchCode::Ok;
  cudaError_t cuda_error = cudaSuccess;
  cublasStatus_t blas_error = CUBLAS_STATUS_SUCCESS;
  bool ok() const noexcept { return code == ScratchCode::Ok; }
};

class ReusableBF16WeightScratch {
 public:
  ReusableBF16WeightScratch() = default;
  ~ReusableBF16WeightScratch();
  ReusableBF16WeightScratch(const ReusableBF16WeightScratch&) = delete;
  ReusableBF16WeightScratch& operator=(const ReusableBF16WeightScratch&) = delete;

  // buffer is an already allocated device pointer, aligned >= 16 bytes.
  // capacity is bytes; minimum is 2*K bytes (one output weight row).
  ScratchResult init(void* buffer, size_t capacity_bytes);
  ScratchResult enqueue(const uint8_t* gguf_w, const uint16_t* x_bf16,
                        float* y_fp32, int T, int N, int K,
                        cudaStream_t stream = nullptr);
  // Synchronizes the last submitted event if initialized. No weights cached.
  // Caller may free buffer only after successful shutdown().
  ScratchResult shutdown();

  size_t capacity_bytes() const noexcept { return capacity_bytes_; }

 private:
  mutable std::mutex submit_mutex_;
  uint16_t* buffer_ = nullptr;
  size_t capacity_bytes_ = 0;
  int device_ = -1;
  cudaEvent_t ready_event_ = nullptr;
  bool event_recorded_ = false;
  bool event_unknown_ = false;
  bool poisoned_ = false;
  cublasHandle_t handle_ = nullptr;
  ScratchResult shutdown_locked();
};

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
