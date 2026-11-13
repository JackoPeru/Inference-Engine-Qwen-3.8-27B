#pragma once
#include <cstdint>
#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#endif
// NVFP4 GEMV (decode path): y[i] = sum_k Wq[i*H+k] * x[k], Wq NVFP4 rows of H.
// Raw GGUF Wq layout (NOT CUTLASS-repacked): F rows, each row H/64 blocks
// of 36B: 4 UE4M3 scales followed by 32 packed E2M1 bytes.
// Each byte j of sub-block s contains weights (s*16+j, s*16+j+8).
// x BF16, y FP32 (accum). On-the-fly dequant, no BF16 upcast of W.
namespace hermes {
uint8_t fp32_to_ue4m3(float x);
void nvfp4_quant_row(const float* w, uint8_t* b36_blocks, int H);
float nvfp4_gemv_cpu(const uint16_t* x, const uint8_t* Wq_row, int H);
#ifdef __CUDACC__
// Explicit opt-in: existing 5-argument entry point stays scalar for ABI and
// numerical regression comparisons. Neither variant synchronizes the device.
enum class Nvfp4GemvVariant : uint8_t { Scalar = 0, Warp4 = 1 };
void nvfp4_gemv_cuda(const uint16_t* x, const uint8_t* Wq, float* y, int H, int F);
void nvfp4_gemv_cuda_variant(const uint16_t* x, const uint8_t* Wq, float* y,
                            int H, int F, Nvfp4GemvVariant variant,
                            cudaStream_t stream = 0);
#endif
}
