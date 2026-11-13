#pragma once
#include <cstdint>
// NVFP4 GEMV (decode path): y[i] = sum_k Wq[i*H+k] * x[k], Wq NVFP4 rows of H.
// Wq layout: F rows, each row H/64 blocks of 36B (4 UE4M3 + 32 E2M1).
// x BF16, y FP32 (accum). On-the-fly dequant, no BF16 upcast of W.
namespace hermes {
uint8_t fp32_to_ue4m3(float x);
void nvfp4_quant_row(const float* w, uint8_t* b36_blocks, int H);
float nvfp4_gemv_cpu(const uint16_t* x, const uint8_t* Wq_row, int H);
#ifdef __CUDACC__
void nvfp4_gemv_cuda(const uint16_t* x, const uint8_t* Wq, float* y, int H, int F);
#endif
}
