#pragma once
#include <cstddef>
#include <cstdint>
// RMSNorm: y[i] = x[i] / sqrt(mean(x^2)+eps) * w[i]
// x,y,w: BF16 bits (uint16_t). CPU ref in float.
namespace hermes {
void rmsnorm_cpu(const uint16_t* x, const uint16_t* w, uint16_t* y, int n, float eps);
#ifdef __CUDACC__
void rmsnorm_cuda(const uint16_t* x, const uint16_t* w, uint16_t* y, int n, float eps);
#endif
float bf16_to_f32(uint16_t b);
uint16_t f32_to_bf16(float f);
}
