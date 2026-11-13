#pragma once
#include <cstdint>
// Scaled-dot-product causal attention, single head, BF16 I/O, FP32 accum.
// q,k,v: [T,hd]; out: [T,hd]. scale = 1/sqrt(hd). Causal mask.
// CPU ref + CUDA (one block per query row). Test sizes T<=256, hd<=256.
namespace hermes {
void attn_causal_cpu(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                     uint16_t* out, int T, int hd, float scale);
#ifdef __CUDACC__
void attn_causal_cuda(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                      uint16_t* out, int T, int hd, float scale);
#endif
}
