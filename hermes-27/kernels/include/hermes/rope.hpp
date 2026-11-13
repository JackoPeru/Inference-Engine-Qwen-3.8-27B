#pragma once
#include <cstdint>
// GPT-J split-half RoPE, BF16 (matches transformers Qwen3_5 apply_rotary_pos_emb
// + rotate_half). Qwen3.8 mRoPE-text ha stream T/H/W identici -> cos/sin standard
// con inv_freq dim 64, theta 10M. Il mapping mRoPE completo (sezioni [11,11,10])
// vive nel layer architecture; il nucleo qui e' verificato contro transformers.
namespace hermes {
void rope_cpu(const uint16_t* x, uint16_t* y, int head_dim, int rotary_dim,
              long pos, float theta);
#ifdef __CUDACC__
void rope_cuda(const uint16_t* x, uint16_t* y, int head_dim, int rotary_dim,
               long pos, float theta);
#endif
}
