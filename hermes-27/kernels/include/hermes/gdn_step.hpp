#pragma once
// Gated DeltaNet decode step, CUDA (real dims: H=48,K=128,V=128, fp32 state).
// S[h,i,j] in/out global; q,k,v,beta,decaylog per head. o[h,j] out.
// CPU reference in architecture/gdn_block (tiny) — test compares vs inline CPU here.
namespace hermes {
void gdn_step_cpu(const float* q, const float* k, const float* v,
                  const float* beta, const float* gl,
                  float* S, float* o, int H, int K, int V);
#ifdef __CUDACC__
void gdn_step_cuda(const float* q, const float* k, const float* v,
                   const float* beta, const float* gl,
                   float* S, float* o, int H, int K, int V);
#endif
}
