#include "hermes/head_bf16_optimized.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace hermes {

// One warp owns one output row. One 32-bit load per lane contains two BF16
// weights and spans 128 contiguous bytes per warp transaction. Four partial
// accumulators shorten the FP32 dependency chains (and reduce rounding drift).
// Output logits remain FP32 to match the PP2 caller. All 32 lanes participate
// in every reduction; an out-of-range vocabulary index exits as a full warp.
__global__ void head_bf16_warp_k(float* __restrict__ logits,
                                const float* __restrict__ xn,
                                const uint16_t* __restrict__ W,
                                int H, int vocab){
  constexpr unsigned full_mask = 0xffffffffu;
  const int lane = (int)threadIdx.x & 31;
  const int v = (int)blockIdx.x * 4 + ((int)threadIdx.x >> 5);
  if(v >= vocab) return;
  const int t = (int)blockIdx.y;
  const float* const input = xn + (size_t)t * (size_t)H;
  const uint32_t* const row = reinterpret_cast<const uint32_t*>(
      W + (size_t)v * (size_t)H);
  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  int k = lane * 2;
  for(; k+64 < H; k+=128){
    // Each lane reads two packed BF16 pairs, 64 features apart. The integer
    // bit expansion is an exact BF16->FP32 conversion with no rounding.
    const uint32_t p0 = row[k / 2];
    const uint32_t p1 = row[(k+64) / 2];
    const float w0 = __uint_as_float((p0 & 0xffffu) << 16);
    const float w1 = __uint_as_float(p0 & 0xffff0000u);
    const float w2 = __uint_as_float((p1 & 0xffffu) << 16);
    const float w3 = __uint_as_float(p1 & 0xffff0000u);
    acc0 = fmaf(input[k],w0,acc0);
    acc1 = fmaf(input[k+1],w1,acc1);
    acc2 = fmaf(input[k+64],w2,acc2);
    acc3 = fmaf(input[k+65],w3,acc3);
  }
  // Handles H divisible by 2 but not by 128 (including H=2 or H=64).
  if(k < H){
    const uint32_t p0=row[k/2];
    acc0=fmaf(input[k],__uint_as_float((p0&0xffffu)<<16),acc0);
    acc1=fmaf(input[k+1],__uint_as_float(p0&0xffff0000u),acc1);
  }
  // A single warp shuffle reduction avoids cross-warp shared memory.
  float sum = (acc0+acc1)+(acc2+acc3);
  for(int delta=16; delta>0; delta>>=1)
    sum += __shfl_down_sync(full_mask,sum,delta);
  if(lane==0) logits[(size_t)t*(size_t)vocab+(size_t)v]=sum;
}

cudaError_t head_bf16_warp_cuda(float* logits, const float* xn,
                               const uint16_t* W, int T, int H, int vocab,
                               cudaStream_t stream){
  if(!logits || !xn || !W || T<=0 || T>65535 || H<=0 || (H&1) ||
     vocab<=0 || ((uintptr_t)W&3u)) return cudaErrorInvalidValue;
  dim3 grid((unsigned)(vocab+3)/4,(unsigned)T);
  head_bf16_warp_k<<<grid,128,0,stream>>>(logits,xn,W,H,vocab);
  return cudaPeekAtLastError();
}

} // namespace hermes
