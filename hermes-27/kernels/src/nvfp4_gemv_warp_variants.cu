#include "hermes/nvfp4_gemv_warp_variants.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstddef>

namespace hermes {
namespace {

__device__ __forceinline__ float q_e2m1(unsigned q){
  const unsigned m=q&7u;
  unsigned bits=((126u+(m>>1))<<23) | (((m&1u)&(m!=1u))<<22);
  bits&=0u-(m!=0u);
  return __uint_as_float(bits | ((q&8u)<<28));
}
__device__ __forceinline__ float q_scale(unsigned q){
  if(q==0u || q==0x7fu) return 0.f;
  const unsigned e=q>>3, m=q&7u;
  return e==0u ? (float)m*0x1p-9f
               : __uint_as_float(((e+120u)<<23) | (m<<20));
}
__device__ __forceinline__ float q_input(const __nv_bfloat16* x,int k){
  return (float)x[k];
}
__device__ __forceinline__ float q_reduce(float acc){
  for(int delta=16;delta;delta>>=1)
    acc+=__shfl_down_sync(0xffffffffu,acc,delta);
  return acc;
}

// A = 2 -> 16 lanes per block, 16-bit load holds 2 GGUF q bytes;
// A = 4 -> 8 lanes per block, 32-bit load holds 4 GGUF q bytes.
// Each warp touches 2 or 4 neighboring blocks per iteration (same output row).
// Bytes are naturally aligned because every 36B block and row are 4B aligned.
template<int A, int ROWS_PER_CTA>
__global__ void packed_row_k(const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ W, float* __restrict__ y,int H,int F){
  constexpr int lanes_per_block=32/A;
  constexpr int bytes_per_lane=A;
  const int lane=(int)threadIdx.x&31;
  const int row=(int)blockIdx.x*ROWS_PER_CTA+((int)threadIdx.x>>5);
  if(row>=F)return; // entire warp exits, no cross-warp barrier
  const int nb=H/64;
  const uint8_t* wrow=W+(size_t)row*nb*36;
  const int part=lane/lanes_per_block;
  const int t=lane%lanes_per_block;
  const int sub=(t*bytes_per_lane)/8;
  const int pos=(t*bytes_per_lane)%8;
  float acc=0.f;
  for(int b=0;b<nb;b+=A){
    const int block_id=b+part;
    if(block_id>=nb)continue;
    const uint8_t* block=wrow+(size_t)block_id*36;
    const unsigned s=*reinterpret_cast<const uint32_t*>(block);
    const float scale=q_scale((s>>(sub*8))&255u);
    const int k=block_id*64+sub*16+pos;
    if constexpr(A==2){
      const unsigned data=*reinterpret_cast<const uint16_t*>(block+4+t*2);
      acc=fmaf(scale,q_input(x,k)*q_e2m1(data&15u)+
            q_input(x,k+8)*q_e2m1((data>>4)&15u),acc);
      acc=fmaf(scale,q_input(x,k+1)*q_e2m1((data>>8)&15u)+
            q_input(x,k+9)*q_e2m1(data>>12),acc);
    } else {
      const unsigned data=*reinterpret_cast<const uint32_t*>(block+4+t*4);
      #pragma unroll
      for(int j=0;j<4;++j){
        const unsigned q=(data>>(j*8))&255u;
        acc=fmaf(scale,q_input(x,k+j)*q_e2m1(q&15u)+
                 q_input(x,k+j+8)*q_e2m1(q>>4),acc);
      }
    }
  }
  acc=q_reduce(acc);
  if(lane==0)y[row]=acc;
}

template<int ROWS_PER_CTA>
__global__ void unroll2_k(const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ W,float* __restrict__ y,int H,int F){
  const int lane=(int)threadIdx.x&31;
  const int row=(int)blockIdx.x*ROWS_PER_CTA+((int)threadIdx.x>>5);
  if(row>=F)return;
  const int nb=H/64;
  const uint8_t* wrow=W+(size_t)row*nb*36;
  const int sub=lane>>3, j=lane&7;
  float a=0.f,c=0.f;
  for(int b=0;b<nb;b+=2){
    const uint8_t* block=wrow+(size_t)b*36;
    const unsigned scales=__shfl_sync(0xffffffffu,
        lane==0?*reinterpret_cast<const uint32_t*>(block):0u,0);
    const unsigned q0=block[4+lane];
    const int k=b*64+sub*16+j;
    const float sf=q_scale((scales>>(sub*8))&255u);
    a=fmaf(sf,q_input(x,k)*q_e2m1(q0&15u)+
             q_input(x,k+8)*q_e2m1(q0>>4),a);
    if(b+1<nb){
      const uint8_t* block2=block+36;
      const unsigned scales2=__shfl_sync(0xffffffffu,
          lane==0?*reinterpret_cast<const uint32_t*>(block2):0u,0);
      const unsigned q1=block2[4+lane];
      const float sf2=q_scale((scales2>>(sub*8))&255u);
      c=fmaf(sf2,q_input(x,k+64)*q_e2m1(q1&15u)+
               q_input(x,k+72)*q_e2m1(q1>>4),c);
    }
  }
  const float sum=q_reduce(a+c);
  if(lane==0)y[row]=sum;
}

// 128-thread CTA: WPR=2 => 2 independent rows, WPR=4 => 1 row.
// Each warp owns alternating GGUF blocks; shared reduction is over the warps
// within a row. Unused warps participate in __syncthreads to avoid deadlocks.
template<int WPR>
__global__ void split_warp_k(const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ W,float* __restrict__ y,int H,int F){
  constexpr int rows_per_cta=4/WPR;
  __shared__ float partial[4];
  const int lane=(int)threadIdx.x&31;
  const int warp=(int)threadIdx.x>>5;
  const int local_row=warp/WPR;
  const int local_warp=warp%WPR;
  const int row=(int)blockIdx.x*rows_per_cta+local_row;
  float acc=0.f;
  if(row<F){
    const int nb=H/64;
    const uint8_t* wrow=W+(size_t)row*nb*36;
    const int sub=lane>>3, j=lane&7;
    for(int b=local_warp;b<nb;b+=WPR){
      const uint8_t* block=wrow+(size_t)b*36;
      const unsigned s=__shfl_sync(0xffffffffu,
          lane==0?*reinterpret_cast<const uint32_t*>(block):0u,0);
      const unsigned q=block[4+lane];
      const float sf=q_scale((s>>(sub*8))&255u);
      const int k=b*64+sub*16+j;
      acc=fmaf(sf,q_input(x,k)*q_e2m1(q&15u)+
                   q_input(x,k+8)*q_e2m1(q>>4),acc);
    }
    acc=q_reduce(acc);
  }
  if(lane==0)partial[warp]=acc;
  __syncthreads();
  if(lane==0 && local_warp==0 && row<F){
    float r=partial[warp];
    #pragma unroll
    for(int w=1;w<WPR;++w)r+=partial[warp+w];
    y[row]=r;
  }
}
} // anonymous namespace

cudaError_t nvfp4_gemv_warp_experiment(const uint16_t* x,
    const uint8_t* W,float* y,int H,int F,Nvfp4WarpExperiment variant,
    cudaStream_t stream){
  if(!x||!W||!y||H<=0||H%64!=0||F<=0||
     (reinterpret_cast<uintptr_t>(W)&3u)!=0u)
    return cudaErrorInvalidValue;
  switch(variant){
    case Nvfp4WarpExperiment::PackedU16x2:
      packed_row_k<2,4><<<(F+3)/4,128,0,stream>>>(
          (const __nv_bfloat16*)x,W,y,H,F);break;
    case Nvfp4WarpExperiment::PackedU32x4:
      packed_row_k<4,4><<<(F+3)/4,128,0,stream>>>(
          (const __nv_bfloat16*)x,W,y,H,F);break;
    case Nvfp4WarpExperiment::Unroll2:
      unroll2_k<4><<<(F+3)/4,128,0,stream>>>(
          (const __nv_bfloat16*)x,W,y,H,F);break;
    case Nvfp4WarpExperiment::WarpSplit2:
      split_warp_k<2><<<(F+1)/2,128,0,stream>>>(
          (const __nv_bfloat16*)x,W,y,H,F);break;
    case Nvfp4WarpExperiment::WarpSplit4:
      split_warp_k<4><<<F,128,0,stream>>>(
          (const __nv_bfloat16*)x,W,y,H,F);break;
    case Nvfp4WarpExperiment::Warp8Rows:
      unroll2_k<8><<<(F+7)/8,256,0,stream>>>(
          (const __nv_bfloat16*)x,W,y,H,F);break;
    default: return cudaErrorInvalidValue;
  }
  return cudaPeekAtLastError();
}
} // hermes
