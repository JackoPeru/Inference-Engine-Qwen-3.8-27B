#include "hermes/nvfp4_prefill_exact.hpp"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <limits>

namespace hermes::prefill_exact {
namespace {

// GGUF decode is raw unsigned E4M3 scale, not half-scale. All finite
// products of this scale and signed E2M1 are exactly representable in BF16.
__device__ __forceinline__ float ue4m3(unsigned c) {
  if(c==0u || c==0x7fu) return 0.f;
  unsigned e=c>>3, m=c&7u;
  if(e==0u) return float(m)*0x1p-9f;
  return __uint_as_float(((e+120u)<<23) | (m<<20));
}
__device__ __forceinline__ float e2m1(unsigned c) {
  switch(c&15u) {
    case 0: case 8: return 0.f;
    case 1: return .5f; case 2: return 1.f; case 3: return 1.5f;
    case 4: return 2.f; case 5: return 3.f; case 6: return 4.f;
    case 7: return 6.f;
    case 9: return -.5f; case 10: return -1.f; case 11: return -1.5f;
    case 12: return -2.f; case 13: return -3.f; case 14: return -4.f;
    default: return -6.f;
  }
}

// One lane per GGUF pair: writes k and k+8 BF16 into the same 16 group.
// Each 64-weight block is 36B: scale[4] followed by 32 E2M1 bytes.
__global__ void unpack_exact_k(const uint8_t* __restrict__ gguf,
                               __nv_bfloat16* __restrict__ dst,
                               int N,int K) {
  size_t tid=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  size_t total=size_t(N)*(K/64)*32;
  if(tid>=total) return;
  size_t block=tid/32, elem=tid%32;
  int n=int(block/(K/64)), b=int(block%(K/64));
  int s=int(elem/8), j=int(elem%8);
  const uint8_t* src=gguf+(size_t(n)*(K/64)+b)*36;
  unsigned q=src[4+s*8+j];
  float scale=ue4m3(src[s]);
  size_t o=size_t(n)*K+size_t(b)*64+s*16+j;
  dst[o]=__float2bfloat16_rn(scale*e2m1(q&15));
  dst[o+8]=__float2bfloat16_rn(scale*e2m1(q>>4));
}

// Four rows per CTA, one independent warp per row. Adjacent GGUF data
// 32B is loaded coalesced; each weight contributes to four T positions.
// Numerics: FP32 FMA in a different associative order than the PP2 scalar
// baseline, but no activation requantization or lossy weight conversion.
__global__ void gguf_batch4_k(const __nv_bfloat16* __restrict__ x,
                              const uint8_t* __restrict__ gguf,
                              float* __restrict__ y,
                              int T,int N,int K) {
  const int lane=int(threadIdx.x)&31;
  const int row=int(blockIdx.x)*4+(int(threadIdx.x)>>5);
  const int t0=int(blockIdx.y)*4;
  if(row>=N) return;  // whole warp exits, warp shuffle remains valid
  const int group=lane>>3, j=lane&7, blocks=K/64;
  const uint8_t* wrow=gguf+size_t(row)*blocks*36;
  float acc0=0.f,acc1=0.f,acc2=0.f,acc3=0.f;
  for(int b=0;b<blocks;++b) {
    const uint8_t* block=wrow+size_t(b)*36;
    unsigned packed=lane==0 ? *reinterpret_cast<const unsigned*>(block) : 0u;
    packed=__shfl_sync(0xffffffffu,packed,0);
    float scale=ue4m3((packed>>(group*8))&0xffu);
    unsigned packed_e2m1=block[4+lane];
    float v0=scale*e2m1(packed_e2m1&15);
    float v1=scale*e2m1(packed_e2m1>>4);
    int k=b*64+group*16+j;
    if(t0<T) {
      const auto* rowx=x+size_t(t0)*K;
      acc0=fmaf(float(rowx[k]),v0,acc0);
      acc0=fmaf(float(rowx[k+8]),v1,acc0);
    }
    if(t0+1<T) {
      const auto* rowx=x+size_t(t0+1)*K;
      acc1=fmaf(float(rowx[k]),v0,acc1);
      acc1=fmaf(float(rowx[k+8]),v1,acc1);
    }
    if(t0+2<T) {
      const auto* rowx=x+size_t(t0+2)*K;
      acc2=fmaf(float(rowx[k]),v0,acc2);
      acc2=fmaf(float(rowx[k+8]),v1,acc2);
    }
    if(t0+3<T) {
      const auto* rowx=x+size_t(t0+3)*K;
      acc3=fmaf(float(rowx[k]),v0,acc3);
      acc3=fmaf(float(rowx[k+8]),v1,acc3);
    }
  }
  for(int delta=16;delta>0;delta>>=1) {
    acc0+=__shfl_down_sync(0xffffffffu,acc0,delta);
    acc1+=__shfl_down_sync(0xffffffffu,acc1,delta);
    acc2+=__shfl_down_sync(0xffffffffu,acc2,delta);
    acc3+=__shfl_down_sync(0xffffffffu,acc3,delta);
  }
  if(lane==0) {
    if(t0<T) y[size_t(t0)*N+row]=acc0;
    if(t0+1<T) y[size_t(t0+1)*N+row]=acc1;
    if(t0+2<T) y[size_t(t0+2)*N+row]=acc2;
    if(t0+3<T) y[size_t(t0+3)*N+row]=acc3;
  }
}

__device__ __forceinline__ float e2m1_warp_bits(unsigned q) {
  unsigned m=q&7u;
  unsigned mant=(m&1u) & (m!=1u);
  unsigned bits=((126u+(m>>1))<<23) | (mant<<22);
  bits &= 0u-(unsigned)(m!=0u);
  bits |= (q&8u)<<28;
  return __uint_as_float(bits);
}
// Same code path and launch shape as production nvfp4_gemv_warp4_k;
// copied only into this experimental file so old production kernels are
// not touched during the fair isolated comparison.
__global__ void gguf_warp4_baseline_k(const __nv_bfloat16* __restrict__ x,
                                    const uint8_t* __restrict__ Wq,
                                    float* __restrict__ y, int H, int F) {
  const int lane=int(threadIdx.x)&31;
  const int row=int(blockIdx.x)*4+(int(threadIdx.x)>>5);
  if(row>=F) return;
  const int nb=H/64;
  const uint8_t* wrow=Wq+size_t(row)*nb*36;
  float acc=0.f;
  const int sub=lane>>3,j=lane&7;
  for(int b=0;b<nb;++b) {
    const uint8_t* block=wrow+size_t(b)*36;
    unsigned packed=(lane==0) ? *reinterpret_cast<const unsigned*>(block) : 0u;
    packed=__shfl_sync(0xffffffffu,packed,0);
    float sf=ue4m3((packed>>(sub*8))&0xffu);
    unsigned q=block[4+lane];
    int k=b*64+sub*16+j;
    float a=float(x[k]), c=float(x[k+8]);
    acc=fmaf(sf,a*e2m1_warp_bits(q&15u)+c*e2m1_warp_bits(q>>4),acc);
  }
  for(int delta=16;delta;delta>>=1)
    acc+=__shfl_down_sync(0xffffffffu,acc,delta);
  if(lane==0)y[row]=acc;
}

}  // namespace

size_t bf16_weight_scratch_bytes(int N,int K) {
  if(N<=0 || K<=0 || (K&63)!=0 ||
     size_t(N)>std::numeric_limits<size_t>::max()/(size_t(K)*2))
    return 0;
  return size_t(N)*K*2;
}

cudaError_t dequantize_gguf_to_bf16(const uint8_t* gguf_w,
                                   uint16_t* weight_bf16,
                                   int N,int K,cudaStream_t stream) {
  if(!gguf_w || !weight_bf16 || !bf16_weight_scratch_bytes(N,K))
    return cudaErrorInvalidValue;
  const size_t threads=size_t(N)*(K/64)*32;
  unpack_exact_k<<<unsigned((threads+255)/256),256,0,stream>>>(
    gguf_w,reinterpret_cast<__nv_bfloat16*>(weight_bf16),N,K);
  return cudaGetLastError();
}

cublasStatus_t gemm_bf16_fp32(cublasHandle_t handle,
                              const uint16_t* x_bf16,
                              const uint16_t* weight_bf16,
                              float* y_fp32,int T,int N,int K,
                              cudaStream_t stream) {
  if(!handle || !x_bf16 || !weight_bf16 || !y_fp32 ||
     T<=0 || N<=0 || K<=0) return CUBLAS_STATUS_INVALID_VALUE;
  auto status=cublasSetStream(handle,stream);
  if(status!=CUBLAS_STATUS_SUCCESS) return status;
  const float alpha=1.f,beta=0.f;
  return cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,
     N,T,K,&alpha,
     weight_bf16,CUDA_R_16BF,K,
     x_bf16,CUDA_R_16BF,K,
     &beta,y_fp32,CUDA_R_32F,N,
     CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

cudaError_t batch4_direct_gguf(const uint16_t* x_bf16,
                              const uint8_t* gguf_w,
                              float* y_fp32,int T,int N,int K,
                              cudaStream_t stream) {
  if(!x_bf16 || !gguf_w || !y_fp32 || T<=0 || N<=0 || K<=0 || K%64!=0)
    return cudaErrorInvalidValue;
  gguf_batch4_k<<<dim3((N+3)/4,(T+3)/4),128,0,stream>>>(
    reinterpret_cast<const __nv_bfloat16*>(x_bf16),gguf_w,y_fp32,T,N,K);
  return cudaGetLastError();
}

cudaError_t repeated_warp4_gguf(const uint16_t* x_bf16,
                                const uint8_t* gguf_w,float* y_fp32,
                                int T,int N,int K,cudaStream_t stream) {
  if(!x_bf16 || !gguf_w || !y_fp32 || T<=0 || N<=0 || K<=0 || K%64!=0)
    return cudaErrorInvalidValue;
  for(int t=0;t<T;++t)
    gguf_warp4_baseline_k<<<(N+3)/4,128,0,stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16+size_t(t)*K),
      gguf_w,y_fp32+size_t(t)*N,K,N);
  return cudaGetLastError();
}

} // namespace hermes::prefill_exact
