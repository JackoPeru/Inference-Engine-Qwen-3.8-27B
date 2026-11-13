#include "hermes/nvfp4_gemv_q8_1_experimental.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <climits>
#include <cstdint>
#include <limits>

namespace hermes::q8_1_experimental {
namespace {

// Exactly the GGUF UE4M3 raw scale convention. 0x7f is reserved -> zero.
__device__ __forceinline__ float gguf_ue4m3(unsigned c) {
  if(c==0u || c==0x7fu) return 0.f;
  const unsigned e=c>>3, m=c&7u;
  return e==0u ? float(m)*0x1p-9f
               : __uint_as_float(((e+120u)<<23)|(m<<20));
}

// Twice the E2M1 value: an EXACT signed integer in [-12,12].
__device__ __forceinline__ int fp4_twice_int(unsigned code) {
  const unsigned mag=code&7u;
  const int a=mag==0u ? 0 : mag==1u ? 1 : mag==2u ? 2
    : mag==3u ? 3 : mag==4u ? 4 : mag==5u ? 6
    : mag==6u ? 8 : 12;
  return (code&8u) ? -a : a;
}

// One warp per 32 floats; all lanes participate in max and sum reduction.
// This intentionally follows llama.cpp quantize.cu instead of using the
// already-FP16 scale for the q integer rounding.
__global__ void quantize_32_kernel(
    const float* __restrict__ x, Q8_1Block* __restrict__ out,
    size_t nblocks) {
  constexpr unsigned FULL=0xffffffffu;
  const unsigned lane=threadIdx.x&31u;
  const unsigned warp=threadIdx.x>>5;
  const size_t ib=size_t(blockIdx.x)*8u+warp;
  if(ib>=nblocks) return; // entire warp exits; no partial warp shuffles
  const float xi=x[ib*32+lane];
  float amax=fabsf(xi);
  float sum=xi;
  #pragma unroll
  for(int step=16;step>0;step>>=1) {
    // Exact source warp_reduce_max/sum pattern is XOR, NOT down:
    // ggml-cuda/common.cuh, float overload on SM120.
    amax=fmaxf(amax,__shfl_xor_sync(FULL,amax,step));
    sum+=__shfl_xor_sync(FULL,sum,step);
  }
  const float max32=__shfl_sync(FULL,amax,0);
  const float d_f32=max32/127.0f;
  // CUDA's roundf() is ties-away-from-zero; __float2int_rn() alone is not.
  // The extra RZ conversion is exact because roundf has integral result.
  const int q=max32==0.f ? 0 : __float2int_rz(roundf(xi/d_f32));
  out[ib].qs[lane]=static_cast<int8_t>(q);
  if(lane==0) {
    const __half d16=__float2half_rn(d_f32);
    const __half s16=__float2half_rn(sum);
    out[ib].d=__half_as_ushort(d16);
    out[ib].s=__half_as_ushort(s16);
  }
}

// Four warps/CTA, each warp owns one output row. Each GGUF weight byte
// contributes two E2M1 weights in positions k and k+8. The int32 dot
// over each 16-weight scale block is exact and accumulated by 8 lanes.
// Then the Q8_1 FP16 d, GGUF UE4M3 scale and 1/2 factor are applied in FP32.
__global__ void gemv_q8_nvfp4_kernel(
    const Q8_1Block* __restrict__ xq, const uint8_t* __restrict__ gguf,
    float* __restrict__ y, int T, int N, int K) {
  constexpr unsigned FULL=0xffffffffu;
  const unsigned lane=threadIdx.x&31u;
  const int row=int(blockIdx.x)*4+(int(threadIdx.x)>>5);
  const int t=int(blockIdx.y);
  if(row>=N || t>=T) return; // uniform across each warp
  const int blocks_per_row=K/64;
  const int sub=int(lane)>>3, j=int(lane)&7;
  const Q8_1Block* const qr=xq+size_t(t)*(K/32);
  const uint8_t* const wr=gguf+size_t(row)*blocks_per_row*36;
  float acc=0.f;
  for(int b=0;b<blocks_per_row;++b) {
    const uint8_t* block=wr+size_t(b)*36;
    unsigned scales=lane==0u?*reinterpret_cast<const uint32_t*>(block):0u;
    scales=__shfl_sync(FULL,scales,0);
    const float sf=gguf_ue4m3((scales>>(sub*8))&0xffu);
    const unsigned packed=block[4+lane];
    const Q8_1Block& q8=qr[b*2+(sub>>1)];
    const int index=(sub&1)*16+j;
    const int8_t x0=q8.qs[index];
    const int8_t x1=q8.qs[index+8];
    int dot=int(x0)*fp4_twice_int(packed&15u)
           +int(x1)*fp4_twice_int(packed>>4);
    #pragma unroll
    for(int step=4;step>0;step>>=1)
      dot+=__shfl_down_sync(FULL,dot,step,8);
    // Only the first lane of each 8-lane group has the complete int sum.
    if((lane&7u)==0u) {
      const float d=__half2float(__ushort_as_half(q8.d));
      const float piece=float(dot)*(sf*0.5f)*d;
      acc+=piece;
    }
  }
  // Four active contributors at lanes 0,8,16,24 (one per GGUF scale);
  // other lanes hold zero, so cross-warp reduction remains well defined.
  acc+=__shfl_down_sync(FULL,acc,16);
  acc+=__shfl_down_sync(FULL,acc,8);
  if(lane==0)y[size_t(t)*N+row]=acc;
}

// Encode four raw GGUF E2M1 nibbles as signed int8 (twice their exact
// E2M1 value). A lane owns one aligned uint32 with four packed FP4 BYTES:
// low nibbles represent j..j+3; high nibbles represent j+8..j+11.
// Both sets can be fed directly to __dp4a alongside four Q8 integers.
__device__ __forceinline__ int fp4_x2_packed4(unsigned raw,
                                              bool high_nibble) {
  unsigned packed=0u;
  #pragma unroll
  for(int i=0;i<4;++i) {
    unsigned q=(raw>>(i*8+(high_nibble?4:0)))&15u;
    // Casting through uint8_t preserves the two's-complement signed
    // byte needed by the signed __dp4a intrinsic.
    packed |= unsigned(static_cast<uint8_t>(
      static_cast<int8_t>(fp4_twice_int(q)))) << (i*8);
  }
  return static_cast<int>(packed);
}

// Fast isolated DP4A experiment: 4 warps/CTA, one output row per warp,
// eight Q8_1 activation rows per warp. 32 lanes decode FOUR 64-weight
// GGUF blocks in parallel: lane>>3 selects a block, lane&7 selects one
// 4-value nibble group within a 16-value UE4M3 subscale.
// Each weight byte is fetched once and reused across up to eight tokens.
// K must be divisible by 256 (four 64-weight blocks per warp iteration).
// No scratch besides caller-owned packed Q8_1 activation blocks.
template<int TOKENS_PER_WARP=8>
__global__ void gemv_q8_nvfp4_dp4a_batched_kernel(
    const Q8_1Block* __restrict__ xq, const uint8_t* __restrict__ gguf,
    float* __restrict__ y, int T, int N, int K) {
  constexpr unsigned FULL=0xffffffffu;
  const int lane=int(threadIdx.x)&31;
  const int row=int(blockIdx.x)*4+(int(threadIdx.x)>>5);
  const int first_t=int(blockIdx.y)*TOKENS_PER_WARP;
  if(row>=N || first_t>=T) return; // entire warp exits uniformly

  const int group=lane>>3;             // 0..3 GGUF 64-weight blocks
  const int sub=(lane&7)>>1;          // 0..3 UE4M3 subscale in that block
  const int chunk=lane&1;             // 0..3 or 4..7 nibble byte positions
  const int qoffset=(sub&1)*16+chunk*4;
  const int qblock_half=sub>>1;
  const int n64=K/64;
  const int n32=K/32;
  const uint8_t* const wr=gguf+size_t(row)*n64*36;

  float accum[TOKENS_PER_WARP]={};
  for(int bbase=0;bbase<n64;bbase+=4) {
    const int block_index=bbase+group;
    const uint8_t* const w=wr+size_t(block_index)*36;
    // 32-lane warp reads 32 consecutive four-byte GGUF weight groups,
    // distributed over four 64-value GGUF blocks. No weight cache.
    const unsigned packed_fp4=*reinterpret_cast<const unsigned*>(
      w+4+sub*8+chunk*4);
    const int wi_lo=fp4_x2_packed4(packed_fp4,false);
    const int wi_hi=fp4_x2_packed4(packed_fp4,true);
    const float sf=gguf_ue4m3(w[sub])*0.5f;

    #pragma unroll
    for(int j=0;j<TOKENS_PER_WARP;++j) {
      const int t=first_t+j;
      if(t<T) {
        const Q8_1Block& q=xq[size_t(t)*n32+block_index*2+qblock_half];
        const int qa_lo=*reinterpret_cast<const int*>(q.qs+qoffset);
        const int qa_hi=*reinterpret_cast<const int*>(q.qs+qoffset+8);
        const int dot=__dp4a(wi_hi,qa_hi,__dp4a(wi_lo,qa_lo,0));
        const float d=__half2float(__ushort_as_half(q.d));
        // Retain the FP16 Q8 scale and exact GGUF code semantics;
        // float reassociation vs llama and baseline is NOT assumed exact.
        accum[j]+=float(dot)*(sf*d);
      }
    }
  }

  // One float sum across 32 lanes per (output row, token).
  #pragma unroll
  for(int off=16;off>0;off>>=1) {
    #pragma unroll
    for(int j=0;j<TOKENS_PER_WARP;++j)
      accum[j]+=__shfl_down_sync(FULL,accum[j],off);
  }
  if(lane==0) {
    #pragma unroll
    for(int j=0;j<TOKENS_PER_WARP;++j) {
      const int t=first_t+j;
      if(t<T)y[size_t(t)*N+row]=accum[j];
    }
  }
}

} // namespace

size_t q8_1_scratch_bytes(int T,int K) {
  if(T<=0 || K<=0 || K%64!=0) return 0;
  const size_t block_count=size_t(T)*size_t(K/32);
  if(block_count>std::numeric_limits<size_t>::max()/sizeof(Q8_1Block))
    return 0;
  return block_count*sizeof(Q8_1Block);
}

cudaError_t quantize_f32_q8_1(
    const float* x_f32,Q8_1Block* q8_scratch,int T,int K,cudaStream_t stream) {
  if(!x_f32 || !q8_scratch || !q8_1_scratch_bytes(T,K))
    return cudaErrorInvalidValue;
  const size_t blocks=size_t(T)*size_t(K/32);
  const size_t ctas=(blocks+7u)/8u;
  if(ctas>unsigned(INT_MAX))return cudaErrorInvalidConfiguration;
  quantize_32_kernel<<<unsigned(ctas),256,0,stream>>>(
    x_f32,q8_scratch,blocks);
  return cudaGetLastError();
}

cudaError_t gemv_q8_1_nvfp4(
    const Q8_1Block* x_q8,const uint8_t* gguf_nvfp4_w,
    float* y_f32,int T,int N,int K,cudaStream_t stream) {
  if(!x_q8 || !gguf_nvfp4_w || !y_f32 || N<=0 ||
     !q8_1_scratch_bytes(T,K))
    return cudaErrorInvalidValue;
  const size_t grid_x=(size_t(N)+3)/4;
  if(grid_x>unsigned(INT_MAX) || T>65535)return cudaErrorInvalidConfiguration;
  gemv_q8_nvfp4_kernel<<<dim3(unsigned(grid_x),unsigned(T)),128,0,stream>>>(
      x_q8,gguf_nvfp4_w,y_f32,T,N,K);
  return cudaGetLastError();
}

cudaError_t gemv_q8_1_nvfp4_dp4a_batched(
    const Q8_1Block* x_q8,const uint8_t* gguf_nvfp4_w,
    float* y_f32,int T,int N,int K,cudaStream_t stream) {
  // Fail closed instead of silently falling back: callers explicitly
  // opt in and use the original gemv_q8_1_nvfp4 for A/B and rollback.
  if(!x_q8 || !gguf_nvfp4_w || !y_f32 || N<=0 ||
     !q8_1_scratch_bytes(T,K) || K%256!=0 ||
     (reinterpret_cast<uintptr_t>(gguf_nvfp4_w)&3u)!=0u)
    return cudaErrorInvalidValue;
  constexpr int TILE=8;
  const size_t gx=(size_t(N)+3)/4;
  const size_t gy=(size_t(T)+TILE-1)/TILE;
  if(gx>unsigned(INT_MAX) || gy>65535u)return cudaErrorInvalidConfiguration;
  gemv_q8_nvfp4_dp4a_batched_kernel<TILE>
    <<<dim3(unsigned(gx),unsigned(gy)),128,0,stream>>>(
      x_q8,gguf_nvfp4_w,y_f32,T,N,K);
  return cudaGetLastError();
}

cudaError_t project_f32_q8_1_nvfp4(
    const float* x_f32,const uint8_t* gguf_nvfp4_w,
    Q8_1Block* q8_scratch,float* y_f32,
    int T,int N,int K,cudaStream_t stream) {
  const cudaError_t rc=quantize_f32_q8_1(x_f32,q8_scratch,T,K,stream);
  if(rc!=cudaSuccess)return rc;
  return gemv_q8_1_nvfp4(q8_scratch,gguf_nvfp4_w,y_f32,T,N,K,stream);
}

cudaError_t project_f32_q8_1_nvfp4_dp4a_batched(
    const float* x_f32,const uint8_t* gguf_nvfp4_w,
    Q8_1Block* q8_scratch,float* y_f32,int T,int N,int K,
    cudaStream_t stream) {
  // Validate geometry before queuing quantization: no partial work when
  // the weight GEMV shape is incompatible with this experimental kernel.
  if(N<=0 || K%256!=0 ||
     !x_f32 || !gguf_nvfp4_w || !q8_scratch || !y_f32)
    return cudaErrorInvalidValue;
  const cudaError_t rc=quantize_f32_q8_1(x_f32,q8_scratch,T,K,stream);
  if(rc!=cudaSuccess)return rc;
  return gemv_q8_1_nvfp4_dp4a_batched(q8_scratch,gguf_nvfp4_w,y_f32,T,N,K,stream);
}

} // namespace hermes::q8_1_experimental
