#include "hermes/nvfp4_prefill_exact.hpp"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>

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

// Branch-free E2M1 float32 bitcast for the vectorized unpack only.
// Preserve +0 for nibble 0x8 (-0) so the resulting BF16 words match the
// original GGUF decoder *bit for bit* (not only as equal floats).
__device__ __forceinline__ float e2m1_exact_bits(unsigned q) {
  unsigned m=q&7u;
  unsigned fraction=(m&1u)&(m!=1u);
  unsigned bits=((126u+(m>>1))<<23) | (fraction<<22) | ((q&8u)<<28);
  bits &= 0u-(unsigned)(m!=0u);
  return __uint_as_float(bits);
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

// GGUF 64-value block: [4 UE4M3 scale bytes, 32 E2M1 packed bytes].
// One warp loads the 32 packed bytes exactly once. Each lane writes a
// *contiguous* 32-bit BF16 pair, rather than two split-half 16-bit stores.
// Reconstruct [0..7,8..15] nibble pairing via shfl; never requantize W.
// PER_WARP lets one warp process multiple blocks in a loop, reducing CTA
// scheduling overhead. Output W remains row-major [N,K].
template<int PER_WARP>
__global__ void unpack_vector_k(const uint8_t* __restrict__ gguf,
                                uint16_t* __restrict__ dst,
                                int K, size_t total_blocks) {
  const int lane=int(threadIdx.x)&31;
  const int warp=int(threadIdx.x)>>5;
  constexpr int WARPS_PER_CTA=8;
  const size_t first=(size_t(blockIdx.x)*WARPS_PER_CTA+size_t(warp))*PER_WARP;
  #pragma unroll
  for(int it=0;it<PER_WARP;++it) {
    size_t block=first+it;
    if(block>=total_blocks) continue; // uniform within this warp
    const uint8_t* src=gguf+block*36;
    unsigned packed_scales=lane==0 ? *reinterpret_cast<const uint32_t*>(src) : 0u;
    packed_scales=__shfl_sync(0xffffffffu,packed_scales,0);
    unsigned byte=src[4+lane];
    const int sub=lane>>3, pair=lane&7;
    const int source_lane=sub*8+(pair&3)*2;
    unsigned q0=__shfl_sync(0xffffffffu,byte,source_lane);
    unsigned q1=__shfl_sync(0xffffffffu,byte,source_lane+1);
    if(pair>=4) {q0>>=4;q1>>=4;}
    const float sf=ue4m3((packed_scales>>(sub*8))&0xffu);
    const __nv_bfloat16 lo=__float2bfloat16_rn(sf*e2m1_exact_bits(q0&15u));
    const __nv_bfloat16 hi=__float2bfloat16_rn(sf*e2m1_exact_bits(q1&15u));
    const unsigned bf16pair=unsigned(__bfloat16_as_ushort(lo)) |
                            (unsigned(__bfloat16_as_ushort(hi))<<16);
    // Block corresponds to contiguous 64 weights, since K%64=0.
    reinterpret_cast<unsigned*>(dst)[block*32+lane]=bf16pair;
  }
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

cudaError_t dequantize_gguf_to_bf16_vector(
    const uint8_t* gguf_w,uint16_t* weight_bf16,
    int N,int K,cudaStream_t stream) {
  if(!gguf_w || !weight_bf16 || !bf16_weight_scratch_bytes(N,K))
    return cudaErrorInvalidValue;
  constexpr int PER_WARP=4;
  constexpr int CTA_WARPS=8;
  const size_t blocks=size_t(N)*size_t(K/64);
  const size_t ctas=(blocks+PER_WARP*CTA_WARPS-1)/(PER_WARP*CTA_WARPS);
  if(ctas>size_t(std::numeric_limits<int>::max()))return cudaErrorInvalidConfiguration;
  unpack_vector_k<PER_WARP><<<unsigned(ctas),256,0,stream>>>(
      gguf_w,weight_bf16,K,blocks);
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

ReusableBF16WeightScratch::~ReusableBF16WeightScratch() {
  (void)shutdown();
}

ScratchResult ReusableBF16WeightScratch::init(
    void* buffer,size_t capacity_bytes) {
  std::lock_guard<std::mutex> lock(submit_mutex_);
  if(buffer_ || handle_ || ready_event_ || poisoned_)
    return {ScratchCode::InvalidArguments};
  if(!buffer || capacity_bytes<2 || (reinterpret_cast<uintptr_t>(buffer)&15u))
    return {ScratchCode::InvalidArguments};
  int device=-1;
  cudaError_t err=cudaGetDevice(&device);
  if(err!=cudaSuccess) return {ScratchCode::CudaFailure,err};
  cudaPointerAttributes attr{};
  err=cudaPointerGetAttributes(&attr,buffer);
  if(err!=cudaSuccess) return {ScratchCode::CudaFailure,err};
  if(attr.device!=device || attr.type!=cudaMemoryTypeDevice)
    return {ScratchCode::WrongDevice};

  cudaEvent_t ev=nullptr;
  err=cudaEventCreateWithFlags(&ev,cudaEventDisableTiming);
  if(err!=cudaSuccess) return {ScratchCode::CudaFailure,err};
  cublasHandle_t blas=nullptr;
  cublasStatus_t st=cublasCreate(&blas);
  if(st!=CUBLAS_STATUS_SUCCESS) {
    cudaEventDestroy(ev);
    return {ScratchCode::CublasFailure,cudaSuccess,st};
  }
  buffer_=reinterpret_cast<uint16_t*>(buffer);
  capacity_bytes_=capacity_bytes;
  device_=device;
  ready_event_=ev;
  handle_=blas;
  event_recorded_=false;
  poisoned_=false;
  return {};
}

ScratchResult ReusableBF16WeightScratch::enqueue(
    const uint8_t* gguf_w,const uint16_t* x_bf16,float* y_fp32,
    int T,int N,int K,cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(submit_mutex_);
  if(!buffer_ || !handle_ || !ready_event_)
    return {ScratchCode::Uninitialized};
  if(poisoned_)return {ScratchCode::Poisoned};
  if(!gguf_w || !x_bf16 || !y_fp32 || T<=0 || N<=0 ||
     K<=0 || K%64!=0 || size_t(K)>SIZE_MAX/2)
    return {ScratchCode::InvalidArguments};
  const size_t row_bytes=size_t(K)*2;
  if(capacity_bytes_<row_bytes)return {ScratchCode::InsufficientCapacity};
  // Memory objects are device-specific. This also detects a caller thread
  // moving the slot to another CUDA device after initialization.
  int active=-1;
  cudaError_t ce=cudaGetDevice(&active);
  if(ce!=cudaSuccess)return {ScratchCode::CudaFailure,ce};
  if(active!=device_)return {ScratchCode::WrongDevice};

  cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
  ce=cudaStreamIsCapturing(stream,&capture);
  if(ce!=cudaSuccess)return {ScratchCode::CudaFailure,ce};
  if(capture!=cudaStreamCaptureStatusNone)
    return {ScratchCode::InvalidArguments}; // Not CUDA-graph-safe.

  // CUDA event wait captures the most recently recorded generation. Holding
  // the mutex through the final record guarantees that next submissions see
  // a well-defined predecessor even on different host threads/streams.
  if(event_recorded_) {
    ce=cudaStreamWaitEvent(stream,ready_event_,0);
    if(ce!=cudaSuccess){poisoned_=true;return {ScratchCode::CudaFailure,ce};}
  }

  // Any failure after the first enqueue POISONS this slot. Attempt to record
  // a completion event for already queued work so shutdown can wait safely.
  auto fail_cuda=[&](cudaError_t failure)->ScratchResult {
    poisoned_=true;
    cudaError_t rec=cudaEventRecord(ready_event_,stream);
    if(rec==cudaSuccess)event_recorded_=true;
    else event_unknown_=true;
    return {ScratchCode::CudaFailure,failure};
  };
  auto fail_blas=[&](cublasStatus_t failure)->ScratchResult {
    poisoned_=true;
    cudaError_t rec=cudaEventRecord(ready_event_,stream);
    if(rec==cudaSuccess)event_recorded_=true;
    else event_unknown_=true;
    return {ScratchCode::CublasFailure,cudaSuccess,failure};
  };
  cublasStatus_t st=cublasSetStream(handle_,stream);
  if(st!=CUBLAS_STATUS_SUCCESS){
    poisoned_=true;
    return {ScratchCode::CublasFailure,cudaSuccess,st};
  }
  const float alpha=1.f,beta=0.f;

  size_t max_rows=capacity_bytes_/row_bytes;
  max_rows=std::min(max_rows,size_t(N));
  // Align tile row count for efficient Tensor Core GEMM, if possible.
  if(max_rows>=128)max_rows=(max_rows/128)*128;
  const int tile_rows=int(max_rows);
  for(int base=0;base<N;base+=tile_rows) {
    const int rows=std::min(tile_rows,N-base);
    const size_t raw_row_bytes=size_t(K/64)*36;
    ce=dequantize_gguf_to_bf16_vector(
      gguf_w+size_t(base)*raw_row_bytes,buffer_,rows,K,stream);
    if(ce!=cudaSuccess)return fail_cuda(ce);
    // Column-major Y[N,T], identical memory to C-row-major Y[T,N].
    // The base offset in Y applies to output rows, ldc remains full N.
    st=cublasGemmEx(handle_,CUBLAS_OP_T,CUBLAS_OP_N,
       rows,T,K,&alpha,
       buffer_,CUDA_R_16BF,K,
       x_bf16,CUDA_R_16BF,K,
       &beta,y_fp32+base,CUDA_R_32F,N,
       CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if(st!=CUBLAS_STATUS_SUCCESS)return fail_blas(st);
  }
  ce=cudaEventRecord(ready_event_,stream);
  if(ce!=cudaSuccess){
    poisoned_=true;
    event_unknown_=true;
    return {ScratchCode::CudaFailure,ce};
  }
  event_recorded_=true;
  return {};
}

ScratchResult ReusableBF16WeightScratch::shutdown_locked() {
  if(!buffer_ && !handle_ && !ready_event_)return {};
  int active=-1;
  cudaError_t ce=cudaGetDevice(&active);
  if(ce!=cudaSuccess)return {ScratchCode::CudaFailure,ce};
  if(active!=device_)return {ScratchCode::WrongDevice};
  if(event_unknown_) {
    // Do not wait on an OLD event generation if the new one failed to
    // record. The global sync is limited to this error-only teardown.
    ce=cudaDeviceSynchronize();
    if(ce!=cudaSuccess)return {ScratchCode::CudaFailure,ce};
  } else if(event_recorded_) {
    ce=cudaEventSynchronize(ready_event_);
    if(ce!=cudaSuccess)return {ScratchCode::CudaFailure,ce};
  } else if(poisoned_) {
    // An event record may have failed after a kernel launch. A full device
    // sync is acceptable on this *exception-only* teardown path.
    ce=cudaDeviceSynchronize();
    if(ce!=cudaSuccess)return {ScratchCode::CudaFailure,ce};
  }
  cublasStatus_t st=cublasDestroy(handle_);
  if(st!=CUBLAS_STATUS_SUCCESS)
    return {ScratchCode::CublasFailure,cudaSuccess,st};
  handle_=nullptr;
  ce=cudaEventDestroy(ready_event_);
  if(ce!=cudaSuccess)return {ScratchCode::CudaFailure,ce};
  ready_event_=nullptr;
  buffer_=nullptr;
  capacity_bytes_=0;
  device_=-1;
  event_recorded_=false;
  event_unknown_=false;
  poisoned_=false;
  return {};
}
ScratchResult ReusableBF16WeightScratch::shutdown() {
  std::lock_guard<std::mutex> lock(submit_mutex_);
  return shutdown_locked();
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
