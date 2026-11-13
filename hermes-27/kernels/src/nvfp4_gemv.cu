#include "hermes/nvfp4_gemv.hpp"
#include "hermes/rmsnorm.hpp"
#include <cuda_bf16.h>
#include <cmath>
#include <cstring>
namespace hermes {
// local UE4M3 (host): raw*0.5 convention, same semantics as model/gguf decoder
static inline float h_ue4m3(uint8_t x){
  if(x==0||x==0x7F) return 0.0f;
  int e=(x>>3)&0xF,m=x&0x7; float r;
  if(e==0) r=ldexpf((float)m,-9); else r=ldexpf(1.0f+(float)m/8.0f,e-7);
  return r*0.5f;
}
// E2M1 true values
static inline float e2m1(int v){
  static const float lut[16]={0,0.5f,1,1.5f,2,3,4,6,0,-0.5f,-1,-1.5f,-2,-3,-4,-6};
  return lut[v&15];
}
uint8_t fp32_to_ue4m3(float x){
  if(!(x>0.0f)) return 0;
  if(x>448.0f) x=448.0f;
  uint32_t bits; memcpy(&bits,&x,4);
  int fp_exp=((bits>>23)&0xFF)-127, fp_man=(bits>>20)&0x7;
  int ue_exp=fp_exp+7;
  if(ue_exp<=0){ int m=(int)(x*512.0f+0.5f); if(m>7)m=7; if(m<1)return 0; return (uint8_t)m; }
  if(ue_exp>=15) return 0x7E;
  int round_bit=(bits>>19)&1, ue_man=fp_man+round_bit;
  if(ue_man>7){ ue_man=0; if(++ue_exp>=15) return 0x7E; }
  return (uint8_t)((ue_exp<<3)|ue_man);
}
void nvfp4_quant_row(const float* w, uint8_t* out, int H){
  for(int b=0;b<H/64;++b) for(int s=0;s<4;++s){
    float amax=0; for(int j=0;j<16;++j){ float a=fabsf(w[b*64+s*16+j]); if(a>amax)amax=a; }
    float raw=amax/6.0f;
    out[b*36+s]=fp32_to_ue4m3(raw);
    float rs=h_ue4m3(out[b*36+s])*2.0f; // true scale
    if(rs==0){ memset(out+b*36+4+s*8,0,8); continue; }
    for(int j=0;j<8;++j){
      float v0=w[b*64+s*16+j]/rs, v1=w[b*64+s*16+j+8]/rs;
      int b0=0,b1=0; float bd0=1e30f,bd1=1e30f;
      for(int q=0;q<16;++q){ float e=e2m1(q); float d0=fabsf(v0-e),d1=fabsf(v1-e); if(d0<bd0){bd0=d0;b0=q;} if(d1<bd1){bd1=d1;b1=q;} }
      out[b*36+4+s*8+j]=(uint8_t)(b0|(b1<<4));
    }
  }
}
float nvfp4_gemv_cpu(const uint16_t* x, const uint8_t* row, int H){
  double acc=0;
  for(int b=0;b<H/64;++b) for(int s=0;s<4;++s){
    float rs=h_ue4m3(row[b*36+s])*2.0f;
    for(int j=0;j<8;++j){
      uint8_t q=row[b*36+4+s*8+j];
      acc+=double(bf16_to_f32(x[b*64+s*16+j]))*double(rs*e2m1(q&15));
      acc+=double(bf16_to_f32(x[b*64+s*16+j+8]))*double(rs*e2m1(q>>4));
    }
  }
  return (float)acc;
}
__device__ __forceinline__ float d_e2m1(int v){
  switch(v&15){case 0:return 0;case 1:return .5f;case 2:return 1;case 3:return 1.5f;
    case 4:return 2;case 5:return 3;case 6:return 4;case 7:return 6;case 8:return 0;
    case 9:return -.5f;case 10:return -1;case 11:return -1.5f;case 12:return -2;
    case 13:return -3;case 14:return -4;default:return -6;}
}
__device__ __forceinline__ float d_ue4m3(uint8_t x){
  if(x==0||x==0x7F) return 0;
  int e=(x>>3)&15,m=x&7,rm;
  float r; if(e==0) r=ldexpf((float)m,-9); else r=ldexpf(1+(float)m/8,e-7);
  return r; // raw scale (no *0.5); NVFP4 val = raw*e2m1
}
__global__ void nvfp4_gemv_k(const __nv_bfloat16* x, const uint8_t* Wq, float* y, int H, int F){
  int i=(int)blockIdx.x; if(i>=F) return;
  const uint8_t* row=Wq+(size_t)i*(H/64*36);
  float acc=0;
  for(int b=threadIdx.x;b<H/64;b+=blockDim.x) for(int s=0;s<4;++s){
    float rs=d_ue4m3(row[b*36+s]);
    for(int j=0;j<8;++j){
      uint8_t q=row[b*36+4+s*8+j];
      acc+=(float)x[b*64+s*16+j]*rs*d_e2m1(q&15);
      acc+=(float)x[b*64+s*16+j+8]*rs*d_e2m1(q>>4);
    }
  }
  __shared__ float sh[256];
  sh[threadIdx.x]=acc; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
  if(threadIdx.x==0) y[i]=sh[0];
}
// One warp per GGUF output row, four rows per CTA. The 32 lanes read one
// contiguous 32-byte E2M1 payload from each 36-byte GGUF block. Each byte
// stores two weights separated by eight elements in its 16-value sub-block.
// All lanes of a participating warp stay active through the reductions.
__device__ __forceinline__ float d_e2m1_warp(unsigned q){
  unsigned m=q&7u;
  // E2M1: {0, .5, 1, 1.5, 2, 3, 4, 6}, with the sign in bit 3.
  unsigned mant=(m&1u) & (m!=1u);
  unsigned bits=((126u+(m>>1))<<23) | (mant<<22);
  bits &= 0u-(unsigned)(m!=0u);
  bits |= (q&8u)<<28;
  return __uint_as_float(bits);
}
__device__ __forceinline__ float d_ue4m3_warp(unsigned q){
  if(q==0u || q==0x7fu) return 0.0f;
  unsigned e=q>>3, m=q&7u;
  if(e==0u) return (float)m * 0x1p-9f;
  return __uint_as_float(((e+120u)<<23) | (m<<20));
}
__global__ void nvfp4_gemv_warp4_k(const __nv_bfloat16* __restrict__ x,
                                    const uint8_t* __restrict__ Wq,
                                    float* __restrict__ y, int H, int F){
  const int lane=(int)threadIdx.x&31;
  const int row=(int)blockIdx.x*4+((int)threadIdx.x>>5);
  if(row>=F) return; // whole warp exits; no inter-warp synchronization
  const int nb=H/64;
  const uint8_t* wrow=Wq+(size_t)row*nb*36;
  float acc=0.0f;
  const int sub=lane>>3, j=lane&7;
  for(int b=0;b<nb;++b){
    const uint8_t* block=wrow+(size_t)b*36;
    // GGUF rows and blocks are four-byte aligned, so one lane can fetch all
    // four scales; warp shuffle broadcasts them without per-lane byte loads.
    unsigned packed=(lane==0) ? *reinterpret_cast<const unsigned*>(block) : 0u;
    packed=__shfl_sync(0xffffffffu,packed,0);
    float scale=d_ue4m3_warp((packed>>(sub*8))&0xffu);
    unsigned q=block[4+lane];
    int k=b*64+sub*16+j;
    float a=(float)x[k], c=(float)x[k+8];
    acc=fmaf(scale, a*d_e2m1_warp(q&15u)+c*d_e2m1_warp(q>>4),acc);
  }
  for(int delta=16;delta;delta>>=1)
    acc+=__shfl_down_sync(0xffffffffu,acc,delta);
  if(lane==0) y[row]=acc;
}
void nvfp4_gemv_cuda_variant(const uint16_t* x, const uint8_t* Wq, float* y,
                            int H, int F, Nvfp4GemvVariant variant,
                            cudaStream_t stream){
  if(variant==Nvfp4GemvVariant::Warp4 && H>=64 && H%64==0 && F>0 &&
     (reinterpret_cast<uintptr_t>(Wq)&3u)==0u){
    nvfp4_gemv_warp4_k<<<(F+3)/4,128,0,stream>>>(
        (const __nv_bfloat16*)x,Wq,y,H,F);
  } else {
    nvfp4_gemv_k<<<F,256,0,stream>>>((const __nv_bfloat16*)x,Wq,y,H,F);
  }
}
void nvfp4_gemv_cuda(const uint16_t* x, const uint8_t* Wq, float* y, int H, int F){
  nvfp4_gemv_cuda_variant(x,Wq,y,H,F,Nvfp4GemvVariant::Scalar);
}
}
