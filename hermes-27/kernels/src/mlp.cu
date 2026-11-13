#include "hermes/mlp.hpp"
#include "hermes/rmsnorm.hpp"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
namespace hermes {
static inline float silu(float v){ return v/(1.0f+expf(-v)); }
void mlp_cpu(const uint16_t* x, const uint16_t* Wg, const uint16_t* Wu,
             const uint16_t* Wd, uint16_t* y, int h, int f){
  for(int j=0;j<h;++j){
    double acc=0;
    for(int i=0;i<f;++i){
      double gg=0,uu=0;
      for(int k=0;k<h;++k){ float xv=bf16_to_f32(x[k]); gg+=double(xv)*double(bf16_to_f32(Wg[k*f+i])); uu+=double(xv)*double(bf16_to_f32(Wu[k*f+i])); }
      float g=(float)gg; float s=silu(g)*(float)uu;
      acc+=double(s)*double(bf16_to_f32(Wd[i*h+j]));
    }
    y[j]=f32_to_bf16((float)acc);
  }
}
// gate/up fused: one thread per f-col
__global__ void mlp_gate_k(const __nv_bfloat16* x, const __nv_bfloat16* Wg,
                           const __nv_bfloat16* Wu, __nv_bfloat16* hbuf, int H, int F){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i>=F) return;
  float gg=0,uu=0;
  for(int k=0;k<H;++k){ float xv=(float)x[k]; gg+=xv*(float)Wg[k*F+i]; uu+=xv*(float)Wu[k*F+i]; }
  float s=(gg/(1.0f+expf(-gg)))*uu;
  hbuf[i]=__nv_bfloat16(s);
}
// down proj: one thread per h-col
__global__ void mlp_down_k(const __nv_bfloat16* hbuf, const __nv_bfloat16* Wd, __nv_bfloat16* y, int H, int F){
  int j=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(j>=H) return;
  float acc=0;
  for(int i=0;i<F;++i) acc+=(float)hbuf[i]*(float)Wd[i*H+j];
  y[j]=__nv_bfloat16(acc);
}
void mlp_cuda(const uint16_t* x, const uint16_t* Wg, const uint16_t* Wu,
              const uint16_t* Wd, uint16_t* y, int h, int f){
  __nv_bfloat16* hbuf=nullptr;
  cudaMalloc(&hbuf,(size_t)f*sizeof(__nv_bfloat16));
  int blk=256;
  mlp_gate_k<<<(f+blk-1)/blk,blk>>>((const __nv_bfloat16*)x,(const __nv_bfloat16*)Wg,(const __nv_bfloat16*)Wu,hbuf,h,f);
  mlp_down_k<<<(h+blk-1)/blk,blk>>>(hbuf,(const __nv_bfloat16*)Wd,(__nv_bfloat16*)y,h,f);
  cudaFree(hbuf);
}
}
