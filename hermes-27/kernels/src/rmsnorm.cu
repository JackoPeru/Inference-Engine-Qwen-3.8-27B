#include "hermes/rmsnorm.hpp"
#include <cuda_bf16.h>
#include <cmath>
namespace hermes {
float bf16_to_f32(uint16_t b){ __nv_bfloat16 h; memcpy(&h,&b,2); return (float)h; }
uint16_t f32_to_bf16(float f){ __nv_bfloat16 h=(__nv_bfloat16)f; uint16_t b; memcpy(&b,&h,2); return b; }
void rmsnorm_cpu(const uint16_t* x, const uint16_t* w, uint16_t* y, int n, float eps){
  double ss=0; for(int i=0;i<n;++i){ float v=bf16_to_f32(x[i]); ss+=double(v)*double(v); }
  float inv=1.0f/sqrtf(float(ss/n)+eps);
  for(int i=0;i<n;++i) y[i]=f32_to_bf16(bf16_to_f32(x[i])*inv*bf16_to_f32(w[i]));
}
__global__ void rmsnorm_k(const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* y, int n, float eps){
  // one block per row, parallel reduction in shared
  extern __shared__ float sh[];
  int tid=threadIdx.x; int stride=blockDim.x;
  float partial=0;
  for(int i=tid;i<n;i+=stride){ float v=(float)x[i]; partial+=v*v; }
  sh[tid]=partial; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(tid<s) sh[tid]+=sh[tid+s]; __syncthreads(); }
  float inv=1.0f/sqrtf(sh[0]/n+eps);
  for(int i=tid;i<n;i+=stride) y[i]=__nv_bfloat16((float)x[i]*inv*(float)w[i]);
  // note: blockDim must be power of two <=1024; n=5120 covered by grid-stride
}
void rmsnorm_cuda(const uint16_t* x, const uint16_t* w, uint16_t* y, int n, float eps){
  rmsnorm_k<<<1,1024,n==0?0:1024*sizeof(float)>>>((const __nv_bfloat16*)x,(const __nv_bfloat16*)w,(__nv_bfloat16*)y,n,eps);
}
}
