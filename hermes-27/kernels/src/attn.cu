#include "hermes/attn.hpp"
#include "hermes/rmsnorm.hpp"
#include <cuda_bf16.h>
#include <cmath>
#include <vector>
namespace hermes {
void attn_causal_cpu(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                     uint16_t* out, int T, int hd, float scale){
  std::vector<float> s(T);
  for(int t=0;t<T;++t){
    float mx=-1e30f;
    for(int t2=0;t2<=t;++t2){ float d=0; for(int d_=0;d_<hd;++d_) d+=bf16_to_f32(q[t*hd+d_])*bf16_to_f32(k[t2*hd+d_]); s[t2]=d*scale; if(s[t2]>mx)mx=s[t2]; }
    float den=0; for(int t2=0;t2<=t;++t2){ s[t2]=expf(s[t2]-mx); den+=s[t2]; }
    for(int d_=0;d_<hd;++d_){ float a=0; for(int t2=0;t2<=t;++t2) a+=s[t2]/den*bf16_to_f32(v[t2*hd+d_]); out[t*hd+d_]=f32_to_bf16(a); }
  }
}
__global__ void attn_k(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
                       __nv_bfloat16* out, int T, int HD, float scale){
  int t=(int)blockIdx.x; int d_=(int)threadIdx.x;
  if(t>=T||d_>=HD) return;
  extern __shared__ float sh[]; // size T
  int tid=threadIdx.x;
  // scores computed by thread 0 of block, broadcast via shared
  if(tid==0){
    float mx=-1e30f;
    for(int t2=0;t2<=t;++t2){ float dd=0; for(int j=0;j<HD;++j) dd+=(float)q[t*HD+j]*(float)k[t2*HD+j]; sh[t2]=dd*scale; if(sh[t2]>mx)mx=sh[t2]; }
    float den=0; for(int t2=0;t2<=t;++t2){ sh[t2]=expf(sh[t2]-mx); den+=sh[t2]; }
    for(int t2=0;t2<=t;++t2) sh[t2]/=den;
  }
  __syncthreads();
  float a=0; for(int t2=0;t2<=t;++t2) a+=sh[t2]*(float)v[t2*HD+d_];
  out[t*HD+d_]=__nv_bfloat16(a);
}
void attn_causal_cuda(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                      uint16_t* out, int T, int hd, float scale){
  attn_k<<<T,hd,(size_t)T*sizeof(float)>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,
    (const __nv_bfloat16*)v,(__nv_bfloat16*)out,T,hd,scale);
}
}
