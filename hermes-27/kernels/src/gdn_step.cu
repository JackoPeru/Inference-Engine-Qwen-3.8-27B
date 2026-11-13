#include "hermes/gdn_step.hpp"
#include <cmath>
namespace hermes {
void gdn_step_cpu(const float* q, const float* k, const float* v,
                  const float* beta, const float* gl,
                  float* S, float* o, int H, int K, int V){
  for(int h=0;h<H;++h){
    float dec=expf(gl[h]), be=beta[h];
    float* Sh=S+h*K*V; const float* qh=q+h*K,*kh=k+h*K,*vh=v+h*V;
    for(int i=0;i<K*V;++i) Sh[i]*=dec;
    for(int j=0;j<V;++j){ double d=0; for(int i=0;i<K;++i) d+=double(Sh[i*V+j])*double(kh[i]);
      double bv=(double(vh[j])-d)*double(be);
      for(int i=0;i<K;++i) Sh[i*V+j]+=float(double(kh[i])*bv); }
    for(int j=0;j<V;++j){ double a=0; for(int i=0;i<K;++i) a+=double(Sh[i*V+j])*double(qh[i]); o[h*V+j]=(float)a; }
  }
}
__global__ void gdn_step_k(const float* q, const float* k, const float* v,
                           const float* beta, const float* gl,
                           float* S, float* o, int H, int K, int V){
  int h=(int)blockIdx.x; if(h>=H) return;
  int j=(int)threadIdx.x;
  extern __shared__ float sh[]; float* sk=sh; float* sq=sh+K;
  for(int i=(int)threadIdx.x;i<K;i+=blockDim.x){ sk[i]=k[h*K+i]; sq[i]=q[h*K+i]; }
  __syncthreads();
  // FIX: predicate work instead of early-return. All threads must reach
  // the __syncthreads() below; returning when j>=V hangs when V<blockDim.
  bool active = (j < V);
  float dec=expf(gl[h]), be=beta[h];
  float* Sh=S+(size_t)h*K*V;
  // scale column j
  if(active){
    for(int i=0;i<K;++i) Sh[i*V+j]*=dec;
  }
  __syncthreads(); // columns share rows; scale is per-element independent, no sync needed but harmless
  if(active){
    double d=0; for(int i=0;i<K;++i) d+=double(Sh[i*V+j])*double(sk[i]);
    double bv=(double(v[h*V+j])-d)*double(be);
    for(int i=0;i<K;++i) Sh[i*V+j]+=float(double(sk[i])*bv);
    double a=0; for(int i=0;i<K;++i) a+=double(Sh[i*V+j])*double(sq[i]);
    o[h*V+j]=(float)a;
  }
}
void gdn_step_cuda(const float* q, const float* k, const float* v,
                   const float* beta, const float* gl,
                   float* S, float* o, int H, int K, int V,
                   cudaStream_t stream){
  size_t shm=(size_t)(K+K)*sizeof(float);
  gdn_step_k<<<H,128,shm,stream>>>(q,k,v,beta,gl,S,o,H,K,V);
}
}
