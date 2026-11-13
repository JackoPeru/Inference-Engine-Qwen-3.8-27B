// test_attn: T=32 hd=64 random, GPU vs CPU. PASS se err < 3e-2.
#include "hermes/attn.hpp"
#include "hermes/rmsnorm.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
using namespace hermes;
int main(){
  const int T=32,hd=64; const float sc=1.0f/sqrtf((float)hd);
  uint16_t *q=new uint16_t[T*hd],*k=new uint16_t[T*hd],*v=new uint16_t[T*hd],
           *oc=new uint16_t[T*hd],*og=new uint16_t[T*hd];
  srand(5);
  for(int i=0;i<T*hd;++i){ q[i]=f32_to_bf16(((float)rand()/RAND_MAX-0.5f)*0.5f); k[i]=f32_to_bf16(((float)rand()/RAND_MAX-0.5f)*0.5f); v[i]=f32_to_bf16(((float)rand()/RAND_MAX-0.5f)*0.5f); }
  attn_causal_cpu(q,k,v,oc,T,hd,sc);
  uint16_t *dq,*dk,*dv,*do_;
  cudaMalloc(&dq,T*hd*2); cudaMalloc(&dk,T*hd*2); cudaMalloc(&dv,T*hd*2); cudaMalloc(&do_,T*hd*2);
  cudaMemcpy(dq,q,T*hd*2,cudaMemcpyHostToDevice); cudaMemcpy(dk,k,T*hd*2,cudaMemcpyHostToDevice); cudaMemcpy(dv,v,T*hd*2,cudaMemcpyHostToDevice);
  attn_causal_cuda(dq,dk,dv,do_,T,hd,sc); cudaDeviceSynchronize();
  cudaMemcpy(og,do_,T*hd*2,cudaMemcpyDeviceToHost);
  float mx=0; for(int i=0;i<T*hd;++i){ float e=fabsf(bf16_to_f32(oc[i])-bf16_to_f32(og[i])); if(e>mx)mx=e; }
  printf("attn T=%d hd=%d max_abs_err=%.6f %s\n",T,hd,mx,mx<3e-2?"PASS":"FAIL");
  return mx<3e-2?0:1;
}
