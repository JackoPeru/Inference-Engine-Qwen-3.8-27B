// test_rope: head_dim 256, rotary 64 (partial 0.25), pos 1000, theta 10M. PASS se err<2e-3.
#include "hermes/rope.hpp"
#include "hermes/rmsnorm.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
using namespace hermes;
int main(){
  const int hd=256, rd=64; const long pos=1000; const float th=10000000.0f;
  uint16_t hx[256],hy_c[256],hy_g[256];
  srand(3);
  for(int i=0;i<hd;++i) hx[i]=f32_to_bf16(((float)rand()/RAND_MAX-0.5f)*2);
  rope_cpu(hx,hy_c,hd,rd,pos,th);
  // identity at pos=0 (cos=1,sin=0) must hold exactly (split-half check)
  uint16_t hz[256],hy0[256];
  for(int i=0;i<hd;++i) hz[i]=hx[i];
  rope_cpu(hz,hy0,hd,rd,0,th);
  float mx0=0; for(int i=0;i<hd;++i){ float e=fabsf(bf16_to_f32(hz[i])-bf16_to_f32(hy0[i])); if(e>mx0)mx0=e; }
  uint16_t *dx,*dy; cudaMalloc(&dx,hd*2); cudaMalloc(&dy,hd*2);
  cudaMemcpy(dx,hx,hd*2,cudaMemcpyHostToDevice);
  rope_cuda(dx,dy,hd,rd,pos,th); cudaDeviceSynchronize();
  cudaMemcpy(hy_g,dy,hd*2,cudaMemcpyDeviceToHost);
  float mx=0; for(int i=0;i<hd;++i){ float e=fabsf(bf16_to_f32(hy_c[i])-bf16_to_f32(hy_g[i])); if(e>mx)mx=e; }
  // pass-through beyond rotary_dim must hold
  float mxp=0; for(int i=rd;i<hd;++i){ float e=fabsf(bf16_to_f32(hx[i])-bf16_to_f32(hy_g[i])); if(e>mxp)mxp=e; }
  printf("rope hd=%d rd=%d gpu/cpu=%.6f pos0=%.6f passthru=%.6f %s\n",hd,rd,mx,mx0,mxp,
    (mx<2e-3&&mx0<1e-6&&mxp<1e-6)?"PASS":"FAIL");
  cudaFree(dx);cudaFree(dy);
  return (mx<2e-3&&mx0<1e-6&&mxp<1e-6)?0:1;
}
