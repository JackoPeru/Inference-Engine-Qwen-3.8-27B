// test_mlp: h=256 f=512 random BF16, GPU vs CPU. PASS se max err < 5e-2 (accum order).
#include "hermes/mlp.hpp"
#include "hermes/rmsnorm.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
using namespace hermes;
int main(){
  const int h=256,f=512;
  uint16_t *x=new uint16_t[h],*Wg=new uint16_t[(size_t)h*f],*Wu=new uint16_t[(size_t)h*f],
           *Wd=new uint16_t[(size_t)f*h],*yc=new uint16_t[h],*yg=new uint16_t[h];
  srand(11);
  for(int i=0;i<h;++i) x[i]=f32_to_bf16(((float)rand()/RAND_MAX-0.5f)*0.5f);
  for(size_t i=0;i<(size_t)h*f;++i){ Wg[i]=f32_to_bf16(((float)rand()/RAND_MAX-0.5f)*0.2f); Wu[i]=f32_to_bf16(((float)rand()/RAND_MAX-0.5f)*0.2f); }
  for(size_t i=0;i<(size_t)f*h;++i) Wd[i]=f32_to_bf16(((float)rand()/RAND_MAX-0.5f)*0.2f);
  mlp_cpu(x,Wg,Wu,Wd,yc,h,f);
  uint16_t *dx,*dWg,*dWu,*dWd,*dy;
  cudaMalloc(&dx,h*2); cudaMalloc(&dWg,(size_t)h*f*2); cudaMalloc(&dWu,(size_t)h*f*2);
  cudaMalloc(&dWd,(size_t)f*h*2); cudaMalloc(&dy,h*2);
  cudaMemcpy(dx,x,h*2,cudaMemcpyHostToDevice); cudaMemcpy(dWg,Wg,(size_t)h*f*2,cudaMemcpyHostToDevice);
  cudaMemcpy(dWu,Wu,(size_t)h*f*2,cudaMemcpyHostToDevice); cudaMemcpy(dWd,Wd,(size_t)f*h*2,cudaMemcpyHostToDevice);
  mlp_cuda(dx,dWg,dWu,dWd,dy,h,f); cudaDeviceSynchronize();
  cudaMemcpy(yg,dy,h*2,cudaMemcpyDeviceToHost);
  float mx=0; for(int i=0;i<h;++i){ float e=fabsf(bf16_to_f32(yc[i])-bf16_to_f32(yg[i])); if(e>mx)mx=e; }
  printf("mlp h=%d f=%d max_abs_err=%.6f %s\n",h,f,mx,mx<5e-2?"PASS":"FAIL");
  return mx<5e-2?0:1;
}
