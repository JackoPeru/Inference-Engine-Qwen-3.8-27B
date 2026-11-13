// test_rmsnorm: GPU vs CPU, hidden 5120, eps 1e-6. Prints max abs err, PASS if <2e-3 (BF16).
#include "hermes/rmsnorm.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
using namespace hermes;
int main(){
  const int n=5120; const float eps=1e-6f;
  uint16_t *hx=new uint16_t[n],*hw=new uint16_t[n],*hy_c=new uint16_t[n],*hy_g=new uint16_t[n];
  srand(7);
  for(int i=0;i<n;++i){ float v=((float)rand()/RAND_MAX-0.5f)*2.0f; hx[i]=f32_to_bf16(v); hw[i]=f32_to_bf16(0.8f+0.4f*((float)rand()/RAND_MAX)); }
  rmsnorm_cpu(hx,hw,hy_c,n,eps);
  uint16_t *dx,*dw,*dy;
  cudaMalloc(&dx,n*2); cudaMalloc(&dw,n*2); cudaMalloc(&dy,n*2);
  cudaMemcpy(dx,hx,n*2,cudaMemcpyHostToDevice); cudaMemcpy(dw,hw,n*2,cudaMemcpyHostToDevice);
  rmsnorm_cuda(dx,dw,dy,n,eps); cudaDeviceSynchronize();
  cudaMemcpy(hy_g,dy,n*2,cudaMemcpyDeviceToHost);
  float mx=0; for(int i=0;i<n;++i){ float a=bf16_to_f32(hy_c[i]),b=bf16_to_f32(hy_g[i]); float e=fabsf(a-b); if(e>mx)mx=e; }
  printf("rmsnorm n=%d max_abs_err=%.6f %s\n",n,mx,mx<2e-3?"PASS":"FAIL");
  cudaFree(dx);cudaFree(dw);cudaFree(dy);
  delete[]hx;delete[]hw;delete[]hy_c;delete[]hy_g;
  return mx<2e-3?0:1;
}
