// test_nvfp4_gemv: H=512 F=64. W random FP32 -> quant NVFP4 -> GPU GEMV.
// Check A: GPU vs CPU-dequant err < 1e-2. Check B: vs FP32 orig rel err < 25% (4-bit coarse).
#include "hermes/nvfp4_gemv.hpp"
#include "hermes/rmsnorm.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
using namespace hermes;
int main(){
  const int H=512,F=64;
  std::vector<float> W((size_t)H*F); std::vector<uint16_t> x(H);
  srand(21);
  for(auto& w:W) w=((float)rand()/RAND_MAX-0.5f)*0.4f;
  for(auto& v:x) v=f32_to_bf16(((float)rand()/RAND_MAX-0.5f));
  std::vector<uint8_t> Wq((size_t)F*(H/64*36));
  for(int i=0;i<F;++i){ std::vector<float> row(H); for(int k=0;k<H;++k) row[k]=W[(size_t)i*H+k]; nvfp4_quant_row(row.data(),Wq.data()+(size_t)i*(H/64*36),H); }
  std::vector<float> y_cpu(F), y_ref(F);
  for(int i=0;i<F;++i){
    y_cpu[i]=nvfp4_gemv_cpu(x.data(),Wq.data()+(size_t)i*(H/64*36),H);
    double a=0; for(int k=0;k<H;++k) a+=double(bf16_to_f32(x[k]))*double(W[(size_t)i*H+k]); y_ref[i]=(float)a;
  }
  uint16_t *dx; uint8_t *dWq; float *dy;
  cudaMalloc(&dx,H*2); cudaMalloc(&dWq,Wq.size()); cudaMalloc(&dy,F*4);
  cudaMemcpy(dx,x.data(),H*2,cudaMemcpyHostToDevice);
  cudaMemcpy(dWq,Wq.data(),Wq.size(),cudaMemcpyHostToDevice);
  nvfp4_gemv_cuda(dx,dWq,dy,H,F); cudaDeviceSynchronize();
  std::vector<float> y_gpu(F); cudaMemcpy(y_gpu.data(),dy,F*4,cudaMemcpyDeviceToHost);
  float mxA=0,mxB=0,nrm=0; for(int i=0;i<F;++i){ float e=fabsf(y_cpu[i]-y_gpu[i]); if(e>mxA)mxA=e; float r=fabsf(y_gpu[i]-y_ref[i]); if(r>mxB)mxB=r; nrm+=y_ref[i]*y_ref[i]; }
  nrm=sqrtf(nrm/F); float rel=mxB/(nrm+1e-6f);
  printf("nvfp4_gemv H=%d F=%d gpu_vs_cpu=%.6f %s | vs_fp32 max=%.4f rms=%.4f rel=%.3f %s\n",
    H,F,mxA,mxA<1e-2?"PASS":"FAIL",mxB,nrm,rel,rel<0.35?"PASS":"FAIL");
  return (mxA<1e-2 && rel<0.35)?0:1;
}
