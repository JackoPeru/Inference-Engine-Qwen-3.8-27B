// Constant-value W4A4 smoke: all e2m1=0x6 (4.0), all scales UE4M3=0x40 (raw 2.0).
// Expected D[m,n] = K * (2*4) * (2*4) = K*64. K=128 -> 8192.
#include "hermes/nvfp4_gemm.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
using namespace hermes;
int main(){
  const int T=128,N=128,K=128;
  std::vector<uint8_t> Aq((size_t)T*K/2,0x77), Bq((size_t)N*K/2,0x77);
  std::vector<uint8_t> SFA(Nvfp4GemmW4A4::sfa_bytes(T,K),0x40), SFB(Nvfp4GemmW4A4::sfb_bytes(T,N,K),0x40);
  uint8_t *dA,*dSFA,*dB,*dSFB; uint16_t *dC,*dD;
  cudaMalloc(&dA,Aq.size());cudaMalloc(&dSFA,SFA.size());cudaMalloc(&dB,Bq.size());cudaMalloc(&dSFB,SFB.size());
  cudaMalloc(&dC,(size_t)T*N*2);cudaMalloc(&dD,(size_t)T*N*2);
  cudaMemcpy(dA,Aq.data(),Aq.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dSFA,SFA.data(),SFA.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,Bq.data(),Bq.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dSFB,SFB.data(),SFB.size(),cudaMemcpyHostToDevice);
  cudaMemset(dC,0,(size_t)T*N*2);
  int rc=Nvfp4GemmW4A4::run(dA,dSFA,dB,dSFB,dC,dD,T,N,K);
  std::vector<uint16_t> D((size_t)T*N);
  cudaMemcpy(D.data(),dD,(size_t)T*N*2,cudaMemcpyDeviceToHost);
  // bf16 to float
  auto b2f=[](uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; };
  double mn=1e30,mx=-1e30,sm=0; for(auto v:D){ float f=b2f(v); mn=std::min<double>(mn,f); mx=std::max<double>(mx,f); sm+=f; }
  double mean=sm/D.size();
  printf("const test: expect 8192, got min=%.1f max=%.1f mean=%.1f rc=%d %s\n",
    mn,mx,mean,rc,(rc==0&&mean>8000&&mean<8400)?"PASS":"FAIL");
  return (rc==0&&mean>8000&&mean<8400)?0:1;
}
