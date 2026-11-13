// Nibble-order probe: A bytes 0x65 (elems 3.0,4.0 alt), B bytes 0x21 (0.5,1.0 alt),
// all scales 0x40 (raw 2.0). If linear low-first: Y = 64*[(6)(1)+(8)(2)] = 1408.
// If swapped within byte: Y = 64*[(8)(1)+(6)(2)] = 1280.
#include "hermes/nvfp4_gemm.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
using namespace hermes;
int main(){
  const int T=128,N=128,K=128;
  std::vector<uint8_t> Aq((size_t)T*K/2,0x65), Bq((size_t)N*K/2,0x21);
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
  auto b2f=[](uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; };
  double mn=1e30,mx=-1e30,sm=0; for(auto v:D){ float f=b2f(v); mn=std::min<double>(mn,f); mx=std::max<double>(mx,f); sm+=f; }
  printf("order probe: expect 1408 (linear) or 1280 (swapped); got min=%.1f max=%.1f mean=%.1f rc=%d\n",mn,mx,sm/D.size(),rc);
  return 0;
}
