// build_B isolation: A uniform 6.0 (exact: amax 6->rs 1.0), SFB=1.0,
// B[n,k]=e2m1((n+k)%8). Expected D = 6 * 16 * 18 = 1728.
#include "hermes/nvfp4_gemm.hpp"
#include "hermes/rmsnorm.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
using namespace hermes;
int main(){
  const int T=128,N=128,K=128;
  std::vector<uint16_t> X((size_t)T*K);
  for(size_t i=0;i<X.size();++i) X[i]=f32_to_bf16(6.0f);
  std::vector<uint8_t> Grow((size_t)N*(K/64*36),0);
  for(int n=0;n<N;++n) for(int b=0;b<K/64;++b) for(int s=0;s<4;++s){
    Grow[(size_t)n*(K/64*36)+b*36+s]=0x38;
    for(int j=0;j<8;++j){ int q0=(n+(s*16+j))%8, q1=(n+(s*16+j+8))%8;
      Grow[(size_t)n*(K/64*36)+b*36+4+s*8+j]=(uint8_t)(q0|(q1<<4)); }
  }
  std::vector<uint8_t> Aq((size_t)T*K/2,0), Bq((size_t)N*K/2,0);
  std::vector<uint8_t> SFA(Nvfp4GemmW4A4::sfa_bytes(T,K),0), SFB(Nvfp4GemmW4A4::sfb_bytes(T,N,K),0);
  Nvfp4GemmW4A4::quant_activation(X.data(),Aq.data(),SFA.data(),T,K);
  Nvfp4GemmW4A4::build_B_from_gguf(Grow.data(),Bq.data(),SFB.data(),N,K);
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
  float mx=0; for(size_t i=0;i<D.size();++i){ float e=fabsf(b2f(D[i])-1728.f); if(e>mx)mx=e; }
  printf("buildB isolation: expect 1728, max_err=%.2f %s (rc=%d)\n",mx,mx<32.0f?"PASS":"FAIL",rc);
  return (rc==0&&mx<32.0f)?0:1;
}
