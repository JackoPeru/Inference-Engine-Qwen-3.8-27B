// Deterministic pattern test: X[t,k]=e2m1((t+k)%16), W[n,k]=e2m1((n+2k)%16).
// amax=6 -> raw=1 -> ue=0x38 exact, all nibbles exact. GPU must match fp32 ref ~1e-2.
#include "hermes/nvfp4_gemm.hpp"
#include "hermes/rmsnorm.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
using namespace hermes;
static float e2m1t(int v){ static const float t[16]={0,.5f,1,1.5f,2,3,4,6,0,-.5f,-1,-1.5f,-2,-3,-4,-6}; return t[v&15]; }
int main(){
  const int T=128,N=128,K=128;
  std::vector<uint16_t> X((size_t)T*K);
  for(int t=0;t<T;++t) for(int k=0;k<K;++k) X[(size_t)t*K+k]=f32_to_bf16(e2m1t((t+k)%16));
  std::vector<uint8_t> Grow((size_t)N*(K/64*36),0);
  for(int n=0;n<N;++n) for(int b=0;b<K/64;++b) for(int s=0;s<4;++s){
    Grow[(size_t)n*(K/64*36)+b*36+s]=0x38;
    for(int j=0;j<8;++j){ int q0=(n+2*(s*16+j))%16, q1=(n+2*(s*16+j+8))%16;
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
  float mx=0,nrm=0; for(int t=0;t<T;++t) for(int n=0;n<N;++n){
    double a=0; for(int k=0;k<K;++k) a+=double(e2m1t((t+k)%16))*double(e2m1t((n+2*k)%16));
    float e=fabsf(b2f(D[(size_t)t*N+n])-(float)a); if(e>mx)mx=e; nrm+=(float)(a*a); }
  nrm=sqrtf(nrm/(T*N));
  printf("pattern T=%d N=%d K=%d max=%.4f rms=%.2f rel=%.4f %s (rc=%d)\n",
    T,N,K,mx,nrm,mx/(nrm+1e-6),(rc==0&&mx/(nrm+1e-6)<0.05)?"PASS":"FAIL",rc);
  return (rc==0&&mx/(nrm+1e-6)<0.05)?0:1;
}
