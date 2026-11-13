// Structured mapping tests: isolate SFA / SFB / B-nibble placement.
// T=N=K=128. Scales raw: ue byte for value v = f2ue(v).
#include "hermes/nvfp4_gemm.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
using namespace hermes;
static uint8_t f2ue(float x){ if(!(x>0))return 0; if(x>448)x=448; uint32_t b; memcpy(&b,&x,4);
  int fe=((b>>23)&0xFF)-127,fm=(b>>20)&7,ee=fe+7;
  if(ee<=0){int m=(int)(x*512+.5f); if(m>7)m=7; return m<1?0:(uint8_t)m;} if(ee>=15)return 0x7E;
  int rb=(b>>19)&1,mm=fm+rb; if(mm>7){mm=0; if(++ee>=15)return 0x7E;} return (uint8_t)((ee<<3)|mm); }
static uint8_t *dA,*dSFA,*dB,*dSFB; static uint16_t *dC,*dD;
static float b2f(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }
static int run_case(const std::vector<uint8_t>& Aq, const std::vector<uint8_t>& SFA,
             const std::vector<uint8_t>& Bq, const std::vector<uint8_t>& SFB,
             std::vector<uint16_t>& D, int T,int N,int K){
  cudaMemcpy(dA,Aq.data(),Aq.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dSFA,SFA.data(),SFA.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,Bq.data(),Bq.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dSFB,SFB.data(),SFB.size(),cudaMemcpyHostToDevice);
  int rc=Nvfp4GemmW4A4::run(dA,dSFA,dB,dSFB,dC,dD,T,N,K);
  cudaMemcpy(D.data(),dD,(size_t)T*N*2,cudaMemcpyDeviceToHost);
  return rc;
}
int main(){
  const int T=128,N=128,K=128;
  cudaMalloc(&dA,(size_t)T*K/2);cudaMalloc(&dSFA,Nvfp4GemmW4A4::sfa_bytes(T,K));
  cudaMalloc(&dB,(size_t)N*K/2);cudaMalloc(&dSFB,Nvfp4GemmW4A4::sfb_bytes(T,N,K));
  cudaMalloc(&dC,(size_t)T*N*2);cudaMalloc(&dD,(size_t)T*N*2);
  cudaMemset(dC,0,(size_t)T*N*2);
  std::vector<uint16_t> D((size_t)T*N);
  int fails=0;
  { // 1. SFA mapping: A=B=1.0 (0x22), SFB=1.0, SFA[m]=raw (m+1)*0.25
    std::vector<uint8_t> Aq((size_t)T*K/2,0x22), Bq((size_t)N*K/2,0x22);
    std::vector<uint8_t> SFA(Nvfp4GemmW4A4::sfa_bytes(T,K),0), SFB(Nvfp4GemmW4A4::sfb_bytes(T,N,K),0x38);
    // fill SFA via repack path: build flat then repack
    std::vector<uint8_t> flat((size_t)T*(K/16));
    for(int m=0;m<T;++m) for(int b=0;b<K/16;++b) flat[m*(K/16)+b]=f2ue((m+1)*0.25f);
    Nvfp4GemmW4A4::repack_sfa_cpu(flat.data(),SFA.data(),T,K);
    int rc=run_case(Aq,SFA,Bq,SFB,D,T,N,K);
    float mx=0;
    for(int m=0;m<T;++m){ float exp=(m+1)*0.25f*128.f; // K * sA * 1 * 1
      for(int n=0;n<N;++n){ float e=fabsf(b2f(D[(size_t)m*N+n])-exp); if(e>mx)mx=e; } }
    printf("SFA mapping: max_err=%.4f %s\n",mx,mx<2.0f?"PASS":"FAIL"); fails+=(mx<2.0f?0:1);
  }
  { // 2. SFB mapping: mirror (vary SFB[n])
    std::vector<uint8_t> Aq((size_t)T*K/2,0x22), Bq((size_t)N*K/2,0x22);
    std::vector<uint8_t> SFA(Nvfp4GemmW4A4::sfa_bytes(T,K),0x38);
    std::vector<uint8_t> SFB(Nvfp4GemmW4A4::sfb_bytes(T,N,K),0);
    // build SFB via engine's build_B path? use direct: need GGUF rows. Instead emulate:
    // SFB logical [N, Kb]: fill with repack-like loop not exposed; use build_B with uniform GGUF rows then overwrite?
    // Simpler: all-zero SFB then set per-n via same layout math is untestable here.
    // Use build_B_from_gguf with GGUF rows carrying per-row scales.
    int nb=K/64;
    std::vector<uint8_t> Grow((size_t)N*nb*36,0);
    for(int n=0;n<N;++n) for(int b=0;b<nb;++b) for(int s=0;s<4;++s){
      Grow[(size_t)n*nb*36+b*36+s]=f2ue((n+1)*0.25f);
      for(int j=0;j<8;++j) Grow[(size_t)n*nb*36+b*36+4+s*8+j]=0x22;
    }
    Nvfp4GemmW4A4::build_B_from_gguf(Grow.data(),Bq.data(),SFB.data(),N,K);
    int rc=run_case(Aq,SFA,Bq,SFB,D,T,N,K);
    float mx=0;
    for(int n=0;n<N;++n){ float exp=(n+1)*0.25f*128.f;
      for(int m=0;m<T;++m){ float e=fabsf(b2f(D[(size_t)m*N+n])-exp); if(e>mx)mx=e; } }
    printf("SFB mapping: max_err=%.4f %s\n",mx,mx<2.0f?"PASS":"FAIL"); fails+=(mx<2.0f?0:1);
  }
  printf(fails==0?"STRUCT PASS\n":"STRUCT FAIL (%d)\n",fails);
  return fails;
}
