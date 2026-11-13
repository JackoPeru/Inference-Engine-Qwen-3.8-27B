// Side isolation with RANDOM on one side, EXACT on the other.
// Case A: A random (quant_activation), B exact pattern (build_B GGUF exact).
// Case B: A exact (direct fill), B random GGUF.
// Reference: fp32 GEMM with A-fp32 / dequant-B as appropriate.
#include "hermes/nvfp4_gemm.hpp"
#include "hermes/rmsnorm.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
using namespace hermes;
static float e2m1t(int v){ static const float t[16]={0,.5f,1,1.5f,2,3,4,6,0,-.5f,-1,-1.5f,-2,-3,-4,-6}; return t[v&15]; }
static float ue2f(uint8_t x){ x=(uint8_t)x; if(x==0||x==0x7F) return 0; int e=(x>>3)&15,m=x&7;
  return e==0?ldexp((float)m,-9):ldexp(1+(float)m/8,e-7); }
static uint8_t f2ue(float x){ if(!(x>0))return 0; if(x>448)x=448; uint32_t b; memcpy(&b,&x,4);
  int fe=((b>>23)&0xFF)-127,fm=(b>>20)&7,ee=fe+7;
  if(ee<=0){int m=(int)(x*512+.5f); if(m>7)m=7; return m<1?0:(uint8_t)m;} if(ee>=15)return 0x7E;
  int rb=(b>>19)&1,mm=fm+rb; if(mm>7){mm=0; if(++ee>=15)return 0x7E;} return (uint8_t)((ee<<3)|mm); }
static float b2f(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }
int main(){
  const int T=128,N=128,K=128;
  uint8_t *dA,*dSFA,*dB,*dSFB; uint16_t *dC,*dD;
  cudaMalloc(&dA,(size_t)T*K/2);cudaMalloc(&dSFA,Nvfp4GemmW4A4::sfa_bytes(T,K));
  cudaMalloc(&dB,(size_t)N*K/2);cudaMalloc(&dSFB,Nvfp4GemmW4A4::sfb_bytes(T,N,K));
  cudaMalloc(&dC,(size_t)T*N*2);cudaMalloc(&dD,(size_t)T*N*2);
  cudaMemset(dC,0,(size_t)T*N*2);
  std::vector<uint16_t> D((size_t)T*N);
  srand(77);
  { // Case A: A random, B exact e2m1((n+k)%8), SFB=1.0
    std::vector<uint16_t> X((size_t)T*K);
    for(size_t i=0;i<X.size();++i) X[i]=f32_to_bf16(((float)rand()/RAND_MAX-0.5f)*0.5f);
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
    cudaMemcpy(dA,Aq.data(),Aq.size(),cudaMemcpyHostToDevice);
    cudaMemcpy(dSFA,SFA.data(),SFA.size(),cudaMemcpyHostToDevice);
    cudaMemcpy(dB,Bq.data(),Bq.size(),cudaMemcpyHostToDevice);
    cudaMemcpy(dSFB,SFB.data(),SFB.size(),cudaMemcpyHostToDevice);
    int rc=Nvfp4GemmW4A4::run(dA,dSFA,dB,dSFB,dC,dD,T,N,K);
    cudaMemcpy(D.data(),dD,(size_t)T*N*2,cudaMemcpyDeviceToHost);
    float mx=0,nrm=0;
    for(int t=0;t<T;++t) for(int n=0;n<N;++n){ double a=0;
      for(int k=0;k<K;++k) a+=double(bf16_to_f32(X[(size_t)t*K+k]))*double(e2m1t((n+k)%8));
      float e=fabsf(b2f(D[(size_t)t*N+n])-(float)a); if(e>mx)mx=e; nrm+=(float)(a*a); }
    nrm=sqrtf(nrm/(T*N));
    printf("case A (A-rand,B-exact): max=%.4f rms=%.3f rel=%.3f %s\n",mx,nrm,mx/(nrm+1e-6),(rc==0&&mx/(nrm+1e-6)<0.25)?"PASS":"FAIL");
  }
  { // Case B: A exact pattern, B random GGUF
    std::vector<uint8_t> Aq((size_t)T*K/2,0);
    for(int t=0;t<T;++t) for(int k=0;k<K/2;++k){ int q0=(t+2*k)%8, q1=(t+2*k+1)%8; Aq[((size_t)t*K+k*2)/2]=(uint8_t)(q0|(q1<<4)); }
    // NOTE: direct linear fill, bypass quant_activation (SFA uniform 1.0)
    std::vector<uint8_t> SFA(Nvfp4GemmW4A4::sfa_bytes(T,K),0x38);
    std::vector<float> W((size_t)N*K);
    for(auto& w:W) w=((float)rand()/RAND_MAX-0.5f)*0.3f;
    std::vector<uint8_t> Grow((size_t)N*(K/64*36),0);
    for(int n=0;n<N;++n) for(int b=0;b<K/64;++b) for(int s=0;s<4;++s){
      float amax=0; for(int j=0;j<16;++j){float a=fabsf(W[(size_t)n*K+b*64+s*16+j]); if(a>amax)amax=a;}
      uint8_t ue=f2ue(amax/6.f); Grow[(size_t)n*(K/64*36)+b*36+s]=ue;
      float rs=ue2f(ue); // true scale (no *0.5 convention)
      for(int j=0;j<8;++j){
        float v0=rs==0?0:W[(size_t)n*K+b*64+s*16+j]/rs, v1=rs==0?0:W[(size_t)n*K+b*64+s*16+j+8]/rs;
        int q0=0,q1=0; float d0=1e30f,d1=1e30f;
        for(int q=0;q<16;++q){float e=e2m1t(q),a0=fabsf(v0-e),a1=fabsf(v1-e); if(a0<d0){d0=a0;q0=q;} if(a1<d1){d1=a1;q1=q;}}
        Grow[(size_t)n*(K/64*36)+b*36+4+s*8+j]=(uint8_t)(q0|(q1<<4));
      }
    }
    std::vector<uint8_t> Bq((size_t)N*K/2,0), SFB(Nvfp4GemmW4A4::sfb_bytes(T,N,K),0);
    Nvfp4GemmW4A4::build_B_from_gguf(Grow.data(),Bq.data(),SFB.data(),N,K);
    cudaMemcpy(dA,Aq.data(),Aq.size(),cudaMemcpyHostToDevice);
    cudaMemcpy(dSFA,SFA.data(),SFA.size(),cudaMemcpyHostToDevice);
    cudaMemcpy(dB,Bq.data(),Bq.size(),cudaMemcpyHostToDevice);
    cudaMemcpy(dSFB,SFB.data(),SFB.size(),cudaMemcpyHostToDevice);
    int rc=Nvfp4GemmW4A4::run(dA,dSFA,dB,dSFB,dC,dD,T,N,K);
    cudaMemcpy(D.data(),dD,(size_t)T*N*2,cudaMemcpyDeviceToHost);
    float mx=0,nrm=0;
    for(int t=0;t<T;++t) for(int n=0;n<N;++n){ double a=0;
      for(int k=0;k<K;++k){ float w=0; int b=k/64,s=(k%64)/16,j=(k%16)%8,hi=((k%16)/8);
        uint8_t q=Grow[(size_t)n*(K/64*36)+b*36+4+s*8+j]; int qq=hi?(q>>4):(q&15);
        float rs=ue2f(Grow[(size_t)n*(K/64*36)+b*36+s]); w=rs*e2m1t(qq);
        a+=double(e2m1t((t+k)%8))*double(w); }
      float e=fabsf(b2f(D[(size_t)t*N+n])-(float)a); if(e>mx)mx=e; nrm+=(float)(a*a); }
    nrm=sqrtf(nrm/(T*N));
    printf("case B (A-exact,B-rand): max=%.4f rms=%.3f rel=%.3f %s\n",mx,nrm,mx/(nrm+1e-6),(rc==0&&mx/(nrm+1e-6)<0.25)?"PASS":"FAIL");
  }
  return 0;
}
