// test_gdn_step: sequential T=8 steps, GPU (state carry) vs CPU.
// Small (H=4,K=16,V=16) exact + real-dims (48,128,128) single-step smoke.
#include "hermes/gdn_step.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
using namespace hermes;
static int run_case(int H,int K,int V,int T){
  std::vector<float> q(T*H*K),k(T*H*K),v(T*H*V),be(T*H),gl(T*H);
  srand(9);
  for(auto& x:q)x=((float)rand()/RAND_MAX-0.5f)*0.5f;
  for(auto& x:k)x=((float)rand()/RAND_MAX-0.5f)*0.5f;
  for(auto& x:v)x=((float)rand()/RAND_MAX-0.5f)*0.5f;
  for(auto& x:be)x=((float)rand()/RAND_MAX)*0.8f+0.1f;
  for(auto& x:gl)x=-((float)rand()/RAND_MAX)*0.5f;
  std::vector<float> Sc(H*K*V,0.f), oc(T*H*V);
  for(int t=0;t<T;++t) gdn_step_cpu(&q[t*H*K],&k[t*H*K],&v[t*H*V],&be[t*H],&gl[t*H],Sc.data(),&oc[t*H*V],H,K,V);
  float *dq,*dk,*dv,*dbe,*dgl,*dS,*doo;
  cudaMalloc(&dq,q.size()*4);cudaMalloc(&dk,k.size()*4);cudaMalloc(&dv,v.size()*4);
  cudaMalloc(&dbe,be.size()*4);cudaMalloc(&dgl,gl.size()*4);
  cudaMalloc(&dS,Sc.size()*4);cudaMalloc(&doo,oc.size()*4);
  cudaMemcpy(dq,q.data(),q.size()*4,cudaMemcpyHostToDevice);
  cudaMemcpy(dk,k.data(),k.size()*4,cudaMemcpyHostToDevice);
  cudaMemcpy(dv,v.data(),v.size()*4,cudaMemcpyHostToDevice);
  cudaMemcpy(dbe,be.data(),be.size()*4,cudaMemcpyHostToDevice);
  cudaMemcpy(dgl,gl.data(),gl.size()*4,cudaMemcpyHostToDevice);
  cudaMemset(dS,0,Sc.size()*4);
  for(int t=0;t<T;++t){
    gdn_step_cuda(dq+t*H*K,dk+t*H*K,dv+t*H*V,dbe+t*H,dgl+t*H,dS,doo+t*H*V,H,K,V);
  }
  cudaDeviceSynchronize();
  std::vector<float> og(oc.size()), Sg(Sc.size());
  cudaMemcpy(og.data(),doo,og.size()*4,cudaMemcpyDeviceToHost);
  cudaMemcpy(Sg.data(),dS,Sg.size()*4,cudaMemcpyDeviceToHost);
  float mx=0,ms=0; for(size_t i=0;i<og.size();++i){float e=fabsf(og[i]-oc[i]); if(e>mx)mx=e;}
  for(size_t i=0;i<Sg.size();++i){float e=fabsf(Sg[i]-Sc[i]); if(e>ms)ms=e;}
  printf("gdn_step H=%d K=%d V=%d T=%d out=%.2e state=%.2e %s\n",H,K,V,T,mx,ms,(mx<1e-3&&ms<1e-3)?"PASS":"FAIL");
  cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(dbe);cudaFree(dgl);cudaFree(dS);cudaFree(doo);
  return (mx<1e-3&&ms<1e-3)?0:1;
}
int main(){ int r=0; r|=run_case(4,16,16,8); r|=run_case(48,128,128,2); return r; }
