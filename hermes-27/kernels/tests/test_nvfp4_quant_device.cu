// test_nvfp4_quant_device (DISABILITATO di default, non in CMakeLists).
// Verifica tiny host-vs-device per quantizzazione attivazioni E2M1.
// Abilitare solo con -DHERMES_ENABLE_DEVICE_QUANT=1 e build manuale:
//   nvcc ... -DHERMES_ENABLE_DEVICE_QUANT=1 test_nvfp4_quant_device.cu ...
// MAI rompere la build: questo file NON e` registrato in CMakeLists.
// Confronta Aq lineare + SFA interleaved (via repack host del flat device).
#ifndef HERMES_ENABLE_DEVICE_QUANT
#error "test_nvfp4_quant_device richiede -DHERMES_ENABLE_DEVICE_QUANT=1 (disabilitato di default)"
#endif
#include "hermes/nvfp4_gemm.hpp"
#include "hermes/rmsnorm.hpp"
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
using namespace hermes;
int main(){
  const int T=32, K=128;
  const int Kb=K/16;
  std::vector<uint16_t> X((size_t)T*K);
  srand(11);
  for(size_t i=0;i<X.size();++i) X[i]=f32_to_bf16(((float)rand()/RAND_MAX-0.5f)*1.0f);
  std::vector<uint8_t> Aq_h((size_t)T*K/2,0), SFA_h(Nvfp4GemmW4A4::sfa_bytes(T,K),0);
  Nvfp4GemmW4A4::quant_activation(X.data(),Aq_h.data(),SFA_h.data(),T,K);
  // device
  uint16_t *dX=0; uint8_t *dAq=0, *dFlat=0;
  cudaMalloc(&dX,X.size()*2); cudaMalloc(&dAq,Aq_h.size()); cudaMalloc(&dFlat,(size_t)T*Kb);
  cudaMemcpy(dX,X.data(),X.size()*2,cudaMemcpyHostToDevice);
  cudaMemset(dAq,0,Aq_h.size());
  int rc=Nvfp4GemmW4A4::quant_activation_device(dX,dAq,dFlat,T,K,0);
  cudaDeviceSynchronize();
  std::vector<uint8_t> Aq_d(Aq_h.size()), flat_d((size_t)T*Kb);
  cudaMemcpy(Aq_d.data(),dAq,Aq_d.size(),cudaMemcpyDeviceToHost);
  cudaMemcpy(flat_d.data(),dFlat,flat_d.size(),cudaMemcpyDeviceToHost);
  std::vector<uint8_t> SFA_d(SFA_h.size(),0);
  Nvfp4GemmW4A4::repack_sfa_cpu(flat_d.data(),SFA_d.data(),T,K);
  int badAq=0; for(size_t i=0;i<Aq_h.size();++i) if(Aq_h[i]!=Aq_d[i]) ++badAq;
  int badS=0; for(size_t i=0;i<SFA_h.size();++i) if(SFA_h[i]!=SFA_d[i]) ++badS;
  printf("quant_device tiny T=%d K=%d rc=%d badAq=%d/%zu badSFA=%d/%zu %s\n",
    T,K,rc,badAq,Aq_h.size(),badS,SFA_h.size(),(rc==0&&badAq==0&&badS==0)?"PASS":"FAIL");
  cudaFree(dX); cudaFree(dAq); cudaFree(dFlat);
  return (rc==0&&badAq==0&&badS==0)?0:1;
}
