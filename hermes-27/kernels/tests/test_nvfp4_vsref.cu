// Kernel vs CUTLASS host reference on identical builder-produced buffers.
// If MATCH: builders+kernel consistent (bug in test reference/quant).
// If DIFFER: kernel reads buffers differently (layout bug).
#include "hermes/nvfp4_gemm.hpp"
#include "hermes/rmsnorm.hpp"
#include <cstdio>
#include <vector>
#include <cuda_bf16.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"
#include "cutlass/util/reference/host/gett.hpp"
using namespace hermes;
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
using ElementD = cutlass::bfloat16_t; using ElementC = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor; using LayoutDTag = cutlass::layout::RowMajor;
using ElementAccumulator = float; using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = cute::Shape<cutlass::_128, cutlass::_128, cutlass::_128>;
using ClusterShape = cute::Shape<cutlass::_1, cutlass::_1, cutlass::_1>;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, 8, ElementD, LayoutDTag, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag, 32, ElementB, LayoutBTag, 32, ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;
using Sm1xxBlkScaledConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;
template <typename T> auto make_iterator(T* ptr) { return cute::recast_ptr<T>(ptr); }
static float e2m1t(int v){ static const float t[16]={0,.5f,1,1.5f,2,3,4,6,0,-.5f,-1,-1.5f,-2,-3,-4,-6}; return t[v&15]; }
static float ue2f(uint8_t x){ x=(uint8_t)x; if(x==0||x==0x7F) return 0; int e=(x>>3)&15,m=x&7;
  return e==0?ldexp((float)m,-9):ldexp(1+(float)m/8,e-7); }
static uint8_t f2ue(float x){ if(!(x>0))return 0; if(x>448)x=448; uint32_t b; memcpy(&b,&x,4);
  int fe=((b>>23)&0xFF)-127,fm=(b>>20)&7,ee=fe+7;
  if(ee<=0){int m=(int)(x*512+.5f); if(m>7)m=7; return m<1?0:(uint8_t)m;} if(ee>=15)return 0x7E;
  int rb=(b>>19)&1,mm=fm+rb; if(mm>7){mm=0; if(++ee>=15)return 0x7E;} return (uint8_t)((ee<<3)|mm); }
int main(){
  const int T=128,N=128,K=128;
  std::vector<float> W((size_t)N*K); std::vector<uint16_t> X((size_t)T*K);
  srand(31);
  for(auto& w:W) w=((float)rand()/RAND_MAX-0.5f)*0.3f;
  for(size_t i=0;i<X.size();++i) X[i]=f32_to_bf16(((float)rand()/RAND_MAX-0.5f)*0.5f);
  std::vector<uint8_t> Grow((size_t)N*(K/64*36),0);
  for(int n=0;n<N;++n) for(int b=0;b<K/64;++b) for(int s=0;s<4;++s){
    float amax=0; for(int j=0;j<16;++j){float a=fabsf(W[(size_t)n*K+b*64+s*16+j]); if(a>amax)amax=a;}
    uint8_t ue=f2ue(amax/6.f); Grow[(size_t)n*(K/64*36)+b*36+s]=ue;
    float rs=ue2f(ue)*2.f;
    for(int j=0;j<8;++j){
      float v0=rs==0?0:W[(size_t)n*K+b*64+s*16+j]/rs, v1=rs==0?0:W[(size_t)n*K+b*64+s*16+j+8]/rs;
      int q0=0,q1=0; float d0=1e30f,d1=1e30f;
      for(int q=0;q<16;++q){float e=e2m1t(q),a0=fabsf(v0-e),a1=fabsf(v1-e); if(a0<d0){d0=a0;q0=q;} if(a1<d1){d1=a1;q1=q;}}
      Grow[(size_t)n*(K/64*36)+b*36+4+s*8+j]=(uint8_t)(q0|(q1<<4));
    }
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
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {T, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {T, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {T, N, 1});
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(T, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(T, N, K, 1));
  auto layout_A = cute::make_layout(cute::make_shape(T, K, 1), stride_A);
  auto layout_B = cute::make_layout(cute::make_shape(N, K, 1), stride_B);
  auto layout_C = cute::make_layout(cute::make_shape(T, N, 1), stride_C);
  auto layout_D = cute::make_layout(cute::make_shape(T, N, 1), stride_D);
  cute::Tensor tensor_A = cute::make_tensor(
      make_iterator(reinterpret_cast<cutlass::float_e2m1_t*>(Aq.data())), layout_A);
  cute::Tensor tensor_SFA = cute::make_tensor(SFA.data(), layout_SFA);
  cute::Tensor tensor_B = cute::make_tensor(
      make_iterator(reinterpret_cast<cutlass::float_e2m1_t*>(Bq.data())), layout_B);
  cute::Tensor tensor_SFB = cute::make_tensor(SFB.data(), layout_SFB);
  std::vector<uint16_t> Cref((size_t)T*N,0), Dref((size_t)T*N,0);
  cute::Tensor tensor_C = cute::make_tensor((ElementC*)Cref.data(), layout_C);
  cute::Tensor tensor_D = cute::make_tensor((ElementD*)Dref.data(), layout_D);
  cutlass::reference::host::GettBlockScalingMainloopParams<ElementAccumulator,
      decltype(tensor_A), decltype(tensor_SFA), decltype(tensor_B), decltype(tensor_SFB)>
      mainloop_params{tensor_A, tensor_SFA, tensor_B, tensor_SFB};
  cutlass::reference::host::GettBlockScalingEpilogueParams<ElementAccumulator, ElementAccumulator,
      float, decltype(tensor_C), decltype(tensor_D)>
      epilogue_params{1.0f, 0.0f, tensor_C, tensor_D};
  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);
  auto b2f=[](uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; };
  float mx=0; for(size_t i=0;i<D.size();++i){ float e=fabsf(b2f(D[i])-b2f(Dref[i])); if(e>mx)mx=e; }
  printf("kernel vs CUTLASS-ref max_abs_err=%.6f %s (rc=%d)\n", mx, (rc==0&&mx<1e-2)?"MATCH":"DIFFER", rc);
  return (rc==0&&mx<1e-2)?0:1;
}
