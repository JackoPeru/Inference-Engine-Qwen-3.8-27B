// Layout cross-check (host-only): build buffers via engine builders with EXACT
// e2m1 values, read back through CuTe tensors (kernel's view). Must match.
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
template <typename T>
auto make_iterator(T* ptr) { return cute::recast_ptr<T>(ptr); }
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
using StrideB = typename GemmKernel::StrideB;
using StrideA = typename GemmKernel::StrideA;
static float e2m1t(int v){ static const float t[16]={0,.5f,1,1.5f,2,3,4,6,0,-.5f,-1,-1.5f,-2,-3,-4,-6}; return t[v&15]; }
int main(){
  const int N=8,K=64; // single GGUF block per row
  // W[n,k] = e2m1((n+k)%16), scale raw 1.0 (ue 0x38)
  std::vector<uint8_t> Grow((size_t)N*36,0);
  for(int n=0;n<N;++n){ Grow[(size_t)n*36+0]=0x38; Grow[(size_t)n*36+1]=0x38; Grow[(size_t)n*36+2]=0x38; Grow[(size_t)n*36+3]=0x38;
    for(int j=0;j<8;++j){ int q0=(n+j)%16, q1=(n+j+8)%16; Grow[(size_t)n*36+4+j]=(uint8_t)(q0|(q1<<4)); } }
  std::vector<uint8_t> Bq((size_t)N*K/2,0), SFB(Nvfp4GemmW4A4::sfb_bytes(8,N,K),0);
  Nvfp4GemmW4A4::build_B_from_gguf(Grow.data(),Bq.data(),SFB.data(),N,K);
  // read back through CuTe tensor (kernel's view)
  auto lb = cute::make_layout(cute::make_shape(N,K,1), cutlass::make_cute_packed_stride(StrideB{},{N,K,1}));
  auto tB = cute::make_tensor(cute::make_iterator(Bq.data()), lb);
  int bad=0;
  for(int n=0;n<N;++n) for(int k=0;k<K;++k){
    float got = (float)tB(n,k,0);
    float want = e2m1t((n+k)%16);
    if(got!=want){ if(bad<8) printf("B mismatch n=%d k=%d got=%.2f want=%.2f\n",n,k,got,want); bad++; }
  }
  printf("B readback %s (%d bad)\n", bad==0?"PASS":"FAIL", bad);
  // A side: X all e2m1 pattern, quant_activation, read back
  const int T=8;
  std::vector<uint16_t> X((size_t)T*K);
  for(int t=0;t<T;++t) for(int k=0;k<K;++k) X[(size_t)t*K+k]=f32_to_bf16(e2m1t((t+k)%16)*1.0f);
  std::vector<uint8_t> Aq((size_t)T*K/2,0), SFA(Nvfp4GemmW4A4::sfa_bytes(T,K),0);
  Nvfp4GemmW4A4::quant_activation(X.data(),Aq.data(),SFA.data(),T,K);
  auto la = cute::make_layout(cute::make_shape(T,K,1), cutlass::make_cute_packed_stride(StrideA{},{T,K,1}));
  auto tA = cute::make_tensor(cute::make_iterator(Aq.data()), la);
  int badA=0;
  for(int t=0;t<T;++t) for(int k=0;k<K;++k){
    float got=(float)tA(t,k,0), want=e2m1t((t+k)%16);
    // quant rescales per-16 group: expected = rs_group * e2m1(nearest(v/rs))
    // here v IS exact e2m1, amax=6 -> raw=1 -> rs=1 -> nearest exact
    if(got!=want){ if(badA<8) printf("A mismatch t=%d k=%d got=%.2f want=%.2f\n",t,k,got,want); badA++; }
  }
  printf("A readback %s (%d bad)\n", badA==0?"PASS":"FAIL", badA);
  return (bad==0&&badA==0)?0:1;
}
