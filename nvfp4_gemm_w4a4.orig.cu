// HERMES-27 W4A4 NVFP4 GEMM via CUTLASS SM120.
// GEMM kernel configuration adapted from NVIDIA CUTLASS example
// 79a_blackwell_geforce_nvfp4_bf16_gemm (BSD-3-Clause, (c) NVIDIA CORPORATION).
// All plumbing (GGUF shuffle/repack, host builders, engine API) is HERMES-27 code.
//
// Orientation: A = weights [N,K] row-major NVFP4 (+SFA), B = activations [T,K]
//   row-major quantized then stored col-major (N=T,K) NVFP4 (+SFB), D = Y^T [N,T].
// GGUF 64B-block nibble order [e_j,e_{j+8}] (llama.cpp dequantize_block_nvfp4)
// is shuffled to linear nibble order on upload (one-time CPU cost).
#include "hermes/nvfp4_gemm.hpp"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <vector>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cute/tensor.hpp"
#include "cutlass/util/packed_stride.hpp"
namespace hermes {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;
using ElementD = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = cute::Shape<cutlass::_128, cutlass::_128, cutlass::_128>;
using ClusterShape = cute::Shape<cutlass::_1, cutlass::_1, cutlass::_1>;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA, ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
static auto sfa_layout(int m, int n, int k) {
  return Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
}
static auto sfb_layout(int m, int n, int k) {
  return Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
}
#endif
size_t Nvfp4GemmW4A4::sfa_bytes(int M, int K) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  return size(cute::filter_zeros(sfa_layout(M, 8, K)));
#else
  (void)M; (void)K; return 0;
#endif
}
void Nvfp4GemmW4A4::repack_sfa_cpu(const uint8_t* flat, uint8_t* out, int M, int K) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  auto l = sfa_layout(M, 8, K);
  int kb = K / 16;
  for (int m = 0; m < M; ++m) {
    auto cm = cute::idx2crd(m, cute::shape<0>(l));
    for (int b = 0; b < kb; ++b) {
      auto ck = cute::idx2crd(b * 16, cute::shape<1>(l));
      out[l(cm, ck, 0)] = flat[m * kb + b];
    }
  }
#else
  (void)flat; (void)out; (void)M; (void)K;
#endif
}
void Nvfp4GemmW4A4::quant_activation(const uint16_t* X_bf16, uint8_t* Aq, uint8_t* SFA, int T, int K) {
  // Host-side activation quant (GPU fused kernel = M3 optimization).
  // Layout: A row-major nibbles + SFA interleaved via same layout fn as run().
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  auto la = cute::make_layout(cute::make_shape(T, K, 1),
      cutlass::make_cute_packed_stride(StrideA{}, {T, K, 1}));
  auto ls = sfa_layout(T, 8, K);
  for (int t = 0; t < T; ++t) {
    const uint16_t* row = X_bf16 + (size_t)t * K;
    for (int b = 0; b < K / 16; ++b) {
      float amax = 0;
      for (int j = 0; j < 16; ++j) { __nv_bfloat16 h; memcpy(&h, &row[b*16+j], 2); float a=fabsf((float)h); if(a>amax)amax=a; }
      float raw = amax / 6.0f;
      uint8_t ue;
      if (!(raw > 0.f)) ue = 0;
      else { float x = raw > 448.f ? 448.f : raw; uint32_t bits; memcpy(&bits,&x,4);
        int fe=((bits>>23)&0xFF)-127, fm=(bits>>20)&0x7, ee=fe+7;
        if (ee<=0){ int m2=(int)(x*512.f+.5f); if(m2>7)m2=7; ue=(m2<1)?0:(uint8_t)m2; }
        else if (ee>=15) ue=0x7E;
        else { int rb=(bits>>19)&1, mm=fm+rb; if(mm>7){mm=0; if(++ee>=15){ue=0x7E;} else ue=(uint8_t)((ee<<3)|mm);} else ue=(uint8_t)((ee<<3)|mm); } }
      float rs; { int e=(ue>>3)&15, m2=ue&7; rs=(ue==0||ue==0x7F)?0.f:(e==0?ldexpf((float)m2,-9):ldexp(1+(float)m2/8,e-7)); }
      SFA[ls(cute::idx2crd(t, cute::shape<0>(ls)), cute::idx2crd(b * 16, cute::shape<1>(ls)), 0)] = ue;
      // Linear nibble order: byte j holds elements (2j, 2j+1) (NOT GGUF j/j+8 pairing)
      for (int j = 0; j < 8; ++j) {
        __nv_bfloat16 h0,h1; memcpy(&h0,&row[b*16+2*j],2); memcpy(&h1,&row[b*16+2*j+1],2);
        float v0 = rs==0?0:(float)h0/rs, v1 = rs==0?0:(float)h1/rs;
        int q0=0,q1=0; float d0=1e30f,d1=1e30f;
        for (int q = 0; q < 16; ++q) {
          float e; switch(q){case 0:e=0;break;case 1:e=.5f;break;case 2:e=1;break;case 3:e=1.5f;break;
            case 4:e=2;break;case 5:e=3;break;case 6:e=4;break;case 7:e=6;break;case 8:e=0;break;
            case 9:e=-.5f;break;case 10:e=-1;break;case 11:e=-1.5f;break;case 12:e=-2;break;
            case 13:e=-3;break;case 14:e=-4;break;default:e=-6;}
          float a0=fabsf(v0-e),a1=fabsf(v1-e); if(a0<d0){d0=a0;q0=q;} if(a1<d1){d1=a1;q1=q;}
        }
        size_t i0 = la(t, (b*16+2*j), 0), i1 = la(t, (b*16+2*j+1), 0);
        Aq[i0/2] |= (uint8_t)(q0 << (4*(i0&1)));
        Aq[i1/2] |= (uint8_t)(q1 << (4*(i1&1)));
      }
    }
  }
#else
  (void)X_bf16; (void)Aq; (void)SFA; (void)T; (void)K;
#endif
}
size_t Nvfp4GemmW4A4::sfb_bytes(int T, int N, int K) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  return size(cute::filter_zeros(sfb_layout(T, N, K)));
#else
  (void)T; (void)N; (void)K; return 0;
#endif
}
void Nvfp4GemmW4A4::build_B_from_gguf(const uint8_t* rows, uint8_t* Bq, uint8_t* SFB, int N, int K) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  auto lb = cute::make_layout(cute::make_shape(N, K, 1),
      cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
  auto ls = sfb_layout(8, N, K);
  int nb = K / 64;
  for (int n = 0; n < N; ++n) {
    auto cn_ls = cute::idx2crd(n, cute::shape<0>(ls));
    const uint8_t* row = rows + (size_t)n * nb * 36;
    for (int b = 0; b < nb; ++b) for (int s = 0; s < 4; ++s) {
      size_t doff = ls(cn_ls, cute::idx2crd((b * 4 + s) * 16, cute::shape<1>(ls)), 0);
      SFB[doff] = row[b * 36 + s];
      for (int j = 0; j < 8; ++j) {
        uint8_t q = row[b * 36 + 4 + s * 8 + j];
        int k0 = b * 64 + s * 16 + j, k1 = k0 + 8;
        size_t i0 = lb(n, k0, 0), i1 = lb(n, k1, 0);
        Bq[i0 / 2] |= (uint8_t)((q & 0xF) << (4 * (i0 & 1)));
        Bq[i1 / 2] |= (uint8_t)(((q >> 4) & 0xF) << (4 * (i1 & 1)));
      }
    }
  }
#else
  (void)rows; (void)Bq; (void)SFB; (void)N; (void)K;
#endif
}
int Nvfp4GemmW4A4::run(const uint8_t* Aq, const uint8_t* SFA,
                       const uint8_t* Bq, const uint8_t* SFB,
                       const uint16_t* C, uint16_t* D, int M_T, int N, int K) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  using namespace cute;
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M_T, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M_T, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M_T, N, 1});
  auto layout_SFA = sfa_layout(M_T, N, K);
  auto layout_SFB = sfb_layout(M_T, N, K);
  using ElementSF = typename CollectiveMainloop::ElementSF;
  using InnerA = cutlass::float_e2m1_t;
  using InnerSF = cutlass::float_ue4m3_t;
  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm, {M_T, N, K, 1},
    { reinterpret_cast<InnerA const*>(Aq), stride_A,
      reinterpret_cast<InnerA const*>(Bq), stride_B,
      reinterpret_cast<InnerSF const*>(SFA), layout_SFA,
      reinterpret_cast<InnerSF const*>(SFB), layout_SFB },
    { {1.0f, 0.0f}, reinterpret_cast<ElementC const*>(C), stride_C,
      reinterpret_cast<ElementD*>(D), stride_D }
  };
  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return -1;
  size_t ws = Gemm::get_workspace_size(args);
  void* wptr = nullptr;
  if (ws) cudaMalloc(&wptr, ws);
  if (gemm.initialize(args, (uint8_t*)wptr) != cutlass::Status::kSuccess) { if (wptr) cudaFree(wptr); return -2; }
  auto st = gemm.run();
  cudaDeviceSynchronize();
  if (wptr) cudaFree(wptr);
  return (st == cutlass::Status::kSuccess) ? 0 : -3;
#else
  (void)Aq;(void)SFA;(void)Bq;(void)SFB;(void)C;(void)D;(void)M_T;(void)N;(void)K;
  return -99;
#endif
}
const char* Nvfp4GemmW4A4::backend_name() { return "cutlass-79a-sm120-w4a4 (BSD NVIDIA)"; }
}
