// Experimental NVFP4 prefill quantization and PP2 baseline, separate from
// the existing CUTLASS wrapper and GEMV kernels.
#include "hermes/nvfp4_prefill_prototype.hpp"
#include "hermes/nvfp4_gemm.hpp"
#include <cuda_bf16.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace hermes::prefill_prototype {

bool build_sfa_scatter_map(int T, int K, std::vector<uint32_t>& map,
                           size_t& bytes, std::string& error) {
  map.clear(); bytes = 0;
  if (T <= 0 || K <= 0 || K % 64 != 0) {
    error = "T must be positive, K must be a positive multiple of 64"; return false;
  }
  const size_t count = size_t(T) * size_t(K / 16);
  if (count >= 0xFFFFFFu) {
    error = "scatter-map index exceeds 24-bit sentinel"; return false;
  }
  bytes = Nvfp4GemmW4A4::sfa_bytes(T, K);
  if (!bytes || bytes > UINT32_MAX) {
    error = "SFA layout unavailable, or SFA bytes exceed uint32 offset"; return false;
  }

  // Invert exactly the currently installed CUTLASS SFA layout without
  // copying its internal tile/layout definitions. Encode logical indices
  // into three distinct byte planes; repack all three planes using the
  // engine's own layout function, then decode the destination offsets.
  // 0xFFFFFF is reserved as an unwritten padding sentinel.
  std::vector<uint8_t> flat0(count), flat1(count), flat2(count);
  for (size_t i = 0; i < count; ++i) {
    flat0[i] = uint8_t(i);
    flat1[i] = uint8_t(i >> 8);
    flat2[i] = uint8_t(i >> 16);
  }
  std::vector<uint8_t> dest0(bytes, 0xFF), dest1(bytes, 0xFF), dest2(bytes, 0xFF);
  Nvfp4GemmW4A4::repack_sfa_cpu(flat0.data(), dest0.data(), T, K);
  Nvfp4GemmW4A4::repack_sfa_cpu(flat1.data(), dest1.data(), T, K);
  Nvfp4GemmW4A4::repack_sfa_cpu(flat2.data(), dest2.data(), T, K);
  map.assign(count, UINT32_MAX);
  size_t discovered = 0;
  for (size_t off = 0; off < bytes; ++off) {
    uint32_t src = uint32_t(dest0[off]) |
                   (uint32_t(dest1[off]) << 8) |
                   (uint32_t(dest2[off]) << 16);
    if (src == 0xFFFFFFu) continue; // CUTLASS padding byte.
    if (src >= count || map[src] != UINT32_MAX) {
      error = "SFA layout is non-bijective or has duplicated logical scales";
      map.clear(); return false;
    }
    map[src] = uint32_t(off);
    ++discovered;
  }
  if (discovered != count) {
    error = "SFA layout failed to map all logical scales";
    map.clear(); return false;
  }
  return true;
}

__device__ __forceinline__ uint8_t ue4m3_encode(float raw) {
  if (!(raw > 0.f)) return 0;
  float x = raw > 448.f ? 448.f : raw;
  uint32_t bits = __float_as_uint(x);
  int fe = int((bits >> 23) & 0xFF) - 127;
  int fm = int((bits >> 20) & 0x7);
  int ee = fe + 7;
  if (ee <= 0) {
    int m2 = int(x * 512.f + .5f);
    if (m2 > 7) m2 = 7;
    return m2 < 1 ? 0 : uint8_t(m2);
  }
  if (ee >= 15) return 0x7E;
  int rb = (bits >> 19) & 1;
  int mm = fm + rb;
  if (mm > 7) {
    mm = 0;
    if (++ee >= 15) return 0x7E;
  }
  return uint8_t((ee << 3) | mm);
}

__device__ __forceinline__ float ue4m3_decode(uint8_t ue) {
  if (ue == 0 || ue == 0x7F) return 0.f;
  int e = (ue >> 3) & 0xF, m = ue & 7;
  return e == 0 ? ldexpf(float(m), -9) : ldexpf(1.f + float(m)/8.f, e - 7);
}

__device__ __forceinline__ float e2m1_decode(int q) {
  switch (q & 15) {
    case 0: case 8: return 0.f;
    case 1: return .5f; case 2: return 1.f; case 3: return 1.5f;
    case 4: return 2.f; case 5: return 3.f; case 6: return 4.f; case 7: return 6.f;
    case 9: return -.5f; case 10: return -1.f; case 11: return -1.5f;
    case 12: return -2.f; case 13: return -3.f; case 14: return -4.f;
    default: return -6.f;
  }
}

__device__ __forceinline__ int nearest_e2m1(float v) {
  int best = 0; float dist = 1.e30f;
  // Identical nearest-value tie rule to Nvfp4GemmW4A4::quant_activation.
  for (int q = 0; q < 16; ++q) {
    float d = fabsf(v - e2m1_decode(q));
    if (d < dist) { dist = d; best = q; }
  }
  return best;
}

__global__ void quantize_bf16_groups(const __nv_bfloat16* __restrict__ x,
                                     const uint32_t* __restrict__ map,
                                     uint8_t* __restrict__ aq,
                                     uint8_t* __restrict__ sfa,
                                     int total_groups, int K) {
  int group = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (group >= total_groups) return;
  const int groups_per_token = K / 16;
  const int token = group / groups_per_token;
  const int block = group % groups_per_token;
  const size_t base = size_t(token)*K + size_t(block)*16;
  float val[16]; float amax = 0.f;
  #pragma unroll
  for (int j=0;j<16;++j) {
    float v = float(x[base+j]); val[j]=v;
    amax=fmaxf(amax, fabsf(v));
  }
  uint8_t ue = ue4m3_encode(amax/6.f);
  sfa[map[group]] = ue;
  float rs = ue4m3_decode(ue);
  uint8_t* packed = aq + base/2;
  #pragma unroll
  for (int j=0;j<8;++j) {
    int q0 = nearest_e2m1(rs==0.f ? 0.f : val[j*2]/rs);
    int q1 = nearest_e2m1(rs==0.f ? 0.f : val[j*2+1]/rs);
    packed[j] = uint8_t(q0 | (q1<<4));
  }
}

cudaError_t quantize_bf16_device(const uint16_t* x_bf16,
                                const uint32_t* offset_map, uint8_t* aq,
                                uint8_t* sfa, int T, int K, cudaStream_t stream) {
  if (!x_bf16 || !offset_map || !aq || !sfa || T<=0 || K<=0 || (K % 64)!=0)
    return cudaErrorInvalidValue;
  const int total = T*(K/16);
  quantize_bf16_groups<<<(total+127)/128,128,0,stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16), offset_map,
      aq, sfa, total, K);
  return cudaGetLastError();
}

// Exact scalar thread/block structure and float accumulation order copied
// from nvfp4_gemv_k in kernels/src/nvfp4_gemv.cu. One kernel launch per token.
__global__ void scalar_gguf_gemv_k(const __nv_bfloat16* x,
                                  const uint8_t* Wq, float* y, int H, int F) {
  int i=int(blockIdx.x); if(i>=F) return;
  const uint8_t* row=Wq+size_t(i)*(H/64*36);
  float acc=0.f;
  for(int b=int(threadIdx.x);b<H/64;b+=int(blockDim.x)) for(int s=0;s<4;++s) {
    float rs=ue4m3_decode(row[b*36+s]);
    for(int j=0;j<8;++j) {
      uint8_t q=row[b*36+4+s*8+j];
      acc+=float(x[b*64+s*16+j])*rs*e2m1_decode(q&15);
      acc+=float(x[b*64+s*16+j+8])*rs*e2m1_decode(q>>4);
    }
  }
  __shared__ float sh[256];
  sh[threadIdx.x]=acc; __syncthreads();
  for(int s=int(blockDim.x)/2;s>0;s>>=1) {
    if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s];
    __syncthreads();
  }
  if(threadIdx.x==0) y[i]=sh[0];
}

cudaError_t repeated_gguf_gemv(const uint16_t* x_bf16,
                               const uint8_t* raw_gguf_weights, float* y,
                               int T, int N, int K, cudaStream_t stream) {
  if (!x_bf16 || !raw_gguf_weights || !y || T<=0 || N<=0 || K<=0 || K%64!=0)
    return cudaErrorInvalidValue;
  for (int t=0;t<T;++t)
    scalar_gguf_gemv_k<<<N,256,0,stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16+size_t(t)*K),
      raw_gguf_weights,y+size_t(t)*N,K,N);
  return cudaGetLastError();
}

} // namespace hermes::prefill_prototype
