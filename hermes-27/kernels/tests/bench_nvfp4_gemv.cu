// Standalone A/B: raw GGUF NVFP4 (36B/64 weights) -> BF16 input -> FP32 GEMV.
// Build separately without changing production CMake:
// nvcc -O3 -std=c++20 -arch=sm_120 -I kernels/include \
//   kernels/src/nvfp4_gemv.cu kernels/src/rmsnorm.cu \
//   kernels/tests/bench_nvfp4_gemv.cu -o bench_nvfp4_gemv
// GPU exclusively reserved; synthetic inputs, CUDA events, no H2D/D2H in timing.
// Observed 2026-11-13, RTX 5060 Ti (SM120), nvcc -O3, 7 rounds x 100 launches:
//  H=5120  F=10240: scalar .94650 ms, Warp4 .13174 ms (7.185x)
//  H=17408 F=5120:  scalar 1.06120 ms, Warp4 .39905 ms (2.659x)
//  H=5120  F=17408: scalar 1.62493 ms, Warp4 .37518 ms (4.331x)
//  All 7 shapes PASS vs independent double CPU (all 64 tiny rows or 286 sampled
//  large rows) and full F-row scalar/warp comparison, max A/B 3.82e-6.
// These are synthetic standalone kernel times, NOT model TPS / end-to-end gains.
#include "hermes/nvfp4_gemv.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using hermes::Nvfp4GemvVariant;

static void cuda_check(cudaError_t result, const char* site){
  if(result!=cudaSuccess)
    throw std::runtime_error(std::string(site)+": "+cudaGetErrorString(result));
}
#define CUDA_CHECK(expr) cuda_check((expr),#expr)

template<class T> class Pinned {
public:
  T* p=nullptr;
  explicit Pinned(size_t n){ CUDA_CHECK(cudaMallocHost((void**)&p,n*sizeof(T))); }
  ~Pinned(){ if(p) cudaFreeHost(p); }
  Pinned(const Pinned&)=delete;
  Pinned& operator=(const Pinned&)=delete;
};
template<class T> class Device {
public:
  T* p=nullptr;
  explicit Device(size_t n){ CUDA_CHECK(cudaMalloc((void**)&p,n*sizeof(T))); }
  ~Device(){ if(p) cudaFree(p); }
  Device(const Device&)=delete;
  Device& operator=(const Device&)=delete;
};
struct EventPair {
  cudaEvent_t start=nullptr, stop=nullptr;
  EventPair(){
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~EventPair(){ if(start) cudaEventDestroy(start); if(stop) cudaEventDestroy(stop); }
};

static uint32_t rng_state=0x7f4a7c15u;
static uint32_t random_u32(){
  uint32_t x=rng_state;
  x^=x<<13; x^=x>>17; x^=x<<5;
  return rng_state=x;
}
static float bf16_to_float(uint16_t b){
  return std::bit_cast<float>((uint32_t)b<<16);
}
static uint16_t float_to_bf16(float f){
  return (uint16_t)(std::bit_cast<uint32_t>(f)>>16);
}
static double decode_scale(uint8_t q){
  if(q==0 || q==0x7f) return 0.0;
  unsigned e=q>>3, m=q&7u;
  return (e==0) ? std::ldexp((double)m,-9)
                : std::ldexp(1.0+(double)m/8.0,(int)e-7);
}
static double decode_e2m1(unsigned q){
  constexpr double lut[16]={0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6};
  return lut[q&15u];
}
// Independent double-precision reference, unpacks GGUF sub-block pairing.
static double cpu_ref(const uint16_t* x, const uint8_t* wrow, int H){
  double total=0;
  for(int b=0;b<H/64;++b){
    const uint8_t* block=wrow+(size_t)b*36;
    for(int s=0;s<4;++s){
      const double scale=decode_scale(block[s]);
      for(int j=0;j<8;++j){
        const uint8_t q=block[4+s*8+j];
        const int k=b*64+s*16+j;
        total+= (double)bf16_to_float(x[k]) * scale * decode_e2m1(q&15u);
        total+= (double)bf16_to_float(x[k+8]) * scale * decode_e2m1(q>>4);
      }
    }
  }
  return total;
}
static void make_input(uint16_t* x, uint8_t* w, int H, int F){
  for(int k=0;k<H;++k){
    // 100% finite BF16 values, including positive/negative, zero and ties.
    float val=(float)((int)(random_u32()%2049)-1024)/1024.0f;
    if(k%127==0) val=0;
    x[k]=float_to_bf16(val);
  }
  const int nb=H/64;
  for(int row=0;row<F;++row){
    uint8_t* rw=w+(size_t)row*nb*36;
    for(int b=0;b<nb;++b){
      uint8_t* block=rw+(size_t)b*36;
      for(int s=0;s<4;++s){
        uint32_t r=random_u32();
        uint8_t q=(uint8_t)(((r%4u)<<3) | ((r>>8)&7u));
        if((r&127u)==0u) q=0;  // zero block scale
        if((r&4095u)==1u) q=0x7f; // reserved zero scale (baseline semantics)
        block[s]=q;
      }
      for(int j=0;j<32;++j) block[4+j]=(uint8_t)random_u32();
    }
    // Deliberate known-pattern rows catch wrong UE, nibble signs and pairing.
    if(row<4){
      for(int b=0;b<nb;++b){
        uint8_t* block=rw+(size_t)b*36;
        for(int s=0;s<4;++s)
          block[s]=(row==0)?0u:(row==3)?0x7fu:0x38u;
        for(int j=0;j<32;++j)
          block[4+j]=(row==2)?0x91u:0x11u;
      }
    }
  }
}
struct Shape { int H,F; };
static std::vector<int> sample_rows(int F){
  std::vector<int> rows;
  if(F<=256){
    for(int i=0;i<F;++i) rows.push_back(i);
  } else {
    for(int i=0;i<16;++i){ rows.push_back(i); rows.push_back(F-1-i); }
    for(int i=0;i<256;++i) rows.push_back((int)(((int64_t)i*(F-1))/255));
    std::sort(rows.begin(),rows.end());
    rows.erase(std::unique(rows.begin(),rows.end()),rows.end());
  }
  return rows;
}
static double median(std::vector<double> times){
  std::sort(times.begin(),times.end());
  return times[times.size()/2];
}
static double measure(Nvfp4GemvVariant variant, const uint16_t* x,
                      const uint8_t* w, float* y, int H, int F, int loops){
  EventPair ev;
  CUDA_CHECK(cudaEventRecord(ev.start));
  for(int i=0;i<loops;++i)
    hermes::nvfp4_gemv_cuda_variant(x,w,y,H,F,variant);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(ev.stop));
  CUDA_CHECK(cudaEventSynchronize(ev.stop));
  float elapsed=0;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed,ev.start,ev.stop));
  return (double)elapsed/loops;
}

static bool run_shape(Shape shape, int loops, int rounds, bool timing){
  const int H=shape.H,F=shape.F;
  const size_t wbytes=(size_t)F*(H/64)*36;
  Pinned<uint16_t> x(H);
  Pinned<uint8_t> weights(wbytes);
  Pinned<float> out_scalar(F+8),out_warp(F+8);
  Device<uint16_t> dx(H);
  Device<uint8_t> dw(wbytes);
  Device<float> dscalar(F+8), dwarp(F+8);
  make_input(x.p,weights.p,H,F);
  CUDA_CHECK(cudaMemcpy(dx.p,x.p,(size_t)H*2,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw.p,weights.p,wbytes,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dscalar.p,0xff,(size_t)(F+8)*sizeof(float)));
  CUDA_CHECK(cudaMemset(dwarp.p,0xff,(size_t)(F+8)*sizeof(float)));
  // The old five-argument API is intentionally the actual baseline under test.
  hermes::nvfp4_gemv_cuda(dx.p,dw.p,dscalar.p,H,F);
  hermes::nvfp4_gemv_cuda_variant(dx.p,dw.p,dwarp.p,H,F,Nvfp4GemvVariant::Warp4);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(out_scalar.p,dscalar.p,(size_t)(F+8)*4,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out_warp.p,dwarp.p,(size_t)(F+8)*4,cudaMemcpyDeviceToHost));

  int fail=0;
  double max_ab=0, max_a_cpu=0, max_b_cpu=0;
  for(int i=0;i<F;++i){
    const double a=out_scalar.p[i],b=out_warp.p[i];
    if(!std::isfinite(a)||!std::isfinite(b)){ ++fail; continue; }
    const double err=std::abs(a-b);
    max_ab=std::max(max_ab,err);
    if(err > 0.001+0.0004*std::max(std::abs(a),std::abs(b))) ++fail;
  }
  for(int i=F;i<F+8;++i){
    if(std::bit_cast<uint32_t>(out_scalar.p[i])!=0xffffffffu ||
       std::bit_cast<uint32_t>(out_warp.p[i])!=0xffffffffu) ++fail;
  }
  const auto rows=sample_rows(F);
  for(int row:rows){
    const double ref=cpu_ref(x.p,weights.p+(size_t)row*(H/64)*36,H);
    const double ea=std::abs((double)out_scalar.p[row]-ref);
    const double eb=std::abs((double)out_warp.p[row]-ref);
    max_a_cpu=std::max(max_a_cpu,ea);
    max_b_cpu=std::max(max_b_cpu,eb);
    const double tol=0.001+0.0004*std::abs(ref);
    if(!(ea<=tol && eb<=tol)) ++fail;
  }
  // A misaligned GGUF pointer must safely select the original scalar fallback.
  // Run only on a small shape to avoid allocating another full matrix.
  if(H==64 && F==5){
    Device<uint8_t> misaligned(wbytes+1);
    CUDA_CHECK(cudaMemcpy(misaligned.p+1,weights.p,wbytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dwarp.p,0xff,(size_t)(F+8)*sizeof(float)));
    hermes::nvfp4_gemv_cuda_variant(dx.p,misaligned.p+1,dwarp.p,H,F,Nvfp4GemvVariant::Warp4);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out_warp.p,dwarp.p,(size_t)F*4,cudaMemcpyDeviceToHost));
    for(int i=0;i<F;++i) if(out_warp.p[i]!=out_scalar.p[i]) ++fail;
  }
  std::printf("H=%d F=%d bytes=%zu: CPU_ref_rows=%zu A_vs_CPU_max=%.7g B_vs_CPU_max=%.7g "
              "A_vs_B_max=%.7g %s\n",H,F,wbytes,rows.size(),max_a_cpu,max_b_cpu,
              max_ab,fail?"FAIL":"PASS");
  if(fail){ std::fprintf(stderr,"  %d correctness/bounds failures; refusing to time shape\n",fail);return false; }

  if(timing){
    for(int i=0;i<10;++i){
      hermes::nvfp4_gemv_cuda_variant(dx.p,dw.p,dscalar.p,H,F,Nvfp4GemvVariant::Scalar);
      hermes::nvfp4_gemv_cuda_variant(dx.p,dw.p,dwarp.p,H,F,Nvfp4GemvVariant::Warp4);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> a,b;
    for(int r=0;r<rounds;++r){
      if((r&1)==0){
        a.push_back(measure(Nvfp4GemvVariant::Scalar,dx.p,dw.p,dscalar.p,H,F,loops));
        b.push_back(measure(Nvfp4GemvVariant::Warp4,dx.p,dw.p,dwarp.p,H,F,loops));
      } else {
        b.push_back(measure(Nvfp4GemvVariant::Warp4,dx.p,dw.p,dwarp.p,H,F,loops));
        a.push_back(measure(Nvfp4GemvVariant::Scalar,dx.p,dw.p,dscalar.p,H,F,loops));
      }
    }
    const double ams=median(a),bms=median(b);
    std::printf("  CUDA events (%d rounds x %d launches, median per launch; uploads excluded): "
                "scalar %.5f ms | warp4 %.5f ms | scalar/warp4 %.3fx | "
                "effective GGUF weight GB/s %.1f / %.1f\n",
                rounds,loops,ams,bms,ams/bms,
                (double)wbytes/1e6/ams,(double)wbytes/1e6/bms);
  }
  return true;
}

int main(int argc,char** argv){
  bool smoke=false,timing=true;
  int loops=100,rounds=7,device=0;
  for(int i=1;i<argc;++i){
    const std::string arg=argv[i];
    if(arg=="--smoke") smoke=true;
    else if(arg=="--no-perf") timing=false;
    else if(arg.rfind("--loops=",0)==0) loops=std::atoi(arg.c_str()+8);
    else if(arg.rfind("--rounds=",0)==0) rounds=std::atoi(arg.c_str()+9);
    else if(arg.rfind("--device=",0)==0) device=std::atoi(arg.c_str()+9);
    else {
      std::fprintf(stderr,"usage: %s [--smoke] [--no-perf] [--loops=N] "
                          "[--rounds=N] [--device=0|1]\n",argv[0]);
      return 2;
    }
  }
  if(loops<1||rounds<1||rounds>101){ std::fprintf(stderr,"invalid loops/rounds\n");return 2; }
  try {
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props,device));
    std::printf("device=%d %s SM%d%d | benchmark=synthetic GGUF NVFP4 "
                "| baseline=original scalar kernel | alternative=warp4 (opt-in)\n",
                device,props.name,props.major,props.minor);
    const std::vector<Shape> shapes=smoke
      ? std::vector<Shape>{{64,1},{64,5},{128,9},{512,64}}
      : std::vector<Shape>{{64,1},{64,5},{128,9},{512,64},
                           {5120,10240},{17408,5120},{5120,17408}};
    bool passed=true;
    for(Shape shape:shapes) passed=run_shape(shape,loops,rounds,timing)&&passed;
    std::printf("overall: %s\n",passed?"PASS":"FAIL");
    return passed?0:1;
  } catch(const std::exception& e){
    std::fprintf(stderr,"CUDA/benchmark error: %s\n",e.what());
    return 1;
  }
}
