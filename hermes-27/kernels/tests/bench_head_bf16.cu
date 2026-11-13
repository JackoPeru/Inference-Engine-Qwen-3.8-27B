// Standalone A/B for the HERMES PP2 BF16 vocabulary head; isolated scratch build:
// nvcc -O3 -std=c++20 -arch=sm_120 -I kernels/include \
//   kernels/src/head_bf16_optimized.cu kernels/tests/bench_head_bf16.cu \
//   -o bench_head_bf16
// ./bench_head_bf16 --smoke --no-perf
// ./bench_head_bf16 --device=1 --loops=5 --rounds=5
// Includes the original per-output-row FP64 kernel reproduced here (benchmark
// only); compares against independent CPU double reference on sampled rows.
// Full vocab: 248320 x 5120 x BF16 = 2,542,796,800 B (~2.37 GiB).
// Synthetic matrix is filled ON GPU to avoid multi-GiB host pinned allocation.
// CUDA-event per-launch latency excludes data initialization, copies, allocation
// and synchronization; result is kernel-only, NOT end-to-end decode TPS.
// Measured on 2026-11-13, RTX 5060 Ti SM120, nvcc -O3, GPU1, median of
// 5 rounds x 5 launches, synthetic weights generated in VRAM:
//  T=1,H=5120,V=248320: scalar FP64 21.45844 ms, warp FP32 5.95840 ms
//    => 3.601x kernel-only. All 248320 logits: max|delta|=1.90735e-6,
//    RMS=4.09947e-7, greedy 135410 matched, reference top1 gap 0.595694.
//    286 CPU double-precision sampled rows matched within 1.19347e-6.
//  T=2,H=5120,V=4096: scalar FP64 0.78925 ms, warp FP32 0.15879 ms
//    => 4.970x kernel-only. All 8192 logits/greedy matched.
// Representative dimensions only; no real-model or near-tie accuracy guarantee.

#include "hermes/head_bf16_optimized.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t status,const char* where){
  if(status!=cudaSuccess)
    throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(status));
}
#define CUDA(expr) check((expr),#expr)

template<class T> struct Device {
  T* ptr=nullptr;
  explicit Device(size_t count){ CUDA(cudaMalloc((void**)&ptr,count*sizeof(T))); }
  ~Device(){if(ptr)cudaFree(ptr);}
  Device(const Device&)=delete;
  Device& operator=(const Device&)=delete;
};
template<class T> struct Pinned {
  T* ptr=nullptr;
  explicit Pinned(size_t count){ CUDA(cudaMallocHost((void**)&ptr,count*sizeof(T))); }
  ~Pinned(){if(ptr)cudaFreeHost(ptr);}
  Pinned(const Pinned&)=delete;
  Pinned& operator=(const Pinned&)=delete;
};
struct Events {
  cudaEvent_t begin=nullptr,end=nullptr;
  Events(){ CUDA(cudaEventCreate(&begin)); CUDA(cudaEventCreate(&end)); }
  ~Events(){ if(begin)cudaEventDestroy(begin); if(end)cudaEventDestroy(end); }
};

// Reference identical to pp2_fwd.cu:228-239: thread per [t,v] and double
// accumulation of FP32 xn * BF16 row. Conversion via bit expansion is exact.
__global__ void head_bf16_scalar_fp64_reference_k(float* logits,const float* xn,
                                                  const uint16_t* W,int T,int H,int V){
  const int idx=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  const int N=T*V;
  if(idx>=N)return;
  const int t=idx/V,v=idx%V;
  const uint16_t* row=W+(size_t)v*H;
  double a=0.0;
  for(int k=0;k<H;++k){
    const float w=__uint_as_float((uint32_t)row[k]<<16);
    a+=(double)xn[(size_t)t*H+k]*(double)w;
  }
  logits[idx]=(float)a;
}

__device__ __forceinline__ uint32_t hash32(uint32_t x){
  x^=x>>16; x*=0x7feb352du;
  x^=x>>15; x*=0x846ca68bu;
  x^=x>>16; return x;
}
__global__ void init_weights_k(uint32_t* packed,size_t pairs,int H,uint32_t seed){
  const size_t tid=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  const size_t stride=(size_t)gridDim.x*blockDim.x;
  for(size_t i=tid;i<pairs;i+=stride){
    const int row=(int)(i / ((size_t)H/2));
    // BF16-exact values in [-0.0625, 0.0625]. Deterministic across devices;
    // special rows test +0/-0, positive uniform, negative uniform.
    int a=0,b=0;
    if(row==1){ a=64;b=64; }
    else if(row==2){ a=-64;b=-64; }
    else if(row>=3){
      uint32_t r0=hash32((uint32_t)i*2u+seed);
      uint32_t r1=hash32((uint32_t)i*2u+seed+0x9e3779b9u);
      a=(int)(r0&127u)-64;
      b=(int)(r1&127u)-64;
    }
    const float fa=(float)a*0x1p-10f,fb=(float)b*0x1p-10f;
    packed[i]=(__float_as_uint(fa)>>16) | (__float_as_uint(fb)&0xffff0000u);
  }
}

static float to_float(uint16_t b){return std::bit_cast<float>((uint32_t)b<<16);}
static uint32_t rng_state=0x4b239b51u;
static uint32_t next_random(){
  uint32_t x=rng_state;
  x^=x<<13; x^=x>>17; x^=x<<5;
  return rng_state=x;
}
static void init_input(float* x,int H,int T){
  for(int t=0;t<T;++t){
    for(int k=0;k<H;++k){
      // FP32 with BF16-inexact fractions to exercise conversion + rounding.
      const int s=(int)(next_random()%2049u)-1024;
      const float v=(float)s/513.f;
      x[(size_t)t*H+k]= k%113==0 ? 0.f : v;
    }
  }
}

struct Shape {int T,H,V;};
static std::vector<int> rows_to_sample(int V){
  std::vector<int> rows;
  if(V<=128){ for(int i=0;i<V;++i)rows.push_back(i); }
  else {
    for(int i=0;i<16;++i){rows.push_back(i);rows.push_back(V-1-i);}
    for(int i=0;i<256;++i)
      rows.push_back((int)((int64_t)i*(V-1)/255));
    std::sort(rows.begin(),rows.end());
    rows.erase(std::unique(rows.begin(),rows.end()),rows.end());
  }
  return rows;
}
static double cpu_dot(const float* x,const uint16_t* row,int H){
  double a=0;
  for(int k=0;k<H;++k) a+=(double)x[k]*(double)to_float(row[k]);
  return a;
}
static double median(std::vector<double> values){
  std::sort(values.begin(),values.end());
  return values[values.size()/2];
}
static void launch_reference(float* y,const float* x,const uint16_t* W,Shape s){
  const int N=s.T*s.V;
  head_bf16_scalar_fp64_reference_k<<<(N+255)/256,256>>>(y,x,W,s.T,s.H,s.V);
}
static void launch_warp(float* y,const float* x,const uint16_t* W,Shape s){
  CUDA(hermes::head_bf16_warp_cuda(y,x,W,s.T,s.H,s.V));
}
static double measure(bool warp, float* y,const float* x,const uint16_t* W,
                      Shape s,int loops){
  Events ev;
  CUDA(cudaEventRecord(ev.begin));
  for(int i=0;i<loops;++i){
    if(warp)launch_warp(y,x,W,s);
    else launch_reference(y,x,W,s);
  }
  CUDA(cudaGetLastError());
  CUDA(cudaEventRecord(ev.end));
  CUDA(cudaEventSynchronize(ev.end));
  float milliseconds=0.f;
  CUDA(cudaEventElapsedTime(&milliseconds,ev.begin,ev.end));
  return (double)milliseconds/loops;
}
static bool run_shape(Shape s,bool perf,int loops,int rounds){
  const size_t weights=(size_t)s.V*(size_t)s.H*2;
  const size_t output=(size_t)s.T*(size_t)s.V;
  if(output>(size_t)std::numeric_limits<int>::max())
    throw std::runtime_error("reference baseline index overflow");
  const auto samples=rows_to_sample(s.V);
  size_t free=0,total=0;
  CUDA(cudaMemGetInfo(&free,&total));
  const size_t request=weights+output*8+(size_t)s.T*s.H*4;
  if(request+128ull*1024*1024>free){
    std::printf("T=%d H=%d V=%d SKIP insufficient free VRAM: requested %.3fGiB, free %.3fGiB\n",
        s.T,s.H,s.V,(double)request/1073741824.,(double)free/1073741824.);
    return false;
  }
  Pinned<float> x((size_t)s.T*s.H);
  Pinned<float> golden(output+16),candidate(output+16);
  Pinned<uint16_t> samples_w(samples.size()*(size_t)s.H);
  Device<float> dx((size_t)s.T*s.H),dgolden(output+16),dcandidate(output+16);
  Device<uint16_t> dw(weights/2);
  init_input(x.ptr,s.H,s.T);
  CUDA(cudaMemcpy(dx.ptr,x.ptr,(size_t)s.T*s.H*4,cudaMemcpyHostToDevice));
  init_weights_k<<<2048,256>>>(reinterpret_cast<uint32_t*>(dw.ptr),weights/4,
                               s.H,0x6f2c571du);
  CUDA(cudaGetLastError());
  CUDA(cudaMemset(dgolden.ptr,0xff,(output+16)*4));
  CUDA(cudaMemset(dcandidate.ptr,0xff,(output+16)*4));
  launch_reference(dgolden.ptr,dx.ptr,dw.ptr,s);
  launch_warp(dcandidate.ptr,dx.ptr,dw.ptr,s);
  CUDA(cudaGetLastError());
  // Stream-ordered device->host transfers only after both kernel launches.
  CUDA(cudaMemcpy(golden.ptr,dgolden.ptr,(output+16)*4,cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpy(candidate.ptr,dcandidate.ptr,(output+16)*4,cudaMemcpyDeviceToHost));
  for(size_t i=0;i<samples.size();++i){
    const uint16_t* src=dw.ptr+(size_t)samples[i]*s.H;
    CUDA(cudaMemcpyAsync(samples_w.ptr+i*(size_t)s.H,src,(size_t)s.H*2,
                         cudaMemcpyDeviceToHost));
  }
  CUDA(cudaStreamSynchronize(0));

  size_t wrong=0,checked_cpu=0,greedy_mismatch=0;
  double max_ab=0,mean_square=0,max_cpu_a=0,max_cpu_b=0,max_rel=0;
  for(size_t i=0;i<output;++i){
    const double a=golden.ptr[i],b=candidate.ptr[i];
    if(!std::isfinite(a)||!std::isfinite(b)){ ++wrong;continue; }
    const double err=std::abs(a-b);
    max_ab=std::max(max_ab,err);
    mean_square+=err*err;
    if(std::abs(a)>=0.1) max_rel=std::max(max_rel,err/std::abs(a));
    if(err>0.0005+0.0003*std::abs(a)) ++wrong;
  }
  for(size_t i=output;i<output+16;++i){
    if(std::bit_cast<uint32_t>(golden.ptr[i])!=0xffffffffu ||
       std::bit_cast<uint32_t>(candidate.ptr[i])!=0xffffffffu) ++wrong;
  }
  for(int t=0;t<s.T;++t){
    const float* r=golden.ptr+(size_t)t*s.V;
    const float* p=candidate.ptr+(size_t)t*s.V;
    const size_t a=(size_t)(std::max_element(r,r+s.V)-r);
    const size_t b=(size_t)(std::max_element(p,p+s.V)-p);
    float second=-std::numeric_limits<float>::infinity();
    for(int i=0;i<s.V;++i)if((size_t)i!=a)second=std::max(second,r[i]);
    std::printf("  t=%d greedy ref=%zu opt=%zu gap_ref=%.7g %s\n",
        t,a,b,r[a]-second,a==b?"MATCH":"MISMATCH");
    greedy_mismatch += a!=b;
    const float* tx=x.ptr+(size_t)t*s.H;
    for(size_t j=0;j<samples.size();++j){
      const int v=samples[j];
      const double ref=cpu_dot(tx,samples_w.ptr+j*(size_t)s.H,s.H);
      const double ea=std::abs((double)r[v]-ref);
      const double eb=std::abs((double)p[v]-ref);
      max_cpu_a=std::max(max_cpu_a,ea);
      max_cpu_b=std::max(max_cpu_b,eb);
      if(ea>0.0001+0.000001*std::abs(ref) ||
         eb>0.0005+0.0003*std::abs(ref))++wrong;
      ++checked_cpu;
    }
  }
  const double rmse=std::sqrt(mean_square/output);
  std::printf("T=%d H=%d V=%d W=%.3f GiB: all %zu logits vs FP64 max_abs=%.9g "
              "rms_abs=%.9g max_rel_ge0.1=%.7g CPU_ref_n=%zu ref_max=%.9g opt_max=%.9g "
              "wrong=%zu greedy_mismatch=%zu %s\n",
      s.T,s.H,s.V,(double)weights/1073741824.,output,max_ab,rmse,max_rel,
      checked_cpu,max_cpu_a,max_cpu_b,wrong,greedy_mismatch,
      (!wrong&&!greedy_mismatch)?"PASS":"FAIL");
  if(wrong || greedy_mismatch) return false;

  if(perf){
    for(int i=0;i<4;++i){launch_reference(dgolden.ptr,dx.ptr,dw.ptr,s);
                             launch_warp(dcandidate.ptr,dx.ptr,dw.ptr,s);}
    CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
    std::vector<double> a,b;
    for(int i=0;i<rounds;++i){
      if(i&1){
        b.push_back(measure(true,dcandidate.ptr,dx.ptr,dw.ptr,s,loops));
        a.push_back(measure(false,dgolden.ptr,dx.ptr,dw.ptr,s,loops));
      } else {
        a.push_back(measure(false,dgolden.ptr,dx.ptr,dw.ptr,s,loops));
        b.push_back(measure(true,dcandidate.ptr,dx.ptr,dw.ptr,s,loops));
      }
    }
    const double oldms=median(a),newms=median(b);
    std::printf("  CUDA events median %d rounds x %d launches: FP64-scalar %.5f ms "
                "FP32-warp %.5f ms speedup %.3fx effective_weight_GBps %.1f/%.1f\n",
        rounds,loops,oldms,newms,oldms/newms,
        (double)weights*s.T/(oldms*1e6),(double)weights*s.T/(newms*1e6));
  }
  return true;
}

int main(int argc,char** argv){
  int device=1,loops=5,rounds=5;
  bool smoke=false,quick=false,perf=true;
  for(int i=1;i<argc;++i){
    std::string arg=argv[i];
    if(arg=="--smoke")smoke=true;
    else if(arg=="--quick")quick=true;
    else if(arg=="--no-perf")perf=false;
    else if(arg.rfind("--device=",0)==0)device=std::atoi(arg.c_str()+9);
    else if(arg.rfind("--loops=",0)==0)loops=std::atoi(arg.c_str()+8);
    else if(arg.rfind("--rounds=",0)==0)rounds=std::atoi(arg.c_str()+9);
    else {
      std::fprintf(stderr,"usage: %s [--smoke|--quick] [--no-perf] [--device=N] "
                          "[--loops=N] [--rounds=N]\n",argv[0]);
      return 2;
    }
  }
  if(device<0||loops<=0||rounds<=0||rounds>30){
    std::fprintf(stderr,"invalid device/loops/rounds\n"); return 2;
  }
  try{
    CUDA(cudaSetDevice(device));
    cudaDeviceProp props{};
    CUDA(cudaGetDeviceProperties(&props,device));
    std::printf("device=%d %s SM%d%d baseline=PP2 FP64 scalar, alt=BF16 FP32 warp\n",
        device,props.name,props.major,props.minor);
    // Invalid inputs never launch or sync (negative test).
    if(hermes::head_bf16_warp_cuda(nullptr,nullptr,nullptr,1,5120,248320)
          !=cudaErrorInvalidValue){
      std::fprintf(stderr,"invalid-pointer validation failed\n");return 1;
    }
    bool ok=true;
    const std::vector<Shape> shapes=smoke
      ? std::vector<Shape>{{1,2,1},{1,64,5},{2,128,17},{3,512,65}}
      : quick
        ? std::vector<Shape>{{1,2,1},{1,64,5},{2,128,17},{3,512,65},
                             {2,5120,4096}}
        : std::vector<Shape>{{1,2,1},{1,64,5},{2,128,17},{3,512,65},
                             {2,5120,4096},{1,5120,248320}};
    for(auto s:shapes)ok=run_shape(s,perf,loops,rounds)&&ok;
    std::printf("overall %s\n",ok?"PASS":"FAIL");
    return ok?0:1;
  }catch(const std::exception& e){
    std::fprintf(stderr,"benchmark error: %s\n",e.what());
    return 1;
  }
}
