// Isolated Warp4 alternatives benchmark, never linked into production.
// Build on SM120 in /tmp: nvcc -O3 -std=c++20 -arch=sm_120 -I kernels/include
//   kernels/src/nvfp4_gemv.cu kernels/src/rmsnorm.cu
//   kernels/src/nvfp4_gemv_warp_variants.cu
//   kernels/tests/bench_warp_variants.cu -o bench_warp_variants
// Run: ./bench_warp_variants --device=0 --loops=20 --rounds=5
// Warmups, CUDA events and order rotation; no upload or allocation timed.
// Synthetic GGUF 36B/64, BF16 activations, FP32 y; independent CPU double
// sampled rows, full-output comparisons and guard sentinels. No model TPS.

#include "hermes/nvfp4_gemv.hpp"
#include "hermes/nvfp4_gemv_warp_variants.hpp"
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
using hermes::Nvfp4WarpExperiment;
static void ck(cudaError_t result,const char* where){
  if(result!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(result));
}
#define CK(expr) ck((expr),#expr)
template<class T> struct Pinned {
  T* p=nullptr;
  explicit Pinned(size_t n){ CK(cudaMallocHost((void**)&p,n*sizeof(T))); }
  ~Pinned(){ if(p)cudaFreeHost(p); }
  Pinned(const Pinned&)=delete; Pinned& operator=(const Pinned&)=delete;
};
template<class T> struct Dev {
  T* p=nullptr;
  explicit Dev(size_t n){ CK(cudaMalloc((void**)&p,n*sizeof(T))); }
  ~Dev(){ if(p)cudaFree(p); }
  Dev(const Dev&)=delete; Dev& operator=(const Dev&)=delete;
};
struct Events {
  cudaEvent_t start=nullptr,stop=nullptr;
  Events(){ CK(cudaEventCreate(&start)); CK(cudaEventCreate(&stop)); }
  ~Events(){ if(start)cudaEventDestroy(start); if(stop)cudaEventDestroy(stop); }
};
static uint32_t seed=0x7f4a7c15u;
static uint32_t random_u32(){
  uint32_t x=seed;x^=x<<13;x^=x>>17;x^=x<<5;return seed=x;
}
static uint16_t to_bf16(float f){return (uint16_t)(std::bit_cast<uint32_t>(f)>>16);}
static float from_bf16(uint16_t f){return std::bit_cast<float>((uint32_t)f<<16);}
static float scale(uint8_t q){
  if(q==0||q==0x7f)return 0.f;
  int e=q>>3,m=q&7;
  return e==0?std::ldexp((float)m,-9):std::ldexp(1.f+m/8.f,e-7);
}
static double e2m1(unsigned q){
  static constexpr double values[16]={0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6};
  return values[q&15];
}
static double cpu_dot(const uint16_t* x,const uint8_t* row,int H){
  double acc=0;
  for(int b=0;b<H/64;++b){
    const uint8_t* block=row+(size_t)b*36;
    for(int s=0;s<4;++s){
      const double sc=scale(block[s]);
      for(int j=0;j<8;++j){
        const uint8_t q=block[4+s*8+j];
        const int k=b*64+s*16+j;
        acc+=(double)from_bf16(x[k])*sc*e2m1(q&15);
        acc+=(double)from_bf16(x[k+8])*sc*e2m1(q>>4);
      }
    }
  }
  return acc;
}
static void make_data(uint16_t* x,uint8_t* W,int H,int F){
  for(int k=0;k<H;++k){
    float f=(float)((int)(random_u32()%2049)-1024)/1024.f;
    if(k%127==0)f=0;
    x[k]=to_bf16(f);
  }
  const int nb=H/64;
  for(int row=0;row<F;++row){
    uint8_t* ptr=W+(size_t)row*nb*36;
    for(int b=0;b<nb;++b){
      uint8_t* block=ptr+(size_t)b*36;
      for(int s=0;s<4;++s){
        uint32_t r=random_u32();
        uint8_t q=(uint8_t)(((r%4u)<<3)|((r>>8)&7u));
        if((r&127u)==0u)q=0;
        if((r&4095u)==1u)q=0x7f;
        block[s]=q;
      }
      for(int j=0;j<32;++j)block[4+j]=(uint8_t)random_u32();
      if(row<4){
        for(int s=0;s<4;++s)
          block[s]=(row==0)?0u:(row==3)?0x7fu:0x38u;
        for(int j=0;j<32;++j)
          block[4+j]=(row==2)?0x91u:0x11u;
      }
    }
  }
}
static std::vector<int> sampled_rows(int F){
  std::vector<int> rows;
  if(F<=256){ for(int i=0;i<F;++i)rows.push_back(i); }
  else {
    for(int i=0;i<16;++i){rows.push_back(i);rows.push_back(F-1-i);}
    for(int i=0;i<256;++i)rows.push_back((int)((int64_t)i*(F-1)/255));
    std::sort(rows.begin(),rows.end());
    rows.erase(std::unique(rows.begin(),rows.end()),rows.end());
  }
  return rows;
}
struct Shape{int H,F;};
struct Variant{ const char* label; int mode; };
static constexpr Variant variants[]={
  {"scalar",-2},{"production-Warp4",-1},
  {"packed-u16x2",0},{"packed-u32x4",1},{"unroll-2",2},
  {"split-2warps",3},{"split-4warps",4},{"warp8-rows",5}
};
static constexpr size_t nvar=sizeof(variants)/sizeof(variants[0]);

static void launch(const Variant& variant,const uint16_t* x,const uint8_t* W,
                   float* y,int H,int F){
  if(variant.mode==-2){
    hermes::nvfp4_gemv_cuda_variant(x,W,y,H,F,Nvfp4GemvVariant::Scalar);
  } else if(variant.mode==-1){
    hermes::nvfp4_gemv_cuda_variant(x,W,y,H,F,Nvfp4GemvVariant::Warp4);
  } else {
    CK(hermes::nvfp4_gemv_warp_experiment(x,W,y,H,F,
          static_cast<Nvfp4WarpExperiment>(variant.mode)));
  }
  CK(cudaPeekAtLastError());
}
static double measure(const Variant& variant,const uint16_t* x,const uint8_t* W,
                      float* y,int H,int F,int loops){
  Events ev;
  CK(cudaEventRecord(ev.start));
  for(int i=0;i<loops;++i)launch(variant,x,W,y,H,F);
  CK(cudaEventRecord(ev.stop));
  CK(cudaEventSynchronize(ev.stop));
  float ms=0;CK(cudaEventElapsedTime(&ms,ev.start,ev.stop));
  return (double)ms/loops;
}
static double median(std::vector<double> values){
  std::sort(values.begin(),values.end());return values[values.size()/2];
}
static bool analyze(int H,int F,int loops,int rounds,bool do_timing){
  const size_t nbytes=(size_t)F*(H/64)*36;
  const auto sampled=sampled_rows(F);
  size_t free_before=0,total=0;
  CK(cudaMemGetInfo(&free_before,&total));
  const size_t request=nbytes+(size_t)H*2+(size_t)(F+8)*nvar*4;
  if(request+64*1024*1024>free_before){
    std::printf("SKIP H=%d F=%d freeVRAM=%zu requested=%zu\n",H,F,free_before,request);
    return false;
  }
  Pinned<uint16_t> input(H);Pinned<uint8_t> weights(nbytes);
  Pinned<float> host((F+8)*nvar);
  Dev<uint16_t> dx(H);Dev<uint8_t> dw(nbytes);
  Dev<float> outputs((F+8)*nvar);
  make_data(input.p,weights.p,H,F);
  CK(cudaMemcpy(dx.p,input.p,H*2,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dw.p,weights.p,nbytes,cudaMemcpyHostToDevice));
  CK(cudaMemset(outputs.p,0xff,(F+8)*nvar*4));
  size_t free_after=0;
  CK(cudaMemGetInfo(&free_after,&total));
  std::printf("H=%d F=%d weights=%.3f MiB allocatedVRAM=%.3f MiB "
              "free_before=%.1f MiB free_after=%.1f MiB\n",
              H,F,nbytes/1048576.,(free_before-free_after)/1048576.,
              free_before/1048576.,free_after/1048576.);
  for(size_t v=0;v<nvar;++v){
    float* out=outputs.p+v*(F+8);
    launch(variants[v],dx.p,dw.p,out,H,F);
  }
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(host.p,outputs.p,(F+8)*nvar*4,cudaMemcpyDeviceToHost));

  // Truly independent double reference. Compute once per row, then compare
  // every variant against it (including scalar and production Warp4).
  std::vector<double> ref(sampled.size());
  for(size_t k=0;k<sampled.size();++k)
    ref[k]=cpu_dot(input.p,weights.p+(size_t)sampled[k]*(H/64)*36,H);
  const float* scalar=host.p;
  const float* warp4=host.p+(F+8);
  const int greedy_ref=(int)(std::max_element(scalar,scalar+F)-scalar);
  int failures=0;
  for(size_t v=0;v<nvar;++v){
    const float* vals=host.p+v*(F+8);
    const int greedy=(int)(std::max_element(vals,vals+F)-vals);
    double abmax=0,bwmax=0,cpumax=0;
    int fail=0;
    for(int r=0;r<F;++r){
      const double got=vals[r],a=scalar[r],b=warp4[r];
      if(!(std::isfinite(got)&&std::isfinite(a)&&std::isfinite(b))){ ++fail;continue; }
      const double ea=std::abs(got-a), eb=std::abs(got-b);
      abmax=std::max(abmax,ea);bwmax=std::max(bwmax,eb);
      const double tol=0.0001+0.0002*std::max(std::abs(a),std::abs(b));
      if(ea>tol||eb>tol)++fail;
    }
    for(int r=F;r<F+8;++r)
      if(std::bit_cast<uint32_t>(vals[r])!=0xffffffffu)++fail;
    for(size_t k=0;k<sampled.size();++k){
      const int r=sampled[k];
      const double e=std::abs((double)vals[r]-ref[k]);
      cpumax=std::max(cpumax,e);
      if(e>0.0001+0.0002*std::abs(ref[k]))++fail;
    }
    if(greedy!=greedy_ref)++fail;
    failures+=fail;
    std::printf("  %-17s Amax=%.7g W4max=%.7g CPUmax=%.7g "
                "CPUrows=%zu greedy=%d ref=%d fail=%d %s\n",
                variants[v].label,abmax,bwmax,cpumax,sampled.size(),greedy,
                greedy_ref,fail,fail?"FAIL":"PASS");
  }
  if(failures){
    std::fprintf(stderr,"Shape H=%d F=%d validation FAILED, no timing\n",H,F);
    return false;
  }
  if(!do_timing)return true;
  for(int i=0;i<5;++i)
    for(size_t v=0;v<nvar;++v)
      launch(variants[v],dx.p,dw.p,outputs.p+v*(F+8),H,F);
  CK(cudaDeviceSynchronize());
  std::vector<double> readings[nvar];
  // Rotate order every round; alternate direction to limit warm-cache bias.
  for(int round=0;round<rounds;++round){
    for(size_t j=0;j<nvar;++j){
      const size_t i=(round&1)?(nvar-1-(j+round)%nvar):((j+round)%nvar);
      const auto val=measure(variants[i],dx.p,dw.p,
                             outputs.p+i*(F+8),H,F,loops);
      readings[i].push_back(val);
    }
  }
  const double baseline=median(readings[1]);
  for(size_t v=0;v<nvar;++v){
    const double ms=median(readings[v]);
    std::printf("  TIME %-17s %.6f ms Warp4/variant %.3fx effectiveBW %.1f GB/s\n",
        variants[v].label,ms,baseline/ms,nbytes/1e6/ms);
  }
  return true;
}
int main(int argc,char** argv){
  bool smoke=false,timing=true;
  int device=0,loops=20,rounds=5;
  for(int i=1;i<argc;++i){
    std::string arg=argv[i];
    if(arg=="--smoke")smoke=true;
    else if(arg=="--no-perf")timing=false;
    else if(arg.rfind("--loops=",0)==0)loops=std::atoi(arg.c_str()+8);
    else if(arg.rfind("--rounds=",0)==0)rounds=std::atoi(arg.c_str()+9);
    else if(arg.rfind("--device=",0)==0)device=std::atoi(arg.c_str()+9);
    else {std::fprintf(stderr,"Unknown argument: %s\n",argv[i]);return 2;}
  }
  if(loops<1||rounds<1||rounds>30||device<0){
    std::fprintf(stderr,"invalid arguments\n");return 2;
  }
  try{
    CK(cudaSetDevice(device));
    cudaDeviceProp prop{};CK(cudaGetDeviceProperties(&prop,device));
    std::printf("GPU=%d %s SM%d%d, loops=%d rounds=%d; "
                "kernel-only CUDA event timing; no persistent GEMV VRAM\n",
        device,prop.name,prop.major,prop.minor,loops,rounds);
    if(hermes::nvfp4_gemv_warp_experiment(nullptr,nullptr,nullptr,64,1,
          Nvfp4WarpExperiment::PackedU32x4)!=cudaErrorInvalidValue){
      std::fprintf(stderr,"Invalid pointer not rejected\n");return 1;
    }
    const std::vector<Shape> shapes=smoke
      ? std::vector<Shape>{{64,1},{64,5},{128,9},{512,64}}
      : std::vector<Shape>{{64,1},{64,5},{128,9},{512,64},
                           {5120,10240},{17408,5120},{5120,17408}};
    bool all=true;
    for(Shape s:shapes)all=analyze(s.H,s.F,loops,rounds,timing)&&all;
    std::printf("overall %s\n",all?"PASS":"FAIL");
    return all?0:1;
  }catch(const std::exception& e){
    std::fprintf(stderr,"benchmark error: %s\n",e.what());return 1;
  }
}
