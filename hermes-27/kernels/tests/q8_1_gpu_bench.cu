// ISOLATED, EXPLICITLY OPT-IN CUDA benchmark. No production route or CMake.
// Never execute without prime granting GPU exclusivity. Compiling is offline.
//
// Builds against existing model libhermes_gguf.a; uses exact captured
// llama.cpp attn_norm-0.f32 input (token 6205, pos0) and genuine raw GGUF
// blk.0.attn_qkv.weight. Extra token rows are deterministic perturbations,
// NOT actual llama runtime captures and NOT a full-model quality test.
#include "hermes/nvfp4_gemv_q8_1_experimental.hpp"
#include "hermes/gguf.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace q8=hermes::q8_1_experimental;
using Block=q8::Q8_1Block;
static_assert(sizeof(_Float16)==2);

static void ck(cudaError_t e,const char* where) {
  if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+
                                            cudaGetErrorString(e));
}
struct Buffers {
  std::vector<void*> ptrs;
  template<class T> T* alloc(size_t n) {
    T* v=nullptr;
    ck(cudaMalloc(reinterpret_cast<void**>(&v),n*sizeof(T)),"cudaMalloc");
    ptrs.push_back(v);return v;
  }
  ~Buffers(){for(void* p:ptrs)cudaFree(p);}
};
struct Events {
  cudaEvent_t a=nullptr,b=nullptr;
  Events(){ck(cudaEventCreate(&a),"event a");ck(cudaEventCreate(&b),"event b");}
  ~Events(){if(a)cudaEventDestroy(a);if(b)cudaEventDestroy(b);}
  template<class Fn> float ms(Fn work,int count) {
    work();ck(cudaDeviceSynchronize(),"warmup");
    ck(cudaEventRecord(a),"start");
    for(int i=0;i<count;++i) work();
    ck(cudaEventRecord(b),"stop");
    ck(cudaEventSynchronize(b),"sync");
    float t=0;ck(cudaEventElapsedTime(&t,a,b),"elapsed");
    return t/count;
  }
};
static std::vector<uint8_t> read_bytes(std::ifstream& f,uint64_t off,size_t len) {
  std::vector<uint8_t> result(len);
  f.clear();f.seekg(static_cast<std::streamoff>(off));
  if(!f.read(reinterpret_cast<char*>(result.data()),
             static_cast<std::streamsize>(len)))
    throw std::runtime_error("GGUF tensor read fail");
  return result;
}
static std::vector<float> f32_file(const std::string& name,size_t expected) {
  std::ifstream f(name,std::ios::binary | std::ios::ate);
  if(!f || f.tellg()!=std::streamoff(expected*4))
    throw std::runtime_error("bad capture size: "+name);
  f.seekg(0);
  std::vector<float> result(expected);
  if(!f.read(reinterpret_cast<char*>(result.data()),expected*4))
    throw std::runtime_error("capture read fail");
  for(float v:result)if(!std::isfinite(v))
    throw std::runtime_error("capture contains nonfinite FP32");
  return result;
}
static std::vector<Block> cpu_quant(const std::vector<float>& x) {
  if(x.size()%32)throw std::runtime_error("Q8 block size mismatch");
  std::vector<Block> b(x.size()/32);
  for(size_t ib=0;ib<b.size();++ib) {
    const float* v=x.data()+ib*32;
    float amax=0.f;
    for(int i=0;i<32;++i)amax=fmaxf(amax,fabsf(v[i]));
    float d=amax/127.f;
    b[ib].d=0;b[ib].s=0;
    const _Float16 d_half=static_cast<_Float16>(d);
    std::memcpy(&b[ib].d,&d_half,2);
    float sum[32];std::memcpy(sum,v,32*4);
    for(int step=16;step;step>>=1) {
      float next[32];
      for(int k=0;k<32;++k) next[k]=sum[k]+sum[k^step];
      std::memcpy(sum,next,sizeof(next));
    }
    const _Float16 s_half=static_cast<_Float16>(sum[0]);
    std::memcpy(&b[ib].s,&s_half,2);
    for(int k=0;k<32;++k) {
      int quant=amax==0.f?0:int(roundf(v[k]/d));
      if(quant>127||quant< -127)throw std::runtime_error("Q8 int8 overflow");
      b[ib].qs[k]=int8_t(quant);
    }
  }
  return b;
}
static float ue4(uint8_t c) {
  if(c==0u || c==127u)return 0.f;
  const int exp=c>>3;
  const int m=c&7;
  return exp==0 ? ldexpf(float(m),-9)
    : ldexpf(1.f+float(m)/8.f,exp-7);
}
static int twice_e2(unsigned q) {
  static const int lut[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
  return lut[q&15u];
}
static double cpu_projection(const std::vector<Block>& q,
                             const uint8_t* w, int K, int t, int n) {
  const uint8_t* wr=w+size_t(n)*(K/64*36);
  double sum=0.;
  for(int b=0;b<K/64;++b) {
    const uint8_t* blk=wr+b*36;
    for(int s=0;s<4;++s) {
      const Block& xb=q[size_t(t)*(K/32)+b*2+(s>>1)];
      _Float16 h;
      std::memcpy(&h,&xb.d,2);
      int dot=0;
      for(int j=0;j<8;++j) {
        int idx=(s&1)*16+j;
        unsigned c=blk[4+s*8+j];
        dot+=int(xb.qs[idx])*twice_e2(c&15);
        dot+=int(xb.qs[idx+8])*twice_e2(c>>4);
      }
      sum+=double(dot)*(.5*double(ue4(blk[s])))*double(h);
    }
  }
  return sum;
}

int main(int argc,char**argv) {
  try {
    std::string gguf,capture;
    int T=64,device=1,loops=10;
    bool dp4a_ab=false;
    for(int i=1;i<argc;++i) {
      std::string key=argv[i];
      auto val=[&] {
        if(i+1==argc)throw std::runtime_error("missing arg for "+key);
        return std::string(argv[++i]);
      };
      if(key=="--gguf")gguf=val();
      else if(key=="--capture")capture=val();
      else if(key=="--T")T=std::stoi(val());
      else if(key=="--device")device=std::stoi(val());
      else if(key=="--loops")loops=std::stoi(val());
      else if(key=="--dp4a-ab")dp4a_ab=true;
      else if(key=="--help") {
        std::puts("usage: q8_1_gpu_bench --gguf REAL_MODEL --capture LLAMA_DIR "
                  "[--T 1|16|64|128|256] [--device 1] [--loops 10] "
                  "[--dp4a-ab (explicit opt-in)]");
        return 0;
      } else throw std::runtime_error("bad option "+key);
    }
    if(gguf.empty()||capture.empty()||T<1||T>256||loops<1||loops>50)
      throw std::runtime_error("invalid gguf/capture/T/loops");
    hermes::GgufFile g;g.open(gguf,true);
    const auto& w=g.find("blk.0.attn_qkv.weight");
    if(w.type!=hermes::GgmlType::NVFP4 || w.ne[0]!=5120 ||
       w.ne[1]!=10240||w.n_bytes!=29491200)
      throw std::runtime_error("unexpected GGUF tensor shape");
    constexpr int K=5120,N=10240;
    std::ifstream f(gguf,std::ios::binary);
    if(!f)throw std::runtime_error("model cannot open");
    auto raw=read_bytes(f,g.data_offset+w.offset,size_t(w.n_bytes));
    auto captured=f32_file(capture+"/attn_norm-0.f32",K);
    auto expected_llama=f32_file(capture+"/linear_attn_qkv_mixed-0.f32",N);
    std::vector<float> x(size_t(T)*K);
    for(int t=0;t<T;++t)
      for(int k=0;k<K;++k)
        x[size_t(t)*K+k]=(t==0) ? captured[k] :
          captured[k]+0.001f*sinf(float(t*13+k*7)*.01f);
    const auto cpu=cpu_quant(x);
    std::printf("INPUT first_token_llama_authentic=true additional_rows=deterministic_perturbations T=%d\n",T);
    ck(cudaSetDevice(device),"CUDA device");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop,device),"device properties");
    std::printf("GPU=%d name=%s sm=%d.%d\n",device,prop.name,prop.major,prop.minor);
    if(prop.major<12)throw std::runtime_error("SM120 required for this bench");
    size_t before=0,total=0,after=0;
    ck(cudaMemGetInfo(&before,&total),"free VRAM before");
    Buffers mem;
    auto* dx=mem.alloc<float>(x.size());
    auto* dw=mem.alloc<uint8_t>(raw.size());
    auto* dq=mem.alloc<Block>(cpu.size());
    auto* dy=mem.alloc<float>(size_t(T)*N);
    // Explicit A/B only; the baseline's allocation and dispatch are
    // unchanged if --dp4a-ab is absent.
    float* dFast=dp4a_ab ? mem.alloc<float>(size_t(T)*N) : nullptr;
    ck(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice),"H2D X");
    ck(cudaMemcpy(dw,raw.data(),raw.size(),cudaMemcpyHostToDevice),"H2D W");
    ck(q8::project_f32_q8_1_nvfp4(dx,dw,dq,dy,T,N,K),"Q8_1 projection");
    ck(cudaDeviceSynchronize(),"projection completion");
    ck(cudaMemGetInfo(&after,&total),"free VRAM after");
    auto qb=std::vector<Block>(cpu.size());
    auto y=std::vector<float>(size_t(T)*N);
    ck(cudaMemcpy(qb.data(),dq,qb.size()*sizeof(Block),cudaMemcpyDeviceToHost),
       "D2H Q8");
    ck(cudaMemcpy(y.data(),dy,y.size()*4,cudaMemcpyDeviceToHost),"D2H Y");
    size_t bad_q=0,bad_d=0,bad_s=0;
    for(size_t ib=0;ib<cpu.size();++ib) {
      bad_q+=std::memcmp(cpu[ib].qs,qb[ib].qs,32)!=0;
      bad_d+=cpu[ib].d!=qb[ib].d;
      bad_s+=cpu[ib].s!=qb[ib].s;
    }
    std::printf("Q8_PACKED blocks=%zu mismatched_q_blocks=%zu mismatched_fp16_d=%zu mismatched_fp16_s=%zu\n",
                cpu.size(),bad_q,bad_d,bad_s);
    if(bad_q||bad_d||bad_s)
      throw std::runtime_error("GPU Q8 blocks not byte-identical to source-style CPU");
    double sse=0,maxerr=0,refss=0;
    std::vector<double> cpu_t0(N);
    // Compare ALL true llama outputs at T=1, and sampled rows for
    // synthetic additional tokens against the exact CPU Q8 projection.
    for(int t=0;t<T;++t)for(int j=0;j<(t==0?N:32);++j) {
      int n=t==0?j:(j*977+19)%N;
      double ref=cpu_projection(cpu,raw.data(),K,t,n);
      if(t==0)cpu_t0[n]=ref;
      double err=double(y[size_t(t)*N+n])-ref;
      if(!std::isfinite(err))throw std::runtime_error("nonfinite GPU Y");
      sse+=err*err;maxerr=std::max(maxerr,std::abs(err));refss+=ref*ref;
    }
    const size_t nchecked=size_t(N)+size_t(T-1)*32;
    std::printf("GPU_Q8_vs_CPU_FP64 samples=%zu max_abs=%.9g rmse=%.9g rel_rmse=%.9g\n",
      nchecked,maxerr,std::sqrt(sse/nchecked),
      std::sqrt(sse/std::max(refss,1.e-12)));
    if(maxerr>1.e-3 || std::sqrt(sse/nchecked)>1.e-4)
      throw std::runtime_error("GPU Q8 vs CPU oracle quality FAIL");

    sse=0;maxerr=0;
    for(int n=0;n<N;++n) {
      double diff=double(y[n])-expected_llama[n];
      sse+=diff*diff;maxerr=std::max(maxerr,std::abs(diff));
    }
    std::printf("GPU_Q8_vs_authentic_llama_QKV T0 samples=%d max_abs=%.9g rmse=%.9g\n",
      N,maxerr,std::sqrt(sse/N));
    // NOT a whole-model quality gate. Only fail absurd local divergence.
    if(maxerr>1.e-3 || std::sqrt(sse/N)>1.e-4)
      throw std::runtime_error("GPU Q8 vs authentic llama first QKV FAIL");

    if(dp4a_ab) {
      // Both kernels read the *exact same* prequantized Q8_1 blocks and
      // raw GGUF weights. This isolates GEMV implementation differences,
      // including output T,N stride, with no quantization drift.
      ck(q8::gemv_q8_1_nvfp4_dp4a_batched(dq,dw,dFast,T,N,K),
         "DP4A batched experiment");
      ck(cudaDeviceSynchronize(),"DP4A warmup and strict compare");
      std::vector<float> fast(y.size());
      ck(cudaMemcpy(fast.data(),dFast,fast.size()*sizeof(float),
                    cudaMemcpyDeviceToHost),"D2H DP4A");
      double errss=0,baseline_ss=0,new_llama_ss=0,cpu_ss=0;
      double max_pair=0,max_llama=0,max_cpu=0;
      size_t nonfinite=0;
      for(size_t i=0;i<fast.size();++i) {
        if(!std::isfinite(fast[i]) || !std::isfinite(y[i])) {
          ++nonfinite; continue;
        }
        const double diff=double(fast[i])-y[i];
        errss+=diff*diff;
        baseline_ss+=double(y[i])*y[i];
        max_pair=std::max(max_pair,std::abs(diff));
      }
      // The real independent llama comparison is valid only for token0.
      // Extra T64 rows are deterministic perturbations, not authentic
      // captured llama layer positions.
      for(int n=0;n<N;++n) {
        if(!std::isfinite(fast[n]))continue;
        const double ll_diff=double(fast[n])-expected_llama[n];
        const double cpu_diff=double(fast[n])-cpu_t0[n];
        new_llama_ss+=ll_diff*ll_diff;
        cpu_ss+=cpu_diff*cpu_diff;
        max_llama=std::max(max_llama,std::abs(ll_diff));
        max_cpu=std::max(max_cpu,std::abs(cpu_diff));
      }
      const double rmse_pair=std::sqrt(errss/fast.size());
      const double rmse_llama=std::sqrt(new_llama_ss/N);
      const double rmse_cpu=std::sqrt(cpu_ss/N);
      std::printf("DP4A_AB_all_outputs T=%d n=%zu nonfinite=%zu "
                  "max_vs_Q8_baseline=%.12g rmse_vs_Q8_baseline=%.12g "
                  "relative_rmse=%.12g\n",T,fast.size(),nonfinite,
                  max_pair,rmse_pair,
                  std::sqrt(errss/std::max(baseline_ss,1.e-20)));
      std::printf("DP4A_AUTHENTIC_llama_token0 n=%d maxabs=%.12g "
                  "rmse=%.12g CPU_fp64_maxabs=%.12g CPU_fp64_rmse=%.12g\n",
                  N,max_llama,rmse_llama,max_cpu,rmse_cpu);

      // Source correctness includes all 10240 output rows per token and
      // 10 consecutive copies from the same packed activation state.
      // Memcmp is intentional: determinism requires bitwise FP32 equality.
      const std::vector<float> first=fast;
      int changed_runs=0;
      for(int rep=0;rep<10;++rep) {
        ck(q8::gemv_q8_1_nvfp4_dp4a_batched(dq,dw,dFast,T,N,K),
           "DP4A ten-repeat enqueue");
        ck(cudaMemcpy(fast.data(),dFast,fast.size()*4,
                      cudaMemcpyDeviceToHost),"DP4A repeat D2H sync");
        if(std::memcmp(fast.data(),first.data(),fast.size()*4))
          ++changed_runs;
      }
      std::printf("DP4A_DETERMINISM samples_per_run=%zu repeats=10 "
                  "changed_runs=%d\n",first.size(),changed_runs);
      if(nonfinite || changed_runs ||
         // Strict A/B: this is *tighter* than the old diagnostic
         // max 0.001 and RMSE 0.0001; never relax those thresholds.
         max_pair>1.e-5 || rmse_pair>2.e-6 ||
         max_llama>7.62939453125e-6 || rmse_llama>2.e-6 ||
         max_cpu>1.e-5 || rmse_cpu>2.e-6)
        throw std::runtime_error(
          "DP4A STRICT A/B NO-GO: QKV accuracy/determinism gate");
      std::puts("DP4A_STRICT_LAYER0_A_B_GATE=PASS (not full-model logits)");
    }

    Events timer;
    auto doquant=[&]{ck(q8::quantize_f32_q8_1(dx,dq,T,K),"quant enqueue");};
    auto dogemv=[&]{ck(q8::gemv_q8_1_nvfp4(dq,dw,dy,T,N,K),"GEMV enqueue");};
    auto both=[&]{ck(q8::project_f32_q8_1_nvfp4(dx,dw,dq,dy,T,N,K),"pipeline");};
    const float quant_ms=timer.ms(doquant,loops);
    const float gemv_ms=timer.ms(dogemv,loops);
    const float combined_ms=timer.ms(both,loops);
    std::printf("GPU_Q8_PERF T=%d loops=%d quant_ms=%.6f gemv_ms=%.6f combined_ms=%.6f "
                "q8_bytes=%zu gpu_allocation_delta_bytes=%zu\n",
                T,loops,quant_ms,gemv_ms,combined_ms,q8::q8_1_scratch_bytes(T,K),
                before-after);
    if(dp4a_ab) {
      auto fast_only=[&] {
        ck(q8::gemv_q8_1_nvfp4_dp4a_batched(dq,dw,dFast,T,N,K),
           "DP4A GEMV timed");
      };
      auto fast_pipeline=[&] {
        ck(q8::project_f32_q8_1_nvfp4_dp4a_batched(
             dx,dw,dq,dFast,T,N,K),"DP4A pipeline timed");
      };
      const float fast_ms=timer.ms(fast_only,loops);
      const float fast_pipeline_ms=timer.ms(fast_pipeline,loops);
      std::printf("DP4A_AB_TIMING T=%d repeats=%d baseline_q8_gemv_ms=%.6f "
                  "batched_dp4a_gemv_ms=%.6f baseline_pipeline_ms=%.6f "
                  "batched_dp4a_pipeline_ms=%.6f "
                  "raw_GGUF_weight_bytes=%zu q8_scratch_bytes=%zu "
                  "extra_weight_copy_bytes=0\n",
                  T,loops,gemv_ms,fast_ms,combined_ms,fast_pipeline_ms,
                  raw.size(),q8::q8_1_scratch_bytes(T,K));
    }
    std::puts("PASS isolated projection only; NOT PP2 latency/full-model parity");
    return 0;
  } catch(const std::exception& e) {
    std::fprintf(stderr,"Q8_1_GPU_BENCH_NO_GO: %s\n",e.what());
    return 1;
  }
}
