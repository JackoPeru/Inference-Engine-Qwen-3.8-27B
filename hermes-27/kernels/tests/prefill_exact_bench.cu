// Standalone GGUF-real exact-BF16 prefill benchmark; NOT in production CMake.
// Tests all 52M weights of blk.0.attn_qkv against GGUF BF16 bit decoding,
// then compares BF16xBF16 cuBLAS GEMM with FP32 accum vs raw GGUF GEMV
// and independent FP64 CPU subset oracle. Supports T=16/64/128/256.
#include "hermes/nvfp4_prefill_exact.hpp"
#include "hermes/nvfp4_gemv.hpp"
#include "hermes/gguf.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pe=hermes::prefill_exact;

static void cu(cudaError_t e,const char* s) {
  if(e!=cudaSuccess)throw std::runtime_error(std::string(s)+": "+cudaGetErrorString(e));
}
static void cb(cublasStatus_t e,const char* s) {
  if(e!=CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(s)+" cublasStatus="+std::to_string(int(e)));
}
static uint16_t f2b(float f) {
  uint32_t bits;std::memcpy(&bits,&f,4);
  bits+=0x7fff+((bits>>16)&1);
  return uint16_t(bits>>16);
}
static float b2f(uint16_t b) {
  uint32_t bits=uint32_t(b)<<16;float f;std::memcpy(&f,&bits,4);return f;
}
static float scale(uint8_t u) {
  if(u==0||u==0x7f)return 0;
  unsigned e=u>>3,m=u&7;
  return e==0?std::ldexp(float(m),-9):std::ldexp(1.f+float(m)/8,int(e)-7);
}
static float e2(unsigned q) {
  static constexpr float lut[16]={0,.5f,1,1.5f,2,3,4,6,0,-.5f,-1,-1.5f,-2,-3,-4,-6};
  return lut[q&15];
}
static float weight(const uint8_t* w,int N,int K,int n,int k) {
  (void)N;
  const uint8_t* block=w+size_t(n)*(K/64*36)+size_t(k/64)*36;
  unsigned sub=unsigned(k%64)/16;
  unsigned j=unsigned(k%16);
  unsigned byte=block[4+sub*8+(j&7)];
  unsigned q=j<8?byte&15:byte>>4;
  return scale(block[sub])*e2(q);
}
struct Metric {
  double max_abs=0,sse=0,ref_ss=0;
  uint64_t n=0;
  void add(double y,double ref) {
    if(!std::isfinite(y)||!std::isfinite(ref))
      throw std::runtime_error("non-finite output: FAIL CLOSED");
    double d=y-ref;
    max_abs=std::max(max_abs,std::abs(d));
    sse+=d*d;ref_ss+=ref*ref;++n;
  }
  double rmse()const{return std::sqrt(sse/double(n));}
  double rel()const{return rmse()/std::max(1.e-12,std::sqrt(ref_ss/double(n)));}
  void report(const char* name)const {
    std::printf("%s samples=%llu max_abs=%.9g rmse=%.9g rel_rmse=%.9g\n",
      name,static_cast<unsigned long long>(n),max_abs,rmse(),rel());
  }
};

struct Gpu {
  std::vector<void*> ptrs;
  cublasHandle_t h=nullptr;
  Gpu(){cb(cublasCreate(&h),"cublasCreate");}
  Gpu(const Gpu&)=delete;
  Gpu&operator=(const Gpu&)=delete;
  template<class T> T* alloc(size_t n) {
    T* p=nullptr;cu(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)),"cudaMalloc");
    ptrs.push_back(p);return p;
  }
  ~Gpu(){for(void* p:ptrs)cudaFree(p);if(h)cublasDestroy(h);}
};
struct EventPair {
  cudaEvent_t start=nullptr,stop=nullptr;
  EventPair(){cu(cudaEventCreate(&start),"cudaEventCreate");cu(cudaEventCreate(&stop),"cudaEventCreate");}
  ~EventPair(){if(start)cudaEventDestroy(start);if(stop)cudaEventDestroy(stop);}
  template<class Fn> float measure(Fn f,int loops) {
    f();cu(cudaDeviceSynchronize(),"warmup synchronize");
    cu(cudaEventRecord(start),"event start");
    for(int i=0;i<loops;++i)f();
    cu(cudaEventRecord(stop),"event stop");
    cu(cudaEventSynchronize(stop),"event sync");
    float elapsed=0;cu(cudaEventElapsedTime(&elapsed,start,stop),"elapsed");
    return elapsed/loops;
  }
};

static std::vector<uint8_t> read_tensor(std::ifstream& f,uint64_t at,size_t bytes) {
  std::vector<uint8_t> v(bytes);
  f.clear();f.seekg(static_cast<std::streamoff>(at));
  if(!f || !f.read(reinterpret_cast<char*>(v.data()),static_cast<std::streamsize>(bytes)))
    throw std::runtime_error("GGUF tensor read failed");
  return v;
}

struct RealGguf {
  int N=0,K=0;
  std::vector<uint8_t> W;
  std::vector<uint16_t> emb;
  std::vector<float> norm;
  RealGguf(const std::string& path,int maxT) {
    hermes::GgufFile gf;gf.open(path,true);
    auto w=gf.find("blk.0.attn_qkv.weight");
    auto e=gf.find("token_embd.weight");
    auto n=gf.find("blk.0.attn_norm.weight");
    if(w.type!=hermes::GgmlType::NVFP4 || w.n_dims!=2 ||
       w.ne[0]%64 || w.ne[0]>65536 || w.ne[1]>65536 ||
       e.type!=hermes::GgmlType::BF16 || e.ne[0]!=w.ne[0] ||
       n.type!=hermes::GgmlType::F32 || n.ne[0]!=w.ne[0])
      throw std::runtime_error("GGUF tensor types/dim mismatch");
    N=int(w.ne[1]);K=int(w.ne[0]);
    std::ifstream f(path,std::ios::binary);
    if(!f)throw std::runtime_error("Cannot open model GGUF");
    W=read_tensor(f,gf.data_offset+w.offset,size_t(w.n_bytes));
    auto nr=read_tensor(f,gf.data_offset+n.offset,size_t(K)*4);
    norm.resize(K);std::memcpy(norm.data(),nr.data(),size_t(K)*4);
    emb.resize(size_t(maxT)*K);
    for(int t=0;t<maxT;++t) {
      int id=(6205+9719*t)%int(e.ne[1]);
      auto row=read_tensor(f,gf.data_offset+e.offset+size_t(id)*K*2,size_t(K)*2);
      std::memcpy(emb.data()+size_t(t)*K,row.data(),size_t(K)*2);
    }
    std::printf("REAL_GGUF target=blk.0.attn_qkv.weight N=%d K=%d bytes=%zu\n",
      N,K,W.size());
  }
  std::vector<uint16_t> x(int T)const {
    std::vector<uint16_t> X(size_t(T)*K);
    for(int t=0;t<T;++t) {
      double sum=0;
      for(int k=0;k<K;++k) {
        float v=b2f(emb[size_t(t)*K+k]);sum+=double(v)*v;
      }
      float inv=1.f/std::sqrt(float(sum/K)+1.e-6f);
      for(int k=0;k<K;++k) {
        float v=b2f(emb[size_t(t)*K+k])*inv*norm[k];
        if(!std::isfinite(v))throw std::runtime_error("nonfinite activation");
        X[size_t(t)*K+k]=f2b(v);
      }
    }
    return X;
  }
};

static void exhaustive_scale_proof() {
  unsigned tested=0;
  for(unsigned u=0;u<256;++u)for(unsigned q=0;q<16;++q) {
    const float v=scale(uint8_t(u))*e2(q);
    uint32_t bits;std::memcpy(&bits,&v,4);
    if((bits&0xffff)!=0 || b2f(f2b(v))!=v)
      throw std::runtime_error("finite UE4M3 x E2M1 is not BF16 exact");
    ++tested;
  }
  std::printf("SCALE_PROOF all_ue4m3_e2m1_pairs=%u EXACT_BF16 PASS\n",tested);
}

static void full_bf16_weight_proof(const RealGguf& m,
                                   const std::vector<uint16_t>& dW) {
  if(dW.size()!=size_t(m.N)*m.K)throw std::runtime_error("weight size mismatch");
  uint64_t bad=0;
  for(int n=0;n<m.N;++n)for(int k=0;k<m.K;++k) {
    uint16_t exp=f2b(weight(m.W.data(),m.N,m.K,n,k));
    bad+=(exp!=dW[size_t(n)*m.K+k]);
    if(bad>3)throw std::runtime_error("GGUF BF16 weight unpack mismatch (first 4 mismatches)");
  }
  if(bad)throw std::runtime_error("GGUF weight unpack non-bit-exact");
  std::printf("WEIGHT_UNPACK_FULL_CHECK N*K=%zu GGUF->BF16 bit_exact PASS\n",dW.size());
}

static void run(const RealGguf& m,int T,int repeats,bool full_weight_proof) {
  const int N=m.N,K=m.K;
  std::printf("\nT=%d N=%d K=%d starts\n",T,N,K);std::fflush(stdout);
  auto X=m.x(T);
  Gpu dev;
  size_t before=0,total=0,after=0;
  cu(cudaMemGetInfo(&before,&total),"mem initial");
  uint16_t* dX=dev.alloc<uint16_t>(X.size());
  uint8_t* dWraw=dev.alloc<uint8_t>(m.W.size());
  uint16_t* dWbf16=dev.alloc<uint16_t>(size_t(N)*K);
  float* dGemv=dev.alloc<float>(size_t(T)*N);
  float* dTc=dev.alloc<float>(size_t(T)*N);
  float* dBatch4=dev.alloc<float>(size_t(T)*N);
  float* dWarp4=dev.alloc<float>(size_t(T)*N);
  cu(cudaMemcpy(dX,X.data(),X.size()*2,cudaMemcpyHostToDevice),"copy X");
  cu(cudaMemcpy(dWraw,m.W.data(),m.W.size(),cudaMemcpyHostToDevice),"copy raw GGUF");
  cu(pe::dequantize_gguf_to_bf16(dWraw,dWbf16,N,K),"device weight decode");
  cu(cudaDeviceSynchronize(),"unpack sync");
  if(full_weight_proof) {
    std::vector<uint16_t> packed(size_t(N)*K);
    cu(cudaMemcpy(packed.data(),dWbf16,packed.size()*2,cudaMemcpyDeviceToHost),
       "copy decoded weights");
    full_bf16_weight_proof(m,packed);
  }
  cu(cudaMemGetInfo(&after,&total),"mem allocated");
  std::printf("VRAM before_free=%zu after_free=%zu delta_bytes=%zu "
              "BF16_scratch_bytes=%zu raw_gguf_bytes=%zu\n",
              before,after,before-after,pe::bf16_weight_scratch_bytes(N,K),m.W.size());

  auto scalar=[&] {
    for(int t=0;t<T;++t) hermes::nvfp4_gemv_cuda(
      dX+size_t(t)*K,dWraw,dGemv+size_t(t)*N,K,N);
    cu(cudaGetLastError(),"scalar CUDA");
  };
  auto direct=[&] {
    cu(pe::batch4_direct_gguf(dX,dWraw,dBatch4,T,N,K),"direct batch4");
  };
  auto warp4=[&] {
    cu(pe::repeated_warp4_gguf(dX,dWraw,dWarp4,T,N,K),"Warp4 baseline");
  };
  auto gemm=[&] {
    cb(pe::gemm_bf16_fp32(dev.h,dX,dWbf16,dTc,T,N,K),"cuBLAS W16 A16");
  };
  auto unpack=[&] {
    cu(pe::dequantize_gguf_to_bf16(dWraw,dWbf16,N,K),"unpack GGUF");
  };
  auto fused=[&] { unpack();gemm(); };
  scalar();direct();warp4();gemm();
  cu(cudaDeviceSynchronize(),"quality sync");
  std::vector<float> b(size_t(T)*N),d(b.size()),g(b.size());
  std::vector<float> w(b.size());
  cu(cudaMemcpy(b.data(),dGemv,b.size()*4,cudaMemcpyDeviceToHost),"read baseline");
  cu(cudaMemcpy(d.data(),dBatch4,d.size()*4,cudaMemcpyDeviceToHost),"read direct");
  cu(cudaMemcpy(g.data(),dTc,g.size()*4,cudaMemcpyDeviceToHost),"read TC");
  cu(cudaMemcpy(w.data(),dWarp4,w.size()*4,cudaMemcpyDeviceToHost),"read Warp4");
  Metric tc,dr,wr;
  for(size_t i=0;i<b.size();++i) {
    tc.add(g[i],b[i]);dr.add(d[i],b[i]);wr.add(w[i],b[i]);
  }
  tc.report("TC_BF16_vs_GGUF_GEMV_FULL");
  dr.report("DIRECT_BATCH4_vs_GGUF_GEMV_FULL");
  wr.report("WARP4_vs_GGUF_GEMV_FULL");
  Metric cpuTC,cpuDR,cpuBaseline,cpuWarp;
  constexpr int sample_rows=32;
  for(int t=0;t<T;++t)for(int i=0;i<sample_rows;++i) {
    int n=(i*977+19)%N;
    double truth=0;
    for(int k=0;k<K;++k)
      truth+=double(b2f(X[size_t(t)*K+k]))*
              double(weight(m.W.data(),N,K,n,k));
    const size_t idx=size_t(t)*N+n;
    cpuTC.add(g[idx],truth);
    cpuDR.add(d[idx],truth);
    cpuWarp.add(w[idx],truth);
    cpuBaseline.add(b[idx],truth);
  }
  cpuBaseline.report("CPU_FP64_vs_GEMV_sample");
  cpuTC.report("CPU_FP64_vs_TENSORCORE_sample");
  cpuDR.report("CPU_FP64_vs_DIRECT_batch4_sample");
  cpuWarp.report("CPU_FP64_vs_WARP4_sample");
  // Fail closed. Thresholds are for a single projection only, not model logits.
  auto gate=[](const Metric& e){return e.max_abs<=0.01 && e.rel()<=0.001;};
  if(!gate(tc)||!gate(dr)||!gate(wr)||!gate(cpuTC)||!gate(cpuDR)||
     !gate(cpuWarp)||!gate(cpuBaseline))
    throw std::runtime_error("FAIL numeric gate: strict projection threshold 0.01 max / .001 relative");

  EventPair timer;
  float scalar_ms=timer.measure(scalar,repeats);
  float warp4_ms=timer.measure(warp4,repeats);
  float direct_ms=timer.measure(direct,repeats);
  float unpack_ms=timer.measure(unpack,repeats);
  float gemm_ms=timer.measure(gemm,repeats);
  float pipeline_ms=timer.measure(fused,repeats);
  std::printf("CUDA_EVENTS repeats=%d scalar_repeated_ms=%.4f warp4_repeated_ms=%.4f "
              "direct_batch4_ms=%.4f device_unpack_ms=%.4f "
              "BF16_TC_GEMM_ms=%.4f unpack_plus_GEMM_ms=%.4f\n",
              repeats,scalar_ms,warp4_ms,direct_ms,unpack_ms,gemm_ms,pipeline_ms);
  std::printf("QUALITY_GATE=PASS per-matrix only (NOT whole-model/logits)\n");
}

int main(int argc,char**argv) {
  try {
    std::string path;
    int T=64,device=1,repeats=3;
    bool all=false,cpu_only=false;
    for(int i=1;i<argc;++i) {
      std::string op=argv[i];
      auto val=[&] {
        if(i+1==argc)throw std::runtime_error("missing argument for "+op);
        return std::string(argv[++i]);
      };
      if(op=="--gguf")path=val();
      else if(op=="--T")T=std::stoi(val());
      else if(op=="--device")device=std::stoi(val());
      else if(op=="--repeats")repeats=std::stoi(val());
      else if(op=="--all")all=true;
      else if(op=="--cpu-only")cpu_only=true;
      else if(op=="--help") {
        std::printf("usage: prefill_exact_bench [--cpu-only] [--gguf FILE] "
                    "[--T 16|64|128|256|--all] [--device 1] [--repeats 3]\n");
        return 0;
      } else throw std::runtime_error("unknown argument "+op);
    }
    exhaustive_scale_proof();
    if(cpu_only)return 0;
    if(path.empty() || (T!=16&&T!=64&&T!=128&&T!=256) ||
       repeats<1 || repeats>30) throw std::runtime_error("invalid --gguf/T/repeats");
    RealGguf m(path,all?256:T);
    cu(cudaSetDevice(device),"cudaSetDevice");
    cudaDeviceProp prop{};
    cu(cudaGetDeviceProperties(&prop,device),"GPU properties");
    std::printf("DEVICE=%d %s SM=%d.%d\n",device,prop.name,prop.major,prop.minor);
    if(prop.major<12) throw std::runtime_error("SM120 required");
    if(all) for(int t:{16,64,128,256})run(m,t,repeats,t==16);
    else run(m,T,repeats,true);
    return 0;
  } catch(const std::exception& e) {
    std::fprintf(stderr,"FAIL / INCONCLUSIVE: %s\n",e.what());
    return 1;
  }
}
