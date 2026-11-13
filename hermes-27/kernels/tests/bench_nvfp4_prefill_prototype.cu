// Independent PP2 NVFP4 prefill microbenchmark; NOT part of production CMake.
// Real GGUF: blk.0.attn_qkv.weight [10240,5120] multiplied by the BF16
// direct-RMSNorm(token_embd.weight[id], blk.0.attn_norm.weight) inputs.
// Measures T=16/64/256, but defaults to only T=64 for a short initial gate.
// See README_nvfp4_prefill_prototype.md for standalone build and safety.
#include "hermes/gguf.hpp"
#include "hermes/nvfp4_gemm.hpp"
#include "hermes/nvfp4_gemv.hpp"
#include "hermes/nvfp4_prefill_prototype.hpp"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using hermes::Nvfp4GemmW4A4;
namespace pr = hermes::prefill_prototype;
using Clock=std::chrono::steady_clock;

static void check(cudaError_t e, const char* where) {
  if(e!=cudaSuccess) throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));
}
static uint16_t float_to_bf16(float f) {
  uint32_t bits; std::memcpy(&bits,&f,4);
  const uint32_t lsb=(bits>>16)&1u;
  bits+=0x7fffu+lsb; // finite IEEE RNE, matching CUDA __float2bfloat16
  return uint16_t(bits>>16);
}
static float bf16_to_float(uint16_t x) {
  uint32_t bits=uint32_t(x)<<16; float f; std::memcpy(&f,&bits,4); return f;
}
static double ms(Clock::time_point a,Clock::time_point b) {
  return std::chrono::duration<double,std::milli>(b-a).count();
}
static std::vector<uint8_t> load_tensor(std::ifstream& f, uint64_t pos, size_t bytes) {
  std::vector<uint8_t> buf(bytes);
  f.clear(); f.seekg(static_cast<std::streamoff>(pos),std::ios::beg);
  if(!f || !f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(bytes)))
    throw std::runtime_error("short GGUF tensor read");
  return buf;
}
static float raw_ue(uint8_t x) {
  if(x==0 || x==0x7f) return 0.f;
  int e=(x>>3)&15,m=x&7;
  return e==0?std::ldexp(float(m),-9):std::ldexp(1.f+float(m)/8.f,e-7);
}
static float e2(int x) {
  static const float lut[16]={0,.5f,1,1.5f,2,3,4,6,0,-.5f,-1,-1.5f,-2,-3,-4,-6};
  return lut[x&15];
}
static float raw_weight(const uint8_t* all, int row, int col, int K) {
  const uint8_t* b=all+size_t(row)*(K/64*36)+size_t(col/64)*36;
  int sub=(col%64)/16, j=col%16;
  uint8_t bits=b[4+sub*8+(j&7)];
  int q=j<8?(bits&15):(bits>>4);
  return raw_ue(b[sub])*e2(q);
}

struct DeviceAlloc {
  std::vector<void*> all;
  template<class T> T* allocate(size_t count,const char* label) {
    T* p=nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(T)),label);
    all.push_back(p); return p;
  }
  ~DeviceAlloc(){ for(auto p:all) if(p) cudaFree(p); }
  DeviceAlloc()=default;
  DeviceAlloc(const DeviceAlloc&)=delete;
  DeviceAlloc& operator=(const DeviceAlloc&)=delete;
};
struct Error {
  double maxabs=0, meanabs=0, rmse=0, reference_rms=0;
  size_t n=0, first_bad=static_cast<size_t>(-1);
  void add(float actual, float expected, size_t id, bool enforce=false) {
    const double e=std::fabs(double(actual)-double(expected));
    maxabs=std::max(maxabs,e);
    meanabs+=e; rmse+=e*e; reference_rms+=double(expected)*double(expected);
    if(enforce && e>0.05+0.02*std::fabs(double(expected)) &&
       first_bad==static_cast<size_t>(-1)) first_bad=id;
    ++n;
  }
  void print(const char* label) const {
    const double n0=n?double(n):1.;
    const double ref=std::sqrt(reference_rms/n0);
    std::printf("%s samples=%zu maxabs=%.6g meanabs=%.6g rmse=%.6g "
                "relative_rmse=%.6g first_bad=%lld\n",
                label,n,maxabs,meanabs/n0,std::sqrt(rmse/n0),
                std::sqrt(rmse/n0)/std::max(ref,1e-12),
                first_bad==static_cast<size_t>(-1)?-1LL:(long long)first_bad);
  }
  bool ok() const {return first_bad==static_cast<size_t>(-1);}
};

template<class Fn> static float cuda_event_ms(Fn fn,int iterations) {
  cudaEvent_t start=nullptr,stop=nullptr;
  check(cudaEventCreate(&start),"cudaEventCreate start");
  check(cudaEventCreate(&stop),"cudaEventCreate stop");
  try {
    check(fn(),"warmup");
    check(cudaDeviceSynchronize(),"sync warmup");
    check(cudaEventRecord(start),"event start");
    for(int r=0;r<iterations;++r) check(fn(),"measured kernel");
    check(cudaEventRecord(stop),"event stop");
    check(cudaEventSynchronize(stop),"event sync");
    float elapsed=0;
    check(cudaEventElapsedTime(&elapsed,start,stop),"elapsed time");
    cudaEventDestroy(start);cudaEventDestroy(stop);
    return elapsed/iterations;
  } catch(...) {
    cudaEventDestroy(start);cudaEventDestroy(stop);throw;
  }
}

struct Model {
  std::ifstream stream;
  hermes::GgufFile gf;
  std::vector<uint8_t> weight_raw;
  std::vector<uint16_t> x_embedding;
  std::vector<float> norm;
  int N=0,K=0;

  Model(const std::string& gguf, int maxT, const std::string& tensor) {
    gf.open(gguf,true);
    const auto& w=gf.find(tensor);
    if(w.type!=hermes::GgmlType::NVFP4 || w.n_dims!=2 || w.ne[0]%64!=0 ||
       w.ne[0]>65536 || w.ne[1]>100000) {
      throw std::runtime_error("selected weight is not a supported NVFP4 [N,K] matrix");
    }
    K=int(w.ne[0]);N=int(w.ne[1]);
    // First linear in blk.0 receives the direct norm of the embedding.
    // This benchmark deliberately supports that exact stage only.
    if(tensor!="blk.0.attn_qkv.weight")
      throw std::runtime_error("only blk.0.attn_qkv.weight has validated embedding->norm input");
    const auto& emb=gf.find("token_embd.weight");
    const auto& nn=gf.find("blk.0.attn_norm.weight");
    if(emb.type!=hermes::GgmlType::BF16 || emb.n_dims!=2 ||
       int(emb.ne[0])!=K || nn.type!=hermes::GgmlType::F32 ||
       nn.ne[0]!=uint64_t(K))
      throw std::runtime_error("embedding or attn_norm GGUF metadata mismatch");
    stream.open(gguf,std::ios::binary);
    if(!stream) throw std::runtime_error("could not open GGUF stream");
    weight_raw=load_tensor(stream,gf.data_offset+w.offset,size_t(w.n_bytes));
    auto nn_raw=load_tensor(stream,gf.data_offset+nn.offset,size_t(K)*4);
    norm.resize(K); std::memcpy(norm.data(),nn_raw.data(),size_t(K)*4);
    x_embedding.resize(size_t(maxT)*K);
    for(int t=0;t<maxT;++t) {
      // Deterministic and distinct IDs, within the real GGUF vocabulary.
      // No tokenizer, no synthetic weights, no position-dependent state.
      const int id=(6205+t*9719)%int(emb.ne[1]);
      auto row=load_tensor(stream,gf.data_offset+emb.offset+
                          size_t(id)*K*sizeof(uint16_t),size_t(K)*2);
      std::memcpy(x_embedding.data()+size_t(t)*K,row.data(),size_t(K)*2);
    }
    std::printf("GGUF version=%u tensors=%zu weight=%s GGUF_bytes=%zu N=%d K=%d "
                "embed=BF16 norm=F32 IDs=(6205+t*9719)%%vocab\n",
                gf.header.version,gf.tensors.size(),tensor.c_str(),
                weight_raw.size(),N,K);
  }
  std::vector<uint16_t> make_input(int T) const {
    std::vector<uint16_t> X(size_t(T)*K);
    for(int t=0;t<T;++t) {
      const uint16_t* e=x_embedding.data()+size_t(t)*K;
      double ss=0;
      for(int k=0;k<K;++k) {
        float v=bf16_to_float(e[k]);ss+=double(v)*v;
      }
      const float inv=1.f/std::sqrt(float(ss/K)+1e-6f);
      for(int k=0;k<K;++k) {
        float v=bf16_to_float(e[k])*inv*norm[k];
        if(!std::isfinite(v)) throw std::runtime_error("nonfinite BF16 input");
        X[size_t(t)*K+k]=float_to_bf16(v);
      }
    }
    return X;
  }
};

static void run_T(const Model& m,int T,int repeats) {
  const int K=m.K,N=m.N;
  if(T!=16 && T!=64 && T!=256)
    throw std::runtime_error("only T=16,64,256 are benchmarked");
  std::printf("\n=== T=%d N=%d K=%d baseline=PP2_scalar_GEMV ===\n",T,N,K);
  std::vector<uint16_t> X=m.make_input(T);
  std::vector<uint8_t> Aq(size_t(T)*K/2,0);
  std::vector<uint8_t> SFA(Nvfp4GemmW4A4::sfa_bytes(T,K),0);
  const size_t sfb8=Nvfp4GemmW4A4::sfb_bytes(8,N,K);
  const size_t sfbT=Nvfp4GemmW4A4::sfb_bytes(T,N,K);
  if(!sfb8 || !sfbT || sfb8!=sfbT)
    throw std::runtime_error("SFB layout depends on T; builder hardcodes M=8. Do not use existing repack.");
  std::vector<uint8_t> Bq(size_t(N)*K/2,0),SFB(sfbT,0);
  std::vector<uint32_t> map;
  size_t sfa_map_bytes=0;
  std::string err;
  if(!pr::build_sfa_scatter_map(T,K,map,sfa_map_bytes,err) ||
     sfa_map_bytes!=SFA.size())
    throw std::runtime_error("SFA map failed: "+err);
  auto t0=Clock::now();
  Nvfp4GemmW4A4::build_B_from_gguf(m.weight_raw.data(),Bq.data(),SFB.data(),N,K);
  auto t1=Clock::now();
  Nvfp4GemmW4A4::quant_activation(X.data(),Aq.data(),SFA.data(),T,K);
  auto t2=Clock::now();
  std::printf("REPACK once CPU_ms=%.3f GGUF_nibble_bytes=%zu CUTLASS_Bq=%zu "
              "SFB=%zu | ACT host_quant_ms=%.3f Aq=%zu SFA=%zu map=%zu\n",
              ms(t0,t1),m.weight_raw.size(),Bq.size(),SFB.size(),ms(t1,t2),
              Aq.size(),SFA.size(),map.size()*sizeof(uint32_t));

  size_t free_start=0,total=0,free_peak=0;
  check(cudaMemGetInfo(&free_start,&total),"mem start");
  DeviceAlloc d;
  uint16_t* dX=d.allocate<uint16_t>(X.size(),"alloc dX");
  uint8_t* dRaw=d.allocate<uint8_t>(m.weight_raw.size(),"alloc raw weights");
  uint8_t* dB=d.allocate<uint8_t>(Bq.size(),"alloc repacked weights");
  uint8_t* dSFB=d.allocate<uint8_t>(SFB.size(),"alloc weight scales");
  uint8_t* dA=d.allocate<uint8_t>(Aq.size(),"alloc activation q4");
  uint8_t* dSFA=d.allocate<uint8_t>(SFA.size(),"alloc activation scales");
  uint32_t* dMap=d.allocate<uint32_t>(map.size(),"alloc scale layout map");
  uint16_t* dD=d.allocate<uint16_t>(size_t(T)*N,"alloc W4A4 output BF16");
  float* dY=d.allocate<float>(size_t(T)*N,"alloc repeated GEMV output FP32");
  const size_t wsbytes=Nvfp4GemmW4A4::workspace_bytes(T,N,K);
  uint8_t* dWs=wsbytes ? d.allocate<uint8_t>(wsbytes,"alloc CUTLASS workspace") : nullptr;
  check(cudaMemcpy(dX,X.data(),X.size()*2,cudaMemcpyHostToDevice),"copy X");
  check(cudaMemcpy(dRaw,m.weight_raw.data(),m.weight_raw.size(),cudaMemcpyHostToDevice),"copy raw");
  check(cudaMemcpy(dB,Bq.data(),Bq.size(),cudaMemcpyHostToDevice),"copy repacked");
  check(cudaMemcpy(dSFB,SFB.data(),SFB.size(),cudaMemcpyHostToDevice),"copy SFB");
  check(cudaMemcpy(dMap,map.data(),map.size()*4,cudaMemcpyHostToDevice),"copy map");
  check(cudaMemset(dSFA,0,SFA.size()),"zero SFA padding");
  check(cudaMemGetInfo(&free_peak,&total),"mem peak");
  std::printf("VRAM total=%zu free_before=%zu free_allocated=%zu peak_delta=%zu "
              "workspace=%zu B_repack_overhead=%lld bytes\n",
              total,free_start,free_peak,free_start-free_peak,wsbytes,
              (long long)Bq.size()+(long long)SFB.size()-
              (long long)m.weight_raw.size());

  auto gemm=[&]()->cudaError_t {
    int rc=Nvfp4GemmW4A4::run_ws(dA,dSFA,dB,dSFB,nullptr,dD,T,N,K,
                                 dWs,wsbytes,0);
    return rc==0 ? cudaGetLastError() : cudaErrorInvalidConfiguration;
  };
  check(cudaMemcpy(dA,Aq.data(),Aq.size(),cudaMemcpyHostToDevice),"host Aq H2D");
  check(cudaMemcpy(dSFA,SFA.data(),SFA.size(),cudaMemcpyHostToDevice),"host SFA H2D");
  check(gemm(),"CUTLASS host quant run");
  check(cudaDeviceSynchronize(),"sync host quant run");
  std::vector<uint16_t> Dhost(size_t(T)*N);
  check(cudaMemcpy(Dhost.data(),dD,Dhost.size()*2,cudaMemcpyDeviceToHost),"copy Dhost");

  // Device quantization must match host SFA/Aq BYTE FOR BYTE before any
  // performance or fidelity statement is accepted.
  check(cudaMemset(dA,0,Aq.size()),"zero Aq before device quant");
  check(cudaMemset(dSFA,0,SFA.size()),"zero SFA before device quant");
  check(pr::quantize_bf16_device(dX,dMap,dA,dSFA,T,K),"quant_device");
  check(cudaDeviceSynchronize(),"sync device quant");
  std::vector<uint8_t> Adevice(Aq.size()),Sdevice(SFA.size());
  check(cudaMemcpy(Adevice.data(),dA,Adevice.size(),cudaMemcpyDeviceToHost),"copy Aq device");
  check(cudaMemcpy(Sdevice.data(),dSFA,Sdevice.size(),cudaMemcpyDeviceToHost),"copy SFA device");
  size_t mismatchesA=0,mismatchesS=0;
  for(size_t i=0;i<Aq.size();++i) mismatchesA+=(Aq[i]!=Adevice[i]);
  for(size_t i=0;i<SFA.size();++i) mismatchesS+=(SFA[i]!=Sdevice[i]);
  std::printf("QUANT_BYTE_PARITY Aq_mismatch=%zu/%zu SFA_mismatch=%zu/%zu\n",
              mismatchesA,Aq.size(),mismatchesS,SFA.size());
  if(mismatchesA || mismatchesS)
    throw std::runtime_error("DEVICE QUANT not bit-identical to engine host quant");

  check(gemm(),"CUTLASS device quant run");
  check(cudaDeviceSynchronize(),"sync device quant run");
  std::vector<uint16_t> Ddev(size_t(T)*N);
  check(cudaMemcpy(Ddev.data(),dD,Ddev.size()*2,cudaMemcpyDeviceToHost),"copy Ddev");
  if(Dhost!=Ddev) throw std::runtime_error("HOST and DEVICE quant differ in CUTLASS output bytes");

  check(pr::repeated_gguf_gemv(dX,dRaw,dY,T,N,K),"baseline GEMV");
  check(cudaDeviceSynchronize(),"sync baseline GEMV");
  std::vector<float> Y(size_t(T)*N);
  check(cudaMemcpy(Y.data(),dY,Y.size()*4,cudaMemcpyDeviceToHost),"copy GEMV output");
  Error all_err;
  for(size_t i=0;i<Y.size();++i) {
    if(!std::isfinite(Y[i]) || !std::isfinite(bf16_to_float(Ddev[i])))
      throw std::runtime_error("GPU output NaN/Inf");
    all_err.add(bf16_to_float(Ddev[i]),Y[i],i);
  }
  all_err.print("W4A4_BF16_vs_PP2_GEMV_FP32_all");

  // CPU independent reference uses raw GGUF scales/nibbles, not CUTLASS
  // Bq/SFB. Full K dot products are evaluated on 32 output rows across T.
  // For quantized-input oracle, interpret the HOST Aq linear nibbles and
  // independently reconstruct each quant group scale from the BF16 X.
  std::vector<int> sample_rows;
  for(int j=0;j<32;++j) sample_rows.push_back((j*977+19)%N);
  Error raw_error, quant_error;
  for(int t=0;t<T;++t) {
    std::vector<float> xraw(K), xq(K);
    for(int b=0;b<K/16;++b) {
      float amax=0;
      for(int j=0;j<16;++j)
        amax=std::max(amax,std::fabs(bf16_to_float(X[size_t(t)*K+b*16+j])));
      float s=raw_ue(hermes::fp32_to_ue4m3(amax/6.f));
      for(int j=0;j<16;++j) {
        int k=b*16+j;
        xraw[k]=bf16_to_float(X[size_t(t)*K+k]);
        size_t nib=size_t(t)*K+k;
        int q=(Aq[nib/2]>>(4*(nib&1)))&15;
        xq[k]=s*e2(q);
      }
    }
    for(int n:sample_rows) {
      double vraw=0,vquant=0;
      for(int k=0;k<K;++k) {
        float w=raw_weight(m.weight_raw.data(),n,k,K);
        vraw+=double(w)*xraw[k]; vquant+=double(w)*xq[k];
      }
      size_t idx=size_t(t)*N+n;
      raw_error.add(Y[idx],float(vraw),idx,true);
      quant_error.add(bf16_to_float(Ddev[idx]),float(vquant),idx,true);
    }
  }
  raw_error.print("PP2_GEMV_vs_CPU_rawGGUF_sample");
  quant_error.print("CUTLASS_W4A4_vs_CPU_dequantA4_W4_sample");
  if(!raw_error.ok() || !quant_error.ok())
    throw std::runtime_error("CPU independent semantic/layout oracle failed");

  const float gemv_ms=cuda_event_ms([&]{
    return pr::repeated_gguf_gemv(dX,dRaw,dY,T,N,K);},repeats);
  const float kernel_ms=cuda_event_ms([&]{return gemm();},repeats);
  const float device_quant_ms=cuda_event_ms([&]{
    return pr::quantize_bf16_device(dX,dMap,dA,dSFA,T,K);},repeats);
  const float gpu_pipeline_ms=cuda_event_ms([&]{
    check(pr::quantize_bf16_device(dX,dMap,dA,dSFA,T,K),"quant pipeline");
    return gemm();},repeats);

  auto ch0=Clock::now();
  std::fill(Aq.begin(),Aq.end(),0);
  std::fill(SFA.begin(),SFA.end(),0);
  Nvfp4GemmW4A4::quant_activation(X.data(),Aq.data(),SFA.data(),T,K);
  auto ch1=Clock::now();
  check(cudaMemcpy(dA,Aq.data(),Aq.size(),cudaMemcpyHostToDevice),"timed host H2D Aq");
  check(cudaMemcpy(dSFA,SFA.data(),SFA.size(),cudaMemcpyHostToDevice),"timed host H2D SFA");
  auto ch2=Clock::now();
  std::printf("TIMING CUDA_EVENTS repeats=%d baseline_repeated_GEMV_ms=%.4f "
              "W4A4_kernel_ms=%.4f GPU_quant_ms=%.4f GPU_quant+W4A4_ms=%.4f\n",
              repeats,gemv_ms,kernel_ms,device_quant_ms,gpu_pipeline_ms);
  std::printf("TIMING wall_host_quant_ms=%.4f wall_host_H2D_ms=%.4f "
              "host_quant+H2D+W4A4_approx_ms=%.4f "
              "kernel_only_speedup=%.3f gpu_pipeline_speedup=%.3f\n",
              ms(ch0,ch1),ms(ch1,ch2),ms(ch0,ch2)+kernel_ms,
              gemv_ms/kernel_ms,gemv_ms/gpu_pipeline_ms);
  std::printf("PASS T=%d QUANT_BYTE_PARITY/CPU_SEMANTICS; "
              "no full-model logit parity claim\n",T);
}

// Pure-CPU validation: CUTLASS swizzle, scale-map bijection and SFB
// allocation size. It calls no CUDA device API and consumes no GPU time.
static void layout_self_test() {
  constexpr int N=10240,K=5120;
  for(int T:{16,64,256}) {
    std::vector<uint32_t> map;
    std::string err; size_t bytes=0;
    if(!pr::build_sfa_scatter_map(T,K,map,bytes,err))
      throw std::runtime_error("CPU SFA layout map T="+std::to_string(T)+": "+err);
    const size_t count=size_t(T)*K/16;
    std::vector<uint8_t> flat(count), swizzled(bytes,0);
    for(size_t i=0;i<count;++i) flat[i]=uint8_t((i*71+13)&255);
    Nvfp4GemmW4A4::repack_sfa_cpu(flat.data(),swizzled.data(),T,K);
    for(size_t i=0;i<count;++i) {
      if(map[i]>=bytes || swizzled[map[i]]!=flat[i])
        throw std::runtime_error("CPU SFA mapping failed at T="+std::to_string(T)+
                                 " logical index="+std::to_string(i));
    }
    auto sfb8=Nvfp4GemmW4A4::sfb_bytes(8,N,K);
    auto sfbT=Nvfp4GemmW4A4::sfb_bytes(T,N,K);
    std::printf("CPU_LAYOUT T=%d K=%d N=%d scales=%zu SFA_bytes=%zu "
                "SFB_bytes_M8=%zu SFB_bytes_T=%zu "
                "scale_mapping=%s SFB_builder_layout=%s\n",
                T,K,N,count,bytes,sfb8,sfbT,
                "PASS",sfb8==sfbT&&sfb8>0?"SIZE_COMPATIBLE":"UNSUPPORTED");
    if(!sfb8 || sfb8!=sfbT)
      throw std::runtime_error("SFB size differs by T; independent repack required");
  }
}

int main(int argc,char**argv) {
  try {
    std::string path;
    int T=64,dev=1,repeats=2;
    bool all=false,layout_only=false;
    for(int i=1;i<argc;++i) {
      std::string arg=argv[i];
      auto next=[&]()->std::string{
        if(i+1>=argc)throw std::runtime_error("missing argument to "+arg);
        return argv[++i];
      };
      if(arg=="--gguf") path=next();
      else if(arg=="--T") T=std::stoi(next());
      else if(arg=="--device") dev=std::stoi(next());
      else if(arg=="--repeats") repeats=std::stoi(next());
      else if(arg=="--all") all=true;
      else if(arg=="--layout-only") layout_only=true;
      else if(arg=="--help") {
        std::printf("Usage: bench_nvfp4_prefill_prototype --gguf FILE.gguf "
                    "[--T 16|64|256 | --all] [--device 1] [--repeats 2]\n"
                    "       bench_nvfp4_prefill_prototype --layout-only "
                    "(CPU only; no GGUF/GPU)\n");
        return 0;
      } else throw std::runtime_error("unknown option "+arg);
    }
    if(layout_only) { layout_self_test(); return 0; }
    if(path.empty() || repeats<1 || repeats>12 || (T!=16&&T!=64&&T!=256))
      throw std::runtime_error("--gguf required, repeats=1..12, T=16|64|256");
    Model m(path,all?256:T,"blk.0.attn_qkv.weight");
    check(cudaSetDevice(dev),"cudaSetDevice");
    cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop,dev),"get properties");
    std::printf("DEVICE=%d name=%s sm=%d.%d free-GPU-only-benchmark; "
                "no CPU full-model logit validation\n",
                dev,prop.name,prop.major,prop.minor);
    if(prop.major<12) throw std::runtime_error("requires Blackwell SM120 CUTLASS");
    if(all)for(int n:{16,64,256})run_T(m,n,repeats);
    else run_T(m,T,repeats);
    return 0;
  } catch(const std::exception& exc) {
    std::fprintf(stderr,"FAIL/INCONCLUSIVE: %s\n",exc.what());
    return 1;
  }
}
