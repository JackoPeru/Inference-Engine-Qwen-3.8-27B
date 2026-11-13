// Optional CPU-only real-capture validation against worker-1 authentic
// llama.cpp layer0 attn_norm-0 -> linear_attn_qkv_mixed-0 FP32 exports.
// NO GPU, cudaMalloc, CUDA driver, model service or production source edits.
//
// Compile standalone:
// g++ -std=c++20 -O3 -I model/include q8_1_real_capture_cpu.cpp
//     build/libhermes_gguf.a -o /tmp/.../q8_1_real_capture_cpu
//
// The CPU FP64 projection is deliberately NOT a bitwise reproduction of
// llama.cpp DP4A/mmq integer accumulation. Compare numerical error only.
#include "hermes/gguf.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(sizeof(_Float16)==2);
struct Q8block {
  _Float16 d,s;
  int8_t q[32];
};
static_assert(sizeof(Q8block)==36);
static_assert(offsetof(Q8block,q)==4);

std::vector<uint8_t> read_at(std::ifstream& f,uint64_t offset,size_t len) {
  f.seekg(static_cast<std::streamoff>(offset));
  if(!f) throw std::runtime_error("seek failed");
  std::vector<uint8_t> out(len);
  if(!f.read(reinterpret_cast<char*>(out.data()),
             static_cast<std::streamsize>(len)))
    throw std::runtime_error("read failed");
  return out;
}
std::vector<float> fp32_file(const std::string& name,int count) {
  std::ifstream f(name,std::ios::binary | std::ios::ate);
  if(!f || f.tellg()!=std::streamoff(size_t(count)*4))
    throw std::runtime_error("capture missing/wrong size: "+name);
  f.seekg(0);
  std::vector<float> out(count);
  if(!f.read(reinterpret_cast<char*>(out.data()),count*4))
    throw std::runtime_error("capture read failed");
  for(float x:out)
    if(!std::isfinite(x))throw std::runtime_error("non-finite real capture");
  return out;
}
float ue4(uint8_t c) {
  if(c==0 || c==127)return 0.f;
  int exp=c>>3;
  int man=c&7;
  return exp==0 ? std::ldexp(float(man),-9)
    : std::ldexp(1.f+float(man)/8,exp-7);
}
int e2twice(uint8_t c) {
  static const int lut[16]={
    0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12
  };
  return lut[c&15];
}
struct Metric {
  double sse=0, ref_ss=0, max_abs=0;
  size_t n=0;
  void add(double a,double reference) {
    if(!std::isfinite(a)||!std::isfinite(reference))
      throw std::runtime_error("non-finite output");
    double d=a-reference;
    sse+=d*d;ref_ss+=reference*reference;
    max_abs=std::max(max_abs,std::abs(d));++n;
  }
  double rmse()const{return std::sqrt(sse/n);}
  double relrmse()const{return rmse()/std::max(1.e-12,std::sqrt(ref_ss/n));}
};

int main(int argc,char** argv) {
  try {
    if(argc!=4) {
      std::fprintf(stderr,"usage: %s MODEL.GGUF LLAMA_LAYER0_DIR EXPECTED_SHA256\n",argv[0]);
      return 2;
    }
    const std::string model=argv[1],dir=argv[2],expected_hash=argv[3];
    if(expected_hash!="db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1")
      throw std::runtime_error("wrong declared GGUF SHA256 for worker1 capture");
    // This checks caller-supplied SHA identifier, NOT an independent
    // rehashing of the 19.7GB GGUF file. Check separately if required.
    hermes::GgufFile gf;
    gf.open(model,true);
    auto t=gf.find("blk.0.attn_qkv.weight");
    if(t.type!=hermes::GgmlType::NVFP4 || t.ne[0]!=5120 ||
       t.ne[1]!=10240 || t.n_bytes!=29491200)
      throw std::runtime_error("GGUF tensor metadata mismatch");
    constexpr int K=5120,N=10240;
    auto x=fp32_file(dir+"/attn_norm-0.f32",K);
    auto ll=fp32_file(dir+"/linear_attn_qkv_mixed-0.f32",N);
    std::ifstream f(model,std::ios::binary);
    if(!f)throw std::runtime_error("cannot open physical GGUF");
    auto w=read_at(f,gf.data_offset+t.offset,static_cast<size_t>(t.n_bytes));

    std::vector<Q8block> q8(K/32);
    double input_abs_max=0.;
    for(int b=0;b<K/32;++b) {
      Q8block& block=q8[b];
      float amax=0.f;
      float sum=0.f;
      for(int j=0;j<32;++j){
        float v=x[b*32+j];
        amax=std::max(amax,std::abs(v));
        sum+=v; // s does not affect NVFP4 dot; association may differ.
      }
      float d=amax/127.f;
      block.d=static_cast<_Float16>(d); // FP16 round-to-nearest
      block.s=static_cast<_Float16>(sum);
      for(int j=0;j<32;++j) {
        float v=x[b*32+j];
        int qi=amax==0.f ? 0 : int(std::round(v/d));
        if(qi < -127 || qi > 127)
          throw std::runtime_error("quant out of int8 range");
        block.q[j]=static_cast<int8_t>(qi);
        input_abs_max=std::max(input_abs_max,double(std::abs(v)));
      }
    }
    Metric metric;
    for(int n=0;n<N;++n) {
      const uint8_t* wr=w.data()+size_t(n)*(K/64*36);
      double result=0.;
      for(int b=0;b<K/64;++b) {
        const uint8_t* p=wr+b*36;
        for(int s=0;s<4;++s) {
          const Q8block& xq=q8[b*2+s/2];
          int dot=0;
          for(int j=0;j<8;++j) {
            uint8_t v=p[4+s*8+j];
            int qidx=(s&1)*16+j;
            dot+=int(xq.q[qidx])*e2twice(v&15);
            dot+=int(xq.q[qidx+8])*e2twice(v>>4);
          }
          result+=double(dot)*(.5*double(ue4(p[s])))*double(xq.d);
        }
      }
      metric.add(float(result),ll[n]);
    }
    std::printf("GGUF_CAPTURE_LABEL_SHA256=%s (not rehashed)\n",expected_hash.c_str());
    std::printf("SOURCE=llama.cpp callback attn_norm-0 -> linear_attn_qkv_mixed-0 token6205 pos0\n");
    std::printf("Q8_1_REFERENCE n=%zu input_amax=%.9g max_abs=%.12g rmse=%.12g rel_rmse=%.12g\n",
      metric.n,input_abs_max,metric.max_abs,metric.rmse(),metric.relrmse());
    const double worker1_rmse=1.5764864708187002e-7;
    const double worker1_max=3.814697265625e-6;
    std::printf("WORKER1_PRIOR_CPU_Q8_APPROX max_abs=%.12g rmse=%.12g\n",
      worker1_max,worker1_rmse);
    // Conservative numerical gate that accounts for reduction ordering:
    // do not misrepresent it as proof of CUDA/runtime identity.
    if(metric.max_abs>1.e-4 || metric.rmse()>1.e-5) {
      std::fprintf(stderr,"NO_GO: CPU Q8_1 reference diverges from llama capture\n");
      return 1;
    }
    std::puts("PASS real GGUF Q8_1 CPU numerical projection only, NO GPU parity claim");
    return 0;
  } catch(const std::exception& e) {
    std::fprintf(stderr,"FAIL CLOSED: %s\n",e.what());
    return 1;
  }
}
