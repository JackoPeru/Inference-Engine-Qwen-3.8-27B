// Explicit real-GGUF prefill/cache/next-token quality gate (manual GPU test).
// Compile/link separately until prime registers a gated target in CMake.
// No run is permitted without exclusive GPU authorization.
#include "pp2_fwd.hpp"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
constexpr int VOC=248320, NL=64;
constexpr int HV=48, KHD=128, VHD=128, CC=10240, NKK=1024;

bool compare_logits(const char* label,const std::vector<float>& a,
                    const std::vector<float>& b,int T,
                    double tolerance,std::string& err){
  if(a.size()!=size_t(T)*VOC || b.size()!=a.size()){
    err=std::string(label)+" wrong logit vector size";
    return false;
  }
  double largest=0;
  for(int t=0;t<T;++t){
    int ai=-1,bi=-1;
    float av=-INFINITY,bv=-INFINITY;
    for(int j=0;j<VOC;++j){
      size_t i=size_t(t)*VOC+j;
      float x=a[i],y=b[i];
      if(!std::isfinite(x)||!std::isfinite(y)){
        err=std::string(label)+" nonfinite logits t="+std::to_string(t);
        return false;
      }
      largest=std::max(largest,std::abs(double(x)-double(y)));
      if(x>av){ai=j;av=x;}
      if(y>bv){bi=j;bv=y;}
    }
    if(ai!=bi){
      err=std::string(label)+" argmax differs t="+std::to_string(t)+
          " "+std::to_string(ai)+" vs "+std::to_string(bi);
      return false;
    }
  }
  std::printf("%s logits max_abs=%.9g gate<=%.9g\n",label,largest,tolerance);
  if(largest>tolerance){
    err=std::string(label)+" full-logit maximum absolute error too high";
    return false;
  }
  return true;
}

bool device_compare(const float* a,const float* b,size_t count,int dev,
                    const char* name,int layer,double tol,std::string& err){
  if(!a||!b){err=std::string(name)+" cache null";return false;}
  cudaError_t rc=cudaSetDevice(dev);
  std::vector<float> lhs(count),rhs(count);
  if(rc==cudaSuccess)rc=cudaMemcpy(lhs.data(),a,count*sizeof(float),cudaMemcpyDeviceToHost);
  if(rc==cudaSuccess)rc=cudaMemcpy(rhs.data(),b,count*sizeof(float),cudaMemcpyDeviceToHost);
  if(rc!=cudaSuccess){
    err=std::string("cache D2H CUDA ")+cudaGetErrorString(rc);
    return false;
  }
  double maxerr=0;
  for(size_t i=0;i<count;++i){
    if(!std::isfinite(lhs[i])||!std::isfinite(rhs[i])){
      err=std::string(name)+" nonfinite cache layer="+std::to_string(layer);
      return false;
    }
    maxerr=std::max(maxerr,std::abs(double(lhs[i])-rhs[i]));
  }
  std::printf("cache %s layer=%d max_abs=%.9g\n",name,layer,maxerr);
  if(maxerr>tol){
    err=std::string(name)+" cache mismatch layer="+std::to_string(layer)+
        " max_abs="+std::to_string(maxerr);
    return false;
  }
  return true;
}

bool compare_states(const pp2::DecodeState& sequential,
                    const pp2::DecodeState& seeded,int T,std::string& err){
  for(int li=0;li<NL;++li){
    int dev=li<=31?0:1;
    if(li%4==3){
      if(sequential.klen[li]!=T||seeded.klen[li]!=T){
        err="K/V cache length mismatch li="+std::to_string(li);
        return false;
      }
      if(!device_compare(sequential.Kc[li],seeded.Kc[li],size_t(T)*NKK,
                         dev,"K",li,1e-3,err)||
         !device_compare(sequential.Vc[li],seeded.Vc[li],size_t(T)*NKK,
                         dev,"V",li,1e-3,err)) return false;
    }else{
      if(!device_compare(sequential.S[li],seeded.S[li],size_t(HV)*KHD*VHD,
                         dev,"GDN S",li,1e-3,err)||
         !device_compare(sequential.CS[li],seeded.CS[li],size_t(CC)*3,
                         dev,"Conv tail",li,1e-3,err)) return false;
    }
  }
  return true;
}

} // namespace

int main(int argc,char** argv){
  int T=11;
  if(argc>3){
    std::fprintf(stderr,"usage: test_prefill_integration_real [T=2|11|64] [model.gguf]\n");
    return 2;
  }
  if(argc>=2)T=std::atoi(argv[1]);
  if(T!=2&&T!=11&&T!=64){std::fprintf(stderr,"unsupported test T\n");return 2;}
  const bool bf16=[]{
    const char* v=std::getenv("HERMES_PP2_PREFILL_BF16");
    return v && std::string(v)=="1";
  }();
  unsetenv("HERMES_PP2_ASYNC_STREAMS");
  pp2::Stats st;
  std::string err;
  const bool loaded=(argc==3)?pp2::upload(st,err,argv[2]):pp2::upload(st,err);
  if(!loaded){std::fprintf(stderr,"upload FAIL %s\n",err.c_str());return 2;}

  std::vector<int> tokens(size_t(T),0);
  constexpr int seed_tokens[]={6205,161421,303,220,17,1374,9911,1147,27053,6059,13};
  for(int i=0;i<T;++i)tokens[i]=seed_tokens[i%11];
  // Keep original full-batch output as independent reference.
  pp2::Stats baseline_stats;
  auto baseline=pp2::prefill(tokens,baseline_stats,err);
  if(baseline.size()!=size_t(T)*VOC){
    std::fprintf(stderr,"raw-GGUF batched reference FAIL %s\n",err.c_str());
    pp2::free_all();return 2;
  }
  // Token-by-token reference is also used to verify recurrent and KV state.
  pp2::DecodeState sequential{};
  bool pass=pp2::decode_init(sequential,err,T+1);
  if(pass){
    for(int i=0;i<T;++i){
      auto v=pp2::decode_step(tokens[i],sequential,i,err);
      if(v.size()!=VOC){pass=false;break;}
    }
  }
  if(!pass){
    std::fprintf(stderr,"sequential reference FAIL %s\n",err.c_str());
    pp2::decode_free(sequential);pp2::free_all();return 2;
  }
  // Seed an ASYNC-enabled DecodeState: first token after batched prefill will
  // exercise existing per-request stream decode continuation on both GPUs.
  setenv("HERMES_PP2_ASYNC_STREAMS","1",1);
  pp2::DecodeState seeded{};
  pass=pp2::decode_init(seeded,err,T+1);
  if(!pass){std::fprintf(stderr,"seed decode_init FAIL %s\n",err.c_str());}
  if(pass && !seeded.async_ctx){pass=false;err="expected per-request async state";}
  std::vector<float> batched;
  if(pass){
    pp2::Stats seed_stats;
    batched=pp2::prefill_into_decode_state(tokens,seeded,seed_stats,err);
    if(batched.size()!=size_t(T)*VOC)pass=false;
  }
  if(pass)pass=compare_logits("full batched prefill",baseline,batched,T,
                             bf16?1e-3:1e-4,err);
  if(pass)pass=compare_states(sequential,seeded,T,err);
  if(pass){
    // Use identical chosen token at position T. This tests next-token
    // integration, rather than only agreeing on the prefill logits.
    int best=0;const float* last=baseline.data()+size_t(T-1)*VOC;
    for(int i=1;i<VOC;++i)if(last[i]>last[best])best=i;
    auto golden=pp2::decode_step(best,sequential,T,err);
    auto cont=pp2::decode_step(best,seeded,T,err);
    pass=compare_logits("post-prefill continuation",golden,cont,1,1e-3,err);
  }
  pp2::decode_free(seeded);
  pp2::decode_free(sequential);
  pp2::free_all();
  std::printf("PREFILL_INTEGRATION T=%d BF16=%d %s %s\n",
              T,int(bf16),pass?"PASS":"FAIL",err.c_str());
  return pass?0:1;
}
