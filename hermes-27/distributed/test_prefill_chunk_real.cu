// Manual real-GGUF numerical gate. REQUIRES EXCLUSIVE GPU GRANT.
// Offline compilation/link is safe. No production service modifications.
// Default n=129 runs an independent token-by-token oracle; 1024/8192
// need an explicit --oracle=1 (expensive) to claim numerical PASS.
#include "pp2_fwd.hpp"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr int VOC=248320, NL=64;
constexpr int HV=48, KH=128, VH=128, CC=10240, NKK=1024;
const int seed_tokens[]={
  6205,161421,303,220,17,1374,9911,1147,27053,6059,13
};

int argmax(const std::vector<float>& v){
  int best=-1;
  for(int i=0;i<(int)v.size();++i)
    if(best<0 || v[i]>v[best])best=i;
  return best;
}

bool compare_logits(const char* label,const std::vector<float>& expected,
                    const std::vector<float>& actual,std::string& err,
                    double threshold=1e-4){
  if(expected.size()!=VOC || actual.size()!=VOC){
    err=std::string(label)+" logits shape mismatch";
    return false;
  }
  double max_abs=0, sumsq=0;
  for(int i=0;i<VOC;++i){
    if(!std::isfinite(expected[i]) || !std::isfinite(actual[i])){
      err=std::string(label)+" nonfinite logit at "+std::to_string(i);
      return false;
    }
    double diff=std::abs(double(expected[i])-actual[i]);
    max_abs=std::max(max_abs,diff);
    sumsq+=diff*diff;
  }
  const int g=argmax(expected),v=argmax(actual);
  std::printf("CHECK %s max_abs=%.9g rmse=%.9g argmax %d/%d\n",
              label,max_abs,std::sqrt(sumsq/VOC),g,v);
  if(max_abs>threshold || g!=v){
    err=std::string(label)+" mismatch max_abs="+std::to_string(max_abs)+
        " argmax="+std::to_string(g)+"/"+std::to_string(v);
    return false;
  }
  return true;
}

bool compare_buffer(const float* reference,const float* seeded,
                    size_t n,int dev,const char* type,int li,
                    double tol,std::string& err){
  if(!reference || !seeded){err="null state pointer";return false;}
  cudaError_t e=cudaSetDevice(dev);
  std::vector<float> a(n),b(n);
  if(e==cudaSuccess)e=cudaMemcpy(a.data(),reference,n*sizeof(float),
                                   cudaMemcpyDeviceToHost);
  if(e==cudaSuccess)e=cudaMemcpy(b.data(),seeded,n*sizeof(float),
                                   cudaMemcpyDeviceToHost);
  if(e!=cudaSuccess){
    err=std::string("state copy CUDA error ")+cudaGetErrorString(e);
    return false;
  }
  double max_abs=0;
  for(size_t k=0;k<n;++k){
    if(!std::isfinite(a[k]) || !std::isfinite(b[k])){
      err=std::string(type)+" nonfinite cache layer="+std::to_string(li);
      return false;
    }
    max_abs=std::max(max_abs,std::abs(double(a[k])-b[k]));
  }
  std::printf("STATE %s layer=%d max_abs=%.9g\n",type,li,max_abs);
  if(max_abs>tol){
    err=std::string(type)+" cache mismatch layer="+std::to_string(li)+
        " max_abs="+std::to_string(max_abs);
    return false;
  }
  return true;
}

bool compare_states(const pp2::DecodeState& reference,
                    const pp2::DecodeState& seeded,int length,
                    std::string& err){
  for(int li=0;li<NL;++li){
    const int dev=li<=31?0:1;
    if(li%4==3){
      if(reference.klen[li]!=length || seeded.klen[li]!=length){
        err="KV length mismatch li="+std::to_string(li);return false;
      }
      if(!compare_buffer(reference.Kc[li],seeded.Kc[li],
                         (size_t)length*NKK,dev,"K",li,1e-4,err) ||
         !compare_buffer(reference.Vc[li],seeded.Vc[li],
                         (size_t)length*NKK,dev,"V",li,1e-4,err))
        return false;
    }else{
      if(!compare_buffer(reference.S[li],seeded.S[li],
                         (size_t)HV*KH*VH,dev,"GDN",li,1e-4,err) ||
         !compare_buffer(reference.CS[li],seeded.CS[li],
                         (size_t)CC*3,dev,"CS",li,1e-4,err))
        return false;
    }
  }
  return true;
}
}

int main(int argc,char** argv){
  int length=129,chunk_tokens=64,oracle=-1;
  const char* model=nullptr;
  for(int i=1;i<argc;++i){
    const std::string arg=argv[i];
    if(arg.rfind("--n=",0)==0)length=std::atoi(arg.c_str()+4);
    else if(arg.rfind("--chunk=",0)==0)chunk_tokens=std::atoi(arg.c_str()+8);
    else if(arg.rfind("--oracle=",0)==0)oracle=std::atoi(arg.c_str()+9);
    else if(arg.rfind("--",0)==0){std::fprintf(stderr,"unknown flag %s\n",arg.c_str());return 2;}
    else if(!model)model=argv[i];
    else{std::fprintf(stderr,"too many positional arguments\n");return 2;}
  }
  if(length!=129 && length!=1024 && length!=8192){
    std::fprintf(stderr,"T must be 129,1024,8192\n");return 2;
  }
  if(chunk_tokens<1 || chunk_tokens>64){std::fprintf(stderr,"invalid chunk size\n");return 2;}
  if(oracle==-1)oracle=length==129?1:0;
  if(oracle!=0 && oracle!=1){std::fprintf(stderr,"invalid oracle flag\n");return 2;}
  setenv("HERMES_PP2_CHUNKED_PREFILL","1",1);
  unsetenv("HERMES_PP2_PREFILL_BF16");
  // Read env before decode_init; async state is the downstream consumer.
  setenv("HERMES_PP2_ASYNC_STREAMS","1",1);

  pp2::Stats upload_stats{};
  std::string err;
  const bool up=model ? pp2::upload(upload_stats,err,std::string(model))
                      : pp2::upload(upload_stats,err);
  if(!up){std::fprintf(stderr,"upload failed: %s\n",err.c_str());return 2;}
  const int capacity=length<8192?length+1:length;
  std::vector<int> tokens((size_t)length);
  for(int t=0;t<length;++t)tokens[t]=seed_tokens[(t*7+3)%11];

  pp2::DecodeState baseline{};
  std::vector<float> golden_last;
  bool ok=true;
  if(oracle){
    if(!pp2::decode_init(baseline,err,capacity))ok=false;
    for(int t=0;ok && t<length;++t){
      golden_last=pp2::decode_step(tokens[t],baseline,t,err);
      if(golden_last.size()!=VOC)ok=false;
    }
  }
  pp2::DecodeState seeded{};
  if(ok && !pp2::decode_init(seeded,err,capacity))ok=false;
  pp2::Stats prefill_stats{};
  std::vector<float> last;
  if(ok){
    last=pp2::prefill_chunks_into_decode_state(
      tokens,seeded,prefill_stats,err,chunk_tokens);
    if(last.size()!=VOC)ok=false;
  }
  if(ok && oracle){
    ok=compare_logits("last token",golden_last,last,err);
    if(ok)ok=compare_states(baseline,seeded,length,err);
    if(ok && length<8192){
      const int next=argmax(golden_last);
      auto g=pp2::decode_step(next,baseline,length,err);
      auto s=pp2::decode_step(next,seeded,length,err);
      ok=compare_logits("continuation",g,s,err);
    }
  }
  std::printf("CHUNK_PP2 n=%d chunk=%d oracle=%d prefill_ms=%.3f xfer_ms=%.3f "
              "status=%s detail=%s\n",
              length,chunk_tokens,oracle,prefill_stats.prefill_ms,
              prefill_stats.xfer_ms,ok?(oracle?"PASS":"UNVERIFIED"):"FAIL",
              err.c_str());
  if(seeded.inited)pp2::decode_free(seeded);
  if(baseline.inited)pp2::decode_free(baseline);
  pp2::free_all();
  // Exit 3 for smoke-only T1024/T8192: without a separate numerical
  // token-by-token oracle this cannot be cited as semantic parity.
  return !ok?1:(oracle?0:3);
}
