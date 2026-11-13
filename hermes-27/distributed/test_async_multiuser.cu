// Real-checkpoint fail-closed PP2 per-request CUDA stream regression.
// This binary intentionally loads the GGUF and runs 2/4 concurrent requests:
// only launch with exclusive ownership of both GPUs.
// Env flags HERMES_NVFP4_GEMV / HERMES_HEAD_BF16 are preserved for A/B.
#include "pp2_fwd.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr int kVocab = 248320;
constexpr int kPrompt = 11;
constexpr int kGreedy = 4;
constexpr int kSteps = kPrompt + kGreedy;
using Clock = std::chrono::steady_clock;

int argmax(const std::vector<float>& logits){
  if(logits.empty()) return -1;
  int best=0;
  for(int i=1;i<(int)logits.size();++i) if(logits[i]>logits[best]) best=i;
  return best;
}

struct Reference {
  std::vector<int> inputs;
  std::vector<std::vector<float>> logits;
  std::vector<int> outputs;
};

const std::vector<std::vector<int>>& prompts(){
  static const std::vector<std::vector<int>> v={
    {6205,161421,303,220,17,1374,9911,1147,27053,6059,13},
    {13,6059,27053,1147,9911,1374,17,220,303,161421,6205},
    {220,17,1374,9911,1147,27053,6059,13,6205,161421,303},
    {9911,1147,27053,6059,13,6205,161421,303,220,17,1374}
  };
  return v;
}

bool build_reference(int id, Reference& ref, std::string& err){
  pp2::DecodeState ds{};
  if(!pp2::decode_init(ds,err,kSteps+1)) return false;
  bool pass=true;
  int next=-1;
  for(int t=0;t<kSteps;++t){
    const int input=(t<kPrompt)?prompts()[id][t]:next;
    if(input<0||input>=kVocab){ err="bad reference token"; pass=false; break; }
    auto logits=pp2::decode_step(input,ds,t,err);
    if(logits.size()!=kVocab){ pass=false; break; }
    next=argmax(logits);
    ref.inputs.push_back(input);
    ref.outputs.push_back(next);
    ref.logits.push_back(std::move(logits));
  }
  pp2::decode_free(ds);
  return pass;
}

bool check_request(int id, const Reference& ref, pp2::DecodeState& ds,
                   std::string& err, double& maxerr){
  maxerr=0;
  for(int t=0;t<kSteps;++t){
    auto logits=pp2::decode_step(ref.inputs[t],ds,t,err);
    if(logits.size()!=kVocab){ err="async logits size != vocab"; return false; }
    const int actual=argmax(logits);
    if(actual!=ref.outputs[t]){
      err="async argmax mismatch req="+std::to_string(id)+" pos="+std::to_string(t)+
          " actual="+std::to_string(actual)+" ref="+std::to_string(ref.outputs[t]);
      return false;
    }
    for(int j=0;j<kVocab;++j){
      if(!std::isfinite(logits[j])||!std::isfinite(ref.logits[t][j])){
        err="nonfinite logit req="+std::to_string(id)+" pos="+std::to_string(t);
        return false;
      }
      const double diff=std::abs(double(logits[j])-ref.logits[t][j]);
      maxerr=std::max(maxerr,diff);
      if(diff>1.e-4){
        err="logits mismatch req="+std::to_string(id)+" pos="+std::to_string(t)+
          " index="+std::to_string(j)+" abs="+std::to_string(diff);
        return false;
      }
    }
  }
  return true;
}

bool run_parallel(int N, const std::vector<Reference>& refs){
  std::vector<pp2::DecodeState> states((size_t)N);
  std::string err;
  int initialized=0;
  for(int i=0;i<N;++i){
    if(!pp2::decode_init(states[i],err,kSteps+1)){
      std::fprintf(stderr,"async init N=%d req=%d failed: %s\n",N,i,err.c_str());
      break;
    }
    ++initialized;
    if(!states[i].async_ctx){
      std::fprintf(stderr,"async init N=%d req=%d legacy fallback, expected opt-in\n",N,i);
      break;
    }
  }
  bool pass=(initialized==N);
  // vector<bool> is bit-packed; concurrent writes to different indices can
  // still touch the same storage word. Use disjoint byte slots instead.
  std::vector<char> outcomes((size_t)N,0);
  std::vector<std::string> errors((size_t)N);
  std::vector<double> maxerrs((size_t)N,0);
  if(pass){
    const auto t0=Clock::now();
    std::vector<std::thread> workers;
    workers.reserve((size_t)N);
    for(int i=0;i<N;++i){
      workers.emplace_back([&,i]{
        outcomes[i]=check_request(i,refs[i],states[i],errors[i],maxerrs[i]);
      });
    }
    for(auto& thread:workers) thread.join();
    const auto t1=Clock::now();
    const double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    std::printf("async N=%d steps/req=%d total_tokens=%d elapsed=%.3fms (correctness workload, NOT throughput benchmark)\n",
                N,kSteps,N*kSteps,ms);
    for(int i=0;i<N;++i){
      std::printf("  req=%d result=%s maxerr=%.9g %s\n",i,
                  outcomes[i]?"PASS":"FAIL",maxerrs[i],errors[i].c_str());
      pass=pass && outcomes[i];
    }
  }
  for(int i=0;i<initialized;++i) pp2::decode_free(states[i]);
  return pass;
}
}

int main(int argc, char** argv){
  // Run each N separately so watchdogs may cap wall time independently.
  // No mode argument: run both N=2 and N=4 in one process.
  const char* model=nullptr;
  int only_n=0;
  for(int i=1;i<argc;++i){
    if(std::strncmp(argv[i],"--n=",4)==0){
      only_n=std::atoi(argv[i]+4);
    } else if(std::strcmp(argv[i],"--n")==0 && i+1<argc){
      only_n=std::atoi(argv[++i]);
    } else if(argv[i][0]!='-' && !model){
      model=argv[i];
    } else {
      std::fprintf(stderr,"usage: test_async_multiuser [model.gguf] [--n=1|2|4]\n");
      return 2;
    }
  }
  if(only_n!=0 && only_n!=1 && only_n!=2 && only_n!=4){
    std::fprintf(stderr,"--n must be 1, 2 or 4\n"); return 2;
  }
#if defined(_WIN32)
  _putenv_s("HERMES_PP2_ASYNC_STREAMS", "");
#else
  unsetenv("HERMES_PP2_ASYNC_STREAMS");
#endif
  pp2::Stats stats;
  std::string err;
  bool uploaded=model
      ? pp2::upload(stats,err,std::string(model))
      : pp2::upload(stats,err);
  if(!uploaded){ std::fprintf(stderr,"upload failed: %s\n",err.c_str()); return 2; }

  const int ref_count=only_n?only_n:4;
  std::vector<Reference> refs((size_t)ref_count);
  for(int i=0;i<ref_count;++i){
    if(!build_reference(i,refs[i],err)){
      std::fprintf(stderr,"legacy reference req=%d failed: %s\n",i,err.c_str());
      pp2::free_all();
      return 2;
    }
  }
  std::printf("legacy golden: %d requests x 15 steps, logits %d floats per step\n",ref_count,kVocab);
#if defined(_WIN32)
  _putenv_s("HERMES_PP2_ASYNC_STREAMS", "1");
#else
  setenv("HERMES_PP2_ASYNC_STREAMS","1",1);
#endif
  const bool pass=(only_n==1)?run_parallel(1,refs)
    :(only_n==2)?run_parallel(2,refs)
    :(only_n==4)?run_parallel(4,refs)
    :(run_parallel(2,refs)&&run_parallel(4,refs));
  pp2::free_all();
  const std::string n_label=only_n?std::to_string(only_n):"2,4";
  std::printf("async real GGUF N=%s regression: %s\n",n_label.c_str(),pass?"PASS":"FAIL");
  return pass?0:1;
}
