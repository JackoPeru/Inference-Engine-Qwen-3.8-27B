// HERMES-27 verified GREEDY MTP diagnostic generator, isolated CLI.
// Both PP2 and MTP use their existing public APIs. The target's step output
// ALWAYS chooses emitted tokens. MTP proposes exactly one token ahead; every
// proposal is checked before use. This is safe iterative verification, not an
// accelerated batched verifier or a claim of speculative token/s speedup.
//
// Standalone build from a scratch directory that links the installed build's
// hermes_mtp/hermes_pp2 libraries. No changes to production CMake required.
#include "pp2_fwd.hpp"
#include "mtp_gpu.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <io.h>
#define dup _dup
#define dup2 _dup2
#define close _close
#define fileno _fileno
#define fdopen _fdopen
#else
#include <unistd.h>
#endif

namespace {
constexpr int VOC=mtp_gpu::VOC;
constexpr int HID=mtp_gpu::HID;
constexpr int MAX_CONTEXT=pp2::DecodeState::MAX_CONTEXT;

struct Options{
  std::string model;
  std::vector<std::vector<int>> prompts;
  int max_new_tokens=4;
  int eos=-1;
  bool mtp=true;
  bool compare_target_only=false;
  bool verify_rollback=false;
  bool verify_prefill_hidden=false;
};

void usage(FILE* out){
  std::fprintf(out,
    "Usage: hermes_mtp_generate --model PATH.gguf --prompt-ids ID[,ID...] "
    "[--prompt-ids ...] [--max-new-tokens N] [--eos-token ID] "
    "[--mtp-off] [--compare-target-only] [--verify-rollback] "
    "[--verify-prefill-hidden] [--help]\n"
    "Target-verified greedy MTP K=1; no sampling and no speculative speedup claim.\n"
    "MTP draft KV is warmed from each verified input token after pos=0.\n"
    "Every MTP proposal is compared with authoritative PP2 logits. A rejection\n"
    "selects the target token and continues with committed target state.\n"
    "--compare-target-only independently replays the whole generated sequence.\n"
    "--verify-rollback checks draft KV rewind/replay on first generated token.\n"
    "Context limit 8192; no batch speculative target state commit API yet.\n");
}
bool parse_int(std::string_view v,int& out){
  if(v.empty())return false;
  const auto [p,ec]=std::from_chars(v.data(),v.data()+v.size(),out);
  return ec==std::errc{} && p==v.data()+v.size();
}
bool parse_ids(std::string_view s,std::vector<int>& ids){
  if(s.empty())return false;
  for(;;){
    auto sep=s.find(',');
    auto v=s.substr(0,sep);
    while(!v.empty() && v.front()==' ')v.remove_prefix(1);
    while(!v.empty() && v.back()==' ')v.remove_suffix(1);
    int id=-1;
    if(!parse_int(v,id)||id<0||id>=VOC)return false;
    ids.push_back(id);
    if(sep==std::string_view::npos)break;
    s.remove_prefix(sep+1);
    if(s.empty())return false;
  }
  return !ids.empty();
}
bool options(int argc,char** argv,Options& opts){
  for(int i=1;i<argc;++i){
    std::string_view key(argv[i]);
    if(key=="--help"||key=="-h"){usage(stdout);return false;}
    if(key=="--mtp-off"){opts.mtp=false;continue;}
    if(key=="--compare-target-only"){opts.compare_target_only=true;continue;}
    if(key=="--verify-rollback"){opts.verify_rollback=true;continue;}
    if(key=="--verify-prefill-hidden"){opts.verify_prefill_hidden=true;continue;}
    if(i+1>=argc){std::fprintf(stderr,"ERROR missing argument for %.*s\n",(int)key.size(),key.data());return false;}
    std::string_view value(argv[++i]);
    if(key=="--model")opts.model=value;
    else if(key=="--prompt-ids"){
      std::vector<int> ids;
      if(!parse_ids(value,ids)){std::fprintf(stderr,"ERROR bad --prompt-ids\n");return false;}
      opts.prompts.push_back(std::move(ids));
    } else if(key=="--max-new-tokens"){
      if(!parse_int(value,opts.max_new_tokens)||opts.max_new_tokens<1||opts.max_new_tokens>MAX_CONTEXT){
        std::fprintf(stderr,"ERROR max-new-tokens must be 1..%d\n",MAX_CONTEXT);return false;
      }
    } else if(key=="--eos-token"){
      if(!parse_int(value,opts.eos)||opts.eos<0||opts.eos>=VOC){
        std::fprintf(stderr,"ERROR invalid eos token\n");return false;
      }
    } else {std::fprintf(stderr,"ERROR unknown option %.*s\n",(int)key.size(),key.data());return false;}
  }
  if(opts.model.empty()||opts.prompts.empty()){
    std::fprintf(stderr,"ERROR --model and --prompt-ids required\n");usage(stderr);return false;
  }
  if(!opts.mtp && (opts.verify_rollback||opts.verify_prefill_hidden)){
    std::fprintf(stderr,"ERROR MTP verification flag used with --mtp-off\n");return false;
  }
  for(size_t i=0;i<opts.prompts.size();++i){
    const size_t L=opts.prompts[i].size();
    if(L==0 || L+(size_t)opts.max_new_tokens-1>MAX_CONTEXT){
      std::fprintf(stderr,"ERROR prompt %zu exceeds PP2 context capacity\n",i);return false;
    }
    if(opts.verify_prefill_hidden && L>64){
      std::fprintf(stderr,"ERROR --verify-prefill-hidden needs prompt <=64\n");return false;
    }
  }
  return true;
}

// PP2 itself writes progress to stdout; keep stdout machine-readable JSON.
class SplitOutput{
 public:
  SplitOutput(){
    std::fflush(stdout);
    int saved=dup(fileno(stdout));
    if(saved<0)return;
    if(dup2(fileno(stderr),fileno(stdout))<0){close(saved);return;}
    json_=fdopen(saved,"w");
    if(!json_)close(saved);
  }
  ~SplitOutput(){if(json_)std::fclose(json_);}
  FILE* json()const{return json_;}
 private:
  FILE* json_=nullptr;
};
void print_ids(FILE* out,const std::vector<int>& ids){
  std::fputc('[',out);
  for(size_t i=0;i<ids.size();++i)std::fprintf(out,"%s%d",i?",":"",ids[i]);
  std::fputc(']',out);
}
bool checked_argmax(const std::vector<float>& logits,int& idx,std::string& err){
  if(logits.size()!=VOC){err="unexpected logits size";return false;}
  float best=-std::numeric_limits<float>::infinity();idx=-1;
  for(int v=0;v<VOC;++v){
    const float x=logits[(size_t)v];
    if(!std::isfinite(x)){err="non-finite logits at "+std::to_string(v);return false;}
    if(x>best){best=x;idx=v;}
  }
  return idx>=0;
}
bool hidden_from_decode(const pp2::DecodeState& ds,std::vector<float>& hidden,std::string& err){
  if(!ds.inited || !ds.x1buf){err="no public PP2 x1buf hidden";return false;}
  hidden.resize(HID);
  cudaError_t e=cudaSetDevice(1);
  if(e==cudaSuccess)e=cudaMemcpy(hidden.data(),ds.x1buf,HID*4,cudaMemcpyDeviceToHost);
  if(e!=cudaSuccess){err="PP2 x1buf copy: "+std::string(cudaGetErrorString(e));return false;}
  for(float v:hidden)if(!std::isfinite(v)){err="PP2 hidden non-finite";return false;}
  return true;
}
struct ManagedStates{
  pp2::DecodeState target{};
  mtp_gpu::MtpState draft{};
  ~ManagedStates(){mtp_gpu::free_state(draft);pp2::decode_free(target);}
  ManagedStates(const ManagedStates&)=delete;
  ManagedStates& operator=(const ManagedStates&)=delete;
  ManagedStates()=default;
};

struct Results{
  std::vector<int> ids;
  int proposals_prompt=0,accepted_prompt=0;
  int proposals_gen=0,accepted_gen=0;
  double seed_ms=0,generation_ms=0,mtp_ms=0,target_verify_ms=0;
  std::string stop="length";
};
bool compare_roll_back(mtp_gpu::MtpState& draft,const std::vector<float>& hidden,
                       int tok,long pos,const std::vector<float>& original,
                       std::string& err){
  const int checkpoint=draft.len-1;
  if(checkpoint<0){err="rollback without last draft";return false;}
  if(!mtp_gpu::rewind(draft,checkpoint,err))return false;
  std::vector<float> replay=mtp_gpu::draft_step(hidden.data(),tok,draft,pos,err);
  if(replay.size()!=VOC || original.size()!=VOC){err="rollback replay missing logits: "+err;return false;}
  double max_abs=0;
  for(size_t i=0;i<replay.size();++i){
    if(!std::isfinite(replay[i]) || !std::isfinite(original[i])){
      err="rollback emitted non-finite";return false;
    }
    max_abs=std::max(max_abs,std::abs((double)replay[i]-original[i]));
  }
  int a=-1,b=-1;
  if(!checked_argmax(original,a,err)||!checked_argmax(replay,b,err))return false;
  std::fprintf(stderr,"REWIND pos=%ld before=%d after=%d max_abs=%.9g argmax=%d/%d\n",
               pos,checkpoint, draft.len,max_abs,a,b);
  if(max_abs>1e-4 || a!=b){err="MTP draft rollback replay mismatch";return false;}
  return true;
}

bool generate(const std::vector<int>& prompt,const Options& opts,int eos,
              size_t pi,pp2::Stats& stats,Results& result,std::string& err){
  ManagedStates st;
  const int capacity=(int)(prompt.size()+(size_t)opts.max_new_tokens-1);
  if(!pp2::decode_init(st.target,err,capacity))return false;
  if(opts.mtp && !mtp_gpu::init(st.draft,err,capacity))return false;
  std::vector<float> previous_hidden, current_hidden, logits;
  const auto seed_start=std::chrono::steady_clock::now();
  for(size_t t=0;t<prompt.size();++t){
    std::vector<float> proposal;
    if(opts.mtp && t>0){
      const auto a=std::chrono::steady_clock::now();
      proposal=mtp_gpu::draft_step(previous_hidden.data(),prompt[t],st.draft,(long)t,err);
      result.mtp_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-a).count();
      if(proposal.empty()){
        err="MTP seed pos="+std::to_string(t)+": "+err;return false;
      }
    }
    logits=pp2::decode_step(prompt[t],st.target,(long)t,err);
    if(logits.empty()){err="PP2 seed pos="+std::to_string(t)+": "+err;return false;}
    int teacher=-1;
    if(!checked_argmax(logits,teacher,err))return false;
    if(opts.mtp){
      if(!hidden_from_decode(st.target,current_hidden,err))return false;
      if(t>0){
        int draft_tok=-1;
        if(!checked_argmax(proposal,draft_tok,err))return false;
        ++result.proposals_prompt;
        result.accepted_prompt+=(draft_tok==teacher);
        std::fprintf(stderr,"MTP seed pos=%zu candidate=%d teacher=%d %s\n",
                     t,draft_tok,teacher,draft_tok==teacher?"ACCEPT":"REJECT");
      }
      previous_hidden.swap(current_hidden);
    }
  }
  result.seed_ms=std::chrono::duration<double,std::milli>(
      std::chrono::steady_clock::now()-seed_start).count();
  if(opts.verify_prefill_hidden){
    std::vector<float> full_hidden;
    std::vector<float> reference=pp2::prefill(prompt,stats,err,&full_hidden);
    if(reference.size()!=prompt.size()*(size_t)VOC || full_hidden.size()!=HID){
      err="full prefill/hidden size mismatch: "+err;return false;
    }
    double logit_delta=0,hidden_delta=0;
    const float* last=reference.data()+(prompt.size()-1)*(size_t)VOC;
    for(int i=0;i<VOC;++i)logit_delta=std::max(logit_delta,
        std::abs((double)logits[(size_t)i]-last[i]));
    for(int i=0;i<HID;++i)hidden_delta=std::max(hidden_delta,
        std::abs((double)previous_hidden[(size_t)i]-full_hidden[(size_t)i]));
    std::fprintf(stderr,"VERIFY prefill-hidden prompt=%zu max_hidden=%.7g max_logit=%.7g\n",
                 pi,hidden_delta,logit_delta);
    int a=-1,b=-1;std::vector<float> final(last,last+VOC);
    if(!checked_argmax(logits,a,err)||!checked_argmax(final,b,err))return false;
    if(a!=b || logit_delta>2e-3 || hidden_delta>5e-3){
      err="PP2 public x1buf hidden differs from prefill reference";return false;
    }
  }

  result.ids.reserve((size_t)opts.max_new_tokens);
  const auto gen_start=std::chrono::steady_clock::now();
  for(int step=0;step<opts.max_new_tokens;++step){
    int current=-1;
    if(!checked_argmax(logits,current,err))return false;
    result.ids.push_back(current); // NEVER emit unverified MTP proposal
    if(current==eos){result.stop="eos";break;}
    if(step+1==opts.max_new_tokens)break;
    const long pos=(long)prompt.size()+step;
    std::vector<float> proposal;
    if(opts.mtp){
      const auto t0=std::chrono::steady_clock::now();
      proposal=mtp_gpu::draft_step(previous_hidden.data(),current,st.draft,pos,err);
      result.mtp_ms+=std::chrono::duration<double,std::milli>(
          std::chrono::steady_clock::now()-t0).count();
      if(proposal.empty()){err="MTP generation pos="+std::to_string(pos)+": "+err;return false;}
      if(opts.verify_rollback && step==0 &&
         !compare_roll_back(st.draft,previous_hidden,current,pos,proposal,err))return false;
    }
    const auto verify_start=std::chrono::steady_clock::now();
    logits=pp2::decode_step(current,st.target,pos,err);
    result.target_verify_ms+=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-verify_start).count();
    if(logits.empty()){err="PP2 verify pos="+std::to_string(pos)+": "+err;return false;}
    if(opts.mtp){
      int draft_tok=-1,teacher=-1;
      if(!checked_argmax(proposal,draft_tok,err)||!checked_argmax(logits,teacher,err))return false;
      ++result.proposals_gen;
      result.accepted_gen+=(draft_tok==teacher);
      std::fprintf(stderr,"MTP verify pos=%ld candidate=%d teacher=%d %s\n",
          pos,draft_tok,teacher,draft_tok==teacher?"ACCEPT":"REJECT");
      if(!hidden_from_decode(st.target,previous_hidden,err))return false;
    }
  }
  result.generation_ms=std::chrono::duration<double,std::milli>(
      std::chrono::steady_clock::now()-gen_start).count();
  return true;
}

// Independent PP2 target-only replay using a fresh decode state, with identical
// prompt and EOS policy. This is the actual oracle for multi-token correctness.
bool target_only(const std::vector<int>& prompt,int count,int eos,
                 std::vector<int>& out,double& generation_ms,std::string& err){
  pp2::DecodeState state{};
  const int capacity=(int)(prompt.size()+(size_t)count-1);
  if(!pp2::decode_init(state,err,capacity))return false;
  std::vector<float> lg;
  for(size_t t=0;t<prompt.size();++t){
    lg=pp2::decode_step(prompt[t],state,(long)t,err);
    if(lg.empty()){pp2::decode_free(state);return false;}
  }
  const auto started=std::chrono::steady_clock::now();
  for(int step=0;step<count;++step){
    int tok=-1;
    if(!checked_argmax(lg,tok,err)){pp2::decode_free(state);return false;}
    out.push_back(tok);
    if(tok==eos||step+1==count)break;
    lg=pp2::decode_step(tok,state,(long)prompt.size()+step,err);
    if(lg.empty()){pp2::decode_free(state);return false;}
  }
  generation_ms=std::chrono::duration<double,std::milli>(
      std::chrono::steady_clock::now()-started).count();
  pp2::decode_free(state);
  return true;
}
} // namespace

int main(int argc,char** argv){
  Options opts;
  if(!options(argc,argv,opts)){
    for(int i=1;i<argc;++i)
      if(std::string_view(argv[i])=="--help" ||std::string_view(argv[i])=="-h")return 0;
    return 2;
  }
  SplitOutput redirect;
  FILE* json=redirect.json();
  if(!json){std::fprintf(stderr,"ERROR cannot reserve JSON stdout\n");return 1;}
  pp2::Stats stats;std::string err;
  if(!pp2::upload(stats,err,opts.model)){
    std::fprintf(stderr,"ERROR upload: %s\n",err.c_str());pp2::free_all();return 1;
  }
  int eos=opts.eos>=0?opts.eos:pp2::eos_token_id();
  if(eos<0){
    std::fprintf(stderr,"ERROR EOS unavailable (pass --eos-token)\n");
    pp2::free_all();return 1;
  }
  std::fprintf(stderr,"MODEL MTP=%s verified_K=1 context_limit=%d eos=%d "
                      "batch_verifier=unavailable\n",
               opts.mtp?"verify":"off",MAX_CONTEXT,eos);
  for(size_t i=0;i<opts.prompts.size();++i){
    Results result;
    if(!generate(opts.prompts[i],opts,eos,i,stats,result,err)){
      std::fprintf(stderr,"ERROR prompt=%zu: %s\n",i,err.c_str());
      pp2::free_all();return 1;
    }
    double baseline_ms=-1.;std::vector<int> reference;
    if(opts.compare_target_only){
      if(!target_only(opts.prompts[i],opts.max_new_tokens,eos,reference,baseline_ms,err)){
        std::fprintf(stderr,"ERROR target-only replay: %s\n",err.c_str());
        pp2::free_all();return 1;
      }
      if(reference!=result.ids){
        std::fprintf(stderr,"ERROR generated MTP-verifier tokens differ from target-only "
                            "on prompt %zu\n",i);
        pp2::free_all();return 2;
      }
      std::fprintf(stderr,"TARGET_ONLY_COMPARE prompt=%zu exact_tokens=PASS "
                          "baseline_gen_ms=%.3f\n",i,baseline_ms);
    }
    std::fprintf(stderr,"MTP_SUMMARY prompt=%zu generated=%zu verified=%d accepted=%d "
                        "rejected=%d prompt_draft=%d/%d seed_ms=%.3f "
                        "gen_ms=%.3f mtp_ms=%.3f target_step_ms=%.3f "
                        "baseline_ms=%.3f (no speculative speedup asserted)\n",
        i,result.ids.size(),result.proposals_gen,result.accepted_gen,
        result.proposals_gen-result.accepted_gen,result.accepted_prompt,
        result.proposals_prompt,result.seed_ms,result.generation_ms,
        result.mtp_ms,result.target_verify_ms,baseline_ms);
    std::fprintf(json,"{\"prompt_index\":%zu,\"prompt_ids\":",i);
    print_ids(json,opts.prompts[i]);
    std::fputs(",\"generated_token_ids\":",json);
    print_ids(json,result.ids);
    std::fprintf(json,",\"stop_reason\":\"%s\",\"mtp_mode\":\"%s\","
                      "\"verified_draft_count\":%d,\"accepted_draft_count\":%d,"
                      "\"rejected_draft_count\":%d,\"generation_ms\":%.3f,"
                      "\"target_only_matched\":%s}\n",
       result.stop.c_str(),opts.mtp?"verified_k1":"off",result.proposals_gen,
       result.accepted_gen,result.proposals_gen-result.accepted_gen,
       result.generation_ms,opts.compare_target_only?"true":"null");
    std::fflush(json);
  }
  pp2::free_all();return 0;
}
