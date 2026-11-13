// HERMES-27 strict, multi-step verified greedy MTP regression test.
// Former k3/sweep tests reused one frozen h_last, verified with full rescans,
// had no target state commit/rollback and used an invented bonus TPS numerator.
// They have been removed. Exit 0 requires real GPU evidence, not just finiteness.
//
// Run only with exclusive GPU access and real GGUF checkpoint uploaded:
//   test_mtp_gpu [verified|verified-b|all]
// Default `verified` uses 11-token prompt and 5 generated greedy positions.
#include "mtp_gpu.hpp"
#include "pp2_fwd.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {
constexpr int VOC=mtp_gpu::VOC;
constexpr int HID=mtp_gpu::HID;
const std::vector<int> A={6205,161421,303,220,17,1374,9911,1147,27053,6059,13};
const std::vector<int> B={6205,161421,303,220,17,1374,9911,1147,27053,6059,13,220,1374};

bool argmax(const std::vector<float>& logits,int& idx,const char* label){
  if(logits.size()!=VOC){std::printf("FAIL %s invalid logit size=%zu\n",label,logits.size());return false;}
  float best=-std::numeric_limits<float>::infinity();idx=-1;
  for(int k=0;k<VOC;++k){
    const float v=logits[(size_t)k];
    if(!std::isfinite(v)){
      std::printf("FAIL %s non-finite logit=%d\n",label,k);return false;
    }
    if(v>best){best=v;idx=k;}
  }
  return idx>=0;
}
bool exact_or_close(const std::vector<float>& a,const std::vector<float>& b,
                    const char* label,double tolerance){
  if(a.size()!=b.size()||a.size()!=VOC){
    std::printf("FAIL %s invalid vectors\n",label);return false;
  }
  double max_err=0;
  for(int k=0;k<VOC;++k){
    if(!std::isfinite(a[(size_t)k])||!std::isfinite(b[(size_t)k])){
      std::printf("FAIL %s non-finite\n",label);return false;
    }
    max_err=std::max(max_err,std::abs((double)a[(size_t)k]-b[(size_t)k]));
  }
  int ta=-1,tb=-1;
  if(!argmax(a,ta,label)||!argmax(b,tb,label))return false;
  std::printf("CHECK %-24s max_abs=%.8g top=%d/%d %s\n",
      label,max_err,ta,tb,(max_err<=tolerance && ta==tb)?"PASS":"FAIL");
  std::fflush(stdout);
  return max_err<=tolerance && ta==tb;
}
bool capture_hidden(const pp2::DecodeState& st,std::vector<float>& hidden,std::string& err){
  if(!st.inited||!st.x1buf){err="PP2 x1buf absent";return false;}
  hidden.resize(HID);
  cudaError_t ce=cudaSetDevice(1);
  if(ce==cudaSuccess)ce=cudaMemcpy(hidden.data(),st.x1buf,HID*4,cudaMemcpyDeviceToHost);
  if(ce!=cudaSuccess){err="capture target hidden: "+std::string(cudaGetErrorString(ce));return false;}
  for(float v:hidden)if(!std::isfinite(v)){err="nonfinite target hidden";return false;}
  return true;
}
struct States{
  pp2::DecodeState target{};
  pp2::DecodeState reference{};
  mtp_gpu::MtpState draft{};
  ~States(){mtp_gpu::free_state(draft);pp2::decode_free(target);pp2::decode_free(reference);}
};

bool run_one(const std::vector<int>& P,int steps,const char* name,pp2::Stats& stats){
  std::printf("\n=== STRICT MTP VERIFIED %s prompt_len=%zu steps=%d ===\n",name,P.size(),steps);
  std::fflush(stdout);
  std::string err;
  States st;
  const int capacity=(int)P.size()+steps-1;
  if(!pp2::decode_init(st.target,err,capacity)||
     !pp2::decode_init(st.reference,err,capacity)||
     !mtp_gpu::init(st.draft,err,capacity)){
    std::printf("FAIL state init: %s\n",err.c_str());return false;
  }
  std::vector<float> prev_hidden,logits,teacher_log;
  int seed_proposals=0,seed_accepted=0;
  for(size_t t=0;t<P.size();++t){
    std::vector<float> proposed;
    if(t>0){
      proposed=mtp_gpu::draft_step(prev_hidden.data(),P[t],st.draft,(long)t,err);
      if(proposed.empty()){
        std::printf("FAIL seed MTP t=%zu: %s\n",t,err.c_str());return false;
      }
    }
    logits=pp2::decode_step(P[t],st.target,(long)t,err);
    teacher_log=pp2::decode_step(P[t],st.reference,(long)t,err);
    if(logits.empty()||teacher_log.empty()||
       !exact_or_close(logits,teacher_log,"independent-target-seed",1e-4)){
      std::printf("FAIL seed teacher t=%zu: %s\n",t,err.c_str());return false;
    }
    if(t>0){
      int td=-1,tt=-1;
      if(!argmax(proposed,td,"seed draft")||!argmax(logits,tt,"seed teacher"))return false;
      ++seed_proposals;
      seed_accepted+=(td==tt);
      std::printf("SEED pos=%zu draft=%d target=%d accepted=%d\n",t,td,tt,(int)(td==tt));
    }
    if(!capture_hidden(st.target,prev_hidden,err)){
      std::printf("FAIL seed hidden t=%zu: %s\n",t,err.c_str());return false;
    }
  }
  // Existing public x1buf must be the pre-output-norm hidden advertised for MTP.
  // Cross-validate with the official pp2::prefill(..., &h_last) accessor.
  std::vector<float> pre_hidden;
  std::vector<float> pre_logits=pp2::prefill(P,stats,err,&pre_hidden);
  if(pre_hidden.size()!=HID || pre_logits.size()!=P.size()*(size_t)VOC){
    std::printf("FAIL prefill hidden reference: %s\n",err.c_str());return false;
  }
  double hidden_err=0;
  for(int i=0;i<HID;++i){
    if(!std::isfinite(pre_hidden[(size_t)i]))return false;
    hidden_err=std::max(hidden_err,std::abs((double)pre_hidden[(size_t)i]-prev_hidden[(size_t)i]));
  }
  const std::vector<float> last_prefill(pre_logits.end()-VOC,pre_logits.end());
  if(!exact_or_close(last_prefill,logits,"prefill-vs-decode",2e-3) || hidden_err>5e-3){
    std::printf("FAIL prefill hidden max_abs=%g\n",hidden_err);return false;
  }
  std::printf("HIDDEN map max_abs=%g PASS\n",hidden_err);

  std::vector<int> outputsA,outputsB;
  int accepted=0,proposed=0;
  for(int step=0;step<steps;++step){
    int tokA=-1,tokB=-1;
    if(!argmax(logits,tokA,"target")||!argmax(teacher_log,tokB,"target-only"))return false;
    outputsA.push_back(tokA);
    outputsB.push_back(tokB);
    if(tokA!=tokB){std::printf("FAIL stream token mismatch step=%d\n",step);return false;}
    if(step+1==steps)break;
    const long pos=(long)P.size()+step;
    const int checkpoint=st.draft.len;
    std::vector<float> proposed_log=mtp_gpu::draft_step(
        prev_hidden.data(),tokA,st.draft,pos,err);
    if(proposed_log.empty()){
      std::printf("FAIL MTP draft step=%d: %s\n",step,err.c_str());return false;
    }
    if(st.draft.len!=checkpoint+1){std::printf("FAIL draft len advance\n");return false;}
    if(step==0){
      // Real GPU rollback/replay: remove a pending draft KV entry and rerun it
      // from the exact same verified target prefix; target state stays untouched.
      if(!mtp_gpu::rewind(st.draft,checkpoint,err))return false;
      if(st.draft.len!=checkpoint){std::printf("FAIL rewind logical length\n");return false;}
      std::vector<float> again=mtp_gpu::draft_step(
          prev_hidden.data(),tokA,st.draft,pos,err);
      if(again.empty()||!exact_or_close(proposed_log,again,"MTP rewind-replay",1e-4))return false;
    }
    logits=pp2::decode_step(tokA,st.target,pos,err);
    teacher_log=pp2::decode_step(tokB,st.reference,pos,err);
    if(logits.empty()||teacher_log.empty()||
       !exact_or_close(logits,teacher_log,"target-after-verification",1e-4)){
      std::printf("FAIL post-verification state step=%d: %s\n",step,err.c_str());return false;
    }
    int draft_tok=-1,teacher_tok=-1;
    if(!argmax(proposed_log,draft_tok,"MTP")||!argmax(logits,teacher_tok,"PP2"))return false;
    ++proposed;accepted+=(draft_tok==teacher_tok);
    std::printf("VERIFY step=%d pos=%ld draft=%d target=%d %s\n",
                step,pos,draft_tok,teacher_tok,draft_tok==teacher_tok?"ACCEPT":"REJECT");
    // Every next state is driven by the target-selected token. Rejections
    // cannot contaminate the target GDN/conv/KV state or the MTP KV history.
    if(!capture_hidden(st.target,prev_hidden,err))return false;
  }
  if(outputsA!=outputsB){std::printf("FAIL target stream differs\n");return false;}
  std::printf("VERIFIED PASS %s: generated %zu target-only-identical tokens, "
              "MTP_seed=%d/%d, MTP_generated=%d/%d, rejects=%d,"
              " rollback_gpu=PASS; TPS claim=NONE\n",
      name,outputsA.size(),seed_accepted,seed_proposals,accepted,proposed,proposed-accepted);
  std::fflush(stdout);
  return true;
}
} // namespace

int main(int argc,char** argv){
  std::string mode=argc>1?argv[1]:"verified";
  if(mode!="verified" && mode!="verified-b" && mode!="all"){
    std::fprintf(stderr,"usage: test_mtp_gpu [verified|verified-b|all]\n"
                        "Legacy k3/sweep modes removed: frozen hidden and full-"
                        "prefill verify had no state commit/rollback.\n");
    return 2;
  }
  pp2::Stats stats;std::string err;
  if(!pp2::upload(stats,err)){
    std::printf("FAIL upload: %s\n",err.c_str());pp2::free_all();return 1;
  }
  const bool a=mode!="verified-b";
  const bool b=mode=="verified-b"||mode=="all";
  bool success=true;
  if(a)success=run_one(A,5,"A11",stats);
  if(success && b)success=run_one(B,5,"B13",stats);
  pp2::free_all();
  std::printf("TEST_MTP_GPU %s\n",success?"PASS":"FAIL");
  return success?0:1;
}
