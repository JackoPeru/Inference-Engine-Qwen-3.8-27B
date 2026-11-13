// MTP K=1 acceptance: draft top-1 vs target top-1.
// Needs target last hidden (pre-output_norm). ModelFwd::prefill currently returns
// only logits, so this test re-runs a hidden-capturing forward via mtp path:
// here we validate wiring sanity (finite, non-degenerate draft) + acceptance vs target.
// Full acceptance-rate sweep K=1..5 on sample prompts follows after golden teacher.
#include "hermes/model_fwd.hpp"
#include <cstdio>
#include <algorithm>
using namespace hermes;
int main(){
  ModelFwd m;
  if(!m.open("/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf")){ printf("open FAIL\n"); return 2; }
  // NOTE: mtp_draft needs h_last (pre-output_norm hidden). Prefill does not expose it yet.
  // This stub only resolves tensors (no forward: teacher is running, avoid RAM contention).
  const char* need[]={"blk.64.nextn.eh_proj.weight","blk.64.nextn.enorm.weight","blk.64.nextn.hnorm.weight",
    "blk.64.nextn.shared_head_norm.weight","blk.64.attn_q.weight","blk.64.attn_k.weight",
    "blk.64.attn_v.weight","blk.64.attn_output.weight","blk.64.attn_norm.weight",
    "blk.64.post_attention_norm.weight","blk.64.ffn_gate.weight","blk.64.ffn_up.weight",
    "blk.64.ffn_down.weight","token_embd.weight","output.weight"};
  bool ok=true;
  for(auto n: need){ if(!m.find(n)){ printf("MISSING %s\n", n); ok=false; } }
  printf("mtp tensors %s\n", ok?"ALL PRESENT":"MISSING");
  m.close();
  return ok?0:1;
}
