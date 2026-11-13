// test_logits_dump_p2: second prompt (EN history, 13 tok), save full logits fp32.
// For cross-prompt E1vsE2 validation (review C). Heavy (~10min), run explicitly.
#include "hermes/model_fwd.hpp"
#include <cstdio>
#include <algorithm>
using namespace hermes;
int main(){
  ModelFwd m;
  if(!m.open("/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf")){ printf("open FAIL\n"); return 2; }
  std::vector<int> ids={8917,5437,537,279,10814,314,279,8323,21440,303,2250,22157,13};
  auto lg = m.prefill(ids);
  FILE* f=fopen("/tmp/e2_logits_p2.bin","wb"); fwrite(lg.data(),4,lg.size(),f); fclose(f);
  int T=(int)ids.size();
  float* last = lg.data()+(T-1)*ModelFwd::VOC;
  int top=0; for(int i=1;i<ModelFwd::VOC;++i) if(last[i]>last[top]) top=i;
  printf("E2-P2 dump top=%d val=%.2f n=%zu saved /tmp/e2_logits_p2.bin\n", top, last[top], lg.size());
  m.close();
  return 0;
}
