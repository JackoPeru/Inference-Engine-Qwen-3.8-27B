// CPU reference prefill on the real GGUF (expensive full-model test).
// Old E1/E2 top-5 fixtures were invalidated by the ssm_a and GDN head-layout fix.
// Independent llama.cpp b9704 with identical GGUF and prompt IDs predicts 271.
#include "hermes/model_fwd.hpp"
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <vector>
using namespace hermes;
int main(){
  ModelFwd m;
  if(!m.open("/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf")){ printf("open FAIL\n"); return 2; }
  std::vector<int> ids={6205,161421,303,220,17,1374,9911,1147,27053,6059,13};
  auto lg = m.prefill(ids);
  int T=(int)ids.size();
  if(lg.size() != (size_t)T*ModelFwd::VOC){ printf("logit size FAIL\n"); m.close(); return 2; }
  std::vector<int> top;
  std::vector<float> last(lg.begin()+(T-1)*ModelFwd::VOC, lg.end());
  if(!std::all_of(last.begin(), last.end(), [](float v){ return std::isfinite(v); })){
    printf("non-finite logits FAIL\n"); m.close(); return 2;
  }
  for(int k=0;k<5;++k){ int bi=std::max_element(last.begin(),last.end())-last.begin();
    top.push_back(bi); printf("top%d id=%d val=%.2f\n",k,bi,last[bi]); last[bi]=-1e30f; }
  bool ok=(top[0]==271);
  printf("model_fwd top1 %s (independent llama.cpp reference token=271)\n", ok?"PASS":"FAIL");
  m.close();
  return ok?0:1;
}
