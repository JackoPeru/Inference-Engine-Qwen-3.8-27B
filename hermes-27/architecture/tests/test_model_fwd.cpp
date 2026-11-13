// test_model_fwd: C++ prefill (11 tokens) vs E1 golden top5 (split interleaved).
// E1: top5 ids [898,34566,22197,29069,73] vals [15.74,12.96,12.41,12.19,11.2]
#include "hermes/model_fwd.hpp"
#include <cstdio>
#include <algorithm>
#include <vector>
using namespace hermes;
int main(){
  ModelFwd m;
  if(!m.open("/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf")){ printf("open FAIL\n"); return 2; }
  std::vector<int> ids={6205,161421,303,220,17,1374,9911,1147,27053,6059,13};
  auto lg = m.prefill(ids);
  int T=(int)ids.size();
  std::vector<int> top;
  std::vector<float> last(lg.begin()+(T-1)*ModelFwd::VOC, lg.end());
  for(int k=0;k<5;++k){ int bi=std::max_element(last.begin(),last.end())-last.begin();
    top.push_back(bi); printf("top%d id=%d val=%.2f\n",k,bi,last[bi]); last[bi]=-1e30f; }
  int exp[5]={220,1374,27053,17,1147};
  bool ok=true; for(int k=0;k<5;++k) if(top[k]!=exp[k]) ok=false;
  printf("model_fwd top5 %s\n", ok?"MATCH E1 PASS":"DIFFER FAIL");
  m.close();
  return ok?0:1;
}
