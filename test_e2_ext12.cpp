// E2 extended cross-check: prefill base+[10] (12 tok), top-1 must equal E1 teacher pred 59.
#include "hermes/model_fwd.hpp"
#include <cstdio>
#include <algorithm>
using namespace hermes;
int main(){
  ModelFwd m;
  if(!m.open("/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf")){ printf("open FAIL\n"); return 2; }
  std::vector<int> ids={6205,161421,303,220,17,1374,9911,1147,27053,6059,13,10};
  auto lg = m.prefill(ids);
  int T=(int)ids.size();
  float* last = lg.data()+(T-1)*ModelFwd::VOC;
  int top=0; for(int i=1;i<ModelFwd::VOC;++i) if(last[i]>last[top]) top=i;
  printf("E2 ext12 top=%d val=%.2f (E1 golden 59) %s\n", top, last[top], top==59?"MATCH":"DIFFER");
  m.close();
  return top==59?0:1;
}
