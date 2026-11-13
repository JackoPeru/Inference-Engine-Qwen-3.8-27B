// test_logits_dump: full prefill, save all logits raw fp32 to /tmp/e2_logits.bin + top5.
// For full-logits E1vsE2 comparison (review C). Heavy (~10min), run only in queue.
#include "hermes/model_fwd.hpp"
#include <cstdio>
#include <algorithm>
using namespace hermes;
int main(){
  ModelFwd m;
  if(!m.open("/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf")){ printf("open FAIL\n"); return 2; }
  std::vector<int> ids={6205,161421,303,220,17,1374,9911,1147,27053,6059,13};
  auto lg = m.prefill(ids);
  FILE* f=fopen("/tmp/e2_logits.bin","wb"); fwrite(lg.data(),4,lg.size(),f); fclose(f);
  int T=(int)ids.size();
  float* last = lg.data()+(T-1)*ModelFwd::VOC;
  int top=0; for(int i=1;i<ModelFwd::VOC;++i) if(last[i]>last[top]) top=i;
  printf("E2 dump top=%d val=%.2f n=%zu saved /tmp/e2_logits.bin\n", top, last[top], lg.size());
  m.close();
  return 0;
}
