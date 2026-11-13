// test_logits_dump: full CPU prefill, export corrected-reference fp32 logits.
// Old /tmp/e2_logits.bin is INVALID after the GGUF ssm_a/head-layout fix.
// Heavy full-model test, run only explicitly during a reserved CPU window.
#include "hermes/model_fwd.hpp"
#include <cstdio>
#include <algorithm>
using namespace hermes;
int main(){
  ModelFwd m;
  if(!m.open("/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf")){ printf("open FAIL\n"); return 2; }
  std::vector<int> ids={6205,161421,303,220,17,1374,9911,1147,27053,6059,13};
  auto lg = m.prefill(ids);
  FILE* f=fopen("/tmp/e2_logits_corrected.bin","wb");
  if(!f){ printf("open corrected logits dump FAIL\n"); m.close(); return 2; }
  size_t written=fwrite(lg.data(),4,lg.size(),f); fclose(f);
  if(written!=lg.size()){ printf("write corrected logits dump FAIL\n"); m.close(); return 2; }
  int T=(int)ids.size();
  float* last = lg.data()+(T-1)*ModelFwd::VOC;
  int top=0; for(int i=1;i<ModelFwd::VOC;++i) if(last[i]>last[top]) top=i;
  printf("E2 corrected dump top=%d val=%.2f n=%zu saved /tmp/e2_logits_corrected.bin\n", top, last[top], lg.size());
  m.close();
  return 0;
}
