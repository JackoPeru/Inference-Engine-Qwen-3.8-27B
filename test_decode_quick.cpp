// quick decode carry check with T=3 (full 64 layers but short seq).
#include "hermes/model_fwd.hpp"
#include <cstdio>
#include <cmath>
#include <algorithm>
using namespace hermes;
int main(){
  ModelFwd m;
  if(!m.open("/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf")){ printf("open FAIL\n"); return 2; }
  std::vector<int> ids={6205,161421,303};
  printf("full prefill %d toks...\n", (int)ids.size()); fflush(stdout);
  auto lg_full = m.prefill(ids);
  int T=(int)ids.size();
  float* last_full = lg_full.data()+(T-1)*ModelFwd::VOC;
  int topf = std::max_element(last_full, last_full+ModelFwd::VOC)-last_full;
  printf("full top=%d val=%.2f\n", topf, last_full[topf]); fflush(stdout);
  printf("incremental prefill_with_state...\n"); fflush(stdout);
  ModelFwd::SeqState st; std::vector<float> lg_last;
  m.prefill_with_state(ids, lg_last, st);
  double me=0, mx=0;
  for(int i=0;i<ModelFwd::VOC;++i){ double d=fabs((double)lg_last[i]-(double)last_full[i]); me+=d; if(d>mx)mx=d; }
  me/=ModelFwd::VOC;
  printf("prefill_vs_incr: maxerr=%.3e meanerr=%.3e %s\n", mx, me, (mx<1e-2?"PASS":"FAIL")); fflush(stdout);
  int nxt = topf;
  printf("decode_step tok=%d pos=%d...\n", nxt, T); fflush(stdout);
  auto lg_step = m.decode_step(nxt, st, (long)T);
  std::vector<int> ids2=ids; ids2.push_back(nxt);
  printf("full prefill %d toks (teacher)...\n", (int)ids2.size()); fflush(stdout);
  auto lg2 = m.prefill(ids2);
  float* last2 = lg2.data()+(ids2.size()-1)*ModelFwd::VOC;
  int top2 = std::max_element(last2, last2+ModelFwd::VOC)-last2;
  int tops = std::max_element(lg_step.begin(), lg_step.end())-lg_step.begin();
  printf("teacher top=%d step top=%d\n", top2, tops);
  me=0; mx=0;
  for(int i=0;i<ModelFwd::VOC;++i){ double d=fabs((double)lg_step[i]-(double)last2[i]); me+=d; if(d>mx)mx=d; }
  me/=ModelFwd::VOC;
  printf("decode_vs_teacher: maxerr=%.3e meanerr=%.3e %s\n", mx, me, (mx<5e-2?"PASS":"FAIL"));
  m.close();
  return (mx<5e-2)?0:1;
}
