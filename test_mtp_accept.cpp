// MTP K=1: target prefill -> h_last + top token -> mtp_draft -> draft top vs teacher next.
// Validates wiring (concat order, norms) via acceptance + sanity (finite, non-degenerate).
// Cost: 1 target prefill (~10min) + 1 draft layer (seconds) + 1 teacher prefill (~10min).
// Exit semantics (HERMES-27):
//   0 = draft finito + scala sana + draft_top == teacher_top (identita` greedy K=1)
//   1 = non-finito o scala assurda (invalido numerico)
//   2 = valido ma mismatch (wiring/acceptance da investigare, NON successo)
// NOTE: non eseguire in job leggeri; richiede 2 prefill ~20min (accodato altrove).
#include "hermes/model_fwd.hpp"
#include <cstdio>
#include <cmath>
#include <algorithm>
using namespace hermes;
int main(){
  ModelFwd m;
  if(!m.open("/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf")){ printf("open FAIL\n"); return 1; }
  std::vector<int> ids={6205,161421,303,220,17,1374,9911,1147,27053,6059,13};
  float h_last[ModelFwd::HID];
  printf("target prefill %d toks...\n", (int)ids.size()); fflush(stdout);
  auto lg = m.prefill(ids, h_last);
  int T=(int)ids.size();
  float* last = lg.data()+(T-1)*ModelFwd::VOC;
  int ttop = std::max_element(last, last+ModelFwd::VOC)-last;
  double mn=last[0], mx=last[0], me=0;
  for(int i=0;i<ModelFwd::VOC;++i){ me+=last[i]; if(last[i]<mn)mn=last[i]; if(last[i]>mx)mx=last[i]; }
  me/=ModelFwd::VOC;
  printf("target top=%d val=%.2f mean=%.3f min=%.2f max=%.2f\n", ttop, last[ttop], me, mn, mx); fflush(stdout);
  // verify next token via teacher: prefill ids+[ttop], take its top as ground truth next
  std::vector<int> ids2=ids; ids2.push_back(ttop);
  printf("teacher prefill %d toks...\n", (int)ids2.size()); fflush(stdout);
  auto lg2 = m.prefill(ids2);
  float* last2 = lg2.data()+(ids2.size()-1)*ModelFwd::VOC;
  int tnext = std::max_element(last2, last2+ModelFwd::VOC)-last2;
  printf("teacher next=%d val=%.2f\n", tnext, last2[tnext]); fflush(stdout);
  // draft K=1 from (h_last, ttop) at pos=T
  ModelFwd::MtpState mst;
  printf("mtp_draft pos=%d...\n", T); fflush(stdout);
  auto dlg = m.mtp_draft(h_last, ttop, mst, (long)T);
  int dtop = std::max_element(dlg.begin(), dlg.end())-dlg.begin();
  double dmn=dlg[0], dmx=dlg[0], dme=0; bool fin=true;
  for(auto v: dlg){ if(!std::isfinite(v)) fin=false; dme+=v; if(v<dmn)dmn=v; if(v>dmx)dmx=v; }
  dme/=dlg.size();
  if(!std::isfinite(dme)) fin=false;
  printf("draft top=%d val=%.2f mean=%.3f min=%.2f max=%.2f finite=%d\n", dtop, dlg[dtop], dme, dmn, dmx, (int)fin);
  printf("K=1 accept: draft %d vs teacher %d -> %s\n", dtop, tnext, (dtop==tnext?"ACCEPT":"REJECT"));
  m.close();
  // 1 = invalido numerico (non-finito o scala assurda)
  if(!fin || dmx >= 1e4 || dmn <= -1e4) return 1;
  // 0 = identita` greedy K=1, 2 = valido ma mismatch (NON successo)
  return (dtop == tnext) ? 0 : 2;
}
