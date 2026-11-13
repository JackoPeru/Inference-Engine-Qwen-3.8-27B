// P2 second prompt (EN, 13 tok): prefill PP2 vs E1 golden top5.
// E1 golden: [2250,279,271,8323,198] (torch fp32, direct norms).
#include "pp2_fwd.hpp"
#include <cstdio>
#include <algorithm>
int main(){
  pp2::Stats st; std::string err;
  if(!pp2::upload(st,err)){ printf("UPLOAD FAIL %s\n",err.c_str()); return 1; }
  std::vector<int> ids={8917,5437,537,279,10814,314,279,8323,21440,303,2250,22157,13};
  auto lg = pp2::prefill(ids,st,err);
  if(lg.empty()){ printf("PREFILL FAIL %s\n",err.c_str()); return 1; }
  int T=(int)ids.size(), VOC=248320;
  int top[5]={-1,-1,-1,-1,-1}; float tv[5]={-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
  float* last=lg.data()+(size_t)(T-1)*VOC;
  for(int i=0;i<VOC;++i){ float v=last[i];
    for(int k=0;k<5;++k) if(v>tv[k]){ for(int j=4;j>k;--j){tv[j]=tv[j-1];top[j]=top[j-1];} tv[k]=v;top[k]=i; break; } }
  printf("P2-P2 top5: [%d,%d,%d,%d,%d] vals [%.2f,%.2f,%.2f,%.2f,%.2f]\n",
    top[0],top[1],top[2],top[3],top[4],tv[0],tv[1],tv[2],tv[3],tv[4]);
  int exp[5]={2250,279,271,8323,198};
  bool ok=true; for(int k=0;k<5;++k) if(top[k]!=exp[k]) ok=false;
  printf("P2-P2 %s (E1 golden [2250,279,271,8323,198])\n", ok?"MATCH E1":"DIFFER");
  pp2::free_all();
  return ok?0:1;
}
