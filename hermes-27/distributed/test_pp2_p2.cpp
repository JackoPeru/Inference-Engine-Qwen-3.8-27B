// P2 second prompt (EN, 13 tokens): PP2 argmax vs llama.cpp b9704 CPU, same GGUF.
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
  const bool ok=top[0]==271;
  printf("P2-P2 %s (llama.cpp argmax 271)\n", ok?"MATCH":"DIFFER");
  pp2::free_all();
  return ok?0:1;
}
