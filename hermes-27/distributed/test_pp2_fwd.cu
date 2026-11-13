// P2 test: GATE1 upload+VRAM, GATE2 singolo layer, GATE3 full prefill T=11 top5, GATE4 full-logits.
// Uso: test_pp2_fwd [gate]  gate=all|g1|g2|g3|g4  (default all, ferma al gate fallito)
#include "pp2_fwd.hpp"
#include <cstdio>
#include <vector>
#include <string>
#include <cmath>

static bool read_bin(const char* p, std::vector<float>& o, size_t n){
  FILE* f=fopen(p,"rb"); if(!f){ printf("OPEN FAIL %s\n",p); return false; }
  fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
  if((size_t)sz!=n*4){ printf("SIZE MISMATCH %s got %ld expect %zu\n",p,sz,n*4); fclose(f); return false; }
  o.resize(n); size_t r=fread(o.data(),4,n,f); fclose(f); return r==n;
}

int main(int argc,char** argv){
  std::string only=argc>1?argv[1]:"all";
  printf("P2 PP2 forward T=11 full 64L test only=%s\n",only.c_str());
  fflush(stdout);
  pp2::Stats st; std::string err;
  bool do_g1=(only=="all"||only=="g1"), do_g2=(only=="all"||only=="g2"),
       do_g3=(only=="all"||only=="g3"), do_g4=(only=="all"||only=="g4");
  // GATE1
  if(do_g1||do_g3||do_g4||do_g2){
    if(!pp2::upload(st,err)){ printf("GATE1 FAIL: %s\n",err.c_str()); return 1; }
    printf("GATE1 PASS upload_sec=%.2f convert_one_ms=%.2f bytes0=%.3fGiB bytes1=%.3fGiB\n",
      st.upload_sec,st.convert_ms_one,st.bytes_gpu0/1024.0/1024/1024,st.bytes_gpu1/1024.0/1024/1024);
    fflush(stdout);
    if(only=="g1") return 0;
  }
  // GATE2
  if(do_g2||only=="all"){
    int r0=pp2::check_blk0_gpu0();
    if(r0!=0){ printf("GATE2 FAIL blk0\n"); return 2; }
    int r1=pp2::check_blk32_gpu1();
    if(r1!=0){ printf("GATE2 FAIL blk32\n"); return 2; }
    printf("GATE2 PASS\n"); fflush(stdout);
    if(only=="g2") return 0;
  }
  // GATE3 full prefill
  std::vector<float> logits;
  if(do_g3||do_g4||only=="all"){
    const int ids_arr[11]={6205,161421,303,220,17,1374,9911,1147,27053,6059,13};
    std::vector<int> ids(ids_arr,ids_arr+11);
    logits=pp2::prefill(ids,st,err);
    if(logits.empty()){ printf("GATE3 FAIL prefill: %s\n",err.c_str()); return 3; }
    int tid[5]; float tv[5];
    pp2::top5(logits.data(),11,248320,tid,tv);
    printf("GATE3 logits T=11 V=248320 prefill_ms=%.1f xfer_ms=%.3f\n",st.prefill_ms,st.xfer_ms);
    printf("GATE3 top5: [%d,%d,%d,%d,%d] vals [%.2f,%.2f,%.2f,%.2f,%.2f]\n",
      tid[0],tid[1],tid[2],tid[3],tid[4],tv[0],tv[1],tv[2],tv[3],tv[4]);
    printf("GATE3 golden top5 [220,1374,27053,17,1147]\n");
    fflush(stdout);
    bool match=(tid[0]==220&&tid[1]==1374&&tid[2]==27053&&tid[3]==17&&tid[4]==1147);
    printf("GATE3 %s\n",match?"PASS top5 MATCH":"FAIL top5 MISMATCH (bisect sopra, stop)");
    if(!match) return 3;
    if(only=="g3") return 0;
  }
  // GATE4 full-logits vs /tmp/e2_logits.bin
  if(do_g4||only=="all"){
    if(logits.empty()){
      const int ids_arr[11]={6205,161421,303,220,17,1374,9911,1147,27053,6059,13};
      std::vector<int> ids(ids_arr,ids_arr+11);
      logits=pp2::prefill(ids,st,err);
      if(logits.empty()){ printf("GATE4 FAIL prefill: %s\n",err.c_str()); return 4; }
    }
    std::vector<float> ref;
    if(!read_bin("/tmp/e2_logits.bin",ref,(size_t)11*248320)){ printf("GATE4 FAIL no golden\n"); return 4; }
    double mx=0,me=0; size_t im=0; int arg_ok=0, top50_ok=0;
    for(int t=0;t<11;++t){
      const float* g=&logits[(size_t)t*248320];
      const float* r=&ref[(size_t)t*248320];
      int ag=-1, ar=-1; float bg=-1e30f, br=-1e30f;
      double mxt=0; int imt=0;
      for(int v=0;v<248320;++v){ double e=fabs((double)g[v]-r[v]); if(e>mxt){mxt=e;imt=v;} if(e>mx){mx=e;im=(size_t)t*248320+v;} me+=e;
        if(g[v]>bg){bg=g[v];ag=v;} if(r[v]>br){br=r[v];ar=v;} }
      if(ag==ar) arg_ok++;
      // top5 per token per diagnosi
      int tg5[5]={-1,-1,-1,-1,-1}, tr5[5]={-1,-1,-1,-1,-1}; float vg5[5]={-1e30f,-1e30f,-1e30f,-1e30f,-1e30f}, vr5[5]={-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
      for(int v=0;v<248320;++v){
        float xg=g[v], xr=r[v];
        for(int k=0;k<5;++k) if(xg>vg5[k]){ for(int j=4;j>k;--j){vg5[j]=vg5[j-1];tg5[j]=tg5[j-1];} vg5[k]=xg;tg5[k]=v; break; }
        for(int k=0;k<5;++k) if(xr>vr5[k]){ for(int j=4;j>k;--j){vr5[j]=vr5[j-1];tr5[j]=tr5[j-1];} vr5[k]=xr;tr5[k]=v; break; }
      }
      printf("GATE4 t=%d arg gpu=%d (%.2f) ref=%d (%.2f) %s maxerr_t=%.3f at %d | top5gpu [%d,%d,%d,%d,%d] top5ref [%d,%d,%d,%d,%d]\n",
        t,ag,bg,ar,br,ag==ar?"OK":"DIFF",mxt,imt,tg5[0],tg5[1],tg5[2],tg5[3],tg5[4],tr5[0],tr5[1],tr5[2],tr5[3],tr5[4]);
      fflush(stdout);
      // top50 overlap
      std::vector<int> tg(50,-1), tr(50,-1); std::vector<float> vg(50,-1e30f), vr(50,-1e30f);
      for(int v=0;v<248320;++v){
        float xg=g[v], xr=r[v];
        for(int k=0;k<50;++k) if(xg>vg[k]){ for(int j=49;j>k;--j){vg[j]=vg[j-1];tg[j]=tg[j-1];} vg[k]=xg;tg[k]=v; break; }
        for(int k=0;k<50;++k) if(xr>vr[k]){ for(int j=49;j>k;--j){vr[j]=vr[j-1];tr[j]=tr[j-1];} vr[k]=xr;tr[k]=v; break; }
      }
      int ov=0; for(int a=0;a<50;++a) for(int b=0;b<50;++b) if(tg[a]==tr[b]){ov++;break;}
      if(t==10) top50_ok=ov;
      if(t==10) printf("GATE4 t=10 top50_overlap=%d/50\n",ov);
    }
    me/=(11.0*248320);
    printf("GATE4 maxerr=%.5f (at %zu t=%zu v=%zu) meanerr=%.6f argmax %d/11 top50_last %d/50\n",mx,im,im/248320,im%248320,me,arg_ok,top50_ok);
    printf("GATE4 golden E2==E1 maxerr 7.75e-03 mean 4.17e-04 argmax 11/11 top50 50/50 (riferimento)\n");
    fflush(stdout);
    bool pass=(arg_ok==11);
    printf("GATE4 %s (maxerr report, non forzare soglia rumore)\n",pass?"PASS argmax 11/11":"FAIL argmax");
    return pass?0:4;
  }
  return 0;
}
