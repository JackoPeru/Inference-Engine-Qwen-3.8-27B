// P4 decode incrementale PP2: GATE1 singolo step 11->12 vs prefill-full 12,
// GATE2 teacher-forcing esteso k=1..3, greedy 4 token da prompt 11.
// Uso: test_pp2_decode [g1|g2|greedy|all] (default all, ferma al gate fallito)
#include "pp2_fwd.hpp"
#include <cstdio>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

static int argmax_v(const std::vector<float>& v){
  int bi=0; float bv=v[0];
  for(size_t i=1;i<v.size();++i) if(v[i]>bv){bv=v[i];bi=(int)i;}
  return bi;
}
static bool finite_all(const std::vector<float>& v){
  for(auto x:v) if(!std::isfinite(x)) return false;
  return true;
}
static void top5_1(const std::vector<float>& logits,int* oid,float* ov){
  for(int k=0;k<5;++k){oid[k]=-1;ov[k]=-1e30f;}
  for(size_t vv=0;vv<logits.size();++vv){
    float x=logits[vv];
    for(int k=0;k<5;++k) if(x>ov[k]){
      for(int j=4;j>k;--j){ov[j]=ov[j-1];oid[j]=oid[j-1];}
      ov[k]=x;oid[k]=(int)vv;break;
    }
  }
}

int main(int argc,char** argv){
  std::string only=argc>1?argv[1]:"all";
  printf("P4 PP2 decode T=11 prompt only=%s\n",only.c_str()); fflush(stdout);
  if(only=="bisect"){
    pp2::Stats bst; std::string berr;
    if(!pp2::upload(bst,berr)){ printf("UPLOAD FAIL %s\n",berr.c_str()); return 1; }
    // single token T=1: prefill vs decode pos0 (no history, S/KV zero)
    std::vector<int> ids1={6205};
    std::vector<float> pre1=pp2::prefill(ids1,bst,berr);
    if(pre1.empty()){ printf("PREFILL1 FAIL %s\n",berr.c_str()); return 1; }
    pp2::DecodeState ds;
    if(!pp2::decode_init(ds,berr)){ printf("INIT FAIL %s\n",berr.c_str()); return 1; }
    std::vector<float> dec1=pp2::decode_step(6205,ds,0L,berr);
    if(dec1.empty()){ printf("DEC1 FAIL %s\n",berr.c_str()); return 1; }
    double mx=0,me=0; size_t im=0;
    for(int v=0;v<248320;++v){ double e=fabs((double)dec1[v]-pre1[v]); if(e>mx){mx=e;im=v;} me+=e; }
    me/=248320.0;
    int ad=argmax_v(dec1), ap=argmax_v(pre1);
    printf("BISECT T=1 pos0: maxerr=%.6f at %zu meanerr=%.7f arg dec=%d pre=%d %s\n",mx,im,me,ad,ap,ad==ap?"SAME":"DIFF"); fflush(stdout);
    // T=2: prefill 2 vs decode 2 steps (conv/GDN/KV carry minimal)
    std::vector<int> ids2={6205,161421};
    std::vector<float> pre2=pp2::prefill(ids2,bst,berr);
    pp2::decode_free(ds);
    pp2::decode_init(ds,berr);
    auto d0=pp2::decode_step(6205,ds,0L,berr);
    auto d1=pp2::decode_step(161421,ds,1L,berr);
    {
      const float* p2=&pre2[(size_t)1*248320];
      double mx2=0,me2=0; size_t im2=0;
      for(int v=0;v<248320;++v){ double e=fabs((double)d1[v]-p2[v]); if(e>mx2){mx2=e;im2=v;} me2+=e; }
      me2/=248320.0;
      int ad2=argmax_v(d1), ap2=-1; float bv=-1e30f; for(int v=0;v<248320;++v) if(p2[v]>bv){bv=p2[v];ap2=v;}
      printf("BISECT T=2 pos1: maxerr=%.6f at %zu meanerr=%.7f arg dec=%d pre=%d %s\n",mx2,im2,me2,ad2,ap2,ad2==ap2?"SAME":"DIFF"); fflush(stdout);
      double mx0=0; for(int v=0;v<248320;++v){ double e=fabs((double)d0[v]-pre2[v]); if(e>mx0)mx0=e; }
      printf("BISECT T=2 pos0 vs pre2_t0: maxerr=%.6f (deve ~0 come T=1)\n",mx0); fflush(stdout);
    }
    pp2::decode_free(ds);
    return 0;
  }
  if(only=="bisect11"){
    pp2::Stats bst; std::string berr;
    if(!pp2::upload(bst,berr)){ printf("UPLOAD FAIL %s\n",berr.c_str()); return 1; }
    const int ids_arr[11]={6205,161421,303,220,17,1374,9911,1147,27053,6059,13};
    std::vector<int> ids(ids_arr,ids_arr+11);
    std::vector<float> pre=pp2::prefill(ids,bst,berr);
    if(pre.empty()){ printf("PREFILL FAIL %s\n",berr.c_str()); return 1; }
    pp2::DecodeState ds;
    if(!pp2::decode_init(ds,berr)){ printf("INIT FAIL %s\n",berr.c_str()); return 1; }
    for(int t=0;t<11;++t){
      std::vector<float> dec=pp2::decode_step(ids[t],ds,(long)t,berr);
      if(dec.empty()){ printf("DEC t=%d FAIL %s\n",t,berr.c_str()); return 1; }
      const float* pp=&pre[(size_t)t*248320];
      double mx=0,me=0; size_t im=0;
      for(int v=0;v<248320;++v){ double e=fabs((double)dec[v]-pp[v]); if(e>mx){mx=e;im=v;} me+=e; }
      me/=248320.0;
      int ad=argmax_v(dec), ap=-1; float bv=-1e30f; for(int v=0;v<248320;++v) if(pp[v]>bv){bv=pp[v];ap=v;}
      printf("BISECT11 t=%d: maxerr=%.6f at %zu meanerr=%.7f arg dec=%d pre=%d %s\n",t,mx,im,me,ad,ap,ad==ap?"SAME":"DIFF"); fflush(stdout);
      if(mx>1e-4) { printf("  DIVERGE at t=%d, stop bisect\n",t); break; }
    }
    pp2::decode_free(ds);
    return 0;
  }
  if(only=="layerbisect"){
    pp2::Stats bst; std::string berr;
    if(!pp2::upload(bst,berr)){ printf("UPLOAD FAIL %s\n",berr.c_str()); return 1; }
    const int ids_arr[4]={6205,161421,303,220};
    std::vector<int> ids(ids_arr,ids_arr+4);
    int f=pp2::debug_bisect_layer(ids);
    printf("LAYERBISECT first_diverge=%d\n",f); fflush(stdout);
    return 0;
  }
  if(only=="detail26"){
    pp2::Stats bst; std::string berr;
    if(!pp2::upload(bst,berr)){ printf("UPLOAD FAIL %s\n",berr.c_str()); return 1; }
    int r=pp2::debug_layer26_detail();
    printf("DETAIL26 done r=%d\n",r); fflush(stdout);
    return 0;
  }
  pp2::Stats st; std::string err;
  bool do_g1=(only=="all"||only=="g1"), do_g2=(only=="all"||only=="g2"), do_gr=(only=="all"||only=="greedy");
  if(!pp2::upload(st,err)){ printf("UPLOAD FAIL: %s\n",err.c_str()); return 1; }
  printf("UPLOAD ok sec=%.2f bytes0=%.3fGiB bytes1=%.3fGiB\n",
    st.upload_sec,st.bytes_gpu0/1024.0/1024/1024,st.bytes_gpu1/1024.0/1024/1024); fflush(stdout);

  const int ids11_arr[11]={6205,161421,303,220,17,1374,9911,1147,27053,6059,13};
  std::vector<int> ids11(ids11_arr,ids11_arr+11);
  // prefill 11 teacher
  std::vector<float> logits11=pp2::prefill(ids11,st,err);
  if(logits11.empty()){ printf("PREFILL11 FAIL %s\n",err.c_str()); return 1; }
  {
    int tid[5]; float tv[5];
    pp2::top5(logits11.data(),11,248320,tid,tv);
    printf("PREFILL11 top5 [%d,%d,%d,%d,%d] vals [%.2f,%.2f,%.2f,%.2f,%.2f] ms=%.1f\n",
      tid[0],tid[1],tid[2],tid[3],tid[4],tv[0],tv[1],tv[2],tv[3],tv[4],st.prefill_ms); fflush(stdout);
  }
  // decode init
  pp2::DecodeState ds;
  if(!pp2::decode_init(ds,err)){ printf("DECODE_INIT FAIL %s\n",err.c_str()); return 1; }
  printf("DECODE_INIT ok\n"); fflush(stdout);
  // warmup 11 steps pos 0..10
  std::vector<float> last_logits;
  {
    auto t0=std::chrono::high_resolution_clock::now();
    for(int t=0;t<11;++t){
      last_logits=pp2::decode_step(ids11[t],ds,(long)t,err);
      if(last_logits.empty()){ printf("WARMUP t=%d FAIL %s\n",t,err.c_str()); return 1; }
      if(!finite_all(last_logits)){ printf("WARMUP t=%d NaN/Inf FAIL\n",t); return 1; }
    }
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    printf("WARMUP 11 steps ms=%.1f avg/step=%.1f\n",ms,ms/11.0); fflush(stdout);
    // warmup last vs prefill11 last: self-consistency check (info, soglia 2e-3 come E2)
    {
      const float* p11=&logits11[(size_t)10*248320];
      double mx=0,me=0; int im=0;
      for(int v=0;v<248320;++v){ double e=fabs((double)last_logits[v]-p11[v]); if(e>mx){mx=e;im=v;} me+=e; }
      me/=248320.0;
      int a1=argmax_v(last_logits), a2=-1; float bv=-1e30f;
      for(int v=0;v<248320;++v) if(p11[v]>bv){bv=p11[v];a2=v;}
      printf("WARMUP check last: maxerr=%.5f at %d meanerr=%.6f arg dec=%d pre=%d %s\n",
        mx,im,me,a1,a2,a1==a2?"SAME":"DIFF"); fflush(stdout);
    }
  }
  int tok12=argmax_v(last_logits);
  printf("TOK12 greedy from warmup = %d (atteso 220 da GATE3-P2)\n",tok12); fflush(stdout);

  std::vector<int> gen; gen.push_back(tok12);
  std::vector<std::vector<float>> dec_logits_list;
  // GATE1: step pos11 vs prefill12 full
  if(do_g1||do_g2||do_gr||only=="all"){
    auto t0=std::chrono::high_resolution_clock::now();
    std::vector<float> dec12=pp2::decode_step(tok12,ds,11L,err);
    auto t1=std::chrono::high_resolution_clock::now();
    double step_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    if(dec12.empty()){ printf("GATE1 decode_step FAIL %s\n",err.c_str()); return 1; }
    printf("GATE1 decode_step tok=%d pos=11 ms=%.1f finite=%d\n",tok12,step_ms,(int)finite_all(dec12)); fflush(stdout);
    if(!finite_all(dec12)){ printf("GATE1 FAIL NaN\n"); return 1; }
    // teacher prefill 12
    std::vector<int> ids12=ids11; ids12.push_back(tok12);
    std::vector<float> logits12=pp2::prefill(ids12,st,err);
    if(logits12.empty()){ printf("GATE1 prefill12 FAIL %s\n",err.c_str()); return 1; }
    const float* p12=&logits12[(size_t)11*248320];
    double mx=0,me=0; size_t im=0;
    for(int v=0;v<248320;++v){ double e=fabs((double)dec12[v]-p12[v]); if(e>mx){mx=e;im=v;} me+=e; }
    me/=248320.0;
    int ad=argmax_v(dec12), ap=-1; float bv=-1e30f;
    for(int v=0;v<248320;++v) if(p12[v]>bv){bv=p12[v];ap=v;}
    int td[5]; float vd[5]; top5_1(dec12,td,vd);
    int tp[5]; float vp[5]; { for(int k=0;k<5;++k){tp[k]=-1;vp[k]=-1e30f;} for(int v=0;v<248320;++v){float x=p12[v]; for(int k=0;k<5;++k) if(x>vp[k]){for(int j=4;j>k;--j){vp[j]=vp[j-1];tp[j]=tp[j-1];}vp[k]=x;tp[k]=v;break;}} }
    printf("GATE1 compare 11->12: maxerr=%.5f at %zu meanerr=%.6f arg dec=%d pre=%d %s step_ms=%.1f prefill12_ms=%.1f\n",
      mx,im,me,ad,ap,ad==ap?"SAME":"DIFF",step_ms,st.prefill_ms); fflush(stdout);
    printf("GATE1 top5 dec [%d,%d,%d,%d,%d] [%.2f,%.2f,%.2f,%.2f,%.2f]\n",td[0],td[1],td[2],td[3],td[4],vd[0],vd[1],vd[2],vd[3],vd[4]); fflush(stdout);
    printf("GATE1 top5 pre [%d,%d,%d,%d,%d] [%.2f,%.2f,%.2f,%.2f,%.2f]\n",tp[0],tp[1],tp[2],tp[3],tp[4],vp[0],vp[1],vp[2],vp[3],vp[4]); fflush(stdout);
    size_t f0=0,t0m=0,f1=0,t1m=0;
    cudaSetDevice(0); cudaMemGetInfo(&f0,&t0m);
    cudaSetDevice(1); cudaMemGetInfo(&f1,&t1m);
    printf("GATE1 vram used0=%zu MiB used1=%zu MiB\n",(t0m-f0)/1048576,(t1m-f1)/1048576); fflush(stdout);
    bool pass=(mx<2e-3 && ad==ap);
    printf("GATE1 %s (soglia 2e-3 come E2 1.57e-03)\n",pass?"PASS":"FAIL"); fflush(stdout);
    dec_logits_list.push_back(dec12);
    if(only=="g1") return pass?0:1;
    if(!pass) return 1;
    // greedy continua: genera altri 3 (tot 4 da prompt)
    if(do_gr||only=="all"||do_g2){
      std::vector<int> seq=ids12;
      std::vector<float> prev=dec12;
      for(int k=2;k<=4;++k){
        int nxt=argmax_v(prev);
        gen.push_back(nxt);
        seq.push_back(nxt);
        long pos=11+k-1;
        size_t pf0=0,pt0=0,pf1=0,pt1=0;
        cudaSetDevice(0); cudaMemGetInfo(&pf0,&pt0);
        cudaSetDevice(1); cudaMemGetInfo(&pf1,&pt1);
        auto s0=std::chrono::high_resolution_clock::now();
        std::vector<float> dec=pp2::decode_step(nxt,ds,pos,err);
        auto s1=std::chrono::high_resolution_clock::now();
        double sms=std::chrono::duration<double,std::milli>(s1-s0).count();
        if(dec.empty()){ printf("GREEDY k=%d FAIL %s\n",k,err.c_str()); return 1; }
        size_t qf0=0,qt0=0,qf1=0,qt1=0;
        cudaSetDevice(0); cudaMemGetInfo(&qf0,&qt0);
        cudaSetDevice(1); cudaMemGetInfo(&qf1,&qt1);
        int td2[5]; float vd2[5]; top5_1(dec,td2,vd2);
        printf("GREEDY k=%d tok=%d pos=%ld ms=%.1f finite=%d top5 [%d,%d,%d,%d,%d] [%.2f,%.2f,%.2f,%.2f,%.2f] vram0 %zu->%zu MiB vram1 %zu->%zu MiB\n",
          k,nxt,pos,sms,(int)finite_all(dec),td2[0],td2[1],td2[2],td2[3],td2[4],vd2[0],vd2[1],vd2[2],vd2[3],vd2[4],
          (pt0-pf0)/1048576,(qt0-qf0)/1048576,(pt1-pf1)/1048576,(qt1-qf1)/1048576); fflush(stdout);
        if(!finite_all(dec)){ printf("GREEDY k=%d FAIL NaN\n",k); return 1; }
        // GATE2: vs prefill-full estesa
        if(do_g2||only=="all"){
          if(k<=3){
            std::vector<float> pre=pp2::prefill(seq,st,err);
            if(pre.empty()){ printf("GATE2 prefill k=%d FAIL %s\n",k,err.c_str()); return 1; }
            const float* pp=&pre[(size_t)(seq.size()-1)*248320];
            double mxx=0,mee=0; size_t imm=0;
            for(int v=0;v<248320;++v){ double e=fabs((double)dec[v]-pp[v]); if(e>mxx){mxx=e;imm=v;} mee+=e; }
            mee/=248320.0;
            int add2=argmax_v(dec), app2=-1; float b2=-1e30f;
            for(int v=0;v<248320;++v) if(pp[v]>b2){b2=pp[v];app2=v;}
            printf("GATE2 k=%d compare: maxerr=%.5f at %zu meanerr=%.6f arg dec=%d pre=%d %s\n",
              k,mxx,imm,mee,add2,app2,add2==app2?"SAME":"DIFF"); fflush(stdout);
            bool pok=(mxx<5e-3 && add2==app2);
            printf("GATE2 k=%d %s (soglia 5e-3)\n",k,pok?"PASS":"FAIL"); fflush(stdout);
            if(!pok) return 2;
          }
        }
        prev=dec;
        dec_logits_list.push_back(dec);
        if(only=="greedy" && k==4) break;
      }
      printf("GEN tokens da prompt11 (4): [%d,%d,%d,%d]\n",gen[0],gen[1],gen[2],gen[3]); fflush(stdout);
      printf("GATE2 PASS k=1..3 (se eseguito)\n"); fflush(stdout);
    }
  }
  pp2::decode_free(ds);
  // no leak check: pesi ancora residenti, free_all alla fine del processo (non qui per riuso)
  printf("P4 DONE gen=["); for(size_t i=0;i<gen.size();++i) printf("%d%s",gen[i],i+1<gen.size()? ",":""); printf("]\n");
  fflush(stdout);
  return 0;
}
