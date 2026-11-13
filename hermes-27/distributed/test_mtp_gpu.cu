// HERMES-27 MTP GPU test: GATE1 (K=1 draft singolo) / GATE2 (K=3 autoregressivo)
// / sweep (verify+accept K=1..4 su 2 prompt).
// Uso: test_mtp_gpu [k1|k3|sweepA|sweepB|sweep|all] (default all, ferma al gate fallito).
// Exit: k1: 0=ACCEPT (draft==teacher), 2=mismatch valido, 1=invalido/errore.
//       k3/sweep: 0=sempre se numeri finiti (report onesto, anche se accept bassa),
//       1=errore/invalido numerico. NON falsare exit code su accept parziale.
// Limiti: VRAM abort >15GB (via pp2::upload + check locali), run <10min per gate.
#include "pp2_fwd.hpp"
#include "mtp_gpu.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>

static const std::vector<int> PA = {6205,161421,303,220,17,1374,9911,1147,27053,6059,13}; // 11-tok base
// 13-tok: base11 + teacher top2 (continuazione teacher-forced, documentata in MTP_GPU.md)
static const std::vector<int> PB = {6205,161421,303,220,17,1374,9911,1147,27053,6059,13,220,1374};

static int argmax_v(const std::vector<float>& v){
  int bi=0; float bv=v[0];
  for(size_t i=1;i<v.size();++i) if(v[i]>bv){bv=v[i];bi=(int)i;}
  return bi;
}
static void top5_of(const float* last, int* oid, float* ov){
  for(int k=0;k<5;++k){ oid[k]=-1; ov[k]=-1e30f; }
  for(int v=0;v<248320;++v){
    float x=last[v];
    for(int k=0;k<5;++k) if(x>ov[k]){
      for(int j=4;j>k;--j){ ov[j]=ov[j-1]; oid[j]=oid[j-1]; }
      ov[k]=x; oid[k]=v; break;
    }
  }
}
static int topover(const int* a, const int* b){
  int n=0;
  for(int i=0;i<5;++i) for(int j=0;j<5;++j) if(a[i]==b[j]){ n++; break; }
  return n;
}
static void vram_check(const char* tag, bool& over){
  size_t f0=0,t0=0,f1=0,t1=0;
  cudaSetDevice(0); cudaMemGetInfo(&f0,&t0);
  cudaSetDevice(1); cudaMemGetInfo(&f1,&t1);
  size_t u0=(t0-f0)/1048576, u1=(t1-f1)/1048576;
  printf("vram %s used0=%zu MiB used1=%zu MiB (limit 15360)\n",tag,u0,u1); fflush(stdout);
  over = (u0>15360||u1>15360);
}

// GATE1: draft singolo K=1. Ritorna 0 accept, 2 mismatch, 1 errore/invalido.
static int gate_k1(){
  printf("\n=== GATE1 MTP draft singolo K=1 (GPU) ===\n"); fflush(stdout);
  pp2::Stats st; std::string err;
  if(!pp2::upload(st,err)){ printf("GATE1 FAIL upload: %s\n",err.c_str()); return 1; }
  printf("upload sec=%.2f bytes0=%.3fGiB bytes1=%.3fGiB\n",
    st.upload_sec,st.bytes_gpu0/1024.0/1024/1024,st.bytes_gpu1/1024.0/1024/1024); fflush(stdout);
  bool over=false; vram_check("post-upload",over);
  if(over){ printf("GATE1 FAIL VRAM>15GB\n"); return 1; }
  // prefill base -> h_last + top
  std::vector<float> h_last;
  std::vector<float> lg11 = pp2::prefill(PA,st,err,&h_last);
  if(lg11.empty()){ printf("GATE1 FAIL prefill11: %s\n",err.c_str()); return 1; }
  if(h_last.size()!=5120){ printf("GATE1 FAIL h_last size=%zu\n",h_last.size()); return 1; }
  const float* last11 = lg11.data()+(PA.size()-1)*248320;
  int top11 = argmax_v(std::vector<float>(last11,last11+248320));
  int t5a[5]; float v5a[5]; top5_of(last11,t5a,v5a);
  printf("prefill11 ms=%.1f top=%d top5=[%d,%d,%d,%d,%d] vals=[%.2f,%.2f,%.2f,%.2f,%.2f]\n",
    st.prefill_ms,top11,t5a[0],t5a[1],t5a[2],t5a[3],t5a[4],v5a[0],v5a[1],v5a[2],v5a[3],v5a[4]); fflush(stdout);
  // teacher: prefill(base+[top11]) -> next
  std::vector<int> ids12=PA; ids12.push_back(top11);
  std::vector<float> lg12 = pp2::prefill(ids12,st,err,nullptr);
  if(lg12.empty()){ printf("GATE1 FAIL prefill12: %s\n",err.c_str()); return 1; }
  const float* last12 = lg12.data()+(ids12.size()-1)*248320;
  std::vector<float> teach(last12,last12+248320);
  int tnext = argmax_v(teach);
  int t5t[5]; float v5t[5]; top5_of(last12,t5t,v5t);
  printf("teacher prefill12 ms=%.1f next=%d top5=[%d,%d,%d,%d,%d]\n",
    st.prefill_ms,tnext,t5t[0],t5t[1],t5t[2],t5t[3],t5t[4]); fflush(stdout);
  // draft singolo
  mtp_gpu::MtpState ms;
  if(!mtp_gpu::init(ms,err)){ printf("GATE1 FAIL mtp_init: %s\n",err.c_str()); return 1; }
  vram_check("mtp-init",over); if(over){ printf("GATE1 FAIL VRAM\n"); return 1; }
  auto t0=std::chrono::high_resolution_clock::now();
  std::vector<float> dlg = mtp_gpu::draft_step(h_last.data(),top11,ms,11,err);
  auto t1=std::chrono::high_resolution_clock::now();
  double dms=std::chrono::duration<double,std::milli>(t1-t0).count();
  if(dlg.empty()){ printf("GATE1 FAIL draft: %s\n",err.c_str()); mtp_gpu::free_state(ms); return 1; }
  double mn,mx,me; bool fin=mtp_gpu::finite_all(dlg,mn,mx,me);
  int dtop=argmax_v(dlg);
  int t5d[5]; float v5d[5]; top5_of(dlg.data(),t5d,v5d);
  int t5teach[5]; float v5teach[5]; top5_of(teach.data(),t5teach,v5teach);
  int ov=topover(t5d,t5teach);
  printf("draft K=1 pos=11 ms=%.1f top=%d val=%.2f mean=%.3f min=%.2f max=%.2f finite=%d\n",
    dms,dtop,dlg[dtop],me,mn,mx,(int)fin);
  printf("draft top5=[%d,%d,%d,%d,%d] overlap_teacher=%d/5\n",t5d[0],t5d[1],t5d[2],t5d[3],t5d[4],ov); fflush(stdout);
  printf("K=1 accept: draft %d vs teacher %d -> %s\n",dtop,tnext,(dtop==tnext?"ACCEPT":"REJECT")); fflush(stdout);
  mtp_gpu::free_state(ms);
  vram_check("post-gate1",over);
  if(!fin||mx>=1e4||mn<=-1e4){ printf("GATE1 FAIL numerico (non-finito o scala assurda)\n"); return 1; }
  if(dtop!=tnext){ printf("GATE1 MISMATCH (valido ma non accettato)\n"); return 2; }
  if(ov<4){ printf("GATE1 WARN overlap %d/5 <4 (argmax ok, rumore top5)\n",ov); fflush(stdout); }
  printf("GATE1 PASS (draft==teacher, overlap %d/5)\n",ov);
  return 0;
}

// GATE2: autoregressivo K=3 con carry MtpState. Ritorna 0 se finiti, 1 altrimenti.
static int gate_k3(){
  printf("\n=== GATE2 MTP autoregressivo K=3 (GPU) ===\n"); fflush(stdout);
  pp2::Stats st; std::string err;
  if(!pp2::upload(st,err)){ printf("GATE2 FAIL upload: %s\n",err.c_str()); return 1; }
  std::vector<float> h_last;
  std::vector<float> lg11 = pp2::prefill(PA,st,err,&h_last);
  if(lg11.empty()){ printf("GATE2 FAIL prefill: %s\n",err.c_str()); return 1; }
  const float* last11 = lg11.data()+(PA.size()-1)*248320;
  int tok = argmax_v(std::vector<float>(last11,last11+248320));
  printf("prefill11 top=%d (tok t+0)\n",tok); fflush(stdout);
  mtp_gpu::MtpState ms;
  if(!mtp_gpu::init(ms,err)){ printf("GATE2 FAIL init: %s\n",err.c_str()); return 1; }
  std::vector<int> dtoks;
  for(int k=0;k<3;++k){
    long pos=11+k;
    auto t0=std::chrono::high_resolution_clock::now();
    std::vector<float> dlg = mtp_gpu::draft_step(h_last.data(),tok,ms,pos,err);
    auto t1=std::chrono::high_resolution_clock::now();
    double dms=std::chrono::duration<double,std::milli>(t1-t0).count();
    if(dlg.empty()){ printf("GATE2 FAIL draft k=%d: %s\n",k,err.c_str()); mtp_gpu::free_state(ms); return 1; }
    double mn,mx,me; bool fin=mtp_gpu::finite_all(dlg,mn,mx,me);
    int dt=argmax_v(dlg);
    printf("draft t+%d pos=%ld ms=%.1f top=%d val=%.2f mean=%.3f min=%.2f max=%.2f finite=%d\n",
      k+1,pos,dms,dt,dlg[dt],me,mn,mx,(int)fin); fflush(stdout);
    if(!fin||mx>=1e4||mn<=-1e4){ printf("GATE2 FAIL numerico k=%d\n",k); mtp_gpu::free_state(ms); return 1; }
    dtoks.push_back(dt);
    tok=dt;
  }
  mtp_gpu::free_state(ms);
  bool over=false; vram_check("post-gate2",over);
  if(over){ printf("GATE2 FAIL VRAM\n"); return 1; }
  printf("GATE2 PASS 3 draft token finiti: [%d,%d,%d]\n",dtoks[0],dtoks[1],dtoks[2]);
  return 0;
}

// Sweep K=1..4 su un prompt. Ritorna 0 se numeri finiti (anche con accept bassa), 1 su errore.
static int sweep_one(const std::vector<int>& P, const char* pname){
  printf("\n=== SWEEP %s len=%zu K=1..4 ===\n",pname,P.size()); fflush(stdout);
  pp2::Stats st; std::string err;
  if(!pp2::upload(st,err)){ printf("SWEEP %s FAIL upload: %s\n",pname,err.c_str()); return 1; }
  std::vector<float> h_last;
  std::vector<float> lg0 = pp2::prefill(P,st,err,&h_last);
  if(lg0.empty()){ printf("SWEEP %s FAIL prefill0: %s\n",pname,err.c_str()); return 1; }
  const float* l0 = lg0.data()+(P.size()-1)*248320;
  int tok0 = argmax_v(std::vector<float>(l0,l0+248320));
  printf("prefill%zu top=%d ms=%.1f (tok t+0)\n",P.size(),tok0,st.prefill_ms); fflush(stdout);
  printf("%-4s %-22s %-8s %-10s %-12s %-12s %-10s\n","K","drafts","accept","rate","draft_ms","verify_ms","eff_tok/s");
  for(int K=1;K<=4;++K){
    // K draft autoregressivi
    mtp_gpu::MtpState ms;
    if(!mtp_gpu::init(ms,err)){ printf("SWEEP FAIL init: %s\n",err.c_str()); return 1; }
    std::vector<int> D; double draft_ms=0; bool bad=false;
    int tok=tok0;
    for(int k=0;k<K;++k){
      auto t0=std::chrono::high_resolution_clock::now();
      std::vector<float> dlg = mtp_gpu::draft_step(h_last.data(),tok,ms,(long)(P.size()+k),err);
      auto t1=std::chrono::high_resolution_clock::now();
      draft_ms+=std::chrono::duration<double,std::milli>(t1-t0).count();
      if(dlg.empty()){ printf("SWEEP K=%d FAIL draft k=%d: %s\n",K,k,err.c_str()); bad=true; break; }
      double mn,mx,me; bool fin=mtp_gpu::finite_all(dlg,mn,mx,me);
      if(!fin||mx>=1e4||mn<=-1e4){ printf("SWEEP K=%d FAIL numerico k=%d\n",K,k); bad=true; break; }
      int dt=argmax_v(dlg); D.push_back(dt); tok=dt;
    }
    mtp_gpu::free_state(ms);
    if(bad) return 1;
    // verify: prefill(P + [tok0] + D); D[j] e` il token in posizione L+1+j,
    // quindi si confronta con argmax(logits[L+j]) (predizione dopo P+tok0+D[..j-1]).
    std::vector<int> ext=P; ext.push_back(tok0); for(int d:D) ext.push_back(d);
    auto t0=std::chrono::high_resolution_clock::now();
    std::vector<float> lgv = pp2::prefill(ext,st,err,nullptr);
    auto t1=std::chrono::high_resolution_clock::now();
    double verify_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    if(lgv.empty()){ printf("SWEEP K=%d FAIL verify: %s\n",K,err.c_str()); return 1; }
    int L=(int)P.size(), acc=0;
    for(int j=0;j<K;++j){
      const float* lj = lgv.data()+(size_t)(L+j)*248320;
      int tt=argmax_v(std::vector<float>(lj,lj+248320));
      if(tt==D[j]) acc++; else break;
    }
    double rate=(double)acc/(double)K;
    double eff=(double)(acc+1)/((draft_ms+verify_ms)/1000.0); // +1 bonus token
    char db[64]; snprintf(db,64,"["); for(int j=0;j<K;++j){ char b[16]; snprintf(b,16,"%d%s",D[j],j+1<K?",":""); strcat(db,b);} strcat(db,"]");
    printf("%-4d %-22s %d/%d      %-10.2f %-12.1f %-12.1f %-10.2f\n",K,db,acc,K,rate,draft_ms,verify_ms,eff); fflush(stdout);
  }
  bool over=false; vram_check("post-sweep",over);
  if(over){ printf("SWEEP %s FAIL VRAM\n",pname); return 1; }
  printf("SWEEP %s DONE (baseline non-spec 1.53 tok/s)\n",pname);
  return 0;
}

int main(int argc,char** argv){
  std::string only=argc>1?argv[1]:"all";
  printf("MTP_GPU test only=%s\n",only.c_str()); fflush(stdout);
  bool do_k1=(only=="all"||only=="k1"), do_k3=(only=="all"||only=="k3"),
       do_sa=(only=="all"||only=="sweep"||only=="sweepA"),
       do_sb=(only=="all"||only=="sweep"||only=="sweepB");
  if(do_k1){ int r=gate_k1(); if(r!=0){ printf("STOP al primo fallimento (k1 rc=%d)\n",r); return r; } if(only=="k1") return 0; }
  if(do_k3){ int r=gate_k3(); if(r!=0){ printf("STOP al primo fallimento (k3 rc=%d)\n",r); return r; } if(only=="k3") return 0; }
  if(do_sa){ int r=sweep_one(PA,"P-A11"); if(r!=0){ printf("STOP (sweepA rc=%d)\n",r); return r; } if(only=="sweepA") return 0; }
  if(do_sb){ int r=sweep_one(PB,"P-B13"); if(r!=0){ printf("STOP (sweepB rc=%d)\n",r); return r; } if(only=="sweepB") return 0; }
  printf("ALL DONE\n");
  return 0;
}
