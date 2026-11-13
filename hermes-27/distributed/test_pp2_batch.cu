// P5 batch concorrente PP2: GATE1 correttezza N=2, GATE2 throughput N=2,4.
// Prompt A = baseline P2/P4 11tok [6205,161421,303,220,17,1374,9911,1147,27053,6059,13].
// Prompt B = reverse(A) [13,6059,27053,1147,9911,1374,17,220,303,161421,6205] (stessa lunghezza 11,
// contesto diverso, pos allineate per batch_step stessa pos).
// Prompt C = rotl(A,3), D = rotl(A,6) per N=4 (documentati in P5_BATCH.md).
// Uso: test_pp2_batch [g1|g2|all] (default all, ferma al gate fallito)
#include "pp2_batch.hpp"
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
static void maxerr_vs(const std::vector<float>& a, const std::vector<float>& b,
                      double& mx, size_t& im, double& me){
  mx=0; me=0; im=0;
  for(size_t i=0;i<a.size();++i){
    double e=fabs((double)a[i]-b[i]);
    if(e>mx){mx=e;im=i;} me+=e;
  }
  me/=a.size();
}

// single-stream ref: warmup prompt + 4 greedy, ritorna logits dei 4 step + gen ids
static bool single_ref(const std::vector<int>& prompt,
                       std::vector<std::vector<float>>& ref_logits,
                       std::vector<int>& gen,
                       double& wall_ms, std::string& err){
  ref_logits.clear(); gen.clear();
  pp2::DecodeState ds;
  if(!pp2::decode_init(ds,err)) return false;
  auto t0=std::chrono::high_resolution_clock::now();
  std::vector<float> last;
  for(size_t t=0;t<prompt.size();++t){
    last=pp2::decode_step(prompt[t],ds,(long)t,err);
    if(last.empty()){ pp2::decode_free(ds); return false; }
  }
  int nxt=argmax_v(last);
  for(int k=0;k<4;++k){
    long pos=(long)(prompt.size()+k);
    std::vector<float> dec=pp2::decode_step(nxt,ds,pos,err);
    if(dec.empty()){ pp2::decode_free(ds); return false; }
    ref_logits.push_back(dec);
    gen.push_back(nxt);
    nxt=argmax_v(dec);
  }
  auto t1=std::chrono::high_resolution_clock::now();
  wall_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
  pp2::decode_free(ds);
  return true;
}

int main(int argc,char** argv){
  std::string only=argc>1?argv[1]:"all";
  printf("P5 PP2 batch N-richeste only=%s\n",only.c_str()); fflush(stdout);
  bool do_g1=(only=="all"||only=="g1"), do_g2=(only=="all"||only=="g2");

  const std::vector<int> PA={6205,161421,303,220,17,1374,9911,1147,27053,6059,13};
  const std::vector<int> PB={13,6059,27053,1147,9911,1374,17,220,303,161421,6205};
  const std::vector<int> PC={220,17,1374,9911,1147,27053,6059,13,6205,161421,303};
  const std::vector<int> PD={9911,1147,27053,6059,13,6205,161421,303,220,17,1374};

  pp2::Stats st; std::string err;
  if(!pp2::upload(st,err)){ printf("UPLOAD FAIL %s\n",err.c_str()); return 1; }
  printf("UPLOAD ok sec=%.2f bytes0=%.3fGiB bytes1=%.3fGiB\n",
    st.upload_sec, st.bytes_gpu0/1024.0/1024/1024, st.bytes_gpu1/1024.0/1024/1024); fflush(stdout);

  size_t u0b=0,u1b=0; pp2batch::vram_used(u0b,u1b);
  printf("VRAM post-upload used0=%zu MiB used1=%zu MiB\n",u0b,u1b); fflush(stdout);

  // ---- GATE1 ----
  if(do_g1){
    printf("\n=== GATE1 N=2 correttezza (seq loop, bit-exact target <1e-4) ===\n"); fflush(stdout);
    std::vector<std::vector<float>> refA, refB;
    std::vector<int> genA, genB;
    double wallA=0, wallB=0;
    if(!single_ref(PA,refA,genA,wallA,err)){ printf("REF A FAIL %s\n",err.c_str()); return 1; }
    if(!single_ref(PB,refB,genB,wallB,err)){ printf("REF B FAIL %s\n",err.c_str()); return 1; }
    printf("REF A gen=[%d,%d,%d,%d] wall15=%.1fms (11 warmup+4 greedy)\n",genA[0],genA[1],genA[2],genA[3],wallA); fflush(stdout);
    printf("REF B gen=[%d,%d,%d,%d] wall15=%.1fms\n",genB[0],genB[1],genB[2],genB[3],wallB); fflush(stdout);
    // baseline single per-tok (15 step wall/15)
    printf("REF single per-step A=%.1fms B=%.1fms (atteso ~654ms)\n",wallA/15.0,wallB/15.0); fflush(stdout);

    pp2batch::BatchState bs;
    if(!pp2batch::batch_init(bs,2,err)){ printf("BATCH_INIT N=2 FAIL %s\n",err.c_str()); return 1; }
    size_t u0a=0,u1a=0; pp2batch::vram_used(u0a,u1a);
    printf("VRAM batch N=2 used0=%zu MiB used1=%zu MiB (delta +%zd/+%zd vs post-upload)\n",
      u0a,u1a,(long)u0a-(long)u0b,(long)u1a-(long)u1b); fflush(stdout);
    if(u0a>15360||u1a>15360){ printf("GATE1 FAIL VRAM>15GB\n"); pp2batch::batch_free(bs); return 1; }

    // warmup 11 batch seq
    for(int t=0;t<11;++t){
      std::vector<int> toks={PA[t],PB[t]};
      auto outs=pp2batch::batch_step(toks,bs,(long)t,err);
      if(outs.empty()){ printf("WARMUP t=%d FAIL %s\n",t,err.c_str()); pp2batch::batch_free(bs); return 1; }
      if(!finite_all(outs[0])||!finite_all(outs[1])){ printf("WARMUP t=%d NaN\n",t); pp2batch::batch_free(bs); return 1; }
    }
    // 4 greedy batch seq, confronto vs ref
    std::vector<int> cur={genA[0],genB[0]}; // placeholder, in realta` tok12 = argmax warmup last
    // Ricalcola tok12 da batch warmup last? Usiamo ref tok12 come starting point per confronto diretto:
    // i ref gen[0] sono i tok12 (argmax dopo warmup). Batch warmup dovrebbe dare stessi tok12.
    // Verifichiamo con uno step extra? No: procediamo con cur = [refA tok12, refB tok12] e pos=11.
    // Se batch warmup diverge, lo step pos11 divergerebbe subito (fail onesto).
    cur[0]=genA[0]; cur[1]=genB[0];
    // Nota: genA[0] e` il token generato a pos11 (primo greedy). Per confronto logits,
    // dobbiamo confrontare batch logits pos11 (generati da tok12) vs refA[0] (stesso input tok12).
    // Ma cur per pos11 deve essere tok12, che e` genA[0]? No: genA[0] e` l'output di pos11,
    // l'input di pos11 e` tok12 = argmax(warmup). In single_ref, nxt iniziale = argmax(warmup),
    // e refA[0] = decode(tok12). Quindi input pos11 = tok12, non genA[0].
    // Dobbiamo recuperare tok12: e` l'input che ha prodotto refA[0]. Lo otteniamo rifacendo warmup single?
    // Semplificazione corretta: rieseguiamo warmup single per ottenere tok12, oppure usiamo
    // il fatto che single_ref warmup last argmax = tok12. Ricaviamo tok12 come: rifacciamo warmup?
    // Piu` semplice e rigoroso: batch warmup last logits -> argmax = tok12_batch, single tok12 = ?
    // Per ora usiamo i tok12 dai ref: dobbiamo salvarli. Ripetiamo warmup single leggero per tok12.
    {
      // Recupera tok12 single (11 warmup, no greedy) per A e B
      pp2::DecodeState dA, dB;
      pp2::decode_init(dA,err); pp2::decode_init(dB,err);
      std::vector<float> lA, lB;
      for(int t=0;t<11;++t){ lA=pp2::decode_step(PA[t],dA,(long)t,err); lB=pp2::decode_step(PB[t],dB,(long)t,err); }
      int tok12A=argmax_v(lA), tok12B=argmax_v(lB);
      pp2::decode_free(dA); pp2::decode_free(dB);
      printf("TOK12 single A=%d B=%d | ref gen0 (pos11 out) A=%d B=%d\n",tok12A,tok12B,genA[0],genB[0]); fflush(stdout);
      cur[0]=tok12A; cur[1]=tok12B;
    }
    bool pass=true;
    std::vector<int> genBatchA, genBatchB;
    auto tg0=std::chrono::high_resolution_clock::now();
    for(int k=0;k<4;++k){
      long pos=11+k;
      // input cur = [tok_pos_A, tok_pos_B]
      auto outs=pp2batch::batch_step(cur,bs,pos,err);
      if(outs.empty()){ printf("BATCH step k=%d pos=%ld FAIL %s\n",k,pos,err.c_str()); pp2batch::batch_free(bs); return 1; }
      double mxA,meA,mxB,meB; size_t imA,imB;
      maxerr_vs(outs[0],refA[k],mxA,imA,meA);
      maxerr_vs(outs[1],refB[k],mxB,imB,meB);
      // Attenzione: refA[k] e` decode(tok_input_k) dove tok_input_0=tok12, tok_input_{k+1}=genA[k].
      // Batch outs[k] usa stesso schema se cur evolve con batch argmax (che deve coincidere).
      // Al primo k input coincide per costruzione; ai successivi coincide solo se arg SAME (verificato).
      int adA=argmax_v(outs[0]), arA=argmax_v(refA[k]);
      int adB=argmax_v(outs[1]), arB=argmax_v(refB[k]);
      printf("GATE1 k=%d pos=%ld A: maxerr=%.6f at %zu mean=%.7f arg batch=%d ref=%d %s | B: maxerr=%.6f at %zu mean=%.7f arg batch=%d ref=%d %s\n",
        k,pos,mxA,imA,meA,adA,arA,adA==arA?"SAME":"DIFF",mxB,imB,meB,adB,arB,adB==arB?"SAME":"DIFF"); fflush(stdout);
      bool okA=(mxA<1e-4 && adA==arA), okB=(mxB<1e-4 && adB==arB);
      printf("GATE1 k=%d A %s B %s (soglia 1e-4 bit-exact)\n",k,okA?"PASS":"FAIL",okB?"PASS":"FAIL"); fflush(stdout);
      if(!okA||!okB) pass=false;
      genBatchA.push_back(adA); genBatchB.push_back(adB);
      // next input = batch argmax (se SAME coincide con ref, confronto resta allineato)
      cur[0]=adA; cur[1]=adB;
      // se DIFF, per onesta` continuiamo con batch path (non forziamo teacher), ma gate gia` FAIL
    }
    auto tg1=std::chrono::high_resolution_clock::now();
    double wall_batch_ms=std::chrono::duration<double,std::milli>(tg1-tg0).count();
    printf("BATCH N=2 4-step wall=%.1fms (8 tok) agg=%.3f tok/s per-req=%.3f tok/s\n",
      wall_batch_ms, 8000.0/wall_batch_ms, 4000.0/wall_batch_ms); fflush(stdout);
    printf("GEN batch A=[%d,%d,%d,%d] vs ref [%d,%d,%d,%d]\n",genBatchA[0],genBatchA[1],genBatchA[2],genBatchA[3],genA[0],genA[1],genA[2],genA[3]); fflush(stdout);
    printf("GEN batch B=[%d,%d,%d,%d] vs ref [%d,%d,%d,%d]\n",genBatchB[0],genBatchB[1],genBatchB[2],genBatchB[3],genB[0],genB[1],genB[2],genB[3]); fflush(stdout);

    // parallel 1-step sanity (stesso input pos11 tok12, deve dare stessi logits del seq)
    {
      pp2batch::BatchState bs2;
      if(!pp2batch::batch_init(bs2,2,err)){ printf("BATCH2 INIT FAIL %s\n",err.c_str()); pp2batch::batch_free(bs); return 1; }
      for(int t=0;t<11;++t){
        std::vector<int> toks={PA[t],PB[t]};
        std::vector<long> poss={(long)t,(long)t};
        auto outs=pp2batch::batch_step_parallel(toks,bs2,poss,err);
        if(outs.empty()){ printf("PAR warmup t=%d FAIL %s\n",t,err.c_str()); pp2batch::batch_free(bs2); pp2batch::batch_free(bs); return 1; }
      }
      // tok12 recuperati sopra? Ricalcola veloce: usa cur iniziale tok12 (pos11 input)
      // Per semplicita` riusa tok12 da prima (cur originario). Ma cur e` evoluto; ricostruisci:
      pp2::DecodeState dA2, dB2;
      pp2::decode_init(dA2,err); pp2::decode_init(dB2,err);
      std::vector<float> lA2,lB2;
      for(int t=0;t<11;++t){ lA2=pp2::decode_step(PA[t],dA2,(long)t,err); lB2=pp2::decode_step(PB[t],dB2,(long)t,err); }
      int t12A=argmax_v(lA2), t12B=argmax_v(lB2);
      pp2::decode_free(dA2); pp2::decode_free(dB2);
      std::vector<int> toks12={t12A,t12B};
      std::vector<long> poss12={11,11};
      auto outs_par=pp2batch::batch_step_parallel(toks12,bs2,poss12,err);
      if(outs_par.empty()){ printf("PAR step pos11 FAIL %s\n",err.c_str()); pp2batch::batch_free(bs2); pp2batch::batch_free(bs); return 1; }
      double mxA,meA,mxB,meB; size_t imA,imB;
      maxerr_vs(outs_par[0],refA[0],mxA,imA,meA);
      maxerr_vs(outs_par[1],refB[0],mxB,imB,meB);
      int adA=argmax_v(outs_par[0]), arA=argmax_v(refA[0]);
      int adB=argmax_v(outs_par[1]), arB=argmax_v(refB[0]);
      printf("PAR sanity pos11 A maxerr=%.6f arg %d vs %d %s | B maxerr=%.6f arg %d vs %d %s\n",
        mxA,adA,arA,adA==arA?"SAME":"DIFF",mxB,adB,arB,adB==arB?"SAME":"DIFF"); fflush(stdout);
      bool okp=(mxA<1e-4&&adA==arA&&mxB<1e-4&&adB==arB);
      printf("PAR sanity %s\n",okp?"PASS":"FAIL"); fflush(stdout);
      if(!okp) pass=false;
      pp2batch::batch_free(bs2);
    }

    size_t u0c=0,u1c=0; pp2batch::vram_used(u0c,u1c);
    printf("VRAM prima free batch used0=%zu MiB used1=%zu MiB (delta vs post-upload %+zd/%+zd)\n",
      u0c,u1c,(long)u0c-(long)u0b,(long)u1c-(long)u1b); fflush(stdout);
    pp2batch::batch_free(bs);
    size_t u0d=0,u1d=0; pp2batch::vram_used(u0d,u1d);
    printf("VRAM dopo free batch used0=%zu MiB used1=%zu MiB (leak check vs post-upload %+zd/%+zd)\n",
      u0d,u1d,(long)u0d-(long)u0b,(long)u1d-(long)u1b); fflush(stdout);
    printf("GATE1 %s\n",pass?"PASS":"FAIL"); fflush(stdout);
    if(only=="g1") return pass?0:1;
    if(!pass) return 1;
  }

  // ---- GATE2 ----
  if(do_g2){
    printf("\n=== GATE2 throughput N=1/2/4 (agg vs baseline 1.53 tok/s) ===\n"); fflush(stdout);
    std::vector<std::vector<int>> prompts={PA,PB,PC,PD};
    const char* pname[4]={"A","B","C","D"};
    for(int N : {1,2,4}){
      for(int mode=0; mode<2; ++mode){
        bool parallel=(mode==1);
        if(N==1 && parallel) continue; // N=1 parallel == seq, skip
        pp2batch::BatchState bs;
        if(!pp2batch::batch_init(bs,N,err)){ printf("G2 INIT N=%d FAIL %s\n",N,err.c_str()); return 2; }
        size_t uv0=0,uv1=0; pp2batch::vram_used(uv0,uv1);
        // warmup 11
        for(int t=0;t<11;++t){
          std::vector<int> toks((size_t)N);
          for(int i=0;i<N;++i) toks[i]=prompts[i][t];
          std::vector<std::vector<float>> outs;
          if(parallel){
            std::vector<long> poss((size_t)N,(long)t);
            outs=pp2batch::batch_step_parallel(toks,bs,poss,err);
          } else {
            outs=pp2batch::batch_step(toks,bs,(long)t,err);
          }
          if(outs.empty()){ printf("G2 warmup N=%d %s t=%d FAIL %s\n",N,parallel?"par":"seq",t,err.c_str()); pp2batch::batch_free(bs); return 2; }
        }
        // tok12 per-request da warmup batch last? Per throughput usiamo greedy batch (self-consistent):
        // prendi ultimo warmup outs argmax come input pos11
        std::vector<int> cur((size_t)N);
        {
          // Per semplicita` e onesta`: un batch_step pos10 gia` fatto; rifacciamo un extra step per tok12?
          // No: usiamo argmax di un passo dedicato? Facciamo: ultimo warmup t=10 ha prodotto logits,
          // ma non salvati. Rifacciamo warmup last argmax via single veloce? Piu` semplice:
          // cur = argmax di un nuovo batch_step con token prompt[10]? No, duplicherebbe pos10.
          // Approccio throughput puro: misura solo 4 greedy step a partire da tok12 single (noto).
          // Calcoliamo tok12 single per ogni prompt (11 decode_step, costo incluso ma non misurato).
          for(int i=0;i<N;++i){
            pp2::DecodeState tmp;
            pp2::decode_init(tmp,err);
            std::vector<float> last;
            for(int t=0;t<11;++t) last=pp2::decode_step(prompts[i][t],tmp,(long)t,err);
            cur[i]=argmax_v(last);
            pp2::decode_free(tmp);
          }
        }
        // Re-inizializza batch per misura pulita (stati zero + warmup 11 dentro misura? No:
        // vogliamo misurare solo decode steady, quindi warmup gia` fatto sopra su bs.
        // Ma cur calcolato da stati single diversi; per coerenza stati, dobbiamo avanzare bs di 0?
        // Soluzione onesta: bs ha gia` stati dopo warmup 11 con prompt tokens.
        // I suoi S/CS/KV corrispondono a prompt, esattamente come i tmp single.
        // Quindi cur (tok12 single) e` valido come input pos11 per bs (se batch warmup corretto,
        // tok12_batch==tok12_single; verifichiamo al primo step con arg check, ma per throughput
        // non confrontiamo logits, solo tempi + finite).
        // Se batch warmup fosse divergente, i tempi restano validi ma la correttezza e` gia` coperta da GATE1.
        auto t0=std::chrono::high_resolution_clock::now();
        bool ok=true;
        for(int k=0;k<4;++k){
          long pos=11+k;
          std::vector<std::vector<float>> outs;
          if(parallel){
            std::vector<long> poss((size_t)N,pos);
            outs=pp2batch::batch_step_parallel(cur,bs,poss,err);
          } else {
            outs=pp2batch::batch_step(cur,bs,pos,err);
          }
          if(outs.empty()){ printf("G2 step N=%d %s k=%d FAIL %s\n",N,parallel?"par":"seq",k,err.c_str()); pp2batch::batch_free(bs); return 2; }
          for(int i=0;i<N;++i){
            if(!finite_all(outs[i])){ printf("G2 NaN N=%d %s k=%d req=%d\n",N,parallel?"par":"seq",k,i); ok=false; }
            cur[i]=argmax_v(outs[i]);
          }
          if(!ok){ pp2batch::batch_free(bs); return 2; }
        }
        auto t1=std::chrono::high_resolution_clock::now();
        double wall_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
        double agg_tps=(N*4)/(wall_ms/1000.0);
        double ms_per_tok_agg=wall_ms/(N*4);
        double speedup_vs_single=agg_tps/1.53;
        size_t uv0b=0,uv1b=0; pp2batch::vram_used(uv0b,uv1b);
        printf("G2 N=%d %s 4-step wall=%.1fmsNt=%d agg=%.3f tok/s (vs 1.53 x%.2f) ms/tok_agg=%.1f vram %zu/%zu MiB prompts=",
          N,parallel?"par":"seq",wall_ms,N*4,agg_tps,speedup_vs_single,ms_per_tok_agg,uv0b,uv1b);
        for(int i=0;i<N;++i) printf("%s%s",pname[i],i+1<N?",":"");
        printf("\n"); fflush(stdout);
        // admission info
        {
          std::string det;
          bool adm=pp2batch::admit_check(N,1024,det);
          printf(" admit_check %s -> %s\n",det.c_str(),adm?"ADMIT":"REJECT"); fflush(stdout);
        }
        pp2batch::batch_free(bs);
        size_t uv0c=0,uv1c=0; pp2batch::vram_used(uv0c,uv1c);
        printf(" G2 N=%d %s post-free vram %zu/%zu MiB (delta vs pre %+zd/%+zd, leak %s)\n",
          N,parallel?"par":"seq",uv0c,uv1c,(long)uv0c-(long)uv0,(long)uv1c-(long)uv1,
          (labs((long)uv0c-(long)uv0)<50 && labs((long)uv1c-(long)uv1)<50)?"NO":"CHECK"); fflush(stdout);
      }
    }
    // admit caps extra (16@1K OK, 17@1K REJECT, 8@8K OK, 2@32K OK, 4@32K REJECT)
    {
      std::string d; bool a;
      a=pp2batch::admit_check(16,1024,d); printf("ADMIT 16@1K %s %s\n",d.c_str(),a?"ADMIT":"REJECT");
      a=pp2batch::admit_check(17,1024,d); printf("ADMIT 17@1K %s %s\n",d.c_str(),a?"ADMIT":"REJECT");
      a=pp2batch::admit_check(8,8192,d);  printf("ADMIT 8@8K %s %s\n",d.c_str(),a?"ADMIT":"REJECT");
      a=pp2batch::admit_check(2,32768,d); printf("ADMIT 2@32K %s %s\n",d.c_str(),a?"ADMIT":"REJECT");
      a=pp2batch::admit_check(4,32768,d); printf("ADMIT 4@32K %s %s\n",d.c_str(),a?"ADMIT":"REJECT");
      fflush(stdout);
    }
    printf("GATE2 DONE (numeri sopra, baseline 1.53 tok/s)\n"); fflush(stdout);
    if(only=="g2") return 0;
  }

  printf("P5 DONE\n"); fflush(stdout);
  return 0;
}
