#include "hermes/model_fwd.hpp"
#include "hermes/gguf.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>
#ifdef __linux__
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif
namespace hermes {
static inline float silu(float v){ return v/(1.f+expf(-v)); }
static inline float softplus(float v){ return v>20.f?v:logf(1.f+expf(v)); } // log(1+exp(v)), NOT exp(-v)
float ModelFwd::bf2f(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }
bool ModelFwd::open(const std::string& p){
  path=p;
  GgufFile gf; gf.open(p, true);
  data_off=gf.data_offset;
  for(auto& t: gf.tensors){
    int ty=-1;
    if(t.type==GgmlType::F32) ty=0; else if(t.type==GgmlType::BF16) ty=30;
    else if(t.type==GgmlType::NVFP4) ty=40; else throw std::runtime_error("unexpected type "+t.name);
    int ne1 = t.n_dims>1 ? (int)t.ne[1] : 1;
    tensors.push_back({t.name,(int)t.ne[0],ne1,ty,t.offset});
  }
#ifdef __linux__
  fd=::open(p.c_str(),O_RDONLY); if(fd<0) return false;
  struct stat st{}; fstat(fd,&st); len=(uint64_t)st.st_size;
  void* m=::mmap(nullptr,len,PROT_READ,MAP_PRIVATE,fd,0);
  if(m==MAP_FAILED) return false; base=(const uint8_t*)m;
#else
  return false;
#endif
  return true;
}
void ModelFwd::close(){
#ifdef __linux__
  if(base){ ::munmap((void*)base,len); base=nullptr; }
  if(fd>=0){ ::close(fd); fd=-1; }
#endif
}
const ModelFwd::TI* ModelFwd::find(const std::string& n) const{
  for(auto& t: tensors) if(t.name==n) return &t; return nullptr;
}
const float* ModelFwd::f32(const std::string& n) const{
  auto t=find(n); if(!t||t->type!=0) throw std::runtime_error("f32 "+n); return (const float*)(base+data_off+t->off);
}
const uint16_t* ModelFwd::bf16(const std::string& n) const{
  auto t=find(n); if(!t||t->type!=30) throw std::runtime_error("bf16 "+n); return (const uint16_t*)(base+data_off+t->off);
}
const uint8_t* ModelFwd::nvfp4(const std::string& n) const{
  auto t=find(n); if(!t||t->type!=40) throw std::runtime_error("nvfp4 "+n); return base+data_off+t->off;
}
void ModelFwd::deq_row(const uint8_t* row, float* out, int ne0){
  for(int b=0;b<ne0/64;++b) nvfp4_block_dequant(row+b*36, out+b*64);
}
void ModelFwd::gemv(const float* W, const float* x, float* y, int rows, int cols){
  for(int i=0;i<rows;++i){ double a=0; for(int k=0;k<cols;++k) a+=double(W[i*cols+k])*double(x[k]); y[i]=(float)a; }
}
static void linear_batch(const float* W, const float* X, float* Y, int T, int out, int in){
  for(int t=0;t<T;++t) for(int i=0;i<out;++i){ double a=0;
    for(int k=0;k<in;++k) a+=double(X[(size_t)t*in+k])*double(W[(size_t)i*in+k]);
    Y[(size_t)t*out+i]=(float)a; }
}
void ModelFwd::linear_nvfp4_batch(const std::string& wname, const float* X, float* Y, int T) const{
  auto t=find(wname);
  linear_nvfp4_rows(wname, X, Y, T, 0, t->ne1);
}
void ModelFwd::linear_nvfp4_rows(const std::string& wname, const float* X, float* Y, int T, int r0, int r1) const{
  auto t=find(wname); int Cc=t->ne0, R=r1-r0;
  std::vector<float> W((size_t)R*Cc);
  if(t->type==40){
    int nb=Cc/64; const uint8_t* rows=nvfp4(wname);
    for(int i=0;i<R;++i) deq_row(rows+(size_t)(r0+i)*nb*36, W.data()+(size_t)i*Cc, Cc);
  } else if(t->type==30){
    const uint16_t* rows=bf16(wname);
    for(int i=0;i<R;++i) for(int k=0;k<Cc;++k) W[(size_t)i*Cc+k]=bf2f(rows[(size_t)(r0+i)*Cc+k]);
  } else throw std::runtime_error("linear_nvfp4_rows: bad type "+wname);
  linear_batch(W.data(), X, Y, T, R, Cc);
}
// NOTE 2026-11-13: GGUF stores DIRECT ~1-scale norm weights (means 0.5..2.3, none ~0),
// and llama.cpp build_norm does rms(x)*w with no +1. So always direct multiply;
// plus_one is kept in the signature (callers unaffected) but ignored.
void ModelFwd::rms(const float* x, const float* w, float* y, int n, bool plus_one) const{
  (void)plus_one;
  double ss=0; for(int i=0;i<n;++i) ss+=double(x[i])*double(x[i]);
  float inv=1.f/sqrtf(float(ss/n)+1e-6f);
  for(int i=0;i<n;++i) y[i]=x[i]*inv*w[i];
}
// q,k l2-normalize per head + scale q
static void qk_norm_scale(float* q, float* k, int H, int K, float scale){
  for(int h=0;h<H;++h){
    double nq=0,nk=0;
    for(int d=0;d<K;++d){ float* qq=q+h*K+d; float* kk=k+h*K+d; nq+=double(*qq)*double(*qq); nk+=double(*kk)*double(*kk); }
    nq=1.0/sqrt(nq+1e-6); nk=1.0/sqrt(nk+1e-6);
    for(int d=0;d<K;++d){ q[h*K+d]=(float)(double(q[h*K+d])*nq*double(scale)); k[h*K+d]=(float)(double(k[h*K+d])*nk); }
  }
}
std::vector<float> ModelFwd::prefill(const std::vector<int>& ids, float* h_last_out) const{
  int T=(int)ids.size();
  std::vector<float> x((size_t)T*HID), h((size_t)T*HID);
  auto emb=bf16("token_embd.weight");
  for(int t=0;t<T;++t) for(int i=0;i<HID;++i) x[(size_t)t*HID+i]=bf2f(emb[(size_t)ids[t]*HID+i]);
  std::vector<float> coss((size_t)T*64), sins((size_t)T*64);
  for(int t=0;t<T;++t) for(int i=0;i<32;++i){
    float f=1.f/powf(10000000.f,(float)(2*i)/64.f), a=(float)t*f;
    coss[(size_t)t*64+i]=cosf(a); coss[(size_t)t*64+32+i]=cosf(a);
    sins[(size_t)t*64+i]=sinf(a); sins[(size_t)t*64+32+i]=sinf(a);
  }
  auto rope_apply=[&](float* QK, int H, int t){ // split-half on first 64 dims
    float* co=&coss[(size_t)t*64], *si=&sins[(size_t)t*64];
    for(int hd_=0;hd_<H;++hd_) for(int i=0;i<32;++i){
      float a=QK[hd_*256+i], b=QK[hd_*256+32+i];
      QK[hd_*256+i]=a*co[i]-b*si[i]; QK[hd_*256+32+i]=a*si[i]+b*co[i];
    }
  };
  std::vector<float> qkv((size_t)T*C), o((size_t)T*HV*VHD);
  std::vector<float> S((size_t)HV*KHD*VHD, 0.f);
  std::vector<float> dbg_cv, dbg_beta, dbg_gl; // blk0 intermediates for E1/E2 bisect
  for(int li=0;li<NL;++li){
    char Pb[16]; snprintf(Pb,16,"blk.%d.",li); std::string P=Pb;
    for(int t=0;t<T;++t) rms(&x[(size_t)t*HID], f32(P+"attn_norm.weight"), &h[(size_t)t*HID], HID, true);
    std::vector<float> core((size_t)T*HID, 0.f);
    if(li%4==3){
      std::vector<float> qqkv((size_t)T*12288), kk((size_t)T*1024), vv((size_t)T*1024);
      linear_nvfp4_batch(P+"attn_q.weight", h.data(), qqkv.data(), T);
      linear_nvfp4_batch(P+"attn_k.weight", h.data(), kk.data(), T);
      linear_nvfp4_batch(P+"attn_v.weight", h.data(), vv.data(), T);
      auto qn=f32(P+"attn_q_norm.weight"), kn=f32(P+"attn_k_norm.weight");
      std::vector<float> Q((size_t)T*NQ*HD), K_((size_t)T*NKV*HD), V((size_t)T*NKV*HD), G((size_t)T*6144);
      for(int t=0;t<T;++t){
        float* qr=&qqkv[(size_t)t*12288];
        for(int hd_=0;hd_<NQ;++hd_){ double ss=0; for(int d=0;d<HD;++d){float v=qr[hd_*512+d]; ss+=double(v)*v;}
          float inv=1.f/sqrtf(float(ss/HD)+1e-6f);
          for(int d=0;d<HD;++d){ Q[((size_t)t*NQ+hd_)*HD+d]=qr[hd_*512+d]*inv*qn[d]; G[((size_t)t*NQ+hd_)*HD+d]=qr[hd_*512+256+d]; } }
        float* kr=&kk[(size_t)t*1024];
        for(int hd_=0;hd_<NKV;++hd_){ double ss=0; for(int d=0;d<HD;++d){float v=kr[hd_*HD+d]; ss+=double(v)*v;}
          float inv=1.f/sqrtf(float(ss/HD)+1e-6f);
          for(int d=0;d<HD;++d) K_[((size_t)t*NKV+hd_)*HD+d]=kr[hd_*HD+d]*inv*kn[d]; }
        for(int i=0;i<1024;++i) V[(size_t)t*1024+i]=vv[(size_t)t*1024+i];
        rope_apply(&Q[(size_t)t*NQ*HD], NQ, t);
        rope_apply(&K_[(size_t)t*NKV*HD], NKV, t);
      }
      // causal SDPA with GQA x6
      std::vector<float> att((size_t)T*6144, 0.f), sc(T);
      for(int t=0;t<T;++t) for(int hd_=0;hd_<NQ;++hd_){
        int g=hd_/6; float mx=-1e30f;
        for(int t2=0;t2<=t;++t2){ float dd=0;
          for(int d=0;d<HD;++d) dd+=Q[((size_t)t*NQ+hd_)*HD+d]*K_[((size_t)t2*NKV+g)*HD+d];
          sc[t2]=dd*(1.f/16.f); if(sc[t2]>mx)mx=sc[t2]; }
        float den=0; for(int t2=0;t2<=t;++t2){ sc[t2]=expf(sc[t2]-mx); den+=sc[t2]; }
        for(int d=0;d<HD;++d){ double a=0;
          for(int t2=0;t2<=t;++t2) a+=double(sc[t2]/den)*double(V[((size_t)t2*NKV+g)*HD+d]);
          att[((size_t)t*NQ+hd_)*HD+d]=(float)a; }
      }
      for(int t=0;t<T;++t){ float* gg=&G[(size_t)t*6144];
        // gate is sigmoid (transformers Qwen3_5Attention line 714), NOT silu
        for(int i=0;i<6144;++i){ float s=gg[i]; gg[i]=att[(size_t)t*6144+i]/(1.f+expf(-s)); } }
      linear_nvfp4_batch(P+"attn_output.weight", G.data(), core.data(), T);
    } else {
      linear_nvfp4_batch(P+"attn_qkv.weight", h.data(), qkv.data(), T);
      // causal depthwise conv k=4 + silu
      std::vector<float> cv((size_t)T*C, 0.f);
      auto Wc=f32(P+"ssm_conv1d.weight");
      for(int t=0;t<T;++t) for(int c=0;c<C;++c){ double a=0;
        for(int j=0;j<4;++j){ int tt=t-(3-j); float xv=(tt<0)?0.f:qkv[(size_t)tt*C+c]; a+=double(xv)*double(Wc[c*4+j]); }
        cv[(size_t)t*C+c]=silu((float)a); }
      std::vector<float> q((size_t)T*HV*KHD), k((size_t)T*HV*KHD), v((size_t)T*HV*VHD);
      for(int t=0;t<T;++t){ float* c0=&cv[(size_t)t*C];
        for(int hh=0;hh<HV;++hh){ int g=hh/3;
          for(int d=0;d<KHD;++d){ q[((size_t)t*HV+hh)*KHD+d]=c0[g*KHD+d]; k[((size_t)t*HV+hh)*KHD+d]=c0[KD+g*KHD+d]; }
          for(int d=0;d<VHD;++d) v[((size_t)t*HV+hh)*VHD+d]=c0[2*KD+hh*VHD+d]; } }
      std::vector<float> zl((size_t)T*VD), b((size_t)T*HV), ar((size_t)T*HV);
      linear_nvfp4_batch(P+"attn_gate.weight", h.data(), zl.data(), T);
      linear_nvfp4_batch(P+"ssm_beta.weight", h.data(), b.data(), T);
      { // ssm_alpha is F32
        auto Wa=f32(P+"ssm_alpha.weight");
        linear_batch(Wa, h.data(), ar.data(), T, HV, HID);
      }
      auto A=f32(P+"ssm_a"), DT=f32(P+"ssm_dt.bias"), Wn=f32(P+"ssm_norm.weight");
      std::vector<float> beta((size_t)T*HV), gl((size_t)T*HV);
      for(int t=0;t<T;++t) for(int hh=0;hh<HV;++hh){
        beta[(size_t)t*HV+hh]=1.f/(1.f+expf(-b[(size_t)t*HV+hh]));
        gl[(size_t)t*HV+hh]=-expf(A[hh])*softplus(ar[(size_t)t*HV+hh]+DT[hh]); }
      for(int t=0;t<T;++t){ qk_norm_scale(&q[(size_t)t*HV*KHD], &k[(size_t)t*HV*KHD], HV, KHD, 1.f/sqrtf(128.f)); }
      std::fill(S.begin(), S.end(), 0.f);
      for(int t=0;t<T;++t) for(int hh=0;hh<HV;++hh){
        float dec=expf(gl[(size_t)t*HV+hh]), be=beta[(size_t)t*HV+hh];
        float* Sh=&S[(size_t)hh*KHD*VHD];
        for(int i=0;i<KHD*VHD;++i) Sh[i]*=dec;
        float* kh=&k[((size_t)t*HV+hh)*KHD], *vh=&v[((size_t)t*HV+hh)*VHD], *qh=&q[((size_t)t*HV+hh)*KHD];
        for(int j=0;j<VHD;++j){ double dd=0; for(int i=0;i<KHD;++i) dd+=double(Sh[i*VHD+j])*double(kh[i]);
          double bv=(double(vh[j])-dd)*double(be);
          for(int i=0;i<KHD;++i) Sh[i*VHD+j]+=float(double(kh[i])*bv); }
        for(int j=0;j<VHD;++j){ double a=0; for(int i=0;i<KHD;++i) a+=double(Sh[i*VHD+j])*double(qh[i]);
          o[((size_t)t*HV+hh)*VHD+j]=(float)a; }
      }
      std::vector<float> gated((size_t)T*VD);
      for(int t=0;t<T;++t) for(int hh=0;hh<HV;++hh){
        float* oo=&o[((size_t)t*HV+hh)*VHD], *zz=&zl[((size_t)t*HV+hh)*VHD];
        double va=0; for(int d=0;d<VHD;++d) va+=double(oo[d])*double(oo[d]);
        float inv=1.f/sqrtf(float(va/VHD)+1e-6f);
        for(int d=0;d<VHD;++d) gated[((size_t)t*HV+hh)*VHD+d]=oo[d]*inv*Wn[d]*silu(zz[d]);
      }
      linear_nvfp4_batch(P+"ssm_out.weight", gated.data(), core.data(), T);
      if(li==0){ dbg_cv=cv; dbg_beta=beta; dbg_gl=gl; }
    }
    for(size_t i=0;i<x.size();++i) x[i]+=core[i]; // residual
    for(int t=0;t<T;++t) rms(&x[(size_t)t*HID], f32(P+"post_attention_norm.weight"), &h[(size_t)t*HID], HID, true);
    std::vector<float> gg((size_t)T*FFN), uu((size_t)T*FFN);
    linear_nvfp4_batch(P+"ffn_gate.weight", h.data(), gg.data(), T);
    linear_nvfp4_batch(P+"ffn_up.weight", h.data(), uu.data(), T);
    for(size_t i=0;i<gg.size();++i) gg[i]=silu(gg[i])*uu[i];
    std::vector<float> down((size_t)T*HID);
    linear_nvfp4_batch(P+"ffn_down.weight", gg.data(), down.data(), T);
    for(size_t i=0;i<x.size();++i) x[i]+=down[i];
    if(getenv("HERMES_BLK0")){ printf("e2 qkv0:"); for(int i=0;i<5;++i) printf(" %.4f",qkv[i]); printf("\n");
      if(getenv("HERMES_DUMPQKV")){ FILE* dd=fopen("/tmp/e2_qkv.bin","wb"); fwrite(qkv.data(),4,qkv.size(),dd); fclose(dd);
        FILE* dc=fopen("/tmp/e2_cv.bin","wb"); fwrite(dbg_cv.data(),4,dbg_cv.size(),dc); fclose(dc);
        FILE* db=fopen("/tmp/e2_beta.bin","wb"); fwrite(dbg_beta.data(),4,dbg_beta.size(),db); fclose(db);
        FILE* dg=fopen("/tmp/e2_gl.bin","wb"); fwrite(dbg_gl.data(),4,dbg_gl.size(),dg); fclose(dg);
        FILE* doo=fopen("/tmp/e2_o.bin","wb"); fwrite(o.data(),4,o.size(),doo); fclose(doo);
        FILE* dcc=fopen("/tmp/e2_core.bin","wb"); fwrite(core.data(),4,core.size(),dcc); fclose(dcc);
        FILE* dy=fopen("/tmp/e2_y.bin","wb"); fwrite(x.data(),4,x.size(),dy); fclose(dy); }
      { float w0[64]; deq_row(nvfp4("blk.0.attn_qkv.weight"), w0, 64);
        printf("e2 Wqkv[0,:5]:"); for(int i=0;i<5;++i) printf(" %.6f",w0[i]); printf("\n"); }
      printf("e2 h[0,:5]:"); for(int i=0;i<5;++i) printf(" %.6f",h[i]); printf("\n");
      double m=0,s=0; for(auto v:x){m+=v;} m/=x.size();
      for(auto v:x){s+=(v-m)*(v-m);} s=sqrt(s/x.size());
      float mn=x[0],mx=x[0]; for(auto v:x){if(v<mn)mn=v; if(v>mx)mx=v;}
      double mc=0,sc=0; for(auto v:core){mc+=v;} mc/=core.size();
      for(auto v:core){sc+=(v-mc)*(v-mc);} sc=sqrt(sc/core.size());
      printf("e2 blk0 out mean=%.5f std=%.5f min=%.2f max=%.2f | core mean=%.5f std=%.5f\n",m,s,mn,mx,mc,sc); fflush(stdout); return x; }
    if(getenv("HERMES_DBG") && (li%8==0 || li==NL-1)){
      double m=0,s=0; for(auto v:x){m+=v;} m/=x.size();
      for(auto v:x){s+=(v-m)*(v-m);} s=sqrt(s/x.size());
      printf("e2 layer %d mean=%.5f std=%.5f\n",li,m,s); fflush(stdout);
    }
  }
  if(h_last_out) for(int i=0;i<HID;++i) h_last_out[i]=x[(size_t)(T-1)*HID+i];
  std::vector<float> xn(x.size()), logits((size_t)T*VOC);
  for(int t=0;t<T;++t) rms(&x[(size_t)t*HID], f32("output_norm.weight"), &xn[(size_t)t*HID], HID, true);
  { // lm head in 8 vocab shards (avoid 5GB fp32 temp)
    int nshard = 8, rows = VOC / nshard;
    for(int s=0;s<nshard;++s){
      std::vector<float> ys((size_t)T*rows);
      linear_nvfp4_rows("output.weight", xn.data(), ys.data(), T, s*rows, (s+1)*rows);
      for(int t=0;t<T;++t) for(int i=0;i<rows;++i) logits[(size_t)t*VOC+s*rows+i]=ys[(size_t)t*rows+i];
    }
  }
  return logits;
}
void ModelFwd::prefill_with_state(const std::vector<int>& ids, std::vector<float>& logits_last, SeqState& st) const{
  // Simplest correct: full prefill, then derive states by re-running recurrence
  // is wasteful; instead run prefill and capture states inline (copy of prefill body
  // with state capture). For now: run prefill, then rebuild states with a second
  // lightweight pass is O(T^2)... Instead: single-token incremental from scratch
  // is exact: feed tokens one by one via decode_step from empty state.
  st.conv.assign((size_t)NL*C*3, 0.f);
  st.gdn.assign((size_t)NL*HV*KHD*VHD, 0.f);
  st.kcache.assign(NL, {}); st.vcache.assign(NL, {}); st.klen.assign(NL, 0);
  logits_last.assign(VOC, 0.f);
  for(size_t t=0;t<ids.size();++t){
    auto lg = decode_step(ids[t], st, (long)t);
    if(t+1==ids.size()) logits_last = lg;
  }
}
std::vector<float> ModelFwd::decode_step(int tok, SeqState& st, long pos) const{
  float c[64], s[64];
  for(int i=0;i<32;++i){ float f=1.f/powf(10000000.f,(float)(2*i)/64.f), a=(float)pos*f;
    c[i]=cosf(a); c[i+32]=cosf(a); s[i]=sinf(a); s[i+32]=sinf(a); }
  auto rope1 = [&](float* QK, int H){
    for(int h=0;h<H;++h){ float* r=&QK[h*HD];
      float a[32],b[32]; for(int i=0;i<32;++i){a[i]=r[i];b[i]=r[32+i];}
      for(int i=0;i<32;++i){ r[i]=a[i]*c[i]-b[i]*s[i]; r[32+i]=a[i]*s[i]+b[i]*c[i]; } }
  };
  std::vector<float> x(HID), h(HID);
  auto emb=bf16("token_embd.weight");
  for(int i=0;i<HID;++i) x[i]=bf2f(emb[(size_t)tok*HID+i]);
  std::vector<float> qkv(C), S_(HV*KHD*VHD);
  for(int li=0;li<NL;++li){
    char Pb[16]; snprintf(Pb,16,"blk.%d.",li); std::string P=Pb;
    rms(x.data(), f32(P+"attn_norm.weight"), h.data(), HID, true);
    std::vector<float> core(HID, 0.f);
    if(li%4==3){
      std::vector<float> qq(12288);
      linear_nvfp4_rows(P+"attn_q.weight", h.data(), qq.data(), 1, 0, 12288);
      float Q[24*256], G[24*256];
      auto qn=f32(P+"attn_q_norm.weight"), kn=f32(P+"attn_k_norm.weight");
      // single RMSNorm (1+w) like prefill (transformers Qwen3_5RMSNorm); NOT double norm
      for(int hd_=0;hd_<24;++hd_){ double ss=0; for(int d=0;d<256;++d){float v=qq[hd_*512+d]; ss+=double(v)*v;}
        float inv=1.f/sqrtf(float(ss/256)+1e-6f);
        for(int d=0;d<256;++d){ Q[hd_*256+d]=qq[hd_*512+d]*inv*qn[d]; G[hd_*256+d]=qq[hd_*512+256+d]; } }
      std::vector<float> kv(1024), vv(1024);
      linear_nvfp4_rows(P+"attn_k.weight", h.data(), kv.data(), 1, 0, 1024);
      linear_nvfp4_rows(P+"attn_v.weight", h.data(), vv.data(), 1, 0, 1024);
      float K[4*256];
      for(int hd_=0;hd_<4;++hd_){ double ss=0; for(int d=0;d<256;++d) ss+=double(kv[hd_*256+d])*double(kv[hd_*256+d]);
        float inv=1.f/sqrtf(float(ss/256)+1e-6f); for(int d=0;d<256;++d) K[hd_*256+d]=kv[hd_*256+d]*inv*kn[d]; }
      rope1(Q,24); rope1(K,4);
      // append KV (post-norm, post-rope, pre-GQA)
      auto& Kc=st.kcache[li]; auto& Vc=st.vcache[li];
      size_t L = st.klen[li]++;
      Kc.resize((L+1)*4*256); Vc.resize((L+1)*4*256);
      for(int i=0;i<4*256;++i){ Kc[L*4*256+i]=K[i]; Vc[L*4*256+i]=vv[i]; }
      // GQA causal SDPA, single query
      float att[24*256]={0};
      for(int hd_=0;hd_<24;++hd_){ int g=hd_/6; float mx=-1e30f; static thread_local float sc[1<<20];
        for(size_t t2=0;t2<=L;++t2){ float dd=0; for(int d=0;d<256;++d) dd+=Q[hd_*256+d]*Kc[t2*4*256+g*256+d]; sc[t2]=dd*(1.f/16.f); if(sc[t2]>mx)mx=sc[t2]; }
        float den=0; for(size_t t2=0;t2<=L;++t2){ sc[t2]=expf(sc[t2]-mx); den+=sc[t2]; }
        for(int d=0;d<256;++d){ double a=0; for(size_t t2=0;t2<=L;++t2) a+=double(sc[t2]/den)*double(Vc[t2*4*256+g*256+d]); att[hd_*256+d]=(float)a; } }
      // gate is sigmoid (transformers line 714), NOT silu
      for(int i=0;i<6144;++i){ float g8=G[i]; att[i]/=(1.f+expf(-g8)); }
      linear_nvfp4_rows(P+"attn_output.weight", att, core.data(), 1, 0, 5120);
    } else {
      std::vector<float> q1(C);
      linear_nvfp4_rows(P+"attn_qkv.weight", h.data(), q1.data(), 1, 0, C);
      float* CS=&st.conv[(size_t)li*C*3];
      float cv[10240];
      auto Wc=f32(P+"ssm_conv1d.weight");
      for(int c=0;c<C;++c){ double a=0;
        for(int j=0;j<3;++j) a+=double(CS[c*3+j])*double(Wc[c*4+j]);
        a+=double(q1[c])*double(Wc[c*4+3]); cv[c]=silu((float)a); }
      for(int c=0;c<C;++c){ CS[c*3]=CS[c*3+1]; CS[c*3+1]=CS[c*3+2]; CS[c*3+2]=q1[c]; }
      float q[48*128],k[48*128],v[48*128];
      for(int hh=0;hh<48;++hh){ int g=hh/3;
        for(int d=0;d<128;++d){ q[hh*128+d]=cv[g*128+d]; k[hh*128+d]=cv[KD+g*128+d]; }
        for(int d=0;d<128;++d) v[hh*128+d]=cv[2*KD+hh*128+d]; }
      // in-proj z/b/a + gate/beta
      std::vector<float> zl(VD);
      linear_nvfp4_rows(P+"attn_gate.weight", h.data(), zl.data(), 1, 0, VD);
      std::vector<float> bl(48), al(48);
      linear_nvfp4_rows(P+"ssm_beta.weight", h.data(), bl.data(), 1, 0, 48);
      { auto Wa=f32(P+"ssm_alpha.weight"); for(int i=0;i<48;++i){ double a=0; for(int kk=0;kk<HID;++kk) a+=double(h[kk])*double(Wa[i*HID+kk]); al[i]=(float)a; } }
      auto A=f32(P+"ssm_a"), DT=f32(P+"ssm_dt.bias"), Wn=f32(P+"ssm_norm.weight");
      // l2norm q,k + scale; keep z (zl) intact, write attention out to separate oe
      float oe[48*128];
      for(int hh=0;hh<48;++hh){
        double nq=0,nk=0; for(int d=0;d<128;++d){ nq+=double(q[hh*128+d])*double(q[hh*128+d]); nk+=double(k[hh*128+d])*double(k[hh*128+d]); }
        nq=1.0/sqrt(nq+1e-6); nk=1.0/sqrt(nk+1e-6);
        float be=1.f/(1.f+expf(-bl[hh])), dec=expf(-expf(A[hh])*softplus(al[hh]+DT[hh]));
        float* Sh=&st.gdn[(size_t)li*48*128*128+hh*128*128];
        for(int i=0;i<128*128;++i) Sh[i]*=dec;
        for(int j=0;j<128;++j){ double dd=0; for(int i=0;i<128;++i) dd+=double(Sh[i*128+j])*double(k[hh*128+i]*nk);
          double bv=(double(v[hh*128+j])-dd)*double(be);
          for(int i=0;i<128;++i) Sh[i*128+j]+=float(double(k[hh*128+i]*nk)*bv); }
        for(int j=0;j<128;++j){ double a=0; for(int i=0;i<128;++i) a+=double(Sh[i*128+j])*double(q[hh*128+i]*nq*0.0883883476); oe[hh*128+j]=(float)a; }
      }
      // RMSNormGated: RMS(oe) * Wn * silu(z), single attn_gate GEMM (zl still holds z)
      for(int hh=0;hh<48;++hh){ double va=0; for(int d=0;d<128;++d) va+=double(oe[hh*128+d])*double(oe[hh*128+d]);
        float inv=1.f/sqrtf(float(va/128)+1e-6f);
        for(int d=0;d<128;++d) zl[hh*128+d]=oe[hh*128+d]*inv*Wn[d]*silu(zl[hh*128+d]); }
      linear_nvfp4_rows(P+"ssm_out.weight", zl.data(), core.data(), 1, 0, 5120);
    }
    for(int i=0;i<HID;++i) x[i]+=core[i];
    rms(x.data(), f32(P+"post_attention_norm.weight"), h.data(), HID, true);
    std::vector<float> gg(FFN), uu(FFN);
    linear_nvfp4_rows(P+"ffn_gate.weight", h.data(), gg.data(), 1, 0, FFN);
    linear_nvfp4_rows(P+"ffn_up.weight", h.data(), uu.data(), 1, 0, FFN);
    for(int i=0;i<FFN;++i) gg[i]=silu(gg[i])*uu[i];
    std::vector<float> down(HID);
    linear_nvfp4_rows(P+"ffn_down.weight", gg.data(), down.data(), 1, 0, HID);
    for(int i=0;i<HID;++i) x[i]+=down[i];
  }
  std::vector<float> xn(HID);
  rms(x.data(), f32("output_norm.weight"), xn.data(), HID, true);
  std::vector<float> logits(VOC);
  { int nshard=8, rows=VOC/nshard;
    for(int s=0;s<nshard;++s){ std::vector<float> ys(rows);
      linear_nvfp4_rows("output.weight", xn.data(), ys.data(), 1, s*rows, (s+1)*rows);
      for(int i=0;i<rows;++i) logits[s*rows+i]=ys[i]; } }
  return logits;
}
// ---- MTP/NextN draft (blk.64) ----
void ModelFwd::mtp_hidden(const float* h_last, int tok, float* x_out) const {
  float e[HID], hn[HID], en[HID];
  auto emb = bf16("token_embd.weight");
  for(int i=0;i<HID;++i) e[i]=bf2f(emb[(size_t)tok*HID+i]);
  rms(h_last, f32("blk.64.nextn.hnorm.weight"), hn, HID, true);
  rms(e, f32("blk.64.nextn.enorm.weight"), en, HID, true);
  float eh[2*HID];
  for(int i=0;i<HID;++i){ eh[i]=en[i]; eh[HID+i]=hn[i]; } // concat [e_norm; h_norm]
  linear_nvfp4_rows("blk.64.nextn.eh_proj.weight", eh, x_out, 1, 0, HID);
}
std::vector<float> ModelFwd::mtp_draft(const float* h_last, int tok, MtpState& mst, long pos) const {
  const std::string P="blk.64.";
  float c[64], s[64];
  for(int i=0;i<32;++i){ float f=1.f/powf(10000000.f,(float)(2*i)/64.f), a=(float)pos*f;
    c[i]=cosf(a); c[i+32]=cosf(a); s[i]=sinf(a); s[i+32]=sinf(a); }
  float x[HID];
  mtp_hidden(h_last, tok, x);
  float h[HID];
  rms(x, f32(P+"attn_norm.weight"), h, HID, true);
  float qq[12288];
  linear_nvfp4_rows(P+"attn_q.weight", h, qq, 1, 0, 12288);
  float Q[NQ*HD], G[NQ*HD];
  auto qn=f32(P+"attn_q_norm.weight"), kn=f32(P+"attn_k_norm.weight");
  for(int hd_=0;hd_<NQ;++hd_){ double ss=0; for(int d=0;d<HD;++d){float v=qq[hd_*512+d]; ss+=double(v)*v;}
    float inv=1.f/sqrtf(float(ss/HD)+1e-6f);
    for(int d=0;d<HD;++d){ Q[hd_*HD+d]=qq[hd_*512+d]*inv*qn[d]; G[hd_*HD+d]=qq[hd_*512+256+d]; } }
  float kv[1024], vv[1024];
  linear_nvfp4_rows(P+"attn_k.weight", h, kv, 1, 0, 1024);
  linear_nvfp4_rows(P+"attn_v.weight", h, vv, 1, 0, 1024);
  float K[NKV*HD];
  for(int hd_=0;hd_<NKV;++hd_){ double ss=0; for(int d=0;d<HD;++d) ss+=double(kv[hd_*HD+d])*double(kv[hd_*HD+d]);
    float inv=1.f/sqrtf(float(ss/HD)+1e-6f); for(int d=0;d<HD;++d) K[hd_*HD+d]=kv[hd_*HD+d]*inv*kn[d]; }
  auto rope1=[&](float* QK, int H){
    for(int hh=0;hh<H;++hh){ float a[32],b[32]; for(int i=0;i<32;++i){a[i]=QK[hh*HD+i];b[i]=QK[hh*HD+32+i];}
      for(int i=0;i<32;++i){ QK[hh*HD+i]=a[i]*c[i]-b[i]*s[i]; QK[hh*HD+32+i]=a[i]*s[i]+b[i]*c[i]; } }
  };
  rope1(Q,NQ); rope1(K,NKV);
  size_t L=(size_t)mst.len++;
  mst.kcache.resize((L+1)*NKV*HD); mst.vcache.resize((L+1)*NKV*HD);
  for(int i=0;i<NKV*HD;++i){ mst.kcache[L*NKV*HD+i]=K[i]; mst.vcache[L*NKV*HD+i]=vv[i]; }
  float att[NQ*HD]={0};
  for(int hd_=0;hd_<NQ;++hd_){ int g=hd_/6; float mx=-1e30f; static thread_local float sc[1<<20];
    for(size_t t2=0;t2<=L;++t2){ float dd=0; for(int d=0;d<HD;++d) dd+=Q[hd_*HD+d]*mst.kcache[t2*NKV*HD+g*HD+d];
      sc[t2]=dd*(1.f/16.f); if(sc[t2]>mx)mx=sc[t2]; }
    float den=0; for(size_t t2=0;t2<=L;++t2){ sc[t2]=expf(sc[t2]-mx); den+=sc[t2]; }
    for(int d=0;d<HD;++d){ double a=0; for(size_t t2=0;t2<=L;++t2) a+=double(sc[t2]/den)*double(mst.vcache[t2*NKV*HD+g*HD+d]);
      att[hd_*HD+d]=(float)a; } }
  for(int i=0;i<6144;++i) att[i]/=(1.f+expf(-G[i])); // sigmoid gate
  float core[HID];
  linear_nvfp4_rows(P+"attn_output.weight", att, core, 1, 0, HID);
  for(int i=0;i<HID;++i) x[i]+=core[i];
  rms(x, f32(P+"post_attention_norm.weight"), h, HID, true);
  float gg[FFN], uu[FFN];
  linear_nvfp4_rows(P+"ffn_gate.weight", h, gg, 1, 0, FFN);
  linear_nvfp4_rows(P+"ffn_up.weight", h, uu, 1, 0, FFN);
  for(int i=0;i<FFN;++i) gg[i]=silu(gg[i])*uu[i];
  float down[HID];
  linear_nvfp4_rows(P+"ffn_down.weight", gg, down, 1, 0, HID);
  for(int i=0;i<HID;++i) x[i]+=down[i];
  float xn[HID];
  rms(x, f32("blk.64.nextn.shared_head_norm.weight"), xn, HID, true);
  std::vector<float> logits(VOC);
  { int nshard=8, rows=VOC/nshard;
    for(int sh=0;sh<nshard;++sh){ std::vector<float> ys(rows);
      linear_nvfp4_rows("output.weight", xn, ys.data(), 1, sh*rows, (sh+1)*rows);
      for(int i=0;i<rows;++i) logits[sh*rows+i]=ys[i]; } }
  return logits;
}
}
