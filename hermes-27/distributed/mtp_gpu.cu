// HERMES-27 MTP/NextN draft su GPU (blk.64). Vedi mtp_gpu.hpp.
// Pesi residenti pp2 (GPU1) via pp2::get_weight_ptr, single-copy, no duplicati.
// Kernel copiati da pp2_fwd.cu (stesse formule) + bf16_gemv per lineari BF16 MTP.
#include "mtp_gpu.hpp"
#include "pp2_fwd.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <climits>

namespace mtp_gpu {

#define MTP_CK(call, err, label) do{ cudaError_t e=(call); \
  if(e!=cudaSuccess){ err=std::string("cuda fail "#call": ")+cudaGetErrorString(e); goto label; } }while(0)

// ---------- kernel (stesse equazioni di pp2_fwd.cu / model_fwd.cpp) ----------
__global__ void rms_f32_k(float* y, const float* x, const float* w, int T, int N){
  int t=(int)blockIdx.x; if(t>=T) return;
  extern __shared__ float sh[];
  int tid=(int)threadIdx.x, B=(int)blockDim.x;
  double part=0;
  for(int i=tid;i<N;i+=B){ float v=x[(size_t)t*N+i]; part+=(double)v*v; }
  sh[tid]=(float)part; __syncthreads();
  for(int s=B/2;s>0;s>>=1){ if(tid<s) sh[tid]+=sh[tid+s]; __syncthreads(); }
  float inv=1.f/sqrtf(sh[0]/N+1e-6f);
  for(int i=tid;i<N;i+=B) y[(size_t)t*N+i]=x[(size_t)t*N+i]*inv*w[i];
}
__global__ void f32_to_bf16_k(uint16_t* out, const float* inp, int N){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i>=N) return;
  __nv_bfloat16 h=(__nv_bfloat16)inp[i];
  uint16_t b; memcpy(&b,&h,2); out[i]=b;
}
// GEMV BF16: y[N] = W[NxK] * x[K], W row-major BF16, x BF16, accumulo fp64
// (stesso schema di head_bf16_k in pp2_fwd.cu).
__global__ void bf16_gemv_k(float* y, const uint16_t* W, const uint16_t* x, int K, int N){
  int r=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(r>=N) return;
  const uint16_t* row=W+(size_t)r*K;
  double a=0;
  for(int k=0;k<K;++k){
    uint16_t wb=row[k];
    __nv_bfloat16 wh; memcpy(&wh,&wb,2);
    uint16_t xb=x[k];
    __nv_bfloat16 xh; memcpy(&xh,&xb,2);
    a+=(double)(float)xh*(double)(float)wh;
  }
  y[r]=(float)a;
}
__global__ void q_norm_rope_pos_k(const float* qq, const float* qn, float* Q, int pos){
  int bh=(int)blockIdx.x; int head=bh; int tid=(int)threadIdx.x;
  __shared__ float sh[256]; __shared__ float sq[256];
  float v=qq[(size_t)head*512+tid];
  sh[tid]=v*v; __syncthreads();
  for(int s=128;s>0;s>>=1){ if(tid<s) sh[tid]+=sh[tid+s]; __syncthreads(); }
  float inv=1.f/sqrtf(sh[0]/256.f+1e-6f);
  float qnrm=v*inv*qn[tid];
  sq[tid]=qnrm; __syncthreads();
  float y;
  if(tid<64){ int i=tid<32?tid:tid-32;
    float fr=1.f/powf(10000000.f,(float)(2*i)/64.f);
    float ang=(float)pos*fr, cc=cosf(ang), ss=sinf(ang);
    float a=sq[i], b=sq[32+i];
    y=(tid<32)?(a*cc-b*ss):(a*ss+b*cc);
  } else y=qnrm;
  Q[(size_t)head*256+tid]=y;
}
__global__ void k_norm_rope_pos_k(const float* kk, const float* kn, float* K_, int pos){
  int bh=(int)blockIdx.x; int head=bh; int tid=(int)threadIdx.x;
  __shared__ float sh[256]; __shared__ float sq[256];
  float v=kk[(size_t)head*256+tid];
  sh[tid]=v*v; __syncthreads();
  for(int s=128;s>0;s>>=1){ if(tid<s) sh[tid]+=sh[tid+s]; __syncthreads(); }
  float inv=1.f/sqrtf(sh[0]/256.f+1e-6f);
  float qnrm=v*inv*kn[tid];
  sq[tid]=qnrm; __syncthreads();
  float y;
  if(tid<64){ int i=tid<32?tid:tid-32;
    float fr=1.f/powf(10000000.f,(float)(2*i)/64.f);
    float ang=(float)pos*fr, cc=cosf(ang), ss=sinf(ang);
    float a=sq[i], b=sq[32+i];
    y=(tid<32)?(a*cc-b*ss):(a*ss+b*cc);
  } else y=qnrm;
  K_[(size_t)head*256+tid]=y;
}
__global__ void extract_G_k(const float* qq, float* G, int N){
  int idx=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(idx>=N) return;
  int hd=idx/256, d=idx%256;
  G[idx]=qq[(size_t)hd*512+256+d];
}
__global__ void sdpa_gqa_decode_k(const float* Q, const float* Kc, const float* Vc, float* ATT, int L){
  int b=(int)blockIdx.x; int hq=b; int g=hq/6; int tid=(int)threadIdx.x;
  // Dynamic scores: L can exceed 127 with extended MTP state. Caller launches
  // with (L+1)*sizeof(float) shared bytes; max 8192 entries fits SM120 limits.
  extern __shared__ float scores[];
  __shared__ float tmp[256];
  float qv=Q[(size_t)hq*256+tid];
  for(int t2=0;t2<=L;++t2){
    float kv=Kc[(size_t)t2*1024+g*256+tid];
    tmp[tid]=qv*kv; __syncthreads();
    for(int s=128;s>0;s>>=1){ if(tid<s) tmp[tid]+=tmp[tid+s]; __syncthreads(); }
    if(tid==0) scores[t2]=tmp[0]*(1.f/16.f);
    __syncthreads();
  }
  if(tid==0){
    float mx=-1e30f; for(int t2=0;t2<=L;++t2) if(scores[t2]>mx)mx=scores[t2];
    float den=0; for(int t2=0;t2<=L;++t2){ scores[t2]=expf(scores[t2]-mx); den+=scores[t2]; }
    for(int t2=0;t2<=L;++t2) scores[t2]/=den;
  }
  __syncthreads();
  float a=0; for(int t2=0;t2<=L;++t2) a+=scores[t2]*Vc[(size_t)t2*1024+g*256+tid];
  ATT[(size_t)hq*256+tid]=a;
}
__global__ void gate_sig_k(const float* att, const float* graw, float* gated, int N){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i>=N) return;
  float s=graw[i]; gated[i]=att[i]/(1.f+expf(-s)); // sigmoid gate (NON silu)
}
__global__ void silu_mul_k(const float* gg, const float* uu, float* hid, int N){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i>=N) return;
  float g=gg[i]; hid[i]=(g/(1.f+expf(-g)))*uu[i];
}
__global__ void add_k(float* x, const float* y, int N){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<N) x[i]+=y[i];
}

static bool wptr(const std::string& n, void*& d, int* ne0=nullptr, int* ne1=nullptr,
                 int* gt=nullptr, int* dev=nullptr){
  return pp2::get_weight_ptr(n, d, ne0, ne1, gt, dev);
}
static bool checked_weight(const char* n, void*& d, int ne0_want,
                           int ne1_want, int dtype_want, int dev_want,
                           std::string& err){
  int ne0=-1,ne1=-1,gt=-1,device=-1;
  if(!wptr(n,d,&ne0,&ne1,&gt,&device)||!d){
    err=std::string("missing MTP weight ")+n;
    return false;
  }
  if(ne0!=ne0_want||ne1!=ne1_want||gt!=dtype_want||device!=dev_want){
    err=std::string("MTP weight shape/type/device mismatch ")+n+
      " got="+std::to_string(ne0)+"x"+std::to_string(ne1)+
      " type="+std::to_string(gt)+" dev="+std::to_string(device);
    return false;
  }
  return true;
}

bool init(MtpState& st, std::string& err){ return init(st, err, MAXL); }

bool init(MtpState& st, std::string& err, int capacity){
  std::lock_guard<std::mutex> lock(st.mutex);
  if(capacity<1 || capacity>MAX_CONTEXT){ err="mtp capacity must be 1..8192"; return false; }
  if(st.poisoned){ err="mtp CUDA state poisoned: free_state then init required"; return false; }
  if(st.inited){
    if(st.capacity!=capacity){ err="mtp already initialized with different capacity"; return false; }
    return true;
  }
  st.Kc=nullptr; st.Vc=nullptr; st.len=0; st.first_pos=-1; st.capacity=capacity;
  cudaError_t ce=cudaSetDevice(1);
  if(ce!=cudaSuccess){ err="mtp cudaSetDevice(1): "+std::string(cudaGetErrorString(ce)); return false; }
  size_t bytes=(size_t)capacity*1024*4;
  float *k=nullptr,*v=nullptr;
  if(cudaMalloc(&k,bytes)!=cudaSuccess){ err="OOM MtpState Kc"; return false; }
  if(cudaMalloc(&v,bytes)!=cudaSuccess){ cudaFree(k); err="OOM MtpState Vc"; return false; }
  ce=cudaMemset(k,0,bytes);
  if(ce==cudaSuccess) ce=cudaMemset(v,0,bytes);
  if(ce!=cudaSuccess){ err="mtp KV init failed: "+std::string(cudaGetErrorString(ce)); cudaFree(k); cudaFree(v); return false; }
  st.Kc=k; st.Vc=v; st.len=0; st.first_pos=-1; st.poisoned=false; st.inited=true;
  return true;
}
void reset(MtpState& st){
  std::lock_guard<std::mutex> lock(st.mutex);
  detail::reset_metadata(st);
}

bool rewind(MtpState& st, int keep_len, std::string& err){
  std::lock_guard<std::mutex> lock(st.mutex);
  if(!detail::rewind_metadata(st,keep_len)){
    err=st.poisoned?"mtp rewind forbidden: state poisoned":
      "mtp rewind outside committed range or uninitialized";
    return false;
  }
  return true;
}
void free_state(MtpState& st){
  std::lock_guard<std::mutex> lock(st.mutex);
  if(!st.inited) return;
  cudaSetDevice(1);
  if(st.Kc){ cudaFree(st.Kc); st.Kc=nullptr; }
  if(st.Vc){ cudaFree(st.Vc); st.Vc=nullptr; }
  st.len=0; st.first_pos=-1; st.capacity=MAXL;
  st.inited=false; st.poisoned=false;
}

static inline float bf2f_cpu(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }

std::vector<float> draft_step(const float* h_last, int tok, MtpState& st, long pos,
                              std::string& err){
  std::lock_guard<std::mutex> lock(st.mutex);
  err.clear();
  if(!st.inited){ err="mtp not inited"; return {}; }
  if(st.poisoned){ err="mtp draft state poisoned; free_state then init required"; return {}; }
  if(!h_last){ err="null h_last"; return {}; }
  if(tok<0||tok>=VOC){ err="tok OOB"; return {}; }
  if(!detail::can_append(st,pos)){
    err="mtp KV position/capacity invalid or nonconsecutive (reset for new prompt)";
    return {};
  }
  // Strictly this accepts the GGUF 65-layer checkpoint, not any arbitrary
  // converted weights with a similar name. Tensor dtype/dims/dev checked below.
  // ---- pesi residenti (tutti su GPU1 tranne emb su GPU0) ----
  void *d_hn_w=nullptr,*d_en_w=nullptr,*d_eh_w=nullptr,
       *d_an_w=nullptr,*d_q_w=nullptr,*d_qn_w=nullptr,*d_k_w=nullptr,*d_kn_w=nullptr,
       *d_v_w=nullptr,*d_o_w=nullptr,*d_post_w=nullptr,
       *d_fg_w=nullptr,*d_fu_w=nullptr,*d_fd_w=nullptr,
       *d_shn_w=nullptr,*d_head_w=nullptr,*d_emb=nullptr;
  struct Spec{ const char* name; void** pointer; int n0,n1,type,device; };
  const Spec specs[]={
    {"blk.64.nextn.hnorm.weight",&d_hn_w,5120,1,0,1},
    {"blk.64.nextn.enorm.weight",&d_en_w,5120,1,0,1},
    {"blk.64.nextn.eh_proj.weight",&d_eh_w,10240,5120,30,1},
    {"blk.64.attn_norm.weight",&d_an_w,5120,1,0,1},
    {"blk.64.attn_q.weight",&d_q_w,5120,12288,30,1},
    {"blk.64.attn_q_norm.weight",&d_qn_w,256,1,0,1},
    {"blk.64.attn_k.weight",&d_k_w,5120,1024,30,1},
    {"blk.64.attn_k_norm.weight",&d_kn_w,256,1,0,1},
    {"blk.64.attn_v.weight",&d_v_w,5120,1024,30,1},
    {"blk.64.attn_output.weight",&d_o_w,6144,5120,30,1},
    {"blk.64.post_attention_norm.weight",&d_post_w,5120,1,0,1},
    {"blk.64.ffn_gate.weight",&d_fg_w,5120,17408,30,1},
    {"blk.64.ffn_up.weight",&d_fu_w,5120,17408,30,1},
    {"blk.64.ffn_down.weight",&d_fd_w,17408,5120,30,1},
    {"blk.64.nextn.shared_head_norm.weight",&d_shn_w,5120,1,0,1},
    {"output.weight",&d_head_w,5120,248320,30,1},
    {"token_embd.weight",&d_emb,5120,248320,30,0}
  };
  for(const auto& spec:specs){
    if(!checked_weight(spec.name,*spec.pointer,spec.n0,spec.n1,
                       spec.type,spec.device,err))return {};
  }

  // ---- embedding tok: riga BF16 da GPU0 -> fp32 host (leggero, 5120 elem) ----
  static thread_local uint16_t emb_row[5120];
  static thread_local float e_fp32[5120];
  cudaError_t ce=cudaSetDevice(0);
  if(ce!=cudaSuccess){ err="mtp embedding device0: "+std::string(cudaGetErrorString(ce)); return {}; }
  if(cudaMemcpy(emb_row,(uint16_t*)d_emb+(size_t)tok*5120,5120*2,cudaMemcpyDeviceToHost)!=cudaSuccess){
    err="emb D2H fail"; return {}; }
  for(int i=0;i<5120;++i) e_fp32[i]=bf2f_cpu(emb_row[i]);

  // ---- upload su GPU1 ----
  ce=cudaSetDevice(1);
  if(ce!=cudaSuccess){ err="mtp device1: "+std::string(cudaGetErrorString(ce)); return {}; }
  float *d_hl=nullptr,*d_e=nullptr,*d_hn=nullptr,*d_en=nullptr,*d_eh=nullptr,
        *d_x=nullptr,*d_h=nullptr,*d_qq=nullptr,*d_kk=nullptr,*d_vv=nullptr,
        *d_Q=nullptr,*d_K1=nullptr,*d_G=nullptr,*d_att=nullptr,*d_core=nullptr,
        *d_h2=nullptr,*d_gg=nullptr,*d_uu=nullptr,*d_hid=nullptr,*d_down=nullptr,
        *d_xn=nullptr,*d_log=nullptr;
  uint16_t *d_ehb=nullptr,*d_hb=nullptr,*d_gb=nullptr,*d_h2b=nullptr,*d_hidb=nullptr,*d_xnb=nullptr;
  std::vector<float> out;
  MTP_CK(cudaMalloc(&d_hl,5120*4),err,fail);
  MTP_CK(cudaMalloc(&d_e,5120*4),err,fail);
  MTP_CK(cudaMemcpy(d_hl,h_last,5120*4,cudaMemcpyHostToDevice),err,fail);
  MTP_CK(cudaMemcpy(d_e,e_fp32,5120*4,cudaMemcpyHostToDevice),err,fail);
  MTP_CK(cudaMalloc(&d_hn,5120*4),err,fail);
  MTP_CK(cudaMalloc(&d_en,5120*4),err,fail);
  MTP_CK(cudaMalloc(&d_eh,10240*4),err,fail);
  MTP_CK(cudaMalloc(&d_ehb,10240*2),err,fail);
  MTP_CK(cudaMalloc(&d_x,5120*4),err,fail);
  // h_norm = RMS(h_last, hnorm); e_norm = RMS(e, enorm) [dirette *w]
  rms_f32_k<<<1,512,(size_t)512*4>>>(d_hn,d_hl,(float*)d_hn_w,1,5120);
  rms_f32_k<<<1,512,(size_t)512*4>>>(d_en,d_e,(float*)d_en_w,1,5120);
  // concat [e_norm; h_norm]
  MTP_CK(cudaMemcpy(d_eh,d_en,5120*4,cudaMemcpyDeviceToDevice),err,fail);
  MTP_CK(cudaMemcpy(d_eh+5120,d_hn,5120*4,cudaMemcpyDeviceToDevice),err,fail);
  f32_to_bf16_k<<<(10240+255)/256,256>>>(d_ehb,d_eh,10240);
  bf16_gemv_k<<<(5120+255)/256,256>>>(d_x,(uint16_t*)d_eh_w,d_ehb,10240,5120);
  cudaFree(d_hl); d_hl=nullptr; cudaFree(d_e); d_e=nullptr;
  cudaFree(d_hn); d_hn=nullptr; cudaFree(d_en); d_en=nullptr;
  cudaFree(d_eh); d_eh=nullptr; cudaFree(d_ehb); d_ehb=nullptr;
  // ---- blocco draft full-attn ----
  MTP_CK(cudaMalloc(&d_h,5120*4),err,fail);
  MTP_CK(cudaMalloc(&d_hb,5120*2),err,fail);
  MTP_CK(cudaMalloc(&d_qq,12288*4),err,fail);
  MTP_CK(cudaMalloc(&d_kk,1024*4),err,fail);
  MTP_CK(cudaMalloc(&d_vv,1024*4),err,fail);
  MTP_CK(cudaMalloc(&d_Q,6144*4),err,fail);
  MTP_CK(cudaMalloc(&d_K1,1024*4),err,fail);
  MTP_CK(cudaMalloc(&d_G,6144*4),err,fail);
  MTP_CK(cudaMalloc(&d_att,6144*4),err,fail);
  rms_f32_k<<<1,512,(size_t)512*4>>>(d_h,d_x,(float*)d_an_w,1,5120);
  f32_to_bf16_k<<<(5120+255)/256,256>>>(d_hb,d_h,5120);
  bf16_gemv_k<<<(12288+255)/256,256>>>(d_qq,(uint16_t*)d_q_w,d_hb,5120,12288);
  bf16_gemv_k<<<(1024+255)/256,256>>>(d_kk,(uint16_t*)d_k_w,d_hb,5120,1024);
  bf16_gemv_k<<<(1024+255)/256,256>>>(d_vv,(uint16_t*)d_v_w,d_hb,5120,1024);
  cudaFree(d_h); d_h=nullptr; cudaFree(d_hb); d_hb=nullptr;
  q_norm_rope_pos_k<<<24,256>>>(d_qq,(float*)d_qn_w,d_Q,(int)pos);
  k_norm_rope_pos_k<<<4,256>>>(d_kk,(float*)d_kn_w,d_K1,(int)pos);
  extract_G_k<<<(6144+255)/256,256>>>(d_qq,d_G,6144);
  cudaFree(d_qq); d_qq=nullptr; cudaFree(d_kk); d_kk=nullptr;
  {
    int L=st.len;
    MTP_CK(cudaMemcpy(st.Kc+(size_t)L*1024,d_K1,1024*4,cudaMemcpyDeviceToDevice),err,fail);
    MTP_CK(cudaMemcpy(st.Vc+(size_t)L*1024,d_vv,1024*4,cudaMemcpyDeviceToDevice),err,fail);
    sdpa_gqa_decode_k<<<24,256,((size_t)L+1)*sizeof(float)>>>(d_Q,st.Kc,st.Vc,d_att,L);
  }
  cudaFree(d_Q); d_Q=nullptr; cudaFree(d_K1); d_K1=nullptr; cudaFree(d_vv); d_vv=nullptr;
  gate_sig_k<<<(6144+255)/256,256>>>(d_att,d_G,d_G,6144);
  cudaFree(d_att); d_att=nullptr;
  MTP_CK(cudaMalloc(&d_gb,6144*2),err,fail);
  MTP_CK(cudaMalloc(&d_core,5120*4),err,fail);
  f32_to_bf16_k<<<(6144+255)/256,256>>>(d_gb,d_G,6144);
  bf16_gemv_k<<<(5120+255)/256,256>>>(d_core,(uint16_t*)d_o_w,d_gb,6144,5120);
  cudaFree(d_G); d_G=nullptr; cudaFree(d_gb); d_gb=nullptr;
  add_k<<<(5120+255)/256,256>>>(d_x,d_core,5120);
  cudaFree(d_core); d_core=nullptr;
  // FFN
  MTP_CK(cudaMalloc(&d_h2,5120*4),err,fail);
  MTP_CK(cudaMalloc(&d_h2b,5120*2),err,fail);
  MTP_CK(cudaMalloc(&d_gg,17408*4),err,fail);
  MTP_CK(cudaMalloc(&d_uu,17408*4),err,fail);
  MTP_CK(cudaMalloc(&d_hid,17408*4),err,fail);
  rms_f32_k<<<1,512,(size_t)512*4>>>(d_h2,d_x,(float*)d_post_w,1,5120);
  f32_to_bf16_k<<<(5120+255)/256,256>>>(d_h2b,d_h2,5120);
  cudaFree(d_h2); d_h2=nullptr;
  bf16_gemv_k<<<(17408+255)/256,256>>>(d_gg,(uint16_t*)d_fg_w,d_h2b,5120,17408);
  bf16_gemv_k<<<(17408+255)/256,256>>>(d_uu,(uint16_t*)d_fu_w,d_h2b,5120,17408);
  cudaFree(d_h2b); d_h2b=nullptr;
  silu_mul_k<<<(17408+255)/256,256>>>(d_gg,d_uu,d_hid,17408);
  cudaFree(d_gg); d_gg=nullptr; cudaFree(d_uu); d_uu=nullptr;
  MTP_CK(cudaMalloc(&d_hidb,17408*2),err,fail);
  MTP_CK(cudaMalloc(&d_down,5120*4),err,fail);
  f32_to_bf16_k<<<(17408+255)/256,256>>>(d_hidb,d_hid,17408);
  cudaFree(d_hid); d_hid=nullptr;
  bf16_gemv_k<<<(5120+255)/256,256>>>(d_down,(uint16_t*)d_fd_w,d_hidb,17408,5120);
  cudaFree(d_hidb); d_hidb=nullptr;
  add_k<<<(5120+255)/256,256>>>(d_x,d_down,5120);
  cudaFree(d_down); d_down=nullptr;
  // shared_head_norm + head condivisa
  MTP_CK(cudaMalloc(&d_xn,5120*4),err,fail);
  MTP_CK(cudaMalloc(&d_xnb,5120*2),err,fail);
  MTP_CK(cudaMalloc(&d_log,248320*4),err,fail);
  rms_f32_k<<<1,512,(size_t)512*4>>>(d_xn,d_x,(float*)d_shn_w,1,5120);
  cudaFree(d_x); d_x=nullptr;
  f32_to_bf16_k<<<(5120+255)/256,256>>>(d_xnb,d_xn,5120);
  cudaFree(d_xn); d_xn=nullptr;
  bf16_gemv_k<<<(248320+255)/256,256>>>(d_log,(uint16_t*)d_head_w,d_xnb,5120,248320);
  cudaFree(d_xnb); d_xnb=nullptr;
  MTP_CK(cudaDeviceSynchronize(),err,fail);
  out.resize(248320);
  MTP_CK(cudaMemcpy(out.data(),d_log,(size_t)248320*4,cudaMemcpyDeviceToHost),err,fail);
  cudaFree(d_log); d_log=nullptr;
  if(!detail::commit_append(st,pos)){
    err="mtp bookkeeping failed after GPU write; state poisoned";
    goto fail;
  }
  return out;
fail:
  // If any CUDA failure occurred (including after writing only one of Kc/Vc),
  // the logical length is not sufficient evidence that retained cache entries
  // are valid. Never allow reset/rewind to silently revive such a state.
  st.poisoned=true;
  if(d_hl)cudaFree(d_hl); if(d_e)cudaFree(d_e);
  if(d_hn)cudaFree(d_hn); if(d_en)cudaFree(d_en);
  if(d_eh)cudaFree(d_eh); if(d_ehb)cudaFree(d_ehb);
  if(d_x)cudaFree(d_x); if(d_h)cudaFree(d_h); if(d_hb)cudaFree(d_hb);
  if(d_qq)cudaFree(d_qq); if(d_kk)cudaFree(d_kk); if(d_vv)cudaFree(d_vv);
  if(d_Q)cudaFree(d_Q); if(d_K1)cudaFree(d_K1);
  if(d_G)cudaFree(d_G); if(d_att)cudaFree(d_att);
  if(d_gb)cudaFree(d_gb); if(d_core)cudaFree(d_core);
  if(d_h2)cudaFree(d_h2); if(d_h2b)cudaFree(d_h2b);
  if(d_gg)cudaFree(d_gg); if(d_uu)cudaFree(d_uu);
  if(d_hid)cudaFree(d_hid); if(d_hidb)cudaFree(d_hidb);
  if(d_down)cudaFree(d_down); if(d_xn)cudaFree(d_xn);
  if(d_xnb)cudaFree(d_xnb); if(d_log)cudaFree(d_log);
  return {};
}

int argmax_v(const std::vector<float>& lg){
  if(lg.empty()) return -1;
  int bi=0; float bv=lg[0];
  for(size_t i=1;i<lg.size();++i) if(lg[i]>bv){bv=lg[i];bi=(int)i;}
  return bi;
}
void top5_v(const std::vector<float>& lg, int* oid, float* ov){
  for(int k=0;k<5;++k){ oid[k]=-1; ov[k]=-1e30f; }
  for(size_t v=0;v<lg.size();++v){
    float x=lg[v];
    for(int k=0;k<5;++k) if(x>ov[k]){
      for(int j=4;j>k;--j){ ov[j]=ov[j-1]; oid[j]=oid[j-1]; }
      ov[k]=x; oid[k]=(int)v; break;
    }
  }
}
int top5_overlap(const std::vector<float>& a, const std::vector<float>& b){
  int ia[5]; float va[5]; top5_v(a,ia,va);
  int ib[5]; float vb[5]; top5_v(b,ib,vb);
  int n=0;
  for(int i=0;i<5;++i) for(int j=0;j<5;++j) if(ia[i]==ib[j]){ n++; break; }
  return n;
}
bool finite_all(const std::vector<float>& v, double& mn, double& mx, double& mean){
  if(v.empty()){ mn=mx=mean=0; return false; }
  mn=v[0]; mx=v[0]; mean=0; bool fin=true;
  for(auto x:v){ if(!std::isfinite(x)) fin=false; if(x<mn)mn=x; if(x>mx)mx=x; mean+=x; }
  mean/=v.size();
  if(!std::isfinite(mean)) fin=false;
  return fin;
}

} // namespace mtp_gpu
