// HERMES-27 PP2 forward prefill — implementazione GEMV precisa (riuso pattern P1).
// PP split: GPU0 embed+blk0-31, GPU1 blk32-63+head. Single-copy raw GGUF resident.
// Conversione CUTLASS misurata su 1 matrice (report), forward usa GEMV diretto (0 conversioni in VRAM).
#include "pp2_fwd.hpp"
#include "hermes/weights.hpp"
#include "hermes/gguf.hpp"
#include "hermes/rmsnorm.hpp"
#include "hermes/nvfp4_gemv.hpp"
#include "hermes/nvfp4_gemm.hpp"
#include "hermes/gdn_step.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <unordered_map>
#include <vector>
#include <string>
#include <mutex>

using namespace hermes;

namespace pp2 {

static const char* CKPT = "/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf";
static const int HID=5120, FFN=17408, NL=64, VOC=248320;
static const int NQ=24, NKV=4, HD=256;
static const int KD=2048, VDD=6144, CC=10240, HV=48, KHD=128, VHD=128;
static const int SPLIT_LAST0=31; // GPU0: 0..31, GPU1: 32..63

// P6 arene persistenti decode T=1 (byte esatti, documentati in P5_BATCH.md appendice):
// GDN decode: h 20480 + hb 10240 + q1 40960 + cv 40960 + zl 24576
//   + b/ar/be/gl 4x192=768 + q/k/v/o 4x24576=98304 + g 24576 + core 20480
//   + gg/uu/hid 3x69632=208896 + down 20480 + h2b 10240 + gb 12288 + hidb 34816
//   = 568064 B.
// FULL decode: h 20480 + hb 10240 + qq 49152 + kk 4096 + vv 4096 + Q 24576
//   + K1 4096 + V1 4096 + G 24576 + att 24576 + core 20480
//   + gg/uu/hid 3x69632=208896 + down 20480 + h2b 10240 + hidb 34816 + gb 12288
//   = 477184 B.
// WS_BYTES = max = 568064. Extra per DecodeState: ids 4 (GPU0), x0/x1/xn 3x20480,
// logits VOC*4=993280 (GPU1). Tot ~2.07 MiB/req. Stessi kernel, stesso ordine,
// stessi input, default-stream sincrono (no Graphs in questo task).
static const size_t WS_GDN_BYTES = 568064;
static const size_t WS_FULL_BYTES = 477184;
static const size_t WS_BYTES = 568064;
static const size_t WS_X_BYTES = 5120u*4u;      // 20480
static const size_t WS_LOG_BYTES = 248320u*4u;  // 993280

struct WEntry { std::string name; int dev=-1; void* d=nullptr; size_t bytes=0; int ne0=0, ne1=0; int gtype=-1; };
static GgufWeightBackend g_be;
static bool g_up=false;
static std::unordered_map<std::string,WEntry> g_w0, g_w1;
static float* g_pinned=nullptr; // pinned host xfer [Tmax*HID]
static const size_t PIN_TMAX=64;
static Stats g_last_stats;
static std::mutex g_xfer_mu; // P5: rende decode_step thread-safe per batch parallelo (xfer atomico)

#define CUDA_OK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ err="cuda fail " #x ": "+std::string(cudaGetErrorString(e)); return false; } } while(0)
#define CUDA_OK_V(x,er) do{ cudaError_t e=(x); if(e!=cudaSuccess){ er="cuda fail " #x ": "+std::string(cudaGetErrorString(e)); return {}; } } while(0)

// ---------- kernels (riuso pattern P1, promossi a funzioni riusabili) ----------
__global__ void embed_gather_k(float* out, const uint16_t* emb, const int* ids, int T){
  int idx=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  int N=T*HID;
  if(idx>=N) return;
  int t=idx/HID, i=idx%HID;
  int tok=ids[t];
  uint16_t b=emb[(size_t)tok*HID+i];
  __nv_bfloat16 h; memcpy(&h,&b,2);
  out[idx]=(float)h;
}
__global__ void rms_f32_k(float* y, const float* x, const float* w, int T, int N){
  // 1 blocco per t
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
__global__ void conv_silu_k(const float* qkv, const float* Wc, float* cv, int T, int C){
  int idx=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  int N=T*C; if(idx>=N) return;
  int t=idx/C, c=idx%C;
  float acc=0.f;
  for(int j=0;j<4;++j){ int tt=t-(3-j); float xv=(tt<0)?0.f:qkv[(size_t)tt*C+c]; acc=__fmaf_rn(xv,Wc[(size_t)c*4+j],acc); } // FMA esplicito: identico a conv_decode_k
  cv[idx]=acc/(1.f+expf(-acc)); // SiLU division form: acc*sigmoid(acc)
}
__global__ void wa_gemm_k(const float* h, const float* Wa, float* ar, int T, int N, int K){
  int idx=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(idx>=T*N) return;
  int t=idx/N, n=idx%N;
  double a=0; for(int k=0;k<K;++k) a+=(double)h[(size_t)t*K+k]*(double)Wa[(size_t)n*K+k];
  ar[idx]=(float)a;
}
__global__ void beta_gl_k(const float* b, const float* ar, const float* A, const float* DT, float* beta, float* gl, int N){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i>=N) return;
  int hh=i%48;
  float bv=b[i];
  beta[i]=1.f/(1.f+expf(-bv));
  float x=ar[i]+DT[hh];
  float sp=x>20.f?x:logf(1.f+expf(x));
  // GGUF ssm_a stores -exp(A_log), already transformed by the converter.
  gl[i]=A[hh]*sp;
}
__global__ void split_qkv_k(const float* cv, float* q, float* k, float* v, int T){
  int idx=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  int N=T*HV*KHD; // q size == v size (KHD==VHD==128)
  if(idx>=N) return;
  int t=idx/(HV*KHD), rem=idx%(HV*KHD);
  int hh=rem/KHD, d=rem%KHD;
  // GGUF tiles V heads by K group so the Q/K heads repeat modulo the group count.
  int g=hh%(KD/KHD);
  q[idx]=cv[(size_t)t*CC+g*KHD+d];
  k[idx]=cv[(size_t)t*CC+KD+g*KHD+d];
  // v stesso idx (HV*VHD == HV*KHD)
  v[idx]=cv[(size_t)t*CC+2*KD+hh*VHD+d];
}
__global__ void qk_norm_scale_k(float* q, float* k, int T){
  // 1 blocco per (t,hh), 128 thread
  int bh=(int)blockIdx.x; int t=bh/HV, hh=bh%HV;
  int tid=(int)threadIdx.x;
  __shared__ float sq[128], sk[128];
  float qv=q[((size_t)t*HV+hh)*KHD+tid];
  float kv=k[((size_t)t*HV+hh)*KHD+tid];
  sq[tid]=qv*qv; sk[tid]=kv*kv; __syncthreads();
  for(int s=64;s>0;s>>=1){ if(tid<s){sq[tid]+=sq[tid+s]; sk[tid]+=sk[tid+s];} __syncthreads(); }
  float iq=1.f/sqrtf(sq[0]+1e-6f), ik=1.f/sqrtf(sk[0]+1e-6f);
  float sc=1.f/sqrtf(128.f);
  q[((size_t)t*HV+hh)*KHD+tid]=qv*iq*sc;
  k[((size_t)t*HV+hh)*KHD+tid]=kv*ik;
}
__global__ void gated_rms_k(const float* o, const float* zl, const float* Wn, float* gated, int T){
  int bh=(int)blockIdx.x; int t=bh/HV, hh=bh%HV;
  int tid=(int)threadIdx.x; // 0..127
  __shared__ float sh[128];
  float ov=o[((size_t)t*HV+hh)*VHD+tid];
  sh[tid]=ov*ov; __syncthreads();
  for(int s=64;s>0;s>>=1){ if(tid<s) sh[tid]+=sh[tid+s]; __syncthreads(); }
  float inv=1.f/sqrtf(sh[0]/VHD+1e-6f);
  float z=zl[((size_t)t*HV+hh)*VHD+tid];
  float s=z/(1.f+expf(-z));
  gated[((size_t)t*HV+hh)*VHD+tid]=ov*inv*Wn[tid]*s;
}
__global__ void add_k(float* x, const float* y, int N){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<N) x[i]+=y[i];
}
__global__ void silu_mul_k(const float* gg, const float* uu, float* hid, int N){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i>=N) return;
  float g=gg[i]; hid[i]=(g/(1.f+expf(-g)))*uu[i];
}
// full-attn riuso da P1 block3
__global__ void q_norm_rope_k(const float* qqkv, const float* qn, float* Q, int T){
  int bh=(int)blockIdx.x; int t=bh/24, head=bh%24; int tid=(int)threadIdx.x;
  __shared__ float sh[256]; __shared__ float sq[256];
  float v=qqkv[(size_t)t*12288+(size_t)head*512+tid];
  sh[tid]=v*v; __syncthreads();
  for(int s=128;s>0;s>>=1){ if(tid<s) sh[tid]+=sh[tid+s]; __syncthreads(); }
  float inv=1.f/sqrtf(sh[0]/256.f+1e-6f);
  float qnrm=v*inv*qn[tid];
  sq[tid]=qnrm; __syncthreads();
  float y;
  if(tid<64){ int i=tid<32?tid:tid-32;
    float fr=1.f/powf(10000000.f,(float)(2*i)/64.f);
    float ang=(float)t*fr, cc=cosf(ang), ss=sinf(ang);
    float a=sq[i], b=sq[32+i];
    y=(tid<32)?(a*cc-b*ss):(a*ss+b*cc);
  } else y=qnrm;
  Q[((size_t)t*24+head)*256+tid]=y;
}
__global__ void k_norm_rope_k(const float* kk, const float* kn, float* K_, int T){
  int bh=(int)blockIdx.x; int t=bh/4, head=bh%4; int tid=(int)threadIdx.x;
  __shared__ float sh[256]; __shared__ float sq[256];
  float v=kk[(size_t)t*1024+(size_t)head*256+tid];
  sh[tid]=v*v; __syncthreads();
  for(int s=128;s>0;s>>=1){ if(tid<s) sh[tid]+=sh[tid+s]; __syncthreads(); }
  float inv=1.f/sqrtf(sh[0]/256.f+1e-6f);
  float qnrm=v*inv*kn[tid];
  sq[tid]=qnrm; __syncthreads();
  float y;
  if(tid<64){ int i=tid<32?tid:tid-32;
    float fr=1.f/powf(10000000.f,(float)(2*i)/64.f);
    float ang=(float)t*fr, cc=cosf(ang), ss=sinf(ang);
    float a=sq[i], b=sq[32+i];
    y=(tid<32)?(a*cc-b*ss):(a*ss+b*cc);
  } else y=qnrm;
  K_[((size_t)t*4+head)*256+tid]=y;
}
__global__ void sdpa_gqa_k(const float* Q, const float* K_, const float* V, float* ATT, int T){
  int b=(int)blockIdx.x; int t=b/24, hq=b%24; int g=hq/6; int tid=(int)threadIdx.x;
  __shared__ float scores[64]; __shared__ float tmp[256];
  float qv=Q[((size_t)t*24+hq)*256+tid];
  for(int t2=0;t2<=t;++t2){
    float kv=K_[((size_t)t2*4+g)*256+tid];
    tmp[tid]=qv*kv; __syncthreads();
    for(int s=128;s>0;s>>=1){ if(tid<s) tmp[tid]+=tmp[tid+s]; __syncthreads(); }
    if(tid==0) scores[t2]=tmp[0]*(1.f/16.f);
    __syncthreads();
  }
  if(tid==0){
    float mx=-1e30f; for(int t2=0;t2<=t;++t2) if(scores[t2]>mx)mx=scores[t2];
    float den=0; for(int t2=0;t2<=t;++t2){ scores[t2]=expf(scores[t2]-mx); den+=scores[t2]; }
    for(int t2=0;t2<=t;++t2) scores[t2]/=den;
  }
  __syncthreads();
  float a=0; for(int t2=0;t2<=t;++t2) a+=scores[t2]*V[((size_t)t2*4+g)*256+tid];
  ATT[((size_t)t*24+hq)*256+tid]=a;
}
__global__ void gate_k(const float* att, const float* graw, float* gated, int N){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i>=N) return;
  float s=graw[i]; gated[i]=att[i]/(1.f+expf(-s));
}
__global__ void head_bf16_k(float* logits, const float* xn, const uint16_t* W, int T){
  int idx=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  int N=T*VOC; if(idx>=N) return;
  int t=idx/VOC, v=idx%VOC;
  const uint16_t* row=W+(size_t)v*HID;
  double a=0;
  for(int k=0;k<HID;++k){
    uint16_t b=row[k]; __nv_bfloat16 h; memcpy(&h,&b,2);
    a+=(double)xn[(size_t)t*HID+k]*(double)(float)h;
  }
  logits[idx]=(float)a;
}

// ---------- helpers ----------
static int parse_blk(const std::string& n){
  if(n.rfind("blk.",0)!=0) return -999;
  size_t a=4, b=n.find('.',a);
  if(b==std::string::npos) return -999;
  return atoi(n.substr(a,b-a).c_str());
}
static int dev_for(const std::string& n){
  int b=parse_blk(n);
  if(b==-999){
    if(n=="token_embd.weight") return 0;
    if(n=="output.weight"||n=="output_norm.weight") return 1;
    return 1; // default MTP etc su GPU1
  }
  if(b<=SPLIT_LAST0) return 0;
  if(b<=63) return 1;
  return 1; // blk64 MTP su GPU1
}
static WEntry* find_w(const std::string& n){
  auto it=g_w0.find(n); if(it!=g_w0.end()) return &it->second;
  auto jt=g_w1.find(n); if(jt!=g_w1.end()) return &jt->second;
  return nullptr;
}
static void gemv_batch(const uint16_t* d_in, const uint8_t* d_W, float* d_out, int H, int N, int T){
  // Experimental CUDA GEMV A/B. Preserve the already-validated scalar path
  // unless explicitly opted into the new raw-GGUF Warp4 implementation.
  // The flag is read only once per process; changing it requires a restart.
  static const Nvfp4GemvVariant variant = [] {
    const char* value=std::getenv("HERMES_NVFP4_GEMV");
    return (value && std::strcmp(value,"warp4")==0)
      ? Nvfp4GemvVariant::Warp4 : Nvfp4GemvVariant::Scalar;
  }();
  for(int t=0;t<T;++t)
    nvfp4_gemv_cuda_variant(d_in+(size_t)t*H,d_W,d_out+(size_t)t*N,H,N,variant);
}

static bool run_gdn_layer(int li, float* d_x, int T, std::string& err){
  char Pb[16]; snprintf(Pb,16,"blk.%d.",li); std::string P=Pb;
  WEntry *wNorm=find_w(P+"attn_norm.weight"), *wQkv=find_w(P+"attn_qkv.weight"),
         *wGate=find_w(P+"attn_gate.weight"), *wBeta=find_w(P+"ssm_beta.weight"),
         *wAlpha=find_w(P+"ssm_alpha.weight"), *wC=find_w(P+"ssm_conv1d.weight"),
         *wA=find_w(P+"ssm_a"), *wDT=find_w(P+"ssm_dt.bias"),
         *wN=find_w(P+"ssm_norm.weight"), *wOut=find_w(P+"ssm_out.weight"),
         *wPost=find_w(P+"post_attention_norm.weight"),
         *wFg=find_w(P+"ffn_gate.weight"), *wFu=find_w(P+"ffn_up.weight"), *wFd=find_w(P+"ffn_down.weight");
  if(!wNorm||!wQkv||!wGate||!wBeta||!wAlpha||!wC||!wA||!wDT||!wN||!wOut||!wPost||!wFg||!wFu||!wFd){ err="missing w gdn "+P; return false; }
  size_t nH=(size_t)T*HID;
  float *d_h=nullptr, *d_qkv=nullptr, *d_cv=nullptr, *d_zl=nullptr, *d_b=nullptr, *d_ar=nullptr,
        *d_be=nullptr, *d_gl=nullptr, *d_q=nullptr, *d_k=nullptr, *d_v=nullptr, *d_o=nullptr,
        *d_S=nullptr, *d_g=nullptr, *d_core=nullptr, *d_gg=nullptr, *d_uu=nullptr, *d_hid=nullptr, *d_down=nullptr;
  uint16_t *d_hb=nullptr, *d_h2b=nullptr, *d_gb=nullptr, *d_hidb=nullptr;
  cudaMalloc(&d_h,nH*4); cudaMalloc(&d_hb,nH*2);
  cudaMalloc(&d_qkv,(size_t)T*CC*4); cudaMalloc(&d_cv,(size_t)T*CC*4);
  cudaMalloc(&d_zl,(size_t)T*VDD*4); cudaMalloc(&d_b,(size_t)T*HV*4); cudaMalloc(&d_ar,(size_t)T*HV*4);
  cudaMalloc(&d_be,(size_t)T*HV*4); cudaMalloc(&d_gl,(size_t)T*HV*4);
  cudaMalloc(&d_q,(size_t)T*HV*KHD*4); cudaMalloc(&d_k,(size_t)T*HV*KHD*4); cudaMalloc(&d_v,(size_t)T*HV*VHD*4);
  cudaMalloc(&d_o,(size_t)T*HV*VHD*4); cudaMalloc(&d_S,(size_t)HV*KHD*VHD*4);
  cudaMalloc(&d_g,(size_t)T*VDD*4); cudaMalloc(&d_core,nH*4);
  cudaMalloc(&d_gg,(size_t)T*FFN*4); cudaMalloc(&d_uu,(size_t)T*FFN*4);
  cudaMalloc(&d_hid,(size_t)T*FFN*4); cudaMalloc(&d_down,nH*4);
  cudaMalloc(&d_h2b,nH*2); cudaMalloc(&d_gb,(size_t)T*VDD*2); cudaMalloc(&d_hidb,(size_t)T*FFN*2);
  // 1 RMS
  rms_f32_k<<<T,512,(size_t)512*4>>>(d_h,d_x,(float*)wNorm->d,T,HID);
  f32_to_bf16_k<<<(int)((nH+255)/256),256>>>(d_hb,d_h,(int)nH);
  // 2 linears GEMV
  gemv_batch(d_hb,(uint8_t*)wQkv->d,d_qkv,HID,CC,T);
  gemv_batch(d_hb,(uint8_t*)wGate->d,d_zl,HID,VDD,T);
  gemv_batch(d_hb,(uint8_t*)wBeta->d,d_b,HID,HV,T);
  wa_gemm_k<<<(int)((T*HV+255)/256),256>>>(d_h,(float*)wAlpha->d,d_ar,T,HV,HID);
  // 3 conv
  conv_silu_k<<<(int)(((size_t)T*CC+255)/256),256>>>(d_qkv,(float*)wC->d,d_cv,T,CC);
  // 4 split
  {
    size_t nq=(size_t)T*HV*KHD, nv=(size_t)T*HV*VHD;
    size_t mx=nq>nv?nq:nv;
    split_qkv_k<<<(int)((mx+255)/256),256>>>(d_cv,d_q,d_k,d_v,T);
  }
  // 5 beta/gl
  beta_gl_k<<<(int)(((size_t)T*HV+255)/256),256>>>(d_b,d_ar,(float*)wA->d,(float*)wDT->d,d_be,d_gl,(int)((size_t)T*HV));
  // 6 norm qk
  qk_norm_scale_k<<<(T*HV),128>>>(d_q,d_k,T);
  // 7 gdn
  cudaMemset(d_S,0,(size_t)HV*KHD*VHD*4);
  for(int t=0;t<T;++t) gdn_step_cuda(d_q+(size_t)t*HV*KHD,d_k+(size_t)t*HV*KHD,d_v+(size_t)t*HV*VHD,
    d_be+(size_t)t*HV,d_gl+(size_t)t*HV,d_S,d_o+(size_t)t*HV*VHD,HV,KHD,VHD);
  // 8 gated
  gated_rms_k<<<(T*HV),128>>>(d_o,d_zl,(float*)wN->d,d_g,T);
  f32_to_bf16_k<<<(int)(((size_t)T*VDD+255)/256),256>>>(d_gb,d_g,(int)((size_t)T*VDD));
  gemv_batch(d_gb,(uint8_t*)wOut->d,d_core,VDD,HID,T);
  add_k<<<(int)((nH+255)/256),256>>>(d_x,d_core,(int)nH);
  if(getenv("HERMES_L26") && li==26){
    std::vector<float> hg((size_t)T*VDD), hc((size_t)T*HID), ho((size_t)T*HV*VHD), hx2((size_t)T*HID);
    cudaMemcpy(hg.data(),d_g,(size_t)T*VDD*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(hc.data(),d_core,nH*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(ho.data(),d_o,(size_t)T*HV*VHD*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(hx2.data(),d_x,nH*4,cudaMemcpyDeviceToHost);
    FILE *ff=fopen("/tmp/l26_pre.bin","wb"); fwrite(hg.data(),4,hg.size(),ff); fwrite(hc.data(),4,hc.size(),ff); fwrite(ho.data(),4,ho.size(),ff); fwrite(hx2.data(),4,hx2.size(),ff); fclose(ff);
    printf("l26 pre saved\n"); fflush(stdout);
  }
  // 9 post + mlp
  rms_f32_k<<<T,512,(size_t)512*4>>>(d_h,d_x,(float*)wPost->d,T,HID);
  f32_to_bf16_k<<<(int)((nH+255)/256),256>>>(d_h2b,d_h,(int)nH);
  gemv_batch(d_h2b,(uint8_t*)wFg->d,d_gg,HID,FFN,T);
  gemv_batch(d_h2b,(uint8_t*)wFu->d,d_uu,HID,FFN,T);
  silu_mul_k<<<(int)(((size_t)T*FFN+255)/256),256>>>(d_gg,d_uu,d_hid,(int)((size_t)T*FFN));
  f32_to_bf16_k<<<(int)(((size_t)T*FFN+255)/256),256>>>(d_hidb,d_hid,(int)((size_t)T*FFN));
  gemv_batch(d_hidb,(uint8_t*)wFd->d,d_down,FFN,HID,T);
  add_k<<<(int)((nH+255)/256),256>>>(d_x,d_down,(int)nH);
  cudaDeviceSynchronize();
  cudaFree(d_h);cudaFree(d_hb);cudaFree(d_qkv);cudaFree(d_cv);cudaFree(d_zl);cudaFree(d_b);cudaFree(d_ar);
  cudaFree(d_be);cudaFree(d_gl);cudaFree(d_q);cudaFree(d_k);cudaFree(d_v);cudaFree(d_o);cudaFree(d_S);
  cudaFree(d_g);cudaFree(d_core);cudaFree(d_gg);cudaFree(d_uu);cudaFree(d_hid);cudaFree(d_down);
  cudaFree(d_h2b);cudaFree(d_gb);cudaFree(d_hidb);
  return true;
}

static bool run_full_layer(int li, float* d_x, int T, std::string& err){
  char Pb[16]; snprintf(Pb,16,"blk.%d.",li); std::string P=Pb;
  WEntry *wNorm=find_w(P+"attn_norm.weight"),
         *wQ=find_w(P+"attn_q.weight"), *wK=find_w(P+"attn_k.weight"), *wV=find_w(P+"attn_v.weight"),
         *wQn=find_w(P+"attn_q_norm.weight"), *wKn=find_w(P+"attn_k_norm.weight"),
         *wO=find_w(P+"attn_output.weight"),
         *wPost=find_w(P+"post_attention_norm.weight"),
         *wFg=find_w(P+"ffn_gate.weight"), *wFu=find_w(P+"ffn_up.weight"), *wFd=find_w(P+"ffn_down.weight");
  if(!wNorm||!wQ||!wK||!wV||!wQn||!wKn||!wO||!wPost||!wFg||!wFu||!wFd){ err="missing w full "+P; return false; }
  const int NQQ=12288, NKK=1024, NQHD=6144;
  size_t nH=(size_t)T*HID;
  float *d_h=nullptr,*d_qq=nullptr,*d_kk=nullptr,*d_vv=nullptr,*d_Q=nullptr,*d_K=nullptr,*d_V=nullptr,
        *d_att=nullptr,*d_G=nullptr,*d_core=nullptr,*d_gg=nullptr,*d_uu=nullptr,*d_hid=nullptr,*d_down=nullptr;
  uint16_t *d_hb=nullptr,*d_h2b=nullptr,*d_hidb=nullptr,*d_gb=nullptr;
  cudaMalloc(&d_h,nH*4); cudaMalloc(&d_hb,nH*2);
  cudaMalloc(&d_qq,(size_t)T*NQQ*4); cudaMalloc(&d_kk,(size_t)T*NKK*4); cudaMalloc(&d_vv,(size_t)T*NKK*4);
  cudaMalloc(&d_Q,(size_t)T*NQHD*4); cudaMalloc(&d_K,(size_t)T*NKK*4); cudaMalloc(&d_V,(size_t)T*NKK*4);
  cudaMalloc(&d_att,(size_t)T*NQHD*4); cudaMalloc(&d_G,(size_t)T*NQHD*4);
  cudaMalloc(&d_core,nH*4);
  cudaMalloc(&d_gg,(size_t)T*FFN*4); cudaMalloc(&d_uu,(size_t)T*FFN*4);
  cudaMalloc(&d_hid,(size_t)T*FFN*4); cudaMalloc(&d_down,nH*4);
  cudaMalloc(&d_h2b,nH*2); cudaMalloc(&d_hidb,(size_t)T*FFN*2); cudaMalloc(&d_gb,(size_t)T*NQHD*2);
  rms_f32_k<<<T,512,(size_t)512*4>>>(d_h,d_x,(float*)wNorm->d,T,HID);
  f32_to_bf16_k<<<(int)((nH+255)/256),256>>>(d_hb,d_h,(int)nH);
  gemv_batch(d_hb,(uint8_t*)wQ->d,d_qq,HID,NQQ,T);
  gemv_batch(d_hb,(uint8_t*)wK->d,d_kk,HID,NKK,T);
  gemv_batch(d_hb,(uint8_t*)wV->d,d_vv,HID,NKK,T);
  q_norm_rope_k<<<(T*NQ),256>>>(d_qq,(float*)wQn->d,d_Q,T);
  k_norm_rope_k<<<(T*NKV),256>>>(d_kk,(float*)wKn->d,d_K,T);
  cudaMemcpy(d_V,d_vv,(size_t)T*NKK*4,cudaMemcpyDeviceToDevice);
  // G raw = second half of qqkv per head: qq[hd*512+256+d]
  // build G on GPU from d_qq
  // simple kernel inline: use gate path — extract via copy kernel
  // implement as: G[(t*24+hd)*256+d] = qqkv[t*12288+hd*512+256+d]
  {
    // launch generic copy
    int N=(int)((size_t)T*NQHD);
    // use gate_k trick? write inline loop kernel via add? do manual:
    // reuse f32_to_bf16? no. Write small lambda kernel via cudaMemcpy2D? easiest: dedicated kernel
    // fallback: copy via host? T*NQHD=67584 floats=270KB, D2H/H2D cheap but adds sync.
    // Do device-to-device strided copy with simple kernel:
    // we inline a kernel here using CUDA dynamic? Instead use simple approach:
    // allocate and run a tiny kernel defined as lambda not allowed. Use cudaMemcpy per head? 264 copies.
    // Simplest correct: D2H qq, CPU extract G, H2D G (270KB, negligible for 16 layers).
    std::vector<float> qq((size_t)T*NQQ), G((size_t)T*NQHD);
    cudaMemcpy(qq.data(),d_qq,(size_t)T*NQQ*4,cudaMemcpyDeviceToHost);
    for(int t=0;t<T;++t) for(int hd=0;hd<24;++hd) for(int d=0;d<256;++d)
      G[((size_t)t*24+hd)*256+d]=qq[(size_t)t*NQQ+hd*512+256+d];
    cudaMemcpy(d_G,G.data(),(size_t)T*NQHD*4,cudaMemcpyHostToDevice);
  }
  sdpa_gqa_k<<<(T*NQ),256>>>(d_Q,d_K,d_V,d_att,T);
  gate_k<<<(int)(((size_t)T*NQHD+255)/256),256>>>(d_att,d_G,d_G,(int)((size_t)T*NQHD));
  f32_to_bf16_k<<<(int)(((size_t)T*NQHD+255)/256),256>>>(d_gb,d_G,(int)((size_t)T*NQHD));
  gemv_batch(d_gb,(uint8_t*)wO->d,d_core,NQHD,HID,T);
  add_k<<<(int)((nH+255)/256),256>>>(d_x,d_core,(int)nH);
  rms_f32_k<<<T,512,(size_t)512*4>>>(d_h,d_x,(float*)wPost->d,T,HID);
  f32_to_bf16_k<<<(int)((nH+255)/256),256>>>(d_h2b,d_h,(int)nH);
  gemv_batch(d_h2b,(uint8_t*)wFg->d,d_gg,HID,FFN,T);
  gemv_batch(d_h2b,(uint8_t*)wFu->d,d_uu,HID,FFN,T);
  silu_mul_k<<<(int)(((size_t)T*FFN+255)/256),256>>>(d_gg,d_uu,d_hid,(int)((size_t)T*FFN));
  f32_to_bf16_k<<<(int)(((size_t)T*FFN+255)/256),256>>>(d_hidb,d_hid,(int)((size_t)T*FFN));
  gemv_batch(d_hidb,(uint8_t*)wFd->d,d_down,FFN,HID,T);
  add_k<<<(int)((nH+255)/256),256>>>(d_x,d_down,(int)nH);
  cudaDeviceSynchronize();
  cudaFree(d_h);cudaFree(d_hb);cudaFree(d_qq);cudaFree(d_kk);cudaFree(d_vv);
  cudaFree(d_Q);cudaFree(d_K);cudaFree(d_V);cudaFree(d_att);cudaFree(d_G);cudaFree(d_core);
  cudaFree(d_gg);cudaFree(d_uu);cudaFree(d_hid);cudaFree(d_down);cudaFree(d_h2b);cudaFree(d_hidb);cudaFree(d_gb);
  return true;
}

// ---------- API ----------
bool upload(Stats& st, std::string& err){
  return upload(st,err,CKPT);
}

bool upload(Stats& st, std::string& err, const std::string& model_path){
  if(model_path.empty()){ err="empty model path"; return false; }
  if(g_up){
    if(g_be.file().path!=model_path){ err="PP2 already uploaded a different model"; return false; }
    st=g_last_stats; return true;
  }
  auto t0=std::chrono::high_resolution_clock::now();
  try{ g_be.open(model_path); }catch(std::exception& e){ err=std::string("GGUF OPEN FAIL: ")+e.what(); return false; }
  const auto& gf=g_be.file();
  size_t f0=0,tt0=0,f1=0,tt1=0;
  cudaSetDevice(0); cudaMemGetInfo(&f0,&tt0);
  cudaSetDevice(1); cudaMemGetInfo(&f1,&tt1);
  st.free0_before=f0; st.free1_before=f1;
  // conversione 1 matrice (misura, non residente doppia)
  {
    auto v=g_be.view("blk.0.attn_qkv.weight");
    int N=10240,K=5120;
    size_t sfa=Nvfp4GemmW4A4::sfa_bytes(11,5120);
    size_t sfb=Nvfp4GemmW4A4::sfb_bytes(11,N,K);
    std::vector<uint8_t> Bq((size_t)N*K/2), SFB(sfb);
    auto tc0=std::chrono::high_resolution_clock::now();
    Nvfp4GemmW4A4::build_B_from_gguf(v.data,Bq.data(),SFB.data(),N,K);
    auto tc1=std::chrono::high_resolution_clock::now();
    st.convert_ms_one=std::chrono::duration<double,std::milli>(tc1-tc0).count();
    printf("pp2 convert one blk0.qkv N=%d K=%d sfa=%zu sfb=%zu ms=%.2f backend=%s\n",
      N,K,sfa,sfb,st.convert_ms_one,Nvfp4GemmW4A4::backend_name());
    fflush(stdout);
  }
  size_t b0=0,b1=0;
  for(auto &t: gf.tensors){
    std::string n=t.name;
    int dev=dev_for(n);
    auto vw=g_be.view(n);
    size_t bytes=vw.size;
    // vw.size = n_bytes? usa info
    void* d=nullptr;
    cudaSetDevice(dev);
    cudaError_t e=cudaMalloc(&d,bytes);
    if(e!=cudaSuccess){ err="OOM "+n+" dev="+std::to_string(dev)+" bytes="+std::to_string(bytes)+" : "+cudaGetErrorString(e); return false; }
    e=cudaMemcpy(d,vw.data,bytes,cudaMemcpyHostToDevice);
    if(e!=cudaSuccess){ err="H2D fail "+n+" : "+std::string(cudaGetErrorString(e)); return false; }
    WEntry we; we.name=n; we.dev=dev; we.d=d; we.bytes=bytes; we.ne0=(int)t.ne[0]; we.ne1=(int)(t.n_dims>1?(int)t.ne[1]:1);
    we.gtype=(int)t.type;
    if(dev==0){ g_w0[n]=we; b0+=bytes; } else { g_w1[n]=we; b1+=bytes; }
  }
  // pinned xfer
  cudaSetDevice(0);
  cudaHostAlloc((void**)&g_pinned, PIN_TMAX*HID*sizeof(float), cudaHostAllocDefault);
  auto t1=std::chrono::high_resolution_clock::now();
  st.upload_sec=std::chrono::duration<double>(t1-t0).count();
  st.bytes_gpu0=b0; st.bytes_gpu1=b1;
  size_t fa0=0,ta0=0,fa1=0,ta1=0;
  cudaSetDevice(0); cudaMemGetInfo(&fa0,&ta0);
  cudaSetDevice(1); cudaMemGetInfo(&fa1,&ta1);
  st.free0_after=fa0; st.free1_after=fa1;
  printf("pp2 upload sec=%.2f bytes0=%zu (%.3f GiB) bytes1=%zu (%.3f GiB)\n",
    st.upload_sec,b0,b0/1024.0/1024/1024,b1,b1/1024.0/1024/1024);
  printf("pp2 vram free0 before=%zu MiB after=%zu MiB | free1 before=%zu MiB after=%zu MiB\n",
    f0/1048576,fa0/1048576,f1/1048576,fa1/1048576);
  fflush(stdout);
  // VRAM hard abort 15GiB used
  {
    size_t used0=ta0-fa0, used1=ta1-fa1;
    const size_t LIM=15ULL*1024*1024*1024;
    if(used0>LIM||used1>LIM){ err="VRAM>15GB abort used0="+std::to_string(used0)+" used1="+std::to_string(used1); return false; }
  }
  // checksum spot: 1 tensore/GPU F32 vs CPU
  {
    auto sum_cpu = [&](const std::string& n)->double{
      auto vw=g_be.view(n); const float* p=(const float*)vw.data;
      size_t N=vw.size/4; double s=0; for(size_t i=0;i<N;++i) s+=p[i]; return s;
    };
    std::string n0="blk.0.attn_norm.weight", n1="output_norm.weight";
    double s0c=sum_cpu(n0), s1c=sum_cpu(n1);
    auto it0=g_w0.find(n0); auto it1=g_w1.find(n1);
    if(it0==g_w0.end()||it1==g_w1.end()){ err="checksum missing"; return false; }
    std::vector<float> h0(it0->second.bytes/4), h1(it1->second.bytes/4);
    cudaSetDevice(0); cudaMemcpy(h0.data(),it0->second.d,it0->second.bytes,cudaMemcpyDeviceToHost);
    cudaSetDevice(1); cudaMemcpy(h1.data(),it1->second.d,it1->second.bytes,cudaMemcpyDeviceToHost);
    double s0g=0,s1g=0; for(auto v:h0)s0g+=v; for(auto v:h1)s1g+=v;
    printf("pp2 checksum %s cpu=%.6f gpu=%.6f diff=%.3e | %s cpu=%.6f gpu=%.6f diff=%.3e\n",
      n0.c_str(),s0c,s0g,fabs(s0c-s0g),n1.c_str(),s1c,s1g,fabs(s1c-s1g));
    fflush(stdout);
    if(fabs(s0c-s0g)>1e-3||fabs(s1c-s1g)>1e-3){ err="checksum mismatch"; return false; }
  }
  g_up=true; g_last_stats=st;
  return true;
}

int eos_token_id(){
  if(!g_up) return -1;
  const auto& kv=g_be.file().kv_u64;
  auto it=kv.find("tokenizer.ggml.eos_token_id");
  if(it==kv.end()||it->second>=(uint64_t)VOC) return -1;
  return (int)it->second;
}

std::vector<float> prefill(const std::vector<int>& ids, Stats& st, std::string& err,
                           std::vector<float>* h_last_out){
  if(!g_up){ err="not uploaded"; return {}; }
  int T=(int)ids.size();
  if(T<=0||T>(int)PIN_TMAX){ err="bad T"; return {}; }
  auto tp0=std::chrono::high_resolution_clock::now();
  size_t nH=(size_t)T*HID;
  // GPU0
  cudaSetDevice(0);
  int* d_ids=nullptr; float* d_x0=nullptr;
  cudaMalloc(&d_ids,T*4); cudaMemcpy(d_ids,ids.data(),T*4,cudaMemcpyHostToDevice);
  cudaMalloc(&d_x0,nH*4);
  {
    auto it=g_w0.find("token_embd.weight");
    if(it==g_w0.end()){ err="no emb"; return {}; }
    embed_gather_k<<<(int)((nH+255)/256),256>>>(d_x0,(uint16_t*)it->second.d,d_ids,T);
    cudaDeviceSynchronize();
  }
  for(int li=0;li<=SPLIT_LAST0;++li){
    bool is_full=(li%4==3);
    bool ok = is_full? run_full_layer(li,d_x0,T,err) : run_gdn_layer(li,d_x0,T,err);
    if(!ok) return {};
    if(li%8==0||li==SPLIT_LAST0){
      std::vector<float> hx(nH); cudaMemcpy(hx.data(),d_x0,nH*4,cudaMemcpyDeviceToHost);
      double m=0; for(auto v:hx)m+=v; m/=hx.size();
      double s=0; for(auto v:hx)s+=(v-m)*(v-m); s=sqrt(s/hx.size());
      printf("pp2 gpu0 layer %d mean=%.5f std=%.5f\n",li,m,s); fflush(stdout);
    }
  }
  // xfer GPU0->host->GPU1 pinned
  auto tx0=std::chrono::high_resolution_clock::now();
  cudaMemcpy(g_pinned,d_x0,nH*4,cudaMemcpyDeviceToHost);
  cudaSetDevice(1);
  float* d_x1=nullptr; cudaMalloc(&d_x1,nH*4);
  cudaMemcpy(d_x1,g_pinned,nH*4,cudaMemcpyHostToDevice);
  auto tx1=std::chrono::high_resolution_clock::now();
  st.xfer_ms=std::chrono::duration<double,std::milli>(tx1-tx0).count();
  printf("pp2 xfer T=%d bytes=%zu ms=%.3f (pinned host)\n",T,nH*4,st.xfer_ms); fflush(stdout);
  cudaFree(d_ids); cudaFree(d_x0);
  // GPU1
  for(int li=SPLIT_LAST0+1;li<NL;++li){
    bool is_full=(li%4==3);
    bool ok = is_full? run_full_layer(li,d_x1,T,err) : run_gdn_layer(li,d_x1,T,err);
    if(!ok){ cudaFree(d_x1); return {}; }
    if(li%8==0||li==NL-1){
      std::vector<float> hx(nH); cudaMemcpy(hx.data(),d_x1,nH*4,cudaMemcpyDeviceToHost);
      double m=0; for(auto v:hx)m+=v; m/=hx.size();
      double s=0; for(auto v:hx)s+=(v-m)*(v-m); s=sqrt(s/hx.size());
      printf("pp2 gpu1 layer %d mean=%.5f std=%.5f\n",li,m,s); fflush(stdout);
    }
  }
  // output norm + head
  {
    // MTP accessor minimale: ultimo hidden pre-output_norm (ultimo token).
    if(h_last_out){
      h_last_out->resize(HID);
      cudaMemcpy(h_last_out->data(),d_x1+(size_t)(T-1)*HID,HID*4,cudaMemcpyDeviceToHost);
    }
    auto itn=g_w1.find("output_norm.weight");
    auto itw=g_w1.find("output.weight");
    if(itn==g_w1.end()||itw==g_w1.end()){ err="no head"; cudaFree(d_x1); return {}; }
    float* d_xn=nullptr; cudaMalloc(&d_xn,nH*4);
    rms_f32_k<<<T,512,(size_t)512*4>>>(d_xn,d_x1,(float*)itn->second.d,T,HID);
    float* d_log=nullptr; cudaMalloc(&d_log,(size_t)T*VOC*4);
    head_bf16_k<<<(int)(((size_t)T*VOC+255)/256),256>>>(d_log,d_xn,(uint16_t*)itw->second.d,T);
    cudaDeviceSynchronize();
    std::vector<float> logits((size_t)T*VOC);
    cudaMemcpy(logits.data(),d_log,(size_t)T*VOC*4,cudaMemcpyDeviceToHost);
    cudaFree(d_xn); cudaFree(d_log); cudaFree(d_x1);
    auto tp1=std::chrono::high_resolution_clock::now();
    st.prefill_ms=std::chrono::duration<double,std::milli>(tp1-tp0).count();
    printf("pp2 prefill T=%d ms=%.1f (incl xfer %.2f)\n",T,st.prefill_ms,st.xfer_ms); fflush(stdout);
    return logits;
  }
}

void free_all(){
  for(auto &kv:g_w0) cudaSetDevice(0), cudaFree(kv.second.d);
  for(auto &kv:g_w1) cudaSetDevice(1), cudaFree(kv.second.d);
  g_w0.clear(); g_w1.clear();
  if(g_pinned){ cudaFreeHost(g_pinned); g_pinned=nullptr; }
  g_up=false;
}

// Accessor minimale MTP (nessuna copy, nessun transfer di ownership).
bool get_weight_ptr(const std::string& name, void*& dptr, int* ne0,
                    int* ne1, int* gtype, int* dev){
  if(!g_up) return false;
  WEntry* w=find_w(name);
  if(!w) return false;
  dptr=w->d;
  if(ne0) *ne0=w->ne0;
  if(ne1) *ne1=w->ne1;
  if(gtype) *gtype=w->gtype;
  if(dev) *dev=w->dev;
  return true;
}

void top5(const float* logits_TxV, int T, int V, int* out_id, float* out_val){
  const float* last=logits_TxV+(size_t)(T-1)*V;
  for(int k=0;k<5;++k){ out_id[k]=-1; out_val[k]=-1e30f; }
  for(int v=0;v<V;++v){
    float x=last[v];
    for(int k=0;k<5;++k) if(x>out_val[k]){
      for(int j=4;j>k;--j){ out_val[j]=out_val[j-1]; out_id[j]=out_id[j-1]; }
      out_val[k]=x; out_id[k]=v; break;
    }
  }
}

// ---------- GATE2 checks ----------
static bool read_bin_f(const char* path, std::vector<float>& out, size_t expect_n){
  FILE* f=fopen(path,"rb"); if(!f){ printf("OPEN FAIL %s\n",path); return false; }
  fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
  if((size_t)sz!=expect_n*4){ printf("SIZE MISMATCH %s got %ld expect %zu\n",path,sz,expect_n*4); fclose(f); return false; }
  out.resize(expect_n); size_t r=fread(out.data(),4,expect_n,f); fclose(f);
  return r==expect_n;
}

int check_blk0_gpu0(){
  printf("\n=== GATE2 blk.0 GPU0 via P2 resident (vs golden P1) ===\n");
  if(!g_up){ printf("not uploaded\n"); return 1; }
  const int T=11;
  const int ids[11]={6205,161421,303,220,17,1374,9911,1147,27053,6059,13};
  // h CPU come P1 (embed+norm da mmap backend)
  auto vEmb=g_be.view("token_embd.weight");
  auto vNorm=g_be.view("blk.0.attn_norm.weight");
  const uint16_t* embB=(const uint16_t*)vEmb.data;
  const float* wN=(const float*)vNorm.data;
  std::vector<float> x((size_t)T*HID), h((size_t)T*HID);
  for(int t=0;t<T;++t){ const uint16_t* row=embB+(size_t)ids[t]*HID;
    for(int i=0;i<HID;++i){ __nv_bfloat16 hb; memcpy(&hb,&row[i],2); x[(size_t)t*HID+i]=(float)hb; } }
  for(int t=0;t<T;++t){ double ss=0; for(int i=0;i<HID;++i) ss+=(double)x[(size_t)t*HID+i]*x[(size_t)t*HID+i];
    float inv=1.f/sqrtf((float)(ss/HID)+1e-6f);
    for(int i=0;i<HID;++i) h[(size_t)t*HID+i]=x[(size_t)t*HID+i]*inv*wN[i]; }
  // golden qkv/cv
  std::vector<float> qkv_ref, cv_ref, beta_ref, gl_ref, o_ref, core_ref, y_ref;
  if(!read_bin_f("/tmp/e1_cv_qkv.bin",qkv_ref,(size_t)T*10240)) return 1;
  if(!read_bin_f("/tmp/e1_cv.bin",cv_ref,(size_t)T*10240)) return 1;
  if(!read_bin_f("/tmp/e1_beta.bin",beta_ref,(size_t)T*48)) return 1;
  if(!read_bin_f("/tmp/e1_gl.bin",gl_ref,(size_t)T*48)) return 1;
  if(!read_bin_f("/tmp/e1_o.bin",o_ref,(size_t)T*48*128)) return 1;
  if(!read_bin_f("/tmp/e1_core.bin",core_ref,(size_t)T*5120)) return 1;
  if(!read_bin_f("/tmp/e1_y.bin",y_ref,(size_t)T*5120)) return 1;
  cudaSetDevice(0);
  // s1 conv con Wc residente
  auto itWc=g_w0.find("blk.0.ssm_conv1d.weight");
  if(itWc==g_w0.end()){ printf("no Wc resident\n"); return 1; }
  float *d_qkv=nullptr,*d_wc=nullptr,*d_cv=nullptr;
  size_t n=(size_t)T*10240;
  cudaMalloc(&d_qkv,n*4); cudaMalloc(&d_cv,n*4);
  cudaMemcpy(d_qkv,qkv_ref.data(),n*4,cudaMemcpyHostToDevice);
  // Wc residente già su device: usa direttamente itWc->second.d
  conv_silu_k<<<(int)((n+255)/256),256>>>(d_qkv,(float*)itWc->second.d,d_cv,T,10240);
  cudaDeviceSynchronize();
  std::vector<float> cv_g(n); cudaMemcpy(cv_g.data(),d_cv,n*4,cudaMemcpyDeviceToHost);
  double mx=0; for(size_t i=0;i<n;++i){ double e=fabs((double)cv_g[i]-cv_ref[i]); if(e>mx)mx=e; }
  printf("blk0 conv via-resident maxerr=%.3e thr 1e-4 %s\n",mx,mx<1e-4?"PASS":"FAIL");
  cudaFree(d_qkv); cudaFree(d_cv);
  if(mx>=1e-4) return 1;
  // s3/s4 lineari GEMV via residenti: Wz, Wb, Wout, ffn
  std::vector<uint16_t> hb((size_t)T*HID);
  for(size_t i=0;i<h.size();++i){ __nv_bfloat16 hh=(__nv_bfloat16)h[i]; uint16_t b; memcpy(&b,&hh,2); hb[i]=b; }
  uint16_t* d_h=nullptr; cudaMalloc(&d_h,(size_t)T*HID*2);
  cudaMemcpy(d_h,hb.data(),(size_t)T*HID*2,cudaMemcpyHostToDevice);
  auto gemv_check=[&](const std::string& wn,int N,int K,const std::vector<float>& ref,float thr,const char* tag)->int{
    auto it=g_w0.find(wn); if(it==g_w0.end()){ printf("missing %s\n",wn.c_str()); return 1; }
    float* d_y=nullptr; cudaMalloc(&d_y,(size_t)T*N*4);
    gemv_batch(d_h,(uint8_t*)it->second.d,d_y,K,N,T);
    cudaDeviceSynchronize();
    std::vector<float> Y((size_t)T*N); cudaMemcpy(Y.data(),d_y,(size_t)T*N*4,cudaMemcpyDeviceToHost);
    double mxx=0; for(size_t i=0;i<Y.size();++i){ double e=fabs((double)Y[i]-ref[i]); if(e>mxx)mxx=e; }
    printf("%s via-resident maxerr=%.3e thr %.3g %s\n",tag,mxx,(double)thr,mxx<thr?"PASS":"FAIL");
    cudaFree(d_y);
    return mxx<thr?0:1;
  };
  // per beta/gl/core/y servono ref intermedi: ricostruiamo zl/b come P1 ma con GEMV residenti
  // Wz
  {
    auto it=g_w0.find("blk.0.attn_gate.weight");
    float* d_zl=nullptr; cudaMalloc(&d_zl,(size_t)T*6144*4);
    gemv_batch(d_h,(uint8_t*)it->second.d,d_zl,HID,6144,T);
    cudaDeviceSynchronize();
    // confronta zl? non abbiamo golden zl diretto; confronta beta finale
    std::vector<float> zl((size_t)T*6144); cudaMemcpy(zl.data(),d_zl,(size_t)T*6144*4,cudaMemcpyDeviceToHost);
    cudaFree(d_zl);
    // Wb -> b -> beta
    auto itb=g_w0.find("blk.0.ssm_beta.weight");
    float* d_b=nullptr; cudaMalloc(&d_b,(size_t)T*48*4);
    gemv_batch(d_h,(uint8_t*)itb->second.d,d_b,HID,48,T);
    cudaDeviceSynchronize();
    std::vector<float> b((size_t)T*48); cudaMemcpy(b.data(),d_b,(size_t)T*48*4,cudaMemcpyDeviceToHost);
    cudaFree(d_b);
    std::vector<float> beta((size_t)T*48);
    for(size_t i=0;i<beta.size();++i) beta[i]=1.f/(1.f+expf(-b[i]));
    double mxb=0; for(size_t i=0;i<beta.size();++i){ double e=fabs((double)beta[i]-beta_ref[i]); if(e>mxb)mxb=e; }
    printf("blk0 beta via-resident maxerr=%.3e thr 5e-3 %s\n",mxb,mxb<5e-3?"PASS":"FAIL");
    if(mxb>=5e-3){ cudaFree(d_h); return 1; }
  }
  // gdn o via residente (usa cv golden + beta/gl golden per isolare, come P1? qui usiamo cv golden)
  {
    // q/k/v da cv golden, l2norm CPU, poi gdn_step_cuda con S su GPU0
    std::vector<float> q((size_t)T*HV*KHD), k((size_t)T*HV*KHD), v((size_t)T*HV*VHD);
    for(int t=0;t<T;++t){ float* c0=&cv_ref[(size_t)t*CC];
      for(int hh=0;hh<HV;++hh){ int g=hh%(KD/KHD);
        for(int d=0;d<KHD;++d){ q[((size_t)t*HV+hh)*KHD+d]=c0[g*KHD+d]; k[((size_t)t*HV+hh)*KHD+d]=c0[KD+g*KHD+d]; }
        for(int d=0;d<VHD;++d) v[((size_t)t*HV+hh)*VHD+d]=c0[2*KD+hh*VHD+d]; } }
    // norm CPU
    std::vector<float> qn(q.size()), kn(k.size());
    for(int t=0;t<T;++t) for(int hh=0;hh<HV;++hh){
      double nq=0,nk=0;
      for(int d=0;d<KHD;++d){ nq+=(double)q[((size_t)t*HV+hh)*KHD+d]*q[((size_t)t*HV+hh)*KHD+d];
        nk+=(double)k[((size_t)t*HV+hh)*KHD+d]*k[((size_t)t*HV+hh)*KHD+d]; }
      float iq=1.f/sqrtf((float)nq+1e-6f), ik=1.f/sqrtf((float)nk+1e-6f);
      float sc=1.f/sqrtf(128.f);
      for(int d=0;d<KHD;++d){ qn[((size_t)t*HV+hh)*KHD+d]=(float)((double)q[((size_t)t*HV+hh)*KHD+d]*iq*sc);
        kn[((size_t)t*HV+hh)*KHD+d]=(float)((double)k[((size_t)t*HV+hh)*KHD+d]*ik); } }
    float *d_qn=nullptr,*d_kn=nullptr,*d_v=nullptr,*d_be=nullptr,*d_gl=nullptr,*d_S=nullptr,*d_o=nullptr;
    cudaMalloc(&d_qn,qn.size()*4); cudaMalloc(&d_kn,kn.size()*4); cudaMalloc(&d_v,v.size()*4);
    cudaMalloc(&d_be,beta_ref.size()*4); cudaMalloc(&d_gl,gl_ref.size()*4);
    cudaMalloc(&d_S,(size_t)HV*KHD*VHD*4); cudaMalloc(&d_o,o_ref.size()*4);
    cudaMemcpy(d_qn,qn.data(),qn.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_kn,kn.data(),kn.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_v,v.data(),v.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_be,beta_ref.data(),beta_ref.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_gl,gl_ref.data(),gl_ref.size()*4,cudaMemcpyHostToDevice);
    cudaMemset(d_S,0,(size_t)HV*KHD*VHD*4);
    for(int t=0;t<T;++t) gdn_step_cuda(d_qn+t*HV*KHD,d_kn+t*HV*KHD,d_v+t*HV*VHD,d_be+t*HV,d_gl+t*HV,d_S,d_o+t*HV*VHD,HV,KHD,VHD);
    cudaDeviceSynchronize();
    std::vector<float> og(o_ref.size()); cudaMemcpy(og.data(),d_o,o_ref.size()*4,cudaMemcpyDeviceToHost);
    double mxo=0; for(size_t i=0;i<og.size();++i){ double e=fabs((double)og[i]-o_ref[i]); if(e>mxo)mxo=e; }
    printf("blk0 o via-resident maxerr=%.3e thr 5e-3 %s\n",mxo,mxo<5e-3?"PASS":"FAIL");
    cudaFree(d_qn);cudaFree(d_kn);cudaFree(d_v);cudaFree(d_be);cudaFree(d_gl);cudaFree(d_S);cudaFree(d_o);
    if(mxo>=5e-3){ cudaFree(d_h); return 1; }
  }
  cudaFree(d_h);
  printf("GATE2 blk0 GPU0 PASS\n");
  return 0;
}

int check_blk32_gpu1(){
  printf("\n=== GATE2 blk.32 GPU1 via P2 resident (GPUvsCPU-dequant) ===\n");
  if(!g_up){ printf("not uploaded\n"); return 1; }
  const int T=11;
  const int ids[11]={6205,161421,303,220,17,1374,9911,1147,27053,6059,13};
  auto vEmb=g_be.view("token_embd.weight");
  auto vN=g_be.view("blk.32.attn_norm.weight");
  const uint16_t* embB=(const uint16_t*)vEmb.data;
  const float* wN=(const float*)vN.data;
  std::vector<float> h((size_t)T*HID);
  {
    std::vector<float> x((size_t)T*HID);
    for(int t=0;t<T;++t){ const uint16_t* row=embB+(size_t)ids[t]*HID;
      for(int i=0;i<HID;++i){ __nv_bfloat16 hb; memcpy(&hb,&row[i],2); x[(size_t)t*HID+i]=(float)hb; } }
    for(int t=0;t<T;++t){ double ss=0; for(int i=0;i<HID;++i) ss+=(double)x[(size_t)t*HID+i]*x[(size_t)t*HID+i];
      float inv=1.f/sqrtf((float)(ss/HID)+1e-6f);
      for(int i=0;i<HID;++i) h[(size_t)t*HID+i]=x[(size_t)t*HID+i]*inv*wN[i]; }
  }
  // CPU-dequant ref per qkv blk32
  auto vQ=g_be.view("blk.32.attn_qkv.weight");
  int N=10240, K=5120, nb=K/64;
  std::vector<float> W((size_t)N*K), t64(64);
  for(int n=0;n<N;++n){ const uint8_t* row=vQ.data+(size_t)n*nb*36; float* out=&W[(size_t)n*K];
    for(int b=0;b<nb;++b){ nvfp4_block_dequant(row+b*36,t64.data()); for(int j=0;j<64;++j) out[b*64+j]=t64[j]; } }
  std::vector<float> Ycpu((size_t)T*N,0.f);
  for(int t=0;t<T;++t) for(int n=0;n<N;++n){ double a=0;
    for(int k=0;k<K;++k) a+=(double)h[(size_t)t*K+k]*(double)W[(size_t)n*K+k];
    Ycpu[(size_t)t*N+n]=(float)a; }
  // GPU via residente GPU1
  cudaSetDevice(1);
  std::vector<uint16_t> hb((size_t)T*HID);
  for(size_t i=0;i<h.size();++i){ __nv_bfloat16 hh=(__nv_bfloat16)h[i]; uint16_t b; memcpy(&b,&hh,2); hb[i]=b; }
  uint16_t* d_h=nullptr; float* d_y=nullptr;
  cudaMalloc(&d_h,(size_t)T*HID*2); cudaMalloc(&d_y,(size_t)T*N*4);
  cudaMemcpy(d_h,hb.data(),(size_t)T*HID*2,cudaMemcpyHostToDevice);
  auto it=g_w1.find("blk.32.attn_qkv.weight");
  if(it==g_w1.end()){ printf("missing blk32 qkv resident\n"); return 1; }
  gemv_batch(d_h,(uint8_t*)it->second.d,d_y,K,N,T);
  cudaDeviceSynchronize();
  std::vector<float> Yg((size_t)T*N); cudaMemcpy(Yg.data(),d_y,(size_t)T*N*4,cudaMemcpyDeviceToHost);
  double mx=0; for(size_t i=0;i<Yg.size();++i){ double e=fabs((double)Yg[i]-Ycpu[i]); if(e>mx)mx=e; }
  double rms=0; for(auto v:Ycpu)rms+=(double)v*v; rms=sqrt(rms/Ycpu.size());
  printf("blk32 qkv GPUvsCPU-dequant maxerr=%.5f rms=%.4f rel=%.4f thr 0.2 %s\n",mx,rms,mx/(rms+1e-12),mx/(rms+1e-12)<0.2?"PASS":"FAIL");
  cudaFree(d_h); cudaFree(d_y);
  // gdn isolated GPUvsCPU
  {
    const int H=48,KH=128,VH=128;
    std::vector<float> q(H*KH,0.1f),k(H*KH,0.1f),vv(H*VH,0.2f),be(H,0.5f),gl(H,-0.1f),S(H*KH*VH,0.f),o1(H*VH),o2(H*VH);
    // cpu
    gdn_step_cpu(q.data(),k.data(),vv.data(),be.data(),gl.data(),S.data(),o1.data(),H,KH,VH);
    // gpu
    float *dq=nullptr,*dk=nullptr,*dv=nullptr,*db=nullptr,*dg=nullptr,*dS=nullptr,*do_=nullptr;
    cudaMalloc(&dq,H*KH*4);cudaMalloc(&dk,H*KH*4);cudaMalloc(&dv,H*VH*4);
    cudaMalloc(&db,H*4);cudaMalloc(&dg,H*4);cudaMalloc(&dS,H*KH*VH*4);cudaMalloc(&do_,H*VH*4);
    cudaMemcpy(dq,q.data(),H*KH*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dk,k.data(),H*KH*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dv,vv.data(),H*VH*4,cudaMemcpyHostToDevice);
    cudaMemcpy(db,be.data(),H*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dg,gl.data(),H*4,cudaMemcpyHostToDevice);
    cudaMemset(dS,0,H*KH*VH*4);
    gdn_step_cuda(dq,dk,dv,db,dg,dS,do_,H,KH,VH);
    cudaDeviceSynchronize();
    cudaMemcpy(o2.data(),do_,H*VH*4,cudaMemcpyDeviceToHost);
    double mxx=0; for(size_t i=0;i<o1.size();++i){ double e=fabs((double)o1[i]-o2[i]); if(e>mxx)mxx=e; }
    printf("blk32 gdn isolated GPUvsCPU maxerr=%.3e thr 1e-5 %s\n",mxx,mxx<1e-5?"PASS":"FAIL");
    cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(db);cudaFree(dg);cudaFree(dS);cudaFree(do_);
    if(mxx>=1e-5) return 1;
  }
  if(mx/(rms+1e-12)>=0.2) return 1;
  printf("GATE2 blk32 GPU1 PASS\n");
  return 0;
}

// ---------- P4 decode incrementale ----------
__global__ void conv_decode_k(const float* q1, const float* Wc, float* CS, float* cv, int C){
  int c=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(c>=C) return;
  float c0=CS[(size_t)c*3+0], c1=CS[(size_t)c*3+1], c2=CS[(size_t)c*3+2];
  float acc=0.f;
  acc=__fmaf_rn(c0,Wc[(size_t)c*4+0],acc);
  acc=__fmaf_rn(c1,Wc[(size_t)c*4+1],acc);
  acc=__fmaf_rn(c2,Wc[(size_t)c*4+2],acc);
  acc=__fmaf_rn(q1[c],Wc[(size_t)c*4+3],acc); // stessa catena FMA di conv_silu_k
  cv[c]=acc/(1.f+expf(-acc)); // SiLU division form: acc*sigmoid(acc)
  CS[(size_t)c*3+0]=c1; CS[(size_t)c*3+1]=c2; CS[(size_t)c*3+2]=q1[c];
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
  extern __shared__ float smem[];
  float* scores=smem;
  float* tmp=scores+L+1;
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

static bool run_gdn_decode_layer(int li, float* d_x, DecodeState& ds, std::string& err){
  char Pb[16]; snprintf(Pb,16,"blk.%d.",li); std::string P=Pb;
  WEntry *wNorm=find_w(P+"attn_norm.weight"), *wQkv=find_w(P+"attn_qkv.weight"),
         *wGate=find_w(P+"attn_gate.weight"), *wBeta=find_w(P+"ssm_beta.weight"),
         *wAlpha=find_w(P+"ssm_alpha.weight"), *wC=find_w(P+"ssm_conv1d.weight"),
         *wA=find_w(P+"ssm_a"), *wDT=find_w(P+"ssm_dt.bias"),
         *wN=find_w(P+"ssm_norm.weight"), *wOut=find_w(P+"ssm_out.weight"),
         *wPost=find_w(P+"post_attention_norm.weight"),
         *wFg=find_w(P+"ffn_gate.weight"), *wFu=find_w(P+"ffn_up.weight"), *wFd=find_w(P+"ffn_down.weight");
  if(!wNorm||!wQkv||!wGate||!wBeta||!wAlpha||!wC||!wA||!wDT||!wN||!wOut||!wPost||!wFg||!wFu||!wFd){ err="missing w gdn dec "+P; return false; }
  if(!ds.S[li]||!ds.CS[li]){ err="decode state null gdn "+P; return false; }
  void* wsbase = (li<=SPLIT_LAST0)? ds.ws0 : ds.ws1;
  if(!wsbase){ err="decode ws null gdn "+P; return false; }
  const int T=1; size_t nH=(size_t)T*HID;
  // P6: carve arena persistente (stesso ordine/tipi dei vecchi malloc, nessun malloc/free).
  uint8_t* ws = (uint8_t*)wsbase;
  float*    d_h    = (float*)(ws+0);        // 20480
  uint16_t* d_hb   = (uint16_t*)(ws+20480); // 10240
  float*    d_q1   = (float*)(ws+30720);    // 40960
  float*    d_cv   = (float*)(ws+71680);    // 40960
  float*    d_zl   = (float*)(ws+112640);   // 24576
  float*    d_b    = (float*)(ws+137216);   // 192
  float*    d_ar   = (float*)(ws+137408);   // 192
  float*    d_be   = (float*)(ws+137600);   // 192
  float*    d_gl   = (float*)(ws+137792);   // 192
  float*    d_q    = (float*)(ws+137984);   // 24576
  float*    d_k    = (float*)(ws+162560);   // 24576
  float*    d_v    = (float*)(ws+187136);   // 24576
  float*    d_o    = (float*)(ws+211712);   // 24576
  float*    d_g    = (float*)(ws+236288);   // 24576
  float*    d_core = (float*)(ws+260864);   // 20480
  float*    d_gg   = (float*)(ws+281344);   // 69632
  float*    d_uu   = (float*)(ws+350976);   // 69632
  float*    d_hid  = (float*)(ws+420608);   // 69632
  float*    d_down = (float*)(ws+490240);   // 20480
  uint16_t* d_h2b  = (uint16_t*)(ws+510720);// 10240
  uint16_t* d_gb   = (uint16_t*)(ws+520960);// 12288
  uint16_t* d_hidb = (uint16_t*)(ws+533248);// 34816 (fine 568064)
  rms_f32_k<<<T,512,(size_t)512*4>>>(d_h,d_x,(float*)wNorm->d,T,HID);
  f32_to_bf16_k<<<(int)((nH+255)/256),256>>>(d_hb,d_h,(int)nH);
  gemv_batch(d_hb,(uint8_t*)wQkv->d,d_q1,HID,CC,T);
  gemv_batch(d_hb,(uint8_t*)wGate->d,d_zl,HID,VDD,T);
  gemv_batch(d_hb,(uint8_t*)wBeta->d,d_b,HID,HV,T);
  wa_gemm_k<<<(int)((T*HV+255)/256),256>>>(d_h,(float*)wAlpha->d,d_ar,T,HV,HID);
  conv_decode_k<<<(int)((CC+255)/256),256>>>(d_q1,(float*)wC->d,ds.CS[li],d_cv,CC);
  {
    size_t nq=(size_t)T*HV*KHD;
    split_qkv_k<<<(int)((nq+255)/256),256>>>(d_cv,d_q,d_k,d_v,T);
  }
  beta_gl_k<<<(int)(((size_t)T*HV+255)/256),256>>>(d_b,d_ar,(float*)wA->d,(float*)wDT->d,d_be,d_gl,(int)((size_t)T*HV));
  qk_norm_scale_k<<<(T*HV),128>>>(d_q,d_k,T);
  gdn_step_cuda(d_q,d_k,d_v,d_be,d_gl,ds.S[li],d_o,HV,KHD,VHD);
  gated_rms_k<<<(T*HV),128>>>(d_o,d_zl,(float*)wN->d,d_g,T);
  f32_to_bf16_k<<<(int)(((size_t)VDD+255)/256),256>>>(d_gb,d_g,(int)((size_t)VDD));
  gemv_batch(d_gb,(uint8_t*)wOut->d,d_core,VDD,HID,T);
  add_k<<<(int)((nH+255)/256),256>>>(d_x,d_core,(int)nH);
  if(getenv("HERMES_L26") && li==26){
    std::vector<float> hg((size_t)VDD), hc((size_t)HID), ho((size_t)HV*VHD), hx2((size_t)HID);
    cudaMemcpy(hg.data(),d_g,(size_t)VDD*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(hc.data(),d_core,(size_t)HID*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(ho.data(),d_o,(size_t)HV*VHD*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(hx2.data(),d_x,(size_t)HID*4,cudaMemcpyDeviceToHost);
    FILE *ff=fopen("/tmp/l26_dec.bin","ab"); fwrite(hg.data(),4,hg.size(),ff); fwrite(hc.data(),4,hc.size(),ff); fwrite(ho.data(),4,ho.size(),ff); fwrite(hx2.data(),4,hx2.size(),ff); fclose(ff);
  }
  rms_f32_k<<<T,512,(size_t)512*4>>>(d_h,d_x,(float*)wPost->d,T,HID);
  f32_to_bf16_k<<<(int)((nH+255)/256),256>>>(d_h2b,d_h,(int)nH);
  gemv_batch(d_h2b,(uint8_t*)wFg->d,d_gg,HID,FFN,T);
  gemv_batch(d_h2b,(uint8_t*)wFu->d,d_uu,HID,FFN,T);
  silu_mul_k<<<(int)(((size_t)FFN+255)/256),256>>>(d_gg,d_uu,d_hid,(int)((size_t)FFN));
  f32_to_bf16_k<<<(int)(((size_t)FFN+255)/256),256>>>(d_hidb,d_hid,(int)((size_t)FFN));
  gemv_batch(d_hidb,(uint8_t*)wFd->d,d_down,FFN,HID,T);
  add_k<<<(int)((nH+255)/256),256>>>(d_x,d_down,(int)nH);
  cudaDeviceSynchronize();
  // P6: nessun free (arena persistente ds.ws0/ws1).
  return true;
}

static bool run_full_decode_layer(int li, float* d_x, DecodeState& ds, long pos, std::string& err){
  char Pb[16]; snprintf(Pb,16,"blk.%d.",li); std::string P=Pb;
  WEntry *wNorm=find_w(P+"attn_norm.weight"),
         *wQ=find_w(P+"attn_q.weight"), *wK=find_w(P+"attn_k.weight"), *wV=find_w(P+"attn_v.weight"),
         *wQn=find_w(P+"attn_q_norm.weight"), *wKn=find_w(P+"attn_k_norm.weight"),
         *wO=find_w(P+"attn_output.weight"),
         *wPost=find_w(P+"post_attention_norm.weight"),
         *wFg=find_w(P+"ffn_gate.weight"), *wFu=find_w(P+"ffn_up.weight"), *wFd=find_w(P+"ffn_down.weight");
  if(!wNorm||!wQ||!wK||!wV||!wQn||!wKn||!wO||!wPost||!wFg||!wFu||!wFd){ err="missing w full dec "+P; return false; }
  if(!ds.Kc[li]||!ds.Vc[li]){ err="decode state null full "+P; return false; }
  int L=ds.klen[li];
  if(ds.capacity<1||ds.capacity>DecodeState::MAX_CONTEXT){ err="decode capacity OOB "+P; return false; }
  if(L<0||L>=ds.capacity){ err="klen OOB "+P; return false; }
  if((long)L!=pos){ err="klen/pos mismatch "+P+" klen="+std::to_string(L)+" pos="+std::to_string(pos); return false; }
  const int T=1; size_t nH=(size_t)T*HID;
  const int NQQ=12288, NKK=1024, NQHD=6144;
  void* wsbase = (li<=SPLIT_LAST0)? ds.ws0 : ds.ws1;
  if(!wsbase){ err="decode ws null full "+P; return false; }
  // P6: carve arena persistente (stesso ordine/tipi dei vecchi malloc, nessun malloc/free).
  uint8_t* ws = (uint8_t*)wsbase;
  float*    d_h    = (float*)(ws+0);        // 20480
  uint16_t* d_hb   = (uint16_t*)(ws+20480); // 10240
  float*    d_qq   = (float*)(ws+30720);    // 49152
  float*    d_kk   = (float*)(ws+79872);    // 4096
  float*    d_vv   = (float*)(ws+83968);    // 4096
  float*    d_Q    = (float*)(ws+88064);    // 24576
  float*    d_K1   = (float*)(ws+112640);   // 4096
  float*    d_V1   = (float*)(ws+116736);   // 4096
  float*    d_G    = (float*)(ws+120832);   // 24576
  float*    d_att  = (float*)(ws+145408);   // 24576
  float*    d_core = (float*)(ws+169984);   // 20480
  float*    d_gg   = (float*)(ws+190464);   // 69632
  float*    d_uu   = (float*)(ws+260096);   // 69632
  float*    d_hid  = (float*)(ws+329728);   // 69632
  float*    d_down = (float*)(ws+399360);   // 20480
  uint16_t* d_h2b  = (uint16_t*)(ws+419840);// 10240
  uint16_t* d_hidb = (uint16_t*)(ws+430080);// 34816
  uint16_t* d_gb   = (uint16_t*)(ws+464896);// 12288 (fine 477184 < 568064)
  rms_f32_k<<<T,512,(size_t)512*4>>>(d_h,d_x,(float*)wNorm->d,T,HID);
  f32_to_bf16_k<<<(int)((nH+255)/256),256>>>(d_hb,d_h,(int)nH);
  gemv_batch(d_hb,(uint8_t*)wQ->d,d_qq,HID,NQQ,T);
  gemv_batch(d_hb,(uint8_t*)wK->d,d_kk,HID,NKK,T);
  gemv_batch(d_hb,(uint8_t*)wV->d,d_vv,HID,NKK,T);
  q_norm_rope_pos_k<<<NQ,256>>>(d_qq,(float*)wQn->d,d_Q,(int)pos);
  k_norm_rope_pos_k<<<NKV,256>>>(d_kk,(float*)wKn->d,d_K1,(int)pos);
  extract_G_k<<<(int)((NQHD+255)/256),256>>>(d_qq,d_G,NQHD);
  cudaMemcpy(ds.Kc[li]+(size_t)L*1024,d_K1,(size_t)NKK*4,cudaMemcpyDeviceToDevice);
  cudaMemcpy(ds.Vc[li]+(size_t)L*1024,d_vv,(size_t)NKK*4,cudaMemcpyDeviceToDevice);
  sdpa_gqa_decode_k<<<NQ,256,(size_t)(L+1+256)*sizeof(float)>>>(d_Q,ds.Kc[li],ds.Vc[li],d_att,L);
  gate_k<<<(int)(((size_t)NQHD+255)/256),256>>>(d_att,d_G,d_G,(int)((size_t)NQHD));
  f32_to_bf16_k<<<(int)(((size_t)NQHD+255)/256),256>>>(d_gb,d_G,(int)((size_t)NQHD));
  gemv_batch(d_gb,(uint8_t*)wO->d,d_core,NQHD,HID,T);
  add_k<<<(int)((nH+255)/256),256>>>(d_x,d_core,(int)nH);
  rms_f32_k<<<T,512,(size_t)512*4>>>(d_h,d_x,(float*)wPost->d,T,HID);
  f32_to_bf16_k<<<(int)((nH+255)/256),256>>>(d_h2b,d_h,(int)nH);
  gemv_batch(d_h2b,(uint8_t*)wFg->d,d_gg,HID,FFN,T);
  gemv_batch(d_h2b,(uint8_t*)wFu->d,d_uu,HID,FFN,T);
  silu_mul_k<<<(int)(((size_t)FFN+255)/256),256>>>(d_gg,d_uu,d_hid,(int)((size_t)FFN));
  f32_to_bf16_k<<<(int)(((size_t)FFN+255)/256),256>>>(d_hidb,d_hid,(int)((size_t)FFN));
  gemv_batch(d_hidb,(uint8_t*)wFd->d,d_down,FFN,HID,T);
  add_k<<<(int)((nH+255)/256),256>>>(d_x,d_down,(int)nH);
  cudaDeviceSynchronize();
  ds.klen[li]=L+1;
  // P6: nessun free (arena persistente ds.ws0/ws1).
  return true;
}

bool decode_init(DecodeState& ds, std::string& err){
  return decode_init(ds,err,DecodeState::MAXL);
}

bool decode_init(DecodeState& ds, std::string& err, int capacity){
  if(ds.inited){ err="decode state already inited"; return false; }
  if(capacity<1||capacity>DecodeState::MAX_CONTEXT){
    err="decode capacity must be 1.."+std::to_string(DecodeState::MAX_CONTEXT); return false;
  }
  if(!g_up){ err="not uploaded"; return false; }
  for(int i=0;i<64;++i){ ds.S[i]=nullptr; ds.CS[i]=nullptr; ds.Kc[i]=nullptr; ds.Vc[i]=nullptr; ds.klen[i]=0; }
  ds.ws0=nullptr; ds.ws1=nullptr; ds.idsbuf=nullptr; ds.x0buf=nullptr;
  ds.x1buf=nullptr; ds.xnbuf=nullptr; ds.logbuf=nullptr; ds.ws_bytes=0;

  const size_t kv_bytes=(size_t)capacity*1024*sizeof(float);
  size_t required[2]={0,0};
  for(int li=0;li<NL;++li){
    const int dev=(li<=SPLIT_LAST0)?0:1;
    if(li%4==3) required[dev]+=2*kv_bytes;
    else required[dev]+=(size_t)HV*KHD*VHD*sizeof(float)+(size_t)CC*3*sizeof(float);
  }
  required[0]+=WS_BYTES+sizeof(int)+WS_X_BYTES;
  required[1]+=WS_BYTES+2*WS_X_BYTES+WS_LOG_BYTES;
  const size_t limit=15ULL*1024*1024*1024;
  for(int dev=0;dev<2;++dev){
    size_t free_bytes=0,total_bytes=0;
    cudaError_t e=cudaSetDevice(dev);
    if(e==cudaSuccess) e=cudaMemGetInfo(&free_bytes,&total_bytes);
    if(e!=cudaSuccess){ err="decode VRAM preflight GPU"+std::to_string(dev)+": "+cudaGetErrorString(e); return false; }
    const size_t used=total_bytes-free_bytes;
    if(used>limit||required[dev]>limit-used||required[dev]>free_bytes){
      err="decode VRAM preflight GPU"+std::to_string(dev)+" required="+std::to_string(required[dev])+" free="+std::to_string(free_bytes)+" used="+std::to_string(used)+" limit="+std::to_string(limit);
      return false;
    }
  }

  ds.capacity=capacity;
  ds.inited=true;
  auto fail=[&](){ decode_free(ds); return false; };
  for(int li=0;li<NL;++li){
    const int dev=(li<=SPLIT_LAST0)?0:1;
    if(cudaSetDevice(dev)!=cudaSuccess){ err="cudaSetDevice li="+std::to_string(li); return fail(); }
    if(li%4==3){
      if(cudaMalloc(&ds.Kc[li],kv_bytes)!=cudaSuccess){ err="OOM Kc li="+std::to_string(li); return fail(); }
      if(cudaMalloc(&ds.Vc[li],kv_bytes)!=cudaSuccess){ err="OOM Vc li="+std::to_string(li); return fail(); }
      if(cudaMemset(ds.Kc[li],0,kv_bytes)!=cudaSuccess||cudaMemset(ds.Vc[li],0,kv_bytes)!=cudaSuccess){ err="zero KV li="+std::to_string(li); return fail(); }
    } else {
      const size_t sbytes=(size_t)HV*KHD*VHD*sizeof(float);
      const size_t cbytes=(size_t)CC*3*sizeof(float);
      if(cudaMalloc(&ds.S[li],sbytes)!=cudaSuccess){ err="OOM S li="+std::to_string(li); return fail(); }
      if(cudaMalloc(&ds.CS[li],cbytes)!=cudaSuccess){ err="OOM CS li="+std::to_string(li); return fail(); }
      if(cudaMemset(ds.S[li],0,sbytes)!=cudaSuccess||cudaMemset(ds.CS[li],0,cbytes)!=cudaSuccess){ err="zero GDN state li="+std::to_string(li); return fail(); }
    }
  }
  // P6: arene persistenti + buffer step (1x alloc, mai per-layer/per-step).
  cudaError_t e=cudaSetDevice(0);
  if(e==cudaSuccess) e=cudaMalloc(&ds.ws0,WS_BYTES);
  if(e==cudaSuccess) e=cudaMalloc(&ds.idsbuf,sizeof(int));
  if(e==cudaSuccess) e=cudaMalloc(&ds.x0buf,WS_X_BYTES);
  if(e==cudaSuccess) e=cudaMemset(ds.ws0,0,WS_BYTES);
  if(e!=cudaSuccess){ err="decode GPU0 workspace: "+std::string(cudaGetErrorString(e)); return fail(); }
  e=cudaSetDevice(1);
  if(e==cudaSuccess) e=cudaMalloc(&ds.ws1,WS_BYTES);
  if(e==cudaSuccess) e=cudaMalloc(&ds.x1buf,WS_X_BYTES);
  if(e==cudaSuccess) e=cudaMalloc(&ds.xnbuf,WS_X_BYTES);
  if(e==cudaSuccess) e=cudaMalloc(&ds.logbuf,WS_LOG_BYTES);
  if(e==cudaSuccess) e=cudaMemset(ds.ws1,0,WS_BYTES);
  if(e!=cudaSuccess){ err="decode GPU1 workspace: "+std::string(cudaGetErrorString(e)); return fail(); }
  ds.ws_bytes=WS_BYTES;

  size_t used_after[2]={0,0};
  for(int dev=0;dev<2;++dev){
    size_t free_bytes=0,total_bytes=0;
    e=cudaSetDevice(dev);
    if(e==cudaSuccess) e=cudaMemGetInfo(&free_bytes,&total_bytes);
    if(e!=cudaSuccess){ err="decode VRAM check GPU"+std::to_string(dev)+": "+cudaGetErrorString(e); return fail(); }
    const size_t used=total_bytes-free_bytes;
    if(used>limit){ err="VRAM>15GB after decode_init GPU"+std::to_string(dev)+" used="+std::to_string(used); return fail(); }
    used_after[dev]=used;
  }
  printf("pp2 decode_init capacity=%d used0=%zu MiB used1=%zu MiB (limit 15360 MiB)\n",
    capacity,used_after[0]/1048576,used_after[1]/1048576); fflush(stdout);
  return true;
}

void decode_free(DecodeState& ds){
  if(!ds.inited) return;
  for(int li=0;li<NL;++li){
    int dev=(li<=SPLIT_LAST0)?0:1;
    cudaSetDevice(dev);
    if(ds.S[li]){ cudaFree(ds.S[li]); ds.S[li]=nullptr; }
    if(ds.CS[li]){ cudaFree(ds.CS[li]); ds.CS[li]=nullptr; }
    if(ds.Kc[li]){ cudaFree(ds.Kc[li]); ds.Kc[li]=nullptr; }
    if(ds.Vc[li]){ cudaFree(ds.Vc[li]); ds.Vc[li]=nullptr; }
    ds.klen[li]=0;
  }
  // P6: rilascia arene + buffer step persistenti.
  cudaSetDevice(0);
  if(ds.ws0){ cudaFree(ds.ws0); ds.ws0=nullptr; }
  if(ds.idsbuf){ cudaFree(ds.idsbuf); ds.idsbuf=nullptr; }
  if(ds.x0buf){ cudaFree(ds.x0buf); ds.x0buf=nullptr; }
  cudaSetDevice(1);
  if(ds.ws1){ cudaFree(ds.ws1); ds.ws1=nullptr; }
  if(ds.x1buf){ cudaFree(ds.x1buf); ds.x1buf=nullptr; }
  if(ds.xnbuf){ cudaFree(ds.xnbuf); ds.xnbuf=nullptr; }
  if(ds.logbuf){ cudaFree(ds.logbuf); ds.logbuf=nullptr; }
  ds.ws_bytes=0;
  ds.capacity=DecodeState::MAXL;
  ds.inited=false;
}

std::vector<float> decode_step(int tok, DecodeState& ds, long pos, std::string& err){
  if(!g_up){ err="not uploaded"; return {}; }
  if(!ds.inited){ err="decode not inited"; return {}; }
  if(ds.capacity<1||ds.capacity>DecodeState::MAX_CONTEXT){ err="decode capacity OOB"; return {}; }
  if(pos<0||pos>=ds.capacity){ err="pos OOB"; return {}; }
  const int T=1; size_t nH=(size_t)T*HID;
  // P6: zero malloc/free — buffer persistenti da decode_init.
  if(!ds.ws0||!ds.ws1||!ds.idsbuf||!ds.x0buf||!ds.x1buf||!ds.xnbuf||!ds.logbuf){
    err="decode ws not inited"; return {};
  }
  cudaSetDevice(0);
  int* d_ids = ds.idsbuf;
  float* d_x0 = ds.x0buf;
  cudaMemcpy(d_ids,&tok,4,cudaMemcpyHostToDevice);
  {
    auto it=g_w0.find("token_embd.weight");
    if(it==g_w0.end()){ err="no emb"; return {}; }
    embed_gather_k<<<(int)((nH+255)/256),256>>>(d_x0,(uint16_t*)it->second.d,d_ids,T);
    cudaDeviceSynchronize();
  }
  for(int li=0;li<=SPLIT_LAST0;++li){
    bool is_full=(li%4==3);
    bool ok=is_full? run_full_decode_layer(li,d_x0,ds,pos,err) : run_gdn_decode_layer(li,d_x0,ds,err);
    if(!ok){ return {}; }
  }
  // P5: xfer atomico sotto lock per batch parallelo (stesso pinned condiviso).
  // Compute GPU0 sopra resta fuori dal lock -> overlap reale con GPU1 di altro thread.
  float* d_x1 = ds.x1buf;
  {
    std::lock_guard<std::mutex> lk(g_xfer_mu);
    cudaMemcpy(g_pinned,d_x0,nH*4,cudaMemcpyDeviceToHost);
    cudaSetDevice(1);
    cudaMemcpy(d_x1,g_pinned,nH*4,cudaMemcpyHostToDevice);
  }
  for(int li=SPLIT_LAST0+1;li<NL;++li){
    bool is_full=(li%4==3);
    bool ok=is_full? run_full_decode_layer(li,d_x1,ds,pos,err) : run_gdn_decode_layer(li,d_x1,ds,err);
    if(!ok){ return {}; }
  }
  {
    auto itn=g_w1.find("output_norm.weight");
    auto itw=g_w1.find("output.weight");
    if(itn==g_w1.end()||itw==g_w1.end()){ err="no head"; return {}; }
    float* d_xn = ds.xnbuf;
    rms_f32_k<<<T,512,(size_t)512*4>>>(d_xn,d_x1,(float*)itn->second.d,T,HID);
    float* d_log = ds.logbuf;
    head_bf16_k<<<(int)(((size_t)VOC+255)/256),256>>>(d_log,d_xn,(uint16_t*)itw->second.d,T);
    cudaDeviceSynchronize();
    std::vector<float> logits((size_t)VOC);
    cudaMemcpy(logits.data(),d_log,(size_t)VOC*4,cudaMemcpyDeviceToHost);
    // P6: nessun free (persistenti).
    return logits;
  }
}

int debug_bisect_layer(const std::vector<int>& ids){
  int T=(int)ids.size();
  if(T<=0||T>DecodeState::MAXL){ printf("debug_bisect bad T=%d\n",T); return -2; }
  if(!g_up){ printf("debug_bisect not uploaded\n"); return -2; }
  std::string err;
  size_t nH=(size_t)T*HID;
  // ---- prefill path con salvataggio ultimo token per layer ----
  std::vector<std::vector<float>> pre_hid(NL);
  {
    cudaSetDevice(0);
    int* d_ids=nullptr; float* d_x0=nullptr;
    cudaMalloc(&d_ids,T*4); cudaMemcpy(d_ids,ids.data(),T*4,cudaMemcpyHostToDevice);
    cudaMalloc(&d_x0,nH*4);
    {
      auto it=g_w0.find("token_embd.weight");
      embed_gather_k<<<(int)((nH+255)/256),256>>>(d_x0,(uint16_t*)it->second.d,d_ids,T);
      cudaDeviceSynchronize();
    }
    cudaFree(d_ids);
    for(int li=0;li<=SPLIT_LAST0;++li){
      bool is_full=(li%4==3);
      bool ok=is_full? run_full_layer(li,d_x0,T,err) : run_gdn_layer(li,d_x0,T,err);
      if(!ok){ printf("debug pre gpu0 li=%d FAIL %s\n",li,err.c_str()); cudaFree(d_x0); return -2; }
      std::vector<float> hx(nH); cudaMemcpy(hx.data(),d_x0,nH*4,cudaMemcpyDeviceToHost);
      pre_hid[li].assign(HID,0.f);
      for(int i=0;i<HID;++i) pre_hid[li][i]=hx[(size_t)(T-1)*HID+i];
    }
    cudaMemcpy(g_pinned,d_x0,nH*4,cudaMemcpyDeviceToHost);
    cudaFree(d_x0);
    cudaSetDevice(1);
    float* d_x1=nullptr; cudaMalloc(&d_x1,nH*4);
    cudaMemcpy(d_x1,g_pinned,nH*4,cudaMemcpyHostToDevice);
    for(int li=SPLIT_LAST0+1;li<NL;++li){
      bool is_full=(li%4==3);
      bool ok=is_full? run_full_layer(li,d_x1,T,err) : run_gdn_layer(li,d_x1,T,err);
      if(!ok){ printf("debug pre gpu1 li=%d FAIL %s\n",li,err.c_str()); cudaFree(d_x1); return -2; }
      std::vector<float> hx(nH); cudaMemcpy(hx.data(),d_x1,nH*4,cudaMemcpyDeviceToHost);
      pre_hid[li].assign(HID,0.f);
      for(int i=0;i<HID;++i) pre_hid[li][i]=hx[(size_t)(T-1)*HID+i];
    }
    cudaFree(d_x1);
  }
  // ---- decode path incrementale con salvataggio ultimo step per layer ----
  std::vector<std::vector<float>> dec_hid(NL);
  {
    DecodeState ds;
    if(!decode_init(ds,err)){ printf("debug dec init FAIL %s\n",err.c_str()); return -2; }
    // warmup 0..T-2
    for(int t=0;t<T-1;++t){
      auto lg=decode_step(ids[t],ds,(long)t,err);
      if(lg.empty()){ printf("debug dec warmup t=%d FAIL %s\n",t,err.c_str()); decode_free(ds); return -2; }
    }
    // ultimo step con trace per layer (replica decode_step con save)
    int tok=ids[T-1]; long pos=T-1;
    cudaSetDevice(0);
    float* d_x0=nullptr; cudaMalloc(&d_x0,(size_t)HID*4);
    {
      int* d_ids=nullptr; cudaMalloc(&d_ids,4); cudaMemcpy(d_ids,&tok,4,cudaMemcpyHostToDevice);
      auto it=g_w0.find("token_embd.weight");
      embed_gather_k<<<(int)((HID+255)/256),256>>>(d_x0,(uint16_t*)it->second.d,d_ids,1);
      cudaDeviceSynchronize(); cudaFree(d_ids);
    }
    for(int li=0;li<=SPLIT_LAST0;++li){
      bool is_full=(li%4==3);
      bool ok=is_full? run_full_decode_layer(li,d_x0,ds,pos,err) : run_gdn_decode_layer(li,d_x0,ds,err);
      if(!ok){ printf("debug dec gpu0 li=%d FAIL %s\n",li,err.c_str()); cudaFree(d_x0); decode_free(ds); return -2; }
      std::vector<float> hx(HID); cudaMemcpy(hx.data(),d_x0,(size_t)HID*4,cudaMemcpyDeviceToHost);
      dec_hid[li]=hx;
    }
    cudaMemcpy(g_pinned,d_x0,(size_t)HID*4,cudaMemcpyDeviceToHost);
    cudaFree(d_x0);
    cudaSetDevice(1);
    float* d_x1=nullptr; cudaMalloc(&d_x1,(size_t)HID*4);
    cudaMemcpy(d_x1,g_pinned,(size_t)HID*4,cudaMemcpyHostToDevice);
    for(int li=SPLIT_LAST0+1;li<NL;++li){
      bool is_full=(li%4==3);
      bool ok=is_full? run_full_decode_layer(li,d_x1,ds,pos,err) : run_gdn_decode_layer(li,d_x1,ds,err);
      if(!ok){ printf("debug dec gpu1 li=%d FAIL %s\n",li,err.c_str()); cudaFree(d_x1); decode_free(ds); return -2; }
      std::vector<float> hx(HID); cudaMemcpy(hx.data(),d_x1,(size_t)HID*4,cudaMemcpyDeviceToHost);
      dec_hid[li]=hx;
    }
    cudaFree(d_x1);
    decode_free(ds);
  }
  int first=-1;
  for(int li=0;li<NL;++li){
    double mx=0,me=0; size_t im=0;
    for(int i=0;i<HID;++i){ double e=fabs((double)dec_hid[li][i]-pre_hid[li][i]); if(e>mx){mx=e;im=i;} me+=e; }
    me/=HID;
    bool is_full=(li%4==3);
    printf("debug layer %2d (%s): maxerr=%.6f at %zu meanerr=%.7f %s\n",li,is_full?"full":"gdn ",mx,im,me,mx<1e-4?"OK":"DIVERGE");
    fflush(stdout);
    if(mx>=1e-4 && first<0) first=li;
  }
  // Secondo passaggio: confronta tutti i token per i primi layer divergenti (26..27) per capire se carry da token precedenti
  if(first>=0){
    printf("debug full-token check for layers %d..%d (T=%d)\n",first,first+1,T); fflush(stdout);
    // prefill all-token hidden gia disponibile? ricostruisci da pre_hid? No, serve full. Rifai prefill salvando tutto.
    std::vector<std::vector<float>> pre_all(NL), dec_all(NL);
    // prefill all
    {
      cudaSetDevice(0);
      int* d_ids=nullptr; float* d_x0=nullptr;
      cudaMalloc(&d_ids,T*4); cudaMemcpy(d_ids,ids.data(),T*4,cudaMemcpyHostToDevice);
      cudaMalloc(&d_x0,nH*4);
      {
        auto it=g_w0.find("token_embd.weight");
        embed_gather_k<<<(int)((nH+255)/256),256>>>(d_x0,(uint16_t*)it->second.d,d_ids,T);
        cudaDeviceSynchronize();
      }
      cudaFree(d_ids);
      for(int li=0;li<=SPLIT_LAST0;++li){
        bool is_full=(li%4==3);
        bool ok=is_full? run_full_layer(li,d_x0,T,err) : run_gdn_layer(li,d_x0,T,err);
        if(!ok){ cudaFree(d_x0); return -2; }
        std::vector<float> hx(nH); cudaMemcpy(hx.data(),d_x0,nH*4,cudaMemcpyDeviceToHost);
        pre_all[li]=hx;
      }
      cudaMemcpy(g_pinned,d_x0,nH*4,cudaMemcpyDeviceToHost);
      cudaFree(d_x0);
      cudaSetDevice(1);
      float* d_x1=nullptr; cudaMalloc(&d_x1,nH*4);
      cudaMemcpy(d_x1,g_pinned,nH*4,cudaMemcpyHostToDevice);
      for(int li=SPLIT_LAST0+1;li<NL;++li){
        bool is_full=(li%4==3);
        bool ok=is_full? run_full_layer(li,d_x1,T,err) : run_gdn_layer(li,d_x1,T,err);
        if(!ok){ cudaFree(d_x1); return -2; }
        std::vector<float> hx(nH); cudaMemcpy(hx.data(),d_x1,nH*4,cudaMemcpyDeviceToHost);
        pre_all[li]=hx;
      }
      cudaFree(d_x1);
    }
    // decode all (trace ogni step)
    {
      DecodeState ds2;
      decode_init(ds2,err);
      for(int li=0;li<NL;++li) dec_all[li].assign(nH,0.f);
      for(int t=0;t<T;++t){
        int tok=ids[t]; long pos=t;
        cudaSetDevice(0);
        float* d_x0=nullptr; cudaMalloc(&d_x0,(size_t)HID*4);
        {
          int* d_ids=nullptr; cudaMalloc(&d_ids,4); cudaMemcpy(d_ids,&tok,4,cudaMemcpyHostToDevice);
          auto it=g_w0.find("token_embd.weight");
          embed_gather_k<<<(int)((HID+255)/256),256>>>(d_x0,(uint16_t*)it->second.d,d_ids,1);
          cudaDeviceSynchronize(); cudaFree(d_ids);
        }
        for(int li=0;li<=SPLIT_LAST0;++li){
          bool is_full=(li%4==3);
          bool ok=is_full? run_full_decode_layer(li,d_x0,ds2,pos,err) : run_gdn_decode_layer(li,d_x0,ds2,err);
          if(!ok){ cudaFree(d_x0); decode_free(ds2); return -2; }
          std::vector<float> hx(HID); cudaMemcpy(hx.data(),d_x0,(size_t)HID*4,cudaMemcpyDeviceToHost);
          for(int i=0;i<HID;++i) dec_all[li][(size_t)t*HID+i]=hx[i];
        }
        cudaMemcpy(g_pinned,d_x0,(size_t)HID*4,cudaMemcpyDeviceToHost);
        cudaFree(d_x0);
        cudaSetDevice(1);
        float* d_x1=nullptr; cudaMalloc(&d_x1,(size_t)HID*4);
        cudaMemcpy(d_x1,g_pinned,(size_t)HID*4,cudaMemcpyHostToDevice);
        for(int li=SPLIT_LAST0+1;li<NL;++li){
          bool is_full=(li%4==3);
          bool ok=is_full? run_full_decode_layer(li,d_x1,ds2,pos,err) : run_gdn_decode_layer(li,d_x1,ds2,err);
          if(!ok){ cudaFree(d_x1); decode_free(ds2); return -2; }
          std::vector<float> hx(HID); cudaMemcpy(hx.data(),d_x1,(size_t)HID*4,cudaMemcpyDeviceToHost);
          for(int i=0;i<HID;++i) dec_all[li][(size_t)t*HID+i]=hx[i];
        }
        cudaFree(d_x1);
      }
      decode_free(ds2);
    }
    for(int li=first;li<=first+1 && li<NL;++li){
      for(int t=0;t<T;++t){
        double mx=0,me=0; size_t im=0;
        for(int i=0;i<HID;++i){
          double e=fabs((double)dec_all[li][(size_t)t*HID+i]-pre_all[li][(size_t)t*HID+i]);
          if(e>mx){mx=e;im=i;} me+=e;
        }
        me/=HID;
        printf("debug layer %d tok %d: maxerr=%.6f at %zu mean=%.7f %s\n",li,t,mx,im,me,mx<1e-4?"OK":"DIVERGE");
        fflush(stdout);
      }
    }
  }
  return first;
}

int debug_layer26_detail(){
  if(!g_up){ printf("debug26 not uploaded\n"); return -2; }
  const int T=4; const int ids[4]={6205,161421,303,220};
  const int li=26;
  std::string err;
  size_t nH=(size_t)T*HID;
  // ---- ottieni h_in (input a layer26 per tutti i token) via prefill 0..25 ----
  std::vector<float> h_in((size_t)T*HID);
  {
    cudaSetDevice(0);
    int* d_ids=nullptr; float* d_x0=nullptr;
    cudaMalloc(&d_ids,T*4); cudaMemcpy(d_ids,ids,T*4,cudaMemcpyHostToDevice);
    cudaMalloc(&d_x0,nH*4);
    {
      auto it=g_w0.find("token_embd.weight");
      embed_gather_k<<<(int)((nH+255)/256),256>>>(d_x0,(uint16_t*)it->second.d,d_ids,T);
      cudaDeviceSynchronize();
    }
    cudaFree(d_ids);
    for(int l=0;l<26;++l){
      bool is_full=(l%4==3);
      bool ok=is_full? run_full_layer(l,d_x0,T,err) : run_gdn_layer(l,d_x0,T,err);
      if(!ok){ printf("debug26 pre-up-to-25 l=%d FAIL %s\n",l,err.c_str()); cudaFree(d_x0); return -2; }
    }
    cudaMemcpy(h_in.data(),d_x0,nH*4,cudaMemcpyDeviceToHost);
    cudaFree(d_x0);
  }
  // ---- q1 pre (batched) vs q1 dec (single) ----
  char Pb[16]; snprintf(Pb,16,"blk.%d.",li);
  std::string P=Pb;
  WEntry *wNorm=find_w(P+"attn_norm.weight"), *wQkv=find_w(P+"attn_qkv.weight"), *wC=find_w(P+"ssm_conv1d.weight");
  WEntry *wBeta=find_w(P+"ssm_beta.weight"), *wAlpha=find_w(P+"ssm_alpha.weight");
  WEntry *wA=find_w(P+"ssm_a"), *wDT=find_w(P+"ssm_dt.bias");
  if(!wNorm||!wQkv||!wC||!wBeta||!wAlpha||!wA||!wDT){ printf("debug26 missing w\n"); return -2; }
  cudaSetDevice(0);
  // rms batched -> h1_pre
  std::vector<float> h1_pre(nH), q1_pre((size_t)T*CC), cv_pre((size_t)T*CC);
  std::vector<float> q1_dec_all((size_t)T*CC), cv_dec_all((size_t)T*CC), h1_dec_all(nH);
  {
    float *d_x=nullptr,*d_h=nullptr,*d_q1=nullptr,*d_cv=nullptr;
    uint16_t *d_hb=nullptr;
    cudaMalloc(&d_x,nH*4); cudaMalloc(&d_h,nH*4); cudaMalloc(&d_hb,nH*2);
    cudaMalloc(&d_q1,(size_t)T*CC*4); cudaMalloc(&d_cv,(size_t)T*CC*4);
    cudaMemcpy(d_x,h_in.data(),nH*4,cudaMemcpyHostToDevice);
    rms_f32_k<<<T,512,(size_t)512*4>>>(d_h,d_x,(float*)wNorm->d,T,HID);
    cudaDeviceSynchronize();
    cudaMemcpy(h1_pre.data(),d_h,nH*4,cudaMemcpyDeviceToHost);
    f32_to_bf16_k<<<(int)((nH+255)/256),256>>>(d_hb,d_h,(int)nH);
    gemv_batch(d_hb,(uint8_t*)wQkv->d,d_q1,HID,CC,T);
    cudaDeviceSynchronize();
    cudaMemcpy(q1_pre.data(),d_q1,(size_t)T*CC*4,cudaMemcpyDeviceToHost);
    conv_silu_k<<<(int)(((size_t)T*CC+255)/256),256>>>((const float*)d_q1,(const float*)wC->d,d_cv,T,CC);
    cudaDeviceSynchronize();
    cudaMemcpy(cv_pre.data(),d_cv,(size_t)T*CC*4,cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_h); cudaFree(d_hb); cudaFree(d_q1); cudaFree(d_cv);
  }
  // decode incr q1/cv
  {
    float *d_CS=nullptr; cudaMalloc(&d_CS,(size_t)CC*3*4); cudaMemset(d_CS,0,(size_t)CC*3*4);
    for(int t=0;t<T;++t){
      float *d_h1=nullptr,*d_q1=nullptr,*d_cv=nullptr;
      uint16_t *d_h1b=nullptr;
      cudaMalloc(&d_h1,(size_t)HID*4); cudaMalloc(&d_h1b,(size_t)HID*2);
      cudaMalloc(&d_q1,(size_t)CC*4); cudaMalloc(&d_cv,(size_t)CC*4);
      // h per singolo token
      cudaMemcpy(d_h1,&h_in[(size_t)t*HID],(size_t)HID*4,cudaMemcpyHostToDevice);
      // rms T=1: serve d_x e d_h separati; riusa d_h1 come input? rms legge x, scrive h. Facciamo x=h_in[t], h=out
      float *d_x1=nullptr; cudaMalloc(&d_x1,(size_t)HID*4);
      cudaMemcpy(d_x1,&h_in[(size_t)t*HID],(size_t)HID*4,cudaMemcpyHostToDevice);
      rms_f32_k<<<1,512,(size_t)512*4>>>(d_h1,d_x1,(float*)wNorm->d,1,HID);
      cudaDeviceSynchronize();
      cudaMemcpy(&h1_dec_all[(size_t)t*HID],d_h1,(size_t)HID*4,cudaMemcpyDeviceToHost);
      f32_to_bf16_k<<<(int)((HID+255)/256),256>>>(d_h1b,d_h1,(int)HID);
      gemv_batch(d_h1b,(uint8_t*)wQkv->d,d_q1,HID,CC,1);
      cudaDeviceSynchronize();
      cudaMemcpy(&q1_dec_all[(size_t)t*CC],d_q1,(size_t)CC*4,cudaMemcpyDeviceToHost);
      conv_decode_k<<<(int)((CC+255)/256),256>>>(d_q1,(const float*)wC->d,d_CS,d_cv,CC);
      cudaDeviceSynchronize();
      cudaMemcpy(&cv_dec_all[(size_t)t*CC],d_cv,(size_t)CC*4,cudaMemcpyDeviceToHost);
      cudaFree(d_h1); cudaFree(d_h1b); cudaFree(d_q1); cudaFree(d_cv); cudaFree(d_x1);
    }
    cudaFree(d_CS);
  }
  for(int t=0;t<T;++t){
    double mx=0,me=0; size_t im=0;
    for(int i=0;i<HID;++i){ double e=fabs((double)h1_dec_all[(size_t)t*HID+i]-h1_pre[(size_t)t*HID+i]); if(e>mx){mx=e;im=i;} me+=e; }
    me/=HID;
    printf("debug26 li26 tok%d h1: maxerr=%.3e at %zu mean=%.2e %s\n",t,mx,im,me,mx==0.0?"BITEXACT":"DIVERGE");
    double mx1=0,me1=0; size_t im1=0;
    for(int i=0;i<CC;++i){ double e=fabs((double)q1_dec_all[(size_t)t*CC+i]-q1_pre[(size_t)t*CC+i]); if(e>mx1){mx1=e;im1=i;} me1+=e; }
    me1/=CC;
    printf("debug26 li26 tok%d q1: maxerr=%.3e at %zu mean=%.2e %s\n",t,mx1,im1,me1,mx1==0.0?"BITEXACT":"DIVERGE");
    double mx2=0,me2=0; size_t im2=0;
    for(int i=0;i<CC;++i){ double e=fabs((double)cv_dec_all[(size_t)t*CC+i]-cv_pre[(size_t)t*CC+i]); if(e>mx2){mx2=e;im2=i;} me2+=e; }
    me2/=CC;
    printf("debug26 li26 tok%d cv: maxerr=%.3e at %zu mean=%.2e %s\n",t,mx2,im2,me2,mx2==0.0?"BITEXACT":"DIVERGE");
    fflush(stdout);
  }
  // beta/ar/gl confronto tok3
  {
    std::vector<float> b_pre((size_t)T*HV), ar_pre((size_t)T*HV), beta_pre((size_t)T*HV), gl_pre((size_t)T*HV);
    std::vector<float> b_dec((size_t)T*HV), ar_dec((size_t)T*HV), beta_dec((size_t)T*HV), gl_dec((size_t)T*HV);
    float *d_h=nullptr,*d_hb=nullptr,*d_b=nullptr,*d_ar=nullptr,*d_be=nullptr,*d_gl=nullptr;
    uint16_t *d_hbu=nullptr;
    cudaMalloc(&d_h,nH*4); cudaMalloc(&d_hbu,nH*2);
    cudaMalloc(&d_b,(size_t)T*HV*4); cudaMalloc(&d_ar,(size_t)T*HV*4);
    cudaMalloc(&d_be,(size_t)T*HV*4); cudaMalloc(&d_gl,(size_t)T*HV*4);
    cudaMemcpy(d_h,h1_pre.data(),nH*4,cudaMemcpyHostToDevice);
    f32_to_bf16_k<<<(int)((nH+255)/256),256>>>(d_hbu,d_h,(int)nH);
    gemv_batch(d_hbu,(uint8_t*)wBeta->d,d_b,HID,HV,T);
    cudaDeviceSynchronize();
    cudaMemcpy(b_pre.data(),d_b,(size_t)T*HV*4,cudaMemcpyDeviceToHost);
    // ar pre via wa_gemm
    {
      float *d_hf=nullptr; cudaMalloc(&d_hf,nH*4); cudaMemcpy(d_hf,h1_pre.data(),nH*4,cudaMemcpyHostToDevice);
      wa_gemm_k<<<(int)(((size_t)T*HV+255)/256),256>>>(d_hf,(float*)wAlpha->d,d_ar,T,HV,HID);
      cudaDeviceSynchronize();
      cudaMemcpy(ar_pre.data(),d_ar,(size_t)T*HV*4,cudaMemcpyDeviceToHost);
      cudaFree(d_hf);
    }
    beta_gl_k<<<(int)(((size_t)T*HV+255)/256),256>>>(d_b,d_ar,(float*)wA->d,(float*)wDT->d,d_be,d_gl,(int)((size_t)T*HV));
    cudaDeviceSynchronize();
    cudaMemcpy(beta_pre.data(),d_be,(size_t)T*HV*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(gl_pre.data(),d_gl,(size_t)T*HV*4,cudaMemcpyDeviceToHost);
    cudaFree(d_h); cudaFree(d_hbu); cudaFree(d_b); cudaFree(d_ar); cudaFree(d_be); cudaFree(d_gl);
    // dec beta/ar/gl per tok (single)
    for(int t=0;t<T;++t){
      float *d_h1=nullptr; uint16_t *d_h1b=nullptr;
      float *d_b1=nullptr,*d_ar1=nullptr,*d_be1=nullptr,*d_gl1=nullptr;
      cudaMalloc(&d_h1,(size_t)HID*4); cudaMalloc(&d_h1b,(size_t)HID*2);
      cudaMalloc(&d_b1,(size_t)HV*4); cudaMalloc(&d_ar1,(size_t)HV*4);
      cudaMalloc(&d_be1,(size_t)HV*4); cudaMalloc(&d_gl1,(size_t)HV*4);
      cudaMemcpy(d_h1,&h1_dec_all[(size_t)t*HID],(size_t)HID*4,cudaMemcpyHostToDevice);
      f32_to_bf16_k<<<(int)((HID+255)/256),256>>>(d_h1b,d_h1,(int)HID);
      gemv_batch(d_h1b,(uint8_t*)wBeta->d,d_b1,HID,HV,1);
      cudaDeviceSynchronize();
      cudaMemcpy(&b_dec[(size_t)t*HV],d_b1,(size_t)HV*4,cudaMemcpyDeviceToHost);
      wa_gemm_k<<<1,256>>>(d_h1,(float*)wAlpha->d,d_ar1,1,HV,HID);
      cudaDeviceSynchronize();
      cudaMemcpy(&ar_dec[(size_t)t*HV],d_ar1,(size_t)HV*4,cudaMemcpyDeviceToHost);
      beta_gl_k<<<1,256>>>(d_b1,d_ar1,(float*)wA->d,(float*)wDT->d,d_be1,d_gl1,HV);
      cudaDeviceSynchronize();
      cudaMemcpy(&beta_dec[(size_t)t*HV],d_be1,(size_t)HV*4,cudaMemcpyDeviceToHost);
      cudaMemcpy(&gl_dec[(size_t)t*HV],d_gl1,(size_t)HV*4,cudaMemcpyDeviceToHost);
      cudaFree(d_h1); cudaFree(d_h1b); cudaFree(d_b1); cudaFree(d_ar1); cudaFree(d_be1); cudaFree(d_gl1);
    }
    for(int t=0;t<T;++t){
      double mx=0,me=0; size_t im=0;
      for(int i=0;i<HV;++i){ double e=fabs((double)b_dec[(size_t)t*HV+i]-b_pre[(size_t)t*HV+i]); if(e>mx){mx=e;im=i;} me+=e; }
      me/=HV;
      printf("debug26 tok%d b: maxerr=%.3e at %zu mean=%.2e %s\n",t,mx,im,me,mx==0.0?"BITEXACT":"DIVERGE");
      double mxa=0,mea=0; size_t ima=0;
      for(int i=0;i<HV;++i){ double e=fabs((double)ar_dec[(size_t)t*HV+i]-ar_pre[(size_t)t*HV+i]); if(e>mxa){mxa=e;ima=i;} mea+=e; }
      mea/=HV;
      printf("debug26 tok%d ar: maxerr=%.3e at %zu mean=%.2e %s\n",t,mxa,ima,mea,mxa==0.0?"BITEXACT":"DIVERGE");
      double mxb=0,meb=0; size_t imb=0;
      for(int i=0;i<HV;++i){ double e=fabs((double)beta_dec[(size_t)t*HV+i]-beta_pre[(size_t)t*HV+i]); if(e>mxb){mxb=e;imb=i;} meb+=e; }
      meb/=HV;
      printf("debug26 tok%d beta: maxerr=%.3e at %zu mean=%.2e %s\n",t,mxb,imb,meb,mxb==0.0?"BITEXACT":"DIVERGE");
      double mxg=0,meg=0; size_t img=0;
      for(int i=0;i<HV;++i){ double e=fabs((double)gl_dec[(size_t)t*HV+i]-gl_pre[(size_t)t*HV+i]); if(e>mxg){mxg=e;img=i;} meg+=e; }
      meg/=HV;
      printf("debug26 tok%d gl: maxerr=%.3e at %zu mean=%.2e %s\n",t,mxg,img,meg,mxg==0.0?"BITEXACT":"DIVERGE");
      fflush(stdout);
    }
  }
  // S-step bisect (chiude il buco qn/kn/v/S/o): stessi h_in, prefill-batched vs decode-carry
  {
    const size_t nq = (size_t)T*HV*KHD, nv = (size_t)T*HV*VHD;
    const size_t nS = (size_t)HV*KHD*VHD, no = (size_t)T*HV*VHD;
    std::vector<float> qn_pre(nq), kn_pre(nq), v_pre(nv), o_pre(no);
    std::vector<float> qn_dec(nq), kn_dec(nq), v_dec(nv), o_dec(no);
    std::vector<std::vector<float>> Sp(T, std::vector<float>(nS, 0.f)), Sd(T, std::vector<float>(nS, 0.f));
    float *d_x=nullptr,*d_h=nullptr,*d_q1=nullptr,*d_cv=nullptr;
    uint16_t *d_hb=nullptr;
    float *d_q=nullptr,*d_k=nullptr,*d_v=nullptr,*d_b=nullptr,*d_ar=nullptr,*d_be=nullptr,*d_gl=nullptr;
    float *d_S=nullptr,*d_o=nullptr;
    cudaMalloc(&d_x,nH*4); cudaMalloc(&d_h,nH*4); cudaMalloc(&d_hb,nH*2);
    cudaMalloc(&d_q1,(size_t)T*CC*4); cudaMalloc(&d_cv,(size_t)T*CC*4);
    cudaMalloc(&d_q,nq*4); cudaMalloc(&d_k,nq*4); cudaMalloc(&d_v,nv*4);
    cudaMalloc(&d_b,(size_t)T*HV*4); cudaMalloc(&d_ar,(size_t)T*HV*4);
    cudaMalloc(&d_be,(size_t)T*HV*4); cudaMalloc(&d_gl,(size_t)T*HV*4);
    cudaMalloc(&d_S,nS*4); cudaMalloc(&d_o,no*4);
    cudaMemcpy(d_x,h_in.data(),nH*4,cudaMemcpyHostToDevice);
    rms_f32_k<<<T,512,(size_t)512*4>>>(d_h,d_x,(float*)wNorm->d,T,HID);
    cudaDeviceSynchronize();
    f32_to_bf16_k<<<(int)((nH+255)/256),256>>>(d_hb,d_h,(int)nH);
    gemv_batch(d_hb,(uint8_t*)wQkv->d,d_q1,HID,CC,T);
    cudaDeviceSynchronize();
    conv_silu_k<<<(int)(((size_t)T*CC+255)/256),256>>>((const float*)d_q1,(const float*)wC->d,d_cv,T,CC);
    cudaDeviceSynchronize();
    { size_t mx=nq>nv?nq:nv; split_qkv_k<<<(int)((mx+255)/256),256>>>(d_cv,d_q,d_k,d_v,T); }
    cudaDeviceSynchronize();
    gemv_batch(d_hb,(uint8_t*)wBeta->d,d_b,HID,HV,T);
    cudaDeviceSynchronize();
    wa_gemm_k<<<(int)(((size_t)T*HV+255)/256),256>>>(d_h,(float*)wAlpha->d,d_ar,T,HV,HID);
    cudaDeviceSynchronize();
    beta_gl_k<<<(int)(((size_t)T*HV+255)/256),256>>>(d_b,d_ar,(float*)wA->d,(float*)wDT->d,d_be,d_gl,(int)((size_t)T*HV));
    cudaDeviceSynchronize();
    qk_norm_scale_k<<<(T*HV),128>>>(d_q,d_k,T);
    cudaDeviceSynchronize();
    cudaMemcpy(qn_pre.data(),d_q,nq*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(kn_pre.data(),d_k,nq*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(v_pre.data(),d_v,nv*4,cudaMemcpyDeviceToHost);
    cudaMemset(d_S,0,nS*4);
    for(int t=0;t<T;++t){
      gdn_step_cuda(d_q+(size_t)t*HV*KHD,d_k+(size_t)t*HV*KHD,d_v+(size_t)t*HV*VHD,
        d_be+(size_t)t*HV,d_gl+(size_t)t*HV,d_S,d_o+(size_t)t*HV*VHD,HV,KHD,VHD);
      cudaDeviceSynchronize();
      cudaMemcpy(Sp[t].data(),d_S,nS*4,cudaMemcpyDeviceToHost);
      cudaMemcpy(&o_pre[(size_t)t*HV*VHD],d_o+(size_t)t*HV*VHD,(size_t)HV*VHD*4,cudaMemcpyDeviceToHost);
    }
    cudaFree(d_x); cudaFree(d_h); cudaFree(d_hb); cudaFree(d_q1); cudaFree(d_cv);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_b); cudaFree(d_ar);
    cudaFree(d_be); cudaFree(d_gl); cudaFree(d_S); cudaFree(d_o);
    // decode-style per tok con carry
    float *d_CS=nullptr;
    cudaMalloc(&d_CS,(size_t)CC*3*4); cudaMemset(d_CS,0,(size_t)CC*3*4);
    float *d_S2=nullptr; cudaMalloc(&d_S2,nS*4); cudaMemset(d_S2,0,nS*4);
    for(int t=0;t<T;++t){
      float *d_h1=nullptr,*d_x1=nullptr; uint16_t *d_h1b=nullptr;
      float *d_q1b=nullptr,*d_cvb=nullptr,*d_qb=nullptr,*d_kb=nullptr,*d_vb=nullptr;
      float *d_bb=nullptr,*d_arb=nullptr,*d_beb=nullptr,*d_glb=nullptr,*d_ob=nullptr;
      const size_t nq1=(size_t)HV*KHD, nv1=(size_t)HV*VHD, no1=(size_t)HV*VHD;
      cudaMalloc(&d_h1,(size_t)HID*4); cudaMalloc(&d_x1,(size_t)HID*4); cudaMalloc(&d_h1b,(size_t)HID*2);
      cudaMalloc(&d_q1b,(size_t)CC*4); cudaMalloc(&d_cvb,(size_t)CC*4);
      cudaMalloc(&d_qb,nq1*4); cudaMalloc(&d_kb,nq1*4); cudaMalloc(&d_vb,nv1*4);
      cudaMalloc(&d_bb,(size_t)HV*4); cudaMalloc(&d_arb,(size_t)HV*4);
      cudaMalloc(&d_beb,(size_t)HV*4); cudaMalloc(&d_glb,(size_t)HV*4); cudaMalloc(&d_ob,no1*4);
      cudaMemcpy(d_x1,&h_in[(size_t)t*HID],(size_t)HID*4,cudaMemcpyHostToDevice);
      rms_f32_k<<<1,512,(size_t)512*4>>>(d_h1,d_x1,(float*)wNorm->d,1,HID);
      cudaDeviceSynchronize();
      f32_to_bf16_k<<<(int)((HID+255)/256),256>>>(d_h1b,d_h1,(int)HID);
      gemv_batch(d_h1b,(uint8_t*)wQkv->d,d_q1b,HID,CC,1);
      cudaDeviceSynchronize();
      conv_decode_k<<<(int)((CC+255)/256),256>>>(d_q1b,(const float*)wC->d,d_CS,d_cvb,CC);
      cudaDeviceSynchronize();
      split_qkv_k<<<(int)((nq1+255)/256),256>>>(d_cvb,d_qb,d_kb,d_vb,1);
      cudaDeviceSynchronize();
      gemv_batch(d_h1b,(uint8_t*)wBeta->d,d_bb,HID,HV,1);
      cudaDeviceSynchronize();
      wa_gemm_k<<<1,256>>>(d_h1,(float*)wAlpha->d,d_arb,1,HV,HID);
      cudaDeviceSynchronize();
      beta_gl_k<<<1,256>>>(d_bb,d_arb,(float*)wA->d,(float*)wDT->d,d_beb,d_glb,HV);
      cudaDeviceSynchronize();
      qk_norm_scale_k<<<HV,128>>>(d_qb,d_kb,1);
      cudaDeviceSynchronize();
      cudaMemcpy(&qn_dec[(size_t)t*HV*KHD],d_qb,nq1*4,cudaMemcpyDeviceToHost);
      cudaMemcpy(&kn_dec[(size_t)t*HV*KHD],d_kb,nq1*4,cudaMemcpyDeviceToHost);
      cudaMemcpy(&v_dec[(size_t)t*HV*VHD],d_vb,nv1*4,cudaMemcpyDeviceToHost);
      gdn_step_cuda(d_qb,d_kb,d_vb,d_beb,d_glb,d_S2,d_ob,HV,KHD,VHD);
      cudaDeviceSynchronize();
      cudaMemcpy(Sd[t].data(),d_S2,nS*4,cudaMemcpyDeviceToHost);
      cudaMemcpy(&o_dec[(size_t)t*HV*VHD],d_ob,no1*4,cudaMemcpyDeviceToHost);
      cudaFree(d_h1); cudaFree(d_x1); cudaFree(d_h1b); cudaFree(d_q1b); cudaFree(d_cvb);
      cudaFree(d_qb); cudaFree(d_kb); cudaFree(d_vb); cudaFree(d_bb); cudaFree(d_arb);
      cudaFree(d_beb); cudaFree(d_glb); cudaFree(d_ob);
    }
    cudaFree(d_CS); cudaFree(d_S2);
    const char* nm[6]={"qn","kn","v","S","o","z"};
    for(int k=0;k<5;++k){
      size_t nn = (k<2)?nq:((k==2)?nv:((k==3)?nS:no));
      const float *A = (k==0)?qn_pre.data():((k==1)?kn_pre.data():((k==2)?v_pre.data():((k==3)?Sp[T-1].data():o_pre.data())));
      const float *B = (k==0)?qn_dec.data():((k==1)?kn_dec.data():((k==2)?v_dec.data():((k==3)?Sd[T-1].data():o_dec.data())));
      (void)nm;
      for(int t=0;t<T;++t){
        const float *At, *Bt;
        size_t per = (k<2)?(size_t)HV*KHD:((k==2)?(size_t)HV*VHD:((k==3)?nS:(size_t)HV*VHD));
        if(k==3){ At=Sp[t].data(); Bt=Sd[t].data(); }
        else { At=A+(size_t)t*per; Bt=B+(size_t)t*per; }
        double mx=0,me=0; size_t im=0; size_t nnz=0;
        for(size_t i=0;i<per;++i){ double e=fabs((double)At[i]-(double)Bt[i]); if(e>mx){mx=e;im=i;} if(e!=0.0)++nnz; me+=e; }
        me/=per;
        printf("sbis %s tok%d: maxerr=%.3e at %zu mean=%.2e nondiff=%zu/%zu %s\n",nm[k],t,mx,im,me,nnz,per,(mx==0.0)?"BITEXACT":"DIVERGE");
        fflush(stdout);
      }
    }
  }
  // zl bisect: gate GEMM prefill-batched vs decode-single (h1 ricostr.)
  {
    WEntry *wG=find_w("blk.26.attn_gate.weight");
    if(!wG){ printf("sbis zl: missing wGate\n"); }
    else {
    const size_t nzl=(size_t)T*VDD;
    std::vector<float> zl_pre(nzl), zl_dec(nzl);
    float *d_h=nullptr,*d_h1=nullptr; uint16_t *d_hb=nullptr; float *d_zl=nullptr;
    cudaMalloc(&d_h,nH*4); cudaMalloc(&d_h1,nH*4); cudaMalloc(&d_hb,nH*2); cudaMalloc(&d_zl,nzl*4);
    cudaMemcpy(d_h,h_in.data(),nH*4,cudaMemcpyHostToDevice);
    rms_f32_k<<<T,512,(size_t)512*4>>>(d_h1,d_h,(float*)wNorm->d,T,HID);
    cudaDeviceSynchronize();
    f32_to_bf16_k<<<(int)((nH+255)/256),256>>>(d_hb,d_h1,(int)nH);
    gemv_batch(d_hb,(uint8_t*)wG->d,d_zl,HID,VDD,T);
    cudaDeviceSynchronize();
    cudaMemcpy(zl_pre.data(),d_zl,nzl*4,cudaMemcpyDeviceToHost);
    cudaFree(d_h); cudaFree(d_h1); cudaFree(d_hb); cudaFree(d_zl);
    for(int t=0;t<T;++t){
      float *d_x1=nullptr,*d_h1b=nullptr; uint16_t *d_h1c=nullptr; float *d_z1=nullptr;
      cudaMalloc(&d_x1,(size_t)HID*4); cudaMalloc(&d_h1b,(size_t)HID*4); cudaMalloc(&d_h1c,(size_t)HID*2);
      cudaMalloc(&d_z1,(size_t)VDD*4);
      cudaMemcpy(d_x1,&h_in[(size_t)t*HID],(size_t)HID*4,cudaMemcpyHostToDevice);
      rms_f32_k<<<1,512,(size_t)512*4>>>(d_h1b,d_x1,(float*)wNorm->d,1,HID);
      cudaDeviceSynchronize();
      f32_to_bf16_k<<<(int)((HID+255)/256),256>>>(d_h1c,d_h1b,(int)HID);
      gemv_batch(d_h1c,(uint8_t*)wG->d,d_z1,HID,VDD,1);
      cudaDeviceSynchronize();
      cudaMemcpy(&zl_dec[(size_t)t*VDD],d_z1,(size_t)VDD*4,cudaMemcpyDeviceToHost);
      cudaFree(d_x1); cudaFree(d_h1b); cudaFree(d_h1c); cudaFree(d_z1);
    }
    for(int t=0;t<T;++t){
      double mx=0,me=0; size_t im=0;
      for(int i=0;i<VDD;++i){ double e=fabs((double)zl_pre[(size_t)t*VDD+i]-zl_dec[(size_t)t*VDD+i]); if(e>mx){mx=e;im=i;} me+=e; }
      me/=VDD;
      printf("sbis zl tok%d: maxerr=%.3e at %zu mean=%.2e %s\n",t,mx,im,me,mx==0.0?"BITEXACT":"DIVERGE");
      fflush(stdout);
    }
    }
  }
  // full layer 26 batched vs incr con stessi h_in
  {
    // prefill-style full layer
    float *d_xp=nullptr; cudaMalloc(&d_xp,nH*4);
    cudaMemcpy(d_xp,h_in.data(),nH*4,cudaMemcpyHostToDevice);
    if(!run_gdn_layer(26,d_xp,T,err)){ printf("debug26 run_gdn_layer FAIL %s\n",err.c_str()); cudaFree(d_xp); return -2; }
    std::vector<float> h_out_pre(nH); cudaMemcpy(h_out_pre.data(),d_xp,nH*4,cudaMemcpyDeviceToHost);
    cudaFree(d_xp);
    // decode-style incr
    DecodeState ds26; decode_init(ds26,err);
    cudaSetDevice(0);
    // azzera tutto tranne layer26 per isolare? usa ds26 completo ma tanto altri layer non usati
    std::vector<float> h_out_dec(nH);
    for(int t=0;t<T;++t){
      cudaSetDevice(0);
      float *d_xs=nullptr; cudaMalloc(&d_xs,(size_t)HID*4);
      cudaMemcpy(d_xs,&h_in[(size_t)t*HID],(size_t)HID*4,cudaMemcpyHostToDevice);
      if(!run_gdn_decode_layer(26,d_xs,ds26,err)){ printf("debug26 dec layer FAIL t=%d %s\n",t,err.c_str()); cudaFree(d_xs); decode_free(ds26); return -2; }
      cudaMemcpy(&h_out_dec[(size_t)t*HID],d_xs,(size_t)HID*4,cudaMemcpyDeviceToHost);
      cudaFree(d_xs);
    }
    decode_free(ds26);
    for(int t=0;t<T;++t){
      double mx=0,me=0; size_t im=0;
      for(int i=0;i<HID;++i){ double e=fabs((double)h_out_dec[(size_t)t*HID+i]-h_out_pre[(size_t)t*HID+i]); if(e>mx){mx=e;im=i;} me+=e; }
      me/=HID;
      printf("debug26 full layer26 tok%d hout: maxerr=%.6f at %zu mean=%.7f %s\n",t,mx,im,me,mx<1e-4?"OK":"DIVERGE");
      fflush(stdout);
    }
  }
  return 0;
}

} // namespace pp2
