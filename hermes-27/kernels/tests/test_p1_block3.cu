// P1 blk.3 full-attention su GPU1 con pesi REALI, verificato vs golden CPU.
// Dims: HID=5120 NQ=24 NKV=4 HD=256 T=11. Goldens /tmp/e3_*.
// s1: q/k/v GEMM NVFP4 (CPU-dequant exact + GEMV preciso + W4A4 rumoroso)
// s2: q/k RMSNorm diretto + split interleaved + RoPE split-half -> vs Q/K post-rope
// s3: SDPA GQA x6 causale -> vs att; gate sigmoid + out_proj GEMV -> core vs e3_core
#include "hermes/weights.hpp"
#include "hermes/gguf.hpp"
#include "hermes/rmsnorm.hpp"
#include "hermes/nvfp4_gemm.hpp"
#include "hermes/nvfp4_gemv.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <cuda_runtime.h>

static const char* CKPT = "/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf";
static const char* F_H = "/tmp/e3_h.bin";
static const char* F_QQKV = "/tmp/e3_qqkv.bin";
static const char* F_KK = "/tmp/e3_kk.bin";
static const char* F_VV = "/tmp/e3_vv.bin";
static const char* F_Q = "/tmp/e3_Q.bin";
static const char* F_K = "/tmp/e3_K.bin";
static const char* F_V = "/tmp/e3_V.bin";
static const char* F_G = "/tmp/e3_G.bin";
static const char* F_ATT = "/tmp/e3_att.bin";
static const char* F_CORE = "/tmp/e3_core.bin";
static const int T = 11, HID = 5120;
static const int NQ = 24, NKV = 4, HD = 256;
static const int NQQ = 12288, NKK = 1024, NQHD = 6144;
static const float THETA = 10000000.f;
static const int ROT = 64;

static bool read_bin(const char* path, std::vector<float>& out, size_t expect_n) {
  FILE* f = fopen(path, "rb");
  if (!f) { printf("OPEN FAIL %s\n", path); return false; }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  if ((size_t)sz != expect_n * 4) { printf("SIZE MISMATCH %s: got %ld expect %zu\n", path, sz, expect_n*4); fclose(f); return false; }
  out.resize(expect_n);
  size_t r = fread(out.data(), 4, expect_n, f); fclose(f);
  if (r != expect_n) { printf("READ FAIL %s\n", path); return false; }
  return true;
}
static void stats(const char* tag, const float* g, const float* r, size_t n, float thr) {
  double mx=0; size_t im=0; double dot=0,na=0,nb=0; double sumr=0;
  for(size_t i=0;i<n;++i){ float e=fabsf(g[i]-r[i]); if(e>mx){mx=e;im=i;}
    dot+=(double)g[i]*(double)r[i]; na+=(double)g[i]*(double)g[i]; nb+=(double)r[i]*(double)r[i]; sumr+=fabsf(r[i]); }
  double cosv=dot/(sqrt(na*nb)+1e-12); double rms=sqrt(nb/n); double rel=mx/(rms+1e-12);
  printf("%s maxerr=%.6g at %zu gpu=%.6f ref=%.6f cos=%.8f rms=%.5f rel=%.5f meanabs=%.5f %s (thr %.g)\n",
    tag,mx,im,g[im],r[im],cosv,rms,rel,sumr/n,(mx<thr?"PASS":"FAIL"),(double)thr);
}

// ---- GPU kernels float (match model_fwd.cpp exactly) ----
__global__ void q_norm_rope_k(const float* qqkv, const float* qn, float* Q, int TT) {
  // grid = TT*24, block=256
  int bh = (int)(blockIdx.x); int t = bh / 24; int head = bh % 24; int tid = (int)threadIdx.x;
  __shared__ float sh[256]; __shared__ float sq[256];
  float v = qqkv[(size_t)t*NQQ + (size_t)head*512 + tid];
  sh[tid] = v*v; __syncthreads();
  for(int s=128;s>0;s>>=1){ if(tid<s) sh[tid]+=sh[tid+s]; __syncthreads(); }
  float inv = 1.f/sqrtf(sh[0]/256.f+1e-6f);
  float qnrm = v*inv*qn[tid];
  sq[tid]=qnrm; __syncthreads();
  float y;
  if(tid<64){
    int i = tid<32? tid : tid-32;
    float freq = 1.f/powf(THETA,(float)(2*i)/64.f);
    float ang = (float)t*freq, cc=cosf(ang), ss=sinf(ang);
    float a=sq[i], b=sq[32+i];
    y = (tid<32)? (a*cc-b*ss) : (a*ss+b*cc);
  } else y=qnrm;
  Q[((size_t)t*24+head)*256+tid]=y;
}
__global__ void k_norm_rope_k(const float* kk, const float* kn, float* K_, int TT) {
  int bh=(int)blockIdx.x; int t=bh/4; int head=bh%4; int tid=(int)threadIdx.x;
  __shared__ float sh[256]; __shared__ float sq[256];
  float v = kk[(size_t)t*NKK + (size_t)head*256 + tid];
  sh[tid]=v*v; __syncthreads();
  for(int s=128;s>0;s>>=1){ if(tid<s) sh[tid]+=sh[tid+s]; __syncthreads(); }
  float inv=1.f/sqrtf(sh[0]/256.f+1e-6f);
  float qnrm=v*inv*kn[tid];
  sq[tid]=qnrm; __syncthreads();
  float y;
  if(tid<64){
    int i=tid<32?tid:tid-32;
    float freq=1.f/powf(THETA,(float)(2*i)/64.f);
    float ang=(float)t*freq, cc=cosf(ang), ss=sinf(ang);
    float a=sq[i], b=sq[32+i];
    y=(tid<32)?(a*cc-b*ss):(a*ss+b*cc);
  } else y=qnrm;
  K_[((size_t)t*4+head)*256+tid]=y;
}
__global__ void sdpa_gqa_k(const float* Q, const float* K_, const float* V, float* ATT, int TT) {
  // grid = TT*24, block=256. one block per (t,hq)
  int b=(int)blockIdx.x; int t=b/24; int hq=b%24; int g=hq/6; int tid=(int)threadIdx.x;
  __shared__ float scores[11]; __shared__ float tmp[256];
  float q = Q[((size_t)t*24+hq)*256+tid];
  for(int t2=0;t2<=t;++t2){
    float kv = K_[((size_t)t2*4+g)*256+tid];
    tmp[tid]=q*kv; __syncthreads();
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
__global__ void gate_k(const float* att, const float* graw, float* gated, int N) {
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i>=N) return;
  float s=graw[i]; gated[i]=att[i]/(1.f+expf(-s));
}

static int run_s1(hermes::GgufWeightBackend& be) {
  printf("\n=== S1 q/k/v GEMM NVFP4 T=%d HID=%d ===\n",T,HID);
  size_t f0=0,t0=0; cudaMemGetInfo(&f0,&t0); printf("vram_before free=%zu MiB\n",f0/(1024*1024));
  std::vector<float> h, qq_ref, kk_ref, vv_ref;
  if(!read_bin(F_H,h,(size_t)T*HID)) return 1;
  if(!read_bin(F_QQKV,qq_ref,(size_t)T*NQQ)) return 1;
  if(!read_bin(F_KK,kk_ref,(size_t)T*NKK)) return 1;
  if(!read_bin(F_VV,vv_ref,(size_t)T*NKK)) return 1;
  printf("h[0..3]=%.6f %.6f %.6f %.6f\n",h[0],h[1],h[2],h[3]);
  auto vQ=be.view("blk.3.attn_q.weight");
  auto vK=be.view("blk.3.attn_k.weight");
  auto vV=be.view("blk.3.attn_v.weight");
  printf("q %s ne=[%llu,%llu] type=%d\n",vQ.info->name.c_str(),(unsigned long long)vQ.info->ne[0],(unsigned long long)vQ.info->ne[1],(int)vQ.info->type);
  printf("k %s ne=[%llu,%llu]\n",vK.info->name.c_str(),(unsigned long long)vK.info->ne[0],(unsigned long long)vK.info->ne[1]);
  printf("v %s ne=[%llu,%llu]\n",vV.info->name.c_str(),(unsigned long long)vV.info->ne[0],(unsigned long long)vV.info->ne[1]);
  if(vQ.info->ne[0]!=5120||vQ.info->ne[1]!=12288){printf("Q SHAPE MISMATCH\n");return 1;}
  if(vK.info->ne[0]!=5120||vK.info->ne[1]!=1024){printf("K SHAPE MISMATCH\n");return 1;}
  // CPU-dequant exact vs golden (isola h)
  {
    int nb=HID/64;
    auto cpu_check=[&](const uint8_t* rows,int N,const std::vector<float>& ref,const char* tag){
      std::vector<float> W((size_t)N*HID); std::vector<float> t64(64);
      for(int n=0;n<N;++n){ const uint8_t* row=rows+(size_t)n*nb*36; float* out=&W[(size_t)n*HID];
        for(int b=0;b<nb;++b){ hermes::nvfp4_block_dequant(row+b*36,t64.data()); for(int j=0;j<64;++j) out[b*64+j]=t64[j]; } }
      std::vector<float> Y((size_t)T*N,0.f);
      for(int t=0;t<T;++t) for(int n=0;n<N;++n){ double a=0;
        for(int k=0;k<HID;++k) a+=(double)h[(size_t)t*HID+k]*(double)W[(size_t)n*HID+k];
        Y[(size_t)t*N+n]=(float)a; }
      char buf[128]; snprintf(buf,128,"CPU-dequant %s vs golden",tag);
      stats(buf,Y.data(),ref.data(),Y.size(),5e-4f);
    };
    cpu_check(vQ.data,NQQ,qq_ref,"qqkv");
    cpu_check(vK.data,NKK,kk_ref,"kk");
    cpu_check(vV.data,NKK,vv_ref,"vv");
  }
  // GPU GEMV preciso (BF16 in, FP32 out, dequant on-the-fly esatta)
  std::vector<uint16_t> h_bf16((size_t)T*HID);
  for(size_t i=0;i<h.size();++i) h_bf16[i]=hermes::f32_to_bf16(h[i]);
  uint16_t* d_h=nullptr; cudaMalloc(&d_h,(size_t)T*HID*2);
  cudaMemcpy(d_h,h_bf16.data(),(size_t)T*HID*2,cudaMemcpyHostToDevice);
  auto gemv_one=[&](const uint8_t* wrows,int N,const std::vector<float>& ref,const char* tag){
    size_t wb=(size_t)N*(HID/64*36);
    uint8_t* dW=nullptr; float* dY=nullptr;
    cudaMalloc(&dW,wb); cudaMalloc(&dY,(size_t)T*N*4);
    cudaMemcpy(dW,wrows,wb,cudaMemcpyHostToDevice);
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for(int t=0;t<T;++t) hermes::nvfp4_gemv_cuda(d_h+t*HID,dW,dY+t*N,HID,N);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms=0; cudaEventElapsedTime(&ms,e0,e1);
    std::vector<float> Y((size_t)T*N); cudaMemcpy(Y.data(),dY,(size_t)T*N*4,cudaMemcpyDeviceToHost);
    char buf[128]; snprintf(buf,128,"GPU-GEMV %s vs golden",tag);
    stats(buf,Y.data(),ref.data(),Y.size(),0.15f);
    printf("  %s GEMV ms=%.3f total T=%d N=%d K=%d\n",tag,ms,T,N,HID);
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaFree(dW); cudaFree(dY);
  };
  gemv_one(vQ.data,NQQ,qq_ref,"qqkv");
  gemv_one(vK.data,NKK,kk_ref,"kk");
  gemv_one(vV.data,NKK,vv_ref,"vv");
  // GPU W4A4 rumoroso su qqkv (documenta rumore 4-bit atteso ~0.38)
  {
    int Mp=T;
    size_t sfaB=hermes::Nvfp4GemmW4A4::sfa_bytes(Mp,HID);
    size_t sfbB=hermes::Nvfp4GemmW4A4::sfb_bytes(Mp,NQQ,HID);
    size_t wsB=hermes::Nvfp4GemmW4A4::workspace_bytes(Mp,NQQ,HID);
    printf("W4A4 qqkv sfa=%zu sfb=%zu ws=%zu backend=%s\n",sfaB,sfbB,wsB,hermes::Nvfp4GemmW4A4::backend_name());
    if(sfaB&&sfbB){
      std::vector<uint16_t> Xp((size_t)Mp*HID); for(int t=0;t<T;++t) memcpy(&Xp[(size_t)t*HID],&h_bf16[(size_t)t*HID],HID*2);
      std::vector<uint8_t> Aq((size_t)Mp*HID/2,0), SFA(sfaB,0);
      hermes::Nvfp4GemmW4A4::quant_activation(Xp.data(),Aq.data(),SFA.data(),Mp,HID);
      std::vector<uint8_t> Bq((size_t)NQQ*HID/2,0), SFB(sfbB,0);
      hermes::Nvfp4GemmW4A4::build_B_from_gguf(vQ.data,Bq.data(),SFB.data(),NQQ,HID);
      uint8_t *dA=nullptr,*dSFA=nullptr,*dB=nullptr,*dSFB=nullptr; uint16_t* dD=nullptr;
      cudaMalloc(&dA,Aq.size()); cudaMalloc(&dSFA,SFA.size()); cudaMalloc(&dB,Bq.size()); cudaMalloc(&dSFB,SFB.size());
      cudaMalloc(&dD,(size_t)Mp*NQQ*2);
      cudaMemcpy(dA,Aq.data(),Aq.size(),cudaMemcpyHostToDevice);
      cudaMemcpy(dSFA,SFA.data(),SFA.size(),cudaMemcpyHostToDevice);
      cudaMemcpy(dB,Bq.data(),Bq.size(),cudaMemcpyHostToDevice);
      cudaMemcpy(dSFB,SFB.data(),SFB.size(),cudaMemcpyHostToDevice);
      hermes::Nvfp4GemmW4A4::Workspace ws{nullptr,0};
      int wrc=hermes::Nvfp4GemmW4A4::workspace_ensure(ws,Mp,NQQ,HID);
      int rc=hermes::Nvfp4GemmW4A4::run_ws(dA,dSFA,dB,dSFB,nullptr,dD,Mp,NQQ,HID,ws.ptr,ws.bytes,nullptr);
      printf("W4A4 run_ws rc=%d wrc=%d\n",rc,wrc);
      if(rc==0){
        cudaDeviceSynchronize();
        cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0); const int REPS=20;
        for(int r=0;r<REPS;++r) hermes::Nvfp4GemmW4A4::run_ws(dA,dSFA,dB,dSFB,nullptr,dD,Mp,NQQ,HID,ws.ptr,ws.bytes,nullptr);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms=0; cudaEventElapsedTime(&ms,e0,e1); ms/=REPS;
        std::vector<uint16_t> D((size_t)Mp*NQQ); cudaMemcpy(D.data(),dD,(size_t)Mp*NQQ*2,cudaMemcpyDeviceToHost);
        std::vector<float> Y((size_t)T*NQQ); for(size_t i=0;i<Y.size();++i) Y[i]=hermes::bf16_to_f32(D[i]);
        stats("GPU-W4A4 qqkv vs golden",Y.data(),qq_ref.data(),Y.size(),2.0f);
        printf("  W4A4 ms=%.4f avg Mp=%d N=%d K=%d\n",ms,Mp,NQQ,HID);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
      }
      cudaFree(dA);cudaFree(dSFA);cudaFree(dB);cudaFree(dSFB);cudaFree(dD);
      hermes::Nvfp4GemmW4A4::workspace_release(ws);
    }
  }
  cudaFree(d_h);
  size_t f1=0,t1=0; cudaMemGetInfo(&f1,&t1);
  printf("vram_after free=%zu MiB delta=%zd MiB\n",f1/(1024*1024),(long)((long)f0-(long)f1)/(1024*1024));
  return 0;
}

static int run_s2(hermes::GgufWeightBackend& be) {
  printf("\n=== S2 q/k RMSNorm+RoPE T=%d NQ=%d NKV=%d HD=%d ===\n",T,NQ,NKV,HD);
  size_t f0=0,t0=0; cudaMemGetInfo(&f0,&t0); printf("vram_before free=%zu MiB\n",f0/(1024*1024));
  std::vector<float> qq, kk, Qref, Kref;
  if(!read_bin(F_QQKV,qq,(size_t)T*NQQ)) return 1;
  if(!read_bin(F_KK,kk,(size_t)T*NKK)) return 1;
  if(!read_bin(F_Q,Qref,(size_t)T*NQHD)) return 1;
  if(!read_bin(F_K,Kref,(size_t)T*NKK)) return 1;
  auto vQn=be.view("blk.3.attn_q_norm.weight");
  auto vKn=be.view("blk.3.attn_k_norm.weight");
  printf("qn %s ne=[%llu] kqn %s\n",vQn.info->name.c_str(),(unsigned long long)vQn.info->ne[0],vKn.info->name.c_str());
  const float* qn=(const float*)vQn.data; const float* kn=(const float*)vKn.data;
  printf("qn[0..3]=%.6f %.6f %.6f %.6f kn[0..3]=%.6f %.6f %.6f %.6f\n",qn[0],qn[1],qn[2],qn[3],kn[0],kn[1],kn[2],kn[3]);
  float *d_qq=nullptr,*d_kk=nullptr,*d_qn=nullptr,*d_kn=nullptr,*d_Q=nullptr,*d_K=nullptr;
  cudaMalloc(&d_qq,(size_t)T*NQQ*4); cudaMalloc(&d_kk,(size_t)T*NKK*4);
  cudaMalloc(&d_qn,256*4); cudaMalloc(&d_kn,256*4);
  cudaMalloc(&d_Q,(size_t)T*NQHD*4); cudaMalloc(&d_K,(size_t)T*NKK*4);
  cudaMemcpy(d_qq,qq.data(),(size_t)T*NQQ*4,cudaMemcpyHostToDevice);
  cudaMemcpy(d_kk,kk.data(),(size_t)T*NKK*4,cudaMemcpyHostToDevice);
  cudaMemcpy(d_qn,qn,256*4,cudaMemcpyHostToDevice);
  cudaMemcpy(d_kn,kn,256*4,cudaMemcpyHostToDevice);
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  q_norm_rope_k<<<T*NQ,256>>>(d_qq,d_qn,d_Q,T);
  k_norm_rope_k<<<T*NKV,256>>>(d_kk,d_kn,d_K,T);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms=0; cudaEventElapsedTime(&ms,e0,e1);
  printf("q/k norm+rope kernel ms=%.4f total (Q blocks=%d K blocks=%d)\n",ms,T*NQ,T*NKV);
  // timed reps
  cudaEventRecord(e0); const int REPS=50;
  for(int r=0;r<REPS;++r){ q_norm_rope_k<<<T*NQ,256>>>(d_qq,d_qn,d_Q,T); k_norm_rope_k<<<T*NKV,256>>>(d_kk,d_kn,d_K,T); }
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float msa=0; cudaEventElapsedTime(&msa,e0,e1); msa/=REPS;
  printf("avg ms=%.4f over %d reps\n",msa,REPS);
  std::vector<float> Q((size_t)T*NQHD), K_((size_t)T*NKK);
  cudaMemcpy(Q.data(),d_Q,(size_t)T*NQHD*4,cudaMemcpyDeviceToHost);
  cudaMemcpy(K_.data(),d_K,(size_t)T*NKK*4,cudaMemcpyDeviceToHost);
  stats("GPU Q post-rope vs golden",Q.data(),Qref.data(),Q.size(),1e-4f);
  stats("GPU K post-rope vs golden",K_.data(),Kref.data(),K_.size(),1e-4f);
  printf("Qgpu[0..3]=%.6f %.6f %.6f %.6f Qref[0..3]=%.6f %.6f %.6f %.6f\n",Q[0],Q[1],Q[2],Q[3],Qref[0],Qref[1],Qref[2],Qref[3]);
  size_t f1=0,t1=0; cudaMemGetInfo(&f1,&t1);
  printf("vram_after free=%zu MiB delta=%zd KiB\n",f1/(1024*1024),(long)((long)f0-(long)f1)/1024);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  cudaFree(d_qq);cudaFree(d_kk);cudaFree(d_qn);cudaFree(d_kn);cudaFree(d_Q);cudaFree(d_K);
  return 0;
}

static int run_s3(hermes::GgufWeightBackend& be) {
  printf("\n=== S3 SDPA GQA + gate + out_proj T=%d ===\n",T);
  size_t f0=0,t0=0; cudaMemGetInfo(&f0,&t0); printf("vram_before free=%zu MiB\n",f0/(1024*1024));
  std::vector<float> Qref,Kref,Vref,att_ref,Ggated_ref,core_ref,qq;
  if(!read_bin(F_Q,Qref,(size_t)T*NQHD)) return 1;
  if(!read_bin(F_K,Kref,(size_t)T*NKK)) return 1;
  if(!read_bin(F_V,Vref,(size_t)T*NKK)) return 1;
  if(!read_bin(F_ATT,att_ref,(size_t)T*NQHD)) return 1;
  if(!read_bin(F_G,Ggated_ref,(size_t)T*NQHD)) return 1;
  if(!read_bin(F_CORE,core_ref,(size_t)T*HID)) return 1;
  if(!read_bin(F_QQKV,qq,(size_t)T*NQQ)) return 1;
  // G_raw from qqkv second half (CPU exact isolate)
  std::vector<float> Graw((size_t)T*NQHD);
  for(int t=0;t<T;++t) for(int hd_=0;hd_<NQ;++hd_) for(int d=0;d<HD;++d)
    Graw[((size_t)t*NQ+hd_)*HD+d]=qq[(size_t)t*NQQ+hd_*512+256+d];
  auto vWo=be.view("blk.3.attn_output.weight");
  printf("Wo %s ne=[%llu,%llu]\n",vWo.info->name.c_str(),(unsigned long long)vWo.info->ne[0],(unsigned long long)vWo.info->ne[1]);
  float *dQ=nullptr,*dK=nullptr,*dV=nullptr,*dA=nullptr,*dGr=nullptr,*dG=nullptr;
  cudaMalloc(&dQ,(size_t)T*NQHD*4); cudaMalloc(&dK,(size_t)T*NKK*4); cudaMalloc(&dV,(size_t)T*NKK*4);
  cudaMalloc(&dA,(size_t)T*NQHD*4); cudaMalloc(&dGr,(size_t)T*NQHD*4); cudaMalloc(&dG,(size_t)T*NQHD*4);
  // usa golden Q/K/V per isolare SDPA (come blk0: golden cv per gdn)
  cudaMemcpy(dQ,Qref.data(),(size_t)T*NQHD*4,cudaMemcpyHostToDevice);
  cudaMemcpy(dK,Kref.data(),(size_t)T*NKK*4,cudaMemcpyHostToDevice);
  cudaMemcpy(dV,Vref.data(),(size_t)T*NKK*4,cudaMemcpyHostToDevice);
  cudaMemcpy(dGr,Graw.data(),(size_t)T*NQHD*4,cudaMemcpyHostToDevice);
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  sdpa_gqa_k<<<T*NQ,256,0>>>(dQ,dK,dV,dA,T);
  int bid=(int)((T*NQHD+255)/256);
  gate_k<<<bid,256>>>(dA,dGr,dG,T*NQHD);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms=0; cudaEventElapsedTime(&ms,e0,e1);
  printf("sdpa+gate ms=%.4f (sdpa grid=%d blk=256)\n",ms,T*NQ);
  cudaEventRecord(e0); const int REPS=20;
  for(int r=0;r<REPS;++r) sdpa_gqa_k<<<T*NQ,256,0>>>(dQ,dK,dV,dA,T);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float mss=0; cudaEventElapsedTime(&mss,e0,e1); mss/=REPS;
  printf("sdpa only avg ms=%.4f over %d\n",mss,REPS);
  std::vector<float> att((size_t)T*NQHD), gated((size_t)T*NQHD);
  cudaMemcpy(att.data(),dA,(size_t)T*NQHD*4,cudaMemcpyDeviceToHost);
  // need gated after gate kernel (re-run both to have dG fresh)
  sdpa_gqa_k<<<T*NQ,256,0>>>(dQ,dK,dV,dA,T);
  gate_k<<<bid,256>>>(dA,dGr,dG,T*NQHD);
  cudaDeviceSynchronize();
  cudaMemcpy(att.data(),dA,(size_t)T*NQHD*4,cudaMemcpyDeviceToHost);
  cudaMemcpy(gated.data(),dG,(size_t)T*NQHD*4,cudaMemcpyDeviceToHost);
  stats("GPU att PRE-gate vs golden",att.data(),att_ref.data(),att.size(),1e-4f);
  stats("GPU gated POST-gate vs golden G",gated.data(),Ggated_ref.data(),gated.size(),1e-4f);
  // out_proj GEMV: gated BF16 -> core
  std::vector<uint16_t> gated_bf16(gated.size());
  for(size_t i=0;i<gated.size();++i) gated_bf16[i]=hermes::f32_to_bf16(gated[i]);
  uint16_t* d_g=nullptr; uint8_t* d_Wo=nullptr; float* d_core=nullptr;
  size_t wo_bytes=(size_t)5120*(6144/64*36);
  cudaMalloc(&d_g,(size_t)T*NQHD*2); cudaMalloc(&d_Wo,wo_bytes); cudaMalloc(&d_core,(size_t)T*HID*4);
  cudaMemcpy(d_g,gated_bf16.data(),(size_t)T*NQHD*2,cudaMemcpyHostToDevice);
  cudaMemcpy(d_Wo,(const uint8_t*)vWo.data,wo_bytes,cudaMemcpyHostToDevice);
  cudaEventRecord(e0);
  for(int t=0;t<T;++t) hermes::nvfp4_gemv_cuda(d_g+t*NQHD,d_Wo,d_core+t*HID,NQHD,HID);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms_o=0; cudaEventElapsedTime(&ms_o,e0,e1);
  std::vector<float> core((size_t)T*HID);
  cudaMemcpy(core.data(),d_core,(size_t)T*HID*4,cudaMemcpyDeviceToHost);
  printf("out_proj GEMV ms=%.4f total N=%d K=%d\n",ms_o,HID,NQHD);
  stats("GPU core vs golden",core.data(),core_ref.data(),core.size(),0.15f);
  printf("core_gpu[0..3]=%.6f %.6f %.6f %.6f core_ref=%.6f %.6f %.6f %.6f\n",
    core[0],core[1],core[2],core[3],core_ref[0],core_ref[1],core_ref[2],core_ref[3]);
  size_t f1=0,t1=0; cudaMemGetInfo(&f1,&t1);
  printf("vram_after free=%zu MiB delta=%zd MiB ms sdpa+gate=%.3f out=%.3f\n",
    f1/(1024*1024),(long)((long)f0-(long)f1)/(1024*1024),ms,ms_o);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dA);cudaFree(dGr);cudaFree(dG);
  cudaFree(d_g);cudaFree(d_Wo);cudaFree(d_core);
  return 0;
}

int main(int argc,char** argv){
  std::string only=(argc>1)?argv[1]:"all";
  hermes::GgufWeightBackend be;
  try{ be.open(CKPT); }catch(std::exception& e){ printf("GGUF OPEN FAIL: %s\n",e.what()); return 1; }
  int rc=0;
  if(only=="all"||only=="s1"){ int r=run_s1(be); if(r!=0) rc=r; if(only=="s1") return rc; }
  if(only=="all"||only=="s2"){ int r=run_s2(be); if(r!=0) rc=r; if(only=="s2") return rc; }
  if(only=="all"||only=="s3"){ int r=run_s3(be); if(r!=0) rc=r; }
  return rc;
}
