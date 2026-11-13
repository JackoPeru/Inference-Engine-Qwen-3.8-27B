// P1 blk.0 GPU1: stage1 conv+silu (PASS) + stage2 qkv GEMM NVFP4.
// Stage1: qkv golden H2D + Wc reale -> conv kernel -> vs /tmp/e1_cv.bin, thr 1e-4.
// Stage2: h=RMS(x,attn_norm) diretto da embed reali -> qkv via NVFP4 W4A4 GPU
//         vs /tmp/e1_cv_qkv.bin. Riporta numeri reali (no soglie inventate).
// Leggero: <150MB VRAM.
#include "hermes/weights.hpp"
#include "hermes/gguf.hpp"
#include "hermes/rmsnorm.hpp"
#include "hermes/nvfp4_gemm.hpp"
#include "hermes/nvfp4_gemv.hpp"
#include "hermes/gdn_step.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <cuda_runtime.h>

static const char* CKPT = "/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf";
static const char* F_QKV = "/tmp/e1_cv_qkv.bin";
static const char* F_CV  = "/tmp/e1_cv.bin";
static const char* F_BETA= "/tmp/e1_beta.bin";
static const char* F_GL  = "/tmp/e1_gl.bin";
static const char* F_O   = "/tmp/e1_o.bin";
static const char* F_CORE= "/tmp/e1_core.bin";
static const char* F_Y   = "/tmp/e1_y.bin";
static const int T = 11;
static const int C = 10240;
static const int HID = 5120;
static const int N_QKV = 10240;
static const int KD = 2048, VDD = 6144, HV = 48, KHD = 128, VHD = 128;

__global__ void conv_silu_k(const float* qkv, const float* Wc, float* cv, int TT, int CC) {
  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = TT * CC;
  if (idx >= N) return;
  int t = idx / CC;
  int c = idx % CC;
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int tt = t - (3 - j);
    float xv = (tt < 0) ? 0.f : qkv[tt * CC + c];
    acc += xv * Wc[c * 4 + j];
  }
  float s = 1.f / (1.f + expf(-acc));
  cv[idx] = acc * s;
}

static bool read_bin(const char* path, std::vector<float>& out, size_t expect_n) {
  FILE* f = fopen(path, "rb");
  if (!f) { printf("OPEN FAIL %s\n", path); return false; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if ((size_t)sz != expect_n * 4) { printf("SIZE MISMATCH %s: got %ld expect %zu\n", path, sz, expect_n * 4); fclose(f); return false; }
  out.resize(expect_n);
  size_t r = fread(out.data(), 4, expect_n, f);
  fclose(f);
  if (r != expect_n) { printf("READ FAIL %s\n", path); return false; }
  return true;
}

static int run_stage1(hermes::GgufWeightBackend& be, float* ms_out, double* mx_out) {
  size_t free0 = 0, tot0 = 0;
  cudaMemGetInfo(&free0, &tot0);
  printf("\n=== STAGE1 conv+silu T=%d C=%d ===\n", T, C);
  printf("vram_before free=%zu MiB\n", free0 / (1024*1024));
  std::vector<float> qkv, cv_ref;
  if (!read_bin(F_QKV, qkv, (size_t)T * C)) return 1;
  if (!read_bin(F_CV, cv_ref, (size_t)T * C)) return 1;
  printf("golden qkv[0..4]: %.6f %.6f %.6f %.6f %.6f\n", qkv[0], qkv[1], qkv[2], qkv[3], qkv[4]);
  auto wv = be.view("blk.0.ssm_conv1d.weight");
  printf("Wc %s ne=[%llu,%llu] bytes=%llu\n", wv.info->name.c_str(),
    (unsigned long long)wv.info->ne[0], (unsigned long long)wv.info->ne[1],
    (unsigned long long)wv.size);
  if (wv.info->ne[0] != 4 || wv.info->ne[1] != 10240) { printf("WC SHAPE MISMATCH\n"); return 1; }
  const float* Wc_host = (const float*)wv.data;
  float *d_qkv=nullptr,*d_wc=nullptr,*d_cv=nullptr;
  size_t n=(size_t)T*C;
  cudaMalloc(&d_qkv,n*4); cudaMalloc(&d_wc,(size_t)C*4*4); cudaMalloc(&d_cv,n*4);
  cudaMemcpy(d_qkv,qkv.data(),n*4,cudaMemcpyHostToDevice);
  cudaMemcpy(d_wc,Wc_host,(size_t)C*4*4,cudaMemcpyHostToDevice);
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  int threads=256, blocks=(int)((n+threads-1)/threads);
  conv_silu_k<<<blocks,threads>>>(d_qkv,d_wc,d_cv,T,C);
  cudaDeviceSynchronize();
  cudaEventRecord(e0);
  const int REPS=100;
  for(int r=0;r<REPS;++r) conv_silu_k<<<blocks,threads>>>(d_qkv,d_wc,d_cv,T,C);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms=0; cudaEventElapsedTime(&ms,e0,e1); ms/=REPS;
  std::vector<float> cv_gpu(n);
  cudaMemcpy(cv_gpu.data(),d_cv,n*4,cudaMemcpyDeviceToHost);
  size_t free1=0,tot1=0; cudaMemGetInfo(&free1,&tot1);
  printf("kernel_ms=%.4f (avg %d) blocks=%d\n",ms,REPS,blocks);
  printf("vram_after free=%zu MiB delta=%zd KiB\n",free1/(1024*1024),(long)((long)free0-(long)free1)/1024);
  double mx=0; size_t imx=0; double dot=0,na=0,nb=0; double suma=0;
  for(size_t i=0;i<n;++i){ float e=fabsf(cv_gpu[i]-cv_ref[i]); if(e>mx){mx=e;imx=i;}
    suma+=fabsf(cv_ref[i]); dot+=(double)cv_gpu[i]*(double)cv_ref[i];
    na+=(double)cv_gpu[i]*(double)cv_gpu[i]; nb+=(double)cv_ref[i]*(double)cv_ref[i]; }
  double cosv=dot/(sqrt(na*nb)+1e-12);
  printf("cv_gpu[0..4]: %.6f %.6f %.6f %.6f %.6f\n",cv_gpu[0],cv_gpu[1],cv_gpu[2],cv_gpu[3],cv_gpu[4]);
  printf("maxerr=%.3e at idx=%zu (t=%zu c=%zu) gpu=%.6f ref=%.6f cos=%.8f meanabs=%.5f\n",
    mx,imx,imx/C,imx%C,cv_gpu[imx],cv_ref[imx],cosv,suma/n);
  bool pass=(mx<1e-4);
  printf("STAGE1 %s (thr 1e-4)\n",pass?"PASS":"FAIL");
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  cudaFree(d_qkv); cudaFree(d_wc); cudaFree(d_cv);
  if(ms_out) *ms_out=ms;
  if(mx_out) *mx_out=mx;
  return pass?0:1;
}

// ---- stage2 helpers ----
static inline float bf2f_u16(uint16_t b){ return hermes::bf16_to_f32(b); }

static int run_stage2(hermes::GgufWeightBackend& be) {
  printf("\n=== STAGE2 qkv GEMM NVFP4 T=%d HID=%d N=%d ===\n",T,HID,N_QKV);
  size_t free0=0,tot0=0; cudaMemGetInfo(&free0,&tot0);
  printf("vram_before free=%zu MiB\n",free0/(1024*1024));
  // ids goldens (E1): base_ids
  const int ids[T]={6205,161421,303,220,17,1374,9911,1147,27053,6059,13};
  // views
  auto vEmb=be.view("token_embd.weight");
  auto vNorm=be.view("blk.0.attn_norm.weight");
  auto vQkv=be.view("blk.0.attn_qkv.weight");
  printf("emb %s ne=[%llu,%llu] type=%d\n",vEmb.info->name.c_str(),
    (unsigned long long)vEmb.info->ne[0],(unsigned long long)vEmb.info->ne[1],(int)vEmb.info->type);
  printf("norm %s ne=[%llu,%llu] type=%d\n",vNorm.info->name.c_str(),
    (unsigned long long)vNorm.info->ne[0],(unsigned long long)vNorm.info->ne[1],(int)vNorm.info->type);
  printf("qkv %s ne=[%llu,%llu] type=%d bytes=%llu\n",vQkv.info->name.c_str(),
    (unsigned long long)vQkv.info->ne[0],(unsigned long long)vQkv.info->ne[1],
    (int)vQkv.info->type,(unsigned long long)vQkv.size);
  // x = emb[ids] float
  const uint16_t* embBase=(const uint16_t*)vEmb.data;
  const float* wNorm=(const float*)vNorm.data;
  std::vector<float> x((size_t)T*HID), h((size_t)T*HID);
  for(int t=0;t<T;++t){
    const uint16_t* row=embBase+(size_t)ids[t]*HID;
    for(int i=0;i<HID;++i) x[(size_t)t*HID+i]=bf2f_u16(row[i]);
  }
  printf("x[0,0..4]: %.6f %.6f %.6f %.6f %.6f\n",x[0],x[1],x[2],x[3],x[4]);
  printf("attn_norm[0..4]: %.6f %.6f %.6f %.6f %.6f\n",wNorm[0],wNorm[1],wNorm[2],wNorm[3],wNorm[4]);
  // h = RMS direct
  for(int t=0;t<T;++t){
    double ss=0; float* xr=&x[(size_t)t*HID];
    for(int i=0;i<HID;++i) ss+=(double)xr[i]*(double)xr[i];
    float inv=1.f/sqrtf((float)(ss/HID)+1e-6f);
    float* hr=&h[(size_t)t*HID];
    for(int i=0;i<HID;++i) hr[i]=xr[i]*inv*wNorm[i];
  }
  printf("h[0,0..4]: %.6f %.6f %.6f %.6f %.6f\n",h[0],h[1],h[2],h[3],h[4]);
  // golden qkv
  std::vector<float> qkv_ref;
  if(!read_bin(F_QKV,qkv_ref,(size_t)T*N_QKV)) return 1;
  // CPU dequant reference: dequant Wqkv -> float GEMM
  const uint8_t* qkvRows=vQkv.data;
  int nb=HID/64; // 80
  std::vector<float> Wdeq((size_t)N_QKV*HID);
  std::vector<float> tmp64(64);
  for(int n=0;n<N_QKV;++n){
    const uint8_t* row=qkvRows+(size_t)n*nb*36;
    float* out=&Wdeq[(size_t)n*HID];
    for(int b=0;b<nb;++b){
      hermes::nvfp4_block_dequant(row+b*36,tmp64.data());
      for(int j=0;j<64;++j) out[b*64+j]=tmp64[j];
    }
  }
  printf("Wqkv deq row0[0..4]: %.6f %.6f %.6f %.6f %.6f\n",
    Wdeq[0],Wdeq[1],Wdeq[2],Wdeq[3],Wdeq[4]);
  std::vector<float> qkv_cpu((size_t)T*N_QKV,0.f);
  for(int t=0;t<T;++t) for(int n=0;n<N_QKV;++n){
    double a=0; float* hr=&h[(size_t)t*HID]; float* wr=&Wdeq[(size_t)n*HID];
    for(int k=0;k<HID;++k) a+=(double)hr[k]*(double)wr[k];
    qkv_cpu[(size_t)t*N_QKV+n]=(float)a;
  }
  double mxc=0; size_t imxc=0; double dotc=0,nac=0,nbc=0;
  for(size_t i=0;i<qkv_cpu.size();++i){ float e=fabsf(qkv_cpu[i]-qkv_ref[i]);
    if(e>mxc){mxc=e;imxc=i;} dotc+=(double)qkv_cpu[i]*(double)qkv_ref[i];
    nac+=(double)qkv_cpu[i]*(double)qkv_cpu[i]; nbc+=(double)qkv_ref[i]*(double)qkv_ref[i]; }
  double cosc=dotc/(sqrt(nac*nbc)+1e-12);
  printf("CPU-dequant vs golden: maxerr=%.3e at %zu (t=%zu n=%zu) cpu=%.6f ref=%.6f cos=%.8f\n",
    mxc,imxc,imxc/N_QKV,imxc%N_QKV,qkv_cpu[imxc],qkv_ref[imxc],cosc);
  printf("qkv_cpu[0..4]: %.6f %.6f %.6f %.6f %.6f\n",qkv_cpu[0],qkv_cpu[1],qkv_cpu[2],qkv_cpu[3],qkv_cpu[4]);
  printf("qkv_ref[0..4]: %.6f %.6f %.6f %.6f %.6f\n",qkv_ref[0],qkv_ref[1],qkv_ref[2],qkv_ref[3],qkv_ref[4]);

  // GPU W4A4 path: h BF16 -> quant -> GEMM
  std::vector<uint16_t> h_bf16((size_t)T*HID);
  for(size_t i=0;i<h.size();++i) h_bf16[i]=hermes::f32_to_bf16(h[i]);
  // try M=T direct, else padded
  for(int attempt=0;attempt<2;++attempt){
    int Mp = (attempt==0)? T : 128;
    printf("--- GPU W4A4 attempt Mp=%d ---\n",Mp);
    size_t sfaB=hermes::Nvfp4GemmW4A4::sfa_bytes(Mp,HID);
    size_t sfbB=hermes::Nvfp4GemmW4A4::sfb_bytes(Mp,N_QKV,HID);
    size_t wsB=hermes::Nvfp4GemmW4A4::workspace_bytes(Mp,N_QKV,HID);
    printf("sfa=%zu sfb=%zu ws=%zu backend=%s\n",sfaB,sfbB,wsB,
      hermes::Nvfp4GemmW4A4::backend_name());
    if(sfaB==0||sfbB==0){ printf("unsupported bytes (0), skip\n"); continue; }
    // host buffers padded
    std::vector<uint16_t> Xp((size_t)Mp*HID,0);
    // copy h_bf16 into first T rows (BF16 bits)
    for(int t=0;t<T;++t) memcpy(&Xp[(size_t)t*HID],&h_bf16[(size_t)t*HID],HID*2);
    std::vector<uint8_t> Aq((size_t)Mp*HID/2,0), SFA(sfaB,0);
    hermes::Nvfp4GemmW4A4::quant_activation(Xp.data(),Aq.data(),SFA.data(),Mp,HID);
    std::vector<uint8_t> Bq((size_t)N_QKV*HID/2,0), SFB(sfbB,0);
    hermes::Nvfp4GemmW4A4::build_B_from_gguf(qkvRows,Bq.data(),SFB.data(),N_QKV,HID);
    // device
    uint8_t *dA=nullptr,*dSFA=nullptr,*dB=nullptr,*dSFB=nullptr;
    uint16_t *dD=nullptr;
    cudaMalloc(&dA,Aq.size()); cudaMalloc(&dSFA,SFA.size());
    cudaMalloc(&dB,Bq.size()); cudaMalloc(&dSFB,SFB.size());
    cudaMalloc(&dD,(size_t)Mp*N_QKV*2);
    cudaMemcpy(dA,Aq.data(),Aq.size(),cudaMemcpyHostToDevice);
    cudaMemcpy(dSFA,SFA.data(),SFA.size(),cudaMemcpyHostToDevice);
    cudaMemcpy(dB,Bq.data(),Bq.size(),cudaMemcpyHostToDevice);
    cudaMemcpy(dSFB,SFB.data(),SFB.size(),cudaMemcpyHostToDevice);
    cudaMemset(dD,0,(size_t)Mp*N_QKV*2);
    hermes::Nvfp4GemmW4A4::Workspace ws{nullptr,0};
    int wrc=hermes::Nvfp4GemmW4A4::workspace_ensure(ws,Mp,N_QKV,HID);
    printf("workspace_ensure rc=%d bytes=%zu\n",wrc,ws.bytes);
    // warmup + timed
    int rc=hermes::Nvfp4GemmW4A4::run_ws(dA,dSFA,dB,dSFB,nullptr,dD,Mp,N_QKV,HID,
      ws.ptr,ws.bytes,nullptr);
    printf("run_ws rc=%d\n",rc);
    if(rc!=0){
      printf("run_ws unsupported/fail, free and try next\n");
      cudaFree(dA);cudaFree(dSFA);cudaFree(dB);cudaFree(dSFB);cudaFree(dD);
      hermes::Nvfp4GemmW4A4::workspace_release(ws);
      continue;
    }
    cudaDeviceSynchronize();
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    const int REPS=20;
    for(int r=0;r<REPS;++r){
      int rr=hermes::Nvfp4GemmW4A4::run_ws(dA,dSFA,dB,dSFB,nullptr,dD,Mp,N_QKV,HID,
        ws.ptr,ws.bytes,nullptr);
      if(rr!=0){ printf("run_ws rep fail %d\n",rr); break; }
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms=0; cudaEventElapsedTime(&ms,e0,e1); ms/=REPS;
    printf("w4a4 kernel_ms=%.4f avg %d reps Mp=%d N=%d K=%d\n",ms,REPS,Mp,N_QKV,HID);
    std::vector<uint16_t> D((size_t)Mp*N_QKV);
    cudaMemcpy(D.data(),dD,(size_t)Mp*N_QKV*2,cudaMemcpyDeviceToHost);
    size_t free1=0,tot1=0; cudaMemGetInfo(&free1,&tot1);
    printf("vram_after free=%zu MiB delta_vs_start=%zd MiB\n",free1/(1024*1024),
      (long)((long)free0-(long)free1)/(1024*1024));
    // compare first T rows
    double mx=0; size_t imx=0; double dot=0,na=0,nb=0; double sumref=0;
    for(int t=0;t<T;++t) for(int n=0;n<N_QKV;++n){
      float g=hermes::bf16_to_f32(D[(size_t)t*N_QKV+n]);
      // NOTE: when Mp=128 padded, D layout is [Mp,N] row-major? Engine D is [M,N] row-major.
      // For padded case with Mp=128, row t is still t*N_QKV (since M rows). OK.
      // But SFB/SFA for padded may shift? We compare directly.
      float r=qkv_ref[(size_t)t*N_QKV+n];
      float e=fabsf(g-r);
      if(e>mx){mx=e;imx=(size_t)t*N_QKV+n;}
      dot+=(double)g*(double)r; na+=(double)g*(double)g; nb+=(double)r*(double)r;
      sumref+=fabsf(r);
    }
    double cosv=dot/(sqrt(na*nb)+1e-12);
    // also relative: maxerr / rms?
    double rms=sqrt(nb/(T*N_QKV));
    double rel=mx/(rms+1e-12);
    printf("GPU-W4A4 vs golden: maxerr=%.5f at %zu (t=%zu n=%zu) gpu=%.5f ref=%.5f cos=%.8f rms=%.5f rel=%.5f\n",
      mx,imx,imx/N_QKV,imx%N_QKV,
      hermes::bf16_to_f32(D[imx]),qkv_ref[imx],cosv,rms,rel);
    printf("D[0..4]: %.5f %.5f %.5f %.5f %.5f (bf16->f32)\n",
      hermes::bf16_to_f32(D[0]),hermes::bf16_to_f32(D[1]),hermes::bf16_to_f32(D[2]),
      hermes::bf16_to_f32(D[3]),hermes::bf16_to_f32(D[4]));
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(dA);cudaFree(dSFA);cudaFree(dB);cudaFree(dSFB);cudaFree(dD);
    hermes::Nvfp4GemmW4A4::workspace_release(ws);
    // report, no threshold inventata: mostriamo numero reale
    printf("STAGE2 GPU-W4A4 Mp=%d maxerr=%.5f rel=%.5f cos=%.8f ms=%.4f\n",Mp,mx,rel,cosv,ms);
    return 0; // un tentativo riuscito basta per report
  }
  printf("STAGE2: nessun run_ws riuscito (Mp=11/128 unsupported?)\n");
  return 2;
}

__global__ void wa_gemm_k(const float* h, const float* Wa, float* ar, int TT, int N, int K) {
  int idx=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(idx>=TT*N) return;
  int t=idx/N, n=idx%N;
  double a=0; for(int k=0;k<K;++k) a+=(double)h[(size_t)t*K+k]*(double)Wa[(size_t)n*K+k];
  ar[idx]=(float)a;
}
static inline float sigmoid_f(float x){ return 1.f/(1.f+expf(-x)); }
static inline float softplus_f(float x){ return x>20.f?x:logf(1.f+expf(x)); }

static int run_stage3(hermes::GgufWeightBackend& be){
  printf("\n=== STAGE3 linears+gdn_step+gated+out_proj T=%d ===\n",T);
  size_t free0=0,tot0=0; cudaMemGetInfo(&free0,&tot0);
  printf("vram_before free=%zu MiB\n",free0/(1024*1024));
  const int ids[T]={6205,161421,303,220,17,1374,9911,1147,27053,6059,13};
  auto vEmb=be.view("token_embd.weight");
  auto vNorm=be.view("blk.0.attn_norm.weight");
  auto vWz=be.view("blk.0.attn_gate.weight");
  auto vWb=be.view("blk.0.ssm_beta.weight");
  auto vWa=be.view("blk.0.ssm_alpha.weight");
  auto vA=be.view("blk.0.ssm_a");
  auto vDT=be.view("blk.0.ssm_dt.bias");
  auto vWn=be.view("blk.0.ssm_norm.weight");
  auto vWout=be.view("blk.0.ssm_out.weight");
  const uint16_t* embBase=(const uint16_t*)vEmb.data;
  const float* wNorm=(const float*)vNorm.data;
  std::vector<float> x((size_t)T*HID), h((size_t)T*HID);
  for(int t=0;t<T;++t){ const uint16_t* row=embBase+(size_t)ids[t]*HID;
    for(int i=0;i<HID;++i) x[(size_t)t*HID+i]=hermes::bf16_to_f32(row[i]); }
  for(int t=0;t<T;++t){ double ss=0; float* xr=&x[(size_t)t*HID];
    for(int i=0;i<HID;++i) ss+=(double)xr[i]*xr[i];
    float inv=1.f/sqrtf((float)(ss/HID)+1e-6f);
    for(int i=0;i<HID;++i) h[(size_t)t*HID+i]=xr[i]*inv*wNorm[i]; }
  std::vector<float> cv_ref, beta_ref, gl_ref, o_ref, core_ref;
  if(!read_bin(F_CV,cv_ref,(size_t)T*C)) return 1;
  if(!read_bin(F_BETA,beta_ref,(size_t)T*HV)) return 1;
  if(!read_bin(F_GL,gl_ref,(size_t)T*HV)) return 1;
  if(!read_bin(F_O,o_ref,(size_t)T*HV*VHD)) return 1;
  if(!read_bin(F_CORE,core_ref,(size_t)T*HID)) return 1;
  // h_bf16 for GEMV
  std::vector<uint16_t> h_bf16((size_t)T*HID);
  for(size_t i=0;i<h.size();++i) h_bf16[i]=hermes::f32_to_bf16(h[i]);
  uint16_t* d_h=nullptr; cudaMalloc(&d_h,(size_t)T*HID*2);
  cudaMemcpy(d_h,h_bf16.data(),(size_t)T*HID*2,cudaMemcpyHostToDevice);
  // --- Wz GEMV ---
  int Nw=6144, Kw=HID;
  size_t wz_bytes=(size_t)Nw*(Kw/64*36);
  uint8_t* d_Wz=nullptr; float* d_zl=nullptr;
  cudaMalloc(&d_Wz,wz_bytes); cudaMalloc(&d_zl,(size_t)T*Nw*4);
  cudaMemcpy(d_Wz,(const uint8_t*)vWz.data,wz_bytes,cudaMemcpyHostToDevice);
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for(int t=0;t<T;++t) hermes::nvfp4_gemv_cuda(d_h+t*HID, d_Wz, d_zl+t*Nw, Kw, Nw);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms_wz=0; cudaEventElapsedTime(&ms_wz,e0,e1);
  std::vector<float> zl((size_t)T*Nw);
  cudaMemcpy(zl.data(),d_zl,(size_t)T*Nw*4,cudaMemcpyDeviceToHost);
  printf("Wz GEMV ms=%.4f total %d steps N=%d K=%d\n",ms_wz,T,Nw,Kw);
  // --- Wb GEMV ---
  int Nb=48;
  size_t wb_bytes=(size_t)Nb*(HID/64*36);
  uint8_t* d_Wb=nullptr; float* d_b=nullptr;
  cudaMalloc(&d_Wb,wb_bytes); cudaMalloc(&d_b,(size_t)T*Nb*4);
  cudaMemcpy(d_Wb,(const uint8_t*)vWb.data,wb_bytes,cudaMemcpyHostToDevice);
  cudaEventRecord(e0);
  for(int t=0;t<T;++t) hermes::nvfp4_gemv_cuda(d_h+t*HID, d_Wb, d_b+t*Nb, HID, Nb);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms_wb=0; cudaEventElapsedTime(&ms_wb,e0,e1);
  std::vector<float> b((size_t)T*Nb);
  cudaMemcpy(b.data(),d_b,(size_t)T*Nb*4,cudaMemcpyDeviceToHost);
  printf("Wb GEMV ms=%.4f N=%d\n",ms_wb,Nb);
  // --- Wa float GEMM on GPU ---
  float* d_hf=nullptr; float* d_Wa=nullptr; float* d_ar=nullptr;
  cudaMalloc(&d_hf,(size_t)T*HID*4); cudaMalloc(&d_Wa,(size_t)Nb*HID*4);
  cudaMalloc(&d_ar,(size_t)T*Nb*4);
  cudaMemcpy(d_hf,h.data(),(size_t)T*HID*4,cudaMemcpyHostToDevice);
  cudaMemcpy(d_Wa,(const float*)vWa.data,(size_t)Nb*HID*4,cudaMemcpyHostToDevice);
  int blks=(T*Nb+255)/256;
  cudaEventRecord(e0);
  wa_gemm_k<<<blks,256>>>(d_hf,d_Wa,d_ar,T,Nb,HID);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms_wa=0; cudaEventElapsedTime(&ms_wa,e0,e1);
  std::vector<float> ar((size_t)T*Nb);
  cudaMemcpy(ar.data(),d_ar,(size_t)T*Nb*4,cudaMemcpyDeviceToHost);
  printf("Wa GEMM ms=%.4f N=%d K=%d\n",ms_wa,Nb,HID);
  cudaFree(d_hf); cudaFree(d_Wa); cudaFree(d_ar);
  // --- beta / gl CPU elementwise ---
  const float* A=(const float*)vA.data; const float* DT=(const float*)vDT.data;
  std::vector<float> beta((size_t)T*HV), gl((size_t)T*HV);
  for(int t=0;t<T;++t) for(int hh=0;hh<HV;++hh){
    beta[(size_t)t*HV+hh]=sigmoid_f(b[(size_t)t*HV+hh]);
    gl[(size_t)t*HV+hh]=-expf(A[hh])*softplus_f(ar[(size_t)t*HV+hh]+DT[hh]);
  }
  double mxb=0,mbg=0; double dotb=0,nab=0,nbb=0,dotg=0,nag=0,nbg=0; size_t ib=0,ig=0;
  for(size_t i=0;i<beta.size();++i){ float e=fabsf(beta[i]-beta_ref[i]); if(e>mxb){mxb=e;ib=i;}
    dotb+=(double)beta[i]*(double)beta_ref[i]; nab+=(double)beta[i]*beta[i]; nbb+=(double)beta_ref[i]*beta_ref[i]; }
  for(size_t i=0;i<gl.size();++i){ float e=fabsf(gl[i]-gl_ref[i]); if(e>mbg){mbg=e;ig=i;}
    dotg+=(double)gl[i]*(double)gl_ref[i]; nag+=(double)gl[i]*gl[i]; nbg+=(double)gl_ref[i]*gl_ref[i]; }
  printf("beta vs golden maxerr=%.3e at %zu gpu=%.6f ref=%.6f cos=%.8f\n",mxb,ib,beta[ib],beta_ref[ib],dotb/(sqrt(nab*nbb)+1e-12));
  printf("gl   vs golden maxerr=%.3e at %zu gpu=%.6f ref=%.6f cos=%.8f\n",mbg,ig,gl[ig],gl_ref[ig],dotg/(sqrt(nag*nbg)+1e-12));
  // --- q/k/v from golden cv, l2norm CPU ---
  std::vector<float> q((size_t)T*HV*KHD), k((size_t)T*HV*KHD), v((size_t)T*HV*VHD);
  for(int t=0;t<T;++t){ float* c0=&cv_ref[(size_t)t*C];
    for(int hh=0;hh<HV;++hh){ int g=hh/3;
      for(int d=0;d<KHD;++d){ q[((size_t)t*HV+hh)*KHD+d]=c0[g*KHD+d]; k[((size_t)t*HV+hh)*KHD+d]=c0[KD+g*KHD+d]; }
      for(int d=0;d<VHD;++d) v[((size_t)t*HV+hh)*VHD+d]=c0[2*KD+hh*VHD+d]; } }
  std::vector<float> qn(q.size()), kn(k.size());
  for(int t=0;t<T;++t) for(int hh=0;hh<HV;++hh){
    double nq=0,nk=0;
    for(int d=0;d<KHD;++d){ nq+=(double)q[((size_t)t*HV+hh)*KHD+d]*(double)q[((size_t)t*HV+hh)*KHD+d];
      nk+=(double)k[((size_t)t*HV+hh)*KHD+d]*(double)k[((size_t)t*HV+hh)*KHD+d]; }
    float iq=1.f/sqrtf((float)nq+1e-6f), ik=1.f/sqrtf((float)nk+1e-6f);
    float sc=1.f/sqrtf(128.f);
    for(int d=0;d<KHD;++d){ qn[((size_t)t*HV+hh)*KHD+d]=(float)((double)q[((size_t)t*HV+hh)*KHD+d]*iq*sc);
      kn[((size_t)t*HV+hh)*KHD+d]=(float)((double)k[((size_t)t*HV+hh)*KHD+d]*ik); }
  }
  // --- gdn_step on GPU sequential ---
  float *d_qn=nullptr,*d_kn=nullptr,*d_v=nullptr,*d_be=nullptr,*d_gl=nullptr,*d_S=nullptr,*d_o=nullptr;
  size_t qnB=(size_t)T*HV*KHD*4, vB=(size_t)T*HV*VHD*4, bgB=(size_t)T*HV*4;
  cudaMalloc(&d_qn,qnB); cudaMalloc(&d_kn,qnB); cudaMalloc(&d_v,vB);
  cudaMalloc(&d_be,bgB); cudaMalloc(&d_gl,bgB);
  cudaMalloc(&d_S,(size_t)HV*KHD*VHD*4); cudaMalloc(&d_o,vB);
  cudaMemcpy(d_qn,qn.data(),qnB,cudaMemcpyHostToDevice);
  cudaMemcpy(d_kn,kn.data(),qnB,cudaMemcpyHostToDevice);
  cudaMemcpy(d_v,v.data(),vB,cudaMemcpyHostToDevice);
  cudaMemcpy(d_be,beta.data(),bgB,cudaMemcpyHostToDevice);
  cudaMemcpy(d_gl,gl.data(),bgB,cudaMemcpyHostToDevice);
  cudaMemset(d_S,0,(size_t)HV*KHD*VHD*4);
  cudaEventRecord(e0);
  for(int t=0;t<T;++t) hermes::gdn_step_cuda(d_qn+t*HV*KHD,d_kn+t*HV*KHD,d_v+t*HV*VHD,
    d_be+t*HV,d_gl+t*HV,d_S,d_o+t*HV*VHD,HV,KHD,VHD);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms_gdn=0; cudaEventElapsedTime(&ms_gdn,e0,e1);
  std::vector<float> o_gpu((size_t)T*HV*VHD);
  cudaMemcpy(o_gpu.data(),d_o,vB,cudaMemcpyDeviceToHost);
  printf("gdn_step GPU total ms=%.4f for T=%d (H=%d K=%d V=%d)\n",ms_gdn,T,HV,KHD,VHD);
  double mxo=0; size_t io=0; double doto=0,nao=0,nbo=0;
  for(size_t i=0;i<o_gpu.size();++i){ float e=fabsf(o_gpu[i]-o_ref[i]); if(e>mxo){mxo=e;io=i;}
    doto+=(double)o_gpu[i]*(double)o_ref[i]; nao+=(double)o_gpu[i]*o_gpu[i]; nbo+=(double)o_ref[i]*o_ref[i]; }
  printf("o vs golden maxerr=%.3e at %zu (t=%zu h=%zu d=%zu) gpu=%.6f ref=%.6f cos=%.8f\n",
    mxo,io,io/(HV*VHD),(io/(VHD))%HV,io%VHD,o_gpu[io],o_ref[io],doto/(sqrt(nao*nbo)+1e-12));
  // --- gated CPU then out_proj GEMV GPU ---
  const float* Wn=(const float*)vWn.data;
  std::vector<float> gated((size_t)T*VDD);
  for(int t=0;t<T;++t) for(int hh=0;hh<HV;++hh){
    float* oo=&o_gpu[((size_t)t*HV+hh)*VHD]; float* zz=&zl[((size_t)t*HV+hh)*VHD];
    double va=0; for(int d=0;d<VHD;++d) va+=(double)oo[d]*oo[d];
    float inv=1.f/sqrtf((float)(va/VHD)+1e-6f);
    for(int d=0;d<VHD;++d){ float s=zz[d]/(1.f+expf(-zz[d])); gated[((size_t)t*HV+hh)*VHD+d]=oo[d]*inv*Wn[d]*s; }
  }
  // out_proj GEMV: H=6144 F=5120
  std::vector<uint16_t> gated_bf16(gated.size());
  for(size_t i=0;i<gated.size();++i) gated_bf16[i]=hermes::f32_to_bf16(gated[i]);
  uint16_t* d_g=nullptr; uint8_t* d_Wo=nullptr; float* d_core=nullptr;
  size_t wo_bytes=(size_t)5120*(6144/64*36);
  cudaMalloc(&d_g,(size_t)T*VDD*2); cudaMalloc(&d_Wo,wo_bytes); cudaMalloc(&d_core,(size_t)T*HID*4);
  cudaMemcpy(d_g,gated_bf16.data(),(size_t)T*VDD*2,cudaMemcpyHostToDevice);
  cudaMemcpy(d_Wo,(const uint8_t*)vWout.data,wo_bytes,cudaMemcpyHostToDevice);
  cudaEventRecord(e0);
  for(int t=0;t<T;++t) hermes::nvfp4_gemv_cuda(d_g+t*VDD, d_Wo, d_core+t*HID, VDD, HID);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms_out=0; cudaEventElapsedTime(&ms_out,e0,e1);
  std::vector<float> core_gpu((size_t)T*HID);
  cudaMemcpy(core_gpu.data(),d_core,(size_t)T*HID*4,cudaMemcpyDeviceToHost);
  printf("out_proj GEMV ms=%.4f total N=%d K=%d\n",ms_out,HID,VDD);
  double mxc=0; size_t ic=0; double dotc=0,nac=0,nbc=0;
  for(size_t i=0;i<core_gpu.size();++i){ float e=fabsf(core_gpu[i]-core_ref[i]); if(e>mxc){mxc=e;ic=i;}
    dotc+=(double)core_gpu[i]*(double)core_ref[i]; nac+=(double)core_gpu[i]*core_gpu[i]; nbc+=(double)core_ref[i]*core_ref[i]; }
  printf("core vs golden maxerr=%.3e at %zu (t=%zu i=%zu) gpu=%.6f ref=%.6f cos=%.8f\n",
    mxc,ic,ic/HID,ic%HID,core_gpu[ic],core_ref[ic],dotc/(sqrt(nac*nbc)+1e-12));
  size_t free1=0,tot1=0; cudaMemGetInfo(&free1,&tot1);
  printf("vram_after free=%zu MiB delta=%zd MiB ms: wz=%.3f wb=%.3f wa=%.3f gdn=%.3f out=%.3f\n",
    free1/(1024*1024),(long)((long)free0-(long)free1)/(1024*1024),ms_wz,ms_wb,ms_wa,ms_gdn,ms_out);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  cudaFree(d_h); cudaFree(d_Wz); cudaFree(d_zl); cudaFree(d_Wb); cudaFree(d_b);
  cudaFree(d_qn); cudaFree(d_kn); cudaFree(d_v); cudaFree(d_be); cudaFree(d_gl);
  cudaFree(d_S); cudaFree(d_o); cudaFree(d_g); cudaFree(d_Wo); cudaFree(d_core);
  bool pass_o=(mxo<5e-3), pass_b=(mxb<5e-3), pass_g=(mbg<5e-3);
  printf("STAGE3 beta %s gl %s o %s core maxerr=%.3e (report)\n",
    pass_b?"PASS":"FAIL",pass_g?"PASS":"FAIL",pass_o?"PASS":"FAIL",mxc);
  return (pass_b&&pass_g&&pass_o)?0:1;
}

static int run_stage4(hermes::GgufWeightBackend& be){
  printf("\n=== STAGE4 residuo+post_norm+MLP T=%d ===\n",T);
  size_t free0=0,tot0=0; cudaMemGetInfo(&free0,&tot0);
  printf("vram_before free=%zu MiB\n",free0/(1024*1024));
  const int FFN=17408;
  const int ids[T]={6205,161421,303,220,17,1374,9911,1147,27053,6059,13};
  auto vEmb=be.view("token_embd.weight");
  auto vNorm=be.view("blk.0.attn_norm.weight");
  auto vWz=be.view("blk.0.attn_gate.weight");
  auto vWb=be.view("blk.0.ssm_beta.weight");
  auto vWa=be.view("blk.0.ssm_alpha.weight");
  auto vA=be.view("blk.0.ssm_a");
  auto vDT=be.view("blk.0.ssm_dt.bias");
  auto vWn=be.view("blk.0.ssm_norm.weight");
  auto vWout=be.view("blk.0.ssm_out.weight");
  auto vPost=be.view("blk.0.post_attention_norm.weight");
  auto vFg=be.view("blk.0.ffn_gate.weight");
  auto vFu=be.view("blk.0.ffn_up.weight");
  auto vFd=be.view("blk.0.ffn_down.weight");
  const uint16_t* embBase=(const uint16_t*)vEmb.data;
  const float* wNorm=(const float*)vNorm.data;
  std::vector<float> x((size_t)T*HID), h((size_t)T*HID);
  for(int t=0;t<T;++t){ const uint16_t* row=embBase+(size_t)ids[t]*HID;
    for(int i=0;i<HID;++i) x[(size_t)t*HID+i]=hermes::bf16_to_f32(row[i]); }
  for(int t=0;t<T;++t){ double ss=0; for(int i=0;i<HID;++i) ss+=(double)x[(size_t)t*HID+i]*x[(size_t)t*HID+i];
    float inv=1.f/sqrtf((float)(ss/HID)+1e-6f);
    for(int i=0;i<HID;++i) h[(size_t)t*HID+i]=x[(size_t)t*HID+i]*inv*wNorm[i]; }
  std::vector<float> cv_ref, y_ref, core_ref;
  if(!read_bin(F_CV,cv_ref,(size_t)T*C)) return 1;
  if(!read_bin(F_Y,y_ref,(size_t)T*HID)) return 1;
  if(!read_bin(F_CORE,core_ref,(size_t)T*HID)) return 1;
  // Recompute full blk.0 chain on GPU where heavy, CPU for tiny elementwise.
  // To isolate stage4, use golden core for residual (exact), then post_norm+MLP on GPU.
  // h1 = x + core_golden
  std::vector<float> h1((size_t)T*HID);
  for(size_t i=0;i<h1.size();++i) h1[i]=x[i]+core_ref[i];
  const float* wPost=(const float*)vPost.data;
  std::vector<float> h2((size_t)T*HID);
  for(int t=0;t<T;++t){ double ss=0; for(int i=0;i<HID;++i) ss+=(double)h1[(size_t)t*HID+i]*h1[(size_t)t*HID+i];
    float inv=1.f/sqrtf((float)(ss/HID)+1e-6f);
    for(int i=0;i<HID;++i) h2[(size_t)t*HID+i]=h1[(size_t)t*HID+i]*inv*wPost[i]; }
  printf("h1[0..2]=%.5f %.5f %.5f h2[0..2]=%.5f %.5f %.5f\n",h1[0],h1[1],h1[2],h2[0],h2[1],h2[2]);
  // MLP via GEMV (NVFP4 direct, BF16 in, FP32 out)
  std::vector<uint16_t> h2_bf16(h2.size());
  for(size_t i=0;i<h2.size();++i) h2_bf16[i]=hermes::f32_to_bf16(h2[i]);
  uint16_t* d_h2=nullptr; cudaMalloc(&d_h2,(size_t)T*HID*2);
  cudaMemcpy(d_h2,h2_bf16.data(),(size_t)T*HID*2,cudaMemcpyHostToDevice);
  size_t fg_bytes=(size_t)FFN*(HID/64*36);
  size_t fd_bytes=(size_t)HID*(FFN/64*36);
  uint8_t *d_Fg=nullptr,*d_Fu=nullptr,*d_Fd=nullptr;
  float *d_gg=nullptr,*d_uu=nullptr,*d_down=nullptr;
  cudaMalloc(&d_Fg,fg_bytes); cudaMalloc(&d_Fu,fg_bytes);
  cudaMalloc(&d_gg,(size_t)T*FFN*4); cudaMalloc(&d_uu,(size_t)T*FFN*4);
  cudaMemcpy(d_Fg,(const uint8_t*)vFg.data,fg_bytes,cudaMemcpyHostToDevice);
  cudaMemcpy(d_Fu,(const uint8_t*)vFu.data,fg_bytes,cudaMemcpyHostToDevice);
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for(int t=0;t<T;++t){ hermes::nvfp4_gemv_cuda(d_h2+t*HID,d_Fg,d_gg+t*FFN,HID,FFN);
    hermes::nvfp4_gemv_cuda(d_h2+t*HID,d_Fu,d_uu+t*FFN,HID,FFN); }
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms_gu=0; cudaEventElapsedTime(&ms_gu,e0,e1);
  std::vector<float> gg((size_t)T*FFN), uu((size_t)T*FFN);
  cudaMemcpy(gg.data(),d_gg,(size_t)T*FFN*4,cudaMemcpyDeviceToHost);
  cudaMemcpy(uu.data(),d_uu,(size_t)T*FFN*4,cudaMemcpyDeviceToHost);
  printf("ffn gate/up GEMV ms=%.4f total (N=%d K=%d T=%d)\n",ms_gu,FFN,HID,T);
  // silu*up on CPU (elementwise exact)
  std::vector<float> hid((size_t)T*FFN);
  for(size_t i=0;i<hid.size();++i){ float g=gg[i]; hid[i]=(g/(1.f+expf(-g)))*uu[i]; }
  // down proj GEMV
  std::vector<uint16_t> hid_bf16(hid.size());
  for(size_t i=0;i<hid.size();++i) hid_bf16[i]=hermes::f32_to_bf16(hid[i]);
  uint16_t* d_hid=nullptr; cudaMalloc(&d_hid,(size_t)T*FFN*2);
  cudaMalloc(&d_Fd,fd_bytes); cudaMalloc(&d_down,(size_t)T*HID*4);
  cudaMemcpy(d_hid,hid_bf16.data(),(size_t)T*FFN*2,cudaMemcpyHostToDevice);
  cudaMemcpy(d_Fd,(const uint8_t*)vFd.data,fd_bytes,cudaMemcpyHostToDevice);
  cudaEventRecord(e0);
  for(int t=0;t<T;++t) hermes::nvfp4_gemv_cuda(d_hid+t*FFN,d_Fd,d_down+t*HID,FFN,HID);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms_down=0; cudaEventElapsedTime(&ms_down,e0,e1);
  std::vector<float> down((size_t)T*HID);
  cudaMemcpy(down.data(),d_down,(size_t)T*HID*4,cudaMemcpyDeviceToHost);
  printf("ffn down GEMV ms=%.4f (N=%d K=%d)\n",ms_down,HID,FFN);
  std::vector<float> y_gpu((size_t)T*HID);
  for(size_t i=0;i<y_gpu.size();++i) y_gpu[i]=h1[i]+down[i];
  double mx=0; size_t im=0; double dot=0,na=0,nb=0;
  for(size_t i=0;i<y_gpu.size();++i){ float e=fabsf(y_gpu[i]-y_ref[i]); if(e>mx){mx=e;im=i;}
    dot+=(double)y_gpu[i]*(double)y_ref[i]; na+=(double)y_gpu[i]*y_gpu[i]; nb+=(double)y_ref[i]*y_ref[i]; }
  printf("y vs golden maxerr=%.3e at %zu (t=%zu i=%zu) gpu=%.6f ref=%.6f cos=%.8f\n",
    mx,im,im/HID,im%HID,y_gpu[im],y_ref[im],dot/(sqrt(na*nb)+1e-12));
  printf("y_gpu[0..4]=%.5f %.5f %.5f %.5f %.5f\n",y_gpu[0],y_gpu[1],y_gpu[2],y_gpu[3],y_gpu[4]);
  printf("y_ref[0..4]=%.5f %.5f %.5f %.5f %.5f\n",y_ref[0],y_ref[1],y_ref[2],y_ref[3],y_ref[4]);
  size_t free1=0,tot1=0; cudaMemGetInfo(&free1,&tot1);
  printf("vram_after free=%zu MiB delta=%zd MiB ms gu=%.3f down=%.3f\n",
    free1/(1024*1024),(long)((long)free0-(long)free1)/(1024*1024),ms_gu,ms_down);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  cudaFree(d_h2); cudaFree(d_Fg); cudaFree(d_Fu); cudaFree(d_gg); cudaFree(d_uu);
  cudaFree(d_hid); cudaFree(d_Fd); cudaFree(d_down);
  bool pass=(mx<0.1);
  printf("STAGE4 %s (thr 0.1 report, core isolato golden)\n",pass?"PASS":"FAIL");
  return pass?0:1;
}

int main(int argc,char** argv){
  std::string only = (argc>1)? argv[1] : "all";
  hermes::GgufWeightBackend be;
  try{ be.open(CKPT); }catch(std::exception& e){ printf("GGUF OPEN FAIL: %s\n",e.what()); return 1; }
  int rc=0;
  if(only=="all"||only=="s1"){ int r=run_stage1(be,nullptr,nullptr); if(r!=0) rc=r; if(only=="s1") return rc; }
  if(only=="all"||only=="s2"){ int r=run_stage2(be); if(r!=0) rc=r; if(only=="s2") return rc; }
  if(only=="all"||only=="s3"){ int r=run_stage3(be); if(r!=0) rc=r; if(only=="s3") return rc; }
  if(only=="all"||only=="s4"){ int r=run_stage4(be); if(r!=0) rc=r; }
  return rc;
}
