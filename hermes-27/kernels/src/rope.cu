#include "hermes/rope.hpp"
#include "hermes/rmsnorm.hpp"
#include <cuda_bf16.h>
#include <cmath>
namespace hermes {
// GPT-J split-half style (matches transformers Qwen3_5 apply_rotary_pos_emb):
// q_rot = x[0:R], q_pass = x[R:]; halves a=x[0:R/2], b=x[R/2:R];
// y[0:R/2]   = a*cos - b*sin ; y[R/2:R] = a*sin + b*cos ; y[R:] = x[R:].
// cos/sin: [R] duplicated halves [c,c] (mRoPE-text: identical T/H/W streams).
void rope_cpu(const uint16_t* x, uint16_t* y, int head_dim, int rotary_dim, long pos, float theta){
  int R = rotary_dim, H2 = R/2;
  float c[256], s[256];
  for(int i=0;i<H2;++i){ float f=1.0f/powf(theta,(float)(2*i)/R); float a=(float)pos*f; c[i]=cosf(a); s[i]=sinf(a); }
  for(int i=0;i<H2;++i){
    float a=bf16_to_f32(x[i]), b=bf16_to_f32(x[H2+i]);
    y[i]=f32_to_bf16(a*c[i]-b*s[i]); y[H2+i]=f32_to_bf16(a*s[i]+b*c[i]);
  }
  for(int i=R;i<head_dim;++i) y[i]=x[i];
}
__global__ void rope_k(const __nv_bfloat16* x, __nv_bfloat16* y, int head_dim, int rotary_dim, long pos, float theta){
  int d=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(d>=head_dim) return;
  if(d>=rotary_dim){ y[d]=x[d]; return; }
  int H2=rotary_dim/2;
  int i = d<H2 ? d : d-H2;           // freq index
  int is_b = d>=H2 ? 1 : 0;          // second half?
  float freq=1.0f/powf(theta,(float)(2*i)/rotary_dim);
  float ang=(float)pos*freq, cc=cosf(ang), ss=sinf(ang);
  float a=(float)x[i], b=(float)x[H2+i];
  y[d]=__nv_bfloat16(is_b ? a*ss+b*cc : a*cc-b*ss);
}
void rope_cuda(const uint16_t* x, uint16_t* y, int head_dim, int rotary_dim, long pos, float theta){
  int blk=256; int grd=(head_dim+blk-1)/blk;
  rope_k<<<grd,blk>>>((const __nv_bfloat16*)x,(__nv_bfloat16*)y,head_dim,rotary_dim,pos,theta);
}
}
