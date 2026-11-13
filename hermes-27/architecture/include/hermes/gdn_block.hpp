#pragma once
// Gated DeltaNet linear block, CPU fp32 reference (tiny-config generic).
// Equations verified vs transformers Qwen3_5GatedDeltaNet + fla pure-torch
// fallbacks: conv+SiLU, split q|k|v, GVA repeat, beta=sigmoid, g=-exp(A)*
// softplus(a+dt), qk-l2norm, scale 1/sqrt(K), delta recurrence, RMSNormGated
// (norm*w*silu(z)), out_proj. Full-prefill == chunked == token-by-token.
#include <vector>
namespace hermes {
struct GdnBlockF32 {
  int B=1,T=1,Hid=128,Hk=2,Khd=16,Hv=4,Vhd=16,convK=4;
  std::vector<float> Wqkv,Wz,Wb,Wa,Wconv,Alog,dt,Wnorm,Wo; // row-major as torch [out,in]
  static GdnBlockF32 load(const char* path); // flat bin from gdn_export.py
  // prefill: x [T,Hid] -> y [T,Hid]; returns final state [Hv,Khd,Vhd]
  std::vector<float> forward_prefill(const std::vector<float>& x, std::vector<float>& state) const;
  // decode one token: x [Hid], state in/out [Hv,Khd,Vhd] -> y [Hid]
  std::vector<float> forward_step(const std::vector<float>& x, std::vector<float>& state,
                                  std::vector<float>& conv_state) const; // conv_state [C,convK-1]
};
}
