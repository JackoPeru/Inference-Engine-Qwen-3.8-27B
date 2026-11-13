#pragma once
// HERMES-27 CPU full-model forward (fp32) for Qwen3.8-27B NVFP4 GGUF.
// Equations verified vs transformers+qwen3_5/fla (see gdn_block).
// Prefill path; decode/MTP follow the same verified recurrences.
#include <cstdint>
#include <string>
#include <vector>
namespace hermes {
struct ModelFwd {
  // dims (verified on real checkpoint)
  static constexpr int HID=5120, FFN=17408, NL=64, VOC=248320;
  static constexpr int NQ=24, NKV=4, HD=256, RD=64;
  static constexpr int KD=16*128, VD=48*128, C=2*KD+VD, HV=48, KHD=128, VHD=128;
  std::string path;
  const uint8_t* base=nullptr; uint64_t len=0; int fd=-1;
  struct TI { std::string name; int ne0, ne1; int type; uint64_t off; };
  std::vector<TI> tensors; uint64_t data_off=0;
  bool open(const std::string& gguf_path);
  void close();
  const TI* find(const std::string& n) const;
  // views (no copy): F32/BF16 raw; NVFP4 raw blocks
  const float* f32(const std::string& n) const;
  const uint16_t* bf16(const std::string& n) const;
  const uint8_t* nvfp4(const std::string& n) const;
  static float bf2f(uint16_t b);
  // dequant one NVFP4 row (ne0 elems, ne0%64==0) into out
  static void deq_row(const uint8_t* row, float* out, int ne0);
  // GEMV: y[i] = sum_k W[i*ld+k]*x[k], W fp32 row-major [rows,cols]
  static void gemv(const float* W, const float* x, float* y, int rows, int cols);
  // GEMV with NVFP4 GGUF row-major [rows,cols] (dequant on the fly per row)
  void gemv_nvfp4(const std::string& wname, const float* x, float* y) const;
  // Batched GEMM with NVFP4 GGUF [out,in] (dequant once, then fp32)
  void linear_nvfp4_batch(const std::string& wname, const float* X, float* Y, int T) const;
  // Row-sharded variant (for the 248320-row lm head): rows [r0,r1)
  void linear_nvfp4_rows(const std::string& wname, const float* X, float* Y, int T, int r0, int r1) const;
  void rms(const float* x, const float* w, float* y, int n, bool plus_one) const;
  // full prefill: ids -> logits[T*VOC] (fp32).
  // h_last_out (optional [HID]): last hidden before output_norm (for MTP draft).
  std::vector<float> prefill(const std::vector<int>& ids, float* h_last_out = nullptr) const;
  // ---- incremental decode (single token, state carry) ----
  struct SeqState {
    // per-layer: linear: conv [C*3], gdn [HV*KHD*VHD]; full: KV handled externally (see below)
    std::vector<float> conv, gdn;
    std::vector<std::vector<float>> kcache, vcache; // per full layer: [L*NKV*HD] grows
    std::vector<int> klen; // per full layer
  };
  void prefill_with_state(const std::vector<int>& ids, std::vector<float>& logits_last, SeqState& st) const;
  // one decode step: tok -> logits[VOC] (fp32, sharded internally); updates st; pos = absolute position
  std::vector<float> decode_step(int tok, SeqState& st, long pos) const;
  // ---- MTP/NextN draft (blk.64, BF16 single layer, verified wiring in MTP_DESIGN.md) ----
  struct MtpState {
    std::vector<float> kcache, vcache; // draft layer KV: [L*NKV*HD] grows
    int len = 0;
  };
  // Last target hidden (pre-output_norm) + next token -> draft hidden.
  // h_last: [HID] fp32 (x after final layer, before output_norm).
  void mtp_hidden(const float* h_last, int tok, float* x_out) const;
  // Draft decode step with KV carry: h_last + tok -> draft logits[VOC]; pos = absolute.
  std::vector<float> mtp_draft(const float* h_last, int tok, MtpState& mst, long pos) const;
};
}
