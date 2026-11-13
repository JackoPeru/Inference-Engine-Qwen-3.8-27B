#pragma once
// HERMES-27 MTP/NextN draft su GPU (blk.64, full-attn BF16, KV propria).
// Usa pesi residenti pp2 (GPU1, single-copy) via pp2::get_weight_ptr: nessun
// duplicato in VRAM. Solo stato draft Kc/Vc (~1 MiB) + workspace per-step.
// Wiring: replica ModelFwd::mtp_draft (concat [e;h], norme dirette, q split
// interleaved, q/k RMSNorm, RoPE mRoPE-text pos assoluta, att*sigmoid(gate),
// out_proj, residui, FFN, shared_head_norm + head condivisa).
// Solo greedy. Sampling distributivo e rollback GDN: fuori scopo (vedi report).
#include <cstdint>
#include <string>
#include <vector>

namespace mtp_gpu {

static const int HID = 5120;
static const int VOC = 248320;
static const int MAXL = 128; // KV draft max (come pp2::DecodeState::MAXL)

struct MtpState {
  float* Kc = nullptr; // [MAXL*1024] fp32, GPU1 (K normate+RoPE)
  float* Vc = nullptr; // [MAXL*1024] fp32, GPU1 (V raw)
  int len = 0;         // posizioni occupate (host mirror)
  bool inited = false;
};

// Alloca Kc/Vc su GPU1 (richiede pp2::upload gia` fatto). Ritorna false su errore/OOM.
bool init(MtpState& st, std::string& err);
// Azzera len (stessa allocazione, per nuovo prompt). Non fallisce se inited.
void reset(MtpState& st);
// Libera Kc/Vc (pesi pp2 restano residenti).
void free_state(MtpState& st);

// Singolo passo draft: h_last host [HID] fp32 (hidden target pre-output_norm,
// da pp2::prefill con h_last_out) + tok (draft token precedente) a pos assoluta
// -> logits [VOC] fp32 host. Aggiorna Kc/Vc (append). Ritorna vuoto su errore.
std::vector<float> draft_step(const float* h_last, int tok, MtpState& st, long pos,
                              std::string& err);

// Utility greedy (stesso criterio test_mtp_accept, su logits fp32 host).
int argmax_v(const std::vector<float>& logits);
void top5_v(const std::vector<float>& logits, int* out_id, float* out_val);
// Overlap top5 (0..5) tra due logits.
int top5_overlap(const std::vector<float>& a, const std::vector<float>& b);
bool finite_all(const std::vector<float>& v, double& mn, double& mx, double& mean);

} // namespace mtp_gpu
