#pragma once
// HERMES-27 MTP/NextN draft su GPU (blk.64, full-attn BF16, KV propria).
// Usa pesi residenti pp2 (GPU1, single-copy) via pp2::get_weight_ptr: nessun
// duplicato in VRAM. Solo stato draft Kc/Vc (~1 MiB) + workspace per-step.
// Wiring: replica ModelFwd::mtp_draft (concat [e;h], norme dirette, q split
// interleaved, q/k RMSNorm, RoPE mRoPE-text pos assoluta, att*sigmoid(gate),
// out_proj, residui, FFN, shared_head_norm + head condivisa).
// Solo greedy. Il blocco draft non modifica mai la cache target/GDN: il target
// deve verificare ogni proposta prima di emettere token speculative.
#include <cstdint>
#include <string>
#include <vector>

namespace mtp_gpu {

static const int HID = 5120;
static const int VOC = 248320;
static const int MAXL = 128; // capacita' default legacy
static const int MAX_CONTEXT = 8192; // limite pubblicato da pp2::DecodeState

struct MtpState {
  float* Kc = nullptr; // [capacity*1024] fp32, GPU1 (K normate+RoPE)
  float* Vc = nullptr; // [capacity*1024] fp32, GPU1 (V raw)
  int len = 0;         // posizioni occupate (host mirror)
  int capacity = MAXL;
  long first_pos = -1; // absolute RoPE position of the first cached token
  bool inited = false;
};

// Alloca Kc/Vc su GPU1 (richiede pp2::upload gia` fatto). Ritorna false su errore/OOM.
bool init(MtpState& st, std::string& err);
bool init(MtpState& st, std::string& err, int capacity);
// Azzera len (stessa allocazione, per nuovo prompt). Non fallisce se inited.
void reset(MtpState& st);
// Truncate only draft KV logical length. All retained positions must have been
// verified against target (or rebuilt from a verified prefix) by the caller.
// Rejected speculative tokens must NEVER advance target state.
bool rewind(MtpState& st, int keep_len, std::string& err);
// Libera Kc/Vc (pesi pp2 restano residenti).
void free_state(MtpState& st);

// Singolo passo draft: h_last host [HID] fp32 (hidden target pre-output_norm,
// da pp2::prefill con h_last_out) + tok (draft token precedente) a pos assoluta
// -> logits [VOC] fp32 host. Aggiorna Kc/Vc solo a step riuscito.
// Le posizioni devono essere consecutive da first_pos (senza buchi); i blocchi
// con prompt > 128 usano init(..., capacity) e attenzione dinamica.
// Il chiamante deve fornire hidden TARGET dell'elemento precedente, mai hidden
// MTP e mai un unico h_last congelato per K draft autoregressivi.
// Ritorna vuoto su errore e lascia len/first_pos logicamente invariati.
std::vector<float> draft_step(const float* h_last, int tok, MtpState& st, long pos,
                              std::string& err);

// Utility greedy (stesso criterio test_mtp_accept, su logits fp32 host).
int argmax_v(const std::vector<float>& logits);
void top5_v(const std::vector<float>& logits, int* out_id, float* out_val);
// Overlap top5 (0..5) tra due logits.
int top5_overlap(const std::vector<float>& a, const std::vector<float>& b);
bool finite_all(const std::vector<float>& v, double& mn, double& mx, double& mean);

} // namespace mtp_gpu
