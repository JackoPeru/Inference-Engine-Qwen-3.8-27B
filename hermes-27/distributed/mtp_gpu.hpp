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
#include <mutex>
#include <string>
#include <vector>

namespace mtp_gpu {

static const int HID = 5120;
static const int VOC = 248320;
static const int MAXL = 128; // capacita' default legacy
static const int MAX_CONTEXT = 8192; // limite pubblicato da pp2::DecodeState

struct MtpState {
  // Host-only serialization of every lifetime and KV bookkeeping operation.
  // Do not share a state while destroying it; caller owns that lifetime.
  // Noncopyable: copies previously aliased CUDA K/V and risked double free.
  mutable std::mutex mutex;
  float* Kc = nullptr; // [capacity*1024] fp32, GPU1 (K normate+RoPE)
  float* Vc = nullptr; // [capacity*1024] fp32, GPU1 (V raw)
  int len = 0;         // posizioni occupate (host mirror)
  int capacity = MAXL;
  long first_pos = -1; // absolute RoPE position of the first cached token
  bool inited = false;
  // Sticky after any CUDA failure during a draft step, including a partial KV
  // append. reset()/rewind() cannot clear this. free_state()+init() required.
  bool poisoned = false;
  MtpState() = default;
  MtpState(const MtpState&) = delete;
  MtpState& operator=(const MtpState&) = delete;
  MtpState(MtpState&&) = delete;
  MtpState& operator=(MtpState&&) = delete;
};

// Strict BLOCKED: public PP2 API has no verification of K>1 tentative tokens
// from a live DecodeState followed by atomic accepted-prefix GDN/CS/KV commit.
// This constexpr is deliberately false and must not be mistaken for a route.
inline constexpr bool target_batch_rollback_supported() noexcept { return false; }

// Pure host bookkeeping shared by the GPU implementation and the CPU-only
// contract test. Must run under MtpState::mutex in all production calls.
namespace detail {
inline bool can_append(const MtpState& st, long pos) noexcept {
  if (!st.inited || st.poisoned || st.len < 0 || st.capacity < 1 ||
      st.capacity > MAX_CONTEXT || st.len >= st.capacity ||
      pos < 0 || pos >= MAX_CONTEXT ||
      st.first_pos < -1 || st.first_pos >= MAX_CONTEXT) return false;
  if (st.first_pos < 0) return st.len == 0;
  // Also binds replay position after rewind(0); only explicit reset() may
  // start a new origin. No jumps or duplicate absolute positions.
  return pos == st.first_pos + st.len;
}
inline bool commit_append(MtpState& st, long pos) noexcept {
  if (!can_append(st,pos)) return false;
  if (st.first_pos < 0) st.first_pos = pos;
  ++st.len;
  return true;
}
inline bool rewind_metadata(MtpState& st, int keep_len) noexcept {
  if (!st.inited || st.poisoned || st.capacity < 1 ||
      st.capacity > MAX_CONTEXT || st.len < 0 || st.len > st.capacity ||
      st.first_pos < -1 || st.first_pos >= MAX_CONTEXT ||
      (st.first_pos < 0 && st.len != 0) ||
      (st.first_pos >= 0 && st.first_pos + st.len > MAX_CONTEXT) ||
      keep_len < 0 || keep_len > st.len)
    return false;
  st.len = keep_len;
  // Preserve first_pos even at zero. Rewind means replay the SAME prefix,
  // while reset() alone grants a new prompt/position.
  return true;
}
inline void reset_metadata(MtpState& st) noexcept {
  if (!st.inited) return;
  st.len = 0;
  st.first_pos = -1;
  // Intentionally leave a poisoned CUDA state poisoned.
}
} // namespace detail

// Alloca Kc/Vc su GPU1 (richiede pp2::upload gia` fatto). Ritorna false su errore/OOM.
bool init(MtpState& st, std::string& err);
bool init(MtpState& st, std::string& err, int capacity);
// Azzera len per nuovo prompt solo se non poisoned. In caso poisoned, rimane
// inutilizzabile fino a free_state()+init() (void legacy API, no hidden retry).
void reset(MtpState& st);
// Truncate only draft KV logical length, preserving the FIRST absolute pos
// even for rewind(0): replay must begin at that same position.
// All retained positions must have been verified against target (or rebuilt
// from a verified prefix) by the caller.
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
// Ritorna vuoto su errore, leaves len/first_pos logically unchanged and, for
// any CUDA failure, latches poisoned=true. Caller must discard that state.
// Does NOT manage or roll back any target GDN S, conv history, KV or EOS.
std::vector<float> draft_step(const float* h_last, int tok, MtpState& st, long pos,
                              std::string& err);

// Utility greedy (stesso criterio test_mtp_accept, su logits fp32 host).
int argmax_v(const std::vector<float>& logits);
void top5_v(const std::vector<float>& logits, int* out_id, float* out_val);
// Overlap top5 (0..5) tra due logits.
int top5_overlap(const std::vector<float>& a, const std::vector<float>& b);
bool finite_all(const std::vector<float>& v, double& mn, double& mx, double& mean);

} // namespace mtp_gpu
