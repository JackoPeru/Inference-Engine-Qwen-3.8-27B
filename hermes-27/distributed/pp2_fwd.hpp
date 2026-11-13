#pragma once
// HERMES-27 PP2 forward prefill (P2): pipeline-parallel 2 stadi su 2x RTX 5060 Ti.
// PP-B effettivo qui: GPU0 embed+blk0-31, GPU1 blk32-63+head (32+32, xfer 1 confine).
// Pesi NVFP4 raw GGUF residenti (single-copy, GEMV preciso come P1 s3/s4).
// Conversione CUTLASS misurata su 1 matrice (report), non duplicata in VRAM.
// API: pp2_upload / pp2_prefill / pp2_free + check singolo layer per GATE2.
#include <cstdint>
#include <string>
#include <vector>

namespace pp2 {

struct Stats {
  double upload_sec = 0;      // tempo totale upload H2D entrambi GPU
  double convert_ms_one = 0;  // build_B_from_gguf su 1 matrice rappresentativa
  double prefill_ms = 0;      // tempo forward completo T=11 (kernel only, senza upload)
  double xfer_ms = 0;         // GPU0->host->GPU1 confine (pinned)
  size_t bytes_gpu0 = 0;      // byte residenti stimati da upload (somma cudaMalloc)
  size_t bytes_gpu1 = 0;
  size_t free0_before = 0, free1_before = 0;
  size_t free0_after = 0, free1_after = 0;
};

// Upload completo pesi su GPU0/GPU1 secondo PP (split 31/32). Ritorna false se VRAM>15GB o errore CUDA.
// Misura tempi, conta byte, fa checksum spot (1 tensore/GPU vs CPU).
bool upload(Stats& st, std::string& err);
// Explicit checkpoint path for CLI/runtime; legacy overload remains compatible.
bool upload(Stats& st, std::string& err, const std::string& model_path);
int eos_token_id(); // tokenizer.ggml.eos_token_id, or -1 if absent/invalid

// Prefill singolo microbatch: ids[T] -> logits[T*VOC] fp32 (solo ultimo? no, tutti T come E2).
// xfer confine via pinned host, misurato in st.xfer_ms. Ritorna vuoto su errore.
// h_last_out (opzionale): se non-null, riempito con ultimo hidden [HID] fp32
// pre-output_norm (per MTP draft). Nessun costo extra oltre 20KB D2H.
std::vector<float> prefill(const std::vector<int>& ids, Stats& st, std::string& err,
                           std::vector<float>* h_last_out = nullptr);

// Accessor minimale MTP: raw device pointer peso residente (no copy, no ownership).
// Ritorna false se non uploaded o nome assente. ne0/ne1/gtype/dev opzionali.
bool get_weight_ptr(const std::string& name, void*& dptr, int* ne0 = nullptr,
                    int* ne1 = nullptr, int* gtype = nullptr, int* dev = nullptr);

// Libera tutto (entrambe GPU + pinned + backend).
void free_all();

// P4 decode incrementale (PP2, stati su GPU che ospita il layer).
// Stati per posizione (cfr VRAM_BUDGET_V2): GDN S [48lin x 48x128x128 fp32],
// conv tail [C=10240 x 3] per layer lineare, KV full-attn [16 x 4x256 x L] fp32,
// RoPE usa pos assoluta (come E2 decode_step pos).
struct DecodeState {
  // Legacy decode_init() default remains 128. Extended states support up to 8K.
  static const int MAXL = 128;
  static constexpr int MAX_CONTEXT = 8192;
  float* S[64];   // GDN S persistente [HV*KHD*VHD], nullptr per full layer
  float* CS[64];  // conv tail [C*3], nullptr per full layer
  float* Kc[64];  // full K cache [capacity*4*256], nullptr per GDN layer
  float* Vc[64];  // full V cache [capacity*4*256], nullptr per GDN layer
  int klen[64];   // len per full layer (host mirror)
  int capacity = MAXL;
  bool inited = false;
  // P6 workspace persistenti (no malloc/free in decode hot path, T=1 fisso):
  // arena riusata layer-per-layer, own device (ws0 GPU0, ws1 GPU1).
  void* ws0 = nullptr;    // [WS_BYTES=568064] GPU0
  void* ws1 = nullptr;    // [WS_BYTES=568064] GPU1
  int* idsbuf = nullptr;  // [1] GPU0 (tok input)
  float* x0buf = nullptr; // [HID=5120] GPU0 (residuo stream 0..31)
  float* x1buf = nullptr; // [HID=5120] GPU1 (residuo stream 32..63)
  float* xnbuf = nullptr; // [HID=5120] GPU1 (head norm)
  float* logbuf = nullptr;// [VOC=248320] GPU1 (head logits)
  size_t ws_bytes = 0;    // == 568064 quando inited
  // Strict opt-in HERMES_PP2_ASYNC_STREAMS=1: opaque per-request CUDA streams,
  // completion events and distinct pinned staging. Owned by decode_init/free.
  // nullptr means the original synchronous decode_step remains active.
  void* async_ctx = nullptr;
};

// Alloca/stati zero su GPU proprietaria (richiede upload già fatto). Ritorna false su errore/OOM.
bool decode_init(DecodeState& ds, std::string& err);
// Explicit KV/attention capacity in tokens (1..MAX_CONTEXT).
bool decode_init(DecodeState& ds, std::string& err, int capacity);
// Singolo step incrementale: tok + pos assoluta -> logits[VOC] fp32 (ultimo token).
// Aggiorna S/CS/KV in place. xfer T=1 via pinned (20KB). Ritorna vuoto su errore.
std::vector<float> decode_step(int tok, DecodeState& ds, long pos, std::string& err);
// Libera stati decode (pesi upload restano residenti).
void decode_free(DecodeState& ds);

// GATE2: singolo layer via path P2 (usa pesi residenti, non mmap diretto).
// blk0 su GPU0 (GDN, vs golden /tmp/e1_*), blk32 su GPU1 (GDN, GPUvsCPU-dequant + GDN isolated).
// Ritorna 0 ok, !=0 fail. Stampa numeri reali (maxerr/cos/ms) come P1.
int check_blk0_gpu0();
int check_blk32_gpu1();

// Utility per report: top5 da logits ultimo token.
void top5(const float* logits_TxV, int T, int V, int* out_id, float* out_val);
// Debug P4: confronta hidden per-layer ultimo token tra prefill T e decode incrementale (stessi ids).
// Ritorna indice primo layer divergente (>1e-4) o -1 se tutti ok. Stampa maxerr/mean per layer.
int debug_bisect_layer(const std::vector<int>& ids);
// Debug dettaglio layer 26 tok3: confronta q1/cv/beta/gl/o tra prefill-batched e decode-incr a parita di input h e S0=0.
int debug_layer26_detail();

} // namespace pp2
