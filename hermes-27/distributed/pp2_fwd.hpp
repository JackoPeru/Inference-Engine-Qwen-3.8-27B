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

// Prefill singolo microbatch: ids[T] -> logits[T*VOC] fp32 (solo ultimo? no, tutti T come E2).
// xfer confine via pinned host, misurato in st.xfer_ms. Ritorna vuoto su errore.
std::vector<float> prefill(const std::vector<int>& ids, Stats& st, std::string& err);

// Libera tutto (entrambe GPU + pinned + backend).
void free_all();

// GATE2: singolo layer via path P2 (usa pesi residenti, non mmap diretto).
// blk0 su GPU0 (GDN, vs golden /tmp/e1_*), blk32 su GPU1 (GDN, GPUvsCPU-dequant + GDN isolated).
// Ritorna 0 ok, !=0 fail. Stampa numeri reali (maxerr/cos/ms) come P1.
int check_blk0_gpu0();
int check_blk32_gpu1();

// Utility per report: top5 da logits ultimo token.
void top5(const float* logits_TxV, int T, int V, int* out_id, float* out_val);

} // namespace pp2
