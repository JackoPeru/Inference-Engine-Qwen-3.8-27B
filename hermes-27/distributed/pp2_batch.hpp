#pragma once
// HERMES-27 P5 batch concorrente PP2 (N richieste, stati separati, overlap pipeline).
// Ri riuso: pesi upload + kernel esistenti via pp2::decode_step (nessuna duplicazione pesi).
// DecodeState attuale = 1 richiesta -> qui BatchState = N DecodeState indipendenti
// (S/CS/KV per request, stessa VRAM budget: 16KB/tok/req/GPU KV + ~75MB/req/GPU GDN).
#include "pp2_fwd.hpp"
#include <vector>
#include <string>

namespace pp2batch {

struct BatchState {
  int N = 0;
  std::vector<pp2::DecodeState> reqs;
  bool inited = false;
};

// Inizializza N stati (chiama pp2::decode_init per ogni request).
// Ritorna false se OOM o VRAM>15GB/GPU (abort hard come P2/P4).
bool batch_init(BatchState& b, int N, std::string& err);

// Libera tutti gli stati (pesi upload restano residenti).
void batch_free(BatchState& b);

// Step sequenziale onesto (loop richieste con kernel esistenti, v1):
// toks.size()==N, stessa pos per tutte (prompt stessa lunghezza, caso GATE1).
// Ritorna N logits[VOC], vuoto su errore.
std::vector<std::vector<float>> batch_step(const std::vector<int>& toks,
                                           BatchState& b, long pos,
                                           std::string& err);

// Step parallelo con overlap pipeline:
// mentre GPU1 fa seconda meta` di req A, GPU0 fa prima meta` di req B.
// Implementato via host threads concorrenti su pp2::decode_step (device diversi
// eseguono davvero in parallelo; xfer pinned protetto da mutex in pp2_fwd).
// poss.size()==N (pos assoluta per-request, permette prompt di lunghezza diversa).
std::vector<std::vector<float>> batch_step_parallel(const std::vector<int>& toks,
                                                    BatchState& b,
                                                    const std::vector<long>& poss,
                                                    std::string& err);

// Overload stessa pos per tutte (comodo per GATE1/GATE2 con prompt pari lunghezza).
std::vector<std::vector<float>> batch_step_parallel(const std::vector<int>& toks,
                                                    BatchState& b, long pos,
                                                    std::string& err);

// Ammissione target da reports/VRAM_BUDGET_V2.md:
// need/GPU = Tctx*16KB + 75MB (+78/72MB split PP-B, qui 75MB medio).
// Cap: max 16 @<=1K, 8 @<=8K, 2 @32K. Ritorna false con detail se oltre cap.
bool admit_check(int N, int Tctx, std::string& detail);

// VRAM usata per GPU (MiB) via cudaMemGetInfo, per report no-leak.
void vram_used(size_t& used0_MiB, size_t& used1_MiB);

} // namespace pp2batch
