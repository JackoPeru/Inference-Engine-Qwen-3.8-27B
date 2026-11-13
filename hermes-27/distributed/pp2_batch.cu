// HERMES-27 P5 batch concorrente PP2 — implementazione v1.
// Sequenziale: loop pp2::decode_step (corretto, overhead onesto).
// Parallelo: N host threads su pp2::decode_step (overlap reale GPU0/GPU1,
// xfer pinned reso thread-safe da mutex in pp2_fwd.cu).
#include "pp2_batch.hpp"
#include <cstdio>
#include <thread>
#include <mutex>
#ifndef HERMES_ADMISSION_CPU_ONLY
#include <cuda_runtime.h>
#endif

namespace pp2batch {

#ifndef HERMES_ADMISSION_CPU_ONLY
bool batch_init(BatchState& b, int N, std::string& err) {
  if (N <= 0 || N > 16) { err = "bad N (1..16)"; return false; }
  b.N = N;
  b.reqs.clear();
  b.reqs.resize((size_t)N);
  // DecodeState default: init a null come pp2::decode_init fa, ma assicuriamo
  for (int i = 0; i < N; ++i) {
    for (int l = 0; l < 64; ++l) {
      b.reqs[i].S[l] = nullptr; b.reqs[i].CS[l] = nullptr;
      b.reqs[i].Kc[l] = nullptr; b.reqs[i].Vc[l] = nullptr;
      b.reqs[i].klen[l] = 0;
    }
    b.reqs[i].inited = false;
  }
  for (int i = 0; i < N; ++i) {
    if (!pp2::decode_init(b.reqs[i], err)) {
      // free parziale
      for (int j = 0; j < i; ++j) pp2::decode_free(b.reqs[j]);
      b.reqs.clear(); b.N = 0; b.inited = false;
      err = "batch_init req " + std::to_string(i) + ": " + err;
      return false;
    }
  }
  // hard abort VRAM>15GB/GPU (come upload/decode_init)
  {
    size_t f0 = 0, t0 = 0, f1 = 0, t1 = 0;
    cudaSetDevice(0); cudaMemGetInfo(&f0, &t0);
    cudaSetDevice(1); cudaMemGetInfo(&f1, &t1);
    size_t u0 = t0 - f0, u1 = t1 - f1;
    const size_t LIM = 15ULL * 1024 * 1024 * 1024;
    printf("pp2batch init N=%d used0=%zu MiB used1=%zu MiB (limit 15360 MiB)\n",
           N, u0 / 1048576, u1 / 1048576);
    fflush(stdout);
    if (u0 > LIM || u1 > LIM) {
      err = "VRAM>15GB after batch_init N=" + std::to_string(N);
      for (int i = 0; i < N; ++i) pp2::decode_free(b.reqs[i]);
      b.reqs.clear(); b.N = 0; b.inited = false;
      return false;
    }
  }
  b.inited = true;
  return true;
}

void batch_free(BatchState& b) {
  if (!b.inited) { b.reqs.clear(); b.N = 0; return; }
  for (int i = 0; i < b.N; ++i) pp2::decode_free(b.reqs[i]);
  b.reqs.clear(); b.N = 0; b.inited = false;
}

std::vector<std::vector<float>> batch_step(const std::vector<int>& toks,
                                           BatchState& b, long pos,
                                           std::string& err) {
  if (!b.inited) { err = "batch not inited"; return {}; }
  if ((int)toks.size() != b.N) { err = "toks size != N"; return {}; }
  std::vector<std::vector<float>> outs;
  outs.reserve((size_t)b.N);
  for (int i = 0; i < b.N; ++i) {
    std::vector<float> lg = pp2::decode_step(toks[i], b.reqs[i], pos, err);
    if (lg.empty()) { err = "batch_step req " + std::to_string(i) + ": " + err; return {}; }
    outs.push_back(std::move(lg));
  }
  return outs;
}

std::vector<std::vector<float>> batch_step_parallel(const std::vector<int>& toks,
                                                    BatchState& b,
                                                    const std::vector<long>& poss,
                                                    std::string& err) {
  if (!b.inited) { err = "batch not inited"; return {}; }
  if ((int)toks.size() != b.N || (int)poss.size() != b.N) { err = "size != N"; return {}; }
  std::vector<std::vector<float>> outs((size_t)b.N);
  std::vector<std::string> errs((size_t)b.N);
  std::vector<char> ok((size_t)b.N, 1);
  std::vector<std::thread> th;
  th.reserve((size_t)b.N);
  for (int i = 0; i < b.N; ++i) {
    th.emplace_back([&, i]() {
      outs[i] = pp2::decode_step(toks[i], b.reqs[i], poss[i], errs[i]);
      if (outs[i].empty()) ok[i] = 0;
    });
  }
  for (auto& t : th) t.join();
  for (int i = 0; i < b.N; ++i) {
    if (!ok[i]) { err = "batch_step_parallel req " + std::to_string(i) + ": " + errs[i]; return {}; }
  }
  return outs;
}

std::vector<std::vector<float>> batch_step_parallel(const std::vector<int>& toks,
                                                    BatchState& b, long pos,
                                                    std::string& err) {
  std::vector<long> poss((size_t)b.N, pos);
  return batch_step_parallel(toks, b, poss, err);
}
#endif // HERMES_ADMISSION_CPU_ONLY

bool admit_check(int N, int Tctx, std::string& detail) {
  // Static admission envelope for the current PP2 FP32 state layout.
  // Each GPU holds 24 GDN layers and 8 full-attention layers (see decode_init).
  // Kc + Vc: 8 layers * 2 tensors * 1024 floats * 4 bytes = 64 KiB/token/GPU.
  // No FP8 KV exists in this implementation. The old 16 KiB/token was wrong.
  constexpr size_t MiB = 1024u * 1024u;
  constexpr size_t GDN = 24u * (48u * 128u * 128u + 10240u * 3u) * sizeof(float);
  constexpr size_t WS = 568064u;
  constexpr size_t STEP0 = sizeof(int) + 5120u * sizeof(float);
  constexpr size_t STEP1 = 2u * 5120u * sizeof(float) + 248320u * sizeof(float);
  constexpr size_t KV_BYTES_PER_TOKEN = 8u * 2u * 1024u * sizeof(float);

  // Baseline P3 post-upload/warm: GPU0 9856 MiB, GPU1 11470 MiB used.
  // These baselines already include a single state, so projecting all N states
  // on top is conservative. Reserve an extra 1 GiB/GPU for runtime variance.
  // Live VRAM is checked separately by serving::admit + decode_init preflight.
  constexpr size_t LIMIT_MIB = 15360u;
  constexpr size_t BASE0_MIB = 9856u;
  constexpr size_t BASE1_MIB = 11470u;
  constexpr size_t SAFETY_MIB = 1024u;
  constexpr int MAX_REQUESTS = 16; // policy limit, not a validated throughput cap
  constexpr size_t BUDGET0 = (LIMIT_MIB - BASE0_MIB - SAFETY_MIB) * MiB;
  constexpr size_t BUDGET1 = (LIMIT_MIB - BASE1_MIB - SAFETY_MIB) * MiB;

  if (N < 1 || N > MAX_REQUESTS || Tctx < 1 || Tctx > pp2::DecodeState::MAX_CONTEXT) {
    char reason[256];
    snprintf(reason, sizeof(reason),
             "admit N=%d Tctx=%d REJECT bounds N=1..%d context=1..%d (FP32 KV)",
             N, Tctx, MAX_REQUESTS, pp2::DecodeState::MAX_CONTEXT);
    detail = reason;
    return false;
  }

  const size_t per0 = GDN + KV_BYTES_PER_TOKEN * (size_t)Tctx + WS + STEP0;
  const size_t per1 = GDN + KV_BYTES_PER_TOKEN * (size_t)Tctx + WS + STEP1;
  const size_t total0 = (size_t)N * per0;
  const size_t total1 = (size_t)N * per1;
  const bool admitted = total0 <= BUDGET0 && total1 <= BUDGET1;
  char buf[512];
  snprintf(buf, sizeof(buf),
           "admit N=%d Tctx=%d projected0=%.1fMiB projected1=%.1fMiB "
           "budget0=%.0fMiB budget1=%.0fMiB (FP32 KV 64KiB/token/GPU, "
           "P3 baseline + 1GiB safety, static estimate, not GPU-validated) %s",
           N, Tctx, (double)total0/MiB, (double)total1/MiB,
           (double)BUDGET0/MiB, (double)BUDGET1/MiB,
           admitted ? "ADMIT" : "REJECT");
  detail = buf;
  return admitted;
}

#ifndef HERMES_ADMISSION_CPU_ONLY
void vram_used(size_t& used0_MiB, size_t& used1_MiB) {
  size_t f0 = 0, t0 = 0, f1 = 0, t1 = 0;
  cudaSetDevice(0); cudaMemGetInfo(&f0, &t0);
  cudaSetDevice(1); cudaMemGetInfo(&f1, &t1);
  used0_MiB = (t0 - f0) / 1048576;
  used1_MiB = (t1 - f1) / 1048576;
}
#endif // HERMES_ADMISSION_CPU_ONLY

} // namespace pp2batch
