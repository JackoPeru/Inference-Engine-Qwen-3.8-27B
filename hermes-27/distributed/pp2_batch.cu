// HERMES-27 P5 batch concorrente PP2 — implementazione v1.
// Sequenziale: loop pp2::decode_step (corretto, overhead onesto).
// Parallelo: N host threads su pp2::decode_step (overlap reale GPU0/GPU1,
// xfer pinned reso thread-safe da mutex in pp2_fwd.cu).
#include "pp2_batch.hpp"
#include <cstdio>
#include <thread>
#include <mutex>
#include <cuda_runtime.h>

namespace pp2batch {

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

bool admit_check(int N, int Tctx, std::string& detail) {
  // Budget V2 §3-§4: 16KB/tok/GPU + 75MB/req/GPU, cap 16@1K / 8@8K / 2@32K.
  int cap = 0;
  if (Tctx <= 1024) cap = 16;
  else if (Tctx <= 8192) cap = 8;
  else if (Tctx <= 32768) cap = 2;
  else cap = 0;
  double need_MB_per_gpu = N * (Tctx * 16.0 / 1024.0 / 1024.0 + 75.0);
  char buf[512];
  snprintf(buf, sizeof(buf),
           "admit N=%d Tctx=%d need=%.1f MB/GPU cap=%d %s",
           N, Tctx, need_MB_per_gpu, cap, (N <= cap && cap > 0) ? "ADMIT" : "REJECT");
  detail = buf;
  if (cap <= 0) return false;
  return N <= cap;
}

void vram_used(size_t& used0_MiB, size_t& used1_MiB) {
  size_t f0 = 0, t0 = 0, f1 = 0, t1 = 0;
  cudaSetDevice(0); cudaMemGetInfo(&f0, &t0);
  cudaSetDevice(1); cudaMemGetInfo(&f1, &t1);
  used0_MiB = (t0 - f0) / 1048576;
  used1_MiB = (t1 - f1) / 1048576;
}

} // namespace pp2batch
