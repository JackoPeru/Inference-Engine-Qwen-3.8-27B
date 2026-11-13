// CPU-only PP2 substitute for testing the native HTTP server.
// Compile with g++ and serving/server.cpp. NEVER link to the real CUDA PP2.
// No GPU calls, model weights, CUDA allocation or inference happen here.
#include "pp2_fwd.hpp"
#include "pp2_batch.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {
int int_env(const char* name, int fallback = 0) {
  const char* value = std::getenv(name);
  return value ? std::atoi(value) : fallback;
}

std::vector<int> fake_ids() {
  const char* env = std::getenv("FAKE_IDS");
  if (!env) return {100, 101, 102, 103};
  std::istringstream stream(env);
  std::string item;
  std::vector<int> ids;
  while (std::getline(stream, item, ',')) ids.push_back(std::stoi(item));
  return ids.empty() ? std::vector<int>{100} : ids;
}
}

namespace pp2 {
bool upload(Stats& stats, std::string& err) {
  return upload(stats, err, "fake");
}

bool upload(Stats& stats, std::string&, const std::string&) {
  stats.upload_sec = 0.01;
  return true;
}

int eos_token_id() { return 248046; }

bool decode_init(DecodeState& state, std::string& error) {
  return decode_init(state, error, DecodeState::MAXL);
}

bool decode_init(DecodeState& state, std::string& error, int capacity) {
  if (std::getenv("FAKE_INIT_FAIL")) {
    error = "forced mock initialization failure";
    return false;
  }
  if (capacity < 1 || capacity > DecodeState::MAX_CONTEXT) {
    error = "mock invalid capacity";
    return false;
  }
  state.capacity = capacity;
  state.inited = true;
  return true;
}

std::vector<float> decode_step(int, DecodeState& state, long position, std::string& error) {
  if (!state.inited || position < 0 || position >= state.capacity) {
    error = "mock invalid position";
    return {};
  }
  static std::atomic<bool> failed_once{false};
  if (std::getenv("FAKE_DECODE_FAIL_ONCE") && !failed_once.exchange(true)) {
    error = "forced mock decode failure";
    return {};
  }
  static const std::vector<int> ids = fake_ids();
  const int token = ids[std::min(static_cast<std::size_t>(position), ids.size() - 1)];
  if (token < 0 || token >= 248320) {
    error = "mock invalid output token";
    return {};
  }
  std::vector<float> logits(248320, -100.0f);
  logits[static_cast<std::size_t>(token)] = 100.0f;
  const int ms = int_env("FAKE_STEP_MS", 8);
  if (ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  return logits;
}

void decode_free(DecodeState& state) { state.inited = false; }
void free_all() {}
}

namespace pp2batch {
bool admit_check(int requests, int context, std::string& reason) {
  if (requests <= 16 && context >= 1 && context <= 8192) return true;
  reason = "mock capacity rejected";
  return false;
}

void vram_used(size_t& gpu0_mib, size_t& gpu1_mib) {
  gpu0_mib = gpu1_mib = static_cast<size_t>(int_env("FAKE_GPU_USED_MIB"));
}
}
