// CUDA-free PP2 fake backend for testing both server scheduling modes.
// Link with distributed/pp2_batch.cu -DHERMES_ADMISSION_CPU_ONLY.
// Unlike mock_pp2_cpu.cpp, this intentionally does NOT define admit_check:
// the real ContinuousScheduler must use worker-6's actual admission policy.
#include "pp2_fwd.hpp"
#include "pp2_batch.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {
std::mutex trace_mutex;
std::atomic<bool> decode_failed_once{false};

int env_integer(const char* name, int default_value = 0) {
  const char* value = std::getenv(name);
  return value ? std::atoi(value) : default_value;
}

void trace(const std::string& operation, long position = -1) {
  const char* output = std::getenv("FAKE_TRACE_FILE");
  if (!output || !*output) return;
  std::lock_guard<std::mutex> lock(trace_mutex);
  std::ofstream file(output, std::ios::app);
  file << operation;
  if (position >= 0) file << ' ' << position;
  file << '\n';
}

std::vector<int> fake_ids() {
  const char* value = std::getenv("FAKE_IDS");
  if (!value) return {100, 101, 102, 103};
  std::istringstream line(value);
  std::string field;
  std::vector<int> ids;
  while (std::getline(line, field, ',')) ids.push_back(std::stoi(field));
  return ids.empty() ? std::vector<int>{100} : ids;
}
}

namespace pp2 {
bool upload(Stats& stats, std::string& error) { return upload(stats, error, "fake"); }

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
    error = "forced mock init failure";
    return false;
  }
  if (capacity < 1 || capacity > DecodeState::MAX_CONTEXT) {
    error = "bad PP2 mock context";
    return false;
  }
  if (state.inited) {
    error = "mock double-initialization of the same PP2 state";
    return false;
  }
  const char* sliding_flag = std::getenv("HERMES_PP2_SLIDING_KV");
  const bool sliding = sliding_flag && std::string(sliding_flag) == "1";
  if (sliding && capacity != DecodeState::MAX_CONTEXT) {
    error = "mock sliding mode requires physical KV capacity exactly 8192";
    return false;
  }
  state.capacity = capacity;
  state.inited = true;
  state.ever_advanced = false;
  state.sliding_kv_enabled = sliding;
  state.sliding_kv_poisoned = false;
  std::fill(std::begin(state.klen), std::end(state.klen), 0);
  trace("open", capacity);
  return true;
}

std::vector<float> decode_step(int, DecodeState& state, long position,
                               std::string& error) {
  const bool invalid_position = state.sliding_kv_enabled
      ? (position >= SLIDING_KV_ABSOLUTE_POSITION_LIMIT ||
         position != state.klen[0] || state.sliding_kv_poisoned)
      : position >= state.capacity;
  if (!state.inited || position < 0 || invalid_position) {
    error = "mock state not initialized or position out of capacity";
    return {};
  }
  state.ever_advanced = true;
  trace("step", position);
  if (std::getenv("FAKE_DECODE_FAIL_ONCE") && !decode_failed_once.exchange(true)) {
    if (state.sliding_kv_enabled) state.sliding_kv_poisoned = true;
    error = "forced mock decode failure";
    return {};
  }
  if (state.sliding_kv_enabled) state.klen[0] = static_cast<int>(position) + 1;
  static const std::vector<int> ids = fake_ids();
  const int token = ids[std::min(static_cast<std::size_t>(position), ids.size() - 1)];
  if (token < 0 || token >= 248320) {
    error = "invalid mock token";
    return {};
  }
  std::vector<float> logits(248320, -100.0f);
  logits[static_cast<std::size_t>(token)] = 100.0f;
  const int delay = env_integer("FAKE_STEP_MS", 8);
  if (delay > 0) std::this_thread::sleep_for(std::chrono::milliseconds(delay));
  return logits;
}

std::vector<float> prefill_into_decode_state(
    const std::vector<int>& ids, DecodeState& state, Stats& stats,
    std::string& error, std::vector<float>* h_last_out) {
  if (!state.inited || state.ever_advanced || ids.empty() || ids.size() > 64 ||
      ids.size() > static_cast<std::size_t>(state.capacity)) {
    error = "mock seeded prefill requires fresh state and 1..64 tokens";
    if (state.inited) decode_free(state);
    return {};
  }
  trace("prefill", static_cast<long>(ids.size()));
  if (std::getenv("FAKE_PREFILL_FAIL")) {
    error = "forced mock seeded prefill failure";
    decode_free(state);
    return {};
  }
  if (h_last_out) h_last_out->clear();
  const auto scripted = fake_ids();
  constexpr std::size_t vocab = 248320;
  std::vector<float> logits(ids.size() * vocab, -100.0f);
  for (std::size_t position = 0; position < ids.size(); ++position) {
    const int result = scripted[std::min(position, scripted.size() - 1)];
    if (result < 0 || result >= static_cast<int>(vocab)) {
      error = "invalid mock output token";
      decode_free(state);
      return {};
    }
    logits[position * vocab + static_cast<std::size_t>(result)] = 100.0f;
  }
  if (std::getenv("FAKE_PREFILL_NONFINITE")) {
    logits[(ids.size() - 1) * vocab] = std::numeric_limits<float>::quiet_NaN();
  }
  stats.prefill_ms = 0.1;
  state.ever_advanced = true;
  if (state.sliding_kv_enabled) state.klen[0] = static_cast<int>(ids.size());
  return logits;
}

// Real PP2's chunked API seeds one untouched state with 65..8192 tokens
// and returns ONLY the final 248320-logit row, not T*VOC.
std::vector<float> prefill_chunks_into_decode_state(
    const std::vector<int>& ids, DecodeState& state, Stats& stats,
    std::string& error, int chunk_tokens, std::vector<float>* h_last_out) {
  const char* enabled = std::getenv("HERMES_PP2_CHUNKED_PREFILL");
  if (!state.inited || state.ever_advanced || ids.size() < 65 ||
      ids.size() > static_cast<std::size_t>(DecodeState::MAX_CONTEXT) ||
      ids.size() > static_cast<std::size_t>(state.capacity) ||
      chunk_tokens < 1 || chunk_tokens > 64 ||
      !enabled || std::string(enabled) != "1") {
    error = "mock chunked prefill requires explicit opt-in, fresh state and 65..8192";
    if (state.inited) decode_free(state);
    return {};
  }
  trace("chunked", static_cast<long>(ids.size()));
  if (std::getenv("FAKE_CHUNKED_FAIL")) {
    error = "forced mock partial chunk failure";
    decode_free(state);
    return {};
  }
  if (h_last_out) h_last_out->clear();
  constexpr std::size_t VOC = 248320;
  auto ids_script = fake_ids();
  int next = ids_script[std::min(ids.size()-1, ids_script.size()-1)];
  if (next < 0 || next >= static_cast<int>(VOC)) {
    error = "invalid mock chunked token";
    decode_free(state);
    return {};
  }
  std::vector<float> logits(VOC, -100.0f);
  logits[static_cast<std::size_t>(next)] = 100.0f;
  if (std::getenv("FAKE_CHUNKED_NONFINITE"))
    logits[0] = std::numeric_limits<float>::quiet_NaN();
  stats.prefill_ms = 0.5;
  state.ever_advanced = true;
  if (state.sliding_kv_enabled) state.klen[0] = static_cast<int>(ids.size());
  return logits;
}

void decode_free(DecodeState& state) {
  if (!state.inited) return;
  state.inited = false;
  trace("close");
}

void free_all() {}
}

namespace pp2batch {
void vram_used(size_t& gpu0_mib, size_t& gpu1_mib) {
  gpu0_mib = gpu1_mib = static_cast<size_t>(env_integer("FAKE_GPU_USED_MIB"));
}
}
