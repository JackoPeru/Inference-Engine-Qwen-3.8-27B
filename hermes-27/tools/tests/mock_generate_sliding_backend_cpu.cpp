// PP2 CLI contract witness: no GGUF, CUDA, physical KV cache or inference.
// Checks the exact capacity and absolute sequence passed to decode_step.
#include "pp2_fwd.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace pp2 {
namespace {
constexpr std::size_t kVocab = 248320;
long next_position = 0;

bool env_is(const char* name, std::string_view expected) {
  const char* value = std::getenv(name);
  return value && std::string_view(value) == expected;
}
}

bool upload(Stats&, std::string&, const std::string&) {
  std::fputs("MOCK_UPLOAD\n", stderr);
  return true;
}

int eos_token_id() { return 1; }

void free_all() { std::fputs("MOCK_FREE_ALL\n", stderr); }

std::vector<float> prefill(const std::vector<int>&, Stats&, std::string&,
                           std::vector<float>*) {
  std::fputs("MOCK_UNEXPECTED_FULL_PREFILL\n", stderr);
  return {};
}

std::vector<float> prefill_into_decode_state(
    const std::vector<int>&, DecodeState&, Stats&, std::string&,
    std::vector<float>*) {
  std::fputs("MOCK_UNEXPECTED_BATCHED_PREFILL\n", stderr);
  return {};
}

bool decode_init(DecodeState& state, std::string& error, int capacity) {
  if(state.inited) {
    error = "mock second decode_init on active state";
    return false;
  }
  const bool sliding = env_is("HERMES_PP2_SLIDING_KV", "1");
  if(sliding && capacity != DecodeState::MAX_CONTEXT) {
    error = "mock sliding capacity must be exactly 8192";
    return false;
  }
  state.inited = true;
  state.capacity = capacity;
  state.sliding_kv_enabled = sliding;
  next_position = 0;
  std::fprintf(stderr, "MOCK_INIT_CAPACITY=%d SLIDING=%d\n",
               capacity, sliding ? 1 : 0);
  return true;
}

std::vector<float> decode_step(int token, DecodeState& state,
                               long position, std::string& error) {
  if(!state.inited || token < 0 || token >= (int)kVocab ||
     position != next_position) {
    error = "mock nonmonotonic, duplicated or uninitialized decode";
    std::fprintf(stderr, "MOCK_BAD_POSITION=%ld EXPECTED=%ld\n",
                 position, next_position);
    return {};
  }
  if(position >= state.capacity && !state.sliding_kv_enabled) {
    error = "mock default context overflow";
    return {};
  }
  if(position >= SLIDING_KV_ABSOLUTE_POSITION_LIMIT) {
    error = "mock absolute position overflow";
    return {};
  }
  const char* fail_step = std::getenv("HERMES_TEST_FAIL_STEP");
  if(fail_step && std::string_view(fail_step) == std::to_string(position)) {
    error = "injected mock ring decode failure";
    std::fputs("MOCK_INJECTED_STEP_FAILURE\n", stderr);
    return {};
  }
  if(position == 8191) std::fputs("MOCK_SEED_COMPLETE=8192\n", stderr);
  if(position == 8192) std::fputs("MOCK_POS=8192\n", stderr);
  ++next_position;
  std::vector<float> result(kVocab, -1.0f);
  result[42] = 1.0f; // Non-EOS token for both requested generated steps.
  if(env_is("HERMES_TEST_BAD_LOGITS", "1") && position == 8192)
    result[42] = std::numeric_limits<float>::quiet_NaN();
  return result;
}

void decode_free(DecodeState& state) {
  if(!state.inited) {
    std::fputs("MOCK_DOUBLE_FREE\n", stderr);
    return;
  }
  std::fprintf(stderr, "MOCK_DECODE_FREE steps=%ld\n", next_position);
  state.inited = false;
}
} // namespace pp2
