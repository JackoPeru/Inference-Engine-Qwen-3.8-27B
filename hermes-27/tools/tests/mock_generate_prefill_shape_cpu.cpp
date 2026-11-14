// CPU-only fake PP2 implementation for hermes_generate shape validation.
// NO real CUDA, GPU, GGUF or model upload occurs in this binary.
#include "pp2_fwd.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace pp2 {
namespace {
constexpr std::size_t kVocab = 248320;
}
bool upload(Stats&, std::string&, const std::string&) { return true; }
int eos_token_id() { return 2; }
void free_all() {}
std::vector<float> prefill(const std::vector<int>& ids, Stats&,
                           std::string&, std::vector<float>*) {
  const char* shape = std::getenv("HERMES_TEST_PREFILL_SHAPE");
  const std::string kind = shape ? shape : "";
  if(kind == "short") return std::vector<float>(ids.size()*kVocab-1, 0.f);
  if(kind == "extra") return std::vector<float>(ids.size()*kVocab+1, 0.f);
  if(kind == "empty") return {};
  return std::vector<float>(ids.size()*kVocab, 0.f);
}
bool decode_init(DecodeState&, std::string& err, int) {
  err="CPU mock decoded state intentionally unavailable";
  return false;
}
std::vector<float> prefill_into_decode_state(
    const std::vector<int>&, DecodeState&, Stats&, std::string&,
    std::vector<float>*) {
  std::fputs("UNEXPECTED_STATEFUL_SEED\n", stderr);
  return {};
}
std::vector<float> decode_step(int, DecodeState&, long, std::string&) {
  std::fputs("UNEXPECTED_DECODE_STEP\n", stderr);
  return {};
}
void decode_free(DecodeState&) {}
} // namespace pp2
