// Independent llama.cpp C-API raw-logit exporter. DO NOT link to HERMES.
// No CMake changes; compile against an independently installed llama.cpp.
//
// Compile with g++ -std=c++17 -O2, both llama/include and ggml/include,
// then link against the independent llama.cpp build's -llama shared library.
//
// Run only in a resource-exclusive window. --n-gpu-layers defaults 0 (CPU);
// >0 requires --gpu-exclusive. Outputs stdout JSONL and FP32 raw files matching
// hermes_generate --dump-logits-dir, with exact numeric token ID prompts.
#include "llama.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
constexpr int VOCAB = 248320;
constexpr int MAX_CTX = 8192;

struct Config {
  std::string model;
  std::filesystem::path dump_dir;
  std::vector<std::vector<int>> prompts;
  int predict = 4;
  int threads = 4;
  int gpu_layers = 0;
  bool gpu_exclusive = false;
  bool check_gguf_only = false;
};

bool integer(std::string_view s, int& value) {
  if (s.empty()) return false;
  auto [ptr, error] = std::from_chars(s.data(), s.data() + s.size(), value);
  return error == std::errc{} && ptr == s.data() + s.size();
}

bool parse_prompt(std::string_view s, std::vector<int>& ids) {
  while (!s.empty()) {
    auto comma = s.find(',');
    std::string_view part = s.substr(0, comma);
    int value = 0;
    if (!integer(part, value) || value < 0 || value >= VOCAB) return false;
    ids.push_back(value);
    if (comma == std::string_view::npos) return true;
    s.remove_prefix(comma + 1);
  }
  return false;
}

bool options(int argc, char** argv, Config& c) {
  for (int i = 1; i < argc; ++i) {
    std::string_view opt = argv[i];
    if (opt == "--gpu-exclusive") { c.gpu_exclusive = true; continue; }
    if (opt == "--check-gguf-only") { c.check_gguf_only = true; continue; }
    if (i + 1 >= argc) return false;
    const std::string arg = argv[++i];
    if (opt == "--model") c.model = arg;
    else if (opt == "--out-dir") c.dump_dir = arg;
    else if (opt == "--prompt-ids") {
      std::vector<int> prompt;
      if (!parse_prompt(arg, prompt)) return false;
      c.prompts.push_back(std::move(prompt));
    } else if (opt == "--max-new-tokens") {
      if (!integer(arg, c.predict) || c.predict < 1 || c.predict > 32) return false;
    } else if (opt == "--threads") {
      if (!integer(arg, c.threads) || c.threads < 1 || c.threads > 64) return false;
    } else if (opt == "--n-gpu-layers") {
      if (!integer(arg, c.gpu_layers) || c.gpu_layers < 0 || c.gpu_layers > 128) return false;
    } else return false;
  }
  if (c.model.empty() || !std::filesystem::is_regular_file(c.model) ||
      (c.gpu_layers && !c.gpu_exclusive))
    return false;
  if (c.check_gguf_only)
    return c.gpu_layers == 0;
  if (c.prompts.empty() || !std::filesystem::is_directory(c.dump_dir))
    return false;
  for (const auto& prompt : c.prompts)
    if (prompt.empty() || prompt.size() + (size_t)c.predict - 1 > MAX_CTX) return false;
  // No stale traces may be silently counted as part of the current run.
  for (const auto& item : std::filesystem::directory_iterator(c.dump_dir))
    if (item.is_regular_file() && item.path().extension() == ".f32") return false;
  return true;
}

void print_ids(const std::vector<int>& ids) {
  std::fputc('[', stdout);
  for (size_t i = 0; i < ids.size(); ++i)
    std::printf("%s%d", i ? "," : "", ids[i]);
  std::fputc(']', stdout);
}

bool dump(const Config& c, size_t prompt, int step, const float* data) {
  auto file = c.dump_dir / ("prompt_" + std::to_string(prompt) + "_step_" +
                            std::to_string(step) + ".f32");
  auto partial = file;
  partial += ".partial";
  FILE* out = std::fopen(partial.string().c_str(), "wb");
  if (!out) return false;
  bool ok = std::fwrite(data, sizeof(float), VOCAB, out) == VOCAB;
  ok = std::fclose(out) == 0 && ok;
  if (!ok || std::rename(partial.string().c_str(), file.string().c_str()) != 0) {
    std::remove(partial.string().c_str());
    return false;
  }
  return true;
}
}

int main(int argc, char** argv) {
  Config c;
  if (!options(argc, argv, c)) {
    std::fprintf(stderr,
      "USAGE: reliability_llama_logits --model FILE.gguf --out-dir EMPTY_DIR "
      "--prompt-ids 1,2,3 [--prompt-ids ...] [--max-new-tokens 4] "
      "[--threads 4] [--n-gpu-layers 0] [--gpu-exclusive]\n"
      "CPU-only header compatibility: --model FILE.gguf --check-gguf-only\n"
      "Requires the same GGUF and numeric prompts as HERMES. No model load on invalid args.\n");
    return 2;
  }

  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = c.gpu_layers; // default 0: true CPU-only independent reference
  mp.vocab_only = c.check_gguf_only; // verify metadata/vocab without loading ~20 GB weights
  llama_model* model = llama_model_load_from_file(c.model.c_str(), mp);
  if (!model) {
    std::fprintf(stderr, "BLOCKED: independent llama.cpp failed to load exact GGUF\n");
    llama_backend_free();
    return 2;
  }
  const llama_vocab* vocab = llama_model_get_vocab(model);
  const int count = llama_vocab_n_tokens(vocab);
  const llama_token eos = llama_vocab_eos(vocab);
  if (count != VOCAB || eos < 0) {
    std::fprintf(stderr, "BLOCKED: GGUF vocab mismatch n=%d eos=%d\n", count, eos);
    llama_model_free(model);
    llama_backend_free();
    return 2;
  }
  if (c.check_gguf_only) {
    std::fprintf(stderr, "LLAMA_GGUF_METADATA_OK vocab=%d eos=%d (CPU-only; weights not loaded)\n",
                 count, (int)eos);
    llama_model_free(model);
    llama_backend_free();
    return 0;
  }
  int rc = 0;
  for (size_t index = 0; index < c.prompts.size(); ++index) {
    const auto& ids = c.prompts[index];
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = std::max<int>(128, (int)ids.size() + c.predict - 1);
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    cp.n_threads = c.threads;
    cp.n_threads_batch = c.threads;
    llama_context* context = llama_init_from_model(model, cp);
    if (!context) {
      std::fprintf(stderr, "BLOCKED: llama context creation failed prompt=%zu\n", index);
      rc = 2;
      break;
    }
    auto eval = [&](llama_token id) -> bool {
      llama_batch b = llama_batch_get_one(&id, 1);
      const int result = llama_decode(context, b);
      return result == 0;
    };
    bool success = true;
    for (size_t pos = 0; pos < ids.size(); ++pos) {
      if (!eval((llama_token)ids[pos])) {
        std::fprintf(stderr, "BLOCKED: llama.decode failed prompt=%zu pos=%zu\n", index, pos);
        success = false;
        break;
      }
    }
    std::vector<int> generated;
    const char* reason = "length";
    if (success) {
      for (int step = 0; step < c.predict; ++step) {
        const float* logits = llama_get_logits(context);
        if (!logits) { success = false; break; }
        int next = -1;
        float best = -std::numeric_limits<float>::infinity();
        for (int v = 0; v < VOCAB; ++v) {
          if (!std::isfinite(logits[v])) { success = false; break; }
          if (logits[v] > best) { best = logits[v]; next = v; }
        }
        if (!success || next < 0 || !dump(c, index, step, logits)) {
          success = false;
          break;
        }
        generated.push_back(next);
        if (next == eos) { reason = "eos"; break; }
        if (step + 1 < c.predict && !eval((llama_token)next)) {
          success = false;
          break;
        }
      }
    }
    if (!success) {
      std::fprintf(stderr, "FAIL/BLOCKED: independent logits invalid prompt=%zu\n", index);
      rc = 1;
      llama_free(context);
      break;
    }
    std::printf("{\"prompt_index\":%zu,\"prompt_ids\":", index);
    print_ids(ids);
    std::fputs(",\"generated_token_ids\":", stdout);
    print_ids(generated);
    std::printf(",\"stop_reason\":\"%s\"}\n", reason);
    std::fflush(stdout);
    llama_free(context);
  }
  llama_model_free(model);
  llama_backend_free();
  return rc;
}
