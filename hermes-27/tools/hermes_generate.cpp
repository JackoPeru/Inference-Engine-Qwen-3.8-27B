#include "pp2_fwd.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <io.h>
#define dup _dup
#define dup2 _dup2
#define close _close
#define fileno _fileno
#define fdopen _fdopen
#else
#include <unistd.h>
#endif

namespace {
constexpr int kVocab = 248320;
constexpr int kMaxContext = pp2::DecodeState::MAX_CONTEXT;
constexpr std::size_t kPrefillMax = 64; // PIN_TMAX in pp2_fwd.cu

struct Options {
  std::string model;
  std::vector<std::vector<int>> prompts;
  int max_new_tokens = 4;
  int eos_token = -1;
  bool verify_prefill = false;
  std::string logits_dump_dir;
};

void usage(FILE* out) {
  std::fprintf(out,
    "Usage: hermes_generate --model FILE.gguf --prompt-ids ID[,ID...] "
    "[--prompt-ids ID[,ID...]] [--max-new-tokens N] [--eos-token ID] "
    "[--verify-prefill] [--dump-logits-dir EXISTING_DIRECTORY]\n"
    "Greedy PP2 token-ID generation. EOS defaults to GGUF metadata.\n"
    "Each prompt gets fresh decode state. Context limit: 8192 tokens.\n"
    "Prompt seed runs decode_step per token (sequential T=1), not chunked prefill.\n"
    "--verify-prefill compares seed logits with full prefill (prompt <= 64).\n"
    "Chunked prefill and MTP are not supported by this path.\n"
    "Optional logits dump creates prompt_<index>_step_<index>.f32 files;\n"
    "step 0 is post-seed logits for the first greedy generated token.\n");
}

bool dump_logits(const std::string& dir, std::size_t prompt_index, int step,
                 const std::vector<float>& logits, std::string& err) {
  if(logits.size() != (std::size_t)kVocab) {
    err = "logits dump: unexpected logit count";
    return false;
  }
  const std::string final_path =
      dir + "/prompt_" + std::to_string(prompt_index) +
      "_step_" + std::to_string(step) + ".f32";
  const std::string partial = final_path + ".partial";
  FILE* out = std::fopen(partial.c_str(), "wb");
  if(!out) {
    err = "logits dump: cannot open " + partial;
    return false;
  }
  const std::size_t written = std::fwrite(logits.data(), sizeof(float), logits.size(), out);
  const bool closed = std::fclose(out) == 0;
  if(written != logits.size() || !closed) {
    std::remove(partial.c_str());
    err = "logits dump: short write or close error at " + partial;
    return false;
  }
  if(std::rename(partial.c_str(), final_path.c_str()) != 0) {
    std::remove(partial.c_str());
    err = "logits dump: failed to finalize " + final_path;
    return false;
  }
  return true;
}

bool parse_int(std::string_view s, int& value) {
  if(s.empty()) return false;
  auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
  return ec == std::errc{} && end == s.data() + s.size();
}

bool parse_ids(std::string_view text, std::vector<int>& ids) {
  if(text.empty()) return false;
  while(!text.empty()) {
    const std::size_t comma = text.find(',');
    std::string_view part = text.substr(0, comma);
    while(!part.empty() && part.front() == ' ') part.remove_prefix(1);
    while(!part.empty() && part.back() == ' ') part.remove_suffix(1);
    int id = -1;
    if(!parse_int(part, id) || id < 0 || id >= kVocab) return false;
    ids.push_back(id);
    if(comma == std::string_view::npos) break;
    text.remove_prefix(comma + 1);
    if(text.empty()) return false;
  }
  return !ids.empty();
}

bool parse_options(int argc, char** argv, Options& opts) {
  for(int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if(arg == "--help" || arg == "-h") { usage(stdout); return false; }
    if(arg == "--verify-prefill") { opts.verify_prefill = true; continue; }
    if(i + 1 >= argc) {
      std::fprintf(stderr, "ERROR: missing value for %.*s\n", (int)arg.size(), arg.data());
      return false;
    }
    const std::string_view value(argv[++i]);
    if(arg == "--model") opts.model.assign(value);
    else if(arg == "--dump-logits-dir") opts.logits_dump_dir.assign(value);
    else if(arg == "--prompt-ids") {
      std::vector<int> ids;
      if(!parse_ids(value, ids)) {
        std::fprintf(stderr, "ERROR: invalid token IDs: %.*s\n", (int)value.size(), value.data());
        return false;
      }
      opts.prompts.push_back(std::move(ids));
    } else if(arg == "--max-new-tokens") {
      if(!parse_int(value, opts.max_new_tokens) || opts.max_new_tokens < 1 || opts.max_new_tokens > kMaxContext) {
        std::fprintf(stderr, "ERROR: max-new-tokens must be 1..%d\n", kMaxContext);
        return false;
      }
    } else if(arg == "--eos-token") {
      if(!parse_int(value, opts.eos_token) || opts.eos_token < 0 || opts.eos_token >= kVocab) {
        std::fprintf(stderr, "ERROR: eos-token must be in 0..%d\n", kVocab - 1);
        return false;
      }
    } else {
      std::fprintf(stderr, "ERROR: unknown option: %.*s\n", (int)arg.size(), arg.data());
      return false;
    }
  }
  if(opts.model.empty() || opts.prompts.empty()) {
    std::fprintf(stderr, "ERROR: --model and at least one --prompt-ids are required\n");
    usage(stderr);
    return false;
  }
  if(!opts.logits_dump_dir.empty()) {
    std::error_code ec;
    if(!std::filesystem::is_directory(opts.logits_dump_dir, ec) || ec) {
      std::fprintf(stderr, "ERROR: --dump-logits-dir requires an existing directory\n");
      return false;
    }
  }
  for(std::size_t i = 0; i < opts.prompts.size(); ++i) {
    const auto size = opts.prompts[i].size();
    if(size > (std::size_t)kMaxContext || size + (std::size_t)opts.max_new_tokens - 1 > (std::size_t)kMaxContext) {
      std::fprintf(stderr, "ERROR: prompt %zu plus max-new-tokens needs more than PP2 context limit %d\n", i, kMaxContext);
      return false;
    }
    if(opts.verify_prefill && size > kPrefillMax) {
      std::fprintf(stderr, "ERROR: prompt %zu exceeds full-prefill limit %zu for --verify-prefill\n", i, kPrefillMax);
      return false;
    }
  }
  return true;
}

// PP2 currently prints diagnostics to stdout. Keep them on stderr; reserve the
// original stdout descriptor for one JSON object per generated prompt.
class SplitOutput {
 public:
  SplitOutput() {
    std::fflush(stdout);
    const int saved = dup(fileno(stdout));
    if(saved < 0) return;
    if(dup2(fileno(stderr), fileno(stdout)) < 0) { close(saved); return; }
    json_ = fdopen(saved, "w");
    if(!json_) close(saved);
  }
  ~SplitOutput() { if(json_) std::fclose(json_); }
  FILE* json() const { return json_; }
 private:
  FILE* json_ = nullptr;
};

bool argmax(const std::vector<float>& logits, int& token, std::string& err) {
  if(logits.size() != (std::size_t)kVocab) { err = "unexpected logits size"; return false; }
  float best = -std::numeric_limits<float>::infinity();
  token = -1;
  for(int i = 0; i < kVocab; ++i) {
    const float value = logits[(std::size_t)i];
    if(!std::isfinite(value)) { err = "non-finite logit at token " + std::to_string(i); return false; }
    if(value > best) { best = value; token = i; }
  }
  return token >= 0;
}

void print_ids(FILE* out, const std::vector<int>& ids) {
  std::fputc('[', out);
  for(std::size_t i = 0; i < ids.size(); ++i)
    std::fprintf(out, "%s%d", i ? "," : "", ids[i]);
  std::fputc(']', out);
}
} // namespace

int main(int argc, char** argv) {
  Options opts;
  if(!parse_options(argc, argv, opts)) {
    for(int i = 1; i < argc; ++i)
      if(std::string_view(argv[i]) == "--help" || std::string_view(argv[i]) == "-h") return 0;
    return 2;
  }

  SplitOutput output;
  FILE* json = output.json();
  if(!json) { std::fprintf(stderr, "ERROR: cannot separate JSON stdout from PP2 diagnostics\n"); return 1; }

  pp2::Stats stats;
  std::string err;
  if(!pp2::upload(stats, err, opts.model)) {
    std::fprintf(stderr, "ERROR: PP2 upload failed: %s\n", err.c_str());
    pp2::free_all();
    return 1;
  }
  const int eos = opts.eos_token >= 0 ? opts.eos_token : pp2::eos_token_id();
  if(eos < 0) {
    std::fprintf(stderr, "ERROR: GGUF lacks valid tokenizer.ggml.eos_token_id; pass --eos-token\n");
    pp2::free_all();
    return 1;
  }
  std::fprintf(stderr, "MODEL %s eos=%d context_limit=%d MTP=off\n", opts.model.c_str(), eos, kMaxContext);

  for(std::size_t pi = 0; pi < opts.prompts.size(); ++pi) {
    const auto& prompt = opts.prompts[pi];
    std::vector<float> reference;
    if(opts.verify_prefill) {
      auto full = pp2::prefill(prompt, stats, err);
      if(full.empty()) {
        std::fprintf(stderr, "ERROR: prompt %zu full prefill failed: %s\n", pi, err.c_str());
        pp2::free_all();
        return 1;
      }
      const std::size_t offset = (prompt.size() - 1) * (std::size_t)kVocab;
      reference.assign(full.begin() + offset, full.begin() + offset + kVocab);
    }

    pp2::DecodeState state{};
    const int context_capacity=(int)(prompt.size()+(std::size_t)opts.max_new_tokens-1);
    if(!pp2::decode_init(state, err, context_capacity)) {
      std::fprintf(stderr, "ERROR: prompt %zu decode init failed: %s\n", pi, err.c_str());
      pp2::free_all();
      return 1;
    }
    std::fprintf(stderr, "SEED prompt=%zu mode=token-by-token capacity=%d\n", pi, context_capacity);
    std::vector<float> logits;
    const auto seed_started = std::chrono::steady_clock::now();
    for(std::size_t pos = 0; pos < prompt.size(); ++pos) {
      logits = pp2::decode_step(prompt[pos], state, (long)pos, err);
      if(logits.empty()) {
        std::fprintf(stderr, "ERROR: prompt %zu token-by-token seed failed at pos %zu: %s\n", pi, pos, err.c_str());
        pp2::decode_free(state);
        pp2::free_all();
        return 1;
      }
    }
    const auto seed_ended = std::chrono::steady_clock::now();

    if(opts.verify_prefill) {
      double max_abs = 0.0, mean_abs = 0.0;
      for(int i = 0; i < kVocab; ++i) {
        const double delta = std::abs((double)logits[(std::size_t)i] - reference[(std::size_t)i]);
        max_abs = std::max(max_abs, delta);
        mean_abs += delta;
      }
      mean_abs /= kVocab;
      int decode_top = -1, prefill_top = -1;
      if(!argmax(logits, decode_top, err) || !argmax(reference, prefill_top, err)) {
        std::fprintf(stderr, "ERROR: prompt %zu prefill compare failed: %s\n", pi, err.c_str());
        pp2::decode_free(state);
        pp2::free_all();
        return 1;
      }
      std::fprintf(stderr, "VERIFY prompt=%zu full_prefill max_abs=%.6g mean_abs=%.6g argmax=%d/%d %s\n",
        pi, max_abs, mean_abs, decode_top, prefill_top,
        max_abs <= 2e-3 && decode_top == prefill_top ? "PASS" : "FAIL");
      if(max_abs > 2e-3 || decode_top != prefill_top) {
        pp2::decode_free(state);
        pp2::free_all();
        return 2;
      }
    }

    std::vector<int> generated;
    generated.reserve((std::size_t)opts.max_new_tokens);
    std::string stop_reason = "length";
    const auto generation_started = std::chrono::steady_clock::now();
    for(int step = 0; step < opts.max_new_tokens; ++step) {
      int next = -1;
      if(!argmax(logits, next, err)) {
        std::fprintf(stderr, "ERROR: prompt %zu greedy selection failed: %s\n", pi, err.c_str());
        pp2::decode_free(state);
        pp2::free_all();
        return 1;
      }
      if(!opts.logits_dump_dir.empty() &&
         !dump_logits(opts.logits_dump_dir, pi, step, logits, err)) {
        std::fprintf(stderr, "ERROR: prompt %zu logit dump step %d: %s\n", pi, step, err.c_str());
        pp2::decode_free(state);
        pp2::free_all();
        return 1;
      }
      generated.push_back(next);
      if(next == eos) { stop_reason = "eos"; break; }
      if(step + 1 == opts.max_new_tokens) break;
      const long pos = (long)prompt.size() + step;
      logits = pp2::decode_step(next, state, pos, err);
      if(logits.empty()) {
        std::fprintf(stderr, "ERROR: prompt %zu decode failed at pos %ld: %s\n", pi, pos, err.c_str());
        pp2::decode_free(state);
        pp2::free_all();
        return 1;
      }
    }
    const auto generation_ended = std::chrono::steady_clock::now();
    const double seed_ms = std::chrono::duration<double,std::milli>(seed_ended-seed_started).count();
    const double generation_ms = std::chrono::duration<double,std::milli>(generation_ended-generation_started).count();
    const std::size_t decode_steps = generated.empty() ? 0 : generated.size()-1;
    std::fprintf(stderr,
      "TIMING prompt=%zu seed_tokens=%zu seed_ms=%.3f decode_steps=%zu generation_ms=%.3f decode_steps_per_s=%.5f\n",
      pi,prompt.size(),seed_ms,decode_steps,generation_ms,
      generation_ms>0.0 ? (1000.0*decode_steps/generation_ms) : 0.0);
    pp2::decode_free(state);

    std::fprintf(json, "{\"prompt_index\":%zu,\"prompt_ids\":", pi);
    print_ids(json, prompt);
    std::fputs(",\"generated_token_ids\":", json);
    print_ids(json, generated);
    std::fprintf(json, ",\"stop_reason\":\"%s\"}\n", stop_reason.c_str());
    std::fflush(json);
    std::fprintf(stderr, "GEN prompt=%zu tokens=%zu stop=%s ids=", pi, generated.size(), stop_reason.c_str());
    for(std::size_t i = 0; i < generated.size(); ++i)
      std::fprintf(stderr, "%s%d", i ? "," : "", generated[i]);
    std::fputc('\n', stderr);
  }

  pp2::free_all();
  return 0;
}
