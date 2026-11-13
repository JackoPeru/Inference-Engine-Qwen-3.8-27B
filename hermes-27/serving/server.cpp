#include "pp2_fwd.hpp"
#include "pp2_batch.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <ctime>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef __linux__
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#else
#error "Hermes serving tokenizer bridge currently requires Linux"
#endif

namespace {
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr int kVocab = 248320; // Fixed vocabulary of the Qwen3.8-27B target model.
constexpr std::size_t kMaxBody = 1024 * 1024;

struct Config {
  std::string model = "/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf";
  std::string tokenizer = "/opt/hermes/models/Qwen3.8-27B-Uncensored-EXL3-3.07bpw/tokenizer.json";
  std::string gguf_dump = "./build/gguf_dump";
  std::string python = "/opt/hermes/runtimes/exllamav3-venv/bin/python";
  std::string model_id = "hermes-27";
  std::string host = "127.0.0.1";
  std::string api_key;
  int port = 0;
  int active = 4;
  int queued = 32;
  int context = pp2::DecodeState::MAXL;
  int queue_timeout_sec = 60;
  int gpu0_start_limit_mib = 2000;
  int gpu1_start_limit_mib = 2500;
};

void usage(FILE* out) {
  std::fprintf(out,
    "Usage: hermes_server --model FILE.gguf [--host 127.0.0.1] [--port 0] "
    "[--active 1..16] [--queue 1..32] [--context 1..8192] [--api-key KEY]\n"
    "Optional: --tokenizer FILE --gguf-dump FILE --python FILE --model-id ID\n"
    "Greedy text-only PP2; context defaults to 128. Unsupported sampling/tools are rejected.\n");
}

bool parse_int(const char* text, int& out) {
  if (!text || !*text) return false;
  char* end = nullptr;
  errno = 0;
  long value = std::strtol(text, &end, 10);
  if (errno || !end || *end || value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) return false;
  out = static_cast<int>(value);
  return true;
}

bool parse_args(int argc, char** argv, Config& cfg, bool& show_help) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") { usage(stdout); show_help = true; return false; }
    if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", arg.c_str()); return false; }
    const char* value = argv[++i];
    if (arg == "--model") cfg.model = value;
    else if (arg == "--tokenizer") cfg.tokenizer = value;
    else if (arg == "--gguf-dump") cfg.gguf_dump = value;
    else if (arg == "--python") cfg.python = value;
    else if (arg == "--model-id") cfg.model_id = value;
    else if (arg == "--host") cfg.host = value;
    else if (arg == "--api-key") cfg.api_key = value;
    else if (arg == "--port") { if (!parse_int(value, cfg.port) || cfg.port < 0 || cfg.port > 65535) return false; }
    else if (arg == "--active") { if (!parse_int(value, cfg.active) || cfg.active < 1 || cfg.active > 16) return false; }
    else if (arg == "--queue") { if (!parse_int(value, cfg.queued) || cfg.queued < 1 || cfg.queued > 32) return false; }
    else if (arg == "--context") { if (!parse_int(value, cfg.context) || cfg.context < 1 || cfg.context > pp2::DecodeState::MAX_CONTEXT) return false; }
    else if (arg == "--queue-timeout") { if (!parse_int(value, cfg.queue_timeout_sec) || cfg.queue_timeout_sec < 1 || cfg.queue_timeout_sec > 600) return false; }
    else if (arg == "--gpu0-used-limit-mib") { if (!parse_int(value, cfg.gpu0_start_limit_mib) || cfg.gpu0_start_limit_mib < 0) return false; }
    else if (arg == "--gpu1-used-limit-mib") { if (!parse_int(value, cfg.gpu1_start_limit_mib) || cfg.gpu1_start_limit_mib < 0) return false; }
    else { std::fprintf(stderr, "unknown option: %s\n", arg.c_str()); return false; }
  }
  return true;
}

class TokenizerBridge {
 public:
  TokenizerBridge(const Config& cfg, const std::string& helper) {
    int to_child[2], from_child[2];
    if (pipe(to_child) || pipe(from_child)) throw std::runtime_error("tokenizer bridge pipe failed");
    pid_ = fork();
    if (pid_ < 0) throw std::runtime_error("tokenizer bridge fork failed");
    if (pid_ == 0) {
      close(to_child[1]); close(from_child[0]);
      if (dup2(to_child[0], STDIN_FILENO) < 0 || dup2(from_child[1], STDOUT_FILENO) < 0) _exit(126);
      close(to_child[0]); close(from_child[1]);
      char* args[] = {
        const_cast<char*>(cfg.python.c_str()), const_cast<char*>(helper.c_str()),
        const_cast<char*>("--tokenizer"), const_cast<char*>(cfg.tokenizer.c_str()),
        const_cast<char*>("--model"), const_cast<char*>(cfg.model.c_str()),
        const_cast<char*>("--gguf-dump"), const_cast<char*>(cfg.gguf_dump.c_str()), nullptr
      };
      execv(cfg.python.c_str(), args);
      _exit(127);
    }
    close(to_child[0]); close(from_child[1]);
    input_ = to_child[1]; output_ = from_child[0];
    json ready = json::parse(read_line());
    if (!ready.value("ready", false)) throw std::runtime_error("tokenizer startup: " + ready.value("error", "no ready response"));
    incremental_ = ready.value("incremental", false);
  }

  ~TokenizerBridge() {
    if (input_ >= 0) close(input_);
    if (output_ >= 0) close(output_);
    if (pid_ > 0) { int status = 0; while (waitpid(pid_, &status, 0) < 0 && errno == EINTR) {} }
  }

  std::vector<int> encode(const json& messages, bool thinking) {
    json response = call({{"op", "encode"}, {"messages", messages}, {"thinking", thinking}});
    std::vector<int> ids;
    for (const auto& id : response.at("ids")) ids.push_back(id.get<int>());
    return ids;
  }

  std::string decode(const std::vector<int>& ids) {
    json response = call({{"op", "decode"}, {"ids", ids}});
    return response.at("text").get<std::string>();
  }

  bool incremental_supported() const { return incremental_; }

  void stream_start(const std::string& id) {
    call({{"op", "stream_start"}, {"id", id}});
  }

  std::string stream_push(const std::string& id, int token) {
    return call({{"op", "stream_push"}, {"id", id}, {"token", token}})
      .at("delta").get<std::string>();
  }

  std::string stream_finish(const std::string& id) {
    return call({{"op", "stream_finish"}, {"id", id}})
      .at("tail").get<std::string>();
  }

  void stream_abort(const std::string& id) {
    call({{"op", "stream_abort"}, {"id", id}});
  }

 private:
  static bool write_all(int fd, const char* data, std::size_t size) {
    while (size) {
      ssize_t n = write(fd, data, size);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) return false;
      data += n; size -= static_cast<std::size_t>(n);
    }
    return true;
  }

  std::string read_line() {
    for (;;) {
      const std::size_t newline = pending_.find('\n');
      if (newline != std::string::npos) {
        std::string line = pending_.substr(0, newline);
        pending_.erase(0, newline + 1);
        return line;
      }
      char buffer[4096];
      ssize_t n = read(output_, buffer, sizeof(buffer));
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) throw std::runtime_error("tokenizer bridge exited");
      pending_.append(buffer, static_cast<std::size_t>(n));
      if (pending_.size() > 4 * kMaxBody) throw std::runtime_error("tokenizer response too large");
    }
  }

  json call(const json& request) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string line = request.dump() + "\n";
    if (!write_all(input_, line.data(), line.size())) throw std::runtime_error("tokenizer bridge write failed");
    json response = json::parse(read_line());
    if (response.contains("error")) throw std::runtime_error(response.value("error", "tokenizer error"));
    return response;
  }

  pid_t pid_ = -1;
  int input_ = -1, output_ = -1;
  std::string pending_;
  bool incremental_ = false;
  std::mutex mutex_;
};

struct Metrics {
  std::atomic<std::uint64_t> requests{0}, accepted{0}, completed{0}, rejected{0};
  std::atomic<std::uint64_t> cancelled{0}, errors{0}, prompt_tokens{0}, completion_tokens{0};
};

struct Job {
  std::string id;
  std::string model;
  std::vector<int> prompt;
  std::vector<int> output;
  std::string text;
  std::size_t sent_bytes = 0;
  std::string error;
  json response;
  bool stream = false;
  int max_new_tokens = 32;
  int http_status = 200;
  std::size_t prompt_pos = 0;
  std::time_t created = 0;
  Clock::time_point queued_at;
  std::atomic<bool> cancel{false}, done{false};
  pp2::DecodeState state{};
  bool state_ready = false;
  bool admitted = false; // Guarded by mutex: streaming HTTP headers wait for GPU admission.
  bool stream_started = false;
  std::mutex mutex;
  std::condition_variable cv;
  std::deque<std::string> stream_chunks;
};

class Scheduler {
 public:
  Scheduler(TokenizerBridge& tokenizer, int eos, const Config& cfg)
      : tokenizer_(tokenizer), eos_(eos), active_limit_(cfg.active), queue_limit_(cfg.queued),
        queue_timeout_(cfg.queue_timeout_sec), worker_(&Scheduler::run, this) {}

  ~Scheduler() { stop(); }

  bool enqueue(const std::shared_ptr<Job>& job) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ || static_cast<int>(queue_.size()) >= queue_limit_) return false;
    job->queued_at = Clock::now();
    jobs_[job->id] = job;
    if (job->stream) push_stream_locked(*job, chunk(*job, {{"role", "assistant"}}, nullptr));
    queue_.push_back(job);
    metrics_.accepted.fetch_add(1);
    metrics_.prompt_tokens.fetch_add(job->prompt.size());
    cv_.notify_all();
    return true;
  }

  bool cancel(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(id);
    if (it == jobs_.end()) return false;
    auto job = it->second.lock();
    if (!job || job->done.load()) return false;
    bool expected = false;
    if (job->cancel.compare_exchange_strong(expected, true)) metrics_.cancelled.fetch_add(1);
    cv_.notify_all();
    return true;
  }

  void retire(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    jobs_.erase(id);
  }

  void rejected() { metrics_.rejected.fetch_add(1); }
  void request() { metrics_.requests.fetch_add(1); }

  std::string metrics_text(const std::string& model) {
    std::size_t queued = 0, active = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queued = queue_.size(); active = active_.size();
    }
    size_t used0 = 0, used1 = 0;
    pp2batch::vram_used(used0, used1);
    const auto& m = metrics_;
    std::string out;
    out += "# HELP hermes_server_ready Server readiness.\n# TYPE hermes_server_ready gauge\nhermes_server_ready 1\n";
    out += "# TYPE hermes_requests_total counter\nhermes_requests_total " + std::to_string(m.requests.load()) + "\n";
    out += "# TYPE hermes_requests_accepted_total counter\nhermes_requests_accepted_total " + std::to_string(m.accepted.load()) + "\n";
    out += "# TYPE hermes_requests_completed_total counter\nhermes_requests_completed_total " + std::to_string(m.completed.load()) + "\n";
    out += "# TYPE hermes_requests_rejected_total counter\nhermes_requests_rejected_total " + std::to_string(m.rejected.load()) + "\n";
    out += "# TYPE hermes_requests_cancelled_total counter\nhermes_requests_cancelled_total " + std::to_string(m.cancelled.load()) + "\n";
    out += "# TYPE hermes_inference_errors_total counter\nhermes_inference_errors_total " + std::to_string(m.errors.load()) + "\n";
    out += "# TYPE hermes_requests_active gauge\nhermes_requests_active " + std::to_string(active) + "\n";
    out += "# TYPE hermes_requests_queued gauge\nhermes_requests_queued " + std::to_string(queued) + "\n";
    out += "# TYPE hermes_prompt_tokens_total counter\nhermes_prompt_tokens_total " + std::to_string(m.prompt_tokens.load()) + "\n";
    out += "# TYPE hermes_completion_tokens_total counter\nhermes_completion_tokens_total " + std::to_string(m.completion_tokens.load()) + "\n";
    out += "# TYPE hermes_vram_used_mib gauge\nhermes_vram_used_mib{gpu=\"0\"} " + std::to_string(used0) + "\n";
    out += "hermes_vram_used_mib{gpu=\"1\"} " + std::to_string(used1) + "\n";
    out += "# TYPE hermes_model_info gauge\nhermes_model_info{model=\"" + model + "\"} 1\n";
    return out;
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_) return;
      stopping_ = true;
      for (auto& job : queue_) job->cancel.store(true);
      for (auto& job : active_) job->cancel.store(true);
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

 private:
  static json chunk(const Job& job, json delta, const char* finish) {
    json choice = {{"index", 0}, {"delta", std::move(delta)}, {"finish_reason", nullptr}};
    if (finish) choice["finish_reason"] = finish;
    return {{"id", job.id}, {"object", "chat.completion.chunk"}, {"created", job.created},
            {"model", job.model}, {"choices", json::array({choice})}};
  }

  static void push_stream_locked(Job& job, const json& payload) {
    job.stream_chunks.push_back("data: " + payload.dump() + "\n\n");
    job.cv.notify_all();
  }

  void stream_text(const std::shared_ptr<Job>& job) {
    if (!job->stream || job->cancel.load()) return;
    // The ByteLevel stream bridge releases only complete, stable UTF-8
    // codepoints; unsupported decoders emit the entire text at completion.
    std::lock_guard<std::mutex> lock(job->mutex);
    if (job->text.size() <= job->sent_bytes) return;
    // All updates append stable decoded characters, so a byte offset is
    // sufficient. Copying/comparing the entire sent prefix was O(n^2).
    const std::string delta = job->text.substr(job->sent_bytes);
    job->sent_bytes = job->text.size();
    if (!delta.empty()) push_stream_locked(*job, chunk(*job, {{"content", delta}}, nullptr));
  }

  void complete(const std::shared_ptr<Job>& job, int status, std::string error,
                const std::string& finish_reason) {
    if (job->state_ready) { pp2::decode_free(job->state); job->state_ready = false; }
    const bool was_cancelled = job->cancel.load();
    if (was_cancelled && status == 200) error.clear();
    const bool was_incremental = job->stream_started;
    if (was_incremental) {
      try {
        if (status == 200) job->text += tokenizer_.stream_finish(job->id);
        else tokenizer_.stream_abort(job->id);
      } catch (const std::exception& e) {
        status = 500;
        error = std::string("incremental decoder finalization failed: ") + e.what();
      }
      job->stream_started = false;
    }
    if (status == 200) {
      try {
        // One reference decode verifies exact final parity (also for invalid
        // Unicode / byte fallback) and serves non-stream clients directly.
        std::string reference = tokenizer_.decode(job->output);
        if (was_incremental && job->text != reference)
          throw std::runtime_error("incremental decoder output mismatch");
        job->text = std::move(reference);
      }
      catch (const std::exception& e) { status = 500; error = std::string("detokenization failed: ") + e.what(); }
    }
    if (status == 200) stream_text(job);

    if (status == 200) {
      const std::string reason = was_cancelled ? "cancelled" : finish_reason;
      job->response = {
        {"id", job->id}, {"object", "chat.completion"}, {"created", job->created}, {"model", job->model},
        {"choices", json::array({{{"index", 0}, {"message", {{"role", "assistant"}, {"content", job->text}}}, {"finish_reason", reason}}})},
        {"usage", {{"prompt_tokens", job->prompt.size()}, {"completion_tokens", job->output.size()},
                    {"total_tokens", job->prompt.size() + job->output.size()}}}
      };
    }

    {
      std::lock_guard<std::mutex> lock(job->mutex);
      job->http_status = status;
      job->error = std::move(error);
      if (job->stream) {
        if (status == 200) push_stream_locked(*job, chunk(*job, json::object(), was_cancelled ? "cancelled" : finish_reason.c_str()));
        else push_stream_locked(*job, {{"error", {{"message", job->error}, {"type", "server_error"}}}});
        job->stream_chunks.push_back("data: [DONE]\n\n");
      }
      job->done.store(true);
      job->cv.notify_all();
    }

    if (status == 200) {
      metrics_.completed.fetch_add(1);
      metrics_.completion_tokens.fetch_add(job->output.size());
    } else {
      metrics_.errors.fetch_add(1);
    }
    std::fprintf(stderr, "request id=%s status=%d prompt_tokens=%zu completion_tokens=%zu finish=%s ids=",
                 job->id.c_str(), status, job->prompt.size(), job->output.size(), finish_reason.c_str());
    for (std::size_t i = 0; i < job->output.size(); ++i) std::fprintf(stderr, "%s%d", i ? "," : "", job->output[i]);
    if (!job->error.empty()) std::fprintf(stderr, " error=%s", job->error.c_str());
    std::fputc('\n', stderr);
    std::fflush(stderr);
  }

  bool admit(const std::shared_ptr<Job>& job, std::string& why) {
    std::size_t active_count = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      active_count = active_.size();
    }
    const int context = static_cast<int>(job->prompt.size()) + job->max_new_tokens - 1;
    if (!pp2batch::admit_check(static_cast<int>(active_count + 1), context, why)) return false;
    std::size_t used0 = 0, used1 = 0;
    pp2batch::vram_used(used0, used1);
    // decode_init uses FP32 KV: 8 full-attention layers per GPU x K/V x
    // 1024 floats = 64 KiB/token/GPU. Allow 96 MiB for GDN states,
    // recurrent convolution and workspaces; decode_init rechecks exactly.
    const std::size_t need = 96 + (static_cast<std::size_t>(context) * 64 + 1023) / 1024;
    constexpr std::size_t limit_mib = 15 * 1024;
    if (used0 + need > limit_mib || used1 + need > limit_mib) {
      why += " vram used=" + std::to_string(used0) + "/" + std::to_string(used1) +
             "MiB need=" + std::to_string(need) + "MiB/GPU limit=15360MiB";
      return false;
    }
    return true;
  }

  void run() {
    for (;;) {
      std::vector<std::shared_ptr<Job>> expired, start;
      bool should_stop = false;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        if (active_.empty() && queue_.empty() && !stopping_) cv_.wait_for(lock, std::chrono::milliseconds(250));
        should_stop = stopping_;
        const auto now = Clock::now();
        // Expire every queued job, including when the active slots are all full.
        // Otherwise queue_timeout_ would only advance when a GPU slot becomes free.
        for (auto it = queue_.begin(); it != queue_.end();) {
          const auto& job = *it;
          if (job->cancel.load() || should_stop ||
              now - job->queued_at > std::chrono::seconds(queue_timeout_)) {
            expired.push_back(job);
            it = queue_.erase(it);
          } else {
            ++it;
          }
        }
        while (!queue_.empty() && static_cast<int>(active_.size() + start.size()) < active_limit_) {
          start.push_back(queue_.front());
          queue_.pop_front();
        }
        if (should_stop) {
          for (auto& job : active_) { job->cancel.store(true); expired.push_back(job); }
          active_.clear();
        }
      }
      for (auto& job : expired) {
        if (job->cancel.load()) complete(job, 200, {}, "cancelled");
        else { metrics_.rejected.fetch_add(1); complete(job, 429, "request queue timeout; retry later", "length"); }
      }
      if (should_stop) return;

      for (auto& job : start) {
        if (job->cancel.load()) { complete(job, 200, {}, "cancelled"); continue; }
        std::string why;
        if (!admit(job, why)) {
          metrics_.rejected.fetch_add(1);
          complete(job, 429, "PP2 request budget rejected: " + why, "length");
          continue;
        }
        std::string err;
        // The final generated token is emitted from the previous logits and
        // therefore needs no additional decode state position.
        const int capacity = static_cast<int>(job->prompt.size()) + job->max_new_tokens - 1;
        if (!pp2::decode_init(job->state, err, capacity)) {
          metrics_.rejected.fetch_add(1);
          complete(job, 503, "PP2 state allocation failed: " + err, "length");
          continue;
        }
        job->state_ready = true;
        if (job->cancel.load()) { complete(job, 200, {}, "cancelled"); continue; }
        if (job->stream && tokenizer_.incremental_supported()) {
          try {
            tokenizer_.stream_start(job->id);
            job->stream_started = true;
          } catch (const std::exception& e) {
            complete(job, 500, std::string("incremental decoder startup failed: ") + e.what(), "length");
            continue;
          }
        }
        {
          std::lock_guard<std::mutex> lock(mutex_);
          active_.push_back(job);
        }
        {
          std::lock_guard<std::mutex> lock(job->mutex);
          job->admitted = true;
          job->cv.notify_all();
        }
      }

      std::vector<std::shared_ptr<Job>> batch;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        batch = active_;
      }
      if (batch.empty()) continue;

      std::vector<int> inputs(batch.size());
      std::vector<long> positions(batch.size());
      std::vector<std::vector<float>> logits(batch.size());
      std::vector<std::string> errors(batch.size());
      std::vector<std::thread> threads;
      threads.reserve(batch.size());
      for (std::size_t i = 0; i < batch.size(); ++i) {
        auto& job = *batch[i];
        if (job.cancel.load()) continue;
        if (job.prompt_pos < job.prompt.size()) {
          inputs[i] = job.prompt[job.prompt_pos];
          positions[i] = static_cast<long>(job.prompt_pos);
        } else {
          inputs[i] = job.output.back();
          positions[i] = static_cast<long>(job.prompt.size() + job.output.size() - 1);
        }
        threads.emplace_back([&, i] {
          logits[i] = pp2::decode_step(inputs[i], batch[i]->state, positions[i], errors[i]);
        });
      }
      for (auto& thread : threads) thread.join();

      std::vector<std::shared_ptr<Job>> finished;
      for (std::size_t i = 0; i < batch.size(); ++i) {
        auto& job = batch[i];
        if (job->cancel.load()) { complete(job, 200, {}, "cancelled"); finished.push_back(job); continue; }
        if (logits[i].empty()) { complete(job, 500, "PP2 decode failed: " + errors[i], "length"); finished.push_back(job); continue; }
        if (job->prompt_pos < job->prompt.size()) {
          ++job->prompt_pos;
          if (job->prompt_pos < job->prompt.size()) continue;
        }
        int token = -1;
        float best = -std::numeric_limits<float>::infinity();
        for (std::size_t t = 0; t < logits[i].size(); ++t) {
          const float value = logits[i][t];
          if (!std::isfinite(value)) { errors[i] = "non-finite logit at token " + std::to_string(t); break; }
          if (value > best) { best = value; token = static_cast<int>(t); }
        }
        if (token < 0) { complete(job, 500, "greedy selection failed: " + errors[i], "length"); finished.push_back(job); continue; }
        job->output.push_back(token);
        if (job->stream_started) {
          try { job->text += tokenizer_.stream_push(job->id, token); }
          catch (const std::exception& e) { complete(job, 500, std::string("incremental detokenization failed: ") + e.what(), "length"); finished.push_back(job); continue; }
          stream_text(job);
        }
        if (token == eos_) { complete(job, 200, {}, "stop"); finished.push_back(job); }
        else if (static_cast<int>(job->output.size()) >= job->max_new_tokens) {
          complete(job, 200, {}, "length"); finished.push_back(job);
        }
      }
      if (!finished.empty()) {
        std::lock_guard<std::mutex> lock(mutex_);
        active_.erase(std::remove_if(active_.begin(), active_.end(), [&](const auto& job) {
          return std::find(finished.begin(), finished.end(), job) != finished.end();
        }), active_.end());
      }
    }
  }

  TokenizerBridge& tokenizer_;
  int eos_;
  int active_limit_, queue_limit_, queue_timeout_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<Job>> queue_;
  std::vector<std::shared_ptr<Job>> active_;
  std::unordered_map<std::string, std::weak_ptr<Job>> jobs_;
  bool stopping_ = false;
  Metrics metrics_;
  std::thread worker_;
};

std::atomic<bool> g_shutdown{false};
void signal_shutdown(int) { g_shutdown.store(true); }

void set_error(httplib::Response& res, int status, const std::string& message,
               const std::string& type = "invalid_request_error",
               const std::string& param = "", const std::string& code = "") {
  json error = {{"message", message}, {"type", type}};
  error["param"] = param.empty() ? json(nullptr) : json(param);
  error["code"] = code.empty() ? json(nullptr) : json(code);
  res.status = status;
  res.set_content(json({{"error", error}}).dump(), "application/json");
}

bool authorize(const httplib::Request& req, const Config& cfg, httplib::Response& res) {
  if (cfg.api_key.empty()) return true;
  if (req.get_header_value("Authorization") == "Bearer " + cfg.api_key) return true;
  set_error(res, 401, "missing or invalid bearer token", "authentication_error", "authorization", "invalid_api_key");
  return false;
}

bool is_loopback(const std::string& host) {
  return host == "127.0.0.1" || host == "localhost" || host == "::1";
}

bool unsupported(const json& body, httplib::Response& res) {
  static const std::set<std::string> allowed = {
    "model", "messages", "max_tokens", "max_completion_tokens", "temperature", "top_p", "n",
    "stream", "reasoning", "presence_penalty", "frequency_penalty", "user"
  };
  for (auto it = body.begin(); it != body.end(); ++it) {
    if (!allowed.count(it.key())) {
      set_error(res, 400, "parameter is not supported by this server", "invalid_request_error", it.key(), "unsupported_parameter");
      return true;
    }
  }
  return false;
}

bool read_messages(const json& body, json& messages, httplib::Response& res) {
  auto it = body.find("messages");
  if (it == body.end() || !it->is_array() || it->empty()) {
    set_error(res, 400, "messages must be a non-empty array", "invalid_request_error", "messages", "invalid_value");
    return false;
  }
  messages = json::array();
  for (const auto& message : *it) {
    if (!message.is_object() || message.size() != 2 || !message.contains("role") || !message.contains("content") ||
        !message["role"].is_string() || !message["content"].is_string()) {
      set_error(res, 400, "messages support only role and plain-text content", "invalid_request_error", "messages", "unsupported_value");
      return false;
    }
    const std::string role = message["role"].get<std::string>();
    if (role != "system" && role != "user" && role != "assistant") {
      set_error(res, 400, "supported message roles: system, user, assistant", "invalid_request_error", "messages.role", "unsupported_value");
      return false;
    }
    messages.push_back({{"role", role}, {"content", message["content"]}});
  }
  return true;
}

bool read_request_options(const json& body, const Config& cfg, bool& stream, bool& thinking,
                          int& max_tokens, httplib::Response& res) {
  stream = false; thinking = true; max_tokens = std::min(32, cfg.context);
  if (body.contains("model")) {
    if (!body["model"].is_string() || body["model"].get<std::string>() != cfg.model_id) {
      set_error(res, 404, "model not found", "invalid_request_error", "model", "model_not_found");
      return false;
    }
  }
  if (body.contains("stream")) {
    if (!body["stream"].is_boolean()) { set_error(res, 400, "stream must be boolean", "invalid_request_error", "stream", "invalid_type"); return false; }
    stream = body["stream"].get<bool>();
  }
  for (const char* name : {"max_tokens", "max_completion_tokens"}) {
    if (!body.contains(name)) continue;
    if (body.contains("max_tokens") && body.contains("max_completion_tokens")) {
      set_error(res, 400, "provide only one token limit", "invalid_request_error", "max_tokens", "invalid_value"); return false;
    }
    if (!body[name].is_number_integer()) { set_error(res, 400, "token limit must be an integer", "invalid_request_error", name, "invalid_type"); return false; }
    const bool within_limit = body[name].is_number_unsigned()
      ? (body[name].get<std::uint64_t>() >= 1 && body[name].get<std::uint64_t>() <= static_cast<std::uint64_t>(cfg.context))
      : (body[name].get<std::int64_t>() >= 1 && body[name].get<std::int64_t>() <= cfg.context);
    if (!within_limit) {
      set_error(res, 400, "token limit must be 1.." + std::to_string(cfg.context), "invalid_request_error", name, "invalid_value");
      return false;
    }
    max_tokens = body[name].get<int>();
  }
  if (body.contains("temperature") && (!body["temperature"].is_number() || body["temperature"].get<double>() != 0.0)) {
    set_error(res, 400, "only greedy temperature=0 is supported", "invalid_request_error", "temperature", "unsupported_value"); return false;
  }
  if (body.contains("top_p") && (!body["top_p"].is_number() || body["top_p"].get<double>() != 1.0)) {
    set_error(res, 400, "only top_p=1 is supported by greedy decoding", "invalid_request_error", "top_p", "unsupported_value"); return false;
  }
  if (body.contains("n") && (!body["n"].is_number_integer() || body["n"].get<int>() != 1)) {
    set_error(res, 400, "only n=1 is supported", "invalid_request_error", "n", "unsupported_value"); return false;
  }
  for (const char* name : {"presence_penalty", "frequency_penalty"}) {
    if (body.contains(name) && (!body[name].is_number() || body[name].get<double>() != 0.0)) {
      set_error(res, 400, "non-zero penalties require sampling and are unsupported", "invalid_request_error", name, "unsupported_value"); return false;
    }
  }
  if (body.contains("reasoning")) {
    const auto& reasoning = body["reasoning"];
    if (!reasoning.is_object() || reasoning.size() != 1) {
      set_error(res, 400, "reasoning accepts only {enabled:boolean} or {effort:string}", "invalid_request_error", "reasoning", "unsupported_value"); return false;
    }
    if (reasoning.contains("enabled") && reasoning["enabled"].is_boolean()) thinking = reasoning["enabled"].get<bool>();
    else if (reasoning.contains("effort") && reasoning["effort"].is_string()) {
      const std::string effort = reasoning["effort"].get<std::string>();
      if (effort == "none") thinking = false;
      else if (effort == "low" || effort == "medium" || effort == "high") thinking = true;
      else { set_error(res, 400, "reasoning effort must be none, low, medium, or high", "invalid_request_error", "reasoning.effort", "unsupported_value"); return false; }
    } else {
      set_error(res, 400, "reasoning accepts only {enabled:boolean} or {effort:string}", "invalid_request_error", "reasoning", "unsupported_value"); return false;
    }
  }
  return true;
}

std::string new_id(std::uint64_t number) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "chatcmpl-%llx-%llu",
                static_cast<unsigned long long>(std::time(nullptr)), static_cast<unsigned long long>(number));
  return buf;
}
} // namespace

int main(int argc, char** argv) {
  std::signal(SIGPIPE, SIG_IGN);
  Config cfg;
  bool show_help = false;
  if (!parse_args(argc, argv, cfg, show_help)) return show_help ? 0 : 2;
  if (!std::filesystem::exists(cfg.model) || !std::filesystem::exists(cfg.tokenizer) ||
      !std::filesystem::exists(cfg.gguf_dump) || !std::filesystem::exists(cfg.python)) {
    std::fprintf(stderr, "model/tokenizer/gguf_dump/python path missing\n"); return 2;
  }
  if (!is_loopback(cfg.host) && cfg.api_key.empty()) {
    std::fprintf(stderr, "non-loopback bind requires --api-key\n"); return 2;
  }
  if (cfg.model_id.empty() || cfg.model_id.size() > 64 ||
      !std::all_of(cfg.model_id.begin(), cfg.model_id.end(), [](unsigned char c) { return std::isalnum(c) || c == '_' || c == '-' || c == '.'; })) {
    std::fprintf(stderr, "model id must contain 1..64 letters, digits, '.', '_' or '-'\n"); return 2;
  }

  size_t used0 = 0, used1 = 0;
  pp2batch::vram_used(used0, used1);
  std::fprintf(stderr, "GPU preflight used=%zu/%zu MiB limits=%d/%d MiB\n",
               used0, used1, cfg.gpu0_start_limit_mib, cfg.gpu1_start_limit_mib);
  if (used0 >= static_cast<size_t>(cfg.gpu0_start_limit_mib) || used1 >= static_cast<size_t>(cfg.gpu1_start_limit_mib)) {
    std::fprintf(stderr, "GPU_BUSY: no upload attempted\n"); return 3;
  }

  try {
    const std::string helper = std::filesystem::absolute(argv[0]).parent_path().parent_path().string() + "/serving/tokenizer_bridge.py";
    TokenizerBridge tokenizer(cfg, helper);
    pp2::Stats stats;
    std::string err;
    if (!pp2::upload(stats, err, cfg.model)) {
      std::fprintf(stderr, "PP2 upload failed: %s\n", err.c_str()); pp2::free_all(); return 1;
    }
    const int eos = pp2::eos_token_id();
    if (eos < 0) { std::fprintf(stderr, "GGUF has no valid EOS token\n"); pp2::free_all(); return 1; }
    std::fprintf(stderr, "MODEL ready id=%s eos=%d context=%d upload_sec=%.2f\n", cfg.model_id.c_str(), eos, cfg.context, stats.upload_sec);

    Scheduler scheduler(tokenizer, eos, cfg);
    httplib::Server server;
    server.set_payload_max_length(kMaxBody);
    server.set_keep_alive_timeout(5);
    server.new_task_queue = [] { return new httplib::ThreadPool(64); };

    server.Get("/health", [&](const httplib::Request& req, httplib::Response& res) {
      if (!authorize(req, cfg, res)) return;
      res.set_content(R"({"status":"ok","ready":true})", "application/json");
    });
    server.Get("/metrics", [&](const httplib::Request& req, httplib::Response& res) {
      if (!authorize(req, cfg, res)) return;
      res.set_content(scheduler.metrics_text(cfg.model_id), "text/plain; version=0.0.4; charset=utf-8");
    });
    server.Get("/v1/models", [&](const httplib::Request& req, httplib::Response& res) {
      if (!authorize(req, cfg, res)) return;
      res.set_content(json({{"object", "list"}, {"data", json::array({{{"id", cfg.model_id}, {"object", "model"}, {"created", std::time(nullptr)}, {"owned_by", "hermes-27"}}})}}).dump(), "application/json");
    });
    server.Post("/v1/cancel", [&](const httplib::Request& req, httplib::Response& res) {
      if (!authorize(req, cfg, res)) return;
      json body;
      try { body = json::parse(req.body); }
      catch (...) { set_error(res, 400, "invalid JSON body", "invalid_request_error", "body", "invalid_json"); return; }
      if (!body.is_object() || !body.contains("id") || !body["id"].is_string()) {
        set_error(res, 400, "body must contain request id", "invalid_request_error", "id", "invalid_value"); return;
      }
      const std::string id = body["id"].get<std::string>();
      if (!scheduler.cancel(id)) { set_error(res, 404, "request is not active", "invalid_request_error", "id", "not_found"); return; }
      res.status = 202;
      res.set_content(json({{"id", id}, {"cancelled", true}}).dump(), "application/json");
    });
    server.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
      scheduler.request();
      if (!authorize(req, cfg, res)) { scheduler.rejected(); return; }
      json body;
      try { body = json::parse(req.body); }
      catch (...) { scheduler.rejected(); set_error(res, 400, "invalid JSON body", "invalid_request_error", "body", "invalid_json"); return; }
      if (!body.is_object()) { scheduler.rejected(); set_error(res, 400, "request body must be a JSON object", "invalid_request_error", "body", "invalid_type"); return; }
      if (unsupported(body, res)) { scheduler.rejected(); return; }
      bool stream = false, thinking = true;
      int max_tokens = 32;
      if (!read_request_options(body, cfg, stream, thinking, max_tokens, res)) { scheduler.rejected(); return; }
      json messages;
      if (!read_messages(body, messages, res)) { scheduler.rejected(); return; }

      std::vector<int> prompt;
      try { prompt = tokenizer.encode(messages, thinking); }
      catch (const std::exception& e) { scheduler.rejected(); set_error(res, 400, std::string("tokenization/template failed: ") + e.what(), "invalid_request_error", "messages", "invalid_value"); return; }
      if (prompt.empty()) { scheduler.rejected(); set_error(res, 400, "chat template produced empty prompt", "invalid_request_error", "messages", "invalid_value"); return; }
      if (std::any_of(prompt.begin(), prompt.end(), [](int id) { return id < 0 || id >= kVocab; })) {
        scheduler.rejected(); set_error(res, 400, "tokenizer returned an ID outside the model vocabulary", "invalid_request_error", "messages", "invalid_value"); return;
      }
      if (prompt.size() > static_cast<std::size_t>(cfg.context) ||
          prompt.size() + static_cast<std::size_t>(max_tokens) - 1 > static_cast<std::size_t>(cfg.context)) {
        scheduler.rejected(); set_error(res, 400, "prompt plus max_tokens exceeds PP2 context limit " + std::to_string(cfg.context), "invalid_request_error", "max_tokens", "context_length_exceeded"); return;
      }

      static std::atomic<std::uint64_t> sequence{1};
      auto job = std::make_shared<Job>();
      job->id = new_id(sequence.fetch_add(1));
      job->model = cfg.model_id;
      job->prompt = std::move(prompt);
      job->max_new_tokens = max_tokens;
      job->stream = stream;
      job->created = std::time(nullptr);
      if (!scheduler.enqueue(job)) {
        scheduler.rejected();
        res.set_header("Retry-After", "1");
        set_error(res, 429, "serving queue is full", "rate_limit_error", "queue", "queue_full");
        return;
      }
      if (stream) {
        // Wait for admission before sending HTTP 200/SSE headers. Requests
        // rejected by the VRAM budget or queue deadline must return HTTP
        // 429/503 rather than a successful stream containing an error event.
        {
          std::unique_lock<std::mutex> lock(job->mutex);
          job->cv.wait(lock, [&] { return job->admitted || job->done.load(); });
          if (!job->admitted && job->http_status != 200) {
            const int status = job->http_status;
            const std::string error = job->error;
            lock.unlock();
            scheduler.retire(job->id);
            if (status == 429 || status == 503) res.set_header("Retry-After", "1");
            set_error(res, status, error, status == 429 ? "rate_limit_error" : "server_error", "",
                      status == 429 ? "overloaded" : "inference_error");
            return;
          }
        }
        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");
        res.set_chunked_content_provider("text/event-stream; charset=utf-8",
          [job, &scheduler](std::size_t, httplib::DataSink& sink) {
            for (;;) {
              if (!sink.is_writable()) { scheduler.cancel(job->id); return false; }
              std::string chunk_text;
              {
                std::unique_lock<std::mutex> lock(job->mutex);
                job->cv.wait_for(lock, std::chrono::milliseconds(250), [&] { return !job->stream_chunks.empty() || job->done.load(); });
                if (!job->stream_chunks.empty()) { chunk_text = std::move(job->stream_chunks.front()); job->stream_chunks.pop_front(); }
                else if (job->done.load()) { sink.done(); return true; }
              }
              if (!chunk_text.empty() && !sink.write(chunk_text.data(), chunk_text.size())) { scheduler.cancel(job->id); return false; }
            }
          },
          [job, &scheduler](bool success) { if (!success) scheduler.cancel(job->id); scheduler.retire(job->id); });
        return;
      }

      std::unique_lock<std::mutex> lock(job->mutex);
      job->cv.wait(lock, [&] { return job->done.load(); });
      const int status = job->http_status;
      const json response = job->response;
      const std::string error = job->error;
      lock.unlock();
      scheduler.retire(job->id);
      if (status != 200) { res.set_header("Retry-After", "1"); set_error(res, status, error, status == 429 ? "rate_limit_error" : "server_error", "", status == 429 ? "overloaded" : "inference_error"); return; }
      res.set_content(response.dump(), "application/json");
    });
    server.set_error_handler([](const httplib::Request&, httplib::Response& res) {
      // cpp-httplib calls this handler for any 4xx/5xx response. Preserve
      // structured API errors already set by the request handlers.
      if (res.body.empty())
        set_error(res, res.status, "route not found", "invalid_request_error", "path", "not_found");
    });

    int port = cfg.port;
    const bool bound = cfg.port == 0 ? ((port = server.bind_to_any_port(cfg.host)) > 0) : server.bind_to_port(cfg.host, cfg.port);
    if (!bound) { std::fprintf(stderr, "HTTP bind failed host=%s port=%d\n", cfg.host.c_str(), cfg.port); scheduler.stop(); pp2::free_all(); return 1; }
    std::fprintf(stderr, "LISTEN http://%s:%d model=%s active=%d queue=%d\n", cfg.host.c_str(), port, cfg.model_id.c_str(), cfg.active, cfg.queued);
    std::signal(SIGINT, signal_shutdown);
    std::signal(SIGTERM, signal_shutdown);
    std::atomic<bool> server_finished{false};
    std::thread listener([&] { server.listen_after_bind(); server_finished.store(true); });
    std::thread watcher([&] {
      while (!g_shutdown.load() && !server_finished.load()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (g_shutdown.load() && !server_finished.load()) server.stop();
    });
    listener.join();
    g_shutdown.store(true);
    watcher.join();
    scheduler.stop();
    pp2::free_all();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "server startup/runtime error: %s\n", e.what());
    pp2::free_all();
    return 1;
  }
  return 0;
}
