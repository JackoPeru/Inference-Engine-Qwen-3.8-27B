// HERMES-27 P5 batch concorrente PP2 — implementazione v1.
// Sequenziale: loop pp2::decode_step (corretto, overhead onesto).
// Parallelo: N host threads su pp2::decode_step (overlap reale GPU0/GPU1,
// xfer pinned reso thread-safe da mutex in pp2_fwd.cu).
#include "pp2_batch.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
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

bool admit_mixed(const std::vector<int>& capacities, std::string& detail) {
  // Same conservative baseline as admit_check; account for the real capacity
  // of each independently allocated DecodeState rather than N * max(cap).
  constexpr size_t MiB = 1024u * 1024u;
  constexpr size_t GDN = 24u * (48u * 128u * 128u + 10240u * 3u) * sizeof(float);
  constexpr size_t WS = 568064u;
  constexpr size_t STEP0 = sizeof(int) + 5120u * sizeof(float);
  constexpr size_t STEP1 = 2u * 5120u * sizeof(float) + 248320u * sizeof(float);
  constexpr size_t KV = 8u * 2u * 1024u * sizeof(float);
  constexpr size_t BUDGET0 = (15360u - 9856u - 1024u) * MiB;
  constexpr size_t BUDGET1 = (15360u - 11470u - 1024u) * MiB;
  if (capacities.empty() || capacities.size() > 16) {
    detail = "mixed admission: request count must be 1..16";
    return false;
  }
  size_t need0 = 0, need1 = 0;
  for (int cap : capacities) {
    if (cap < 1 || cap > pp2::DecodeState::MAX_CONTEXT) {
      detail = "mixed admission: context must be 1.." +
               std::to_string(pp2::DecodeState::MAX_CONTEXT);
      return false;
    }
    need0 += GDN + KV * static_cast<size_t>(cap) + WS + STEP0;
    need1 += GDN + KV * static_cast<size_t>(cap) + WS + STEP1;
  }
  bool admitted = need0 <= BUDGET0 && need1 <= BUDGET1;
  char buf[320];
  std::snprintf(buf, sizeof(buf),
                "mixed admission requests=%zu need0=%.1fMiB need1=%.1fMiB "
                "budget0=%.0fMiB budget1=%.0fMiB FP32_KV %s",
                capacities.size(), double(need0) / MiB, double(need1) / MiB,
                double(BUDGET0) / MiB, double(BUDGET1) / MiB,
                admitted ? "ADMIT" : "REJECT");
  detail = buf;
  return admitted;
}

struct ContinuousScheduler::Job {
  ScheduledRequest request;
  SchedulerClock::time_point queued_at;
  int capacity = 0;
  JobStatus status = JobStatus::queued;
  bool cancel_requested = false;
  std::size_t prompt_consumed = 0;
  std::vector<int> generated;
  std::string error;
  void* state = nullptr; // exclusively owned by the scheduler pump
};

ContinuousScheduler::ContinuousScheduler(SchedulerConfig config, StepBackend& backend)
    : config_(config), backend_(backend) {
  if (config.max_active < 1 || config.max_active > 16 ||
      config.max_queued < 1 || config.max_queued > 32 ||
      config.steps_per_tick < 1 || config.steps_per_tick > 16 ||
      config.max_retained > 1024 ||
      config.queue_timeout.count() <= 0 ||
      config.max_context < 1 ||
      config.max_context > pp2::DecodeState::MAX_CONTEXT ||
      config.vocab_size < 1 ||
      config.eos_token_id < -1 ||
      config.eos_token_id >= config.vocab_size) {
    throw std::invalid_argument("invalid continuous scheduler configuration");
  }
}

ContinuousScheduler::~ContinuousScheduler() { shutdown(); }

bool ContinuousScheduler::closed(JobStatus status) const {
  return status == JobStatus::completed || status == JobStatus::cancelled ||
         status == JobStatus::expired || status == JobStatus::failed;
}

void ContinuousScheduler::trim_finished_locked() {
  while (finished_order_.size() > config_.max_retained) {
    auto id = std::move(finished_order_.front());
    finished_order_.pop_front();
    jobs_.erase(id);
  }
}

SubmitResult ContinuousScheduler::enqueue(ScheduledRequest request,
                                           SchedulerClock::time_point now) {
  const auto invalid = [this]() -> SubmitResult {
    std::lock_guard<std::mutex> guard(mutex_);
    ++counters_.rejected_total;
    return {SubmitCode::invalid, "invalid id, prompt, token, or context capacity"};
  };
  if (request.id.empty() || request.id.size() > 128 ||
      request.prompt.empty() || request.max_new_tokens < 1 ||
      request.prompt.size() > static_cast<size_t>(config_.max_context) ||
      static_cast<size_t>(request.max_new_tokens - 1) >
          static_cast<size_t>(config_.max_context) - request.prompt.size() ||
      std::any_of(request.prompt.begin(), request.prompt.end(), [this](int t) {
        return t < 0 || t >= config_.vocab_size;
      })) return invalid();

  int capacity = static_cast<int>(request.prompt.size()) +
                 request.max_new_tokens - 1;
  std::string estimate;
  if (!admit_mixed({capacity}, estimate)) {
    std::lock_guard<std::mutex> guard(mutex_);
    ++counters_.rejected_total;
    return {SubmitCode::memory_limit, std::move(estimate)};
  }

  std::lock_guard<std::mutex> guard(mutex_);
  if (stopping_) { ++counters_.rejected_total; return {SubmitCode::stopped, "scheduler stopped"}; }
  if (jobs_.count(request.id)) {
    ++counters_.rejected_total;
    return {SubmitCode::duplicate, "duplicate request id"};
  }
  if (waiting_.size() >= config_.max_queued) {
    ++counters_.rejected_total;
    return {SubmitCode::queue_full, "waiting queue full"};
  }
  auto job = std::make_shared<Job>();
  job->capacity = capacity;
  job->queued_at = now;
  job->request = std::move(request);
  jobs_[job->request.id] = job;
  waiting_.push_back(std::move(job));
  ++counters_.accepted_total;
  return {SubmitCode::accepted, {}};
}

bool ContinuousScheduler::cancel(const std::string& id) {
  std::shared_ptr<Job> immediate;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = jobs_.find(id);
    if (it == jobs_.end() || closed(it->second->status) ||
        it->second->cancel_requested) return false;
    auto job = it->second; // own a ref: terminal-history trim can erase jobs_[id]
    job->cancel_requested = true;
    // Atomically remove a queued request. Never race finish() against a pump
    // thread opening GPU state after it popped the request for admission.
    auto queued = std::find(waiting_.begin(), waiting_.end(), job);
    if (queued != waiting_.end()) {
      waiting_.erase(queued);
      job->status = JobStatus::cancelled;
      ++counters_.cancelled_total;
      finished_order_.push_back(job->request.id);
      trim_finished_locked();
      immediate = job;
    }
  }
  // An opening or active request is reaped by the pump after the backend call.
  if (immediate) emit_event(immediate, EventKind::terminal);
  return true;
}

bool ContinuousScheduler::inspect(const std::string& id, JobSnapshot& out) const {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = jobs_.find(id);
  if (it == jobs_.end()) return false;
  const Job& job = *it->second;
  out = {job.status, job.prompt_consumed, job.generated, job.capacity, job.error};
  return true;
}

bool ContinuousScheduler::retire(const std::string& id) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = jobs_.find(id);
  if (it == jobs_.end() || !closed(it->second->status)) return false;
  jobs_.erase(it);
  finished_order_.erase(std::remove(finished_order_.begin(), finished_order_.end(), id),
                        finished_order_.end());
  return true;
}

SchedulerMetrics ContinuousScheduler::metrics() const {
  std::lock_guard<std::mutex> guard(mutex_);
  SchedulerMetrics result = counters_;
  result.stopping = stopping_;
  result.active = running_.size();
  result.queued = waiting_.size();
  result.retained = jobs_.size();
  return result;
}

void ContinuousScheduler::emit_event(const std::shared_ptr<Job>& job,
                                      EventKind kind, int token_id) {
  if (!config_.on_event) return;
  SchedulerEvent event;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    event.kind = kind;
    event.id = job->request.id;
    event.token_id = token_id;
    event.status = job->status;
    event.prompt_consumed = job->prompt_consumed;
    if (kind == EventKind::terminal) {
      event.generated = job->generated;
      event.error = job->error;
    }
  }
  // Callbacks can signal the HTTP request's condition variable or invoke
  // cancel/inspect/retire safely. A throwing consumer cannot tear down the
  // GPU scheduling loop; callback_errors_total exposes this failure.
  try {
    config_.on_event(event);
  } catch (...) {
    std::lock_guard<std::mutex> guard(mutex_);
    ++counters_.callback_errors_total;
  }
}

void ContinuousScheduler::finish(const std::shared_ptr<Job>& job, JobStatus status,
                                  const std::string& error) {
  void* state = nullptr;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (closed(job->status)) return;
    waiting_.erase(std::remove(waiting_.begin(), waiting_.end(), job), waiting_.end());
    running_.erase(std::remove(running_.begin(), running_.end(), job), running_.end());
    state = job->state;
    job->state = nullptr;
    job->status = status;
    job->error = error;
    if (status == JobStatus::completed) ++counters_.completed_total;
    else if (status == JobStatus::cancelled) ++counters_.cancelled_total;
    else if (status == JobStatus::expired) ++counters_.expired_total;
    else if (status == JobStatus::failed) ++counters_.failed_total;
    finished_order_.push_back(job->request.id);
    trim_finished_locked();
  }
  if (state) backend_.close(state);
  emit_event(job, EventKind::terminal);
}

void ContinuousScheduler::expire_waiting(SchedulerClock::time_point now) {
  std::vector<std::shared_ptr<Job>> expired;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    for (const auto& job : waiting_)
      if (now >= job->queued_at && now - job->queued_at >= config_.queue_timeout)
        expired.push_back(job);
  }
  for (const auto& job : expired) finish(job, JobStatus::expired, "queue timeout");
}

void ContinuousScheduler::promote(std::size_t& remaining_admissions) {
  for (;;) {
    std::shared_ptr<Job> job;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (stopping_ || remaining_admissions == 0 || waiting_.empty() ||
          running_.size() >= config_.max_active)
        return;
      // FIFO prevents a large request from being overtaken indefinitely. The
      // head stays queued when active allocations cannot fit its FP32 state.
      std::vector<int> capacities;
      capacities.reserve(running_.size() + 1);
      for (const auto& current : running_) capacities.push_back(current->capacity);
      job = waiting_.front();
      capacities.push_back(job->capacity);
      std::string detail;
      if (!admit_mixed(capacities, detail)) return;
      waiting_.pop_front();
      --remaining_admissions; // includes failed opens, avoiding retry storms
    }

    void* state = nullptr;
    std::string error;
    bool opened = false;
    try {
      opened = backend_.open(job->capacity, state, error);
    } catch (const std::exception& e) {
      error = e.what();
    } catch (...) {
      error = "backend open threw an unknown exception";
    }
    if (!opened || !state) {
      if (state) backend_.close(state); // defend against broken backend open
      bool cancelled;
      {
        std::lock_guard<std::mutex> guard(mutex_);
        cancelled = job->cancel_requested || stopping_;
      }
      finish(job, cancelled ? JobStatus::cancelled : JobStatus::failed,
             cancelled ? "" : "PP2 state admission failed: " + error);
      continue;
    }
    bool cancelled;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      job->state = state;
      cancelled = job->cancel_requested || stopping_;
      if (!cancelled) {
        job->status = JobStatus::active;
        running_.push_back(job);
      }
    }
    if (cancelled) finish(job, JobStatus::cancelled);
    else emit_event(job, EventKind::admitted);
  }
}

std::size_t ContinuousScheduler::tick(SchedulerClock::time_point now) {
  std::lock_guard<std::mutex> pump(pump_mutex_);
  const auto entered_at = SchedulerClock::now();
  // Advance even an injected deterministic clock by actual time spent in
  // backend calls. A long decode must not make a queued request appear
  // unexpired until the next separate tick() invocation.
  const auto effective_now = [&]() {
    return now + (SchedulerClock::now() - entered_at);
  };
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (stopping_) return 0;
  }
  std::size_t decoded = 0;
  // Event callbacks may enqueue replacement requests for a failed allocation;
  // bound total state-open attempts in the entire tick, including failures.
  std::size_t remaining_admissions = 16;
  // Skipping cancelled slots must not consume the decode fairness quantum.
  std::size_t skipped = 0;
  while (decoded < config_.steps_per_tick && skipped <= config_.max_active) {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (stopping_) break; // shutdown can flag stop while a GPU step runs
    }
    expire_waiting(effective_now());
    promote(remaining_admissions);
    std::shared_ptr<Job> job;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (running_.empty()) break;
      job = running_.front();
      running_.pop_front();
      running_.push_back(job); // rotate after exactly one step per request
    }
    bool cancelled;
    int token;
    long position;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      cancelled = job->cancel_requested;
      if (job->prompt_consumed < job->request.prompt.size()) {
        position = static_cast<long>(job->prompt_consumed);
        token = job->request.prompt[job->prompt_consumed];
      } else {
        position = static_cast<long>(job->request.prompt.size() + job->generated.size() - 1);
        token = job->generated.back();
      }
    }
    if (cancelled) {
      finish(job, JobStatus::cancelled);
      ++skipped;
      continue;
    }
    std::vector<float> logits;
    std::string error;
    bool ok = false;
    try {
      ok = backend_.step(job->state, token, position, logits, error);
    } catch (const std::exception& e) {
      error = e.what();
    } catch (...) {
      error = "backend step threw an unknown exception";
    }
    ++decoded;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      ++counters_.decode_steps_total;
      cancelled = job->cancel_requested;
    }
    if (cancelled) { finish(job, JobStatus::cancelled); continue; }
    if (!ok || logits.size() != static_cast<size_t>(config_.vocab_size)) {
      finish(job, JobStatus::failed,
             "PP2 decode failed: " + (error.empty() ? "wrong logits size or empty output" : error));
      continue;
    }
    int best_id = -1;
    float best = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < logits.size(); ++i) {
      if (!std::isfinite(logits[i])) { best_id = -1; error = "non-finite logit"; break; }
      if (logits[i] > best) { best = logits[i]; best_id = static_cast<int>(i); }
    }
    if (best_id < 0) {
      finish(job, JobStatus::failed, "invalid logits: " + error);
      continue;
    }
    bool done = false, produced_token = false;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (job->prompt_consumed < job->request.prompt.size()) {
        ++job->prompt_consumed;
        if (job->prompt_consumed == job->request.prompt.size()) {
          job->generated.push_back(best_id);
          produced_token = true;
        }
      } else {
        job->generated.push_back(best_id);
        produced_token = true;
      }
      done = !job->generated.empty() &&
             (job->generated.back() == config_.eos_token_id ||
              job->generated.size() >= static_cast<size_t>(job->request.max_new_tokens));
    }
    if (produced_token) emit_event(job, EventKind::token, best_id);
    if (done) {
      bool cancelled_now;
      {
        std::lock_guard<std::mutex> guard(mutex_);
        cancelled_now = job->cancel_requested;
      }
      finish(job, cancelled_now ? JobStatus::cancelled : JobStatus::completed);
    }
  }
  return decoded;
}

void ContinuousScheduler::shutdown() {
  // Fail closed promptly while the pump is blocked on a noninterruptible
  // CUDA step: prevent new enqueue and flag active requests before joining.
  {
    std::lock_guard<std::mutex> guard(mutex_);
    stopping_ = true;
    for (const auto& job : waiting_) job->cancel_requested = true;
    for (const auto& job : running_) job->cancel_requested = true;
    // One opening request may have been popped from waiting_, but promote()
    // checks stopping_ after its GPU allocation and releases that state.
  }
  std::lock_guard<std::mutex> pump(pump_mutex_);
  std::vector<std::shared_ptr<Job>> jobs;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    jobs.insert(jobs.end(), waiting_.begin(), waiting_.end());
    jobs.insert(jobs.end(), running_.begin(), running_.end());
  }
  for (const auto& job : jobs) finish(job, JobStatus::cancelled);
}

#ifndef HERMES_ADMISSION_CPU_ONLY
bool PP2StepBackend::open(int capacity, void*& state, std::string& error) {
  state = nullptr;
  if (capacity < 1 || capacity > pp2::DecodeState::MAX_CONTEXT) {
    error = "invalid PP2 context capacity";
    return false;
  }
  constexpr size_t LIMIT_BYTES = 15360ULL * 1024 * 1024;
  constexpr size_t MiB = 1024u * 1024u;
  const size_t reservation = 96u * MiB + static_cast<size_t>(capacity) * 64u * 1024u;
  for (int device = 0; device < 2; ++device) {
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaSetDevice(device) != cudaSuccess ||
        cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
        total_bytes < free_bytes || total_bytes - free_bytes > LIMIT_BYTES ||
        reservation > LIMIT_BYTES - (total_bytes - free_bytes)) {
      error = "live VRAM preflight rejected PP2 state on GPU" + std::to_string(device);
      return false;
    }
  }
  std::unique_ptr<pp2::DecodeState> ds(new (std::nothrow) pp2::DecodeState{});
  if (!ds) { error = "host allocation failed"; return false; }
  if (!pp2::decode_init(*ds, error, capacity)) {
    pp2::decode_free(*ds);
    return false;
  }
  for (int device = 0; device < 2; ++device) {
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaSetDevice(device) != cudaSuccess ||
        cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
        total_bytes < free_bytes || total_bytes - free_bytes > LIMIT_BYTES) {
      error = "live VRAM over limit after PP2 state allocation";
      pp2::decode_free(*ds);
      return false;
    }
  }
  state = ds.release();
  return true;
}

bool PP2StepBackend::step(void* state, int token, long position,
                           std::vector<float>& logits, std::string& error) {
  if (!state) { error = "null PP2 state"; return false; }
  auto& ds = *static_cast<pp2::DecodeState*>(state);
  if (!ds.inited || position < 0 || position >= ds.capacity) {
    error = "PP2 decode position outside state capacity";
    return false;
  }
  logits = pp2::decode_step(token, ds, position, error);
  return !logits.empty();
}

void PP2StepBackend::close(void* state) noexcept {
  if (!state) return;
  auto* ds = static_cast<pp2::DecodeState*>(state);
  pp2::decode_free(*ds);
  delete ds;
}

void vram_used(size_t& used0_MiB, size_t& used1_MiB) {
  size_t f0 = 0, t0 = 0, f1 = 0, t1 = 0;
  cudaSetDevice(0); cudaMemGetInfo(&f0, &t0);
  cudaSetDevice(1); cudaMemGetInfo(&f1, &t1);
  used0_MiB = (t0 - f0) / 1048576;
  used1_MiB = (t1 - f1) / 1048576;
}
#endif // HERMES_ADMISSION_CPU_ONLY

} // namespace pp2batch
