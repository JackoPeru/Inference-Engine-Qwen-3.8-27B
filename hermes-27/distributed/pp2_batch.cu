// HERMES-27 P5 batch concorrente PP2 — implementazione v1.
// Sequenziale: loop pp2::decode_step (corretto, overhead onesto).
// Parallelo: N host threads su pp2::decode_step (overlap reale GPU0/GPU1,
// xfer pinned reso thread-safe da mutex in pp2_fwd.cu).
#include "pp2_batch.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
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
  // Anonymous per-state timing scratch. The pump owns admission/commit;
  // exactly one backend worker writes step start/end for a state at a time.
  // Relaxed atomics are safe because the pump joins before reading them.
  std::int64_t observed_queued_ns = 0;
  std::atomic<std::int64_t> observed_step_begin_ns{0};
  std::atomic<std::int64_t> observed_step_end_ns{0};
  int capacity = 0;
  JobStatus status = JobStatus::queued;
  bool cancel_requested = false;
  // Only used by opt-in ready completion; exclusively set/cleared under
  // metadata mutex by the one scheduler pump. Prevents same-state overlap.
  bool ready_step_in_flight = false;
  std::size_t prompt_consumed = 0;
  std::vector<int> generated;
  std::string error;
  void* state = nullptr; // exclusively owned by the scheduler pump
};

namespace {
void saturating_observation(std::uint64_t& count, std::uint64_t& total,
                            std::uint64_t& maximum, std::int64_t begin,
                            std::int64_t end) noexcept {
  if (begin <= 0 || end < begin) return;
  const auto elapsed = static_cast<std::uint64_t>(end - begin);
  if (count != UINT64_MAX) ++count;
  total = elapsed > UINT64_MAX - total ? UINT64_MAX : total + elapsed;
  maximum = std::max(maximum, elapsed);
}
}  // namespace

std::int64_t ContinuousScheduler::observation_now_ns() const noexcept {
  if (!config_.enable_observability) return 0;
  try {
#ifdef HERMES_ADMISSION_CPU_ONLY
    const auto now = config_.observability_now
                         ? config_.observability_now() : SchedulerClock::now();
#else
    const auto now = SchedulerClock::now();
#endif
    const auto nanos =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();
    return nanos > 0 ? nanos : 0;
  } catch (...) {
    // Host observation is never allowed to make the inference loop fail.
    return 0;
  }
}

void ContinuousScheduler::observe_step_begin(Job& job) noexcept {
  job.observed_step_end_ns.store(0, std::memory_order_relaxed);
  job.observed_step_begin_ns.store(observation_now_ns(),
                                   std::memory_order_relaxed);
}

void ContinuousScheduler::observe_step_end(Job& job) noexcept {
  job.observed_step_end_ns.store(observation_now_ns(),
                                 std::memory_order_relaxed);
}

void ContinuousScheduler::observe_step_joined_locked(const Job& job) noexcept {
  if (!config_.enable_observability) return;
  saturating_observation(
      counters_.observed_backend_steps,
      counters_.backend_step_ns_total, counters_.backend_step_ns_max,
      job.observed_step_begin_ns.load(std::memory_order_relaxed),
      job.observed_step_end_ns.load(std::memory_order_relaxed));
}

void ContinuousScheduler::observe_queue_admission(
    const std::shared_ptr<Job>& job) noexcept {
  if (!config_.enable_observability) return;
  const auto end = observation_now_ns();
  try {
    std::lock_guard<std::mutex> guard(mutex_);
    saturating_observation(counters_.observed_queue_admissions,
                           counters_.queue_wait_ns_total,
                           counters_.queue_wait_ns_max,
                           job->observed_queued_ns, end);
  } catch (...) {}
}

void ContinuousScheduler::observe_token_commit(
    const std::shared_ptr<Job>& job) noexcept {
  if (!config_.enable_observability) return;
  const auto end = observation_now_ns();
  try {
    std::lock_guard<std::mutex> guard(mutex_);
    saturating_observation(
        counters_.observed_token_commits,
        counters_.backend_ready_to_token_commit_ns_total,
        counters_.backend_ready_to_token_commit_ns_max,
        job->observed_step_end_ns.load(std::memory_order_relaxed), end);
  } catch (...) {}
}

void ContinuousScheduler::observe_terminal_locked(
    const Job& job, JobStatus status, std::int64_t end) noexcept {
  if (!config_.enable_observability || status != JobStatus::completed)
    return;
  const auto begin = job.observed_step_end_ns.load(std::memory_order_relaxed);
  if (config_.enable_ready_completion_dispatch)
    saturating_observation(counters_.observed_ready_terminals,
                           counters_.ready_terminal_ns_total,
                           counters_.ready_terminal_ns_max, begin, end);
  else if (config_.enable_parallel_steps)
    saturating_observation(counters_.observed_wave_terminals,
                           counters_.wave_terminal_ns_total,
                           counters_.wave_terminal_ns_max, begin, end);
  else
    saturating_observation(counters_.observed_serial_terminals,
                           counters_.serial_terminal_ns_total,
                           counters_.serial_terminal_ns_max, begin, end);
}

void ContinuousScheduler::observe_callback(
    EventKind kind, std::int64_t began_ns) noexcept {
  if (!config_.enable_observability) return;
  const auto end = observation_now_ns();
  try {
    std::lock_guard<std::mutex> guard(mutex_);
    saturating_observation(counters_.observed_callbacks,
                           counters_.callback_ns_total,
                           counters_.callback_ns_max, began_ns, end);
    if (kind == EventKind::terminal)
      saturating_observation(counters_.observed_terminal_callbacks,
                             counters_.terminal_callback_ns_total,
                             counters_.terminal_callback_ns_max,
                             began_ns, end);
  } catch (...) {}
}

void ContinuousScheduler::observe_outstanding(
    std::size_t outstanding) noexcept {
  if (!config_.enable_observability) return;
  try {
    std::lock_guard<std::mutex> guard(mutex_);
    counters_.outstanding_step_high_water =
        std::max(counters_.outstanding_step_high_water,
                 static_cast<std::uint64_t>(outstanding));
  } catch (...) {}
}

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
  if (config_.enable_parallel_steps && !backend_.supports_parallel_steps()) {
    throw std::invalid_argument(
        "parallel PP2 steps require explicit thread-safe async backend capability");
  }
  if (config_.enable_ready_completion_dispatch &&
      (!config_.enable_parallel_steps || config_.enable_batched_prefill ||
       !backend_.supports_parallel_steps())) {
    throw std::invalid_argument(
        "ready completion requires parallel async backend and forbids seeded prefill");
  }
  counters_.observability_enabled = config_.enable_observability;
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

// With mutex_ held, terminal retention is best effort. A failed allocation
// must never skip the DecodeState close or the terminal notification.
bool ContinuousScheduler::retain_finished_locked(
    const std::shared_ptr<Job>& job) noexcept {
  try {
#ifdef HERMES_ADMISSION_CPU_ONLY
    if (config_.terminal_before_history_append)
      config_.terminal_before_history_append();
#endif
    finished_order_.push_back(job->request.id);
    trim_finished_locked();
    return true;
  } catch (...) {
    ++counters_.terminal_history_failures_total;
    return false;
  }
}

// A terminal record without a history entry cannot be retained indefinitely.
// Deliver its event first, then remove the exact job if still present; a
// reentrant callback may already have retired it.
void ContinuousScheduler::forget_unretained(
    const std::shared_ptr<Job>& job) noexcept {
  try {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto it = jobs_.find(job->request.id);
    if (it != jobs_.end() && it->second == job) jobs_.erase(it);
  } catch (...) {
    // Emergency cleanup cannot make another allocation or throw.
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

  const auto queued_observation = config_.enable_observability
                                      ? observation_now_ns() : 0;
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
  job->observed_queued_ns = queued_observation;
  jobs_[job->request.id] = job;
  waiting_.push_back(std::move(job));
  ++counters_.accepted_total;
  if (config_.enable_observability)
    counters_.queued_high_water =
        std::max(counters_.queued_high_water,
                 static_cast<std::uint64_t>(waiting_.size()));
  return {SubmitCode::accepted, {}};
}

bool ContinuousScheduler::cancel(const std::string& id) {
  std::shared_ptr<Job> immediate;
  bool retained = true;
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
      retained = retain_finished_locked(job);
      immediate = job;
    }
  }
  // An opening or active request is reaped by the pump after the backend call.
  if (immediate) {
    emit_event(immediate, EventKind::terminal);
    if (!retained) forget_unretained(immediate);
  }
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
                                      EventKind kind, int token_id) noexcept {
  if (!config_.on_event) return;
  SchedulerEvent event;
  // A terminal callback can be the only wakeup for an HTTP/SSE waiter.
  // Both the event snapshot and the consumer can allocate/throw. A failed
  // snapshot gets a minimal *failed* terminal event; if delivery also fails,
  // the consumer's borrowed-ID emergency notifier must wake its waiters.
  const auto record_failure = [&]() noexcept {
    try {
      std::lock_guard<std::mutex> guard(mutex_);
      ++counters_.callback_errors_total;
      if (kind == EventKind::terminal)
        ++counters_.terminal_delivery_failures_total;
    } catch (...) {}
  };
  const auto wake_failure = [&]() noexcept {
    if (kind != EventKind::terminal || !config_.on_terminal_delivery_failure)
      return;
    try {
      config_.on_terminal_delivery_failure(job->request.id, job->status);
    } catch (...) {
      record_failure();
    }
  };
  // Timing wraps only the actual sink invocation, never holds mutex_ during
  // user code and samples even if the sink throws.
  const auto deliver = [&](const SchedulerEvent& snapshot) noexcept {
    const auto began = config_.enable_observability ? observation_now_ns() : 0;
    bool success = true;
    try {
      config_.on_event(snapshot);
    } catch (...) {
      success = false;
      record_failure();
    }
    if (config_.enable_observability) observe_callback(kind, began);
    return success;
  };
  try {
    std::lock_guard<std::mutex> guard(mutex_);
    event.kind = kind;
#ifdef HERMES_ADMISSION_CPU_ONLY
    if (kind == EventKind::terminal && config_.terminal_before_event_id_copy)
      config_.terminal_before_event_id_copy();
#endif
    event.id = job->request.id;
    event.token_id = token_id;
    event.status = job->status;
    event.prompt_consumed = job->prompt_consumed;
    if (kind == EventKind::terminal) {
#ifdef HERMES_ADMISSION_CPU_ONLY
      if (config_.terminal_before_snapshot_copy)
        config_.terminal_before_snapshot_copy();
#endif
      event.generated = job->generated;
      event.error = job->error;
    }
  } catch (...) {
    record_failure();
    if (kind == EventKind::terminal) {
      // If copying the ID itself exhausted memory, a minimal event has no
      // trustworthy recipient. Prefer the borrowed-ID emergency notifier
      // over another allocation. Preserve the historical best-effort event
      // retry when a caller has NOT installed that emergency notifier.
      // This avoids regressing standalone consumers after a transient OOM.
      if (event.id.empty() && config_.on_terminal_delivery_failure) {
        wake_failure();
        return;
      }
      try {
        event.kind = EventKind::terminal;
        if (event.id.empty()) event.id = job->request.id;
        event.status = JobStatus::failed;
        event.generated.clear();
        event.error.clear();
        if (deliver(event)) return;
      } catch (...) { record_failure(); }
      wake_failure();
    }
    return;
  }
  // Callbacks can signal the HTTP request's condition variable or invoke
  // cancel/inspect/retire safely. A throwing consumer cannot tear down the
  // GPU scheduling loop; callback_errors_total exposes this failure.
  if (!deliver(event)) wake_failure();
}

void ContinuousScheduler::finish(const std::shared_ptr<Job>& job, JobStatus status,
                                  const std::string& error) {
  const auto terminal_observation =
      config_.enable_observability ? observation_now_ns() : 0;
  void* state = nullptr;
  bool retained = true;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (closed(job->status)) return;
    waiting_.erase(std::remove(waiting_.begin(), waiting_.end(), job), waiting_.end());
    running_.erase(std::remove(running_.begin(), running_.end(), job), running_.end());
    // All fallible metadata work precedes raw-state ownership transfer.
    // Failed error text can be dropped; terminal status must remain visible.
    try {
#ifdef HERMES_ADMISSION_CPU_ONLY
      if (config_.terminal_before_error_copy)
        config_.terminal_before_error_copy();
#endif
      job->error = error;
    } catch (...) {
      job->error.clear();
    }
    retained = retain_finished_locked(job);
    state = job->state;
    job->state = nullptr;
    job->status = status;
    if (config_.enable_observability)
      observe_terminal_locked(*job, status, terminal_observation);
    if (status == JobStatus::completed) ++counters_.completed_total;
    else if (status == JobStatus::cancelled) ++counters_.cancelled_total;
    else if (status == JobStatus::expired) ++counters_.expired_total;
    else if (status == JobStatus::failed) ++counters_.failed_total;
  }
  if (state) backend_.close(state);
  emit_event(job, EventKind::terminal);
  if (!retained) forget_unretained(job);
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

void ContinuousScheduler::promote(std::size_t& remaining_admissions,
                                  SchedulerClock::time_point logical_now,
                                  SchedulerClock::time_point entered_at) {
  for (;;) {
    // A seeded prefill can spend substantial time in a single CUDA call.
    // Re-evaluate the waiting queue deadline before EACH next admission,
    // otherwise a later request may start despite an expired wait deadline.
    expire_waiting(logical_now + (SchedulerClock::now() - entered_at));
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
        if (config_.enable_observability)
          counters_.active_high_water =
              std::max(counters_.active_high_water,
                       static_cast<std::uint64_t>(running_.size()));
      }
    }
    if (cancelled) {
      finish(job, JobStatus::cancelled);
      continue;
    }
    if (config_.enable_observability) observe_queue_admission(job);

    // Explicit opt-in: seed the SAME freshly opened state in a single
    // T-token PP2 pass, before HTTP sees 'admitted'. An attempted prefill
    // may poison/deallocate the GPU state on failure. NEVER replay the prompt
    // or allocate a second state after PrefillResult::failed.
    if (config_.enable_batched_prefill && job->request.prompt.size() <= 64) {
      std::vector<float> last_logits;
      std::string prefill_error;
      PrefillResult result = PrefillResult::failed;
      try {
        result = backend_.prefill(state, job->request.prompt,
                                  last_logits, prefill_error);
      } catch (const std::exception& e) {
        prefill_error = e.what();
      } catch (...) {
        prefill_error = "backend seeded prefill threw an unknown exception";
      }
      {
        std::lock_guard<std::mutex> guard(mutex_);
        ++counters_.prefill_attempts_total;
        if (result == PrefillResult::seeded) {
          ++counters_.prefill_seeded_total;
          counters_.prefill_prompt_tokens_total += job->request.prompt.size();
        } else if (result == PrefillResult::unsupported) {
          ++counters_.prefill_unsupported_total;
        } else {
          ++counters_.prefill_failed_total;
        }
        cancelled = job->cancel_requested || stopping_;
      }
      if (cancelled) {
        finish(job, JobStatus::cancelled);
        continue;
      }
      if (result == PrefillResult::failed) {
        finish(job, JobStatus::failed,
               "PP2 seeded prefill failed: " +
               (prefill_error.empty() ? "invalid or poisoned state" : prefill_error));
        continue;
      }
      if (result == PrefillResult::seeded) {
        // Only the final prompt position yields the first completion token;
        // the T-1 earlier logits must never count toward usage or SSE.
        if (last_logits.size() != static_cast<size_t>(config_.vocab_size)) {
          finish(job, JobStatus::failed, "PP2 seeded prefill invalid final logits size");
          continue;
        }
        int best_id = -1;
        float best = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < last_logits.size(); ++i) {
          if (!std::isfinite(last_logits[i])) { best_id = -1; break; }
          if (last_logits[i] > best) {
            best = last_logits[i];
            best_id = static_cast<int>(i);
          }
        }
        if (best_id < 0) {
          finish(job, JobStatus::failed, "PP2 seeded prefill invalid/nonfinite logits");
          continue;
        }
        {
          std::lock_guard<std::mutex> guard(mutex_);
          job->prompt_consumed = job->request.prompt.size();
        }
        // Guarantee event order: admitted -> token -> optional terminal.
        // An admitted callback may cancel for disconnected/backpressured SSE.
        emit_event(job, EventKind::admitted);
        {
          std::lock_guard<std::mutex> guard(mutex_);
          cancelled = job->cancel_requested || stopping_;
          if (!cancelled) job->generated.push_back(best_id);
        }
        if (cancelled) {
          finish(job, JobStatus::cancelled);
          continue;
        }
        emit_event(job, EventKind::token, best_id);
        bool done, cancelled_now;
        {
          std::lock_guard<std::mutex> guard(mutex_);
          cancelled_now = job->cancel_requested || stopping_;
          done = job->generated.size() >=
                     static_cast<size_t>(job->request.max_new_tokens) ||
                 best_id == config_.eos_token_id;
        }
        if (done || cancelled_now)
          finish(job, cancelled_now ? JobStatus::cancelled : JobStatus::completed);
        continue;
      }
      // 'unsupported' guarantees no state mutation: normal prompt steps
      // start from position 0 on this same allocated DecodeState.
    }
    emit_event(job, EventKind::admitted);
  }
}

std::size_t ContinuousScheduler::tick(SchedulerClock::time_point now) {
  std::lock_guard<std::mutex> pump(pump_mutex_);
  const auto entered_at = SchedulerClock::now();
  if (config_.enable_ready_completion_dispatch) return tick_ready(now, entered_at);
  if (config_.enable_parallel_steps) return tick_parallel(now, entered_at);
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
    promote(remaining_admissions, now, entered_at);
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
    if (config_.enable_observability) {
      observe_outstanding(1);
      observe_step_begin(*job);
    }
    try {
      ok = backend_.step(job->state, token, position, logits, error);
    } catch (const std::exception& e) {
      error = e.what();
    } catch (...) {
      error = "backend step threw an unknown exception";
    }
    if (config_.enable_observability) observe_step_end(*job);
    ++decoded;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      ++counters_.decode_steps_total;
      if (config_.enable_observability) observe_step_joined_locked(*job);
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
    if (produced_token) {
      if (config_.enable_observability) observe_token_commit(job);
      emit_event(job, EventKind::token, best_id);
    }
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

std::size_t ContinuousScheduler::tick_ready(
    SchedulerClock::time_point now, SchedulerClock::time_point entered_at) {
  // The caller owns the SINGLE pump_mutex_ until ALL started workers have
  // joined. No second scheduler, no detached threads, no callback from a
  // backend worker. Work is limited to 16 total steps and max_active states.
  struct Work {
    std::shared_ptr<Job> job;
    int token = -1;
    long position = -1;
    std::vector<float> logits;
    std::string error;
    bool ok = false;
    bool accounted = false;
    std::thread worker;
  };
  constexpr std::size_t kMaxSteps = 16;
  std::array<Work, kMaxSteps> tasks;
  std::array<std::size_t, kMaxSteps> ready_indices{};
  std::mutex ready_mutex;
  std::condition_variable ready_cv;
  std::size_t ready_published = 0;
  std::size_t ready_consumed = 0;
  std::size_t launched = 0;
  std::size_t reaped = 0;
  std::size_t outstanding = 0;
  std::size_t remaining_admissions = 16;
  bool dispatch_failed = false;
  // Construct before any thread launch. The unexpected-error recovery is
  // noexcept and must not allocate another error string under memory stress.
  const std::string internal_error =
      "ready completion pump internal exception: scheduler stopped";
  // Captures a job selected under mutex_ but not yet added to tasks[] if
  // some unexpected pump-side allocation/callback operation throws.
  std::shared_ptr<Job> pending_launch;
  const auto effective_now = [&] {
    return now + (SchedulerClock::now() - entered_at);
  };

  // A never-throwing safety net for an unexpected exception in the pump
  // (e.g. allocation during an SSE callback). All CUDA worker threads must
  // be joined BEFORE pump_mutex_ can be released and shutdown can free
  // their DecodeStates. Normal completion joins each worker individually
  // as its ready record is removed, before ANY commit/close/relaunch.
  struct JoinRemaining {
    std::array<Work, kMaxSteps>& tasks;
    std::size_t& launched;
    ~JoinRemaining() {
      for (std::size_t i = 0; i < launched; ++i)
        if (tasks[i].worker.joinable()) tasks[i].worker.join();
    }
  } join_remaining{tasks, launched};

  // Unexpected pump exceptions must NEVER strand an in-flight marker (which
  // would make all future ticks skip that state), nor release GPU state
  // before every host worker has returned. Treat the scheduler as poisoned,
  // join EVERYTHING first, then fail/close ALL remaining jobs. This does not
  // catch a genuine fatal process OOM in backend/CUDA, but no unsafe retry
  // or continuation is ever attempted after this host-side failure.
  auto fail_closed_after_exception = [&]() noexcept {
    for (std::size_t i = 0; i < launched; ++i)
      if (tasks[i].worker.joinable()) tasks[i].worker.join();
    constexpr std::size_t kMaxJobs = 16 + 32 + 2;
    std::array<std::shared_ptr<Job>, kMaxJobs> affected;
    std::size_t count = 0;
    try {
      std::lock_guard<std::mutex> guard(mutex_);
      stopping_ = true;
      for (std::size_t i = 0; i < launched; ++i) {
        if (!tasks[i].accounted) {
          ++counters_.decode_steps_total;
          ++counters_.parallel_dispatched_steps_total;
          if (config_.enable_observability)
            observe_step_joined_locked(*tasks[i].job);
          tasks[i].accounted = true;
        }
      }
      for (const auto& item : jobs_) {
        const auto& job = item.second;
        if (closed(job->status)) continue;
        job->ready_step_in_flight = false;
        // Admission is bounded to max_active <=16 + queue<=32; one
        // opening request may additionally be temporarily outside queues.
        if (count < affected.size()) affected[count++] = job;
      }
      if (pending_launch && !closed(pending_launch->status)) {
        const bool found = std::any_of(affected.begin(), affected.begin()+count,
            [&](const std::shared_ptr<Job>& j) { return j == pending_launch; });
        if (!found && count < affected.size()) {
          pending_launch->ready_step_in_flight = false;
          affected[count++] = pending_launch;
        }
      }
    } catch (...) {
      // An unrecoverable metadata lock/allocation failure still cannot free
      // a running worker early, because all workers have already joined.
      return;
    }
    for (std::size_t i = 0; i < count; ++i) {
      try { finish(affected[i], JobStatus::failed, internal_error); }
      catch (...) {
        // Do not retry corrupted state or reopen ingress after OOM.
        // stopping_ remains latched; higher-level shutdown is required.
      }
    }
  };

  try {
  while (true) {
    bool stopping;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      stopping = stopping_;
    }
    if (!stopping) {
      expire_waiting(effective_now());
      // Existing PP2 decode_init and RAW seed allocations can synchronize
      // default streams. Admit only when NO backend step is in flight.
      // This conservative boundary preserves the existing async ownership
      // contract; fast active requests can still advance during slow peers.
      if (outstanding == 0 && launched < config_.steps_per_tick)
        promote(remaining_admissions, now, entered_at);
    }

    while (!stopping && !dispatch_failed &&
           launched < config_.steps_per_tick &&
           outstanding < config_.max_active) {
      std::shared_ptr<Job> job;
      std::vector<std::shared_ptr<Job>> idle_cancelled;
      int token = -1;
      long position = -1;
      {
        std::lock_guard<std::mutex> guard(mutex_);
        if (stopping_) {
          stopping = true;
        } else {
          // Preserve FIFO admission and round-robin LAUNCH fairness.
          // Skip states with an unfinished prior step. Never enqueue
          // two threads with the same request state/absolute position.
          const std::size_t candidates = running_.size();
          for (std::size_t n = 0; n < candidates; ++n) {
            auto candidate = running_.front();
            running_.pop_front();
            running_.push_back(candidate);
            if (candidate->ready_step_in_flight) continue;
            if (candidate->cancel_requested) {
              idle_cancelled.push_back(candidate);
              continue;
            }
            job = candidate;
            job->ready_step_in_flight = true;
            pending_launch = job;
            if (job->prompt_consumed < job->request.prompt.size()) {
              position = static_cast<long>(job->prompt_consumed);
              token = job->request.prompt[job->prompt_consumed];
            } else {
              position = static_cast<long>(
                  job->request.prompt.size() + job->generated.size() - 1);
              token = job->generated.back();
            }
            break;
          }
        }
      }
      for (const auto& cancelled : idle_cancelled)
        finish(cancelled, JobStatus::cancelled);
      if (!job) break;

      bool was_cancelled;
      {
        std::lock_guard<std::mutex> guard(mutex_);
        was_cancelled = stopping_ || job->cancel_requested;
        if (was_cancelled) job->ready_step_in_flight = false;
      }
      if (was_cancelled) {
        finish(job, JobStatus::cancelled);
        pending_launch.reset();
        continue;
      }

      // Every worker owns ONE stable array element. No task-vector
      // reallocation, no unbounded queue allocation or thread detach.
      const std::size_t index = launched;
      Work& work = tasks[index];
      work.job = job;
      work.token = token;
      work.position = position;
      try {
#ifdef HERMES_ADMISSION_CPU_ONLY
        // Pure CPU deterministic injection for partial std::thread creation
        // failure; completely absent from production CUDA builds.
        if (config_.ready_before_worker_spawn)
          config_.ready_before_worker_spawn(launched);
#endif
        work.worker = std::thread([this, &tasks, &ready_mutex, &ready_cv,
                                   &ready_indices, &ready_published, index] {
          auto& own = tasks[index];
          if (config_.enable_observability) observe_step_begin(*own.job);
          try {
            own.ok = backend_.step(own.job->state, own.token, own.position,
                                   own.logits, own.error);
          } catch (const std::exception& e) {
            // An allocation exception while formatting the error must not
            // escape a std::thread and call std::terminate. The pump will
            // still fail this request if the message cannot be retained.
            try { own.error = e.what(); } catch (...) { own.error.clear(); }
          } catch (...) {
            try { own.error = "backend ready step threw unknown exception"; }
            catch (...) { own.error.clear(); }
          }
          if (config_.enable_observability) observe_step_end(*own.job);
          // The queue is bounded by steps_per_tick <= 16. A worker only
          // publishes its index after writing the result. The CV mutex
          // and a subsequent join establish happens-before for the pump.
          {
            std::lock_guard<std::mutex> guard(ready_mutex);
            ready_indices[ready_published++] = index;
          }
          ready_cv.notify_one();
        });
      } catch (const std::exception& e) {
        {
          std::lock_guard<std::mutex> guard(mutex_);
          job->ready_step_in_flight = false;
        }
        finish(job, JobStatus::failed,
               std::string("ready host worker creation failed: ") + e.what());
        dispatch_failed = true; // never silently retry a poisoned admission
        pending_launch.reset();
        break;
      } catch (...) {
        {
          std::lock_guard<std::mutex> guard(mutex_);
          job->ready_step_in_flight = false;
        }
        finish(job, JobStatus::failed,
               "ready host worker creation failed: unknown exception");
        dispatch_failed = true;
        pending_launch.reset();
        break;
      }
      if (outstanding == 0) {
        std::lock_guard<std::mutex> guard(mutex_);
        ++counters_.parallel_waves_total;
      }
      ++launched;
      ++outstanding;
      if (config_.enable_observability)
        observe_outstanding(outstanding);
      pending_launch.reset();
    }

    if (outstanding == 0) break;

    // Wait only for THE NEXT finished worker, not the full launched group.
    // A fast client's SSE result can now be delivered while a slow sibling
    // is still inside backend_.step. Producer threads never touch scheduler
    // counters, callbacks, request positions, or DecodeState lifetime.
    std::size_t index;
    {
      std::unique_lock<std::mutex> guard(ready_mutex);
      // Even if every in-flight backend is slow, recheck waiting-request
      // deadlines and the shutdown flag at bounded 25ms intervals. This
      // cannot preempt CUDA, but queued work can expire without waiting
      // for an unrelated long decode_step to finish.
      if (!ready_cv.wait_for(guard, std::chrono::milliseconds(25),
                             [&] { return ready_consumed < ready_published; }))
        continue;
      index = ready_indices[ready_consumed++];
    }
    Work& work = tasks[index];
    if (work.worker.joinable()) work.worker.join();
    --outstanding;
    ++reaped;
#ifdef HERMES_ADMISSION_CPU_ONLY
    // Deterministic CPU-only fault after OWN worker joined but before job
    // commit. The catch joins remaining workers and fails all active jobs.
    if (config_.ready_before_commit)
      config_.ready_before_commit(reaped - 1);
#endif

    bool cancelled;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      work.job->ready_step_in_flight = false;
      ++counters_.decode_steps_total;
      ++counters_.parallel_dispatched_steps_total;
      if (config_.enable_observability)
        observe_step_joined_locked(*work.job);
      work.accounted = true;
      cancelled = stopping_ || work.job->cancel_requested;
    }
    if (cancelled) {
      finish(work.job, JobStatus::cancelled);
      continue;
    }
    if (!work.ok ||
        work.logits.size() != static_cast<std::size_t>(config_.vocab_size)) {
      finish(work.job, JobStatus::failed,
             "PP2 ready decode failed: " +
             (work.error.empty() ? "wrong logits size or empty output" : work.error));
      continue;
    }
    int best_id = -1;
    float best = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < work.logits.size(); ++i) {
      if (!std::isfinite(work.logits[i])) {
        best_id = -1;
        work.error = "non-finite logit";
        break;
      }
      if (work.logits[i] > best) {
        best = work.logits[i];
        best_id = static_cast<int>(i);
      }
    }
    if (best_id < 0) {
      finish(work.job, JobStatus::failed, "invalid ready logits: " + work.error);
      continue;
    }

    bool done = false, produced_token = false;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (work.job->prompt_consumed < work.job->request.prompt.size()) {
        ++work.job->prompt_consumed;
        if (work.job->prompt_consumed == work.job->request.prompt.size()) {
          work.job->generated.push_back(best_id);
          produced_token = true;
        }
      } else {
        work.job->generated.push_back(best_id);
        produced_token = true;
      }
      done = !work.job->generated.empty() &&
             (work.job->generated.back() == config_.eos_token_id ||
              work.job->generated.size() >=
                  static_cast<std::size_t>(work.job->request.max_new_tokens));
    }
    if (produced_token) {
      if (config_.enable_observability) observe_token_commit(work.job);
      emit_event(work.job, EventKind::token, best_id);
    }
    bool cancelled_after_event;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      cancelled_after_event = stopping_ || work.job->cancel_requested;
    }
    if (done || cancelled_after_event)
      finish(work.job, cancelled_after_event ? JobStatus::cancelled
                                            : JobStatus::completed);
  }
  } catch (...) {
    fail_closed_after_exception();
  }
  // All launched worker threads were reaped individually. Shutdown can
  // safely acquire pump_mutex_ after this return and close surviving states.
  return reaped;
}

std::size_t ContinuousScheduler::tick_parallel(
    SchedulerClock::time_point now, SchedulerClock::time_point entered_at) {
  // The caller owns pump_mutex_, so no second tick() can touch these states.
  // Only backend_.step() runs on helper threads: state bookkeeping, all
  // callbacks, token selection and GPU state release belong to this pump.
  const auto effective_now = [&]() {
    return now + (SchedulerClock::now() - entered_at);
  };
  struct Work {
    std::shared_ptr<Job> job;
    int token = -1;
    long position = -1;
    std::vector<float> logits;
    std::string error;
    bool ok = false;
    bool launched = false;
  };
  std::size_t decoded = 0;
  std::size_t remaining_admissions = 16;
  while (decoded < config_.steps_per_tick) {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (stopping_) break;
    }
    expire_waiting(effective_now());
    // Allocations/seeded prefill stay serial and always precede wave launch.
    // A failed/partly seeded state is never dispatched for decoding.
    promote(remaining_admissions, now, entered_at);
    std::vector<Work> wave;
    std::vector<std::shared_ptr<Job>> cancelled_before_launch;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (stopping_) break;
      // Exactly ONE step per distinct state in the wave. Rotating the deque
      // by the selected count preserves the same FIFO/round-robin priority
      // as the serial scheduler across successive waves.
      const std::size_t selected = std::min(
          {running_.size(), config_.max_active, config_.steps_per_tick - decoded});
      wave.reserve(selected);
      for (std::size_t i = 0; i < selected; ++i) {
        auto job = running_.front();
        running_.pop_front();
        running_.push_back(job);
        if (job->cancel_requested) {
          cancelled_before_launch.push_back(std::move(job));
          continue;
        }
        Work work;
        work.job = job;
        if (job->prompt_consumed < job->request.prompt.size()) {
          work.position = static_cast<long>(job->prompt_consumed);
          work.token = job->request.prompt[job->prompt_consumed];
        } else {
          // A seeded prefill commits the first sampled completion token
          // before a job reaches this branch.
          work.position = static_cast<long>(
              job->request.prompt.size() + job->generated.size() - 1);
          work.token = job->generated.back();
        }
        wave.push_back(std::move(work));
      }
    }
    for (const auto& job : cancelled_before_launch)
      finish(job, JobStatus::cancelled);
    if (wave.empty()) {
      // If all selected jobs were cancelled, they were removed by finish().
      // Re-evaluate the next ready group, never spin on an unchanged state.
      if (cancelled_before_launch.empty()) break;
      continue;
    }

    // At most max_active threads; each job/state appears only once in wave.
    // After the join barrier, all logs/events are committed in selection
    // order, not in nondeterministic CUDA completion order.
    std::vector<std::thread> workers;
    bool dispatch_failed = false;
    std::string dispatch_error;
    try {
      workers.reserve(wave.size());
      for (std::size_t i = 0; i < wave.size(); ++i) {
        bool cancelled;
        {
          std::lock_guard<std::mutex> guard(mutex_);
          cancelled = stopping_ || wave[i].job->cancel_requested;
        }
        if (cancelled) continue;
        workers.emplace_back([this, &wave, i]() {
          auto& work = wave[i];
          if (config_.enable_observability) observe_step_begin(*work.job);
          try {
            work.ok = backend_.step(work.job->state, work.token, work.position,
                                    work.logits, work.error);
          } catch (const std::exception& e) {
            work.error = e.what();
          } catch (...) {
            work.error = "backend parallel step threw an unknown exception";
          }
          if (config_.enable_observability) observe_step_end(*work.job);
        });
        wave[i].launched = true;
      }
    } catch (const std::exception& e) {
      dispatch_failed = true;
      dispatch_error = e.what();
    } catch (...) {
      dispatch_failed = true;
      dispatch_error = "unknown host thread creation exception";
    }
    // Never allow a GPU state to be closed or reused while its host thread
    // may still be enqueueing/reading CUDA work, even after dispatch failure.
    for (auto& worker : workers) if (worker.joinable()) worker.join();
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (!workers.empty()) ++counters_.parallel_waves_total;
      counters_.parallel_dispatched_steps_total += workers.size();
      counters_.decode_steps_total += workers.size();
      if (config_.enable_observability) {
        counters_.outstanding_step_high_water =
            std::max(counters_.outstanding_step_high_water,
                     static_cast<std::uint64_t>(workers.size()));
        for (const auto& work : wave)
          if (work.launched) observe_step_joined_locked(*work.job);
      }
    }
    decoded += workers.size();
    for (auto& work : wave) {
      bool cancelled;
      {
        std::lock_guard<std::mutex> guard(mutex_);
        cancelled = stopping_ || work.job->cancel_requested;
      }
      if (cancelled) {
        finish(work.job, JobStatus::cancelled);
        continue;
      }
      if (!work.launched) {
        if (dispatch_failed) {
          finish(work.job, JobStatus::failed,
                 "parallel host dispatch failed before step: " + dispatch_error);
        }
        // Otherwise cancellation was detected before launch and finish()
        // will be requested on the next branch/tick.
        continue;
      }
      if (!work.ok || work.logits.size() != static_cast<std::size_t>(config_.vocab_size)) {
        finish(work.job, JobStatus::failed,
               "PP2 parallel decode failed: " +
               (work.error.empty() ? "wrong logits size or empty output" : work.error));
        continue;
      }
      int best_id = -1;
      float best = -std::numeric_limits<float>::infinity();
      for (std::size_t i = 0; i < work.logits.size(); ++i) {
        if (!std::isfinite(work.logits[i])) {
          best_id = -1;
          work.error = "non-finite logit";
          break;
        }
        if (work.logits[i] > best) {
          best = work.logits[i];
          best_id = static_cast<int>(i);
        }
      }
      if (best_id < 0) {
        finish(work.job, JobStatus::failed, "invalid logits: " + work.error);
        continue;
      }
      bool done = false, produced_token = false;
      {
        std::lock_guard<std::mutex> guard(mutex_);
        if (work.job->prompt_consumed < work.job->request.prompt.size()) {
          ++work.job->prompt_consumed;
          if (work.job->prompt_consumed == work.job->request.prompt.size()) {
            work.job->generated.push_back(best_id);
            produced_token = true;
          }
        } else {
          work.job->generated.push_back(best_id);
          produced_token = true;
        }
        done = !work.job->generated.empty() &&
               (work.job->generated.back() == config_.eos_token_id ||
                work.job->generated.size() >=
                    static_cast<std::size_t>(work.job->request.max_new_tokens));
      }
      if (produced_token) {
        if (config_.enable_observability) observe_token_commit(work.job);
        emit_event(work.job, EventKind::token, best_id);
      }
      // Token callbacks may request cancellation for a full SSE queue.
      // Do not issue another decode step for that request; promptly reap it.
      bool cancelled_after_event;
      {
        std::lock_guard<std::mutex> guard(mutex_);
        cancelled_after_event = stopping_ || work.job->cancel_requested;
      }
      if (done || cancelled_after_event)
        finish(work.job, cancelled_after_event ? JobStatus::cancelled
                                              : JobStatus::completed);
    }
    // A partial host-thread creation failure is a strict per-request
    // terminal failure (not an unsafe fallback to sync decode after some
    // unknown subset has already advanced CUDA state).
    if (dispatch_failed) break;
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

bool PP2StepBackend::supports_parallel_steps() const noexcept {
  // This adapter is provably per-request stream/private-pinned only in
  // async mode. The sync fallback may use g_pinned and the legacy stream;
  // do not advertise it as a throughput-safe parallel backend.
  const char* flag = std::getenv("HERMES_PP2_ASYNC_STREAMS");
  return flag && flag[0] == '1' && flag[1] == '\0';
}

PrefillResult PP2StepBackend::prefill(void* state,
                                     const std::vector<int>& prompt,
                                     std::vector<float>& last_logits,
                                     std::string& error) {
  last_logits.clear();
  if (prompt.empty() || prompt.size() > 64)
    return PrefillResult::unsupported; // untouched; caller may step normally
  if (!state) {
    error = "null PP2 prefill state";
    return PrefillResult::failed;
  }
  auto& ds = *static_cast<pp2::DecodeState*>(state);
  if (!ds.inited || ds.ever_advanced || prompt.size() > static_cast<size_t>(ds.capacity)) {
    error = "PP2 seeded prefill requires an untouched allocated state";
    return PrefillResult::failed;
  }
  pp2::Stats stats;
  const std::vector<float> full =
      pp2::prefill_into_decode_state(prompt, ds, stats, error);
  constexpr size_t VOC = 248320u;
  if (full.size() != prompt.size() * VOC) {
    if (error.empty()) error = "seeded prefill returned incomplete logits";
    return PrefillResult::failed;
  }
  last_logits.assign(full.end() - VOC, full.end());
  return PrefillResult::seeded;
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
