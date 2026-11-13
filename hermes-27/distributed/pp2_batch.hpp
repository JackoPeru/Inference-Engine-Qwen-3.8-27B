#pragma once
// HERMES-27 P5 batch concorrente PP2 (N richieste, stati separati, overlap pipeline).
// Ri riuso: pesi upload + kernel esistenti via pp2::decode_step (nessuna duplicazione pesi).
// DecodeState attuale = 1 richiesta -> qui BatchState = N DecodeState indipendenti
// (S/CS/KV per request; KV attualmente FP32, 64KiB/tok/req/GPU,
// piu` ~75MiB/req/GPU per stati GDN/conv e workspace).
#include "pp2_fwd.hpp"
#include <chrono>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <string>

namespace pp2batch {

struct BatchState {
  int N = 0;
  std::vector<pp2::DecodeState> reqs;
  bool inited = false;
};

// Inizializza N stati (chiama pp2::decode_init per ogni request).
// Ritorna false se OOM o VRAM>15GB/GPU (abort hard come P2/P4).
bool batch_init(BatchState& b, int N, std::string& err);

// Libera tutti gli stati (pesi upload restano residenti).
void batch_free(BatchState& b);

// Step sequenziale onesto (loop richieste con kernel esistenti, v1):
// toks.size()==N, stessa pos per tutte (prompt stessa lunghezza, caso GATE1).
// Ritorna N logits[VOC], vuoto su errore.
std::vector<std::vector<float>> batch_step(const std::vector<int>& toks,
                                           BatchState& b, long pos,
                                           std::string& err);

// Step parallelo con overlap pipeline:
// mentre GPU1 fa seconda meta` di req A, GPU0 fa prima meta` di req B.
// Implementato via host threads concorrenti su pp2::decode_step (device diversi
// eseguono davvero in parallelo; xfer pinned protetto da mutex in pp2_fwd).
// poss.size()==N (pos assoluta per-request, permette prompt di lunghezza diversa).
std::vector<std::vector<float>> batch_step_parallel(const std::vector<int>& toks,
                                                    BatchState& b,
                                                    const std::vector<long>& poss,
                                                    std::string& err);

// Overload stessa pos per tutte (comodo per GATE1/GATE2 con prompt pari lunghezza).
std::vector<std::vector<float>> batch_step_parallel(const std::vector<int>& toks,
                                                    BatchState& b, long pos,
                                                    std::string& err);

// Ammissione statica (conservativa) per PP2 corrente: 64KiB/tok/GPU KV FP32,
// stati GDN/conv + arene/step, N=1..16, Tctx=1..MAX_CONTEXT (8192).
// Budget basato su VRAM steady P3 con 1GiB/GPU riservato; non conferma
// disponibilita` VRAM live e non garantisce throughput per N ammessi.
// Il chiamante deve mantenere il preflight live tramite cudaMemGetInfo/decode_init.
bool admit_check(int N, int Tctx, std::string& detail);

// Same FP32/GDN projection as admit_check, but each request reserves its
// actual context capacity. This is a static envelope, never a live CUDA check.
bool admit_mixed(const std::vector<int>& capacities, std::string& detail);

// VRAM usata per GPU (MiB) via cudaMemGetInfo, per report no-leak.
void vram_used(size_t& used0_MiB, size_t& used1_MiB);

// Bounded, single-pump continuous request scheduler. A request owns one
// independent PP2 DecodeState for its entire lifetime (no GDN/KV sharing).
// Enqueue, cancel, inspect, retire and metrics may run on other host threads.
// Exactly one host thread should call tick(); an internal mutex enforces this.
// Backend calls are made outside the scheduler metadata mutex, so cancel()
// remains responsive while a GPU decode_step is running. Cancellation takes
// effect after the current non-interruptible decode_step returns.
using SchedulerClock = std::chrono::steady_clock;

struct ScheduledRequest {
  std::string id;
  std::vector<int> prompt;
  int max_new_tokens = 1;
};

enum class SubmitCode { accepted, invalid, duplicate, queue_full, stopped, memory_limit };
struct SubmitResult {
  SubmitCode code = SubmitCode::invalid;
  std::string detail;
};

enum class JobStatus { queued, active, completed, cancelled, expired, failed };
struct JobSnapshot {
  JobStatus status = JobStatus::queued;
  std::size_t prompt_consumed = 0;
  std::vector<int> generated;
  int capacity = 0;
  std::string error;
};

enum class EventKind { token, terminal };
struct SchedulerEvent {
  EventKind kind = EventKind::terminal;
  std::string id;
  int token_id = -1;       // valid only when kind==token
  JobStatus status = JobStatus::queued;
  std::size_t prompt_consumed = 0;
  std::vector<int> generated; // populated on terminal events
  std::string error;
};

struct SchedulerConfig {
  std::size_t max_active = 4;          // 1..16, GPU state capacity bound
  std::size_t max_queued = 32;         // 1..32, waiting requests bound
  std::size_t steps_per_tick = 4;      // 1..16, total PP2 decode calls / tick
  std::size_t max_retained = 64;       // 0..1024, bounded terminal snapshots
  std::chrono::milliseconds queue_timeout{60000};
  int max_context = pp2::DecodeState::MAX_CONTEXT;
  int vocab_size = 248320;
  int eos_token_id = -1;              // -1 disables EOS stopping
  // Synchronous, optional event sink: no callbacks under metadata lock.
  // Must be thread-safe: queued cancellation calls it from the canceller,
  // other events from the pump. Never call tick/shutdown from this callback.
  std::function<void(const SchedulerEvent&)> on_event;
};

struct SchedulerMetrics {
  std::size_t active = 0;
  std::size_t queued = 0;
  std::size_t retained = 0;
  std::size_t accepted_total = 0;
  std::size_t rejected_total = 0;
  std::size_t completed_total = 0;
  std::size_t cancelled_total = 0;
  std::size_t expired_total = 0;
  std::size_t failed_total = 0;
  std::size_t decode_steps_total = 0;
  std::size_t callback_errors_total = 0;
};

class StepBackend {
 public:
  virtual ~StepBackend() = default;
  // open owns the capacity-sized KV buffers; state must be non-null on success.
  // A failed open must release its partial allocations before returning.
  virtual bool open(int capacity, void*& state, std::string& error) = 0;
  virtual bool step(void* state, int token, long position,
                    std::vector<float>& logits, std::string& error) = 0;
  virtual void close(void* state) noexcept = 0;
};

class ContinuousScheduler {
 public:
  ContinuousScheduler(SchedulerConfig config, StepBackend& backend);
  ~ContinuousScheduler();
  ContinuousScheduler(const ContinuousScheduler&) = delete;
  ContinuousScheduler& operator=(const ContinuousScheduler&) = delete;

  SubmitResult enqueue(ScheduledRequest request,
                       SchedulerClock::time_point now = SchedulerClock::now());
  bool cancel(const std::string& id);
  bool inspect(const std::string& id, JobSnapshot& out) const;
  bool retire(const std::string& id);  // only terminal jobs
  SchedulerMetrics metrics() const;
  // Returns number of backend decode steps executed (<=steps_per_tick).
  std::size_t tick(SchedulerClock::time_point now = SchedulerClock::now());
  void shutdown();

 private:
  struct Job;
  void finish(const std::shared_ptr<Job>& job, JobStatus status,
              const std::string& error = {});
  void expire_waiting(SchedulerClock::time_point now);
  void promote();
  void emit_event(const std::shared_ptr<Job>& job, EventKind kind, int token_id = -1);
  bool closed(JobStatus status) const;
  void trim_finished_locked();

  SchedulerConfig config_;
  StepBackend& backend_;
  mutable std::mutex mutex_;
  std::mutex pump_mutex_;
  bool stopping_ = false;
  std::deque<std::shared_ptr<Job>> waiting_;
  std::deque<std::shared_ptr<Job>> running_;
  std::deque<std::string> finished_order_;
  std::unordered_map<std::string, std::shared_ptr<Job>> jobs_;
  SchedulerMetrics counters_;
};

#ifndef HERMES_ADMISSION_CPU_ONLY
// Real PP2 adapter. Calls synchronous decode_step, one request per state.
// Live free-VRAM preflight and decode_init both fail closed on allocation.
class PP2StepBackend final : public StepBackend {
 public:
  bool open(int capacity, void*& state, std::string& error) override;
  bool step(void* state, int token, long position,
            std::vector<float>& logits, std::string& error) override;
  void close(void* state) noexcept override;
};
#endif

} // namespace pp2batch
