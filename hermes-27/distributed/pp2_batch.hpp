#pragma once
// HERMES-27 P5 batch concorrente PP2 (N richieste, stati separati, overlap pipeline).
// Ri riuso: pesi upload + kernel esistenti via pp2::decode_step (nessuna duplicazione pesi).
// DecodeState attuale = 1 richiesta -> qui BatchState = N DecodeState indipendenti
// (S/CS/KV per request; KV attualmente FP32, 64KiB/tok/req/GPU,
// piu` ~75MiB/req/GPU per stati GDN/conv e workspace).
#include "pp2_fwd.hpp"
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
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
// Optional parallel dispatch uses at most max_active host threads per wave,
// one independent request state per thread, and deterministic pump-side
// token/event commits AFTER joining. No second scheduler owns those states.
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

// admitted: PP2 state is allocated and active; the consumer may now start
// HTTP 200/SSE headers. A request that fails, expires or is cancelled while
// queued emits terminal only, so the HTTP layer can still return 429/503.
// Each successful admission emits exactly one admitted event before any token.
enum class EventKind { token, terminal, admitted };
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
  // Experimental opt-in: use the existing freshly allocated DecodeState
  // for a single T-token PP2 stateful prefill when 1 <= prompt.size() <= 64.
  // Default false preserves the incremental prompt loop exactly.
  bool enable_batched_prefill = false;
  // Strict OPT-IN. Run at most one decode_step per active request per wave,
  // concurrently on distinct backend states. Requires backend capability
  // supports_parallel_steps() and (for PP2) HERMES_PP2_ASYNC_STREAMS=1.
  // Default false retains the original one-step-at-a-time scheduling path.
  bool enable_parallel_steps = false;
  // Independent strict opt-in layered on enable_parallel_steps: commit each
  // request as soon as its own backend step has returned and that worker has
  // been joined, WITHOUT waiting for the slowest sibling in the host wave.
  // Requires a backend advertising supports_parallel_steps(); real PP2 only
  // advertises this under HERMES_PP2_ASYNC_STREAMS=1. Seeded prefill is
  // incompatible with this path. Default false keeps old wave-join behavior.
  bool enable_ready_completion_dispatch = false;
  // Opt-in anonymous, aggregate host-side scheduler timing. Never records
  // prompt text, token IDs, request IDs, or any per-request history; default
  // false costs only a few predictable branches along the scheduling path.
  // No GPU synchronization is performed to collect these CPU wall times.
  bool enable_observability = false;
#ifdef HERMES_ADMISSION_CPU_ONLY
  // CPU-ONLY deterministic std::thread-construction failure injection.
  // Not included in production/CUDA SchedulerConfig builds.
  std::function<void(std::size_t)> ready_before_worker_spawn;
  // CPU-ONLY unexpected pump exception injection AFTER the selected worker
  // was joined but BEFORE its per-job in-flight marker/result is committed.
  // Validates global fail-closed cleanup; absent from production builds.
  std::function<void(std::size_t)> ready_before_commit;
  // CPU-only deterministic allocation failures inside terminalization.
  // These are deliberately called while the metadata lock is held.
  std::function<void()> terminal_before_error_copy;
  std::function<void()> terminal_before_history_append;
  // Simulates allocation failure before the event ID is copied. Unlike the
  // payload hook below, this leaves no safe recipient for a minimal event.
  std::function<void()> terminal_before_event_id_copy;
  // Called after the terminal ID is prepared but before copying its vector.
  std::function<void()> terminal_before_snapshot_copy;
  // CPU-only monotonic, thread-safe, nonblocking clock for deterministic
  // tests. Must not call back into the scheduler or take application locks.
  // Used only with enable_observability=true; production uses steady_clock.
  std::function<SchedulerClock::time_point()> observability_now;
#endif
  // Synchronous, optional event sink: no callbacks under metadata lock.
  // Must be thread-safe: queued cancellation calls it from the canceller,
  // admission, tokens and other terminal events from the pump. May call
  // cancel/inspect/retire/enqueue, but never tick/shutdown from this callback.
  // Keep it nonblocking and bound HTTP/SSE queues; cancel on backpressure.
  std::function<void(const SchedulerEvent&)> on_event;
  // Last-resort allocation-free terminal wakeup when event construction or
  // on_event throws. The ID is borrowed from a live job; copy nothing before
  // setting the consumer's done/error flag and notifying its waiters.
  // Must not throw, call tick/shutdown, or retain the string reference.
  std::function<void(const std::string&, JobStatus)>
      on_terminal_delivery_failure;
};

struct SchedulerMetrics {
  bool stopping = false;
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
  std::size_t parallel_waves_total = 0;
  std::size_t parallel_dispatched_steps_total = 0;
  std::size_t prefill_attempts_total = 0;
  std::size_t prefill_seeded_total = 0;
  std::size_t prefill_unsupported_total = 0;
  std::size_t prefill_failed_total = 0;
  std::size_t prefill_prompt_tokens_total = 0;
  std::size_t callback_errors_total = 0;
  std::size_t terminal_delivery_failures_total = 0;
  std::size_t terminal_history_failures_total = 0;
  // Enabled only by SchedulerConfig::enable_observability. Values are
  // saturating uint64 nanosecond sums/maxima with bounded constant-size
  // storage. A count == 0 means the duration was never sampled.
  bool observability_enabled = false;
  std::uint64_t observed_queue_admissions = 0;
  std::uint64_t queue_wait_ns_total = 0, queue_wait_ns_max = 0;
  std::uint64_t observed_backend_steps = 0;
  std::uint64_t backend_step_ns_total = 0, backend_step_ns_max = 0;
  std::uint64_t observed_token_commits = 0;
  std::uint64_t backend_ready_to_token_commit_ns_total = 0;
  std::uint64_t backend_ready_to_token_commit_ns_max = 0;
  // Terminal latency is measured only for successful completed jobs after a
  // backend step, not for queue cancellation, expiration or backend failure.
  std::uint64_t observed_ready_terminals = 0;
  std::uint64_t ready_terminal_ns_total = 0, ready_terminal_ns_max = 0;
  std::uint64_t observed_wave_terminals = 0;
  std::uint64_t wave_terminal_ns_total = 0, wave_terminal_ns_max = 0;
  std::uint64_t observed_serial_terminals = 0;
  std::uint64_t serial_terminal_ns_total = 0, serial_terminal_ns_max = 0;
  std::uint64_t observed_callbacks = 0;
  std::uint64_t callback_ns_total = 0, callback_ns_max = 0;
  std::uint64_t observed_terminal_callbacks = 0;
  std::uint64_t terminal_callback_ns_total = 0, terminal_callback_ns_max = 0;
  std::uint64_t queued_high_water = 0;
  std::uint64_t active_high_water = 0;
  std::uint64_t outstanding_step_high_water = 0;
};

enum class PrefillResult { unsupported, success, failed, seeded = success };

class StepBackend {
 public:
  virtual ~StepBackend() = default;
  // open owns the capacity-sized KV buffers; state must be non-null on success.
  // A failed open must release its partial allocations before returning.
  virtual bool open(int capacity, void*& state, std::string& error) = 0;
  virtual bool step(void* state, int token, long position,
                    std::vector<float>& logits, std::string& error) = 0;
  // Must be explicitly true before opt-in can dispatch overlapping calls on
  // *distinct* states. Backend must be thread-safe across states; concurrent
  // calls for the SAME state remain forbidden. Default fail-closed.
  virtual bool supports_parallel_steps() const noexcept { return false; }
  // Optional single-pass seeded prefill on THE SAME fresh open() state.
  // Output contains only the final token's VOC logits; state must be ready
  // for continuation at absolute position prompt.size() on 'seeded'.
  // 'unsupported' MUST leave state completely untouched, so the scheduler
  // can execute the legacy token-by-token prompt loop on that same state.
  // 'failed' may leave state poisoned/freed: scheduler NEVER retries or
  // replays, only calls close() and sends terminal failure. close() must be
  // safe when the underlying prefill implementation already freed GPU data.
  virtual PrefillResult prefill(void* state, const std::vector<int>& prompt,
                               std::vector<float>& last_logits,
                               std::string& error) {
    (void)state;
    (void)prompt;
    last_logits.clear();
    error.clear();
    return PrefillResult::unsupported;
  }
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
  void promote(std::size_t& remaining_admissions,
               SchedulerClock::time_point logical_now,
               SchedulerClock::time_point entered_at);
  void emit_event(const std::shared_ptr<Job>& job, EventKind kind,
                  int token_id = -1) noexcept;
  bool closed(JobStatus status) const;
  void trim_finished_locked();
  bool retain_finished_locked(const std::shared_ptr<Job>& job) noexcept;
  void forget_unretained(const std::shared_ptr<Job>& job) noexcept;
  std::int64_t observation_now_ns() const noexcept;
  void observe_step_begin(Job& job) noexcept;
  void observe_step_end(Job& job) noexcept;
  // Must be called with mutex_ held, exactly once for each joined step.
  void observe_step_joined_locked(const Job& job) noexcept;
  void observe_queue_admission(const std::shared_ptr<Job>& job) noexcept;
  void observe_token_commit(const std::shared_ptr<Job>& job) noexcept;
  void observe_terminal_locked(const Job& job, JobStatus status,
                               std::int64_t at_ns) noexcept;
  void observe_callback(EventKind kind, std::int64_t began_ns) noexcept;
  void observe_outstanding(std::size_t outstanding) noexcept;
  // Called with pump_mutex_ held. Never holds mutex_ across backend calls;
  // callbacks and frees are pump-owned and occur only after wave joins.
  std::size_t tick_parallel(SchedulerClock::time_point now,
                            SchedulerClock::time_point entered_at);
  // Same single pump owner; per-tick bounded worker-completion CV queue.
  // Every worker is JOINED before its state is advanced, reused or closed.
  std::size_t tick_ready(SchedulerClock::time_point now,
                         SchedulerClock::time_point entered_at);

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
  PrefillResult prefill(void* state, const std::vector<int>& prompt,
                       std::vector<float>& last_logits,
                       std::string& error) override;
  bool supports_parallel_steps() const noexcept override;
  void close(void* state) noexcept override;
};
#endif

} // namespace pp2batch
