// P5 batch concorrente PP2: GATE1 correttezza N=2, GATE2 throughput N=2,4.
// Prompt A = baseline P2/P4 11tok [6205,161421,303,220,17,1374,9911,1147,27053,6059,13].
// Prompt B = reverse(A) [13,6059,27053,1147,9911,1374,17,220,303,161421,6205] (stessa lunghezza 11,
// contesto diverso, pos allineate per batch_step stessa pos).
// Prompt C = rotl(A,3), D = rotl(A,6) per N=4 (documentati in P5_BATCH.md).
// Uso: test_pp2_batch [admission|g1|g2|all]
// CPU-only: g++ -DHERMES_ADMISSION_CPU_ONLY -x c++ pp2_batch.cu test_pp2_batch.cu
#include "pp2_batch.hpp"
#include "pp2_fwd.hpp"
#include <cstdio>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#ifndef HERMES_ADMISSION_CPU_ONLY
#include <cuda_runtime.h>
#endif

// Test host-only della stessa admit_check usata da serving, senza CUDA context.
// Questi sono esiti del budget statico, NON cap di throughput misurati su GPU.
static int run_admission_tests(){
  struct Case { int n, ctx; bool expected; };
  const Case checks[]={
    {0,1024,false}, {17,1024,false}, {1,0,false}, {1,8193,false},
    {2,32768,false}, {1,8192,true}, {4,8192,true}, {5,8192,false},
    {4,1024,true}, {16,8192,false}, {1,1,true}
  };
  bool pass=true;
  for(const auto& test : checks){
    std::string detail;
    bool actual=pp2batch::admit_check(test.n,test.ctx,detail);
    bool ok=(actual==test.expected && !detail.empty());
    if(test.n==1 && test.ctx==8192)
      ok=ok && detail.find("64KiB/token/GPU")!=std::string::npos;
    printf("ADMISSION N=%d context=%d got=%s expected=%s %s | %s\n",
           test.n,test.ctx,actual?"ADMIT":"REJECT",test.expected?"ADMIT":"REJECT",
           ok?"PASS":"FAIL",detail.c_str());
    if(!ok) pass=false;
  }
  printf("ADMISSION CPU GATE %s (static memory policy, no GPU benchmark)\n",pass?"PASS":"FAIL");
  return pass?0:1;
}

#ifdef HERMES_ADMISSION_CPU_ONLY
namespace {
struct FakeBackend final : pp2batch::StepBackend {
  struct State { int serial, capacity, next_pos; };
  struct Call { int serial, token; long pos; };
  int attempts = 0, opens = 0, closes = 0, live = 0, peak_live = 0;
  bool fail_next_open = false, fail_next_step = false, wrong_logits = false;
  bool nan_logits = false, throw_step = false;
  bool support_prefill = false, fail_next_prefill = false;
  bool wrong_prefill_logits = false, nan_prefill_logits = false, throw_prefill = false;
  int prefill_calls = 0;
  std::vector<std::vector<int>> prefill_prompts;
  std::vector<Call> calls;
  std::function<void()> during_open;
  std::function<void()> during_step;
  std::function<void()> during_prefill;

  bool open(int capacity, void*& state, std::string& error) override {
    state = nullptr;
    ++attempts;
    if (fail_next_open) {
      fail_next_open = false;
      error = "injected open OOM";
      return false;
    }
    State* s = new State{++opens, capacity, 0};
    state = s;
    ++live;
    peak_live = std::max(peak_live, live);
    if (during_open) during_open();
    return true;
  }
  bool step(void* p, int token, long position,
            std::vector<float>& logits, std::string& error) override {
    auto* s = static_cast<State*>(p);
    calls.push_back({s->serial, token, position});
    if (position != s->next_pos || position < 0 || position >= s->capacity) {
      error = "position not monotonic or exceeds allocated KV capacity";
      return false;
    }
    ++s->next_pos;
    if (during_step) during_step();
    if (throw_step) throw std::runtime_error("injected step exception");
    if (fail_next_step) {
      fail_next_step = false;
      error = "injected decode failure";
      return false;
    }
    logits.assign(wrong_logits ? 3u : 8u, -1.f);
    if (nan_logits) logits[0] = std::numeric_limits<float>::quiet_NaN();
    logits[2] = 1.f; // deterministic greedy token; EOS=7 never selected
    return true;
  }
  pp2batch::PrefillResult prefill(void* p, const std::vector<int>& prompt,
                                 std::vector<float>& last_logits,
                                 std::string& error) override {
    ++prefill_calls;
    prefill_prompts.push_back(prompt);
    if (!support_prefill) return pp2batch::PrefillResult::unsupported;
    auto* s = static_cast<State*>(p);
    if (prompt.empty() || prompt.size() > 64 ||
        prompt.size() > static_cast<size_t>(s->capacity) || s->next_pos != 0) {
      error = "prefill requires fresh state and bounded prompt";
      return pp2batch::PrefillResult::failed;
    }
    // The mock represents the exact one-pass persistent GDN/KV mutation.
    s->next_pos = static_cast<int>(prompt.size());
    if (during_prefill) during_prefill();
    if (throw_prefill) throw std::runtime_error("injected seeded prefill exception");
    if (fail_next_prefill) {
      fail_next_prefill = false;
      error = "injected partially seeded state failure";
      return pp2batch::PrefillResult::failed;
    }
    last_logits.assign(wrong_prefill_logits ? 3u : 8u, -1.f);
    if (nan_prefill_logits) last_logits[0] =
        std::numeric_limits<float>::quiet_NaN();
    last_logits[2] = 1.f;
    return pp2batch::PrefillResult::seeded;
  }
  void close(void* p) noexcept override {
    delete static_cast<State*>(p);
    ++closes;
    --live;
  }
};

// Independent, thread-safe mock: each open() returns one exclusive state,
// while step() records the maximum number of SIMULTANEOUS in-flight calls.
// A serial scheduler cannot pass the overlap requirement below.
struct ParallelProbeBackend final : pp2batch::StepBackend {
  struct State { int serial, capacity, next_pos; };
  struct Call { int serial, token; long position; };
  mutable std::mutex mu;
  std::vector<Call> calls;
  std::atomic<int> inflight{0}, peak_inflight{0}, opens{0}, closes{0};
  std::atomic<int> serial_counter{0};
  std::atomic<int> prefill_calls{0};
  std::atomic<int> fail_serial{0}, nan_serial{0}, throw_serial{0};
  bool permit_parallel = true;
  bool support_seeded = false;
  std::chrono::milliseconds latency{12};
  std::function<void(int)> after_enter;

  bool supports_parallel_steps() const noexcept override { return permit_parallel; }
  bool open(int capacity, void*& state, std::string&) override {
    state = new State{++serial_counter, capacity, 0};
    ++opens;
    return true;
  }
  bool step(void* state, int token, long pos,
            std::vector<float>& logits, std::string& error) override {
    auto* s = static_cast<State*>(state);
    const int current = inflight.fetch_add(1) + 1;
    for (int prior = peak_inflight.load(); current > prior &&
         !peak_inflight.compare_exchange_weak(prior, current);) {}
    if (after_enter) after_enter(s->serial);
    std::this_thread::sleep_for(latency);
    bool valid = pos == s->next_pos && pos >= 0 && pos < s->capacity;
    if (valid) ++s->next_pos;
    {
      std::lock_guard<std::mutex> guard(mu);
      calls.push_back({s->serial, token, pos});
    }
    inflight.fetch_sub(1);
    if (!valid) { error = "duplicate or out-of-order per-request position"; return false; }
    if (throw_serial == s->serial)
      throw std::runtime_error("injected request-local parallel CUDA failure");
    if (fail_serial == s->serial) {
      error = "injected request-local step error";
      return false;
    }
    logits.assign(8, -1.f);
    if (nan_serial == s->serial)
      logits[0] = std::numeric_limits<float>::quiet_NaN();
    logits[2] = 1.f;
    return true;
  }
  pp2batch::PrefillResult prefill(void* state, const std::vector<int>& prompt,
                                 std::vector<float>& last_logits,
                                 std::string& error) override {
    ++prefill_calls;
    if (!support_seeded) return pp2batch::PrefillResult::unsupported;
    auto* s = static_cast<State*>(state);
    if (s->next_pos != 0 || prompt.empty() ||
        prompt.size() > 64 || prompt.size() > static_cast<size_t>(s->capacity)) {
      error = "invalid/advanced parallel-seed state";
      return pp2batch::PrefillResult::failed;
    }
    s->next_pos = static_cast<int>(prompt.size());
    last_logits.assign(8, -1.f);
    last_logits[2] = 1.f;
    return pp2batch::PrefillResult::success;
  }
  void close(void* state) noexcept override {
    delete static_cast<State*>(state);
    ++closes;
  }
};

int checks = 0, failures = 0;
void expect(bool pass, const char* name) {
  ++checks;
  if (!pass) { ++failures; std::printf("SCHEDULER FAIL %s\n", name); }
}
pp2batch::SchedulerConfig config() {
  pp2batch::SchedulerConfig c;
  c.max_active = 2;
  c.max_queued = 3;
  c.steps_per_tick = 1;
  c.max_retained = 8;
  c.vocab_size = 8;
  c.eos_token_id = 7;
  c.max_context = 12;
  c.queue_timeout = std::chrono::milliseconds(1000);
  return c;
}
bool has_status(pp2batch::ContinuousScheduler& s, const std::string& id,
                pp2batch::JobStatus status, size_t n_gen = 0) {
  pp2batch::JobSnapshot snapshot;
  return s.inspect(id, snapshot) && snapshot.status == status &&
         snapshot.generated.size() == n_gen;
}

int run_scheduler_tests() {
  using namespace pp2batch;
  using namespace std::chrono;
  // RED gate: a scheduler that merely rotates jobs serially inside tick()
  // cannot pass overlap N4, even if CUDA streams exist underneath.
  // Exact terminal, token and usage ordering must match the serial oracle.
  {
    auto execute = [&](bool parallel) {
      ParallelProbeBackend backend;
      auto cfg = config();
      cfg.max_active = 4;
      cfg.max_queued = 4;
      cfg.steps_per_tick = 4;
      cfg.queue_timeout = seconds(60);
      cfg.enable_parallel_steps = parallel;
      std::vector<std::string> events;
      cfg.on_event = [&](const SchedulerEvent& e) {
        events.push_back(e.id + (e.kind == EventKind::admitted ? ":admitted" :
                                 e.kind == EventKind::token ? ":token" : ":terminal"));
      };
      ContinuousScheduler scheduler(cfg, backend);
      for (int i = 0; i < 4; ++i)
        expect(scheduler.enqueue({"P" + std::to_string(i), {i + 1}, 2}).code ==
               SubmitCode::accepted, "parallel N4 enqueues");
      std::size_t steps = scheduler.tick();
      steps += scheduler.tick();
      std::vector<JobSnapshot> outputs(4);
      for (int i = 0; i < 4; ++i)
        expect(scheduler.inspect("P" + std::to_string(i), outputs[i]) &&
               outputs[i].status == JobStatus::completed,
               "parallel N4 all requests complete");
      expect(steps == 8 && scheduler.metrics().decode_steps_total == 8 &&
             backend.opens == 4 && backend.closes == 4 && backend.inflight == 0,
             "parallel N4 state lifecycle / exact eight steps");
      return std::make_tuple(events, outputs, backend.peak_inflight.load());
    };
    auto serial = execute(false);
    auto parallel = execute(true);
    expect(std::get<0>(serial) == std::get<0>(parallel),
           "parallel N4 admitted/token/terminal order identical to sequential");
    for (int i = 0; i < 4; ++i)
      expect(std::get<1>(serial)[i].prompt_consumed ==
                 std::get<1>(parallel)[i].prompt_consumed &&
             std::get<1>(serial)[i].generated == std::get<1>(parallel)[i].generated &&
             std::get<1>(parallel)[i].generated == std::vector<int>({2, 2}),
             "parallel N4 prompt/usage/output bit-exact to sequential oracle");
    expect(std::get<2>(serial) == 1 && std::get<2>(parallel) >= 2 &&
           std::get<2>(parallel) <= 4,
           "parallel N4 genuinely overlaps independent backend steps");
  }
  {
    ParallelProbeBackend backend;
    backend.permit_parallel = false;
    auto cfg = config();
    cfg.enable_parallel_steps = true;
    bool rejected = false;
    try { ContinuousScheduler incompatible(cfg, backend); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "parallel opt-in rejects backend without thread-safety contract");
  }
  // N8 must remain bounded at four concurrently resident states and a
  // four-thread wave. The second FIFO cohort starts only after release.
  {
    ParallelProbeBackend backend;
    auto cfg = config();
    cfg.enable_parallel_steps = true;
    cfg.max_active = 4;
    cfg.max_queued = 8;
    cfg.steps_per_tick = 4;
    cfg.queue_timeout = seconds(60);
    std::vector<std::string> admitted;
    cfg.on_event = [&](const SchedulerEvent& event) {
      if (event.kind == EventKind::admitted)
        admitted.push_back(event.id);
    };
    ContinuousScheduler scheduler(cfg, backend);
    for (int i = 0; i < 8; ++i)
      expect(scheduler.enqueue({"bound-P" + std::to_string(i), {(i % 7) + 1}, 1}).code ==
                 SubmitCode::accepted,
             "parallel N8 bounded queue enqueue");
    expect(scheduler.tick() == 4 && scheduler.metrics().queued == 4 &&
           scheduler.metrics().active == 0,
           "parallel N8 only first FIFO cohort executes during first tick");
    expect(scheduler.tick() == 4 && scheduler.metrics().queued == 0 &&
           scheduler.metrics().completed_total == 8,
           "parallel N8 second cohort admitted when first frees states");
    expect(backend.peak_inflight == 4 && backend.opens == 8 &&
           backend.closes == 8 && backend.inflight == 0 &&
           scheduler.metrics().parallel_dispatched_steps_total == 8,
           "parallel N8 maximum threads bounded by max_active four");
    expect(admitted == std::vector<std::string>({
             "bound-P0", "bound-P1", "bound-P2", "bound-P3",
             "bound-P4", "bound-P5", "bound-P6", "bound-P7"}),
           "parallel N8 strict FIFO admission across active waves");
  }
  // EOS and callback-driven backpressure are terminal after one sampled
  // output; a cancelled job is never sent another step in a later wave.
  {
    ParallelProbeBackend backend;
    auto cfg = config();
    cfg.enable_parallel_steps = true;
    cfg.max_active = 4;
    cfg.max_queued = 4;
    cfg.steps_per_tick = 4;
    cfg.queue_timeout = seconds(60);
    cfg.eos_token_id = 2;
    ContinuousScheduler scheduler(cfg, backend);
    for (int i = 0; i < 4; ++i)
      scheduler.enqueue({"eos-P" + std::to_string(i), {i + 1}, 8});
    expect(scheduler.tick() == 4 && scheduler.metrics().completed_total == 4 &&
           backend.opens == 4 && backend.closes == 4,
           "parallel EOS sampled once per request, no duplicate GPU steps");
    for (int i = 0; i < 4; ++i)
      expect(has_status(scheduler, "eos-P" + std::to_string(i),
                        JobStatus::completed, 1),
             "parallel EOS exact one completion token per request");
  }
  {
    ParallelProbeBackend backend;
    auto cfg = config();
    cfg.enable_parallel_steps = true;
    cfg.max_active = 4;
    cfg.max_queued = 4;
    cfg.steps_per_tick = 4;
    cfg.queue_timeout = seconds(60);
    std::vector<std::string> terminals;
    ContinuousScheduler* ptr = nullptr;
    cfg.on_event = [&](const SchedulerEvent& event) {
      if (event.kind == EventKind::token && event.id == "backpressure-P1")
        expect(ptr->cancel(event.id), "parallel SSE backpressure cancel from callback");
      if (event.kind == EventKind::terminal) {
        terminals.push_back(event.id);
        expect(ptr->retire(event.id), "parallel terminal callback reentrantly retires");
      }
    };
    ContinuousScheduler scheduler(cfg, backend);
    ptr = &scheduler;
    for (int i = 0; i < 4; ++i)
      scheduler.enqueue({"backpressure-P" + std::to_string(i), {i + 1}, 2});
    expect(scheduler.tick() == 4 &&
           scheduler.metrics().cancelled_total == 1 &&
           scheduler.metrics().active == 3,
           "parallel backpressure cancels only its request in first wave");
    expect(scheduler.tick() == 3 &&
           scheduler.metrics().completed_total == 3 &&
           backend.opens == 4 && backend.closes == 4,
           "parallel cancelled request gets no second decode step");
    JobSnapshot removed;
    expect(!scheduler.inspect("backpressure-P1", removed) &&
           terminals == std::vector<std::string>({
             "backpressure-P1", "backpressure-P0",
             "backpressure-P2", "backpressure-P3"}),
           "parallel SSE exact terminal ordering and safe retire callback");
  }
  // Two simultaneous tick() callers are serialized by pump_mutex_. A
  // second host caller cannot submit position 0 again on any live state.
  {
    ParallelProbeBackend backend;
    auto cfg = config();
    cfg.enable_parallel_steps = true;
    cfg.max_active = 4;
    cfg.max_queued = 4;
    cfg.steps_per_tick = 4;
    cfg.queue_timeout = seconds(60);
    ContinuousScheduler scheduler(cfg, backend);
    for (int i = 0; i < 4; ++i)
      scheduler.enqueue({"dual-tick-P" + std::to_string(i), {i + 1}, 2});
    std::atomic<std::size_t> total{0};
    std::thread a([&] { total += scheduler.tick(); });
    std::thread b([&] { total += scheduler.tick(); });
    a.join();
    b.join();
    expect(total == 8 && scheduler.metrics().completed_total == 4 &&
           scheduler.metrics().failed_total == 0 &&
           backend.opens == 4 && backend.closes == 4,
           "parallel simultaneous tick callers cannot replay same position");
    std::size_t first_steps = 0, second_steps = 0;
    for (const auto& call : backend.calls) {
      first_steps += call.position == 0;
      second_steps += call.position == 1;
    }
    expect(first_steps == 4 && second_steps == 4,
           "parallel duplicate tick prevented by one pump mutex");
  }
  // Both opt-in flags may coexist without duplicate state seeding. Compare
  // mixed T1/11/64/65 to the serial seeded reference, including fallback.
  {
    auto execute = [&](bool parallel) {
      ParallelProbeBackend backend;
      backend.support_seeded = true;
      backend.latency = milliseconds(2);
      auto cfg = config();
      cfg.enable_parallel_steps = parallel;
      cfg.enable_batched_prefill = true;
      cfg.max_active = 4;
      cfg.max_queued = 4;
      cfg.max_context = 128;
      cfg.steps_per_tick = 4;
      cfg.queue_timeout = seconds(60);
      std::vector<std::string> events;
      cfg.on_event = [&](const SchedulerEvent& e) {
        events.push_back(e.id + (e.kind == EventKind::admitted ? ":admitted" :
                                 e.kind == EventKind::token ? ":token" : ":terminal"));
      };
      ContinuousScheduler scheduler(cfg, backend);
      for (int t : {1, 11, 64, 65})
        scheduler.enqueue({"mixed-" + std::to_string(t),
                           std::vector<int>(static_cast<size_t>(t), 1), 2});
      std::size_t total = 0;
      for (int i = 0; i < 40 && scheduler.metrics().completed_total != 4; ++i)
        total += scheduler.tick();
      std::vector<JobSnapshot> jobs(4);
      for (std::size_t i = 0; i < jobs.size(); ++i) {
        const int t = (i == 0 ? 1 : i == 1 ? 11 : i == 2 ? 64 : 65);
        expect(scheduler.inspect("mixed-" + std::to_string(t), jobs[i]) &&
               jobs[i].status == JobStatus::completed &&
               jobs[i].prompt_consumed == static_cast<size_t>(t) &&
               jobs[i].generated == std::vector<int>({2, 2}),
               "parallel and serial seeded mixed request usage/outputs");
      }
      expect(backend.prefill_calls == 3 &&
             scheduler.metrics().prefill_seeded_total == 3 &&
             scheduler.metrics().prefill_prompt_tokens_total == 76 &&
             total == 69 && backend.opens == 4 && backend.closes == 4,
             "parallel+seeded exact three prefill passes and 69 incremental steps");
      return std::make_tuple(events, jobs, backend.peak_inflight.load());
    };
    auto serial = execute(false);
    auto parallel = execute(true);
    expect(std::get<0>(serial) == std::get<0>(parallel),
           "parallel+seeded N4 admission/token/terminal event order identical");
    for (std::size_t i = 0; i < 4; ++i)
      expect(std::get<1>(serial)[i].prompt_consumed ==
                 std::get<1>(parallel)[i].prompt_consumed &&
             std::get<1>(serial)[i].generated == std::get<1>(parallel)[i].generated,
             "parallel+seeded N4 exact positions and token count parity");
    expect(std::get<2>(parallel) <= 4 && std::get<2>(parallel) >= 2,
           "parallel+seeded bounded concurrent continuation on independent states");
  }
  // A smaller step quantum than active occupancy must preserve round-robin
  // fairness. A second pump must not run on the first pump's live state.
  {
    ParallelProbeBackend backend;
    auto cfg = config();
    cfg.enable_parallel_steps = true;
    cfg.max_active = 4;
    cfg.max_queued = 4;
    cfg.steps_per_tick = 2;
    cfg.queue_timeout = seconds(60);
    std::vector<std::string> tokens;
    cfg.on_event = [&](const SchedulerEvent& event) {
      if (event.kind == EventKind::token)
        tokens.push_back(event.id);
    };
    ContinuousScheduler scheduler(cfg, backend);
    for (int i = 0; i < 4; ++i)
      scheduler.enqueue({"fair-P" + std::to_string(i), {i + 1}, 2});
    for (int i = 0; i < 4; ++i)
      expect(scheduler.tick() == 2, "parallel fairness tick dispatches exact quantum");
    expect(tokens == std::vector<std::string>({
             "fair-P0", "fair-P1", "fair-P2", "fair-P3",
             "fair-P0", "fair-P1", "fair-P2", "fair-P3"}),
           "parallel fairness FIFO round-robin with quantum smaller than active");
    expect(backend.peak_inflight == 2 && backend.opens == backend.closes &&
           scheduler.metrics().parallel_dispatched_steps_total == 8 &&
           scheduler.metrics().parallel_waves_total == 4,
           "parallel fairness thread bound, waves and state lifetime");
  }
  // Partial backend failures must terminate ONLY the bad state, after all
  // workers have joined; no failed state may be retried or sampled.
  for (int mode = 0; mode < 3; ++mode) {
    ParallelProbeBackend backend;
    if (mode == 0) backend.fail_serial = 2;
    if (mode == 1) backend.nan_serial = 2;
    if (mode == 2) backend.throw_serial = 2;
    auto cfg = config();
    cfg.enable_parallel_steps = true;
    cfg.max_active = 4;
    cfg.max_queued = 4;
    cfg.steps_per_tick = 4;
    cfg.queue_timeout = seconds(60);
    std::vector<std::string> terminal;
    cfg.on_event = [&](const SchedulerEvent& e) {
      if (e.kind == EventKind::terminal) terminal.push_back(e.id);
    };
    ContinuousScheduler scheduler(cfg, backend);
    for (int i = 0; i < 4; ++i)
      scheduler.enqueue({"fail-P" + std::to_string(i), {i + 1}, 1});
    expect(scheduler.tick() == 4 &&
           has_status(scheduler, "fail-P1", JobStatus::failed) &&
           scheduler.metrics().failed_total == 1,
           "parallel backend error/nonfinite/throw isolated to one request");
    expect(has_status(scheduler, "fail-P0", JobStatus::completed, 1) &&
           has_status(scheduler, "fail-P2", JobStatus::completed, 1) &&
           has_status(scheduler, "fail-P3", JobStatus::completed, 1) &&
           backend.opens == 4 && backend.closes == 4,
           "parallel sibling requests survive one poisoned state and close all states");
    expect(terminal == std::vector<std::string>({
             "fail-P0", "fail-P1", "fail-P2", "fail-P3"}),
           "parallel terminal event order independent of host worker completion");
  }
  // Cancel in-flight while four states each have an outstanding GPU step.
  // The cancelled output must never leak into SSE or be processed twice.
  {
    ParallelProbeBackend backend;
    backend.latency = milliseconds(1);
    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    bool release = false;
    std::atomic<int> entered{0};
    backend.after_enter = [&](int) {
      entered.fetch_add(1);
      gate_cv.notify_all();
      std::unique_lock<std::mutex> guard(gate_mutex);
      gate_cv.wait(guard, [&] { return release; });
    };
    auto cfg = config();
    cfg.enable_parallel_steps = true;
    cfg.max_active = 4;
    cfg.max_queued = 4;
    cfg.steps_per_tick = 4;
    cfg.queue_timeout = seconds(60);
    std::vector<std::string> token_events, terminal_events;
    cfg.on_event = [&](const SchedulerEvent& event) {
      if (event.kind == EventKind::token) token_events.push_back(event.id);
      if (event.kind == EventKind::terminal) terminal_events.push_back(event.id);
    };
    ContinuousScheduler scheduler(cfg, backend);
    for (int i = 0; i < 4; ++i)
      scheduler.enqueue({"cancel-P" + std::to_string(i), {i + 1}, 2});
    std::atomic<std::size_t> first_steps{99};
    std::thread pump([&] { first_steps = scheduler.tick(); });
    {
      std::unique_lock<std::mutex> guard(gate_mutex);
      const bool all_started = gate_cv.wait_for(
          guard, seconds(2), [&] { return entered.load() == 4; });
      expect(all_started, "parallel cancel test overlaps all four backend calls");
    }
    expect(scheduler.cancel("cancel-P1"),
           "parallel cancelling one in-flight request is responsive");
    {
      std::lock_guard<std::mutex> guard(gate_mutex);
      release = true;
    }
    gate_cv.notify_all();
    pump.join();
    expect(first_steps == 4 &&
           has_status(scheduler, "cancel-P1", JobStatus::cancelled, 0) &&
           scheduler.metrics().active == 3,
           "parallel in-flight cancel discards only selected request logits");
    expect(scheduler.tick() == 3 &&
           scheduler.metrics().completed_total == 3 &&
           scheduler.metrics().cancelled_total == 1 &&
           backend.opens == 4 && backend.closes == 4,
           "parallel cancel continues three independent requests without leaks");
    expect(std::count(token_events.begin(), token_events.end(), "cancel-P1") == 0 &&
           std::count(terminal_events.begin(), terminal_events.end(), "cancel-P1") == 1,
           "parallel cancellation produces zero token and one terminal for cancelled state");
  }
  // Same shutdown ordering as the existing serial gate: immediately reject
  // new requests while a full wave is blocked in backend_.step(), then
  // join, discard any results and close exactly once.
  {
    ParallelProbeBackend backend;
    backend.latency = milliseconds(1);
    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    bool release = false;
    std::atomic<int> entered{0};
    backend.after_enter = [&](int) {
      entered.fetch_add(1);
      gate_cv.notify_all();
      std::unique_lock<std::mutex> guard(gate_mutex);
      gate_cv.wait(guard, [&] { return release; });
    };
    auto cfg = config();
    cfg.enable_parallel_steps = true;
    cfg.max_active = 4;
    cfg.max_queued = 5;
    cfg.steps_per_tick = 4;
    cfg.queue_timeout = seconds(60);
    std::vector<SchedulerEvent> events;
    cfg.on_event = [&](const SchedulerEvent& event) {
      events.push_back(event);
    };
    ContinuousScheduler scheduler(cfg, backend);
    for (int i = 0; i < 4; ++i)
      scheduler.enqueue({"stop-P" + std::to_string(i), {i + 1}, 3});
    scheduler.enqueue({"stop-wait", {5}, 1});
    std::atomic<std::size_t> first_steps{99};
    std::thread pump([&] { first_steps = scheduler.tick(); });
    {
      std::unique_lock<std::mutex> guard(gate_mutex);
      const bool all_started = gate_cv.wait_for(
          guard, seconds(2), [&] { return entered.load() == 4; });
      expect(all_started, "parallel shutdown test overlaps all active states");
    }
    std::thread stopping([&] { scheduler.shutdown(); });
    bool flagged = false;
    for (int i = 0; i < 2000 && !flagged; ++i) {
      flagged = scheduler.metrics().stopping;
      if (!flagged) std::this_thread::sleep_for(milliseconds(1));
    }
    expect(flagged, "parallel stop flag visible before wave finishes");
    if (flagged)
      expect(scheduler.enqueue({"stop-late", {1}, 1}).code == SubmitCode::stopped,
             "parallel shutdown refuses late enqueue while backend blocked");
    {
      std::lock_guard<std::mutex> guard(gate_mutex);
      release = true;
    }
    gate_cv.notify_all();
    pump.join();
    stopping.join();
    expect(first_steps == 4 && backend.opens == 4 && backend.closes == 4 &&
           scheduler.metrics().active == 0 && scheduler.metrics().queued == 0 &&
           scheduler.metrics().cancelled_total == 5,
           "parallel shutdown waits for four steps, releases all states and queue");
    std::size_t admitted = 0, tokens = 0, terminal = 0;
    for (const auto& event : events) {
      admitted += event.kind == EventKind::admitted;
      tokens += event.kind == EventKind::token;
      terminal += event.kind == EventKind::terminal;
    }
    expect(admitted == 4 && tokens == 0 && terminal == 5 &&
           scheduler.tick() == 0 && backend.inflight == 0,
           "parallel shutdown yields no post-stop token and one terminal per request");
  }
  // Stateful seeded prefill consumes the full prompt on the SAME newly
  // allocated backend state, emits only the LAST prompt logit's greedy token,
  // and resumes the normal decoder at absolute position T (never replays).
  for (int t : {1, 11, 64}) {
    FakeBackend backend;
    backend.support_prefill = true;
    auto cfg = config();
    cfg.enable_batched_prefill = true;
    cfg.max_context = 128;
    cfg.steps_per_tick = 8;
    std::vector<SchedulerEvent> events;
    cfg.on_event = [&](const SchedulerEvent& e) { events.push_back(e); };
    ContinuousScheduler s(cfg, backend);
    std::vector<int> prompt(static_cast<size_t>(t), 3);
    const std::string id = "seeded-" + std::to_string(t);
    expect(s.enqueue({id, prompt, 3}).code == SubmitCode::accepted,
           "seeded short prompt accepted");
    expect(s.tick() == 2, "seeded pass followed by exactly two decode steps");
    JobSnapshot snapshot;
    expect(s.inspect(id, snapshot) && snapshot.status == JobStatus::completed &&
           snapshot.prompt_consumed == static_cast<size_t>(t) &&
           snapshot.generated == std::vector<int>({2, 2, 2}),
           "seeded completion usage matches incremental token accounting");
    expect(backend.prefill_calls == 1 &&
           backend.prefill_prompts == std::vector<std::vector<int>>({prompt}) &&
           backend.calls.size() == 2 && backend.calls[0].pos == t &&
           backend.calls[1].pos == t + 1,
           "seeded cache continuation starts at T without double-seed");
    expect(backend.opens == 1 && backend.closes == 1 && backend.live == 0,
           "seeded request uses exactly one allocated/deallocated state");
    auto m = s.metrics();
    expect(m.prefill_attempts_total == 1 && m.prefill_seeded_total == 1 &&
           m.prefill_prompt_tokens_total == static_cast<size_t>(t) &&
           m.decode_steps_total == 2 && m.prefill_failed_total == 0,
           "seeded metrics distinguish prompt prefill from decode steps");
    expect(events.size() == 5 &&
           events[0].kind == EventKind::admitted &&
           events[0].prompt_consumed == static_cast<size_t>(t) &&
           events[1].kind == EventKind::token && events[1].token_id == 2 &&
           events[2].kind == EventKind::token &&
           events[3].kind == EventKind::token &&
           events[4].kind == EventKind::terminal &&
           events[4].status == JobStatus::completed &&
           events[4].generated == std::vector<int>({2, 2, 2}),
           "seeded events admitted/token/token/token/terminal exact order");
  }
  {
    FakeBackend backend;
    backend.support_prefill = true;
    auto cfg = config();
    cfg.enable_batched_prefill = true;
    cfg.max_context = 128;
    cfg.eos_token_id = 2;
    std::vector<EventKind> kinds;
    cfg.on_event = [&](const SchedulerEvent& e) { kinds.push_back(e.kind); };
    ContinuousScheduler s(cfg, backend);
    s.enqueue({"seed-eos", std::vector<int>(11, 1), 8});
    expect(s.tick() == 0 && has_status(s, "seed-eos", JobStatus::completed, 1),
           "seeded EOS completes without an extra decode call");
    expect(kinds == std::vector<EventKind>({
             EventKind::admitted, EventKind::token, EventKind::terminal}) &&
           backend.prefill_calls == 1 && backend.calls.empty() && backend.live == 0,
           "seeded first-token EOS callback and release exactly once");
  }
  // Flag-off is byte-for-byte legacy behavior even if backend supports
  // prefill. Unsupported MUST leave state untouched for legal fallback.
  {
    FakeBackend backend;
    backend.support_prefill = true;
    auto cfg = config();
    cfg.steps_per_tick = 4;
    ContinuousScheduler s(cfg, backend);
    s.enqueue({"default-off", {1, 2, 3}, 2});
    expect(s.tick() == 4 && has_status(s, "default-off", JobStatus::completed, 2),
           "default-off consumes all prompt tokens incrementally");
    expect(backend.prefill_calls == 0 && backend.calls.size() == 4 &&
           backend.calls[0].pos == 0 && backend.calls[3].pos == 3,
           "flag-off never touches experimental backend prefill");
  }
  {
    FakeBackend backend; // prefill intentionally unsupported by default
    auto cfg = config();
    cfg.enable_batched_prefill = true;
    cfg.steps_per_tick = 4;
    ContinuousScheduler s(cfg, backend);
    s.enqueue({"unsupported", {1, 2, 3}, 2});
    expect(s.tick() == 4 && has_status(s, "unsupported", JobStatus::completed, 2),
           "unsupported prefill falls back only on untouched same state");
    expect(backend.prefill_calls == 1 && backend.calls.size() == 4 &&
           backend.opens == 1 && backend.closes == 1,
           "unsupported fallback never duplicates state or token");
    expect(s.metrics().prefill_unsupported_total == 1 &&
           s.metrics().prefill_seeded_total == 0,
           "unsupported fallback metrics exact");
  }
  // Prompt T>64 MUST bypass experimental prefill and use normal absolute
  // positions. No backend.prefill() call is allowed for long prompts.
  {
    FakeBackend backend;
    backend.support_prefill = true;
    auto cfg = config();
    cfg.enable_batched_prefill = true;
    cfg.max_context = 128;
    cfg.steps_per_tick = 16;
    ContinuousScheduler s(cfg, backend);
    s.enqueue({"long65", std::vector<int>(65, 1), 1});
    size_t calls = 0;
    for (int i = 0; i < 6; ++i) calls += s.tick();
    expect(calls == 65 && has_status(s, "long65", JobStatus::completed, 1),
           "T65 legacy fallback consumes exactly 65 tokens");
    expect(backend.prefill_calls == 0 && backend.calls.size() == 65 &&
           backend.calls.back().pos == 64 && backend.live == 0,
           "long fallback never invokes prefill or doubles state");
  }
  // Multi-user mixed 1/11/64/65 prompt lengths share weights but never
  // share per-request S/CS/KV; the fourth request must use fallback while
  // the first three resume generated tokens at their own absolute positions.
  {
    FakeBackend backend;
    backend.support_prefill = true;
    auto cfg = config();
    cfg.enable_batched_prefill = true;
    cfg.max_active = 4;
    cfg.max_queued = 4;
    cfg.max_context = 128;
    cfg.steps_per_tick = 4;
    cfg.queue_timeout = hours(1);
    std::vector<SchedulerEvent> events;
    cfg.on_event = [&](const SchedulerEvent& e) { events.push_back(e); };
    ContinuousScheduler s(cfg, backend);
    for (int t : {1, 11, 64, 65})
      expect(s.enqueue({"mix-" + std::to_string(t),
                        std::vector<int>(static_cast<size_t>(t), 1), 2}).code ==
             SubmitCode::accepted, "mixed seeded+fallback request accepted");
    std::size_t decoded = 0;
    for (int i = 0; i < 40; ++i) decoded += s.tick();
    for (int t : {1, 11, 64, 65})
      expect(has_status(s, "mix-" + std::to_string(t),
                        JobStatus::completed, 2),
             "mixed request completed with exactly two sampled output IDs");
    expect(decoded == 69 && backend.calls.size() == 69 &&
           backend.prefill_calls == 3,
           "mixed scheduler saves 1+11+64 prompt decode steps without replay");
    expect(backend.prefill_prompts.size() == 3 &&
           backend.prefill_prompts[0].size() == 1 &&
           backend.prefill_prompts[1].size() == 11 &&
           backend.prefill_prompts[2].size() == 64,
           "only short prompts receive one stateful prefill each");
    expect(backend.opens == 4 && backend.closes == 4 &&
           backend.peak_live == 4 && backend.live == 0,
           "mixed prefill/legacy states each allocated and released once");
    const auto m = s.metrics();
    expect(m.prefill_attempts_total == 3 && m.prefill_seeded_total == 3 &&
           m.prefill_prompt_tokens_total == 76 &&
           m.decode_steps_total == 69 && m.completed_total == 4,
           "mixed prefill metrics report exact input and decode operations");
    std::size_t admitted = 0, tokens = 0, terminal = 0;
    for (const auto& e : events) {
      if (e.kind == EventKind::admitted) ++admitted;
      if (e.kind == EventKind::token) ++tokens;
      if (e.kind == EventKind::terminal) ++terminal;
    }
    expect(admitted == 4 && tokens == 8 && terminal == 4,
           "mixed multiuser delivery exactly once per admission/output/terminal");
  }
  // A partially mutated/failed prefill can never be replayed token-by-token.
  // Failure occurs before admitted, permitting the HTTP layer to send 503.
  for (int mode = 0; mode < 4; ++mode) {
    FakeBackend backend;
    backend.support_prefill = true;
    backend.fail_next_prefill = mode == 0;
    backend.wrong_prefill_logits = mode == 1;
    backend.nan_prefill_logits = mode == 2;
    backend.throw_prefill = mode == 3;
    auto cfg = config();
    cfg.enable_batched_prefill = true;
    std::vector<SchedulerEvent> events;
    cfg.on_event = [&](const SchedulerEvent& e) { events.push_back(e); };
    ContinuousScheduler s(cfg, backend);
    s.enqueue({"bad-seed", {1, 2}, 2});
    expect(s.tick() == 0 && has_status(s, "bad-seed", JobStatus::failed),
           "failed or invalid seeded prefill terminates without replay");
    expect(events.size() == 1 && events[0].kind == EventKind::terminal &&
           events[0].status == JobStatus::failed && !events[0].error.empty(),
           "failed seeded prefill emits single pre-admission terminal");
    expect(backend.prefill_calls == 1 && backend.calls.empty() &&
           backend.opens == 1 && backend.closes == 1 && backend.live == 0,
           "failed seeded prefill closes poisoned state exactly once");
    expect(s.metrics().prefill_failed_total == (mode == 0 || mode == 3 ? 1u : 0u),
           "seeded backend failures counted separately from logits validation");
  }
  // A concurrent disconnect while prefill operates must be visible without
  // holding a metadata mutex across PP2. Discard sampled logits on cancel.
  {
    FakeBackend backend;
    backend.support_prefill = true;
    auto cfg = config();
    cfg.enable_batched_prefill = true;
    std::vector<EventKind> kinds;
    cfg.on_event = [&](const SchedulerEvent& e) { kinds.push_back(e.kind); };
    ContinuousScheduler s(cfg, backend);
    s.enqueue({"cancel-seed", {1, 2, 3}, 3});
    backend.during_prefill = [&] {
      expect(s.cancel("cancel-seed"), "cancel during seeded CUDA pass responsive");
    };
    expect(s.tick() == 0 && has_status(s, "cancel-seed", JobStatus::cancelled),
           "cancelled seed discards output and never starts decode loop");
    expect(kinds == std::vector<EventKind>({EventKind::terminal}) &&
           backend.calls.empty() && backend.live == 0,
           "seed cancellation returns exactly one terminal without SSE headers");
  }
  // SSE admission and token callbacks may cancel a seeded request. Respect
  // transport backpressure before emitting any additional tokens.
  for (int cancel_at = 0; cancel_at < 2; ++cancel_at) {
    FakeBackend backend;
    backend.support_prefill = true;
    auto cfg = config();
    cfg.enable_batched_prefill = true;
    std::vector<EventKind> kinds;
    ContinuousScheduler* scheduler = nullptr;
    cfg.on_event = [&](const SchedulerEvent& e) {
      kinds.push_back(e.kind);
      if ((cancel_at == 0 && e.kind == EventKind::admitted) ||
          (cancel_at == 1 && e.kind == EventKind::token))
        expect(scheduler->cancel(e.id), "SSE seeded callback cancellation");
    };
    ContinuousScheduler s(cfg, backend);
    scheduler = &s;
    s.enqueue({"sse-seed", {1, 2}, 3});
    expect(s.tick() == 0 && has_status(s, "sse-seed", JobStatus::cancelled,
                                        cancel_at == 0 ? 0 : 1),
           "SSE seeded cancel retains only already delivered tokens");
    expect(kinds == (cancel_at == 0 ?
             std::vector<EventKind>({EventKind::admitted, EventKind::terminal}) :
             std::vector<EventKind>({EventKind::admitted, EventKind::token, EventKind::terminal})) &&
           backend.live == 0 && backend.calls.empty(),
           "SSE seeded cancel suppresses all further decode");
  }
  // Queue deadline may expire DURING a seeded prefill of a prior request,
  // in the very same tick and before the next admission/open call.
  {
    FakeBackend backend;
    backend.support_prefill = true;
    backend.during_prefill = [] { std::this_thread::sleep_for(milliseconds(10)); };
    auto cfg = config();
    cfg.enable_batched_prefill = true;
    cfg.max_active = 1;
    cfg.queue_timeout = milliseconds(2);
    ContinuousScheduler s(cfg, backend);
    const auto t0 = SchedulerClock::time_point{};
    s.enqueue({"seed-slow", {1}, 1}, t0);
    s.enqueue({"seed-expired", {1}, 1}, t0);
    expect(s.tick(t0) == 0 && has_status(s, "seed-slow", JobStatus::completed, 1) &&
           has_status(s, "seed-expired", JobStatus::expired),
           "deadline rechecked after slow seeded prefill in same tick");
    expect(backend.prefill_calls == 1 && backend.opens == 1 && backend.live == 0,
           "expired waiting job never gets GPU state after long prefill");
  }
  // Shutdown must flag all active states before waiting for a noninterruptible
  // seeded CUDA pass. No SSE admission/first token may escape after stop.
  {
    FakeBackend backend;
    backend.support_prefill = true;
    auto cfg = config();
    cfg.enable_batched_prefill = true;
    cfg.queue_timeout = seconds(60);
    std::mutex gate_mutex, events_mutex;
    std::condition_variable gate_cv;
    bool entered_prefill = false, release_prefill = false;
    std::vector<SchedulerEvent> events;
    backend.during_prefill = [&] {
      std::unique_lock<std::mutex> lk(gate_mutex);
      entered_prefill = true;
      gate_cv.notify_all();
      gate_cv.wait(lk, [&] { return release_prefill; });
    };
    cfg.on_event = [&](const SchedulerEvent& e) {
      std::lock_guard<std::mutex> lk(events_mutex);
      events.push_back(e);
    };
    ContinuousScheduler s(cfg, backend);
    s.enqueue({"shutdown-seed", {1, 2}, 4});
    s.enqueue({"shutdown-queued", {1, 2}, 2});
    std::atomic<std::size_t> executed{99};
    std::thread pump([&] { executed.store(s.tick()); });
    {
      std::unique_lock<std::mutex> lk(gate_mutex);
      gate_cv.wait(lk, [&] { return entered_prefill; });
    }
    std::thread stopping([&] { s.shutdown(); });
    bool signalled = false;
    const auto deadline = SchedulerClock::now() + seconds(2);
    while (SchedulerClock::now() < deadline) {
      if (s.metrics().stopping) { signalled = true; break; }
      std::this_thread::sleep_for(milliseconds(1));
    }
    expect(signalled, "shutdown visible while seeded prefill is in flight");
    if (signalled)
      expect(s.enqueue({"late-seed", {1}, 2}).code == SubmitCode::stopped,
             "no new admission while seeded prefill is in flight at shutdown");
    {
      std::lock_guard<std::mutex> lk(gate_mutex);
      release_prefill = true;
    }
    gate_cv.notify_all();
    pump.join();
    stopping.join();
    expect(executed.load() == 0 && backend.prefill_calls == 1 &&
           backend.calls.empty() && backend.live == 0 &&
           backend.opens == backend.closes,
           "shutdown drains seeded and queued states without extra decode");
    expect(has_status(s, "shutdown-seed", JobStatus::cancelled) &&
           has_status(s, "shutdown-queued", JobStatus::cancelled) &&
           s.metrics().cancelled_total == 2,
           "shutdown seeded jobs terminal once and states freed");
    expect(events.size() == 2 &&
           events[0].kind == EventKind::terminal &&
           events[1].kind == EventKind::terminal &&
           events[0].status == JobStatus::cancelled &&
           events[1].status == JobStatus::cancelled,
           "shutdown during seeded prefill sends no admitted or token event");
  }

  // Static mixed-context projection is exact for the FP32 per-state layout.
  std::string why;
  expect(admit_mixed({1, 8192, 1}, why) && !why.empty(), "mixed contexts allowed");
  expect(admit_mixed({8192, 8192, 8192, 8192}, why), "four 8K states admitted");
  expect(!admit_mixed({8192, 8192, 8192, 8192, 8192}, why), "five 8K states rejected");
  expect(!admit_mixed({}, why), "empty mixed context rejected");
  expect(!admit_mixed({0}, why), "zero context rejected");
  expect(!admit_mixed({8193}, why), "too long context rejected");
  expect(!admit_mixed(std::vector<int>(17, 1), why), "too many states rejected");

  // Invalid contexts must fail before any backend allocation, and a full
  // waiting queue must reject immediately with no unbounded memory growth.
  {
    FakeBackend backend;
    auto cfg = config();
    cfg.max_context = 4;
    cfg.max_queued = 2;
    int cancel_events = 0;
    cfg.on_event = [&](const SchedulerEvent& event) {
      if (event.kind == EventKind::terminal &&
          event.status == JobStatus::cancelled) ++cancel_events;
    };
    ContinuousScheduler s(cfg, backend);
    expect(s.enqueue({"", {1}, 1}).code == SubmitCode::invalid, "empty ID");
    expect(s.enqueue({std::string(129, 'x'), {1}, 1}).code ==
           SubmitCode::invalid, "request ID length bounded");
    expect(s.enqueue({"blank", {}, 1}).code == SubmitCode::invalid, "empty prompt");
    expect(s.enqueue({"negative", {-1}, 1}).code == SubmitCode::invalid, "negative token");
    expect(s.enqueue({"vocab", {8}, 1}).code == SubmitCode::invalid, "out of vocab");
    expect(s.enqueue({"zero", {1}, 0}).code == SubmitCode::invalid, "zero generation");
    expect(s.enqueue({"long", {1, 2, 3, 4}, 2}).code == SubmitCode::invalid,
           "prompt+generation over capacity");
    expect(s.enqueue({"huge", {1}, std::numeric_limits<int>::max()}).code ==
           SubmitCode::invalid, "huge generation overflow");
    expect(s.enqueue({"a", {1}, 1}).code == SubmitCode::accepted, "queue a");
    expect(s.enqueue({"a", {1}, 1}).code == SubmitCode::duplicate, "duplicate ID");
    expect(s.enqueue({"b", {1}, 1}).code == SubmitCode::accepted, "queue b");
    expect(s.enqueue({"c", {1}, 1}).code == SubmitCode::queue_full, "bounded waiting");
    expect(s.metrics().queued == 2 && s.metrics().active == 0 &&
           backend.opens == 0, "no implicit GPU initialization");
    expect(s.cancel("b") && has_status(s, "b", JobStatus::cancelled),
           "queued cancellation immediate");
    expect(cancel_events == 1, "queued cancellation event once");
    expect(s.metrics().queued == 1, "cancel releases waiting slot");
    expect(s.enqueue({"c", {1}, 1}).code == SubmitCode::accepted,
           "queue space restored");
    s.shutdown();
    expect(s.metrics().active == 0 && s.metrics().queued == 0 && backend.live == 0,
           "shutdown drains waiting");
    expect(cancel_events == 3, "shutdown sends two terminal callbacks");
    expect(s.enqueue({"d", {1}, 1}).code == SubmitCode::stopped,
           "enqueue after shutdown denied");
  }
  // Round-robin uses one decode_step per request turn. The states have
  // different prompt lengths, KV capacities, positions and generation limits.
  {
    FakeBackend backend;
    auto cfg = config();
    cfg.queue_timeout = seconds(60);
    std::vector<SchedulerEvent> events;
    cfg.on_event = [&](const SchedulerEvent& event) { events.push_back(event); };
    ContinuousScheduler s(cfg, backend);
    const auto start = SchedulerClock::time_point{};
    expect(s.enqueue({"A", {1, 3, 5}, 2}, start).code == SubmitCode::accepted, "fair A");
    expect(s.enqueue({"B", {4}, 2}, start).code == SubmitCode::accepted, "fair B");
    expect(s.enqueue({"C", {6}, 1}, start).code == SubmitCode::accepted, "fair C");
    int executed = 0;
    for (int i = 0; i < 8; ++i) executed += int(s.tick(start));
    const std::vector<int> serials = {1, 2, 1, 2, 1, 3, 1};
    std::vector<int> actual;
    for (const auto& call : backend.calls) actual.push_back(call.serial);
    expect(actual == serials, "fair scheduler interleaves A B C");
    expect(backend.calls.size() == 7 && executed == 7, "seven exact GPU steps");
    const std::vector<long> positions = {0, 0, 1, 1, 2, 0, 3};
    std::vector<long> actual_pos;
    for (const auto& call : backend.calls) actual_pos.push_back(call.pos);
    expect(actual_pos == positions, "separate per-request positions");
    expect(has_status(s, "A", JobStatus::completed, 2), "A completed");
    expect(has_status(s, "B", JobStatus::completed, 2), "B completed");
    expect(has_status(s, "C", JobStatus::completed, 1), "C completed");
    expect(backend.peak_live == 2 && backend.live == 0 &&
           backend.opens == backend.closes, "active GPU state cap and cleanup");
    auto m = s.metrics();
    expect(m.active == 0 && m.queued == 0 && m.completed_total == 3 &&
           m.decode_steps_total == 7, "completed counters and no active leak");
    std::vector<std::string> event_signature;
    for (const auto& event : events)
      event_signature.push_back(event.id + (event.kind == EventKind::token ? ":token" :
                                             event.kind == EventKind::admitted ? ":admitted" : ":end"));
    expect(event_signature == std::vector<std::string>({
           "A:admitted", "B:admitted", "B:token", "B:token", "B:end",
           "C:admitted", "A:token", "C:token", "C:end", "A:token", "A:end"}),
           "admission, token and terminal events in generation order");
    expect(events[4].generated == std::vector<int>({2, 2}) &&
           events[8].generated == std::vector<int>({2}) &&
           events[10].generated == std::vector<int>({2, 2}),
           "terminal callback owns complete generated output");
    expect(events[0].kind == EventKind::admitted &&
           events[0].status == JobStatus::active &&
           events[0].generated.empty() && events[0].token_id == -1,
           "admission signals active GPU state before sampling");
    JobSnapshot removed;
    expect(s.retire("A") && !s.inspect("A", removed), "retire terminal state");
  }
  // EOS is a sampled output after the final prompt step; that output must not
  // cause an extra model step. Single-token EOS finalizes after one step.
  {
    FakeBackend backend;
    auto cfg = config();
    cfg.eos_token_id = 2;
    ContinuousScheduler s(cfg, backend);
    s.enqueue({"eos", {1, 2, 3}, 9});
    expect(s.tick() == 1 && s.tick() == 1 && s.tick() == 1, "EOS prompt processed");
    expect(has_status(s, "eos", JobStatus::completed, 1) &&
           backend.calls.size() == 3 && backend.live == 0, "EOS early stop");
  }
  // Backend can call cancel() synchronously within decode_step: scheduler
  // holds no metadata mutex across CUDA and discards cancelled step output.
  {
    FakeBackend backend;
    auto cfg = config();
    ContinuousScheduler s(cfg, backend);
    s.enqueue({"inflight", {1}, 3});
    backend.during_step = [&]() {
      expect(s.cancel("inflight"), "inflight cancellation responsive");
    };
    expect(s.tick() == 1, "inflight decode step returned");
    backend.during_step = {};
    expect(has_status(s, "inflight", JobStatus::cancelled) &&
           backend.live == 0 && backend.calls.size() == 1, "cancel no partial output");
    expect(!s.cancel("inflight"), "terminal cancel idempotent");
  }
  // A request can be cancelled while PP2 decode_init is still allocating.
  // Admission must not resurrect a cancelled job or leak a newly opened state.
  {
    FakeBackend backend;
    auto cfg = config();
    ContinuousScheduler s(cfg, backend);
    s.enqueue({"opening", {1}, 2});
    backend.during_open = [&]() {
      expect(s.cancel("opening"), "cancellation during state allocation");
    };
    expect(s.tick() == 0, "cancelled opening never decoded");
    backend.during_open = {};
    expect(has_status(s, "opening", JobStatus::cancelled) &&
           backend.live == 0 && backend.opens == backend.closes,
           "cancelled opening state immediately freed");
  }
  // Cancellation from admitted callback is legal. An accepted but cancelled
  // stream emits admitted then terminal; no generated token is produced.
  {
    FakeBackend backend;
    auto cfg = config();
    std::vector<EventKind> received;
    ContinuousScheduler* scheduler = nullptr;
    cfg.on_event = [&](const SchedulerEvent& event) {
      received.push_back(event.kind);
      if (event.kind == EventKind::admitted)
        expect(scheduler->cancel(event.id), "cancel from admitted callback");
    };
    ContinuousScheduler s(cfg, backend);
    scheduler = &s;
    s.enqueue({"cancel-admitted", {1, 2}, 2});
    expect(s.tick() == 0, "cancel after admitted avoids decode");
    expect(received == std::vector<EventKind>({EventKind::admitted, EventKind::terminal}),
           "admitted event precedes cancellation terminal exactly once");
    expect(has_status(s, "cancel-admitted", JobStatus::cancelled) && backend.live == 0,
           "admission callback cancellation releases GPU state");
  }
  // A bounded SSE sink may discover backpressure while handling a token.
  // The reentrant cancellation must suppress any further decode steps and
  // deliver terminal exactly once, without holding a scheduler metadata lock.
  {
    FakeBackend backend;
    auto cfg = config();
    cfg.steps_per_tick = 4;
    ContinuousScheduler* scheduler = nullptr;
    std::vector<EventKind> events;
    cfg.on_event = [&](const SchedulerEvent& event) {
      events.push_back(event.kind);
      if (event.kind == EventKind::token)
        expect(scheduler->cancel(event.id), "SSE backpressure cancels active job");
      if (event.kind == EventKind::terminal)
        expect(scheduler->retire(event.id), "terminal callback may retire job");
    };
    ContinuousScheduler s(cfg, backend);
    scheduler = &s;
    s.enqueue({"bounded-sse", {1}, 4});
    expect(s.tick() == 1 && backend.calls.size() == 1,
           "cancel after first token prevents further decoding in same tick");
    expect(events == std::vector<EventKind>({
           EventKind::admitted, EventKind::token, EventKind::terminal}),
           "SSE cancellation emits admission/token/terminal once");
    JobSnapshot j;
    expect(!s.inspect("bounded-sse", j) && backend.live == 0 &&
           s.metrics().cancelled_total == 1, "terminal callback retire no deadlock");
  }
  // Queue deadline runs even when all active slots are occupied.
  {
    FakeBackend backend;
    auto cfg = config();
    cfg.max_active = 1;
    ContinuousScheduler s(cfg, backend);
    const auto t0 = SchedulerClock::time_point{};
    s.enqueue({"current", {1, 1, 1, 1}, 1}, t0);
    s.enqueue({"wait", {2}, 1}, t0);
    s.tick(t0);
    expect(s.metrics().active == 1 && s.metrics().queued == 1, "one active slot");
    s.tick(t0 + milliseconds(1001));
    expect(has_status(s, "wait", JobStatus::expired) &&
           s.metrics().expired_total == 1, "queued request expires under pressure");
    expect(s.metrics().queued == 0 && backend.opens == 1, "expired never opens state");
  }
  // Stop from a separate host thread while the backend holds a GPU step:
  // new enqueue is rejected immediately, the step returns, and no later
  // tokens or duplicate terminals are delivered. shutdown() waits for safe
  // device-state cleanup, but never holds the metadata mutex across CUDA.
  {
    FakeBackend backend;
    auto cfg = config();
    cfg.max_active = 1;
    cfg.steps_per_tick = 8;
    cfg.queue_timeout = seconds(60);
    std::mutex gate_mutex, events_mutex;
    std::condition_variable gate_cv;
    bool entered_step = false, release_step = false;
    std::vector<std::string> events;
    backend.during_step = [&]() {
      std::unique_lock<std::mutex> lk(gate_mutex);
      entered_step = true;
      gate_cv.notify_all();
      gate_cv.wait(lk, [&] { return release_step; });
    };
    cfg.on_event = [&](const SchedulerEvent& event) {
      std::lock_guard<std::mutex> lk(events_mutex);
      events.push_back(event.id + (event.kind == EventKind::admitted ? ":admitted" :
                                   event.kind == EventKind::token ? ":token" : ":end"));
    };
    ContinuousScheduler s(cfg, backend);
    s.enqueue({"running-stop", {1, 2, 3}, 3});
    s.enqueue({"queued-stop", {4}, 1});
    std::atomic<std::size_t> executed{100};
    std::thread pump([&] { executed.store(s.tick()); });
    {
      std::unique_lock<std::mutex> lk(gate_mutex);
      gate_cv.wait(lk, [&] { return entered_step; });
    }
    std::thread stopping([&] { s.shutdown(); });
    // Wait until shutdown has flagged stop before releasing the fake CUDA step.
    bool signalled = false;
    const auto deadline = SchedulerClock::now() + seconds(2);
    while (SchedulerClock::now() < deadline) {
      if (s.metrics().stopping) { signalled = true; break; }
      std::this_thread::sleep_for(milliseconds(1));
    }
    expect(signalled, "stop visible while backend decode remains blocked");
    if (signalled)
      expect(s.enqueue({"late-stop", {3}, 1}).code == SubmitCode::stopped,
             "new work rejected before in-flight CUDA step returns");
    {
      std::lock_guard<std::mutex> lk(gate_mutex);
      release_step = true;
    }
    gate_cv.notify_all();
    pump.join();
    stopping.join();
    expect(executed.load() == 1 && backend.calls.size() == 1 &&
           backend.live == 0 && backend.opens == backend.closes,
           "shutdown drains active state after exactly one in-flight step");
    expect(has_status(s, "running-stop", JobStatus::cancelled) &&
           has_status(s, "queued-stop", JobStatus::cancelled) &&
           s.metrics().cancelled_total == 2,
           "shutdown cancels running and waiting jobs exactly once");
    expect(events == std::vector<std::string>({
           "running-stop:admitted", "running-stop:end", "queued-stop:end"}),
           "concurrent stop emits admitted then one terminal per job");
    expect(s.tick() == 0 && s.metrics().active == 0 &&
           s.metrics().queued == 0, "stopped scheduler stays drained");
  }
  // The clock also advances within a single tick. A slow backend step must
  // not admit an already expired waiting job on the next iteration.
  {
    FakeBackend backend;
    auto cfg = config();
    cfg.max_active = 1;
    cfg.steps_per_tick = 2;
    cfg.queue_timeout = milliseconds(2);
    ContinuousScheduler s(cfg, backend);
    const auto t0 = SchedulerClock::time_point{};
    s.enqueue({"slow", {1, 2}, 1}, t0);
    s.enqueue({"deadline", {3}, 1}, t0);
    backend.during_step = [] { std::this_thread::sleep_for(milliseconds(10)); };
    expect(s.tick(t0) == 2, "two steps in a slow pump");
    expect(has_status(s, "deadline", JobStatus::expired) &&
           backend.opens == 1 && backend.live == 0,
           "timeout evaluated between steps of one tick");
  }
  // Admission uses the sum of all active FP32 capacities. Five full 8K
  // requests are allowed to queue, but at most four GPU states may open
  // under the conservative budget. A cancelled state frees an admission slot.
  {
    FakeBackend backend;
    auto cfg = config();
    cfg.max_active = 5; // larger than the static FP32 GPU1 memory envelope
    cfg.max_context = 8192;
    cfg.max_queued = 5;
    cfg.queue_timeout = hours(1);
    ContinuousScheduler s(cfg, backend);
    for (int i = 0; i < 5; ++i)
      expect(s.enqueue({"big" + std::to_string(i), {1}, 8192}).code ==
             SubmitCode::accepted, "8K request queued");
    s.tick();
    expect(s.metrics().active == 4 && s.metrics().queued == 1 &&
           backend.opens == 4, "aggregate FP32 memory admission limits slots");
    expect(s.cancel("big0"), "cancel active 8K state");
    for (int i = 0; i < 5; ++i) s.tick();
    expect(s.metrics().active == 4 && s.metrics().queued == 0 &&
           backend.opens == 5 && backend.peak_live == 4,
           "freed state admits waiting 8K request");
    s.shutdown();
    expect(backend.live == 0 && backend.opens == backend.closes,
           "mixed 8K admission no state leak");
  }
  // State allocation and decode failures must not poison later requests.
  {
    FakeBackend backend;
    auto cfg = config();
    cfg.max_active = 1;
    std::vector<std::string> event_order;
    cfg.on_event = [&](const SchedulerEvent& e) {
      event_order.push_back(e.id + (e.kind == EventKind::admitted ? ":admitted" :
                                     e.kind == EventKind::token ? ":token" : ":terminal"));
    };
    ContinuousScheduler s(cfg, backend);
    s.enqueue({"oom", {1}, 1});
    s.enqueue({"next", {2}, 1});
    backend.fail_next_open = true;
    s.tick();
    expect(has_status(s, "oom", JobStatus::failed) &&
           has_status(s, "next", JobStatus::completed, 1), "failed open frees admission");
    expect(event_order == std::vector<std::string>({
             "oom:terminal", "next:admitted", "next:token", "next:terminal"}),
           "failed open has terminal without admitted or token");
    expect(backend.opens == 1 && backend.closes == 1, "open failure no GPU leak");
    backend.nan_logits = true;
    s.enqueue({"nan", {3}, 1});
    s.tick();
    expect(has_status(s, "nan", JobStatus::failed) && backend.live == 0,
           "NaN logits fail closed");
    backend.nan_logits = false;
    backend.wrong_logits = true;
    s.enqueue({"bad-size", {3}, 1});
    s.tick();
    expect(has_status(s, "bad-size", JobStatus::failed) && backend.live == 0,
           "wrong logits size fail closed");
    backend.wrong_logits = false;
    backend.throw_step = true;
    s.enqueue({"throw", {4}, 1});
    s.tick();
    expect(has_status(s, "throw", JobStatus::failed) && backend.live == 0,
           "decode exceptions release GPU state");
  }
  // An HTTP consumer might retry a failed state allocation from on_event.
  // A single tick must bound ALL open attempts, even when every failure
  // immediately injects new queued work.
  {
    FakeBackend backend;
    auto cfg = config();
    cfg.max_active = 1;
    cfg.max_queued = 32;
    cfg.max_retained = 0;
    cfg.queue_timeout = seconds(60);
    ContinuousScheduler* scheduler = nullptr;
    int retries = 0;
    cfg.on_event = [&](const SchedulerEvent& event) {
      if (event.kind != EventKind::terminal || event.status != JobStatus::failed)
        return;
      backend.fail_next_open = true;
      scheduler->enqueue({"retry-" + std::to_string(++retries), {1}, 1});
    };
    ContinuousScheduler s(cfg, backend);
    scheduler = &s;
    backend.fail_next_open = true;
    s.enqueue({"retry-0", {1}, 1});
    expect(s.tick() == 0, "failed allocation retry storm executes no decode");
    expect(backend.attempts == 16 && retries == 16 &&
           s.metrics().queued == 1 && s.metrics().active == 0,
           "single tick bounds failed admission attempts to 16");
    expect(s.metrics().failed_total == 16 && backend.live == 0,
           "retry storm releases all failed state attempts");
    s.shutdown();
  }
  // Completion must reach its consumer without depending on retained history.
  // Throwing consumer callbacks are observable but cannot leak GPU state.
  {
    FakeBackend backend;
    auto cfg = config();
    cfg.max_retained = 0;
    int tokens = 0, terminal = 0, cancelled = 0;
    cfg.on_event = [&](const SchedulerEvent& event) {
      if (event.kind == EventKind::token) ++tokens;
      if (event.kind == EventKind::terminal &&
          event.status == JobStatus::completed &&
          event.generated == std::vector<int>({2})) ++terminal;
      if (event.kind == EventKind::terminal &&
          event.status == JobStatus::cancelled) ++cancelled;
    };
    ContinuousScheduler s(cfg, backend);
    s.enqueue({"zero-history", {1}, 1});
    s.tick();
    JobSnapshot j;
    expect(tokens == 1 && terminal == 1, "callback delivers terminal without history");
    expect(!s.inspect("zero-history", j) && s.metrics().retained == 0,
           "zero terminal retention still bounded");
    expect(backend.live == 0, "zero-history release");
    expect(s.enqueue({"queued-zero", {1}, 1}).code == SubmitCode::accepted,
           "zero-history queued request");
    expect(s.cancel("queued-zero") && cancelled == 1 &&
           !s.inspect("queued-zero", j) && s.metrics().retained == 0,
           "zero-history queued cancellation no use-after-free");
  }
  {
    FakeBackend backend;
    auto cfg = config();
    cfg.on_event = [](const SchedulerEvent&) {
      throw std::runtime_error("injected consumer exception");
    };
    ContinuousScheduler s(cfg, backend);
    s.enqueue({"throws-cb", {1}, 1});
    s.tick();
    expect(has_status(s, "throws-cb", JobStatus::completed, 1) &&
           s.metrics().callback_errors_total == 3 && backend.live == 0,
           "callback exception accounted without model state leak");
  }
  // Terminal snapshots have a configured finite bound, preventing retaining
  // every finished request indefinitely in a long-running server.
  {
    FakeBackend backend;
    auto cfg = config();
    cfg.max_retained = 2;
    ContinuousScheduler s(cfg, backend);
    s.enqueue({"one", {1}, 1});
    s.tick();
    s.enqueue({"two", {1}, 1});
    s.tick();
    s.enqueue({"three", {1}, 1});
    s.tick();
    JobSnapshot j;
    expect(!s.inspect("one", j) && s.inspect("two", j) &&
           s.inspect("three", j), "bounded terminal snapshots");
    expect(s.metrics().retained == 2, "bounded metadata total");
    expect(s.retire("two") && s.metrics().retained == 1, "explicit terminal retire");
  }
  bool threw = false;
  try {
    auto cfg = config();
    cfg.max_active = 0;
    FakeBackend b;
    ContinuousScheduler s(cfg, b);
  } catch (const std::invalid_argument&) { threw = true; }
  expect(threw, "bad scheduler config rejected");
  std::printf("SCHEDULER CPU MOCK %s: %d assertions, %d failures\n",
              failures ? "FAIL" : "PASS", checks, failures);
  return failures ? 1 : 0;
}
} // namespace

int main() {
  const int admission = run_admission_tests();
  const int scheduler = run_scheduler_tests();
  return admission || scheduler ? 1 : 0;
}
#else

static int argmax_v(const std::vector<float>& v){
  int bi=0; float bv=v[0];
  for(size_t i=1;i<v.size();++i) if(v[i]>bv){bv=v[i];bi=(int)i;}
  return bi;
}
static bool finite_all(const std::vector<float>& v){
  for(auto x:v) if(!std::isfinite(x)) return false;
  return true;
}
static void maxerr_vs(const std::vector<float>& a, const std::vector<float>& b,
                      double& mx, size_t& im, double& me){
  mx=0; me=0; im=0;
  for(size_t i=0;i<a.size();++i){
    double e=fabs((double)a[i]-b[i]);
    if(e>mx){mx=e;im=i;} me+=e;
  }
  me/=a.size();
}

// single-stream ref: warmup prompt + 4 greedy, ritorna logits dei 4 step + gen ids
static bool single_ref(const std::vector<int>& prompt,
                       std::vector<std::vector<float>>& ref_logits,
                       std::vector<int>& gen,
                       double& wall_ms, std::string& err){
  ref_logits.clear(); gen.clear();
  pp2::DecodeState ds;
  if(!pp2::decode_init(ds,err)) return false;
  auto t0=std::chrono::high_resolution_clock::now();
  std::vector<float> last;
  for(size_t t=0;t<prompt.size();++t){
    last=pp2::decode_step(prompt[t],ds,(long)t,err);
    if(last.empty()){ pp2::decode_free(ds); return false; }
  }
  int nxt=argmax_v(last);
  for(int k=0;k<4;++k){
    long pos=(long)(prompt.size()+k);
    std::vector<float> dec=pp2::decode_step(nxt,ds,pos,err);
    if(dec.empty()){ pp2::decode_free(ds); return false; }
    ref_logits.push_back(dec);
    gen.push_back(nxt);
    nxt=argmax_v(dec);
  }
  auto t1=std::chrono::high_resolution_clock::now();
  wall_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
  pp2::decode_free(ds);
  return true;
}

int main(int argc,char** argv){
  std::string only=argc>1?argv[1]:"all";
  if(only=="admission") return run_admission_tests();
  printf("P5 PP2 batch N-richeste only=%s\n",only.c_str()); fflush(stdout);
  bool do_g1=(only=="all"||only=="g1"), do_g2=(only=="all"||only=="g2");

  const std::vector<int> PA={6205,161421,303,220,17,1374,9911,1147,27053,6059,13};
  const std::vector<int> PB={13,6059,27053,1147,9911,1374,17,220,303,161421,6205};
  const std::vector<int> PC={220,17,1374,9911,1147,27053,6059,13,6205,161421,303};
  const std::vector<int> PD={9911,1147,27053,6059,13,6205,161421,303,220,17,1374};

  pp2::Stats st; std::string err;
  if(!pp2::upload(st,err)){ printf("UPLOAD FAIL %s\n",err.c_str()); return 1; }
  printf("UPLOAD ok sec=%.2f bytes0=%.3fGiB bytes1=%.3fGiB\n",
    st.upload_sec, st.bytes_gpu0/1024.0/1024/1024, st.bytes_gpu1/1024.0/1024/1024); fflush(stdout);

  size_t u0b=0,u1b=0; pp2batch::vram_used(u0b,u1b);
  printf("VRAM post-upload used0=%zu MiB used1=%zu MiB\n",u0b,u1b); fflush(stdout);

  // ---- GATE1 ----
  if(do_g1){
    printf("\n=== GATE1 N=2 correttezza (seq loop, bit-exact target <1e-4) ===\n"); fflush(stdout);
    std::vector<std::vector<float>> refA, refB;
    std::vector<int> genA, genB;
    double wallA=0, wallB=0;
    if(!single_ref(PA,refA,genA,wallA,err)){ printf("REF A FAIL %s\n",err.c_str()); return 1; }
    if(!single_ref(PB,refB,genB,wallB,err)){ printf("REF B FAIL %s\n",err.c_str()); return 1; }
    printf("REF A gen=[%d,%d,%d,%d] wall15=%.1fms (11 warmup+4 greedy)\n",genA[0],genA[1],genA[2],genA[3],wallA); fflush(stdout);
    printf("REF B gen=[%d,%d,%d,%d] wall15=%.1fms\n",genB[0],genB[1],genB[2],genB[3],wallB); fflush(stdout);
    // baseline single per-tok (15 step wall/15)
    printf("REF single per-step A=%.1fms B=%.1fms (atteso ~654ms)\n",wallA/15.0,wallB/15.0); fflush(stdout);

    pp2batch::BatchState bs;
    if(!pp2batch::batch_init(bs,2,err)){ printf("BATCH_INIT N=2 FAIL %s\n",err.c_str()); return 1; }
    size_t u0a=0,u1a=0; pp2batch::vram_used(u0a,u1a);
    printf("VRAM batch N=2 used0=%zu MiB used1=%zu MiB (delta +%zd/+%zd vs post-upload)\n",
      u0a,u1a,(long)u0a-(long)u0b,(long)u1a-(long)u1b); fflush(stdout);
    if(u0a>15360||u1a>15360){ printf("GATE1 FAIL VRAM>15GB\n"); pp2batch::batch_free(bs); return 1; }

    // warmup 11 batch seq
    for(int t=0;t<11;++t){
      std::vector<int> toks={PA[t],PB[t]};
      auto outs=pp2batch::batch_step(toks,bs,(long)t,err);
      if(outs.empty()){ printf("WARMUP t=%d FAIL %s\n",t,err.c_str()); pp2batch::batch_free(bs); return 1; }
      if(!finite_all(outs[0])||!finite_all(outs[1])){ printf("WARMUP t=%d NaN\n",t); pp2batch::batch_free(bs); return 1; }
    }
    // 4 greedy batch seq, confronto vs ref
    std::vector<int> cur={genA[0],genB[0]}; // placeholder, in realta` tok12 = argmax warmup last
    // Ricalcola tok12 da batch warmup last? Usiamo ref tok12 come starting point per confronto diretto:
    // i ref gen[0] sono i tok12 (argmax dopo warmup). Batch warmup dovrebbe dare stessi tok12.
    // Verifichiamo con uno step extra? No: procediamo con cur = [refA tok12, refB tok12] e pos=11.
    // Se batch warmup diverge, lo step pos11 divergerebbe subito (fail onesto).
    cur[0]=genA[0]; cur[1]=genB[0];
    // Nota: genA[0] e` il token generato a pos11 (primo greedy). Per confronto logits,
    // dobbiamo confrontare batch logits pos11 (generati da tok12) vs refA[0] (stesso input tok12).
    // Ma cur per pos11 deve essere tok12, che e` genA[0]? No: genA[0] e` l'output di pos11,
    // l'input di pos11 e` tok12 = argmax(warmup). In single_ref, nxt iniziale = argmax(warmup),
    // e refA[0] = decode(tok12). Quindi input pos11 = tok12, non genA[0].
    // Dobbiamo recuperare tok12: e` l'input che ha prodotto refA[0]. Lo otteniamo rifacendo warmup single?
    // Semplificazione corretta: rieseguiamo warmup single per ottenere tok12, oppure usiamo
    // il fatto che single_ref warmup last argmax = tok12. Ricaviamo tok12 come: rifacciamo warmup?
    // Piu` semplice e rigoroso: batch warmup last logits -> argmax = tok12_batch, single tok12 = ?
    // Per ora usiamo i tok12 dai ref: dobbiamo salvarli. Ripetiamo warmup single leggero per tok12.
    {
      // Recupera tok12 single (11 warmup, no greedy) per A e B
      pp2::DecodeState dA, dB;
      pp2::decode_init(dA,err); pp2::decode_init(dB,err);
      std::vector<float> lA, lB;
      for(int t=0;t<11;++t){ lA=pp2::decode_step(PA[t],dA,(long)t,err); lB=pp2::decode_step(PB[t],dB,(long)t,err); }
      int tok12A=argmax_v(lA), tok12B=argmax_v(lB);
      pp2::decode_free(dA); pp2::decode_free(dB);
      printf("TOK12 single A=%d B=%d | ref gen0 (pos11 out) A=%d B=%d\n",tok12A,tok12B,genA[0],genB[0]); fflush(stdout);
      cur[0]=tok12A; cur[1]=tok12B;
    }
    bool pass=true;
    std::vector<int> genBatchA, genBatchB;
    auto tg0=std::chrono::high_resolution_clock::now();
    for(int k=0;k<4;++k){
      long pos=11+k;
      // input cur = [tok_pos_A, tok_pos_B]
      auto outs=pp2batch::batch_step(cur,bs,pos,err);
      if(outs.empty()){ printf("BATCH step k=%d pos=%ld FAIL %s\n",k,pos,err.c_str()); pp2batch::batch_free(bs); return 1; }
      double mxA,meA,mxB,meB; size_t imA,imB;
      maxerr_vs(outs[0],refA[k],mxA,imA,meA);
      maxerr_vs(outs[1],refB[k],mxB,imB,meB);
      // Attenzione: refA[k] e` decode(tok_input_k) dove tok_input_0=tok12, tok_input_{k+1}=genA[k].
      // Batch outs[k] usa stesso schema se cur evolve con batch argmax (che deve coincidere).
      // Al primo k input coincide per costruzione; ai successivi coincide solo se arg SAME (verificato).
      int adA=argmax_v(outs[0]), arA=argmax_v(refA[k]);
      int adB=argmax_v(outs[1]), arB=argmax_v(refB[k]);
      printf("GATE1 k=%d pos=%ld A: maxerr=%.6f at %zu mean=%.7f arg batch=%d ref=%d %s | B: maxerr=%.6f at %zu mean=%.7f arg batch=%d ref=%d %s\n",
        k,pos,mxA,imA,meA,adA,arA,adA==arA?"SAME":"DIFF",mxB,imB,meB,adB,arB,adB==arB?"SAME":"DIFF"); fflush(stdout);
      bool okA=(mxA<1e-4 && adA==arA), okB=(mxB<1e-4 && adB==arB);
      printf("GATE1 k=%d A %s B %s (soglia 1e-4 bit-exact)\n",k,okA?"PASS":"FAIL",okB?"PASS":"FAIL"); fflush(stdout);
      if(!okA||!okB) pass=false;
      genBatchA.push_back(adA); genBatchB.push_back(adB);
      // next input = batch argmax (se SAME coincide con ref, confronto resta allineato)
      cur[0]=adA; cur[1]=adB;
      // se DIFF, per onesta` continuiamo con batch path (non forziamo teacher), ma gate gia` FAIL
    }
    auto tg1=std::chrono::high_resolution_clock::now();
    double wall_batch_ms=std::chrono::duration<double,std::milli>(tg1-tg0).count();
    printf("BATCH N=2 4-step wall=%.1fms (8 tok) agg=%.3f tok/s per-req=%.3f tok/s\n",
      wall_batch_ms, 8000.0/wall_batch_ms, 4000.0/wall_batch_ms); fflush(stdout);
    printf("GEN batch A=[%d,%d,%d,%d] vs ref [%d,%d,%d,%d]\n",genBatchA[0],genBatchA[1],genBatchA[2],genBatchA[3],genA[0],genA[1],genA[2],genA[3]); fflush(stdout);
    printf("GEN batch B=[%d,%d,%d,%d] vs ref [%d,%d,%d,%d]\n",genBatchB[0],genBatchB[1],genBatchB[2],genBatchB[3],genB[0],genB[1],genB[2],genB[3]); fflush(stdout);

    // parallel 1-step sanity (stesso input pos11 tok12, deve dare stessi logits del seq)
    {
      pp2batch::BatchState bs2;
      if(!pp2batch::batch_init(bs2,2,err)){ printf("BATCH2 INIT FAIL %s\n",err.c_str()); pp2batch::batch_free(bs); return 1; }
      for(int t=0;t<11;++t){
        std::vector<int> toks={PA[t],PB[t]};
        std::vector<long> poss={(long)t,(long)t};
        auto outs=pp2batch::batch_step_parallel(toks,bs2,poss,err);
        if(outs.empty()){ printf("PAR warmup t=%d FAIL %s\n",t,err.c_str()); pp2batch::batch_free(bs2); pp2batch::batch_free(bs); return 1; }
      }
      // tok12 recuperati sopra? Ricalcola veloce: usa cur iniziale tok12 (pos11 input)
      // Per semplicita` riusa tok12 da prima (cur originario). Ma cur e` evoluto; ricostruisci:
      pp2::DecodeState dA2, dB2;
      pp2::decode_init(dA2,err); pp2::decode_init(dB2,err);
      std::vector<float> lA2,lB2;
      for(int t=0;t<11;++t){ lA2=pp2::decode_step(PA[t],dA2,(long)t,err); lB2=pp2::decode_step(PB[t],dB2,(long)t,err); }
      int t12A=argmax_v(lA2), t12B=argmax_v(lB2);
      pp2::decode_free(dA2); pp2::decode_free(dB2);
      std::vector<int> toks12={t12A,t12B};
      std::vector<long> poss12={11,11};
      auto outs_par=pp2batch::batch_step_parallel(toks12,bs2,poss12,err);
      if(outs_par.empty()){ printf("PAR step pos11 FAIL %s\n",err.c_str()); pp2batch::batch_free(bs2); pp2batch::batch_free(bs); return 1; }
      double mxA,meA,mxB,meB; size_t imA,imB;
      maxerr_vs(outs_par[0],refA[0],mxA,imA,meA);
      maxerr_vs(outs_par[1],refB[0],mxB,imB,meB);
      int adA=argmax_v(outs_par[0]), arA=argmax_v(refA[0]);
      int adB=argmax_v(outs_par[1]), arB=argmax_v(refB[0]);
      printf("PAR sanity pos11 A maxerr=%.6f arg %d vs %d %s | B maxerr=%.6f arg %d vs %d %s\n",
        mxA,adA,arA,adA==arA?"SAME":"DIFF",mxB,adB,arB,adB==arB?"SAME":"DIFF"); fflush(stdout);
      bool okp=(mxA<1e-4&&adA==arA&&mxB<1e-4&&adB==arB);
      printf("PAR sanity %s\n",okp?"PASS":"FAIL"); fflush(stdout);
      if(!okp) pass=false;
      pp2batch::batch_free(bs2);
    }

    size_t u0c=0,u1c=0; pp2batch::vram_used(u0c,u1c);
    printf("VRAM prima free batch used0=%zu MiB used1=%zu MiB (delta vs post-upload %+zd/%+zd)\n",
      u0c,u1c,(long)u0c-(long)u0b,(long)u1c-(long)u1b); fflush(stdout);
    pp2batch::batch_free(bs);
    size_t u0d=0,u1d=0; pp2batch::vram_used(u0d,u1d);
    printf("VRAM dopo free batch used0=%zu MiB used1=%zu MiB (leak check vs post-upload %+zd/%+zd)\n",
      u0d,u1d,(long)u0d-(long)u0b,(long)u1d-(long)u1b); fflush(stdout);
    printf("GATE1 %s\n",pass?"PASS":"FAIL"); fflush(stdout);
    if(only=="g1") return pass?0:1;
    if(!pass) return 1;
  }

  // ---- GATE2 ----
  if(do_g2){
    printf("\n=== GATE2 throughput N=1/2/4 (agg vs baseline 1.53 tok/s) ===\n"); fflush(stdout);
    std::vector<std::vector<int>> prompts={PA,PB,PC,PD};
    const char* pname[4]={"A","B","C","D"};
    for(int N : {1,2,4}){
      for(int mode=0; mode<2; ++mode){
        bool parallel=(mode==1);
        if(N==1 && parallel) continue; // N=1 parallel == seq, skip
        pp2batch::BatchState bs;
        if(!pp2batch::batch_init(bs,N,err)){ printf("G2 INIT N=%d FAIL %s\n",N,err.c_str()); return 2; }
        size_t uv0=0,uv1=0; pp2batch::vram_used(uv0,uv1);
        // warmup 11
        for(int t=0;t<11;++t){
          std::vector<int> toks((size_t)N);
          for(int i=0;i<N;++i) toks[i]=prompts[i][t];
          std::vector<std::vector<float>> outs;
          if(parallel){
            std::vector<long> poss((size_t)N,(long)t);
            outs=pp2batch::batch_step_parallel(toks,bs,poss,err);
          } else {
            outs=pp2batch::batch_step(toks,bs,(long)t,err);
          }
          if(outs.empty()){ printf("G2 warmup N=%d %s t=%d FAIL %s\n",N,parallel?"par":"seq",t,err.c_str()); pp2batch::batch_free(bs); return 2; }
        }
        // tok12 per-request da warmup batch last? Per throughput usiamo greedy batch (self-consistent):
        // prendi ultimo warmup outs argmax come input pos11
        std::vector<int> cur((size_t)N);
        {
          // Per semplicita` e onesta`: un batch_step pos10 gia` fatto; rifacciamo un extra step per tok12?
          // No: usiamo argmax di un passo dedicato? Facciamo: ultimo warmup t=10 ha prodotto logits,
          // ma non salvati. Rifacciamo warmup last argmax via single veloce? Piu` semplice:
          // cur = argmax di un nuovo batch_step con token prompt[10]? No, duplicherebbe pos10.
          // Approccio throughput puro: misura solo 4 greedy step a partire da tok12 single (noto).
          // Calcoliamo tok12 single per ogni prompt (11 decode_step, costo incluso ma non misurato).
          for(int i=0;i<N;++i){
            pp2::DecodeState tmp;
            pp2::decode_init(tmp,err);
            std::vector<float> last;
            for(int t=0;t<11;++t) last=pp2::decode_step(prompts[i][t],tmp,(long)t,err);
            cur[i]=argmax_v(last);
            pp2::decode_free(tmp);
          }
        }
        // Re-inizializza batch per misura pulita (stati zero + warmup 11 dentro misura? No:
        // vogliamo misurare solo decode steady, quindi warmup gia` fatto sopra su bs.
        // Ma cur calcolato da stati single diversi; per coerenza stati, dobbiamo avanzare bs di 0?
        // Soluzione onesta: bs ha gia` stati dopo warmup 11 con prompt tokens.
        // I suoi S/CS/KV corrispondono a prompt, esattamente come i tmp single.
        // Quindi cur (tok12 single) e` valido come input pos11 per bs (se batch warmup corretto,
        // tok12_batch==tok12_single; verifichiamo al primo step con arg check, ma per throughput
        // non confrontiamo logits, solo tempi + finite).
        // Se batch warmup fosse divergente, i tempi restano validi ma la correttezza e` gia` coperta da GATE1.
        auto t0=std::chrono::high_resolution_clock::now();
        bool ok=true;
        for(int k=0;k<4;++k){
          long pos=11+k;
          std::vector<std::vector<float>> outs;
          if(parallel){
            std::vector<long> poss((size_t)N,pos);
            outs=pp2batch::batch_step_parallel(cur,bs,poss,err);
          } else {
            outs=pp2batch::batch_step(cur,bs,pos,err);
          }
          if(outs.empty()){ printf("G2 step N=%d %s k=%d FAIL %s\n",N,parallel?"par":"seq",k,err.c_str()); pp2batch::batch_free(bs); return 2; }
          for(int i=0;i<N;++i){
            if(!finite_all(outs[i])){ printf("G2 NaN N=%d %s k=%d req=%d\n",N,parallel?"par":"seq",k,i); ok=false; }
            cur[i]=argmax_v(outs[i]);
          }
          if(!ok){ pp2batch::batch_free(bs); return 2; }
        }
        auto t1=std::chrono::high_resolution_clock::now();
        double wall_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
        double agg_tps=(N*4)/(wall_ms/1000.0);
        double ms_per_tok_agg=wall_ms/(N*4);
        double speedup_vs_single=agg_tps/1.53;
        size_t uv0b=0,uv1b=0; pp2batch::vram_used(uv0b,uv1b);
        printf("G2 N=%d %s 4-step wall=%.1fmsNt=%d agg=%.3f tok/s (vs 1.53 x%.2f) ms/tok_agg=%.1f vram %zu/%zu MiB prompts=",
          N,parallel?"par":"seq",wall_ms,N*4,agg_tps,speedup_vs_single,ms_per_tok_agg,uv0b,uv1b);
        for(int i=0;i<N;++i) printf("%s%s",pname[i],i+1<N?",":"");
        printf("\n"); fflush(stdout);
        // admission info
        {
          std::string det;
          bool adm=pp2batch::admit_check(N,1024,det);
          printf(" admit_check %s -> %s\n",det.c_str(),adm?"ADMIT":"REJECT"); fflush(stdout);
        }
        pp2batch::batch_free(bs);
        size_t uv0c=0,uv1c=0; pp2batch::vram_used(uv0c,uv1c);
        printf(" G2 N=%d %s post-free vram %zu/%zu MiB (delta vs pre %+zd/%+zd, leak %s)\n",
          N,parallel?"par":"seq",uv0c,uv1c,(long)uv0c-(long)uv0,(long)uv1c-(long)uv1,
          (labs((long)uv0c-(long)uv0)<50 && labs((long)uv1c-(long)uv1)<50)?"NO":"CHECK"); fflush(stdout);
      }
    }
    if(run_admission_tests()!=0) return 2;
    printf("GATE2 DONE (numeri sopra, baseline 1.53 tok/s)\n"); fflush(stdout);
    if(only=="g2") return 0;
  }

  printf("P5 DONE\n"); fflush(stdout);
  return 0;
}
#endif // HERMES_ADMISSION_CPU_ONLY
