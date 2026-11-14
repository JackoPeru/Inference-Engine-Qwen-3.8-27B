// HERMES-27 CPU/OFFLINE. RED first: enable_ready_completion_dispatch must
// deliver a fast request's token WHILE a separate request remains blocked
// in its backend step. No PP2 CUDA, model, service or network is used.
#include "pp2_batch.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
using namespace pp2batch;
using namespace std::chrono;

std::atomic<int> checks{0};
void require(bool ok, const char* msg) {
  ++checks;
  if (!ok) throw std::runtime_error(msg);
}

class Backend final : public StepBackend {
 public:
  struct State {
    int serial;
    int position = 0;
    std::atomic<bool> in_step{false};
  };
  std::atomic<int> opened{0}, closed{0}, inflight{0}, peak{0}, leaked_free{0};
  std::atomic<int> fail_serial{0}, throw_serial{0}, nan_serial{0}, shape_serial{0};
  bool capability = true;
  std::mutex mu;
  std::condition_variable cv;
  bool release_slow = false;
  bool slow_entered = false;
  bool fast_finished = false;
  std::vector<std::pair<int, long>> input_positions;

  bool supports_parallel_steps() const noexcept override { return capability; }
  bool open(int, void*& ptr, std::string&) override {
    ptr = new State{++opened};
    return true;
  }
  bool step(void* ptr, int, long pos, std::vector<float>& logits,
            std::string& error) override {
    auto& state = *static_cast<State*>(ptr);
    if (state.in_step.exchange(true)) {
      error = "duplicate concurrent same-state step";
      return false;
    }
    const int n = inflight.fetch_add(1) + 1;
    for (int observed = peak.load(); n > observed &&
         !peak.compare_exchange_weak(observed, n);) {}
    bool valid = false;
    {
      std::unique_lock<std::mutex> lock(mu);
      input_positions.emplace_back(state.serial, pos);
      if (state.serial == 1 && pos == 0) {
        slow_entered = true;
        cv.notify_all();
        cv.wait(lock, [&] { return release_slow; });
      }
      valid = pos == state.position;
      if (valid) ++state.position;
      if (state.serial != 1) fast_finished = true;
      cv.notify_all();
    }
    inflight.fetch_sub(1);
    state.in_step = false;
    if (!valid) {
      error = "duplicate or skipped position";
      return false;
    }
    if (throw_serial == state.serial) throw std::runtime_error("injected step throw");
    if (fail_serial == state.serial) {
      error = "injected step fail";
      return false;
    }
    logits.assign(shape_serial == state.serial ? 3 : 8, -1.f);
    logits[2] = 1.f;
    if (nan_serial == state.serial) logits[0] = std::numeric_limits<float>::quiet_NaN();
    return true;
  }
  void close(void* ptr) noexcept override {
    auto* state = static_cast<State*>(ptr);
    if (state->in_step.load()) ++leaked_free;
    delete state;
    ++closed;
  }
  bool wait_slow() {
    std::unique_lock<std::mutex> lock(mu);
    return cv.wait_for(lock, seconds(3), [&] { return slow_entered; });
  }
  void release() {
    {
      std::lock_guard<std::mutex> lock(mu);
      release_slow = true;
    }
    cv.notify_all();
  }
};

struct Events {
  std::mutex mu;
  std::condition_variable cv;
  std::vector<std::pair<std::string, EventKind>> observed;
  void add(const SchedulerEvent& ev) {
    {
      std::lock_guard<std::mutex> lock(mu);
      observed.emplace_back(ev.id, ev.kind);
    }
    cv.notify_all();
  }
  bool wait(const std::string& id, EventKind kind, milliseconds budget) {
    std::unique_lock<std::mutex> lock(mu);
    return cv.wait_for(lock, budget, [&] {
      return std::find(observed.begin(), observed.end(),
                       std::make_pair(id, kind)) != observed.end();
    });
  }
  int count(const std::string& id, EventKind kind) {
    std::lock_guard<std::mutex> lock(mu);
    return static_cast<int>(std::count(observed.begin(), observed.end(),
                                       std::make_pair(id, kind)));
  }
};

SchedulerConfig cfg(int active, int quantum, Events& events, bool ready = true) {
  SchedulerConfig c;
  c.vocab_size = 8;
  c.eos_token_id = 7;
  c.max_context = 32;
  c.max_active = static_cast<size_t>(active);
  c.max_queued = 8;
  c.steps_per_tick = static_cast<size_t>(quantum);
  c.queue_timeout = seconds(60);
  c.enable_parallel_steps = true;
  c.enable_ready_completion_dispatch = ready;
  c.on_event = [&](const SchedulerEvent& e) { events.add(e); };
  return c;
}

void assert_completed(ContinuousScheduler& scheduler, const char* id, int n) {
  JobSnapshot s;
  require(scheduler.inspect(id, s), "request must be inspectable");
  require(s.status == JobStatus::completed, "request must have completed");
  require(s.prompt_consumed == 1 && s.generated == std::vector<int>(n, 2),
          "request output/usage/positions");
}

void test_red_fast_before_slow_and_default_unchanged() {
  for (bool ready : {false, true}) {
    Backend backend;
    Events events;
    ContinuousScheduler scheduler(cfg(2, 2, events, ready), backend);
    require(scheduler.enqueue({"A", {1}, 1}).code == SubmitCode::accepted, "A enqueue");
    require(scheduler.enqueue({"B", {2}, 1}).code == SubmitCode::accepted, "B enqueue");
    std::thread pump([&] { scheduler.tick(); });
    const bool slow = backend.wait_slow();
    const bool fast_token_before_release =
        events.wait("B", EventKind::token, milliseconds(600));
    // If RED fails, first release the held worker and join safely.
    backend.release();
    pump.join();
    require(slow, "slow A must have been running");
    require(fast_token_before_release == ready,
            "RED ready B must receive token while A blocked; wave default must not");
    assert_completed(scheduler, "A", 1);
    assert_completed(scheduler, "B", 1);
    require(backend.opened == 2 && backend.closed == 2 && backend.leaked_free == 0,
            "exactly one state per request and safe close");
    require(events.count("A", EventKind::terminal) == 1 &&
            events.count("B", EventKind::terminal) == 1, "one terminal per request");
    std::printf("PASS ready=%d B_token_before_A_release=%d\n",
                ready ? 1 : 0, fast_token_before_release ? 1 : 0);
  }
}

void test_fast_relaunch_while_slow_blocked() {
  Backend backend;
  Events events;
  ContinuousScheduler scheduler(cfg(2, 4, events), backend);
  require(scheduler.enqueue({"A", {1}, 1}).code == SubmitCode::accepted, "slow enqueue");
  require(scheduler.enqueue({"B", {2}, 3}).code == SubmitCode::accepted, "fast enqueue");
  std::thread pump([&] { scheduler.tick(); });
  const bool slow = backend.wait_slow();
  const bool b_done = events.wait("B", EventKind::terminal, milliseconds(1800));
  backend.release();
  pump.join();
  require(slow && b_done, "B must commit 3 tokens and terminal before A returns");
  assert_completed(scheduler, "A", 1);
  assert_completed(scheduler, "B", 3);
  require(backend.peak <= 2 && backend.opened == 2 && backend.closed == 2,
          "bounded N2 workers/one PP2 state per job");
  require(scheduler.metrics().decode_steps_total == 4 &&
          scheduler.metrics().parallel_dispatched_steps_total == 4,
          "four backend steps counted once");
  require(events.count("B", EventKind::token) == 3, "B three ordered tokens");
}

void test_bounded_N8_two_cohorts_and_fifo() {
  Backend backend;
  Events events;
  auto config = cfg(4, 4, events);
  config.max_queued = 8;
  ContinuousScheduler scheduler(config, backend);
  for (int i = 0; i < 8; ++i)
    require(scheduler.enqueue({std::string(1, static_cast<char>('A' + i)),
                                {(i % 7) + 1}, 1}).code ==
                SubmitCode::accepted,
            "N8 enqueued within bounded eight-client waiting queue");
  std::atomic<std::size_t> first{0};
  std::thread pump([&] { first = scheduler.tick(); });
  const bool slow = backend.wait_slow();
  // Ready worker completion order is intentionally nondeterministic across
  // requests. Wait for ALL independent terminals, never assume D is last.
  const bool fast_three_done =
      events.wait("B", EventKind::terminal, milliseconds(1600)) &&
      events.wait("C", EventKind::terminal, milliseconds(1600)) &&
      events.wait("D", EventKind::terminal, milliseconds(1600));
  const auto active = scheduler.metrics();
  backend.release();
  pump.join();
  require(slow && fast_three_done && active.active == 1 &&
          active.queued == 4 && first == 4,
          "N8 fast B/C/D close while A remains in flight, four requests remain FIFO");
  for (const auto* id : {"A", "B", "C", "D"}) assert_completed(scheduler, id, 1);
  require(scheduler.tick() == 4, "N8 second cohort executes with bounded quantum");
  for (const auto* id : {"E", "F", "G", "H"}) assert_completed(scheduler, id, 1);
  require(backend.peak <= 4 && backend.opened == 8 &&
          backend.closed == 8 && backend.leaked_free == 0,
          "N8 cap on simultaneous independent states and safe lifetime");
  std::vector<std::string> admitted;
  {
    std::lock_guard<std::mutex> lock(events.mu);
    for (const auto& event : events.observed)
      if (event.second == EventKind::admitted) admitted.push_back(event.first);
  }
  require(admitted == std::vector<std::string>({"A", "B", "C", "D",
                                                "E", "F", "G", "H"}),
          "N8 admission remains FIFO across completed cohorts");
}

void test_queue_timeout_during_slow_inflight_step() {
  Backend backend;
  Events events;
  auto config = cfg(1, 1, events);
  config.queue_timeout = milliseconds(75);
  ContinuousScheduler scheduler(config, backend);
  require(scheduler.enqueue({"A", {1}, 1}).code == SubmitCode::accepted,
          "queue deadline A accepted");
  require(scheduler.enqueue({"C", {3}, 1}).code == SubmitCode::accepted,
          "queue deadline C accepted");
  std::thread pump([&] { scheduler.tick(); });
  const bool slow = backend.wait_slow();
  const bool expired_before_A_return =
      events.wait("C", EventKind::terminal, milliseconds(1200));
  backend.release();
  pump.join();
  require(slow && expired_before_A_return,
          "C queued deadline expires while A backend still in flight");
  JobSnapshot c;
  require(scheduler.inspect("C", c) && c.status == JobStatus::expired,
          "queued C gets exact expired status without being opened");
  require(backend.opened == 1 && backend.closed == 1 && backend.leaked_free == 0,
          "queued expired C did not allocate/replay an active PP2 state");
}

void test_EOS_completion_is_exactly_one_step() {
  Backend backend;
  Events events;
  auto config = cfg(2, 4, events);
  config.eos_token_id = 2;
  ContinuousScheduler scheduler(config, backend);
  scheduler.enqueue({"A", {1}, 3});
  scheduler.enqueue({"B", {2}, 3});
  std::thread pump([&] { scheduler.tick(); });
  const bool slow = backend.wait_slow();
  const bool b_done = events.wait("B", EventKind::terminal, milliseconds(1200));
  backend.release();
  pump.join();
  require(slow && b_done && scheduler.metrics().decode_steps_total == 2,
          "EOS terminates each state after one sampled token");
  assert_completed(scheduler, "A", 1);
  assert_completed(scheduler, "B", 1);
  require(events.count("A", EventKind::token) == 1 &&
          events.count("B", EventKind::token) == 1,
          "EOS token emitted once per user");
}

void test_two_concurrent_ticks_reuse_one_pump() {
  Backend backend;
  Events events;
  ContinuousScheduler scheduler(cfg(2, 2, events), backend);
  scheduler.enqueue({"A", {1}, 1});
  scheduler.enqueue({"B", {2}, 1});
  std::atomic<std::size_t> sum{0};
  std::thread pump1([&] { sum.fetch_add(scheduler.tick()); });
  const bool slow = backend.wait_slow();
  std::thread pump2([&] { sum.fetch_add(scheduler.tick()); });
  const bool b_done = events.wait("B", EventKind::terminal, milliseconds(1200));
  backend.release();
  pump1.join();
  pump2.join();
  require(slow && b_done && sum == 2 &&
          scheduler.metrics().decode_steps_total == 2,
          "concurrent tick callers never replay same token position");
  require(backend.opened == 2 && backend.closed == 2 && backend.leaked_free == 0,
          "single pump retains one PP2 state per request");
}

void test_cancel_slow_does_not_free_while_backend_owns_state() {
  Backend backend;
  Events events;
  ContinuousScheduler scheduler(cfg(2, 2, events), backend);
  scheduler.enqueue({"A", {1}, 1});
  scheduler.enqueue({"B", {2}, 1});
  std::thread pump([&] { scheduler.tick(); });
  const bool slow = backend.wait_slow();
  const bool b_done = events.wait("B", EventKind::terminal, milliseconds(1500));
  const bool cancelled = scheduler.cancel("A");
  const int pre_close = backend.closed.load();
  backend.release();
  pump.join();
  require(slow && b_done && cancelled, "cancel slow remains responsive after B completes");
  require(pre_close == 1, "only fast B may be closed while A is in flight");
  require(backend.closed == 2 && backend.leaked_free == 0, "A freed only after join");
  JobSnapshot a;
  require(scheduler.inspect("A", a) && a.status == JobStatus::cancelled &&
          a.generated.empty(), "cancelled A has no visible token");
  require(events.count("A", EventKind::token) == 0 &&
          events.count("A", EventKind::terminal) == 1, "cancelled no token exactly terminal");
}

void test_fail_closed_capability_and_flags() {
  Backend backend;
  Events events;
  {
    auto c = cfg(2, 2, events);
    c.enable_parallel_steps = false;
    bool rejected = false;
    try { ContinuousScheduler wrong(c, backend); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "ready requires enable_parallel_steps flag");
  }
  {
    backend.capability = false;
    bool rejected = false;
    try { ContinuousScheduler wrong(cfg(2, 2, events), backend); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "ready requires thread-safe backend capability");
    backend.capability = true;
  }
  {
    auto c = cfg(2, 2, events);
    c.enable_batched_prefill = true;
    bool rejected = false;
    try { ContinuousScheduler wrong(c, backend); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "ready path excludes default-stream seeded prefill");
  }
}

void test_error_isolation() {
  for (int injected = 0; injected < 4; ++injected) {
    Backend backend;
    Events events;
    if (injected == 0) backend.fail_serial = 2;
    if (injected == 1) backend.throw_serial = 2;
    if (injected == 2) backend.nan_serial = 2;
    if (injected == 3) backend.shape_serial = 2;
    ContinuousScheduler scheduler(cfg(2, 2, events), backend);
    scheduler.enqueue({"A", {1}, 1});
    scheduler.enqueue({"B", {2}, 1});
    std::thread pump([&] { scheduler.tick(); });
    const bool slow = backend.wait_slow();
    const bool failed = events.wait("B", EventKind::terminal, milliseconds(1500));
    backend.release();
    pump.join();
    require(slow && failed, "B failure must not wait for slow A");
    assert_completed(scheduler, "A", 1);
    JobSnapshot b;
    require(scheduler.inspect("B", b) && b.status == JobStatus::failed &&
            b.generated.empty(), "failure isolated and no token from poisoned B");
    require(backend.opened == 2 && backend.closed == 2 && backend.leaked_free == 0,
            "isolated failure closes exactly once");
    require(events.count("B", EventKind::terminal) == 1 &&
            events.count("B", EventKind::token) == 0, "failure never sends invalid token");
  }
}

void test_callback_cancel_and_reentrant_retire() {
  Backend backend;
  Events events;
  auto c = cfg(2, 4, events);
  ContinuousScheduler* ptr = nullptr;
  c.on_event = [&](const SchedulerEvent& event) {
    events.add(event);
    if (event.id == "B" && event.kind == EventKind::token)
      require(ptr->cancel(event.id), "callback can cancel its own job");
    if (event.id == "B" && event.kind == EventKind::terminal)
      require(ptr->retire(event.id), "terminal may retire reentrantly");
  };
  ContinuousScheduler scheduler(c, backend);
  ptr = &scheduler;
  scheduler.enqueue({"A", {1}, 1});
  scheduler.enqueue({"B", {2}, 3});
  std::thread pump([&] { scheduler.tick(); });
  const bool slow = backend.wait_slow();
  const bool b_terminal = events.wait("B", EventKind::terminal, milliseconds(1500));
  backend.release();
  pump.join();
  require(slow && b_terminal, "callback cancels before unrelated A finishes");
  require(events.count("B", EventKind::token) == 1 &&
          events.count("B", EventKind::terminal) == 1, "one token then cancellation");
  JobSnapshot removed;
  require(!scheduler.inspect("B", removed), "retire removes terminal B");
  require(backend.opened == 2 && backend.closed == 2 && backend.leaked_free == 0,
          "callback does not double free");
}

void test_throwing_user_callback_is_contained() {
  Backend backend;
  backend.release();
  Events events;
  auto config = cfg(2, 2, events);
  config.on_event = [&](const SchedulerEvent& event) {
    events.add(event);
    if (event.id == "B" && event.kind == EventKind::token)
      throw std::runtime_error("user SSE sink injected failure");
  };
  ContinuousScheduler scheduler(config, backend);
  scheduler.enqueue({"A", {1}, 1});
  scheduler.enqueue({"B", {2}, 1});
  require(scheduler.tick() == 2, "throwing external callback cannot tear down pump");
  require(scheduler.metrics().callback_errors_total == 1 &&
          scheduler.metrics().failed_total == 0,
          "throwing callback accounted, not mistaken for failed backend");
  assert_completed(scheduler, "A", 1);
  assert_completed(scheduler, "B", 1);
  require(backend.opened == 2 && backend.closed == 2 && backend.leaked_free == 0,
          "throwing callback cannot leak state or worker");
}

#ifdef HERMES_ADMISSION_CPU_ONLY
void test_unexpected_pump_bad_alloc_joins_and_fails_closed() {
  Backend backend;
  Events events;
  std::mutex gate_mu;
  std::condition_variable gate_cv;
  bool fault_reached = false;
  auto config = cfg(2, 2, events);
  config.ready_before_commit = [&](std::size_t completed_index) {
    if (completed_index == 0) {
      {
        std::lock_guard<std::mutex> lock(gate_mu);
        fault_reached = true;
      }
      gate_cv.notify_all();
      throw std::bad_alloc();
    }
  };
  ContinuousScheduler scheduler(config, backend);
  scheduler.enqueue({"A", {1}, 1});
  scheduler.enqueue({"B", {2}, 1});
  std::thread pump([&] { scheduler.tick(); });
  const bool slow = backend.wait_slow();
  bool injected = false;
  {
    std::unique_lock<std::mutex> lock(gate_mu);
    injected = gate_cv.wait_for(lock, seconds(3), [&] { return fault_reached; });
  }
  // In the recovery path, A is still in backend_.step; absolutely no
  // close/free is permitted until A returns and the guard joins its worker.
  const bool before_release_safe = backend.leaked_free == 0 && backend.closed == 0;
  backend.release();
  pump.join();
  require(slow && injected && before_release_safe,
          "unexpected allocation error cannot close in-flight A/B states");
  const auto metrics = scheduler.metrics();
  require(metrics.stopping && metrics.active == 0 && metrics.queued == 0 &&
          metrics.failed_total == 2 && metrics.decode_steps_total == 2,
          "bad_alloc recovery joins two workers, fails both and closes scheduler");
  require(scheduler.enqueue({"late", {1}, 1}).code == SubmitCode::stopped,
          "aborted scheduler never resumes unsafe state after exception");
  JobSnapshot a, b;
  require(scheduler.inspect("A", a) && a.status == JobStatus::failed &&
          scheduler.inspect("B", b) && b.status == JobStatus::failed,
          "exception did not strand ready_step_in_flight or replay A/B");
  require(events.count("A", EventKind::token) == 0 &&
          events.count("B", EventKind::token) == 0 &&
          events.count("A", EventKind::terminal) == 1 &&
          events.count("B", EventKind::terminal) == 1,
          "fatal pump failure is all-terminal with zero unverified tokens");
  require(backend.opened == 2 && backend.closed == 2 && backend.leaked_free == 0,
          "fatal pump failure joins before every close and leaks no state");
}
#endif

void test_shutdown_during_blocked_backend() {
  Backend backend;
  Events events;
  ContinuousScheduler scheduler(cfg(2, 2, events), backend);
  scheduler.enqueue({"A", {1}, 1});
  scheduler.enqueue({"B", {2}, 1});
  std::thread pump([&] { scheduler.tick(); });
  const bool slow = backend.wait_slow();
  std::thread stop([&] { scheduler.shutdown(); });
  bool flagged = false;
  for (int tries = 0; tries < 1000 && !flagged; ++tries) {
    flagged = scheduler.metrics().stopping;
    if (!flagged) std::this_thread::sleep_for(milliseconds(1));
  }
  const auto refused = scheduler.enqueue({"late", {1}, 1});
  const bool not_freed = backend.leaked_free == 0 && backend.closed < 2;
  backend.release();
  pump.join();
  stop.join();
  require(slow && flagged && refused.code == SubmitCode::stopped && not_freed,
          "stop blocks new work but does not prematurely release A");
  require(scheduler.metrics().active == 0 && scheduler.metrics().queued == 0 &&
          backend.opened == backend.closed && backend.leaked_free == 0,
          "shutdown drains all outstanding backend calls");
  JobSnapshot a;
  require(scheduler.inspect("A", a) && a.status == JobStatus::cancelled,
          "slow A marked cancelled on shutdown");
}

#ifdef HERMES_ADMISSION_CPU_ONLY
void test_partial_thread_creation_fail_closed() {
  Backend backend;
  backend.release(); // inject a worker-start failure without hanging A
  Events events;
  auto c = cfg(2, 2, events);
  c.ready_before_worker_spawn = [](std::size_t started) {
    if (started == 1) throw std::runtime_error("injected second std::thread failure");
  };
  ContinuousScheduler scheduler(c, backend);
  scheduler.enqueue({"A", {1}, 1});
  scheduler.enqueue({"B", {2}, 1});
  const std::size_t launched = scheduler.tick();
  require(launched == 1 && scheduler.metrics().decode_steps_total == 1,
          "partial creation counts only actually started backend calls");
  require(backend.opened == 2 && backend.closed == 2 && backend.leaked_free == 0,
          "all affected states closed after successful worker join");
  JobSnapshot a, b;
  require(scheduler.inspect("A", a) && a.status == JobStatus::completed,
          "started A may complete successfully");
  require(scheduler.inspect("B", b) && b.status == JobStatus::failed &&
          b.generated.empty(), "unstarted B is failed, never replayed");
}
#endif
} // namespace

int main() {
  try {
    test_fail_closed_capability_and_flags();
    test_red_fast_before_slow_and_default_unchanged();
    test_fast_relaunch_while_slow_blocked();
    test_bounded_N8_two_cohorts_and_fifo();
    test_queue_timeout_during_slow_inflight_step();
    test_EOS_completion_is_exactly_one_step();
    test_two_concurrent_ticks_reuse_one_pump();
    test_cancel_slow_does_not_free_while_backend_owns_state();
    test_error_isolation();
    test_callback_cancel_and_reentrant_retire();
    test_throwing_user_callback_is_contained();
    test_shutdown_during_blocked_backend();
#ifdef HERMES_ADMISSION_CPU_ONLY
    test_unexpected_pump_bad_alloc_joins_and_fails_closed();
    test_partial_thread_creation_fail_closed();
#endif
    std::printf("PASS ready-completion optin CPU contract checks=%d\n", checks.load());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL ready-completion optin: %s\n", e.what());
    return 1;
  }
}
