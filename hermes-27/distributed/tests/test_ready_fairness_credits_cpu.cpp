// HERMES-27 CPU/OFFLINE acceptance witnesses on the ACTUAL ready scheduler.
// No CUDA/model/server. Synchronization is by CV gates (never timed sleeps).
// Timeouts in assertions are deadlock guards, not throughput measurements.
#include "pp2_batch.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
using namespace pp2batch;
using namespace std::chrono;

int assertions = 0;
void check(bool ok, const char* reason) {
  ++assertions;
  if (!ok) throw std::runtime_error(reason);
}

struct FakeClock {
  std::atomic<std::int64_t> t{1'000'000'000};
  SchedulerClock::time_point now() noexcept {
    return SchedulerClock::time_point(
        nanoseconds(t.fetch_add(1000, std::memory_order_relaxed)));
  }
  void advance(std::int64_t ns) noexcept {
    t.fetch_add(ns, std::memory_order_relaxed);
  }
};

struct CallbackGate {
  std::mutex mu;
  std::condition_variable cv;
  bool callback_entered = false;
  bool callback_release = false;
  int other_backends_finished = 0;

  bool wait_callback() {
    std::unique_lock<std::mutex> lock(mu);
    return cv.wait_for(lock, seconds(3), [&] { return callback_entered; });
  }
  bool wait_other_backends(int expected) {
    std::unique_lock<std::mutex> lock(mu);
    return cv.wait_for(lock, seconds(3), [&] {
      return other_backends_finished >= expected;
    });
  }
  void release() {
    {
      std::lock_guard<std::mutex> lock(mu);
      // If an assertion timeout prevented the intended callback handoff,
      // unblock backend workers anyway so the test can join and report FAIL
      // instead of hanging the sanitizer runner indefinitely.
      callback_entered = true;
      callback_release = true;
    }
    cv.notify_all();
  }
};

struct Backend final : StepBackend {
  struct State {
    int serial;
    int next_position = 0;
    std::atomic<bool> in_step{false};
  };
  std::atomic<int> opened{0}, closed{0}, freed_in_step{0};
  std::atomic<int> live_steps{0}, peak_steps{0}, overlap{0};
  std::atomic<int> failure_serial{0};
  bool block_first_A = false;
  CallbackGate* callback_gate = nullptr;
  std::mutex mu;
  std::condition_variable cv;
  bool slow_entered = false, slow_release = false;
  std::vector<std::pair<int, long>> start_order;

  bool supports_parallel_steps() const noexcept override { return true; }
  bool open(int, void*& ptr, std::string&) override {
    ptr = new State{++opened};
    return true;
  }
  bool step(void* ptr, int, long pos, std::vector<float>& logits,
            std::string& error) override {
    State& state = *static_cast<State*>(ptr);
    if (state.in_step.exchange(true)) {
      ++overlap;
      throw std::logic_error("overlap: two concurrent steps on same state");
    }
    const int inflight = live_steps.fetch_add(1) + 1;
    for (int observed = peak_steps.load(); observed < inflight &&
             !peak_steps.compare_exchange_weak(observed, inflight);) {}
    bool valid;
    {
      std::unique_lock<std::mutex> lock(mu);
      start_order.emplace_back(state.serial, pos);
      valid = pos == state.next_position;
      if (valid) ++state.next_position;
      if (block_first_A && state.serial == 1 && pos == 0) {
        slow_entered = true;
        cv.notify_all();
        cv.wait(lock, [&] { return slow_release; });
      }
    }
    if (callback_gate && state.serial >= 3) {
      {
        std::unique_lock<std::mutex> lock(callback_gate->mu);
        callback_gate->cv.wait(lock,
                               [&] { return callback_gate->callback_entered; });
        ++callback_gate->other_backends_finished;
      }
      callback_gate->cv.notify_all();
    }
    live_steps.fetch_sub(1);
    state.in_step = false;
    if (!valid) {
      error = "token position duplicate or skipped";
      return false;
    }
    if (failure_serial.load() == state.serial) {
      error = "injected backend error";
      return false;
    }
    logits.assign(8, -1.f);
    logits[2] = 1.f;
    return true;
  }
  void close(void* ptr) noexcept override {
    auto* state = static_cast<State*>(ptr);
    if (state->in_step) ++freed_in_step;
    delete state;
    ++closed;
  }
  bool wait_slow() {
    std::unique_lock<std::mutex> lock(mu);
    return cv.wait_for(lock, seconds(3), [&] { return slow_entered; });
  }
  void release_slow() {
    {
      std::lock_guard<std::mutex> lock(mu);
      slow_release = true;
    }
    cv.notify_all();
  }
};

struct Events {
  struct Item {
    std::string id;
    EventKind kind;
    JobStatus status;
    int token_id;
  };
  std::mutex mu;
  std::condition_variable cv;
  std::vector<Item> items;
  void record(const SchedulerEvent& e) {
    {
      std::lock_guard<std::mutex> lock(mu);
      items.push_back({e.id, e.kind, e.status, e.token_id});
    }
    cv.notify_all();
  }
  bool wait(const std::string& id, EventKind kind) {
    std::unique_lock<std::mutex> lock(mu);
    return cv.wait_for(lock, seconds(3), [&] {
      return std::any_of(items.begin(), items.end(), [&](const Item& item) {
        return item.id == id && item.kind == kind;
      });
    });
  }
  int count(const std::string& id, EventKind kind) {
    std::lock_guard<std::mutex> lock(mu);
    return static_cast<int>(std::count_if(items.begin(), items.end(),
        [&](const Item& item) { return item.id == id && item.kind == kind; }));
  }
  JobStatus terminal_status(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu);
    for (const auto& item : items)
      if (item.id == id && item.kind == EventKind::terminal)
        return item.status;
    throw std::runtime_error("terminal not present");
  }
  std::vector<std::string> admissions() {
    std::lock_guard<std::mutex> lock(mu);
    std::vector<std::string> result;
    for (const auto& item : items)
      if (item.kind == EventKind::admitted) result.push_back(item.id);
    return result;
  }
};

SchedulerConfig cfg(int active, int steps, Events& events, FakeClock& clock) {
  SchedulerConfig c;
  c.vocab_size = 8;
  c.max_context = 32;
  c.max_active = static_cast<std::size_t>(active);
  c.max_queued = static_cast<std::size_t>(active);
  c.steps_per_tick = static_cast<std::size_t>(steps);
  c.max_retained = 64;
  c.queue_timeout = seconds(60);
  c.enable_parallel_steps = true;
  c.enable_ready_completion_dispatch = true;
  c.enable_observability = true;
  c.observability_now = [&] { return clock.now(); };
  c.on_event = [&](const SchedulerEvent& e) { events.record(e); };
  return c;
}

void check_closed(const Backend& backend) {
  check(backend.opened == backend.closed &&
            backend.freed_in_step == 0 && backend.overlap == 0 &&
            backend.live_steps == 0,
        "all backend states closed exactly once, no overlap or early free");
}

void test_arrival_and_bounded_backpressure(int n) {
  Backend backend;
  backend.block_first_A = true;
  FakeClock clock;
  Events events;
  auto config = cfg(n, n, events, clock);
  ContinuousScheduler scheduler(config, backend);
  for (int i = 0; i < n; ++i) {
    check(scheduler.enqueue({std::string(1, char('A' + i)), {1}, 1}).code ==
              SubmitCode::accepted, "initial cohort FIFO accepted");
  }
  std::atomic<std::size_t> dispatched{0};
  std::thread pump([&] { dispatched = scheduler.tick(); });
  const bool slow_in_backend = backend.wait_slow();
  bool fast_terminals_before_A = true;
  for (int i = 1; i < n; ++i)
    fast_terminals_before_A &=
        events.wait(std::string(1, char('A' + i)), EventKind::terminal);
  // The new cohort arrives after fast siblings finish while A still owns
  // one in-flight state. The scheduler conservatively forbids *admission*
  // during any outstanding backend step, including free capacity.
  std::vector<std::string> queued;
  for (int i = 0; i < n; ++i) {
    auto id = "Q" + std::to_string(i);
    queued.push_back(id);
    check(scheduler.enqueue({id, {2}, 1}).code == SubmitCode::accepted,
          "mid-flight arrival accepted into bounded waiting queue");
  }
  check(scheduler.enqueue({"overflow", {2}, 1}).code ==
            SubmitCode::queue_full,
        "full waiting queue rejects extra request without allocating state");
  check(scheduler.cancel("Q1"),
        "cancel a queued request to reclaim exactly one waiting credit");
  check(events.wait("Q1", EventKind::terminal),
        "cancelled queued request delivers immediate terminal");
  check(scheduler.enqueue({"replacement", {2}, 1}).code ==
            SubmitCode::accepted, "cancelled queue credit can be reused");
  const auto during = scheduler.metrics();
  const int opened_during = backend.opened;
  const auto admitted_during = events.admissions();
  const bool before_release_A = events.count("A", EventKind::terminal) == 0;
  backend.release_slow();
  pump.join();
  check(slow_in_backend && fast_terminals_before_A && before_release_A,
        "all fast siblings finish while A is still held in backend");
  check(during.queued == static_cast<std::size_t>(n) &&
            during.active == 1 && opened_during == n &&
            admitted_during.size() == static_cast<std::size_t>(n) &&
            during.rejected_total == 1 && during.cancelled_total == 1,
        "in-flight A prevents new backend admission; bounded queue and reject");
  check(dispatched == static_cast<std::size_t>(n) &&
            scheduler.metrics().observed_backend_steps ==
                static_cast<std::uint64_t>(n),
        "first tick executes exactly N distinct in-flight backend steps");
  check(scheduler.tick() == static_cast<std::size_t>(n),
        "second pump tick admits all remaining FIFO waiting requests");
  auto expected = admitted_during;
  for (int i = 0; i < n; ++i)
    if (i != 1) expected.push_back("Q" + std::to_string(i));
  expected.push_back("replacement");
  check(events.admissions() == expected,
        "new arrivals preserve strict FIFO admission despite cancellation");
  const auto done = scheduler.metrics();
  check(done.completed_total == static_cast<std::size_t>(2 * n) &&
            done.cancelled_total == 1 &&
            done.rejected_total == 1 &&
            done.accepted_total == static_cast<std::size_t>(2 * n + 1) &&
            done.observed_backend_steps ==
                static_cast<std::uint64_t>(2 * n) &&
            done.observed_token_commits ==
                static_cast<std::uint64_t>(2 * n) &&
            done.queued_high_water == static_cast<std::uint64_t>(n) &&
            done.active_high_water == static_cast<std::uint64_t>(n) &&
            done.outstanding_step_high_water <= static_cast<std::uint64_t>(n),
        "accepted/rejected/cancelled counts and anonymous metric watermarks");
  check(backend.peak_steps <= n &&
            backend.opened == 2 * n, "hard active-state and backend concurrency cap");
  check_closed(backend);
  std::printf("PASS N=%d midflight FIFO/backpressure: peak_backend=%d "
              "highwater=%llu complete=%llu queue_rejected=%llu\n", n,
              backend.peak_steps.load(),
              static_cast<unsigned long long>(done.outstanding_step_high_water),
              static_cast<unsigned long long>(done.completed_total),
              static_cast<unsigned long long>(done.rejected_total));
}

void test_round_robin_credits(int n) {
  Backend backend;
  FakeClock clock;
  Events events;
  const int quantum = n / 2;
  ContinuousScheduler scheduler(cfg(n, quantum, events, clock), backend);
  for (int i = 0; i < n; ++i)
    check(scheduler.enqueue({std::string(1, char('A' + i)), {1}, 2}).code ==
              SubmitCode::accepted, "RR cohort accepted");
  for (int tick = 0; tick < 4; ++tick) {
    check(scheduler.tick() == static_cast<std::size_t>(quantum),
          "ready per-tick quantum respected");
    for (int i = 0; i < n; ++i) {
      const int expected = (tick + 1) / 2 +
                           (i < quantum && tick % 2 == 0 ? 1 : 0);
      // tick0: first half 1, second half 0; tick1: all 1.
      // tick2: first half 2, second half 1; tick3: all 2.
      JobSnapshot snapshot;
      check(scheduler.inspect(std::string(1, char('A' + i)), snapshot) &&
                static_cast<int>(snapshot.generated.size()) == expected,
            "FIFO RR equalizes per-state token credits across four ticks");
    }
  }
  const auto metrics = scheduler.metrics();
  check(metrics.completed_total == static_cast<std::size_t>(n) &&
            metrics.observed_backend_steps == static_cast<std::uint64_t>(2 * n) &&
            metrics.observed_token_commits == static_cast<std::uint64_t>(2 * n) &&
            metrics.outstanding_step_high_water <=
                static_cast<std::uint64_t>(quantum),
        "RR quantum and aggregate steps bounded without overcount");
  const auto start_order = [&]() {
    std::lock_guard<std::mutex> lock(backend.mu);
    return backend.start_order;
  }();
  check(start_order.size() == static_cast<std::size_t>(2 * n),
        "exactly two backend steps per state");
  for (int i = 1; i <= n; ++i) {
    auto first = std::count(start_order.begin(), start_order.end(),
                            std::make_pair(i, 0L));
    auto second = std::count(start_order.begin(), start_order.end(),
                             std::make_pair(i, 1L));
    check(first == 1 && second == 1,
          "no request skipped or replayed absolute token position");
  }
  check_closed(backend);
  std::printf("PASS N=%d RR quantum=%d ticks=4 steps=%llu "
              "mean_step_host_fake_ns=%llu\n", n, quantum,
              static_cast<unsigned long long>(metrics.observed_backend_steps),
              static_cast<unsigned long long>(metrics.observed_backend_steps
                  ? metrics.backend_step_ns_total / metrics.observed_backend_steps
                  : 0));
}

void test_fast_peers_reuse_credits_while_slow_A_blocks(int n) {
  Backend backend;
  backend.block_first_A = true;
  FakeClock clock;
  Events events;
  // N8 => 16-slot bounded ready task array; all B..N may take two steps
  // while A is held in its *first* backend step.
  ContinuousScheduler scheduler(cfg(n, 2 * n, events, clock), backend);
  for (int i = 0; i < n; ++i) {
    const std::string id(1, char('A' + i));
    const int output_tokens = i == 0 ? 1 : 2;
    check(scheduler.enqueue({id, {1}, output_tokens}).code ==
              SubmitCode::accepted,
          "mixed slow/fast credit cohort enqueued");
  }
  std::atomic<std::size_t> dispatched{0};
  std::thread pump([&] { dispatched = scheduler.tick(); });
  const bool A_started = backend.wait_slow();
  bool peers_all_complete = true;
  for (int i = 1; i < n; ++i)
    peers_all_complete &=
        events.wait(std::string(1, char('A' + i)), EventKind::terminal);
  const auto while_A = scheduler.metrics();
  const auto A_not_committed = events.count("A", EventKind::token) == 0;
  backend.release_slow();
  pump.join();
  check(A_started && peers_all_complete && A_not_committed &&
            while_A.completed_total == static_cast<std::size_t>(n - 1) &&
            while_A.observed_token_commits ==
                static_cast<std::uint64_t>(2 * (n - 1)),
        "fast peers used two backend credits each before slow A completed one");
  const auto done = scheduler.metrics();
  check(dispatched == static_cast<std::size_t>(2 * n - 1) &&
            done.parallel_dispatched_steps_total ==
                static_cast<std::size_t>(2 * n - 1) &&
            done.observed_backend_steps ==
                static_cast<std::uint64_t>(2 * n - 1) &&
            done.completed_total == static_cast<std::size_t>(n) &&
            done.outstanding_step_high_water <= static_cast<std::uint64_t>(n),
        "2N quantum bounds all fast re-launch work without re-running slow state");
  for (int i = 0; i < n; ++i) {
    const std::string id(1, char('A' + i));
    const int expected_tokens = i == 0 ? 1 : 2;
    JobSnapshot snapshot;
    check(scheduler.inspect(id, snapshot) &&
              snapshot.status == JobStatus::completed &&
              snapshot.generated.size() == static_cast<std::size_t>(expected_tokens) &&
              events.count(id, EventKind::token) == expected_tokens,
          "exact per-state generation count despite overlapped re-launch");
  }
  check_closed(backend);
  std::printf("PASS N=%d throughput credits while A held: fast_tokens=%llu "
              "steps=%llu\n", n,
              static_cast<unsigned long long>(while_A.observed_token_commits),
              static_cast<unsigned long long>(done.observed_backend_steps));
}

void test_queued_expiry_cancel_and_failed_step(int n) {
  Backend backend;
  backend.block_first_A = true;
  backend.failure_serial = 2;
  FakeClock clock;
  Events events;
  auto config = cfg(n, n, events, clock);
  config.queue_timeout = milliseconds(80);
  ContinuousScheduler scheduler(config, backend);
  for (int i = 0; i < n; ++i)
    check(scheduler.enqueue({std::string(1, char('A' + i)), {1}, 1}).code ==
              SubmitCode::accepted, "error/expiry cohort accepted");
  std::thread pump([&] { scheduler.tick(); });
  const bool slow = backend.wait_slow();
  // Explicitly backdate only the QUEUED request: no waiting on wall time.
  check(scheduler.enqueue({"expired", {2}, 1},
                          SchedulerClock::now() - seconds(1)).code ==
            SubmitCode::accepted,
        "already expired queued request accepted for tick deadline check");
  check(scheduler.enqueue({"cancelled", {2}, 1}).code ==
            SubmitCode::accepted, "cancel candidate enqueued");
  const bool cancel_succeeded = scheduler.cancel("cancelled");
  const bool cancelled_woke = events.wait("cancelled", EventKind::terminal);
  const bool expired_woke = events.wait("expired", EventKind::terminal);
  const bool failure_woke = events.wait("B", EventKind::terminal);
  const auto during = scheduler.metrics();
  const bool before_slow_release = events.count("A", EventKind::terminal) == 0;
  backend.release_slow();
  pump.join();
  check(slow && cancel_succeeded && cancelled_woke && expired_woke &&
            failure_woke && before_slow_release,
        "queued expiry and cancellation and B backend error delivered while A blocked");
  check(events.terminal_status("cancelled") == JobStatus::cancelled &&
            events.terminal_status("expired") == JobStatus::expired &&
            events.terminal_status("B") == JobStatus::failed,
        "exact non-completed terminal reasons and one event per job");
  const auto metrics = scheduler.metrics();
  check(during.observed_backend_steps >= 1 &&
            metrics.failed_total == 1 && metrics.cancelled_total == 1 &&
            metrics.expired_total == 1 &&
            metrics.completed_total == static_cast<std::size_t>(n - 1) &&
            metrics.observed_queue_admissions == static_cast<std::uint64_t>(n) &&
            metrics.observed_backend_steps == static_cast<std::uint64_t>(n) &&
            metrics.observed_token_commits ==
                static_cast<std::uint64_t>(n - 1) &&
            metrics.observed_ready_terminals ==
                static_cast<std::uint64_t>(n - 1) &&
            backend.opened == n,
        "queued cancellation and deadline do not consume admission or success");
  check(events.count("B", EventKind::token) == 0 &&
            events.count("expired", EventKind::admitted) == 0 &&
            events.count("cancelled", EventKind::admitted) == 0,
        "no token on failed step and no state opened for terminated queue jobs");
  check_closed(backend);
  std::printf("PASS N=%d deadline/cancel/error while slow A in-flight: "
              "expired=%llu failed=%llu cancelled=%llu\n", n,
              static_cast<unsigned long long>(metrics.expired_total),
              static_cast<unsigned long long>(metrics.failed_total),
              static_cast<unsigned long long>(metrics.cancelled_total));
}

void test_callback_head_of_line_and_backpressure() {
  Backend backend;
  backend.block_first_A = true;
  CallbackGate gate;
  backend.callback_gate = &gate;
  FakeClock clock;
  Events events;
  auto config = cfg(4, 4, events, clock);
  std::atomic<bool> reentrant_ok{false}, backpressure_cancelled{false};
  ContinuousScheduler* ptr = nullptr;
  config.on_event = [&](const SchedulerEvent& event) {
    events.record(event);
    if (event.id == "B" && event.kind == EventKind::token) {
      JobSnapshot snapshot;
      reentrant_ok = ptr->inspect("A", snapshot) &&
                     ptr->metrics().active >= 2;
      // Simulate a bounded SSE queue rejecting this request's next token.
      backpressure_cancelled = ptr->cancel("B");
      std::unique_lock<std::mutex> lock(gate.mu);
      gate.callback_entered = true;
      gate.cv.notify_all();
      gate.cv.wait(lock, [&] { return gate.callback_release; });
      clock.advance(7'000'000); // SYNTHETIC callback latency, not wall time.
    }
  };
  ContinuousScheduler scheduler(config, backend);
  ptr = &scheduler;
  for (const char* id : {"A", "B", "C", "D"})
    check(scheduler.enqueue({id, {1}, 2}).code == SubmitCode::accepted,
          "callback backpressure test enqueues distinct states");
  std::thread pump([&] { scheduler.tick(); });
  const bool A_inflight = backend.wait_slow();
  const bool callback_held = gate.wait_callback();
  const bool sibling_backends_completed = gate.wait_other_backends(2);
  const bool no_C_or_D_events = events.count("C", EventKind::token) == 0 &&
                                events.count("D", EventKind::token) == 0;
  const auto during = scheduler.metrics();
  gate.release();
  const bool C_received = events.wait("C", EventKind::token);
  const bool D_received = events.wait("D", EventKind::token);
  const bool B_cancelled = events.wait("B", EventKind::terminal);
  const auto A_still_inflight = events.count("A", EventKind::token) == 0;
  backend.release_slow();
  pump.join();
  check(A_inflight && callback_held && sibling_backends_completed &&
            no_C_or_D_events && during.active == 4,
        "single pump callback holds C/D delivery despite completed C/D workers");
  check(reentrant_ok && backpressure_cancelled && B_cancelled &&
            events.terminal_status("B") == JobStatus::cancelled,
        "callback can inspect/metrics/cancel under no scheduler lock");
  check(C_received && D_received && A_still_inflight &&
            events.count("B", EventKind::token) == 1,
        "bounded SSE cancellation limits B to one token and releases C/D before A");
  const auto metrics = scheduler.metrics();
  check(metrics.cancelled_total == 1 && metrics.failed_total == 0 &&
            metrics.observed_backend_steps == 4 &&
            metrics.observed_token_commits == 4 &&
            metrics.callback_ns_max >= 7'000'000,
        "backpressure and synthetic 7ms synchronous callback stall measured");
  check(backend.peak_steps <= 4 && backend.opened == 4,
        "maximum N4 backend state concurrency");
  // The three remaining jobs require a second token on another tick.
  check(scheduler.tick() == 3, "three surviving requests progress on next tick");
  check(scheduler.metrics().completed_total == 3 &&
            scheduler.metrics().cancelled_total == 1,
        "backpressure cancellation does not poison siblings or fairness");
  check_closed(backend);
  std::printf("PASS N=4 backpressure callback HOL: fake_callback_ns_max=%llu "
              "survivors=%llu\n",
              static_cast<unsigned long long>(metrics.callback_ns_max),
              static_cast<unsigned long long>(
                  scheduler.metrics().completed_total));
}
}  // namespace

int main() {
  try {
    for (int n : {2, 4, 8}) test_arrival_and_bounded_backpressure(n);
    for (int n : {2, 4, 8}) test_round_robin_credits(n);
    for (int n : {2, 4, 8})
      test_fast_peers_reuse_credits_while_slow_A_blocks(n);
    for (int n : {2, 4, 8}) test_queued_expiry_cancel_and_failed_step(n);
    test_callback_head_of_line_and_backpressure();
    std::printf("PASS ready fairness/backpressure CPU assertions=%d\n",
                assertions);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL ready fairness/backpressure: %s\n",
                 error.what());
    return 1;
  }
}
