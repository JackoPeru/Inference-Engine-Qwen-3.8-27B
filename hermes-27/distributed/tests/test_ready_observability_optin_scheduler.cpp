// CPU-only deterministic observable scheduler test. No CUDA/model/HTTP.
// The clock advances only on explicit test events and observation calls;
// backend ordering is controlled by a condition-variable gate, not sleeps.
#include "pp2_batch.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace pp2batch;
using namespace std::chrono;

int checks = 0;
// Test-only marker set at the final line of B's backend step. The scheduler
// samples the fake clock immediately after returning from that step. Waiting
// for that sample ensures A's simulated delay cannot precede B's timestamp.
thread_local bool fast_step_has_returned = false;
void check(bool value, const char* message) {
  ++checks;
  if (!value) throw std::runtime_error(message);
}

struct Clock {
  std::atomic<std::int64_t> nanos{1'000'000'000};
  std::mutex sample_mu;
  std::condition_variable sample_cv;
  std::atomic<bool> fast_end_sampled{false};
  SchedulerClock::time_point now() noexcept {
    // Uniquely ordered timestamps without wall-clock dependence. No locks,
    // scheduler callbacks, allocations, or relative-time assumptions.
    const auto value = nanos.fetch_add(1000);
    if (fast_step_has_returned) {
      fast_step_has_returned = false;
      fast_end_sampled = true;
      sample_cv.notify_all();
    }
    return SchedulerClock::time_point(std::chrono::nanoseconds(value));
  }
  void advance(std::int64_t delta_ns) noexcept { nanos.fetch_add(delta_ns); }
  bool wait_fast_end_sample() {
    std::unique_lock<std::mutex> guard(sample_mu);
    return sample_cv.wait_for(
        guard, seconds(3), [&] { return fast_end_sampled.load(); });
  }
};

struct Backend final : StepBackend {
  struct State { int serial; std::atomic<bool> in_step{false}; };
  Clock& clock;
  std::atomic<int> opened{0}, closed{0}, premature_close{0};
  bool fail_fast = false;
  std::mutex mutex;
  std::condition_variable cv;
  bool slow_entered = false, fast_returned = false, released = false;
  explicit Backend(Clock& c) : clock(c) {}
  bool supports_parallel_steps() const noexcept override { return true; }
  bool open(int, void*& state, std::string&) override {
    state = new State{++opened};
    return true;
  }
  bool step(void* ptr, int, long, std::vector<float>& logits,
            std::string& error) override {
    auto& state = *static_cast<State*>(ptr);
    if (state.in_step.exchange(true)) throw std::runtime_error("state overlap");
    fast_step_has_returned = false;
    if (state.serial == 1) {
      std::unique_lock<std::mutex> lock(mutex);
      slow_entered = true;
      cv.notify_all();
      cv.wait(lock, [&] { return released; });
      clock.advance(50'000'000);  // B has been ready for 50 simulated ms.
    } else {
      clock.advance(1'000'000);
    }
    state.in_step = false;
    if (state.serial != 1) {
      {
        std::lock_guard<std::mutex> lock(mutex);
        fast_returned = true;
      }
      cv.notify_all();
      if (fail_fast) {
        error = "injected fast backend error";
        fast_step_has_returned = true;
        return false;
      }
    }
    logits.assign(8, -1.f);
    logits[2] = 1.f;
    if (state.serial != 1) fast_step_has_returned = true;
    return true;
  }
  void close(void* ptr) noexcept override {
    State* state = static_cast<State*>(ptr);
    if (state->in_step) ++premature_close;
    delete state;
    ++closed;
  }
  bool wait_backend() {
    std::unique_lock<std::mutex> lock(mutex);
    return cv.wait_for(lock, seconds(3),
                       [&] { return slow_entered && fast_returned; });
  }
  void release() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      released = true;
    }
    cv.notify_all();
  }
};

struct Events {
  std::mutex mu;
  std::condition_variable cv;
  std::vector<SchedulerEvent> records;
  void add(const SchedulerEvent& event) {
    {
      std::lock_guard<std::mutex> lock(mu);
      records.push_back(event);
    }
    cv.notify_all();
  }
  bool wait(const std::string& id, EventKind kind) {
    std::unique_lock<std::mutex> lock(mu);
    return cv.wait_for(lock, seconds(3), [&] {
      for (const auto& e : records)
        if (e.id == id && e.kind == kind) return true;
      return false;
    });
  }
  int count(const std::string& id, EventKind kind) {
    std::lock_guard<std::mutex> lock(mu);
    int result = 0;
    for (const auto& e : records)
      if (e.id == id && e.kind == kind) ++result;
    return result;
  }
};

SchedulerConfig config(Clock& clock, Events& events, bool ready,
                       bool observe = true) {
  SchedulerConfig c;
  c.vocab_size = 8;
  c.max_context = 32;
  c.max_active = 2;
  c.max_queued = 4;
  c.steps_per_tick = 2;
  c.queue_timeout = seconds(60);
  c.enable_parallel_steps = true;
  c.enable_ready_completion_dispatch = ready;
  c.enable_observability = observe;
  c.observability_now = [&] { return clock.now(); };
  c.on_event = [&](const SchedulerEvent& e) { events.add(e); };
  return c;
}

void test_ready_vs_wave_terminal_latency() {
  for (bool ready : {false, true}) {
    Clock clock;
    Backend backend(clock);
    Events events;
    ContinuousScheduler scheduler(config(clock, events, ready), backend);
    check(scheduler.enqueue({"A", {1}, 1}).code == SubmitCode::accepted,
          "slow enqueue accepted");
    check(scheduler.enqueue({"B", {2}, 1}).code == SubmitCode::accepted,
          "fast enqueue accepted");
    std::thread pump([&] { scheduler.tick(); });
    const bool both_in_backend = backend.wait_backend();
    const bool fast_timestamped = clock.wait_fast_end_sample();
    // For ready, wait until B is emitted before releasing A. In wave mode,
    // B's result must remain behind the join barrier.
    const bool b_early = ready ? events.wait("B", EventKind::terminal) : false;
    const auto while_slow = scheduler.metrics();
    const auto early_b_terminal = events.count("B", EventKind::terminal);
    backend.release();
    pump.join();
    const auto done = scheduler.metrics();
    check(both_in_backend && fast_timestamped,
          "both backend steps reached and B finish timestamp recorded");
    check(ready ? (b_early && early_b_terminal == 1 &&
                   while_slow.observed_ready_terminals == 1)
                : (early_b_terminal == 0 &&
                   while_slow.observed_wave_terminals == 0),
          "ready B terminal while A blocked; wave join delays B");
    check(done.observability_enabled && done.observed_queue_admissions == 2 &&
              done.observed_backend_steps == 2 &&
              done.observed_token_commits == 2 &&
              done.queued_high_water == 2 &&
              done.active_high_water == 2 &&
              done.outstanding_step_high_water == 2,
          "bounded admission/step/commit and high-water sample counts");
    check(done.queue_wait_ns_total > 0 &&
              done.backend_step_ns_total >= 50'000'000 &&
              done.backend_ready_to_token_commit_ns_max > 0 &&
              done.callback_ns_total > 0,
          "fake clock produces non-zero queue/backend/commit/callback timings");
    check(ready ? (done.observed_ready_terminals == 2 &&
                   done.observed_wave_terminals == 0)
                : (done.observed_wave_terminals == 2 &&
                   done.observed_ready_terminals == 0 &&
                   done.wave_terminal_ns_max >= 50'000'000),
          "exactly two successful terminal latency samples in selected mode");
    check(done.observed_terminal_callbacks == 2 &&
              done.observed_callbacks == 6 &&
              done.completed_total == 2 && done.failed_total == 0 &&
              done.cancelled_total == 0 &&
              backend.opened == 2 && backend.closed == 2 &&
              backend.premature_close == 0,
          "exact terminal/callback counts and join-before-close");
    std::printf("PASS observability ready=%d B_before_A=%d tokens=%llu "
                "terminals=%llu wave_delay_max_ns=%llu\n",
                ready ? 1 : 0, early_b_terminal ? 1 : 0,
                static_cast<unsigned long long>(done.observed_token_commits),
                static_cast<unsigned long long>(
                    ready ? done.observed_ready_terminals
                          : done.observed_wave_terminals),
                static_cast<unsigned long long>(done.wave_terminal_ns_max));
  }
}

void test_cancel_and_backend_failure_not_successful_terminal() {
  Clock clock;
  Backend backend(clock);
  backend.fail_fast = true;
  Events events;
  ContinuousScheduler scheduler(config(clock, events, true), backend);
  scheduler.enqueue({"A", {1}, 1});
  scheduler.enqueue({"B", {2}, 1});
  scheduler.enqueue({"C", {3}, 1});  // Must remain queued.
  std::thread pump([&] { scheduler.tick(); });
  const bool started = backend.wait_backend();
  const bool cancelled = scheduler.cancel("C");
  const bool cancelled_terminal = events.wait("C", EventKind::terminal);
  const bool failed_terminal = events.wait("B", EventKind::terminal);
  const auto before_release = scheduler.metrics();
  backend.release();
  pump.join();
  const auto stats = scheduler.metrics();
  check(started && cancelled && cancelled_terminal && failed_terminal &&
            before_release.observed_ready_terminals == 0 &&
            before_release.observed_token_commits == 0,
        "cancelled queue C and failed B are not successful ready completions");
  check(stats.observed_queue_admissions == 2 &&
            stats.observed_backend_steps == 2 &&
            stats.observed_token_commits == 1 &&
            stats.observed_ready_terminals == 1 &&
            stats.observed_wave_terminals == 0 &&
            stats.cancelled_total == 1 && stats.failed_total == 1 &&
            stats.completed_total == 1,
        "failed/cancelled requests counted without admission or success inflation");
  check(stats.observed_terminal_callbacks == 3 &&
            stats.queued_high_water == 3 &&
            stats.active_high_water == 2 &&
            stats.outstanding_step_high_water == 2 &&
            backend.opened == 2 && backend.closed == 2 &&
            backend.premature_close == 0,
        "queue/history caps and all backend states safely released");
}

void test_callback_stall_and_disabled_clock_fast_path() {
  for (bool observe : {false, true}) {
    Clock clock;
    Backend backend(clock);
    backend.release();
    Events events;
    auto c = config(clock, events, true, observe);
    std::atomic<int> fake_clock_uses{0};
    c.observability_now = [&]() {
      ++fake_clock_uses;
      if (!observe) throw std::logic_error("disabled clock must never be used");
      return clock.now();
    };
    c.on_event = [&](const SchedulerEvent& event) {
      if (event.kind == EventKind::token) clock.advance(1'500'000);
      if (event.kind == EventKind::terminal) clock.advance(2'500'000);
      events.add(event);
    };
    ContinuousScheduler scheduler(c, backend);
    scheduler.enqueue({"A", {1}, 1});
    check(scheduler.tick() == 1, "single ready step completed");
    const auto stats = scheduler.metrics();
    check(observe ? (fake_clock_uses > 0 &&
                     stats.callback_ns_max >= 2'500'000 &&
                     stats.terminal_callback_ns_max >= 2'500'000 &&
                     stats.observed_terminal_callbacks == 1 &&
                     stats.observed_backend_steps == 1 &&
                     stats.observed_token_commits == 1)
                  : (fake_clock_uses == 0 &&
                     stats.observed_callbacks == 0 &&
                     stats.observed_backend_steps == 0 &&
                     stats.outstanding_step_high_water == 0),
          "callback stall is measured iff explicitly opted in");
    check(backend.opened == 1 && backend.closed == 1 &&
              backend.premature_close == 0,
          "observability never changes backend lifetime");
  }
}

void test_serial_observability_does_not_claim_wave_terminal() {
  Clock clock;
  Backend backend(clock);
  backend.release();
  Events events;
  auto c = config(clock, events, false);
  c.enable_parallel_steps = false;
  c.max_active = 1;
  c.steps_per_tick = 1;
  ContinuousScheduler scheduler(c, backend);
  scheduler.enqueue({"serial", {1}, 1});
  check(scheduler.tick() == 1, "serial backend step completed");
  const auto stats = scheduler.metrics();
  check(stats.observed_serial_terminals == 1 &&
            stats.observed_wave_terminals == 0 &&
            stats.observed_ready_terminals == 0 &&
            stats.observed_backend_steps == 1 &&
            stats.observed_token_commits == 1 &&
            stats.outstanding_step_high_water == 1,
        "serial step metrics never mislabeled as parallel terminal");
  check(events.count("serial", EventKind::terminal) == 1 &&
            backend.opened == 1 && backend.closed == 1,
        "serial metrics maintain one notification and safe lifetime");
}
}  // namespace

int main() {
  try {
    test_ready_vs_wave_terminal_latency();
    test_cancel_and_backend_failure_not_successful_terminal();
    test_callback_stall_and_disabled_clock_fast_path();
    test_serial_observability_does_not_claim_wave_terminal();
    std::printf("PASS ready observability optin CPU checks=%d\n", checks);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL ready observability optin: %s\n", e.what());
    return 1;
  }
}
