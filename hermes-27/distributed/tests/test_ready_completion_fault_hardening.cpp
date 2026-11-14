// Offline scheduler exception contracts. Compile with
// -DHERMES_ADMISSION_CPU_ONLY and pp2_batch.cu treated as C++.
// There are no CUDA devices, model weights, services or network calls.
#include "pp2_batch.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace pp2batch;
using namespace std::chrono;

int checks = 0;
void require(bool value, const char* why) {
  ++checks;
  if (!value) throw std::runtime_error(why);
}

struct FakeBackend final : StepBackend {
  struct State {
    int serial;
    std::atomic<bool> in_step{false};
  };
  std::atomic<int> opened{0}, closed{0}, freed_inflight{0};
  int fail_serial = 0;
  std::mutex mu;
  std::condition_variable cv;
  bool slow_entered = false;
  bool release_slow = false;

  bool supports_parallel_steps() const noexcept override { return true; }
  bool open(int, void*& ptr, std::string&) override {
    ptr = new State{++opened};
    return true;
  }
  bool step(void* ptr, int, long pos, std::vector<float>& logits,
            std::string& error) override {
    State& state = *static_cast<State*>(ptr);
    if (state.in_step.exchange(true)) throw std::runtime_error("overlap");
    {
      std::unique_lock<std::mutex> lock(mu);
      if (state.serial == 1 && pos == 0) {
        slow_entered = true;
        cv.notify_all();
        cv.wait(lock, [&] { return release_slow; });
      }
    }
    state.in_step = false;
    if (fail_serial == state.serial) {
      error = std::string(200, 'x');
      return false;
    }
    logits.assign(8, -1.f);
    logits[2] = 1.f;
    return true;
  }
  void close(void* ptr) noexcept override {
    State* state = static_cast<State*>(ptr);
    if (state->in_step.load()) ++freed_inflight;
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
  std::vector<SchedulerEvent> values;
  void record(const SchedulerEvent& event) {
    {
      std::lock_guard<std::mutex> lock(mu);
      values.push_back(event);
    }
    cv.notify_all();
  }
  bool wait_terminal(const std::string& id) {
    std::unique_lock<std::mutex> lock(mu);
    return cv.wait_for(lock, seconds(3), [&] {
      for (const auto& item : values)
        if (item.id == id && item.kind == EventKind::terminal) return true;
      return false;
    });
  }
  int count(const std::string& id, EventKind kind) {
    std::lock_guard<std::mutex> lock(mu);
    int result = 0;
    for (const auto& event : values)
      if (event.id == id && event.kind == kind) ++result;
    return result;
  }
  SchedulerEvent terminal(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu);
    for (const auto& event : values)
      if (event.id == id && event.kind == EventKind::terminal) return event;
    throw std::runtime_error("terminal missing");
  }
};

SchedulerConfig config(Events& events, int active = 2, int quantum = 2) {
  SchedulerConfig c;
  c.vocab_size = 8;
  c.max_context = 32;
  c.max_active = active;
  c.max_queued = 4;
  c.steps_per_tick = quantum;
  c.enable_parallel_steps = true;
  c.enable_ready_completion_dispatch = true;
  c.on_event = [&](const SchedulerEvent& event) { events.record(event); };
  return c;
}

void test_history_bad_alloc_closes_fast_before_slow() {
  FakeBackend backend;
  Events events;
  auto c = config(events);
  std::atomic<int> hooks{0};
  c.terminal_before_history_append = [&] {
    if (hooks++ == 0) throw std::bad_alloc();
  };
  ContinuousScheduler scheduler(c, backend);
  require(scheduler.enqueue({"A", {1}, 1}).code == SubmitCode::accepted,
          "accept slow");
  require(scheduler.enqueue({"B", {2}, 1}).code == SubmitCode::accepted,
          "accept fast");
  std::thread pump([&] { scheduler.tick(); });
  const bool slow = backend.wait_slow();
  const bool fast_terminal = events.wait_terminal("B");
  const int early_closed = backend.closed;
  const int early_bad_free = backend.freed_inflight;
  backend.release();
  pump.join();
  require(slow && fast_terminal && early_closed == 1 && early_bad_free == 0,
          "history bad_alloc must not delay B notification or free A");
  require(backend.opened == 2 && backend.closed == 2 &&
              backend.freed_inflight == 0, "all states closed exactly once");
  require(events.count("B", EventKind::terminal) == 1 &&
              events.count("A", EventKind::terminal) == 1,
          "history bad_alloc never loses or duplicates terminal");
  const auto m = scheduler.metrics();
  require(m.terminal_history_failures_total == 1 &&
              m.completed_total == 2 && m.active == 0,
          "history failure separately counted and both jobs completed");
  JobSnapshot discarded;
  require(!scheduler.inspect("B", discarded),
          "failed history cannot retain unbounded terminal jobs");
}

void test_queued_cancel_history_bad_alloc_wakes_client() {
  FakeBackend backend;
  Events events;
  auto c = config(events, 1, 1);
  c.terminal_before_history_append = [] { throw std::bad_alloc(); };
  ContinuousScheduler scheduler(c, backend);
  require(scheduler.enqueue({"A", {1}, 1}).code == SubmitCode::accepted,
          "accept active");
  require(scheduler.enqueue({"C", {3}, 1}).code == SubmitCode::accepted,
          "accept queued");
  std::thread pump([&] { scheduler.tick(); });
  const bool slow = backend.wait_slow();
  const bool cancelled = scheduler.cancel("C");
  const bool client_woke = events.wait_terminal("C");
  const int close_before_release = backend.closed;
  backend.release();
  pump.join();
  require(slow && cancelled && client_woke && close_before_release == 0,
          "queued cancellation wakes even when history append throws");
  require(events.count("C", EventKind::terminal) == 1 &&
              events.terminal("C").status == JobStatus::cancelled,
          "queued cancellation exact terminal");
  require(scheduler.metrics().terminal_history_failures_total == 2 &&
              backend.opened == 1 && backend.closed == 1 &&
              backend.freed_inflight == 0,
          "both history faults contained; no queued state allocated");
}

void test_terminal_snapshot_bad_alloc_gets_failed_terminal() {
  FakeBackend backend;
  backend.release();
  Events events;
  auto c = config(events, 1, 1);
  c.terminal_before_snapshot_copy = [] { throw std::bad_alloc(); };
  ContinuousScheduler scheduler(c, backend);
  scheduler.enqueue({"A", {1}, 1});
  require(scheduler.tick() == 1, "one ready decode after snapshot fault");
  const auto event = events.terminal("A");
  require(events.count("A", EventKind::terminal) == 1 &&
              event.status == JobStatus::failed && event.generated.empty(),
          "uncopyable output is never exposed as successful terminal");
  require(scheduler.metrics().terminal_delivery_failures_total == 1 &&
              backend.opened == 1 && backend.closed == 1 &&
              backend.freed_inflight == 0,
          "snapshot OOM accounted and state closed exactly once");
}

void test_terminal_callback_bad_alloc_calls_nothrow_wakeup() {
  FakeBackend backend;
  backend.release();
  Events events;
  auto c = config(events, 1, 1);
  std::atomic<int> attempted{0}, wakeups{0};
  c.on_event = [&](const SchedulerEvent& event) {
    if (event.kind == EventKind::terminal) {
      ++attempted;
      throw std::bad_alloc();
    }
  };
  c.on_terminal_delivery_failure = [&](const std::string& id, JobStatus) {
    if (id == "A") ++wakeups;
  };
  ContinuousScheduler scheduler(c, backend);
  scheduler.enqueue({"A", {1}, 1});
  require(scheduler.tick() == 1 && attempted == 1 && wakeups == 1,
          "throwing terminal consumer triggers exactly one emergency wakeup");
  require(scheduler.metrics().terminal_delivery_failures_total == 1 &&
              backend.opened == backend.closed, "callback failure accounted");
}

void test_double_snapshot_and_callback_fault_uses_emergency_wakeup() {
  FakeBackend backend;
  backend.release();
  Events events;
  auto c = config(events, 1, 1);
  std::atomic<int> wakeups{0};
  c.terminal_before_snapshot_copy = [] { throw std::bad_alloc(); };
  c.on_event = [](const SchedulerEvent& event) {
    if (event.kind == EventKind::terminal) throw std::bad_alloc();
  };
  c.on_terminal_delivery_failure = [&](const std::string& id, JobStatus) {
    if (id == "A") ++wakeups;
  };
  ContinuousScheduler scheduler(c, backend);
  scheduler.enqueue({"A", {1}, 1});
  require(scheduler.tick() == 1 && wakeups == 1,
          "uncopyable snapshot plus throwing callback still wakes HTTP owner");
  require(scheduler.metrics().terminal_delivery_failures_total == 2 &&
              backend.opened == backend.closed,
          "both terminal failures recorded without a backend leak");
}

void test_error_copy_bad_alloc_still_closes_and_notifies() {
  FakeBackend backend;
  backend.fail_serial = 1;
  backend.release();
  Events events;
  auto c = config(events, 1, 1);
  c.terminal_before_error_copy = [] { throw std::bad_alloc(); };
  ContinuousScheduler scheduler(c, backend);
  scheduler.enqueue({"A", {1}, 1});
  require(scheduler.tick() == 1 && events.wait_terminal("A"),
          "failed backend terminal survives error text allocation failure");
  require(events.terminal("A").status == JobStatus::failed &&
              backend.opened == 1 && backend.closed == 1,
          "error text loss never loses failure status or allocated state");
}
} // namespace

int main() {
  try {
    test_history_bad_alloc_closes_fast_before_slow();
    test_queued_cancel_history_bad_alloc_wakes_client();
    test_terminal_snapshot_bad_alloc_gets_failed_terminal();
    test_terminal_callback_bad_alloc_calls_nothrow_wakeup();
    test_double_snapshot_and_callback_fault_uses_emergency_wakeup();
    test_error_copy_bad_alloc_still_closes_and_notifies();
    std::printf("PASS ready fault-hardening CPU checks=%d\n", checks);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL ready fault-hardening: %s\n", e.what());
    return 1;
  }
  return 0;
}
