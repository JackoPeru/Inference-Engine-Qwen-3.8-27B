// HERMES-27 STRICT opt-in async callback isolation: actual ready scheduler,
// fake stateful backend, CV-gated callbacks, zero CUDA/network/model.
// Timing guards guard deadlocks; no measured wall or GPU speedup is claimed.
#include "pp2_batch.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
using namespace pp2batch;
using namespace std::chrono;

int checks = 0;
void check(bool good, const char* reason) {
  ++checks;
  if (!good) throw std::runtime_error(reason);
}

struct Gate {
  std::mutex mu;
  std::condition_variable cv;
  bool block_A = false, block_B = false;
  bool A_entered = false, B_entered = false, A_released = false,
       B_released = false, A_left = false, B_left = false;
  bool B_backend_returned = false;
  bool cancel_A_on_release = false;
  ContinuousScheduler* scheduler = nullptr;
  std::atomic<int> cancelled_by_backpressure{0};
  void enter(const std::string& id) {
    if (id != "A" && id != "B") return;
    std::unique_lock<std::mutex> guard(mu);
    if (id == "A" && block_A) {
      A_entered = true;
      cv.notify_all();
      cv.wait(guard, [&] { return A_released; });
      A_left = true;
      guard.unlock();
      if (cancel_A_on_release && scheduler && scheduler->cancel("A"))
        ++cancelled_by_backpressure;
      cv.notify_all();
    } else if (id == "B" && block_B) {
      B_entered = true;
      cv.notify_all();
      cv.wait(guard, [&] { return B_released; });
      B_left = true;
      cv.notify_all();
    }
  }
  bool wait_A() {
    std::unique_lock<std::mutex> guard(mu);
    return cv.wait_for(guard, seconds(3), [&] { return A_entered; });
  }
  bool wait_B() {
    std::unique_lock<std::mutex> guard(mu);
    return cv.wait_for(guard, seconds(3), [&] { return B_entered; });
  }
  bool wait_B_backend() {
    std::unique_lock<std::mutex> guard(mu);
    return cv.wait_for(guard, seconds(3), [&] { return B_backend_returned; });
  }
  void release_A() {
    {
      std::lock_guard<std::mutex> guard(mu);
      A_released = true;
    }
    cv.notify_all();
  }
  void release_B() {
    {
      std::lock_guard<std::mutex> guard(mu);
      B_released = true;
    }
    cv.notify_all();
  }
  void release_all() {
    {
      std::lock_guard<std::mutex> guard(mu);
      A_released = true;
      B_released = true;
    }
    cv.notify_all();
  }
};

struct Backend : StepBackend {
  struct State {
    int serial;
    int pos = 0;
    std::atomic<bool> running{false};
  };
  Gate* gate = nullptr;
  std::atomic<int> open_count{0}, close_count{0}, step_count{0},
      premature_close{0}, overlap{0};
  std::mutex mu;
  std::mutex closed_mu;
  std::condition_variable closed_cv;
  std::vector<std::pair<int, long>> positions;
  bool supports_parallel_steps() const noexcept override { return true; }
  bool open(int, void*& ptr, std::string&) override {
    ptr = new State{++open_count};
    return true;
  }
  bool step(void* ptr, int, long position, std::vector<float>& logits,
            std::string& error) override {
    auto& state = *static_cast<State*>(ptr);
    if (state.running.exchange(true)) {
      ++overlap;
      error = "overlapping state step";
      return false;
    }
    bool valid;
    {
      std::lock_guard<std::mutex> guard(mu);
      positions.emplace_back(state.serial, position);
      valid = state.pos == position;
      if (valid) ++state.pos;
    }
    ++step_count;
    // In the baseline comparison the callback of A must be blocking before
    // the backend for B can publish its result. Otherwise B could win the
    // ready queue before A's callback starts by ordinary OS scheduling.
    if (gate && state.serial == 2 && position == 0) {
      std::unique_lock<std::mutex> guard(gate->mu);
      gate->cv.wait_for(guard, seconds(3),
                        [&] { return gate->A_entered; });
      gate->B_backend_returned = true;
      gate->cv.notify_all();
    }
    state.running = false;
    if (!valid) {
      error = "duplicate/skipped token position";
      return false;
    }
    logits.assign(8, -1.f);
    logits[2] = 1.f;
    return true;
  }
  void close(void* ptr) noexcept override {
    State* state = static_cast<State*>(ptr);
    if (state->running) ++premature_close;
    delete state;
    ++close_count;
    closed_cv.notify_all();
  }
  bool wait_closed(int expected) {
    std::unique_lock<std::mutex> guard(closed_mu);
    return closed_cv.wait_for(guard, seconds(3), [&] {
      return close_count.load() >= expected;
    });
  }
};

struct Events {
  struct Item {
    std::string id;
    EventKind kind;
    JobStatus status;
    std::size_t prompt_consumed;
    std::vector<int> generated;
    std::string error;
  };
  std::mutex mu;
  std::condition_variable cv;
  std::vector<Item> items;
  std::unordered_map<std::string, int> callback_active;
  std::atomic<int> per_job_overlap{0};
  Gate* gate = nullptr;
  ContinuousScheduler* scheduler = nullptr;
  bool throw_on_B_token = false;
  void deliver(const SchedulerEvent& ev) {
    {
      std::lock_guard<std::mutex> guard(mu);
      if (++callback_active[ev.id] != 1) ++per_job_overlap;
      items.push_back({ev.id, ev.kind, ev.status, ev.prompt_consumed,
                       ev.generated, ev.error});
    }
    cv.notify_all();
    if (gate && ev.kind == EventKind::token) gate->enter(ev.id);
    if (throw_on_B_token && ev.id == "B" && ev.kind == EventKind::token) {
      {
        std::lock_guard<std::mutex> guard(mu);
        --callback_active[ev.id];
      }
      throw std::bad_alloc();
    }
    {
      std::lock_guard<std::mutex> guard(mu);
      --callback_active[ev.id];
    }
    cv.notify_all();
  }
  bool wait(const std::string& id, EventKind kind) {
    std::unique_lock<std::mutex> guard(mu);
    return cv.wait_for(guard, seconds(3), [&] {
      for (const auto& item : items)
        if (item.id == id && item.kind == kind) return true;
      return false;
    });
  }
  int count(const std::string& id, EventKind kind) {
    std::lock_guard<std::mutex> guard(mu);
    return static_cast<int>(std::count_if(items.begin(), items.end(),
        [&](const Item& item) {
          return item.id == id && item.kind == kind;
        }));
  }
  std::vector<EventKind> order(const std::string& id) {
    std::lock_guard<std::mutex> guard(mu);
    std::vector<EventKind> result;
    for (const auto& item : items)
      if (item.id == id) result.push_back(item.kind);
    return result;
  }
  JobStatus terminal(const std::string& id) {
    std::lock_guard<std::mutex> guard(mu);
    for (const auto& item : items)
      if (item.id == id && item.kind == EventKind::terminal)
        return item.status;
    throw std::runtime_error("terminal missing");
  }
  Item terminal_snapshot(const std::string& id) {
    std::lock_guard<std::mutex> guard(mu);
    for (const auto& item : items)
      if (item.id == id && item.kind == EventKind::terminal) return item;
    throw std::runtime_error("terminal snapshot missing");
  }
};

SchedulerConfig config(Events& events, bool async) {
  SchedulerConfig c;
  c.vocab_size = 8;
  c.max_context = 256;
  c.max_active = 2;
  c.max_queued = 32;
  c.steps_per_tick = 16;
  c.max_retained = 64;
  c.enable_parallel_steps = true;
  c.enable_ready_completion_dispatch = true;
  c.enable_ready_callback_delivery = async;
  c.on_event = [&](const SchedulerEvent& e) { events.deliver(e); };
  // The borrow-only terminal emergency callback is required for opt-in.
  c.on_terminal_delivery_failure =
      [](const std::string&, JobStatus) noexcept {};
  return c;
}

// Keep tick() driving while an async callback is blocked. In sync mode its
// pump thread is expected to block until the CV release; destructor joins it.
struct Driver {
  ContinuousScheduler& scheduler;
  std::atomic<bool> stop{false};
  std::thread worker;
  explicit Driver(ContinuousScheduler& s) : scheduler(s) {
    worker = std::thread([this] {
      while (!stop.load()) {
        scheduler.tick();
        std::this_thread::yield();
      }
    });
  }
  void join() {
    stop = true;
    if (worker.joinable()) worker.join();
  }
  ~Driver() { join(); }
};

void validate_backend(Backend& backend) {
  check(backend.open_count == backend.close_count &&
            backend.premature_close == 0 &&
            backend.overlap == 0,
        "all state closes exactly once, joined before close, no same-state overlap");
}

void test_blocked_A_callback_does_not_block_B_in_optin() {
  for (bool async : {false, true}) {
    Gate gate;
    gate.block_A = true;
    Backend backend;
    backend.gate = &gate;
    Events events;
    events.gate = &gate;
    ContinuousScheduler scheduler(config(events, async), backend);
    gate.scheduler = &scheduler;
    check(scheduler.enqueue({"A", {1}, 1}).code == SubmitCode::accepted,
          "A admitted");
    check(scheduler.enqueue({"B", {2}, 1}).code == SubmitCode::accepted,
          "B admitted");
    Driver driver(scheduler);
    const bool A_blocked = gate.wait_A();
    const bool B_backend_done = gate.wait_B_backend();
    // In the synchronous default, the pump remains in the A callback.
    // In async mode, another managed callback worker can deliver B's token
    // and terminal while A still holds its own callback.
    const bool B_terminal_early =
        async ? events.wait("B", EventKind::terminal)
              : events.count("B", EventKind::terminal) != 0;
    const auto before_release = scheduler.metrics();
    gate.release_all();
    const bool A_terminal = events.wait("A", EventKind::terminal);
    driver.join();
    scheduler.shutdown();
    check(A_blocked && B_backend_done && A_terminal,
          "A callback was blocked and both backend results completed safely");
    check(B_terminal_early == async,
          "async isolates slow A callback; default synchronous path still HOL");
    check(before_release.completed_total >= (async ? 1u : 0u) &&
              events.count("A", EventKind::terminal) == 1 &&
              events.count("B", EventKind::terminal) == 1,
          "one terminal per job and snapshot finalization after own callback");
    check(events.order("A") ==
                  std::vector<EventKind>({EventKind::admitted,
                                          EventKind::token,
                                          EventKind::terminal}) &&
              events.order("B") ==
                  std::vector<EventKind>({EventKind::admitted,
                                          EventKind::token,
                                          EventKind::terminal}) &&
              events.per_job_overlap == 0,
          "strict admitted/token/terminal order and no same-job callback overlap");
    const auto A_usage = events.terminal_snapshot("A");
    const auto B_usage = events.terminal_snapshot("B");
    check(A_usage.prompt_consumed == 1 &&
              B_usage.prompt_consumed == 1 &&
              A_usage.generated == std::vector<int>({2}) &&
              B_usage.generated == std::vector<int>({2}) &&
              A_usage.error.empty() && B_usage.error.empty(),
          "terminal snapshots preserve prompt usage, generated tokens and error status");
    validate_backend(backend);
    std::printf("PASS callback HOL async=%d B_terminal_while_A_held=%d\n",
                async ? 1 : 0, B_terminal_early ? 1 : 0);
  }
}

void test_backpressure_cancel_is_acknowledged_before_next_A_step() {
  Gate gate;
  gate.block_A = true;
  gate.cancel_A_on_release = true;
  Backend backend;
  backend.gate = &gate;
  Events events;
  events.gate = &gate;
  ContinuousScheduler scheduler(config(events, true), backend);
  gate.scheduler = &scheduler;
  scheduler.enqueue({"A", {1}, 3});
  scheduler.enqueue({"B", {2}, 2});
  Driver driver(scheduler);
  const bool A_blocked = gate.wait_A();
  const bool B_terminated = events.wait("B", EventKind::terminal);
  const auto before_release = scheduler.metrics();
  const auto A_event_count = events.count("A", EventKind::token);
  const auto A_steps_before_release = [&] {
    std::lock_guard<std::mutex> guard(backend.mu);
    return std::count_if(backend.positions.begin(), backend.positions.end(),
                         [](const auto& item) { return item.first == 1; });
  }();
  gate.release_all();
  const bool A_terminated = events.wait("A", EventKind::terminal);
  driver.join();
  scheduler.shutdown();
  check(A_blocked && B_terminated && A_terminated &&
            before_release.completed_total == 1 &&
            A_event_count == 1 && A_steps_before_release == 1 &&
            gate.cancelled_by_backpressure == 1,
        "slow A token callback cancels A before another backend step; B advances");
  check(events.terminal("A") == JobStatus::cancelled &&
            events.terminal("B") == JobStatus::completed &&
            events.count("A", EventKind::token) == 1 &&
            events.count("B", EventKind::token) == 2 &&
            events.per_job_overlap == 0,
        "backpressure does not fabricate success, duplicate or reorder tokens");
  const auto B_usage = events.terminal_snapshot("B");
  check(B_usage.prompt_consumed == 1 &&
            B_usage.generated == std::vector<int>({2, 2}) &&
            B_usage.error.empty(),
        "fast B terminal payload matches both previously emitted tokens");
  validate_backend(backend);
}

void test_shutdown_waits_for_blocked_callback_then_joins() {
  Gate gate;
  gate.block_A = true;
  Backend backend;
  Events events;
  events.gate = &gate;
  ContinuousScheduler scheduler(config(events, true), backend);
  gate.scheduler = &scheduler;
  scheduler.enqueue({"A", {1}, 1});
  Driver driver(scheduler);
  const bool entered = gate.wait_A();
  driver.join();
  std::atomic<bool> started{false}, finished{false};
  std::mutex started_mu;
  std::condition_variable started_cv;
  std::thread stopping([&] {
    started = true;
    started_cv.notify_all();
    scheduler.shutdown();
    finished = true;
  });
  bool was_started;
  {
    std::unique_lock<std::mutex> lock(started_mu);
    was_started = started_cv.wait_for(
        lock, seconds(3), [&] { return started.load(); });
  }
  const bool held = !finished.load();
  gate.release_all();
  stopping.join();
  scheduler.shutdown(); // idempotent; no double-join
  check(entered && was_started && held && finished.load(),
        "shutdown waits for managed callback completion, then joins workers");
  check(events.wait("A", EventKind::terminal) &&
            events.count("A", EventKind::terminal) == 1,
        "shutdown terminal delivered exactly once");
  validate_backend(backend);
}

void test_throwing_sink_uses_one_terminal_emergency_and_cancel() {
  Backend backend;
  Events events;
  events.throw_on_B_token = true;
  auto c = config(events, true);
  std::atomic<int> emergency{0};
  std::atomic<bool> status_failed{false};
  std::mutex emergency_mu;
  std::condition_variable emergency_cv;
  c.on_terminal_delivery_failure =
      [&](const std::string& id, JobStatus status) noexcept {
        if (id == "B") {
          ++emergency;
          if (status == JobStatus::failed) status_failed = true;
          emergency_cv.notify_all();
        }
      };
  ContinuousScheduler scheduler(c, backend);
  scheduler.enqueue({"A", {1}, 1});
  scheduler.enqueue({"B", {2}, 3});
  Driver driver(scheduler);
  const bool A_terminal = events.wait("A", EventKind::terminal);
  // The B token sink deliberately throws. It may have logged the token
  // before throwing; it must receive no *later* normal B terminal.
  bool woke;
  {
    std::unique_lock<std::mutex> guard(emergency_mu);
    woke = emergency_cv.wait_for(
        guard, seconds(3), [&] { return emergency.load() == 1; });
  }
  const auto mid = scheduler.metrics();
  driver.join();
  scheduler.shutdown();
  check(A_terminal && woke && status_failed &&
            events.count("B", EventKind::terminal) == 0 &&
            events.count("A", EventKind::terminal) == 1,
        "B throwing token consumer gets one emergency, A unaffected");
  check(emergency == 1 && mid.terminal_delivery_failures_total >= 1 &&
            mid.callback_errors_total >= 1,
        "missing B token fails closed once and reports callback failure");
  validate_backend(backend);
}

void test_throwing_sink_ack_cannot_race_false_success() {
  Backend backend;
  Events events;
  auto c = config(events, true);
  std::mutex hook_mu, notifier_mu;
  std::condition_variable hook_cv, notifier_cv;
  bool hook_entered = false, hook_released = false;
  std::atomic<int> emergency{0};
  c.on_event = [&](const SchedulerEvent& event) {
    events.deliver(event);
    if (event.id == "A" && event.kind == EventKind::token)
      throw std::bad_alloc();
  };
  c.callback_before_failure_recovery = [&] {
    std::unique_lock<std::mutex> guard(hook_mu);
    hook_entered = true;
    hook_cv.notify_all();
    hook_cv.wait(guard, [&] { return hook_released; });
  };
  c.on_terminal_delivery_failure =
      [&](const std::string& id, JobStatus status) noexcept {
        if (id == "A" && status == JobStatus::failed) {
          ++emergency;
          notifier_cv.notify_all();
        }
      };
  ContinuousScheduler scheduler(c, backend);
  check(scheduler.enqueue({"A", {1}, 1}).code == SubmitCode::accepted,
        "adversarial callback race request accepted");
  Driver driver(scheduler);
  bool entered;
  {
    std::unique_lock<std::mutex> guard(hook_mu);
    entered = hook_cv.wait_for(guard, seconds(3),
                               [&] { return hook_entered; });
  }
  // Force another independent tick while the failing callback is frozen
  // exactly before cancellation. With the vulnerable ordering, the worker
  // would have released callback_unconsumed and tick() would mark completed.
  const std::size_t extra_work = scheduler.tick();
  const auto blocked = scheduler.metrics();
  JobSnapshot snapshot;
  const bool inspectable = scheduler.inspect("A", snapshot);
  const bool prematurely_closed = backend.close_count != 0;
  {
    std::lock_guard<std::mutex> guard(hook_mu);
    hook_released = true;
  }
  hook_cv.notify_all();
  bool woke;
  {
    std::unique_lock<std::mutex> guard(notifier_mu);
    woke = notifier_cv.wait_for(
        guard, seconds(3), [&] { return emergency.load() == 1; });
  }
  const bool closed = backend.wait_closed(1);
  driver.join();
  scheduler.shutdown();
  const auto done = scheduler.metrics();
  check(entered && extra_work == 0 && inspectable &&
            snapshot.status == JobStatus::active &&
            !prematurely_closed && blocked.completed_total == 0 &&
            blocked.failed_total == 0 && blocked.cancelled_total == 0,
        "unacknowledged throwing token cannot appear completed before poison");
  check(woke && closed && emergency == 1 &&
            done.completed_total == 0 && done.cancelled_total == 1 &&
            events.count("A", EventKind::terminal) == 0 &&
            done.terminal_delivery_failures_total >= 1,
        "after throw recovery: cancelled terminal state and exactly one emergency");
  validate_backend(backend);
}

void test_async_terminal_copy_failures_keep_transport_bounded() {
  for (bool fail_id : {false, true}) {
    Backend backend;
    Events events;
    auto c = config(events, true);
    std::atomic<int> emergency{0};
    std::mutex emergency_mu;
    std::condition_variable emergency_cv;
    c.on_terminal_delivery_failure =
        [&](const std::string& id, JobStatus status) noexcept {
          if (id == "A" && status == JobStatus::failed) {
            ++emergency;
            emergency_cv.notify_all();
          }
        };
    if (fail_id) {
      c.terminal_before_event_id_copy = [] { throw std::bad_alloc(); };
    } else {
      c.terminal_before_snapshot_copy = [] { throw std::bad_alloc(); };
    }
    ContinuousScheduler scheduler(c, backend);
    scheduler.enqueue({"A", {1}, 1});
    Driver driver(scheduler);
    bool reached;
    if (fail_id) {
      std::unique_lock<std::mutex> guard(emergency_mu);
      reached = emergency_cv.wait_for(
          guard, seconds(3), [&] { return emergency.load() == 1; });
    } else {
      reached = events.wait("A", EventKind::terminal);
    }
    driver.join();
    scheduler.shutdown();
    const auto metrics = scheduler.metrics();
    check(reached && metrics.completed_total == 1 &&
              metrics.terminal_delivery_failures_total >= 1 &&
              events.count("A", EventKind::token) == 1,
          "async terminal allocation failure preserves successful backend result");
    if (fail_id) {
      check(emergency == 1 && events.count("A", EventKind::terminal) == 0,
            "uncopyable ID uses exactly one emergency without reallocation");
    } else {
      const auto final = events.terminal_snapshot("A");
      check(emergency == 0 && final.status == JobStatus::failed &&
                final.generated.empty() && final.error.empty(),
            "uncopyable terminal payload sends exactly one safe failed terminal");
    }
    validate_backend(backend);
  }
}

void test_queued_cancel_isolated_from_blocked_A_callback() {
  Gate gate;
  gate.block_A = true;
  Backend backend;
  Events events;
  events.gate = &gate;
  ContinuousScheduler scheduler(config(events, true), backend);
  scheduler.enqueue({"A", {1}, 1});
  Driver driver(scheduler);
  const bool entered = gate.wait_A();
  const bool queued = scheduler.enqueue({"C", {3}, 1}).code ==
                      SubmitCode::accepted;
  const bool cancelled = scheduler.cancel("C");
  const bool C_terminal = events.wait("C", EventKind::terminal);
  const int A_terminal_before = events.count("A", EventKind::terminal);
  gate.release_all();
  const bool A_terminal = events.wait("A", EventKind::terminal);
  driver.join();
  scheduler.shutdown();
  check(entered && queued && cancelled && C_terminal && A_terminal &&
            A_terminal_before == 0 &&
            events.terminal("C") == JobStatus::cancelled,
        "queued cancelled C terminal bypasses blocked active A callback");
  check(events.count("C", EventKind::admitted) == 0 &&
            events.count("C", EventKind::token) == 0 &&
            events.count("C", EventKind::terminal) == 1 &&
            backend.open_count == 1,
        "queued C cancellation emits terminal without backend open or token");
  validate_backend(backend);
}

void test_bounded_mailbox_overflow_is_fail_closed() {
  Gate gate;
  gate.block_A = true;
  gate.block_B = true;
  Backend backend;
  backend.gate = &gate;
  Events events;
  events.gate = &gate;
  auto c = config(events, true);
  c.max_queued = 2;  // Both A and B must enter before cancellations start.
  std::mutex emergency_mu;
  std::unordered_map<std::string, int> emergency;
  c.on_terminal_delivery_failure =
      [&](const std::string& id, JobStatus) noexcept {
        std::lock_guard<std::mutex> guard(emergency_mu);
        ++emergency[id];
      };
  ContinuousScheduler scheduler(c, backend);
  gate.scheduler = &scheduler;
  check(scheduler.enqueue({"A", {1}, 3}).code == SubmitCode::accepted,
        "both gate owners: A enqueued");
  check(scheduler.enqueue({"B", {2}, 3}).code == SubmitCode::accepted,
        "both gate owners: B enqueued");
  Driver driver(scheduler);
  const bool both_callbacks_blocked = gate.wait_A() && gate.wait_B();
  std::vector<std::string> queued_ids;
  for (int i = 0; i < 72; ++i) {
    const std::string id = "Q" + std::to_string(i);
    queued_ids.push_back(id);
    check(scheduler.enqueue({id, {3}, 1}).code == SubmitCode::accepted,
          "bounded queued request accepted after previous cancellation");
    check(scheduler.cancel(id), "queued cancel terminal submitted to mailbox");
  }
  std::size_t emergency_count_before = 0;
  {
    std::lock_guard<std::mutex> guard(emergency_mu);
    for (const auto& item : emergency) emergency_count_before += item.second;
  }
  gate.release_all();
  driver.join();
  scheduler.shutdown();
  std::size_t delivered_cancelled = 0, overflow_failed = 0;
  for (const auto& id : queued_ids) {
    const int count = events.count(id, EventKind::terminal);
    int failed = 0;
    {
      std::lock_guard<std::mutex> guard(emergency_mu);
      const auto found = emergency.find(id);
      if (found != emergency.end()) failed = found->second;
    }
    check(count + failed == 1,
          "every queued cancellation gets exactly one normal terminal or fail-closed emergency");
    delivered_cancelled += static_cast<std::size_t>(count);
    overflow_failed += static_cast<std::size_t>(failed);
  }
  const auto metrics = scheduler.metrics();
  check(both_callbacks_blocked && emergency_count_before >= 8 &&
            overflow_failed >= 8 && delivered_cancelled <= 64,
        "fixed 64-slot callback queue overflows deterministically and wakes clients");
  check(metrics.accepted_total == 74 &&
            metrics.cancelled_total >= 72 &&
            metrics.cancelled_total + metrics.completed_total == 74 &&
            metrics.terminal_delivery_failures_total >= overflow_failed,
        "bounded queue overflow and shutdown cancellations are accounted");
  for (const char* id : {"A", "B"}) {
    int failed = 0;
    {
      std::lock_guard<std::mutex> guard(emergency_mu);
      const auto it = emergency.find(id);
      if (it != emergency.end()) failed = it->second;
    }
    check(events.count(id, EventKind::terminal) + failed == 1,
          "even shutdown racing a full callback queue gives one terminal or emergency");
  }
  validate_backend(backend);
}

void test_fail_closed_invalid_optin_and_spawn_rollback() {
  Backend backend;
  Events events;
  {
    auto c = config(events, true);
    c.on_terminal_delivery_failure = {};
    bool rejected = false;
    try { ContinuousScheduler forbidden(c, backend); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "async requires guaranteed borrowed-ID emergency notifier");
  }
  {
    auto c = config(events, true);
    c.max_active = 1;
    bool rejected = false;
    try { ContinuousScheduler forbidden(c, backend); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "async requires at least two independent callback workers");
  }
  {
    auto c = config(events, true);
    c.callback_before_worker_spawn = [](std::size_t index) {
      if (index == 1) throw std::bad_alloc();
    };
    bool caught = false;
    try { ContinuousScheduler partial(c, backend); }
    catch (const std::bad_alloc&) { caught = true; }
    check(caught, "partial dispatcher startup joins started workers before rethrow");
  }
}
}  // namespace

int main() {
  try {
    test_fail_closed_invalid_optin_and_spawn_rollback();
    test_blocked_A_callback_does_not_block_B_in_optin();
    test_backpressure_cancel_is_acknowledged_before_next_A_step();
    test_shutdown_waits_for_blocked_callback_then_joins();
    test_throwing_sink_uses_one_terminal_emergency_and_cancel();
    test_throwing_sink_ack_cannot_race_false_success();
    test_async_terminal_copy_failures_keep_transport_bounded();
    test_queued_cancel_isolated_from_blocked_A_callback();
    test_bounded_mailbox_overflow_is_fail_closed();
    std::printf("PASS ready callback isolation CPU checks=%d\n", checks);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL ready callback isolation: %s\n", e.what());
    return 1;
  }
}
