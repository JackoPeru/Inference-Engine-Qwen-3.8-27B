// HERMES-27 CPU-only terminal OOM/error safety regression. No CUDA/model/HTTP.
// The initial RED test requires fail-closed emergency wakeup if allocating
// the first terminal event ID fails, without a second allocating ID retry.
#include "pp2_batch.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace pp2batch;

int checks = 0;
void require(bool condition, const char* message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

struct Backend final : StepBackend {
  struct State { bool in_step = false; };
  std::atomic<int> opens{0}, closes{0}, closed_in_step{0}, steps{0};
  bool fail_step = false;
  bool supports_parallel_steps() const noexcept override { return true; }
  bool open(int, void*& ptr, std::string&) override {
    ptr = new State();
    ++opens;
    return true;
  }
  bool step(void* ptr, int, long, std::vector<float>& logits,
            std::string& error) override {
    auto& state = *static_cast<State*>(ptr);
    if (state.in_step) throw std::logic_error("state overlap");
    state.in_step = true;
    ++steps;
    if (fail_step) {
      error = std::string(160, 'e');
      state.in_step = false;
      return false;
    }
    logits.assign(8, -1.f);
    logits[2] = 1.f;
    state.in_step = false;
    return true;
  }
  void close(void* ptr) noexcept override {
    auto* state = static_cast<State*>(ptr);
    if (state->in_step) ++closed_in_step;
    delete state;
    ++closes;
  }
};

SchedulerConfig configure() {
  SchedulerConfig c;
  c.vocab_size = 8;
  c.max_context = 32;
  c.max_active = 1;
  c.max_queued = 4;
  c.steps_per_tick = 1;
  c.enable_parallel_steps = true;
  c.enable_ready_completion_dispatch = true;
  return c;
}

void test_id_copy_oom_uses_borrowed_id_without_second_allocation() {
  Backend backend;
  auto config = configure();
  const std::string id = "terminal-id-" + std::string(96, 'x');
  std::atomic<int> terminal_attempts{0}, admitted{0}, tokens{0};
  std::atomic<int> emergency_calls{0}, id_copy_faults{0};
  std::atomic<bool> callback_can_reenter{false};
  ContinuousScheduler* ptr = nullptr;

  config.terminal_before_event_id_copy = [&] {
    ++id_copy_faults;
    throw std::bad_alloc();
  };
  config.on_event = [&](const SchedulerEvent& event) {
    if (event.kind == EventKind::admitted) ++admitted;
    if (event.kind == EventKind::token) ++tokens;
    if (event.kind == EventKind::terminal) ++terminal_attempts;
  };
  config.on_terminal_delivery_failure =
      [&](const std::string& borrowed_id, JobStatus status) noexcept {
        if (borrowed_id == id && status == JobStatus::completed) {
          ++emergency_calls;
          // Called outside scheduler metadata lock; these APIs must not
          // deadlock even when terminal payload construction has failed.
          const auto snapshot = ptr->metrics();
          JobSnapshot output;
          callback_can_reenter = snapshot.completed_total == 1 &&
                                 ptr->inspect(id, output) &&
                                 output.status == JobStatus::completed;
        }
      };

  ContinuousScheduler scheduler(config, backend);
  ptr = &scheduler;
  require(scheduler.enqueue({id, {1}, 1}).code == SubmitCode::accepted,
          "accepted request with heap-backed ID");
  require(scheduler.tick() == 1, "one backend decode step processed");
  const auto metrics = scheduler.metrics();
  require(id_copy_faults == 1 && admitted == 1 && tokens == 1,
          "fault was after admission and token, only on terminal ID copy");
  require(terminal_attempts == 0 && emergency_calls == 1 &&
              callback_can_reenter,
          "no allocation retry after ID OOM; emergency borrowed ID wakes");
  require(metrics.completed_total == 1 && metrics.failed_total == 0 &&
              metrics.callback_errors_total == 1 &&
              metrics.terminal_delivery_failures_total == 1,
          "backend success preserved; terminal transport failure counted once");
  require(backend.opens == 1 && backend.closes == 1 &&
              backend.steps == 1 && backend.closed_in_step == 0,
          "backend state closed only after step worker joined");
  require(!scheduler.cancel(id),
          "completed job cannot be cancelled/replayed after terminal fault");
}

void test_id_copy_oom_without_emergency_preserves_best_effort_terminal() {
  Backend backend;
  auto config = configure();
  const std::string id = "standalone-" + std::string(96, 's');
  int injected = 0, terminals = 0;
  JobStatus seen = JobStatus::queued;
  std::string received_id;
  config.terminal_before_event_id_copy = [&] {
    ++injected;
    throw std::bad_alloc();
  };
  // No on_terminal_delivery_failure: standalone scheduler clients should
  // still receive the minimal failed event when a later ID copy can succeed.
  config.on_event = [&](const SchedulerEvent& event) {
    if (event.kind != EventKind::terminal) return;
    ++terminals;
    seen = event.status;
    received_id = event.id;
  };
  ContinuousScheduler scheduler(config, backend);
  require(scheduler.enqueue({id, {1}, 1}).code == SubmitCode::accepted,
          "standalone request accepted");
  require(scheduler.tick() == 1, "standalone backend step finished");
  const auto metrics = scheduler.metrics();
  require(injected == 1 && terminals == 1 && received_id == id &&
              seen == JobStatus::failed,
          "without emergency hook a transient OOM retains minimal terminal");
  require(metrics.completed_total == 1 &&
              metrics.terminal_delivery_failures_total == 1 &&
              backend.opens == backend.closes &&
              backend.closed_in_step == 0,
          "standalone fallback preserves state lifetime and error metric");
}

void test_history_and_snapshot_oom_still_deliver_one_failed_terminal() {
  Backend backend;
  auto config = configure();
  const std::string id = "snapshot-id-" + std::string(96, 'y');
  int terminal_attempts = 0, emergency = 0, history_faults = 0;
  JobStatus delivered = JobStatus::queued;
  std::vector<int> generated;
  config.terminal_before_history_append = [&] {
    ++history_faults;
    throw std::bad_alloc();
  };
  config.terminal_before_snapshot_copy = [] { throw std::bad_alloc(); };
  config.on_event = [&](const SchedulerEvent& event) {
    if (event.kind != EventKind::terminal) return;
    ++terminal_attempts;
    delivered = event.status;
    generated = event.generated;
  };
  config.on_terminal_delivery_failure =
      [&](const std::string&, JobStatus) noexcept { ++emergency; };
  ContinuousScheduler scheduler(config, backend);
  require(scheduler.enqueue({id, {1}, 1}).code == SubmitCode::accepted,
          "second test accepted");
  require(scheduler.tick() == 1, "second backend decode processed");
  const auto metrics = scheduler.metrics();
  require(terminal_attempts == 1 && emergency == 0 &&
              delivered == JobStatus::failed && generated.empty(),
          "snapshot copy failure produces one safe minimal failed terminal");
  require(metrics.terminal_history_failures_total == 1 &&
              metrics.terminal_delivery_failures_total == 1 &&
              metrics.completed_total == 1,
          "history/snapshot errors tracked without changing backend result");
  require(backend.opens == backend.closes && backend.closed_in_step == 0,
          "double metadata OOM never leaks backend state");
  JobSnapshot forgotten;
  require(!scheduler.inspect(id, forgotten),
          "history failure does not leave unbounded retained terminal");
}

void test_throwing_terminal_sink_once_then_emergency() {
  Backend backend;
  backend.fail_step = true;
  auto config = configure();
  std::atomic<int> attempts{0}, emergency{0};
  const std::string id = "backend-failure";
  config.on_event = [&](const SchedulerEvent& event) {
    if (event.kind == EventKind::terminal) {
      ++attempts;
      throw std::bad_alloc();
    }
  };
  config.on_terminal_delivery_failure =
      [&](const std::string& borrowed_id, JobStatus status) noexcept {
        if (borrowed_id == id && status == JobStatus::failed) ++emergency;
      };
  ContinuousScheduler scheduler(config, backend);
  require(scheduler.enqueue({id, {1}, 1}).code == SubmitCode::accepted,
          "failed backend request accepted");
  require(scheduler.tick() == 1, "failed backend step still accounted");
  const auto metrics = scheduler.metrics();
  require(attempts == 1 && emergency == 1 &&
              metrics.callback_errors_total == 1 &&
              metrics.terminal_delivery_failures_total == 1 &&
              metrics.failed_total == 1 && metrics.completed_total == 0,
          "throwing terminal callback causes exactly one emergency wake");
  require(backend.opens == backend.closes && backend.closed_in_step == 0,
          "failed backend state closed once before callback");
}
}  // namespace

int main() {
  try {
    test_id_copy_oom_uses_borrowed_id_without_second_allocation();
    test_id_copy_oom_without_emergency_preserves_best_effort_terminal();
    test_history_and_snapshot_oom_still_deliver_one_failed_terminal();
    test_throwing_terminal_sink_once_then_emergency();
    std::printf("PASS ready terminal exception CPU checks=%d\n", checks);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL ready terminal exception: %s\n", e.what());
    return 1;
  }
}
