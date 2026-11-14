// HERMES-27: actual ContinuousScheduler CPU-only wave/event HOL witness.
//
// Compile with unchanged PP2 source:
// g++ -std=c++17 -O1 -g -Wall -Wextra -Werror -pedantic -pthread -Idistributed
//   -DHERMES_ADMISSION_CPU_ONLY -x c++ distributed/pp2_batch.cu
//   distributed/tests/test_throughput_stream_event_scheduler.cpp -o /tmp/stream_event
//
// No CUDA context, GGUF, HTTP, services, or implementation edits.
// This is intentionally a witness of CURRENT behavior, not a throughput test.
#include "pp2_batch.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
using namespace pp2batch;
using namespace std::chrono;

void check(bool yes, const char* name) {
  if (!yes) throw std::runtime_error(name);
}

// Opened states are exclusive; a slow state deliberately stops inside
// backend.step while all fast requests return. Step numbers are deterministic
// and never use a clock to decide whether the backend finishes successfully.
class ControlledBackend final : public StepBackend {
 public:
  struct State {
    int serial = 0;
    int next_pos = 0;
  };

  explicit ControlledBackend(int expected_workers) : expected_workers_(expected_workers) {}

  bool supports_parallel_steps() const noexcept override { return true; }
  bool open(int, void*& opaque, std::string&) override {
    opaque = new State{++opens_, 0};
    return true;
  }
  bool step(void* opaque, int, long pos, std::vector<float>& logits,
            std::string& error) override {
    auto& state = *static_cast<State*>(opaque);
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (inflight_[state.serial]) {
        error = "duplicate in-flight state";
        return false;
      }
      inflight_[state.serial] = true;
      ++entered_;
      peak_ = std::max(peak_, entered_ - finished_);
      cv_.notify_all();
      // Ensure all N steps are demonstrably in-flight simultaneously,
      // regardless of host thread scheduling order or CPU core count.
      if (!cv_.wait_for(lock, seconds(3), [&] { return entered_ == expected_workers_; })) {
        error = "not all backend workers entered in time";
        return false;
      }
      if (state.serial == 1) {
        cv_.wait(lock, [&] { return release_slow_; });
      }
      if (pos != state.next_pos) {
        error = "duplicate/out-of-order absolute token position";
        inflight_[state.serial] = false;
        ++finished_;
        cv_.notify_all();
        return false;
      }
      ++state.next_pos;
      inflight_[state.serial] = false;
      ++finished_;
      completed_ids_.push_back(state.serial);
      cv_.notify_all();
    }
    logits.assign(8, -1.f);
    logits[2] = 1.f;
    return true;
  }
  void close(void* opaque) noexcept override {
    delete static_cast<State*>(opaque);
    ++closes_;
  }
  bool wait_for_fast(int n) {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, seconds(3), [&] {
      if (entered_ != n) return false;
      return std::count_if(completed_ids_.begin(), completed_ids_.end(),
                           [](int serial) { return serial != 1; }) == n - 1;
    });
  }
  void release_slow() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      release_slow_ = true;
    }
    cv_.notify_all();
  }
  int peak() const {
    std::lock_guard<std::mutex> lock(mu_);
    return peak_;
  }
  int opens() const { return opens_.load(); }
  int closes() const { return closes_.load(); }
 private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::map<int, bool> inflight_;
  std::vector<int> completed_ids_;
  int entered_ = 0, finished_ = 0, peak_ = 0;
  int expected_workers_ = 0;
  bool release_slow_ = false;
  std::atomic<int> opens_{0}, closes_{0};
};

struct CapturedEvents {
  std::mutex mu;
  std::vector<std::string> events;
  void add(const SchedulerEvent& event) {
    std::lock_guard<std::mutex> lock(mu);
    const char* kind = event.kind == EventKind::admitted ? "admitted" :
                       event.kind == EventKind::token ? "token" : "terminal";
    events.push_back(event.id + ":" + kind);
  }
  std::vector<std::string> get() {
    std::lock_guard<std::mutex> lock(mu);
    return events;
  }
};

SchedulerConfig config(int max_active, CapturedEvents& events) {
  SchedulerConfig c;
  c.max_active = static_cast<size_t>(max_active);
  c.max_queued = 8;
  c.max_context = 64;
  c.steps_per_tick = static_cast<size_t>(max_active);
  c.queue_timeout = seconds(30);
  c.vocab_size = 8;
  c.eos_token_id = 7;
  c.enable_parallel_steps = true;
  c.on_event = [&](const SchedulerEvent& event) { events.add(event); };
  return c;
}

void witness_wave_join(int n, bool cancel_slow) {
  ControlledBackend backend(n);
  CapturedEvents captured;
  ContinuousScheduler scheduler(config(n, captured), backend);
  for (int i = 0; i < n; ++i) {
    const auto submitted = scheduler.enqueue(
        {"P" + std::to_string(i), {i + 1}, 1});
    check(submitted.code == SubmitCode::accepted, "bounded FIFO enqueue");
  }

  std::atomic<size_t> steps{0};
  std::exception_ptr pump_error;
  std::thread pump([&] {
    try { steps = scheduler.tick(); }
    catch (...) { pump_error = std::current_exception(); }
  });
  // Avoid std::terminate on any assertion: release/join before checking
  // results. An unexpected backend deadlock is a test failure, not a success.
  const bool fast_finished = backend.wait_for_fast(n);
  std::vector<std::string> before = captured.get();
  bool fast_token_before_release = false;
  for (int i = 1; i < n; ++i)
    fast_token_before_release |=
        std::find(before.begin(), before.end(),
                  "P" + std::to_string(i) + ":token") != before.end();
  bool slow_cancelled = false;
  if (cancel_slow) slow_cancelled = scheduler.cancel("P0");
  backend.release_slow();
  pump.join();
  if (pump_error) std::rethrow_exception(pump_error);

  check(fast_finished, "all fast backend calls completed while one slow call was blocked");
  check(!fast_token_before_release,
        "CURRENT source joins whole wave before delivering any ready fast token");
  check(backend.peak() == n, "all host backend steps did truly overlap");
  check(steps == static_cast<size_t>(n), "one step per unique state");
  check(backend.opens() == n && backend.closes() == n,
        "one open/close per distinct request state");

  std::vector<std::string> after = captured.get();
  for (int i = 0; i < n; ++i) {
    JobSnapshot job;
    const bool observed = scheduler.inspect("P" + std::to_string(i), job);
    check(observed, "terminal job retained after wave");
    if (cancel_slow && i == 0) {
      check(slow_cancelled, "in-flight cancellation responsive under pump join");
      check(job.status == JobStatus::cancelled && job.generated.empty(),
            "cancelled slow token discarded before state freed");
      check(std::find(after.begin(), after.end(), "P0:token") == after.end(),
            "cancelled token must not reach SSE");
    } else {
      check(job.status == JobStatus::completed &&
            job.prompt_consumed == 1 && job.generated == std::vector<int>({2}),
            "fast request exact prompt+completion state");
    }
    check(std::count(after.begin(), after.end(),
                     "P" + std::to_string(i) + ":terminal") == 1,
          "exactly one terminal event per request");
  }
  std::printf("PASS current-wave-join N=%d cancel_slow=%d peak=%d "
              "fast_results_blocked_until_slow_return=1\n",
              n, cancel_slow ? 1 : 0, backend.peak());
}

// Even with all GPU steps finished, synchronous callbacks may block the
// scheduler's single committing pump. The slow callback here represents an
// injected tokenizer IPC stall, NOT a claim that the real IPC is slow.
void witness_callback_head_of_line() {
  struct ImmediateBackend final : StepBackend {
    struct State { int serial; };
    std::atomic<int> count{0}, closes{0};
    bool supports_parallel_steps() const noexcept override { return true; }
    bool open(int, void*& ptr, std::string&) override {
      ptr = new State{++count};
      return true;
    }
    bool step(void*, int, long, std::vector<float>& logits, std::string&) override {
      logits.assign(8, -1.f);
      logits[2] = 1.f;
      return true;
    }
    void close(void* ptr) noexcept override { delete static_cast<State*>(ptr); ++closes; }
  } backend;

  std::mutex mu;
  std::condition_variable cv;
  bool callback_entered = false, callback_release = false;
  CapturedEvents events;
  auto cfg = config(2, events);
  cfg.on_event = [&](const SchedulerEvent& event) {
    events.add(event);
    if (event.kind == EventKind::token && event.id == "P0") {
      std::unique_lock<std::mutex> lock(mu);
      callback_entered = true;
      cv.notify_all();
      cv.wait(lock, [&] { return callback_release; });
    }
  };
  ContinuousScheduler scheduler(cfg, backend);
  check(scheduler.enqueue({"P0", {1}, 1}).code == SubmitCode::accepted, "P0 enqueue");
  check(scheduler.enqueue({"P1", {2}, 1}).code == SubmitCode::accepted, "P1 enqueue");
  std::thread pump([&] { scheduler.tick(); });
  bool entered;
  {
    std::unique_lock<std::mutex> lock(mu);
    entered = cv.wait_for(lock, seconds(3), [&] { return callback_entered; });
  }
  const auto before = events.get();
  const bool fast_sse_blocked =
      std::find(before.begin(), before.end(), "P1:token") == before.end();
  const bool cancelled = scheduler.cancel("P1");
  {
    std::lock_guard<std::mutex> lock(mu);
    callback_release = true;
  }
  cv.notify_all();
  pump.join();
  check(entered, "controlled callback entered");
  check(fast_sse_blocked, "synchronous token callback blocks sibling ready output");
  check(cancelled, "cancel can still mark sibling while callback blocks pump");
  JobSnapshot sibling;
  check(scheduler.inspect("P1", sibling) && sibling.status == JobStatus::cancelled &&
        sibling.generated.empty(), "sibling token correctly dropped after cancellation");
  check(backend.closes == 2, "callback block does not double-free PP2 states");
  std::printf("PASS current-synchronous-callback stalls sibling token delivery\n");
}
} // namespace

int main() {
  try {
    witness_wave_join(2, false);
    witness_wave_join(4, false);
    witness_wave_join(4, true);
    witness_callback_head_of_line();
    std::puts("PASS actual ContinuousScheduler CPU event-blocking witness: 4 cases");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL stream-event scheduler witness: %s\n", e.what());
    return 1;
  }
}
