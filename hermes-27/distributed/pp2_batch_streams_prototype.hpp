#pragma once

// Standalone two-GPU stream/event scheduler prototype. No dependency on PP2
// internals and no changes to the production decode path.
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace pp2streams {

struct DeviceBuffers {
  // Caller owns these buffers for the complete lifetime of the pipeline.
  // Every request needs its own buffers and device-side recurrent state.
  void* gpu0_hidden = nullptr; // stage 0 writes hidden on GPU 0
  void* gpu1_hidden = nullptr; // stage 1 reads hidden on GPU 1
  void* gpu1_logits = nullptr; // stage 1 writes logits on GPU 1
};

using Stage0Enqueue = cudaError_t (*)(cudaStream_t stream, void* gpu0_hidden,
                                      void* user);
using Stage1Enqueue = cudaError_t (*)(cudaStream_t stream, void* gpu1_hidden,
                                      void* gpu1_logits, void* user);

class TwoStagePipeline {
 public:
  struct Config {
    int gpu0 = 0;
    int gpu1 = 1;
    std::size_t hidden_bytes = 5120u * sizeof(float);
    std::size_t logits_bytes = 248320u * sizeof(float);
  };
  enum class Phase : unsigned char { Idle, Stage0, Stage1, Fault };

  explicit TwoStagePipeline(Config config) : config_(config) {}
  ~TwoStagePipeline() { shutdown(); }
  TwoStagePipeline(const TwoStagePipeline&) = delete;
  TwoStagePipeline& operator=(const TwoStagePipeline&) = delete;
  TwoStagePipeline(TwoStagePipeline&&) = delete;
  TwoStagePipeline& operator=(TwoStagePipeline&&) = delete;

  // Allocates per-request nonblocking streams/events and pinned host buffers.
  // DeviceBuffers must remain alive and distinct across concurrent requests.
  bool initialize(const std::vector<DeviceBuffers>& buffers, std::string& err);

  // A callback only ENQUEUES operations on the provided stream and returns;
  // kernel launches must use <<<..., 0, stream>>> and return launch errors.
  bool submit_stage0(std::size_t id, Stage0Enqueue enqueue, void* user,
                     std::string& err);

  // Nonblocking host poll. When GPU0 D2H is complete, enqueue GPU1 H2D,
  // compute, logits D2H, and GPU1 completion event in the request stream.
  bool try_submit_stage1(std::size_t id, Stage1Enqueue enqueue, void* user,
                         bool& submitted, std::string& err);

  // Nonblocking host poll. Output is copied only after GPU1 event completes.
  // The slot becomes reusable once ready=true. No overlapping tokens in slot.
  bool try_collect(std::size_t id, std::vector<std::uint8_t>& output,
                   bool& ready, std::string& err);

  Phase phase(std::size_t id) const;
  std::size_t size() const;
  void shutdown() noexcept;

 private:
  struct Slot {
    DeviceBuffers device;
    cudaStream_t stream0 = nullptr;
    cudaStream_t stream1 = nullptr;
    cudaEvent_t stage0_copied = nullptr;
    cudaEvent_t stage1_copied = nullptr;
    void* pinned_hidden = nullptr;
    void* pinned_logits = nullptr;
    Phase phase = Phase::Idle;
  };

  bool validate(std::size_t id, Phase expected, std::string& err) const;
  void shutdown_unlocked() noexcept;
  static bool checked(cudaError_t status, const char* operation,
                      std::string& err);
  Config config_;
  // Protects slot phase/lifetime against simultaneous host callers. This
  // mutex is not a device synchronization barrier; all GPU work remains async.
  mutable std::mutex mutex_;
  std::vector<Slot> slots_;
  bool initialized_ = false;
};

} // namespace pp2streams
