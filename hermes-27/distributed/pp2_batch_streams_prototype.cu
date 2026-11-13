#include "pp2_batch_streams_prototype.hpp"

#include <cstring>
#include <string>
#include <utility>

namespace pp2streams {
namespace {

// This prototype does not permanently change the device of its caller.
struct RestoreDevice {
  int original = -1;
  bool valid = cudaGetDevice(&original) == cudaSuccess;
  ~RestoreDevice() {
    if (valid) (void)cudaSetDevice(original);
  }
};

} // namespace

bool TwoStagePipeline::checked(cudaError_t status, const char* operation,
                               std::string& err) {
  if (status == cudaSuccess) return true;
  err = std::string(operation) + ": " + cudaGetErrorString(status);
  return false;
}

bool TwoStagePipeline::initialize(const std::vector<DeviceBuffers>& buffers,
                                  std::string& err) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_ || !slots_.empty()) {
    err = "pipeline already initialized; call shutdown first";
    return false;
  }
  if (config_.gpu0 < 0 || config_.gpu1 < 0 || config_.gpu0 == config_.gpu1 ||
      config_.hidden_bytes == 0 || config_.logits_bytes == 0 || buffers.empty()) {
    err = "two different GPUs, nonzero buffer sizes and requests are required";
    return false;
  }
  for (const auto& b : buffers) {
    if (!b.gpu0_hidden || !b.gpu1_hidden || !b.gpu1_logits) {
      err = "request device buffers must all be non-null";
      return false;
    }
  }

  RestoreDevice restore;
  slots_.resize(buffers.size());
  for (std::size_t i = 0; i < slots_.size(); ++i) {
    Slot& slot = slots_[i];
    slot.device = buffers[i];
    if (!checked(cudaSetDevice(config_.gpu0), "select GPU0", err) ||
        !checked(cudaStreamCreateWithFlags(&slot.stream0, cudaStreamNonBlocking),
                 "GPU0 stream", err) ||
        !checked(cudaEventCreateWithFlags(&slot.stage0_copied, cudaEventDisableTiming),
                 "GPU0 copy event", err) ||
        !checked(cudaHostAlloc(&slot.pinned_hidden, config_.hidden_bytes,
                               cudaHostAllocDefault), "pinned hidden", err) ||
        !checked(cudaSetDevice(config_.gpu1), "select GPU1", err) ||
        !checked(cudaStreamCreateWithFlags(&slot.stream1, cudaStreamNonBlocking),
                 "GPU1 stream", err) ||
        !checked(cudaEventCreateWithFlags(&slot.stage1_copied, cudaEventDisableTiming),
                 "GPU1 logits event", err) ||
        !checked(cudaHostAlloc(&slot.pinned_logits, config_.logits_bytes,
                               cudaHostAllocDefault), "pinned logits", err)) {
      // Even partially initialized resources are cleaned, after any streams drain.
      shutdown_unlocked();
      return false;
    }
  }
  initialized_ = true;
  return true;
}

bool TwoStagePipeline::validate(std::size_t id, Phase expected,
                                 std::string& err) const {
  if (!initialized_ || id >= slots_.size()) {
    err = "pipeline not initialized or request id out of range";
    return false;
  }
  if (slots_[id].phase != expected) {
    err = "request slot has wrong phase or is already in flight";
    return false;
  }
  return true;
}

bool TwoStagePipeline::submit_stage0(std::size_t id, Stage0Enqueue enqueue,
                                      void* user, std::string& err) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (!validate(id, Phase::Idle, err) || !enqueue) {
    if (!enqueue) err = "stage0 callback is null";
    return false;
  }
  RestoreDevice restore;
  Slot& slot = slots_[id];
  slot.phase = Phase::Fault; // Fail closed on partially enqueued work.
  if (!checked(cudaSetDevice(config_.gpu0), "select GPU0", err) ||
      !checked(enqueue(slot.stream0, slot.device.gpu0_hidden, user),
               "enqueue stage0", err) ||
      !checked(cudaMemcpyAsync(slot.pinned_hidden, slot.device.gpu0_hidden,
                               config_.hidden_bytes, cudaMemcpyDeviceToHost,
                               slot.stream0), "GPU0 hidden D2H", err) ||
      !checked(cudaEventRecord(slot.stage0_copied, slot.stream0),
               "record GPU0 copy event", err)) return false;
  slot.phase = Phase::Stage0;
  return true;
}

bool TwoStagePipeline::try_submit_stage1(std::size_t id, Stage1Enqueue enqueue,
                                          void* user, bool& submitted,
                                          std::string& err) {
  const std::lock_guard<std::mutex> lock(mutex_);
  submitted = false;
  if (!validate(id, Phase::Stage0, err) || !enqueue) {
    if (!enqueue) err = "stage1 callback is null";
    return false;
  }
  RestoreDevice restore;
  Slot& slot = slots_[id];
  if (!checked(cudaSetDevice(config_.gpu0), "select GPU0", err)) {
    slot.phase = Phase::Fault;
    return false;
  }
  const cudaError_t query = cudaEventQuery(slot.stage0_copied);
  if (query == cudaErrorNotReady) return true;
  if (!checked(query, "GPU0 copy completion", err)) {
    slot.phase = Phase::Fault;
    return false;
  }
  slot.phase = Phase::Fault;
  // Only after GPU0's D2H event can GPU1 read the pinned boundary.
  // GPU1 H2D and all subsequent kernels/copies use this request's stream.
  if (!checked(cudaSetDevice(config_.gpu1), "select GPU1", err) ||
      !checked(cudaMemcpyAsync(slot.device.gpu1_hidden, slot.pinned_hidden,
                               config_.hidden_bytes, cudaMemcpyHostToDevice,
                               slot.stream1), "GPU1 hidden H2D", err) ||
      !checked(enqueue(slot.stream1, slot.device.gpu1_hidden,
                       slot.device.gpu1_logits, user), "enqueue stage1", err) ||
      !checked(cudaMemcpyAsync(slot.pinned_logits, slot.device.gpu1_logits,
                               config_.logits_bytes, cudaMemcpyDeviceToHost,
                               slot.stream1), "GPU1 logits D2H", err) ||
      !checked(cudaEventRecord(slot.stage1_copied, slot.stream1),
               "record GPU1 logits event", err)) return false;
  slot.phase = Phase::Stage1;
  submitted = true;
  return true;
}

bool TwoStagePipeline::try_collect(std::size_t id,
                                    std::vector<std::uint8_t>& output,
                                    bool& ready, std::string& err) {
  const std::lock_guard<std::mutex> lock(mutex_);
  ready = false;
  if (!validate(id, Phase::Stage1, err)) return false;
  RestoreDevice restore;
  Slot& slot = slots_[id];
  if (!checked(cudaSetDevice(config_.gpu1), "select GPU1", err)) {
    slot.phase = Phase::Fault;
    return false;
  }
  const cudaError_t query = cudaEventQuery(slot.stage1_copied);
  if (query == cudaErrorNotReady) return true;
  if (!checked(query, "GPU1 logits completion", err)) {
    slot.phase = Phase::Fault;
    return false;
  }
  const auto* bytes = static_cast<const std::uint8_t*>(slot.pinned_logits);
  output.assign(bytes, bytes + config_.logits_bytes);
  // At this point all writes to this request's GDN/KV buffers and pinned
  // staging buffers are complete. The next token may safely reuse the slot.
  slot.phase = Phase::Idle;
  ready = true;
  return true;
}

TwoStagePipeline::Phase TwoStagePipeline::phase(std::size_t id) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return id < slots_.size() ? slots_[id].phase : Phase::Fault;
}

std::size_t TwoStagePipeline::size() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return slots_.size();
}

void TwoStagePipeline::shutdown() noexcept {
  const std::lock_guard<std::mutex> lock(mutex_);
  shutdown_unlocked();
}

void TwoStagePipeline::shutdown_unlocked() noexcept {
  RestoreDevice restore;
  // If an error happened after enqueueing work, drain before freeing any
  // buffers that a pending CUDA operation could still reference.
  for (auto& slot : slots_) {
    (void)cudaSetDevice(config_.gpu0);
    if (slot.stream0) (void)cudaStreamSynchronize(slot.stream0);
    (void)cudaSetDevice(config_.gpu1);
    if (slot.stream1) (void)cudaStreamSynchronize(slot.stream1);
  }
  for (auto& slot : slots_) {
    (void)cudaSetDevice(config_.gpu0);
    if (slot.stage0_copied) (void)cudaEventDestroy(slot.stage0_copied);
    if (slot.stream0) (void)cudaStreamDestroy(slot.stream0);
    (void)cudaSetDevice(config_.gpu1);
    if (slot.stage1_copied) (void)cudaEventDestroy(slot.stage1_copied);
    if (slot.stream1) (void)cudaStreamDestroy(slot.stream1);
    if (slot.pinned_hidden) (void)cudaFreeHost(slot.pinned_hidden);
    if (slot.pinned_logits) (void)cudaFreeHost(slot.pinned_logits);
  }
  slots_.clear();
  initialized_ = false;
}

} // namespace pp2streams
