// Synthetic two-GPU event/stream correctness test. NEVER asserts model parity.
// Compile separately; do not run while any other agent owns the GPUs.
#include "pp2_batch_streams_prototype.hpp"

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr int kRequests = 4;
constexpr int kHidden = 5120;
constexpr int kLogits = 5120; // Small synthetic stand-in, not the 248320-vocab head.

struct Context {
  int seed = 0;
  int bias = 0;
};

__global__ void stage0_kernel(float* dst, int seed) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < kHidden) dst[i] = static_cast<float>(seed + i % 11);
}
__global__ void stage1_kernel(const float* src, float* dst, int bias) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < kLogits) dst[i] = src[i % kHidden] * 2.0f + static_cast<float>(bias);
}
cudaError_t launch0(cudaStream_t stream, void* hidden, void* user) {
  auto* context = static_cast<Context*>(user);
  stage0_kernel<<<(kHidden + 255) / 256, 256, 0, stream>>>(
      static_cast<float*>(hidden), context->seed);
  return cudaPeekAtLastError();
}
cudaError_t launch1(cudaStream_t stream, void* hidden, void* logits, void* user) {
  auto* context = static_cast<Context*>(user);
  stage1_kernel<<<(kLogits + 255) / 256, 256, 0, stream>>>(
      static_cast<const float*>(hidden), static_cast<float*>(logits),
      context->bias);
  return cudaPeekAtLastError();
}

bool status(cudaError_t result, const char* operation) {
  if (result == cudaSuccess) return true;
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
  return false;
}
} // namespace

int main() {
  int count = 0;
  if (!status(cudaGetDeviceCount(&count), "device count") || count < 2) {
    std::fprintf(stderr, "requires two available CUDA GPUs\n");
    return 2;
  }

  std::vector<pp2streams::DeviceBuffers> buffers(kRequests);
  bool allocated = true;
  for (auto& b : buffers) {
    if (!status(cudaSetDevice(0), "GPU0 selection") ||
        !status(cudaMalloc(&b.gpu0_hidden, kHidden * sizeof(float)), "GPU0 hidden") ||
        !status(cudaSetDevice(1), "GPU1 selection") ||
        !status(cudaMalloc(&b.gpu1_hidden, kHidden * sizeof(float)), "GPU1 hidden") ||
        !status(cudaMalloc(&b.gpu1_logits, kLogits * sizeof(float)), "GPU1 logits")) {
      allocated = false;
      break;
    }
  }

  pp2streams::TwoStagePipeline::Config cfg;
  cfg.hidden_bytes = kHidden * sizeof(float);
  cfg.logits_bytes = kLogits * sizeof(float);
  pp2streams::TwoStagePipeline pipeline(cfg);
  std::string err;
  bool pass = allocated && pipeline.initialize(buffers, err);
  if (!pass) std::fprintf(stderr, "initialize failed: %s\n", err.c_str());

  std::vector<Context> contexts(kRequests);
  for (int step = 0; pass && step < 3; ++step) {
    for (int i = 0; i < kRequests; ++i) {
      contexts[i] = {100 * i + step, 3 * i + step};
      if (!pipeline.submit_stage0(i, launch0, &contexts[i], err)) {
        std::fprintf(stderr, "stage0 request %d: %s\n", i, err.c_str());
        pass = false;
        break;
      }
    }
    std::vector<bool> complete(kRequests, false);
    int completed = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (pass && completed < kRequests) {
      if (std::chrono::steady_clock::now() >= deadline) {
        std::fprintf(stderr, "timed out waiting for async stage events\n");
        pass = false;
        break;
      }
      for (int i = 0; i < kRequests && pass; ++i) {
        if (complete[i]) continue;
        const auto phase = pipeline.phase(i);
        if (phase == pp2streams::TwoStagePipeline::Phase::Stage0) {
          bool launched = false;
          pass = pipeline.try_submit_stage1(i, launch1, &contexts[i], launched, err);
        } else if (phase == pp2streams::TwoStagePipeline::Phase::Stage1) {
          bool ready = false;
          std::vector<std::uint8_t> bytes;
          pass = pipeline.try_collect(i, bytes, ready, err);
          if (!pass || !ready) continue;
          if (bytes.size() != kLogits * sizeof(float)) pass = false;
          for (int j = 0; pass && j < kLogits; ++j) {
            float actual = 0;
            std::memcpy(&actual, bytes.data() + j * sizeof(float), sizeof(float));
            const float expected = static_cast<float>(
                2 * (contexts[i].seed + j % 11) + contexts[i].bias);
            if (!std::isfinite(actual) || actual != expected) {
              std::fprintf(stderr, "mismatch step=%d req=%d idx=%d got=%.9g expect=%.9g\n",
                           step, i, j, actual, expected);
              pass = false;
            }
          }
          if (pass) { complete[i] = true; ++completed; }
        } else {
          std::fprintf(stderr, "invalid phase step=%d req=%d\n", step, i);
          pass = false;
        }
        if (!pass) std::fprintf(stderr, "pipeline stage failure: %s\n", err.c_str());
      }
      if (pass && completed < kRequests) std::this_thread::yield();
    }
  }

  // Drains both streams even after a failed enqueue before releasing memory.
  pipeline.shutdown();
  for (auto& b : buffers) {
    (void)cudaSetDevice(0);
    if (b.gpu0_hidden) (void)cudaFree(b.gpu0_hidden);
    (void)cudaSetDevice(1);
    if (b.gpu1_hidden) (void)cudaFree(b.gpu1_hidden);
    if (b.gpu1_logits) (void)cudaFree(b.gpu1_logits);
  }
  std::printf("synthetic two-stage stream test: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
