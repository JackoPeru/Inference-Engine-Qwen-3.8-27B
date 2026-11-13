// Standalone opt-in GPU A/B. Requires explicit GPU clearance from the prime.
// Three ACTUAL production kernel entry points on identical raw GGUF W / BF16 X.
// No throughput timing, no model inference, no weight repacking or model state.
#include "gemv_real_common.hpp"
#include "hermes/nvfp4_gemv.hpp"
#include "hermes/nvfp4_gemv_warp_variants.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int kGuardWords = 16;
constexpr std::uint32_t kSentinel = 0x5a37c4b2u;

// Mirrors the production cast expression, but is independently compiled in
// this isolated harness. This checks CUDA's actual RNE behavior for every
// captured F32 value; it is not a readback of production's d_gb allocation.
__global__ void probe_bf16_cast(const float* input, std::uint16_t* out, int n) {
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    const __nv_bfloat16 v = (__nv_bfloat16)input[i];
    std::uint16_t bits;
    memcpy(&bits, &v, sizeof(bits));
    out[i] = bits;
  }
}

void ck(cudaError_t result, const char* operation) {
  if (result != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
}

template <typename T> struct Device {
  T* ptr = nullptr;
  Device() = default;
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  explicit Device(std::size_t count) {
    ck(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)), "cudaMalloc");
  }
  ~Device() { if (ptr) (void)cudaFree(ptr); }
};

enum class Path { Scalar, Warp4, Packed4 };

const char* label(Path path) {
  switch (path) {
    case Path::Scalar: return "scalar";
    case Path::Warp4: return "warp4";
    case Path::Packed4: return "packed4";
  }
  return "unknown";
}

void launch(Path path, const std::uint16_t* x, const std::uint8_t* w,
            float* output, int H, int F) {
  if (path == Path::Packed4) {
    ck(hermes::nvfp4_gemv_warp_experiment(
           x, w, output, H, F, hermes::Nvfp4WarpExperiment::PackedU32x4),
       "launch packed4");
  } else {
    hermes::nvfp4_gemv_cuda_variant(x, w, output, H, F,
      path == Path::Scalar ? hermes::Nvfp4GemvVariant::Scalar
                           : hermes::Nvfp4GemvVariant::Warp4);
  }
  ck(cudaGetLastError(), "kernel launch");
  ck(cudaDeviceSynchronize(), "kernel completion");
}

std::vector<float> run_repeat(Path path, const Device<std::uint16_t>& d_x,
                              const Device<std::uint8_t>& d_w,
                              const Device<std::uint32_t>& d_y,
                              int H, int F) {
  std::vector<std::uint32_t> before(F + 2 * kGuardWords, kSentinel);
  std::vector<std::uint32_t> after(before.size());
  std::vector<std::uint32_t> first;
  first.reserve(F);
  for (int n = 0; n < gemv_real::kRepeats; ++n) {
    ck(cudaMemcpy(d_y.ptr, before.data(), before.size() * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice), "initialize per-iteration output guards");
    launch(path, d_x.ptr, d_w.ptr,
           reinterpret_cast<float*>(d_y.ptr + kGuardWords), H, F);
    ck(cudaMemcpy(after.data(), d_y.ptr, after.size() * sizeof(std::uint32_t),
                  cudaMemcpyDeviceToHost), "read complete output and guards");
    for (int i = 0; i < kGuardWords; ++i)
      if (after[i] != kSentinel || after[kGuardWords + F + i] != kSentinel)
        throw std::runtime_error(std::string(label(path)) +
                                 " wrote outside its F output rows");
    for (int i = 0; i < F; ++i) {
      const float f = std::bit_cast<float>(after[kGuardWords + i]);
      if (!std::isfinite(f))
        throw std::runtime_error(std::string(label(path)) +
                                 " produced nonfinite output row " + std::to_string(i));
    }
    if (n == 0)
      first.assign(after.begin() + kGuardWords,
                   after.begin() + kGuardWords + F);
    else
      for (int i = 0; i < F; ++i)
        if (after[kGuardWords + i] != first[i])
          throw std::runtime_error(std::string(label(path)) +
                                   " nondeterministic output row " + std::to_string(i) +
                                   " repeat " + std::to_string(n + 1));
  }
  std::vector<float> values(F);
  for (int i = 0; i < F; ++i) values[i] = std::bit_cast<float>(first[i]);
  return values;
}

struct Metrics {
  double max_abs = 0.0;
  double max_ulp = 0.0;
  double sum_sq = 0.0;
  int worst = -1;
  int empirical_fail = 0;
  int envelope_fail = 0;
};

void verify_cuda_rne_cast(const gemv_real::Replay& replay) {
  if (replay.original_f32.empty()) {
    std::cout << "CUDA_BF16_RNE unavailable (no original F32 input)\n";
    return;
  }
  Device<float> original(replay.original_f32.size());
  Device<std::uint16_t> cast(replay.x.size());
  std::vector<std::uint16_t> returned(replay.x.size());
  ck(cudaMemcpy(original.ptr, replay.original_f32.data(),
                replay.original_f32.size() * sizeof(float), cudaMemcpyHostToDevice),
     "upload original FP32 vector");
  probe_bf16_cast<<<(replay.H + 255) / 256, 256>>>(
      original.ptr, cast.ptr, replay.H);
  ck(cudaGetLastError(), "CUDA BF16 probe launch");
  ck(cudaDeviceSynchronize(), "CUDA BF16 probe completion");
  ck(cudaMemcpy(returned.data(), cast.ptr, replay.x.size() * 2,
                cudaMemcpyDeviceToHost), "read CUDA BF16 probe");
  if (returned != replay.x)
    throw std::runtime_error("CPU RNE conversion disagrees with CUDA BF16 cast");
  std::cout << "CUDA_BF16_RNE PASS all " << returned.size()
            << " real captured F32-to-BF16 elements bit-exact\n";
}

Metrics compare_reference(const char* name, const std::vector<float>& actual,
                          const std::vector<gemv_real::ReferenceRow>& ref,
                          int H) {
  Metrics m;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double deviation = std::fabs(double(actual[i]) - ref[i].dot);
    const double ulp = gemv_real::ulp_float(ref[i].dot);
    const double empirical = 1.e-4 + 2.e-4 * std::fabs(ref[i].dot);
    const double bound = gemv_real::fp32_error_envelope(H, ref[i]);
    if (deviation > empirical) ++m.empirical_fail;
    if (deviation > bound) ++m.envelope_fail;
    if (deviation > m.max_abs) {
      m.max_abs = deviation;
      m.worst = static_cast<int>(i);
    }
    m.max_ulp = std::max(m.max_ulp, deviation / ulp);
    m.sum_sq += deviation * deviation;
  }
  std::cout << std::setprecision(10)
            << "CPU_FP64_VS_" << name << " full_rows=" << actual.size()
            << " max_abs=" << m.max_abs << " max_ulp=" << m.max_ulp
            << " rmse=" << std::sqrt(m.sum_sq / actual.size())
            << " worst_row=" << m.worst
            << " empirical_gate_fail=" << m.empirical_fail
            << " fp32_envelope_fail=" << m.envelope_fail << '\n';
  return m;
}

bool compare_pairs(const std::vector<float>& a, const std::vector<float>& b,
                   const char* left, const char* right) {
  double max_abs = 0.0, rms = 0.0;
  int worst = -1, failures = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double diff = std::fabs(double(a[i]) - double(b[i]));
    if (diff > max_abs) { max_abs = diff; worst = static_cast<int>(i); }
    rms += diff * diff;
    const double tolerance = 1.e-4 + 2.e-4 *
      std::max(std::fabs(double(a[i])), std::fabs(double(b[i])));
    if (diff > tolerance) ++failures;
  }
  const auto argmax = [](const std::vector<float>& v) {
    return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
  };
  const auto top1_a = argmax(a), top1_b = argmax(b);
  std::cout << std::setprecision(10)
            << "PAIR " << left << "_vs_" << right
            << " max_abs=" << max_abs << " rmse=" << std::sqrt(rms / a.size())
            << " worst_row=" << worst << " empirical_gate_fail=" << failures
            << " argmax=" << top1_a << "," << top1_b << '\n';
  return failures == 0 && top1_a == top1_b;
}

void verify_inputs(const Device<std::uint16_t>& d_x,
                   const Device<std::uint8_t>& d_w,
                   const gemv_real::Replay& r) {
  std::vector<std::uint16_t> input(r.x.size());
  std::vector<std::uint8_t> weights(r.weights.size());
  ck(cudaMemcpy(input.data(), d_x.ptr, input.size() * 2, cudaMemcpyDeviceToHost),
     "verify immutable BF16 input");
  ck(cudaMemcpy(weights.data(), d_w.ptr, weights.size(), cudaMemcpyDeviceToHost),
     "verify immutable GGUF weights");
  if (input != r.x || weights != r.weights)
    throw std::runtime_error("kernel modified the shared GGUF weights or BF16 input");
}

} // namespace

int main(int argc, char** argv) {
  try {
    const auto options = gemv_real::parse(argc, argv, true);
    if (options.selftest) {
      gemv_real::selftest();
      return 0;
    }
    if (options.inspect_weights)
      throw std::runtime_error("--inspect-weights belongs to CPU-only executable");
    auto replay = gemv_real::load_weights(options);
    gemv_real::load_activation(replay, options);
    gemv_real::print_identity(replay);
    // Compute all reference rows before any CUDA runtime call. If any real
    // input/weight is invalid, fail before touching the GPU.
    const auto reference = gemv_real::cpu_fp64_all(replay);
    gemv_real::print_activation_rounding(replay, reference);
    ck(cudaSetDevice(options.device), "cudaSetDevice");
    verify_cuda_rne_cast(replay);
    Device<std::uint16_t> d_x(replay.x.size());
    Device<std::uint8_t> d_w(replay.weights.size());
    Device<std::uint32_t> d_y(replay.F + 2 * kGuardWords);
    ck(cudaMemcpy(d_x.ptr, replay.x.data(), replay.x.size() * 2,
                  cudaMemcpyHostToDevice), "upload BF16 activation");
    ck(cudaMemcpy(d_w.ptr, replay.weights.data(), replay.weights.size(),
                  cudaMemcpyHostToDevice), "upload real GGUF weights");
    const auto scalar = run_repeat(Path::Scalar, d_x, d_w, d_y, replay.H, replay.F);
    const auto warp4 = run_repeat(Path::Warp4, d_x, d_w, d_y, replay.H, replay.F);
    const auto packed4 = run_repeat(Path::Packed4, d_x, d_w, d_y, replay.H, replay.F);
    verify_inputs(d_x, d_w, replay);
    const auto s = compare_reference("SCALAR", scalar, reference, replay.H);
    const auto w = compare_reference("WARP4", warp4, reference, replay.H);
    const auto p = compare_reference("PACKED4", packed4, reference, replay.H);
    // Evaluate all three pairs even if the first comparison fails. Every
    // failure should retain its own complete-row numerical diagnostics.
    const bool scalar_warp4 = compare_pairs(scalar, warp4, "scalar", "warp4");
    const bool scalar_packed4 = compare_pairs(scalar, packed4, "scalar", "packed4");
    const bool warp4_packed4 = compare_pairs(warp4, packed4, "warp4", "packed4");
    const bool pairs_ok = scalar_warp4 && scalar_packed4 && warp4_packed4;
    const bool per_kernel_ok =
      s.empirical_fail == 0 && w.empirical_fail == 0 && p.empirical_fail == 0 &&
      s.envelope_fail == 0 && w.envelope_fail == 0 && p.envelope_fail == 0;
    if (!pairs_ok || !per_kernel_ok) {
      std::cerr << "FAIL real GEMV same-input full-row CPU FP64 comparison\n";
      return 1;
    }
    std::cout << "PASS real GEMV three kernels x 10 deterministic iterations, "
                 "all rows, FP64, error bounds, guards, immutable W/X";
    if (replay.reconstructed && !replay.direct_bf16_confirmed)
      std::cout << " (BF16 reconstructed from d_g F32, device bytes not confirmed)";
    std::cout << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL_OR_BLOCKED gemv_real_compare: " << e.what() << '\n';
    return 1;
  }
}
