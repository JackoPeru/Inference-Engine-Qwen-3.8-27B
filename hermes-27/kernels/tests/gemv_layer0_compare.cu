// Standalone HERMES scalar/Warp4/Packed4 GEMV replay on verified llama layer0.
// Never invokes model inference; default manifest verification is performed
// by gemv_layer0_oracle.py BEFORE this private GPU executable can run.
#include "gemv_real_common.hpp"
#include "hermes/nvfp4_gemv.hpp"
#include "hermes/nvfp4_gemv_warp_variants.hpp"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int H = 5120, F = 10240, GUARD = 16, REPEATS = 10;
constexpr std::uint32_t SENTINEL = 0x5a37c4b2u;

struct Args {
  std::string gguf, input_f32, input_bf16, llama_qkv;
  int device = 0;
  bool selftest = false;
};

Args options(int argc, char** argv) {
  Args a;
  if (argc == 2 && std::string(argv[1]) == "--selftest") {
    a.selftest = true;
    return a;
  }
  for (int i = 1; i < argc; ++i) {
    const std::string flag(argv[i]);
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
    const std::string value(argv[++i]);
    if (flag == "--gguf") a.gguf = value;
    else if (flag == "--original-f32") a.input_f32 = value;
    else if (flag == "--activation-bf16") a.input_bf16 = value;
    else if (flag == "--llama-qkv") a.llama_qkv = value;
    else if (flag == "--device") {
      auto v = gemv_real::parse_nonnegative(value, "--device");
      if (v >= 128) throw std::runtime_error("GPU device index out of range");
      a.device = static_cast<int>(v);
    } else throw std::runtime_error("unknown option: " + flag);
  }
  if (a.gguf.empty() || a.input_f32.empty() || a.input_bf16.empty() ||
      a.llama_qkv.empty())
    throw std::runtime_error("require --gguf, --original-f32, "
                             "--activation-bf16, --llama-qkv");
  return a;
}

void ck(cudaError_t result, const char* call) {
  if (result != cudaSuccess)
    throw std::runtime_error(std::string(call) + ": " +
                             cudaGetErrorString(result));
}

template<class T> struct Device {
  T* ptr = nullptr;
  explicit Device(std::size_t count) {
    ck(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)),
       "cudaMalloc");
  }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  ~Device() { if (ptr) cudaFree(ptr); }
};

std::vector<float> load_llama(const std::string& path) {
  if (gemv_real::checked_size(path) != F * sizeof(float))
    throw std::runtime_error("llama captured QKV must be exactly 10240 FP32");
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open llama QKV capture");
  std::vector<float> y(F);
  gemv_real::read_exact(f, y.data(), y.size() * sizeof(float), "llama QKV");
  for (float z : y)
    if (!std::isfinite(z))
      throw std::runtime_error("llama QKV contains nonfinite float");
  return y;
}

__global__ void probe_rne(const float* src, std::uint16_t* dst, int n) {
  int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  __nv_bfloat16 converted = (__nv_bfloat16)src[i];
  std::uint16_t bits;
  memcpy(&bits, &converted, sizeof(bits));
  dst[i] = bits;
}

void verify_cuda_rne(const gemv_real::Replay& r) {
  Device<float> original(H);
  Device<std::uint16_t> converted(H);
  std::vector<std::uint16_t> returned(H);
  ck(cudaMemcpy(original.ptr, r.original_f32.data(), H * sizeof(float),
                cudaMemcpyHostToDevice), "copy source F32 to CUDA");
  probe_rne<<<(H + 255) / 256, 256>>>(original.ptr, converted.ptr, H);
  ck(cudaGetLastError(), "probe BF16 launch");
  ck(cudaDeviceSynchronize(), "probe BF16 completion");
  ck(cudaMemcpy(returned.data(), converted.ptr, H * sizeof(std::uint16_t),
                cudaMemcpyDeviceToHost), "copy CUDA BF16 input probe");
  if (returned != r.x)
    throw std::runtime_error("CUDA BF16_RNE differs from verified CPU conversion");
  std::cout << "CUDA_RNE_MATCH count=5120 bitwise=PASS\n";
}

enum class Variant { Scalar, Warp4, Packed4 };
const char* name(Variant v) {
  switch (v) {
    case Variant::Scalar: return "Scalar";
    case Variant::Warp4: return "Warp4";
    case Variant::Packed4: return "Packed4";
  }
  return "invalid";
}

void launch(Variant variant, const std::uint16_t* x, const std::uint8_t* w,
            float* y) {
  if (variant == Variant::Packed4) {
    ck(hermes::nvfp4_gemv_warp_experiment(
      x, w, y, H, F, hermes::Nvfp4WarpExperiment::PackedU32x4),
      "Packed4 launch");
  } else {
    hermes::nvfp4_gemv_cuda_variant(x, w, y, H, F,
      variant == Variant::Scalar ? hermes::Nvfp4GemvVariant::Scalar
                                 : hermes::Nvfp4GemvVariant::Warp4);
  }
  ck(cudaGetLastError(), "kernel launch");
  ck(cudaDeviceSynchronize(), "kernel execution");
}

std::vector<float> verify_replay(Variant v,
    const Device<std::uint16_t>& dx,
    const Device<std::uint8_t>& dw,
    const Device<std::uint32_t>& dout) {
  std::vector<std::uint32_t> pattern(F + GUARD * 2, SENTINEL);
  std::vector<std::uint32_t> returned(pattern.size()), prior;
  for (int repeat = 0; repeat < REPEATS; ++repeat) {
    ck(cudaMemcpy(dout.ptr, pattern.data(), pattern.size() * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice), "write output sentinels");
    launch(v, dx.ptr, dw.ptr, reinterpret_cast<float*>(dout.ptr + GUARD));
    ck(cudaMemcpy(returned.data(), dout.ptr,
                  returned.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
                  "read complete kernel output");
    for (int k = 0; k < GUARD; ++k)
      if (returned[k] != SENTINEL || returned[F + GUARD + k] != SENTINEL)
        throw std::runtime_error(std::string(name(v)) + " corrupted guard");
    for (int k = 0; k < F; ++k)
      if (!std::isfinite(std::bit_cast<float>(returned[GUARD + k])))
        throw std::runtime_error(std::string(name(v)) + " produced NaN/Inf");
    if (repeat == 0) prior.assign(returned.begin() + GUARD,
                                  returned.begin() + GUARD + F);
    else if (!std::equal(prior.begin(), prior.end(), returned.begin() + GUARD))
      throw std::runtime_error(std::string(name(v)) +
                               " non-repeatable FP32 bit-pattern");
  }
  std::vector<float> output(F);
  for (int k = 0; k < F; ++k) output[k] = std::bit_cast<float>(prior[k]);
  std::cout << "KERNEL " << name(v)
            << " repeats=10 bitwise_deterministic=yes guard_before_after=16+16 OK\n";
  return output;
}

void verify_immutable(const gemv_real::Replay& r,
    const Device<std::uint16_t>& dx,
    const Device<std::uint8_t>& dw) {
  std::vector<std::uint16_t> x(H);
  std::vector<std::uint8_t> w(r.weights.size());
  ck(cudaMemcpy(x.data(), dx.ptr, x.size() * sizeof(std::uint16_t),
                cudaMemcpyDeviceToHost), "read input immutability");
  ck(cudaMemcpy(w.data(), dw.ptr, w.size(), cudaMemcpyDeviceToHost),
     "read GGUF weight immutability");
  if (x != r.x || w != r.weights)
    throw std::runtime_error("GPU kernel altered immutable W/X");
}

struct Metrics { double max_abs = 0., rmse = 0., max_ulp = 0.; int worst = -1; int fail = 0; };

Metrics report(const std::string& title, const std::vector<float>& values,
    const std::vector<double>& reference, bool kernel_gate,
    const std::vector<gemv_real::ReferenceRow>* cpu = nullptr) {
  if (values.size() != F || reference.size() != F)
    throw std::runtime_error("comparison rows mismatch");
  Metrics m;
  double ss = 0.;
  for (int i = 0; i < F; ++i) {
    const double delta = std::fabs(double(values[i]) - reference[i]);
    if (delta > m.max_abs) { m.max_abs = delta; m.worst = i; }
    ss += delta * delta;
    if (kernel_gate) {
      // Use both strict empirical gate and conservative rounding envelope.
      const double empirical = 1.e-4 + 2.e-4 * std::fabs(reference[i]);
      const double envelope = gemv_real::fp32_error_envelope(H, (*cpu)[i]);
      if (delta > empirical || delta > envelope) ++m.fail;
      m.max_ulp = std::max(m.max_ulp,
          delta / gemv_real::ulp_float(reference[i]));
    }
  }
  m.rmse = std::sqrt(ss / F);
  std::cout << std::setprecision(10) << title
            << " full_rows=10240 max_abs=" << m.max_abs
            << " rmse=" << m.rmse << " worst_row=" << m.worst;
  if (kernel_gate)
    std::cout << " max_ulp=" << m.max_ulp
              << " fp32_vs_fp64_gate_fail_rows=" << m.fail;
  else
    std::cout << " PARITY_DIAGNOSTIC_ONLY=true";
  std::cout << '\n';
  return m;
}

void report_pair(const char* title, const std::vector<float>& left,
                 const std::vector<float>& right, bool& ok) {
  double maximum = 0., ss = 0.;
  int wrong = 0;
  for (int i = 0; i < F; ++i) {
    const double e = std::fabs(double(left[i]) - right[i]);
    maximum = std::max(maximum, e);
    ss += e * e;
    if (e > 1.e-4 + 2.e-4 * std::max(
          std::fabs(double(left[i])), std::fabs(double(right[i])))) ++wrong;
  }
  if (wrong) ok = false;
  std::cout << std::setprecision(10) << title
            << " max_abs=" << maximum << " rmse=" << std::sqrt(ss / F)
            << " per_row_gate_failures=" << wrong << '\n';
}
} // namespace

int main(int argc, char** argv) {
  try {
    const Args a = options(argc, argv);
    if (a.selftest) { gemv_real::selftest(); return 0; }
    gemv_real::Arguments loaded;
    loaded.gguf = a.gguf;
    loaded.tensor = "blk.0.attn_qkv.weight";
    loaded.bf16_path = a.input_bf16;
    loaded.original_f32_path = a.input_f32;
    auto r = gemv_real::load_weights(loaded);
    if (r.H != H || r.F != F) throw std::runtime_error("invalid GGUF QKV shape");
    gemv_real::load_activation(r, loaded);
    const auto llama = load_llama(a.llama_qkv);
    const auto oracle_bf16 = gemv_real::cpu_fp64_all(r);
    const auto oracle_f32 = gemv_real::cpu_fp64_all(r, true);
    std::vector<double> ref_bf16(F), ref_f32(F);
    for (int k = 0; k < F; ++k) {
      ref_bf16[k] = oracle_bf16[k].dot;
      ref_f32[k] = oracle_f32[k].dot;
    }
    std::cout << "CAPTURE_INPUT original_llama_attn_norm FP32[5120] "
                 "converted_to_HERMES_BF16_RNE; GGUF raw W[10240,5120]\n";
    gemv_real::print_identity(r);
    gemv_real::print_activation_rounding(r, oracle_bf16);
    // Never read/initialize any CUDA before CPU validity checks succeed.
    ck(cudaSetDevice(a.device), "cudaSetDevice");
    verify_cuda_rne(r);
    Device<std::uint16_t> dx(H);
    Device<std::uint8_t> dw(r.weights.size());
    Device<std::uint32_t> dout(F + GUARD * 2);
    ck(cudaMemcpy(dx.ptr, r.x.data(), H * 2, cudaMemcpyHostToDevice),
       "upload same BF16 input");
    ck(cudaMemcpy(dw.ptr, r.weights.data(), r.weights.size(),
                  cudaMemcpyHostToDevice), "upload same GGUF weights");
    const auto scalar = verify_replay(Variant::Scalar, dx, dw, dout);
    const auto warp = verify_replay(Variant::Warp4, dx, dw, dout);
    const auto packed = verify_replay(Variant::Packed4, dx, dw, dout);
    verify_immutable(r, dx, dw);
    bool ok = true;
    for (const auto& item : std::vector<std::pair<const char*, const std::vector<float>*>>{
        {"SCALAR", &scalar}, {"WARP4", &warp}, {"PACKED4", &packed}}) {
      const std::string label = item.first;
      const auto result = report(label + "_vs_CPU_FP64_BF16",
        *item.second, ref_bf16, true, &oracle_bf16);
      if (result.fail) ok = false;
      report(label + "_vs_LLAMA_CAPTURED_QKV",
        *item.second, std::vector<double>(llama.begin(), llama.end()), false);
    }
    report_pair("SCALAR_vs_WARP4", scalar, warp, ok);
    report_pair("SCALAR_vs_PACKED4", scalar, packed, ok);
    report_pair("WARP4_vs_PACKED4", warp, packed, ok);
    // Llama-vs-original and llama-vs-rounded are diagnostics. Same input is
    // identical only at recorded attn_norm, not known internal CUDA Q8_1 math.
    std::vector<double> ref_llama(llama.begin(), llama.end());
    report("LLAMA_CAPTURE_vs_CPU_FP64_F32",
      llama, ref_f32, false);
    report("LLAMA_CAPTURE_vs_CPU_FP64_BF16",
      llama, ref_bf16, false);
    if (!ok) {
      std::cerr << "FAIL kernel arithmetic gate on real llama input\n";
      return 1;
    }
    std::cout << "GEMV_KERNEL_REPLAY_PASS all three variants; "
                 "LLAMA_SEMANTIC_PARITY=UNPROVEN (reported as metric only)\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL_OR_BLOCKED gemv_layer0_compare: " << e.what() << '\n';
    return 1;
  }
}
