// CPU-only numerical diagnostic for llama.cpp CUDA's Q8_1 activation format.
// Approximates the quantize_q8_1 -> NVFP4 x Q8_1 mathematical dot product,
// not CUDA's dp4a reduction or an independent llama.cpp hidden activation.
#include "gemv_real_common.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

// Portable nearest-even F32 -> binary16 (the Q8_1 block scale is stored as
// FP16 by CUDA make_half2(d,sum)). Inputs must already be finite.
std::uint16_t f32_to_fp16_rne(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const int exponent = int((bits >> 23) & 255u);
  const std::uint32_t mantissa = bits & 0x7fffffu;
  if (exponent == 255) gemv_real::error("Q8_1 nonfinite scale");
  int half_exp = exponent - 127 + 15;
  if (half_exp >= 31) gemv_real::error("Q8_1 half scale overflow");
  if (half_exp <= 0) {
    if (half_exp < -10) return std::uint16_t(sign);
    const std::uint32_t full_m = mantissa | 0x800000u;
    const int shift = 14 - half_exp;
    std::uint32_t rounded = full_m >> shift;
    const std::uint32_t remainder = full_m & ((1u << shift) - 1u);
    const std::uint32_t midpoint = 1u << (shift - 1);
    if (remainder > midpoint || (remainder == midpoint && (rounded & 1u)))
      ++rounded;
    return std::uint16_t(sign | rounded);
  }
  const std::uint32_t rounded = mantissa + 0xfffu + ((mantissa >> 13) & 1u);
  if (rounded >= 0x800000u) {
    ++half_exp;
    if (half_exp >= 31) gemv_real::error("Q8_1 rounded half scale overflow");
    return std::uint16_t(sign | (std::uint32_t(half_exp) << 10));
  }
  return std::uint16_t(sign | (std::uint32_t(half_exp) << 10) |
                       (rounded >> 13));
}

float fp16_to_f32(std::uint16_t bits) {
  const int e = (bits >> 10) & 31;
  const int m = bits & 1023;
  if (e == 31) gemv_real::error("Q8_1 nonfinite half scale");
  const float value = e == 0 ? std::ldexp(float(m), -24)
                             : std::ldexp(float(1024 + m), e - 25);
  return (bits & 0x8000u) ? -value : value;
}

void selftest() {
  const struct { float val; std::uint16_t bits; } samples[] = {
      {0.f, 0x0000}, {1.f, 0x3c00}, {-1.f, 0xbc00},
      {2.f, 0x4000}, {0.5f, 0x3800}, {65504.f, 0x7bff},
      {std::ldexp(1.f, -24), 0x0001}, {std::ldexp(1.f, -14), 0x0400}};
  for (auto s : samples)
    if (f32_to_fp16_rne(s.val) != s.bits || fp16_to_f32(s.bits) != s.val)
      gemv_real::error("FP16 scale selftest failed");
  // Even/odd tie cases at half-precision mantissa boundary.
  if (f32_to_fp16_rne(1.f + std::ldexp(1.f, -11)) != 0x3c00 ||
      f32_to_fp16_rne(1.f + 3.f * std::ldexp(1.f, -11)) != 0x3c02)
    gemv_real::error("FP16 RNE tie selftest failed");
  std::cout << "Q8_1_CPU_SELFTEST PASS FP16 round-nearest-even and decode\n";
}

std::vector<double> simulate_q8_1(const gemv_real::Replay& r) {
  if (r.original_f32.size() != r.x.size())
    gemv_real::error("Q8_1 comparison requires original FP32 capture");
  if (r.H % 32) gemv_real::error("Q8_1 requires H multiple of 32");
  std::vector<double> q8(r.H);
  double max_input_abs = 0.0;
  double activation_error_sq = 0.0;
  std::size_t zero_scale_blocks = 0;
  for (int block = 0; block < r.H / 32; ++block) {
    float amax = 0.f;
    for (int j = 0; j < 32; ++j)
      amax = std::max(amax, std::fabs(r.original_f32[block * 32 + j]));
    const float scale = amax / 127.f;
    const auto half_bits = f32_to_fp16_rne(scale);
    const float stored_scale = fp16_to_f32(half_bits);
    if (stored_scale == 0.f) ++zero_scale_blocks;
    for (int j = 0; j < 32; ++j) {
      const int k = block * 32 + j;
      const float input = r.original_f32[k];
      const float normalized = amax == 0.f ? 0.f : input / scale;
      // CUDA roundf() uses ties away from zero. A host CPU's division can
      // differ in last-bit behavior; this remains a CPU model, not a CUDA
      // bitwise reimplementation.
      const float rounded = std::round(normalized);
      if (!std::isfinite(rounded) || rounded < -127.f || rounded > 127.f)
        gemv_real::error("Q8_1 quantized int8 outside legal [-127,127]");
      const int8_t qi = static_cast<int8_t>(rounded);
      q8[k] = double(qi) * double(stored_scale);
      const double err = q8[k] - double(input);
      max_input_abs = std::max(max_input_abs, std::fabs(err));
      activation_error_sq += err * err;
    }
  }
  std::cout << std::setprecision(10)
            << "Q8_1_CPU_QUANT H=" << r.H << " blocks32=" << r.H / 32
            << " half_scale_zero_blocks=" << zero_scale_blocks
            << " input_vs_q8_max_abs=" << max_input_abs
            << " input_vs_q8_rmse=" << std::sqrt(activation_error_sq / r.H)
            << '\n';
  return q8;
}

// Independent FP64 NVFP4 dot on the original GGUF layout for arbitrary
// higher-precision input. Only the activation representation differs.
double dot_q8_cpu_fp64(const gemv_real::Replay& r,
                      const std::vector<double>& x, int row) {
  const int blocks = r.H / 64;
  const auto* w = r.weights.data() + std::size_t(row) * blocks * 36;
  double out = 0.0;
  for (int b = 0; b < blocks; ++b)
    for (int s = 0; s < 4; ++s) {
      const double scale = gemv_real::ue4m3_raw(w[b * 36 + s]);
      for (int j = 0; j < 8; ++j) {
        const auto q = w[b * 36 + 4 + s * 8 + j];
        const int k = b * 64 + s * 16 + j;
        out += x[k] * scale * gemv_real::e2m1(q);
        out += x[k + 8] * scale * gemv_real::e2m1(q >> 4);
      }
    }
  if (!std::isfinite(out)) gemv_real::error("Q8_1 CPU FP64 output nonfinite");
  return out;
}

struct Comparison {
  double max = 0.0, mse = 0.0, base_sumsq = 0.0;
  int worst = -1;
  void add(double x, double ref, int row) {
    const double d = std::fabs(x - ref);
    if (d > max) { max = d; worst = row; }
    mse += d * d;
    base_sumsq += ref * ref;
  }
  void print(const char* name, int count) const {
    const double rmse = std::sqrt(mse / count);
    const double ref_rms = std::sqrt(base_sumsq / count);
    std::cout << std::setprecision(10) << "Q8_1_CPU_FP64 " << name
              << " full_rows=" << count << " max_abs=" << max
              << " rmse=" << rmse
              << " rel_rmse=" << rmse / std::max(ref_rms, 1.e-30)
              << " worst_row=" << worst << '\n';
  }
};

} // namespace

int main(int argc, char** argv) {
  try {
    const auto options = gemv_real::parse(argc, argv, false);
    if (options.selftest) { selftest(); return 0; }
    if (options.inspect_weights)
      gemv_real::error("--inspect-weights belongs to gemv_real_cpu executable");
    auto r = gemv_real::load_weights(options);
    gemv_real::load_activation(r, options);
    gemv_real::print_identity(r);
    const auto input_q8 = simulate_q8_1(r);
    const auto bf16_ref = gemv_real::cpu_fp64_all(r);
    const auto f32_ref = gemv_real::cpu_fp64_all(r, true);
    Comparison q8_v_f32, q8_v_bf16, bf16_v_f32;
    for (int row = 0; row < r.F; ++row) {
      const auto q8 = dot_q8_cpu_fp64(r, input_q8, row);
      q8_v_f32.add(q8, f32_ref[row].dot, row);
      q8_v_bf16.add(q8, bf16_ref[row].dot, row);
      bf16_v_f32.add(bf16_ref[row].dot, f32_ref[row].dot, row);
    }
    q8_v_f32.print("Q8_vs_original_F32", r.F);
    q8_v_bf16.print("Q8_vs_BF16", r.F);
    bf16_v_f32.print("BF16_vs_original_F32", r.F);
    std::cout << "PASS CPU Q8_1-like activation rounding error decomposition "
                 "over every real-GGUF output row; no CUDA launched\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL_OR_BLOCKED gemv_real_q8_cpu: " << e.what() << '\n';
    return 1;
  }
}
